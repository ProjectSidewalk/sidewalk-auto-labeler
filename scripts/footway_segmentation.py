"""Footway: does an off-the-shelf segmenter agree with GSV depth on the walkable surface?
(issue #47, step 2).

A STUDY, not production: nothing here is wired into main.py, geo.py, fuse_sites.py or
sources/. Step 1 (scripts/depth_at_detection.py, docs/depth-at-detection-study.md) found
that GSV's depth model puts a plane under every below-horizon pixel and that a verdict-True
detection sits on the modelled surface ~92% of the time -- but so does a verdict-False one,
so depth alone carries no ramp-vs-not signal. This step asks whether a semantic segmenter
supplies the missing object/surface-class term, and how the two disagree.

Sample. Every judged RampNet benchmark pano of the four harvested GSV cities
(eval_sites.judged_gt_panos: same drift gate and skip rules as every sibling study) whose
local depth payload is `measured`. Pixels come from ../RampNet/benchmark/<city>/panos/
(read-only).

Segmenter. facebook/mask2former-swin-large-mapillary-vistas-semantic, pinned to
MODEL_REVISION (Mapillary Vistas v1.2, 65 classes: it has Sidewalk, Curb, Curb Cut,
Crosswalk - Plain and Pedestrian Area, the vocabulary this question needs). The HF image
processor's default resizes every input to 384x384; that is switched off (`do_resize=False`)
and the model sees the tiles at their rendered 1024x1024. Semantic scores are combined the
standard Mask2Former way (softmax class scores without the no-object column x sigmoid mask
probabilities), at mask resolution, then bilinearly upsampled to the target and argmaxed
-- not through HF's post_process_semantic_segmentation, which squashes every mask through a
fixed 384x384 intermediate (anisotropically, for the 2:1 direct arm).

Tiling (the trap the issue names: the segmenters are trained on perspective images).
Each pano is resized to 4096x2048 (LANCZOS) and rendered to 16 perspective tiles, 90 deg
FOV, 1024x1024 (the same ~11.4 px/deg as the resized pano), bilinear: 8 headings every
45 deg x 2 pitch rows (0 deg and -35 deg). The label maps are back-projected onto a
1024x512 equirect grid by voting over every tile whose frustum contains the pixel (nearest
pixel within each tile); a tie goes to the tied label of the tile whose optical axis is
nearest in angle. Pixels no tile sees (only the far nadir, below ~-75 deg at the tile
corners) get NONE_LABEL and are counted. A CONTROL arm feeds the equirect directly
(2048x1024) and reads the label map on the same grid; it quantifies the trap (poles, seam)
and is reported beside the tiled arm, never in the headline.

Image frame throughout: x_norm from the left of the JPEG, y_norm from the top; longitude
increases to the right (x_norm = 0.5 + lon / 2pi), latitude up (y_norm = 0.5 - lat / pi).
Tile geometry is pure image-frame geometry (no compass). The depth lookups are imported
from scripts/depth_at_detection.py (PayloadIndex, plane_class, classify), which uses
depth._plane_at in the image frame and the exact ray -- never depth.depth_at.

Collapsed classes (COLLAPSE, every Vistas name mapped explicitly; an unmapped name refuses):
WALK, ROAD, OBJECT, STRUCTURE, SKY, OTHER, and NONE for tiled-arm pixels no tile covers.

Subcommands (run in this order; only `segment` needs torch + transformers, which are
deliberately NOT in requirements.txt -- this is an analysis tool, not the pipeline; it says
so when they are missing):

    python scripts/footway_segmentation.py sample   [--run-root runs] [--benchmark-root ../RampNet/benchmark]
    python scripts/footway_segmentation.py tiles    # -> <work>/tiles/, <work>/direct/
    python scripts/footway_segmentation.py segment --in <work>/tiles --out <work>/tile_labels
    python scripts/footway_segmentation.py segment --in <work>/direct --out <work>/direct_labels --target 1024x512
    python scripts/footway_segmentation.py stitch   # -> <work>/stitched/
    python scripts/footway_segmentation.py compare  # -> runs/<city>/footway/, runs/_pooled/footway/
    python scripts/footway_segmentation.py verdict
    python scripts/footway_segmentation.py figures  # -> docs/figures/footway-depth/

`<work>` defaults to runs/_pooled/footway/work/ (untracked: tiles, label maps). The
reports and CSVs under runs/<city>/footway/ and runs/_pooled/footway/ are tracked, with
the label maps' sha256 in masks_manifest.json.

Stdlib + numpy + Pillow on top of depth.py / geo.py / fuse_sites.py / eval_sites.py /
depth_at_detection.py; scipy + matplotlib only for `figures`; torch + transformers only for
`segment`.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT / 'scripts'), str(REPO_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import depth as depthlib  # noqa: E402
import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import depth_at_detection as dad  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE  # noqa: E402

Image.MAX_IMAGE_PIXELS = None   # the benchmark JPEGs are 16384x8192, a known size

DEFAULT_CITIES = ['bend', 'paterson', 'gainesville', 'sao_paulo']
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'footway-depth'
SEED = 47

MODEL_ID = 'facebook/mask2former-swin-large-mapillary-vistas-semantic'
MODEL_REVISION = '4772b6bf101d91f2534c106dc524d906aeb3c68a'
FALLBACK_MODEL_ID = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'

PANO_W, PANO_H = 4096, 2048
TILE_SIZE = 1024
TILE_FOV_DEG = 90.0
TILE_HEADINGS_DEG = tuple(range(0, 360, 45))
TILE_PITCHES_DEG = (0, -35)
GRID_W, GRID_H = 1024, 512
DIRECT_W, DIRECT_H = 2048, 1024
NONE_LABEL = 255
JPEG_QUALITY = 95

# --- class collapse (Mapillary Vistas v1.2 names -> the six groups) ---------------------
WALK, ROAD, OBJECT, STRUCTURE, SKY, OTHER, NONE = (
    'WALK', 'ROAD', 'OBJECT', 'STRUCTURE', 'SKY', 'OTHER', 'NONE')
SEG_GROUPS = [WALK, ROAD, OBJECT, STRUCTURE, SKY, OTHER, NONE]
SURFACE_GROUPS = {WALK, ROAD}
COLLAPSE = {
    # WALK: where a pedestrian walks, including the ramp itself
    'Sidewalk': WALK, 'Curb': WALK, 'Curb Cut': WALK, 'Crosswalk - Plain': WALK,
    'Pedestrian Area': WALK,
    # ROAD: the carriageway and its markings; flush fixtures in it (manhole, catch basin,
    # pothole) and rail track are part of the drivable surface, not objects on it
    'Road': ROAD, 'Lane Marking - Crosswalk': ROAD, 'Lane Marking - General': ROAD,
    'Parking': ROAD, 'Bike Lane': ROAD, 'Service Lane': ROAD, 'Manhole': ROAD,
    'Catch Basin': ROAD, 'Pothole': ROAD, 'Rail Track': ROAD,
    # OBJECT: every vehicle, person, rider, animal, vegetation, pole, sign, street
    # furniture and barrier -- the things GSV depth is said to omit
    'Bird': OBJECT, 'Ground Animal': OBJECT, 'Person': OBJECT, 'Bicyclist': OBJECT,
    'Motorcyclist': OBJECT, 'Other Rider': OBJECT, 'Vegetation': OBJECT,
    'Guard Rail': OBJECT, 'Barrier': OBJECT, 'Banner': OBJECT, 'Bench': OBJECT,
    'Bike Rack': OBJECT, 'Billboard': OBJECT, 'CCTV Camera': OBJECT,
    'Fire Hydrant': OBJECT, 'Junction Box': OBJECT, 'Mailbox': OBJECT,
    'Phone Booth': OBJECT, 'Street Light': OBJECT, 'Pole': OBJECT,
    'Traffic Sign Frame': OBJECT, 'Utility Pole': OBJECT, 'Traffic Light': OBJECT,
    'Traffic Sign (Back)': OBJECT, 'Traffic Sign (Front)': OBJECT, 'Trash Can': OBJECT,
    'Bicycle': OBJECT, 'Boat': OBJECT, 'Bus': OBJECT, 'Car': OBJECT, 'Caravan': OBJECT,
    'Motorcycle': OBJECT, 'On Rails': OBJECT, 'Other Vehicle': OBJECT, 'Trailer': OBJECT,
    'Truck': OBJECT, 'Wheeled Slow': OBJECT, 'Car Mount': OBJECT, 'Ego Vehicle': OBJECT,
    # STRUCTURE
    'Building': STRUCTURE, 'Wall': STRUCTURE, 'Fence': STRUCTURE, 'Bridge': STRUCTURE,
    'Tunnel': STRUCTURE,
    SKY.title(): SKY,
    # OTHER: natural ground and the rest
    'Terrain': OTHER, 'Mountain': OTHER, 'Sand': OTHER, 'Snow': OTHER, 'Water': OTHER,
}

# Depth classes: depth_at_detection's five, with `floor` split by whether the plane is
# Google's exactly-level stand-in (depth.is_standin; step 1 section 5.4).
FLOOR_STANDIN = 'floor_standin'
DEPTH_CLASSES = [dad.GROUND, dad.FLOOR, FLOOR_STANDIN, dad.HORIZONTAL_NONFLOOR,
                 dad.NON_HORIZONTAL, dad.NO_PLANE]
DEPTH_SURFACE = {dad.GROUND, dad.FLOOR, FLOOR_STANDIN}

# Flat-raycast range bins (geo.detection_ground_point at 2.6 m, apply_pose=False).
RANGE_BINS = [(0, 5), (5, 10), (10, 15), (15, 25)]
BEYOND = 'beyond'
RANGE_LABELS = [f'{lo}-{hi}' for lo, hi in RANGE_BINS] + [BEYOND]

# --- the pre-registered reading (dated #47 comment, posted before any GT-conditioned
# number was computed) --------------------------------------------------------------------
#
# (C) FP signal, mirroring step 1's rule (ii): `non_surface` = the tiled segmenter class at
#     the detection pixel is not WALK or ROAD (OBJECT, STRUCTURE, SKY, OTHER, or NONE).
#     SUPPORTED iff the pooled non_surface share among verdict-False detections exceeds the
#     verdict-True share by >= RULE_C_MARGIN AND the False share's Wilson 95% lower bound
#     exceeds the True share's upper bound; otherwise NOT SUPPORTED. `underpowered` is
#     flagged when the False share's Wilson interval is wider than the margin. Benchmark
#     tier (0.55) only, measured payloads only (the whole sample is measured).
# (C') Surface reading, descriptive: the share of verdict-True detections on WALK or ROAD.
# (B) Curb height: over tiled-`Sidewalk` pixels on a measured `floor` plane (neither the
#     plane nor its local reference a stand-in) within 25 m, the median across panos of each
#     pano's median offset_local (depth_at_detection.classify) is "consistent with ~0.15 m"
#     iff it lies in CLAIM_BAND_M; ROAD pixels on a floor plane are the control (~0 m).
RULE_C_MARGIN = dad.RULE_II_MARGIN
CLAIM_BAND_M = dad.CLAIM_BAND_M
CURB_MAX_RANGE_M = geo.DEFAULT_MAX_RANGE_M
CURB_MAX_PIXELS = 1500     # per pano and group, a seeded subsample (runtime bound)
CURB_MIN_PIXELS = 20       # a pano enters the across-pano median with >= this many


# --- small io helpers --------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else
                            '' if v is None else v) for k, v in r.items() if k in fields})


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def work_dir(args):
    return args.work or (args.out_root / '_pooled' / 'footway' / 'work')


def pooled_dir(out_root):
    d = out_root / '_pooled' / 'footway'
    d.mkdir(parents=True, exist_ok=True)
    return d


def city_out(out_root, city):
    d = out_root / city / 'footway'
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_sample(args):
    path = pooled_dir(args.out_root) / 'sample.csv'
    if not path.exists():
        raise SystemExit(f'{path} not found; run `sample` first')
    rows = read_csv(path)
    return [r for r in rows if r['city'] in args.cities]


# --- tile geometry (pure numpy; tests/test_footway_segmentation.py) ---------------------

def tile_rotation(heading_deg, pitch_deg):
    """Camera -> world rotation for a tile looking at (heading, pitch).

    World/camera axes: X right, Y up, Z forward (at heading 0 = the pano's centre column).
    Pitch rotates about X (positive = look up), then heading about Y (positive = to the
    right, i.e. toward larger x_norm).
    """
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    r_pitch = np.array([[1, 0, 0],
                        [0, math.cos(p), math.sin(p)],
                        [0, -math.sin(p), math.cos(p)]])
    r_yaw = np.array([[math.cos(h), 0, math.sin(h)],
                      [0, 1, 0],
                      [-math.sin(h), 0, math.cos(h)]])
    return r_yaw @ r_pitch


def focal_px(size=TILE_SIZE, fov_deg=TILE_FOV_DEG):
    return (size / 2) / math.tan(math.radians(fov_deg) / 2)


def dirs_to_equirect(d):
    """World unit directions (..., 3) -> (x_norm, y_norm) in the image frame."""
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return np.mod(0.5 + lon / (2 * math.pi), 1.0), 0.5 - lat / math.pi


def equirect_to_dirs(x_norm, y_norm):
    """(x_norm, y_norm) -> world unit directions (..., 3); inverse of dirs_to_equirect."""
    lon = (np.asarray(x_norm) - 0.5) * 2 * math.pi
    lat = (0.5 - np.asarray(y_norm)) * math.pi
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)],
                    axis=-1)


def tile_pixel_dirs(heading_deg, pitch_deg, size=TILE_SIZE, fov_deg=TILE_FOV_DEG):
    """World unit ray through every tile pixel centre, shape (size, size, 3); row 0 is the
    top of the tile, column 0 its left."""
    f = focal_px(size, fov_deg)
    c = np.arange(size) + 0.5 - size / 2
    xc, yc = np.meshgrid(c / f, -c / f)          # yc: up is positive, rows go down
    cam = np.stack([xc, yc, np.ones_like(xc)], axis=-1)
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
    return cam @ tile_rotation(heading_deg, pitch_deg).T


def tile_pixel_to_equirect(heading_deg, pitch_deg, size=TILE_SIZE, fov_deg=TILE_FOV_DEG):
    return dirs_to_equirect(tile_pixel_dirs(heading_deg, pitch_deg, size, fov_deg))


def equirect_to_tile(x_norm, y_norm, heading_deg, pitch_deg, size=TILE_SIZE,
                     fov_deg=TILE_FOV_DEG):
    """(col, row, valid, angle_rad): the tile pixel containing each equirect direction,
    whether the tile sees it, and the angle between the ray and the tile's optical axis."""
    d = equirect_to_dirs(x_norm, y_norm)
    cam = d @ tile_rotation(heading_deg, pitch_deg)        # world -> camera (R^T d)
    cz = cam[..., 2]
    f = focal_px(size, fov_deg)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = f * cam[..., 0] / cz + size / 2
        v = size / 2 - f * cam[..., 1] / cz
    col = np.floor(np.where(cz > 0, u, -1)).astype(np.int64)
    row = np.floor(np.where(cz > 0, v, -1)).astype(np.int64)
    valid = (cz > 0) & (col >= 0) & (col < size) & (row >= 0) & (row < size)
    angle = np.arccos(np.clip(cz, -1.0, 1.0))
    return col, row, valid, angle


def sample_bilinear(img, x_norm, y_norm):
    """Bilinear sample of an equirect image (H, W[, C]) at image-frame coordinates:
    wraps across the seam in x, clamps in y."""
    h, w = img.shape[:2]
    px = np.asarray(x_norm) * w - 0.5
    py = np.clip(np.asarray(y_norm) * h - 0.5, 0, h - 1)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx, fy = px - x0, py - y0
    x0w, x1w = np.mod(x0, w), np.mod(x0 + 1, w)
    y1 = np.minimum(y0 + 1, h - 1)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    a = img[y0, x0w].astype(np.float64)
    b = img[y0, x1w].astype(np.float64)
    c = img[y1, x0w].astype(np.float64)
    d = img[y1, x1w].astype(np.float64)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def tile_specs():
    """[(name, heading, pitch)] in a fixed order."""
    return [(f'h{h:03d}_p{p:+03d}', h, p)
            for p in TILE_PITCHES_DEG for h in TILE_HEADINGS_DEG]


def render_tiles(pano_rgb):
    """{name: uint8 (TILE_SIZE, TILE_SIZE, 3)} for a (PANO_H, PANO_W, 3) array."""
    out = {}
    for name, h, p in tile_specs():
        x, y = tile_pixel_to_equirect(h, p)
        out[name] = np.clip(np.rint(sample_bilinear(pano_rgb, x, y)), 0, 255).astype(np.uint8)
    return out


def grid_centres(w=GRID_W, h=GRID_H):
    """(x_norm, y_norm) of every grid pixel centre, shape (h, w)."""
    return np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)


class StitchGeometry:
    """Per tile: the flat index of the tile pixel each grid pixel falls in, whether the
    tile sees it, and the per-pixel tile order by angle to the optical axis. Identical for
    every pano, so computed once."""

    def __init__(self, w=GRID_W, h=GRID_H, tile_size=TILE_SIZE):
        x, y = grid_centres(w, h)
        self.shape = (h, w)
        self.names = [n for n, _, _ in tile_specs()]
        flat, valid, angle = [], [], []
        for _, hd, pt in tile_specs():
            col, row, ok, ang = equirect_to_tile(x, y, hd, pt, size=tile_size)
            flat.append(np.where(ok, row * tile_size + col, 0).ravel())
            valid.append(ok.ravel())
            angle.append(np.where(ok, ang, np.inf).ravel())
        self.flat = np.stack(flat)
        self.valid = np.stack(valid)
        self.order = np.argsort(np.stack(angle), axis=0, kind='stable')
        self.n_cover = self.valid.sum(axis=0).reshape(self.shape)


def vote(tile_labels, geom):
    """Stitch {name: (TILE_SIZE, TILE_SIZE) uint8 label map} onto the grid.

    Each grid pixel takes the label most tiles that see it agree on; a tie goes to the
    tied label of the tile nearest in angle. Unseen pixels get NONE_LABEL.
    Returns (labels (h, w) uint8, votes (h, w) uint8 = how many tiles agreed).
    """
    t = len(geom.names)
    n = geom.flat.shape[1]
    stack = np.full((t, n), NONE_LABEL, dtype=np.uint8)
    for i, name in enumerate(geom.names):
        lab = tile_labels[name].ravel()
        stack[i] = np.where(geom.valid[i], lab[geom.flat[i]], NONE_LABEL)
    counts = np.zeros((t, n), dtype=np.uint8)
    for i in range(t):
        eq = stack == stack[i]
        counts[i] = np.where(stack[i] != NONE_LABEL, eq.sum(axis=0), 0)
    best = counts.max(axis=0)
    out = np.full(n, NONE_LABEL, dtype=np.uint8)
    done = best == 0
    cols = np.arange(n)
    for rank in range(t):
        ti = geom.order[rank]
        take = ~done & (counts[ti, cols] == best)
        out[take] = stack[ti, cols][take]
        done |= take
    return out.reshape(geom.shape), best.reshape(geom.shape)


def collapse_table(id2label):
    """uint8 id -> index into SEG_GROUPS (NONE_LABEL -> NONE). Refuses an unmapped name."""
    missing = sorted(v for v in id2label.values() if v not in COLLAPSE)
    if missing:
        raise SystemExit(f'Vistas labels without a group in COLLAPSE: {missing}')
    table = np.full(256, SEG_GROUPS.index(OTHER), dtype=np.uint8)
    for i, name in id2label.items():
        table[int(i)] = SEG_GROUPS.index(COLLAPSE[name])
    table[NONE_LABEL] = SEG_GROUPS.index(NONE)
    return table


# --- sample ------------------------------------------------------------------------------

def payload_path(run_root, city, pano_id):
    return run_root / city / 'depth' / f'{pano_id}.json.gz'


def load_city(city, args):
    """(verdict panos, bundle ops, run panos by id) for one city."""
    verdicts, bundle_ops, run_panos = es.load_city_files(
        city, args.benchmark_root, args.run_root / city, read_heights=False)
    return verdicts, bundle_ops, {p.pano_id: p for p in run_panos}


def cmd_sample(args):
    rows, counts_rows = [], []
    for city in args.cities:
        verdicts, bundle_ops, by_id = load_city(city, args)
        counts, warnings = es.gt_counts(), []
        counts['gt_panos'] = len(verdicts)
        status_n = Counter()
        for pid, entry, run_pano, ops, in_pool in es.judged_gt_panos(
                verdicts, bundle_ops, by_id, counts, warnings):
            jpg = args.benchmark_root / city / 'panos' / f'{pid}.jpg'
            ppath = payload_path(args.run_root, city, pid)
            if not jpg.exists():
                status_n['no_jpeg'] += 1
                continue
            if not ppath.exists():
                status_n[dad.NO_FILE] += 1
                continue
            status, _ = dad.measure_pano((str(ppath), []))
            status_n[status] += 1
            if status != depthlib.MEASURED:
                continue
            rows.append({'city': city, 'pano_id': pid, 'label_uid': f'{city}:{pid}',
                         'capture_date': run_pano.capture_date,
                         'n_det_0p55': len(ops), 'in_pool': int(bool(in_pool)),
                         'jpeg_sha256': sha256_file(jpg),
                         'payload_sha256': sha256_file(ppath)})
        counts_rows.append({'city': city, **counts, 'n_warnings': len(warnings),
                            **{f'payload_{k}': v for k, v in sorted(status_n.items())},
                            'n_sample': sum(r['city'] == city for r in rows)})
        print(f'{city}: judged {counts["judged"]}, skipped {counts["skipped"]}, '
              f'partial {counts["partial"]}; payload {dict(status_n)}; '
              f'sample {counts_rows[-1]["n_sample"]}', flush=True)
    uids = [r['label_uid'] for r in rows]
    assert len(uids) == len(set(uids)), 'duplicate (city, pano_id) in the sample'
    pd = pooled_dir(args.out_root)
    write_csv(pd / 'sample.csv', rows)
    write_csv(pd / 'sample_counts.csv', counts_rows,
              sorted({k for r in counts_rows for k in r}, key=lambda k: (k != 'city', k)))
    print(f'sample: {len(rows)} panos -> {pd / "sample.csv"}')


# --- tiles -------------------------------------------------------------------------------

def load_pano_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((PANO_W, PANO_H), Image.LANCZOS))


def tiles_one(task):
    """Worker: render one pano's 16 tiles + the direct-arm equirect (skips done work)."""
    src, tdir, direct = (Path(t) for t in task)
    names = [nm for nm, _, _ in tile_specs()]
    if all((tdir / f'{nm}.jpg').exists() for nm in names) and direct.exists():
        return 0
    rgb = load_pano_rgb(src)
    tdir.mkdir(parents=True, exist_ok=True)
    for nm, arr in render_tiles(rgb).items():
        Image.fromarray(arr).save(tdir / f'{nm}.jpg', quality=JPEG_QUALITY)
    direct.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).resize((DIRECT_W, DIRECT_H), Image.LANCZOS).save(
        direct, quality=JPEG_QUALITY)
    return 1


def cmd_tiles(args):
    from concurrent.futures import ProcessPoolExecutor
    wd = work_dir(args)
    sample = read_sample(args)
    if args.limit:
        sample = sample[:args.limit]
    tasks = [(str(args.benchmark_root / r['city'] / 'panos' / f"{r['pano_id']}.jpg"),
              str(wd / 'tiles' / r['city'] / r['pano_id']),
              str(wd / 'direct' / r['city'] / f"{r['pano_id']}.jpg")) for r in sample]
    t0 = time.time()
    manifest = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, done in enumerate(ex.map(tiles_one, tasks), 1):
            manifest.append(done)
            if n % 25 == 0:
                print(f'  tiles {n}/{len(tasks)} ({time.time() - t0:.0f} s)', flush=True)
    meta = {'pano_size': [PANO_W, PANO_H], 'resize': 'PIL LANCZOS',
            'tile_size': TILE_SIZE, 'tile_fov_deg': TILE_FOV_DEG,
            'headings_deg': list(TILE_HEADINGS_DEG), 'pitches_deg': list(TILE_PITCHES_DEG),
            'sampling': 'bilinear, x wraps, y clamps', 'jpeg_quality': JPEG_QUALITY,
            'direct_size': [DIRECT_W, DIRECT_H], 'n_panos': len(manifest)}
    (wd / 'tiles_manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'tiles: {len(manifest)} panos x {len(tile_specs())} tiles -> {wd / "tiles"}')


# --- segment (the GPU step) ---------------------------------------------------------------

def cmd_segment(args):
    try:
        import torch
        import torch.nn.functional as F
        import transformers
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError as e:
        raise SystemExit(f'`segment` needs torch + transformers ({e}); they are deliberately '
                         f'not in requirements.txt: pip install torch transformers')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if (args.fp16 and device == 'cuda') else torch.float32
    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype).to(device).eval()
    revision = getattr(model.config, '_commit_hash', None) or args.revision
    files = sorted(p for p in args.inp.rglob('*.jpg'))
    todo = [p for p in files
            if not (args.out / p.relative_to(args.inp)).with_suffix('.png').exists()]
    target = tuple(int(v) for v in args.target.split('x')) if args.target else None
    print(f'segment: {len(files)} images, {len(todo)} to do, device {device}, {dtype}, '
          f'revision {revision}', flush=True)
    t0 = time.time()
    for b in range(0, len(todo), args.batch_size):
        batch = todo[b:b + args.batch_size]
        images = [Image.open(p).convert('RGB') for p in batch]
        inputs = processor(images=images, return_tensors='pt', do_resize=False)
        with torch.inference_mode():
            out = model(pixel_values=inputs['pixel_values'].to(device, dtype),
                        pixel_mask=inputs['pixel_mask'].to(device))
            cls = out.class_queries_logits.float().softmax(-1)[..., :-1]   # (B, Q, C)
            masks = out.masks_queries_logits.float().sigmoid()             # (B, Q, h, w)
            sem = torch.einsum('bqc,bqhw->bchw', cls, masks)
            for img, path, s in zip(images, batch, sem):
                w, h = target or img.size
                lab = F.interpolate(s[None], size=(h, w), mode='bilinear',
                                    align_corners=False)[0].argmax(0)
                dst = (args.out / path.relative_to(args.inp)).with_suffix('.png')
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(lab.to(torch.uint8).cpu().numpy()).save(dst)
        done = b + len(batch)
        if done % (args.batch_size * 50) < args.batch_size or done == len(todo):
            rate = done / max(1e-9, time.time() - t0)
            print(f'  {done}/{len(todo)} ({rate:.2f} img/s, peak '
                  f'{torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0:.1f}'
                  f' GiB)', flush=True)
    outs = sorted(args.out.rglob('*.png'))
    manifest = {'model_id': args.model, 'model_revision': revision,
                'id2label': {int(k): v for k, v in model.config.id2label.items()},
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                'dtype': str(dtype), 'do_resize': False, 'target': args.target,
                'inference': 'softmax(class)[:-1] x sigmoid(mask) at mask resolution, '
                             'bilinear to target, argmax',
                'n_inputs': len(files), 'n_outputs': len(outs),
                'outputs_sha256': {str(p.relative_to(args.out)).replace('\\', '/'):
                                   sha256_file(p) for p in outs}}
    (args.out / 'segment_manifest.json').write_text(json.dumps(manifest, indent=1),
                                                    encoding='utf-8')
    print(f'segment: {len(outs)} label maps -> {args.out}')


# --- stitch ------------------------------------------------------------------------------

def cmd_stitch(args):
    wd = work_dir(args)
    geom = StitchGeometry()
    uncovered = int((geom.n_cover == 0).sum())
    ys = np.nonzero((geom.n_cover == 0).any(axis=1))[0]
    print(f'stitch: {uncovered} of {GRID_W * GRID_H} grid pixels seen by no tile '
          f'(rows {ys.min() if len(ys) else None}-{ys.max() if len(ys) else None}, '
          f'y_norm >= {(ys.min() + 0.5) / GRID_H if len(ys) else None})', flush=True)
    sample = read_sample(args)
    for n, r in enumerate(sample, 1):
        city, pid = r['city'], r['pano_id']
        dst = wd / 'stitched' / city / f'{pid}.png'
        if dst.exists():
            continue
        src = wd / 'tile_labels' / city / pid
        labels = {nm: np.asarray(Image.open(src / f'{nm}.png')) for nm in geom.names}
        lab, votes = vote(labels, geom)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(lab).save(dst)
        if n % 50 == 0:
            print(f'  stitched {n}/{len(sample)}', flush=True)
    np.save(wd / 'stitch_coverage.npy', geom.n_cover)
    print(f'stitch: {len(sample)} panos -> {wd / "stitched"}')


# --- compare -----------------------------------------------------------------------------

def depth_class_table(pix):
    """plane index -> index into DEPTH_CLASSES (dad.plane_class, floor split by stand-in)."""
    n = max(max(pix.counts), len(pix.payload.planes)) + 1
    table = np.zeros(n, dtype=np.uint8)
    for idx in range(n):
        cls = dad.plane_class(pix, idx)
        if cls == dad.FLOOR and depthlib.is_standin(pix.payload.planes[idx]):
            cls = FLOOR_STANDIN
        table[idx] = DEPTH_CLASSES.index(cls)
    return table


def depth_class_at(pix, x, y):
    """(depth class, classify() fields) at one image-frame coordinate."""
    out = dad.classify(pix, x, y)
    cls = out['plane_class']
    if cls == dad.FLOOR:
        plane, _, _ = depthlib._plane_at(pix.payload, x, y)
        if depthlib.is_standin(plane):
            cls = FLOOR_STANDIN
    return cls, out


def flat_range(pose, y_norm):
    """Flat-raycast horizontal range at 2.6 m (geo), None beyond 25 m or at the horizon."""
    g = geo.detection_ground_point(pose, 0.5, float(y_norm), max_range_m=math.inf,
                                   apply_pose=False)
    if g is None or g.range_m > geo.DEFAULT_MAX_RANGE_M:
        return None
    return g.range_m


def range_label(r):
    if r is None:
        return BEYOND
    for lo, hi in RANGE_BINS:
        if lo <= r < hi:
            return f'{lo}-{hi}'
    return BEYOND


LAT_BANDS = list(range(-90, 90, 10))     # band start, degrees of latitude (up positive)
SEAM_HALF_WIDTH = 0.03                  # x_norm within this of the seam = seam band


def lat_band_of_rows(h=GRID_H):
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    return np.clip(((lat + 90) // 10).astype(int), 0, len(LAT_BANDS) - 1)


def detection_items(city, pid, entry, run_pano, ops):
    """Every stored detection >= 0.30 on the pano, the 0.55 ones with their verdict, plus
    the reviewer's missed marks."""
    verdict_of = {stored_i: v for v, (stored_i, _, _, _) in zip(entry['dets'], ops)}
    items = []
    for i, x, y, c in run_pano.detections:
        if c < OPERATIONAL_CONFIDENCE:
            continue
        v = verdict_of.get(i)
        group = dad._VERDICT_GROUP[v] if c >= BENCHMARK_CONFIDENCE else 'band_0p30'
        items.append({'city': city, 'pano_id': pid, 'kind': 'det', 'det_index': i,
                      'x': x, 'y': y, 'confidence': c, 'gt_group': group})
    for k, m in enumerate(entry.get('missed', ())):
        items.append({'city': city, 'pano_id': pid, 'kind': 'missed', 'det_index': k,
                      'x': m['x'], 'y': m['y'], 'confidence': None,
                      'gt_group': 'unsure_missed' if m.get('unsure') else 'missed'})
    return items


def compare_pano(pix, run_pano, tiled, direct, fine_names, table_seg, ctx):
    """Accumulate one pano into ctx; returns (per-pano row, detection-free curb rows)."""
    payload = pix.payload
    pw, ph = payload.width, payload.height
    ind = np.asarray(payload.indices, dtype=np.int64).reshape(ph, pw)
    dtab = depth_class_table(pix)
    # (A) every payload pixel below the horizon, read at the grid pixel whose centre falls
    # in it (the grid is 2x the payload in each axis).
    sy, sx = GRID_H // ph, GRID_W // pw
    rows = np.arange(ph // 2, ph)
    dcls = dtab[ind[rows]]                                   # (rows, pw)
    grid = tiled[rows * sy][:, np.arange(pw) * sx]
    scls = table_seg[grid]
    pose = fs.pano_pose(run_pano, fs.POSE_OFF)
    ranges = [flat_range(pose, (r * sy + 0.5) / GRID_H) for r in rows]
    rbin = np.array([RANGE_LABELS.index(range_label(r)) for r in ranges])[:, None]
    rbin = np.broadcast_to(rbin, dcls.shape)
    key = (dcls.astype(np.int64) * len(SEG_GROUPS) + scls) * len(RANGE_LABELS) + rbin
    counts = np.bincount(key.ravel(), minlength=len(DEPTH_CLASSES) * len(SEG_GROUPS)
                         * len(RANGE_LABELS)).reshape(len(DEPTH_CLASSES), len(SEG_GROUPS),
                                                      len(RANGE_LABELS))
    ctx['agree'] += counts
    surf = np.isin(dcls, [DEPTH_CLASSES.index(c) for c in DEPTH_SURFACE])
    ws = np.isin(scls, [SEG_GROUPS.index(g) for g in SURFACE_GROUPS])
    obj = scls == SEG_GROUPS.index(OBJECT)
    pano_row = {'n_surface_px': int(surf.sum()),
                'p_walkroad_given_surface': float((surf & ws).sum() / max(1, surf.sum())),
                'p_object_given_surface': float((surf & obj).sum() / max(1, surf.sum()))}
    # trap: tiled vs direct, every grid pixel, by latitude band and seam
    tg, dg = table_seg[tiled], table_seg[direct]
    band = np.broadcast_to(lat_band_of_rows()[:, None], tg.shape)
    x = (np.arange(GRID_W) + 0.5) / GRID_W
    seam = np.broadcast_to(((x < SEAM_HALF_WIDTH) | (x > 1 - SEAM_HALF_WIDTH))[None, :],
                           tg.shape)
    seen = tg != SEG_GROUPS.index(NONE)
    for arm, g in (('tiled', tg), ('direct', dg)):
        k = (band * 2 + seam) * len(SEG_GROUPS) + g
        ctx[f'trap_{arm}'] += np.bincount(k.ravel(), minlength=len(LAT_BANDS) * 2
                                          * len(SEG_GROUPS)).reshape(len(LAT_BANDS), 2,
                                                                     len(SEG_GROUPS))
    k = (band * 2 + seam)[seen]
    ctx['trap_agree'] += np.bincount(k[(tg == dg)[seen]], minlength=len(LAT_BANDS) * 2
                                     ).reshape(len(LAT_BANDS), 2)
    ctx['trap_seen'] += np.bincount(k, minlength=len(LAT_BANDS) * 2).reshape(
        len(LAT_BANDS), 2)
    # (B) curb height: Sidewalk (headline), all WALK, and ROAD (control) on a floor plane
    fine = tiled[rows * sy][:, np.arange(pw) * sx]
    in_range = np.broadcast_to(np.array([r is not None for r in ranges])[:, None],
                               dcls.shape)
    floor_any = np.isin(dcls, [DEPTH_CLASSES.index(dad.FLOOR),
                               DEPTH_CLASSES.index(FLOOR_STANDIN)])
    groups = {'sidewalk': fine == fine_names.index('Sidewalk'),
              'walk': scls == SEG_GROUPS.index(WALK),
              'road': scls == SEG_GROUPS.index(ROAD)}
    rng = random.Random(f'{SEED}:{run_pano.pano_id}')
    curb = []
    for gname, gm in groups.items():
        rr, cc = np.nonzero(gm & floor_any & in_range)
        pano_row[f'n_{gname}_px'] = int((gm).sum())
        pano_row[f'n_{gname}_floor_px'] = len(rr)
        pano_row[f'n_{gname}_ground_px'] = int((gm & (dcls == DEPTH_CLASSES.index(
            dad.GROUND)) & in_range).sum())
        pick = list(range(len(rr)))
        if len(pick) > CURB_MAX_PIXELS:
            pick = sorted(rng.sample(pick, CURB_MAX_PIXELS))
        for i in pick:
            r, c = int(rows[rr[i]]), int(cc[i])
            xn, yn = (c + 0.5) / pw, (r + 0.5) / ph
            cls, f = depth_class_at(pix, xn, yn)
            off = f.get('offset_local_m')
            if off is None:
                continue
            curb.append({'group': gname, 'standin_plane': int(cls == FLOOR_STANDIN),
                         'standin_ref': int(f.get('ref_tilt_deg') == 0.0),
                         'offset_local_m': off,
                         'offset_level_ref_m': f.get('offset_local_level_ref_m'),
                         'range_bin': range_label(ranges[rr[i]])})
    return pano_row, curb


def curb_summary(curb_by_pano):
    """Per group and reading: the median across panos of per-pano median offsets."""
    out = []
    for group in ('sidewalk', 'walk', 'road'):
        for reading in ('measured', 'with_standins'):
            meds, npx = [], 0
            for pid, rows in curb_by_pano.items():
                v = [r['offset_local_m'] for r in rows if r['group'] == group
                     and (reading == 'with_standins'
                          or not (r['standin_plane'] or r['standin_ref']))]
                npx += len(v)
                if len(v) >= CURB_MIN_PIXELS:
                    meds.append(float(np.median(v)))
            med, lo, hi = dad.median_ci(meds)
            allv = [r['offset_local_m'] for rows in curb_by_pano.values() for r in rows
                    if r['group'] == group and (reading == 'with_standins' or not (
                        r['standin_plane'] or r['standin_ref']))]
            out.append({'group': group, 'reading': reading, 'n_panos': len(meds),
                        'n_pixels': npx, 'median_of_pano_medians': med,
                        'median_lo': lo, 'median_hi': hi,
                        'pixel_p10': float(np.percentile(allv, 10)) if allv else None,
                        'pixel_median': float(np.median(allv)) if allv else None,
                        'pixel_p90': float(np.percentile(allv, 90)) if allv else None,
                        'share_in_claim_band': (sum(CLAIM_BAND_M[0] <= v < CLAIM_BAND_M[1]
                                                    for v in allv) / len(allv))
                        if allv else None})
    return out


def new_ctx():
    return {'agree': np.zeros((len(DEPTH_CLASSES), len(SEG_GROUPS), len(RANGE_LABELS)),
                              dtype=np.int64),
            'trap_tiled': np.zeros((len(LAT_BANDS), 2, len(SEG_GROUPS)), dtype=np.int64),
            'trap_direct': np.zeros((len(LAT_BANDS), 2, len(SEG_GROUPS)), dtype=np.int64),
            'trap_agree': np.zeros((len(LAT_BANDS), 2), dtype=np.int64),
            'trap_seen': np.zeros((len(LAT_BANDS), 2), dtype=np.int64)}


def add_ctx(a, b):
    for k in a:
        a[k] += b[k]


def agreement_rows(label, agree):
    """Long-form rows of the depth x segmenter matrix: per range bin and 'all'."""
    rows = []
    for ri, rl in enumerate(RANGE_LABELS + ['all']):
        m = agree.sum(axis=2) if rl == 'all' else agree[:, :, ri]
        for di, dc in enumerate(DEPTH_CLASSES):
            tot = int(m[di].sum())
            rows.append({'city': label, 'range_bin': rl, 'depth_class': dc, 'n_px': tot,
                         **{f'n_{g}': int(m[di, gi]) for gi, g in enumerate(SEG_GROUPS)},
                         **{f'share_{g}': (m[di, gi] / tot if tot else None)
                            for gi, g in enumerate(SEG_GROUPS)}})
    return rows


def headline(agree):
    """P(WALK or ROAD | depth surface), OBJECT share of depth surface, and the reverse
    reading P(depth surface | WALK or ROAD), over every range (and within 25 m).

    Denominators are the pixels some tile SEES: the NONE column (the far nadir, below
    ~-79 deg, within ~0.5 m of the car) is reported as its own count, never as a
    disagreement. (Clarified 2026-09-30, before the first scoring run.)"""
    out = {}
    for tag, m in (('all', agree.sum(axis=2)),
                   ('le25m', agree[:, :, :len(RANGE_BINS)].sum(axis=2))):
        out[f'{tag}_n_unseen_px'] = int(m[:, SEG_GROUPS.index(NONE)].sum())
        m = m.copy()
        m[:, SEG_GROUPS.index(NONE)] = 0
        surf = [DEPTH_CLASSES.index(c) for c in DEPTH_SURFACE]
        wr = [SEG_GROUPS.index(g) for g in SURFACE_GROUPS]
        s = m[surf].sum()
        out[f'{tag}_n_surface_px'] = int(s)
        out[f'{tag}_p_walkroad_given_surface'] = float(m[surf][:, wr].sum() / s) if s else None
        for g in (OBJECT, STRUCTURE, OTHER, SKY):
            out[f'{tag}_p_{g.lower()}_given_surface'] = \
                float(m[surf][:, SEG_GROUPS.index(g)].sum() / s) if s else None
        w = m[:, wr].sum()
        out[f'{tag}_p_surface_given_walkroad'] = float(m[surf][:, wr].sum() / w) if w else None
        o = m[:, SEG_GROUPS.index(OBJECT)].sum()
        out[f'{tag}_p_surface_given_object'] = \
            float(m[surf][:, SEG_GROUPS.index(OBJECT)].sum() / o) if o else None
        wall = m[DEPTH_CLASSES.index(dad.NON_HORIZONTAL)].sum()
        out[f'{tag}_p_walkroad_given_nonhorizontal'] = \
            float(m[DEPTH_CLASSES.index(dad.NON_HORIZONTAL), wr].sum() / wall) if wall else None
    return out


def trap_rows(label, ctx):
    rows = []
    for bi, b in enumerate(LAT_BANDS):
        for si, sname in enumerate(('interior', 'seam')):
            seen = int(ctx['trap_seen'][bi, si])
            row = {'city': label, 'lat_band_deg': f'{b}..{b + 10}', 'band': sname,
                   'n_seen_px': seen,
                   'agreement': ctx['trap_agree'][bi, si] / seen if seen else None}
            for arm in ('tiled', 'direct'):
                t = ctx[f'trap_{arm}'][bi, si]
                tot = t.sum()
                for gi, g in enumerate(SEG_GROUPS):
                    row[f'{arm}_share_{g}'] = t[gi] / tot if tot else None
            rows.append(row)
    return rows


DET_FIELDS = ['city', 'pano_id', 'kind', 'det_index', 'x', 'y', 'confidence', 'gt_group',
              'seg_group', 'seg_class', 'direct_group', 'direct_class', 'depth_class',
              'plane_tilt_deg', 'offset_local_m', 'range_flat_2p6_m', 'on_camera_rig']


def cmd_compare(args):
    wd = work_dir(args)
    manifest = json.loads((wd / 'tile_labels' / 'segment_manifest.json').read_text(
        encoding='utf-8'))
    id2label = {int(k): v for k, v in manifest['id2label'].items()}
    fine_names = [id2label[i] for i in range(len(id2label))]
    table_seg = collapse_table(id2label)
    sample = read_sample(args)
    by_city = defaultdict(list)
    for r in sample:
        by_city[r['city']].append(r['pano_id'])
    pooled = new_ctx()
    pooled_det, pooled_curb, pooled_pano = [], {}, []
    head_rows, agree_all, trap_all, curb_all = [], [], [], []
    for city in args.cities:
        verdicts, bundle_ops, by_id = load_city(city, args)
        counts, warnings = es.gt_counts(), []
        want = set(by_city[city])
        ctx, dets, curb_by_pano, pano_rows = new_ctx(), [], {}, []
        for pid, entry, run_pano, ops, in_pool in es.judged_gt_panos(
                verdicts, bundle_ops, by_id, counts, warnings):
            if pid not in want:
                continue
            pix = dad.PayloadIndex(dad.read_payload(payload_path(args.run_root, city, pid)))
            assert pix.status() == depthlib.MEASURED, f'{city}:{pid} is not measured'
            tiled = np.asarray(Image.open(wd / 'stitched' / city / f'{pid}.png'))
            direct = np.asarray(Image.open(wd / 'direct_labels' / city / f'{pid}.png'))
            assert tiled.shape == direct.shape == (GRID_H, GRID_W), (city, pid)
            c1 = new_ctx()
            prow, curb = compare_pano(pix, run_pano, tiled, direct, fine_names, table_seg, c1)
            add_ctx(ctx, c1)
            pano_rows.append({'city': city, 'pano_id': pid, **prow})
            curb_by_pano[f'{city}:{pid}'] = curb
            pose = fs.pano_pose(run_pano, fs.POSE_OFF)
            for it in detection_items(city, pid, entry, run_pano, ops):
                x, y = it['x'], it['y']
                col = min(GRID_W - 1, int(x * GRID_W))
                row = min(GRID_H - 1, int(y * GRID_H))
                lt, ld = int(tiled[row, col]), int(direct[row, col])
                cls, f = depth_class_at(pix, x, y)
                g = geo.detection_ground_point(pose, x, y, max_range_m=math.inf,
                                               apply_pose=False)
                it.update({'seg_group': SEG_GROUPS[table_seg[lt]],
                           'seg_class': id2label.get(lt, 'NONE'),
                           'direct_group': SEG_GROUPS[table_seg[ld]],
                           'direct_class': id2label.get(ld, 'NONE'),
                           'depth_class': cls, 'plane_tilt_deg': f.get('plane_tilt_deg'),
                           'offset_local_m': f.get('offset_local_m'),
                           'range_flat_2p6_m': g.range_m if g else None,
                           'on_camera_rig': int(dad_on_rig(y))})
                dets.append(it)
        missing = want - {r['pano_id'] for r in pano_rows}
        if missing:
            raise SystemExit(f'{city}: {len(missing)} sampled panos no longer pass the GT '
                             f'join (drift?): {sorted(missing)[:5]}')
        od = city_out(args.out_root, city)
        agree_all += agreement_rows(city, ctx['agree'])
        trap_all += trap_rows(city, ctx)
        cs = curb_summary(curb_by_pano)
        curb_all += [{'city': city, **r} for r in cs]
        head_rows.append({'city': city, 'n_panos': len(pano_rows), **headline(ctx['agree'])})
        write_csv(od / 'detections.csv', dets, DET_FIELDS)
        write_csv(od / 'panos.csv', pano_rows)
        write_csv(od / 'agreement.csv', agreement_rows(city, ctx['agree']))
        write_csv(od / 'curb_height.csv', [{'city': city, **r} for r in cs])
        add_ctx(pooled, ctx)
        pooled_det += dets
        pooled_curb.update(curb_by_pano)
        pooled_pano += pano_rows
        print(f'{city}: {len(pano_rows)} panos, {len(dets)} detection/mark rows; '
              f'P(WALK|ROAD | depth surface) {head_rows[-1]["all_p_walkroad_given_surface"]:.3f}',
              flush=True)
    pd = pooled_dir(args.out_root)
    agree_all += agreement_rows('pooled', pooled['agree'])
    trap_all += trap_rows('pooled', pooled)
    curb_all += [{'city': 'pooled', **r} for r in curb_summary(pooled_curb)]
    head_rows.append({'city': 'pooled', 'n_panos': len(pooled_pano),
                      **headline(pooled['agree'])})
    write_csv(pd / 'agreement.csv', agree_all)
    write_csv(pd / 'trap.csv', trap_all)
    write_csv(pd / 'curb_height.csv', curb_all)
    write_csv(pd / 'headline.csv', head_rows)
    write_csv(pd / 'detections.csv', pooled_det, DET_FIELDS)
    write_csv(pd / 'detection_classes.csv', detection_class_rows(pooled_det))
    write_masks_manifest(args, wd, sample)
    v = cmd_verdict(args)
    write_reports(args, head_rows, curb_all, trap_all, v)


def dad_on_rig(y):
    from detectors import on_camera_rig
    return on_camera_rig(y)


def detection_class_rows(dets):
    """Segmenter-group shares (tiled and direct) per city x tier x GT group, with the
    non_surface share and its Wilson interval."""
    rows = []
    cities = sorted({d['city'] for d in dets}) + ['pooled']
    groups = ['true', 'false', 'missed', 'unsure', 'duplicate', 'unsure_missed',
              'all_0p55', 'all_0p30']
    for city in cities:
        cd = [d for d in dets if city == 'pooled' or d['city'] == city]
        for g in groups:
            if g == 'all_0p55':
                gi = [d for d in cd if d['kind'] == 'det' and d['gt_group'] != 'band_0p30']
            elif g == 'all_0p30':
                gi = [d for d in cd if d['kind'] == 'det']
            else:
                gi = [d for d in cd if d['gt_group'] == g]
            for arm in ('seg', 'direct'):
                c = Counter(d[f'{arm}_group'] for d in gi)
                n = len(gi)
                ns = n - c[WALK] - c[ROAD]
                lo, hi = es.wilson(ns, n)
                fine = Counter(d[f'{arm}_class'] for d in gi)
                rows.append({'city': city, 'group': g, 'arm': 'tiled' if arm == 'seg'
                             else 'direct', 'n': n, 'n_non_surface': ns,
                             'share_non_surface': ns / n if n else None,
                             'non_surface_lo': lo, 'non_surface_hi': hi,
                             **{f'share_{k}': (c[k] / n if n else None) for k in SEG_GROUPS},
                             'share_curb_cut': fine['Curb Cut'] / n if n else None,
                             'top_classes': ';'.join(f'{k}={v}'
                                                     for k, v in fine.most_common(6))})
    return rows


def write_masks_manifest(args, wd, sample):
    seg = json.loads((wd / 'tile_labels' / 'segment_manifest.json').read_text(
        encoding='utf-8'))
    dseg = json.loads((wd / 'direct_labels' / 'segment_manifest.json').read_text(
        encoding='utf-8'))
    stitched = {f'{r["city"]}/{r["pano_id"]}.png':
                sha256_file(wd / 'stitched' / r['city'] / f'{r["pano_id"]}.png')
                for r in sample}
    agg = hashlib.sha256()
    for k in sorted(seg['outputs_sha256']):
        agg.update(f'{k} {seg["outputs_sha256"][k]}\n'.encode())
    man = {'model_id': seg['model_id'], 'model_revision': seg['model_revision'],
           'torch': seg['torch'], 'transformers': seg['transformers'],
           'device': seg['device'], 'dtype': seg['dtype'], 'inference': seg['inference'],
           'tiles': json.loads((wd / 'tiles_manifest.json').read_text(encoding='utf-8')),
           'n_tile_label_maps': seg['n_outputs'],
           'tile_label_maps_aggregate_sha256': agg.hexdigest(),
           'direct_label_maps_sha256': dseg['outputs_sha256'],
           'stitched_label_maps_sha256': stitched}
    (pooled_dir(args.out_root) / 'masks_manifest.json').write_text(
        json.dumps(man, indent=1, sort_keys=True), encoding='utf-8')


# --- verdict -----------------------------------------------------------------------------

def verdict(true_n, true_ns, false_n, false_ns, true_surface_n=None):
    """The pre-registered FP reading (C). Pure.

    Example:
        >>> verdict(700, 70, 42, 30)['rule_c']
        'SUPPORTED'
    """
    t = true_ns / true_n if true_n else None
    f = false_ns / false_n if false_n else None
    tlo, thi = es.wilson(true_ns, true_n)
    flo, fhi = es.wilson(false_ns, false_n)
    ok = (t is not None and f is not None and f - t >= RULE_C_MARGIN - 1e-12
          and flo > thi)
    return {'rule_c': 'SUPPORTED' if ok else 'NOT SUPPORTED',
            'true_n': true_n, 'true_non_surface': true_ns, 'true_share': t,
            'true_lo': tlo, 'true_hi': thi,
            'false_n': false_n, 'false_non_surface': false_ns, 'false_share': f,
            'false_lo': flo, 'false_hi': fhi,
            'gap': (f - t) if t is not None and f is not None else None,
            'margin': RULE_C_MARGIN, 'underpowered': (fhi - flo) > RULE_C_MARGIN}


def cmd_verdict(args):
    pd = pooled_dir(args.out_root)
    src = pd if (pd / 'detections.csv').exists() else FIG_DIR / 'data'
    dets = read_csv(src / 'detections.csv')
    curb = read_csv(src / 'curb_height.csv')
    out = {'tiers': {}}
    for tier_name, tier in (('0.55', BENCHMARK_CONFIDENCE),):
        t = [d for d in dets if d['gt_group'] == 'true']
        f = [d for d in dets if d['gt_group'] == 'false']
        ns = lambda rows: sum(d['seg_group'] not in SURFACE_GROUPS for d in rows)  # noqa
        v = verdict(len(t), ns(t), len(f), ns(f))
        v['per_city'] = {c: verdict(sum(d['city'] == c for d in t),
                                    ns([d for d in t if d['city'] == c]),
                                    sum(d['city'] == c for d in f),
                                    ns([d for d in f if d['city'] == c]))
                         for c in sorted({d['city'] for d in dets})}
        tr = len(t) - ns(t)
        lo, hi = es.wilson(tr, len(t))
        v['surface_reading'] = {'true_on_walkroad': tr, 'true_n': len(t),
                                'share': tr / len(t) if t else None, 'lo': lo, 'hi': hi}
        out['tiers'][tier_name] = v
    row = next((r for r in curb if r['city'] == 'pooled' and r['group'] == 'sidewalk'
                and r['reading'] == 'measured'), None)
    if row and row['median_of_pano_medians'] not in ('', None):
        med = float(row['median_of_pano_medians'])
        out['curb_height'] = {
            'median_of_pano_medians_m': med, 'lo': float(row['median_lo']),
            'hi': float(row['median_hi']), 'n_panos': int(row['n_panos']),
            'claim_band_m': list(CLAIM_BAND_M),
            'reading': 'consistent' if CLAIM_BAND_M[0] <= med < CLAIM_BAND_M[1]
            else 'not consistent'}
    v = out['tiers']['0.55']
    print(f"(C) FP signal: {v['rule_c']} -- non-surface False {v['false_non_surface']}/"
          f"{v['false_n']} = {v['false_share']:.3f} [{v['false_lo']:.3f}, {v['false_hi']:.3f}] "
          f"vs True {v['true_non_surface']}/{v['true_n']} = {v['true_share']:.3f} "
          f"[{v['true_lo']:.3f}, {v['true_hi']:.3f}]; underpowered {v['underpowered']}")
    s = v['surface_reading']
    print(f"(C') True on WALK|ROAD: {s['true_on_walkroad']}/{s['true_n']} = {s['share']:.3f}")
    if 'curb_height' in out:
        ch = out['curb_height']
        print(f"(B) curb height (Sidewalk, measured planes): {ch['median_of_pano_medians_m']:.3f} m "
              f"[{ch['lo']:.3f}, {ch['hi']:.3f}], {ch['n_panos']} panos -> {ch['reading']}")
    (pd / 'verdict.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    return out


# --- reports -----------------------------------------------------------------------------

def _f(v, nd=3):
    if v in (None, ''):
        return '--'
    return f'{float(v):.{nd}f}'


def write_reports(args, head_rows, curb_all, trap_all, v):
    for h in head_rows:
        city = h['city']
        d = pooled_dir(args.out_root) if city == 'pooled' else city_out(args.out_root, city)
        cb = [r for r in curb_all if r['city'] == city]
        lines = [f'# Footway: segmenter vs GSV depth -- {city}', '',
                 'Generated by `scripts/footway_segmentation.py compare` (issue #47 step 2); '
                 'write-up in docs/footway-depth-study.md. Tiled arm unless marked direct.', '',
                 f"Panos: {h['n_panos']}.", '',
                 '| reading (every below-horizon pixel) | all ranges | within 25 m |',
                 '|---|---:|---:|']
        for k, name in (('p_walkroad_given_surface', 'P(WALK or ROAD | depth surface)'),
                        ('p_object_given_surface', 'P(OBJECT | depth surface)'),
                        ('p_structure_given_surface', 'P(STRUCTURE | depth surface)'),
                        ('p_other_given_surface', 'P(OTHER | depth surface)'),
                        ('p_surface_given_walkroad', 'P(depth surface | WALK or ROAD)'),
                        ('p_surface_given_object', 'P(depth surface | OBJECT)'),
                        ('p_walkroad_given_nonhorizontal', 'P(WALK or ROAD | depth wall)')):
            lines.append(f"| {name} | {_f(h['all_' + k])} | {_f(h['le25m_' + k])} |")
        lines += ['', '| curb height (offset_local, m) | reading | panos | median of pano '
                  'medians [95% CI] | pixel p10 / median / p90 |', '|---|---|---:|---|---|']
        for r in cb:
            lines.append(f"| {r['group']} | {r['reading']} | {r['n_panos']} | "
                         f"{_f(r['median_of_pano_medians'])} [{_f(r['median_lo'])}, "
                         f"{_f(r['median_hi'])}] | {_f(r['pixel_p10'])} / "
                         f"{_f(r['pixel_median'])} / {_f(r['pixel_p90'])} |")
        if city == 'pooled':
            t = v['tiers']['0.55']
            lines += ['', '## Pre-registered reading', '',
                      f"- (C) FP signal: **{t['rule_c']}**. Non-surface share: False "
                      f"{t['false_non_surface']}/{t['false_n']} = {_f(t['false_share'])} "
                      f"[{_f(t['false_lo'])}, {_f(t['false_hi'])}]; True "
                      f"{t['true_non_surface']}/{t['true_n']} = {_f(t['true_share'])} "
                      f"[{_f(t['true_lo'])}, {_f(t['true_hi'])}]; gap {_f(t['gap'])} vs "
                      f"margin {t['margin']}; underpowered: {t['underpowered']}.",
                      f"- (C') True detections on WALK or ROAD: "
                      f"{t['surface_reading']['true_on_walkroad']}/{t['surface_reading']['true_n']}"
                      f" = {_f(t['surface_reading']['share'])}."]
            if 'curb_height' in v:
                ch = v['curb_height']
                lines.append(f"- (B) Sidewalk plane above the local road: "
                             f"{_f(ch['median_of_pano_medians_m'])} m [{_f(ch['lo'])}, "
                             f"{_f(ch['hi'])}] over {ch['n_panos']} panos -> "
                             f"**{ch['reading']}** with ~0.15 m (band {ch['claim_band_m']}).")
            tr = [r for r in trap_all if r['city'] == 'pooled']
            lines += ['', '## Tiled vs direct equirect (the trap)', '',
                      '| latitude band (deg) | interior agreement | seam agreement |',
                      '|---|---:|---:|']
            for b in LAT_BANDS:
                lab = f'{b}..{b + 10}'
                ri = next(r for r in tr if r['lat_band_deg'] == lab and r['band'] == 'interior')
                rs = next(r for r in tr if r['lat_band_deg'] == lab and r['band'] == 'seam')
                lines.append(f"| {lab} | {_f(ri['agreement'])} | {_f(rs['agreement'])} |")
        (d / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# --- figures -----------------------------------------------------------------------------

def cmd_figures(args):
    import figures_footway  # noqa: F401  (kept separate: matplotlib + scipy only there)
    figures_footway.main(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['sample', 'tiles', 'segment', 'stitch', 'compare',
                                        'verdict', 'figures'])
    ap.add_argument('cities', nargs='*', default=DEFAULT_CITIES)
    ap.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--out-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--work', type=Path, default=None,
                    help='tiles / label maps (default <out-root>/_pooled/footway/work)')
    ap.add_argument('--in', dest='inp', type=Path, help='segment: input dir of *.jpg')
    ap.add_argument('--out', type=Path, help='segment: output dir for *.png label maps')
    ap.add_argument('--target', default=None,
                    help='segment: WxH of the label map (default: the input size)')
    ap.add_argument('--model', default=MODEL_ID)
    ap.add_argument('--revision', default=MODEL_REVISION)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--workers', type=int, default=6, help='tiles: worker processes')
    ap.add_argument('--limit', type=int, default=0, help='tiles: first N sampled panos')
    args = ap.parse_args()
    {'sample': cmd_sample, 'tiles': cmd_tiles, 'segment': cmd_segment,
     'stitch': cmd_stitch, 'compare': cmd_compare, 'verdict': cmd_verdict,
     'figures': cmd_figures}[args.command](args)


if __name__ == '__main__':
    main()
