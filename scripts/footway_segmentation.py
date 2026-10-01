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
nearest in angle. Pixels no tile sees (the far nadir below ~-79 deg, and the zenith above
~+43 deg) get NONE_LABEL and are counted. Two CONTROL arms feed the equirect directly and
read the label map on the same grid: the registered 2048x1024 (5.7 px/deg, half the tiles'
resolution) and, added on review of #124, 4096x2048 (the tiles' 11.4 px/deg), so projection
and resolution separate. They quantify the trap and are never in a headline.

The (A) agreement matrix is reported beside a NULL (added on review): the same pixels with
the depth index rotated 180 deg in azimuth or mirrored, plus marginals, lifts, Cohen's kappa
on a common surface / vertical / unmodelled partition, a solid-angle (cos latitude) weighted
column and a per-pano median column. Below the horizon depth puts a floor almost everywhere,
so a conditional like P(WALK or ROAD | depth surface) is close to its base rate; read it
against the null.

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
    python scripts/footway_segmentation.py examples  # fixed-rule example panels (needs work/)
    python scripts/footway_segmentation.py figures   # committed files only -> docs/figures/footway-depth/

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
        w = csv.DictWriter(f, fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else
                            '' if v is None else v) for k, v in r.items() if k in fields})


def write_text(path, text):
    """Write text with LF line endings on every platform (the tracked blobs are LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


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
            rows.append({'city': city, 'pano_id': pid, 'pano_uid': f'{city}:{pid}',
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
    uids = [r['pano_uid'] for r in rows]
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


def direct_one(task):
    """Worker: one pano resized to a direct-arm equirect of the given width (skips done)."""
    src, dst, width = Path(task[0]), Path(task[1]), int(task[2])
    if dst.exists():
        return 0
    with Image.open(src) as im:
        img = im.convert('RGB').resize((width, width // 2), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=JPEG_QUALITY)
    return 1


def direct_dir_name(width):
    """`direct` for the registered 2048 px control arm, `direct<W>` for any other width
    (the 4096 px arm added on review: the same angular resolution as the tiles)."""
    return 'direct' if width == DIRECT_W else f'direct{width}'


def cmd_tiles(args):
    from concurrent.futures import ProcessPoolExecutor
    wd = work_dir(args)
    sample = read_sample(args)
    if args.limit:
        sample = sample[:args.limit]
    if args.direct_width != DIRECT_W:
        # Only the extra direct arm: the tiles and the 2048 px arm are already rendered.
        name = direct_dir_name(args.direct_width)
        tasks = [(str(args.benchmark_root / r['city'] / 'panos' / f"{r['pano_id']}.jpg"),
                  str(wd / name / r['city'] / f"{r['pano_id']}.jpg"), args.direct_width)
                 for r in sample]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            n = sum(ex.map(direct_one, tasks))
        print(f'tiles: {n} new {args.direct_width}x{args.direct_width // 2} direct-arm '
              f'equirects -> {wd / name}')
        return
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
                'notes': args.note,
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
    unseen = geom.n_cover == 0
    below = unseen[GRID_H // 2:]
    rows = np.nonzero(below.any(axis=1))[0] + GRID_H // 2
    print(f'stitch: {int(unseen.sum())} of {GRID_W * GRID_H} grid pixels seen by no tile; '
          f'below the horizon {int(below.sum())} of {below.size}, all at latitude <= '
          f'{90 - (rows.min() + 0.5) / GRID_H * 180:.1f} deg', flush=True)
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

def payload_indices(payload):
    """The payload's per-pixel plane indices as an (height, width) int64 array (image
    frame: raw column == image column)."""
    idx = payload.indices
    arr = (np.frombuffer(bytes(idx), dtype=np.uint8) if isinstance(idx, (bytes, bytearray))
           else np.asarray(idx))
    return arr.astype(np.int64).reshape(payload.height, payload.width)


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


DIRECT_ARMS = ('direct', 'direct4096')   # the registered 2048 px control; 4096 px added on
                                         # review (the tiles' 11.4 px/deg, so the two direct
                                         # arms separate projection from resolution)
NULL_ARMS = ('yaw180', 'mirror')         # (A) null, added on review: the depth index rotated
                                         # 180 deg in azimuth, or mirrored left-right -- same
                                         # pixels, same denominators, wrong geometry
EGO_CLASSES = ('Ego Vehicle', 'Car Mount')

# Cohen's kappa needs one partition on both sides; this common 3-way one is a choice made on
# review and stated in the doc. `unmodelled` pairs depth's "no plane / overhang" with the
# segmenter's objects and sky -- so an object on a depth floor plane counts AGAINST kappa.
KAPPA_CATS = ('surface', 'vertical', 'unmodelled')
KAPPA_DEPTH = {dad.GROUND: 'surface', dad.FLOOR: 'surface', FLOOR_STANDIN: 'surface',
               dad.NON_HORIZONTAL: 'vertical', dad.HORIZONTAL_NONFLOOR: 'unmodelled',
               dad.NO_PLANE: 'unmodelled'}
KAPPA_SEG = {WALK: 'surface', ROAD: 'surface', OTHER: 'surface', STRUCTURE: 'vertical',
             OBJECT: 'unmodelled', SKY: 'unmodelled'}


def null_indices(ind, arm):
    """The payload index array as the (A) arm sees it (`correct`, or a null)."""
    if arm == 'correct':
        return ind
    if arm == 'yaw180':
        return np.roll(ind, ind.shape[1] // 2, axis=1)
    if arm == 'mirror':
        return ind[:, ::-1]
    raise ValueError(arm)


def grid_rows_cols(ph, pw):
    """(payload rows below the horizon, the grid row read for each, the grid column read
    for each payload column): one grid pixel per payload pixel, the one whose centre falls
    in it (the grid is GRID_H // ph = 2x the payload in each axis)."""
    rows = np.arange(ph // 2, ph)
    return rows, rows * (GRID_H // ph), np.arange(pw) * (GRID_W // pw)


def reference_plane(pix, ind, row, col, walked):
    """The local reference plane depth_at_detection.local_reference stopped at: the plane
    `walked` payload rows below (row, col) in the same raw column. None without one."""
    if not walked:
        return None
    return pix.payload.planes[ind[row + walked, col]]


def compare_pano(pix, run_pano, tiled, directs, fine_names, table_seg, ctx):
    """Accumulate one pano into ctx; returns (per-pano row, curb rows).

    `tiled` and every array in `directs` are (GRID_H, GRID_W) Vistas label maps."""
    payload = pix.payload
    pw, ph = payload.width, payload.height
    ind = payload_indices(payload)
    dtab = depth_class_table(pix)
    rows, grow, gcol = grid_rows_cols(ph, pw)
    lab = tiled[grow][:, gcol]
    scls = table_seg[lab]
    pose = fs.pano_pose(run_pano, fs.POSE_OFF)
    ranges = [flat_range(pose, (g + 0.5) / GRID_H) for g in grow]
    rbin = np.broadcast_to(np.array([RANGE_LABELS.index(range_label(r))
                                     for r in ranges])[:, None], scls.shape)
    shape = (len(DEPTH_CLASSES), len(SEG_GROUPS), len(RANGE_LABELS))
    nb = int(np.prod(shape))
    # solid-angle weight of a pixel row: cos(latitude) (equirect rows oversample the nadir)
    lat = (0.5 - (grow + 0.5) / GRID_H) * math.pi
    wrow = np.broadcast_to(np.cos(lat)[:, None], scls.shape)
    ego = np.isin(lab, [fine_names.index(n) for n in EGO_CLASSES])
    pano_counts = {}
    for arm in ('correct',) + NULL_ARMS:
        dc = dtab[null_indices(ind, arm)[rows]]
        key = (dc.astype(np.int64) * len(SEG_GROUPS) + scls) * len(RANGE_LABELS) + rbin
        counts = np.bincount(key.ravel(), minlength=nb).reshape(shape)
        ctx['agree' if arm == 'correct' else f'agree_{arm}'] += counts
        pano_counts[arm] = counts
        if arm == 'correct':
            dcls = dc
            ctx['agree_w'] += np.bincount(key.ravel(), weights=wrow.ravel(),
                                          minlength=nb).reshape(shape)
            ego_counts = np.bincount(key[ego], minlength=nb).reshape(shape)
            ctx['ego'] += ego_counts
    near = slice(0, len(RANGE_BINS))
    pano_row = {'n_seen_px_le25m': int(pano_counts['correct'][:, :SEG_GROUPS.index(NONE),
                                                              near].sum())}
    for k, v in headline_stats(pano_counts['correct'][:, :, near].sum(axis=2),
                               ego_counts[:, :, near].sum(axis=2)).items():
        pano_row[f'le25m_{k}'] = v
    # trap: tiled vs each direct arm, every grid pixel the tiled arm SEES (both arms' shares
    # over the same pixels), by latitude band and seam band
    tg = table_seg[tiled]
    band = np.broadcast_to(lat_band_of_rows()[:, None], tg.shape)
    x = (np.arange(GRID_W) + 0.5) / GRID_W
    seam = np.broadcast_to(((x < SEAM_HALF_WIDTH) | (x > 1 - SEAM_HALF_WIDTH))[None, :],
                           tg.shape)
    seen = tg != SEG_GROUPS.index(NONE)
    cell = (band * 2 + seam)
    nc = len(LAT_BANDS) * 2
    G = len(SEG_GROUPS)
    ctx['trap_seen'] += np.bincount(cell[seen], minlength=nc).reshape(len(LAT_BANDS), 2)
    ctx['trap_tiled'] += np.bincount((cell * G + tg)[seen], minlength=nc * G).reshape(
        len(LAT_BANDS), 2, G)
    for arm, d in directs.items():
        dg = table_seg[d]
        ctx[f'trap_{arm}'] += np.bincount((cell * G + dg)[seen], minlength=nc * G).reshape(
            len(LAT_BANDS), 2, G)
        ctx[f'trap_agree_{arm}'] += np.bincount(cell[seen & (tg == dg)], minlength=nc
                                                ).reshape(len(LAT_BANDS), 2)
    # (B) curb height: Sidewalk (headline), all WALK, and ROAD (control) on a floor plane
    in_range = np.broadcast_to(np.array([r is not None for r in ranges])[:, None],
                               dcls.shape)
    floor_any = np.isin(dcls, [DEPTH_CLASSES.index(dad.FLOOR),
                               DEPTH_CLASSES.index(FLOOR_STANDIN)])
    groups = {'sidewalk': lab == fine_names.index('Sidewalk'),
              'walk': scls == SEG_GROUPS.index(WALK),
              'road': scls == SEG_GROUPS.index(ROAD)}
    rng = random.Random(f'{SEED}:{run_pano.pano_id}')
    curb = []
    for gname, gm in groups.items():
        rr, cc = np.nonzero(gm & floor_any & in_range)
        pano_row[f'n_{gname}_px'] = int((gm & in_range).sum())
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
            walked = f.get('rows_to_ref')
            ref = reference_plane(pix, ind, r, c, walked)
            ref_lab = tiled[(r + walked) * (GRID_H // ph), c * (GRID_W // pw)]
            curb.append({'group': gname, 'standin_plane': int(cls == FLOOR_STANDIN),
                         'ref_group': SEG_GROUPS[table_seg[ref_lab]],
                         'standin_ref': int(depthlib.is_standin(ref)),
                         'offset_local_m': off,
                         'offset_level_ref_m': f.get('offset_local_level_ref_m'),
                         'range_bin': range_label(ranges[rr[i]])})
    return pano_row, curb


# `measured` is the registered reading; `with_standins` is registered as reported-beside;
# `measured_ref_road` is EXPLORATORY (added after the first scoring run): only pixels whose
# local reference -- the first different floor plane met walking the column toward the
# nadir -- is itself labelled ROAD by the segmenter at the pixel where the walk met it, i.e.
# the step really is sidewalk-to-road rather than sidewalk-to-another-sidewalk-segment.
CURB_READINGS = ('measured', 'with_standins', 'measured_ref_road')


def curb_keep(r, reading):
    measured = not (r['standin_plane'] or r['standin_ref'])
    if reading == 'with_standins':
        return True
    if reading == 'measured_ref_road':
        return measured and r['ref_group'] == ROAD
    return measured


def curb_summary(curb_by_pano):
    """Per group and reading: the median across panos of per-pano median offsets."""
    out = []
    for group in ('sidewalk', 'walk', 'road'):
        for reading in CURB_READINGS:
            meds, npx = [], 0
            for pid, rows in curb_by_pano.items():
                v = [r['offset_local_m'] for r in rows
                     if r['group'] == group and curb_keep(r, reading)]
                npx += len(v)
                if len(v) >= CURB_MIN_PIXELS:
                    meds.append(float(np.median(v)))
            med, lo, hi = dad.median_ci(meds)
            allv = [r['offset_local_m'] for rows in curb_by_pano.values() for r in rows
                    if r['group'] == group and curb_keep(r, reading)]
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


def curb_pano_medians(curb_by_pano):
    """One row per (pano, group, reading) with >= CURB_MIN_PIXELS offsets."""
    out = []
    for uid, rows in sorted(curb_by_pano.items()):
        for group in ('sidewalk', 'walk', 'road'):
            for reading in CURB_READINGS:
                v = [r['offset_local_m'] for r in rows
                     if r['group'] == group and curb_keep(r, reading)]
                if len(v) >= CURB_MIN_PIXELS:
                    out.append({'pano_uid': uid, 'group': group, 'reading': reading,
                                'n_pixels': len(v), 'median_offset_m': float(np.median(v))})
    return out


def new_ctx():
    shape = (len(DEPTH_CLASSES), len(SEG_GROUPS), len(RANGE_LABELS))
    ctx = {'agree': np.zeros(shape, dtype=np.int64),
           'agree_w': np.zeros(shape, dtype=np.float64),
           'ego': np.zeros(shape, dtype=np.int64),
           'trap_tiled': np.zeros((len(LAT_BANDS), 2, len(SEG_GROUPS)), dtype=np.int64),
           'trap_seen': np.zeros((len(LAT_BANDS), 2), dtype=np.int64)}
    for arm in NULL_ARMS:
        ctx[f'agree_{arm}'] = np.zeros(shape, dtype=np.int64)
    for arm in DIRECT_ARMS:
        ctx[f'trap_{arm}'] = np.zeros((len(LAT_BANDS), 2, len(SEG_GROUPS)), dtype=np.int64)
        ctx[f'trap_agree_{arm}'] = np.zeros((len(LAT_BANDS), 2), dtype=np.int64)
    return ctx


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


def _div(a, b):
    return float(a / b) if b else None


def cohen_kappa(m):
    """Cohen's kappa of a (DEPTH_CLASSES x SEG_GROUPS) count matrix over the common
    KAPPA_CATS partition (NONE pixels are left out)."""
    c = np.zeros((len(KAPPA_CATS), len(KAPPA_CATS)))
    for di, dcn in enumerate(DEPTH_CLASSES):
        for gi, g in enumerate(SEG_GROUPS):
            if g == NONE:
                continue
            c[KAPPA_CATS.index(KAPPA_DEPTH[dcn]), KAPPA_CATS.index(KAPPA_SEG[g])] += m[di, gi]
    n = c.sum()
    if not n:
        return None
    po = np.trace(c) / n
    pe = float((c.sum(axis=1) * c.sum(axis=0)).sum()) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def headline_stats(m, ego=None):
    """The (A) statistics of one (DEPTH_CLASSES x SEG_GROUPS) matrix (counts or weights).

    Denominators are the pixels some tile SEES (the NONE column -- the far nadir -- is
    dropped). Every conditional comes with its marginal, so the lift over the base rate is
    explicit: below the horizon nearly every pixel is a depth floor and most are road, so a
    conditional near 0.9 can be a base rate, not agreement (the review of #124)."""
    m = np.asarray(m, dtype=np.float64).copy()
    m[:, SEG_GROUPS.index(NONE)] = 0
    ego = np.zeros_like(m) if ego is None else np.asarray(ego, dtype=np.float64)
    surf = [DEPTH_CLASSES.index(c) for c in DEPTH_SURFACE]
    wall = DEPTH_CLASSES.index(dad.NON_HORIZONTAL)
    wr = [SEG_GROUPS.index(g) for g in SURFACE_GROUPS]
    ob, st = SEG_GROUPS.index(OBJECT), SEG_GROUPS.index(STRUCTURE)
    tot = m.sum()
    s, w = m[surf].sum(), m[:, wr].sum()
    o, o_s = m[:, ob].sum(), m[surf, ob].sum()
    e, e_s = ego[:, ob].sum(), ego[surf, ob].sum()
    out = {'n': tot, 'p_surface': _div(s, tot), 'p_walkroad': _div(w, tot),
           'p_object': _div(o, tot), 'p_structure': _div(m[:, st].sum(), tot),
           'p_wall': _div(m[wall].sum(), tot),
           'p_walkroad_given_surface': _div(m[surf][:, wr].sum(), s),
           'p_surface_given_walkroad': _div(m[surf][:, wr].sum(), w),
           'p_object_given_surface': _div(o_s, s),
           'p_object_given_surface_excl_ego': _div(o_s - e_s, s),
           'p_surface_given_object': _div(o_s, o),
           'p_surface_given_object_excl_ego': _div(o_s - e_s, o - e),
           'ego_share_of_surface_object': _div(e_s, o_s),
           'p_structure_given_surface': _div(m[surf, st].sum(), s),
           'p_other_given_surface': _div(m[surf, SEG_GROUPS.index(OTHER)].sum(), s),
           'p_structure_given_wall': _div(m[wall, st].sum(), m[wall].sum()),
           'p_walkroad_given_wall': _div(m[wall, wr].sum(), m[wall].sum()),
           'kappa': cohen_kappa(m)}
    for k, cond, marg in (('lift_walkroad_given_surface', 'p_walkroad_given_surface',
                           'p_walkroad'),
                          ('lift_surface_given_object', 'p_surface_given_object',
                           'p_surface'),
                          ('lift_structure_given_wall', 'p_structure_given_wall',
                           'p_structure')):
        out[k] = _div(out[cond], out[marg]) if out[cond] is not None else None
    return out


HEADLINE_KEYS = ['p_walkroad_given_surface', 'p_walkroad', 'lift_walkroad_given_surface',
                 'p_surface_given_walkroad', 'p_surface',
                 'p_object_given_surface', 'p_object_given_surface_excl_ego',
                 'p_surface_given_object', 'p_surface_given_object_excl_ego',
                 'lift_surface_given_object', 'p_structure_given_wall', 'p_structure',
                 'lift_structure_given_wall', 'p_walkroad_given_wall', 'kappa']


def headline_rows(label, ctx, pano_rows):
    """Long-form (A) rows: range x weighting x depth arm. Weightings: `pixels` (pooled
    pixels, the registered reading), `solid_angle` (cos(latitude) per pixel), and
    `per_pano_median` (le25m only, each pano one vote). Depth arms: `correct` and the
    NULL_ARMS (pixel weighting)."""
    near = slice(0, len(RANGE_BINS))
    out = []
    for rng_tag, sl in (('all', slice(None)), ('le25m', near)):
        def mat(a):
            return a[:, :, sl].sum(axis=2)
        cells = [('pixels', 'correct', ctx['agree'], ctx['ego']),
                 ('solid_angle', 'correct', ctx['agree_w'], None)] + \
                [('pixels', arm, ctx[f'agree_{arm}'], None) for arm in NULL_ARMS]
        for weighting, arm, a, ego in cells:
            st = headline_stats(mat(a), mat(ego) if ego is not None else None)
            if ego is None:
                for k in ('p_object_given_surface_excl_ego',
                          'p_surface_given_object_excl_ego', 'ego_share_of_surface_object'):
                    st[k] = None
            out.append({'city': label, 'range': rng_tag, 'weighting': weighting,
                        'depth_arm': arm, **st})
    seen_near = ctx['agree'][:, :SEG_GROUPS.index(NONE), near].sum()
    seen_05 = ctx['agree'][:, :SEG_GROUPS.index(NONE), 0].sum()
    pp = {'city': label, 'range': 'le25m', 'weighting': 'per_pano_median',
          'depth_arm': 'correct', 'n': len(pano_rows)}
    for k in HEADLINE_KEYS + ['p_other_given_surface']:
        vals = [r[f'le25m_{k}'] for r in pano_rows if r.get(f'le25m_{k}') is not None]
        pp[k] = float(np.median(vals)) if vals else None
    out.append(pp)
    for r in out:
        r['share_0_5m_of_le25m_px'] = _div(seen_05, seen_near)
    return out


def trap_rows(label, ctx):
    """Per latitude band x (interior, seam): every share over the pixels the TILED arm
    sees (NONE excluded from every denominator, both arms over the same pixels)."""
    rows = []
    for bi, b in enumerate(LAT_BANDS):
        for si, sname in enumerate(('interior', 'seam')):
            seen = int(ctx['trap_seen'][bi, si])
            row = {'city': label, 'lat_band_deg': f'{b}..{b + 10}', 'band': sname,
                   'n_seen_px': seen}
            for arm in DIRECT_ARMS:
                row[f'agreement_{arm}'] = _div(ctx[f'trap_agree_{arm}'][bi, si], seen)
            for arm in ('tiled',) + DIRECT_ARMS:
                t = ctx[f'trap_{arm}'][bi, si]
                for gi, g in enumerate(SEG_GROUPS):
                    if g != NONE:
                        row[f'{arm}_share_{g}'] = _div(t[gi], seen)
            rows.append(row)
    return rows


DET_FIELDS = ['city', 'pano_id', 'kind', 'det_index', 'x', 'y', 'confidence', 'gt_group',
              'seg_group', 'seg_class'] + \
    [f'{a}_{k}' for a in DIRECT_ARMS for k in ('group', 'class')] + \
    ['depth_class', 'plane_tilt_deg', 'offset_local_m', 'range_flat_2p6_m', 'on_camera_rig']
DET_ARMS = (('tiled', 'seg'),) + tuple((a, a) for a in DIRECT_ARMS)


ARM_KEYS = (('correct', 'agree'),) + tuple((a, f'agree_{a}') for a in NULL_ARMS)


def pano_count_row(city, pid, c):
    """One pano's within-25 m (depth class x seen segmenter group) counts for every (A)
    arm: the unit `figures` bootstraps over (pixels inside a pano are not independent)."""
    near = slice(0, len(RANGE_BINS))
    row = {'city': city, 'pano_id': pid}
    for arm, key in ARM_KEYS:
        m = c[key][:, :, near].sum(axis=2)
        for di, d in enumerate(DEPTH_CLASSES):
            for gi, g in enumerate(SEG_GROUPS):
                if g != NONE:
                    row[f'{arm}__{d}__{g}'] = int(m[di, gi])
    return row


def matrix_from_row(row, arm):
    """Inverse of pano_count_row for one arm: a (DEPTH_CLASSES x SEG_GROUPS) array, NONE
    column zero."""
    m = np.zeros((len(DEPTH_CLASSES), len(SEG_GROUPS)))
    for di, d in enumerate(DEPTH_CLASSES):
        for gi, g in enumerate(SEG_GROUPS):
            if g != NONE:
                m[di, gi] = float(row[f'{arm}__{d}__{g}'])
    return m


def trap_pano_rows(city, pid, c):
    """One pano's trap counts per latitude band x (interior, seam) with any seen pixel."""
    out = []
    for bi, b in enumerate(LAT_BANDS):
        for si, sname in enumerate(('interior', 'seam')):
            n = int(c['trap_seen'][bi, si])
            if n:
                out.append({'city': city, 'pano_id': pid, 'lat_band_deg': f'{b}..{b + 10}',
                            'band': sname, 'n_seen_px': n,
                            **{f'n_agree_{a}': int(c[f'trap_agree_{a}'][bi, si])
                               for a in DIRECT_ARMS}})
    return out


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
    pano_counts, trap_pano = [], []
    head_all, agree_all, trap_all, curb_all = [], [], [], []
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
            directs = {arm: np.asarray(Image.open(wd / f'{arm}_labels' / city / f'{pid}.png'))
                       for arm in DIRECT_ARMS}
            assert tiled.shape == (GRID_H, GRID_W) and all(
                d.shape == tiled.shape for d in directs.values()), (city, pid)
            c1 = new_ctx()
            prow, curb = compare_pano(pix, run_pano, tiled, directs, fine_names, table_seg,
                                      c1)
            add_ctx(ctx, c1)
            pano_rows.append({'city': city, 'pano_id': pid, **prow})
            pano_counts.append(pano_count_row(city, pid, c1))
            trap_pano.extend(trap_pano_rows(city, pid, c1))
            curb_by_pano[f'{city}:{pid}'] = curb
            pose = fs.pano_pose(run_pano, fs.POSE_OFF)
            for it in detection_items(city, pid, entry, run_pano, ops):
                x, y = it['x'], it['y']
                col = min(GRID_W - 1, int(x * GRID_W))
                row = min(GRID_H - 1, int(y * GRID_H))
                cls, f = depth_class_at(pix, x, y)
                g = geo.detection_ground_point(pose, x, y, max_range_m=math.inf,
                                               apply_pose=False)
                lt = int(tiled[row, col])
                it.update({'seg_group': SEG_GROUPS[table_seg[lt]],
                           'seg_class': id2label.get(lt, 'NONE')})
                for arm, d in directs.items():
                    ld = int(d[row, col])
                    it[f'{arm}_group'] = SEG_GROUPS[table_seg[ld]]
                    it[f'{arm}_class'] = id2label.get(ld, 'NONE')
                it.update({'depth_class': cls, 'plane_tilt_deg': f.get('plane_tilt_deg'),
                           'offset_local_m': f.get('offset_local_m'),
                           'range_flat_2p6_m': g.range_m if g else None,
                           'on_camera_rig': int(dad_on_rig(y))})
                dets.append(it)
        missing = want - {r['pano_id'] for r in pano_rows}
        if missing:
            raise SystemExit(f'{city}: {len(missing)} sampled panos no longer pass the GT '
                             f'join (drift?): {sorted(missing)[:5]}')
        od = city_out(args.out_root, city)
        hr = headline_rows(city, ctx, pano_rows)
        head_all += hr
        agree_all += agreement_rows(city, ctx['agree'])
        trap_all += trap_rows(city, ctx)
        cs = curb_summary(curb_by_pano)
        curb_all += [{'city': city, **r} for r in cs]
        write_csv(od / 'detections.csv', dets, DET_FIELDS)
        write_csv(od / 'panos.csv', pano_rows)
        write_csv(od / 'agreement.csv', agreement_rows(city, ctx['agree']))
        write_csv(od / 'curb_height.csv', [{'city': city, **r} for r in cs])
        add_ctx(pooled, ctx)
        pooled_det += dets
        pooled_curb.update(curb_by_pano)
        pooled_pano += pano_rows
        h = next(r for r in hr if r['range'] == 'le25m' and r['weighting'] == 'pixels'
                 and r['depth_arm'] == 'correct')
        print(f'{city}: {len(pano_rows)} panos, {len(dets)} detection/mark rows; within 25 m '
              f'P(WALK|ROAD | depth surface) {h["p_walkroad_given_surface"]:.3f} vs marginal '
              f'{h["p_walkroad"]:.3f}; kappa {h["kappa"]:.3f}', flush=True)
    pd = pooled_dir(args.out_root)
    head_all += headline_rows('pooled', pooled, pooled_pano)
    agree_all += agreement_rows('pooled', pooled['agree'])
    trap_all += trap_rows('pooled', pooled)
    curb_all += [{'city': 'pooled', **r} for r in curb_summary(pooled_curb)]
    write_csv(pd / 'agreement.csv', agree_all)
    write_csv(pd / 'trap.csv', trap_all)
    write_csv(pd / 'curb_height.csv', curb_all)
    write_csv(pd / 'curb_pano_medians.csv', curb_pano_medians(pooled_curb))
    write_csv(pd / 'headline.csv', head_all)
    write_csv(pd / 'panos.csv', pooled_pano)
    write_csv(pd / 'pano_counts_le25m.csv', pano_counts)
    write_csv(pd / 'trap_pano.csv', trap_pano)
    write_csv(pd / 'detections.csv', pooled_det, DET_FIELDS)
    write_csv(pd / 'detection_classes.csv', detection_class_rows(pooled_det))
    write_masks_manifest(args, wd, sample)
    write_inputs_manifest(args, pd)
    v = cmd_verdict(args)
    write_reports(args, head_all, curb_all, trap_all, v)


def dad_on_rig(y):
    from detectors import on_camera_rig
    return on_camera_rig(y)


def detection_class_rows(dets):
    """Segmenter-group shares (tiled and both direct arms) per city x GT group, with the
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
            for arm, col in DET_ARMS:
                c = Counter(d[f'{col}_group'] for d in gi)
                n = len(gi)
                ns = n - c[WALK] - c[ROAD]
                lo, hi = es.wilson(ns, n)
                fine = Counter(d[f'{col}_class'] for d in gi)
                rows.append({'city': city, 'group': g, 'arm': arm, 'n': n,
                             'n_non_surface': ns,
                             'share_non_surface': ns / n if n else None,
                             'non_surface_lo': lo, 'non_surface_hi': hi,
                             **{f'share_{k}': (c[k] / n if n else None) for k in SEG_GROUPS},
                             'share_curb_cut': fine['Curb Cut'] / n if n else None,
                             'top_classes': ';'.join(f'{k}={v}'
                                                     for k, v in fine.most_common(6))})
    return rows


def write_masks_manifest(args, wd, sample):
    def load(name):
        return json.loads((wd / name / 'segment_manifest.json').read_text(encoding='utf-8'))
    seg = load('tile_labels')
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
           'tile_label_maps_notes': seg.get('notes', []),
           'stitched_label_maps_sha256': stitched}
    for arm in DIRECT_ARMS:
        d = load(f'{arm}_labels')
        man[f'{arm}_label_maps_sha256'] = d['outputs_sha256']
        man[f'{arm}_notes'] = d.get('notes', [])
    write_text(pooled_dir(args.out_root) / 'masks_manifest.json',
               json.dumps(man, indent=1, sort_keys=True))


def write_inputs_manifest(args, pd):
    """runs/_pooled/footway/inputs.json: the sha256 of every input `compare` reads (per city
    results.jsonl and the RampNet bundle's verdicts.json + records.jsonl; depth/index.csv is
    recorded but NOT read -- payloads are hashed per pano in sample.csv), the RampNet commit,
    the masks manifest, and the analysis environment."""
    import platform
    import subprocess
    import scipy
    out = {'cities': {}, 'environment': {
        'python': platform.python_version(), 'numpy': np.__version__,
        'pillow': Image.__version__, 'scipy': scipy.__version__}}
    for city in args.cities:
        rd, bd = args.run_root / city, args.benchmark_root / city
        out['cities'][city] = {
            'results.jsonl': sha256_file(rd / 'results.jsonl'),
            'depth/index.csv (not read)': sha256_file(rd / 'depth' / 'index.csv')
            if (rd / 'depth' / 'index.csv').exists() else None,
            'rampnet verdicts.json': sha256_file(bd / 'verdicts.json'),
            'rampnet records.jsonl': sha256_file(bd / 'records.jsonl')}
    try:
        out['rampnet_commit'] = subprocess.run(
            ['git', '-C', str(args.benchmark_root), 'rev-parse', 'HEAD'], capture_output=True,
            text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out['rampnet_commit'] = None
    out['sample.csv'] = sha256_file(pd / 'sample.csv')
    out['masks_manifest.json'] = sha256_file(pd / 'masks_manifest.json')
    write_text(pd / 'inputs.json', json.dumps(out, indent=1, sort_keys=True))


# --- verdict -----------------------------------------------------------------------------

def verdict(true_n, true_ns, false_n, false_ns):
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
    if not (pd / 'detections.csv').exists():
        raise SystemExit(f'{pd / "detections.csv"} not found; run `compare` first')
    print(f'verdict: reading {pd}')
    dets = read_csv(pd / 'detections.csv')
    curb = read_csv(pd / 'curb_height.csv')
    t = [d for d in dets if d['gt_group'] == 'true']
    f = [d for d in dets if d['gt_group'] == 'false']

    def ns(rows):
        return sum(d['seg_group'] not in SURFACE_GROUPS for d in rows)
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
    # EXPLORATORY / POST HOC, outside the registered reading (added after the first scoring
    # run): the fine Vistas class `Curb Cut` at the pixel, per arm. Descriptive only.
    expl = {}
    for arm, col in DET_ARMS:
        for g in ('true', 'false', 'missed', 'unsure_missed', 'band_0p30'):
            rows = [d for d in dets if d['gt_group'] == g]
            k = sum(d[f'{col}_class'] == 'Curb Cut' for d in rows)
            lo, hi = es.wilson(k, len(rows))
            expl[f'{arm}_{g}'] = {'n': len(rows), 'curb_cut': k,
                                  'share': k / len(rows) if rows else None,
                                  'lo': lo, 'hi': hi}
    v['exploratory_curb_cut'] = expl
    out = {'tiers': {'0.55': v}}
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
    print(f"(C) FP signal: {v['rule_c']} -- non-surface False {v['false_non_surface']}/"
          f"{v['false_n']} = {v['false_share']:.3f} [{v['false_lo']:.3f}, {v['false_hi']:.3f}] "
          f"vs True {v['true_non_surface']}/{v['true_n']} = {v['true_share']:.3f} "
          f"[{v['true_lo']:.3f}, {v['true_hi']:.3f}]; underpowered {v['underpowered']}")
    s = v['surface_reading']
    print(f"(C') True on WALK|ROAD: {s['true_on_walkroad']}/{s['true_n']} = {s['share']:.3f}")
    if 'curb_height' in out:
        ch = out['curb_height']
        print(f"(B) curb height (Sidewalk, measured planes): "
              f"{ch['median_of_pano_medians_m']:.3f} m [{ch['lo']:.3f}, {ch['hi']:.3f}], "
              f"{ch['n_panos']} panos -> {ch['reading']}")
    write_text(pd / 'verdict.json', json.dumps(out, indent=1))
    return out


# --- reports -----------------------------------------------------------------------------

def _f(v, nd=3):
    if v in (None, ''):
        return '--'
    return f'{float(v):.{nd}f}'


REPORT_STATS = [
    ('p_walkroad_given_surface', 'P(WALK or ROAD | depth surface)'),
    ('p_walkroad', '  marginal P(WALK or ROAD)'),
    ('lift_walkroad_given_surface', '  lift'),
    ('p_surface_given_walkroad', 'P(depth surface | WALK or ROAD)'),
    ('p_surface', '  marginal P(depth surface)'),
    ('p_object_given_surface', 'P(OBJECT | depth surface)'),
    ('p_object_given_surface_excl_ego', '  ... excluding Ego Vehicle / Car Mount'),
    ('p_surface_given_object', 'P(depth surface | OBJECT)'),
    ('p_surface_given_object_excl_ego', '  ... excluding Ego Vehicle / Car Mount'),
    ('lift_surface_given_object', '  lift over P(depth surface)'),
    ('p_structure_given_wall', 'P(STRUCTURE | depth wall)'),
    ('p_structure', '  marginal P(STRUCTURE)'),
    ('lift_structure_given_wall', '  lift'),
    ('p_walkroad_given_wall', 'P(WALK or ROAD | depth wall)'),
    ('kappa', "Cohen's kappa (surface / vertical / unmodelled)"),
]


def write_reports(args, head_all, curb_all, trap_all, v):
    for city in sorted({h['city'] for h in head_all}, key=lambda c: c == 'pooled'):
        d = pooled_dir(args.out_root) if city == 'pooled' else city_out(args.out_root, city)
        hs = [h for h in head_all if h['city'] == city]

        def cell(rng, weighting, arm):
            return next(h for h in hs if h['range'] == rng and h['weighting'] == weighting
                        and h['depth_arm'] == arm)
        cols = [('all ranges, pixels', cell('all', 'pixels', 'correct')),
                ('25 m, pixels', cell('le25m', 'pixels', 'correct')),
                ('25 m, solid angle', cell('le25m', 'solid_angle', 'correct')),
                ('25 m, per-pano median', cell('le25m', 'per_pano_median', 'correct')),
                ('25 m, NULL yaw 180', cell('le25m', 'pixels', 'yaw180')),
                ('25 m, NULL mirror', cell('le25m', 'pixels', 'mirror'))]
        cb = [r for r in curb_all if r['city'] == city]
        npano = cell('le25m', 'per_pano_median', 'correct')['n']
        lines = [f'# Footway: segmenter vs GSV depth -- {city}', '',
                 'Generated by `scripts/footway_segmentation.py compare` (issue #47 step 2); '
                 'write-up in docs/footway-depth-study.md. Tiled arm unless marked direct.', '',
                 f'Panos: {npano}. Below-horizon pixels some tile sees; NULL = the depth index '
                 'rotated 180 deg in azimuth / mirrored (same pixels). Of the within-25 m '
                 f"pixels, {_f(hs[0]['share_0_5m_of_le25m_px'])} are in the 0-5 m bin.", '',
                 '| (A) statistic | ' + ' | '.join(c for c, _ in cols) + ' |',
                 '|---|' + '---:|' * len(cols)]
        for k, name in REPORT_STATS:
            lines.append(f'| {name} | ' + ' | '.join(_f(h.get(k)) for _, h in cols) + ' |')
        lines += ['', '| curb height (offset_local, m) | reading | panos | median of pano '
                  'medians [95% CI] | pixel p10 / median / p90 | share in [0.05, 0.30) |',
                  '|---|---|---:|---|---|---:|']
        for r in cb:
            lines.append(f"| {r['group']} | {r['reading']} | {r['n_panos']} | "
                         f"{_f(r['median_of_pano_medians'])} [{_f(r['median_lo'])}, "
                         f"{_f(r['median_hi'])}] | {_f(r['pixel_p10'])} / "
                         f"{_f(r['pixel_median'])} / {_f(r['pixel_p90'])} | "
                         f"{_f(r['share_in_claim_band'])} |")
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
                      f"{t['surface_reading']['true_on_walkroad']}/"
                      f"{t['surface_reading']['true_n']} = {_f(t['surface_reading']['share'])}."]
            if 'curb_height' in v:
                ch = v['curb_height']
                lines.append(f"- (B) Sidewalk plane above the local road: "
                             f"{_f(ch['median_of_pano_medians_m'])} m [{_f(ch['lo'])}, "
                             f"{_f(ch['hi'])}] over {ch['n_panos']} panos -> "
                             f"**{ch['reading']}** with ~0.15 m (band "
                             f"[{CLAIM_BAND_M[0]:.2f}, {CLAIM_BAND_M[1]:.2f})).")
            lines.append('- (A) was registered as descriptive; the null columns above were '
                         'added on review and change its interpretation, not a rule.')
            ex = t['exploratory_curb_cut']
            lines += ['', '## Post hoc, descriptive (not part of any registered reading)', '',
                      '`Curb Cut` (fine Vistas class) at the detection / mark pixel, per arm:',
                      '', '| group | n | tiled [95% CI] | direct 2048 [95% CI] | '
                      'direct 4096 [95% CI] |', '|---|---:|---|---|---|']
            for g in ('true', 'false', 'missed', 'unsure_missed', 'band_0p30'):
                a = [ex[f'{arm}_{g}'] for arm, _ in DET_ARMS]
                lines.append(f"| {g} | {a[0]['n']} | " + ' | '.join(
                    f"{_f(e['share'])} [{_f(e['lo'])}, {_f(e['hi'])}]" for e in a) + ' |')
            tr = [r for r in trap_all if r['city'] == 'pooled']
            lines += ['', '## Tiled vs direct equirect (the trap; registered: by latitude '
                      'band and seam)', '',
                      'Agreement on the collapsed group over pixels the tiled arm sees. The '
                      'seam band is straight behind the car (mostly road), so seam vs interior '
                      'compares content as well as the wrap.', '',
                      '| latitude band (deg) | interior, 2048 | interior, 4096 | seam, 2048 | '
                      'seam, 4096 |', '|---|---:|---:|---:|---:|']
            for b in LAT_BANDS:
                lab = f'{b}..{b + 10}'
                ri = next(r for r in tr if r['lat_band_deg'] == lab and r['band'] == 'interior')
                rs = next(r for r in tr if r['lat_band_deg'] == lab and r['band'] == 'seam')
                lines.append(f"| {lab} | {_f(ri['agreement_direct'])} | "
                             f"{_f(ri['agreement_direct4096'])} | {_f(rs['agreement_direct'])} "
                             f"| {_f(rs['agreement_direct4096'])} |")
        write_text(d / 'report.md', '\n'.join(lines) + '\n')


# --- figures -----------------------------------------------------------------------------

# --- where each number lives (checked by `figures`) -------------------------------------
#
# Every number quoted in docs/footway-depth-study.md, the PR body, the CLAUDE.md block and
# the #47 comments, with the committed file, row selector and column it comes from. `figures`
# re-reads each one, rounds it the way it is quoted, and writes data/numbers.csv with a
# `status` column; a mismatch is printed. Selectors: headline.csv rows are keyed by
# (city, range, weighting, depth_arm); curb_height.csv by (city, group, reading);
# detection_classes.csv by (city, group, arm); trap.csv by (city, lat_band_deg, band);
# verdict.json by a dotted path; sample_counts.csv by city.

def _H(city, rng, w, arm):
    return ('headline.csv', {'city': city, 'range': rng, 'weighting': w, 'depth_arm': arm})


_POOL25 = 'pooled', 'le25m'
QUOTED = [
    # (A) pooled, within 25 m
    ('0.897', _H(*_POOL25, 'pixels', 'correct'), 'p_walkroad_given_surface', 3),
    ('0.894', _H(*_POOL25, 'pixels', 'yaw180'), 'p_walkroad_given_surface', 3),
    ('0.895', _H(*_POOL25, 'pixels', 'mirror'), 'p_walkroad_given_surface', 3),
    ('0.884', _H(*_POOL25, 'pixels', 'correct'), 'p_walkroad', 3),
    ('0.983', _H(*_POOL25, 'pixels', 'correct'), 'p_surface', 3),
    ('1.01', _H(*_POOL25, 'pixels', 'correct'), 'lift_walkroad_given_surface', 2),
    ('0.997', _H(*_POOL25, 'pixels', 'correct'), 'p_surface_given_walkroad', 3),
    ('0.863', _H(*_POOL25, 'solid_angle', 'correct'), 'p_walkroad_given_surface', 3),
    ('0.845', _H(*_POOL25, 'solid_angle', 'correct'), 'p_walkroad', 3),
    ('0.917', _H(*_POOL25, 'per_pano_median', 'correct'), 'p_walkroad_given_surface', 3),
    ('0.904', _H(*_POOL25, 'per_pano_median', 'correct'), 'p_walkroad', 3),
    ('0.71', _H(*_POOL25, 'pixels', 'correct'), 'share_0_5m_of_le25m_px', 2),
    ('0.661', _H(*_POOL25, 'pixels', 'correct'), 'p_structure_given_wall', 3),
    ('0.415', _H(*_POOL25, 'pixels', 'yaw180'), 'p_structure_given_wall', 3),
    ('0.470', _H(*_POOL25, 'pixels', 'mirror'), 'p_structure_given_wall', 3),
    ('0.023', _H(*_POOL25, 'pixels', 'correct'), 'p_structure', 3),
    ('29', _H(*_POOL25, 'pixels', 'correct'), 'lift_structure_given_wall', 0),
    ('0.250', _H(*_POOL25, 'pixels', 'correct'), 'kappa', 3),
    ('0.169', _H(*_POOL25, 'pixels', 'yaw180'), 'kappa', 3),
    ('0.188', _H(*_POOL25, 'pixels', 'mirror'), 'kappa', 3),
    ('0.097', _H(*_POOL25, 'per_pano_median', 'correct'), 'kappa', 3),
    ('0.430', _H('pooled', 'all', 'pixels', 'correct'), 'kappa', 3),
    ('0.341', _H('pooled', 'all', 'pixels', 'yaw180'), 'kappa', 3),
    ('0.364', _H('pooled', 'all', 'pixels', 'mirror'), 'kappa', 3),
    ('0.947', _H(*_POOL25, 'pixels', 'correct'), 'p_surface_given_object', 3),
    ('0.935', _H(*_POOL25, 'pixels', 'yaw180'), 'p_surface_given_object', 3),
    ('0.936', _H(*_POOL25, 'pixels', 'mirror'), 'p_surface_given_object', 3),
    ('0.944', _H(*_POOL25, 'pixels', 'correct'), 'p_surface_given_object_excl_ego', 3),
    ('0.96', _H(*_POOL25, 'pixels', 'correct'), 'lift_surface_given_object', 2),
    ('0.062', _H(*_POOL25, 'pixels', 'correct'), 'ego_share_of_surface_object', 3),
    ('0.055', _H(*_POOL25, 'pixels', 'correct'), 'p_object_given_surface', 3),
    ('0.051', _H(*_POOL25, 'pixels', 'correct'), 'p_object_given_surface_excl_ego', 3),
    ('0.071', _H(*_POOL25, 'solid_angle', 'correct'), 'p_object_given_surface', 3),
    ('0.153', _H(*_POOL25, 'pixels', 'correct'), 'p_walkroad_given_wall', 3),
    ('0.336', _H(*_POOL25, 'pixels', 'yaw180'), 'p_walkroad_given_wall', 3),
    ('0.280', _H(*_POOL25, 'pixels', 'mirror'), 'p_walkroad_given_wall', 3),
    ('0.871', _H('pooled', 'all', 'pixels', 'correct'), 'p_walkroad_given_surface', 3),
    ('0.826', _H('pooled', 'all', 'pixels', 'correct'), 'p_walkroad', 3),
    ('0.070', _H('pooled', 'all', 'pixels', 'correct'), 'p_object_given_surface', 3),
    ('0.804', _H('pooled', 'all', 'pixels', 'correct'), 'p_surface_given_object', 3),
    ('0.674', _H('pooled', 'all', 'pixels', 'correct'), 'p_structure_given_wall', 3),
    # (B)
    ('0.022', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk', 'reading': 'measured'}),
     'median_of_pano_medians', 3),
    ('0.014', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk', 'reading': 'measured'}),
     'median_lo', 3),
    ('0.030', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk', 'reading': 'measured'}),
     'median_hi', 3),
    ('399', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk', 'reading': 'measured'}),
     'n_panos', 0),
    ('0.339', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk', 'reading': 'measured'}),
     'share_in_claim_band', 3),
    ('0.033', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk',
                                   'reading': 'with_standins'}), 'median_of_pano_medians', 3),
    ('0.042', ('curb_height.csv', {'city': 'pooled', 'group': 'sidewalk',
                                   'reading': 'measured_ref_road'}), 'median_of_pano_medians', 3),
    ('0.021', ('curb_height.csv', {'city': 'pooled', 'group': 'walk', 'reading': 'measured'}),
     'median_of_pano_medians', 3),
    ('0.008', ('curb_height.csv', {'city': 'pooled', 'group': 'road', 'reading': 'measured'}),
     'median_of_pano_medians', 3),
    ('0.234', ('curb_height.csv', {'city': 'pooled', 'group': 'road', 'reading': 'measured'}),
     'share_in_claim_band', 3),
    ('0.031', ('curb_height.csv', {'city': 'bend', 'group': 'sidewalk', 'reading': 'measured'}),
     'median_of_pano_medians', 3),
    ('0.024', ('curb_height.csv', {'city': 'paterson', 'group': 'sidewalk',
                                   'reading': 'measured'}), 'median_of_pano_medians', 3),
    ('0.020', ('curb_height.csv', {'city': 'gainesville', 'group': 'sidewalk',
                                   'reading': 'measured'}), 'median_of_pano_medians', 3),
    ('0.009', ('curb_height.csv', {'city': 'sao_paulo', 'group': 'sidewalk',
                                   'reading': 'measured'}), 'median_of_pano_medians', 3),
    # (C), (C')
    ('2', ('verdict.json', {}), 'tiers.0.55.false_non_surface', 0),
    ('42', ('verdict.json', {}), 'tiers.0.55.false_n', 0),
    ('0.048', ('verdict.json', {}), 'tiers.0.55.false_share', 3),
    ('0.013', ('verdict.json', {}), 'tiers.0.55.false_lo', 3),
    ('0.158', ('verdict.json', {}), 'tiers.0.55.false_hi', 3),
    ('43', ('verdict.json', {}), 'tiers.0.55.true_non_surface', 0),
    ('770', ('verdict.json', {}), 'tiers.0.55.true_n', 0),
    ('0.056', ('verdict.json', {}), 'tiers.0.55.true_share', 3),
    ('0.042', ('verdict.json', {}), 'tiers.0.55.true_lo', 3),
    ('0.074', ('verdict.json', {}), 'tiers.0.55.true_hi', 3),
    ('0.944', ('verdict.json', {}), 'tiers.0.55.surface_reading.share', 3),
    ('727', ('verdict.json', {}), 'tiers.0.55.surface_reading.true_on_walkroad', 0),
    ('0.030', ('detection_classes.csv', {'city': 'pooled', 'group': 'missed', 'arm': 'tiled'}),
     'share_non_surface', 3),
    # Curb Cut at the pixel (post hoc)
    ('0.271', ('detection_classes.csv', {'city': 'pooled', 'group': 'true', 'arm': 'tiled'}),
     'share_curb_cut', 3),
    ('0.071', ('detection_classes.csv', {'city': 'pooled', 'group': 'false', 'arm': 'tiled'}),
     'share_curb_cut', 3),
    ('0.254', ('detection_classes.csv', {'city': 'pooled', 'group': 'missed', 'arm': 'tiled'}),
     'share_curb_cut', 3),
    ('0.179', ('detection_classes.csv', {'city': 'pooled', 'group': 'true', 'arm': 'direct'}),
     'share_curb_cut', 3),
    ('0.130', ('detection_classes.csv', {'city': 'pooled', 'group': 'missed', 'arm': 'direct'}),
     'share_curb_cut', 3),
    ('0.301', ('detection_classes.csv', {'city': 'pooled', 'group': 'true', 'arm': 'direct4096'}),
     'share_curb_cut', 3),
    ('0.269', ('detection_classes.csv', {'city': 'pooled', 'group': 'missed',
                                         'arm': 'direct4096'}), 'share_curb_cut', 3),
    ('0.119', ('detection_classes.csv', {'city': 'pooled', 'group': 'false',
                                         'arm': 'direct4096'}), 'share_curb_cut', 3),
    # the trap
    ('0.957', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-80..-70', 'band': 'interior'}),
     'agreement_direct', 3),
    ('0.856', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-80..-70', 'band': 'interior'}),
     'agreement_direct4096', 3),
    ('0.948', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-80..-70', 'band': 'seam'}),
     'agreement_direct', 3),
    ('0.819', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-80..-70', 'band': 'seam'}),
     'agreement_direct4096', 3),
    ('0.907', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-10..0', 'band': 'seam'}),
     'agreement_direct', 3),
    ('0.919', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-10..0', 'band': 'interior'}),
     'agreement_direct', 3),
    ('0.959', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-10..0', 'band': 'interior'}),
     'agreement_direct4096', 3),
    ('0.12', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-80..-70', 'band': 'interior'}),
     'direct4096_share_SKY', 2),
    ('0.15', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-80..-70', 'band': 'seam'}),
     'direct4096_share_SKY', 2),
    ('0.06', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-50..-40', 'band': 'interior'}),
     'direct4096_share_SKY', 2),
    ('0.08', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-50..-40', 'band': 'seam'}),
     'direct4096_share_SKY', 2),
    ('0.95', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-20..-10', 'band': 'seam'}),
     'tiled_share_ROAD', 2),
    ('0.44', ('trap.csv', {'city': 'pooled', 'lat_band_deg': '-20..-10', 'band': 'interior'}),
     'tiled_share_ROAD', 2),
    # sample
    ('90', ('sample_counts.csv', {'city': 'bend'}), 'n_sample', 0),
    ('113', ('sample_counts.csv', {'city': 'paterson'}), 'n_sample', 0),
    ('112', ('sample_counts.csv', {'city': 'gainesville'}), 'n_sample', 0),
    ('101', ('sample_counts.csv', {'city': 'sao_paulo'}), 'n_sample', 0),
]
# the per-city (A) table, within 25 m: (correct, yaw180, mirror) per statistic
_CITY_TABLE = {
    'bend': {'p_walkroad_given_surface': ('0.906', '0.905', '0.905'), 'p_walkroad': ('0.902',),
             'p_surface_given_object': ('0.957', '0.966', '0.963'),
             'p_structure_given_wall': ('0.488', '0.209', '0.222'),
             'kappa': ('0.104', '0.053', '0.057')},
    'paterson': {'p_walkroad_given_surface': ('0.905', '0.902', '0.903'),
                 'p_walkroad': ('0.892',),
                 'p_surface_given_object': ('0.955', '0.941', '0.946'),
                 'p_structure_given_wall': ('0.674', '0.403', '0.486'),
                 'kappa': ('0.225', '0.146', '0.170')},
    'gainesville': {'p_walkroad_given_surface': ('0.882', '0.881', '0.881'),
                    'p_walkroad': ('0.878',),
                    'p_surface_given_object': ('0.966', '0.960', '0.955'),
                    'p_structure_given_wall': ('0.553', '0.174', '0.216'),
                    'kappa': ('0.117', '0.055', '0.065')},
    'sao_paulo': {'p_walkroad_given_surface': ('0.896', '0.889', '0.891'),
                  'p_walkroad': ('0.866',),
                  'p_surface_given_object': ('0.929', '0.907', '0.907'),
                  'p_structure_given_wall': ('0.684', '0.466', '0.512'),
                  'kappa': ('0.341', '0.242', '0.266')}}
for _c, _stats in _CITY_TABLE.items():
    for _k, _vals in _stats.items():
        for _arm, _v in zip(('correct', 'yaw180', 'mirror'), _vals):
            QUOTED.append((_v, _H(_c, 'le25m', 'pixels', _arm), _k, 3))


def derived_numbers(pd):
    """Numbers the write-up quotes that are not a single CSV cell, each recomputed here from
    committed files with its formula written beside it (data/derived_numbers.csv)."""
    dets = read_csv(pd / 'detections.csv')
    agree = read_csv(pd / 'agreement.csv')
    panos = read_csv(pd / 'panos.csv')
    out = []

    def add(name, value, formula):
        out.append({'name': name, 'value': value, 'formula': formula})
    for g in ('true', 'false'):
        gi = [d for d in dets if d['gt_group'] == g]
        c = Counter(d['depth_class'] for d in gi)
        for k in DEPTH_CLASSES:
            add(f'det_{g}_depth_{k}', c[k], f"detections.csv: gt_group={g}, count depth_class={k}")
    f = [d for d in dets if d['gt_group'] == 'false']
    c = Counter(d['seg_class'] for d in f)
    for k in ('Sidewalk', 'Curb', 'Curb Cut', 'Road', 'Catch Basin', 'Manhole'):
        add(f'false_seg_{k}', c[k], f"detections.csv: gt_group=false, count seg_class={k}")
    for grp in (WALK, ROAD):
        add(f'false_group_{grp}', sum(d['seg_group'] == grp for d in f),
            f'detections.csv: gt_group=false, count seg_group={grp}')
    rows = {r['depth_class']: r for r in agree if r['city'] == 'pooled' and r['range_bin'] == 'all'}

    def seen(r):
        return float(r['n_px']) - float(r['n_NONE'])
    for dc in (dad.GROUND, dad.FLOOR, FLOOR_STANDIN):
        add(f'walk_px_millions_{dc}', float(rows[dc]['n_WALK']) / 1e6,
            f'agreement.csv: city=pooled, range_bin=all, depth_class={dc}, n_WALK / 1e6')
        add(f'walk_share_of_seen_{dc}', float(rows[dc]['n_WALK']) / seen(rows[dc]),
            f'agreement.csv: same row, n_WALK / (n_px - n_NONE)')
    add('walk_px_millions_secondary', (float(rows[dad.FLOOR]['n_WALK'])
                                       + float(rows[FLOOR_STANDIN]['n_WALK'])) / 1e6,
        'agreement.csv: floor + floor_standin n_WALK / 1e6')
    for rl in RANGE_LABELS:
        rr = [r for r in agree if r['city'] == 'pooled' and r['range_bin'] == rl
              and r['depth_class'] in DEPTH_SURFACE]
        tot = sum(seen(r) for r in rr)
        add(f'object_share_of_surface_{rl}', sum(float(r['n_OBJECT']) for r in rr) / tot,
            f'agreement.csv: city=pooled, range_bin={rl}, surface rows, n_OBJECT / seen')
    sw = [(int(r['n_sidewalk_ground_px']), int(r['n_sidewalk_px'])) for r in panos
          if int(r['n_sidewalk_px']) > 0]
    add('sidewalk_on_dominant_pixel_pooled', sum(a for a, _ in sw) / sum(b for _, b in sw),
        'panos.csv (pooled): sum n_sidewalk_ground_px / sum n_sidewalk_px')
    add('sidewalk_on_dominant_pano_median', float(np.median([a / b for a, b in sw])),
        'panos.csv (pooled): median over panos with n_sidewalk_px > 0 of the ratio')
    add('sidewalk_on_dominant_n_panos', len(sw), 'panos.csv: panos with n_sidewalk_px > 0')
    lats = np.array([(0.5 - float(d['y'])) * 180 for d in dets
                     if d['gt_group'] in ('true', 'false', 'missed')])
    add('det_lat_p01_deg', float(np.percentile(lats, 1)),
        'detections.csv: True + False + missed, (0.5 - y) * 180, 1st percentile')
    add('det_lat_p99_deg', float(np.percentile(lats, 99)), '... 99th percentile')
    add('det_lat_min_deg', float(lats.min()), '... minimum')
    add('det_lat_max_deg', float(lats.max()), '... maximum')
    for city in DEFAULT_CITIES:
        t = [d for d in dets if d['gt_group'] == 'true' and d['city'] == city]
        add(f'true_on_walkroad_{city}', sum(d['seg_group'] in SURFACE_GROUPS for d in t) / len(t),
            f'detections.csv: city={city}, gt_group=true, share seg_group in WALK/ROAD')
    x = [d for d in dets if d['gt_group'] in ('true', 'false', 'missed')]
    for arm in DIRECT_ARMS:
        add(f'det_group_agree_{arm}', sum(d['seg_group'] == d[f'{arm}_group'] for d in x) / len(x),
            f'detections.csv: True + False + missed, share seg_group == {arm}_group')
        add(f'det_class_agree_{arm}', sum(d['seg_class'] == d[f'{arm}_class'] for d in x) / len(x),
            f'detections.csv: True + False + missed, share seg_class == {arm}_class')
    add('det_n_true_false_missed', len(x), 'detections.csv: rows with gt_group in true/false/missed')
    b = [d for d in dets if d['gt_group'] == 'band_0p30']
    add('band_non_surface_share', sum(d['seg_group'] not in SURFACE_GROUPS for d in b) / len(b),
        'detections.csv: gt_group=band_0p30, share seg_group not in WALK/ROAD')
    return out

def _D(name):
    return ('derived_numbers.csv', {'name': name})


QUOTED += [
    ('500', _D('det_true_depth_floor'), 'value', 0),
    ('143', _D('det_true_depth_ground'), 'value', 0),
    ('122', _D('det_true_depth_floor_standin'), 'value', 0),
    ('4', _D('det_true_depth_non_horizontal'), 'value', 0),
    ('1', _D('det_true_depth_horizontal_nonfloor'), 'value', 0),
    ('26', _D('det_false_depth_floor'), 'value', 0),
    ('10', _D('det_false_depth_floor_standin'), 'value', 0),
    ('5', _D('det_false_depth_ground'), 'value', 0),
    ('1', _D('det_false_depth_non_horizontal'), 'value', 0),
    ('25', _D('false_seg_Sidewalk'), 'value', 0),
    ('5', _D('false_seg_Curb'), 'value', 0),
    ('3', _D('false_seg_Curb Cut'), 'value', 0),
    ('5', _D('false_seg_Road'), 'value', 0),
    ('1', _D('false_seg_Catch Basin'), 'value', 0),
    ('1', _D('false_seg_Manhole'), 'value', 0),
    ('33', _D('false_group_WALK'), 'value', 0),
    ('7', _D('false_group_ROAD'), 'value', 0),
    ('0.97', _D('walk_px_millions_ground'), 'value', 2),
    ('1.03', _D('walk_px_millions_secondary'), 'value', 2),
    ('0.91', _D('walk_px_millions_floor'), 'value', 2),
    ('0.12', _D('walk_px_millions_floor_standin'), 'value', 2),
    ('0.069', _D('walk_share_of_seen_ground'), 'value', 3),
    ('0.118', _D('walk_share_of_seen_floor'), 'value', 3),
    ('0.02', _D('object_share_of_surface_0-5'), 'value', 2),
    ('0.24', _D('object_share_of_surface_15-25'), 'value', 2),
    ('0.505', _D('sidewalk_on_dominant_pixel_pooled'), 'value', 3),
    ('0.27', _D('sidewalk_on_dominant_pano_median'), 'value', 2),
    ('410', _D('sidewalk_on_dominant_n_panos'), 'value', 0),
    ('-2', _D('det_lat_p99_deg'), 'value', 0),
    ('-36', _D('det_lat_p01_deg'), 'value', 0),
    ('-49', _D('det_lat_min_deg'), 'value', 0),
    ('4', _D('det_lat_max_deg'), 'value', 0),
    ('0.955', _D('true_on_walkroad_bend'), 'value', 3),
    ('0.930', _D('true_on_walkroad_paterson'), 'value', 3),
    ('0.924', _D('true_on_walkroad_gainesville'), 'value', 3),
    ('0.975', _D('true_on_walkroad_sao_paulo'), 'value', 3),
    ('0.896', _D('det_group_agree_direct'), 'value', 3),
    ('0.684', _D('det_class_agree_direct'), 'value', 3),
    ('0.958', _D('det_group_agree_direct4096'), 'value', 3),
    ('0.843', _D('det_class_agree_direct4096'), 'value', 3),
    ('1143', _D('det_n_true_false_missed'), 'value', 0),
    ('1', ('verdict.json', {}), 'tiers.0.55.per_city.bend.false_non_surface', 0),
    ('7', ('verdict.json', {}), 'tiers.0.55.per_city.bend.false_n', 0),
    ('0', ('verdict.json', {}), 'tiers.0.55.per_city.gainesville.false_non_surface', 0),
    ('9', ('verdict.json', {}), 'tiers.0.55.per_city.gainesville.false_n', 0),
    ('0', ('verdict.json', {}), 'tiers.0.55.per_city.paterson.false_non_surface', 0),
    ('5', ('verdict.json', {}), 'tiers.0.55.per_city.paterson.false_n', 0),
    ('1', ('verdict.json', {}), 'tiers.0.55.per_city.sao_paulo.false_non_surface', 0),
    ('21', ('verdict.json', {}), 'tiers.0.55.per_city.sao_paulo.false_n', 0),
    ('0.024', ('detection_classes.csv', {'city': 'pooled', 'group': 'false', 'arm': 'direct'}),
     'share_curb_cut', 3),
    ('0.191', ('detection_classes.csv', {'city': 'pooled', 'group': 'unsure_missed',
                                         'arm': 'tiled'}), 'share_curb_cut', 3),
    ('0.056', ('detection_classes.csv', {'city': 'pooled', 'group': 'unsure_missed',
                                         'arm': 'tiled'}), 'share_non_surface', 3),
    ('0.870', ('detection_classes.csv', {'city': 'pooled', 'group': 'true', 'arm': 'tiled'}),
     'share_WALK', 3),
    ('0.786', ('detection_classes.csv', {'city': 'pooled', 'group': 'false', 'arm': 'tiled'}),
     'share_WALK', 3),
    ('0.167', ('detection_classes.csv', {'city': 'pooled', 'group': 'false', 'arm': 'tiled'}),
     'share_ROAD', 3),
    ('0.825', ('detection_classes.csv', {'city': 'pooled', 'group': 'missed', 'arm': 'tiled'}),
     'share_WALK', 3),
    ('0.145', ('detection_classes.csv', {'city': 'pooled', 'group': 'missed', 'arm': 'tiled'}),
     'share_ROAD', 3),
    ('0.074', ('detection_classes.csv', {'city': 'pooled', 'group': 'true', 'arm': 'tiled'}),
     'share_ROAD', 3),
    ('0.107', _D('band_non_surface_share'), 'value', 3),
]


def check_numbers(pd, data):
    """Resolve every QUOTED number against the committed files; one row per number."""
    cache = {}

    def load(name):
        if name not in cache:
            path = (data if name in ('derived_numbers.csv',) else
                    pd if (pd / name).exists() else data) / name
            cache[name] = (json.loads(path.read_text(encoding='utf-8'))
                           if name.endswith('.json') else read_csv(path))
        return cache[name]
    out = []
    for quoted, (name, sel), col, nd in QUOTED:
        src = load(name)
        if name.endswith('.json'):
            v = src
            for part in col.replace('0.55', '0_55').split('.'):
                v = v['0.55' if part == '0_55' else part]
            row_desc = '-'
        else:
            hits = [r for r in src if all(r.get(k) == v for k, v in sel.items())]
            v = hits[0][col.removeprefix('1-')] if len(hits) == 1 else None
            row_desc = ', '.join(f'{k}={v}' for k, v in sel.items())
        actual = None if v in (None, '') else float(v)
        if col.startswith('1-') and actual is not None:
            actual = 1.0 - actual
        ok = actual is not None and round(actual, nd) == round(float(quoted), nd)
        out.append({'quoted': quoted, 'file': name, 'row': row_desc, 'column': col,
                    'value': actual, 'rounding': nd, 'status': 'ok' if ok else 'MISMATCH'})
    return out


VISTAS_V12_ORDER = [
    'Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall', 'Bike Lane',
    'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road',
    'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist',
    'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk', 'Lane Marking - General',
    'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench',
    'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box',
    'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light', 'Pole',
    'Traffic Sign Frame', 'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)',
    'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan',
    'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount',
    'Ego Vehicle']   # model.config.id2label at MODEL_REVISION; cmd_examples asserts the match


# --- examples and figures ----------------------------------------------------------------
#
# Two steps, so that every figure is byte-reproducible from COMMITTED files alone:
#   `examples` (needs the untracked work/ label maps, the payloads and the benchmark JPEGs;
#   no GPU, no network) selects every example by a fixed, stated rule and seed and writes
#   small JPEG panels + one CSV per set under docs/figures/footway-depth/examples/;
#   `figures` reads only committed CSVs (docs/figures/footway-depth/data/ and
#   runs/_pooled/footway/) and those panels, and writes every figure (PNG 200 dpi + SVG for
#   charts; JPEG 200 dpi for photographic contact sheets) plus data/figures_manifest.json
#   (sha256 of every output) and data/numbers.csv (every quoted number, checked).

# Categorical slots in fixed order (the dataviz reference palette, validated: adjacent CVD
# dE >= 9.1, normal-vision >= 19.6; three slots sit below 3:1 on white, so every chart carries
# visible labels and a CSV beside it). NONE is neutral gray.
GROUP_COLORS = {WALK: '#2a78d6', ROAD: '#eb6834', OBJECT: '#1baf7a', STRUCTURE: '#eda100',
                SKY: '#e87ba4', OTHER: '#008300', NONE: '#b0aea5'}
DEPTH_COLORS = {dad.GROUND: '#2a78d6', dad.FLOOR: '#1baf7a', FLOOR_STANDIN: '#eda100',
                dad.HORIZONTAL_NONFLOOR: '#e87ba4', dad.NON_HORIZONTAL: '#4a3aa7',
                dad.NO_PLANE: '#e34948'}
ARM_STYLE = {'correct': dict(color='#2a78d6', label='measured (correct geometry)'),
             'yaw180': dict(color='#8a8880', label='null: depth rotated 180 deg'),
             'mirror': dict(color='#c3c2b7', hatch='///', label='null: depth mirrored')}
DIRECT_STYLE = {'direct': dict(color='#2a78d6', label='direct equirect, 2048 px (5.7 px/deg)'),
                'tiled': dict(color='#0b0b0b', label='tiled (registered, 11.4 px/deg)'),
                'direct4096': dict(color='#eb6834', label='direct equirect, 4096 px (11.4 px/deg)')}
CITY_LABEL = {'bend': 'Bend', 'paterson': 'Paterson', 'gainesville': 'Gainesville',
              'sao_paulo': 'São Paulo', 'pooled': 'pooled'}
INK, INK2, GRID_INK, SURFACE_BG = '#0b0b0b', '#52514e', '#e4e3dd', '#fcfcfb'
EX_DIR = FIG_DIR / 'examples'
PANEL_W, PANEL_H = 320, 240
CROP_W_NORM, CROP_H_NORM = 0.1, 0.15          # a 36 x 27 deg equirect window
BOOT_N = 1000

OBJECT_EXAMPLE_CLASSES = [('car', ('Car',)), ('bus_truck', ('Bus', 'Truck', 'Other Vehicle')),
                          ('person', ('Person', 'Bicyclist', 'Motorcyclist')),
                          ('vegetation', ('Vegetation',)),
                          ('camera_car', ('Ego Vehicle', 'Car Mount')),
                          ('pole', ('Pole', 'Utility Pole', 'Street Light'))]
OBJECT_PER_CLASS, OBJECT_TOP = 2, 10
CURB_EXAMPLES, CURB_TOP, CURB_MIN_EX_PX = 6, 30, 200
SKY_EXAMPLES, SKY_TOP, SKY_MAX_LAT_DEG = 6, 20, -40.0
GALLERY_CATEGORIES = [
    ('surface_object', 'depth surface,\nsegmenter OBJECT'),
    ('surface_structure', 'depth surface,\nsegmenter STRUCTURE'),
    ('wall_walkroad', 'depth wall,\nsegmenter WALK/ROAD'),
    ('det_non_surface', 'detection >= 0.55,\nsegmenter non-surface'),
]
GALLERY_PER_CATEGORY = 6
GALLERY_TOP_PANOS = 20


def _hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def crop_window(city, pid, x, y, args):
    """(panel RGB array, xn grid, yn grid, marker px): a CROP_W_NORM x CROP_H_NORM equirect
    window centred on (x, y) at native resolution, wrapping across the seam, resized to
    PANEL_W x PANEL_H; xn/yn are the image-frame coordinates of every panel pixel centre."""
    y0 = min(max(0.0, y - CROP_H_NORM / 2), 1.0 - CROP_H_NORM)
    x0 = x - CROP_W_NORM / 2
    with Image.open(args.benchmark_root / city / 'panos' / f'{pid}.jpg') as im:
        W, H = im.size
        left, top = int(round(x0 * W)), int(round(y0 * H))
        w, h = int(round(CROP_W_NORM * W)), int(round(CROP_H_NORM * H))
        canvas = Image.new('RGB', (w, h))
        pos = 0
        while pos < w:
            src = (left + pos) % W
            take = min(w - pos, W - src)
            canvas.paste(im.crop((src, top, src + take, top + h)), (pos, 0))
            pos += take
    img = np.asarray(canvas.convert('RGB').resize((PANEL_W, PANEL_H), Image.LANCZOS))
    jj, ii = np.meshgrid(np.arange(PANEL_W), np.arange(PANEL_H))
    xn = np.mod(x0 + (jj + 0.5) / PANEL_W * CROP_W_NORM, 1.0)
    yn = y0 + (ii + 0.5) / PANEL_H * CROP_H_NORM
    return img, xn, yn, (PANEL_W / 2, (y - y0) / CROP_H_NORM * PANEL_H)


def overlay(img, classes, colors, alpha=0.55):
    """Blend a per-pixel class-colour layer over the panel."""
    layer = np.stack([_hex_rgb(colors[c]) for c in classes.ravel()]).reshape(img.shape)
    return np.clip(img * (1 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def seg_panel(img, xn, yn, labels, table_seg):
    g = table_seg[labels[np.minimum((yn * GRID_H).astype(int), GRID_H - 1),
                         np.minimum((xn * GRID_W).astype(int), GRID_W - 1)]]
    return overlay(img, np.array(SEG_GROUPS, dtype=object)[g], GROUP_COLORS)


def depth_panel(img, xn, yn, pix, dtab):
    ind = payload_indices(pix.payload)
    ph, pw = ind.shape
    d = dtab[ind[np.minimum((yn * ph).astype(int), ph - 1),
                 np.minimum((xn * pw).astype(int), pw - 1)]]
    return overlay(img, np.array(DEPTH_CLASSES, dtype=object)[d], DEPTH_COLORS)


def save_panel(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=85, optimize=True)


def largest_component_point(mask):
    """(size, row, col) of the mask's largest 8-connected component and its pixel nearest
    the component centroid; None for an empty mask."""
    from scipy import ndimage
    comp, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return None
    sizes = np.bincount(comp.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    rr, cc = np.nonzero(comp == k)
    i = int(np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2))
    return int(sizes[k - 1]), int(rr[i]), int(cc[i]), comp == k


def pano_context(args, city, pid, wd, table_seg):
    pix = dad.PayloadIndex(dad.read_payload(payload_path(args.run_root, city, pid)))
    ph, pw = pix.payload.height, pix.payload.width
    ind = payload_indices(pix.payload)
    dtab = depth_class_table(pix)
    rows, grow, gcol = grid_rows_cols(ph, pw)
    tiled = np.asarray(Image.open(wd / 'stitched' / city / f'{pid}.png'))
    lab = tiled[grow][:, gcol]
    yv = (grow + 0.5) / GRID_H
    min_dip = math.atan2(geo.DEFAULT_CAMERA_HEIGHT_M, geo.DEFAULT_MAX_RANGE_M)
    near = np.broadcast_to(((yv - 0.5) * math.pi > min_dip)[:, None], lab.shape)
    return {'pix': pix, 'ind': ind, 'dtab': dtab, 'rows': rows, 'tiled': tiled, 'lab': lab,
            'scls': table_seg[lab], 'dcls': dtab[ind[rows]], 'near': near, 'pw': pw, 'ph': ph}


def cmd_examples(args):
    """Select and render every example panel by a fixed rule and seed (module docstring)."""
    wd = work_dir(args)
    man = json.loads((wd / 'tile_labels' / 'segment_manifest.json').read_text(
        encoding='utf-8'))
    id2label = {int(k): v for k, v in man['id2label'].items()}
    fine_names = [id2label[i] for i in range(len(id2label))]
    assert fine_names == VISTAS_V12_ORDER, 'the model id order changed: update VISTAS_V12_ORDER'
    table_seg = collapse_table(id2label)
    surf_ids = [DEPTH_CLASSES.index(c) for c in DEPTH_SURFACE]
    sample = read_sample(args)
    cand = defaultdict(list)
    for n, r in enumerate(sample, 1):
        city, pid = r['city'], r['pano_id']
        c = pano_context(args, city, pid, wd, table_seg)
        surf = np.isin(c['dcls'], surf_ids) & c['near']
        for key, names in OBJECT_EXAMPLE_CLASSES:
            hit = largest_component_point(surf & np.isin(c['lab'], [fine_names.index(x)
                                                                    for x in names]))
            if hit:
                size, rr, cc, comp = hit
                cand[f'obj_{key}'].append(dict(
                    city=city, pano_id=pid, size=size, row=int(c['rows'][rr]), col=cc,
                    seg_class=Counter(c['lab'][comp].tolist()).most_common(1)[0][0],
                    depth_class=Counter(c['dcls'][comp].tolist()).most_common(1)[0][0]))
        sw = (c['lab'] == fine_names.index('Sidewalk')) & c['near'] & \
            (c['dcls'] == DEPTH_CLASSES.index(dad.FLOOR))
        if sw.sum() >= CURB_MIN_EX_PX:
            hit = largest_component_point(sw)
            cand['curb'].append(dict(city=city, pano_id=pid, size=int(sw.sum()), comp=None,
                                     row=int(c['rows'][hit[1]]), col=hit[2], mask=hit[3]))
        d4 = np.asarray(Image.open(wd / 'direct4096_labels' / city / f'{pid}.png'))
        lat = 90.0 - (np.arange(GRID_H) + 0.5) / GRID_H * 180.0
        low = np.broadcast_to((lat < SKY_MAX_LAT_DEG)[:, None], d4.shape)
        sky = low & (table_seg[c['tiled']] == SEG_GROUPS.index(ROAD)) & \
            (table_seg[d4] == SEG_GROUPS.index(SKY))
        hit = largest_component_point(sky)
        if hit:
            cand['sky'].append(dict(city=city, pano_id=pid, size=int(sky.sum()), row=hit[1],
                                    col=hit[2],
                                    share_low_road_as_sky=float(sky.sum() / max(1, (
                                        low & (table_seg[c['tiled']] == SEG_GROUPS.index(
                                            ROAD))).sum()))))
        if n % 100 == 0:
            print(f'  examples: scanned {n}/{len(sample)}', flush=True)

    def draw(key, top, k):
        pool = sorted(cand[key], key=lambda e: (-e['size'], e['city'], e['pano_id']))[:top]
        return random.Random(f'{SEED}:{key}').sample(pool, min(k, len(pool)))

    out = {}
    # 1. objects on a depth floor plane: two per class family
    rows = []
    for key, names in OBJECT_EXAMPLE_CLASSES:
        for e in draw(f'obj_{key}', OBJECT_TOP, OBJECT_PER_CLASS):
            c = pano_context(args, e['city'], e['pano_id'], wd, table_seg)
            x, y = (e['col'] + 0.5) / c['pw'], (e['row'] + 0.5) / c['ph']
            img, xn, yn, mk = crop_window(e['city'], e['pano_id'], x, y, args)
            i = len(rows)
            save_panel(img, EX_DIR / 'objects' / f'{i:02d}_crop.jpg')
            save_panel(seg_panel(img, xn, yn, c['tiled'], table_seg),
                       EX_DIR / 'objects' / f'{i:02d}_seg.jpg')
            save_panel(depth_panel(img, xn, yn, c['pix'], c['dtab']),
                       EX_DIR / 'objects' / f'{i:02d}_depth.jpg')
            rng_m = flat_range(fs.pano_pose(None, fs.POSE_OFF), y) if False else \
                range_from_y(y)
            rows.append({'idx': i, 'family': key, 'city': e['city'],
                         'pano_id': e['pano_id'], 'x': x, 'y': y,
                         'component_px': e['size'], 'seg_class': id2label[e['seg_class']],
                         'depth_class': DEPTH_CLASSES[e['depth_class']],
                         'range_flat_2p6_m': rng_m, 'marker_px': f'{mk[0]:.1f};{mk[1]:.1f}'})
    out['objects'] = rows
    # 2. curb: sidewalk on a measured secondary floor plane, the offset at one pixel
    rows = []
    for e in draw('curb', CURB_TOP, CURB_EXAMPLES):
        c = pano_context(args, e['city'], e['pano_id'], wd, table_seg)
        rr, cc = np.nonzero(e['mask'])
        r0 = int(np.searchsorted(c['rows'], e['row']))
        order = np.argsort((rr - r0) ** 2 + (cc - e['col']) ** 2, kind='stable')
        pick = None
        for j in order:
            r, col = int(c['rows'][rr[j]]), int(cc[j])
            x, y = (col + 0.5) / c['pw'], (r + 0.5) / c['ph']
            cls, f = depth_class_at(c['pix'], x, y)
            ref = reference_plane(c['pix'], c['ind'], r, col, f.get('rows_to_ref'))
            if f.get('offset_local_m') is not None and cls == dad.FLOOR and \
                    not depthlib.is_standin(ref):
                pick = (x, y, f)
                break
        if pick is None:
            continue
        x, y, f = pick
        img, xn, yn, mk = crop_window(e['city'], e['pano_id'], x, y, args)
        i = len(rows)
        save_panel(img, EX_DIR / 'curb' / f'{i:02d}_crop.jpg')
        save_panel(depth_panel(img, xn, yn, c['pix'], c['dtab']),
                   EX_DIR / 'curb' / f'{i:02d}_depth.jpg')
        rows.append({'idx': i, 'city': e['city'], 'pano_id': e['pano_id'], 'x': x, 'y': y,
                     'sidewalk_floor_px': e['size'], 'offset_local_m': f['offset_local_m'],
                     'rows_to_ref': f.get('rows_to_ref'),
                     'range_flat_2p6_m': range_from_y(y),
                     'marker_px': f'{mk[0]:.1f};{mk[1]:.1f}'})
    out['curb'] = rows
    # 3. the 4096 px direct arm's SKY under the car
    rows = []
    for e in draw('sky', SKY_TOP, SKY_EXAMPLES):
        c = pano_context(args, e['city'], e['pano_id'], wd, table_seg)
        d4 = np.asarray(Image.open(wd / 'direct4096_labels' / e['city'] / f"{e['pano_id']}.png"))
        x, y = (e['col'] + 0.5) / GRID_W, (e['row'] + 0.5) / GRID_H
        img, xn, yn, mk = crop_window(e['city'], e['pano_id'], x, y, args)
        i = len(rows)
        save_panel(img, EX_DIR / 'sky' / f'{i:02d}_crop.jpg')
        save_panel(seg_panel(img, xn, yn, c['tiled'], table_seg),
                   EX_DIR / 'sky' / f'{i:02d}_tiled.jpg')
        save_panel(seg_panel(img, xn, yn, d4, table_seg), EX_DIR / 'sky' / f'{i:02d}_d4096.jpg')
        rows.append({'idx': i, 'city': e['city'], 'pano_id': e['pano_id'], 'x': x, 'y': y,
                     'road_as_sky_px': e['size'],
                     'share_low_road_as_sky': e['share_low_road_as_sky'],
                     'marker_px': f'{mk[0]:.1f};{mk[1]:.1f}'})
    out['sky'] = rows
    # 4. the disagreement gallery (the earlier fixed rule, gallery_candidates)
    rows = []
    for p in gallery_candidates(args):
        img, xn, yn, mk = crop_window(p['city'], p['pano_id'], p['x'], p['y'], args)
        i = len(rows)
        save_panel(img, EX_DIR / 'disagreement' / f'{i:02d}_crop.jpg')
        rows.append({'idx': i, **p, 'marker_px': f'{mk[0]:.1f};{mk[1]:.1f}'})
    out['disagreement'] = rows
    # 5. the tiling figure's pano: the first row of sample.csv
    r = sample[0]
    t = EX_DIR / 'tiling'
    t.mkdir(parents=True, exist_ok=True)
    with Image.open(args.benchmark_root / r['city'] / 'panos' / f"{r['pano_id']}.jpg") as im:
        im.convert('RGB').resize((GRID_W, GRID_H), Image.LANCZOS).save(
            t / 'pano.jpg', quality=85, optimize=True)
    Image.open(wd / 'stitched' / r['city'] / f"{r['pano_id']}.png").save(t / 'stitched.png')
    out['tiling'] = [{'idx': 0, 'city': r['city'], 'pano_id': r['pano_id']}]
    for name, rows in out.items():
        write_csv(EX_DIR / f'{name}.csv', rows,
                  sorted({k for row in rows for k in row},
                         key=lambda k: (k not in ('idx', 'family', 'category', 'city',
                                                  'pano_id'), k)))
    size = sum(p.stat().st_size for p in EX_DIR.rglob('*') if p.is_file())
    print(f'examples: {sum(len(v) for v in out.values())} examples, '
          f'{size / 2**20:.2f} MiB -> {EX_DIR}')


def range_from_y(y):
    """Flat-raycast horizontal range at 2.6 m for an image row (geo; the range does not
    depend on heading or position), None beyond 25 m or at the horizon."""
    pano = fs.SlimPano(pano_id='-', lat=0.0, lng=0.0, camera_heading=0.0, camera_pitch=None,
                       camera_roll=None, capture_date=None, source='launch', detections=[])
    return flat_range(fs.pano_pose(pano, fs.POSE_OFF), y)


def gallery_candidates(args):
    """Fixed-rule disagreement picks: for each pixel category, rank panos by the size of
    the category's largest 8-connected component within 25 m (payload grid), take the top
    GALLERY_TOP_PANOS and draw GALLERY_PER_CATEGORY with random.Random(SEED); the point
    is the component pixel nearest its centroid. Detections: every >= 0.55 detection
    whose tiled group is non-surface, drawn the same way. Needs the untracked work/."""
    wd = work_dir(args)
    man = json.loads((wd / 'tile_labels' / 'segment_manifest.json').read_text(
        encoding='utf-8'))
    id2label = {int(k): v for k, v in man['id2label'].items()}
    table_seg = collapse_table(id2label)
    cands = defaultdict(list)
    surf_ids = [DEPTH_CLASSES.index(c) for c in DEPTH_SURFACE]
    for r in read_sample(args):
        city, pid = r['city'], r['pano_id']
        c = pano_context(args, city, pid, wd, table_seg)
        dcls, scls, lab, near = c['dcls'], c['scls'], c['lab'], c['near']
        surf = np.isin(dcls, surf_ids)
        masks = {'surface_object': surf & (scls == SEG_GROUPS.index(OBJECT)),
                 'surface_structure': surf & (scls == SEG_GROUPS.index(STRUCTURE)),
                 'wall_walkroad': (dcls == DEPTH_CLASSES.index(dad.NON_HORIZONTAL))
                 & np.isin(scls, [SEG_GROUPS.index(WALK), SEG_GROUPS.index(ROAD)])}
        for cat, m in masks.items():
            hit = largest_component_point(m & near)
            if not hit:
                continue
            size, rr, cc, _ = hit
            cands[cat].append({'city': city, 'pano_id': pid, 'size': size,
                               'x': (cc + 0.5) / c['pw'],
                               'y': (int(c['rows'][rr]) + 0.5) / c['ph'],
                               'depth': DEPTH_CLASSES[dcls[rr, cc]],
                               'seg': id2label.get(int(lab[rr, cc]), 'NONE')})
    picks = []
    for cat, _ in GALLERY_CATEGORIES[:3]:
        top = sorted(cands[cat], key=lambda e: (-e['size'], e['city'], e['pano_id']))
        top = top[:GALLERY_TOP_PANOS]
        chosen = random.Random(SEED).sample(top, min(GALLERY_PER_CATEGORY, len(top)))
        picks += [{'category': cat, **e} for e in chosen]
    dets = [d for d in read_csv(pooled_dir(args.out_root) / 'detections.csv')
            if d['kind'] == 'det' and d['gt_group'] != 'band_0p30'
            and d['seg_group'] not in SURFACE_GROUPS]
    dets.sort(key=lambda d: (d['city'], d['pano_id'], int(d['det_index'])))
    for d in random.Random(SEED).sample(dets, min(GALLERY_PER_CATEGORY, len(dets))):
        picks.append({'category': 'det_non_surface', 'city': d['city'],
                      'pano_id': d['pano_id'], 'size': None, 'x': float(d['x']),
                      'y': float(d['y']), 'depth': d['depth_class'], 'seg': d['seg_class'],
                      'verdict': d['gt_group']})
    return picks


# --- figures (committed inputs only) -----------------------------------------------------

def _boot_stats(rows, arm, keys, seed=SEED, n=BOOT_N):
    """(point, lo, hi) per key of headline_stats over the panos in `rows` (pano bootstrap,
    95% percentile interval): pixels inside one pano are not independent, so the pano is
    the resampling unit."""
    mats = np.stack([matrix_from_row(r, arm) for r in rows])
    point = headline_stats(mats.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(n):
        st = headline_stats(mats[rng.integers(0, len(rows), len(rows))].sum(axis=0))
        for k in keys:
            if st[k] is not None:
                draws[k].append(st[k])
    return {k: (point[k], float(np.percentile(draws[k], 2.5)),
                float(np.percentile(draws[k], 97.5))) for k in keys}


def _suptitle(fig, text, x=0.01, ha='left', fontsize=12.5, fontweight='bold', **kw):
    import textwrap
    fig.suptitle(textwrap.fill(text, 96), x=x, ha=ha, fontsize=fontsize,
                 fontweight=fontweight, **kw)


def _note(fig, x, y, text, fontsize=8.5, color=INK2, **kw):
    import textwrap
    fig.text(x, y, textwrap.fill(text, 165), fontsize=fontsize, color=color, va='top', **kw)


def _style(plt):
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.hashsalt': 'footway47', 'svg.fonttype': 'path', 'path.simplify': False,
        'font.family': 'DejaVu Sans'})


def _save(fig, out_dir, stem, svg=True, photo=False):
    """PNG (charts) or JPEG (photographic sheets) at 200 dpi, plus SVG for charts; metadata
    that would carry a timestamp or version is dropped so the bytes are reproducible."""
    paths = []
    if photo:
        p = out_dir / f'{stem}.jpg'
        fig.savefig(p, dpi=200, pil_kwargs={'quality': 88, 'optimize': True})
        paths.append(p)
    else:
        p = out_dir / f'{stem}.png'
        fig.savefig(p, dpi=200, metadata={'Software': None})
        paths.append(p)
    if svg:
        import io
        p = out_dir / f'{stem}.svg'
        buf = io.BytesIO()     # bytes, so the SVG is LF on every platform (the blob is LF)
        fig.savefig(buf, format='svg', metadata={'Date': None, 'Creator': None})
        p.write_bytes(buf.getvalue())
        paths.append(p)
    return paths


def _panel(ax, path):
    ax.imshow(np.asarray(Image.open(path)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _tag(ax, text, loc='top'):
    """A label written INSIDE the image panel (white box, dark ink)."""
    y, va = (0.03, 'bottom') if loc == 'bottom' else (0.97, 'top')
    ax.text(0.03, y, text, transform=ax.transAxes, ha='left', va=va, fontsize=7.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))


def _marker(ax, mk):
    x, y = (float(v) for v in mk.split(';'))
    ax.plot([x], [y], marker='o', ms=12, mfc='none', mec='white', mew=2.6)
    ax.plot([x], [y], marker='o', ms=12, mfc='none', mec='#e34948', mew=1.3)


def _class_legend(fig, colors, title, y, x0=0.02):
    from matplotlib.patches import Patch
    handles = [Patch(fc=c, ec='none', label=k) for k, c in colors.items()]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(x0, y), ncol=len(handles),
               frameon=False, title=title, title_fontsize=9, fontsize=8.5,
               handlelength=1.2, columnspacing=1.0, alignment='left')


def wilson_err(k, n):
    lo, hi = es.wilson(k, n)
    p = k / n if n else 0.0
    return p, p - lo, hi - p, lo, hi


def fig_base_rate(plt, pd, fig_dir):
    counts = read_csv(pd / 'pano_counts_le25m.csv')
    stats = [('p_walkroad_given_surface', 'P(WALK or ROAD |\ndepth surface)', 'p_walkroad'),
             ('p_structure_given_wall', 'P(STRUCTURE |\ndepth wall)', 'p_structure'),
             ('p_surface_given_object', 'P(depth surface |\nOBJECT)', 'p_surface'),
             ('kappa', "Cohen's kappa\n(surface / vertical / unmodelled)", None)]
    keys = [k for k, _, _ in stats] + [m for _, _, m in stats if m]
    arms = ('correct', 'yaw180', 'mirror')
    pooled = {a: _boot_stats(counts, a, keys) for a in arms}
    cities = DEFAULT_CITIES
    per_city = {c: {a: _boot_stats([r for r in counts if r['city'] == c], a, keys)
                    for a in arms} for c in cities}
    rows_out = []
    for scope, res in [('pooled', pooled)] + [(c, per_city[c]) for c in cities]:
        for a in arms:
            for k in keys:
                p, lo, hi = res[a][k]
                rows_out.append({'scope': scope, 'arm': a, 'statistic': k, 'value': p,
                                 'boot_lo': lo, 'boot_hi': hi})
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.62, wspace=0.32,
                          left=0.08, right=0.98, top=0.8, bottom=0.11)
    ax = fig.add_subplot(gs[0, :])
    width = 0.26
    for si, (k, label, marg) in enumerate(stats):
        for ai, a in enumerate(arms):
            p, lo, hi = pooled[a][k]
            st = ARM_STYLE[a]
            ax.bar(si + (ai - 1) * width, p, width * 0.92, color=st['color'],
                   hatch=st.get('hatch'), edgecolor='white', linewidth=1.5,
                   label=st['label'] if si == 0 else None)
            ax.errorbar(si + (ai - 1) * width, p, yerr=[[p - lo], [hi - p]], fmt='none',
                        ecolor=INK, elinewidth=1.2, capsize=3)
            if a == 'correct':
                ax.text(si + (ai - 1) * width, p / 2, f'{p:.3f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white', rotation=90)
        if marg:
            m = pooled['correct'][marg][0]
            ax.plot([si - 1.6 * width, si + 1.6 * width], [m, m], color=INK, lw=1.6,
                    ls=(0, (4, 2)), label='marginal (base rate)' if si == 0 else None)
    ax.set_xticks(range(len(stats)), [lab for _, lab, _ in stats])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('value, within 25 m (pooled, 416 panos)')
    ax.grid(axis='y', color=GRID_INK, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False)
    _suptitle(fig, 'Surface "agreement" is a base rate: scrambling depth leaves it unchanged; '
                 'only the walls (and kappa) carry signal', x=0.07, ha='left', fontsize=12.5,
                 fontweight='bold')
    for si, (k, label, marg) in enumerate(stats):
        a2 = fig.add_subplot(gs[1, si])
        for ci, c in enumerate(cities):
            for ai, a in enumerate(arms):
                p, lo, hi = per_city[c][a][k]
                yy = ci + (ai - 1) * 0.22
                a2.plot([lo, hi], [yy, yy], color=ARM_STYLE[a]['color'] if a != 'mirror'
                        else '#a8a69e', lw=2)
                a2.plot([p], [yy], 'o', ms=6, color=ARM_STYLE[a]['color'] if a != 'mirror'
                        else '#a8a69e', mec='white', mew=1)
            if marg:
                m = per_city[c]['correct'][marg][0]
                a2.plot([m, m], [ci - 0.4, ci + 0.4], color=INK, lw=1.4, ls=(0, (3, 2)))
        a2.set_yticks(range(len(cities)), [CITY_LABEL[c] for c in cities])
        a2.invert_yaxis()
        a2.set_title("Cohen's kappa" if k == 'kappa' else label.replace('\n', ' '),
                     fontsize=9.5)
        a2.grid(axis='x', color=GRID_INK, lw=0.8)
        a2.set_axisbelow(True)
    _note(fig, 0.08, 0.05, 'Bars and dots: pooled pixel statistic; whiskers: 95% pano-bootstrap '
             f'interval ({BOOT_N} resamples, seed {SEED}). Dashed: the marginal the '
             'conditional should be read against. Bottom row: per city (blue measured, '
             'grays the two nulls).', fontsize=8.5, color=INK2)
    paths = _save(fig, fig_dir, 'fig1_base_rate')
    plt.close(fig)
    return paths, {'fig1_base_rate.csv': rows_out}


def fig_objects(plt, fig_dir):
    ex = read_csv(EX_DIR / 'objects.csv')
    n = len(ex)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, 3 * ncol, figsize=(12, 1.55 * nrow + 1.6),
                             gridspec_kw=dict(wspace=0.03, hspace=0.08, left=0.01,
                                              right=0.99, top=0.86, bottom=0.13))
    for i, e in enumerate(ex):
        r, c0 = i // ncol, 3 * (i % ncol)
        stem = EX_DIR / 'objects' / f"{int(e['idx']):02d}"
        for j, (suffix, tag) in enumerate((('crop', None), ('seg', 'segmenter'),
                                           ('depth', 'depth plane class'))):
            ax = axes[r, c0 + j]
            _panel(ax, f'{stem}_{suffix}.jpg')
            _marker(ax, e['marker_px'])
            if j == 0:
                rng = e['range_flat_2p6_m']
                _tag(ax, f"{e['seg_class']} on depth {e['depth_class']}"
                     + (f", {float(rng):.0f} m" if rng else ''))
                _tag(ax, f"{e['city']}:{e['pano_id'][:11]}", loc='bottom')
            else:
                _tag(ax, tag)
    for ax in axes.ravel()[n * 3:]:
        ax.axis('off')
    _suptitle(fig, 'Depth draws the ground through objects: under every OBJECT the segmenter '
                 'finds, depth reports a floor plane', x=0.01, ha='left', fontsize=12.5,
                 fontweight='bold')
    _note(fig, 0.01, 0.905, 'Rule: for each of six OBJECT families, the 2 panos drawn (seed 47) '
             'from the 10 with the largest within-25 m component of that family on a depth '
             'floor plane; red ring = the component pixel nearest its centroid.',
             fontsize=8.5, color=INK2)
    _class_legend(fig, {g: GROUP_COLORS[g] for g in SEG_GROUPS if g != NONE},
                  'segmenter group (middle panels)', 0.05)
    _class_legend(fig, DEPTH_COLORS, 'depth plane class (right panels)', 0.0)
    paths = _save(fig, fig_dir, 'fig2_objects_on_floor', svg=False, photo=True)
    plt.close(fig)
    return paths, {}


def fig_matrix(plt, pd, fig_dir):
    counts = read_csv(pd / 'pano_counts_le25m.csv')
    groups = [g for g in SEG_GROUPS if g != NONE]
    mats = {a: sum(matrix_from_row(r, a) for r in counts) for a in ('correct', 'yaw180')}
    share = {a: m[:, :len(groups)] / np.maximum(1, m[:, :len(groups)].sum(axis=1,
                                                                          keepdims=True))
             for a, m in mats.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(wspace=0.62,
                                                                        left=0.17, right=0.98,
                                                                        top=0.8, bottom=0.14))
    out_rows = []
    for ax, a, title in ((axes[0], 'correct', 'measured'),
                         (axes[1], 'yaw180', 'null: depth rotated 180 deg')):
        ax.imshow(share[a], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(share[a].shape[0]):
            for j in range(share[a].shape[1]):
                v = share[a][i, j]
                txt = f'{v:.2f}'
                if a == 'correct':
                    nv = share['yaw180'][i, j]
                    lift = v / nv if nv > 0 else None
                    if lift is not None and v >= 0.05 and mats['correct'][i].sum() >= 10000:
                        txt += f'\nx{lift:.2f}'
                    out_rows.append({'depth_class': DEPTH_CLASSES[i], 'group': groups[j],
                                     'share_correct': v, 'share_yaw180': nv,
                                     'lift': lift, 'n_correct': mats['correct'][i, j]})
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if v > 0.55 else INK)
        ax.set_xticks(range(len(groups)), groups, fontsize=8.5)
        ax.set_yticks(range(len(DEPTH_CLASSES)),
                      [f'{d} (n={int(mats[a][i].sum()):,})' for i, d in enumerate(DEPTH_CLASSES)],
                      fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel('segmenter group (tiled arm)')
        for sp in ax.spines.values():
            sp.set_visible(False)
    _suptitle(fig, 'Where the signal is: the wall row (STRUCTURE x1.6 over the null); the floor '
                 'rows barely move', x=0.01, ha='left', fontsize=12.5, fontweight='bold')
    _note(fig, 0.01, 0.88, 'Row shares over below-horizon pixels within 25 m, pooled 416 panos; '
             'left cells also show the lift over the null (where the share >= 0.05 and the row has >= 10,000 pixels).',
             fontsize=8.5, color=INK2)
    paths = _save(fig, fig_dir, 'fig3_where_signal')
    plt.close(fig)
    return paths, {'fig3_where_signal.csv': out_rows}


def fig_curb(plt, pd, data, fig_dir):
    med = read_csv(pd / 'curb_pano_medians.csv')
    summ = {(r['group'], r['reading']): r for r in read_csv(data / 'curb_height.csv')
            if r['city'] == 'pooled'}
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.27)
    bins = np.arange(-0.30, 0.41, 0.02)
    ax.axvspan(*CLAIM_BAND_M, color='#efeee9', zorder=0)
    ax.text(0.24, 0.5, 'claimed curb band\n[0.05, 0.30) m', transform=ax.get_xaxis_transform(),
            ha='center', va='center', fontsize=9.5, color=INK2)
    for g, col, lab in (('sidewalk', GROUP_COLORS[WALK], 'Sidewalk on a secondary floor plane'),
                        ('road', GROUP_COLORS[ROAD], 'Road on a secondary floor plane (control)')):
        v = [float(r['median_offset_m']) for r in med
             if r['group'] == g and r['reading'] == 'measured']
        s = summ[(g, 'measured')]
        m, lo, hi = (float(s[k]) for k in ('median_of_pano_medians', 'median_lo', 'median_hi'))
        ax.hist(np.clip(v, bins[0], bins[-1] - 1e-9), bins=bins, histtype='step', lw=2,
                color=col, label=f'{lab} ({len(v)} panos): median of pano medians '
                                 f'{m:.3f} m [{lo:.3f}, {hi:.3f}]; '
                                 f'{float(s["share_in_claim_band"]):.0%} of pixel offsets in band')
        ax.axvspan(lo, hi, color=col, alpha=0.25, lw=0)
        ax.axvline(m, color=col, lw=1.5)
    ax.axvline(0.15, color=INK, lw=1.2, ls=(0, (4, 2)))
    ax.text(0.152, 0.62, 'the quoted\n~0.15 m', transform=ax.get_xaxis_transform(),
            fontsize=9, color=INK)
    ax.set_xlabel('per-pano median height above the local road, offset_local (m); '
                  'stand-ins excluded; within 25 m')
    ax.set_ylabel('panos')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, -0.17), fontsize=9)
    _suptitle(fig, 'GSV depth carries no curb step: the sidewalk plane sits 0.02 m above the '
                 'road, far below the claimed band', x=0.07, ha='left', fontsize=12.5,
                 fontweight='bold')
    _note(fig, 0.07, 0.885, 'Vertical line and shading: median of per-pano medians and its 95% '
             'order-statistic CI. Values outside [-0.30, 0.40] are drawn in the end bins.',
             fontsize=8.5, color=INK2)
    paths = _save(fig, fig_dir, 'fig4_curb_height')
    plt.close(fig)
    ex = read_csv(EX_DIR / 'curb.csv')
    fig, axes = plt.subplots(2, 2 * 3, figsize=(12, 3.9),
                             gridspec_kw=dict(wspace=0.03, hspace=0.08, left=0.01, right=0.99,
                                              top=0.74, bottom=0.17))
    for i, e in enumerate(ex):
        r, c0 = i // 3, 2 * (i % 3)
        stem = EX_DIR / 'curb' / f"{int(e['idx']):02d}"
        _panel(axes[r, c0], f'{stem}_crop.jpg')
        _panel(axes[r, c0 + 1], f'{stem}_depth.jpg')
        for ax in axes[r, c0:c0 + 2]:
            _marker(ax, e['marker_px'])
        _tag(axes[r, c0], f"offset {float(e['offset_local_m']):+.3f} m")
        _tag(axes[r, c0], f"{e['city']}:{e['pano_id'][:11]}", loc='bottom')
        _tag(axes[r, c0 + 1], 'depth plane class')
    for ax in axes.ravel()[len(ex) * 2:]:
        ax.axis('off')
    _suptitle(fig, 'Sidewalk on its own floor plane: the measured step above the road is '
                 'centimetres', x=0.01, ha='left', fontsize=12.5, fontweight='bold')
    _note(fig, 0.01, 0.87, 'Rule: 6 panos drawn (seed 47) from the 30 with the most '
             'Sidewalk-on-measured-floor pixels within 25 m; the ring is the pixel nearest the '
             'largest component\'s centroid with a non-stand-in local reference. Single-pixel '
             'offsets scatter by +-0.2 m; the reading uses per-pano medians (fig. 4).',
             fontsize=8.5, color=INK2)
    _class_legend(fig, DEPTH_COLORS, 'depth plane class', 0.0)
    paths += _save(fig, fig_dir, 'fig4b_curb_examples', svg=False, photo=True)
    plt.close(fig)
    return paths, {}


def fig_fp(plt, pd, fig_dir):
    dets = read_csv(pd / 'detections.csv')
    groups = [('true', 'verdict True'), ('false', 'verdict False'),
              ('missed', 'missed mark')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5),
                             gridspec_kw=dict(wspace=0.5, width_ratios=[1.1, 1.3, 1.1],
                                              left=0.09, right=0.97, top=0.78, bottom=0.27))
    out_rows = []
    ax = axes[0]
    for i, (g, lab) in enumerate(groups):
        gi = [d for d in dets if d['gt_group'] == g]
        k = sum(d['seg_group'] not in SURFACE_GROUPS for d in gi)
        p, el, eh, lo, hi = wilson_err(k, len(gi))
        ax.errorbar([p], [i], xerr=[[el], [eh]], fmt='o', ms=8, color=GROUP_COLORS[WALK],
                    ecolor=INK, capsize=4, mec='white', mew=1)
        ax.text(hi + 0.01, i, f'{k}/{len(gi)} = {p:.3f}', va='center', fontsize=9)
        out_rows.append({'panel': 'non_surface', 'group': g, 'k': k, 'n': len(gi), 'share': p,
                         'wilson_lo': lo, 'wilson_hi': hi})
    t_share = out_rows[0]['share']
    ax.axvline(t_share + RULE_C_MARGIN, color='#e34948', lw=1.6, ls=(0, (4, 2)))
    ax.text(t_share + RULE_C_MARGIN + 0.005, 2.45, 'registered bar:\nTrue + 20 pts', fontsize=8.5,
            color=INK, va='bottom')
    ax.set_yticks(range(len(groups)), [lab for _, lab in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.42)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel('share NOT on WALK or ROAD (Wilson 95%)')
    ax.set_title('(C) registered: NOT SUPPORTED', fontsize=10.5)
    ax.grid(axis='x', color=GRID_INK, lw=0.8)
    ax = axes[1]
    for i, (g, lab) in enumerate(groups):
        gi = [d for d in dets if d['gt_group'] == g]
        c = Counter(d['seg_group'] for d in gi)
        left = 0.0
        for grp in SEG_GROUPS:
            w = c[grp] / len(gi)
            if w:
                ax.barh(i, w, left=left, color=GROUP_COLORS[grp], height=0.6,
                        edgecolor='white', lw=2, label=grp if i == 0 or grp not in c else None)
                if w >= 0.06:
                    ax.text(left + w / 2, i, f'{w:.2f}', ha='center', va='center', fontsize=8.5,
                            color='white' if grp in (WALK, OBJECT) else INK)
            left += w
            out_rows.append({'panel': 'groups', 'group': g, 'seg_group': grp, 'k': c[grp],
                             'n': len(gi), 'share': w})
    handles, labels = axes[1].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend([seen[k] for k in SEG_GROUPS if k in seen], [k for k in SEG_GROUPS if k in seen],
              ncol=3, frameon=False, fontsize=8.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.17))
    ax.set_yticks(range(len(groups)), [lab for _, lab in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('segmenter group at the pixel (tiled)')
    ax = axes[2]
    for i, (g, lab) in enumerate(groups):
        gi = [d for d in dets if d['gt_group'] == g]
        k = sum(d['seg_class'] == 'Curb Cut' for d in gi)
        p, el, eh, lo, hi = wilson_err(k, len(gi))
        ax.errorbar([p], [i], xerr=[[el], [eh]], fmt='D', ms=7, color=GROUP_COLORS[OBJECT],
                    ecolor=INK, capsize=4, mec='white', mew=1)
        ax.text(hi + 0.01, i, f'{p:.3f}', va='center', fontsize=9)
        out_rows.append({'panel': 'curb_cut_exploratory', 'group': g, 'k': k, 'n': len(gi),
                         'share': p, 'wilson_lo': lo, 'wilson_hi': hi})
    ax.set_yticks(range(len(groups)), [lab for _, lab in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.set_xlabel('share on Vistas Curb Cut (Wilson 95%)')
    ax.set_title('EXPLORATORY (post hoc):\nCurb Cut at the pixel', fontsize=10.5,
                 color='#c0392b')
    ax.grid(axis='x', color=GRID_INK, lw=0.8)
    _suptitle(fig, 'The segmenter class at the peak pixel is no false-positive filter: '
                 'the 42 false positives are footway too', x=0.01, ha='left', fontsize=12.5,
                 fontweight='bold')
    paths = _save(fig, fig_dir, 'fig5_fp_rule')
    plt.close(fig)
    return paths, {'fig5_fp_rule.csv': out_rows}


def fig_trap(plt, pd, fig_dir):
    dets = read_csv(pd / 'detections.csv')
    tp = read_csv(pd / 'trap_pano.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1.5], left=0.07,
                                              right=0.98, top=0.76, bottom=0.13))
    out_rows = []
    ax = axes[0]
    arms = [('direct', 'direct_class'), ('tiled', 'seg_class'),
            ('direct4096', 'direct4096_class')]
    for gi_, (g, lab) in enumerate((('true', 'under verdict-True\ndetections'),
                                    ('missed', 'under missed marks'))):
        rows = [d for d in dets if d['gt_group'] == g]
        for ai, (arm, col) in enumerate(arms):
            k = sum(d[col] == 'Curb Cut' for d in rows)
            p, el, eh, lo, hi = wilson_err(k, len(rows))
            x = gi_ + (ai - 1) * 0.25
            ax.bar(x, p, 0.23, color=DIRECT_STYLE[arm]['color'], edgecolor='white', lw=1.5,
                   label=DIRECT_STYLE[arm]['label'] if gi_ == 0 else None)
            ax.errorbar(x, p, yerr=[[el], [eh]], fmt='none', ecolor=INK, capsize=3)
            ax.text(x, hi + 0.01, f'{p:.2f}', ha='center', fontsize=9)
            out_rows.append({'panel': 'curb_cut', 'group': g, 'arm': arm, 'k': k,
                             'n': len(rows), 'share': p, 'wilson_lo': lo, 'wilson_hi': hi})
    ax.set_xticks([0, 1], ['under verdict-True\ndetections', 'under missed marks'])
    ax.set_ylabel('share on Vistas `Curb Cut` (Wilson 95%)')
    ax.set_ylim(0, 0.52)
    ax.set_title('Curb Cut at the detection (post hoc)', fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(axis='y', color=GRID_INK, lw=0.8)
    ax.set_axisbelow(True)
    ax = axes[1]
    bands = [b for b in LAT_BANDS if any(r['lat_band_deg'] == f'{b}..{b + 10}' for r in tp)]
    rng = np.random.default_rng(SEED)
    pids = sorted({(r['city'], r['pano_id']) for r in tp})
    idx = {p: i for i, p in enumerate(pids)}
    for arm in DIRECT_ARMS:
        for band, ls in (('interior', '-'), ('seam', (0, (4, 2)))):
            seen = np.zeros((len(pids), len(bands)))
            agree = np.zeros_like(seen)
            for r in tp:
                if r['band'] != band:
                    continue
                bi = bands.index(int(r['lat_band_deg'].split('..')[0]))
                i = idx[(r['city'], r['pano_id'])]
                seen[i, bi] = float(r['n_seen_px'])
                agree[i, bi] = float(r[f'n_agree_{arm}'])
            pt = agree.sum(0) / np.maximum(1, seen.sum(0))
            boots = []
            for _ in range(BOOT_N):
                s = rng.integers(0, len(pids), len(pids))
                boots.append(agree[s].sum(0) / np.maximum(1, seen[s].sum(0)))
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            xs = np.array(bands) + 5
            col = DIRECT_STYLE[arm]['color']
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(xs, pt, ls=ls, color=col, lw=2, marker='o', ms=4,
                    label=f"{'2048' if arm == 'direct' else '4096'} px, {band}")
            for b, v, l_, h_ in zip(bands, pt, lo, hi):
                out_rows.append({'panel': 'agreement', 'arm': arm, 'band': band,
                                 'lat_band_deg': f'{b}..{b + 10}', 'agreement': v,
                                 'boot_lo': l_, 'boot_hi': h_})
    v = next(r['agreement'] for r in out_rows if r.get('arm') == 'direct4096'
             and r.get('band') == 'interior' and r.get('lat_band_deg') == '-80..-70')
    ax.annotate(f'4096 px at the nadir: {v:.2f}\n(the road under the car read as SKY)',
                xy=(-75, v), xytext=(-62, 0.81), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=INK, lw=1))
    ax.axvline(0, color=INK2, lw=1, ls=':')
    ax.set_xlabel('latitude band centre (deg; 0 = horizon, negative = below)')
    ax.set_ylabel('share of pixels where the direct arm\nagrees with the tiled arm (group)')
    ax.set_ylim(0.78, 1.005)
    ax.grid(axis='y', color=GRID_INK, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc='lower right', ncol=2)
    ax.set_title('Agreement with the tiled arm by latitude band (registered)', fontsize=10.5)
    _suptitle(fig, 'Resolution vs projection: the Curb Cut loss was resolution; at matched '
                 'resolution the equirect fails at the nadir', x=0.01, ha='left',
                 fontsize=12.5, fontweight='bold')
    _note(fig, 0.01, 0.87, f'Bands: 95% pano-bootstrap intervals ({BOOT_N} resamples, seed '
             f'{SEED}). Seam = within {SEAM_HALF_WIDTH * 360:.0f} deg of x = 0/1, straight '
             'behind the car (mostly road), so seam vs interior compares content as well as the '
             'wrap.', fontsize=8.5, color=INK2)
    paths = _save(fig, fig_dir, 'fig6_resolution_projection')
    plt.close(fig)
    ex = read_csv(EX_DIR / 'sky.csv')
    fig, axes = plt.subplots(2, 9, figsize=(12, 3.2),
                             gridspec_kw=dict(wspace=0.03, hspace=0.06, left=0.01, right=0.99,
                                              top=0.7, bottom=0.17))
    for i, e in enumerate(ex):
        r, c0 = i // 3, 3 * (i % 3)
        stem = EX_DIR / 'sky' / f"{int(e['idx']):02d}"
        for j, (suf, tag) in enumerate((('crop', None), ('tiled', 'tiled'),
                                         ('d4096', 'direct 4096'))):
            ax = axes[r, c0 + j]
            _panel(ax, f'{stem}_{suf}.jpg')
            if tag:
                _tag(ax, tag)
        _tag(axes[r, c0], f"{float(e['share_low_road_as_sky']):.0%} of road\nbelow -40 deg as SKY")
    _suptitle(fig, 'The projection trap at matched resolution: the 4096 px equirect paints the '
                 'road under the car as SKY', x=0.01, ha='left', fontsize=12.5,
                 fontweight='bold')
    _note(fig, 0.01, 0.82, 'Rule: 6 panos drawn (seed 47) from the 20 with the most pixels below '
             '-40 deg that the tiled arm calls ROAD and the 4096 px arm SKY; crop centred on the '
             'largest such component. The crops show the featureless road under the car: the tiled '
             'arm reads it as ROAD, the 4096 px equirect as SKY.', fontsize=8.5, color=INK2)
    _class_legend(fig, {g: GROUP_COLORS[g] for g in SEG_GROUPS if g != NONE},
                  'segmenter group', 0.0)
    paths += _save(fig, fig_dir, 'fig6b_sky_examples', svg=False, photo=True)
    plt.close(fig)
    return paths, {'fig6_resolution_projection.csv': out_rows}


def tile_outline(heading, pitch, n=64):
    """(x_norm, y_norm) along a tile's border, split where it wraps across the seam."""
    e = np.linspace(0, TILE_SIZE - 1, n)
    border = np.concatenate([np.stack([e, np.zeros(n)], 1), np.stack([np.full(n, TILE_SIZE - 1), e], 1),
                             np.stack([e[::-1], np.full(n, TILE_SIZE - 1)], 1),
                             np.stack([np.zeros(n), e[::-1]], 1)])
    d = tile_pixel_dirs(heading, pitch)[border[:, 1].astype(int), border[:, 0].astype(int)]
    x, y = dirs_to_equirect(d)
    segs, cur = [], [(x[0], y[0])]
    for a, b, c, dd in zip(x[:-1], y[:-1], x[1:], y[1:]):
        if abs(c - a) > 0.5:
            segs.append(cur)
            cur = []
        cur.append((c, dd))
    segs.append(cur)
    return segs


def fig_tiling(plt, fig_dir):
    from matplotlib.colors import ListedColormap
    meta = read_csv(EX_DIR / 'tiling.csv')[0]
    pano = np.asarray(Image.open(EX_DIR / 'tiling' / 'pano.jpg'))
    lab = np.asarray(Image.open(EX_DIR / 'tiling' / 'stitched.png'))
    geom = StitchGeometry()
    fig, axes = plt.subplots(3, 1, figsize=(12, 15.2),
                             gridspec_kw=dict(hspace=0.22, left=0.1, right=0.98, top=0.92,
                                              bottom=0.05))
    ax = axes[0]
    ax.imshow(pano, extent=(0, 1, 1, 0), aspect='auto')
    for name, h, p in tile_specs():
        col = '#2a78d6' if p == 0 else '#eb6834'
        for seg in tile_outline(h, p):
            if len(seg) > 1:
                xs, ys = zip(*seg)
                ax.plot(xs, ys, color='white', lw=3)
                ax.plot(xs, ys, color=col, lw=1.6)
    ax.plot([], [], color='#2a78d6', lw=2, label='8 tiles at pitch 0 deg')
    ax.plot([], [], color='#eb6834', lw=2, label='8 tiles at pitch -35 deg')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    ax.set_title(f"(a) {meta['city']}:{meta['pano_id']} with the 16 perspective tile "
                 'footprints (90 deg FOV, 1024 px each)', fontsize=10.5, loc='left')
    ax = axes[1]
    cov = geom.n_cover
    cmap = ListedColormap(['#e34948'] + [plt.get_cmap('Blues')(0.25 + 0.75 * i / 5)
                                         for i in range(1, 6)])
    ax.imshow(np.minimum(cov, 5), cmap=cmap, vmin=-0.5, vmax=5.5, extent=(0, 1, 1, 0),
              aspect='auto', interpolation='nearest')
    ax.text(0.5, 0.06, f'unseen zenith (above ~+43 deg)', ha='center', color='white',
            fontsize=10, fontweight='bold')
    ax.text(0.5, 0.975, 'unseen nadir (below ~-79 deg, within ~0.5 m of the car)', ha='center',
            color='white', fontsize=10, fontweight='bold', va='bottom')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc='#e34948', label='no tile (NONE)')] +
              [Patch(fc=cmap(i), label=f'{i} tile' + ('s' if i > 1 else '') +
                     (' or more' if i == 5 else '')) for i in range(1, 6)],
              loc='center right', fontsize=8.5, framealpha=0.9)
    ax.set_title('(b) How many tiles see each grid pixel (vote weight); red = no tile',
                 fontsize=10.5, loc='left')
    ax = axes[2]
    # the stitched map holds Vistas ids; VISTAS_V12_ORDER is the pinned model's id order
    table = collapse_table(dict(enumerate(VISTAS_V12_ORDER)))
    rgb = np.stack([_hex_rgb(GROUP_COLORS[SEG_GROUPS[t]]) for t in range(len(SEG_GROUPS))])
    ax.imshow(rgb[table[lab]].astype(np.uint8), extent=(0, 1, 1, 0), aspect='auto',
              interpolation='nearest')
    ax.set_title('(c) Stitched label map on the 1024x512 grid (collapsed groups)', fontsize=10.5,
                 loc='left')
    from matplotlib.patches import Patch as P2
    ax.legend(handles=[P2(fc=GROUP_COLORS[k], label=k) for k in SEG_GROUPS], ncol=7,
              loc='upper center', bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)
    for a in axes:
        a.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0 (seam)', '0.25', '0.5 (heading)', '0.75',
                                               '1 (seam)'])
        a.set_yticks([0, 0.5, 1], ['+90', '0 (horizon)', '-90'])
        a.set_ylabel('latitude (deg)')
    _suptitle(fig, 'The tiling: 16 perspective tiles cover everything from +43 to -79 deg, and '
                 'their votes are stitched back onto the equirect', x=0.1, ha='left',
                 fontsize=12.5, fontweight='bold')
    paths = _save(fig, fig_dir, 'fig7_tiling', svg=False, photo=True)
    plt.close(fig)
    return paths, {}


def fig_gallery(plt, fig_dir):
    ex = read_csv(EX_DIR / 'disagreement.csv')
    nc = GALLERY_PER_CATEGORY
    fig, axes = plt.subplots(len(GALLERY_CATEGORIES), nc, figsize=(12, 8.6),
                             gridspec_kw=dict(wspace=0.03, hspace=0.1, left=0.07, right=0.995,
                                              top=0.86, bottom=0.01))
    for ci, (cat, title) in enumerate(GALLERY_CATEGORIES):
        cp = [e for e in ex if e['category'] == cat]
        for k in range(nc):
            ax = axes[ci, k]
            if k >= len(cp):
                ax.axis('off')
                continue
            e = cp[k]
            _panel(ax, EX_DIR / 'disagreement' / f"{int(e['idx']):02d}_crop.jpg")
            _marker(ax, e['marker_px'])
            v = f" [{e['verdict']}]" if e.get('verdict') else ''
            _tag(ax, f"depth {e['depth']}\nseg {e['seg']}{v}")
            _tag(ax, f"{e['city']}:{e['pano_id'][:11]}", loc='bottom')
            if k == 0:
                ax.set_ylabel(title, fontsize=9)
    _suptitle(fig, 'Where depth and the segmenter disagree: objects on the floor, fences, '
                 'sloped pavement, and peak pixels off the ramp', x=0.07, ha='left',
                 fontsize=12.5, fontweight='bold')
    _note(fig, 0.07, 0.9, 'Rule: per category, 6 panos drawn (seed 47) from the 20 with the '
             'largest within-25 m component; row 4: 6 of the 45 detections >= 0.55 the segmenter '
             'calls non-surface (seed 47).', fontsize=8.5, color=INK2)
    paths = _save(fig, fig_dir, 'fig8_disagreement_gallery', svg=False, photo=True)
    plt.close(fig)
    return paths, {}


def cmd_figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd = pooled_dir(args.out_root)
    fig_dir = args.fig_dir
    data = fig_dir / 'data'
    data.mkdir(parents=True, exist_ok=True)
    # Small aggregates are copied beside the figures; the large per-row tables are tracked
    # ONCE, in runs/_pooled/footway/, and read from there.
    for name in ('agreement.csv', 'headline.csv', 'curb_height.csv', 'trap.csv',
                 'detection_classes.csv', 'verdict.json', 'sample_counts.csv'):
        if (pd / name).exists():
            (data / name).write_bytes((pd / name).read_bytes())
    print(f'figures: reading {pd} and {EX_DIR} -> {fig_dir}')
    _style(plt)
    outputs, tables = [], {}
    for fn in (lambda: fig_base_rate(plt, pd, fig_dir), lambda: fig_objects(plt, fig_dir),
               lambda: fig_matrix(plt, pd, fig_dir), lambda: fig_curb(plt, pd, data, fig_dir),
               lambda: fig_fp(plt, pd, fig_dir), lambda: fig_trap(plt, pd, fig_dir),
               lambda: fig_tiling(plt, fig_dir), lambda: fig_gallery(plt, fig_dir)):
        paths, t = fn()
        outputs += paths
        tables.update(t)
    for name, rows in tables.items():
        write_csv(data / name, rows, list(dict.fromkeys(k for r in rows for k in r)))
    write_csv(data / 'derived_numbers.csv', derived_numbers(pd))
    nums = check_numbers(pd, data)
    write_csv(data / 'numbers.csv', nums)
    bad = [n for n in nums if n['status'] != 'ok']
    man = {str(p.relative_to(fig_dir)).replace('\\', '/'): sha256_file(p)
           for p in sorted(outputs)}
    write_text(data / 'figures_manifest.json', json.dumps(man, indent=1, sort_keys=True))
    print(f'figures: {len(outputs)} files; numbers checked {len(nums)}, mismatched {len(bad)}')
    for b in bad:
        print('  MISMATCH', b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['sample', 'tiles', 'segment', 'stitch', 'compare',
                                        'verdict', 'examples', 'figures'])
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
    ap.add_argument('--fig-dir', type=Path, default=FIG_DIR,
                    help='figures: output dir (default docs/figures/footway-depth)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--workers', type=int, default=6, help='tiles: worker processes')
    ap.add_argument('--direct-width', type=int, default=DIRECT_W,
                    help='tiles: render ONLY a direct-arm equirect of this width into '
                         'work/direct<W>/ (the registered control arm is 2048)')
    ap.add_argument('--note', action='append', default=[],
                    help='segment: a provenance note recorded in segment_manifest.json')
    ap.add_argument('--limit', type=int, default=0, help='tiles: first N sampled panos')
    args = ap.parse_args()
    {'sample': cmd_sample, 'tiles': cmd_tiles, 'segment': cmd_segment,
     'stitch': cmd_stitch, 'compare': cmd_compare, 'verdict': cmd_verdict,
     'examples': cmd_examples, 'figures': cmd_figures}[args.command](args)


if __name__ == '__main__':
    main()
