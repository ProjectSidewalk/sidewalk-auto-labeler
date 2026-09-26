"""Depth at the detection (issue #47, step 1): does GSV's harvested depth alone locate a
curb ramp's surface?

A STUDY, not production: nothing here is wired into main.py, geo.py, fuse_sites.py or
sources/. It needs no network and no GPU. Inputs are the four harvested GSV runs'
``results.jsonl`` + ``depth/<pano_id>.json.gz`` payloads (scripts/harvest_depth.py) and
RampNet's benchmark bundles, all read in place and never rewritten.

For every stored detection (at the operating point, 0.30, and the benchmark tier, 0.55)
on a pano with a payload, read the depth plane under the detection's pixel and ask:

  - which KIND of plane is it (`plane_class`, below)?
  - for a near-horizontal floor plane that is not the dominant ground: how far above the
    road beside it does it sit (`offset_local_m`; the "~0.15 m above the modelled road"
    claim in sidewalk-panorama-tools' docs/depth.md)?
  - what horizontal range does the plane give, against the flat-ground raycast at 2.6 m
    and at the pano's own depth-measured height (a third range instrument beside
    triangulation, docs/camera-height-study.md, and the reprojection range-scale fit)?

Frames. Detections are stored in the IMAGE frame (x from the left of the panorama JPEG).
A raw depth-index column IS an image column (#80), so the plane lookup is
``depth._plane_at(payload, x, y)`` -- never ``depth.depth_at``, which takes streetlevel's
mirrored raster frame and snaps the ray to a pixel centre (worth up to 6.6% of range near
the horizon). Only the plane LOOKUP is quantized (the segmentation is per pixel); the hit
point is the exact ray ``depth._direction_continuous(x, y)`` intersected with that plane,
exactly as ``depth.ground_range_at`` computes the range. Depth frame: +z is DOWN.

Classes (``classify``), in the order they are tested:

    no_plane             index 0 (sky / unreconstructed) at the pixel
    ground               the pixel's plane IS the dominant ground plane
                         (gsv_ground_plane.dominant_ground, cross-checked against
                         depth.ground_plane)
    floor                another plane meeting ground_plane's candidate rule: tilt <=
                         GROUND_MAX_TILT_DEG (18) and >= 90% of its pixels below the horizon
    horizontal_nonfloor  tilt <= 18 deg but fails the below-horizon share (an overpass or
                         awning underside)
    non_horizontal       tilt > 18 deg (a wall reads ~90 deg; the tilt is reported)

Subcommands. Per-city CSVs go to ``<out-root>/<city>/depth_at_detection/`` (untracked,
like ground_plane/), the aggregates to ``<out-root>/_summary/depth_at_detection/``, and
those aggregates are committed under ``docs/figures/depth-at-detection/data/``, which
``verdict`` and ``figures`` fall back to:

    python scripts/depth_at_detection.py measure [cities] [--limit N] [--workers N]
    python scripts/depth_at_detection.py gt      [cities] [--benchmark-root ../RampNet/benchmark]
    python scripts/depth_at_detection.py verdict
    python scripts/depth_at_detection.py figures

``measure`` must run first; ``gt`` reads its detections.csv. ``--limit N`` reads the first
N detection-bearing panos per city and writes no summaries (a smoke test).

Stdlib on top of depth.py / geo.py / fuse_sites.py / eval_sites.py / gsv_ground_plane.py /
mapillary_tilt.py, except ``figures`` (matplotlib).
"""
import argparse
import csv
import gzip
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same order tests/conftest.py forces: the repo root FIRST, because
# scripts/position_check.py is a shim sharing the root module's name.
for p in (str(REPO_ROOT / 'scripts'), str(REPO_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import depth as depthlib  # noqa: E402
import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import gsv_ground_plane as gp  # noqa: E402
import mapillary_tilt as mt  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE  # noqa: E402

DEFAULT_CITIES = ['bend', 'paterson', 'gainesville', 'sao_paulo']
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'depth-at-detection'
TIERS = (('0.30', OPERATIONAL_CONFIDENCE), ('0.55', BENCHMARK_CONFIDENCE))

NO_PLANE, GROUND, FLOOR = 'no_plane', 'ground', 'floor'
HORIZONTAL_NONFLOOR, NON_HORIZONTAL = 'horizontal_nonfloor', 'non_horizontal'
CLASSES = [GROUND, FLOOR, HORIZONTAL_NONFLOOR, NON_HORIZONTAL, NO_PLANE]
NO_FILE = 'no_file'   # payload status when the pano has no harvested payload

# --- the pre-registered reading (plan comment on #47; fixed before the full run) --------
#
# (i) Depth carries the surface. `surface` = `ground`, or `floor` with
#     SURFACE_BAND_M[0] <= offset_local < SURFACE_BAND_M[1]. SUPPORTED iff the pooled
#     surface share of verdict-True detections is >= RULE_I_SUPPORT_POOLED and >=
#     RULE_I_SUPPORT_CITY in every city; UNDERCUT iff pooled < RULE_I_UNDERCUT; otherwise
#     NOT SUPPORTED. The ~0.15 m claim is read descriptively: the median offset_local of
#     True `floor` detections (with its CI) is "consistent" if it lies in CLAIM_BAND_M.
# (ii) A free false-positive signal. `non_surface` = everything else. SUPPORTED iff the
#     pooled non_surface share among verdict-False detections exceeds the True share by
#     >= RULE_II_MARGIN AND the False share's Wilson lower bound exceeds the True share's
#     upper bound; otherwise NOT SUPPORTED. `underpowered` is true when the False share's
#     Wilson 95% interval is wider than RULE_II_MARGIN: a null is then weak evidence.
# Only `measured` payloads enter either rule; the benchmark tier only (GT was judged there).
#
# Clarification made while implementing, before any GT-conditioned number was computed:
# a `floor` detection whose column walk finds no local reference plane has no
# offset_local, so it is not inside the band and counts as non_surface (the plan's probe
# found 2 such of 2,478 floor hits).
SURFACE_BAND_M = (-0.30, 0.30)
CLAIM_BAND_M = (0.05, 0.30)
RULE_I_SUPPORT_POOLED = 0.85
RULE_I_SUPPORT_CITY = 0.80
RULE_I_UNDERCUT = 0.70
RULE_II_MARGIN = 0.20

# Descriptive bins, fixed by the plan.
OFFSET_BINS = [(-math.inf, -0.30), (-0.30, -0.05), (-0.05, 0.05), (0.05, 0.30),
               (0.30, math.inf)]
RANGE_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, math.inf)]
HORIZON_BIN = 'horizon'
VINTAGE_MIN_N = 300

DET_FIELDS = ['pano_id', 'det_index', 'x', 'y', 'confidence', 'capture_year',
              'payload_status', 'plane_class', 'plane_tilt_deg', 'plane_pixel_share',
              'offset_local_m', 'offset_dominant_m', 'rows_to_ref', 'range_depth_m',
              'range_flat_2p6_m', 'range_flat_pp_m', 'camera_height_pp_m', 'beyond_25m',
              'no_plane_3x3']
_STR_FIELDS = {'pano_id', 'capture_year', 'payload_status', 'plane_class'}
_INT_FIELDS = {'det_index', 'rows_to_ref', 'beyond_25m', 'no_plane_3x3'}


# --- geometry on one payload (stdlib; tests/test_depth_at_detection.py) -----------------

def plane_tilt_deg(plane):
    """Angle between a plane's normal and vertical, degrees (depth.ground_plane's)."""
    return math.degrees(math.acos(min(1.0, abs(plane.nz))))


class PayloadIndex:
    """The per-payload facts every detection on it needs, computed once: pixel counts,
    the floor-candidate set (depth.ground_plane's rule) and the dominant ground plane's
    index, cross-checked against depth.ground_plane so the copies cannot drift."""

    def __init__(self, payload):
        self.payload = payload
        w, h = payload.width, payload.height
        self.counts = Counter(payload.indices)
        below = Counter(payload.indices[(h // 2) * w:])
        self.floor = set()
        for idx, count in self.counts.items():
            if idx == depthlib.SKY or idx >= len(payload.planes):
                continue
            p = payload.planes[idx]
            if (plane_tilt_deg(p) <= depthlib.GROUND_MAX_TILT_DEG
                    and below[idx] / count >= depthlib.GROUND_MIN_BELOW_HORIZON):
                self.floor.add(idx)
        self.ground_info = depthlib.ground_plane(payload)
        self.dominant = None
        dom = gp.dominant_ground(payload)
        if dom is not None:
            self.dominant = next(i for i, p in enumerate(payload.planes) if p is dom)
            if (self.ground_info is None or dom.d != self.ground_info.camera_height_m
                    or abs(plane_tilt_deg(dom) - self.ground_info.tilt_deg) > 1e-9):
                raise RuntimeError('dominant_ground disagrees with depth.ground_plane '
                                   '-- the copy drifted')

    def status(self):
        """depth.classify_height's status for this payload."""
        if self.payload.degenerate:
            return depthlib.DEGENERATE
        g = self.ground_info
        if g is None:
            return depthlib.NO_GROUND
        return depthlib.classify_height(g.camera_height_m, g.tilt_deg,
                                        exactly_level=g.exactly_level)

    def is_valid(self, idx):
        return idx != depthlib.SKY and idx < len(self.payload.planes)


def plane_class(pix, idx):
    """The class of plane index `idx` on PayloadIndex `pix` (module docstring)."""
    if not pix.is_valid(idx):
        return NO_PLANE
    if idx == pix.dominant:
        return GROUND
    if idx in pix.floor:
        return FLOOR
    if plane_tilt_deg(pix.payload.planes[idx]) <= depthlib.GROUND_MAX_TILT_DEG:
        return HORIZONTAL_NONFLOOR
    return NON_HORIZONTAL


def floor_z(plane, px, py):
    """z (depth frame, +down) of a floor plane at horizontal position (px, py).

    A floor plane (tilt <= 18 deg, below the horizon) lies BELOW the camera, so of the two
    planes n.X = +-d the one meant is the one crossing the vertical under the camera at
    positive z: n.X = d * sign(nz).
    """
    s = plane.d * math.copysign(1.0, plane.nz)
    return (s - plane.nx * px - plane.ny * py) / plane.nz


def floor_hit(plane, x_norm, y_norm):
    """Exact-ray hit point on a floor plane at an image-frame coordinate, or None when the
    ray does not meet the plane in front of the camera."""
    v = depthlib._direction_continuous(x_norm, y_norm)
    denom = v[0] * plane.nx + v[1] * plane.ny + v[2] * plane.nz
    if denom == 0:
        return None
    t = plane.d * math.copysign(1.0, plane.nz) / denom
    if t <= 0:
        return None
    return (v[0] * t, v[1] * t, v[2] * t)


def vertical_offset(plane, ref, x_norm, y_norm):
    """Height of `plane` above `ref` (metres, positive = above) at the point where the
    exact ray through (x_norm, y_norm) meets `plane`: z_ref(P_xy) - P_z. Both are floor
    planes. None if the ray misses `plane` or `ref` is vertical."""
    if ref.nz == 0:
        return None
    hit = floor_hit(plane, x_norm, y_norm)
    if hit is None:
        return None
    return floor_z(ref, hit[0], hit[1]) - hit[2]


def local_reference(pix, row, col, own_idx):
    """(ref_idx, rows_walked) for the first floor-candidate plane DIFFERENT from `own_idx`
    met walking raw column `col` down from `row` toward the nadir (the road in front of
    the ramp), or (None, None) when the column has none."""
    payload = pix.payload
    w = payload.width
    for r in range(row + 1, payload.height):
        j = payload.indices[r * w + col]
        if j != own_idx and j in pix.floor:
            return j, r - row
    return None, None


def no_plane_near(pix, row, col):
    """1 if any pixel of the 3x3 payload neighbourhood (rows clamped, columns wrapping
    round the seam) has no plane, else 0."""
    payload = pix.payload
    w, h = payload.width, payload.height
    for r in range(max(0, row - 1), min(h, row + 2)):
        for dc in (-1, 0, 1):
            if not pix.is_valid(payload.indices[r * w + (col + dc) % w]):
                return 1
    return 0


def classify(pix, x_norm, y_norm):
    """The depth-derived fields of one image-frame coordinate on PayloadIndex `pix`."""
    payload = pix.payload
    plane, row, col = depthlib._plane_at(payload, x_norm, y_norm)
    idx = payload.indices[row * payload.width + col]
    out = {'plane_class': plane_class(pix, idx), 'plane_tilt_deg': None,
           'plane_pixel_share': None, 'offset_local_m': None, 'offset_dominant_m': None,
           'rows_to_ref': None, 'range_depth_m': None,
           'no_plane_3x3': no_plane_near(pix, row, col)}
    if plane is None:
        return out
    out['plane_tilt_deg'] = plane_tilt_deg(plane)
    out['plane_pixel_share'] = pix.counts[idx] / len(payload.indices)
    # Horizontal range exactly as depth.ground_range_at computes it (same lookup).
    d = depthlib._intersect(plane, depthlib._direction_continuous(x_norm, y_norm))
    if d is not None:
        out['range_depth_m'] = d * math.cos((0.5 - y_norm) * math.pi)
    if out['plane_class'] == FLOOR:
        ref_idx, walked = local_reference(pix, row, col, idx)
        if ref_idx is not None:
            out['offset_local_m'] = vertical_offset(plane, payload.planes[ref_idx],
                                                    x_norm, y_norm)
            out['rows_to_ref'] = walked
        if pix.dominant is not None:
            out['offset_dominant_m'] = vertical_offset(
                plane, payload.planes[pix.dominant], x_norm, y_norm)
    return out


def read_payload(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return depthlib.parse(json.load(f)['depth_b64'])


def measure_pano(task):
    """Worker: (payload_path, [(key, x, y), ...]) -> (status, [(key, fields), ...]).

    Module-level so ProcessPoolExecutor can pickle it on Windows. A payload that does not
    parse is recorded as UNPARSED, never raised."""
    path_str, points = task
    try:
        pix = PayloadIndex(read_payload(path_str))
    except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError):
        return depthlib.UNPARSED, [(k, {}) for k, _, _ in points]
    return pix.status(), [(k, classify(pix, x, y)) for k, x, y in points]


# --- aggregation (pure) -----------------------------------------------------------------

def is_surface(row):
    """The pre-registered `surface` test: ground, or floor within SURFACE_BAND_M of the
    local road."""
    if row['plane_class'] == GROUND:
        return True
    off = row.get('offset_local_m')
    return (row['plane_class'] == FLOOR and off is not None
            and SURFACE_BAND_M[0] <= off < SURFACE_BAND_M[1])


def offset_bin(v):
    for lo, hi in OFFSET_BINS:
        if lo <= v < hi:
            return bin_label(lo, hi)
    return None


def bin_label(lo, hi):
    if lo == -math.inf:
        return f'<{hi:g}'
    if hi == math.inf:
        return f'>={lo:g}'
    return f'[{lo:g},{hi:g})'


OFFSET_LABELS = [bin_label(lo, hi) for lo, hi in OFFSET_BINS]


def range_bin(row):
    r = row.get('range_flat_2p6_m')
    if r is None:
        return HORIZON_BIN
    for lo, hi in RANGE_BINS:
        if lo <= r < hi:
            return f'{lo}+' if hi == math.inf else f'{lo}-{hi}'
    return HORIZON_BIN


RANGE_LABELS = [f'{lo}+' if hi == math.inf else f'{lo}-{hi}' for lo, hi in RANGE_BINS] \
    + [HORIZON_BIN]


def median_ci(values, z=1.96):
    """(median, lo, hi): the sample median with a distribution-free 95% interval from
    binomial order statistics. (None, None, None) for an empty list."""
    v = sorted(values)
    n = len(v)
    if not n:
        return None, None, None
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    half = z * math.sqrt(n) / 2
    lo = max(0, int(math.floor(n / 2 - half)) - 1)
    hi = min(n - 1, int(math.ceil(n / 2 + half)))
    return med, v[lo], v[hi]


def class_counts(rows):
    c = Counter(r['plane_class'] for r in rows)
    n = len(rows)
    out = {'n': n}
    for k in CLASSES:
        out[f'n_{k}'] = c[k]
        out[f'share_{k}'] = c[k] / n if n else None
    k = sum(is_surface(r) for r in rows)
    out['n_surface'] = k
    out['share_surface'] = k / n if n else None
    return out


def ratio(a, b):
    return a / b if a is not None and b else None


def verdict(gt, offset_median=None):
    """Apply the pre-registered reading.

    `gt` = {city: {'true_n', 'true_surface', 'false_n', 'false_non_surface'}}, measured
    payloads, benchmark tier. `offset_median` = (median, lo, hi) of offset_local over
    True floor detections, or None. Returns a dict with rule_i, rule_ii and the numbers.

    Example:
        >>> v = verdict({'a': {'true_n': 100, 'true_surface': 90,
        ...                    'false_n': 10, 'false_non_surface': 2}})
        >>> v['rule_i'], v['rule_ii'], v['underpowered']
        ('SUPPORTED', 'NOT SUPPORTED', True)
    """
    tn = sum(g['true_n'] for g in gt.values())
    ts = sum(g['true_surface'] for g in gt.values())
    fn = sum(g['false_n'] for g in gt.values())
    fns = sum(g['false_non_surface'] for g in gt.values())
    pooled = ts / tn if tn else None
    per_city = {c: (g['true_surface'] / g['true_n'] if g['true_n'] else None)
                for c, g in gt.items()}
    detail = {'true_surface_share_pooled': pooled, 'true_n_pooled': tn,
              'true_surface_share_by_city': per_city}
    if pooled is not None and pooled < RULE_I_UNDERCUT:
        rule_i = 'UNDERCUT'
    elif (pooled is not None and pooled >= RULE_I_SUPPORT_POOLED
          and all(s is not None and s >= RULE_I_SUPPORT_CITY for s in per_city.values())):
        rule_i = 'SUPPORTED'
    else:
        rule_i = 'NOT SUPPORTED'

    true_ns = (tn - ts) / tn if tn else None
    false_ns = fns / fn if fn else None
    t_lo, t_hi = es.wilson(tn - ts, tn)
    f_lo, f_hi = es.wilson(fns, fn)
    detail.update({'true_non_surface_share': true_ns, 'true_non_surface_ci': [t_lo, t_hi],
                   'false_non_surface_share': false_ns, 'false_n_pooled': fn,
                   'false_non_surface_ci': [f_lo, f_hi]})
    # rounded so a difference of exactly 20 points is not lost to float subtraction
    diff = None if true_ns is None or false_ns is None else round(false_ns - true_ns, 12)
    detail['non_surface_diff'] = diff
    rule_ii = ('SUPPORTED' if diff is not None and diff >= RULE_II_MARGIN and f_lo > t_hi
               else 'NOT SUPPORTED')
    underpowered = (f_hi - f_lo) > RULE_II_MARGIN

    claim = None
    if offset_median is not None and offset_median[0] is not None:
        claim = ('consistent' if CLAIM_BAND_M[0] <= offset_median[0] < CLAIM_BAND_M[1]
                 else 'not consistent')
        detail['true_floor_offset_local_median'] = list(offset_median)
    detail['offset_claim_0p15'] = claim
    return {'rule_i': rule_i, 'rule_ii': rule_ii, 'underpowered': underpowered, **detail}


# --- IO ---------------------------------------------------------------------------------

def city_dir(out_root, city):
    d = out_root / city / 'depth_at_detection'
    d.mkdir(parents=True, exist_ok=True)
    return d


def summary_dir(out_root):
    d = out_root / '_summary' / 'depth_at_detection'
    d.mkdir(parents=True, exist_ok=True)
    return d


# Written at full precision: a confidence rounded to 0.55 would move a detection between
# tiers, and x/y are the stored coordinates.
_EXACT_FIELDS = {'x', 'y', 'confidence'}


def _fmt(k, v):
    if v is None:
        return ''
    if isinstance(v, float) and k not in _EXACT_FIELDS:
        return f'{v:.6g}'
    return v


def write_rows(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(k, r.get(k)) for k in fields})


def read_detections(path):
    """detections.csv with explicit types (pano ids must never be float-parsed)."""
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v == '':
                    row[k] = None
                elif k in _STR_FIELDS:
                    row[k] = v
                elif k in _INT_FIELDS:
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            out.append(row)
    return out


def summary_csv(out_root, name):
    """An aggregate from the run tree, else the committed copy."""
    for base in (out_root / '_summary' / 'depth_at_detection', FIG_DIR / 'data'):
        if (base / name).exists():
            return mt._read_csv(base / name)
    raise SystemExit(f'{name} not found in the run tree or {FIG_DIR / "data"}; '
                     f'run `measure` / `gt` first')


# --- measure ----------------------------------------------------------------------------

def cmd_measure(args):
    coverage, city_rows = [], {}
    for city in args.cities:
        run_dir = args.run_root / city
        panos, _ = fs.load_results(run_dir / 'results.jsonl')
        panos = [p for p in panos
                 if any(c >= OPERATIONAL_CONFIDENCE for _, _, _, c in p.detections)]
        if args.limit:
            panos = panos[:args.limit]
        tasks, no_file = [], []
        for p in panos:
            path = run_dir / 'depth' / f'{p.pano_id}.json.gz'
            pts = [(i, x, y) for i, x, y, c in p.detections if c >= OPERATIONAL_CONFIDENCE]
            (tasks if path.exists() else no_file).append((str(path), pts, p))
        print(f'{city}: {len(panos):,} panos with a detection >= {OPERATIONAL_CONFIDENCE}, '
              f'{len(tasks):,} with a payload', flush=True)
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, (task, res) in enumerate(zip(tasks, ex.map(
                    measure_pano, [(t[0], t[1]) for t in tasks], chunksize=200)), 1):
                results[task[2].pano_id] = res
                if n % 5000 == 0:
                    print(f'  {city}: {n:,}/{len(tasks):,}', flush=True)
        rows = []
        for p in panos:
            status, fields = results.get(p.pano_id, (NO_FILE, None))
            fields = dict(fields or [])
            pose = fs.pano_pose(p, fs.POSE_OFF)
            for i, x, y, c in p.detections:
                if c < OPERATIONAL_CONFIDENCE:
                    continue
                g = geo.detection_ground_point(pose, x, y, max_range_m=math.inf,
                                               apply_pose=False)
                h = p.camera_height_m
                gpp = (geo.detection_ground_point(pose, x, y, camera_height=h,
                                                  max_range_m=math.inf, apply_pose=False)
                       if h is not None else None)
                row = {'pano_id': p.pano_id, 'det_index': i, 'x': x, 'y': y,
                       'confidence': c, 'capture_year': (p.capture_date or '')[:4] or None,
                       'payload_status': status, 'plane_class': None,
                       'range_flat_2p6_m': g.range_m if g else None,
                       'range_flat_pp_m': gpp.range_m if gpp else None,
                       'camera_height_pp_m': h,
                       'beyond_25m': int(g is None or g.range_m > geo.DEFAULT_MAX_RANGE_M)}
                row.update(fields.get(i, {}))
                rows.append(row)
        write_rows(city_dir(args.out_root, city) / 'detections.csv', rows, DET_FIELDS)
        city_rows[city] = rows
        st = Counter(r['payload_status'] for r in rows)
        pst = Counter(res[0] for res in results.values())
        coverage.append({'city': city, 'n_panos_with_detection': len(panos),
                         'n_panos_with_payload': len(tasks),
                         **{f'panos_{k}': v for k, v in sorted(pst.items())},
                         'n_detections_0p30': len(rows),
                         'n_detections_0p55': sum(r['confidence'] >= BENCHMARK_CONFIDENCE
                                                  for r in rows),
                         **{f'dets_{k}': v for k, v in sorted(st.items())}})
        meas = [r for r in rows if r['payload_status'] == depthlib.MEASURED]
        cc = class_counts(meas)
        print(f'{city}: {len(rows):,} detections; status {dict(st)}; measured-only classes '
              + ', '.join(f"{k} {cc[f'share_{k}']:.3f}" for k in CLASSES if cc['n']),
              flush=True)
    if not args.limit:
        write_summaries(args.out_root, city_rows, coverage)


def _groups(city_rows):
    """(label, rows) for every city and the pool."""
    for city, rows in city_rows.items():
        yield city, rows
    yield 'pooled', [r for rows in city_rows.values() for r in rows]


def write_summaries(out_root, city_rows, coverage):
    sd = summary_dir(out_root)
    fields = sorted({k for c in coverage for k in c}, key=lambda k: (k != 'city', k))
    write_rows(sd / 'coverage.csv', coverage, fields)
    shares, by_year, offs, ratios, noplane, hist = [], [], [], [], [], []
    for tier, floor in TIERS:
        for label, rows in _groups(city_rows):
            rows = [r for r in rows if r['confidence'] >= floor]
            by_status = defaultdict(list)
            for r in rows:
                by_status[r['payload_status']].append(r)
            for status in [depthlib.MEASURED] + sorted(s for s in by_status
                                                       if s != depthlib.MEASURED):
                sr = by_status.get(status, [])
                tilts = [r['plane_tilt_deg'] for r in sr if r['plane_class'] == NON_HORIZONTAL]
                shares.append({'tier': tier, 'city': label, 'payload_status': status,
                               **(class_counts(sr) if status != NO_FILE else {'n': len(sr)}),
                               'non_horizontal_tilt_p50': mt.pct(tilts, .5)})
            meas = by_status.get(depthlib.MEASURED, [])
            # offsets of `floor` detections, local reference (primary) and dominant
            fl = [r for r in meas if r['plane_class'] == FLOOR]
            for ref in ('local', 'dominant'):
                vals = [r[f'offset_{ref}_m'] for r in fl if r[f'offset_{ref}_m'] is not None]
                bins = Counter(offset_bin(v) for v in vals)
                med, lo, hi = median_ci(vals)
                offs.append({'tier': tier, 'city': label, 'reference': ref,
                             'n_floor': len(fl), 'n_with_ref': len(vals),
                             **{f'n_{b}': bins[b] for b in OFFSET_LABELS},
                             'p10': mt.pct(vals, .1), 'p25': mt.pct(vals, .25),
                             'median': med, 'median_lo': lo, 'median_hi': hi,
                             'p75': mt.pct(vals, .75), 'p90': mt.pct(vals, .9),
                             'rows_to_ref_p50': mt.pct([r['rows_to_ref'] for r in fl
                                                        if r['rows_to_ref'] is not None], .5)
                             if ref == 'local' else None})
                for v in vals:
                    b = math.floor(max(-1.0, min(0.999, v)) / 0.05)
                    hist.append((tier, label, ref, b))
            for cls in (GROUND, FLOOR):
                cr = [r for r in meas if r['plane_class'] == cls]
                ok = [r for r in cr if not r['beyond_25m']]
                rf = [ratio(r['range_depth_m'], r['range_flat_2p6_m']) for r in ok]
                rf = [v for v in rf if v is not None]
                rp = [ratio(r['range_depth_m'], r['range_flat_pp_m']) for r in ok]
                rp = [v for v in rp if v is not None]
                ratios.append({'tier': tier, 'city': label, 'plane_class': cls, 'n': len(cr),
                               'n_beyond_25m': len(cr) - len(ok),
                               'n_ratio_flat': len(rf), 'ratio_flat_p10': mt.pct(rf, .1),
                               'ratio_flat_p50': mt.pct(rf, .5), 'ratio_flat_p90': mt.pct(rf, .9),
                               'n_ratio_pp': len(rp), 'ratio_pp_p10': mt.pct(rp, .1),
                               'ratio_pp_p50': mt.pct(rp, .5), 'ratio_pp_p90': mt.pct(rp, .9)})
            bins = defaultdict(list)
            for r in meas:
                bins[range_bin(r)].append(r)
            for b in RANGE_LABELS:
                br = bins.get(b, [])
                noplane.append({'tier': tier, 'city': label, 'range_bin': b, 'n': len(br),
                                'n_no_plane': sum(r['plane_class'] == NO_PLANE for r in br),
                                'n_no_plane_3x3': sum(r['no_plane_3x3'] or 0 for r in br),
                                'rate_no_plane': (sum(r['plane_class'] == NO_PLANE for r in br)
                                                  / len(br)) if br else None,
                                'rate_no_plane_3x3': (sum(r['no_plane_3x3'] or 0 for r in br)
                                                      / len(br)) if br else None})
            if label == 'pooled':
                continue
            years = defaultdict(list)
            for r in meas:
                years[r['capture_year'] or '????'].append(r)
            for y, yr in sorted(years.items()):
                if len(yr) < VINTAGE_MIN_N:
                    continue
                ok = [r for r in yr if r['plane_class'] in (GROUND, FLOOR)
                      and not r['beyond_25m']]
                rf = [v for v in (ratio(r['range_depth_m'], r['range_flat_2p6_m'])
                                  for r in ok) if v is not None]
                rp = [v for v in (ratio(r['range_depth_m'], r['range_flat_pp_m'])
                                  for r in ok) if v is not None]
                by_year.append({'tier': tier, 'city': label, 'year': y, **class_counts(yr),
                                'ratio_flat_p50': mt.pct(rf, .5),
                                'ratio_pp_p50': mt.pct(rp, .5),
                                'camera_height_pp_p50': mt.pct(
                                    [r['camera_height_pp_m'] for r in yr
                                     if r['camera_height_pp_m'] is not None], .5)})
    share_fields = ['tier', 'city', 'payload_status', 'n'] \
        + [f'n_{k}' for k in CLASSES] + [f'share_{k}' for k in CLASSES] \
        + ['n_surface', 'share_surface', 'non_horizontal_tilt_p50']
    write_rows(sd / 'class_shares.csv', shares, share_fields)
    write_rows(sd / 'class_by_year.csv', by_year, list(by_year[0]) if by_year else ['tier'])
    write_rows(sd / 'offset_bins.csv', offs, list(offs[0]))
    write_rows(sd / 'range_ratio.csv', ratios, list(ratios[0]))
    write_rows(sd / 'no_plane_by_range.csv', noplane, list(noplane[0]))
    hc = Counter(hist)
    write_rows(sd / 'offset_hist.csv',
               [{'tier': t, 'city': c, 'reference': ref, 'bin_lo_m': round(b * 0.05, 2),
                 'n': n} for (t, c, ref, b), n in sorted(hc.items())],
               ['tier', 'city', 'reference', 'bin_lo_m', 'n'])
    print(f'summaries -> {sd}')


# --- gt ---------------------------------------------------------------------------------

GT_GROUPS = ['true', 'false', 'missed', 'unsure', 'duplicate', 'unsure_missed']
_VERDICT_GROUP = {True: 'true', False: 'false', 'unsure': 'unsure', 'duplicate': 'duplicate'}


def cmd_gt(args):
    """Join the benchmark tier to RampNet's reviewed verdicts and missed marks, with the
    same drift gate and skip rules as eval_sites / mined_precision."""
    out, counts_rows, per_city = [], [], {}
    for city in args.cities:
        run_dir = args.run_root / city
        det_path = args.out_root / city / 'depth_at_detection' / 'detections.csv'
        dets = {(r['pano_id'], r['det_index']): r for r in read_detections(det_path)}
        verdicts, bundle_ops, run_panos = es.load_city_files(city, args.benchmark_root,
                                                             run_dir, read_heights=False)
        by_id = {p.pano_id: p for p in run_panos}
        counts, warnings = es.gt_counts(), []
        counts['gt_panos'] = len(verdicts)
        items, missed_tasks = [], defaultdict(list)
        for pid, entry, _run_pano, ops, in_pool in es.judged_gt_panos(
                verdicts, bundle_ops, by_id, counts, warnings):
            for v, (stored_i, x, y, c) in zip(entry['dets'], ops):
                r = dets.get((pid, stored_i))
                if r is None:
                    raise SystemExit(f'{city}:{pid} detection {stored_i} is not in '
                                     f'{det_path} -- re-run `measure`')
                items.append({**r, 'city': city, 'gt_group': _VERDICT_GROUP[v],
                              'in_pool': int(bool(in_pool))})
            for k, mark in enumerate(entry.get('missed', ())):
                missed_tasks[pid].append((k, mark['x'], mark['y'],
                                          'unsure_missed' if mark.get('unsure') else 'missed',
                                          int(bool(in_pool))))
        for pid, marks in missed_tasks.items():
            path = run_dir / 'depth' / f'{pid}.json.gz'
            if path.exists():
                status, fields = measure_pano((str(path), [(k, x, y) for k, x, y, _, _
                                                           in marks]))
                fields = dict(fields)
            else:
                status, fields = NO_FILE, {}
            for k, x, y, group, in_pool in marks:
                items.append({'city': city, 'pano_id': pid, 'det_index': k, 'x': x, 'y': y,
                              'confidence': None, 'payload_status': status,
                              'plane_class': None, 'gt_group': group, 'in_pool': in_pool,
                              **fields.get(k, {})})
        write_rows(city_dir(args.out_root, city) / 'gt_rows.csv', items,
                   ['city', 'gt_group', 'in_pool'] + DET_FIELDS)
        per_city[city] = items
        counts_rows.append({'city': city, **counts, 'n_warnings': len(warnings)})
        print(f'{city}: {counts} ({len(warnings)} warnings); '
              + ', '.join(f'{g} {sum(i["gt_group"] == g for i in items)}' for g in GT_GROUPS))
    rows = []
    for label, items in _groups(per_city):
        for group in GT_GROUPS:
            gi = [i for i in items if i['gt_group'] == group]
            for subset, si in (('measured', [i for i in gi
                                             if i['payload_status'] == depthlib.MEASURED]),
                               ('not_measured', [i for i in gi
                                                 if i['payload_status'] != depthlib.MEASURED])):
                cc = class_counts(si)
                lo, hi = es.wilson(cc['n_surface'], cc['n'])
                fl = [i['offset_local_m'] for i in si
                      if i['plane_class'] == FLOOR and i['offset_local_m'] is not None]
                bins = Counter(offset_bin(v) for v in fl)
                med, mlo, mhi = median_ci(fl)
                rows.append({'city': label, 'gt_group': group, 'subset': subset, **cc,
                             'n_non_surface': cc['n'] - cc['n_surface'],
                             'surface_lo': lo, 'surface_hi': hi,
                             'n_floor_with_ref': len(fl),
                             **{f'offset_n_{b}': bins[b] for b in OFFSET_LABELS},
                             'offset_local_median': med, 'offset_local_median_lo': mlo,
                             'offset_local_median_hi': mhi,
                             'statuses': ';'.join(f'{k}={v}' for k, v in sorted(Counter(
                                 i['payload_status'] for i in si).items()))})
    sd = summary_dir(args.out_root)
    write_rows(sd / 'gt_classes.csv', rows, list(rows[0]))
    write_rows(sd / 'gt_counts.csv', counts_rows, list(counts_rows[0]))
    print(f'gt -> {sd}')


# --- verdict ----------------------------------------------------------------------------

def cmd_verdict(args):
    rows = summary_csv(args.out_root, 'gt_classes.csv')
    meas = [r for r in rows if r['subset'] == 'measured']
    gt = {}
    for c in args.cities:
        t = next(r for r in meas if r['city'] == c and r['gt_group'] == 'true')
        f = next(r for r in meas if r['city'] == c and r['gt_group'] == 'false')
        gt[c] = {'true_n': int(t['n']), 'true_surface': int(t['n_surface']),
                 'false_n': int(f['n']), 'false_non_surface': int(f['n_non_surface'])}
    pt = next(r for r in meas if r['city'] == 'pooled' and r['gt_group'] == 'true')
    med = (pt['offset_local_median'], pt['offset_local_median_lo'],
           pt['offset_local_median_hi'])
    v = verdict(gt, med)
    v['per_city_counts'] = gt
    print(json.dumps(v, indent=2))
    with open(summary_dir(args.out_root) / 'verdict.json', 'w', encoding='utf-8') as f:
        json.dump(v, f, indent=2)


# --- figures ----------------------------------------------------------------------------

def cmd_figures(args):
    """Redraw the four figures into docs/figures/depth-at-detection/ from the aggregates
    (run tree first, then the committed copies), and refresh the committed data copies
    when the run tree has them."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sd = args.out_root / '_summary' / 'depth_at_detection'
    (FIG_DIR / 'data').mkdir(parents=True, exist_ok=True)
    if sd.exists():
        for f in sd.iterdir():
            if f.suffix in ('.csv', '.json'):
                shutil.copy2(f, FIG_DIR / 'data' / f.name)
    # Reference categorical palette (dataviz skill, light mode), fixed order per class.
    colour = {GROUND: '#2a78d6', FLOOR: '#1baf7a', HORIZONTAL_NONFLOOR: '#eda100',
              NON_HORIZONTAL: '#eb6834', NO_PLANE: '#52514e'}
    ink, muted = '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                         'xtick.color': muted, 'ytick.color': muted,
                         'axes.spines.top': False, 'axes.spines.right': False})
    cities = args.cities
    labels = cities + ['pooled']

    # fig1: class shares, measured payloads, both tiers
    shares = [r for r in summary_csv(args.out_root, 'class_shares.csv')
              if r['payload_status'] == depthlib.MEASURED]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (tier, _) in zip(axes, TIERS):
        bottom = [0.0] * len(labels)
        for k in CLASSES:
            vals = []
            for c in labels:
                r = next((r for r in shares if r['city'] == c and r['tier'] == float(tier)),
                         None)
                vals.append((r or {}).get(f'share_{k}') or 0.0)
            ax.bar(labels, vals, bottom=bottom, color=colour[k], label=k,
                   edgecolor='white', linewidth=1)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f'confidence >= {tier}', color=ink)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelrotation=20)
    axes[0].set_ylabel('share of detections (measured payloads)')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle('Depth plane under the detection pixel', color=ink)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_classes.png', dpi=150)
    plt.close(fig)

    # fig2: offset_local histogram of `floor` detections (0.55 tier), local vs dominant
    hist = summary_csv(args.out_root, 'offset_hist.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, ref, col in zip(axes, ('local', 'dominant'), ('#1baf7a', '#52514e')):
        hr = [r for r in hist if r['tier'] == 0.55 and r['city'] == 'pooled'
              and r['reference'] == ref]
        tot = sum(r['n'] for r in hr) or 1
        ax.bar([r['bin_lo_m'] + 0.025 for r in hr], [r['n'] / tot for r in hr], width=0.045,
               color=col)
        ax.axvspan(*CLAIM_BAND_M, color='#eda100', alpha=0.15, lw=0)
        for b in SURFACE_BAND_M:
            ax.axvline(b, color=muted, lw=0.8, ls='--')
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f'vs {ref} reference plane', color=ink)
        ax.set_xlabel('height above the reference (m); shaded = [0.05, 0.30)')
    axes[0].set_ylabel('share of floor detections (pooled, >= 0.55)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_offset.png', dpi=150)
    plt.close(fig)

    # fig3: range ratio p10/p50/p90 per city, ground vs floor, flat 2.6 m vs per-pano
    rr = [r for r in summary_csv(args.out_root, 'range_ratio.csv') if r['tier'] == 0.30]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, title in zip(axes, ('ratio_flat', 'ratio_pp'),
                              ('depth range / flat range at 2.6 m',
                               'depth range / flat range at per-pano height')):
        for j, (cls, col) in enumerate(((GROUND, colour[GROUND]), (FLOOR, colour[FLOOR]))):
            for i, c in enumerate(labels):
                r = next((r for r in rr if r['city'] == c and r['plane_class'] == cls), None)
                if not r or r.get(f'{key}_p50') is None:
                    continue
                xpos = i + (j - 0.5) * 0.25
                ax.plot([xpos, xpos], [r[f'{key}_p10'], r[f'{key}_p90']], color=col, lw=2)
                ax.plot(xpos, r[f'{key}_p50'], 'o', color=col, ms=6,
                        label=cls if i == 0 else None)
        ax.axhline(1.0, color=muted, lw=0.8, ls='--')
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_title(title, color=ink)
    axes[0].set_ylabel('ratio (dot p50, bar p10-p90)')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_range_ratio.png', dpi=150)
    plt.close(fig)

    # fig4: GT groups, measured payloads, class shares + surface-share Wilson CI
    try:
        gt = [r for r in summary_csv(args.out_root, 'gt_classes.csv')
              if r['subset'] == 'measured' and r['city'] == 'pooled']
    except SystemExit:
        gt = []
    if gt:
        groups = [g for g in ('true', 'false', 'missed') if any(r['gt_group'] == g for r in gt)]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bottom = [0.0] * len(groups)
        for k in CLASSES:
            vals = [next(r for r in gt if r['gt_group'] == g).get(f'share_{k}') or 0.0
                    for g in groups]
            ax.bar(groups, vals, bottom=bottom, color=colour[k], label=k, edgecolor='white',
                   linewidth=1, width=0.6)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for i, g in enumerate(groups):
            r = next(r for r in gt if r['gt_group'] == g)
            ax.errorbar(i + 0.38, r['share_surface'],
                        yerr=[[r['share_surface'] - r['surface_lo']],
                              [r['surface_hi'] - r['share_surface']]],
                        fmt='D', color=ink, ms=4, capsize=3)
            ax.text(i, 1.02, f"n={int(r['n'])}", ha='center', color=muted)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('share (pooled, measured payloads, >= 0.55)')
        ax.set_title('Plane class by reviewer verdict  (diamond: surface share, 95% CI)',
                     color=ink, fontsize=9)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig4_gt.png', dpi=150)
        plt.close(fig)
    print(f'figures -> {FIG_DIR}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('cities', nargs='*', default=DEFAULT_CITIES)
        p.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs',
                       help='where runs/<city>/{results.jsonl,depth/} are read from (read-only)')
        p.add_argument('--out-root', type=Path, default=REPO_ROOT / 'runs',
                       help='where <city>/depth_at_detection/ and _summary/ are written')
        p.add_argument('--benchmark-root', type=Path,
                       default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    for name, fn in (('measure', cmd_measure), ('gt', cmd_gt), ('verdict', cmd_verdict),
                     ('figures', cmd_figures)):
        p = sub.add_parser(name)
        common(p)
        if name == 'measure':
            p.add_argument('--limit', type=int, default=0,
                           help='first N detection-bearing panos per city; writes no summaries')
            p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    if getattr(args, 'cities', None) == []:
        args.cities = DEFAULT_CITIES
    args.fn(args)


if __name__ == '__main__':
    main()
