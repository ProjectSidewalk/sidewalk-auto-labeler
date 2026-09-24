"""GSV ground-plane study (issue #52): read the ground plane Google's own depth
reconstruction observes under every GSV panorama, and use it to test the road-relative
raycast claim that the Mapillary tilt study (#42 / #50) could only infer.

A STUDY, not production: nothing here is wired into main.py, geo.py, fuse_sites.py or
sources/. It needs no network and no GPU. Inputs are a finished GSV run's
``results.jsonl``, its harvested ``depth/<pano_id>.json.gz`` payloads
(scripts/harvest_depth.py) and RampNet's benchmark bundles, all read in place.

Why GSV can test the claim cleanly. #50 found that on a vehicle rig the flat-ground
raycast wants the camera's orientation *relative to the local road*, not to gravity, and
supported that only indirectly (camera pitch regressed on an SfM altitude-profile grade).
GSV separates the two frames: its equirectangulars are gravity-rectified (the #27 pose
ablation, quoted in ``geo._world_ray``), so the camera frame IS the gravity frame, and the
depth payload's dominant ground plane gives the road's normal in that frame per pano. If
the road-relative claim is right, rotating each GSV ray into the observed ground frame
should tighten multi-view agreement, most on the steepest panoramas; if it is extra
degrees of freedom that help, a shuffled normal will help as much.

Frames. ``depth.py``'s coordinate convention (transcribed from streetlevel, checked
against its raster in tests/test_depth.py) puts a stored coordinate ``(x, y)`` at
``phi = 2*pi*x + pi/2`` with ``v = (sin t cos phi, sin t sin phi, cos t)``: so in the
depth frame **+x is camera-right, -y is camera-forward (x = 0.5, the heading) and +z is
down**. ``camera_frame_normal`` converts a plane normal to the camera frame
``(forward, right, up)`` with the normal pointing up; ``slopes`` turns that into
along-travel grade (rise ahead, positive uphill) and cross-slope (rise toward the
camera's right). ``road_pose`` turns it into the (pitch, roll) of the camera *relative
to the ground plane* in exactly ``geo._world_ray``'s composition, so the ground-normal
raycast goes through the production code path (``detection_ground_point`` /
``fuse_sites.fuse`` / ``eval_sites.evaluate_city``) with the angles (and the foot-shifted ray origin) injected -- the same
injection ``scripts/mapillary_tilt.py`` uses. tests/test_gsv_ground_plane.py pins the
frame mapping against ``depth.ground_range_at`` and the injection against an exact
ray-plane intersection.

Subcommands. Per-pano CSVs go to ``<out-root>/<city>/ground_plane/`` and the aggregated
ones to ``<out-root>/_summary/ground_plane/`` (out-root defaults to this checkout's
``runs/``); the aggregated ones are also committed under
``docs/figures/gsv-ground-plane/data/``, which ``figures`` falls back to:

    python scripts/gsv_ground_plane.py planes     [cities] --run-root ../labeler/runs
    python scripts/gsv_ground_plane.py chain      [cities] ...  # grade persistence (step 2)
    python scripts/gsv_ground_plane.py ablation   [cities] ...  # frozen-association spread
    python scripts/gsv_ground_plane.py eval       [cities] ...  # world P/R vs RampNet GT
    python scripts/gsv_ground_plane.py crossslope [cities] ...  # slope at ramp bearings
    python scripts/gsv_ground_plane.py verdict                  # the pre-registered reading
    python scripts/gsv_ground_plane.py figures                  # docs/figures/gsv-ground-plane

``planes`` must run first: every other subcommand reads its ``planes.csv``.

Camera height is held at ``geo.DEFAULT_CAMERA_HEIGHT_M`` (2.6 m) in every arm, so no
number here depends on the #40/#68 height question. Detections are taken at the
benchmark tier (``BENCHMARK_CONFIDENCE``) with ``mask_rig=False`` wherever they are joined
to GT or compared with #50's tables, for the reasons ``mapillary_tilt.cmd_ablation``
gives; ``crossslope`` describes what production ships, so it uses the operating point
and the rig mask.

Stdlib on top of depth.py / geo.py / fuse_sites.py / eval_sites.py / mapillary_tilt.py,
except ``figures`` (numpy + matplotlib).
"""
import argparse
import csv
import gzip
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
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
import mapillary_tilt as mt  # noqa: E402
from detectors import (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       on_camera_rig)

DEFAULT_CITIES = ['bend', 'paterson', 'gainesville', 'sao_paulo']
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'gsv-ground-plane'
ARMS = ['off', 'ground-normal', 'shuffled-normal']
# Added after step 1, NOT part of the pre-registered reading: the plane implied by the
# rig's own metadata attitude (a car follows the road). Step 1 found the depth normal is
# a noisy per-pano estimate of the road whose agreement with that attitude rises to
# r ~0.9 where the ground plane is large, while the attitude itself persists along the
# street at r 0.72-0.87 -- so this arm separates "the depth normal is too noisy" from
# "the mechanism does not hold on GSV". It is scored on its own four-arm site set
# (`*_4arm`) so the pre-registered three-arm rows are untouched by it.
EXPLORATORY_ARM = 'rig-attitude-normal'
ALL_ARMS = ARMS + [EXPLORATORY_ARM]
# Added on review of PR #78, also outside the pre-registered reading, and scored on their
# own site set (`*_review`: every arm below places every member) so neither the
# three-arm nor the four-arm rows move:
#   - `travel-only`: the observed plane with its cross-slope zeroed -- the like-for-like
#     test of #50's road-relative correction, which removes only the along-travel grade
#     (the Mapillary SfM altitude profile has no cross term);
#   - `travel-shuffled`: its within-city control (another pano's grade, no cross);
#   - `travel-bucket-shuffled` / `bucket-shuffled-normal`: MAGNITUDE-MATCHED controls --
#     the grade (resp. the whole normal) permuted only among panos in the same |grade|
#     bucket, so in the 4+ deg bucket the control applies >= 4 deg too. The city-wide
#     shuffle applies a typical ~1 deg there, so comparing the arm with it in a steep
#     bucket compares two different correction sizes;
#   - `cross-flipped`, `cross-only`, `cross-only-flipped`: the cross-slope sign check
#     (is the frame mirrored, or is one plane the wrong model across a crowned road?),
#     read with the detection-side split `cmd_ablation` writes for every arm.
TRAVEL_ARM = 'travel-only'
REVIEW_ARMS = [TRAVEL_ARM, 'travel-shuffled', 'travel-bucket-shuffled',
               'bucket-shuffled-normal', 'cross-flipped', 'cross-only', 'cross-only-flipped']
ABLATION_ARMS = ALL_ARMS + REVIEW_ARMS
# The GT eval runs the pre-registered arms, the rig arm and the travel pair (the
# statistic #50's precondition asks for, on the arm that mirrors #50's correction).
EVAL_ARMS = ALL_ARMS + [TRAVEL_ARM, 'travel-shuffled']
SHUFFLE_SEED = 52
# Detection side relative to the camera heading: x_normalized < 0.5 is left of it.
SIDES = ('left', 'right', 'mixed')
# Pairs whose steeper |cross-slope| reaches this are where a cross-slope sign would show.
CROSS_SPLIT_DEG = 2.0
# Ground-plane pixel share above which the depth normal is called well determined.
WELL_DETERMINED_SHARE = 0.3

# Grade buckets fixed by the plan (issue #52), before any number was seen.
GRADE_BUCKETS = [(0, 1), (1, 2), (2, 4), (4, math.inf)]


# --- The pre-registered reading -------------------------------------------------------
#
# Written and committed BEFORE the full run (see the commit that introduced this file).
# The plan's words: "The claim is supported if `ground-normal` beats `off` on the top
# grade buckets in at least three of four cities on median spread and does not lose p90
# GT distance anywhere, while `shuffled-normal` does not. It is undercut if `off` wins or
# the control matches the treatment." Operationalised here, with no freedom left:
#
#   - "top grade buckets" = 2-4 deg and 4+ deg of the pair's larger |along-travel grade|,
#     counting only buckets where the `off` arm has >= MIN_BUCKET_PAIRS pairs. A city
#     with no qualifying bucket is "not testable" and counts as a non-win.
#   - "beats on median spread" = the arm's median within-site pair distance is strictly
#     lower than off's in EVERY qualifying top bucket, on the primary site set
#     (`uncapped`: #50 section 4.4's design, every arm placing every member).
#   - "does not lose p90 GT distance anywhere" = ground-normal's p90 GT-to-site distance
#     (5 m match radius, production 25 m cap) <= off's, to the 0.01 m it is reported at,
#     in every city.
#   - "shuffled-normal does not" = shuffled-normal wins (same bucket rule) in fewer than
#     three cities.
#   - SUPPORTED iff ground-normal wins >= 3 cities AND the p90 rule holds AND shuffled
#     wins < 3 cities.
#   - UNDERCUT iff off wins -- ground-normal wins <= 1 city -- OR the control matches the
#     treatment -- shuffled wins >= ground-normal's wins.
#   - otherwise NOT SUPPORTED (the specific failing clause is printed).

TOP_BUCKETS = ('2-4', '4+')
MIN_BUCKET_PAIRS = 100
PRIMARY_SET = 'uncapped'


def bucket_win(abl_rows, city, arm, site_set=PRIMARY_SET):
    """True/False: does `arm` beat `off` on median spread in every qualifying top grade
    bucket of `city`? None when no top bucket has enough pairs to read."""
    rows = {r['arm']: r for r in abl_rows
            if r['city'] == city and r['site_set'] == site_set}
    off, a = rows.get('off'), rows.get(arm)
    if off is None or a is None:
        return None
    qualifying = [b for b in TOP_BUCKETS
                  if (off.get(f'n_pairs_grade_{b}') or 0) >= MIN_BUCKET_PAIRS]
    if not qualifying:
        return None
    return all(a[f'median_pair_m_grade_{b}'] < off[f'median_pair_m_grade_{b}']
               for b in qualifying)


def verdict(abl_rows, eval_rows, cities):
    """Apply the pre-registered reading. Returns (label, detail dict)."""
    gn = {c: bucket_win(abl_rows, c, 'ground-normal') for c in cities}
    sh = {c: bucket_win(abl_rows, c, 'shuffled-normal') for c in cities}
    p90 = {}
    for c in cities:
        rows = {r['arm']: r for r in eval_rows if r['city'] == c}
        if 'off' in rows and 'ground-normal' in rows:
            p90[c] = (round(rows['ground-normal']['match_dist_p90'], 2)
                      <= round(rows['off']['match_dist_p90'], 2))
    gn_wins = sum(v is True for v in gn.values())
    sh_wins = sum(v is True for v in sh.values())
    p90_ok = len(p90) == len(cities) and all(p90.values())
    detail = {'ground_normal_bucket_win': gn, 'shuffled_bucket_win': sh,
              'ground_normal_wins': gn_wins, 'shuffled_wins': sh_wins,
              'p90_not_worse': p90, 'p90_rule_holds': p90_ok}
    if gn_wins >= 3 and p90_ok and sh_wins < 3:
        return 'SUPPORTED', detail
    if gn_wins <= 1 or sh_wins >= gn_wins:
        detail['why'] = ('off wins (ground-normal wins <= 1 city)' if gn_wins <= 1
                         else 'the shuffled control matches the treatment')
        return 'UNDERCUT', detail
    why = []
    if gn_wins < 3:
        why.append(f'ground-normal wins only {gn_wins} of {len(cities)} cities')
    if not p90_ok:
        why.append('ground-normal loses p90 GT distance in '
                   + ', '.join(c for c, ok in p90.items() if not ok))
    if sh_wins >= 3:
        why.append(f'shuffled-normal also wins {sh_wins} cities')
    detail['why'] = '; '.join(why)
    return 'NOT SUPPORTED', detail


# --- Frames and the decomposition (stdlib; tests/test_gsv_ground_plane.py) -------------

def camera_frame_normal(nx, ny, nz):
    """A depth-frame plane normal -> the unit UPWARD normal (forward, right, up) in the
    camera frame.

    depth.py's frame is +x right, -y forward, +z down (module docstring), and a payload
    stores a ground normal as ~(0, 0, -1) but either orientation describes the plane, so
    the result is flipped to point up.

    Example:
        >>> camera_frame_normal(0.0, 0.0, -1.0)
        (-0.0, 0.0, 1.0)
    """
    f, r, u = -ny, nx, -nz
    n = math.sqrt(f * f + r * r + u * u)
    if u < 0:
        n = -n
    return f / n, r / n, u / n


def depth_frame_normal(n_f, n_r, n_u):
    """Inverse of camera_frame_normal (up-pointing), for building synthetic payloads."""
    return n_r, -n_f, -n_u


def slopes(n_f, n_r, n_u):
    """(grade_deg, cross_deg) of a plane with upward camera-frame normal (f, r, u).

    grade = the angle the plane rises moving FORWARD (along the camera heading, which on
    a GSV car is the direction of travel); cross = the angle it rises moving toward the
    camera's RIGHT. Both are exact plane angles, not small-angle approximations.

    Example:
        >>> g, c = slopes(-math.sin(math.radians(3)), 0.0, math.cos(math.radians(3)))
        >>> round(g, 9), round(c, 9)
        (3.0, -0.0)
    """
    return (math.degrees(math.atan2(-n_f, n_u)), math.degrees(math.atan2(-n_r, n_u)))


def normal_from_slopes(grade_deg, cross_deg):
    """Upward unit normal (f, r, u) of the plane with this grade and cross-slope."""
    tg, tc = math.tan(math.radians(grade_deg)), math.tan(math.radians(cross_deg))
    n = math.sqrt(tg * tg + tc * tc + 1.0)
    return -tg / n, -tc / n, 1.0 / n


def slope_along(grade_deg, cross_deg, azimuth_deg):
    """Rise angle of the plane along a horizontal direction `azimuth_deg` clockwise from
    camera-forward -- what a ray at that azimuth actually crosses.

    Example:
        >>> round(slope_along(2.0, 0.0, 180.0), 9)   # looking back down a 2 deg hill
        -2.0
    """
    a = math.radians(azimuth_deg)
    return math.degrees(math.atan(math.tan(math.radians(grade_deg)) * math.cos(a)
                                  + math.tan(math.radians(cross_deg)) * math.sin(a)))


def road_pose(n_f, n_r, n_u):
    """(pitch_deg, roll_deg) of the gravity-level camera RELATIVE TO the ground plane,
    in geo._world_ray's composition (pitch about right, + raises the view axis; then roll
    about forward, + lifts the image's right side).

    Derivation: after pitch a and roll r, _world_ray's camera axes expressed in the
    'world' (here: road) frame make the road's up axis read (sin a, sin r cos a,
    cos r cos a) in camera coordinates -- set that equal to the observed normal. On a
    road rising ahead the camera is pitched DOWN relative to it (a < 0), which is what
    lengthens a forward ray's depression and shortens its range.
    """
    return (math.degrees(math.asin(max(-1.0, min(1.0, n_f)))),
            math.degrees(math.atan2(n_r, n_u)))


def fold180(a):
    """Fold an angle difference into [0, 90]: the angle between two undirected axes."""
    a = abs(geo.norm_deg(a)) % 180.0
    return min(a, 180.0 - a)


# --- planes: one row per panorama ------------------------------------------------------

def dominant_ground(payload):
    """The plane depth.ground_plane picks, returned as the Plane itself (the module
    returns only its tilt magnitude). Same candidate rule, same ordering; the caller
    cross-checks the pick against depth.ground_plane so the two cannot drift apart."""
    w, h = payload.width, payload.height
    counts = Counter(payload.indices)
    below = Counter(payload.indices[(h // 2) * w:])
    best = None
    for idx, count in counts.items():
        if idx == depthlib.SKY or idx >= len(payload.planes):
            continue
        p = payload.planes[idx]
        tilt = math.degrees(math.acos(min(1.0, abs(p.nz))))
        if tilt > depthlib.GROUND_MAX_TILT_DEG:
            continue
        if below[idx] / count < depthlib.GROUND_MIN_BELOW_HORIZON:
            continue
        if best is None or count > best[0]:
            best = (count, p)
    return None if best is None else best[1]


def _read_one(path_str):
    """Worker: one payload -> the plane fields of its row (picklable dict)."""
    path = Path(path_str)
    pid = path.name[:-len('.json.gz')]
    row = {'pano_id': pid, 'status': None, 'n_planes': None, 'ground_d_m': None,
           'tilt_deg': None, 'pixel_share': None, 'n_f': None, 'n_r': None, 'n_u': None}
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            payload = depthlib.parse(json.load(f)['depth_b64'])
    except Exception:   # noqa: BLE001 -- recorded as a status, never raised
        row['status'] = depthlib.UNPARSED
        return row
    row['n_planes'] = payload.n_planes
    if payload.degenerate:
        row['status'] = depthlib.DEGENERATE
        return row
    ground = depthlib.ground_plane(payload)
    if ground is None:
        row['status'] = depthlib.NO_GROUND
        return row
    plane = dominant_ground(payload)
    tilt = math.degrees(math.acos(min(1.0, abs(plane.nz))))
    if plane.d != ground.camera_height_m or abs(tilt - ground.tilt_deg) > 1e-9:
        raise RuntimeError(f'{pid}: dominant_ground disagrees with depth.ground_plane '
                           f'({plane.d} vs {ground.camera_height_m}) -- the copy drifted')
    row['status'] = depthlib.classify_height(ground.camera_height_m, ground.tilt_deg,
                                             exactly_level=ground.exactly_level)
    n_f, n_r, n_u = camera_frame_normal(plane.nx, plane.ny, plane.nz)
    row.update({'ground_d_m': plane.d, 'tilt_deg': tilt,
                'pixel_share': ground.pixel_share, 'n_f': n_f, 'n_r': n_r, 'n_u': n_u})
    return row


def load_run_meta(run_dir):
    """Per pano: position, heading, the rig's metadata pitch/roll, capture date, link
    yaws and operational-detection count, straight from results.jsonl."""
    meta = {}
    with open(run_dir / 'results.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec['pano']
            if p.get('lat') is None or p.get('camera_heading') is None:
                continue
            meta[p['panorama_id']] = {
                'lat': p['lat'], 'lng': p['lng'],
                'heading': geo.norm_deg(float(p['camera_heading'])) % 360.0,
                'meta_pitch': None if p.get('camera_pitch') is None
                else geo.norm_deg(float(p['camera_pitch'])),
                'meta_roll': None if p.get('camera_roll') is None
                else geo.norm_deg(float(p['camera_roll'])),
                'capture_date': p.get('capture_date'),
                'links': [(lk.get('target_gsv_panorama_id') or lk.get('target_pano_id'),
                           lk.get('yaw_deg')) for lk in (p.get('links') or [])],
                'n_operational': sum(1 for d in rec.get('detections', [])
                                     if d['confidence'] >= BENCHMARK_CONFIDENCE)}
    return meta


def planes_path(out_root, city):
    return out_root / city / 'ground_plane' / 'planes.csv'


def summary_dir(out_root):
    d = out_root / '_summary' / 'ground_plane'
    d.mkdir(parents=True, exist_ok=True)
    return d


def ols(xs, ys):
    """(slope, r) of y on x, or (None, None)."""
    if len(xs) < 10:
        return None, None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if not sxx or not syy:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def cmd_planes(args):
    """Step 1: the observed ground plane per panorama, decomposed into along-travel grade
    and cross-slope, with the frame checks that tie it to the world."""
    summary, by_year, rig_rows = [], [], []
    for city in args.cities:
        run_dir = args.run_root / city
        meta = load_run_meta(run_dir)
        files = sorted((run_dir / 'depth').glob('*.json.gz'))
        files = [f for f in files if f.name[:-len('.json.gz')] in meta]
        if args.limit:
            files = files[:args.limit]
        print(f'{city}: {len(meta):,} panos in results.jsonl, {len(files):,} payloads to read',
              flush=True)
        rows = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(_read_one, [str(f) for f in files],
                                           chunksize=200), 1):
                rows[row['pano_id']] = row
                if i % 10000 == 0:
                    print(f'  {city}: {i:,}/{len(files):,}', flush=True)
        out_rows = []
        pids = list(rows) if args.limit else list(meta)
        for pid in pids:
            m = meta[pid]
            row = rows.get(pid) or {'pano_id': pid, 'status': 'no_file'}
            r = {'pano_id': pid, 'capture_date': m['capture_date'], 'lat': m['lat'],
                 'lng': m['lng'], 'heading_deg': m['heading'],
                 'meta_pitch_deg': m['meta_pitch'], 'meta_roll_deg': m['meta_roll'],
                 'n_operational': m['n_operational'], 'n_links': len(m['links'])}
            r.update({k: row.get(k) for k in ('status', 'n_planes', 'ground_d_m', 'tilt_deg',
                                              'pixel_share', 'n_f', 'n_r', 'n_u')})
            if row.get('n_u') is not None:
                g, c = slopes(row['n_f'], row['n_r'], row['n_u'])
                r['grade_deg'], r['cross_deg'] = g, c
            yaws = [y for _, y in m['links'] if y is not None]
            r['link_axis_diff_deg'] = (min(fold180(y - m['heading']) for y in yaws)
                                       if yaws else None)
            out_rows.append(r)
        path = planes_path(args.out_root, city)
        path.parent.mkdir(parents=True, exist_ok=True)
        mt.write_csv(path, out_rows)

        status = Counter(r['status'] for r in out_rows)
        meas = [r for r in out_rows if r['status'] == depthlib.MEASURED]
        with_plane = [r for r in out_rows if r.get('grade_deg') is not None]
        ag = [abs(r['grade_deg']) for r in meas]
        ac = [abs(r['cross_deg']) for r in meas]
        links = [r['link_axis_diff_deg'] for r in out_rows if r['link_axis_diff_deg'] is not None]
        imu = [(r['meta_pitch_deg'], r['grade_deg']) for r in meas
               if r['meta_pitch_deg'] is not None]
        imr = [(r['meta_roll_deg'], r['cross_deg']) for r in meas
               if r['meta_roll_deg'] is not None]
        sp, rp = ols([a for a, _ in imu], [b for _, b in imu])
        sr, rr = ols([a for a, _ in imr], [b for _, b in imr])
        wd = [r for r in meas if r['pixel_share'] >= WELL_DETERMINED_SHARE
              and r['meta_pitch_deg'] is not None and r['meta_roll_deg'] is not None]
        wsp, wrp = ols([r['meta_pitch_deg'] for r in wd], [r['grade_deg'] for r in wd])
        wsr, wrr = ols([r['meta_roll_deg'] for r in wd], [r['cross_deg'] for r in wd])
        s = {'city': city, 'n_panos': len(out_rows), 'n_measured': len(meas),
             'frac_measured': len(meas) / len(out_rows),
             **{f'n_{k}': v for k, v in sorted(status.items())},
             'frac_synthetic_of_nondegenerate':
                 status[depthlib.SYNTHETIC_GROUND]
                 / max(1, len(with_plane) + status[depthlib.NO_GROUND]),
             'grade_abs_p50': mt.pct(ag, .5), 'grade_abs_p90': mt.pct(ag, .9),
             'grade_abs_p99': mt.pct(ag, .99),
             'cross_abs_p50': mt.pct(ac, .5), 'cross_abs_p90': mt.pct(ac, .9),
             'cross_abs_p99': mt.pct(ac, .99),
             'grade_p50_signed': mt.pct([r['grade_deg'] for r in meas], .5),
             'cross_p50_signed': mt.pct([r['cross_deg'] for r in meas], .5),
             'cross_mean_signed': statistics.mean(r['cross_deg'] for r in meas) if meas else None,
             'frac_cross_falls_right': (sum(r['cross_deg'] < 0 for r in meas) / len(meas)
                                        if meas else None),
             'tilt_p50': mt.pct([r['tilt_deg'] for r in meas], .5),
             'tilt_p90': mt.pct([r['tilt_deg'] for r in meas], .9),
             'frac_grade_gt_1': sum(a > 1 for a in ag) / len(ag) if ag else None,
             'frac_grade_gt_2': sum(a > 2 for a in ag) / len(ag) if ag else None,
             'frac_grade_gt_4': sum(a > 4 for a in ag) / len(ag) if ag else None,
             'link_axis_diff_p50': mt.pct(links, .5), 'link_axis_diff_p90': mt.pct(links, .9),
             'frac_link_axis_within_5': (sum(x <= 5 for x in links) / len(links)
                                         if links else None),
             'slope_grade_on_meta_pitch': sp, 'r_grade_meta_pitch': rp,
             'slope_cross_on_meta_roll': sr, 'r_cross_meta_roll': rr,
             'n_well_determined': len(wd),
             'frac_well_determined': len(wd) / len(meas) if meas else None,
             'slope_grade_on_meta_pitch_welldet': wsp, 'r_grade_meta_pitch_welldet': wrp,
             'slope_cross_on_meta_roll_welldet': wsr, 'r_cross_meta_roll_welldet': wrr,
             'pixel_share_p50': mt.pct([r['pixel_share'] for r in meas], .5)}
        summary.append(s)
        # The frame check, binned: depth grade / cross-slope against the rig's metadata
        # attitude. A car rides on the road, so a gravity-frame ground normal should track
        # it with slope ~ +/-1 wherever the plane is well determined.
        for axis, mkey, dkey in (('pitch', 'meta_pitch_deg', 'grade_deg'),
                                 ('roll', 'meta_roll_deg', 'cross_deg')):
            for subset, rs in (('all', meas),
                               ('well_determined', [r for r in meas if r['pixel_share']
                                                    >= WELL_DETERMINED_SHARE])):
                bins = defaultdict(list)
                for r in rs:
                    if r[mkey] is not None:
                        bins[max(-8, min(8, round(r[mkey])))].append(r[dkey])
                for b, v in sorted(bins.items()):
                    rig_rows.append({'city': city, 'axis': axis, 'subset': subset,
                                     'meta_bin_deg': b, 'n': len(v),
                                     'depth_p25': mt.pct(v, .25), 'depth_p50': mt.pct(v, .5),
                                     'depth_p75': mt.pct(v, .75)})
        years = defaultdict(list)
        for r in meas:
            years[(r['capture_date'] or '????')[:4]].append(r)
        for y, rs in sorted(years.items()):
            by_year.append({'city': city, 'year': y, 'n_measured': len(rs),
                            'grade_abs_p50': mt.pct([abs(r['grade_deg']) for r in rs], .5),
                            'grade_abs_p90': mt.pct([abs(r['grade_deg']) for r in rs], .9),
                            'cross_abs_p50': mt.pct([abs(r['cross_deg']) for r in rs], .5),
                            'cross_p50_signed': mt.pct([r['cross_deg'] for r in rs], .5),
                            'ground_d_p50': mt.pct([r['ground_d_m'] for r in rs], .5)})
        print(f"{city}: measured {len(meas):,}/{len(out_rows):,} ({s['frac_measured']:.1%}); "
              f"status {dict(status)}; |grade| p50/p90 {s['grade_abs_p50']:.2f}/"
              f"{s['grade_abs_p90']:.2f}; |cross| p50/p90 {s['cross_abs_p50']:.2f}/"
              f"{s['cross_abs_p90']:.2f}; cross falls right {s['frac_cross_falls_right']:.1%}; "
              f"link axis p50 {s['link_axis_diff_p50']:.2f}; grade~meta_pitch slope "
              f"{sp if sp is None else round(sp, 3)} r {rp if rp is None else round(rp, 3)}",
              flush=True)
    if not args.limit:
        mt.write_csv(summary_dir(args.out_root) / 'planes_summary.csv', summary)
        mt.write_csv(summary_dir(args.out_root) / 'planes_by_year.csv', by_year)
        mt.write_csv(summary_dir(args.out_root) / 'planes_vs_rig.csv', rig_rows)


def load_planes(out_root, city):
    """{pano_id: row} of planes.csv with numbers parsed."""
    rows = mt._read_csv(planes_path(out_root, city))
    return {r['pano_id']: r for r in rows}


def measured(planes):
    return {pid: r for pid, r in planes.items()
            if r['status'] == depthlib.MEASURED and r['n_u'] is not None}


# --- chain: how far a grade persists along the link graph (offline part of step 2) -----

DIST_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 80)]
MAX_HOPS = 8


CHAIN_SIGNALS = ('depth', 'depth_well_determined', 'rig_attitude')


def cmd_chain(args):
    """Autocorrelation of the observed slope along the street, pano to pano.

    Pairs are panos joined by the GSV link graph (up to MAX_HOPS hops, both measured).
    For each pair the slope of EACH pano's own plane is taken along the one world bearing
    A->B, so both numbers describe the same stretch of street; a persistent road grade
    and a consistent measurement give r -> 1 at short separation, and the RMS of the
    difference / sqrt(2) bounds the per-pano noise from above. Split by whether the two
    panos share a capture month (same drive, same rig) or not.

    Three signals, so the depth plane is read against something:
      - `depth`: the dominant ground plane, every measured pano;
      - `depth_well_determined`: the same, both panos' plane covering >=
        WELL_DETERMINED_SHARE of the image;
      - `rig_attitude`: the plane implied by the GSV metadata pitch/roll
        (rig_attitude_normal) -- a car follows the road, so this is an independent,
        IMU-borne road-grade estimate on exactly the same pairs."""
    rows = []
    for city in args.cities:
        run_dir = args.run_root / city
        meta = load_run_meta(run_dir)
        meas = measured(load_planes(args.out_root, city))
        adj = defaultdict(set)
        for pid in meas:
            for tgt, _ in meta[pid]['links']:
                if tgt in meas and tgt != pid:
                    adj[pid].add(tgt)
                    adj[tgt].add(pid)
        bins = defaultdict(list)
        for a in sorted(adj):
            seen = {a: 0}
            q = deque([a])
            while q:
                u = q.popleft()
                if seen[u] >= MAX_HOPS:
                    continue
                for v in adj[u]:
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        q.append(v)
            ma, pa = meta[a], meas[a]
            for b, hops in seen.items():
                if b <= a:
                    continue
                mb, pb = meta[b], meas[b]
                d = geo.haversine_m(ma['lat'], ma['lng'], mb['lat'], mb['lng'])
                if d < 1.0:
                    continue
                e, n = geo.LocalFrame(ma['lat'], ma['lng']).to_enu(mb['lat'], mb['lng'])
                bearing = math.degrees(math.atan2(e, n))
                sa = slope_along(pa['grade_deg'], pa['cross_deg'], bearing - ma['heading'])
                sb = slope_along(pb['grade_deg'], pb['cross_deg'], bearing - mb['heading'])
                vals = {'depth': (sa, sb)}
                if min(pa['pixel_share'], pb['pixel_share']) >= WELL_DETERMINED_SHARE:
                    vals['depth_well_determined'] = (sa, sb)
                if None not in (ma['meta_pitch'], ma['meta_roll'], mb['meta_pitch'],
                                mb['meta_roll']):
                    vals['rig_attitude'] = (
                        slope_along(ma['meta_pitch'], -ma['meta_roll'], bearing - ma['heading']),
                        slope_along(mb['meta_pitch'], -mb['meta_roll'], bearing - mb['heading']))
                same = (ma['capture_date'] or 'a')[:7] == (mb['capture_date'] or 'b')[:7]
                for lo, hi in DIST_BINS:
                    if lo <= d < hi:
                        for sig, v in vals.items():
                            bins[(sig, f'{lo}-{hi}', 'same' if same else 'different')].append(v)
                            bins[(sig, f'{lo}-{hi}', 'all')].append(v)
                        break
        for sig in CHAIN_SIGNALS:
            for lo, hi in DIST_BINS:
                for vint in ('all', 'same', 'different'):
                    prs = bins.get((sig, f'{lo}-{hi}', vint), [])
                    _, r = ols([a for a, _ in prs], [b for _, b in prs])
                    diffs = [a - b for a, b in prs]
                    rows.append({'city': city, 'signal': sig, 'dist_bin_m': f'{lo}-{hi}',
                                 'vintage': vint, 'n_pairs': len(prs), 'r': r,
                                 'rms_diff_over_sqrt2': (
                                     math.sqrt(sum(x * x for x in diffs) / len(diffs))
                                     / math.sqrt(2) if diffs else None),
                                 'abs_diff_p50': mt.pct([abs(x) for x in diffs], .5),
                                 'slope_abs_p50': mt.pct([abs(a) for a, _ in prs], .5)})
            for vint in ('same', 'different'):
                line = [f'{city:>11} {sig:>22} {vint:>9}:']
                for r_ in rows[-3 * len(DIST_BINS):]:
                    if r_['vintage'] == vint and r_['signal'] == sig:
                        line.append(f"{r_['dist_bin_m']}m " + ('  n/a' if r_['r'] is None
                                                               else f"{r_['r']:.2f}"))
                print('  '.join(line), flush=True)
    mt.write_csv(summary_dir(args.out_root) / 'chain.csv', rows)


# --- The arms --------------------------------------------------------------------------

def ground_frame_fields(lat, lng, heading_deg, normal, h=geo.DEFAULT_CAMERA_HEIGHT_M):
    """Pano-block overrides that make geo.detection_ground_point raycast onto the plane
    with upward camera-frame normal `normal`, `h` below the camera along that normal.

    Two parts. (1) camera_pitch/camera_roll = road_pose(normal): _world_ray then
    expresses the ray in the plane's frame, and d = h / tan(depression) is the distance
    IN the plane. (2) The ray origin moves to the foot of the perpendicular from the
    camera to the plane, which is not under the camera on a slope: it sits h*sin(slope)
    uphill (0.045 m per degree at 2.6 m). Without (2) every point of a sloped pano
    shifts downhill by that constant -- a per-pano bias the arms would then be scored on.
    What remains is the in-plane vs horizontal distance, a 1/cos factor (0.24% at 4 deg);
    tests/test_gsv_ground_plane.py pins the total against an exact intersection."""
    n_f, n_r, _ = normal
    pitch, roll = road_pose(*normal)
    b = math.radians(heading_deg)
    hf, hr = -h * n_f, -h * n_r                      # foot offset, camera frame
    e = hf * math.sin(b) + hr * math.cos(b)
    n = hf * math.cos(b) - hr * math.sin(b)
    flat, flng = geo.LocalFrame(lat, lng).to_latlng(e, n)
    return {'camera_pitch': pitch, 'camera_roll': roll, 'lat': flat, 'lng': flng}


def rig_attitude_normal(meta_pitch_deg, meta_roll_deg):
    """The ground normal implied by the capture rig's metadata attitude, for a car that
    rides on the road: grade = pitch, cross-slope = -roll. The signs are not assumed;
    they are the ones the depth planes themselves select (`planes` regresses depth grade
    on metadata pitch and depth cross-slope on metadata roll: slopes > 0 and < 0)."""
    return normal_from_slopes(meta_pitch_deg, -meta_roll_deg)


def _bucket_shuffle(pids, values, key, seed):
    """Permute `values` (aligned with `pids`) only among pids sharing `key(pid)`."""
    groups = defaultdict(list)
    for i, pid in enumerate(pids):
        groups[key(pid)].append(i)
    out = list(values)
    rng = random.Random(seed)
    for b in sorted(groups):
        idx = groups[b]
        vs = [values[i] for i in idx]
        rng.shuffle(vs)
        for i, v in zip(idx, vs):
            out[i] = v
    return out


def arm_normals(meas, seed=SHUFFLE_SEED):
    """{arm: {pano_id: upward camera-frame normal}} for every non-off arm.

    'ground-normal' uses each pano's own observed plane. 'shuffled-normal' hands every
    measured pano ANOTHER measured pano's normal -- a fixed-seed permutation over the
    city -- so it has the same distribution of corrections and the same extra degrees of
    freedom with the pano-specific information removed. Panos without a measured plane
    get no entry in any arm and are raycast flat, exactly as in production.

    The review arms (REVIEW_ARMS; see there) decompose the same plane: travel-only keeps
    the grade and zeroes the cross-slope, cross-only the reverse, the *-flipped arms
    negate the cross-slope, and the two bucket-shuffled arms permute within the pano's
    own |grade| bucket (GRADE_BUCKETS) so the control's magnitude matches the arm's in
    every bucket.

    Example:
        >>> m = {'a': dict(zip(('n_f', 'n_r', 'n_u'), normal_from_slopes(3.0, -1.0)))}
        >>> [round(v, 9) for v in slopes(*arm_normals(m)['travel-only']['a'])]
        [3.0, 0.0]
    """
    pids = sorted(meas)
    real = {pid: (meas[pid]['n_f'], meas[pid]['n_r'], meas[pid]['n_u']) for pid in pids}
    values = [real[p] for p in pids]
    random.Random(seed).shuffle(values)
    rig = {pid: rig_attitude_normal(meas[pid]['meta_pitch_deg'], meas[pid]['meta_roll_deg'])
           for pid in pids if meas[pid].get('meta_pitch_deg') is not None
           and meas[pid].get('meta_roll_deg') is not None}
    gc = {pid: slopes(*real[pid]) for pid in pids}
    grades = [gc[p][0] for p in pids]
    shuffled_grades = list(grades)
    random.Random(seed).shuffle(shuffled_grades)

    def bucket(pid):
        return grade_bucket(abs(gc[pid][0]))

    bucket_grades = _bucket_shuffle(pids, grades, bucket, seed)
    bucket_normals = _bucket_shuffle(pids, [real[p] for p in pids], bucket, seed)
    return {'ground-normal': real, 'shuffled-normal': dict(zip(pids, values)),
            EXPLORATORY_ARM: rig,
            TRAVEL_ARM: {p: normal_from_slopes(gc[p][0], 0.0) for p in pids},
            'travel-shuffled': {p: normal_from_slopes(g, 0.0)
                                for p, g in zip(pids, shuffled_grades)},
            'travel-bucket-shuffled': {p: normal_from_slopes(g, 0.0)
                                       for p, g in zip(pids, bucket_grades)},
            'bucket-shuffled-normal': dict(zip(pids, bucket_normals)),
            'cross-flipped': {p: (f, -r, u) for p, (f, r, u) in real.items()},
            'cross-only': {p: normal_from_slopes(0.0, gc[p][1]) for p in pids},
            'cross-only-flipped': {p: normal_from_slopes(0.0, -gc[p][1]) for p in pids}}


def pair_side(xa, xb):
    """'left' / 'right' when both detections sit on that side of their camera's heading
    (x_normalized < 0.5 is left), else 'mixed'. A cross-slope correction moves a left
    and a right detection in opposite senses, so its sign is read per side.

    Example:
        >>> pair_side(0.2, 0.4), pair_side(0.7, 0.9), pair_side(0.2, 0.9)
        ('left', 'right', 'mixed')
    """
    a, b = xa < 0.5, xb < 0.5
    if a and b:
        return 'left'
    if not a and not b:
        return 'right'
    return 'mixed'


def arm_fields(p, normals, arm):
    """pose_fields overrides for SlimPano `p` under `arm` (None for production's flat)."""
    if arm == 'off':
        return {'camera_pitch': None, 'camera_roll': None}
    n = normals[arm].get(p.pano_id)
    if n is None:
        return {'camera_pitch': 0.0, 'camera_roll': 0.0}     # == the flat raycast
    return ground_frame_fields(p.lat, p.lng, p.camera_heading, n)


def arm_panos(panos, normals, arm):
    """SlimPanos carrying the arm's ground frame (pitch, roll and the foot-shifted
    origin) so fuse_sites / eval_sites raycast under it with apply_pose=True."""
    return [replace(p, **arm_fields(p, normals, arm)) for p in panos]


def grade_bucket(g):
    for lo, hi in GRADE_BUCKETS:
        if lo <= g < hi:
            return mt.bucket_label(lo, hi)
    return mt.bucket_label(*GRADE_BUCKETS[-1])


GRADE_LABELS = [mt.bucket_label(lo, hi) for lo, hi in GRADE_BUCKETS]


def benchmark_params(**kw):
    """The tier every GT-joined number here uses; see the module docstring."""
    return fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False, **kw)


# --- ablation: frozen-association multi-view spread (step 3, instrument 1) -------------

def cmd_ablation(args):
    """#50 section 4.4's instrument on GSV, three arms.

    Association is frozen from the production (off) fuse; every operational member is
    re-projected under each arm and the within-site pairwise member distance is compared.
    Two site sets, both reported:

      - `uncapped` (primary, and the one the pre-registered reading uses): #50's design,
        max_range_m = inf, a site kept only if every arm places every member.
      - `capped`: the production 25 m cap, a site kept only if every arm places every
        member within it -- the survivorship #50 warned about, reported beside it because
        #50's open question was the tail under the cap.

    Pairs are bucketed by the pair's larger |along-travel grade| (the plan's buckets),
    by the TRUE grade in every arm -- so the shuffled control is scored on the same pairs,
    bucketed by how steep they really are. A pair enters the buckets only if both panos
    carry a measured plane; the overall rows include every pair.

    Two more site-set pairs: `*_4arm` adds the rig-attitude arm, `*_review` every arm in
    ABLATION_ARMS. Each set is the intersection over its own arms, so adding an arm never
    moves a row of a smaller set. Every row also carries the detection-side split
    (pair_side; both-measured pairs, and those whose steeper |cross-slope| is >=
    CROSS_SPLIT_DEG), which is how the cross-slope sign is read."""
    params = benchmark_params()
    all_rows = []
    for city in args.cities:
        meas = measured(load_planes(args.out_root, city))
        normals = arm_normals(meas)
        panos, _ = fs.load_results(args.run_root / city / 'results.jsonl', read_heights=False)
        sites, frame, _ = fs.fuse(arm_panos(panos, normals, 'off'), params)
        groups = [[d for d, _ in s.members if d.operational] for s in sites]
        groups = [g for g in groups if len(g) >= 2]
        by_id = {p.pano_id: p for p in panos}
        grade = {pid: abs(r['grade_deg']) for pid, r in meas.items()}
        cross = {pid: abs(r['cross_deg']) for pid, r in meas.items()}
        tilt = {pid: r['tilt_deg'] for pid, r in meas.items()}
        welldet = {pid for pid, r in meas.items() if r['pixel_share'] >= WELL_DETERMINED_SHARE}

        def place(d, arm):
            p = by_id[d.pano_id]
            pose = geo.pano_pose(p.pose_fields(**arm_fields(p, normals, arm)))
            return geo.detection_ground_point(pose, d.x, d.y, max_range_m=math.inf,
                                              errors=geo.error_model_for(p.source),
                                              apply_pose=arm != 'off')

        placed = {}   # (arm, pano, det) -> GroundEstimate or None
        for g in groups:
            for d in g:
                for arm in ABLATION_ARMS:
                    placed[(arm, d.pano_id, d.det_index)] = place(d, arm)

        def ok(g, cap, arms):
            return all(placed[(arm, d.pano_id, d.det_index)] is not None
                       and placed[(arm, d.pano_id, d.det_index)].range_m <= cap
                       for d in g for arm in arms)

        sets = {'uncapped': (math.inf, ARMS), 'capped': (geo.DEFAULT_MAX_RANGE_M, ARMS),
                'uncapped_4arm': (math.inf, ALL_ARMS),
                'capped_4arm': (geo.DEFAULT_MAX_RANGE_M, ALL_ARMS),
                'uncapped_review': (math.inf, ABLATION_ARMS),
                'capped_review': (geo.DEFAULT_MAX_RANGE_M, ABLATION_ARMS)}
        for set_name, (cap, arms) in sets.items():
            gs = [g for g in groups if ok(g, cap, arms)]
            print(f'{city}: {set_name}: {len(gs):,} of {len(groups):,} multi-member sites '
                  f'placeable under every arm', flush=True)
            for arm in arms:
                dists, norm, both_meas, wd, wd_steep = [], [], [], [], []
                by_grade, by_tilt = defaultdict(list), defaultdict(list)
                by_side, by_side_cross = defaultdict(list), defaultdict(list)
                for g in gs:
                    pts = []
                    for d in g:
                        est = placed[(arm, d.pano_id, d.det_index)]
                        pts.append((frame.to_enu(est.lat, est.lng), est.range_m, d.pano_id,
                                    d.x))
                    for i in range(len(pts)):
                        for j in range(i + 1, len(pts)):
                            (a, ra, pa, xa), (b, rb, pb, xb) = pts[i], pts[j]
                            dd = math.hypot(a[0] - b[0], a[1] - b[1])
                            dists.append(dd)
                            norm.append(dd / (0.5 * (ra + rb)))
                            if pa in welldet and pb in welldet:
                                wd.append(dd)
                                if max(grade[pa], grade[pb]) >= 2:
                                    wd_steep.append(dd)
                            if pa in grade and pb in grade:
                                both_meas.append(dd)
                                by_grade[grade_bucket(max(grade[pa], grade[pb]))].append(dd)
                                by_tilt[grade_bucket(max(tilt[pa], tilt[pb]))].append(dd)
                                side = pair_side(xa, xb)
                                by_side[side].append(dd)
                                if max(cross[pa], cross[pb]) >= CROSS_SPLIT_DEG:
                                    by_side_cross[side].append(dd)
                row = {'city': city, 'site_set': set_name, 'arm': arm, 'n_sites': len(gs),
                       'n_sites_before_intersection': len(groups), 'n_pairs': len(dists),
                       'median_pair_m': mt.pct(dists, .5),
                       'mean_pair_m': statistics.mean(dists) if dists else None,
                       'p90_pair_m': mt.pct(dists, .9),
                       'mean_pair_over_range': statistics.mean(norm) if norm else None,
                       'n_pairs_both_measured': len(both_meas),
                       'median_pair_m_both_measured': mt.pct(both_meas, .5),
                       'n_pairs_welldet': len(wd), 'median_pair_m_welldet': mt.pct(wd, .5),
                       'n_pairs_welldet_grade_2+': len(wd_steep),
                       'median_pair_m_welldet_grade_2+': mt.pct(wd_steep, .5)}
                for b in GRADE_LABELS:
                    row[f'median_pair_m_grade_{b}'] = mt.pct(by_grade[b], .5)
                    row[f'mean_pair_m_grade_{b}'] = (statistics.mean(by_grade[b])
                                                     if by_grade[b] else None)
                    row[f'p90_pair_m_grade_{b}'] = mt.pct(by_grade[b], .9)
                    row[f'n_pairs_grade_{b}'] = len(by_grade[b])
                    row[f'median_pair_m_tilt_{b}'] = mt.pct(by_tilt[b], .5)
                    row[f'n_pairs_tilt_{b}'] = len(by_tilt[b])
                cs = f'cross_{CROSS_SPLIT_DEG:g}+'
                for sd in SIDES:
                    row[f'median_pair_m_side_{sd}'] = mt.pct(by_side[sd], .5)
                    row[f'n_pairs_side_{sd}'] = len(by_side[sd])
                    row[f'median_pair_m_side_{sd}_{cs}'] = mt.pct(by_side_cross[sd], .5)
                    row[f'n_pairs_side_{sd}_{cs}'] = len(by_side_cross[sd])
                all_rows.append(row)
                print(f"{city:>11} {set_name:>8} {arm:>22}  median {row['median_pair_m']:.3f}  "
                      f"mean {row['mean_pair_m']:.3f}  p90 {row['p90_pair_m']:.3f}  norm "
                      f"{row['mean_pair_over_range']:.4f}  pairs {len(dists):,}  | by grade "
                      + '  '.join(f"{b}: {row[f'median_pair_m_grade_{b}'] or float('nan'):.2f}"
                                  f" ({row[f'n_pairs_grade_{b}']:,})" for b in GRADE_LABELS)
                      + '  | by side ' + '  '.join(
                          f"{sd}: {row[f'median_pair_m_side_{sd}'] or float('nan'):.2f}"
                          for sd in SIDES), flush=True)
    mt.write_csv(summary_dir(args.out_root) / 'ablation.csv', all_rows)


# --- eval: world P/R against RampNet GT, production cap (step 3, instrument 2) ---------

def cmd_eval(args):
    """``eval_sites.evaluate_city`` re-run per arm with the arm's angles injected into
    every SlimPano and apply_pose=True -- re-association, GT raycasting and matching all
    use the arm's ground model, under the production 25 m cap.

    Carries #50 section 4.5's caveat: GT marks are raycast under the arm too, so the
    recall pool moves; ``recall_vs_off_pool_*`` puts each arm's numerator over the off
    arm's pool. The p90 GT-to-site distance (5 m match radius) is the pre-registered
    tail statistic."""
    rows = []
    for city in args.cities:
        meas = measured(load_planes(args.out_root, city))
        normals = arm_normals(meas)
        verdict_panos, bundle_ops = mt.load_gt_files(city, args.benchmark_root)
        panos, _ = fs.load_results(args.run_root / city / 'results.jsonl', read_heights=False)
        for arm in EVAL_ARMS:
            ps_ = arm_panos(panos, normals, arm)
            params = benchmark_params(apply_pose=arm != 'off')
            prefused = fs.fuse(ps_, params)
            r5 = es.evaluate_city(verdict_panos, bundle_ops, ps_, params, match_radius_m=5.0,
                                  prefused=prefused)
            r25 = es.evaluate_city(verdict_panos, bundle_ops, ps_, params,
                                   match_radius_m=2.5, prefused=prefused)
            dists, counts = mt.gt_matched_distances(verdict_panos, bundle_ops, ps_, params,
                                                    prefused)
            row = {'city': city, 'arm': arm, 'n_sites': r5['fuse']['n_sites'],
                   'n_multi_pano_sites': r5['fuse']['n_multi_pano_sites'],
                   'gt_placeable': counts['placeable'], 'gt_unplaceable': counts['unplaceable'],
                   'pool_ramps': r5['n_pool_ramps'],
                   'world_recall_5m': r5['world_recall'],
                   'own_view_recall_5m': r5['own_view_recall'],
                   'precision_5m': r5['precision']['value'],
                   'tp_5m': r5['precision']['tp'], 'fp_5m': r5['precision']['fp'],
                   'world_recall_2p5m': r25['world_recall'],
                   'matched_5m': round(r5['world_recall'] * r5['n_pool_ramps']),
                   'matched_2p5m': round(r25['world_recall'] * r25['n_pool_ramps']),
                   'match_dist_p50': mt.pct(dists, .5), 'match_dist_p90': mt.pct(dists, .9),
                   'n_matched': len(dists)}
            rows.append(row)
            print(f"{city:>11} {arm:>22}  R5 {row['world_recall_5m']:.3f}  R2.5 "
                  f"{row['world_recall_2p5m']:.3f}  P {row['precision_5m']:.3f}  GT->site "
                  f"p50/p90 {row['match_dist_p50']:.2f}/{row['match_dist_p90']:.2f} m  pool "
                  f"{row['pool_ramps']}  unplaceable {counts['unplaceable']}  sites "
                  f"{row['n_sites']:,}", flush=True)
        pool_off = next(r['pool_ramps'] for r in rows
                        if r['city'] == city and r['arm'] == 'off')
        for r in (r for r in rows if r['city'] == city):
            r['pool_ramps_off'] = pool_off
            r['recall_vs_off_pool_5m'] = r['matched_5m'] / pool_off
            r['recall_vs_off_pool_2p5m'] = r['matched_2p5m'] / pool_off
    mt.write_csv(summary_dir(args.out_root) / 'gt_eval.csv', rows)


# --- crossslope: the slope a ray toward a ramp actually crosses (step 4) ---------------

def cmd_crossslope(args):
    """For every operational detection on a pano with a measured plane: the plane's slope
    along the detection's bearing, what an along-travel-only model (the Mapillary SfM
    grade: no cross-slope term) predicts there, and the difference -- the cross-slope
    component ``~ cross * sin(azimuth)`` that model cannot see. Then the same in meters:
    the ground point under the full plane vs under the travel-only plane (cross-slope
    zeroed), both at the production 2.6 m height, for detections production would place
    (flat range <= 25 m)."""
    summary, dets_all = [], []
    for city in args.cities:
        meas = measured(load_planes(args.out_root, city))
        panos, _ = fs.load_results(args.run_root / city / 'results.jsonl', read_heights=False)
        rows = []
        for p in panos:
            pl = meas.get(p.pano_id)
            if pl is None:
                continue
            g, c = pl['grade_deg'], pl['cross_deg']
            full = ground_frame_fields(p.lat, p.lng, p.camera_heading,
                                       (pl['n_f'], pl['n_r'], pl['n_u']))
            travel = ground_frame_fields(p.lat, p.lng, p.camera_heading,
                                         normal_from_slopes(g, 0.0))
            poses = {'flat': (geo.pano_pose(p.pose_fields(camera_pitch=None, camera_roll=None)),
                              False),
                     'full': (geo.pano_pose(p.pose_fields(**full)), True),
                     'travel': (geo.pano_pose(p.pose_fields(**travel)), True)}
            for i, x, y, conf in p.detections:
                if conf < OPERATIONAL_CONFIDENCE or on_camera_rig(y):
                    continue
                az = (x - 0.5) * 360.0
                s_full = slope_along(g, c, az)
                s_travel = slope_along(g, 0.0, az)
                est = {k: geo.detection_ground_point(pose, x, y, max_range_m=math.inf,
                                                     apply_pose=ap)
                       for k, (pose, ap) in poses.items()}
                row = {'city': city, 'pano_id': p.pano_id, 'det_index': i, 'confidence': conf,
                       'azimuth_deg': az, 'grade_deg': g, 'cross_deg': c,
                       'slope_along_bearing_deg': s_full, 'slope_travel_model_deg': s_travel,
                       'cross_component_deg': s_full - s_travel,
                       'range_flat_m': est['flat'] and est['flat'].range_m,
                       'range_full_m': est['full'] and est['full'].range_m,
                       'range_travel_m': est['travel'] and est['travel'].range_m}
                if all(est.values()) and est['flat'].range_m <= geo.DEFAULT_MAX_RANGE_M:
                    fr = geo.LocalFrame(p.lat, p.lng)
                    pts = {k: fr.to_enu(v.lat, v.lng) for k, v in est.items()}
                    row['shift_full_vs_flat_m'] = math.dist(pts['full'], pts['flat'])
                    row['shift_full_vs_travel_m'] = math.dist(pts['full'], pts['travel'])
                    row['shift_travel_vs_flat_m'] = math.dist(pts['travel'], pts['flat'])
                rows.append(row)
        out = args.out_root / city / 'ground_plane'
        out.mkdir(parents=True, exist_ok=True)
        mt.write_csv(out / 'crossslope.csv', rows)
        dets_all.extend(rows)
        cc = [abs(r['cross_component_deg']) for r in rows]
        sa = [abs(r['slope_along_bearing_deg']) for r in rows]
        st = [abs(r['slope_travel_model_deg']) for r in rows]
        placed = [r for r in rows if r.get('shift_full_vs_travel_m') is not None]
        s = {'city': city, 'n_detections': len(rows), 'n_placed_25m': len(placed),
             'abs_sin_azimuth_p50': mt.pct([abs(math.sin(math.radians(r['azimuth_deg'])))
                                             for r in rows], .5),
             'cross_component_abs_p50': mt.pct(cc, .5), 'cross_component_abs_p90': mt.pct(cc, .9),
             'slope_along_bearing_abs_p50': mt.pct(sa, .5),
             'slope_along_bearing_abs_p90': mt.pct(sa, .9),
             'slope_travel_model_abs_p50': mt.pct(st, .5),
             'slope_travel_model_abs_p90': mt.pct(st, .9),
             'pano_cross_abs_p50_label_weighted': mt.pct([abs(r['cross_deg']) for r in rows], .5),
             'frac_cross_component_gt_0p5': sum(x > 0.5 for x in cc) / len(cc) if cc else None,
             'frac_cross_component_gt_1': sum(x > 1 for x in cc) / len(cc) if cc else None}
        for k in ('full_vs_flat', 'full_vs_travel', 'travel_vs_flat'):
            v = [r[f'shift_{k}_m'] for r in placed]
            s[f'shift_{k}_p50_m'] = mt.pct(v, .5)
            s[f'shift_{k}_p90_m'] = mt.pct(v, .9)
        summary.append(s)
        print(f"{city:>11}: {len(rows):,} operational detections on measured panos; |cross "
              f"component| p50/p90 {s['cross_component_abs_p50']:.2f}/"
              f"{s['cross_component_abs_p90']:.2f} deg; |slope along bearing| p50 "
              f"{s['slope_along_bearing_abs_p50']:.2f}; ground-point shift full-vs-travel "
              f"p50/p90 {s['shift_full_vs_travel_p50_m']:.2f}/{s['shift_full_vs_travel_p90_m']:.2f} m",
              flush=True)
    mt.write_csv(summary_dir(args.out_root) / 'crossslope_summary.csv', summary)


# --- verdict ---------------------------------------------------------------------------

def summary_csv(out_root, name):
    """Aggregated CSV from the run tree, else the copy committed beside the report."""
    p = out_root / '_summary' / 'ground_plane' / name
    return mt._read_csv(p if p.exists() else FIG_DIR / 'data' / name)


def cmd_verdict(args):
    abl = summary_csv(args.out_root, 'ablation.csv')
    ev = summary_csv(args.out_root, 'gt_eval.csv')
    label, detail = verdict(abl, ev, args.cities)
    print(json.dumps({'verdict': label, **detail}, indent=2))
    with open(summary_dir(args.out_root) / 'verdict.json', 'w', encoding='utf-8') as f:
        json.dump({'verdict': label, **detail}, f, indent=2)


# --- figures ---------------------------------------------------------------------------

def cmd_figures(args):
    """Redraw every figure from the aggregated CSVs (run tree first, then the committed
    copies under docs/figures/gsv-ground-plane/data/). Figure 1 also needs the
    per-panorama planes.csv (not committed; re-run `planes`) and is skipped without it."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cities = args.cities
    colors = {'bend': '#1f77b4', 'paterson': '#ff7f0e', 'gainesville': '#2ca02c',
              'sao_paulo': '#d62728'}

    # Fig 1: grade and cross-slope distributions (needs per-pano planes.csv)
    have = [c for c in cities if planes_path(args.out_root, c).exists()]
    if len(have) == len(cities):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for city in cities:
            meas = measured(load_planes(args.out_root, city)).values()
            g = np.sort([abs(r['grade_deg']) for r in meas])
            c = np.array([r['cross_deg'] for r in meas])
            axes[0].plot(g, np.linspace(0, 1, len(g)), color=colors[city],
                         label=f'{city} (n={len(g):,})')
            ca = np.sort(np.abs(c))
            axes[1].plot(ca, np.linspace(0, 1, len(ca)), color=colors[city])
            axes[2].hist(np.clip(c, -6, 6), bins=np.linspace(-6, 6, 97), histtype='step',
                         density=True, color=colors[city], label=city)
        for ax, xl in zip(axes[:2], ('|along-travel grade| (deg)', '|cross-slope| (deg)')):
            ax.set_xlim(0, 8)
            ax.set_xlabel(xl)
            ax.set_ylabel('fraction of measured panoramas')
            ax.grid(alpha=.3)
            for b in (1, 2, 4):
                ax.axvline(b, color='k', lw=.5, ls=':')
        axes[0].legend(fontsize=7)
        axes[2].axvline(0, color='k', lw=.8)
        axes[2].set_xlabel('signed cross-slope (deg, clipped at 6; < 0 = falls to the right)')
        axes[2].set_ylabel('density')
        axes[2].legend(fontsize=7)
        fig.suptitle('The ground plane GSV depth observes under each panorama (stand-in grounds '
                     'excluded); dotted = the plan\'s grade buckets', fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig1_slope_distributions.png', dpi=150)
    else:
        print('skipping fig1: no planes.csv for', ', '.join(set(cities) - set(have)))

    # Fig 2: slope persistence along the link graph, depth plane vs rig attitude
    ch = summary_csv(args.out_root, 'chain.csv')
    labels = [f'{lo}-{hi}' for lo, hi in DIST_BINS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, sig, title in zip(axes, CHAIN_SIGNALS,
                              ('depth ground plane, every measured pano',
                               f'depth ground plane, both covering >= {WELL_DETERMINED_SHARE:.0%}',
                               'rig attitude (GSV metadata pitch, -roll)')):
        for city in cities:
            for vint, ls in (('same', '-'), ('different', '--')):
                rs = {r['dist_bin_m']: r for r in ch if r['city'] == city
                      and r['vintage'] == vint and r['signal'] == sig}
                ys = [rs[b]['r'] if b in rs and rs[b]['r'] is not None and rs[b]['n_pairs'] >= 100
                      else np.nan for b in labels]
                ax.plot(range(len(labels)), ys, ls=ls, marker='o', ms=3, color=colors[city],
                        label=city if vint == 'same' else None)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([b for b in labels], fontsize=7)
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(0, color='k', lw=.5)
        ax.set_xlabel('separation of two linked panoramas (m)')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('correlation of slope along the A->B bearing')
    axes[0].legend(fontsize=7)
    fig.suptitle('Does an observed slope persist along the street? solid = same capture month '
                 '(one drive), dashed = different months; bins >= 100 pairs', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_grade_persistence.png', dpi=150)

    # Fig 3: ablation by grade bucket, relative to off
    abl = summary_csv(args.out_root, 'ablation.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, set_name in zip(axes, ('uncapped', 'capped')):
        for city in cities:
            rs = {r['arm']: r for r in abl if r['city'] == city and r['site_set'] == set_name}
            rs4 = {r['arm']: r for r in abl if r['city'] == city
                   and r['site_set'] == f'{set_name}_4arm'}
            for arm, ls, mk, src in (('ground-normal', '-', 'o', rs),
                                     ('shuffled-normal', '--', 's', rs),
                                     (EXPLORATORY_ARM, ':', '^', rs4)):
                if arm not in src:
                    continue
                vals = []
                for b in GRADE_LABELS:
                    o = src['off'][f'median_pair_m_grade_{b}']
                    a = src[arm][f'median_pair_m_grade_{b}']
                    n = src['off'][f'n_pairs_grade_{b}'] or 0
                    vals.append(np.nan if (o is None or a is None or not o or n < MIN_BUCKET_PAIRS)
                                else a / o)
                ax.plot(range(len(GRADE_LABELS)), vals, ls=ls, marker=mk, color=colors[city],
                        label=city if arm == 'ground-normal' else None)
        ax.axhline(1.0, color='k', lw=.8)
        ax.set_xticks(range(len(GRADE_LABELS)))
        ax.set_xticklabels([f'{b}°' for b in GRADE_LABELS])
        ax.set_xlabel("pair's larger |along-travel grade|")
        ax.set_title(f'{set_name} site set', fontsize=9)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('median within-site pair distance, relative to off')
    axes[0].legend(fontsize=7)
    fig.suptitle('solid = ground-normal, dashed = shuffled-normal control, dotted = rig-attitude '
                 '(exploratory, own 4-arm site set)\nbelow 1 = tighter multi-view agreement than '
                 f'the flat raycast (buckets with >= {MIN_BUCKET_PAIRS} pairs)', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_ablation_by_grade.png', dpi=150)

    # Fig 4: GT eval per arm
    ev = summary_csv(args.out_root, 'gt_eval.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(axes, ('recall_vs_off_pool_2p5m', 'match_dist_p50', 'match_dist_p90'),
                              ('world recall @2.5 m (off pool)', 'GT-to-site distance p50 (m)',
                               'GT-to-site distance p90 (m)')):
        w = 0.8 / len(cities)
        for k, city in enumerate(cities):
            rs = {r['arm']: r for r in ev if r['city'] == city}
            arms = [a for a in EVAL_ARMS if a in rs]
            ax.bar(np.arange(len(arms)) + (k - len(cities) / 2 + .5) * w,
                   [rs[a][key] for a in arms], w, color=colors[city], label=city)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(arms, fontsize=7, rotation=20, ha='right')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3, axis='y')
    axes[0].set_ylim(0.6, 1.0)
    axes[0].legend(fontsize=7)
    fig.suptitle('Against RampNet ground truth, production 25 m cap (GT marks raycast with the '
                 'same ground model)', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_gt_eval.png', dpi=150)

    # Fig 5: cross-slope at ramp bearings
    cs = summary_csv(args.out_root, 'crossslope_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    keys = [('slope_travel_model_abs', 'along-travel model'),
            ('cross_component_abs', 'cross-slope component'),
            ('slope_along_bearing_abs', 'full slope along bearing')]
    w = 0.8 / len(cities)
    for k, city in enumerate(cities):
        r = next(r for r in cs if r['city'] == city)
        x = np.arange(len(keys)) + (k - len(cities) / 2 + .5) * w
        axes[0].bar(x, [r[f'{kk}_p50'] for kk, _ in keys], w, color=colors[city], label=city)
        axes[0].scatter(x, [r[f'{kk}_p90'] for kk, _ in keys], color=colors[city], marker='_', s=120)
        sk = [('shift_travel_vs_flat', 'travel-only vs flat'),
              ('shift_full_vs_travel', 'full vs travel-only'),
              ('shift_full_vs_flat', 'full vs flat')]
        x2 = np.arange(len(sk)) + (k - len(cities) / 2 + .5) * w
        axes[1].bar(x2, [r[f'{kk}_p50_m'] for kk, _ in sk], w, color=colors[city])
        axes[1].scatter(x2, [r[f'{kk}_p90_m'] for kk, _ in sk], color=colors[city],
                        marker='_', s=120)
    axes[0].axhline(0.5, color='k', ls='--', lw=.8)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels([l for _, l in keys], fontsize=8)
    axes[0].set_ylabel('|slope| at operational detections (deg)')
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(len(sk)))
    axes[1].set_xticklabels([l for _, l in sk], fontsize=8)
    axes[1].set_ylabel('ground-point shift (m), placeable <= 25 m')
    for ax in axes:
        ax.grid(alpha=.3, axis='y')
    fig.suptitle('Slope a ray toward a detected ramp crosses (bars = median, ticks = p90; '
                 'dashed = the 0.5° closure line from #52)', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_crossslope.png', dpi=150)
    # Fig 6: the frame check -- depth slope against the rig's metadata attitude
    rig = summary_csv(args.out_root, 'planes_vs_rig.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, axis, ylab, sign in zip(axes, ('pitch', 'roll'),
                                    ('depth along-travel grade (deg)',
                                     'depth cross-slope (deg, + rises to the right)'),
                                    (1, -1)):
        for city in cities:
            for subset, ls in (('all', '--'), ('well_determined', '-')):
                rs = sorted((r for r in rig if r['city'] == city and r['axis'] == axis
                             and r['subset'] == subset and r['n'] >= 30),
                            key=lambda r: r['meta_bin_deg'])
                ax.plot([r['meta_bin_deg'] for r in rs], [r['depth_p50'] for r in rs], ls=ls,
                        marker='o', ms=3, color=colors[city],
                        label=city if subset == 'well_determined' else None)
        ax.plot([-8, 8], [-8 * sign, 8 * sign], color='k', lw=.8, ls=':')
        ax.set_xlabel(f'GSV metadata {axis} of the capture rig (deg, 1-deg bins, n >= 30)')
        ax.set_ylabel(ylab + ', median')
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Frame check: solid = plane covers >= {WELL_DETERMINED_SHARE:.0%} of the image, '
                 'dashed = every measured pano; dotted = slope 1 (a plane riding with the car)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_frame_check.png', dpi=150)

    # Fig 7: the review arms -- travel-only against its two controls by grade bucket, and
    # the cross-slope sign by detection side
    rv = [r for r in abl if r['site_set'] == 'uncapped_review']
    if rv:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        for city in cities:
            rs = {r['arm']: r for r in rv if r['city'] == city}
            for arm, ls, mk in ((TRAVEL_ARM, '-', 'o'), ('travel-shuffled', '--', 's'),
                                ('travel-bucket-shuffled', ':', '^')):
                vals = []
                for b in GRADE_LABELS:
                    o, a = rs['off'][f'median_pair_m_grade_{b}'], rs[arm][f'median_pair_m_grade_{b}']
                    n = rs['off'][f'n_pairs_grade_{b}'] or 0
                    vals.append(np.nan if (not o or a is None or n < MIN_BUCKET_PAIRS) else a / o)
                ax.plot(range(len(GRADE_LABELS)), vals, ls=ls, marker=mk, color=colors[city],
                        label=city if arm == TRAVEL_ARM else None)
        ax.axhline(1.0, color='k', lw=.8)
        ax.set_xticks(range(len(GRADE_LABELS)))
        ax.set_xticklabels([f'{b}°' for b in GRADE_LABELS])
        ax.set_xlabel("pair's larger |along-travel grade|")
        ax.set_ylabel('median within-site pair distance, relative to off')
        ax.set_title('solid = travel-only, dashed = city-wide grade shuffle,\n'
                     'dotted = magnitude-matched (within-bucket) shuffle', fontsize=9)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        ax = axes[1]
        cs = f'cross_{CROSS_SPLIT_DEG:g}+'
        w = 0.8 / len(cities)
        labels = []
        for j, (sd, arm) in enumerate([(sd, arm) for sd in ('left', 'right')
                                       for arm in ('cross-only', 'cross-only-flipped')]):
            labels.append(f"{sd}\n{'measured' if arm == 'cross-only' else 'flipped'}")
            for k, city in enumerate(cities):
                rs = {r['arm']: r for r in rv if r['city'] == city}
                o = rs['off'][f'median_pair_m_side_{sd}_{cs}']
                a = rs[arm][f'median_pair_m_side_{sd}_{cs}']
                ax.bar(j + (k - len(cities) / 2 + .5) * w, a / o if o and a else np.nan, w,
                       color=colors[city], label=city if j == 0 else None)
        ax.axhline(1.0, color='k', lw=.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('median pair distance, relative to off')
        ax.set_title(f'cross-slope only, measured vs flipped sign, by detection side\n'
                     f'(both detections on that side; steeper |cross| >= {CROSS_SPLIT_DEG:g}°)',
                     fontsize=9)
        ax.grid(alpha=.3, axis='y')
        fig.suptitle('Review arms (uncapped_review site set; exploratory, outside the '
                     'pre-registered reading)', fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig7_review_arms.png', dpi=150)
    print('wrote figures to', FIG_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('cities', nargs='*', default=DEFAULT_CITIES)
        p.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs',
                       help='where runs/<city>/{results.jsonl,depth/} are read from (read-only)')
        p.add_argument('--out-root', type=Path, default=REPO_ROOT / 'runs',
                       help='where <city>/ground_plane/ and _summary/ground_plane/ are written')
        p.add_argument('--benchmark-root', type=Path,
                       default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    for name, fn in (('planes', cmd_planes), ('chain', cmd_chain), ('ablation', cmd_ablation),
                     ('eval', cmd_eval), ('crossslope', cmd_crossslope),
                     ('verdict', cmd_verdict), ('figures', cmd_figures)):
        p = sub.add_parser(name)
        common(p)
        if name == 'planes':
            p.add_argument('--limit', type=int, default=0,
                           help='read only the first N payloads per city (a smoke test; '
                                'writes no summary CSV)')
            p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    if getattr(args, 'cities', None) == []:
        args.cities = DEFAULT_CITIES
    args.fn(args)


if __name__ == '__main__':
    main()
