"""GSV partial pose (issue #116): does a fraction of the rig tilt improve the flat raycast?

GSV equirectangulars are in the capture rig's frame, not gravity-rectified (#113), yet
rotating rays by the full stored pose loosens multi-view agreement (#27, #52): the car
rides the road, so the local ground shares most of the rig's tilt and only a fraction of
it is a placement error. #113 measured that fraction in-sample (roughly 0.15-0.25 of the
pitch term, 0.4-0.55 of the roll term) and saw a partial correction tighten frozen-at-flat
sites. This script is the pre-registered test of that (rule posted on #116 before any
scoring run; `verdict()` below applies it).

Design, as pre-registered:

- HALF SPLIT. Each city's panos are split by a seeded hash of the pano id (SEED = 116):
  the TRAIN half fits the coefficients, the TEST half is a separate run that every arm
  is scored on. No test pano contributes to any fit.
- FIT (#113's estimator). The train half is fused flat; every pair of operational
  members of a multi-view site whose flat bearings cross at >= 30 deg fixes the ramp's
  range by bearing-only triangulation (3-30 m). The row's elevation residual
  (observed - expected at that range and the pano's camera height) is regressed by OLS
  on `pitch cos b` and `roll sin b` (streetlevel's stored sign, roll wrapped to
  [-180, 180), b the detection's azimuth from the centre column) with an intercept,
  pano-clustered standard errors. The slopes (k_pitch, k_roll) are the leaked fractions;
  the arm feeds geo._world_ray the pose (-k_pitch * pitch, +k_roll * roll). Fitted per
  city, leave-one-city-out (LOCO: the other cities' train halves pooled) and pooled.
- ARMS (TEST half): off (production); partial (the city's own train-half fit);
  partial-loco (the LOCO fit); full (-pitch, +roll); partial-shuffled / partial-loco-
  shuffled (the same coefficients on another test pano's (pitch, roll), permuted only
  within |tilt| buckets -- the magnitude-matched control of #52); partial-mirror (the
  partial fit with both signs flipped); fixed-113 ((-0.25, +0.5), #113's arm, descriptive).
- ASSOCIATION NOT FROZEN (the primary frame, `reassoc`): every arm re-fuses the test half
  under its own pose. A PAIR is two operational members (different panos) of one site; the
  common set is the pairs co-associated under every DECISION arm (off, both candidates,
  both shuffles). Each arm's distance for a pair is between its own placements of the two
  members. Sites/pairs each arm forms and keeps in the common set are reported.
  Secondary frame `frozen@off`: membership from the off fuse, every arm re-places each
  operational member, a site counts only if every arm places every member (#113's table).
- EXTERNAL REFEREE (Bend, Gainesville): the city curb-ramp inventory (#79), through
  inventory_oracle.score_frozen -- association frozen from `off`, pool anchored on off,
  every arm re-solving the same sites; plus each arm's own-association coverage.
- SURVIVORSHIP (RampNet GT, test-half judged panos): recall at 2.5 m on the OFF pool
  against each arm's own sites, unplaceable GT marks, and unplaceable operational members.
- HEIGHTS: `auto` (the fuse default, per-rig by capture year; decides) and 2.6 m.

Usage:
    python scripts/gsv_partial_pose.py fit                 # all cities, both heights
    python scripts/gsv_partial_pose.py score               # needs fit's coefficients.csv
    python scripts/gsv_partial_pose.py tieback             # #113's frozen table, full run
    python scripts/gsv_partial_pose.py verdict
    python scripts/gsv_partial_pose.py all

Inputs are read in place from runs/<city>/ (results.jsonl, depth/index.csv, and for the
inventory cities inventory_oracle/inventory.geojson); RampNet GT from --benchmark-root.
Outputs: runs/<city>/partial_pose/{report.md,*.csv}, runs/_pooled/partial_pose/.
No network, no GPU.
"""
import argparse
import csv
import hashlib
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import inventory_oracle as oracle  # noqa: E402
from agree_rate import Pt, match_one_to_one  # noqa: E402
from detectors import OPERATIONAL_CONFIDENCE, on_camera_rig  # noqa: E402

SEED = 116
CITIES = ('paterson', 'bend', 'gainesville', 'sao_paulo', 'laurens_gsv')
INVENTORY_CITIES = ('bend', 'gainesville')
HEIGHTS = (fs.HEIGHT_AUTO, geo.DEFAULT_CAMERA_HEIGHT_M)
PRIMARY_HEIGHT = fs.HEIGHT_AUTO
OUT_NAME = 'partial_pose'
POOLED_DIR = REPO_ROOT / 'runs' / '_pooled' / OUT_NAME
DEFAULT_BENCHMARK_ROOT = REPO_ROOT.parent / 'RampNet' / 'benchmark'

# The fit (#113's estimator): bearing-only triangulation of multi-view pairs.
FIT_MIN_ANGLE_DEG = 30.0
FIT_RANGE_M = (3.0, 30.0)
FIT_TIERS = (OPERATIONAL_CONFIDENCE, 0.55)   # 0.30 is used; 0.55 (#113's tier) is reported

# |tilt| = hypot(pitch, roll) buckets (deg) for the magnitude-matched shuffle.
TILT_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, math.inf)

ARM_OFF = 'off'
ARM_PARTIAL = 'partial'
ARM_LOCO = 'partial-loco'
ARM_FULL = 'full'
ARM_SHUFFLED = 'partial-shuffled'
ARM_LOCO_SHUFFLED = 'partial-loco-shuffled'
ARM_MIRROR = 'partial-mirror'
ARM_FIXED_113 = 'fixed-113'
FIXED_113 = (0.25, 0.5)            # #113's (-0.25 pitch, +0.5 roll)
ARMS = (ARM_OFF, ARM_PARTIAL, ARM_LOCO, ARM_SHUFFLED, ARM_LOCO_SHUFFLED, ARM_FULL,
        ARM_MIRROR, ARM_FIXED_113)
# The exploratory full-run reading (added 2026-09-30 AFTER the verdict, never gated): the
# whole run, LOCO coefficients only -- they never saw the city, so it stays held out.
EXPLORE_ARMS = (ARM_OFF, ARM_LOCO, ARM_LOCO_SHUFFLED, ARM_FULL, ARM_FIXED_113)
DECISION_ARMS = (ARM_OFF, ARM_PARTIAL, ARM_LOCO, ARM_SHUFFLED, ARM_LOCO_SHUFFLED)
CANDIDATES = {ARM_PARTIAL: ARM_SHUFFLED, ARM_LOCO: ARM_LOCO_SHUFFLED}  # arm -> its control
FRAME_REASSOC = 'reassoc'
FRAME_FROZEN = 'frozen@off'

# The rule (pre-registered on #116; verdict()). All at `auto`, operational tier, 25 m cap.
RULE_MIN_COMMON_PAIRS = 500        # a city is "scored" by clause (i) with at least this many
RULE_MAX_RECALL_DROP = 0.010       # (ii) off-pool recall@2.5 m, as #42
RULE_MAX_UNPLACEABLE_FRAC = 0.05   # (iii) extra unplaceable GT marks / off pool, as #42
RULE_INVENTORY_MARGIN_M = 0.10     # (iv) inventory guard, the #79 margin
INVENTORY_RADIUS_M = oracle.PRIMARY_RADIUS_M


# ------------------------------------------------------------------ split, tilt, arms

def in_test_half(pano_id, seed=SEED):
    """True for the TEST half: the low bit of sha256('<seed>:<pano_id>')."""
    return hashlib.sha256(f'{seed}:{pano_id}'.encode()).digest()[0] & 1 == 1


def split_halves(panos, seed=SEED):
    """(train, test) lists of panos, by in_test_half."""
    train, test = [], []
    for p in panos:
        (test if in_test_half(p.pano_id, seed) else train).append(p)
    return train, test


def stored_tilt(pano):
    """(pitch, roll) in streetlevel's stored sign, each wrapped to [-180, 180): GSV stores
    roll unwrapped (359.4 means -0.6). None when the pano carries no pose."""
    if pano.camera_pitch is None or pano.camera_roll is None:
        return None
    return geo.norm_deg(float(pano.camera_pitch)), geo.norm_deg(float(pano.camera_roll))


def arm_pose(pitch, roll, k_pitch, k_roll):
    """The (pitch_deg, roll_deg) geo._world_ray takes for a stored GSV (pitch, roll) under
    leak fractions (k_pitch, k_roll). Streetlevel's pitch > 0 is nose DOWN while _world_ray's
    raises the view axis, so the pitch term flips sign; roll is already Project Sidewalk's.

    Example:
        >>> arm_pose(2.0, -1.0, 1.0, 1.0)      # the full pose
        (-2.0, -1.0)
        >>> arm_pose(2.0, -1.0, 0.25, 0.5)     # #113's partial arm
        (-0.5, -0.5)
    """
    return -k_pitch * pitch, k_roll * roll


def posed_panos(panos, k_pitch, k_roll, tilts=None):
    """Copies of `panos` whose camera_pitch/camera_roll hold the arm's _world_ray pose, to
    fuse under fs.POSE_GRAVITY (which rotates by the block's angles as given). `tilts`
    ({pano_id: (pitch, roll)}) substitutes another pano's stored tilt (the shuffle).
    Unposed panos are returned unchanged (and raycast flat)."""
    out = []
    for p in panos:
        tilt = (tilts or {}).get(p.pano_id) or stored_tilt(p)
        if tilt is None:
            out.append(p)
            continue
        pitch, roll = arm_pose(tilt[0], tilt[1], k_pitch, k_roll)
        out.append(replace(p, camera_pitch=pitch, camera_roll=roll))
    return out


def tilt_bucket(pitch, roll, edges=TILT_EDGES):
    """Index of hypot(pitch, roll) in `edges` (lower edge inclusive)."""
    mag = math.hypot(pitch, roll)
    for i in range(len(edges) - 1):
        if edges[i] <= mag < edges[i + 1]:
            return i
    return len(edges) - 2


def shuffled_tilts(panos, seed=SEED, edges=TILT_EDGES):
    """({pano_id: (pitch, roll)}, n_kept_own): every posed pano's stored tilt PAIR permuted
    only among panos in the same |tilt| bucket, so the control applies corrections of the
    same size distribution (and the same mean tilt per bucket) with the pano's own tilt
    removed. Deterministic for a seed; a pano may draw its own tilt (counted)."""
    rng = random.Random(seed)
    groups = {}
    for p in sorted(panos, key=lambda q: q.pano_id):
        t = stored_tilt(p)
        if t is not None:
            groups.setdefault(tilt_bucket(*t, edges=edges), []).append((p.pano_id, t))
    out, own = {}, 0
    for b in sorted(groups):
        ids = [pid for pid, _ in groups[b]]
        vals = [t for _, t in groups[b]]
        rng.shuffle(vals)
        for pid, t, orig in zip(ids, vals, (t for _, t in groups[b])):
            out[pid] = t
            own += t == orig
    return out, own


def arm_coefficients(coefs):
    """{arm: (k_pitch, k_roll) or None (flat), shuffled?} from {'partial': (kp, kr),
    'partial-loco': (kp, kr)}."""
    kp, kr = coefs[ARM_PARTIAL]
    lp, lr = coefs[ARM_LOCO]
    return {ARM_OFF: (None, False), ARM_PARTIAL: ((kp, kr), False),
            ARM_LOCO: ((lp, lr), False), ARM_SHUFFLED: ((kp, kr), True),
            ARM_LOCO_SHUFFLED: ((lp, lr), True), ARM_FULL: ((1.0, 1.0), False),
            ARM_MIRROR: ((-kp, -kr), False), ARM_FIXED_113: (FIXED_113, False)}


def build_arms(panos, coefs, height, tier=OPERATIONAL_CONFIDENCE, arms=ARMS, seed=SEED):
    """{arm: (panos, FuseParams)} for the test half, and the shuffle's own-tilt count."""
    tilts, own = shuffled_tilts(panos, seed)
    base = fs.FuseParams(min_confidence=tier, camera_height_m=height)
    spec = arm_coefficients(coefs)
    out = {}
    for arm in arms:
        k, shuffled = spec[arm]
        if k is None:
            out[arm] = (panos, replace(base, apply_pose=fs.POSE_OFF))
        else:
            out[arm] = (posed_panos(panos, k[0], k[1], tilts if shuffled else None),
                        replace(base, apply_pose=fs.POSE_GRAVITY))
    return out, own


# --------------------------------------------------------------------------- fit

def triangulate(ci, cj, bi_deg, bj_deg, min_angle_deg=FIT_MIN_ANGLE_DEG):
    """(range from i, range from j) of the 2D intersection of two bearing rays from camera
    ENU positions ci, cj; None when they cross at less than min_angle_deg (or diverge).
    Same algebra as fuse_sites.implied_heights."""
    bi, bj = math.radians(bi_deg), math.radians(bj_deg)
    ui, uj = (math.sin(bi), math.cos(bi)), (math.sin(bj), math.cos(bj))
    cross = ui[0] * uj[1] - ui[1] * uj[0]
    if abs(cross) < math.sin(math.radians(min_angle_deg)):
        return None
    dx, dy = cj[0] - ci[0], cj[1] - ci[1]
    return (dx * uj[1] - dy * uj[0]) / cross, (dx * ui[1] - dy * ui[0]) / cross


def fit_rows(panos, height, tier=OPERATIONAL_CONFIDENCE):
    """Regression rows from a flat fuse of `panos`: [(residual_deg, pitch*cos b,
    roll*sin b, pano_id)], one per member of each triangulable operational pair."""
    params = fs.FuseParams(min_confidence=tier, camera_height_m=height,
                           apply_pose=fs.POSE_OFF)
    sites, frame, _ = fs.fuse(panos, params)
    by_id = {p.pano_id: p for p in panos}
    heights = {}
    rows = []
    lo, hi = FIT_RANGE_M
    for site in sites:
        ms = [d for d, _ in site.members if d.operational]
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                di, dj = ms[a], ms[b]
                pi, pj = by_id[di.pano_id], by_id[dj.pano_id]
                got = triangulate(frame.to_enu(pi.lat, pi.lng), frame.to_enu(pj.lat, pj.lng),
                                  di.ground.bearing_deg, dj.ground.bearing_deg)
                if got is None or not (lo < got[0] < hi and lo < got[1] < hi):
                    continue
                for d, p, r in ((di, pi, got[0]), (dj, pj, got[1])):
                    tilt = stored_tilt(p)
                    if tilt is None:
                        continue
                    h = heights.get(p.pano_id)
                    if h is None:
                        h = heights[p.pano_id] = geo.camera_height_for(
                            geo.pano_pose(p.pose_fields()), geo.error_model_for(p.source),
                            height)[0]
                    el_obs = (0.5 - d.y) * 180.0
                    el_exp = -math.degrees(math.atan2(h, r))
                    phi = (d.x - 0.5) * 2.0 * math.pi
                    rows.append((el_obs - el_exp, tilt[0] * math.cos(phi),
                                 tilt[1] * math.sin(phi), p.pano_id))
    return rows


def ols_clustered(y, X, groups):
    """OLS of y on [1, X] with cluster-robust (CR1) standard errors.
    Returns (beta, se, n_clusters); beta[0] is the intercept.

    Example (exact data recovers the slopes):
        >>> beta, se, g = ols_clustered([1, 3, 5, 7], [[0], [1], [2], [3]], [0, 0, 1, 1])
        >>> [round(b, 6) for b in beta]
        [1.0, 2.0]
    """
    import numpy as np
    y = np.asarray(y, dtype=float)
    X1 = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    xtx_inv = np.linalg.inv(X1.T @ X1)
    beta = xtx_inv @ X1.T @ y
    u = y - X1 @ beta
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    g = int(inv.max()) + 1
    scores = np.zeros((g, X1.shape[1]))
    np.add.at(scores, inv, X1 * u[:, None])
    n, k = X1.shape
    c = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    cov = c * xtx_inv @ (scores.T @ scores) @ xtx_inv
    return [float(b) for b in beta], [float(s) for s in np.sqrt(np.diag(cov))], g


def fit_coefficients(rows):
    """{'k_pitch', 'se_pitch', 'k_roll', 'se_roll', 'intercept', 'n_rows', 'n_panos'}."""
    if len(rows) < 10:
        return None
    beta, se, g = ols_clustered([r[0] for r in rows], [(r[1], r[2]) for r in rows],
                                [r[3] for r in rows])
    return {'k_pitch': beta[1], 'se_pitch': se[1], 'k_roll': beta[2], 'se_roll': se[2],
            'intercept': beta[0], 'n_rows': len(rows), 'n_panos': g}


# ---------------------------------------------------------------------- loading

def height_label(height):
    return 'auto' if height == fs.HEIGHT_AUTO else f'{float(height):g}'


def parse_height(label):
    return fs.HEIGHT_AUTO if label == 'auto' else float(label)


def load_city(city, height, runs_root):
    """(all panos with heights resolved, FuseParams height). `auto` is resolved on the FULL
    run (the production assignment), before the split."""
    path = Path(runs_root) / city / 'results.jsonl'
    if height == fs.HEIGHT_AUTO and not (path.parent / 'depth' / 'index.csv').exists():
        raise SystemExit(f'{path.parent}/depth/index.csv is missing: `auto` needs the '
                         'harvested depth heights; refusing to resolve it to 2.6 m silently')
    panos, _skipped, fuse_height, _auto = fs.load_at_height(path, height)
    return panos, fuse_height


# ------------------------------------------------------------------------ scoring

def _dist_stats(dists):
    return {'median_m': es._pct(dists, 0.5), 'p90_m': es._pct(dists, 0.9),
            'mean_m': sum(dists) / len(dists) if dists else None}


def _det_key(d):
    return (d.pano_id, d.det_index)


def site_pairs(sites):
    """{(key_a, key_b)} over pairs of operational members (different panos) of one site."""
    pairs = set()
    for s in sites:
        ms = sorted(_det_key(d) for d, _ in s.members if d.operational)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                if ms[a][0] != ms[b][0]:
                    pairs.add((ms[a], ms[b]))
    return pairs


def make_placer(arms):
    """place(arm, pano_id, x, y) -> (e, n) or None, in a frame set later; wraps
    inventory_oracle.placer (poses cached per arm)."""
    return oracle.placer(arms)


def score_reassoc(arms, fused):
    """Rows for the primary frame: pairs co-associated under every DECISION arm."""
    pairs = {arm: site_pairs(fused[arm][0]) for arm in arms}
    decision = [a for a in DECISION_ARMS if a in arms]
    common = set.intersection(*(pairs[a] for a in decision))
    place = make_placer(arms)
    frame = fused[ARM_OFF][1]
    det_xy = {}
    by_pano = {p.pano_id: p for p in arms[ARM_OFF][0]}
    for pr in common:
        for pid, i in pr:
            _, x, y, _c = by_pano[pid].detections[i]
            det_xy[(pid, i)] = (x, y)
    rows = []
    for arm in arms:
        pos, dists, unplaced = {}, [], 0
        for pr in sorted(common):
            pts = []
            for key in pr:
                if key not in pos:
                    g = place(arm, key[0], *det_xy[key])
                    pos[key] = None if g is None else frame.to_enu(g.lat, g.lng)
                pts.append(pos[key])
            if pts[0] is None or pts[1] is None:
                unplaced += 1
                continue
            dists.append(math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
        sites = fused[arm][0]
        multi = [s for s in sites if s.n_operational > 0
                 and len({d.pano_id for d, _ in s.members if d.operational}) > 1]
        in_common = sum(1 for s in multi
                        if any(pr in common for pr in site_pairs([s])))
        own = []
        for s in multi:
            ms = [d for d, _ in s.members if d.operational]
            for a in range(len(ms)):
                for b in range(a + 1, len(ms)):
                    if ms[a].pano_id != ms[b].pano_id:
                        own.append(math.hypot(ms[a].e - ms[b].e, ms[a].n - ms[b].n))
        rows.append({'frame': FRAME_REASSOC, 'arm': arm,
                     'op_sites': sum(1 for s in sites if s.n_operational > 0),
                     'multi_view_sites': len(multi), 'sites_in_common': in_common,
                     'own_pairs': len(pairs[arm]), 'common_pairs': len(common),
                     'pairs_scored': len(dists), 'pairs_unplaced': unplaced,
                     **_dist_stats(dists),
                     **{'own_' + k: v for k, v in _dist_stats(own).items()}})
    return rows


def score_frozen_pairs(arms, fused):
    """#113's table: membership from the off fuse; a multi-view site counts only if every
    arm places every operational member (25 m cap); within-site pairwise distances."""
    sites, frame = fused[ARM_OFF][:2]
    place = make_placer(arms)
    per_arm = {arm: [] for arm in arms}
    kept = dropped = 0
    for s in sites:
        ms = [d for d, _ in s.members if d.operational]
        if len({d.pano_id for d in ms}) < 2:
            continue
        placed = {}
        for arm in arms:
            gs = [place(arm, d.pano_id, d.x, d.y) for d in ms]
            if any(g is None for g in gs):
                break
            placed[arm] = [frame.to_enu(g.lat, g.lng) for g in gs]
        else:
            kept += 1
            for arm in arms:
                pts = placed[arm]
                for a in range(len(ms)):
                    for b in range(a + 1, len(ms)):
                        if ms[a].pano_id != ms[b].pano_id:
                            per_arm[arm].append(math.hypot(pts[a][0] - pts[b][0],
                                                           pts[a][1] - pts[b][1]))
            continue
        dropped += 1
    return [{'frame': FRAME_FROZEN, 'arm': arm, 'multi_view_sites': kept + dropped,
             'sites_in_common': kept, 'common_pairs': len(per_arm[arm]),
             'pairs_scored': len(per_arm[arm]), **_dist_stats(per_arm[arm])}
            for arm in arms]


def unplaceable_members(arms, tier):
    """{arm: operational detections (rig mask on) the arm cannot place inside 25 m}."""
    place = make_placer(arms)
    out = {}
    for arm, (panos, _params) in arms.items():
        n = 0
        for p in panos:
            for i, x, y, c in p.detections:
                if c >= tier and not on_camera_rig(y) and place(arm, p.pano_id, x, y) is None:
                    n += 1
        out[arm] = n
    return out


def score_gt(arms, fused, verdict_panos, bundle_ops, gt_merge_m=2.5):
    """Survivorship on RampNet GT, test-half judged panos. Per arm: unplaceable GT marks,
    recall at 2.5 / 5 m on the OFF pool (ramps grouped under off from every mark off places;
    recalled if the arm places a mark and the ramp is self-detected or one-to-one matched to
    one of the arm's OWN operational sites), and GT-to-site distance over the ramps every
    arm matches within 5 m (descriptive). Returns (rows, info)."""
    panos = arms[ARM_OFF][0]
    by_id = {p.pano_id: p for p in panos}
    test_verdicts = {pid: v for pid, v in verdict_panos.items() if pid in by_id}
    counts, warnings = es.gt_counts(), []
    marks = []
    for pid, entry, _run_pano, ops, in_pool in es.judged_gt_panos(
            test_verdicts, bundle_ops, by_id, counts, warnings):
        for verdict, (_i, x, y, _c) in zip(entry['dets'], ops):
            if verdict is True:
                marks.append((pid, x, y, 'det', in_pool))
        for mark in entry.get('missed', ()):
            if not mark.get('unsure'):
                marks.append((pid, mark['x'], mark['y'], 'missed', in_pool))
    frame = fused[ARM_OFF][1]
    place = make_placer(arms)
    enu = []
    for pid, x, y, _k, _p in marks:
        row = {}
        for arm in arms:
            g = place(arm, pid, x, y)
            row[arm] = None if g is None else frame.to_enu(g.lat, g.lng)
        enu.append(row)

    def ramps_from(ids):
        pts = [es.GTPoint(marks[i][0], marks[i][3], *enu[i][ARM_OFF], marks[i][4]) for i in ids]
        index_of = {id(pt): i for pt, i in zip(pts, ids)}
        return [(r, [index_of[id(pt)] for pt in r.points])
                for r in es.merge_gt_points(pts, gt_merge_m) if r.in_pool]

    def place_ramps(arm, ramps):
        out = []
        for k, (_r, ids) in enumerate(ramps):
            got = [enu[i][arm] for i in ids if enu[i][arm] is not None]
            out.append(None if not got else Pt(k, sum(e for e, _ in got) / len(got),
                                                 sum(n for _, n in got) / len(got)))
        return out

    off_pool = ramps_from([i for i, r in enumerate(enu) if r[ARM_OFF] is not None])
    common_ids = [i for i, r in enumerate(enu) if all(v is not None for v in r.values())]
    pool = ramps_from(common_ids)
    own_sites = {arm: [Pt(s.id, s.e, s.n) for s in fused[arm][0] if s.n_operational > 0]
                 for arm in arms}
    matched, recalled = {}, {}
    rows = []
    for arm in arms:
        placed = place_ramps(arm, pool)
        hits = match_one_to_one([p for p in placed], own_sites[arm], 5.0)
        matched[arm] = {i: math.hypot(placed[i].e - s.e, placed[i].n - s.n)
                        for i, s in hits.items()}
    common = set(range(len(pool)))
    for arm in arms:
        common &= set(matched[arm])
    for arm in arms:
        placed_off = place_ramps(arm, off_pool)
        present = [p for p in placed_off if p is not None]
        row = {'arm': arm, 'gt_panos_judged': counts['judged'], 'gt_marks': len(marks),
               'gt_marks_unplaceable': sum(1 for r in enu if r[arm] is None),
               'off_pool_ramps': len(off_pool)}
        for radius in (2.5, 5.0):
            hits = match_one_to_one(present, own_sites[arm], radius)
            hit_ids = {present[i].id for i in hits}
            got = {k for k, (r, _ids) in enumerate(off_pool)
                   if placed_off[k] is not None and (r.self_detected or k in hit_ids)}
            recalled[arm, radius] = got
            row[f'recall_off_pool_{radius:g}m'.replace('.', 'p')] = \
                len(got) / len(off_pool) if off_pool else None
        dists = [matched[arm][i] for i in sorted(common)]
        row.update({'gt_common_ramps': len(common),
                    **{'gt_to_site_' + k: v for k, v in _dist_stats(dists).items()}})
        rows.append(row)
    for row in rows:   # paired: which off-pool ramps this arm loses / gains against off
        base, mine = recalled[ARM_OFF, 2.5], recalled[row['arm'], 2.5]
        row['lost_vs_off_2p5m'] = len(base - mine)
        row['gained_vs_off_2p5m'] = len(mine - base)
    info = {'counts': counts, 'warnings': len(warnings)}
    return rows, info


def score_inventory(city, arms, fused, by_id, runs_root):
    """inventory_oracle.score_frozen anchored (membership and pool) on off, plus each arm's
    own-association match (score_own)."""
    inventory, _record = oracle.load_inventory(city)   # this checkout's runs/<city>/inventory_oracle/
    frame = fused[ARM_OFF][1]
    points = [Pt(k, *frame.to_enu(lat, lng)) for k, lat, lng, _props in inventory]
    names = tuple(arms)
    frozen, _v = oracle.score_frozen(ARM_OFF, fused, points, names, arms, by_id,
                                     oracle.RADII_M, 1.0, pool_arm=ARM_OFF)
    own = []
    for arm in names:
        own += oracle.score_own(arm, fused, points, oracle.RADII_M)
    rows = []
    for r in frozen + own:
        r = dict(r)
        r['frame'] = FRAME_FROZEN if r['frame'].startswith(oracle.FROZEN) else FRAME_REASSOC
        rows.append(r)
    return rows


def score_city(city, heights, coef_table, runs_root, benchmark_root, tier=OPERATIONAL_CONFIDENCE,
               seed=SEED, explore=False):
    """Every table for one city at each height. Returns {'pairs', 'gt', 'inventory',
    'members', 'info'} row lists.

    explore=True is the EXPLORATORY full-run reading added after the verdict (see
    EXPLORE_ARMS): every pano of the run, and only arms whose coefficients never saw this
    city (LOCO), so the run stays held out for them. It never feeds the #116 verdict."""
    out = {'pairs': [], 'gt': [], 'inventory': [], 'members': [], 'info': []}
    bench = None
    if benchmark_root and (Path(benchmark_root) / city / 'verdicts.json').exists():
        bench = es.load_benchmark(city, Path(benchmark_root))
    for height in heights:
        t0 = time.time()
        label = height_label(height)
        panos, fuse_height = load_city(city, height, runs_root)
        _train, test = split_halves(panos, seed)
        coefs = {ARM_PARTIAL: coef_pair(coef_table, label, 'city', city),
                 ARM_LOCO: coef_pair(coef_table, label, 'loco', city)}
        if explore:
            test = panos
            arms, own_tilt = build_arms(test, coefs, fuse_height, tier, arms=EXPLORE_ARMS,
                                        seed=seed)
        else:
            arms, own_tilt = build_arms(test, coefs, fuse_height, tier, seed=seed)
        fused = {}
        for arm, (ps, params) in arms.items():
            sites, frame, _st = fs.fuse(ps, params)
            fused[arm] = (sites, frame)
        frame = fused[ARM_OFF][1]
        assert all(abs(f.lat0 - frame.lat0) < 1e-12 and abs(f.lng0 - frame.lng0) < 1e-12
                   for _s, f in fused.values())
        tag = {'city': city, 'height': label}
        for r in score_reassoc(arms, fused) + score_frozen_pairs(arms, fused):
            out['pairs'].append({**tag, **r})
        members = unplaceable_members(arms, tier)
        placed_ops = sum(1 for p in test for _i, _x, y, c in p.detections
                         if c >= tier and not on_camera_rig(y))
        for arm in arms:
            out['members'].append({**tag, 'arm': arm, 'op_members': placed_ops,
                                   'unplaceable': members[arm],
                                   'unplaceable_vs_off': members[arm] - members[ARM_OFF]})
        if bench is not None:
            gt_rows, gt_info = score_gt(arms, fused, *bench)
            out['gt'] += [{**tag, **r} for r in gt_rows]
        if city in INVENTORY_CITIES:
            by_id = {p.pano_id: p for p in test}
            out['inventory'] += [{**tag, **r} for r in
                                 score_inventory(city, arms, fused, by_id, runs_root)]
        out['info'].append({**tag, 'panos': len(panos), 'test_panos': len(test),
                            'shuffle_kept_own': own_tilt,
                            'k_partial': coefs[ARM_PARTIAL], 'k_loco': coefs[ARM_LOCO],
                            'runtime_s': round(time.time() - t0, 1)})
        print(f'  {city} @ {label}: {len(test)} test panos, {time.time() - t0:.0f} s',
              file=sys.stderr)
    return out


def coef_pair(table, height_label_, scope, city):
    for r in table:
        if (r['height'] == height_label_ and r['scope'] == scope and r['city'] == city
                and float(r['tier']) == OPERATIONAL_CONFIDENCE):
            return float(r['k_pitch']), float(r['k_roll'])
    raise SystemExit(f'no {scope} coefficients for {city} @ {height_label_}: run `fit` first')


# ----------------------------------------------------------------------- tie-back

def tieback(city, runs_root, tiers=FIT_TIERS):
    """#113's frozen-at-flat table on the FULL run at 2.6 m: off / full / mirror-full /
    fixed-113. In-sample, no control -- it only ties this code to #113's numbers."""
    panos, h = load_city(city, geo.DEFAULT_CAMERA_HEIGHT_M, runs_root)
    rows = []
    for tier in tiers:
        coefs = {ARM_PARTIAL: (1.0, 1.0), ARM_LOCO: (1.0, 1.0)}   # placeholders, unused
        arms, _ = build_arms(panos, coefs, h, tier, arms=(ARM_OFF, ARM_FULL, ARM_MIRROR,
                                                         ARM_FIXED_113))
        # mirror of the FULL pose here, as in #113's table
        arms[ARM_MIRROR] = (posed_panos(panos, -1.0, -1.0), arms[ARM_FULL][1])
        sites, frame, _ = fs.fuse(*arms[ARM_OFF])
        fused = {ARM_OFF: (sites, frame)}
        for r in score_frozen_pairs(arms, fused):
            rows.append({'city': city, 'height': '2.6', 'tier': tier,
                         'arm': 'full-mirror' if r['arm'] == ARM_MIRROR else r['arm'],
                         **{k: v for k, v in r.items() if k != 'arm'}})
    return rows


# ------------------------------------------------------------------------ verdict

def _same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _get(rows, **kw):
    for r in rows:
        if all(_same(r.get(k), v) for k, v in kw.items()):
            return r
    return None


def _num(v):
    return None if v in (None, '') else float(v)


def verdict(pairs, gt, inventory, height=PRIMARY_HEIGHT, cities=CITIES,
            candidates=CANDIDATES):
    """Apply the pre-registered rule. Returns (passes, clauses, lines).

    (i)   every scored city (>= RULE_MIN_COMMON_PAIRS common pairs, reassoc frame):
          each candidate's median AND p90 pair distance strictly below off's AND below
          its own shuffled control's;
    (ii)  off-pool recall@2.5 m drops by <= 1.0 pt vs off in every city with GT;
    (iii) extra unplaceable GT marks <= 5% of the off pool in every city with GT;
    (iv)  inventory (Bend, Gainesville; frozen@off, pool on off, 5 m): median and p90
          no worse than off by more than 0.10 m.
    Passes only if all four hold for BOTH candidates (partial, partial-loco)."""
    h = height_label(height) if not isinstance(height, str) else height
    lines, fails = [], {'i': [], 'ii': [], 'iii': [], 'iv': []}
    scored = []
    for city in cities:
        off = _get(pairs, city=city, height=h, frame=FRAME_REASSOC, arm=ARM_OFF)
        if off is None:
            continue
        n = int(float(off['pairs_scored']))
        if n < RULE_MIN_COMMON_PAIRS:
            lines.append(f'{city}: {n} common pairs < {RULE_MIN_COMMON_PAIRS}: not scored by (i)')
            continue
        scored.append(city)
        for cand, ctrl in candidates.items():
            c = _get(pairs, city=city, height=h, frame=FRAME_REASSOC, arm=cand)
            s = _get(pairs, city=city, height=h, frame=FRAME_REASSOC, arm=ctrl)
            dm_off = _num(c['median_m']) - _num(off['median_m'])
            d9_off = _num(c['p90_m']) - _num(off['p90_m'])
            dm_s = _num(c['median_m']) - _num(s['median_m'])
            d9_s = _num(c['p90_m']) - _num(s['p90_m'])
            ok = dm_off < 0 and d9_off < 0 and dm_s < 0 and d9_s < 0
            if not ok:
                fails['i'].append(f'{city}/{cand}')
            lines.append(f'(i) {city} {cand}: median {dm_off:+.3f} m vs off, {dm_s:+.3f} m vs '
                         f'{ctrl}; p90 {d9_off:+.3f} vs off, {d9_s:+.3f} vs {ctrl} '
                         f'-> {"ok" if ok else "FAIL"}')
    for city in cities:
        off = _get(gt, city=city, height=h, arm=ARM_OFF)
        if off is None:
            continue
        for cand in candidates:
            c = _get(gt, city=city, height=h, arm=cand)
            drec = _num(c['recall_off_pool_2p5m']) - _num(off['recall_off_pool_2p5m'])
            extra = int(float(c['gt_marks_unplaceable'])) - int(float(off['gt_marks_unplaceable']))
            limit = RULE_MAX_UNPLACEABLE_FRAC * int(float(off['off_pool_ramps']))
            if drec < -RULE_MAX_RECALL_DROP:
                fails['ii'].append(f'{city}/{cand}')
            if extra > limit:
                fails['iii'].append(f'{city}/{cand}')
            lines.append(f'(ii/iii) {city} {cand}: off-pool R@2.5 {100 * drec:+.1f} pt; '
                         f'unplaceable GT marks {extra:+d} (limit {limit:.1f})')
    for city in INVENTORY_CITIES:
        off = _get(inventory, city=city, height=h, frame=FRAME_FROZEN, arm=ARM_OFF,
                   radius_m=INVENTORY_RADIUS_M)
        if off is None:
            fails['iv'].append(f'{city}: not scored')
            continue
        for cand in candidates:
            c = _get(inventory, city=city, height=h, frame=FRAME_FROZEN, arm=cand,
                     radius_m=INVENTORY_RADIUS_M)
            dm = _num(c['median_m']) - _num(off['median_m'])
            d9 = _num(c['p90_m']) - _num(off['p90_m'])
            if dm > RULE_INVENTORY_MARGIN_M or d9 > RULE_INVENTORY_MARGIN_M:
                fails['iv'].append(f'{city}/{cand}')
            lines.append(f'(iv) {city} {cand}: inventory median {dm:+.3f} m, p90 {d9:+.3f} m '
                         f'vs off (limit +{RULE_INVENTORY_MARGIN_M:.2f})')
    if not scored:
        fails['i'].append('no city scored')
    clauses = {k: not v for k, v in fails.items()}
    passes = all(clauses.values())
    for k in ('i', 'ii', 'iii', 'iv'):
        lines.append(f'clause ({k}): {"PASS" if clauses[k] else "FAIL"}'
                     + (f' -- {", ".join(fails[k])}' if fails[k] else ''))
    lines.append(f'VERDICT @ {h}: {"PASS" if passes else "FAIL"}')
    return passes, clauses, lines


# ------------------------------------------------------------------------- output

def _fmt(v, nd=4):
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    if isinstance(v, tuple):
        return ' '.join(_fmt(x) for x in v)
    return '' if v is None else str(v)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _md_table(rows, cols, heads=None):
    heads = heads or cols
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join(_fmt(r.get(c), 3) if isinstance(r.get(c), float)
                                     else (f'{float(r[c]):.3f}' if _isfloat(r.get(c)) else
                                           str(r.get(c, '')))
                                     for c in cols) + ' |')
    return '\n'.join(out)


def _isfloat(v):
    if not isinstance(v, str) or v == '' or v.isdigit():
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False


def city_report(city, res):
    lines = [f'# {city}: GSV partial pose (#116)', '',
             'Generated by `scripts/gsv_partial_pose.py score`. TEST half only (seed 116); '
             'operational tier 0.30, rig mask on, 25 m cap. Pair distances in metres.', '']
    for info in res['info']:
        lines.append(f"- @ {info['height']}: {info['test_panos']} of {info['panos']} panos in "
                     f"the test half; partial k = {_fmt(info['k_partial'])}, LOCO k = "
                     f"{_fmt(info['k_loco'])} (k_pitch k_roll); shuffle kept its own tilt on "
                     f"{info['shuffle_kept_own']} panos")
    for h in sorted({r['height'] for r in res['pairs']}):
        for frame in (FRAME_REASSOC, FRAME_FROZEN):
            rows = [r for r in res['pairs'] if r['height'] == h and r['frame'] == frame]
            lines += ['', f'## Within-site pair distance, {frame}, height {h}', '',
                      _md_table(rows, ['arm', 'multi_view_sites', 'sites_in_common',
                                       'pairs_scored', 'median_m', 'mean_m', 'p90_m'])]
        rows = [r for r in res['members'] if r['height'] == h]
        lines += ['', f'## Operational members not placed inside 25 m, height {h}', '',
                  _md_table(rows, ['arm', 'op_members', 'unplaceable', 'unplaceable_vs_off'])]
        rows = [r for r in res['gt'] if r['height'] == h]
        if rows:
            lines += ['', f'## RampNet GT survivorship (test-half judged panos), height {h}', '',
                      _md_table(rows, ['arm', 'gt_marks', 'gt_marks_unplaceable',
                                       'off_pool_ramps', 'recall_off_pool_2p5m',
                                       'lost_vs_off_2p5m', 'gained_vs_off_2p5m',
                                       'recall_off_pool_5m', 'gt_common_ramps',
                                       'gt_to_site_median_m', 'gt_to_site_p90_m'])]
        rows = [r for r in res['inventory'] if r['height'] == h
                and float(r['radius_m']) == INVENTORY_RADIUS_M]
        if rows:
            fz = [r for r in rows if r['frame'] == FRAME_FROZEN]
            ow = [r for r in rows if r['frame'] == FRAME_REASSOC]
            lines += ['', f'## Inventory referee, frozen@off, pool on off, 5 m, height {h}', '',
                      _md_table(fz, ['arm', 'sites_scored', 'n_pool', 'median_m', 'p90_m',
                                     'share_le_3m', 'coverage']),
                      '', f'Own association (each arm fused under itself), 5 m:', '',
                      _md_table(ow, ['arm', 'op_sites', 'own_matched', 'coverage',
                                     'own_median_m', 'own_p90_m'])]
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------ commands

def cmd_fit(args):
    table = []
    rows_by = {}
    for height in args.heights:
        label = height_label(height)
        for city in args.cities:
            panos, fuse_height = load_city(city, height, args.runs_root)
            train, _test = split_halves(panos, args.seed)
            for tier in FIT_TIERS:
                rows = fit_rows(train, fuse_height, tier)
                rows_by[label, tier, city] = [(r[0], r[1], r[2], f'{city}:{r[3]}') for r in rows]
                c = fit_coefficients(rows)
                table.append({'height': label, 'tier': tier, 'scope': 'city', 'city': city,
                              'train_panos': len(train), **(c or {})})
                print(f'fit {city} @ {label} t{tier}: {c}', file=sys.stderr)
        for tier in FIT_TIERS:
            for city in args.cities:
                rows = [r for other in args.cities if other != city
                        for r in rows_by[label, tier, other]]
                table.append({'height': label, 'tier': tier, 'scope': 'loco', 'city': city,
                              **(fit_coefficients(rows) or {})})
            rows = [r for c in args.cities for r in rows_by[label, tier, c]]
            table.append({'height': label, 'tier': tier, 'scope': 'pooled', 'city': 'all',
                          **(fit_coefficients(rows) or {})})
    write_csv(args.pooled_dir / 'coefficients.csv', table)
    print(f'wrote {args.pooled_dir / "coefficients.csv"}', file=sys.stderr)


def cmd_score(args):
    table = read_csv(args.pooled_dir / 'coefficients.csv')
    allres = {'pairs': [], 'gt': [], 'inventory': [], 'members': [], 'info': []}
    for city in args.cities:
        res = score_city(city, args.heights, table, args.runs_root, args.benchmark_root,
                         seed=args.seed)
        out = Path(args.runs_root) / city / OUT_NAME
        for key in ('pairs', 'gt', 'inventory', 'members'):
            if res[key]:
                write_csv(out / f'{key}.csv', res[key])
        (out / 'report.md').write_text(city_report(city, res), encoding='utf-8', newline='\n')
        for k in allres:
            allres[k] += res[k]
    for key in ('pairs', 'gt', 'inventory', 'members'):
        write_csv(args.pooled_dir / f'{key}.csv', allres[key])


def cmd_explore(args):
    """EXPLORATORY (added after the verdict; see score_city(explore=True))."""
    table = read_csv(args.pooled_dir / 'coefficients.csv')
    allres = {'pairs': [], 'gt': [], 'inventory': [], 'members': [], 'info': []}
    for city in args.cities:
        res = score_city(city, args.heights, table, args.runs_root, args.benchmark_root,
                         seed=args.seed, explore=True)
        out = Path(args.runs_root) / city / OUT_NAME / 'explore_full'
        for key in ('pairs', 'gt', 'inventory', 'members'):
            if res[key]:
                write_csv(out / f'{key}.csv', res[key])
        for k in allres:
            allres[k] += res[k]
    d = args.pooled_dir / 'explore_full'
    for key in ('pairs', 'gt', 'inventory', 'members'):
        write_csv(d / f'{key}.csv', allres[key])
    text = ['# EXPLORATORY: full run, LOCO coefficients (not the #116 verdict)', '',
            'The pre-registered clauses applied to the full-run LOCO arm, for reading only.', '']
    for h in args.heights:
        _p, _c, lines = verdict(_str_rows(allres['pairs']), _str_rows(allres['gt']),
                                _str_rows(allres['inventory']), height=height_label(h),
                                cities=args.cities, candidates={ARM_LOCO: ARM_LOCO_SHUFFLED})
        text += [f'## height {height_label(h)}', ''] + [f'- {ln}' for ln in lines] + ['']
    (d / 'verdict.md').write_text('\n'.join(text), encoding='utf-8', newline='\n')
    print('\n'.join(text))


def _str_rows(rows):
    return [{k: (v if isinstance(v, str) else _fmt(v)) for k, v in r.items()} for r in rows]


def cmd_tieback(args):
    rows = []
    for city in args.cities:
        rows += tieback(city, args.runs_root)
    write_csv(args.pooled_dir / 'tieback_113.csv', rows)


def cmd_verdict(args):
    d = args.pooled_dir
    pairs, gt, inv = read_csv(d / 'pairs.csv'), read_csv(d / 'gt.csv'), read_csv(d / 'inventory.csv')
    out = ['# #116 verdict (pre-registered rule)', '']
    for h in args.heights:
        passes, _clauses, lines = verdict(pairs, gt, inv, height=height_label(h),
                                          cities=args.cities)
        out += [f'## height {height_label(h)}'
                + (' (decides)' if h == PRIMARY_HEIGHT else ' (reported)'), '']
        out += [f'- {ln}' for ln in lines] + ['']
    text = '\n'.join(out)
    (d / 'verdict.md').write_text(text, encoding='utf-8', newline='\n')
    print(text)
    (d / 'summary.md').write_text(summary_markdown(d, args.cities), encoding='utf-8', newline='\n')


def _f3(v):
    return '—' if v in (None, '') else f'{float(v):.3f}'


def summary_markdown(d, cities=CITIES):
    """The cross-city tables docs/gsv-partial-pose-study.md quotes, from the pooled CSVs."""
    coef = read_csv(d / 'coefficients.csv')
    pairs, gt = read_csv(d / 'pairs.csv'), read_csv(d / 'gt.csv')
    inv, mem = read_csv(d / 'inventory.csv'), read_csv(d / 'members.csv')
    out = ['# #116 summary tables', '',
           'Regenerated by `scripts/gsv_partial_pose.py verdict` from the CSVs beside it.', '',
           '## Coefficients (tier 0.30; k = leaked fraction, SE pano-clustered)', '',
           '| height | scope | city | k_pitch (SE) | k_roll (SE) | intercept (deg) | rows | panos |',
           '|---|---|---|---|---|---|---|---|']
    for r in coef:
        if not _same(r['tier'], OPERATIONAL_CONFIDENCE) or not r.get('k_pitch'):
            continue
        out.append(f"| {r['height']} | {r['scope']} | {r['city']} | {_f3(r['k_pitch'])} "
                   f"({_f3(r['se_pitch'])}) | {_f3(r['k_roll'])} ({_f3(r['se_roll'])}) | "
                   f"{_f3(r['intercept'])} | {r['n_rows']} | {r['n_panos']} |")
    arms_shown = (ARM_OFF, ARM_PARTIAL, ARM_LOCO, ARM_SHUFFLED, ARM_LOCO_SHUFFLED, ARM_FULL,
                  ARM_MIRROR, ARM_FIXED_113)
    for h in ('auto', '2.6'):
        for frame in (FRAME_REASSOC, FRAME_FROZEN):
            out += ['', f'## Pair distance median / p90 (m), {frame}, height {h}', '',
                    '| city | pairs | ' + ' | '.join(arms_shown) + ' |',
                    '|---|---|' + '---|' * len(arms_shown)]
            for city in cities:
                row = [_get(pairs, city=city, height=h, frame=frame, arm=a) for a in arms_shown]
                if row[0] is None:
                    continue
                out.append(f'| {city} | {row[0]["pairs_scored"]} | ' + ' | '.join(
                    '—' if r is None else f'{_f3(r["median_m"])} / {_f3(r["p90_m"])}'
                    for r in row) + ' |')
        out += ['', f'## Sites formed under re-association, height {h} '
                '(multi-view sites / of them in the common set)', '',
                '| city | ' + ' | '.join(arms_shown) + ' |', '|---|' + '---|' * len(arms_shown)]
        for city in cities:
            row = [_get(pairs, city=city, height=h, frame=FRAME_REASSOC, arm=a) for a in arms_shown]
            if row[0] is None:
                continue
            out.append(f'| {city} | ' + ' | '.join(
                f'{r["multi_view_sites"]} / {r["sites_in_common"]}' for r in row) + ' |')
        out += ['', f'## Survivorship, height {h}: off-pool recall@2.5 m (lost/gained ramps '
                'vs off), unplaceable GT marks, unplaceable operational members', '',
                '| city | arm | off pool | recall@2.5 | lost / gained | GT marks unplaceable '
                '| members unplaceable (vs off) |', '|---|---|---|---|---|---|---|']
        for city in cities:
            for a in (ARM_OFF, ARM_PARTIAL, ARM_LOCO, ARM_SHUFFLED, ARM_LOCO_SHUFFLED, ARM_FULL):
                g = _get(gt, city=city, height=h, arm=a)
                m = _get(mem, city=city, height=h, arm=a)
                if g is None or m is None:
                    continue
                out.append(f"| {city} | {a} | {g['off_pool_ramps']} | "
                           f"{_f3(g['recall_off_pool_2p5m'])} | {g['lost_vs_off_2p5m']} / "
                           f"{g['gained_vs_off_2p5m']} | {g['gt_marks_unplaceable']} | "
                           f"{m['unplaceable']} ({m['unplaceable_vs_off']}) |")
        out += ['', f'## Inventory referee, height {h}: frozen@off, pool on off, 5 m '
                '(median / p90 m, share <= 3 m); own-association coverage', '',
                '| city | arm | pool | median / p90 | share <= 3 m | own coverage |',
                '|---|---|---|---|---|---|']
        for city in INVENTORY_CITIES:
            for a in arms_shown:
                fz = _get(inv, city=city, height=h, frame=FRAME_FROZEN, arm=a,
                          radius_m=INVENTORY_RADIUS_M)
                ow = _get(inv, city=city, height=h, frame=FRAME_REASSOC, arm=a,
                          radius_m=INVENTORY_RADIUS_M)
                if fz is None:
                    continue
                out.append(f"| {city} | {a} | {fz['n_pool']} | {_f3(fz['median_m'])} / "
                           f"{_f3(fz['p90_m'])} | {_f3(fz['share_le_3m'])} | "
                           f"{_f3(ow['coverage']) if ow else '—'} |")
    ex = d / 'explore_full'
    if (ex / 'pairs.csv').exists():
        ep, eg, ei = (read_csv(ex / 'pairs.csv'), read_csv(ex / 'gt.csv'),
                      read_csv(ex / 'inventory.csv'))
        out += ['', '## EXPLORATORY (added after the verdict): full run, LOCO coefficients', '',
                '| height | city | pairs | off med / p90 | loco med / p90 | loco-shuffled med / p90 '
                '| off-pool R@2.5 off -> loco (lost/gained) | inventory median / p90 off -> loco |',
                '|---|---|---|---|---|---|---|---|']
        for h in ('auto', '2.6'):
            for city in cities:
                o = _get(ep, city=city, height=h, frame=FRAME_REASSOC, arm=ARM_OFF)
                if o is None:
                    continue
                lo = _get(ep, city=city, height=h, frame=FRAME_REASSOC, arm=ARM_LOCO)
                ls = _get(ep, city=city, height=h, frame=FRAME_REASSOC, arm=ARM_LOCO_SHUFFLED)
                go, gl = (_get(eg, city=city, height=h, arm=a) for a in (ARM_OFF, ARM_LOCO))
                io_, il = (_get(ei, city=city, height=h, frame=FRAME_FROZEN, arm=a,
                                radius_m=INVENTORY_RADIUS_M) for a in (ARM_OFF, ARM_LOCO))
                rec = (f"{_f3(go['recall_off_pool_2p5m'])} -> {_f3(gl['recall_off_pool_2p5m'])} "
                       f"({gl['lost_vs_off_2p5m']}/{gl['gained_vs_off_2p5m']})") if go else '—'
                invs = (f"{_f3(io_['median_m'])} / {_f3(io_['p90_m'])} -> {_f3(il['median_m'])} / "
                        f"{_f3(il['p90_m'])}") if io_ else '—'
                out.append(f"| {h} | {city} | {o['pairs_scored']} | {_f3(o['median_m'])} / "
                           f"{_f3(o['p90_m'])} | {_f3(lo['median_m'])} / {_f3(lo['p90_m'])} | "
                           f"{_f3(ls['median_m'])} / {_f3(ls['p90_m'])} | {rec} | {invs} |")
    tb = d / 'tieback_113.csv'
    if tb.exists():
        out += ['', '## Tie-back to #113 (full run, 2.6 m, frozen@off; median / mean / p90 m)', '',
                '| city | tier | pairs | off | full | full-mirror | fixed-113 |',
                '|---|---|---|---|---|---|---|']
        rows = read_csv(tb)
        for city in cities:
            for tier in FIT_TIERS:
                rs = {r['arm']: r for r in rows if r['city'] == city and _same(r['tier'], tier)}
                if not rs:
                    continue
                out.append(f"| {city} | {tier:g} | {rs[ARM_OFF]['pairs_scored']} | " + ' | '.join(
                    f"{_f3(rs[a]['median_m'])} / {_f3(rs[a]['mean_m'])} / {_f3(rs[a]['p90_m'])}"
                    for a in (ARM_OFF, ARM_FULL, 'full-mirror', ARM_FIXED_113)) + ' |')
    return '\n'.join(out) + '\n'


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=('fit', 'score', 'tieback', 'verdict', 'explore', 'all'))
    ap.add_argument('cities', nargs='*', default=list(CITIES))
    ap.add_argument('--heights', nargs='+', default=[height_label(h) for h in HEIGHTS])
    ap.add_argument('--runs-root', default=str(REPO_ROOT / 'runs'))
    ap.add_argument('--benchmark-root', default=str(DEFAULT_BENCHMARK_ROOT))
    ap.add_argument('--out', default=None, help='pooled output dir (default runs/_pooled/partial_pose)')
    ap.add_argument('--seed', type=int, default=SEED)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.heights = [parse_height(h) for h in args.heights]
    args.pooled_dir = Path(args.out) if args.out else POOLED_DIR
    cmds = {'fit': [cmd_fit], 'score': [cmd_score], 'tieback': [cmd_tieback],
            'verdict': [cmd_verdict], 'explore': [cmd_explore], 'all': [cmd_fit, cmd_score, cmd_tieback, cmd_verdict]}
    for cmd in cmds[args.command]:
        cmd(args)


if __name__ == '__main__':
    main()
