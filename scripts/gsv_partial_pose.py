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
    python scripts/gsv_partial_pose.py figures        # docs/figures/gsv-partial-pose/

Added by the #116 follow-up (after the verdict; nothing above changes):
    python scripts/gsv_partial_pose.py consistency paterson   # production `partial` beside
                                                              # the study arms (reported)
    python scripts/gsv_partial_pose.py loss-bar               # clause (ii)'s k*(n) table
    python scripts/gsv_partial_pose.py confirm <city>         # the pre-registered
        # confirmatory run on a NEW GSV benchmark city (whole run, frozen pooled constants
        # geo.PARTIAL_POSE_K_GSV); --camera-height-m 2.6 for the reported height;
        # --exploratory to allow a #116 train city (outputs labelled EXPLORATORY)

Inputs are read in place from runs/<city>/ (results.jsonl, depth/index.csv, and for the
inventory cities inventory_oracle/inventory.geojson); RampNet GT from --benchmark-root.
Outputs: runs/<city>/partial_pose/{report.md,*.csv}, runs/_pooled/partial_pose/.
No network, no GPU.
"""
import argparse
import csv
import hashlib
import json
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


# The arm's pose IS production's (#116 follow-up): geo.partial_pitch_roll, the function
# `fuse_sites --apply-pose partial` applies, so what this study measures cannot drift from
# what fusion does (the mapillary_tilt.py precedent). Signature (pitch, roll, k_pitch, k_roll);
# its doctest carries the sign convention.
arm_pose = geo.partial_pitch_roll


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
    try:   # scored against argmax-keyed bundles (#111)
        es.require_bundle_frame(panos, path.parent)
    except ValueError as e:
        raise SystemExit(str(e))
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


def score_gt(arms, fused, verdict_panos, bundle_ops, gt_merge_m=2.5, details=None):
    """Survivorship on RampNet GT, test-half judged panos. Per arm: unplaceable GT marks,
    recall at 2.5 / 5 m on the OFF pool (ramps grouped under off from every mark off places;
    recalled if the arm places a mark and the ramp is self-detected or one-to-one matched to
    one of the arm's OWN operational sites), and GT-to-site distance over the ramps every
    arm matches within 5 m (descriptive). Returns (rows, info).

    `details`, when a list, receives one dict per off-pool ramp (its marks, whether it is
    self-detected, and per arm its placement, nearest own operational site, one-to-one
    match distance at 2.5 m and whether it was recalled). Only `figures` passes it; the
    returned rows are identical either way."""
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
    hit_dist, placed_by_arm = {}, {}
    for arm in arms:
        placed_off = place_ramps(arm, off_pool)
        present = [p for p in placed_off if p is not None]
        row = {'arm': arm, 'gt_panos_judged': counts['judged'], 'gt_marks': len(marks),
               'gt_marks_unplaceable': sum(1 for r in enu if r[arm] is None),
               'off_pool_ramps': len(off_pool)}
        for radius in (2.5, 5.0):
            hits = match_one_to_one(present, own_sites[arm], radius)
            hit_ids = {present[i].id for i in hits}
            if radius == 2.5:
                hit_dist[arm] = {present[i].id: math.hypot(present[i].e - s.e, present[i].n - s.n)
                                 for i, s in hits.items()}
                placed_by_arm[arm] = placed_off
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
    if details is not None:
        for k, (r, ids) in enumerate(off_pool):
            entry = {'ramp': k, 'self_detected': bool(r.self_detected),
                     'marks': [marks[i][:4] for i in ids], 'arms': {}}
            for arm in arms:
                pt = placed_by_arm[arm][k]
                near = None if pt is None or not own_sites[arm] else min(
                    math.hypot(pt.e - s.e, pt.n - s.n) for s in own_sites[arm])
                entry['arms'][arm] = {
                    'e': None if pt is None else pt.e, 'n': None if pt is None else pt.n,
                    'nearest_site_m': near, 'match_2p5m': hit_dist[arm].get(k),
                    'recalled_2p5m': k in recalled[arm, 2.5]}
            details.append(entry)
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


# ------------------------------------------------- production arm + confirmatory run
#
# Added by the #116 follow-up (2026-09-30), after the verdict: `--apply-pose partial` is
# wired into fuse_sites as an OPT-IN mode with the frozen pooled constants
# (geo.PARTIAL_POSE_K_GSV), and the confirmatory run below is pre-registered on #116
# before any new data is looked at. Neither changes the #116 verdict or its outputs.

ARM_POOLED = 'partial-pooled'   # production's --apply-pose partial (frozen pooled constants)
CONFIRM_ARMS = (ARM_OFF, ARM_PARTIAL, ARM_SHUFFLED, ARM_MIRROR)
CONFIRM_CANDIDATES = {ARM_PARTIAL: ARM_SHUFFLED}
# (ii), re-sized to the pool: FAIL iff (lost - gained) >= recall_loss_bar(n). The 1.0 pt
# limit the study used, read as a per-ramp loss RATE, at one-sided alpha 0.05.
CONFIRM_LOSS_RATE = 0.01
CONFIRM_ALPHA = 0.05
CONFIRM_MIN_POOL = 50           # below this, a (ii) pass reads INCONCLUSIVE (a FAIL stands)
CONFIRM_MIN_VINTAGE_ROWS = 50   # per-vintage k is reported only with this many fit rows


def recall_loss_bar(n, rate=CONFIRM_LOSS_RATE, alpha=CONFIRM_ALPHA):
    """k*(n): the smallest k with P(Binomial(n, rate) >= k) <= alpha (exact, one-sided).
    Clause (ii) of the confirmatory run FAILS iff ramps lost - ramps gained >= k*(n),
    where n is the off-pool recall denominator at 2.5 m.

    Example:
        >>> [recall_loss_bar(n) for n in (100, 157, 300)]
        [4, 5, 7]
    """
    if n <= 0:
        return 1
    pmf = (1.0 - rate) ** n        # P(X = 0)
    cdf = 0.0                      # P(X <= k - 1)
    for k in range(0, n + 1):
        if 1.0 - cdf <= alpha:     # P(X >= k)
            return k
        cdf += pmf
        pmf *= (n - k) / (k + 1) * rate / (1.0 - rate)
    return n + 1


def recall_loss_bar_table(lo=50, hi=800, rate=CONFIRM_LOSS_RATE, alpha=CONFIRM_ALPHA):
    """[(n_first, n_last, k*)] runs of equal k*(n) over n in [lo, hi]."""
    out = []
    for n in range(lo, hi + 1):
        k = recall_loss_bar(n, rate, alpha)
        if out and out[-1][2] == k:
            out[-1] = (out[-1][0], n, k)
        else:
            out.append((n, n, k))
    return out


def production_pose_mismatches(panos, k=geo.PARTIAL_POSE_K_GSV):
    """Pano ids whose study-built pose (posed_panos under POSE_GRAVITY, the path every #116
    arm used) differs from production's fs.pano_pose(p, POSE_PARTIAL). Empty means the
    study's arm and the shipped mode raycast identically."""
    bad = []
    for p, q in zip(panos, posed_panos(panos, *k)):
        a = fs.pano_pose(q, fs.POSE_GRAVITY)
        b = fs.pano_pose(p, fs.POSE_PARTIAL)
        if (a.pitch_deg, a.roll_deg, a.has_pitch_roll) != (b.pitch_deg, b.roll_deg,
                                                         b.has_pitch_roll):
            bad.append(p.pano_id)
    return bad


def _fuse_arms(arms):
    fused = {}
    for arm, (ps, params) in arms.items():
        sites, frame, _st = fs.fuse(ps, params)
        fused[arm] = (sites, frame)
    frame = fused[ARM_OFF][1]
    assert all(abs(f.lat0 - frame.lat0) < 1e-12 and abs(f.lng0 - frame.lng0) < 1e-12
               for _s, f in fused.values())
    return fused


def consistency_city(city, heights, coef_table, runs_root, benchmark_root,
                     tier=OPERATIONAL_CONFIDENCE, seed=SEED):
    """The study's TEST-half arms plus ARM_POOLED -- production's `--apply-pose partial`
    on the unmodified panos -- re-fused and scored with the same machinery, seed and
    common pairs (the common set is the DECISION arms', so adding an arm moves nothing
    else). Also checks production_pose_mismatches is empty on the whole run.
    Returns {'pairs', 'gt', 'info'} (reassoc frame only: the frozen frame's site set is
    'every arm places every member', so an extra arm would change it)."""
    out = {'pairs': [], 'gt': [], 'info': []}
    bench = None
    if benchmark_root and (Path(benchmark_root) / city / 'verdicts.json').exists():
        bench = es.load_benchmark(city, Path(benchmark_root))
    for height in heights:
        label = height_label(height)
        panos, fuse_height = load_city(city, height, runs_root)
        bad = production_pose_mismatches(panos)
        if bad:
            raise SystemExit(f'{city}: {len(bad)} panos pose differently under production '
                             f'`partial` than under the study arm, e.g. {bad[:3]}')
        _train, test = split_halves(panos, seed)
        coefs = {ARM_PARTIAL: coef_pair(coef_table, label, 'city', city),
                 ARM_LOCO: coef_pair(coef_table, label, 'loco', city)}
        arms, _own = build_arms(test, coefs, fuse_height, tier, seed=seed)
        arms[ARM_POOLED] = (test, replace(arms[ARM_OFF][1], apply_pose=fs.POSE_PARTIAL))
        fused = _fuse_arms(arms)
        tag = {'city': city, 'height': label}
        out['pairs'] += [{**tag, **r} for r in score_reassoc(arms, fused)]
        if bench is not None:
            rows, _info = score_gt(arms, fused, *bench)
            out['gt'] += [{**tag, **r} for r in rows]
        out['info'].append({**tag, 'panos': len(panos), 'test_panos': len(test),
                            'pose_checked_panos': len(panos),
                            'k_partial': coefs[ARM_PARTIAL], 'k_loco': coefs[ARM_LOCO],
                            'k_pooled': geo.PARTIAL_POSE_K_GSV})
        print(f'  consistency {city} @ {label}: {len(test)} test panos, '
              f'{len(panos)} poses checked', file=sys.stderr)
    return out


def consistency_mismatches(new_rows, committed_rows):
    """[(key, column, committed, new)] where a row of the consistency run differs from the
    committed #116 row for the same (height, frame, arm), over every column both carry.
    Rows only one side has (partial-pooled; the frozen frame) are not compared. Returns
    (n_rows_compared, mismatches)."""
    def key(r):
        return (r['height'], r.get('frame', ''), r['arm'])
    old = {key(r): r for r in committed_rows}
    n, bad = 0, []
    for r in new_rows:
        o = old.get(key(r))
        if o is None:
            continue
        n += 1
        bad += [(key(r), c, o[c], r[c]) for c in r if c in o and o[c] != r[c]]
    return n, bad


def cmd_consistency(args):
    """Consistency row (#116 follow-up; reported, gates nothing): production's pooled
    `partial` beside the study's per-city and LOCO arms on the same TEST half."""
    table = read_csv(args.pooled_dir / 'coefficients.csv')
    for city in args.cities:
        res = consistency_city(city, args.heights, table, args.runs_root, args.benchmark_root,
                               seed=args.seed)
        out = REPO_ROOT / 'runs' / city / OUT_NAME
        write_csv(out / 'consistency_pairs.csv', res['pairs'])
        if res['gt']:
            write_csv(out / 'consistency_gt.csv', res['gt'])
        lines = [f'# {city}: production `--apply-pose partial` vs the #116 arms', '',
                 'Generated by `scripts/gsv_partial_pose.py consistency`. TEST half (seed 116), '
                 'tier 0.30, re-associated per arm, on the pairs every #116 decision arm '
                 "co-associates. `partial-pooled` is production's mode (frozen pooled "
                 f'constants {geo.PARTIAL_POSE_K_GSV}); reported, gates nothing.', '']
        for info in res['info']:
            lines.append(f"- @ {info['height']}: production pose == study arm pose on all "
                         f"{info['pose_checked_panos']} panos of the run; partial k = "
                         f"{_fmt(info['k_partial'])}, LOCO k = {_fmt(info['k_loco'])}, pooled "
                         f"k = {_fmt(info['k_pooled'])}")
        shown = (ARM_OFF, ARM_PARTIAL, ARM_LOCO, ARM_POOLED, ARM_SHUFFLED)
        for h in args.heights:
            label = height_label(h)
            rows = [r for r in _str_rows(res['pairs']) if r['height'] == label
                    and r['arm'] in shown]
            lines += ['', f'## Pair distance, reassoc, height {label}', '',
                      _md_table(rows, ['arm', 'multi_view_sites', 'pairs_scored', 'median_m',
                                       'mean_m', 'p90_m'])]
            rows = [r for r in _str_rows(res['gt']) if r['height'] == label
                    and r['arm'] in shown]
            if rows:
                lines += ['', f'## Off-pool recall, height {label}', '',
                          _md_table(rows, ['arm', 'off_pool_ramps', 'recall_off_pool_2p5m',
                                           'lost_vs_off_2p5m', 'gained_vs_off_2p5m',
                                           'gt_marks_unplaceable'])]
        (out / 'consistency.md').write_text('\n'.join(lines) + '\n', encoding='utf-8',
                                            newline='\n')
        print('\n'.join(lines))
        # The automated half of the check: every study-arm row must reproduce the
        # committed #116 outputs (pairs.csv, gt.csv) column for column.
        failed = False
        for new, committed in (('consistency_pairs.csv', 'pairs.csv'),
                               ('consistency_gt.csv', 'gt.csv')):
            if not (out / new).exists() or not (out / committed).exists():
                print(f'{city}: {new} or the committed {committed} is missing: cannot check',
                      file=sys.stderr)
                failed = True
                continue
            n, bad = consistency_mismatches(read_csv(out / new), read_csv(out / committed))
            print(f'{city}: {new} vs committed {committed}: {n} rows, {len(bad)} mismatches',
                  file=sys.stderr)
            failed |= bool(bad) or n == 0
        if failed:
            raise SystemExit(f'{city}: the consistency run does not reproduce #116\'s rows')


def vintage_coefficients(panos, height, tier=OPERATIONAL_CONFIDENCE,
                         min_rows=CONFIRM_MIN_VINTAGE_ROWS):
    """Descriptive only (never scored): #116's fit on the WHOLE confirm run, overall and
    per capture year. A group below min_rows reports its row count and no slopes."""
    rows = fit_rows(panos, height, tier)
    year = {p.pano_id: (p.capture_date or '')[:4] or 'undated' for p in panos}
    groups = {'all': rows}
    for r in rows:
        groups.setdefault(year[r[3]], []).append(r)
    out = []
    for key in ['all'] + sorted(k for k in groups if k != 'all'):
        rs = groups[key]
        c = fit_coefficients(rs) if len(rs) >= min_rows else None
        out.append({'vintage': key, 'n_rows': len(rs),
                    'n_panos': len({r[3] for r in rs}), **(c or {})})
    return out


def confirm_verdict(pairs, gt, inventory, candidates=CONFIRM_CANDIDATES,
                    inventory_cities=INVENTORY_CITIES):
    """The confirmatory rule (pre-registered on #116, 2026-09-30). One city, one height.

    (i)   >= RULE_MIN_COMMON_PAIRS common pairs (reassoc): the candidate's median AND p90
          strictly below off's AND below its magnitude-matched shuffle's. Fewer pairs
          -> INCONCLUSIVE, never a pass.
    (ii)  L = ramps lost - ramps gained vs off (off pool, 2.5 m); FAIL iff
          L >= recall_loss_bar(n), n = the off-pool denominator. Below
          CONFIRM_MIN_POOL a FAIL stands; only a would-be pass becomes INCONCLUSIVE.
    (iii) extra unplaceable GT marks <= 5% of the off pool.
    (iv)  Bend and Gainesville inventories (frozen@off, pool on off, 5 m): median and p90
          no worse than off by more than 0.10 m.
    Returns (outcome in {'PASS', 'FAIL', 'INCONCLUSIVE'}, {clause: True/False/None}, lines)."""
    lines, state = [], {}
    off = _get(pairs, frame=FRAME_REASSOC, arm=ARM_OFF)
    n_pairs = int(float(off['pairs_scored']))
    if n_pairs < RULE_MIN_COMMON_PAIRS:
        state['i'] = None
        lines.append(f'(i) {n_pairs} common pairs < {RULE_MIN_COMMON_PAIRS}: INCONCLUSIVE')
    else:
        ok_all = True
        for cand, ctrl in candidates.items():
            c = _get(pairs, frame=FRAME_REASSOC, arm=cand)
            s = _get(pairs, frame=FRAME_REASSOC, arm=ctrl)
            d = {k: _num(c[k]) - _num(off[k]) for k in ('median_m', 'p90_m')}
            e = {k: _num(c[k]) - _num(s[k]) for k in ('median_m', 'p90_m')}
            ok = all(v < 0 for v in (*d.values(), *e.values()))
            ok_all = ok_all and ok
            lines.append(f"(i) {cand} ({n_pairs} pairs): median {d['median_m']:+.3f} m vs off, "
                         f"{e['median_m']:+.3f} m vs {ctrl}; p90 {d['p90_m']:+.3f} vs off, "
                         f"{e['p90_m']:+.3f} vs {ctrl} -> {'ok' if ok else 'FAIL'}")
        state['i'] = ok_all
    goff = _get(gt, arm=ARM_OFF)
    n = int(float(goff['off_pool_ramps']))
    bar = recall_loss_bar(n)
    ok2 = ok3 = True
    for cand in candidates:
        c = _get(gt, arm=cand)
        lost, gained = int(float(c['lost_vs_off_2p5m'])), int(float(c['gained_vs_off_2p5m']))
        loss = lost - gained
        ok = loss < bar
        ok2 = ok2 and ok
        lines.append(f'(ii) {cand}: n = {n}, lost {lost}, gained {gained}, L = {loss:+d}; '
                     f'k*(n) = {bar} -> {"ok" if ok else "FAIL"}')
        extra = int(float(c['gt_marks_unplaceable'])) - int(float(goff['gt_marks_unplaceable']))
        limit = RULE_MAX_UNPLACEABLE_FRAC * n
        ok3 = ok3 and extra <= limit
        lines.append(f'(iii) {cand}: unplaceable GT marks {extra:+d} (limit {limit:.1f}) -> '
                     f'{"ok" if extra <= limit else "FAIL"}')
    if ok2 and n < CONFIRM_MIN_POOL:
        ok2 = None
        lines.append(f'(ii) off pool n = {n} < {CONFIRM_MIN_POOL}: INCONCLUSIVE')
    state['ii'], state['iii'] = ok2, ok3
    ok4 = True
    for city in inventory_cities:
        ioff = _get(inventory, city=city, frame=FRAME_FROZEN, arm=ARM_OFF,
                    radius_m=INVENTORY_RADIUS_M)
        if ioff is None:
            ok4 = False
            lines.append(f'(iv) {city}: not scored -> FAIL')
            continue
        for cand in candidates:
            c = _get(inventory, city=city, frame=FRAME_FROZEN, arm=cand,
                     radius_m=INVENTORY_RADIUS_M)
            dm = _num(c['median_m']) - _num(ioff['median_m'])
            d9 = _num(c['p90_m']) - _num(ioff['p90_m'])
            ok = dm <= RULE_INVENTORY_MARGIN_M and d9 <= RULE_INVENTORY_MARGIN_M
            ok4 = ok4 and ok
            lines.append(f'(iv) {city} {cand}: inventory median {dm:+.3f} m, p90 {d9:+.3f} m '
                         f'vs off (limit +{RULE_INVENTORY_MARGIN_M:.2f}) -> '
                         f'{"ok" if ok else "FAIL"}')
    state['iv'] = ok4
    if any(v is False for v in state.values()):
        outcome = 'FAIL'
    elif any(v is None for v in state.values()):
        outcome = 'INCONCLUSIVE'
    else:
        outcome = 'PASS'
    for k in ('i', 'ii', 'iii', 'iv'):
        v = state[k]
        lines.append(f'clause ({k}): ' + ('INCONCLUSIVE' if v is None else
                                          'PASS' if v else 'FAIL'))
    return outcome, state, lines


# STORE-POSE GATE (review of PR #123). A run rebuilt from the PS pano store
# (detect_from_store.py; Vancouver) carries the PS pano row's pitch/roll, whose convention
# against streetlevel's -- the angles PARTIAL_POSE_K_GSV was fit on -- is unverified; a
# sign flip would make `partial` behave as the mirror arm. fuse_sites therefore raycasts
# such panos flat under `partial`. `confirm` may score a store-built city only after this
# gate: a seeded sample of its store panos is resolved live through streetlevel (METADATA
# only, sources/gsv.fetch_metadata_with_retry; no imagery) and the PS angles are compared
# with streetlevel's on the same ids under each of the four sign mappings. The gate passes
# only if one mapping agrees within GATE_TOL_DEG on BOTH angles for >= GATE_MIN_AGREE of
# at least GATE_MIN_COMPARED comparable panos; confirm then applies that mapping and
# re-labels the blocks fs.STORE_POSE_VERIFIED_DETAIL. This gate is what makes a
# store-built city (Vancouver) eligible for the confirmatory run.
GATE_SAMPLE = 200
GATE_TOL_DEG = 0.1
GATE_MIN_AGREE = 0.95
GATE_MIN_COMPARED = 50
GATE_SPACING_S = 0.2
SIGN_MAPPINGS = ((1, 1), (1, -1), (-1, 1), (-1, -1))   # (pitch sign, roll sign)


def streetlevel_pitch_roll(pano_id):
    """(pitch_deg, roll_deg) streetlevel reports for a pano id -- the convention
    sources/gsv.build_pano_record stores -- or None when it no longer resolves.
    Metadata only (the same call main.py makes; no imagery)."""
    from sources import gsv as gsv_source
    metadata, _depth = gsv_source.fetch_metadata_with_retry(pano_id)
    if metadata is None:
        return None
    return math.degrees(metadata.pitch), math.degrees(metadata.roll)


def store_panos(panos):
    return [p for p in panos if p.source_detail == fs.STORE_SOURCE_DETAIL]


def store_pose_gate(panos, fetch=streetlevel_pitch_roll, n=GATE_SAMPLE, seed=SEED,
                    tol=GATE_TOL_DEG, spacing_s=GATE_SPACING_S):
    """The store-pose convention gate (see GATE_SAMPLE). Returns a JSON-able dict with
    `status` in {'not_required', 'pass', 'fail', 'skipped'} and, on a pass, `mapping`
    (pitch sign, roll sign) to apply to the PS angles. `fetch(pano_id)` returns
    streetlevel's (pitch, roll) or None; an exception from it counts as a network error,
    and a sample where every fetch raised is `skipped` (no network), which confirm treats
    as a refusal."""
    store = sorted(store_panos(panos), key=lambda p: p.pano_id)
    out = {'store_panos': len(store), 'run_panos': len(panos), 'sample_seed': seed,
           'tol_deg': tol, 'min_agree': GATE_MIN_AGREE, 'min_compared': GATE_MIN_COMPARED}
    if not store:
        return {**out, 'status': 'not_required',
                'reason': "no store-built panos: every angle is streetlevel's own"}
    sample = random.Random(seed).sample(store, min(n, len(store)))
    rows, errors = [], 0
    for i, p in enumerate(sample):
        if i and spacing_s:
            time.sleep(spacing_s)
        try:
            got = fetch(p.pano_id)
            err = None
        except Exception as e:           # network / parse: recorded, never fatal here
            got, err = None, f'{type(e).__name__}: {e}'
            errors += 1
        rows.append({'pano_id': p.pano_id, 'ps_pitch': p.camera_pitch, 'ps_roll': p.camera_roll,
                     'sl_pitch': None if got is None else got[0],
                     'sl_roll': None if got is None else got[1], 'error': err})
    out.update({'sample': len(sample), 'fetch_errors': errors,
                'streetlevel_unresolved': sum(1 for r in rows if r['sl_pitch'] is None
                                              and r['error'] is None),
                'ps_null_pitch_share': sum(r['ps_pitch'] is None for r in rows) / len(rows),
                'ps_null_roll_share': sum(r['ps_roll'] is None for r in rows) / len(rows),
                'ps_roll_unwrapped_share': sum(
                    1 for r in rows if r['ps_roll'] is not None
                    and not -180.0 <= float(r['ps_roll']) < 180.0) / len(rows),
                'rows': rows})
    if errors == len(rows):
        return {**out, 'status': 'skipped',
                'reason': 'no streetlevel metadata reachable (every fetch raised; no network?)'}
    comp = [r for r in rows if None not in (r['ps_pitch'], r['ps_roll'], r['sl_pitch'],
                                            r['sl_roll'])]
    agree = {}
    for sp, sr in SIGN_MAPPINGS:
        ok = sum(1 for r in comp
                 if abs(geo.norm_deg(sp * float(r['ps_pitch']) - float(r['sl_pitch']))) <= tol
                 and abs(geo.norm_deg(sr * float(r['ps_roll']) - float(r['sl_roll']))) <= tol)
        agree[f'{sp:+d}{sr:+d}'] = ok / len(comp) if comp else None
    best = max(SIGN_MAPPINGS, key=lambda m: agree[f'{m[0]:+d}{m[1]:+d}'] or 0.0)
    share = agree[f'{best[0]:+d}{best[1]:+d}']
    # The mapping must be UNIQUE: two mappings over the bar (e.g. angles all within the
    # tolerance of zero) cannot tell the sign, so the gate fails rather than taking the first.
    over = [m for m in SIGN_MAPPINGS
            if (agree[f'{m[0]:+d}{m[1]:+d}'] or 0.0) >= GATE_MIN_AGREE]
    passed = (len(comp) >= GATE_MIN_COMPARED and share is not None
              and share >= GATE_MIN_AGREE and len(over) == 1)
    out.update({'compared': len(comp), 'agreement_by_mapping': agree,
                'best_mapping': list(best), 'best_agreement': share,
                'mappings_over_bar': [list(m) for m in over],
                'status': 'pass' if passed else 'fail'})
    if passed:
        out['mapping'] = list(best)
    elif len(over) > 1:
        out['reason'] = (f'{len(over)} sign mappings clear the {GATE_MIN_AGREE} bar '
                         f'({over}): the sign is not identified')
    else:
        out['reason'] = (f'{len(comp)} comparable panos (need {GATE_MIN_COMPARED}); best '
                         f'mapping {best} agrees on {share} (need {GATE_MIN_AGREE})')
    return out


def apply_store_mapping(panos, mapping):
    """Store-built panos with their PS angles put into streetlevel's convention (the
    gate's sign mapping) and re-labelled fs.STORE_POSE_VERIFIED_DETAIL, so `partial`
    poses them; every other pano is returned unchanged."""
    sp, sr = mapping
    out = []
    for p in panos:
        if p.source_detail != fs.STORE_SOURCE_DETAIL:
            out.append(p)
            continue
        out.append(replace(
            p, camera_pitch=None if p.camera_pitch is None else sp * float(p.camera_pitch),
            camera_roll=None if p.camera_roll is None else sr * float(p.camera_roll),
            source_detail=fs.STORE_POSE_VERIFIED_DETAIL))
    return out


def check_confirm_city(city, exploratory, seed):
    """Refusals for `confirm` before anything is loaded: a path-like name, a #116 train
    city (case-insensitively: on Windows/macOS `Bend` resolves runs/bend/) without
    --exploratory, and an unfrozen seed without --exploratory."""
    if not city or any(c in city for c in ('/', '\\', ':')) or '..' in city:
        raise SystemExit(f'{city!r} is not a city name (no paths)')
    if city.lower() in CITIES and not exploratory:
        raise SystemExit(f"{city} was in #116's train set ({', '.join(CITIES)}): its panos "
                         'fitted the constants, so it cannot confirm them. Pass '
                         '--exploratory to run it anyway (every output is labelled so)')
    if seed != SEED and not exploratory:
        raise SystemExit(f'the confirmatory run is pre-registered at seed {SEED}; --seed '
                         f'{seed} is only allowed with --exploratory')


def confirm_arms(panos, height, tier=OPERATIONAL_CONFIDENCE, seed=SEED,
                 k=geo.PARTIAL_POSE_K_GSV):
    """{arm: (panos, FuseParams)} for the confirmatory run: off and partial through
    PRODUCTION's modes on the unmodified panos; the shuffle and the mirror built as #116
    built them (posed_panos under POSE_GRAVITY), with the frozen constants."""
    tilts, own = shuffled_tilts(panos, seed)
    base = fs.FuseParams(min_confidence=tier, camera_height_m=height)
    grav = replace(base, apply_pose=fs.POSE_GRAVITY)
    return {ARM_OFF: (panos, replace(base, apply_pose=fs.POSE_OFF)),
            ARM_PARTIAL: (panos, replace(base, apply_pose=fs.POSE_PARTIAL)),
            ARM_SHUFFLED: (posed_panos(panos, k[0], k[1], tilts), grav),
            ARM_MIRROR: (posed_panos(panos, -k[0], -k[1]), grav)}, own


def cmd_confirm(args):
    """The pre-registered confirmatory run (#116): one NEW GSV benchmark city, whole run,
    frozen pooled constants. See confirm_verdict for the rule."""
    if len(args.cities) != 1 or args.cities == list(CITIES):
        raise SystemExit('confirm takes exactly one city')
    city = args.cities[0]
    exploratory = args.exploratory
    check_confirm_city(city, exploratory, args.seed)
    bench_root = Path(args.benchmark_root)
    if not (bench_root / city / 'verdicts.json').exists():
        raise SystemExit(f'{bench_root / city}/verdicts.json is missing: clauses (ii) and '
                         '(iii) need RampNet GT for the city')
    height = parse_height(args.camera_height_m)
    label = height_label(height)
    deciding = height == PRIMARY_HEIGHT and not exploratory
    tag_x = 'EXPLORATORY ' if exploratory else ''
    out = REPO_ROOT / 'runs' / city / OUT_NAME / (
        f'confirm_exploratory_{label}' + (f'_seed{args.seed}' if args.seed != SEED else '')
        if exploratory else f'confirm_{label}')
    t0 = time.time()
    panos, fuse_height = load_city(city, height, args.runs_root)
    gate = store_pose_gate(panos)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'store_pose_gate.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(gate, f, indent=1)
    print(f"  store-pose gate: {gate['status']}"
          + (f" ({gate.get('reason')})" if gate.get('reason') else ''), file=sys.stderr)
    if gate['status'] in ('fail', 'skipped'):
        raise SystemExit(f"{city}: {gate['store_panos']} store-built panos and the store-pose "
                         f"gate did not pass ({gate['status']}: {gate.get('reason')}); "
                         f'refusing to score. See {out / "store_pose_gate.json"}')
    if gate['status'] == 'pass':
        panos = apply_store_mapping(panos, gate['mapping'])
    bad = production_pose_mismatches(panos)
    if bad:
        raise SystemExit(f'{city}: production `partial` and the study arm disagree on '
                         f'{len(bad)} panos, e.g. {bad[:3]}')
    arms, own = confirm_arms(panos, fuse_height, seed=args.seed)
    fused = _fuse_arms(arms)
    tag = {'city': city, 'height': label, 'exploratory': exploratory}
    pairs = [{**tag, **r} for r in score_reassoc(arms, fused)]
    gt_rows, gt_info = score_gt(arms, fused, *es.load_benchmark(city, bench_root))
    gt = [{**tag, **r} for r in gt_rows]
    members = unplaceable_members(arms, OPERATIONAL_CONFIDENCE)
    member_rows = [{**tag, 'arm': a, 'unplaceable': members[a],
                    'unplaceable_vs_off': members[a] - members[ARM_OFF]} for a in arms]
    print(f'  confirm {city} @ {label}: {len(panos)} panos, {time.time() - t0:.0f} s',
          file=sys.stderr)
    inventory = []
    referee = list(INVENTORY_CITIES)
    own_inventory = (REPO_ROOT / 'runs' / city / 'inventory_oracle' /
                     'inventory.geojson').exists()
    for icity in referee + ([city] if own_inventory and city not in referee else []):
        if icity == city:
            ipanos, iarms, ifused = panos, arms, fused
        else:
            ipanos, ih = load_city(icity, height, args.runs_root)
            iarms, _ = confirm_arms(ipanos, ih, seed=args.seed)
            ifused = _fuse_arms(iarms)
        by_id = {p.pano_id: p for p in ipanos}
        role = 'referee' if icity in referee else 'reported'
        inventory += [{'city': icity, 'height': label, 'exploratory': exploratory,
                       'role': role, **r}
                      for r in score_inventory(icity, iarms, ifused, by_id, args.runs_root)]
        print(f'  inventory {icity} ({role}) @ {label}: {time.time() - t0:.0f} s',
              file=sys.stderr)
    vint = [{**tag, **r} for r in vintage_coefficients(panos, fuse_height)]
    outcome, _state, lines = confirm_verdict(
        _str_rows(pairs), _str_rows(gt),
        _str_rows([r for r in inventory if r['role'] == 'referee']))
    write_csv(out / 'pairs.csv', pairs)
    write_csv(out / 'gt.csv', gt)
    write_csv(out / 'members.csv', member_rows)
    write_csv(out / 'inventory.csv', inventory)
    write_csv(out / 'vintage_k.csv', vint)
    k = geo.PARTIAL_POSE_K_GSV
    if exploratory:
        head = ("**EXPLORATORY: this city was in #116's train set, so this output proves the "
                'code path and is not a confirmation.**')
    else:
        head = 'This height DECIDES.' if deciding else 'Reported at this height; `auto` decides.'
    inv_rows = [r for r in _str_rows(inventory) if r['frame'] == FRAME_FROZEN
                and _same(r['radius_m'], INVENTORY_RADIUS_M)]
    text = [f'# {tag_x}#116 confirmatory run: {city} @ {label}', '', head, '',
            f'Frozen constants k_pitch {k[0]}, k_roll {k[1]} (geo.PARTIAL_POSE_K_GSV); whole '
            f'run, {len(panos)} panos, tier {OPERATIONAL_CONFIDENCE}, 25 m cap, seed '
            f'{args.seed} (the shuffle kept its own tilt on {own} panos). Production\'s pose '
            f"equals the study arm's on all {len(panos)} panos.", '',
            f"Store-pose gate: **{gate['status']}**"
            + (f" -- {gate['reason']}" if gate.get('reason') else '')
            + (f" (mapping {gate['mapping']}, agreement {gate['best_agreement']:.3f} on "
               f"{gate['compared']} panos; PS null-roll share {gate['ps_null_roll_share']:.3f})"
               if gate['status'] == 'pass' else '') + '; `store_pose_gate.json`.', '',
            f'## Outcome: {tag_x}{outcome}', ''] + [f'- {ln}' for ln in lines]
    text += ['', '## Pair distance, reassoc (m)', '',
             _md_table(_str_rows(pairs), ['arm', 'multi_view_sites', 'pairs_scored',
                                          'median_m', 'mean_m', 'p90_m']),
             '', '## RampNet GT survivorship (off pool)', '',
             _md_table(_str_rows(gt), ['arm', 'gt_marks', 'gt_marks_unplaceable',
                                       'off_pool_ramps', 'recall_off_pool_2p5m',
                                       'lost_vs_off_2p5m', 'gained_vs_off_2p5m']),
             '', f"GT panos judged: {gt_info['counts'].get('judged')}", '',
             '## Operational members not placed inside 25 m', '',
             _md_table(_str_rows(member_rows), ['arm', 'unplaceable', 'unplaceable_vs_off']),
             '', '## Inventories (frozen@off, pool on off, 5 m)', '',
             _md_table(inv_rows, ['city', 'role', 'arm', 'n_pool', 'median_m', 'p90_m']),
             '', '## k per capture vintage (descriptive; never scored)', '',
             _md_table(_str_rows(vint), ['vintage', 'n_rows', 'n_panos', 'k_pitch', 'se_pitch',
                                         'k_roll', 'se_roll', 'intercept'])]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verdict.md').write_text('\n'.join(text) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(text))


def cmd_loss_bar(args):
    """Print the confirmatory clause (ii)'s k*(n) table."""
    print('| n | k*(n) |\n|---|---|')
    for lo, hi, k in recall_loss_bar_table():
        print(f'| {lo}' + (f'-{hi}' if hi != lo else '') + f' | {k} |')


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


# ------------------------------------------------------------------------ figures

FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'gsv-partial-pose'
FIG_DATA = FIG_DIR / 'data'
FIG_SITE_CITY = 'gainesville'   # the deciding inventory city (64% 2026 rig, #79)
FIG_RECALL_CITY = 'bend'        # where clause (ii) failed
FIG_RADIUS_M = 2.5
# Categorical slots from the dataviz reference palette, in fixed order; `off` is ink grey.
FIG_COLORS = {ARM_OFF: '#52514e', ARM_PARTIAL: '#2a78d6', ARM_LOCO: '#4a3aa7',
              ARM_SHUFFLED: '#eb6834', ARM_LOCO_SHUFFLED: '#eda100', ARM_FULL: '#1baf7a',
              ARM_MIRROR: '#e87ba4'}
FIG_MARKERS = {ARM_OFF: 'o', ARM_PARTIAL: 'o', ARM_LOCO: 's', ARM_SHUFFLED: 'v',
               ARM_LOCO_SHUFFLED: '^', ARM_FULL: 'D', ARM_MIRROR: 'X'}


def pano_crop(bundle_dir, pano_id, x, y, half_w=0.045, half_h=0.09, out_px=360):
    """(crop array, mark row in the crop) for a bundle pano around normalized (x, y),
    wrapping across the seam; the mark sits on the crop's centre column. None when the
    pano is not in the local bundle. Decodes at quarter scale (JPEG draft)."""
    import numpy as np
    from PIL import Image
    path = Path(bundle_dir) / 'panos' / f'{pano_id}.jpg'
    if not path.exists():
        return None
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        im.draft('RGB', (im.width // 4, im.height // 4))
        arr = np.asarray(im.convert('RGB'))
    h, w = arr.shape[:2]
    cx, cy = int(x * w), int(y * h)
    dx, dy = int(half_w * w), int(half_h * h)
    cols = [(cx + i) % w for i in range(-dx, dx)]
    y0, y1 = max(0, cy - dy), min(h, cy + dy)
    img = Image.fromarray(arr[y0:y1][:, cols])
    img.thumbnail((out_px, out_px))
    return np.asarray(img), (cy - y0) / (y1 - y0) * img.height


def figure_recall_data(args):
    """Per off-pool ramp, per arm, for FIG_RECALL_CITY at `auto`: the TEST half (off,
    partial, partial-loco -- the decision) and, exploratory, the full run (off,
    partial-loco). Asserts the lost/gained counts reproduce the committed gt.csv."""
    city = FIG_RECALL_CITY
    table = read_csv(args.pooled_dir / 'coefficients.csv')
    bench = es.load_benchmark(city, Path(args.benchmark_root))
    panos, fuse_height = load_city(city, fs.HEIGHT_AUTO, args.runs_root)
    _train, test = split_halves(panos, args.seed)
    coefs = {ARM_PARTIAL: coef_pair(table, 'auto', 'city', city),
             ARM_LOCO: coef_pair(table, 'auto', 'loco', city)}
    rows = []
    for scope, ps, arm_set, gt_name in (
            ('test_half', test, (ARM_OFF, ARM_PARTIAL, ARM_LOCO), 'gt.csv'),
            ('full_run_exploratory', panos, (ARM_OFF, ARM_LOCO), 'explore_full/gt.csv')):
        arms, _own = build_arms(ps, coefs, fuse_height, arms=arm_set, seed=args.seed)
        fused = {arm: fs.fuse(p, params)[:2] for arm, (p, params) in arms.items()}
        details = []
        gt_rows, _info = score_gt(arms, fused, *bench, details=details)
        committed = {r['arm']: r for r in read_csv(args.pooled_dir / gt_name)
                     if r['city'] == city and r['height'] == 'auto'}
        for r in gt_rows:
            c = committed[r['arm']]
            assert (int(c['lost_vs_off_2p5m']), int(c['gained_vs_off_2p5m'])) == \
                (r['lost_vs_off_2p5m'], r['gained_vs_off_2p5m']), (scope, r['arm'])
        for d in details:
            pid, x, y, kind = d['marks'][0]
            for arm, a in d['arms'].items():
                rows.append({'scope': scope, 'ramp': d['ramp'], 'arm': arm,
                             'self_detected': d['self_detected'], 'n_marks': len(d['marks']),
                             'pano_id': pid, 'x': x, 'y': y, 'mark_kind': kind,
                             'nearest_site_m': a['nearest_site_m'],
                             'match_2p5m': a['match_2p5m'],
                             'recalled_2p5m': a['recalled_2p5m']})
        print(f'  figure data: {city} {scope}: {len(details)} off-pool ramps', file=sys.stderr)
    write_csv(FIG_DATA / 'bend_recall_ramps.csv', rows)


def figure_site_data(args):
    """One representative multi-view site in FIG_SITE_CITY (TEST half, `auto`), off vs the
    city's own partial fit, association frozen from off (as clause (iv) scores it).

    Selection rule (fixed before looking at any candidate): a site is eligible when it has
    >= 3 operational members on >= 3 panos and is matched one-to-one to an inventory point
    within 2.5 m under off. Its spread under an arm is the median pairwise distance between
    its members' placements; the site drawn is the eligible one whose partial/off spread
    ratio is closest to the median ratio over all eligible sites (ties: lowest site id)."""
    import json
    city = FIG_SITE_CITY
    table = read_csv(args.pooled_dir / 'coefficients.csv')
    panos, fuse_height = load_city(city, fs.HEIGHT_AUTO, args.runs_root)
    _train, test = split_halves(panos, args.seed)
    coefs = {ARM_PARTIAL: coef_pair(table, 'auto', 'city', city),
             ARM_LOCO: coef_pair(table, 'auto', 'loco', city)}
    names = (ARM_OFF, ARM_PARTIAL)
    arms, _own = build_arms(test, coefs, fuse_height, arms=names, seed=args.seed)
    sites, frame = fs.fuse(*arms[ARM_OFF])[:2]
    op_sites = [s for s in sites if s.n_operational > 0]
    kept, pos, placed = es.refit_frozen(op_sites, frame, oracle.placer(arms), names)
    inventory, _rec = oracle.load_inventory(city)
    points = [Pt(k, *frame.to_enu(lat, lng)) for k, lat, lng, _p in inventory]
    pool = match_one_to_one(points, pos[ARM_OFF], FIG_RADIUS_M)
    index = {s.id: k for k, s in enumerate(kept)}

    def spread(gs):
        en = [frame.to_enu(g.lat, g.lng) for g in gs]
        return es._pct([math.hypot(a[0] - b[0], a[1] - b[1])
                        for i, a in enumerate(en) for b in en[i + 1:]], 0.5)

    cands = []
    for pi, s in pool.items():
        k = index[s.id]
        members = [d for d, _ in kept[k].members if d.operational]
        if len(members) < 3 or len({d.pano_id for d in members}) < 3:
            continue
        off_s = spread(placed[ARM_OFF][k])
        if off_s and off_s > 0:
            cands.append((spread(placed[ARM_PARTIAL][k]) / off_s, s.id, k, pi))
    med = es._pct(sorted(c[0] for c in cands), 0.5)
    ratio, site_id, k, pi = min(cands, key=lambda c: (abs(c[0] - med), c[1]))
    by_id = {p.pano_id: p for p in test}
    members = [d for d, _ in kept[k].members if d.operational]
    out = {'city': city, 'height': 'auto', 'site_id': site_id, 'eligible_sites': len(cands),
           'median_ratio': med, 'site_ratio': ratio,
           'share_tighter': sum(1 for c in cands if c[0] < 1) / len(cands),
           'k_partial': coefs[ARM_PARTIAL],
           'inventory_en': [points[pi].e, points[pi].n],
           'site_en': {arm: [pos[arm][k].e, pos[arm][k].n] for arm in names},
           'members': []}
    for j, d in enumerate(members):
        p = by_id[d.pano_id]
        tilt = stored_tilt(p)
        out['members'].append({
            'pano_id': d.pano_id, 'x': d.x, 'y': d.y,
            'camera_en': list(frame.to_enu(p.lat, p.lng)),
            'pitch': None if tilt is None else tilt[0],
            'roll': None if tilt is None else tilt[1], 'capture_date': p.capture_date,
            'placed_en': {arm: list(frame.to_enu(placed[arm][k][j].lat, placed[arm][k][j].lng))
                          for arm in names}})
    (FIG_DATA / 'site_example.json').write_text(json.dumps(out, indent=1), encoding='utf-8',
                                                 newline='\n')
    print(f'  figure data: {city} site {site_id}, ratio {ratio:.3f} (median {med:.3f} over '
          f'{len(cands)} eligible)', file=sys.stderr)


def _style(plt):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb', 'grid.color': '#e4e3df'})


def fig_pair_distance(plt, pairs):
    """Change vs off in median and p90 re-associated pair distance, per city and arm."""
    arms = (ARM_PARTIAL, ARM_LOCO, ARM_SHUFFLED, ARM_LOCO_SHUFFLED, ARM_FULL, ARM_MIRROR)
    rows = {(r['city'], r['arm']): r for r in pairs
            if r['height'] == 'auto' and r['frame'] == FRAME_REASSOC}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    step = 0.12
    for ax, stat, name in ((axes[0], 'median_m', 'median'), (axes[1], 'p90_m', 'p90')):
        for ci, city in enumerate(CITIES):
            off = float(rows[city, ARM_OFF][stat])
            for ai, arm in enumerate(arms):
                yv = ci + (ai - (len(arms) - 1) / 2) * step
                dv = float(rows[city, arm][stat]) - off
                ax.plot([0, dv], [yv, yv], color=FIG_COLORS[arm], lw=1.2, alpha=.55)
                ax.plot(dv, yv, FIG_MARKERS[arm], color=FIG_COLORS[arm], ms=6,
                        mec='#fcfcfb', mew=.8, label=arm if ci == 0 else None)
            ax.annotate(f'off = {off:.2f} m', (0, ci - .42), fontsize=7, color='#52514e',
                        ha='left', xytext=(3, 0), textcoords='offset points')
        ax.axvline(0, color='#0b0b0b', lw=1)
        ax.grid(axis='x')
        ax.set_xlabel(f'{name} pair distance minus off (m)\n<- members agree better | worse ->')
        ax.set_title(f'{name} within-site pair distance', fontsize=10)
    labels = []
    for city in CITIES:
        n = int(rows[city, ARM_OFF]['common_pairs'])
        tag = '' if n >= RULE_MIN_COMMON_PAIRS else '\n(not gated)'
        labels.append(f'{city}\n{n:,} pairs{tag}')
    axes[0].set_yticks(range(len(CITIES)), labels)
    axes[0].set_ylim(len(CITIES) - .45, -.6)
    axes[1].legend(loc='lower right', fontsize=7.5, frameon=False, title='arm',
                   title_fontsize=7.5)
    fig.suptitle('Held-out GSV panos, auto height: the partial pose tightens multi-view sites;\n'
                 'its magnitude-matched shuffle and the mirror sign do not', fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_coefficients(plt, coefs):
    """k_pitch and k_roll, 95% CI, per scope and city, at both heights."""
    rows = [r for r in coefs if r['tier'] == '0.3000']
    order = [('city', c) for c in CITIES] + [('loco', c) for c in CITIES] + [('pooled', 'all')]
    labels = [f'{c} (own fit)' if s == 'city' else f'all but {c} (LOCO)' if s == 'loco'
              else 'pooled, all five' for s, c in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    colors = {'auto': '#2a78d6', '2.6': '#eb6834'}
    for ax, (k, se, name, ref) in zip(axes, (('k_pitch', 'se_pitch', 'pitch', 0.25),
                                              ('k_roll', 'se_roll', 'roll', 0.5))):
        for hi, h in enumerate(('auto', '2.6')):
            for yi, (scope, city) in enumerate(order):
                r = next(r for r in rows if r['height'] == h and r['scope'] == scope
                         and r['city'] == city)
                ax.errorbar(float(r[k]), yi + (hi - .5) * 0.3, xerr=1.96 * float(r[se]),
                            fmt='o', color=colors[h], ms=5, lw=1.4, capsize=2,
                            label=(f'camera height {h}' + (' m' if h == '2.6' else ''))
                            if yi == 0 else None)
        ax.axvline(ref, color='#52514e', lw=1, ls='--')
        ax.annotate(f'#113 fixed arm ({ref})', (ref, -0.9), fontsize=7, color='#52514e',
                    ha='center')
        ax.axvline(0, color='#0b0b0b', lw=.8)
        for yb in (4.5, 9.5):
            ax.axhline(yb, color='#e4e3df', lw=1)
        ax.set_xlabel(f'k_{name}: fraction of the stored {name}\ntreated as placement error')
        ax.grid(axis='x')
    axes[0].set_yticks(range(len(order)), labels)
    axes[0].set_ylim(len(order) - .4, -1.3)
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    fig.suptitle('Leak fractions fitted on the TRAIN halves (95% CI, pano-clustered SE):\n'
                 'about a fifth of the pitch and two fifths of the roll', fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_site_example(plt, site, bundle_dir):
    """Plan view of the selected site (every view, and a zoom on the inventory ramp), plus
    crops of its member views when they are in the local bundle."""
    members = site['members']
    crops = [pano_crop(bundle_dir, m['pano_id'], m['x'], m['y']) for m in members]
    n_have = sum(c is not None for c in crops)
    ncol = max(2, n_have)
    fig = plt.figure(figsize=(12, 7.6 if n_have else 5.8))
    gs = fig.add_gridspec(2 if n_have else 1, ncol, height_ratios=[3, 1.2] if n_have else [1])
    half = ncol // 2
    _site_panel(fig.add_subplot(gs[0, :half]), site, zoom=False)
    _site_panel(fig.add_subplot(gs[0, half:]), site, zoom=True)
    ie, inn = site['inventory_en']
    d = {a: math.hypot(site['site_en'][a][0] - ie, site['site_en'][a][1] - inn)
         for a in (ARM_OFF, ARM_PARTIAL)}
    fig.suptitle(f"{site['city']}, held-out site {site['site_id']}, chosen as the median-ratio "
                 f"site: member spread partial/off = {site['site_ratio']:.2f} (median over "
                 f"{site['eligible_sites']} eligible sites {site['median_ratio']:.2f})\n"
                 f"fused site to the city inventory ramp: {d[ARM_OFF]:.2f} m off, "
                 f"{d[ARM_PARTIAL]:.2f} m partial", fontsize=9.5)
    j = 0
    for i, c in enumerate(crops):
        if c is None:
            continue
        img, my = c
        cax = fig.add_subplot(gs[1, j])
        cax.imshow(img)
        cax.plot(img.shape[1] / 2, my, marker='o', ms=16, mfc='none', mec='#eda100', mew=2)
        cax.set_title(f'view {i + 1}', fontsize=8)
        cax.axis('off')
        j += 1
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, n_have, len(members)


def _site_panel(ax, site, zoom):
    """One plan-view panel of fig 3: every camera and ray, or +-2.2 m around the inventory
    ramp with an arrow from each member's off placement to its partial placement."""
    ie, inn = site['inventory_en']
    for i, m in enumerate(site['members']):
        ce, cn = m['camera_en']
        pe = {a: m['placed_en'][a] for a in (ARM_OFF, ARM_PARTIAL)}
        far = max(pe.values(), key=lambda p: math.hypot(p[0] - ce, p[1] - cn))
        ax.plot([ce, far[0]], [cn, far[1]], color='#b5b4ae', lw=.9, zorder=1)
        ax.plot(ce, cn, marker='^', color='#0b0b0b', ms=7, ls='none', zorder=3,
                label='camera' if i == 0 else None)
        if zoom:
            ax.annotate('', xy=pe[ARM_PARTIAL], xytext=pe[ARM_OFF],
                        arrowprops={'arrowstyle': '->', 'color': '#52514e', 'lw': .8})
        else:
            tilt = ('' if m['pitch'] is None else
                    f"\npitch {m['pitch']:+.1f}°, roll {m['roll']:+.1f}°")
            ax.annotate(f'view {i + 1}{tilt}', (ce, cn), textcoords='offset points',
                        xytext=(6, -14), fontsize=7, color='#52514e')
        for arm, mk in ((ARM_OFF, 'o'), (ARM_PARTIAL, 'D')):
            ax.plot(*pe[arm], mk, color=FIG_COLORS[arm], ms=7 if zoom else 4, mec='#fcfcfb',
                    mew=.8, ls='none', zorder=4,
                    label=f'member placement, {arm}' if i == 0 else None)
            if zoom:
                ax.annotate(str(i + 1), pe[arm], textcoords='offset points', xytext=(5, 3),
                            fontsize=8, color=FIG_COLORS[arm])
    for arm, mk in ((ARM_OFF, 'o'), (ARM_PARTIAL, 'D')):
        ax.plot(*site['site_en'][arm], mk, color=FIG_COLORS[arm], ms=14, mfc='none', mew=2,
                ls='none', zorder=5, label=f'fused site, {arm}')
    ax.plot(ie, inn, marker='*', color='#0b0b0b', ms=14, ls='none', zorder=6,
            label='city inventory ramp')
    ax.set_aspect('equal', adjustable='datalim')
    if zoom:
        ax.set_xlim(ie - 2.2, ie + 2.2)
        ax.set_ylim(inn - 2.2, inn + 2.2)
        ax.set_title('zoom, +-2 m around the inventory ramp (arrows: off -> partial)',
                     fontsize=8.5)
        ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    else:
        ax.set_title('every member view: cameras and rays; numbers = view', fontsize=8.5)
    ax.set_xlabel('east (m)')
    ax.set_ylabel('north (m)')
    ax.grid(True)


def fig_recall(plt, rows, bundle_dir):
    """Every off-pool Bend ramp's nearest-site distance, off vs candidate, with the ramps a
    candidate lost or gained against off at 2.5 m highlighted and cropped."""
    by = {}
    for r in rows:
        by.setdefault((r['scope'], int(r['ramp'])), {})[r['arm']] = r
    panels = (('test_half', ARM_PARTIAL, 'decision: held-out half, own fit'),
              ('test_half', ARM_LOCO, 'decision: held-out half, LOCO fit'),
              ('full_run_exploratory', ARM_LOCO, 'EXPLORATORY: full run, LOCO fit'))
    num = lambda v: None if v in ('', None) else float(v)  # noqa: E731
    changed = []
    fig = plt.figure(figsize=(13, 8.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[2.4, 1.2])
    cap = 8.0
    for pi, (scope, arm, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * pi:2 * pi + 2])
        same, lost, gained, lost_unplaced = [], [], [], []
        for (sc, k), a in sorted(by.items()):
            if sc != scope:
                continue
            o, c = a[ARM_OFF], a[arm]
            ro, rc = o['recalled_2p5m'] == 'True', c['recalled_2p5m'] == 'True'
            if ro != rc:
                changed.append((scope, arm, k, 'lost' if ro else 'gained', o, c))
            xo, yc = num(o['nearest_site_m']), num(c['nearest_site_m'])
            if xo is None:
                continue
            if yc is None:   # the arm cannot place the GT mark (past the 25 m raycast cap)
                (lost_unplaced if ro and not rc else same).append((min(xo, cap), cap))
                continue
            pt = (min(xo, cap), min(yc, cap))
            (lost if ro and not rc else gained if rc and not ro else same).append(pt)
        for pts, mk, col, lab, ms in (
                (same, 'o', '#b5b4ae', 'recall unchanged', 3.5),
                (lost, 'X', '#e87ba4', 'lost vs off', 10),
                (lost_unplaced, 'v', '#e87ba4', 'lost: mark now past 25 m (drawn at 8)', 10),
                (gained, 'P', '#1baf7a', 'gained vs off', 10)):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, ms=ms, color=col,
                    mec='#0b0b0b' if ms > 5 else col, mew=.6, ls='none',
                    label=f'{lab} ({len(pts)})')
        ax.plot([0, cap], [0, cap], color='#52514e', lw=.7, ls=':')
        ax.axvline(FIG_RADIUS_M, color='#0b0b0b', lw=.9, ls='--')
        ax.axhline(FIG_RADIUS_M, color='#0b0b0b', lw=.9, ls='--')
        ax.set_xlim(0, cap + .3)
        ax.set_ylim(0, cap + .3)
        ax.set_aspect('equal')
        ax.set_xlabel('off: GT ramp to nearest site (m)')
        ax.set_ylabel(f'{arm}: GT ramp to nearest site (m)')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.8, frameon=False, loc='lower right')
    fig.text(0.5, 0.012, 'Distances beyond 8 m are drawn at 8 m. Dashed: the 2.5 m match radius. '
             'A ramp counts as recalled if it is self-detected or matched one-to-one within '
             '2.5 m, so a ramp inside the radius can still be lost\nto a neighbouring ramp that '
             'claims the same site. Crops: the GT mark (circle) in its judged pano, from the '
             'local RampNet bundle.', ha='center', fontsize=7.5, color='#52514e')
    seen, j = set(), 0
    for scope, arm, k, what, o, c in changed:
        if j >= 6 or (scope, k) in seen:
            continue
        seen.add((scope, k))
        cax = fig.add_subplot(gs[1, j])
        crop = pano_crop(bundle_dir, o['pano_id'], float(o['x']), float(o['y']),
                         half_w=0.03, half_h=0.06)
        if crop is not None:
            img, my = crop
            cax.imshow(img)
            cax.plot(img.shape[1] / 2, my, marker='o', ms=16, mfc='none', mec='#eda100', mew=2)
        else:
            cax.text(.5, .5, 'pano not in\nlocal bundle', ha='center', va='center', fontsize=8)
        fm = lambda v: 'unplaced' if num(v) is None else f'{num(v):.2f} m'  # noqa: E731
        tag = 'held-out' if scope == 'test_half' else 'full run'
        cax.set_title(f'{what}: {tag}, {arm}\nnearest site {fm(o["nearest_site_m"])} -> '
                      f'{fm(c["nearest_site_m"])}', fontsize=7)
        cax.axis('off')
        j += 1
    fig.suptitle(f'{FIG_RECALL_CITY}, auto height: the ramps behind the failed recall clause',
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    return fig, changed


def cmd_figures(args):
    """Draw docs/figures/gsv-partial-pose/*.png. Figures 1-2 read the committed pooled CSVs
    (copied into data/). Figures 3-4 need a re-fuse (runs/<city>/ + the RampNet bundle):
    their data is written once to data/ and re-read after that (--refresh redoes it).
    Crops come from the local bundle panos only (no network)."""
    import json
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    FIG_DATA.mkdir(parents=True, exist_ok=True)
    for name in ('pairs.csv', 'coefficients.csv', 'gt.csv'):
        shutil.copyfile(args.pooled_dir / name, FIG_DATA / name)
    shutil.copyfile(args.pooled_dir / 'explore_full' / 'gt.csv',
                    FIG_DATA / 'explore_full_gt.csv')
    recall_csv, site_json = FIG_DATA / 'bend_recall_ramps.csv', FIG_DATA / 'site_example.json'
    if args.refresh or not recall_csv.exists():
        figure_recall_data(args)
    if args.refresh or not site_json.exists():
        figure_site_data(args)
    bench = Path(args.benchmark_root)
    figs = {'fig1_pair_distance.png': fig_pair_distance(plt, read_csv(FIG_DATA / 'pairs.csv')),
            'fig2_coefficients.png': fig_coefficients(plt,
                                                      read_csv(FIG_DATA / 'coefficients.csv'))}
    site = json.loads(site_json.read_text(encoding='utf-8'))
    figs['fig3_site_example.png'], n_crops, n_members = fig_site_example(
        plt, site, bench / site['city'])
    figs['fig4_bend_recall.png'], changed = fig_recall(plt, read_csv(recall_csv),
                                                       bench / FIG_RECALL_CITY)
    for name, fig in figs.items():
        fig.savefig(FIG_DIR / name, dpi=80 if name.startswith('fig4') else 110)  # < 300 KB
        plt.close(fig)
        print(f'  wrote {FIG_DIR / name} ({(FIG_DIR / name).stat().st_size // 1024} KB)')
    print(f'  site example: {n_crops} of {n_members} member views cropped from the bundle')
    for scope, arm, k, what, o, c in changed:
        print(f'  {scope} {arm} ramp {k}: {what}; off match={o["match_2p5m"] or "-"} '
              f'nearest={o["nearest_site_m"] or "-"}; {arm} match={c["match_2p5m"] or "-"} '
              f'nearest={c["nearest_site_m"] or "-"}; self_detected={o["self_detected"]}; '
              f'marks={o["n_marks"]} ({o["mark_kind"]}); pano {o["pano_id"]} '
              f'x={float(o["x"]):.3f} y={float(o["y"]):.3f}')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=('fit', 'score', 'tieback', 'verdict', 'explore', 'figures',
                                        'consistency', 'confirm', 'loss-bar', 'all'))
    ap.add_argument('cities', nargs='*', default=list(CITIES))
    ap.add_argument('--heights', nargs='+', default=[height_label(h) for h in HEIGHTS])
    ap.add_argument('--runs-root', default=str(REPO_ROOT / 'runs'))
    ap.add_argument('--benchmark-root', default=str(DEFAULT_BENCHMARK_ROOT))
    ap.add_argument('--out', default=None, help='pooled output dir (default runs/_pooled/partial_pose)')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--refresh', action='store_true',
                    help='figures: recompute the re-fused figure data')
    ap.add_argument('--camera-height-m', default='auto', choices=('auto', '2.6'),
                    help='confirm: the height to score at (auto decides; 2.6 is reported)')
    ap.add_argument('--exploratory', action='store_true',
                    help="confirm: allow a city from #116's train set; every output is "
                         'labelled EXPLORATORY')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.heights = [parse_height(h) for h in args.heights]
    args.pooled_dir = Path(args.out) if args.out else POOLED_DIR
    cmds = {'fit': [cmd_fit], 'score': [cmd_score], 'tieback': [cmd_tieback],
            'verdict': [cmd_verdict], 'explore': [cmd_explore],
            'figures': [cmd_figures], 'consistency': [cmd_consistency],
            'confirm': [cmd_confirm], 'loss-bar': [cmd_loss_bar],
            'all': [cmd_fit, cmd_score, cmd_tieback, cmd_verdict]}
    for cmd in cmds[args.command]:
        cmd(args)


if __name__ == '__main__':
    main()
