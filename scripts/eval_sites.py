"""Stage-3 world-space evaluation of multi-view fusion (issue #27).

Joins a city's RampNet ground truth (benchmark/<city>/{verdicts.json,
records.jsonl} in the RampNet repo — read as data, no code import) against the
full run's results.jsonl, re-fuses sites in memory (fuse_sites.fuse, so every
ablation shares one code path), and scores in world space:

- world GT: verdict-true operational detections and non-unsure missed marks are
  raycast with the same projector and envelope as fusion; within one pano GT
  points are distinct ramps by construction (never merged); across panos they
  greedily merge within --gt-merge-m. GT panos are used only if fully judged
  (mirrors rampnet.validation.collect: any None verdict excludes the pano), and
  contribute to the recall pool only if their missed-ramp check is confirmed
  (no_missed set or a missed mark present).
- world recall: fraction of placeable recall-pool GT ramps that are
  self-detected or matched one-to-one by an operational site within
  --match-radius-m, with the union decomposition #27 asks for:
  self_detected / recovered_other_view / subthreshold_only / unmatched.
- world precision, GT-incompleteness-safe: over operational sites with at least
  one operational member from a judged GT pano — TP if any such member verdict
  is true/duplicate (a duplicate proves the physical ramp), FP if all decided
  verdicts are false, excluded if unsure-only. P(site is real | a GT pano
  observed it), unbiased because the GT sample is spatially random.

The eval never *requires* the run's manifest and never touches the network.

Usage:
    python scripts/eval_sites.py paterson
    python scripts/eval_sites.py sao_paulo --benchmark-root ../RampNet/benchmark
"""
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (same form RampNet uses)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class GTPoint:
    pano_id: str
    kind: str          # 'det' (verdict-true operational detection) or 'missed'
    e: float
    n: float
    in_pool: bool      # pano's missed-ramp check confirmed -> counts for recall


class GTRamp:
    def __init__(self, pt):
        self.points = [pt]
        self.e, self.n = pt.e, pt.n

    def add(self, pt):
        self.points.append(pt)
        self.e = sum(p.e for p in self.points) / len(self.points)
        self.n = sum(p.n for p in self.points) / len(self.points)

    @property
    def pano_ids(self):
        return {p.pano_id for p in self.points}

    @property
    def self_detected(self):
        return any(p.kind == 'det' for p in self.points)

    @property
    def in_pool(self):
        return any(p.in_pool for p in self.points)


def judged_gt_panos(verdict_panos, bundle_ops, run_panos_by_id, counts, warnings):
    """Yield (pano_id, entry, run_pano, ops, in_pool) for every GT pano whose
    verdicts can be trusted against the run, updating counts/warnings in place.

    This is the guard, split out so consumers that need the reviewer's PIXEL marks
    rather than world points (scripts/mapillary_tilt.py's horizon test) apply the
    same skip rules: a pano is dropped when it is missing from the run, when its
    operational detections drifted from the frozen bundle, or when the verdict list
    does not line up with them — the three ways a verdict can end up attached to a
    detection the reviewer never saw.
    """
    for pid in sorted(verdict_panos):
        entry = verdict_panos[pid]
        run_pano = run_panos_by_id.get(pid)
        if run_pano is None:
            warnings.append(f'{pid}: GT pano missing from the run')
            counts['skipped'] += 1
            continue
        ops = [(i, x, y, c) for i, x, y, c in run_pano.detections
               if c >= BENCHMARK_CONFIDENCE]
        expected = bundle_ops.get(pid)
        if expected is None or [(x, y, c) for _, x, y, c in ops] != expected:
            warnings.append(f'{pid}: run detections drifted from the bundle')
            counts['skipped'] += 1
            continue
        if len(entry['dets']) != len(ops):
            warnings.append(f"{pid}: verdicts don't match operational detections")
            counts['skipped'] += 1
            continue
        if any(d is None for d in entry['dets']):
            counts['partial'] += 1
            continue
        counts['judged'] += 1
        in_pool = (entry['no_missed'] or entry['missed']) \
            if 'no_missed' in entry else True
        if not in_pool:
            counts['no_pool'] += 1
        yield pid, entry, run_pano, ops, in_pool


def gt_counts():
    return {'gt_panos': 0, 'judged': 0, 'partial': 0, 'skipped': 0, 'no_pool': 0,
            'unplaceable': 0, 'placeable': 0, 'unsure_missed': 0}


def build_gt(verdict_panos, bundle_ops, run_panos_by_id, params, frame):
    """World GT points + the (pano_id, stored_index) -> verdict map.

    Returns (points, op_verdicts, counts, warnings). Panos are skipped (with a
    warning, mirroring rampnet.validation's skip-and-warn) when they are missing
    from the run or their operational detections drifted from the frozen bundle.
    """
    points, op_verdicts, warnings = [], {}, []
    counts = gt_counts()
    counts['gt_panos'] = len(verdict_panos)
    for pid, entry, run_pano, ops, in_pool in judged_gt_panos(
            verdict_panos, bundle_ops, run_panos_by_id, counts, warnings):
        pose = fs.pano_pose(run_pano, params.apply_pose)
        errors = geo.error_model_for(run_pano.source)

        def place(x, y, kind):
            g = geo.detection_ground_point(
                pose, x, y, camera_height=params.camera_height_m,
                max_range_m=params.max_range_m, errors=errors,
                apply_pose=params.rotates)
            if g is None:
                counts['unplaceable'] += 1
                return
            counts['placeable'] += 1
            e, n = frame.to_enu(g.lat, g.lng)
            points.append(GTPoint(pid, kind, e, n, in_pool))

        for verdict, (stored_i, x, y, _c) in zip(entry['dets'], ops):
            op_verdicts[(pid, stored_i)] = verdict
            if verdict is True:
                place(x, y, 'det')
        for mark in entry.get('missed', ()):
            if mark.get('unsure'):
                counts['unsure_missed'] += 1
                continue
            place(mark['x'], mark['y'], 'missed')
    return points, op_verdicts, counts, warnings


def merge_gt_points(points, merge_m):
    """Greedy cross-pano merge of GT points into physical ramps. Within one pano
    points never merge (the reviewer marked them as distinct ramps)."""
    ramps = []
    for pt in points:  # build_gt emits in sorted pano order -> deterministic
        best = None
        for ramp in ramps:
            if pt.pano_id in ramp.pano_ids:
                continue
            d = math.hypot(ramp.e - pt.e, ramp.n - pt.n)
            if d <= merge_m and (best is None or d < best[0]):
                best = (d, ramp)
        if best:
            best[1].add(pt)
        else:
            ramps.append(GTRamp(pt))
    return ramps


def match_one_to_one(ramps, sites, radius_m):
    """Greedy ascending-distance one-to-one matching. Returns {ramp_index: site}."""
    pairs = []
    for gi, ramp in enumerate(ramps):
        for site in sites:
            d = math.hypot(ramp.e - site.e, ramp.n - site.n)
            if d <= radius_m:
                pairs.append((d, gi, site.id, site))
    pairs.sort(key=lambda p: p[:3])
    matched, used_sites = {}, set()
    for d, gi, sid, site in pairs:
        if gi in matched or sid in used_sites:
            continue
        matched[gi] = site
        used_sites.add(sid)
    return matched


CAL_FLOORS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
VINTAGE_BUCKETS = ((0, '0'), (18, '1-18'), (36, '19-36'), (10 ** 9, '>36'))


def _vintage_bucket(delta_months):
    if delta_months is None:
        return 'unknown'
    for bound, name in VINTAGE_BUCKETS:
        if delta_months <= bound:
            return name


def evaluate_city(verdict_panos, bundle_ops, run_panos, params,
                  match_radius_m=5.0, gt_merge_m=2.5, prefused=None):
    """The full stage-3 join for one city; returns a result dict (no I/O).

    prefused=(sites, frame, fuse_stats) skips re-association — used by the
    match-radius sweep, where fusion is identical across radii."""
    sites, frame, fuse_stats = prefused or fs.fuse(run_panos, params)
    op_sites = [s for s in sites if s.n_operational > 0]
    sub_sites = [s for s in sites if s.n_operational == 0]
    run_by_id = {p.pano_id: p for p in run_panos}

    points, op_verdicts, counts, warnings = build_gt(
        verdict_panos, bundle_ops, run_by_id, params, frame)
    ramps = merge_gt_points(points, gt_merge_m)
    counts['gt_ramps'] = len(ramps)
    counts['cross_pano_merges'] = len(points) - len(ramps)
    pool = [r for r in ramps if r.in_pool]

    matched_op = match_one_to_one(pool, op_sites, match_radius_m)
    unmatched_idx = [i for i in range(len(pool)) if i not in matched_op]
    matched_sub = match_one_to_one([pool[i] for i in unmatched_idx],
                                   sub_sites, match_radius_m)
    sub_site_of = {unmatched_idx[j]: site for j, site in matched_sub.items()}
    sub_hits = set(sub_site_of)

    buckets = {'self_detected': 0, 'recovered_other_view': 0,
               'subthreshold_only': 0, 'unmatched': 0}
    self_detected_without_site = 0
    for i, ramp in enumerate(pool):
        if ramp.self_detected:
            buckets['self_detected'] += 1
            if i not in matched_op:
                self_detected_without_site += 1  # diagnostic: should be rare
        elif i in matched_op:
            buckets['recovered_other_view'] += 1
        elif i in sub_hits:
            buckets['subthreshold_only'] += 1
        else:
            buckets['unmatched'] += 1

    # GT-incompleteness-safe world precision over operational sites seen by GT panos
    tp = fp = unsure_only = 0
    tp_site_ids, fp_site_ids = set(), set()
    for site in op_sites:
        verdicts = [op_verdicts[(d.pano_id, d.det_index)]
                    for d, _ in site.members
                    if d.operational and (d.pano_id, d.det_index) in op_verdicts]
        if not verdicts:
            continue
        if any(v is True or v == 'duplicate' for v in verdicts):
            tp += 1
            tp_site_ids.add(site.id)
        elif any(v is False for v in verdicts):
            fp += 1
            fp_site_ids.add(site.id)
        else:
            unsure_only += 1

    n_pool = len(pool)
    world_recalled = buckets['self_detected'] + buckets['recovered_other_view']

    # (d) stage-4 promotion calibration: support profiles k(f) for GT ramps not
    # recovered at the operational threshold, and the promotion curve.
    has_subfloor = any(c < BENCHMARK_CONFIDENCE
                       for p in run_panos for _, _, _, c in p.detections)
    calibration = None
    if has_subfloor:
        profiles = []
        for i, ramp in enumerate(pool):
            if ramp.self_detected or i in matched_op:
                continue
            site = sub_site_of.get(i)
            ks = {f: (0 if site is None else
                      len({d.pano_id for d, _ in site.members if d.conf >= f}))
                  for f in CAL_FLOORS}
            profiles.append({'ramp_index': i,
                             'panos': sorted(ramp.pano_ids),
                             'site_id': None if site is None else site.id,
                             'k': ks})
        promotion = []
        for f in CAL_FLOORS:
            for k in (1, 2, 3):
                promoted = sum(1 for pr in profiles if pr['k'][f] >= k)
                promotion.append({
                    'floor': f, 'k': k, 'promoted_ramps': promoted,
                    'recall_if_promoted':
                        (world_recalled + promoted) / n_pool if n_pool else None})
        # ghost check: other-pano support of judged operational detections —
        # does consensus separate real ramps from view-consistent false positives?
        site_of_member = {(d.pano_id, d.det_index): s
                          for s in sites for d, _ in s.members}
        ghost_raw = {'true': [], 'false': []}
        for (pid, si), v in sorted(op_verdicts.items()):
            key = 'true' if v is True else 'false' if v is False else None
            site = site_of_member.get((pid, si))
            if key is None or site is None:
                continue
            ghost_raw[key].append(
                {f: len({d.pano_id for d, _ in site.members
                         if d.pano_id != pid and d.conf >= f})
                 for f in CAL_FLOORS})
        ghost = []
        for f in CAL_FLOORS:
            row = {'floor': f,
                   'n_true': len(ghost_raw['true']),
                   'n_false': len(ghost_raw['false'])}
            for k in (1, 2):
                for key in ('true', 'false'):
                    n = len(ghost_raw[key])
                    row[f'{key}_ge{k}'] = (
                        sum(1 for ks in ghost_raw[key] if ks[f] >= k) / n
                        if n else None)
            ghost.append(row)
        calibration = {'profiles': profiles, 'promotion': promotion,
                       'ghost': ghost}

    # (e) member-pair capture-date deltas, overall and within judged TP/FP sites
    vintage = {'all': {}, 'tp_sites': {}, 'fp_sites': {}}
    for site in sites:
        if len(site.members) < 2:
            continue
        keys = ['all']
        if site.id in tp_site_ids:
            keys.append('tp_sites')
        elif site.id in fp_site_ids:
            keys.append('fp_sites')
        ms = [d.months for d, _ in site.members]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                delta = None if ms[i] is None or ms[j] is None \
                    else abs(ms[i] - ms[j])
                b = _vintage_bucket(delta)
                for key in keys:
                    vintage[key][b] = vintage[key].get(b, 0) + 1

    # (f) dual-ramp separation: same-pano GT points < 5 m apart -> how often do
    # both end up with their own operational site? (one-to-one matching means
    # "both matched" == "kept separate")
    ramp_of_point = {}
    for ri, ramp in enumerate(ramps):
        for ptx in ramp.points:
            ramp_of_point[id(ptx)] = ri
    pool_index = {id(ramp): i for i, ramp in enumerate(pool)}
    dual = {'pairs': 0, 'both_matched': 0, 'one_matched': 0, 'neither': 0}
    by_pano = {}
    for ptx in points:
        by_pano.setdefault(ptx.pano_id, []).append(ptx)
    for pid in sorted(by_pano):
        pts = by_pano[pid]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if math.hypot(pts[i].e - pts[j].e, pts[i].n - pts[j].n) >= 5.0:
                    continue
                dual['pairs'] += 1
                hits = 0
                for ptx in (pts[i], pts[j]):
                    ramp = ramps[ramp_of_point[id(ptx)]]
                    pi = pool_index.get(id(ramp))
                    if pi is not None and pi in matched_op:
                        hits += 1
                dual['both_matched' if hits == 2
                     else 'one_matched' if hits == 1 else 'neither'] += 1

    return {
        'params': {'match_radius_m': match_radius_m, 'gt_merge_m': gt_merge_m,
                   'min_confidence': params.min_confidence,
                   'max_range_m': params.max_range_m,
                   'camera_height_m': params.camera_height_m,
                   'apply_pose': params.apply_pose},
        'counts': counts, 'warnings': warnings,
        'fuse': {k: fuse_stats[k] for k in
                 ('n_panos', 'n_projected', 'n_sites', 'n_operational_sites',
                  'n_multi_pano_sites')},
        'n_pool_ramps': n_pool,
        'world_recall': world_recalled / n_pool if n_pool else None,
        'world_recall_ci': wilson(world_recalled, n_pool),
        'own_view_recall': buckets['self_detected'] / n_pool if n_pool else None,
        'own_view_recall_ci': wilson(buckets['self_detected'], n_pool),
        'buckets': buckets,
        'self_detected_without_site': self_detected_without_site,
        'precision': {'tp': tp, 'fp': fp, 'unsure_only': unsure_only,
                      'value': tp / (tp + fp) if tp + fp else None,
                      'ci': wilson(tp, tp + fp)},
        'calibration': calibration,
        'vintage': vintage,
        'dual_ramp': dual,
    }


# --- The #42 pose precondition: p90 placement under the production range cap ----------

ARM_SHUFFLED_WITHIN = 'road-shuffled-within'
ARM_SHUFFLED_ACROSS = 'road-shuffled-across'
PRECONDITION_ARMS = (fs.POSE_OFF, fs.POSE_GRAVITY, fs.POSE_ROAD,
                     ARM_SHUFFLED_WITHIN, ARM_SHUFFLED_ACROSS)
# #51: road mode with a grade from somewhere other than production's SfM two-point
# grade. `road-dem` takes the USGS 3DEP grade (scripts/dem_grade.py), which never saw the
# SfM; its control is the SAME within-sequence shuffle applied to the DEM grades, so it
# keeps each sequence's DEM grade distribution and destroys only the per-frame alignment.
# `road-sfm-smoothed` (a +-20 m fit on the SfM altitude) is descriptive only.
ARM_ROAD_DEM = 'road-dem'
ARM_ROAD_DEM_SHUFFLED_WITHIN = 'road-dem-shuffled-within'
ARM_ROAD_SFM_SMOOTHED = 'road-sfm-smoothed'
PRECONDITION_ARMS_DEM = PRECONDITION_ARMS + (ARM_ROAD_SFM_SMOOTHED, ARM_ROAD_DEM,
                                             ARM_ROAD_DEM_SHUFFLED_WITHIN)
PRECONDITION_RADII = (2.5, 5.0)
# Fixed seed for the shuffled-grade controls, so the table is reproducible.
SHUFFLE_SEED = 42
# The FIRST decision rule, pre-registered on issue #42 before the first run: `road` becomes
# fusion's Mapillary default only if, against `off`, its p90 GT-to-site distance is no
# worse in ANY city (within this tolerance) and its median improves (strictly) in at least
# PRECONDITION_MIN_MEDIAN_WINS of the cities. Road passed it (study section 10.3).
PRECONDITION_P90_TOLERANCE_M = 0.1
PRECONDITION_MIN_MEDIAN_WINS = 3
# The SECOND rule, the shuffled-grade control, fixed before any shuffled arm was run
# (after #52 found, on GSV, that a shuffled ground normal "improves" p90 too and that p90
# can improve by survivorship). Road keeps the `auto` default only if ALL of:
#  (i)   road beats road-shuffled-within on BOTH p90 and median GT-to-site distance by more
#        than CONTROL_MARGIN_M in at least CONTROL_MIN_CITIES cities -- i.e. the frame's
#        OWN grade matters, not the sequence's grade distribution or its SfM error;
#  (ii)  recall at 2.5 m on the OFF pool does not drop by more than CONTROL_MAX_RECALL_DROP
#        (1.0 point) against off in any city;
#  (iii) road's count of unplaceable GT marks exceeds off's by no more than
#        CONTROL_MAX_UNPLACEABLE_FRAC of the off pool in any city.
CONTROL_MARGIN_M = 0.1
CONTROL_MIN_CITIES = 4
CONTROL_MAX_RECALL_DROP = 0.010
CONTROL_MAX_UNPLACEABLE_FRAC = 0.05
# The #51 rule, pre-registered on the issue before any DEM arm was scored: `road` with
# the DEM grade earns AUTO_ROAD_SOURCES = ('mapillary',) only if ALL FOUR hold on the
# eight-arm intersected set, with the constants above unchanged --
#  (i)   road-dem beats road-dem-shuffled-within on BOTH p90 and median by more than
#        CONTROL_MARGIN_M in at least CONTROL_MIN_CITIES cities;
#  (ii)  its off-pool recall at 2.5 m drops by no more than CONTROL_MAX_RECALL_DROP
#        against off in every city;
#  (iii) its unplaceable marks exceed off's by no more than CONTROL_MAX_UNPLACEABLE_FRAC
#        of the off pool in every city;
#  (iv)  the FIRST rule against off: p90 no worse by more than PRECONDITION_P90_TOLERANCE_M
#        in any city, median better in at least PRECONDITION_MIN_MEDIAN_WINS.
# (i)-(iii) are control_verdict(rows, ARM_ROAD_DEM, ARM_ROAD_DEM_SHUFFLED_WITHIN) and (iv)
# is precondition_verdict(rows, ARM_ROAD_DEM); dem_verdict combines them. If any clause
# fails, #51 closes negative and the road frame stays opt-in.


def _pct(values, q):
    """Nearest-rank percentile (the convention of scripts/mapillary_tilt.py's tables)."""
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round((len(v) - 1) * q)))]


@dataclass
class _Placed:
    """A GT ramp or a site at one arm's position (what match_one_to_one reads)."""
    id: int
    e: float
    n: float


def shuffled_grades(run_panos, within_sequence, seed=SHUFFLE_SEED):
    """Copies of the panos with each frame's road grade swapped for another frame's (#42
    control). The frame keeps its OWN travel bearing -- only the grade value moves -- so
    the control destroys exactly one thing: which grade belongs to which frame.

    within_sequence=True: each graded frame takes the grade of another randomly chosen
    graded frame of the SAME sequence (with replacement). That keeps the sequence's grade
    distribution and anything sequence-level SfM error puts into it. A sequence with a
    single graded frame has nothing to swap with and keeps its own (counted).
    within_sequence=False: the grades are permuted across every graded frame of the run.

    Returns (panos, n_unshuffled).
    """
    import random
    rng = random.Random(seed)
    graded = [i for i, p in enumerate(run_panos) if p.grade_deg is not None]
    new_grade, unshuffled = {}, 0
    if within_sequence:
        by_seq = {}
        for i in graded:
            by_seq.setdefault(run_panos[i].sequence_id, []).append(i)
        for members in by_seq.values():
            for i in members:
                others = [j for j in members if j != i]
                if not others:
                    new_grade[i] = run_panos[i].grade_deg
                    unshuffled += 1
                else:
                    new_grade[i] = run_panos[rng.choice(others)].grade_deg
    else:
        values = [run_panos[i].grade_deg for i in graded]
        rng.shuffle(values)
        new_grade = dict(zip(graded, values))
    return ([replace(p, grade_deg=new_grade[i]) if i in new_grade else p
             for i, p in enumerate(run_panos)], unshuffled)


def refit_frozen(sites, frame, place, arms, sigma_scale=1.0, counts=None):
    """Re-place a frozen site set under several arms (the #42 precondition design, also
    scripts/inventory_oracle.py's for #79).

    Membership is fixed -- whatever fuse built `sites` -- and each arm re-solves every site
    from its own raycast of the site's OPERATIONAL members, with the same inverse-covariance
    refit fuse_sites.Site uses. A site is kept only if EVERY arm places EVERY one of those
    members (place() returning None anywhere drops it), so all arms describe one site set.
    A site with NO operational member has nothing to refit from and is dropped too (its
    information matrix would be zero); when `counts` (a dict) is given, those drops are
    tallied in counts['no_operational_members']. Today's callers pass only sites with
    n_operational > 0, so this is a guard, not a behaviour change.

    place(arm, pano_id, x, y) -> geo.GroundEstimate or None; `frame` is the fuse's
    LocalFrame. Returns (kept, site_pos, placed): the kept sites in input order,
    {arm: [_Placed(site.id, e, n)]} aligned with kept, and {arm: [[GroundEstimate per
    operational member]]} aligned with kept (callers that need each member's ray, e.g.
    its bearing and range, read it there).

    Example (two arms that agree reproduce each other exactly):
        kept, pos, _ = refit_frozen(op_sites, frame, place, ('off', 'road'))
        assert [p.id for p in pos['off']] == [s.id for s in kept]
    """
    s2 = sigma_scale ** 2
    kept, site_pos, member_placed = [], {arm: [] for arm in arms}, {arm: [] for arm in arms}
    for site in sites:
        members = [d for d, _ in site.members if d.operational]
        if not members:
            if counts is not None:
                counts['no_operational_members'] = counts.get('no_operational_members', 0) + 1
            continue
        placed = {}
        for arm in arms:
            gs = [place(arm, d.pano_id, d.x, d.y) for d in members]
            if any(g is None for g in gs):
                break
            placed[arm] = gs
        else:
            kept.append(site)
            for arm in arms:
                lam, eta_e, eta_n = (0.0, 0.0, 0.0), 0.0, 0.0
                for d, g in zip(members, placed[arm]):
                    e, n = frame.to_enu(g.lat, g.lng)
                    cov = g.cov_en(geo.error_model_for(d.source).sigma_gps_m)
                    w = geo.sym2_inv((cov[0] * s2, cov[1] * s2, cov[2] * s2))
                    lam = geo.sym2_add(lam, w)
                    eta_e += w[0] * e + w[1] * n
                    eta_n += w[1] * e + w[2] * n
                inv = geo.sym2_inv(lam)
                site_pos[arm].append(_Placed(site.id, inv[0] * eta_e + inv[1] * eta_n,
                                             inv[1] * eta_e + inv[2] * eta_n))
                member_placed[arm].append(placed[arm])
    return kept, site_pos, member_placed


def pose_precondition(verdict_panos, bundle_ops, run_panos, base_params,
                      arms=PRECONDITION_ARMS, gt_merge_m=2.5, graded_panos=None):
    """Per-arm GT-to-site placement on ONE site set and ONE GT set (issue #42, study
    section 8 rec 3). Returns (rows, info): one row per arm.

    Why not just run evaluate_city once per arm: each arm then scores a different subset.
    A pose that pushes one member of a site above the horizon or past the 25 m cap drops
    that member, and re-association and GT placement drift with it, so the arms' p90s would
    describe different ramps -- the trap the #42 study hit in its first ablation. So:

    - Association is frozen from the `off` fuse (production's current output), and a site
      is scored only if EVERY arm places EVERY one of its operational members at the
      production cap (base_params.max_range_m). Each arm then refits the site's position
      from its own raycast of those members (the same inverse-covariance refit fuse uses).
    - A GT mark (verdict-true operational detection, or a non-unsure missed mark) is used
      only if every arm places it; ramps are grouped once, under `off`, and each arm puts
      a ramp at the mean of its own raycasts of that ramp's marks.
    - Distances are taken over the pool ramps that ALL arms match to a site within 5 m, so
      median and p90 compare the same ramps; recall at 2.5 m and 5 m is over the same pool.

    Because that intersection hides survivorship (a correction that makes hard marks
    unplaceable is scored on the easy rest -- #52's finding on GSV), each row also carries
    the arm's count of unplaceable GT marks and its recall on the OFF pool: ramps grouped
    from every mark `off` places; a ramp counts as recalled under an arm only if the arm
    places at least one of its marks AND it is self-detected or matched to a scored site
    within the radius (the ramp's position = the mean of the marks the arm places).

    The two `road-shuffled-*` arms are road mode on shuffled_grades copies of the panos:
    the control #52 asked for, since a Mapillary frame's pitch and its sequence grade come
    from the same SfM and subtracting one from the other might cancel shared SfM error
    rather than road slope.

    graded_panos (#51): {fs.GRADE_DEM: panos, fs.GRADE_SFM_SMOOTHED: panos} -- the same
    panos loaded with that grade source (fs.load_results(..., grade_source=...)). They
    feed the `road-dem`, `road-dem-shuffled-within` and `road-sfm-smoothed` arms; only
    grade_deg may differ from run_panos. Required when `arms` names any of those.

    Freezing the association from `off` favours `off` (its sites were built from its own
    geometry); so does the arms' shared GT being grouped under `off`. World precision is
    identical across arms by construction (membership is fixed); it is reported so the row
    reads like evaluate_city's.
    """
    by_id = {p.pano_id: p for p in run_panos}
    shuffled = {ARM_SHUFFLED_WITHIN: shuffled_grades(run_panos, True),
                ARM_SHUFFLED_ACROSS: shuffled_grades(run_panos, False)}
    graded_panos = graded_panos or {}
    if ARM_ROAD_DEM in arms or ARM_ROAD_DEM_SHUFFLED_WITHIN in arms:
        shuffled[ARM_ROAD_DEM] = (graded_panos[fs.GRADE_DEM], 0)
        shuffled[ARM_ROAD_DEM_SHUFFLED_WITHIN] = shuffled_grades(graded_panos[fs.GRADE_DEM],
                                                                 True)
    if ARM_ROAD_SFM_SMOOTHED in arms:
        shuffled[ARM_ROAD_SFM_SMOOTHED] = (graded_panos[fs.GRADE_SFM_SMOOTHED], 0)
    lookup = {arm: ({p.pano_id: p for p in shuffled[arm][0]} if arm in shuffled else by_id)
              for arm in arms}
    mode = {arm: fs.POSE_ROAD if arm in shuffled else arm for arm in arms}
    params = {arm: replace(base_params, apply_pose=mode[arm]) for arm in arms}
    off = params[fs.POSE_OFF]
    sites, frame, fuse_stats = fs.fuse(run_panos, off)
    poses = {arm: {} for arm in arms}

    def place(arm, pano_id, x, y):
        pose = poses[arm].get(pano_id)
        pano = lookup[arm][pano_id]
        if pose is None:
            pose = poses[arm][pano_id] = fs.pano_pose(pano, mode[arm])
        return geo.detection_ground_point(
            pose, x, y, camera_height=params[arm].camera_height_m,
            max_range_m=params[arm].max_range_m, errors=geo.error_model_for(pano.source),
            apply_pose=params[arm].rotates)

    # Sites: frozen membership, kept only if every arm places every operational member.
    op_sites = [s for s in sites if s.n_operational > 0]
    kept, site_pos, _placed = refit_frozen(op_sites, frame, place, arms,
                                           base_params.sigma_scale)

    # GT marks, each raycast under every arm (None = unplaceable under that arm).
    counts, warnings = gt_counts(), []
    counts['gt_panos'] = len(verdict_panos)
    marks, op_verdicts = [], {}
    for pid, entry, run_pano, ops, in_pool in judged_gt_panos(
            verdict_panos, bundle_ops, by_id, counts, warnings):
        for verdict, (stored_i, x, y, _c) in zip(entry['dets'], ops):
            op_verdicts[(pid, stored_i)] = verdict
            if verdict is True:
                marks.append((pid, x, y, 'det', in_pool))
        for mark in entry.get('missed', ()):
            if not mark.get('unsure'):
                marks.append((pid, mark['x'], mark['y'], 'missed', in_pool))
    enu = []   # per mark: {arm: (e, n) or None}
    for pid, x, y, _kind, _in_pool in marks:
        row = {}
        for arm in arms:
            g = place(arm, pid, x, y)
            row[arm] = None if g is None else frame.to_enu(g.lat, g.lng)
        enu.append(row)
    unplaceable = {arm: sum(1 for r in enu if r[arm] is None) for arm in arms}

    def ramps_from(mark_ids):
        """Pool ramps grouped under `off` from these marks: [(ramp, [mark index])]."""
        pts = [GTPoint(marks[i][0], marks[i][3], *enu[i][fs.POSE_OFF], marks[i][4])
               for i in mark_ids]
        index_of = {id(pt): i for pt, i in zip(pts, mark_ids)}
        return [(r, [index_of[id(pt)] for pt in r.points])
                for r in merge_gt_points(pts, gt_merge_m) if r.in_pool]

    def place_ramps(arm, ramps):
        out = []
        for k, (_r, ids) in enumerate(ramps):
            got = [enu[i][arm] for i in ids if enu[i][arm] is not None]
            out.append(None if not got else _Placed(k, sum(e for e, _ in got) / len(got),
                                                    sum(n for _, n in got) / len(got)))
        return out

    # (1) the shared GT set: marks every arm places.
    common_ids = [i for i, r in enumerate(enu) if all(v is not None for v in r.values())]
    pool = ramps_from(common_ids)
    # (2) the OFF pool: every mark off places, whatever the other arms do.
    off_pool = ramps_from([i for i, r in enumerate(enu) if r[fs.POSE_OFF] is not None])

    tp = fp = 0
    for site in kept:
        vs = [op_verdicts[(d.pano_id, d.det_index)] for d, _ in site.members
              if d.operational and (d.pano_id, d.det_index) in op_verdicts]
        if any(v is True or v == 'duplicate' for v in vs):
            tp += 1
        elif any(v is False for v in vs):
            fp += 1

    matched, off_pool_recall = {}, {}
    for arm in arms:
        placed = place_ramps(arm, pool)
        for radius in PRECONDITION_RADII:
            hits = match_one_to_one(placed, site_pos[arm], radius)
            matched[arm, radius] = {i: math.hypot(placed[i].e - s.e, placed[i].n - s.n)
                                    for i, s in hits.items()}
        placed_off = place_ramps(arm, off_pool)
        present = [p for p in placed_off if p is not None]
        for radius in PRECONDITION_RADII:
            hits = match_one_to_one(present, site_pos[arm], radius)
            hit_ids = {present[i].id for i in hits}
            recalled = sum(1 for k, (r, _ids) in enumerate(off_pool)
                           if placed_off[k] is not None
                           and (r.self_detected or k in hit_ids))
            off_pool_recall[arm, radius] = recalled / len(off_pool) if off_pool else None
    common = set(range(len(pool)))
    for arm in arms:
        common &= set(matched[arm, 5.0])

    posed = [p for p in run_panos if p.camera_pitch is not None and p.camera_roll is not None]
    graded = sum(1 for p in posed if p.grade_deg is not None)
    member_panos = [by_id[d.pano_id] for s in kept for d, _ in s.members if d.operational]
    posed_members = [p for p in member_panos
                     if p.camera_pitch is not None and p.camera_roll is not None]
    info = {'op_sites': len(op_sites), 'sites_scored': len(kept),
            'gt_marks': len(marks), 'gt_marks_dropped': len(marks) - len(common_ids),
            'pool_ramps': len(pool), 'off_pool_ramps': len(off_pool),
            'common_matched_5m': len(common),
            'panos': len(run_panos), 'posed_panos': len(posed),
            'gravity_fallback_panos': len(posed) - graded,
            'gravity_fallback_share_panos': (len(posed) - graded) / len(posed) if posed else None,
            'gravity_fallback_share_members':
                sum(1 for p in posed_members if p.grade_deg is None) / len(posed_members)
                if posed_members else None,
            'shuffle_within_unshuffled_frames': shuffled[ARM_SHUFFLED_WITHIN][1]
            if ARM_SHUFFLED_WITHIN in arms else None,
            'dem_shuffle_within_unshuffled_frames': shuffled[ARM_ROAD_DEM_SHUFFLED_WITHIN][1]
            if ARM_ROAD_DEM_SHUFFLED_WITHIN in arms else None,
            'warnings': warnings}

    def fallback_shares(arm):
        """Gravity-fallback shares under this arm's own grades (#51: the DEM arms leave a
        different set of frames ungraded; for the SfM arms this equals info's)."""
        ps = [lookup[arm][p.pano_id] for p in posed]
        ms = [lookup[arm][p.pano_id] for p in posed_members]
        return (sum(1 for p in ps if p.grade_deg is None) / len(ps) if ps else None,
                sum(1 for p in ms if p.grade_deg is None) / len(ms) if ms else None)
    rows = []
    for arm in arms:
        dists = [matched[arm, 5.0][i] for i in sorted(common)]
        row = {'arm': arm, 'sites_scored': len(kept),
               'sites_dropped_by_intersection': len(op_sites) - len(kept),
               'pool_ramps': len(pool), 'gt_marks_dropped_by_intersection': info['gt_marks_dropped'],
               'n_common_matched': len(common),
               'median_gt_to_site_m': _pct(dists, 0.5), 'p90_gt_to_site_m': _pct(dists, 0.9),
               'mean_gt_to_site_m': sum(dists) / len(dists) if dists else None,
               'precision': tp / (tp + fp) if tp + fp else None, 'tp': tp, 'fp': fp}
        for radius in PRECONDITION_RADII:
            hit = matched[arm, radius]
            recalled = sum(1 for i, (r, _ids) in enumerate(pool) if r.self_detected or i in hit)
            tag = f'{radius:g}'.replace('.', 'p')
            row[f'world_recall_{tag}m'] = recalled / len(pool) if pool else None
        row['off_pool_ramps'] = len(off_pool)
        for radius in PRECONDITION_RADII:
            tag = f'{radius:g}'.replace('.', 'p')
            row[f'recall_off_pool_{tag}m'] = off_pool_recall[arm, radius]
        row['gt_marks_unplaceable'] = unplaceable[arm]
        row['gt_marks_unplaceable_vs_off'] = unplaceable[arm] - unplaceable[fs.POSE_OFF]
        road_like = mode[arm] == fs.POSE_ROAD
        share_panos, share_members = fallback_shares(arm) if road_like else (None, None)
        row['gravity_fallback_share_panos'] = share_panos
        row['gravity_fallback_share_members'] = share_members
        rows.append(row)
    return rows, info


def precondition_verdict(rows_by_city, road_arm=fs.POSE_ROAD):
    """Apply the FIRST pre-registered rule (road_arm vs off) to {city: rows}.
    Returns (passes, reasons). #51 applies it to ARM_ROAD_DEM as its clause (iv)."""
    p90_fail, median_wins, reasons = [], [], []
    for city, rows in rows_by_city.items():
        r = {row['arm']: row for row in rows}
        off, road = r[fs.POSE_OFF], r[road_arm]
        dp90 = road['p90_gt_to_site_m'] - off['p90_gt_to_site_m']
        dmed = road['median_gt_to_site_m'] - off['median_gt_to_site_m']
        if dp90 > PRECONDITION_P90_TOLERANCE_M:
            p90_fail.append(city)
        if dmed < 0:
            median_wins.append(city)
        reasons.append(f'{city}: p90 {road_arm}-off {dp90:+.3f} m, '
                       f'median {road_arm}-off {dmed:+.3f} m')
    passes = not p90_fail and len(median_wins) >= PRECONDITION_MIN_MEDIAN_WINS
    reasons.append(f'p90 worse than off by > {PRECONDITION_P90_TOLERANCE_M} m in: '
                   f'{", ".join(p90_fail) or "none"}; median improves in '
                   f'{len(median_wins)} of {len(rows_by_city)} '
                   f'(needs {PRECONDITION_MIN_MEDIAN_WINS}): {", ".join(median_wins) or "none"}')
    return passes, reasons


def control_verdict(rows_by_city, road_arm=fs.POSE_ROAD, shuffle_arm=ARM_SHUFFLED_WITHIN):
    """Apply the SECOND pre-registered rule (the shuffled-grade control) to {city: rows}.
    Returns (passes, clauses, reasons) with clauses = {'i': bool, 'ii': bool, 'iii': bool}.
    #51 applies it to (ARM_ROAD_DEM, ARM_ROAD_DEM_SHUFFLED_WITHIN) as its (i)-(iii)."""
    beats, recall_fail, unplace_fail, reasons = [], [], [], []
    for city, rows in rows_by_city.items():
        r = {row['arm']: row for row in rows}
        off, road, shuf = r[fs.POSE_OFF], r[road_arm], r[shuffle_arm]
        d90 = shuf['p90_gt_to_site_m'] - road['p90_gt_to_site_m']
        dmed = shuf['median_gt_to_site_m'] - road['median_gt_to_site_m']
        if d90 > CONTROL_MARGIN_M and dmed > CONTROL_MARGIN_M:
            beats.append(city)
        drec = road['recall_off_pool_2p5m'] - off['recall_off_pool_2p5m']
        if drec < -CONTROL_MAX_RECALL_DROP:
            recall_fail.append(city)
        extra = road['gt_marks_unplaceable'] - off['gt_marks_unplaceable']
        limit = CONTROL_MAX_UNPLACEABLE_FRAC * off['off_pool_ramps']
        if extra > limit:
            unplace_fail.append(city)
        if (road_arm, shuffle_arm) == (fs.POSE_ROAD, ARM_SHUFFLED_WITHIN):
            name_s, name_r = 'shuffled-within', 'road'      # #74's committed wording
        else:
            name_s, name_r = shuffle_arm, road_arm
        reasons.append(f'{city}: {name_s} minus {name_r} p90 {d90:+.3f} m, median '
                       f'{dmed:+.3f} m; off-pool R@2.5 {name_r}-off {100 * drec:+.1f} pt; '
                       f'unplaceable marks {name_r}-off {extra:+d} (limit {limit:.1f})')
    clauses = {'i': len(beats) >= CONTROL_MIN_CITIES, 'ii': not recall_fail,
               'iii': not unplace_fail}
    reasons.append(f'(i) {road_arm} beats {shuffle_arm} by > {CONTROL_MARGIN_M} m on p90 AND '
                   f'median in {len(beats)} of {len(rows_by_city)} (needs {CONTROL_MIN_CITIES}): '
                   f'{", ".join(beats) or "none"} -> {"PASS" if clauses["i"] else "FAIL"}')
    reasons.append(f'(ii) off-pool recall@2.5 drops > {100 * CONTROL_MAX_RECALL_DROP:.1f} pt in: '
                   f'{", ".join(recall_fail) or "none"} -> {"PASS" if clauses["ii"] else "FAIL"}')
    reasons.append(f'(iii) unplaceable marks exceed off by > '
                   f'{100 * CONTROL_MAX_UNPLACEABLE_FRAC:.0f}% of the off pool in: '
                   f'{", ".join(unplace_fail) or "none"} -> {"PASS" if clauses["iii"] else "FAIL"}')
    return all(clauses.values()), clauses, reasons


def dem_verdict(rows_by_city):
    """The #51 rule (see the comment above CONTROL_MARGIN_M): clauses (i)-(iii) are the
    shuffled-grade control on road-dem vs road-dem-shuffled-within, (iv) the first rule on
    road-dem vs off. Returns (passes, clauses, reasons); passes only if all four do."""
    _ok, clauses, reasons = control_verdict(rows_by_city, ARM_ROAD_DEM,
                                            ARM_ROAD_DEM_SHUFFLED_WITHIN)
    first, first_reasons = precondition_verdict(rows_by_city, ARM_ROAD_DEM)
    clauses = {**clauses, 'iv': first}
    reasons = reasons + first_reasons + [
        f'(iv) first rule, {ARM_ROAD_DEM} vs off -> {"PASS" if first else "FAIL"}']
    passes = all(clauses.values())
    reasons.append(f'#51 VERDICT: {"PASS" if passes else "FAIL"} on '
                   + ', '.join(f'({k}) {"pass" if v else "FAIL"}' for k, v in clauses.items())
                   + (" -> AUTO_ROAD_SOURCES = ('mapillary',) with the DEM grade" if passes
                      else ' -> #51 closes negative: AUTO_ROAD_SOURCES stays (), '
                           '--apply-pose road --grade-source dem stays opt-in'))
    return passes, clauses, reasons


def format_precondition(city, rows, info):
    fmt = lambda v, f='.3f': '—' if v is None else format(v, f)  # noqa: E731
    lines = [f"== {city}: pose precondition (#42) -- one site set, one GT set, "
             f"{rows[0]['sites_scored']} of {info['op_sites']} operational sites scored "
             f"({rows[0]['sites_dropped_by_intersection']} dropped: some arm cannot place a "
             f"member), {info['pool_ramps']} pool ramps, {info['common_matched_5m']} matched "
             f"within 5 m under every arm; {info['gt_marks_dropped']} of {info['gt_marks']} "
             f"GT marks unplaceable under some arm; off pool {info['off_pool_ramps']} ramps",
             f"{'arm':>24}  {'median':>7}  {'p90':>7}  {'mean':>7}  {'R@2.5':>6}  "
             f"{'R@5':>6}  {'P':>6}  {'offR@2.5':>8}  {'offR@5':>6}  {'unplaced':>8}"]
    for r in rows:
        lines.append(f"{r['arm']:>24}  {fmt(r['median_gt_to_site_m']):>7}  "
                     f"{fmt(r['p90_gt_to_site_m']):>7}  {fmt(r['mean_gt_to_site_m']):>7}  "
                     f"{fmt(r['world_recall_2p5m']):>6}  {fmt(r['world_recall_5m']):>6}  "
                     f"{fmt(r['precision']):>6}  {fmt(r['recall_off_pool_2p5m']):>8}  "
                     f"{fmt(r['recall_off_pool_5m']):>6}  {r['gt_marks_unplaceable']:>8d}")
    lines.append(f"road-mode gravity fallback: {info['gravity_fallback_panos']} of "
                 f"{info['posed_panos']} posed panos ({fmt(info['gravity_fallback_share_panos'], '.1%')}), "
                 f"{fmt(info['gravity_fallback_share_members'], '.1%')} of scored members; "
                 f"shuffle-within kept its own grade on {info['shuffle_within_unshuffled_frames']} "
                 f"frames (single-graded-frame sequences)")
    for w in info['warnings']:
        lines.append(f'warning: {w}')
    return '\n'.join(lines)


def format_report(city, r):
    c, b, p = r['counts'], r['buckets'], r['precision']
    pct = lambda v: 'n/a' if v is None else f'{v:.3f}'  # noqa: E731
    ci = lambda t: f'[{t[0]:.3f}, {t[1]:.3f}]'          # noqa: E731
    lines = [
        f"== {city}: world-space fusion eval "
        f"(match radius {r['params']['match_radius_m']} m, "
        f"GT merge {r['params']['gt_merge_m']} m, "
        f"camera height {r['params']['camera_height_m']}, "
        f"pose {r['params']['apply_pose']})",
        f"run: {r['fuse']['n_panos']} panos -> {r['fuse']['n_sites']} sites "
        f"({r['fuse']['n_operational_sites']} operational, "
        f"{r['fuse']['n_multi_pano_sites']} multi-pano)",
        f"GT: {c['gt_panos']} panos ({c['judged']} fully judged, "
        f"{c['partial']} partial, {c['skipped']} skipped, "
        f"{c['no_pool']} without missed-check) -> {c['placeable']} placeable "
        f"points (+{c['unplaceable']} unplaceable) -> {c['gt_ramps']} ramps "
        f"({c['cross_pano_merges']} cross-pano merges), "
        f"{r['n_pool_ramps']} in the recall pool",
        f"world recall     {pct(r['world_recall'])} {ci(r['world_recall_ci'])}",
        f"own-view recall  {pct(r['own_view_recall'])} "
        f"{ci(r['own_view_recall_ci'])}",
        f"union lift       +{b['recovered_other_view']} ramps recovered from "
        f"other views "
        f"({b['recovered_other_view'] / r['n_pool_ramps']:.3f} of the pool)"
        if r['n_pool_ramps'] else "union lift       n/a",
        f"buckets          self {b['self_detected']}, other-view "
        f"{b['recovered_other_view']}, subthreshold-only "
        f"{b['subthreshold_only']}, unmatched {b['unmatched']}",
        f"world precision  {pct(p['value'])} {ci(p['ci'])}  "
        f"(TP {p['tp']}, FP {p['fp']}, unsure-only {p['unsure_only']} excluded)",
    ]
    d = r['dual_ramp']
    if d['pairs']:
        lines.append(f"dual ramps       {d['pairs']} same-pano GT pairs < 5 m: "
                     f"{d['both_matched']} kept separate, {d['one_matched']} "
                     f"half-matched, {d['neither']} unmatched")
    v = r['vintage']['all']
    if v:
        order = ['0', '1-18', '19-36', '>36', 'unknown']
        lines.append("vintage          member pairs by capture delta (months): "
                     + ', '.join(f"{b} = {v[b]}" for b in order if b in v))
        for key, label in (('tp_sites', 'TP-site'), ('fp_sites', 'FP-site')):
            vv = r['vintage'][key]
            if vv:
                lines.append(f"                 {label} pairs: "
                             + ', '.join(f"{b} = {vv[b]}"
                                         for b in order if b in vv))
    cal = r['calibration']
    if cal is None:
        lines.append("promotion        skipped: this run stores no "
                     "sub-threshold detections")
    else:
        base = r['world_recall']
        lines.append(f"promotion        {len(cal['profiles'])} pool ramps missed "
                     f"at {r['params']['min_confidence']}; world recall if "
                     "sub-threshold sites with >=k views at conf>=f were "
                     "accepted (base "
                     f"{'n/a' if base is None else format(base, '.3f')}):")
        lines.append(f"{'floor':>18} " + ' '.join(f'{f:>6.2f}'
                                                  for f in CAL_FLOORS))
        for k in (1, 2, 3):
            row = {p['floor']: p['recall_if_promoted']
                   for p in cal['promotion'] if p['k'] == k}
            lines.append(f"{'k>=' + str(k):>18} "
                         + ' '.join('   n/a' if row[f] is None
                                    else f'{row[f]:>6.3f}' for f in CAL_FLOORS))
        g25 = next(g for g in cal['ghost'] if abs(g['floor'] - 0.25) < 1e-9)
        fmt = lambda x: 'n/a' if x is None else f'{x:.3f}'  # noqa: E731
        lines.append(
            "ghost check      other-pano support >=1 view at conf>=0.25: "
            f"verdict-true dets {fmt(g25['true_ge1'])} "
            f"(n={g25['n_true']}) vs verdict-false {fmt(g25['false_ge1'])} "
            f"(n={g25['n_false']})")
    if r['self_detected_without_site']:
        lines.append(f"note: {r['self_detected_without_site']} self-detected "
                     "ramps had no operational site within radius (projector/"
                     "associator disagreement — should be rare)")
    for w in r['warnings']:
        lines.append(f'warning: {w}')
    return '\n'.join(lines)


def radius_sweep_table(results_by_radius):
    lines = ["match-radius sweep:",
             f"{'radius_m':>9} {'world_recall':>13} {'own_view':>9} "
             f"{'precision':>10} {'unmatched':>10}"]
    for radius, r in results_by_radius:
        pct = lambda x: 'n/a' if x is None else f'{x:.3f}'  # noqa: E731
        lines.append(f"{radius:>9.1f} {pct(r['world_recall']):>13} "
                     f"{pct(r['own_view_recall']):>9} "
                     f"{pct(r['precision']['value']):>10} "
                     f"{r['buckets']['unmatched']:>10}")
    return '\n'.join(lines)


def vintage_ablation_table(results_by_window):
    lines = ["vintage ablation (max member capture delta enforced at fusion):",
             f"{'window':>10} {'world_recall':>13} {'precision':>10} "
             f"{'multi_pano':>11}"]
    for window, r in results_by_window:
        pct = lambda x: 'n/a' if x is None else f'{x:.3f}'  # noqa: E731
        name = 'none' if window is None else f'{window} mo'
        lines.append(f"{name:>10} {pct(r['world_recall']):>13} "
                     f"{pct(r['precision']['value']):>10} "
                     f"{r['fuse']['n_multi_pano_sites']:>11}")
    return '\n'.join(lines)


def write_outputs(out_dir, report_text, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.md').write_text(report_text + '\n', encoding='utf-8')
    cal = result['calibration']
    if cal is None:
        return
    with open(out_dir / 'calibration.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, ['floor', 'k', 'promoted_ramps',
                               'recall_if_promoted'])
        w.writeheader()
        w.writerows(cal['promotion'])
    with open(out_dir / 'ghost_check.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, list(cal['ghost'][0].keys()))
        w.writeheader()
        w.writerows(cal['ghost'])
    with open(out_dir / 'support_profiles.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ramp_index', 'panos', 'site_id']
                   + [f'k_at_{f:.2f}' for f in CAL_FLOORS])
        for pr in cal['profiles']:
            w.writerow([pr['ramp_index'], ';'.join(pr['panos']),
                        '' if pr['site_id'] is None else pr['site_id']]
                       + [pr['k'][f] for f in CAL_FLOORS])


def load_city_files(city, benchmark_root, run_dir, read_heights=True, height_table=None):
    with open(benchmark_root / city / 'verdicts.json', encoding='utf-8') as f:
        verdicts = json.load(f)
    bundle_ops = {}
    with open(benchmark_root / city / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bundle_ops[rec['pano']['panorama_id']] = \
                    [(d['x_normalized'], d['y_normalized'], d['confidence'])
                     for d in rec['detections']]
    run_panos, skipped = fs.load_results(run_dir / 'results.jsonl',
                                         read_heights=read_heights, height_table=height_table)
    return verdicts['panos'], bundle_ops, run_panos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', help='benchmark split name, e.g. paterson')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--gt-merge-m', type=float, default=2.5)
    ap.add_argument('--radius-sweep', type=float, nargs='*',
                    default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument('--camera-height-m', type=fs.camera_height_arg,
                    default=geo.DEFAULT_CAMERA_HEIGHT_M,
                    help='raycast height for fusion AND GT placement: meters, or '
                         '"per-pano" for GSV depth-measured heights (#40), or "per-rig" '
                         "for the run's camera_heights.json (#53). A non-default "
                         'value needs --out, so it cannot overwrite the published report')
    # Default OFF, not FuseParams' `auto`: runs/<city>/fusion_eval/ reports were all
    # produced flat, and re-running with the defaults must still reproduce them.
    ap.add_argument('--apply-pose', choices=fs.POSE_MODES, default=fs.POSE_OFF,
                    help='camera pose for fusion AND GT placement (#42; see fuse_sites.py; '
                         'default off, which reproduces the published reports). Any other '
                         'value needs --out, like --camera-height-m')
    ap.add_argument('--pose-precondition', action='store_true',
                    help='instead of the report: GT-to-site median/p90 and world P/R under '
                         'every --apply-pose arm, on one site set and one GT set at the '
                         'production range cap (#42; see pose_precondition). Needs --out')
    ap.add_argument('--vintage-ablation', action='store_true',
                    help='re-fuse at capture-delta windows 0/18/36/none and '
                         'compare world P/R (the #27 open question)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/fusion_eval)')
    args = ap.parse_args()

    if args.camera_height_m != geo.DEFAULT_CAMERA_HEIGHT_M and args.out is None:
        ap.error('--camera-height-m other than the default changes the scoring frame; '
                 'pass --out so the default fusion_eval/ report is not overwritten')
    if args.pose_precondition and args.out is None:
        ap.error('--pose-precondition needs --out')
    if args.apply_pose != fs.POSE_OFF and args.out is None:
        ap.error('--apply-pose other than the default changes the scoring frame; '
                 'pass --out so the default fusion_eval/ report is not overwritten')
    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    height_table = None
    if args.camera_height_m == geo.PER_RIG:
        height_table = run_dir / fs.HEIGHT_TABLE_NAME
        if not height_table.exists():
            sys.exit(f'per-rig needs a camera-height table; none at {height_table} '
                     '(scripts/mapillary_height.py writes it)')
    try:
        verdict_panos, bundle_ops, run_panos = load_city_files(
            args.city, args.benchmark_root, run_dir,
            read_heights=args.camera_height_m == geo.PER_PANO, height_table=height_table)
    except ValueError as e:        # a table measured on another file, or a GSV run
        sys.exit(str(e))
    # Fusion at the BENCHMARK threshold, not the production operating point: the bundle's
    # verdicts and the committed reports are keyed to it (detectors/__init__.py).
    # mask_rig=False alongside the pinned tier: runs/<city>/fusion_eval/ is git-tracked by
    # the same convention as the tilt CSVs, so re-running must still reproduce it.
    params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                           camera_height_m=args.camera_height_m,
                           apply_pose=args.apply_pose)
    if args.pose_precondition:
        rows, info = pose_precondition(verdict_panos, bundle_ops, run_panos,
                                       replace(params, apply_pose=fs.POSE_OFF),
                                       gt_merge_m=args.gt_merge_m)
        print(format_precondition(args.city, rows, info))
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / 'pose_precondition.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.out}')
        return
    prefused = fs.fuse(run_panos, params)

    result = evaluate_city(verdict_panos, bundle_ops, run_panos, params,
                           match_radius_m=args.match_radius_m,
                           gt_merge_m=args.gt_merge_m, prefused=prefused)
    sections = [format_report(args.city, result)]

    sweep = [r for r in args.radius_sweep
             if abs(r - args.match_radius_m) > 1e-9]
    if sweep:
        by_radius = [(args.match_radius_m, result)]
        for radius in sweep:
            by_radius.append((radius, evaluate_city(
                verdict_panos, bundle_ops, run_panos, params,
                match_radius_m=radius, gt_merge_m=args.gt_merge_m,
                prefused=prefused)))
        by_radius.sort(key=lambda t: t[0])
        sections.append(radius_sweep_table(by_radius))

    if args.vintage_ablation:
        by_window = []
        for window in (0, 18, 36, None):
            p = replace(params, max_vintage_months=window)
            by_window.append((window, evaluate_city(
                verdict_panos, bundle_ops, run_panos, p,
                match_radius_m=args.match_radius_m,
                gt_merge_m=args.gt_merge_m)))
        sections.append(vintage_ablation_table(by_window))

    report_text = '\n\n'.join(sections)
    print(report_text)
    out_dir = args.out or run_dir / 'fusion_eval'
    write_outputs(out_dir, report_text, result)
    print(f'\nwrote {out_dir}')


if __name__ == '__main__':
    main()
