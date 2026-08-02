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
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
from detectors import OPERATIONAL_CONFIDENCE  # noqa: E402


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


def build_gt(verdict_panos, bundle_ops, run_panos_by_id, params, frame):
    """World GT points + the (pano_id, stored_index) -> verdict map.

    Returns (points, op_verdicts, counts, warnings). Panos are skipped (with a
    warning, mirroring rampnet.validation's skip-and-warn) when they are missing
    from the run or their operational detections drifted from the frozen bundle.
    """
    points, op_verdicts, warnings = [], {}, []
    counts = {'gt_panos': len(verdict_panos), 'judged': 0, 'partial': 0,
              'skipped': 0, 'no_pool': 0, 'unplaceable': 0, 'placeable': 0,
              'unsure_missed': 0}
    for pid in sorted(verdict_panos):
        entry = verdict_panos[pid]
        run_pano = run_panos_by_id.get(pid)
        if run_pano is None:
            warnings.append(f'{pid}: GT pano missing from the run')
            counts['skipped'] += 1
            continue
        ops = [(i, x, y, c) for i, x, y, c in run_pano.detections
               if c >= OPERATIONAL_CONFIDENCE]
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

        pose = geo.pano_pose({'lat': run_pano.lat, 'lng': run_pano.lng,
                              'camera_heading': run_pano.camera_heading,
                              'camera_pitch': run_pano.camera_pitch,
                              'camera_roll': run_pano.camera_roll,
                              'source': run_pano.source})
        errors = geo.error_model_for(run_pano.source)

        def place(x, y, kind):
            g = geo.detection_ground_point(
                pose, x, y, camera_height=params.camera_height_m,
                max_range_m=params.max_range_m, errors=errors,
                apply_pose=params.apply_pose)
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


def evaluate_city(verdict_panos, bundle_ops, run_panos, params,
                  match_radius_m=5.0, gt_merge_m=2.5):
    """The full stage-3 join for one city; returns a result dict (no I/O)."""
    sites, frame, fuse_stats = fs.fuse(run_panos, params)
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
    sub_hits = {unmatched_idx[j] for j in matched_sub}

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
    for site in op_sites:
        verdicts = [op_verdicts[(d.pano_id, d.det_index)]
                    for d, _ in site.members
                    if d.operational and (d.pano_id, d.det_index) in op_verdicts]
        if not verdicts:
            continue
        if any(v is True or v == 'duplicate' for v in verdicts):
            tp += 1
        elif any(v is False for v in verdicts):
            fp += 1
        else:
            unsure_only += 1

    n_pool = len(pool)
    world_recalled = buckets['self_detected'] + buckets['recovered_other_view']
    return {
        'params': {'match_radius_m': match_radius_m, 'gt_merge_m': gt_merge_m,
                   'min_confidence': params.min_confidence,
                   'max_range_m': params.max_range_m},
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
    }


def format_report(city, r):
    c, b, p = r['counts'], r['buckets'], r['precision']
    pct = lambda v: 'n/a' if v is None else f'{v:.3f}'  # noqa: E731
    ci = lambda t: f'[{t[0]:.3f}, {t[1]:.3f}]'          # noqa: E731
    lines = [
        f"== {city}: world-space fusion eval "
        f"(match radius {r['params']['match_radius_m']} m, "
        f"GT merge {r['params']['gt_merge_m']} m)",
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
    if r['self_detected_without_site']:
        lines.append(f"note: {r['self_detected_without_site']} self-detected "
                     "ramps had no operational site within radius (projector/"
                     "associator disagreement — should be rare)")
    for w in r['warnings']:
        lines.append(f'warning: {w}')
    return '\n'.join(lines)


def load_city_files(city, benchmark_root, run_dir):
    verdicts = json.load(open(benchmark_root / city / 'verdicts.json',
                              encoding='utf-8'))
    bundle_ops = {}
    with open(benchmark_root / city / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bundle_ops[rec['pano']['panorama_id']] = \
                    [(d['x_normalized'], d['y_normalized'], d['confidence'])
                     for d in rec['detections']]
    run_panos, skipped = fs.load_results(run_dir / 'results.jsonl')
    return verdicts['panos'], bundle_ops, run_panos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', help='benchmark split name, e.g. paterson')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--gt-merge-m', type=float, default=2.5)
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    verdict_panos, bundle_ops, run_panos = load_city_files(
        args.city, args.benchmark_root, run_dir)
    result = evaluate_city(verdict_panos, bundle_ops, run_panos, fs.FuseParams(),
                           match_radius_m=args.match_radius_m,
                           gt_merge_m=args.gt_merge_m)
    print(format_report(args.city, result))


if __name__ == '__main__':
    main()
