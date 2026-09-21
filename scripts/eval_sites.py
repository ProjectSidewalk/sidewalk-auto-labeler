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
    has_subfloor = any(c < OPERATIONAL_CONFIDENCE
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
        'calibration': calibration,
        'vintage': vintage,
        'dual_ramp': dual,
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


def load_city_files(city, benchmark_root, run_dir):
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
    ap.add_argument('--radius-sweep', type=float, nargs='*',
                    default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument('--vintage-ablation', action='store_true',
                    help='re-fuse at capture-delta windows 0/18/36/none and '
                         'compare world P/R (the #27 open question)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/fusion_eval)')
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    verdict_panos, bundle_ops, run_panos = load_city_files(
        args.city, args.benchmark_root, run_dir)
    params = fs.FuseParams()
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
