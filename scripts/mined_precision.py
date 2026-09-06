"""Precision of hard positives mined from multi-view consensus (RampNet#158 step 1).

RampNet#102 proposes a training-label source that needs no inventory and no human
review: a fused site corroborated by >= 3 operational panos is almost certainly a
ramp, so every nearby pano that produced no detection for it is a miss at a known
world position - a training target once projected back into that pano. Before
anyone builds that miner, #102 asks for the precision of what it would produce,
measured against verdicts that already exist. This script is that measurement.

For one benchmark city it re-fuses the run in memory (fuse_sites.fuse, same code
path as production and eval_sites), takes every fully judged benchmark pano that
stood within R of a strong site without being one of its members, projects the
site into that pano (geo.ground_point_to_pano), and asks the reviewer's ground
truth what is there, in world space with the eval's own match radius:

    tp                the reviewer marked a MISSED ramp there: a true mined target
    fp                the pano was attested clean there: occlusion, ghost, out of view
    already_detected  a verdict-true detection is there: an association gap, not a miss
    unsure            an unsure missed mark is there
    false_det_nearby  a verdict-false detection is there (the reviewer looked and said no)
    unadjudicable     nothing there, but the pano's missed-ramp check was never attested

Headline precision = tp / (tp + fp), Wilson intervals, stratified by range, by
site support and by site confidence. Reads the same files as eval_sites.py
(benchmark/<city>/{verdicts.json,records.jsonl} as data, runs/<city>/results.jsonl)
and never touches the network. Writes runs/<city>/mined_precision/{report.md,
candidates.csv}; the CSV has one row per (site, pano) candidate so any bucket can
be eyeballed.

Usage:
    python scripts/mined_precision.py richmond
    python scripts/mined_precision.py paterson --radius 10 15 25 --min-panos 3
"""
import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
from detectors import OPERATIONAL_CONFIDENCE  # noqa: E402

BUCKETS = ('tp', 'fp', 'already_detected', 'unsure', 'false_det_nearby',
           'unadjudicable')
HEADLINE = ('tp', 'fp')
RANGE_BUCKETS = ((8.0, '0-8'), (12.0, '8-12'), (18.0, '12-18'), (25.0, '18-25'))
CONF_SPLIT = 0.9
# The pre-registered decision rule (RampNet#158 step 1).
RULE_BUILD, RULE_VISIBILITY = 0.80, 0.50


@dataclass
class Candidate:
    site_id: int
    pano_id: str
    range_m: float          # camera-to-site distance == projected range (flat ground)
    bearing_deg: float
    x_norm: float
    y_norm: float
    n_op_panos: int
    best_conf: float
    bucket: str
    nearest_gt_kind: str    # '' when nothing within the match radius
    nearest_gt_m: float     # inf when nothing within the match radius


def judged_panos(verdict_panos, bundle_ops, run_by_id):
    """{pano_id: verdict entry} for panos eval_sites.build_gt would score - present
    in the run, operational detections unchanged since the bundle was frozen, one
    verdict per detection and none of them None. Mirrors build_gt's gate so the
    candidate set and the GT points describe the same panos (a judged pano with
    no detections and no marks emits no GT point, yet is exactly the pano that
    yields an fp candidate)."""
    out = {}
    for pid, entry in verdict_panos.items():
        run_pano = run_by_id.get(pid)
        if run_pano is None:
            continue
        ops = [(x, y, c) for _, x, y, c in run_pano.detections
               if c >= OPERATIONAL_CONFIDENCE]
        if bundle_ops.get(pid) != ops or len(entry['dets']) != len(ops):
            continue
        if any(d is None for d in entry['dets']):
            continue
        out[pid] = entry
    return out


def _in_pool(entry):
    return (bool(entry['no_missed']) or bool(entry['missed'])) \
        if 'no_missed' in entry else True


def _pose_and_errors(run_pano):
    pose = geo.pano_pose({'lat': run_pano.lat, 'lng': run_pano.lng,
                          'camera_heading': run_pano.camera_heading,
                          'camera_pitch': run_pano.camera_pitch,
                          'camera_roll': run_pano.camera_roll,
                          'source': run_pano.source})
    return pose, geo.error_model_for(run_pano.source)


def gt_points_by_pano(verdict_panos, bundle_ops, run_by_id, params, frame):
    """{pano_id: [(kind, e, n)]} over judged panos. kinds: 'det' (verdict-true
    operational detection) and 'missed' come from eval_sites.build_gt; 'unsure'
    (unsure missed marks) and 'false_det' (verdict-false detections), which
    build_gt deliberately skips, are placed here with the same projector."""
    points, _op_verdicts, _counts, warnings = es.build_gt(
        verdict_panos, bundle_ops, run_by_id, params, frame)
    by_pano = {pid: [] for pid in judged_panos(verdict_panos, bundle_ops, run_by_id)}
    for pt in points:
        by_pano.setdefault(pt.pano_id, []).append((pt.kind, pt.e, pt.n))
    for pid in by_pano:
        entry, run_pano = verdict_panos[pid], run_by_id[pid]
        pose, errors = _pose_and_errors(run_pano)

        def place(x, y, kind):
            g = geo.detection_ground_point(
                pose, x, y, camera_height=params.camera_height_m,
                max_range_m=params.max_range_m, errors=errors,
                apply_pose=params.apply_pose)
            if g is not None:
                e, n = frame.to_enu(g.lat, g.lng)
                by_pano[pid].append((kind, e, n))

        ops = [(x, y) for _, x, y, c in run_pano.detections
               if c >= OPERATIONAL_CONFIDENCE]
        for verdict, (x, y) in zip(entry['dets'], ops):
            if verdict is False:
                place(x, y, 'false_det')
        for mark in entry.get('missed', ()):
            if mark.get('unsure'):
                place(mark['x'], mark['y'], 'unsure')
    return by_pano, warnings


def strong_sites(sites, min_panos):
    """Sites with >= min_panos distinct panos contributing an operational member,
    as [(site, n_op_panos, best_conf)]."""
    out = []
    for site in sites:
        op_panos = {d.pano_id for d, _ in site.members if d.operational}
        if len(op_panos) >= min_panos:
            out.append((site, len(op_panos),
                        max(d.conf for d, _ in site.members if d.operational)))
    return out


def mine_candidates(strong, judged, run_by_id, gt_by_pano, frame, params,
                    max_radius_m, match_m):
    """Every (strong site, judged non-member pano within max_radius_m) pair,
    classified. Also returns how many pairs were excluded because the pano is a
    member through a sub-threshold detection only (the model did respond, so it
    is not a miss - but not a training positive either)."""
    cands, excluded_subfloor = [], 0
    pano_enu = {pid: frame.to_enu(run_by_id[pid].lat, run_by_id[pid].lng)
                for pid in judged}
    for site, n_op_panos, best_conf in strong:
        site_lat, site_lng = frame.to_latlng(site.e, site.n)
        op_panos = {d.pano_id for d, _ in site.members if d.operational}
        for pid, (pe, pn) in pano_enu.items():
            if math.hypot(pe - site.e, pn - site.n) > max_radius_m:
                continue
            if pid in site.pano_ids:
                if pid not in op_panos:
                    excluded_subfloor += 1
                continue
            pose, _ = _pose_and_errors(run_by_id[pid])
            proj = geo.ground_point_to_pano(
                pose, site_lat, site_lng, camera_height=params.camera_height_m,
                max_range_m=max_radius_m)
            if proj is None:
                continue
            kind, dist = '', math.inf
            for k, e, n in gt_by_pano.get(pid, ()):
                d = math.hypot(e - site.e, n - site.n)
                if d <= match_m and d < dist:
                    kind, dist = k, d
            if kind == 'missed':
                bucket = 'tp'
            elif kind == 'det':
                bucket = 'already_detected'
            elif kind == 'unsure':
                bucket = 'unsure'
            elif kind == 'false_det':
                bucket = 'false_det_nearby'
            else:
                bucket = 'fp' if _in_pool(judged[pid]) else 'unadjudicable'
            cands.append(Candidate(site.id, pid, proj.range_m, proj.bearing_deg,
                                   proj.x_norm, proj.y_norm, n_op_panos,
                                   best_conf, bucket, kind, dist))
    cands.sort(key=lambda c: (c.range_m, c.site_id, c.pano_id))
    return cands, excluded_subfloor


def yield_all_panos(strong, run_panos, frame, radii_m):
    """#102's yield table, recomputed: for each radius, how many (strong site,
    non-member run pano) pairs exist across the WHOLE run - the number of targets
    a miner would emit - and that count as a multiple of the run's operational
    detections. A gross mismatch with #102's table means the candidate definition
    drifted."""
    grid = geo.GridIndex(max(radii_m))
    enu = {}
    for p in run_panos:
        enu[p.pano_id] = frame.to_enu(p.lat, p.lng)
        grid.add(*enu[p.pano_id], p)
    counts = {r: 0 for r in radii_m}
    for site, _, _ in strong:
        for p in grid.near(site.e, site.n):
            if p.pano_id in site.pano_ids:
                continue
            pe, pn = enu[p.pano_id]
            d = math.hypot(pe - site.e, pn - site.n)
            for r in radii_m:
                if d <= r:
                    counts[r] += 1
    n_ops = sum(1 for p in run_panos for _, _, _, c in p.detections
                if c >= OPERATIONAL_CONFIDENCE)
    return counts, n_ops


def tally(cands):
    t = {b: 0 for b in BUCKETS}
    for c in cands:
        t[c.bucket] += 1
    return t


def precision_row(label, cands):
    t = tally(cands)
    n = t['tp'] + t['fp']
    p = t['tp'] / n if n else None
    lo, hi = es.wilson(t['tp'], n)
    return {'stratum': label, 'candidates': len(cands), 'tp': t['tp'], 'fp': t['fp'],
            'precision': p, 'ci_lo': lo, 'ci_hi': hi,
            'already_detected': t['already_detected'], 'unsure': t['unsure'],
            'false_det_nearby': t['false_det_nearby'],
            'unadjudicable': t['unadjudicable']}


def strata(cands, radii_m):
    rows = {'radius': [], 'range': [], 'support': [], 'confidence': []}
    for r in sorted(radii_m):
        rows['radius'].append(precision_row(f'<= {r:g} m',
                                            [c for c in cands if c.range_m <= r]))
    lo = 0.0
    for hi, name in RANGE_BUCKETS:
        if lo >= max(radii_m):
            break
        rows['range'].append(precision_row(
            f'{name} m', [c for c in cands if lo < c.range_m <= hi]))
        lo = hi
    for label, test in (('3 panos', lambda c: c.n_op_panos == 3),
                        ('4 panos', lambda c: c.n_op_panos == 4),
                        ('>= 5 panos', lambda c: c.n_op_panos >= 5)):
        rows['support'].append(precision_row(label, [c for c in cands if test(c)]))
    rows['confidence'].append(precision_row(
        f'best member >= {CONF_SPLIT}', [c for c in cands if c.best_conf >= CONF_SPLIT]))
    rows['confidence'].append(precision_row(
        f'best member < {CONF_SPLIT}', [c for c in cands if c.best_conf < CONF_SPLIT]))
    return rows


def rule_reading(p):
    if p is None:
        return 'no adjudicable candidates'
    if p >= RULE_BUILD:
        return f'>= {RULE_BUILD:.2f}: build the miner'
    if p >= RULE_VISIBILITY:
        return f'{RULE_VISIBILITY:.2f}-{RULE_BUILD:.2f}: add the visibility test before mining'
    return f'< {RULE_VISIBILITY:.2f}: drop this label source'


def run_city(verdict_panos, bundle_ops, run_panos, params, radii_m=(10.0, 15.0),
             min_panos=3, match_m=5.0):
    """The whole check for one city; returns (result dict, candidates). No I/O."""
    sites, frame, fuse_stats = fs.fuse(run_panos, params)
    run_by_id = {p.pano_id: p for p in run_panos}
    judged = judged_panos(verdict_panos, bundle_ops, run_by_id)
    gt_by_pano, warnings = gt_points_by_pano(
        verdict_panos, bundle_ops, run_by_id, params, frame)
    strong = strong_sites(sites, min_panos)
    cands, excluded_subfloor = mine_candidates(
        strong, judged, run_by_id, gt_by_pano, frame, params, max(radii_m), match_m)
    yields, n_ops = yield_all_panos(strong, run_panos, frame, radii_m)
    headline = precision_row('headline', cands)
    return {
        'params': {'radii_m': list(radii_m), 'min_panos': min_panos,
                   'match_m': match_m, 'camera_height_m': params.camera_height_m},
        'fuse': {'n_panos': fuse_stats['n_panos'], 'n_sites': fuse_stats['n_sites']},
        'n_strong_sites': len(strong),
        'n_judged_panos': len(judged),
        'n_attested_panos': sum(1 for e in judged.values() if _in_pool(e)),
        'excluded_subfloor_members': excluded_subfloor,
        'headline': headline,
        'reading': rule_reading(headline['precision']),
        'strata': strata(cands, radii_m),
        'yield': {'counts': yields, 'n_operational_detections': n_ops},
        'warnings': warnings,
    }, cands


def _fmt(row):
    p = 'n/a' if row['precision'] is None else f"{row['precision']:.3f}"
    ci = f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]" if row['tp'] + row['fp'] else ''
    return (f"| {row['stratum']} | {row['candidates']} | {row['tp']} | {row['fp']} | "
            f"{p} {ci} | {row['already_detected']} | {row['unsure']} | "
            f"{row['false_det_nearby']} | {row['unadjudicable']} |")


TABLE_HEAD = ('| stratum | cand. | tp | fp | precision [95% CI] | already det. | '
              'unsure | false det. nearby | unadjudicable |\n'
              '|---|--:|--:|--:|---|--:|--:|--:|--:|')


def format_report(city, r):
    h = r['headline']
    headline_p = 'n/a' if h['precision'] is None else f"{h['precision']:.3f}"
    lines = [
        f"## {city}: precision of mined hard positives (RampNet#158 step 1)",
        '',
        f"run: {r['fuse']['n_panos']} panos -> {r['fuse']['n_sites']} sites, "
        f"{r['n_strong_sites']} with >= {r['params']['min_panos']} operational panos",
        f"GT: {r['n_judged_panos']} fully judged panos, {r['n_attested_panos']} with "
        f"the missed-ramp check attested; match radius {r['params']['match_m']:g} m; "
        f"candidates within {max(r['params']['radii_m']):g} m; camera height "
        f"{r['params']['camera_height_m']:g} m",
        f"excluded: {r['excluded_subfloor_members']} (site, pano) pairs where the pano "
        f"is a member through a sub-threshold detection only",
        '',
        f"**headline precision {headline_p} [{h['ci_lo']:.3f}, {h['ci_hi']:.3f}] "
        f"(tp {h['tp']} / fp {h['fp']}) -> {r['reading']}**",
        '',
        TABLE_HEAD, _fmt(h),
    ]
    for key, title in (('radius', 'cumulative by camera-to-site distance'),
                       ('range', 'by range bucket'),
                       ('support', 'by site support (operational panos)'),
                       ('confidence', 'by best member confidence')):
        lines += ['', f'### {title}', '', TABLE_HEAD]
        lines += [_fmt(row) for row in r['strata'][key]]
    y = r['yield']
    lines += ['', '### mined yield over the whole run (#102 sanity check)', '',
              '| radius | (site, non-member pano) pairs | x operational detections |',
              '|---|--:|--:|']
    for radius, n in sorted(y['counts'].items()):
        ratio = n / y['n_operational_detections'] if y['n_operational_detections'] else 0
        lines.append(f'| <= {radius:g} m | {n} | {ratio:.2f}x |')
    if r['warnings']:
        lines += ['', f"{len(r['warnings'])} GT panos skipped (see eval_sites):"]
        lines += [f'- {w}' for w in r['warnings'][:10]]
    return '\n'.join(lines)


def write_outputs(out_dir, report_text, cands):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.md').write_text(report_text + '\n', encoding='utf-8')
    with open(out_dir / 'candidates.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(Candidate.__dataclass_fields__))
        w.writeheader()
        for c in cands:
            row = asdict(c)
            row['nearest_gt_m'] = '' if math.isinf(c.nearest_gt_m) else f'{c.nearest_gt_m:.2f}'
            for k in ('range_m', 'bearing_deg'):
                row[k] = f'{row[k]:.2f}'
            for k in ('x_norm', 'y_norm', 'best_conf'):
                row[k] = f'{row[k]:.5f}'
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', help='benchmark split name, e.g. richmond')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--radius', type=float, nargs='+', default=[10.0, 15.0],
                    help='camera-to-site distances to report; candidates are '
                         'collected out to the largest')
    ap.add_argument('--min-panos', type=int, default=3,
                    help='operational panos a site needs to count as a ramp')
    ap.add_argument('--match-m', type=float, default=5.0,
                    help='world-space radius within which a GT point adjudicates '
                         'a candidate (the eval\'s match radius)')
    ap.add_argument('--camera-height', type=float, default=None,
                    help='override geo.DEFAULT_CAMERA_HEIGHT_M for fusion, GT '
                         'placement and the projection alike (sensitivity check '
                         'for the #101 range anchoring; the measured medians are '
                         '~2.2 m GSV, ~1.7 m for the richmond Mapillary rig)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/mined_precision)')
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    verdict_panos, bundle_ops, run_panos = es.load_city_files(
        args.city, args.benchmark_root, run_dir)
    params = fs.FuseParams() if args.camera_height is None \
        else fs.FuseParams(camera_height_m=args.camera_height)
    result, cands = run_city(verdict_panos, bundle_ops, run_panos, params,
                             radii_m=tuple(args.radius), min_panos=args.min_panos,
                             match_m=args.match_m)
    report_text = format_report(args.city, result)
    print(report_text)
    out_dir = args.out or run_dir / 'mined_precision'
    write_outputs(out_dir, report_text, cands)
    print(f'\nwrote {out_dir}')


if __name__ == '__main__':
    main()
