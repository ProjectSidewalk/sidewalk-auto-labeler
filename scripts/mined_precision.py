"""Precision of hard positives mined from multi-view consensus (RampNet#158 step 1).

RampNet#102 proposes a training-label source that needs no inventory and no human
review: a fused site corroborated by >= 3 operational panos is almost certainly a
ramp, so every nearby pano that is not already one of its members is a training
target at a known world position, once projected back into that pano. Before
anyone builds that miner, #102 asks for the precision of what it would produce,
measured against verdicts that already exist. This script is that measurement.

For one benchmark city it re-fuses the run in memory (fuse_sites.fuse, same code
path as production and eval_sites), takes every fully judged benchmark pano that
stood within R of a strong site without being one of its members, projects the
site into that pano (geo.ground_point_to_pano), and asks the reviewer's ground
truth what is there, in world space with the eval's own match radius:

    tp                the reviewer marked a MISSED ramp there: a true mined target
    fp                the pano was attested clean there: occlusion, ghost, out of view
    already_detected  a verdict-true detection is there: the label is CORRECT, but
                      the model already produces it, so it is not a hard positive
    unsure            an unsure missed mark is there
    false_det_nearby  a verdict-false detection is there (the reviewer looked and
                      said no) - counted as a false positive, see FP_BUCKETS
    unadjudicable     nothing within the match radius, and the pano's missed-ramp
                      check was never attested, so silence there means nothing

TWO DENOMINATORS, both reported side by side, because the choice changes the
reading of the pre-registered rule and the mining rule the code models (a pano is
a candidate iff it is not a member of the site) cannot tell them apart:

    hard-only   tp / (tp + fp)                    - of the targets the miner emits,
                                                    how many are misses the model
                                                    does not already make
    all-mined   (tp + already_detected) / (... )  - of the labels the miner emits,
                                                    how many are CORRECT

A miner has no verdicts at mining time, so it cannot filter `already_detected`
out: those labels ship. `all-mined` is therefore the precision of the training
data; `hard-only` is the rate at which that data is *new*. Wilson intervals on
both, stratified by range, by site support and by site confidence.

Reads the same files as eval_sites.py (benchmark/<city>/{verdicts.json,
records.jsonl} as data, runs/<city>/results.jsonl) and never touches the network.
Writes runs/<city>/mined_precision/{report.md,candidates.csv}; the CSV has one row
per (site, pano) candidate, with the nearest GT point and its distance always
recorded (`within_match` says whether it was close enough to adjudicate), so any
bucket can be eyeballed and the localization hypothesis can be tested directly.

Usage:
    python scripts/mined_precision.py richmond
    python scripts/mined_precision.py paterson --radius 10 15 20 --min-panos 3
    # several cities also write the POOLED headline, which is what the RampNet#158
    # decision reads; --camera-height takes one value or one per city, in order
    python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo
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
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

BUCKETS = ('tp', 'fp', 'already_detected', 'unsure', 'false_det_nearby',
           'unadjudicable')
# What counts as a false positive under BOTH denominators. `false_det_nearby` is in
# here deliberately: the reviewer looked at that spot, in that pano, and rejected a
# detection of it, which is evidence AGAINST the site, not neutral. Leaving it out
# would make the same attested-clean pano drop out of the denominator purely because
# a rejected detection happened to sit within the match radius. It keeps its own
# bucket (and CSV row) so the two can still be counted separately.
FP_BUCKETS = ('fp', 'false_det_nearby')
# Numerators: `hard-only` counts new misses, `all-mined` counts correct labels.
TP_HARD = ('tp',)
TP_ALL = ('tp', 'already_detected')
RANGE_EDGES = (8.0, 12.0, 18.0, 25.0)
CONF_SPLIT = 0.9
# The pre-registered decision rule (RampNet#158 step 1).
RULE_BUILD, RULE_VISIBILITY = 0.80, 0.50
BANDS = (('build', RULE_BUILD, 'build the miner'),
         ('visibility', RULE_VISIBILITY, 'add the visibility test before mining'),
         ('drop', 0.0, 'drop this label source'))


@dataclass
class Candidate:
    # `city` is first and is part of the row key. `site_id` is fuse_sites' per-run
    # serial (Site(len(sites), det)), so it is NOT unique across cities - the same
    # trap as a Project Sidewalk label_id, which identifies a label only together
    # with its city. Anything that pools cities keys on (city, site_id), and
    # pool_cities checks (city, site_id, pano_id) for uniqueness before counting.
    city: str
    site_id: int
    pano_id: str
    range_m: float          # camera-to-site distance == projected range (flat ground)
    bearing_deg: float
    x_norm: float
    y_norm: float
    n_op_panos: int
    best_conf: float
    bucket: str
    nearest_gt_kind: str    # nearest GT point in THIS pano, at any distance ('' = none)
    nearest_gt_m: float     # its distance to the site (inf when the pano has no GT point)
    within_match: bool      # was it close enough to adjudicate (<= match_m)?


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
               if c >= BENCHMARK_CONFIDENCE]
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
               if c >= BENCHMARK_CONFIDENCE]
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
                    max_radius_m, match_m, city):
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
                max_range_m=max_radius_m, apply_pose=params.apply_pose)
            if proj is None:
                continue
            # The nearest GT point in this pano is recorded whatever its distance:
            # for an fp, "how far away was the nearest missed mark" is the whole
            # localization diagnostic, and blanking it past match_m threw it away.
            kind, dist = '', math.inf
            for k, e, n in gt_by_pano.get(pid, ()):
                d = math.hypot(e - site.e, n - site.n)
                if d < dist:
                    kind, dist = k, d
            within = dist <= match_m
            adjudicating = kind if within else ''
            if adjudicating == 'missed':
                bucket = 'tp'
            elif adjudicating == 'det':
                bucket = 'already_detected'
            elif adjudicating == 'unsure':
                bucket = 'unsure'
            elif adjudicating == 'false_det':
                # NOT gated on _in_pool: a rejected detection adjudicates that spot
                # in that pano by itself, so whether the pano's separate missed-ramp
                # sweep was attested has no bearing on it. Gating it here would drop
                # exactly the rejected-detection candidates out of the denominator
                # in any bundle where the sweep was skipped, reading precision high
                # - the inversion counting them as fp exists to prevent.
                bucket = 'false_det_nearby'
            else:
                bucket = 'fp' if _in_pool(judged[pid]) else 'unadjudicable'
            cands.append(Candidate(city, site.id, pid,
                                   proj.range_m, proj.bearing_deg,
                                   proj.x_norm, proj.y_norm, n_op_panos,
                                   best_conf, bucket, kind, dist, within))
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
                if c >= BENCHMARK_CONFIDENCE)
    return counts, n_ops


def tally(cands):
    t = {b: 0 for b in BUCKETS}
    for c in cands:
        t[c.bucket] += 1
    return t


def _ratio(k, n):
    """(point estimate or None, wilson lo, wilson hi)."""
    lo, hi = es.wilson(k, n)
    return (k / n if n else None), lo, hi


def precision_row(label, cands):
    """One stratum under BOTH denominators. `fp` is the total false-positive count
    (`fp` + `false_det_nearby`); `fp_rejected_det` breaks the second out."""
    t = tally(cands)
    fp = sum(t[b] for b in FP_BUCKETS)
    hard_k = sum(t[b] for b in TP_HARD)
    all_k = sum(t[b] for b in TP_ALL)
    p_hard, hard_lo, hard_hi = _ratio(hard_k, hard_k + fp)
    p_all, all_lo, all_hi = _ratio(all_k, all_k + fp)
    return {'stratum': label, 'candidates': len(cands), 'tp': t['tp'], 'fp': fp,
            'fp_rejected_det': t['false_det_nearby'],
            'already_detected': t['already_detected'], 'unsure': t['unsure'],
            'unadjudicable': t['unadjudicable'],
            'n_hard': hard_k + fp, 'p_hard': p_hard,
            'hard_lo': hard_lo, 'hard_hi': hard_hi,
            'n_all': all_k + fp, 'p_all': p_all,
            'all_lo': all_lo, 'all_hi': all_hi}


def strata(cands, radii_m):
    rows = {'radius': [], 'range': [], 'support': [], 'confidence': []}
    for r in sorted(radii_m):
        rows['radius'].append(precision_row(f'<= {r:g} m',
                                            [c for c in cands if c.range_m <= r]))
    lo, covered = 0.0, 0
    for hi in RANGE_EDGES:
        if lo >= max(radii_m):
            break
        band = [c for c in cands if lo < c.range_m <= hi]
        rows['range'].append(precision_row(f'{lo:g}-{hi:g} m', band))
        covered += len(band)
        lo = hi
    # ...and never silently drop the tail: an open-ended bucket catches anything past
    # the last edge (reachable when a caller raises FuseParams.max_range_m).
    tail = [c for c in cands if c.range_m > lo]
    if tail:
        rows['range'].append(precision_row(f'> {lo:g} m', tail))
        covered += len(tail)
    if covered != len(cands):   # not an assert: python -O would strip it
        raise AssertionError(
            f'range buckets accounted for {covered} of {len(cands)} candidates')
    for label, test in (('3 panos', lambda c: c.n_op_panos == 3),
                        ('4 panos', lambda c: c.n_op_panos == 4),
                        ('>= 5 panos', lambda c: c.n_op_panos >= 5)):
        rows['support'].append(precision_row(label, [c for c in cands if test(c)]))
    rows['confidence'].append(precision_row(
        f'best member >= {CONF_SPLIT}', [c for c in cands if c.best_conf >= CONF_SPLIT]))
    rows['confidence'].append(precision_row(
        f'best member < {CONF_SPLIT}', [c for c in cands if c.best_conf < CONF_SPLIT]))
    return rows


def _band(p):
    for name, floor, _text in BANDS:
        if p >= floor:
            return name
    return 'drop'


def rule_reading(p, lo, hi, radius_m):
    """The pre-registered rule as read off a point estimate AND its interval.

    Reading the point estimate alone hid that every city's 95% CI straddles at
    least one band edge, so the bolded sentence over-stated the decision. The
    radius is named because it is chosen (max of --radius) and moves the answer.
    """
    if p is None:
        return 'no adjudicable candidates'
    band = _band(p)
    text = next(t for name, _f, t in BANDS if name == band)
    out = f'read at <= {radius_m:g} m: {text}'
    lo_band, hi_band = _band(lo), _band(hi)
    if lo_band != hi_band:
        order = [n for n, _f, _t in reversed(BANDS)]   # drop -> visibility -> build
        spanned = [n for n in order
                   if order.index(lo_band) <= order.index(n) <= order.index(hi_band)]
        out += (f'; the 95% CI [{lo:.2f}, {hi:.2f}] spans the '
                + '/'.join(spanned) + ' bands, so this is not decisive')
    return out


def run_city(verdict_panos, bundle_ops, run_panos, params, radii_m=(10.0, 15.0),
             min_panos=3, match_m=5.0, city=''):
    """The whole check for one city; returns (result dict, candidates). No I/O."""
    # A candidate past the ground-raycast range can never be adjudicated: the GT
    # marks that would answer for it are placed through the same raycast and dropped
    # beyond params.max_range_m, so everything out there falls to `fp` by default and
    # the precision reads far lower than it is. Refuse rather than mislead.
    if max(radii_m) > params.max_range_m + 1e-9:
        raise ValueError(
            f'radius {max(radii_m):g} m exceeds the {params.max_range_m:g} m ground-'
            f'raycast range, beyond which no GT mark can be placed; candidates out '
            f'there could only ever be counted false. Lower --radius; raising '
            f'the range for both is a FuseParams.max_range_m code change, not '
            f'an option.')
    sites, frame, fuse_stats = fs.fuse(run_panos, params)
    run_by_id = {p.pano_id: p for p in run_panos}
    judged = judged_panos(verdict_panos, bundle_ops, run_by_id)
    gt_by_pano, warnings = gt_points_by_pano(
        verdict_panos, bundle_ops, run_by_id, params, frame)
    strong = strong_sites(sites, min_panos)
    cands, excluded_subfloor = mine_candidates(
        strong, judged, run_by_id, gt_by_pano, frame, params, max(radii_m),
        match_m, city)
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
        'reading': {
            'hard': rule_reading(headline['p_hard'], headline['hard_lo'],
                                 headline['hard_hi'], max(radii_m)),
            'all': rule_reading(headline['p_all'], headline['all_lo'],
                                headline['all_hi'], max(radii_m)),
        },
        'strata': strata(cands, radii_m),
        'yield': {'counts': yields, 'n_operational_detections': n_ops},
        'warnings': warnings,
    }, cands


def pool_cities(per_city, radii_m, min_panos, match_m):
    """One result dict over several cities' (result, candidates) pairs.

    The decision numbers on RampNet#158 are pooled, so pooling lives here rather
    than in whatever ad-hoc script last needed it. Candidates are keyed on
    (city, site_id, pano_id): `site_id` alone is a per-run serial and collides
    across cities, so concatenating two cities' candidates.csv and grouping on it
    would cross-count different ramps.
    """
    all_cands = [c for _r, cands in per_city for c in cands]
    keys = {(c.city, c.site_id, c.pano_id) for c in all_cands}
    if len(keys) != len(all_cands):
        raise AssertionError(
            f'{len(all_cands) - len(keys)} duplicate (city, site_id, pano_id) rows; '
            f'were two runs of the same city pooled?')
    all_cands.sort(key=lambda c: (c.range_m, c.city, c.site_id, c.pano_id))
    results = [r for r, _c in per_city]
    heights = sorted({r['params']['camera_height_m'] for r in results})
    counts = {r_m: sum(res['yield']['counts'][r_m] for res in results)
              for r_m in radii_m}
    headline = precision_row('headline', all_cands)
    return {
        'params': {'radii_m': list(radii_m), 'min_panos': min_panos,
                   'match_m': match_m,
                   'camera_height_m': '/'.join(f'{h:g}' for h in heights)},
        'fuse': {'n_panos': sum(r['fuse']['n_panos'] for r in results),
                 'n_sites': sum(r['fuse']['n_sites'] for r in results)},
        'n_strong_sites': sum(r['n_strong_sites'] for r in results),
        'n_judged_panos': sum(r['n_judged_panos'] for r in results),
        'n_attested_panos': sum(r['n_attested_panos'] for r in results),
        'excluded_subfloor_members': sum(r['excluded_subfloor_members']
                                         for r in results),
        'headline': headline,
        'reading': {
            'hard': rule_reading(headline['p_hard'], headline['hard_lo'],
                                 headline['hard_hi'], max(radii_m)),
            'all': rule_reading(headline['p_all'], headline['all_lo'],
                                headline['all_hi'], max(radii_m)),
        },
        'strata': strata(all_cands, radii_m),
        'yield': {'counts': counts,
                  'n_operational_detections': sum(
                      r['yield']['n_operational_detections'] for r in results)},
        'warnings': [w for r in results for w in r['warnings']],
    }, all_cands


def _num(v):
    return v if isinstance(v, str) else f'{v:g}'


def _pct(row, key):
    p, lo, hi, n = row[f'p_{key}'], row[f'{key}_lo'], row[f'{key}_hi'], row[f'n_{key}']
    if p is None:
        return 'n/a'
    return f'{p:.3f} [{lo:.2f}, {hi:.2f}] (n={n})'


def _fmt(row):
    return (f"| {row['stratum']} | {row['candidates']} | {row['tp']} | {row['fp']} | "
            f"{row['fp_rejected_det']} | {row['already_detected']} | "
            f"{row['unsure']} | {row['unadjudicable']} | "
            f"{_pct(row, 'hard')} | {_pct(row, 'all')} |")


TABLE_HEAD = ('| stratum | cand. | tp | fp | of which rej. det. | already det. | '
              'unsure | unadj. | precision hard-only [95% CI] | '
              'precision all-mined [95% CI] |\n'
              '|---|--:|--:|--:|--:|--:|--:|--:|---|---|')


def format_report(city, r):
    h = r['headline']
    lines = [
        f"## {city}: precision of mined positives (RampNet#158 step 1)",
        '',
        f"run: {r['fuse']['n_panos']} panos -> {r['fuse']['n_sites']} sites, "
        f"{r['n_strong_sites']} with >= {r['params']['min_panos']} operational panos",
        f"GT: {r['n_judged_panos']} fully judged panos, {r['n_attested_panos']} with "
        f"the missed-ramp check attested; match radius {r['params']['match_m']:g} m; "
        f"candidates within {max(r['params']['radii_m']):g} m; camera height "
        f"{_num(r['params']['camera_height_m'])} m",
        f"excluded: {r['excluded_subfloor_members']} (site, pano) pairs where the pano "
        f"is a member through a sub-threshold detection only",
        '',
        'Two denominators (see the module docstring). A miner has no verdicts, so it '
        'cannot filter `already_detected` out; those labels ship and they are correct.',
        '',
        f"- **hard-only** (new misses): {_pct(h, 'hard')} -> {r['reading']['hard']}",
        f"- **all-mined** (correct labels): {_pct(h, 'all')} -> {r['reading']['all']}",
        '',
        TABLE_HEAD, _fmt(h),
    ]
    for key, title in (('radius', 'cumulative by camera-to-site distance'),
                       ('range', 'by range bucket'),
                       ('support', 'by site support (operational panos)'),
                       ('confidence', 'by best member confidence')):
        lines += ['', f'### {title}', '', TABLE_HEAD]
        lines += [_fmt(row) for row in r['strata'][key]]
        if key == 'confidence':
            lines += ['', f'Read the confidence split with care: `best_conf` saturates '
                          f'(values run above 1.0), so the >= {CONF_SPLIT} bin holds '
                          f'most candidates, and it is confounded with support and '
                          f'range - stronger sites are seen by more panos, hence from '
                          f'further away, where precision falls for geometric reasons. '
                          f'It is not evidence that confident sites mine worse.']
    y = r['yield']
    lines += ['', '### mined yield over the whole run (#102 sanity check)', '',
              'An upper bound: these are (strong site, non-member pano) pairs, and the '
              '`already_detected` column above shows a share of them are ramps the '
              'model already detects from that pano rather than misses.', '',
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
            row['nearest_gt_m'] = '' if math.isinf(c.nearest_gt_m) \
                else f'{c.nearest_gt_m:.2f}'
            row['within_match'] = '1' if c.within_match else '0'
            for k in ('range_m', 'bearing_deg'):
                row[k] = f'{row[k]:.2f}'
            for k in ('x_norm', 'y_norm', 'best_conf'):
                row[k] = f'{row[k]:.5f}'
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', nargs='+',
                    help='benchmark split name(s), e.g. richmond. Give several to '
                         'also print and write the POOLED headline - the decision '
                         'numbers on RampNet#158 are pooled, so they have to be '
                         'regenerable in one command')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs',
                    help='where runs/<city>/results.jsonl live (default runs/)')
    ap.add_argument('--run-dir', type=Path, default=None,
                    help='one run directory, overriding --runs-root; single city only')
    ap.add_argument('--radius', type=float, nargs='+', default=[10.0, 15.0],
                    help='camera-to-site distances to report; candidates are '
                         'collected out to the largest, which may not exceed the '
                         'ground-raycast range (25 m) - past it no GT mark can be '
                         'placed, so a candidate could only ever be counted false')
    ap.add_argument('--min-panos', type=int, default=3,
                    help='operational panos a site needs to count as a ramp')
    ap.add_argument('--match-radius-m', '--match-m', type=float, default=5.0,
                    dest='match_radius_m',
                    help='world-space radius within which a GT point adjudicates '
                         'a candidate (the eval\'s match radius; same name as '
                         'eval_sites.py, --match-m still accepted)')
    ap.add_argument('--camera-height', type=float, nargs='+', default=None,
                    help='override geo.DEFAULT_CAMERA_HEIGHT_M for fusion, GT '
                         'placement and the projection alike (the #101 range-'
                         'anchoring sensitivity knob). One value for every city, '
                         'or one per city in the order they are named. GSV: 2.2 m, '
                         'the median measured from GSV depth payloads (#40/#41). '
                         'Mapillary serves no depth, so richmond has NO measured '
                         'height - a sweep there is flat at 0.31-0.32 for 2.0-2.6 m '
                         'and worse below, i.e. no constant fixes it')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/mined_precision, and '
                         'runs/_pooled/mined_precision for the pooled report)')
    args = ap.parse_args()

    cities = args.city
    if args.run_dir is not None and len(cities) > 1:
        ap.error('--run-dir names one run directory; with several cities use '
                 '--runs-root, which resolves <runs-root>/<city> for each.')
    heights = args.camera_height
    if heights is not None and len(heights) not in (1, len(cities)):
        ap.error(f'--camera-height takes one value or one per city '
                 f'({len(cities)} named), got {len(heights)}.')

    per_city, out_dirs = [], []
    for i, city in enumerate(cities):
        run_dir = args.run_dir or args.runs_root / city
        try:
            verdict_panos, bundle_ops, run_panos = es.load_city_files(
                city, args.benchmark_root, run_dir)
        except FileNotFoundError as exc:
            ap.error(f'{exc.filename}: not found. The benchmark lives in the RampNet '
                     f'checkout (default {ap.get_default("benchmark_root")}); point '
                     f'--benchmark-root at it, and --runs-root at the runs, if '
                     f'either sits elsewhere (e.g. in a git worktree).')
        # Fusion at the BENCHMARK threshold (the verdicts' tier), not the production one.
        params = (fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE) if heights is None
                  else fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE,
                                     camera_height_m=heights[i] if len(heights) > 1
                                     else heights[0]))
        try:
            result, cands = run_city(verdict_panos, bundle_ops, run_panos, params,
                                     radii_m=tuple(args.radius),
                                     min_panos=args.min_panos,
                                     match_m=args.match_radius_m, city=city)
        except ValueError as exc:
            ap.error(str(exc))
        report_text = format_report(city, result)
        print(report_text)
        out_dir = (args.out / city if args.out and len(cities) > 1
                   else args.out or run_dir / 'mined_precision')
        write_outputs(out_dir, report_text, cands)
        out_dirs.append(out_dir)
        per_city.append((result, cands))

    if len(cities) > 1:
        pooled, pooled_cands = pool_cities(
            per_city, tuple(args.radius), args.min_panos, args.match_radius_m)
        text = format_report('pooled over ' + ', '.join(cities), pooled)
        print('\n\n' + text)
        pooled_dir = (args.out / '_pooled' if args.out
                      else args.runs_root / '_pooled' / 'mined_precision')
        write_outputs(pooled_dir, text, pooled_cands)
        out_dirs.append(pooled_dir)
    print('\nwrote ' + ', '.join(str(d) for d in out_dirs))


if __name__ == '__main__':
    main()
