"""POST HOC: which ramp did an image-placement "fix" land on? (RampNet#158 phase 2 review)

NOT part of the pre-registered phase-2 design. The phase-2 plan (RampNet#158, comment
5894832666) fixed the source rule, the arms and the adjudication: a placed target is
right when the nearest GT point in the target pano lies within the 5 m match radius and
is a missed mark (`tp`) or a verdict-true detection (`already_detected`). The all-mined
numbers stand under that rubric - a label on any real ramp is a correct label. This
script was written AFTER scoring, in response to the independent review of
sidewalk-auto-labeler#110 / RampNet#219, to answer a different question the rubric
cannot: when a placement arm moves a candidate into `already_detected`, is the target
detection it lands on the SAME physical ramp as the mined site, or a neighbouring one
the target pano also detected?

The only identity we have for "which ramp" is the fuse itself. For every candidate
whose bucket changes between the flat run and a placement arm, this re-runs the exact
step-2 fuse and adjudication (mined_precision.run_city's code path, same FuseParams as
mined_precision.main), finds the GT point that adjudicated each side, and - when it is
one of the target pano's own detections - looks that detection up in the fused sites:

    own_site          the detection is a member of the candidate's own site. This is
                      IMPOSSIBLE by construction: a pano is a candidate only if it has no
                      member in the site at all. It is counted (and must be 0) so the
                      table says so rather than leaving the category out.
    other_multi       a member of a DIFFERENT site that >= 2 panos corroborate - per the
                      fuse, a distinct ramp. `src_in_landed` marks the sharpest case: the
                      candidate's own source pano (the view the arm transferred FROM) is
                      also a member of the landed site, i.e. the source view itself shows
                      the two ramps separately.
    other_singleton   a site of that one detection: the fuse linked it to nothing, so
                      whether it is the candidate's ramp (an association miss) or another
                      one is undetermined here. Its distance to the candidate site is
                      recorded.
    unfused           a detection not in any site (should not happen at the benchmark
                      threshold; counted, not hidden).
    none              the adjudicating GT point is not a detection (a missed or unsure
                      mark), or nothing is within the match radius.

`fixed` (flat wrong -> arm right) and `tp_to_ad` are attributed on the ARM side (where
the placed point landed); `broken` (flat right -> arm wrong) on the FLAT side (what the
flat point had matched). Every recomputed bucket is checked against the committed
candidates.csv of the same run (--verify), so the attribution describes exactly the
candidates the tables count.

Usage (frozen inputs as in docs/mined-precision.md; RampNet benchmark at 4a859f1):
    D=docs/figures/mined-precision/data/frozen
    python scripts/mined_placement_attribution.py richmond paterson bend gainesville sao_paulo \\
        --runs-root $FROZEN --benchmark-root ../RampNet/benchmark \\
        --camera-height per-rig per-pano per-pano per-pano per-pano \\
        --arm roma=docs/figures/mined-precision/data/placement/roma.jsonl \\
        --arm roma_local=docs/figures/mined-precision/data/placement/roma_local.jsonl \\
        --arm mapa_posed_pair=docs/figures/mined-precision/data/placement/mapa_posed_pair.jsonl \\
        --verify $D --mapillary richmond \\
        --out $D/attribution_phase2.csv --md $D/attribution_phase2.md
"""
import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import mined_precision as mp  # noqa: E402
import mined_precision_compare as mpc  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

CATEGORIES = ('own_site', 'other_multi', 'other_singleton', 'unfused', 'none')
TRANSITIONS = ('fixed', 'broken', 'tp_to_ad', 'ad_to_tp', 'other_change')
FIELDS = ('city', 'site_id', 'pano_id', 'arm', 'range_m', 'flat_bucket', 'arm_bucket',
          'transition', 'side', 'category', 'landed_site_id', 'landed_site_panos',
          'site_to_landed_m', 'src_pano', 'src_in_landed', 'point_to_det_m',
          'x_to_det_deg')


def _state(bucket):
    """mined_precision_compare._state on a bare bucket: right / wrong / out."""
    return ('right' if bucket in mp.TP_ALL else
            'wrong' if bucket in mp.FP_BUCKETS else 'out')


def transition(flat_bucket, arm_bucket):
    """Name a bucket change between the flat run and an arm (None when unchanged).
    fixed / broken use compare's right / wrong states, so they are the sign test's."""
    if flat_bucket == arm_bucket:
        return None
    a, b = _state(flat_bucket), _state(arm_bucket)
    if a == 'wrong' and b == 'right':
        return 'fixed'
    if a == 'right' and b == 'wrong':
        return 'broken'
    if flat_bucket == 'tp' and arm_bucket == 'already_detected':
        return 'tp_to_ad'
    if flat_bucket == 'already_detected' and arm_bucket == 'tp':
        return 'ad_to_tp'
    return 'other_change'


def target_detections(entry, run_pano, params, frame):
    """[(stored det_index, verdict, x, e, n)] for the target pano's OPERATIONAL
    detections, raycast exactly as gt_points_by_pano / build_gt place them."""
    pose = fs.pano_pose(run_pano, params.apply_pose)
    ops = [(i, x, y) for i, x, y, c in run_pano.detections if c >= BENCHMARK_CONFIDENCE]
    out = []
    for verdict, (i, x, y) in zip(entry['dets'], ops):
        enu = mp.raycast_pixel(pose, (x, y), run_pano, params, frame)
        if enu is not None:
            out.append((i, verdict, x, enu[0], enu[1]))
    return out


def attribute(point, kind, dist, match_m, dets, det_site, cand_site, src_pano, x_px):
    """Category and details for the GT point that adjudicated `point` (ENU)."""
    row = {'category': 'none', 'landed_site_id': '', 'landed_site_panos': '',
           'site_to_landed_m': '', 'src_in_landed': '', 'point_to_det_m': '',
           'x_to_det_deg': ''}
    if dist > match_m or kind not in ('det', 'false_det'):
        return row
    want = kind == 'det'
    best = None
    for i, verdict, x, e, n in dets:
        if (verdict is True) != want:
            continue
        d = math.hypot(e - point[0], n - point[1])
        if best is None or d < best[0]:
            best = (d, i, x)
    if best is None or abs(best[0] - dist) > 1e-6:
        raise AssertionError(f'could not re-identify the adjudicating {kind} '
                             f'(nearest {best}, classify said {dist})')
    d, i, x = best
    row['point_to_det_m'] = d
    row['x_to_det_deg'] = abs(geo.norm_deg((x_px - x) * 360.0))
    site = det_site.get(i)
    if site is None:
        row['category'] = 'unfused'
        return row
    row['landed_site_id'] = site.id
    row['landed_site_panos'] = len(site.pano_ids)
    row['site_to_landed_m'] = math.hypot(site.e - cand_site.e, site.n - cand_site.n)
    row['src_in_landed'] = int(src_pano in site.pano_ids) if src_pano else ''
    if site.id == cand_site.id:
        row['category'] = 'own_site'
    elif len(site.pano_ids) >= 2:
        row['category'] = 'other_multi'
    else:
        row['category'] = 'other_singleton'
    return row


def run_city(city, verdict_panos, bundle_ops, run_panos, params, arms, radius_m,
             match_m, verify_root=None):
    """Attribution rows for every changed candidate of every arm in one city."""
    placements = {label: {k: v for k, v in pl.items() if k[0] == city}
                  for label, pl in arms}
    sites, frame, _ = fs.fuse(run_panos, params)
    by_site = {s.id: s for s in sites}
    run_by_id = {p.pano_id: p for p in run_panos}
    judged = mp.judged_panos(verdict_panos, bundle_ops, run_by_id)
    gt_by_pano, _ = mp.gt_points_by_pano(verdict_panos, bundle_ops, run_by_id, params,
                                         frame)
    strong = mp.strong_sites(sites, 3)
    sources = []
    flat, _ = mp.mine_candidates(strong, judged, run_by_id, gt_by_pano, frame, params,
                                 radius_m, match_m, city, sources=sources)
    src = {(s['city'], s['site_id'], s['pano_id']): s for s in sources}
    runs = {}
    for label, pl in placements.items():
        placed = []
        cands, _ = mp.mine_candidates(strong, judged, run_by_id, gt_by_pano, frame,
                                      params, radius_m, match_m, city, placement=pl,
                                      placed=placed)
        runs[label] = (cands, {(r['city'], r['site_id'], r['pano_id']): r
                               for r in placed})
    if verify_root is not None:
        _verify(Path(verify_root) / 'perrig_perpano' / city / 'candidates.csv', flat)
        for label, (cands, _p) in runs.items():
            _verify(Path(verify_root) / f'place_{label}' / city / 'candidates.csv', cands)

    det_sites = {}
    for s in sites:
        for d, _ in s.members:
            det_sites[(d.pano_id, d.det_index)] = s
    flat_by = {(c.city, c.site_id, c.pano_id): c for c in flat}

    def landed(c, point, kind, dist, x_px):
        dets = target_detections(judged[c.pano_id], run_by_id[c.pano_id], params, frame)
        det_site = {i: det_sites.get((c.pano_id, i)) for i, *_ in dets}
        s = src.get((c.city, c.site_id, c.pano_id))
        return attribute(point, kind, dist, match_m, dets, det_site, by_site[c.site_id],
                         s['src_pano'] if s else '', x_px)

    # every candidate's adjudicating point, flat and per arm, for the strict re-tally
    every = []
    for c in flat:
        site = by_site[c.site_id]
        every.append({'city': city, 'arm': 'flat', 'bucket': c.bucket, 'category': landed(
            c, (site.e, site.n), c.nearest_gt_kind, c.nearest_gt_m, c.x_norm)['category']})
    for label, (cands, placed) in runs.items():
        for c in cands:
            p = placed[(c.city, c.site_id, c.pano_id)]
            x_px = p['x'] if p['status'] == 'placed' else c.x_norm
            every.append({'city': city, 'arm': label, 'bucket': c.bucket,
                          'category': landed(c, (p['adj_e'], p['adj_n']),
                                             c.nearest_gt_kind, c.nearest_gt_m,
                                             x_px)['category']})
    out = []
    for label, (cands, placed) in runs.items():
        for c in cands:
            key = (c.city, c.site_id, c.pano_id)
            f = flat_by[key]
            t = transition(f.bucket, c.bucket)
            if t is None:
                continue
            site = by_site[c.site_id]
            s = src.get(key)
            src_pano = s['src_pano'] if s else ''
            if t == 'broken':          # what the flat point had matched
                side, point, kind, dist = ('flat', (site.e, site.n), f.nearest_gt_kind,
                                           f.nearest_gt_m)
                x_px = f.x_norm
            else:                      # where the placed point landed
                p = placed[key]
                side, point, kind, dist = ('arm', (p['adj_e'], p['adj_n']),
                                           c.nearest_gt_kind, c.nearest_gt_m)
                x_px = p['x'] if p['status'] == 'placed' else f.x_norm
            row = landed(c, point, kind, dist, x_px)
            row.update({'city': city, 'site_id': c.site_id, 'pano_id': c.pano_id,
                        'arm': label, 'range_m': c.range_m, 'flat_bucket': f.bucket,
                        'arm_bucket': c.bucket, 'transition': t, 'side': side,
                        'src_pano': src_pano})
            out.append(row)
    return out, every


def _verify(path, cands):
    """The recomputed buckets must be the committed run's, candidate for candidate."""
    committed = {(c.city, c.site_id, c.pano_id): c.bucket
                 for c in mpc.read_candidates(path)}
    got = {(c.city, c.site_id, c.pano_id): c.bucket for c in cands}
    if committed != got:
        diff = sorted(k for k in set(committed) | set(got)
                      if committed.get(k) != got.get(k))
        raise AssertionError(f'{path}: recomputed buckets differ from the committed run '
                             f'for {len(diff)} candidates (e.g. {diff[0]})')


def summarise(rows, cities, mapillary=()):
    """[(group, arm, transition, {category: n}, n_src_in_landed)] in a stable order."""
    groups = [(c, [c]) for c in cities] + [('pooled', list(cities))]
    gsv = [c for c in cities if c not in set(mapillary)]
    if gsv and len(gsv) != len(cities):
        groups.append(('pooled GSV', gsv))
    arms = []
    for r in rows:
        if r['arm'] not in arms:
            arms.append(r['arm'])
    out = []
    for g, cs in groups:
        for a in arms:
            for t in TRANSITIONS:
                sub = [r for r in rows if r['city'] in cs and r['arm'] == a
                       and r['transition'] == t]
                if not sub:
                    continue
                n = {k: sum(1 for r in sub if r['category'] == k) for k in CATEGORIES}
                out.append((g, a, t, n, sum(1 for r in sub if r['src_in_landed'] == 1)))
    return out


def strict_retally(every, cities, mapillary=()):
    """All-mined under the rubric AND with `already_detected` that landed on a detection
    of ANOTHER multi-pano site counted as wrong (a label on a different ramp than the
    mined site, per the fuse). A sensitivity reading, post hoc; hard-only is unchanged by
    construction (it has no `already_detected` in it)."""
    groups = [(c, [c]) for c in cities] + [('pooled', list(cities))]
    gsv = [c for c in cities if c not in set(mapillary)]
    if gsv and len(gsv) != len(cities):
        groups.append(('pooled GSV', gsv))
    arms = []
    for r in every:
        if r['arm'] not in arms:
            arms.append(r['arm'])
    out = []
    for g, cs in groups:
        for a in arms:
            sub = [r for r in every if r['city'] in cs and r['arm'] == a]
            tp = sum(1 for r in sub if r['bucket'] == 'tp')
            ad = sum(1 for r in sub if r['bucket'] == 'already_detected')
            ad_other = sum(1 for r in sub if r['bucket'] == 'already_detected'
                           and r['category'] == 'other_multi')
            fp = sum(1 for r in sub if r['bucket'] in mp.FP_BUCKETS)
            out.append({'group': g, 'arm': a, 'tp': tp, 'ad': ad, 'ad_other': ad_other,
                        'fp': fp, 'all': mp._ratio(tp + ad, tp + ad + fp),
                        'strict': mp._ratio(tp + ad - ad_other, tp + ad + fp)})
    return out


def strict_markdown(rows):
    def cell(r, k, n):
        p, lo, hi = r
        return '-' if p is None else f'{p:.3f} [{lo:.2f}, {hi:.2f}] ({k}/{n})'
    lines = ['POST HOC sensitivity: all-mined with `already_detected` on another '
             'multi-pano site counted wrong.', '',
             '| group | arm | tp | already det. | of which on another multi-pano site '
             '| fp | all-mined (rubric) | all-mined, strict |',
             '|---|---|--:|--:|--:|--:|---|---|']
    for r in rows:
        n = r['tp'] + r['ad'] + r['fp']
        lines.append(f"| {r['group']} | {r['arm']} | {r['tp']} | {r['ad']} | {r['ad_other']} "
                     f"| {r['fp']} | {cell(r['all'], r['tp'] + r['ad'], n)} "
                     f"| {cell(r['strict'], r['tp'] + r['ad'] - r['ad_other'], n)} |")
    return '\n'.join(lines)


def to_markdown(summary):
    lines = ['POST HOC diagnostic (not pre-registered); see the script docstring. '
             '`fixed` and `tp_to_ad` are attributed where the placed point landed, '
             '`broken` where the flat point had matched.', '',
             '| group | arm | transition | n | own site | other multi-pano site '
             '(of which the source pano is a member) | other singleton | unfused | none |',
             '|---|---|---|--:|--:|--:|--:|--:|--:|']
    for g, a, t, n, sil in summary:
        lines.append(f"| {g} | {a} | {t} | {sum(n.values())} | {n['own_site']} "
                     f"| {n['other_multi']} ({sil}) | {n['other_singleton']} "
                     f"| {n['unfused']} | {n['none']} |")
    return '\n'.join(lines)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, FIELDS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.2f}' if isinstance(r[k], float) else r[k])
                        for k in FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', nargs='+')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--camera-height', type=fs.fuse_camera_height_arg, nargs='+',
                    required=True, help='as mined_precision.py; one value or one per city')
    ap.add_argument('--arm', action='append', required=True,
                    help='label=PLACEMENT_JSONL (the file mined_precision --placement read)')
    ap.add_argument('--radius', type=float, default=15.0)
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--verify', type=Path, default=None,
                    help='frozen data dir holding perrig_perpano/ and place_<arm>/: refuse '
                         'unless every recomputed bucket equals the committed one')
    ap.add_argument('--mapillary', nargs='*', default=())
    ap.add_argument('--out', type=Path, default=None, help='per-candidate CSV')
    ap.add_argument('--md', type=Path, default=None, help='summary table (markdown)')
    args = ap.parse_args()
    heights = args.camera_height
    if len(heights) not in (1, len(args.city)):
        ap.error('--camera-height takes one value or one per city')
    arms = []
    for a in args.arm:
        label, path = a.split('=', 1)
        arms.append((label, mp.read_placement(path)))
    rows, every = [], []
    for i, city in enumerate(args.city):
        mode = heights[i] if len(heights) > 1 else heights[0]
        verdict_panos, bundle_ops, run_panos, height, _auto = es.load_city_at_height(
            city, args.benchmark_root, args.runs_root / city, mode, read_heights=True)
        # the same FuseParams mined_precision.main uses
        params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                               apply_pose=fs.POSE_OFF, camera_height_m=height)
        changed, all_c = run_city(city, verdict_panos, bundle_ops, run_panos, params,
                                  arms, args.radius, args.match_radius_m, args.verify)
        rows += changed
        every += all_c
    rows.sort(key=lambda r: (r['arm'], r['city'], r['transition'], r['site_id'],
                             r['pano_id']))
    md = (to_markdown(summarise(rows, args.city, args.mapillary)) + '\n\n'
          + strict_markdown(strict_retally(every, args.city, args.mapillary)))
    print(md)
    if args.out:
        write_csv(args.out, rows)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
