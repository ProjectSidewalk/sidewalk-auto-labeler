"""Side-by-side table of mined_precision.py runs that differ in one setting (RampNet#158 step 2).

mined_precision.py writes one candidates.csv per city per run. Step 2 of RampNet#158
reruns it under different camera-height frames and has to be read *beside* step 1,
per city, per range band, on both denominators. Doing that by hand from five
report.md files per arm is where transcription errors come from, so this reads the
committed candidates.csv files back and recomputes every cell with
mined_precision's own precision_row (same buckets, same Wilson interval) - it adds
no new statistics, only the layout.

Each arm is `label=DIR`, where DIR is a mined_precision.py `--out` directory holding
<city>/candidates.csv. Rows are keyed on (city, site_id, pano_id); site_id is a
per-run serial, so candidates are never joined ACROSS arms (a different height is a
different fuse, with different site ids) - each arm is tallied on its own.

Besides the per-city rows it writes two pooled rows per arm: all cities, and the GSV
cities only (every named city except the Mapillary ones given by --mapillary).

`fp placed` counts false positives whose nearest GT point in that pano is a
verdict-true detection (nearest_gt_kind == 'det'): the pano saw a real ramp there and
the projection put the site > match radius off it. It is the localization signal the
step-1 correction measured (48 of 65), read per arm.

Usage:
    python scripts/mined_precision_compare.py \\
        --arm step1_2.6=docs/figures/mined-precision/data/frozen/h2.6 \\
        --arm step2=docs/figures/mined-precision/data/frozen/perrig_perpano \\
        --cities richmond paterson bend gainesville sao_paulo --mapillary richmond \\
        --out docs/figures/mined-precision/data/frozen/compare.csv
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

import mined_precision as mp  # noqa: E402

# (label, predicate on range_m). The bands are mined_precision's RANGE_EDGES up to the
# 15 m candidate radius, so 12-18 m holds 12-15 m in practice.
STRATA = (('<= 15 m', lambda r: r <= 15.0),
          ('<= 10 m', lambda r: r <= 10.0),
          ('0-8 m', lambda r: 0.0 < r <= 8.0),
          ('8-12 m', lambda r: 8.0 < r <= 12.0),
          ('12-18 m', lambda r: 12.0 < r <= 18.0))


def read_candidates(path):
    """candidates.csv back into mp.Candidate rows (inverse of mp.write_outputs)."""
    out = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.append(mp.Candidate(
                city=row['city'], site_id=int(row['site_id']), pano_id=row['pano_id'],
                range_m=float(row['range_m']), bearing_deg=float(row['bearing_deg']),
                x_norm=float(row['x_norm']), y_norm=float(row['y_norm']),
                n_op_panos=int(row['n_op_panos']), best_conf=float(row['best_conf']),
                bucket=row['bucket'], nearest_gt_kind=row['nearest_gt_kind'],
                nearest_gt_m=(math.inf if row['nearest_gt_m'] == ''
                              else float(row['nearest_gt_m'])),
                within_match=row['within_match'] == '1'))
    keys = [(c.city, c.site_id, c.pano_id) for c in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{path}: (city, site_id, pano_id) is not unique')
    return out


def rows_for(arm, group, cands):
    """One output row per stratum for one (arm, city-or-pool) group."""
    rows = []
    for label, test in STRATA:
        sub = [c for c in cands if test(c.range_m)]
        r = mp.precision_row(label, sub)
        r['fp_placed'] = sum(1 for c in sub
                             if c.bucket in mp.FP_BUCKETS and c.nearest_gt_kind == 'det')
        r['arm'], r['group'] = arm, group
        rows.append(r)
    return rows


def compare(arms, cities, mapillary=()):
    """arms: [(label, dir)]. Returns the flat row list (per city, then pools)."""
    out = []
    for label, d in arms:
        per_city = {c: read_candidates(Path(d) / c / 'candidates.csv') for c in cities}
        for c in cities:
            out += rows_for(label, c, per_city[c])
        out += rows_for(label, 'pooled', [x for c in cities for x in per_city[c]])
        gsv = [c for c in cities if c not in set(mapillary)]
        if gsv and len(gsv) != len(cities):
            out += rows_for(label, 'pooled GSV', [x for c in gsv for x in per_city[c]])
    return out


def _state(c):
    """'right' (tp / already_detected), 'wrong' (a false positive) or 'out' (unsure,
    unadjudicable: in neither denominator)."""
    if c.bucket in mp.TP_ALL:
        return 'right'
    if c.bucket in mp.FP_BUCKETS:
        return 'wrong'
    return 'out'


def sign_test_p(k, n):
    """Two-sided exact binomial (p = 0.5) p-value for k of n discordant pairs."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(base, arm, cities, mapillary=()):
    """Candidate-by-candidate transitions from `base` to `arm` (both (label, dir)), for
    runs made on the SAME candidate set (an image-placement arm against its flat run).
    Per city, pooled and pooled GSV: fixed (wrong -> right), broken (right -> wrong),
    tp gained / lost, and a two-sided sign test on fixed vs broken. Two runs whose
    candidate keys differ are refused: they are not paired."""
    per = {}
    for c in cities:
        a = {(x.city, x.site_id, x.pano_id): x
             for x in read_candidates(Path(base[1]) / c / 'candidates.csv')}
        b = {(x.city, x.site_id, x.pano_id): x
             for x in read_candidates(Path(arm[1]) / c / 'candidates.csv')}
        if set(a) != set(b):
            raise ValueError(f'{c}: {base[0]} and {arm[0]} are not the same candidates')
        # site_id is a per-run serial: two different fuses can share key sets by
        # coincidence. A placement arm keeps the flat run's range_m (it is the site's
        # range, never the placed point's), so any difference means a different fuse.
        moved = [k for k in a if a[k].range_m != b[k].range_m]
        if moved:
            raise ValueError(f'{c}: {base[0]} and {arm[0]} disagree on range_m for '
                             f'{len(moved)} candidates (e.g. {moved[0]}); they are '
                             f'not the same fuse, so not paired')
        per[c] = [(a[k], b[k]) for k in sorted(a)]
    groups = [(c, [c]) for c in cities] + [('pooled', list(cities))]
    gsv = [c for c in cities if c not in set(mapillary)]
    if gsv and len(gsv) != len(cities):
        groups.append(('pooled GSV', gsv))
    out = []
    for g, cs in groups:
        pp = [x for c in cs for x in per[c]]
        fixed = sum(1 for x, y in pp if _state(x) == 'wrong' and _state(y) == 'right')
        broken = sum(1 for x, y in pp if _state(x) == 'right' and _state(y) == 'wrong')
        out.append({'group': g, 'base': base[0], 'arm': arm[0], 'candidates': len(pp),
                    'changed': sum(1 for x, y in pp if x.bucket != y.bucket),
                    'fixed': fixed, 'broken': broken,
                    'sign_p': sign_test_p(fixed, fixed + broken),
                    'tp_gained': sum(1 for x, y in pp if x.bucket != 'tp' and y.bucket == 'tp'),
                    'tp_lost': sum(1 for x, y in pp if x.bucket == 'tp' and y.bucket != 'tp')})
    return out


def paired_markdown(rows):
    lines = ['| group | base -> arm | cand. | buckets changed | fixed (fp -> right) '
             '| broken (right -> fp) | sign-test p | tp gained | tp lost |',
             '|---|---|--:|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        lines.append(f"| {r['group']} | {r['base']} -> {r['arm']} | {r['candidates']} "
                     f"| {r['changed']} | {r['fixed']} | {r['broken']} | {r['sign_p']:.3f} "
                     f"| {r['tp_gained']} | {r['tp_lost']} |")
    return '\n'.join(lines)


def _cell(p, lo, hi, k, n):
    if p is None:
        return '- (n=0)'
    return f'{p:.3f} [{lo:.2f}, {hi:.2f}] ({k}/{n})'


def to_markdown(rows, strata=None):
    strata = strata or [s for s, _ in STRATA]
    lines = ['| group | arm | stratum | cand. | tp | fp | fp placed | already det. | unsure '
             '| hard-only [95% CI] (k/n) | all-mined [95% CI] (k/n) |',
             '|---|---|---|--:|--:|--:|--:|--:|--:|---|---|']
    order = []
    for r in rows:
        if r['group'] not in order:
            order.append(r['group'])
    for g in order:
        for s in strata:
            for r in rows:
                if r['group'] != g or r['stratum'] != s:
                    continue
                k_all = r['tp'] + r['already_detected']
                lines.append(
                    f"| {g} | {r['arm']} | {s} | {r['candidates']} | {r['tp']} | {r['fp']} "
                    f"| {r['fp_placed']} | {r['already_detected']} | {r['unsure']} "
                    f"| {_cell(r['p_hard'], r['hard_lo'], r['hard_hi'], r['tp'], r['n_hard'])} "
                    f"| {_cell(r['p_all'], r['all_lo'], r['all_hi'], k_all, r['n_all'])} |")
    return '\n'.join(lines)


FIELDS = ('group', 'arm', 'stratum', 'candidates', 'tp', 'fp', 'fp_rejected_det',
          'fp_placed', 'already_detected', 'unsure', 'unadjudicable',
          'n_hard', 'p_hard', 'hard_lo', 'hard_hi', 'n_all', 'p_all', 'all_lo', 'all_hi')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, FIELDS, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else
                            '' if v is None else v) for k, v in r.items() if k in FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', action='append', required=True,
                    help='label=DIR, DIR a mined_precision.py --out holding <city>/candidates.csv')
    ap.add_argument('--cities', nargs='+', required=True)
    ap.add_argument('--mapillary', nargs='*', default=(),
                    help='cities left out of the "pooled GSV" row')
    ap.add_argument('--out', type=Path, default=None, help='CSV of every cell')
    ap.add_argument('--md', type=Path, default=None, help='markdown copy of the table')
    ap.add_argument('--paired-base', default=None,
                    help='label of an --arm run on the SAME candidates as --paired-arms; '
                         'each gets a candidate-by-candidate transition table against it '
                         '(image-placement arms against their flat run)')
    ap.add_argument('--paired-arms', nargs='*', default=None,
                    help='arms to pair against --paired-base (default: every other arm)')
    args = ap.parse_args()
    arms = []
    for a in args.arm:
        if '=' not in a:
            ap.error(f'--arm {a!r}: expected label=DIR')
        label, d = a.split('=', 1)
        arms.append((label, d))
    rows = compare(arms, args.cities, args.mapillary)
    md = to_markdown(rows)
    print(md)
    if args.out:
        write_csv(args.out, rows)
    if args.paired_base:
        base = next(a for a in arms if a[0] == args.paired_base)
        prow = []
        for a in arms:
            if a is base or (args.paired_arms is not None and a[0] not in args.paired_arms):
                continue
            prow += paired(base, a, args.cities, args.mapillary)
        pmd = paired_markdown(prow)
        print('\n' + pmd)
        md = md + '\n\n' + pmd
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
