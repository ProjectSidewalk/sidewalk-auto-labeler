"""How far the site-median label point moved each label, against the corner-point build (#169).

Compares two builds of the same corner table, the old corner-point build and a
`--label-point site-median` build: refuses unless both hold exactly the same corner keys with
the same chosen panos, then writes <new>/label_shift.json (shift and range statistics by
stratum and over a pilot order's positions, cases, clamps) and lists the labels whose new
pano-to-label range is under the build's min-view distance. Stdlib only.

    python scripts/corner_label_shift.py --old runs/vancouver/corner_labels169_split/na_discount_sw \
        --new runs/vancouver/corner_labels169_split/na_discount_sw_sitemed \
        --pilot runs/vancouver/corner_labels169_split/na_discount_sw/pilot_residential/pilot.csv
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import corner_labels as cl  # noqa: E402


def read(d):
    return {p['key']: p for p in (json.loads(x) for x in
                                  open(Path(d) / 'proposals.jsonl', encoding='utf-8'))}


def stats(xs):
    if not xs:
        return None
    return {'n': len(xs), 'p10': round(cl.quantile(xs, 0.1), 2),
            'median': round(cl.quantile(xs, 0.5), 2), 'p90': round(cl.quantile(xs, 0.9), 2),
            'max': round(max(xs), 2)}


def summarize(keys, old, new):
    lps = [new[k]['label_point'] for k in keys]
    return {'labels': len(keys),
            'shift_m': stats([lp['shift_m'] for lp in lps]),
            'label_dist_from_centre_m': stats([lp['dist_m'] for lp in lps]),
            'range_old_m': stats([old[k]['range_m'] for k in keys]),
            'range_new_m': stats([new[k]['range_m'] for k in keys]),
            'by_case': dict(sorted(collections.Counter(lp['case'] for lp in lps).items())),
            'share_unit_case': round(sum(lp['case'] == 'unit' for lp in lps) / len(lps), 3),
            'clamped': dict(sorted(collections.Counter(str(lp['clamped']) for lp in lps).items()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--pilot', help='a pilot order CSV: also summarize positions 1-203')
    ap.add_argument('--results', action='append', default=[],
                    help='an exported results file of the new build: also summarize its corners '
                         '(repeatable), and the first 120 of them in pilot order (rate + reserve)')
    a = ap.parse_args(argv)
    old, new = read(a.old), read(a.new)
    if set(old) != set(new):
        raise SystemExit(f'corner sets differ: {len(set(old) - set(new))} only in old, '
                         f'{len(set(new) - set(old))} only in new')
    pano_diff = sorted(k for k in old if old[k]['pano']['pano_id'] != new[k]['pano']['pano_id'])
    if pano_diff:
        raise SystemExit(f'{len(pano_diff)} chosen panos differ, e.g. {pano_diff[:3]}')
    out = {'old': str(a.old), 'new': str(a.new), 'same_corner_keys': True,
           'same_chosen_panos': True, 'n': len(new),
           'all': summarize(sorted(new), old, new),
           'by_stratum': {s: summarize(sorted(k for k in new if new[k]['stratum'] == s), old, new)
                          for s in cl.STRATA}}
    short = sorted((new[k]['range_m'], k) for k in new if new[k]['range_m'] < cl.MIN_VIEW_M)
    out['range_under_min_view'] = [
        {'key': k, 'stratum': new[k]['stratum'], 'pano_id': new[k]['pano']['pano_id'],
         'range_new_m': r, 'range_old_m': old[k]['range_m'],
         'case': new[k]['label_point']['case']} for r, k in short]
    if a.pilot:
        order = cl.read_corner_order(a.pilot)
        pos = {k: i for i, k in enumerate(order, 1)}
        out['pilot_positions_1_203'] = summarize(order[:203], old, new)
        for row in out['range_under_min_view']:
            row['pilot_position'] = pos.get(row['key'])
    for f in a.results:
        dets = sorted((d.get('pilot_order') or 0, d['corner'])
                      for ln in (json.loads(x) for x in open(f, encoding='utf-8'))
                      for d in ln['detections'])
        keys = [k for _o, k in dets]
        out[f'file:{Path(f).name}'] = summarize(keys, old, new)
        out[f'file:{Path(f).name}']['first_120'] = summarize(keys[:120], old, new)
        out[f'file:{Path(f).name}']['range_under_min_view'] = [
            (o, k, new[k]['range_m']) for o, k in dets if new[k]['range_m'] < cl.MIN_VIEW_M]
    path = Path(a.new) / 'label_shift.json'
    path.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'range_under_min_view'}, indent=1))
    print(f"{len(short)} labels with a new range under {cl.MIN_VIEW_M:g} m -> {path}")


if __name__ == '__main__':
    main()
