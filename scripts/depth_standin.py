"""How much of the per-pano height spread was Google's stand-in planes (issue #47).

Step 1 of #47 (docs/depth-at-detection-study.md 5.4) found an exactly level plane at
2.500 m under 10-27% of `floor` detections: Google's stand-in ground, appearing as a
*secondary* ground plane. `depth.classify_height` tests only the dominant plane for that
pattern, so until #47's follow-up `depth.ground_plane`'s `height_spread_m` -- the per-pano
sigma `depth.believe_height` hands `geo` under `--camera-height-m per-pano` -- took the
stand-ins in. It now leaves them out (`depth.is_standin`). This script measures what that
moved, pre-registered on #47 (the plan comment of 2026-09-27) before any number was run:

  - per city and capture year, the share of MEASURED panos with >= 1 stand-in secondary
    plane (`n_standin_planes` >= 1 on a measured pano: the dominant plane of a measured
    pano is never a stand-in, so every stand-in there is secondary);
  - `height_spread_m` old vs new over the measured panos: median and p90 of each, and how
    many panos move by >= 0.05 m;
  - reported beside those (no rule reads them): how many panos widened (removing a
    weighted point can move a percentile either way), and how many sit above geo's sigma
    floor under each definition (spread * SIGMA_PER_P10_P90 > the GSV error model's
    sigma_height_m), and the union that decides it: panos whose spread differs AND where
    max(old, new) is above the floor -- exactly the panos whose per-pano raycast sigma the
    change moves (added on the #97 review; no rule reads it).

It also ASSERTS the plan's invariant: every index column other than the spread and the two
new stand-in columns is byte-identical between the old and the reindexed index.csv for
every panorama (`camera_height_m`, tilt, pixel share, plane count, degenerate flag, sha256,
...), so the dominant-plane choice and every status are untouched. Any difference is
printed and the run exits 1.

Two steps, because the old index has to be captured before `harvest_depth.py --reindex`
rewrites it:

    python scripts/depth_standin.py snapshot            # copy each depth/index.csv aside
    python scripts/harvest_depth.py runs/paterson --reindex    # ...per city
    python scripts/depth_standin.py measure             # old vs new -> runs/_summary/depth_standin/

`measure` writes spread_change.csv (per city + pooled) and by_year.csv (per city x capture
year) to runs/_summary/depth_standin/ and copies both to docs/figures/camera-height/data/
(tracked). No GPU, no network; stdlib on top of depth.py and geo.py.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import depth as depthlib  # noqa: E402
import geo  # noqa: E402

CITIES = ('bend', 'paterson', 'gainesville', 'sao_paulo')
CHANGE_M = 0.05                      # pre-registered: "how many panos change by >= 0.05 m"
NEW_COLUMNS = ('n_standin_planes', 'standin_pixel_share')
MOVED_COLUMNS = ('height_spread_m',) + NEW_COLUMNS
# The spread at which geo's floor stops binding: spread * SIGMA_PER_P10_P90 > sigma floor.
SIGMA_FLOOR_SPREAD_M = geo.GSV_ERRORS.sigma_height_m / depthlib.SIGMA_PER_P10_P90
DOCS_DATA = REPO_ROOT / 'docs' / 'figures' / 'camera-height' / 'data'


def summary_dir(runs_root):
    return Path(runs_root) / '_summary' / 'depth_standin'


def old_index_path(runs_root, city):
    return summary_dir(runs_root) / 'old_index' / f'{city}.csv'


def read_index(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, {r['panorama_id']: r for r in reader}


def status_of(row):
    """depth.classify_height from an index row, exactly as fuse_sites.load_depth_index."""
    h = float(row['camera_height_m']) if row['camera_height_m'] else None
    tilt = float(row['ground_tilt_deg']) if row['ground_tilt_deg'] else None
    return depthlib.classify_height(h, tilt, degenerate=row['degenerate'] == '1')


def capture_years(results_path):
    """{panorama_id: 'YYYY' or '????'} from a run's results.jsonl (read-only)."""
    years = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pano = json.loads(line)['pano']
                years[pano['panorama_id']] = (pano.get('capture_date') or '????')[:4]
    return years


def pct(values, q):
    """Nearest-rank percentile (harvest_depth.summarize's), None for no values."""
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def invariant_violations(old, new):
    """[(pano_id, column, old, new)] for every difference outside MOVED_COLUMNS, plus any
    pano present in one index and not the other (column '<row>')."""
    out = [(pid, '<row>', 'present', 'absent') for pid in sorted(set(old) - set(new))]
    out += [(pid, '<row>', 'absent', 'present') for pid in sorted(set(new) - set(old))]
    for pid in sorted(set(old) & set(new)):
        for col, value in old[pid].items():
            if col not in MOVED_COLUMNS and new[pid].get(col) != value:
                out.append((pid, col, value, new[pid].get(col)))
    return out


def pano_rows(old, new, years):
    """One dict per MEASURED pano: year, old/new spread, stand-in count and share.

    A pano missing from the old index is skipped: invariant_violations has already
    reported it as a lost row, and cmd_measure exits 1 on that, not on a KeyError here."""
    rows = []
    for pid, n in new.items():
        if pid not in old or status_of(n) != depthlib.MEASURED:
            continue
        rows.append({'pano_id': pid, 'year': years.get(pid, '????'),
                     'spread_old': float(old[pid]['height_spread_m']),
                     'spread_new': float(n['height_spread_m']),
                     'n_standin': int(n['n_standin_planes']),
                     'standin_share': float(n['standin_pixel_share'])})
    return rows


def summarize(rows):
    """The pre-registered numbers (and the reported-only ones) over pano rows."""
    n = len(rows)
    with_si = [r for r in rows if r['n_standin'] >= 1]
    old = [r['spread_old'] for r in rows]
    new = [r['spread_new'] for r in rows]
    # Rounded first: the index values carry 4 decimals, and without it float error makes a
    # difference of exactly 0.0500 read as 0.04999... and miss the pre-registered bar.
    changed = [r for r in rows
               if round(abs(r['spread_new'] - r['spread_old']), 4) >= CHANGE_M]

    def rnd(v, nd=4):
        return None if v is None else round(v, nd)

    return {
        'n_measured': n,
        'n_with_standin': len(with_si),
        'share_with_standin': rnd(len(with_si) / n if n else None),
        'standin_pixel_share_p50': rnd(pct([r['standin_share'] for r in with_si], .5)),
        'spread_old_p50': rnd(pct(old, .5)), 'spread_old_p90': rnd(pct(old, .9)),
        'spread_new_p50': rnd(pct(new, .5)), 'spread_new_p90': rnd(pct(new, .9)),
        'n_changed_ge_0p05': len(changed),
        'share_changed_ge_0p05': rnd(len(changed) / n if n else None),
        # reported only
        'n_narrowed': sum(r['spread_new'] < r['spread_old'] for r in rows),
        'n_widened': sum(r['spread_new'] > r['spread_old'] for r in rows),
        'n_old_above_sigma_floor': sum(v > SIGMA_FLOOR_SPREAD_M for v in old),
        'n_new_above_sigma_floor': sum(v > SIGMA_FLOOR_SPREAD_M for v in new),
        'n_sigma_changed': sum(1 for r in rows if r['spread_new'] != r['spread_old']
                               and max(r['spread_old'], r['spread_new'])
                               > SIGMA_FLOOR_SPREAD_M),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def cmd_snapshot(args):
    for city in args.cities:
        src = args.runs_root / city / 'depth' / 'index.csv'
        dst = old_index_path(args.runs_root, city)
        if dst.exists():
            sys.exit(f'{dst} already exists; it is the pre-#47 index -- refusing to '
                     f'overwrite it (delete it by hand to re-snapshot)')
        fields, _ = read_index(src)
        if set(NEW_COLUMNS) <= set(fields):
            sys.exit(f'{src} already has the #47 columns; it is not the old index')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'{city}: {src} -> {dst}')


def cmd_measure(args):
    spread_rows, year_rows, pooled, bad = [], [], [], 0
    for city in args.cities:
        _, old = read_index(old_index_path(args.runs_root, city))
        fields, new = read_index(args.runs_root / city / 'depth' / 'index.csv')
        if not set(NEW_COLUMNS) <= set(fields):
            sys.exit(f'{city}: depth/index.csv has not been reindexed '
                     f'(python scripts/harvest_depth.py runs/{city} --reindex)')
        violations = invariant_violations(old, new)
        if violations:
            bad += len(violations)
            print(f'{city}: INVARIANT FAILED -- {len(violations)} difference(s) outside '
                  f'{MOVED_COLUMNS}, e.g. {violations[:3]}')
        else:
            print(f'{city}: invariant holds -- {len(new)} rows identical outside '
                  f'{MOVED_COLUMNS}')
        years = capture_years(args.runs_root / city / 'results.jsonl')
        rows = pano_rows(old, new, years)
        pooled += rows
        spread_rows.append({'city': city, 'n_indexed': len(new), **summarize(rows)})
        for year in sorted({r['year'] for r in rows}):
            year_rows.append({'city': city, 'capture_year': year,
                              **summarize([r for r in rows if r['year'] == year])})
    if len(args.cities) > 1:
        spread_rows.append({'city': 'pooled', 'n_indexed': sum(r['n_indexed']
                                                                for r in spread_rows),
                            **summarize(pooled)})

    out = summary_dir(args.runs_root)
    write_csv(out / 'spread_change.csv', spread_rows)
    write_csv(out / 'by_year.csv', year_rows)
    if not args.no_docs_copy:
        DOCS_DATA.mkdir(parents=True, exist_ok=True)
        for name in ('spread_change.csv', 'by_year.csv'):
            shutil.copy2(out / name, DOCS_DATA / f'standin_{name}')
    for r in spread_rows:
        print(f"{r['city']:>12}: {r['share_with_standin']:.3f} of {r['n_measured']} measured "
              f"with a stand-in; spread p50 {r['spread_old_p50']} -> {r['spread_new_p50']}, "
              f"p90 {r['spread_old_p90']} -> {r['spread_new_p90']}; "
              f"{r['n_changed_ge_0p05']} moved >= {CHANGE_M} m")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=('snapshot', 'measure'))
    ap.add_argument('cities', nargs='*', default=list(CITIES))
    ap.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--no-docs-copy', action='store_true',
                    help='measure: do not copy the CSVs into docs/figures/camera-height/data/')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    {'snapshot': cmd_snapshot, 'measure': cmd_measure}[args.command](args)


if __name__ == '__main__':
    main()
