"""Part 1 of issue #106: PS label clustering scored on every RampNet benchmark city, pooled.

Runs scripts/eval_ps_clustering.py in --offline mode (labels synthesized from the run and
placed with the server's own estimator; see ps_placement.py) once per (city, tier, frame)
cell, then pools the cells into one table per (tier, frame):

  - tier 0.55 in the 2.6 m frame: the HEADLINE. Every judged RampNet bundle was reviewed
    at 0.55 and every published GT number is in the 2.6 m frame.
  - tier 0.30 in the 2.6 m frame: the operating point. The 0.30-0.55 band is UNJUDGED, so
    precision still counts only judged (>= 0.55) labels while coverage may rise.
  - tier 0.55 in the `auto` frame, GSV cities only (fuse_sites' default since #79).

Cities with a Project Sidewalk server get regions from its street network (GET
/v3/api/streets, cached at runs/<run>/ps_streets.geojson); the others are one region.
Pooled rows SUM numerators and denominators (covered/pool, fragments/ramps, dual pairs,
TP/FP, size-bucket T/F) over cities -- never an average of rates, and never a join on a
per-run id (label and cluster ids are per city).

Resumable: a cell whose arms.csv exists and whose report.md records the same results
sha256 and SCORER_VERSION is read back instead of re-run; --force re-runs it.

Usage:
    python scripts/clustering_eval_pooled.py                      # every cell, then pool
    python scripts/clustering_eval_pooled.py --cities laurens_gsv # one city (smoke test)
    python scripts/clustering_eval_pooled.py --pool-only          # re-pool existing cells
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fuse_sites as fs  # noqa: E402
import eval_ps_clustering as epc  # noqa: E402

SERVERS = {
    'paterson': 'https://sidewalk-paterson.cs.washington.edu',
    'gainesville': 'https://sidewalk-gainesville.cs.washington.edu',
    'sao_paulo': 'https://sidewalk-sao-paulo.cs.washington.edu',
    'laurens': 'https://sidewalk-laurens.cs.washington.edu',
    'richmond': 'https://sidewalk-richmond.cs.washington.edu',
}

# (name, benchmark split, run dir under runs/, results file, server key or None, source)
CITIES = [
    ('paterson', 'paterson', 'paterson', 'results.jsonl', 'paterson', 'gsv'),
    ('gainesville', 'gainesville', 'gainesville', 'results.jsonl', 'gainesville', 'gsv'),
    ('sao_paulo', 'sao_paulo', 'sao_paulo', 'results.jsonl', 'sao_paulo', 'gsv'),
    ('bend', 'bend', 'bend', 'results.jsonl', None, 'gsv'),
    # the same city as laurens (Mapillary), so the same server's street network
    ('laurens_gsv', 'laurens_gsv', 'laurens_gsv', 'results.jsonl', 'laurens', 'gsv'),
    ('richmond', 'richmond', 'richmond', 'results.jsonl', 'richmond', 'mapillary'),
    ('clovis', 'clovis', 'clovis', 'results.jsonl', None, 'mapillary'),
    ('morgantown', 'morgantown', 'morgantown', 'results.jsonl', None, 'mapillary'),
    ('annapolis', 'annapolis', 'annapolis', 'results.jsonl', None, 'mapillary'),
    # prod went live from results.raw.jsonl (raw GPS), so that is the file scored
    ('laurens', 'laurens_mapillary', 'laurens', 'results.raw.jsonl', 'laurens', 'mapillary'),
]
EXCLUDED = {
    'budapest_district5': 'partial local file: runs/budapest_district5/results.jsonl holds '
                          '300 of the 18,183 panos the run processed, and 2 of the 125 '
                          'benchmark panos (the full file was never copied back here)',
}

# (tier, frame, GSV only)
CELLS = [(0.55, 2.6, False), (0.30, 2.6, False), (0.55, 'auto', True)]
HEADLINE = (0.55, 2.6)

# Live reports whose `deployed` row joins the per-city table, by (city, tier, frame).
LIVE = {('richmond', 0.55, 2.6): 'runs/richmond/ps_clustering_eval',
        ('laurens', 0.30, 2.6): 'runs/laurens/ps_clustering_eval'}

ARMS = ['deployed', 'ps @ 7.5 m', 'ps_citywide @ 7.5 m', 'ps @ 12.5 m', 'ps @ 15 m',
        'ps_raycast @ 7.5 m', 'fusion', 'fusion_server', 'fusion_server+attach']
SUM_COLS = ['n_clusters', 'n_placed', 'n_labels', 'tp', 'fp', 'covered', 'n_pool',
            'frag3_ramps', 'frag3_with_extra', 'frag3_extra', 'frag5_with_extra',
            'frag5_extra', 'dual_pairs', 'dual_both', 'dual_one', 'dual_neither',
            'same_pano_pairs']
SIZE_COLS = [f'{k}_{f}' for k in ('sz_1', 'sz_2', 'sz_3p', 'sz_unpl')
             for f in ('n', 'judged', 't', 'f')]
OUT = REPO_ROOT / 'runs' / '_pooled' / 'ps_clustering_eval'


def cell_dir(run_dir, tier, frame):
    return run_dir / (f'ps_clustering_eval_offline{fs.frame_suffix(frame)}_t{tier:g}')


def cell_current(out, results_sha):
    """True when out/ holds arms.csv and a report stamped with this results file's sha256
    and this SCORER_VERSION."""
    rep, arms = out / 'report.md', out / 'arms.csv'
    if not (rep.exists() and arms.exists()):
        return False
    head = rep.read_text(encoding='utf-8')[:4000]
    return (f'scorer {epc.SCORER_VERSION};' in head and f'sha256 `{results_sha}`' in head)


def streets_for(run_dir, server_key, refresh=False):
    """The server's street network (GET, cached beside the run), or None."""
    if server_key is None:
        return None
    path = run_dir / 'ps_streets.geojson'
    epc.fetch(SERVERS[server_key] + epc.API_STREETS, path, refresh)
    return path


def run_cell(city, tier, frame, force=False, refresh=False, bench=None):
    name, split, run, results, server_key, _source = city
    run_dir = REPO_ROOT / 'runs' / run
    results_path = run_dir / results
    out = cell_dir(run_dir, tier, frame)
    sha = fs.file_sha256(results_path)
    if not force and cell_current(out, sha):
        print(f'[{name} t{tier:g} {frame}] current, reading back', flush=True)
        return out
    streets = streets_for(run_dir, server_key, refresh)
    argv = [name, '--split', split, '--run-dir', str(run_dir), '--results', str(results_path),
            '--offline', '--min-confidence', str(tier), '--camera-height-m', str(frame),
            '--out', str(out)]
    if streets:
        argv += ['--streets', str(streets)]
    if bench:
        argv += ['--benchmark-root', str(bench)]
    print(f'[{name} t{tier:g} {frame}] running: eval_ps_clustering.py {" ".join(argv)}',
          flush=True)
    epc.run(epc.build_parser().parse_args(argv))
    return out


def read_arms(path):
    with open(path, encoding='utf-8', newline='') as f:
        return {r['arm']: r for r in csv.DictReader(f)}


ATTACH_RE = re.compile(r'unplaceable labels: (\d+); attached (\d+)')


def read_attach(report):
    m = ATTACH_RE.search(report.read_text(encoding='utf-8'))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def num(v):
    return 0 if v in (None, '') else int(float(v))


def rate(k, n, nd=3):
    return 'n/a' if not n else f'{k / n:.{nd}f}'


def sized(r):
    """'t/(t+f)' precision per size bucket."""
    out = []
    for k in ('sz_1', 'sz_2', 'sz_3p', 'sz_unpl'):
        t, f = num(r.get(f'{k}_t')), num(r.get(f'{k}_f'))
        out.append(f'{rate(t, t + f, 2)} ({t}/{t + f})')
    return out


TABLE_HEAD = ('| {first} | arm | clusters | labels/cluster | coverage | frag 3 m | frag 5 m '
              '| dual both/one/neither | precision (TP/FP) | prec. size 1 | size 2 | size 3+ '
              '| unplaceable |\n'
              '|---|---|---:|---:|---|---|---|---|---|---|---|---|---|')


def table_row(first, arm, r):
    return (f"| {first} | {arm} | {num(r['n_clusters'])} | "
            f"{rate(num(r['n_labels']), num(r['n_clusters']), 2)} | "
            f"{rate(num(r['covered']), num(r['n_pool']))} ({num(r['covered'])}/"
            f"{num(r['n_pool'])}) | {rate(num(r['frag3_with_extra']), num(r['frag3_ramps']), 2)} | "
            f"{rate(num(r['frag5_with_extra']), num(r['frag3_ramps']), 2)} | "
            f"{num(r['dual_both'])}/{num(r['dual_one'])}/{num(r['dual_neither'])} | "
            f"{rate(num(r['tp']), num(r['tp']) + num(r['fp']))} ({num(r['tp'])}/"
            f"{num(r['fp'])}) | " + ' | '.join(sized(r)) + ' |')


def pool(rows):
    """Sum every count column over rows (one arm, several cities)."""
    out = {c: sum(num(r.get(c)) for r in rows) for c in SUM_COLS + SIZE_COLS}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cities', nargs='*', default=None,
                    help='names from the CITY table (default: all)')
    ap.add_argument('--force', action='store_true', help='re-run cells that are current')
    ap.add_argument('--refresh', action='store_true', help='re-pull the street networks')
    ap.add_argument('--pool-only', action='store_true',
                    help='only pool cells already on disk (no scoring)')
    ap.add_argument('--benchmark-root', type=Path, default=None)
    args = ap.parse_args(argv)
    names = {c[0] for c in CITIES}
    unknown = set(args.cities or ()) - names
    if unknown:
        raise SystemExit(f'unknown cities {sorted(unknown)}; choose from {sorted(names)}'
                         + ''.join(f'\n  {k} excluded: {v}' for k, v in EXCLUDED.items()))
    cities = [c for c in CITIES if args.cities is None or c[0] in args.cities]

    if not args.pool_only:
        for city in cities:
            for tier, frame, gsv_only in CELLS:
                if gsv_only and city[5] != 'gsv':
                    continue
                run_cell(city, tier, frame, args.force, args.refresh, args.benchmark_root)

    per_city, attach_rows, missing = [], [], []
    for name, _split, run, results, _srv, source in CITIES:
        run_dir = REPO_ROOT / 'runs' / run
        sha = fs.file_sha256(run_dir / results)
        for tier, frame, gsv_only in CELLS:
            if gsv_only and source != 'gsv':
                continue
            out = cell_dir(run_dir, tier, frame)
            if not cell_current(out, sha):
                missing.append(f'{name} t{tier:g} {frame}')
                continue
            arms = read_arms(out / 'arms.csv')
            live = LIVE.get((name, tier, frame))
            if live:
                arms = {'deployed': read_arms(REPO_ROOT / live / 'arms.csv')['deployed'],
                        **arms}
            for arm in ARMS:
                if arm in arms:
                    per_city.append({'tier': tier, 'frame': str(frame), 'city': name,
                                     'source': source, 'arm': arm,
                                     **{c: arms[arm].get(c, '') for c in SUM_COLS + SIZE_COLS}})
            unpl, att = read_attach(out / 'report.md')
            attach_rows.append({'tier': tier, 'frame': str(frame), 'city': name,
                                'source': source, 'unplaceable': unpl, 'attached': att,
                                'clusters_server': num(arms['fusion_server']['n_clusters']),
                                'clusters_attach':
                                    num(arms['fusion_server+attach']['n_clusters'])})

    # A run from before the storage floor stored only >= 0.55 detections, so its 0.30 cell
    # is its 0.55 cell over again: detected by an identical label count, listed, and kept
    # out of the 0.30 table (it would otherwise dilute the band's effect with no band).
    labels_at = {(r['city'], r['tier'], r['frame']): num(r['n_labels']) for r in per_city
                 if r['arm'] == 'ps @ 7.5 m'}
    no_band = sorted(c for (c, t, f), n in labels_at.items()
                     if t == 0.30 and labels_at.get((c, 0.55, f)) == n)
    per_city = [r for r in per_city if not (r['tier'] == 0.30 and r['city'] in no_band)]
    attach_rows = [r for r in attach_rows if not (r['tier'] == 0.30 and r['city'] in no_band)]

    pooled = []
    groups = (('GSV', ('gsv',)), ('Mapillary', ('mapillary',)), ('all', ('gsv', 'mapillary')))
    for tier, frame, _g in CELLS:
        for gname, srcs in groups:
            for arm in ARMS:
                if arm == 'deployed':
                    continue            # two cities at different tiers: never pooled
                rows = [r for r in per_city if r['tier'] == tier and r['frame'] == str(frame)
                        and r['source'] in srcs and r['arm'] == arm]
                if rows:
                    pooled.append({'tier': tier, 'frame': str(frame), 'group': gname,
                                   'arm': arm, 'n_cities': len(rows), **pool(rows)})

    OUT.mkdir(parents=True, exist_ok=True)
    for fname, rows in (('per_city.csv', per_city), ('pooled.csv', pooled)):
        if rows:
            with open(OUT / fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    lines = ['# PS label clustering on every benchmark city (issue #106, Part 1)', '',
             'Every cell is `scripts/eval_ps_clustering.py --offline`: one label per stored '
             'detection at the tier (rig-masked), placed where the server would place it '
             "(ps_placement, the server's estimator at 2.341 m), regions from the city "
             "server's street network where one exists (else one region, so `ps @ t` == "
             '`ps_citywide`). Scored against RampNet GT at the 5 m match radius; every '
             "cluster at the mean of its members' raycast positions in the named frame. "
             'Pooled rows sum counts over cities, never rates. Per-city reports: '
             '`runs/<run>/ps_clustering_eval_offline<frame>_t<tier>/report.md`.', '',
             f'scorer {epc.SCORER_VERSION}. Excluded: '
             + '; '.join(f'`{k}` ({v})' for k, v in EXCLUDED.items()) + '.',
             'Bend was a RampNet training city: its rows measure clustering, not detector '
             'generalization.']
    if missing:
        lines.append('warning: cells missing or stale (not pooled): ' + ', '.join(missing))
    lines += ['', '**Columns.** coverage = pool GT ramps with a cluster within 5 m '
              '(one-to-one); frag r = share of covered ramps with an extra cluster within r '
              'that is no GT ramp\'s match; dual = same-pano GT pairs < 5 m apart, both / one '
              '/ neither matched to distinct clusters; precision = clusters with a judged '
              'member, TP if any member is verdict-true; prec. size k = per-label precision '
              'bucketed by the size of the label\'s cluster, `unplaceable` = labels the raycast '
              'cannot place. `fusion_server+attach` differs from `fusion_server` only in '
              'cluster count and size buckets (its attached labels have no raycast position).']
    for tier, frame, gsv_only in CELLS:
        tag = f'tier {tier:g}, {fs.frame_label(frame)} frame' + (' (GSV only)' if gsv_only
                                                                   else '')
        head = ' -- HEADLINE' if (tier, frame) == HEADLINE else ''
        lines += ['', f'## {tag}{head}', '']
        if tier == 0.30:
            lines.append('The 0.30-0.55 band is unjudged: precision counts only judged '
                         '(>= 0.55) labels, while coverage can rise with the extra labels. '
                         'Left out: ' + (', '.join(f'`{c}`' for c in no_band) or 'none')
                         + ' -- runs from before the storage floor, which stored only '
                         '>= 0.55 detections, so they have no 0.30-0.55 band to add.')
            lines.append('')
        lines += ['### Pooled', '', TABLE_HEAD.format(first='group (cities)')]
        for r in pooled:
            if r['tier'] == tier and r['frame'] == str(frame):
                lines.append(table_row(f"{r['group']} ({r['n_cities']})", r['arm'], r))
        lines += ['', '### Per city', '', TABLE_HEAD.format(first='city')]
        for r in per_city:
            if r['tier'] == tier and r['frame'] == str(frame):
                lines.append(table_row(r['city'], r['arm'], r))
        lines += ['', '### Unplaceable labels: attach by bearing', '',
                  '| city | unplaceable | attached | share | clusters fusion_server -> +attach |',
                  '|---|---:|---:|---:|---|']
        tot_u = tot_a = 0
        for r in attach_rows:
            if r['tier'] == tier and r['frame'] == str(frame):
                tot_u += r['unplaceable']
                tot_a += r['attached']
                lines.append(f"| {r['city']} | {r['unplaceable']} | {r['attached']} | "
                             f"{rate(r['attached'], r['unplaceable'], 2)} | "
                             f"{r['clusters_server']} -> {r['clusters_attach']} |")
        lines.append(f'| all | {tot_u} | {tot_a} | {rate(tot_a, tot_u, 2)} | |')
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'wrote {OUT / "report.md"}, per_city.csv, pooled.csv')


if __name__ == '__main__':
    main()
