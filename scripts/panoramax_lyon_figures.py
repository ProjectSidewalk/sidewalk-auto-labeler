"""Figures for the first Lyon Panoramax slice (RampNet#159; docs/panoramax-lyon.md).

Two subcommands, so that what needs the run files and what needs neither stay apart:

    census-scan  runs/lyon/scan.json -> docs/figures/panoramax-lyon/data/scan_years.csv
                 Capture-year mix of the raw scan, each thinning (5/10/20 m), and the
                 slice --limit would pick (first N sorted ids of the 10 m set). No
                 network, no GPU; scan.json is gitignored, so this is the one step that
                 needs a local scan.
    rig-producers  runs/lyon/results.jsonl -> data/rig_producer_detections.csv
                 (--out data/full for the completed 10 m run, docs section 7)
                 Detections per pano per (rig, year, producer) group of >= 200 panos;
                 separates rig from operator. No network, no GPU.
    figures      committed CSVs only -> fig1_rig_rates, fig2_years as PNG (200 dpi) +
                 SVG. Reads data/scan_years.csv, data/census/{lyon,bayonne}/
                 rig_detections.csv + years.csv, and the six-run reference rates
                 committed with the Bayonne report
                 (docs/figures/panoramax-bayonne/data/fig5_detections.csv). No run
                 files, no network; byte-reproducible for a fixed matplotlib.

Usage:
    python scripts/panoramax_lyon_figures.py census-scan
    python scripts/panoramax_lyon_figures.py rig-producers
    python scripts/panoramax_lyon_figures.py figures
"""
import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / 'docs' / 'figures' / 'panoramax-lyon'
DATA = OUT / 'data'
CENSUS = DATA / 'census'
REFERENCE = REPO_ROOT / 'docs' / 'figures' / 'panoramax-bayonne' / 'data' / 'fig5_detections.csv'
SPACINGS = (5, 10, 20)
SLICE_SPACING, SLICE_LIMIT = 10, 40000     # the pre-registered slice (doc section 1)
MIN_GROUP_PANOS = 200                      # rig x year groups smaller than this are not drawn


# Same rule as #150's run_census.mask_producer (issue #149): Panoramax serves producer
# names publicly and some accounts use an email address as theirs, so a committed table
# never republishes one. TODO: import run_census.mask_producer once #150 is on main.
EMAIL_RE = re.compile(r'[^@\s]+@[^@\s]+\.[^@\s]+')


def mask_producer(name):
    """Mask a wholly email-shaped producer name to its first character; pass the rest.

        >>> mask_producer('jane.doe@example.org'), mask_producer('grand lyon')
        ('j***@***', 'grand lyon')
    """
    if isinstance(name, str) and EMAIL_RE.fullmatch(name.strip()):
        return name.strip()[0] + '***@***'
    return name

# dataviz reference palette, light mode (same tokens as panoramax_bayonne_figures.py)
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, BLUE_LIGHT = '#2a78d6', '#86b6ef'
SEQ = ('#e4eefb', '#cde2fb', '#a9cbf5', '#86b6ef', '#5f9de9', '#3987e5', '#1c5cab',
       '#0d366b')


def year_bucket(year):
    """Capture year -> the bucket fig2 draws ('<=2019', '2020'..'2026', 'unknown').

    Example:
        >>> [year_bucket(y) for y in ('2016', '2019', '2020', '2026', 'unknown', '')]
        ['<=2019', '<=2019', '2020', '2026', 'unknown', 'unknown']
    """
    if not year or not str(year)[:4].isdigit():
        return 'unknown'
    return '<=2019' if int(str(year)[:4]) <= 2019 else str(year)[:4]


# ===================================================================== census-scan
def census_scan(scan_path):
    """Year counts per set from scan.json rows [id, lat, lon, ts, density]."""
    from sources import panoramax
    scan = json.loads(Path(scan_path).read_text(encoding='utf-8'))
    panos = {row[0]: tuple(row[1:]) for row in scan['panos']}
    sets = {'raw': panos}
    for m in SPACINGS:
        sets[f'thin_{m}m'] = panoramax.thin_panos(panos, m)
    ids = sorted(sets[f'thin_{SLICE_SPACING}m'])[:SLICE_LIMIT]
    sets[f'slice_{SLICE_SPACING}m_first{SLICE_LIMIT}'] = {i: panos[i] for i in ids}
    counts = {name: Counter((v[2] or '')[:4] or 'unknown' for v in s.values())
              for name, s in sets.items()}
    years = sorted(set().union(*counts.values()))
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / 'scan_years.csv', 'w', newline='\n', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['capture_year', *counts])
        for y in years:
            w.writerow([y, *(counts[n][y] for n in counts)])
        w.writerow(['total', *(len(s) for s in sets.values())])
    print(f'wrote {DATA / "scan_years.csv"} (scanned_at {scan["scanned_at"]}, '
          f'{scan["pano_count"]} panos, failed_tiles {scan["failed_tiles"]})')


# =================================================================== rig-producers
def rig_producers(results_path, min_panos=200, out_dir=DATA):
    """Detections per pano per (rig, capture year, producer) for groups >= min_panos.

    In Lyon a camera model is mostly one operator's (ecartip drives most GoPro Max 2026
    panos, the Metropole the 8192x4096 Ladybug), so a per-camera rate is also a
    per-operator rate; this table is how far the two can be pulled apart. Besides the
    rates it carries, at the benchmark tier, the raw detection count (so a group's rate
    without one producer is a subtraction), the count on the camera rig
    (detectors.on_camera_rig), the share above the horizon (y < 0.5) and the median dip
    of the below-horizon detections in degrees -- the mount-geometry signal (a higher
    camera sees a ramp at a given distance at a steeper dip) -- and the share of panos
    whose reported pitch and roll are both exactly 0 (pose not levelled, or not known). Rates include on-rig
    detections, as detections.csv does. Reads the uncommitted results.jsonl.
    """
    import statistics
    from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE, on_camera_rig
    tiers = (OPERATIONAL_CONFIDENCE, BENCHMARK_CONFIDENCE)
    n, dets, seqs = Counter(), {t: Counter() for t in tiers}, {}
    on_rig, above, dips, zeros = Counter(), Counter(), {}, Counter()
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            key = (p.get('camera_make'), p.get('camera_model'),
                   f"{p.get('width')}x{p.get('height')}",
                   (p.get('capture_date') or '')[:4] or 'unknown', mask_producer(p.get('copyright')))
            n[key] += 1
            seqs.setdefault(key, set()).add(p.get('sequence_id'))
            zeros[key] += (p.get('camera_pitch') == 0 and p.get('camera_roll') == 0)
            for t in tiers:
                dets[t][key] += sum(d['confidence'] >= t for d in rec.get('detections') or [])
            for d in rec.get('detections') or []:
                if d['confidence'] < BENCHMARK_CONFIDENCE:
                    continue
                y = d['y_normalized']
                on_rig[key] += on_camera_rig(y)
                if y < 0.5:
                    above[key] += 1
                else:
                    dips.setdefault(key, []).append((y - 0.5) * 180.0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = f'{BENCHMARK_CONFIDENCE:g}'
    with open(out_dir / 'rig_producer_detections.csv', 'w', newline='\n', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['camera_make', 'camera_model', 'dimensions', 'capture_year', 'producer',
                    'panos', 'sequences',
                    *(f'detections_per_pano_{t:g}' for t in tiers),
                    f'detections_{b}', f'on_rig_{b}', f'above_horizon_share_{b}',
                    f'median_dip_deg_below_horizon_{b}', 'pose_zeros_share'])
        for key, k in sorted(n.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            if k >= min_panos:
                kb = dets[BENCHMARK_CONFIDENCE][key]
                dip = dips.get(key)
                w.writerow([*key, k, len(seqs[key]),
                            *(round(dets[t][key] / k, 3) for t in tiers),
                            kb, on_rig[key], round(above[key] / kb, 4) if kb else '',
                            round(statistics.median(dip), 1) if dip else '',
                            round(zeros[key] / k, 4)])
    print(f'wrote {out_dir / "rig_producer_detections.csv"}')


# ========================================================================= figures
def _read(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'svg.hashsalt': 'panoramax-lyon-159', 'svg.fonttype': 'none',
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'legend.frameon': False})
    return plt


def _save(fig, name):
    """PNG (200 dpi) + SVG with no timestamps; the SVG is written with LF endings."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight',
                metadata={'Software': None})
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', metadata={'Date': None,
                                                                  'Creator': None})
    with open(OUT / f'{name}.svg', 'w', encoding='utf-8', newline='\n') as f:
        f.write(buf.getvalue())
    print(f'wrote {OUT / name}.png/.svg')


def _rig_label(r):
    make = r['camera_make'] or '(no make)'
    model = r['camera_model'] or ''
    return f"{make} {model}".strip() + f"  {r['dimensions']}  {r['capture_year']}"


def fig1_rig_rates(plt):
    """Detections per pano by rig x capture year, Lyon slice beside Bayonne, at both
    tiers, with the six whole-run rates (0.55) as the reference panel."""
    panels = []
    for city, title in (('lyon', 'Lyon slice 1 (Panoramax)'),
                        ('bayonne', 'Bayonne, whole run (Panoramax)')):
        rows = [r for r in _read(CENSUS / city / 'rig_detections.csv')
                if int(r['panos']) >= MIN_GROUP_PANOS]
        panels.append((title, [(_rig_label(r), int(r['panos']),
                                float(r['detections_per_pano_0.55']),
                                float(r['detections_per_pano_0.3'])) for r in rows]))
    ref = [r for r in _read(REFERENCE) if r['tier'] == '0.55']
    ref.sort(key=lambda r: float(r['per_pano']))
    panels.append(('Six earlier runs, whole-run rate at 0.55 (Bayonne report fig5)',
                   [(r['run'].capitalize(), int(r['panos']), float(r['per_pano']), None)
                    for r in ref]))
    heights = [max(len(rows), 1) for _, rows in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 0.34 * sum(heights) + 2.6),
                             sharex=True, gridspec_kw={'height_ratios': heights})
    xmax = max(max(v for v in (row[2], row[3]) if v is not None)
               for _, rows in panels for row in rows) * 1.18
    for ax, (title, rows) in zip(axes, panels):
        rows = rows[::-1]
        ys = range(len(rows))
        for y, (label, n, r55, r30) in zip(ys, rows):
            if r30 is not None:
                ax.barh(y, r30, height=0.62, color=BLUE_LIGHT, edgecolor=SURFACE, lw=1)
            ax.barh(y, r55, height=0.62, color=BLUE, edgecolor=SURFACE, lw=1)
            txt = f'{r55:.3f}' + (f' / {r30:.3f}' if r30 is not None else '')
            ax.text(max(r55, r30 or 0) + xmax * 0.01, y, f'{txt}   n={n:,}', va='center',
                    fontsize=8, color=INK2)
        ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', visible=False)
        ax.set_xlim(0, xmax)
    axes[-1].set_xlabel('detections per processed pano')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=BLUE, label='at 0.55 (benchmark tier)'),
                            Patch(color=BLUE_LIGHT, label='at 0.30 (operational tier)')],
                   loc='lower right', fontsize=8)
    fig.suptitle('Detections per pano by camera model and capture year: Lyon slice vs Bayonne',
                 x=0.01, ha='left', fontweight='bold', fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig1_rig_rates')
    plt.close(fig)


def fig2_years(plt):
    """Capture-year mix: raw scan, 10 m thinned set, the slice as drawn from the scan,
    and the slice as processed (census) -- is the slice representative of the 10 m set?"""
    scan = {r['capture_year']: r for r in _read(DATA / 'scan_years.csv')}
    total = scan.pop('total')
    processed = {r['capture_year']: int(r['panos'])
                 for r in _read(CENSUS / 'lyon' / 'years.csv')}
    slice_col = f'slice_{SLICE_SPACING}m_first{SLICE_LIMIT}'
    bars = [('raw scan', {y: int(r['raw']) for y, r in scan.items()}, int(total['raw'])),
            (f'thinned {SLICE_SPACING} m', {y: int(r[f'thin_{SLICE_SPACING}m'])
                                           for y, r in scan.items()},
             int(total[f'thin_{SLICE_SPACING}m'])),
            ('slice (from scan)', {y: int(r[slice_col]) for y, r in scan.items()},
             int(total[slice_col])),
            ('slice (processed)', processed, sum(processed.values()))]
    buckets = sorted({year_bucket(y) for _, c, _ in bars for y in c},
                     key=lambda b: (b == 'unknown', b != '<=2019', b))
    colors = {b: SEQ[min(i, len(SEQ) - 1)] for i, b in enumerate(
        [b for b in buckets if b != 'unknown'])}
    colors['unknown'] = '#c3c2b7'
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for y, (label, counts, n) in enumerate(bars[::-1]):
        agg = Counter()
        for yr, k in counts.items():
            agg[year_bucket(yr)] += k
        left = 0.0
        for b in buckets:
            share = agg[b] / n if n else 0
            if share <= 0:
                continue
            ax.barh(y, share, left=left, height=0.62, color=colors[b], edgecolor=SURFACE,
                    lw=1.5, label=b if y == 0 else None)
            if share >= 0.06:
                dark = SEQ.index(colors[b]) >= 5 if colors[b] in SEQ else False
                ax.text(left + share / 2, y, f'{100 * share:.0f}%', ha='center', va='center',
                        fontsize=7.5, color='#ffffff' if dark else INK)
            left += share
        ax.text(1.01, y, f'n={n:,}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars[::-1]])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f}%'))
    ax.grid(axis='y', visible=False)
    ax.legend(ncol=len(buckets), loc='upper center', bbox_to_anchor=(0.5, -0.16),
              fontsize=8, title='capture year', title_fontsize=8)
    ax.set_title('Capture-year mix: thinning favours the newest captures; the slice '
                 'matches the 10 m set', fontsize=10)
    fig.tight_layout()
    _save(fig, 'fig2_years')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    cs = sub.add_parser('census-scan')
    cs.add_argument('--scan', default=str(REPO_ROOT / 'runs' / 'lyon' / 'scan.json'))
    rp = sub.add_parser('rig-producers')
    rp.add_argument('--results', default=str(REPO_ROOT / 'runs' / 'lyon' / 'results.jsonl'))
    rp.add_argument('--out', default=str(DATA),
                    help='output dir (default the slice-1 data dir; the full 10 m run writes data/full/)')
    sub.add_parser('figures')
    args = ap.parse_args()
    if args.cmd == 'census-scan':
        census_scan(args.scan)
    elif args.cmd == 'rig-producers':
        rig_producers(args.results, out_dir=args.out)
    else:
        plt = _style()
        fig1_rig_rates(plt)
        fig2_years(plt)


if __name__ == '__main__':
    main()
