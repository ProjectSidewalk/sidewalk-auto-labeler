"""Figures for docs/panoramax-asset-study.md (issue #161), via
`python scripts/panoramax_asset_study.py figures` (and `contact-sheet --data <dir>`).

`main()` reads only the committed CSVs / detections.jsonl under
runs/_summary/panoramax_asset/ (no GPU, no network) and writes PNG (200 dpi) + SVG with no
timestamps, so a re-run is byte-identical. `contact_sheet()` is the one output that needs
the local pixels: it cuts crops from the archived HD and the downloaded SD of a seeded set
of disagreements, each captioned with its producer and licence (CC BY-SA / Etalab).
"""
import csv
import io
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'runs' / '_summary' / 'panoramax_asset'
OUT = REPO / 'docs' / 'figures' / 'panoramax-asset'

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SERIES = {'sd': '#2a78d6', 'sd_sim': '#eb6834'}     # categorical slots 1, 2 (fixed order)
ARM_LABEL = {'sd': 'SD asset (real)', 'sd_sim': 'HD resampled to 2048 px (exploratory)'}
SEED = 161


def _read(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'svg.hashsalt': 'panoramax-asset-161', 'svg.fonttype': 'none',
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'legend.frameon': False})
    return plt


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight', metadata={'Software': None})
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', metadata={'Date': None, 'Creator': None})
    with open(OUT / f'{name}.svg', 'w', encoding='utf-8', newline='\n') as f:
        f.write(buf.getvalue())
    print(f'wrote {OUT / name}.png/.svg')


def _f(v):
    return float(v) if v not in ('', None) else float('nan')


def fig_agreement(plt):
    """Q: at the operational tier, does SD find what HD finds (recall) and only that
    (precision), pooled and per rig? Dots = point estimates, whiskers = 95% pano-bootstrap
    intervals, dashed lines = the pre-registered bars (0.95 pooled, 0.90 per rig)."""
    summary = [r for r in _read(DATA / 'summary.csv') if _f(r['tier']) == 0.3]
    rigs = [r for r in _read(DATA / 'per_rig.csv') if _f(r['tier']) == 0.3]
    names = sorted({r['rig'] for r in rigs},
                   key=lambda n: (-int(next(r['panos'] for r in rigs if r['rig'] == n)), n))
    rows = ['All sampled panos'] + names
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(rows) + 1.8), sharey=True)
    for ax, metric, title in ((axes[0], 'recall', 'Recall of HD detections'),
                              (axes[1], 'precision', 'Precision against HD')):
        for k, arm in enumerate(('sd', 'sd_sim')):
            off = -0.15 if k == 0 else 0.15
            for i, name in enumerate(rows):
                src = (next(r for r in summary if r['arm'] == arm) if i == 0 else
                       next((r for r in rigs if r['arm'] == arm and r['rig'] == name), None))
                if src is None or src[metric] == '':
                    continue
                v, lo, hi = _f(src[metric]), _f(src[f'{metric}_lo']), _f(src[f'{metric}_hi'])
                ax.plot([lo, hi], [i + off] * 2, color=SERIES[arm], lw=2, solid_capstyle='round')
                ax.plot(v, i + off, 'o', ms=8 if i == 0 else 6, color=SERIES[arm],
                        mec=SURFACE, mew=1.5, label=ARM_LABEL[arm] if i == 0 else None)
                if arm == 'sd':
                    ax.text(1.005, i + off, f'{v:.3f}  (n={src["panos"]})', va='center',
                            fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
        ax.axvline(0.95, color=INK2, ls='--', lw=1)
        ax.axvline(0.90, color=MUTED, ls=':', lw=1)
        ax.text(0.952, -0.9, 'pooled\nbar 0.95', fontsize=7, color=INK2, ha='left',
                va='center')
        ax.text(0.898, -0.9, 'rig bar\n0.90', fontsize=7, color=MUTED, ha='right',
                va='center')
        ax.set_title(title)
        ax.set_xlim(0.4 if metric == 'recall' else 0.6, 1.0)
        ax.set_xlabel('share at confidence >= 0.30 (rig-masked)')
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows)
    axes[0].set_ylim(len(rows) - 0.4, -1.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Detecting on the SD derivative vs the HD original, Lyon sample (seed 161)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'agreement_by_rig')
    plt.close(fig)


def _matched_pairs(arm, tier=0.30):
    import panoramax_asset_study as study
    dets = {}
    with open(DATA / 'detections.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            dets.setdefault(r['panorama_id'], {})[r['arm']] = [tuple(d) for d in r['detections']]
    pairs, misses, extras = [], [], []
    for pid in sorted(dets):
        d = dets[pid]
        if 'hd' not in d or arm not in d:
            continue
        h = study.masked(d['hd'], tier)
        a = study.masked(d[arm], study.DETECTION_STORAGE_FLOOR)
        m = study.match(h, a)
        used = {j for j, _ in m.values()}
        for i, (j, _) in m.items():
            pairs.append((h[i][2], a[j][2]))
        misses += [(pid, h[i]) for i in range(len(h)) if i not in m]
        # an extra: SD >= tier with no HD peak >= the floor near it
        hf = study.masked(d['hd'], study.DETECTION_STORAGE_FLOOR)
        at = study.masked(d[arm], tier)
        m2 = study.match(at, hf)
        extras += [(pid, at[i]) for i in range(len(at)) if i not in m2]
    return pairs, misses, extras


def fig_confidence(plt):
    """Q: when SD and HD agree on a ramp, does SD score it lower? Each dot is one HD peak
    >= 0.30 and its one-to-one SD partner (any SD peak >= 0.1); the diagonal is "same
    confidence", the dashed lines the 0.30 operating point."""
    pairs, misses, _ = _matched_pairs('sd')
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.axhline(0.30, color=INK2, ls='--', lw=1)
    ax.axvline(0.30, color=INK2, ls='--', lw=1)
    ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=10, color=SERIES['sd'],
               alpha=0.55, edgecolors='none')
    below = sum(1 for h, s in pairs if s < 0.30)
    ax.text(0.98, 0.04, f'{len(pairs):,} matched HD peaks\n{below} fall below 0.30 on SD\n'
            f'{len(misses)} HD peaks with no SD partner >= 0.1', ha='right', va='bottom',
            fontsize=8, color=INK2, transform=ax.transAxes)
    ax.set_xlim(0.28, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('HD confidence')
    ax.set_ylabel('SD confidence (same ramp)')
    ax.set_title('Confidence of the same peak on SD vs HD')
    fig.tight_layout()
    _save(fig, 'confidence_sd_vs_hd')
    plt.close(fig)


def main():
    plt = _style()
    fig_agreement(plt)
    fig_confidence(plt)


def _window(img, cx, cy, r):
    """A (2r x 2r) crop centred on (cx, cy) that keeps the peak at the centre: wrapped
    across the 360-degree seam horizontally, padded with grey past the top/bottom edge."""
    from PIL import Image
    w, h = img.size
    side = int(round(2 * r))
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    out = Image.new('RGB', (side, side), (128, 128, 128))
    for shift in (-w, 0, w):                       # the image and its two seam neighbours
        left, right = max(0, x0 - shift), min(w, x0 - shift + side)
        top, bottom = max(0, y0), min(h, y0 + side)
        if left < right and top < bottom:
            out.paste(img.crop((left, top, right, bottom)), (left + shift - x0, top - y0))
    return out


def contact_sheet(data, per_kind=6):
    """A seeded sheet of disagreements at 0.30: HD peaks SD misses (no SD peak >= 0.1
    within the radius) and SD peaks HD has nothing near, `per_kind` of each drawn with
    random.Random(161) from the sorted lists. Each row shows the HD crop and the SD crop
    of the same 12-degree window, with its producer and licence."""
    plt = _style()
    from PIL import Image
    sample = {r['panorama_id']: r for r in _read(DATA / 'sample.csv')}
    _, misses, extras = _matched_pairs('sd')
    rng = random.Random(SEED)
    picks = []
    for kind, pool in (('HD only (SD misses it)', misses), ('SD only (not on HD)', extras)):
        pool = sorted(pool)
        rng.shuffle(pool)
        picks += [(kind, pid, pt) for pid, pt in pool[:per_kind]]
    fig, axes = plt.subplots(len(picks), 2, figsize=(7, 2.15 * len(picks)))
    half = 1 / 60    # +-6 degrees of the 360-degree width around the peak
    for row, (kind, pid, (x, y, c)) in enumerate(picks):
        s = sample[pid]
        for col, arm in enumerate(('hd', 'sd')):
            img = Image.open(data / arm / f'{pid}.jpg').convert('RGB')
            w, h = img.size
            crop = _window(img, x * w, y * h, half * w)
            ax = axes[row][col]
            ax.imshow(crop)
            ax.plot(crop.size[0] / 2, crop.size[1] / 2, marker='+', ms=14, mew=1.5,
                    color='#e34948')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            label = f'{arm.upper()} {w}x{h}'
            if col == 0:
                label = f'{kind}, conf {c:.2f}\n' + label
            ax.set_title(label, fontsize=8, loc='left', fontweight='normal')
        axes[row][1].text(1.02, 0.5, f"{s['rig']}\n© {s['producer'] or 'unknown'}\n"
                          f"{s['license'] or 'CC-BY-SA-4.0'}, via Panoramax\n{pid[:8]}",
                          transform=axes[row][1].transAxes, fontsize=7, color=INK2,
                          va='center')
    fig.suptitle('SD/HD disagreements at 0.30, seeded (161); red cross = the peak',
                 x=0.01, y=0.998, ha='left', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    # PNG only: an SVG would embed the crops twice over for no gain.
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'contact_sheet_disagreements.png', dpi=150, bbox_inches='tight',
                metadata={'Software': None})
    print(f'wrote {OUT / "contact_sheet_disagreements.png"}')
    plt.close(fig)
