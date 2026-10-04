"""The RampNet heatmap's 8-cell grid, measured on this repo's runs (issue #111).

RampNet's head upsamples a stride-32 (64x128) map 8x bilinearly, and a bilinear surface
peaks only at its coarse sample points. With align_corners=False coarse sample k sits at
hi-res coordinate 8k + 3.5, so an integer argmax lands on column/row 3 or 4 mod 8, and the
real quantum of a stored detection is one coarse cell (8 heatmap px), not one heatmap px.
Two subcommands, no GPU, no network:

    # 1. confirm the grid: round(x*1024) % 8 and round(y*512) % 8 per city, per tier
    python scripts/heatmap_grid.py grid paterson bend gainesville sao_paulo richmond

    # 2. what sigma_peak_px 1.0 -> 2.31 (one coarse cell's uniform quantization) does to
    #    fusion: re-fuse at the benchmark tier and score against RampNet GT, at 2.6 m and
    #    at `auto`, then apply the rule pre-registered on #111
    python scripts/heatmap_grid.py sigma paterson bend gainesville sao_paulo richmond

Two more draw the figures in docs/heatmap-grid.md (docs/figures/heatmap-grid/*.png):

    # 3. GPU, offline: run RampNet on a local RampNet bundle's panos (default Richmond,
    #    Mapillary imagery under CC BY-SA) and cache two heatmap examples, a clean detection
    #    and an adjacent-coarse-cell flip under a resample of the same pano (the store-run
    #    perturbation, #56). Writes data/examples.npz, examples.json and one RGB crop.
    python scripts/heatmap_grid.py examples --city richmond

    # 4. no GPU, no network: draw every figure from the committed data
    python scripts/heatmap_grid.py figures

`sigma` fuses exactly as eval_sites.py does (tier 0.55, rig mask off, pose off) and reads
../RampNet/benchmark/<city> as data. Outputs land in docs/figures/heatmap-grid/data/
(grid.csv, sigma_peak.csv), which docs/heatmap-grid.md cites.

**The sigma rule** (posted on #111 before the run; `sigma_verdict`). Noise bound per cell =
one binomial standard error of the baseline value, sqrt(p(1-p)/n), n = the recall pool or
the judged operational sites. The new sigma becomes geo.SIGMA_PEAK_PX_DEFAULT iff in every
cell (city x height) |delta world recall| and |delta world precision| are each <= that
bound, AND neither chi-square gate rejections nor residual rejections rise by more than
10% (relative) in any cell. Otherwise the default stays and the difference is reported.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE  # noqa: E402

HEATMAP_W, HEATMAP_H = 1024, 512
CELL = geo.HEATMAP_COARSE_CELL_PX
OUT_DIR = REPO_ROOT / 'docs' / 'figures' / 'heatmap-grid' / 'data'
TIERS = (('stored', 0.0), ('0.30', OPERATIONAL_CONFIDENCE), ('0.55', BENCHMARK_CONFIDENCE))
GRID_RESIDUES = (3, 4)            # where a bilinear-8x argmax can land (module docstring)

# ---- Pre-registered on #111 (before any number was measured). Do not tune. ----
SIGMAS = (1.0, round(geo.SIGMA_PEAK_COARSE_CELL_PX, 2))   # baseline, candidate (2.31)
REJECTION_RISE_MAX = 0.10         # relative rise allowed in gate / residual rejections
# ----------------------------------------------------------------------------------


def grid_counts(results_path):
    """{tier: Counter(('x'|'y', residue) -> n) plus 'n'} over every stored detection.

    Example:
        >>> round(0.25 * HEATMAP_W) % CELL        # a detection at column 256
        0
    """
    out = {name: Counter() for name, _ in TIERS}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            for d in json.loads(line).get('detections', []):
                rx = round(d['x_normalized'] * HEATMAP_W) % CELL
                ry = round(d['y_normalized'] * HEATMAP_H) % CELL
                for name, floor in TIERS:
                    if d['confidence'] >= floor:
                        c = out[name]
                        c['n'] += 1
                        c[('x', rx)] += 1
                        c[('y', ry)] += 1
                        c['both_on_grid'] += rx in GRID_RESIDUES and ry in GRID_RESIDUES
    return out


def cmd_grid(args):
    rows = []
    for city in args.cities:
        path = args.runs_root / city / 'results.jsonl'
        if not path.exists():
            print(f'{city}: no {path}, skipped', file=sys.stderr)
            continue
        for tier, c in grid_counts(path).items():
            n = c['n']
            row = {'city': city, 'tier': tier, 'detections': n,
                   'both_on_grid_share': round(c['both_on_grid'] / n, 5) if n else None}
            for axis in ('x', 'y'):
                for r in range(CELL):
                    row[f'{axis}_mod8_{r}'] = c[(axis, r)]
            rows.append(row)
    print(f"{'city':<20} {'tier':>6} {'detections':>10} {'3|4 both axes':>14}  "
          f"x residues 0..7 / y residues 0..7")
    for r in rows:
        share = 'n/a' if r['both_on_grid_share'] is None else f"{r['both_on_grid_share']:.4f}"
        xs = ' '.join(str(r[f'x_mod8_{i}']) for i in range(CELL))
        ys = ' '.join(str(r[f'y_mod8_{i}']) for i in range(CELL))
        print(f"{r['city']:<20} {r['tier']:>6} {r['detections']:>10,} {share:>14}  "
              f"[{xs}] / [{ys}]")
    write_csv(args.out / 'grid.csv', rows)


def binomial_se(p, n):
    return math.sqrt(p * (1 - p) / n) if n and p is not None else None


def score(city, benchmark_root, run_dir, height, sigma):
    """One cell: fuse at the benchmark tier (as eval_sites.py does) and score it."""
    import eval_sites as es
    import fuse_sites as fs
    verdict_panos, bundle_ops, run_panos, resolved, auto = es.load_city_at_height(
        city, benchmark_root, run_dir, height)
    params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                           camera_height_m=resolved, apply_pose=fs.POSE_OFF,
                           sigma_peak_px=sigma)
    r = es.evaluate_city(verdict_panos, bundle_ops, run_panos, params)
    p, b, fz = r['precision'], r['buckets'], r['fuse']
    n_pool = r['n_pool_ramps']
    return {
        'city': city, 'height': str(height),
        'auto_resolved': '' if not auto else str(auto.get('resolved', '')),
        'sigma_peak_px': sigma,
        'n_pool': n_pool, 'world_recall': r['world_recall'],
        'own_view_recall': r['own_view_recall'],
        'union_lift': b['recovered_other_view'] / n_pool if n_pool else None,
        'recovered_other_view': b['recovered_other_view'],
        'judged_sites': p['tp'] + p['fp'], 'world_precision': p['value'],
        'tp': p['tp'], 'fp': p['fp'],
        'n_sites': fz['n_sites'], 'n_operational_sites': fz['n_operational_sites'],
        'n_multi_pano_sites': fz['n_multi_pano_sites'],
        'chi2_gate_rejections': fz['rejections']['chi2_gate'],
        'residual_rejections': fz['rejections']['residual'],
    }


def sigma_verdict(rows, baseline=SIGMAS[0], candidate=SIGMAS[1]):
    """(adopt, per-cell comparison rows) under the pre-registered rule (module docstring).

    Example:
        >>> base = {'city': 'x', 'height': '2.6', 'sigma_peak_px': 1.0, 'n_pool': 400,
        ...         'world_recall': 0.95, 'judged_sites': 400, 'world_precision': 0.95,
        ...         'chi2_gate_rejections': 100, 'residual_rejections': 10}
        >>> cand = dict(base, sigma_peak_px=2.31, world_recall=0.945,
        ...             chi2_gate_rejections=60, residual_rejections=11)
        >>> sigma_verdict([base, cand])[0]
        True
        >>> sigma_verdict([base, dict(cand, residual_rejections=12)])[0]   # +20%
        False
    """
    cells = {}
    for r in rows:
        cells.setdefault((r['city'], r['height']), {})[r['sigma_peak_px']] = r
    out, adopt = [], True
    for (city, height), by in sorted(cells.items()):
        b, c = by.get(baseline), by.get(candidate)
        if b is None or c is None:
            continue
        cmp = {'city': city, 'height': height}
        ok = True
        for metric, n_key in (('world_recall', 'n_pool'), ('world_precision', 'judged_sites')):
            se = binomial_se(b[metric], b[n_key])
            delta = (c[metric] - b[metric]) if None not in (b[metric], c[metric]) else None
            within = delta is not None and se is not None and abs(delta) <= se + 1e-12
            cmp[f'{metric}_delta'] = delta
            cmp[f'{metric}_se'] = se
            cmp[f'{metric}_within_noise'] = within
            ok &= within
        for metric in ('chi2_gate_rejections', 'residual_rejections'):
            rise = ((c[metric] - b[metric]) / b[metric]) if b[metric] else (
                0.0 if not c[metric] else math.inf)
            cmp[f'{metric}_rise'] = rise
            cmp[f'{metric}_ok'] = rise <= REJECTION_RISE_MAX
            ok &= cmp[f'{metric}_ok']
        cmp['pass'] = ok
        adopt &= ok
        out.append(cmp)
    return adopt and bool(out), out


def cmd_sigma(args):
    import fuse_sites as fs
    heights = [fs.fuse_camera_height_arg(h) for h in args.heights]
    rows = []
    for city in args.cities:
        run_dir = args.runs_root / city
        for height in heights:
            for sigma in SIGMAS:
                row = score(city, args.benchmark_root, run_dir, height, sigma)
                rows.append(row)
                print(f"{city:<12} h={str(height):<5} sigma={sigma:<4}  R {row['world_recall']:.4f}"
                      f"  P {row['world_precision']:.4f}  lift {row['union_lift']:.4f}"
                      f"  sites {row['n_sites']:,} (op {row['n_operational_sites']:,}, multi "
                      f"{row['n_multi_pano_sites']:,})  gate-rej {row['chi2_gate_rejections']:,}"
                      f"  resid-rej {row['residual_rejections']:,}", flush=True)
    write_csv(args.out / 'sigma_peak.csv', rows)
    adopt, cmp = sigma_verdict(rows)
    write_csv(args.out / 'sigma_peak_verdict.csv', cmp)
    print(f"\nrule (pre-registered on #111): |dR|, |dP| <= 1 binomial SE of the baseline in "
          f"every cell; gate / residual rejections rise <= {REJECTION_RISE_MAX:.0%}")
    for c in cmp:
        print(f"  {c['city']:<12} h={c['height']:<5} dR {c['world_recall_delta']:+.4f} "
              f"(SE {c['world_recall_se']:.4f})  dP {c['world_precision_delta']:+.4f} "
              f"(SE {c['world_precision_se']:.4f})  gate {c['chi2_gate_rejections_rise']:+.1%}"
              f"  resid {c['residual_rejections_rise']:+.1%}  -> {'pass' if c['pass'] else 'FAIL'}")
    print(f"verdict: {'ADOPT' if adopt else 'KEEP'} sigma_peak_px {SIGMAS[1]} "
          f"{'as the default' if adopt else '(default stays ' + str(SIGMAS[0]) + ')'}")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')


# ---- figures (docs/heatmap-grid.md) ------------------------------------------------------
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'heatmap-grid'
EXAMPLE_HALF = 32                 # heatmap crop half-width: 64x64 heatmap px = 8x8 coarse cells
RESAMPLE_SCALE = 0.75             # the perturbation: bilinear resample to 3/4 size + JPEG q90
FLIP_MIN, FLIP_MAX = 7, 9         # Chebyshev heatmap px that count as a coarse-cell flip
# Palette: the dataviz reference categorical slots 1-2 (blue, orange; validated for CVD
# separation on the light surface) and recessive ink for everything that is not data.
C_BLUE, C_ORANGE, C_INK, C_INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
C_MUTED, C_GRID, C_AXIS, C_FAIL = '#898781', '#e1e0d9', '#c3c2b7', '#d03b3b'


def _heat_px(d):
    """(col, row) of a normalized detection on the 1024x512 heatmap.

    Example:
        >>> _heat_px((0.5, 0.25, 0.9))
        (512, 128)
    """
    return round(d[0] * HEATMAP_W), round(d[1] * HEATMAP_H)


def _crop(a, cx, cy, half=EXAMPLE_HALF):
    return a[cy - half:cy + half, cx - half:cx + half]


def _resample(img, scale=RESAMPLE_SCALE, quality=90):
    """The same pano re-encoded at another resolution: what a store JPEG differs by (#56)."""
    import io
    from PIL import Image
    w, h = img.size
    small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def cmd_examples(args):
    """GPU: find a clean detection and a flip on local bundle panos; cache crops for figures.

    Reads panos in sorted-id order (deterministic), stops at the first pano that yields each
    example or at --max-panos. A flip is a >= 0.55 detection whose nearest >= 0.55 detection on
    the resampled pano sits FLIP_MIN..FLIP_MAX heatmap px away (Chebyshev).
    """
    import numpy as np
    from PIL import Image
    from detectors.curb_ramp import CurbRampDetector, detections_from_heatmap
    bundle = args.benchmark_root / args.city
    records = {}
    with open(bundle / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r['pano']['panorama_id']] = r['pano']
    det = CurbRampDetector()
    edge = EXAMPLE_HALF + 2

    def inside(c, r):
        return edge <= c < HEATMAP_W - edge and edge <= r < HEATMAP_H - edge

    def cheb(d, c, r):
        dc, dr = _heat_px(d)
        return max(abs(dc - c), abs(dr - r))

    clean = flip = None
    for n, pano_id in enumerate(sorted(records)):
        path = bundle / 'panos' / f'{pano_id}.jpg'
        if not path.exists():
            continue
        img = Image.open(path).convert('RGB')
        h0 = det.heatmap(img)
        d0 = [d for d in detections_from_heatmap(h0) if d[2] >= BENCHMARK_CONFIDENCE]
        if clean is None:
            for d in d0:
                if inside(*_heat_px(d)):
                    clean = (pano_id, img, h0, d)
                    break
        if flip is None and d0:
            h1 = det.heatmap(_resample(img))
            d1 = [d for d in detections_from_heatmap(h1) if d[2] >= BENCHMARK_CONFIDENCE]
            for a in d0:
                ca, ra = _heat_px(a)
                if not inside(ca, ra) or not d1:
                    continue
                b = min(d1, key=lambda d: cheb(d, ca, ra))
                if FLIP_MIN <= cheb(b, ca, ra) <= FLIP_MAX:
                    flip = (pano_id, h0, h1, a, b)
                    break
        print(f'{n + 1:>3} {pano_id}: clean={"yes" if clean else "-"} '
              f'flip={"yes" if flip else "-"}', flush=True)
        if (clean and flip) or n + 1 >= args.max_panos:
            break
    if clean is None:
        sys.exit('no clean example found')
    arrays = {}
    meta = {'city': args.city, 'model_id': det.provenance.get('model_id'),
            'resample': {'scale': RESAMPLE_SCALE, 'jpeg_quality': 90},
            'crop_half_heatmap_px': EXAMPLE_HALF}
    pano_id, img, h0, d = clean
    c, r = _heat_px(d)
    arrays['clean'] = _crop(h0, c, r).astype(np.float16)
    w, h = img.size
    sx, sy = w / HEATMAP_W, h / HEATMAP_H
    box = (round((c - EXAMPLE_HALF) * sx), round((r - EXAMPLE_HALF) * sy),
           round((c + EXAMPLE_HALF) * sx), round((r + EXAMPLE_HALF) * sy))
    (FIG_DIR / 'data').mkdir(parents=True, exist_ok=True)
    img.crop(box).resize((384, 384), Image.LANCZOS).save(
        FIG_DIR / 'data' / 'example_clean_rgb.jpg', quality=85)
    meta['clean'] = {'pano_id': pano_id, 'copyright': records[pano_id].get('copyright'),
                     'col': c, 'row': r, 'confidence': round(d[2], 4)}
    if flip:
        pano_id, h0, h1, a, b = flip
        ca, ra = _heat_px(a)
        cb, rb = _heat_px(b)
        arrays['flip_original'] = _crop(h0, ca, ra).astype(np.float16)
        arrays['flip_resampled'] = _crop(h1, ca, ra).astype(np.float16)
        meta['flip'] = {'pano_id': pano_id, 'copyright': records[pano_id].get('copyright'),
                        'original': {'col': ca, 'row': ra, 'confidence': round(a[2], 4)},
                        'resampled': {'col': cb, 'row': rb, 'confidence': round(b[2], 4)}}
    else:
        print('no flip within --max-panos; the figure shows the clean example only',
              file=sys.stderr)
    np.savez_compressed(FIG_DIR / 'data' / 'examples.npz', **arrays)
    with open(FIG_DIR / 'data' / 'examples.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')
    print(json.dumps(meta, indent=2))


def _style(ax, grid_axis='y'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelcolor=C_INK2, labelsize=8)
    ax.grid(False)
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fig_grid(plt):
    """Pooled residue histograms per axis, plus each run's on-grid share."""
    rows = [r for r in _read_csv(FIG_DIR / 'data' / 'grid.csv')
            if r['tier'] == 'stored' and r['city'] != 'vancouver_zoom3_control']
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8),
                             gridspec_kw={'width_ratios': [1, 1, 1.3]})
    total = sum(int(r['detections']) for r in rows)
    for ax, axis, label in ((axes[0], 'x', 'column'), (axes[1], 'y', 'row')):
        counts = [sum(int(r[f'{axis}_mod8_{k}']) for r in rows) for k in range(CELL)]
        colors = [C_BLUE if k in GRID_RESIDUES else C_MUTED for k in range(CELL)]
        ax.bar(range(CELL), [max(n, 0) for n in counts], color=colors, width=0.8)
        ax.set_yscale('log')
        ax.set_ylim(0.8, total * 4)
        for k, n in enumerate(counts):
            ax.text(k, max(n, 1) * 1.3, f'{n / 1000:.0f}k' if n >= 10000 else f'{n:,}',
                    ha='center', fontsize=7, color=C_INK2)
        ax.set_xticks(range(CELL))
        ax.set_xlabel(f'heatmap {label} mod 8', color=C_INK2, fontsize=9)
        ax.set_ylabel('detections (log scale)', color=C_INK2, fontsize=9)
        ax.set_title(f'Heatmap {label} residue, all runs pooled', fontsize=10, color=C_INK,
                     loc='left')
        _style(ax)
    ax = axes[2]
    rows.sort(key=lambda r: float(r['both_on_grid_share']))
    y = list(range(len(rows)))
    ax.scatter([100 * float(r['both_on_grid_share']) for r in rows], y, s=28, color=C_BLUE,
               zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['city'].replace('_', ' ')} ({int(r['detections']):,})"
                        for r in rows], fontsize=7)
    ax.set_xlim(99.7, 100.02)
    ax.set_xlabel('% on residue 3 or 4 in both axes', color=C_INK2, fontsize=9)
    ax.set_title('Per run (stored detections)', fontsize=10, color=C_INK, loc='left')
    _style(ax, grid_axis='x')
    fig.suptitle(f'RampNet detections land on 2 of every 8 heatmap columns and rows '
                 f'({total:,} detections, {len(rows)} runs)', fontsize=11, color=C_INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _edges(origin, n):
    """Crop coordinates of the absolute coarse-cell edges (8k - 0.5) in a crop of n px.

    Argmax residues 3/4 sit mid-cell, so a cell spans heatmap px 8k..8k+7.

    Example:
        >>> _edges(411, 20)      # crop starting at column 411 (residue 3)
        [4.5, 12.5]
    """
    first = (-origin) % CELL
    return [e - 0.5 for e in range(first, n + 1, CELL) if 0 < e < n]


def _heat_panel(ax, a, marks, title, origin):
    im = ax.imshow(a, cmap='viridis', vmin=0, vmax=max(1e-6, float(a.max())),
                   interpolation='nearest')
    for e in _edges(origin[0], a.shape[1]):
        ax.axvline(e, color='white', linewidth=0.5, alpha=0.6)
    for e in _edges(origin[1], a.shape[0]):
        ax.axhline(e, color='white', linewidth=0.5, alpha=0.6)
    for (c, r), color, marker, label in marks:
        ax.scatter([c], [r], s=80, facecolors='none', edgecolors=color, linewidths=2,
                   marker=marker, label=label, zorder=3)
    ax.set_xlim(-0.5, a.shape[1] - 0.5)
    ax.set_ylim(a.shape[0] - 0.5, -0.5)
    ax.set_title(title, fontsize=9, color=C_INK, loc='left')
    ax.set_xlabel('heatmap column in crop (px)', fontsize=8, color=C_INK2)
    ax.set_ylabel('heatmap row in crop (px)', fontsize=8, color=C_INK2)
    ax.tick_params(labelsize=7, colors=C_MUTED, labelcolor=C_INK2)
    return im


def fig_examples(plt):
    """A real heatmap crop at native 1-px resolution, and a coarse-cell flip if one was cached."""
    import numpy as np
    from PIL import Image
    meta = json.loads((FIG_DIR / 'data' / 'examples.json').read_text(encoding='utf-8'))
    arr = np.load(FIG_DIR / 'data' / 'examples.npz')
    has_flip = 'flip' in meta
    fig, axes = plt.subplots(2 if has_flip else 1, 3, figsize=(12, 8 if has_flip else 4.2),
                             squeeze=False)
    half = meta['crop_half_heatmap_px']
    cl = meta['clean']
    ax = axes[0][0]
    rgb = Image.open(FIG_DIR / 'data' / 'example_clean_rgb.jpg')
    ax.imshow(rgb)
    ctr = rgb.size[0] / 2
    ax.scatter([ctr], [ctr], s=140, facecolors='none', edgecolors='white', linewidths=2)
    ax.set_title(f"Pano crop around a detection (confidence {cl['confidence']:.2f})",
                 fontsize=9, color=C_INK, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{meta['city']} pano {cl['pano_id']}\n{cl['copyright']}", fontsize=7,
                  color=C_INK2)
    a = arr['clean'].astype(float)
    origin = (cl['col'] - half, cl['row'] - half)
    im = _heat_panel(axes[0][1], a, [((half, half), 'white', 'o', 'stored detection')],
                     'Its heatmap; white grid = 8-px coarse cells', origin)
    fig.colorbar(im, ax=axes[0][1], fraction=0.046, pad=0.03).set_label(
        'heatmap value', fontsize=8, color=C_INK2)
    ax = axes[0][2]
    ax.plot(range(a.shape[1]), a[half], color=C_BLUE, linewidth=2, label='row through detection')
    ax.plot(range(a.shape[0]), a[:, half], color=C_ORANGE, linewidth=2, linestyle='--',
            label='column through detection')
    # coarse samples sit at 8k + 3.5 (align_corners=False): the kinks of the profile
    for e in _edges(origin[0], a.shape[1]):
        ax.axvline(e + 4, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel('position in crop (heatmap px); grey = coarse samples', fontsize=8,
                  color=C_INK2)
    ax.set_ylabel('heatmap value', fontsize=8, color=C_INK2)
    ax.set_title('Profiles: linear between coarse samples', fontsize=9, color=C_INK,
                 loc='left')
    ax.legend(fontsize=7, frameon=False)
    _style(ax, grid_axis='y')
    if has_flip:
        fl = meta['flip']
        o, rs = fl['original'], fl['resampled']
        dc, dr = rs['col'] - o['col'], rs['row'] - o['row']
        a0, a1 = arr['flip_original'].astype(float), arr['flip_resampled'].astype(float)
        marks = [((half, half), 'white', 'o', f"original ({o['confidence']:.2f})"),
                 ((half + dc, half + dr), C_ORANGE, 's', f"resampled ({rs['confidence']:.2f})")]
        forigin = (o['col'] - half, o['row'] - half)
        _heat_panel(axes[1][0], a0, marks, 'Flip example: the original pano', forigin)
        axes[1][0].legend(fontsize=7, loc='lower left', framealpha=0.85)
        _heat_panel(axes[1][1], a1, marks,
                    f"Same pano resampled to {meta['resample']['scale']:g}x, "
                    f"JPEG q{meta['resample']['jpeg_quality']}", forigin)
        ax = axes[1][2]
        # sample both heatmaps along the line through the two argmaxes (nearest pixel)
        steps = max(abs(dc), abs(dr))
        ux, uy = dc / steps, dr / steps
        ts = [t for t in range(-2 * half, 2 * half + 1)
              if 0 <= round(half + t * ux) < a0.shape[1] and 0 <= round(half + t * uy) < a0.shape[0]]
        dist = math.hypot(ux, uy)

        def along(m):
            return [m[round(half + t * uy), round(half + t * ux)] for t in ts]
        xs = [t * dist for t in ts]
        ax.plot(xs, along(a0), color=C_BLUE, linewidth=2, label='original')
        ax.plot(xs, along(a1), color=C_ORANGE, linewidth=2, linestyle='--', label='resampled')
        ax.axvline(0, color=C_BLUE, linewidth=1, linestyle=':')
        ax.axvline(steps * dist, color=C_ORANGE, linewidth=1, linestyle=':')
        ax.set_xlabel('along the line through both argmaxes (heatmap px)', fontsize=8, color=C_INK2)
        ax.set_ylabel('heatmap value', fontsize=8, color=C_INK2)
        ax.set_title(f"Near-tie: argmax moves ({dc:+d}, {dr:+d}) px",
                     fontsize=9, color=C_INK, loc='left')
        ax.legend(fontsize=7, frameon=False)
        _style(ax, grid_axis='y')
        axes[1][0].set_xlabel(f"heatmap column in crop (px)\n{meta['city']} pano "
                              f"{fl['pano_id']}, {fl['copyright']}", fontsize=7, color=C_INK2)
    fig.suptitle('The heatmap is a bilinear 8x upsample: peaks sit on coarse samples, and '
                 'near-ties flip by one cell', fontsize=11, color=C_INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_sigma(plt):
    """Per-cell change in world recall (with the rule's 1-SE bound) and in rejections."""
    rows = _read_csv(FIG_DIR / 'data' / 'sigma_peak_verdict.csv')
    rows.sort(key=lambda r: (r['city'], r['height']))
    labels = [f"{r['city'].replace('_', ' ')}, {'2.6 m' if r['height'] == '2.6' else 'auto'}"
              for r in rows]
    y = list(range(len(rows)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    ax = axes[0]
    for yi, r in zip(y, rows):
        se = 100 * float(r['world_recall_se'])
        ax.fill_betweenx([yi - 0.32, yi + 0.32], -se, se, color=C_GRID, zorder=1)
        dv = 100 * float(r['world_recall_delta'])
        fail = r['pass'] != 'True'
        ax.scatter([dv], [yi], s=50, zorder=3, color=C_FAIL if fail else C_BLUE,
                   marker='X' if fail else 'o')
        if fail:
            ax.annotate(f'{dv:+.2f} pts, bound {se:.2f}: FAIL', (dv, yi), xytext=(8, 6),
                        textcoords='offset points', fontsize=7, color=C_FAIL)
    ax.axvline(0, color=C_AXIS, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-2.6, 2.6)
    ax.set_xlabel('change in world recall (percentage points)', fontsize=8, color=C_INK2)
    ax.set_title('World recall; grey band = allowed (1 binomial SE)\n'
                 'World precision: unchanged in every cell', fontsize=9, color=C_INK,
                 loc='left')
    _style(ax, grid_axis='x')
    ax = axes[1]
    for off, key, color, marker, name in (
            (0.14, 'chi2_gate_rejections_rise', C_BLUE, 'o', 'chi-square gate rejections'),
            (-0.14, 'residual_rejections_rise', C_ORANGE, 's', 'residual rejections')):
        ax.scatter([100 * float(r[key]) for r in rows], [yi + off for yi in y], s=36,
                   color=color, marker=marker, label=name, zorder=3)
    ax.axvline(100 * REJECTION_RISE_MAX, color=C_FAIL, linewidth=1, linestyle='--')
    ax.text(100 * REJECTION_RISE_MAX + 1, 0, 'rise allowed\nup to +10%', fontsize=7,
            color=C_FAIL, va='bottom')
    ax.axvline(0, color=C_AXIS, linewidth=0.8)
    ax.set_xlim(-65, 25)
    ax.set_xlabel('relative change in rejections (%)', fontsize=8, color=C_INK2)
    ax.set_title('Association rejections fall (Richmond: none either way)', fontsize=9,
                 color=C_INK, loc='left')
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    _style(ax, grid_axis='x')
    fig.suptitle('sigma_peak_px 1.0 -> 2.31 px: nine of ten cells inside the pre-registered '
                 'bound; Sao Paulo at 2.6 m fails on recall', fontsize=11, color=C_INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def cmd_figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    todo = [('grid_mod8.png', fig_grid), ('sigma_sweep.png', fig_sigma)]
    if (FIG_DIR / 'data' / 'examples.json').exists():
        todo.append(('heatmap_crop.png', fig_examples))
    else:
        print('no data/examples.json: run `examples` first (GPU) for heatmap_crop.png',
              file=sys.stderr)
    for name, fn in todo:
        fig = fn(plt)
        fig.savefig(FIG_DIR / name, dpi=args.dpi)
        plt.close(fig)
        print(f'wrote {FIG_DIR / name} ({(FIG_DIR / name).stat().st_size // 1024} KB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('grid', 'sigma'):
        p = sub.add_parser(name)
        p.add_argument('cities', nargs='+')
        p.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs')
        p.add_argument('--out', type=Path, default=OUT_DIR)
    sub.choices['sigma'].add_argument('--benchmark-root', type=Path,
                                      default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    sub.choices['sigma'].add_argument('--heights', nargs='+', default=['2.6', 'auto'])
    p = sub.add_parser('examples')
    p.add_argument('--city', default='richmond')
    p.add_argument('--benchmark-root', type=Path,
                   default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    p.add_argument('--max-panos', type=int, default=80)
    p = sub.add_parser('figures')
    p.add_argument('--dpi', type=int, default=100)
    args = ap.parse_args(argv)
    {'grid': cmd_grid, 'sigma': cmd_sigma, 'examples': cmd_examples,
     'figures': cmd_figures}[args.cmd](args)


if __name__ == '__main__':
    main()
