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
    args = ap.parse_args(argv)
    {'grid': cmd_grid, 'sigma': cmd_sigma}[args.cmd](args)


if __name__ == '__main__':
    main()
