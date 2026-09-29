"""POST HOC: why does step 3's peak rule drop GSV's true misses? (RampNet#158 step 3 review)

NOT part of the pre-registered step-3 design (RampNet#158, comment 5895808813). The rule,
the gate and every step-3 number stand as posted. This diagnostic was written after the
independent review of sidewalk-auto-labeler#112, which found that on GSV the rule's
"operational peak in window" exclusion mostly removes verified misses because a
reviewer-judged-TRUE detection of a NEIGHBOURING ramp sits inside the 0.022 window, not
because the model already fires on the mined ramp.

For every flat `tp` at <= --radius in the given cities (the step-2 flat arm,
`frozen/perrig_perpano`), it:

1. re-runs the exact step-2 fuse and adjudication (mined_precision's code path, the
   FuseParams mined_precision.main uses) and refuses to go on unless every recomputed
   bucket equals the committed candidates.csv;
2. re-identifies the MISSED mark that adjudicated the candidate (the reviewer's mark whose
   raycast point is the adjudicating GT point);
3. re-applies the step-3 rule (peak_anchor.anchor_peak) to the flat anchor with the
   pano's stored peaks, and checks it against the committed placement/peak_flat.jsonl;
4. for `operational_in_window`: lists the in-window >= 0.55 peaks, their reviewer verdict
   (benchmark verdicts.json, matched by position in the operational list, as build_gt
   does), their angular distance to the missed mark, and their offset from it in heatmap
   cells (the 512x1024 grid). `peak_local_max(min_distance=10)` keeps a pixel only if it is
   the maximum of the 21x21 cells around it, so a missed mark within 10 cells (Chebyshev,
   about 3.5 deg) of a stronger peak CANNOT hold a stored peak of its own;
5. for every row: the stored peaks (floor 0.1) within the window of the MISSED MARK, and
   the nearest stored peak to it.

--heatmap DIR adds the part the stored peaks cannot answer: is there a local maximum at
the missed mark BEFORE min_distance suppression? It re-fetches each target pano at zoom 3
through the production path (panorama.fetch_panorama: the pixels the production run fed
the model) and runs the published model (CPU is fine, ~35 s a pano). Each pano is first
checked the way the step-3 gate checks a floor pass (floor_infer_archive.reproduces
against the stored >= 0.55 detections), and its full stored peak set is compared, so the
heatmap is shown to be the production one before it is read. Then peaks are re-extracted
with min_distance=1 (every local maximum >= 0.1), and the nearest one to the missed mark
is reported with the heatmap's max within 2 cells of the mark. Images and heatmaps are
cached in DIR (gitignored scratch, not an input); the model revision and torch /
transformers versions are written into the output. Network: Google's unofficial GSV
endpoints (the same ones the production run used).

Usage (frozen inputs as in docs/mined-precision.md, step 2 "Reproduce"; RampNet
benchmark at 4a859f1):
    FROZEN=/path/to/frozen
    D=docs/figures/mined-precision/data
    python scripts/step3_exclusion_diag.py paterson gainesville sao_paulo \\
        --runs-root $FROZEN --benchmark-root ../RampNet/benchmark \\
        --camera-height per-pano per-pano per-pano --verify $D/frozen \\
        --peak-flat $D/placement/peak_flat.jsonl --heatmap /scratch/step3_heatmaps \\
        --out $D/step3/posthoc_exclusion.csv --md $D/step3/posthoc_exclusion.md
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import mined_precision as mp  # noqa: E402
import mined_placement_attribution as mpa  # noqa: E402
import peak_anchor as pa  # noqa: E402
import floor_infer_archive as fia  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR  # noqa: E402

GRID_W, GRID_H = 1024, 512          # the model's heatmap, = RampNet's pano match geometry
SUPPRESS_CELLS = 10                 # detectors.curb_ramp: peak_local_max(min_distance=10)
LOCAL_CELLS = 2                     # heatmap max within this many cells of the mark

FIELDS = ('city', 'site_id', 'pano_id', 'range_m', 'status', 'mark_x', 'mark_y',
          'n_op_in_window', 'op_conf', 'op_verdict', 'op_to_mark_deg', 'op_to_mark_cells',
          'suppressed_by_construction', 'stored_sub_near_mark', 'nearest_stored_deg',
          'nearest_stored_conf', 'hm_reproduces', 'hm_same_peak_set', 'hm_local_max_deg',
          'hm_local_max_conf', 'hm_local_max_is_stored', 'hm_max_near_mark')


def degrees(ax, ay, bx, by):
    """Angular distance on the equirect in RampNet's pano geometry (0.022 = 7.92 deg)."""
    return pa.pano_distance(ax, ay, bx, by) * 360.0


def cells(ax, ay, bx, by):
    """Chebyshev offset in heatmap cells, x cyclic at the seam."""
    dx = abs(ax - bx) % 1.0
    dx = min(dx, 1.0 - dx)
    return max(dx * GRID_W, abs(ay - by) * GRID_H)


def adjudicating_mark(entry, run_pano, params, frame, site, dist):
    """The non-unsure missed mark whose raycast point adjudicated the candidate: the one
    at exactly `dist` from the site (build_gt places marks the same way)."""
    pose = fs.pano_pose(run_pano, params.apply_pose)
    hits = []
    for m in entry.get('missed', ()):
        if m.get('unsure'):
            continue
        enu = mp.raycast_pixel(pose, (m['x'], m['y']), run_pano, params, frame)
        if enu is not None and abs(math.hypot(enu[0] - site.e, enu[1] - site.n) - dist) < 1e-6:
            hits.append(m)
    if len(hits) != 1:
        raise AssertionError(f'{run_pano.pano_id}: {len(hits)} missed marks at {dist} m')
    return hits[0]['x'], hits[0]['y']


def diagnose_city(city, verdict_panos, bundle_ops, run_panos, params, radius_m, match_m,
                  peaks_all, peak_flat_rows, verify_root=None):
    sites, frame, _ = fs.fuse(run_panos, params)
    by_site = {s.id: s for s in sites}
    run_by_id = {p.pano_id: p for p in run_panos}
    judged = mp.judged_panos(verdict_panos, bundle_ops, run_by_id)
    gt_by_pano, _ = mp.gt_points_by_pano(verdict_panos, bundle_ops, run_by_id, params, frame)
    strong = mp.strong_sites(sites, 3)
    flat, _ = mp.mine_candidates(strong, judged, run_by_id, gt_by_pano, frame, params,
                                 radius_m, match_m, city)
    if verify_root is not None:
        mpa._verify(Path(verify_root) / 'perrig_perpano' / city / 'candidates.csv', flat)
    rows = []
    for c in flat:
        if c.bucket != 'tp':
            continue
        key = (city, c.site_id, c.pano_id)
        peaks = peaks_all[c.pano_id]
        status, _p = pa.anchor_peak((c.x_norm, c.y_norm), peaks)
        committed = peak_flat_rows[key]['status']
        if status != committed:
            raise AssertionError(f'{key}: rule gives {status}, peak_flat.jsonl {committed}')
        mx, my = adjudicating_mark(judged[c.pano_id], run_by_id[c.pano_id], params, frame,
                                   by_site[c.site_id], c.nearest_gt_m)
        ops = [(x, y) for _i, x, y, conf in run_by_id[c.pano_id].detections
               if conf >= BENCHMARK_CONFIDENCE]
        verdicts = judged[c.pano_id]['dets']
        in_win = [p for p in peaks if p[2] >= BENCHMARK_CONFIDENCE
                  and pa.pano_distance(c.x_norm, c.y_norm, p[0], p[1]) <= pa.WINDOW]
        row = {'city': city, 'site_id': c.site_id, 'pano_id': c.pano_id,
               'range_m': c.range_m, 'status': status, 'mark_x': mx, 'mark_y': my,
               'n_op_in_window': len(in_win)}
        if in_win:
            # the in-window operational peak nearest the missed mark
            p = min(in_win, key=lambda q: degrees(q[0], q[1], mx, my))
            k = ops.index((p[0], p[1]))
            row.update({'op_conf': p[2], 'op_verdict': verdicts[k],
                        'op_to_mark_deg': degrees(p[0], p[1], mx, my),
                        'op_to_mark_cells': cells(p[0], p[1], mx, my),
                        'suppressed_by_construction':
                            int(cells(p[0], p[1], mx, my) <= SUPPRESS_CELLS)})
        sub = [q for q in peaks if DETECTION_STORAGE_FLOOR <= q[2] < BENCHMARK_CONFIDENCE
               and pa.pano_distance(mx, my, q[0], q[1]) <= pa.WINDOW]
        row['stored_sub_near_mark'] = len(sub)
        if peaks:
            q = min(peaks, key=lambda q: degrees(q[0], q[1], mx, my))
            row['nearest_stored_deg'] = degrees(q[0], q[1], mx, my)
            row['nearest_stored_conf'] = q[2]
        rows.append(row)
    return rows


def software_versions():
    import torch
    import transformers
    return {'torch': torch.__version__, 'transformers': transformers.__version__}


def add_heatmap(rows, records, cache):
    """Fill the hm_* columns (see the module docstring). Returns provenance."""
    import numpy as np
    from skimage.feature import peak_local_max
    from streetlevel import streetview
    import panorama
    from detectors.curb_ramp import CurbRampDetector, detections_from_heatmap
    cache.mkdir(parents=True, exist_ok=True)
    det = None
    for r in rows:
        pid = r['pano_id']
        hm_path = cache / f'{pid}.z3.npy'
        if not hm_path.exists():
            if det is None:
                det = CurbRampDetector()
            meta = streetview.find_panorama_by_id(pid)
            img = None if meta is None else panorama.fetch_panorama(meta)
            if img is None:
                raise SystemExit(f'{pid}: could not re-fetch at zoom 3')
            np.save(hm_path, det.heatmap(img))
        hm = np.load(hm_path)
        new = [{'x_normalized': x, 'y_normalized': y, 'confidence': conf}
               for x, y, conf in detections_from_heatmap(hm)]
        old = records[pid]['detections']
        r['hm_reproduces'] = int(fia.reproduces(old, new))
        # the FULL stored set (floor 0.1): same count, each within one cell
        r['hm_same_peak_set'] = int(len(old) == len(new) and fia.reproduces(
            [dict(d, confidence=1.0) for d in old], [dict(d, confidence=1.0) for d in new]))
        mx, my = r['mark_x'], r['mark_y']
        allmax = peak_local_max(np.clip(hm, 0, 1), min_distance=1,
                                threshold_abs=DETECTION_STORAGE_FLOOR)
        pts = [(cc / GRID_W, rr / GRID_H, float(hm[rr][cc])) for rr, cc in allmax]
        if pts:
            q = min(pts, key=lambda q: degrees(q[0], q[1], mx, my))
            r['hm_local_max_deg'] = degrees(q[0], q[1], mx, my)
            r['hm_local_max_conf'] = q[2]
            # is the nearest local maximum one the production extraction (min_distance=10)
            # also keeps? If so, dropping the suppression reveals nothing new near the mark.
            r['hm_local_max_is_stored'] = int(any(
                cells(q[0], q[1], d['x_normalized'], d['y_normalized']) < 0.5 for d in new))
        ci, ri = int(round(mx * GRID_W)) % GRID_W, int(round(my * GRID_H))
        cols = [(ci + d) % GRID_W for d in range(-LOCAL_CELLS, LOCAL_CELLS + 1)]
        rws = [min(max(ri + d, 0), GRID_H - 1) for d in range(-LOCAL_CELLS, LOCAL_CELLS + 1)]
        r['hm_max_near_mark'] = float(hm[np.ix_(rws, cols)].max())
    if det is None:                   # every heatmap was cached: load for provenance only
        det = CurbRampDetector()
    prov = {'imagery': 'zoom 3 via panorama.fetch_panorama (the production path)',
            'model': det.provenance}
    try:
        prov.update(software_versions())
    except ImportError:
        pass
    return prov


def summarise(rows, radius_m, mapillary=()):
    lines = ['POST HOC diagnostic (not pre-registered); see scripts/step3_exclusion_diag.py.',
             f'Every flat `tp` at <= {radius_m:g} m, by the `peak_flat` rule\'s status.', '',
             '| imagery | status | n | in-window op. detection judged true | ... within 10 cells '
             '(suppressed by construction) | op. to missed mark, deg (min-max) '
             '| stored sub-threshold peak near the missed mark |',
             '|---|---|--:|--:|--:|---|--:|']
    groups = [('GSV', [r for r in rows if r['city'] not in set(mapillary)]),
              ('Mapillary', [r for r in rows if r['city'] in set(mapillary)])]
    for (g, grows), s in ((g, s) for g in groups
                          for s in ('operational_in_window', 'no_peak', 'emit')):
        sub = [r for r in grows if r['status'] == s]
        if not sub:
            continue
        op = [r for r in sub if r.get('op_verdict') is not None]
        degs = [r['op_to_mark_deg'] for r in op]
        rng = f'{min(degs):.1f}-{max(degs):.1f}' if degs else '-'
        lines.append(f"| {g} | {s} | {len(sub)} | {sum(1 for r in op if r['op_verdict'] is True)}"
                     f" of {len(op)} | {sum(r['suppressed_by_construction'] for r in op)} "
                     f"| {rng} | {sum(1 for r in sub if r['stored_sub_near_mark'])} |")
    hm = [r for r in rows if r.get('hm_reproduces') is not None]
    if hm:
        panos = {r['pano_id']: r for r in hm}
        lines += ['', f"Zoom-3 re-inference over the {len(panos)} distinct target panos of "
                      f"these {len(hm)} candidates: "
                      f"{sum(r['hm_reproduces'] for r in panos.values())} reproduce their >= 0.55 "
                      f"detections and {sum(r['hm_same_peak_set'] for r in panos.values())} their "
                      "full stored peak set (floor 0.1, one-cell tolerance).",
                  '', '| status | n | nearest local max to the missed mark with NO suppression '
                      '(min_distance=1) is a stored peak | ... and is the in-window op. detection '
                      '| heatmap max within 2 cells of the mark: median; n >= 0.55 |',
                  '|---|--:|--:|--:|---|']
        for s in ('operational_in_window', 'no_peak', 'emit'):
            sub = [r for r in hm if r['status'] == s]
            if not sub:
                continue
            stored = sum(1 for r in sub if r.get('hm_local_max_is_stored'))
            is_op = sum(1 for r in sub if r.get('op_to_mark_deg') is not None
                        and abs(r['hm_local_max_deg'] - r['op_to_mark_deg']) < 0.05)
            vals = sorted(r['hm_max_near_mark'] for r in sub)
            med = vals[len(vals) // 2] if len(vals) % 2 else \
                (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
            lines.append(f'| {s} | {len(sub)} | {stored} | {is_op} | {med:.3f}; '
                         f'{sum(v >= BENCHMARK_CONFIDENCE for v in vals)} |')
    return '\n'.join(lines)


def _fmt(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f'{v:.4f}'
    return '' if v is None else v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', nargs='+')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--runs-root', type=Path, required=True)
    ap.add_argument('--camera-height', type=fs.fuse_camera_height_arg, nargs='+',
                    required=True)
    ap.add_argument('--radius', type=float, default=15.0)
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--verify', type=Path, default=None)
    ap.add_argument('--peak-flat', type=Path, required=True)
    ap.add_argument('--peaks', nargs='*', default=[],
                    help="city=PATH: read that city's peaks from a floor pass (richmond)")
    ap.add_argument('--mapillary', nargs='*', default=(),
                    help='cities whose imagery is not GSV: --heatmap skips them')
    ap.add_argument('--heatmap', type=Path, default=None,
                    help='cache dir: re-fetch at zoom 3 and re-infer (network + model)')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--md', type=Path, default=None)
    args = ap.parse_args()
    mpa.require_verify((args.out, args.md), args.verify, mpa.FROZEN_DIR.parent)
    heights = args.camera_height
    if len(heights) not in (1, len(args.city)):
        ap.error('--camera-height takes one value or one per city')
    peak_flat = {}
    with open(args.peak_flat, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                peak_flat[(r['city'], int(r['site_id']), str(r['pano_id']))] = r
    rows, records = [], {}
    for i, city in enumerate(args.city):
        mode = heights[i] if len(heights) > 1 else heights[0]
        verdict_panos, bundle_ops, run_panos, height, _a = es.load_city_at_height(
            city, args.benchmark_root, args.runs_root / city, mode, read_heights=True)
        params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                               apply_pose=fs.POSE_OFF, camera_height_m=height)
        want = set(verdict_panos)
        path = dict(kv.split('=', 1) for kv in args.peaks).get(
            city, args.runs_root / city / 'results.jsonl')
        peaks = pa.read_peaks(path, want)
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if str(rec['pano']['panorama_id']) in want:
                        records[str(rec['pano']['panorama_id'])] = rec
        rows += diagnose_city(city, verdict_panos, bundle_ops, run_panos, params,
                              args.radius, args.match_radius_m, peaks, peak_flat,
                              args.verify)
    prov = None
    if args.heatmap:
        prov = add_heatmap([r for r in rows if r['city'] not in set(args.mapillary)],
                           records, args.heatmap)
    md = summarise(rows, args.radius, args.mapillary)
    if prov:
        md += '\n\nProvenance: ' + json.dumps(prov, sort_keys=True)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, FIELDS, lineterminator='\n')
            w.writeheader()
            for r in rows:
                w.writerow({k: _fmt(r.get(k)) for k in FIELDS})
    if args.md:
        args.md.write_text(md + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
