"""The peak-anchored miner over a WHOLE run (RampNet#158 step 4).

Step 3 measured the ``peak_flat`` rule on benchmark candidates only. This applies the same
rule, unchanged, to every (strong site, non-member pano within R) pair of a run, and writes
the labels a miner would emit. It reads no verdict.

    fuse       the pinned run exactly as mined_precision.py does (benchmark threshold,
               pose off, the given camera-height frame): fs.fuse over results.jsonl
    sites      every site with >= 3 operational panos (mined_precision.strong_sites)
    pairs      every run pano within R (15 m) of a site that is not one of its members
               (a pano that is a member only through a sub-threshold detection is skipped
               and counted, as mined_precision does)
    anchor     the flat projection of the site into the pano (geo.ground_point_to_pano)
    rule       peak_anchor.anchor_peak over that pano's stored peaks (a floor pass):
               a >= 0.55 peak in the 0.022 window -> not emitted; else the strongest
               [0.1, 0.55) peak -> emitted at its pixel; else not emitted

Writes ``<out>/labels.jsonl`` (one row per pair: status, anchor, emitted pixel and peak
confidence, range, band, whether the target pano is a benchmark pano, the source-rule
member for context) and ``<out>/summary.json`` (counts by status and by band).

Usage:
    python scripts/mine_city.py --results FROZEN/richmond/results.jsonl \\
        --peaks richmond.all.floor.jsonl --camera-height per-rig --city richmond \\
        --benchmark-ids docs/figures/mined-precision/data/step3/richmond_ids.txt \\
        --out docs/figures/mined-precision/data/step4/richmond
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import mined_precision as mp  # noqa: E402
import peak_anchor as pa  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

BANDS = ((0.0, 8.0, '0-8 m'), (8.0, 12.0, '8-12 m'), (12.0, 15.0, '12-15 m'))
NL = '\n'


def band_of(range_m):
    for lo, hi, name in BANDS:
        if lo < range_m <= hi:
            return name
    return f'> {BANDS[-1][1]:g} m' if range_m > BANDS[-1][1] else '0 m'


def mine(run_panos, params, peaks, radius_m=15.0, min_panos=3, city='',
         benchmark_ids=frozenset()):
    """(rows, stats). ``peaks`` = {pano_id: [(x, y, conf)]} for every run pano."""
    sites, frame, fuse_stats = fs.fuse(run_panos, params)
    strong = mp.strong_sites(sites, min_panos)
    run_by_id = {p.pano_id: p for p in run_panos}
    grid = geo.GridIndex(radius_m)
    enu = {}
    for p in run_panos:
        enu[p.pano_id] = frame.to_enu(p.lat, p.lng)
        grid.add(*enu[p.pano_id], p)
    rows, subfloor = [], 0
    for site, n_op, best_conf in strong:
        site_lat, site_lng = frame.to_latlng(site.e, site.n)
        op_panos = {d.pano_id for d, _ in site.members if d.operational}
        for p in grid.near(site.e, site.n):
            pid = p.pano_id
            pe, pn = enu[pid]
            if math.hypot(pe - site.e, pn - site.n) > radius_m:
                continue
            if pid in site.pano_ids:
                if pid not in op_panos:
                    subfloor += 1
                continue
            pose, _ = mp._pose_and_errors(p, params)
            proj = geo.ground_point_to_pano(
                pose, site_lat, site_lng, camera_height=params.camera_height_m,
                max_range_m=radius_m, apply_pose=params.rotates)
            if proj is None:
                continue
            if pid not in peaks:
                raise SystemExit(f'no floor peaks for run pano {pid}')
            status, pk = pa.anchor_peak((proj.x_norm, proj.y_norm), peaks[pid])
            src = mp.choose_source(site, pid, run_by_id, frame)
            rows.append({
                'city': city, 'site_id': site.id, 'pano_id': pid, 'status': status,
                'emit': status == 'emit', 'range_m': round(proj.range_m, 3),
                'band': band_of(proj.range_m), 'anchor_x': round(proj.x_norm, 6),
                'anchor_y': round(proj.y_norm, 6),
                'x': None if pk is None else pk[0], 'y': None if pk is None else pk[1],
                'peak_conf': None if pk is None else round(pk[2], 6),
                'n_op_panos': n_op, 'benchmark_pano': pid in benchmark_ids,
                'capture_date': p.capture_date,
                'src_pano': None if src is None else src[0].pano_id,
                'src_x': None if src is None else round(src[0].x, 6),
                'src_y': None if src is None else round(src[0].y, 6),
                'src_capture_date': (None if src is None
                                     else run_by_id[src[0].pano_id].capture_date)})
    rows.sort(key=lambda r: (r['site_id'], r['pano_id']))
    keys = [(r['site_id'], r['pano_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError('duplicate (site_id, pano_id) rows')
    em = [r for r in rows if r['emit']]
    stats = {
        'city': city, 'n_panos': fuse_stats['n_panos'], 'n_sites': fuse_stats['n_sites'],
        'n_strong_sites': len(strong), 'radius_m': radius_m, 'min_panos': min_panos,
        'pairs': len(rows), 'excluded_subfloor_members': subfloor,
        'by_status': dict(Counter(r['status'] for r in rows)),
        'emitted': len(em),
        'emitted_by_band': {name: sum(1 for r in em if r['band'] == name)
                            for _, _, name in BANDS},
        'pairs_by_band': {name: sum(1 for r in rows if r['band'] == name)
                          for _, _, name in BANDS},
        'emitted_on_benchmark_panos': sum(1 for r in em if r['benchmark_pano']),
        'emitted_off_benchmark': sum(1 for r in em if not r['benchmark_pano']),
        'emitted_sites': len({r['site_id'] for r in em}),
        'emitted_panos': len({r['pano_id'] for r in em}),
        'operational_detections': sum(1 for p in run_panos for _, _, _, c in p.detections
                                      if c >= BENCHMARK_CONFIDENCE)}
    return rows, stats


def read_peaks(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r['pano']['panorama_id'])] = [
                    (d['x_normalized'], d['y_normalized'], d['confidence'])
                    for d in r['detections']]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument('--results', type=Path, required=True, help='the pinned results.jsonl')
    ap.add_argument('--peaks', type=Path, required=True, help='a whole-run floor pass')
    ap.add_argument('--camera-height', type=fs.fuse_camera_height_arg, default='per-rig')
    ap.add_argument('--city', required=True)
    ap.add_argument('--radius', type=float, default=15.0)
    ap.add_argument('--benchmark-ids', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    run_panos, _skipped, height, _auto = fs.load_at_height(
        args.results, args.camera_height, read_heights=True)
    params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                           apply_pose=fs.POSE_OFF, camera_height_m=height)
    bench = set()
    if args.benchmark_ids:
        bench = {line.strip() for line in open(args.benchmark_ids, encoding='utf-8')
                 if line.strip()}
    rows, stats = mine(run_panos, params, read_peaks(args.peaks), radius_m=args.radius,
                       city=args.city, benchmark_ids=bench)
    stats['camera_height'] = str(args.camera_height)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / 'labels.jsonl', 'w', encoding='utf-8', newline=NL) as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + NL)
    (args.out / 'summary.json').write_text(json.dumps(stats, indent=1, sort_keys=True) + NL,
                                           encoding='utf-8', newline=NL)
    print(json.dumps(stats, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
