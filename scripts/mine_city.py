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
    rig mask   an emitted peak on the camera rig (detectors.on_camera_rig, 49 deg of dip)
               is withheld as status `on_rig`, as send_to_ps.transform_record drops it
               (0 Richmond rows: the deepest emitted peak sits at y 0.771484, the mask
               starts above 0.772222)

TIERS. "Operational" in this script and in peak_anchor is the BENCHMARK tier, 0.55
(detectors.BENCHMARK_CONFIDENCE), the tier step 3 was measured at: status
`operational_in_window`, the `n_op_panos` column and summary.json's
`operational_detections` all count >= 0.55. It is not the production operating point,
detectors.OPERATIONAL_CONFIDENCE = 0.30. So an emitted peak can sit in [0.30, 0.55):
production already ships a label at that pixel, and only the benchmark tier calls it a
miss. summary.json splits the emitted counts at 0.30 (`emitted_ge_0_30*`).

TARGETS vs ROWS. A row is a (site, pano) pair. Two sites can resolve to the same stored
peak in one pano, so emitted rows overcount distinct training targets (pano, x, y);
`emitted_distinct_targets*` counts those.

Writes ``<out>/labels.jsonl`` (one row per pair: status, anchor, emitted pixel and peak
confidence, range, band, whether the target pano is a benchmark pano, the source-rule
member for context) and ``<out>/summary.json`` (counts by status, band, tier and
distinct target; the inputs' sha256; the pinned parameters; the camera height each pano
resolved to, in metres). summary.json files written before the 2026-11-03 review of
labeler#114 carry only the counts by status and band.

Usage:
    python scripts/mine_city.py --results FROZEN/richmond/results.jsonl \\
        --peaks richmond.all.floor.jsonl --camera-height per-rig --city richmond \\
        --benchmark-ids docs/figures/mined-precision/data/step3/richmond_ids.txt \\
        --out docs/figures/mined-precision/data/step4/richmond
    # offline, from a committed labels.jsonl: the row counts alone (tiers, distinct
    # targets), to tally.json beside it
    python scripts/mine_city.py --tally docs/figures/mined-precision/data/step4/richmond/labels.jsonl
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import mined_precision as mp  # noqa: E402
import peak_anchor as pa  # noqa: E402
from detectors import (BENCHMARK_CONFIDENCE, NADIR_MASK_Y,  # noqa: E402
                       OPERATIONAL_CONFIDENCE, on_camera_rig)

BANDS = ((0.0, 8.0, '0-8 m'), (8.0, 12.0, '8-12 m'), (12.0, 15.0, '12-15 m'))
NL = '\n'


def band_of(range_m):
    for lo, hi, name in BANDS:
        if lo < range_m <= hi:
            return name
    return f'> {BANDS[-1][1]:g} m' if range_m > BANDS[-1][1] else '0 m'


def cli_params(height):
    """The FuseParams the CLI mines at: step 3's frame, pinned. Benchmark tier (a bare
    FuseParams() would follow OPERATIONAL_CONFIDENCE down to 0.30), rig mask and pose
    off in fusion, as mined_precision.py fuses; ``height`` is load_at_height's value.

        >>> p = cli_params(2.6)
        >>> p.min_confidence, p.mask_rig, p.apply_pose, p.camera_height_m
        (0.55, False, 'off', 2.6)
    """
    return fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                         apply_pose=fs.POSE_OFF, camera_height_m=height)


def target_status(anchor, peaks):
    """peak_anchor.anchor_peak, then the camera-rig mask on what it would emit: a peak
    past 49 deg of dip is on the vehicle, never a ramp (detectors.on_camera_rig, the mask
    send_to_ps.transform_record applies), so it is withheld as ``on_rig``. The peak stays
    in the row for context; only ``emit`` rows are training targets.

        >>> target_status((0.5, 0.6), [(0.505, 0.6, 0.3)])
        ('emit', (0.505, 0.6, 0.3))
        >>> target_status((0.5, 0.77), [(0.5, 0.775, 0.3)])
        ('on_rig', (0.5, 0.775, 0.3))
    """
    status, pk = pa.anchor_peak(anchor, peaks)
    if status == 'emit' and on_camera_rig(pk[1]):
        return 'on_rig', pk
    return status, pk


def mine(run_panos, params, peaks, radius_m=15.0, min_panos=3, city='',
         benchmark_ids=frozenset()):
    """(rows, stats). ``peaks`` = {pano_id: [(x, y, conf)]} for every run pano.
    ``params`` should be cli_params(height): the CLI's pinned frame."""
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
            status, pk = target_status((proj.x_norm, proj.y_norm), peaks[pid])
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
    stats = {
        'city': city, 'n_panos': fuse_stats['n_panos'], 'n_sites': fuse_stats['n_sites'],
        'n_strong_sites': len(strong), 'radius_m': radius_m, 'min_panos': min_panos,
        'excluded_subfloor_members': subfloor,
        **tally(rows),
        'operational_detections': sum(1 for p in run_panos for _, _, _, c in p.detections
                                      if c >= BENCHMARK_CONFIDENCE)}
    return rows, stats


def _target(r):
    return (r['pano_id'], r['x'], r['y'])


def _ge_production(rows):
    return sum(1 for r in rows if r['peak_conf'] >= OPERATIONAL_CONFIDENCE)


def _multi(by_target, key, off=False):
    """Targets whose emitted rows take more than one value of ``key`` (benchmark_pano is
    a property of the pano, so a target is off-benchmark in all its rows or none)."""
    return sum(1 for v in by_target.values()
               if len({r[key] for r in v}) > 1 and (not off or not v[0]['benchmark_pano']))


def tally(rows):
    """Every summary count that the rows alone determine (no fusion, no peaks), so a
    committed labels.jsonl re-tallies offline (--tally) to the same numbers.

    Tiers: ``emitted_ge_0_30*`` counts emitted rows whose peak is >= the production
    OPERATIONAL_CONFIDENCE (0.30), i.e. pixels where production already ships a label;
    "operational" elsewhere in this script is the benchmark tier, 0.55 (``tiers``).
    Targets: ``emitted_distinct_targets*`` counts distinct (pano, x, y) among emitted
    rows (two sites can resolve to one peak); ``emitted_targets_multi_site`` = targets
    emitted for more than one site, ``emitted_targets_multi_band`` = targets whose rows
    fall in more than one range band (the per-band distinct counts include each of those
    once per band).

        >>> r = dict(emit=True, benchmark_pano=False, pano_id='p', x=0.5, y=0.6, site_id=1,
        ...          band='0-8 m', status='emit', peak_conf=0.4)
        >>> t = tally([r, dict(r, site_id=2, band='8-12 m', peak_conf=0.2)])
        >>> t['emitted'], t['emitted_distinct_targets'], t['emitted_ge_0_30']
        (2, 1, 1)
        >>> t['emitted_targets_multi_site'], t['emitted_targets_multi_band']
        (1, 1)
    """
    em = [r for r in rows if r['emit']]
    off = [r for r in em if not r['benchmark_pano']]
    by_target = defaultdict(list)
    for r in em:
        by_target[_target(r)].append(r)
    names = [name for _, _, name in BANDS]
    in_band = {n: [r for r in em if r['band'] == n] for n in names}
    return {
        'pairs': len(rows),
        'by_status': dict(Counter(r['status'] for r in rows)),
        'emitted': len(em),
        'emitted_by_band': {n: len(in_band[n]) for n in names},
        'pairs_by_band': {n: sum(1 for r in rows if r['band'] == n) for n in names},
        'emitted_on_benchmark_panos': len(em) - len(off),
        'emitted_off_benchmark': len(off),
        'emitted_sites': len({r['site_id'] for r in em}),
        'emitted_panos': len({r['pano_id'] for r in em}),
        'emitted_ge_0_30': _ge_production(em),
        'emitted_ge_0_30_by_band': {n: _ge_production(in_band[n]) for n in names},
        'emitted_ge_0_30_off_benchmark': _ge_production(off),
        'emitted_ge_0_30_off_benchmark_by_band': {
            n: _ge_production([r for r in off if r['band'] == n]) for n in names},
        'emitted_off_benchmark_by_band': {
            n: sum(1 for r in off if r['band'] == n) for n in names},
        'emitted_distinct_targets': len(by_target),
        'emitted_distinct_targets_by_band': {
            n: len({_target(r) for r in in_band[n]}) for n in names},
        'emitted_distinct_targets_off_benchmark': len({_target(r) for r in off}),
        'emitted_distinct_targets_off_benchmark_by_band': {
            n: len({_target(r) for r in off if r['band'] == n}) for n in names},
        'emitted_targets_multi_site': _multi(by_target, 'site_id'),
        'emitted_targets_multi_site_off_benchmark': _multi(by_target, 'site_id', off=True),
        'emitted_targets_multi_band': _multi(by_target, 'band'),
        'emitted_targets_multi_band_off_benchmark': _multi(by_target, 'band', off=True),
        'tiers': {'operational_in_this_script': BENCHMARK_CONFIDENCE,
                  'production_operational': OPERATIONAL_CONFIDENCE}}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolved_heights_m(run_panos, params):
    """{metres: panos}: the camera height each pano is raycast at under ``params``
    (geo.camera_height_for, the resolution ground_point_to_pano applies), so a summary
    says the frame in metres, not only the mode name (Richmond per-rig: {'2.6': 9091})."""
    out = Counter()
    for p in run_panos:
        pose, errors = mp._pose_and_errors(p, params)
        h, _ = geo.camera_height_for(pose, errors, camera_height=params.camera_height_m)
        out[f'{h:g}'] += 1
    return dict(sorted(out.items()))


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


def read_labels(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_tally(labels, out):
    """--tally: tally() over a labels.jsonl, written to ``out`` with the file's sha256."""
    res = {'labels': Path(labels).as_posix(), 'labels_sha256': sha256_file(labels),
           **tally(read_labels(labels))}
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + NL,
                         encoding='utf-8', newline=NL)
    print(json.dumps(res, indent=1, sort_keys=True))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument('--results', type=Path, help='the pinned results.jsonl')
    ap.add_argument('--peaks', type=Path, help='a whole-run floor pass')
    ap.add_argument('--camera-height', type=fs.fuse_camera_height_arg, default='per-rig')
    ap.add_argument('--city')
    ap.add_argument('--radius', type=float, default=15.0)
    ap.add_argument('--benchmark-ids', type=Path, default=None)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--tally', type=Path, default=None, metavar='LABELS',
                    help='offline: re-derive the row counts of a labels.jsonl (tiers, '
                         'distinct targets) into --tally-out, then exit')
    ap.add_argument('--tally-out', type=Path, default=None,
                    help='--tally output (default: tally.json beside LABELS)')
    args = ap.parse_args(argv)
    if args.tally:
        return cmd_tally(args.tally, args.tally_out or args.tally.with_name('tally.json'))
    missing = [f'--{k}' for k in ('results', 'peaks', 'city', 'out') if not getattr(args, k)]
    if missing:
        ap.error(f'mining needs {" ".join(missing)} (or --tally LABELS)')
    try:
        run_panos, _skipped, height, _auto = fs.load_at_height(
            args.results, args.camera_height, read_heights=True)
    except ValueError as e:      # e.g. per-rig on a GSV run, or a table for another file
        ap.error(str(e))
    params = cli_params(height)
    bench = set()
    if args.benchmark_ids:
        bench = {line.strip() for line in open(args.benchmark_ids, encoding='utf-8')
                 if line.strip()}
    rows, stats = mine(run_panos, params, read_peaks(args.peaks), radius_m=args.radius,
                       city=args.city, benchmark_ids=bench)
    stats['camera_height'] = str(args.camera_height)
    stats['camera_height_resolved_m'] = resolved_heights_m(run_panos, params)
    table = fs.height_table_path(args.camera_height, args.results)
    stats['inputs'] = {
        'results_sha256': sha256_file(args.results),
        'peaks_sha256': sha256_file(args.peaks),
        'camera_heights_sha256': None if table is None else sha256_file(table),
        'benchmark_ids_sha256': (None if args.benchmark_ids is None
                                 else sha256_file(args.benchmark_ids))}
    stats['params'] = {'min_confidence': params.min_confidence, 'mask_rig': params.mask_rig,
                       'apply_pose': params.apply_pose, 'window': pa.WINDOW,
                       'peak_floor': pa.FLOOR, 'withheld_at': pa.OPERATIONAL,
                       'target_rig_mask_y': NADIR_MASK_Y}
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / 'labels.jsonl', 'w', encoding='utf-8', newline=NL) as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + NL)
    (args.out / 'summary.json').write_text(json.dumps(stats, indent=1, sort_keys=True) + NL,
                                           encoding='utf-8', newline=NL)
    print(json.dumps(stats, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
