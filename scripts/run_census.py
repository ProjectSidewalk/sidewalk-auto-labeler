"""Census of a finished run: who captured it, when, with what pose, and what it detected.

Written for the first Panoramax city (issue #57, docs/panoramax-bayonne.md), but reads
any results.jsonl. Stdlib only; no network, no GPU. Every count is over the panos the
run PROCESSED (one line each); a detection count is over stored detections filtered at
a tier (detectors.OPERATIONAL_CONFIDENCE / BENCHMARK_CONFIDENCE), never the raw
storage floor.

Tables (CSV + report.md under --out):
    rigs.csv       panos per camera_make / camera_model (+ source dimensions)
    years.csv      panos per capture year
    pose.csv       panos per pose group: tilt (pitch/roll reported, not both 0), zeros
                   (reported 0/0), absent -- the Panoramax mixture #57 measured
    producers.csv  panos per producer (`copyright`), instance and license
    detections.csv detections per pano at each tier, and the share of them on the
                   camera rig (detectors.on_camera_rig, dip >= NADIR_MASK_DEG) and
                   inside a nadir band y >= --band-y (default 0.8, the bottom 20% of
                   the equirect: Bayonne's municipal rig burns a white logo band there)
    rig_detections.csv
                   panos and, per tier, detections / detections_per_pano /
                   panos_with_detection / share_with_detection per (camera_make,
                   camera_model, dimensions, capture_year) -- the per-rig rate a city
                   with several rigs needs (Lyon, docs/panoramax-lyon.md)

Usage:
    python scripts/run_census.py runs/bayonne --out docs/figures/panoramax-bayonne/data
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       on_camera_rig)

TIERS = (OPERATIONAL_CONFIDENCE, BENCHMARK_CONFIDENCE)


def pose_group(pano):
    """'tilt' | 'zeros' | 'absent' from a pano block's camera_pitch / camera_roll.

    Example:
        >>> pose_group({'camera_pitch': None, 'camera_roll': None})
        'absent'
        >>> pose_group({'camera_pitch': 0.0, 'camera_roll': 0.0})
        'zeros'
        >>> pose_group({'camera_pitch': -2.5, 'camera_roll': 0.0})
        'tilt'
    """
    pitch, roll = pano.get('camera_pitch'), pano.get('camera_roll')
    if pitch is None or roll is None:
        return 'absent'
    return 'zeros' if float(pitch) == 0.0 and float(roll) == 0.0 else 'tilt'


def census(lines, band_y=0.8):
    """Counters over an iterable of results.jsonl records (dicts)."""
    c = {'rigs': Counter(), 'years': Counter(), 'pose': Counter(), 'producers': Counter(),
         'panos': 0, 'rig_years': Counter(),
         'rig_dets': {t: Counter() for t in TIERS},
         'rig_with': {t: Counter() for t in TIERS}}
    det = {t: {'panos_with': 0, 'detections': 0, 'on_rig': 0, 'in_band': 0}
           for t in TIERS}
    for rec in lines:
        p = rec['pano']
        c['panos'] += 1
        rig = (p.get('camera_make'), p.get('camera_model'),
               f"{p.get('width')}x{p.get('height')}")
        year = (p.get('capture_date') or '')[:4] or 'unknown'
        c['rigs'][rig] += 1
        c['years'][year] += 1
        c['rig_years'][(*rig, year)] += 1
        c['pose'][pose_group(p)] += 1
        c['producers'][(p.get('copyright'), p.get('panoramax_instance'),
                        p.get('license'))] += 1
        for t in TIERS:
            ds = [d for d in rec.get('detections') or [] if d['confidence'] >= t]
            det[t]['panos_with'] += bool(ds)
            c['rig_dets'][t][(*rig, year)] += len(ds)
            c['rig_with'][t][(*rig, year)] += bool(ds)
            det[t]['detections'] += len(ds)
            det[t]['on_rig'] += sum(on_camera_rig(d['y_normalized']) for d in ds)
            det[t]['in_band'] += sum(d['y_normalized'] >= band_y for d in ds)
    c['detections'] = det
    return c


RIG_DETECTION_HEADER = (
    ['camera_make', 'camera_model', 'dimensions', 'capture_year', 'panos']
    + [f'{col}_{t:g}' for t in TIERS
       for col in ('detections', 'detections_per_pano', 'panos_with_detection',
                   'share_with_detection')])


def rig_detection_rows(c):
    """rig_detections.csv rows from census() output, largest group first.

    One row per (camera_make, camera_model, dimensions, capture_year); per tier in
    TIERS: detections, detections per pano (3 dp), panos with >= 1 detection, and
    that share (4 dp). Ties in pano count break on the key, so the order is stable.
    """
    rows = []
    for key, n in sorted(c['rig_years'].items(),
                         key=lambda kv: (-kv[1], tuple(str(x) for x in kv[0]))):
        row = [*key, n]
        for t in TIERS:
            k, w = c['rig_dets'][t][key], c['rig_with'][t][key]
            row += [k, round(k / n, 3), w, round(w / n, 4)]
        rows.append(row)
    return rows


def _write(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--results', type=Path, default=None,
                    help='results file (default <run_dir>/results.jsonl)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--band-y', type=float, default=0.8)
    args = ap.parse_args()
    path = args.results or args.run_dir / 'results.jsonl'

    def records():
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    c = census(records(), args.band_y)
    n = c['panos']
    args.out.mkdir(parents=True, exist_ok=True)
    share = lambda k: round(k / n, 4) if n else None
    _write(args.out / 'rigs.csv', ['camera_make', 'camera_model', 'dimensions', 'panos',
                                   'share'],
           [(*k, v, share(v)) for k, v in c['rigs'].most_common()])
    _write(args.out / 'years.csv', ['capture_year', 'panos', 'share'],
           [(k, v, share(v)) for k, v in sorted(c['years'].items())])
    _write(args.out / 'pose.csv', ['pose_group', 'panos', 'share'],
           [(k, v, share(v)) for k, v in c['pose'].most_common()])
    _write(args.out / 'producers.csv', ['producer', 'instance', 'license', 'panos', 'share'],
           [(*k, v, share(v)) for k, v in c['producers'].most_common()])
    det_rows = []
    for t, d in c['detections'].items():
        k = d['detections']
        det_rows.append((t, n, d['panos_with'], share(d['panos_with']), k,
                         round(k / n, 3) if n else None, d['on_rig'],
                         round(d['on_rig'] / k, 4) if k else None, args.band_y,
                         d['in_band'], round(d['in_band'] / k, 4) if k else None))
    _write(args.out / 'detections.csv',
           ['min_confidence', 'panos', 'panos_with_detection', 'share_with_detection',
            'detections', 'detections_per_pano', 'on_rig', 'on_rig_share', 'band_y',
            'in_band', 'in_band_share'], det_rows)
    rig_rows = rig_detection_rows(c)
    _write(args.out / 'rig_detections.csv', RIG_DETECTION_HEADER, rig_rows)
    lines = [f'# Run census: {args.run_dir.name}', '',
             f'Generated by scripts/run_census.py from `{path.name}` ({n} panos).', '',
             '| tier | panos with a detection | detections | per pano | on rig '
             f'(dip >= mask) | in band (y >= {args.band_y:g}) |', '|---|---:|---:|---:|---:|---:|']
    for r in det_rows:
        lines.append(f'| {r[0]} | {r[2]} ({100 * (r[3] or 0):.1f}%) | {r[4]} | {r[5]} | '
                     f'{r[6]} ({100 * (r[7] or 0):.2f}%) | {r[9]} ({100 * (r[10] or 0):.2f}%) |')
    for name, header in (('rigs', 'camera_make, camera_model, dimensions'),
                         ('years', 'capture year'), ('pose', 'pose group'),
                         ('producers', 'producer, instance, license')):
        lines += ['', f'## {name}', '', f'| {header} | panos | share |', '|---|---:|---:|']
        items = (sorted(c[name].items()) if name == 'years'
                 else c[name].most_common(12))
        for k, v in items:
            label = ', '.join(str(x) for x in k) if isinstance(k, tuple) else k
            lines.append(f'| {label} | {v} | {100 * share(v):.1f}% |')
    lines += ['', '## detections per pano by rig and capture year', '',
              '| camera_make, camera_model, dimensions | year | panos | '
              + ' | '.join(f'per pano @{t:g}' for t in TIERS) + ' |',
              '|---|---|---:|' + '---:|' * len(TIERS)]
    for r in rig_rows:
        rates = [r[6 + 4 * i] for i in range(len(TIERS))]
        lines.append(f'| {r[0]}, {r[1]}, {r[2]} | {r[3]} | {r[4]} | '
                     + ' | '.join(str(x) for x in rates) + ' |')
    (args.out / 'census.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
