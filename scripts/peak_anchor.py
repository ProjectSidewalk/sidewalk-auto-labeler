"""Peak-anchored mined targets (RampNet#158 step 3).

Steps 1-2 and phase 2 place a mined target by GEOMETRY alone: the fused site projected
into the target pano (flat ground), or an image transfer from a member view. Nothing asks
whether the model responded there at all. Step 1's plan proposed a stricter definition:
emit a target only where the target pano holds a SUB-THRESHOLD peak near the geometric
anchor, and put the target on that peak's pixel -- geometry proposes, the model localizes.

This script turns an anchor per candidate into a placement file for
``mined_precision.py --placement``. It reads no verdict: only the candidates' anchors (the
step-2 flat projection from ``sources.csv``, or another arm's placement JSONL) and the
target panos' stored peaks.

THE DEFINITION (fixed and posted on RampNet#158 before any inference or scoring):

  window   the benchmark's own per-pano match radius, 0.022 in RampNet's pano geometry
           (x scaled by 1024, y by 512, x cyclic at the seam; agree_rate.pano_distance),
           around the anchor pixel -- about 7.9 deg.
  peaks    the target pano's stored peaks, down to the storage floor (0.1,
           detectors.DETECTION_STORAGE_FLOOR), at most 50 per pano, from the published
           RampNet model. For runs that predate the floor (richmond, bend) these come from
           one inference pass over the archived panos (scripts/floor_infer_archive.py).
  rule     if any peak >= the benchmark threshold (0.55) is in the window, the model
           already fires there: NOT EMITTED (not a miss). Otherwise, if any peak in
           [0.1, 0.55) is in the window, EMIT the highest-confidence one's pixel.
           Otherwise (no response at all): NOT EMITTED.

An arm's anchor that fell back (x null) is replaced by the flat anchor, as in phase 2.

Usage:
    python scripts/peak_anchor.py --sources-root DIR --cities richmond paterson ... \\
        --peaks richmond=runs/richmond/floor/results.floor.jsonl bend=... \\
        --runs-root FROZEN --out placement_peak_flat.jsonl
    # ...anchored on another arm's placement instead of the flat projection:
    python scripts/peak_anchor.py ... --anchor-placement placement/roma_local.jsonl
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR  # noqa: E402

# RampNet's pano geometry (rampnet.detection_eval / agree_rate.pano_distance)
PANO_SCALE_X, PANO_SCALE_Y = 1024, 512
WINDOW = 0.022
FLOOR = DETECTION_STORAGE_FLOOR
OPERATIONAL = BENCHMARK_CONFIDENCE


def pano_distance(ax, ay, bx, by):
    """Normalized-x distance between two pano points, x cyclic at the seam."""
    dx = abs(ax - bx) % 1.0
    dx = min(dx, 1.0 - dx) * PANO_SCALE_X
    dy = (ay - by) * PANO_SCALE_Y
    return math.hypot(dx, dy) / PANO_SCALE_X


def anchor_peak(anchor, peaks, window=WINDOW, floor=FLOOR, operational=OPERATIONAL):
    """The rule above for one candidate. ``peaks`` = [(x, y, conf)]. Returns
    (status, peak or None): 'emit' with the chosen sub-threshold peak, 'operational_in_window'
    or 'no_peak'."""
    ax, ay = anchor
    near = [p for p in peaks if p[2] >= floor and pano_distance(ax, ay, p[0], p[1]) <= window]
    if any(p[2] >= operational for p in near):
        return 'operational_in_window', None
    if not near:
        return 'no_peak', None
    return 'emit', max(near, key=lambda p: (p[2], -p[0], -p[1]))


def read_sources(root, cities):
    rows = []
    for city in cities:
        with open(Path(root) / city / 'sources.csv', encoding='utf-8', newline='') as f:
            rows += list(csv.DictReader(f))
    return rows


def read_peaks(path, want):
    """{pano_id: [(x, y, conf)]} for the wanted panos, from a results-style JSONL."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = str(rec['pano']['panorama_id'])
            if pid in want:
                out[pid] = [(d['x_normalized'], d['y_normalized'], d['confidence'])
                            for d in rec['detections']]
    return out


def build(sources, peaks_by_city, anchors=None):
    """Placement rows (one per candidate) for mined_precision --placement."""
    rows = []
    for s in sources:
        key = (s['city'], int(s['site_id']), s['pano_id'])
        anchor, via = (float(s['proj_x']), float(s['proj_y'])), 'flat'
        if anchors is not None:
            a = anchors.get(key, 'missing')
            if a == 'missing':
                raise SystemExit(f'anchor placement has no row for {key}')
            if a is not None:
                anchor, via = a, 'arm'
        peaks = peaks_by_city[s['city']].get(s['pano_id'])
        if peaks is None:
            raise SystemExit(f'no stored peaks for target pano {key}')
        status, p = anchor_peak(anchor, peaks)
        rows.append({'city': key[0], 'site_id': key[1], 'pano_id': key[2],
                     'x': None if p is None else p[0], 'y': None if p is None else p[1],
                     'emit': status == 'emit', 'status': status, 'anchor_via': via,
                     'anchor_x': anchor[0], 'anchor_y': anchor[1],
                     'peak_conf': None if p is None else p[2]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sources-root', required=True, type=Path)
    ap.add_argument('--cities', nargs='+', required=True)
    ap.add_argument('--runs-root', type=Path, required=True,
                    help='<runs-root>/<city>/results.jsonl: stored peaks unless --peaks names '
                         'another file for that city')
    ap.add_argument('--peaks', nargs='*', default=[],
                    help='city=PATH: a floor re-inference to read that city\'s peaks from')
    ap.add_argument('--anchor-placement', type=Path, default=None,
                    help='anchor on this arm\'s placement JSONL instead of the flat projection')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sources = read_sources(args.sources_root, args.cities)
    override = dict(kv.split('=', 1) for kv in args.peaks)
    peaks_by_city = {}
    for city in args.cities:
        want = {s['pano_id'] for s in sources if s['city'] == city}
        path = override.get(city) or args.runs_root / city / 'results.jsonl'
        peaks_by_city[city] = read_peaks(path, want)
    anchors = None
    if args.anchor_placement:
        anchors = {}
        with open(args.anchor_placement, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    k = (r['city'], int(r['site_id']), str(r['pano_id']))
                    anchors[k] = (None if r.get('x') is None or r.get('y') is None
                                  else (float(r['x']), float(r['y'])))
    rows = build(sources, peaks_by_city, anchors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    n = {}
    for r in rows:
        n[(r['city'], r['status'])] = n.get((r['city'], r['status']), 0) + 1
    for city in args.cities:
        print(city, {s: n.get((city, s), 0)
                     for s in ('emit', 'operational_in_window', 'no_peak')})
    print(f'{sum(r["emit"] for r in rows)} of {len(rows)} emitted -> {args.out}')


if __name__ == '__main__':
    main()
