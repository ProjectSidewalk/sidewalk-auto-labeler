"""POST HOC: why bend's floor pass failed the step-3 gate, and whether richmond's existing
full-run re-inference already passes it (RampNet#158 step 3 review).

NOT part of the pre-registered step-3 design. The gate (>= 95% of a city's panos
reproduce their >= 0.55 detections within one heatmap cell), its verdicts (richmond
124/124 passes, bend 77/110 fails) and every step-3 number stand as posted. Both
subcommands only read committed or pinned files; no model, no network.

``displacement`` -- for every pinned operational (>= 0.55) detection of the judged panos,
the nearest re-inferred peak of the floor pass (any confidence >= the 0.1 floor): is it the
same position exactly, within one cell (1/1024 in x, 1/512 in y, seam-wrapped), and by how
much did its confidence move? A pixel-source difference (bend: the production run fed the
model Google's ZOOM-3 rendition, the floor pass the archive's max-zoom JPEG) shows up as
small shifts on every detection; a weights difference would not reproduce richmond either.
It also tallies the archived images' sizes, since zoom 3 is 4096 px wide only for a
16384-px pano (a 13312-px pano's zoom 3 is 3328 px, upscaled to 4096 in production).

``f01`` -- runs the gate's own check (floor_infer_archive.reproduces) over every pano of a
full-run re-inference against the pinned run, and compares its FULL peak sets (floor 0.1
included, which the gate does not check) with the floor pass on the judged panos.

Usage:
    D=docs/figures/mined-precision/data/step3
    python scripts/step3_gate_diag.py displacement --city richmond \\
        --results $FROZEN/richmond/results.jsonl --floor $D/richmond.floor.jsonl \\
        --json-out $D/posthoc_gate_displacement_richmond.json
    python scripts/step3_gate_diag.py displacement --city bend \\
        --results $FROZEN/bend/results.jsonl --floor $D/bend.floor.jsonl \\
        --json-out $D/posthoc_gate_displacement_bend.json
    python scripts/step3_gate_diag.py f01 --results $FROZEN/richmond/results.jsonl \\
        --f01 runs/richmond/results.f01.jsonl --floor $D/richmond.floor.jsonl \\
        --json-out $D/posthoc_f01_gate.json
"""
import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import floor_infer_archive as fia  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402


def read_jsonl(path, want=None):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = str(r['pano']['panorama_id'])
                if want is None or pid in want:
                    out[pid] = r
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _dx(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def displacement(pinned, floor):
    """Per pinned operational detection: (exact = same heatmap cell, within_one_cell,
    |dconf|) against the nearest floor-pass peak of the same pano."""
    rows = []
    for pid, rec in floor.items():
        new = [(d['x_normalized'], d['y_normalized'], d['confidence'])
               for d in rec['detections']]
        for d in pinned[pid]['detections']:
            if d['confidence'] < BENCHMARK_CONFIDENCE:
                continue
            x, y, c = d['x_normalized'], d['y_normalized'], d['confidence']
            q = min(new, key=lambda q: (_dx(q[0], x) * 1024) ** 2 + ((q[1] - y) * 512) ** 2)
            rows.append({'pano': pid, 'exact': q[0] == x and q[1] == y,
                         'one_cell': _dx(q[0], x) <= fia.CELL_X + 1e-9
                         and abs(q[1] - y) <= fia.CELL_Y + 1e-9,
                         'dconf': abs(q[2] - c)})
    return rows


def cmd_displacement(args):
    floor = read_jsonl(args.floor)
    pinned = read_jsonl(args.results, set(floor))
    rows = displacement(pinned, floor)
    dc = [r['dconf'] for r in rows]
    sizes = Counter(tuple(r['floor_reinfer']['image_size']) for r in floor.values())
    out = {'what': 'POST HOC (RampNet#158 step-3 review): pinned operational detections vs '
                   'the nearest floor-pass peak (step3_gate_diag.py displacement)',
           'results_sha256': sha256(args.results), 'floor_sha256': sha256(args.floor),
           'city': args.city, 'operational_detections': len(rows),
           'same_cell': sum(r['exact'] for r in rows),
           'same_cell_and_confidence': sum(r['exact'] and r['dconf'] == 0 for r in rows),
           'within_one_cell': sum(r['one_cell'] for r in rows),
           'dconf_median': round(statistics.median(dc), 6),
           'dconf_max': round(max(dc), 6),
           'archive_image_sizes': {f'{w}x{h}': n for (w, h), n in sorted(sizes.items())}}
    print(json.dumps(out))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8',
                                 newline='\n')
    return out


def cmd_f01(args):
    pinned = read_jsonl(args.results)
    f01 = read_jsonl(args.f01)
    ids = sorted(set(pinned) & set(f01))
    ok = [p for p in ids if fia.reproduces(pinned[p]['detections'], f01[p]['detections'])]
    floor = read_jsonl(args.floor)
    same_set, max_dconf = 0, 0.0
    for pid, rec in floor.items():
        a = sorted((d['x_normalized'], d['y_normalized'], d['confidence'])
                   for d in rec['detections'])
        b = sorted((d['x_normalized'], d['y_normalized'], d['confidence'])
                   for d in f01[pid]['detections'])
        if len(a) == len(b) and all(p[:2] == q[:2] for p, q in zip(a, b)):
            same_set += 1
            max_dconf = max([max_dconf] + [abs(p[2] - q[2]) for p, q in zip(a, b)])
    out = {'what': 'POST HOC (RampNet#158 step-3 review): the step-3 gate check run over a '
                   'full-run re-inference, and its full peak sets vs the floor pass',
           'results_sha256': sha256(args.results), 'f01_sha256': sha256(args.f01),
           'floor_sha256': sha256(args.floor),
           'pinned_panos': len(pinned), 'f01_panos': len(f01), 'shared_panos': len(ids),
           'reproduce': len(ok), 'share': round(len(ok) / len(ids), 6),
           'gate': fia.GATE, 'passes': len(ok) / len(ids) >= fia.GATE,
           'not_reproducing': [p for p in ids if p not in set(ok)],
           'judged_panos': len(floor), 'judged_same_peak_positions': same_set,
           'judged_max_abs_dconf': round(max_dconf, 8)}
    print(json.dumps({k: v for k, v in out.items() if k != 'not_reproducing'}))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8',
                                 newline='\n')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('displacement')
    d.add_argument('--city', required=True)
    d.add_argument('--results', type=Path, required=True, help='the pinned results.jsonl')
    d.add_argument('--floor', type=Path, required=True, help='the step-3 floor pass')
    d.add_argument('--json-out', type=Path, default=None)
    f = sub.add_parser('f01')
    f.add_argument('--results', type=Path, required=True, help='the pinned results.jsonl')
    f.add_argument('--f01', type=Path, required=True, help='the full-run re-inference')
    f.add_argument('--floor', type=Path, required=True, help='the step-3 floor pass')
    f.add_argument('--json-out', type=Path, default=None)
    args = ap.parse_args()
    (cmd_displacement if args.cmd == 'displacement' else cmd_f01)(args)


if __name__ == '__main__':
    main()
