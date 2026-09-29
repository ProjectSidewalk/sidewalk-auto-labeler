"""Re-run RampNet at the storage floor on ARCHIVED panos of a pinned run (RampNet#158 step 3).

Runs from before the storage floor (#28) -- richmond, bend -- hold only peaks >= 0.55, so
the peak-anchored target definition (scripts/peak_anchor.py) has nothing to read there.
``scripts/reinfer.py`` re-fetches imagery and metadata from the source, which moves the
input; this script instead reads the native-resolution JPEGs already in the makelab2 run
archive (``<archive>/<city>/panos/<id>.jpg``) and keeps each pano's block from the pinned
``results.jsonl`` untouched, so only the detections change.

Each output record is the pinned record with ``detections`` replaced by the model's peaks
down to the floor (detectors.DETECTION_STORAGE_FLOOR, at most MAX_PEAKS_PER_PANO), plus a
``floor_reinfer`` block (model provenance, image path and size).

``--check`` is the pre-registered instrument check, run before anything is scored: per
pano, do the re-inferred peaks >= 0.55 reproduce the pinned run's >= 0.55 detections (same
count, each within one heatmap cell, 1/1024 in x and 1/512 in y)? The gate (RampNet#158
step-3 plan): at least 95% of panos must reproduce, or the pass is not used.

Step 4 runs it over a WHOLE run (no ``--ids``: every pano of ``--results``, in file order)
and gates it on the 124 benchmark panos inside that run (``--check --ids <benchmark ids>``).
``--workers N`` decodes JPEGs in N threads; the forward pass stays serialized
(CurbRampDetector at batch_size 1), so every pano's peaks equal a sequential pass's and only
the order of output lines changes. ``--digest`` prints an order-independent sha256.

Usage (makelab2, A40):
    python scripts/floor_infer_archive.py --results ARCHIVE/richmond/results.jsonl \\
        --panos ARCHIVE/richmond/panos --ids ids.txt --out richmond.floor.jsonl
    python scripts/floor_infer_archive.py --results ... --out richmond.floor.jsonl --check
    # whole run, then the gate on the benchmark panos inside it
    python scripts/floor_infer_archive.py --results ARCHIVE/richmond/results.jsonl \\
        --panos ARCHIVE/richmond/panos --out richmond.all.floor.jsonl --workers 6
    python scripts/floor_infer_archive.py --results ... --out richmond.all.floor.jsonl \\
        --ids docs/figures/mined-precision/data/step3/richmond_ids.txt --check
    python scripts/floor_infer_archive.py --results ... --out richmond.all.floor.jsonl --digest
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import (BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR,  # noqa: E402
                       MAX_PEAKS_PER_PANO)

CELL_X, CELL_Y = 1.0 / 1024, 1.0 / 512     # one heatmap cell
GATE = 0.95
NL = '\n'


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def all_ids(results):
    """Every pano id of a pinned results.jsonl, in file order (the whole-run pass)."""
    with open(results, encoding='utf-8') as f:
        return [str(json.loads(line)['pano']['panorama_id']) for line in f if line.strip()]


def pinned_records(results, ids):
    want, out = set(ids), {}
    with open(results, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                pid = str(rec['pano']['panorama_id'])
                if pid in want:
                    out[pid] = rec
    missing = want - set(out)
    if missing:
        raise SystemExit(f'{len(missing)} ids not in {results}, e.g. {sorted(missing)[0]}')
    return out


def ops(dets):
    return sorted((d['x_normalized'], d['y_normalized']) for d in dets
                  if d['confidence'] >= BENCHMARK_CONFIDENCE)


def reproduces(old_dets, new_dets):
    """The instrument check for one pano (see module docstring)."""
    a, b = ops(old_dets), ops(new_dets)
    if len(a) != len(b):
        return False
    left = list(b)
    for x, y in a:
        hit = next((q for q in left if min(abs(q[0] - x), 1 - abs(q[0] - x)) <= CELL_X + 1e-9
                    and abs(q[1] - y) <= CELL_Y + 1e-9), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def _ids(args):
    return read_ids(args.ids) if args.ids else all_ids(args.results)


def cmd_infer(args):
    from PIL import Image
    from detectors.curb_ramp import CurbRampDetector
    Image.MAX_IMAGE_PIXELS = None
    ids = _ids(args)
    recs = pinned_records(args.results, ids)
    done = set()
    if args.out.exists():
        with open(args.out, encoding='utf-8') as f:
            done = {str(json.loads(line)['pano']['panorama_id']) for line in f if line.strip()}
    det = CurbRampDetector()
    todo = [pid for pid in ids if pid not in done]
    t0 = time.time()

    def one(pid):
        # Decode + preprocess run in this worker thread; detect() serializes only the
        # forward pass, so each pano's peaks are what a sequential pass gives.
        path = args.panos / f'{pid}.jpg'
        img = Image.open(path).convert('RGB')
        peaks = det.detect(img)
        rec = dict(recs[pid])
        rec['detections'] = [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                             for x, y, c in peaks]
        rec['floor_reinfer'] = {'image': str(path), 'image_size': list(img.size),
                                'storage_floor': DETECTION_STORAGE_FLOOR,
                                'max_peaks': MAX_PEAKS_PER_PANO,
                                'model': det.provenance}
        return json.dumps(rec) + NL

    n = 0
    with open(args.out, 'a', encoding='utf-8', newline=NL) as f, \
            ThreadPoolExecutor(max(1, args.workers)) as pool:
        for line in pool.map(one, todo):
            f.write(line)
            n += 1
            if n % 500 == 0:
                f.flush()
                print(f'{n} / {len(todo)} panos, {time.time() - t0:.0f} s', flush=True)
    print(f'{n} panos in {time.time() - t0:.1f} s -> {args.out}')


def canonical_sha256(path):
    """sha256 over the output's lines sorted by pano id: independent of --workers order,
    so two passes can be compared (and a committed digest checked) whatever their order."""
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip(NL) for line in f if line.strip()]
    lines.sort(key=lambda s: str(json.loads(s)['pano']['panorama_id']))
    return hashlib.sha256((NL.join(lines) + NL).encode('utf-8')).hexdigest()


def cmd_check(args):
    ids = _ids(args)
    recs = pinned_records(args.results, ids)
    new = {}
    with open(args.out, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                new[str(r['pano']['panorama_id'])] = r
    missing = [p for p in ids if p not in new]
    ok = [p for p in ids if p in new and reproduces(recs[p]['detections'], new[p]['detections'])]
    n_ops_old = sum(len(ops(recs[p]['detections'])) for p in ids)
    n_ops_new = sum(len(ops(new[p]['detections'])) for p in ids if p in new)
    share = len(ok) / len(ids) if ids else 0.0
    out = {'panos': len(ids), 'missing': len(missing), 'reproduce': len(ok),
           'share': round(share, 4), 'gate': GATE, 'passes': share >= GATE and not missing,
           'ops_old': n_ops_old, 'ops_new': n_ops_new,
           'not_reproducing': [p for p in ids if p in new and p not in set(ok)]}
    print(json.dumps({k: v for k, v in out.items() if k != 'not_reproducing'}))
    if args.check_out:
        args.check_out.write_text(json.dumps(out, indent=1) + NL, encoding='utf-8')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument('--results', type=Path, required=True, help='the pinned results.jsonl')
    ap.add_argument('--panos', type=Path, help='the archive panos/ directory')
    ap.add_argument('--ids', type=Path, default=None,
                    help='pano ids to run / check (default: every pano of --results)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=1,
                    help='threads decoding JPEGs in parallel; the forward pass stays '
                         'serialized, so peaks are identical to --workers 1')
    ap.add_argument('--check', action='store_true', help='run the instrument check only')
    ap.add_argument('--check-out', type=Path, default=None)
    ap.add_argument('--digest', action='store_true',
                    help='print the order-independent sha256 of --out and exit')
    args = ap.parse_args()
    if args.digest:
        print(canonical_sha256(args.out))
    elif args.check:
        cmd_check(args)
    else:
        if not args.panos:
            ap.error('--panos is required for inference')
        cmd_infer(args)


if __name__ == '__main__':
    main()
