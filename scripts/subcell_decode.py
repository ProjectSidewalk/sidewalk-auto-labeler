"""The sub-cell decode in this repo's detector path, measured (issue #111, decode half).

RampNet#221 measured a sub-cell ("gaussian") decode of RampNet's heatmap peaks against human
box centres and shipped it in RampNet. ``detectors.decode`` carries the same rule here as an
opt-in. This script measures what it does through THIS repo's detector, in three steps:

    # 1. GPU (makelab2 A40). ONE forward pass per pano, BOTH decodes from that one heatmap
    #    (never two inference runs: near-tied coarse cells flip between passes). Writes one
    #    JSONL line per pano: the labeler's detections under argmax and gaussian (storage floor
    #    0.1, top 50, exclude_border as production), RampNet's own extraction under both
    #    (rampnet_subcell.detect_peaks at 0.30, clip, exclude_border=False -- the #221
    #    protocol), and the recovered 64x128 coarse map's sha256 (map saved with --coarse-dir).
    python scripts/subcell_decode.py detect --panos-dir <bundle>/panos --ids ids.txt \\
        --out decode_<split>.jsonl --coarse-dir coarse/<split>
    #    ...for an archived run, --results gives the ids AND the pano blocks, and
    #    --source-resize hands the detector a 4096x2048 image exactly as sources/ do:
    python scripts/subcell_decode.py detect --panos-dir <archive>/laurens/panos \\
        --results <archive>/laurens/results.jsonl --source-resize --out decode_laurens.jsonl

The CPU steps (``residual``, ``sigma``, ``world``, ``figures``) read those files; see their
own help.
"""
import argparse
import hashlib
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detectors import decode as dec  # noqa: E402
from detectors import rampnet_subcell as sc  # noqa: E402

#: RampNet#221's extraction floor (the #79 recommended operating point), used for the
#: reproduction of its residual table.
RN_FLOOR = 0.30
#: What every source module hands the detector (panorama.TARGET_SIZE, sources/mapillary.py).
SOURCE_SIZE = (4096, 2048)


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def results_ids(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(str(json.loads(line)['pano']['panorama_id']))
    return out


def _rn(heatmap, decode):
    """RampNet#221's extraction of one heatmap: (row, col, score) rows at 0.30."""
    rcs = sc.detect_peaks(heatmap, RN_FLOOR, decode=decode, clip=True)
    return [[float(r), float(c), float(s)] for r, c, s in rcs]


def pano_line(pid, heatmap, image_size, coarse_dir=None):
    """Everything the CPU steps need from one heatmap (see the module docstring)."""
    import numpy as np
    coarse = sc.coarse_from_heatmap(heatmap)
    resid = sc.upsample_residual(heatmap, coarse)
    c32 = coarse.astype(np.float32)
    if coarse_dir is not None:
        np.save(Path(coarse_dir) / f'{pid}_coarse.npy', c32)
    both = dec.detections_both(heatmap)
    return {'pano_id': pid, 'image_size': list(image_size),
            'upsample_residual': float(f'{resid:.3e}'),
            'coarse_sha256': hashlib.sha256(c32.tobytes()).hexdigest(),
            'argmax': [list(d) for d in both[dec.ARGMAX]],
            'gaussian': [list(d) for d in both[dec.GAUSSIAN]],
            'rn_argmax': _rn(heatmap, dec.ARGMAX),
            'rn_gaussian': _rn(heatmap, dec.GAUSSIAN)}


def cmd_detect(args):
    from PIL import Image
    from detectors.curb_ramp import CurbRampDetector
    import floor_infer_archive as fia
    Image.MAX_IMAGE_PIXELS = None
    ids = read_ids(args.ids) if args.ids else results_ids(args.results)
    if args.limit:
        ids = ids[:args.limit]
    done = set()
    if args.out.exists():
        with open(args.out, encoding='utf-8') as f:
            done = {json.loads(line)['pano_id'] for line in f if line.strip()}
    todo = [p for p in ids if p not in done]
    if args.coarse_dir:
        args.coarse_dir.mkdir(parents=True, exist_ok=True)
    det = CurbRampDetector()          # decode is applied below, both ways, per heatmap

    def work(pid):
        path = args.panos_dir / f'{pid}.jpg'
        img = Image.open(path).convert('RGB')
        size = img.size
        if args.source_resize and img.size != SOURCE_SIZE:
            img = img.resize(SOURCE_SIZE, Image.BILINEAR)
        h = det.heatmap(img)
        return pano_line(pid, h, size, args.coarse_dir)

    started = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    t0 = time.perf_counter()
    n = 0
    print(f'{len(ids)} ids, {len(done)} already in {args.out}, {len(todo)} to run', flush=True)
    with open(args.out, 'a', encoding='utf-8', newline='\n') as f, \
            ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(work, todo):
            f.write(json.dumps(line) + '\n')
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f'  {n}/{len(todo)}  {time.perf_counter() - t0:.0f} s', flush=True)
    wall = time.perf_counter() - t0
    meta_path = args.out.with_name(args.out.name + '.meta.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {
        'passes': []}
    meta.update({'model': det.provenance, 'software': fia.software_versions(),
                 'rampnet_subcell_commit': dec.RAMPNET_SUBCELL_COMMIT,
                 'panos_dir': str(args.panos_dir),
                 'ids_from': str(args.ids or args.results),
                 'source_resize': list(SOURCE_SIZE) if args.source_resize else None,
                 'storage_floor': dec.DETECTION_STORAGE_FLOOR,
                 'max_peaks': dec.MAX_PEAKS_PER_PANO, 'rn_floor': RN_FLOOR})
    stats = det.stats()
    meta['passes'].append({'started': started, 'host': socket.getfqdn(), 'panos': n,
                           'wall_s': round(wall, 1),
                           'forward_s': stats.get('forward_seconds'),
                           'workers': args.workers})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n', encoding='utf-8',
                         newline='\n')
    print(f'done: {n} panos in {wall:.0f} s -> {args.out} (+ {meta_path.name})', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('detect', help='GPU: one forward pass per pano, both decodes')
    p.add_argument('--panos-dir', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--ids', type=Path, help='one pano id per line')
    g.add_argument('--results', type=Path, help="a run's results.jsonl (ids in its order)")
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--coarse-dir', type=Path, default=None)
    p.add_argument('--source-resize', action='store_true',
                   help='resize to 4096x2048 (PIL bilinear) first, as sources/ do')
    p.add_argument('--workers', type=int, default=4, help='image-decode threads')
    p.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == 'detect':
        cmd_detect(args)


if __name__ == '__main__':
    main()
