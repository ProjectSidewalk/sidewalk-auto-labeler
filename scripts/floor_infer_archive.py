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

``--manifest`` (added after the step-3 review, 2026-09-29; changes no detection and no
gate) writes a sidecar ``<city>.floor.inputs.json``: the sha256 of every input image (from
the archive's ``index.csv``, or by hashing ``--panos``), checked against RampNet's
``benchmark/<city>/imagery_manifest.json`` so a replicator can run the pass on the
published benchmark imagery instead of the makelab2 archive, plus the software versions of
the environment the pass ran in. Inference itself now also writes ``<out>.software.json``
(torch / transformers / torchvision / Pillow / scikit-image versions) beside its output.

Usage (makelab2, A40):
    python scripts/floor_infer_archive.py --results ARCHIVE/richmond/results.jsonl \\
        --panos ARCHIVE/richmond/panos --ids ids.txt --out richmond.floor.jsonl
    python scripts/floor_infer_archive.py --results ... --out richmond.floor.jsonl --check
    python scripts/floor_infer_archive.py --results ... --ids ids.txt --out richmond.floor.jsonl \\
        --manifest --index ARCHIVE/richmond/index.csv \\
        --imagery-manifest ../RampNet/benchmark/richmond/imagery_manifest.json \\
        --software-json richmond.floor.software.json --manifest-out richmond.floor.inputs.json
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import (BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR,  # noqa: E402
                       MAX_PEAKS_PER_PANO)

CELL_X, CELL_Y = 1.0 / 1024, 1.0 / 512     # one heatmap cell
GATE = 0.95


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


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


SOFTWARE_PACKAGES = ('torch', 'transformers', 'torchvision', 'pillow', 'scikit-image', 'numpy')


def software_versions():
    """Versions of the packages that decide the pixels and the heatmap."""
    import importlib.metadata as md
    out = {}
    for name in SOFTWARE_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    try:
        import torch
        out['torch'] = torch.__version__          # carries the +cuXXX build tag
    except ImportError:
        pass
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_software_json(path):
    """--software-json: versions recorded elsewhere. Refuse one that omits a package
    software_versions() records (numpy was missing from the 2026-09-29 step-3 sidecars),
    so a hand-written block cannot silently record less than a pass writes itself."""
    sw = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = [k for k in SOFTWARE_PACKAGES if k not in sw]
    if missing:
        raise SystemExit(f'{path}: --software-json lacks {missing}; record every one of '
                         f'{list(SOFTWARE_PACKAGES)} (null if not installed)')
    return sw

def input_manifest(ids, floor_recs, index=None, panos=None, imagery_manifest=None):
    """{images: {id: {sha256, bytes, image_size}}, rampnet_check: ...} for the floor pass's
    inputs. Hashes come from the archive ``index`` rows ({id: {sha256, bytes}}) when given,
    else from hashing ``panos/<id>.jpg``."""
    images = {}
    for pid in ids:
        if index is not None:
            row = index[pid]
            sha, size = row['sha256'], int(row['bytes'])
        else:
            path = Path(panos) / f'{pid}.jpg'
            sha, size = sha256_file(path), path.stat().st_size
        images[pid] = {'sha256': sha, 'bytes': size,
                       'image_size': floor_recs[pid]['floor_reinfer']['image_size']}
    check = None
    if imagery_manifest is not None:
        man = imagery_manifest['panos']
        equal = [p for p in ids if p in man and man[p]['sha256'] == images[p]['sha256']
                 and [man[p]['width'], man[p]['height']] == images[p]['image_size']]
        check = {'digest': imagery_manifest.get('digest'), 'n': len(ids),
                 'sha256_and_size_equal': len(equal),
                 'differ': [p for p in ids if p not in set(equal)]}
    return {'images': images, 'rampnet_check': check}


def cmd_manifest(args):
    ids = read_ids(args.ids)
    floor = {}
    with open(args.out, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                floor[str(r['pano']['panorama_id'])] = r
    index = None
    if args.index:
        with open(args.index, encoding='utf-8', newline='') as f:
            index = {r['panorama_id']: r for r in csv.DictReader(f)}
    man = None
    if args.imagery_manifest:
        man = json.loads(Path(args.imagery_manifest).read_text(encoding='utf-8'))
    out = input_manifest(ids, floor, index=index, panos=args.panos, imagery_manifest=man)
    out = {'floor_output': args.out.name, 'floor_output_sha256': sha256_file(args.out),
           'hash_source': ('archive index.csv (sha256 recorded when the archive was '
                           'exported and reconciled; not re-hashed at pass time)'
                           if index is not None else 'hashed from --panos'),
           # RampNet-relative (benchmark/<city>/...), so the file is machine-independent
           'imagery_manifest': (None if not man else
                                'RampNet:' + '/'.join(Path(args.imagery_manifest).parts[
                                    Path(args.imagery_manifest).parts.index('benchmark'):])
                                if 'benchmark' in Path(args.imagery_manifest).parts
                                else Path(args.imagery_manifest).as_posix()),
           'software': (read_software_json(args.software_json)
                        if args.software_json else software_versions()),
           **out}
    chk = out['rampnet_check']
    print(json.dumps({'images': len(out['images']), 'rampnet_check':
                      None if chk is None else {k: v for k, v in chk.items() if k != 'differ'}}))
    Path(args.manifest_out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n',
                                       encoding='utf-8', newline='\n')
    return out


def cmd_infer(args):
    from PIL import Image
    from detectors.curb_ramp import CurbRampDetector
    ids = read_ids(args.ids)
    recs = pinned_records(args.results, ids)
    done = set()
    if args.out.exists():
        with open(args.out, encoding='utf-8') as f:
            done = {str(json.loads(line)['pano']['panorama_id']) for line in f if line.strip()}
    det = CurbRampDetector()
    t0 = time.time()
    n = 0
    with open(args.out, 'a', encoding='utf-8', newline='\n') as f:
        for pid in ids:
            if pid in done:
                continue
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
            f.write(json.dumps(rec) + '\n')
            n += 1
    sw = args.out.with_name(args.out.name + '.software.json')
    sw.write_text(json.dumps({'software': software_versions(), 'model': det.provenance},
                             indent=1, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    print(f'{n} panos in {time.time() - t0:.1f} s -> {args.out} (+ {sw.name})')


def cmd_check(args):
    ids = read_ids(args.ids)
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
        args.check_out.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', type=Path, required=True, help='the pinned results.jsonl')
    ap.add_argument('--panos', type=Path, help='the archive panos/ directory')
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--check', action='store_true', help='run the instrument check only')
    ap.add_argument('--check-out', type=Path, default=None)
    ap.add_argument('--manifest', action='store_true',
                    help='write the input sidecar (sha256 per image, software) only')
    ap.add_argument('--index', type=Path, default=None,
                    help="--manifest: the archive's index.csv (else hash --panos)")
    ap.add_argument('--imagery-manifest', type=Path, default=None,
                    help="--manifest: RampNet benchmark/<city>/imagery_manifest.json")
    ap.add_argument('--software-json', type=Path, default=None,
                    help='--manifest: software versions recorded elsewhere (JSON object); '
                         "default is this environment's")
    ap.add_argument('--manifest-out', type=Path, default=None)
    args = ap.parse_args()
    if args.manifest:
        if not args.manifest_out or not (args.index or args.panos):
            ap.error('--manifest needs --manifest-out and --index or --panos')
        cmd_manifest(args)
    elif args.check:
        cmd_check(args)
    else:
        if not args.panos:
            ap.error('--panos is required for inference')
        cmd_infer(args)


if __name__ == '__main__':
    main()
