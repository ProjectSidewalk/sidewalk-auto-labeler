"""Re-run inference over an existing run's panos, by id, at the current storage floor.

Why this exists (issue #20): runs made before the storage floor (#28) hold only peaks at
or above 0.55, so when the operating point moved to 0.30 there was nothing stored to
submit for the 0.30-0.55 band. A fresh `main.py` run would re-scan and re-thin against
today's coverage and pick a *different* pano set; this fetches exactly the panos the
run already has, in the file's own order, through the same fetch -> detect -> record
path as main.py, and writes a sibling file. The pano block is rebuilt from fresh
metadata (positions included), so the run's `--mapillary-position` binding is honoured
from the manifest.

    python scripts/reinfer.py runs/richmond                 # -> runs/richmond/results.f01.jsonl
    python scripts/reinfer.py runs/richmond --verify        # compare, no network, no GPU

Resumable: `<out>.processed` caches the ids already written (and the deterministic
skips - a pano the source no longer serves). `--verify` compares the new file's
detections at BENCHMARK_CONFIDENCE to the old file's, per pano, on the pixel key PS
stores (`round(x*W)`, `round(y*H)`): the band may ship only when every pano the new
file holds reproduces the old file's operational set exactly, since anything else means
the labels already live on the server are not the ones this file describes.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import BENCHMARK_CONFIDENCE  # noqa: E402


def read_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pixel_set(record, floor):
    w, h = record['pano']['width'], record['pano']['height']
    return {(round(d['x_normalized'] * w), round(d['y_normalized'] * h))
            for d in record['detections'] if d['confidence'] >= floor}


def verify(old_path, new_path, floor=BENCHMARK_CONFIDENCE, band_floor=None):
    """Compare the new file to the old one at `floor`; return (summary dict, mismatches)."""
    new_by_id = {r['pano']['panorama_id']: r for r in read_records(new_path)}
    summary = {'old_panos': 0, 'new_panos': len(new_by_id), 'missing': 0, 'exact': 0,
               'mismatch': 0, 'old_labels': 0, 'band_labels': 0}
    mismatches = []
    for old in read_records(old_path):
        pid = old['pano']['panorama_id']
        summary['old_panos'] += 1
        old_set = pixel_set(old, floor)
        summary['old_labels'] += len(old_set)
        new = new_by_id.get(pid)
        if new is None:
            summary['missing'] += 1
            continue
        new_set = pixel_set(new, floor)
        if new_set == old_set:
            summary['exact'] += 1
            if band_floor is not None:
                summary['band_labels'] += sum(
                    1 for d in new['detections'] if band_floor <= d['confidence'] < floor)
        else:
            summary['mismatch'] += 1
            mismatches.append((pid, sorted(old_set - new_set), sorted(new_set - old_set)))
    return summary, mismatches


def reinfer(run_dir, out_path, workers, limit):
    import main  # noqa: E402  (loads torch lazily below, like main.py does)
    from sources import get_source
    from detectors.curb_ramp import CurbRampDetector

    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    source_name = manifest.get('imagery_source') or manifest.get('source') or 'gsv'
    source = get_source(source_name)
    if source_name == 'mapillary':
        source.POSITION_FIELD = manifest.get('mapillary_position', 'sfm')
    source.prepare()
    main.curb_ramp_detector = CurbRampDetector()

    cache_path = Path(f"{out_path}.processed")
    done = set()
    if cache_path.exists():
        done = {l.strip() for l in cache_path.read_text(encoding='utf-8').splitlines() if l.strip()}
    todo = [(r['pano']['panorama_id'], r['pano']['lat'], r['pano']['lng'])
            for r in read_records(run_dir / 'results.jsonl')
            if r['pano']['panorama_id'] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"-> {len(done)} already done, {len(todo)} to re-infer from {source_name} "
          f"(storage floor {main.DETECTION_STORAGE_FLOOR}) -> {out_path}")

    counts = {'success': 0, 'skipped': 0, 'failed': 0}
    # Futures are consumed in submission order so the output keeps the input's line order;
    # downloads still overlap (the pool runs ahead) and the GPU serialises inference.
    with open(cache_path, 'a', encoding='utf-8') as f_cache, \
         open(out_path, 'a', encoding='utf-8') as f_out, \
         ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(main.process_pano, source, pid, lat, lon) for pid, lat, lon in todo]
        try:
            from tqdm import tqdm
            it = tqdm(futures, desc="Re-inferring")
        except ImportError:
            it = futures
        for future in it:
            counts[main.handle_result(future.result(), f_cache, f_out)] += 1
    print(f"-> Re-inference: {counts['success']} written, {counts['skipped']} skipped "
          f"(no longer served / unusable), {counts['failed']} failed (retry by re-running).")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir', type=Path, help='runs/<name>, holding results.jsonl + manifest.json')
    ap.add_argument('--out', type=Path, default=None,
                    help='output file (default <run_dir>/results.f01.jsonl)')
    ap.add_argument('--verify', action='store_true',
                    help='compare --out against results.jsonl at the benchmark threshold '
                         'and exit 1 on any mismatch; no network, no GPU')
    ap.add_argument('--band-floor', type=float, default=None,
                    help='with --verify: also count labels in [band-floor, threshold) on '
                         'the panos that reproduced, i.e. what a band campaign would ship')
    ap.add_argument('--workers', type=int, default=None,
                    help='fetch threads (default main.PROCESSING_CONCURRENCY)')
    ap.add_argument('--limit', type=int, default=None, help='re-infer at most this many panos')
    args = ap.parse_args()

    out_path = args.out or (args.run_dir / 'results.f01.jsonl')
    if args.verify:
        summary, mismatches = verify(args.run_dir / 'results.jsonl', out_path,
                                     band_floor=args.band_floor)
        print(json.dumps(summary, indent=2))
        for pid, only_old, only_new in mismatches[:10]:
            print(f"  {pid}: old-only {only_old} new-only {only_new}")
        if summary['mismatch']:
            print(f"{summary['mismatch']} pano(s) do not reproduce the old operational set: "
                  f"do NOT ship a band from {out_path.name}.")
            raise SystemExit(1)
        return

    import main  # noqa: E402
    reinfer(args.run_dir, out_path, args.workers or main.PROCESSING_CONCURRENCY, args.limit)


if __name__ == '__main__':
    main_cli()
