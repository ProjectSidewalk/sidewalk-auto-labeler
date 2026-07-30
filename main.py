import argparse
import math
import hashlib
import json
import socket
import sys
import traceback
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

# Concurrency is plain OS threads (not gevent): streetlevel's sync imagery API runs its
# own asyncio event loop internally (asyncio.run + aiohttp), which needs one thread per
# concurrent call. Under gevent's monkey-patching all workers share one OS thread, so
# those event loops collide and every pano download stalls for minutes.

# streetlevel makes HTTP requests with no timeout; without this, one black-holed
# connection (e.g. Google throttling) hangs its worker thread forever.
socket.setdefaulttimeout(30)

# Progress/report messages use emoji; on Windows the console may be cp1252, where printing
# them raises UnicodeEncodeError and would abort the run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

import geojson
from dotenv import load_dotenv
from shapely.geometry import shape
from tqdm import tqdm

from detectors import DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO
from sources import get_source, SOURCE_NAMES

# Local secrets (e.g. MAPILLARY_ACCESS_TOKEN) from ./.env; real env vars win.
load_dotenv()
# CurbRampDetector is imported lazily in main(): pulling in torch/transformers takes
# many seconds and --scan-only doesn't need them.

# --- Configuration ---
COVERAGE_API_CONCURRENCY = 100
PROCESSING_CONCURRENCY = 50

# Provenance recorded in every JSONL line and in each run's manifest.json.
MODEL_ID = "rampnet-model"
MODEL_TRAINING_DATE = "08-21-2025"
API_VERSION = "1.0.0"


def latlon_to_tile(lat_deg, lon_deg, zoom):
    """Converts latitude and longitude to Slippy Map tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def get_geojson_hash(geojson_data):
    """Creates a stable SHA256 hash of the GeoJSON geometry."""
    # The root of the geojson is the geometry object itself.
    geometry = geojson_data
    canonical_json = json.dumps(geometry, sort_keys=True, separators=(',', ':'))
    sha_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    return sha_hash

def load_processed_ids(cache_file_path):
    """Loads a set of already processed panorama IDs from the cache file."""
    if not os.path.exists(cache_file_path):
        return set()
    with open(cache_file_path, 'r') as f:
        return {line.strip() for line in f}

def _process(pano_id, fetch):
    """
    Fetches one pano (via the given thunk), runs detection, and returns a result
    dict. Designed to be run concurrently in a thread pool.
    """
    try:
        fetched = fetch()
        if fetched['status'] != 'success':
            return {'status': fetched['status'], 'pano_id': pano_id, 'reason': fetched.get('reason', 'Unknown')}

        # The detector returns the list of center points and confidence: [[x, y, confidence], ...] or an empty list [].
        detections = curb_ramp_detector.detect(fetched['image'])

        return {
            'status': 'success',
            'pano_id': pano_id,
            'pano': fetched['pano'],
            'detections': detections
        }
    except Exception as e:
        return {'status': 'failure', 'pano_id': pano_id, 'reason': str(e)}

def process_pano(source, pano_id, lat, lon):
    return _process(pano_id, lambda: source.fetch_pano(pano_id, lat, lon))

def process_gap_pano(source, pano_id, area_shape):
    return _process(pano_id, lambda: source.fetch_pano_by_id(pano_id, area_shape))

def handle_result(result, f_cache, f_jsonl):
    """
    Writes one process result to the run's JSONL/cache files (both flushed
    line-by-line so the run stays resumable) and returns its outcome:
    'success', 'skipped', or 'failed'.
    """
    if result['status'] == 'success':
        # ALWAYS write a line to the JSONL file for a successful process.
        # The 'detections' key will be an empty list [] if none were found.
        # Guarded so a single malformed pano cannot abort the whole run.
        try:
            json_line = json.dumps(build_output_line(result))
        except Exception as e:
            print(f"  ❌ Failed to build output for {result['pano_id']}. Reason: {e}. Will retry on next run.")
            return 'failed'
        f_jsonl.write(json_line + '\n')
        f_jsonl.flush()

        # Mark successfully processed pano in the cache.
        f_cache.write(f"{result['pano_id']}\n")
        f_cache.flush()
        return 'success'
    if result['status'] == 'skipped':
        # Deterministic skips (indoor pano, non-360 image, incomplete metadata,
        # gap-fill target outside the area): cache so they aren't refetched every run.
        f_cache.write(f"{result['pano_id']}\n")
        f_cache.flush()
        return 'skipped'
    print(f"  ❌ Failed to process {result['pano_id']}. Reason: {result.get('reason', 'Unknown')}. Will retry on next run.")
    return 'failed'

def build_output_line(result):
    """
    Builds one JSONL record for a successfully processed panorama: the detections
    plus model provenance around the source-built 'pano' block.
    """
    return {
        "detections": [
            {
                "x_normalized": x_normalized,
                "y_normalized": y_normalized,
                "confidence": confidence
            } for x_normalized, y_normalized, confidence in result['detections']
        ],
        "label_type": "CurbRamp",
        "model_id": MODEL_ID,
        "model_training_date": MODEL_TRAINING_DATE,
        "api_version": API_VERSION,
        "pano": result['pano'],
    }

def save_manifest(manifest_path, manifest):
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

def load_or_init_run_dir(run_dir, geojson_path, geojson_data, area_hash, source_name):
    """
    Creates or validates the run directory (runs/<name>/), which holds all per-area
    state: results.jsonl, already_processed.txt, manifest.json, and a copy of the
    exact geometry used. A run directory is permanently bound to one geometry and one
    imagery source; reusing the name with a different geometry or source is refused
    so that a renamed/edited geojson or a --source change can't silently fork or
    corrupt the run's state.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        if manifest.get('area_hash') != area_hash:
            sys.exit(
                f"❌ Run '{run_dir.name}' was created for a different area geometry\n"
                f"   (manifest hash {manifest.get('area_hash', '?')[:12]}…, this geojson {area_hash[:12]}…).\n"
                f"   Use a new --name for the new geometry, or restore the original geojson\n"
                f"   (an exact copy is kept at {run_dir / 'area.geojson'})."
            )
        # Manifests predating the --source flag are all GSV runs.
        if manifest.get('imagery_source', 'gsv') != source_name:
            sys.exit(
                f"❌ Run '{run_dir.name}' was created with imagery source "
                f"'{manifest.get('imagery_source', 'gsv')}', not '{source_name}'.\n"
                f"   Use a new --name for a different source."
            )
        # Manifests predating the storage floor stored only >= 0.55 peaks.
        run_floor = manifest.get('detection_storage_floor', 0.55)
        if run_floor != DETECTION_STORAGE_FLOOR:
            sys.exit(
                f"❌ Run '{run_dir.name}' stores detections down to confidence {run_floor}; "
                f"this code stores down to {DETECTION_STORAGE_FLOOR}.\n"
                f"   Appending would mix confidence semantics in one results.jsonl.\n"
                f"   Use a new --name for this area (or run the code version that matches)."
            )
        return manifest

    manifest = {
        'run_name': run_dir.name,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_geojson': str(geojson_path),
        'area_hash': area_hash,
        'imagery_source': source_name,
        'model_id': MODEL_ID,
        'model_training_date': MODEL_TRAINING_DATE,
        'api_version': API_VERSION,
        'detection_storage_floor': DETECTION_STORAGE_FLOOR,
        'max_peaks_per_pano': MAX_PEAKS_PER_PANO,
        'streetlevel_version': pkg_version('streetlevel'),
        'runs': [],
    }
    with open(run_dir / "area.geojson", 'w') as f:
        geojson.dump(geojson_data, f)
    save_manifest(manifest_path, manifest)
    return manifest

def record_run(manifest_path, manifest, started_at, found, success, skipped, failed, phase=None):
    """Appends one entry to the manifest's run history."""
    entry = {
        'started_at': started_at,
        'finished_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'panos_found_in_area': found,
        'processed': success,
        'skipped': skipped,
        'failed': failed,
    }
    if phase:
        entry['phase'] = phase
    manifest['runs'].append(entry)
    save_manifest(manifest_path, manifest)

def dangling_link_targets(results_path, processed_ids):
    """
    Link-target pano ids that results.jsonl records reference but the run never
    processed — panos the run's own view graph points at that we don't have (mostly
    coverage churn between the tile scan and the per-pano pass; issue #32).
    processed_ids is the resume cache, which covers deterministic skips, so cached
    indoor/outside-the-area targets don't reappear. The records' own ids are
    subtracted too — a run pulled from a cluster has results.jsonl but no cache.
    """
    targets = set()
    have = set(processed_ids)
    with open(results_path, 'r') as f:
        for line in f:
            pano = json.loads(line)['pano']
            have.add(pano['panorama_id'])
            for link in pano.get('links') or []:
                target = link.get('target_gsv_panorama_id')
                if target:
                    targets.add(target)
    return targets - have

def run_gap_fill(source, area_shape, run_dir, scan_only=False, limit=None):
    """
    Post-run link-graph closure (issue #32): fetches the panos that
    dangling_link_targets finds, keeping those whose own position falls inside the
    area, and appends them to results.jsonl/cache exactly like main-pass panos.
    Iterates until closed, since each new record introduces new links (round 2 is
    normally near-empty). Returns (candidates, success, skipped, failed) totals,
    or None if nothing ran.
    """
    results_path = run_dir / "results.jsonl"
    cache_file = run_dir / "already_processed.txt"
    if not results_path.exists():
        print("-> Gap fill: no results.jsonl yet — nothing to close.")
        return None

    # Reload rather than track: the main pass appends to the cache file directly.
    processed_ids = load_processed_ids(cache_file)
    attempted = set()  # failures stay uncached (retryable next run) but must not loop now
    totals = [0, 0, 0, 0]  # candidates, success, skipped, failed
    while True:
        candidates = sorted(dangling_link_targets(results_path, processed_ids | attempted))
        if scan_only:
            print(f"-> Gap fill scan: {len(candidates)} dangling link targets to try "
                  f"(in-area count unknown until their metadata is fetched).")
            return None
        if limit is not None:
            candidates = candidates[:max(0, limit - totals[0])]
        if not candidates:
            break
        totals[0] += len(candidates)

        counts = {'success': 0, 'skipped': 0, 'failed': 0}
        with open(cache_file, 'a') as f_cache, \
             open(results_path, 'a') as f_jsonl, \
             ThreadPoolExecutor(max_workers=PROCESSING_CONCURRENCY) as pool:
            futures = [pool.submit(process_gap_pano, source, pid, area_shape) for pid in candidates]
            with tqdm(total=len(futures), desc="Gap-filling Link Targets") as pbar:
                for future in as_completed(futures):
                    counts[handle_result(future.result(), f_cache, f_jsonl)] += 1
                    pbar.update(1)

        attempted.update(candidates)
        totals[1] += counts['success']
        totals[2] += counts['skipped']
        totals[3] += counts['failed']
        print(f"-> Gap fill: {counts['success']} added, {counts['skipped']} skipped "
              f"(outside area/indoor), {counts['failed']} failed.")
    return tuple(totals)

def run_labeler(geojson_path, run_name, source, scan_only=False, limit=None, thin_spacing=None,
                gap_fill=True, gap_fill_only=False):
    """
    Finds and processes all panoramas from the given imagery source within a GeoJSON
    area, writing all per-area state to runs/<run_name>/.

    With scan_only=True, stops after the coverage scan and prints a size/runtime
    estimate — use this to scope a city before committing to a multi-day run.

    After the main pass, sources with a link graph get a gap-fill phase closing it
    (issue #32); gap_fill_only skips straight to that phase on an existing run.
    """
    print("--- Sidewalk Auto-Labeler ---")

    # 1. Load GeoJSON and set up the run directory
    print(f"-> Loading GeoJSON from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        geojson_data = geojson.load(f)

    area_hash = get_geojson_hash(geojson_data)
    started_at = datetime.now(timezone.utc).isoformat(timespec='seconds')

    run_dir = Path("runs") / run_name
    manifest = load_or_init_run_dir(run_dir, geojson_path, geojson_data, area_hash, source.NAME)
    manifest_path = run_dir / "manifest.json"
    output_jsonl_file = run_dir / "results.jsonl"
    cache_file = run_dir / "already_processed.txt"

    print(f"-> Run directory: {run_dir}")
    print(f"-> Imagery source: {source.NAME}")
    print(f"-> Area hash: {area_hash}")

    processed_ids = load_processed_ids(cache_file)
    print(f"-> Found {len(processed_ids)} already processed panoramas in cache.")

    # 2. Find all panorama IDs in the area
    # The geojson_data is the geometry object itself, which shapely can read directly.
    area_shape = shape(geojson_data)

    if gap_fill_only:
        if not hasattr(source, 'fetch_pano_by_id'):
            sys.exit(f"❌ --gap-fill-only: source '{source.NAME}' has no by-id fetch "
                     f"(its records carry no link graph, so there is nothing to close).")
        gf = run_gap_fill(source, area_shape, run_dir, scan_only=scan_only, limit=limit)
        if gf is not None:
            record_run(manifest_path, manifest, started_at, gf[0], gf[1], gf[2], gf[3],
                       phase='gap_fill')
            print(f"\n--- Gap Fill Report ---\n"
                  f"Dangling link targets tried: {gf[0]}\n"
                  f"Added to the run:            {gf[1]}\n"
                  f"Skipped (outside/indoor):    {gf[2]}\n"
                  f"Failed (will retry):         {gf[3]}\n"
                  f"-----------------------")
        return

    bounds = area_shape.bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    top_left_x, top_left_y = latlon_to_tile(max_lat, min_lon, source.COVERAGE_TILE_ZOOM)
    bottom_right_x, bottom_right_y = latlon_to_tile(min_lat, max_lon, source.COVERAGE_TILE_ZOOM)

    tiles_to_scan = [(x, y) for x in range(top_left_x, bottom_right_x + 1) for y in range(top_left_y, bottom_right_y + 1)]

    print(f"-> Scanning {len(tiles_to_scan)} coverage tiles using {COVERAGE_API_CONCURRENCY} concurrent workers...")

    all_panos_in_area = {}

    failed_tiles = 0
    with ThreadPoolExecutor(max_workers=COVERAGE_API_CONCURRENCY) as find_pool:
        futures = [find_pool.submit(source.fetch_panos_for_tile, x, y, area_shape) for x, y in tiles_to_scan]
        with tqdm(total=len(futures), desc="Finding Panoramas") as pbar:
            for future in as_completed(futures):
                tile_panos = future.result()
                if tile_panos is None:
                    failed_tiles += 1
                else:
                    all_panos_in_area.update(tile_panos)
                pbar.update(1)

    if failed_tiles:
        print(f"⚠ {failed_tiles} coverage tiles failed after retries — panos there are "
              f"missing from this run. A re-run rescans all tiles and picks them up.")

    # Optional per-source spatial thinning (e.g. Mapillary's near-duplicate coverage).
    if hasattr(source, 'thin_panos') and all_panos_in_area and thin_spacing != 0:
        spacing = thin_spacing or source.THIN_CELL_METERS
        found_before_thinning = len(all_panos_in_area)
        all_panos_in_area = source.thin_panos(all_panos_in_area, spacing)
        print(f"-> Spatial thinning: {found_before_thinning} → {len(all_panos_in_area)} panos "
              f"(best per ~{spacing} m grid cell).")

    # 3. Determine which panoramas to process
    panos_to_process_ids = sorted(list(set(all_panos_in_area.keys()) - processed_ids))
    if limit is not None and len(panos_to_process_ids) > limit:
        print(f"-> --limit {limit}: processing only the first {limit} of {len(panos_to_process_ids)} new panoramas.")
        panos_to_process_ids = panos_to_process_ids[:limit]

    print("\n--- Processing Summary ---")
    print(f"Total panoramas found in area: {len(all_panos_in_area)}")
    print(f"Already processed (skipped):   {len(processed_ids)}")
    print(f"New panoramas to process:      {len(panos_to_process_ids)}")
    print("--------------------------\n")

    if scan_only:
        # ~1.5 s/pano is the measured single-GPU steady state (RTX 3070, concurrency 10+);
        # downloads overlap but inference is serialized, so the GPU sets the rate.
        est_hours = len(panos_to_process_ids) * 1.5 / 3600
        print("Scan only — no panoramas processed, nothing recorded in the manifest.")
        print(f"Estimated full-run time: ~{est_hours:.1f} h at 1.5 s/pano "
              f"(single GPU; measure your machine's rate on a smoke run first).")
        return

    # 4. Process new panoramas
    success_count, skip_count, fail_count = 0, 0, 0

    if not panos_to_process_ids:
        print("🎉 No new panoramas to process.")
        record_run(manifest_path, manifest, started_at, len(all_panos_in_area), 0, 0, 0)
    else:
        processing_tasks = [
            (pid, all_panos_in_area[pid][0], all_panos_in_area[pid][1])
            for pid in panos_to_process_ids
        ]

        with open(cache_file, 'a') as f_cache, \
             open(output_jsonl_file, 'a') as f_jsonl, \
             ThreadPoolExecutor(max_workers=PROCESSING_CONCURRENCY) as process_pool:

            futures = [process_pool.submit(process_pano, source, *task) for task in processing_tasks]

            with tqdm(total=len(futures), desc="Processing New Panoramas") as pbar:
                for future in as_completed(futures):
                    outcome = handle_result(future.result(), f_cache, f_jsonl)
                    if outcome == 'success':
                        success_count += 1
                    elif outcome == 'skipped':
                        skip_count += 1
                    else:
                        fail_count += 1
                    pbar.update(1)

        record_run(manifest_path, manifest, started_at, len(all_panos_in_area), success_count, skip_count, fail_count)

    # 5. Close the link graph (issue #32): fetch in-area panos the new records
    # reference but the scan never enumerated (mostly coverage churn). Done promptly,
    # inside the same run, so the filled panos match the run's imagery vintage.
    # A --limit budget is shared with the main pass so smoke runs stay small.
    gf = None
    if gap_fill and hasattr(source, 'fetch_pano_by_id'):
        gap_limit = None if limit is None else max(0, limit - len(panos_to_process_ids))
        if gap_limit == 0:
            print("-> Gap fill: skipped, --limit budget consumed by the main pass.")
        else:
            gap_started_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
            gf = run_gap_fill(source, area_shape, run_dir, limit=gap_limit)
            if gf is not None:
                record_run(manifest_path, manifest, gap_started_at, gf[0], gf[1], gf[2], gf[3],
                           phase='gap_fill')

    print("\n--- Final Report ---")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (cached):       {skip_count}")
    print(f"Failed to process:      {fail_count}")
    if gf is not None:
        print(f"Gap fill (link graph):  {gf[1]} added of {gf[0]} dangling targets "
              f"({gf[2]} outside/skipped, {gf[3]} failed)")
    print(f"Results saved to: {output_jsonl_file}")
    print(f"Export benchmark bundle (imagery for RampNet GT/scoring): python scripts/export_benchmark.py {output_jsonl_file} --out <dir>")
    print("----------------------")


def main():
    global PROCESSING_CONCURRENCY, COVERAGE_API_CONCURRENCY

    parser = argparse.ArgumentParser(
        description="Finds and processes all street-level panoramas within a GeoJSON area, saving results to a .jsonl file."
    )
    parser.add_argument(
        "geojson_file",
        help="Path to the GeoJSON file defining the area of interest."
    )
    parser.add_argument(
        "--name",
        help="Run name; all per-area state (results, cache, manifest) lives in runs/<name>/ "
             "(default: the geojson filename without extension)."
    )
    parser.add_argument(
        "--source", choices=SOURCE_NAMES, default="gsv",
        help="Imagery source to scan and fetch from (default: %(default)s). "
             "'mapillary' needs a client token in MAPILLARY_ACCESS_TOKEN."
    )
    parser.add_argument(
        "--processing-concurrency", type=int, default=PROCESSING_CONCURRENCY,
        help="Concurrent workers downloading and detecting panos (default: %(default)s). "
             "Lower this if Google starts dropping connections."
    )
    parser.add_argument(
        "--coverage-concurrency", type=int, default=COVERAGE_API_CONCURRENCY,
        help="Concurrent workers scanning coverage tiles (default: %(default)s)."
    )
    parser.add_argument(
        "--thin-spacing", type=int, default=None, metavar="METERS",
        help="Spatial thinning grid size in meters for sources with near-duplicate "
             "coverage (default: the source's own, e.g. Mapillary 5 m); 0 disables "
             "thinning. No effect on sources without a thinning hook (GSV)."
    )
    parser.add_argument(
        "--limit", type=int,
        help="Process at most N new panoramas this run (for smoke tests and rate "
             "measurement); the rest stay uncached and process on a later run."
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Only scan coverage and report the pano count and a runtime estimate; "
             "skips model loading and processes nothing. With --gap-fill-only, "
             "reports the dangling link-target count instead."
    )
    gap_group = parser.add_mutually_exclusive_group()
    gap_group.add_argument(
        "--no-gap-fill", action="store_true",
        help="Skip the post-run link-graph closure phase (issue #32)."
    )
    gap_group.add_argument(
        "--gap-fill-only", action="store_true",
        help="Skip the coverage scan and main pass; only close the link graph of an "
             "existing run (fetch in-area panos its records reference but it never "
             "processed). For retrofits and smoke tests."
    )
    args = parser.parse_args()

    PROCESSING_CONCURRENCY = args.processing_concurrency
    COVERAGE_API_CONCURRENCY = args.coverage_concurrency

    # Fail fast on source misconfiguration (e.g. missing Mapillary token) before the
    # slow model load below.
    source = get_source(args.source)
    source.prepare()

    # Initialize detectors (skipped for a scan: importing torch + loading the model takes a while):
    if not args.scan_only:
        from detectors.curb_ramp import CurbRampDetector
        global curb_ramp_detector
        curb_ramp_detector = CurbRampDetector()

    try:
        run_labeler(args.geojson_file, args.name or Path(args.geojson_file).stem, source, args.scan_only,
                    args.limit, args.thin_spacing,
                    gap_fill=not args.no_gap_fill, gap_fill_only=args.gap_fill_only)
    except FileNotFoundError:
        print(f"❌ Error: The file '{args.geojson_file}' was not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
