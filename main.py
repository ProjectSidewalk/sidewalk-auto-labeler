from gevent import monkey
monkey.patch_all()
from gevent.pool import Pool

import argparse
import math
import hashlib
import json
import random
import socket
import sys
import time
import traceback
import os
from pathlib import Path

# streetlevel makes HTTP requests with no timeout; without this, one black-holed
# connection (e.g. Google throttling) hangs its worker greenlet forever.
socket.setdefaulttimeout(30)

# Progress/report messages use emoji; on Windows the console may be cp1252, where printing
# them raises UnicodeEncodeError and would abort the run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

import geojson
from streetlevel import streetview
from shapely.geometry import shape, Point
from tqdm import tqdm

from panorama import fetch_panorama
from detectors.curb_ramp import CurbRampDetector

# --- Configuration ---
COVERAGE_API_CONCURRENCY = 100
PROCESSING_CONCURRENCY = 50
COVERAGE_TILE_ZOOM = 17
METADATA_ATTEMPTS = 3


def fetch_metadata_with_retry(pano_id):
    """
    Fetches pano metadata, retrying with backoff: Google's metadata endpoint
    intermittently returns empty responses (see sk-zk/streetlevel#40).
    Returns None if all attempts fail.
    """
    for attempt in range(METADATA_ATTEMPTS):
        metadata = streetview.find_panorama_by_id(pano_id)
        if metadata is not None:
            return metadata
        if attempt < METADATA_ATTEMPTS - 1:
            time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


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

def fetch_panos_for_tile(tile_x, tile_y, area_shape):
    """
    Worker function for a single tile. Fetches panos and filters them.
    Returns a dictionary of {pano_id: (lat, lon)} for valid panos.
    """
    try:
        panos_in_tile = streetview.get_coverage_tile(tile_x, tile_y)
        valid_panos = {p.id: (p.lat, p.lon) for p in panos_in_tile if Point(p.lon, p.lat).within(area_shape)}
        return valid_panos
    except Exception:
        return {}

def process_pano(pano_id, lat, lon):
    """
    Downloads image and metadata, runs detection on, and returns results for a single panorama.
    Designed to be run concurrently in a gevent pool.
    """
    try:
        # Get metadata. A None result can be transient (Google's metadata endpoint
        # intermittently returns empty responses), so retry here and treat exhaustion
        # as a retryable failure, not a cacheable skip.
        metadata = fetch_metadata_with_retry(pano_id)
        if metadata is None:
            return {'status': 'failure', 'pano_id': pano_id, 'reason': 'Metadata unavailable (transient?)'}
        if metadata.source in ['innerspace', 'cultural_institute', 'photos:legacy_innerspace']:
            return {'status': 'skipped', 'pano_id': pano_id, 'reason': 'Indoor panorama source'}

        # Download the image (streetlevel stitches it using the metadata's tile grid).
        equi = fetch_panorama(metadata)
        if equi is None:
            return {'status': 'failure', 'pano_id': pano_id, 'reason': 'Failed to download equirectangular image'}

        # The detector returns the list of center points and confidence: [[x, y, confidence], ...] or an empty list [].
        detections = curb_ramp_detector.detect(equi)

        return {
            'status': 'success',
            'pano_id': pano_id,
            'lat': float(lat),
            'lon': float(lon),
            'metadata': metadata,
            'detections': detections
        }
    except Exception as e:
        return {'status': 'failure', 'pano_id': pano_id, 'reason': str(e)}

class IncompleteMetadataError(Exception):
    """Raised when a pano's metadata lacks fields required by the output record."""

def build_output_line(result):
    """
    Builds one JSONL record for a successfully processed panorama.

    Raises IncompleteMetadataError when the metadata is missing fields the record
    (and the Project Sidewalk endpoint) requires; such panos are deterministic
    skips and should be cached rather than retried.
    """
    metadata = result['metadata']
    if metadata.date is None or not metadata.image_sizes or metadata.tile_size is None:
        raise IncompleteMetadataError('Pano metadata missing date, image sizes, or tile size')

    return {
        "detections": [
            {
                "x_normalized": x_normalized,
                "y_normalized": y_normalized,
                "confidence": confidence
            } for x_normalized, y_normalized, confidence in result['detections']
        ],
        "label_type": "CurbRamp",
        "model_id": "rampnet-model",
        "model_training_date": "08-21-2025",
        "api_version": "1.0.0",
        "pano": {
            "panorama_id": result['pano_id'],
            "capture_date": f"{metadata.date.year}-{metadata.date.month:02d}",
            "width": metadata.image_sizes[-1].x,
            "height": metadata.image_sizes[-1].y,
            "tile_width": metadata.tile_size.x,
            "tile_height": metadata.tile_size.y,
            "lat": result['lat'],
            "lng": result['lon'],
            "camera_heading": math.degrees(metadata.heading),
            "camera_pitch": math.degrees(metadata.pitch),
            "camera_roll": math.degrees(metadata.roll),
            "copyright": metadata.copyright_message,
            "source": metadata.source,
            "history": [
                {
                    "pano_id": old_pano.id,
                    "date": f"{old_pano.date.year}-{old_pano.date.month:02d}"
                } for old_pano in (metadata.historical or []) if old_pano.date is not None
            ],
            "links": [
                {
                    "target_gsv_panorama_id": linked_pano.pano.id,
                    "yaw_deg": math.degrees(linked_pano.direction),
                    "description": linked_pano.pano.address[0].value if linked_pano.pano.address else ""
                } for linked_pano in (metadata.links or [])
            ]
        },
    }

def run_labeler(geojson_path):
    """
    Finds and processes all GSV panoramas within a GeoJSON area,
    outputting results to a .jsonl file and using a cache.
    """
    print("--- Sidewalk Auto-Labeler ---")

    # 1. Load GeoJSON and set up cache and output paths
    print(f"-> Loading GeoJSON from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        geojson_data = geojson.load(f)

    area_hash = get_geojson_hash(geojson_data)

    output_jsonl_file = Path(f"{os.path.splitext(os.path.basename(geojson_path))[0]}.jsonl")
    cache_dir = Path("cache") / area_hash
    cache_file = cache_dir / "already_processed.txt"

    print(f"-> Area hash: {area_hash}")
    print(f"-> Output file: {output_jsonl_file}")
    print(f"-> Cache file:  {cache_file}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(cache_file)
    print(f"-> Found {len(processed_ids)} already processed panoramas in cache.")

    # 2. Find all panorama IDs in the area
    # The geojson_data is the geometry object itself, which shapely can read directly.
    area_shape = shape(geojson_data)
    bounds = area_shape.bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    top_left_x, top_left_y = latlon_to_tile(max_lat, min_lon, COVERAGE_TILE_ZOOM)
    bottom_right_x, bottom_right_y = latlon_to_tile(min_lat, max_lon, COVERAGE_TILE_ZOOM)

    tiles_to_scan = [(x, y) for x in range(top_left_x, bottom_right_x + 1) for y in range(top_left_y, bottom_right_y + 1)]

    print(f"-> Scanning {len(tiles_to_scan)} coverage tiles using {COVERAGE_API_CONCURRENCY} concurrent workers...")

    find_pool = Pool(COVERAGE_API_CONCURRENCY)
    all_panos_in_area = {}

    tasks = [(x, y, area_shape) for x, y in tiles_to_scan]

    with tqdm(total=len(tasks), desc="Finding Panoramas") as pbar:
        def find_worker_wrapper(args): return fetch_panos_for_tile(*args)
        for pano_dict in find_pool.imap_unordered(find_worker_wrapper, tasks):
            all_panos_in_area.update(pano_dict)
            pbar.update(1)

    # 3. Determine which panoramas to process
    panos_to_process_ids = sorted(list(set(all_panos_in_area.keys()) - processed_ids))

    print("\n--- Processing Summary ---")
    print(f"Total panoramas found in area: {len(all_panos_in_area)}")
    print(f"Already processed (skipped):   {len(processed_ids)}")
    print(f"New panoramas to process:      {len(panos_to_process_ids)}")
    print("--------------------------\n")

    if not panos_to_process_ids:
        print("🎉 No new panoramas to process. All done!")
        return

    # 4. Process new panoramas
    success_count, skip_count, fail_count = 0, 0, 0

    process_pool = Pool(PROCESSING_CONCURRENCY)
    processing_tasks = [
        (pid, all_panos_in_area[pid][0], all_panos_in_area[pid][1])
        for pid in panos_to_process_ids
    ]

    with open(cache_file, 'a') as f_cache, \
         open(output_jsonl_file, 'a') as f_jsonl:

        def process_worker_wrapper(args): return process_pano(*args)

        with tqdm(total=len(processing_tasks), desc="Processing New Panoramas") as pbar:
            for result in process_pool.imap_unordered(process_worker_wrapper, processing_tasks):
                if result['status'] == 'success':
                    # ALWAYS write a line to the JSONL file for a successful process.
                    # The 'detections' key will be an empty list [] if none were found.
                    # Guarded so a single malformed pano cannot abort the whole run.
                    try:
                        output_line = build_output_line(result)
                    except IncompleteMetadataError:
                        # Deterministic skip: cache it so it isn't retried on every run.
                        f_cache.write(f"{result['pano_id']}\n")
                        f_cache.flush()
                        skip_count += 1
                    except Exception as e:
                        print(f"  ❌ Failed to build output for {result['pano_id']}. Reason: {e}. Will retry on next run.")
                        fail_count += 1
                    else:
                        f_jsonl.write(json.dumps(output_line) + '\n')
                        f_jsonl.flush()

                        # Mark successfully processed pano in the cache.
                        f_cache.write(f"{result['pano_id']}\n")
                        f_cache.flush()
                        success_count += 1
                elif result['status'] == 'skipped':
                    # Deterministic skips (indoor pano): cache so they aren't refetched.
                    f_cache.write(f"{result['pano_id']}\n")
                    f_cache.flush()
                    skip_count += 1
                else:
                    print(f"  ❌ Failed to process {result['pano_id']}. Reason: {result.get('reason', 'Unknown')}. Will retry on next run.")
                    fail_count += 1
                pbar.update(1)

    print("\n--- Final Report ---")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (cached):       {skip_count}")
    print(f"Failed to process:      {fail_count}")
    print(f"Results saved to: ./{output_jsonl_file}")
    print("----------------------")


def main():
    global PROCESSING_CONCURRENCY, COVERAGE_API_CONCURRENCY

    parser = argparse.ArgumentParser(
        description="Finds and processes all GSV panoramas within a GeoJSON area, saving results to a .jsonl file."
    )
    parser.add_argument(
        "geojson_file",
        help="Path to the GeoJSON file defining the area of interest."
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
    args = parser.parse_args()

    PROCESSING_CONCURRENCY = args.processing_concurrency
    COVERAGE_API_CONCURRENCY = args.coverage_concurrency

    # Initialize detectors:
    global curb_ramp_detector
    curb_ramp_detector = CurbRampDetector()

    try:
        run_labeler(args.geojson_file)
    except FileNotFoundError:
        print(f"❌ Error: The file '{args.geojson_file}' was not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
