from gevent import monkey
monkey.patch_all()
from gevent.pool import Pool

import argparse
import math
import hashlib
import json
import traceback
import os
from pathlib import Path

import geojson
from streetlevel import streetview
from shapely.geometry import shape, Point
from tqdm import tqdm
import cv2
from PIL import Image

from panorama import fetch_panorama
from detectors.curb_ramp import CurbRampDetector

# --- Configuration ---
COVERAGE_API_CONCURRENCY = 100
PROCESSING_CONCURRENCY = 50
COVERAGE_TILE_ZOOM = 17


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
        # Get metadata. Skip if indoor panorama.
        metadata = streetview.find_panorama_by_id(pano_id)
        if metadata is None or metadata.source in ['innerspace', 'cultural_institute', 'photos:legacy_innerspace']:
            return {'status': 'skipped', 'pano_id': pano_id, 'reason': 'Indoor panorama source'}

        # Download the image.
        equi = fetch_panorama(pano_id)
        if equi is None:
            return {'status': 'failure', 'pano_id': pano_id, 'reason': 'Failed to download equirectangular image'}

        equi_rgb = cv2.cvtColor(equi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(equi_rgb)

        # The detector returns the list of center points and confidence: [[x, y, confidence], ...] or an empty list [].
        detections = curb_ramp_detector.detect(pil_image)

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
    success_count, fail_count = 0, 0

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
                    output_line = {
                        "pano_id": result['pano_id'],
                        "pano_lat": result['lat'],
                        "pano_lon": result['lon'],
                        "detections": result['detections'],
                        "capture_date": f"{result['metadata'].date.year}-{result['metadata'].date.month:02d}",
                        "copyright": result['metadata'].copyright_message,
                        "camera_heading": result['metadata'].heading,
                        "camera_pitch": result['metadata'].pitch,
                        "tile_width": result['metadata'].tile_size.x,
                        "tile_height": result['metadata'].tile_size.y,
                        "width": result['metadata'].image_sizes[len(result['metadata'].image_sizes) - 1].x,
                        "height": result['metadata'].image_sizes[len(result['metadata'].image_sizes) - 1].y,
                        "history": [
                            {
                                "pano_id": old_pano.id,
                                "date": f"{old_pano.date.year}-{old_pano.date.month:02d}"
                            } for old_pano in result['metadata'].historical
                        ],
                        "links": [
                            {
                                "target_gsv_panorama_id": linked_pano.pano.id,
                                "yaw_deg": math.degrees(linked_pano.direction),
                                "description": linked_pano.pano.address[0].value if linked_pano.pano.address else ""
                            } for linked_pano in result['metadata'].links
                        ]
                    }
                    f_jsonl.write(json.dumps(output_line) + '\n')
                    f_jsonl.flush()

                    # Mark successfully processed pano in the cache.
                    f_cache.write(f"{result['pano_id']}\n")
                    f_cache.flush()
                    success_count += 1
                else:
                    print(f"  ❌ Failed to process {result['pano_id']}. Reason: {result.get('reason', 'Unknown')}. Will retry on next run.")
                    fail_count += 1
                pbar.update(1)

    print("\n--- Final Report ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process:      {fail_count}")
    print(f"Results saved to: ./{output_jsonl_file}")
    print("----------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Finds and processes all GSV panoramas within a GeoJSON area, saving results to a .jsonl file."
    )
    parser.add_argument(
        "geojson_file",
        help="Path to the GeoJSON file defining the area of interest."
    )
    args = parser.parse_args()

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
