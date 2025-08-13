from gevent import monkey
monkey.patch_all()
from gevent.pool import Pool

import argparse
import math
import geojson
import hashlib
import json
import traceback
import os

from streetlevel import streetview
from shapely.geometry import shape, Point
from tqdm import tqdm
import cv2
from PIL import Image

from panorama import Panorama
from detectors.curb_ramp import CurbRampDetector

COVERAGE_TILE_ZOOM = 17
CONCURRENCY_LEVEL = 100


def latlon_to_tile(lat_deg, lon_deg, zoom):
    """Converts latitude and longitude to Slippy Map tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def get_geojson_hash(geojson_data):
    """Creates a stable SHA256 hash of the GeoJSON geometry."""
    geometry = geojson_data['features'][0]['geometry']
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
    Designed to be run concurrently in a gevent pool.
    """
    try:
        panos_in_tile = streetview.get_coverage_tile(tile_x, tile_y)
        
        valid_panos = set()
        for pano in panos_in_tile:
            if Point(pano.lon, pano.lat).within(area_shape):
                valid_panos.add(pano.id)
        return valid_panos
    except Exception:
        return set()

def process_pano(pano_id):
    print(f"  -> Processing {pano_id}...")
    try:
        equi = Panorama(pano_id).get_equi()
        if equi is None:
            return False
        equi = cv2.cvtColor(equi, cv2.COLOR_BGR2RGB)
        equi = Image.fromarray(equi)

        print(curb_ramp_detector.detect(equi))
        print(f"  ✅ Successfully processed {pano_id}")
        return True
    except:
        traceback.print_exc()
        return False

def run_labeler(geojson_path):
    """
    Finds and processes all GSV panoramas within a GeoJSON area,
    using a cache to skip already processed panoramas.
    """
    print("--- Sidewalk Auto-Labeler ---")
    
    # 1. Load GeoJSON and set up cache
    print(f"-> Loading GeoJSON from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        geojson_data = geojson.load(f)
    
    area_hash = get_geojson_hash(geojson_data)
    cache_dir = os.path.join("cache", area_hash)
    cache_file = os.path.join(cache_dir, "already_processed.txt")
    
    print(f"-> Area hash: {area_hash[:12]}...")
    print(f"-> Cache file: {cache_file}")

    os.makedirs(cache_dir, exist_ok=True)
    processed_ids = load_processed_ids(cache_file)
    print(f"-> Found {len(processed_ids)} already processed panoramas in cache.")

    # 2. Find all panorama IDs in the area (CONCURRENTLY)
    area_shape = shape(geojson_data['features'][0]['geometry'])
    bounds = area_shape.bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    top_left_x, top_left_y = latlon_to_tile(max_lat, min_lon, COVERAGE_TILE_ZOOM)
    bottom_right_x, bottom_right_y = latlon_to_tile(min_lat, max_lon, COVERAGE_TILE_ZOOM)
    
    # Create a list of all tile coordinates to scan
    tiles_to_scan = [(x, y) for x in range(top_left_x, bottom_right_x + 1) for y in range(top_left_y, bottom_right_y + 1)]
    
    print(f"-> Scanning {len(tiles_to_scan)} coverage tiles using {CONCURRENCY_LEVEL} concurrent workers...")
    
    # Setup the gevent pool
    pool = Pool(CONCURRENCY_LEVEL)
    all_panos_in_area = set()
    
    # Create argument list for the worker function
    tasks = [(x, y, area_shape) for x, y in tiles_to_scan]

    # Use pool.imap_unordered for efficient processing with a progress bar
    # It yields results as they complete, not in the original order.
    with tqdm(total=len(tiles_to_scan), desc="Finding Panoramas (Concurrent)") as pbar:
        # Using a wrapper function to unpack arguments for imap
        def worker_wrapper(args):
            return fetch_panos_for_tile(*args)

        for pano_id_set in pool.imap_unordered(worker_wrapper, tasks):
            all_panos_in_area.update(pano_id_set)
            pbar.update(1)

    # 3. Determine which panoramas to process and run the loop
    panos_to_process = sorted(list(all_panos_in_area - processed_ids))
    
    print("\n--- Processing Summary ---")
    print(f"Total panoramas found in area: {len(all_panos_in_area)}")
    print(f"Already processed (skipped):   {len(processed_ids)}")
    print(f"New panoramas to process:      {len(panos_to_process)}")
    print("--------------------------\n")

    if not panos_to_process:
        print("🎉 No new panoramas to process. All done!")
        return

    success_count = 0
    fail_count = 0
    
    with open(cache_file, 'a') as f_cache:
        for pano_id in tqdm(panos_to_process, desc="Processing New Panoramas"):
            if process_pano(pano_id):
                f_cache.write(f"{pano_id}\n")
                f_cache.flush()
                success_count += 1
            else:
                print(f"  ❌ Failed to process {pano_id}. Will retry on next run.")
                fail_count += 1

    print("\n--- Final Report ---")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process:      {fail_count}")
    print("----------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Finds and processes all Google Street View panoramas within a GeoJSON area, skipping those already processed."
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

if __name__ == "__main__":
    main()