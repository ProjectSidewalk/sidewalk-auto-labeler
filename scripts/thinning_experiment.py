"""
Thinning-assumption experiment: does denser Mapillary sampling actually improve
curb-ramp coverage, and does camera proximity improve detection?

Assumptions under test (from the 2026-07 Richmond spike, see sources/mapillary.py):
  A1. Detection is more reliable the closer the camera is to the ramp.
  A2. Denser sampling (smaller --thin-spacing) finds more distinct ramps, with
      diminishing returns; PS-style clustering absorbs the duplicate labels.
  A3. thin_panos' newest-capture/quality selection loses no more ramps than a
      random selection of the same size.

Protocol:
  1. Choose a compact sub-area (a few blocks, ~1-2k raw panos) and save it as a
     bare-geometry geojson, e.g. example_geojson/richmond_thinexp.geojson.
  2. Run detection UN-thinned over it (the only GPU step):
       python main.py example_geojson/richmond_thinexp.geojson --name thinexp \
           --source mapillary --thin-spacing 0
  3. Analyze (offline apart from a cheap coverage rescan):
       python scripts/thinning_experiment.py runs/thinexp
  4. Read runs/thinexp/thinning_experiment/report.md (+ CSVs for plotting).

Method:
  - Every detection is projected to an estimated ground point: flat-ground
    projection with distance = camera_height / tan(angle below horizon) from
    y_normalized, bearing = camera_heading + (x_normalized - 0.5) * 360.
  - Ground points are greedily clustered within --cluster-radius (default 7.5 m,
    approximating PS label clustering) into pseudo-ground-truth "ramp sites".
  - A2: for each candidate spacing, select the panos thin_panos would keep and
    report panos kept, ramp sites retained (>=1 member detection from a kept
    pano), and estimated GPU hours — the coverage-vs-cost curve.
  - A3: compare against random same-count pano selections (mean of N seeds).
  - A1: among (pano, site) pairs within --opportunity-radius of a site, bin the
    fraction that produced a member detection by camera-to-site distance.

Caveats: the projection assumes flat ground and a fixed camera height, so
distances are approximate (fine for binned/relative comparisons); ramp sites are
model-derived, so results measure detection coverage, not true recall — validate
a sample with the spot-check gallery before leaning on precision claims.
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import main as labeler
from sources import mapillary

CAMERA_HEIGHT_M = 2.6      # typical roof-mounted 360 rig
MAX_GROUND_DIST_M = 30.0   # beyond this the flat-ground estimate is mush
SECONDS_PER_PANO = 1.5     # measured single-GPU rate (see main.py)
METERS_PER_DEG_LAT = 111320.0


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get('imagery_source') != 'mapillary':
        sys.exit("This experiment is about Mapillary thinning; run it on a --source mapillary run.")
    with open(run_dir / "results.jsonl", 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    area = shape(json.loads((run_dir / "area.geojson").read_text()))
    return run_dir, records, area


def rescan_coverage(area):
    """Re-enumerates the area's tiles for exact per-pano (lat, lon, captured_at,
    quality_score) — the attributes thin_panos selects on."""
    min_lon, min_lat, max_lon, max_lat = area.bounds
    x0, y0 = labeler.latlon_to_tile(max_lat, min_lon, mapillary.COVERAGE_TILE_ZOOM)
    x1, y1 = labeler.latlon_to_tile(min_lat, max_lon, mapillary.COVERAGE_TILE_ZOOM)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    panos = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(mapillary.fetch_panos_for_tile, x, y, area) for x, y in tiles]
        for future in as_completed(futures):
            tile_panos = future.result()
            if tile_panos is None:
                sys.exit("A coverage tile failed after retries; re-run the experiment.")
            panos.update(tile_panos)
    return panos


def ground_point(pano, det):
    """Flat-ground estimate of a detection's (lat, lon), or None if at/above horizon."""
    below_horizon = (det['y_normalized'] - 0.5) * math.pi  # radians; equirect y=0.5 is the horizon
    if below_horizon <= 0.02:
        return None
    dist = min(CAMERA_HEIGHT_M / math.tan(below_horizon), MAX_GROUND_DIST_M)
    bearing = math.radians(pano['camera_heading'] + (det['x_normalized'] - 0.5) * 360.0)
    lat = pano['lat'] + dist * math.cos(bearing) / METERS_PER_DEG_LAT
    lon = pano['lng'] + dist * math.sin(bearing) / (METERS_PER_DEG_LAT * math.cos(math.radians(pano['lat'])))
    return lat, lon


def meters_between(a, b):
    dlat = (a[0] - b[0]) * METERS_PER_DEG_LAT
    dlon = (a[1] - b[1]) * METERS_PER_DEG_LAT * math.cos(math.radians(a[0]))
    return math.hypot(dlat, dlon)


def cluster_detections(records, radius_m):
    """Greedy centroid clustering of projected detections into ramp sites.
    Returns a list of {'centroid': (lat, lon), 'members': {pano_id, ...}}."""
    projected = []  # (confidence, point, pano_id)
    for record in records:
        pano = record['pano']
        for det in record['detections']:
            point = ground_point(pano, det)
            if point is not None:
                projected.append((det['confidence'], point, pano['panorama_id']))
    projected.sort(reverse=True)  # high-confidence detections seed clusters

    sites = []
    for _, point, pano_id in projected:
        for site in sites:
            if meters_between(point, site['centroid']) <= radius_m:
                n = site['n']
                site['centroid'] = ((site['centroid'][0] * n + point[0]) / (n + 1),
                                    (site['centroid'][1] * n + point[1]) / (n + 1))
                site['n'] = n + 1
                site['members'].add(pano_id)
                break
        else:
            sites.append({'centroid': point, 'n': 1, 'members': {pano_id}})
    return sites


def sites_retained(sites, kept_pano_ids):
    return sum(1 for site in sites if site['members'] & kept_pano_ids)


def spacing_curve(sites, scan, spacings, random_seeds):
    rows = []
    for spacing in spacings:
        kept = set(scan) if spacing == 0 else set(mapillary.thin_panos(scan, spacing))
        random_retained = []
        for seed in range(random_seeds):
            sample = set(random.Random(seed).sample(sorted(scan), len(kept)))
            random_retained.append(sites_retained(sites, sample))
        rows.append({
            'spacing_m': spacing,
            'panos_kept': len(kept),
            'gpu_hours': round(len(kept) * SECONDS_PER_PANO / 3600, 2),
            'sites_retained': sites_retained(sites, kept),           # A2
            'sites_retained_random_mean': round(sum(random_retained) / len(random_retained), 1),  # A3
            'sites_total': len(sites),
        })
    return rows


def distance_bins(sites, records, opportunity_radius, bin_edges):
    """A1: among panos within opportunity_radius of a site, detection rate by distance."""
    pano_pos = {r['pano']['panorama_id']: (r['pano']['lat'], r['pano']['lng']) for r in records}
    hits = defaultdict(int)
    opportunities = defaultdict(int)
    for site in sites:
        for pano_id, pos in pano_pos.items():
            dist = meters_between(pos, site['centroid'])
            if dist > opportunity_radius:
                continue
            for lo, hi in zip(bin_edges, bin_edges[1:]):
                if lo <= dist < hi:
                    opportunities[(lo, hi)] += 1
                    hits[(lo, hi)] += pano_id in site['members']
                    break
    return [{'bin_m': f"{lo}-{hi}", 'opportunities': opportunities[(lo, hi)],
             'detection_rate': round(hits[(lo, hi)] / opportunities[(lo, hi)], 3) if opportunities[(lo, hi)] else None}
            for lo, hi in zip(bin_edges, bin_edges[1:])]


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(rows[0].keys()) + '\n')
        for row in rows:
            f.write(','.join(str(v) for v in row.values()) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Analyze an un-thinned Mapillary run for the thinning experiment.")
    parser.add_argument("run_dir", help="Run directory of an UN-thinned (--thin-spacing 0) Mapillary run.")
    parser.add_argument("--spacings", type=int, nargs='+', default=[0, 3, 5, 8, 10, 15, 20])
    parser.add_argument("--cluster-radius", type=float, default=7.5,
                        help="Meters within which detections merge into one ramp site (default %(default)s).")
    parser.add_argument("--opportunity-radius", type=float, default=20.0)
    parser.add_argument("--random-seeds", type=int, default=20)
    args = parser.parse_args()

    run_dir, records, area = load_run(args.run_dir)
    print(f"-> {len(records)} processed panos; rescanning coverage for thinning attributes...")
    scan = rescan_coverage(area)
    # Only panos that were actually processed can retain sites; warn if the run looks partial.
    processed = {r['pano']['panorama_id'] for r in records}
    missing = len([p for p in scan if p not in processed])
    if missing:
        print(f"⚠ {missing} scanned panos have no results record (partial or thinned run?) — "
              f"curves will understate dense-sampling coverage.")

    sites = cluster_detections(records, args.cluster_radius)
    print(f"-> {sum(len(r['detections']) for r in records)} detections → {len(sites)} ramp sites "
          f"(cluster radius {args.cluster_radius} m).")

    curve = spacing_curve(sites, scan, args.spacings, args.random_seeds)
    bins = distance_bins(sites, records, args.opportunity_radius, [0, 4, 8, 12, 16, 20])

    out_dir = run_dir / "thinning_experiment"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / "spacing_curve.csv", curve)
    write_csv(out_dir / "distance_bins.csv", bins)

    def table(rows):
        header = '| ' + ' | '.join(rows[0].keys()) + ' |'
        sep = '|' + '---|' * len(rows[0])
        return '\n'.join([header, sep] + ['| ' + ' | '.join(str(v) for v in row.values()) + ' |' for row in rows])

    (out_dir / "report.md").write_text(
        f"# Thinning experiment — {run_dir.name}\n\n"
        f"{len(records)} panos processed un-thinned; {len(sites)} model-derived ramp sites "
        f"(cluster radius {args.cluster_radius} m). See the module docstring for assumptions A1-A3 and caveats.\n\n"
        f"## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)\n\n{table(curve)}\n\n"
        f"## A1 — detection rate by camera-to-site distance\n\n{table(bins)}\n",
        encoding='utf-8')
    print(f"-> Wrote {out_dir / 'report.md'} (+ CSVs).")


if __name__ == "__main__":
    main()
