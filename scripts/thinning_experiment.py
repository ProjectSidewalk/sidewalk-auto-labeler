"""
Thinning-assumption experiment: does denser open-imagery sampling (Mapillary or
Panoramax, the two sources with a thin_panos hook) actually improve curb-ramp
coverage, and does camera proximity improve detection?

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
     (--source panoramax works the same way.)
  3. Analyze (offline: the pre-thinning pano list comes from the run's own scan.json;
     only a run without one falls back to a coverage rescan):
       python scripts/thinning_experiment.py runs/thinexp
     (--min-confidence 0.3 scores the operational tier into thinning_experiment_t0.3/.)
  4. Read runs/thinexp/thinning_experiment/report.md (+ CSVs for plotting).
  Findings for a Bayonne and a Richmond sub-area: docs/thinning-experiment.md.

Method:
  - Every detection at or above --min-confidence (default the benchmark tier 0.55)
    and off the camera rig (detectors.on_camera_rig) is projected to an estimated
    ground point with geo.detection_ground_point (flat ground, 2.6 m, no pose):
    rays beyond its 25 m range are DROPPED, not clamped to a fake range.
  - Ground points are greedily clustered within --cluster-radius (default 7.5 m,
    approximating PS label clustering) into pseudo-ground-truth "ramp sites".
  - A2: for each candidate spacing, select the panos thin_panos would keep and
    report panos kept, ramp sites retained (>=1 member detection from a kept
    pano), estimated GPU hours — the coverage-vs-cost curve — plus two stricter
    columns: "robust" sites (>= 3 member panos at full density, the ones least
    likely to be a one-off false positive) retained, and sites still seen from
    >= 2 kept panos (what multi-view fusion needs).
  - A3: compare against random same-count pano selections (mean of N seeds).
  - A1: among (pano, site) pairs within --opportunity-radius of a site, bin the
    fraction that produced a member detection by camera-to-site distance.

Caveats: the projection assumes flat ground and a fixed camera height, so
distances are approximate (fine for binned/relative comparisons); ramp sites are
model-derived, so results measure detection coverage, not true recall — validate
a sample in RampNet (GT gallery + scorer) before leaning on precision claims.
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
import geo
from detectors import BENCHMARK_CONFIDENCE, on_camera_rig
from sources import get_source

SECONDS_PER_PANO = 1.5     # main.py's planning rate; --seconds-per-pano overrides
METERS_PER_DEG_LAT = geo.METERS_PER_DEG_LAT
ROBUST_MIN_PANOS = 3       # a site seen from >= this many panos at full density
THINNABLE_SOURCES = ('mapillary', 'panoramax')  # the sources with a thin_panos hook


def load_run(run_dir, min_confidence=BENCHMARK_CONFIDENCE):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    source_name = manifest.get('imagery_source')
    if source_name not in THINNABLE_SOURCES:
        sys.exit(f"This experiment is about thinning, which only {THINNABLE_SOURCES} do; "
                 f"{run_dir} is a {source_name!r} run.")
    with open(run_dir / "results.jsonl", 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    # Ramp sites approximate PS clustering of believed labels, so build them from
    # detections at the scored tier only (results.jsonl also carries sub-floor
    # candidates, detectors/__init__.py), minus the camera-rig band production masks.
    for r in records:
        r['detections'] = [d for d in r['detections']
                           if d['confidence'] >= min_confidence
                           and not on_camera_rig(d['y_normalized'])]
    area = shape(json.loads((run_dir / "area.geojson").read_text()))
    return run_dir, records, area, get_source(source_name), source_name


def load_scan(run_dir, source_name):
    """The run's own pre-thinning pano list ({id: (lat, lon, captured, quality)}) from
    scan.json, or None when there is none (or it is for another source, or partial)."""
    path = Path(run_dir) / labeler.SCAN_CACHE_FILE
    if not path.exists():
        return None
    cache = json.loads(path.read_text())
    if cache.get('source') != source_name or cache.get('failed_tiles'):
        return None
    return {row[0]: tuple(row[1:]) for row in cache['panos']}


def rescan_coverage(area, source):
    """Re-enumerates the area's tiles for exact per-pano (lat, lon, captured_at,
    quality) — the attributes thin_panos selects on. Uses main.py's tile rule."""
    tiles = labeler.tiles_intersecting(area, source.COVERAGE_TILE_ZOOM)
    panos = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(source.fetch_panos_for_tile, x, y, area) for x, y in tiles]
        for future in as_completed(futures):
            tile_panos = future.result()
            if tile_panos is None:
                sys.exit("A coverage tile failed after retries; re-run the experiment.")
            panos.update(tile_panos)
    return panos


def ground_point(pano, det):
    """Flat-ground (lat, lon) of a detection at 2.6 m, or None if unplaceable (at/above
    the horizon or beyond geo's 25 m range — dropped, never clamped)."""
    if pano.get('camera_heading') is None:
        return None
    est = geo.detection_ground_point(geo.pano_pose(pano), det['x_normalized'],
                                     det['y_normalized'], apply_pose=False)
    return None if est is None else (est.lat, est.lng)


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
    for conf, point, pano_id in projected:
        for site in sites:
            if meters_between(point, site['centroid']) <= radius_m:
                n = site['n']
                site['centroid'] = ((site['centroid'][0] * n + point[0]) / (n + 1),
                                    (site['centroid'][1] * n + point[1]) / (n + 1))
                site['n'] = n + 1
                site['members'].add(pano_id)
                break
        else:
            sites.append({'centroid': point, 'n': 1, 'members': {pano_id}, 'max_conf': conf})
    return sites


def sites_retained(sites, kept_pano_ids, min_views=1):
    return sum(1 for site in sites if len(site['members'] & kept_pano_ids) >= min_views)


def site_table(sites, scan, spacings, thin):
    """One row per ramp site: its member count at full density, its highest member
    confidence, and how many of its member panos each spacing keeps (0 = lost) — what
    tells a lost one-view, low-confidence site from a lost well-seen one."""
    kept = {sp: set(scan) if sp == 0 else set(thin(scan, sp)) for sp in spacings}
    rows = []
    for i, site in enumerate(sites):
        row = {'site': i, 'lat': round(site['centroid'][0], 7), 'lon': round(site['centroid'][1], 7),
               'member_panos': len(site['members']), 'max_confidence': round(site['max_conf'], 3)}
        for sp in spacings:
            row[f'views_at_{sp:g}m'] = len(site['members'] & kept[sp])
        rows.append(row)
    return rows


def spacing_curve(sites, scan, spacings, random_seeds, thin, seconds_per_pano=SECONDS_PER_PANO):
    """One row per spacing: what `thin` (the source's thin_panos) keeps, and what a
    random selection of the same size keeps (A3, mean over seeds)."""
    robust = [s for s in sites if len(s['members']) >= ROBUST_MIN_PANOS]
    rows = []
    for spacing in spacings:
        kept = set(scan) if spacing == 0 else set(thin(scan, spacing))
        rand_all, rand_robust, rand_2v = [], [], []
        for seed in range(random_seeds):
            sample = set(random.Random(seed).sample(sorted(scan), len(kept)))
            rand_all.append(sites_retained(sites, sample))
            rand_robust.append(sites_retained(robust, sample))
            rand_2v.append(sites_retained(sites, sample, min_views=2))
        mean = lambda xs: round(sum(xs) / len(xs), 1)
        rows.append({
            'spacing_m': spacing,
            'panos_kept': len(kept),
            'gpu_hours': round(len(kept) * seconds_per_pano / 3600, 2),
            'sites_retained': sites_retained(sites, kept),           # A2
            'sites_retained_random_mean': mean(rand_all),            # A3
            'sites_total': len(sites),
            'robust_retained': sites_retained(robust, kept),
            'robust_retained_random_mean': mean(rand_robust),
            'robust_total': len(robust),
            'sites_2plus_views': sites_retained(sites, kept, min_views=2),
            'sites_2plus_views_random_mean': mean(rand_2v),
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
    parser = argparse.ArgumentParser(description="Analyze an un-thinned Mapillary/Panoramax run for the thinning experiment.")
    parser.add_argument("run_dir", help="Run directory of an UN-thinned (--thin-spacing 0) Mapillary or Panoramax run.")
    parser.add_argument("--spacings", type=float, nargs='+', default=[0, 2.5, 5, 7.5, 10, 15, 20])
    parser.add_argument("--min-confidence", type=float, default=BENCHMARK_CONFIDENCE,
                        help="Detection tier the ramp sites are built from (default the benchmark "
                             "tier %(default)s; a non-default tier writes thinning_experiment_t<tier>/).")
    parser.add_argument("--seconds-per-pano", type=float, default=SECONDS_PER_PANO,
                        help="Rate for the gpu_hours column (default %(default)s, main.py's planning rate).")
    parser.add_argument("--cluster-radius", type=float, default=7.5,
                        help="Meters within which detections merge into one ramp site (default %(default)s).")
    parser.add_argument("--opportunity-radius", type=float, default=20.0)
    parser.add_argument("--random-seeds", type=int, default=20)
    args = parser.parse_args()

    run_dir, records, area, source, source_name = load_run(args.run_dir, args.min_confidence)
    scan = load_scan(run_dir, source_name)
    if scan is not None:
        print(f"-> {len(records)} processed panos; {len(scan)} panos in the run's scan.json.")
    else:
        print(f"-> {len(records)} processed panos; no usable scan.json, rescanning coverage...")
        scan = rescan_coverage(area, source)
    # Only panos that were actually processed can retain sites; warn if the run looks partial.
    processed = {r['pano']['panorama_id'] for r in records}
    missing = len([p for p in scan if p not in processed])
    if missing:
        print(f"⚠ {missing} scanned panos have no results record (partial or thinned run?) — "
              f"curves will understate dense-sampling coverage.")

    sites = cluster_detections(records, args.cluster_radius)
    print(f"-> {sum(len(r['detections']) for r in records)} detections → {len(sites)} ramp sites "
          f"(cluster radius {args.cluster_radius} m).")

    curve = spacing_curve(sites, scan, args.spacings, args.random_seeds, source.thin_panos,
                          args.seconds_per_pano)
    bins = distance_bins(sites, records, args.opportunity_radius, [0, 4, 8, 12, 16, 20])

    suffix = '' if args.min_confidence == BENCHMARK_CONFIDENCE else f"_t{args.min_confidence:g}"
    out_dir = run_dir / f"thinning_experiment{suffix}"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / "spacing_curve.csv", curve)
    write_csv(out_dir / "distance_bins.csv", bins)
    write_csv(out_dir / "sites.csv", site_table(sites, scan, args.spacings, source.thin_panos))

    def table(rows):
        header = '| ' + ' | '.join(rows[0].keys()) + ' |'
        sep = '|' + '---|' * len(rows[0])
        return '\n'.join([header, sep] + ['| ' + ' | '.join(str(v) for v in row.values()) + ' |' for row in rows])

    (out_dir / "report.md").write_text(
        f"# Thinning experiment — {run_dir.name}\n\n"
        f"{source_name}; {len(records)} panos processed un-thinned ({len(scan)} in scan.json); "
        f"{len(sites)} model-derived ramp sites from detections >= {args.min_confidence:g} off the rig "
        f"(cluster radius {args.cluster_radius} m; robust = >= {ROBUST_MIN_PANOS} member panos). "
        f"gpu_hours at {args.seconds_per_pano:g} s/pano. "
        f"See the module docstring for assumptions A1-A3 and caveats.\n\n"
        f"## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)\n\n{table(curve)}\n\n"
        f"## A1 — detection rate by camera-to-site distance\n\n{table(bins)}\n",
        encoding='utf-8')
    print(f"-> Wrote {out_dir / 'report.md'} (+ CSVs).")


if __name__ == "__main__":
    main()
