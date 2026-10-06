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
  4. Read runs/thinexp/thinning_experiment/report.md (+ CSVs; vintages.csv is the
     capture-year mix of the scan and of each spacing's kept set).
  5. Redraw the doc's figures from the committed CSVs only (no run data, no network):
       python scripts/thinning_experiment.py figures
     (--runs names the run dirs, default runs/thinexp_bayonne runs/thinexp_richmond;
     writes PNG + SVG to docs/figures/thinning-experiment/.)
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
import csv
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
FIG_DIR = REPO_ROOT / "docs" / "figures" / "thinning-experiment"
DEFAULT_FIGURE_RUNS = ("runs/thinexp_bayonne", "runs/thinexp_richmond")


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


def site_columns(spacings):
    return (['site', 'lat', 'lon', 'member_panos', 'max_confidence']
            + [f'views_at_{sp:g}m' for sp in spacings])


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
        mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # --random-seeds 0
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


def capture_month(captured):
    """'YYYY-MM' of a scan row's capture field: Panoramax stores an ISO string,
    Mapillary epoch milliseconds."""
    if isinstance(captured, (int, float)):
        return datetime.fromtimestamp(captured / 1000, tz=timezone.utc).strftime('%Y-%m')
    return str(captured)[:7]


def vintage_table(scan, spacings, thin):
    """Capture-year mix of the scan (spacing 0) and of each spacing's kept set: one
    'all' row per spacing, then one row per year. capture_months counts distinct
    year-months, so the 'all' row's is the number of capture months in that set."""
    rows = []
    for spacing in spacings:
        kept = list(scan) if spacing == 0 else list(thin(scan, spacing))
        months = [capture_month(scan[p][2]) for p in kept]
        by_year = Counter(m[:4] for m in months)
        rows.append({'spacing_m': spacing, 'capture_year': 'all', 'panos': len(kept),
                     'share_of_kept': 1.0 if kept else None, 'capture_months': len(set(months))})
        for year in sorted(by_year):
            rows.append({'spacing_m': spacing, 'capture_year': year, 'panos': by_year[year],
                         'share_of_kept': round(by_year[year] / len(kept), 3),
                         'capture_months': len({m for m in months if m[:4] == year})})
    return rows


def write_csv(path, rows, columns=None):
    """rows -> CSV (None as a blank cell). With no rows, writes the header from
    `columns` (or an empty file), so a run with no site at the tier still writes."""
    cols = list(rows[0].keys()) if rows else list(columns or [])
    with open(path, 'w', encoding='utf-8') as f:
        if cols:
            f.write(','.join(cols) + '\n')
        for row in rows:
            f.write(','.join('' if v is None else str(v) for v in row.values()) + '\n')


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
    write_csv(out_dir / "sites.csv", site_table(sites, scan, args.spacings, source.thin_panos),
              columns=site_columns(args.spacings))
    vintages = vintage_table(scan, args.spacings, source.thin_panos)
    write_csv(out_dir / "vintages.csv", vintages)
    for row in vintages:
        if row['capture_year'] == 'all':
            years = {r['capture_year']: r['panos'] for r in vintages
                     if r['spacing_m'] == row['spacing_m'] and r['capture_year'] != 'all'}
            print(f"   vintages at {row['spacing_m']:g} m: {row['panos']} panos, "
                  f"{row['capture_months']} capture months; by year {years}")

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


# ---------------------------------------------------------------------------------------
# figures: redrawn from the committed CSVs only (spacing_curve.csv, distance_bins.csv)

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
THIN_COLOR, RANDOM_COLOR = '#2a78d6', '#eb6834'      # validated pair (dataviz palette 1, 2)
CITY_STYLES = (  # color + marker + dash, so identity never rests on color alone
    dict(color='#2a78d6', marker='o', linestyle='-'),
    dict(color='#eb6834', marker='s', linestyle='--'),
)
TIERS = (('0.3', '_t0.3'), ('0.55', ''))             # (label, output-dir suffix)


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _num(v):
    return float('nan') if v in ('', 'None', None) else float(v)


def _city_label(run_dir):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    name = Path(run_dir).name.replace('thinexp_', '').replace('_', ' ').title()
    return f"{name} ({str(manifest.get('imagery_source', '?')).title()})"


def _style(plt):
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 9,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'grid.color': GRID,
        'lines.linewidth': 2, 'lines.markersize': 6, 'hatch.color': '#b5b3ad',
        'svg.hashsalt': 'thinexp144', 'svg.fonttype': 'path', 'path.simplify': False,
        'font.family': 'DejaVu Sans'})


def _save(fig, out_dir, stem):
    """PNG + SVG with timestamp/version metadata dropped, so re-runs are byte-stable."""
    fig.savefig(out_dir / f'{stem}.png', dpi=150, metadata={'Software': None})
    buf = io.BytesIO()   # bytes, so the SVG is LF on every platform
    fig.savefig(buf, format='svg', metadata={'Date': None, 'Creator': None})
    (out_dir / f'{stem}.svg').write_bytes(buf.getvalue())
    return [out_dir / f'{stem}.png', out_dir / f'{stem}.svg']


def figure_coverage(plt, runs, tier, suffix, out_dir):
    """Sites kept vs spacing (% of the full-density count): rows = all / robust / 2+
    views, columns = city; thin_panos solid circles, random same-count dashed squares."""
    metrics = (('sites_retained', 'all sites'),
               ('robust_retained', f'robust sites (>= {ROBUST_MIN_PANOS} panos at full density)'),
               ('sites_2plus_views', 'sites seen from >= 2 kept panos'))
    fig, axes = plt.subplots(len(metrics), len(runs), figsize=(4.8 * len(runs), 9.6),
                             sharex=True, sharey=True, squeeze=False)
    for col, run in enumerate(runs):
        rows = read_csv(Path(run) / f"thinning_experiment{suffix}" / "spacing_curve.csv")
        x = [_num(r['spacing_m']) for r in rows]
        for i, (key, label) in enumerate(metrics):
            ax = axes[i][col]
            full = _num(rows[0][key])   # spacing 0 = full density
            pct = lambda k: [100 * _num(r[k]) / full if full else float('nan') for r in rows]
            thin, rand = pct(key), pct(f"{key}_random_mean")
            for ref in (5, 10):
                ax.axvline(ref, color=GRID, linewidth=1, zorder=0)
            ax.plot(x, rand, color=RANDOM_COLOR, marker='s', linestyle='--',
                    label='random selection, same pano count (mean of seeds)')
            ax.plot(x, thin, color=THIN_COLOR, marker='o', linestyle='-',
                    label='thin_panos (newest capture per cell)')
            for ref in (5, 10):   # selective direct labels: the two spacings in question
                if ref in x:
                    j = x.index(ref)
                    below = thin[j] < rand[j]   # keep the label off the other line
                    ax.annotate(f"{int(_num(rows[j][key]))}", (ref, thin[j]),
                                xytext=(5, -13 if below else 6), textcoords='offset points',
                                fontsize=8.5, color=INK)
            ax.set_ylim(0, 108)
            ax.grid(axis='y', linewidth=0.6)
            title = f"{label}, n = {int(full)}"
            ax.set_title(f"{_city_label(run)}\n{title}" if i == 0 else title, loc='left')
            if col == 0:
                ax.set_ylabel('kept, % of full density')
            if i == len(metrics) - 1:
                ax.set_xlabel('thinning cell (m); 0 = every pano')
                ax.set_xticks(x)
                ax.set_xticklabels([f"{v:g}" for v in x])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"Ramp sites kept vs thinning cell, detections >= {tier} "
                 f"(labels: thin_panos count at 5 m and 10 m)", y=0.968, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    paths = _save(fig, out_dir, f"coverage_vs_spacing_t{tier}")
    plt.close(fig)
    return paths


def figure_distance(plt, runs, out_dir):
    """A1: per-pano detection rate by camera-to-site distance, one panel per tier,
    one line per city (color + marker + dash), with the rig-mask zone shaded."""
    from detectors import NADIR_MASK_DEG
    mask_m = geo.DEFAULT_CAMERA_HEIGHT_M / math.tan(math.radians(NADIR_MASK_DEG))
    fig, axes = plt.subplots(1, len(TIERS), figsize=(10, 4.4), sharey=True, squeeze=False)
    for k, (tier, suffix) in enumerate(TIERS):
        ax = axes[0][k]
        ax.axvspan(0, mask_m, facecolor='#f1f0ec', hatch='///', edgecolor='#b5b3ad',
                   linewidth=0, zorder=0)
        ax.text(mask_m / 2, 0.53, f"rig mask: no detection\nnearer than {mask_m:.2f} m",
                ha='center', va='top', fontsize=7.5, color=INK2, rotation=90)
        for c, run in enumerate(runs):
            rows = read_csv(Path(run) / f"thinning_experiment{suffix}" / "distance_bins.csv")
            edges = [tuple(float(v) for v in r['bin_m'].split('-')) for r in rows]
            mids = [(lo + hi) / 2 for lo, hi in edges]
            rate = [_num(r['detection_rate']) for r in rows]
            ax.plot(mids, rate, label=_city_label(run), **CITY_STYLES[c % len(CITY_STYLES)])
            ax.annotate(_city_label(run).split(' (')[0], (mids[-1], rate[-1]), xytext=(6, 0),
                        textcoords='offset points', va='center', fontsize=8.5, color=INK)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 4, 8, 12, 16, 20])
        ax.set_ylim(0, 0.55)
        ax.grid(axis='y', linewidth=0.6)
        ax.set_title(f"detections >= {tier}", loc='left')
        ax.set_xlabel('camera-to-site distance (m), 4 m bins')
        if k == 0:
            ax.set_ylabel('share of (pano, site) pairs\nwhere the pano detected the site')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(runs), frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("A1: per-view detection rate by camera-to-site distance (pairs within 20 m)",
                 y=0.925, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    paths = _save(fig, out_dir, "detection_rate_by_distance")
    plt.close(fig)
    return paths


def figures_main(argv):
    parser = argparse.ArgumentParser(
        prog="thinning_experiment.py figures",
        description="Redraw docs/figures/thinning-experiment/ from the committed CSVs only.")
    parser.add_argument("--runs", nargs='+', default=list(DEFAULT_FIGURE_RUNS),
                        help="Run dirs holding thinning_experiment[_t0.3]/ (default %(default)s).")
    parser.add_argument("--out", default=str(FIG_DIR))
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) if Path(r).is_absolute() else REPO_ROOT / r for r in args.runs]
    written = []
    for tier, suffix in TIERS:
        written += figure_coverage(plt, runs, tier, suffix, out_dir)
    written += figure_distance(plt, runs, out_dir)
    for path in written:
        print(f"-> {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ['figures']:
        figures_main(sys.argv[2:])
    else:
        main()
