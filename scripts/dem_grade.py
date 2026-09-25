"""Road grade from a DEM, independent of the SfM (issue #51).

Fusion's road-relative raycast (--apply-pose road, #42) subtracts a road grade from each
Mapillary frame's pitch. Production takes that grade from the sequence's own SfM altitude
(fuse_sites.sequence_grades on `computed_altitude`) -- the same reconstruction that
produced the pitch, so subtracting one from the other may cancel shared SfM error rather
than road slope (#52's reading of the #42 control, docs/mapillary-tilt-study.md 10.5).
This script builds the independent arm: the grade from USGS 3DEP elevation.

For each city it:

1. Fetches 3DEP over the run's area.geojson bbox (padded PAD_M) from the ImageServer's
   exportImage endpoint as float32 GeoTIFFs at --posts-m spacing, in tiles of at most
   MAX_TILE_PX a side (the service advertises more, but a 3000 px request returned HTTP
   500 while 1500 px always succeeded). Tiles are cached in runs/<city>/dem/tiles/ and
   recorded in runs/<city>/dem/tiles.json (service, request parameters, grid, per-tile
   sha256 and fetch time); a re-run reuses every tile whose sha256 still matches and
   refetches the rest. A response that is not the float32 TIFF asked for -- an HTML error
   page, a JSON error, the wrong size or the wrong georeference -- is refused and never
   cached. --verify re-hashes the cache and exits 1 on any mismatch.
2. Samples the DEM at every pano's own position (bilinear on the 4326 pixel grid; pixel
   centres at half-pixel offsets from the TIFF's own tie point) and writes
   runs/<city>/dem/grades.csv, one row per pano:
     grade_sfm_deg           fuse_sites.sequence_grades on computed_altitude -- exactly
                             production's grade (load_results reproduces it bit for bit)
     grade_dem_2pt_deg       the same call with the DEM elevation substituted: same frames,
                             same 2-40 m baseline, same bearing; only the altitude differs
     grade_dem_deg           a least-squares line through the DEM elevation of every frame
                             of the sequence within +-FIT_HALF_WINDOW_M of path distance
                             (>= 3 frames, else the two-point value) -- what `--grade-source
                             dem` subtracts
     grade_sfm_smoothed_deg  the identical fit on computed_altitude (the issue's smoothing
                             arm: "a less noisy SfM" vs "an independent source")
     travel_bearing_deg      sequence_grades' bearing (every grade source keeps it)
     baseline_m, n_frames_fit  path span and frame count of the DEM fit window
     dem_z_m                 the DEM elevation at the pano
     dem_relief_ratio        the sequence's OLS slope of computed_altitude on DEM elevation
                             (>= RELIEF_MIN_FRAMES frames, >= RELIEF_MIN_M of DEM relief),
                             repeated on every row of the sequence
3. Writes runs/<city>/dem/report.md (DEM-vs-SfM agreement per city and rig, by |grade|
   bucket, frame-to-frame roughness, the relief-ratio distribution, coverage) and one row
   per (city, rig) to runs/_summary/dem_grade.csv.

Only elevation.nationalmap.gov is contacted, and at most one request per
REQUEST_INTERVAL_S. Needs requests, numpy and Pillow (all pipeline dependencies).

Usage:
    python scripts/dem_grade.py richmond clovis morgantown annapolis laurens
    python scripts/dem_grade.py richmond --verify           # re-hash the tile cache only
    python scripts/fuse_sites.py runs/richmond --apply-pose road --grade-source dem
"""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT / 'scripts'), str(REPO_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

import fuse_sites as fs  # noqa: E402
import geo  # noqa: E402

SERVICE_URL = ('https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/'
               'ImageServer/exportImage')
REQUEST_PARAMS = {'bboxSR': 4326, 'imageSR': 4326, 'format': 'tiff', 'pixelType': 'F32',
                  'interpolation': 'RSP_BilinearInterpolation', 'f': 'image'}
MAX_TILE_PX = 1500
POSTS_M = 2.0              # 3DEP is 1 m lidar in all five cities; >= 10 m baselines lose nothing
PAD_M = 100.0
M_PER_DEG_LAT = 111_320.0  # nominal; sets the post spacing, never a measured distance
REQUEST_INTERVAL_S = 0.5
ATTEMPTS = 3
TIMEOUT_S = 120
NODATA_BELOW_M = -1000.0   # the service's NoData is a huge negative float
FIT_HALF_WINDOW_M = 20.0   # ten 2 m posts; the upper half of the 2-40 m production baseline
RELIEF_MIN_FRAMES, RELIEF_MIN_M = 30, 0.5
GRADE_BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, math.inf))
RIG_MIN_FRAMES = 200       # rigs with fewer graded frames are pooled as 'other'

GRADES_FIELDS = ['panorama_id', 'sequence_id', 'grade_sfm_deg', 'grade_sfm_smoothed_deg',
                 'grade_dem_deg', 'grade_dem_2pt_deg', 'travel_bearing_deg', 'baseline_m',
                 'n_frames_fit', 'dem_z_m', 'dem_relief_ratio']


class FetchRefused(RuntimeError):
    """The service answered, but not with the raster asked for. Never cached."""


# --- the tile grid ------------------------------------------------------------------

def geometry_bbox(geom):
    """(min_lng, min_lat, max_lng, max_lat) of a bare GeoJSON geometry."""
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for sub in c:
                walk(sub)
    if geom.get('type') == 'GeometryCollection':
        for g in geom['geometries']:
            b = geometry_bbox(g)
            xs.extend((b[0], b[2]))
            ys.extend((b[1], b[3]))
    else:
        walk(geom['coordinates'])
    return min(xs), min(ys), max(xs), max(ys)


def tile_grid(bbox, posts_m=POSTS_M, pad_m=PAD_M, max_px=MAX_TILE_PX):
    """The request grid over bbox padded by pad_m: one global pixel grid of SQUARE
    degree-pixels (the service returns square pixels in 4326 whatever size is asked, so a
    non-square request would come back resampled onto a different extent), split into
    tiles of at most max_px a side that cover it exactly once.

    Returns {'step_deg', 'x0', 'y1', 'width', 'height', 'tiles': [tile]} with each tile
    {'name', 'row', 'col', 'px_offset': [r0, c0], 'size': [w, h], 'bbox': [xmin, ymin,
    xmax, ymax]}. x0/y1 are the grid's west and north edges.
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    step = posts_m / M_PER_DEG_LAT
    mid = math.radians((min_lat + max_lat) / 2)
    pad_lat = pad_m / M_PER_DEG_LAT
    pad_lng = pad_m / (M_PER_DEG_LAT * math.cos(mid))
    x0, y1 = min_lng - pad_lng, max_lat + pad_lat
    width = math.ceil((max_lng + pad_lng - x0) / step)
    height = math.ceil((y1 - (min_lat - pad_lat)) / step)
    tiles = []
    for ri, r0 in enumerate(range(0, height, max_px)):
        for ci, c0 in enumerate(range(0, width, max_px)):
            h, w = min(max_px, height - r0), min(max_px, width - c0)
            tiles.append({'name': f'{ri}_{ci}', 'row': ri, 'col': ci, 'px_offset': [r0, c0],
                          'size': [w, h],
                          'bbox': [x0 + c0 * step, y1 - (r0 + h) * step,
                                   x0 + (c0 + w) * step, y1 - r0 * step]})
    return {'step_deg': step, 'x0': x0, 'y1': y1, 'width': width, 'height': height,
            'tiles': tiles}


# --- fetch and cache ----------------------------------------------------------------

def decode_tile(content, tile, step):
    """Validate a response body as the float32 GeoTIFF this tile asked for and return it
    as an array (NoData -> NaN). Raises FetchRefused on anything else."""
    try:
        im = Image.open(io.BytesIO(content))
        im.load()
    except Exception as e:  # noqa: BLE001 -- any undecodable body is a refusal
        raise FetchRefused(f'not an image ({len(content)} bytes: {content[:80]!r}): {e}')
    w, h = tile['size']
    if im.mode != 'F' or im.size != (w, h):
        raise FetchRefused(f'expected a {w}x{h} float32 raster, got mode {im.mode} '
                           f'size {im.size}')
    tie, scale = im.tag_v2.get(33922), im.tag_v2.get(33550)
    if tie is None or scale is None:
        raise FetchRefused('GeoTIFF carries no tie point / pixel scale')
    xmin, _ymin, _xmax, ymax = tile['bbox']
    if (abs(tie[3] - xmin) > 0.01 * step or abs(tie[4] - ymax) > 0.01 * step
            or abs(scale[0] - step) > 1e-6 * step or abs(scale[1] - step) > 1e-6 * step):
        raise FetchRefused(f'georeference {tie[3:5]} / {scale[:2]} does not match the '
                           f'requested grid {(xmin, ymax)} / {step}')
    a = np.asarray(im, dtype=np.float32).copy()
    a[~np.isfinite(a) | (a < NODATA_BELOW_M)] = np.nan
    return a


def fetch_tile(tile, step, attempts=ATTEMPTS, pause=time.sleep):
    """GET one tile; returns the validated body bytes. Retries with backoff on transport
    errors, HTTP errors and refusals; raises the last error after `attempts`."""
    w, h = tile['size']
    params = {**REQUEST_PARAMS, 'bbox': ','.join(repr(v) for v in tile['bbox']),
              'size': f'{w},{h}'}
    last = None
    for attempt in range(attempts):
        if attempt:
            pause(2.0 ** attempt)
        try:
            r = requests.get(SERVICE_URL, params=params, timeout=TIMEOUT_S)
            if r.status_code != 200:
                raise FetchRefused(f'HTTP {r.status_code}')
            decode_tile(r.content, tile, step)
            return r.content
        except (requests.RequestException, FetchRefused) as e:
            last = e
    raise last


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch_city(dem_dir, grid, bbox, posts_m, pause=time.sleep, log=print):
    """Fetch or reuse every tile of the grid into dem_dir/tiles/, write tiles.json.
    Returns the tiles.json record."""
    tiles_dir = dem_dir / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dem_dir / 'tiles.json'
    old = {}
    if manifest_path.exists():
        old = {t['name']: t for t in json.loads(manifest_path.read_text('utf-8'))['tiles']}
    records, fetched, reused = [], 0, 0
    for tile in grid['tiles']:
        path = tiles_dir / f"{tile['name']}.tif"
        prev = old.get(tile['name'])
        if (prev and path.exists() and prev['bbox'] == tile['bbox']
                and prev['size'] == tile['size']
                and sha256_bytes(path.read_bytes()) == prev['sha256']):
            records.append(prev)
            reused += 1
            continue
        if fetched:
            pause(REQUEST_INTERVAL_S)
        body = fetch_tile(tile, grid['step_deg'], pause=pause)
        tmp = path.with_suffix('.tif.part')
        tmp.write_bytes(body)
        tmp.replace(path)
        fetched += 1
        records.append({**tile, 'sha256': sha256_bytes(body), 'bytes': len(body),
                        'fetched_at': datetime.now(timezone.utc).isoformat(
                            timespec='seconds')})
        log(f"  tile {tile['name']} {tile['size'][0]}x{tile['size'][1]} "
            f"({len(body) / 1e6:.1f} MB)")
    manifest = {'service': SERVICE_URL, 'request_params': REQUEST_PARAMS,
                'posts_m': posts_m, 'pad_m': PAD_M, 'max_tile_px': MAX_TILE_PX,
                'area_bbox': list(bbox),
                'grid': {k: grid[k] for k in ('step_deg', 'x0', 'y1', 'width', 'height')},
                'tiles': records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', 'utf-8')
    log(f'  {fetched} tile(s) fetched, {reused} reused')
    return manifest


def verify_city(dem_dir):
    """[problem strings] for a cached city: every recorded tile present, hash matching."""
    manifest_path = dem_dir / 'tiles.json'
    if not manifest_path.exists():
        return [f'{manifest_path} missing']
    problems = []
    for t in json.loads(manifest_path.read_text('utf-8'))['tiles']:
        path = dem_dir / 'tiles' / f"{t['name']}.tif"
        if not path.exists():
            problems.append(f'{path} missing')
        elif sha256_bytes(path.read_bytes()) != t['sha256']:
            problems.append(f'{path} sha256 mismatch')
    return problems


# --- sampling -----------------------------------------------------------------------

class Mosaic:
    """All of a city's tiles in one array, sampled bilinearly in the 4326 pixel grid
    (rows are latitude from the north edge, columns longitude from the west edge; pixel
    (r, c) is centred at (x0 + (c + .5) step, y1 - (r + .5) step))."""

    def __init__(self, array, x0, y1, step):
        self.a, self.x0, self.y1, self.step = array, x0, y1, step

    @classmethod
    def from_cache(cls, dem_dir, manifest):
        g = manifest['grid']
        a = np.full((g['height'], g['width']), np.nan, dtype=np.float32)
        for t in manifest['tiles']:
            body = (dem_dir / 'tiles' / f"{t['name']}.tif").read_bytes()
            tile = decode_tile(body, t, g['step_deg'])
            r0, c0 = t['px_offset']
            a[r0:r0 + tile.shape[0], c0:c0 + tile.shape[1]] = tile
        return cls(a, g['x0'], g['y1'], g['step_deg'])

    def sample(self, lats, lngs):
        """Bilinear elevations (float array, NaN outside the raster or next to NoData)."""
        lats, lngs = np.asarray(lats, float), np.asarray(lngs, float)
        fc = (lngs - self.x0) / self.step - 0.5
        fr = (self.y1 - lats) / self.step - 0.5
        c0, r0 = np.floor(fc).astype(int), np.floor(fr).astype(int)
        H, W = self.a.shape
        ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < W) & (r0 + 1 < H)
        out = np.full(lats.shape, np.nan)
        c, r = c0[ok], r0[ok]
        dc, dr = fc[ok] - c, fr[ok] - r
        a = self.a
        out[ok] = ((1 - dr) * ((1 - dc) * a[r, c] + dc * a[r, c + 1])
                   + dr * ((1 - dc) * a[r + 1, c] + dc * a[r + 1, c + 1]))
        return out


# --- grades -------------------------------------------------------------------------

def read_frames(results_path):
    """[(pano_id, sequence_id, captured_at, lat, lng, computed_altitude, rig)] in file
    order, with exactly fuse_sites.load_results' skip rule (no lat/lng/heading), so the
    frames sequence_grades sees here are the ones it sees in production."""
    frames = []
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)['pano']
            if p.get('lat') is None or p.get('lng') is None \
                    or p.get('camera_heading') is None:
                continue
            meta = p.get('source_metadata') or {}
            rig = ' '.join(x for x in (p.get('camera_make') or meta.get('make'),
                                       p.get('camera_model') or meta.get('model')) if x)
            frames.append((p['panorama_id'], p.get('sequence_id'), meta.get('captured_at'),
                           p['lat'], p['lng'], meta.get('computed_altitude'), rig or '?'))
    return frames


def _segments(frames_idx, frames):
    """Split one sequence's time-ordered frame indices where consecutive frames are more
    than 2 * GRADE_MAX_GAP_S or GRADE_MAX_DIST_M apart (the bounds sequence_grades uses),
    returning [(indices, cumulative path distance)]."""
    segs, cur, s = [], [], []
    for i in frames_idx:
        if cur:
            j = cur[-1]
            d = geo.haversine_m(frames[j][3], frames[j][4], frames[i][3], frames[i][4])
            if (frames[i][2] - frames[j][2]) / 1000.0 > 2 * fs.GRADE_MAX_GAP_S \
                    or d > fs.GRADE_MAX_DIST_M:
                segs.append((cur, s))
                cur, s = [], []
            else:
                cur.append(i)
                s.append(s[-1] + d)
                continue
        cur, s = [i], [0.0]
    if cur:
        segs.append((cur, s))
    return segs


def fitted_grades(frames, z, half_window_m=FIT_HALF_WINDOW_M):
    """{frame index: (grade_deg, span_m, n)} from a least-squares line of z on path
    distance over every frame of the sequence segment within +-half_window_m, where at
    least 3 frames with a z value span >= GRADE_MIN_DIST_M. Positive uphill along travel
    (time order), the sign sequence_grades uses."""
    by_seq = {}
    for i, fr in enumerate(frames):
        if fr[1] is None or fr[2] is None:
            continue
        by_seq.setdefault(fr[1], []).append(i)
    out = {}
    for idx in by_seq.values():
        idx.sort(key=lambda i: frames[i][2])      # stable, like sequence_grades
        for seg, s in _segments(idx, frames):
            n = len(seg)
            lo = hi = 0
            for k in range(n):
                while s[k] - s[lo] > half_window_m:
                    lo += 1
                hi = max(hi, k)
                while hi + 1 < n and s[hi + 1] - s[k] <= half_window_m:
                    hi += 1
                pts = [(s[m], z[seg[m]]) for m in range(lo, hi + 1) if z[seg[m]] is not None]
                if len(pts) < 3:
                    continue
                xs = [p[0] for p in pts]
                span = max(xs) - min(xs)
                if span < fs.GRADE_MIN_DIST_M:
                    continue
                mx = sum(xs) / len(xs)
                mz = sum(p[1] for p in pts) / len(pts)
                sxx = sum((x - mx) ** 2 for x in xs)
                sxz = sum((x - mx) * (p[1] - mz) for x, p in zip(xs, pts))
                out[seg[k]] = (math.degrees(math.atan(sxz / sxx)), span, len(pts))
    return out


def relief_ratios(frames, dem_z):
    """{sequence_id: OLS slope of computed_altitude on DEM elevation} for sequences with
    >= RELIEF_MIN_FRAMES frames holding both and >= RELIEF_MIN_M of DEM relief."""
    by_seq = {}
    for i, fr in enumerate(frames):
        if fr[1] is not None and fr[5] is not None and dem_z[i] is not None:
            by_seq.setdefault(fr[1], []).append((dem_z[i], fr[5]))
    out = {}
    for seq, pts in by_seq.items():
        if len(pts) < RELIEF_MIN_FRAMES:
            continue
        xs = [p[0] for p in pts]
        if max(xs) - min(xs) < RELIEF_MIN_M:
            continue
        mx, my = sum(xs) / len(xs), sum(p[1] for p in pts) / len(pts)
        out[seq] = (sum((x - mx) * (p[1] - my) for x, p in zip(xs, pts))
                    / sum((x - mx) ** 2 for x in xs))
    return out


def compute_grades(frames, dem_z, half_window_m=FIT_HALF_WINDOW_M):
    """grades.csv rows (dicts, GRADES_FIELDS order) for frames + per-frame DEM elevation
    (None where out of raster)."""
    def seq_frames(alt):
        return [(i, fr[1], fr[2], fr[3], fr[4], alt[i]) for i, fr in enumerate(frames)]
    sfm_alt = [fr[5] for fr in frames]
    sfm = fs.sequence_grades(seq_frames(sfm_alt))
    dem2 = fs.sequence_grades(seq_frames(dem_z))
    dem_fit = fitted_grades(frames, dem_z, half_window_m)
    sfm_fit = fitted_grades(frames, sfm_alt, half_window_m)
    relief = relief_ratios(frames, dem_z)
    rows = []
    for i, fr in enumerate(frames):
        g_sfm = sfm.get(i, (None, None))[0]
        g_dem2 = dem2.get(i, (None, None))[0]
        fit = dem_fit.get(i)
        rows.append({
            'panorama_id': fr[0], 'sequence_id': fr[1],
            'grade_sfm_deg': g_sfm,
            'grade_sfm_smoothed_deg': sfm_fit[i][0] if i in sfm_fit else g_sfm,
            'grade_dem_deg': fit[0] if fit else g_dem2,
            'grade_dem_2pt_deg': g_dem2,
            'travel_bearing_deg': sfm.get(i, (None, None))[1],
            'baseline_m': fit[1] if fit else None,
            'n_frames_fit': fit[2] if fit else None,
            'dem_z_m': dem_z[i],
            'dem_relief_ratio': relief.get(fr[1]),
        })
    return rows


def write_grades(path, rows):
    """CSV with repr floats (round-trip exact) and empty cells for None."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, GRADES_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: '' if v is None else (repr(v) if isinstance(v, float) else v)
                        for k, v in r.items()})


# --- report -------------------------------------------------------------------------

def _pct(v, q):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]


def agreement(xs, ys):
    """{n, r, slope (y on x), mad (median |y-x|), p90_abs} over paired values."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    if n < 3:
        return {'n': n, 'r': None, 'slope': None, 'mad': None, 'p90_abs': None}
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    d = np.abs(y - x).tolist()
    sx, sy = x.std(), y.std()
    return {'n': n, 'r': float(np.corrcoef(x, y)[0, 1]) if sx and sy else None,
            'slope': float(np.polyfit(x, y, 1)[0]) if sx else None,
            'mad': statistics.median(d), 'p90_abs': _pct(d, 0.9)}


def lag1_rms(frames, rows, key):
    """RMS of the change in `key` between consecutive (time-ordered) frames of a sequence
    that both carry it -- the grade's frame-to-frame roughness."""
    by_seq = {}
    for i, fr in enumerate(frames):
        if fr[1] is not None and fr[2] is not None:
            by_seq.setdefault(fr[1], []).append(i)
    sq = []
    for idx in by_seq.values():
        idx.sort(key=lambda i: frames[i][2])
        for a, b in zip(idx, idx[1:]):
            va, vb = rows[a][key], rows[b][key]
            if va is not None and vb is not None:
                sq.append((vb - va) ** 2)
    return math.sqrt(sum(sq) / len(sq)) if sq else None


PAIRS = [('grade_dem_deg', 'grade_sfm_deg', 'DEM (fit) vs SfM (production)'),
         ('grade_dem_2pt_deg', 'grade_sfm_deg', 'DEM (2-point) vs SfM (production)'),
         ('grade_dem_deg', 'grade_sfm_smoothed_deg', 'DEM (fit) vs SfM (fit)'),
         ('grade_sfm_deg', 'grade_sfm_smoothed_deg', 'SfM (production) vs SfM (fit)'),
         ('grade_dem_2pt_deg', 'grade_dem_deg', 'DEM (2-point) vs DEM (fit)')]
GRADE_KEYS = ['grade_sfm_deg', 'grade_sfm_smoothed_deg', 'grade_dem_2pt_deg', 'grade_dem_deg']


def summarize(city, frames, rows, n_out_of_raster):
    """(summary rows for the CSV, report markdown lines)."""
    rigs = {}
    for i, fr in enumerate(frames):
        rigs.setdefault(fr[6], []).append(i)
    big = {k: v for k, v in rigs.items()
           if sum(1 for i in v if rows[i]['grade_sfm_deg'] is not None) >= RIG_MIN_FRAMES}
    groups = [('all', list(range(len(frames))))] + sorted(big.items())
    other = [i for k, v in rigs.items() if k not in big for i in v]
    if big and any(rows[i]['grade_sfm_deg'] is not None for i in other):
        groups.append(('other', other))

    relief = {}
    for r in rows:
        if r['dem_relief_ratio'] is not None:
            relief[r['sequence_id']] = r['dem_relief_ratio']
    rv = list(relief.values())

    summary, lines = [], []
    lines += [f'## {city}', '',
              f'Panos {len(frames)}; out of raster {n_out_of_raster}; with a grade: '
              + ', '.join(f"{k.replace('grade_', '').replace('_deg', '')} "
                          f"{sum(1 for r in rows if r[k] is not None)}" for k in GRADE_KEYS)
              + f"; DEM fit fell back to the two-point value on "
                f"{sum(1 for r in rows if r['grade_dem_deg'] is not None and r['n_frames_fit'] is None)}.",
              '']
    lines += ['### DEM vs SfM agreement (degrees)', '',
              '| rig | pair | n | r | slope | median abs diff | p90 abs diff |',
              '|---|---|---:|---:|---:|---:|---:|']
    fmt = lambda v, f='.2f': '—' if v is None else format(v, f)  # noqa: E731
    for name, idx in groups:
        sub = [rows[i] for i in idx]
        srow = {'city': city, 'rig': name, 'panos': len(idx)}
        for a, b, label in PAIRS:
            ag = agreement([r[a] for r in sub], [r[b] for r in sub])
            lines.append(f"| {name} | {label} | {ag['n']} | {fmt(ag['r'])} | "
                         f"{fmt(ag['slope'])} | {fmt(ag['mad'])} | {fmt(ag['p90_abs'])} |")
            tag = f"{a.replace('grade_', '').replace('_deg', '')}__{b.replace('grade_', '').replace('_deg', '')}"
            for k, v in ag.items():
                srow[f'{tag}_{k}'] = v
        sub_frames = [frames[i] for i in idx]
        for k in GRADE_KEYS:
            srow[f"lag1_rms_{k.replace('grade_', '').replace('_deg', '')}"] = \
                lag1_rms(sub_frames, sub, k)
            srow[f"median_abs_{k.replace('grade_', '').replace('_deg', '')}"] = \
                statistics.median([abs(r[k]) for r in sub if r[k] is not None]) \
                if any(r[k] is not None for r in sub) else None
        seqs = {r['sequence_id'] for r in sub}
        rvs = [v for s, v in relief.items() if s in seqs]
        srow.update({'relief_sequences': len(rvs), 'relief_p10': _pct(rvs, 0.1),
                     'relief_p50': _pct(rvs, 0.5), 'relief_p90': _pct(rvs, 0.9),
                     'relief_share_0p8_1p2': sum(1 for v in rvs if 0.8 <= v <= 1.2) / len(rvs)
                     if rvs else None,
                     'relief_share_negative': sum(1 for v in rvs if v < 0) / len(rvs)
                     if rvs else None})
        summary.append(srow)
    lines.append('')
    lines += ['### Frame-to-frame roughness and magnitude (degrees)', '',
              '| rig | grade | lag-1 RMS | median abs grade |', '|---|---|---:|---:|']
    for srow in summary:
        for k in GRADE_KEYS:
            t = k.replace('grade_', '').replace('_deg', '')
            lines.append(f"| {srow['rig']} | {t} | {fmt(srow[f'lag1_rms_{t}'])} | "
                         f"{fmt(srow[f'median_abs_{t}'])} |")
    lines.append('')
    lines += ['### Relief ratio: per-sequence slope of SfM altitude on DEM elevation', '',
              f'Sequences with >= {RELIEF_MIN_FRAMES} frames and >= {RELIEF_MIN_M} m of DEM '
              'relief. 1.0 = the SfM altitude profile carries all of the terrain\'s relief.', '',
              '| rig | sequences | p10 | p50 | p90 | share 0.8-1.2 | share negative |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for srow in summary:
        lines.append(f"| {srow['rig']} | {srow['relief_sequences']} | "
                     f"{fmt(srow['relief_p10'])} | {fmt(srow['relief_p50'])} | "
                     f"{fmt(srow['relief_p90'])} | {fmt(srow['relief_share_0p8_1p2'], '.0%')} | "
                     f"{fmt(srow['relief_share_negative'], '.0%')} |")
    lines.append('')
    lines += ['### DEM (fit) vs SfM (production) by |grade| bucket', '',
              'Bucketed on each source in turn, because they disagree on which frames are steep.',
              '', '| bucketed on | abs grade | n | r | median abs diff | median abs DEM | '
              'median abs SfM |', '|---|---|---:|---:|---:|---:|---:|']
    for on in ('grade_dem_deg', 'grade_sfm_deg'):
        for lo, hi in GRADE_BUCKETS:
            sub = [r for r in rows if r['grade_dem_deg'] is not None
                   and r['grade_sfm_deg'] is not None and lo <= abs(r[on]) < hi]
            ag = agreement([r['grade_dem_deg'] for r in sub], [r['grade_sfm_deg'] for r in sub])
            md = statistics.median([abs(r['grade_dem_deg']) for r in sub]) if sub else None
            ms = statistics.median([abs(r['grade_sfm_deg']) for r in sub]) if sub else None
            label = f'{lo:g}-{hi:g}' if hi != math.inf else f'{lo:g}+'
            lines.append(f"| {'DEM' if on == 'grade_dem_deg' else 'SfM'} | {label} | "
                         f"{ag['n']} | {fmt(ag['r'])} | {fmt(ag['mad'])} | {fmt(md)} | "
                         f"{fmt(ms)} |")
    lines.append('')
    return summary, lines, rv


def write_report(path, city, manifest, grades_path, lines):
    sha = sha256_bytes(Path(grades_path).read_bytes())
    head = [f'# DEM road grade: {city} (#51)', '',
            'Written by `scripts/dem_grade.py`; the study is `docs/dem-grade-study.md`.', '',
            f"- DEM: USGS 3DEP via `{manifest['service']}`, {manifest['posts_m']:g} m posts "
            f"(square {manifest['grid']['step_deg']:.3e} deg pixels), "
            f"{manifest['grid']['width']}x{manifest['grid']['height']} px in "
            f"{len(manifest['tiles'])} tile(s), "
            f"{sum(t['bytes'] for t in manifest['tiles']) / 1e6:.1f} MB; per-tile sha256 in "
            '`tiles.json`.',
            f"- grades.csv sha256 `{sha}` ({Path(grades_path).stat().st_size} bytes; not "
            'tracked -- regenerate with the command above and compare).',
            f'- Fit window +-{FIT_HALF_WINDOW_M:g} m of path distance, >= 3 frames.', '']
    Path(path).write_text('\n'.join(head + lines) + '\n', 'utf-8')


# --- driver -------------------------------------------------------------------------

def run_city(city, runs_root, posts_m, half_window_m, no_fetch=False, log=print):
    run_dir = runs_root / city
    dem_dir = run_dir / 'dem'
    geom = json.loads((run_dir / 'area.geojson').read_text('utf-8'))
    bbox = geometry_bbox(geom)
    grid = tile_grid(bbox, posts_m)
    log(f'{city}: {grid["width"]}x{grid["height"]} px, {len(grid["tiles"])} tile(s)')
    if no_fetch:
        manifest = json.loads((dem_dir / 'tiles.json').read_text('utf-8'))
        problems = verify_city(dem_dir)
        if problems:
            raise SystemExit('\n'.join(problems))
    else:
        manifest = fetch_city(dem_dir, grid, bbox, posts_m, log=log)
    mosaic = Mosaic.from_cache(dem_dir, manifest)
    frames = read_frames(run_dir / 'results.jsonl')
    z = mosaic.sample([f[3] for f in frames], [f[4] for f in frames])
    dem_z = [None if math.isnan(v) else float(v) for v in z]
    n_out = sum(1 for v in dem_z if v is None)
    rows = compute_grades(frames, dem_z, half_window_m)
    grades_path = dem_dir / 'grades.csv'
    write_grades(grades_path, rows)
    summary, lines, _ = summarize(city, frames, rows, n_out)
    write_report(dem_dir / 'report.md', city, manifest, grades_path, lines)
    log(f'  wrote {grades_path} and {dem_dir / "report.md"}')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cities', nargs='+', help='run directory names under --runs-root')
    ap.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--posts-m', type=float, default=POSTS_M)
    ap.add_argument('--fit-half-window-m', type=float, default=FIT_HALF_WINDOW_M)
    ap.add_argument('--verify', action='store_true',
                    help='re-hash the cached tiles against tiles.json and exit (1 = mismatch)')
    ap.add_argument('--no-fetch', action='store_true',
                    help='use the cached tiles only (verified first); never touch the network')
    ap.add_argument('--summary', type=Path, default=None,
                    help='summary CSV (default: <runs-root>/_summary/dem_grade.csv)')
    args = ap.parse_args(argv)
    if args.verify:
        bad = False
        for city in args.cities:
            problems = verify_city(args.runs_root / city / 'dem')
            print(f'{city}: ' + ('ok' if not problems else '; '.join(problems)))
            bad |= bool(problems)
        sys.exit(1 if bad else 0)
    rows = []
    for city in args.cities:
        rows.extend(run_city(city, args.runs_root, args.posts_m, args.fit_half_window_m,
                             no_fetch=args.no_fetch))
    out = args.summary or args.runs_root / '_summary' / 'dem_grade.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
