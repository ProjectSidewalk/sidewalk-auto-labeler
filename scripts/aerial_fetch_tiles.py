"""Fetch slippy-map aerial tiles for an area into Tile2Net's `--input` layout (issue #104).

Tile2Net (VIDA-NYU/tile2net) downloads its own imagery for the sources it ships, but its
Oregon source is commented out upstream ("Oregon also has some SSL issues" -- and on
2026-09-28 the OSIP server's certificate had expired two days earlier), so Bend's tiles are
fetched here instead and handed to `tile2net generate --input <dir>/{z}/{x}/{y}.png`.
Stdlib only, so it runs in any Python on the GPU host. TLS is always verified.

The default source is the City of Bend's 2019 orthoimagery cache on ArcGIS Online
(`City_of_Bend_2019_Imagery`, cached to level 21), which sits on the standard Google/OSM
tile grid (origin -20037508.34, 256 px), so its `tile/{z}/{y}/{x}` endpoint IS a slippy
tile. Every tile's sha256 goes into tiles.csv, and their digest into manifest.json, so the
exact pixels the study used are pinned.

Politeness: a small fixed worker count (default 4), retries with backoff, and resumable
(existing files are skipped), so a re-run never re-downloads.

Usage (on the GPU host):
    python aerial_fetch_tiles.py --area area.geojson --zoom 19 --out tiles/
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BEND_2019 = ('https://tiles.arcgis.com/tiles/JisFYcK2mIVg9ueP/arcgis/rest/services/'
             'City_of_Bend_2019_Imagery/MapServer/tile/{z}/{y}/{x}')
USER_AGENT = 'sidewalk-auto-labeler aerial study (issue 104; Makeability Lab, UW)'


def lnglat_to_tile(lng, lat, z):
    """(x, y) slippy tile holding a WGS84 point.

    Example:
        >>> lnglat_to_tile(-121.3074, 44.0619, 19)
        (85477, 190516)
    """
    n = 2 ** z
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def bbox_of_geojson(path):
    """(west, south, east, north) of every coordinate in a (Multi)Polygon geojson."""
    with open(path, encoding='utf-8') as f:
        g = json.load(f)
    if g.get('type') == 'FeatureCollection':
        geoms = [ft['geometry'] for ft in g['features']]
    elif g.get('type') == 'Feature':
        geoms = [g['geometry']]
    else:
        geoms = [g]
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for cc in c:
                walk(cc)
    for geom in geoms:
        walk(geom['coordinates'])
    return min(xs), min(ys), max(xs), max(ys)


def fetch_one(url, dest, retries=5):
    """Download url to dest (atomic). Returns (status, bytes, sha256)."""
    if dest.exists() and dest.stat().st_size > 0:
        data = dest.read_bytes()
        return 'cached', len(data), hashlib.sha256(data).hexdigest()
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                ctype = r.headers.get('Content-Type', '')
            if not ctype.startswith('image/'):
                return f'not-image:{ctype}', 0, None
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.part')
            tmp.write_bytes(data)
            tmp.replace(dest)
            return 'fetched', len(data), hashlib.sha256(data).hexdigest()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 'missing', 0, None
            if attempt == retries - 1:
                return f'http-{e.code}', 0, None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == retries - 1:
                return f'error:{type(e).__name__}', 0, None
        time.sleep(delay)
        delay *= 2
    return 'error', 0, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--area', help='area geojson; its bbox is fetched')
    ap.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'),
                    help='fetch this WGS84 bbox instead of an area file')
    ap.add_argument('--zoom', type=int, default=19)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--url', default=BEND_2019)
    ap.add_argument('--ext', default='png')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--margin-tiles', type=int, default=1)
    args = ap.parse_args(argv)
    if bool(args.area) == bool(args.bbox):
        ap.error('give exactly one of --area / --bbox')

    w, s, e, n = args.bbox if args.bbox else bbox_of_geojson(args.area)
    x0, y0 = lnglat_to_tile(w, n, args.zoom)
    x1, y1 = lnglat_to_tile(e, s, args.zoom)
    m = args.margin_tiles
    xs = range(x0 - m, x1 + m + 1)
    ys = range(y0 - m, y1 + m + 1)
    jobs = [(x, y) for x in xs for y in ys]
    print(f'bbox {w},{s},{e},{n} -> z{args.zoom} x {xs.start}..{xs.stop - 1} '
          f'y {ys.start}..{ys.stop - 1}: {len(jobs)} tiles', flush=True)
    counts, rows, t0 = {}, [], time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_one, args.url.format(z=args.zoom, x=x, y=y),
                          args.out / str(args.zoom) / str(x) / f'{y}.{args.ext}'): (x, y)
                for x, y in jobs}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            x, y = futs[fut]
            status, nbytes, sha = fut.result()
            counts[status.split(':')[0]] = counts.get(status.split(':')[0], 0) + 1
            rows.append((x, y, status, nbytes, sha))
            if k % 2000 == 0:
                print(f'{k}/{len(jobs)} {counts} {time.time() - t0:.0f} s', flush=True)
    rows.sort()
    manifest = {
        'url_template': args.url, 'zoom': args.zoom,
        'area': str(args.area) if args.area else None,
        'bbox_wgs84': [w, s, e, n], 'x_range': [xs.start, xs.stop - 1],
        'y_range': [ys.start, ys.stop - 1], 'n_tiles': len(jobs), 'counts': counts,
        'workers': args.workers,
        'fetched_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'seconds': round(time.time() - t0, 1),
        'tiles_sha256': hashlib.sha256(
            '\n'.join(f'{x},{y},{sha}' for x, y, _st, _nb, sha in rows).encode()).hexdigest(),
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    with open(args.out / 'tiles.csv', 'w', encoding='utf-8') as f:
        f.write('x,y,status,bytes,sha256\n')
        for r in rows:
            f.write(','.join('' if v is None else str(v) for v in r) + '\n')
    print(json.dumps(manifest, indent=1))
    bad = any(k.startswith(('error', 'http', 'not-image')) for k in counts)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
