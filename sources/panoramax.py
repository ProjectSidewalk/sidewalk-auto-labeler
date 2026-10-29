"""Panoramax imagery source: coverage via z15 vector tiles, imagery via the STAC API.

Panoramax (panoramax.fr) is the federated street-level imagery commons started by IGN
(the French national mapping agency) and OpenStreetMap France. Every instance runs the
open-source GeoVisio stack and exposes the same STAC API; the meta-catalog at
api.panoramax.xyz federates all of them (23 instances as of 2026-09, from IGN and OSM-FR
at ~56M pictures each down to single-contributor instances such as the US Northwest one).
No credentials are needed to read. Pictures are CC BY-SA 4.0 on nearly every instance
and Etalab Open License 2.0 on IGN's; the per-picture license is recorded in each record.

Coverage: the catalog's vector tiles carry individual pictures only at z15 (the highest
zoom it serves — z16 is refused), as points in the `pictures` layer with `id`, `ts`,
`heading`, `type` (`equirectangular` / `flat`), `model` and `h_pixel_density` — so the
360° filter is applied during enumeration with no per-picture API calls, exactly as the
Mapillary source does with `is_pano`. Tiles with no coverage answer 204 with an empty
body.

Fetching: one STAC item request per picture returns position, capture time, heading
(`view:azimuth`), pitch/roll, the camera's interior orientation (field of view and
sensor dimensions), producer, license and the `hd` asset — a plain, unsigned URL on the
origin instance serving the original upload (a GoPro Max pano is 5760×2880, ~3 MB).

Orientation: GeoVisio defines a picture's heading as the bearing of the picture's
center, which for an equirectangular is the center column — the same convention as GSV
and Mapillary and exactly what Project Sidewalk's panoX -> heading math assumes — so
images are never rotated and `view:azimuth` is recorded as `camera_heading`.
`pers:pitch`/`pers:roll` are recorded as given, and unlike GSV (always present) or
Mapillary (always absent) Panoramax is *mixed*: measured over 757 live panos across six
French cities, 72% carry neither key and 28% carry both — of those only a fifth are
0.0/0.0, the rest real tilt reaching -24.6 deg pitch and +-17 deg roll. See geo.pano_pose
for what that mixture means for --pose-ablation.

`PANORAMAX_API_URL` overrides the catalog root, e.g. to run against a single instance
(including a self-hosted one) instead of the federation.

Politeness (issue #161). Panoramax's maintainers asked us to pull fewer full-resolution
pictures and to pace downloads at about one per second; before #161 this module made two
requests per pano from 50 threads with no limit. Now:

- Every request goes through `_get(url, kind, ...)`, which waits on a per-kind
  `sources.ratelimit.RateLimiter`: `picture` (image assets) 1/s, `metadata` (item GETs,
  `/search`, the landing-page probe) 1/s, `tile` (coverage MVTs) 5/s. Env vars
  PANORAMAX_PICTURES_PER_SECOND / _METADATA_ / _TILES_ override them (0 = unlimited, for a
  private mirror). A 429 or 503 pauses the whole kind for min(Retry-After, 300) s, so every
  thread backs off, not only the one that was told. Limits are per process.
- Metadata is batched: main.py hands the processing order to `queue_hint`, and a cache miss
  in `fetch_item` fetches that picture plus the next <= 99 hinted ones in one `/search?ids=`
  request (~1,750 metadata requests for a Lyon-size run instead of 174k). Without a hint
  (export_benchmark, reinfer) it is one `/pictures/{id}` GET per picture, as before.
- `IMAGE_ASSET` ('hd' default, main.py --panoramax-asset) picks the downloaded asset. Under
  'sd' the 2048-px derivative is downloaded but the record keeps the SENSOR frame
  (`sensor_array_dimensions`) -- the frame PS's viewer and the PS pano store use -- and a
  picture that cannot be read that way is downloaded in HD instead (counted as an
  asset fallback). See docs/panoramax-asset-study.md for whether SD costs detections.
- MAX_PROCESSING_CONCURRENCY caps main.py's pool: threads beyond a few only queue at the
  limiter while holding connections open.
"""
import email.utils
import os
import random
import threading
import time
from collections import OrderedDict
from io import BytesIO
from urllib.parse import urlsplit

import requests
from PIL import Image
from shapely.geometry import Point

from sources import TARGET_IMAGE_SIZE as TARGET_SIZE
# MVT plumbing is shared with the Mapillary source: same slippy-map math, same decoder.
from sources.mapillary import tile_point_to_lonlat, _decode_tile
from sources import mapillary as _mapillary
from sources.ratelimit import RateLimiter

NAME = 'panoramax'
COVERAGE_TILE_ZOOM = 15  # the only zoom whose tiles carry individual pictures
DEFAULT_API_URL = 'https://api.panoramax.xyz/api'
API_URL_ENV_VAR = 'PANORAMAX_API_URL'
# The API asks production tools to identify themselves explicitly.
USER_AGENT = 'sidewalk-auto-labeler/1.0 (+https://github.com/ProjectSidewalk/sidewalk-auto-labeler)'
ATTEMPTS = 3

# Request pacing per kind (issue #161), in requests per second. `picture` is the one the
# maintainers asked about (full-resolution downloads, ~1/s); `metadata` covers item GETs,
# /search and the landing-page probe; `tile` the coverage MVTs, which are small and served
# from cache. 0 in an env var means unlimited (a private mirror or a local instance).
RATE_DEFAULTS = {'picture': 1.0, 'metadata': 1.0, 'tile': 5.0}
RATE_ENV_VARS = {'picture': 'PANORAMAX_PICTURES_PER_SECOND',
                 'metadata': 'PANORAMAX_METADATA_PER_SECOND',
                 'tile': 'PANORAMAX_TILES_PER_SECOND'}
# A 429/503 is the server saying it is overloaded: the whole kind waits Retry-After, capped
# so a mis-set header cannot stall a run for hours, or DEFAULT_PAUSE_S when it names none.
OVERLOAD_STATUSES = (429, 503)
MAX_PAUSE_S = 300
DEFAULT_PAUSE_S = 30

# More threads than this only queue at the picture limiter (1/s) while each holds a decoded
# 4096x2048 image and an open connection; main.py takes min(--processing-concurrency, this).
MAX_PROCESSING_CONCURRENCY = 4

# /search?ids= returns full items for many pictures in one request (100 measured to work on
# the federation catalog, 2026-10-28). The hinted cache holds at most CACHE_CHUNKS of them.
SEARCH_CHUNK = 100
CACHE_CHUNKS = 4

# Which asset fetch_pano downloads: 'hd' (the original upload) or 'sd' (the 2048-px-wide
# derivative Panoramax's own viewer loads). Set by main.py --panoramax-asset and bound in
# the run manifest (`panoramax_asset`); manifests from before #161 read as 'hd'.
IMAGE_ASSETS = ('hd', 'sd')
IMAGE_ASSET = 'hd'
MANIFEST_ASSET_KEY = 'panoramax_asset'
LEGACY_MANIFEST_BINDINGS = {MANIFEST_ASSET_KEY: 'hd'}

# The catalog answers 404 for a picture id it doesn't know (deleted by its uploader, an
# instance that left the federation, or never valid) — verified live for both a
# well-formed unknown UUID and a malformed id. A 404 is not retried within one call (it is
# one request, and the next one would say the same), but fetch_pano reports it as a
# RETRYABLE failure, never a cached skip: Panoramax's infrastructure is young and a
# federated instance can drop out and come back, so a 404 today is not proof the pixels
# are gone for good (Jon's call, 2026-10-05, reversing #137's permanent skip). The same
# holds for a 404 on the image asset itself (_download_image). The cost of a picture
# that really is gone is one request per resume, forever — cheap, and it keeps the
# decision reversible. export_benchmark.py still reads fetch_item's `gone` as GONE.
GONE_STATUSES = (404,)

# A full 360x180 equirectangular is exactly 2:1 (same reasoning as the Mapillary source:
# resizing a cropped-FOV upload to 4096x2048 would shift every detection's pano_y).
ASPECT_TOLERANCE = 0.02

# Panoramax contributors, like Mapillary's, re-drive the same streets and shoot at
# 1-3 m intervals, so raw 360 coverage is far denser than GSV's ~10 m. Thinning keeps
# the best pano per grid cell: newest capture wins, higher pixel density (px/degree —
# the tile's proxy for camera resolution) breaks ties. Same default as Mapillary,
# for the same reasons; override per run with --thin-spacing (0 disables).
THIN_CELL_METERS = 5

# Bulky per-picture tiling descriptors the viewer needs and nobody else does.
VOLATILE_PROPERTY_KEYS = {'tiles:tile_matrix_sets'}

# Written into source_metadata from the item itself, so a property of the same (unprefixed,
# therefore non-STAC) name never silently shadows them — see provenance_fields.
ITEM_LEVEL_KEYS = {'collection', 'providers', 'hd_url'}


def api_url():
    return os.environ.get(API_URL_ENV_VAR, DEFAULT_API_URL).rstrip('/')


def prepare():
    """Probe the STAC landing page so a wrong API root fails here, before the model loads.

    No credentials to check — the catalog is anonymous — but the root is configurable, and
    a mistyped one is otherwise invisible: the tile endpoint answers 404 for a bad path
    and the scan would just report zero panos. One request settles it, and it works for a
    single instance as well as the federation (panoramax.ign.fr and .openstreetmap.fr both
    answer a STAC Catalog here).
    """
    root = api_url()
    try:
        response = _get(root, 'metadata', timeout=30)
        response.raise_for_status()
        catalog = response.json()
    except Exception as e:
        raise SystemExit(
            f"\u274c Panoramax API root {root} is not reachable ({e}).\n"
            f"   Check {API_URL_ENV_VAR}, or unset it to use the federation catalog\n"
            f"   ({DEFAULT_API_URL})."
        )
    if not catalog.get('stac_version'):
        raise SystemExit(
            f"\u274c {root} answered, but it is not a STAC catalog (no stac_version).\n"
            f"   {API_URL_ENV_VAR} should point at an instance's /api root, e.g.\n"
            f"   https://panoramax.openstreetmap.fr/api."
        )


def _headers():
    return {'User-Agent': USER_AGENT}


def rate_limits():
    """{kind: requests per second} in force, from RATE_ENV_VARS or RATE_DEFAULTS; None means
    unlimited (an env value of 0). Recorded in each run entry (`panoramax_rate_limits`)."""
    limits = {}
    for kind, default in RATE_DEFAULTS.items():
        raw = os.environ.get(RATE_ENV_VARS[kind])
        if raw is None or raw.strip() == '':
            limits[kind] = default
            continue
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit(f"\u274c {RATE_ENV_VARS[kind]}={raw!r} is not a number "
                             f"(requests per second; 0 = unlimited).")
        if value < 0:
            raise SystemExit(f"\u274c {RATE_ENV_VARS[kind]} must be >= 0 (0 = unlimited).")
        limits[kind] = value or None
    return limits


_LIMITERS = {}
_limiters_lock = threading.Lock()


def limiter(kind):
    """The process-wide RateLimiter for one request kind, built on first use from
    rate_limits() (so env overrides set before the first request take effect)."""
    with _limiters_lock:
        if kind not in _LIMITERS:
            _LIMITERS[kind] = RateLimiter(rate_limits()[kind])
        return _LIMITERS[kind]


def retry_after_seconds(headers):
    """The pause a 429/503 asks for: Retry-After as seconds or as an HTTP date, capped at
    MAX_PAUSE_S; DEFAULT_PAUSE_S when the header is absent or unreadable."""
    value = headers.get('Retry-After') if hasattr(headers, 'get') else None
    seconds = None
    if value is not None:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            try:
                seconds = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, IndexError, OverflowError):
                seconds = None
    if seconds is None:
        seconds = DEFAULT_PAUSE_S
    return max(0.0, min(float(seconds), MAX_PAUSE_S))


def _get(url, kind, **kwargs):
    """One GET to a Panoramax host, paced by the limiter for `kind`. Every attempt pays,
    retries included. A 429/503 answer pauses the whole kind (every thread) for the time
    the server asks, then is returned as-is: the caller's raise_for_status turns it into an
    ordinary failed attempt, and its next attempt waits out the pause. Calls requests.get,
    so the tests' monkeypatches see every request."""
    lim = limiter(kind)
    lim.wait()
    kwargs.setdefault('headers', _headers())
    response = requests.get(url, **kwargs)
    if getattr(response, 'status_code', None) in OVERLOAD_STATUSES:
        lim.pause(retry_after_seconds(getattr(response, 'headers', None)))
    return response


def min_seconds_per_pano():
    """The floor the picture limiter puts under each pano, for main.py's --scan-only
    estimate: one asset download per pano at the `picture` rate."""
    rate = rate_limits()['picture']
    return 1.0 / rate if rate else 0.0


def fetch_panos_for_tile(tile_x, tile_y, area_shape):
    """
    Worker function for a single z15 coverage tile. Returns {picture_id: (lat, lon, ...)}
    for 360° pictures inside the area, or None if the tile failed after retries.
    """
    url = f'{api_url()}/map/{COVERAGE_TILE_ZOOM}/{tile_x}/{tile_y}.mvt'
    for attempt in range(ATTEMPTS):
        try:
            response = _get(url, 'tile', timeout=60)
            # 204 is the catalog's documented — and measured — answer for a tile with no
            # coverage. A 404 is NOT: it means the path is wrong (a mistyped
            # PANORAMAX_API_URL), so it must surface as a failed tile rather than as an
            # empty one. Same reason raise_for_status() comes before the empty-body check:
            # a 502 with no body is a failure to retry, not a tile without pictures.
            if response.status_code == 204:
                return {}
            response.raise_for_status()
            if not response.content:
                return {}
            return panos_from_tile(response.content, tile_x, tile_y, area_shape)
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def panos_from_tile(tile_bytes, tile_x, tile_y, area_shape):
    """Decodes one coverage tile into {picture_id: (lat, lon, captured_at, pixel_density)}
    for equirectangular pictures in the area. main.py only relies on the first two
    elements; captured_at (the tile's ISO timestamp string — all UTC, so it sorts) and
    pixel_density feed thin_panos."""
    layer = _decode_tile(tile_bytes).get('pictures')
    if not layer:
        return {}
    extent = layer.get('extent', 4096)
    panos = {}
    for feature in layer['features']:
        properties = feature.get('properties', {})
        if properties.get('type') != 'equirectangular':
            continue
        px, py = feature['geometry']['coordinates']
        lon, lat = tile_point_to_lonlat(px, py, tile_x, tile_y, COVERAGE_TILE_ZOOM, extent)
        if Point(lon, lat).within(area_shape):
            panos[str(properties['id'])] = (
                lat, lon, properties.get('ts') or '', properties.get('h_pixel_density') or 0)
    return panos


def thin_panos(panos, cell_meters=None):
    """Keeps one pano per grid cell (default THIN_CELL_METERS): newest capture wins,
    higher pixel density breaks ties. The cell logic is the Mapillary source's; only
    the tuple contents differ (ISO timestamp + px/degree instead of ms + quality)."""
    return _mapillary.thin_panos(panos, cell_meters or THIN_CELL_METERS)


def fetch_pano(pano_id, lat, lon):
    """
    Fetches the STAC item and the equirectangular image for one Panoramax picture (see
    the interface contract in sources/__init__.py). Metadata is validated before the
    (much more expensive) image download.

    The record's width/height are the frame PS's normalized -> pixel transform must match.
    Under IMAGE_ASSET 'hd' that is the downloaded original's own dimensions, as always.
    Under 'sd' it is the item's `sensor_array_dimensions`, never the 2048-px derivative's:
    PS's Panoramax viewer sizes its PanoData from the sensor dimensions and the PS pano
    store keeps the HD file, so an SD-sized record would put every label in a frame
    nothing else uses. A picture whose item lacks 2:1 sensor dimensions, has no `sd`
    asset, or whose SD answers 404 / undecodable / not 2:1 is downloaded in HD instead
    and counted (`asset_fallbacks`). Detections are normalized, so the asset changes no
    downstream code; under 'sd' source_metadata says which asset was read.
    """
    item, gone = fetch_item(pano_id)
    if gone:
        # Retryable, not a cached skip — see GONE_STATUSES.
        return {'status': 'failure',
                'reason': 'Picture not found on Panoramax (HTTP 404; retried next run)'}
    if item is None:
        return {'status': 'failure', 'reason': 'STAC item unavailable (transient?)'}
    props = item.get('properties') or {}
    orientation = props.get('pers:interior_orientation') or {}
    # Belt and suspenders over the tile scan's type == equirectangular filter — so it only
    # fires when the item positively contradicts it. `pers:interior_orientation` is
    # optional and `field_of_view` within it more so, and a skip here is cached forever:
    # treating "absent" as "not 360" would permanently drop a picture the scan already
    # certified, with no way to retry it. The downloaded image's own 2:1 check below is
    # the real backstop.
    field_of_view = orientation.get('field_of_view')
    if field_of_view is not None and field_of_view != 360:
        return {'status': 'skipped',
                'reason': f"Not a 360 picture (field_of_view={field_of_view})"}
    if not props.get('datetime'):
        return {'status': 'skipped', 'reason': 'No capture timestamp'}
    # PS's pano_x -> heading math requires camera_heading.
    if props.get('view:azimuth') is None:
        return {'status': 'skipped', 'reason': 'No heading (view:azimuth)'}
    url = image_url(item)
    if not url:
        return {'status': 'skipped', 'reason': 'Metadata missing the hd image asset'}
    declared = orientation.get('sensor_array_dimensions')
    if declared and len(declared) == 2 and not _is_equirectangular(*declared):
        return {'status': 'skipped',
                'reason': f'Not a full 360x180 equirectangular ({declared[0]}x{declared[1]})'}

    asset = 'hd'
    if IMAGE_ASSET == 'sd':
        sd_url = image_url(item, asset='sd')
        if sd_url and declared and len(declared) == 2:
            downloaded, undecodable, detail = _download_image(sd_url)
            if downloaded is None and not undecodable and not _is_asset_404(detail):
                return {'status': 'failure',
                        'reason': 'Failed to download sd image' + (f' ({detail})' if detail else '')}
            if downloaded is not None and _is_equirectangular(*downloaded[1]):
                image, size = downloaded
                return {'status': 'success',
                        'pano': build_pano_record(pano_id, lat, lon, item, int(declared[0]),
                                                  int(declared[1]),
                                                  image_info=_image_info('sd', sd_url, size)),
                        'image': image}
        _count_fallback()   # no usable SD for this picture: read it in HD

    downloaded, undecodable, detail = _download_image(url)
    if undecodable:
        return {'status': 'skipped',
                'reason': detail or 'Image asset is not a decodable image'}
    if downloaded is None:
        return {'status': 'failure',
                'reason': 'Failed to download equirectangular image'
                          + (f' ({detail})' if detail else '')}
    image, (width, height) = downloaded
    if not _is_equirectangular(width, height):
        return {'status': 'skipped',
                'reason': f'Not a full 360x180 equirectangular ({width}x{height})'}

    image_info = None if IMAGE_ASSET == 'hd' else _image_info(asset, url, (width, height))
    return {'status': 'success',
            'pano': build_pano_record(pano_id, lat, lon, item, width, height,
                                      image_info=image_info),
            'image': image}


def _is_asset_404(detail):
    return bool(detail) and detail.startswith('Image asset HTTP 404')


def _image_info(asset, url, size):
    """source_metadata keys written under IMAGE_ASSET 'sd' (absent under 'hd', so resumed
    HD runs keep their line shape): which asset was read, from where, and its real size."""
    return {'image_asset': asset, 'image_url': url,
            'image_dimensions_downloaded': [int(size[0]), int(size[1])]}


_counter_lock = threading.Lock()
_COUNTERS = {'asset_fallbacks': 0}


def _count_fallback():
    with _counter_lock:
        _COUNTERS['asset_fallbacks'] += 1


def _is_equirectangular(width, height):
    return bool(width and height) and abs(width - 2 * height) <= ASPECT_TOLERANCE * width


def image_url(item, asset='hd'):
    """The URL of one asset. 'hd': the original upload (the `hd` asset, else GeoVisio's own
    image link). 'sd': the 2048-px-wide derivative, or None when the item has none."""
    assets = item.get('assets') or {}
    if asset == 'sd':
        return (assets.get('sd') or {}).get('href')
    if asset != 'hd':
        raise ValueError(f"unknown Panoramax asset {asset!r} (expected one of {IMAGE_ASSETS})")
    hd = assets.get('hd') or {}
    return hd.get('href') or (item.get('properties') or {}).get('geovisio:image')


# --- batched metadata (issue #161) --------------------------------------------------------
# queue_hint stores the processing order; fetch_item serves hinted pictures from a small
# cache filled by /search?ids= in chunks. One lock guards everything, and is held across the
# search request itself: at 1 metadata request/s the lock is never the bottleneck, and
# holding it means two threads can never search for the same chunk.
_hint_lock = threading.Lock()
_HINT = {'ids': None, 'positions': {}, 'cursor': 0, 'requested': set(), 'cache': OrderedDict()}


def queue_hint(ordered_ids):
    """Optional source hook (sources/__init__.py): main.py passes the ids it is about to
    process, in submission order, right before the processing pool starts. Replaces any
    earlier hint."""
    ids = [str(i) for i in ordered_ids]
    with _hint_lock:
        _HINT.update(ids=ids, positions={pid: i for i, pid in enumerate(ids)}, cursor=0,
                     requested=set(), cache=OrderedDict())


def clear_hint():
    queue_hint([])
    with _hint_lock:
        _HINT['ids'] = None


def fetch_items(ids):
    """{picture_id: STAC item} for up to SEARCH_CHUNK ids in one /search request (`limit`
    is always the chunk's length: the endpoint's default page is shorter), or None if the
    request failed after ATTEMPTS. An id missing from the answer is simply absent; the
    caller must not read that as "gone" (fetch_item falls back to the per-id GET)."""
    ids = list(ids)
    if not ids:
        return {}
    url = f'{api_url()}/search'
    params = {'ids': ','.join(ids), 'limit': len(ids)}
    for attempt in range(ATTEMPTS):
        try:
            response = _get(url, 'metadata', params=params, timeout=60)
            response.raise_for_status()
            wanted = set(ids)
            return {str(f['id']): f for f in (response.json().get('features') or [])
                    if str(f.get('id')) in wanted}
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def _hinted_item(picture_id):
    """The item for a hinted picture from the cache, searching on a miss for it plus the
    next <= SEARCH_CHUNK-1 hinted, not yet requested ids from the cursor. None (so the
    caller falls back to the per-id GET) when the picture was never hinted, the search
    failed, the answer did not include it, or it was requested before and is not cached."""
    with _hint_lock:
        if _HINT['ids'] is None or picture_id not in _HINT['positions']:
            return None
        cache = _HINT['cache']
        if picture_id in cache:
            return cache.pop(picture_id)
        ids, requested = _HINT['ids'], _HINT['requested']
        if picture_id in requested:
            # Already asked for and not cached: the search left it out (or it was evicted).
            # Searching for it again alone would cost the same request as the per-id GET.
            return None
        chunk = [picture_id]
        requested.add(picture_id)
        cursor = _HINT['cursor']
        while cursor < len(ids) and len(chunk) < SEARCH_CHUNK:
            pid = ids[cursor]
            cursor += 1
            if pid not in requested:
                chunk.append(pid)
                requested.add(pid)
        _HINT['cursor'] = cursor
        found = fetch_items(chunk)
        if not found:
            return None
        for pid in chunk[1:]:
            if pid in found:
                cache[pid] = found[pid]
        while len(cache) > CACHE_CHUNKS * SEARCH_CHUNK:
            cache.popitem(last=False)   # evicted ids are searched again on their own miss
        return found.get(picture_id)


def fetch_item(picture_id):
    """(STAC item, gone) for one picture: from the hinted /search cache when main.py gave
    a queue_hint (see _hinted_item), else -- and whenever the batch did not return the
    picture -- one /pictures/{id} GET, as before #161.

    `gone` is True only when the catalog positively says it has no such picture (404),
    which no amount of retrying will fix, so the retry loop short-circuits. A transient
    failure (network, 5xx) returns (None, False) after ATTEMPTS tries — the same
    "source decayed" vs "our fetch failed" distinction the Mapillary source draws.
    """
    item = _hinted_item(picture_id)
    if item is not None:
        return item, False
    return _fetch_item_by_id(picture_id)


def _fetch_item_by_id(picture_id):
    """fetch_item's per-id path: one /pictures/{id} GET (with retries), never the cache."""
    url = f'{api_url()}/pictures/{picture_id}'
    for attempt in range(ATTEMPTS):
        try:
            response = _get(url, 'metadata', timeout=30)
            if response.status_code in GONE_STATUSES:
                return None, True
            response.raise_for_status()
            return response.json(), False
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None, False


def _download_image(url):
    """((image, (original_width, original_height)), permanent, detail) for one picture's
    asset. `detail` is None on success and otherwise the reason to log: an asset 404 has
    its own failure reason (so a run log can count them -- the transient-404 question on
    #127), undecodable bytes their own skip reason, and any other retryable failure carries
    the last attempt's exception, so a server-side truncation reads apart from a flaky
    network.

    Splits the two failure kinds the retry loop used to conflate, for the same reason
    fetch_item does: main.py caches a deterministic skip and retries a failure forever.

    - Network/HTTP failures are retryable and return (None, False, last error) after
      ATTEMPTS tries.
    - A 404 on the asset returns (None, False, "Image asset HTTP 404") after ONE request:
      retryable on the next run, but not hammered within this one. The `hd` URL is plain
      and unsigned (unlike Mapillary's signed, expiring thumbnail), so #137 read a 404 as
      "pixels gone" and cached it; reversed 2026-10-05 because Panoramax instances are
      young and flaky (see GONE_STATUSES). A picture that really is gone costs one request
      per resume.
    - A decode failure on complete bytes is permanent. Bytes that arrived whole and that
      PIL cannot read (sources.mapillary.decode_failure_is_permanent) — garbage, a
      decompression bomb past PIL's ceiling — will not become an image on the fourth
      try, so they are a permanent skip on the first attempt.
    - A body that did NOT arrive whole is retryable and never cached (#127): a byte count
      other than its Content-Length (sources.mapillary.body_is_complete), or PIL's
      truncated-file OSError ("image file is truncated" / "Truncated File Read"). So is
      any non-decode exception raised while decoding, e.g. a MemoryError under load.
      Before #127 every one of these was cached as a skip, so a transfer cut short lost
      the pano for good. The rule is shared with Mapillary; only the 404 differs: Mapillary
      retries it within the call (its signed URL may have expired), Panoramax returns it
      after one request. Both leave it uncached.
    """
    last_error = None
    for attempt in range(ATTEMPTS):
        try:
            response = _get(url, 'picture', timeout=180)
            if response.status_code in GONE_STATUSES:
                return None, False, f"Image asset HTTP {response.status_code}"
            response.raise_for_status()
            payload = response.content
            if not _mapillary.body_is_complete(response):
                raise ValueError(f"short body: Content-Length "
                                 f"{response.headers.get('Content-Length')}, got {len(payload)} bytes")
            # Past this point the whole body is in hand, so a permanent decode failure is
            # about the bytes; a truncation or any other exception is retried.
            try:
                image = Image.open(BytesIO(payload)).convert('RGB')
            except Exception as e:
                if _mapillary.decode_failure_is_permanent(e):
                    return None, True, f"Undecodable image bytes ({_mapillary.describe_exception(e)})"
                raise
        except Exception as e:
            last_error = _mapillary.describe_exception(e)
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
            continue
        original_size = image.size
        if image.size != TARGET_SIZE:
            image = image.resize(TARGET_SIZE, Image.BILINEAR)
        return (image, original_size), False, None
    return None, False, last_error


def provenance_fields(item):
    """Extended-provenance keys added to a pano block: curated camera fields (top-level,
    for easy analysis — make/model is the signal that explains image quality) plus the
    federation member holding the picture and `source_metadata`, the STAC item's
    properties (EXIF included: that is where the camera's own GPano pose lives) minus the
    viewer's tiling descriptors, with the collection and providers folded in."""
    props = item.get('properties') or {}
    orientation = props.get('pers:interior_orientation') or {}
    url = image_url(item)
    return {
        'camera_make': orientation.get('camera_manufacturer'),
        'camera_model': orientation.get('camera_model'),
        'camera_type': 'equirectangular',
        'panoramax_instance': urlsplit(url).netloc if url else None,
        # Item-level fields folded in beside the properties. STAC namespaces its property
        # keys (`geovisio:`, `pers:`, `view:`) so these three cannot realistically collide,
        # but state the precedence rather than leaving it to dict-literal ordering: the
        # item's own collection/providers are authoritative, and hd_url is ours.
        'source_metadata': {
            **{k: v for k, v in sorted(props.items())
               if k not in VOLATILE_PROPERTY_KEYS and k not in ITEM_LEVEL_KEYS},
            'collection': item.get('collection'),
            'providers': item.get('providers'),
            'hd_url': url,
        },
    }


def _as_float(value):
    return None if value is None else float(value)


def build_pano_record(pano_id, lat, lon, item, width, height, image_info=None):
    """
    Builds the Stage-1 JSONL 'pano' block from a STAC item, in the same internal
    field-name contract as the GSV and Mapillary sources (send_to_ps.transform_pano maps
    it onto the Project Sidewalk reader; `source: "panoramax"` passes through to PS's
    pano_source enum). width/height are the record's pixel frame (fetch_pano: the HD
    download's own size, or the sensor dimensions under 'sd'). `image_info` (the 'sd'
    run's image_asset / image_url / image_dimensions_downloaded) is merged into
    source_metadata; None writes nothing new.
    """
    props = item.get('properties') or {}
    coordinates = (item.get('geometry') or {}).get('coordinates')
    if coordinates:  # the item's position when available; tile position otherwise
        lon, lat = coordinates[0], coordinates[1]
    captured = props['datetime']  # RFC 3339; the year-month prefix is all PS stores
    # STAC lets an item name several providers, and instances routinely list themselves
    # as the `host` alongside the contributor — so when `geovisio:producer` is absent,
    # take the entry that declares the producer role before the first merely-named one,
    # or the credit becomes whoever happens to be listed first.
    named = [p for p in item.get('providers') or [] if p.get('name')]
    producer = props.get('geovisio:producer') or next(
        (p['name'] for p in named if 'producer' in (p.get('roles') or [])),
        next((p['name'] for p in named), None))
    license_id = props.get('license') or 'CC-BY-SA-4.0'
    provenance = provenance_fields(item)
    if image_info:
        provenance['source_metadata'].update(image_info)
    return {
        "panorama_id": pano_id,
        "capture_date": captured[:7],
        "width": int(width),
        "height": int(height),
        "lat": float(lat),
        "lng": float(lon),
        "camera_heading": float(props['view:azimuth']),
        "camera_pitch": _as_float(props.get('pers:pitch')),
        "camera_roll": _as_float(props.get('pers:roll')),
        # The producer's bare name, which is what PS's pano_data.copyright holds for this
        # source; PS composes "© <name> · Panoramax · <licence>" itself wherever it shows
        # its own copy of the imagery. None when the item names no producer — the licence
        # is in `license` below and the provider follows from `source`.
        "copyright": producer,
        "source": "panoramax",
        "sequence_id": item.get('collection'),
        "license": license_id,
        **provenance,
        "history": [],
        "links": []
    }


# --- generic source hooks main.py reads (issue #161) --------------------------------------

def manifest_bindings():
    """{manifest key: value} a run directory is bound to (main.bind_source_settings): the
    asset, since a resume under another asset would mix two image qualities in one
    results.jsonl. LEGACY_MANIFEST_BINDINGS gives the value a pre-#161 run with records
    was made under (hd: the only asset there was)."""
    return {MANIFEST_ASSET_KEY: IMAGE_ASSET}


def run_record():
    """Merged into the manifest's run entry: the pacing in force, the asset, how many
    pictures fell back to HD under 'sd', and per-kind request counts for this process."""
    with _limiters_lock:
        requests_by_kind = {kind: lim.stats() for kind, lim in sorted(_LIMITERS.items())}
    with _counter_lock:
        fallbacks = _COUNTERS['asset_fallbacks']
    return {'panoramax_rate_limits': rate_limits(),
            MANIFEST_ASSET_KEY: IMAGE_ASSET,
            'asset_fallbacks': fallbacks,
            'panoramax_requests': requests_by_kind}
