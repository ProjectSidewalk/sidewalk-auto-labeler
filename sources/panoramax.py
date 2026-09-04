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
`pers:pitch`/`pers:roll` are recorded as given (consumer 360 cameras upload 0.0).

`PANORAMAX_API_URL` overrides the catalog root, e.g. to run against a single instance
(including a self-hosted one) instead of the federation.
"""
import os
import random
import time
from io import BytesIO
from urllib.parse import urlsplit

import requests
from PIL import Image
from shapely.geometry import Point

from sources import TARGET_IMAGE_SIZE as TARGET_SIZE
# MVT plumbing is shared with the Mapillary source: same slippy-map math, same decoder.
from sources.mapillary import tile_point_to_lonlat, _decode_tile
from sources import mapillary as _mapillary

NAME = 'panoramax'
COVERAGE_TILE_ZOOM = 15  # the only zoom whose tiles carry individual pictures
DEFAULT_API_URL = 'https://api.panoramax.xyz/api'
API_URL_ENV_VAR = 'PANORAMAX_API_URL'
# The API asks production tools to identify themselves explicitly.
USER_AGENT = 'sidewalk-auto-labeler/1.0 (+https://github.com/ProjectSidewalk/sidewalk-auto-labeler)'
ATTEMPTS = 3

# The catalog answers 404 for a picture id it doesn't know (deleted by its uploader, an
# instance that left the federation, or never valid) — verified live for both a
# well-formed unknown UUID and a malformed id. Retrying never helps, so it's a
# deterministic skip; anything else is treated as transient.
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


def api_url():
    return os.environ.get(API_URL_ENV_VAR, DEFAULT_API_URL).rstrip('/')


def prepare():
    """Nothing to check: the catalog is anonymous. A wrong PANORAMAX_API_URL surfaces on
    the first tile request, which main.py reports as a failed tile."""


def _headers():
    return {'User-Agent': USER_AGENT}


def fetch_panos_for_tile(tile_x, tile_y, area_shape):
    """
    Worker function for a single z15 coverage tile. Returns {picture_id: (lat, lon, ...)}
    for 360° pictures inside the area, or None if the tile failed after retries.
    """
    url = f'{api_url()}/map/{COVERAGE_TILE_ZOOM}/{tile_x}/{tile_y}.mvt'
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(url, headers=_headers(), timeout=60)
            if response.status_code in (204, 404) or not response.content:  # no coverage here
                return {}
            response.raise_for_status()
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
    (much more expensive) image download; the downloaded image's own dimensions are what
    the record stores, since they are what PS's normalized -> pixel transform must match.
    """
    item, gone = fetch_item(pano_id)
    if gone:
        return {'status': 'skipped', 'reason': 'Picture no longer exists on Panoramax'}
    if item is None:
        return {'status': 'failure', 'reason': 'STAC item unavailable (transient?)'}
    props = item.get('properties') or {}
    orientation = props.get('pers:interior_orientation') or {}
    # Belt and suspenders: the tile scan already filtered on type == equirectangular.
    if orientation.get('field_of_view') != 360:
        return {'status': 'skipped',
                'reason': f"Not a 360 picture (field_of_view={orientation.get('field_of_view')})"}
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

    downloaded = _download_image(url)
    if downloaded is None:
        return {'status': 'failure', 'reason': 'Failed to download equirectangular image'}
    image, (width, height) = downloaded
    if not _is_equirectangular(width, height):
        return {'status': 'skipped',
                'reason': f'Not a full 360x180 equirectangular ({width}x{height})'}

    return {'status': 'success',
            'pano': build_pano_record(pano_id, lat, lon, item, width, height),
            'image': image}


def _is_equirectangular(width, height):
    return bool(width and height) and abs(width - 2 * height) <= ASPECT_TOLERANCE * width


def image_url(item):
    """The original upload: the `hd` asset, else GeoVisio's own image link."""
    hd = (item.get('assets') or {}).get('hd') or {}
    return hd.get('href') or (item.get('properties') or {}).get('geovisio:image')


def fetch_item(picture_id):
    """(STAC item, gone) for one picture.

    `gone` is True only when the catalog positively says it has no such picture (404),
    which no amount of retrying will fix, so the retry loop short-circuits. A transient
    failure (network, 5xx) returns (None, False) after ATTEMPTS tries — the same
    "source decayed" vs "our fetch failed" distinction the Mapillary source draws.
    """
    url = f'{api_url()}/pictures/{picture_id}'
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(url, headers=_headers(), timeout=30)
            if response.status_code in GONE_STATUSES:
                return None, True
            response.raise_for_status()
            return response.json(), False
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None, False


def _download_image(url):
    """Downloads the original upload and normalizes it to the detector's 4096x2048.
    Returns (image, (original_width, original_height)), or None on failure (caller
    treats as retryable)."""
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(url, headers=_headers(), timeout=180)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            original_size = image.size
            if image.size != TARGET_SIZE:
                image = image.resize(TARGET_SIZE, Image.BILINEAR)
            return image, original_size
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


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
        'source_metadata': {
            **{k: v for k, v in sorted(props.items()) if k not in VOLATILE_PROPERTY_KEYS},
            'collection': item.get('collection'),
            'providers': item.get('providers'),
            'hd_url': url,
        },
    }


def _as_float(value):
    return None if value is None else float(value)


def build_pano_record(pano_id, lat, lon, item, width, height):
    """
    Builds the Stage-1 JSONL 'pano' block from a STAC item, in the same internal
    field-name contract as the GSV and Mapillary sources (send_to_ps.transform_pano maps
    it onto the Project Sidewalk reader; `source: "panoramax"` passes through to PS's
    pano_source enum). width/height are the downloaded image's own dimensions.
    """
    props = item.get('properties') or {}
    coordinates = (item.get('geometry') or {}).get('coordinates')
    if coordinates:  # the item's position when available; tile position otherwise
        lon, lat = coordinates[0], coordinates[1]
    captured = props['datetime']  # RFC 3339; the year-month prefix is all PS stores
    producer = props.get('geovisio:producer') or next(
        (p.get('name') for p in item.get('providers') or [] if p.get('name')), None)
    license_id = props.get('license') or 'CC-BY-SA-4.0'
    credit = f"{producer} / Panoramax" if producer else "Panoramax"
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
        "copyright": f"© {credit} ({license_id})",
        "source": "panoramax",
        "sequence_id": item.get('collection'),
        "license": license_id,
        **provenance_fields(item),
        "history": [],
        "links": []
    }
