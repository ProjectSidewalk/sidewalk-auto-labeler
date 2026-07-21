"""Mapillary imagery source: coverage via z14 vector tiles, imagery via the Graph API v4.

Coverage: Mapillary's vector tiles (`mly1_public`) carry individual images only at
z14, as points in the `image` layer with `id` and `is_pano` properties — so filtering
to 360° panoramas happens during enumeration, with no per-image API calls.

Fetching: one Graph API call per image returns the signed full-resolution thumbnail
URL (`thumb_original_url` — the original uploaded dimensions, despite the name) plus
dimensions, SfM-computed position and compass angle. The signed URL expires, so the
image is downloaded in the same worker pass.

Orientation: the center column of a Mapillary equirectangular is the camera's compass
bearing — the same convention as GSV panos and exactly what Project Sidewalk's
panoX -> heading math assumes — so the image is never rotated; `computed_compass_angle`
is recorded as `camera_heading`. Pitch/roll only exist inside `computed_rotation`
(an axis-angle vector) and are left null.

Requires a Mapillary client token (mapillary.com/dashboard/developers) in the
MAPILLARY_ACCESS_TOKEN environment variable. Rate limits (60k entity requests/min,
50k tiles/day per app) are far above what a city-scale run generates.
"""
import os
import math
import random
import time
from datetime import datetime, timezone
from io import BytesIO

import requests
from PIL import Image
from shapely.geometry import Point

from sources import TARGET_IMAGE_SIZE as TARGET_SIZE

NAME = 'mapillary'
COVERAGE_TILE_ZOOM = 14  # the only zoom whose tiles carry individual images
TILE_URL = 'https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}'
GRAPH_URL = 'https://graph.mapillary.com'
TOKEN_ENV_VAR = 'MAPILLARY_ACCESS_TOKEN'
ATTEMPTS = 3

IMAGE_FIELDS = ','.join([
    'camera_type', 'captured_at', 'computed_geometry', 'geometry',
    'computed_compass_angle', 'compass_angle', 'thumb_original_url',
    'width', 'height', 'creator', 'sequence', 'quality_score',
])

# A full 360x180 equirectangular is exactly 2:1. Narrower "spherical" uploads exist
# (cropped vertical FOV); resizing one to 4096x2048 would stretch it and shift every
# detection's pano_y, so they're skipped rather than padded. Revisit if a scan shows
# they're a meaningful share of coverage.
ASPECT_TOLERANCE = 0.02

# Mapillary contributors re-drive the same streets, so raw 360 coverage runs ~1 pano
# per 1.5 m of street (downtown Richmond: 35k panos in 4.5 km²). Thinning keeps one
# pano per grid cell. The default is deliberately denser than GSV's ~10 m spacing:
# Mapillary image quality is generally lower, detection works better the closer a
# ramp is, and Project Sidewalk's clustering consolidates proximal duplicate labels —
# so extra redundancy mostly buys detection robustness at GPU-time cost. Override
# per run with --thin-spacing (0 disables thinning).
THIN_CELL_METERS = 5


def prepare():
    if not os.environ.get(TOKEN_ENV_VAR):
        raise SystemExit(
            f"❌ {TOKEN_ENV_VAR} is not set. The Mapillary source needs a client token\n"
            f"   (create one at https://www.mapillary.com/dashboard/developers)."
        )


def _token():
    return os.environ[TOKEN_ENV_VAR]


def fetch_panos_for_tile(tile_x, tile_y, area_shape):
    """
    Worker function for a single z14 coverage tile. Returns {image_id: (lat, lon)}
    for 360° panos inside the area, or None if the tile failed after retries.
    """
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(
                TILE_URL.format(z=COVERAGE_TILE_ZOOM, x=tile_x, y=tile_y),
                params={'access_token': _token()}, timeout=30)
            if response.status_code == 404:  # no Mapillary coverage in this tile
                return {}
            response.raise_for_status()
            return panos_from_tile(response.content, tile_x, tile_y, area_shape)
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def _decode_tile(tile_bytes):
    # Imported lazily so GSV-only runs don't need the MVT decoder installed.
    import mapbox_vector_tile
    # y_coord_down keeps tile-local y growing southward (the raw MVT convention),
    # matching the slippy-map math in tile_point_to_lonlat. The options API changed
    # in mapbox-vector-tile 2.0; support both.
    try:
        return mapbox_vector_tile.decode(tile_bytes, default_options={'y_coord_down': True})
    except TypeError:
        return mapbox_vector_tile.decode(tile_bytes, y_coord_down=True)


def panos_from_tile(tile_bytes, tile_x, tile_y, area_shape):
    """Decodes one coverage tile into {image_id: (lat, lon, captured_at_ms,
    quality_score)} for 360° panos in the area. main.py only relies on the first two
    elements; captured_at/quality_score feed thin_panos."""
    image_layer = _decode_tile(tile_bytes).get('image')
    if not image_layer:
        return {}
    extent = image_layer.get('extent', 4096)
    panos = {}
    for feature in image_layer['features']:
        properties = feature.get('properties', {})
        if not properties.get('is_pano'):
            continue
        px, py = feature['geometry']['coordinates']
        lon, lat = tile_point_to_lonlat(px, py, tile_x, tile_y, COVERAGE_TILE_ZOOM, extent)
        if Point(lon, lat).within(area_shape):
            panos[str(properties['id'])] = (
                lat, lon, properties.get('captured_at', 0), properties.get('quality_score', 0.0))
    return panos


def thin_panos(panos, cell_meters=None):
    """Keeps one pano per grid cell (default THIN_CELL_METERS): newest capture wins,
    higher quality_score breaks ties. Called by main.py after the coverage scan (the
    hook is optional per source; GSV coverage is already ~10 m spaced and has none)."""
    cell_meters = cell_meters or THIN_CELL_METERS
    cells = {}
    for pano_id, (lat, lon, captured_at, quality) in panos.items():
        cell_lat = cell_meters / 111320.0
        cell_lon = cell_meters / (111320.0 * max(0.01, math.cos(math.radians(lat))))
        cell = (int(lat // cell_lat), int(lon // cell_lon))
        best = cells.get(cell)
        if best is None or (captured_at, quality) > best[1]:
            cells[cell] = (pano_id, (captured_at, quality))
    keep = {pano_id for pano_id, _ in cells.values()}
    return {pano_id: value for pano_id, value in panos.items() if pano_id in keep}


def tile_point_to_lonlat(px, py, tile_x, tile_y, zoom, extent):
    """Converts tile-local MVT point coordinates (y down) to lon/lat — the inverse
    of main.latlon_to_tile with sub-tile precision."""
    n = 2.0 ** zoom
    x = (tile_x + px / extent) / n
    y = (tile_y + py / extent) / n
    lon = x * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y))))
    return lon, lat


def fetch_pano(pano_id, lat, lon):
    """
    Fetches Graph API metadata and the equirectangular image for one Mapillary image
    (see the interface contract in sources/__init__.py). Metadata is validated before
    the (much more expensive) image download.
    """
    meta = _fetch_image_metadata(pano_id)
    if meta is None:
        return {'status': 'failure', 'reason': 'Graph API metadata unavailable (transient?)'}
    # Belt and suspenders: the tile scan already filtered on is_pano.
    if meta.get('camera_type') not in ('spherical', 'equirectangular'):
        return {'status': 'skipped', 'reason': f"Not a 360 pano (camera_type={meta.get('camera_type')})"}
    if not meta.get('captured_at'):
        return {'status': 'skipped', 'reason': 'No capture timestamp'}
    width, height = meta.get('width'), meta.get('height')
    if not meta.get('thumb_original_url') or not width or not height:
        return {'status': 'skipped', 'reason': 'Metadata missing thumbnail URL or dimensions'}
    if abs(width - 2 * height) > ASPECT_TOLERANCE * width:
        return {'status': 'skipped', 'reason': f'Not a full 360x180 equirectangular ({width}x{height})'}
    # PS's pano_x -> heading math requires camera_heading; a pano without any compass
    # angle can't be submitted.
    if _compass_angle(meta) is None:
        return {'status': 'skipped', 'reason': 'No compass angle'}

    image = _download_image(meta['thumb_original_url'])
    if image is None:
        return {'status': 'failure', 'reason': 'Failed to download equirectangular image'}

    return {'status': 'success', 'pano': build_pano_record(pano_id, lat, lon, meta), 'image': image}


def _compass_angle(meta):
    """SfM-corrected compass when available, camera EXIF compass otherwise."""
    angle = meta.get('computed_compass_angle')
    return angle if angle is not None else meta.get('compass_angle')


def _fetch_image_metadata(image_id):
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(
                f'{GRAPH_URL}/{image_id}',
                params={'access_token': _token(), 'fields': IMAGE_FIELDS}, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def fetch_image(pano_id):
    """Just the normalized 4096x2048 equirectangular for one image ID, or None.
    Used by the spot-check gallery to re-download display imagery; the full
    pipeline path is fetch_pano."""
    meta = _fetch_image_metadata(pano_id)
    url = (meta or {}).get('thumb_original_url')
    return _download_image(url) if url else None


def _download_image(url):
    """Downloads the signed thumbnail and normalizes it to the detector's 4096x2048.
    Returns None on failure (caller treats as retryable)."""
    for attempt in range(ATTEMPTS):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            if image.size != TARGET_SIZE:
                image = image.resize(TARGET_SIZE, Image.BILINEAR)
            return image
        except Exception:
            if attempt < ATTEMPTS - 1:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def build_pano_record(pano_id, lat, lon, meta):
    """
    Builds the Stage-1 JSONL 'pano' block from Graph API metadata, in the same
    internal field-name contract as the GSV source (send_to_ps.transform_pano maps
    it onto the Project Sidewalk reader; `source: "mapillary"` passes through to
    PS's pano_source enum). sequence_id and quality_score are extra provenance the
    PS reader ignores.
    """
    coordinates = (meta.get('computed_geometry') or meta.get('geometry') or {}).get('coordinates')
    if coordinates:  # SfM-corrected position when available; tile position otherwise
        lon, lat = coordinates
    captured = datetime.fromtimestamp(meta['captured_at'] / 1000.0, tz=timezone.utc)
    creator = (meta.get('creator') or {}).get('username')
    return {
        "panorama_id": pano_id,
        "capture_date": f"{captured.year}-{captured.month:02d}",
        "width": int(meta['width']),
        "height": int(meta['height']),
        "lat": float(lat),
        "lng": float(lon),
        "camera_heading": float(_compass_angle(meta)),
        "camera_pitch": None,
        "camera_roll": None,
        "copyright": f"© {creator} / Mapillary (CC BY-SA 4.0)" if creator else "© Mapillary (CC BY-SA 4.0)",
        "source": "mapillary",
        "sequence_id": meta.get('sequence'),
        "quality_score": meta.get('quality_score'),
        "history": [],
        "links": []
    }
