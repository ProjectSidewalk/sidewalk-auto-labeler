"""GSV imagery source: coverage enumeration and per-pano fetching via streetlevel.

This is the original pipeline behavior, moved out of main.py unchanged: scan GSV
coverage tiles at z17, fetch pano metadata (with retries — Google's metadata endpoint
intermittently returns empty responses, sk-zk/streetlevel#40), download and stitch the
equirectangular via panorama.py, and build the Stage-1 JSONL pano block.

The metadata request also asks for GSV's depth payload (issue #40) -- a flag on the same
URL, not a second fetch -- because its ground plane is the camera height, which the
raycast needs per pano. Only the handful of derived fields is stored (see
depth.camera_height_fields); scripts/harvest_depth.py archives the payload itself.

Imagery provenance (issue #23) comes from the same metadata object: see
provenance_fields, the GSV counterpart of the Mapillary/Panoramax ones.
"""
import math
import random
import time

from shapely.geometry import Point

# streetlevel hard-imports pyexiv2 (for EXIF writing), and every published pyexiv2
# wheel bundles a libexiv2 built against glibc >= 2.29 — on older hosts (e.g. Hyak's
# Rocky 8, glibc 2.28) the dlopen fails and takes the whole GSV source down with it.
# This pipeline never writes EXIF, so when pyexiv2 can't load, register a stub that
# satisfies streetlevel's import (including the eagerly-evaluated pyexiv2.ImageData
# annotation in streetlevel.exif) and fails loudly if EXIF writing is ever invoked.
try:
    import pyexiv2  # noqa: F401
except Exception:
    import sys
    import types

    class _PyExiv2Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "pyexiv2 could not be loaded on this host (its bundled libexiv2 "
                "needs a newer glibc); EXIF writing is unavailable."
            )

    # Expose ONLY what streetlevel touches at import time. No catch-all
    # __getattr__: answering hasattr(module, '__file__') with a non-string
    # breaks inspect.getmodule() for every stack walker in the process
    # (torch's op registration among them).
    _stub = types.ModuleType('pyexiv2')
    _stub.ImageData = _PyExiv2Unavailable
    sys.modules['pyexiv2'] = _stub

from streetlevel import streetview
from streetlevel.streetview import api
from streetlevel.streetview.parse import parse_panorama_id_response

import depth as depthlib
from panorama import fetch_panorama

NAME = 'gsv'
COVERAGE_TILE_ZOOM = 17
METADATA_ATTEMPTS = 3

# Indoor panorama sources; there are no curb ramps indoors.
INDOOR_SOURCES = ('innerspace', 'cultural_institute', 'photos:legacy_innerspace')


def prepare():
    """No setup needed: streetlevel's GSV endpoints are anonymous."""


def fetch_panos_for_tile(tile_x, tile_y, area_shape):
    """
    Worker function for a single coverage tile. Fetches panos and filters them.
    Returns a dictionary of {pano_id: (lat, lon)} for valid panos, or None if the
    tile failed after retries (throttling loses ~10% of tiles on large scans if
    failures are swallowed silently — the caller counts and reports these).
    """
    for attempt in range(3):
        try:
            panos_in_tile = streetview.get_coverage_tile(tile_x, tile_y)
            return {p.id: (p.lat, p.lon) for p in panos_in_tile if Point(p.lon, p.lat).within(area_shape)}
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def find_panorama_with_depth(pano_id):
    """`streetview.find_panorama_by_id(pano_id)`, plus the camera-height fields of the
    depth payload that rides on the same response.

    Returns (metadata or None, fields). The payload is pulled out of the raw response and
    then removed from it before streetlevel parses the rest, for two reasons: streetlevel
    would otherwise rasterize it (a pure-Python loop over 131k pixels per pano, for a
    raster we never use), and its parser reads the header's offset byte as a uint16 and
    throws on ~0.3-0.5% of panoramas (see depth.parse) -- which would turn a perfectly good
    pano into a metadata failure. depth.parse reads it correctly.

    Depth must never fail a pano: the path into the response is undocumented and
    positional, so if Google reshapes it, every pano would otherwise become a retryable
    failure that no rerun can clear. Anything unexpected there is recorded as
    camera_height_status 'unparsed' (main.py alarms when that is widespread) and the
    metadata is parsed as usual.
    """
    response = api.find_panorama_by_id(pano_id, download_depth=True)
    blob = depthlib.blob_from_response(response)
    if blob is None:
        return parse_panorama_id_response(response), depthlib.camera_height_fields(None)
    try:
        response[1][0][5][0][5][1][2] = None  # the path blob_from_response just read
        fields = depthlib.camera_height_fields(depthlib.parse(blob))
    except Exception:
        fields = {**depthlib.camera_height_fields(None),
                  'camera_height_status': depthlib.UNPARSED}
    return parse_panorama_id_response(response), fields


def fetch_metadata_with_retry(pano_id):
    """
    Fetches pano metadata, retrying with backoff: Google's metadata endpoint
    intermittently returns empty responses (see sk-zk/streetlevel#40).
    Returns (metadata, camera-height fields), or (None, None) if all attempts fail.
    """
    for attempt in range(METADATA_ATTEMPTS):
        metadata, depth_fields = find_panorama_with_depth(pano_id)
        if metadata is not None:
            return metadata, depth_fields
        if attempt < METADATA_ATTEMPTS - 1:
            time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None, None


def _metadata_problem(metadata):
    """Deterministic-skip reason for metadata the pipeline can't use, else None."""
    if metadata.source in INDOOR_SOURCES:
        return 'Indoor panorama source'
    if metadata.date is None or not metadata.image_sizes or metadata.tile_size is None:
        return 'Pano metadata missing date, image sizes, or tile size'
    return None


def _download_and_build(pano_id, lat, lon, metadata, depth_fields):
    image = fetch_panorama(metadata)
    if image is None:
        return {'status': 'failure', 'reason': 'Failed to download equirectangular image'}
    return {'status': 'success',
            'pano': build_pano_record(pano_id, lat, lon, metadata, depth_fields),
            'image': image}


def fetch_pano(pano_id, lat, lon):
    """
    Fetches metadata and the equirectangular image for one pano (see the interface
    contract in sources/__init__.py). Metadata is validated before the (much more
    expensive) image download.
    """
    metadata, depth_fields = fetch_metadata_with_retry(pano_id)
    if metadata is None:
        return {'status': 'failure', 'reason': 'Metadata unavailable (transient?)'}
    problem = _metadata_problem(metadata)
    if problem:
        return {'status': 'skipped', 'reason': problem}
    return _download_and_build(pano_id, lat, lon, metadata, depth_fields)


def fetch_pano_by_id(pano_id, area_shape):
    """
    Gap-fill entry point (see sources/__init__.py): fetches a pano known only by id —
    a link target the coverage scan never enumerated — locating it from its own
    metadata. Outside-the-area is a deterministic skip (the run geometry is
    immutable), decided before the expensive image download.
    """
    metadata, depth_fields = fetch_metadata_with_retry(pano_id)
    if metadata is None:
        return {'status': 'failure', 'reason': 'Metadata unavailable (transient?)'}
    problem = _metadata_problem(metadata)
    if problem:
        return {'status': 'skipped', 'reason': problem}
    if metadata.lat is None or metadata.lon is None:
        return {'status': 'skipped', 'reason': 'Pano metadata carries no position'}
    if not Point(metadata.lon, metadata.lat).within(area_shape):
        return {'status': 'skipped', 'reason': 'Outside the run area'}
    return _download_and_build(pano_id, metadata.lat, metadata.lon, metadata, depth_fields)


def build_pano_record(pano_id, lat, lon, metadata, depth_fields=None):
    """
    Builds the Stage-1 JSONL 'pano' block from streetlevel metadata. Field names are
    the pipeline's internal contract (send_to_ps.transform_pano maps them onto the
    Project Sidewalk reader). Heading/pitch/roll are radians in the metadata, degrees
    on the wire.

    `depth_fields` (from depth.camera_height_fields) adds camera_height_m and its
    provenance; None records the pano as having no depth, so the keys are always present.
    provenance_fields adds the imagery provenance shared in shape with the other sources.
    """
    return {
        "panorama_id": pano_id,
        "capture_date": f"{metadata.date.year}-{metadata.date.month:02d}",
        "width": metadata.image_sizes[-1].x,
        "height": metadata.image_sizes[-1].y,
        "tile_width": metadata.tile_size.x,
        "tile_height": metadata.tile_size.y,
        "lat": float(lat),
        "lng": float(lon),
        "camera_heading": math.degrees(metadata.heading),
        "camera_pitch": math.degrees(metadata.pitch),
        "camera_roll": math.degrees(metadata.roll),
        "copyright": metadata.copyright_message,
        "source": metadata.source,
        "history": [
            {
                "pano_id": old_pano.id,
                "date": f"{old_pano.date.year}-{old_pano.date.month:02d}"
            } for old_pano in (metadata.historical or []) if old_pano.date is not None
        ],
        "links": [
            {
                "target_gsv_panorama_id": linked_pano.pano.id,
                "yaw_deg": math.degrees(linked_pano.direction),
                "description": linked_pano.pano.address[0].value if linked_pano.pano.address else ""
            } for linked_pano in (metadata.links or [])
        ],
        **provenance_fields(metadata),
        **(depth_fields or depthlib.camera_height_fields(None)),
    }


# --- Extended provenance (issue #23) -------------------------------------------------
# streetlevel's StreetViewPanorama is a dataclass of nested dataclasses, enums, numpy
# arrays and further StreetViewPanorama objects, so it is not JSON-serializable and must
# never be dumped wholesale (vars()/asdict()): the depth raster alone is 131k floats, and
# neighbors/links/historical recurse. Instead each field kept is named here with its own
# converter to JSON-native values. A field streetlevel adds later is therefore left out
# until someone names it, so the record shape cannot drift with a library upgrade.
# Angles are radians in streetlevel and degrees on the wire, as for the camera pose.

def _localized(value):
    """streetlevel LocalizedString -> {'value', 'language'}."""
    if value is None:
        return None
    return {'value': value.value, 'language': value.language}


def _deg(radians):
    return None if radians is None else math.degrees(radians)


def _upload_date(value):
    """UploadDate (year/month/day/hour; third-party panos only) -> a dict of ints, kept
    as fields rather than a string because 'YYYY-MM-DD HH' has no standard spelling."""
    if value is None:
        return None
    return {part: getattr(value, part, None) for part in ('year', 'month', 'day', 'hour')}


def _street_names(labels):
    return [{'name': _localized(label.name),
             'angles_deg': [math.degrees(a) for a in label.angles or []]}
            for label in labels]


def _building_level(level):
    return {'level': level.level, 'name': _localized(level.name),
            'short_name': _localized(level.short_name)}


def _places(places):
    return [{
        'feature_id': place.feature_id,
        'cid': place.cid,
        'name': _localized(place.name),
        'type': _localized(place.type),
        # BusinessStatus enum -> its name ('Operational', 'PermanentlyClosed', ...)
        'status': place.status.name if place.status is not None else None,
        'marker_yaw_deg': _deg(place.marker_yaw),
        'marker_pitch_deg': _deg(place.marker_pitch),
        'marker_distance': place.marker_distance,
        'marker_icon_url': place.marker_icon_url,
    } for place in places]


def _artworks(artworks):
    return [{
        'id': art.id,
        'title': _localized(art.title),
        'creator': _localized(art.creator),
        'description': _localized(art.description),
        'thumbnail': art.thumbnail,
        'url': art.url,
        'attributes': {k: _localized(v) for k, v in (art.attributes or {}).items()},
        'marker_yaw_deg': _deg(art.marker_yaw),
        'marker_pitch_deg': _deg(art.marker_pitch),
        'marker_icon_url': art.marker_icon_url,
        'link': (None if art.link is None else
                 {'pano_id': art.link.panoid, 'link_text': _localized(art.link.link_text)}),
    } for art in artworks]


def _pano_ids(panos):
    """Nested StreetViewPanorama objects -> ids only; each is a pano of its own."""
    return [pano.id for pano in panos]


def _plain(value):
    return value


# name -> converter for the non-None value. Order is the key order on the wire.
SOURCE_METADATA_FIELDS = (
    ('uploader', _plain),
    ('uploader_icon_url', _plain),
    ('upload_date', _upload_date),
    ('elevation', float),
    ('country_code', _plain),
    ('street_names', _street_names),
    ('address', lambda parts: [_localized(p) for p in parts]),
    ('building_level', _building_level),
    ('building_levels', _pano_ids),
    ('places', _places),
    ('artworks', _artworks),
    ('neighbors', _pano_ids),
)


def _project(metadata, name, convert):
    """One source_metadata field, or None when streetlevel did not set it (an optional
    attribute, or one an older streetlevel does not have). Provenance must never fail a
    pano -- it is record-keeping, not an input to detection -- so a value whose shape a
    future streetlevel changed under a converter is also recorded as None."""
    value = getattr(metadata, name, None)
    if value is None:
        return None
    try:
        return convert(value)
    except Exception:
        return None


def provenance_fields(metadata):
    """Extended-provenance keys added to a GSV pano block, in the same contract as
    sources/mapillary.py and sources/panoramax.py (issue #23): curated top-level fields
    plus a `source_metadata` projection of the streetlevel object.

    GSV exposes no camera make/model, so those are explicit nulls (the shape is the same
    across sources; the value says 'not known'). Google always serves an equirectangular.
    The GSV analogue of make/model is `source_detail` + `uploader`: Google's own car
    imagery (`launch`, `scout`) versus a user photosphere (`photos:...`) is what explains
    why one pano is sharp and another is not. `source_detail` duplicates the block's
    `source` here on purpose -- send_to_ps.transform_pano coerces `source` onto PS's
    pano_source enum ('gsv'), and this is the copy that survives that.

    `source_metadata` is an explicit per-field projection (SOURCE_METADATA_FIELDS), never
    a dump: see the comment above the helpers. Every named key is always present, None
    when streetlevel did not set it, so consumers never have to test for the key.
    """
    return {
        'camera_make': None,
        'camera_model': None,
        'camera_type': 'equirectangular',
        'source_detail': getattr(metadata, 'source', None),
        'uploader': getattr(metadata, 'uploader', None),
        'source_metadata': {name: _project(metadata, name, convert)
                            for name, convert in SOURCE_METADATA_FIELDS},
    }
