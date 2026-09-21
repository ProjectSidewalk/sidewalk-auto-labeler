"""GSV imagery source: coverage enumeration and per-pano fetching via streetlevel.

This is the original pipeline behavior, moved out of main.py unchanged: scan GSV
coverage tiles at z17, fetch pano metadata (with retries — Google's metadata endpoint
intermittently returns empty responses, sk-zk/streetlevel#40), download and stitch the
equirectangular via panorama.py, and build the Stage-1 JSONL pano block.
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


def fetch_metadata_with_retry(pano_id):
    """
    Fetches pano metadata, retrying with backoff: Google's metadata endpoint
    intermittently returns empty responses (see sk-zk/streetlevel#40).
    Returns None if all attempts fail.
    """
    for attempt in range(METADATA_ATTEMPTS):
        metadata = streetview.find_panorama_by_id(pano_id)
        if metadata is not None:
            return metadata
        if attempt < METADATA_ATTEMPTS - 1:
            time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
    return None


def _metadata_problem(metadata):
    """Deterministic-skip reason for metadata the pipeline can't use, else None."""
    if metadata.source in INDOOR_SOURCES:
        return 'Indoor panorama source'
    if metadata.date is None or not metadata.image_sizes or metadata.tile_size is None:
        return 'Pano metadata missing date, image sizes, or tile size'
    return None


def _download_and_build(pano_id, lat, lon, metadata):
    image = fetch_panorama(metadata)
    if image is None:
        return {'status': 'failure', 'reason': 'Failed to download equirectangular image'}
    return {'status': 'success', 'pano': build_pano_record(pano_id, lat, lon, metadata), 'image': image}


def fetch_pano(pano_id, lat, lon):
    """
    Fetches metadata and the equirectangular image for one pano (see the interface
    contract in sources/__init__.py). Metadata is validated before the (much more
    expensive) image download.
    """
    metadata = fetch_metadata_with_retry(pano_id)
    if metadata is None:
        return {'status': 'failure', 'reason': 'Metadata unavailable (transient?)'}
    problem = _metadata_problem(metadata)
    if problem:
        return {'status': 'skipped', 'reason': problem}
    return _download_and_build(pano_id, lat, lon, metadata)


def fetch_pano_by_id(pano_id, area_shape):
    """
    Gap-fill entry point (see sources/__init__.py): fetches a pano known only by id —
    a link target the coverage scan never enumerated — locating it from its own
    metadata. Outside-the-area is a deterministic skip (the run geometry is
    immutable), decided before the expensive image download.
    """
    metadata = fetch_metadata_with_retry(pano_id)
    if metadata is None:
        return {'status': 'failure', 'reason': 'Metadata unavailable (transient?)'}
    problem = _metadata_problem(metadata)
    if problem:
        return {'status': 'skipped', 'reason': problem}
    if metadata.lat is None or metadata.lon is None:
        return {'status': 'skipped', 'reason': 'Pano metadata carries no position'}
    if not Point(metadata.lon, metadata.lat).within(area_shape):
        return {'status': 'skipped', 'reason': 'Outside the run area'}
    return _download_and_build(pano_id, metadata.lat, metadata.lon, metadata)


def build_pano_record(pano_id, lat, lon, metadata):
    """
    Builds the Stage-1 JSONL 'pano' block from streetlevel metadata. Field names are
    the pipeline's internal contract (send_to_ps.transform_pano maps them onto the
    Project Sidewalk reader). Heading/pitch/roll are radians in the metadata, degrees
    on the wire.
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
        ]
    }
