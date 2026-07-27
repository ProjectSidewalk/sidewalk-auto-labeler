from PIL import Image
from streetlevel import streetview

from sources import TARGET_IMAGE_SIZE as TARGET_SIZE
# Zoom 3 is 4096x2048 for standard GSV panos - matches the detector input, so no detail is lost.
PREFERRED_ZOOM = 3


def fetch_panorama(metadata):
    """
    Downloads the equirectangular panorama for an already-fetched StreetViewPanorama
    metadata object. streetlevel stitches the tile grid and crops to the pano's true
    dimensions (both known from the metadata, so no probing is needed).

    Returns a 4096x2048 RGB PIL image, or None on any failure (caller treats as a skip).
    """
    try:
        # Older/third-party panos may not have zoom 3; use the highest available below it.
        zoom = min(PREFERRED_ZOOM, len(metadata.image_sizes) - 1)
        pano = streetview.get_panorama(metadata, zoom=zoom)
        if pano is None:
            return None

        # Defensive: clamp to a 2:1 aspect ratio, then normalize to the detector's size.
        width, height = pano.size
        max_width = height * 2
        if width > max_width:
            pano = pano.crop((0, 0, max_width, height))
        if pano.size != TARGET_SIZE:
            pano = pano.resize(TARGET_SIZE, Image.BILINEAR)
        return pano
    except Exception as e:
        print(f"Error fetching panorama {metadata.id}: {e}")
        return None
