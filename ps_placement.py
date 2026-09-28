"""Project Sidewalk's server-side label placement, ported exactly -- stdlib only.

The server computes an AI label's lat/lng ONCE, at insert, from the pano position it was
sent, the camera heading and the label's integer pixel (`PanoDataService.toLatLng` in
SidewalkWebpage `app/service/PanoDataService.scala`, estimator "approximation3" from
SidewalkWebpage#4819 / #5084):

1. bearing and pitch from the pixel (`calculatePovFromPanoXY`):
   heading = (camera_heading - 180 + x / w * 360) % 360   (Scala `%`: keeps the sign)
   pitch   = 90 - 180 * y / h
2. distance from the depression d = -pitch (`estimateDistanceFromPanoM`): the flat-ground
   cotangent h / tan(d) down to BLEND_DEG, then a linear tail with the cotangent's value
   and slope at BLEND_DEG, evaluated at max(d, 0) and capped at MAX_DISTANCE_M (which
   never binds: the tail's maximum, at the horizon, is ~23.85 m);
3. the spherical destination point (`CommonUtils.calculateDestination`, R = 6371 km).

This module exists so a city WITHOUT a server can still be scored under the server's own
label positions (issue #106): `scripts/eval_ps_clustering.py --offline` places every
synthesized label here. `pano_x`/`pano_y` must be the integers send_to_ps.py sent,
`round(x_normalized * width)` / `round(y_normalized * height)`.

Pinned against SidewalkWebpage's own cross-implementation parity fixture
(`test/fixtures/latLngEstimationParity.json`, copied into tests/fixtures/) in
tests/test_ps_placement.py.

Example:
    >>> lat, lng = label_latlng(47.6553, -122.3035, 6656, 4160, 13312, 6656, 90.0)
    >>> round(lat, 9), round(lng, 9)
    (47.6553, -122.303424536)
"""
import math

# LatLngEstimation in PanoDataService.scala.
CAMERA_HEIGHT_M = 2.341219672825709
BLEND_DEG = 11.25
MAX_DISTANCE_M = 50.0
# CommonUtils.EARTH_RADIUS_KM
EARTH_RADIUS_KM = 6371.0


def pov_from_pano_xy(x, y, width, height, camera_heading):
    """(heading_deg, pitch_deg) of pixel (x, y), exactly as calculatePovFromPanoXY.

    The heading is NOT wrapped into [0, 360): Scala's `%` keeps the dividend's sign, so a
    negative unwrapped heading stays negative. The destination point is the same either way.

    Example:
        >>> pov_from_pano_xy(1664, 4160, 13312, 6656, 10.0)
        (-125.0, -22.5)
    """
    heading = math.fmod(camera_heading - 180 + (float(x) / width) * 360, 360)
    pitch = 90.0 - 180.0 * y / height
    return heading, pitch


def estimate_distance_m(depression_deg, camera_height_m=CAMERA_HEIGHT_M):
    """Distance (m) to a label `depression_deg` below the horizon (estimateDistanceFromPanoM).

    Example:
        >>> round(estimate_distance_m(45.0), 9) == round(CAMERA_HEIGHT_M, 9)   # 45 deg: range = h
        True
        >>> round(estimate_distance_m(-5.0), 2)   # above the horizon: the tail's maximum
        23.85
    """
    blend = math.radians(BLEND_DEG)
    if depression_deg >= BLEND_DEG:
        return camera_height_m / math.tan(math.radians(depression_deg))
    tail = (camera_height_m / math.tan(blend)
            + camera_height_m * (math.pi / 180.0) / math.sin(blend) ** 2
            * (BLEND_DEG - max(depression_deg, 0.0)))
    return min(tail, MAX_DISTANCE_M)


def destination(lat, lng, distance_m, bearing_deg):
    """(lat, lng) `distance_m` from (lat, lng) along `bearing_deg` on the 6371 km sphere
    (CommonUtils.calculateDestination; the server passes estDistanceM / 1000)."""
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)
    brg = math.radians(bearing_deg)
    ang = (distance_m / 1000.0) / EARTH_RADIUS_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(ang)
                     + math.cos(lat1) * math.sin(ang) * math.cos(brg))
    lng2 = lng1 + math.atan2(math.sin(brg) * math.sin(ang) * math.cos(lat1),
                             math.cos(ang) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lng2)


def label_latlng(pano_lat, pano_lng, pano_x, pano_y, width, height, camera_heading):
    """The server's (lat, lng) for a label at integer pixel (pano_x, pano_y) (toLatLng)."""
    heading, pitch = pov_from_pano_xy(pano_x, pano_y, width, height, camera_heading)
    return destination(pano_lat, pano_lng, estimate_distance_m(-pitch), heading)


def label_offset(pano_x, pano_y, width, height, camera_heading):
    """(distance_m, bearing_deg) from the camera to the label, the two quantities
    label_latlng walks along. Used to invert a server position back to its camera."""
    heading, pitch = pov_from_pano_xy(pano_x, pano_y, width, height, camera_heading)
    return estimate_distance_m(-pitch), heading
