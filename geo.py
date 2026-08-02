"""Shared geodesy for the labeler — torch-free, numpy-free, stdlib only.

This is the single home for the constants and coordinate math that were
previously duplicated across scripts/ (export_benchmark, thinning_experiment)
and sources/. Everything here is importable everywhere (main.py, send_to_ps.py,
scripts/, tests) without pulling in the model stack, same as detectors/__init__.py.

Two coordinate models coexist deliberately:

- ``haversine_m`` is the exact great-circle distance, used where a distance is
  compared against a threshold in isolation (benchmark spacing).
- ``LocalFrame`` is a flat equirectangular tangent plane (fixed cos(lat0)),
  used where many points must live in one Euclidean frame (site association,
  covariance math). At city scale the two disagree by well under 0.5% — for a
  10 m separation 15 km from the frame origin the error is centimeters.
"""
import math
from dataclasses import dataclass

METERS_PER_DEG_LAT = 111320.0
EARTH_RADIUS_M = 6371000.0

# Ground raycast envelope. The 0.02 rad (~1.15 deg) guard rejects at/above-horizon
# peaks and the near-horizon band where tan() explodes; both bounds are inherited
# from scripts/thinning_experiment.py, but the raycast here DROPS beyond max range
# where that script clamped (a clamp fabricates ranges — ~20% of paterson's stored
# detections sat on the old 30 m clamp).
MIN_DEPRESSION_RAD = 0.02
DEFAULT_CAMERA_HEIGHT_M = 2.6  # typical roof-mounted 360 rig
DEFAULT_MAX_RANGE_M = 25.0

# RampNet's heatmap is 1024x512 over the full equirect, so detections are quantized
# to that grid — and both axes step by the same angle: 2*pi/1024 == pi/512 rad/px.
RAD_PER_HEATMAP_PX = math.pi / 512.0


def norm_deg(a):
    """Fold any angle in degrees into [-180, 180). GSV stores roll (and sometimes
    pitch) as [0, 360) — e.g. a -0.6 degree roll arrives as 359.4 — so every angle
    read from a pano block must pass through here before trigonometry."""
    return (a + 180.0) % 360.0 - 180.0


def haversine_m(lat1, lng1, lat2, lng2):
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class LocalFrame:
    """Flat local tangent frame: (lat, lng) <-> (east_m, north_m) about a fixed origin.

    The longitude scale is frozen at cos(lat0), so the frame is a plain linear map
    — safe to invert, average in, and do covariance algebra in. Pick the origin
    near the data centroid (a run's mean pano position).
    """

    def __init__(self, lat0, lng0):
        self.lat0, self.lng0 = lat0, lng0
        self._m_per_deg_lng = METERS_PER_DEG_LAT * max(0.01, math.cos(math.radians(lat0)))

    def to_enu(self, lat, lng):
        return (norm_deg(lng - self.lng0) * self._m_per_deg_lng,
                (lat - self.lat0) * METERS_PER_DEG_LAT)

    def to_latlng(self, e, n):
        return (self.lat0 + n / METERS_PER_DEG_LAT,
                self.lng0 + e / self._m_per_deg_lng)


class GridIndex:
    """Uniform grid over ENU coordinates holding arbitrary payloads.

    ``near(e, n)`` yields every item in the 3x3 cell neighborhood — a superset of
    all items within ``cell_m`` meters, so with cell size equal to the query cap
    the caller only re-checks true distance, never misses a candidate. Items are
    never removed; callers that move a payload add it again under the new cell
    and treat stale entries as harmless (they fail the distance re-check).
    """

    def __init__(self, cell_m):
        self.cell = max(cell_m, 1e-9)
        self.cells = {}

    def _key(self, e, n):
        return (math.floor(e / self.cell), math.floor(n / self.cell))

    def add(self, e, n, item):
        self.cells.setdefault(self._key(e, n), []).append(item)

    def near(self, e, n):
        kx, ky = self._key(e, n)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from self.cells.get((kx + dx, ky + dy), ())


class LatLngSpacingIndex:
    """Grid of accepted points for fast 'is anything within min_spacing?' checks.

    Cell size == min_spacing, so any point within range lies in one of the nine
    neighbouring cells — the acceptance test touches a handful of points, not the
    whole accepted set, so selection stays cheap on city-sized candidate pools.
    (Moved verbatim from scripts/export_benchmark.py, where it declusters the
    benchmark sample.)
    """

    def __init__(self, min_spacing):
        self.s = max(min_spacing, 1e-9)
        self.cells = {}

    def _key(self, lat, lng):
        clat = self.s / METERS_PER_DEG_LAT
        clng = self.s / (METERS_PER_DEG_LAT * max(0.01, math.cos(math.radians(lat))))
        return (math.floor(lat / clat), math.floor(lng / clng))

    def far_enough(self, lat, lng):
        kx, ky = self._key(lat, lng)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for plat, plng in self.cells.get((kx + dx, ky + dy), ()):
                    if haversine_m(lat, lng, plat, plng) < self.s:
                        return False
        return True

    def add(self, lat, lng):
        self.cells.setdefault(self._key(lat, lng), []).append((lat, lng))


# --- Symmetric 2x2 matrices as (a, b, c) == [[a, b], [b, c]] --------------------------
# Covariances here always include an isotropic GPS floor, so determinants stay
# strictly positive and the inverse is safe.

def sym2_add(m, n):
    return (m[0] + n[0], m[1] + n[1], m[2] + n[2])


def sym2_inv(m):
    a, b, c = m
    det = a * c - b * b
    return (c / det, -b / det, a / det)


def sym2_quadform(m, dx, dy):
    """dx, dy row-vector times m times its transpose: a*dx^2 + 2b*dx*dy + c*dy^2."""
    a, b, c = m
    return a * dx * dx + 2.0 * b * dx * dy + c * dy * dy


# --- Camera pose and the ground-plane raycast -----------------------------------------

@dataclass(frozen=True)
class Pose:
    lat: float
    lng: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    has_pitch_roll: bool
    source: str


def pano_pose(pano):
    """Extract a Pose from a results.jsonl pano block.

    GSV blocks carry heading/pitch/roll in degrees (sometimes as [0, 360) — always
    normalized here). Mapillary blocks store pitch/roll as null (the OpenSfM
    rotation is only inside source_metadata, deliberately not parsed here); those
    panos get pitch=roll=0 with has_pitch_roll=False and the wider Mapillary error
    model absorbs the unknown tilt.
    """
    src = pano.get('source') or ''
    heading = norm_deg(float(pano['camera_heading']))
    pitch, roll = pano.get('camera_pitch'), pano.get('camera_roll')
    if pitch is None or roll is None:
        return Pose(pano['lat'], pano['lng'], heading, 0.0, 0.0, False, src)
    return Pose(pano['lat'], pano['lng'], heading,
                norm_deg(float(pitch)), norm_deg(float(roll)), True, src)


@dataclass(frozen=True)
class ErrorModel:
    """1-sigma inputs for the ground-point covariance.

    sigma_peak_px is heatmap-peak localization jitter in heatmap pixels; the
    angular quantum is RAD_PER_HEATMAP_PX on both axes. GPS sigma enters the ENU
    covariance isotropically (it moves the ray origin, not the ray)."""
    sigma_peak_px: float = 1.0
    sigma_pitch_rad: float = math.radians(0.3)
    sigma_heading_rad: float = math.radians(0.5)
    sigma_height_m: float = 0.15
    sigma_gps_m: float = 1.0


GSV_ERRORS = ErrorModel()
# Mapillary: no pitch/roll (rig tilt lands in sigma_pitch), consumer rigs, SfM
# positions with meters of scatter between sequences.
MAPILLARY_ERRORS = ErrorModel(sigma_pitch_rad=math.radians(1.5),
                              sigma_heading_rad=math.radians(1.0),
                              sigma_height_m=0.30,
                              sigma_gps_m=3.0)


def error_model_for(source):
    return MAPILLARY_ERRORS if source == 'mapillary' else GSV_ERRORS


@dataclass(frozen=True)
class GroundEstimate:
    lat: float
    lng: float
    range_m: float
    bearing_deg: float       # world bearing of the ray, [0, 360)
    sigma_along_m: float     # 1-sigma along the ray (range uncertainty)
    sigma_cross_m: float     # 1-sigma across the ray (bearing uncertainty)

    def cov_en(self, sigma_gps_m):
        """ENU covariance (sym2 tuple) of this ground point: the along/cross
        anisotropy rotated to the ray bearing, plus isotropic GPS variance."""
        b = math.radians(self.bearing_deg)
        sb, cb = math.sin(b), math.cos(b)
        sa2, sc2 = self.sigma_along_m ** 2, self.sigma_cross_m ** 2
        g2 = sigma_gps_m ** 2
        return (sa2 * sb * sb + sc2 * cb * cb + g2,   # var_ee
                (sa2 - sc2) * sb * cb,                # cov_en
                sa2 * cb * cb + sc2 * sb * sb + g2)   # var_nn


def _world_ray(pose, phi, theta):
    """(elevation, bearing) in world frame of a pano-frame direction, applying the
    pano's yaw/pitch/roll exactly.

    Rotation: intrinsic yaw (about up) -> pitch (about the right axis; positive
    raises the view axis) -> roll (about the forward axis; positive lifts the
    right side of the image). First-order effect on elevation:
    elev ~= theta + pitch*cos(phi) + roll*sin(phi). Vector components are
    (north, east, up).

    MEASURED (fuse_sites.py --pose-ablation, 2026-08-02, paterson + bend,
    ~123k within-site member pairs): applying GSV metadata pitch/roll under ANY
    sign convention LOOSENS multi-view agreement — mean pairwise member distance
    2.61 m -> 3.9-4.7 m (paterson), 2.06 m -> 2.8-3.9 m (bend) — i.e. the
    equirectangulars streetlevel serves are already gravity-rectified and the
    metadata angles describe the capture rig, not the stitched pano frame. So
    production fusion runs with apply_pose=False and this rotation exists for
    experiments (and any future source whose imagery is NOT rectified).
    """
    psi = math.radians(pose.heading_deg)
    alpha = math.radians(pose.pitch_deg)
    rho = math.radians(pose.roll_deg)

    # Pano-frame basis in world coordinates, after yaw:
    f = (math.cos(psi), math.sin(psi), 0.0)      # forward (center column)
    r = (-math.sin(psi), math.cos(psi), 0.0)     # right
    u = (0.0, 0.0, 1.0)                          # up
    # ...pitch about r:
    ca, sa = math.cos(alpha), math.sin(alpha)
    f, u = tuple(ca * fi + sa * ui for fi, ui in zip(f, u)), \
           tuple(-sa * fi + ca * ui for fi, ui in zip(f, u))
    # ...roll about f:
    cr, sr = math.cos(rho), math.sin(rho)
    r, u = tuple(cr * ri + sr * ui for ri, ui in zip(r, u)), \
           tuple(-sr * ri + cr * ui for ri, ui in zip(r, u))

    ct = math.cos(theta)
    v_f, v_r, v_u = ct * math.cos(phi), ct * math.sin(phi), math.sin(theta)
    v = tuple(v_f * fi + v_r * ri + v_u * ui for fi, ri, ui in zip(f, r, u))
    elev = math.asin(max(-1.0, min(1.0, v[2])))
    bearing = math.atan2(v[1], v[0])
    return elev, bearing


def detection_ground_point(pose, x_norm, y_norm, *,
                           camera_height=DEFAULT_CAMERA_HEIGHT_M,
                           max_range_m=DEFAULT_MAX_RANGE_M,
                           errors=GSV_ERRORS,
                           apply_pose=True):
    """Flat-ground world estimate of a normalized detection, or None if unplaceable.

    The equirect convention (verified for both sources, see CLAUDE.md): the center
    column x=0.5 is the camera heading, y=0.5 is the pano-frame horizon, so
    phi = (x-0.5)*2*pi and theta = (0.5-y)*pi (positive up). apply_pose=True
    additionally rotates the direction by the pano's pitch/roll — measured to
    HURT on GSV (see _world_ray: streetlevel's equirects are already
    gravity-rectified), so fusion passes apply_pose=False; the flat path is
    also always used when the pose carries no pitch/roll (Mapillary).

    Returns None for rays at/above the horizon (within MIN_DEPRESSION_RAD) and
    for ranges beyond max_range_m — dropped, never clamped.
    """
    phi = (x_norm - 0.5) * 2.0 * math.pi
    theta = (0.5 - y_norm) * math.pi

    if apply_pose and pose.has_pitch_roll:
        elev, bearing = _world_ray(pose, phi, theta)
    else:
        elev = theta
        bearing = math.radians(pose.heading_deg) + phi

    depression = -elev
    if depression <= MIN_DEPRESSION_RAD:
        return None
    d = camera_height / math.tan(depression)
    if d > max_range_m:
        return None

    # Anisotropic 1-sigma errors. Along-ray: exact sensitivity of d = h/tan(delta)
    # is |dd/d(delta)| = (h^2 + d^2)/h, driven by peak jitter + pitch uncertainty;
    # height error scales as d/h. Cross-ray: range times angular error.
    sigma_theta2 = (errors.sigma_peak_px * RAD_PER_HEATMAP_PX) ** 2 \
        + errors.sigma_pitch_rad ** 2
    dd_ddelta = (camera_height ** 2 + d * d) / camera_height
    sigma_along = math.sqrt(dd_ddelta ** 2 * sigma_theta2
                            + (d / camera_height) ** 2 * errors.sigma_height_m ** 2)
    sigma_cross = d * math.sqrt((errors.sigma_peak_px * RAD_PER_HEATMAP_PX) ** 2
                                + errors.sigma_heading_rad ** 2)

    lat = pose.lat + d * math.cos(bearing) / METERS_PER_DEG_LAT
    lng = pose.lng + d * math.sin(bearing) / (
        METERS_PER_DEG_LAT * max(0.01, math.cos(math.radians(pose.lat))))
    return GroundEstimate(lat, lng, d, math.degrees(bearing) % 360.0,
                          sigma_along, sigma_cross)
