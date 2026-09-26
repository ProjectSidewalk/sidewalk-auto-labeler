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

import depth as depthlib

METERS_PER_DEG_LAT = 111320.0
EARTH_RADIUS_M = 6371000.0

# Ground raycast envelope. The 0.02 rad (~1.15 deg) guard rejects at/above-horizon
# peaks and the near-horizon band where tan() explodes; both bounds are inherited
# from scripts/thinning_experiment.py, but the raycast here DROPS beyond max range
# where that script clamped (a clamp fabricates ranges — ~20% of paterson's stored
# detections sat on the old 30 m clamp).
MIN_DEPRESSION_RAD = 0.02
DEFAULT_MAX_RANGE_M = 25.0

# Camera height (issue #40). Every raycast uses DEFAULT_CAMERA_HEIGHT_M unless the caller
# asks for PER_PANO, which uses the pano's own height where it has a measurement (GSV's
# depth ground plane; see depth.camera_height_fields and fuse_sites.load_results) and the
# default where it does not.
#
# PER_PANO is opt-in, not the default, on evidence (docs/camera-height-study.md). The
# depth ground plane ranks rigs correctly -- the 2025-26 GSV rig really is lower -- but
# measured against the imagery's own geometry (bearing-only triangulation of multi-view
# ramps, iterated to a self-consistent height) it runs 6-16% short, city by city, and
# world P/R against RampNet GT cannot tell any of the height models apart. The 2.6 m
# constant runs ranges ~2-4% long for pre-2025 GSV rigs and 31-35% long for the 2025-26
# one (2.6/1.98, 2.6/1.92); panos with no measurement triangulate to >= 2.6 m, which is
# why they fall back to it rather than to the measured median.
DEFAULT_CAMERA_HEIGHT_M = 2.6
PER_PANO = 'per-pano'
# Under PER_PANO a measured height goes through depth.believe_height (#44): today it is
# always kept (the one candidate gate only flags), with its sigma from the p90-p10 spread of
# camera height across its ground planes -- p90-p10 of a normal is 2.563 sigma. A height a
# future rule rejects falls back to the default like an unmeasured pano.
SIGMA_PER_P10_P90 = depthlib.SIGMA_PER_P10_P90

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

    def key(self, e, n):
        """Cell key for (e, n) — public so movers can detect cell crossings."""
        return (math.floor(e / self.cell), math.floor(n / self.cell))

    def add(self, e, n, item):
        self.cells.setdefault(self.key(e, n), []).append(item)

    def near(self, e, n):
        kx, ky = self.key(e, n)
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
    camera_height_m: float | None = None         # measured (GSV depth); None = unknown
    camera_height_spread_m: float | None = None  # p90-p10 over the pano's ground planes
    ground_tilt_deg: float | None = None         # that ground plane's tilt (#44 QC input)
    camera_height_vintage_m: float | None = None  # median measured height of the pano's
                                                  # vintage in its run (#44 QC input)


def pano_pose(pano):
    """Extract a Pose from a results.jsonl pano block.

    GSV blocks carry heading/pitch/roll in degrees (sometimes as [0, 360) — always
    normalized here). Mapillary blocks written since #42 carry the gravity-relative
    pitch/roll parsed from OpenSfM's rotation (mapillary_pitch_roll; roll in Project
    Sidewalk's sign, the one _world_ray consumes); older blocks, and panos whose
    rotation is missing or failed (tilt past MAX_POSE_TILT_DEG), store null. A null
    pose gets pitch=roll=0 with has_pitch_roll=False and the wider Mapillary error
    model absorbs the unknown tilt. This reads the block only: the road-relative
    angles fusion can apply need the whole sequence, so fuse_sites.load_results
    derives them.

    `camera_height_m` / `camera_height_spread_m` are read when present (GSV blocks written
    since #40 carry them; null or absent means unmeasured). They only take effect for a
    raycast asked for with camera_height=PER_PANO -- see camera_height_for.
    """
    src = pano.get('source') or ''
    heading = norm_deg(float(pano['camera_heading']))
    pitch, roll = pano.get('camera_pitch'), pano.get('camera_roll')
    height = pano.get('camera_height_m')
    spread = pano.get('camera_height_spread_m')
    height = None if height is None else float(height)
    spread = None if spread is None else float(spread)
    tilt = pano.get('ground_tilt_deg')
    tilt = None if tilt is None else float(tilt)
    vintage = pano.get('camera_height_vintage_m')   # set by fuse_sites.load_results only
    vintage = None if vintage is None else float(vintage)
    if pitch is None or roll is None:
        return Pose(pano['lat'], pano['lng'], heading, 0.0, 0.0, False, src,
                    height, spread, tilt, vintage)
    return Pose(pano['lat'], pano['lng'], heading,
                norm_deg(float(pitch)), norm_deg(float(roll)), True, src,
                height, spread, tilt, vintage)


# --- OpenSfM / Mapillary rotation -> (heading, pitch, roll) ----------------------------
# Mapillary serves no pitch/roll fields, only OpenSfM's `computed_rotation`: an axis-angle
# (Rodrigues) vector for the WORLD->CAMERA rotation, world = topocentric ENU, camera =
# OpenCV (x right, y down, z forward). The convention was locked four independent ways in
# the #42 study (docs/mapillary-tilt-study.md section 5.1): the matrix's yaw equals
# Mapillary's own computed_compass_angle to 1e-11 deg on all 191,286 records, pitch and
# roll equal what Project Sidewalk's MapillaryViewer.extractPitchRoll derives, re-rendering
# panos with it levels them, and no reviewer-marked ramp raycasts above the horizon under it.

# A pose past this tilt (camera-up vs world-up) is a failed reconstruction, not a rig: 30
# of 191,286 panos across the five Mapillary runs exceed it, reaching 170 deg (25 in
# Clovis, i.e. upside down). Such a pano keeps a null pitch/roll -- an honest "unknown"
# beats a confidently wrong pose, which would move every ray -- and the flat raycast plus
# the wide Mapillary error model handle it exactly as before #42.
MAX_POSE_TILT_DEG = 45.0


def rotation_matrix(rvec):
    """Rodrigues: axis-angle vector -> 3x3 rotation matrix (nested lists)."""
    rx, ry, rz = (float(v) for v in rvec)
    th = math.sqrt(rx * rx + ry * ry + rz * rz)
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = rx / th, ry / th, rz / th
    c, s = math.cos(th), math.sin(th)
    v = 1.0 - c
    return [[c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v]]


def opensfm_pose(rvec):
    """(heading_deg, pitch_deg, roll_deg) of a world->camera axis-angle rotation, in
    _world_ray's convention (roll in Project Sidewalk's sign), plus tilt_deg (angle
    between camera-up and world-up). Rows of R are the camera axes in ENU:
    R[0] = right, R[1] = down, R[2] = forward.

    Example (a level camera facing north is a quarter turn about east):
        >>> p = opensfm_pose([math.pi / 2, 0.0, 0.0])
        >>> [round(p[k], 9) + 0.0 for k in ('heading_deg', 'pitch_deg', 'roll_deg', 'tilt_deg')]
        [0.0, 0.0, 0.0, 0.0]
    """
    R = rotation_matrix(rvec)
    fwd_e, fwd_n, fwd_u = R[2]
    right_e, right_n, right_u = R[0]
    up_u = -R[1][2]
    heading = math.atan2(fwd_e, fwd_n)
    pitch = math.asin(max(-1.0, min(1.0, fwd_u)))
    # Roll: angle of the camera's right axis about the (pitched) forward axis,
    # measured from the level right axis toward the pitched up axis, then negated into
    # PS's sign -- exactly the roll _world_ray applies after yaw and pitch.
    cp, sp = math.cos(heading), math.sin(heading)
    ca, sa = math.cos(pitch), math.sin(pitch)
    level_right = (cp, -sp, 0.0)                    # ENU: (E, N, U)
    pitched_up = (-sa * sp, -sa * cp, ca)
    roll = math.atan2(right_e * pitched_up[0] + right_n * pitched_up[1] + right_u * pitched_up[2],
                      right_e * level_right[0] + right_n * level_right[1] + right_u * level_right[2])
    tilt = math.acos(max(-1.0, min(1.0, up_u)))
    return {'heading_deg': math.degrees(heading) % 360.0,
            'pitch_deg': math.degrees(pitch),
            'roll_deg': -math.degrees(roll),
            'tilt_deg': math.degrees(tilt)}


POSE_OK, POSE_MISSING, POSE_MALFORMED, POSE_OVER_TILT = 'ok', 'missing', 'malformed', 'over_tilt'


def mapillary_pose_status(rvec):
    """Why a Mapillary rotation does or does not yield a pose: POSE_OK, POSE_MISSING (no
    computed_rotation -- an unreconstructed image), POSE_MALFORMED (not three finite
    numbers) or POSE_OVER_TILT (tilt past MAX_POSE_TILT_DEG, a failed reconstruction)."""
    if rvec is None:
        return POSE_MISSING
    if not isinstance(rvec, (list, tuple)) or len(rvec) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in rvec):
        return POSE_MALFORMED
    if opensfm_pose(rvec)['tilt_deg'] > MAX_POSE_TILT_DEG:
        return POSE_OVER_TILT
    return POSE_OK


def mapillary_pitch_roll(rvec):
    """(camera_pitch, camera_roll) in degrees for a pano block, from Mapillary's
    `computed_rotation`, or (None, None) when mapillary_pose_status is not POSE_OK.

    Gravity-relative, roll in Project Sidewalk's sign: the values PS's own Mapillary
    viewer stores in pano_data for the same image, and the ones _world_ray consumes. This
    is the single decomposition in the tree -- sources/mapillary.py writes it into new
    records, scripts/backfill_metadata.py --pose into old ones, fuse_sites.load_results
    derives it for records that have neither, and scripts/mapillary_tilt.py studies it.

    Example (Richmond pano 2163793620710887; PS's viewer derived 2.2318, -6.3550):
        >>> pitch, roll = mapillary_pitch_roll([1.3857263583832, 0.71804330335161,
        ...                                     -0.58250746512038])
        >>> round(pitch, 4), round(roll, 4)
        (2.2318, -6.355)
        >>> mapillary_pitch_roll([math.pi, 0.0, 0.0])      # upside down: not a pose
        (None, None)
    """
    if mapillary_pose_status(rvec) != POSE_OK:
        return None, None
    pose = opensfm_pose(rvec)
    return pose['pitch_deg'], pose['roll_deg']


def road_relative_pitch_roll(pitch_deg, roll_deg, heading_deg, grade_deg,
                             travel_bearing_deg):
    """Gravity-relative (pitch, roll) re-expressed relative to a road sloping at
    ``grade_deg`` (positive uphill) along ``travel_bearing_deg`` (#42).

    A flat-ground raycast needs the camera's tilt relative to the ground it meets, and
    Mapillary's rotation gives it relative to gravity; on a slope the two differ by the
    grade. The #42 study measured why that matters: a camera riding level on a car in a
    hill town pitches with the road (slope 0.97 in Morgantown), so correcting it to
    gravity moves rays OFF the road, while a rig tilted on its own mount (Clovis) wants
    the gravity correction. Subtracting the grade is right in both regimes
    (docs/mapillary-tilt-study.md sections 5.3-5.4).

    First order, the ground in pano direction phi rises at grade*cos(phi - phi_travel),
    and _world_ray's elevation is theta + pitch*cos(phi) - roll*sin(phi) (PS roll sign),
    so the grade comes off the pitch along the camera's forward axis and goes ONTO the
    roll across it.

    Example (driving straight uphill at 5 deg with the camera pitched up 5 deg: level
    relative to the road):
        >>> [round(v, 9) + 0.0 for v in road_relative_pitch_roll(5.0, 0.0, 90.0, 5.0, 90.0)]
        [0.0, 0.0]
    """
    phi = math.radians(norm_deg(travel_bearing_deg - heading_deg))
    return (pitch_deg - grade_deg * math.cos(phi),
            roll_deg + grade_deg * math.sin(phi))


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
# Mapillary: consumer rigs, SfM positions with meters of scatter between sequences, and a
# pose that is known (#42) but only road-relative-corrected to ~0.5-1.9 deg median (the
# residual after subtracting a ~1-deg-noisy SfM grade), which is what sigma_pitch covers --
# roughly right for the corrected raycast, far too small for a flat one (median tilt 3 deg);
# fusion still raycasts Mapillary flat by default (fuse_sites.AUTO_ROAD_SOURCES says why),
# so for the default path this sigma is optimistic. Panoramax shares every one of
# those traits and adds raw GPS positions (no SfM; the catalog's own accuracy figure
# is a 4 m 95% interval), so it gets the same model until measured otherwise. Note
# sigma_pitch_rad=1.5 deg is the unknown-tilt budget, and the Panoramax panos that do
# report tilt reach far past it (see pano_pose) — so this is a guess with a known-suspect
# constant, not a measured model. Issue #57 tracks measuring it on a real Panoramax run.
MAPILLARY_ERRORS = ErrorModel(sigma_pitch_rad=math.radians(1.5),
                              sigma_heading_rad=math.radians(1.0),
                              sigma_height_m=0.30,
                              sigma_gps_m=3.0)
CROWDSOURCED_SOURCES = ('mapillary', 'panoramax')


def error_model_for(source):
    return MAPILLARY_ERRORS if source in CROWDSOURCED_SOURCES else GSV_ERRORS


def camera_height_for(pose, errors=None, camera_height=DEFAULT_CAMERA_HEIGHT_M):
    """(camera height, its 1-sigma) to raycast a pano with, in meters.

    ``camera_height`` is a number -- every pano gets that height and the error model's
    flat sigma, which is what every raycast did before #40 -- or PER_PANO: the pano's
    measured height as depth.believe_height returns it (#44), with its sigma floored at the
    error model's, falling back to DEFAULT_CAMERA_HEIGHT_M and the model's sigma for a pano
    with no measurement (or, should a future rule reject one, a rejected one).

    Example:
        >>> pose = pano_pose({'lat': 40.0, 'lng': -74.0, 'camera_heading': 0.0,
        ...                   'camera_pitch': 0.0, 'camera_roll': 0.0, 'source': 'launch',
        ...                   'camera_height_m': 1.73, 'camera_height_spread_m': 0.02})
        >>> camera_height_for(pose)
        (2.6, 0.15)
        >>> camera_height_for(pose, camera_height=PER_PANO)
        (1.73, 0.15)
    """
    errors = errors or error_model_for(pose.source)
    if camera_height != PER_PANO:
        return float(camera_height), errors.sigma_height_m
    if pose.camera_height_m is None:
        return DEFAULT_CAMERA_HEIGHT_M, errors.sigma_height_m
    height, sigma, _ = depthlib.believe_height(
        pose.camera_height_m, pose.camera_height_spread_m, pose.ground_tilt_deg,
        vintage_median_m=pose.camera_height_vintage_m)
    if height is None:
        return DEFAULT_CAMERA_HEIGHT_M, errors.sigma_height_m
    return height, max(errors.sigma_height_m, sigma)


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
    raises the view axis) -> roll (about the forward axis; positive LOWERS the
    camera's right axis, i.e. the camera is rolled clockwise as seen from behind it).
    First-order effect on elevation: elev ~= theta + pitch*cos(phi) - roll*sin(phi).
    Vector components are (north, east, up).

    The roll sign is Project Sidewalk's (MapillaryViewer.extractPitchRoll), which is
    what results.jsonl stores and submits (issue #42): one convention from the record
    to the raycast, with no per-source special case. Until #42 this function used the
    opposite sign (positive lifted the image's right side), so any roll quoted from
    before then -- including the GSV ablation below and docs/mapillary-tilt-study.md's
    per-pano CSVs -- has the other sign. The sign was locked on Mapillary four ways
    (compass identity, PS's own viewer, pixel re-rectification, reviewer marks above
    the horizon; docs/mapillary-tilt-study.md section 5.1).

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
    rho = -math.radians(pose.roll_deg)   # PS sign in, geometric rotation out (see above)

    # Pano-frame basis in world coordinates, after yaw:
    f = (math.cos(psi), math.sin(psi), 0.0)      # forward (center column)
    r = (-math.sin(psi), math.cos(psi), 0.0)     # right
    u = (0.0, 0.0, 1.0)                          # up
    # ...pitch about r:
    ca, sa = math.cos(alpha), math.sin(alpha)
    f, u = tuple(ca * fi + sa * ui for fi, ui in zip(f, u)), \
           tuple(-sa * fi + ca * ui for fi, ui in zip(f, u))
    # ...roll about f (rho > 0 lifts r, so a positive PS roll, which lowers it, is -rho):
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

    ``camera_height`` is a height in meters, or PER_PANO for the pano's own measured
    one where it has it. See camera_height_for.

    Returns None for rays at/above the horizon (within MIN_DEPRESSION_RAD) and
    for ranges beyond max_range_m — dropped, never clamped.
    """
    camera_height, sigma_height = camera_height_for(pose, errors, camera_height)
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
                            + (d / camera_height) ** 2 * sigma_height ** 2)
    sigma_cross = d * math.sqrt((errors.sigma_peak_px * RAD_PER_HEATMAP_PX) ** 2
                                + errors.sigma_heading_rad ** 2)

    lat = pose.lat + d * math.cos(bearing) / METERS_PER_DEG_LAT
    lng = pose.lng + d * math.sin(bearing) / (
        METERS_PER_DEG_LAT * max(0.01, math.cos(math.radians(pose.lat))))
    return GroundEstimate(lat, lng, d, math.degrees(bearing) % 360.0,
                          sigma_along, sigma_cross)


@dataclass(frozen=True)
class PanoProjection:
    x_norm: float
    y_norm: float
    range_m: float
    bearing_deg: float       # world bearing from the camera to the point, [0, 360)


def ground_point_to_pano(pose, lat, lng, *,
                         camera_height=DEFAULT_CAMERA_HEIGHT_M,
                         max_range_m=DEFAULT_MAX_RANGE_M,
                         apply_pose=False):
    """Where a known ground point lands in a pano: the exact inverse of the flat
    path of detection_ground_point (apply_pose=False, which is what production
    fusion uses), or None where the forward function would have dropped it.

    ``camera_height`` resolves exactly as in detection_ground_point, so the pair stays
    inverse per pano whatever that pano's height is.

    ``apply_pose`` exists only for parity with detection_ground_point, so a caller
    that threads ``params.apply_pose`` through both cannot silently end up with a
    forward and an inverse that disagree: the posed rotation is not inverted here,
    and True raises. Production fusion runs apply_pose=False (see _world_ray).

    This is the projection a hard-positive miner needs (RampNet#102): a fused site
    at a known world position becomes a normalized (x, y) training target in a pano
    that produced no detection for it. It inverts the forward function's own
    linearized lat/lng step (a LocalFrame at the camera, longitude scale frozen at
    the camera latitude) rather than the great-circle distance, so
    detection_ground_point -> ground_point_to_pano round-trips to floating-point
    precision; the two agree to ~1e-6 relative at 25 m anyway.

    Example:
        >>> pose = pano_pose({'lat': 40.0, 'lng': -74.0, 'camera_heading': 90.0,
        ...                   'camera_pitch': None, 'camera_roll': None,
        ...                   'source': 'mapillary'})
        >>> g = detection_ground_point(pose, 0.4, 0.6)
        >>> p = ground_point_to_pano(pose, g.lat, g.lng)
        >>> round(p.x_norm, 9), round(p.y_norm, 9)
        (0.4, 0.6)
    """
    if apply_pose and pose.has_pitch_roll:
        raise NotImplementedError(
            'ground_point_to_pano inverts only the flat (gravity-rectified) path; '
            'pass apply_pose=False, as production fusion does')
    camera_height, _ = camera_height_for(pose, camera_height=camera_height)
    e, n = LocalFrame(pose.lat, pose.lng).to_enu(lat, lng)
    d = math.hypot(e, n)
    if d > max_range_m or d < 1e-6:
        return None
    depression = math.atan2(camera_height, d)
    if depression <= MIN_DEPRESSION_RAD:
        return None
    bearing = math.degrees(math.atan2(e, n))
    x_norm = 0.5 + norm_deg(bearing - pose.heading_deg) / 360.0
    y_norm = 0.5 + depression / math.pi
    return PanoProjection(x_norm, y_norm, d, bearing % 360.0)
