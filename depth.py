"""GSV depth payloads: parsing, the ground plane, and per-detection range.

Stdlib-only and torch/numpy-free, like `geo.py` and `detectors/__init__.py`, so anything
in the repo can import it — the harvester, analysis scripts, and eventually `geo.py`
itself.

**A GSV depth payload is not a raster.** It is a list of planes (`normal`, `distance`)
plus one plane index per pixel of a 512x256 grid; a per-pixel depth is the ray-plane
intersection. `streetlevel` computes that raster and then discards the planes, which
throws away the two numbers that matter most here:

  - the dominant ground plane's `distance` IS the camera height, exactly. No fitting,
    no assumed constant. `geo.DEFAULT_CAMERA_HEIGHT_M = 2.6` is above every value
    observed in a 160-pano survey (min 1.114, median 2.206) — see labeler #40.
  - its normal IS the ground tilt, 1-2 degrees even on levelled professional rigs,
    against the perfectly-level ground the raycast currently assumes.

Coordinate convention, transcribed from `streetlevel.streetview.depth.compute_depth_map`
so this module and streetlevel's raster agree exactly (`tests/test_depth.py` checks that
they do, and `scripts/harvest_depth.py --check-convention` re-checks it against live
panoramas):

    theta = (height - row - 0.5) / height * pi        # pi at the top, 0 at nadir
    phi   = (width  - col - 0.5) / width  * 2pi + pi/2
    v     = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    depth = |distance / dot(v, normal)|

so **+z points at the nadir, i.e. DOWN**, and a ground plane's normal is ~(0, 0, -1).
Note also that `compute_depth_map` writes the value computed at column `col` into stored
column `width - col - 1`: the raster is mirrored relative to the raw index array. That
flip is the easiest thing to get wrong here, so it lives in exactly one place
(`_raw_column`) rather than being open-coded at each call site.
"""
import base64
import math
import struct
from collections import Counter
from dataclasses import dataclass

# A ground plane must be within this of vertical, and have at least this share of its
# pixels below the horizon, to be a floor rather than a wall or a ceiling.
GROUND_MAX_TILT_DEG = 18.0
GROUND_MIN_BELOW_HORIZON = 0.9

# Google returns a trivial fallback reconstruction for some panoramas: a single ground
# plane at exactly 2.500 m with a normal of exactly (0, 0, -1) and no tilt. Those are
# defaults, not measurements, and must not be fed to a fit. The structural signal (how
# many planes the whole payload has) is far more robust than testing the value 2.5,
# which also occurs as one plane among many in perfectly good reconstructions.
DEGENERATE_MAX_PLANES = 2

SKY = 0  # plane index 0 means "no plane" -- sky, or unreconstructed


@dataclass(frozen=True)
class Plane:
    nx: float
    ny: float
    nz: float
    d: float


@dataclass(frozen=True)
class DepthPayload:
    width: int
    height: int
    planes: list          # Plane, indexed by plane id; entry 0 is the sky sentinel
    indices: bytes        # width*height plane ids, in RAW (unmirrored) column order.
                          # Kept as bytes, not a list: Counter() over bytes iterates at
                          # C speed, which matters when this runs over ~170k panoramas.
                          # One byte per index is the format's own limit, so a payload
                          # can never carry more than 255 planes.

    @property
    def n_planes(self):
        return len(self.planes)

    @property
    def degenerate(self):
        """A fallback reconstruction, not a measurement. See DEGENERATE_MAX_PLANES."""
        return self.n_planes <= DEGENERATE_MAX_PLANES

    @property
    def sky_fraction(self):
        return self.indices.count(SKY) / len(self.indices)


@dataclass(frozen=True)
class GroundPlane:
    camera_height_m: float    # the plane's perpendicular distance from the camera
    tilt_deg: float           # angle between its normal and vertical
    pixel_share: float        # share of the whole image on this plane
    n_ground_planes: int      # roadways are segmented; more than one is normal
    height_spread_m: float    # p90-p10 of camera height across all ground planes,
                              # weighted by pixel count -- a free per-pano uncertainty


def blob_from_response(response):
    """The base64 depth payload inside a raw `streetview.api.find_panorama_by_id` response.

    Taking the blob straight from the API response, rather than going through
    `streetview.find_panorama_by_id(..., download_depth=True)`, skips streetlevel's
    per-pixel raster loop — pure Python over 131k pixels, and the dominant cost when
    harvesting at city scale. We want the payload itself anyway; the raster is
    reconstructible from it at any time.

    The index path mirrors `streetlevel.streetview.parse`: a by-id response wraps the
    panorama message at `[1][0]`, and the depth blob hangs off `[5][0][5][1][2]` of it.
    That is undocumented protobuf-shaped positional data and could move in any Google
    change, so it fails soft — a None here means "no depth for this panorama", which the
    caller records rather than crashes on.
    """
    try:
        return response[1][0][5][0][5][1][2]
    except (IndexError, KeyError, TypeError):
        return None


def parse(b64_string):
    """Parse a base64 depth payload into planes + per-pixel indices.

    Deliberately reimplemented from streetlevel's parser (~40 lines of struct work)
    rather than imported, so this module stays stdlib-only and keeps working if
    streetlevel's internal `parse_planes` moves or changes shape.
    """
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    raw = base64.urlsafe_b64decode(b64_string)

    n_planes, width, height, offset = struct.unpack_from("<HHHH", raw, 1)
    indices = raw[offset:offset + width * height]
    if len(indices) != width * height:
        raise ValueError(f"depth payload truncated: {len(indices)} of {width * height} indices")

    planes = []
    base = offset + width * height
    for i in range(n_planes):
        nx, ny, nz, d = struct.unpack_from("<ffff", raw, base + i * 16)
        planes.append(Plane(nx, ny, nz, d))
    return DepthPayload(width, height, planes, indices)


def _raw_column(payload, col):
    """Stored/rastered column -> the column in the raw index array.

    `compute_depth_map` reads index[row][col] but writes the result to stored column
    width-col-1, so the raster is mirrored relative to the raw indices.
    """
    return payload.width - col - 1


def _direction(payload, row, col):
    """Unit ray for a RAW (unmirrored) row/column, per the module docstring."""
    theta = (payload.height - row - 0.5) / payload.height * math.pi
    phi = (payload.width - col - 0.5) / payload.width * 2.0 * math.pi + math.pi / 2.0
    st = math.sin(theta)
    return st * math.cos(phi), st * math.sin(phi), math.cos(theta)


def ground_plane(payload):
    """The dominant ground plane, or None if the payload has no usable floor.

    Picks by pixel share among planes that are near-horizontal and sit below the
    horizon. Returns the camera height directly -- this is a read, not a fit.
    """
    w, h = payload.width, payload.height
    counts = Counter(payload.indices)
    below = Counter(payload.indices[(h // 2) * w:])

    candidates = []
    for idx, count in counts.items():
        if idx == SKY or idx >= len(payload.planes):
            continue
        p = payload.planes[idx]
        tilt = math.degrees(math.acos(min(1.0, abs(p.nz))))
        if tilt > GROUND_MAX_TILT_DEG:
            continue
        if below[idx] / count < GROUND_MIN_BELOW_HORIZON:
            continue
        candidates.append((count, idx, p, tilt))
    if not candidates:
        return None

    candidates.sort(key=lambda c: -c[0])
    total = len(payload.indices)
    count, _, best, tilt = candidates[0]

    # Spread across every ground plane, pixel-weighted: the roadway is segmented into
    # several planes, and how much they disagree about the camera height is a genuine
    # per-pano uncertainty -- better than the flat sigma_height_m the error model uses.
    heights = []
    for c, _, p, _ in candidates:
        heights.extend([p.d] * c)
    heights.sort()
    spread = (heights[int(0.9 * (len(heights) - 1))]
              - heights[int(0.1 * (len(heights) - 1))])

    return GroundPlane(camera_height_m=best.d, tilt_deg=tilt, pixel_share=count / total,
                       n_ground_planes=len(candidates), height_spread_m=spread)


def depth_at(payload, x_norm, y_norm):
    """Euclidean ray distance at a normalized equirect coordinate, or None for sky.

    (x_norm, y_norm) use the pipeline's convention -- x from the left of the rastered
    image, y from the top -- the same one detections are stored in, so this can be
    called directly with a detection's coordinates.
    """
    col = min(payload.width - 1, max(0, int(x_norm * payload.width)))
    row = min(payload.height - 1, max(0, int(y_norm * payload.height)))
    raw_col = _raw_column(payload, col)

    idx = payload.indices[row * payload.width + raw_col]
    if idx == SKY or idx >= len(payload.planes):
        return None
    p = payload.planes[idx]
    vx, vy, vz = _direction(payload, row, raw_col)
    denom = vx * p.nx + vy * p.ny + vz * p.nz
    if denom == 0:
        return None
    return abs(p.d / denom)


def ground_range_at(payload, x_norm, y_norm):
    """Horizontal distance from the camera at a normalized coordinate, or None.

    The horizontal component of `depth_at` -- directly comparable to what
    `geo.detection_ground_point` computes as `camera_height / tan(depression)`.
    """
    d = depth_at(payload, x_norm, y_norm)
    if d is None:
        return None
    theta = (0.5 - y_norm) * math.pi        # elevation, positive up
    return d * math.cos(theta)
