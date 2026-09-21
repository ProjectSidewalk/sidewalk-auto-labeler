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

Two ways to sample, and the distinction matters:

  - `depth_at` snaps to a payload pixel and so reproduces streetlevel's raster exactly.
    Use it to reason about the payload itself.
  - `ray_depth_at` / `ground_range_at` keep the exact coordinate and intersect the true
    ray. Use these for a *measurement* -- a detection does not land on a pixel centre,
    and at 256 rows the snap is worth up to 6.6% of the horizontal range near the
    horizon. See `_direction_continuous`.

Only the plane lookup is quantized either way: the segmentation really is per-pixel.
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

# Fixed prefix: a header-size byte, three uint16 fields (plane count, width, height), and
# the offset byte. That fourth field is a **uint8**, not a uint16 -- reading it wider is
# the upstream bug `parse` documents -- so the prefix is eight bytes, not nine. Verified
# on 3,000 archived payloads across four cities: raw[0] == raw[7] == 8 and
# len(raw) == 8 + width*height + 16*n_planes, without exception. `parse` checks byte 0
# against this so a future layout change fails loudly instead of misparsing silently.
HEADER_BYTES = 8


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

    if len(raw) < HEADER_BYTES:
        raise ValueError(f"depth payload truncated: {len(raw)} bytes, need at least "
                         f"{HEADER_BYTES} for the header")
    # Byte 0 states the header size and is 8 in every payload observed (3,000 sampled
    # across four cities). Nothing below would notice if it changed -- the fields would
    # simply be read from the wrong offsets and yield plausible garbage -- so check it.
    if raw[0] != HEADER_BYTES:
        raise ValueError(
            f"unexpected depth header size {raw[0]}, expected {HEADER_BYTES} -- the payload "
            f"layout has changed and every derived number is suspect. Re-run "
            f"scripts/harvest_depth.py --check-convention before trusting this module.")

    # `offset` is a **uint8 at byte 7**, not a uint16. This matters: streetlevel (and the
    # GSVPanoDepth.js it derives from) reads it as a uint16, which absorbs byte 8 — the
    # *first plane index* — as a high byte. When the top-left pixel is sky (index 0) the
    # misread is invisible and offset comes out as 8; when it is anything else, offset
    # comes out as 8 + 256*index, the plane list is then read past the end of the buffer,
    # and the parse throws. That is ~0.3% of panoramas, and it is why those are
    # unreadable upstream rather than merely unusual.
    n_planes, width, height = struct.unpack_from("<HHH", raw, 1)
    offset = raw[7]

    body = width * height
    expected = offset + body + 16 * n_planes
    if offset < HEADER_BYTES or len(raw) < expected:
        raise ValueError(
            f"depth payload truncated: {len(raw)} bytes, need {expected} for "
            f"{width}x{height} and {n_planes} planes at offset {offset}")

    indices = raw[offset:offset + body]
    planes = []
    base = offset + body
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
    """Unit ray for a RAW (unmirrored) row/column, per the module docstring.

    This is the *quantized* ray: the direction through the centre of one payload pixel.
    It is what reproduces streetlevel's raster, so `depth_at` uses it -- but it is the
    wrong ray for a coordinate that did not come from a pixel centre. See
    `_direction_continuous`.
    """
    theta = (payload.height - row - 0.5) / payload.height * math.pi
    phi = (payload.width - col - 0.5) / payload.width * 2.0 * math.pi + math.pi / 2.0
    st = math.sin(theta)
    return st * math.cos(phi), st * math.sin(phi), math.cos(theta)


def _direction_continuous(x_norm, y_norm):
    """Unit ray for an exact normalized STORED coordinate -- no pixel snapping.

    A detection lands at an arbitrary (x, y), not at a payload pixel centre. Snapping the
    ray to the nearest of 256 rows costs up to half a row of elevation (0.30 deg), and
    near the horizon that error is amplified by 1/(sin d * cos d): measured against a flat
    plane at the real 512x256 payload size it reaches **6.6% -- +/-1.2 m at 20-25 m**,
    with an alternating sign that reads as noise rather than bias. Since the whole point
    of harvesting depth is to remove a range bias (#40), spending a chunk of it back on
    quantization would be self-defeating, so range queries intersect the true ray.

    Only the *plane lookup* is legitimately quantized (the segmentation really is
    per-pixel); once the plane is known it is a continuous surface and the intersection
    should be exact.

    Note there is **no width-col-1 flip here**: the mirror cancels. `_direction` takes a
    RAW column and counts azimuth down from `width`, whereas a stored column already counts
    up, so the two compose to `phi = x_norm * 2pi + pi/2`. That is easy to get backwards,
    and hard to catch: mirroring phi flips the sign of vx and nothing else, so any plane
    whose normal has nx == 0 -- which includes every level ground plane, the case one
    naturally reaches for -- returns the identical answer either way. Only a plane with a
    nonzero normal-x pins the convention down; tests/test_depth.py uses several. At a pixel
    centre this agrees with `_direction` bit-for-bit.
    """
    theta = (1.0 - y_norm) * math.pi
    phi = x_norm * 2.0 * math.pi + math.pi / 2.0
    st = math.sin(theta)
    return st * math.cos(phi), st * math.sin(phi), math.cos(theta)


def _plane_at(payload, x_norm, y_norm):
    """(plane, row, raw_col) under a normalized coordinate; plane is None for sky.

    The row needs no un-mirroring -- only columns are flipped -- but the column does, so
    both are returned ready for `_direction`.

    The lookup snaps to a pixel because the plane segmentation is genuinely per-pixel --
    this is the one place quantization is correct rather than merely convenient.
    """
    col = min(payload.width - 1, max(0, int(x_norm * payload.width)))
    row = min(payload.height - 1, max(0, int(y_norm * payload.height)))
    raw_col = _raw_column(payload, col)
    idx = payload.indices[row * payload.width + raw_col]
    if idx == SKY or idx >= len(payload.planes):
        return None, row, raw_col
    return payload.planes[idx], row, raw_col


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
    # Taken as a weighted percentile over (height, pixel count) pairs rather than by
    # materializing one float per pixel: identical result, but it does not build a
    # 131k-element list per panorama across a 170k-panorama archive.
    weighted = sorted((p.d, c) for c, _, p, _ in candidates)
    total_px = sum(c for _, c in weighted)

    def wpct(q):
        rank, seen = int(q * (total_px - 1)), 0
        for value, c in weighted:
            seen += c
            if seen > rank:
                return value
        return weighted[-1][0]

    spread = wpct(0.9) - wpct(0.1)

    return GroundPlane(camera_height_m=best.d, tilt_deg=tilt, pixel_share=count / total,
                       n_ground_planes=len(candidates), height_spread_m=spread)


def _intersect(plane, direction):
    """Ray-plane intersection distance, or None if the ray runs parallel to the plane."""
    vx, vy, vz = direction
    denom = vx * plane.nx + vy * plane.ny + vz * plane.nz
    if denom == 0:
        return None
    return abs(plane.d / denom)


def depth_at(payload, x_norm, y_norm):
    """Euclidean ray distance at the payload PIXEL containing a normalized coordinate.

    Raster-faithful: this snaps to a pixel centre and so reproduces streetlevel's own
    depth map exactly, which is what `--check-convention` and the offline cross-check in
    tests/test_depth.py compare against. Use it to reason about the payload.

    For a measurement at an arbitrary coordinate -- a detection, say -- use
    `ray_depth_at` / `ground_range_at`, which intersect the true ray instead of a
    quantized one.

    (x_norm, y_norm) use the pipeline's convention -- x from the left of the rastered
    image, y from the top -- the same one detections are stored in.
    """
    p, row, raw_col = _plane_at(payload, x_norm, y_norm)
    if p is None:
        return None
    return _intersect(p, _direction(payload, row, raw_col))


def ray_depth_at(payload, x_norm, y_norm):
    """Euclidean distance along the exact ray at a normalized coordinate, or None for sky.

    Same plane lookup as `depth_at`, but intersected at the un-snapped ray direction, so
    the result is continuous in (x_norm, y_norm) rather than piecewise-constant across a
    pixel. Identical to `depth_at` at pixel centres. See `_direction_continuous` for why
    the difference is worth up to 6.6% of the range.
    """
    p, _, _ = _plane_at(payload, x_norm, y_norm)
    if p is None:
        return None
    return _intersect(p, _direction_continuous(x_norm, y_norm))


def ground_range_at(payload, x_norm, y_norm):
    """Horizontal distance from the camera at a normalized coordinate, or None.

    The horizontal component of `ray_depth_at` -- directly comparable to what
    `geo.detection_ground_point` computes as `camera_height / tan(depression)`, and the
    intended consumer of this whole archive (#40). Both factors come from the same
    un-snapped coordinate; mixing a snapped depth with an un-snapped cosine is what the
    6.6% error in `_direction_continuous` refers to.
    """
    d = ray_depth_at(payload, x_norm, y_norm)
    if d is None:
        return None
    theta = (0.5 - y_norm) * math.pi        # elevation, positive up
    return d * math.cos(theta)
