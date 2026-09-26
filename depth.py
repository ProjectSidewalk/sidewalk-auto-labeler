"""GSV depth payloads: parsing, the ground plane, and per-detection range.

Stdlib-only and torch/numpy-free, like `geo.py` and `detectors/__init__.py`, so anything
in the repo can import it — the harvester, analysis scripts, and eventually `geo.py`
itself.

**A GSV depth payload is not a raster.** It is a list of planes (`normal`, `distance`)
plus one plane index per pixel of a 512x256 grid; a per-pixel depth is the ray-plane
intersection. `streetlevel` computes that raster and then discards the planes, which
throws away the two numbers that matter most here:

  - the dominant ground plane's `distance` is the camera height *in the depth frame*.
    It ranks capture rigs correctly, but runs 6-16% short of the height the imagery's
    own geometry implies (bearing-only triangulation, labeler #40), so it is evidence
    about the rig rather than a drop-in raycast constant -- see
    docs/camera-height-study.md. Not every payload has a real one: see SYNTHETIC_GROUND.
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

**Two column frames, and which one a function takes matters** (labeler #80):

  - the **image frame** -- x from the left of the panorama JPEG, the frame detections,
    Project Sidewalk's `pano_x` and the RampNet benchmark use. Raw index column `c` IS
    image column `c`: the payload's planes overlay the imagery with no flip. In this
    frame the heading is x = 0.5 and x = 0.75 is heading + 90 deg (the camera's right),
    which is **-x** in the depth frame above: `phi = (1 - x) * 2pi + pi/2`.
  - the **raster frame** -- streetlevel's rastered depth map. `compute_depth_map` writes
    the value computed at raw column `col` into stored column `width - col - 1`, so that
    raster is the MIRROR of the imagery: raster column `c` is image column `width-1-c`.

`ray_depth_at` / `ground_range_at` take image-frame coordinates. `depth_at` alone takes
raster-frame coordinates, because its job is to reproduce streetlevel's raster; for an
image coordinate `(x, y)` the raster-faithful value is `depth_at(payload, 1 - x, y)`.
The raster flip lives in exactly one place (`_raw_column`) rather than being open-coded.
Until #80 the range queries used the raster frame too, and so answered for the point at
`(1 - x, y)`; no production path called them.

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

# ...and a second, far commoner fallback that the plane count cannot see: a full
# reconstruction (100-200 planes of real facades) whose *ground* is a stand-in, a plane at
# exactly 2.500 m with a normal of exactly (0, 0, -1). Measured over the four harvested GSV
# runs it is 24,529 of 170,462 non-degenerate payloads (14%; 16% in bend alone), and it
# separates on the normal alone: of the payloads whose dominant ground is exactly level,
# all but 55 sit at exactly 2.5000 m, while a measured ground plane is 1-2 deg off level
# (median 1.2-1.4 deg in bend and gainesville). So the test is structural again -- "is
# the normal exactly vertical?" -- not "is the value 2.5?".
#
# Heights outside this window are not a camera on a vehicle or a backpack; they are a
# ground plane picked from the wrong surface (0.04 m at 11.9 deg of tilt, say). Measured
# ground heights put 1% below 1.09 m and 1% above 2.49 m.
PLAUSIBLE_HEIGHT_M = (0.8, 3.5)

# Why a ground plane is or is not a camera-height measurement. Stored per pano in
# results.jsonl (`camera_height_status`), so a consumer can tell "no depth served" from
# "depth served, ground was a stand-in" without re-fetching anything.
MEASURED = "measured"
NO_DEPTH = "no_depth"            # no payload in the response
UNPARSED = "unparsed"            # a payload was there but did not parse (retryable: a
                                 # later harvest may still supply the height)
DEGENERATE = "degenerate"        # the whole payload is a fallback (DEGENERATE_MAX_PLANES)
NO_GROUND = "no_ground"          # a real payload with no plane that qualifies as a floor
SYNTHETIC_GROUND = "synthetic_ground"  # exactly-level stand-in ground (see above)
IMPLAUSIBLE = "implausible"      # outside PLAUSIBLE_HEIGHT_M

# Which MEASURED heights to flag (#44). Pre-registered tests on the four harvested GSV runs
# (scripts/height_qc.py; runs/_pooled/height_qc/report.md), judged against bearing-only
# triangulation, found ONE candidate gate that passed when the cities were pooled: a depth
# height QC_VINTAGE_DEVIATION_M or more from its vintage's median (same run, same capture
# year). It is NOT adopted as a rejection: per city it fails its own rule in gainesville
# (11% flagged) and sao_paulo (1.38x ratio) -- the pooled pass was carried by bend, 45% of
# the pool -- and on the 2025-26 rig raycasting a flagged pano at the 2.6 m default is worse
# than keeping its depth height (docs/camera-height-study.md, "QC rule (#44)"). So the gate
# only FLAGS: believe_height reports it, sites_meta.json counts it, the height is kept.
# Every other candidate (tilt, plane spread, ground pixel share, plane count, sky share,
# the sub-1.5 m tail by tilt) failed its rule.
QC_VINTAGE_DEVIATION_M = 0.40
# A vintage median needs this many measured panos (T1's minimum); a smaller or undated
# vintage gets none, and the gate cannot fire on it.
QC_MIN_VINTAGE_PANOS = 300
# A measured height's sigma is the p90-p10 spread of camera height across its ground planes
# (a segmented roadway disagrees with itself where it slopes or crowns); p90-p10 of a normal
# is 2.563 sigma, and geo floors it at the error model's. T4 found the spread does NOT
# predict the triangulation residual (its p68 is flat across spread quartiles) and the
# pre-registered rule swapped in a constant 0.259 m (the kept-pano p68 of |implied - k *
# depth|). That constant was then scored against RampNet GT and made p90 GT-to-site
# placement worse in all four cities (+0.12 to +0.30 m; P/R unchanged within noise), so
# it was reverted and the spread sigma stays -- per-pano output is exactly #68's.
SIGMA_PER_P10_P90 = 1.0 / 2.563
FLAGGED_QC = "flagged_qc"        # prefix of believe_height's flag reasons

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
    exactly_level: bool = False  # normal is exactly (0, 0, +-1): Google's stand-in
                                 # ground, not a measurement (see SYNTHETIC_GROUND)


def classify_height(height_m, tilt_deg, *, degenerate=False, exactly_level=None):
    """Whether a ground plane's distance is a camera-height measurement, as a status.

    Returns one of the status constants above; only MEASURED means `height_m` may be used
    as the camera height. `exactly_level` is the test on the plane normal; callers that
    only have index.csv's tilt (3 decimals) pass None and it is compared against 0 instead.
    That is the same test: the smallest nonzero tilt a float32 unit normal can carry is
    0.0198 deg, which rounds to 0.020, never to 0.000.

    Example:
        >>> classify_height(2.5, 0.0)
        'synthetic_ground'
        >>> classify_height(1.73, 2.01)
        'measured'
    """
    if degenerate:
        return DEGENERATE
    if height_m is None:
        return NO_GROUND
    if exactly_level if exactly_level is not None else tilt_deg == 0.0:
        return SYNTHETIC_GROUND
    lo, hi = PLAUSIBLE_HEIGHT_M
    if not lo <= height_m <= hi:
        return IMPLAUSIBLE
    return MEASURED


def believe_height(height_m, spread_m, tilt_deg, *, vintage_median_m=None):
    """(height, sigma, reason) to raycast a MEASURED height with, per the #44 QC tests.

    Applied by consumers (geo.camera_height_for under PER_PANO), never written to a pano
    block -- the block records what was measured. Today no gate rejects: the height always
    comes back with its spread-based sigma (SIGMA_PER_P10_P90; geo floors it), and `reason`
    is `measured` or `flagged_qc:<gate>` (counted in sites_meta.json). A height of None is reserved for a future rule that does reject; the
    caller then falls back to its default. The vintage gate needs `vintage_median_m`
    (fuse_sites.load_results supplies it); `spread_m` and `tilt_deg` are taken so a gate on
    them would not change the signature -- neither passed its test.

    Example:
        >>> believe_height(1.80, 0.0, 1.2, vintage_median_m=1.76)
        (1.8, 0.0, 'measured')
        >>> believe_height(1.20, 0.0, 1.2, vintage_median_m=1.76)
        (1.2, 0.0, 'flagged_qc:vintage_deviation')
    """
    sigma = (spread_m or 0.0) * SIGMA_PER_P10_P90
    if (vintage_median_m is not None
            and abs(height_m - vintage_median_m) >= QC_VINTAGE_DEVIATION_M):
        return height_m, sigma, f"{FLAGGED_QC}:vintage_deviation"
    return height_m, sigma, MEASURED


def camera_height_fields(payload):
    """The camera-height fields of a results.jsonl pano block, from a parsed payload.

    `payload` may be None (no depth in the response). Every key is always present so a
    consumer never has to distinguish a missing key from a null; `camera_height_m` is
    non-null only when the status is MEASURED, so nothing downstream can use a stand-in
    by accident. The raw ground distance of a non-measurement is deliberately not kept --
    the harvested archive has it (scripts/harvest_depth.py) for anyone studying them.
    """
    fields = {"camera_height_m": None, "camera_height_spread_m": None,
              "ground_tilt_deg": None, "depth_planes": None,
              "camera_height_status": NO_DEPTH}
    if payload is None:
        return fields
    fields["depth_planes"] = payload.n_planes
    ground = None if payload.degenerate else ground_plane(payload)
    status = classify_height(ground and ground.camera_height_m,
                             ground and ground.tilt_deg,
                             degenerate=payload.degenerate,
                             exactly_level=ground and ground.exactly_level)
    fields["camera_height_status"] = status
    if status == MEASURED:
        fields["camera_height_m"] = round(ground.camera_height_m, 4)
        fields["camera_height_spread_m"] = round(ground.height_spread_m, 4)
        fields["ground_tilt_deg"] = round(ground.tilt_deg, 3)
    return fields


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
    # and the parse throws. That is ~0.3-0.5% of panoramas, and it is why those are
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
    """RASTER-frame column -> the column in the raw index array.

    `compute_depth_map` reads index[row][col] but writes the result to stored column
    width-col-1, so streetlevel's raster is mirrored relative to the raw indices -- and
    so relative to the imagery, whose columns ARE the raw ones. Only `depth_at` (the
    raster-faithful path) uses this; an image-frame column needs no mapping.
    """
    return payload.width - col - 1


def _direction(payload, row, col):
    """Unit ray for a RAW (unmirrored) row/column, per the module docstring. A raw column
    is also an image column, so this is the image-frame ray through a pixel centre.

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
    """Unit ray for an exact normalized IMAGE-frame coordinate -- no pixel snapping.

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

    This is `_direction` made continuous: an image column is a raw column, and
    `_direction` counts azimuth down from `width`, so `phi = (1 - x_norm) * 2pi + pi/2`.
    At a pixel centre the two agree bit-for-bit. Until #80 this was `x_norm * 2pi + pi/2`,
    the raster frame's ray. That is easy to get backwards and hard to catch: mirroring
    phi flips the sign of vx and nothing else, so any plane whose normal has nx == 0 --
    every level ground plane, the case one naturally reaches for -- returns the identical
    answer either way, and a synthetic payload built under the wrong assumption passes
    its own tests. What pins it is the imagery: tests/test_depth.py checks the lookup
    against sky and surface read off a real panorama.
    """
    theta = (1.0 - y_norm) * math.pi
    phi = (1.0 - x_norm) * 2.0 * math.pi + math.pi / 2.0
    st = math.sin(theta)
    return st * math.cos(phi), st * math.sin(phi), math.cos(theta)


def _plane_at(payload, x_norm, y_norm, *, raster=False):
    """(plane, row, raw_col) under a normalized coordinate; plane is None for sky.

    The coordinate is in the IMAGE frame (raw column == image column) unless `raster` is
    set, when it is in streetlevel's mirrored raster frame -- `depth_at`'s, and no other
    caller's. Rows are never flipped. Both are returned ready for `_direction`.

    The lookup snaps to a pixel because the plane segmentation is genuinely per-pixel --
    this is the one place quantization is correct rather than merely convenient.
    """
    col = min(payload.width - 1, max(0, int(x_norm * payload.width)))
    row = min(payload.height - 1, max(0, int(y_norm * payload.height)))
    raw_col = _raw_column(payload, col) if raster else col
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
                       n_ground_planes=len(candidates), height_spread_m=spread,
                       exactly_level=best.nx == 0.0 and best.ny == 0.0)


def _intersect(plane, direction):
    """Ray-plane intersection distance, or None if the ray runs parallel to the plane."""
    vx, vy, vz = direction
    denom = vx * plane.nx + vy * plane.ny + vz * plane.nz
    if denom == 0:
        return None
    return abs(plane.d / denom)


def depth_at(payload, x_norm, y_norm):
    """Euclidean ray distance at the payload PIXEL containing a RASTER-frame coordinate.

    Raster-faithful: this snaps to a pixel centre and so reproduces streetlevel's own
    depth map exactly, which is what `--check-convention` and the offline cross-check in
    tests/test_depth.py compare against. Use it to reason about the payload.

    (x_norm, y_norm) are x from the left of streetlevel's RASTER, y from the top. That
    raster is the mirror of the imagery (module docstring), so this is NOT the frame
    detections are stored in: for an image coordinate use `depth_at(payload, 1 - x, y)`,
    or, for a measurement at an arbitrary coordinate -- a detection, say --
    `ray_depth_at` / `ground_range_at`, which take the image frame and intersect the
    true ray instead of a quantized one.
    """
    p, row, raw_col = _plane_at(payload, x_norm, y_norm, raster=True)
    if p is None:
        return None
    return _intersect(p, _direction(payload, row, raw_col))


def ray_depth_at(payload, x_norm, y_norm):
    """Euclidean distance along the exact ray at a normalized IMAGE-frame coordinate
    (x from the left of the panorama JPEG, y from the top), or None for sky.

    Same pixel-snapped plane lookup as `depth_at`, but in the image frame and intersected
    at the un-snapped ray direction, so the result is continuous in (x_norm, y_norm)
    rather than piecewise-constant across a pixel. At pixel centres it equals
    `depth_at(payload, 1 - x_norm, y_norm)`. See `_direction_continuous` for why the
    snap is worth up to 6.6% of the range.
    """
    p, _, _ = _plane_at(payload, x_norm, y_norm)
    if p is None:
        return None
    return _intersect(p, _direction_continuous(x_norm, y_norm))


def ground_range_at(payload, x_norm, y_norm):
    """Horizontal distance from the camera at a normalized IMAGE-frame coordinate, or None.

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
