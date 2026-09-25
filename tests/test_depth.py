"""Contract tests for depth.py — the GSV depth payload parser.

Deliberately narrow, per the repo's lean-test rule. These lock the two things that would
corrupt every derived number silently rather than loudly:

  - the **two column frames** (#80). `compute_depth_map` reads the raw index at column
    `col` but writes the result to stored column `width-col-1`, so streetlevel's raster
    is the mirror of the imagery, while a raw column IS an image column. Getting either
    backwards still returns plausible distances — just for the wrong side of the
    panorama — and a synthetic payload built under the same wrong assumption passes its
    own tests, which is how the range queries shipped mirrored. So the image frame is
    pinned against a real panorama (test_image_frame_matches_the_real_panorama).
  - the **ground plane read**, since the camera height it returns replaces a hardcoded
    constant in the raycast (labeler #40).
  - the **snapped/continuous split**: a range query must intersect the true ray, not a
    pixel-quantized one, which is worth up to 6.6% of the range near the horizon.

Agreement with streetlevel's own raster is checked here too, offline against a synthetic
payload — `streetlevel` is already a test dependency, so the check that actually matters
runs on every `pytest` rather than only when someone remembers
`python scripts/harvest_depth.py <run> --check-convention` (which stays, for re-checking
against live panoramas after a streetlevel upgrade).
"""
import math
import base64
import gzip
import hashlib
import json
import os
import random
import struct
from pathlib import Path

import pytest

import depth as depthlib

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_payload(width, height, planes, indices):
    """Assemble a synthetic depth blob in Google's layout.

    planes: list of (nx, ny, nz, d), index 0 being the unused sky slot.
    indices: width*height plane ids in RAW (unmirrored) column order.
    """
    offset = depthlib.HEADER_BYTES
    header = bytes([offset]) + struct.pack("<HHH", len(planes), width, height) + bytes([offset])
    body = bytes(indices)
    tail = b"".join(struct.pack("<ffff", *p) for p in planes)
    return base64.urlsafe_b64encode(header + body + tail).decode().rstrip("=")


GROUND = (0.0, 0.0, -1.0, 2.5)      # level floor, camera 2.5 m above it
WALL = (1.0, 0.0, 0.0, 12.0)        # vertical surface 12 m away


def flat_payload(width=8, height=4):
    """Sky in the top half, ground in the bottom half."""
    indices = [depthlib.SKY] * (width * height // 2) + [1] * (width * height // 2)
    return depthlib.parse(build_payload(width, height, [GROUND, GROUND], indices))


def test_parse_reads_header_and_planes():
    payload = flat_payload()
    assert (payload.width, payload.height) == (8, 4)
    assert payload.n_planes == 2
    assert len(payload.indices) == 32
    assert payload.planes[1].d == pytest.approx(2.5)
    assert payload.sky_fraction == pytest.approx(0.5)


def test_ground_plane_reads_camera_height_exactly():
    g = depthlib.ground_plane(flat_payload())
    assert g.camera_height_m == pytest.approx(2.5)
    assert g.tilt_deg == pytest.approx(0.0, abs=1e-6)
    assert g.pixel_share == pytest.approx(0.5)


def test_ground_plane_reports_tilt():
    tilted = (0.0, math.sin(math.radians(10)), -math.cos(math.radians(10)), 2.5)
    indices = [depthlib.SKY] * 16 + [1] * 16
    payload = depthlib.parse(build_payload(8, 4, [tilted, tilted], indices))
    assert depthlib.ground_plane(payload).tilt_deg == pytest.approx(10.0, abs=1e-3)


def test_ground_plane_rejects_walls():
    """A vertical surface filling the lower half is not a floor."""
    indices = [depthlib.SKY] * 16 + [1] * 16
    payload = depthlib.parse(build_payload(8, 4, [WALL, WALL], indices))
    assert depthlib.ground_plane(payload) is None


def test_depth_at_matches_the_plane_geometry():
    """Bottom row of an 8x4 grid sits 22.5 deg off nadir, so the ray is longer than 2.5 m."""
    payload = flat_payload()
    theta = (4 - 3 - 0.5) / 4 * math.pi          # row 3, per the module convention
    expected = 2.5 / math.cos(theta)
    assert depthlib.depth_at(payload, 0.5, 7 / 8) == pytest.approx(expected)


def test_sky_returns_none():
    assert depthlib.depth_at(flat_payload(), 0.5, 1 / 8) is None


def test_stored_columns_are_mirrored_relative_to_raw_indices():
    """depth_at is in streetlevel's RASTER frame: raw column 0 is raster column width-1."""
    width, height = 8, 4
    indices = [depthlib.SKY] * (width * height)
    for row in range(height // 2, height):       # ground everywhere below the horizon...
        for col in range(width):
            indices[row * width + col] = 1
    indices[3 * width + 0] = 2                   # ...except RAW column 0 of the last row

    near_ground = (0.0, 0.0, -1.0, 1.0)          # distinctly closer than GROUND
    payload = depthlib.parse(build_payload(width, height, [GROUND, GROUND, near_ground],
                                           indices))
    theta = (height - 3 - 0.5) / height * math.pi
    at_left = depthlib.depth_at(payload, 0.5 / width, 3.5 / height)
    at_right = depthlib.depth_at(payload, (width - 0.5) / width, 3.5 / height)

    assert at_right == pytest.approx(1.0 / math.cos(theta))   # raw col 0 -> stored col 7
    assert at_left == pytest.approx(2.5 / math.cos(theta))


def test_degenerate_reconstructions_are_flagged():
    """Google's fallback: one ground plane at exactly 2.500, no tilt. Not a measurement."""
    assert flat_payload().degenerate is True          # 2 planes
    indices = [depthlib.SKY] * 16 + [1] * 16
    rich = depthlib.parse(build_payload(8, 4, [GROUND] * 5, indices))
    assert rich.degenerate is False


def test_nonzero_first_index_still_parses():
    """The upstream bug: `offset` is a uint8 at byte 7, and streetlevel reads it as a
    uint16 that swallows byte 8 — the first plane index. So a panorama whose top-left
    pixel is anything but sky reports offset = 8 + 256*index, sends the plane list past
    the end of the buffer, and throws. ~0.3% of panoramas, and entirely readable."""
    width, height = 8, 4
    # Plane 2 (a wall) fills the top half, so the FIRST index byte is 2 rather than sky's
    # 0 — which is exactly what makes the upstream uint16 misread fire. Ground stays
    # confined below the horizon so it is still recognizable as a floor.
    indices = [2] * 16 + [1] * 16
    payload = depthlib.parse(build_payload(width, height, [GROUND, GROUND, WALL], indices))

    assert payload.indices[0] == 2
    assert payload.n_planes == 3
    assert payload.planes[1].d == pytest.approx(2.5)
    assert depthlib.ground_plane(payload).camera_height_m == pytest.approx(2.5)


def test_truncated_payload_is_rejected():
    blob = build_payload(8, 4, [GROUND, GROUND], [1] * 32)
    raw = base64.urlsafe_b64decode(blob + "=" * ((4 - len(blob) % 4) % 4))
    short = base64.urlsafe_b64encode(raw[:20]).decode().rstrip("=")
    with pytest.raises(ValueError, match="truncated"):
        depthlib.parse(short)


def test_unexpected_header_size_is_rejected():
    """The tripwire: 3,000 archived payloads all have raw[0] == 8. If that ever changes,
    every derived number is suspect, so it must fail loudly rather than misparse."""
    blob = build_payload(8, 4, [GROUND, GROUND], [1] * 32)
    raw = bytearray(base64.urlsafe_b64decode(blob + "=" * ((4 - len(blob) % 4) % 4)))
    raw[0] = 9
    with pytest.raises(ValueError, match="header size"):
        depthlib.parse(base64.urlsafe_b64encode(bytes(raw)).decode().rstrip("="))


# --- the snapped / continuous split ---------------------------------------------------


def tilted_ground(deg, bearing=35.0, height=2.5):
    """A ground plane tilted `deg` from level, leaning towards `bearing`.

    The bearing is not decoration: mirroring phi flips the sign of the ray's **x**
    component and nothing else, so a normal with nx == 0 gives the same answer either way.
    A plane tilted purely north-south would let a mirrored convention pass unnoticed.
    """
    t, b = math.radians(deg), math.radians(bearing)
    return (math.sin(t) * math.cos(b), math.sin(t) * math.sin(b), -math.cos(t), height)


def test_ray_depth_matches_depth_at_on_pixel_centres():
    """The continuous ray must reduce to the quantized one exactly at a pixel centre --
    otherwise it would be a different convention rather than a refinement. The two take
    different frames (#80): ray_depth_at the image, depth_at streetlevel's raster, which
    is its mirror, so image x meets raster 1 - x.

    Planes with a nonzero normal-x are what pin the azimuth down; see tilted_ground.
    """
    W, H = 16, 8
    planes = [GROUND, tilted_ground(12), WALL, tilted_ground(-7, 200.0, 3.1)]
    indices = [1 + (i * 7 + i // W) % 3 for i in range(W * H)]
    payload = depthlib.parse(build_payload(W, H, planes, indices))
    for row in range(H):
        for col in range(W):
            x, y = (col + 0.5) / W, (row + 0.5) / H
            assert depthlib.ray_depth_at(payload, x, y) == pytest.approx(
                depthlib.depth_at(payload, 1.0 - x, y), rel=1e-12)


def test_ground_range_is_exact_between_pixel_centres():
    """A detection lands at an arbitrary y, and snapping it to one of 256 rows costs up to
    6.6% of the range near the horizon. On flat ground the answer is h/tan(depression) for
    every y, not just at row centres."""
    W, H, h = 512, 256, 2.2                       # the real payload grid
    payload = depthlib.parse(build_payload(
        W, H, [(0, 0, -1, h), (0, 0, -1, h)], [0] * (W * H // 2) + [1] * (W * H // 2)))
    for target in (3, 5, 8, 12, 16, 20, 25):
        y = 0.5 + math.atan(h / target) / math.pi
        # rel=1e-6 is float32: plane distances are stored as f32 in the payload, so h comes
        # back as 2.20000004. Snapping the row instead would miss by up to 6.6e-2.
        assert depthlib.ground_range_at(payload, 0.5, y) == pytest.approx(target, rel=1e-6)


def test_ground_range_uses_the_plane_under_the_detection():
    """The (quantized) plane lookup drives the answer, in the IMAGE frame: a nearer patch
    at raw column 0 is under image column 0 -- the left edge -- and nowhere else (#80)."""
    W, H = 64, 32
    indices = [0] * (W * H)
    for row in range(H // 2, H):
        for col in range(W):
            indices[row * W + col] = 1
    indices[(H - 1) * W + 0] = 2                  # RAW column 0 == image column 0
    payload = depthlib.parse(build_payload(W, H, [GROUND, GROUND, (0, 0, -1, 1.0)], indices))
    y = (H - 0.5) / H
    left = depthlib.ground_range_at(payload, 0.5 / W, y)
    right = depthlib.ground_range_at(payload, (W - 0.5) / W, y)
    middle = depthlib.ground_range_at(payload, 0.5, y)
    assert left == pytest.approx(middle * 1.0 / 2.5, rel=1e-6)   # the 1.0 m patch
    assert right == pytest.approx(middle, rel=1e-12)             # its mirror is plain ground


def test_image_frame_matches_the_real_panorama():
    """The regression guard for #80, pinned to reality rather than to a synthetic payload
    built under our own assumption. tests/fixtures/depth_image_frame.json holds 64 points
    of one real paterson panorama labelled sky/surface from its JPEG alone, at cells where
    the image frame and its mirror disagree about sky; ray_depth_at (None = sky) must side
    with the image on nearly all of them, and the mirror on nearly none.

    The payload is read from the local depth archive (runs/<city>/depth, written by
    scripts/harvest_depth.py; LABELER_RUNS points elsewhere) and the test is skipped
    where that file is absent; a present file that is not the fixture's payload fails. The offline synthetic tests above pin the code relation between
    the frames; this pins which frame the imagery is in. Measured when written: 59 of 64
    agree in the image frame, 5 of 64 mirrored.
    """
    fx = json.loads((REPO_ROOT / "tests" / "fixtures" / "depth_image_frame.json").read_text())
    runs = Path(os.environ.get("LABELER_RUNS", REPO_ROOT / "runs"))
    path = runs / fx["city"] / "depth" / f"{fx['panorama_id']}.json.gz"
    if not path.exists():
        pytest.skip(f"no local depth archive at {path}")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == fx["payload_file_sha256"], (
        f"{path} is not the payload the fixture was derived from")
    payload = depthlib.parse(json.loads(gzip.decompress(raw))["depth_b64"])

    def agree(mirror):
        return sum((depthlib.ray_depth_at(payload, 1.0 - x if mirror else x, y) is None)
                   == (label == "sky") for x, y, label in fx["points"])

    n = len(fx["points"])
    assert agree(mirror=False) >= 0.85 * n
    assert agree(mirror=True) <= 0.15 * n


def test_agrees_with_streetlevel_raster_offline():
    """The check `--check-convention` makes against live panoramas, run offline on a
    synthetic payload so it guards every commit. streetlevel is already a test dependency.

    The first plane index is forced to sky: streetlevel misreads `offset` as a uint16 and
    cannot parse a payload whose first index is nonzero — the bug depth.py fixes and
    test_nonzero_first_index_still_parses covers.
    """
    from streetlevel.streetview.depth import parse as sl_parse

    rng = random.Random(7)
    W, H = 32, 16
    planes = [GROUND]
    for _ in range(6):
        a, b = rng.uniform(0, 2 * math.pi), rng.uniform(-1.2, 1.2)
        planes.append((math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), math.sin(b),
                       rng.uniform(1.0, 25.0)))
    indices = [rng.randrange(0, len(planes)) for _ in range(W * H)]
    indices[0] = depthlib.SKY

    blob = build_payload(W, H, planes, indices)
    ref = sl_parse(blob).data
    payload = depthlib.parse(blob)
    for row in range(H):
        for col in range(W):
            mine = depthlib.depth_at(payload, (col + 0.5) / W, (row + 0.5) / H)
            theirs = ref[row][col]
            if theirs < 0:                        # streetlevel's INFINITELY_FAR sentinel
                assert mine is None
            else:
                assert mine == pytest.approx(theirs, rel=1e-12)


def test_height_spread_matches_the_pixel_weighted_percentiles():
    """height_spread_m takes a weighted percentile over (height, count) pairs instead of
    materializing one float per pixel; it must agree with the naive form exactly."""
    W, H = 32, 16
    planes = [GROUND] + [(0.0, 0.0, -1.0, d) for d in (2.0, 2.4, 2.6, 3.0, 2.2)]
    rng = random.Random(3)
    indices = [0] * (W * H // 2) + [rng.randrange(1, len(planes)) for _ in range(W * H // 2)]
    payload = depthlib.parse(build_payload(W, H, planes, indices))

    heights = []
    for idx in range(1, len(planes)):
        heights.extend([planes[idx][3]] * indices.count(idx))
    heights.sort()
    naive = (heights[int(0.9 * (len(heights) - 1))] - heights[int(0.1 * (len(heights) - 1))])
    assert depthlib.ground_plane(payload).height_spread_m == pytest.approx(naive)


# --- camera-height classification (labeler #40)

@pytest.mark.parametrize("height, tilt, degenerate, level, expected", [
    (1.87, 1.7, False, False, depthlib.MEASURED),
    (2.5, 0.0, False, True, depthlib.SYNTHETIC_GROUND),   # stand-in ground, full payload
    (2.5, 0.0, False, None, depthlib.SYNTHETIC_GROUND),   # ...as read from index.csv
    (2.5, 1.1, False, False, depthlib.MEASURED),          # 2.5 itself is not suspicious
    (0.04, 11.9, False, False, depthlib.IMPLAUSIBLE),     # a plane from the wrong surface
    (2.5, 0.0, True, True, depthlib.DEGENERATE),
    (None, None, False, None, depthlib.NO_GROUND),
])
def test_classify_height(height, tilt, degenerate, level, expected):
    assert depthlib.classify_height(height, tilt, degenerate=degenerate,
                                    exactly_level=level) == expected


def test_camera_height_fields_null_everything_but_a_measurement():
    assert depthlib.camera_height_fields(None)["camera_height_status"] == depthlib.NO_DEPTH
    # flat_payload is the 2-plane degenerate fallback: a status and a plane count, no height
    fields = depthlib.camera_height_fields(flat_payload())
    assert fields["camera_height_status"] == depthlib.DEGENERATE
    assert fields["depth_planes"] == 2 and fields["camera_height_m"] is None

    tilted = (0.03, 0.0, -0.99955, 1.87)
    indices = [depthlib.SKY] * 16 + [1] * 16
    payload = depthlib.parse(build_payload(8, 4, [tilted, tilted, WALL], indices))
    assert depthlib.ground_plane(payload).exactly_level is False
    fields = depthlib.camera_height_fields(payload)
    assert fields["camera_height_status"] == depthlib.MEASURED
    assert fields["camera_height_m"] == pytest.approx(1.87)
