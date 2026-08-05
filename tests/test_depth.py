"""Contract tests for depth.py — the GSV depth payload parser.

Deliberately narrow, per the repo's lean-test rule. These lock the two things that would
corrupt every derived number silently rather than loudly:

  - the **mirrored column**. `compute_depth_map` reads the raw index at column `col` but
    writes the result to stored column `width-col-1`, so a parser that skips the flip
    still returns plausible distances — just for the wrong side of the panorama. Nothing
    downstream could catch that.
  - the **ground plane read**, since the camera height it returns replaces a hardcoded
    constant in the raycast (labeler #40).

Live agreement with streetlevel's own raster is checked separately and needs the network:
`python scripts/harvest_depth.py <run> --check-convention`.
"""
import math
import base64
import struct

import pytest

import depth as depthlib


def build_payload(width, height, planes, indices):
    """Assemble a synthetic depth blob in Google's layout.

    planes: list of (nx, ny, nz, d), index 0 being the unused sky slot.
    indices: width*height plane ids in RAW (unmirrored) column order.
    """
    offset = 9
    header = bytes([offset]) + struct.pack("<HHHH", len(planes), width, height, offset)
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
    """The flip that a naive parser gets wrong: raw column 0 is stored column width-1."""
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


def test_truncated_payload_is_rejected():
    blob = build_payload(8, 4, [GROUND, GROUND], [1] * 32)
    raw = base64.urlsafe_b64decode(blob + "=" * ((4 - len(blob) % 4) % 4))
    short = base64.urlsafe_b64encode(raw[:20]).decode().rstrip("=")
    with pytest.raises(ValueError, match="truncated"):
        depthlib.parse(short)
