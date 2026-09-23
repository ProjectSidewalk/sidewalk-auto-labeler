"""Unit tests for the GSV imagery source: the pano record shape (what PS ultimately
consumes) and fetch_pano's skip/failure semantics (what the resume cache depends on)."""
import pytest
from shapely.geometry import box

import depth as depthlib
from conftest import make_metadata
from sources import gsv
from test_depth import build_payload


def test_build_pano_record_shape_and_units():
    pano = gsv.build_pano_record("PID", 44.05, -121.31, make_metadata())
    assert pano["panorama_id"] == "PID"
    assert pano["capture_date"] == "2021-06"
    # width/height come from the highest-resolution entry.
    assert (pano["width"], pano["height"]) == (16384, 8192)
    # Radians in the metadata, degrees on the wire.
    assert pano["camera_heading"] == pytest.approx(90.0)
    assert pano["camera_pitch"] == pytest.approx(1.5)
    assert pano["camera_roll"] == pytest.approx(-0.5)
    assert (pano["lat"], pano["lng"]) == (44.05, -121.31)
    # Historical pano without a date is dropped.
    assert pano["history"] == [{"pano_id": "OLD1", "date": "2019-08"}]
    assert pano["links"] == []


def _patch_fetch(monkeypatch, metadata, image="IMAGE", depth_fields=None):
    fields = None if metadata is None else depth_fields or depthlib.camera_height_fields(None)
    monkeypatch.setattr(gsv, "fetch_metadata_with_retry", lambda pano_id: (metadata, fields))
    monkeypatch.setattr(gsv, "fetch_panorama", lambda md: image)


def test_fetch_pano_success(monkeypatch):
    _patch_fetch(monkeypatch, make_metadata())
    result = gsv.fetch_pano("PID", 44.05, -121.31)
    assert result["status"] == "success"
    assert result["image"] == "IMAGE"
    assert result["pano"]["panorama_id"] == "PID"


def test_fetch_pano_metadata_unavailable_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, None)
    assert gsv.fetch_pano("PID", 0, 0)["status"] == "failure"


def test_fetch_pano_indoor_is_deterministic_skip(monkeypatch):
    _patch_fetch(monkeypatch, make_metadata(source="innerspace"))
    assert gsv.fetch_pano("PID", 0, 0)["status"] == "skipped"


@pytest.mark.parametrize("broken", [
    {"date": None}, {"image_sizes": []}, {"tile_size": None}])
def test_fetch_pano_incomplete_metadata_is_deterministic_skip(monkeypatch, broken):
    _patch_fetch(monkeypatch, make_metadata(**broken))
    assert gsv.fetch_pano("PID", 0, 0)["status"] == "skipped"


def test_fetch_pano_download_failure_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, make_metadata(), image=None)
    assert gsv.fetch_pano("PID", 0, 0)["status"] == "failure"


# --- fetch_pano_by_id (gap fill, issue #32): position comes from the metadata and
# --- the area test happens before the image download.

AREA = box(-122.0, 44.0, -121.0, 45.0)


def test_fetch_pano_by_id_inside_area_uses_metadata_position(monkeypatch):
    _patch_fetch(monkeypatch, make_metadata(lat=44.05, lon=-121.31))
    result = gsv.fetch_pano_by_id("PID", AREA)
    assert result["status"] == "success"
    assert (result["pano"]["lat"], result["pano"]["lng"]) == (44.05, -121.31)


def test_fetch_pano_by_id_outside_area_is_deterministic_skip(monkeypatch):
    def no_download(md):
        raise AssertionError("image downloaded for an outside-the-area pano")
    monkeypatch.setattr(gsv, "fetch_metadata_with_retry",
                        lambda pano_id: (make_metadata(lat=40.0, lon=-121.31), None))
    monkeypatch.setattr(gsv, "fetch_panorama", no_download)
    result = gsv.fetch_pano_by_id("PID", AREA)
    assert result == {"status": "skipped", "reason": "Outside the run area"}


def test_fetch_pano_by_id_positionless_metadata_is_deterministic_skip(monkeypatch):
    _patch_fetch(monkeypatch, make_metadata(lat=None, lon=None))
    assert gsv.fetch_pano_by_id("PID", AREA)["status"] == "skipped"


def test_fetch_pano_by_id_keeps_fetch_pano_semantics(monkeypatch):
    _patch_fetch(monkeypatch, None)
    assert gsv.fetch_pano_by_id("PID", AREA)["status"] == "failure"
    _patch_fetch(monkeypatch, make_metadata(lat=44.05, lon=-121.31, source="innerspace"))
    assert gsv.fetch_pano_by_id("PID", AREA)["status"] == "skipped"


# --- camera height from the depth payload on the metadata response (issue #40)

TILTED_GROUND = (0.03, 0.0, -0.99955, 1.87)   # ~1.7 deg off level: a measurement
STAND_IN_GROUND = (0.0, 0.0, -1.0, 2.5)       # exactly level: Google's stand-in
FACADE = (1.0, 0.0, 0.0, 12.0)


def _response_with_depth(ground):
    """A by-id response shaped like the real one, carrying a 3-plane payload."""
    width, height = 8, 4
    indices = [depthlib.SKY] * (width * height // 2) + [1] * (width * height // 2)
    blob = build_payload(width, height, [ground, ground, FACADE], indices)
    depth_node = [None, None, blob]
    return [None, [[[1], None, None, None, None, [[None, None, None, None, None,
                                                   [None, depth_node]]]]]]


def _patch_api(monkeypatch, response):
    seen = {}

    def fake_parse(resp):
        seen["blob_left"] = depthlib.blob_from_response(resp)
        return make_metadata()
    monkeypatch.setattr(gsv.api, "find_panorama_by_id",
                        lambda pano_id, download_depth: response)
    monkeypatch.setattr(gsv, "parse_panorama_id_response", fake_parse)
    return seen


def test_measured_ground_lands_in_the_pano_block(monkeypatch):
    seen = _patch_api(monkeypatch, _response_with_depth(TILTED_GROUND))
    metadata, fields = gsv.fetch_metadata_with_retry("PID")
    # streetlevel's own depth parser never sees the payload (it throws on ~0.3% of them)
    assert seen["blob_left"] is None
    assert fields["camera_height_status"] == depthlib.MEASURED
    assert fields["camera_height_m"] == pytest.approx(1.87)
    assert fields["ground_tilt_deg"] == pytest.approx(1.72, abs=0.01)
    pano = gsv.build_pano_record("PID", 44.05, -121.31, metadata, fields)
    assert pano["camera_height_m"] == pytest.approx(1.87)
    assert pano["depth_planes"] == 3


def test_stand_in_ground_is_recorded_but_never_used_as_a_height(monkeypatch):
    _patch_api(monkeypatch, _response_with_depth(STAND_IN_GROUND))
    _, fields = gsv.fetch_metadata_with_retry("PID")
    assert fields["camera_height_status"] == depthlib.SYNTHETIC_GROUND
    assert fields["camera_height_m"] is None
    assert fields["depth_planes"] == 3


@pytest.mark.parametrize("node", [
    "AAAA",            # decodes, but is no payload
    123, ["x"], {"a": 1},  # the undocumented path now holds something else entirely
])
def test_corrupt_depth_is_never_a_metadata_failure(monkeypatch, node):
    response = _response_with_depth(TILTED_GROUND)
    response[1][0][5][0][5][1][2] = node
    seen = _patch_api(monkeypatch, response)
    metadata, fields = gsv.fetch_metadata_with_retry("PID")
    assert metadata is not None
    assert seen["blob_left"] is None        # streetlevel's parser never sees it either
    assert fields["camera_height_status"] == depthlib.UNPARSED
    assert fields["camera_height_m"] is None


def test_absent_depth_is_no_depth(monkeypatch):
    response = _response_with_depth(TILTED_GROUND)
    response[1][0][5][0][5] = [None]        # the whole depth node is missing
    _patch_api(monkeypatch, response)
    _, fields = gsv.fetch_metadata_with_retry("PID")
    assert fields["camera_height_status"] == depthlib.NO_DEPTH


def test_pano_block_always_carries_the_height_keys():
    pano = gsv.build_pano_record("PID", 44.05, -121.31, make_metadata())
    assert pano["camera_height_status"] == depthlib.NO_DEPTH
    for key in ("camera_height_m", "camera_height_spread_m", "ground_tilt_deg",
                "depth_planes"):
        assert pano[key] is None
