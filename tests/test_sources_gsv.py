"""Unit tests for the GSV imagery source: the pano record shape (what PS ultimately
consumes) and fetch_pano's skip/failure semantics (what the resume cache depends on)."""
import pytest
from shapely.geometry import box

from conftest import make_metadata
from sources import gsv


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


def _patch_fetch(monkeypatch, metadata, image="IMAGE"):
    monkeypatch.setattr(gsv, "fetch_metadata_with_retry", lambda pano_id: metadata)
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
                        lambda pano_id: make_metadata(lat=40.0, lon=-121.31))
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
