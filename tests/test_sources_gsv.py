"""Unit tests for the GSV imagery source: the pano record shape (what PS ultimately
consumes) and fetch_pano's skip/failure semantics (what the resume cache depends on)."""
import pytest

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
