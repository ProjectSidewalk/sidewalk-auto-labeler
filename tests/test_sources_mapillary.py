"""Unit tests for the Mapillary imagery source: tile geometry, coverage decoding
(round-tripped through a real MVT encode), and the fetch_pano record contract."""
import math

import mapbox_vector_tile
import pytest
from shapely.geometry import Point

import main
from sources import mapillary


# A plausible Graph API response for a spherical image (downtown Richmond, VA).
def make_meta(**overrides):
    base = {
        "camera_type": "spherical",
        "captured_at": 1687000000000,  # 2023-06-17 UTC
        "computed_geometry": {"type": "Point", "coordinates": [-77.4360, 37.5407]},
        "computed_compass_angle": 123.4,
        "compass_angle": 120.0,
        "thumb_original_url": "https://example.test/signed.jpg",
        "width": 5760,
        "height": 2880,
        "creator": {"username": "rva-rider", "id": "42"},
        "sequence": "seq-1",
        "quality_score": 0.8,
    }
    base.update(overrides)
    # Graph API omits absent fields entirely rather than sending nulls.
    return {k: v for k, v in base.items() if v is not None}


def test_tile_point_to_lonlat_inverts_latlon_to_tile():
    lat, lon = 37.5407, -77.4360
    zoom = mapillary.COVERAGE_TILE_ZOOM
    n = 2 ** zoom
    x_float = (lon + 180.0) / 360.0 * n
    y_float = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    tile_x, tile_y = int(x_float), int(y_float)
    assert (tile_x, tile_y) == main.latlon_to_tile(lat, lon, zoom)
    lon2, lat2 = mapillary.tile_point_to_lonlat(
        (x_float - tile_x) * 4096, (y_float - tile_y) * 4096, tile_x, tile_y, zoom, 4096)
    assert lon2 == pytest.approx(lon, abs=1e-6)
    assert lat2 == pytest.approx(lat, abs=1e-6)


def _encode_image_layer(features):
    layer = {"name": "image", "features": features}
    # The options API changed in mapbox-vector-tile 2.0; support both.
    try:
        return mapbox_vector_tile.encode(
            [layer], default_options={"y_coord_down": True, "extents": 4096})
    except TypeError:
        return mapbox_vector_tile.encode([layer], y_coord_down=True, extents=4096)


def test_panos_from_tile_filters_pano_flag_and_area():
    tile_x, tile_y = main.latlon_to_tile(37.5407, -77.4360, mapillary.COVERAGE_TILE_ZOOM)
    inside_lon, inside_lat = mapillary.tile_point_to_lonlat(
        1000, 1000, tile_x, tile_y, mapillary.COVERAGE_TILE_ZOOM, 4096)
    area = Point(inside_lon, inside_lat).buffer(0.0005)

    tile_bytes = _encode_image_layer([
        {"geometry": "POINT(1000 1000)", "properties": {"id": 111, "is_pano": True}},
        {"geometry": "POINT(1000 1000)", "properties": {"id": 222, "is_pano": False}},
        {"geometry": "POINT(3900 3900)", "properties": {"id": 333, "is_pano": True}},  # outside area
    ])

    panos = mapillary.panos_from_tile(tile_bytes, tile_x, tile_y, area)
    assert list(panos) == ["111"]  # IDs are strings — PS pano IDs are strings
    lat, lon = panos["111"]
    assert lat == pytest.approx(inside_lat, abs=1e-5)
    assert lon == pytest.approx(inside_lon, abs=1e-5)


def _patch_fetch(monkeypatch, meta, image="IMAGE"):
    monkeypatch.setattr(mapillary, "_fetch_image_metadata", lambda image_id: meta)
    monkeypatch.setattr(mapillary, "_download_image", lambda url: image)


def test_fetch_pano_success_record_contract(monkeypatch):
    _patch_fetch(monkeypatch, make_meta())
    result = mapillary.fetch_pano("123456", 0.0, 0.0)
    assert result["status"] == "success"
    assert result["image"] == "IMAGE"
    pano = result["pano"]
    assert pano["panorama_id"] == "123456"
    assert pano["source"] == "mapillary"
    assert pano["capture_date"] == "2023-06"
    assert (pano["width"], pano["height"]) == (5760, 2880)
    # SfM-computed values beat the tile position and EXIF compass.
    assert (pano["lat"], pano["lng"]) == (37.5407, -77.4360)
    assert pano["camera_heading"] == pytest.approx(123.4)
    # Pitch/roll only exist inside computed_rotation (axis-angle); left null.
    assert pano["camera_pitch"] is None and pano["camera_roll"] is None
    assert pano["history"] == [] and pano["links"] == []
    assert "rva-rider" in pano["copyright"]


def test_fetch_pano_falls_back_to_exif_compass_and_tile_position(monkeypatch):
    _patch_fetch(monkeypatch, make_meta(computed_compass_angle=None, computed_geometry=None,
                                        geometry=None))
    pano = mapillary.fetch_pano("123456", 37.5, -77.4)["pano"]
    assert pano["camera_heading"] == pytest.approx(120.0)
    assert (pano["lat"], pano["lng"]) == (37.5, -77.4)


@pytest.mark.parametrize("broken, reason_fragment", [
    ({"camera_type": "perspective"}, "camera_type"),
    ({"captured_at": None}, "timestamp"),
    ({"thumb_original_url": None}, "thumbnail"),
    ({"width": 5760, "height": 2000}, "equirectangular"),  # cropped vertical FOV
    ({"computed_compass_angle": None, "compass_angle": None}, "compass"),
])
def test_fetch_pano_deterministic_skips(monkeypatch, broken, reason_fragment):
    _patch_fetch(monkeypatch, make_meta(**broken))
    result = mapillary.fetch_pano("123456", 0.0, 0.0)
    assert result["status"] == "skipped"
    assert reason_fragment in result["reason"]


def test_fetch_pano_metadata_unavailable_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, None)
    assert mapillary.fetch_pano("123456", 0.0, 0.0)["status"] == "failure"


def test_fetch_pano_download_failure_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, make_meta(), image=None)
    assert mapillary.fetch_pano("123456", 0.0, 0.0)["status"] == "failure"
