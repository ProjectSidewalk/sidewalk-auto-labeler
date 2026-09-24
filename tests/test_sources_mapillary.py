"""Unit tests for the Mapillary imagery source: tile geometry, coverage decoding
(round-tripped through a real MVT encode), and the fetch_pano record contract."""
import math
from types import SimpleNamespace

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
        "make": "GoPro",
        "model": "Fusion",
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
        {"geometry": "POINT(1000 1000)",
         "properties": {"id": 111, "is_pano": True, "captured_at": 1687000000000, "quality_score": 0.8}},
        {"geometry": "POINT(1000 1000)", "properties": {"id": 222, "is_pano": False}},
        {"geometry": "POINT(3900 3900)", "properties": {"id": 333, "is_pano": True}},  # outside area
    ])

    panos = mapillary.panos_from_tile(tile_bytes, tile_x, tile_y, area)
    assert list(panos) == ["111"]  # IDs are strings — PS pano IDs are strings
    lat, lon, captured_at, quality = panos["111"]
    assert lat == pytest.approx(inside_lat, abs=1e-5)
    assert lon == pytest.approx(inside_lon, abs=1e-5)
    assert captured_at == 1687000000000
    assert quality == pytest.approx(0.8)


def test_thin_panos_newest_per_cell_wins():
    panos = {
        "old": (37.5400, -77.4360, 100, 0.9),
        "new": (37.5400, -77.4360, 200, 0.4),   # same spot: newer wins despite lower quality
        "far": (37.5410, -77.4360, 50, 0.5),    # ~110 m away: its own cell survives
    }
    assert set(mapillary.thin_panos(panos)) == {"new", "far"}
    # Surviving values keep the (lat, lon, ...) shape main.py consumes.
    assert mapillary.thin_panos(panos)["far"][:2] == (37.5410, -77.4360)


def test_thin_panos_quality_breaks_capture_time_ties():
    panos = {
        "meh": (37.5400, -77.4360, 100, 0.4),
        "good": (37.5400, -77.4360, 100, 0.9),
    }
    assert set(mapillary.thin_panos(panos)) == {"good"}


def test_thin_panos_spacing_is_tunable():
    panos = {
        "a": (37.5400, -77.4360, 100, 0.5),
        "b": (37.5410, -77.4360, 200, 0.5),  # ~110 m apart
    }
    assert set(mapillary.thin_panos(panos)) == {"a", "b"}      # separate 5 m cells
    # A cell size that dwarfs the 110 m separation (so no cell-boundary luck):
    # both collapse into one cell and the newest wins.
    assert set(mapillary.thin_panos(panos, 100_000)) == {"b"}


def _patch_fetch(monkeypatch, meta, image="IMAGE", gone=False, undecodable=False):
    monkeypatch.setattr(mapillary, "fetch_image_metadata", lambda image_id: (meta, gone))
    monkeypatch.setattr(mapillary, "_download_image", lambda url: (image, undecodable))


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
    # PS's pano_data.copyright holds the contributor's BARE name for this source — it
    # composes the ©, the provider and the licence itself (SidewalkWebpage#5360).
    assert pano["copyright"] == "rva-rider"
    assert pano["license"] == "CC-BY-SA-4.0"
    # Camera hardware provenance for post-hoc image-quality analysis.
    assert (pano["camera_make"], pano["camera_model"]) == ("GoPro", "Fusion")
    assert pano["camera_type"] == "spherical"
    # Full stable Graph metadata kept verbatim, minus the volatile signed thumb URL.
    sm = pano["source_metadata"]
    assert sm["quality_score"] == 0.8 and sm["make"] == "GoPro"
    assert sm["creator"] == {"username": "rva-rider", "id": "42"}
    assert "thumb_original_url" not in sm


@pytest.mark.parametrize("creator", [
    None,                              # Graph API omits the field entirely
    {"id": "42"},                      # ...or returns the object without a username
    {"username": "", "id": "42"},      # ...or a blank one (deleted/renamed account)
    {"username": "   ", "id": "42"},
])
def test_fetch_pano_copyright_is_none_without_a_creator(monkeypatch, creator):
    # No name to credit: leave it null rather than storing a blank credit, so PS credits
    # Mapillary itself. The licence stays in its own field either way.
    _patch_fetch(monkeypatch, make_meta(creator=creator))
    pano = mapillary.fetch_pano("123456", 0.0, 0.0)["pano"]
    assert pano["copyright"] is None
    assert pano["license"] == "CC-BY-SA-4.0"


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


def test_fetch_pano_deleted_image_is_skipped_not_retried(monkeypatch):
    """The API saying "no such image" is deterministic, so it must be cached as skipped
    rather than retried on every future run of the area."""
    _patch_fetch(monkeypatch, None, gone=True)
    result = mapillary.fetch_pano("123456", 0.0, 0.0)
    assert result["status"] == "skipped" and "no longer exists" in result["reason"]


def test_fetch_pano_transient_metadata_failure_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, None, gone=False)
    assert mapillary.fetch_pano("123456", 0.0, 0.0)["status"] == "failure"


@pytest.mark.parametrize("status, expected", [
    (404, (None, True)),        # deleted image: never retried
    (400, (None, True)),        # invalid/withdrawn id: same
])
def test_fetch_image_metadata_reports_gone_without_retrying(monkeypatch, status, expected):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=status, json=lambda: {},
                               raise_for_status=lambda: None)

    monkeypatch.setenv(mapillary.TOKEN_ENV_VAR, "tok")
    monkeypatch.setattr(mapillary.requests, "get", fake_get)
    assert mapillary.fetch_image_metadata("123456") == expected
    assert len(calls) == 1                       # one call, not ATTEMPTS


def test_fetch_image_metadata_transient_error_exhausts_retries(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        raise OSError("connection reset")

    monkeypatch.setenv(mapillary.TOKEN_ENV_VAR, "tok")
    monkeypatch.setattr(mapillary.requests, "get", fake_get)
    monkeypatch.setattr(mapillary.time, "sleep", lambda s: None)
    assert mapillary.fetch_image_metadata("123456") == (None, False)
    assert len(calls) == mapillary.ATTEMPTS


def test_fetch_pano_metadata_unavailable_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, None)
    assert mapillary.fetch_pano("123456", 0.0, 0.0)["status"] == "failure"


def test_fetch_pano_download_failure_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, make_meta(), image=None)
    assert mapillary.fetch_pano("123456", 0.0, 0.0)["status"] == "failure"


def test_fetch_pano_undecodable_image_is_skipped_not_retried(monkeypatch):
    # main.py caches a skip, so the same unreadable bytes are never re-downloaded.
    _patch_fetch(monkeypatch, make_meta(), image=None, undecodable=True)
    result = mapillary.fetch_pano("123456", 0.0, 0.0)
    assert result == {"status": "skipped", "reason": "Undecodable image bytes"}


def _fake_image_get(monkeypatch, respond):
    """Route mapillary.requests.get through `respond(call_number)`; returns the call log."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return respond(len(calls))

    monkeypatch.setattr(mapillary.requests, "get", fake_get)
    monkeypatch.setattr(mapillary.time, "sleep", lambda s: None)
    return calls


def _response(status, content=b"", content_type="image/jpeg"):
    def raise_for_status():
        if status >= 400:
            raise mapillary.requests.HTTPError(f"{status}")
    return SimpleNamespace(status_code=status, content=content,
                           headers={"Content-Type": content_type},
                           raise_for_status=raise_for_status)


def _jpeg_bytes():
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (200, 100)).save(buf, format="JPEG")
    return buf.getvalue()


def test_download_image_non_image_bytes_are_permanent_after_one_request(monkeypatch):
    calls = _fake_image_get(monkeypatch, lambda n: _response(200, b"<html>not a jpeg</html>"))
    assert mapillary._download_image("https://example.test/signed.jpg") == (None, True)
    assert len(calls) == 1                       # decode failures are never re-looped


def test_download_image_connection_error_is_retryable_after_all_attempts(monkeypatch):
    def respond(n):
        raise OSError("connection reset")
    calls = _fake_image_get(monkeypatch, respond)
    assert mapillary._download_image("https://example.test/signed.jpg") == (None, False)
    assert len(calls) == mapillary.ATTEMPTS


def test_download_image_404_is_retryable_because_the_url_is_signed(monkeypatch):
    # Unlike Panoramax's plain hd URL, a 404 on an expiring signed URL is transient:
    # it must come back as a retryable failure, not a cached skip.
    calls = _fake_image_get(monkeypatch, lambda n: _response(404))
    assert mapillary._download_image("https://example.test/signed.jpg") == (None, False)
    assert len(calls) == mapillary.ATTEMPTS


def test_download_image_success_normalizes_to_detector_size(monkeypatch):
    calls = _fake_image_get(monkeypatch, lambda n: _response(200, _jpeg_bytes()))
    image, permanent = mapillary._download_image("https://example.test/signed.jpg")
    assert permanent is False
    assert image.size == mapillary.TARGET_SIZE
    assert len(calls) == 1


def test_download_image_memory_error_during_decode_stays_retryable(monkeypatch):
    # Only "these bytes are not an image" errors are permanent. A MemoryError under load
    # is about this process, so it must never become a cached skip.
    calls = _fake_image_get(monkeypatch, lambda n: _response(200, _jpeg_bytes()))

    def oom(*args, **kwargs):
        raise MemoryError()
    monkeypatch.setattr(mapillary.Image, "open", oom)
    assert mapillary._download_image("https://example.test/signed.jpg") == (None, False)
    assert len(calls) == mapillary.ATTEMPTS


def test_download_image_non_image_content_type_is_retryable(monkeypatch):
    # An HTML error page served with a 200 says nothing about the image itself.
    calls = _fake_image_get(
        monkeypatch, lambda n: _response(200, b"<html>busy</html>", content_type="text/html"))
    assert mapillary._download_image("https://example.test/signed.jpg") == (None, False)
    assert len(calls) == mapillary.ATTEMPTS


def test_download_image_network_error_then_bad_bytes_is_permanent_after_two(monkeypatch):
    # A transient failure is retried; once bytes arrive and do not decode, the loop stops.
    def respond(n):
        if n == 1:
            raise OSError("connection reset")
        return _response(200, b"not a jpeg")
    calls = _fake_image_get(monkeypatch, respond)
    assert mapillary._download_image("https://example.test/signed.jpg") == (None, True)
    assert len(calls) == 2
