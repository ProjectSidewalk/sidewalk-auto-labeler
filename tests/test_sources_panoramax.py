"""Unit tests for the Panoramax imagery source: coverage decoding (round-tripped through
a real MVT encode), thinning, and the fetch_pano record contract against a STAC item."""
import copy
from types import SimpleNamespace

import mapbox_vector_tile
import pytest
from shapely.geometry import Point

import main
from sources import panoramax


# A plausible STAC item for a GoPro Max picture in Bayonne, trimmed to what the source
# reads (the real item also carries ~60 EXIF keys and the viewer's tile matrix).
def make_item(**overrides):
    item = {
        "id": "4ac3b7f9-89ee-4285-9ec3-132002a28003",
        "type": "Feature",
        "collection": "seq-1",
        "geometry": {"type": "Point", "coordinates": [-1.4748, 43.4929]},
        "providers": [{"id": "p1", "name": "Arretche", "roles": ["producer"]}],
        "assets": {
            "hd": {"href": "https://panoramax.openstreetmap.fr/images/4a/c3/b7f9.jpg",
                   "type": "image/jpeg"},
            "sd": {"href": "https://panoramax.openstreetmap.fr/derivates/4a/sd.jpg"},
        },
        "properties": {
            "datetime": "2025-05-03T11:17:42+00:00",
            "license": "CC-BY-SA-4.0",
            "view:azimuth": 112,
            "pers:pitch": 0.0,
            "pers:roll": 0.0,
            "pers:interior_orientation": {
                "camera_manufacturer": "GoPro", "camera_model": "Max",
                "field_of_view": 360, "focal_length": 3.0,
                "sensor_array_dimensions": [5760, 2880],
            },
            "geovisio:producer": "Arretche",
            "geovisio:image": "https://panoramax.openstreetmap.fr/images/4a/c3/b7f9.jpg",
            "exif": {"Exif.Image.Model": "GoPro Max", "Xmp.GPano.ProjectionType": "equirectangular"},
            "tiles:tile_matrix_sets": {"geovisio": {"bulky": True}},
        },
    }
    item = copy.deepcopy(item)
    for key, value in overrides.items():
        # "properties.view:azimuth": None  ->  set (or delete, when None) a nested key
        target, _, leaf = key.rpartition(".")
        node = item
        for part in target.split(".") if target else []:
            node = node[part]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return item


def _encode_pictures_layer(features):
    layer = {"name": "pictures", "features": features}
    try:
        return mapbox_vector_tile.encode(
            [layer], default_options={"y_coord_down": True, "extents": 4096})
    except TypeError:
        return mapbox_vector_tile.encode([layer], y_coord_down=True, extents=4096)


def test_panos_from_tile_keeps_equirectangular_pictures_in_area():
    zoom = panoramax.COVERAGE_TILE_ZOOM
    tile_x, tile_y = main.latlon_to_tile(43.4929, -1.4748, zoom)
    inside_lon, inside_lat = panoramax.tile_point_to_lonlat(1000, 1000, tile_x, tile_y, zoom, 4096)
    area = Point(inside_lon, inside_lat).buffer(0.0005)

    tile_bytes = _encode_pictures_layer([
        {"geometry": "POINT(1000 1000)",
         "properties": {"id": "aaa", "type": "equirectangular", "ts": "2025-05-03 11:17:42+00",
                        "h_pixel_density": 16, "heading": 112}},
        {"geometry": "POINT(1000 1000)", "properties": {"id": "bbb", "type": "flat"}},
        {"geometry": "POINT(1000 1000)", "properties": {"id": "ccc"}},          # no type
        {"geometry": "POINT(3900 3900)",
         "properties": {"id": "ddd", "type": "equirectangular"}},               # outside area
    ])

    panos = panoramax.panos_from_tile(tile_bytes, tile_x, tile_y, area)
    assert list(panos) == ["aaa"]
    lat, lon, captured_at, density = panos["aaa"]
    assert lat == pytest.approx(inside_lat, abs=1e-5)
    assert lon == pytest.approx(inside_lon, abs=1e-5)
    assert captured_at == "2025-05-03 11:17:42+00"
    assert density == 16


def test_panos_from_tile_without_a_pictures_layer_is_empty():
    # z14 tiles carry only the `sequences` layer; a sequence-only payload must not crash.
    layer = {"name": "sequences", "features": [
        {"geometry": "LINESTRING(0 0, 10 10)", "properties": {"id": "s", "type": "equirectangular"}}]}
    try:
        tile_bytes = mapbox_vector_tile.encode([layer], default_options={"y_coord_down": True})
    except TypeError:
        tile_bytes = mapbox_vector_tile.encode([layer], y_coord_down=True)
    assert panoramax.panos_from_tile(tile_bytes, 0, 0, Point(0, 0).buffer(1)) == {}


def test_thin_panos_newest_capture_wins_then_pixel_density():
    panos = {
        "old": (43.4929, -1.4748, "2023-06-01 10:00:00+00", 30),
        "new": (43.4929, -1.4748, "2025-05-03 11:17:42+00", 16),   # newer wins despite lower density
        "far": (43.4939, -1.4748, "2020-01-01 00:00:00+00", 16),   # ~110 m away: its own cell
        "sharp": (43.4949, -1.4748, "2025-05-03 11:17:42+00", 33),
        "soft": (43.4949, -1.4748, "2025-05-03 11:17:42+00", 16),  # same instant: density breaks the tie
    }
    assert set(panoramax.thin_panos(panos)) == {"new", "far", "sharp"}
    assert panoramax.thin_panos(panos)["far"][:2] == (43.4939, -1.4748)
    # One cell for everything: the newest instant wins and density breaks its tie.
    assert set(panoramax.thin_panos(panos, 100_000)) == {"sharp"}


def _patch_fetch(monkeypatch, item, image=("IMAGE", (5760, 2880)), gone=False,
                 undecodable=False):
    monkeypatch.setattr(panoramax, "fetch_item", lambda picture_id: (item, gone))
    monkeypatch.setattr(panoramax, "_download_image", lambda url: (image, undecodable))


def test_fetch_pano_success_record_contract(monkeypatch):
    _patch_fetch(monkeypatch, make_item())
    result = panoramax.fetch_pano("4ac3b7f9", 0.0, 0.0)
    assert result["status"] == "success"
    assert result["image"] == "IMAGE"
    pano = result["pano"]
    assert pano["panorama_id"] == "4ac3b7f9"
    assert pano["source"] == "panoramax"
    assert pano["capture_date"] == "2025-05"
    assert (pano["width"], pano["height"]) == (5760, 2880)
    # The item's own position beats the tile position passed in.
    assert (pano["lat"], pano["lng"]) == (43.4929, -1.4748)
    assert pano["camera_heading"] == pytest.approx(112.0)
    assert pano["camera_pitch"] == 0.0 and pano["camera_roll"] == 0.0
    assert pano["history"] == [] and pano["links"] == []
    assert pano["copyright"] == "© Arretche / Panoramax (CC-BY-SA-4.0)"
    assert pano["license"] == "CC-BY-SA-4.0"
    assert pano["sequence_id"] == "seq-1"
    # Camera hardware provenance + which federation member holds the picture.
    assert (pano["camera_make"], pano["camera_model"]) == ("GoPro", "Max")
    assert pano["camera_type"] == "equirectangular"
    assert pano["panoramax_instance"] == "panoramax.openstreetmap.fr"
    sm = pano["source_metadata"]
    assert sm["view:azimuth"] == 112 and sm["exif"]["Exif.Image.Model"] == "GoPro Max"
    assert sm["collection"] == "seq-1" and sm["providers"][0]["name"] == "Arretche"
    assert sm["hd_url"].endswith("b7f9.jpg")
    assert "tiles:tile_matrix_sets" not in sm


def test_fetch_pano_records_the_downloaded_dimensions(monkeypatch):
    # The record's width/height are what PS scales normalized detections by, so they
    # must describe the pixels we actually ran on, not what the metadata declared.
    _patch_fetch(monkeypatch, make_item(), image=("IMAGE", (11008, 5504)))
    pano = panoramax.fetch_pano("x", 0.0, 0.0)["pano"]
    assert (pano["width"], pano["height"]) == (11008, 5504)


def test_fetch_pano_falls_back_to_tile_position_and_provider_name(monkeypatch):
    _patch_fetch(monkeypatch, make_item(**{"geometry": None, "properties.geovisio:producer": None}))
    pano = panoramax.fetch_pano("x", 43.5, -1.47)["pano"]
    assert (pano["lat"], pano["lng"]) == (43.5, -1.47)
    assert pano["copyright"].startswith("© Arretche / Panoramax")


def test_fetch_pano_uses_geovisio_image_when_the_hd_asset_is_missing(monkeypatch):
    seen = []
    monkeypatch.setattr(panoramax, "fetch_item", lambda pid: (make_item(**{"assets.hd": None}), False))
    monkeypatch.setattr(panoramax, "_download_image",
                        lambda url: seen.append(url) or (("IMAGE", (5760, 2880)), False))
    assert panoramax.fetch_pano("x", 0.0, 0.0)["status"] == "success"
    assert seen == ["https://panoramax.openstreetmap.fr/images/4a/c3/b7f9.jpg"]


def test_fetch_pano_accepts_a_picture_whose_item_omits_field_of_view(monkeypatch):
    # The tile scan already certified `type == equirectangular`, and a skip here is cached
    # forever — so an absent field_of_view must not permanently drop a valid picture.
    # (`pers:interior_orientation` is optional in STAC, and so is field_of_view within it.)
    for broken in ({"properties.pers:interior_orientation.field_of_view": None},
                   {"properties.pers:interior_orientation": None}):
        _patch_fetch(monkeypatch, make_item(**broken))
        assert panoramax.fetch_pano("x", 0.0, 0.0)["status"] == "success"


@pytest.mark.parametrize("broken, reason_fragment", [
    ({"properties.pers:interior_orientation.field_of_view": 100}, "field_of_view"),
    ({"properties.datetime": None}, "timestamp"),
    ({"properties.view:azimuth": None}, "heading"),
    ({"assets.hd": None, "properties.geovisio:image": None}, "hd image"),
    ({"properties.pers:interior_orientation.sensor_array_dimensions": [5760, 2000]}, "equirectangular"),
])
def test_fetch_pano_deterministic_skips(monkeypatch, broken, reason_fragment):
    _patch_fetch(monkeypatch, make_item(**broken))
    result = panoramax.fetch_pano("x", 0.0, 0.0)
    assert result["status"] == "skipped"
    assert reason_fragment in result["reason"]


def test_fetch_pano_skips_a_download_that_is_not_2_to_1(monkeypatch):
    _patch_fetch(monkeypatch, make_item(), image=("IMAGE", (5760, 2000)))
    result = panoramax.fetch_pano("x", 0.0, 0.0)
    assert result["status"] == "skipped" and "5760x2000" in result["reason"]


def test_fetch_pano_deleted_picture_is_skipped_not_retried(monkeypatch):
    _patch_fetch(monkeypatch, None, gone=True)
    result = panoramax.fetch_pano("x", 0.0, 0.0)
    assert result["status"] == "skipped" and "no longer exists" in result["reason"]


def test_fetch_pano_transient_metadata_failure_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, None, gone=False)
    assert panoramax.fetch_pano("x", 0.0, 0.0)["status"] == "failure"


def test_fetch_pano_download_failure_is_retryable(monkeypatch):
    _patch_fetch(monkeypatch, make_item(), image=None)
    assert panoramax.fetch_pano("x", 0.0, 0.0)["status"] == "failure"


def test_fetch_pano_undecodable_asset_is_skipped_not_retried(monkeypatch):
    # Bytes that arrived and are not a readable image (or a 404 on the unsigned hd URL)
    # will not become one on a later run — caching the skip is what stops the area
    # re-downloading the same dead megabytes forever.
    _patch_fetch(monkeypatch, make_item(), image=None, undecodable=True)
    result = panoramax.fetch_pano("x", 0.0, 0.0)
    assert result["status"] == "skipped" and "decodable" in result["reason"]


def test_download_image_separates_decode_failure_from_network_failure(monkeypatch):
    calls = []

    def answer(status, body):
        def raise_for_status():
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")
        return SimpleNamespace(status_code=status, content=body,
                               raise_for_status=raise_for_status)

    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)

    # Not an image: one request, permanent.
    monkeypatch.setattr(panoramax.requests, "get",
                        lambda url, **kw: calls.append(url) or answer(200, b"not a jpeg"))
    assert panoramax._download_image("u") == (None, True)
    assert len(calls) == 1

    # A 404 on the plain, unsigned asset URL is equally permanent.
    calls.clear()
    monkeypatch.setattr(panoramax.requests, "get",
                        lambda url, **kw: calls.append(url) or answer(404, b""))
    assert panoramax._download_image("u") == (None, True)
    assert len(calls) == 1

    # A network error is not: retried, then reported as retryable.
    calls.clear()

    def boom(url, **kw):
        calls.append(url)
        raise OSError("connection reset")

    monkeypatch.setattr(panoramax.requests, "get", boom)
    assert panoramax._download_image("u") == (None, False)
    assert len(calls) == panoramax.ATTEMPTS


def test_fetch_item_404_is_gone_without_retrying(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("headers", {}).get("User-Agent")))
        return SimpleNamespace(status_code=404, json=lambda: {}, raise_for_status=lambda: None)

    monkeypatch.setattr(panoramax.requests, "get", fake_get)
    assert panoramax.fetch_item("abc") == (None, True)
    assert len(calls) == 1
    assert calls[0][0] == "https://api.panoramax.xyz/api/pictures/abc"
    assert calls[0][1] == panoramax.USER_AGENT     # the API asks tools to identify themselves


def test_fetch_item_transient_error_exhausts_retries(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        raise OSError("connection reset")

    monkeypatch.setattr(panoramax.requests, "get", fake_get)
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)
    assert panoramax.fetch_item("abc") == (None, False)
    assert len(calls) == panoramax.ATTEMPTS


def test_fetch_panos_for_tile_treats_204_as_no_coverage(monkeypatch):
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: SimpleNamespace(
        status_code=204, content=b"", raise_for_status=lambda: None))
    assert panoramax.fetch_panos_for_tile(1, 2, Point(0, 0)) == {}


def _tile_response(status, content=b"\x1a\x00"):
    def raise_for_status():
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
    return SimpleNamespace(status_code=status, content=content,
                           raise_for_status=raise_for_status)


def test_fetch_panos_for_tile_reports_a_404_as_a_failed_tile(monkeypatch):
    # Measured live: an empty tile answers 204, and a 404 means the API root is wrong.
    # Reading 404 as "no coverage" would turn a mistyped PANORAMAX_API_URL into a
    # silent zero-pano run instead of a scan that says every tile failed.
    calls = []
    monkeypatch.setattr(panoramax.requests, "get",
                        lambda url, **kw: calls.append(url) or _tile_response(404, b""))
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)
    assert panoramax.fetch_panos_for_tile(1, 2, Point(0, 0)) is None
    assert len(calls) == panoramax.ATTEMPTS


def test_fetch_panos_for_tile_reports_an_empty_bodied_5xx_as_a_failed_tile(monkeypatch):
    # A 502 with no body is a failure to retry, not a tile without pictures — so the
    # empty-body shortcut must not run before raise_for_status().
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: _tile_response(502, b""))
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)
    assert panoramax.fetch_panos_for_tile(1, 2, Point(0, 0)) is None


def test_prepare_accepts_a_stac_catalog_and_rejects_a_bad_root(monkeypatch):
    def answer(payload, status=200):
        def raise_for_status():
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")
        return SimpleNamespace(status_code=status, raise_for_status=raise_for_status,
                               json=lambda: payload)

    seen = []
    monkeypatch.setattr(panoramax.requests, "get",
                        lambda url, **kw: seen.append(url) or answer({"stac_version": "1.1.0"}))
    panoramax.prepare()                                  # no raise
    assert seen == [panoramax.DEFAULT_API_URL]

    # A mistyped root 404s; prepare must fail before the model loads, not at scan time.
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: answer({}, status=404))
    with pytest.raises(SystemExit):
        panoramax.prepare()
    # ...and so must a root that answers but isn't a STAC catalog.
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: answer({"hello": "world"}))
    with pytest.raises(SystemExit):
        panoramax.prepare()


def test_api_url_env_override_points_at_a_single_instance(monkeypatch):
    monkeypatch.setenv(panoramax.API_URL_ENV_VAR, "https://pano.locus.sbs/api/")
    seen = []
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: seen.append(url) or SimpleNamespace(
        status_code=404, json=lambda: {}, raise_for_status=lambda: None))
    panoramax.fetch_item("abc")
    assert seen == ["https://pano.locus.sbs/api/pictures/abc"]
