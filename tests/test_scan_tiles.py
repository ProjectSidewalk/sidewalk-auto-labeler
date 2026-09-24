"""Unit tests for main.py's coverage-tile prefilter and scan cache (issue #4): which
tiles a scan requests, and when a saved scan may stand in for a fresh one. No network —
the one end-to-end test drives run_labeler with a fake source in --scan-only mode."""
import json
import random
from types import SimpleNamespace

import pytest
from shapely.geometry import Point, Polygon, mapping

import main

ZOOM = 14


def _l_shape():
    """An L over a 4x4-tile patch in Virginia: the whole west column plus the south row
    (7 tiles), leaving the 3x3 north-east corner of the bbox empty."""
    x0, y0 = 4600, 6300                               # north-west tile of the patch
    west, _, _, north = main.tile_lonlat_bounds(x0, y0, ZOOM)
    _, south, east, _ = main.tile_lonlat_bounds(x0 + 3, y0 + 3, ZOOM)
    col_east = main.tile_lonlat_bounds(x0, y0, ZOOM)[2]
    row_north = main.tile_lonlat_bounds(x0, y0 + 3, ZOOM)[3]
    # Inset by a hair so the L doesn't touch the neighbouring tiles' edges.
    e = 1e-6
    return Polygon([(west + e, north - e), (col_east - e, north - e), (col_east - e, row_north - e),
                    (east - e, row_north - e), (east - e, south + e), (west + e, south + e)])


def test_tile_lonlat_bounds_inverts_latlon_to_tile():
    lat, lon = 37.5407, -77.4360
    x, y = main.latlon_to_tile(lat, lon, 17)
    west, south, east, north = main.tile_lonlat_bounds(x, y, 17)
    assert west <= lon < east and south < lat <= north


def test_tile_prefilter_drops_the_empty_corner_of_an_l():
    tiles, bbox_count = main.tiles_intersecting(_l_shape(), ZOOM)
    assert bbox_count == 16
    assert len(tiles) == 7                            # 4 in the column + 3 more in the row
    assert (4603, 6300) not in tiles and (4601, 6301) not in tiles
    assert (4600, 6300) in tiles and (4603, 6303) in tiles


def test_every_in_area_pano_keeps_its_tile():
    # The correctness argument, checked: a point inside the area is always in a kept tile.
    area = _l_shape().union(Point(-77.43, 37.54).buffer(0.01))   # concave AND multi-part
    tiles = set(main.tiles_intersecting(area, ZOOM)[0])
    rng = random.Random(4)
    w, s, e, n = area.bounds
    hits = 0
    while hits < 500:
        lon, lat = rng.uniform(w, e), rng.uniform(s, n)
        if area.contains(Point(lon, lat)):
            assert main.latlon_to_tile(lat, lon, ZOOM) in tiles
            hits += 1


def _cache(tmp_path, failed=0, **overrides):
    args = dict(area_hash="abc", source_name="mapillary", zoom=14)
    args.update(overrides)
    path = tmp_path / main.SCAN_CACHE_FILE
    main.write_scan_cache(path, args["area_hash"], args["source_name"], args["zoom"],
                          tile_count=7, failed_tiles=failed,
                          panos={"111": (37.5, -77.4, 1687000000000, 0.8)},
                          scanned_at="2026-09-20T00:00:00+00:00")
    return path


def test_scan_cache_round_trips_the_scan_values(tmp_path):
    panos, cache, reason = main.load_scan_cache(_cache(tmp_path), "abc", "mapillary", 14)
    assert reason is None
    assert panos == {"111": (37.5, -77.4, 1687000000000, 0.8)}   # tuples, as the scan gives
    assert cache["tile_count"] == 7 and cache["scanned_at"].startswith("2026-09-20")


@pytest.mark.parametrize("load_args, failed, fragment", [
    (("other", "mapillary", 14), 0, "area geometry"),
    (("abc", "gsv", 14), 0, "imagery source"),
    (("abc", "mapillary", 17), 0, "tile zoom"),
    (("abc", "mapillary", 14), 3, "3 failed tiles"),
])
def test_scan_cache_refuses_a_mismatch(tmp_path, load_args, failed, fragment):
    path = _cache(tmp_path, failed=failed)
    panos, cache, reason = main.load_scan_cache(path, *load_args)
    assert panos is None and fragment in reason


def test_scan_cache_missing_or_corrupt_is_refused_not_raised(tmp_path):
    assert main.load_scan_cache(tmp_path / "nope.json", "abc", "gsv", 17)[2] == "no scan cache yet"
    bad = tmp_path / "bad.json"
    bad.write_text("{truncated")
    assert "unreadable" in main.load_scan_cache(bad, "abc", "gsv", 17)[2]


def _fake_source(calls):
    def fetch_panos_for_tile(x, y, area_shape):
        calls.append((x, y))
        west, south, east, north = main.tile_lonlat_bounds(x, y, ZOOM)
        return {f"{x}-{y}": ((south + north) / 2, (west + east) / 2)}
    return SimpleNamespace(NAME="fake", COVERAGE_TILE_ZOOM=ZOOM,
                           fetch_panos_for_tile=fetch_panos_for_tile)


def test_scan_only_scans_intersecting_tiles_and_reuse_skips_the_tile_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    area_file = tmp_path / "l.geojson"
    area_file.write_text(json.dumps(mapping(_l_shape())))
    calls = []
    main.run_labeler(str(area_file), "l", _fake_source(calls), scan_only=True)
    assert len(calls) == 7                            # the empty corner is never requested
    assert (tmp_path / "runs" / "l" / main.SCAN_CACHE_FILE).exists()

    # Default: a resume rescans even though the cache is there.
    main.run_labeler(str(area_file), "l", _fake_source(calls), scan_only=True)
    assert len(calls) == 14
    # --reuse-scan: no tile requests at all.
    main.run_labeler(str(area_file), "l", _fake_source(calls), scan_only=True, reuse_scan=True)
    assert len(calls) == 14


def test_record_run_carries_the_scan_provenance(tmp_path):
    manifest = {"runs": []}
    main.record_run(tmp_path / "m.json", manifest, "t0", 5, 5, 0, 0,
                    scan={"scan": "reused", "scan_scanned_at": "t", "scan_age_hours": 3.5})
    assert manifest["runs"][0]["scan"] == "reused"
    assert manifest["runs"][0]["scan_age_hours"] == 3.5
