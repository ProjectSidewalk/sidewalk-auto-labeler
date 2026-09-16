"""Tests for the pano-position check and the reposition tool (SidewalkWebpage#5361):
offsets against a synthetic street grid, the per-sequence verdict, the position-field
switch in the Mapillary record builder, and the run-dir binding to one field."""
import json

import pytest

import main
import position_check
import reposition
import send_to_ps
from sources import mapillary
from test_sources_mapillary import make_meta

LAT0, LNG0 = 42.85, -94.85


def _frame():
    return position_check.Frame(LAT0, LNG0)


def _grid_osm(frame, spacing_m=100.0, count=3):
    """A count x count grid of named N-S and E-W streets around the frame origin."""
    elements = []
    lo, hi = -spacing_m * (count - 1) / 2, spacing_m * (count - 1) / 2
    for i in range(count):
        c = lo + i * spacing_m
        (lat_a, lng_a), (lat_b, lng_b) = frame.latlng(c, lo), frame.latlng(c, hi)
        elements.append({"type": "way", "tags": {"name": f"NS{i}"},
                         "geometry": [{"lat": lat_a, "lon": lng_a}, {"lat": lat_b, "lon": lng_b}]})
        (lat_a, lng_a), (lat_b, lng_b) = frame.latlng(lo, c), frame.latlng(hi, c)
        elements.append({"type": "way", "tags": {"name": f"EW{i}"},
                         "geometry": [{"lat": lat_a, "lon": lng_a}, {"lat": lat_b, "lon": lng_b}]})
    return {"elements": elements}


def test_measure_point_signs_and_axes():
    frame = _frame()
    index = position_check.StreetIndex(position_check.street_segments(_grid_osm(frame), frame))
    # A cardinal grid: family a is N-S (positive east), family b is E-W (positive north).
    assert index.theta0 == pytest.approx(0.0, abs=1e-6)
    assert index.axes()["a"] == {"streets": "N-S streets", "pos": "east", "neg": "west"}
    # 5 m east of the central N-S street (x=0), far from any E-W street.
    m = position_check.measure_point(index, 5.0, 40.0)
    assert m["street"] == "NS1"
    assert m["a"] == pytest.approx(5.0) and m["b"] is None and m["cross"] == pytest.approx(5.0)
    # 3 m south of the central E-W street (y=0).
    m = position_check.measure_point(index, 40.0, -3.0)
    assert m["street"] == "EW1"
    assert m["b"] == pytest.approx(-3.0) and m["a"] is None
    # Nothing within 30 m of a street: not scored.
    assert position_check.measure_point(index, 1000.0, 1000.0) is None


def _mapillary_record(pano_id, seq, raw_en, sfm_en, frame):
    raw_lat, raw_lng = frame.latlng(*raw_en)
    sfm_lat, sfm_lng = frame.latlng(*sfm_en)
    meta = make_meta(sequence=seq,
                     geometry={"type": "Point", "coordinates": [raw_lng, raw_lat]},
                     computed_geometry={"type": "Point", "coordinates": [sfm_lng, sfm_lat]})
    return {"detections": [], "pano": mapillary.build_pano_record(pano_id, 0.0, 0.0, meta)}


def test_check_run_flags_the_drifted_sequence_and_recommends_raw():
    frame = _frame()
    records = []
    # Sequence A drives north on NS1: raw GPS on the line, SfM 8 m west of it.
    for k in range(25):
        y = -80 + k * 6
        records.append(_mapillary_record(f"a{k}", "A", (0.5, y), (-7.5, y), frame))
    # Sequence B drives east on EW1: both fields within a lane of the line.
    for k in range(25):
        x = -80 + k * 6
        records.append(_mapillary_record(f"b{k}", "B", (x, 1.0), (x, -1.0), frame))
    manifest = {"run_name": "synthetic", "imagery_source": "mapillary"}
    area = {"type": "Polygon", "coordinates": [[[LNG0 - 0.01, LAT0 - 0.01], [LNG0 + 0.01, LAT0 - 0.01],
                                                 [LNG0 + 0.01, LAT0 + 0.01], [LNG0 - 0.01, LAT0 + 0.01],
                                                 [LNG0 - 0.01, LAT0 - 0.01]]]}
    result = position_check.check_run(manifest, area, records, _grid_osm(frame), threshold_m=3.0, min_sequence=20)

    assert result["submitted_field"] == "sfm"  # build_pano_record wrote computed_geometry
    assert result["flagged_sequences"] == ["A"]
    by_id = {s["sequence_id"]: s for s in result["sequences"]}
    assert by_id["A"]["recommended"] == "raw" and by_id["A"]["flagged"]
    assert by_id["A"]["sfm_minus_raw_east_m"] == pytest.approx(-8.0, abs=0.05)
    assert not by_id["B"]["flagged"]
    assert by_id["A"]["bias_m"]["sfm"]["a"] == pytest.approx(-7.5, abs=0.05)
    # Whole-run across-N-S summary sees A's SfM positions 7.5 m west of the line.
    assert result["fields"]["sfm"]["across_a"]["median"] == pytest.approx(-7.5, abs=0.05)
    assert result["fields"]["raw"]["across_a"]["median"] == pytest.approx(0.5, abs=0.05)
    assert result["fields"]["raw"]["across_a"]["neg_share"] == 0.0


def test_reposition_pano_switches_field_and_stays_submittable():
    meta = make_meta(geometry={"type": "Point", "coordinates": [-77.4300, 37.5400]})
    pano = mapillary.build_pano_record("img-1", 0.0, 0.0, meta)
    assert (pano["lng"], pano["lat"]) == (-77.4360, 37.5407)  # SfM by default

    moved = reposition.reposition_pano(pano, "raw")
    assert (moved["lng"], moved["lat"]) == (-77.4300, 37.5400)
    assert moved["position_field"] == "geometry"
    assert pano["lat"] == 37.5407  # input untouched
    assert reposition.reposition_pano(moved, "sfm")["lat"] == 37.5407  # round trip

    # The extra key rides along to PS like every other provenance field.
    payload = send_to_ps.transform_record({"detections": [], "pano": moved})
    assert payload["pano"]["position_field"] == "geometry"
    assert payload["pano"]["lat"] == 37.5400

    # Non-Mapillary blocks are refused rather than guessed at.
    assert reposition.reposition_pano({"source": "gsv", "lat": 1, "lng": 2}, "raw") is None


def test_build_pano_record_honours_position_field(monkeypatch):
    meta = make_meta(geometry={"type": "Point", "coordinates": [-77.4300, 37.5400]})
    monkeypatch.setattr(mapillary, "POSITION_FIELD", "raw")
    pano = mapillary.build_pano_record("img-1", 0.0, 0.0, meta)
    assert (pano["lng"], pano["lat"]) == (-77.4300, 37.5400)
    # An unreconstructed image (no computed_geometry) falls back to the GPS fix.
    monkeypatch.setattr(mapillary, "POSITION_FIELD", "sfm")
    pano = mapillary.build_pano_record("img-2", 0.0, 0.0, make_meta(computed_geometry=None,
                                        geometry={"type": "Point", "coordinates": [-77.4300, 37.5400]}))
    assert (pano["lng"], pano["lat"]) == (-77.4300, 37.5400)


def test_run_dir_is_bound_to_one_position_field(tmp_path):
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    run_dir = tmp_path / "city"
    manifest = main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="raw")
    assert manifest["mapillary_position"] == "raw"
    assert json.load(open(run_dir / "manifest.json"))["mapillary_position"] == "raw"
    main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="raw")
    with pytest.raises(SystemExit):
        main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="sfm")
    # A manifest from before the flag is an SfM run.
    del manifest["mapillary_position"]
    main.save_manifest(run_dir / "manifest.json", manifest)
    main.load_or_init_run_dir(run_dir, "city.geojson", geom, "hash", "mapillary", position_field="sfm")
