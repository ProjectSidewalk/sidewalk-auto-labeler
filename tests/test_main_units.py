"""Unit tests for main.py's pure helpers (no network, no model)."""
import json

import pytest

import main
from conftest import make_process_result as _result


def test_latlon_to_tile_known_values():
    # Zoom 0 is a single world tile; the origin is (0, 0).
    assert main.latlon_to_tile(0.0, 0.0, 0) == (0, 0)
    # Bend, OR at zoom 17 (the coverage-scan zoom) — regression-pinned values,
    # cross-checked against the standard Slippy Map formula.
    x, y = main.latlon_to_tile(44.058, -121.315, 17)
    assert (x, y) == (21366, 47630)


def test_latlon_to_tile_monotonic():
    # x grows eastward, y grows southward.
    x_w, y_n = main.latlon_to_tile(45.0, -122.0, 12)
    x_e, y_s = main.latlon_to_tile(44.0, -121.0, 12)
    assert x_e > x_w and y_s > y_n


def test_geojson_hash_stable_across_key_order():
    a = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    b = {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]], "type": "Polygon"}
    assert main.get_geojson_hash(a) == main.get_geojson_hash(b)
    # Any coordinate change must change the hash (this is the run-dir guard).
    c = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 2], [0, 0]]]}
    assert main.get_geojson_hash(a) != main.get_geojson_hash(c)


def test_load_processed_ids(tmp_path):
    assert main.load_processed_ids(tmp_path / "missing.txt") == set()
    cache = tmp_path / "cache.txt"
    cache.write_text("abc\ndef\nabc\n")
    assert main.load_processed_ids(cache) == {"abc", "def"}


def test_build_output_line_shape():
    line = main.build_output_line(_result())
    assert line["detections"] == [
        {"x_normalized": 0.5, "y_normalized": 0.25, "confidence": 0.9}]
    assert line["label_type"] == "CurbRamp"
    assert line["model_id"] == main.MODEL_ID
    # The source-built pano block passes through untouched (its shape is covered by
    # the per-source tests).
    assert line["pano"]["panorama_id"] == "PID"
    json.dumps(line)  # must be JSON-serializable as written


def test_build_output_line_zero_detections():
    assert main.build_output_line(_result(detections=[]))["detections"] == []


def _results_file(tmp_path, links_per_record):
    path = tmp_path / "results.jsonl"
    lines = []
    for pid, targets in links_per_record.items():
        links = [{"target_gsv_panorama_id": t, "yaw_deg": 0.0, "description": ""} for t in targets]
        lines.append(json.dumps({"pano": {"panorama_id": pid, "links": links}, "detections": []}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_dangling_link_targets_subtracts_the_cache(tmp_path):
    # A links to B (processed) and X (dangling); B links back to A (processed).
    path = _results_file(tmp_path, {"A": ["B", "X"], "B": ["A"]})
    assert main.dangling_link_targets(path, {"A", "B"}) == {"X"}
    # A cached deterministic skip (e.g. gap-fill target outside the area) stays gone.
    assert main.dangling_link_targets(path, {"A", "B", "X"}) == set()
    # Runs pulled from a cluster have results.jsonl but no cache: the records' own
    # ids must not count as dangling.
    assert main.dangling_link_targets(path, set()) == {"X"}


def test_dangling_link_targets_tolerates_linkless_records(tmp_path):
    # Mapillary records carry links: [] — the phase must be a no-op over them.
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"pano": {"panorama_id": "M", "links": []}, "detections": []}) + "\n" +
                    json.dumps({"pano": {"panorama_id": "N"}, "detections": []}) + "\n")
    assert main.dangling_link_targets(path, set()) == set()


def test_dangling_link_targets_skips_truncated_final_line(tmp_path):
    # A run killed mid-write can leave a truncated last line; the intact records
    # must still be read (its own pano is uncached, so the main pass retries it).
    path = _results_file(tmp_path, {"A": ["X"]})
    with open(path, 'a') as f:
        f.write('{"pano": {"panorama_id": "TRUNC", "li')
    assert main.dangling_link_targets(path, set()) == {"X"}


def test_run_position_check_records_the_verdict_and_survives_failure(tmp_path, monkeypatch):
    """Every run ends with the position check (SidewalkWebpage#5361). A failure (no
    network, Overpass down) is recorded and warned about but never fails the run: the
    detections are on disk and send_to_ps.py refuses the file until the check passes."""
    import position_check
    run_dir = tmp_path / "city"
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    manifest = {"runs": []}

    monkeypatch.setattr(position_check, "run_check", lambda rd: {
        "checked_at": "2026-09-16T00:00:00Z", "results_sha256": "abc", "submitted_field": "sfm",
        "flagged_sequences": ["A"], "both_off_sequences": [], "panos_not_near_a_street": 3})
    assert main.run_position_check(run_dir, manifest_path, manifest)["flagged_sequences"] == ["A"]
    saved = json.load(open(manifest_path))
    assert saved["position_check"] == {"checked_at": "2026-09-16T00:00:00Z", "results_sha256": "abc",
                                       "submitted_field": "sfm", "flagged": 1, "both_off": 0,
                                       "panos_not_near_a_street": 3}

    def boom(rd):
        raise OSError("Overpass query failed on every endpoint")
    monkeypatch.setattr(position_check, "run_check", boom)
    assert main.run_position_check(run_dir, manifest_path, manifest) is None
    failed = json.load(open(manifest_path))["position_check"]
    assert "Overpass" in failed["error"] and "checked_at" in failed

    # Idempotent: a check already pinned to the current results.jsonl (hash + report on
    # disk) is reused, so a no-op resume never rewrites the two git-tracked outputs.
    results = run_dir / "results.jsonl"
    results.write_text("{}\n")
    pinned = {"checked_at": "2026-09-16T01:00:00Z", "results_sha256": position_check.file_sha256(results),
              "submitted_field": None, "flagged_sequences": [], "both_off_sequences": [],
              "panos_not_near_a_street": 0}
    position_check.check_path_for(results).write_text(json.dumps(pinned))
    position_check.report_path_for(results).write_text("<html>")
    calls = []
    monkeypatch.setattr(position_check, "run_check", lambda rd: calls.append(rd) or pinned)
    assert main.run_position_check(run_dir, manifest_path, manifest)["checked_at"] == "2026-09-16T01:00:00Z"
    assert calls == []
    results.write_text("{}\n{}\n")  # the file changed: the check runs again
    main.run_position_check(run_dir, manifest_path, manifest)
    assert calls == [run_dir]


# --- GeoJSON input normalization (issue #7) --------------------------------------------

SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
FAR_SQUARE = {"type": "Polygon", "coordinates": [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]}


def _feature(geometry):
    return {"type": "Feature", "properties": {"name": "x"}, "geometry": geometry}


def _hash_of(data):
    return main.get_geojson_hash(main.extract_geometry(data)[0])


def test_bare_feature_and_single_feature_collection_hash_alike():
    # Wrapping or unwrapping the same polygon must bind to the same run directory,
    # and the bare case must hash exactly as it did before #7 (no run forks).
    bare = _hash_of(SQUARE)
    assert bare == main.get_geojson_hash(SQUARE)
    assert _hash_of(_feature(SQUARE)) == bare
    assert _hash_of({"type": "FeatureCollection", "features": [_feature(SQUARE)]}) == bare
    assert main.extract_geometry(_feature(SQUARE))[1] == "Feature"


def test_committed_area_hash_survives_a_feature_wrapper(tmp_path):
    # The same check on a real file loaded the way run_labeler loads it.
    import geojson
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "example_geojson" / "richmond.geojson"
    bare = json.loads(src.read_text())
    wrapped = tmp_path / "wrapped.geojson"
    wrapped.write_text(json.dumps({"type": "FeatureCollection", "features": [_feature(bare)]}))
    with open(src) as f:
        bare_hash = main.get_geojson_hash(main.extract_geometry(geojson.load(f))[0])
    with open(wrapped) as f:
        wrapped_hash = main.get_geojson_hash(main.extract_geometry(geojson.load(f))[0])
    assert wrapped_hash == bare_hash


def test_multi_feature_collection_dissolves_to_one_multipolygon(capsys):
    geometry, input_type = main.extract_geometry(
        {"type": "FeatureCollection", "features": [_feature(SQUARE), _feature(FAR_SQUARE)]})
    assert input_type == "FeatureCollection"
    assert geometry["type"] == "MultiPolygon"
    assert len(geometry["coordinates"]) == 2
    assert main.shape(geometry).area == 2.0
    json.dumps(geometry)                          # plain lists, storable as area.geojson
    assert "dissolved" in capsys.readouterr().out


def test_touching_features_dissolve_to_a_multipolygon_not_a_polygon():
    right = {"type": "Polygon", "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]]}
    geometry, _ = main.extract_geometry(
        {"type": "FeatureCollection", "features": [_feature(SQUARE), _feature(right)]})
    assert geometry["type"] == "MultiPolygon" and len(geometry["coordinates"]) == 1


@pytest.mark.parametrize("data, found", [
    (_feature({"type": "Point", "coordinates": [0, 0]}), "Point"),
    ({"type": "GeometryCollection", "geometries": [SQUARE]}, "GeometryCollection"),
    ({"type": "FeatureCollection", "features": []}, "no features"),
    (_feature(None), "empty"),
])
def test_non_polygonal_input_is_refused_naming_what_was_found(data, found):
    with pytest.raises(ValueError, match=found):
        main.extract_geometry(data)


def test_manifest_records_input_type_and_area_geojson_stays_bare(tmp_path):
    geometry, input_type = main.extract_geometry(_feature(SQUARE))
    manifest = main.load_or_init_run_dir(tmp_path / "run", "in.geojson", geometry,
                                         main.get_geojson_hash(geometry), "gsv",
                                         input_geojson_type=input_type)
    assert manifest["input_geojson_type"] == "Feature"
    assert json.loads((tmp_path / "run" / "area.geojson").read_text())["type"] == "Polygon"


def test_committed_run_area_hash_still_reproduces_from_its_area_geojson():
    # Backward compatibility pinned against REAL data, not new code against new code:
    # paterson's manifest was written before #7, and a resume re-hashes whatever file it
    # is pointed at, so the committed area.geojson must still produce the committed hash.
    import geojson
    from pathlib import Path
    run_dir = Path(__file__).resolve().parent.parent / "runs" / "paterson"
    recorded = json.loads((run_dir / "manifest.json").read_text())["area_hash"]
    assert recorded == "7a3e679281f6318a8d7d152d131307d7d76efdec6f247dbce318f56b52ca8cb4"
    with open(run_dir / "area.geojson") as f:
        geometry, input_type = main.extract_geometry(geojson.load(f))
    assert input_type == "Polygon"
    assert main.get_geojson_hash(geometry) == recorded


def test_dissolved_area_resumes_from_its_own_area_geojson(tmp_path, capsys):
    # unary_union computes new vertices at full float precision (here where the two
    # squares' edges cross, x=1, y=0.5555555...), while geojson.load rounds to 6
    # decimals. The dissolved geometry must already be rounded, or the area.geojson a run
    # stores could never re-hash to the manifest's area_hash and a resume would be refused.
    import geojson
    tilted = {"type": "Polygon", "coordinates": [[[0.3, 0.123457], [1.7, 0.987654],
                                                  [1.7, 1.5], [0.3, 1.5], [0.3, 0.123457]]]}
    collection = geojson.loads(json.dumps(
        {"type": "FeatureCollection", "features": [_feature(SQUARE), _feature(tilted)]}))
    geometry, _ = main.extract_geometry(collection)
    area_hash = main.get_geojson_hash(geometry)
    main.load_or_init_run_dir(tmp_path / "run", "in.geojson", geometry, area_hash, "gsv",
                              input_geojson_type="FeatureCollection")
    with open(tmp_path / "run" / "area.geojson") as f:
        reloaded, input_type = main.extract_geometry(geojson.load(f))
    assert input_type == "MultiPolygon"
    assert main.get_geojson_hash(reloaded) == area_hash


def test_invalid_features_are_refused_cleanly_not_with_a_geos_traceback(tmp_path):
    # A self-intersecting ("bowtie") part makes unary_union throw a GEOS
    # TopologyException; it must surface as the same clean ValueError / sys.exit.
    bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    overlapping = {"type": "Polygon",
                   "coordinates": [[[0.5, 0], [2, 0], [2, 2], [0.5, 2], [0.5, 0]]]}
    data = {"type": "FeatureCollection", "features": [_feature(bowtie), _feature(overlapping)]}
    with pytest.raises(ValueError, match="could not be dissolved"):
        main.extract_geometry(data)
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as exc:
        main.run_labeler(str(path), "bad", source=None)
    assert "could not be dissolved" in str(exc.value.code)
