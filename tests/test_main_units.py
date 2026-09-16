"""Unit tests for main.py's pure helpers (no network, no model)."""
import json

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
    assert "Overpass" in json.load(open(manifest_path))["position_check"]["error"]
