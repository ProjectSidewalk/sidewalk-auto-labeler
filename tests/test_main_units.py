"""Unit tests for main.py's pure helpers (no network, no model)."""
import json
from pathlib import Path

import main
from conftest import make_process_result as _result, make_provenance


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
    line = main.build_output_line(_result(), make_provenance())
    assert line["detections"] == [
        {"x_normalized": 0.5, "y_normalized": 0.25, "confidence": 0.9}]
    assert line["label_type"] == "CurbRamp"
    # Provenance comes from the resolved snapshot (issue #39): the three keys PS stores per
    # label, in the formats it parses, plus the full revision for everything else.
    assert line["model_id"] == "rampnet-model@606a11956743"
    assert line["model_training_date"] == "08-21-2025"
    assert line["api_version"] == "1.0.0"
    assert line["model_repo"] == "projectsidewalk/rampnet-model"
    assert line["model_revision"] == "606a11956743f7eb328d9207769034752f6191f4"
    # The source-built pano block passes through untouched (its shape is covered by
    # the per-source tests).
    assert line["pano"]["panorama_id"] == "PID"
    json.dumps(line)  # must be JSON-serializable as written


def test_build_output_line_zero_detections():
    assert main.build_output_line(_result(detections=[]), make_provenance())["detections"] == []


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


# --- Model provenance (issues #39, #6): resolved from the snapshot, refusing the unknown ---

import pytest  # noqa: E402

import detectors  # noqa: E402

PAPER_MAIN = "606a11956743f7eb328d9207769034752f6191f4"   # HF main since 2026-07-24
UNKNOWN_SHA = "0123456789abcdef0123456789abcdef01234567"


def _snapshot(tmp_path, sha):
    d = tmp_path / "hub" / "models--projectsidewalk--rampnet-model" / "snapshots" / sha
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d / "config.json"


def test_provenance_from_snapshot_dir_resolves_the_cache_path(tmp_path):
    """The offline fallback: with no commit hash from transformers, the snapshots/<sha>
    directory the weights were read from IS the revision."""
    prov = detectors.provenance_from_snapshot_dir(_snapshot(tmp_path, PAPER_MAIN))
    assert prov == {
        "model_repo": "projectsidewalk/rampnet-model",
        "model_revision": PAPER_MAIN,
        "model_id": "rampnet-model@606a11956743",
        # PS parses MM-dd-yyyy into a NOT NULL timestamp (ExploreService.submitAiLabelData);
        # the paper weights were trained 2025-08-21, which is the date every live row holds.
        "model_training_date": "08-21-2025",
        "api_version": "1.0.0",
    }
    # The directory alone works too, and a commit hash from transformers takes precedence.
    assert detectors.provenance_from_snapshot_dir(
        _snapshot(tmp_path, "1078bcd6771d63bd845d9fd36042904473e20b07").parent,
        commit_hash=PAPER_MAIN)["model_revision"] == PAPER_MAIN


def test_model_id_keeps_the_prefix_consumers_match_on():
    model_id = detectors.model_id_for(PAPER_MAIN)
    assert model_id.startswith("rampnet-model") and model_id != "rampnet-model"
    assert model_id.split("@")[1] == PAPER_MAIN[:12]


def test_every_known_revision_is_a_full_sha_with_an_iso_date():
    for sha, row in detectors.KNOWN_REVISIONS.items():
        assert detectors.is_revision(sha)
        assert detectors.ps_training_date(row["training_date"])  # parses as ISO
        assert row["note"]


def test_unknown_revision_refuses_and_names_the_sha_and_the_table(tmp_path):
    with pytest.raises(detectors.ModelProvenanceError) as e:
        detectors.provenance_from_snapshot_dir(_snapshot(tmp_path, UNKNOWN_SHA))
    assert UNKNOWN_SHA in str(e.value) and "KNOWN_REVISIONS" in str(e.value)
    # The override runs, but the date is null — never a stale default that reads as fact.
    prov = detectors.provenance_from_snapshot_dir(None, commit_hash=UNKNOWN_SHA, allow_unknown=True)
    assert prov["model_training_date"] is None
    assert prov["model_id"] == "rampnet-model@0123456789ab"


def test_unresolvable_revision_refuses_even_with_the_override(tmp_path):
    """A plain local folder has no snapshots/<sha>: there is nothing to attribute to, and
    --allow-unknown-model-revision covers an unknown SHA, not a missing one."""
    folder = tmp_path / "my-local-model" / "config.json"
    for commit_hash in (None, "main", PAPER_MAIN[:12]):
        with pytest.raises(detectors.ModelProvenanceError):
            detectors.provenance_from_snapshot_dir(folder, commit_hash=commit_hash, allow_unknown=True)


def test_load_falls_back_to_the_local_cache_when_the_hub_is_unreachable():
    calls = []

    def loader(repo, **kw):
        calls.append(kw)
        if not kw.get("local_files_only"):
            raise OSError("We couldn't connect to 'https://huggingface.co'")
        return "model"

    assert detectors.load_with_offline_fallback(loader, trust_remote_code=True) == "model"
    assert calls == [{"trust_remote_code": True},
                     {"trust_remote_code": True, "local_files_only": True}]

    def always_fails(repo, **kw):
        raise OSError("not in the cache either")
    with pytest.raises(OSError):
        detectors.load_with_offline_fallback(always_fails)


# --- The run directory is bound to one model revision (issue #39) ---

GEOM = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


def _init(run_dir, provenance):
    return main.load_or_init_run_dir(run_dir, "city.geojson", GEOM, "hash", "gsv",
                                     provenance=provenance)


def test_manifest_records_the_model_and_refuses_a_different_revision(tmp_path):
    run_dir = tmp_path / "city"
    manifest = _init(run_dir, make_provenance())
    assert manifest["model_revision"] == PAPER_MAIN
    assert manifest["model_id"] == "rampnet-model@606a11956743"
    main.record_run(run_dir / "manifest.json", manifest, "t0", 1, 1, 0, 0,
                    provenance=make_provenance())
    assert json.load(open(run_dir / "manifest.json"))["runs"][0]["model"]["model_revision"] == PAPER_MAIN

    _init(run_dir, make_provenance())  # same revision resumes
    other = make_provenance(UNKNOWN_SHA, allow_unknown=True)
    with pytest.raises(SystemExit) as e:
        _init(run_dir, other)
    assert "revision" in str(e.value) and "new --name" in str(e.value)


def test_scan_only_binds_no_model_and_a_later_run_binds_quietly(tmp_path, capsys):
    run_dir = tmp_path / "city"
    assert "model_revision" not in _init(run_dir, None)
    assert _init(run_dir, make_provenance())["model_revision"] == PAPER_MAIN
    assert "predates" not in capsys.readouterr().out
    assert json.load(open(run_dir / "manifest.json"))["model_revision"] == PAPER_MAIN


def test_legacy_manifest_resumes_once_with_a_note(tmp_path, capsys):
    """Manifests from before #39 carry the three literals and no revision. Same training
    date = the paper weights, so the run resumes and is bound, and the note prints once."""
    run_dir = tmp_path / "city"
    _init(run_dir, None)
    path = run_dir / "manifest.json"
    legacy = json.load(open(path))
    legacy.update(model_id="rampnet-model", model_training_date="08-21-2025", api_version="1.0.0")
    path.write_text(json.dumps(legacy))

    assert _init(run_dir, make_provenance())["model_revision"] == PAPER_MAIN
    assert "predates model revisions" in capsys.readouterr().out
    saved = json.load(open(path))
    assert saved["legacy_model_provenance"]["model_id"] == "rampnet-model"
    _init(run_dir, make_provenance())
    assert "predates" not in capsys.readouterr().out  # one-time

    # A legacy run whose recorded date differs from the loaded model's is different weights.
    legacy["model_training_date"] = "01-01-2027"
    path.write_text(json.dumps(legacy))
    with pytest.raises(SystemExit):
        _init(run_dir, make_provenance())


def test_unknown_revision_override_is_recorded_in_the_manifest(tmp_path):
    run_dir = tmp_path / "city"
    manifest = _init(run_dir, make_provenance(UNKNOWN_SHA, allow_unknown=True))
    assert manifest["model_training_date"] is None
    assert manifest["unknown_revision_allowed"] is True
    assert "unknown_revision_allowed" not in main.manifest_model_block(make_provenance())


def test_override_run_is_refused_once_its_revision_becomes_known(tmp_path, monkeypatch):
    """A run made under --allow-unknown-model-revision holds null-date records. Once that
    SHA gains a KNOWN_REVISIONS row the loaded provenance has a date, and resuming would
    append dated lines after undated ones from the same weights — refuse, and say to start
    a fresh --name, since the null-date lines already written can never be submitted."""
    run_dir = tmp_path / "city"
    _init(run_dir, make_provenance(UNKNOWN_SHA, allow_unknown=True))
    _init(run_dir, make_provenance(UNKNOWN_SHA, allow_unknown=True))  # still unknown: resumes
    monkeypatch.setitem(detectors.KNOWN_REVISIONS, UNKNOWN_SHA,
                        {"training_date": "2027-01-01", "note": "test row"})
    with pytest.raises(SystemExit) as e:
        _init(run_dir, make_provenance(UNKNOWN_SHA))
    assert "--allow-unknown-model-revision" in str(e.value) and "fresh --name" in str(e.value)
    # Nothing was rewritten: the manifest still shows the override the records were made under.
    saved = json.load(open(run_dir / "manifest.json"))
    assert saved["unknown_revision_allowed"] is True and saved["model_training_date"] is None


def test_scan_only_stays_torch_free():
    """--scan-only must not pay for (or require) torch: main imports the detector lazily,
    and the provenance machinery it now imports eagerly lives in torch-free detectors."""
    import subprocess
    import sys
    code = ("import sys, main, detectors; "
            "bad = [m for m in ('torch', 'transformers', 'detectors.curb_ramp') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=str(Path(main.__file__).parent)).returncode == 0
