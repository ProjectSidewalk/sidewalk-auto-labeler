"""send_to_ps refuses a run rebuilt from the Project Sidewalk pano store (#56, PR #96 review).

detect_from_store.py's >= 0.55 detections are, by construction, the labels the city
already has; PS is insert-only, so sending such a file duplicates them for good. Offline:
the POST is stubbed.
"""
import json
from types import SimpleNamespace

import pytest

import detect_from_store as dfs
import main
import send_to_ps
from conftest import make_process_result, make_provenance

URL = "https://ps.example/ai"


def _store_record(pid):
    meta = {"panoId": pid, "width": 64, "height": 32, "lat": 45.6, "lng": -122.5,
            "cameraHeading": 12.5, "cameraPitch": 1.0, "captureDate": "2023-05"}
    return main.build_output_line({"pano": dfs.pano_record_from_ps(pid, meta),
                                   "detections": [(0.5, 0.25, 0.9)]}, make_provenance())


def _capture(monkeypatch):
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text="")
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk",
                        lambda payload, url, key=None: sent.append(payload) or ok)
    return sent


def test_store_records_are_refused_unless_allowed(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(_store_record("P1")) + "\n", encoding="utf-8")
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match="ps_store"):
        send_to_ps.process_jsonl_file(str(path), URL)
    assert sent == []
    send_to_ps.process_jsonl_file(str(path), URL, dry_run=True)       # a dry run previews it
    assert sent == []
    send_to_ps.process_jsonl_file(str(path), URL, allow_store_file=True)
    assert len(sent) == 1


def test_a_manifest_with_pixels_is_refused(tmp_path, monkeypatch):
    """Even when the records themselves do not say so (e.g. hand-edited), the run's
    manifest does."""
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(main.build_output_line(make_process_result(), make_provenance()))
                    + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"pixels": {"store": "/s"}}),
                                            encoding="utf-8")
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match="pixels"):
        send_to_ps.process_jsonl_file(str(path), URL)
    assert sent == []


def test_an_ordinary_gsv_file_is_not_affected(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(main.build_output_line(make_process_result(), make_provenance()))
                    + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"imagery_source": "gsv"}),
                                            encoding="utf-8")
    send_to_ps.check_store_built(path)


# --- type-aware (#168/#169): the store's live AI labels are all CurbRamp -----------------

NCR = "NoCurbRamp"
NCR_MODEL = {"model_id": "ncr-model@abcdef123456", "model_training_date": "10-01-2026",
             "api_version": "1.0.0", "model_repo": "projectsidewalk/ncr-model",
             "model_revision": "abcdef1234567890abcdef1234567890abcdef12"}


@pytest.fixture
def ncr_spec(monkeypatch):
    from detectors import registry
    monkeypatch.setitem(registry.SPECS, NCR, registry.DetectorSpec(
        NCR, "detectors.ncr", "NcrDetector", 0.1, 0.5, 0.6, False))


def _typed_store_record(pid, types):
    meta = {"panoId": pid, "width": 64, "height": 32, "lat": 45.6, "lng": -122.5,
            "cameraHeading": 12.5, "cameraPitch": 1.0, "captureDate": "2023-05"}
    pmap = {t: (make_provenance() if t == "CurbRamp" else dict(NCR_MODEL)) for t in types}
    return main.build_output_line(
        {"pano_id": pid, "pano": dfs.pano_record_from_ps(pid, meta), "label_types": types,
         "detections": [(0.5, 0.25, 0.9, t) for t in types]}, pmap)


def test_a_no_curb_ramp_only_store_file_passes_and_is_recorded(tmp_path, monkeypatch, ncr_spec):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(_typed_store_record("P1", [NCR])) + "\n", encoding="utf-8")
    assert "only NoCurbRamp" in send_to_ps.check_store_built(path)
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), URL)
    assert [p["label_type"] for p in sent] == [NCR]
    record = json.loads((tmp_path / "results.jsonl.submission.json").read_text(encoding="utf-8"))
    assert "type-aware store guard" in record["store_built"]["reason"]


def test_a_mixed_store_file_is_refused(tmp_path, monkeypatch, ncr_spec):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(_typed_store_record("P1", ["CurbRamp", NCR])) + "\n",
                    encoding="utf-8")
    sent = _capture(monkeypatch)
    with pytest.raises(ValueError, match="CurbRamp labels"):
        send_to_ps.process_jsonl_file(str(path), URL)
    assert sent == []


def test_an_untyped_legacy_store_file_reads_as_curb_ramp_and_is_refused(tmp_path):
    rec = _store_record("P1")
    rec.pop("label_type", None)
    assert "label_types" not in rec
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CurbRamp labels"):
        send_to_ps.check_store_built(path)


def test_a_manifest_store_run_of_no_curb_ramp_only_passes(tmp_path, ncr_spec):
    path = tmp_path / "results.jsonl"
    rec = _typed_store_record("P1", [NCR])
    rec["pano"].pop("source_detail", None)
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"pixels": {"store": "/s"}}),
                                            encoding="utf-8")
    assert "pixels" in send_to_ps.check_store_built(path)
