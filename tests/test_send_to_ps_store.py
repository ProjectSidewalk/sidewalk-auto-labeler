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
