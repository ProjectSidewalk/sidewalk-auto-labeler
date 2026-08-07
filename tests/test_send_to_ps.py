"""Unit tests for send_to_ps.py's record transform, endpoint guard and resume sidecar."""
import json
from types import SimpleNamespace

import pytest

import main
import send_to_ps
from conftest import make_process_result


def _record(detections):
    return {
        "detections": detections,
        "label_type": "CurbRamp",
        "model_id": "rampnet-model",
        "pano": {"panorama_id": "PID", "width": 16384, "height": 8192},
    }


def test_transform_record_pixel_conversion():
    payload = send_to_ps.transform_record(
        _record([{"x_normalized": 0.5, "y_normalized": 0.25, "confidence": 0.91}]))
    assert payload["labels"] == [{"pano_x": 8192, "pano_y": 2048, "confidence": 0.91}]
    # The normalized-coordinate key must not survive the transform.
    assert "detections" not in payload
    # Everything else passes through untouched.
    assert payload["label_type"] == "CurbRamp"
    assert payload["pano"]["pano_id"] == "PID"


def test_transform_record_rounds_to_int_pixels():
    payload = send_to_ps.transform_record(
        _record([{"x_normalized": 0.333333, "y_normalized": 0.666666, "confidence": 0.6}]))
    label = payload["labels"][0]
    assert isinstance(label["pano_x"], int) and isinstance(label["pano_y"], int)
    assert label["pano_x"] == round(0.333333 * 16384)


def test_transform_record_zero_detections():
    payload = send_to_ps.transform_record(_record([]))
    assert payload["labels"] == [] and "detections" not in payload


def test_transform_record_filters_below_operational_confidence():
    """results.jsonl stores candidate peaks below the operational threshold (the
    storage floor, issue #27); only operational detections may become PS labels.
    A detection exactly at the threshold is kept (>=, not >)."""
    payload = send_to_ps.transform_record(_record([
        {"x_normalized": 0.5, "y_normalized": 0.25, "confidence": 0.91},
        {"x_normalized": 0.25, "y_normalized": 0.75, "confidence": 0.55},
        {"x_normalized": 0.1, "y_normalized": 0.75, "confidence": 0.2},
    ]))
    assert [label["confidence"] for label in payload["labels"]] == [0.91, 0.55]


def test_transform_record_all_subthreshold_still_submits_empty():
    """A pano whose stored peaks are all sub-threshold is still a processed pano: it
    must submit with empty labels ('checked, nothing found'), never be dropped."""
    payload = send_to_ps.transform_record(
        _record([{"x_normalized": 0.5, "y_normalized": 0.25, "confidence": 0.2}]))
    assert payload["labels"] == [] and "detections" not in payload


def test_transform_record_min_confidence_is_a_knob():
    record = _record([{"x_normalized": 0.5, "y_normalized": 0.25, "confidence": 0.2}])
    assert len(send_to_ps.transform_record(record, min_confidence=0)["labels"]) == 1


def test_transform_record_does_not_mutate_input():
    record = _record([{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.7}])
    send_to_ps.transform_record(record)
    assert "detections" in record and "labels" not in record
    assert "panorama_id" in record["pano"] and "pano_id" not in record["pano"]


def test_transform_pano_maps_legacy_fields_to_ps_reader():
    """The server's PanoSubmission reader expects pano_id, an enum source, and
    target_pano_id links; legacy Stage-1 records carry streetlevel-flavored names."""
    pano = send_to_ps.transform_pano({
        "panorama_id": "PID",
        "source": "launch",  # raw streetlevel source string, not a pano_source enum value
        "links": [{"target_gsv_panorama_id": "NEXT", "yaw_deg": 12.5, "description": ""}],
    })
    assert pano["pano_id"] == "PID" and "panorama_id" not in pano
    assert pano["source"] == "gsv"
    assert pano["links"] == [{"target_pano_id": "NEXT", "yaw_deg": 12.5, "description": ""}]
    # history is a required (possibly empty) array server-side.
    assert pano["history"] == []


def test_transform_pano_forwards_all_provenance():
    """We submit every field we have, so provenance is already in the payload the day PS
    learns to store it. The server's reader ignores keys it doesn't name, so nothing here
    can break a submission — but nothing may quietly strip them either."""
    pano = send_to_ps.transform_pano({
        "panorama_id": "123456789",
        "source": "mapillary",
        "camera_make": "GoPro",
        "sequence_id": "seq-1",
        "source_metadata": {"make": "GoPro", "camera_parameters": [0.4, 0.0, 0.0]},
    })
    assert pano["source_metadata"] == {"make": "GoPro", "camera_parameters": [0.4, 0.0, 0.0]}
    assert pano["camera_make"] == "GoPro" and pano["sequence_id"] == "seq-1"


def test_transform_pano_passes_through_canonical_fields():
    pano = send_to_ps.transform_pano({
        "pano_id": "123456789",
        "source": "mapillary",
        "links": [],
        "history": [{"pano_id": "OLD", "date": "2019-08"}],
    })
    assert pano["pano_id"] == "123456789"
    assert pano["source"] == "mapillary"
    assert pano["history"] == [{"pano_id": "OLD", "date": "2019-08"}]


def test_transform_accepts_real_stage1_records():
    """Producer→consumer contract: feed transform_record an actual build_output_line
    record (round-tripped through JSON like the JSONL file), so a key rename or
    reshape on either side fails here instead of at submission time."""
    record = json.loads(json.dumps(main.build_output_line(make_process_result())))
    payload = send_to_ps.transform_record(record)
    # 0.5 * 16384, 0.25 * 8192 — dimensions come from the record's own pano block.
    assert payload["labels"] == [{"pano_x": 8192, "pano_y": 2048, "confidence": 0.9}]
    assert "detections" not in payload
    assert payload["label_type"] == "CurbRamp"
    # The pano block must come out in the shape the PS reader accepts.
    pano = payload["pano"]
    assert pano["pano_id"] == "PID" and "panorama_id" not in pano
    assert pano["source"] in send_to_ps.PS_PANO_SOURCES
    assert isinstance(pano["links"], list) and isinstance(pano["history"], list)
    assert all("target_pano_id" in link for link in pano["links"])


def test_load_submitted_lines(tmp_path):
    assert send_to_ps.load_submitted_lines(tmp_path / "missing") == set()
    sidecar = tmp_path / "r.jsonl.submitted"
    sidecar.write_text("1\n3\n\n3\n")
    assert send_to_ps.load_submitted_lines(sidecar) == {1, 3}


# --- Endpoint security guard (issue #9) -------------------------------------------------
#
# PS's internal API key is a shared server secret, so a mistyped --endpoint that drops the
# 's' must fail loudly rather than leak it to every hop in between.

@pytest.mark.parametrize("url", [
    "https://sidewalk-richmond.cs.washington.edu/ai/submitLabelsOnPano",
    "http://localhost:9000/ai/submitLabelsOnPano",
    "http://127.0.0.1:9000/ai/submitLabelsOnPano",
    "http://[::1]:9000/ai/submitLabelsOnPano",
    "http://0.0.0.0:9000/ai/submitLabelsOnPano",   # the bind address of a local dev server
])
def test_check_endpoint_security_allows_https_and_loopback(url):
    send_to_ps.check_endpoint_security(url, "SECRET")


@pytest.mark.parametrize("url", [
    "http://sidewalk-richmond.cs.washington.edu/ai/submitLabelsOnPano",  # the typo that matters
    "http://192.168.1.50:9000/ai/submitLabelsOnPano",                    # LAN is still the wire
    "ftp://sidewalk-richmond.cs.washington.edu/ai/submitLabelsOnPano",   # only https is trusted
])
def test_check_endpoint_security_refuses_cleartext_key(url):
    with pytest.raises(ValueError, match="cleartext"):
        send_to_ps.check_endpoint_security(url, "SECRET")


def test_check_endpoint_security_ignores_plain_http_without_a_key():
    """No key, nothing to protect: deployments that don't require auth stay submittable."""
    send_to_ps.check_endpoint_security("http://example.org/ai/submitLabelsOnPano", None)


def test_error_message_never_echoes_the_key():
    with pytest.raises(ValueError) as excinfo:
        send_to_ps.check_endpoint_security("http://example.org/ai/submitLabelsOnPano", "SECRET")
    assert "SECRET" not in str(excinfo.value)


# --- Staged submission (--limit) --------------------------------------------------------

def _jsonl(tmp_path, count):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(
        json.dumps(_record([{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}])) + "\n"
        for _ in range(count)))
    return path


def _capture_posts(monkeypatch):
    """Record every payload that would be POSTed, and hand back a stand-in for requests'
    Response — truthy, and carrying the attributes a caller might come to inspect, so a
    later `response.status_code` check doesn't silently break these tests."""
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text="")
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk",
                        lambda payload, url, key=None: sent.append(payload) or ok)
    return sent


def test_limit_caps_records_submitted_and_resumes(tmp_path, monkeypatch):
    """The first-submission checklist is 'send 2-3, look at them in Validate, then the
    rest'. Successive capped runs must walk the file rather than resend the same head."""
    path = _jsonl(tmp_path, 10)
    sent = _capture_posts(monkeypatch)

    send_to_ps.process_jsonl_file(str(path), "https://ps.example/ai", limit=3)
    assert len(sent) == 3
    assert send_to_ps.load_submitted_lines(tmp_path / "results.jsonl.submitted") == {1, 2, 3}

    send_to_ps.process_jsonl_file(str(path), "https://ps.example/ai", limit=3)
    assert len(sent) == 6
    assert send_to_ps.load_submitted_lines(tmp_path / "results.jsonl.submitted") == set(range(1, 7))


def test_no_limit_submits_everything(tmp_path, monkeypatch):
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), "https://ps.example/ai")
    assert len(sent) == 4


def test_submission_refuses_a_cleartext_remote_endpoint(tmp_path, monkeypatch):
    """The guard must fire before any request is made, not after the first leak."""
    path = _jsonl(tmp_path, 2)
    sent = _capture_posts(monkeypatch)
    with pytest.raises(ValueError):
        send_to_ps.process_jsonl_file(str(path), "http://ps.example/ai", api_key="SECRET")
    assert sent == []


def test_dry_run_is_exempt_from_the_endpoint_guard(tmp_path, capsys):
    """A dry run sends nothing, so it must stay usable for previewing a payload against
    whatever URL is at hand."""
    path = _jsonl(tmp_path, 2)
    send_to_ps.process_jsonl_file(str(path), "http://ps.example/ai", api_key="SECRET",
                                  dry_run=True, limit=1)
    assert "SECRET" not in capsys.readouterr().out
    # A dry run records nothing, so it never creates the resume sidecar.
    assert not (tmp_path / "results.jsonl.submitted").exists()
