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


TEST = "https://ps-test.example/ai/submitLabelsOnPano"
PROD = "https://ps.example/ai/submitLabelsOnPano"


def _read_record(tmp_path):
    return json.loads((tmp_path / "results.jsonl.submission.json").read_text())


def _sidecar(tmp_path):
    return send_to_ps.load_submitted_lines(tmp_path / "results.jsonl.submitted")


# --- Submission record + resume guard ---------------------------------------------------

def test_submission_record_tracks_the_campaign(tmp_path, monkeypatch):
    """The record is the campaign's only git-committable memory of what went where: it must
    agree with the sidecar line-for-line, per endpoint, across resumed runs."""
    path = _jsonl(tmp_path, 5)
    _capture_posts(monkeypatch)

    send_to_ps.process_jsonl_file(str(path), PROD, limit=2)
    record = _read_record(tmp_path)
    assert record["total_lines"] == 5
    assert list(record["endpoints"]) == [PROD]
    state = record["endpoints"][PROD]
    assert (state["submitted_lines"], state["labels_submitted"]) == (2, 2)   # one label/record

    send_to_ps.process_jsonl_file(str(path), PROD)
    state = _read_record(tmp_path)["endpoints"][PROD]
    assert state["submitted_lines"] == 5 == len(_sidecar(tmp_path))
    assert state["labels_submitted"] == 5
    assert state["first_submission_utc"] <= state["last_submission_utc"]


def test_record_counts_come_from_the_sidecar_not_the_run(tmp_path, monkeypatch):
    """A sidecar that predates the record (a campaign started under older code) must be
    counted in full, and a run that lands nothing new must not restamp the record."""
    path = _jsonl(tmp_path, 4)
    _capture_posts(monkeypatch)
    (tmp_path / "results.jsonl.submitted").write_text("1\n2\n")

    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)
    state = _read_record(tmp_path)["endpoints"][PROD]
    assert (state["submitted_lines"], state["labels_submitted"]) == (3, 3)

    send_to_ps.process_jsonl_file(str(path), PROD)
    before = (tmp_path / "results.jsonl.submission.json").read_text()
    send_to_ps.process_jsonl_file(str(path), PROD)          # everything already submitted
    assert (tmp_path / "results.jsonl.submission.json").read_text() == before


def test_record_written_on_interrupt(tmp_path, monkeypatch):
    """Ctrl-C mid-run: the lines that landed are in the sidecar, so the record must catch up
    with them rather than lose them until the next clean finish."""
    path = _jsonl(tmp_path, 5)
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text="")

    def post(payload, url, key=None):
        if len(sent) == 2:
            raise KeyboardInterrupt
        sent.append(payload)
        return ok
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk", post)

    with pytest.raises(SystemExit):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert _sidecar(tmp_path) == {1, 2}
    assert _read_record(tmp_path)["endpoints"][PROD]["submitted_lines"] == 2


def test_no_record_when_nothing_landed(tmp_path, monkeypatch):
    """A first run where every POST fails must not create a record claiming an endpoint
    that took nothing - editing the file afterwards would then trip the hash guard."""
    path = _jsonl(tmp_path, 2)
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk", lambda *a, **k: None)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert not (tmp_path / "results.jsonl.submission.json").exists()


def test_guard_refuses_a_changed_input_file(tmp_path, monkeypatch):
    """Line numbers against an edited file point at the wrong records, so a resume must stop
    rather than skip some and re-send others - unless overridden by hand."""
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=2)

    path.write_text(path.read_text().replace("PID", "OTHER", 1))
    del sent[:]
    with pytest.raises(ValueError, match="has changed"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True)
    assert len(sent) == 2


def test_guard_refuses_when_the_sidecar_is_gone(tmp_path, monkeypatch):
    """The failure that doubles a city's labels: the sidecar is lost (or the run moves to a
    second machine) and every already-live record is POSTed again."""
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 4

    (tmp_path / "results.jsonl.submitted").unlink()
    del sent[:]
    with pytest.raises(ValueError, match="already"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    # ...and the override is what lets a checked-by-hand case through.
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True)
    assert len(sent) == 4


def test_guard_refuses_an_unreadable_record(tmp_path, monkeypatch):
    """A conflict-markered or truncated record must not decay into 'nothing was sent': with
    the sidecar gone too, that would re-POST the whole city with only a warning printed."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    record_path = tmp_path / "results.jsonl.submission.json"
    (tmp_path / "results.jsonl.submitted").unlink()
    del sent[:]

    for broken in ["<<<<<<< HEAD\n{}\n=======\n", record_path.read_text()[:40]]:
        record_path.write_text(broken)
        with pytest.raises(ValueError, match="not a readable submission record"):
            send_to_ps.process_jsonl_file(str(path), PROD)
        assert sent == []

    # The override starts a fresh record rather than crashing.
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True)
    assert len(sent) == 3
    assert _read_record(tmp_path)["endpoints"][PROD]["submitted_lines"] == 3


def test_record_is_written_atomically(tmp_path, monkeypatch):
    """No half-written record can be left behind, and no temp file either."""
    path = _jsonl(tmp_path, 2)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert not list(tmp_path.glob("*.tmp"))
    raw = (tmp_path / "results.jsonl.submission.json").read_bytes()
    assert raw.endswith(b"}\n") and b"\r\n" not in raw


def test_test_then_prod_is_the_documented_path(tmp_path, monkeypatch):
    """The runbook's sequence: stage everything on a test instance, then submit to prod.
    The sidecar is endpoint-agnostic, so the guard must refuse to silently send prod only
    the remainder, and moving the sidecar aside must be enough - no override."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), TEST)
    assert len(sent) == 3

    del sent[:]
    with pytest.raises(ValueError, match="went to"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    (tmp_path / "results.jsonl.submitted").rename(tmp_path / "results.jsonl.submitted.staging")
    send_to_ps.process_jsonl_file(str(path), PROD, limit=2)
    assert len(sent) == 2
    endpoints = _read_record(tmp_path)["endpoints"]
    assert endpoints[TEST]["submitted_lines"] == 3          # test's count survives
    assert endpoints[PROD]["submitted_lines"] == 2

    # Resuming prod is an ordinary resume, and going back to test sends nothing: test is
    # complete, and a half-done prod sidecar can't be mistaken for it.
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 3
    send_to_ps.process_jsonl_file(str(path), TEST)
    assert len(sent) == 3
    (tmp_path / "results.jsonl.submitted").write_text("1\n")
    with pytest.raises(ValueError, match="ran to completion"):
        send_to_ps.process_jsonl_file(str(path), TEST)


def test_endpoint_spelling_does_not_split_a_campaign(tmp_path, monkeypatch):
    path = _jsonl(tmp_path, 2)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)
    send_to_ps.process_jsonl_file(str(path), "HTTPS://PS.EXAMPLE/ai/submitLabelsOnPano/")
    endpoints = _read_record(tmp_path)["endpoints"]
    assert list(endpoints) == [PROD] and endpoints[PROD]["submitted_lines"] == 2


def test_hash_and_count_handles_a_missing_trailing_newline(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"a":1}\n{"b":2}')
    digest, lines = send_to_ps.hash_and_count(path)
    assert lines == 2 and len(digest) == 64
    path.write_bytes(b'{"a":1}\n{"b":2}\n')
    assert send_to_ps.hash_and_count(path)[1] == 2


def test_dry_run_writes_no_submission_record(tmp_path):
    """A dry run POSTs nothing, so it must leave no trace and stay usable on a stale file."""
    path = _jsonl(tmp_path, 2)
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    assert not (tmp_path / "results.jsonl.submission.json").exists()


def test_guard_refuses_a_record_in_the_old_flat_shape(tmp_path, monkeypatch):
    """A record from before the per-endpoint format is a real record of a real campaign, so
    it must be migrated by hand, not read as 'nothing sent'."""
    path = _jsonl(tmp_path, 2)
    sent = _capture_posts(monkeypatch)
    (tmp_path / "results.jsonl.submission.json").write_text(json.dumps({
        "input_file": "results.jsonl", "sha256": "0" * 64, "total_lines": 2,
        "submitted_lines": 2, "labels_submitted": 2, "endpoints": [PROD]}))
    with pytest.raises(ValueError, match="not a readable submission record"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_failed_and_malformed_lines_stay_out_of_the_sidecar(tmp_path, monkeypatch):
    """Only lines that got a 200 are recorded, so a failed POST or a corrupt line is retried
    next run rather than silently counted as submitted; blank lines are ignored."""
    path = tmp_path / "results.jsonl"
    good = json.dumps(_record([{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}]))
    path.write_text(f"{good}\n\nnot json\n{good}\n{good}\n")
    ok = SimpleNamespace(status_code=200, ok=True, text="")
    calls = []
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk",
                        lambda p, u, k=None: calls.append(p) or (ok if len(calls) != 2 else None))

    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(calls) == 3                                  # the 3 parseable records
    assert _sidecar(tmp_path) == {1, 5}                     # line 4's POST failed
    assert _read_record(tmp_path)["endpoints"][PROD]["submitted_lines"] == 2

    send_to_ps.process_jsonl_file(str(path), PROD)          # the failed one is retried
    assert len(calls) == 4 and _sidecar(tmp_path) == {1, 4, 5}


def test_missing_input_file_is_reported_not_raised(tmp_path, capsys):
    send_to_ps.process_jsonl_file(str(tmp_path / "nope.jsonl"), PROD)
    assert "does not exist" in capsys.readouterr().out


# --- HTTP client: what gets retried ------------------------------------------------------

def _fake_post(monkeypatch, outcomes):
    """requests.post stand-in that yields each outcome in turn: an int status code, or an
    exception instance to raise. Backoff sleeps are recorded, not slept."""
    calls, sleeps = [], []
    outcomes = list(outcomes)

    def post(url, json=None, headers=None, timeout=None):
        calls.append(headers)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(status_code=outcome, json=lambda: {"status": outcome}, text="")
    monkeypatch.setattr(send_to_ps.requests, "post", post)
    monkeypatch.setattr(send_to_ps.time, "sleep", sleeps.append)
    return calls, sleeps


def test_post_sends_the_key_as_a_bearer_token(monkeypatch):
    calls, _ = _fake_post(monkeypatch, [200])
    assert send_to_ps.send_to_project_sidewalk({}, PROD, api_key="SECRET").status_code == 200
    assert calls[0]["Authorization"] == "Bearer SECRET"
    calls, _ = _fake_post(monkeypatch, [200])
    send_to_ps.send_to_project_sidewalk({}, PROD)
    assert "Authorization" not in calls[0]


def test_post_does_not_retry_a_4xx(monkeypatch):
    """401/400 mean the key or payload is wrong; hammering the server won't change that."""
    calls, sleeps = _fake_post(monkeypatch, [401, 200])
    assert send_to_ps.send_to_project_sidewalk({}, PROD) is None
    assert len(calls) == 1 and sleeps == []


def test_post_retries_5xx_and_connection_errors_with_backoff(monkeypatch):
    calls, sleeps = _fake_post(monkeypatch, [503, send_to_ps.requests.exceptions.ConnectionError("down"), 200])
    assert send_to_ps.send_to_project_sidewalk({}, PROD).status_code == 200
    assert len(calls) == 3 and sleeps == send_to_ps.RETRY_BACKOFF_SECONDS


def test_post_gives_up_after_max_attempts(monkeypatch):
    calls, sleeps = _fake_post(monkeypatch, [500] * send_to_ps.MAX_ATTEMPTS + [200])
    assert send_to_ps.send_to_project_sidewalk({}, PROD) is None
    assert len(calls) == send_to_ps.MAX_ATTEMPTS
    assert len(sleeps) == send_to_ps.MAX_ATTEMPTS - 1     # no sleep after the last attempt
