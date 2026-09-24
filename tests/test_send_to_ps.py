"""Unit tests for send_to_ps.py's record transform, endpoint guard and resume sidecar."""
import json
from types import SimpleNamespace

import pytest

import detectors
import main
import send_to_ps
from conftest import make_process_result, make_provenance


def _record(detections, pano_id="PID"):
    return {
        "detections": detections,
        "label_type": "CurbRamp",
        "model_id": "rampnet-model",
        "pano": {"panorama_id": pano_id, "width": 16384, "height": 8192},
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


@pytest.mark.parametrize("source, expected", [
    ("mapillary", "mapillary"),
    ("panoramax", "panoramax"),   # ahead of the server's enum: rejected there, never relabeled here
    ("launch", "gsv"),            # legacy raw streetlevel string
])
def test_transform_pano_source_enum(source, expected):
    assert send_to_ps.transform_pano({"pano_id": "x", "source": source})["source"] == expected


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
    record = json.loads(json.dumps(main.build_output_line(make_process_result(), make_provenance())))
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
        json.dumps(_record([{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}],
                           pano_id=f"PID{i}")) + "\n"
        for i in range(1, count + 1)))
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

    with pytest.raises(SystemExit) as excinfo:
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert excinfo.value.code == 130
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


def _append_records(path, pano_ids):
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        for pano_id in pano_ids:
            f.write(json.dumps(
                _record([{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}],
                        pano_id=pano_id)) + "\n")


def _append(path, count):
    """Add `count` records for panos the file does not already hold - what a gap-fill does."""
    held = sum(1 for line in path.read_text().splitlines() if line.strip())
    _append_records(path, [f"PID{held + i}" for i in range(1, count + 1)])


def test_gap_fill_append_resumes_and_reports_the_new_lines(tmp_path, monkeypatch, capsys):
    """Issue #59: `main.py --gap-fill-only` APPENDS to results.jsonl, so 'submit, gap-fill,
    submit the rest' is the expected sequence. Every recorded line keeps its number, so the
    sidecar still describes the campaign and the new lines must simply be sent."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 3
    before = _read_record(tmp_path)

    _append(path, 2)
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 2                                   # only the appended lines
    assert _sidecar(tmp_path) == {1, 2, 3, 4, 5}
    out = capsys.readouterr().out
    assert "3 line(s) are byte-for-byte unchanged and 2 new line(s) were appended" in out
    assert "none repeating a panorama_id already submitted" in out

    # ...and the record now describes the grown file, so the next run needs no override.
    record = _read_record(tmp_path)
    assert record["sha256"] != before["sha256"]
    assert record["sha256"] == send_to_ps.hash_and_count(path)[0]
    assert (record["total_lines"], record["total_bytes"]) == (5, path.stat().st_size)
    assert record["endpoints"][PROD]["submitted_lines"] == 5
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_a_doubled_file_refuses_even_though_the_prefix_is_intact(tmp_path, monkeypatch):
    """New bytes are not new panos. A file appended to itself (a bad copy, or a re-run after
    the gitignored already_processed.txt was lost) passes the prefix proof byte-for-byte, yet
    every appended line is a pano already live on the server."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 3

    with open(path, 'ab') as f:                             # exactly the same three records
        f.write(path.read_bytes())
    del sent[:]
    with pytest.raises(ValueError, match="3 of the 3 appended line"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_append_repeating_one_submitted_pano_refuses(tmp_path, monkeypatch):
    """One repeat is enough: it would POST that pano's labels a second time, and the server
    has no way to retire them."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)

    _append_records(path, ["PID4", "PID2", "PID5"])         # PID2 is already submitted
    del sent[:]
    with pytest.raises(ValueError, match="first: PID2"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    # Drop the repeated line and the same append is an ordinary resume.
    lines = path.read_text().splitlines(keepends=True)
    del lines[4]                                            # the PID2 repeat
    path.write_text("".join(lines))
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 2
    assert _sidecar(tmp_path) == {1, 2, 3, 4, 5}


def test_guard_refuses_an_edit_that_also_grew_the_file(tmp_path, monkeypatch):
    """A longer file is not automatically an append: the recorded prefix has to still hash
    the same, or the already-submitted lines were edited under their line numbers."""
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=2)

    path.write_text(path.read_text().replace("PID", "A_LONGER_PID", 1))
    _append(path, 1)
    del sent[:]
    with pytest.raises(ValueError, match="no longer hash to the recorded digest"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_guard_refuses_a_truncated_file(tmp_path, monkeypatch):
    """Losing lines is the opposite of an append and still points the sidecar at the wrong
    records."""
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)

    path.write_text("".join(path.read_text().splitlines(keepends=True)[:3]))
    del sent[:]
    with pytest.raises(ValueError, match="truncated"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_a_lost_sidecar_on_a_grown_file_reports_the_growth(tmp_path, monkeypatch):
    """A complete campaign plus a gap-fill is NOT 'nothing left to send': saying so either
    loses the new labels or pushes the user to an override that re-POSTs the whole file."""
    path = _jsonl(tmp_path, 5)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)

    _append(path, 3)
    (tmp_path / "results.jsonl.submitted").unlink()
    with pytest.raises(ValueError, match="covered the whole file as recorded, but .* grown by 3"):
        send_to_ps.process_jsonl_file(str(path), PROD)


def test_a_partial_campaign_on_a_grown_file_is_not_called_complete(tmp_path, monkeypatch):
    """Same shape, but only part of the file ever went to the endpoint: the advice must not
    assert the whole file was covered when two recorded lines were never sent."""
    path = _jsonl(tmp_path, 5)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=3)

    _append(path, 2)
    (tmp_path / "results.jsonl.submitted").unlink()
    with pytest.raises(ValueError, match="missing, truncated") as excinfo:
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert "also grown by 2 line" in str(excinfo.value)
    assert "covered the whole file" not in str(excinfo.value)


def test_append_to_a_record_with_no_byte_length_refuses_with_the_append_hint(tmp_path, monkeypatch):
    """A record written before the append check cannot prove the prefix, so it must keep
    refusing - but name the append case, since that is what the user is looking at."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)

    record_path = tmp_path / "results.jsonl.submission.json"
    legacy = json.loads(record_path.read_text())
    legacy_bytes = legacy.pop("total_bytes")
    record_path.write_text(json.dumps(legacy))
    _append(path, 2)
    del sent[:]
    with pytest.raises(ValueError, match="--prefix-digest 3"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    # ...and that migration works: --prefix-digest over the recorded total_lines reproduces
    # the recorded digest and hands back the total_bytes to add to the record by hand.
    assert send_to_ps.hash_through_lines(path, 3)[:2] == (legacy["sha256"], legacy_bytes)
    legacy["total_bytes"] = legacy_bytes
    record_path.write_text(json.dumps(legacy))
    send_to_ps.process_jsonl_file(str(path), PROD)           # no override needed
    assert len(sent) == 2
    assert json.loads(record_path.read_text())["total_bytes"] == path.stat().st_size


def test_prefix_digest_skips_blank_lines_like_the_submit_loop(tmp_path):
    """total_lines counts non-blank lines, so the migration helper must too - a blank line in
    the file would otherwise make an intact prefix look edited."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"a":1}\n\n{"b":2}\n{"c":3}\n')
    digest, size, blank_digest, blank_size = send_to_ps.hash_through_lines(path, 2)
    assert size == len(b'{"a":1}\n\n{"b":2}\n')
    assert send_to_ps.hash_prefix(path, size)[0] == digest
    assert (blank_digest, blank_size) == (None, None)       # line 3 is not blank
    assert send_to_ps.hash_through_lines(path, 9) == (None, None, None, None)


def test_prefix_digest_offers_the_blank_tail_a_record_would_cover(tmp_path):
    """A record always describes a WHOLE file, so if that file ended with a blank line its
    sha256 covers it - and a count of non-blank lines stops short. The migration helper has
    to hand back a digest the user can actually match, so it prints that candidate too."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"a":1}\n{"b":2}\n\n')
    whole_digest, whole_lines, whole_bytes = send_to_ps.hash_and_count(path)
    digest, size, blank_digest, blank_size = send_to_ps.hash_through_lines(path, whole_lines)
    assert (digest, size) != (whole_digest, whole_bytes)    # the trap the review found
    assert (blank_digest, blank_size) == (whole_digest, whole_bytes)


def test_append_naming_no_pano_refuses(tmp_path, monkeypatch):
    """A pano block with no panorama_id still POSTs (as pano_id null) but is invisible to the
    duplicate check, so an append holding one is not provably new."""
    path = _jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)

    _append_records(path, ["PID4"])
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        nameless = _record([], pano_id="PID5")
        del nameless["pano"]["panorama_id"]
        f.write(json.dumps(nameless) + "\n")
    del sent[:]
    with pytest.raises(ValueError, match="1 of the 2 appended line\\(s\\) name no pano"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_a_record_with_a_byte_length_but_no_line_count_refuses(tmp_path):
    """The line-count check is part of the proof, so a record that cannot supply it fails
    closed rather than resuming on the prefix hash alone."""
    path = _jsonl(tmp_path, 2)
    digest, _, size = send_to_ps.hash_and_count(path)
    _append(path, 1)
    note, reason = send_to_ps.append_check({"sha256": digest, "total_bytes": size}, path, 3,
                                           path.stat().st_size)
    assert note is None and "no line count" in reason


def test_a_record_without_total_lines_is_not_interpolated_as_none(tmp_path):
    """A record with a hash but no line count has no prefix to check; the refusal must say so
    rather than print 'the first None recorded line(s)'."""
    path = _jsonl(tmp_path, 2)
    note, reason = send_to_ps.append_check({"sha256": "0" * 64}, path, 2, path.stat().st_size)
    assert note is None and "None" not in reason


def test_append_onto_a_last_line_with_no_newline_refuses(tmp_path, monkeypatch):
    """The ambiguous case that must stay fail-closed: the recorded bytes end mid-line, so
    anything added ran onto the last submitted record instead of starting a new one."""
    path = tmp_path / "results.jsonl"
    line = json.dumps(_record([{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}]))
    path.write_text(f"{line}\n{line}")                       # no trailing newline
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 2

    _append(path, 1)
    del sent[:]
    with pytest.raises(ValueError, match="do not end with a newline"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []


def test_guard_refuses_when_the_sidecar_is_gone(tmp_path, monkeypatch):
    """The failure that doubles a city's labels: the sidecar is lost (or the run moves to a
    second machine) and every already-live record is POSTed again."""
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 4

    (tmp_path / "results.jsonl.submitted").unlink()
    del sent[:]
    with pytest.raises(ValueError, match="ran to completion"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    # A campaign still in progress gets the lost-sidecar diagnosis instead.
    (tmp_path / "results.jsonl.submission.json").unlink()
    (tmp_path / "results.jsonl.submitted").write_text("1\n2\n")
    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)   # record: 3 of 4
    (tmp_path / "results.jsonl.submitted").write_text("1\n")
    del sent[:]
    with pytest.raises(ValueError, match="missing, truncated"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []
    (tmp_path / "results.jsonl.submitted").unlink()

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


def test_record_is_written_lf_with_no_temp_file_left(tmp_path, monkeypatch):
    """The record is committed, so it must be LF on every platform, and the temp file the
    atomic write goes through must not be left behind."""
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
    digest, lines, size = send_to_ps.hash_and_count(path)
    assert lines == 2 and len(digest) == 64 and size == 15
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


def test_guard_refuses_a_sidecar_that_outgrew_this_endpoint(tmp_path, monkeypatch):
    """The gap a plain 'endpoint unknown' check leaves: prod took a pilot, the full run went
    to test on a fresh sidecar, and 'prod for the rest' would then skip everything. The
    sidecar holding MORE than prod's recorded count while test has a count is the tell."""
    path = _jsonl(tmp_path, 5)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=2)                 # prod pilot
    (tmp_path / "results.jsonl.submitted").rename(tmp_path / "results.jsonl.submitted.prod")
    send_to_ps.process_jsonl_file(str(path), TEST)                          # full test run
    assert len(sent) == 7

    del sent[:]
    with pytest.raises(ValueError, match="says only 2 went to"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    # The override sends only the remainder (nothing here) and the message says so.
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True)
    assert sent == []

    # Restoring prod's own sidecar is an ordinary resume.
    (tmp_path / "results.jsonl.submitted.prod").replace(tmp_path / "results.jsonl.submitted")
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 3
    assert _read_record(tmp_path)["endpoints"][PROD]["submitted_lines"] == 5


def test_guard_refuses_a_changed_min_confidence(tmp_path, monkeypatch):
    """One server, one threshold: the record counts labels at a single --min-confidence,
    and a campaign that switched mid-way would certify a number the server never got."""
    path = _jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=2, min_confidence=0.55)
    del sent[:]
    with pytest.raises(ValueError, match="--min-confidence 0.55"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3)
    assert sent == []
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    assert len(sent) == 2


def test_dry_run_is_exempt_from_the_record_guard(tmp_path, capsys):
    """Step 1 of the runbook must always be safe: a dry run beside a broken record, or on
    a file that has changed since submission, still previews and touches nothing."""
    path = _jsonl(tmp_path, 2)
    record_path = tmp_path / "results.jsonl.submission.json"
    record_path.write_text("<<<<<<< HEAD\n")
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    assert "Successfully processed:        2" in capsys.readouterr().out
    assert record_path.read_text() == "<<<<<<< HEAD\n"


def test_hash_and_count_counts_only_non_blank_lines(tmp_path):
    """total_lines is what a complete sidecar reaches, so blank lines - which the submit
    loop skips - must not count, or a finished campaign reads as an unfinished one."""
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"a":1}\r\n\r\n{"b":2}\n\n')
    assert send_to_ps.hash_and_count(path)[1] == 2


# --- Position gate (SidewalkWebpage#5361) -----------------------------------------------

def _mapillary_jsonl(tmp_path, count, name="results.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps({**_record([]), "pano": {**_record([])["pano"], "source": "mapillary",
                                                                "sequence_id": "A"}}) + "\n"
                            for _ in range(count)))
    return path


def _write_check(path, flagged=(), digest=None):
    import position_check
    check = {"checked_at": "2026-09-16T00:00:00Z", "flagged_sequences": list(flagged),
             "results_sha256": digest or position_check.file_sha256(path)}
    position_check.check_path_for(path).write_text(json.dumps(check), encoding="utf-8")


def test_position_gate_refuses_unchecked_stale_and_flagged_mapillary_files(tmp_path, monkeypatch):
    """The only way to submit a Mapillary file whose pano positions were not checked, were
    checked before it changed, or are flagged is to type --ignore-position-check."""
    path = _mapillary_jsonl(tmp_path, 3)
    sent = _capture_posts(monkeypatch)

    with pytest.raises(ValueError, match="no position check"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    _write_check(path, flagged=["A"])
    with pytest.raises(ValueError, match="reposition.py"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    _write_check(path, digest="0" * 64)
    with pytest.raises(ValueError, match="different version"):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []

    # Dry runs are exempt (they POST nothing); the override warns and sends.
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    assert sent == []
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_position_check=True)
    assert len(sent) == 3
    # ...and the record says the gate was bypassed, and why.
    state = json.loads(send_to_ps.submission_record_path(str(path)).read_text())["endpoints"][
        send_to_ps.canonical_endpoint(PROD)]
    assert state["position_check"]["overridden"] and "different version" in state["position_check"]["reason"]


def test_position_gate_passes_a_clean_matching_check_and_records_it(tmp_path, monkeypatch):
    path = _mapillary_jsonl(tmp_path, 2)
    sent = _capture_posts(monkeypatch)
    _write_check(path)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 2
    record = json.loads(send_to_ps.submission_record_path(str(path)).read_text())
    state = record["endpoints"][send_to_ps.canonical_endpoint(PROD)]
    assert state["position_check"]["flagged"] == 0
    assert state["position_check"]["results_sha256"] == record["sha256"]

    # A reposition.py output is gated on the check written beside IT.
    out = _mapillary_jsonl(tmp_path, 2, name="results.check.jsonl")
    with pytest.raises(ValueError, match="results.check.position_check.json missing"):
        send_to_ps.process_jsonl_file(str(out), PROD)
    _write_check(out)
    send_to_ps.process_jsonl_file(str(out), PROD)
    assert len(sent) == 4


def test_position_gate_ignores_non_mapillary_files(tmp_path, monkeypatch):
    """GSV/Panoramax carry one position; there is nothing to switch, so nothing to gate."""
    path = _jsonl(tmp_path, 2)   # the GSV fixture, no check beside it
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == 2


# --- Band campaigns (issue #20: a live city gets the labels a lower threshold adds) --------

def _banded_jsonl(tmp_path, count):
    """Every record has one 0.9 label; odd records also carry one in [0.3, 0.55)."""
    path = tmp_path / "results.jsonl"
    lines = []
    for i in range(1, count + 1):
        dets = [{"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9}]
        if i % 2:
            dets.append({"x_normalized": 0.2, "y_normalized": 0.6, "confidence": 0.4})
        dets.append({"x_normalized": 0.1, "y_normalized": 0.1, "confidence": 0.2})  # sub-band
        lines.append(json.dumps(_record(dets, pano_id=f"PID{i}")) + "\n")
    path.write_text("".join(lines))
    return path


def test_transform_record_band_is_half_open():
    payload = send_to_ps.transform_record(_record([
        {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.55},   # excluded: == max
        {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.549},
        {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.3},    # included: == min
        {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.29},
    ]), min_confidence=0.3, max_confidence=0.55)
    assert [l["confidence"] for l in payload["labels"]] == [0.549, 0.3]


def test_band_ships_only_the_new_labels_and_skips_empty_records(tmp_path, monkeypatch):
    """After a complete 0.55 campaign, the 0.3-0.55 band POSTs exactly the band labels,
    for exactly the records that have one, from its own sidecar, and the record says so."""
    path = _banded_jsonl(tmp_path, 6)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    assert len(sent) == 6
    del sent[:]

    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert [p["pano"]["pano_id"] for p in sent] == ["PID1", "PID3", "PID5"]
    assert all([l["confidence"] for l in p["labels"]] == [0.4] for p in sent)
    # The band's sidecar marks every line handled, POSTed or not; the base one is untouched.
    band_sidecar = tmp_path / "results.jsonl.band-0.3-0.55.submitted"
    assert send_to_ps.load_submitted_lines(band_sidecar) == set(range(1, 7))
    assert _sidecar(tmp_path) == set(range(1, 7))

    state = _read_record(tmp_path)["endpoints"][PROD]
    assert state["bands"]["0.3-0.55"]["submitted_lines"] == 6
    assert state["bands"]["0.3-0.55"]["labels_submitted"] == 3
    assert state["min_confidence"] == 0.3            # the server now holds down to 0.3
    assert state["labels_submitted"] == 6 + 3
    assert state["submitted_lines"] == 6             # the base campaign's count is kept

    # Re-running the band sends nothing more.
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []


def test_band_resumes_from_its_own_sidecar(tmp_path, monkeypatch):
    path = _banded_jsonl(tmp_path, 6)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55,
                                  limit=3)                      # lines 1-3: PID1, PID3 POSTed
    assert [p["pano"]["pano_id"] for p in sent] == ["PID1", "PID3"]
    state = _read_record(tmp_path)["endpoints"][PROD]
    assert state["bands"]["0.3-0.55"]["submitted_lines"] == 3
    assert state["min_confidence"] == 0.55           # not complete yet: server still at 0.55
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert [p["pano"]["pano_id"] for p in sent] == ["PID1", "PID3", "PID5"]
    assert _read_record(tmp_path)["endpoints"][PROD]["min_confidence"] == 0.3


@pytest.mark.parametrize("setup, message", [
    ("none", "records no campaign"),
    ("incomplete", "COMPLETE campaign"),
    ("wrong_max", "--max-confidence 0.55, not 0.5"),
    ("changed_file", "is not the file"),
])
def test_band_refusals(tmp_path, monkeypatch, setup, message):
    """A band is allowed only on top of a complete campaign at exactly --max-confidence,
    on the unchanged file; anything else would double labels or leave a gap."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    max_conf = 0.55
    if setup == "incomplete":
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55, limit=2)
    elif setup in ("wrong_max", "changed_file"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
        if setup == "wrong_max":
            max_conf = 0.5
        else:
            with open(path, "a") as f:
                f.write(json.dumps(_record([], pano_id="PID9")) + "\n")
    del sent[:]
    with pytest.raises(ValueError, match=message):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3,
                                      max_confidence=max_conf)
    assert sent == []


def test_band_sidecar_from_another_endpoint_is_refused(tmp_path, monkeypatch):
    """Test then prod: the band sidecar is endpoint-agnostic like the base one, so the
    prod band must start from its own sidecar, not skip the lines test took."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    for url in (TEST, PROD):
        send_to_ps.process_jsonl_file(str(path), url, min_confidence=0.55)
        (tmp_path / "results.jsonl.submitted").unlink()
    send_to_ps.process_jsonl_file(str(path), TEST, min_confidence=0.3, max_confidence=0.55)
    del sent[:]
    with pytest.raises(ValueError, match="Move the band sidecar aside"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []
    (tmp_path / "results.jsonl.band-0.3-0.55.submitted").unlink()
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert [p["pano"]["pano_id"] for p in sent] == ["PID1", "PID3"]
    record = _read_record(tmp_path)
    assert record["endpoints"][TEST]["min_confidence"] == 0.3
    assert record["endpoints"][PROD]["min_confidence"] == 0.3


def test_base_resume_after_a_band_never_suggests_an_upward_band(tmp_path, monkeypatch):
    """Once a server holds 0.3, a plain run at 0.55 is refused WITHOUT a band hint — a band
    is [min, max) and only ever adds a lower tier, so there is no route upward. A plain run
    at 0.3 with the base sidecar gone is still caught as a lost sidecar."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    del sent[:]
    with pytest.raises(ValueError, match="already holds every label") as excinfo:
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    # The bug this guards: the hint used to be built unconditionally, so it printed the
    # impossible "--min-confidence 0.55 --max-confidence 0.3" (min must be below max).
    assert "--max-confidence 0.3" not in str(excinfo.value)
    (tmp_path / "results.jsonl.submitted").unlink()
    with pytest.raises(ValueError, match="accounts for only 0"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3)
    assert sent == []


def test_base_resume_below_the_recorded_tier_still_names_the_band(tmp_path, monkeypatch):
    """The downward case keeps the hint: a server holding 0.55 told to run at 0.3 is sent
    to the band route, which is the one thing that adds a tier without re-sending."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    del sent[:]
    with pytest.raises(ValueError,
                       match=r"--min-confidence 0\.3 --max-confidence 0\.55"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3)
    assert sent == []


def test_band_dry_run_prints_only_band_payloads_and_records_nothing(tmp_path, monkeypatch, capsys):
    path = _banded_jsonl(tmp_path, 4)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    before = (tmp_path / "results.jsonl.submission.json").read_text()
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True, min_confidence=0.3,
                                  max_confidence=0.55)
    out = capsys.readouterr().out
    assert out.count('"pano_id"') == 2
    assert not (tmp_path / "results.jsonl.band-0.3-0.55.submitted").exists()
    assert (tmp_path / "results.jsonl.submission.json").read_text() == before


# --- Band campaigns: the ways a second campaign could duplicate live labels ---------------

def _append_banded(path, pano_ids):
    """Append panos that carry a band label, i.e. what a gap-fill adds to a banded file."""
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        for pano_id in pano_ids:
            f.write(json.dumps(_record([
                {"x_normalized": 0.5, "y_normalized": 0.5, "confidence": 0.9},
                {"x_normalized": 0.2, "y_normalized": 0.6, "confidence": 0.4},
            ], pano_id=pano_id)) + "\n")


def test_band_re_run_after_a_gap_fill_refuses_instead_of_resending(tmp_path, monkeypatch):
    """The completed-band exemption must be judged against the file as it is NOW.

    Sequence that used to duplicate: band completes -> gap-fill (#32) appends panos -> the
    base campaign ships them at the band's floor via the append path -> the band command is
    re-run by accident. The band sidecar never saw the appended line numbers, so their
    [0.3, 0.55) labels were POSTed a second time, and PS is insert-only.
    """
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert _read_record(tmp_path)["endpoints"][PROD]["min_confidence"] == 0.3

    _append_banded(path, ["PID5", "PID6"])
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3)   # the append path
    assert {p["pano"]["pano_id"] for p in sent} == {"PID5", "PID6"}

    del sent[:]
    with pytest.raises(ValueError, match="belong to the BASE campaign"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []


def test_band_over_a_file_with_nothing_in_the_band_is_refused(tmp_path, monkeypatch):
    """A legacy file holds nothing below 0.55, so a band over it POSTs nothing, marks every
    line done, and then records the endpoint as holding 0.3 - a record that lies about a
    live server. That is the first command anyone would try on a pre-storage-floor city."""
    path = _jsonl(tmp_path, 4)          # every record has one 0.9 label and nothing else
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    del sent[:]
    with pytest.raises(ValueError, match="holds no labels at all in"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []
    # The record must be exactly as the base campaign left it.
    assert _read_record(tmp_path)["endpoints"][PROD]["min_confidence"] == 0.55
    assert "bands" not in _read_record(tmp_path)["endpoints"][PROD]


def test_band_refusal_names_the_storage_floor_when_the_run_records_one(tmp_path, monkeypatch):
    path = _jsonl(tmp_path, 2)
    (tmp_path / "manifest.json").write_text(json.dumps({"detection_storage_floor": 0.55}))
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    with pytest.raises(ValueError, match="detection_storage_floor is 0.55"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)


def test_lost_band_sidecar_on_a_complete_band_is_a_noop(tmp_path, monkeypatch, capsys):
    """Nothing is left to send, so refusing would only push the user to the override - which
    with an empty sidecar would re-POST the entire band."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    before = (tmp_path / "results.jsonl.submission.json").read_text()
    (tmp_path / "results.jsonl.band-0.3-0.55.submitted").unlink()

    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []
    assert "Nothing to do" in capsys.readouterr().out
    assert (tmp_path / "results.jsonl.submission.json").read_text() == before


def test_lost_band_sidecar_noop_is_not_downgraded_by_the_override(tmp_path, monkeypatch):
    """--ignore-submission-guard turns refusals into warnings and carries on. Here that
    would re-POST the whole band, so the completion stop must not be a refusal at all."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    (tmp_path / "results.jsonl.band-0.3-0.55.submitted").unlink()
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55,
                                  ignore_guard=True)
    assert sent == []


def test_a_second_band_chains_below_the_first(tmp_path, monkeypatch):
    """Once 0.3-0.55 has landed the endpoint holds 0.3, so the next tier down is
    --max-confidence 0.3. Every record in the fixture carries a 0.2 label."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    del sent[:]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.2, max_confidence=0.3)
    assert len(sent) == 4 and all(len(p["labels"]) == 1 for p in sent)
    state = _read_record(tmp_path)["endpoints"][PROD]
    assert state["min_confidence"] == 0.2
    assert set(state["bands"]) == {"0.3-0.55", "0.2-0.3"}
    # 4 base (0.9) + 2 first band (odd records only) + 4 second band (0.2 on every record).
    assert state["labels_submitted"] == 10


def test_band_completion_adds_its_labels_exactly_once(tmp_path, monkeypatch):
    """`was_complete` is what stops a re-run from adding the band's labels to the endpoint
    total a second time."""
    path = _banded_jsonl(tmp_path, 4)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    total = _read_record(tmp_path)["endpoints"][PROD]["labels_submitted"]
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert _read_record(tmp_path)["endpoints"][PROD]["labels_submitted"] == total


def test_a_band_forced_onto_a_file_with_no_record_still_writes_a_readable_one(tmp_path,
                                                                             monkeypatch):
    """Reachable only via the override, but the record it leaves must not be one the next
    run refuses as unreadable - that would strand the campaign with no way back."""
    path = _banded_jsonl(tmp_path, 4)
    _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55,
                                  ignore_guard=True)
    record_path = tmp_path / "results.jsonl.submission.json"
    send_to_ps.load_submission_record(record_path)          # must not raise
    state = json.loads(record_path.read_text())["endpoints"][PROD]
    assert state["submitted_lines"] == 0                    # honest: no base campaign ran
    assert state["bands"]["0.3-0.55"]["submitted_lines"] == 4


def test_a_band_on_a_record_with_no_threshold_says_so(tmp_path, monkeypatch):
    """A record from before min_confidence was tracked cannot say what tier the server
    holds, so it must be repaired by hand rather than guessed at."""
    path = _banded_jsonl(tmp_path, 4)
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    record_path = tmp_path / "results.jsonl.submission.json"
    record = json.loads(record_path.read_text())
    del record["endpoints"][PROD]["min_confidence"]
    record_path.write_text(json.dumps(record))
    del sent[:]
    with pytest.raises(ValueError, match="predates threshold tracking"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []


# --- The nadir / camera-rig mask ----------------------------------------------------------

def _at_dip(deg, conf=0.4):
    """A detection `deg` degrees below the horizon: y_normalized 0.5 is the horizon."""
    return {"x_normalized": 0.5, "y_normalized": 0.5 + deg / 180.0, "confidence": conf}


def test_rig_mask_threshold_sits_between_the_measured_true_and_false_populations():
    """Laurens validations, 2026-09-22: the shallowest label a validator marked correct is
    46.1 deg below the horizon; the shallowest rig false positive is 51.7 deg. The constant
    must separate them, or it is either dropping real ramps or keeping vehicle roofs."""
    assert 46.1 < detectors.NADIR_MASK_DEG < 51.7
    assert not detectors.on_camera_rig(0.5 + 46.1 / 180.0)   # a real ramp, 2.5 m out
    assert detectors.on_camera_rig(0.5 + 51.7 / 180.0)       # the rig, 2.1 m out
    assert not detectors.on_camera_rig(0.5)                  # the horizon
    assert detectors.on_camera_rig(1.0)                      # straight down


def test_transform_record_drops_detections_on_the_camera_rig():
    payload = send_to_ps.transform_record(_record([
        _at_dip(10), _at_dip(40), _at_dip(46), _at_dip(52), _at_dip(60),
    ]), min_confidence=0.30)
    dips = [round((lbl["pano_y"] / 8192 - 0.5) * 180) for lbl in payload["labels"]]
    assert dips == [10, 40, 46]


def test_the_rig_mask_can_be_turned_off_to_reconstruct_an_older_campaign():
    """What is already live is already live: a record derived from a campaign that shipped
    before the mask existed has to count what that campaign actually sent."""
    rec = _record([_at_dip(10), _at_dip(60)])
    assert len(send_to_ps.transform_record(rec, 0.30)["labels"]) == 1
    assert len(send_to_ps.transform_record(rec, 0.30, mask_rig=False)["labels"]) == 2


def test_rig_detections_are_never_posted(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        _record([_at_dip(20, 0.9)], "A"),                  # a real ramp
        _record([_at_dip(60, 0.9)], "B"),                  # a roof rack, above threshold
        _record([_at_dip(20, 0.9), _at_dip(60, 0.9)], "C"),
    ]), encoding="utf-8", newline="\n")
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    # B has nothing left once the rig detection is dropped, but it is still submitted as a
    # "checked, nothing found" pano — exactly like a record whose peaks are all sub-threshold.
    assert [(p["pano"]["pano_id"], len(p["labels"])) for p in sent] == [("A", 1), ("B", 0), ("C", 1)]


def test_count_band_labels_in_file_honours_the_mask_both_ways(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        _record([_at_dip(20, 0.4), _at_dip(60, 0.4)], "A"),
        _record([_at_dip(60, 0.4)], "B"),
    ]), encoding="utf-8", newline="\n")
    assert send_to_ps.count_band_labels_in_file(path, 0.30, 0.55) == 1
    assert send_to_ps.count_band_labels_in_file(path, 0.30, 0.55, mask_rig=False) == 3


def test_a_band_that_is_all_rig_detections_is_refused(tmp_path, monkeypatch):
    """The zero-band-labels guard counts what would actually SHIP, so a file whose entire
    band is vehicle roof is caught by it rather than completing silently."""
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        _record([_at_dip(20, 0.9), _at_dip(60, 0.4)], "A"),
        _record([_at_dip(20, 0.9), _at_dip(60, 0.4)], "B"),
    ]), encoding="utf-8", newline="\n")
    sent = _capture_posts(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.55)
    del sent[:]
    with pytest.raises(ValueError, match="holds no labels at all in"):
        send_to_ps.process_jsonl_file(str(path), PROD, min_confidence=0.3, max_confidence=0.55)
    assert sent == []
