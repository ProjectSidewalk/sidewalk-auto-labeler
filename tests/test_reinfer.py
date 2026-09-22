"""reinfer: the re-inference verifier and the band-ready file it builds (issue #20).

A run made before the storage floor (#28) holds nothing below 0.55, so its band has to
come from a fresh re-inference — and re-inference produces *new numbers*. These tests
cover the one thing that matters about that: a band built from the new file must never
POST a label the server already holds, because PS is insert-only (SidewalkWebpage#5382)
and a duplicate can never be retired.

No network, no GPU, no torch: `verify` and the band writer are pure file logic.
"""
import json

import pytest

import reinfer
import send_to_ps

PROD = "https://ps.example/ai/submitLabelsOnPano"


def _record(detections, pano_id="PID", **pano):
    block = {"panorama_id": pano_id, "width": 16384, "height": 8192,
             "lat": 37.5, "lng": -77.4, "camera_heading": 90.0}
    block.update(pano)
    return {"detections": detections, "label_type": "CurbRamp", "model_id": "rampnet-model",
            "pano": block}


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8",
                    newline="\n")
    return path


def _det(conf, x=0.5, y=0.5):
    return {"x_normalized": x, "y_normalized": y, "confidence": conf}


def _old_and_new(tmp_path, old_records, new_records, submit=True):
    """A legacy results.jsonl (with a completed 0.55 campaign) and its re-inferred sibling."""
    old = _write(tmp_path / "results.jsonl", old_records)
    new = _write(tmp_path / "results.f01.jsonl", new_records)
    if submit:
        send_to_ps.send_to_project_sidewalk = lambda payload, url, key=None: _OK
        send_to_ps.process_jsonl_file(str(old), PROD, min_confidence=0.55)
    return old, new


class _OKResponse:
    status_code, ok, text = 200, True, ""


_OK = _OKResponse()


@pytest.fixture(autouse=True)
def _no_posts(monkeypatch):
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk",
                        lambda payload, url, key=None: _OK)


# --- verify -------------------------------------------------------------------------------

def test_verify_reports_exact_when_the_new_file_reproduces(tmp_path):
    old, new = _old_and_new(tmp_path,
                            [_record([_det(0.9)], "A"), _record([_det(0.8)], "B")],
                            [_record([_det(0.9), _det(0.4, x=0.2)], "A"),
                             _record([_det(0.8), _det(0.35, x=0.3)], "B")])
    summary, mismatches, carry = reinfer.verify(old, new, floor=0.55, band_floor=0.30)
    assert (summary["exact"], summary["mismatch"], summary["pano_drift"]) == (2, 0, 0)
    assert summary["band_labels"] == 2 and mismatches == [] and carry == set()


def test_a_peak_that_slips_below_the_tier_does_not_reproduce(tmp_path):
    """The Richmond case: 0.550011 in July, 0.549993 now. The pixel is live, so a naive
    band would insert it a second time."""
    old, new = _old_and_new(tmp_path, [_record([_det(0.550011)], "A")],
                            [_record([_det(0.549993)], "A")])
    summary, _, carry = reinfer.verify(old, new, floor=0.55)
    assert summary["mismatch"] == 1 and carry == {"A"}


def test_a_peak_that_moves_while_staying_above_the_tier_does_not_reproduce(tmp_path):
    old, new = _old_and_new(tmp_path, [_record([_det(0.87, x=0.125, y=0.1888)], "A")],
                            [_record([_det(0.87, x=0.126, y=0.1840)], "A")])
    summary, _, carry = reinfer.verify(old, new, floor=0.55)
    assert summary["mismatch"] == 1 and carry == {"A"}


@pytest.mark.parametrize("field,value", [
    ("width", 8192), ("height", 4096), ("lat", 37.6), ("lng", -77.5), ("camera_heading", 91.0)])
def test_pano_drift_alone_stops_a_pano_reproducing(tmp_path, field, value):
    """A pano whose frame moved fails even when every detection is bit-identical: width and
    height decide the pixel key, and position is what each label inherits (SW#5361)."""
    old, new = _old_and_new(tmp_path, [_record([_det(0.9)], "A")],
                            [_record([_det(0.9)], "A", **{field: value})])
    summary, mismatches, carry = reinfer.verify(old, new, floor=0.55)
    assert summary["pano_drift"] == 1 and carry == {"A"}
    assert field in mismatches[0][1][0]


def test_a_pano_with_no_labels_at_all_is_still_checked_for_drift(tmp_path):
    """Nothing to compare on the pixel key, so this pano used to pass `exact` whatever
    happened to its metadata."""
    old, new = _old_and_new(tmp_path, [_record([_det(0.2)], "A")],
                            [_record([_det(0.2)], "A", lat=37.9)])
    summary, _, carry = reinfer.verify(old, new, floor=0.55)
    assert summary["pano_drift"] == 1 and carry == {"A"}


def test_verify_compares_at_the_tier_the_record_holds_not_the_benchmark_constant(tmp_path):
    old = _write(tmp_path / "results.jsonl", [_record([_det(0.9)], "A")])
    send_to_ps.process_jsonl_file(str(old), PROD, min_confidence=0.7)
    record = send_to_ps.load_submission_record(send_to_ps.submission_record_path(old))
    assert reinfer.server_tier(record) == 0.7


def test_server_tier_refuses_a_record_holding_two_thresholds(tmp_path):
    """Two servers at different tiers have no single "what is live" to verify against, so
    guessing one would silently pick the wrong comparison for the other."""
    record = {"endpoints": {PROD: {"submitted_lines": 1, "min_confidence": 0.55},
                            "https://other.example/ai/x": {"submitted_lines": 1,
                                                           "min_confidence": 0.7}}}
    with pytest.raises(SystemExit, match="more than one threshold"):
        reinfer.server_tier(record)


def test_server_tier_falls_back_to_the_benchmark_constant_without_a_record(tmp_path):
    assert reinfer.server_tier({}) == reinfer.BENCHMARK_CONFIDENCE


# --- the band-ready file ------------------------------------------------------------------

def test_band_file_takes_the_old_record_for_every_pano_that_did_not_reproduce(tmp_path):
    old, new = _old_and_new(
        tmp_path,
        [_record([_det(0.9)], "A"), _record([_det(0.550011)], "B")],
        [_record([_det(0.9), _det(0.4, x=0.2)], "A"),
         _record([_det(0.549993), _det(0.45, x=0.7)], "B")])
    _, _, carry = reinfer.verify(old, new, floor=0.55)
    band = tmp_path / "results.band.jsonl"
    assert reinfer.write_band_file(old, new, band, carry, 0.55) == 1

    by_id = {r["pano"]["panorama_id"]: r for r in reinfer.read_records(band)}
    # A reproduced: the new record, band label included.
    assert [d["confidence"] for d in by_id["A"]["detections"]] == [0.9, 0.4]
    # B carried over: the old record, so its 0.549993 and 0.45 are both absent and the
    # 0.550011 that is actually live is what the file describes.
    assert [d["confidence"] for d in by_id["B"]["detections"]] == [0.550011]
    assert send_to_ps.count_band_labels_in_file(band, 0.30, 0.55) == 1


def test_band_file_reproduces_the_live_label_set_exactly(tmp_path):
    """The invariant the whole design rests on: the band file's labels at the server's tier
    ARE the ones already live, so nothing it ships can be a duplicate."""
    old, new = _old_and_new(
        tmp_path,
        [_record([_det(0.9)], "A"), _record([_det(0.550011)], "B"), _record([_det(0.8)], "C")],
        [_record([_det(0.9), _det(0.4, x=0.2)], "A"),
         _record([_det(0.549993)], "B"),
         _record([_det(0.8), _det(0.31, x=0.9)], "C", lat=37.99)])
    _, _, carry = reinfer.verify(old, new, floor=0.55)
    band = tmp_path / "results.band.jsonl"
    reinfer.write_band_file(old, new, band, carry, 0.55)

    def keys(path, lo, hi=None):
        return {(r["pano"]["panorama_id"], px) for r in reinfer.read_records(path)
                for px in reinfer.pixel_set(r, lo)} if hi is None else None

    assert keys(band, 0.55) == keys(old, 0.55)
    # B slipped below the tier and C's pano moved, so only A contributes a band label.
    assert send_to_ps.count_band_labels_in_file(band, 0.30, 0.55) == 1


def test_band_file_is_identical_to_the_new_file_when_everything_reproduces(tmp_path):
    old, new = _old_and_new(tmp_path,
                            [_record([_det(0.9)], "A"), _record([_det(0.8)], "B")],
                            [_record([_det(0.9), _det(0.4, x=0.2)], "A"),
                             _record([_det(0.8)], "B")])
    _, _, carry = reinfer.verify(old, new, floor=0.55)
    band = tmp_path / "results.band.jsonl"
    assert reinfer.write_band_file(old, new, band, carry, 0.55) == 0
    assert band.read_text() == new.read_text()


def test_write_band_file_deletes_the_output_if_the_invariant_fails(tmp_path):
    """Belt and braces: the property is asserted against what was actually written, and a
    file that fails it must not be left where someone could ship from it.

    Forced here by passing an empty carry_over for a pano that genuinely drifted, which is
    what a future bug in `verify` would look like from the writer's side.
    """
    old, new = _old_and_new(tmp_path, [_record([_det(0.9)], "A")],
                            [_record([_det(0.9, x=0.6)], "A")])
    band = tmp_path / "results.band.jsonl"
    with pytest.raises(SystemExit, match="does not reproduce"):
        reinfer.write_band_file(old, new, band, carry_over=set(), tier=0.55)
    assert not band.exists()


# --- the derived submission record ---------------------------------------------------------

def test_derived_record_lets_the_ordinary_band_guard_pass(tmp_path, monkeypatch):
    """The point of the whole exercise: no --ignore-submission-guard."""
    old, new = _old_and_new(
        tmp_path,
        [_record([_det(0.9)], "A"), _record([_det(0.550011)], "B")],
        [_record([_det(0.9), _det(0.4, x=0.2)], "A"), _record([_det(0.549993)], "B")])
    _, _, carry = reinfer.verify(old, new, floor=0.55)
    band = tmp_path / "results.band.jsonl"
    reinfer.write_band_file(old, new, band, carry, 0.55)
    record_path, endpoints = reinfer.write_derived_record(old, band, 0.55)
    assert list(endpoints) == [PROD]
    assert json.loads(record_path.read_text())["derived_from"]["input_file"] == "results.jsonl"

    sent = []
    monkeypatch.setattr(send_to_ps, "send_to_project_sidewalk",
                        lambda payload, url, key=None: sent.append(payload) or _OK)
    send_to_ps.process_jsonl_file(str(band), PROD, min_confidence=0.30, max_confidence=0.55)
    # Only A had anything in the band; B was carried over and so is skipped without a POST.
    assert [p["pano"]["pano_id"] for p in sent] == ["A"]
    assert [lbl["confidence"] for lbl in sent[0]["labels"]] == [0.4]


def test_derived_record_refuses_an_incomplete_base_campaign(tmp_path):
    old = _write(tmp_path / "results.jsonl",
                 [_record([_det(0.9)], "A"), _record([_det(0.9)], "B")])
    new = _write(tmp_path / "results.f01.jsonl",
                 [_record([_det(0.9)], "A"), _record([_det(0.9)], "B")])
    send_to_ps.process_jsonl_file(str(old), PROD, min_confidence=0.55, limit=1)
    band = tmp_path / "results.band.jsonl"
    reinfer.write_band_file(old, new, band, set(), 0.55)
    with pytest.raises(SystemExit, match="COMPLETE campaign"):
        reinfer.write_derived_record(old, band, 0.55)


def test_derived_record_refuses_when_the_label_count_disagrees(tmp_path):
    """If the band file does not hold exactly the recorded number of labels at the tier, it
    is not the file that campaign describes, whatever the pixel sets said."""
    old, new = _old_and_new(tmp_path, [_record([_det(0.9)], "A")], [_record([_det(0.9)], "A")])
    band = _write(tmp_path / "results.band.jsonl",
                  [_record([_det(0.9), _det(0.95, x=0.8)], "A")])
    with pytest.raises(SystemExit, match="does not describe that campaign"):
        reinfer.write_derived_record(old, band, 0.55)


def test_derived_record_refuses_a_file_with_no_campaign(tmp_path):
    old, new = _old_and_new(tmp_path, [_record([_det(0.9)], "A")], [_record([_det(0.9)], "A")],
                            submit=False)
    band = tmp_path / "results.band.jsonl"
    reinfer.write_band_file(old, new, band, set(), 0.55)
    with pytest.raises(SystemExit, match="no submission record"):
        reinfer.write_derived_record(old, band, 0.55)
