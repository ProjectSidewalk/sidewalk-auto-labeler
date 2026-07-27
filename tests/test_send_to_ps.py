"""Unit tests for send_to_ps.py's record transform and resume sidecar."""
import json

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
