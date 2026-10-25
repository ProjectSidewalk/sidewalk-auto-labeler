"""Committed producer tables never republish an email-shaped Panoramax producer name
(issue #149; #156 review): thinning_experiment's crosscheck producers.csv and
panoramax_lyon_figures' rig-producers table. No network, no model, no torch."""
import json

import panoramax_lyon_figures as plf
import thinning_experiment as te

EMAIL = "jane.doe@example.org"


def test_mask_producer_masks_only_email_shaped_names():
    for mask in (te.mask_producer, plf.mask_producer):
        assert mask(EMAIL) == "j***@***"
        assert mask(" " + EMAIL + " ") == "j***@***"
        for keep in ("grand lyon", "mike en gyroroue", "a@b", None, ""):
            assert mask(keep) == keep


def test_crosscheck_producer_rows_mask_the_name():
    scan = {"p1": (45.0, 4.0, "2026-01-01", 1.0), "p2": (45.0, 4.001, "2025-01-01", 1.0)}
    records = [{"pano": {"panorama_id": p, "copyright": EMAIL}, "detections": []} for p in scan]
    rows = te.producer_rows(scan, records, lambda s, sp: list(s), 10, 5, (0.3,))
    assert [r["producer"] for r in rows] == ["j***@***"]
    assert EMAIL not in json.dumps(rows)


def test_rig_producers_masks_the_name(tmp_path):
    pano = {"panorama_id": "p", "camera_make": "GoPro", "camera_model": "Max", "width": 5760,
            "height": 2880, "capture_date": "2026-01-01", "copyright": EMAIL, "sequence_id": "s",
            "camera_pitch": 0, "camera_roll": 0}
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps({"pano": pano, "detections": []}) + "\n", encoding="utf-8")
    plf.rig_producers(results, min_panos=1, out_dir=tmp_path / "out")
    text = (tmp_path / "out" / "rig_producer_detections.csv").read_text(encoding="utf-8")
    assert "j***@***" in text and EMAIL not in text
