"""Every rebuild of Project Sidewalk's pixel key agrees with what send_to_ps ships (PR #152).

A detection stored at x just below, or at, 1.0 (a sub-cell decode under --border wrap) ships
at pano_x 0, so each analysis that joins a run to the server's labels on (pano_id, pano_x,
pano_y) must key it at column 0 too, or the join silently misses it. One synthetic
results.jsonl, every key builder. No network, no GPU.
"""
import json

import pytest

import agree_rate
import provenance_gate
import send_to_ps
import server_agree_check

W, H = 16384, 8192
SEAM_XS = (0.99998, 1.0)


def _results(tmp_path, xs=SEAM_XS):
    rec = {"detections": [{"x_normalized": x, "y_normalized": 0.5, "confidence": 0.9}
                          for x in xs],
           "label_type": "CurbRamp", "model_id": "rampnet-model",
           "pano": {"panorama_id": "P", "width": W, "height": H, "lat": 47.6, "lng": -122.3,
                    "camera_heading": 0.0}}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8", newline="\n")
    return path, rec


def test_send_to_ps_ships_the_seam_at_column_zero(tmp_path):
    _, rec = _results(tmp_path)
    assert {(l["pano_x"], l["pano_y"]) for l in send_to_ps.transform_record(rec)["labels"]} \
        == {(0, H // 2)}


def test_agree_rate_contamination_keys(tmp_path):
    path, _ = _results(tmp_path)
    assert agree_rate.scan_run(path)[2] == {("P", 0, H // 2)}


def test_server_agree_check_tier_keys(tmp_path):
    path, _ = _results(tmp_path)
    keys, _ = server_agree_check.load_tiers([path])
    assert set(keys) == {("P", 0, H // 2)}
    assert server_agree_check.label_tier({"pano_id": "P", "pano_x": 0, "pano_y": H // 2},
                                         keys) != server_agree_check.TIERS[4]


def test_provenance_gate_detection_keys(tmp_path):
    path, _ = _results(tmp_path)
    w, h, dets = provenance_gate.load_run(path)["P"]
    assert [(px, py) for px, py, _ in dets] == [(0, H // 2)] * len(SEAM_XS)


def test_eval_ps_clustering_label_join_and_synthesized_labels(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    pytest.importorskip("haversine")
    import eval_ps_clustering as epc
    path, _ = _results(tmp_path)
    labels, _ = epc.synthesize_labels(path, 0.3)
    assert set(zip(labels["pano_x"], labels["pano_y"])) == {(0, H // 2)}
    # The live join: a server label at (0, H/2) maps back to the seam detection.
    path, _ = _results(tmp_path, xs=(0.99998,))
    server = pd.DataFrame([{"label_id": 1, "pano_id": "P", "pano_x": 0, "pano_y": H // 2}])
    mapping, n_ambiguous, _ = epc.label_to_detection(path, server)
    assert mapping == {1: ("P", 0)} and n_ambiguous == 0
