"""provenance_gate: joining a city's deployed AI labels to a rebuilt run (#56). Offline."""
import csv
import json

import pytest

import provenance_gate as pg

W, H = 16384, 8192
AI, HUMAN = "ai-uuid", "human-uuid"


def _det(px, py, conf):
    return {"x_normalized": px / W, "y_normalized": py / H, "confidence": conf}


def _run(tmp_path, panos):
    run_dir = tmp_path / "runs" / "city"
    run_dir.mkdir(parents=True)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for pid, dets, *dims in panos:
            w, h = dims[0] if dims else (W, H)
            f.write(json.dumps({"detections": dets, "pano": {
                "panorama_id": pid, "width": w, "height": h}}) + "\n")
    return run_dir


def _labels(tmp_path, rows):
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
              "properties": {"label_id": i + 1, "user_id": user, "pano_id": pid,
                             "label_type": "CurbRamp", "pano_x": x, "pano_y": y,
                             "pano_width": W, "pano_height": H}}
             for i, (pid, x, y, user) in enumerate(rows)]
    path = tmp_path / "raw.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return path


def test_verdict_boundary_is_pre_registered():
    assert (pg.PASS_SHARE, pg.TOLERANCE_PX, pg.TIER) == (0.98, 1, 0.55)
    assert pg.verdict(98, 100) == ("PASS", 0.98)
    assert pg.verdict(4899, 5000)[0] == "STOP"          # 0.9798
    assert pg.verdict(0, 0) == ("UNDETERMINED", None)


def test_join_exact_tolerance_and_misses(tmp_path):
    run_dir = _run(tmp_path, [
        ("P1", [_det(1000, 4000, 0.9), _det(2000, 4000, 0.6), _det(3000, 4000, 0.4)]),
        ("P2", [_det(16383, 5000, 0.8)]),             # at the seam
        ("P3", [_det(500, 4500, 0.9)]),
        ("EMPTY", [_det(700, 4100, 0.95)]),            # a sampled pano: no labels
    ])
    labels_path = _labels(tmp_path, [
        ("P1", 1000, 4000, AI),        # exact
        ("P1", 2001, 3999, AI),        # +/-1 px
        ("P1", 3000, 4000, AI),        # a detection there, but below the tier
        ("P2", 0, 5001, AI),           # across the seam: dx 1
        ("P3", 516, 4500, AI),         # one heatmap cell off (16 px)
        ("GONE", 10, 10, AI),          # pano not in the run
        ("P3", 500, 4500, HUMAN),      # a human label, not scored
    ])
    labels, ai, users = pg.load_ai_labels(labels_path)
    assert ai == AI and users[HUMAN] == 1

    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"))

    assert res["joinable"] == 5 and res["matched"] == 3 and res["exact"] == 1
    assert res["sub_tier"] == 1
    reasons = {r["pano_x"]: r["reason"] for r in res["unmatched"]}
    assert reasons == {3000: "below tier at the pixel", 516: "no detection within tolerance"}
    assert res["buckets"]["<= 1 heatmap cell"] == 1 and res["buckets"]["<= 2 px"] == 1
    assert res["panos"] == {"all matched": 1, "partly matched": 1, "none matched": 1}
    assert res["not_in_run"] == {"not processed yet (pending or failed)": 1}
    assert res["unlabeled_dets"] == {"labeled_panos": 1, "unlabeled_panos": 1}
    assert pg.verdict(res["matched"], res["joinable"])[0] == "STOP"


def test_a_pano_with_other_dimensions_cannot_match(tmp_path):
    """If the run's pano block has other dimensions than the pano the label was made on,
    the pixel key moves with them: counted, never silently rescaled."""
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4000, 0.9)], (13312, 6656))])
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [("P1", 1000, 4000, AI)]))
    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"))
    assert res["dims_differ"] == 1 and res["matched"] == 0
    assert res["unmatched"][0]["reason"] == "dims differ"


def test_not_in_run_says_why(tmp_path):
    run_dir = _run(tmp_path, [("P1", [])])
    (run_dir / "store_ids.txt").write_text("P1\nNOJPEG\nPENDING\n", encoding="utf-8")
    (run_dir / "store_skipped.jsonl").write_text(
        json.dumps({"pano_id": "NOJPEG", "reason": "jpg_missing"}) + "\n", encoding="utf-8")
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [
        ("NOJPEG", 1, 1, AI), ("PENDING", 1, 1, AI), ("OUTSIDE", 1, 1, AI), ("P1", 5, 5, AI)]))
    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"), *pg.load_selection(run_dir))
    assert res["not_in_run"] == {"skipped: jpg_missing": 1, "not selected": 1,
                                 "not processed yet (pending or failed)": 1}
    assert res["panos"] == {"none matched": 1}
    assert res["buckets"]["no detection on the pano"] == 1


def test_cli_writes_report_and_csv_and_exits_on_the_verdict(tmp_path):
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4000, 0.9)]), ("P2", [])])
    labels_path = _labels(tmp_path, [("P1", 1000, 4000, AI), ("P2", 10, 10, AI)])
    out = tmp_path / "gate"
    code = pg.main(["city", "--run-dir", str(run_dir), "--labels", str(labels_path),
                    "--out", str(out)])
    assert code == 1                                   # 1 of 2: STOP
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "**Verdict: STOP**" in report and "0.5000" in report
    rows = list(csv.DictReader(open(out / "unmatched.csv", newline="", encoding="utf-8")))
    assert [r["label_uid"] for r in rows] == ["city:2"]

    labels_path = _labels(tmp_path, [("P1", 1000, 4000, AI)])
    assert pg.main(["city", "--run-dir", str(run_dir), "--labels", str(labels_path),
                    "--out", str(out)]) == 0


def test_fetch_only_needs_no_run(tmp_path, capsys):
    labels_path = _labels(tmp_path, [("P1", 1, 1, AI), ("P2", 1, 1, AI), ("P1", 3, 3, HUMAN)])
    assert pg.main(["nowhere", "--run-dir", str(tmp_path / "none"), "--labels",
                    str(labels_path), "--out", str(tmp_path / "g"), "--fetch-only"]) == 0
    assert f"AI account (most labels unless --ai-user): {AI}, 2 labels on 2 panos" \
        in capsys.readouterr().out


def test_explicit_ai_user_overrides_the_majority(tmp_path):
    labels, ai, _ = pg.load_ai_labels(
        _labels(tmp_path, [("P1", 1, 1, AI), ("P2", 1, 1, AI), ("P1", 3, 3, HUMAN)]), HUMAN)
    assert ai == HUMAN and len(labels) == 1


def test_missing_results_is_an_error(tmp_path):
    labels_path = _labels(tmp_path, [("P1", 1, 1, AI)])
    with pytest.raises(SystemExit):
        pg.main(["city", "--run-dir", str(tmp_path / "none"), "--labels", str(labels_path),
                 "--out", str(tmp_path / "g")])
