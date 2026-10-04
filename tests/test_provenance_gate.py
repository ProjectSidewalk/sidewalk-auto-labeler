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


def test_rule_constants_are_pre_registered():
    assert (pg.PASS_SHARE, pg.TOLERANCE_PX, pg.STORE_SLACK_PX, pg.TIER) == (0.98, 1, 1, 0.55)
    assert (pg.COVERAGE_FLOOR, pg.EXTRA_DETECTION_BOUND) == (0.95, 0.02)
    assert (pg.CONTROL_SIZE, pg.CONTROL_SEED) == (200, 56)
    assert pg.store_tolerance(16384, 8192) == (17, 17)
    assert pg.exact_tolerance(16384, 8192) == (1, 1)
    # #111: both arms +/-1 coarse cell (8 heatmap cells) + the rounding pixel
    assert (pg.COARSE_CELL, pg.COARSE_CELLS) == (8, 1)
    assert pg.coarse_tolerance(16384, 8192) == (129, 129)
    assert pg.RULES[pg.RULE_COARSE_CELL] == (pg.coarse_tolerance, pg.coarse_tolerance)
    assert pg.RULES[pg.RULE_PIXEL_96] == (pg.store_tolerance, pg.exact_tolerance)


def test_skip_reasons_match_the_runner():
    import detect_from_store as dfs
    assert pg.SKIP_NO_JPEG == dfs.SKIP_NO_JPEG
    assert pg.SKIP_METADATA_UNSERVED == dfs.SKIP_METADATA_UNSERVED


def test_verdict_arms_and_floors():
    # Arm S boundary
    assert pg.verdict(98, 100, 100, 0, 0)[0] == "PASS"
    v, d = pg.verdict(4899, 5000, 5000, 0, 0)            # 0.9798
    assert v == "STOP" and d["failing"] == ["S"]
    # precision bound: <= 0.02 x joinable
    assert pg.verdict(100, 100, 100, 0, 2)[0] == "PASS"
    assert pg.verdict(100, 100, 100, 0, 3)[1]["failing"] == ["P"]
    # Arm Z, only when a control is given
    assert pg.verdict(100, 100, 100, 0, 0, 196, 200)[0] == "PASS"
    v, d = pg.verdict(97, 100, 100, 0, 0, 195, 200)
    assert v == "STOP" and d["failing"] == ["S", "Z"]
    # coverage floor: UNDETERMINED, and it takes precedence over STOP
    v, d = pg.verdict(50, 100, 100, 1, 0)
    assert v == "UNDETERMINED" and "pending" in d["reason"]
    assert pg.verdict(95, 95, 100, 0, 0)[0] == "PASS"     # exactly at the floor
    assert pg.verdict(94, 94, 100, 0, 0)[0] == "UNDETERMINED"
    assert pg.verdict(0, 0, 0, 0, 0)[0] == "UNDETERMINED"
    assert pg.verdict(100, 100, 100, 0, 0, 0, 0)[0] == "UNDETERMINED"


def test_join_store_tolerance_exact_share_and_misses(tmp_path):
    run_dir = _run(tmp_path, [
        ("P1", [_det(1000, 4000, 0.9), _det(2000, 4000, 0.6), _det(3000, 4000, 0.4)]),
        ("P2", [_det(16383, 5000, 0.8)]),             # at the seam
        ("P3", [_det(500, 4500, 0.9)]),
        ("P4", [_det(800, 4000, 0.9)]),
        ("EMPTY", [_det(700, 4100, 0.95)]),            # a sampled pano: no labels
    ])
    labels_path = _labels(tmp_path, [
        ("P1", 1000, 4000, AI),        # exact
        ("P1", 2001, 3999, AI),        # +/-1 px
        ("P1", 3000, 4000, AI),        # a detection there, but below the tier
        ("P2", 0, 5001, AI),           # across the seam: dx 1
        ("P3", 516, 4500, AI),         # one heatmap cell off (16 px): S matches, 1 px does not
        ("P4", 818, 4000, AI),         # 18 px: outside a cell + 1
        ("GONE", 10, 10, AI),          # pano not in the run
        ("P3", 500, 4500, HUMAN),      # a human label, not scored
    ])
    labels, ai, users = pg.load_ai_labels(labels_path)
    assert ai == AI and users[HUMAN] == 1

    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"))

    assert res["joinable"] == 6 and res["matched"] == 4 and res["on_pixel"] == 1
    assert res["within_1px"] == 3                     # exact_share numerator
    assert res["sub_tier"] == 1
    reasons = {r["pano_x"]: r["reason"] for r in res["unmatched"]}
    assert reasons == {3000: "below tier within tolerance", 818: "no detection within tolerance"}
    assert res["buckets"]["<= 2 heatmap cells"] == 1 and res["buckets"]["<= 2 px"] == 1
    assert res["panos"] == {"all matched": 2, "partly matched": 1, "none matched": 1}
    assert res["not_in_run"] == {"not processed yet (pending or failed)": 1}
    assert res["unlabeled_dets"] == {"labeled_panos": 1, "unlabeled_panos": 1}

    exact = pg.join(labels, pg.load_run(run_dir / "results.jsonl"), tolerance=pg.exact_tolerance)
    assert exact["matched"] == 3 and exact["matched_ids"] < res["matched_ids"]


def test_sub_tier_uses_the_tolerance_box_not_the_nearest_detection(tmp_path):
    """A below-tier detection inside the box counts as a threshold flip even when a nearer
    detection sits just outside it."""
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4018, 0.9), _det(1017, 4017, 0.3)])])
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [("P1", 1000, 4000, AI)]))
    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"))
    assert res["matched"] == 0 and res["sub_tier"] == 1


def test_duplicate_label_keys_are_counted(tmp_path):
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4000, 0.9)])])
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [
        ("P1", 1000, 4000, AI), ("P1", 1000, 4000, AI), ("P1", 5, 5, AI)]))
    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"))
    assert (res["duplicate_keys"], res["duplicate_labels"]) == (1, 2)
    assert res["matched"] == 2                        # many-to-one: both copies match


def test_coverage_counts_pending_jpegless_and_the_404_set(tmp_path):
    run_dir = _run(tmp_path, [("P1", [])])
    (run_dir / "store_ids.txt").write_text("P1\nNOJPEG\nUNSERVED\nPENDING\nSAMPLE\n",
                                           encoding="utf-8")
    (run_dir / "store_skipped.jsonl").write_text(
        json.dumps({"pano_id": "NOJPEG", "reason": pg.SKIP_NO_JPEG}) + "\n" +
        json.dumps({"pano_id": "UNSERVED", "reason": pg.SKIP_METADATA_UNSERVED}) + "\n" +
        json.dumps({"pano_id": "SAMPLE", "reason": pg.SKIP_METADATA_UNSERVED}) + "\n",
        encoding="utf-8")
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [
        ("P1", 5, 5, AI), ("NOJPEG", 1, 1, AI), ("UNSERVED", 1, 1, AI), ("UNSERVED", 2, 2, AI),
        ("PENDING", 1, 1, AI), ("OUTSIDE", 1, 1, AI)]))
    selected, skips = pg.load_selection(run_dir)
    cov = pg.coverage(labels, pg.load_run(run_dir / "results.jsonl"), selected, skips)
    assert cov["pending"] == 1 and cov["pending_examples"] == ["PENDING"]
    assert cov["labels_with_jpeg"] == 5                     # all but the NOJPEG one
    assert (cov["unserved_panos"], cov["unserved_labeled_panos"], cov["unserved_labels"]) == (2, 1, 2)


def test_draw_control_is_seeded_and_labeled_only(tmp_path):
    run = {f"P{i:03d}": (W, H, []) for i in range(300)}
    labels = [{"pano_id": f"P{i:03d}"} for i in range(0, 300, 2)] + [{"pano_id": "NOTRUN"}]
    a = pg.draw_control(labels, run, 20, seed=56)
    assert a == pg.draw_control(list(reversed(labels)), run, 20, seed=56)
    assert a != pg.draw_control(labels, run, 20, seed=57)
    assert len(a) == 20 and all(int(p[1:]) % 2 == 0 for p in a)
    assert len(pg.draw_control(labels, run, 1000)) == 150


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


def test_cli_control_arm_and_draw(tmp_path, capsys):
    """Arm S passes on a one-cell shift; the zoom-3 control fails Arm Z; the report names
    Z and the flips between the arms."""
    run_dir = _run(tmp_path, [("P1", [_det(1016, 4000, 0.9)]), ("P2", [_det(3000, 3000, 0.9)])])
    labels_path = _labels(tmp_path, [("P1", 1000, 4000, AI), ("P2", 3000, 3000, AI)])
    out = tmp_path / "gate"
    base = ["city", "--run-dir", str(run_dir), "--labels", str(labels_path), "--out", str(out)]
    assert pg.main(base) == 0                               # S: both within a cell

    assert pg.main(base + ["--draw-control", "5"]) == 0
    ids = (out / pg.CONTROL_IDS_FILE).read_text().split()
    assert ids == ["P1", "P2"]
    with pytest.raises(SystemExit, match="made once"):
        pg.main(base + ["--draw-control"])

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control = control_dir / "control.jsonl"
    with open(control, "w", encoding="utf-8") as f:
        f.write(json.dumps({"detections": [_det(1000, 4000, 0.9)],
                            "pano": {"panorama_id": "P1", "width": W, "height": H}}) + "\n")
        f.write(json.dumps({"detections": [_det(3000, 3000, 0.5)],
                            "pano": {"panorama_id": "P2", "width": W, "height": H}}) + "\n")
    assert pg.main(base + ["--control", str(control)]) == 1
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "**Verdict: STOP** -- failing: Arm Z (pipeline identity)." in report
    assert "| S only | 1 |" in report and "| both | 1 |" in report
    assert "2 of the 2 drawn panos are in the control file; 0 absent" in report


def test_control_is_checked_against_the_draw(tmp_path):
    """S4 of the PR #108 review: a control pano outside control_ids.txt refuses, a drawn pano
    the control lacks is reported as absent, and the report is LF with no absolute path."""
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4000, 0.9)]), ("P2", [_det(3000, 3000, 0.9)])])
    labels_path = _labels(tmp_path, [("P1", 1000, 4000, AI), ("P2", 3000, 3000, AI)])
    out = tmp_path / "gate"
    base = ["city", "--run-dir", str(run_dir), "--labels", str(labels_path), "--out", str(out)]
    control = tmp_path / "control.jsonl"
    control.write_text(json.dumps({"detections": [_det(1000, 4000, 0.9)], "pano": {
        "panorama_id": "P1", "width": W, "height": H}}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not exist"):
        pg.main(base + ["--control", str(control)])          # no draw to check against
    out.mkdir(exist_ok=True)
    (out / pg.CONTROL_IDS_FILE).write_text("P2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="1 control pano"):
        pg.main(base + ["--control", str(control)])          # P1 was never drawn
    (out / pg.CONTROL_IDS_FILE).write_text("P1\nP2\n", encoding="utf-8")
    assert pg.main(base + ["--control", str(control)]) == 0
    raw = (out / "report.md").read_bytes()
    assert b"\r\n" not in raw and b"\r\n" not in (out / "unmatched.csv").read_bytes()
    assert b"1 of the 2 drawn panos are in the control file; 1 absent" in raw
    assert pg.shown_path(pg.REPO_ROOT / "runs" / "x" / "results.jsonl") == "runs/x/results.jsonl"


def test_cli_undetermined_while_panos_are_pending(tmp_path):
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4000, 0.9)])])
    (run_dir / "store_ids.txt").write_text("P1\nLATER\n", encoding="utf-8")
    labels_path = _labels(tmp_path, [("P1", 1000, 4000, AI)])
    code = pg.main(["city", "--run-dir", str(run_dir), "--labels", str(labels_path),
                    "--out", str(tmp_path / "g")])
    assert code == 2
    assert "UNDETERMINED** -- 1 selected pano(s) still pending" in \
        (tmp_path / "g" / "report.md").read_text(encoding="utf-8")


def test_coarse_cell_rule_matches_a_flip_and_names_it(tmp_path):
    """#111: a detection 7 heatmap cells away (112 px at 16 px/cell) is the same ramp at the
    neighbouring coarse cell. The coarse-cell rule matches it and counts it as a flip; the
    #96 rule missed it. 10 cells away is outside one coarse cell and still misses."""
    run_dir = _run(tmp_path, [
        ("P1", [_det(1112, 4000, 0.9)]),               # flip: 7 cells in x
        ("P2", [_det(16384 - 56, 4000 + 128, 0.9)]),   # over the seam: 3.5 cells x, 8 cells y
        ("P3", [_det(3016, 3000, 0.9)]),               # grid neighbour: 1 cell
        ("P4", [_det(5160, 3000, 0.9)]),               # 10 cells: no match
    ])
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [
        ("P1", 1000, 4000, AI), ("P2", 0, 4000, AI), ("P3", 3000, 3000, AI),
        ("P4", 5000, 3000, AI)]))
    run = pg.load_run(run_dir / "results.jsonl")
    res = pg.join(labels, run, tolerance=pg.coarse_tolerance)
    assert res["matched"] == 3
    assert res["match_classes"] == {"flip": 2, "grid_neighbour": 1}
    assert res["buckets"] == {"<= 2 coarse cells": 1}
    assert pg.join(labels, run, tolerance=pg.store_tolerance)["matched"] == 1


def test_shared_claims_are_counted(tmp_path):
    """Two labels at different pixels within one coarse cell of the same detection."""
    run_dir = _run(tmp_path, [("P1", [_det(1000, 4000, 0.9)])])
    labels, _, _ = pg.load_ai_labels(_labels(tmp_path, [("P1", 1000, 4000, AI),
                                                         ("P1", 1100, 4000, AI)]))
    res = pg.join(labels, pg.load_run(run_dir / "results.jsonl"), tolerance=pg.coarse_tolerance)
    assert res["matched"] == 2 and res["shared_claims"] == 1


def test_cli_rule_flag_reproduces_the_96_rule(tmp_path, capsys):
    run_dir = _run(tmp_path, [("P1", [_det(1112, 4000, 0.9)])])
    labels_path = _labels(tmp_path, [("P1", 1000, 4000, AI)])
    base = ["city", "--run-dir", str(run_dir), "--labels", str(labels_path),
            "--out", str(tmp_path / "g")]
    assert pg.main(base) == 0                                # coarse-cell: a flip matches
    report = (tmp_path / "g" / "report.md").read_text(encoding="utf-8")
    assert "`coarse-cell`" in report and "| adjacent-coarse-cell flip (7-8) | 1 |" in report
    assert "1 of 1 matches are adjacent-coarse-cell flips" in capsys.readouterr().out
    assert pg.main(base + ["--rule", "pixel-96", "--exploratory", "a check"]) == 1
    report = (tmp_path / "g" / "report.md").read_text(encoding="utf-8")
    assert "amended after the PR #96 review" in report and "**Exploratory:** a check" in report
