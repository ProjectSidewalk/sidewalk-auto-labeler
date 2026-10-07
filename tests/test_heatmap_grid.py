"""heatmap_grid: the 8-cell grid census and the pre-registered sigma rule (#111). Offline."""
import doctest
import importlib.util
import json

import pytest

import heatmap_grid as hg


def test_doctests():
    assert doctest.testmod(hg).failed == 0


def test_grid_counts_by_tier(tmp_path):
    path = tmp_path / "results.jsonl"
    dets = [{"x_normalized": 259 / 1024, "y_normalized": 131 / 512, "confidence": 0.9},  # 3, 3
            {"x_normalized": 260 / 1024, "y_normalized": 130 / 512, "confidence": 0.4},  # 4, 2
            {"x_normalized": 0.0, "y_normalized": 0.5, "confidence": 0.2}]              # 0, 0
    path.write_text(json.dumps({"detections": dets}) + "\n", encoding="utf-8")
    c = hg.grid_counts(path)
    assert c["stored"]["n"] == 3 and c["stored"]["both_on_grid"] == 1
    assert c["0.30"]["n"] == 2 and c["0.30"][("x", 4)] == 1 and c["0.30"][("y", 2)] == 1
    assert c["0.55"]["n"] == 1 and c["0.55"]["both_on_grid"] == 1


def test_sigma_rule_constants_are_pre_registered():
    assert hg.SIGMAS == (1.0, 2.31) and hg.REJECTION_RISE_MAX == 0.10


# ---- #151 question 2: off-grid peaks are clipped-plateau tie-breaks ----------------------
needs_skimage = pytest.mark.skipif(importlib.util.find_spec("skimage") is None,
                                   reason="needs scikit-image (requirements-test.txt)")


@needs_skimage
def test_plateau_demo_column_residue_7():
    # c6 = 0.99, c7 = 1.015 on the peak row: the top above 1.0 clips flat from column 55 to
    # the knot (59.5), and the raster-first pixel of the plateau wins -- residue 7 of cell 6,
    # vancouver:62190's shape. The row stays on the knot (275, residue 3).
    col, row, conf = hg.plateau_demo(**hg.DEMO_COL)
    assert (col, row) == (55, 275) and conf > 1.0
    assert (col % 8, row % 8) == (7, 3)


@needs_skimage
def test_plateau_demo_row_residue_1():
    # Same mechanism along y: r36 slightly > 1 clips rows 289..291, row 289 (residue 1) wins,
    # the shape of the store side of vancouver:41404 and vancouver:63918.
    col, row, conf = hg.plateau_demo(**hg.DEMO_ROW)
    assert (col, row) == (483, 289) and conf > 1.0
    assert (col % 8, row % 8) == (3, 1)


@needs_skimage
def test_unclipped_peak_stays_on_the_knot():
    # Control: the same map scaled below 1.0 has no plateau, so the argmax is on the grid.
    demo = dict(hg.DEMO_COL, col_knots=[v * 0.95 for v in hg.DEMO_COL["col_knots"]])
    col, row, conf = hg.plateau_demo(**demo)
    assert conf < 1.0 and (col % 8, row % 8) == (3, 3)


def test_offgrid_misses_picks_the_2_to_6_cell_tier_neighbour(tmp_path):
    run = tmp_path / "vancouver" / "results.jsonl"
    gate = run.parent / "provenance_gate"
    gate.mkdir(parents=True)
    w, h = 16384, 8192
    dets = [{"x_normalized": 435 / 1024, "y_normalized": 283 / 512, "confidence": 0.99},
            {"x_normalized": 900 / 1024, "y_normalized": 283 / 512, "confidence": 0.9}]
    rec = {"pano": {"panorama_id": "p1", "width": w, "height": h}, "detections": dets}
    run.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    cols = "label_id,pano_id,pano_x,pano_y,reason\n"
    rows = [f"85767,p1,{433 * 16},{283 * 16},no detection within tolerance",   # 2 cells: kept
            f"1,p1,{899 * 16},{283 * 16},no detection within tolerance",       # 1 cell: not
            f"2,p1,{433 * 16},{283 * 16},below tier within tolerance"]         # other reason
    (gate / "unmatched.csv").write_text(cols + "\n".join(rows) + "\n", encoding="utf-8")
    out = hg.offgrid_misses(gate / "unmatched.csv", run, "vancouver")
    assert [m["label_uid"] for m in out] == ["vancouver:85767"]
    m = out[0]
    assert (m["label_residue"], m["det_residue"], m["coarse_cells"]) == ("1,3", "3,3", "same")
    assert m["off_grid_side"] == "label" and m["off_grid_is_raster_earlier"] is True
