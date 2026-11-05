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
def test_peak_local_max_keeps_the_row_major_first_pixel_of_a_clipped_plateau():
    # The library behaviour the whole explanation rests on (scikit-image is pinned in
    # requirements.txt since PR 158; this test catches a drift in an env built without it):
    # candidates come in np.nonzero (row-major) order, a stable descending sort keeps that
    # order among ties, and ensure_spacing keeps the first. The raw top (30, 31) loses to the
    # plateau's first pixel because every pixel of the plateau clips to exactly 1.0.
    import numpy as np
    from detectors.decode import _peaks
    h = np.zeros((64, 64), dtype=np.float32)
    h[30, 28:35] = 1.01
    h[31, 25:32] = 1.02
    h[30, 31] = 1.2
    assert _peaks(h).tolist() == [[30, 28]]


def test_plateau_check_on_a_synthetic_plateau():
    import numpy as np
    h = np.zeros((64, 64), dtype=np.float32)
    h[30, 26:35] = [0.95, 0.99, 1.001, 1.01, 1.03, 1.05, 1.03, 1.0, 0.98]   # top at col 31
    r = hg.plateau_check(h, 28, 30)
    assert r["clipped"] and r["plateau_px"] == 6 and r["raster_first"]
    assert (r["raw_max_col"], r["raw_max_row"]) == (31, 30)
    assert not hg.plateau_check(h, 27, 30)["clipped"]


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


@needs_skimage
def test_tie_peak_reversed_order_keeps_the_raster_last_pixel_of_an_exact_tie():
    # offgrid_tie_sweep.csv's reversed_col/reversed_row (PR 160 review, S1): an exact tie is
    # broken by skimage's raster order, so the flipped decode picks the other end of the run;
    # a strict maximum inside the same run decodes the same both ways.
    import numpy as np
    h = np.zeros((512, 1024), np.float32)
    h[260, 466:478] = 0.5
    assert hg.tie_peak(h) == (466, 260)
    assert hg.tie_peak(h, reverse=True) == (477, 260)
    assert (hg.tie_size(h, 466, 260), hg.tie_size(h, 477, 260)) == (11, 11)   # 21-px window
    h[260, 470] = np.nextafter(np.float32(0.5), np.float32(1))   # one ulp above the run
    assert hg.tie_peak(h) == hg.tie_peak(h, reverse=True) == (470, 260)
    assert hg.tie_size(h, 470, 260) == 1


def test_tie_size_counts_on_the_clipped_map():
    # The decoder sees np.clip(h, 0, 1): two raw values above 1.0 are a tie there.
    import numpy as np
    h = np.zeros((40, 40))
    h[20, 19], h[20, 21] = 1.02, 1.05
    assert hg.tie_size(h, 21, 20) == 2


def test_tie_env_warning_only_when_overwriting_the_committed_dir_from_another_stack(tmp_path):
    # offgrid-tie(-backend) default --out is the committed data dir, and its outputs reproduce
    # only on the recorded stack (PR 160 final review, nit 1): warn on that overwrite only.
    env = {"gpu": "NVIDIA GeForce RTX 3070", "torch_cuda": "12.6", "cudnn_allow_tf32": True,
           "software": {"numpy": "2.2.6"}}
    (tmp_path / "offgrid_tie.meta.json").write_text(json.dumps(dict(env, pano="p")),
                                                    encoding="utf-8")
    assert hg.tie_env_warning(tmp_path, "offgrid_tie.meta.json", env, tmp_path) is None
    other = dict(env, gpu="NVIDIA L40S", software={"numpy": "2.3.0"})
    w = hg.tie_env_warning(tmp_path, "offgrid_tie.meta.json", other, tmp_path)
    assert "gpu, software" in w and "--out" in w
    scratch = tmp_path / "scratch"
    assert hg.tie_env_warning(scratch, "offgrid_tie.meta.json", other, tmp_path) is None
    assert hg.tie_env_warning(tmp_path, "missing.meta.json", other, tmp_path) is None
