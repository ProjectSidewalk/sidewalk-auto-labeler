"""heatmap_grid: the 8-cell grid census and the pre-registered sigma rule (#111). Offline."""
import doctest
import json

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
