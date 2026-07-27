"""Unit tests for spot_check_gallery.py: rendering geometry and HTML build.

The sampler it renders (choose_panos) is tested in test_export_benchmark.py, where the
code now lives — this viewer is transitional and its tests leave with it.
Network calls (streetlevel) are monkeypatched; no test downloads anything.
"""
import json
from types import SimpleNamespace

from PIL import Image

import spot_check_gallery as g

PANO_W, PANO_H = 4096, 2048


def _record(pid, n_dets, coords=None):
    dets = coords or [
        {"x_normalized": 0.1 * (i + 1), "y_normalized": 0.5, "confidence": 0.9}
        for i in range(n_dets)]
    return {"pano": {"panorama_id": pid, "capture_date": "2024-06"}, "detections": dets}


# --- render_pano ---

def _patch_streetview(monkeypatch):
    monkeypatch.setattr(g.streetview, "find_panorama_by_id",
                        lambda pid: SimpleNamespace(image_sizes=[None] * 4))
    monkeypatch.setattr(g.streetview, "get_panorama",
                        lambda meta, zoom: Image.new("RGB", (PANO_W, PANO_H)))


def test_render_pano_centered_detection(monkeypatch, tmp_path):
    _patch_streetview(monkeypatch)
    rec = _record("PID", 1, coords=[
        {"x_normalized": 0.5, "y_normalized": 0.6, "confidence": 0.912}])
    entry = g.render_pano(rec, "random", tmp_path)

    assert entry["pid"] == "PID" and entry["group"] == "random"
    (crop,) = entry["crops"]
    # Pano-space position passes through for the overlay circle.
    assert (crop["x"], crop["y"]) == (0.5, 0.6)
    # An interior detection sits at the center of its crop.
    assert abs(crop["cx"] - 0.5) < 0.01 and abs(crop["cy"] - 0.5) < 0.01
    assert crop["conf"] == 0.912
    # Images exist; the full pano is resized to display resolution.
    assert Image.open(tmp_path / entry["full"]).size == (
        g.DISPLAY_WIDTH, g.DISPLAY_WIDTH // 2)
    assert Image.open(tmp_path / crop["img"]).size == (g.CROP_SIZE, g.CROP_SIZE)


def test_render_pano_corner_detection_clamps_crop(monkeypatch, tmp_path):
    _patch_streetview(monkeypatch)
    rec = _record("PID2", 1, coords=[
        {"x_normalized": 0.99, "y_normalized": 0.02, "confidence": 0.6}])
    (crop,) = g.render_pano(rec, "random", tmp_path)["crops"]
    # Crop clamped to the pano's top-right corner: ring lands off-center but
    # exactly on the detection.
    px, py = 0.99 * PANO_W, 0.02 * PANO_H
    assert crop["cx"] == round((px - (PANO_W - g.CROP_SIZE)) / g.CROP_SIZE, 4)
    assert crop["cy"] == round(py / g.CROP_SIZE, 4)
    assert 0.0 <= crop["cx"] <= 1.0 and 0.0 <= crop["cy"] <= 1.0


def test_render_pano_zero_detections(monkeypatch, tmp_path):
    _patch_streetview(monkeypatch)
    entry = g.render_pano(_record("PID3", 0), "empty", tmp_path)
    assert entry["crops"] == []
    assert (tmp_path / entry["full"]).exists()


# --- build_html / load_records ---

def _entries():
    return [
        {"pid": "PID", "date": "2024-06", "group": "random", "full": "PID_full.jpg",
         "crops": [{"img": "PID_det0.jpg", "conf": 0.91,
                    "x": 0.5, "y": 0.6, "cx": 0.5, "cy": 0.5}]},
        {"pid": "EMPTY", "date": "2024-01", "group": "empty",
         "full": "EMPTY_full.jpg", "crops": []},
    ]


def test_build_html_substitutes_everything(tmp_path):
    html = g.build_html(_entries(), tmp_path / "results.jsonl", "hash123", "bend")
    for placeholder in ("__ENTRIES__", "__RUN_KEY__", "__RUN_NAME__", "__SOURCE__"):
        assert placeholder not in html
    assert '"hash123"' in html and '"bend"' in html
    # The verdict schema the scorer depends on must be produced by the viewer.
    assert "no_missed" in html and "noMissed" in html


def test_load_records_run_dir_and_manifest_key(tmp_path):
    (tmp_path / "results.jsonl").write_text(
        json.dumps(_record("PID", 1)) + "\n\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"area_hash": "hash123"}), encoding="utf-8")
    jsonl_path, records, run_key = g.load_records(str(tmp_path))
    assert jsonl_path == tmp_path / "results.jsonl"
    assert len(records) == 1 and run_key == "hash123"


def test_load_records_bare_file_without_manifest(tmp_path):
    p = tmp_path / "foo.jsonl"
    p.write_text(json.dumps(_record("PID", 0)) + "\n", encoding="utf-8")
    jsonl_path, records, run_key = g.load_records(str(p))
    assert jsonl_path == p and run_key == str(p)
