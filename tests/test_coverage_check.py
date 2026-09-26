"""scripts/coverage_check.py (issue #46): which panos the records say were sent labels, and
how the server's backup state classifies them. Offline: the pulls and the metadata probe
are monkeypatched (conftest blocks the network)."""
import json

import pytest

import coverage_check as cc

SERVER = "https://ps.example.org"
PROD = SERVER + "/ai/submitLabelsOnPano"
TEST = "https://ps-test.example.org/ai/submitLabelsOnPano"


def _line(pid, *dets):
    """A results.jsonl line; each det is (confidence, y_normalized)."""
    return json.dumps({"label_type": "CurbRamp",
                       "pano": {"panorama_id": pid, "width": 100, "height": 50,
                                "lat": 37.5, "lng": -77.4, "source": "mapillary"},
                       "detections": [{"x_normalized": 0.5, "y_normalized": y, "confidence": c}
                                      for c, y in dets]})


def _write(tmp_path, name, lines, endpoints, sidecars=()):
    (tmp_path / name).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    (tmp_path / f"{name}.submission.json").write_text(json.dumps(
        {"input_file": name, "total_lines": len(lines), "endpoints": endpoints}), encoding="utf-8")
    for suffix, numbers in sidecars:
        (tmp_path / f"{name}{suffix}").write_text("".join(f"{n}\n" for n in numbers), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path):
    """Three files, as a city's records look after a band and a partial fix:
    a.jsonl  base campaign (predates the mask; min_confidence moved down to the band's floor)
             + a masked band [0.3, 0.55);
    c.jsonl  a partial campaign at 0.3, masked, whose sidecar holds lines 1 and 3;
    d.jsonl  sent only to another server."""
    _write(tmp_path, "a.jsonl",
           [_line("A", (0.9, 0.6)),       # base
            _line("R", (0.9, 0.9)),       # base, on the rig: the base shipped unmasked
            _line("L", (0.4, 0.6)),       # band
            _line("B", (0.4, 0.9)),       # band, on the rig: masked, never sent
            _line("N", (0.2, 0.6))],      # below every range
           {PROD: {"submitted_lines": 5, "min_confidence": 0.3,
                   "bands": {"0.3-0.55": {"submitted_lines": 5, "rig_masked": True}}}})
    _write(tmp_path, "c.jsonl", [_line("C1", (0.5, 0.6)), _line("C2", (0.5, 0.6)),
                                 _line("C3", (0.5, 0.6))],
           {PROD: {"submitted_lines": 2, "min_confidence": 0.3, "rig_masked": True}},
           sidecars=[(".submitted", [1, 3])])
    _write(tmp_path, "d.jsonl", [_line("D", (0.9, 0.6))],
           {TEST: {"submitted_lines": 1, "min_confidence": 0.3}})
    return tmp_path


def test_expected_set_replays_each_campaign_at_what_it_sent(run_dir):
    expected, campaigns, problems = cc.expected_panos(run_dir, PROD)
    assert problems == []
    assert set(expected) == {"A", "R", "L", "C1", "C3"}
    ranges = {c["campaign"]: (c["select_min"], c["select_max"], c["rig_masked"]) for c in campaigns}
    assert ranges == {"a.jsonl": (0.55, None, False), "a.jsonl (band 0.3-0.55)": (0.3, 0.55, True),
                      "c.jsonl": (0.3, None, True)}


def test_partial_campaign_without_sidecar_cannot_be_determined(run_dir):
    (run_dir / "c.jsonl.submitted").unlink()
    expected, _, problems = cc.expected_panos(run_dir, PROD)
    assert "C1" not in expected and len(problems) == 1 and "sidecar" in problems[0]
    assert cc.exit_code({"covered": 3}, problems) == cc.EXIT_UNDETERMINED
    _, _, none = cc.expected_panos(run_dir, "https://nowhere.example.org/ai/submitLabelsOnPano")
    assert none and "no submission record" in none[0]


def _feature(label_id, pano_id=None, ai=True, backup=True, pitch=None, label_type="CurbRamp"):
    props = {"label_id": label_id, "label_type": label_type, "ai_generated": ai, "has_backup": backup}
    if pano_id is not None:
        props.update(pano_id=pano_id, camera_pitch=pitch)
    return {"type": "Feature", "geometry": None, "properties": props}


def test_server_join_on_label_id():
    labels_all = [_feature(1, backup=True), _feature(2, backup=False), _feature(3, ai=False),
                  _feature(4, backup=False), _feature(9)]          # 9: not in rawLabels
    raw = [_feature(1, "P1"), _feature(2, "P1", pitch=-1.0), _feature(3, "P3"),
           _feature(4, "P4")]
    panos, unjoinable, any_label = cc.server_ai_panos(labels_all, {"CurbRamp": raw})
    assert panos == {"P1": {"ai_labels": 2, "backed": True, "pitch": -1.0},
                     "P4": {"ai_labels": 1, "backed": False, "pitch": None}}
    assert unjoinable == 1 and set(any_label) == {"P1", "P3", "P4"}


def test_classification_false_is_never_missing_without_a_decisive_probe():
    expected = {p: {"labels": 1} for p in ("GONE", "OK", "META", "MISS", "NOPITCH")}
    server = {"OK": {"ai_labels": 1, "backed": True, "pitch": None},
              "META": {"ai_labels": 1, "backed": False, "pitch": None},
              "MISS": {"ai_labels": 1, "backed": False, "pitch": -2.0},
              "NOPITCH": {"ai_labels": 1, "backed": False, "pitch": None}}
    answers = {"META": 200, "MISS": 404, "NOPITCH": 404}
    rows, problems = cc.classify(expected, server, answers.get, spacing_s=0)
    assert {p: r["status"] for p, r in rows.items()} == {
        "GONE": "retired", "OK": "covered", "META": "covered_metadata",
        "MISS": "missing", "NOPITCH": "unconfirmed"}
    assert problems == []
    rows, _ = cc.classify(expected, server, answers.get, confirm=False)
    assert rows["MISS"]["status"] == "unconfirmed"      # has_backup false alone is not absence


def test_max_confirm_overflow_is_undetermined():
    expected = {f"P{i}": {"labels": 1} for i in range(5)}
    server = {p: {"ai_labels": 1, "backed": False, "pitch": 1.0} for p in expected}
    calls = []
    rows, problems = cc.classify(expected, server, lambda p: calls.append(p) or 200,
                                 max_confirm=2, spacing_s=0)
    assert len(calls) == 2 and len(problems) == 1
    assert sum(r["status"] == "unconfirmed" for r in rows.values()) == 3
    counts = {"covered_metadata": 2, "unconfirmed": 3}
    assert cc.exit_code(counts, problems) == cc.EXIT_UNDETERMINED


def test_archive_index_as_file_or_dir(tmp_path):
    (tmp_path / "panos").mkdir()
    (tmp_path / "index.csv").write_text("panorama_id,filename,bytes,sha256\nX,X.jpg,10,ab\n",
                                        encoding="utf-8")
    (tmp_path / "decayed.txt").write_text("Y\n", encoding="utf-8")
    for arg in (tmp_path / "index.csv", tmp_path, tmp_path / "panos"):
        index, decayed, _ = cc.load_archive(arg)
        assert [cc.archive_state(p, index, decayed) for p in ("X", "Y", "Z")] == [
            "archived", "decayed", "not_archived"]


def test_empty_pull_is_refused_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "fetch_json", lambda url: b'{"type":"FeatureCollection","features":[]}')
    with pytest.raises(cc.PullError, match="zero rows"):
        cc.pull(SERVER + "/labels/all", tmp_path / "labels_all.geojson")
    assert not (tmp_path / "labels_all.geojson").exists()


def test_main_end_to_end_exit_codes_and_report(run_dir, monkeypatch):
    backed = {"A": True, "R": True, "L": True, "C1": True, "C3": False}
    ids = {p: i for i, p in enumerate(sorted(backed), 1)}
    payloads = {
        "/adminapi/panos": [{"pano_id": p, "has_labels": True} for p in backed],
        "/labels/all": {"type": "FeatureCollection", "features": [
            _feature(ids[p], backup=b) for p, b in backed.items()]},
        "/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson": {
            "type": "FeatureCollection", "features": [
                _feature(ids[p], p, pitch=1.0 if p == "C3" else None) for p in backed]}}
    monkeypatch.setattr(cc, "fetch_json",
                        lambda url: json.dumps(payloads[url[len(SERVER):]]).encode())
    monkeypatch.setattr(cc.time, "sleep", lambda s: None)
    status = {"C3": 404}
    monkeypatch.setattr(cc, "probe_metadata", lambda server, pid, session=None: status[pid])
    argv = [str(run_dir / "a.jsonl"), "--server", SERVER + "/"]
    assert cc.main(argv) == cc.EXIT_GAPS                   # C3: 404 with a pitch
    report = (run_dir / "coverage" / "report.md").read_text(encoding="utf-8")
    assert SERVER + "/labels/all" in report and "| missing | 1 |" in report
    side = json.loads((run_dir / "coverage" / "labels_all.geojson.source.json").read_text())
    assert side["sha256"] in report and side["n_features"] == 5
    assert "C3" in (run_dir / "coverage" / "missing.csv").read_text()

    status["C3"] = 200
    assert cc.main(argv) == cc.EXIT_OK                     # cached pulls, probe settles it
    assert not (run_dir / "coverage" / "missing.csv").exists()

    payloads["/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson"]["features"][-1][
        "properties"]["camera_pitch"] = None               # C3 now has no pose on the server
    status["C3"] = 404
    assert cc.main(argv + ["--refresh"]) == cc.EXIT_OK     # unconfirmed only
    assert cc.main(argv + ["--strict"]) == cc.EXIT_GAPS
