"""scripts/review_order.py (issue #168 item 5): the validator-ranked review list of the AI
labels live on one server. Offline, tmp_path only: the run dirs, the submission records and
the validator-ai ranker run are all built here."""
import csv
import doctest
import json

import pytest

import review_order as ro
import send_to_ps

PROD = "https://ps.example.org/ai/submitLabelsOnPano"
TEST = "https://ps-test.example.org/ai/submitLabelsOnPano"
HOST = "ps.example.org"
W, H = 1000, 500


# ------------------------------------------------------------------------------- builders

def _line(pid, *dets, types=None):
    """A results.jsonl line; each det is (confidence, x_normalized, y_normalized[, type])."""
    rec = {"label_type": "CurbRamp",
           "pano": {"panorama_id": pid, "width": W, "height": H, "lat": 37.5, "lng": -77.4,
                    "source": "mapillary"},
           "detections": [{"x_normalized": d[1], "y_normalized": d[2], "confidence": d[0],
                           **({"label_type": d[3]} if len(d) > 3 else {})} for d in dets]}
    if types:
        rec["label_types"] = sorted(types)
        rec["label_type"] = None
    return json.dumps(rec)


def _write(run_dir, name, lines, endpoints, sidecars=()):
    """A results file, its record (with the file's real sha256) and its sidecars."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    digest, total, size = send_to_ps.hash_and_count(path)
    (run_dir / f"{name}.submission.json").write_text(json.dumps(
        {"input_file": name, "sha256": digest, "total_lines": total, "total_bytes": size,
         "endpoints": endpoints}, indent=2) + "\n", encoding="utf-8")
    for suffix, numbers in sidecars:
        (run_dir / f"{name}{suffix}").write_text("".join(f"{n}\n" for n in numbers), encoding="utf-8")


def _px(x, y):
    return round(x * W) % W, round(y * H)


def _label(label_id, pid, x, y, city="city-a", author="ai", px=None):
    """A labels.csv row at the pixel the labeler would have sent for (x, y)."""
    pano_x, pano_y = px if px is not None else _px(x, y)
    return {"label_uid": f"{city}:{label_id}", "city": city, "label_id": str(label_id),
            "pano_id": pid, "author": author, "tier": "core", "n_votes": "0", "clean": "1",
            "pano_x": str(pano_x), "pano_y": str(pano_y)}


def _csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _ranker(path, labels, scores, all_ai=True, label_type="CurbRamp", score_column=False):
    """A validator-ai `select --all-ai` / crops / score run. `scores` = {label_uid: p_incorrect}."""
    path.mkdir(parents=True, exist_ok=True)
    _csv(path / "labels.csv", labels, list(labels[0]) if labels else list(ro.LABEL_COLUMNS))
    manifest = [{"label_uid": r["label_uid"], "pano_id": r["pano_id"], "status": "ok"}
                for r in labels if r["label_uid"] in scores]
    _csv(path / "manifest.csv", manifest, ["label_uid", "pano_id", "status"])
    preds = [{"label_uid": u, "split": "test", "true": "", "p_correct": repr(1 - p),
              "p_incorrect": repr(p), **({"rank_score": "100"} if score_column else {})}
             for u, p in scores.items()]
    _csv(path / "predictions.csv", preds,
         ["label_uid", "split", "true", "p_correct", "p_incorrect"] + (["rank_score"] if score_column else []))
    (path / "selection.json").write_text(json.dumps({
        "all_ai": all_ai, "label_type": label_type, "selected_on": "2026-11-01T00:00:00+00:00",
        "cities": sorted({r["city"] for r in labels}),
        "labels_csv_sha256": ro._sha256(path / "labels.csv")}), encoding="utf-8")
    (path / "predictions_provenance.json").write_text(json.dumps({
        "manifest_sha256": ro._sha256(path / "manifest.csv"), "model_sha256": "m" * 64,
        "checkpoint": "ckpt", "scored": "2026-11-01T01:00:00+00:00"}), encoding="utf-8")
    return path


def _run(run_dirs, ranker, *extra, server=HOST, endpoint=PROD):
    return ro.main([str(d) for d in run_dirs] + ["--endpoint", endpoint, "--ranker-run", str(ranker),
                                                  "--ranker-server", server, *extra])


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _out(run_dir):
    return run_dir / "review_order" / HOST


@pytest.fixture
def city(tmp_path):
    """One run dir, as a city looks after a band:
    a.jsonl  base campaign (predates the mask; min_confidence moved down to the band floor)
             + a masked band [0.3, 0.55)."""
    run_dir = tmp_path / "cityrun"
    _write(run_dir, "a.jsonl",
           [_line("A", (0.9, 0.5, 0.6)),       # base
            _line("R", (0.9, 0.5, 0.9)),       # base, on the rig: the base shipped unmasked
            _line("L", (0.4, 0.5, 0.6)),       # band
            _line("B", (0.4, 0.5, 0.9)),       # band, on the rig: masked, never sent
            _line("N", (0.2, 0.5, 0.6))],      # below every range
           {PROD: {"submitted_lines": 5, "min_confidence": 0.3, "labels_submitted": 3,
                   "last_submission_utc": "2026-09-22T19:00:00Z",
                   "bands": {"0.3-0.55": {"submitted_lines": 5, "rig_masked": True,
                                          "labels_submitted": 1,
                                          "last_submission_utc": "2026-09-22T19:00:00Z"}}}})
    return run_dir


# ---------------------------------------------------------------------------------- tests

def test_rank_score_matches_validator_doctest_and_keeps_saturated_order():
    assert [round(ro.rank_score(c, i), 1) for c, i in [(0.5, 0.5), (1e-6, 1.0), (1e-3, 1.0)]] == \
        [0.0, 13.8, 6.9]
    assert ro.rank_score(1e-6, 1.0) > ro.rank_score(1e-3, 1.0)   # both p_incorrect == 1.0
    assert ro.rank_score(0.0, 1.0) == pytest.approx(-ro.math.log(1e-45))   # clipped, finite
    failures, _ = doctest.testmod(ro, verbose=False)
    assert failures == 0


def test_base_and_band_rows_carry_their_campaign_and_unshipped_rows_are_unmatched(city, tmp_path):
    labels = [_label(1, "A", 0.5, 0.6), _label(2, "R", 0.5, 0.9), _label(3, "L", 0.5, 0.6),
              _label(4, "B", 0.5, 0.9), _label(5, "N", 0.5, 0.6)]
    ranker = _ranker(tmp_path / "rk", labels,
                     {"city-a:1": 0.1, "city-a:2": 0.9, "city-a:3": 0.5, "city-a:4": 0.2, "city-a:5": 0.3})
    assert _run([city], ranker) == 0
    rows = _read(_out(city) / ro.LIST_FILE)
    assert [r["label_uid"] for r in rows] == ["city-a:2", "city-a:3", "city-a:1"]   # by score
    assert [r["review_rank"] for r in rows] == ["1", "2", "3"]
    by_uid = {r["label_uid"]: r for r in rows}
    assert by_uid["city-a:1"]["campaign"] == "cityrun/a.jsonl"
    assert by_uid["city-a:3"]["campaign"] == "cityrun/a.jsonl (band 0.3-0.55)"
    assert by_uid["city-a:2"]["verdict"] == "incorrect" and by_uid["city-a:1"]["verdict"] == "correct"
    # the masked band detection and the below-range one were never sent: not queue rows
    assert {r["label_uid"] for r in _read(_out(city) / ro.UNMATCHED_FILE)} == {"city-a:4", "city-a:5"}
    summary = json.loads((_out(city) / ro.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["counts"]["shipped_keys"] == 3 and summary["counts"]["match_rate"] == 1.0
    replayed = {c["campaign"]: c["replayed_labels"] for c in summary["campaigns"]}
    assert replayed == {"cityrun/a.jsonl": 2, "cityrun/a.jsonl (band 0.3-0.55)": 1}
    assert not (_out(city) / ro.NOT_ON_SERVER_FILE).exists()
    assert (_out(city) / ro.REPORT_FILE).read_text(encoding="utf-8").startswith("# Review order")


def test_join_is_on_the_ps_pixel_including_the_seam(tmp_path):
    run_dir = tmp_path / "seam"
    _write(run_dir, "s.jsonl", [_line("S", (0.9, 0.99998, 0.6), (0.9, 0.25, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.55, "labels_submitted": 2,
                   "rig_masked": True}})
    labels = [_label(1, "S", 0, 0, px=(0, 300)),                  # x just below 1.0 wraps to 0
              _label(2, "S", 0, 0, px=(251, 300))]                # one pixel off 250
    ranker = _ranker(tmp_path / "rk", labels, {"city-a:1": 0.5, "city-a:2": 0.5})
    assert _run([run_dir], ranker) == 0
    assert [r["label_uid"] for r in _read(_out(run_dir) / ro.LIST_FILE)] == ["city-a:1"]
    assert [r["label_uid"] for r in _read(_out(run_dir) / ro.UNMATCHED_FILE)] == ["city-a:2"]
    gone = _read(_out(run_dir) / ro.NOT_ON_SERVER_FILE)
    assert [(g["pano_id"], g["pano_x"], g["pano_y"]) for g in gone] == [("S", "250", "300")]


def test_ranked_rows_come_first_then_no_prediction_by_confidence(tmp_path):
    run_dir = tmp_path / "r"
    _write(run_dir, "a.jsonl", [_line("P", (0.9, 0.1, 0.6), (0.6, 0.2, 0.6), (0.7, 0.3, 0.6),
                                      (0.8, 0.4, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.55, "labels_submitted": 4,
                   "rig_masked": True}})
    labels = [_label(i, "P", x, 0.6) for i, x in [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]]
    ranker = _ranker(tmp_path / "rk", labels, {"city-a:1": 0.2, "city-a:4": 0.7})  # 2, 3: crop failed
    assert _run([run_dir], ranker) == 0
    rows = _read(_out(run_dir) / ro.LIST_FILE)
    assert [(r["label_uid"], r["rank_status"]) for r in rows] == [
        ("city-a:4", "ranked"), ("city-a:1", "ranked"),
        ("city-a:2", "no_prediction"), ("city-a:3", "no_prediction")]   # 0.6 before 0.7
    assert [int(r["review_rank"]) for r in rows] == [1, 2, 3, 4]
    assert rows[2]["rank_score"] == "" and rows[2]["confidence"] == "0.6"


def test_city_is_part_of_the_key(city, tmp_path):
    labels = [_label(1, "A", 0.5, 0.6, city="city-a"), _label(1, "A", 0.5, 0.6, city="city-b")]
    ranker = _ranker(tmp_path / "rk", labels, {"city-a:1": 0.2, "city-b:1": 0.8})
    assert _run([city], ranker) == 2                         # two cities, no --city
    assert not _out(city).exists()                           # exit 2 writes nothing
    assert _run([city], ranker, "--city", "city-b") == 0
    rows = _read(_out(city) / ro.LIST_FILE)
    assert [(r["label_uid"], r["city"], r["label_id"]) for r in rows] == [("city-b:1", "city-b", "1")]
    assert _run([city], ranker, "--city", "city-z") == 2


def test_partial_campaign_lists_only_its_sidecar_lines(tmp_path):
    run_dir = tmp_path / "p"
    _write(run_dir, "c.jsonl", [_line(f"C{i}", (0.5, 0.5, 0.6)) for i in (1, 2, 3)],
           {PROD: {"submitted_lines": 2, "min_confidence": 0.3, "rig_masked": True,
                   "labels_submitted": 2}},
           sidecars=[(".submitted", [1, 3])])
    labels = [_label(i, f"C{i}", 0.5, 0.6) for i in (1, 2, 3)]
    ranker = _ranker(tmp_path / "rk", labels, {f"city-a:{i}": 0.5 for i in (1, 2, 3)})
    assert _run([run_dir], ranker) == 0
    assert {r["pano_id"] for r in _read(_out(run_dir) / ro.LIST_FILE)} == {"C1", "C3"}
    assert [r["pano_id"] for r in _read(_out(run_dir) / ro.UNMATCHED_FILE)] == ["C2"]


def test_multitype_rank_attaches_to_its_own_type_only(tmp_path, capsys):
    run_dir = tmp_path / "m"
    types = ["Crosswalk", "CurbRamp"]
    _write(run_dir, "t.jsonl",
           [_line("T", (0.9, 0.5, 0.6, "CurbRamp"), (0.9, 0.5, 0.6, "Crosswalk"), types=types)],
           {PROD: {"submitted_lines": 1, "labels_submitted": 2, "min_confidence": 0.23,
                   "types": {"CurbRamp": {"submitted_lines": 1, "labels_submitted": 1,
                                          "min_confidence": 0.3, "rig_masked": True},
                             "Crosswalk": {"submitted_lines": 1, "labels_submitted": 1,
                                           "min_confidence": 0.23, "rig_masked": True}}}},
           sidecars=[(".type-CurbRamp.submitted", [1]), (".type-Crosswalk.submitted", [1])])
    ranker = _ranker(tmp_path / "rk", [_label(1, "T", 0.5, 0.6)], {"city-a:1": 0.5})
    assert _run([run_dir], ranker) == 0
    rows = _read(_out(run_dir) / ro.LIST_FILE)
    assert [(r["label_uid"], r["label_type"], r["campaign"]) for r in rows] == [
        ("city-a:1", "CurbRamp", "m/t.jsonl (type CurbRamp)")]
    assert not (_out(run_dir) / ro.NOT_ON_SERVER_FILE).exists()   # the Crosswalk key is not "gone"
    summary = json.loads((_out(run_dir) / ro.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["counts"]["other_type_keys"] == {"Crosswalk": 1}
    per = {c["campaign"]: c for c in summary["campaigns"]}
    assert per["m/t.jsonl (type Crosswalk)"]["other_type"] == 1
    assert per["m/t.jsonl (type Crosswalk)"]["on_server"] == 0
    assert "no ranker input covers type Crosswalk" in capsys.readouterr().err


def test_richmond_shape_one_row_per_server_label(tmp_path):
    """Two records share a core (results.band.jsonl copies results.jsonl's), and a third file
    resent one pano at the same pixel after its old labels were soft-deleted."""
    run_dir = tmp_path / "richmond"
    core = [_line("A", (0.9, 0.5, 0.6)), _line("Q", (0.9, 0.2, 0.6), (0.4, 0.3, 0.6))]
    _write(run_dir, "results.jsonl", core,
           {PROD: {"submitted_lines": 2, "min_confidence": 0.55, "labels_submitted": 2,
                   "last_submission_utc": "2026-09-05T01:39:04Z"}})
    _write(run_dir, "results.band.jsonl", core,
           {PROD: {"submitted_lines": 2, "min_confidence": 0.3, "labels_submitted": 3,
                   "last_submission_utc": "2026-09-22T19:03:45Z",
                   "bands": {"0.3-0.55": {"submitted_lines": 2, "labels_submitted": 1,
                                          "rig_masked": True,
                                          "last_submission_utc": "2026-09-22T19:03:45Z"}}}})
    _write(run_dir, "results.posfix.jsonl", [_line("Q", (0.9, 0.2, 0.6), (0.4, 0.3, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.3, "labels_submitted": 2,
                   "rig_masked": True, "last_submission_utc": "2026-09-24T13:30:56Z"}})
    labels = [_label(1, "A", 0.5, 0.6), _label(20, "Q", 0.2, 0.6), _label(21, "Q", 0.3, 0.6)]
    ranker = _ranker(tmp_path / "rk", labels, {"city-a:1": 0.1, "city-a:20": 0.6, "city-a:21": 0.3})
    assert _run([run_dir], ranker) == 0
    rows = {r["label_uid"]: r for r in _read(_out(run_dir) / ro.LIST_FILE)}
    assert len(rows) == 3                                    # each server label once
    assert rows["city-a:1"]["campaigns"] == "richmond/results.band.jsonl;richmond/results.jsonl"
    assert rows["city-a:1"]["campaign"] == "richmond/results.band.jsonl"        # newest
    assert rows["city-a:20"]["campaign"] == "richmond/results.posfix.jsonl"
    assert rows["city-a:20"]["campaigns"].count(";") == 2
    assert rows["city-a:21"]["campaigns"] == ("richmond/results.band.jsonl (band 0.3-0.55);"
                                              "richmond/results.posfix.jsonl")
    summary = json.loads((_out(run_dir) / ro.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["counts"]["rows_in_several_campaigns"] == 3
    assert summary["counts"]["shipped_keys"] == 3


def test_sibling_run_dirs_share_one_list(tmp_path):
    a, b = tmp_path / "juneau", tmp_path / "juneau_expansion"
    _write(a, "results.jsonl", [_line("A", (0.9, 0.5, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.55, "labels_submitted": 1, "rig_masked": True}})
    _write(b, "results.jsonl", [_line("B", (0.9, 0.5, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.55, "labels_submitted": 1, "rig_masked": True}})
    ranker = _ranker(tmp_path / "rk", [_label(1, "A", 0.5, 0.6), _label(2, "B", 0.5, 0.6)],
                     {"city-a:1": 0.3, "city-a:2": 0.4})
    assert _run([a], ranker) == 0
    assert [r["label_uid"] for r in _read(_out(a) / ro.UNMATCHED_FILE)] == ["city-a:2"]
    assert _run([a, b], ranker) == 0
    assert {r["campaign"] for r in _read(_out(a) / ro.LIST_FILE)} == {
        "juneau/results.jsonl", "juneau_expansion/results.jsonl"}
    assert not (_out(a) / ro.UNMATCHED_FILE).exists()       # a stale unmatched.csv is removed


def test_author_filter_and_zero_matches(city, tmp_path):
    human = _label(9, "A", 0.5, 0.6, author="human")          # a human label at an AI pixel
    ranker = _ranker(tmp_path / "rk", [human, _label(1, "Z", 0.5, 0.6)], {"city-a:9": 0.9, "city-a:1": 0.1})
    assert _run([city], ranker) == 2                          # nothing joins: exit 2, nothing written
    assert not _out(city).exists()
    ranker = _ranker(tmp_path / "rk2", [human, _label(1, "A", 0.5, 0.6)], {"city-a:9": 0.9, "city-a:1": 0.1})
    assert _run([city], ranker) == 0
    assert [r["label_uid"] for r in _read(_out(city) / ro.LIST_FILE)] == ["city-a:1"]
    summary = json.loads((_out(city) / ro.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["ranker"]["human_rows_dropped"] == 1
    gone = _read(_out(city) / ro.NOT_ON_SERVER_FILE)          # R and L have no server row
    assert {g["pano_id"] for g in gone} == {"R", "L"}
    assert summary["counts"]["match_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert any("under 90%" in w for w in summary["warnings"])


def test_records_are_never_written_and_check_reports_staleness(city, tmp_path):
    record = city / "a.jsonl.submission.json"
    before = record.read_bytes()
    ranker = _ranker(tmp_path / "rk", [_label(1, "A", 0.5, 0.6)], {"city-a:1": 0.2})
    assert _run([city], ranker) == 0
    assert record.read_bytes() == before
    assert sorted(p.name for p in city.glob("*.submission.json")) == ["a.jsonl.submission.json"]
    check = ["--endpoint", PROD, "--check"]
    assert ro.main([str(city)] + check) == 0
    # a later send moves the record's counts and timestamp: the list is stale
    data = json.loads(before)
    data["endpoints"][PROD]["last_submission_utc"] = "2026-11-02T00:00:00Z"
    data["endpoints"][PROD]["labels_submitted"] = 4
    record.write_text(json.dumps(data), encoding="utf-8")
    assert ro.main([str(city)] + check) == 1
    assert ro.stale_reasons({"x/a": {"sha256": "1"}}, {}) == ["x/a: no longer holds a campaign on this endpoint"]
    assert ro.main([str(tmp_path / "rk")] + check) == 2      # no summary there


def test_results_file_that_changed_since_its_record_is_refused(city, tmp_path):
    with open(city / "a.jsonl", "a", encoding="utf-8") as f:
        f.write(_line("Z", (0.1, 0.5, 0.6)) + "\n")       # sends nothing, but changes the hash
    ranker = _ranker(tmp_path / "rk", [_label(1, "A", 0.5, 0.6)], {"city-a:1": 0.2})
    assert _run([city], ranker) == 2


def test_ranker_contract_is_checked(city, tmp_path, capsys):
    labels = [_label(1, "A", 0.5, 0.6)]
    assert _run([city], _ranker(tmp_path / "voted", labels, {"city-a:1": 0.2}, all_ai=False)) == 2
    assert "all_ai" in capsys.readouterr().err

    rk = _ranker(tmp_path / "edited", labels, {"city-a:1": 0.2})
    with open(rk / "labels.csv", "a", encoding="utf-8") as f:
        f.write("city-a:2,city-a,2,A,ai,core,0,1,500,300\n")
    assert _run([city], rk) == 2
    assert "labels_csv_sha256" in capsys.readouterr().err

    rk = _ranker(tmp_path / "rescored", labels, {"city-a:1": 0.2})
    (rk / "manifest.csv").write_text("label_uid,pano_id,status\ncity-a:1,OTHER,ok\n", encoding="utf-8")
    assert _run([city], rk) == 2
    assert "manifest_sha256" in capsys.readouterr().err

    rk = _ranker(tmp_path / "pano", labels, {"city-a:1": 0.2})
    (rk / "manifest.csv").write_text("label_uid,pano_id,status\ncity-a:1,OTHER,ok\n", encoding="utf-8")
    prov = json.loads((rk / "predictions_provenance.json").read_text(encoding="utf-8"))
    prov["manifest_sha256"] = ro._sha256(rk / "manifest.csv")
    (rk / "predictions_provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    assert _run([city], rk) == 2
    assert "manifest.csv pano_id" in capsys.readouterr().err

    ok = _ranker(tmp_path / "ok", labels, {"city-a:1": 0.2})
    assert _run([city], ok, server="ps-test.example.org") == 2    # a pull from another server
    assert _run([city], ok, server="https://PS.example.org/") == 0
    assert ro.main([str(city), "--endpoint", PROD]) == 2         # no --ranker-run


def test_pixels_and_pano_ids_are_parsed_exactly(tmp_path):
    run_dir = tmp_path / "ids"
    _write(run_dir, "a.jsonl", [_line("0012345", (0.9, 0.5, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.55, "labels_submitted": 1, "rig_masked": True}})
    row = _label(1, "0012345", 0, 0, px=(500, 300))
    row["pano_x"] = "500.0"                                      # pandas writes floats like this
    rk = _ranker(tmp_path / "rk", [row], {"city-a:1": 0.2})
    assert _run([run_dir], rk) == 0
    assert _read(_out(run_dir) / ro.LIST_FILE)[0]["pano_id"] == "0012345"
    row["pano_x"] = "500.5"
    assert _run([run_dir], _ranker(tmp_path / "rk2", [row], {"city-a:1": 0.2})) == 2


def test_input_rank_score_column_wins(city, tmp_path):
    rk = _ranker(tmp_path / "rk", [_label(1, "A", 0.5, 0.6)], {"city-a:1": 0.2}, score_column=True)
    assert _run([city], rk) == 0
    assert _read(_out(city) / ro.LIST_FILE)[0]["rank_score"] == "100"
    summary = json.loads((_out(city) / ro.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["ranker"]["rank_score_source"] == "input"


def test_campaign_sent_after_the_pull_is_warned(tmp_path):
    run_dir = tmp_path / "late"
    _write(run_dir, "a.jsonl", [_line("A", (0.9, 0.5, 0.6))],
           {PROD: {"submitted_lines": 1, "min_confidence": 0.55, "labels_submitted": 1,
                   "rig_masked": True, "last_submission_utc": "2026-11-05T00:00:00Z"}})
    assert _run([run_dir], _ranker(tmp_path / "rk", [_label(1, "A", 0.5, 0.6)], {"city-a:1": 0.2})) == 0
    summary = json.loads((_out(run_dir) / ro.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert any("after the ranker's rawLabels pull" in w for w in summary["warnings"])
