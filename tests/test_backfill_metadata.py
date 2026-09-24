"""Unit tests for scripts/backfill_metadata.py.

It rewrites finished runs in place — the makelab archive, the local run, and the
benchmark bundle each hold a copy of a city — so the merge rules and the "only touch
mapillary lines that need it" gate are what these cover. No test fetches anything.
"""
import json

import pytest

import backfill_metadata as bf


def _line(pid, source="mapillary", **pano):
    return {"pano": {"panorama_id": pid, "source": source, **pano}, "detections": []}


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def _run(monkeypatch, jsonl, fetched, argv_extra=()):
    """Drive main() with a stubbed fetch; returns the ids it asked for."""
    asked = []

    def fake_fetch(pano_id):
        asked.append(pano_id)
        return fetched.get(pano_id)

    monkeypatch.setattr(bf, "fetch_fields", fake_fetch)
    monkeypatch.setattr(bf.mapillary, "prepare", lambda: None)
    monkeypatch.setattr("sys.argv", ["backfill_metadata.py", str(jsonl), *argv_extra])
    bf.main()
    return asked


def test_backfills_only_mapillary_lines_that_lack_metadata(monkeypatch, tmp_path):
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [
        _line("M_NEW"),                                        # needs a backfill
        _line("M_DONE", source_metadata={"make": "NCTECH"}),    # already has one
        _line("G1", source="gsv"),                             # never touched
    ])
    fields = {"camera_make": "GoPro", "source_metadata": {"make": "GoPro"}}

    asked = _run(monkeypatch, jsonl, {"M_NEW": fields})

    assert asked == ["M_NEW"]
    out = {r["pano"]["panorama_id"]: r["pano"] for r in _read(jsonl)}
    assert out["M_NEW"]["camera_make"] == "GoPro"
    assert out["M_DONE"]["source_metadata"] == {"make": "NCTECH"}   # untouched
    assert "source_metadata" not in out["G1"]
    assert len(out) == 3                                           # nothing dropped


def test_unfetchable_panos_leave_the_line_intact_for_a_retry(monkeypatch, tmp_path):
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M_GONE"), _line("M_OK")])
    fields = {"camera_make": "GoPro", "source_metadata": {"make": "GoPro"}}

    _run(monkeypatch, jsonl, {"M_OK": fields})   # M_GONE returns None

    out = {r["pano"]["panorama_id"]: r["pano"] for r in _read(jsonl)}
    assert "source_metadata" not in out["M_GONE"]      # still eligible next run
    assert out["M_OK"]["source_metadata"] == {"make": "GoPro"}


def test_cache_is_written_incrementally_and_reused_across_copies(monkeypatch, tmp_path):
    """The whole point of --cache: one fetch pass serves a city's several copies."""
    cache = tmp_path / ".meta_cache.jsonl"
    fields = {"camera_make": "GoPro", "source_metadata": {"make": "GoPro"}}

    run_jsonl = tmp_path / "results.jsonl"
    _write(run_jsonl, [_line("M1")])
    assert _run(monkeypatch, run_jsonl, {"M1": fields}, ["--cache", str(cache)]) == ["M1"]
    assert bf.load_cache(cache) == {"M1": fields}

    bundle_jsonl = tmp_path / "records.jsonl"
    _write(bundle_jsonl, [_line("M1")])
    # Second copy of the same city: served entirely from the cache, no fetch.
    assert _run(monkeypatch, bundle_jsonl, {}, ["--cache", str(cache)]) == []
    assert _read(bundle_jsonl)[0]["pano"]["source_metadata"] == {"make": "GoPro"}


def test_force_refetches_instead_of_replaying_the_cache(monkeypatch, tmp_path):
    """--force exists to refresh stale provenance; honouring the cache would make it a
    no-op that silently rewrites the same values."""
    cache = tmp_path / ".meta_cache.jsonl"
    cache.write_text(json.dumps({"id": "M1", "fields": {"source_metadata": {"v": "old"}}}) + "\n",
                     encoding="utf-8")
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M1", source_metadata={"v": "old"})])

    asked = _run(monkeypatch, jsonl, {"M1": {"source_metadata": {"v": "new"}}},
                 ["--cache", str(cache), "--force"])

    assert asked == ["M1"]
    assert _read(jsonl)[0]["pano"]["source_metadata"] == {"v": "new"}
    # The refreshed value wins on the next load — the cache file is append-only.
    assert bf.load_cache(cache)["M1"]["source_metadata"] == {"v": "new"}


# --- --pose: the offline camera_pitch/camera_roll pass (#42) ---------------------------

# A real rotation (runs/richmond pano 2163793620710887); PS's viewer derived these angles.
RVEC = [1.3857263583832, 0.71804330335161, -0.58250746512038]
PITCH, ROLL = 2.231776541825151, -6.354952531976068


def _pose_run(monkeypatch, jsonl, *argv_extra):
    def no_network(*a, **k):
        raise AssertionError("--pose must not touch the network or need a token")
    monkeypatch.setattr(bf, "fetch_fields", no_network)
    monkeypatch.setattr(bf.mapillary, "prepare", no_network)
    monkeypatch.setattr("sys.argv", ["backfill_metadata.py", str(jsonl), "--pose", *argv_extra])
    bf.main()


def test_pose_pass_fills_from_source_metadata_and_keeps_line_order(monkeypatch, tmp_path):
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [
        _line("M_OK", camera_pitch=None, camera_roll=None,
              source_metadata={"computed_rotation": RVEC}),
        _line("G1", source="gsv", camera_pitch=1.0, camera_roll=0.5),
        _line("M_FLIPPED", camera_pitch=None, camera_roll=None,
              source_metadata={"computed_rotation": [3.14159, 0.0, 0.0]}),   # upside down
        _line("M_BARE", camera_pitch=None, camera_roll=None),                 # no metadata
    ])
    _pose_run(monkeypatch, jsonl)
    out = [r["pano"] for r in _read(jsonl)]
    # Same lines, same order: send_to_ps.py's sidecars are line numbers.
    assert [p["panorama_id"] for p in out] == ["M_OK", "G1", "M_FLIPPED", "M_BARE"]
    assert (out[0]["camera_pitch"], out[0]["camera_roll"]) == (
        pytest.approx(PITCH, abs=1e-9), pytest.approx(ROLL, abs=1e-9))
    assert out[0]["camera_pose_source"] == "mapillary_computed_rotation"
    assert "camera_pose_source" not in out[1]                                # GSV: no key
    assert out[2]["camera_pose_source"] is None                              # key, no angles
    assert out[3]["camera_pose_source"] is None
    assert (out[1]["camera_pitch"], out[1]["camera_roll"]) == (1.0, 0.5)     # GSV untouched
    assert out[2]["camera_pitch"] is None and out[2]["camera_roll"] is None  # not a pose
    assert out[3]["camera_pitch"] is None


def test_pose_dry_run_counts_and_writes_nothing(monkeypatch, tmp_path):
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M1", camera_pitch=None, camera_roll=None,
                         source_metadata={"computed_rotation": RVEC})])
    before = jsonl.read_bytes()
    counts = bf.backfill_pose(jsonl, dry_run=True)
    assert counts == {bf.POSE_FILLED: 1}
    assert jsonl.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_pose_refuses_in_place_rewrite_of_a_submitted_file(monkeypatch, tmp_path):
    """A submitted file's sha256 is what send_to_ps.py's guard holds; changing it in place
    would make that campaign refuse to resume. --out is the route, and it leaves the input
    byte-identical."""
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M1", camera_pitch=None, camera_roll=None,
                         source_metadata={"computed_rotation": RVEC})])
    (tmp_path / "results.jsonl.submission.json").write_text("{}", encoding="utf-8")
    before = jsonl.read_bytes()
    with pytest.raises(SystemExit, match="campaign state"):
        _pose_run(monkeypatch, jsonl)
    assert jsonl.read_bytes() == before

    out = tmp_path / "results.pose.jsonl"
    _pose_run(monkeypatch, jsonl, "--out", str(out))
    assert jsonl.read_bytes() == before
    assert _read(out)[0]["pano"]["camera_roll"] == pytest.approx(ROLL, abs=1e-9)


def test_pose_already_set_gains_the_missing_provenance_key():
    # A block a fresh run wrote before camera_pose_source existed: angles untouched,
    # the key filled in.
    rec = _line("M1", camera_pitch=1.5, camera_roll=-0.5,
                source_metadata={"computed_rotation": RVEC})
    assert bf.fill_pose(rec) == bf.POSE_ALREADY
    assert (rec["pano"]["camera_pitch"], rec["pano"]["camera_roll"]) == (1.5, -0.5)
    assert rec["pano"]["camera_pose_source"] == "mapillary_computed_rotation"


@pytest.mark.parametrize("sidecar", ["results.jsonl.submitted",
                                     "results.jsonl.submitted.laurens-prod",
                                     "results.jsonl.band-0.3-0.55.submitted"])
def test_pose_refuses_a_file_with_only_a_resume_sidecar(monkeypatch, tmp_path, sidecar):
    """A sidecar with no record beside it (a campaign begun before records existed, a
    moved-aside copy, a band) still numbers this file's lines, so it guards the file too."""
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M1", camera_pitch=None, camera_roll=None,
                         source_metadata={"computed_rotation": RVEC})])
    (tmp_path / sidecar).write_text("1\n", encoding="utf-8")
    before = jsonl.read_bytes()
    with pytest.raises(SystemExit, match="campaign state"):
        _pose_run(monkeypatch, jsonl)
    assert jsonl.read_bytes() == before


def test_pose_refuses_an_out_target_under_a_campaign(monkeypatch, tmp_path):
    """--out onto a file some campaign already holds would change THAT file's hash."""
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M1", camera_pitch=None, camera_roll=None,
                         source_metadata={"computed_rotation": RVEC})])
    out = tmp_path / "results.pose.jsonl"
    _write(out, [_line("M1")])
    (tmp_path / "results.pose.jsonl.submitted").write_text("1\n", encoding="utf-8")
    before = out.read_bytes()
    with pytest.raises(SystemExit, match="overwriting it as --out"):
        _pose_run(monkeypatch, jsonl, "--out", str(out))
    assert out.read_bytes() == before


def test_pose_refuses_out_equal_to_input(monkeypatch, tmp_path):
    jsonl = tmp_path / "results.jsonl"
    _write(jsonl, [_line("M1", camera_pitch=None, camera_roll=None,
                         source_metadata={"computed_rotation": RVEC})])
    before = jsonl.read_bytes()
    with pytest.raises(SystemExit, match="--out is the input file"):
        _pose_run(monkeypatch, jsonl, "--out", str(tmp_path / "." / "results.jsonl"))
    assert jsonl.read_bytes() == before


def test_rewrite_submitted_rewrites_in_place_and_keeps_line_numbers(monkeypatch, tmp_path):
    """The deliberate override is allowed, and a blank line stays a line: the sidecar's
    numbers count physical lines, blanks included, so they still name the same panos."""
    jsonl = tmp_path / "results.jsonl"
    posed = _line("M2", camera_pitch=None, camera_roll=None,
                  source_metadata={"computed_rotation": RVEC})
    jsonl.write_text(json.dumps(_line("G1", source="gsv")) + "\n\n" + json.dumps(posed) + "\n",
                     encoding="utf-8")
    (tmp_path / "results.jsonl.submission.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results.jsonl.submitted").write_text("1\n3\n", encoding="utf-8")
    _pose_run(monkeypatch, jsonl, "--rewrite-submitted")
    lines = jsonl.read_text(encoding="utf-8").split("\n")
    assert len(lines) == 4 and lines[1] == "" and lines[3] == ""   # 3 lines, final newline
    assert json.loads(lines[0])["pano"]["panorama_id"] == "G1"
    third = json.loads(lines[2])["pano"]
    assert third["panorama_id"] == "M2"
    assert third["camera_roll"] == pytest.approx(ROLL, abs=1e-9)
