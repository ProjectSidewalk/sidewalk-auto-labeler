"""Unit tests for scripts/backfill_metadata.py.

It rewrites finished runs in place — the makelab archive, the local run, and the
benchmark bundle each hold a copy of a city — so the merge rules and the "only touch
mapillary lines that need it" gate are what these cover. No test fetches anything.
"""
import json

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
