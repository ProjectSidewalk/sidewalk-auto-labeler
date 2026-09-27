"""Reconciliation tests for harvest_depth.py — the bookkeeping that decides whether a
depth archive is trustworthy.

Narrow on purpose: these cover the state that survives *between* runs, because that is
where a mistake is silent and permanent. `gone.txt` was previously rebuilt from each
pass's own failures, so `--verify` (which fetches nothing, and so has no failures) deleted
it and downgraded a complete archive to PARTIAL. Nothing downstream could notice.

No network — reconcile is pure filesystem, and conftest blocks streetlevel besides.
"""
import base64
import gzip
import json
import struct

import pytest

import depth as depthlib
import harvest_depth as hd

GROUND = (0.0, 0.0, -1.0, 2.5)


def _blob(width=8, height=4):
    planes = [GROUND, GROUND]
    indices = [depthlib.SKY] * (width * height // 2) + [1] * (width * height // 2)
    off = depthlib.HEADER_BYTES
    header = bytes([off]) + struct.pack("<HHH", len(planes), width, height) + bytes([off])
    tail = b"".join(struct.pack("<ffff", *p) for p in planes)
    return base64.urlsafe_b64encode(header + bytes(indices) + tail).decode().rstrip("=")


def archive(depth_dir, pid):
    depth_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(depth_dir / f"{pid}.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"pano_id": pid, "depth_b64": _blob(), "fetched_at": "2026-08-09T00:00:00+00:00"}, f)


def test_verify_pass_preserves_gone_ids(tmp_path, capsys):
    """--verify fetches nothing, so it has no failures to rebuild gone.txt from. It must
    read the existing list rather than overwrite it with an empty set."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    depth_dir.joinpath("gone.txt").write_text("B\n", encoding="utf-8")

    gone, failed, pending, anomalies = hd.reconcile(depth_dir, ["A", "B"])

    assert (gone, failed, pending, anomalies) == (1, 0, 0, 0)
    assert hd.load_gone(depth_dir) == {"B"}
    assert "OK with gaps" in capsys.readouterr().out


def test_partial_pass_preserves_gone_ids(tmp_path):
    """A --limit pass attempts a subset, so it does not re-encounter the gone ids either."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    depth_dir.joinpath("gone.txt").write_text("B\n", encoding="utf-8")

    hd.reconcile(depth_dir, ["A", "B", "C"], failures={}, attempted={"A"})

    assert hd.load_gone(depth_dir) == {"B"}


def test_newly_found_gone_ids_are_merged_not_replaced(tmp_path):
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    depth_dir.joinpath("gone.txt").write_text("B\n", encoding="utf-8")

    hd.reconcile(depth_dir, ["A", "B", "C"], failures={"C": (hd.GONE, "response code 2")},
                 attempted={"C"})

    assert hd.load_gone(depth_dir) == {"B", "C"}


def test_recovered_pano_drops_out_of_gone(tmp_path):
    """The one case that must still shrink the list: the payload is actually here now."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    archive(depth_dir, "B")
    depth_dir.joinpath("gone.txt").write_text("B\n", encoding="utf-8")

    hd.reconcile(depth_dir, ["A", "B"])

    assert hd.load_gone(depth_dir) == set()


def test_gone_ids_are_not_refetched(tmp_path):
    """GONE is documented as deterministic and never retried, so it must be a real skip
    cache — not merely a report written by reconcile and never read back."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    depth_dir.joinpath("gone.txt").write_text("B\n", encoding="utf-8")
    depth_dir.joinpath("no_depth.txt").write_text("C\n", encoding="utf-8")

    skip = hd.load_no_depth(depth_dir) | hd.load_gone(depth_dir)
    todo = [pid for pid in ["A", "B", "C", "D"]
            if pid not in skip and not (depth_dir / f"{pid}.json.gz").exists()]

    assert todo == ["D"]


def alter_in_place(path):
    """Change a file's bytes without changing its size or breaking it.

    Bytes 4-7 of a gzip member are the MTIME field: informational, ignored on read. So the
    file still decompresses to exactly the same payload and only its sha256 moves — which
    is the case the size-only shortcut is blind to, and the one --rehash exists for.
    """
    raw = bytearray(path.read_bytes())
    raw[4] ^= 0xFF
    path.write_bytes(bytes(raw))


def test_rehash_detects_a_file_altered_in_place(tmp_path, capsys):
    """Same byte count, still readable, different bytes: the incremental index trusts size
    alone, so only --rehash can catch this. It has to surface as an anomaly."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    hd.reconcile(depth_dir, ["A"])
    alter_in_place(depth_dir / "A.json.gz")

    _, _, _, anomalies = hd.reconcile(depth_dir, ["A"], rehash=True)

    assert anomalies == 1
    assert "changed on disk" in capsys.readouterr().out


def test_an_altered_file_keeps_its_recorded_digest(tmp_path):
    """#97 review: reporting the change once and then writing the new sha256 over the
    recorded one launders it -- the next --verify --rehash would come back clean. The prior
    row is carried forward instead, so the anomaly stands until someone deals with it."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    hd.reconcile(depth_dir, ["A"])
    _, rows = _read_index(depth_dir)
    recorded = rows[0]["sha256"]
    alter_in_place(depth_dir / "A.json.gz")

    assert hd.reconcile(depth_dir, ["A"], rehash=True)[3] == 1
    assert _read_index(depth_dir)[1][0]["sha256"] == recorded
    assert hd.reconcile(depth_dir, ["A"], rehash=True)[3] == 1    # the next --verify --rehash
    assert _read_index(depth_dir)[1][0]["sha256"] == recorded


def test_size_only_reconcile_misses_it(tmp_path):
    """The flip side, stated so the limitation is deliberate rather than assumed: without
    --rehash a same-size change is invisible."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    hd.reconcile(depth_dir, ["A"])
    alter_in_place(depth_dir / "A.json.gz")

    assert hd.reconcile(depth_dir, ["A"])[3] == 0


def test_rehash_reports_damage_that_breaks_decompression(tmp_path, capsys):
    """Real bit rot usually trips gzip's own CRC first. That must land as an anomaly too,
    rather than quietly dropping the panorama out of index.csv."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    hd.reconcile(depth_dir, ["A"])

    path = depth_dir / "A.json.gz"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF                                   # breaks the trailer, keeps the size
    path.write_bytes(bytes(raw))

    _, _, _, anomalies = hd.reconcile(depth_dir, ["A"], rehash=True)

    assert anomalies == 1
    assert "unreadable" in capsys.readouterr().out
    assert [r["panorama_id"] for r in _read_index(depth_dir)[1]] == ["A"]   # row kept


def test_interrupted_part_files_are_swept(tmp_path):
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    (depth_dir / "B.json.gz.part").write_bytes(b"half a payload")

    hd.reconcile(depth_dir, ["A", "B"], failures={}, attempted={"A"})

    assert not (depth_dir / "B.json.gz.part").exists()


def _read_index(depth_dir):
    import csv
    with open(depth_dir / "index.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_reindex_rewrites_an_old_index_with_the_new_columns(tmp_path):
    """#47: an index written under the old INDEX_FIELDS (no stand-in columns, the old
    spread) is rebuilt from the archived files, offline."""
    import csv
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    hd.reconcile(depth_dir, ["A"])
    old_fields = [f for f in hd.INDEX_FIELDS
                  if f not in ("n_standin_planes", "standin_pixel_share")]
    _, rows = _read_index(depth_dir)
    with open(depth_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "height_spread_m": "9.9999"})   # a stale derived value

    assert hd.reindex(depth_dir, ["A"])[3] == 0
    fields, rows = _read_index(depth_dir)
    assert fields == hd.INDEX_FIELDS
    # the fixture's floor is one exactly level plane: a stand-in, the dominant one
    assert rows[0]["n_standin_planes"] == "1"
    assert rows[0]["standin_pixel_share"] == "0.5000"
    assert rows[0]["height_spread_m"] == "0.0000"


def test_reconcile_does_not_reuse_rows_from_an_older_schema(tmp_path):
    """Without --reindex too: an index missing a current column is recomputed, never
    copied through with blanks and a stale spread. Its sha256s are still compared: the
    recorded digest is the real one here, so the pass is clean."""
    import csv
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    path = depth_dir / "A.json.gz"
    with open(depth_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hd.INDEX_FIELDS[:-2])
        w.writerow(["A", "A.json.gz", path.stat().st_size, hd._sha256(path), 2, 1,
                    "2.5000", "0.000", "0.5000", "9.9999", "0.5000"])
    assert hd.reconcile(depth_dir, ["A"])[3] == 0
    _, rows = _read_index(depth_dir)
    assert rows[0]["height_spread_m"] == "0.0000"
    assert rows[0]["n_standin_planes"] == "1"


def test_an_old_schema_row_with_a_wrong_digest_stays_an_anomaly(tmp_path):
    """The same recompute with a recorded digest that does not match: an anomaly, and the
    row is carried forward as recorded (it is evidence, not a value to refresh). A later
    pass without --rehash re-reads it rather than trusting its blank stand-in columns."""
    import csv
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    with open(depth_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hd.INDEX_FIELDS[:-2])
        w.writerow(["A", "A.json.gz", (depth_dir / "A.json.gz").stat().st_size, "x" * 64,
                    2, 1, "2.5000", "0.000", "0.5000", "9.9999", "0.5000"])
    assert hd.reconcile(depth_dir, ["A"])[3] == 1
    assert _read_index(depth_dir)[1][0]["sha256"] == "x" * 64
    assert hd.reconcile(depth_dir, ["A"])[3] == 1                  # not short-circuited


def test_reindex_refuses_a_missing_file(tmp_path):
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    archive(depth_dir, "B")
    hd.reconcile(depth_dir, ["A", "B"])
    before = (depth_dir / "index.csv").read_bytes()
    (depth_dir / "B.json.gz").unlink()

    with pytest.raises(SystemExit) as exc:
        hd.reindex(depth_dir, ["A", "B"])
    assert "REFUSING" in str(exc.value)
    assert (depth_dir / "index.csv").read_bytes() == before


def test_reindex_refuses_a_zero_byte_file(tmp_path):
    """reconcile deletes an empty file before anything else, which would drop its row;
    the reindex pre-check has to count it as missing and refuse first."""
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    archive(depth_dir, "B")
    hd.reconcile(depth_dir, ["A", "B"])
    before = (depth_dir / "index.csv").read_bytes()
    (depth_dir / "B.json.gz").write_bytes(b"")

    with pytest.raises(SystemExit) as exc:
        hd.reindex(depth_dir, ["A", "B"])
    assert "REFUSING" in str(exc.value)
    assert (depth_dir / "index.csv").read_bytes() == before
    assert (depth_dir / "B.json.gz").exists()                      # not deleted either


def test_reindex_writes_nothing_when_it_finds_an_anomaly(tmp_path):
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    archive(depth_dir, "B")
    hd.reconcile(depth_dir, ["A", "B"])
    before = (depth_dir / "index.csv").read_bytes()
    alter_in_place(depth_dir / "B.json.gz")

    assert hd.reindex(depth_dir, ["A", "B"])[3] == 1
    assert (depth_dir / "index.csv").read_bytes() == before
    assert not (depth_dir / "index.csv.tmp").exists()


def test_reindex_refuses_without_an_index(tmp_path):
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    with pytest.raises(SystemExit):
        hd.reindex(depth_dir, ["A"])


def test_no_depth_alarm_fires_only_on_an_implausible_rate():
    """NO_DEPTH fired zero times across 170,932 production panoramas, so a sudden crop of
    it means the response shape moved — and caching that would skip them forever."""
    assert not hd.no_depth_looks_poisoned(0, 1000)
    assert not hd.no_depth_looks_poisoned(2, 3)            # too few to mean anything
    assert not hd.no_depth_looks_poisoned(20, 1000)        # 2%, under the threshold
    assert hd.no_depth_looks_poisoned(200, 1000)           # 20%, not plausible
    assert hd.no_depth_looks_poisoned(1000, 1000)          # everything — the real signal


def test_gsv_runs_are_not_refused_by_their_raw_source_string():
    """GSV records carry streetlevel's raw source ('launch'), never 'gsv'; refusing
    anything but 'gsv' refused every real GSV run."""
    assert hd.is_gsv_source("launch") and hd.is_gsv_source("scout")
    assert hd.is_gsv_source("gsv") and hd.is_gsv_source(None)
    assert not hd.is_gsv_source("mapillary") and not hd.is_gsv_source("panoramax")


def test_every_registered_non_gsv_source_is_refused():
    """Derived from the registry, not a literal list: a source added to SOURCE_NAMES is
    refused without touching harvest_depth."""
    import sources
    assert hd.NON_GSV_SOURCES == (set(sources.SOURCE_NAMES) | {"infra3d"}) - {"gsv"}
    assert all(not hd.is_gsv_source(s) for s in sources.SOURCE_NAMES if s != "gsv")
    assert not hd.is_gsv_source("infra3d")


def _results(run_dir, sources):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for i, src in enumerate(sources):
            f.write(json.dumps({"pano": {"panorama_id": f"P{i}", "source": src}}) + "\n")


def test_run_pano_ids_collects_every_records_source(tmp_path):
    _results(tmp_path, ["launch", None, "scout", "launch"])
    ids, sources = hd.run_pano_ids(tmp_path)
    assert ids == ["P0", "P1", "P2", "P3"]
    assert sources == {"launch", "scout"}
    assert hd.record_source_problem(sources) is None


def test_a_mixed_file_is_refused(tmp_path):
    """Only the first non-null source used to be read, so a GSV-first file with Mapillary
    records later passed."""
    _results(tmp_path, ["launch", "launch", "mapillary"])
    problem = hd.record_source_problem(hd.run_pano_ids(tmp_path)[1])
    assert problem and "mixed" in problem and "mapillary" in problem
    assert "mixed" not in hd.record_source_problem({"panoramax"})
    assert hd.record_source_problem(set()) is None


def _producer_run(run_dir, pano_ids, imagery_source="gsv"):
    """A run dir whose records come from the real producer chain (sources/gsv's pano
    block through main.build_output_line), so the test sees whatever `source` string GSV
    records actually store -- 'launch' today -- rather than one a test author assumed."""
    import main
    from conftest import make_process_result, make_provenance
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for pid in pano_ids:
            result = make_process_result(pano_id=pid)
            result["pano"] = dict(result["pano"], panorama_id=pid)
            f.write(json.dumps(main.build_output_line(result, make_provenance())) + "\n")
    (run_dir / "manifest.json").write_text(
        json.dumps({"imagery_source": imagery_source}), encoding="utf-8")


def test_main_verify_accepts_a_real_gsv_run(tmp_path, capsys):
    """Issue #99: main() refused every real GSV run, --verify included, because records
    store 'launch', not 'gsv'. Driven through main() so the whole source check is covered."""
    run_dir = tmp_path / "city"
    _producer_run(run_dir, ["A", "B"])
    assert {json.loads(l)["pano"]["source"]
            for l in (run_dir / "results.jsonl").read_text().splitlines()} == {"launch"}
    for pid in ("A", "B"):
        archive(run_dir / "depth", pid)

    with pytest.raises(SystemExit) as exc:
        hd.main([str(run_dir), "--verify"])
    assert exc.value.code in (0, None)
    out = capsys.readouterr().out
    assert "only GSV serves depth" not in out
    assert "matches the run 1:1" in out


def test_main_refuses_mapillary_records_without_a_manifest(tmp_path):
    """The records check is the one that still refuses when manifest.json is missing."""
    run_dir = tmp_path / "city"
    _results(run_dir, ["mapillary", "mapillary"])
    with pytest.raises(SystemExit) as exc:
        hd.main([str(run_dir), "--verify"])
    assert "mapillary" in str(exc.value.code) and "only GSV serves depth" in str(exc.value.code)
