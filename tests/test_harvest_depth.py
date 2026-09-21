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


def test_interrupted_part_files_are_swept(tmp_path):
    depth_dir = tmp_path / "depth"
    archive(depth_dir, "A")
    (depth_dir / "B.json.gz.part").write_bytes(b"half a payload")

    hd.reconcile(depth_dir, ["A", "B"], failures={}, attempted={"A"})

    assert not (depth_dir / "B.json.gz.part").exists()


def test_no_depth_alarm_fires_only_on_an_implausible_rate():
    """NO_DEPTH fired zero times across 170,932 production panoramas, so a sudden crop of
    it means the response shape moved — and caching that would skip them forever."""
    assert not hd.no_depth_looks_poisoned(0, 1000)
    assert not hd.no_depth_looks_poisoned(2, 3)            # too few to mean anything
    assert not hd.no_depth_looks_poisoned(20, 1000)        # 2%, under the threshold
    assert hd.no_depth_looks_poisoned(200, 1000)           # 20%, not plausible
    assert hd.no_depth_looks_poisoned(1000, 1000)          # everything — the real signal
