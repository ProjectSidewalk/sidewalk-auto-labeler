"""Unit tests for export_benchmark.py: benchmark sampling, bundle records, and the
archive reconciliation that makes "is this the same data we processed?" mechanical.

The sampler lives here rather than with the transitional spot_check_gallery, which
imports it; deleting that viewer must not take the sampler's coverage with it.
No test touches the network — fetch_native is never called.
"""
import csv
import json

import export_benchmark as eb

# --- sampling: choose_panos -----------------------------------------------------------


def _record(pid, n_dets, coords=None, lat=None, lng=None):
    dets = coords or [
        {"x_normalized": 0.1 * (i + 1), "y_normalized": 0.5, "confidence": 0.9}
        for i in range(n_dets)]
    pano = {"panorama_id": pid, "capture_date": "2024-06"}
    if lat is not None:
        pano["lat"], pano["lng"] = lat, lng
    return {"pano": pano, "detections": dets}


def test_choose_panos_groups_and_counts():
    records = ([_record(f"D{i}", (i % 9) + 1) for i in range(30)]
               + [_record(f"E{i}", 0) for i in range(8)])
    chosen = eb.choose_panos(records, sample=15, empty_sample=3, seed=0)
    groups = [grp for _, grp in chosen]
    assert groups.count("top") == eb.TOP_N_BY_COUNT
    assert groups.count("empty") == 3
    assert len(chosen) == 15 + 3
    # No pano appears twice (top panos must be excluded from the random pool).
    pids = [r["pano"]["panorama_id"] for r, _ in chosen]
    assert len(pids) == len(set(pids))
    # The densest panos really are the top group.
    top_counts = [len(r["detections"]) for r, grp in chosen if grp == "top"]
    assert min(top_counts) >= max(
        len(r["detections"]) for r, grp in chosen if grp == "random")


def test_choose_panos_seed_reproducible():
    records = ([_record(f"D{i}", 1) for i in range(50)]
               + [_record(f"E{i}", 0) for i in range(10)])
    pick = lambda seed: [r["pano"]["panorama_id"]
                         for r, _ in eb.choose_panos(records, 10, 2, seed)]
    assert pick(7) == pick(7)
    assert pick(7) != pick(8)


def test_choose_panos_small_inputs():
    # Fewer panos than requested: everything is included, nothing crashes.
    # (Coordinate-less records degrade gracefully to the non-spatial path.)
    records = [_record("D0", 2), _record("E0", 0)]
    chosen = eb.choose_panos(records, sample=100, empty_sample=10, seed=0)
    assert {grp for _, grp in chosen} == {"top", "empty"}
    assert len(chosen) == 2


# --- sampling: spatial de-clustering ---------------------------------------------------

def _pairwise_min_m(chosen):
    pts = [eb._coords(r) for r, _ in chosen]
    return min((eb._haversine_m(*pts[i], *pts[j])
                for i in range(len(pts)) for j in range(i + 1, len(pts))), default=float("inf"))


def test_choose_panos_declusters_to_one_per_location():
    # Four tight clusters of 5 panos (~13 m within a cluster, ~780 m between
    # clusters). At 30 m spacing at most one pano per cluster can be selected,
    # even though 20 are requested.
    recs = []
    for ci, lng0 in enumerate([-122.00, -121.99, -121.98, -121.97]):
        for j in range(5):
            recs.append(_record(f"D{ci}_{j}", 3, lat=45.0 + 0.00003 * j, lng=lng0))
    chosen = eb.choose_panos(recs, sample=20, empty_sample=0, seed=1, min_spacing=30)
    assert len(chosen) == 4                       # one survivor per cluster
    assert _pairwise_min_m(chosen) >= 30 - 1e-6   # and they respect the spacing


def test_choose_panos_top_from_distinct_intersections():
    # Five very dense panos at ONE corner + six moderately dense, far apart.
    # Only one same-corner pano can be 'top'; the rest come from the far ones.
    recs = [_record(f"C{j}", 10, lat=45.0 + 0.00002 * j, lng=-122.0) for j in range(5)]
    recs += [_record(f"F{k}", 6, lat=45.0 + 0.01 * (k + 1), lng=-122.0) for k in range(6)]
    chosen = eb.choose_panos(recs, sample=5, empty_sample=0, seed=0, min_spacing=30)
    top = [r for r, grp in chosen if grp == "top"]
    assert len(top) == eb.TOP_N_BY_COUNT
    assert sum(r["pano"]["panorama_id"].startswith("C") for r in top) == 1
    assert _pairwise_min_m(chosen) >= 30 - 1e-6


def test_choose_panos_min_spacing_zero_keeps_clusters():
    # Ten panos all within ~30 m; with spacing disabled, none are dropped.
    recs = [_record(f"D{j}", 2, lat=45.0 + 0.00003 * j, lng=-122.0) for j in range(10)]
    chosen = eb.choose_panos(recs, sample=10, empty_sample=0, seed=0, min_spacing=0)
    assert len(chosen) == 10


def test_choose_panos_spacing_is_cross_stratum():
    # An 'empty' pano sitting right next to a detection pano must be dropped:
    # spacing holds across strata, not just within them.
    recs = [_record("D0", 4, lat=45.0, lng=-122.0),
            _record("E_near", 0, lat=45.00002, lng=-122.0),      # ~2 m from D0
            _record("E_far", 0, lat=45.02, lng=-122.0)]          # ~2.2 km away
    chosen = eb.choose_panos(recs, sample=5, empty_sample=5, seed=0, min_spacing=30)
    ids = {r["pano"]["panorama_id"] for r, _ in chosen}
    assert ids == {"D0", "E_far"}                # E_near dropped for being too close to D0


# --- write_bundle_records --------------------------------------------------------------

def _write_results(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_write_bundle_records_tags_groups_and_writes_provenance(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_results(results, [_record(f"D{i}", 2, lat=45.0 + 0.01 * i, lng=-122.0)
                             for i in range(10)]
                   + [_record(f"E{i}", 0, lat=46.0 + 0.01 * i, lng=-122.0) for i in range(5)])
    bundle = tmp_path / "bundle"

    records_path = eb.write_bundle_records(results, bundle, sample=8, empty_sample=3,
                                           seed=0, min_spacing=30)

    rows = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 + 3
    assert {r["benchmark_group"] for r in rows} == {"top", "random", "empty"}
    # Records keep their Stage-1 shape, and land sorted by id for a stable diff.
    assert all("detections" in r and "pano" in r for r in rows)
    assert [r["pano"]["panorama_id"] for r in rows] == sorted(
        r["pano"]["panorama_id"] for r in rows)

    meta = json.loads((bundle / "sample.json").read_text(encoding="utf-8"))
    assert meta["selected"] == 11 and meta["source_record_count"] == 15
    assert meta["seed"] == 0 and meta["min_spacing_m"] == 30
    assert meta["groups"]["empty"] == 3


def test_write_bundle_records_never_resamples_an_existing_bundle(tmp_path):
    """A reviewer's verdicts are keyed to the sample; silently redrawing it would
    invalidate them."""
    results = tmp_path / "results.jsonl"
    _write_results(results, [_record(f"D{i}", 2, lat=45.0 + 0.01 * i, lng=-122.0)
                             for i in range(10)])
    bundle = tmp_path / "bundle"
    first = eb.write_bundle_records(results, bundle, 5, 0, seed=0, min_spacing=30)
    before = first.read_text(encoding="utf-8")

    again = eb.write_bundle_records(results, bundle, 9, 0, seed=99, min_spacing=0)
    assert again == first and again.read_text(encoding="utf-8") == before


# --- reconcile -------------------------------------------------------------------------

def _archive(tmp_path, panos, dirname="panos"):
    """Build a bundle dir with {pano_id: bytes} written into <dirname>/."""
    panos_dir = tmp_path / dirname
    panos_dir.mkdir(parents=True)
    for pid, blob in panos.items():
        (panos_dir / f"{pid}.jpg").write_bytes(blob)
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    return records, panos_dir


def _index(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["panorama_id"]: row for row in csv.DictReader(f)}


def test_reconcile_clean_archive_writes_index(tmp_path):
    records, panos_dir = _archive(tmp_path, {"A": b"aaa", "B": b"bb"})
    decayed, incomplete, extra = eb.reconcile(records, panos_dir, ["A", "B"])
    assert (decayed, incomplete, extra) == (0, 0, 0)

    rows = _index(tmp_path / "index.csv")
    assert set(rows) == {"A", "B"}
    assert rows["A"]["filename"] == "A.jpg" and rows["A"]["bytes"] == "3"
    assert len(rows["A"]["sha256"]) == 64
    assert not (tmp_path / "decayed.txt").exists()


def test_reconcile_separates_decay_from_a_failed_fetch(tmp_path):
    records, panos_dir = _archive(tmp_path, {"A": b"aaa"})
    failures = {"GONE1": (eb.GONE, "image no longer exists on Mapillary"),
                "FLAKY": (eb.ERROR, "connection reset")}
    decayed, incomplete, extra = eb.reconcile(
        records, panos_dir, ["A", "GONE1", "FLAKY"], failures)

    # Only the pano the source declared gone counts as decay; the flaky one keeps the
    # archive marked incomplete so the caller exits non-zero and re-runs.
    assert (decayed, incomplete, extra) == (1, 1, 0)
    assert (tmp_path / "decayed.txt").read_text(encoding="utf-8") == "GONE1\n"


def test_reconcile_missing_without_a_failure_record_is_not_decay(tmp_path):
    """An unexplained gap is 'incomplete', never silently blessed as decay."""
    records, panos_dir = _archive(tmp_path, {"A": b"aaa"})
    decayed, incomplete, _ = eb.reconcile(records, panos_dir, ["A", "B"])
    assert (decayed, incomplete) == (0, 1)
    assert not (tmp_path / "decayed.txt").exists()


def test_reconcile_flags_contamination(tmp_path):
    records, panos_dir = _archive(tmp_path, {"A": b"aaa", "STRAY": b"x"})
    decayed, incomplete, extra = eb.reconcile(records, panos_dir, ["A"])
    assert (decayed, incomplete, extra) == (0, 0, 1)
    assert set(_index(tmp_path / "index.csv")) == {"A"}   # unbacked files never indexed


def test_reconcile_clears_a_stale_decayed_list(tmp_path):
    records, panos_dir = _archive(tmp_path, {"A": b"aaa"})
    eb.reconcile(records, panos_dir, ["A", "B"], {"B": (eb.GONE, "gone")})
    assert (tmp_path / "decayed.txt").exists()

    (panos_dir / "B.jpg").write_bytes(b"bbb")            # a resume recovered it
    eb.reconcile(records, panos_dir, ["A", "B"])
    assert not (tmp_path / "decayed.txt").exists()


def test_reconcile_reuses_hashes_for_unchanged_panos(tmp_path, monkeypatch):
    records, panos_dir = _archive(tmp_path, {"A": b"aaa", "B": b"bb"})
    eb.reconcile(records, panos_dir, ["A", "B"])
    first = _index(tmp_path / "index.csv")

    (panos_dir / "B.jpg").write_bytes(b"bbbb")           # different size -> must re-hash
    hashed = []
    real_sha = eb._sha256
    monkeypatch.setattr(eb, "_sha256", lambda p, *a, **k: hashed.append(p.stem) or real_sha(p))
    eb.reconcile(records, panos_dir, ["A", "B"])

    assert hashed == ["B"]                               # A's cached sha was reused
    second = _index(tmp_path / "index.csv")
    assert second["A"]["sha256"] == first["A"]["sha256"]
    assert second["B"]["sha256"] != first["B"]["sha256"] and second["B"]["bytes"] == "4"


def test_reconcile_drops_empty_files_so_they_refetch(tmp_path):
    """A 0-byte file carries no imagery but would satisfy the resume check forever."""
    records, panos_dir = _archive(tmp_path, {"A": b"aaa", "TRUNC": b""})
    decayed, incomplete, extra = eb.reconcile(records, panos_dir, ["A", "TRUNC"])
    assert (decayed, incomplete, extra) == (0, 1, 0)
    assert not (panos_dir / "TRUNC.jpg").exists()
    assert set(_index(tmp_path / "index.csv")) == {"A"}


# --- manifest placement ----------------------------------------------------------------

def test_manifest_dir_sits_beside_a_panos_dir(tmp_path):
    assert eb.manifest_dir(tmp_path / "clovis" / "panos") == tmp_path / "clovis"


def test_manifest_dir_stays_inside_a_bare_output_dir(tmp_path):
    """`--out /archive/<city>` must keep its manifest in the city dir — one level up is
    shared with every other city, whose runs would overwrite it."""
    assert eb.manifest_dir(tmp_path / "clovis") == tmp_path / "clovis"


def test_reconcile_writes_manifest_inside_a_bare_output_dir(tmp_path):
    records, panos_dir = _archive(tmp_path, {"A": b"aaa"}, dirname="clovis")
    eb.reconcile(records, panos_dir, ["A"])
    assert (panos_dir / "index.csv").exists()
    assert not (tmp_path / "index.csv").exists()
