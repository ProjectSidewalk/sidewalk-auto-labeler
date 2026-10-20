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


def test_write_bundle_records_applies_operational_threshold(tmp_path):
    """results.jsonl stores candidate peaks below the operational threshold (the
    storage floor, issue #27). The benchmark contract is operational detections:
    sub-floor candidates never reach a bundle, and a pano with only sub-floor
    peaks has no believed detections — it belongs in the 'empty' stratum."""
    results = tmp_path / "results.jsonl"
    mixed = _record("MIX", 0, coords=[
        {"x_normalized": 0.5, "y_normalized": 0.6, "confidence": 0.9},
        {"x_normalized": 0.2, "y_normalized": 0.6, "confidence": 0.2}], lat=45.0, lng=-122.0)
    weak = _record("WEAK", 0, coords=[
        {"x_normalized": 0.4, "y_normalized": 0.6, "confidence": 0.3}], lat=45.5, lng=-122.0)
    empty = _record("EMPTY", 0, lat=46.0, lng=-122.0)
    _write_results(results, [mixed, weak, empty])

    records_path = eb.write_bundle_records(results, tmp_path / "bundle", sample=5,
                                           empty_sample=5, seed=0, min_spacing=0)
    rows = {json.loads(l)["pano"]["panorama_id"]: json.loads(l)
            for l in records_path.read_text(encoding="utf-8").splitlines()}
    assert all(d["confidence"] >= eb.BENCHMARK_CONFIDENCE
               for r in rows.values() for d in r["detections"])
    assert rows["WEAK"]["benchmark_group"] == "empty"
    assert rows["MIX"]["benchmark_group"] in ("top", "random")
    assert len(rows["MIX"]["detections"]) == 1


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


# --- stratified sampling (--group-key; RampNet #159/#263) ------------------------------

def _rig_record(pid, n_dets, model, producer, lat, lng):
    rec = _record(pid, n_dets, lat=lat, lng=lng)
    rec["pano"].update(camera_model=model, copyright=producer)
    return rec


def _rig_city(seed=0):
    """A synthetic city: three rig x producer groups of very different size, panos on a
    ~110 m grid so 30 m spacing never binds, about a third of each group empty."""
    import random
    rng = random.Random(seed)
    recs, k = [], 0
    for model, producer, n in (("Ladybug", "grand lyon", 300), ("Max", "ecartip", 150),
                               ("THETA S", "luppano", 30)):
        for i in range(n):
            lat, lng = 45.0 + 0.001 * (k // 60), 4.8 + 0.0014 * (k % 60)
            recs.append(_rig_record(f"{producer[:3]}{i}", 0 if i % 3 == 0 else
                                    rng.randint(1, 4), model, producer, lat, lng))
            k += 1
    return recs


GROUP_MAP = [["Ladybug|grand lyon", "grand_lyon"], ["Max|*", "gopro_max"],
             ["*", "other_cameras"]]


def test_group_label_joins_keys_and_reads_missing_as_empty():
    rec = _rig_record("A", 1, None, "grand lyon", 45.0, 4.8)
    assert eb.group_label(rec, ["camera_model", "copyright"]) == "|grand lyon"
    assert eb.group_label(rec, ["copyright"]) == "grand lyon"


def test_group_map_first_match_wins_and_unmatched_keeps_raw_label(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps([["Max|ecartip", "ecartip"], ["Max|*", "other_max"]]),
                    encoding="utf-8")
    stratum_of = eb.make_stratum_of(["camera_model", "copyright"], eb.load_group_map(path))
    rec = lambda m, p: _rig_record("A", 1, m, p, 45.0, 4.8)
    assert stratum_of(rec("Max", "ecartip")) == "ecartip"
    assert stratum_of(rec("Max", "trucbidule")) == "other_max"
    assert stratum_of(rec("Ladybug", "grand lyon")) == "Ladybug|grand lyon"   # never pooled
    assert stratum_of(rec("max", "ecartip")) == "max|ecartip"   # case-sensitive everywhere


def test_load_group_map_rejects_a_dict(tmp_path):
    import pytest
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"Max|*": "gopro"}), encoding="utf-8")
    with pytest.raises(ValueError):
        eb.load_group_map(path)


def test_allocate_is_proportional_with_a_floor():
    assert eb.allocate(100, {"a": 900, "b": 90, "c": 10}, 20) == {"a": 70, "b": 20, "c": 10}
    # Without floors binding it is plain largest remainder, and sums to the total.
    alloc = eb.allocate(10, {"a": 50, "b": 30, "c": 20}, 0)
    assert alloc == {"a": 5, "b": 3, "c": 2}
    # Empty groups get nothing and do not appear.
    assert eb.allocate(10, {"a": 5, "b": 0}, 3) == {"a": 5}


def test_allocate_never_exceeds_a_group_and_reports_overshoot_by_floors():
    alloc = eb.allocate(30, {"a": 100, "b": 4}, 10)
    assert alloc == {"a": 26, "b": 4} and sum(alloc.values()) == 30
    # Floors alone exceed the total: every group keeps its floor (the caller reports it).
    assert eb.allocate(10, {"a": 100, "b": 100, "c": 100}, 5) == {"a": 5, "b": 5, "c": 5}


def test_allocate_sums_to_total_on_many_shapes():
    import random
    rng = random.Random(3)
    for _ in range(200):
        sizes = {f"g{i}": rng.randint(0, 400) for i in range(rng.randint(1, 7))}
        total, minimum = rng.randint(0, 300), rng.randint(0, 30)
        alloc = eb.allocate(total, sizes, minimum)
        assert all(0 <= n <= sizes[g] for g, n in alloc.items())
        assert all(n >= min(minimum, sizes[g]) for g, n in alloc.items())
        floors = sum(min(minimum, n) for n in sizes.values())
        if floors < total:
            assert sum(alloc.values()) == min(total, sum(sizes.values()))


def test_choose_panos_stratified_gives_every_group_its_floor():
    recs = _rig_city()
    report = {}
    stratum_of = eb.make_stratum_of(["camera_model", "copyright"], GROUP_MAP)
    chosen = eb.choose_panos(recs, sample=70, empty_sample=9, seed=1, top_n=4,
                             stratum_of=stratum_of, group_min=12, empty_group_min=3,
                             report=report)
    by = {}
    for rec, grp in chosen:
        by.setdefault((rec["benchmark_stratum"], grp), 0)
        by[(rec["benchmark_stratum"], grp)] += 1
    # The small group (20 detection panos) still gets its floor of 12 random panos.
    assert by[("other_cameras", "random")] == 12
    assert sum(n for (s, g), n in by.items() if g == "random") == 70 - 4
    assert sum(n for (s, g), n in by.items() if g == "top") == 4
    assert all(by[(s, "empty")] >= 3 for s in ("grand_lyon", "gopro_max", "other_cameras"))
    assert sum(n for (s, g), n in by.items() if g == "empty") == 9
    # The bigger group gets the bigger share of the rest.
    assert by[("grand_lyon", "random")] > by[("gopro_max", "random")]
    # The report agrees with what was drawn.
    for s, row in report["strata"].items():
        assert row["random"] == by.get((s, "random"), 0)
        assert row["empty"] == by.get((s, "empty"), 0)
        assert row["random"] == row["random_target"]
    # Every record is tagged and no pano appears twice.
    pids = [r["pano"]["panorama_id"] for r, _ in chosen]
    assert len(pids) == len(set(pids))
    assert all(r["benchmark_stratum"] == stratum_of(r) for r, _ in chosen)


def test_choose_panos_stratified_keeps_spacing_across_groups():
    # Two groups sharing one corner: only one pano there can be drawn, whatever its group.
    recs = [_rig_record(f"L{j}", 2, "Ladybug", "grand lyon", 45.0 + 0.00002 * j, 4.8)
            for j in range(5)]
    recs += [_rig_record(f"M{j}", 2, "Max", "ecartip", 45.00001 + 0.00002 * j, 4.8)
             for j in range(5)]
    recs += [_rig_record(f"F{k}", 1, "Max", "ecartip", 45.0 + 0.01 * (k + 1), 4.8)
             for k in range(5)]
    chosen = eb.choose_panos(recs, 20, 0, seed=0, min_spacing=30, top_n=0,
                             stratum_of=eb.make_stratum_of(["copyright"]), group_min=5)
    assert _pairwise_min_m(chosen) >= 30 - 1e-6
    corner = [r for r, _ in chosen if r["pano"]["panorama_id"][0] in "LM"]
    assert len(corner) == 1


def test_choose_panos_stratified_is_seed_reproducible():
    recs = _rig_city()
    stratum_of = eb.make_stratum_of(["camera_model", "copyright"], GROUP_MAP)
    pick = lambda seed: [r["pano"]["panorama_id"] for r, _ in eb.choose_panos(
        recs, 40, 6, seed, top_n=3, stratum_of=stratum_of, group_min=8)]
    assert pick(5) == pick(5)
    assert pick(5) != pick(6)


# The first 8 ids of choose_panos(_rig_city(), 30, 5, seed=2) as drawn by origin/main at
# 533eaf0, before the stratified option existed (the whole draw matched, not only these).
PRE_CHANGE_FIRST_8 = ['gra1', 'gra2', 'gra7', 'gra8', 'gra11', 'eca41', 'eca145', 'gra41']


def test_choose_panos_default_is_unchanged_by_the_stratified_option():
    # Default arguments must draw exactly what they drew before the option existed; the
    # unstratified call and an explicit stratum_of=None agree, and nothing is tagged.
    recs = _rig_city()
    a = eb.choose_panos(recs, 30, 5, seed=2)
    b = eb.choose_panos(recs, 30, 5, seed=2, stratum_of=None, top_n=eb.TOP_N_BY_COUNT)
    assert [(r["pano"]["panorama_id"], g) for r, g in a] == \
           [(r["pano"]["panorama_id"], g) for r, g in b]
    assert not any("benchmark_stratum" in r for r, _ in a)
    # Pinned against the pre-change draw (recomputed with the code before this option).
    assert [r["pano"]["panorama_id"] for r, _ in a][:8] == PRE_CHANGE_FIRST_8


def test_write_bundle_records_stratified_writes_strata_and_lower_tier(tmp_path):
    results = tmp_path / "results.jsonl"
    recs = _rig_city()
    # Give one Ladybug pano a 0.40 peak next to its 0.9 one, and one empty pano a lone
    # 0.40 peak: the strata are still decided at 0.55.
    recs[1]["detections"].append({"x_normalized": 0.7, "y_normalized": 0.6,
                                  "confidence": 0.40})
    recs[0]["detections"] = [{"x_normalized": 0.3, "y_normalized": 0.6, "confidence": 0.40}]
    _write_results(results, recs)
    gmap = tmp_path / "map.json"
    gmap.write_text(json.dumps(GROUP_MAP), encoding="utf-8")
    bundle = tmp_path / "bundle"
    records_path = eb.write_bundle_records(
        results, bundle, sample=60, empty_sample=9, seed=0, min_spacing=30, top_n=3,
        group_keys=["camera_model", "copyright"], group_map=eb.load_group_map(gmap),
        group_min=10, empty_group_min=3, record_confidence=0.30)
    rows = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines()]
    assert {r["benchmark_stratum"] for r in rows} == {"grand_lyon", "gopro_max",
                                                       "other_cameras"}
    assert all(d["confidence"] >= 0.30 for r in rows for d in r["detections"])
    for r in rows:      # strata at 0.55, whatever the lower tier shows
        n55 = sum(d["confidence"] >= eb.BENCHMARK_CONFIDENCE for d in r["detections"])
        assert (n55 == 0) == (r["benchmark_group"] == "empty")
    meta = json.loads((bundle / "sample.json").read_text(encoding="utf-8"))
    assert meta["record_confidence"] == 0.30 and meta["top_n"] == 3
    st = meta["stratified"]
    assert st["group_keys"] == ["camera_model", "copyright"] and st["group_map"] == GROUP_MAP
    assert sum(row["random"] for row in st["strata"].values()) == meta["groups"]["random"]


def test_write_bundle_records_default_sample_json_has_no_new_keys(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_results(results, _rig_city())
    eb.write_bundle_records(results, tmp_path / "b", 20, 5, seed=0, min_spacing=30)
    meta = json.loads((tmp_path / "b" / "sample.json").read_text(encoding="utf-8"))
    assert set(meta) == {"source_records", "source_record_count", "sample", "empty_sample",
                         "seed", "min_spacing_m", "min_confidence", "selected", "groups"}
    rows = (tmp_path / "b" / "records.jsonl").read_text(encoding="utf-8")
    assert "benchmark_stratum" not in rows


def test_write_bundle_records_refuses_a_record_tier_above_the_benchmark(tmp_path):
    import pytest
    results = tmp_path / "results.jsonl"
    _write_results(results, _rig_city())
    with pytest.raises(ValueError):
        eb.write_bundle_records(results, tmp_path / "b", 20, 5, seed=0, min_spacing=30,
                                record_confidence=0.9)


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
