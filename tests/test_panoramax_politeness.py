"""Issue #161: Panoramax pacing, batched metadata, the SD asset and its record frame, the
manifest binding, and the concurrency cap. No network: requests.get is monkeypatched, and
the limiter runs on a fake clock."""
import json
import threading
from types import SimpleNamespace

import pytest

import main
from sources import panoramax
from sources.ratelimit import RateLimiter
from test_sources_panoramax import make_item


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _limited(monkeypatch, clock, **rates):
    lims = {kind: RateLimiter(rates.get(kind), clock=clock, sleep=clock.sleep)
            for kind in panoramax.RATE_DEFAULTS}
    monkeypatch.setattr(panoramax, "_LIMITERS", lims)
    return lims


def _resp(status=200, payload=None, content=b"", headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {}, content=content,
                           json=lambda: payload,
                           raise_for_status=lambda: None if status < 400 else
                           (_ for _ in ()).throw(RuntimeError(f"HTTP {status}")))


# --- limiter --------------------------------------------------------------------------

def test_limiter_spaces_requests_at_the_rate():
    clock = FakeClock()
    lim = RateLimiter(1.0, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        lim.wait()
    assert clock.t == pytest.approx(3.0)       # 4 requests at 1/s: first free, then 1 s each
    assert lim.requests == 4


def test_limiter_burst_lets_the_first_requests_through():
    clock = FakeClock()
    lim = RateLimiter(1.0, burst=3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        lim.wait()
    assert clock.t == 0.0
    lim.wait()
    assert clock.t == pytest.approx(1.0)


def test_limiter_unlimited_never_sleeps():
    clock = FakeClock()
    lim = RateLimiter(None, clock=clock, sleep=clock.sleep)
    for _ in range(100):
        lim.wait()
    assert clock.sleeps == [] and lim.unlimited


def test_pause_holds_every_waiter_and_never_shortens():
    clock = FakeClock()
    lim = RateLimiter(None, clock=clock, sleep=clock.sleep)
    lim.pause(60)
    lim.pause(10)               # a shorter pause must not cut the longer one
    lim.wait()
    assert clock.t == pytest.approx(60)
    assert lim.pauses == 2


def test_limiter_is_thread_safe_under_a_real_clock():
    lim = RateLimiter(1000.0)
    threads = [threading.Thread(target=lambda: [lim.wait() for _ in range(25)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert lim.requests == 200


def test_rate_limits_env_overrides_and_zero_means_unlimited(monkeypatch):
    assert panoramax.rate_limits() == {"picture": 1.0, "metadata": 1.0, "tile": 5.0}
    monkeypatch.setenv("PANORAMAX_PICTURES_PER_SECOND", "0.5")
    monkeypatch.setenv("PANORAMAX_TILES_PER_SECOND", "0")
    assert panoramax.rate_limits() == {"picture": 0.5, "metadata": 1.0, "tile": None}
    assert panoramax.min_seconds_per_pano() == 2.0
    monkeypatch.setenv("PANORAMAX_METADATA_PER_SECOND", "fast")
    with pytest.raises(SystemExit):
        panoramax.rate_limits()


@pytest.mark.parametrize("headers, expected", [
    ({"Retry-After": "12"}, 12.0),
    ({"Retry-After": "9999"}, panoramax.MAX_PAUSE_S),
    ({}, panoramax.DEFAULT_PAUSE_S),
    ({"Retry-After": "soon"}, panoramax.DEFAULT_PAUSE_S),
    (None, panoramax.DEFAULT_PAUSE_S),
])
def test_retry_after_seconds(headers, expected):
    assert panoramax.retry_after_seconds(headers) == expected


def test_a_429_pauses_the_whole_kind_then_the_retry_goes_through(monkeypatch):
    clock = FakeClock()
    lims = _limited(monkeypatch, clock, picture=1.0)
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)   # the retry loop's backoff
    answers = [_resp(429, headers={"Retry-After": "40"}),
               _resp(200, content=b"x", headers={})]
    seen = []

    def fake_get(url, **kw):
        seen.append((clock.t, kw.get("headers", {}).get("User-Agent")))
        return answers.pop(0)
    monkeypatch.setattr(panoramax.requests, "get", fake_get)
    monkeypatch.setattr(panoramax.Image, "open", lambda b: SimpleNamespace(
        convert=lambda mode: SimpleNamespace(size=(4096, 2048))))
    (image, size), permanent, detail = panoramax._download_image("https://x/hd.jpg")
    assert size == (4096, 2048) and detail is None
    assert [t for t, _ in seen] == [0.0, pytest.approx(40.0)]       # the retry waited out the pause
    assert seen[0][1] == panoramax.USER_AGENT
    assert lims["picture"].pauses == 1 and lims["metadata"].pauses == 0
    # another thread's request of the same kind also waits: the pause is the kind's
    lims["picture"].pause(5)
    t0 = clock.t
    lims["picture"].wait()
    assert clock.t - t0 == pytest.approx(5.0)


def test_overload_exhaustion_is_a_retryable_failure(monkeypatch):
    clock = FakeClock()
    lims = _limited(monkeypatch, clock)               # unlimited rate, but pauses apply
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: _resp(503))
    assert panoramax.fetch_item("p1") == (None, False)                   # retryable, not gone
    assert panoramax._download_image("https://x/hd.jpg")[:2] == (None, False)
    assert panoramax.fetch_panos_for_tile(1, 2, None) is None             # failed tile
    assert lims["metadata"].pauses == lims["picture"].pauses == lims["tile"].pauses == 3
    assert clock.t >= 2 * 3 * panoramax.DEFAULT_PAUSE_S                    # each retry waited


def test_every_request_kind_goes_through_its_limiter(monkeypatch):
    calls = []
    monkeypatch.setattr(panoramax, "limiter", lambda kind: calls.append(kind) or RateLimiter(None))
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: _resp(
        200, payload={"stac_version": "1.0.0", "features": []}))
    panoramax.prepare()
    panoramax.fetch_item("p1")
    panoramax.fetch_items(["a", "b"])
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: _resp(204))
    panoramax.fetch_panos_for_tile(1, 2, None)
    assert calls == ["metadata", "metadata", "metadata", "tile"]


# --- batched metadata -----------------------------------------------------------------

def _search_server(monkeypatch, missing=(), fail_search=False):
    """A fake catalog: /search answers the requested ids (minus `missing`), /pictures/{id}
    answers any id. Returns the request log."""
    log = []

    def fake_get(url, params=None, **kw):
        if url.endswith("/search"):
            ids = params["ids"].split(",")
            log.append(("search", ids, params["limit"]))
            if fail_search:
                return _resp(502)
            return _resp(200, payload={"features": [dict(make_item(), id=i) for i in ids
                                                    if i not in missing]})
        pid = url.rsplit("/", 1)[1]
        log.append(("picture", pid))
        return _resp(200, payload=dict(make_item(), id=pid))
    monkeypatch.setattr(panoramax.requests, "get", fake_get)
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)
    return log


def test_without_a_hint_fetch_item_is_one_get_per_picture(monkeypatch):
    log = _search_server(monkeypatch)
    for pid in ("a", "b"):
        assert panoramax.fetch_item(pid)[0]["id"] == pid
    assert log == [("picture", "a"), ("picture", "b")]


def test_hinted_fetches_come_in_chunks_with_limit_equal_to_chunk(monkeypatch):
    monkeypatch.setattr(panoramax, "SEARCH_CHUNK", 3)
    log = _search_server(monkeypatch)
    ids = [f"p{i}" for i in range(7)]
    panoramax.queue_hint(ids)
    got = [panoramax.fetch_item(pid)[0]["id"] for pid in ids]
    assert got == ids
    assert log == [("search", ["p0", "p1", "p2"], 3), ("search", ["p3", "p4", "p5"], 3),
                   ("search", ["p6"], 1)]
    assert panoramax._HINT["cache"] == {}          # popped on use


def test_out_of_order_miss_fetches_the_id_plus_the_window_from_the_cursor(monkeypatch):
    monkeypatch.setattr(panoramax, "SEARCH_CHUNK", 3)
    log = _search_server(monkeypatch)
    panoramax.queue_hint(["a", "b", "c", "d", "e"])
    panoramax.fetch_item("d")                   # a thread that got ahead
    assert log[-1] == ("search", ["d", "a", "b"], 3)
    for pid in ("a", "b"):
        panoramax.fetch_item(pid)               # cached: no request
    panoramax.fetch_item("c")
    assert log[-1] == ("search", ["c", "e"], 2)  # d already requested: never asked twice
    assert len(log) == 2


def test_absent_from_search_falls_back_to_the_per_id_get(monkeypatch):
    log = _search_server(monkeypatch, missing={"b"})
    panoramax.queue_hint(["a", "b"])
    assert panoramax.fetch_item("a")[0]["id"] == "a"
    item, gone = panoramax.fetch_item("b")
    assert item["id"] == "b" and gone is False
    assert log == [("search", ["a", "b"], 2), ("picture", "b")]


def test_failed_search_falls_back_to_the_per_id_get(monkeypatch):
    log = _search_server(monkeypatch, fail_search=True)
    panoramax.queue_hint(["a", "b"])
    assert panoramax.fetch_item("a")[0]["id"] == "a"
    assert log[-1] == ("picture", "a")


def test_an_unhinted_id_uses_the_per_id_get(monkeypatch):
    log = _search_server(monkeypatch)
    panoramax.queue_hint(["a"])
    panoramax.fetch_item("zzz")
    assert log == [("picture", "zzz")]


def test_the_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(panoramax, "SEARCH_CHUNK", 4)
    monkeypatch.setattr(panoramax, "CACHE_CHUNKS", 1)
    log = _search_server(monkeypatch)
    ids = [f"p{i}" for i in range(12)]
    panoramax.queue_hint(ids)
    panoramax.fetch_item("p0")                      # caches p1..p3
    panoramax.fetch_item("p5")                      # caches p6, p7, p4 -> evicts beyond 4
    assert len(panoramax._HINT["cache"]) <= 4
    for pid in ids:                                 # every id is still served
        assert panoramax.fetch_item(pid)[0]["id"] == pid
    searched = [i for entry in log if entry[0] == "search" for i in entry[1]]
    assert len(searched) == len(set(searched))      # no id is ever searched twice


def test_main_hands_the_processing_order_to_queue_hint():
    src = open(main.__file__, encoding="utf-8").read()
    assert "getattr(source, 'queue_hint', None)" in src
    assert "hint(panos_to_process_ids)" in src


# --- SD asset and the record frame ----------------------------------------------------

def _patch_downloads(monkeypatch, answers):
    """answers: {url_substring: (image_tuple_or_None, permanent, detail)}"""
    seen = []

    def fake(url):
        seen.append(url)
        for key, value in answers.items():
            if key in url:
                return value
        raise AssertionError(url)
    monkeypatch.setattr(panoramax, "_download_image", fake)
    monkeypatch.setattr(panoramax, "fetch_item", lambda picture_id: (make_item(), False))
    return seen


def test_hd_run_writes_no_new_keys(monkeypatch):
    _patch_downloads(monkeypatch, {"b7f9.jpg": (("IMG", (5760, 2880)), False, None)})
    result = panoramax.fetch_pano("pid", 43.49, -1.47)
    meta = result["pano"]["source_metadata"]
    assert not {"image_asset", "image_url", "image_dimensions_downloaded"} & set(meta)
    assert (result["pano"]["width"], result["pano"]["height"]) == (5760, 2880)


def test_sd_run_records_the_sensor_frame_not_the_sd_size(monkeypatch):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    seen = _patch_downloads(monkeypatch, {"sd.jpg": (("SDIMG", (2048, 1024)), False, None)})
    result = panoramax.fetch_pano("pid", 43.49, -1.47)
    assert result["status"] == "success" and result["image"] == "SDIMG"
    pano = result["pano"]
    assert (pano["width"], pano["height"]) == (5760, 2880)      # sensor_array_dimensions
    meta = pano["source_metadata"]
    assert meta["image_asset"] == "sd" and meta["image_url"].endswith("sd.jpg")
    assert meta["image_dimensions_downloaded"] == [2048, 1024]
    assert meta["hd_url"].endswith("b7f9.jpg")                  # hd_url stays the original
    assert len(seen) == 1 and panoramax.run_record()["asset_fallbacks"] == 0


@pytest.mark.parametrize("override, sd_answer", [
    ({"assets.sd": None}, None),                                              # no SD asset
    ({"properties.pers:interior_orientation.sensor_array_dimensions": None}, None),  # no sensor dims
    ({}, (None, False, "Image asset HTTP 404")),                              # SD 404
    ({}, (None, True, "Undecodable image bytes (x)")),                        # SD undecodable
    ({}, (("SDIMG", (2048, 1536)), False, None)),                             # SD not 2:1
])
def test_sd_falls_back_to_hd_and_counts_it(monkeypatch, override, sd_answer):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    item = make_item(**override)
    answers = {"b7f9.jpg": (("HDIMG", (5760, 2880)), False, None)}
    if sd_answer is not None:
        answers["sd.jpg"] = sd_answer
    _patch_downloads(monkeypatch, answers)
    monkeypatch.setattr(panoramax, "fetch_item", lambda picture_id: (item, False))
    result = panoramax.fetch_pano("pid", 43.49, -1.47)
    assert result["status"] == "success" and result["image"] == "HDIMG"
    meta = result["pano"]["source_metadata"]
    assert meta["image_asset"] == "hd" and meta["image_dimensions_downloaded"] == [5760, 2880]
    assert (result["pano"]["width"], result["pano"]["height"]) == (5760, 2880)
    assert panoramax.run_record()["asset_fallbacks"] == 1


def test_sd_transient_failure_is_retryable_not_a_fallback(monkeypatch):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    _patch_downloads(monkeypatch, {"sd.jpg": (None, False, "ConnectionError: reset")})
    result = panoramax.fetch_pano("pid", 43.49, -1.47)
    assert result["status"] == "failure" and "reset" in result["reason"]
    assert panoramax.run_record()["asset_fallbacks"] == 0


def test_image_url_assets():
    item = make_item()
    assert panoramax.image_url(item).endswith("b7f9.jpg")
    assert panoramax.image_url(item, asset="sd").endswith("sd.jpg")
    assert panoramax.image_url(make_item(**{"assets.sd": None}), asset="sd") is None
    with pytest.raises(ValueError):
        panoramax.image_url(item, asset="thumb")


def test_run_record_reports_pacing_and_asset(monkeypatch):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    rec = panoramax.run_record()
    assert rec["panoramax_rate_limits"] == {"picture": 1.0, "metadata": 1.0, "tile": 5.0}
    assert rec["panoramax_asset"] == "sd" and set(rec["panoramax_requests"]) == set(
        panoramax.RATE_DEFAULTS)
    assert main.source_run_record(panoramax) == rec
    assert main.source_run_record(object()) == {}


# --- manifest binding -----------------------------------------------------------------

def _run_dir(tmp_path, records=False):
    run_dir = tmp_path / "lyon"
    run_dir.mkdir()
    if records:
        (run_dir / "results.jsonl").write_text('{"pano": {}}\n')
    return run_dir


def test_binding_new_run_records_the_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    manifest = {}
    assert main.bind_source_settings(manifest, panoramax, _run_dir(tmp_path)) is True
    assert manifest == {"panoramax_asset": "sd"}


def test_binding_legacy_run_with_records_is_hd(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path, records=True)
    manifest = {}
    assert main.bind_source_settings(manifest, panoramax, run_dir) is False   # not rewritten
    assert manifest == {}
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    with pytest.raises(SystemExit) as e:
        main.bind_source_settings({}, panoramax, run_dir)
    assert "panoramax_asset 'hd'" in str(e.value) and "new --name" in str(e.value)


def test_binding_refuses_a_resume_under_the_other_asset(tmp_path):
    with pytest.raises(SystemExit):
        main.bind_source_settings({"panoramax_asset": "sd"}, panoramax,
                                  _run_dir(tmp_path, records=True))
    assert main.bind_source_settings({"panoramax_asset": "hd"}, panoramax,
                                     tmp_path / "lyon") is False      # same asset resumes


def test_binding_is_a_noop_for_sources_without_the_hook(tmp_path):
    from sources import gsv
    manifest = {}
    assert main.bind_source_settings(manifest, gsv, _run_dir(tmp_path)) is False
    assert manifest == {}


# --- concurrency cap, CLI, export_benchmark, reinfer ----------------------------------

def _run_main(monkeypatch, argv):
    seen = {}

    def fake_run_labeler(*args, **kwargs):
        seen["concurrency"] = main.PROCESSING_CONCURRENCY
        seen["asset"] = panoramax.IMAGE_ASSET
    monkeypatch.setattr(main, "run_labeler", fake_run_labeler)
    monkeypatch.setattr(panoramax, "prepare", lambda: None)
    monkeypatch.setattr(main, "PROCESSING_CONCURRENCY", main.PROCESSING_CONCURRENCY)
    monkeypatch.setattr("sys.argv", ["main.py", "area.geojson", "--scan-only"] + argv)
    main.main()
    return seen


def test_concurrency_is_capped_for_panoramax(monkeypatch):
    seen = _run_main(monkeypatch, ["--source", "panoramax"])
    assert seen["concurrency"] == panoramax.MAX_PROCESSING_CONCURRENCY == 4
    seen = _run_main(monkeypatch, ["--source", "panoramax", "--processing-concurrency", "2"])
    assert seen["concurrency"] == 2


def test_panoramax_asset_flag_sets_the_module(monkeypatch):
    assert _run_main(monkeypatch, ["--source", "panoramax"])["asset"] == "hd"
    assert _run_main(monkeypatch, ["--source", "panoramax", "--panoramax-asset", "sd"])[
        "asset"] == "sd"


def test_fetch_native_downloads_hd_through_the_picture_limiter(monkeypatch, tmp_path):
    import export_benchmark
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")      # a run on SD must still archive HD
    monkeypatch.setattr(panoramax, "fetch_item", lambda pid: (make_item(), False))
    kinds, urls = [], []
    real_limiter = panoramax.limiter
    monkeypatch.setattr(panoramax, "limiter", lambda kind: kinds.append(kind) or real_limiter(kind))
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: urls.append(url) or _resp(
        200, content=b"JPEGBYTES"))
    out = tmp_path / "pid.jpg"
    assert export_benchmark.fetch_native("pid", "panoramax", out) is None
    assert out.read_bytes() == b"JPEGBYTES"
    assert urls == [make_item()["assets"]["hd"]["href"]] and kinds == ["picture"]


def test_reinfer_sets_the_asset_from_the_manifest():
    from conftest import REPO_ROOT
    src = (REPO_ROOT / "scripts" / "reinfer.py").read_text(encoding="utf-8")
    assert "source.IMAGE_ASSET = manifest.get(source.MANIFEST_ASSET_KEY, 'hd')" in src


# --- the asset study's matcher and rule (torch-free) ----------------------------------

import panoramax_asset_study as study  # noqa: E402


def test_study_matcher_wraps_the_seam_and_is_one_to_one():
    hd = [(0.999, 0.6, 0.9), (0.5, 0.6, 0.8)]
    sd = [(0.001, 0.6, 0.85), (0.505, 0.6, 0.7), (0.506, 0.6, 0.6)]
    m = study.match(hd, sd)
    assert m[0][0] == 0                     # across the seam: 0.002 apart
    assert m[1][0] == 1 and len(m) == 2     # one-to-one: the third peak is left over
    assert study.match([(0.5, 0.6, 1)], [(0.5 + 0.022, 0.6, 1)]) == {}   # strictly within


def test_study_offset_classes_follow_the_stride_8_grid():
    a = (3 / 1024, 0.6, 1)
    assert study.offset_class(a, (4 / 1024, 0.6, 1)) == "same_cell"
    assert study.offset_class(a, (11 / 1024, 0.6, 1)) == "adjacent_cell"
    assert study.offset_class(a, (3 / 1024, 0.6 + 4 / 512, 1)) == "other"


def test_study_compare_pano_strict_lenient_and_rig_mask():
    hd = [(0.2, 0.6, 0.8), (0.4, 0.6, 0.5), (0.6, 0.95, 0.9)]     # the last is on the rig
    sd = [(0.2, 0.6, 0.7), (0.4, 0.6, 0.2), (0.8, 0.6, 0.4)]
    c = study.compare_pano(hd, sd, 0.30)
    assert (c["hd_n"], c["arm_n"], c["matched"], c["matched_lenient"]) == (2, 2, 1, 2)
    assert c["deltas"] == [pytest.approx(-0.1)]


def test_study_rule():
    rigs = [{"rig": "A", "panos": 40, "recall_hi": 0.97, "precision_hi": 0.95},
            {"rig": "B", "panos": 10, "recall_hi": 0.50, "precision_hi": 0.50}]   # too few
    assert study.apply_rule({"recall": 0.96, "precision": 0.95}, rigs) == ("sd", [])
    default, reasons = study.apply_rule({"recall": 0.949, "precision": 0.99}, rigs)
    assert default == "hd" and "pooled recall" in reasons[0]
    rigs[0]["precision_hi"] = 0.89
    default, reasons = study.apply_rule({"recall": 0.99, "precision": 0.99}, rigs)
    assert default == "hd" and reasons == ["A: precision upper bound 0.890 < 0.9"]
    rigs[0]["precision_hi"] = float("nan")          # a rig with no detections never fails
    assert study.apply_rule({"recall": 0.99, "precision": 0.99}, rigs)[0] == "sd"


def test_study_gate_ignores_threshold_straddlers():
    stored = [(0.1, 0.6, 0.5), (0.2, 0.6, 0.302)]
    same, dmax = study.gate_pano(stored, [(0.1, 0.6, 0.5000001), (0.2, 0.6, 0.299)])
    assert same and dmax < 0.01
    assert not study.gate_pano(stored, [(0.2, 0.6, 0.302)])[0]


def test_study_allocation():
    alloc = study.allocate({"big": 1000, "mid": 100, "small": 10}, 100, 30)
    assert alloc["mid"] >= 30 and alloc["small"] == 0 and sum(alloc.values()) == 100
    assert study.allocate({"a": 3, "b": 2}, 10, 0) == {"a": 3, "b": 2}   # capped at pools


def test_study_bootstrap_is_seeded_and_brackets_the_estimate():
    rows = [{"m": i % 3, "n": 3} for i in range(30)]
    lo, hi = study.bootstrap_ci(rows, "m", "n")
    assert lo <= 1 / 3 <= hi and (lo, hi) == study.bootstrap_ci(rows, "m", "n")


# --- review of PR #173 ----------------------------------------------------------------

def test_prepare_says_overloaded_not_misconfigured_on_429(monkeypatch):
    clock = FakeClock()
    lims = _limited(monkeypatch, clock)
    monkeypatch.setattr(panoramax.time, "sleep", lambda s: None)
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: _resp(
        429, headers={"Retry-After": "7"}))
    with pytest.raises(SystemExit) as e:
        panoramax.prepare()
    assert "overloaded" in str(e.value) and "PANORAMAX_API_URL" not in str(e.value)
    assert lims["metadata"].pauses == panoramax.ATTEMPTS        # every attempt waited
    answers = [_resp(503), _resp(200, payload={"stac_version": "1.0.0"})]
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: answers.pop(0))
    panoramax.prepare()                                          # a retry gets through


def test_a_failed_search_lets_a_later_miss_rebatch_the_followers(monkeypatch):
    monkeypatch.setattr(panoramax, "SEARCH_CHUNK", 3)
    log = _search_server(monkeypatch, fail_search=True)
    panoramax.queue_hint(["a", "b", "c"])
    panoramax.fetch_item("a")                                   # search fails -> per-id
    monkeypatch.setattr(panoramax, "fetch_items", lambda ids: log.append(("batch", ids)) or {
        i: dict(make_item(), id=i) for i in ids})
    assert panoramax.fetch_item("b")[0]["id"] == "b"
    assert log[-1] == ("batch", ["b", "c"])                     # b, c were un-marked


def test_refusal_names_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    with pytest.raises(SystemExit) as e:
        main.bind_source_settings({"panoramax_asset": "hd"}, panoramax, _run_dir(tmp_path))
    assert "pass --panoramax-asset hd for this run" in str(e.value)


def test_fallback_counts_once_per_pano_written(monkeypatch):
    monkeypatch.setattr(panoramax, "IMAGE_ASSET", "sd")
    _patch_downloads(monkeypatch, {"sd.jpg": (None, False, "Image asset HTTP 404"),
                                   "b7f9.jpg": (None, False, "reset")})
    assert panoramax.fetch_pano("pid", 43.49, -1.47)["status"] == "failure"
    assert panoramax.run_record()["asset_fallbacks"] == 0       # HD failed: nothing written


def test_bayonne_figures_download_through_the_limiter(monkeypatch):
    import io
    import panoramax_bayonne_figures as bf
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 2)).save(buf, "JPEG")
    kinds = []
    real = panoramax.limiter
    monkeypatch.setattr(panoramax, "limiter", lambda kind: kinds.append(kind) or real(kind))
    monkeypatch.setattr(panoramax.requests, "get", lambda url, **kw: _resp(
        200, content=buf.getvalue()))
    assert bf._fetch_image("https://x/hd.jpg").size == (4, 2)
    assert kinds == ["picture"]


def test_contact_sheet_window_keeps_the_peak_centred():
    from PIL import Image
    import panoramax_asset_figures as figs
    img = Image.new("RGB", (100, 50), (255, 0, 0))
    img.paste((0, 0, 255), (0, 0, 10, 50))                     # blue strip at the left edge
    win = figs._window(img, 98, 2, 6)                           # straddles seam and top
    assert win.size == (12, 12)
    assert win.getpixel((11, 6)) == (0, 0, 255)                 # wrapped from x = 4
    assert win.getpixel((2, 6)) == (255, 0, 0)
    assert win.getpixel((6, 0)) == (128, 128, 128)              # padded above the top
