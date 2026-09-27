"""detect_from_store: a results.jsonl from a local pano store + the server's metadata (#56).

No network (requests.get is stubbed per test), no GPU, no torch: the detector is a stub
and the store holds tiny JPEGs.
"""
import json

import pytest
from PIL import Image

import detect_from_store as dfs
import main
from conftest import make_process_result, make_provenance
from sources import TARGET_IMAGE_SIZE

SERVER = "https://ps.example"


def _meta(pid, **over):
    m = {"panoId": pid, "width": 64, "height": 32, "tileWidth": 512, "tileHeight": 512,
         "lat": 45.6, "lng": -122.5, "cameraHeading": 12.5, "cameraPitch": 1.0,
         "cameraRoll": None, "captureDate": "2023-05", "copyright": "© 2025 Google"}
    m.update(over)
    return m


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(dfs, "REQUEST_SPACING_S", 0.0)
    monkeypatch.setattr(dfs.time, "sleep", lambda s: None)


def serve(monkeypatch, answers):
    """Stub requests.get: `answers` maps pano id -> list of _Resp (consumed in order) or a
    single _Resp. Returns the list of pano ids requested."""
    calls = []

    def get(url, timeout=None, allow_redirects=True):
        assert allow_redirects is False
        pid = url.split("/backupImage/")[1].split("/")[0]
        calls.append(pid)
        a = answers[pid]
        return a.pop(0) if isinstance(a, list) else a
    monkeypatch.setattr(dfs.requests, "get", get)
    return calls


def make_store(root, ids, sharded=True, size=(64, 32)):
    for pid in ids:
        d = root / pid[:2] if sharded else root
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (90, 90, 90)).save(d / f"{pid}.jpg")
    return root


class StubDetector:
    provenance = make_provenance()

    def __init__(self):
        self.seen = []

    def detect(self, image):
        self.seen.append(image.size)
        return [(0.5, 0.25, 0.9), (0.125, 0.75, 0.2)]


@pytest.fixture
def detector(monkeypatch):
    stub = StubDetector()
    monkeypatch.setattr(main, "curb_ramp_detector", stub, raising=False)
    return stub


# ------------------------------------------------------------------------ store layouts

def test_layout_autodetect_and_paths(tmp_path):
    flat = make_store(tmp_path / "flat", ["AAx", "BBy"], sharded=False)
    shard = make_store(tmp_path / "shard", ["AAx", "BBy"])
    assert dfs.store_layout(flat, ["AAx", "ZZmissing"]) == "flat"
    assert dfs.store_layout(shard, ["AAx", "BBy"]) == "sharded"
    assert dfs.jpg_path(shard, "sharded", "AAx") == shard / "AA" / "AAx.jpg"
    assert dfs.jpg_path(flat, "flat", "AAx") == flat / "AAx.jpg"
    assert dfs.store_ids(flat, "flat") == dfs.store_ids(shard, "sharded") == ["AAx", "BBy"]


def test_store_ids_ignore_downscaled_sidecars(tmp_path):
    store = make_store(tmp_path, ["AAx"])
    Image.new("RGB", (8, 4)).save(store / "AA" / "AAx.w8192.jpg")
    (store / "AA" / "AAx.depth.npz").write_bytes(b"")
    assert dfs.store_ids(store, "sharded") == ["AAx"]


# ------------------------------------------------------------------------- id selection

def test_label_ids_read_pano_id_and_filter_user(tmp_path):
    feats = [{"properties": {"pano_id": "P1", "user_id": "ai"}},
             {"properties": {"pano_id": "P2", "user_id": "human"}},
             {"properties": {"gsv_panorama_id": "P3", "user_id": "ai"}},
             {"properties": {"pano_id": "P1", "user_id": "ai"}}]
    path = tmp_path / "raw.geojson"
    path.write_text(json.dumps({"features": feats}), encoding="utf-8")
    assert dfs.label_pano_ids(path) == ["P1", "P2", "P3"]
    assert dfs.label_pano_ids(path, user="ai") == ["P1", "P3"]


def test_sample_unlabeled_is_deterministic_and_excludes_the_list():
    pool = [f"id{i:03d}" for i in range(200)]
    a = dfs.sample_unlabeled(pool, pool[:50], 20, seed=56)
    assert a == dfs.sample_unlabeled(list(reversed(pool)), pool[:50], 20, seed=56)
    assert a != dfs.sample_unlabeled(pool, pool[:50], 20, seed=57)
    assert not set(a) & set(pool[:50]) and len(set(a)) == 20
    with pytest.raises(SystemExit):
        dfs.sample_unlabeled(pool, pool[:50], 151, seed=1)


def test_selection_is_frozen_on_resume(tmp_path):
    store = make_store(tmp_path / "store", ["AA1", "AA2", "BB1", "BB2", "CC1"])
    run_dir = tmp_path / "run"
    ids, sel = dfs.select_ids(run_dir, store, "sharded", ["AA1"], 2, seed=3, server=SERVER)
    assert ids[0] == "AA1" and len(ids) == 3 and sel["n_sampled_unlabeled"] == 2
    # The store grows; the resume must still process the original list.
    make_store(store, ["DD1", "DD2", "DD3"])
    again, _ = dfs.select_ids(run_dir, store, "sharded", ["AA1"], 2, seed=3, server=SERVER)
    assert again == ids
    with pytest.raises(SystemExit, match="seed"):
        dfs.select_ids(run_dir, store, "sharded", ["AA1"], 2, seed=4, server=SERVER)
    with pytest.raises(SystemExit, match="input_ids_sha256"):
        dfs.select_ids(run_dir, store, "sharded", ["AA1", "BB1"], 2, seed=3, server=SERVER)


# ---------------------------------------------------------------------------- metadata

def test_metadata_is_cached_and_never_refetched(tmp_path, monkeypatch):
    calls = serve(monkeypatch, {"P": _Resp(200, _meta("P"))})
    assert dfs.fetch_metadata(SERVER, "P", tmp_path) == ("ok", _meta("P"))
    assert dfs.fetch_metadata(SERVER, "P", tmp_path) == ("ok", _meta("P"))
    assert calls == ["P"]


def test_metadata_404_is_a_skip_5xx_and_junk_are_failures(tmp_path, monkeypatch):
    serve(monkeypatch, {"GONE": _Resp(404),
                        "DOWN": _Resp(503),
                        "HTML": _Resp(200, ValueError("not json")),
                        "MOVED": _Resp(302)})
    assert dfs.fetch_metadata(SERVER, "GONE", tmp_path) == ("skipped", dfs.SKIP_METADATA_UNSERVED)
    for pid in ("DOWN", "HTML", "MOVED"):
        status, _ = dfs.fetch_metadata(SERVER, pid, tmp_path)
        assert status == "failure"
    assert not list(tmp_path.glob("*.json"))


def test_429_honours_retry_after_then_succeeds(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(dfs.time, "sleep", slept.append)
    serve(monkeypatch, {"P": [_Resp(429, headers={"Retry-After": "7"}), _Resp(200, _meta("P"))]})
    assert dfs.fetch_metadata(SERVER, "P", tmp_path)[0] == "ok"
    assert 7.0 in slept


def test_pano_block_has_the_gsv_record_keys():
    gsv_block = make_process_result()["pano"]
    block = dfs.pano_record_from_ps("P", _meta("P", width=16384, height=8192))
    assert list(block) == list(gsv_block)
    assert block["source_detail"] == "ps_store"
    assert block["camera_height_status"] == "no_depth" and block["camera_height_m"] is None
    assert block["links"] == [] and block["history"] == [] and block["source"] is None
    assert (block["width"], block["height"], block["capture_date"]) == (16384, 8192, "2023-05")
    with pytest.raises(KeyError):
        dfs.pano_record_from_ps("P", _meta("P", lat=None))


def test_normalize_image_is_the_detector_input():
    import panorama
    assert panorama.normalize_image(Image.new("RGB", (300, 100))).size == TARGET_IMAGE_SIZE


# --------------------------------------------------------------------------- the run

def _args(run_dir, store, *extra):
    return ["--run-dir", str(run_dir), "--store", str(store), "--server", SERVER, *extra]


def test_run_writes_records_skips_and_resumes(tmp_path, monkeypatch, detector):
    store = make_store(tmp_path / "store", ["AA1", "BB1", "CC1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\nBB1\nCC1\nNOJPEG\nGONE1\n", encoding="utf-8")
    make_store(store, ["GONE1"])
    calls = serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1")), "BB1": _Resp(200, _meta("BB1")),
                                "CC1": _Resp(503), "GONE1": _Resp(404)})
    run_dir = tmp_path / "run"

    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file), "--workers", "2"))

    lines = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [r["pano"]["panorama_id"] for r in lines] == ["AA1", "BB1"]
    expected = main.build_output_line(
        {"pano": dfs.pano_record_from_ps("AA1", _meta("AA1")),
         "detections": [(0.5, 0.25, 0.9), (0.125, 0.75, 0.2)]}, detector.provenance)
    assert lines[0] == expected
    assert detector.seen == [TARGET_IMAGE_SIZE] * 2
    assert "NOJPEG" not in calls                                   # no GET without pixels
    skips = dfs.load_skips(run_dir)
    assert skips == {"NOJPEG": dfs.SKIP_NO_JPEG, "GONE1": dfs.SKIP_METADATA_UNSERVED}
    done = main.load_processed_ids(run_dir / "already_processed.txt")
    assert done == {"AA1", "BB1", "NOJPEG", "GONE1"}                # CC1 failed: retried
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["imagery_source"] == "gsv"
    assert manifest["model_revision"] == detector.provenance["model_revision"]
    assert manifest["pixels"]["store"] == str(store.resolve()) and manifest["pixels"]["n_ids"] == 5
    assert manifest["runs"][-1]["phase"] == "store"
    assert manifest["runs"][-1]["processed"] == 2 and manifest["runs"][-1]["failed"] == 1

    # Resume: only the failed pano is retried, and its metadata is fetched once.
    calls.clear()
    serve_again = serve(monkeypatch, {"CC1": _Resp(200, _meta("CC1"))})
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file)))
    assert serve_again == ["CC1"]
    ids = [json.loads(l)["pano"]["panorama_id"]
           for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert ids == ["AA1", "BB1", "CC1"]


def test_resume_refuses_another_store(tmp_path, monkeypatch, detector):
    store = make_store(tmp_path / "store", ["AA1"])
    other = make_store(tmp_path / "other", ["AA1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\n", encoding="utf-8")
    serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1"))})
    dfs.main_cli(_args(tmp_path / "run", store, "--ids", str(ids_file)))
    with pytest.raises(SystemExit, match="store"):
        dfs.main_cli(_args(tmp_path / "run", other, "--ids", str(ids_file)))


def test_unreadable_jpg_is_a_cached_skip(tmp_path, monkeypatch, detector):
    store = make_store(tmp_path / "store", ["AA1"])
    (store / "AA" / "AA1.jpg").write_bytes(b"not a jpeg at all")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\n", encoding="utf-8")
    serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1"))})
    dfs.main_cli(_args(tmp_path / "run", store, "--ids", str(ids_file)))
    assert dfs.load_skips(tmp_path / "run") == {"AA1": dfs.SKIP_NOT_AN_IMAGE}


def test_metadata_only_loads_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "curb_ramp_detector", None, raising=False)
    store = make_store(tmp_path / "store", ["AA1", "GONE1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\nGONE1\nNOJPEG\n", encoding="utf-8")
    serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1")), "GONE1": _Resp(404)})
    run_dir = tmp_path / "run"
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file), "--metadata-only"))
    assert (run_dir / "store_metadata" / "AA1.json").exists()
    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "results.jsonl").exists()
    assert main.load_processed_ids(run_dir / "already_processed.txt") == {"GONE1", "NOJPEG"}


def test_existing_scan_manifest_is_adopted_and_bound(tmp_path, monkeypatch, detector):
    """runs/vancouver has a scan-only manifest (area hash, pre-#39 model keys, no runs)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": "run", "area_hash": "abc", "imagery_source": "gsv",
        "model_id": "rampnet-model", "model_training_date": detector.provenance["model_training_date"],
        "api_version": "1.0.0", "detection_storage_floor": main.DETECTION_STORAGE_FLOOR,
        "runs": []}), encoding="utf-8")
    store = make_store(tmp_path / "store", ["AA1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\n", encoding="utf-8")
    serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1"))})
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file)))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["area_hash"] == "abc"
    assert manifest["model_revision"] == detector.provenance["model_revision"]
    assert "legacy_model_provenance" in manifest and "pixels" in manifest


# ------------------------------------------------------------ review fixes (PR #96)

def test_a_stray_root_jpeg_does_not_flip_a_sharded_store(tmp_path):
    store = make_store(tmp_path / "store", ["AA1", "BB1"])
    Image.new("RGB", (8, 4)).save(store / "stray.jpg")
    assert dfs.store_layout(store, ["AA1", "BB1"]) == "sharded"
    with pytest.raises(SystemExit, match="none of the first"):
        dfs.store_layout(tmp_path / "store" / "AA", ["ZZ1", "YY1"])     # wrong path


def test_store_ids_only_read_two_character_shards_with_their_own_ids(tmp_path):
    store = make_store(tmp_path / "store", ["AA1", "BB1"])
    (store / "thumbs").mkdir()
    Image.new("RGB", (8, 4)).save(store / "thumbs" / "CC1.jpg")
    Image.new("RGB", (8, 4)).save(store / "AA" / "ZZ9.jpg")             # misfiled
    assert dfs.store_ids(store, "sharded") == ["AA1", "BB1"]


def test_poisoned_skip_rate_is_not_cached(tmp_path, monkeypatch, detector):
    """An empty or unmounted store makes every pano jpg_missing: that must not become a
    permanent skip for the whole run."""
    ids = [f"AA{i:02d}" for i in range(30)]
    store = make_store(tmp_path / "store", ids[:1])                     # only one JPEG
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    serve(monkeypatch, {"AA00": _Resp(200, _meta("AA00"))})
    run_dir = tmp_path / "run"
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file)))
    assert main.load_processed_ids(run_dir / "already_processed.txt") == {"AA00"}
    assert dfs.load_skips(run_dir) == {}
    entry = json.loads((run_dir / "manifest.json").read_text())["runs"][-1]
    assert entry["guarded_skips_refused"] == {dfs.SKIP_NO_JPEG: 29} and entry["failed"] == 29
    assert entry["native_size_mismatch"] == 0
    # ...and the operator can accept the rate after checking by hand.
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file), "--accept-skip-rate"))
    assert len(main.load_processed_ids(run_dir / "already_processed.txt")) == 30


def test_poisoned_404_rate_is_not_cached_by_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "curb_ramp_detector", None, raising=False)
    ids = [f"AA{i:02d}" for i in range(25)]
    store = make_store(tmp_path / "store", ids)
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    serve(monkeypatch, {pid: _Resp(404) for pid in ids})               # a wrong --server
    run_dir = tmp_path / "run"
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file), "--metadata-only"))
    assert main.load_processed_ids(run_dir / "already_processed.txt") == set()


def test_server_is_bound_on_every_call(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "curb_ramp_detector", None, raising=False)
    store = make_store(tmp_path / "store", ["AA1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\n", encoding="utf-8")
    serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1"))})
    run_dir = tmp_path / "run"
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file), "--metadata-only"))
    sel = json.loads((run_dir / dfs.SELECTION_FILE).read_text())
    assert sel["server"] == SERVER and sel["layout"] == "sharded"
    # A trailing slash is the same server; another host is not.
    dfs.main_cli(["--run-dir", str(run_dir), "--store", str(store), "--server", SERVER + "/",
                  "--ids", str(ids_file), "--metadata-only"])
    with pytest.raises(SystemExit, match="server"):
        dfs.main_cli(["--run-dir", str(run_dir), "--store", str(store), "--server",
                      "https://other.example", "--ids", str(ids_file), "--metadata-only"])


def test_store_path_is_compared_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "curb_ramp_detector", None, raising=False)
    store = make_store(tmp_path / "store", ["AA1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\n", encoding="utf-8")
    serve(monkeypatch, {"AA1": _Resp(200, _meta("AA1"))})
    run_dir = tmp_path / "run"
    dfs.main_cli(_args(run_dir, store, "--ids", str(ids_file), "--metadata-only"))
    dfs.main_cli(_args(run_dir, str(store) + "/", "--ids", str(ids_file), "--metadata-only"))
    dfs.main_cli(_args(run_dir, store / "AA" / "..", "--ids", str(ids_file), "--metadata-only"))


def test_sampled_ids_are_written_beside_the_selection(tmp_path):
    store = make_store(tmp_path / "store", ["AA1", "AA2", "BB1", "BB2"])
    run_dir = tmp_path / "run"
    ids, sel = dfs.select_ids(run_dir, store, "sharded", ["AA1"], 2, seed=3, server=SERVER)
    sampled = dfs.read_id_file(run_dir / dfs.SAMPLED_IDS_FILE)
    assert sampled == ids[1:] and len(sampled) == 2
    assert sel["sampled_ids_sha256"] == dfs._sha256_lines(sampled)


def test_bind_manifest_refuses_a_run_main_py_wrote(tmp_path, detector):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": "run", "area_hash": "abc", "imagery_source": "gsv",
        "detection_storage_floor": main.DETECTION_STORAGE_FLOOR, "runs": []}), encoding="utf-8")
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    pixels = {"store": "/s", "layout": "sharded", "server": SERVER}
    with pytest.raises(SystemExit, match="main.py"):
        dfs.bind_manifest(run_dir, detector.provenance, pixels)


def test_bind_manifest_refuses_other_pixels(tmp_path, detector):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pixels = {"store": "/s", "layout": "sharded", "server": SERVER}
    dfs.bind_manifest(run_dir, detector.provenance, pixels)
    with pytest.raises(SystemExit, match="server"):
        dfs.bind_manifest(run_dir, detector.provenance, {**pixels, "server": "https://b.example"})


def test_main_py_refuses_a_store_built_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": "run", "area_hash": "abc", "imagery_source": "gsv",
        "detection_storage_floor": main.DETECTION_STORAGE_FLOOR,
        "pixels": {"store": "/s"}, "runs": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main.load_or_init_run_dir(run_dir, tmp_path / "a.geojson", {}, "abc", "gsv")


def test_a_store_record_goes_through_transform_record():
    """send_to_ps's own transform accepts a store record (the refusal to SEND one is a
    separate, file-level guard: tests/test_send_to_ps_store.py)."""
    import send_to_ps
    record = main.build_output_line(
        {"pano": dfs.pano_record_from_ps("P", _meta("P", width=16384, height=8192)),
         "detections": [(0.5, 0.25, 0.9)]}, make_provenance())
    out = send_to_ps.transform_record(record)
    assert out["pano"]["source"] == "gsv" and out["pano"]["pano_id"] == "P"
    assert out["pano"]["source_metadata"]["source_detail"] == "ps_store"
    assert (out["labels"][0]["pano_x"], out["labels"][0]["pano_y"]) == (8192, 2048)
