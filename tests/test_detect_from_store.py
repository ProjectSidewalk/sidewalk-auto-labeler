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


# --------------------------------------------------------------- records mode (#168 pilot)

from detectors import checkpoints as ck  # noqa: E402

XW_PROV = ck.provenance_for_checkpoint(
    'ebad379b5f7c1645dc39ccb820cd18500109723ae5b1e5706c1e4521b2ab42b4')


class StubCrosswalk:
    """CrosswalkDetector's surface (frame fixed by the spec), without torch."""
    decode, border, storage_floor = 'argmax', 'keep', 0.05
    provenance = XW_PROV

    def detect(self, image):
        return [(0.5, 0.6, 0.9)]


@pytest.fixture
def crosswalk(monkeypatch):
    stub = StubCrosswalk()
    monkeypatch.setattr(main, "detectors", {"Crosswalk": stub})
    monkeypatch.setattr(main, "curb_ramp_detector", None, raising=False)
    return stub


def _xw(tmp_path):
    """--types Crosswalk plus a checkpoint path the CLI can find (the stub ignores it)."""
    ckpt = tmp_path / "s4.pth"
    ckpt.write_bytes(b"")
    return ["--types", "Crosswalk", "--crosswalk-checkpoint", str(ckpt)]


def _mapillary_block(pid, lat):
    return {"panorama_id": pid, "width": 64, "height": 32, "lat": lat, "lng": -134.4,
            "camera_heading": 90.0, "source": "mapillary", "source_detail": "mapillary",
            "copyright": "a contributor", "links": [], "history": []}


TEST_EP = "https://sidewalk-juneau-test.cs.washington.edu/ai/submitLabelsOnPano"
PROD_EP = "https://sidewalk-juneau.cs.washington.edu/ai/submitLabelsOnPano"


def write_record(path, endpoints=(TEST_EP,), sha256=None, sent=None):
    """The send_to_ps.py submission record beside `path`: every line sent to each endpoint
    (or `sent` lines), hashed as the file now stands unless `sha256` says otherwise."""
    n = len(path.read_text(encoding="utf-8").splitlines())
    rec = {"input_file": path.name, "sha256": sha256 or dfs.file_sha256(path), "total_lines": n,
           "endpoints": {ep: {"submitted_lines": n if sent is None else sent,
                              "min_confidence": 0.3} for ep in endpoints}}
    (path.parent / f"{path.name}.submission.json").write_text(json.dumps(rec), encoding="utf-8")


def make_source_run(root, ids, name="results.check.jsonl", record=True):
    """A finished Mapillary run dir: manifest, area and a records file PS already received
    (with its submission record, unless `record` is False)."""
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(
        {"imagery_source": "mapillary", "mapillary_position": "sfm"}), encoding="utf-8")
    (root / "area.geojson").write_text('{"type": "Polygon", "coordinates": []}', encoding="utf-8")
    path = root / name
    path.write_text("".join(json.dumps({"detections": [], "label_type": "CurbRamp",
                                        "pano": _mapillary_block(pid, 58.3 + i / 1000)}) + "\n"
                            for i, pid in enumerate(ids)), encoding="utf-8")
    if record:
        write_record(path)
    return path


def _no_network(monkeypatch):
    def get(*a, **k):
        raise AssertionError("records mode must not ask a server")
    monkeypatch.setattr(dfs.requests, "get", get)


def test_records_mode_keeps_blocks_verbatim_and_inherits_the_source(tmp_path, monkeypatch,
                                                                     crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a", "22b"])
    records = make_source_run(tmp_path / "juneau", ["11a", "22b", "33c"])     # 33c: no JPEG
    run_dir = tmp_path / "juneau_crosswalk"

    dfs.main_cli(["--run-dir", str(run_dir), "--store", str(store), "--records", str(records),
                  "--accept-skip-rate", *_xw(tmp_path)])

    lines = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    source = [json.loads(l)["pano"] for l in records.read_text().splitlines()]
    assert [l["pano"] for l in lines] == source[:2]                # verbatim, never rebuilt
    assert all(l["label_types"] == ["Crosswalk"] for l in lines)
    assert lines[0]["detections"] == [{"x_normalized": 0.5, "y_normalized": 0.6,
                                       "confidence": 0.9, "label_type": "Crosswalk"}]
    assert dfs.load_skips(run_dir) == {"33c": dfs.SKIP_NO_JPEG}
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["imagery_source"] == "mapillary" and m["mapillary_position"] == "sfm"
    assert m["label_types"] == ["Crosswalk"] and m["detection_storage_floor"] is None
    assert m["models"]["Crosswalk"]["detection_border"] == "keep"
    assert m["pixels"]["pano_blocks"] == "records"
    assert m["pixels"]["records"]["sha256"] == dfs.file_sha256(records)
    assert m["pixels"]["records_received_by"] == [TEST_EP]
    assert (run_dir / "area.geojson").read_bytes() == (records.parent / "area.geojson").read_bytes()

    # send_to_ps: store-built (the `pixels` block), but Crosswalk only, so it passes (#179).
    import send_to_ps
    assert "pixels" in send_to_ps.check_store_built(run_dir / "results.jsonl")


def test_records_mode_refuses_an_edited_records_file_on_resume(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    argv = ["--run-dir", str(tmp_path / "run"), "--store", str(store), "--records", str(records),
            *_xw(tmp_path)]
    dfs.main_cli(argv)
    records.write_text(records.read_text().replace("58.3", "58.4"), encoding="utf-8")
    write_record(records)      # as if the edited file had been sent too: only the binding refuses
    with pytest.raises(SystemExit, match="different pano selection"):
        dfs.main_cli(argv)


def test_records_line_without_a_pano_block_names_the_line(tmp_path):
    records = tmp_path / "results.jsonl"
    records.write_text('{"pano": {"panorama_id": "11a"}}\n{"detections": []}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="line 2"):
        dfs.read_records(records)


def test_records_mode_needs_the_source_run_dir(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    (records.parent / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="no manifest.json beside it"):
        dfs.main_cli(_records_argv(tmp_path, store, records))


def test_records_mode_refuses_an_area_that_differs_on_resume(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    argv = _records_argv(tmp_path, store, records)
    dfs.main_cli(argv)
    (tmp_path / "run" / "area.geojson").write_text('{"type": "Polygon", "coordinates": [[]]}',
                                                   encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from"):
        dfs.main_cli(argv)


def test_records_mode_refuses_another_imagery_source_on_resume(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    argv = _records_argv(tmp_path, store, records)
    dfs.main_cli(argv)
    (records.parent / "manifest.json").write_text(json.dumps({"imagery_source": "panoramax"}),
                                                  encoding="utf-8")
    with pytest.raises(SystemExit, match="'mapillary' run"):
        dfs.main_cli(argv)


def test_records_file_is_bound_by_resolved_path_too(tmp_path, monkeypatch, crosswalk):
    # The same bytes under the same name in another run dir are another records file.
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    dfs.main_cli(_records_argv(tmp_path, store, records))
    copy = make_source_run(tmp_path / "copy", ["11a"])
    assert copy.read_bytes() == records.read_bytes() and copy.name == records.name
    with pytest.raises(SystemExit, match="records: recorded"):
        dfs.main_cli(_records_argv(tmp_path, store, copy))


def test_run_checks_each_loaded_detectors_frame(tmp_path, monkeypatch):
    # The stubbed tests preset main.detectors, so run() returns before check_fixed_frames;
    # here run() loads the (stub) detector itself, with a frame its spec does not have.
    from detectors import registry
    _no_network(monkeypatch)

    class WrongFrame(StubCrosswalk):
        border = "exclude"

        def close(self):
            pass
    monkeypatch.setattr(main, "detectors", {})
    monkeypatch.setattr(main, "curb_ramp_detector", None, raising=False)
    monkeypatch.setattr(registry, "load_detectors", lambda types, **k: {"Crosswalk": WrongFrame()})
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    with pytest.raises(SystemExit, match="spec says"):
        dfs.main_cli(_records_argv(tmp_path, store, records))
    assert not (tmp_path / "run" / "results.jsonl").exists()
    assert main.detectors == {}


def _received(tmp_path, record_text):
    records = tmp_path / "results.jsonl"
    records.write_text('{"pano": {"panorama_id": "11a"}}\n', encoding="utf-8")
    (tmp_path / "results.jsonl.submission.json").write_text(
        record_text.replace("SHA", dfs.file_sha256(records)), encoding="utf-8")
    return dfs.records_received(records, dfs.file_sha256(records))


def test_records_received_reads_a_multi_type_record(tmp_path):
    # A multi-type record counts its lines per type (endpoints[ep].types[T]).
    assert _received(tmp_path, json.dumps({"sha256": "SHA", "endpoints": {
        TEST_EP: {"types": {"Crosswalk": {"submitted_lines": 1}}},
        PROD_EP: {"types": {"Crosswalk": {"submitted_lines": 0}}}}})) == [TEST_EP]


def test_records_received_refuses_unreadable_json(tmp_path):
    with pytest.raises(SystemExit, match="not readable JSON"):
        _received(tmp_path, '{"sha256": "SHA", ')


@pytest.mark.parametrize("entry", [{"submitted_lines": "abc"}, {"submitted_lines": [1]},
                                   {"types": {"Crosswalk": {"submitted_lines": "abc"}}}])
def test_records_received_refuses_a_line_count_that_is_not_a_count(tmp_path, entry):
    with pytest.raises(SystemExit, match="not a line count"):
        _received(tmp_path, json.dumps({"sha256": "SHA", "endpoints": {TEST_EP: entry}}))


def test_records_mode_refreshes_received_by_on_resume(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a", "22b"])
    records = make_source_run(tmp_path / "juneau", ["11a", "22b"])
    argv = _records_argv(tmp_path, store, records)
    dfs.main_cli(argv + ["--limit", "1"])
    path = tmp_path / "run" / "manifest.json"
    assert json.loads(path.read_text())["pixels"]["records_received_by"] == [TEST_EP]
    write_record(records, endpoints=(TEST_EP, PROD_EP))      # the source file later went to prod
    dfs.main_cli(argv)
    assert json.loads(path.read_text())["pixels"]["records_received_by"] == sorted([TEST_EP, PROD_EP])


def _records_argv(tmp_path, store, records, run="run"):
    return ["--run-dir", str(tmp_path / run), "--store", str(store), "--records", str(records),
            *_xw(tmp_path)]


def test_records_mode_needs_the_files_submission_record(tmp_path, monkeypatch, crosswalk):
    # A results file PS never received (no record beside it), e.g. results.jsonl where
    # results.check.jsonl (the repositioned file) is what was sent.
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"], record=False)
    with pytest.raises(SystemExit, match="no submission record"):
        dfs.main_cli(_records_argv(tmp_path, store, records))
    assert not (tmp_path / "run").exists()


def test_records_mode_refuses_a_record_of_another_file_version(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    write_record(records, sha256="0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        dfs.main_cli(_records_argv(tmp_path, store, records))
    assert not (tmp_path / "run").exists()


def test_records_mode_refuses_a_record_that_sent_nothing(tmp_path, monkeypatch, crosswalk):
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    write_record(records, sent=0)
    with pytest.raises(SystemExit, match="no lines sent"):
        dfs.main_cli(_records_argv(tmp_path, store, records))


def test_records_mode_resumes_a_manifest_without_received_by(tmp_path, monkeypatch, crosswalk):
    # The committed Juneau manifests predate `records_received_by`: a resume must adopt the
    # key, not read its absence as other pixels. The endpoint list is not bound either: the
    # source file reaching prod later must not refuse a resume.
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a", "22b"])
    records = make_source_run(tmp_path / "juneau", ["11a", "22b"])
    argv = _records_argv(tmp_path, store, records)
    dfs.main_cli(argv + ["--limit", "1"])
    path = tmp_path / "run" / "manifest.json"
    m = json.loads(path.read_text())
    del m["pixels"]["records_received_by"]
    path.write_text(json.dumps(m), encoding="utf-8")

    write_record(records, endpoints=(TEST_EP, PROD_EP))
    dfs.main_cli(argv)
    m = json.loads(path.read_text())
    assert m["pixels"]["records_received_by"] == sorted([TEST_EP, PROD_EP])
    assert len((tmp_path / "run" / "results.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("flag", [["--server", SERVER], ["--ids", "ids.txt"],
                                  ["--labels", "labels.geojson"], ["--labels-user", "u"],
                                  ["--sample-unlabeled", "3"], ["--seed", "5"],
                                  ["--metadata-only"]])
def test_records_mode_refuses_each_server_mode_flag(tmp_path, monkeypatch, capsys, crosswalk,
                                                     flag):
    # A clean records file, so the only possible refusal is the flag's own: without the
    # exclusion the run would go through (and write a results file).
    _no_network(monkeypatch)
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a"])
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as e:
        dfs.main_cli(["--run-dir", str(run_dir), "--store", str(store), "--records", str(records),
                      *_xw(tmp_path), *flag])
    assert e.value.code == 2
    assert f"{flag[0]} does not apply with --records" in capsys.readouterr().err
    assert not (run_dir / "results.jsonl").exists()


def test_records_mode_refuses_a_duplicated_pano(tmp_path, crosswalk):
    store = make_store(tmp_path / "store", ["11a"])
    records = make_source_run(tmp_path / "juneau", ["11a", "11a"])
    with pytest.raises(SystemExit, match="twice"):
        dfs.main_cli(["--run-dir", str(tmp_path / "run"), "--store", str(store),
                      "--records", str(records), *_xw(tmp_path)])


def test_server_mode_stays_curb_ramp_only(tmp_path):
    store = make_store(tmp_path / "store", ["AA1"])
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("AA1\n", encoding="utf-8")
    ckpt = tmp_path / "x.pth"
    ckpt.write_bytes(b"")
    with pytest.raises(SystemExit):
        dfs.main_cli(_args(tmp_path / "run", store, "--ids", str(ids_file), "--types",
                           "Crosswalk", "--crosswalk-checkpoint", str(ckpt)))
