"""harvest_depth --from-store: indexing pano-tools' depth artifacts (#56), offline.

The artifact under test is built by hand with the layout pano-tools documents for format
v3 (sidewalk-panorama-tools downloaders/gsv.py, _write_depth_artifact): `plane_indices`
verbatim in the payload's column order, `planes_n`/`planes_d` verbatim, and `depth` the
reconstruction identity in the SAME column order (sidewalk-panorama-tools#58 un-mirror). The raster here is
computed from that identity with numpy, not through depth.py, so the raster comparison is
an independent check of depth.py's image frame (#80) rather than a tautology.

The payload is deliberately asymmetric -- a wall on the left half only and a ground plane
tilted sideways -- because a level, symmetric payload cannot tell a mirrored index array
from a correct one (depth._direction_continuous).
"""
import base64
import csv
import math
import struct

import numpy as np
import pytest

import depth as depthlib
import fuse_sites
import harvest_depth as hd

W, H = 32, 16
TILT = math.radians(2.0)
PLANES = [(0.0, 0.0, 0.0, 0.0),                                  # 0: sky sentinel
          (math.sin(TILT), 0.0, -math.cos(TILT), 2.1),           # 1: ground, tilted sideways
          (0.0, 0.0, -1.0, 2.3),                                  # 2: a second ground patch
          (1.0, 0.0, 0.0, 6.0)]                                   # 3: a wall


def _indices():
    idx = np.zeros((H, W), np.uint8)
    idx[H // 2:, :] = 1                      # ground below the horizon
    idx[H // 2:, W // 2:W // 2 + 4] = 2      # a patch of the second ground plane
    idx[H // 4:H // 2 + 2, :W // 3] = 3      # a wall on the LEFT third only
    return idx


def _blob(idx=None, planes=PLANES):
    idx = _indices() if idx is None else idx
    off = depthlib.HEADER_BYTES
    header = bytes([off]) + struct.pack("<HHH", len(planes), W, H) + bytes([off])
    tail = b"".join(struct.pack("<ffff", *p) for p in planes)
    return base64.urlsafe_b64encode(header + idx.tobytes() + tail).decode().rstrip("=")


def _raster(idx, normals, dists):
    """pano-tools' reconstruction identity, in payload (= image) column order:
    depth[r, c] = |d_i / (v(r, c) . n_i)|, v at theta = (h-r-0.5)/h*pi,
    phi = (w-c-0.5)/w*2pi + pi/2; -1 where the index is 0."""
    h, w = idx.shape
    theta = (h - np.arange(h) - 0.5) / h * np.pi
    phi = (w - np.arange(w) - 0.5) / w * 2.0 * np.pi + np.pi / 2.0
    v = np.stack([np.sin(theta)[:, None] * np.cos(phi)[None, :],
                  np.sin(theta)[:, None] * np.sin(phi)[None, :],
                  np.broadcast_to(np.cos(theta)[:, None], (h, w))], axis=-1)
    n = normals.astype(np.float64)[idx]
    d = dists.astype(np.float64)[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(d / np.einsum("hwc,hwc->hw", v, n))
    return np.where(idx == 0, -1.0, r).astype(np.float32)


def _artifact(idx=None, version=3):
    idx = _indices() if idx is None else idx
    arr = np.asarray(PLANES, np.float32)
    fields = {"depth": _raster(idx, arr[:, :3], arr[:, 3]), "plane_indices": idx,
              "planes_n": arr[:, :3], "planes_d": arr[:, 3],
              "heading": 0.0, "pitch": 0.0, "roll": 0.0, "format_version": version}
    return fields


def _write_npz(store, pid, fields):
    d = store / pid[:2]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{pid}{hd.STORE_SUFFIX}"
    with open(path, "wb") as f:
        np.savez_compressed(f, **fields)
    return path


# ------------------------------------------------------------------------------ rebuild

def test_rebuilt_payload_is_the_wire_parse():
    raw = depthlib.parse(_blob())
    stored = hd.payload_from_npz(_artifact())
    assert (stored.width, stored.height) == (raw.width, raw.height)
    assert stored.planes == raw.planes
    assert stored.indices == raw.indices


def test_frame_check_passes_on_a_faithful_artifact():
    fields = _artifact()
    res = hd.compare_store_frame(depthlib.parse(_blob()), hd.payload_from_npz(fields),
                                 fields["depth"])
    assert res["ok"] and res["match"] == "identical", res
    assert res["raster_mismatches"] == 0
    assert res["discriminating"] > 0          # the points can see a mirror on this payload


def test_frame_check_catches_a_mirrored_index_array():
    """The failure the check exists for: an artifact whose plane indices were flipped
    left-right (e.g. copied from streetlevel's raster frame)."""
    fields = _artifact(idx=_indices()[:, ::-1].copy())
    res = hd.compare_store_frame(depthlib.parse(_blob()), hd.payload_from_npz(fields))
    assert not res["ok"] and res["match"] == "mismatch"
    assert not res["indices_identical"]
    assert not res["range_points"]            # ...and the range queries alone see it
    assert res["mirrored_agreement"] > res["index_agreement"]


def test_a_revised_live_payload_is_not_a_frame_error():
    """Google re-serves a slightly re-segmented reconstruction now and then (1 of the first
    5 Vancouver panos checked). Same frame, same ground: reported as `revised`, not failed."""
    live = _indices()
    live[H - 3:, 2:6] = 2                      # a few ground pixels re-assigned
    fields = _artifact()
    res = hd.compare_store_frame(depthlib.parse(_blob(idx=live)), hd.payload_from_npz(fields),
                                 fields["depth"])
    assert not res["indices_identical"]
    assert res["match"] == "revised" and res["ok"], res


def test_frame_check_catches_a_raster_in_the_wrong_frame():
    fields = _artifact()
    fields["depth"] = fields["depth"][:, ::-1].copy()   # streetlevel's mirrored raster
    res = hd.compare_store_frame(depthlib.parse(_blob()), hd.payload_from_npz(fields),
                                 fields["depth"])
    assert res["raster_mismatches"] > 0 and not res["ok"]


def test_v2_artifact_without_planes_is_refused():
    fields = _artifact(version=2)
    for k in ("plane_indices", "planes_n", "planes_d"):
        del fields[k]
    with pytest.raises(hd.NoPlanes):
        hd.payload_from_npz(fields)


def test_out_of_range_index_is_an_error_not_a_guess():
    fields = _artifact()
    fields["plane_indices"] = fields["plane_indices"].copy()
    fields["plane_indices"][0, 0] = len(PLANES)
    with pytest.raises(ValueError):
        hd.payload_from_npz(fields)


# ------------------------------------------------------------------------------ indexing

def test_store_index_has_the_harvest_schema_and_feeds_fuse_sites(tmp_path, capsys):
    store = tmp_path / "store"
    _write_npz(store, "AAone", _artifact())
    v2 = _artifact(version=2)
    for k in ("plane_indices", "planes_n", "planes_d"):
        del v2[k]
    _write_npz(store, "BBold", v2)
    (store / hd.STORE_LEDGER).write_text("pano_id,status\nCCgone,unavailable\nAAone,saved\n",
                                         encoding="utf-8")
    depth_dir = tmp_path / "run" / "depth"

    gone, no_planes, pending, anomalies = hd.reconcile_store(
        depth_dir, store, ["AAone", "BBold", "CCgone", "DDlater"])

    assert (gone, no_planes, pending, anomalies) == (1, 1, 1, 0)
    with open(depth_dir / "index.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        assert next(reader) == hd.INDEX_FIELDS
        rows = list(reader)
    assert [r[0] for r in rows] == ["AAone"]
    assert rows[0][1] == "AA/AAone.depth.npz"
    assert rows[0][3] == hd._sha256(store / "AA" / "AAone.depth.npz")
    assert hd.load_gone(depth_dir) == {"CCgone"}
    assert (depth_dir / hd.NO_PLANES_FILE).read_text().split() == ["BBold"]
    assert "PARTIAL" in capsys.readouterr().out

    # The index is read by the consumers unchanged: fuse_sites takes its heights.
    heights = fuse_sites.load_depth_index(depth_dir / "index.csv")
    ground = depthlib.ground_plane(depthlib.parse(_blob()))
    assert heights["AAone"][0] == pytest.approx(ground.camera_height_m, abs=1e-4)


def test_rehash_catches_an_altered_artifact(tmp_path, capsys):
    store = tmp_path / "store"
    _write_npz(store, "AAone", _artifact())
    depth_dir = tmp_path / "depth"
    hd.reconcile_store(depth_dir, store, ["AAone"])
    rows = list(csv.DictReader(open(depth_dir / "index.csv", newline="", encoding="utf-8")))
    rows[0]["sha256"] = "0" * 64                      # as if the file had changed under it
    with open(depth_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hd.INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert hd.reconcile_store(depth_dir, store, ["AAone"])[3] == 0      # size match: trusted
    assert hd.reconcile_store(depth_dir, store, ["AAone"], rehash=True)[3] == 1
    assert "ANOMALY" in capsys.readouterr().out


def test_store_index_refuses_a_harvest_dir_and_a_second_store(tmp_path):
    store = tmp_path / "store"
    _write_npz(store, "AAone", _artifact())
    harvested = tmp_path / "harvest"
    harvested.mkdir()
    (harvested / "AAone.json.gz").write_bytes(b"x")
    with pytest.raises(SystemExit):
        hd.reconcile_store(harvested, store, ["AAone"])
    depth_dir = tmp_path / "depth"
    hd.reconcile_store(depth_dir, store, ["AAone"])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit):
        hd.reconcile_store(depth_dir, other, ["AAone"])


def test_check_store_frame_fetches_through_fetch_depth(tmp_path, monkeypatch):
    """The live check goes through fetch_depth (stubbed here to write the wire payload)."""
    store = tmp_path / "store"
    _write_npz(store, "AAone", _artifact())

    def fake_fetch(pid, out_path):
        import gzip
        import json
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump({"pano_id": pid, "depth_b64": _blob()}, f)
        return None
    monkeypatch.setattr(hd, "fetch_depth", fake_fetch)
    results = hd.check_store_frame(store, ["AAone", "ZZnone"], 5, scratch=tmp_path / "s")
    assert [r["pano_id"] for r in results] == ["AAone"] and results[0]["ok"]

    _write_npz(store, "AAone", _artifact(idx=_indices()[:, ::-1].copy()))
    with pytest.raises(SystemExit, match="FAILED"):
        hd.check_store_frame(store, ["AAone"], 5, scratch=tmp_path / "s2")
