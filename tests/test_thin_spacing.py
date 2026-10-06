"""Issue #126: main.py records the thinning spacing and binds the run directory to it.
No network, no model, no torch."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from conftest import make_provenance

REPO = Path(__file__).resolve().parents[1]
GEOM = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


def _init(run_dir, source="mapillary", **kw):
    return main.load_or_init_run_dir(run_dir, "city.geojson", GEOM, "hash", source,
                                     provenance=make_provenance(), **kw)


def _with_records(run_dir, n=3):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps({"detections": [], "i": i}) + "\n" for i in range(n)),
        encoding="utf-8")


def test_resolve_thin_spacing():
    hook = SimpleNamespace(THIN_CELL_METERS=5, thin_panos=lambda p, m: p)
    assert main.resolve_thin_spacing(hook, None) == 5
    assert main.resolve_thin_spacing(hook, 10) == 10
    assert main.resolve_thin_spacing(hook, 0) == 0
    assert main.resolve_thin_spacing(SimpleNamespace(), 10) is None
    assert main.resolve_thin_spacing(SimpleNamespace(), None) is None


@pytest.mark.parametrize("value,other", [(10, 5), (0, 5), (None, 5), (5, None)])
def test_bind_fresh_same_and_mismatch(tmp_path, value, other):
    m = {}
    assert main.bind_thin_spacing(m, value, tmp_path) is True
    assert m == {"thin_spacing_m": value}                    # no records: bound silently
    assert main.bind_thin_spacing(m, value, tmp_path) is False
    with pytest.raises(SystemExit) as e:
        main.bind_thin_spacing(m, other, tmp_path)
    msg = str(e.value)
    assert "new --name" in msg and "#126" in msg
    if value is not None:
        assert f"--thin-spacing {value}" in msg


def test_legacy_run_with_records_binds_once_with_a_note(tmp_path, capsys):
    _with_records(tmp_path, 3)
    m = {}
    assert main.bind_thin_spacing(m, 10, tmp_path) is True
    out = capsys.readouterr().out
    assert out.count("predates thin-spacing binding") == 1
    assert "3 existing records" in out and "10 m" in out
    assert "saved to manifest.json now" in out and "removing those two keys" in out
    assert m["thin_spacing_m"] == 10 and "thin_spacing_bound_on_resume" in m
    assert main.bind_thin_spacing(m, 10, tmp_path) is False
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit, match="--thin-spacing 10"):
        main.bind_thin_spacing(m, 5, tmp_path)


def test_whitespace_only_results_bind_silently(tmp_path, capsys):
    (tmp_path / "results.jsonl").write_text("\n\n", encoding="utf-8")
    m = {}
    assert main.bind_thin_spacing(m, 10, tmp_path) is True
    assert capsys.readouterr().out == ""
    assert m == {"thin_spacing_m": 10}


def test_legacy_hookless_run_binds_silently(tmp_path, capsys):
    """A GSV run's legacy value is known (no hook has never thinned): no note."""
    _with_records(tmp_path, 3)
    m = {}
    assert main.bind_thin_spacing(m, None, tmp_path) is True
    assert capsys.readouterr().out == ""
    assert m == {"thin_spacing_m": None}


def test_load_or_init_run_dir(tmp_path):
    # Default (UNBOUND): --scan-only / --gap-fill-only / every existing caller binds nothing.
    m = _init(tmp_path / "scan")
    assert "thin_spacing_m" not in m
    m = _init(tmp_path / "scan")
    assert "thin_spacing_m" not in m
    assert "thin_spacing_m" not in json.loads(
        (tmp_path / "scan" / "manifest.json").read_text(encoding="utf-8"))

    run = tmp_path / "bayonne"
    assert _init(run, thin_spacing=10)["thin_spacing_m"] == 10
    assert _init(run, thin_spacing=10)["thin_spacing_m"] == 10        # same spacing resumes
    with pytest.raises(SystemExit, match="thin_spacing_m"):
        _init(run, thin_spacing=5)
    _init(run)                                                        # scan-only: no check

    gsv = tmp_path / "gsv"
    m = _init(gsv, source="gsv", thin_spacing=None)
    assert "thin_spacing_m" in m and m["thin_spacing_m"] is None
    on_disk = json.loads((gsv / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["thin_spacing_m"] is None
    _init(gsv, source="gsv", thin_spacing=None)                       # resume passes


def test_load_or_init_run_dir_binds_a_legacy_manifest(tmp_path, capsys):
    run = tmp_path / "richmond"
    _init(run)                                     # a manifest without the key
    _with_records(run, 2)
    m = _init(run, thin_spacing=5)
    assert "predates thin-spacing binding" in capsys.readouterr().out
    on_disk = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["thin_spacing_m"] == 5 and "thin_spacing_bound_on_resume" in on_disk
    assert m["thin_spacing_m"] == 5


def test_record_run_thinning_keys(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"runs": []}
    main.record_run(path, manifest, "t0", 100, 1, 2, 3, phase="gap_fill")
    main.record_run(path, manifest, "t0", 100, 1, 2, 3,
                    thinning={"thin_spacing_m": 10, "panos_before_thinning": 400})
    main.record_run(path, manifest, "t0", 100, 1, 2, 3,
                    thinning={"thin_spacing_m": None, "panos_before_thinning": 100})
    plain, thinned, gsv = manifest["runs"]
    assert "thin_spacing_m" not in plain and "panos_before_thinning" not in plain
    assert thinned["thin_spacing_m"] == 10 and thinned["panos_before_thinning"] == 400
    assert gsv["thin_spacing_m"] is None                  # always present on a main pass
    plain.pop("phase")
    for entry in (thinned, gsv):
        rest = {k: v for k, v in entry.items()
                if k not in ("thin_spacing_m", "panos_before_thinning")}
        assert rest.keys() == plain.keys()
        assert {k: v for k, v in rest.items() if k != "finished_at"} == \
               {k: v for k, v in plain.items() if k != "finished_at"}


def _main_py(*args):
    return subprocess.run([sys.executable, str(REPO / "main.py"), *args], capture_output=True,
                          text=True, cwd=REPO, timeout=120)


def test_help_mentions_the_binding():
    out = _main_py("--help")
    assert out.returncode == 0, out.stderr
    assert "thin_spacing_m" in out.stdout


def test_negative_thin_spacing_is_a_usage_error():
    """argparse rejects it before get_source(), so nothing touches the network."""
    out = _main_py("x.geojson", "--thin-spacing", "-1", "--scan-only")
    assert out.returncode == 2
    assert "thin-spacing" in out.stderr


# --- run_labeler wiring (PR #136 review) -------------------------------------------------
# A fake thinning source over one zoom-14 tile; already_processed.txt is pre-filled with every
# pano the thinning keeps, so the main pass takes the "no new panoramas" branch and nothing
# is fetched. No gap fill (no fetch_pano_by_id), no position check: no network.

WIRING_ZOOM = 14
FAKE_PANOS = {f"p{i}": (37.5 + i * 1e-5, -77.4) for i in range(6)}
KEPT = sorted(FAKE_PANOS)[:2]


def _thinning_source(calls):
    def fetch_panos_for_tile(x, y, area_shape):
        calls.append((x, y))
        return dict(FAKE_PANOS)

    def thin_panos(panos, cell_meters):
        calls.append(("thin", cell_meters))
        return {k: v for k, v in panos.items() if k in KEPT}

    return SimpleNamespace(NAME="fake", COVERAGE_TILE_ZOOM=WIRING_ZOOM, THIN_CELL_METERS=5,
                           fetch_panos_for_tile=fetch_panos_for_tile, thin_panos=thin_panos)


def _area(tmp_path):
    x, y = main.latlon_to_tile(37.5, -77.4, WIRING_ZOOM)
    west, south, east, north = main.tile_lonlat_bounds(x, y, WIRING_ZOOM)
    w, s, e, n = (west + (east - west) / 4, south + (north - south) / 4,
                  east - (east - west) / 4, north - (north - south) / 4)
    path = tmp_path / "t.geojson"
    path.write_text(json.dumps({"type": "Polygon",
                                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}))
    return path


def _run(tmp_path, calls, **kw):
    main.run_labeler(str(_area(tmp_path)), "t", _thinning_source(calls), check_positions=False,
                     **kw)
    return json.loads((tmp_path / "runs" / "t" / "manifest.json").read_text(encoding="utf-8"))


def test_run_labeler_binds_and_records_on_the_no_new_panos_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "t"
    run.mkdir(parents=True)
    (run / "already_processed.txt").write_text("".join(f"{k}\n" for k in KEPT))
    calls = []
    m = _run(tmp_path, calls, thin_spacing=7, provenance=make_provenance())
    assert ("thin", 7) in calls
    assert m["thin_spacing_m"] == 7                         # bound (not UNBOUND)
    (entry,) = m["runs"]
    assert entry["processed"] == 0                          # the "no new panoramas" branch
    assert entry["thin_spacing_m"] == 7
    assert entry["panos_before_thinning"] == len(FAKE_PANOS)
    assert entry["panos_found_in_area"] == len(KEPT)

    # A resume at another spacing is refused before any tile is requested.
    calls.clear()
    with pytest.raises(SystemExit, match="thin_spacing_m"):
        _run(tmp_path, calls, thin_spacing=5, provenance=make_provenance())
    assert calls == []


def test_run_labeler_scan_only_binds_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    m = _run(tmp_path, calls, thin_spacing=7, scan_only=True)
    assert "thin_spacing_m" not in m and m["runs"] == []
    # ...and does not refuse a dir bound at another spacing.
    run = tmp_path / "runs" / "t"
    bound = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    bound["thin_spacing_m"] = 10
    (run / "manifest.json").write_text(json.dumps(bound), encoding="utf-8")
    assert _run(tmp_path, calls, thin_spacing=7, scan_only=True)["thin_spacing_m"] == 10
