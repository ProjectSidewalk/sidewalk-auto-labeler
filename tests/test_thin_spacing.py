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
    assert m["thin_spacing_m"] == 10 and "thin_spacing_bound_on_resume" in m
    assert main.bind_thin_spacing(m, 10, tmp_path) is False
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit, match="--thin-spacing 10"):
        main.bind_thin_spacing(m, 5, tmp_path)


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
