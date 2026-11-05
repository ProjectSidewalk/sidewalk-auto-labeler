"""The partial-run guard of panoramax_bayonne_figures.py `data` (PR 154 final review, S1)."""
import hashlib
from types import SimpleNamespace

import pytest

import panoramax_bayonne_figures as bf


class _Loaded(Exception):
    """Raised by the stubbed loader: the guard let the run through to the data."""


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(bf, "DATA", data)
    monkeypatch.setattr(bf, "_load_city", lambda *a: (_ for _ in ()).throw(_Loaded()))
    root = tmp_path / "runs"
    (root / "bayonne").mkdir(parents=True)
    certified = b'{"pano": "10 m"}\n'
    (data / "inputs.csv").write_text(
        "file,sha256,used_for\n"
        f"runs/bayonne/results.jsonl,{hashlib.sha256(certified).hexdigest()},the measurement\n",
        encoding="utf-8")
    return root, certified


def _data(root, only, allow=False):
    bf.cmd_data(SimpleNamespace(run_root=root, only=only, allow_changed_inputs=allow))


@pytest.mark.parametrize("only", [["fig5"], ["inputs", "fig5"]])
def test_partial_run_on_changed_inputs_is_refused_with_or_without_inputs(run_root, only):
    root, _ = run_root
    (root / "bayonne" / "results.jsonl").write_bytes(b'{"pano": "5 m"}\n')
    with pytest.raises(SystemExit, match="runs/bayonne/results.jsonl"):
        _data(root, only)
    with pytest.raises(_Loaded):                        # the explicit override
        _data(root, only, allow=True)


def test_partial_run_on_certified_inputs_and_full_run_pass(run_root):
    root, certified = run_root
    (root / "bayonne" / "results.jsonl").write_bytes(certified)
    with pytest.raises(_Loaded):
        _data(root, ["inputs", "fig5"])
    (root / "bayonne" / "results.jsonl").write_bytes(b'{"pano": "5 m"}\n')
    with pytest.raises(_Loaded):                        # a full run moves everything together
        _data(root, None)
    with pytest.raises(_Loaded):
        _data(root, list(bf.DATA_PARTS))
