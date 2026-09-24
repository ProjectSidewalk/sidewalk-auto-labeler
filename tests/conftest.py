"""Shared test setup: imports, a network kill-switch, and the Stage-1 record builders.

The suite needs only the lightweight dependencies (no torch/transformers — the
detector is imported lazily by main.py and never loaded here), and no test may
touch the network: every streetlevel entry point is stubbed to fail loudly unless
a test explicitly overrides it.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Root goes first: scripts/position_check.py is a shim sharing the root module's name.
# Forced, not gap-filled: `python -m pytest` already has the cwd (the root) on sys.path,
# and a "not in" guard would then leave scripts/ in front and the shim shadowing the module.
for p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import requests  # noqa: E402
from streetlevel import streetview  # noqa: E402  (needs the path setup above)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test reaches for Google or Mapillary; tests override these
    explicitly."""
    def blocked(*args, **kwargs):
        raise AssertionError("network call attempted in tests — monkeypatch the entry point")
    for fn in ("find_panorama_by_id", "get_panorama", "get_coverage_tile"):
        monkeypatch.setattr(streetview, fn, blocked)
    for fn in ("get", "post"):
        monkeypatch.setattr(requests, fn, blocked)


def make_metadata(**overrides):
    """A minimal streetlevel-shaped metadata object accepted by main.build_output_line."""
    base = dict(
        date=SimpleNamespace(year=2021, month=6),
        image_sizes=[SimpleNamespace(x=512, y=256), SimpleNamespace(x=16384, y=8192)],
        tile_size=SimpleNamespace(x=512, y=512),
        heading=math.radians(90.0),
        pitch=math.radians(1.5),
        roll=math.radians(-0.5),
        copyright_message="© Google",
        source="launch",
        historical=[
            SimpleNamespace(id="OLD1", date=SimpleNamespace(year=2019, month=8)),
            SimpleNamespace(id="OLD2", date=None),  # must be dropped, not crash
        ],
        links=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_process_result(**overrides):
    """A main.process_pano success result, ready for main.build_output_line. The pano
    block comes from the real GSV record builder so producer→consumer contract tests
    exercise the actual chain."""
    from sources import gsv
    base = dict(pano_id="PID",
                pano=gsv.build_pano_record("PID", 44.05, -121.31, make_metadata()),
                detections=[(0.5, 0.25, 0.9)])
    base.update(overrides)
    return base


# HF main since 2026-07-24 (the paper weights); in detectors.KNOWN_REVISIONS.
PAPER_REVISION = "606a11956743f7eb328d9207769034752f6191f4"


def make_provenance(revision=PAPER_REVISION, allow_unknown=False):
    """The provenance block CurbRampDetector resolves for `revision`, built by the same
    torch-free function it uses — so tests exercise the real table, not a copy of it."""
    import detectors
    return detectors.provenance_for_revision(revision, allow_unknown=allow_unknown)
