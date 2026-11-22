"""scripts/area_difference.py: the new-only area of a grown boundary. A wrong difference
re-runs panos whose labels are live, and PS is insert-only, so pin the geometry."""
import importlib.util
from pathlib import Path

from shapely.geometry import box, mapping, shape

_spec = importlib.util.spec_from_file_location(
    "area_difference", Path(__file__).resolve().parents[1] / "scripts" / "area_difference.py")
area_difference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(area_difference)

D = 0.01  # degrees; ~1 km at the equator


def _fc(*geoms):
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(g)} for g in geoms]}


def test_grown_boundary_yields_only_the_added_area():
    old = box(0, 0, D, D)
    # Two regions; the second adds a block to the south. A hair of float noise on the
    # shared edge must come out as a dropped sliver, not as a second new area.
    new = _fc(box(0, 0, D, D + 1e-12), box(0, -D, D, 0))
    union, diff, dropped = area_difference.area_difference(new, mapping(old))
    assert diff.geom_type == "Polygon"
    assert diff.symmetric_difference(box(0, -D, D, 0)).area < 1e-12
    assert union.contains(old)
    assert dropped >= 0
    assert shape(mapping(diff)).intersection(old).area < 1e-12


def test_unchanged_boundary_yields_nothing():
    old = box(0, 0, D, D)
    _, diff, _ = area_difference.area_difference(_fc(old), {"type": "Feature", "geometry": mapping(old)})
    assert diff is None
