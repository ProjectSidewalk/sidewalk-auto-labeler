"""Geometry helpers of scripts/aerial_sidewalks.py (#104): anchor extraction and the
on-surface distance. No data, no network."""
import math

import pytest

pytest.importorskip('shapely')

from shapely.geometry import box  # noqa: E402

import aerial_sidewalks as asw  # noqa: E402


def test_crosswalk_across_a_street_gives_one_anchor_per_curb():
    cw = [box(0, 0, 3, 12)]                                   # 12 m street crossing
    sw = [box(-10, -3, 13, 0), box(-10, 12, 13, 15)]          # the two sidewalks
    anchors = sorted(asw.crosswalk_anchors(cw, sw))
    assert len(anchors) == 2
    (e0, n0), (e1, n1) = anchors
    assert e0 == pytest.approx(1.5) and e1 == pytest.approx(1.5)
    assert n0 < 1.0 and n1 > 11.0                             # at the curbs, not mid-street


def test_crosswalk_touching_no_sidewalk_gives_no_anchor():
    assert asw.crosswalk_anchors([box(0, 0, 3, 12)], [box(50, 50, 60, 52)]) == []


def test_network_endpoints_merge_with_polygon_anchors():
    cw = [box(0, 0, 3, 12)]
    sw = [box(-10, -3, 13, 0), box(-10, 12, 13, 15)]
    # the network's crosswalk endpoints sit on the same two corners -> still two anchors
    anchors = asw.crosswalk_anchors(cw, sw, extra_points=[(1.5, 0.5), (1.5, 11.5)])
    assert len(anchors) == 2
    # ...and a far endpoint is its own anchor
    assert len(asw.crosswalk_anchors(cw, sw, extra_points=[(40.0, 0.0)])) == 3


def test_surface_distance_is_zero_inside_and_metric_outside():
    s = asw.Surface([box(0, 0, 10, 2), box(20, 0, 22, 10)])
    d = s.distance([(5, 1), (5, 5), (15, 1), (21, 12)])
    assert list(d) == pytest.approx([0.0, 3.0, 5.0, 2.0])
    assert math.isinf(asw.Surface([]).distance([(0, 0)])[0])


def test_heatmap_offset_wraps_the_seam():
    dx, dy = asw.heatmap_offset(0.999, 0.5, 0.001, 0.51)
    assert dx == pytest.approx(2.048) and dy == pytest.approx(5.12)


def test_bootstrap_gap_needs_both_sides():
    assert asw.bootstrap_gap([[[(True, False)]]]) == (None, None, None)
