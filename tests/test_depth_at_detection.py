"""depth_at_detection (#47 step 1): the plane class, the local height offset, the column
walk and the pre-registered verdict.

All payloads are synthetic (tests/test_depth.py's build_payload); no real payload bytes
are used. The planes that matter carry a nonzero normal-x or sit at an asymmetric image
column, so a mirrored (raster-frame, `1 - x`) lookup fails here rather than passing
unnoticed (#80).
"""
import math

import pytest

import depth
import depth_at_detection as dad
from test_depth import GROUND, WALL, build_payload, tilted_ground

W, H = 64, 32
PATCH_COLS, PATCH_ROWS = range(8, 12), range(20, 24)       # floor patch, image columns 8-11
WALL_COLS, WALL_ROWS = range(40, 44), range(18, 22)
LID_COLS, LID_ROWS = range(24, 28), range(10, 18)          # 6 of 8 rows above the horizon


def _index(ground=GROUND, patch_d=2.35):
    """Ground below the horizon, a level floor patch `2.5 - patch_d` m above it, a wall,
    and a level plane mostly ABOVE the horizon (an overpass underside)."""
    planes = [GROUND, ground, (0.0, 0.0, -1.0, patch_d), WALL, (0.0, 0.0, 1.0, 3.0)]
    idx = [depth.SKY] * (W * H)
    for r in range(H // 2, H):
        for c in range(W):
            idx[r * W + c] = 1
    for plane, cols, rows in ((2, PATCH_COLS, PATCH_ROWS), (3, WALL_COLS, WALL_ROWS),
                              (4, LID_COLS, LID_ROWS)):
        for r in rows:
            for c in cols:
                idx[r * W + c] = plane
    payload = depth.parse(build_payload(W, H, planes, idx))
    return dad.PayloadIndex(payload)


def _xy(col, row):
    return (col + 0.5) / W, (row + 0.5) / H


def test_classes_and_range_in_the_image_frame():
    pix = _index()
    cases = {dad.GROUND: _xy(30, 26), dad.FLOOR: _xy(9, 21),
             dad.NON_HORIZONTAL: _xy(41, 19), dad.HORIZONTAL_NONFLOOR: _xy(25, 17),
             dad.NO_PLANE: _xy(5, 3)}
    for cls, (x, y) in cases.items():
        out = dad.classify(pix, x, y)
        assert out['plane_class'] == cls, cls
        want = depth.ground_range_at(pix.payload, x, y)
        if want is None:
            assert out['range_depth_m'] is None
        else:
            assert out['range_depth_m'] == pytest.approx(want, rel=1e-12)
    assert dad.classify(pix, *cases[dad.NON_HORIZONTAL])['plane_tilt_deg'] == \
        pytest.approx(90.0)
    # The #80 guard: the mirrored column of the floor patch is plain ground.
    x, y = cases[dad.FLOOR]
    assert dad.classify(pix, 1.0 - x, y)['plane_class'] == dad.GROUND


def test_offset_local_is_the_height_above_the_road():
    pix = _index()
    out = dad.classify(pix, *_xy(9, 21))
    assert out['offset_local_m'] == pytest.approx(0.15, abs=1e-6)
    assert out['rows_to_ref'] == 3                     # rows 22, 23 are patch; 24 is road
    assert out['offset_dominant_m'] == pytest.approx(0.15, abs=1e-6)


def test_offset_against_a_tilted_reference_is_analytic():
    t, b, h, patch_d = 3.0, 35.0, 2.5, 2.35
    pix = _index(ground=tilted_ground(t, b, h), patch_d=patch_d)
    x, y = 9.3 / W, 21.7 / H                            # not a pixel centre
    # The exact ray, written out independently of depth.py: image frame, +z down.
    theta, phi = (1.0 - y) * math.pi, (1.0 - x) * 2.0 * math.pi + math.pi / 2.0
    v = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    p = [c * patch_d / v[2] for c in v]                 # hit on the level patch
    tr, br = math.radians(t), math.radians(b)
    z_ref = h / math.cos(tr) + math.tan(tr) * (math.cos(br) * p[0] + math.sin(br) * p[1])
    out = dad.classify(pix, x, y)
    assert out['plane_class'] == dad.FLOOR
    assert out['offset_local_m'] == pytest.approx(z_ref - p[2], abs=1e-5)
    # a mirrored azimuth would put the hit elsewhere on the tilted plane
    pm = [p[0] * -1, p[1], p[2]]
    z_m = h / math.cos(tr) + math.tan(tr) * (math.cos(br) * pm[0] + math.sin(br) * pm[1])
    assert abs((z_m - pm[2]) - (z_ref - p[2])) > 1e-3


def test_column_walk_skips_non_floor_and_returns_the_first_different_floor():
    w, h = 8, 16
    planes = [GROUND, GROUND, (0, 0, -1, 2.3), WALL, (0, 0, 1, 3.0), (0, 0, -1, 2.45)]
    idx = [depth.SKY] * (w * h)
    for r in range(h // 2, h):
        for c in range(w):
            idx[r * w + c] = 1
    for r in range(0, 7):
        idx[r * w + 0] = 4                  # plane 4 lives above the horizon: not a floor
    col = 3
    for r, j in ((8, 2), (9, 3), (10, 2), (11, 4), (12, 5), (13, 1)):
        idx[r * w + col] = j
    for r, j in ((12, 2), (13, 3), (14, 3), (15, 3)):
        idx[r * w + 6] = j
    pix = dad.PayloadIndex(depth.parse(build_payload(w, h, planes, idx)))
    assert 4 not in pix.floor and 3 not in pix.floor and {1, 2, 5} <= pix.floor
    assert dad.local_reference(pix, 8, col, 2) == (5, 4)
    assert dad.local_reference(pix, 12, 6, 2) == (None, None)


def _gt(*cities):
    return {f'c{i}': {'true_n': n, 'true_surface': k, 'false_n': fn,
                      'false_non_surface': fk}
            for i, (n, k, fn, fk) in enumerate(cities)}


@pytest.mark.parametrize('cities,expected', [
    ([(100, 85, 10, 1), (100, 85, 10, 1)], 'SUPPORTED'),        # pooled exactly 0.85
    ([(100, 84, 10, 1)], 'NOT SUPPORTED'),
    ([(200, 190, 10, 1), (100, 80, 10, 1)], 'SUPPORTED'),       # a city exactly at 0.80
    ([(200, 190, 10, 1), (100, 79, 10, 1)], 'NOT SUPPORTED'),   # pooled 0.90, a city 0.79
    ([(100, 70, 10, 1)], 'NOT SUPPORTED'),                      # exactly 0.70
    ([(100, 69, 10, 1)], 'UNDERCUT'),
])
def test_verdict_rule_i_boundaries(cities, expected):
    assert dad.verdict(_gt(*cities))['rule_i'] == expected


@pytest.mark.parametrize('false_ns,expected', [(30, 'SUPPORTED'), (29, 'NOT SUPPORTED')])
def test_verdict_rule_ii_margin_is_twenty_points(false_ns, expected):
    # True non-surface share 10/100; False 30/100 is exactly 20 points above it.
    v = dad.verdict(_gt((100, 90, 100, false_ns)))
    assert v['rule_ii'] == expected


def test_verdict_rule_ii_needs_separated_intervals_and_flags_underpower():
    big = dad.verdict(_gt((1000, 900, 100, 60)))
    assert big['rule_ii'] == 'SUPPORTED' and big['underpowered'] is False
    # n = 49, the pooled verdict-False count: a 40-point gap still carries a wide interval
    small = dad.verdict(_gt((1000, 900, 49, 25)))
    assert small['underpowered'] is True
    # a large gap whose intervals overlap is not enough
    tiny = dad.verdict(_gt((20, 16, 5, 3)))
    assert tiny['non_surface_diff'] >= 0.20 and tiny['rule_ii'] == 'NOT SUPPORTED'


def test_offset_claim_is_read_descriptively():
    assert dad.verdict(_gt((10, 9, 1, 0)), (0.12, 0.10, 0.14))['offset_claim_0p15'] == \
        'consistent'
    assert dad.verdict(_gt((10, 9, 1, 0)), (0.02, 0.0, 0.04))['offset_claim_0p15'] == \
        'not consistent'


def test_doctests_run():
    import doctest
    assert doctest.testmod(dad).failed == 0
