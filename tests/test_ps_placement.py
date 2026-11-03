"""ps_placement.py must reproduce Project Sidewalk's server-side label placement exactly.

The fixture is SidewalkWebpage's own cross-implementation parity file, copied verbatim from
SidewalkWebpage origin/develop `test/fixtures/latLngEstimationParity.json` (d84ecbbca,
2026-09-26); the Scala server and the JS client are pinned to the same cases.
"""
import json
import math
from pathlib import Path

import pytest

import ps_placement as pp

FIXTURE = json.loads((Path(__file__).parent / 'fixtures'
                      / 'latlng_estimation_parity.json').read_text(encoding='utf-8'))


def test_constants_match_fixture():
    c = FIXTURE['constants']
    assert pp.CAMERA_HEIGHT_M == c['camera_height_m']
    assert pp.BLEND_DEG == c['blend_deg']
    assert pp.MAX_DISTANCE_M == c['max_distance_m']
    assert pp.EARTH_RADIUS_KM * 1000.0 == c['earth_radius_m']


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['name'])
def test_parity_fixture(case):
    tol = FIXTURE['tolerance']
    exp = case['expected']
    heading, pitch = pp.pov_from_pano_xy(case['pano_x'], case['pano_y'], case['pano_width'],
                                         case['pano_height'], case['camera_heading'])
    # The server keeps Scala's signed %, the reference wraps into [0, 360): same bearing.
    assert heading % 360 == pytest.approx(exp['heading_deg'], abs=tol['heading_deg'])
    assert pitch == pytest.approx(exp['pitch_deg'], abs=tol['heading_deg'])
    assert pp.estimate_distance_m(-pitch) == pytest.approx(exp['distance_m'],
                                                           abs=tol['distance_m'])
    lat, lng = pp.label_latlng(case['pano_lat'], case['pano_lng'], case['pano_x'],
                               case['pano_y'], case['pano_width'], case['pano_height'],
                               case['camera_heading'])
    assert lat == pytest.approx(exp['lat'], abs=tol['lat_lng_deg'])
    assert lng == pytest.approx(exp['lng'], abs=tol['lat_lng_deg'])


def test_45_degrees_is_camera_height_and_nadir_is_zero():
    assert pp.estimate_distance_m(45.0) == pytest.approx(pp.CAMERA_HEIGHT_M, abs=1e-12)
    assert pp.estimate_distance_m(90.0) == pytest.approx(0.0, abs=1e-12)


def test_tail_maximum_at_and_above_horizon():
    top = pp.estimate_distance_m(0.0)
    assert top == pytest.approx(23.85, abs=0.01)
    assert pp.estimate_distance_m(-30.0) == top      # clamped at the horizon's value
    assert top < pp.MAX_DISTANCE_M                   # the 50 m cap never binds


def test_tail_continuous_in_value_and_slope_at_blend():
    b, eps = pp.BLEND_DEG, 1e-6
    assert pp.estimate_distance_m(b - 1e-12) == pytest.approx(pp.estimate_distance_m(b),
                                                              abs=1e-9)
    above = (pp.estimate_distance_m(b + 2 * eps) - pp.estimate_distance_m(b + eps)) / eps
    below = (pp.estimate_distance_m(b - eps) - pp.estimate_distance_m(b - 2 * eps)) / eps
    assert above == pytest.approx(below, rel=1e-4)


def test_signed_modulo_keeps_negative_heading():
    heading, _ = pp.pov_from_pano_xy(1664, 4160, 13312, 6656, 10.0)
    assert heading == -125.0
    assert math.isclose(heading % 360, 235.0)
