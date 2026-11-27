"""Pure-logic checks for scripts/recall_by_tilt.py (no data, no network)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geo  # noqa: E402
import recall_by_tilt as rbt  # noqa: E402


def test_gravity_elevation_matches_geo_world_ray():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = rng.uniform(0, 1), rng.uniform(0.3, 0.95)
        p113, r113 = rng.uniform(-25, 25), rng.uniform(-25, 25)
        pose = geo.Pose(0.0, 0.0, 0.0, -p113, r113, True, '')
        elev, _ = geo._world_ray(pose, (x - 0.5) * 2 * math.pi, (0.5 - y) * math.pi)
        assert abs(math.degrees(elev) - float(rbt.gravity_elev_deg(x, y, p113, r113))) < 1e-9


def test_T_first_order_and_sign():
    # #113: pitch > 0 nose down -> the horizon sits ABOVE the middle row straight ahead.
    T_ahead = -float(rbt.gravity_elev_deg(0.5, 0.5, 2.0, 0.0))
    assert abs(T_ahead - 2.0) < 1e-6
    # roll > 0 left side up -> right side down -> horizon above the middle row at b = +90.
    T_right = -float(rbt.gravity_elev_deg(0.75, 0.5, 0.0, 2.0))
    assert abs(T_right - 2.0) < 1e-6
    T_left = -float(rbt.gravity_elev_deg(0.25, 0.5, 0.0, 2.0))
    assert abs(T_left + 2.0) < 1e-6


def test_rig_class():
    assert rbt.rig_class('bayonne', {'camera_make': 'GoPro', 'camera_model': 'Max'}) == 'gopro_max'
    assert rbt.rig_class('lyon', {'camera_make': 'GoPro', 'camera_model': 'MAX2'}) == 'other_360'
    assert rbt.rig_class('richmond', {'camera_make': 'none', 'camera_model': 'none'}) == 'unknown'
    assert rbt.rig_class('paterson', {}) == 'gsv'


def test_logistic_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(0.3 - 0.8 * x)))).astype(float)
    beta = rbt.logistic_fit(np.column_stack([np.ones_like(x), x]), y)
    assert abs(beta[1] + 0.8) < 0.15
