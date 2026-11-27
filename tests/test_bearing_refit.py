"""Pure-geometry checks for scripts/bearing_refit.py (no run data, no network)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import bearing_refit as br  # noqa: E402


def _ray(ce, cn, target, s2t=1e-4, s2g=1.0):
    b = math.degrees(math.atan2(target[0] - ce, target[1] - cn)) % 360.0
    return (ce, cn, b, s2t, s2g)


def test_exact_rays_recover_the_point():
    target = (3.0, 12.0)
    rays = [_ray(0, 0, target), _ray(5, 0, target), _ray(10, 0, target)]
    (e, n), ranges = br.bearing_solve(rays)
    assert math.isclose(e, 3.0, abs_tol=1e-9) and math.isclose(n, 12.0, abs_tol=1e-9)
    assert all(r > 0 for r in ranges)


def test_parallel_rays_are_singular():
    assert br.bearing_solve([(0, 0, 10.0, 1e-4, 1.0), (5, 0, 10.0, 1e-4, 1.0)]) is None


def test_b_solution_refuses_rays_meeting_behind_a_camera():
    # Two rays that diverge: their lines cross behind both cameras.
    pos, sp = br.b_solution([(0, 0, 300.0, 1e-4, 1.0), (10, 0, 60.0, 1e-4, 1.0)])
    assert pos is None and sp > 50


def test_spread_folds_opposed_rays():
    assert math.isclose(br.spread_deg([0.0, 175.0]), 5.0, abs_tol=1e-9)
    assert math.isclose(br.spread_deg([0.0, 90.0]), 90.0, abs_tol=1e-9)


def test_azimuth_error_wraps():
    assert math.isclose(br.azimuth_error((0, 0), (0, -1), 179.0), 1.0, abs_tol=1e-9)
    assert math.isclose(br.azimuth_error((0, 0), (0, -1), -179.0), -1.0, abs_tol=1e-9)
