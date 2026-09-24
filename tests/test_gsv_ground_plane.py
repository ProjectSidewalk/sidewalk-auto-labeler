"""gsv_ground_plane: the depth-frame -> grade/cross-slope decomposition and the ray
injection it rests on (#52).

These pin the frame mapping against depth.py's own ray convention (a synthetic payload
with a known slope), not against our reading of it, and pin the (pitch, roll) injection
through geo._world_ray against an exact ray-plane intersection -- so a wrong sign in
either place fails here rather than showing up as a plausible-looking ablation.
"""
import math

import pytest

import depth
import geo
import gsv_ground_plane as gp


def _payload(grade_deg, cross_deg, d=2.6):
    """A 64x32 payload that is one ground plane everywhere, with the given slopes."""
    nx, ny, nz = gp.depth_frame_normal(*gp.normal_from_slopes(grade_deg, cross_deg))
    planes = [depth.Plane(0.0, 0.0, 0.0, 0.0), depth.Plane(nx, ny, nz, d)]
    return depth.DepthPayload(64, 32, planes, bytes([1]) * (64 * 32))


@pytest.mark.parametrize('grade,cross', [(3.0, 0.0), (0.0, -2.0), (-4.5, 1.25),
                                         (0.7, 0.3), (10.0, -6.0)])
def test_slopes_round_trip_through_the_depth_frame(grade, cross):
    nx, ny, nz = gp.depth_frame_normal(*gp.normal_from_slopes(grade, cross))
    # either orientation of the stored normal describes the same plane
    for sign in (1, -1):
        g, c = gp.slopes(*gp.camera_frame_normal(sign * nx, sign * ny, sign * nz))
        assert g == pytest.approx(grade, abs=1e-9)
        assert c == pytest.approx(cross, abs=1e-9)


def test_uphill_ahead_is_closer_ahead_in_depth_py():
    # A road rising ahead meets a forward ray (x = 0.5, the heading) sooner than a
    # backward one (x = 0.0) at the same depression. This ties "forward = -y" to the
    # coordinate convention depth.py actually uses, so a flipped axis fails here.
    p = _payload(3.0, 0.0)
    ahead = depth.ground_range_at(p, 0.5, 0.6)
    behind = depth.ground_range_at(p, 0.0, 0.6)
    assert ahead < behind
    # ...and by the right amount: exact intersection with a 3 deg plane, 18 deg down
    dep, s = math.radians(18.0), math.radians(3.0)
    h = 2.6 / math.cos(s)                 # vertical drop to the plane under the camera
    expect = h / (math.tan(dep) + math.tan(s))
    assert ahead == pytest.approx(expect, rel=1e-9)


def test_rising_to_the_right_is_closer_on_the_right_in_depth_py():
    p = _payload(0.0, 5.0)
    right = depth.ground_range_at(p, 0.75, 0.6)   # heading + 90 deg
    left = depth.ground_range_at(p, 0.25, 0.6)
    assert right < left


def test_slope_along_matches_the_components():
    assert gp.slope_along(2.0, -1.0, 0.0) == pytest.approx(2.0)
    assert gp.slope_along(2.0, -1.0, 90.0) == pytest.approx(-1.0)
    assert gp.slope_along(2.0, -1.0, -90.0) == pytest.approx(1.0)
    assert gp.slope_along(2.0, -1.0, 180.0) == pytest.approx(-2.0)


def _exact_ground_point(heading_deg, n, x, y, h=geo.DEFAULT_CAMERA_HEIGHT_M):
    """(east, north) of the exact ray/plane intersection; plane h below the camera
    along its upward normal n = (f, r, u) in the gravity-level camera frame."""
    phi, theta = (x - 0.5) * 2 * math.pi, (0.5 - y) * math.pi
    v = (math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), math.sin(theta))
    t = -h / sum(a * b for a, b in zip(n, v))
    pf, pr = t * v[0], t * v[1]
    b = math.radians(heading_deg)
    return pf * math.sin(b) + pr * math.cos(b), pf * math.cos(b) - pr * math.sin(b)


@pytest.mark.parametrize('grade,cross', [(3.0, 0.0), (0.0, 3.0), (-4.0, 2.0), (2.0, -2.0)])
def test_road_pose_injection_matches_the_exact_plane_intersection(grade, cross):
    """The ground-normal arm goes through geo.detection_ground_point with the camera's
    pose relative to the plane injected. That places a point at the in-plane range along
    the in-plane bearing from the foot of the perpendicular; the exact intersection's
    horizontal projection differs only by the in-plane/horizontal cos factor. Pin that it
    stays under 0.5% of the range out to 25 m at these slopes -- far below the 2-3 m
    within-site spread the arms are compared on. (Without the foot shift it is ~1.6%:
    a constant h*sin(slope) downhill bias per pano.)"""
    n = gp.normal_from_slopes(grade, cross)
    pose = geo.pano_pose({'lat': 40.0, 'lng': -74.0, 'camera_heading': 30.0, 'source': 'launch',
                          **gp.ground_frame_fields(40.0, -74.0, 30.0, n)})
    frame = geo.LocalFrame(40.0, -74.0)
    for x in (0.1, 0.3, 0.45, 0.5, 0.62, 0.75, 0.9):
        for y in (0.56, 0.6, 0.7, 0.85):
            g = geo.detection_ground_point(pose, x, y, max_range_m=math.inf)
            e_x, n_x = _exact_ground_point(30.0, n, x, y)
            rng = math.hypot(e_x, n_x)
            if g is None or rng > 25.0:
                continue
            e_g, n_g = frame.to_enu(g.lat, g.lng)
            assert math.hypot(e_g - e_x, n_g - n_x) < 0.005 * rng


def test_zero_pose_is_the_flat_raycast():
    # A pano without a measured plane gets (0, 0) in the non-off arms; that has to be
    # the production flat raycast, or those panos would move between arms.
    base = {'lat': 40.0, 'lng': -74.0, 'camera_heading': 211.0, 'source': 'launch'}
    flat = geo.pano_pose({**base, 'camera_pitch': None, 'camera_roll': None})
    zero = geo.pano_pose({**base, 'camera_pitch': 0.0, 'camera_roll': 0.0})
    for x, y in ((0.2, 0.6), (0.5, 0.7), (0.93, 0.58)):
        a = geo.detection_ground_point(flat, x, y, apply_pose=False)
        b = geo.detection_ground_point(zero, x, y, apply_pose=True)
        assert a.lat == pytest.approx(b.lat, abs=1e-12)
        assert a.lng == pytest.approx(b.lng, abs=1e-12)


def test_shuffled_arm_is_a_permutation_of_the_real_one():
    meas = {f'p{i}': dict(zip(('n_f', 'n_r', 'n_u'), gp.normal_from_slopes(i * 0.3, -i * 0.1)))
            for i in range(20)}
    arms = gp.arm_normals(meas)
    assert sorted(arms['ground-normal'].values()) == sorted(arms['shuffled-normal'].values())
    assert arms['ground-normal'] != arms['shuffled-normal']
    assert arms == gp.arm_normals(meas)         # fixed seed: reproducible


def _abl(city, arm, top):
    """An ablation row whose two top buckets read `top` (off reads 3.0 in both)."""
    row = {'city': city, 'arm': arm, 'site_set': 'uncapped'}
    for b in gp.TOP_BUCKETS:
        row[f'n_pairs_grade_{b}'] = 500
        row[f'median_pair_m_grade_{b}'] = top
    return row


def _eval(city, arm, p90):
    return {'city': city, 'arm': arm, 'match_dist_p90': p90}


def test_verdict_reads_the_preregistered_rule():
    cities = ['a', 'b', 'c', 'd']

    def rows(gn, sh, p90_gn=2.0):
        abl, ev = [], []
        for c, g, s in zip(cities, gn, sh):
            abl += [_abl(c, 'off', 3.0), _abl(c, 'ground-normal', g),
                    _abl(c, 'shuffled-normal', s)]
            ev += [_eval(c, 'off', 2.0), _eval(c, 'ground-normal', p90_gn)]
        return abl, ev

    assert gp.verdict(*rows([2.5, 2.5, 2.5, 3.5], [3.1] * 4), cities)[0] == 'SUPPORTED'
    # the control matching the treatment undercuts, however well the treatment does
    assert gp.verdict(*rows([2.5] * 4, [2.6] * 4), cities)[0] == 'UNDERCUT'
    # off winning undercuts
    assert gp.verdict(*rows([3.2, 3.2, 3.1, 2.5], [3.1] * 4), cities)[0] == 'UNDERCUT'
    # a p90 loss anywhere blocks SUPPORTED without making it UNDERCUT
    label, detail = gp.verdict(*rows([2.5] * 4, [3.1] * 4, p90_gn=2.01), cities)
    assert label == 'NOT SUPPORTED' and 'p90' in detail['why']


def test_a_thin_top_bucket_is_not_testable():
    abl = [_abl('a', 'off', 3.0), _abl('a', 'ground-normal', 2.0)]
    for r in abl:
        for b in gp.TOP_BUCKETS:
            r[f'n_pairs_grade_{b}'] = gp.MIN_BUCKET_PAIRS - 1
    assert gp.bucket_win(abl, 'a', 'ground-normal') is None


def _meas(n=40):
    """Measured-plane rows spanning every grade bucket, both grade signs, both cross signs."""
    out = {}
    for i in range(n):
        g = (-1) ** i * (i * 0.15)          # 0 .. ~5.9 deg, alternating sign
        c = (-1) ** (i // 2) * (0.2 + (i % 7) * 0.4)
        out[f'p{i:02d}'] = dict(zip(('n_f', 'n_r', 'n_u'), gp.normal_from_slopes(g, c)))
    return out


def _gc(n):
    g, c = gp.slopes(*n)
    return round(g, 9), round(c, 9)


def test_review_arms_decompose_the_observed_plane():
    # travel-only is #50's correction on GSV: the grade kept, the cross-slope zeroed.
    # cross-only is the reverse, and the flipped arms negate only the cross-slope.
    meas = _meas()
    arms = gp.arm_normals(meas)
    for pid in meas:
        g, c = _gc(arms['ground-normal'][pid])
        assert _gc(arms['travel-only'][pid]) == (g, 0.0)
        assert _gc(arms['cross-only'][pid]) == (0.0, c)
        assert _gc(arms['cross-only-flipped'][pid]) == (0.0, -c)
        assert _gc(arms['cross-flipped'][pid]) == (g, -c)


def test_travel_shuffled_is_a_permutation_of_the_grades_with_no_cross():
    arms = gp.arm_normals(_meas())
    real = sorted(_gc(n)[0] for n in arms['ground-normal'].values())
    for arm in ('travel-shuffled', 'travel-bucket-shuffled'):
        gcs = [_gc(n) for n in arms[arm].values()]
        assert sorted(g for g, _ in gcs) == real
        assert all(c == 0.0 for _, c in gcs)
    assert arms['travel-shuffled'] != arms['travel-only']


@pytest.mark.parametrize('arm,real_arm', [('travel-bucket-shuffled', 'travel-only'),
                                          ('bucket-shuffled-normal', 'ground-normal')])
def test_bucket_shuffles_are_magnitude_matched(arm, real_arm):
    # The magnitude-matched control: every pano gets a grade from its OWN |grade| bucket,
    # so in the 4+ deg bucket the control applies >= 4 deg too -- unlike the city-wide
    # shuffle, which hands a steep pano a typical (~1 deg) correction.
    arms = gp.arm_normals(_meas())
    moved = 0
    for pid, n in arms[arm].items():
        own = arms[real_arm][pid]
        assert gp.grade_bucket(abs(_gc(n)[0])) == gp.grade_bucket(abs(_gc(own)[0]))
        moved += n != own
    assert moved > len(arms[arm]) // 2       # a real permutation, not the identity
    assert sorted(arms[arm].values()) == sorted(arms[real_arm].values())
    assert gp.arm_normals(_meas())[arm] == arms[arm]      # fixed seed


def test_existing_arms_are_unchanged_by_the_review_arms():
    # The pre-registered shuffle must draw the same permutation it always did, or the
    # committed three-arm rows would stop reproducing.
    meas = _meas()
    pids = sorted(meas)
    values = [(meas[p]['n_f'], meas[p]['n_r'], meas[p]['n_u']) for p in pids]
    import random
    random.Random(gp.SHUFFLE_SEED).shuffle(values)
    assert gp.arm_normals(meas)['shuffled-normal'] == dict(zip(pids, values))


def test_pair_side_reads_x_against_the_heading():
    # x_normalized < 0.5 is left of the heading (azimuth (x - 0.5) * 360 < 0)
    assert gp.pair_side(0.1, 0.49) == 'left'
    assert gp.pair_side(0.5, 0.99) == 'right'
    assert gp.pair_side(0.3, 0.7) == 'mixed'
    assert gp.pair_side(0.7, 0.3) == 'mixed'


def test_a_flipped_cross_slope_mirrors_left_and_right_placements():
    # The side split only reads the cross-slope sign if flipping it swaps what happens
    # on the two sides: a detection on the left under +c lands where its mirror image on
    # the right lands under -c (same range), and the plane rising to the right shortens
    # right-side ranges.
    base = {'lat': 40.0, 'lng': -74.0, 'camera_heading': 0.0, 'source': 'launch'}
    up_right = gp.normal_from_slopes(0.0, 3.0)
    up_left = gp.normal_from_slopes(0.0, -3.0)
    pose_r = geo.pano_pose({**base, **gp.ground_frame_fields(40.0, -74.0, 0.0, up_right)})
    pose_l = geo.pano_pose({**base, **gp.ground_frame_fields(40.0, -74.0, 0.0, up_left)})
    for x, y in ((0.25, 0.6), (0.35, 0.7)):
        left_r = geo.detection_ground_point(pose_r, x, y, max_range_m=math.inf)
        right_r = geo.detection_ground_point(pose_r, 1 - x, y, max_range_m=math.inf)
        right_l = geo.detection_ground_point(pose_l, 1 - x, y, max_range_m=math.inf)
        assert right_r.range_m < left_r.range_m
        assert right_l.range_m == pytest.approx(left_r.range_m, rel=1e-9)
