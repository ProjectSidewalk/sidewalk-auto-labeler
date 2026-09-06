"""geo.py: the shared geodesy the fusion pipeline builds on."""
import math

import pytest

import geo


def test_norm_deg_folds_gsv_style_angles():
    assert geo.norm_deg(359.4) == pytest.approx(-0.6)
    assert geo.norm_deg(-181.0) == pytest.approx(179.0)
    assert geo.norm_deg(180.0) == pytest.approx(-180.0)
    assert geo.norm_deg(45.0) == pytest.approx(45.0)


def test_haversine_one_degree_of_latitude():
    # 1 degree of latitude is ~111.19 km on the sphere geo.py uses
    assert geo.haversine_m(40.0, -74.0, 41.0, -74.0) == pytest.approx(111195, rel=2e-3)


def test_local_frame_round_trip_and_haversine_agreement():
    frame = geo.LocalFrame(40.9168, -74.1718)  # paterson-ish
    lat, lng = 40.9203, -74.1655
    e, n = frame.to_enu(lat, lng)
    back = frame.to_latlng(e, n)
    assert back[0] == pytest.approx(lat, abs=1e-9)
    assert back[1] == pytest.approx(lng, abs=1e-9)
    # ENU Euclidean distance matches the great-circle distance at ~600 m scale
    assert math.hypot(e, n) == pytest.approx(
        geo.haversine_m(frame.lat0, frame.lng0, lat, lng), rel=5e-3)


def test_grid_index_finds_neighbors_across_cell_borders():
    grid = geo.GridIndex(cell_m=8.0)
    grid.add(7.9, 0.0, "a")   # just left of a cell border
    grid.add(8.1, 0.0, "b")   # just right of it
    grid.add(100.0, 100.0, "far")
    near = set(grid.near(8.0, 0.0))
    assert near == {"a", "b"}


def _flat_pose(lat=44.05, lng=-121.31, heading=0.0, pitch=None, roll=None):
    return geo.pano_pose({'lat': lat, 'lng': lng, 'camera_heading': heading,
                          'camera_pitch': pitch, 'camera_roll': roll,
                          'source': 'launch'})


def _y_for_depression(depression_rad):
    """y_normalized whose pano-frame ray points depression_rad below the horizon."""
    return 0.5 + depression_rad / math.pi


def test_ground_point_to_pano_inverts_the_flat_raycast():
    # exact inverse of detection_ground_point's flat path, including across the
    # seam (x near 0 and near 1) and at both ends of the usable range
    for heading in (0.0, 10.0, 187.5, 359.0):
        pose = _flat_pose(heading=heading)
        for x, y in ((0.02, 0.6), (0.98, 0.55), (0.5, 0.75), (0.25, 0.54)):
            g = geo.detection_ground_point(pose, x, y, apply_pose=False)
            assert g is not None
            p = geo.ground_point_to_pano(pose, g.lat, g.lng)
            assert p.x_norm == pytest.approx(x, abs=1e-9)
            assert p.y_norm == pytest.approx(y, abs=1e-9)
            assert p.range_m == pytest.approx(g.range_m, abs=1e-6)
            assert p.bearing_deg == pytest.approx(g.bearing_deg, abs=1e-6)


def test_ground_point_to_pano_drops_what_the_forward_path_drops():
    pose = _flat_pose(heading=90.0)
    near = geo.detection_ground_point(pose, 0.5, 0.75, apply_pose=False)  # ~2.6 m
    assert geo.ground_point_to_pano(pose, near.lat, near.lng) is not None
    # beyond max range: 40 m due east of the camera
    far_lat, far_lng = geo.LocalFrame(pose.lat, pose.lng).to_latlng(40.0, 0.0)
    assert geo.ground_point_to_pano(pose, far_lat, far_lng) is None
    assert geo.ground_point_to_pano(pose, far_lat, far_lng, max_range_m=50.0) \
        .range_m == pytest.approx(40.0, abs=1e-6)
    # the camera's own footprint has no bearing
    assert geo.ground_point_to_pano(pose, pose.lat, pose.lng) is None


def test_raycast_hand_computed_flat_case():
    pose = _flat_pose(heading=90.0)
    y = _y_for_depression(math.atan(2.6 / 10.0))
    g = geo.detection_ground_point(pose, 0.5, y)
    assert g.range_m == pytest.approx(10.0)
    assert g.bearing_deg == pytest.approx(90.0)
    # 10 m due east of the pano
    assert g.lat == pytest.approx(pose.lat, abs=1e-7)
    east_m = (g.lng - pose.lng) * geo.METERS_PER_DEG_LAT * math.cos(math.radians(pose.lat))
    assert east_m == pytest.approx(10.0, rel=1e-3)


def test_raycast_bearing_wraps_across_north():
    pose = _flat_pose(heading=350.0)
    g = geo.detection_ground_point(pose, 0.6, _y_for_depression(0.3))
    assert g.bearing_deg == pytest.approx(350.0 + 36.0 - 360.0)


def test_raycast_horizon_guard_and_range_drop():
    pose = _flat_pose()
    assert geo.detection_ground_point(pose, 0.5, 0.5) is None      # exactly at horizon
    assert geo.detection_ground_point(pose, 0.5, _y_for_depression(0.015)) is None
    # just past max range: dropped, never clamped
    y = _y_for_depression(math.atan(2.6 / 25.5))
    assert geo.detection_ground_point(pose, 0.5, y, max_range_m=25.0) is None
    assert geo.detection_ground_point(pose, 0.5, y, max_range_m=26.0) is not None


def test_pitch_shifts_dead_ahead_elevation():
    # +2 deg pitch raises the forward axis: a pixel at pano-frame depression 5 deg
    # sits at world depression 3 deg. (Fusion runs with apply_pose=False — the
    # ablation showed GSV equirects are pre-rectified — but the rotation itself
    # must stay correct for experiments and unrectified sources.)
    pose = _flat_pose(pitch=2.0, roll=0.0)
    g = geo.detection_ground_point(pose, 0.5, _y_for_depression(math.radians(5.0)),
                                   max_range_m=100.0)
    assert g.range_m == pytest.approx(2.6 / math.tan(math.radians(3.0)), rel=1e-6)


def test_roll_shifts_side_elevation():
    # +2 deg roll lifts the right side of the image (phi=90): same 5->3 deg shift
    pose = _flat_pose(pitch=0.0, roll=2.0)
    g = geo.detection_ground_point(pose, 0.75, _y_for_depression(math.radians(5.0)),
                                   max_range_m=100.0)
    assert g.range_m == pytest.approx(2.6 / math.tan(math.radians(3.0)), rel=1e-6)


def test_exact_rotation_matches_first_order_formula():
    # elev ~= theta + pitch*cos(phi) + roll*sin(phi), good to second order
    pose = _flat_pose(heading=120.0, pitch=3.0, roll=1.0)
    phi_deg, depression_deg = 40.0, 6.0
    g = geo.detection_ground_point(
        pose, 0.5 + phi_deg / 360.0, _y_for_depression(math.radians(depression_deg)),
        max_range_m=200.0)
    expected_elev = (-math.radians(depression_deg)
                     + math.radians(3.0) * math.cos(math.radians(phi_deg))
                     + math.radians(1.0) * math.sin(math.radians(phi_deg)))
    assert g.range_m == pytest.approx(2.6 / math.tan(-expected_elev), rel=5e-3)


def test_pose_off_and_missing_pitch_roll_fall_back_to_flat():
    tilted = _flat_pose(pitch=2.0, roll=1.0)
    flat = geo.detection_ground_point(tilted, 0.3, 0.62, apply_pose=False)
    mapillary = geo.pano_pose({'lat': 44.05, 'lng': -121.31, 'camera_heading': 0.0,
                               'camera_pitch': None, 'camera_roll': None,
                               'source': 'mapillary'})
    assert not mapillary.has_pitch_roll
    assert geo.detection_ground_point(mapillary, 0.3, 0.62).range_m \
        == pytest.approx(flat.range_m)


def test_pano_pose_normalizes_wrapped_angles():
    pose = _flat_pose(pitch=1.5, roll=359.4)  # GSV stores roll in [0, 360)
    assert pose.roll_deg == pytest.approx(-0.6)
    assert pose.has_pitch_roll


def test_covariance_grows_along_ray_with_range():
    pose = _flat_pose()
    near = geo.detection_ground_point(pose, 0.5, _y_for_depression(math.atan(2.6 / 10)))
    far = geo.detection_ground_point(pose, 0.5, _y_for_depression(math.atan(2.6 / 20)))
    # super-linear growth: the angular term scales ~d^2, the height term ~d
    assert far.sigma_along_m / near.sigma_along_m > 2.2
    assert far.sigma_cross_m / near.sigma_cross_m == pytest.approx(2.0, rel=1e-2)
    assert near.sigma_along_m > near.sigma_cross_m          # anisotropy, always


def test_cov_en_orientation_follows_bearing():
    g_north = geo.GroundEstimate(0, 0, 10, 0.0, sigma_along_m=2.0, sigma_cross_m=0.5)
    a, b, c = g_north.cov_en(sigma_gps_m=1.0)
    assert c > a and b == pytest.approx(0.0, abs=1e-12)     # along-ray = north
    g_east = geo.GroundEstimate(0, 0, 10, 90.0, sigma_along_m=2.0, sigma_cross_m=0.5)
    a, b, c = g_east.cov_en(sigma_gps_m=1.0)
    assert a > c
    g_diag = geo.GroundEstimate(0, 0, 10, 45.0, sigma_along_m=2.0, sigma_cross_m=0.5)
    assert g_diag.cov_en(sigma_gps_m=1.0)[1] > 0


def test_sym2_identities():
    m = (4.0, 1.0, 3.0)
    inv = geo.sym2_inv(m)
    # m @ inv == identity
    assert m[0] * inv[0] + m[1] * inv[1] == pytest.approx(1.0)
    assert m[0] * inv[1] + m[1] * inv[2] == pytest.approx(0.0)
    assert m[1] * inv[1] + m[2] * inv[2] == pytest.approx(1.0)
    # quadratic form against a hand computation: [1, 2] m [1, 2]^T
    assert geo.sym2_quadform(m, 1.0, 2.0) == pytest.approx(4.0 + 2 * 1.0 * 2.0 + 3.0 * 4.0)
    assert geo.sym2_add((1, 2, 3), (10, 20, 30)) == (11, 22, 33)
