"""mapillary_tilt: the computed_rotation -> (heading, pitch, roll) convention lock (#42).

These pin the parse against things that do not depend on our own reading of the
docs: a hand-built rotation, the round trip through geo._world_ray, and one real
record whose values Project Sidewalk's own Mapillary viewer independently derived.
"""
import math
import random

import pytest

import geo
import mapillary_tilt as mt


def _mat_close(a, b, tol=1e-9):
    return all(abs(a[i][j] - b[i][j]) < tol for i in range(3) for j in range(3))


def test_level_camera_facing_north_is_a_quarter_turn_about_east():
    # world->camera for a level camera looking north: camera x = east, camera y = down,
    # camera z = north, i.e. a +90 deg rotation about the world x (east) axis.
    R = mt.rotation_matrix([math.pi / 2, 0.0, 0.0])
    assert _mat_close(R, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    pose = mt.opensfm_pose([math.pi / 2, 0.0, 0.0])
    assert pose['heading_deg'] == pytest.approx(0.0, abs=1e-9)
    assert pose['pitch_deg'] == pytest.approx(0.0, abs=1e-9)
    assert pose['roll_deg'] == pytest.approx(0.0, abs=1e-9)
    assert pose['tilt_deg'] == pytest.approx(0.0, abs=1e-9)


def test_decomposition_round_trips_and_matches_world_ray():
    # For random rotations the decomposed (heading, pitch, roll) must rebuild the same
    # matrix AND geo._world_ray must reproduce the direct R^T * bearing raycast.
    rng = random.Random(42)
    for _ in range(200):
        rvec = [rng.uniform(-math.pi, math.pi) for _ in range(3)]
        R = mt.rotation_matrix(rvec)
        pose = mt.opensfm_pose(rvec)
        if abs(pose['pitch_deg']) > 85:      # gimbal lock is the one excluded region
            continue
        assert _mat_close(R, mt.matrix_from_pose(pose['heading_deg'], pose['pitch_deg'],
                                                 pose['roll_deg']))
        gpose = geo.Pose(0.0, 0.0, pose['heading_deg'], pose['pitch_deg'], pose['roll_deg'],
                         True, 'mapillary')
        for _ in range(5):
            x, y = rng.random(), rng.random()
            phi, theta = (x - 0.5) * 2 * math.pi, (0.5 - y) * math.pi
            elev_a, bear_a = geo._world_ray(gpose, phi, theta)
            elev_b, bear_b = mt.world_direction(R, x, y)
            assert elev_a == pytest.approx(elev_b, abs=1e-9)
            assert geo.norm_deg(math.degrees(bear_a - bear_b)) == pytest.approx(0.0, abs=1e-7)


def test_pure_pitch_and_pure_roll_have_the_documented_signs():
    # Camera pitched up 10 deg: forward has +elevation -> pitch +10, roll 0.
    R = mt.matrix_from_pose(30.0, 10.0, 0.0)
    fwd = R[2]
    assert math.degrees(math.asin(fwd[2])) == pytest.approx(10.0)
    # Camera rolled so the image's right side lifts: the right axis gains +up.
    R = mt.matrix_from_pose(30.0, 0.0, 10.0)
    assert R[0][2] == pytest.approx(math.sin(math.radians(10.0)))
    # ...and Project Sidewalk's sign is the negative of that.
    pose = mt.opensfm_pose([0, 0, 0])
    assert pose['roll_ps_deg'] == -pose['roll_deg']


def test_real_record_matches_mapillary_compass_and_project_sidewalk_viewer():
    # runs/richmond pano 2163793620710887 (iSTAR Pulsar). computed_compass_angle is
    # Mapillary's own number; camera_pitch/camera_roll are what Project Sidewalk's
    # MapillaryViewer.extractPitchRoll wrote into pano_data for the same pano
    # (read back from sidewalk-richmond-test on 2026-09-04: 2.231776541825151,
    # -6.354952531976068).
    rvec = [1.3857263583832, 0.71804330335161, -0.58250746512038]
    pose = mt.opensfm_pose(rvec)
    assert pose['heading_deg'] == pytest.approx(311.5711847973, abs=1e-8)
    assert pose['pitch_deg'] == pytest.approx(2.231776541825151, abs=1e-9)
    assert pose['roll_ps_deg'] == pytest.approx(-6.354952531976068, abs=1e-9)
    assert pose['roll_deg'] == pytest.approx(6.354952531976068, abs=1e-9)
