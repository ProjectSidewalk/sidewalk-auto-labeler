"""label_frame_beta: the tilt term's sign, the pairing window and the clustered fit that
the AI-vs-human label-frame offset (issue #113) rests on.

Offline and torch-free: synthetic panos and labels with a planted beta.
"""
import math
import random

import pytest

import label_frame_beta as lfb


def _label(i, key, x, y, era='post179', vouched=False):
    return lfb.Label(uid=f'{key[0]}:{i}', key=key, x=x, y=y, width=16384, height=8192,
                     era=era, vouched=vouched)


def _pano(pitch, roll, dets):
    return lfb.Pano(pitch, roll, 16384, 8192, 'results.jsonl', dets)


# ------------------------------------------------------------------------- geometry

def test_tilt_terms_follow_streetlevels_sign_at_the_four_bearings():
    # ahead (x = 0.5) the tilt is the pitch; to the right (x = 0.75) it is the roll
    assert lfb.tilt_terms(3.0, 1.0, 0.5) == pytest.approx((3.0, 0.0))
    assert lfb.tilt_terms(3.0, 1.0, 0.75) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert lfb.tilt_terms(3.0, 1.0, 0.0) == pytest.approx((-3.0, 0.0), abs=1e-12)
    assert lfb.tilt_terms(3.0, 1.0, 0.25) == pytest.approx((0.0, -1.0), abs=1e-12)


def test_an_unwrapped_roll_is_read_as_the_small_negative_angle_it_is():
    # GSV stores camera_roll unwrapped: 359.4 is -0.6, not a 359-degree roll
    assert lfb.wrap180(359.4) == pytest.approx(-0.6)
    assert lfb.tilt_terms(0.0, 359.4, 0.75)[1] == pytest.approx(-0.6)


def test_xml_tilt_converts_with_the_fitted_convention():
    # tilt direction straight ahead: all of it is pitch, with the fitted minus sign
    assert lfb.xml_pitch_roll(90.0, 90.0, 2.0) == pytest.approx((-2.0, 0.0), abs=1e-12)
    # tilt direction 90 deg right of the pano's yaw: all of it is roll
    assert lfb.xml_pitch_roll(90.0, 180.0, 2.0) == pytest.approx((0.0, -2.0), abs=1e-12)


def test_eras_split_on_the_ps_release_boundaries():
    assert lfb.era_of('2020-12-31T23:59:59Z') == 'legacy'
    assert lfb.era_of('2021-01-01 00:00:00+00:00') == 'mid'
    assert lfb.era_of('2023-03-28T00:00:00Z') == 'mid'
    assert lfb.era_of('2023-03-29T00:00:00Z') == 'post179'
    assert lfb.era_of(None) == 'unknown'


# -------------------------------------------------------------------------- pairing

def test_pairing_takes_the_nearest_detection_inside_the_window_and_at_the_tier():
    key = ('c', 'p')
    panos = {key: _pano(0.0, 0.0, [(0.500, 0.60, 0.9),      # 1.8 deg below the label
                                   (0.500, 0.595, 0.4),     # nearer, but under the tier
                                   (0.520, 0.59, 0.9)])}    # 7.2 deg away in bearing
    pairs, counts = lfb.pair_labels([_label(1, key, 0.5, 0.59)], panos, 0.55, 6.0)
    assert len(pairs) == 1
    # the detection is lower in the image, so its elevation is lower: a negative diff
    assert pairs[0].diff == pytest.approx(-1.8)
    assert pairs[0].confidence == 0.9
    assert counts['no_detection_in_window'] == 0


def test_pairing_wraps_in_bearing_at_the_seam():
    key = ('c', 'p')
    panos = {key: _pano(0.0, 0.0, [(0.998, 0.6, 0.9)])}
    pairs, _ = lfb.pair_labels([_label(1, key, 0.001, 0.6)], panos, 0.55, 6.0)
    assert len(pairs) == 1


def test_a_label_made_on_other_pixels_is_dropped_not_paired():
    key = ('c', 'p')
    lab = _label(1, key, 0.5, 0.6)
    lab.width, lab.height = 13312, 6656
    pairs, counts = lfb.pair_labels([lab], {key: _pano(0, 0, [(0.5, 0.6, 0.9)])}, 0.55, 6.0)
    assert pairs == [] and counts['dims_differ'] == 1


def test_the_window_in_elevation_is_what_lets_a_large_shift_through():
    key = ('c', 'p')
    panos = {key: _pano(0.0, 0.0, [(0.5, 0.6 + 5.0 / 180.0, 0.9)])}
    lab = [_label(1, key, 0.5, 0.6)]
    assert lfb.pair_labels(lab, panos, 0.55, 4.0)[0] == []
    assert len(lfb.pair_labels(lab, panos, 0.55, 6.0)[0]) == 1


# ------------------------------------------------------------------------------ fit

def _planted(beta, intercept, n_panos=300, seed=7):
    """Labels whose pano_y is off from the detection's row by beta * T + intercept."""
    rng = random.Random(seed)
    labels, panos = [], {}
    for i in range(n_panos):
        key = ('c', f'p{i}')
        pitch, roll = rng.gauss(0, 2.0), rng.gauss(0, 1.5)
        dets, labs = [], []
        for j in range(2):
            x = rng.random()
            y_det = 0.55 + 0.1 * rng.random()
            tp, tr = lfb.tilt_terms(pitch, roll, x)
            diff = beta * (tp + tr) + intercept + rng.gauss(0, 0.3)
            dets.append((x, y_det, 0.9))
            # diff = el_det - el_human = (y_human - y_det) * 180
            labs.append(_label(i * 10 + j, key, x, y_det + diff / 180.0))
        panos[key] = _pano(pitch, roll, dets)
        labels += labs
    return labels, panos


@pytest.mark.parametrize('beta', [0.0, 0.6, 1.0])
def test_the_fit_recovers_a_planted_beta_on_both_axes(beta):
    labels, panos = _planted(beta, intercept=-0.3)
    pairs, _ = lfb.pair_labels(labels, panos, 0.55, 10.0)
    row = lfb.fit(pairs)
    assert row['beta'] == pytest.approx(beta, abs=0.03)
    assert row['beta_pitch'] == pytest.approx(beta, abs=0.05)
    assert row['beta_roll'] == pytest.approx(beta, abs=0.05)
    assert row['intercept'] == pytest.approx(-0.3, abs=0.05)
    assert abs(row['beta'] - beta) < 4 * row['beta_se']


def test_a_mirrored_pose_sign_reads_as_a_negative_beta():
    # the sign is part of the claim: feeding the pose with the wrong sign must not
    # come back looking like the right answer
    labels, panos = _planted(1.0, intercept=0.0)
    for p in panos.values():
        p.pitch, p.roll = -p.pitch, -p.roll
    pairs, _ = lfb.pair_labels(labels, panos, 0.55, 10.0)
    assert lfb.fit(pairs)['beta'] == pytest.approx(-1.0, abs=0.03)


def test_clustering_by_pano_widens_the_standard_error_when_labels_share_a_pose():
    # ten labels at ONE bearing of each pano share one T and one pano-level error: they
    # are close to one observation, and the clustered SE has to say so
    rng = random.Random(3)
    X, y, cl = [], [], []
    for i in range(200):
        t, shared = rng.gauss(0, 2.0), rng.gauss(0, 0.5)
        for _ in range(10):
            X.append([1.0, t])
            y.append(0.9 * t + shared + rng.gauss(0, 0.05))
            cl.append(i)
    _, se_clustered = lfb.ols_clustered(X, y, cl)
    _, se_naive = lfb.ols_clustered(X, y, list(range(len(y))))
    assert se_clustered[1] > 2.0 * se_naive[1]


def test_too_few_pairs_give_no_fit_rather_than_a_number():
    labels, panos = _planted(1.0, 0.0, n_panos=5)
    pairs, _ = lfb.pair_labels(labels, panos, 0.55, 10.0)
    assert len(pairs) < lfb.MIN_FIT and lfb.fit(pairs) is None


def test_the_solver_refuses_a_singular_design():
    with pytest.raises(ValueError):
        lfb.ols_clustered([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [1.0, 2.0, 3.0], [0, 1, 2])


def test_rig_borne_detections_never_enter_a_pair():
    # y = 0.9 is 72 deg below the horizon: the camera vehicle, masked in production
    assert lfb._masked([(0.5, 0.9, 0.99), (0.5, 0.6, 0.9)]) == [(0.5, 0.6, 0.9)]
    assert math.isclose(lfb.WINDOW_BEARING_DEG, 3.0)


# ------------------------------------------------------------------------- pool mode

POSE_COLS = ['city', 'pano_id', 'jpg_present', 'jpg_width', 'jpg_height', 'npz_present',
             'pitch_deg', 'roll_deg', 'xml_present', 'xml_pano_yaw_deg', 'xml_tilt_yaw_deg',
             'xml_tilt_pitch_deg']
POOL_COLS = ['label_uid', 'city', 'pano_id', 'label_type', 'pano_source', 'pano_x', 'pano_y',
             'pano_width', 'pano_height', 'era']


def _gz_csv(path, cols, rows):
    import csv
    import gzip
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def test_the_xml_pose_wins_over_the_modern_one_and_a_missing_pose_is_none():
    both = dict(xml_present='1', xml_pano_yaw_deg='90', xml_tilt_yaw_deg='90',
                xml_tilt_pitch_deg='2', npz_present='1', pitch_deg='5', roll_deg='5')
    pitch, roll, source = lfb.pose_from_scan_row(both)
    assert source == 'xml'
    assert pitch == pytest.approx(-2.0) and roll == pytest.approx(0.0, abs=1e-12)
    npz_only = dict(xml_present='0', xml_pano_yaw_deg='', xml_tilt_yaw_deg='',
                    xml_tilt_pitch_deg='', npz_present='1', pitch_deg='5', roll_deg='-1')
    assert lfb.pose_from_scan_row(npz_only) == (5.0, -1.0, 'npz')
    none = dict(xml_present='0', xml_pano_yaw_deg='', xml_tilt_yaw_deg='',
                xml_tilt_pitch_deg='', npz_present='0', pitch_deg='', roll_deg='')
    assert lfb.pose_from_scan_row(none) is None


def test_load_pool_keeps_only_labels_with_a_jpeg_a_pose_matching_dims_and_a_detection(tmp_path):
    import json
    pose = tmp_path / 'pose.csv.gz'
    pool = tmp_path / 'pool.csv.gz'
    dets = tmp_path / 'dets.jsonl'
    posed = dict(jpg_present='1', jpg_width='16384', jpg_height='8192', npz_present='1',
                 pitch_deg='1', roll_deg='2', xml_present='0')
    _gz_csv(pose, POSE_COLS, [
        dict(city='c', pano_id='ok', **posed),
        dict(city='c', pano_id='nojpg', **{**posed, 'jpg_present': '0'}),
        dict(city='c', pano_id='nopose', **{**posed, 'npz_present': '0'}),
        dict(city='c', pano_id='undetected', **posed),
    ])
    lab = dict(label_type='CurbRamp', pano_source='gsv', pano_x='8192', pano_y='4915',
               pano_width='16384', pano_height='8192', era='mid')
    _gz_csv(pool, POOL_COLS, [
        dict(label_uid='c:1', city='c', pano_id='ok', **lab),
        dict(label_uid='c:2', city='c', pano_id='ok', **{**lab, 'pano_width': '13312',
                                                          'pano_height': '6656'}),
        dict(label_uid='c:3', city='c', pano_id='nojpg', **lab),
        dict(label_uid='c:4', city='c', pano_id='nopose', **lab),
        dict(label_uid='c:5', city='c', pano_id='undetected', **lab),
        dict(label_uid='c:6', city='c', pano_id='noscan', **lab),
        dict(label_uid='c:7', city='c', pano_id='ok', **{**lab, 'label_type': 'Obstacle'}),
    ])
    dets.write_text(json.dumps({'city': 'c', 'pano_id': 'ok', 'native_w': 16384,
                                'native_h': 8192, 'detections': [[0.5, 0.6, 0.9]]}) + '\n')
    labels, panos, counts = lfb.load_pool(pool, pose, dets, 'CurbRamp')
    assert [lab.uid for lab in labels] == ['c:1']
    assert list(panos) == [('c', 'ok')] and panos[('c', 'ok')].pose_source == 'npz'
    assert counts == {'other_type_or_source': 1, 'no_pose_row': 1, 'no_jpg': 1,
                      'no_pose': 1, 'dims_differ': 1, 'not_detected': 1}
    pairs, _ = lfb.pair_labels(labels, panos, 0.55, 6.0)
    assert len(pairs) == 1 and pairs[0].era == 'mid' and pairs[0].pose_source == 'npz'


def test_load_pool_refuses_a_detection_made_on_other_pixels_than_the_scan_saw(tmp_path):
    import json
    pose, pool, dets = tmp_path / 'pose.csv.gz', tmp_path / 'pool.csv.gz', tmp_path / 'd.jsonl'
    _gz_csv(pose, POSE_COLS, [dict(city='c', pano_id='p', jpg_present='1', jpg_width='16384',
                                   jpg_height='8192', npz_present='1', pitch_deg='1',
                                   roll_deg='2', xml_present='0')])
    _gz_csv(pool, POOL_COLS, [])
    dets.write_text(json.dumps({'city': 'c', 'pano_id': 'p', 'native_w': 13312,
                                'native_h': 6656, 'detections': []}) + '\n')
    with pytest.raises(SystemExit, match='different pixels'):
        lfb.load_pool(pool, pose, dets, 'CurbRamp')
