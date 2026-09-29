"""mined_precision_compare: reads mined_precision's candidates.csv back and re-tallies
it per city, per range band and pooled, without changing any number."""
import math

import pytest

import mined_precision as mp
import mined_precision_compare as mpc


def _cand(city, site, pano, rng, bucket, kind=''):
    return mp.Candidate(city=city, site_id=site, pano_id=pano, range_m=rng,
                        bearing_deg=0.0, x_norm=0.5, y_norm=0.6, n_op_panos=3,
                        best_conf=0.9, bucket=bucket, nearest_gt_kind=kind,
                        nearest_gt_m=math.inf if not kind else 3.0,
                        within_match=bool(kind))


def _write(root, arm, city, cands):
    mp.write_outputs(root / arm / city, 'report', cands)


def test_round_trip_and_tallies(tmp_path):
    # site_id 1 in BOTH cities: a per-run serial, so the key must include the city
    a = [_cand('a', 1, 'p1', 5.0, 'tp', 'missed'),
         _cand('a', 2, 'p1', 9.0, 'fp', 'det'),          # placed: a real ramp nearby
         _cand('a', 3, 'p2', 14.0, 'already_detected', 'det')]
    b = [_cand('b', 1, 'q1', 11.0, 'fp'),                 # no GT near: not "placed"
         _cand('b', 2, 'q2', 7.0, 'false_det_nearby', 'false_det')]
    _write(tmp_path, 'x', 'a', a)
    _write(tmp_path, 'x', 'b', b)
    back = mpc.read_candidates(tmp_path / 'x' / 'a' / 'candidates.csv')
    assert [(c.site_id, c.bucket, c.range_m) for c in back] == \
        [(c.site_id, c.bucket, c.range_m) for c in a]
    rows = mpc.compare([('x', tmp_path / 'x')], ['a', 'b'], mapillary=['b'])
    get = {(r['group'], r['stratum']): r for r in rows}
    head = get[('pooled', '<= 15 m')]
    # fp = fp + false_det_nearby; hard = tp/(tp+fp); all = (tp+already)/(... )
    assert (head['tp'], head['fp'], head['already_detected']) == (1, 3, 1)
    assert head['p_hard'] == pytest.approx(1 / 4)
    assert head['p_all'] == pytest.approx(2 / 5)
    assert head['fp_placed'] == 1
    # bands partition the <= 15 m set
    bands = sum(get[('pooled', s)]['candidates'] for s in ('0-8 m', '8-12 m', '12-18 m'))
    assert bands == head['candidates'] == 5
    assert get[('pooled', '<= 10 m')]['candidates'] == 3
    # the GSV pool leaves out the city named as Mapillary
    assert get[('pooled GSV', '<= 15 m')]['candidates'] == 3
    # numbers match mined_precision's own precision_row on the same rows
    ref = mp.precision_row('<= 15 m', a)
    assert get[('a', '<= 15 m')]['p_all'] == ref['p_all']
    assert '| a | x | <= 15 m |' in mpc.to_markdown(rows)


def test_duplicate_key_refused(tmp_path):
    dup = [_cand('a', 1, 'p1', 5.0, 'tp', 'missed'), _cand('a', 1, 'p1', 6.0, 'fp')]
    _write(tmp_path, 'x', 'a', dup)
    with pytest.raises(ValueError, match='not unique'):
        mpc.read_candidates(tmp_path / 'x' / 'a' / 'candidates.csv')


def test_paired_transitions_and_sign_test(tmp_path):
    base = [_cand('a', 1, 'p1', 5.0, 'fp', 'det'), _cand('a', 2, 'p2', 9.0, 'tp', 'missed'),
            _cand('a', 3, 'p3', 12.0, 'unsure', 'unsure')]
    arm = [_cand('a', 1, 'p1', 5.0, 'tp', 'missed'), _cand('a', 2, 'p2', 9.0, 'fp'),
           _cand('a', 3, 'p3', 12.0, 'already_detected', 'det')]
    _write(tmp_path, 'flat', 'a', base)
    _write(tmp_path, 'img', 'a', arm)
    rows = mpc.paired(('flat', tmp_path / 'flat'), ('img', tmp_path / 'img'), ['a'])
    r = next(x for x in rows if x['group'] == 'a')
    assert (r['fixed'], r['broken'], r['changed']) == (1, 1, 3)   # unsure -> right is neither
    assert (r['tp_gained'], r['tp_lost']) == (1, 1)
    assert r['sign_p'] == 1.0
    assert mpc.sign_test_p(0, 6) == pytest.approx(2 / 64)
    # different candidate sets are not paired
    _write(tmp_path, 'other', 'a', base[:2])
    with pytest.raises(ValueError, match='not the same candidates'):
        mpc.paired(('flat', tmp_path / 'flat'), ('other', tmp_path / 'other'), ['a'])


def test_paired_refuses_a_different_fuse_with_coinciding_keys(tmp_path):
    """site_id is a per-run serial: equal key sets can still be two different fuses. A
    placement arm keeps the flat run's range_m, so a range mismatch is refused."""
    base = [_cand('a', 1, 'p1', 5.0, 'fp', 'det')]
    other = [_cand('a', 1, 'p1', 6.5, 'tp', 'missed')]
    _write(tmp_path, 'flat', 'a', base)
    _write(tmp_path, 'img', 'a', other)
    with pytest.raises(ValueError, match='range_m'):
        mpc.paired(('flat', tmp_path / 'flat'), ('img', tmp_path / 'img'), ['a'])


def test_outputs_are_lf_on_every_platform(tmp_path):
    mp.write_outputs(tmp_path, 'line one\nline two', [_cand('a', 1, 'p1', 5.0, 'tp', 'missed')])
    for name in ('candidates.csv', 'report.md'):
        assert b'\r' not in (tmp_path / name).read_bytes()
