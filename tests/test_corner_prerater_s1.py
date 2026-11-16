"""scripts/corner_prerater_s1.py (issue #185, Stage 1) on synthetic corners: what reaches the
model (rubric verbatim, no filename, no inventory, no verdict), the call ids per arm, the
nearest-crop rule, the unit-level catch, the flag rate with no_answer counted as a flag, and
the pre-registered kill rule. CPU only, no network, no model."""
import pytest

import corner_prerater_s1 as s1


def req(key, blind, crops=((5.0, '2023-05'), (3.0, '2021-01')), final=None, gallery='g243',
        unit=None):
    unit = unit or key.split(':', 1)[1].split('#')[0]
    return {'key': key, 'gallery': gallery, 'unit': unit, 'corner': int(key.split('#')[1]),
            'rubric': 'RUBRIC TEXT v1\n', 'aerial': {'path': 'D:/x/aerial/secret_name.png'},
            'crops': [{'i': i + 1, 'path': f'D:/x/crops/pano_{i}.jpg', 'dist_m': d, 'date': dt}
                      for i, (d, dt) in enumerate(crops)],
            'nearest': s1.nearest_view([{'dist_m': d} for d, _ in crops]) + 1,
            'truth': {'blind': blind, 'blind_kind': None, 'final': final or blind,
                      'used_live_imagery': False}}


def texts(parts):
    return ' '.join(p for k, p in parts if k == 'text')


def test_prompt_carries_the_rubric_verbatim_and_no_filename_or_truth():
    q = req('g243:vancouver:res:n1#2', 'present')
    for mode in ('multi', 'noaerial'):
        parts = s1.prompt_parts(q, mode)
        t = texts(parts)
        assert 'RUBRIC TEXT v1\n' in t and 'corner 2' in t
        assert 'secret_name' not in t and 'pano_' not in t and 'present,' in t
        assert 'inventory is not available' in t
    multi = s1.prompt_parts(q, 'multi')
    assert [p for k, p in multi if k == 'image'][0].endswith('secret_name.png')
    assert sum(k == 'image' for k, _ in multi) == 3
    assert sum(k == 'image' for k, _ in s1.prompt_parts(q, 'noaerial')) == 2
    single = s1.prompt_parts(q, 'single', 2)
    assert [p for k, p in single if k == 'image'] == ['D:/x/crops/pano_1.jpg']
    assert 'Crop 1: captured 2021-01, camera 3.0 m' in texts(single)


def test_prompt_digest_is_stable_and_covers_the_schema():
    a = s1.prompt_sha256()
    assert a == s1.prompt_sha256() and len(a) == 64


def test_nearest_view_and_call_keys():
    assert s1.nearest_view([{'dist_m': 4}, {'dist_m': 2}, {'dist_m': 2}]) == 1
    reqs = [req('g243:vancouver:res:n1#0', 'absent'), req('g243:vancouver:res:n2#1', 'absent')]
    assert s1.call_keys(reqs, 'argon_multi_r1') == [r['key'] for r in reqs]
    assert s1.call_keys(reqs, 'flash_single', views='nearest') == [
        'g243:vancouver:res:n1#0@2', 'g243:vancouver:res:n2#1@2']
    assert s1.call_keys(reqs, 'flash_single', views='rest') == [
        'g243:vancouver:res:n1#0@1', 'g243:vancouver:res:n2#1@1']
    assert len(s1.call_keys(reqs, 'argon_single', keys=[reqs[0]['key']])) == 2


def test_smoke_keys_are_seeded():
    reqs = [req(f'g243:vancouver:res:n{i}#0', 'absent') for i in range(50)]
    assert s1.smoke_keys(reqs) == s1.smoke_keys(list(reversed(reqs)))
    assert len(s1.smoke_keys(reqs)) == s1.SMOKE_N


def build(n_absent=20):
    reqs = [req('g243:vancouver:art:n9#0', 'present'), req('g243:vancouver:art:n9#1', 'present'),
            req('g243:vancouver:res:n8#0', 'present'), req('g169:vancouver:res:n7#2', 'present',
                                                            gallery='g169', final='cant_tell')]
    reqs += [req(f'g243:vancouver:res:a{i}#0', 'absent') for i in range(n_absent)]
    return reqs


def test_unit_is_caught_only_when_every_present_corner_is():
    reqs = build()
    ans = {q['key']: 'absent' for q in reqs}
    ans['g243:vancouver:art:n9#0'] = 'present'              # one of two: unit not caught
    ans['g243:vancouver:res:n8#0'] = 'present'
    s = s1.score_arm(reqs, ans)
    assert s['present_blind']['units'] == 3
    assert s['present_blind']['units_caught'] == 1
    assert s['present_blind']['corners_caught'] == 2
    assert s['present_final']['units'] == 2                 # the g169 corner left the final set


def test_flag_rate_counts_present_cant_tell_and_no_answer():
    reqs = build(n_absent=20)
    ans = {q['key']: 'absent' for q in reqs}
    ans['g243:vancouver:res:a0#0'] = 'present'
    ans['g243:vancouver:res:a1#0'] = 'cant_tell'
    ans['g243:vancouver:res:a2#0'] = 'no_answer'
    s = s1.score_arm(reqs, ans)
    assert s['flagged'] == 3 and s['flag_rate'] == pytest.approx(3 / 20)
    assert (s['flag_present'], s['flag_cant_tell'], s['flag_no_answer']) == (1, 1, 1)


def scores_for(units_caught, flag):
    return {'called': 10, 'corners': 10, 'flag_rate': flag,
            'present_blind': {'units_caught': units_caught, 'units': 6}}


def test_kill_rule():
    all_fail = {a: scores_for(2, 0.05) for a in s1.KILL_ARMS}
    assert s1.kill_verdict(all_fail)[0].startswith('KILL')
    one_ok = dict(all_fail, flash_single=scores_for(3, 0.10))   # 10% exactly is not > 10%
    v, per = s1.kill_verdict(one_ok)
    assert v.startswith('NO KILL') and per['flash_single'] == 'pass'
    flagged = dict(all_fail, flash_single=scores_for(6, 0.101))
    assert s1.kill_verdict(flagged)[0].startswith('KILL')
    partial = dict(all_fail)
    partial['argon_single'] = dict(scores_for(6, 0.0), called=9)
    assert s1.kill_verdict(partial)[0].startswith('NOT DECIDED')


def test_arm_answers_reads_the_nearest_crop_for_single_view():
    reqs = [req('g243:vancouver:res:n1#0', 'absent')]
    resp = [{'call_id': 'g243:vancouver:res:n1#0@1', 'verdict': 'present'},
            {'call_id': 'g243:vancouver:res:n1#0@2', 'verdict': 'absent'}]
    assert s1.arm_answers(reqs, resp, 'argon_single') == {'g243:vancouver:res:n1#0': 'absent'}
    resp = [{'call_id': 'g243:vancouver:res:n1#0@2', 'verdict': None}]
    assert s1.arm_answers(reqs, resp, 'argon_single') == {'g243:vancouver:res:n1#0': 'no_answer'}


def test_kappa():
    assert s1.kappa([('a', 'a'), ('b', 'b')]) == pytest.approx(1.0)
    assert s1.kappa([]) is None


def test_to_img_maps_the_bbox_corners():
    a = {'zoom': 20, 'px': 512}
    x0, y0 = s1.world_px(45.6, -122.6, 20)
    a['world_px'] = {'x0': x0, 'x1': x0 + 512, 'y0': y0, 'y1': y0 + 512}
    assert s1.to_img(a, 45.6, -122.6) == pytest.approx((0, 0))
