"""mined_placement_attribution (POST HOC, RampNet#158 phase-2 review): a placed target
that lands on a detection of ANOTHER fused site is labelled as such, so a "fix" under
the 5 m world match can be told apart from a transfer onto a neighbouring ramp."""
import fuse_sites as fs
import mined_precision as mp
import mined_placement_attribution as mpa
from test_fuse_sites import make_pano
from test_eval_sites import _entry, _xy, _bundle_ops


def _two_ramp_scene():
    """Ramp A at (0, 0) and ramp B at (12, 5), 13 m apart. g2 stands between them and
    detects only B; n2 (A's member nearest g2, so g2's source view) detects BOTH."""
    panos = [
        make_pano('n1', 0, -10, [(0, 0, 0.9)]),
        make_pano('n2', 10, 0, [(0, 0, 0.8), (12, 5, 0.7)]),
        make_pano('n3', -10, 0, [(0, 0, 0.95)]),
        make_pano('g1', 0, 8, [], heading_deg=180.0),
        make_pano('g2', 5, 5, [(12, 5, 0.9)], heading_deg=225.0),
        make_pano('m1', 12, 15, [(12, 5, 0.9)]),
        make_pano('m2', 22, 5, [(12, 5, 0.9)]),
    ]
    verdicts = {'g1': _entry(missed=[_xy(0, 8, 180.0, 0, 0)], no_missed=False),
                'g2': _entry(dets=[True], no_missed=True)}
    return panos, verdicts


def test_transfer_onto_a_neighbouring_sites_detection_is_other_multi():
    panos, verdicts = _two_ramp_scene()
    ops = _bundle_ops(panos, verdicts)
    params = fs.FuseParams()
    _, flat = mp.run_city(verdicts, ops, panos, params, city='s')
    g2_a = next(c for c in flat if c.pano_id == 'g2')
    assert g2_a.bucket == 'fp'                    # A's flat point: nothing within 5 m
    g2 = next(p for p in panos if p.pano_id == 'g2')
    _, bx, by, _ = g2.detections[0]
    # the "arm" carries A's click onto g2's detection of B; everything else falls back
    place = {('s', c.site_id, c.pano_id): None for c in flat}
    place[('s', g2_a.site_id, 'g2')] = (bx, by)
    rows, every = mpa.run_city('s', verdicts, ops, panos, params, [('img', place)],
                               15.0, 5.0)
    row = next(r for r in rows if r['pano_id'] == 'g2' and r['site_id'] == g2_a.site_id)
    assert (row['flat_bucket'], row['arm_bucket']) == ('fp', 'already_detected')
    assert row['transition'] == 'fixed'           # a "fix" under the rubric...
    assert row['category'] == 'other_multi'       # ...onto ANOTHER multi-pano site
    assert row['landed_site_panos'] >= 3
    assert row['src_pano'] == 'n2' and row['src_in_landed'] == 1
    assert row['site_to_landed_m'] > 12.0
    # own_site is impossible by construction: a candidate pano has no member in the site
    assert all(r['category'] != 'own_site' for r in every)
    strict = {(r['group'], r['arm']): r for r in mpa.strict_retally(every, ['s'])}
    s = strict[('s', 'img')]
    assert (s['ad'], s['ad_other']) == (1, 1)
    assert s['strict'][0] < s['all'][0]


def test_transition_names():
    assert mpa.transition('fp', 'already_detected') == 'fixed'
    assert mpa.transition('tp', 'false_det_nearby') == 'broken'
    assert mpa.transition('tp', 'already_detected') == 'tp_to_ad'
    assert mpa.transition('already_detected', 'tp') == 'ad_to_tp'
    assert mpa.transition('unsure', 'fp') == 'other_change'
    assert mpa.transition('fp', 'fp') is None


def _g2_arm(flat):
    """The test arm: carries A's click onto g2's detection of B; everything else falls back."""
    g2_a = next(c for c in flat if c.pano_id == 'g2')
    place = {('s', c.site_id, c.pano_id): None for c in flat}
    return g2_a, place


def test_verify_refuses_a_bucket_that_differs_from_the_committed_run(tmp_path, monkeypatch):
    """run_city(verify_root=...) re-reads each committed candidates.csv and refuses on any
    bucket mismatch. Planting a no-op _verify must make this test fail."""
    import pytest
    panos, verdicts = _two_ramp_scene()
    ops = _bundle_ops(panos, verdicts)
    params = fs.FuseParams()
    _, flat = mp.run_city(verdicts, ops, panos, params, city='s')
    g2_a, place = _g2_arm(flat)
    g2 = next(p for p in panos if p.pano_id == 'g2')
    place[('s', g2_a.site_id, 'g2')] = g2.detections[0][1:3]
    arms = [('img', place)]
    # write the "committed" runs from the same code path, so they verify cleanly
    seen = []
    with monkeypatch.context() as m:
        m.setattr(mpa, '_verify', lambda path, cands: seen.append((path, cands)))
        mpa.run_city('s', verdicts, ops, panos, params, arms, 15.0, 5.0,
                     verify_root=tmp_path)
    assert [p.parent.parent.name for p, _ in seen] == ['perrig_perpano', 'place_img']
    for path, cands in seen:
        mp.write_outputs(path.parent, '', cands)
    mpa.run_city('s', verdicts, ops, panos, params, arms, 15.0, 5.0, verify_root=tmp_path)
    # flip one committed bucket in the arm's run: refused
    arm_csv = tmp_path / 'place_img' / 's' / 'candidates.csv'
    text = arm_csv.read_text(encoding='utf-8')
    assert text.count(',already_detected,') == 1
    arm_csv.write_text(text.replace(',already_detected,', ',fp,'), encoding='utf-8',
                       newline='\n')
    with pytest.raises(AssertionError, match='recomputed buckets differ'):
        mpa.run_city('s', verdicts, ops, panos, params, arms, 15.0, 5.0,
                     verify_root=tmp_path)


def test_writing_into_frozen_requires_verify(tmp_path):
    import pytest
    frozen = tmp_path / 'frozen'
    with pytest.raises(SystemExit, match='--verify'):
        mpa.require_verify((None, frozen / 'attribution_phase2.md'), None, frozen)
    with pytest.raises(SystemExit, match='--verify'):
        mpa.require_verify((frozen / 'sub' / 'a.csv', None), None, frozen)
    mpa.require_verify((frozen / 'a.csv',), frozen, frozen)      # verified: allowed
    mpa.require_verify((tmp_path / 'scratch.csv', None), None, frozen)  # elsewhere: allowed
    assert mpa.FROZEN_DIR.parts[-5:] == ('docs', 'figures', 'mined-precision', 'data',
                                         'frozen')
