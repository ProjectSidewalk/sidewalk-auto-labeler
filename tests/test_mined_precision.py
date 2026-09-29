"""mined_precision: the RampNet#158 step-1 check, end to end on synthetic geometry.

One ramp, seen by enough panos to make a strong site; then judged panos that stood
next to it, one per outcome the check has to tell apart — including the three the
first version of this file never exercised (`already_detected`, `false_det_nearby`,
`unadjudicable`), which is where the two denominators differ."""
import csv
import math

import pytest

import fuse_sites as fs
import mined_precision as mp
from test_fuse_sites import make_pano
from test_eval_sites import _entry, _xy, _bundle_ops


def _scene():
    panos = [
        # ramp A at (0, 0): three neighbours detect it -> a strong site
        make_pano('n1', 0, -10, [(0, 0, 0.9)]),
        make_pano('n2', 10, 0, [(0, 0, 0.8)]),
        make_pano('n3', -10, 0, [(0, 0, 0.95)]),
        # g1 stood 8 m north, detected nothing, reviewer marked the miss -> tp
        make_pano('g1', 0, 8, [], heading_deg=180.0),
        # g2 stood 5 m NE, detected nothing, reviewer attested no misses -> fp
        make_pano('g2', 5, 5, [], heading_deg=225.0),
        # g3 stood 6 m south and DID detect it (verdict true) -> a member, not a candidate
        make_pano('g3', 0, -6, [(0, 0, 0.7)]),
        # g4 stood 40 m away: outside every radius, never a candidate
        make_pano('g4', 40, 0, [], heading_deg=270.0),
    ]
    verdicts = {
        'g1': _entry(missed=[_xy(0, 8, 180.0, 0, 0)], no_missed=False),
        'g2': _entry(no_missed=True),
        'g3': _entry(dets=[True]),
        'g4': _entry(no_missed=True),
    }
    return panos, verdicts


def _run(panos, verdicts, params=None, city='synthetic', **kw):
    return mp.run_city(verdicts, _bundle_ops(panos, verdicts), panos,
                       params or fs.FuseParams(), city=city, **kw)


# A tight association cap forces the site SPLIT that the `already_detected` and
# `false_det_nearby` buckets describe: a pano that responded to the ramp, but whose
# detection landed in a different site, so it is mined as a non-member anyway. The
# base scene is unaffected (n1-n3/g3 all raycast to exactly (0, 0), distance 0).
SPLIT = fs.FuseParams(max_match_m=2.0)


def test_buckets_and_headline():
    panos, verdicts = _scene()
    result, cands = _run(panos, verdicts, radii_m=(10.0, 15.0))
    assert result['n_strong_sites'] == 1
    assert result['n_judged_panos'] == 4
    by_pano = {c.pano_id: c for c in cands}
    assert set(by_pano) == {'g1', 'g2'}          # g3 is a member, g4 out of range
    assert by_pano['g1'].bucket == 'tp'
    assert by_pano['g1'].nearest_gt_kind == 'missed'
    assert by_pano['g1'].within_match
    assert by_pano['g2'].bucket == 'fp'
    assert by_pano['g1'].n_op_panos == 4         # n1, n2, n3, g3
    assert by_pano['g1'].best_conf == 0.95
    # the projection points g1 at the ramp: dead ahead (heading 180 looks at it), 8 m out
    assert abs(by_pano['g1'].x_norm - 0.5) < 1e-6
    assert abs(by_pano['g1'].range_m - 8.0) < 0.05
    h = result['headline']
    assert (h['tp'], h['fp'], h['p_hard']) == (1, 1, 0.5)
    assert 'add the visibility test' in result['reading']['hard']
    # cumulative-by-distance strata: g2 (7.1 m) and g1 (8 m) both inside 10 m
    assert [r['candidates'] for r in result['strata']['radius']] == [2, 2]


def test_already_detected_is_a_correct_label_not_a_miss():
    """The bucket the two denominators disagree on: a pano that detects the ramp but
    lands in a different site still gets mined, and the label it gets is right."""
    panos, verdicts = _scene()
    # g5 responds to ramp A, but 3 m off — far enough to become its own site under
    # SPLIT, close enough that its verdict-true GT point still adjudicates site A.
    panos.append(make_pano('g5', 0, 12, [(0.0, 3.0, 0.9)], heading_deg=180.0))
    verdicts['g5'] = _entry(dets=[True], no_missed=True)
    result, cands = _run(panos, verdicts, params=SPLIT)
    g5 = next(c for c in cands if c.pano_id == 'g5')
    assert g5.bucket == 'already_detected'
    assert g5.nearest_gt_kind == 'det'
    h = result['headline']
    # hard-only drops it from both sides; all-mined counts it as a correct label
    assert (h['p_hard'], h['n_hard']) == (0.5, 2)
    assert (h['p_all'], h['n_all']) == (pytest.approx(2 / 3), 3)


def test_rejected_detection_counts_against_and_unattested_is_unadjudicable():
    panos, verdicts = _scene()
    # g6 fired 3 m off the ramp and the reviewer REJECTED it: the reviewer looked at
    # that spot, in that pano, and said no -> evidence against the site, not neutral.
    # no_missed=False deliberately: the rejected detection adjudicates that spot on
    # its own, so an unattested missed-ramp sweep must NOT downgrade it.
    panos.append(make_pano('g6', -12, 0, [(-3.0, 0.0, 0.9)], heading_deg=90.0))
    verdicts['g6'] = _entry(dets=[False], no_missed=False)
    # g7 stood next to the site but its missed-ramp check was never attested
    panos.append(make_pano('g7', 0, 6, [], heading_deg=180.0))
    verdicts['g7'] = _entry(no_missed=False)
    result, cands = _run(panos, verdicts, params=SPLIT)
    by_pano = {c.pano_id: c for c in cands}
    assert by_pano['g6'].bucket == 'false_det_nearby'
    assert by_pano['g6'].nearest_gt_kind == 'false_det'
    assert by_pano['g7'].bucket == 'unadjudicable'
    h = result['headline']
    # the rejected detection is a false positive under BOTH denominators...
    assert h['fp'] == 2 and h['fp_rejected_det'] == 1
    assert h['p_hard'] == pytest.approx(1 / 3)
    # ...and the unattested pano is in neither
    assert h['n_hard'] == 3


def test_nearest_gt_is_recorded_even_beyond_the_match_radius():
    """An fp keeps the distance to the nearest missed mark — the localization
    diagnostic the first version blanked out."""
    panos, verdicts = _scene()
    # g1's mark is real but its raycast lands 3 m off the site — exactly the
    # localization error this whole check is about
    verdicts['g1'] = _entry(missed=[_xy(0, 8, 180.0, 3, 0)], no_missed=False)
    tp_at_5m = next(c for c in _run(panos, verdicts)[1] if c.pano_id == 'g1')
    assert tp_at_5m.bucket == 'tp' and tp_at_5m.within_match

    g1 = next(c for c in _run(panos, verdicts, match_m=1.0)[1] if c.pano_id == 'g1')
    assert g1.bucket == 'fp'                      # too far to adjudicate
    assert not g1.within_match
    assert g1.nearest_gt_kind == 'missed'         # ...but still recorded
    assert g1.nearest_gt_m == pytest.approx(3.0, abs=0.2)


def test_radius_beyond_the_raycast_range_is_refused():
    panos, verdicts = _scene()
    with pytest.raises(ValueError, match='ground-raycast range'):
        _run(panos, verdicts, radii_m=(10.0, 30.0))


def test_range_buckets_account_for_every_candidate():
    panos, verdicts = _scene()
    result, cands = _run(panos, verdicts)
    assert sum(r['candidates'] for r in result['strata']['range']) == len(cands)


def test_yield_counts_non_member_pairs_over_the_whole_run():
    panos, verdicts = _scene()
    result, _ = _run(panos, verdicts, radii_m=(10.0,))
    y = result['yield']
    # site A has 4 member panos (n1-n3, g3); of the other three, g1 (8 m) and
    # g2 (7.1 m) are inside 10 m and g4 (40 m) is not
    assert y['counts'][10.0] == 2
    assert y['n_operational_detections'] == 4


def test_subthreshold_only_members_are_excluded_and_counted():
    panos, verdicts = _scene()
    # g8 responds to the ramp, but below the operational floor: the model did fire,
    # so it is neither a miss nor a training positive
    panos.append(make_pano('g8', 0, 9, [(0, 0, 0.2)], heading_deg=180.0))
    verdicts['g8'] = _entry(no_missed=True)
    result, cands = _run(panos, verdicts)
    assert result['excluded_subfloor_members'] == 1
    assert 'g8' not in {c.pano_id for c in cands}


def test_rule_reading_names_the_radius_and_a_straddling_interval():
    decisive = mp.rule_reading(0.9, 0.85, 0.95, 15.0)
    assert 'build the miner' in decisive and '<= 15 m' in decisive
    assert 'spans' not in decisive
    straddles = mp.rule_reading(0.70, 0.40, 0.89, 10.0)
    assert 'drop/visibility/build bands' in straddles
    assert 'not decisive' in straddles
    assert mp.rule_reading(None, 0.0, 1.0, 15.0) == 'no adjudicable candidates'


def test_pooling_keys_on_city_not_the_per_run_site_serial():
    """fuse_sites hands every run its own site_id serial, so two cities collide on
    it. Pooling must key on (city, site_id, pano_id) - the same composite rule a
    Project Sidewalk label_id needs."""
    panos, verdicts = _scene()
    a = _run(panos, verdicts, city='alpha')
    b = _run(panos, verdicts, city='beta')
    # identical scenes: the site ids ARE the same integers in both cities
    assert {c.site_id for c in a[1]} == {c.site_id for c in b[1]}
    pooled, cands = mp.pool_cities([a, b], (10.0, 15.0), 3, 5.0)
    assert len(cands) == len(a[1]) + len(b[1]) == 4
    assert {c.city for c in cands} == {'alpha', 'beta'}
    h, ha = pooled['headline'], a[0]['headline']
    assert (h['tp'], h['fp']) == (2 * ha['tp'], 2 * ha['fp'])
    assert h['p_hard'] == ha['p_hard']          # same ratio, tighter interval
    assert h['hard_hi'] - h['hard_lo'] < ha['hard_hi'] - ha['hard_lo']
    assert pooled['n_judged_panos'] == 2 * a[0]['n_judged_panos']
    assert pooled['yield']['counts'][10.0] == 2 * a[0]['yield']['counts'][10.0]
    # ...and pooling the SAME city twice is the mistake the key exists to catch
    with pytest.raises(AssertionError, match='duplicate'):
        mp.pool_cities([a, a], (10.0, 15.0), 3, 5.0)


def test_csv_has_one_row_per_candidate(tmp_path):
    panos, verdicts = _scene()
    result, cands = _run(panos, verdicts)
    mp.write_outputs(tmp_path, mp.format_report('synthetic', result), cands)
    with open(tmp_path / 'candidates.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(cands) == 2
    assert list(rows[0])[0] == 'city'            # first field, part of the row key
    assert {r['city'] for r in rows} == {'synthetic'}
    assert {r['bucket'] for r in rows} == {'tp', 'fp'}
    assert {r['within_match'] for r in rows} == {'1', '0'}
    report = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert 'hard-only' in report and 'all-mined' in report


# --------------------------------------------------------------------------- #
# RampNet#158 phase 2: the source rule and external (image-based) placement
# --------------------------------------------------------------------------- #


def test_source_rule_picks_the_nearest_member_camera_then_confidence(tmp_path):
    panos, verdicts = _scene()
    result, cands = _run(panos, verdicts, emit_sources=True)
    src = {s['pano_id']: s for s in result['sources']}
    assert set(src) == {c.pano_id for c in cands}
    # g1 stands at (0, 8). Member cameras: n1 18 m, n2 and n3 both ~12.8 m, g3 14 m.
    # n2 / n3 tie on distance, so the higher confidence (n3, 0.95) wins.
    assert src['g1']['src_pano'] == 'n3'
    assert src['g1']['baseline_m'] == pytest.approx(math.hypot(10, 8), abs=0.05)
    # nothing GT-derived travels with a source row
    assert set(src['g1']) == set(mp.SOURCE_FIELDS)
    mp.write_outputs(tmp_path, 'r', cands, result['sources'])
    with open(tmp_path / 'sources.csv', newline='', encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == len(cands)


def test_source_rule_never_uses_the_target_or_a_subfloor_member():
    panos, verdicts = _scene()
    # g9 is 1 m from g1's camera but only responds below the operational floor
    panos.append(make_pano('g9', 1, 8, [(0, 0, 0.2)], heading_deg=180.0))
    result, _ = _run(panos, verdicts, emit_sources=True)
    assert next(s for s in result['sources'] if s['pano_id'] == 'g1')['src_pano'] == 'n3'


def test_placement_is_adjudicated_where_it_lands_and_paired():
    panos, verdicts = _scene()
    _, flat = _run(panos, verdicts)
    by = {c.pano_id: c for c in flat}
    g1, g2 = by['g1'], by['g2']
    assert g1.bucket == 'tp'
    # g1: the arm puts the target 90 deg off, on the ground 8 m to one side -> fp.
    # g2: the arm fell back (None) -> the flat position, same bucket as before.
    place = {('synthetic', g1.site_id, 'g1'): (g1.x_norm + 0.25, g1.y_norm),
             ('synthetic', g2.site_id, 'g2'): None}
    result, cands = _run(panos, verdicts, placement=place)
    by2 = {c.pano_id: c for c in cands}
    assert [(c.site_id, c.pano_id, c.range_m) for c in cands] == \
        [(c.site_id, c.pano_id, c.range_m) for c in flat]      # paired: same candidates
    assert by2['g1'].bucket == 'fp'
    assert by2['g2'].bucket == by['g2'].bucket
    status = {r['pano_id']: r for r in result['placement']}
    assert status['g1']['status'] == 'placed' and status['g1']['shift_m'] > 5.0
    assert status['g2']['status'] == 'fallback' and status['g2']['shift_m'] == 0.0
    # the flat pixel itself reproduces the flat adjudication
    same = {k: (by[k[2]].x_norm, by[k[2]].y_norm) for k in place}
    _, again = _run(panos, verdicts, placement=same)
    assert [c.bucket for c in again] == [c.bucket for c in flat]


def test_placement_must_cover_exactly_the_candidates():
    panos, verdicts = _scene()
    _, flat = _run(panos, verdicts)
    full = {('synthetic', c.site_id, c.pano_id): None for c in flat}
    missing = dict(list(full.items())[:1])
    with pytest.raises(ValueError, match='no row for candidate'):
        _run(panos, verdicts, placement=missing)
    extra = {**full, ('synthetic', 999, 'zz'): None}
    with pytest.raises(ValueError, match='match no candidate'):
        _run(panos, verdicts, placement=extra)


def test_read_placement_refuses_duplicates(tmp_path):
    p = tmp_path / 'p.jsonl'
    p.write_text('{"city": "a", "site_id": 1, "pano_id": "x", "x": 0.5, "y": 0.6}\n'
                 '{"city": "a", "site_id": 1, "pano_id": "y", "x": null, "y": null}\n',
                 encoding='utf-8')
    assert mp.read_placement(p) == {('a', 1, 'x'): (0.5, 0.6), ('a', 1, 'y'): None}
    with open(p, 'a', encoding='utf-8') as f:
        f.write('{"city": "a", "site_id": 1, "pano_id": "x", "x": 0.1, "y": 0.6}\n')
    with pytest.raises(ValueError, match='duplicate'):
        mp.read_placement(p)


def test_not_emitted_candidates_leave_both_denominators():
    """Step 3: a placement row with emit=false means the miner would not emit that
    candidate; it is counted, not scored."""
    panos, verdicts = _scene()
    _, flat = _run(panos, verdicts)
    by = {c.pano_id: c for c in flat}
    place = {('synthetic', by['g1'].site_id, 'g1'): mp.NOT_EMITTED,
             ('synthetic', by['g2'].site_id, 'g2'): None}
    result, cands = _run(panos, verdicts, placement=place)
    assert [c.pano_id for c in cands] == ['g2']
    assert {r['pano_id']: r['status'] for r in result['placement']} == \
        {'g1': 'not_emitted', 'g2': 'fallback'}
    assert 'NOT EMITTED' in mp.placement_line(result)


def test_read_placement_reads_emit_false(tmp_path):
    p = tmp_path / 'p.jsonl'
    p.write_text('{"city": "a", "site_id": 1, "pano_id": "x", "emit": false, "x": null}\n',
                 encoding='utf-8')
    assert mp.read_placement(p) == {('a', 1, 'x'): mp.NOT_EMITTED}


def test_own_site_read_refuses_a_neighbouring_sites_ramp():
    """Step 3's own-site read: a right adjudication whose GT point sits closer to ANOTHER
    fused site counts as other_site (false under both denominators)."""
    panos, verdicts = _scene()
    # ramp B, 4 m east of A, is its own strong site (three views of it)
    panos += [make_pano('m1', 4, -10, [(4, 0, 0.9)]), make_pano('m2', 14, 0, [(4, 0, 0.9)]),
              make_pano('m3', 4, 10, [(4, 0, 0.9)])]
    # g1's reviewer marked B (not A) as its missed ramp: inside A's 5 m match
    verdicts['g1'] = _entry(missed=[_xy(0, 8, 180.0, 4, 0)], no_missed=False)
    plain = {c.pano_id: c for c in _run(panos, verdicts)[1] if c.site_id == 0}
    assert plain['g1'].bucket == 'tp'
    result, cands = _run(panos, verdicts, own_site=True)
    own = {c.pano_id: c for c in cands if c.site_id == 0}
    assert own['g1'].bucket == 'other_site'
    assert result['headline']['other_site'] >= 1
    assert 'other_site' in mp.FP_BUCKETS
