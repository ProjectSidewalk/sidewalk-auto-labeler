"""eval_sites: the stage-3 world-space scoring against RampNet verdicts.

Reuses test_fuse_sites' synthetic-geometry builder: GT panos and neighbors are
placed in ENU, aimed at known ramp positions, and scored end to end through
evaluate_city (fuse -> GT raycast -> merge -> match -> metrics).
"""
import geo
import fuse_sites as fs
import eval_sites as es
from detectors import BENCHMARK_CONFIDENCE
from test_fuse_sites import BASE, make_pano


def _xy(pe, pn, heading, te, tn):
    """Pano-space normalized coordinates from which a pano at (pe, pn) with the
    given heading sees the ground point (te, tn) — for building missed marks."""
    p = make_pano('tmp', pe, pn, [(te, tn, 1.0)], heading_deg=heading)
    _, x, y, _ = p.detections[0]
    return {'x': x, 'y': y}


def _entry(dets=(), missed=(), no_missed=True):
    return {'group': 'random', 'dets': list(dets), 'missed': list(missed),
            'no_missed': no_missed}


def _bundle_ops(panos, gt_ids):
    return {p.pano_id: [(x, y, c) for _, x, y, c in p.detections if c >= BENCHMARK_CONFIDENCE]
            for p in panos if p.pano_id in gt_ids}


def test_all_four_recall_buckets():
    panos = [
        # ramp A (0,0): GT pano g1 detects it -> self_detected
        make_pano('g1', 0, -10, [(0, 0, 0.9)]),
        # ramp B (30,0): g2 misses it, neighbor n1 detects it -> recovered
        make_pano('g2', 30, -10, [], heading_deg=0.0),
        make_pano('n1', 30, 10, [(30, 0, 0.9)]),
        # ramp C (60,0): g3 misses it, two sub-threshold views -> subthreshold_only
        make_pano('g3', 60, -10, [], heading_deg=0.0),
        make_pano('n2', 60, 10, [(60, 0, 0.30)]),
        make_pano('n3', 68, 0, [(60, 0, 0.28)], capture='2022-03'),
        # ramp D (90,0): g4 misses it, nobody else sees it -> unmatched
        make_pano('g4', 90, -10, [], heading_deg=0.0),
    ]
    verdicts = {
        'g1': _entry(dets=[True]),
        'g2': _entry(missed=[_xy(30, -10, 0.0, 30, 0)], no_missed=False),
        'g3': _entry(missed=[_xy(60, -10, 0.0, 60, 0)], no_missed=False),
        'g4': _entry(missed=[_xy(90, -10, 0.0, 90, 0)], no_missed=False),
    }
    r = es.evaluate_city(verdicts, _bundle_ops(panos, verdicts), panos,
                         fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    assert r['buckets'] == {'self_detected': 1, 'recovered_other_view': 1,
                            'subthreshold_only': 1, 'unmatched': 1}
    assert r['n_pool_ramps'] == 4
    assert r['world_recall'] == 0.5      # self + recovered
    assert r['own_view_recall'] == 0.25
    assert r['self_detected_without_site'] == 0
    # precision: only g1's site has a judged member -> TP
    assert (r['precision']['tp'], r['precision']['fp']) == (1, 0)
    assert not r['warnings']

    # (d) promotion calibration: ramp C's site has views at 0.30 and 0.28, ramp D
    # has none, so k(f) drops from 2 to 1 to 0 as the floor crosses them
    cal = r['calibration']
    assert len(cal['profiles']) == 2   # ramps C and D
    k_by_ramp = {tuple(p['panos']): p['k'] for p in cal['profiles']}
    c_k = next(k for panos, k in k_by_ramp.items() if 'g3' in panos)
    assert (c_k[0.10], c_k[0.25], c_k[0.30], c_k[0.35]) == (2, 2, 1, 0)
    promo = {(p['floor'], p['k']): p['recall_if_promoted']
             for p in cal['promotion']}
    assert promo[(0.25, 2)] == 0.75    # base 0.5 + ramp C
    assert promo[(0.30, 2)] == 0.5     # n3's 0.28 view falls below the floor
    assert promo[(0.30, 1)] == 0.75
    # ghost check: g1's true detection has no other-pano support in this city
    g25 = next(g for g in cal['ghost'] if abs(g['floor'] - 0.25) < 1e-9)
    assert g25['true_ge1'] == 0.0 and g25['n_true'] == 1
    assert g25['n_false'] == 0 and g25['false_ge1'] is None

    # (e) the only multi-member site is C's (n2 2024-06, n3 2022-03: 27 months)
    assert r['vintage']['all'] == {'19-36': 1}


def test_dual_ramps_keep_separate_sites():
    # One GT pano marks two missed ramps 3 m apart; a neighbor detects both.
    # The neighbor's two peaks are cannot-linked into two sites, and one-to-one
    # matching must give each GT ramp its own site.
    panos = [make_pano('g1', 0, -10, [], heading_deg=0.0),
             make_pano('n1', 0, 10, [(-1.5, 0, 0.9), (1.5, 0, 0.85)],
                       heading_deg=180.0)]
    verdicts = {'g1': _entry(missed=[_xy(0, -10, 0.0, -1.5, 0),
                                     _xy(0, -10, 0.0, 1.5, 0)],
                             no_missed=False)}
    r = es.evaluate_city(verdicts, _bundle_ops(panos, verdicts), panos,
                         fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    assert r['dual_ramp'] == {'pairs': 1, 'both_matched': 1,
                              'one_matched': 0, 'neither': 0}
    assert r['buckets']['recovered_other_view'] == 2


def test_precision_duplicate_false_and_unsure_semantics():
    panos = [make_pano('g1', 0, -10, [(0, 0, 0.9)]),
             make_pano('g2', 100, -10, [(100, 0, 0.9)]),
             make_pano('g3', 200, -10, [(200, 0, 0.9)])]
    verdicts = {'g1': _entry(dets=['duplicate']),   # proves the ramp -> TP
                'g2': _entry(dets=[False]),         # all-decided-false -> FP
                'g3': _entry(dets=['unsure'])}      # abstains -> excluded
    r = es.evaluate_city(verdicts, _bundle_ops(panos, verdicts), panos,
                         fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    p = r['precision']
    assert (p['tp'], p['fp'], p['unsure_only']) == (1, 1, 1)
    assert p['value'] == 0.5
    # none of those verdicts creates a GT point, so there is no recall pool
    assert r['n_pool_ramps'] == 0 and r['world_recall'] is None


def test_gt_merge_is_cross_pano_only():
    panos = [make_pano('g1', 0, -10, [], heading_deg=0.0),
             make_pano('g2', 0, 10, [], heading_deg=180.0),
             make_pano('g3', 50, -10, [], heading_deg=0.0)]
    verdicts = {
        # same ramp seen by two panos, marks 0.5 m apart -> one ramp
        'g1': _entry(missed=[_xy(0, -10, 0.0, 0, 0)], no_missed=False),
        'g2': _entry(missed=[_xy(0, 10, 180.0, 0.5, 0)], no_missed=False),
        # one pano, two marks 1.5 m apart: reviewer says two ramps -> never merged
        'g3': _entry(missed=[_xy(50, -10, 0.0, 49, 0), _xy(50, -10, 0.0, 50.5, 0)],
                     no_missed=False),
    }
    r = es.evaluate_city(verdicts, _bundle_ops(panos, verdicts), panos,
                         fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    assert r['counts']['gt_ramps'] == 3
    assert r['counts']['cross_pano_merges'] == 1
    assert r['buckets']['unmatched'] == 3


def test_verdicts_map_to_stored_indices_through_subfloor_interleaving():
    # stored detections: [0]=0.3 sub, [1]=0.9 op, [2]=0.4 sub, [3]=0.7 op
    pano = make_pano('p1', 0, -10,
                     [(0, 0, 0.3), (3, 0, 0.9), (-3, 0, 0.4), (6, 0, 0.7)],
                     heading_deg=0.0)
    frame = geo.LocalFrame(*BASE)
    points, op_verdicts, counts, warnings = es.build_gt(
        {'p1': _entry(dets=[True, False])},
        {'p1': [(x, y, c) for _, x, y, c in pano.detections if c >= BENCHMARK_CONFIDENCE]},
        {'p1': pano}, fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE), frame)
    assert op_verdicts == {('p1', 1): True, ('p1', 3): False}
    assert len(points) == 1 and points[0].kind == 'det'
    assert not warnings


def test_missed_check_gates_the_recall_pool_but_not_precision():
    panos = [make_pano('g1', 0, -10, [(0, 0, 0.9)])]
    verdicts = {'g1': _entry(dets=[True], missed=[], no_missed=False)}
    r = es.evaluate_city(verdicts, _bundle_ops(panos, verdicts), panos,
                         fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    assert r['n_pool_ramps'] == 0          # unconfirmed missed-check: no recall
    assert r['precision']['tp'] == 1       # ...but the site verdict still counts
    assert r['counts']['no_pool'] == 1


def test_bundle_drift_skips_the_pano_with_a_warning():
    panos = [make_pano('g1', 0, -10, [(0, 0, 0.9)])]
    stale = {'g1': [(0.25, 0.6, 0.88)]}    # not what the run produced
    r = es.evaluate_city({'g1': _entry(dets=[True])}, stale, panos,
                         fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    assert r['counts']['skipped'] == 1
    assert any('drifted' in w for w in r['warnings'])
    assert r['n_pool_ramps'] == 0


# --- the #42 pose precondition ----------------------------------------------------------

def test_pose_precondition_scores_every_arm_on_the_same_sites():
    """One member that some arm cannot place drops its whole site from EVERY arm, so the
    arms' distances describe the same ramps (the trap the #42 study hit)."""
    panos = [make_pano('g1', 0, -10, [(0, 0, 0.9)]), make_pano('n1', 0, 10, [(0, 0, 0.9)]),
             make_pano('g2', 30, -10, [(30, 0, 0.9)]),
             make_pano('n2', 30, 22, [(30, 0, 0.9)])]
    # n2 sees ramp B 22 m away, and its rig is pitched up 3 deg: rotated by that pose the
    # ray meets the ground ~40 m out, past the 25 m cap -- so B cannot be scored at all.
    panos[3].camera_pitch, panos[3].camera_roll = 3.0, 0.0
    verdicts = {'g1': _entry(dets=[True]), 'g2': _entry(dets=[True])}
    rows, info = es.pose_precondition(
        verdicts, _bundle_ops(panos, verdicts), panos,
        fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE))
    assert (info['op_sites'], info['sites_scored']) == (2, 1)
    assert [r['arm'] for r in rows] == ['off', 'gravity', 'road', 'road-shuffled-within',
                                        'road-shuffled-across']
    # only ramp A is left, and nobody on it is posed: the arms must agree exactly, and
    # the off arm's refit must land where fuse put the site (on the ramp).
    for r in rows:
        assert r['n_common_matched'] == 1
        assert r['median_gt_to_site_m'] < 1e-6
        assert (r['world_recall_5m'], r['precision']) == (1.0, 1.0)
    # n2 has no sequence grade, so road mode fell back to gravity for it -- and said so
    assert info['gravity_fallback_panos'] == 1
    # survivorship is visible: nothing here becomes unplaceable, the off pool is both ramps
    for r in rows:
        assert r['gt_marks_unplaceable_vs_off'] == 0 and r['off_pool_ramps'] == 2


def test_shuffled_grades_keep_the_distribution_and_move_only_the_grade():
    panos = [make_pano(f'p{i}', 0, 10.0 * i, [(5, 10.0 * i, 0.9)]) for i in range(6)]
    for i, p in enumerate(panos):
        p.sequence_id = 'a' if i < 3 else 'b'
        p.grade_deg, p.travel_bearing_deg = float(i), 7.0
    panos[5].grade_deg = None                     # ungraded: stays ungraded
    within, unshuffled = es.shuffled_grades(panos, within_sequence=True)
    for orig, new in zip(panos, within):
        assert new.travel_bearing_deg == orig.travel_bearing_deg   # own bearing kept
    # each graded frame draws from ANOTHER frame of its own sequence
    assert all(within[i].grade_deg in {0.0, 1.0, 2.0} - {float(i)} for i in range(3))
    assert {within[3].grade_deg, within[4].grade_deg} == {3.0, 4.0}
    assert within[5].grade_deg is None and unshuffled == 0
    across, _ = es.shuffled_grades(panos, within_sequence=False)
    assert sorted(p.grade_deg for p in across[:5]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert es.shuffled_grades(panos, True)[0] == within          # seeded: reproducible
    assert [p.grade_deg for p in panos[:5]] == [0.0, 1.0, 2.0, 3.0, 4.0]   # input untouched


def test_control_rule_clauses():
    def city(road=(1.0, 2.0), shuf=(1.2, 2.2), rec=(0.90, 0.90), unpl=(10, 10), pool=100):
        base = {'off_pool_ramps': pool}
        return [dict(base, arm='off', recall_off_pool_2p5m=rec[0], gt_marks_unplaceable=unpl[0],
                     median_gt_to_site_m=1.5, p90_gt_to_site_m=3.0),
                dict(base, arm='road', median_gt_to_site_m=road[0], p90_gt_to_site_m=road[1],
                     recall_off_pool_2p5m=rec[1], gt_marks_unplaceable=unpl[1]),
                dict(base, arm='road-shuffled-within', median_gt_to_site_m=shuf[0],
                     p90_gt_to_site_m=shuf[1], recall_off_pool_2p5m=rec[1],
                     gt_marks_unplaceable=unpl[1])]
    good = {c: city() for c in 'abcde'}
    assert es.control_verdict(good)[0]
    # (i) a shuffled grade within 0.1 m of the real one (on either statistic) in 2 of 5
    # cities fails the control: only 3 cities show the frame's own grade mattering
    tie = dict(good, d=city(shuf=(1.05, 2.5)), e=city(shuf=(1.0, 2.0)))
    ok, clauses, _ = es.control_verdict(tie)
    assert not ok and clauses == {'i': False, 'ii': True, 'iii': True}
    # (ii) 1.1 points of off-pool recall lost in one city
    ok, clauses, _ = es.control_verdict(dict(good, a=city(rec=(0.900, 0.889))))
    assert not ok and not clauses['ii']
    # (iii) six more unplaceable marks on a 100-ramp pool (limit 5)
    ok, clauses, _ = es.control_verdict(dict(good, a=city(unpl=(10, 16))))
    assert not ok and not clauses['iii']


def test_precondition_rule_needs_p90_everywhere_and_three_median_wins():
    def rows(off_med, off_p90, road_med, road_p90):
        return [{'arm': 'off', 'median_gt_to_site_m': off_med, 'p90_gt_to_site_m': off_p90},
                {'arm': 'road', 'median_gt_to_site_m': road_med, 'p90_gt_to_site_m': road_p90}]
    better = rows(2.0, 4.0, 1.5, 3.5)
    assert es.precondition_verdict({c: better for c in 'abcde'})[0]
    # within the 0.1 m tolerance is "no worse"; three median wins are enough
    ok = {'a': better, 'b': better, 'c': better,
          'd': rows(2.0, 4.0, 2.1, 4.09), 'e': rows(2.0, 4.0, 2.0, 4.0)}
    assert es.precondition_verdict(ok)[0]
    # one city's p90 worse by more than the tolerance vetoes it, whatever the medians say
    assert not es.precondition_verdict({**ok, 'e': rows(2.0, 4.0, 1.0, 4.2)})[0]
    # two median wins are not enough
    assert not es.precondition_verdict({'a': better, 'b': better, 'c': rows(2, 4, 2, 4)})[0]
