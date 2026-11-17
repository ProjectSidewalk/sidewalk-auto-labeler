"""scripts/corner_prerater.py (issue #185), Stage 0 on synthetic cases: the P_free formula and
its break-even v, the pool the formula adds back, the over-merge filter, the blind tally, and
the figure rows. CPU only, no network, no real data."""
import pytest

import corner_prerater as cp

K = cp.ARM_KEY


def corner(i, state, inv=None, sites=(), lat=0.0, lng=0.0, inventory=()):
    counts = dict.fromkeys(('Available', 'NA_noramp', 'NA_typed', 'RMV', 'Expired/Removed',
                            'other'), 0)
    counts.update(inv or {})
    return {'corner': i, 'state': {K: state}, 'inv_counts': counts, 'site_ids': list(sites),
            'lat': lat, 'lng': lng, 'inventory': list(inventory)}


def test_p_free_matches_the_plan_numbers():
    # the plan's worst case and point estimate (review F1)
    assert cp.p_free(3070, 0.944, 283, 0.235) == pytest.approx(0.9289, abs=1e-4)
    assert cp.p_free(3070, 96 / 97, 283, 5 / 45) == pytest.approx(0.9812, abs=1e-4)
    # nothing removed: P_free is p
    assert cp.p_free(100, 0.9, 0, 0.5) == pytest.approx(0.9)


def test_breakeven_v_is_where_p_free_meets_the_bar():
    v = cp.breakeven_v(3070, 0.944, 283)
    assert cp.p_free(3070, 0.944, 283, v) == pytest.approx(0.90)
    # at a high enough p the bar holds even if every removed corner is a ramp
    assert cp.breakeven_v(3070, 0.990, 283) > 1


def test_pool_is_absent_and_excluded_under_the_discount_read():
    rec = {'unit': 'u', 'type': 'residential', 'corners': [
        corner(0, 'absent', {'Available': 1}),
        corner(1, 'absent', {'NA_noramp': 2}),      # discount read: not excluded
        corner(2, 'present', {'Available': 1}),     # present: never in the pool
        corner(3, 'absent', {'RMV': 1}),            # excluded without an Available point
        corner(4, 'unobservable', {'Available': 1}),
    ]}
    mid = {'unit': 'm', 'type': 'mid_block', 'corners': [corner(0, 'absent', {'Available': 1})]}
    got = [c['corner'] for _, c in cp.pool([rec, mid])]
    assert got == [0, 3]


def test_adjacent_present_wraps_and_skips_the_opposite_corner():
    cs = [corner(0, 'absent'), corner(1, 'present'), corner(2, 'present'), corner(3, 'present')]
    assert [c['corner'] for c in cp.adjacent_present(cs, 0)] == [1, 3]
    assert cp.adjacent_present([corner(0, 'absent')], 0) == []
    two = [corner(0, 'absent'), corner(1, 'present')]
    assert [c['corner'] for c in cp.adjacent_present(two, 0)] == [1]


def test_reassign_moves_only_points_nearer_a_neighbour_site():
    # one point near the neighbour's site, one near the corner itself
    assert cp.reassign((0, 0), [(9, 0), (1, 0)], [(10, 0)], False) == (1, False)
    assert cp.reassign((0, 0), [(9, 0), (8, 1)], [(10, 0)], False) == (2, True)
    # a tie stays put (strictly nearer)
    assert cp.reassign((0, 0), [(5, 0)], [(10, 0)], False) == (0, False)
    # no present neighbour, nothing moves
    assert cp.reassign((0, 0), [(9, 0)], [], False) == (0, False)
    # another excluding class keeps the corner removed
    assert cp.reassign((0, 0), [(9, 0)], [(10, 0)], True) == (1, False)


def test_blind_tally_reads_blind_not_final_verdicts():
    verdicts = {'units': {'u': {
        'blind': {'0': {'verdict': 'present'}, '1': {'verdict': 'cant_tell'}},
        'corners': {'0': {'verdict': 'absent'}, '1': {'verdict': 'absent'}}}}}
    t = cp.blind_tally(verdicts, [('u', 0), ('u', 1), ('u', 2), ('x', 0)])
    assert t == {'absent': 0, 'present': 1, 'cant_tell': 1, 'unrated': 2}


def test_judged_pool_refuses_a_corner_not_absent_in_the_table():
    item = {'unit': 'u', 'corners': [{'corner': 0, 'lat': 1.0, 'lng': 2.0,
                                      'inv_counts': {'Available': 1}}]}
    ok = {'u': {'corners': [corner(0, 'absent', lat=1.0, lng=2.0)]}}
    assert len(cp.judged_pool([item], ok)) == 1
    bad = {'u': {'corners': [corner(0, 'present', lat=1.0, lng=2.0)]}}
    with pytest.raises(SystemExit):
        cp.judged_pool([item], bad)


def test_figure_rows_end_points():
    gate = {'counts': {'proposed': 3070, 'excluded_inventory': 283},
            'p': {'share': 0.99, 'wilson95': [0.944, 0.998]},
            'cant_tell_against': {'p_wilson_lower': 0.90}}
    rows = cp.figure_rows(gate)
    assert len(rows) == 3 * 101
    first = [r for r in rows if r['curve'] == 'p Wilson lower'][0]
    assert first['v'] == 0 and first['p_free'] == pytest.approx(cp.p_free(3070, 0.944, 283, 0),
                                                                abs=1e-5)
