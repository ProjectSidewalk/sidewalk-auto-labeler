"""dual_corner_split.py: the partition, the split rule, the random control, emission, the
cluster bootstrap and the verdict, on toy inputs (pure logic; no runs, no inventory).
Needs numpy, scipy and shapely (skips without them)."""
from types import SimpleNamespace as N

import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')

import numpy as np  # noqa: E402

import dual_corner_split as dcs  # noqa: E402

I = (0.25, 0.0, 0.25)   # isotropic 0.5 m sigma


def test_gls_matches_the_mean_and_residual():
    e, n, c = dcs.gls([(0.0, 0.0, (1.0, 0.0, 1.0)), (2.0, 0.0, (1.0, 0.0, 1.0))])
    assert (round(e, 9), round(n, 9), round(c, 9)) == (1.0, 0.0, 2.0)


def test_bipartition_finds_two_separated_groups():
    pts = [(6, 0, I), (0, 0, I), (6.2, 0.1, I), (0.2, -0.1, I), (0.1, 0.1, I)]
    d, a, b = dcs.bipartition(pts)
    assert {a, b} == {(1, 3, 4), (0, 2)}
    assert a[0] < b[0]                    # canonical order
    whole = dcs.gls(pts)[2]
    assert d == pytest.approx(whole - dcs.gls([pts[i] for i in a])[2]
                              - dcs.gls([pts[i] for i in b])[2])


def test_bipartition_of_two_points_and_identical_points():
    d, a, b = dcs.bipartition([(0, 0, I), (1, 0, I)])
    assert (a, b) == ((0,), (1,)) and d == pytest.approx(2.0)
    d, a, b = dcs.bipartition([(0, 0, I)] * 3)
    assert d == pytest.approx(0.0) and len(a) + len(b) == 3 and a and b


def test_chi2_prefix_matches_gls():
    rng = np.random.default_rng(0)
    pts = [(float(x), float(y), (0.3 + float(v), 0.05, 0.4)) for x, y, v in rng.random((6, 3))]
    w = np.array([dcs._info(p[2]) for p in pts])
    xy = np.array([(p[0], p[1]) for p in pts])
    pre = dcs._chi2_prefix(w, xy[:, 0], xy[:, 1])
    for k in range(1, 7):
        assert pre[k - 1] == pytest.approx(dcs.gls(pts[:k])[2], abs=1e-9)


def plan_row(sid, dchi2, sep, rng_m, n=3):
    return N(site=N(id=sid), dchi2=dchi2, label_sep=sep, other_range=rng_m,
             members=[0] * n)


def test_rule_splits_needs_all_three_conditions():
    plan = [plan_row(0, 5.0, 3.0, 9.0), plan_row(1, 1.0, 3.0, 9.0),
            plan_row(2, 5.0, 1.0, 9.0), plan_row(3, 5.0, 3.0, 20.0),
            N(site=N(id=4), dchi2=None, label_sep=None, other_range=None, members=[0])]
    assert dcs.rule_splits(plan, 2.0, 2.0, 14.0) == {0}
    assert dcs.rule_splits(plan, 0.0, 0.0, float('inf')) == {0, 1, 2, 3}


def test_random_split_is_seeded_sized_and_skips_single_member_sites():
    plan = [plan_row(i, 0, 0, 0, n=1 + i % 4) for i in range(40)]
    a, b = dcs.random_split(plan, 10, 192), dcs.random_split(plan, 10, 192)
    assert a == b and len(a) == 10
    sizes = {p.site.id: len(p.members) for p in plan}
    for sid, (pa, pb) in a.items():
        assert sizes[sid] >= 2 and pa and pb
        assert sorted(pa + pb) == list(range(sizes[sid]))
    assert dcs.random_split(plan, 10, 7) != a
    with pytest.raises(ValueError):
        dcs.random_split(plan, 1000, 1)


def test_emit_one_best_key_per_part():
    cands = [('a', 0, 0.60, 0.9), ('b', 0, 0.70, 0.5), ('c', 0, 0.80, 0.5)]
    plan = [N(site=N(id=0), cands=cands, best=('c', 0)),
            N(site=N(id=1), cands=cands[:1], best=('a', 0))]
    assert dcs.emit(plan, {}) == [(0, ('c', 0)), (2, ('a', 0))]
    assert dcs.emit(plan, {0: ((0, 1), (2,))}) == [(0, ('b', 0)), (1, ('c', 0)),
                                                    (2, ('a', 0))]


def test_randmember_is_seeded_and_draws_from_the_part():
    cands = [('a', i, 0.6, 0.5) for i in range(5)]
    plan = [N(site=N(id=0), cands=cands)]
    r = dcs.randmember(plan, {0: ((0, 1), (2, 3, 4))}, 3)
    assert r == dcs.randmember(plan, {0: ((0, 1), (2, 3, 4))}, 3)
    assert r[0][1] in (0, 1) and r[1][1] in (2, 3, 4)


def test_ramp_clusters_and_cluster_bootstrap():
    cl = dcs.ramp_clusters(np.array([[0, 0], [1, 0], [50, 0], [9, 9]]),
                           np.array([1, 1, 1, 0], bool), np.array([[0.5, 0.0]]))
    assert cl.tolist() == [0, 0, 1]
    x, y = np.array([1, 1, 0, 0]), np.zeros(4)
    p, lo, hi = dcs.cluster_bootstrap_diff(x, y, np.array([0, 0, 1, 1]), 500, 1)
    assert p == 0.5 and 0.0 <= lo <= p <= hi <= 1.0
    p, lo, hi = dcs.cluster_bootstrap_diff(x, x, np.array([0, 0, 1, 1]), 50, 1)
    assert (p, lo, hi) == (0.0, 0.0, 0.0)


def test_select_follows_the_preregistered_objective():
    rows = [dict(p90_gain=1.2, covered_r5=0.900, covered_r5_all=0.91, n_labels=10, n_split=1),
            dict(p90_gain=1.2, covered_r5=0.906, covered_r5_all=0.91, n_labels=20, n_split=5),
            dict(p90_gain=1.2, covered_r5=0.907, covered_r5_all=0.91, n_labels=30, n_split=6),
            dict(p90_gain=0.8, covered_r5=0.909, covered_r5_all=0.91, n_labels=12, n_split=2)]
    assert dcs.select(rows)['n_labels'] == 20          # fewest reaching all - 0.005
    rows = [dict(r, covered_r5=r['covered_r5'] - 0.01) for r in rows]
    assert dcs.select(rows)['n_labels'] == 30          # none reach it: highest coverage


def _rows(cov_all, cov_split, gain, lo):
    base = dict(city='gainesville', tier='0.3000', frame='auto')
    rows = [dict(base, arm='all', setting='rule', covered_r5=str(cov_all)),
            dict(base, arm='best+split', setting='rule', covered_r5=str(cov_split))]
    boots = [dict(base, setting='rule', p90_gain=str(gain), p90_gain_lo='0', p90_gain_hi='0',
                  cov_diff_vs_best_random_split='0.01',
                  cov_diff_vs_best_random_split_lo=str(lo),
                  cov_diff_vs_best_random_split_hi='0.02')]
    return rows, boots


def test_verdict_clauses():
    assert dcs.verdict(*_rows(0.89, 0.885, 0.9, 0.001))[0] == dcs.PASS
    assert dcs.verdict(*_rows(0.89, 0.875, 0.9, 0.001))[0] == dcs.FAIL    # (i)
    assert dcs.verdict(*_rows(0.89, 0.885, 0.4, 0.001))[0] == dcs.FAIL    # (ii)
    assert dcs.verdict(*_rows(0.89, 0.885, 0.9, -0.001))[0] == dcs.FAIL   # (iii)
    assert dcs.verdict([], [])[0] == dcs.UNDETERMINED
