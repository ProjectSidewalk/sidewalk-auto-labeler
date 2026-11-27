"""Pure-logic checks for scripts/per_rig_threshold.py (no data, no network)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np


import per_rig_threshold as prt  # noqa: E402

R = prt.RADIUS_NORMALIZED  # match radius in x-normalized units


def test_build_ground_truth_semantics():
    dets = [dict(x_normalized=0.1, y_normalized=0.6), dict(x_normalized=0.2, y_normalized=0.6),
            dict(x_normalized=0.3, y_normalized=0.6), dict(x_normalized=0.4, y_normalized=0.6)]
    gt, ign = prt.build_ground_truth(dets, [True, False, 'unsure', 'duplicate'],
                                     [{'x': 0.5, 'y': 0.6}, {'x': 0.7, 'y': 0.6, 'unsure': True}])
    assert gt == [(0.1, 0.6), (0.5, 0.6)]
    assert ign == [(0.3, 0.6), (0.7, 0.6)]


def test_classify_greedy_redundant_ignore_and_seam():
    gt = [(0.5, 0.6), (0.001, 0.6)]
    ign = [(0.8, 0.6)]
    preds = [(0.5, 0.6, 0.9),            # claims the first GT
             (0.5 + R / 2, 0.6, 0.4),    # second hit on a claimed ramp -> FP
             (0.999, 0.6, 0.2),          # across the seam from (0.001) -> TP (wrapped)
             (0.8, 0.6, 0.15),           # near an ignore point -> ignored
             (0.3, 0.6, 0.12)]           # nothing near -> FP
    out = prt.classify(preds, gt, ign)
    assert out == [(0.9, 'tp'), (0.4, 'fp'), (0.2, 'tp'), (0.15, 'ignore'), (0.12, 'fp')]


def test_classify_strict_radius():
    out = prt.classify([(0.5 + R, 0.6, 0.7)], [(0.5, 0.6)], [])
    assert out == [(0.7, 'fp')]   # exactly at the radius is outside (strict <)


def test_counts_on_grid_is_a_prefix_count():
    tp, fp = prt.counts_on_grid([(0.9, 'tp'), (0.4, 'fp'), (0.3, 'tp'), (0.2, 'ignore')])
    i = {round(t, 2): k for k, t in enumerate(prt.GRID)}
    assert (tp[i[0.10]], fp[i[0.10]]) == (2, 1)
    assert (tp[i[0.30]], fp[i[0.30]]) == (2, 1)   # conf >= t keeps 0.30
    assert (tp[i[0.31]], fp[i[0.31]]) == (1, 1)
    assert (tp[i[0.41]], fp[i[0.41]]) == (1, 0)


def test_in_border_matches_exclude_border_10():
    assert prt.in_border(9 / 1024, 0.6)
    assert not prt.in_border(10 / 1024, 0.6)
    assert not prt.in_border(1013 / 1024, 0.6)
    assert prt.in_border(1014 / 1024, 0.6)
    assert prt.in_border(0.5, 502 / 512)


def test_best_threshold_ties_go_low():
    tp = np.array([5, 5, 4])
    fp = np.array([2, 2, 0])
    assert prt.best_threshold_index(tp, fp, 1.0) == 2     # 3, 3, 4
    tp = np.array([5, 5, 3])
    assert prt.best_threshold_index(tp, fp, 1.0) == 0     # 3, 3, 3 -> lowest


def test_best_threshold_lambda():
    tp = np.array([10, 8])
    fp = np.array([4, 1])
    assert prt.best_threshold_index(tp, fp, 0.5) == 0     # 8 vs 7.5
    assert prt.best_threshold_index(tp, fp, 1.0) == 1     # 6 vs 7


def test_eligible_rigs_needs_two_cities():
    g = {'a': {'x': 150, 'y': 5}, 'b': {'x': 60, 'y': 60}, 'c': {'x': 30, 'y': 30},
         'other': {'x': 500, 'y': 500}}
    assert prt.eligible_rigs(g) == {'b'}
    assert prt.eligible_rigs(g, min_cities=1) == {'a', 'b'}


def test_fit_per_rig_and_other_fallback():
    n_grid = 3
    # Rig 0 in two cities, 60 GT each: prefers index 2. Rig 1 in one city: pooled into other.
    tp = np.array([[10, 9, 9], [10, 9, 9], [5, 5, 1]], dtype=float)
    fp = np.array([[5, 3, 0], [5, 3, 0], [0, 0, 0]], dtype=float)
    ngt = np.array([60.0, 60.0, 50.0])
    rig = np.array([0, 0, 1])
    city = np.array(['x', 'y', 'x'])
    w = np.ones(3)
    rig_idx, other_idx, elig = prt.fit_per_rig(tp, fp, ngt, rig, city, w, 1.0, 1, ['r0', 'r1'])
    assert elig == {'r0'} and rig_idx == {'r0': 2}
    assert other_idx == 1          # rig 1 holds 50 < 100 GT -> the global index passed in
    assert tp.shape[1] == n_grid


def test_shuffle_within_preserves_each_group():
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 0])
    groups = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    out = prt.shuffle_within(labels, groups, rng)
    for g in ('a', 'b'):
        assert Counter(out[groups == g]) == Counter(labels[groups == g])


def test_score_per_pano_thresholds():
    tp = np.array([[3, 2], [4, 1]], dtype=float)
    fp = np.array([[2, 0], [3, 1]], dtype=float)
    assert prt.score(tp, fp, np.array([1, 0]), np.array([1.0, 2.0])) == (10.0, 6.0)
