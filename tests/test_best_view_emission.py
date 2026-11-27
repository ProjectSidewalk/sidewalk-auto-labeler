"""best_view_emission.py: arm selection, the far-field gate, the anchored metric and the
pre-registered rule, on toy inputs. Needs numpy, scipy and shapely (skips without them)."""
from types import SimpleNamespace

import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')

import numpy as np  # noqa: E402

import best_view_emission as bve  # noqa: E402


def det(pid, idx, y, conf, operational=True):
    return SimpleNamespace(pano_id=pid, det_index=idx, y=y, conf=conf, operational=operational)


def site(sid, dets, e=0.0, n=0.0):
    return SimpleNamespace(id=sid, members=[(d, True) for d in dets], e=e, n=n,
                           n_operational=sum(d.operational for d in dets))


def lab(key, range_m, e=0.0, n=0.0, y=0.6):
    return bve.Label(key, 0.0, 0.0, range_m, y, 0.5, e, n)


def test_best_takes_the_largest_dip_then_confidence():
    s = site(0, [det('a', 0, 0.60, 0.9), det('b', 1, 0.70, 0.4), det('c', 2, 0.70, 0.5),
                 det('d', 3, 0.95, 0.9, operational=False)])   # sub-tier: never a candidate
    assert bve.select_best([s]) == {0: ('c', 2)}


def test_sites_without_an_operational_member_emit_nothing():
    s = site(0, [det('a', 0, 0.6, 0.2, operational=False)])
    assert bve.select_best([s]) == {} and bve.select_random([s]) == {}


def test_random_is_seeded_and_picks_a_member():
    sites = [site(i, [det(f'p{i}', j, 0.6, 0.5) for j in range(5)]) for i in range(20)]
    a, b = bve.select_random(sites, 3), bve.select_random(sites, 3)
    assert a == b
    assert all(k[0] == f'p{sid}' for sid, k in a.items())
    assert bve.select_random(sites, 4) != a


def test_gate_drops_far_labels_only_when_the_site_has_a_near_member():
    s0 = site(0, [det('a', 0, 0.6, 0.5), det('b', 0, 0.52, 0.5)])   # near + far
    s1 = site(1, [det('c', 0, 0.52, 0.5)])                          # far only
    labels = {('a', 0): lab(('a', 0), 8.0), ('b', 0): lab(('b', 0), 22.0),
              ('c', 0): lab(('c', 0), 23.0), ('u', 0): lab(('u', 0), 23.5)}  # u: in no site
    assert sorted(bve.gate_keep(labels, [s0, s1])) == [('a', 0), ('c', 0), ('u', 0)]


def test_build_arms_counts_and_unplaced():
    s0 = site(0, [det('a', 0, 0.6, 0.5), det('b', 0, 0.7, 0.5)])
    labels = {('a', 0): lab(('a', 0), 8.0), ('b', 0): lab(('b', 0), 5.0),
              ('u', 0): lab(('u', 0), 23.0)}
    arms, per_site, unplaced = bve.build_arms(labels, [s0])
    assert unplaced == [('u', 0)]
    assert arms[bve.ALL] == sorted(labels)
    assert arms[bve.BEST] == [('b', 0)]
    assert arms[bve.BEST_U] == [('b', 0), ('u', 0)]
    assert per_site[bve.ALL] == {0: [('a', 0), ('b', 0)]}


def test_build_arms_refuses_a_member_without_a_label():
    with pytest.raises(ValueError):
        bve.build_arms({}, [site(0, [det('a', 0, 0.6, 0.5)])])


def test_anchored_pairs_and_distances():
    inv = np.array([[0.0, 0.0], [20.0, 0.0], [100.0, 0.0]])
    pool = np.array([True, True, False])          # the third ramp is not visible
    s0 = site(0, [det('a', 0, 0.6, 0.5)], e=1.0)  # pairs with ramp 0
    s1 = site(1, [det('b', 0, 0.6, 0.5)], e=26.0)  # 6 m from ramp 1: beyond 5 m
    pairs = bve.anchored_pairs(inv, pool, [s0, s1])
    assert pairs == [(0, 0)]
    labels = {('a', 0): lab(('a', 0), 5.0, e=3.0, n=4.0)}
    one, every, n_none = bve.anchored_distances(pairs, inv, {0: [('a', 0)]}, labels)
    assert one == [5.0] and every == [5.0] and n_none == 0


def test_ps_clusters_cannot_link_and_threshold():
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [9.0, 0.0], [40.0, 0.0]])
    # complete linkage: 0-1-2 spans 9 m > 7.5 m, so {0, 1} and {2}; 3 alone
    assert bve.ps_clusters(pts, ['a', 'b', 'c', 'd']).tolist() == [0, 0, 1, 2]
    assert bve.ps_clusters(pts[:2], ['a', 'a']).tolist() == [0, 1]


def test_pct_nearest_rank():
    assert bve.pct([3.0, 1.0, 2.0], 0.5) == 2.0
    assert bve.pct([], 0.5) is None


def test_bootstrap_ci_brackets_a_constant_shift():
    a = np.linspace(0, 10, 200)
    ci = bve.bootstrap_diff(a, a + 1.0, n_boot=200)
    assert ci[0.9][0] == pytest.approx(1.0) and ci[0.9][1] == pytest.approx(1.0)


def _row(city, arm, p90, cov):
    return {'city': city, 'tier': '0.3', 'frame': 'auto', 'arm': arm,
            'm1_one_p90': p90, 'covered_r5': cov}


def _rows(rand_p90_g=4.0, best_cov_g=0.890):
    rows = []
    for city, rp in (('bend', 4.0), ('gainesville', rand_p90_g)):
        cov = best_cov_g if city == 'gainesville' else 0.890
        rows += [_row(city, bve.ALL, None, 0.895), _row(city, bve.BEST, 3.0, cov),
                 _row(city, bve.RANDOM, rp, cov)]
    return rows


def test_verdict_pass_fail_undetermined():
    ok = {'bend': 'verified', 'gainesville': 'verified'}
    assert bve.verdict(_rows(), ok)[0] == bve.PASS
    assert bve.verdict(_rows(rand_p90_g=3.4), ok)[0] == bve.FAIL       # margin 0.4 < 0.5
    assert bve.verdict(_rows(best_cov_g=0.880), ok)[0] == bve.FAIL     # coverage -1.5 pts
    sub = {'bend': 'SUBSTITUTE', 'gainesville': 'verified'}
    assert bve.verdict(_rows(), sub)[0] == bve.UNDETERMINED
    only_bend = [r for r in _rows() if r['city'] == 'bend']
    assert bve.verdict(only_bend, {'bend': 'verified'})[0] == bve.UNDETERMINED
    # a failure on a substitute inventory is not a verified failure
    assert bve.verdict(_rows(rand_p90_g=3.4), {'bend': 'verified',
                                                'gainesville': 'SUBSTITUTE'})[0] \
        == bve.UNDETERMINED


def test_synthesize_matches_the_documented_example(tmp_path):
    import json
    rec = {'pano': {'panorama_id': 'p', 'lat': 47.0, 'lng': -122.0, 'camera_heading': 0.0,
                    'width': 4096, 'height': 2048},
           'detections': [{'x_normalized': 0.5, 'y_normalized': 0.6, 'confidence': 0.7},
                          {'x_normalized': 0.5, 'y_normalized': 0.99, 'confidence': 0.9},
                          {'x_normalized': 0.5, 'y_normalized': 0.6, 'confidence': 0.2}]}
    p = tmp_path / 'results.jsonl'
    p.write_text(json.dumps(rec) + '\n')
    labels = bve.synthesize(p, 0.30)
    assert list(labels) == [('p', 0)]           # rig detection masked, sub-tier dropped
    assert labels[('p', 0)].range_m == pytest.approx(7.20, abs=0.01)
    assert labels[('p', 0)].lat > 47.0
