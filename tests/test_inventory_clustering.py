"""inventory_clustering.py (issue #106 Part 2): the pre-registered metrics and decision rule
on a toy layout. Needs scipy (and eval_ps_clustering's analysis deps), so skips in CI."""
import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')
pytest.importorskip('shapely')

import numpy as np  # noqa: E402

import inventory_clustering as ic  # noqa: E402

# ramps: A (0, 0) and B (12, 0) 12 m apart; a dual pair C (40, 0) / D (42, 0) 2 m apart;
# E (0, 80) is 40+ m from every pano, so it is not in the visible pool
INV = np.array([[0.0, 0.0], [12.0, 0.0], [40.0, 0.0], [42.0, 0.0], [0.0, 80.0]])
PANOS = np.array([[0.0, -10.0], [20.0, -10.0], [41.0, -10.0]])


def _cluster(*members):
    pts = [tuple(m) for m in members]
    cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return cen, pts


def test_visible_pool_excludes_the_unseen_ramp():
    assert ic.visible_pool(INV, PANOS).tolist() == [True, True, True, True, False]


def test_covered_split_and_merge():
    pool = ic.visible_pool(INV, PANOS)
    clusters = [
        _cluster((0.5, 0.0), (-0.5, 0.0)),           # A, once
        _cluster((12.5, 0.5)),                       # B ...
        _cluster((11.5, -0.5)),                      # ... twice: a split
        _cluster((40.0, 0.2), (40.1, -0.1), (42.0, 0.1), (41.9, -0.2)),  # C+D: a merge
        (None, []),                                  # unplaceable: counted, not placed
    ]
    m = ic.inventory_metrics(clusters, INV, pool, 5.0)
    assert m['pool'] == 4 and m['n_clusters'] == 5 and m['n_placed'] == 4
    # A, B and one of C/D (the merged centroid sits at 41 m, nearest C or D) are covered
    assert m['covered'] == 3 and m['split'] == 1
    assert m['extra_per_covered'] == pytest.approx(1 / 3)
    assert (m['merge_k'], m['merge_n']) == (1, 2)
    # at r = 0.3 m only A (centroid exactly on it) is still covered
    assert ic.inventory_metrics(clusters, INV, pool, 0.3)["covered"] == 1


def _rows(split_fusion_auto, split_fusion_26=None, merge_fusion=0.02, cov_fusion=0.90):
    rows = []
    for city in ('bend', 'gainesville'):
        for frame, split_f in (('auto', split_fusion_auto),
                               ('2.6', split_fusion_26 if split_fusion_26 is not None
                                else split_fusion_auto)):
            rows.append({'city': city, 'tier': '0.3', 'frame': frame, 'arm': ic.PS_ARM,
                         'r': '5.0', 'split_rate': 0.30, 'merge_rate': 0.02,
                         'covered_rate': 0.90})
            rows.append({'city': city, 'tier': '0.3', 'frame': frame, 'arm': ic.FUSION_ARM,
                         'r': '5.0', 'split_rate': split_f, 'merge_rate': merge_fusion,
                         'covered_rate': cov_fusion})
    return rows


def test_verdict_rule():
    assert ic.verdict(_rows(0.20))[0] == ic.PASS
    assert ic.verdict(_rows(0.27))[0] == ic.NOT_ESTABLISHED          # only 3 points lower
    assert ic.verdict(_rows(0.20, merge_fusion=0.04))[0] == ic.NOT_ESTABLISHED
    assert ic.verdict(_rows(0.20, cov_fusion=0.88))[0] == ic.NOT_ESTABLISHED
    assert ic.verdict(_rows(0.20, split_fusion_26=0.31))[0] == ic.FRAME_DEPENDENT
