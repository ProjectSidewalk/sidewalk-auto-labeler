"""Geometry helpers of scripts/aerial_sidewalks.py (#104): anchor extraction and the
on-surface distance. No data, no network."""
import math

import pytest

pytest.importorskip('shapely')

from shapely.geometry import box  # noqa: E402

import aerial_sidewalks as asw  # noqa: E402


def test_crosswalk_across_a_street_gives_one_anchor_per_curb():
    cw = [box(0, 0, 3, 12)]                                   # 12 m street crossing
    sw = [box(-10, -3, 13, 0), box(-10, 12, 13, 15)]          # the two sidewalks
    anchors = sorted(asw.crosswalk_anchors(cw, sw))
    assert len(anchors) == 2
    (e0, n0), (e1, n1) = anchors
    assert e0 == pytest.approx(1.5) and e1 == pytest.approx(1.5)
    assert n0 < 1.0 and n1 > 11.0                             # at the curbs, not mid-street


def test_crosswalk_touching_no_sidewalk_gives_no_anchor():
    assert asw.crosswalk_anchors([box(0, 0, 3, 12)], [box(50, 50, 60, 52)]) == []


def test_network_endpoints_merge_with_polygon_anchors():
    cw = [box(0, 0, 3, 12)]
    sw = [box(-10, -3, 13, 0), box(-10, 12, 13, 15)]
    # the network's crosswalk endpoints sit on the same two corners -> still two anchors
    anchors = asw.crosswalk_anchors(cw, sw, extra_points=[(1.5, 0.5), (1.5, 11.5)])
    assert len(anchors) == 2
    # ...and a far endpoint is its own anchor
    assert len(asw.crosswalk_anchors(cw, sw, extra_points=[(40.0, 0.0)])) == 3


def test_surface_distance_is_zero_inside_and_metric_outside():
    s = asw.Surface([box(0, 0, 10, 2), box(20, 0, 22, 10)])
    d = s.distance([(5, 1), (5, 5), (15, 1), (21, 12)])
    assert list(d) == pytest.approx([0.0, 3.0, 5.0, 2.0])
    assert math.isinf(asw.Surface([]).distance([(0, 0)])[0])


def test_edge_px_wraps_the_seam_and_handles_no_edges():
    import numpy as np
    px, dx, dy = asw.edge_px(np.array([[0.001, 0.51], [0.5, 0.5]]), 0.999, 0.5)
    assert dx == pytest.approx(2.048) and dy == pytest.approx(5.12)
    assert px == pytest.approx(math.hypot(2.048, 5.12), abs=1e-3)
    assert asw.edge_px(np.zeros((0, 2)), 0.5, 0.5) == (None, None, None)


def test_displaced_mark_moves_40px_and_is_order_free():
    xs = [asw.displaced_x('pano', f'det:{k}', 0.999) for k in range(20)]
    for x in xs:
        dx = ((x - 0.999 + 0.5) % 1.0 - 0.5) * asw.HEATMAP_W
        assert abs(dx) == pytest.approx(asw.Q2_DISPLACE_PX)
    assert len({round(x, 6) for x in xs}) == 2          # both sides drawn
    assert asw.displaced_x('pano', 'det:3', 0.999) == xs[3]


def test_bootstrap_gap_needs_both_sides():
    assert asw.bootstrap_gap([[[(True, False)]]]) == (None, None, None)


# ---- the pre-registered rules (#104), at their boundaries

def test_q1_verdict_bar_is_inclusive():
    assert asw.q1_verdict({'share_within_2m': 0.90, 'n': 10})[0] == 'USABLE'
    assert asw.q1_verdict({'share_within_2m': 0.8999, 'n': 10})[0] == 'NOT USABLE'
    assert asw.q1_verdict(None)[0] == 'NOT MEASURED'


def _q3_rows(d_frag, d_cov, d_dual, cities=asw.CITIES):
    rows = []
    for city in cities:
        for base in asw.Q3_BASES:
            for arm in asw.Q3_ARMS:
                rows.append({'city': city, 'base': base, 'arm': arm,
                             'base_frag5': 0.20, 'base_coverage': 0.90, 'base_dual': 0.80,
                             'frag5': 0.20 + d_frag, 'coverage': 0.90 + d_cov,
                             'dual': 0.80 + d_dual})
    return rows


def test_q3_verdict_boundaries():
    # exactly at the bars passes: frag -0.03, coverage and dual -0.01
    label, cells = asw.q3_verdict(_q3_rows(-0.03, -0.01, -0.01))
    assert label == 'YES' and cells['fusion / anchored']['verdict'] == 'PASS'
    # frag cut just short of 0.03 fails
    assert asw.q3_verdict(_q3_rows(-0.0299, 0.0, 0.0))[0] == 'NO'
    # Bend's real fusion / anchored coverage change (-0.0101) fails the 0.01 tolerance
    assert asw.q3_verdict(_q3_rows(-0.05, -0.0101, 0.0))[0] == 'NO'
    assert asw.q3_verdict(_q3_rows(-0.05, 0.0, -0.0101))[0] == 'NO'
    # one city alone never passes
    assert asw.q3_verdict(_q3_rows(-0.05, 0.0, 0.0, cities=('paterson',)))[0] == 'NO'


def _q4(true_on, gap, lo, n_false):
    cities = [{'city': c, 'true_on_share': t} for c, t in zip(asw.CITIES, true_on)]
    return asw.q4_verdict(cities, {'gap': gap, 'gap_lo': lo, 'gap_hi': 0.5,
                                   'n_false': n_false})[0]


def test_q4_verdict_boundaries():
    assert _q4((0.90, 0.95), 0.15, 0.01, 30) == 'FILTER'
    assert _q4((0.8999, 0.95), 0.15, 0.01, 30) == 'NOT A FILTER'   # clause (i)
    assert _q4((0.90, 0.95), 0.1499, 0.01, 30) == 'NOT A FILTER'   # gap below 0.15
    assert _q4((0.90, 0.95), 0.20, 0.0, 30) == 'NOT A FILTER'      # CI touches 0
    assert _q4((0.90, 0.95), 0.20, 0.05, 29) == 'NOT ESTABLISHED (underpowered)'


# ---- Q3 arms

def test_anchor_clusters_snaps_within_radius_and_merges_shared_anchors():
    from types import SimpleNamespace as N
    cs = [N(id=0, members=[('a', 0)], n_labels=1, e=0.5, n=0.0, label_ids=[1]),
          N(id=1, members=[('b', 0)], n_labels=2, e=-0.5, n=0.0, label_ids=[2]),
          N(id=2, members=[('c', 0)], n_labels=1, e=3.01, n=0.0, label_ids=[]),
          N(id=3, members=[('d', 0)], n_labels=1, e=None, n=None, label_ids=[])]
    anchored, merged = asw.anchor_clusters(cs, [(0.0, 0.0)], radius_m=3.0)
    assert [(c.e, c.n) for c in anchored[:2]] == [(0.0, 0.0), (0.0, 0.0)]
    assert anchored[2].e == 3.01 and anchored[3].e is None          # beyond r_a / unplaced
    assert cs[0].e == 0.5                                            # inputs untouched
    assert len(merged) == 3
    m = next(c for c in merged if len(c.members) == 2)
    assert m.id == 4 and m.n_labels == 3 and sorted(m.label_ids) == [1, 2]
    a0, m0 = asw.anchor_clusters(cs, [])                             # no anchors: no-op
    assert [(c.e, c.n) for c in a0] == [(c.e, c.n) for c in cs] and len(m0) == len(cs)


def test_merge_points_is_greedy_on_the_running_mean():
    assert asw.merge_points([(0, 0), (1, 0), (10, 0)], 2.0) == [(0.5, 0.0), (10.0, 0.0)]
    assert asw.merge_points([], 2.0) == []


# ---- provenance helpers

def test_tiles_digest_matches_the_fetchers_manifest_format():
    import hashlib
    rows = [(2, 1, 'bb'), (1, 9, 'aa')]
    want = hashlib.sha256('1,9,aa\n2,1,bb'.encode()).hexdigest()
    assert asw.tiles_digest(rows) == want


def test_install_year_treats_the_1900_placeholder_as_unknown():
    assert asw.install_year({'InstallDate': 1598918400000}) == 2020
    assert asw.install_year({'InstallDate': -2208988800000}) is None
    assert asw.install_year({}) is None


def _aerial_dir(tmp_path, monkeypatch, record):
    import json
    d = tmp_path / 'runs' / 'x' / asw.OUT_NAME
    d.mkdir(parents=True)
    poly = d / 'polygons.geojson'
    poly.write_text('{"type": "FeatureCollection", "features": []}', encoding='utf-8')
    record = {'bbox_wgs84': [0, 0, 1, 1],
              'polygons_geojson_sha256': asw.sha256_file(poly), **record}
    (d / 'tile2net.json').write_text(json.dumps(record), encoding='utf-8')
    monkeypatch.setattr(asw, 'REPO_ROOT', tmp_path)
    return d


def test_load_aerial_refuses_a_missing_or_altered_network(tmp_path, monkeypatch):
    d = _aerial_dir(tmp_path, monkeypatch, {'network_geojson_sha256': '0' * 64})
    with pytest.raises(SystemExit, match='network'):
        asw.load_aerial('x')                                         # recorded, absent
    (d / 'network.geojson').write_text('{"features": []}', encoding='utf-8')
    with pytest.raises(SystemExit, match='does not match'):
        asw.load_aerial('x')                                         # present, wrong bytes


def test_load_aerial_refuses_an_unbound_network(tmp_path, monkeypatch):
    d = _aerial_dir(tmp_path, monkeypatch, {})
    asw.load_aerial('x')                                             # no network: fine
    (d / 'network.geojson').write_text('{"features": []}', encoding='utf-8')
    with pytest.raises(SystemExit, match='records no network'):
        asw.load_aerial('x')
