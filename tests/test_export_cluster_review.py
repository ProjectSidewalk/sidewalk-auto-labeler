"""export_cluster_review.py (RampNet#224): the pre-registered unit sampler on synthetic
OSM payloads, window membership, the corners.jsonl label shape, and the crop worker's
sharded-store path. No network, no pandas: the module's heavy imports are lazy."""
import json
import math
import subprocess
import sys

import pytest

import export_cluster_review as ecr
import geo

LAT0, LNG0 = 45.6, -122.6
FR = geo.LocalFrame(LAT0, LNG0)


def ll(e, n):
    lat, lng = FR.to_latlng(e, n)
    return {'lat': lat, 'lon': lng}


def way(wid, hw, pts, node_ids):
    return {'type': 'way', 'id': wid, 'tags': {'highway': hw}, 'nodes': node_ids,
            'geometry': [ll(*p) for p in pts]}


def node(nid, e, n, **tags):
    d = ll(e, n)
    return {'type': 'node', 'id': nid, 'lat': d['lat'], 'lon': d['lon'], 'tags': tags}


# a residential "+" at (0, 0) [node 2]; a primary with a residential T at (200, 0)
# [node 11] and a second T 10 m further [node 14] (merged with 11); a signal 20 m from
# node 2; a residential "+" at (0, 400) [node 32] with no signal
PAYLOAD = {'elements': [
    way(1, 'residential', [(-100, 0), (0, 0), (100, 0)], [1, 2, 3]),
    way(2, 'residential', [(0, -100), (0, 0), (0, 100)], [4, 2, 5]),
    way(3, 'primary', [(100, 0), (200, 0), (210, 0), (400, 0)], [3, 11, 14, 12]),
    way(4, 'residential', [(200, 0), (200, 80)], [11, 13]),
    way(5, 'residential', [(210, 0), (210, -80)], [14, 15]),
    way(6, 'residential', [(-100, 400), (0, 400), (100, 400)], [31, 32, 33]),
    way(7, 'residential', [(0, 300), (0, 400), (0, 500)], [34, 32, 35]),
    way(8, 'service', [(0, 0), (0, -50)], [2, 99]),          # not a street: ignored
    node(50, 20, 0, highway='traffic_signals'),
]}


def test_street_filter_matches_position_check():
    import position_check
    assert ecr.STREET_HIGHWAY_RE == position_check.STREET_HIGHWAY_RE


def test_intersections_legs_and_types():
    ways = ecr.street_ways(PAYLOAD)
    assert {w['id'] for w in ways} == {1, 2, 3, 4, 5, 6, 7}
    nodes = ecr.intersection_nodes(ways)
    # node 3 joins two way ends plus nothing else: 2 legs, not an intersection
    assert set(nodes) == {2, 11, 14, 32}
    assert nodes[2]['legs'] == 4 and nodes[11]['legs'] == 3
    cands, stats = ecr.build_candidates(PAYLOAD, FR)
    units = {tuple(c['node_ids']): c for c in cands if c['type'] != 'mid_block'}
    assert set(units) == {(2,), (11, 14), (32,)}        # 11 and 14 merged (10 m apart)
    assert units[(2,)]['type'] == 'signalised'           # a signal node within 30 m
    assert units[(11, 14)]['type'] == 'arterial'         # a primary leg
    assert units[(32,)]['type'] == 'residential'
    e, n = FR.to_enu(units[(11, 14)]['lat'], units[(11, 14)]['lng'])
    assert e == pytest.approx(205.0, abs=0.01) and n == pytest.approx(0.0, abs=0.01)
    assert stats['intersection_units'] == 3 and stats['signal_nodes'] == 1


def test_merge_is_single_linkage():
    pts = {k: dict(zip(('lat', 'lng'), FR.to_latlng(e, 0.0)), highways=['residential'])
           for k, e in ((1, 0.0), (2, 20.0), (3, 40.0), (4, 100.0))}
    groups = ecr.merge_nodes(pts, 25.0)
    assert [g['node_ids'] for g in groups] == [[1, 2, 3], [4]]    # chained, not complete


def test_midblock_candidates_every_step_beyond_the_nodes():
    w = way(1, 'residential', [(0, 0), (150, 0), (300, 0)], [1, 2, 3])
    ends = [FR.to_latlng(0, 0), FR.to_latlng(300, 0)]
    pts = ecr.midblock_candidates([w], [(a, b) for a, b in ends], FR, 70.0, 20.0)
    es = [round(FR.to_enu(lat, lng)[0], 6) for lat, lng, _hw in pts]
    assert es == [80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0]


def test_in_area_respects_holes():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
    g = {'type': 'MultiPolygon', 'coordinates': [[sq, hole]]}
    assert ecr.in_area(2, 2, g) and not ecr.in_area(5, 5, g) and not ecr.in_area(20, 2, g)


def test_segment_index_eligibility():
    lines = [[FR.to_latlng(0, 0), FR.to_latlng(100, 0)]]
    idx = ecr.SegmentIndex(FR, lines, 20.0)
    assert idx.near(*FR.to_latlng(50, 19.0), 20.0)
    assert not idx.near(*FR.to_latlng(50, 21.0), 20.0)
    assert not idx.near(*FR.to_latlng(125, 0.0), 20.0)


def _cands(n_lab, n_empty, stratum, step_m=100.0, row=0.0):
    out = []
    for k in range(n_lab + n_empty):
        lat, lng = FR.to_latlng(k * step_m, row)
        out.append({'type': stratum, 'lat': lat, 'lng': lng, 'node_ids': [k],
                    'highways': [], 'label_idx': [k] if k < n_lab else []})
    return out


def test_draw_is_deterministic_spaced_and_keeps_the_empty_share():
    cands = []
    for i, s in enumerate(ecr.STRATA):
        cands += _cands(30, 10, s, row=i * 1000.0)
    a, st = ecr.draw(cands, 'city', per_stratum=20, empty_share=0.15, seed=224)
    b, _ = ecr.draw(cands, 'city', per_stratum=20, empty_share=0.15, seed=224)
    assert [u['corner_id'] for u in a] == [u['corner_id'] for u in b]
    assert [(u['lat'], u['lng']) for u in a] == [(u['lat'], u['lng']) for u in b]
    assert len(a) == 80
    for s in ecr.STRATA:
        us = [u for u in a if u['type'] == s]
        assert len(us) == 20 and sum(1 for u in us if not u['has_labels']) == 3
    c, _ = ecr.draw(cands, 'city', per_stratum=20, empty_share=0.15, seed=225)
    assert {(u['lat'], u['lng']) for u in c} != {(u['lat'], u['lng']) for u in a}
    assert a[0]['corner_id'] == 'city:sig:000001'


def test_draw_spacing_rejects_close_units_across_strata():
    cands = _cands(10, 0, 'signalised', step_m=100.0) + \
        _cands(10, 0, 'arterial', step_m=100.0, row=30.0)      # 30 m from the first row
    units, st = ecr.draw(cands, 'city', per_stratum=10, empty_share=0.0, seed=1)
    for i, u in enumerate(units):
        for v in units[i + 1:]:
            assert geo.haversine_m(u['lat'], u['lng'], v['lat'], v['lng']) >= 60.0
    assert len([u for u in units if u['type'] == 'arterial']) == 0
    assert st['arterial']['labelled_spacing_rejected'] == 10


def test_pilot_quota_and_rater_b_alternation():
    cands = []
    for i, s in enumerate(ecr.STRATA):
        cands += _cands(30, 10, s, row=i * 1000.0)
    units, _ = ecr.draw(cands, 'city', seed=224)
    ecr.mark_pilot(units)
    pilot = [u for u in units if u['pilot']]
    assert len(pilot) == 30
    assert {s: sum(1 for u in pilot if u['type'] == s) for s in ecr.STRATA} == ecr.PILOT_QUOTA
    assert sum(1 for u in pilot if not u['has_labels']) == 4
    order = sorted(pilot, key=lambda u: ecr.sha1_hex(u['corner_id']))
    assert [u['rater_b_seed'] for u in order[:4]] == ['fusion', 'deployed', 'fusion', 'deployed']
    assert all(u['double_rate'] for u in pilot)
    rest = [u for u in units if not u['pilot']]
    assert sum(1 for u in rest if u['double_rate']) == 10       # 20% of 50
    assert all(u['rater_b_seed'] is None for u in rest if not u['double_rate'])


def test_window_membership_by_server_position():
    pts = [FR.to_latlng(29.9, 0.0), FR.to_latlng(30.1, 0.0), FR.to_latlng(0.0, -12.0)]
    idx = ecr.PointIndex(FR, pts, 30.0)
    assert sorted(idx.within(LAT0, LNG0, 30.0)) == [0, 2]


def test_label_entry_shape_and_seed_groups():
    dep, fus = ecr.seed_groups([(7, [101, 102]), (3, [102])], [[101], [102, 103]])
    assert dep[102] == ['d3', 'd7'] and fus[103] == 'f1'
    row = {'label_id': 102, 'pano_id': 'P', 'user_kind': 'ai', 'pano_x': 2048,
           'pano_y': 1100, 'pano_width': 4096, 'pano_height': float('nan'),
           'lat': LAT0, 'lng': LNG0, 'image_capture_date': '2024-05'}
    cam = {'lat': LAT0, 'lng': LNG0, 'heading_deg': 10.0, 'source': 'run'}
    e = ecr.label_entry(row, dep, fus, cam)
    assert e['key'] == '102' and e['x'] == 0.5 and e['y'] is None     # NaN height -> None
    assert e['seed_group'] == {'deployed': 'd3', 'fusion': 'f1'}
    assert e['seed_group_all'] == {'deployed': ['d3', 'd7']}
    assert e['crop'] == 'crops/102.jpg'
    json.dumps(e)   # plain JSON types only
    for k in ('key', 'label_id', 'pano_id', 'user_kind', 'pano_x', 'pano_y', 'pano_width',
              'pano_height', 'x', 'y', 'lat', 'lng', 'camera', 'capture_date', 'crop',
              'seed_group'):
        assert k in e


def test_aerial_window_is_mercator_exact():
    win, bbox = ecr.aerial_window(LAT0, LNG0)
    x, y = ecr.world_px(LAT0, LNG0)
    assert win['x0'] < x < win['x1'] and win['y0'] < y < win['y1']
    # 70 m at z20 and 45.6 deg is ~671 px on each axis
    assert abs((win['x1'] - win['x0']) - 671) <= 3 and abs((win['y1'] - win['y0']) - 671) <= 3
    lat, lng = ecr.world_to_latlng(*ecr.world_px(45.61, -122.61))
    assert lat == pytest.approx(45.61, abs=1e-9) and lng == pytest.approx(-122.61, abs=1e-9)
    assert bbox['north'] > LAT0 > bbox['south'] and bbox['west'] < LNG0 < bbox['east']


def test_crop_worker_reads_a_sharded_store(tmp_path):
    from PIL import Image
    import site_explorer as se
    store = tmp_path / 'store'
    for pid in ('abPANO1', 'cdPANO2'):
        (store / pid[:2]).mkdir(parents=True)
        Image.new('RGB', (512, 256), (40, 90, 140)).save(store / pid[:2] / f'{pid}.jpg')
    items = [{'pano_id': 'abPANO1', 'x': 0.5, 'y': 0.5, 'fov_deg': 45, 'px': 64,
              'name': '1.jpg', 'path': 'ab/abPANO1.jpg'},
             {'pano_id': 'cdPANO2', 'x': 0.99, 'y': 0.6, 'fov_deg': 45, 'px': 64,
              'name': '2.jpg', 'path': 'cd/cdPANO2.jpg'},
             {'pano_id': 'efGONE', 'x': 0.5, 'y': 0.5, 'fov_deg': 45, 'px': 64,
              'name': '3.jpg', 'path': 'ef/efGONE.jpg'}]
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'items': items, 'workers': 1, 'panos_root': str(store),
                               'draft_width': 4096}), encoding='utf-8')
    worker = tmp_path / 'crop.py'
    worker.write_text(se.CROP_WORKER, encoding='utf-8')
    out = tmp_path / 'out'
    proc = subprocess.run([sys.executable, str(worker), str(job), str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in out.glob('*.jpg')) == ['1.jpg', '2.jpg']
    assert (out / '_missing.txt').read_text() == '3.jpg'
    with Image.open(out / '1.jpg') as im:
        assert im.size == (64, 64)
    # the same job cut locally (fetch_crops_local) honours the path too
    local = tmp_path / 'local'
    local.mkdir()
    assert se.fetch_crops_local(json.loads(job.read_text()), local, store) == {'3.jpg'}
    assert sorted(p.name for p in local.glob('*.jpg')) == ['1.jpg', '2.jpg']


def test_crop_items_skip_existing_and_shard(tmp_path):
    corners = [{'labels': [{'key': '5', 'pano_id': 'xyPANO', 'x': 0.1, 'y': 0.5,
                            'crop': 'crops/5.jpg'},
                           {'key': '6', 'pano_id': 'xyPANO', 'x': 0.2, 'y': 0.5,
                            'crop': 'crops/6.jpg'}]}]
    (tmp_path / '6.jpg').write_bytes(b'x')
    items = ecr.crop_items(corners, tmp_path, sharded=True)
    assert items == [{'pano_id': 'xyPANO', 'x': 0.1, 'y': 0.5, 'fov_deg': 45, 'px': 512,
                      'name': '5.jpg', 'path': 'xy/xyPANO.jpg'}]


def test_reconcile_flags_every_gap(tmp_path):
    corners = [{'corner_id': 'c:sig:000001', 'aerial': {'file': 'aerial/c_sig_000001.jpg'},
                'labels': [{'key': '1', 'pano_id': 'P', 'crop': 'crops/1.jpg'},
                           {'key': '2', 'pano_id': 'P', 'crop': 'crops/2.jpg'},
                           {'key': '3', 'pano_id': 'Q', 'crop': 'crops/3.jpg'}]}]
    (tmp_path / 'crops').mkdir()
    (tmp_path / 'crops' / '1.jpg').write_bytes(b'x')
    (tmp_path / 'crops' / '9.jpg').write_bytes(b'x')
    ecr.write_missing(tmp_path / 'crops_missing.csv', {'2': {'pano_id': 'P', 'reason': 'r'}})
    problems, counts = ecr.reconcile(corners, tmp_path)
    assert counts == {'crops': 1, 'crops_missing': 1, 'aerials': 0}
    assert any('no aerial' in p for p in problems)
    assert any('label 3' in p for p in problems)
    assert any('no label' in p for p in problems)
    (tmp_path / 'aerial').mkdir()
    (tmp_path / 'aerial' / 'c_sig_000001.jpg').write_bytes(b'x')
    (tmp_path / 'crops' / '3.jpg').write_bytes(b'x')
    (tmp_path / 'crops' / '9.jpg').unlink()
    assert ecr.reconcile(corners, tmp_path)[0] == []


def test_overpass_query_unions_signals():
    q = ecr.overpass_query((-122.7, 45.5, -122.5, 45.7))
    assert 'node["highway"="traffic_signals"]' in q and 'node["crossing"="traffic_signals"]' in q
    assert q.endswith('out geom;') and ecr.STREET_HIGHWAY_RE in q
    assert ecr.validate_osm({'elements': [], 'remark': 'runtime error'}).startswith('Overpass')
    assert math.isclose(ecr.round_half_up(2.5), 3) and ecr.round_half_up(1.05) == 1


# --- rule 6b (rule_version 2): grade-separated windows ---------------------------------------

def tagged(wid, tags, pts):
    return {'type': 'way', 'id': wid, 'tags': tags, 'nodes': list(range(wid * 10, wid * 10 + len(pts))),
            'geometry': [ll(*p) for p in pts]}


def test_grade_reason_rules():
    assert ecr.grade_reason({'bridge': 'yes'}, False) == 'bridge'
    assert ecr.grade_reason({'bridge': 'no'}, True) is None
    assert ecr.grade_reason({'covered': 'yes'}, False) == 'covered'
    assert ecr.grade_reason({'layer': '1'}, False) == 'layer_above'
    assert ecr.grade_reason({'layer': '1;2'}, False) == 'layer_above'
    assert ecr.grade_reason({'layer': 'x'}, True) is None
    assert ecr.grade_reason({'tunnel': 'yes'}, True) == 'street_tunnel'
    assert ecr.grade_reason({'layer': '-1'}, True) == 'street_below'
    # underground and not a street (a subway, a culvert): nobody's corner is hidden by it
    assert ecr.grade_reason({'tunnel': 'yes', 'railway': 'subway', 'layer': '-2'}, False) is None


def _grade_payload(extra):
    return {'elements': PAYLOAD['elements'] + extra}


def test_grade_rule_excludes_windows_over_structures_only():
    base, st0 = ecr.build_candidates(PAYLOAD, FR)
    assert st0['grade_separated'] == 0
    near_plus = [(-50, 20), (50, 20)]            # 20 m from the "+" at (0, 0)
    far = [(-50, 700), (50, 700)]                # 300 m from everything
    cases = [
        ([tagged(90, {'railway': 'rail', 'bridge': 'yes'}, near_plus)], True),   # overhead rail
        ([tagged(90, {'railway': 'subway', 'tunnel': 'yes', 'layer': '-2'}, near_plus)], False),
        ([tagged(90, {'highway': 'primary', 'tunnel': 'yes', 'layer': '-1'}, near_plus)], True),
        ([tagged(90, {'highway': 'footway', 'bridge': 'yes'}, far)], False),
    ]
    for extra, excluded in cases:
        cands, st = ecr.build_candidates(_grade_payload(extra), FR)
        at_plus = any(abs(c['lat'] - ll(0, 0)['lat']) < 1e-7 and abs(c['lng'] - ll(0, 0)['lon']) < 1e-7
                      for c in cands)
        assert at_plus is not excluded, extra
        assert (st['grade_separated'] > 0) is excluded
        assert sum(st['grade_separated_by_stratum'].values()) == st['grade_separated']
    cands, st = ecr.build_candidates(_grade_payload(cases[0][0]), FR, grade_rule=False)
    assert len(cands) == len(base) and st['grade_separated'] == 0


def test_street_ways_ignore_the_structure_ways():
    extra = [tagged(90, {'railway': 'rail', 'bridge': 'yes'}, [(-50, 20), (50, 20)]),
             tagged(91, {'highway': 'footway', 'bridge': 'yes'}, [(-50, 30), (50, 30)])]
    assert {w['id'] for w in ecr.street_ways(_grade_payload(extra))} == \
        {w['id'] for w in ecr.street_ways(PAYLOAD)}


def test_overpass_query_fetches_rule_6b_structures():
    q = ecr.overpass_query((-122.7, 45.5, -122.5, 45.7))
    for part in ('way["bridge"]["bridge"!="no"]', 'way["covered"]["covered"!="no"]',
                 'way["layer"~"^[+]?[1-9]"]'):
        assert part in q
    assert ecr.RULE_VERSION == 2
