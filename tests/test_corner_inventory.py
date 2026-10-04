"""corner_inventory.py (RampNet#238): legs and corners on synthetic geometry, sector
assignment, the observability radii, states, the inventory-gap read, the #224 assignments
scorer on a synthetic file, the census fetch policy with a stubbed API, and LF byte
stability. No network and no Vancouver file; the end-to-end smoke test on the real run is
behind CORNER238_SMOKE=1."""
import json
import math
import os
from pathlib import Path

import pytest

import corner_inventory as ci
import geo

LAT0, LNG0 = 45.6, -122.6
FR = geo.LocalFrame(LAT0, LNG0)


def ll(e, n):
    return FR.to_latlng(e, n)


def way(wid, hw, pts, node_ids):
    return {'type': 'way', 'id': wid, 'tags': {'highway': hw}, 'nodes': node_ids,
            'geometry': [dict(zip(('lat', 'lon'), ll(*p))) for p in pts]}


# a "+" at node 2 (0, 0); a T at node 12 (300, 0): through street east-west, stem south.
# way 3 is split at node 20 (330, 0), 10 m inside the probe circle of nothing: a way
# split that the leg walk must pass through.
WAYS = [
    way(1, 'residential', [(-100, 0), (0, 0), (100, 0)], [1, 2, 3]),
    way(2, 'residential', [(0, -100), (0, 0), (0, 100)], [4, 2, 5]),
    way(3, 'residential', [(200, 0), (300, 0), (310, 0)], [11, 12, 20]),
    way(4, 'residential', [(310, 0), (400, 0)], [20, 13]),
    way(5, 'residential', [(300, 0), (300, -100)], [12, 14]),
]


def graph():
    adj, pos = ci.street_graph(WAYS)
    return adj, {k: FR.to_enu(*v) for k, v in pos.items()}


def test_cross_has_four_legs_and_quadrants():
    adj, pos = graph()
    legs = ci.merge_bearings(ci.leg_bearings([2], (0.0, 0.0), adj, pos))
    assert [round(b) % 360 for b in legs] == [0, 90, 180, 270]
    secs = ci.corner_sectors(legs)
    assert [round(w) for _, w in secs] == [90, 90, 90, 90]
    # NE point -> sector 0 (0..90), SE -> 1, SW -> 2, NW -> 3
    assert ci.sector_index(ci.bearing_deg(5, 5), secs) == 0
    assert ci.sector_index(ci.bearing_deg(5, -5), secs) == 1
    assert ci.sector_index(ci.bearing_deg(-5, -5), secs) == 2
    assert ci.sector_index(ci.bearing_deg(-5, 5), secs) == 3
    e, n = ci.corner_point((0.0, 0.0), *secs[0])
    assert math.isclose(e, 12 / math.sqrt(2), abs_tol=1e-6)
    assert math.isclose(n, 12 / math.sqrt(2), abs_tol=1e-6)


def test_t_junction_three_legs_one_wide_and_walks_through_split():
    adj, pos = graph()
    legs = ci.merge_bearings(ci.leg_bearings([12], (300.0, 0.0), adj, pos))
    assert [round(b) for b in legs] == [90, 180, 270]
    secs = ci.corner_sectors(legs)
    widths = sorted(round(w) for _, w in secs)
    assert widths == [90, 90, 180]
    assert sum(1 for _, w in secs if w > ci.WIDE_SECTOR_DEG) == 1


def test_merge_bearings_dual_carriageway_and_wraparound():
    assert ci.merge_bearings([350, 10, 90, 180, 270]) == [0.0, 90.0, 180.0, 270.0]
    # 40 deg apart: two legs
    assert len(ci.merge_bearings([0, 40])) == 2
    assert len(ci.merge_bearings([0, 20])) == 1
    assert ci.corner_sectors([]) == [(0.0, 360.0)]


def test_internal_edges_skipped_for_merged_units():
    # two nodes 10 m apart on one street, each with a stem: the edge between them is
    # internal; legs = west, east, north stem, south stem
    ways = [way(1, 'residential', [(-100, 0), (0, 0), (10, 0), (100, 0)], [1, 2, 3, 4]),
            way(2, 'residential', [(0, 0), (0, 100)], [2, 5]),
            way(3, 'residential', [(10, 0), (10, -100)], [3, 6])]
    adj, pos = ci.street_graph(ways)
    pos = {k: FR.to_enu(*v) for k, v in pos.items()}
    legs = ci.merge_bearings(ci.leg_bearings([2, 3], (5.0, 0.0), adj, pos))
    assert len(legs) == 4


def make_idx(panos=(), sites=(), clusters=(), inventory=()):
    def at(items):
        return [(*ll(e, n), p) for e, n, p in items]
    return {'panos': ci.Points(FR, at(panos), 37.0), 'sites': ci.Points(FR, at(sites), 30.0),
            'clusters': ci.Points(FR, at(clusters), 30.0),
            'inventory': ci.Points(FR, at(inventory), 30.0)}


def inv(cls, uid='CR1'):
    status = {'Available': 'Available', 'NA_noramp': 'NA', 'RMV': 'RMV'}[cls]
    props = {'UNITID': uid, 'STATUS': status, 'RAMPTYPE': None if cls == 'NA_noramp' else 'PERP'}
    return ci.inv_record(props)


LEGS4 = [0.0, 90.0, 180.0, 270.0]
UNIT = {'unit': 'u', 'type': 'residential', 'lat': LAT0, 'lng': LNG0, 'node_ids': [2],
        'highways': ['residential']}


def test_observability_radii_and_states():
    # a pano 20 m north of the centre: within 25 m of the unit centre and of the NE / NW
    # corner points, > 15 m of every corner point except the NE / NW ones (dist ~11.7 m)
    idx = make_idx(panos=[(0, 20, 'p1')])
    r = ci.describe_unit(UNIT, (0.0, 0.0), LEGS4, idx, 25.0)
    assert r['n_panos_25'] == 1 and r['n_panos_15'] == 0
    assert r['state']['fusion/primary'] == 'absent'
    assert r['state']['fusion/ge2'] == 'unobservable'
    assert r['state']['fusion/le15'] == 'unobservable'
    ne, se = r['corners'][0], r['corners'][1]
    assert ne['n_panos_25'] == 1 and ne['n_panos_15'] == 1
    # SE corner point (8.5, -8.5): distance to (0, 20) = 29.7 m > 25 m
    assert se['n_panos_25'] == 0 and se['state']['fusion/primary'] == 'unobservable'


def test_unit_with_no_panos_is_unobservable():
    r = ci.describe_unit(UNIT, (0.0, 0.0), LEGS4, make_idx(), 25.0)
    assert r['nearest_pano_m'] is None
    assert all(v == 'unobservable' for v in r['state'].values())
    assert all(c['state']['fusion/primary'] == 'unobservable' for c in r['corners'])


def site(sid, panos=('p1',)):
    return {'site_id': sid, 'op_panos': list(panos)}


def test_present_precedence_and_inventory_gap():
    # a site in the NE sector, no inventory anywhere: present, and an inventory gap;
    # no pano at all, so present-not-observed is counted
    idx = make_idx(sites=[(6, 6, site(7))])
    r = ci.describe_unit(UNIT, (0.0, 0.0), LEGS4, idx, 25.0)
    assert r['state']['fusion/primary'] == 'present'
    assert r['state']['deployed/primary'] == 'unobservable'
    ne = r['corners'][0]
    assert ne['site_ids'] == [7] and ne['site_pano_ids'] == ['p1']
    t = ci.tally([ne], 'fusion/primary')
    assert t['present_no_inventory'] == 1 and t['present_not_observed'] == 1
    gaps = ci.gap_rows([r])
    assert len(gaps) == 1 and gaps[0]['site_ids'] == '7' and gaps[0]['pano_ids'] == 'p1'


def test_absence_precision_buckets():
    idx = make_idx(panos=[(0, 0, 'p0')],
                   inventory=[(6, 6, inv('Available')), (6, -6, inv('NA_noramp')),
                              (-6, -6, inv('RMV'))],
                   clusters=[(6, 6, 99)])
    r = ci.describe_unit(UNIT, (0.0, 0.0), LEGS4, idx, 25.0)
    cs = r['corners']
    assert [c['state']['fusion/primary'] for c in cs] == ['absent'] * 4
    assert cs[0]['state']['deployed/primary'] == 'present'
    t = ci.tally(cs, 'fusion/primary')
    assert (t['absent_clean'], t['absent_false'], t['absent_rmvna_only']) == (1, 1, 2)
    assert (t['n_avail'], t['avail_absent']) == (1, 1)
    assert cs[1]['inv_counts']['NA_noramp'] == 1


def test_decision_rule():
    def rec(state, inv_counts):
        st = {f'{a}/{v}': state for a in ci.ARMS for v in ci.VARIANTS}
        c = {'state': st, 'inv_counts': inv_counts, 'n_panos_25': 1, 'wide': False}
        return {'type': 'residential', 'state': st, 'inv_counts': inv_counts,
                'n_panos_25': 1, 'corners': [c]}
    empty = {k: 0 for k in ci.INV_CLASSES}
    avail = dict(empty, Available=1)
    recs = [rec('absent', empty)] * 9 + [rec('absent', avail)]
    d = ci.decision(ci.score_rows(recs))
    assert d['precision'] == 0.9 and d['outcome'].startswith('PASS')
    recs = [rec('absent', empty)] * 8 + [rec('absent', avail)] * 2
    assert ci.decision(ci.score_rows(recs))['outcome'].startswith('FAIL')


def test_score_assignments_synthetic():
    idx = make_idx(panos=[(0, 0, 'p0')], sites=[(6, 6, site(1))])
    u = ci.describe_unit(dict(UNIT, unit='vancouver:res:000001'), (0.0, 0.0), LEGS4, idx, 25.0)
    u2 = ci.describe_unit(dict(UNIT, unit='vancouver:res:000002'), (0.0, 0.0), LEGS4,
                          make_idx(panos=[(0, 0, 'p0')]), 25.0)
    u2['lat'], u2['lng'] = LAT0, LNG0
    ne_lat, ne_lng = ll(6, 6)
    sw_lat, sw_lng = ll(-6, -6)
    assignments = {'schema': 'rampnet.cluster_review/1', 'rater': 'synthetic', 'role': 'a',
                   'corners': {
                       'vancouver:res:000001': {
                           'labels': {'1': 'r1'},
                           'ramps': {'r1': {'lat': ne_lat, 'lng': ne_lng, 'placed': False}},
                           'uncovered': [{'lat': sw_lat, 'lng': sw_lng, 'unsure': False}],
                           'complete': True},
                       'vancouver:res:000002': {'labels': {}, 'ramps': {}, 'uncovered': [],
                                                'complete': True},
                       'vancouver:res:000003': {'cant_judge': True, 'complete': False,
                                                'cant_judge_reason': 'dark'}}}
    rows, summary = ci.score_assignments([u, u2], assignments)
    unit_rows = [r for r in rows if r['corner'] == '']
    assert [r['rater'] for r in unit_rows] == ['present', 'absent']
    c1 = {r['corner']: r for r in rows if r['unit'] == 'vancouver:res:000001'
          and r['corner'] != ''}
    assert c1[0]['rater'] == 'present' and c1[0]['ours_fusion_primary'] == 'present'
    # SW: the rater's uncovered ramp, we call it absent -> a false absence
    assert c1[2]['rater'] == 'present' and c1[2]['ours_fusion_primary'] == 'absent'
    reads = summary['reads']
    assert reads['fusion/primary/unit']['absence_precision'] == [1, 1]
    assert reads['fusion/primary/corner']['absence_precision'] == [6, 7]
    assert summary['excluded']['not_in_units'] == ['vancouver:res:000003']
    # a can't-judge unit that IS one of ours is excluded as such, not scored
    u3 = dict(u2, unit='vancouver:res:000003')
    _rows, s3 = ci.score_assignments([u, u2, u3], assignments)
    assert s3['excluded']['cant_judge'] == ['vancouver:res:000003']


def test_census_fetch_policy():
    calls = []

    class Api:
        def __init__(self, answers):
            self.answers = list(answers)

        def find_panorama_by_id(self, pid, download_depth=False):
            calls.append(pid)
            a = self.answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return a
    nf = [[], [[[2], [2, 'x']]]]
    raw, status, err, n = ci.fetch_raw('x', Api([nf]), sleep=lambda s: None)
    assert status == 'not_found' and n == 1
    ok = [[], [[[1], [2, 'y']]]]
    raw, status, err, n = ci.fetch_raw('y', Api([OSError('down'), ok]), sleep=lambda s: None)
    assert status == 'ok' and n == 2
    raw, status, err, n = ci.fetch_raw('z', Api([OSError('down')] * 3), sleep=lambda s: None)
    assert status == 'error' and 'down' in err and n == 3


def test_census_sample_is_seeded_and_stratified():
    recs = [{'unit': f'u{s}{i}', 'type': s} for s in ci.INTERSECTION_STRATA for i in range(100)]
    a, b = ci.census_sample(recs), ci.census_sample(list(reversed(recs)))
    assert [r['unit'] for r in a] == [r['unit'] for r in b]
    assert len(a) == 200
    assert {s: sum(1 for r in a if r['type'] == s) for s in ci.INTERSECTION_STRATA} == ci.CENSUS_N


def test_outputs_are_lf_and_byte_stable(tmp_path):
    idx = make_idx(panos=[(0, 20, 'p1')], sites=[(6, 6, site(3))],
                   inventory=[(6, -6, inv('Available'))])
    r = ci.describe_unit(UNIT, (0.0, 0.0), LEGS4, idx, 25.0)
    for k in (1, 2):
        d = tmp_path / str(k)
        ci.write_flat(d, [r])
        ci.write_jsonl_lf(d / 'x.jsonl', [r])
        ci.write_csv_lf(d / 'counts.csv', ci.COUNT_FIELDS, ci.score_rows([r]))
    for name in ('units.csv', 'corners.csv', 'x.jsonl', 'counts.csv'):
        a, b = (tmp_path / '1' / name).read_bytes(), (tmp_path / '2' / name).read_bytes()
        assert a == b and b'\r\n' not in a


@pytest.mark.skipif(os.environ.get('CORNER238_SMOKE') != '1',
                    reason='needs the Vancouver run dir; set CORNER238_SMOKE=1')
def test_smoke_vancouver(tmp_path):
    run = Path(os.environ.get('CORNER238_RUN', '../sal-vancouver/runs/vancouver'))
    osm = Path(os.environ.get('CORNER238_OSM',
                              '../sal-cluster-review/runs/vancouver/cluster_review/osm.json'))
    ci.main(['build', '--run-dir', str(run), '--osm', str(osm), '--out', str(tmp_path)])
    dec = ci.main(['score', '--out', str(tmp_path)])
    assert dec['n_absent'] >= 0
    assert json.loads((tmp_path / 'build.json').read_text())['counts']['units'] > 0


def test_score_assignments_cli_end_to_end(tmp_path):
    idx = make_idx(panos=[(0, 0, 'p0')], sites=[(6, 6, site(1))])
    u = ci.describe_unit(dict(UNIT, unit='vancouver:res:000001'), (0.0, 0.0), LEGS4, idx, 25.0)
    ci.write_jsonl_lf(tmp_path / 'corners_224.jsonl', [u])
    ne_lat, ne_lng = ll(6, 6)
    far_lat, far_lng = ll(0, 40)
    a = {'schema': 'rampnet.cluster_review/1', 'rater': 'synthetic', 'role': 'a',
         'rubric_version': 1, 'snapshot_sha256': 'abc',
         'corners': {'vancouver:res:000001': {
             'labels': {'1': 'r1'}, 'complete': True,
             'uncovered': [{'lat': far_lat, 'lng': far_lng, 'unsure': False}],
             'ramps': {'r1': {'lat': ne_lat, 'lng': ne_lng, 'placed': False}}}}}
    (tmp_path / 'assignments.json').write_text(json.dumps(a), encoding='utf-8')
    (tmp_path / 'snapshot.json').write_text(json.dumps({'labels': {'sha256': 'abc'}}),
                                            encoding='utf-8')
    rows, summary = ci.main(['score-assignments', '--assignments',
                             str(tmp_path / 'assignments.json'), '--out', str(tmp_path),
                             '--snapshot', str(tmp_path / 'snapshot.json')])
    assert (tmp_path / 'assignments_score' / 'rows.csv').exists()
    s = json.loads((tmp_path / 'assignments_score' / 'summary.json').read_text())
    assert s['reads']['fusion/primary/unit']['recall_vs_rater'] == [1, 1]
    assert s['reads']['fusion/primary/corner']['absence_precision'] == [3, 3]
    # the uncovered point 40 m out counts for the unit, and is counted (not lost) at corners
    assert s['outside_window_at_corner_level'] == {'ramps': 0, 'uncovered': 1}
    for bad in (dict(a, schema='nope'), dict(a, rubric_version=2),
                dict(a, snapshot_sha256='other')):
        (tmp_path / 'bad.json').write_text(json.dumps(bad), encoding='utf-8')
        with pytest.raises(SystemExit):
            ci.main(['score-assignments', '--assignments', str(tmp_path / 'bad.json'),
                     '--out', str(tmp_path), '--snapshot', str(tmp_path / 'snapshot.json')])


def test_census_summary_counts_history_and_run_dates():
    rec = {'unit': 'u1', 'type': 'residential', 'pano_ids_25': ['a', 'b'],
           'corners': [{'pano_ids_25': ['a']}, {'pano_ids_25': ['b']}]}
    info = {'a': {'status': 'ok', 'error': None, 'current': '2024-05',
                  'hist': [('h1', '2014-06'), ('h2', '2019-07'), ('b', '2018-01')]},
            'b': {'status': 'not_found', 'error': None, 'current': None, 'hist': []}}
    s = ci.census_summary([rec], info, [rec], run_dates={'a': '2024-05', 'b': '2018-01'})
    u = s['per_unit'][0]
    # captures: 2024-05, 2014-06, 2019-07, 2018-01; b is a run pano, so not "historical"
    assert u['n_captures'] == 4 and u['n_hist'] == 2 and u['n_captures_gsv'] == 4
    assert u['earliest'] == 2014 and u['span_years'] == 10
    assert s['n_hist_distinct'] == 2 and s['status'] == {'ok': 1, 'not_found': 1}
    assert s['units_with_panos_none_served'] == 0


def tagged(ways, nodes, centre):
    adj, pos = ci.street_graph(ways)
    pos = {k: FR.to_enu(*v) for k, v in pos.items()}
    return ci.leg_bearings(nodes, centre, adj, pos, edge_tags=ci.street_edge_tags(ways))


def test_divided_arterial_cross_has_four_corners_not_six():
    # an east-west divided road (two oneway carriageways 14 m apart, same name) crossing a
    # north-south street: unit nodes A (0, 7) and B (0, -7), merged; centre (0, 0)
    def w(wid, hw, pts, ids, **tags):
        d = way(wid, hw, pts, ids)
        d['tags'].update(tags)
        return d
    ways = [w(1, 'secondary', [(-100, 7), (0, 7), (100, 7)], [11, 1, 12], name='Main',
              oneway='yes'),
            w(2, 'secondary', [(100, -7), (0, -7), (-100, -7)], [13, 2, 14], name='Main',
              oneway='yes'),
            w(3, 'residential', [(0, 100), (0, 7), (0, -7), (0, -100)], [15, 1, 2, 16],
              name='Cross')]
    walks = tagged(ways, [1, 2], (0.0, 0.0))
    assert len(walks) == 6
    # the original rule keeps the carriageways apart (about 41 deg at the 20 m probe)
    assert len(ci.merge_bearings([x['bearing'] for x in walks])) == 6
    legs = ci.merge_legs(walks)
    assert [round(b) for b in legs] == [0, 90, 180, 270]
    assert len(ci.corner_sectors(legs)) == 4


def test_slip_lane_joins_its_parent_leg():
    walks = [{'bearing': b, 'highway': hw, 'name': '', 'ref': '', 'oneway': hw.endswith('_link')}
             for b, hw in ((0, 'primary'), (50, 'primary_link'), (90, 'primary'),
                           (180, 'primary'), (270, 'primary'))]
    assert len(ci.merge_legs(walks)) == 4


def test_internal_path_through_shape_node_is_not_a_leg():
    # unit nodes 2 (-6, 0) and 3 (6, 0) joined by a way through shape node 9 (0, 4);
    # real legs: SW, N from node 2; NE, S from node 3
    ways = [way(1, 'residential', [(-60, -80), (-6, 0)], [1, 2]),
            way(2, 'residential', [(-6, 0), (0, 4), (6, 0)], [2, 9, 3]),
            way(3, 'residential', [(6, 0), (66, 80)], [3, 4]),
            way(4, 'residential', [(-6, 0), (-6, 100)], [2, 5]),
            way(5, 'residential', [(6, 0), (6, -100)], [3, 6])]
    walks = tagged(ways, [2, 3], (0.0, 0.0))
    assert len(walks) == 4
    assert len(ci.merge_legs(walks)) == 4
