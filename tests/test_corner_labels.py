"""scripts/corner_labels.py (issue #169) on synthetic corners: the NA-read rule, pano choice
(range, the opt-in view test, newest date), the projection round trip through geo, the exported record
and its PS payload, the registry's rule-produced NoCurbRamp spec, the ranked-CSV join, the
seeded sample, and the pre-registered score. CPU only, no network, no torch."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

import corner_inventory as ci
import corner_labels as cl
import geo
import send_to_ps
from detectors import NADIR_MASK_Y, ps_pixel
from detectors import registry

LAT0, LNG0 = 45.6, -122.6
EMPTY = {'Available': 0, 'NA_noramp': 0, 'NA_typed': 0, 'RMV': 0, 'Expired/Removed': 0,
         'other': 0}
STATE_KEYS = [f'{a}/{v}' for a in ci.ARMS for v in ci.VARIANTS]


def at(e, n, lat0=LAT0, lng0=LNG0):
    """(lat, lng) e metres east and n metres north of (lat0, lng0)."""
    return geo.LocalFrame(lat0, lng0).to_latlng(e, n)


def pano(heading, e=0.0, n=0.0, date='2020-01', run='base'):
    lat, lng = at(e, n)
    return {'lat': lat, 'lng': lng, 'heading': heading, 'capture_date': date,
            'width': 16384, 'height': 8192, 'source': 'gsv', 'run': run}


def args(**kw):
    base = dict(arm='fusion', variant='primary', na_read='discount', max_range_m=25.0,
                min_view_m=3.0, top_k=3, camera_height_m=2.5, hemisphere_deg=None)
    base.update(kw)
    return SimpleNamespace(**base)


def corner(idx, e, n, state='absent', inv=None, panos=()):
    lat, lng = at(e, n)
    return {'corner': idx, 'lat': lat, 'lng': lng, 'start_deg': 0.0, 'width_deg': 90.0,
            'wide': False, 'state': dict.fromkeys(STATE_KEYS, state),
            'inv_counts': dict(EMPTY, **(inv or {})), 'inventory': [],
            'pano_ids_25': list(panos), 'n_panos_25': len(panos), 'n_panos_15': len(panos),
            'nearest_pano_m': 5.0}


class NoSites:
    def nearest(self, lat, lng):
        return None


# ------------------------------------------------------------------------ the rule

@pytest.mark.parametrize('inv, discount, count', [
    ({}, False, False),
    ({'NA_noramp': 3}, False, True),
    ({'NA_typed': 1}, True, True),
    ({'Available': 1}, True, True),
    ({'RMV': 1}, True, True),
    ({'Expired/Removed': 1, 'NA_noramp': 1}, True, True),
    ({'other': 1}, True, True),
])
def test_inventory_excludes_under_both_reads(inv, discount, count):
    counts = dict(EMPTY, **inv)
    assert cl.inventory_excludes(counts, 'discount') is discount
    assert cl.inventory_excludes(counts, 'count') is count


def test_inventory_excludes_refuses_an_unknown_read():
    with pytest.raises(ValueError):
        cl.inventory_excludes(EMPTY, 'clean')


def test_na_read_is_required_on_the_cli():
    with pytest.raises(SystemExit):
        cl.main(['build', '--corners', 'x', '--run-dir', 'y', '--out', 'z'])


# ---------------------------------------------------------------------- pano choice

def _view_panos():
    return {
        'ok': pano(0.0),                     # at the origin facing north: dead ahead
        'side': pano(90.0, e=0, n=0),        # facing east: point at 90 deg
        'behind': pano(180.0),               # facing south: point dead behind
        'near': pano(0.0, n=8.5),            # 1.5 m away
        'far': pano(0.0, n=-20.0),           # 30 m away
        'nopose': dict(pano(0.0), heading=None),
    }


def test_candidate_views_range_and_no_view_test_by_default():
    """Since 2026-11-05 a pano facing away is a candidate: a GSV equirect sees behind the car
    as well as ahead, and PS's pixel key wraps x."""
    lat, lng = at(0, 10)               # the corner point, 10 m north of the origin
    panos = _view_panos()
    cands, rej = cl.candidate_views(lat, lng, list(panos) + ['missing'], panos)
    assert sorted(c['pano_id'] for c in cands) == ['behind', 'ok', 'side']
    assert rej == {'no_pose': 2, 'too_far': 1, 'too_near': 1, 'behind': 0}
    ok = next(c for c in cands if c['pano_id'] == 'ok')
    assert ok['dist_m'] == pytest.approx(10.0, abs=1e-6) and ok['off_axis_deg'] < 1e-6
    back = next(c for c in cands if c['pano_id'] == 'behind')
    assert back['off_axis_deg'] == pytest.approx(180.0, abs=1e-6)
    # ...and it projects: the point lands on the seam, x 0 or 1 (PS wraps it modulo width)
    proj = cl.project(panos['behind'], lat, lng)
    assert min(proj.x_norm, 1.0 - proj.x_norm) == pytest.approx(0.0, abs=1e-6)


def test_candidate_views_opt_in_hemisphere_rejects_a_rear_pano():
    lat, lng = at(0, 10)
    panos = _view_panos()
    cands, rej = cl.candidate_views(lat, lng, list(panos), panos, hemisphere_deg=90.0)
    assert sorted(c['pano_id'] for c in cands) == ['ok', 'side']      # 90 deg is still forward
    assert rej['behind'] == 1
    cands, rej = cl.candidate_views(lat, lng, list(panos), panos, hemisphere_deg=45.0)
    assert [c['pano_id'] for c in cands] == ['ok'] and rej['behind'] == 2


def test_build_refuses_a_bad_hemisphere():
    with pytest.raises(SystemExit, match='hemisphere'):
        cl.cmd_build(SimpleNamespace(max_range_m=25.0, hemisphere_deg=0.0))


def test_choose_pano_prefers_newest_among_top_k():
    c = [{'pano_id': 'a', 'dist_m': 4.0, 'capture_date': '2014-08'},
         {'pano_id': 'b', 'dist_m': 9.0, 'capture_date': '2014-08'},
         {'pano_id': 'c', 'dist_m': 12.0, 'capture_date': '2022-07'},
         {'pano_id': 'd', 'dist_m': 20.0, 'capture_date': '2024-01'}]
    chosen, runners = cl.choose_pano(c, top_k=3)
    assert chosen['pano_id'] == 'c'                    # newest of a, b, c; d is not top-3
    assert [r['pano_id'] for r in runners] == ['a', 'b', 'd']
    assert cl.choose_pano(c, top_k=1)[0]['pano_id'] == 'a'
    assert cl.choose_pano(c, top_k=4)[0]['pano_id'] == 'd'
    # same date: nearest wins
    same = [dict(x, capture_date='2020-01') for x in c]
    assert cl.choose_pano(same)[0]['pano_id'] == 'a'
    assert cl.choose_pano([]) == (None, [])


def test_min_view_excludes_near_panos_from_the_choice():
    lat, lng = at(0, 10)
    panos = {'near': pano(0.0, n=8.0, date='2024-01'), 'far': pano(0.0, date='2010-01')}
    cands, rej = cl.candidate_views(lat, lng, panos, panos, min_view_m=3.0)
    assert cl.choose_pano(cands)[0]['pano_id'] == 'far' and rej['too_near'] == 1


# ----------------------------------------------------------------------- projection

@pytest.mark.parametrize('heading, x, y', [(0.0, 0.5, 0.6), (123.4, 0.31, 0.55),
                                           (300.0, 0.7, 0.58)])
def test_projection_round_trips_through_geo(heading, x, y):
    p = pano(heading, e=3.0, n=-2.0)
    pose = geo.pano_pose({'lat': p['lat'], 'lng': p['lng'], 'camera_heading': heading,
                          'camera_pitch': None, 'camera_roll': None, 'source': 'gsv'})
    g = geo.detection_ground_point(pose, x, y, camera_height=2.5, apply_pose=False)
    proj = cl.project(p, g.lat, g.lng, camera_height_m=2.5)
    assert (proj.x_norm, proj.y_norm) == (pytest.approx(x, abs=1e-9), pytest.approx(y, abs=1e-9))


# ----------------------------------------------------------------------------- build

def unit(corners, typ='residential', name='vancouver:res:n1'):
    return {'unit': name, 'type': typ, 'corners': corners}


def test_build_proposals_counts_every_reason():
    panos = {'p': pano(0.0)}
    units = [unit([
        corner(0, 0, 10, panos=['p']),                                  # proposed
        corner(1, 0, 12, inv={'NA_noramp': 1}, panos=['p']),            # proposed under discount
        corner(2, 0, 14, inv={'Available': 1}, panos=['p']),            # excluded
        corner(3, 0, -10, panos=['p']),                                 # behind: proposed
        corner(5, 0, 40, panos=['p']),                                  # 40 m -> no candidate
        corner(4, 5, 10, state='present', panos=['p']),                 # not absent
    ]), unit([corner(0, 0, 8, panos=['p'])], typ='mid_block', name='vancouver:mid:m000001')]
    props, unl, counts = cl.build_proposals(units, panos, NoSites(), args())
    assert [p['key'] for p in props] == ['vancouver:res:n1#0', 'vancouver:res:n1#1',
                                         'vancouver:res:n1#3']
    assert counts['residential'] == {'corners': 6, 'absent': 5, 'excluded_inventory': 1,
                                     'dropped_no_sidewalk': 0, 'no_candidate': 1,
                                     'dropped_projection': 0, 'proposed': 3}
    assert [(r['corner'], r['reason']) for r in unl] == [(2, 'inventory'), (5, 'no_candidate')]
    p = props[0]
    assert p['pano']['pano_id'] == 'p' and p['label_type'] == 'NoCurbRamp'
    assert p['x_normalized'] == pytest.approx(0.5, abs=1e-6)
    assert p['source']['inventory']['status'] == 'none'
    assert p['rule']['na_read'] == 'discount'

    props, _unl, counts = cl.build_proposals(units, panos, NoSites(), args(na_read='count'))
    assert [p['key'] for p in props] == ['vancouver:res:n1#0', 'vancouver:res:n1#3']
    assert counts['residential']['excluded_inventory'] == 2

    # the opt-in forward hemisphere turns the rear corner back into no_candidate
    props, unl, counts = cl.build_proposals(units, panos, NoSites(), args(hemisphere_deg=90.0))
    assert [p['corner'] for p in props] == [0, 1]
    assert [(r['corner'], r['reason']) for r in unl] == [(2, 'inventory'), (3, 'no_candidate'),
                                                         (5, 'no_candidate')]


def test_build_proposals_applies_a_sidewalk_layer():
    class Walks:
        def has_sidewalk(self, lat, lng):
            return lat < at(0, 11)[0]
    panos = {'p': pano(0.0)}
    units = [unit([corner(0, 0, 10, panos=['p']), corner(1, 0, 12, panos=['p'])])]
    props, unl, counts = cl.build_proposals(units, panos, NoSites(), args(), Walks())
    assert [p['corner'] for p in props] == [0]
    assert unl[0]['reason'] == 'no_sidewalk' and counts['residential']['dropped_no_sidewalk'] == 1


def test_sidewalk_presence_reads_geojson(tmp_path):
    a, b = at(-20, 5), at(20, 5)          # an east-west sidewalk 5 m north of the origin
    gj = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {},
          'geometry': {'type': 'LineString', 'coordinates': [[a[1], a[0]], [b[1], b[0]]]}}]}
    path = tmp_path / 'walks.geojson'
    path.write_text(json.dumps(gj), encoding='utf-8')
    sp = cl.SidewalkPresence(path, radius_m=10.0)
    assert sp.has_sidewalk(*at(0, 12)) and not sp.has_sidewalk(*at(0, 20))


def test_sidewalk_presence_keeps_only_the_named_statuses(tmp_path):
    """A city layer that digitizes street sides without a sidewalk (Vancouver WA STATUS 'NA')
    must not count those as sidewalk: --sidewalk-keep filters them out."""
    def line(n, status):
        a, b = at(-20, n), at(20, n)
        return {'type': 'Feature', 'properties': {'STATUS': status},
                'geometry': {'type': 'LineString', 'coordinates': [[a[1], a[0]], [b[1], b[0]]]}}
    gj = {'type': 'FeatureCollection', 'features': [line(5, 'Available'), line(40, 'NA')]}
    path = tmp_path / 'walks.geojson'
    path.write_text(json.dumps(gj), encoding='utf-8')
    assert cl.SidewalkPresence(path).has_sidewalk(*at(0, 40))
    keep = cl.parse_keep(['STATUS=Available,Pathway'])
    sp = cl.SidewalkPresence(path, keep=keep)
    assert (sp.n_features_in_file, sp.n_features) == (2, 1)
    assert sp.has_sidewalk(*at(0, 5)) and not sp.has_sidewalk(*at(0, 40))
    with pytest.raises(SystemExit, match='no sidewalk geometry'):
        cl.SidewalkPresence(path, keep={'STATUS': {'RMV'}})
    with pytest.raises(SystemExit, match='FIELD=V1,V2'):
        cl.parse_keep(['STATUS'])


# ---------------------------------------------------------------------------- export

PROV = {'model_id': 'corner-rule@0123456789ab', 'model_training_date': '11-05-2026',
        'api_version': '1.0.0', 'model_repo': cl.RULE_REPO, 'model_revision': '0' * 40}


def block(pid, heading=0.0):
    p = pano(heading)
    return {'panorama_id': pid, 'lat': p['lat'], 'lng': p['lng'], 'camera_heading': heading,
            'width': 16384, 'height': 8192, 'source': 'launch', 'links': [], 'history': []}


def proposal(key, pid, x, y, rank=None):
    u, c = key.split('#')
    return {'key': key, 'unit': u, 'corner': int(c), 'stratum': 'residential',
            'pano': {'pano_id': pid, 'width': 16384, 'height': 8192},
            'x_normalized': x, 'y_normalized': y, 'rank': rank, 'source': {'k': key}}


def test_export_one_line_per_pano_and_its_payload():
    props = [proposal('u1#1', 'P', 0.4, 0.6), proposal('u1#0', 'P', 0.6, 0.8),
             proposal('u2#0', 'Q', 0.5, 0.55)]
    lines = cl.export_lines(props, {(None, 'P'): block('P'), (None, 'Q'): block('Q')}, PROV)
    assert [ln['pano']['panorama_id'] for ln in lines] == ['P', 'Q']
    line = lines[0]
    assert line['label_type'] == 'NoCurbRamp' and line['label_types'] == ['NoCurbRamp']
    assert line['models']['NoCurbRamp']['model_training_date'] == '11-05-2026'
    assert line['models']['NoCurbRamp']['model_id'].startswith('corner-rule@')
    assert [d['corner'] for d in line['detections']] == ['u1#0', 'u1#1']
    assert all(d['label_type'] == 'NoCurbRamp' and d['confidence'] == 1.0
               for d in line['detections'])
    assert registry.record_types(line) == ['NoCurbRamp']

    parts = send_to_ps.split_record(line)
    assert [t for t, _ in parts] == ['NoCurbRamp']
    payload = parts[0][1]
    assert payload['label_type'] == 'NoCurbRamp'
    assert payload['model_training_date'] == '11-05-2026'
    # y 0.8 is past the nadir mask; the rule's spec does not mask (mask_rig False)
    assert 0.8 > NADIR_MASK_Y
    assert payload['labels'] == [
        {'pano_x': ps_pixel(0.6, 0.8, 16384, 8192)[0], 'pano_y': ps_pixel(0.6, 0.8, 16384, 8192)[1],
         'confidence': 1.0},
        {'pano_x': ps_pixel(0.4, 0.6, 16384, 8192)[0], 'pano_y': ps_pixel(0.4, 0.6, 16384, 8192)[1],
         'confidence': 1.0}]
    assert 'source' not in json.dumps(payload['labels'])
    assert send_to_ps.transform_record(line, label_type='NoCurbRamp')['labels'] == payload['labels']


def test_export_orders_lines_by_best_rank():
    props = [proposal('a#0', 'P', .5, .6, rank=None), proposal('b#0', 'Q', .5, .6, rank=2),
             proposal('c#0', 'R', .5, .6, rank=1)]
    lines = cl.export_lines(props, {(None, k): block(k) for k in 'PQR'}, PROV)
    assert [ln['pano']['panorama_id'] for ln in lines] == ['R', 'Q', 'P']
    assert [ln['detections'][0]['rank'] for ln in lines] == [1, 2, None]


def test_rule_provenance_needs_a_commit():
    good = {'tool': {'git_sha': 'a' * 40}, 'rule_version_date': '2026-11-05'}
    p = cl.rule_provenance(good)
    assert p['model_id'] == 'corner-rule@' + 'a' * 12 and p['model_training_date'] == '11-05-2026'
    assert set(p) == set(registry.PROVENANCE_KEYS)
    with pytest.raises(SystemExit):
        cl.rule_provenance({'tool': {'git_sha': 'unknown'}})


# -------------------------------------------------------------------------- registry

def test_registry_no_curb_ramp_is_rule_produced():
    s = registry.spec_for('NoCurbRamp')
    assert s.module is None and s.mask_rig is False
    assert s.operational_confidence <= cl.CONFIDENCE
    assert registry.parse_types('CurbRamp,NoCurbRamp') == ['CurbRamp', 'NoCurbRamp']
    with pytest.raises(registry.NoDetectorError, match='scripts/corner_labels.py'):
        registry.load_detectors(['NoCurbRamp'])
    with pytest.raises(registry.NoDetectorError):
        registry.load_detectors(['CurbRamp', 'NoCurbRamp'])   # refused before any import


# ------------------------------------------------------------------------------ rank

def test_rank_joins_on_the_ps_pixel(tmp_path):
    props = [proposal('a#0', 'P', 0.25, 0.6), proposal('b#0', 'P', 0.75, 0.55),
             proposal('c#0', 'Q', 0.5, 0.6)]
    xa, ya = ps_pixel(0.25, 0.6, 16384, 8192)
    xb, yb = ps_pixel(0.75, 0.55, 16384, 8192)
    csv_path = tmp_path / 'ranked.csv'
    csv_path.write_text(f'pano_id,pano_x,pano_y,score\nP,{xa},{ya},0.2\nP,{xb},{yb},0.9\n'
                        f'Q,1,1,0.5\n', encoding='utf-8')
    ranked = cl.rank_proposals(props, cl.read_ranked_csv(csv_path))
    assert [(p['key'], p['rank'], p['rank_score']) for p in ranked] == [
        ('a#0', 2, 0.2), ('b#0', 1, 0.9), ('c#0', None, None)]


def test_ranked_csv_refuses_conflicting_rows(tmp_path):
    p = tmp_path / 'r.csv'
    p.write_text('pid,x,y,s\nP,1,2,0.5\nP,1,2,0.6\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        cl.read_ranked_csv(p, 'pid', 'x', 'y', 's')
    with pytest.raises(SystemExit):
        cl.read_ranked_csv(p)                       # default columns missing


# ----------------------------------------------------------------------- sample/score

def test_sample_is_seeded_and_refuses_a_short_population():
    props = [{'key': f'u{i}#0'} for i in range(150)]
    a, b = cl.draw_sample(props, 100, 169), cl.draw_sample(list(reversed(props)), 100, 169)
    assert a == b and len(set(a)) == 100
    with pytest.raises(SystemExit):
        cl.draw_sample(props[:50], 100, 169)


def _verdicts(rows):
    units = {}
    for u, c, v, kind in rows:
        units.setdefault(u, {'blind': {}})['blind'][str(c)] = {'verdict': v, 'absent_kind': kind}
    return {'units': units}


def test_score_not_scored_below_100_or_without_a_drawn_sample():
    props = [{'key': f'u{i}#0', 'unit': f'u{i}', 'corner': 0, 'stratum': 'residential'}
             for i in range(10)]
    v = _verdicts([(f'u{i}', 0, 'absent', None) for i in range(8)] +
                  [('u8', 0, 'cant_tell', None)])
    res = cl.score(props, v)
    assert res['n'] == 8 and res['cant_tell'] == 1 and res['n_rated'] == 9
    assert res['verdict'].startswith('NOT SCORED') and 'sample not drawn' in res['verdict']
    res = cl.score(props, v, sample_keys=[p['key'] for p in props])
    assert res['verdict'].startswith('NOT SCORED (n = 8 < 100)')


def test_score_wilson_on_a_drawn_100_corner_sample():
    props = [{'key': f'u{i}#0', 'unit': f'u{i}', 'corner': 0, 'stratum': 'arterial'}
             for i in range(110)]
    rows = ([(f'u{i}', 0, 'absent', 'curb_no_ramp') for i in range(93)] +
            [(f'u{i}', 0, 'absent', 'no_sidewalk') for i in range(93, 95)] +
            [(f'u{i}', 0, 'present', None) for i in range(95, 100)] +
            [(f'u{i}', 0, 'cant_tell', None) for i in range(100, 110)])
    res = cl.score(props, _verdicts(rows), sample_keys=[p['key'] for p in props])
    lo, hi = ci.wilson(95, 100)
    assert res['n'] == 100 and res['precision'] == 0.95
    assert res['wilson95'] == [round(lo, 4), round(hi, 4)]
    assert res['wilson95'][0] == pytest.approx(0.8882, abs=1e-4)
    assert res['cant_tell_share'] == round(10 / 110, 4)
    assert res['label_precision'] == 0.93 and res['absent_kinds']['no_sidewalk'] == 2
    assert res['verdict'].startswith('PASS')
    rows[0:10] = [(f'u{i}', 0, 'present', None) for i in range(10)]
    res = cl.score(props, _verdicts(rows), sample_keys=[p['key'] for p in props])
    assert res['precision'] == 0.85 and res['verdict'].startswith('FAIL')


# ------------------------------------------------------------------ end to end (tmp)

def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_build_export_and_dry_run_end_to_end(tmp_path):
    run = tmp_path / 'run'
    _write_jsonl(run / 'results.jsonl', [
        {'detections': [], 'label_type': 'CurbRamp', 'pano': block('P')},
        {'detections': [], 'label_type': 'CurbRamp', 'pano': block('Q', heading=180.0)}])
    _write_jsonl(run / 'sites.jsonl', [{'site_id': 7, 'lat': at(3, 3)[0], 'lng': at(3, 3)[1],
                                        'n_operational': 1,
                                        'members': [{'pano_id': 'P', 'confidence': 0.9}]}])
    (run / 'sites_meta.json').write_text(json.dumps({'params': {'min_confidence': 0.3}}),
                                         encoding='utf-8')
    _write_jsonl(tmp_path / 'corners_full.jsonl', [
        unit([corner(0, 0, 10, panos=['P', 'Q']), corner(1, 6, 10, panos=['P', 'Q'])])])
    out = tmp_path / 'out'
    with redirect_stdout(io.StringIO()):
        cl.main(['build', '--city', 'testcity', '--na-read', 'discount',
                 '--corners', str(tmp_path / 'corners_full.jsonl'), '--run-dir', str(run),
                 '--out', str(out)])
    build = json.loads((out / 'build.json').read_text(encoding='utf-8'))
    assert build['counts']['proposed'] == 2 and build['sidewalk_rule'] == 'not applied'
    assert build['outputs']['proposals']['lines'] == 2 and build['pins'] == 'unpinned city'
    props = ci.read_jsonl(out / 'proposals.jsonl')
    assert build['params']['hemisphere_deg'] is None
    # P and Q stand at the same spot; Q faces away and is still a candidate (tie -> P)
    assert {p['pano']['pano_id'] for p in props} == {'P'}
    assert all(p['source']['n_candidates'] == 2 for p in props)
    assert props[0]['source']['runner_ups'][0]['pano_id'] == "Q"
    assert props[0]['source']['nearest_site']['site_id'] == '7'
    assert (out / 'proposals.jsonl').read_bytes().count(b'\r') == 0

    # with a sidewalk layer: the file's sha256 and the keep filter land in build.json
    a, b = at(-20, 10), at(-1, 10)       # reaches corner 0 (0, 10), not corner 1 (6, 10)
    walks = tmp_path / 'walks.geojson'
    walks.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'STATUS': s},
         'geometry': {'type': 'LineString', 'coordinates': [[a[1], a[0]], [b[1] + d, b[0]]]}}
        for s, d in (('Available', 0.0), ('NA', 0.001))]}), encoding='utf-8')
    out_sw = tmp_path / 'out_sw'
    with redirect_stdout(io.StringIO()):
        cl.main(['build', '--city', 'testcity', '--na-read', 'discount',
                 '--corners', str(tmp_path / 'corners_full.jsonl'), '--run-dir', str(run),
                 '--out', str(out_sw), '--sidewalk-presence', str(walks),
                 '--sidewalk-radius-m', '3', '--sidewalk-keep', 'STATUS=Available',
                 '--na-read-settled'])
    bsw = json.loads((out_sw / 'build.json').read_text(encoding='utf-8'))
    assert build['params']['na_read_settled'] is False and 'na_read_settlement' not in build['params']
    assert bsw['params']['na_read_settled'] is True and bsw['params']['na_read_is_primary'] is True
    assert bsw['params']['na_read_settlement']['score_json_sha256'] ==         cl.NA_READ_SETTLEMENT['score_json_sha256']
    rule = bsw['sidewalk_rule']
    assert rule['sha256'] == bsw['inputs']['sidewalk_presence']['sha256'] == ci.sha256_file(walks)
    assert rule['keep'] == {'STATUS': ['Available']}
    assert (rule['n_features_in_file'], rule['n_features']) == (2, 1)
    assert bsw['counts']['proposed'] == 1 and bsw['counts']['dropped_no_sidewalk'] == 1

    build['tool']['git_sha'] = 'b' * 40          # the test tree may be uncommitted
    build['tool']['dirty'] = False
    (out / 'build.json').write_text(json.dumps(build), encoding='utf-8')
    with redirect_stdout(io.StringIO()):
        cl.main(['export', '--dir', str(out)])
    lines = ci.read_jsonl(out / 'results.jsonl')
    assert len(lines) == 1 and len(lines[0]['detections']) == 2
    assert send_to_ps.file_label_types(out / 'results.jsonl') == ['NoCurbRamp']
    buf = io.StringIO()
    with redirect_stdout(buf):
        send_to_ps.process_jsonl_file(str(out / 'results.jsonl'), 'https://ps.example/ai/x',
                                      dry_run=True)
    assert '"label_type": "NoCurbRamp"' in buf.getvalue() or 'NoCurbRamp' in buf.getvalue()

    # an edited proposals file is refused
    with open(out / 'proposals.jsonl', 'a', encoding='utf-8') as f:
        f.write('{}\n')
    with pytest.raises(SystemExit, match='rebuild'):
        cl.main(['export', '--dir', str(out)])


def test_build_refuses_a_range_past_the_table():
    with pytest.raises(SystemExit, match='pano_ids_25'):
        cl.cmd_build(SimpleNamespace(max_range_m=30.0))


def test_check_pins():
    rec = {r: {'sha256': s[0] if isinstance(s, tuple) else s}
           for r, s in cl.PINNED_SHA256['vancouver'].items()}
    assert cl.check_pins('vancouver', rec, False) == 'pinned'
    # the #169 revision: the --split-wide corners_full is pinned beside the #241 one
    split = dict(rec, corners_full={'sha256': cl.CORNERS_FULL_SPLIT})
    assert cl.check_pins('vancouver', split, False) == 'pinned'
    assert cl.PINNED_SHA256['vancouver']['corners_full'] == (cl.CORNERS_FULL_241,
                                                            cl.CORNERS_FULL_SPLIT)
    other = dict(rec, corners_full={'sha256': '1' * 64})
    with pytest.raises(SystemExit, match='corners_full'):
        cl.check_pins('vancouver', other, False)
    bad = dict(rec, results={'sha256': '0' * 64})
    with pytest.raises(SystemExit, match='results'):
        cl.check_pins('vancouver', bad, False)
    assert cl.check_pins('vancouver', bad, True).startswith('overridden')


def test_tool_version_names_the_repo_head():
    """Review of #177: provenance names HEAD (the whole tree that built the labels), not the
    last commit touching corner_labels.py."""
    import subprocess
    want = subprocess.run(['git', '-C', str(cl.REPO_ROOT), 'log', '-1', '--format=%H'],
                          capture_output=True, text=True).stdout.strip()
    tv = cl.tool_version()
    assert tv['file'] == 'scripts/corner_labels.py' and tv['git_sha'] == want


def _two_run_setup(tmp_path):
    """Pano P in run A (width 8192) and run B (width 16384); corners seen only from P."""
    runs = {}
    for name, width in (('runA', 8192), ('runB', 16384)):
        d = tmp_path / name
        b = dict(block('P'), width=width, height=width // 2)
        if name == 'runB':
            b = dict(b, panorama_id='P')
        _write_jsonl(d / 'results.jsonl', [{'detections': [], 'label_type': 'CurbRamp',
                                            'pano': b}])
        _write_jsonl(d / 'sites.jsonl', [])
        (d / 'sites_meta.json').write_text(json.dumps({'params': {'min_confidence': 0.3}}),
                                           encoding='utf-8')
        runs[name] = d
    return runs


def _fake_build(out, inputs, proposals_path):
    rec = cl.file_record(proposals_path)
    build = {'schema': cl.BUILD_SCHEMA, 'tool': {'git_sha': 'c' * 40, 'dirty': False},
             'rule_version_date': '2026-11-05',
             'inputs': {role: {'path': str(path), 'sha256': ci.sha256_file(path)}
                        for role, path in inputs.items()},
             'outputs': {'proposals': rec}}
    (out / 'build.json').write_text(json.dumps(build), encoding='utf-8')
    return build


def test_export_takes_the_pano_block_from_the_proposals_run(tmp_path):
    runs = _two_run_setup(tmp_path)
    out = tmp_path / 'out'
    prop = dict(proposal('u#0', 'P', 0.25, 0.6), pano={'pano_id': 'P', 'run': 'runB',
                                                        'width': 16384, 'height': 8192})
    _write_jsonl(out / 'proposals.jsonl', [prop])
    # base run is runA (role `results`); runB is an extra -- first-wins would pick runA
    _fake_build(out, {'results': runs['runA'] / 'results.jsonl',
                      'extra:runB:results': runs['runB'] / 'results.jsonl'},
                out / 'proposals.jsonl')
    with redirect_stdout(io.StringIO()):
        cl.main(['export', '--dir', str(out)])
    line = ci.read_jsonl(out / 'results.jsonl')[0]
    assert line['pano']['width'] == 16384
    payload = send_to_ps.split_record(line)[0][1]
    assert (payload['labels'][0]['pano_x'], payload['labels'][0]['pano_y']) == \
        ps_pixel(0.25, 0.6, 16384, 8192)

    # a run build.json does not record is refused
    prop['pano']['run'] = 'runC'
    _write_jsonl(out / 'proposals.jsonl', [prop])
    _fake_build(out, {'results': runs['runA'] / 'results.jsonl'}, out / 'proposals.jsonl')
    with pytest.raises(SystemExit, match="runC"):
        cl.main(['export', '--dir', str(out)])


def test_rank_then_export_ranked_and_the_mismatch_refusal(tmp_path):
    runs = _two_run_setup(tmp_path)
    out = tmp_path / 'out'
    props = [dict(proposal(f'u{i}#0', 'P', x, 0.6), pano={'pano_id': 'P', 'run': 'runA',
                                                           'width': 8192, 'height': 4096})
             for i, x in enumerate((0.25, 0.75))]
    _write_jsonl(out / 'proposals.jsonl', props)
    _fake_build(out, {'results': runs['runA'] / 'results.jsonl'}, out / 'proposals.jsonl')
    x1, y1 = ps_pixel(0.75, 0.6, 8192, 4096)
    (tmp_path / 'r.csv').write_text(f'pano_id,pano_x,pano_y,score\nP,{x1},{y1},0.9\n',
                                    encoding='utf-8')
    with redirect_stdout(io.StringIO()):
        cl.main(['rank', '--dir', str(out), '--ranked-csv', str(tmp_path / 'r.csv')])
        cl.main(['export', '--dir', str(out), '--proposals',
                 str(out / 'proposals.ranked.jsonl')])
    rank = json.loads((out / 'rank.json').read_text(encoding='utf-8'))
    assert (rank['matched'], rank['unmatched']) == (1, 1)
    dets = ci.read_jsonl(out / 'results.jsonl')[0]['detections']
    assert {d['corner']: d['rank'] for d in dets} == {'u0#0': None, 'u1#0': 1}

    # proposals rebuilt after ranking: rank.json no longer ties the ranked file to build.json
    _write_jsonl(out / 'proposals.jsonl', props[:1])
    _fake_build(out, {'results': runs['runA'] / 'results.jsonl'}, out / 'proposals.jsonl')
    with pytest.raises(SystemExit, match='ranks other proposals'):
        cl.main(['export', '--dir', str(out), '--proposals',
                 str(out / 'proposals.ranked.jsonl')])
    # and an edited ranked file is refused by its own hash
    rank_ok = dict(rank, proposals_sha256=ci.sha256_file(out / 'proposals.jsonl'))
    (out / 'rank.json').write_text(json.dumps(rank_ok), encoding='utf-8')
    with open(out / 'proposals.ranked.jsonl', 'a', encoding='utf-8') as f:
        f.write('{}\n')
    with pytest.raises(SystemExit, match='rebuild'):
        cl.main(['export', '--dir', str(out), '--proposals',
                 str(out / 'proposals.ranked.jsonl')])


def test_load_detectors_refuses_no_curb_ramp_before_any_import(monkeypatch):
    import importlib

    def boom(name, *a, **kw):
        raise AssertionError(f'import of {name} attempted')
    monkeypatch.setattr(importlib, 'import_module', boom)
    with pytest.raises(registry.NoDetectorError):
        registry.load_detectors(['CurbRamp', 'NoCurbRamp'])


def test_points_sort_with_a_null_unit_id():
    c = corner(0, 0, 10, inv={'NA_noramp': 2}, panos=['p'])
    c['inventory'] = [{'unit_id': None, 'class': 'NA_noramp'},
                      {'unit_id': 'CR1', 'class': 'NA_noramp'}]
    props, _u, _c = cl.build_proposals([unit([c])], {'p': pano(0.0)}, NoSites(), args())
    assert props[0]['source']['inventory']['points'] == [[None, 'NA_noramp'],
                                                         ['CR1', 'NA_noramp']]


def test_score_reports_sample_keys_not_in_the_proposals():
    props = [{'key': 'u0#0', 'unit': 'u0', 'corner': 0, 'stratum': 'residential'}]
    res = cl.score(props, _verdicts([('u0', 0, 'absent', None)]),
                   sample_keys=['u0#0', 'zz#1'])
    assert res['n_proposals'] == 1 and res['sample_keys_not_in_proposals'] == ['zz#1']


def test_na_read_settled_flag_records_the_settlement():
    """Without the flag build.json says what it always said; with it, the gallery 243
    settlement (PR #180) and whether this build's read is the primary one."""
    assert cl.na_read_params(args(na_read='discount')) == {'na_read_settled': False}
    d = cl.na_read_params(args(na_read='discount', na_read_settled=True))
    assert d['na_read_settled'] is True and d['na_read_primary'] == 'discount'
    assert d['na_read_is_primary'] is True
    s = d['na_read_settlement']
    assert s['pr'].endswith('#180') and len(s['score_json_sha256']) == 64
    assert s == cl.NA_READ_SETTLEMENT and s is not cl.NA_READ_SETTLEMENT
    c = cl.na_read_params(args(na_read='count', na_read_settled=True))
    assert c['na_read_settled'] is True and c['na_read_is_primary'] is False


def test_sector_record_phantom_flags_only_from_a_split_inventory():
    old = {'start_deg': 270.0, 'width_deg': 180.0, 'wide': True}
    assert cl.sector_record(old) == old
    new = dict(old, width_deg=90.0, wide=False, phantom_start=False, phantom_end=True)
    assert cl.sector_record(new) == {'start_deg': 270.0, 'width_deg': 90.0, 'wide': False,
                                     'phantom_start': False, 'phantom_end': True}


# ------------------------------------------------------- test-server pilot (#169 pilot)

def _strat_props():
    return [{'key': f'r{i}#0', 'unit': f'r{i}', 'corner': 0, 'stratum': 'residential'}
            for i in range(6)] + \
           [{'key': f'a{i}#0', 'unit': f'a{i}', 'corner': 0, 'stratum': 'arterial'}
            for i in range(3)]


def test_select_proposals_stratum_filter_and_only_corners():
    props = _strat_props()
    kept, rec = cl.select_proposals(props, strata=['residential'])
    assert [p['key'] for p in kept] == [f'r{i}#0' for i in range(6)]
    assert rec['dropped_by_stratum'] == 3 and rec['kept_by_stratum'] == {'residential': 6}
    kept, rec = cl.select_proposals(props, strata=['residential'], only=['r4#0', 'r1#0'])
    assert [p['key'] for p in kept] == ['r1#0', 'r4#0']          # the proposals' own order
    assert rec['dropped_by_only_corners'] == 4 and rec['kept'] == 2
    kept, _rec = cl.select_proposals(props, only=['a2#0'])
    assert [p['key'] for p in kept] == ['a2#0']


def test_select_proposals_refuses_unknown_keys_and_strata():
    props = _strat_props()
    with pytest.raises(SystemExit, match='not in the proposals'):
        cl.select_proposals(props, only=['r1#0', 'zz#9'])
    with pytest.raises(SystemExit, match='outside --stratum'):
        cl.select_proposals(props, strata=['residential'], only=['a0#0'])
    with pytest.raises(SystemExit, match='not a stratum'):
        cl.select_proposals(props, strata=['highway'])


def test_pilot_order_is_one_seeded_permutation():
    props = _strat_props()
    a = cl.pilot_order(props, ['residential'], 169)
    b = cl.pilot_order(list(reversed(props)), ['residential'], 169)
    assert a == b and sorted(a) == sorted(f'r{i}#0' for i in range(6))
    assert len(cl.pilot_order(props, None, 169)) == 9
    big = [{'key': f'u{i:04d}#0', 'stratum': 'residential'} for i in range(500)]
    full = cl.pilot_order(big, ['residential'], 169)
    assert full == cl.pilot_order(list(reversed(big)), ['residential'], 169)
    assert full != cl.pilot_order(big, ['residential'], 170)
    assert cl.pilot_roles(6, 2, 1, 4) == ['rate', 'rate', 'reserve', 'send', 'rest', 'rest']


def test_stop_rule_arithmetic():
    r = cl.stop_rule()
    assert [(x['look'], x['pass_if_wrong_at_most']) for x in r['looks']] == \
        [(40, 0), (60, 2), (100, 6)]
    assert r['fail_once_wrong_exceeds'] == 6
    assert r['z'] == pytest.approx(2.394, abs=1e-3)
    # unadjusted (z 1.96) the same looks would allow 1 / 3 / 8 wrong: the correction costs that
    assert [x['unadjusted_z196_pass_if_wrong_at_most'] for x in r['looks']] == [1, 3, 8]


def test_sequential_read_reserve_skip_and_stops():
    keys = [f'k{i:03d}' for i in range(200)]
    # can't-tell absorbed by the reserve: 2 unsure, 40 agree -> PASS at look 40, order 42
    v = {k: 'agree' for k in keys[:42]}
    v[keys[3]] = v[keys[10]] = 'unsure'
    res = cl.sequential_read(keys, v)
    assert (res['status'], res['look'], res['stopped_at_order'], res['unsure']) == \
        ('PASS', 40, 42, 2)
    # one wrong among the first 40 -> no pass at 40; 59 right of 60 -> PASS at 60
    v = {k: 'agree' for k in keys[:60]}
    v[keys[5]] = 'disagree'
    res = cl.sequential_read(keys, v)
    assert (res['status'], res['look'], res['wrong']) == ('PASS', 60, 1)
    # still rating: CONTINUE
    assert cl.sequential_read(keys, {k: 'agree' for k in keys[:30]})['status'] == 'CONTINUE'
    # a skipped label is refused
    v = {k: 'agree' for k in keys[:30]}
    del v[keys[7]]
    with pytest.raises(SystemExit, match='cannot be skipped'):
        cl.sequential_read(keys, v)
    # the 7th wrong label stops it as FAIL at once
    v = {k: ('disagree' if i < 7 else 'agree') for i, k in enumerate(keys[:20])}
    res = cl.sequential_read(keys, v)
    assert (res['status'], res['stopped_at_order'], res['wrong']) == ('FAIL', 7, 7)
    with pytest.raises(SystemExit, match='unknown verdict'):
        cl.sequential_read(keys, {keys[0]: 'maybe'})


def test_pilot_end_to_end_sample_export_dry_run_and_score(tmp_path):
    """build -> sample --stratum --reserve -> export --stratum --only-corners --only-first ->
    send_to_ps dry run, the store guard on a NoCurbRamp store-built file, and pilot-score."""
    run = tmp_path / 'run'
    blocks = [dict(block(f'P{i}'), lat=at(30 * i, 0)[0], lng=at(30 * i, 0)[1],
                   source_detail='ps_store') for i in range(4)]
    _write_jsonl(run / 'results.jsonl', [{'detections': [], 'label_type': 'CurbRamp', 'pano': b}
                                         for b in blocks])
    _write_jsonl(run / 'sites.jsonl', [])
    (run / 'sites_meta.json').write_text(json.dumps({'params': {'min_confidence': 0.3}}),
                                         encoding='utf-8')
    units = []
    for i in range(4):
        typ = 'arterial' if i == 3 else 'residential'
        units.append(unit([corner(0, 30 * i, 10, panos=[f'P{i}']),
                           corner(1, 30 * i + 6, 10, panos=[f'P{i}'])],
                          typ=typ, name=f'vancouver:{typ[:3]}:n{i}'))
    _write_jsonl(tmp_path / 'corners_full.jsonl', units)
    out = tmp_path / 'out'
    with redirect_stdout(io.StringIO()):
        cl.main(['build', '--city', 'testcity', '--na-read', 'discount',
                 '--corners', str(tmp_path / 'corners_full.jsonl'), '--run-dir', str(run),
                 '--out', str(out)])
    build = json.loads((out / 'build.json').read_text(encoding='utf-8'))
    assert build['counts']['proposed'] == 8
    build['tool']['git_sha'], build['tool']['dirty'] = 'd' * 40, False
    (out / 'build.json').write_text(json.dumps(build), encoding='utf-8')

    pilot = out / 'pilot' / 'pilot.csv'
    pilot.parent.mkdir()
    with redirect_stdout(io.StringIO()):
        cl.main(['sample', '--proposals', str(out / 'proposals.jsonl'), '--stratum',
                 'residential', '--n', '2', '--reserve', '1', '--send', '4', '--looks', '2',
                 '--out', str(pilot)])
    rec = json.loads(pilot.with_suffix('.json').read_text(encoding='utf-8'))
    assert rec['population'] == 6
    assert rec['roles'] == {'rate': 2, 'reserve': 1, 'send': 1, 'rest': 2}
    order = cl.read_corner_order(pilot)
    assert all(':res:' in k for k in order) and len(order) == 6

    sub = out / 'pilot' / 'results.pilot4.jsonl'
    with redirect_stdout(io.StringIO()):
        cl.main(['export', '--dir', str(out), '--stratum', 'residential', '--only-corners',
                 str(pilot), '--only-first', '4', '--out', str(sub)])
        cl.main(['export', '--dir', str(out), '--stratum', 'residential',
                 '--out', str(out / 'pilot' / 'results.residential.jsonl')])
    erec = json.loads((out / 'pilot' / 'results.pilot4.export.json').read_text(encoding='utf-8'))
    assert erec['labels'] == 4 and erec['selection']['kept'] == 4
    assert erec['selection']['only_corners']['first'] == 4
    assert erec['output']['sha256'] == ci.sha256_file(sub)
    assert erec['output']['lines'] == erec['lines']
    assert not (out / 'export.json').exists()          # the build dir's record is untouched
    lines = ci.read_jsonl(sub)
    assert sorted(d['pilot_order'] for ln in lines for d in ln['detections']) == [1, 2, 3, 4]
    heads = [min(d['pilot_order'] for d in ln['detections']) for ln in lines]
    assert heads == sorted(heads)                        # lines follow the draw order
    rrec = json.loads((out / 'pilot' / 'results.residential.export.json')
                      .read_text(encoding='utf-8'))
    assert rrec['labels'] == 6 and rrec['selection']['dropped_by_stratum'] == 2
    with pytest.raises(SystemExit, match='only 6 corners'):
        cl.main(['export', '--dir', str(out), '--only-corners', str(pilot), '--only-first', '9',
                 '--out', str(tmp_path / 'x.jsonl')])

    # send_to_ps: one type, the dry run records nothing, and the type-aware store guard passes
    assert send_to_ps.file_label_types(sub) == ['NoCurbRamp']
    buf = io.StringIO()
    with redirect_stdout(buf):
        send_to_ps.process_jsonl_file(str(sub), 'https://ps.example/ai/x', dry_run=True)
    assert 'NoCurbRamp' in buf.getvalue()
    assert not (sub.parent / (sub.name + '.submitted')).exists()
    note = send_to_ps.check_store_built(sub)
    assert note and 'NoCurbRamp' in note
    send_to_ps.check_model_provenance(sub)

    # pilot-score: the lead's validations, keyed on PS's pixel key
    props = {p['key']: p for p in ci.read_jsonl(out / 'proposals.jsonl')}
    rows = ['pano_id,pano_x,pano_y,validation']
    for k, v in zip(order[:3], ('Agree', 'Unsure', 'Agree')):
        pid, x, y = cl.pixel_key(props[k])
        rows.append(f'{pid},{x},{y},{v}')
    vcsv = tmp_path / 'validations.csv'
    vcsv.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    with redirect_stdout(io.StringIO()):
        cl.main(['pilot-score', '--pilot', str(pilot), '--proposals',
                 str(out / 'proposals.jsonl'), '--validations', str(vcsv),
                 '--out', str(tmp_path / 'read.json')])
    read = json.loads((tmp_path / 'read.json').read_text(encoding='utf-8'))
    assert (read['decided'], read['unsure'], read['look']) == (2, 1, 2)
    assert read['status'] == 'FAIL'       # 2 of 2 cannot clear a Wilson lower bound of 0.85

    # pilot-links: server label ids back in draw order (only the corners the server holds)
    srv = [{'label_id': str(900 + j), 'pano_id': cl.pixel_key(props[k])[0],
            'pano_x': cl.pixel_key(props[k])[1], 'pano_y': cl.pixel_key(props[k])[2]}
           for j, k in reversed(list(enumerate(order[:3])))]
    got = cl.pilot_label_ids(order[:4], list(props.values()), srv)
    assert got == [(1, order[0], 900), (2, order[1], 901), (3, order[2], 902)]
    with pytest.raises(SystemExit, match='two server labels'):
        cl.pilot_label_ids(order, list(props.values()), srv + [dict(srv[0], label_id='999')])
