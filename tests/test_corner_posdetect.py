"""corner_posdetect.py and corner_inventory.py --extra-run (RampNet#241), on synthetic
geometry: position selection (radius, run and no-JPEG exclusions, GSV fallback after
skips), the 25 m circle, the extra-run loader (id prefixing, the emulated deployed arm,
refusals), the three absence reads, the nearest-pano-only read, transitions, and LF
byte-stable compare outputs. CPU only, no network, no Vancouver file."""
import json
import math

import pytest

import corner_inventory as ci
import corner_posdetect as cp
import geo

LAT0, LNG0 = 45.6, -122.6
FR = geo.LocalFrame(LAT0, LNG0)
BUILD = {'frame': {'lat0': LAT0, 'lng0': LNG0}, 'params': {'obs_m': 25.0}}
LEGS4 = [0.0, 90.0, 180.0, 270.0]
EMPTY = {k: 0 for k in ci.INV_CLASSES}


def ll(e, n):
    return FR.to_latlng(e, n)


def unit_rec(key, e, n, state='unobservable', typ='residential', inv=None):
    lat, lng = ll(e, n)
    st = {f'{a}/{v}': state for a in ci.ARMS for v in ci.VARIANTS}
    return {'unit': key, 'type': typ, 'lat': lat, 'lng': lng, 'state': st,
            'inv_counts': dict(EMPTY, **(inv or {})), 'pano_ids_25': [], 'site_ids': [],
            'corners': []}


def ps(pid, e, n):
    lat, lng = ll(e, n)
    return {'pano_id': pid, 'lat': lat, 'lng': lng}


def test_select_radius_and_exclusions():
    recs = [unit_rec('u1', 0, 0), unit_rec('u2', 1000, 0), unit_rec('u3', 0, 0, 'present'),
            unit_rec('m', 0, 0, typ='mid_block')]
    panos = [ps('near', 0, 10), ps('edge', 24.9, 0), ps('far', 0, 26), ps('inrun', 5, 0),
             ps('nojpg', -5, 0), ps('lonely', 1000, 3)]
    rows, chosen = cp.select(recs, BUILD, panos, store_ids=['near', 'edge', 'far', 'inrun'],
                             run_ids=['inrun'], radius_m=25.0)
    assert chosen == ['edge', 'near']             # sorted; far, in-run and no-JPEG out
    assert [r['unit'] for r in rows] == ['u1', 'u2']   # only unobservable intersections
    u1, u2 = rows
    assert (u1['n_ps_panos'], u1['n_in_run'], u1['n_no_jpg'], u1['n_selected']) == (4, 1, 1, 2)
    assert u1['nearest_selected_id'] == 'near' and u1['nearest_selected_m'] == 10.0
    assert u2['n_selected'] == 0 and u2['n_no_jpg'] == 1 and 'source' not in u2
    # after the metadata pass: a unit whose only selected pano was skipped goes to GSV
    rows, _ = cp.select(recs, BUILD, panos, ['near', 'edge'], [], 25.0,
                        skips={'near': 'jpg_missing', 'edge': 'jpg_missing'})
    assert rows[0]['source'] == 'gsv' and rows[0]['n_usable'] == 0


def test_circle_is_closed_and_has_the_radius():
    lat, lng = ll(100, -50)
    ring = cp.circle(FR, lat, lng, 25.0)
    assert ring[0] == ring[-1] and len(ring) == cp.CIRCLE_VERTICES + 1
    for lo, la in ring[:-1]:
        e, n = FR.to_enu(la, lo)
        assert math.isclose(math.hypot(e - 100, n + 50), 25.0, abs_tol=0.05)


def write_run(d, panos, sites, params=None):
    d.mkdir(parents=True)
    with open(d / 'results.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for pid, e, n, date in panos:
            lat, lng = ll(e, n)
            f.write(json.dumps({'pano': {'panorama_id': pid, 'lat': lat, 'lng': lng,
                                         'capture_date': date}, 'detections': []}) + '\n')
    with open(d / 'sites.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for sid, e, n, members in sites:
            lat, lng = ll(e, n)
            f.write(json.dumps({'site_id': sid, 'lat': lat, 'lng': lng,
                                'n_operational': sum(c >= 0.3 for _, c in members),
                                'members': [{'pano_id': p, 'confidence': c}
                                            for p, c in members]}) + '\n')
    meta = {'params': dict({'floor': 0.1, 'min_confidence': 0.3, 'max_range_m': 25.0,
                            'mask_rig': True, 'camera_height_m': 2.5}, **(params or {}))}
    (d / 'sites_meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return meta


BASE_META = {'params': {'floor': 0.1, 'min_confidence': 0.3, 'max_range_m': 25.0,
                        'mask_rig': True, 'camera_height_m': 'per-rig'}}


def test_load_extra_run_prefixes_and_emulates_deployed(tmp_path):
    d = tmp_path / 'vancouver_posdetect241'
    write_run(d, [('a', 0, 5, '2024-05'), ('b', 0, -5, '2022-01')],
              [(0, 6, 6, [('a', 0.4)]), (1, -6, 6, [('a', 0.2), ('b', 0.6)]),
               (2, 0, 0, [('b', 0.2)])])
    panos, sites, dep, rec = ci.load_extra_run(d, BASE_META, ['zzz'])
    assert [p[2] for p in panos] == ['a', 'b']
    # site 2 is not operational; site 1 has op pano b only
    assert [s[2]['site_id'] for s in sites] == ['vancouver_posdetect241:0',
                                                'vancouver_posdetect241:1']
    assert sites[1][2]['op_panos'] == ['b']
    # only site 1 has a member >= 0.55
    assert [x[2] for x in dep] == ['vancouver_posdetect241:s1']
    assert rec['n_deployed_emulated'] == 1 and rec['deployed_tier'] == 0.55
    with pytest.raises(SystemExit, match='already in the base run'):
        ci.load_extra_run(d, BASE_META, ['a'])
    d2 = tmp_path / 'other'
    write_run(d2, [('c', 0, 0, None)], [], params={'min_confidence': 0.5})
    with pytest.raises(SystemExit, match='min_confidence'):
        ci.load_extra_run(d2, BASE_META, [])


def test_bucket_and_three_reads():
    assert cp.bucket(EMPTY) == 'clean'
    na = dict(EMPTY, NA_noramp=1)
    assert cp.bucket(na) == 'rmvna' and cp.bucket(na, ('NA_noramp',)) == 'clean'
    assert cp.bucket(dict(na, Available=1), ('NA_noramp',)) == 'false'
    units = [unit_rec('c', 0, 0, 'absent'), unit_rec('n', 0, 0, 'absent', inv={'NA_noramp': 2}),
             unit_rec('f', 0, 0, 'absent', inv={'Available': 1}),
             unit_rec('p', 0, 0, 'present')]
    units[1]['state']['deployed/primary'] = 'present'
    rows = {r['read']: r for r in cp.absence_reads(units)}
    a = rows['a_fusion_clean_as_written']
    assert (a['n_absent'], a['clean'], a['no_available']) == (3, 1, 2)
    b = rows['b_fusion_clean_ignoring_NA_noramp']
    assert (b['n_absent'], b['clean']) == (3, 2)
    c = rows['c_deployed_clean']
    assert (c['n_absent'], c['clean'], c['no_available']) == (2, 1, 1)
    assert cp.absence_reads([])[0]['clean_p'] is None


def test_nearest_only_states():
    added = {'near': ll(0, 3), 'far': ll(0, -20)}
    u = unit_rec('t', 0, 0, 'present')
    u['pano_ids_25'] = ['far', 'near']
    # the only site was seen from `far` alone -> nearest-only: absent
    u['site_ids'] = ['x:1']
    st = cp.nearest_only_states([u], {'t'}, added, {'x:1': ['far']}, FR, 25.0)
    assert st['t'][cp.KEY] == 'absent'
    st = cp.nearest_only_states([u], {'t'}, added, {'x:1': ['far', 'near']}, FR, 25.0)
    assert st['t'][cp.KEY] == 'present'
    # a base-run site in the window still counts
    u['site_ids'] = [17]
    st = cp.nearest_only_states([u], {'t'}, added, {}, FR, 25.0)
    assert st['t'][cp.KEY] == 'present'


def test_transitions():
    old = [unit_rec('a', 0, 0), unit_rec('b', 0, 0), unit_rec('c', 0, 0, 'absent'),
           unit_rec('m', 0, 0, typ='mid_block')]
    new = [unit_rec('a', 0, 0, 'absent'), unit_rec('b', 0, 0, 'present'),
           unit_rec('c', 0, 0, 'absent'), unit_rec('m', 0, 0, 'present', typ='mid_block')]
    t = cp.transitions(old, new, cp.KEY)
    assert t == {('residential', 'unobservable', 'absent'): 1,
                 ('residential', 'unobservable', 'present'): 1,
                 ('residential', 'absent', 'absent'): 1}


def make_idx(panos=(), sites=(), clusters=(), inventory=()):
    def pts(xs):
        return ci.Points(FR, [(*ll(e, n), p) for e, n, p in xs], 37.0)
    return {'panos': pts(panos), 'sites': pts(sites), 'clusters': pts(clusters),
            'inventory': pts(inventory)}


def test_compare_end_to_end_is_byte_stable(tmp_path):
    """Two units, both unobservable before; the extra run puts a pano at each, and a
    site at one. The decision read moves from undefined to 1 absent, 1 clean."""
    run = tmp_path / 'xrun'
    write_run(run, [('a', 0, 5, '2024-05'), ('b', 500, 5, '2025-08')],
              [(0, 6, 6, [('a', 0.9)])])
    panos, sites, dep, _rec = ci.load_extra_run(run, BASE_META, [])
    base = {'UNIT': None}
    units = []
    for key, e in (('vancouver:res:n1', 0), ('vancouver:res:n2', 500)):
        lat, lng = ll(e, 0)
        units.append({'unit': key, 'type': 'residential', 'lat': lat, 'lng': lng,
                      'node_ids': [1], 'highways': ['residential']})
    old = [ci.describe_unit(u, FR.to_enu(u['lat'], u['lng']), LEGS4, make_idx(), 25.0)
           for u in units]
    idx_new = {'panos': ci.Points(FR, panos, 37.0), 'sites': ci.Points(FR, sites, 30.0),
               'clusters': ci.Points(FR, dep, 30.0), 'inventory': ci.Points(FR, [], 30.0)}
    new = [ci.describe_unit(u, FR.to_enu(u['lat'], u['lng']), LEGS4, idx_new, 25.0)
           for u in units]
    del base
    for name, recs in (('old', old), ('new', new)):
        d = tmp_path / name
        ci.write_jsonl_lf(d / 'corners_full.jsonl', recs)
        b = dict(BUILD, inputs={'results': {'path': str(run / 'results.jsonl'),
                                            'sha256': 'base'}})
        if name == 'new':
            b['extra_runs'] = [_rec]
            for k, rel in (('results', 'results.jsonl'), ('sites', 'sites.jsonl'),
                           ('sites_meta', 'sites_meta.json')):
                b['inputs'][f'extra:xrun:{k}'] = {'path': str(run / rel), 'sha256': k}
        (d / 'build.json').write_text(json.dumps(b), encoding='utf-8')
    outs = []
    for k in (1, 2):
        res = cp.main(['compare', '--old', str(tmp_path / 'old'), '--new',
                       str(tmp_path / 'new'), '--out', str(tmp_path / f'out{k}')])
        outs.append(tmp_path / f'out{k}' / 'compare')
    assert res['target_units'] == 2 and res['added_panos'] == 2
    assert res['target_states']['fusion'] == {'absent': 1, 'present': 1}
    assert res['target_states']['deployed'] == {'absent': 1, 'present': 1}
    assert res['decision_rule']['n_absent'] == 1 and res['decision_rule']['clean'] == 1
    assert res['capture_years_added'] == {'2024': 1, '2025': 1}
    moved = [r for r in res['reads'] if r['subset'] == 'moved'
             and r['stratum'] == 'intersections']
    assert {r['read']: r['n_absent'] for r in moved}['a_fusion_clean_as_written'] == 1
    for name in ('compare.json', 'transitions.csv', 'reads.csv', 'report.md'):
        a, b = (outs[0] / name).read_bytes(), (outs[1] / name).read_bytes()
        assert a == b and b'\r\n' not in a
