"""corner_gallery.py / corner_gallery_score.py (RampNet#243) on synthetic units: the three
populations, the seeded draw and review order, the pano projection, view choice, the
inventory join (UNITID is not unique), validation refusals, unit outcomes, the false-absence
classes, the control and NA shares, agreement, the rendered page, and the page's state
bootstrap under node. Plus a consistency check of the committed Vancouver bundle (items
sha256, parts, seed). CPU only, no network."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import corner_gallery as cg
import corner_gallery_page as cgp
import corner_gallery_score as cs
import geo

LAT0, LNG0 = 45.6, -122.6
FR = geo.LocalFrame(LAT0, LNG0)
EMPTY = {'Available': 0, 'NA_noramp': 0, 'NA_typed': 0, 'RMV': 0, 'Expired/Removed': 0, 'other': 0}
BUNDLE = Path(__file__).resolve().parents[1] / 'runs/vancouver/corner_gallery243'


def rec(unit, state='absent', typ='residential', inv=None):
    st = {'fusion/primary': state, 'deployed/primary': state}
    return {'unit': unit, 'type': typ, 'state': st, 'inv_counts': dict(EMPTY, **(inv or {}))}


# ---------------------------------------------------------------------------- sampling

def test_populations_split_by_inventory_and_target():
    recs = [rec('fa_t', inv={'Available': 1}), rec('fa_other', inv={'Available': 1, 'NA_noramp': 2}),
            rec('na', inv={'NA_noramp': 2}), rec('na_rmv', inv={'NA_noramp': 1, 'RMV': 1}),
            rec('clean'), rec('present', state='present'), rec('mid', typ='mid_block'),
            rec('unobs', state='unobservable')]
    pops = cg.populations(recs, targets={'fa_t', 'na', 'clean'})
    assert pops == {'false_absence': ['fa_t'], 'na_noramp': ['na'], 'clean': ['clean']}


def test_draw_is_seeded_and_stable():
    pops = {'false_absence': ['f1', 'f2'], 'na_noramp': [f'n{i:03d}' for i in range(100)],
            'clean': [f'c{i:03d}' for i in range(50)]}
    a = cg.draw(pops, seed=243, n_na=40, n_clean=20)
    b = cg.draw(pops, seed=243, n_na=40, n_clean=20)
    assert a == b
    assert a['false_absence'] == ['f1', 'f2']
    assert len(a['na_noramp']) == 40 and len(set(a['na_noramp'])) == 40
    assert len(a['clean']) == 20
    assert a != cg.draw(pops, seed=244, n_na=40, n_clean=20)
    order = cg.review_order(a['na_noramp'] + a['clean'])
    assert order == sorted(order, key=lambda u: hashlib.sha1(u.encode()).hexdigest())


# ---------------------------------------------------------------------------- geometry

@pytest.mark.parametrize('heading,e,n,x', [(0.0, 0, 10, 0.5), (0.0, 10, 0, 0.75),
                                          (90.0, 0, 10, 0.25), (350.0, 0, 10, 0.527778),
                                          (0.0, 0, -10, 0.0)])
def test_pano_view_x_follows_heading(heading, e, n, x):
    lat, lng = FR.to_latlng(e, n)
    v = cg.pano_view(LAT0, LNG0, heading, lat, lng)
    assert abs(((v['x'] - x + 0.5) % 1.0) - 0.5) < 1e-4
    assert abs(v['dist_m'] - 10.0) < 0.05
    assert 0.5 < v['y'] < 1.0


def test_pano_view_nearer_is_lower():
    far = cg.pano_view(LAT0, LNG0, 0.0, *FR.to_latlng(0, 20))
    near = cg.pano_view(LAT0, LNG0, 0.0, *FR.to_latlng(0, 5))
    assert near['y'] > far['y']


def test_choose_views_prefers_min_distance():
    c = [{'pano_id': 'a', 'dist_m': 1.0}, {'pano_id': 'b', 'dist_m': 4.0},
         {'pano_id': 'c', 'dist_m': 9.0}, {'pano_id': 'd', 'dist_m': 50.0}]
    assert [v['pano_id'] for v in cg.choose_views(c, k=2)] == ['b', 'c']
    assert [v['pano_id'] for v in cg.choose_views(c, k=3)] == ['b', 'c', 'a']
    assert [v['pano_id'] for v in cg.choose_views(c, k=5)] == ['b', 'c', 'a']   # 50 m dropped


def test_inventory_position_resolves_duplicate_ids_by_position():
    near, far = FR.to_latlng(5, 0), FR.to_latlng(5000, 0)
    inv = {'CR1': [(far[0], far[1], 'Available'), (near[0], near[1], 'NA_noramp')]}
    assert cg.inventory_position(inv, 'CR1', 'NA_noramp', LAT0, LNG0) == near
    assert cg.inventory_position(inv, 'CR1', 'Available', LAT0, LNG0) == (None, None)
    assert cg.inventory_position(inv, 'CR9', 'Available', LAT0, LNG0) == (None, None)


def full_record(unit='vancouver:res:n1'):
    corners = []
    for k, (start, width) in enumerate([(0.0, 90.0), (90.0, 270.0)]):
        b = start + width / 2
        import math
        e, n = 12 * math.sin(math.radians(b)), 12 * math.cos(math.radians(b))
        lat, lng = FR.to_latlng(e, n)
        corners.append({'corner': k, 'start_deg': start, 'width_deg': width, 'wide': width > 150,
                        'lat': lat, 'lng': lng, 'n_panos_25': 2, 'pano_ids_25': ['p1', 'p2'],
                        'inv_counts': dict(EMPTY, Available=1 if k == 0 else 0),
                        'inventory': [{'unit_id': 'CR1', 'class': 'Available', 'status': 'Available',
                                       'ramptype': 'PERP', 'instdate': '2025-09-12'}] if k == 0 else []})
    return {'unit': unit, 'type': 'residential', 'lat': LAT0, 'lng': LNG0, 'legs': [0.0, 90.0],
            'n_legs': 2, 'node_ids': [1], 'highways': ['residential'],
            'state': {'fusion/primary': 'absent'}, 'n_panos_25': 2, 'nearest_pano_m': 3.0,
            'inv_counts': dict(EMPTY, Available=1), 'pano_ids_25': ['p1'], 'corners': corners}


def test_build_item_views_and_inventory():
    r = full_record()
    panos = {'p1': {'lat': FR.to_latlng(-10, 0)[0], 'lng': FR.to_latlng(-10, 0)[1], 'heading': 0.0,
                    'capture_date': '2023-05', 'run': 'results'},
             'p2': {'lat': FR.to_latlng(0, -15)[0], 'lng': FR.to_latlng(0, -15)[1], 'heading': 90.0,
                    'capture_date': '2024-06', 'run': 'results'}}
    p = FR.to_latlng(8, 8)
    it = cg.build_item(r, 'false_absence', panos, {'CR1': [(p[0], p[1], 'Available')]})
    assert it['part'] == 'false_absence' and len(it['corners']) == 2
    c0 = it['corners'][0]
    assert [v['pano_id'] for v in c0['views']] == ['p1', 'p2']        # p1 is nearer corner 0
    assert c0['views'][0]['crop'] == 'crops/vancouver_res_n1_c0_p1.jpg'
    assert (c0['inventory'][0]['lat'], c0['inventory'][0]['lng']) == p
    assert it['aerial']['file'] == 'aerial/vancouver_res_n1.jpg'
    jobs = cg.crop_jobs([it], Path('/nonexistent'))
    assert sum(len(v) for v in jobs.values()) == 4
    assert jobs['results'][0]['path'] == 'p1/p1.jpg'


# ------------------------------------------------------------------------------ scoring

def item(unit, part, inv=None, views_date='2023-05'):
    inv = inv or {}
    cs_ = []
    for k in range(3):
        cls = inv.get(k)
        cs_.append({'corner': k, 'inv_counts': dict(EMPTY, **({cls[0]: 1} if cls else {})),
                    'views': [{'capture_date': views_date}],
                    'inventory': [{'class': cls[0], 'instdate': cls[1], 'unit_id': f'CR{k}'}]
                    if cls else []})
    return {'unit': unit, 'part': part, 'corners': cs_}


def unit_verdicts(vs, kinds=None, complete=True, blind=None, edited=False):
    kinds = kinds or {}
    m = {str(k): {'verdict': v, 'absent_kind': kinds.get(k)} for k, v in enumerate(vs)}
    if not complete:
        bl = None
    elif blind is None:
        bl = json.loads(json.dumps(m))
    else:
        bl = {str(k): {'verdict': v, 'absent_kind': None} for k, v in enumerate(blind)}
    return {'corners': m, 'blind': bl, 'complete': complete, 'elapsed_s': 3.0,
            'inventory_seen': complete, 'edited_after_inventory': edited}


def vfile(units, sha='S'):
    return {'schema': cs.SCHEMA, 'items_sha256': sha, 'rubric_version': 1, 'rubric': cg.RUBRIC,
            'rater': 'jonf', 'units': units}


def test_outcome():
    assert cs.outcome({'0': 'absent', '1': 'present'}) == 'present'
    assert cs.outcome({'0': 'absent', '1': 'absent'}) == 'absent'
    assert cs.outcome({'0': 'absent', '1': 'cant_tell'}) == 'undetermined'


@pytest.mark.parametrize('vs,inst,cls', [
    (['present', 'absent', 'absent'], '2020-01-01', 'miss_at_inventory_corner'),
    (['absent', 'present', 'absent'], '2020-01-01', 'miss_elsewhere'),
    (['absent', 'absent', 'absent'], '2025-09-12', 'artifact_built_after_imagery'),
    (['absent', 'absent', 'absent'], '2020-01-01', 'artifact_inventory_or_geometry'),
    (['absent', 'absent', 'absent'], None, 'artifact_inventory_or_geometry'),
    (['cant_tell', 'absent', 'absent'], '2020-01-01', 'undetermined'),
])
def test_false_absence_classes(vs, inst, cls):
    it = item('u', 'false_absence', {0: ('Available', inst)})
    assert cs.classify_false_absence(it, {str(k): v for k, v in enumerate(vs)}) == cls


def test_validate_refusals():
    items = [item('u1', 'clean')]
    good = vfile({'u1': unit_verdicts(['absent', 'absent', 'present'], {0: 'no_sidewalk'})})
    assert cs.validate(good, items, 'S') == []
    bad = vfile({'u1': unit_verdicts(['absent', None, 'present'])}, sha='X')
    bad['rubric'] = ''
    bad['units']['u9'] = unit_verdicts(['absent'])
    bad['units']['u1']['corners']['0']['absent_kind'] = 'no_sidewalk'
    bad['units']['u1']['corners']['2']['absent_kind'] = 'no_sidewalk'
    bad['units']['u1']['inventory_seen'] = False
    p = '\n'.join(cs.validate(bad, items, 'S'))
    for frag in ('items_sha256', 'rubric text', 'u9: not a unit', 'corner 1 has no verdict',
                 'absent_kind with verdict', 'inventory_seen is false'):
        assert frag in p, frag
    nob = vfile({'u1': dict(unit_verdicts(['absent'] * 3), blind=None)})
    assert any('no blind' in x for x in cs.validate(nob, items, 'S'))


def test_score_parts_blind_vs_final_and_stratified():
    items = ([item(f'n{i}', 'na_noramp', {0: ('NA_noramp', None)}) for i in range(4)] +
             [item(f'c{i}', 'clean') for i in range(2)] +
             [item('f0', 'false_absence', {1: ('Available', '2019-01-01')}),
              item('f1', 'false_absence', {1: ('Available', '2019-01-01')})])
    units = {'n0': unit_verdicts(['absent'] * 3, {0: 'no_sidewalk'}),
             'n1': unit_verdicts(['absent'] * 3, {0: 'curb_no_ramp'}),
             'n2': unit_verdicts(['present', 'absent', 'absent']),
             'n3': unit_verdicts(['cant_tell', 'absent', 'absent']),
             'c0': unit_verdicts(['absent'] * 3),
             # blind said absent everywhere; changed to present after the reveal
             'c1': unit_verdicts(['present', 'absent', 'absent'], blind=['absent'] * 3, edited=True),
             'f0': unit_verdicts(['absent', 'present', 'absent']),
             'f1': unit_verdicts(['absent'] * 3)}
    snap = {'populations': {'false_absence': 2, 'na_noramp': 40, 'clean': 20}}
    b = cs.score_rater(items, vfile(units), snap, 'blind')
    na = b['parts']['na_noramp']
    assert na['outcomes'] == {'present': 1, 'absent': 2, 'undetermined': 1}
    assert na['unit_absent_share']['k'] == 2 and na['unit_absent_share']['n'] == 3
    assert na['unit_absent_conservative']['n'] == 4
    assert na['corner_verdicts'] == {'present': 1, 'absent': 2, 'cant_tell': 1}  # NA corners only
    assert na['absent_kinds'] == {'curb_no_ramp': 1, 'no_sidewalk': 1, 'unspecified': 0}
    assert b['parts']['clean']['unit_absent_share']['k'] == 2
    fa = b['parts']['false_absence']
    assert fa['classes']['miss_at_inventory_corner'] == 1
    assert fa['classes']['artifact_inventory_or_geometry'] == 1
    assert fa['real_miss_share']['n'] == 2
    assert b['edited_after_inventory'] == 1
    f = cs.score_rater(items, vfile(units), snap, 'final')
    assert f['parts']['clean']['unit_absent_share']['k'] == 1
    st = b['stratified']
    # (20 * 1 + 40 * 2/3 + 2 * 1/2) / 62
    assert abs(st['estimate'] - (20 + 40 * 2 / 3 + 1) / 62) < 1e-4
    assert st['not_covered'] == cs.ABSENT_TOTAL - 62
    rep = cs.render_report({'items_sha256': 'S', 'seed': 243, 'populations': snap['populations']},
                           items, {'jonf': {'blind': b, 'final': f}}, {'jonf': []})
    assert 'Rater `jonf`' in rep and '2/3' in rep


def test_incomplete_units_are_not_scored():
    items = [item('c0', 'clean')]
    v = vfile({'c0': unit_verdicts(['absent', None, None], complete=False)})
    assert cs.validate(v, items, 'S') == []
    assert cs.score_rater(items, v, {}, 'blind')['parts']['clean']['complete'] == 0


def test_agreement_and_kappa():
    items = [item('a', 'clean'), item('b', 'clean')]
    va = vfile({'a': unit_verdicts(['absent'] * 3), 'b': unit_verdicts(['present', 'absent', 'absent'])})
    vb = vfile({'a': unit_verdicts(['absent'] * 3), 'b': unit_verdicts(['absent'] * 3)})
    ag = cs.agreement(items, va, vb)
    assert ag['units_both_complete'] == 2 and ag['corners'] == 6
    assert ag['corner_agree']['k'] == 5
    assert ag['unit_outcome_agree']['k'] == 1
    assert cs.cohen_kappa([('a', 'a')] * 3, cats=('a', 'b')) is None


# --------------------------------------------------------------------------------- page

def test_render_has_no_placeholders_and_no_part():
    it = cg.build_item(full_record(), 'na_noramp', {}, {})
    html = cg.build_html([cg.viewer_unit(it, '../../')], 'S' * 64, 'jonf', None, 'Imagery: Esri')
    assert '__' not in html.replace('verdicts__', '')
    assert 'na_noramp' not in html                       # the part is never sent to the page
    assert '"verdicts__jonf.json"' in html
    assert cg.RUBRIC.splitlines()[0] in html
    with pytest.raises(ValueError):
        cg.rater_file_name('bad name')


NODE = shutil.which('node')


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_bootstrap_state_under_node(tmp_path):
    units = [{'id': 'u1', 'corners': [{'k': '0'}, {'k': '1'}]},
             {'id': 'u2', 'corners': [{'k': '0'}]}, {'id': 'u3', 'corners': [{'k': '0'}]}]
    initial = {'items_sha256': 'S', 'units': {
        'u1': {'corners': {'0': {'verdict': 'absent', 'absent_kind': 'no_sidewalk'},
                           '1': {'verdict': 'present', 'absent_kind': None}},
               'blind': {'0': {'verdict': 'absent'}, '1': {'verdict': 'present'}},
               'complete': True, 'inventory_seen': True},
        'u2': {'corners': {'0': {'verdict': 'present'}}, 'complete': True, 'blind': {'0': {'verdict': 'present'}}},
        'u3': {'corners': {'0': {'verdict': 'absent'}, '9': {'verdict': 'absent'}},
               'blind': {'0': {'verdict': 'absent'}, '9': {'verdict': 'absent'}}, 'complete': True},
        'gone': {'corners': {}, 'complete': False}}}
    local = {'u2': {'corners': {'0': {'verdict': 'absent', 'absent_kind': None}}, 'complete': False,
                    'seen': True}}
    js = cgp.STATE_BOOTSTRAP_JS + f"""
const r = bootstrapState({json.dumps(initial)}, {json.dumps(local)}, {json.dumps(units)}, 'S');
const ig = bootstrapState({json.dumps(initial)}, {{}}, {json.dumps(units)}, 'OTHER');
console.log(JSON.stringify({{r: r, ig: ig}}));
"""
    p = tmp_path / 'b.js'
    p.write_text(js, encoding='utf-8')
    out = json.loads(subprocess.run([NODE, str(p)], capture_output=True, text=True, check=True).stdout)
    r, ig = out['r'], out['ig']
    assert r['prefilled'] == 3 and r['conflicts'] == ['u2'] and r['reopened'] == 1
    assert r['state']['u1']['corners']['0'] == {'verdict': 'absent', 'absent_kind': 'no_sidewalk'}
    assert r['state']['u1']['complete'] is True
    assert r['state']['u2']['corners']['0']['verdict'] == 'absent'        # local work wins
    assert '9' not in r['state']['u3']['corners'] and r['state']['u3']['complete'] is False
    assert 'gone' in r['state']                                             # kept verbatim
    assert ig['initialIgnored'] is True and ig['prefilled'] == 0
    assert ig['state']['u1']['corners']['0']['verdict'] is None


# ------------------------------------------------------------------ committed bundle

@pytest.mark.skipif(not (BUNDLE / 'snapshot.json').exists(), reason='bundle not built')
def test_committed_bundle_is_consistent():
    snap = json.loads((BUNDLE / 'snapshot.json').read_text(encoding='utf-8'))
    assert cs.sha256_file(BUNDLE / 'items.jsonl') == snap['items_sha256']
    items = cg.read_jsonl(BUNDLE / 'items.jsonl')
    assert snap['seed'] == cg.SEED and snap['populations'] == cg.EXPECTED_POPULATION
    parts = {p: sorted(i['unit'] for i in items if i['part'] == p) for p in cg.PARTS}
    assert parts == {p: sorted(snap['draw'][p]) for p in cg.PARTS}
    assert {p: len(v) for p, v in parts.items()} == {'false_absence': 35, 'na_noramp': 40, 'clean': 20}
    assert [i['unit'] for i in items] == cg.review_order([i['unit'] for i in items])
    assert all(c['views'] for i in items for c in i['corners'])
