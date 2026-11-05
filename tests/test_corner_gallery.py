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
EXPECTED_POPS = {'false_absence': 35, 'na_noramp': 944, 'clean': 333}
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


def test_choose_views_always_includes_the_newest_capture():
    """B1 of the review: three near 2014 panos must not hide a farther 2021 one."""
    c = [{'pano_id': 'a', 'dist_m': 5.0, 'capture_date': '2014-08'},
         {'pano_id': 'b', 'dist_m': 6.0, 'capture_date': '2014-08'},
         {'pano_id': 'c', 'dist_m': 7.0, 'capture_date': '2014-08'},
         {'pano_id': 'd', 'dist_m': 30.0, 'capture_date': '2021-12'},
         {'pano_id': 'e', 'dist_m': 25.0, 'capture_date': '2021-12'},
         {'pano_id': 'z', 'dist_m': 45.0, 'capture_date': '2025-01'}]     # outside 40 m
    got = cg.choose_views(c)
    assert [v['pano_id'] for v in got] == ['a', 'b', 'e']    # nearest of the newest date
    assert cg.newest_date(got) == '2021-12'
    # the newest pano is too near (< 3 m): still shown, the rest fill by distance
    c2 = [{'pano_id': 'n', 'dist_m': 1.0, 'capture_date': '2023-01'},
          {'pano_id': 'a', 'dist_m': 5.0, 'capture_date': '2014-08'},
          {'pano_id': 'b', 'dist_m': 6.0, 'capture_date': '2014-08'},
          {'pano_id': 'c', 'dist_m': 7.0, 'capture_date': '2014-08'}]
    assert [v['pano_id'] for v in cg.choose_views(c2)] == ['a', 'b', 'n']
    # undated candidates count as oldest; nothing dated -> plain distance order
    c3 = [{'pano_id': 'a', 'dist_m': 5.0, 'capture_date': None},
          {'pano_id': 'b', 'dist_m': 9.0, 'capture_date': '2019-01'}]
    assert [v['pano_id'] for v in cg.choose_views(c3, k=1)] == ['b']
    assert cg.choose_views([]) == []


def test_inventory_position_resolves_duplicate_ids_by_position():
    near, far = FR.to_latlng(5, 0), FR.to_latlng(5000, 0)
    inv = {'CR1': [(far[0], far[1], 'Available'), (near[0], near[1], 'NA_noramp')]}
    assert cg.inventory_position(inv, 'CR1', 'NA_noramp', LAT0, LNG0) == near
    assert cg.inventory_position(inv, 'CR1', 'Available', LAT0, LNG0) == (None, None)
    assert cg.inventory_position(inv, 'CR9', 'Available', LAT0, LNG0) == (None, None)
    # N8: two same-id same-class points inside the window are ambiguous; co-located is fine
    twin = FR.to_latlng(-8, 3)
    with pytest.raises(ValueError, match='ambiguous'):
        cg.inventory_position({'CR2': [(near[0], near[1], 'NA_noramp'),
                                       (twin[0], twin[1], 'NA_noramp')]}, 'CR2', 'NA_noramp',
                              LAT0, LNG0)
    assert cg.inventory_position({'CR3': [(near[0], near[1], 'NA_noramp')] * 2}, 'CR3',
                                 'NA_noramp', LAT0, LNG0) == near


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


def test_image_manifest_and_check(tmp_path):
    """S4: every shown image is hashed, and check catches a changed, missing or unlisted one."""
    it = cg.build_item(full_record(), 'clean', {
        'p1': {'lat': LAT0, 'lng': LNG0 - 0.0001, 'heading': 0.0, 'capture_date': '2023-05',
               'run': 'results'}}, {})
    files = cg.shown_images([it])
    assert files == ['aerial/vancouver_res_n1.jpg', 'crops/vancouver_res_n1_c0_p1.jpg',
                     'crops/vancouver_res_n1_c1_p1.jpg']
    for n, f in enumerate(files):
        (tmp_path / f).parent.mkdir(exist_ok=True)
        (tmp_path / f).write_bytes(bytes([n]) * 10)
    meta = cg.write_image_manifest(tmp_path, [it])
    assert meta['n_aerial'] == 1 and meta['n_crop'] == 2
    snap = {'images': meta}
    assert cg.check_images(tmp_path, snap, [it]) == []
    (tmp_path / files[1]).write_bytes(b'other pixels')
    (tmp_path / files[2]).unlink()
    it['corners'][0]['views'].append(dict(it['corners'][0]['views'][0], crop='crops/new.jpg'))
    p = '\n'.join(cg.check_images(tmp_path, snap, [it]))
    assert 'c0_p1.jpg: sha256 differs' in p and 'c1_p1.jpg: missing' in p
    assert 'crops/new.jpg: shown to the rater but not in the manifest' in p
    (tmp_path / cg.IMAGES_MANIFEST).write_text('x', encoding='utf-8')
    assert any('not the one snapshot.json recorded' in x for x in cg.check_images(tmp_path, snap, [it]))
    assert cg.check_images(tmp_path, {}, [it])[0].startswith('snapshot.json has no image manifest')


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


def test_false_absence_dating_uses_shown_and_records_available():
    it = item('u', 'false_absence', {0: ('Available', '2022-03-01')}, views_date='2014-08')
    it['corners'][0]['newest_available'] = '2021-12'
    # judged on the 2014 crop: the 2022 install is after it, so built-after-imagery
    verdicts = {'0': 'absent', '1': 'absent', '2': 'absent'}
    assert cs.classify_false_absence(it, verdicts) == 'artifact_built_after_imagery'
    d = cs.fa_dating(it)
    assert d == {'0': {'newest_shown': '2014-08', 'newest_available': '2021-12',
                       'instdates': ['2022-03']}}
    sc = cs.score_rater([it], vfile({'u': unit_verdicts(['absent'] * 3)}), {}, 'blind')
    assert sc['parts']['false_absence']['shown_older_than_available'] == ['u']


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
    assert cs.validate(good, items, 'S', rubric=cg.RUBRIC) == []
    assert any('rubric text differs' in x
               for x in cs.validate(good, items, 'S', rubric=cg.RUBRIC + 'edited'))
    # N10: final differs from blind with no edited_after_inventory flag -> altered file
    unflagged = vfile({'u1': unit_verdicts(['present', 'absent', 'absent'], blind=['absent'] * 3)})
    assert any('edited_after_inventory is false' in x for x in cs.validate(unflagged, items, 'S'))
    unflagged['units']['u1']['edited_after_inventory'] = True
    assert cs.validate(unflagged, items, 'S') == []
    assert cg.portable_path(cg.REPO_ROOT / 'runs' / 'x.json') == 'runs/x.json'
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


def test_partial_file_report_is_interim_and_ci_not_degenerate():
    """Review N2: a partial file is marked INTERIM with the scored count, and a stratum at
    k/n = 1 still contributes variance (the old CI was [1, 1] when every stratum was at 1)."""
    items = ([item(f'n{i}', 'na_noramp', {0: ('NA_noramp', None)}) for i in range(3)] +
             [item('c0', 'clean'), item('c1', 'clean'),
              item('f0', 'false_absence', {1: ('Available', '2019-01-01')}),
              item('f1', 'false_absence', {1: ('Available', '2019-01-01')})])
    units = {u: unit_verdicts(['absent'] * 3) for u in ('n0', 'n1', 'c0', 'f0')}
    snap = {'items_sha256': 'S', 'seed': 243,
            'populations': {'false_absence': 2, 'na_noramp': 40, 'clean': 20}}
    b = cs.score_rater(items, vfile(units), snap, 'blind')
    st = b['stratified']
    assert st['estimate'] == 1.0 and st['ci95'][0] < 0.95
    rep = cs.render_report(snap, items, {'jonf': {'blind': b, 'final': b}}, {'jonf': []})
    assert '**INTERIM: 4 of 7 units rated.**' in rep
    assert '### The 2 false absences (1 of 2 scored)' in rep


def test_incomplete_units_are_not_scored():
    items = [item('c0', 'clean')]
    v = vfile({'c0': unit_verdicts(['absent', None, None], complete=False)})
    assert cs.validate(v, items, 'S') == []
    assert cs.score_rater(items, v, {}, 'blind')['parts']['clean']['scored'] == 0


def test_reopened_unit_keeps_its_blind_read():
    """S5: completed (blind frozen), reopened, edited and not completed again: the blind read
    still scores it; the final read does not."""
    items = [item('c0', 'clean'), item('c1', 'clean')]
    reopened = unit_verdicts(['present', 'absent', 'absent'], blind=['absent'] * 3, edited=True)
    reopened['complete'] = False
    v = vfile({'c0': reopened, 'c1': unit_verdicts(['absent'] * 3)})
    assert cs.validate(v, items, 'S') == []
    b = cs.score_rater(items, v, {}, 'blind')
    assert b['scored'] == 2 and b['complete'] == 1 and b['reopened_with_blind'] == 1
    assert b['parts']['clean']['outcomes']['absent'] == 2
    f = cs.score_rater(items, v, {}, 'final')
    assert f['scored'] == 1 and f['parts']['clean']['outcomes'] == {'present': 0, 'absent': 1,
                                                                    'undetermined': 0}
    vb = vfile({'c0': reopened, 'c1': unit_verdicts(['absent'] * 3)})
    assert cs.agreement(items, v, vb)['units_both_blind'] == 2


def test_agreement_and_kappa():
    items = [item('a', 'clean'), item('b', 'clean')]
    va = vfile({'a': unit_verdicts(['absent'] * 3), 'b': unit_verdicts(['present', 'absent', 'absent'])})
    vb = vfile({'a': unit_verdicts(['absent'] * 3), 'b': unit_verdicts(['absent'] * 3)})
    ag = cs.agreement(items, va, vb)
    assert ag['units_both_blind'] == 2 and ag['corners'] == 6
    assert ag['corner_agree']['k'] == 5
    assert ag['unit_outcome_agree']['k'] == 1
    assert cs.cohen_kappa([('a', 'a')] * 3, cats=('a', 'b')) is None


def test_live_imagery_units_are_reported_with_and_without(tmp_path):
    """PR #141 review S2: units rated with live imagery (page flag, or the hand-recorded
    sidecar for units rated before the flag) are named, and the blind read is also given
    without them; the sidecar must quote the unit's note it rests on."""
    items = [item('f0', 'false_absence', {0: ('Available', '2019-01-01')}),
             item('f1', 'false_absence', {0: ('Available', '2019-01-01')}),
             item('c0', 'clean')]
    v = vfile({'f0': dict(unit_verdicts(['present', 'absent', 'absent']), note='I went into PS'),
               'f1': unit_verdicts(['absent'] * 3),
               'c0': dict(unit_verdicts(['absent'] * 3), used_live_imagery=True)})
    assert cs.validate(v, items, 'S') == []
    side = {'schema': cs.LIVE_SCHEMA, 'rater': 'jonf', 'items_sha256': 'S',
            'units': {'f0': {'used_live_imagery': True, 'note_quote': 'went into PS'}}}
    assert cs.validate_live_sidecar(side, v, 'S') == []
    assert cs.live_units(v, side) == ['c0', 'f0']
    w = cs.score_rater(items, v, {}, 'blind', cs.live_units(v, side))
    assert w['scored'] == 1 and w['parts']['false_absence']['real_misses'] == 0
    assert cs.score_rater(items, v, {}, 'blind')['parts']['false_absence']['real_misses'] == 1
    bad = json.loads(json.dumps(side))
    bad['units']['f0']['note_quote'] = 'checked Street View'
    bad['units']['zz'] = {'used_live_imagery': True, 'note_quote': 'x'}
    p = '\n'.join(cs.validate_live_sidecar(bad, v, 'X'))
    assert 'items_sha256' in p and 'zz is not rated' in p and 'note does not contain' in p
    v['units']['c0']['used_live_imagery'] = 'yes'
    assert any('used_live_imagery' in x for x in cs.validate(v, items, 'S'))
    # end to end: the report names them and gives the read without them
    bundle = tmp_path / 'b'
    bundle.mkdir()
    lines = [json.dumps(i) for i in items]
    (bundle / 'items.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sha = cs.sha256_file(bundle / 'items.jsonl')
    v['units']['c0']['used_live_imagery'] = True
    v['items_sha256'] = side['items_sha256'] = sha
    (bundle / 'snapshot.json').write_text(json.dumps(
        {'items_sha256': sha, 'seed': 243, 'rubric': cg.RUBRIC,
         'populations': {'false_absence': 2, 'na_noramp': 40, 'clean': 20}}), encoding='utf-8')
    (bundle / 'verdicts__jonf.json').write_text(json.dumps(v), encoding='utf-8')
    (bundle / 'live_imagery__jonf.json').write_text(json.dumps(side), encoding='utf-8')
    assert cs.main([str(bundle)]) == 0
    rep = (bundle / 'score/report.md').read_text(encoding='utf-8')
    assert '2 unit(s): `c0`, `f0`' in rep and 'real-miss share 0/1' in rep
    sc = json.loads((bundle / 'score/score.json').read_text(encoding='utf-8'))
    assert sc['results']['jonf']['used_live_imagery'] == ['c0', 'f0']


# ------------------------------------------------- frozen blind reads vs HEAD (review SF1)

def test_blind_regressions():
    """PR #141 final review SF1: a blind read frozen in the committed file that the new file
    lost, removed or changed is a regression; added units, post-reveal edits and drafts are not."""
    old = vfile({'u1': unit_verdicts(['absent'] * 3), 'u2': unit_verdicts(['present'] * 3),
                 'd': unit_verdicts(['absent', None, None], complete=False)})
    assert cs.blind_regressions(old, json.loads(json.dumps(old))) == []          # unchanged
    grown = json.loads(json.dumps(old))
    grown['units']['u3'] = unit_verdicts(['absent'] * 3)                          # new units added
    grown['units']['u1'] = unit_verdicts(['present', 'absent', 'absent'], blind=['absent'] * 3,
                                         edited=True)                            # edited after reveal
    del grown['units']['d']                                                      # a draft dropped
    assert cs.blind_regressions(old, grown) == []
    lost = json.loads(json.dumps(old))
    del lost['units']['u1']                                                      # missing unit
    lost['units']['u2']['blind']['0']['verdict'] = 'absent'                      # changed blind
    assert cs.blind_regressions(old, lost) == [
        'u1: missing (its blind read is frozen in HEAD)',
        'u2: its blind read differs from the one frozen in HEAD']
    lost['units']['u2']['blind'] = None
    assert cs.blind_regressions(old, lost)[1] == 'u2: its blind read was removed'


GIT = shutil.which('git')


def _git(cwd, *a):
    subprocess.run([GIT, '-C', str(cwd), *a], check=True, capture_output=True)


@pytest.mark.skipif(GIT is None, reason='git not installed')
def test_score_refuses_an_export_that_lost_blind_reads_frozen_in_head(tmp_path, monkeypatch, capsys):
    """SF1 end to end: the scorer (doc step 6, run before every commit) compares the working
    verdicts file with HEAD's, refuses (exit 2, nothing written) when a frozen blind read is
    missing, says the export came from a stale page, and scores with --allow-blind-change. No
    file in HEAD yet, or no git checkout at all, scores with a note."""
    for k, v in {'GIT_CONFIG_GLOBAL': str(tmp_path / 'empty.gitconfig'), 'GIT_CONFIG_NOSYSTEM': '1',
                 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't',
                 'GIT_COMMITTER_EMAIL': 't@t'}.items():
        monkeypatch.setenv(k, v)
    (tmp_path / 'empty.gitconfig').write_text('', encoding='utf-8')
    bundle = tmp_path / 'repo' / 'b'
    bundle.mkdir(parents=True)
    items = [item('c0', 'clean'), item('c1', 'clean')]
    (bundle / 'items.jsonl').write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')
    sha = cs.sha256_file(bundle / 'items.jsonl')
    (bundle / 'snapshot.json').write_text(json.dumps(
        {'items_sha256': sha, 'seed': 243, 'rubric': cg.RUBRIC,
         'populations': {'false_absence': 2, 'na_noramp': 40, 'clean': 20}}), encoding='utf-8')
    vpath = bundle / 'verdicts__jonf.json'
    full = vfile({'c0': unit_verdicts(['absent'] * 3), 'c1': unit_verdicts(['absent'] * 3)}, sha=sha)
    vpath.write_text(json.dumps(full), encoding='utf-8')
    assert cs.head_bytes(vpath) == (None, 'not a git checkout')
    _git(bundle.parent, 'init', '-q')
    assert cs.head_bytes(vpath)[1] == 'the checkout has no commit yet'
    _git(bundle.parent, 'add', 'b/items.jsonl', 'b/snapshot.json')
    _git(bundle.parent, 'commit', '-q', '-m', 'bundle')
    assert cs.head_bytes(vpath)[1] == 'not in HEAD yet (first commit)'
    assert cs.main([str(bundle)]) == 0                               # first commit: a note only
    _git(bundle.parent, 'add', 'b/verdicts__jonf.json')
    _git(bundle.parent, 'commit', '-q', '-m', 'verdicts')
    assert cs.head_bytes(vpath)[0] == vpath.read_bytes()
    assert cs.main([str(bundle)]) == 0                               # unchanged vs HEAD
    # an export from a page rendered before c1 was committed: valid, but c1's blind read is gone
    stale = vfile({'c0': unit_verdicts(['absent'] * 3)}, sha=sha)
    stale['prefilled_from_sha256'] = 'a' * 64
    vpath.write_text(json.dumps(stale), encoding='utf-8')
    assert cs.validate(stale, items, sha, cg.RUBRIC) == []
    shutil.rmtree(bundle / 'score')
    capsys.readouterr()
    assert cs.main([str(bundle)]) == 2 and not (bundle / 'score').exists()
    err = capsys.readouterr().err
    assert 'REFUSED: 1 blind read(s)' in err and 'c1: missing' in err and 'STALE page' in err
    assert cs.main([str(bundle), '--allow-blind-change']) == 0
    assert (bundle / 'score/report.md').exists()
    # more units on top of HEAD's is the normal case and passes
    grown = vfile({'c0': unit_verdicts(['absent'] * 3), 'c1': unit_verdicts(['absent'] * 3)}, sha=sha)
    grown['prefilled_from_sha256'] = None
    grown['units']['c0']['note'] = 'added later'
    vpath.write_text(json.dumps(grown), encoding='utf-8')
    assert cs.main([str(bundle)]) == 0


def test_validate_accepts_prefilled_from_as_optional():
    """SF1: exports carry prefilled_from_sha256 (null when the page had no file); files from
    before it (Jon's committed 50 units) still validate; a malformed value does not."""
    items = [item('u1', 'clean')]
    v = vfile({'u1': unit_verdicts(['absent'] * 3)})
    assert cs.validate(v, items, 'S') == []
    for ok in (None, 'ab' * 32):
        assert cs.validate(dict(v, prefilled_from_sha256=ok), items, 'S') == []
    assert any('prefilled_from_sha256' in x
               for x in cs.validate(dict(v, prefilled_from_sha256='stale'), items, 'S'))


# --------------------------------------------------------------------------------- page

def test_render_has_no_placeholders_and_no_part():
    it = cg.build_item(full_record(), 'na_noramp', {}, {})
    html = cg.build_html([cg.viewer_unit(it, '../../')], 'S' * 64, 'jonf', None, 'Imagery: Esri')
    assert '__' not in html.replace('verdicts__', '')
    assert 'na_noramp' not in html                       # the part is never sent to the page
    # S3: no inventory in the page before the reveal (status, RAMPTYPE, INSTDATE, ids)
    for leak in ('2025-09-12', 'PERP', 'CR1', '"instdate"', '"ramptype"', '"inventory"'):
        assert leak not in html, leak
    assert '"reveal": "reveal/vancouver_res_n1.js"' in html
    assert '"verdicts__jonf.json"' in html
    assert cg.RUBRIC.splitlines()[0] in html
    # review S2: the live-imagery checkbox, set by opening a look-around link, is exported
    assert 'id="liveimg"' in html and 'markLive(); return;' in html
    assert 'used_live_imagery: !!s.used_live_imagery' in html
    # final review SF1: the export names the file the page was rendered from; N5: the hint is
    # tied to the checkbox
    assert 'const PREFILLED_FROM = null;' in html and 'prefilled_from_sha256: PREFILLED_FROM' in html
    assert 'const PREFILLED_FROM = "' + 'ab' * 32 + '";' in cg.build_html(
        [cg.viewer_unit(it, '../../')], 'S' * 64, 'jonf', None, 'Imagery: Esri', 'ab' * 32)
    assert 'aria-describedby="liveimghint"' in html and 'id="liveimghint"' in html
    # N1: the page cannot know which machine completed a unit first
    assert 'completed on another machine first' not in html
    assert 'already holds a frozen blind read for the unit; it was kept' in html
    with pytest.raises(ValueError):
        cg.rater_file_name('bad name')


NODE = shutil.which('node')


def test_viewer_unit_carries_heading_and_dates():
    """Review S4(a): each view's heading is the camera-to-corner bearing (the look-around
    links open at it), its date is the capture date, and the aerial carries its dates."""
    panos = {'p1': {'lat': FR.to_latlng(-10, 0)[0], 'lng': FR.to_latlng(-10, 0)[1], 'heading': 0.0,
                    'capture_date': '2023-05', 'run': 'results'}}
    it = cg.build_item(full_record(), 'clean', panos, {})
    vu = cg.viewer_unit(it, '../../', {'vancouver:res:n1': ['2025-04-14', '2025-09-19']})
    assert vu['aerial']['dates'] == ['2025-04-14', '2025-09-19']
    assert cg.viewer_unit(it, '../../')['aerial']['dates'] == []
    for c_it, c_vu in zip(it['corners'], vu['corners']):
        for v_it, v_vu in zip(c_it['views'], c_vu['views']):
            assert v_vu['heading'] == v_it['bearing_deg'] and v_vu['date'] == '2023-05'
            assert v_vu['crop'] == '../../' + v_it['crop'] and v_vu['cam'] == v_it['cam']


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_pano_links_under_node(tmp_path):
    """Review S4(b): the two look-around URLs, lat/lng order, heading and pitch, &amp; for
    innerHTML, and the new-tab / look-only wording (N4, N5)."""
    v = {'pano_id': 'J1zLlmbrSzYZYo6kwyArwA', 'heading': 55.3,
         'cam': {'lat': 45.6252251, 'lng': -122.5291138}}
    js = cgp.PANO_LINKS_JS + f"console.log(panoLinks({json.dumps(v)}, {json.dumps(cg.PS_HOST)}));\n"
    p = tmp_path / 'l.js'
    p.write_text(js, encoding='utf-8')
    html = subprocess.run([NODE, str(p)], capture_output=True, text=True, check=True).stdout
    assert ('href="https://sidewalk-vancouver.cs.washington.edu/explore?lat=45.6252251&amp;'
            'lng=-122.5291138&amp;panoId=J1zLlmbrSzYZYo6kwyArwA&amp;heading=55&amp;pitch=-20"') in html
    assert ('href="https://www.google.com/maps/@45.6252251,-122.5291138,3a,90y,55h,70t/data='
            '!3m4!1e1!3m2!1sJ1zLlmbrSzYZYo6kwyArwA!2e0"') in html
    assert html.count('(opens in a new tab)') == 2 and 'do not label there' in html
    assert '<span aria-hidden="true">(each opens in a new tab)</span>' in html    # N5


def test_load_aerial_dates_refuses_another_item_list(tmp_path):
    """Review S4(c): dates made on another items.jsonl are not shown."""
    doc = {'items_sha256': 'A', 'units': {'u1': {'dates': ['2025-09-19']}}}
    (tmp_path / cg.AERIAL_DATES_FILE).write_text(json.dumps(doc), encoding='utf-8')
    assert cg.load_aerial_dates(tmp_path, 'A') == {'u1': ['2025-09-19']}
    assert cg.load_aerial_dates(tmp_path, 'B') == {}
    assert cg.load_aerial_dates(tmp_path / 'none', 'A') == {}


def test_esri_metadata_picks_the_layer_drawn_at_the_zoom(monkeypatch):
    """Review S4(d): the layer whose Min/MaxMapLevel covers the zoom, 'Null' dates skipped,
    M/D/YYYY -> ISO. No network: urlopen is replaced."""
    import io
    import urllib.request

    def layer(name, lo, hi, date, release='R1'):
        return {'layerName': name, 'attributes': {'MinMapLevel': lo, 'MaxMapLevel': hi,
                                                  'SRC_DATE2': date, 'ReleaseName': release,
                                                  'SRC_DESC': 'src'}}
    res = [layer('Imagery Footprints', 0, 23, '1/1/2000'),          # not a metadata layer
           layer('0.3m Resolution Metadata', 12, 19, '12/30/2025'),
           layer('0.15m Resolution Metadata', 20, 23, 'Null'),
           layer('0.15m Resolution Metadata', 'x', 23, '1/1/2001'),  # unreadable levels
           layer('0.15m Resolution Metadata', 20, 23, '9/19/2025', 'R2')]
    seen = []

    def fake(url, timeout=None):
        seen.append(url)
        return io.BytesIO(json.dumps({'results': res}).encode('utf-8'))
    monkeypatch.setattr(urllib.request, 'urlopen', fake)
    assert cg.esri_metadata_at(45.6, -122.6, 20) == {
        'date': '2025-09-19', 'layer': '0.15m Resolution Metadata', 'release': 'R2', 'source': 'src'}
    assert cg.esri_metadata_at(45.6, -122.6, 17)['date'] == '2025-12-30'
    assert cg.esri_metadata_at(45.6, -122.6, 5) is None
    assert 'geometry=-122.6%2C45.6' in seen[0]                       # lng,lat order


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_reveal_script_delivers_inventory(tmp_path):
    it = cg.build_item(full_record(), 'false_absence', {}, {})
    js = ("let got = null; function revealInventory(id, d) { got = [id, d]; }\n" +
          cg.reveal_script(it) + "console.log(JSON.stringify(got));\n")
    p = tmp_path / 'r.js'
    p.write_text(js, encoding='utf-8')
    got = json.loads(subprocess.run([shutil.which('node'), str(p)], capture_output=True, text=True,
                                    check=True).stdout)
    assert got[0] == 'vancouver:res:n1'
    assert got[1]['0'][0]['instdate'] == '2025-09-12' and got[1]['1'] == []


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_bootstrap_state_under_node(tmp_path):
    units = [{'id': 'u1', 'corners': [{'k': '0'}, {'k': '1'}]},
             {'id': 'u2', 'corners': [{'k': '0'}]}, {'id': 'u3', 'corners': [{'k': '0'}]}]
    initial = {'items_sha256': 'S', 'units': {
        'u1': {'corners': {'0': {'verdict': 'absent', 'absent_kind': 'no_sidewalk'},
                           '1': {'verdict': 'present', 'absent_kind': None}},
               'blind': {'0': {'verdict': 'absent'}, '1': {'verdict': 'present'}},
               'complete': True, 'inventory_seen': True},
        # an unfinished draft in the file (no blind read): local work on it wins
        'u2': {'corners': {'0': {'verdict': 'present'}}, 'complete': False, 'blind': None},
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


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_bootstrap_never_drops_a_blind_read_frozen_in_the_file(tmp_path):
    """PR #141 review S1: X started on machine A, completed on B (blind frozen, inventory
    shown), exported and committed; A reopens. A must take B's blind read, not keep its
    own pre-completion draft (which would freeze a 'blind' read given after the reveal)."""
    units = [{'id': 'X', 'corners': [{'k': '0'}, {'k': '1'}]},
             {'id': 'Y', 'corners': [{'k': '0'}]}]
    blind_b = {'0': {'verdict': 'absent', 'absent_kind': None},
               '1': {'verdict': 'present', 'absent_kind': None}}
    initial = {'items_sha256': 'S', 'units': {
        'X': {'corners': blind_b, 'blind': blind_b, 'complete': True, 'inventory_seen': True},
        'Y': {'corners': {'0': {'verdict': 'absent', 'absent_kind': None}},
              'blind': {'0': {'verdict': 'absent', 'absent_kind': None}},
              'complete': True, 'inventory_seen': True}}}
    local = {'X': {'corners': {'0': {'verdict': 'present', 'absent_kind': None},
                               '1': {'verdict': None, 'absent_kind': None}},
                   'blind': None, 'complete': False, 'seen': True, 'inventory_seen': False},
             # both machines completed Y independently, with different blind reads
             'Y': {'corners': {'0': {'verdict': 'present', 'absent_kind': None}},
                   'blind': {'0': {'verdict': 'present', 'absent_kind': None}},
                   'complete': True, 'seen': True, 'inventory_seen': True}}
    js = cgp.STATE_BOOTSTRAP_JS + f"""
const r = bootstrapState({json.dumps(initial)}, {json.dumps(local)}, {json.dumps(units)}, 'S');
console.log(JSON.stringify(r));
"""
    p = tmp_path / 's1.js'
    p.write_text(js, encoding='utf-8')
    r = json.loads(subprocess.run([NODE, str(p)], capture_output=True, text=True, check=True).stdout)
    x, y = r['state']['X'], r['state']['Y']
    assert x['blind'] == blind_b and x['inventory_seen'] is True and x['complete'] is True
    assert x['corners'] == blind_b                     # the draft is superseded whole
    assert y['blind'] == initial['units']['Y']['blind'] and y['inventory_seen'] is True
    assert y['corners']['0']['verdict'] == 'present'   # later local work kept, flagged
    assert y['edited_after_inventory'] is True
    assert sorted(r['blindFromFile']) == ['X', 'Y'] and r['conflicts'] == ['Y']


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_plan_key_absent_then_next_key_advances(tmp_path):
    """B2 of the review: after `a`, p / a / t rate the NEXT corner; b / n still describe the
    corner just rated absent; once all are rated, a further verdict key is refused."""
    js = cgp.KEYS_JS + """
const out = [];
let v = [null, null, null], active = 0, last = null;
function step(k) {
  const r = planKey(v, active, last, k);
  if (r.op === 'verdict') v[r.ci] = r.value;
  if (r.op === 'kind') out.push(['kind', r.ci, r.value]);
  active = r.active; last = r.last;
  out.push([k, r.op, r.ci === undefined ? null : r.ci, active, v.slice()]);
}
step('a'); step('p'); step('b');           // b with no absent corner just rated: refused
step('t');                                  // all rated now, active stays on corner 2
step('p');                                  // refused: would overwrite corner 2
active = 0; last = null; step('b');         // the rater picks corner 0 (absent): b applies
// a then b: b goes to the absent corner although the next corner is active
v = [null, null]; active = 0; last = null; step('a'); step('n');
// N6: b on a present corner is refused
v = ['present', null]; active = 0; last = null; step('b');
// re-review S1: b on the LAST corner must not lift the all-rated guard (p p a b p)
v = [null, null, null]; active = 0; last = null; step('p'); step('p'); step('a'); step('b'); step('p');
console.log(JSON.stringify(out));
"""
    p = tmp_path / 'k.js'
    p.write_text(js, encoding='utf-8')
    out = json.loads(subprocess.run([NODE, str(p)], capture_output=True, text=True, check=True).stdout)
    assert out[0] == ['a', 'verdict', 0, 1, ['absent', None, None]]
    assert out[1] == ['p', 'verdict', 1, 2, ['absent', 'present', None]]   # not corner 0
    assert out[2][:2] == ['b', 'refuse']
    assert out[3] == ['t', 'verdict', 2, 2, ['absent', 'present', 'cant_tell']]
    assert out[4][:2] == ['p', 'refuse'] and out[4][4][2] == 'cant_tell'
    assert out[5] == ['kind', 0, 'curb_no_ramp'] and out[6][:2] == ['b', 'kind']
    assert out[7][:4] == ['a', 'verdict', 0, 1]
    assert out[8] == ['kind', 0, 'no_sidewalk']
    assert out[10][:2] == ['b', 'refuse']
    # re-review S1: the stray p after a, b on the last corner is refused; corner 2 stays absent
    assert out[-2][:2] == ['b', 'kind'] and out[-1][:2] == ['p', 'refuse']
    assert out[-1][4] == ['present', 'present', 'absent']


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
    # S4: the image manifest is committed and is the one snapshot.json recorded
    man = (BUNDLE / snap['images']['manifest']).read_bytes()
    assert hashlib.sha256(man).hexdigest() == snap['images']['manifest_sha256']
    listed = [line.split('  ', 1)[1] for line in man.decode().splitlines()]
    assert listed == cg.shown_images(items)
    assert snap['images']['n_aerial'] == len(items)
    # B1: every corner shows the newest capture date in its candidate pool
    for i in items:
        for c in i['corners']:
            assert cg.newest_date(c['views']) == (c['newest_available'] or ''), (i['unit'], c['corner'])
    # review S4(e): every committed verdicts file (human GT) still validates against the
    # bundle, and so does its live-imagery sidecar (review S2)
    for path in sorted(BUNDLE.glob('verdicts__*.json')):
        v = json.loads(path.read_text(encoding='utf-8'))
        assert cs.validate(v, items, snap['items_sha256'], snap['rubric']) == [], path.name
        side = cs.load_live_sidecar(BUNDLE, v['rater'])
        if side is not None:
            assert cs.validate_live_sidecar(side, v, snap['items_sha256']) == [], path.name


# ------------------------------------------------------------------ reasons after the reveal

def test_reasons_added_after_reveal_are_not_a_reedit():
    """Adding b / n after the reveal leaves the verdicts alone: it carries its own flag
    (kinds_after_inventory), validates, is not counted as a verdict edit, and its reasons are
    reported on their own row, never in the blind read's kinds of absence (review S3: the
    revealed NA status can bias a no-sidewalk reason)."""
    items = [item('n0', 'na_noramp', {0: ('NA_noramp', None)})]
    u = unit_verdicts(['absent'] * 3, {0: 'no_sidewalk'}, blind=['absent'] * 3)
    v = vfile({'n0': u})
    assert any('kinds_after_inventory is false' in x for x in cs.validate(v, items, 'S'))
    u['kinds_after_inventory'] = True
    assert cs.validate(v, items, 'S') == []
    b = cs.score_rater(items, v, {}, 'blind')
    assert b['edited_after_inventory'] == 0 and b['kinds_after_inventory'] == 1
    assert b['kinds_from_after_corners'] == 1
    na = b['parts']['na_noramp']
    assert na['absent_kinds'] == {'curb_no_ramp': 0, 'no_sidewalk': 0, 'unspecified': 1}
    assert na['absent_kinds_added_after_reveal'] == {'curb_no_ramp': 0, 'no_sidewalk': 1}
    assert cs.kind_map(u, 'blind')['0'] is None and cs.after_kind_map(u) == {'0': 'no_sidewalk'}
    rep = cs.render_report({'items_sha256': 'S', 'seed': 243, 'populations': EXPECTED_POPS},
                           items, {'jonf': {'blind': b, 'final': b}}, {'jonf': []})
    assert "Reasons added after the reveal at those corners" in rep
    # an older export flagged the same change as a verdict edit: measured from the data
    u['kinds_after_inventory'] = False
    u['edited_after_inventory'] = True
    assert cs.validate(v, items, 'S') == []
    assert cs.score_rater(items, v, {}, 'blind')['edited_after_inventory'] == 0
    # a reason is never taken from after the reveal for a corner whose verdict changed
    w = unit_verdicts(['absent', 'absent', 'absent'], {0: 'no_sidewalk'},
                      blind=['present', 'absent', 'absent'], edited=True)
    assert cs.kind_map(w, 'blind')['0'] is None and cs.kinds_from_after(w) == 0


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_bootstrap_reflags_reason_only_edits(tmp_path):
    units = [{'id': 'u1', 'corners': [{'k': '0'}]}, {'id': 'u2', 'corners': [{'k': '0'}]}]
    blind = {'0': {'verdict': 'absent', 'absent_kind': None}}
    local = {'u1': {'corners': {'0': {'verdict': 'absent', 'absent_kind': 'curb_no_ramp'}}, 'blind': blind,
                    'complete': True, 'seen': True, 'inventory_seen': True, 'edited_after_inventory': True},
             'u2': {'corners': {'0': {'verdict': 'present', 'absent_kind': None}}, 'blind': blind,
                    'complete': True, 'seen': True, 'inventory_seen': True, 'edited_after_inventory': True}}
    js = cgp.STATE_BOOTSTRAP_JS + f"""
const r = bootstrapState(null, {json.dumps(local)}, {json.dumps(units)}, 'S');
console.log(JSON.stringify(r.state));
"""
    p = tmp_path / 'r.js'
    p.write_text(js, encoding='utf-8')
    st = json.loads(subprocess.run([NODE, str(p)], capture_output=True, text=True, check=True).stdout)
    assert st['u1']['edited_after_inventory'] is False and st['u1']['kinds_after_inventory'] is True
    assert st['u2']['edited_after_inventory'] is True       # a verdict changed: stays a re-edit
