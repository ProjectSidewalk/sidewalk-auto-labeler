"""corner_gallery.py build --items-from (the #169 precision gallery) and the scorer path
corner_labels.py score --sample --bundle, on synthetic units. CPU only, no network, no SSH:
crops are never cut (--no-crops) and aerials are skipped (--skip-aerial)."""
import csv
import hashlib
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

import corner_gallery as cg
import corner_gallery_score as cgs
import corner_labels as cl
import geo

LAT0, LNG0 = 45.6, -122.6
FR = geo.LocalFrame(LAT0, LNG0)
EMPTY = {'Available': 0, 'NA_noramp': 0, 'NA_typed': 0, 'RMV': 0, 'Expired/Removed': 0, 'other': 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit_record(unit):
    """A two-corner unit at (LAT0, LNG0), both corners absent, observed by p1 and p2."""
    corners = []
    for k, (start, width) in enumerate([(0.0, 90.0), (90.0, 270.0)]):
        b = math.radians(start + width / 2)
        lat, lng = FR.to_latlng(12 * math.sin(b), 12 * math.cos(b))
        corners.append({'corner': k, 'start_deg': start, 'width_deg': width, 'wide': width > 150,
                        'lat': lat, 'lng': lng, 'n_panos_25': 2, 'pano_ids_25': ['p1', 'p2'],
                        'inv_counts': dict(EMPTY), 'inventory': []})
    return {'unit': unit, 'type': 'residential', 'lat': LAT0, 'lng': LNG0, 'legs': [0.0, 90.0],
            'n_legs': 2, 'node_ids': [1], 'highways': ['residential'],
            'state': {'fusion/primary': 'absent'}, 'n_panos_25': 2, 'nearest_pano_m': 3.0,
            'inv_counts': dict(EMPTY), 'pano_ids_25': ['p1'], 'corners': corners}


def pano_line(pid, e, n, heading, date):
    lat, lng = FR.to_latlng(e, n)
    return {'detections': [], 'pano': {'panorama_id': pid, 'lat': lat, 'lng': lng,
                                       'camera_heading': heading, 'capture_date': date}}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def proposal(unit, corner, pid):
    return {'key': f'{unit}#{corner}', 'unit': unit, 'corner': corner, 'stratum': 'residential',
            'pano': {'pano_id': pid, 'run': 'base'}, 'x_normalized': 0.25, 'y_normalized': 0.6,
            'range_m': 11.0}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # split_figures needs pandas, which the test environment does not have; build only reads
    # two strings from it
    monkeypatch.setitem(sys.modules, 'split_figures',
                        SimpleNamespace(TILE_URL='https://tiles/{z}/{y}/{x}', ATTRIBUTION='Esri'))
    units = ['vancouver:res:n1', 'vancouver:res:n2', 'vancouver:res:n3']
    write_jsonl(tmp_path / 'corners_full.jsonl', [unit_record(u) for u in units])
    write_jsonl(tmp_path / 'run' / 'results.jsonl',
                [pano_line('p1', -10, 0, 0.0, '2023-05'), pano_line('p2', 0, -15, 90.0, '2024-06')])
    inv = tmp_path / 'inventory.geojson'
    inv.write_text(json.dumps({'type': 'FeatureCollection', 'features': []}), encoding='utf-8')
    b241 = tmp_path / 'build241.json'
    b241.write_text(json.dumps({'inputs': {
        'inventory': {'path': str(inv), 'sha256': sha(inv)},
        'results': {'path': str(tmp_path / 'run' / 'results.jsonl'),
                    'sha256': sha(tmp_path / 'run' / 'results.jsonl')}}}), encoding='utf-8')
    d = tmp_path / 'na_discount_sw'
    props = [proposal(u, c, 'p2') for u in units for c in (0, 1)]
    write_jsonl(d / 'proposals.jsonl', props)
    (d / 'build.json').write_text(json.dumps({
        'outputs': {'proposals': {'sha256': sha(d / 'proposals.jsonl')}},
        'inputs': {'corners_full': {'sha256': sha(tmp_path / 'corners_full.jsonl')}},
        'params': {'na_read': 'discount', 'na_read_settled': True},
        'tool': {'git_sha': 'a' * 40}, 'sidewalk_rule': {'applied': True}}), encoding='utf-8')
    sample = d / 'sample.csv'
    with open(sample, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['unit', 'corner'])
        w.writerows([['vancouver:res:n2', 1], ['vancouver:res:n1', 0], ['vancouver:res:n2', 0]])
    (d / 'sample.json').write_text(json.dumps({
        'seed': 169, 'n': 3, 'population': len(props), 'rule': 'r',
        'proposals': {'sha256': sha(d / 'proposals.jsonl')},
        'sample': {'sha256': sha(sample)}}), encoding='utf-8')
    bundle = tmp_path / 'corner_gallery169'
    argv = ['build', '--items-from', str(sample), '--corners-full',
            str(tmp_path / 'corners_full.jsonl'), '--build-json', str(b241), '--bundle',
            str(bundle), '--skip-aerial', '--no-crops']
    return SimpleNamespace(tmp=tmp_path, d=d, sample=sample, bundle=bundle, argv=argv)


def build(argv):
    with redirect_stdout(io.StringIO()):
        return cg.main(argv)


def test_items_from_builds_only_the_sampled_corners(setup):
    assert build(setup.argv) == 0
    b = setup.bundle
    items = cg.read_jsonl(b / 'items.jsonl')
    got = {i['unit']: [c['corner'] for c in i['corners']] for i in items}
    assert got == {'vancouver:res:n1': [0], 'vancouver:res:n2': [0, 1]}   # n3 not sampled
    assert all(i['part'] == cg.SAMPLE_PART and i['n_corners_unit'] == 2 for i in items)
    assert [i['unit'] for i in items] == cg.review_order(list(got))
    c = items[0]['corners'][0]
    assert c['proposal']['pano_id'] == 'p2' and c['proposal']['shown'] is True
    assert len(c['views']) == 2                       # the whole unit's pool, as gallery 243
    snap = json.loads((b / 'snapshot.json').read_text(encoding='utf-8'))
    assert snap['mode'] == 'sample' and snap['sample']['sha256'] == sha(setup.sample)
    assert snap['proposals']['sha256'] == sha(setup.d / 'proposals.jsonl')
    assert snap['draw'][cg.SAMPLE_PART][0] == 'vancouver:res:n2#1'      # file order kept
    assert snap['rubric'] == cg.RUBRIC and snap['rubric_version'] == cg.RUBRIC_VERSION
    assert snap['items_sha256'] == sha(b / 'items.jsonl')
    # no crops: CROPS_NEEDED.md and the crop worker's job say what is missing
    assert len(snap['crops_missing']) == 6
    job = json.loads((b / cg.CROP_JOB_FILE).read_text(encoding='utf-8'))
    assert len(job['items']) == 6 and job['panos_root'] == cg.STORE
    assert job['items'][0]['path'] in ('p1/p1.jpg', 'p2/p2.jpg')
    need = (b / cg.CROPS_NEEDED).read_text(encoding='utf-8')
    assert 'Crops needed: 6 of 6' in need and '--items-from' in need
    # the reveal stays the city inventory; the proposal is never sent to the page
    html = cg.build_html([cg.viewer_unit(i, '../../') for i in items], 'S' * 64, 'jonf', None, 'E')
    assert 'proposal' not in html and cg.SAMPLE_PART not in html
    # render refuses while the crops are missing
    with pytest.raises(SystemExit):
        build(['render', '--bundle', str(b), '--rater', 'jonf'])
    # a second build over the same sample keeps items.jsonl
    assert build(setup.argv) == 0 and sha(b / 'items.jsonl') == snap['items_sha256']


def test_items_from_refuses_a_sample_that_is_not_its_record(setup):
    with open(setup.sample, 'a', encoding='utf-8', newline='\n') as f:
        f.write('vancouver:res:n3,0\n')
    with pytest.raises(SystemExit, match='sha256'):
        build(setup.argv)


def test_crops_needed_is_removed_once_the_crops_exist(setup):
    build(setup.argv)
    items = cg.read_jsonl(setup.bundle / 'items.jsonl')
    for i in items:
        for c in i['corners']:
            for v in c['views']:
                (setup.bundle / v['crop']).parent.mkdir(exist_ok=True)
                (setup.bundle / v['crop']).write_bytes(b'jpeg')
    build(setup.argv)
    assert not (setup.bundle / cg.CROPS_NEEDED).exists()
    assert not (setup.bundle / cg.CROP_JOB_FILE).exists()


def export(bundle, verdicts):
    """A verdicts file as the page's Export writes it: blind = the first completion."""
    snap = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
    units = {}
    for uid, cs_ in verdicts.items():
        m = {str(k): {'verdict': v, 'absent_kind': ak} for k, (v, ak) in cs_.items()}
        units[uid] = {'type': 'residential', 'corners': m, 'blind': json.loads(json.dumps(m)),
                      'complete': True, 'inventory_seen': True, 'elapsed_s': 12, 'note': ''}
    doc = {'schema': cgs.SCHEMA, 'city': 'vancouver', 'items_sha256': snap['items_sha256'],
           'rater': 'jonf', 'rubric_version': snap['rubric_version'], 'rubric': snap['rubric'],
           'units': units}
    if snap.get('tag_set_version') is not None:          # the page exports the bundle's (#184)
        doc.update(tag_set_version=snap['tag_set_version'], tag_set=snap['tag_set'])
    path = bundle / cg.rater_file_name('jonf')
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path, doc


def score(setup, verdicts_path, *extra):
    out = setup.tmp / 'score.json'
    with redirect_stdout(io.StringIO()):
        cl.main(['score', '--proposals', str(setup.d / 'proposals.jsonl'), '--sample',
                 str(setup.sample), '--verdicts', str(verdicts_path), '--out', str(out), *extra])
    return json.loads(out.read_text(encoding='utf-8'))


def test_score_reads_the_bundle_export(setup):
    build(setup.argv)
    vpath, doc = export(setup.bundle, {
        'vancouver:res:n1': {0: ('absent', 'curb_no_ramp')},
        'vancouver:res:n2': {0: ('present', None), 1: ('absent', 'no_sidewalk')}})
    res = score(setup, vpath, '--bundle', str(setup.bundle))
    assert (res['absent'], res['present'], res['cant_tell']) == (2, 1, 0)
    assert res['n_proposals'] == 3 and res['sample_drawn_for_rule'] is True
    assert res['absent_kinds'] == {'curb_no_ramp': 1, 'no_sidewalk': 1, 'unspecified': 0}
    assert res['verdict'].startswith('NOT SCORED (n = 3 < 100)')
    assert res['bundle']['items_sha256'] == doc['items_sha256']
    assert res['verdicts']['items_sha256'] == doc['items_sha256']


def test_score_bundle_refusals(setup):
    build(setup.argv)
    vpath, doc = export(setup.bundle, {'vancouver:res:n1': {0: ('absent', None)}})
    with pytest.raises(SystemExit, match='--bundle needs --sample'):
        with redirect_stdout(io.StringIO()):
            cl.main(['score', '--proposals', str(setup.d / 'proposals.jsonl'), '--verdicts',
                     str(vpath), '--bundle', str(setup.bundle)])
    doc['items_sha256'] = 'f' * 64                       # an export made on other items
    vpath.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(SystemExit, match='cannot be scored'):
        score(setup, vpath, '--bundle', str(setup.bundle))
    _p, doc = export(setup.bundle, {'vancouver:res:n3': {0: ('absent', None)}})
    with pytest.raises(SystemExit, match='cannot be scored'):   # not a unit of this bundle
        score(setup, vpath, '--bundle', str(setup.bundle))
    # a sample other than the bundle's
    export(setup.bundle, {'vancouver:res:n1': {0: ('absent', None)}})
    other = setup.tmp / 'other.csv'
    other.write_text('unit,corner\nvancouver:res:n1,0\n', encoding='utf-8')
    probs = cl.check_bundle(setup.bundle, json.loads(vpath.read_text(encoding='utf-8')), other,
                            setup.d / 'proposals.jsonl')
    assert any('is not the sample the bundle was built from' in p for p in probs)


def test_generic_scorer_validates_a_sample_bundle_and_stops(setup, capsys):
    """corner_gallery_score.py on a sample bundle: validate, then point at corner_labels.py
    score instead of crashing on gallery 243's populations (KeyError 'na_noramp')."""
    build(setup.argv)
    export(setup.bundle, {'vancouver:res:n1': {0: ('absent', 'curb_no_ramp')},
                          'vancouver:res:n2': {0: ('absent', None), 1: ('cant_tell', None)}})
    out = setup.tmp / 'cgs_out'
    assert cgs.main([str(setup.bundle), '--out', str(out)]) == 0
    cap = capsys.readouterr()
    assert 'verdicts__jonf.json: valid' in cap.out
    assert "sample bundle" in cap.err and 'corner_labels.py score' in cap.err
    assert not out.exists()                                  # no gallery 243 report written


# ------------------------------------------- #169 revision (2026-11-06): split-wide bundle

def test_views_per_corner_and_landscape_job_items(setup):
    """--views-per-corner N keeps the newest-date slot and fills by distance; --crop-aspect 2
    writes the worker's optional landscape keys and records the geometry."""
    assert build(setup.argv + ['--views-per-corner', '1', '--crop-aspect', '2',
                               '--crop-fov-deg', '120', '--crop-px', '896']) == 0
    b = setup.bundle
    items = cg.read_jsonl(b / 'items.jsonl')
    for i in items:
        assert i['crop_geometry'] == {'aspect': 2.0, 'fov_h_deg': 120.0, 'fov_v_deg': 60.0,
                                      'px_w': 896, 'px_h': 448}
        for c in i['corners']:
            assert [v['capture_date'] for v in c['views']] == ['2024-06']   # newest slot kept
    snap = json.loads((b / 'snapshot.json').read_text(encoding='utf-8'))
    assert snap['views']['per_corner'] == 1
    assert snap['views']['crop_geometry']['px_h'] == 448
    job = json.loads((b / cg.CROP_JOB_FILE).read_text(encoding='utf-8'))
    assert all((j['fov_h_deg'], j['px_w'], j['px_h']) == (120.0, 896, 448) for j in job['items'])
    need = (b / cg.CROPS_NEEDED).read_text(encoding='utf-8')
    assert '--views-per-corner 1' in need and '--crop-aspect 2' in need
    # a re-run without the flags keeps the bundle's settings (it only cuts missing crops)
    assert build(setup.argv) == 0
    assert cg.read_jsonl(b / 'items.jsonl') == items
    report = (b / 'report.md').read_text(encoding='utf-8')
    assert 'up to 1 per corner' in report and '120 x 60 deg at 896 x 448 px' in report


def test_default_build_is_square_and_three_views(setup):
    """No new flag: the job items carry exactly the pre-#169 keys and items no new key."""
    assert build(setup.argv) == 0
    items = cg.read_jsonl(setup.bundle / 'items.jsonl')
    assert all('crop_geometry' not in i and 'phantom_legs' not in i for i in items)
    assert all('phantom_start' not in c for i in items for c in i['corners'])
    job = json.loads((setup.bundle / cg.CROP_JOB_FILE).read_text(encoding='utf-8'))
    assert {tuple(sorted(j)) for j in job['items']} == {
        ('fov_deg', 'name', 'pano_id', 'path', 'px', 'x', 'y')}
    assert {(j['fov_deg'], j['px']) for j in job['items']} == {(cg.CROP_FOV_DEG, cg.CROP_PX)}
    snap = json.loads((setup.bundle / 'snapshot.json').read_text(encoding='utf-8'))
    assert snap['views']['per_corner'] == cg.VIEWS_PER_CORNER == 3
    assert 'crop_geometry' not in snap['views']


def test_choose_views_k():
    c = [{'pano_id': p, 'dist_m': d, 'capture_date': dt} for p, d, dt in
         [('a', 5, '2014-08'), ('b', 6, '2014-08'), ('c', 7, '2014-08'), ('d', 30, '2021-12'),
          ('e', 8, '2014-08'), ('f', 9, '2014-08'), ('g', 10, '2014-08')]]
    assert [v['pano_id'] for v in cg.choose_views(c, 6)] == ['a', 'b', 'c', 'e', 'f', 'd']
    assert [v['pano_id'] for v in cg.choose_views(c, 3)] == ['a', 'b', 'd']
    assert [v['pano_id'] for v in cg.choose_views(c, 1)] == ['d']


def test_split_corner_table_carries_phantoms_to_the_page(setup):
    """A --split-wide corners_full: phantom legs reach the item and the page (dashed), per-corner
    phantom flags reach the legend; a unit without them sends no new key."""
    import corner_inventory as ci
    rec = unit_record('vancouver:res:n9')
    rec['legs'], rec['n_legs'], rec['phantom_legs'] = [90.0, 180.0, 270.0], 3, [0.0]
    for c, (s0, w, a, z) in zip(rec['corners'], ci.split_sectors([90.0, 180.0, 270.0], [0.0])):
        c.update(start_deg=s0, width_deg=w, wide=False, phantom_start=a, phantom_end=z)
    it = cg.build_item(rec, cg.SAMPLE_PART, {}, {}, 6, cg.crop_geometry(2, 120, 896))
    assert it['phantom_legs'] == [0.0]
    assert [(c['phantom_start'], c['phantom_end']) for c in it['corners']] == \
        [(True, False), (False, False)]
    vu = cg.viewer_unit(it, '../../')
    assert vu['phantom_legs'] == [0.0] and vu['crop_aspect'] == 2.0
    assert vu['corners'][0]['phantom'] == [True, False] and 'phantom' not in vu['corners'][1]
    html = cg.build_html([vu], 'S' * 64, 'jonf', None, 'E', None, 2, cg.RUBRIC_V2)
    assert 'stroke-dasharray="9 6"' in html and '"phantom_legs": [0.0]' in html
    assert cg.RUBRIC_V2.splitlines()[0] in html and 'const RUBRIC_VERSION = 2;' in html
    old = cg.viewer_unit(cg.build_item(unit_record('vancouver:res:n1'), 'p', {}, {}), '../../')
    assert 'phantom_legs' not in old and 'crop_aspect' not in old


def test_rubric_versions_kept_per_bundle(setup):
    """v1 is gallery 243's text verbatim; a new bundle gets v2; render shows the bundle's own
    rubric; the validators accept v1 and v2 but only the bundle's version."""
    assert cg.RUBRICS[1] == cg.RUBRIC_V1 and cg.RUBRIC_VERSION == 2
    assert cg.RUBRIC_V2.startswith('Rubric v2') and 'DASHED' in cg.RUBRIC_V2
    assert cg.RUBRIC_V1.splitlines()[1:] == cg.RUBRIC_V2.splitlines()[1:len(
        cg.RUBRIC_V1.splitlines())]
    snap243 = json.loads((cg.REPO_ROOT / 'runs/vancouver/corner_gallery243/snapshot.json')
                         .read_text(encoding='utf-8'))
    assert cg.rubric_for(snap243) == (1, cg.RUBRIC_V1)
    with pytest.raises(SystemExit, match='edited'):
        cg.rubric_for({'rubric_version': 1, 'rubric': cg.RUBRIC_V1 + 'x'})
    build(setup.argv)
    snap = json.loads((setup.bundle / 'snapshot.json').read_text(encoding='utf-8'))
    assert (snap['rubric_version'], snap['rubric']) == (2, cg.RUBRIC_V2)
    vpath, doc = export(setup.bundle, {'vancouver:res:n1': {0: ('absent', None)},
                                       'vancouver:res:n2': {0: ('present', None),
                                                            1: ('absent', None)}})
    res = score(setup, vpath, '--bundle', str(setup.bundle))
    assert res['verdicts']['rubric_version'] == 2
    items = cg.read_jsonl(setup.bundle / 'items.jsonl')
    doc = {k: v for k, v in doc.items() if k not in ('tag_set_version', 'tag_set')}   # rubric only
    assert cgs.validate(doc, items, snap['items_sha256'], snap['rubric'], 2) == []
    v1 = dict(doc, rubric_version=1, rubric=cg.RUBRIC_V1)
    assert any('rubric_version 1' in p for p in
               cgs.validate(v1, items, snap['items_sha256'], snap['rubric'], 2))
    assert cgs.validate(v1, items, snap['items_sha256'], cg.RUBRIC_V1, 1) == []
    assert any('rubric_version 3' in p for p in
               cgs.validate(dict(doc, rubric_version=3), items, snap['items_sha256'], None, 3))
    # an explicit --rubric-version 1 builds a v1 bundle
    build(setup.argv + ['--rubric-version', '1', '--rebuild'])
    snap = json.loads((setup.bundle / 'snapshot.json').read_text(encoding='utf-8'))
    assert (snap['rubric_version'], snap['rubric']) == (1, cg.RUBRIC_V1)


def test_tag_set_kept_per_bundle_and_counted_by_stratum(setup):
    """#184: a new sample bundle records tag set v1 (report.md says so); score --bundle adds a
    descriptive tags block, blind and final, and leaves every precision field alone; an export
    without the bundle's tag set, or with a tag outside it, cannot be scored (review M3); a
    rebuild never adds tags to a bundle built without them, and a bundle holding verdicts but
    no snapshot.json is refused (review M2)."""
    build(setup.argv)
    snap_path = setup.bundle / 'snapshot.json'
    snap = json.loads(snap_path.read_text(encoding='utf-8'))
    assert (snap['tag_set_version'], snap['tag_set']) == (1, cg.TAG_SET_V1)
    assert '- Tags: set v1 (unsafe_crossing, ' in (setup.bundle / 'report.md').read_text(encoding='utf-8')
    vpath, doc = export(setup.bundle, {'vancouver:res:n1': {0: ('absent', None)},
                                       'vancouver:res:n2': {0: ('present', None),
                                                            1: ('absent', None)}})
    ids = [t['id'] for t in cg.TAG_SET_V1]
    plain = score(setup, vpath, '--bundle', str(setup.bundle))
    assert plain['tags']['corners'] == {'all': 3, 'residential': 3}
    assert plain['tags']['blind']['all'] == dict.fromkeys(ids, 0)
    u = doc['units']['vancouver:res:n2']
    u['blind']['1']['tags'] = ['nothing_to_cross_to', 'unsafe_crossing']
    u['corners']['1']['tags'] = ['nothing_to_cross_to', 'unsafe_crossing']
    u['corners']['0']['tags'] = ['occluded_or_blurry']                 # after the reveal
    u['tags_after_inventory'] = True
    vpath.write_text(json.dumps(doc), encoding='utf-8')
    res = score(setup, vpath, '--bundle', str(setup.bundle))
    blind, final = res['tags']['blind']['residential'], res['tags']['final']['residential']
    assert blind == dict(dict.fromkeys(ids, 0), nothing_to_cross_to=1, unsafe_crossing=1)
    assert final == dict(blind, occluded_or_blurry=1)

    def precision_only(r):
        return {k: v for k, v in r.items() if k not in ('tags', 'verdicts')}
    assert precision_only(res) == precision_only(plain)
    # M3: an export from a page rendered without the tag set is refused, not read as no tags
    vpath.write_text(json.dumps({k: v for k, v in doc.items()
                                 if k not in ('tag_set_version', 'tag_set')}), encoding='utf-8')
    with pytest.raises(SystemExit, match='cannot be scored'):
        score(setup, vpath, '--bundle', str(setup.bundle))
    u['blind']['1']['tags'] = u['corners']['1']['tags'] = ['made_up']
    vpath.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(SystemExit, match='cannot be scored'):
        score(setup, vpath, '--bundle', str(setup.bundle))
    # M2: a bundle built before tags stays tag-less on a rebuild
    snap_path.write_text(json.dumps({k: v for k, v in snap.items()
                                     if k not in ('tag_set_version', 'tag_set')}), encoding='utf-8')
    build(setup.argv)
    assert 'tag_set_version' not in json.loads(snap_path.read_text(encoding='utf-8'))
    snap_path.unlink()
    with pytest.raises(SystemExit, match='no snapshot.json'):
        build(setup.argv)


def _cut(tmp_path, worker_src, items):
    import subprocess
    from PIL import Image
    store = tmp_path / 'store'
    if not store.exists():
        (store / 'ab').mkdir(parents=True)
        im = Image.new('RGB', (1024, 512), (40, 90, 140))
        for x in range(0, 1024, 8):
            for y in range(0, 512, 64):
                im.putpixel((x, y), (255, 255, 255))
        im.save(store / 'ab' / 'abP.jpg', quality=95)
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'items': items, 'workers': 1, 'panos_root': str(store),
                               'draft_width': 4096}), encoding='utf-8')
    w = tmp_path / 'crop.py'
    w.write_text(worker_src, encoding='utf-8')
    out = tmp_path / f'out{len(list(tmp_path.glob("out*")))}'
    proc = subprocess.run([sys.executable, str(w), str(job), str(out)], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return out, store


def test_crop_worker_landscape_and_square_back_compat(tmp_path):
    """The worker's optional fov_h_deg / px_w / px_h: a landscape tile, crosshair centred,
    seam-wrapped; an item without them is the square crop, and a square item spelled with the
    new keys gives the same bytes (the worker is shared with export_cluster_review and
    site_explorer)."""
    pytest.importorskip('PIL')
    from PIL import Image
    import site_explorer as se
    sq = {'pano_id': 'abP', 'x': 0.5, 'y': 0.6, 'fov_deg': 60, 'px': 128, 'name': 'sq.jpg',
          'path': 'ab/abP.jpg'}
    sq_new = dict(sq, name='sq2.jpg', fov_h_deg=60, px_w=128, px_h=128)
    wide = dict(sq, name='w.jpg', fov_h_deg=120, px_w=256, px_h=128)
    seam = dict(wide, name='seam.jpg', x=0.98)
    out, store = _cut(tmp_path, se.CROP_WORKER, [sq, sq_new, wide, seam])
    assert (out / 'sq.jpg').read_bytes() == (out / 'sq2.jpg').read_bytes()
    with Image.open(out / 'sq.jpg') as im:
        assert im.size == (128, 128)
    for n in ('w.jpg', 'seam.jpg'):
        with Image.open(out / n) as im:
            assert im.size == (256, 128)
            rgb = im.convert('RGB')
            # the orange crosshair arm right of the ring sits on the corner point's row:
            # tile 341 x 170 px of the 1024-px equirect, corner at y = 0.6 * 512 = 307.2,
            # y0 = round(307.2 - 85) = 222, so (307.2 - 222) * 128 / 170 = 64.1 in the output
            col = [rgb.getpixel((140, y)) for y in range(128)]
            row = max(range(128), key=lambda y: col[y][0] - col[y][2])
            assert abs(row - 64) <= 2 and col[row][0] > 150, (n, row, col[row])
    local = tmp_path / 'local'
    local.mkdir()
    assert se.fetch_crops_local({'items': [sq, wide], 'draft_width': 4096}, local, store) == set()
    with Image.open(local / 'w.jpg') as im:
        assert im.size == (256, 128)
