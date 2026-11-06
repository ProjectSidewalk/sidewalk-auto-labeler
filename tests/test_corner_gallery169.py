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
