"""The multi-type record contract (issue #168): the detector registry, the reader helpers
every consumer goes through, and the writer rule that keeps a CurbRamp-only run
byte-identical to the pre-#168 format. Torch-free."""
import pytest

from detectors import (BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR,
                       OPERATIONAL_CONFIDENCE)
from detectors import registry as reg

PROV_A = {'model_id': 'rampnet-model@aaaaaaaaaaaa', 'model_training_date': '08-21-2025',
          'api_version': '1.0.0', 'model_repo': 'projectsidewalk/rampnet-model',
          'model_revision': 'a' * 40}
PROV_B = {'model_id': 'crosswalk@bbbbbbbbbbbb', 'model_training_date': '10-01-2026',
          'api_version': '1.0.0', 'model_repo': 'example/crosswalk',
          'model_revision': 'b' * 40}


def _det(x, y, c, t=None):
    d = {'x_normalized': x, 'y_normalized': y, 'confidence': c}
    if t is not None:
        d['label_type'] = t
    return d


def _legacy_line():
    return {'detections': [_det(.1, .6, .9), _det(.2, .7, .2)], 'label_type': 'CurbRamp',
            **PROV_A, 'pano': {'panorama_id': 'p1'}}


def _two_type_line():
    return {'detections': [_det(.1, .6, .9, 'CurbRamp'), _det(.3, .6, .8, 'Crosswalk')],
            'label_type': None, 'label_types': ['Crosswalk', 'CurbRamp'],
            'models': {'CurbRamp': PROV_A, 'Crosswalk': PROV_B}, **PROV_A,
            'pano': {'panorama_id': 'p1'}}


# --- registry ------------------------------------------------------------------------

def test_curb_ramp_spec_references_the_module_constants():
    s = reg.spec_for('CurbRamp')
    assert (s.storage_floor, s.operational_confidence, s.benchmark_confidence, s.mask_rig) \
        == (DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE, BENCHMARK_CONFIDENCE, True)
    assert (s.module, s.cls) == ('detectors.curb_ramp', 'CurbRampDetector')
    assert list(reg.SPECS) == ['CurbRamp']


def test_spec_for_unknown_lists_known_types():
    with pytest.raises(reg.UnknownLabelTypeError, match='known: CurbRamp'):
        reg.spec_for('Crosswalk')


def test_parse_types():
    assert reg.parse_types('CurbRamp') == ['CurbRamp']
    assert reg.parse_types(' CurbRamp,CurbRamp ') == ['CurbRamp']
    for bad in ('Crosswalk', 'CurbRamp,', ''):
        with pytest.raises(reg.UnknownLabelTypeError):
            reg.parse_types(bad)


def test_load_detectors_imports_lazily(monkeypatch):
    import sys
    import types
    built = []

    class Fake:
        def __init__(self, **kw):
            built.append(kw)

    mod = types.ModuleType('fake_det_mod')
    mod.FakeDetector = Fake
    monkeypatch.setitem(sys.modules, 'fake_det_mod', mod)
    monkeypatch.setitem(reg.SPECS, 'Fake', reg.SPECS['CurbRamp']._replace(
        label_type='Fake', module='fake_det_mod', cls='FakeDetector'))
    dets = reg.load_detectors(['Fake'], batch_size=2)
    assert list(dets) == ['Fake'] and isinstance(dets['Fake'], Fake)
    assert built == [{'batch_size': 2}]


# --- reader helpers ------------------------------------------------------------------

def test_helpers_on_a_legacy_line():
    rec = _legacy_line()
    assert reg.record_types(rec) == ['CurbRamp']
    assert [reg.detection_type(rec, d) for d in rec['detections']] == ['CurbRamp'] * 2
    assert reg.provenance_for(rec, 'CurbRamp') == PROV_A
    assert reg.detections_of_type(rec, 'CurbRamp') == rec['detections']
    assert reg.detections_of_type(rec, 'Crosswalk') == []


def test_helpers_on_a_line_with_no_type_keys_at_all():
    rec = {'detections': [_det(.1, .6, .9)], 'model_id': 'm'}
    assert reg.record_types(rec) == ['CurbRamp']
    assert reg.detection_type(rec, rec['detections'][0]) == 'CurbRamp'
    assert reg.provenance_for(rec, 'CurbRamp')['model_id'] == 'm'


def test_helpers_on_a_two_type_line():
    rec = _two_type_line()
    assert reg.record_types(rec) == ['Crosswalk', 'CurbRamp']
    assert [reg.detection_type(rec, d) for d in rec['detections']] == ['CurbRamp', 'Crosswalk']
    assert reg.provenance_for(rec, 'Crosswalk') == PROV_B
    assert reg.provenance_for(rec, 'CurbRamp') == PROV_A
    assert [d['confidence'] for d in reg.detections_of_type(rec, 'Crosswalk')] == [.8]
    with pytest.raises(KeyError):
        reg.provenance_for(rec, 'NoCurbRamp')


def test_untyped_detection_on_a_multi_type_line_raises():
    rec = _two_type_line()
    rec['detections'].append(_det(.5, .6, .5))
    with pytest.raises(reg.UntypedDetectionError):
        reg.detection_type(rec, rec['detections'][-1])
    with pytest.raises(reg.UntypedDetectionError):
        reg.detections_of_type(rec, 'CurbRamp')


def test_empty_multi_type_line_still_says_what_ran():
    rec = _two_type_line()
    rec['detections'] = []
    assert reg.record_types(rec) == ['Crosswalk', 'CurbRamp']


def test_require_curb_ramp_only():
    reg.require_curb_ramp_only(_legacy_line(), 'x')
    with pytest.raises(ValueError, match='CurbRamp-only'):
        reg.require_curb_ramp_only(_two_type_line(), 'x')


# --- the writer rule (main.build_output_line / main._process) ------------------------

import hashlib  # noqa: E402
import json  # noqa: E402

from conftest import make_process_result, make_provenance  # noqa: E402
from test_border import PRE_130_LINE_SHA256  # noqa: E402


class _FakeDetector:
    """The CurbRampDetector surface load_detectors promises, without torch."""
    decode, border = 'argmax', 'exclude'

    def __init__(self, provenance, dets):
        self.provenance, self._dets, self.closed = provenance, dets, False

    def detect(self, image):
        return list(self._dets)

    def stats(self):
        return {'images': 0}

    def close(self):
        self.closed = True


@pytest.fixture
def crosswalk_spec(monkeypatch):
    """A second registered type for the duration of one test (none exists yet, #168)."""
    monkeypatch.setitem(reg.SPECS, 'Crosswalk', reg.DetectorSpec(
        'Crosswalk', 'nowhere', 'Nothing', 0.2, 0.5, 0.6, False))


def _run_process(monkeypatch, dets):
    import main
    monkeypatch.setattr(main, 'detectors', dets)
    monkeypatch.setattr(main, 'curb_ramp_detector', dets.get('CurbRamp'))
    fake = make_process_result()
    return main._process('PID', lambda: {'status': 'success', 'pano': fake['pano'],
                                         'image': object()})


def test_curb_ramp_only_line_keeps_the_byte_pin_through_the_registry_path(monkeypatch):
    """The issue's done-when: a {CurbRamp} run is byte-identical to the pre-#168 line,
    whether build_output_line gets one provenance dict or a {type: provenance} map, and
    when the detection comes out of _process through main.detectors."""
    import main
    prov = make_provenance()
    fake = make_process_result()
    for p in (prov, {'CurbRamp': prov}):
        line = json.dumps(main.build_output_line(fake, p))
        assert hashlib.sha256(line.encode()).hexdigest() == PRE_130_LINE_SHA256
    result = _run_process(monkeypatch, {'CurbRamp': _FakeDetector(prov, fake['detections'])})
    assert result['status'] == 'success', result
    assert 'label_types' not in result and 'type_markers' not in result
    line = json.dumps(main.build_output_line(result, {'CurbRamp': prov}))
    assert hashlib.sha256(line.encode()).hexdigest() == PRE_130_LINE_SHA256


def test_two_type_line_shape(monkeypatch, crosswalk_spec):
    import main
    prov_c = make_provenance()
    result = _run_process(monkeypatch, {
        'CurbRamp': _FakeDetector(prov_c, [(0.5, 0.25, 0.9)]),
        'Crosswalk': _FakeDetector(PROV_B, [(0.1, 0.7, 0.8), (0.2, 0.7, 0.4)])})
    line = main.build_output_line(result, {'CurbRamp': prov_c, 'Crosswalk': PROV_B})
    assert line['label_types'] == ['Crosswalk', 'CurbRamp']
    assert line['label_type'] is None
    assert [(d['label_type'], d['confidence']) for d in line['detections']] == \
        [('Crosswalk', 0.8), ('Crosswalk', 0.4), ('CurbRamp', 0.9)]
    # line-level model_* are CurbRamp's whenever CurbRamp ran
    assert {k: line[k] for k in reg.PROVENANCE_KEYS} == \
        {k: prov_c[k] for k in reg.PROVENANCE_KEYS}
    assert line['models']['CurbRamp'] == {k: prov_c[k] for k in reg.PROVENANCE_KEYS}
    assert line['models']['Crosswalk'] == {**PROV_B, 'detection_decode': 'argmax',
                                           'detection_border': 'exclude'}
    assert 'detection_decode' not in line and 'detection_border' not in line
    # the helpers read it back
    assert reg.record_types(line) == ['Crosswalk', 'CurbRamp']
    assert reg.provenance_for(line, 'Crosswalk') == PROV_B
    assert len(reg.detections_of_type(line, 'CurbRamp')) == 1
    json.dumps(line)  # serializable


def test_empty_two_type_line_still_lists_what_ran(monkeypatch, crosswalk_spec):
    import main
    result = _run_process(monkeypatch, {'CurbRamp': _FakeDetector(make_provenance(), []),
                                        'Crosswalk': _FakeDetector(PROV_B, [])})
    line = main.build_output_line(result, {'CurbRamp': make_provenance(), 'Crosswalk': PROV_B})
    assert line['detections'] == [] and line['label_types'] == ['Crosswalk', 'CurbRamp']


def test_single_non_curb_ramp_type_line(monkeypatch, crosswalk_spec):
    import main
    result = _run_process(monkeypatch, {'Crosswalk': _FakeDetector(PROV_B, [(0.1, 0.7, 0.8)])})
    line = main.build_output_line(result, {'Crosswalk': PROV_B})
    assert line['label_type'] == 'Crosswalk' and line['label_types'] == ['Crosswalk']
    assert line['model_id'] == PROV_B['model_id']
    assert line['detections'][0]['label_type'] == 'Crosswalk'


def test_multitype_writer_refuses_an_untagged_detection(crosswalk_spec):
    import main
    fake = make_process_result()   # 3-tuple detections, no type tag
    with pytest.raises(ValueError, match='not tagged'):
        main.build_output_line(fake, {'CurbRamp': make_provenance(), 'Crosswalk': PROV_B})


# --- the manifest binding --------------------------------------------------------------

GEOM = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


def _init(run_dir, provenance):
    import main
    return main.load_or_init_run_dir(run_dir, 'city.geojson', GEOM, 'hash', 'gsv',
                                     provenance=provenance)


def test_manifest_binds_label_types_new_match_mismatch(tmp_path, crosswalk_spec):
    prov = make_provenance()
    run = tmp_path / 'city'
    m = _init(run, prov)
    assert m['label_types'] == ['CurbRamp'] and 'models' not in m
    _init(run, {'CurbRamp': prov})                     # same set resumes
    with pytest.raises(SystemExit, match='new --name for this type set'):
        _init(run, {'CurbRamp': prov, 'Crosswalk': PROV_B})


def test_legacy_manifest_with_records_binds_curb_ramp_without_a_rewrite(tmp_path, crosswalk_spec):
    prov = make_provenance()
    run = tmp_path / 'city'
    m = _init(run, prov)
    del m['label_types']
    (run / 'manifest.json').write_text(json.dumps(m), encoding='utf-8')
    (run / 'results.jsonl').write_text('{}\n', encoding='utf-8')
    before = (run / 'manifest.json').read_bytes()
    _init(run, prov)                                   # pre-#168 run resumes as CurbRamp
    assert (run / 'manifest.json').read_bytes() == before
    with pytest.raises(SystemExit, match='label types CurbRamp'):
        _init(run, {'CurbRamp': prov, 'Crosswalk': PROV_B})


def test_scan_only_binds_no_types(tmp_path):
    m = _init(tmp_path / 'city', None)
    assert 'label_types' not in m


def test_multi_type_manifest_binds_every_model(tmp_path, crosswalk_spec):
    import main
    prov = make_provenance()
    run = tmp_path / 'city'
    m = _init(run, {'CurbRamp': prov, 'Crosswalk': PROV_B})
    assert m['label_types'] == ['Crosswalk', 'CurbRamp']
    assert m['model_revision'] == prov['model_revision']           # top-level = CurbRamp
    assert m['models']['Crosswalk']['model_revision'] == PROV_B['model_revision']
    main.record_run(run / 'manifest.json', m, 't0', 1, 1, 0, 0,
                    provenance={'CurbRamp': prov, 'Crosswalk': PROV_B})
    entry = json.loads((run / 'manifest.json').read_text())['runs'][0]
    assert set(entry['models']) == {'Crosswalk', 'CurbRamp'}
    other = {**PROV_B, 'model_revision': 'c' * 40}
    with pytest.raises(SystemExit, match='city/Crosswalk'):
        _init(run, {'CurbRamp': prov, 'Crosswalk': other})


# --- --types -------------------------------------------------------------------------

def test_types_flag_parses_and_rejects_unknown(monkeypatch, capsys):
    import argparse
    import sys
    import main
    assert main.parse_types_arg('CurbRamp') == ['CurbRamp']
    with pytest.raises(argparse.ArgumentTypeError, match='known: CurbRamp'):
        main.parse_types_arg('Crosswalk')
    monkeypatch.setattr(sys, 'argv', ['main.py', 'x.geojson', '--types', 'Crosswalk'])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 2 and 'known: CurbRamp' in capsys.readouterr().err


# --- scripts/detect_from_store.bind_manifest -----------------------------------------

def test_detect_from_store_binds_curb_ramp_and_refuses_other_types(tmp_path, crosswalk_spec):
    import detect_from_store as dfs
    prov = make_provenance()
    pixels = {'store': '/s', 'layout': 'sharded', 'server': 'https://a.example'}
    run = tmp_path / 'run'
    run.mkdir()
    m = dfs.bind_manifest(run, prov, pixels)
    assert m['label_types'] == ['CurbRamp']
    dfs.bind_manifest(run, {'CurbRamp': prov}, pixels)        # map form resumes
    with pytest.raises(SystemExit, match='CurbRamp only'):
        dfs.bind_manifest(run, {'CurbRamp': prov, 'Crosswalk': PROV_B}, pixels)
    m['label_types'] = ['Crosswalk']
    (run / 'manifest.json').write_text(json.dumps(m), encoding='utf-8')
    with pytest.raises(SystemExit, match='label types Crosswalk'):
        dfs.bind_manifest(run, prov, pixels)


# --- consumers --------------------------------------------------------------------------

def _write(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    return path


def _real_legacy_line():
    import main
    fake = make_process_result(detections=[(0.5, 0.6, 0.9), (0.25, 0.7, 0.2)])
    return main.build_output_line(fake, make_provenance())


def _as_two_type(line):
    """The same pano as a {CurbRamp, Crosswalk} line: CurbRamp's detections first (so their
    record indices match), then a Crosswalk one."""
    return {**line, 'detections': [{**d, 'label_type': 'CurbRamp'} for d in line['detections']]
            + [_det(0.75, 0.6, 0.95, 'Crosswalk')], 'label_type': None,
            'label_types': ['Crosswalk', 'CurbRamp'],
            'models': {'CurbRamp': {k: line[k] for k in reg.PROVENANCE_KEYS},
                       'Crosswalk': PROV_B}}


def test_fuse_sites_load_results_equivalence(tmp_path):
    """A legacy line, and the same pano's CurbRamp detections inside a two-type line, give
    identical fusion input; the Crosswalk detection never reaches a CurbRamp fuse."""
    import fuse_sites as fs
    legacy = _real_legacy_line()
    a, _ = fs.load_results(_write(tmp_path / 'a.jsonl', [legacy]), read_heights=False)
    b, _ = fs.load_results(_write(tmp_path / 'b.jsonl', [_as_two_type(legacy)]),
                           read_heights=False)
    assert a == b
    assert a[0].detections == [(0, 0.5, 0.6, 0.9), (1, 0.25, 0.7, 0.2)]
    c, _ = fs.load_results(tmp_path / 'b.jsonl', read_heights=False, label_type='Crosswalk')
    assert c[0].detections == [(2, 0.75, 0.6, 0.95)]
    with pytest.raises(ValueError, match='nothing to fuse'):
        fs.load_results(tmp_path / 'a.jsonl', read_heights=False, label_type='Crosswalk')


def test_export_benchmark_label_type(crosswalk_spec):
    import export_benchmark as eb
    legacy = _real_legacy_line()
    two = _as_two_type(legacy)
    assert eb._of_type(legacy, 'CurbRamp') is legacy               # unchanged, byte for byte
    # CurbRamp's line-level border marker must not survive into a Crosswalk record, and the
    # Crosswalk decode must surface at line level
    two['detection_border'] = 'keep'
    two['models']['Crosswalk'] = {**PROV_B, 'detection_decode': 'gaussian',
                                  'detection_border': 'exclude'}
    cw = eb._of_type(two, 'Crosswalk')
    assert cw['label_type'] == 'Crosswalk' and cw['label_types'] == ['Crosswalk']
    assert [d['confidence'] for d in cw['detections']] == [0.95]
    assert list(cw['models']) == ['Crosswalk']
    assert {k: cw[k] for k in reg.PROVENANCE_KEYS} == PROV_B
    assert cw['detection_decode'] == 'gaussian' and 'detection_border' not in cw
    assert reg.record_types(cw) == ['Crosswalk']
    # ...and a CurbRamp record cut from the same line keeps CurbRamp's markers
    cr = eb._of_type(two, 'CurbRamp')
    assert cr['detection_border'] == 'keep' and 'detection_decode' not in cr
    assert cr['model_id'] == legacy['model_id'] and list(cr['models']) == ['CurbRamp']
    with pytest.raises(ValueError, match='not Crosswalk'):
        eb._of_type(legacy, 'Crosswalk')


def test_position_check_reads_each_detection_at_its_types_spec(crosswalk_spec):
    import geo
    import position_check as pc
    legacy = _real_legacy_line()
    two = _as_two_type(legacy)
    two['detections'].append(_det(0.8, 0.6, 0.55, 'Crosswalk'))   # above Crosswalk's 0.5
    two['detections'].append(_det(0.9, 0.6, 0.45, 'Crosswalk'))   # below it
    frame = geo.LocalFrame(legacy['pano']['lat'], legacy['pano']['lng'])
    ids = [p['id'].split('#')[1] for p in pc.operational_points([two], frame)]
    # CurbRamp 0.9 (idx 0) passes 0.30 and 0.2 (1) does not; Crosswalk 0.95 (2) and 0.55
    # (3) pass its 0.5, 0.45 (4) does not
    assert ids == ['0', '2', '3']


def test_curb_ramp_only_refusals(tmp_path):
    legacy = _write(tmp_path / 'a.jsonl', [_real_legacy_line()])
    reg.require_curb_ramp_file(legacy, 'x')
    two = _write(tmp_path / 'b.jsonl', [_as_two_type(_real_legacy_line())])
    with pytest.raises(ValueError, match='line 1'):
        reg.require_curb_ramp_file(two, 'x')


# --- review fixes (PR #170) -----------------------------------------------------------

def test_legacy_writer_refuses_a_multi_type_result(crosswalk_spec):
    """A tagged result handed only CurbRamp's provenance must not be written as a CurbRamp
    line (it would relabel another type's detections)."""
    import main
    fake = make_process_result(detections=[(0.5, 0.25, 0.9, 'Crosswalk')])
    with pytest.raises(ValueError, match='multi-type result'):
        main.build_output_line(fake, make_provenance())
    with pytest.raises(ValueError, match='multi-type result'):
        main.build_output_line({**make_process_result(), 'label_types': ['Crosswalk', 'CurbRamp']},
                               make_provenance())


def test_provenance_map_refuses_a_misspelled_type():
    import main
    with pytest.raises(reg.UnknownLabelTypeError):
        main.provenance_map({'Curbramp': make_provenance()})


def test_one_detector_raising_fails_the_whole_pano(monkeypatch, crosswalk_spec, tmp_path):
    import main

    class Boom(_FakeDetector):
        def detect(self, image):
            raise RuntimeError('CUDA out of memory')

    result = _run_process(monkeypatch, {'CurbRamp': _FakeDetector(make_provenance(), [(.5, .5, .9)]),
                                        'Crosswalk': Boom(PROV_B, [])})
    assert result['status'] == 'failure' and 'out of memory' in result['reason']
    cache, jsonl = tmp_path / 'c.txt', tmp_path / 'r.jsonl'
    with open(cache, 'w') as fc, open(jsonl, 'w') as fj:
        assert main.handle_result(result, fc, fj, {'CurbRamp': make_provenance(),
                                                   'Crosswalk': PROV_B}) == 'failed'
    assert cache.read_text() == '' and jsonl.read_text() == ''   # retryable, nothing written


def test_load_detectors_shares_one_lock_across_types(monkeypatch):
    import sys
    import types
    built = []

    class Fake:
        def __init__(self, **kw):
            built.append(kw)

    mod = types.ModuleType('fake_det_mod2')
    mod.FakeDetector = Fake
    monkeypatch.setitem(sys.modules, 'fake_det_mod2', mod)
    for t in ('FakeA', 'FakeB'):
        monkeypatch.setitem(reg.SPECS, t, reg.SPECS['CurbRamp']._replace(
            label_type=t, module='fake_det_mod2', cls='FakeDetector'))
    reg.load_detectors(['FakeA', 'FakeB'], batch_size=1)
    assert built[0]['inference_lock'] is built[1]['inference_lock'] is not None
    built.clear()
    reg.load_detectors(['FakeA'], batch_size=1)          # one type: constructed as before
    assert built == [{'batch_size': 1}]


def test_fuse_sites_reads_the_types_own_markers(tmp_path):
    """A Crosswalk fuse takes its decode/border from models['Crosswalk'], never from the
    line-level markers, which describe CurbRamp."""
    import fuse_sites as fs
    two = _as_two_type(_real_legacy_line())
    two['detection_border'] = 'keep'                                  # CurbRamp's
    two['models']['Crosswalk'] = {**PROV_B, 'detection_decode': 'gaussian',
                                  'detection_border': 'exclude'}
    path = _write(tmp_path / 'b.jsonl', [two])
    cw, _ = fs.load_results(path, read_heights=False, label_type='Crosswalk')
    cr, _ = fs.load_results(path, read_heights=False)
    assert (cw[0].decode, cw[0].border) == ('gaussian', 'exclude')
    assert (cr[0].decode, cr[0].border) == ('argmax', 'keep')


def test_fuse_sites_require_types(tmp_path):
    import fuse_sites as fs
    legacy = _write(tmp_path / 'a.jsonl', [_real_legacy_line()])
    two = _write(tmp_path / 'b.jsonl', [_as_two_type(_real_legacy_line())])
    fs.load_results(legacy, read_heights=False, require_types=('CurbRamp',))
    with pytest.raises(ValueError, match='scores'):
        fs.load_results(two, read_heights=False, require_types=('CurbRamp',))


def test_consumer_refusals_on_a_multi_type_file(tmp_path):
    """agree_rate, provenance_gate and eval_sites (mined_precision's loader) refuse a
    multi-type file and stay silent on a legacy one."""
    import agree_rate
    import eval_sites
    import provenance_gate
    legacy = _write(tmp_path / 'a.jsonl', [_real_legacy_line()])
    two = _write(tmp_path / 'b.jsonl', [_as_two_type(_real_legacy_line())])
    bench = tmp_path / 'bench' / 'city'
    bench.mkdir(parents=True)
    (bench / 'verdicts.json').write_text(json.dumps({'panos': {}}), encoding='utf-8')
    (bench / 'records.jsonl').write_text('', encoding='utf-8')
    for fn in (agree_rate.scan_run, provenance_gate.load_run,
               lambda p: eval_sites.load_city_at_height('city', tmp_path / 'bench', tmp_path,
                                                        2.6, results_path=p)):
        fn(legacy)
        with pytest.raises(ValueError):
            fn(two)


def test_gap_fill_only_binds_label_types(tmp_path, monkeypatch):
    import types
    import main
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'a.geojson').write_text(json.dumps(GEOM), encoding='utf-8')
    source = types.SimpleNamespace(NAME='gsv', fetch_pano_by_id=lambda *a: None)
    main.run_labeler('a.geojson', 'gf', source, gap_fill_only=True, check_positions=False,
                     provenance={'CurbRamp': make_provenance()})
    m = json.loads((tmp_path / 'runs' / 'gf' / 'manifest.json').read_text(encoding='utf-8'))
    assert m['label_types'] == ['CurbRamp']


def test_bind_label_types_writes_the_key_on_a_legacy_manifest_without_records(tmp_path):
    import main
    run = tmp_path / 'r'
    run.mkdir()
    manifest = {}
    assert main.bind_label_types(manifest, ['CurbRamp'], run) is True
    assert manifest == {'label_types': ['CurbRamp']}
