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
