"""The Crosswalk detector spec (RampNet#274 crosswalk_s4 in the labeler): its thresholds, the
decode floor, checkpoint provenance, the vendored network, and how Crosswalk runs describe
themselves in records, manifests and submission records. Torch-free: the detector itself is
never constructed here (scripts/crosswalk_parity.py is its GPU acceptance test)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

import detectors
from detectors import checkpoints as ck
from detectors import decode as dec
from detectors import registry as reg

REPO = Path(__file__).resolve().parents[1]
S4_SHA = 'ebad379b5f7c1645dc39ccb820cd18500109723ae5b1e5706c1e4521b2ab42b4'


# --- the spec --------------------------------------------------------------------------

def test_crosswalk_spec_values():
    s = reg.spec_for('Crosswalk')
    assert (s.module, s.cls) == ('detectors.crosswalk', 'CrosswalkDetector')
    assert (s.storage_floor, s.operational_confidence, s.benchmark_confidence) == (0.05, 0.23, 0.23)
    assert s.mask_rig is True and s.border == 'keep'
    assert reg.type_frame('Crosswalk') == {'detection_decode': 'argmax',
                                           'detection_border': 'keep',
                                           'detection_storage_floor': 0.05}


def test_border_field_defaults_for_seven_argument_specs():
    """Every 7-argument DetectorSpec (CurbRamp's, the tests' stand-ins) keeps working."""
    assert reg.spec_for('CurbRamp').border == 'exclude'
    assert reg.DetectorSpec('X', 'm', 'C', 0.1, 0.3, 0.5, False).border == 'exclude'
    with pytest.raises(ValueError, match='per run'):
        reg.type_frame('CurbRamp')


# --- decode floor -----------------------------------------------------------------------

def test_decode_floor_keyword():
    """0.05 finds a 0.07 peak; the default (CurbRamp's 0.1) does not."""
    h = np.zeros((512, 1024), dtype=np.float32)
    h[200, 300] = 0.07
    h[300, 700] = 0.5
    assert [round(c, 2) for _, _, c in dec.detections_from_heatmap(h)] == [0.5]
    assert [round(c, 2) for _, _, c in dec.detections_from_heatmap(h, floor=0.05)] == [0.5, 0.07]
    assert len(dec.detections_from_heatmap(h, border='wrap', floor=0.05)) == 2
    assert len(dec.detections_from_heatmap(h, border='wrap')) == 1


# --- checkpoint provenance --------------------------------------------------------------

def test_known_checkpoint_provenance():
    p = ck.provenance_for_checkpoint(S4_SHA)
    assert p == {'model_id': 'rampnet-crosswalk@ebad379b5f7c',
                 'model_training_date': '11-09-2026',
                 'api_version': detectors.API_VERSION,
                 'model_repo': 'unpublished:RampNet#274/crosswalk_s4',
                 'model_revision': f'sha256:{S4_SHA}'}
    assert set(p) == set(reg.PROVENANCE_KEYS)


def test_unknown_checkpoint_refuses_unless_allowed():
    other = 'f' * 64
    with pytest.raises(detectors.ModelProvenanceError, match='KNOWN_CHECKPOINTS'):
        ck.provenance_for_checkpoint(other)
    p = ck.provenance_for_checkpoint(other, allow_unknown=True)
    assert p['model_training_date'] is None
    assert p['model_repo'] == ck.UNKNOWN_CHECKPOINT_REPO
    assert p['model_id'] == 'rampnet-crosswalk@ffffffffffff'
    with pytest.raises(detectors.ModelProvenanceError, match='not a sha256'):
        ck.provenance_for_checkpoint('abc', allow_unknown=True)


def test_checkpoint_of_another_type_refuses(monkeypatch):
    monkeypatch.setitem(ck.KNOWN_CHECKPOINTS, 'a' * 64,
                        {**ck.KNOWN_CHECKPOINTS[S4_SHA], 'label_type': 'NoCurbRamp'})
    with pytest.raises(detectors.ModelProvenanceError, match='not a Crosswalk one'):
        ck.provenance_for_checkpoint('a' * 64, allow_unknown=True)


def test_sha256_hex():
    assert ck.sha256_hex(b'') == hashlib.sha256(b'').hexdigest()


def test_vendored_keypoint_model_is_rampnets_file_verbatim():
    """detectors/rampnet_keypoint.py must stay byte-identical to RampNet's rampnet/model.py at
    RAMPNET_KEYPOINT_COMMIT. To update: copy the file verbatim and change both constants."""
    data = (REPO / 'detectors' / 'rampnet_keypoint.py').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(data).hexdigest() == ck.RAMPNET_KEYPOINT_SHA256


# --- load_detectors: per-type kwargs ------------------------------------------------------

def test_load_detectors_passes_type_kwargs_to_that_type_only(monkeypatch):
    import sys
    import types
    built = {}

    def fake(name):
        class Fake:
            def __init__(self, **kw):
                built[name] = kw
        return Fake

    mod = types.ModuleType('fake_det_mod_xw')
    mod.A, mod.B = fake('A'), fake('B')
    monkeypatch.setitem(sys.modules, 'fake_det_mod_xw', mod)
    for t in ('A', 'B'):
        monkeypatch.setitem(reg.SPECS, t, reg.SPECS['CurbRamp']._replace(
            label_type=t, module='fake_det_mod_xw', cls=t))
    reg.load_detectors(['B'], type_kwargs={'B': {'checkpoint': 'x.pth'}}, batch_size=1)
    assert built == {'B': {'batch_size': 1, 'checkpoint': 'x.pth'}}
    built.clear()
    reg.load_detectors(['A', 'B'], type_kwargs={'A': {'decode': 'gaussian'}}, batch_size=1)
    assert built['A']['decode'] == 'gaussian' and 'decode' not in built['B']
    assert built['A']['inference_lock'] is built['B']['inference_lock']
    with pytest.raises(ValueError, match='not among the requested types'):
        reg.load_detectors(['A'], type_kwargs={'B': {}})


def test_crosswalk_module_is_not_imported_by_the_registry():
    import subprocess
    import sys
    code = ("import sys; from detectors import registry; registry.spec_for('Crosswalk'); "
            "bad = [m for m in ('torch', 'detectors.crosswalk', 'detectors.rampnet_keypoint') "
            "if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True)


# --- records written by main.py ------------------------------------------------------------

import json  # noqa: E402

from conftest import make_process_result, make_provenance  # noqa: E402

XW_PROV = ck.provenance_for_checkpoint(S4_SHA)


class _FakeXwalk:
    """CrosswalkDetector's surface (frame fixed by the spec), without torch."""
    decode, border, storage_floor = 'argmax', 'keep', 0.05

    def __init__(self, dets):
        self.provenance, self._dets = XW_PROV, dets

    def detect(self, image):
        return list(self._dets)


class _FakeCurb(_FakeXwalk):
    decode, border, storage_floor = 'argmax', 'exclude', 0.1

    def __init__(self, dets):
        super().__init__(dets)
        self.provenance = make_provenance()


def _process(monkeypatch, dets):
    import main
    monkeypatch.setattr(main, 'detectors', dets)
    monkeypatch.setattr(main, 'curb_ramp_detector', dets.get('CurbRamp'))
    fake = make_process_result()
    return main._process('PID', lambda: {'status': 'success', 'pano': fake['pano'],
                                         'image': object()})


def _crosswalk_line(monkeypatch, dets=((0.5, 0.6, 0.9), (0.25, 0.6, 0.07))):
    import main
    result = _process(monkeypatch, {'Crosswalk': _FakeXwalk(dets)})
    return json.loads(json.dumps(main.build_output_line(result, {'Crosswalk': XW_PROV})))


def test_crosswalk_only_line(monkeypatch):
    line = _crosswalk_line(monkeypatch)
    assert line['label_type'] == 'Crosswalk' and line['label_types'] == ['Crosswalk']
    assert line['models']['Crosswalk'] == {**XW_PROV, 'detection_decode': 'argmax',
                                           'detection_border': 'keep',
                                           'detection_storage_floor': 0.05}
    assert {k: line[k] for k in reg.PROVENANCE_KEYS} == XW_PROV
    # the line-level markers describe CurbRamp, which did not run
    assert 'detection_border' not in line and 'detection_decode' not in line
    assert reg.type_markers(line, 'Crosswalk') == ('argmax', 'keep')
    assert [d['label_type'] for d in line['detections']] == ['Crosswalk'] * 2


def test_curb_ramp_plus_crosswalk_line(monkeypatch):
    import main
    prov_c = make_provenance()
    result = _process(monkeypatch, {'CurbRamp': _FakeCurb([(0.5, 0.25, 0.9)]),
                                    'Crosswalk': _FakeXwalk([(0.1, 0.6, 0.3)])})
    line = main.build_output_line(result, {'CurbRamp': prov_c, 'Crosswalk': XW_PROV})
    assert line['label_types'] == ['Crosswalk', 'CurbRamp'] and line['label_type'] is None
    assert line['model_id'] == prov_c['model_id']              # line level = CurbRamp's
    assert line['models']['CurbRamp'] == {k: prov_c[k] for k in reg.PROVENANCE_KEYS}
    assert line['models']['Crosswalk']['detection_border'] == 'keep'
    assert 'detection_border' not in line                       # CurbRamp ran `exclude`
    assert reg.type_markers(line, 'CurbRamp') == ('argmax', 'exclude')
    assert reg.type_markers(line, 'Crosswalk') == ('argmax', 'keep')


# --- the manifest ----------------------------------------------------------------------------

GEOM = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


def _init(run_dir, provenance, decode='argmax', border='exclude'):
    import main
    return main.load_or_init_run_dir(run_dir, 'city.geojson', GEOM, 'hash', 'gsv',
                                     provenance=provenance, decode=decode, border=border)


def test_crosswalk_only_manifest_and_resume(tmp_path):
    run = tmp_path / 'city'
    m = _init(run, {'Crosswalk': XW_PROV})
    assert m['label_types'] == ['Crosswalk']
    # CurbRamp's top-level frame does not apply: null, never absent (absent = legacy 0.55)
    assert (m['detection_storage_floor'], m['detection_decode'], m['detection_border']) == \
        (None, None, None)
    assert m['models']['Crosswalk'] == {**XW_PROV, **reg.type_frame('Crosswalk')}
    assert m['model_revision'] == f'sha256:{S4_SHA}'
    (run / 'results.jsonl').write_text('{}\n', encoding='utf-8')
    before = (run / 'manifest.json').read_bytes()
    _init(run, {'Crosswalk': XW_PROV})                       # same model resumes, no rewrite
    assert (run / 'manifest.json').read_bytes() == before
    other = ck.provenance_for_checkpoint('e' * 64, allow_unknown=True)
    with pytest.raises(SystemExit, match='was produced by'):
        _init(run, {'Crosswalk': other})                     # a changed checkpoint is refused
    with pytest.raises(SystemExit, match='label types Crosswalk'):
        _init(run, {'CurbRamp': make_provenance()})          # and so is CurbRamp


def test_crosswalk_frame_change_is_refused_on_resume(tmp_path, monkeypatch):
    run = tmp_path / 'city'
    _init(run, {'Crosswalk': XW_PROV})
    monkeypatch.setitem(reg.SPECS, 'Crosswalk', reg.SPECS['Crosswalk']._replace(border='exclude'))
    with pytest.raises(SystemExit, match="detection_border 'keep'"):
        _init(run, {'Crosswalk': XW_PROV})


def test_scan_only_manifest_then_crosswalk_run(tmp_path):
    """--scan-only writes CurbRamp's floor; the first Crosswalk run on it nulls the top level."""
    run = tmp_path / 'city'
    m = _init(run, None, decode=None, border=None)
    assert m['detection_storage_floor'] == detectors.DETECTION_STORAGE_FLOOR
    m = _init(run, {'Crosswalk': XW_PROV})
    assert m['detection_storage_floor'] is None and m['label_types'] == ['Crosswalk']
    saved = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['models']['Crosswalk']['detection_border'] == 'keep'
    _init(run, None, decode=None, border=None)               # a later --scan-only still passes


def test_mixed_manifest_top_level_is_curb_ramps(tmp_path):
    m = _init(tmp_path / 'city', {'CurbRamp': make_provenance(), 'Crosswalk': XW_PROV},
              border='keep')
    assert (m['detection_storage_floor'], m['detection_border']) == (0.1, 'keep')
    assert 'detection_border' not in m['models']['CurbRamp']
    assert m['models']['Crosswalk']['detection_storage_floor'] == 0.05


def test_curb_ramp_only_manifest_is_unchanged(tmp_path):
    m = _init(tmp_path / 'city', make_provenance())
    assert 'models' not in m
    assert (m['detection_storage_floor'], m['detection_decode'], m['detection_border']) == \
        (0.1, 'argmax', 'exclude')


# --- main.py flags ----------------------------------------------------------------------------

@pytest.mark.parametrize('flags', [('--decode', 'gaussian'), ('--border', 'keep')])
def test_curb_ramp_flags_refused_without_curb_ramp(monkeypatch, capsys, tmp_path, flags):
    import sys
    import main
    ckpt = tmp_path / 'x.pth'
    ckpt.write_bytes(b'')
    monkeypatch.setattr(sys, 'argv', ['main.py', 'x.geojson', '--types', 'Crosswalk',
                                      '--crosswalk-checkpoint', str(ckpt), *flags])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 2 and 'applies to CurbRamp only' in capsys.readouterr().err


def test_crosswalk_needs_a_checkpoint(capsys, tmp_path):
    import argparse
    import main
    args = argparse.Namespace(types=['Crosswalk'], decode='argmax', border='exclude',
                              scan_only=False, crosswalk_checkpoint=None)
    with pytest.raises(SystemExit):
        main.check_type_flags(argparse.ArgumentParser(), args)
    assert '--crosswalk-checkpoint' in capsys.readouterr().err
    args.crosswalk_checkpoint = str(tmp_path / 'missing.pth')
    with pytest.raises(SystemExit):
        main.check_type_flags(argparse.ArgumentParser(), args)
    assert 'no such file' in capsys.readouterr().err
    args.scan_only = True                                    # a scan loads no model
    main.check_type_flags(argparse.ArgumentParser(), args)


def test_detector_type_kwargs():
    import argparse
    import main
    args = argparse.Namespace(types=['Crosswalk', 'CurbRamp'], decode='gaussian',
                              border='keep', crosswalk_checkpoint='c.pth')
    assert main.detector_type_kwargs(args) == {
        'CurbRamp': {'decode': 'gaussian', 'border': 'keep'},
        'Crosswalk': {'checkpoint': 'c.pth'}}
    args.types = ['CurbRamp']
    assert list(main.detector_type_kwargs(args)) == ['CurbRamp']


def test_check_fixed_frames():
    import main
    main.check_fixed_frames({'CurbRamp': _FakeCurb([]), 'Crosswalk': _FakeXwalk([])})
    bad = _FakeXwalk([])
    bad.border = 'exclude'
    with pytest.raises(SystemExit, match='spec says'):
        main.check_fixed_frames({'Crosswalk': bad})


# --- send_to_ps.py ----------------------------------------------------------------------------

import shutil  # noqa: E402
from types import SimpleNamespace  # noqa: E402

PROD = 'https://ps.example/ai/submitLabelsOnPano'
GOLDEN = REPO / 'tests' / 'fixtures' / 'multitype_submit' / 'curbramp_only.jsonl'
# (x, y, c): kept at 0.23 / dropped below it / on the camera rig / below the 0.23 default
XW_DETS = [(0.5, 0.6, 0.9), (0.25, 0.6, 0.25), (0.3, 0.95, 0.8), (0.2, 0.6, 0.2),
           (0.1, 0.6, 0.07)]


def _crosswalk_file(monkeypatch, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    line = _crosswalk_line(monkeypatch, XW_DETS)
    path = run_dir / 'results.jsonl'
    path.write_text(json.dumps(line) + '\n', encoding='utf-8', newline='\n')
    return path


def _capture(monkeypatch):
    import send_to_ps
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text='')
    monkeypatch.setattr(send_to_ps, 'send_to_project_sidewalk',
                        lambda payload, url, key=None: sent.append(payload) or ok)
    return sent


def _record(path):
    return json.loads(Path(f'{path}.submission.json').read_text(encoding='utf-8'))


def test_crosswalk_dry_run(monkeypatch, tmp_path, capsys):
    import send_to_ps
    path = _crosswalk_file(monkeypatch, tmp_path / 'runs' / 'xw')
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    out = capsys.readouterr().out
    payload = json.loads(out[out.index('{'):out.rindex('}') + 1])
    assert payload['label_type'] == 'Crosswalk'
    # 0.23 by default; the y=0.95 detection is on the camera rig (mask on for Crosswalk)
    assert [lab['confidence'] for lab in payload['labels']] == [0.9, 0.25]
    assert {k: payload[k] for k in reg.PROVENANCE_KEYS} == XW_PROV
    assert not {'models', 'label_types', 'detections', 'detection_decode',
                'detection_border'} & set(payload)
    assert not list(path.parent.glob('*.submitted'))      # a dry run records nothing


def test_crosswalk_submission_record_and_no_clash_with_curb_ramp(monkeypatch, tmp_path):
    """The record says Crosswalk went out under `keep`; the city's `exclude` CurbRamp campaign
    in a sibling run dir on the same endpoint neither refuses it nor is refused by it."""
    import send_to_ps
    curb = tmp_path / 'runs' / 'city'
    curb.mkdir(parents=True)
    shutil.copyfile(GOLDEN, curb / 'results.jsonl')
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(curb / 'results.jsonl'), PROD)
    n_curb = len(sent)

    path = _crosswalk_file(monkeypatch, tmp_path / 'runs' / 'city_crosswalk')
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert len(sent) == n_curb + 1 and sent[-1]['label_type'] == 'Crosswalk'
    rec = _record(path)
    assert rec['label_types'] == ['Crosswalk']
    assert rec['type_frames'] == {'Crosswalk': {'detection_decode': 'argmax',
                                                'detection_border': 'keep'}}
    assert 'detection_border' not in rec and 'detection_decode' not in rec
    state = rec['endpoints'][PROD]
    assert (state['min_confidence'], state['rig_masked'], state['labels_submitted']) == \
        (0.23, True, 2)
    assert send_to_ps.record_label_types(_record(curb / 'results.jsonl')) == ['CurbRamp']
    # ...and the CurbRamp file still passes its own guard beside the Crosswalk record
    assert send_to_ps.check_detection_border(curb / 'results.jsonl', PROD) == 'exclude'


def test_crosswalk_frame_clash_with_a_crosswalk_campaign(monkeypatch, tmp_path):
    import send_to_ps
    path = _crosswalk_file(monkeypatch, tmp_path / 'runs' / 'xw2')
    other = tmp_path / 'runs' / 'xw1'
    other.mkdir(parents=True)
    (other / 'results.jsonl.submission.json').write_text(json.dumps({
        'input_file': 'results.jsonl', 'sha256': '0' * 64, 'total_lines': 1, 'total_bytes': 1,
        'endpoints': {PROD: {'submitted_lines': 1, 'labels_submitted': 1}},
        'label_types': ['Crosswalk'],
        'type_frames': {'Crosswalk': {'detection_decode': 'argmax',
                                      'detection_border': 'exclude'}}}), encoding='utf-8')
    with pytest.raises(ValueError, match=r'Crosswalk detections.*xw1/.*\(exclude\)'):
        send_to_ps.check_detection_border(path, PROD)
    assert send_to_ps.check_detection_decode(path, PROD) == 'argmax'   # same decode: fine


def test_band_refused_on_a_crosswalk_file(monkeypatch, tmp_path):
    import send_to_ps
    path = _crosswalk_file(monkeypatch, tmp_path / 'runs' / 'xw')
    sent = _capture(monkeypatch)
    for dry in (True, False):
        with pytest.raises(ValueError, match='not CurbRamp-only'):
            send_to_ps.process_jsonl_file(str(path), PROD, dry_run=dry, min_confidence=0.1,
                                          max_confidence=0.23)
    assert sent == []


def test_type_frame_counts_of_a_curb_ramp_file_match_the_line_level_counters():
    import send_to_ps
    counts = send_to_ps.type_frame_counts(GOLDEN)
    assert list(counts) == ['CurbRamp']
    assert counts['CurbRamp'] == (detectors.decodes_in_file(GOLDEN),
                                  detectors.borders_in_file(GOLDEN))
    assert send_to_ps.type_frame_record_values(GOLDEN) == {}


# --- scripts/crosswalk_parity.py (the GPU script's matching rule) -----------------------------

def test_parity_compare_rule():
    import crosswalk_parity as cp
    a = [(512 / 1024, 256 / 512, 0.9), (100 / 1024, 300 / 512, 0.4)]
    assert cp.compare(a, [(x, y, c + 1e-4) for x, y, c in a])['exact'] == 2
    r = cp.compare(a, [(512 / 1024, 256 / 512, 0.91), (101 / 1024, 300 / 512, 0.401)])
    assert (r['bad_value'], r['moved']) == (1, 1)            # value off at the same pixel; a 1-px move
    r = cp.compare(a + [(0.7, 0.7, 0.052)], a)              # a peak at the floor on one side only
    assert (r['unmatched_stored'], r['excused']) == ([], 1)
    r = cp.compare(a + [(0.7, 0.7, 0.3)], a)
    assert r['unmatched_stored'] == [0.3]
    r = cp.compare(a, [a[0], (130 / 1024, 300 / 512, 0.4)])  # beyond the NMS window: no match
    assert (r['unmatched_stored'], r['unmatched_labeler']) == ([0.4], [0.4])
