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


def test_type_frame_counts_refuses_a_line_whose_models_lack_a_listed_type(tmp_path):
    """Skipping it would also hide the frame of the line's other type, which can still be sent."""
    import send_to_ps
    line = {'label_types': ['Crosswalk', 'CurbRamp'], 'detections': [],
            'models': {'CurbRamp': {'model_id': 'rampnet-model@abc'}}}
    path = tmp_path / 'results.jsonl'
    path.write_text('not json\n' + json.dumps(line) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r"line 2: .*no entry for 'Crosswalk'"):
        send_to_ps.type_frame_counts(path)


def test_frame_helpers_use_counts_they_are_given(tmp_path):
    """process_jsonl_file parses the file once and hands the counts to every frame helper."""
    from collections import Counter
    import send_to_ps
    counts = {'Crosswalk': (Counter({'argmax': 3}), Counter({'keep': 3}))}
    missing = tmp_path / 'runs' / 'x' / 'results.jsonl'     # never opened
    assert send_to_ps.type_frame_record_values(missing, counts) == {
        'Crosswalk': {'detection_decode': 'argmax', 'detection_border': 'keep'}}
    assert send_to_ps.check_detection_decode(missing, None, counts) == 'argmax'
    assert send_to_ps.check_detection_border(missing, None, counts) == 'exclude'


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
    # Near the operating point is not near the floor: such a peak must exist on both sides.
    r = cp.compare(a, a + [(0.7, 0.7, 0.232)])
    assert (r['unmatched_labeler'], r['excused']) == ([0.232], 0)
    r = cp.compare(a, [a[0], (130 / 1024, 300 / 512, 0.4)])  # beyond the NMS window: no match
    assert (r['unmatched_stored'], r['unmatched_labeler']) == ([0.4], [0.4])


def test_parity_first_peaks_needs_finder_order():
    import crosswalk_parity as cp
    stored = [(0, 0, 1.3), (0, 0, 1.1), (0, 0, 0.6), (0, 0, 0.2)]   # >1.0 ties once clipped
    assert cp.first_peaks(stored, cap=3) == stored[:3]
    with pytest.raises(ValueError):
        cp.first_peaks([(0, 0, 0.2), (0, 0, 0.6)], cap=1)


def test_parity_jpeg_round_trip_and_pixel_mad(tmp_path):
    import crosswalk_parity as cp
    from PIL import Image
    im = Image.new('RGB', (64, 32), (120, 60, 30))
    im.save(tmp_path / 'p.jpg', 'JPEG', quality=92)
    rt = cp.jpeg_round_trip(im, 92)
    assert rt.size == im.size and rt.mode == 'RGB'
    assert cp.pixel_mad(rt, tmp_path / 'p.jpg') == 0.0         # same pixels, same encoder
    assert cp.pixel_mad(im, tmp_path / 'missing.jpg') is None
    assert cp.pixel_mad(im.resize((32, 16)), tmp_path / 'p.jpg') is None


# --- the range gate (issue #192) ---------------------------------------------------------------

# Coarse-cell rows (residue 3 or 4 mod 8) on both sides of the 5.625-deg cell edge at row 272,
# plus the edge itself. (x, y, confidence); the confidences tell the peaks apart.
EDGE_DETS = [(0.10, 267 / 512, 0.50), (0.20, 268 / 512, 0.51), (0.30, 272 / 512, 0.52),
             (0.40, 275 / 512, 0.53), (0.50, 276 / 512, 0.54)]
FAR = 268 / 512      # 4.2 deg below the horizon: beyond Crosswalk's gate


def test_range_gate_rule_and_spec_values():
    assert detectors.depression_deg(272 / 512) == 5.625          # a cell edge, exactly
    assert detectors.beyond_range_gate(271 / 512, 5.625)
    assert not detectors.beyond_range_gate(272 / 512, 5.625)     # at the gate: kept
    assert not detectors.beyond_range_gate(0.0, None)            # ungated
    assert reg.spec_for('Crosswalk').min_depression_deg == 5.625
    assert reg.spec_for('CurbRamp').min_depression_deg is None


def test_range_gate_boundary_drops_crosswalk_rows_above_the_cell_edge(monkeypatch):
    import send_to_ps
    line = _crosswalk_line(monkeypatch, EDGE_DETS)
    kept = [lab['confidence'] for lab in send_to_ps.transform_record(line)['labels']]
    assert kept == [0.52, 0.53, 0.54]          # rows 267/268 dropped; 272, 275, 276 kept
    assert len(send_to_ps.transform_record(line, range_gate=False)['labels']) == 5
    assert len(send_to_ps.transform_record(line, range_gate=4.0)['labels']) == 4   # a number
    # The same peaks on a CurbRamp line ship untouched: CurbRamp is ungated.
    curb = json.loads(GOLDEN.read_text(encoding='utf-8').splitlines()[0])
    curb['detections'] = [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                          for x, y, c in EDGE_DETS]
    assert len(send_to_ps.transform_record(curb)['labels']) == 5


def _gate_file(monkeypatch, run_dir, per_line):
    """A Crosswalk-only file, one line per entry of `per_line` (a list of detections)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, dets in enumerate(per_line, 1):
        line = _crosswalk_line(monkeypatch, dets)
        line['pano']['panorama_id'] = f'G{i}'
        lines.append(json.dumps(line) + '\n')
    path = run_dir / 'results.jsonl'
    path.write_text(''.join(lines), encoding='utf-8', newline='\n')
    return path


def _strip_gate(path):
    """Make the record look like a campaign sent before the gate existed (no key)."""
    rec_path = Path(f'{path}.submission.json')
    rec = json.loads(rec_path.read_text(encoding='utf-8'))
    del rec['endpoints'][PROD]['range_gate_deg']
    rec_path.write_text(json.dumps(rec), encoding='utf-8')


def test_crosswalk_record_carries_the_gate_and_the_gated_count(monkeypatch, tmp_path, capsys):
    import send_to_ps
    path = _gate_file(monkeypatch, tmp_path / 'xw', [[(0.5, 0.6, 0.9), (0.2, FAR, 0.8)],
                                                     [(0.5, 0.6, 0.7)]])
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    assert 'Held back by the range gate' in capsys.readouterr().out
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    out = capsys.readouterr().out
    assert [len(p['labels']) for p in sent] == [1, 1]
    assert 'Detections below --min-confidence 0.23 (not submitted): 0' in out
    assert 'Crosswalk 1 (shallower than 5.625 deg below the horizon)' in out
    state = _record(path)['endpoints'][PROD]
    assert (state['range_gate_deg'], state['rig_masked'], state['labels_submitted']) == \
        (5.625, True, 2)


def test_resume_refuses_a_gate_change_that_touches_sent_labels(monkeypatch, tmp_path):
    """Line 1 went out ungated with a far label; gating the rest would leave the campaign
    under two rules. The override finishes it under the RECORDED gate (none)."""
    import send_to_ps
    path = _gate_file(monkeypatch, tmp_path / 'xw', [[(0.5, 0.6, 0.9), (0.2, FAR, 0.8)],
                                                     [(0.5, 0.6, 0.7), (0.2, FAR, 0.6)]])
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)
    _strip_gate(path)
    del sent[:]
    with pytest.raises(ValueError, match=r'range gate none.*1 line\(s\) already sent carry 2'):
        send_to_ps.process_jsonl_file(str(path), PROD)
    assert sent == []
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True)
    assert [len(p['labels']) for p in sent] == [2]              # line 2, ungated
    state = _record(path)['endpoints'][PROD]
    assert state['range_gate_deg'] is None and state['labels_submitted'] == 4


def test_resume_adopts_the_gate_when_no_sent_label_is_affected(monkeypatch, tmp_path, capsys):
    """Juneau's shape: a Crosswalk campaign recorded before the gate, none of whose sent
    lines holds a far label, adopts it with a note instead of refusing."""
    import send_to_ps
    path = _gate_file(monkeypatch, tmp_path / 'xw', [[(0.5, 0.6, 0.9)],
                                                     [(0.5, 0.6, 0.7), (0.2, FAR, 0.6)]])
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)
    _strip_gate(path)
    capsys.readouterr()
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert 'NOTE (range gate)' in capsys.readouterr().out
    assert [len(p['labels']) for p in sent] == [1, 1]          # line 2's far label held back
    state = _record(path)['endpoints'][PROD]
    assert (state['range_gate_deg'], state['labels_submitted']) == (5.625, 2)


def test_the_recount_uses_the_recorded_gate(monkeypatch, tmp_path):
    import send_to_ps
    path = _gate_file(monkeypatch, tmp_path / 'xw', [[(0.5, 0.6, 0.9), (0.2, FAR, 0.8)]])
    digest, n, size = send_to_ps.hash_and_count(path)
    rec_path = send_to_ps.submission_record_path(str(path))
    entry = {'submitted_lines': 1, 'labels_submitted': 2, 'min_confidence': 0.23,
             'rig_masked': True}
    for recorded, labels in ((None, 2), (5.625, 1), ('absent', 2)):
        prev = dict(entry) if recorded == 'absent' else dict(entry, range_gate_deg=recorded)
        send_to_ps.write_submission_record(rec_path, path, digest, n, size, {1}, PROD, 0.23,
                                           {'endpoints': {PROD: prev}},
                                           label_types=['Crosswalk'])
        state = json.loads(rec_path.read_text(encoding='utf-8'))['endpoints'][PROD]
        assert (state['range_gate_deg'], state['labels_submitted']) == \
            (None if recorded == 'absent' else recorded, labels)


def test_multi_type_record_gates_crosswalk_only(monkeypatch, tmp_path):
    import main
    import send_to_ps
    result = _process(monkeypatch, {'CurbRamp': _FakeCurb([(0.5, 0.25, 0.9), (0.6, FAR, 0.9)]),
                                    'Crosswalk': _FakeXwalk([(0.1, 0.6, 0.3), (0.2, FAR, 0.9)])})
    line = main.build_output_line(result, {'CurbRamp': make_provenance(), 'Crosswalk': XW_PROV})
    path = tmp_path / 'results.jsonl'
    path.write_text(json.dumps(line) + '\n', encoding='utf-8', newline='\n')
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert {p['label_type']: len(p['labels']) for p in sent} == {'Crosswalk': 1, 'CurbRamp': 2}
    types = _record(path)['endpoints'][PROD]['types']
    assert types['Crosswalk']['range_gate_deg'] == 5.625
    assert 'range_gate_deg' not in types['CurbRamp']


def test_above_the_horizon_is_gated_and_zero_is_a_horizon_gate(monkeypatch):
    import send_to_ps
    line = _crosswalk_line(monkeypatch, [(0.5, 0.4, 0.9), (0.6, 0.6, 0.8)])   # 18 deg up / down
    assert [lab['confidence'] for lab in send_to_ps.transform_record(line)['labels']] == [0.8]
    assert len(send_to_ps.transform_record(line, range_gate=0)['labels']) == 1   # horizon gate
    assert len(send_to_ps.transform_record(line, range_gate=False)['labels']) == 2
    with pytest.raises(ValueError, match='cannot be negative'):
        send_to_ps.transform_record(line, range_gate=-1.0)


def _lose_ungated_far_line(monkeypatch, tmp_path, far_conf=0.8):
    """Line 1 live UNGATED with a far label (record stripped of its gate, count made honest);
    line 2 still to send."""
    import send_to_ps
    path = _gate_file(monkeypatch, tmp_path / 'xw', [[(0.5, 0.6, 0.9), (0.2, FAR, far_conf)],
                                                     [(0.5, 0.6, 0.7), (0.2, FAR, 0.6)]])
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)
    _strip_gate(path)
    rec_path = Path(f'{path}.submission.json')
    rec = json.loads(rec_path.read_text(encoding='utf-8'))
    rec['endpoints'][PROD]['labels_submitted'] = 2
    rec_path.write_text(json.dumps(rec), encoding='utf-8')
    del sent[:]
    return path, sent


def test_override_with_a_lost_sidecar_keeps_the_recorded_gate(monkeypatch, tmp_path, capsys):
    """Review P1: with the sidecar gone, "0 sent lines carry 0 labels" proves nothing, so the
    override must not adopt; the campaign stays ungated, as line 1 went out."""
    import send_to_ps
    path, sent = _lose_ungated_far_line(monkeypatch, tmp_path)
    Path(f'{path}.submitted').unlink()
    capsys.readouterr()
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True)
    assert 'keeps its recorded range gate' in capsys.readouterr().out
    assert _record(path)['endpoints'][PROD]['range_gate_deg'] is None
    assert [len(p['labels']) for p in sent] == [2, 2]           # re-sent ungated, as recorded


def test_override_with_a_changed_threshold_keeps_the_recorded_gate(monkeypatch, tmp_path):
    """Review P2: counted at 0.5 the far 0.3 label on line 1 vanishes from both counts, which
    would read as "no sent label affected". The override keeps the recorded gate instead."""
    import send_to_ps
    path, sent = _lose_ungated_far_line(monkeypatch, tmp_path, far_conf=0.3)
    send_to_ps.process_jsonl_file(str(path), PROD, ignore_guard=True, min_confidence=0.5)
    assert _record(path)['endpoints'][PROD]['range_gate_deg'] is None
    assert [len(p['labels']) for p in sent] == [2]              # line 2, ungated


def test_adoption_counts_at_the_recorded_threshold_and_needs_every_sent_line(monkeypatch,
                                                                            tmp_path):
    """range_gate_on_resume itself, trusted: a changed --min-confidence cannot hide a far label
    (it counts at the entry's 0.23), and a sidecar short of the record never adopts."""
    import send_to_ps
    path = _gate_file(monkeypatch, tmp_path / 'xw', [[(0.5, 0.6, 0.9), (0.2, FAR, 0.3)]])
    entry = {'submitted_lines': 1, 'min_confidence': 0.23, 'rig_masked': True}
    rp = Path(f'{path}.submission.json')
    with pytest.raises(ValueError, match='range gate'):
        send_to_ps.range_gate_on_resume(entry, 'Crosswalk', path, {1}, 0.5, None, 'Crosswalk',
                                        PROD, rp)
    gate, note = send_to_ps.range_gate_on_resume(dict(entry, submitted_lines=2), 'Crosswalk',
                                                 path, {1}, 0.23, None, 'Crosswalk', PROD, rp)
    assert gate is None and 'do not account for the 2 recorded' in note


def test_dry_run_says_what_a_real_run_would_do_with_the_gate(monkeypatch, tmp_path, capsys):
    import send_to_ps
    path, _ = _lose_ungated_far_line(monkeypatch, tmp_path)
    capsys.readouterr()
    send_to_ps.process_jsonl_file(str(path), PROD, dry_run=True)
    assert 'a real run would REFUSE' in capsys.readouterr().out
    clean = _gate_file(monkeypatch, tmp_path / 'xw2', [[(0.5, 0.6, 0.9)],
                                                       [(0.5, 0.6, 0.7), (0.2, FAR, 0.6)]])
    send_to_ps.process_jsonl_file(str(clean), PROD, limit=1)
    _strip_gate(clean)
    capsys.readouterr()
    send_to_ps.process_jsonl_file(str(clean), PROD, dry_run=True)
    assert 'a real run would adopt' in capsys.readouterr().out


def _two_type_gate_file(monkeypatch, tmp_path, first_far):
    """CurbRamp + Crosswalk, two lines; line 2 has a far Crosswalk peak, line 1 one iff
    `first_far`. Both lines carry a far CurbRamp peak, which is never gated."""
    import main
    lines = []
    for i, xw in enumerate(([(0.1, 0.6, 0.3)] + ([(0.2, FAR, 0.9)] if first_far else []),
                            [(0.1, 0.6, 0.3), (0.2, FAR, 0.9)]), 1):
        result = _process(monkeypatch, {'CurbRamp': _FakeCurb([(0.5, 0.25, 0.9), (0.6, FAR, 0.9)]),
                                        'Crosswalk': _FakeXwalk(xw)})
        line = json.loads(json.dumps(main.build_output_line(
            result, {'CurbRamp': make_provenance(), 'Crosswalk': XW_PROV})))
        line['pano']['panorama_id'] = f'M{i}'
        lines.append(json.dumps(line) + '\n')
    path = tmp_path / 'results.jsonl'
    path.write_text(''.join(lines), encoding='utf-8', newline='\n')
    return path


@pytest.mark.parametrize('first_far', [True, False])
def test_multi_type_resume_refuses_or_adopts_per_type(monkeypatch, tmp_path, first_far):
    import send_to_ps
    path = _two_type_gate_file(monkeypatch, tmp_path, first_far)
    sent = _capture(monkeypatch)
    send_to_ps.process_jsonl_file(str(path), PROD, limit=1)
    rec_path = Path(f'{path}.submission.json')
    rec = json.loads(rec_path.read_text(encoding='utf-8'))
    del rec['endpoints'][PROD]['types']['Crosswalk']['range_gate_deg']   # sent before the gate
    rec_path.write_text(json.dumps(rec), encoding='utf-8')
    del sent[:]
    if first_far:
        with pytest.raises(ValueError, match='Crosswalk campaign'):
            send_to_ps.process_jsonl_file(str(path), PROD)
        assert sent == []
        return
    send_to_ps.process_jsonl_file(str(path), PROD)
    assert {p['label_type']: len(p['labels']) for p in sent} == {'Crosswalk': 1, 'CurbRamp': 2}
    types = _record(path)['endpoints'][PROD]['types']
    assert types['Crosswalk']['range_gate_deg'] == 5.625
    assert 'range_gate_deg' not in types['CurbRamp']
