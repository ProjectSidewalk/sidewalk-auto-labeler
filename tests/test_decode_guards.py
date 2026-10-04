"""The peak decode's guards (issue #111): wherever two runs could be combined, a decode is
bound and a mix refused. CPU only, no model, no network, no scikit-image."""
import json
from collections import Counter
from types import SimpleNamespace

import pytest

import detectors


def test_record_decode_and_single_decode():
    assert detectors.record_decode({}) == 'argmax'
    assert detectors.single_decode(Counter(), 'f') == 'argmax'
    with pytest.raises(ValueError, match='mixes peak decodes'):
        detectors.single_decode(Counter({'argmax': 1, 'gaussian': 1}), 'f')


def _line(decode=None, pid='p1'):
    rec = {'detections': [{'x_normalized': 0.5, 'y_normalized': 0.6, 'confidence': 0.7}],
           'pano': {'panorama_id': pid}}
    if decode is not None:
        rec['detection_decode'] = decode
    return json.dumps(rec) + '\n'


def test_build_output_line_marks_only_non_argmax():
    import main
    from conftest import make_process_result, make_provenance
    fake_result, prov = make_process_result(), make_provenance()
    plain = main.build_output_line(fake_result, prov)
    assert 'detection_decode' not in plain
    assert main.build_output_line({**fake_result, 'decode': 'argmax'}, prov) == plain
    g = main.build_output_line({**fake_result, 'decode': 'gaussian'}, prov)
    assert g['detection_decode'] == 'gaussian'
    assert {k: v for k, v in g.items() if k != 'detection_decode'} == plain


def test_bind_decode(tmp_path):
    import main
    run = tmp_path / 'run'
    run.mkdir()
    m = {}
    assert main.bind_decode(m, 'gaussian', run) is True          # fresh: binds
    assert m['detection_decode'] == 'gaussian'
    assert main.bind_decode(m, 'gaussian', run) is False
    with pytest.raises(SystemExit):
        main.bind_decode(m, 'argmax', run)
    # a pre-#111 manifest over existing records is argmax, and is not rewritten
    legacy = {}
    (run / 'results.jsonl').write_text(_line(), encoding='utf-8')
    assert main.bind_decode(legacy, 'argmax', run) is False and legacy == {}
    with pytest.raises(SystemExit):
        main.bind_decode(legacy, 'gaussian', run)


def test_load_or_init_run_dir_binds_decode(tmp_path, monkeypatch):
    import main
    monkeypatch.chdir(tmp_path)
    run = tmp_path / 'runs' / 'x'
    m = main.load_or_init_run_dir(run, 'a.geojson', {'type': 'Polygon', 'coordinates': []},
                                  'h', 'gsv', decode='gaussian')
    assert m['detection_decode'] == 'gaussian'
    with pytest.raises(SystemExit):
        main.load_or_init_run_dir(run, 'a.geojson', {}, 'h', 'gsv', decode='argmax')
    main.load_or_init_run_dir(run, 'a.geojson', {}, 'h', 'gsv', decode=None)   # scan-only


def test_reinfer_resolve_decode(tmp_path):
    import reinfer
    out = tmp_path / 'f01.jsonl'
    assert reinfer.resolve_decode({}, out) == 'argmax'
    assert reinfer.resolve_decode({'detection_decode': 'gaussian'}, out) == 'gaussian'
    out.write_text(_line('gaussian'), encoding='utf-8')
    with pytest.raises(SystemExit):
        reinfer.resolve_decode({}, out)                    # argmax into a gaussian file
    assert reinfer.resolve_decode({}, out, 'gaussian') == 'gaussian'


def _reinfer_pair(tmp_path, old_decode, new_decode):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'manifest.json').write_text('{}', encoding='utf-8')
    (run / 'results.jsonl').write_text(_line(old_decode), encoding='utf-8')
    (run / 'results.f01.jsonl').write_text(_line(new_decode), encoding='utf-8')
    return run


def test_reinfer_verify_refuses_mixed_decodes(tmp_path):
    import reinfer
    run = _reinfer_pair(tmp_path, None, 'gaussian')
    with pytest.raises(SystemExit, match='cannot reproduce'):
        reinfer.main_cli([str(run), '--verify'])
    with pytest.raises(SystemExit, match='another frame'):
        reinfer.main_cli([str(run), '--verify', '--allow-mixed-decode', '--write-band-file'])


def test_fuse_refuses_a_mixed_file(tmp_path):
    import fuse_sites as fs
    pano = {'panorama_id': 'p', 'lat': 47.6, 'lng': -122.3, 'camera_heading': 0.0,
            'source': 'launch'}
    path = tmp_path / 'results.jsonl'
    path.write_text(json.dumps({'detections': [], 'pano': pano}) + '\n'
                    + json.dumps({'detections': [], 'pano': {**pano, 'panorama_id': 'q'},
                                  'detection_decode': 'gaussian'}) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='mixes peak decodes'):
        fs.load_results(path, read_heights=False)
    panos, _ = fs.load_results(path, read_heights=False, allow_mixed_decode=True)
    assert [p.decode for p in panos] == ['argmax', 'gaussian']
    with pytest.raises(ValueError):
        fs.fuse(panos, fs.FuseParams())


def test_eval_sites_refuses_a_gaussian_run():
    import eval_sites as es
    import fuse_sites as fs
    p = fs.SlimPano('p', 0.0, 0.0, 0.0, None, None, None, 'launch', [], decode='gaussian')
    with pytest.raises(ValueError, match='keyed to argmax'):
        es.require_argmax([p], 'runs/x')
    es.require_argmax([fs.SlimPano('q', 0.0, 0.0, 0.0, None, None, None, 'launch', [])], 'x')


def test_send_to_ps_decode_guard(tmp_path):
    import send_to_ps as sp
    f = tmp_path / 'results.jsonl'
    f.write_text(_line() + _line('gaussian', 'p2'), encoding='utf-8')
    with pytest.raises(ValueError, match='mixes peak decodes'):
        sp.check_detection_decode(f)
    g = tmp_path / 'results.gauss.jsonl'
    g.write_text(_line('gaussian'), encoding='utf-8')
    assert sp.check_detection_decode(g) == 'gaussian'
    # an argmax campaign already recorded beside it: a frame change for the city
    (tmp_path / 'results.band.jsonl.submission.json').write_text(
        json.dumps({'sha256': 'x', 'endpoints': {'https://e': {'submitted_lines': 1}}}),
        encoding='utf-8')
    with pytest.raises(ValueError, match='frame change'):
        sp.check_detection_decode(g)


def test_provenance_gate_refuses_a_gaussian_arm(tmp_path):
    import provenance_gate as pg
    rec = {'detections': [], 'detection_decode': 'gaussian',
           'pano': {'panorama_id': 'p', 'width': 16384, 'height': 8192}}
    f = tmp_path / 'results.jsonl'
    f.write_text(json.dumps(rec) + '\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='placed by argmax'):
        pg.load_run(f)
    rec.pop('detection_decode')
    f.write_text(json.dumps(rec) + '\n', encoding='utf-8')
    assert pg.load_run(f) == {'p': (16384, 8192, [])}


# --- send_to_ps: the override branch and the argmax record (review S2, M9) ---------------

PROD = 'https://ps.example/ai/submitLabelsOnPano'

# What an argmax campaign's submission record holds, byte for byte, under a frozen clock and
# host. Generated with origin/main's send_to_ps.py (before #111) on the same input, so it pins
# that the decode work changed nothing an argmax campaign writes.
ARGMAX_RECORD = '''{
  "input_file": "results.jsonl",
  "sha256": "c9e8efdd7d83a062d68f830403aedf98eb55d40e293383bc12066d2844d0c933",
  "total_lines": 3,
  "total_bytes": 600,
  "endpoints": {
    "https://ps.example/ai/submitLabelsOnPano": {
      "rig_masked": true,
      "submitted_lines": 3,
      "labels_submitted": 3,
      "min_confidence": 0.3,
      "first_submission_utc": "2026-10-02T12:00:00Z",
      "last_submission_utc": "2026-10-02T12:00:00Z",
      "last_run_host": "host"
    }
  }
}
'''


def _frozen_send(monkeypatch):
    import send_to_ps as sp
    from datetime import datetime as real, timezone

    class Frozen(real):
        @classmethod
        def now(cls, tz=None):
            return real(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(sp, 'datetime', Frozen)
    monkeypatch.setattr(sp, 'socket', SimpleNamespace(gethostname=lambda: 'host'))
    sent = []
    ok = SimpleNamespace(status_code=200, ok=True, text='')
    monkeypatch.setattr(sp, 'send_to_project_sidewalk',
                        lambda payload, url, key=None: sent.append(payload) or ok)
    return sp, sent


def _results(path, decodes):
    lines = []
    for i, d in enumerate(decodes, 1):
        rec = {'detections': [{'x_normalized': 0.5, 'y_normalized': 0.5, 'confidence': 0.9}],
               'label_type': 'CurbRamp', 'model_id': 'rampnet-model',
               'pano': {'panorama_id': f'PID{i}', 'width': 16384, 'height': 8192}}
        if d is not None:
            rec['detection_decode'] = d
        lines.append(json.dumps(rec) + '\n')
    path.write_bytes(''.join(lines).encode())
    return path


def test_argmax_submission_record_is_unchanged(tmp_path, monkeypatch):
    sp, sent = _frozen_send(monkeypatch)
    f = _results(tmp_path / 'results.jsonl', [None, None, None])
    sp.process_jsonl_file(str(f), PROD)
    assert len(sent) == 3
    assert (tmp_path / 'results.jsonl.submission.json').read_bytes() == ARGMAX_RECORD.encode()


def test_mixed_override_records_the_mix_and_blocks_later_argmax(tmp_path, monkeypatch):
    sp, sent = _frozen_send(monkeypatch)
    f = _results(tmp_path / 'results.jsonl', [None, None, 'gaussian'])   # mostly argmax
    with pytest.raises(ValueError, match='mixes peak decodes'):
        sp.process_jsonl_file(str(f), PROD)
    assert not sent
    sp.process_jsonl_file(str(f), PROD, allow_mixed_decode=True)
    assert len(sent) == 3
    record = json.loads((tmp_path / 'results.jsonl.submission.json').read_text())
    assert record['detection_decode'] == {'argmax': 2, 'gaussian': 1}
    assert 'mixes peak decodes' in record['mixed_decode_override']['reason']
    assert sp.recorded_decode(record) == 'mixed'
    # a later, purely argmax campaign beside it is refused: gaussian labels are live there
    g = _results(tmp_path / 'results.band.jsonl', [None, None])
    with pytest.raises(ValueError, match='frame change'):
        sp.check_detection_decode(g)


def test_recorded_decode():
    import send_to_ps as sp
    assert sp.recorded_decode({}) == 'argmax'
    assert sp.recorded_decode({'detection_decode': 'gaussian'}) == 'gaussian'
    assert sp.recorded_decode({'detection_decode': {'argmax': 1, 'gaussian': 1}}) == 'mixed'
    # a single-decode file sent under the override (it differed from its neighbours)
    assert sp.recorded_decode({'detection_decode': 'gaussian',
                               'mixed_decode_override': {'reason': 'x'}}) == 'mixed'


# --- fuse_sites' report modes and the research scorers (review M5, M6) -------------------

def _mixed_results(tmp_path):
    pano = {'panorama_id': 'p', 'lat': 47.6, 'lng': -122.3, 'camera_heading': 0.0,
            'source': 'launch'}
    path = tmp_path / 'results.jsonl'
    path.write_text(json.dumps({'detections': [], 'pano': pano}) + '\n'
                    + json.dumps({'detections': [], 'pano': {**pano, 'panorama_id': 'q'},
                                  'detection_decode': 'gaussian'}) + '\n', encoding='utf-8')
    return path


@pytest.mark.parametrize('mode', ['--pose-ablation', '--implied-height'])
def test_fuse_report_modes_honour_allow_mixed_decode(tmp_path, mode, capsys):
    import fuse_sites as fs
    path = _mixed_results(tmp_path)
    with pytest.raises(SystemExit):
        fs.main([str(path), mode, '--camera-height-m', '2.6'])
    fs.main([str(path), mode, '--camera-height-m', '2.6', '--allow-mixed-decode'])


def test_gsv_partial_pose_refuses_a_gaussian_run(tmp_path):
    import gsv_partial_pose as gpp
    run = tmp_path / 'x'
    run.mkdir()
    pano = {'panorama_id': 'p', 'lat': 47.6, 'lng': -122.3, 'camera_heading': 0.0,
            'source': 'launch'}
    (run / 'results.jsonl').write_text(json.dumps(
        {'detections': [], 'pano': pano, 'detection_decode': 'gaussian'}) + '\n',
        encoding='utf-8')
    with pytest.raises(SystemExit, match='keyed to argmax'):
        gpp.load_city('x', 2.6, tmp_path)
