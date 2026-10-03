"""The peak border rule (issue #130): `exclude` (default) is the extractor every live label
used, bit for bit; `keep` is RampNet's rule since RampNet#132 (exclude_border=False, no NMS
across the seam). CPU only, no model, no network; the peak finder needs scikit-image."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import detectors
from detectors import decode as dec
from detectors import rampnet_subcell as sc

REPO = Path(__file__).resolve().parents[1]
needs_skimage = pytest.mark.skipif(importlib.util.find_spec('skimage') is None,
                                   reason='needs scikit-image (requirements-test.txt)')
FIXTURE = REPO / 'tests' / 'fixtures' / 'decode_heatmaps.npz'
FIXTURE_EXPECTED = REPO / 'tests' / 'fixtures' / 'decode_expected.json'


def coarse_map(peaks, shape=(64, 128), sigma=1.25):
    """A 64x128 coarse map holding Gaussians at coarse-cell centres (cy, cx, amp)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    c = np.zeros(shape)
    for cy, cx, amp in peaks:
        c += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return c


# One interior peak and one in each of coarse column 0 (left seam), column 127 (right seam)
# and row 0 (zenith). Amplitudes are distinct so the expected order is unambiguous.
INTERIOR = (30, 60, 0.9)
LEFT, RIGHT, TOP = (20, 0, 0.8), (45, 127, 0.7), (0, 90, 0.6)


def edge_heatmap():
    return sc.upsample(coarse_map([INTERIOR, LEFT, RIGHT, TOP])).astype(np.float32)


def px(dets):
    return [(round(x * 1024), round(y * 512)) for x, y, _ in dets]


@pytest.mark.skipif(not FIXTURE.exists(), reason='stored heatmap fixture not present')
@needs_skimage
@pytest.mark.parametrize('decode', ['argmax', 'gaussian'])
def test_default_border_is_bit_identical_on_stored_heatmaps(decode):
    """Test 1: `border` absent or `exclude` is the pre-#130 extractor on every stored heatmap,
    for both decodes, and matches the outputs committed beside the fixture."""
    maps = np.load(FIXTURE)
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding='utf-8'))
    for pid in maps.files:
        h = sc.upsample(maps[pid].astype(np.float64)).astype(np.float32)
        base = dec.detections_from_heatmap(h, decode)
        assert dec.detections_from_heatmap(h, decode, border='exclude') == base
        if decode == 'argmax':
            assert px(base) == px(expected[pid][decode])
        assert len(base) == len(expected[pid][decode])
        for d, e in zip(base, expected[pid][decode]):
            assert d == pytest.approx(e, abs=1e-6)


@needs_skimage
def test_keep_returns_edge_peaks_and_exclude_drops_them():
    """Test 2: `exclude` returns only the interior peak; `keep` returns all four, highest
    first, each scored by the heatmap at its pixel; both decodes run at the edge."""
    h = edge_heatmap()
    ex = dec.detections_from_heatmap(h, 'argmax', border='exclude')
    assert len(ex) == 1
    kp = dec.detections_from_heatmap(h, 'argmax', border='keep')
    assert len(kp) == 4
    scores = [c for _, _, c in kp]
    assert scores == sorted(scores, reverse=True)
    for (x, y, c), (cy, cx, amp) in zip(kp, [INTERIOR, LEFT, RIGHT, TOP]):
        r, col = round(y * 512), round(x * 1024)
        assert c == float(h[r][col])                    # the raw value at the pixel
        # ...which is the coarse peak, less up to 1/16 cell of bilinear blur per axis
        assert c == pytest.approx(amp, abs=0.04)
        assert abs(col - (8 * cx + 3.5)) <= 4 and abs(r - (8 * cy + 3.5)) <= 4
    cols = [round(x * 1024) for x, _, _ in kp]
    rows = [round(y * 512) for _, y, _ in kp]
    assert cols[1] < dec.MIN_DISTANCE and cols[2] >= 1024 - dec.MIN_DISTANCE
    assert rows[3] < dec.MIN_DISTANCE
    # the interior peak is the same under both rules
    assert kp[0] == ex[0]

    g = dec.detections_from_heatmap(h, 'gaussian', border='keep')
    assert [c for _, _, c in g] == scores
    # Off-map neighbours are NaN, so an edge axis gets no sub-cell offset: the peak sits at
    # the edge cell's centre (8*0 + 3.5) / 1024 on that axis -- RampNet's own handling.
    assert g[1][0] == pytest.approx(3.5 / 1024, abs=1e-12)
    assert g[2][0] == pytest.approx((8 * 127 + 3.5) / 1024, abs=1e-12)
    assert g[3][1] == pytest.approx(3.5 / 512, abs=1e-12)
    assert len(dec.detections_from_heatmap(h, 'gaussian', border='exclude')) == 1
    both = dec.detections_both(h, border='keep')
    assert both['argmax'] == kp and both['gaussian'] == g


@needs_skimage
def test_keep_is_rampnets_rule_and_does_not_wrap_nms():
    """Test 3: RampNet's detect_peaks (exclude_border=False) does NOT suppress across the
    seam, so neither does `keep`: a ramp straddling it gives two peaks, one in column 0-9 and
    one in column 1014-1023, and `keep` finds exactly RampNet's pixels."""
    h = sc.upsample(coarse_map([(25, 0, 0.8), (25, 127, 0.75), INTERIOR])).astype(np.float32)
    kp = dec.detections_from_heatmap(h, 'argmax', border='keep')
    cols = sorted(round(x * 1024) for x, _, _ in kp)
    assert len(kp) == 3 and cols[0] < 10 and cols[-1] >= 1014       # both seam peaks kept
    rn = sc.detect_peaks(h, detectors.DETECTION_STORAGE_FLOOR, clip=True,
                         exclude_border=False)
    assert sorted((int(c), int(r)) for r, c, _ in rn) == sorted(px(kp))
    ex = sc.detect_peaks(h, detectors.DETECTION_STORAGE_FLOOR, clip=True, exclude_border=True)
    assert sorted((int(c), int(r)) for r, c, _ in ex) == \
        sorted(px(dec.detections_from_heatmap(h, 'argmax')))


@needs_skimage
def test_unknown_border_is_refused():
    with pytest.raises(ValueError, match='unknown border'):
        dec.detections_from_heatmap(edge_heatmap(), border='wrap')


def test_border_constants():
    assert detectors.DEFAULT_BORDER == detectors.BORDER_EXCLUDE == 'exclude'
    assert detectors.BORDERS == ('exclude', 'keep')
    assert detectors.RECORD_BORDER_KEY == 'detection_border'


# --- The opt-in, bound wherever two runs could be combined (tests 4-8). None of these need
# scikit-image, so they run on every CI configuration. ------------------------------------

import hashlib  # noqa: E402
from collections import Counter  # noqa: E402

# sha256 of json.dumps(main.build_output_line(make_process_result(), make_provenance())) as
# origin/subcell-decode-111's main.py (before #130) writes it: 1,206 bytes.
PRE_130_LINE_SHA256 = '2d9d23b21c31664edbd674971892897de808eed34001675f778894fbcf20d8e1'


def test_record_border_and_single_border():
    """Test 6 (part): a record that predates the key reads as exclude."""
    assert detectors.record_border({'detections': []}) == 'exclude'
    assert detectors.record_border({'detection_border': 'keep'}) == 'keep'
    assert detectors.single_border(Counter(), 'f') == 'exclude'
    with pytest.raises(ValueError, match='mixes peak border rules'):
        detectors.single_border(Counter({'exclude': 1, 'keep': 1}), 'f')


def test_border_band_edge_matches_skimage_exclusion():
    """The band is exactly what exclude_border=True blanks: [0, 10) and [size-10, size)."""
    e = detectors.border_band_edge
    cols = [e(c / 1024, 0.5) for c in (0, 9, 10, 1013, 1014, 1023)]
    assert cols == ['left', 'left', None, None, 'right', 'right']
    rows = [e(0.5, r / 512) for r in (0, 9, 10, 501, 502, 511)]
    assert rows == ['top', 'top', None, None, 'bottom', 'bottom']
    assert e(0.0, 0.0) == 'top'                   # a corner is never counted as seam


def test_build_output_line_marks_only_keep():
    """Test 4: `keep` records carry detection_border; an exclude record is byte-identical
    to the line main.py wrote before #130."""
    import main
    from conftest import make_process_result, make_provenance
    fake, prov = make_process_result(), make_provenance()
    plain = main.build_output_line(fake, prov)
    assert 'detection_border' not in plain
    assert hashlib.sha256(json.dumps(plain).encode()).hexdigest() == PRE_130_LINE_SHA256
    ex = main.build_output_line({**fake, 'border': 'exclude'}, prov)
    assert json.dumps(ex) == json.dumps(plain)
    kp = main.build_output_line({**fake, 'border': 'keep'}, prov)
    assert kp['detection_border'] == 'keep'
    assert {k: v for k, v in kp.items() if k != 'detection_border'} == plain
    both = main.build_output_line({**fake, 'border': 'keep', 'decode': 'gaussian'}, prov)
    assert both['detection_border'] == 'keep' and both['detection_decode'] == 'gaussian'


def _rec(pid='p1', border=None, dets=((0.5, 0.6, 0.7),)):
    rec = {'detections': [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                          for x, y, c in dets],
           'pano': {'panorama_id': pid, 'width': 16384, 'height': 8192, 'lat': 47.6,
                    'lng': -122.3, 'camera_heading': 0.0, 'source': 'launch'}}
    if border is not None:
        rec['detection_border'] = border
    return rec


def _line(**kw):
    return json.dumps(_rec(**kw)) + '\n'


def test_bind_border(tmp_path):
    """Test 5: fresh run binds; a pre-#130 manifest over records is exclude and is not
    rewritten; a mismatch exits naming --border <bound>."""
    import main
    run = tmp_path / 'run'
    run.mkdir()
    m = {}
    assert main.bind_border(m, 'keep', run) is True
    assert m['detection_border'] == 'keep'
    assert main.bind_border(m, 'keep', run) is False
    with pytest.raises(SystemExit, match='--border keep'):
        main.bind_border(m, 'exclude', run)
    legacy = {}
    (run / 'results.jsonl').write_text(_line(), encoding='utf-8')
    assert main.bind_border(legacy, 'exclude', run) is False and legacy == {}
    with pytest.raises(SystemExit, match='--border exclude'):
        main.bind_border(legacy, 'keep', run)


def test_load_or_init_run_dir_binds_border(tmp_path, monkeypatch):
    import main
    monkeypatch.chdir(tmp_path)
    run = tmp_path / 'runs' / 'x'
    m = main.load_or_init_run_dir(run, 'a.geojson', {'type': 'Polygon', 'coordinates': []},
                                  'h', 'gsv', decode='argmax', border='keep')
    assert m['detection_border'] == 'keep' and m['detection_decode'] == 'argmax'
    with pytest.raises(SystemExit, match='border rule'):         # resume under the other rule
        main.load_or_init_run_dir(run, 'a.geojson', {}, 'h', 'gsv', decode='argmax',
                                  border='exclude')
    main.load_or_init_run_dir(run, 'a.geojson', {}, 'h', 'gsv', border=None)   # scan-only
    m2 = main.load_or_init_run_dir(tmp_path / 'runs' / 'y', 'a.geojson',
                                   {'type': 'Polygon', 'coordinates': []}, 'h', 'gsv')
    assert 'detection_border' not in m2                          # --scan-only binds none


def test_reinfer_resolve_border(tmp_path):
    import reinfer
    out = tmp_path / 'f01.jsonl'
    assert reinfer.resolve_border({}, out) == 'exclude'
    assert reinfer.resolve_border({'detection_border': 'keep'}, out) == 'keep'
    out.write_text(_line(border='keep'), encoding='utf-8')
    with pytest.raises(SystemExit, match='#130'):
        reinfer.resolve_border({}, out)                   # exclude into a keep file
    assert reinfer.resolve_border({}, out, 'keep') == 'keep'


def _pair(tmp_path, old, new):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'manifest.json').write_text('{}', encoding='utf-8')
    (run / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in old),
                                       encoding='utf-8')
    (run / 'results.f01.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in new),
                                           encoding='utf-8')
    return run


def test_reinfer_verify_names_the_seam_band(tmp_path, capsys):
    """Test 7: an exclude/keep pair is refused; under the diagnostic, a pano whose new file
    gained a seam peak does NOT reproduce and the report names the border band; and no band
    file can be built from the pair."""
    import reinfer
    seam = (0.0, 0.55, 0.8)                               # column 0: inside the band
    run = _pair(tmp_path, [_rec('p1'), _rec('p2')],
                [_rec('p1', 'keep', ((0.5, 0.6, 0.7), seam)), _rec('p2', 'keep')])
    with pytest.raises(SystemExit, match='--allow-mixed-border'):
        reinfer.main_cli([str(run), '--verify'])
    with pytest.raises(SystemExit, match='seam-band labels'):
        reinfer.main_cli([str(run), '--verify', '--allow-mixed-border', '--write-band-file'])
    assert not (run / 'results.band.jsonl').exists()
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        reinfer.main_cli([str(run), '--verify', '--allow-mixed-border'])
    assert e.value.code == 1                                # p1 does not reproduce
    out = capsys.readouterr().out
    assert '"exact": 1' in out and '"mismatch": 1' in out
    assert 'border-band diagnostic' in out and '1 pano(s) differ ONLY there' in out
    summary, _, carry = reinfer.verify(run / 'results.jsonl', run / 'results.f01.jsonl')
    assert carry == {'p1'}
    assert summary['border_band'] == {'panos_only_in_band': 1, 'keys_old': 0, 'keys_new': 1}


def _mixed_file(tmp_path):
    path = tmp_path / 'results.jsonl'
    path.write_text(_line(pid='p') + _line(pid='q', border='keep'), encoding='utf-8')
    return path


def _consumer_fuse_load(tmp_path):
    import fuse_sites as fs
    fs.load_results(_mixed_file(tmp_path), read_heights=False)


def _consumer_fuse(tmp_path):
    import fuse_sites as fs
    panos, _ = fs.load_results(_mixed_file(tmp_path), read_heights=False,
                               allow_mixed_border=True)
    assert [p.border for p in panos] == ['exclude', 'keep']
    fs.fuse(panos, fs.FuseParams())


def _consumer_send_to_ps(tmp_path):
    import send_to_ps as sp
    sp.check_detection_border(_mixed_file(tmp_path))


def _consumer_reinfer_out(tmp_path):
    import reinfer
    reinfer.resolve_border({}, _mixed_file(tmp_path))


def _consumer_reinfer_verify(tmp_path):
    import reinfer
    run = _pair(tmp_path, [_rec('p'), _rec('q', 'keep')], [_rec('p'), _rec('q')])
    reinfer.main_cli([str(run), '--verify'])


MIX_CONSUMERS = {'fuse_sites.load_results': (_consumer_fuse_load, ValueError),
                 'fuse_sites.fuse': (_consumer_fuse, ValueError),
                 'send_to_ps.check_detection_border': (_consumer_send_to_ps, ValueError),
                 'reinfer.resolve_border': (_consumer_reinfer_out, SystemExit),
                 'reinfer --verify': (_consumer_reinfer_verify, SystemExit)}


@pytest.mark.parametrize('name', sorted(MIX_CONSUMERS))
def test_mix_is_refused(tmp_path, name):
    """Test 6: every consumer that combines records refuses an exclude/keep mix."""
    fn, exc = MIX_CONSUMERS[name]
    with pytest.raises(exc, match='mixes peak border rules'):
        fn(tmp_path)


@pytest.mark.parametrize('mode', ['--pose-ablation', '--implied-height', None])
def test_fuse_cli_honours_allow_mixed_border(tmp_path, mode):
    import fuse_sites as fs
    path = _mixed_file(tmp_path)
    args = [str(path), '--camera-height-m', '2.6'] + ([mode] if mode else [])
    with pytest.raises(SystemExit):
        fs.main(args)
    fs.main(args + ['--allow-mixed-border'])
    if mode is None:                            # a fuse records the mix in sites_meta.json
        text = (tmp_path / 'sites_meta.json').read_text(encoding='utf-8')
        assert '"exclude": 1' in text and '"keep": 1' in text


def test_sites_meta_records_the_border(tmp_path):
    import fuse_sites as fs
    path = tmp_path / 'results.jsonl'
    path.write_text(_line(pid='p', border='keep'), encoding='utf-8')
    fs.main([str(path), '--camera-height-m', '2.6'])
    text = (tmp_path / 'sites_meta.json').read_text(encoding='utf-8')
    assert '"detection_border": "keep"' in text


def test_bundle_scorers_refuse_a_keep_run(tmp_path, monkeypatch):
    """eval_sites (and every script that calls es.require_bundle_frame: agree_rate,
    site_explorer, gsv_partial_pose, mapillary_tilt) refuses a keep run."""
    import eval_sites as es
    import fuse_sites as fs
    keep = fs.SlimPano('p', 0.0, 0.0, 0.0, None, None, None, 'launch', [], border='keep')
    with pytest.raises(ValueError, match='exported from exclude runs'):
        es.require_bundle_frame([keep], 'runs/x')
    gauss = fs.SlimPano('p', 0.0, 0.0, 0.0, None, None, None, 'launch', [], decode='gaussian')
    with pytest.raises(ValueError, match='keyed to argmax'):
        es.require_bundle_frame([gauss], 'runs/x')
    es.require_bundle_frame([fs.SlimPano('q', 0.0, 0.0, 0.0, None, None, None, 'launch', [])],
                            'x')
    import gsv_partial_pose as gpp
    run = tmp_path / 'x'
    run.mkdir()
    (run / 'results.jsonl').write_text(_line(border='keep'), encoding='utf-8')
    with pytest.raises(SystemExit, match='exclude runs'):
        gpp.load_city('x', 2.6, tmp_path)


def test_provenance_gate_refuses_a_keep_arm(tmp_path):
    import provenance_gate as pg
    f = tmp_path / 'results.jsonl'
    f.write_text(json.dumps({'detections': [], 'detection_border': 'keep',
                             'pano': {'panorama_id': 'p', 'width': 16384,
                                      'height': 8192}}) + '\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='--border exclude'):
        pg.load_run(f)


def test_send_to_ps_border_guard_and_records(tmp_path, monkeypatch):
    """send_to_ps refuses a keep file beside an exclude campaign unless --allow-mixed-border,
    records the rule in *.submission.json, and leaves an exclude record byte-identical."""
    from test_decode_guards import ARGMAX_RECORD, PROD, _frozen_send, _results
    sp, sent = _frozen_send(monkeypatch)
    f = _results(tmp_path / 'results.jsonl', [None, None, None])
    sp.process_jsonl_file(str(f), PROD)
    assert (tmp_path / 'results.jsonl.submission.json').read_bytes() == ARGMAX_RECORD.encode()
    # a keep file beside that live exclude campaign: refused, then sent under the override
    k = tmp_path / 'results.keep.jsonl'
    k.write_bytes(b''.join(
        (json.dumps({**json.loads(line), 'detection_border': 'keep'}) + '\n').encode()
        for line in f.read_text().splitlines()))
    n = len(sent)
    with pytest.raises(ValueError, match='seam band'):
        sp.process_jsonl_file(str(k), PROD)
    assert len(sent) == n
    sp.process_jsonl_file(str(k), PROD, allow_mixed_border=True)
    record = json.loads((tmp_path / 'results.keep.jsonl.submission.json').read_text())
    assert record['detection_border'] == 'keep'
    assert 'whole-city frame change' in record['mixed_border_override']['reason']
    assert sp.recorded_border(record) == 'mixed'
    assert sp.recorded_border({}) == 'exclude'
    assert sp.recorded_border({'detection_border': {'exclude': 1, 'keep': 1}}) == 'mixed'


@pytest.mark.parametrize('script', ['main.py', 'scripts/reinfer.py'])
def test_help_shows_the_flag(script):
    """Test 8: --help imports cleanly (torch-free) and documents --border."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, str(REPO / script), '--help'], capture_output=True,
                         text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr
    assert '--border' in out.stdout and 'keep' in out.stdout
