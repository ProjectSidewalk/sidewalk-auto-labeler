"""The peak decode (issue #111): argmax unchanged, gaussian = RampNet#221's rule, and the
decode bound everywhere two runs could be combined. CPU only, no model, no network."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('skimage')

import detectors  # noqa: E402
from detectors import decode as dec  # noqa: E402
from detectors import rampnet_subcell as sc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / 'tests' / 'fixtures' / 'decode_heatmaps.npz'
FIXTURE_EXPECTED = REPO / 'tests' / 'fixtures' / 'decode_expected.json'


def legacy_detections_from_heatmap(heatmap):
    """detectors/curb_ramp.py::detections_from_heatmap as it was before #111, verbatim."""
    from skimage.feature import peak_local_max
    peaks = peak_local_max(np.clip(heatmap, 0, 1), min_distance=10,
                           threshold_abs=detectors.DETECTION_STORAGE_FLOOR,
                           num_peaks=detectors.MAX_PEAKS_PER_PANO)
    return [(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c]))
            for r, c in peaks]


def gaussian_map(peaks, shape=(64, 128), sigma=1.25):
    """A coarse map holding Gaussians at continuous coarse-cell centres (cy, cx, amp)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    c = np.zeros(shape)
    for cy, cx, amp in peaks:
        c += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return c


SYNTH = [(30.3, 60.8, 0.9), (34.45, 20.1, 0.62), (40.0, 100.4, 1.3),   # 1.3: clipped top
         (26.7, 90.6, 0.35), (45.2, 3.3, 0.5)]


def synth_heatmap():
    return sc.upsample(gaussian_map(SYNTH)).astype(np.float32)


def test_vendored_subcell_is_rampnets_file_verbatim():
    """detectors/rampnet_subcell.py must stay byte-identical to RampNet's rampnet/subcell.py at
    RAMPNET_SUBCELL_COMMIT. To update: copy the file verbatim and change both constants."""
    data = (REPO / 'detectors' / 'rampnet_subcell.py').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(data).hexdigest() == dec.RAMPNET_SUBCELL_SHA256


def test_argmax_is_the_legacy_extractor_on_synthetic_maps():
    h = synth_heatmap()
    assert dec.detections_from_heatmap(h) == legacy_detections_from_heatmap(h)
    assert dec.detections_from_heatmap(h, 'argmax') == legacy_detections_from_heatmap(h)


@pytest.mark.skipif(not FIXTURE.exists(), reason='stored heatmap fixture not present')
def test_argmax_is_the_legacy_extractor_on_stored_heatmaps():
    """Pinned on real RampNet heatmaps (coarse maps recovered from A40 forward passes on
    benchmark panos, rebuilt by the exact bilinear operator): argmax output equals the
    pre-#111 extractor, and both equal the outputs committed beside the fixture."""
    maps = np.load(FIXTURE)
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding='utf-8'))
    assert sorted(maps.files) == sorted(expected)
    for pid in maps.files:
        h = sc.upsample(maps[pid].astype(np.float64)).astype(np.float32)
        got = dec.detections_from_heatmap(h, 'argmax')
        assert got == legacy_detections_from_heatmap(h)          # bit-identical, any platform
        # against the committed outputs: pixels exact, values to float noise (BLAS differs
        # by platform in the last bits of the rebuilt heatmap)
        px = lambda ds: [(round(d[0] * 1024), round(d[1] * 512)) for d in ds]  # noqa: E731
        assert px(got) == px(expected[pid]['argmax'])
        g = dec.detections_from_heatmap(h, 'gaussian')
        assert len(g) == len(expected[pid]['gaussian'])
        for d, e in zip(g + got, expected[pid]['gaussian'] + expected[pid]['argmax']):
            assert d == pytest.approx(e, abs=1e-6)


def test_gaussian_keeps_peaks_scores_and_order():
    h = synth_heatmap()
    a = dec.detections_from_heatmap(h, 'argmax')
    g = dec.detections_from_heatmap(h, 'gaussian')
    assert len(a) == len(g) >= 4
    assert [d[2] for d in a] == [d[2] for d in g]
    for (ax, ay, _), (gx, gy, _) in zip(a, g):
        # a refined peak stays within its coarse cell's reach: half a cell + the 0.5-px lean,
        # plus at most one cell when a diagonal neighbour pulled the argmax across (climb)
        assert abs(gx - ax) * 1024 <= 12.5 and abs(gy - ay) * 512 <= 12.5


def test_gaussian_recovers_subcell_positions():
    h = synth_heatmap()
    got = dec.detections_from_heatmap(h, 'gaussian')
    for cy, cx, _amp in SYNTH[:2]:
        want = ((8 * cx + 3.5) / 1024, (8 * cy + 3.5) / 512)
        best = min(got, key=lambda d: (d[0] - want[0]) ** 2 + (d[1] - want[1]) ** 2)
        assert abs(best[0] - want[0]) * 1024 < 0.05 and abs(best[1] - want[1]) * 512 < 0.05


def test_gaussian_equals_rampnet_detect_peaks():
    """The labeler's gaussian positions are RampNet's detect_peaks positions for the same
    peaks (the labeler keeps its own peak finder: floor, top 50, exclude_border)."""
    h = synth_heatmap()
    mine = dec.detections_from_heatmap(h, 'gaussian')
    rn = sc.detect_peaks(h, detectors.DETECTION_STORAGE_FLOOR, decode='gaussian', clip=True,
                         exclude_border=True)
    rn_xy = {(round(c / 1024, 12), round(r / 512, 12)) for r, c, _ in rn}
    for x, y, _ in mine:
        assert (round(x, 12), round(y, 12)) in rn_xy


def test_gaussian_refuses_a_clipped_heatmap():
    h = np.clip(synth_heatmap(), 0, 1)      # the 1.3 peak is flattened: not an upsample
    with pytest.raises(ValueError, match='not an exact'):
        dec.detections_from_heatmap(h, 'gaussian')
    assert dec.detections_from_heatmap(h, 'argmax')       # argmax never needs the coarse map


def test_unknown_decode_and_empty_heatmap():
    with pytest.raises(ValueError):
        dec.detections_from_heatmap(synth_heatmap(), 'dark')
    assert dec.detections_from_heatmap(np.zeros((512, 1024), np.float32), 'gaussian') == []


def test_detections_both_is_one_heatmap_two_decodes():
    h = synth_heatmap()
    both = dec.detections_both(h)
    assert both['argmax'] == dec.detections_from_heatmap(h)
    assert both['gaussian'] == dec.detections_from_heatmap(h, 'gaussian')


# --- the record marker and the file-level guards --------------------------------------

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


# --- the committed measurement re-derives from the committed data ----------------------

def _rederive(tmp_path, argv, names):
    import subcell_decode as sd
    sd.main(argv + ['--out', str(tmp_path)])
    for n in names:
        assert (tmp_path / n).read_bytes() == (sd.DATA_DIR / n).read_bytes(), n


def test_stability_csvs_rederive(tmp_path):
    _rederive(tmp_path, ['stability', 'annapolis', 'paterson', 'richmond', 'sao_paulo'],
              ['decode_stability.csv', 'decode_stability_hist.csv'])


RAMPNET = REPO.parent / 'RampNet'


@pytest.mark.skipif(not (RAMPNET / 'manual_labels').exists() or
                    not (RAMPNET / 'benchmark' / 'paterson' / 'boxes.json').exists(),
                    reason='needs a RampNet checkout beside this repo (benchmark GT)')
def test_residual_csvs_rederive(tmp_path):
    _rederive(tmp_path, ['residual', '--rampnet-root', str(RAMPNET)],
              ['decode_residual.csv', 'decode_sigma.csv', 'decode_agreement.csv',
               'decode_inputs.csv'])
