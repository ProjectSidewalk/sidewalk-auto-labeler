"""The peak decode (issue #111): argmax unchanged, gaussian = RampNet#221's rule, and the
decode bound everywhere two runs could be combined. CPU only, no model, no network."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import importlib.util
import os

import numpy as np
import pytest

import detectors  # noqa: E402
from detectors import decode as dec  # noqa: E402
from detectors import rampnet_subcell as sc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# Only the peak finder needs scikit-image (requirements-test.txt carries it); the guards
# that bind a decode are in tests/test_decode_guards.py and need nothing beyond the stdlib.
needs_skimage = pytest.mark.skipif(importlib.util.find_spec('skimage') is None,
                                   reason='needs scikit-image (requirements-test.txt)')
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


@needs_skimage
def test_argmax_is_the_legacy_extractor_on_synthetic_maps():
    h = synth_heatmap()
    assert dec.detections_from_heatmap(h) == legacy_detections_from_heatmap(h)
    assert dec.detections_from_heatmap(h, 'argmax') == legacy_detections_from_heatmap(h)


@pytest.mark.skipif(not FIXTURE.exists(), reason='stored heatmap fixture not present')
@needs_skimage
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


@needs_skimage
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


@needs_skimage
def test_gaussian_recovers_subcell_positions():
    h = synth_heatmap()
    got = dec.detections_from_heatmap(h, 'gaussian')
    for cy, cx, _amp in SYNTH[:2]:
        want = ((8 * cx + 3.5) / 1024, (8 * cy + 3.5) / 512)
        best = min(got, key=lambda d: (d[0] - want[0]) ** 2 + (d[1] - want[1]) ** 2)
        assert abs(best[0] - want[0]) * 1024 < 0.05 and abs(best[1] - want[1]) * 512 < 0.05


@needs_skimage
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


@needs_skimage
def test_gaussian_refuses_a_clipped_heatmap():
    h = np.clip(synth_heatmap(), 0, 1)      # the 1.3 peak is flattened: not an upsample
    with pytest.raises(ValueError, match='not an exact'):
        dec.detections_from_heatmap(h, 'gaussian')
    assert dec.detections_from_heatmap(h, 'argmax')       # argmax never needs the coarse map


@needs_skimage
def test_unknown_decode_and_empty_heatmap():
    with pytest.raises(ValueError):
        dec.detections_from_heatmap(synth_heatmap(), 'dark')
    assert dec.detections_from_heatmap(np.zeros((512, 1024), np.float32), 'gaussian') == []


@needs_skimage
def test_detections_both_is_one_heatmap_two_decodes():
    h = synth_heatmap()
    both = dec.detections_both(h)
    assert both['argmax'] == dec.detections_from_heatmap(h)
    assert both['gaussian'] == dec.detections_from_heatmap(h, 'gaussian')


# --- the committed measurement re-derives from the committed data ----------------------

def _rederive(tmp_path, argv, names):
    import subcell_decode as sd
    sd.main(argv + ['--out', str(tmp_path)])
    for n in names:
        assert (tmp_path / n).read_bytes() == (sd.DATA_DIR / n).read_bytes(), n


def test_stability_csvs_rederive(tmp_path):
    _rederive(tmp_path, ['stability', 'annapolis', 'paterson', 'richmond', 'sao_paulo'],
              ['decode_stability.csv', 'decode_stability_hist.csv'])


RAMPNET = Path(os.environ.get('RAMPNET_ROOT', REPO.parent / 'RampNet'))


def _rampnet_matches():
    """(ok, why): does RAMPNET hold, byte for byte, the RampNet inputs the committed residual
    CSVs were made from (decode_inputs.csv's `rampnet:` rows, RampNet main 459ea9e)?"""
    import csv
    import subcell_decode as sd
    if not RAMPNET.exists():
        return False, f'no RampNet checkout at {RAMPNET}'
    with open(sd.DATA_DIR / 'decode_inputs.csv', encoding='utf-8') as f:
        want = {r['input'][len('rampnet:'):]: r['sha256'] for r in csv.DictReader(f)
                if r['input'].startswith('rampnet:')}
    got = dict(sd.rampnet_inputs(RAMPNET))
    bad = sorted(k for k in want if got.get(k) != want[k])
    return (not bad, f'{RAMPNET} differs from RampNet main 459ea9e in {", ".join(bad)}; set '
                     'RAMPNET_ROOT to a RampNet checkout whose files match main 459ea9e to run it')


@needs_skimage
def test_residual_csvs_rederive(tmp_path):
    ok, why = _rampnet_matches()
    if not ok:
        pytest.skip(why)
    _rederive(tmp_path, ['residual', '--rampnet-root', str(RAMPNET)],
              ['decode_residual.csv', 'decode_sigma.csv', 'decode_agreement.csv',
               'decode_inputs.csv'])
