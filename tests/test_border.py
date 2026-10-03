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
