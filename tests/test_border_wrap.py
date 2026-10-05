"""The `wrap` peak border rule (#130 follow-up): `keep` plus non-maximum suppression wrapped
across the 360-degree seam, this repo's own rule. Pins that `exclude` and `keep` are
byte-identical to origin/main (tests/fixtures/border_expected.json, generated before the
change), what `wrap` does at the seam, and that the opt-in is bound and guarded like `keep`.
CPU only, no model, no network; the peak finder needs scikit-image."""
import hashlib
import importlib.util
import json
from collections import Counter
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
BORDER_EXPECTED = REPO / 'tests' / 'fixtures' / 'border_expected.json'
W = 1024
D = dec.MIN_DISTANCE

# Copied from tests/test_border.py (no cross-test imports beyond the sanctioned one below).
# sha256 of the line main.py wrote before #130 for conftest's fake result.
PRE_130_LINE_SHA256 = '2d9d23b21c31664edbd674971892897de808eed34001675f778894fbcf20d8e1'
INTERIOR = (30, 60, 0.9)
LEFT, RIGHT, TOP = (20, 0, 0.8), (45, 127, 0.7), (0, 90, 0.6)


def coarse_map(peaks, shape=(64, 128), sigma=1.25):
    """A 64x128 coarse map holding Gaussians at coarse-cell centres (cy, cx, amp)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    c = np.zeros(shape)
    for cy, cx, amp in peaks:
        c += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return c


def hm(peaks):
    return sc.upsample(coarse_map(peaks)).astype(np.float32)


def edge_heatmap():
    return hm([INTERIOR, LEFT, RIGHT, TOP])


def straddle_heatmap(a=0.8, b=0.75):
    """One ramp across the seam (coarse columns 0 and 127, same row) plus an interior peak."""
    return hm([(25, 0, a), (25, 127, b), INTERIOR])


def px(dets):
    return [(round(x * W), round(y * 512)) for x, y, _ in dets]


def all_maps():
    """Every heatmap the fixture pins, by the fixture's names."""
    maps = {}
    if FIXTURE.exists():
        fx = np.load(FIXTURE)
        for pid in fx.files:
            maps[pid] = sc.upsample(fx[pid].astype(np.float64)).astype(np.float32)
    maps['synthetic:edge'] = edge_heatmap()
    maps['synthetic:straddle'] = straddle_heatmap()
    return maps


def interior(dets, rows=False):
    """Peaks off the seam band (columns 10-1013); with ``rows``, off the zenith/nadir band
    too (the peaks `exclude` keeps)."""
    return [d for d in dets if D <= round(d[0] * W) < W - D
            and (not rows or D <= round(d[1] * 512) < 512 - D)]


def has_wrapped_pair(dets):
    """seam_band_130.straddle_flags' test: two peaks within wrapped Chebyshev D."""
    keys = px(dets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dx = abs(keys[i][0] - keys[j][0]) % W
            if min(dx, W - dx) <= D and abs(keys[i][1] - keys[j][1]) <= D:
                return True
    return False


# --- 1. byte identity: the test that protects live labels -----------------------------------

@needs_skimage
@pytest.mark.parametrize('border', ['exclude', 'keep'])
@pytest.mark.parametrize('decode', ['argmax', 'gaussian'])
def test_exclude_and_keep_are_byte_identical_to_origin_main(border, decode):
    expected = json.loads(BORDER_EXPECTED.read_text(encoding='utf-8'))
    maps = all_maps()
    assert set(expected) <= set(maps) or not FIXTURE.exists()
    for name, want in expected.items():
        if name not in maps:
            continue
        got = dec.detections_from_heatmap(maps[name], decode, border=border)
        want = [tuple(t) for t in want[border][decode]]
        if decode == 'argmax':
            assert got == want, name                         # every bit
        else:
            assert len(got) == len(want), name
            for g, w in zip(got, want):
                assert g[2] == w[2], name                    # scores: every bit
                assert g[:2] == pytest.approx(w[:2], abs=1e-9), name


# --- 2-3. the seam ---------------------------------------------------------------------------

@needs_skimage
def test_straddling_pair_collapses_to_the_stronger_half():
    h = straddle_heatmap(0.8, 0.75)
    kp = dec.detections_from_heatmap(h, 'argmax', border='keep')
    wr = dec.detections_from_heatmap(h, 'argmax', border='wrap')
    assert len(kp) == 3 and len(wr) == 2
    cols = [c for c, _ in px(wr)]
    assert wr[0] == kp[0]                                     # the interior peak, untouched
    assert cols[1] < D and wr[1][2] == pytest.approx(0.8, abs=0.04)
    assert wr[1] in kp
    # amplitudes swapped: the survivor is the right-edge half
    h2 = straddle_heatmap(0.75, 0.8)
    wr2 = dec.detections_from_heatmap(h2, 'argmax', border='wrap')
    assert len(wr2) == 2 and px(wr2)[1][0] >= W - D
    assert wr2[1] in dec.detections_from_heatmap(h2, 'argmax', border='keep')


@needs_skimage
def test_pair_just_outside_the_window_keeps_both_halves():
    """Coarse columns 0 and 126: the peaks sit at hi-res 0 and 1011, wrapped |dx| = 13 > 10."""
    h = hm([(25, 0, 0.8), (25, 126, 0.75), INTERIOR])
    kp = dec.detections_from_heatmap(h, 'argmax', border='keep')
    wr = dec.detections_from_heatmap(h, 'argmax', border='wrap')
    assert len(kp) == 3 and wr == kp


# --- 4-5. what wrap leaves alone, and the set relation ---------------------------------------

@needs_skimage
def test_interior_peaks_are_identical_under_all_three_rules():
    for name, h in all_maps().items():
        for decode in ('argmax', 'gaussian'):
            ex, kp, wr = (dec.detections_from_heatmap(h, decode, border=b)
                          for b in ('exclude', 'keep', 'wrap'))
            assert interior(wr) == interior(kp), (name, decode)
            assert interior(wr, rows=True) == interior(ex, rows=True), (name, decode)


@needs_skimage
def test_exclude_subset_wrap_subset_keep():
    outcomes = {}
    for name, h in all_maps().items():
        ex, kp, wr = (set(px(dec.detections_from_heatmap(h, 'argmax', border=b)))
                      for b in ('exclude', 'keep', 'wrap'))
        assert ex <= wr <= kp, name
        keep_dets = dec.detections_from_heatmap(h, 'argmax', border='keep')
        outcomes[name] = ('pair' if has_wrapped_pair(keep_dets) else 'no pair', wr == kp)
    # No pair across the seam -> nothing to suppress; with one, wrap drops exactly one half.
    for name, (pair, equal) in outcomes.items():
        assert equal == (pair == 'no pair'), outcomes
    assert outcomes['synthetic:straddle'] == ('pair', False)


# --- 6-7. ties and the cap -------------------------------------------------------------------

@needs_skimage
def test_equal_plateau_across_the_seam_gives_one_peak():
    """Coarse columns 0 and 127 exactly equal in one row: keep returns both; wrap one, and the
    survivor is the right-edge half (its left-pad copy comes first in the padded frame's
    row-major order -- decode._peaks' docstring)."""
    c = np.zeros((64, 128))
    c[25, 0] = c[25, 127] = 0.9
    c[40, 60] = 0.5
    h = sc.upsample(c).astype(np.float32)
    kp = dec.detections_from_heatmap(h, 'argmax', border='keep')
    wr = dec.detections_from_heatmap(h, 'argmax', border='wrap')
    assert len([k for k in px(kp) if k[1] == 203]) == 2
    seam = [k for k in px(wr) if k[1] == 203]
    assert seam == [(1020, 203)]
    assert len(wr) == 2


@needs_skimage
def test_cap_counts_real_peaks_only_and_keeps_the_highest():
    rng = np.random.default_rng(130)
    peaks = [(25, 0, 0.99), (25, 127, 0.98)]                  # a straddling pair at the top
    cells = [(r, c) for r in range(4, 62, 4) for c in range(6, 122, 4)]
    for k in rng.choice(len(cells), 58, replace=False):
        r, c = cells[k]
        peaks.append((r, c, float(rng.uniform(0.2, 0.95))))
    h = hm(peaks)
    from skimage.feature import peak_local_max
    raw = peak_local_max(np.clip(h, 0, 1), min_distance=D, exclude_border=False,
                         threshold_abs=detectors.DETECTION_STORAGE_FLOOR)
    assert len(raw) >= 55, len(raw)                           # the cap really binds
    wr = dec.detections_from_heatmap(h, 'argmax', border='wrap')
    assert len(wr) == detectors.MAX_PEAKS_PER_PANO
    scores = [s for _, _, s in wr]
    assert scores == sorted(scores, reverse=True)
    assert len(set(px(wr))) == len(wr)
    # the 50 highest of keep's uncapped peaks, minus the suppressed (weaker, 0.98) seam half
    keep_sorted = sorted(((float(h[r][c]), (int(c), int(r))) for r, c in raw), reverse=True)
    weaker = (1020, 203)
    assert weaker in [k for _, k in keep_sorted]
    want = [k for _, k in keep_sorted if k != weaker][:detectors.MAX_PEAKS_PER_PANO]
    assert px(wr) == want


# --- 8. the gaussian decode under wrap -------------------------------------------------------

@needs_skimage
def test_gaussian_under_wrap():
    for h in (edge_heatmap(), straddle_heatmap(), straddle_heatmap(0.75, 0.8)):
        a = dec.detections_from_heatmap(h, 'argmax', border='wrap')
        g = dec.detections_from_heatmap(h, 'gaussian', border='wrap')
        assert [s for _, _, s in g] == [s for _, _, s in a]   # same peaks, same order
        assert dec.detections_both(h, border='wrap') == {'argmax': a, 'gaussian': g}
        for (ax, ay, _), (gx, gy, _) in zip(a, g):
            assert 0.0 <= gx < 1.0 and np.isfinite(gx) and np.isfinite(gy)
            if round(ax * W) < D or round(ax * W) >= W - D:  # a seam peak: within half a
                d = abs(gx - (8 * (round(ax * W) // 8) + 3.5) / W)   # coarse cell, mod 1
                assert min(d, 1 - d) <= 4 / W + 1e-12
    assert len(dec.detections_from_heatmap(straddle_heatmap(), 'gaussian', border='wrap')) == 2
    # wrap_x=True gives a seam peak a sub-cell x offset, where keep (RampNet) gives none
    g_keep = dec.detections_from_heatmap(straddle_heatmap(), 'gaussian', border='keep')
    g_wrap = dec.detections_from_heatmap(straddle_heatmap(), 'gaussian', border='wrap')
    assert g_keep[1][0] == pytest.approx(3.5 / W, abs=1e-12)
    assert g_wrap[1][0] != pytest.approx(3.5 / W, abs=1e-6)


@needs_skimage
def test_unknown_border_still_refused():
    with pytest.raises(ValueError, match='unknown border'):
        dec.detections_from_heatmap(edge_heatmap(), border='mirror')


# --- 9. the contract: bound and guarded like keep (no scikit-image needed) --------------------

def _rec(pid='p1', border=None):
    rec = {'detections': [{'x_normalized': 0.5, 'y_normalized': 0.6, 'confidence': 0.7}],
           'pano': {'panorama_id': pid, 'width': 16384, 'height': 8192, 'lat': 47.6,
                    'lng': -122.3, 'camera_heading': 0.0, 'source': 'launch'}}
    if border is not None:
        rec['detection_border'] = border
    return rec


def _line(**kw):
    return json.dumps(_rec(**kw)) + '\n'


def test_record_and_single_border():
    assert detectors.BORDER_WRAP == 'wrap' and 'wrap' in detectors.BORDERS
    assert detectors.DEFAULT_BORDER == 'exclude'
    assert detectors.record_border({'detection_border': 'wrap'}) == 'wrap'
    assert detectors.single_border(Counter({'wrap': 2}), 'f') == 'wrap'
    for mix in ({'keep': 1, 'wrap': 1}, {'exclude': 1, 'wrap': 1}):
        with pytest.raises(ValueError, match='mixes peak border rules'):
            detectors.single_border(Counter(mix), 'f')


def test_build_output_line_marks_wrap():
    import main
    from conftest import make_process_result, make_provenance
    fake, prov = make_process_result(), make_provenance()
    plain = main.build_output_line(fake, prov)
    assert hashlib.sha256(json.dumps(plain).encode()).hexdigest() == PRE_130_LINE_SHA256
    wr = main.build_output_line({**fake, 'border': 'wrap'}, prov)
    assert wr['detection_border'] == 'wrap'
    assert {k: v for k, v in wr.items() if k != 'detection_border'} == plain


def test_bind_border_wrap(tmp_path):
    import main
    run = tmp_path / 'run'
    run.mkdir()
    m = {}
    assert main.bind_border(m, 'wrap', run) is True and m['detection_border'] == 'wrap'
    assert main.bind_border(m, 'wrap', run) is False
    with pytest.raises(SystemExit, match='--border wrap'):
        main.bind_border(m, 'keep', run)
    legacy = {}
    (run / 'results.jsonl').write_text(_line(), encoding='utf-8')
    with pytest.raises(SystemExit, match='--border exclude'):
        main.bind_border(legacy, 'wrap', run)


def test_reinfer_resolve_border_wrap(tmp_path):
    import reinfer
    out = tmp_path / 'f01.jsonl'
    assert reinfer.resolve_border({'detection_border': 'wrap'}, out) == 'wrap'
    out.write_text(_line(border='wrap'), encoding='utf-8')
    with pytest.raises(SystemExit, match='#130'):
        reinfer.resolve_border({}, out)                       # exclude into a wrap file
    with pytest.raises(SystemExit, match='#130'):
        reinfer.resolve_border({'detection_border': 'keep'}, out)


def test_send_to_ps_refuses_wrap_beside_exclude(tmp_path, monkeypatch):
    from test_decode_guards import PROD, _frozen_send, _results
    sp, sent = _frozen_send(monkeypatch)
    f = _results(tmp_path / 'results.jsonl', [None, None, None])
    sp.process_jsonl_file(str(f), PROD)
    w = tmp_path / 'results.wrap.jsonl'
    w.write_bytes(b''.join(
        (json.dumps({**json.loads(line), 'detection_border': 'wrap'}) + '\n').encode()
        for line in f.read_text().splitlines()))
    n = len(sent)
    with pytest.raises(ValueError, match='seam band'):
        sp.process_jsonl_file(str(w), PROD)
    assert len(sent) == n
    sp.process_jsonl_file(str(w), PROD, allow_mixed_border=True)
    record = json.loads((tmp_path / 'results.wrap.jsonl.submission.json').read_text())
    assert record['detection_border'] == 'wrap'


def test_bundle_scorers_and_gate_refuse_wrap(tmp_path):
    import eval_sites as es
    import fuse_sites as fs
    import provenance_gate as pg
    wr = fs.SlimPano('p', 0.0, 0.0, 0.0, None, None, None, 'launch', [], border='wrap')
    with pytest.raises(ValueError, match='exported from exclude runs'):
        es.require_bundle_frame([wr], 'runs/x')
    f = tmp_path / 'results.jsonl'
    f.write_text(json.dumps({'detections': [], 'detection_border': 'wrap',
                             'pano': {'panorama_id': 'p', 'width': 16384,
                                      'height': 8192}}) + '\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='--border exclude'):
        pg.load_run(f)


def test_fuse_refuses_a_keep_wrap_mix(tmp_path):
    import fuse_sites as fs
    path = tmp_path / 'results.jsonl'
    path.write_text(_line(pid='p', border='keep') + _line(pid='q', border='wrap'),
                    encoding='utf-8')
    with pytest.raises(ValueError, match='mixes peak border rules'):
        fs.load_results(path, read_heights=False)


@pytest.mark.parametrize('script', ['main.py', 'scripts/reinfer.py'])
def test_help_names_wrap(script):
    import subprocess
    import sys
    out = subprocess.run([sys.executable, str(REPO / script), '--help'], capture_output=True,
                         text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr
    assert 'wrap' in out.stdout


# --- 10. the offline estimate in seam_band_130.summary ---------------------------------------

def test_wrap_estimate_arithmetic():
    import seam_band_130 as sb
    tiers = {'0.3': {'gained': 15, 'keep_peaks': 1745, 'straddle_pairs': 3}}
    est = sb.wrap_tiers(tiers)['0.3']
    assert est['gained_under_wrap'] == 12 and est['wrap_peaks'] == 1742
    assert est['gained_share_of_wrap']['k'] == 12 and est['gained_share_of_wrap']['n'] == 1742

    def row(pid, score, cls):
        return {'pano_id': pid, 'score': str(score), 'world_class': cls, 'straddle_pair': '1',
                'edge': 'right'}
    rows = [row('a', 0.708, 'view'), row('a', 0.666, 'split'),       # the duplicate
            row('b', 0.49, 'not_projected'), row('b', 0.40, 'not_projected'),
            row('c', 0.50, 'promoted'),                                  # partner < 0.30
            row('d', 0.6, 'split'), row('d', 0.5, 'view'),               # weaker joined a view
            {**row('e', 0.4, 'new'), 'straddle_pair': '0'}]
    w = sb.wrap_world(rows, 333)
    assert w == {'straddle_halves_fused_as_second_site': 1, 'straddle_rows_not_projected': 2,
                 'operational_sites_keep': 333, 'operational_sites_wrap_estimate': 332}


def test_committed_wrap_estimate():
    """The numbers the PR and docs/seam-band-130.md section 9 quote."""
    s = json.loads((REPO / 'docs' / 'figures' / 'seam-band-130' / 'data' / 'summary.json')
                   .read_text(encoding='utf-8'))
    m = s['arms']['laurens']['wrap_estimate']
    assert [m[t]['gained_under_wrap'] for t in ('0.1', '0.3', '0.55')] == [89, 12, 2]
    assert m['world']['operational_sites_wrap_estimate'] == 332
    g = s['arms']['laurens_gsv_archive']['wrap_estimate']
    assert [g[t]['gained_under_wrap'] for t in ('0.1', '0.3', '0.55')] == [9, 2, 1]
