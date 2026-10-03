"""scripts/seam_band_130.py (issue #130): the instrument check, the exclude/keep peak diff,
the world classifier and the GT match on synthetic data, and `verify` against the committed
results. CPU only, no network; the peak steps need scikit-image."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import seam_band_130 as sb
from detectors import decode as dec
from detectors import rampnet_subcell as sc

REPO = Path(__file__).resolve().parents[1]
needs_skimage = pytest.mark.skipif(importlib.util.find_spec('skimage') is None,
                                   reason='needs scikit-image (requirements-test.txt)')


def coarse(peaks, shape=(64, 128), sigma=1.25):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    c = np.zeros(shape)
    for cy, cx, amp in peaks:
        c += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return c.astype(np.float32)


# p_seam: one interior peak, one at the left seam. p_top: interior + zenith row.
# p_pair: a seam-straddling pair (columns 0 and 127, same row). p_bad: stored detections that
# the heatmap does not reproduce.
MAPS = {'p_seam': [(30, 60, 0.9), (25, 0, 0.62)],
        'p_top': [(32, 40, 0.7), (0, 90, 0.4)],
        'p_pair': [(28, 0, 0.8), (28, 127, 0.75), (40, 70, 0.35)],
        'p_bad': [(30, 60, 0.9)]}


def make_run(tmp_path):
    cdir = tmp_path / 'coarse'
    cdir.mkdir()
    run = tmp_path / 'runs' / 'laurens'
    run.mkdir(parents=True)
    lines = []
    for pid, pk in MAPS.items():
        c = coarse(pk)
        np.save(cdir / f'{pid}_coarse.npy', c)
        h = sb.heatmap_from_coarse(c)
        dets = dec.detections_from_heatmap(h)           # what production stored
        if pid == 'p_bad':
            dets = [(0.25, 0.5, 0.5)]
        lines.append(json.dumps({
            'detections': [{'x_normalized': x, 'y_normalized': y, 'confidence': s}
                           for x, y, s in dets],
            'pano': {'panorama_id': pid, 'lat': 42.85, 'lng': -94.85, 'camera_heading': 0.0,
                     'width': 5760, 'height': 2880, 'source': 'mapillary'}}) + '\n')
    (run / 'results.jsonl').write_text(''.join(lines), encoding='utf-8')
    (run / 'manifest.json').write_text('{"detection_storage_floor": 0.1}', encoding='utf-8')
    return run, cdir


@needs_skimage
def test_check_peaks_summary_on_synthetic_maps(tmp_path):
    run, cdir = make_run(tmp_path)
    data = tmp_path / 'data'
    rep = sb.main(['check', 'laurens', '--results', str(run / 'results.jsonl'),
                   '--coarse-dir', str(cdir), '--data-dir', str(data)])
    assert rep['panos'] == 4 and rep['reproduce'] == 3 and not rep['passes_gate']
    rows = {r['pano_id']: r for r in sb.read_csv(data / 'laurens_check.csv')}
    assert rows['p_bad']['reproduces'] == '0' and rows['p_seam']['reproduces'] == '1'

    counts = sb.main(['peaks', 'laurens', '--results', str(run / 'results.jsonl'),
                      '--coarse-dir', str(cdir), '--data-dir', str(data)])
    assert counts['panos_counted'] == 3 and counts['lost_peaks'] == 0
    gained = [r for r in sb.read_csv(data / 'laurens_gained.csv') if r['decode'] == 'argmax']
    edges = sorted((r['pano_id'], r['edge']) for r in gained)
    assert edges == [('p_pair', 'left'), ('p_pair', 'right'), ('p_seam', 'left'),
                     ('p_top', 'top')]
    assert {r['pano_id'] for r in gained if r['straddle_pair'] == '1'} == {'p_pair'}
    assert len(counts['straddle_pairs']) == 1
    # both decodes recorded, same peaks
    assert len(sb.read_csv(data / 'laurens_gained.csv')) == 2 * len(gained)

    s = sb.build_summary(data, arms=('laurens',), pool=('laurens',))
    t = s['arms']['laurens']['tiers']['0.3']
    assert t['seam_gained'] == 3 and t['top_bottom_gained'] == 1 and t['gained'] == 4
    assert t['panos_with_seam_gain'] == 2 and t['straddle_pairs'] == 1
    assert t['keep_peaks'] - t['exclude_peaks'] == t['gained']
    assert s['arms']['laurens']['tiers']['0.55']['seam_gained'] == 3
    assert s['arms']['laurens']['geometric_expectation'] == pytest.approx(20 / 1024, abs=1e-6)

    # the archive frame counts every pano, including the one the run does not reproduce
    arch = sb.main(['peaks', 'laurens', '--results', str(run / 'results.jsonl'),
                    '--coarse-dir', str(cdir), '--data-dir', str(data), '--frame', 'archive'])
    assert arch['panos_counted'] == 4 and arch['frame'] == 'archive'
    assert (data / 'laurens_archive_gained.csv').exists()
    assert (run / 'seam_band_130' / 'peaks.archive.jsonl').exists()


def _site(sid, members, n_operational):
    return SimpleNamespace(id=sid, n_operational=n_operational,
                           members=[(SimpleNamespace(pano_id=p, det_index=i), True)
                                    for p, i in members])


def test_classify_view_promoted_new():
    """A gained detection that joins a site the exclude run had as operational is a lost
    VIEW; one whose stored partners were only sub-threshold support is a PROMOTION; one with
    no stored partner seeded a NEW site; one that did not fuse is absent."""
    ex = [_site(0, [('a', 0), ('b', 0)], 1),            # operational in exclude
          _site(1, [('c', 0)], 0)]                      # support only in exclude
    kp = [_site(0, [('a', 0), ('b', 0), ('g1', 1)], 2),
          _site(1, [('c', 0), ('g2', 1)], 1),
          _site(2, [('g3', 1), ('g4', 1)], 2)]
    got = sb.classify([('g1', 1), ('g2', 1), ('g3', 1), ('g4', 1), ('g5', 1)], ex, kp)
    assert got == {('g1', 1): 'view', ('g2', 1): 'promoted', ('g3', 1): 'new',
                   ('g4', 1): 'new'}


def test_gt_verdicts_wrap_the_seam():
    entry = {'dets': [True], 'missed': [{'x': 0.999, 'y': 0.55}], 'no_missed': False}
    bundle = [(0.5, 0.6, 0.9)]
    got = sb.gt_verdicts([(0.001, 0.55), (0.5, 0.601), (0.3, 0.6)], entry, bundle)
    assert got == ['tp', 'dup', 'fp']        # 0.001 matches 0.999 across the seam
    assert sb.gt_verdicts([(0.3, 0.6)], {'dets': [], 'missed': [], 'no_missed': False},
                          []) == ['unjudged']


def test_straddle_flags():
    dets = [(0 / 1024, 0.5, 0.8), (1014 / 1024, 0.5 + 3 / 512, 0.7), (500 / 1024, 0.5, 0.6)]
    flags, pairs = sb.straddle_flags(dets)
    assert flags == [True, True, False] and pairs == [(0, 1)]
    assert sb.straddle_flags([(0.0, 0.5, 1), (1013 / 1024, 0.5, 1)])[1] == []   # 11 px


@pytest.mark.skipif(not sb.SUMMARY.exists(), reason='no committed summary.json yet')
def test_committed_summary_rederives():
    """Step e: summary.json re-derives byte for byte from the committed CSV/JSON files."""
    sb.main(['verify'])
