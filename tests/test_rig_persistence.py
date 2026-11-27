"""Pure-logic tests for scripts/rig_persistence.py (no data files, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import rig_persistence as rp  # noqa: E402


def _m(group, col, row, seq='s', conf=0.4, idx=0):
    return rp.Member(seq, group, float(col), float(row), conf, f'p{group}', idx)


def test_support_counts_distinct_frames_and_seam_wrap():
    # two detections in frame 1 count once; frame 2 sits across the seam (col 1020 vs 2)
    ms = [_m(0, 2, 400), _m(1, 4, 402), _m(1, 5, 401, idx=1), _m(2, 1020, 400)]
    assert rp.support_counts(ms[:1], ms, 8) == [2]
    # outside the tolerance in y
    assert rp.support_counts([_m(0, 2, 420)], ms, 8) == [0]


def test_support_counts_other_sequence_and_window():
    ms = [_m(g, 500, 400) for g in range(30)] + [_m(5, 500, 400, seq='other')]
    t = _m(0, 500, 400)
    assert rp.support_counts([t], ms, 8) == [29]
    assert rp.support_counts([t], ms, 8, window=10) == [10]


def test_flags_from_k_and_fraction():
    ts = [_m(0, 0, 0), _m(1, 0, 0)]
    n_frames = {'s': 20}
    assert rp.flags_from(ts, [2, 1], n_frames, K=3, F=0.0) == [True, False]
    assert rp.flags_from(ts, [2, 1], n_frames, K=3, F=0.2) == [False, False]


def test_frame_groups_collapses_stationary_frames():
    P = rp.RunPano
    panos = {
        'a': P('a', 's', 0.0, -1, 47.0, -122.0, []),
        'b': P('b', 's', 1.0, -1, 47.000005, -122.0, []),   # ~0.6 m: stationary
        'c': P('c', 's', 2.0, -1, 47.0001, -122.0, []),     # ~11 m: new frame
    }
    assign, n_frames, along = rp.frame_groups(panos)
    assert assign['a'] == assign['b'] == ('s', 0)
    assert assign['c'] == ('s', 1)
    assert n_frames == {'s': 2}
    assert 10 < along[('s', 1)] < 12
