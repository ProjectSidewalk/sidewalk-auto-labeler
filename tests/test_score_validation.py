"""Unit tests for score_validation.py: interval math and verdict collection."""
import pytest

import score_validation as sv


def test_wilson_interval_edges():
    assert sv.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = sv.wilson_interval(10, 10)
    assert 0.0 < lo < 1.0 and hi == 1.0
    lo, hi = sv.wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0


def test_wilson_interval_contains_p_and_narrows():
    lo_s, hi_s = sv.wilson_interval(5, 10)
    lo_l, hi_l = sv.wilson_interval(500, 1000)
    assert lo_s < 0.5 < hi_s and lo_l < 0.5 < hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)  # more data, tighter interval


CONFS = {
    "P_TWO_DETS": [0.91, 0.63],
    "P_EMPTY": [],
    "P_ONE_DET": [0.7],
}


def _panos(no_missed_flags=True):
    """Three panos: fully judged w/ a missed mark, empty+affirmed, judged-unaffirmed."""
    panos = {
        "P_TWO_DETS": {"group": "random", "dets": [True, True],
                       "missed": [{"x": 0.2, "y": 0.3}], "no_missed": False},
        "P_EMPTY": {"group": "empty", "dets": [], "missed": [], "no_missed": True},
        "P_ONE_DET": {"group": "random", "dets": [False], "missed": [], "no_missed": False},
    }
    if not no_missed_flags:  # old-schema export: flag absent everywhere
        for entry in panos.values():
            del entry["no_missed"]
    return panos


def test_collect_unconfirmed_pano_counts_for_precision_not_recall():
    judged, recall_judged, missed, n_seen, n_judged, n_unconfirmed = \
        sv.collect(_panos(), CONFS)
    assert (n_seen, n_judged, n_unconfirmed) == (3, 3, 1)
    # P_ONE_DET's crop verdict is valid for precision even though its missed-ramp
    # check was never confirmed...
    assert sorted(judged) == [(0.63, True), (0.7, False), (0.91, True)]
    # ...but it is excluded from the recall pool.
    assert sorted(recall_judged) == [(0.63, True), (0.91, True)]
    assert missed == 1


def test_collect_old_schema_keeps_legacy_behavior():
    judged, recall_judged, missed, n_seen, n_judged, n_unconfirmed = \
        sv.collect(_panos(no_missed_flags=False), CONFS)
    # Legacy entries (no no_missed key) are trusted for recall, as before.
    assert (n_seen, n_judged, n_unconfirmed) == (3, 3, 0)
    assert len(judged) == 3 and len(recall_judged) == 3 and missed == 1


def test_collect_mixed_schema_gates_per_entry():
    # A legacy entry hand-merged into a new-schema file must stay trusted; the
    # new-schema unconfirmed entry must still be excluded from recall.
    panos = _panos()
    del panos["P_TWO_DETS"]["no_missed"]  # legacy entry, missed mark
    _, recall_judged, missed, _, _, n_unconfirmed = sv.collect(panos, CONFS)
    assert n_unconfirmed == 1  # P_ONE_DET only
    assert sorted(recall_judged) == [(0.63, True), (0.91, True)]
    assert missed == 1


def test_collect_skips_partially_judged():
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True, None],
                            "missed": [], "no_missed": True}}
    judged, recall_judged, missed, n_seen, n_judged, n_unconfirmed = \
        sv.collect(panos, CONFS)
    assert n_seen == 1 and n_judged == 0
    assert judged == [] and recall_judged == [] and missed == 0


def test_collect_skips_mismatched_detection_counts(capsys):
    panos = {"P_TWO_DETS": {"group": "random", "dets": [True],  # results.jsonl has 2
                            "missed": [], "no_missed": True}}
    judged, _, _, n_seen, n_judged, _ = sv.collect(panos, CONFS)
    assert n_seen == 0 and n_judged == 0 and judged == []
    assert "don't match" in capsys.readouterr().out


def test_collect_exclude_top():
    panos = _panos()
    panos["P_TWO_DETS"]["group"] = "top"
    judged, recall_judged, missed, n_seen, n_judged, n_unconfirmed = \
        sv.collect(panos, CONFS, exclude_top=True)
    assert n_seen == 2 and n_judged == 2
    # P_EMPTY (affirmed, no dets) and P_ONE_DET (unconfirmed) remain.
    assert judged == [(0.7, False)] and recall_judged == [] and missed == 0
    assert n_unconfirmed == 1
