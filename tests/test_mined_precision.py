"""mined_precision: the RampNet#158 step-1 check, end to end on synthetic geometry.

One ramp, seen by enough panos to make a strong site; then three judged panos
that stood next to it without detecting it, one per outcome the check has to
tell apart."""
import csv

import fuse_sites as fs
import mined_precision as mp
from test_fuse_sites import make_pano
from test_eval_sites import _entry, _xy, _bundle_ops


def _scene():
    panos = [
        # ramp A at (0, 0): three neighbours detect it -> a strong site
        make_pano('n1', 0, -10, [(0, 0, 0.9)]),
        make_pano('n2', 10, 0, [(0, 0, 0.8)]),
        make_pano('n3', -10, 0, [(0, 0, 0.95)]),
        # g1 stood 8 m north, detected nothing, reviewer marked the miss -> tp
        make_pano('g1', 0, 8, [], heading_deg=180.0),
        # g2 stood 5 m NE, detected nothing, reviewer attested no misses -> fp
        make_pano('g2', 5, 5, [], heading_deg=225.0),
        # g3 stood 6 m south and DID detect it (verdict true) -> a member, not a candidate
        make_pano('g3', 0, -6, [(0, 0, 0.7)]),
        # g4 stood 40 m away: outside every radius, never a candidate
        make_pano('g4', 40, 0, [], heading_deg=270.0),
    ]
    verdicts = {
        'g1': _entry(missed=[_xy(0, 8, 180.0, 0, 0)], no_missed=False),
        'g2': _entry(no_missed=True),
        'g3': _entry(dets=[True]),
        'g4': _entry(no_missed=True),
    }
    return panos, verdicts


def test_buckets_and_headline():
    panos, verdicts = _scene()
    result, cands = mp.run_city(verdicts, _bundle_ops(panos, verdicts), panos,
                                fs.FuseParams(), radii_m=(10.0, 15.0))
    assert result['n_strong_sites'] == 1
    assert result['n_judged_panos'] == 4
    by_pano = {c.pano_id: c for c in cands}
    assert set(by_pano) == {'g1', 'g2'}          # g3 is a member, g4 out of range
    assert by_pano['g1'].bucket == 'tp'
    assert by_pano['g1'].nearest_gt_kind == 'missed'
    assert by_pano['g2'].bucket == 'fp'
    assert by_pano['g2'].nearest_gt_kind == ''
    assert by_pano['g1'].n_op_panos == 4         # n1, n2, n3, g3
    assert by_pano['g1'].best_conf == 0.95
    # the projection points g1 at the ramp: dead ahead (heading 180 looks at it), 8 m out
    assert abs(by_pano['g1'].x_norm - 0.5) < 1e-6
    assert abs(by_pano['g1'].range_m - 8.0) < 0.05
    h = result['headline']
    assert (h['tp'], h['fp'], h['precision']) == (1, 1, 0.5)
    assert result['reading'].startswith('0.50-0.80')
    # cumulative-by-distance strata: g2 (7.1 m) and g1 (8 m) both inside 10 m
    assert [r['candidates'] for r in result['strata']['radius']] == [2, 2]


def test_csv_has_one_row_per_candidate(tmp_path):
    panos, verdicts = _scene()
    result, cands = mp.run_city(verdicts, _bundle_ops(panos, verdicts), panos,
                                fs.FuseParams())
    mp.write_outputs(tmp_path, mp.format_report('synthetic', result), cands)
    with open(tmp_path / 'candidates.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(cands) == 2
    assert {r['bucket'] for r in rows} == {'tp', 'fp'}
    assert 'headline precision 0.500' in (tmp_path / 'report.md').read_text(encoding='utf-8')
