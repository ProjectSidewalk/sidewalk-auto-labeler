"""mine_city (RampNet#158 step 4): the step-3 peak_flat rule over every non-member pano."""
import fuse_sites as fs
import geo
import mine_city as mc
from detectors import BENCHMARK_CONFIDENCE
from test_fuse_sites import make_pano
from test_mined_precision import _scene, _run

CLI = mc.cli_params(geo.DEFAULT_CAMERA_HEIGHT_M)


def test_cli_params_pin_the_benchmark_tier_with_rig_mask_and_pose_off():
    assert CLI.min_confidence == BENCHMARK_CONFIDENCE
    assert (CLI.mask_rig, CLI.apply_pose) == (False, fs.POSE_OFF)
    assert mc.cli_params(geo.PER_RIG).camera_height_m == geo.PER_RIG


def test_mine_covers_the_same_pairs_as_the_candidates_and_applies_the_rule():
    panos, verdicts = _scene()
    _, cands = _run(panos, verdicts, params=CLI)
    by = {c.pano_id: c for c in cands}
    # g1 has a sub-threshold peak on the flat anchor -> emitted; g2 has an operational
    # peak there -> not emitted; every other pano has no peaks
    peaks = {p.pano_id: [] for p in panos}
    peaks['g1'] = [(by['g1'].x_norm + 0.002, by['g1'].y_norm, 0.3)]
    peaks['g2'] = [(by['g2'].x_norm, by['g2'].y_norm, 0.7)]
    rows, stats = mc.mine(panos, CLI, peaks, radius_m=15.0, city='synthetic',
                          benchmark_ids={'g1'})
    got = {r['pano_id']: r for r in rows}
    # the judged candidates are a subset of the city-wide pairs, with the same anchor
    assert {'g1', 'g2'} <= set(got)
    assert abs(got['g1']['anchor_x'] - by['g1'].x_norm) < 1e-6
    assert got['g1']['status'] == 'emit' and got['g1']['peak_conf'] == 0.3
    assert got['g2']['status'] == 'operational_in_window'
    assert got['g1']['src_pano'] == 'n3'                      # the step-2 source rule
    assert stats['emitted'] == 1 and stats['emitted_on_benchmark_panos'] == 1
    assert 'g3' not in got                                     # a member is never mined
    assert all(r['range_m'] <= 15.0 for r in rows)


def test_cli_tier_is_the_benchmark_tier_not_the_operating_point():
    """A pano whose only view of the ramp is a 0.4 detection: an operational member at
    0.30 (FuseParams()), a sub-threshold member at the CLI's 0.55, skipped and counted."""
    panos, _ = _scene()
    panos = panos + [make_pano('g5', 3, 3, [(0, 0, 0.4)])]
    peaks = {p.pano_id: [] for p in panos}
    rows, stats = mc.mine(panos, CLI, peaks, radius_m=15.0)
    assert stats['excluded_subfloor_members'] == 1
    assert {r['n_op_panos'] for r in rows} == {4}              # n1 n2 n3 g3
    _, at_030 = mc.mine(panos, fs.FuseParams(), peaks, radius_m=15.0)
    assert at_030['excluded_subfloor_members'] == 0


def test_an_emitted_peak_on_the_camera_rig_is_withheld():
    """g6 stands 1.5 m from the ramp: its anchor dips ~60 deg, past the 49 deg rig mask,
    so the sub-threshold peak there is the vehicle, not a training target."""
    panos, _ = _scene()
    panos = panos + [make_pano('g6', 0, 1.5, [], heading_deg=180.0)]
    peaks = {p.pano_id: [] for p in panos}
    rows, _ = mc.mine(panos, CLI, peaks, radius_m=15.0)
    anchor = next(r for r in rows if r['pano_id'] == 'g6')
    assert anchor['anchor_y'] > 0.5 + 49 / 180
    peaks['g6'] = [(anchor['anchor_x'], anchor['anchor_y'], 0.3)]
    rows, stats = mc.mine(panos, CLI, peaks, radius_m=15.0)
    g6 = next(r for r in rows if r['pano_id'] == 'g6')
    assert (g6['status'], g6['emit'], g6['peak_conf']) == ('on_rig', False, 0.3)
    assert stats['by_status']['on_rig'] == 1 and stats['emitted'] == 0
    # just above the mask the same rule still emits
    assert mc.target_status((0.5, 0.77), [(0.5, 0.772, 0.3)])[0] == 'emit'


def test_tally_splits_tiers_and_counts_distinct_targets():
    base = dict(emit=True, status='emit', benchmark_pano=False, pano_id='p', x=0.5, y=0.6)
    rows = [dict(base, site_id=1, band='0-8 m', peak_conf=0.4),     # >= 0.30
            dict(base, site_id=2, band='8-12 m', peak_conf=0.4),    # same pixel, 2nd site
            dict(base, site_id=3, band='8-12 m', peak_conf=0.2, x=0.7),
            dict(base, site_id=3, band='8-12 m', peak_conf=0.2, pano_id='q',
                 benchmark_pano=True),
            dict(base, site_id=4, band='12-15 m', peak_conf=None, emit=False,
                 status='no_peak', x=None, y=None)]
    t = mc.tally(rows)
    assert (t['pairs'], t['emitted'], t['emitted_off_benchmark']) == (5, 4, 3)
    assert t['emitted_ge_0_30'] == 2 and t['emitted_ge_0_30_off_benchmark'] == 2
    assert t['emitted_ge_0_30_by_band'] == {'0-8 m': 1, '8-12 m': 1, '12-15 m': 0}
    assert t['emitted_distinct_targets'] == 3
    assert t['emitted_distinct_targets_off_benchmark'] == 2
    assert t['emitted_distinct_targets_by_band'] == {'0-8 m': 1, '8-12 m': 3, '12-15 m': 0}
    assert (t['emitted_targets_multi_site'], t['emitted_targets_multi_band']) == (1, 1)
    assert t['emitted_targets_multi_band_off_benchmark'] == 1
    assert t['emitted_distinct_targets_off_benchmark_by_band']['8-12 m'] == 2
    on_bench = mc.tally([dict(r, benchmark_pano=True) for r in rows])
    assert on_bench['emitted_targets_multi_site_off_benchmark'] == 0
    assert t['tiers'] == {'operational_in_this_script': 0.55, 'production_operational': 0.3}


def test_tally_of_the_committed_labels_reproduces_the_committed_summary():
    """The committed step-4 summary.json predates the tier/target keys; every key it has
    that the rows determine must come back unchanged from its labels.jsonl."""
    import json
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / 'docs/figures/mined-precision/data/step4/richmond'
    summary = json.loads((d / 'summary.json').read_text(encoding='utf-8'))
    t = mc.tally(mc.read_labels(d / 'labels.jsonl'))
    shared = set(summary) & set(t)
    assert len(shared) == 9
    assert {k: t[k] for k in shared} == {k: summary[k] for k in shared}


def test_per_rig_without_a_table_is_a_usage_error_not_a_traceback(tmp_path, capsys):
    import pytest
    res = tmp_path / 'results.jsonl'
    res.write_text('', encoding='utf-8')
    with pytest.raises(SystemExit) as e:
        mc.main(['--results', str(res), '--peaks', str(res), '--city', 'c',
                 '--camera-height', 'per-rig', '--out', str(tmp_path / 'o')])
    assert e.value.code == 2 and 'camera-height table' in capsys.readouterr().err


def test_band_edges():
    assert mc.band_of(8.0) == '0-8 m' and mc.band_of(8.01) == '8-12 m'
    assert mc.band_of(15.0) == '12-15 m'
