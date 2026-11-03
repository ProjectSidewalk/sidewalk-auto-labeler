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


def test_band_edges():
    assert mc.band_of(8.0) == '0-8 m' and mc.band_of(8.01) == '8-12 m'
    assert mc.band_of(15.0) == '12-15 m'
