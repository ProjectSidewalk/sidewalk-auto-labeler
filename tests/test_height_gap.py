"""height_gap: why the #53 instruments disagree (issue #87).

Synthetic geometry only (test_mapillary_height's scene): no network, no run data.
"""
import math

import pytest

import geo
import mapillary_height as mh
import height_gap as hg
from test_mapillary_height import CAMERAS, FRAME, synthetic_panos


RAMPS = [(0.0, 0.0), (16.0, 0.0), (0.0, 16.0), (16.0, 16.0)]


def _uniform(h):
    """test_mapillary_height's scene with every camera at true height h (one rig), its
    detections projected with geo.ground_point_to_pano so they are the exact inverse of
    fusion's raycast (the flat-ENU scene there is exact only to ~1e-8)."""
    panos = synthetic_panos()
    for p in panos:
        pose = geo.pano_pose(p.pose_fields())
        dets = []
        for te, tn in RAMPS:
            proj = geo.ground_point_to_pano(pose, *FRAME.to_latlng(te, tn), camera_height=h,
                                            max_range_m=22.0)
            if proj is not None:
                dets.append((len(dets), proj.x_norm, proj.y_norm, 0.9))
        p.detections = dets
    return panos


def _key(_pid):
    return 'r1'


def test_b_raw_scale_is_exact_at_every_association_height():
    heights = (1.6, 2.1, 2.6, 3.0)
    sw = hg.sweep(_uniform(2.1), {'rig': _key}, heights=heights, null_seeds=range(2), min_rows=1)
    for h in heights:
        row = sw['b']['rig'][h]['r1']
        assert row['scale_s'] == pytest.approx(1 - 2.1 / h, abs=1e-6)
        assert row['h_b_raw'] == pytest.approx(2.1, abs=1e-6)
    h_star, _a, b = mh.fixed_point(list(heights),
                                   [sw['b']['rig'][h]['r1']['h_b_raw'] for h in heights])
    assert h_star == pytest.approx(2.1, abs=0.05) and abs(b) < 1e-6


def test_resynthesis_round_trips():
    panos = _uniform(2.6)
    truth, _stats = hg.planted_sites(panos)
    syn, counts = hg.resynthesize(panos, truth, lambda _p: 2.6, noise_scale=0.0)
    assert counts['no_site'] == counts['unprojectable'] == 0
    for a, b in zip(panos, syn):
        assert (a.lat, a.lng) == (b.lat, b.lng)
        assert len(a.detections) == len(b.detections)
        for da, db in zip(a.detections, b.detections):
            assert da[0] == db[0] and da[3] == db[3]
            assert da[1] == pytest.approx(db[1], abs=1e-9)
            assert da[2] == pytest.approx(db[2], abs=1e-9)
    assert hg.planted_sites(syn)[0] == pytest.approx(truth)


def test_offset_biases_both_instruments_alike():
    h, eps_px = 2.1, 2.0
    panos = _uniform(h)
    truth, _ = hg.planted_sites(panos)
    heights = (1.6, 2.1, 2.6, 3.0)
    got = {}
    for eps in (0.0, eps_px):
        syn, _ = hg.resynthesize(panos, truth, lambda _p: h, offset_px=eps)
        sw = hg.sweep(syn, {'rig': _key}, heights=heights, null_seeds=range(1), min_rows=1)
        a = mh.instrument_a(sw['a'], _key, n_boot=0)['r1']['h_star']
        b_raw = mh.fixed_point(list(heights),
                               [sw['b']['rig'][hh]['r1']['h_b_raw'] for hh in heights])[0]
        got[eps] = (a, b_raw)
    da = got[eps_px][0] - got[0.0][0]
    db = got[eps_px][1] - got[0.0][1]
    # first order: eps * (r^2 + h^2) / r, averaged over the scene's views
    eps = eps_px * geo.RAD_PER_HEATMAP_PX
    rs = []
    for _pid, pe, pn, _rig, _h in CAMERAS:
        for te, tn in RAMPS:
            d = math.hypot(te - pe, tn - pn)
            if d <= 22.0:
                rs.append(d)
    expect = eps * sum((r * r + h * h) / r for r in rs) / len(rs)
    assert da > 0 and db > 0
    assert da == pytest.approx(db, rel=0.3)
    assert da == pytest.approx(expect, rel=0.3) and db == pytest.approx(expect, rel=0.3)


def test_verdict_rules():
    real_ok = {'h_star_a': 1.98, 'h_star_b': 2.10, 'b_slope': 0.45}
    sim_ok = {n: {'b_2p6': 2.30, 'h_star_a': 2.03, 'h_star_b': 1.95} for n in (0.5, 1.0)}
    assert hg.verdict_candidate3(real_ok, sim_ok)[0] == 'confirmed'
    assert hg.verdict_candidate3(dict(real_ok, b_slope=0.1), sim_ok)[0] == 'rejected'
    recovers = {n: dict(s, b_2p6=2.05) for n, s in sim_ok.items()}
    assert hg.verdict_candidate3(real_ok, recovers)[0] == 'rejected'
    far = dict(real_ok, h_star_b=2.30)
    assert hg.verdict_candidate3(far, sim_ok)[0] == 'partial'
    assert hg.candidate3_share(far, 0.4) == pytest.approx(1 - 0.32 / 0.4)
    # candidate 1: a CI reaching eps = +2 px where the gap moves 0.25 m confirms ...
    grid = {-4.0: -0.5, -2.0: -0.25, 0.0: 0.0, 2.0: 0.25, 4.0: 0.5}
    assert hg.verdict_candidate1((1.0, 2.0), grid)[0] == 'confirmed'
    # ... a CI near zero does not, and a flat response rejects
    assert hg.verdict_candidate1((-0.5, 0.5), grid)[0] == 'partial'
    assert hg.verdict_candidate1((-4.0, 4.0), {e: 0.01 * e for e in grid})[0] == 'rejected'
    # candidate 2
    assert hg.verdict_candidate2([0.05, -0.02, 0.08], 0.8)[0] == 'confirmed'
    assert hg.verdict_candidate2([0.30, 0.40, 0.20], 0.1)[0] == 'rejected'
    assert hg.verdict_candidate2([0.15, 0.20, 0.10], 0.5)[0] == 'partial'
    # E5 and the overall label
    assert hg.verdict_e5(0.20, 0.03)[:2] == (True, 'A (2.0 m rig)')
    assert hg.verdict_e5(0.20, 0.08)[0] is False
    assert hg.verdict_explained(real_ok, 'confirmed', 'rejected', [0.35, 0.42], 0.394)[0] \
        == 'explained'
    assert hg.verdict_explained(real_ok, 'partial', 'rejected', [0.35], 0.394)[0] \
        == 'partially explained'
    assert hg.verdict_explained(far, 'confirmed', 'rejected', [0.35, 0.42], 0.394)[0] \
        == 'partially explained'
