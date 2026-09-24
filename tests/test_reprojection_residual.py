"""reprojection_residual (issue #36): the leave-one-view-out residual on synthetic sites.

Noise-free views of one ramp must agree exactly with each other when every pano is
raycast at its true height, and a deliberately wrong height must show up as the exact
range-scale identity the scale fit relies on (member - held-out = s * g, s = 1 - 1/k).
"""
import math

import pytest

import geo
import fuse_sites as fs
import reprojection_residual as rr
from test_fuse_sites import FRAME, make_pano
from test_eval_sites import _entry, _xy, _bundle_ops

TRUE_H = 2.45
# One ramp at (0, 0), four views at different ranges and bearings; every camera's
# true (and recorded, "measured") height is TRUE_H.
VIEWS = [('a', 0, -6), ('b', 8, 0), ('c', -10, 0), ('d', 3, 12)]


def _panos(height=TRUE_H):
    return [make_pano(pid, pe, pn, [(0, 0, 0.9)], height=height) for pid, pe, pn in VIEWS]


def _gtfree(panos, camera_height):
    params = fs.FuseParams(camera_height_m=camera_height)
    sites, frame, _ = fs.fuse(panos, params)
    assert len(sites) == 1, 'the synthetic views must associate into one site'
    svs = [rr.views_from_site(s) for s in sites]
    by_id = {p.pano_id: p for p in panos}
    return svs, rr.gtfree_rows('synthetic', 'x', svs, frame, by_id, camera_height)


def test_noise_free_leave_one_out_is_exactly_zero():
    rows = _gtfree(_panos(), geo.PER_PANO)[1]
    assert len(rows) == 4
    for r in rows:
        assert abs(r['dist_m']) < 1e-6
        assert abs(r['dx_px']) < 1e-3 and abs(r['dy_px']) < 1e-3
        assert r['camera_height_m'] == TRUE_H


@pytest.mark.parametrize('used_h', [2.6, 2.8])
def test_wrong_height_is_the_exact_range_scale_identity(used_h):
    """At a fixed wrong height every range is k = used/true times too long, and each
    view's along-ray residual is s * g_along with s = 1 - 1/k - so it grows linearly
    with the height error and, view by view, with range."""
    svs, rows = _gtfree(_panos(), used_h)
    s_true = 1.0 - TRUE_H / used_h
    for r in rows:
        assert r['along_m'] == pytest.approx(s_true * r['g_along'], abs=1e-3)
        assert r['along_m'] > 0          # ranges run long -> the view lands beyond
    by_range = sorted(rows, key=lambda r: r['range_m'])
    assert by_range[-1]['along_m'] > by_range[0]['along_m']
    table = rr.scale_table('synthetic', 'x', svs, rows)
    assert table[0]['scale_s'] == pytest.approx(s_true, abs=1e-4)
    assert table[0]['implied_range_scale_k'] == pytest.approx(used_h / TRUE_H, abs=1e-3)


def test_per_group_fit_separates_two_rigs():
    """Two rigs sharing sites: the joint per-group fit recovers each one's scale,
    which a pooled fit cannot."""
    rng_views = [((0, -6), 'o'), ((8, 0), 'o'), ((-10, 0), 'n'), ((3, 12), 'n'),
                 ((-7, 7), 'o'), ((6, -9), 'n')]
    views = []
    for i, ((pe, pn), rig) in enumerate(rng_views):
        k = 1.05 if rig == 'o' else 1.30
        r = math.hypot(pe, pn)
        b = math.atan2(-pe, -pn)           # bearing camera -> ramp at the origin
        e = pe + k * r * math.sin(b)
        n = pn + k * r * math.cos(b)
        views.append(rr.View(f'p{i}', 0, 0.5, 0.6, 0.9, e, n, (1.0, 0.2, 1.5 + i * 0.1),
                             k * r, math.degrees(b) % 360, 'launch', None))
    groups = [rig for _, rig in rng_views]
    design = rr.scale_design(views, groups, ['n', 'o'])
    loo = rr.leave_one_out(views)
    block = [(cols, (v.e - he, v.n - hn)) for v, ((he, hn), _), cols in
             zip(views, loo, design)]
    fit = rr.fit_scale([block], ['n', 'o'])
    assert fit['o'][0] == pytest.approx(1 - 1 / 1.05, abs=1e-9)
    assert fit['n'][0] == pytest.approx(1 - 1 / 1.30, abs=1e-9)


def test_leave_one_out_matches_a_refit_without_the_view():
    """The subtraction is the re-fuse: held-out position == GLS over the others."""
    views = [rr.View(f'p{i}', 0, 0.5, 0.6, 0.9, e, n, cov, 10.0, 0.0, 'launch', None)
             for i, (e, n, cov) in enumerate([(0.1, 0.0, (1.0, 0.1, 2.0)),
                                              (0.5, -0.3, (2.0, -0.4, 1.0)),
                                              (-0.2, 0.4, (1.5, 0.0, 1.5))])]
    for i, ((he, hn), _) in enumerate(rr.leave_one_out(views)):
        ee, nn = rr.solve_views(views[:i] + views[i + 1:])
        assert (he, hn) == pytest.approx((ee, nn), abs=1e-12)


def test_gt_anchored_box_missed_and_peak():
    """A judged member pano with a boxed true detection, and a judged non-member pano
    with a missed mark: on noise-free geometry at the true height every residual is 0,
    the member gets a left-out number, and the non-member's full == left-out."""
    panos = _panos() + [make_pano('g', 0, 9, [], heading_deg=180.0, height=TRUE_H)]
    params = fs.FuseParams(min_confidence=0.55, mask_rig=False,
                           camera_height_m=geo.PER_PANO)
    sites, frame, _ = fs.fuse(panos, params)
    svs = [rr.views_from_site(s) for s in sites]
    mark = _xy(0, 9, 180.0, 0, 0)
    # _xy raycasts at the default height; re-aim it for this camera's true height
    mark['y'] = 0.5 + math.atan(TRUE_H / 9.0) / math.pi
    a = next(p for p in panos if p.pano_id == 'a')
    _, ax, ay, _ = a.detections[0]
    verdicts = {'a': _entry(dets=[True]), 'g': _entry(missed=[mark], no_missed=False),
                'b': _entry(dets=[True])}
    boxes = {'a': {'det:0': (ax, ay, {'x': ax, 'y': ay})}}
    rows, counts, warnings = rr.gt_anchored_rows(
        'synthetic', 'per-pano', verdicts, _bundle_ops(panos, verdicts), boxes, panos,
        svs, frame, params)
    assert not warnings
    kinds = {r['pano_id']: r for r in rows}
    assert kinds['a']['ref_kind'] == 'box' and kinds['a']['pano_in_site']
    assert kinds['b']['ref_kind'] == 'peak'
    assert kinds['g']['ref_kind'] == 'missed' and not kinds['g']['pano_in_site']
    for r in rows:
        assert abs(r['full_px']) < 1e-3 and abs(r['loo_px']) < 1e-3
    assert kinds['g']['full_px'] == kinds['g']['loo_px']
    assert kinds['a']['n_views_loo'] == 3
    assert counts['refs_missed'] == 1 and counts['missed_no_site'] == 0


def test_box_with_mismatched_anchor_is_refused():
    warnings = []
    boxes = {'p': {'det:0': (0.3, 0.6, {'x': 0.9, 'y': 0.6})}}
    assert rr._box_for(boxes, 'p', 'det:0', 0.3, 0.6, warnings) is None
    assert warnings


def test_sites_json_round_trip_rebuilds_the_same_views(tmp_path):
    """Reading sites.jsonl back must reproduce the in-memory site: same members, same
    position to the rounding of the stored fields."""
    panos = _panos()
    params = fs.FuseParams(min_confidence=0.55, mask_rig=False)
    sites, frame, stats = fs.fuse(panos, params)
    fs.write_sites(sites, frame, stats, params, tmp_path / 'sites.jsonl',
                   tmp_path / 'sites_meta.json')
    svs, frame2, p, worst = rr.load_sites_json(tmp_path)
    assert p['camera_height_m'] == geo.DEFAULT_CAMERA_HEIGHT_M
    assert [v.pano_id for v in svs[0].views] == \
        [v.pano_id for v in rr.views_from_site(sites[0]).views]
    assert worst < 0.02


def test_pixel_residual_wraps_the_seam():
    dx, dy = rr.pixel_residual(0.999, 0.6, 0.001, 0.59)
    assert dx == pytest.approx(-0.002 * 1024)
    assert dy == pytest.approx(0.01 * 512)


def test_along_cross_signs():
    # a ray heading due north; a residual pointing north-east
    a, c = rr.along_cross(1.0, 2.0, 0.0)
    assert (a, c) == pytest.approx((2.0, 1.0))


def test_null_scale_is_near_zero_without_a_scale_error():
    """Re-drawn at the fused positions with the error model's noise, the estimator must
    report (near) no scale error - that is the baseline a measured s is read against."""
    panos = []
    for k in range(40):
        oe = 60.0 * k
        panos += [make_pano(f'{pid}{k}', pe + oe, pn, [(oe, 0, 0.9)], height=TRUE_H)
                  for pid, pe, pn in VIEWS]
    sites, _, _ = fs.fuse(panos, fs.FuseParams(camera_height_m=geo.PER_PANO))
    svs = [rr.views_from_site(s) for s in sites]
    assert sum(len(s.views) == 4 for s in svs) == 40
    null, naive = rr.null_scale(svs, seed=1)
    assert abs(null) < 0.05
    # the naive along~range slope is NOT near zero on the same null draw: a view whose
    # range reads long is both farther and beyond the consensus (errors in variables)
    assert naive > 0.03 and naive > 3 * abs(null)
    assert rr.null_scale(svs, seed=1) == (null, naive)   # deterministic per seed
