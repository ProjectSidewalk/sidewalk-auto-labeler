"""One camera-height resolver for every script that raycasts (issue #56).

fuse_sites.load_at_height turns a --camera-height-m value into loaded panos and a
FuseParams height; fuse_sites' CLI, eval_sites, eval_ps_clustering and mined_precision
all go through it. These tests pin the parser, the resolution on a synthetic run with a
harvested depth index, the report lines that state the resolution, and output naming.
"""
import sys

import pytest

import geo
import fuse_sites as fs
import mined_precision as mp
from test_fuse_sites import FRAME, _record_line, make_pano
from test_mined_precision import _run, _scene


def _run_dir(tmp_path, n=80, measured=60, height=1.8):
    """A pre-#40 GSV run of `n` 2026 panos whose only heights are a harvested
    depth/index.csv covering the first `measured` of them."""
    panos = [make_pano(f'p{i}', 12.0 * i, -8, [(12.0 * i + 2, 0, 0.9)], capture='2026-06')
             for i in range(n)]
    (tmp_path / 'results.jsonl').write_text(
        '\n'.join(_record_line(p) for p in panos) + '\n', encoding='utf-8')
    (tmp_path / 'depth').mkdir()
    (tmp_path / 'depth' / 'index.csv').write_text(
        'panorama_id,degenerate,camera_height_m,ground_tilt_deg,height_spread_m,'
        'n_standin_planes\n'
        + ''.join(f'p{i},0,{height},1.4,0.1,0\n' for i in range(measured)),
        encoding='utf-8')
    return tmp_path / 'results.jsonl'


@pytest.mark.parametrize('text, value', [('2.6', 2.6), ('2.341219672825709',
                                                        2.341219672825709),
                                         ('auto', 'auto'), ('per-pano', geo.PER_PANO),
                                         ('per-rig', geo.PER_RIG)])
def test_parser_takes_metres_and_the_three_modes(text, value):
    assert fs.fuse_camera_height_arg(text) == value


@pytest.mark.parametrize('text', ['garbage', 'per_pano', 'AUTO', ''])
def test_parser_refuses_anything_else(text, monkeypatch):
    with pytest.raises(ValueError):
        fs.fuse_camera_height_arg(text)
    # ...and a CLI using it exits with a usage error instead of running
    monkeypatch.setattr(sys, 'argv', ['mined_precision.py', 'paterson',
                                      '--camera-height', text])
    with pytest.raises(SystemExit):
        mp.main()


def test_per_pano_takes_measured_heights_and_counts_the_fallback(tmp_path):
    src = _run_dir(tmp_path)
    panos, skipped, height, auto = fs.load_at_height(src, geo.PER_PANO)
    assert (skipped, height, auto) == (0, geo.PER_PANO, None)
    by_id = {p.pano_id: p for p in panos}
    assert by_id['p0'].camera_height_m == 1.8 and by_id['p0'].height_from_index
    assert by_id['p79'].camera_height_m is None               # no index row: falls back
    params = fs.FuseParams(camera_height_m=height)
    counts = fs.resolved_height_counts(panos, params)
    assert (counts['measured'], counts['fallback']) == (60, 20)
    assert counts['spread_definition'] == 'measured_planes_only'
    lines = fs.height_resolution_lines(geo.PER_PANO, panos, params)
    assert lines[0] == '- camera height mode `per-pano`'
    assert '60 of 80 panos took a measured height; 20 fell back to the 2.6 m' in lines[1]
    assert 'ALL' not in lines[1]
    assert any('60 heights read from depth/index.csv' in ln for ln in lines)


def test_a_number_reads_no_heights_and_reports_nothing_new(tmp_path):
    src = _run_dir(tmp_path)
    panos, _, height, auto = fs.load_at_height(src, 2.6)
    assert (height, auto) == (2.6, None)
    assert all(p.camera_height_m is None for p in panos)
    # ...unless asked (mined_precision keeps its pre-#56 load exactly)
    panos, _, _, _ = fs.load_at_height(src, 2.2, read_heights=True)
    assert sum(p.camera_height_m is not None for p in panos) == 60
    assert fs.height_resolution_lines(2.2, panos, fs.FuseParams(camera_height_m=2.2)) == []


def test_auto_resolves_per_rig_and_reports_the_year_table(tmp_path):
    src = _run_dir(tmp_path)                           # 60 of 80 measured: qualifies
    panos, _, height, auto = fs.load_at_height(src, fs.HEIGHT_AUTO)
    assert height == geo.PER_RIG and auto['resolved'] == 'gsv-per-rig'
    assert {p.camera_height_m for p in panos} == {fs.GSV_RIG_LOW_M}   # 1.8 < 2.1 cut
    params = fs.FuseParams(camera_height_m=height)
    lines = fs.height_resolution_lines(fs.HEIGHT_AUTO, panos, params, auto)
    assert lines[0] == '- camera height mode `auto`: auto -> gsv-per-rig'
    assert '80 of 80 panos took a per-rig height; 0 fell back' in lines[1]
    assert '  - 2026: median 1.80 m, 60 of 80 measured -> 2 m' in lines
    # resolve_auto=False (--pose-ablation / --implied-height) keeps the constant
    _, _, height, auto = fs.load_at_height(src, fs.HEIGHT_AUTO, resolve_auto=False)
    assert (height, auto) == (geo.DEFAULT_CAMERA_HEIGHT_M, None)


def test_every_pano_falling_back_is_said_out_loud(tmp_path):
    """Richmond's shape: Mapillary-like coverage with no measured height at all."""
    src = _run_dir(tmp_path, n=5, measured=0)
    panos, _, height, _ = fs.load_at_height(src, geo.PER_PANO)
    lines = fs.height_resolution_lines(geo.PER_PANO, panos,
                                       fs.FuseParams(camera_height_m=height))
    assert '0 of 5 panos took a measured height; 5 fell back' in lines[1]
    assert 'ALL of them' in lines[1]
    panos, _, height, auto = fs.load_at_height(src, fs.HEIGHT_AUTO)
    assert height == geo.DEFAULT_CAMERA_HEIGHT_M and 'reason' in auto
    lines = fs.height_resolution_lines(fs.HEIGHT_AUTO, panos,
                                       fs.FuseParams(camera_height_m=height), auto)
    assert lines[1] == ('- all 5 panos raycast at the constant 2.6 m (no pano took a '
                        'measured or per-rig height)')


def test_per_rig_without_a_table_is_refused(tmp_path):
    src = _run_dir(tmp_path, n=3, measured=0)
    with pytest.raises(ValueError, match='per-rig needs a camera-height table'):
        fs.load_at_height(src, geo.PER_RIG)


@pytest.mark.parametrize('height, suffix, label', [
    (2.6, '', '2.6 m'), (2.341219672825709, '_h2.34', '2.34122 m'),
    (geo.PER_PANO, '_per-pano', 'per-pano'), (fs.HEIGHT_AUTO, '_auto', 'auto'),
    (geo.PER_RIG, '_per-rig', 'per-rig')])
def test_output_dir_suffix_and_label(height, suffix, label):
    """eval_ps_clustering writes ps_clustering_eval<suffix>/: a frame never
    overwrites another, and the committed 2.6 m / _h2.34 dirs keep their names."""
    assert fs.frame_suffix(height) == suffix
    assert fs.frame_label(height) == label


def test_mined_precision_records_the_mode_and_pools_mixed_heights():
    panos, verdicts = _scene()
    a = _run(panos, verdicts, city='alpha', params=fs.FuseParams(camera_height_m=2.6))
    b = _run(panos, verdicts, city='beta', params=fs.FuseParams(camera_height_m=2.6),
             height_mode=fs.HEIGHT_AUTO,
             auto={'mode': 'auto', 'resolved': 2.6, 'reason': 'no GSV panos'})
    assert a[0]['params']['camera_height_m'] == 2.6 and a[0]['height_resolution'] == []
    assert b[0]['params']['camera_height_m'] == 'auto'
    assert 'camera height auto\n' in mp.format_report('beta', b[0]) + '\n'
    assert 'camera-height resolution (#56):' in mp.format_report('beta', b[0])
    assert 'camera height 2.6 m\n' in mp.format_report('alpha', a[0]) + '\n'
    pooled, _ = mp.pool_cities([a, b], (10.0, 15.0), 3, 5.0, cities=['alpha', 'beta'])
    assert pooled['params']['camera_height_m'] == '2.6/auto'
    assert pooled['height_resolution'][0] == '- beta: camera height mode `auto`: ' \
                                             'auto -> 2.6 (no GSV panos)'
    # a numeric pool reads exactly as before #56
    numeric, _ = mp.pool_cities([a, _run(panos, verdicts, city='gamma',
                                         params=fs.FuseParams(camera_height_m=2.2))],
                                (10.0, 15.0), 3, 5.0)
    assert numeric['params']['camera_height_m'] == '2.2/2.6'
    assert 'camera height 2.2/2.6 m' in mp.format_report('pooled', numeric)


def test_fuse_sites_doctests_run():
    """The helpers' docstring examples (frame_suffix, the parser, ...) stay true."""
    import doctest
    assert doctest.testmod(fs).failed == 0


def test_the_frame_moves_the_detection_and_the_gt_raycast_together(tmp_path):
    """eval_ps_clustering / eval_sites / mined_precision place detections (fs.project)
    and GT (es.build_gt) with the SAME params, so a per-pano frame must move both. A ramp
    at (0, 0) seen from 10 m south by g1's detection and from 10 m north as g2's missed
    mark, both cameras truly at 1.8 m (measured in the index): per-pano puts the two on
    the ramp, the 2.6 m constant pushes each ~4.4 m long, in opposite directions."""
    import eval_sites as es
    true_h = 1.8
    g1 = make_pano('g1', 0, -10, [(0, 0, 0.9)], height=true_h)
    g2 = make_pano('g2', 0, 10, [], heading_deg=180.0)
    mark = make_pano('tmp', 0, 10, [(0, 0, 1.0)], heading_deg=180.0, height=true_h)
    _, mx, my, _ = mark.detections[0]
    (tmp_path / 'results.jsonl').write_text(
        _record_line(g1) + '\n' + _record_line(g2) + '\n', encoding='utf-8')
    (tmp_path / 'depth').mkdir()
    (tmp_path / 'depth' / 'index.csv').write_text(
        'panorama_id,degenerate,camera_height_m,ground_tilt_deg,height_spread_m,'
        f'n_standin_planes\ng1,0,{true_h},1.4,0.01,0\ng2,0,{true_h},1.4,0.01,0\n',
        encoding='utf-8')
    verdicts = {'g1': {'group': 'random', 'dets': [True], 'missed': [], 'no_missed': True},
                'g2': {'group': 'random', 'dets': [], 'no_missed': False,
                       'missed': [{'x': mx, 'y': my}]}}
    bundle = {'g1': [(x, y, c) for _, x, y, c in g1.detections], 'g2': []}

    def gap(mode):
        panos, _, height, _ = fs.load_at_height(tmp_path / 'results.jsonl', mode)
        params = fs.FuseParams(camera_height_m=height, apply_pose=fs.POSE_OFF)
        dets, frame, _ = fs.project(panos, params)
        points, _, _, _ = es.build_gt(verdicts, bundle, {p.pano_id: p for p in panos},
                                      params, frame)
        (det,) = dets
        (missed,) = [pt for pt in points if pt.pano_id == 'g2']
        return abs(det.n - missed.n), frame.to_enu(*FRAME.to_latlng(0, 0))[1] - det.n

    per_pano_gap, per_pano_off = gap(geo.PER_PANO)
    const_gap, const_off = gap(2.6)
    assert per_pano_gap < 0.05 and abs(per_pano_off) < 0.05
    assert const_gap == pytest.approx(2 * 10 * (2.6 / true_h - 1), abs=0.1)
