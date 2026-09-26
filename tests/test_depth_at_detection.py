"""depth_at_detection (#47 step 1): the plane class, the local height offset, the column
walk and the pre-registered verdict.

All payloads are synthetic (tests/test_depth.py's build_payload). The planes that matter carry a nonzero normal-x or sit at an asymmetric image
column, so a mirrored (raster-frame, `1 - x`) lookup fails here rather than passing
unnoticed (#80).
"""
import math
import struct
import types
import zlib

import pytest

import depth
import depth_at_detection as dad
from test_depth import GROUND, WALL, build_payload, tilted_ground

W, H = 64, 32
PATCH_COLS, PATCH_ROWS = range(8, 12), range(20, 24)       # floor patch, image columns 8-11
WALL_COLS, WALL_ROWS = range(40, 44), range(18, 22)
LID_COLS, LID_ROWS = range(24, 28), range(10, 18)          # 6 of 8 rows above the horizon


def _index(ground=GROUND, patch_d=2.35):
    """Ground below the horizon, a level floor patch `2.5 - patch_d` m above it, a wall,
    and a level plane mostly ABOVE the horizon (an overpass underside)."""
    planes = [GROUND, ground, (0.0, 0.0, -1.0, patch_d), WALL, (0.0, 0.0, 1.0, 3.0)]
    idx = [depth.SKY] * (W * H)
    for r in range(H // 2, H):
        for c in range(W):
            idx[r * W + c] = 1
    for plane, cols, rows in ((2, PATCH_COLS, PATCH_ROWS), (3, WALL_COLS, WALL_ROWS),
                              (4, LID_COLS, LID_ROWS)):
        for r in rows:
            for c in cols:
                idx[r * W + c] = plane
    payload = depth.parse(build_payload(W, H, planes, idx))
    return dad.PayloadIndex(payload)


def _xy(col, row):
    return (col + 0.5) / W, (row + 0.5) / H


def test_classes_and_range_in_the_image_frame():
    pix = _index()
    cases = {dad.GROUND: _xy(30, 26), dad.FLOOR: _xy(9, 21),
             dad.NON_HORIZONTAL: _xy(41, 19), dad.HORIZONTAL_NONFLOOR: _xy(25, 17),
             dad.NO_PLANE: _xy(5, 3)}
    for cls, (x, y) in cases.items():
        out = dad.classify(pix, x, y)
        assert out['plane_class'] == cls, cls
        want = depth.ground_range_at(pix.payload, x, y)
        if want is None:
            assert out['range_depth_m'] is None
        else:
            assert out['range_depth_m'] == pytest.approx(want, rel=1e-12)
    assert dad.classify(pix, *cases[dad.NON_HORIZONTAL])['plane_tilt_deg'] == \
        pytest.approx(90.0)
    # The #80 guard: the mirrored column of the floor patch is plain ground.
    x, y = cases[dad.FLOOR]
    assert dad.classify(pix, 1.0 - x, y)['plane_class'] == dad.GROUND


def test_offset_local_is_the_height_above_the_road():
    pix = _index()
    out = dad.classify(pix, *_xy(9, 21))
    assert out['offset_local_m'] == pytest.approx(0.15, abs=1e-6)
    # a level reference: the tilt term is zero, so the split gives the whole offset
    assert out['offset_local_level_ref_m'] == pytest.approx(0.15, abs=1e-6)
    assert out['rows_to_ref'] == 3                     # rows 22, 23 are patch; 24 is road
    assert out['offset_dominant_m'] == pytest.approx(0.15, abs=1e-6)


def test_offset_against_a_tilted_reference_is_analytic():
    t, b, h, patch_d = 3.0, 35.0, 2.5, 2.35
    pix = _index(ground=tilted_ground(t, b, h), patch_d=patch_d)
    x, y = 9.3 / W, 21.7 / H                            # not a pixel centre
    # The exact ray, written out independently of depth.py: image frame, +z down.
    theta, phi = (1.0 - y) * math.pi, (1.0 - x) * 2.0 * math.pi + math.pi / 2.0
    v = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    p = [c * patch_d / v[2] for c in v]                 # hit on the level patch
    tr, br = math.radians(t), math.radians(b)
    z_ref = h / math.cos(tr) + math.tan(tr) * (math.cos(br) * p[0] + math.sin(br) * p[1])
    out = dad.classify(pix, x, y)
    assert out['plane_class'] == dad.FLOOR
    assert out['offset_local_m'] == pytest.approx(z_ref - p[2], abs=1e-5)
    # the descriptive split: a level reference at the tilted road's nadir height, plus the
    # reference-tilt term carried out to the hit point (#91 review finding 2)
    assert out['offset_local_level_ref_m'] == pytest.approx(h / math.cos(tr) - p[2], abs=1e-5)
    assert out['offset_local_m'] - out['offset_local_level_ref_m'] == pytest.approx(
        math.tan(tr) * (math.cos(br) * p[0] + math.sin(br) * p[1]), abs=1e-5)
    assert out['plane_d_m'] == pytest.approx(patch_d, abs=1e-6)
    assert out['ref_d_m'] == pytest.approx(h, abs=1e-6)
    assert out['ref_tilt_deg'] == pytest.approx(t, abs=1e-4)
    # a mirrored azimuth would put the hit elsewhere on the tilted plane
    pm = [p[0] * -1, p[1], p[2]]
    z_m = h / math.cos(tr) + math.tan(tr) * (math.cos(br) * pm[0] + math.sin(br) * pm[1])
    assert abs((z_m - pm[2]) - (z_ref - p[2])) > 1e-3


def test_column_walk_skips_non_floor_and_returns_the_first_different_floor():
    w, h = 8, 16
    planes = [GROUND, GROUND, (0, 0, -1, 2.3), WALL, (0, 0, 1, 3.0), (0, 0, -1, 2.45)]
    idx = [depth.SKY] * (w * h)
    for r in range(h // 2, h):
        for c in range(w):
            idx[r * w + c] = 1
    for r in range(0, 7):
        idx[r * w + 0] = 4                  # plane 4 lives above the horizon: not a floor
    col = 3
    for r, j in ((8, 2), (9, 3), (10, 2), (11, 4), (12, 5), (13, 1)):
        idx[r * w + col] = j
    for r, j in ((12, 2), (13, 3), (14, 3), (15, 3)):
        idx[r * w + 6] = j
    pix = dad.PayloadIndex(depth.parse(build_payload(w, h, planes, idx)))
    assert 4 not in pix.floor and 3 not in pix.floor and {1, 2, 5} <= pix.floor
    assert dad.local_reference(pix, 8, col, 2) == (5, 4)
    assert dad.local_reference(pix, 12, 6, 2) == (None, None)


def _gt(*cities):
    return {f'c{i}': {'true_n': n, 'true_surface': k, 'false_n': fn,
                      'false_non_surface': fk}
            for i, (n, k, fn, fk) in enumerate(cities)}


@pytest.mark.parametrize('cities,expected', [
    ([(100, 85, 10, 1), (100, 85, 10, 1)], 'SUPPORTED'),        # pooled exactly 0.85
    ([(100, 84, 10, 1)], 'NOT SUPPORTED'),
    ([(200, 190, 10, 1), (100, 80, 10, 1)], 'SUPPORTED'),       # a city exactly at 0.80
    ([(200, 190, 10, 1), (100, 79, 10, 1)], 'NOT SUPPORTED'),   # pooled 0.90, a city 0.79
    ([(100, 70, 10, 1)], 'NOT SUPPORTED'),                      # exactly 0.70
    ([(100, 69, 10, 1)], 'UNDERCUT'),
])
def test_verdict_rule_i_boundaries(cities, expected):
    assert dad.verdict(_gt(*cities))['rule_i'] == expected


@pytest.mark.parametrize('false_ns,expected', [(30, 'SUPPORTED'), (29, 'NOT SUPPORTED')])
def test_verdict_rule_ii_margin_is_twenty_points(false_ns, expected):
    # True non-surface share 10/100; False 30/100 is exactly 20 points above it.
    v = dad.verdict(_gt((100, 90, 100, false_ns)))
    assert v['rule_ii'] == expected


def test_verdict_rule_ii_needs_separated_intervals_and_flags_underpower():
    big = dad.verdict(_gt((1000, 900, 100, 60)))
    assert big['rule_ii'] == 'SUPPORTED' and big['underpowered'] is False
    # n = 49, the pooled verdict-False count: a 40-point gap still carries a wide interval
    small = dad.verdict(_gt((1000, 900, 49, 25)))
    assert small['underpowered'] is True
    # a large gap whose intervals overlap is not enough
    tiny = dad.verdict(_gt((20, 16, 5, 3)))
    assert tiny['non_surface_diff'] >= 0.20 and tiny['rule_ii'] == 'NOT SUPPORTED'


@pytest.mark.parametrize('median,expected', [
    (0.12, 'consistent'), (0.02, 'not consistent'),
    # the knife-edge the real result sits on (0.0498): the band is [0.05, 0.30)
    (0.05, 'consistent'), (0.0499, 'not consistent'),
    (0.2999, 'consistent'), (0.30, 'not consistent')])
def test_offset_claim_is_read_descriptively(median, expected):
    v = dad.verdict(_gt((10, 9, 1, 0)), (median, median - 0.01, median + 0.01))
    assert v['offset_claim_0p15'] == expected


def test_payload_index_refuses_a_drifted_dominant_plane(monkeypatch):
    pix = _index()
    other = pix.payload.planes[2]                       # the patch, not the ground
    monkeypatch.setattr(dad.gp, 'dominant_ground', lambda payload: other)
    with pytest.raises(RuntimeError, match='drifted'):
        dad.PayloadIndex(pix.payload)


def test_no_plane_near_wraps_the_seam_and_clamps_rows():
    w, h = 8, 4
    idx = [1] * (w * h)
    idx[2 * w + (w - 1)] = depth.SKY                    # row 2, the LAST column
    pix = dad.PayloadIndex(depth.parse(build_payload(w, h, [GROUND, GROUND], idx)))
    assert dad.no_plane_near(pix, 2, 0) == 1            # column 0's left neighbour wraps
    assert dad.no_plane_near(pix, 2, 3) == 0
    assert dad.no_plane_near(pix, 0, w - 1) == 0        # row 0: row -1 clamped, row 2 unseen
    assert dad.no_plane_near(pix, 3, w - 1) == 1        # row 3: row 4 clamped, row 2 seen


def test_is_level_floor_tests_the_normal_not_the_distance():
    assert dad.is_level_floor({'plane_class': dad.FLOOR, 'plane_tilt_deg': 0.0})
    assert not dad.is_level_floor({'plane_class': dad.FLOOR, 'plane_tilt_deg': 1e-7})
    assert not dad.is_level_floor({'plane_class': dad.GROUND, 'plane_tilt_deg': 0.0})
    assert not dad.is_level_floor({'plane_class': dad.FLOOR, 'plane_tilt_deg': None})
    # and the level patch of the fixture is one, measured through classify
    assert dad.is_level_floor(dad.classify(_index(), *_xy(9, 21)))


@pytest.mark.parametrize('exc', [zlib.error('invalid stored block lengths'),
                                 struct.error('unpack_from requires a buffer'),
                                 EOFError(), OSError('bad gzip')])
def test_an_unreadable_payload_is_unparsed_never_raised(monkeypatch, exc):
    def boom(path):
        raise exc
    monkeypatch.setattr(dad, 'read_payload', boom)
    status, fields = dad.measure_pano(('x.json.gz', [(0, 0.5, 0.7), (3, 0.1, 0.8)]))
    assert status == depth.UNPARSED and fields == [(0, {}), (3, {})]


def test_a_corrupt_deflate_stream_is_unparsed(tmp_path):
    import gzip
    path = tmp_path / 'p.json.gz'
    raw = bytearray(gzip.compress(b'{"depth_b64": "' + b'A' * 4000 + b'"}'))
    for i in range(12, len(raw) - 8):                  # scramble the deflate body only
        raw[i] ^= 0x5A
    path.write_bytes(bytes(raw))
    status, fields = dad.measure_pano((str(path), [(0, 0.5, 0.7)]))
    assert status == depth.UNPARSED and fields == [(0, {})]


def test_summaries_only_refuses_a_partial_detections_file():
    cov = [{'city': 'a', 'n_detections_0p30': 3}, {'city': 'b', 'n_detections_0p30': 2}]
    dad.check_row_counts({'a': [{}] * 3, 'b': [{}] * 2}, cov)
    with pytest.raises(SystemExit, match='coverage.csv'):
        dad.check_row_counts({'a': [{}] * 3, 'b': [{}]}, cov)
    with pytest.raises(SystemExit):
        dad.check_row_counts({'c': [{}]}, cov)             # a city coverage never saw


def test_limit_writes_its_own_file(tmp_path, monkeypatch):
    """`measure --limit` must not overwrite the full run's detections.csv."""
    (tmp_path / 'runs' / 'x').mkdir(parents=True)
    out = tmp_path / 'out'
    (out / 'x' / 'depth_at_detection').mkdir(parents=True)
    full = out / 'x' / 'depth_at_detection' / 'detections.csv'
    full.write_text('the full run\n')
    pano = types.SimpleNamespace(pano_id='p1', detections=[(0, 0.5, 0.7, 0.9)],
                                 camera_height_m=None, capture_date='2024-05')
    monkeypatch.setattr(dad.fs, 'load_results', lambda path: ([pano], None))
    monkeypatch.setattr(dad.fs, 'pano_pose', lambda p, mode: None)
    monkeypatch.setattr(dad.geo, 'detection_ground_point', lambda *a, **k: None)
    args = types.SimpleNamespace(cities=['x'], run_root=tmp_path / 'runs', out_root=out,
                                 limit=1, workers=1, summaries_only=False)
    dad.cmd_measure(args)
    assert full.read_text() == 'the full run\n'
    rows = dad.read_detections(out / 'x' / 'depth_at_detection' / 'detections.limit.csv')
    assert [(r['pano_id'], r['payload_status']) for r in rows] == [('p1', dad.NO_FILE)]
    assert not (out / '_summary').exists()


def _floor_row(level, d, off, off_level, ref_d=2.36, ref_tilt=1.8):
    return {'plane_class': dad.FLOOR, 'plane_tilt_deg': 0.0 if level else 2.0,
            'plane_d_m': d, 'offset_local_m': off, 'offset_local_level_ref_m': off_level,
            'ref_d_m': ref_d, 'ref_tilt_deg': ref_tilt, 'rows_to_ref': 12,
            'range_depth_m': 10.0}


def test_level_floor_rows_count_the_exact_stand_in_distance():
    rows = [_floor_row(True, 2.5, 0.12, -0.14), _floor_row(True, 2.5, 0.14, -0.14),
            _floor_row(True, 2.4999999, 0.1, -0.1), _floor_row(False, 2.3, 0.03, 0.03)]
    by = {r['split']: r for r in dad.level_floor_rows('0.55', 'c', rows)}
    assert by['all']['n_floor'] == 4 and by['level']['n_floor'] == 3
    assert by['level']['n_plane_d_exactly_2p5'] == 2        # 2.4999999 is not 2.5
    assert by['nonlevel']['n_plane_d_exactly_2p5'] == 0
    assert by['level']['offset_ref_tilt_term_p50'] == pytest.approx(0.26)


def test_plane_distance_survives_the_csv_round_trip(tmp_path):
    rows = [{'pano_id': '0012e', 'det_index': 1, 'plane_d_m': 2.4999999, 'ref_d_m': 2.5,
             'plane_tilt_deg': 0.0, 'plane_class': dad.FLOOR}]
    dad.write_rows(tmp_path / 'd.csv', rows, dad.DET_FIELDS)
    back = dad.read_detections(tmp_path / 'd.csv')[0]
    assert back['plane_d_m'] == 2.4999999 and back['ref_d_m'] == 2.5
    assert back['pano_id'] == '0012e' and dad.is_level_floor(back)


def test_figure_errorbar_skips_an_empty_gt_group():
    assert dad.surface_errorbar({'n': 0.0, 'share_surface': None, 'surface_lo': None,
                                 'surface_hi': None}) is None
    assert dad.surface_errorbar({'n': None, 'share_surface': None}) is None
    s, lo, hi = dad.surface_errorbar({'n': 10.0, 'share_surface': 0.9, 'surface_lo': 0.6,
                                      'surface_hi': 0.98})
    assert (s, lo, hi) == pytest.approx((0.9, 0.3, 0.08))


def test_figures_draw_with_an_empty_gt_group(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    import shutil
    import mapillary_tilt as mt
    fig_dir = tmp_path / 'fig'
    (fig_dir / 'data').mkdir(parents=True)
    for f in (dad.REPO_ROOT / 'docs' / 'figures' / 'depth-at-detection' / 'data').iterdir():
        shutil.copy2(f, fig_dir / 'data' / f.name)
    rows = mt._read_csv(fig_dir / 'data' / 'gt_classes.csv')
    for r in rows:                                      # empty the pooled measured False group
        if r['city'] == 'pooled' and r['subset'] == 'measured' and r['gt_group'] == 'false':
            for k in r:
                if k not in ('city', 'gt_group', 'subset'):
                    r[k] = 0 if k.startswith('n') else None
    dad.write_rows(fig_dir / 'data' / 'gt_classes.csv', rows, list(rows[0]))
    monkeypatch.setattr(dad, 'FIG_DIR', fig_dir)
    args = types.SimpleNamespace(cities=list(dad.DEFAULT_CITIES), out_root=tmp_path / 'none')
    dad.cmd_figures(args)
    assert (fig_dir / 'fig4_gt.png').stat().st_size > 0


def test_doctests_run():
    import doctest
    assert doctest.testmod(dad).failed == 0
