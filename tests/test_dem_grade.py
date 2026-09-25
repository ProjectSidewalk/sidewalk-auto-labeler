"""dem_grade (#51): the DEM tile grid, sampler and grades, the grade-source plumbing in
fuse_sites.load_results, and the pre-registered #51 rule. Synthetic rasters only; the
network is blocked by conftest and faked here where a fetch is exercised."""
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image, TiffImagePlugin

import dem_grade as dg
import eval_sites as es
import fuse_sites as fs
import geo
from test_fuse_sites import FRAME, _mapillary_line

LAT0, LNG0 = FRAME.lat0, FRAME.lng0
M_PER_DEG = geo.EARTH_RADIUS_M * math.pi / 180.0      # haversine's, along a meridian


def _plane_mosaic(fn, lat_span=0.01, lng_span=0.01, step=2e-5):
    """A Mosaic whose pixel centres hold fn(lat, lng)."""
    x0, y1 = LNG0 - lng_span / 2, LAT0 + lat_span / 2
    h, w = int(lat_span / step), int(lng_span / step)
    lat_c = y1 - (np.arange(h) + 0.5) * step
    lng_c = x0 + (np.arange(w) + 0.5) * step
    return dg.Mosaic(fn(lat_c[:, None], lng_c[None, :]).astype(np.float64), x0, y1, step)


def test_tile_grid_covers_the_padded_bbox_exactly_once_in_capped_tiles():
    bbox = (-77.47, 37.52, -77.40, 37.56)
    g = dg.tile_grid(bbox, posts_m=2.0)
    assert g['width'] > 1500 and g['height'] > 1500       # really several tiles
    cover = np.zeros((g['height'], g['width']), dtype=int)
    for t in g['tiles']:
        w, h = t['size']
        assert 0 < w <= dg.MAX_TILE_PX and 0 < h <= dg.MAX_TILE_PX
        r0, c0 = t['px_offset']
        cover[r0:r0 + h, c0:c0 + w] += 1
        # the tile's bbox is its pixel range on the one global grid
        assert t['bbox'][0] == pytest.approx(g['x0'] + c0 * g['step_deg'], abs=1e-12)
        assert t['bbox'][3] == pytest.approx(g['y1'] - r0 * g['step_deg'], abs=1e-12)
    assert (cover == 1).all()
    s = g['step_deg']
    assert g['x0'] < bbox[0] and g['y1'] > bbox[3]
    assert g['x0'] + g['width'] * s > bbox[2] and g['y1'] - g['height'] * s < bbox[1]


def test_bilinear_sampling_of_a_plane_returns_the_plane():
    fn = lambda lat, lng: 50.0 + 3000.0 * (lat - LAT0) - 1200.0 * (lng - LNG0)  # noqa: E731
    m = _plane_mosaic(fn)
    rng = np.random.default_rng(0)
    lats = LAT0 + rng.uniform(-0.004, 0.004, 50)
    lngs = LNG0 + rng.uniform(-0.004, 0.004, 50)
    assert np.allclose(m.sample(lats, lngs), fn(lats, lngs), atol=1e-6)
    assert np.isnan(m.sample([LAT0 + 1.0], [LNG0]))[0]            # off the raster


def _north_sequence(n=10, spacing_m=5.0, slope=0.05, seq='s'):
    """Frames driving north up a `slope` road, one per second; SfM altitude on it too."""
    frames = []
    for i in range(n):
        lat = LAT0 + i * spacing_m / M_PER_DEG
        frames.append((f'p{i}', seq, i * 1000, lat, LNG0,
                       100.0 + slope * i * spacing_m, 'GoPro Max'))
    return frames


def test_a_5_percent_slope_gives_2_86_deg_from_both_dem_grades_and_sfm_matches_production():
    fn = lambda lat, lng: 100.0 + 0.05 * (lat - LAT0) * M_PER_DEG + 0 * lng  # noqa: E731
    m = _plane_mosaic(fn)
    frames = _north_sequence()
    dem_z = [float(v) for v in m.sample([f[3] for f in frames], [f[4] for f in frames])]
    rows = dg.compute_grades(frames, dem_z)
    want = math.degrees(math.atan(0.05))
    assert want == pytest.approx(2.862, abs=1e-3)
    for r in rows:
        assert r['grade_dem_2pt_deg'] == pytest.approx(want, abs=1e-3)
        assert r['grade_dem_deg'] == pytest.approx(want, abs=1e-3)
        assert r['grade_sfm_smoothed_deg'] == pytest.approx(want, abs=1e-6)
        assert r['travel_bearing_deg'] == pytest.approx(0.0, abs=1e-6)
    assert rows[5]['n_frames_fit'] == 9 and rows[5]['baseline_m'] == pytest.approx(40.0)
    # the SfM column IS production's grade: the same sequence_grades call, same frames
    prod = fs.sequence_grades([(i, f[1], f[2], f[3], f[4], f[5]) for i, f in enumerate(frames)])
    assert [r['grade_sfm_deg'] for r in rows] == [prod[i][0] for i in range(len(frames))]


def _write_run(tmp_path, n=4):
    run = tmp_path / 'city'
    (run / 'dem').mkdir(parents=True)
    rise = 5.0 * math.tan(math.radians(3.0))
    lines = [_mapillary_line(f'm{i}', 5.0 * i, 100.0 + rise * i, 2 * i, 'seq', 3.0)
             for i in range(n)] + [_mapillary_line('lone', 500.0, 100.0, 90, 'other', 0.0)]
    (run / 'results.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return run


def test_grades_csv_sfm_column_reproduces_load_results_bit_for_bit(tmp_path):
    run = _write_run(tmp_path)
    frames = dg.read_frames(run / 'results.jsonl')
    rows = dg.compute_grades(frames, [100.0 + i for i in range(len(frames))])
    dg.write_grades(run / 'dem' / 'grades.csv', rows)
    panos, _ = fs.load_results(run / 'results.jsonl')
    with open(run / 'dem' / 'grades.csv', newline='', encoding='utf-8') as f:
        csv_sfm = {r['panorama_id']: r['grade_sfm_deg'] for r in csv.DictReader(f)}
    for p in panos:
        assert (float(csv_sfm[p.pano_id]) if csv_sfm[p.pano_id] else None) == p.grade_deg


def test_load_results_replaces_the_grade_from_the_csv_and_keeps_the_bearing(tmp_path):
    run = _write_run(tmp_path, n=3)
    (run / 'dem' / 'grades.csv').write_text(
        'panorama_id,grade_dem_deg,grade_sfm_smoothed_deg\n'
        'm0,1.5,2.5\nm1,-0.25,2.75\nm2,,\n', encoding='utf-8')
    base = {p.pano_id: p for p in fs.load_results(run / 'results.jsonl')[0]}
    dem = {p.pano_id: p for p in fs.load_results(run / 'results.jsonl',
                                                 grade_source=fs.GRADE_DEM)[0]}
    assert (dem['m0'].grade_deg, dem['m1'].grade_deg) == (1.5, -0.25)
    assert dem['m2'].grade_deg is None                   # no value -> gravity fallback
    assert dem['lone'].grade_deg is None and base['lone'].grade_deg is None
    for pid in ('m0', 'm1', 'm2'):
        assert base[pid].grade_deg is not None
        assert dem[pid].travel_bearing_deg == base[pid].travel_bearing_deg
    counts = fs.pose_counts(list(dem.values()),
                            fs.FuseParams(apply_pose=fs.POSE_ROAD, grade_source=fs.GRADE_DEM))
    assert (counts['grade_source'], counts['grade_replaced']) == ('dem', 2)
    assert (counts['road_relative'], counts['gravity_fallback']) == (2, 2)
    assert fs.pose_counts(list(base.values()), fs.FuseParams())['grade_replaced'] == 0
    smooth = fs.load_results(run / 'results.jsonl', grade_source=fs.GRADE_SFM_SMOOTHED)[0]
    assert [p.grade_deg for p in smooth][:3] == [2.5, 2.75, None]
    # a missing file refuses, and names the command that writes it
    (run / 'dem' / 'grades.csv').unlink()
    with pytest.raises(SystemExit, match='dem_grade.py'):
        fs.load_results(run / 'results.jsonl', grade_source=fs.GRADE_DEM)
    with pytest.raises(ValueError):
        fs.FuseParams(grade_source='lidar')


# --- fetch --------------------------------------------------------------------------

class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status


def _geotiff(tile, step, value=42.0):
    w, h = tile['size']
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33922] = (0.0, 0.0, 0.0, tile['bbox'][0], tile['bbox'][3], 0.0)
    ifd[33550] = (step, step, 0.0)
    ifd.tagtype[33922] = ifd.tagtype[33550] = 12
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w), value, np.float32), 'F').save(
        buf, format='TIFF', tiffinfo=ifd)
    return buf.getvalue()


def _small_grid():
    return dg.tile_grid((LNG0, LAT0, LNG0 + 0.001, LAT0 + 0.001), posts_m=2.0)


def test_a_non_image_response_is_refused_and_never_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, 'get',
                        lambda *a, **k: _Resp(b'<html>Service Unavailable</html>'))
    g = _small_grid()
    with pytest.raises(dg.FetchRefused, match='not an image'):
        dg.fetch_city(tmp_path, g, (0, 0, 0, 0), 2.0, pause=lambda s: None, log=lambda m: None)
    assert not list((tmp_path / 'tiles').glob('*'))
    assert not (tmp_path / 'tiles.json').exists()


def test_a_valid_tile_is_cached_hashed_and_reused(tmp_path, monkeypatch):
    g = _small_grid()
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        tile = next(t for t in g['tiles']
                    if ','.join(repr(v) for v in t['bbox']) == params['bbox'])
        return _Resp(_geotiff(tile, g['step_deg']))
    monkeypatch.setattr(requests, 'get', fake_get)
    quiet = dict(pause=lambda s: None, log=lambda m: None)
    manifest = dg.fetch_city(tmp_path, g, (0, 0, 0, 0), 2.0, **quiet)
    assert len(calls) == len(g['tiles']) == 1
    assert calls[0]['pixelType'] == 'F32' and calls[0]['imageSR'] == 4326
    assert dg.verify_city(tmp_path) == []
    m = dg.Mosaic.from_cache(tmp_path, manifest)
    assert m.sample([LAT0 + 0.0005], [LNG0 + 0.0005])[0] == pytest.approx(42.0)
    dg.fetch_city(tmp_path, g, (0, 0, 0, 0), 2.0, **quiet)        # all reused
    assert len(calls) == 1
    (tmp_path / 'tiles' / '0_0.tif').write_bytes(b'corrupt')
    assert dg.verify_city(tmp_path)                                # mismatch reported
    # a tile georeferenced somewhere else is refused
    shifted = dict(g['tiles'][0], bbox=[v + 0.01 for v in g['tiles'][0]['bbox']])
    with pytest.raises(dg.FetchRefused, match='georeference'):
        dg.decode_tile(_geotiff(shifted, g['step_deg']), g['tiles'][0], g['step_deg'])


# --- the #51 rule -------------------------------------------------------------------

def _city(dem=(1.0, 2.0), dem_shuf=(1.2, 2.2), rec=(0.90, 0.90), unpl=(10, 10), pool=100,
          off=(1.5, 3.0)):
    base = {'off_pool_ramps': pool}
    return [dict(base, arm='off', median_gt_to_site_m=off[0], p90_gt_to_site_m=off[1],
                 recall_off_pool_2p5m=rec[0], gt_marks_unplaceable=unpl[0]),
            dict(base, arm='road-dem', median_gt_to_site_m=dem[0], p90_gt_to_site_m=dem[1],
                 recall_off_pool_2p5m=rec[1], gt_marks_unplaceable=unpl[1]),
            dict(base, arm='road-dem-shuffled-within', median_gt_to_site_m=dem_shuf[0],
                 p90_gt_to_site_m=dem_shuf[1], recall_off_pool_2p5m=rec[1],
                 gt_marks_unplaceable=unpl[1])]


def test_dem_rule_passes_only_on_all_four_clauses():
    good = {c: _city() for c in 'abcde'}
    ok, clauses, reasons = es.dem_verdict(good)
    assert ok and clauses == {'i': True, 'ii': True, 'iii': True, 'iv': True}
    assert 'road-dem beats road-dem-shuffled-within' in '\n'.join(reasons)
    # (i): the DEM shuffle ties the DEM grade in two cities
    ok, c, _ = es.dem_verdict(dict(good, d=_city(dem_shuf=(1.05, 2.5)),
                                   e=_city(dem_shuf=(1.0, 2.0))))
    assert not ok and c == {'i': False, 'ii': True, 'iii': True, 'iv': True}
    # (ii): 1.1 points of off-pool recall lost in one city
    ok, c, _ = es.dem_verdict(dict(good, a=_city(rec=(0.900, 0.889))))
    assert not ok and c == {'i': True, 'ii': False, 'iii': True, 'iv': True}
    # (iii): six more unplaceable marks on a 100-ramp pool (limit 5)
    ok, c, _ = es.dem_verdict(dict(good, a=_city(unpl=(10, 16))))
    assert not ok and c == {'i': True, 'ii': True, 'iii': False, 'iv': True}
    # (iv): p90 worse than off by more than 0.1 m in one city
    ok, c, _ = es.dem_verdict(dict(good, a=_city(dem=(1.0, 3.2), dem_shuf=(1.2, 3.4))))
    assert not ok and c == {'i': True, 'ii': True, 'iii': True, 'iv': False}


def test_default_rule_arguments_reproduce_the_committed_42_verdict():
    """control_verdict / precondition_verdict with their defaults still read #74's
    five-arm table exactly as it was published: first rule PASS, control FAIL on (i),(ii)."""
    path = (Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'mapillary-tilt'
            / 'data' / 'pose_precondition.csv')
    by_city = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            row = {k: (v if k in ('city', 'arm') else float(v) if v else None)
                   for k, v in r.items()}
            row['gt_marks_unplaceable'] = int(row['gt_marks_unplaceable'])
            by_city.setdefault(row['city'], []).append(row)
    assert es.precondition_verdict(by_city)[0]
    ok, clauses, reasons = es.control_verdict(by_city)
    assert not ok and clauses == {'i': False, 'ii': False, 'iii': True}
    assert reasons[0].startswith('richmond: shuffled-within minus road p90 +0.344 m')
