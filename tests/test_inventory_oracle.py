"""inventory_oracle (#79): paging, the frozen refit, the arms, the anchored pool, the sign
convention and the pre-registered verdict -- all offline and synthetic."""
import json
import math
from types import SimpleNamespace

import pytest

import geo
import fuse_sites as fs
import eval_sites as es
import inventory_oracle as io
from test_fuse_sites import FRAME, make_pano


# ----------------------------------------------------------------------------- fetch

LAYER = 'https://services2.arcgis.com/x/arcgis/rest/services/y/FeatureServer/3'


def _feat(oid, lng, lat, status='Active'):
    return {'type': 'Feature', 'id': oid, 'geometry': {'type': 'Point',
                                                       'coordinates': [lng, lat]},
            'properties': {'OBJECTID': oid, 'LIFECYCLE': status, 'CORNER': 'NE'}}


def _server(features, page=2, break_at=None):
    """A fake ArcGIS layer: maxRecordCount `page`; `break_at` = an offset at which it
    claims exceededTransferLimit but returns nothing."""
    calls = []

    def get(url, params):
        calls.append(params)
        if url == LAYER:
            return {'name': 'CurbRamp', 'maxRecordCount': page, 'objectIdField': 'OBJECTID'}
        if params.get('returnCountOnly'):
            return {'count': len(features) + (1 if 'geometry' not in params else 0)}
        off = params['resultOffset']
        if break_at is not None and off >= break_at:
            return {'type': 'FeatureCollection', 'features': [],
                    'properties': {'exceededTransferLimit': True}}
        got = features[off:off + params['resultRecordCount']]
        return {'type': 'FeatureCollection', 'features': got,
                'properties': {'exceededTransferLimit': off + len(got) < len(features)}}
    return get, calls


def test_fetch_pages_with_result_offset_until_the_count_is_reached():
    feats = [_feat(i, -82.33 + i * 1e-4, 29.64) for i in range(1, 6)]
    get, calls = _server(feats, page=2)
    got, info = io.fetch_features(LAYER, (-83, 29, -82, 30), get=get, pause_s=0)
    assert [f['id'] for f in got] == [1, 2, 3, 4, 5]
    offsets = [c['resultOffset'] for c in calls if 'resultOffset' in c]
    assert offsets == [0, 2, 4]
    assert info['pages'] == 3 and info['in_bbox'] == 5 and info['layer_total'] == 6
    assert all(c['orderByFields'] == 'OBJECTID ASC' for c in calls if 'resultOffset' in c)


def test_fetch_refuses_an_unresolved_transfer_limit():
    feats = [_feat(i, -82.33, 29.64) for i in range(1, 6)]
    get, _ = _server(feats, page=2, break_at=2)
    with pytest.raises(RuntimeError, match='did not resolve'):
        io.fetch_features(LAYER, (-83, 29, -82, 30), get=get, pause_s=0)


def test_fetch_refuses_a_short_pull():
    feats = [_feat(i, -82.33, 29.64) for i in range(1, 4)]
    get, _ = _server(feats, page=2)

    def lying(url, params):
        if params.get('returnCountOnly') and 'geometry' in params:
            return {'count': 7}
        return get(url, params)
    with pytest.raises(RuntimeError, match='partial pull'):
        io.fetch_features(LAYER, (-83, 29, -82, 30), get=lying, pause_s=0)


def test_http_get_refuses_other_hosts():
    with pytest.raises(ValueError, match='not an inventory host'):
        io.http_get_json('https://example.com/query', {'f': 'json'})


def test_fetch_city_filters_to_the_area_and_caches(tmp_path, monkeypatch):
    run = tmp_path / 'runs' / 'gainesville'
    run.mkdir(parents=True)
    square = {'type': 'Polygon', 'coordinates': [[[-82.34, 29.63], [-82.32, 29.63],
                                                  [-82.32, 29.65], [-82.34, 29.65],
                                                  [-82.34, 29.63]]]}
    (run / 'area.geojson').write_text(json.dumps(square), encoding='utf-8')
    feats = [_feat(1, -82.33, 29.64), _feat(2, -82.331, 29.641, 'Retired'),
             _feat(3, -82.30, 29.64)]            # 3: in the bbox query answer, outside
    get, _ = _server(feats, page=2)
    monkeypatch.setattr(io, 'PAGE_PAUSE_S', 0)
    monkeypatch.setitem(io.INVENTORIES['gainesville'], 'url', LAYER)
    rec = io.fetch_city('gainesville', get=get, runs_root=tmp_path / 'runs',
                        out=tmp_path / 'out')
    assert (rec['in_bbox'], rec['in_area'], rec['kept']) == (3, 2, 1)
    assert rec['histograms_in_area']['LIFECYCLE'] == {'Active': 1, 'Retired': 1}
    pts, _ = io.load_inventory('gainesville', out=tmp_path / 'out')
    assert [(k, lat, lng) for k, lat, lng, _ in pts] == [(0, 29.64, -82.33)]

    def offline(url, params):
        raise AssertionError('a cached snapshot must not touch the network')
    again = io.fetch_city('gainesville', get=offline, runs_root=tmp_path / 'runs',
                          out=tmp_path / 'out')
    assert again['sha256'] == rec['sha256']
    # a cache is bound to the area it was pulled for: an edited area.geojson is refused
    bigger = {'type': 'Polygon', 'coordinates': [[[-82.35, 29.63], [-82.32, 29.63],
                                                  [-82.32, 29.65], [-82.35, 29.65],
                                                  [-82.35, 29.63]]]}
    (run / 'area.geojson').write_text(json.dumps(bigger), encoding='utf-8')
    with pytest.raises(SystemExit, match='the area changed'):
        io.fetch_city('gainesville', get=offline, runs_root=tmp_path / 'runs',
                      out=tmp_path / 'out')
    (run / 'area.geojson').write_text(json.dumps(square), encoding='utf-8')
    # an edited cache is refused, not silently scored
    p = tmp_path / 'out' / 'gainesville' / 'inventory.geojson'
    p.write_bytes(p.read_bytes() + b' ')
    with pytest.raises(SystemExit, match='sha256'):
        io.load_inventory('gainesville', out=tmp_path / 'out')


# ------------------------------------------------------------------------------ arms

def _slim(pid, year, height):
    return fs.SlimPano(pid, 40.0, -74.0, 0.0, None, None, year and f'{year}-06', 'launch',
                       [], camera_height_m=height)


def test_per_rig_assigns_by_vintage_median_including_unmeasured_panos():
    panos = [_slim('l1', 2025, 1.8), _slim('l2', 2025, 1.8), _slim('l3', 2025, None),
             _slim('h1', 2024, 2.4), _slim('h2', 2024, 2.4), _slim('h3', 2024, None),
             _slim('u', None, 1.8), _slim('n', 2019, None)]
    out, rig = io.per_rig_panos(panos)
    h = {p.pano_id: p.camera_height_m for p in out}
    assert h == {'l1': 2.0, 'l2': 2.0, 'l3': 2.0, 'h1': 2.5, 'h2': 2.5, 'h3': 2.5,
                 'u': None, 'n': None}
    assert rig == {'2025': (1.8, 2.0), '2024': (2.4, 2.5)}
    assert all(p.camera_height_spread_m is None for p in out)
    # under PER_RIG an unassigned pano raycasts at the 2.6 m default
    pose = geo.pano_pose(next(p for p in out if p.pano_id == 'u').pose_fields())
    assert geo.camera_height_for(pose, camera_height=geo.PER_RIG)[0] == 2.6


def test_scaled_panos_scale_measured_heights_only():
    out = io.scaled_panos([_slim('m', 2024, 2.0), _slim('u', 2024, None)], 1.08)
    assert out[0].camera_height_m == pytest.approx(2.16)
    assert out[1].camera_height_m is None


# ------------------------------------------------------------------- frozen refit

def test_frozen_refit_moves_the_site_along_the_ray_by_the_height_ratio():
    pano = make_pano('p', 0, 0, [(0, 10, 0.9)], heading_deg=0.0)   # target 10 m north
    arms = {'hi': ([pano], fs.FuseParams(camera_height_m=2.6)),
            'lo': ([pano], fs.FuseParams(camera_height_m=2.0))}
    sites, frame, _ = fs.fuse([pano], arms['hi'][1])
    kept, pos, members = es.refit_frozen(sites, frame, io.placer(arms), ('hi', 'lo'))
    assert len(kept) == 1
    pe, pn = frame.to_enu(pano.lat, pano.lng)
    r_hi = math.hypot(pos['hi'][0].e - pe, pos['hi'][0].n - pn)
    r_lo = math.hypot(pos['lo'][0].e - pe, pos['lo'][0].n - pn)
    assert r_hi == pytest.approx(10.0, abs=1e-6)
    assert r_lo / r_hi == pytest.approx(2.0 / 2.6, abs=1e-9)
    assert members['lo'][0][0].range_m == pytest.approx(10.0 * 2.0 / 2.6)
    # the refit under the association's own arm IS the fused position
    assert (pos['hi'][0].e, pos['hi'][0].n) == pytest.approx((sites[0].e, sites[0].n))


def test_frozen_refit_drops_a_site_any_arm_cannot_place():
    near = make_pano('near', 0, 0, [(0, 10, 0.9)], heading_deg=0.0)
    far = make_pano('far', 100, 0, [(100, 24, 0.9)], heading_deg=0.0)   # 24 m at 2.6
    arms = {'a': ([near, far], fs.FuseParams()),
            'tall': ([near, far], fs.FuseParams(camera_height_m=2.9))}   # 26.8 m > cap
    sites, frame, _ = fs.fuse([near, far], arms['a'][1])
    kept, pos, _ = es.refit_frozen(sites, frame, io.placer(arms), ('a', 'tall'))
    assert [s.members[0][0].pano_id for s in kept] == ['near']
    assert [p.id for p in pos['a']] == [kept[0].id]


def test_frozen_refit_drops_and_counts_a_site_with_no_operational_member():
    """A site with nothing to refit from is dropped (not a ZeroDivisionError in
    geo.sym2_inv), and counted when the caller asks."""
    pano = make_pano('p', 0, 0, [(0, 10, 0.9)], heading_deg=0.0)
    arms = {'a': ([pano], fs.FuseParams())}
    sites, frame, _ = fs.fuse([pano], arms['a'][1])
    empty = SimpleNamespace(id=-1, members=[(d, r) for d, r in sites[0].members
                                            if not d.operational])
    counts = {}
    kept, pos, _ = es.refit_frozen([empty, sites[0]], frame, io.placer(arms), ('a',),
                                   counts=counts)
    assert [s.id for s in kept] == [sites[0].id]
    assert [p.id for p in pos['a']] == [sites[0].id]
    assert counts == {'no_operational_members': 1}
    kept, _, _ = es.refit_frozen([empty], frame, io.placer(arms), ('a',))   # no counts
    assert kept == []


# ------------------------------------------------------------------- along-ray sign

def _ge(bearing, rng):
    return geo.GroundEstimate(0.0, 0.0, rng, bearing, 0.1, 0.1)


def test_along_ray_is_positive_when_the_site_lies_beyond_the_point():
    # camera looking north (bearing 0): a site 1 m north of the point = ranges long
    assert io.along_ray(0.0, 11.0, 0.0, 10.0, [_ge(0.0, 11.0)]) == (pytest.approx(1.0), 11.0)
    assert io.along_ray(0.0, 9.0, 0.0, 10.0, [_ge(0.0, 9.0)])[0] == pytest.approx(-1.0)
    # an across-ray offset does not count
    assert io.along_ray(3.0, 10.0, 0.0, 10.0, [_ge(0.0, 10.0)])[0] == pytest.approx(0.0)
    # exactly opposite rays cancel: no direction to project on
    assert io.along_ray(0.0, 0.0, 0.0, 0.0, [_ge(0.0, 5.0), _ge(180.0, 5.0)]) is None


# ------------------------------------------------------------ score: anchored pool

def test_score_anchors_the_pool_on_a_and_reads_the_same_site_under_every_arm():
    """Cameras really at 2.0 m (measured), 2026 imagery. Under (a) 2.6 m every range runs
    30% long; (c) puts the vintage (median 2.0 < 2.1) at 2.0 m, i.e. exact."""
    targets = [(0.0, 8.0), (40.0, 10.0), (80.0, 6.0)]
    panos = [make_pano(f'p{k}', te, 0.0, [(te, tn, 0.9)], heading_deg=0.0,
                       capture='2026-04', height=2.0) for k, (te, tn) in enumerate(targets)]
    inventory = [(k, *FRAME.to_latlng(te, tn), {}) for k, (te, tn) in enumerate(targets)]
    res = io.score_city('synthetic', radii=(2.5, 5.0), panos=panos, inventory=inventory)
    rows = {(r['frame'], r['arm'], r['radius_m']): r for r in res['rows']}
    a5 = rows['frozen@a', 'a', 5.0]
    c5 = rows['frozen@a', 'c', 5.0]
    b5 = rows['frozen@a', 'b', 5.0]
    assert a5['n_pool'] == c5['n_pool'] == 3
    assert a5['median_m'] == pytest.approx(0.3 * 8.0)          # 2.6/2.0 - 1 of 8 m
    assert a5['p90_m'] == pytest.approx(0.3 * 10.0)
    assert c5['p90_m'] == pytest.approx(0.0, abs=1e-6)
    assert b5['median_m'] == pytest.approx(0.08 * 8.0, abs=1e-6)  # x1.08 of the truth
    # at 2.5 m (a) matches only the 6 and 8 m ramps (1.8, 2.4 m off); its pool is what
    # every arm is scored on, while (c) matches all three on its own
    a25, c25 = rows['frozen@a', 'a', 2.5], rows['frozen@a', 'c', 2.5]
    assert a25['n_pool'] == c25['n_pool'] == 2
    assert (a25['own_matched'], c25['own_matched']) == (2, 3)
    assert c25['coverage'] == pytest.approx(1.0)
    # along-ray: (a) ranges long by 30% of each range, (c) by nothing
    v = {(r['frame'], r['arm'], r['vintage']): r for r in res['vintage_rows']}
    assert v['frozen@a', 'a', '2026']['median_along_m'] == pytest.approx(2.4)
    assert v['frozen@a', 'a', '2026']['median_along_over_range'] == pytest.approx(0.3 / 1.3)
    assert v['frozen@a', 'c', '2026']['median_along_m'] == pytest.approx(0.0, abs=1e-6)
    # the own-association frame and the other memberships are there for rule 3
    assert rows['own', 'c', 5.0]['own_p90_m'] == pytest.approx(0.0, abs=1e-6)
    assert rows['frozen@c', 'a', 5.0]['p90_m'] == pytest.approx(3.0)
    assert res['rig'] == {'2026': (2.0, 2.0)}


# -------------------------------------------------------------------------- verdict

def _city_rows(a, b, c, own=None, frozen_x=None, cov=None, dropped=0):
    """arms.csv-like rows: a/b/c = (median, p90) in frozen@a at 5 m."""
    cov = cov or {}
    rows = []
    for arm, (med, p90) in (('a', a), ('b', b), ('c', c)):
        rows.append({'frame': 'frozen@a', 'arm': arm, 'radius_m': '5.0', 'median_m': med,
                     'p90_m': p90, 'coverage': cov.get(arm, 0.80), 'op_sites': 1000,
                     'sites_dropped': dropped})
    for x in ('b', 'c'):
        fx = (frozen_x or {}).get(x, (dict(b=b, c=c)[x][1], a[1]))
        rows.append({'frame': f'frozen@{x}', 'arm': x, 'radius_m': '5.0', 'p90_m': fx[0]})
        rows.append({'frame': f'frozen@{x}', 'arm': 'a', 'radius_m': '5.0', 'p90_m': fx[1]})
        rows.append({'frame': 'own', 'arm': x, 'radius_m': '5.0',
                     'own_p90_m': (own or {}).get(x, dict(b=b, c=c)[x][1])})
    return rows


def _vint(b, c):
    return [{'frame': 'frozen@a', 'arm': 'b', 'vintage': '2026', 'median_along_m': b},
            {'frame': 'frozen@a', 'arm': 'c', 'vintage': '2026', 'median_along_m': c}]


def test_verdict_cli_takes_no_city_list():
    """The rule names its own cities; a positional list would be silently ignored."""
    assert io.build_parser().parse_args(['verdict']).cmd == 'verdict'
    with pytest.raises(SystemExit):
        io.build_parser().parse_args(['verdict', 'bend'])


def test_verdict_selects_c_when_only_c_passes():
    rows = {'gainesville': _city_rows(a=(2.0, 4.0), b=(1.95, 3.5), c=(1.2, 2.6)),
            'bend': _city_rows(a=(0.7, 1.5), b=(0.7, 1.5), c=(0.7, 1.55))}
    vint = {'gainesville': _vint(1.0, 0.2), 'bend': []}
    selected, res, _ = io.verdict(rows, vint)
    assert selected == 'c'
    assert res['c']['pass'] and not res['b']['1'] and not res['b']['5']


def test_verdict_tie_break_prefers_c_unless_b_is_clearly_better_in_both_cities():
    g = _city_rows(a=(2.0, 4.0), b=(1.2, 2.6), c=(1.2, 2.6))
    bd = _city_rows(a=(0.7, 1.5), b=(0.7, 1.5), c=(0.7, 1.5))
    vint = {'gainesville': _vint(0.1, 0.2), 'bend': []}
    selected, res, _ = io.verdict({'gainesville': g, 'bend': bd}, vint)
    assert res['b']['pass'] and res['c']['pass'] and selected == 'c'
    g = _city_rows(a=(2.0, 4.0), b=(1.1, 2.3), c=(1.2, 2.6))
    bd = _city_rows(a=(0.7, 1.7), b=(0.7, 1.3), c=(0.7, 1.55))
    selected, res, _ = io.verdict({'gainesville': g, 'bend': bd}, vint)
    assert res['b']['pass'] and res['c']['pass'] and selected == 'b'


def test_verdict_rule_4_fails_on_a_coverage_drop_and_caveats_a_site_drop():
    good_c = dict(a=(2.0, 4.0), b=(1.95, 3.9), c=(1.2, 2.6))
    rows = {'gainesville': _city_rows(**good_c, cov={'a': 0.80, 'c': 0.785}),
            'bend': _city_rows(a=(0.7, 1.5), b=(0.7, 1.5), c=(0.7, 1.5))}
    vint = {'gainesville': _vint(1.0, 0.2), 'bend': []}
    selected, res, _ = io.verdict(rows, vint)
    assert not res['c']['4'] and not res['c']['caveat'] and selected == 'a'
    rows['gainesville'] = _city_rows(**good_c, dropped=60)     # 6% of 1000
    selected, res, reasons = io.verdict(rows, vint)
    assert res['c']['caveat'] and not res['c']['pass'] and selected == 'a'
    assert any('CAVEAT' in r for r in reasons)


def test_verdict_rule_3_and_rule_2_guard_against_construction_and_regression():
    vint = {'gainesville': _vint(1.0, 0.2), 'bend': []}
    base = dict(a=(2.0, 4.0), b=(1.95, 3.9), c=(1.2, 2.6))
    # (c) wins only inside (a)'s membership: under its own, (a) is no worse
    rows = {'gainesville': _city_rows(**base, frozen_x={'c': (2.9, 2.8)}),
            'bend': _city_rows(a=(0.7, 1.5), b=(0.7, 1.5), c=(0.7, 1.5))}
    selected, res, _ = io.verdict(rows, vint)
    assert not res['c']['3'] and selected == 'a'
    # bend's old rig regresses by more than 0.10 m
    rows = {'gainesville': _city_rows(**base),
            'bend': _city_rows(a=(0.7, 1.5), b=(0.7, 1.5), c=(0.8, 1.65))}
    selected, res, _ = io.verdict(rows, vint)
    assert not res['c']['2'] and selected == 'a'
