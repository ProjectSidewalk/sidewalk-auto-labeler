"""The #169 placement check (2026-11-15): the opt-in sidewalk-line label rules
(corner_labels.SidewalkCrossingPlacement) on a synthetic sidewalk layer, the seeded draw, the
pre-registered bar, and the scorer's join of blind tags to rules. CPU only, no network."""
import json

import pytest

pytest.importorskip('shapely')

import corner_labels as cl  # noqa: E402
import corner_placement_check as cpc  # noqa: E402
import geo  # noqa: E402

LAT0, LNG0 = 45.6, -122.6


def at(e, n):
    return geo.LocalFrame(LAT0, LNG0).to_latlng(e, n)


def layer(tmp_path, lines):
    """A GeoJSON of LineStrings given as ([(e, n), ...], UNITTYPE) in metres from the origin."""
    feats = []
    for pts, unittype in lines:
        coords = [list(reversed(at(e, n))) for e, n in pts]
        feats.append({'type': 'Feature', 'properties': {'UNITTYPE': unittype, 'STATUS': 'Available'},
                      'geometry': {'type': 'LineString', 'coordinates': coords}})
    p = tmp_path / 'sw.geojson'
    p.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')
    return p


class Fallback:
    unit_d = {'u': [9.0, 11.0]}

    def place(self, rec, c):
        return {'lat': 0.0, 'lng': 0.0, 'dist_m': 10.0, 'case': 'unit', 'base_m': 10.0,
                'n_unit_sites': 2, 'clamped': None, 'bearing_deg': 0.0}

    def record(self):
        return {'rule': 'site-median'}


UNIT = {'unit': 'u', 'lat': LAT0, 'lng': LNG0}
NORTH = {'corner': 0, 'start_deg': 315.0, 'width_deg': 90.0}     # bisector due north
EAST = {'corner': 1, 'start_deg': 45.0, 'width_deg': 90.0}       # bisector due east


def dist(lp):
    e, n = geo.LocalFrame(LAT0, LNG0).to_enu(lp['lat'], lp['lng'])
    return (e * e + n * n) ** 0.5


def test_crossing_finds_the_first_line_and_applies_the_unittype_inset(tmp_path):
    p = layer(tmp_path, [([(-5, 12), (5, 12)], 'DETACHED'), ([(-5, 16), (5, 16)], 'ATTACHED')])
    b = cl.SidewalkCrossingPlacement(p, Fallback(), 'sidewalk-crossing')
    lp = b.place(UNIT, NORTH)
    assert lp['case'] == 'sidewalk_crossing' and lp['n_unit_sites'] == 2
    assert lp['sidewalk']['unittype'] == 'DETACHED' and lp['sidewalk']['inset_m'] == 0.0
    assert dist(lp) == pytest.approx(12.0, abs=0.05)
    c = cl.SidewalkCrossingPlacement(p, Fallback(), 'sidewalk-crossing-inset')
    lp = c.place(UNIT, NORTH)
    assert lp['sidewalk']['inset_m'] == cl.SIDEWALK_INSET_BY_TYPE['DETACHED']
    assert dist(lp) == pytest.approx(12.0 - cl.SIDEWALK_INSET_BY_TYPE['DETACHED'], abs=0.05)
    assert lp['base_m'] == pytest.approx(12.0, abs=0.05)


def test_no_crossing_falls_back_and_the_ray_ignores_lines_too_near(tmp_path):
    p = layer(tmp_path, [([(-5, 2), (5, 2)], 'ATTACHED')])      # inside the 3 m ray start
    lp = cl.SidewalkCrossingPlacement(p, Fallback(), 'sidewalk-crossing').place(UNIT, NORTH)
    assert lp['case'] == 'site_median_fallback' and lp['sidewalk'] is None
    lp = cl.SidewalkCrossingPlacement(p, Fallback(), 'sidewalk-crossing').place(UNIT, EAST)
    assert lp['case'] == 'site_median_fallback'


def test_the_new_rules_are_opt_in():
    assert cl.LABEL_POINT_DEFAULT == 'site-median'
    assert set(cl.SidewalkCrossingPlacement.RULES) <= set(cl.LABEL_POINT_RULES)


def test_score_joins_blind_tags_to_rules_and_keeps_calibration_apart():
    key = [{'card': 'c1', 'key': 'k1', 'rule': 'A', 'calibration': False},
           {'card': 'c2', 'key': 'k1', 'rule': 'B', 'calibration': False},
           {'card': 'c3', 'key': 'k1', 'rule': 'C', 'calibration': False},
           {'card': 'c4', 'key': 'vancouver:res:n47279117#3', 'rule': 'A', 'calibration': True}]
    s = cpc.score(key, {'c1': 'in_street', 'c2': 'at_curb', 'c3': 'at_curb', 'c4': 'in_street'})
    assert s['rules']['A']['in_street'] == 1 and s['rules']['B']['at_curb'] == 1
    assert s['calibration'][0]['jon'] == 'in_street' and s['calibration'][0]['tag'] == 'in_street'
    assert [d['key'] for d in s['disagreements']] == ['k1']
    with pytest.raises(SystemExit):
        cpc.score(key, {'c1': 'at_curb'})                        # untagged cards refuse
    with pytest.raises(SystemExit):
        cpc.score(key, {'c1': 'x', 'c2': 'at_curb', 'c3': 'at_curb', 'c4': 'at_curb'})
