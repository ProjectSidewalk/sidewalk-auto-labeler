"""Pure logic of scripts/nosidewalk_zeroshot.py: view geometry and side assignment, pano
selection, band geometry, per-view scoring, pooling, metrics and the decision rule."""
import math

import numpy as np
import pytest

import nosidewalk_zeroshot as nz

NORTH = [(-122.30, 47.60), (-122.30, 47.601)]      # digitized due north, ~111 m


def test_project_side_sign_and_tangent():
    g = nz.EdgeGeom(NORTH)
    s, cross, tangent = g.project(47.6005, -122.30005)     # west of a northbound edge
    assert cross > 0                                        # left = positive (#5222 convention)
    assert tangent == pytest.approx(0.0, abs=0.01)
    assert s == pytest.approx(g.length / 2, abs=1.0)
    _, cross_e, _ = g.project(47.6005, -122.29995)
    assert cross_e < 0


def test_project_reversed_edge_flips_side():
    g = nz.EdgeGeom(list(reversed(NORTH)))
    _, cross, tangent = g.project(47.6005, -122.30005)     # same point, edge now southbound
    assert cross < 0 and tangent == pytest.approx(180.0, abs=0.01)


def test_face_bearing_and_view_heading():
    assert nz.face_bearing(0.0, 'left') == pytest.approx(270.0)
    assert nz.face_bearing(0.0, 'right') == pytest.approx(90.0)
    assert nz.face_bearing(350.0, 'right') == pytest.approx(80.0)
    # pano facing east: the left face of a northbound edge (west) is behind-left = 180
    assert nz.view_heading(270.0, 90.0) == pytest.approx(180.0)
    assert nz.view_heading(90.0, 90.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        nz.face_bearing(0.0, 'middle')


def test_tangent_follows_a_bend():
    # east for ~78 m, then north
    g = nz.EdgeGeom([(-122.300, 47.600), (-122.299, 47.600), (-122.299, 47.601)])
    assert g.tangent_at(10) == pytest.approx(90.0, abs=0.5)
    assert g.tangent_at(g.length - 10) == pytest.approx(0.0, abs=0.5)


def test_select_panos_rules():
    c = [dict(pano_id='old', s_m=30, cross_m=2, date='2018-06'),
         dict(pano_id='new', s_m=33, cross_m=2, date='2025-06'),     # too close to 'old'? newer wins
         dict(pano_id='end', s_m=8, cross_m=1, date='2026-01'),      # inside the end buffer
         dict(pano_id='far', s_m=50, cross_m=14, date='2026-01'),    # too far off the centerline
         dict(pano_id='mid', s_m=45, cross_m=-3, date='2020-01'),
         dict(pano_id='mid2', s_m=60, cross_m=0, date='2019-01')]
    got = [x['pano_id'] for x in nz.select_panos(c, 100)]
    assert got == ['new', 'mid', 'mid2', 'old'][:len(got)]
    assert 'end' not in got and 'far' not in got
    assert all(abs(a['s_m'] - b['s_m']) >= nz.MIN_PANO_SPACING_M
               for i, a in enumerate(nz.select_panos(c, 100))
               for b in nz.select_panos(c, 100)[i + 1:])
    many = [dict(pano_id=f'p{i}', s_m=15 + 9 * i, cross_m=0, date='2024-01') for i in range(9)]
    assert len(nz.select_panos(many, 200)) == nz.MAX_PANOS_PER_STREET


def test_band_pixels_geometry():
    bp = nz.band_pixels()
    assert len(bp['row']) > 1000
    assert bp['row'].min() >= 0 and bp['row'].max() < nz.VIEW_SIZE
    assert bp['col'].min() >= 0 and bp['col'].max() < nz.VIEW_SIZE
    # nearer ground is lower in the image
    near = bp['d'] < 5
    far = bp['d'] > 12
    assert bp['row'][near].mean() > bp['row'][far].mean()
    # straight ahead projects to the centre column, symmetric left/right
    ahead = np.abs(bp['a']) < 1e-9
    assert np.all(np.abs(bp['col'][ahead] - nz.VIEW_SIZE / 2) <= 1)
    assert np.isclose(bp['a'].mean(), 0.0, atol=1e-6)
    # a ground point straight ahead at d metres sits at depression atan(h/d)
    f = (nz.VIEW_SIZE / 2) / math.tan(math.radians(nz.VIEW_FOV_DEG / 2))
    d = 6.0
    dep = math.degrees(math.atan(nz.CAMERA_HEIGHT_M / d))
    want_row = nz.VIEW_SIZE / 2 + f * math.tan(math.radians(dep + nz.VIEW_PITCH_DEG))
    i = np.where(ahead & np.isclose(bp['d'], d))[0][0]
    assert abs(bp['row'][i] - want_row) <= 1


ID2LABEL = {0: 'Road', 1: 'Sidewalk', 2: 'Curb', 3: 'Car', 4: 'Pedestrian Area',
            5: 'Curb Cut', 6: 'Terrain', 7: 'Person'}
for _i, _n in enumerate(nz.OCCLUDER_CLASSES):
    if _n not in ID2LABEL.values():
        ID2LABEL[100 + _i] = _n


def hist(**counts):
    inv = {v: k for k, v in ID2LABEL.items()}
    h = np.zeros(max(ID2LABEL) + 1, dtype=int)
    for name, n in counts.items():
        h[inv[name.replace('_', ' ')]] = n
    return h.tolist()


def test_view_score_occluders_are_unobserved():
    s = nz.view_score(hist(Road=20, Sidewalk=20, Car=60), ID2LABEL)
    assert not s['observed']                       # only 40% unoccluded
    assert s['walk_share'] == pytest.approx(0.5)   # share of the observed part
    s = nz.view_score(hist(Road=40, Sidewalk=10, Curb_Cut=5, Pedestrian_Area=5, Curb=10,
                           Person=30), ID2LABEL)
    assert s['observed'] and s['occluded_frac'] == pytest.approx(0.3)
    assert s['walk_share'] == pytest.approx(20 / 70)
    assert s['sidewalk_share'] == pytest.approx(10 / 70)
    assert s['curb_share'] == pytest.approx(10 / 70)
    empty = nz.view_score(hist(), ID2LABEL)
    assert not empty['observed'] and empty['walk_share'] is None


def test_view_score_refuses_unknown_vocabulary():
    with pytest.raises(ValueError):
        nz.view_score([1, 2], {0: 'Road', 1: 'Sidewalk'})


def test_pool():
    assert nz.pool([0.0, 0.0, 0.6, 0.1]) == pytest.approx(0.225)
    assert nz.pool([0.2, 0.4], 'max') == 0.4
    assert nz.pool([0.2, 0.4, 0.9], 'median') == 0.4
    assert nz.pool([0.5]) is None                  # < 2 observed -> unobserved
    assert nz.pool([None, 0.3]) is None
    with pytest.raises(ValueError):
        nz.pool([0.1, 0.2], 'mode')


def test_weighted_auc():
    assert nz.weighted_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 1.0
    assert nz.weighted_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 0.0
    assert nz.weighted_auc([0.5, 0.5], [1, 0]) == 0.5
    # weights: the misordered absent face dominates
    a = nz.weighted_auc([0.1, 0.95, 0.9], [1, 1, 0], [1, 9, 1])
    assert a == pytest.approx(0.1)
    assert nz.weighted_auc([0.1], [1]) is None


def test_threshold_and_precision_recall():
    s = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6]
    y = [1, 1, 1, 0, 1, 0]
    tau = nz.threshold_for_recall(s, y, target=0.67)
    assert tau == 0.2                               # 3 of 4 absent at <= 0.2
    p, r = nz.precision_recall([x <= tau for x in s], y)
    assert (p, r) == (1.0, 0.75)
    # weights change precision: a heavy present face under tau
    p, r = nz.precision_recall([True, True, False], [1, 0, 1], [1, 3, 1])
    assert p == pytest.approx(0.25) and r == pytest.approx(0.5)


def test_decide_clauses():
    v, c = nz.decide(0.85, 0.85, 0.20, True, 0.9)
    assert v == 'CONTINUE' and all(val for k, val in c.items() if k.startswith('continue'))
    assert nz.decide(0.85, 0.85, 0.60, True, 0.9)[0] == 'INCONCLUSIVE'   # swap didn't collapse
    assert nz.decide(0.85, 0.85, 0.20, False, 0.9)[0] == 'INCONCLUSIVE'  # prior not beaten
    assert nz.decide(0.85, 0.69, 0.31, True, 0.9)[0] == 'STOP'           # disc AUC < 0.70
    assert nz.decide(0.69, 0.85, 0.20, True, 0.9)[0] == 'STOP'           # precision < 0.70
    assert nz.decide(0.85, 0.85, 0.20, True, 0.49)[0] == 'STOP'          # coverage < 50%
    v, c = nz.decide(0.85, 0.74, 0.71, True, 0.9)
    assert v == 'STOP' and c['stop:disc_auc_within_0.05_of_swap_null']


def _face(e, side, truth, way='residential', length=100):
    return {'street_edge_id': str(e), 'ps_side': side, 'truth2': truth, 'truth3': truth,
            'way_type': way, 'len_m': str(length), 'in_scope': 't'}


def test_draw_sample_strata_weights_and_split():
    faces = []
    for e in range(1, 41):          # discordant streets
        faces += [_face(e, 'left', 'none'), _face(e, 'right', 'present')]
    for e in range(100, 160):       # concordant absent
        faces += [_face(e, 'left', 'none'), _face(e, 'right', 'none')]
    for e in range(200, 400):       # concordant present
        faces += [_face(e, 'left', 'present'), _face(e, 'right', 'present')]
    faces.append(_face(500, 'left', 'none'))     # other face ineligible: not discordant
    smp, pop, n = nz.draw_sample(faces, seed=1, n_d=10, n_a=20, n_p=30, dev_fraction=0.2)
    assert pop == {'D': 80, 'A': 121, 'P': 400} and n == {'D': 20, 'A': 20, 'P': 30}
    d = [r for r in smp if r['stratum'] == 'D']
    assert len({r['street_edge_id'] for r in d}) == 10                   # both faces per street
    assert all(r['weight'] == pytest.approx(pop[r['stratum']] / n[r['stratum']]) for r in smp)
    split = {}
    for r in smp:                                                        # split is per street
        assert split.setdefault(r['street_edge_id'], r['split']) == r['split']
    assert smp == nz.draw_sample(faces, seed=1, n_d=10, n_a=20, n_p=30, dev_fraction=0.2)[0]
    assert all(r['face_uid'].startswith('seattle:') for r in smp)
    assert len({r['face_uid'] for r in smp}) == len(smp)


def test_eligibility_filters():
    streets = {1: {'status': 'open'}, 2: {'status': 'closed'}, 3: {'status': 'open'},
               4: {'status': 'open'}}
    rows = [_face(1, 'left', 'none'), _face(2, 'left', 'none'),
            _face(3, 'left', 'none', way='trunk'), _face(4, 'left', 'none', length=30),
            dict(_face(1, 'right', 'none'), in_scope='f')]
    got = nz.eligible_faces(rows, streets)
    assert [(r['street_edge_id'], r['ps_side']) for r in got] == [('1', 'left')]


def test_other_face_and_osm_side():
    assert nz.other_face('seattle:12:left') == 'seattle:12:right'
    assert nz.osm_side_value({'sidewalk': 'right'}, 'right') == 'present'
    assert nz.osm_side_value({'sidewalk': 'right'}, 'left') == 'absent'
    assert nz.osm_side_value({'sidewalk': 'no'}, 'left') == 'absent'
    assert nz.osm_side_value({'sidewalk:left': 'separate'}, 'left') == 'present'
    assert nz.osm_side_value({'sidewalk:both': 'no'}, 'right') == 'absent'
    assert nz.osm_side_value({'highway': 'residential'}, 'left') is None


def test_bootstrap_resamples_within_strata():
    rows = [{'stratum': 'D', 'street_edge_id': 1, 'v': 1},
            {'stratum': 'D', 'street_edge_id': 1, 'v': 2},
            {'stratum': 'A', 'street_edge_id': 2, 'v': 3}]
    reps = nz.bootstrap(rows, lambda rr: {'n': len(rr),
                                          'strata': sorted({r['stratum'] for r in rr})}, n_boot=20)
    assert all(r['n'] == 3 and r['strata'] == ['A', 'D'] for r in reps)
