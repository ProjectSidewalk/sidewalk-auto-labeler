"""agree_rate: the pixel matcher, the grid matcher's equivalence, region bucketing and
the crowd-label parsing that the AI-vs-crowd comparison (issue #31) rests on.

Offline and torch-free: synthetic panos, synthetic regions, a synthetic feed.
"""
import json
import random

import pytest

import agree_rate as ar
import eval_sites as es
import fuse_sites as fs
from detectors import NADIR_MASK_DEG


def _pano(pano_id, dets):
    return fs.SlimPano(pano_id, 29.64, -82.33, 0.0, None, None, '2026-04', 'launch',
                       [(i, x, y, c) for i, (x, y, c) in enumerate(dets)])


def _label(lid, pano_id, x, y, human='unvalidated'):
    return ar.CrowdLabel(uid=f'gainesville:{lid}', label_id=lid, label_type='CurbRamp',
                         user_id='u', pano_id=pano_id, region_id=5, lat=29.64,
                         lng=-82.33, x=x, y=y, capture_date='2026-04', human=human,
                         ps_correct=None, has_ai_vote=False, id=lid)


# ------------------------------------------------------------------ pano frame

def test_pano_distance_wraps_at_the_seam_and_scales_y_by_half():
    # 0.99 and 0.01 are 0.02 apart across the seam, not 0.98
    assert ar.pano_distance(0.99, 0.5, 0.01, 0.5) == pytest.approx(0.02)
    # y is scaled 512/1024: a 0.02 step in y is 0.01 in x units (equal angle)
    assert ar.pano_distance(0.3, 0.50, 0.3, 0.52) == pytest.approx(0.01)


def test_match_pano_is_one_to_one_and_nearest_first():
    crowd = [(0.500, 0.55), (0.510, 0.55)]
    ai = [(0.505, 0.55)]
    m = ar.match_pano(crowd, ai, 0.022)
    # both are 0.005 away; the tie goes to the lower index, and only one may claim it
    assert m == {0: (0, pytest.approx(0.005))}


def test_match_pano_radius_is_strict_like_rampnet():
    assert ar.match_pano([(0.5, 0.5)], [(0.522, 0.5)], 0.022) == {}
    assert 0 in ar.match_pano([(0.5, 0.5)], [(0.5219, 0.5)], 0.022)


def test_pano_frame_applies_tier_and_rig_mask_and_matches_across_the_seam():
    rig_y = 0.5 + (NADIR_MASK_DEG + 5) / 180.0          # on the camera vehicle
    panos = {'p1': _pano('p1', [(0.995, 0.56, 0.9),        # seam neighbour of label 0
                                (0.300, 0.56, 0.40),       # operational only at 0.30
                                (0.700, rig_y, 0.95)])}    # masked: never a match
    crowd = [_label(0, 'p1', 0.005, 0.56), _label(1, 'p1', 0.300, 0.56),
             _label(2, 'p1', 0.700, rig_y), _label(3, 'elsewhere', 0.5, 0.5)]
    hit_op, ai_tot_op, ai_hit_op = ar.pano_frame(crowd, panos, 0.30, 0.022)
    hit_bm, ai_tot_bm, _ = ar.pano_frame(crowd, panos, 0.55, 0.022)
    assert set(hit_op) == {0, 1}
    assert set(hit_bm) == {0}
    assert (ai_tot_op, ai_hit_op, ai_tot_bm) == (2, 2, 1)


def test_pixel_alignment_reports_signed_offsets():
    panos = {'p1': _pano('p1', [(0.5 + 1 / 360, 0.56, 0.9)])}   # AI 1 deg right
    n, mdx, mdy, _, _ = ar.pixel_alignment([_label(0, 'p1', 0.5, 0.56)], panos, 0.30)
    assert n == 1 and mdx == pytest.approx(1.0) and mdy == pytest.approx(0.0)


# ------------------------------------------------------------------ world frame

def test_grid_matcher_is_identical_to_eval_sites():
    rng = random.Random(7)
    ramps = [ar.Pt(i, rng.uniform(0, 200), rng.uniform(0, 200)) for i in range(150)]
    sites = [ar.Pt(i, rng.uniform(0, 200), rng.uniform(0, 200)) for i in range(220)]
    for r in (2.5, 5.0, 7.5):
        fast = {k: v.id for k, v in ar.match_one_to_one(ramps, sites, r).items()}
        slow = {k: v.id for k, v in es.match_one_to_one(ramps, sites, r).items()}
        assert fast == slow and fast


def test_any_within_returns_nearest_distance():
    pts = [ar.Pt(0, 0.0, 0.0), ar.Pt(1, 100.0, 0.0)]
    others = [ar.Pt(0, 3.0, 4.0), ar.Pt(1, 1.0, 0.0)]
    assert ar.any_within(pts, others, 5.0) == {0: pytest.approx(1.0)}


# ------------------------------------------------------------------ regions

def _square(x0, y0, x1, y1):
    return {'type': 'MultiPolygon',
            'coordinates': [[[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]]}


def test_region_bucketing_full_partial_outside(tmp_path):
    from shapely.geometry import shape
    feed = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': _square(0, 0, 1, 1),
         'properties': {'region_id': 5, 'name': 'a', 'completion_rate': 1.0}},
        {'type': 'Feature', 'geometry': _square(1, 0, 2, 1),
         'properties': {'region_id': 7, 'name': 'b', 'completion_rate': 0.9065}},
        # opened after the run: outside the run polygon however complete it is
        {'type': 'Feature', 'geometry': _square(5, 5, 6, 6),
         'properties': {'region_id': 9, 'name': 'c', 'completion_rate': 1.0}},
        # a sliver of it inside the run polygon is not membership
        {'type': 'Feature', 'geometry': _square(1.9, 1, 2.9, 2),
         'properties': {'region_id': 11, 'name': 'd', 'completion_rate': 1.0}}]}
    path = tmp_path / 'regions.geojson'
    path.write_text(json.dumps(feed), encoding='utf-8')
    area = shape(_square(0, 0, 2, 1.1))
    regions = ar.load_regions(path, area)
    cls = {rid: ar.region_class(rid, regions, 0.999) for rid in (5, 7, 9, 11, 404, None)}
    assert cls == {5: 'full', 7: 'partial', 9: 'outside', 11: 'outside',
                   404: 'outside', None: 'outside'}
    loc = ar.RegionLocator(regions)
    assert (loc.locate(0.5, 0.5), loc.locate(0.5, 1.5), loc.locate(5.5, 5.5)) == (5, 7, None)


# ------------------------------------------------------------------ crowd feed

def test_human_status_ignores_ai_and_self_votes():
    ai = {'user_id': 'bot', 'validation': 'Agree', 'validator_type': 'AI'}
    own = {'user_id': 'me', 'validation': 'Agree', 'validator_type': 'Human'}
    agree = {'user_id': 'v1', 'validation': 'Agree', 'validator_type': 'Human'}
    dis = {'user_id': 'v2', 'validation': 'Disagree', 'validator_type': 'Human'}
    unsure = {'user_id': 'v3', 'validation': 'Unsure', 'validator_type': 'Human'}
    assert ar.human_status([ai, own], 'me') == 'unvalidated'
    assert ar.human_status([ai, agree], 'me') == 'agreed'
    assert ar.human_status([dis, unsure], 'me') == 'disagreed'
    assert ar.human_status([agree, dis], 'me') == 'unvalidated'   # a tie is no verdict


def test_load_crowd_normalizes_pixels_and_keys_on_city(tmp_path):
    def feat(lid, lng, **kw):
        props = {'label_id': lid, 'user_id': 'u', 'pano_id': 'p', 'label_type': 'CurbRamp',
                 'region_id': 5, 'pano_x': 4096, 'pano_y': 4506, 'pano_width': 16384,
                 'pano_height': 8192, 'image_capture_date': '2026-04', 'correct': True,
                 'validations': [{'user_id': 'bot', 'validation': 'Agree',
                                  'validator_type': 'AI'}]}
        props.update(kw)
        return {'type': 'Feature', 'properties': props,
                'geometry': {'type': 'Point', 'coordinates': [lng, 29.64]}}
    path = tmp_path / 'raw.geojson'
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        feat(12345, -82.33), feat(2, 1e14), feat(3, -82.33, pano_width=None)]}),
        encoding='utf-8')
    labs, dropped = ar.load_crowd(path, 'gainesville')
    assert dropped == 1                                  # the corrupt lng
    assert labs[0].uid == 'gainesville:12345'
    assert (labs[0].x, labs[0].y) == (0.25, pytest.approx(0.55, abs=1e-4))
    assert labs[0].has_ai_vote and labs[0].human == 'unvalidated'
    assert labs[1].x is None                             # no size: world frame only


def test_vintage_stratum_prefers_history_and_signs_months():
    run_meta = {'ours': ('2026-04', 16384, 8192, 0)}
    history = {'old': [('ours', '2026-04'), ('also', '2025-01')]}
    lab = _label(0, 'ours', 0.5, 0.5)
    assert ar.vintage_stratum(lab, run_meta, history, None) == ('same pano', 0)
    lab = _label(1, 'old', 0.5, 0.5)
    lab.capture_date = '2024-04'
    # the newest run pano listing it (2026-04) is the reference: 24 months older
    assert ar.vintage_stratum(lab, run_meta, history, None) == ('19-36 mo older', 24)
    lab = _label(2, 'unlinked', 0.5, 0.5)
    lab.capture_date = '2026-06'
    assert ar.vintage_stratum(lab, run_meta, history,
                              fs._months('2026-04')) == ('newer than ours', -2)
    assert ar.vintage_stratum(lab, run_meta, history, None) == ('no run pano nearby', None)


def test_chance_floor_breaks_real_correspondence():
    labels = [ar.Pt(i, 100.0 * i, 0.0) for i in range(20)]
    sites = [ar.Pt(i, 100.0 * i, 0.0) for i in range(20)]      # every label has its site
    assert len(ar.match_one_to_one(labels, sites, 5.0)) == 20
    assert ar.chance_floor(labels, sites, 5.0) == 0            # 25 m away, sites 100 m apart
    assert ar.chance_floor(labels, sites, 5.0) == ar.chance_floor(labels, sites, 5.0)
