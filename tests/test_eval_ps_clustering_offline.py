"""eval_ps_clustering's offline machinery (issue #106): the blocked PS partition, label
synthesis from results.jsonl, region assignment and the unplaceable-label attach rule.

Needs pandas/scipy/haversine (analysis-only, not in requirements-test.txt), so the file
skips where they are missing (CI)."""
import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eval_ps_clustering as epc  # noqa: E402

LAT0, LNG0 = 37.54, -77.43


def _partition_sets(df, arr):
    groups = {}
    for lid, c in zip(df['label_id'].to_numpy(), arr):
        groups.setdefault(int(c), set()).add(int(lid))
    return {frozenset(v) for v in groups.values()}


def test_blocked_partition_equals_dense():
    rng = np.random.default_rng(106)
    n = 300
    # clumps of labels along a few streets, so blocks are neither all singletons nor one
    centres = rng.uniform(0, 400, size=(40, 2))
    pts = centres[rng.integers(0, 40, n)] + rng.normal(0, 4.0, size=(n, 2))
    lat = LAT0 + pts[:, 1] / 111320.0
    lng = LNG0 + pts[:, 0] / (111320.0 * np.cos(np.radians(LAT0)))
    panos = [f'p{i // 3}' for i in range(n)]
    panos[1] = panos[0]            # same (user, pano) pair 0.1 m apart: cannot-link
    lat[1], lng[1] = lat[0] + 1e-6, lng[0]
    df = pd.DataFrame({'label_id': np.arange(n), 'user_id': 'ai', 'pano_id': panos,
                       'region_id': rng.integers(0, 3, n), 'lat': lat, 'lng': lng})
    ts = [0.0025, 0.0075, 0.0125, 0.015]
    for per_region in (True, False):
        dense = epc.ps_partition(df, ts, per_region, blocked=False)
        stats = {}
        blocked = epc.ps_partition(df, ts, per_region, blocked=True, stats=stats)
        assert stats['largest_block'] < n
        for t in ts:
            assert _partition_sets(df, dense[t]) == _partition_sets(df, blocked[t])
            assert blocked[t][0] != blocked[t][1]     # the cannot-link held


def _record(pano_id, lat, lng, heading, dets, w=4096, h=2048):
    return {'pano': {'panorama_id': pano_id, 'lat': lat, 'lng': lng,
                     'camera_heading': heading, 'width': w, 'height': h,
                     'source': 'mapillary', 'capture_date': '2025-06'},
            'detections': [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                           for x, y, c in dets]}


def test_synthesized_labels_match_the_pixel_map_and_the_server_estimator(tmp_path):
    import json
    import ps_placement
    recs = [_record('a', LAT0, LNG0, 30.0, [(0.40, 0.62, 0.81), (0.70, 0.58, 0.20),
                                            (0.55, 0.95, 0.90)]),     # 0.95: on the rig
            _record('b', LAT0 + 1e-4, LNG0, 200.0, [(0.10, 0.60, 0.56)]),
            {'pano': {'panorama_id': 'c', 'lat': None, 'lng': None, 'camera_heading': 0,
                      'width': 4096, 'height': 2048},
             'detections': [{'x_normalized': 0.5, 'y_normalized': 0.6, 'confidence': 0.9}]}]
    path = tmp_path / 'results.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in recs) + '\n', encoding='utf-8')
    labels, det_of = epc.synthesize_labels(path, 0.55)
    assert det_of == {1: ('a', 0), 2: ('b', 0)}       # tier, rig mask, no-position skip
    pix_map, n_ambiguous, n_dup = epc.label_to_detection(path, labels)
    assert pix_map == det_of and n_ambiguous == 0 and n_dup == 0
    row = labels.iloc[0]
    assert (row.pano_x, row.pano_y) == (round(0.40 * 4096), round(0.62 * 2048))
    assert (row.lat, row.lng) == ps_placement.label_latlng(LAT0, LNG0, row.pano_x,
                                                           row.pano_y, 4096, 2048, 30.0)
    unmasked, _ = epc.synthesize_labels(path, 0.55, mask_rig=False)
    assert len(unmasked) == 3


def test_regions_come_from_the_nearest_street():
    from shapely.geometry import LineString
    streets = [(10, 1, LineString([(LNG0, LAT0), (LNG0 + 0.001, LAT0)])),
               (11, 2, LineString([(LNG0, LAT0 + 0.0005), (LNG0 + 0.001, LAT0 + 0.0005)]))]
    lat = [LAT0 + 0.0001, LAT0 + 0.0004, LAT0 - 0.001]
    lng = [LNG0 + 0.0005] * 3
    reg, ties = epc.assign_regions(lat, lng, streets)
    assert reg.tolist() == [1, 2, 1] and ties == 0
    assert epc.assign_regions(lat, lng, [])[0].tolist() == [epc.NO_REGION] * 3


def test_load_streets_keeps_open_streets_only(tmp_path):
    """The server snaps a label to its nearest OPEN street; /v3/api/streets returns all."""
    pytest.importorskip('shapely')
    import json

    def feat(sid, rid, status):
        props = {'street_edge_id': sid, 'region_id': rid}
        if status is not None:
            props['status'] = status
        return {'type': 'Feature', 'properties': props, 'geometry': {
            'type': 'LineString', 'coordinates': [[LNG0, LAT0], [LNG0 + 0.001, LAT0]]}}
    path = tmp_path / 'streets.geojson'
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        feat(1, 7, 'open'), feat(2, 8, 'closed'), feat(3, 9, 'no_imagery')]}))
    assert [(s, r) for s, r, _ in epc.load_streets(path)] == [(1, 7)]
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        feat(1, 7, 'open'), feat(2, 8, None)]}))
    with pytest.raises(SystemExit):
        epc.load_streets(path)


class _Site:
    def __init__(self, sid, e, n, members):
        self.id, self.e, self.n = sid, e, n
        self.members = [(_D(p, i), True) for p, i in members]
        self.pano_ids = {p for p, _i in members}


class _D:
    def __init__(self, pano_id, det_index):
        self.pano_id, self.det_index = pano_id, det_index


def test_attach_rule_by_range_and_perpendicular_distance():
    import fuse_sites as fs
    import geo
    frame = geo.LocalFrame(LAT0, LNG0)
    # camera at the frame origin looking north: heading 0 and x = 0.5 is due north
    cam = fs.SlimPano('cam', LAT0, LNG0, 0.0, None, None, None, 'gsv',
                      [(0, 0.5, 0.51, 0.9)])
    sites = [_Site(0, 0.0, 10.0, [('o1', 0)]),     # too near (10 m)
             _Site(1, 5.0, 30.0, [('o2', 0)]),     # 5 m off the ray
             _Site(2, 1.0, 30.0, [('o3', 0)]),     # 30 m ahead, 1 m off: the one
             _Site(3, -0.5, 45.0, [('o4', 0)])]    # also on the ray, but farther
    clusters, attached = epc.attach_unplaceable(sites, [cam], frame)
    assert attached == {('cam', 0): 2}
    assert [c.n_labels for c in clusters] == [1, 1, 2, 1]
    # the cannot-link: a site already holding a label from this pano is skipped
    sites[2].pano_ids.add('cam')
    _clusters, attached = epc.attach_unplaceable(sites, [cam], frame)
    assert attached == {('cam', 0): 3}
    # beyond 60 m nothing qualifies, and the label stays a singleton
    far = [_Site(0, 0.0, 70.0, [('o5', 0)])]
    clusters, attached = epc.attach_unplaceable(far, [cam], frame)
    assert attached == {} and [c.n_labels for c in clusters] == [1, 1]
    # a label in the rig band was dropped as on_rig, never for range: it stays a singleton
    rig = fs.SlimPano('rig', LAT0, LNG0, 0.0, None, None, None, 'gsv',
                      [(0, 0.5, 0.95, 0.9)])
    sites = [_Site(0, 1.0, 30.0, [('o6', 0)])]
    assert epc.attach_unplaceable(sites, [rig], frame)[1] == {('rig', 0): 0}
    assert epc.attach_unplaceable(sites, [rig], frame, mask_rig=True)[1] == {}
