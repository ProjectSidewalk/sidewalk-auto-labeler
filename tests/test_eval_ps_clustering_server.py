"""The fusion_server arm of eval_ps_clustering: fusion fed only what the server holds.

eval_ps_clustering needs pandas/scipy/haversine, which are analysis-only and not in
requirements-test.txt, so this file skips where they are missing (CI)."""
import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')

import pandas as pd  # noqa: E402

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_ps_clustering as epc  # noqa: E402

LAT0, LNG0 = 37.54, -77.43
W, H = 8192, 4096


def _label(label_id, pano_id, x, y, heading, cam=(LAT0, LNG0), user='ai'):
    """A rawLabels row whose lat/lng is the server's flat raycast from `cam`."""
    probe = fs.SlimPano(pano_id, cam[0], cam[1], heading, None, None, None, 'gsv', [])
    g = geo.detection_ground_point(fs.pano_pose(probe, fs.POSE_OFF), x, y,
                                   camera_height=epc.SERVER_CAMERA_HEIGHT_M,
                                   max_range_m=1e9, apply_pose=False)
    return {'label_id': label_id, 'user_id': user, 'pano_id': pano_id, 'region_id': 1,
            'lat': g.lat, 'lng': g.lng, 'pano_x': round(x * W), 'pano_y': round(y * H),
            'label_type': 'CurbRamp', 'pano_width': W, 'pano_height': H,
            'camera_heading': heading, 'pano_source': 'gsv',
            'image_capture_date': '2025-06'}


def test_inversion_recovers_the_camera_from_its_labels():
    cam = (LAT0 + 0.0003, LNG0 - 0.0002)
    rows = [_label(1, 'p', 0.40, 0.62, 30.0, cam), _label(2, 'p', 0.70, 0.58, 30.0, cam)]
    lat, lng, n = epc.invert_camera_position(rows)
    assert n == 2
    assert geo.haversine_m(lat, lng, *cam) < 0.05   # pixel rounding only


def test_labels_beyond_the_inversion_range_are_not_used():
    # y just below the horizon: tens of metres out, where the server's tail departs
    # from the flat raycast, so it must not steer the camera estimate
    rows = [_label(1, 'p', 0.5, 0.515, 0.0)]
    assert epc.invert_camera_position(rows) is None


def test_server_panos_keep_ai_indices_and_seed_human_labels():
    run = fs.SlimPano('p', LAT0, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81), (3, 0.70, 0.58, 0.62)])
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0),
                           _label(11, 'p', 0.70, 0.58, 30.0),
                           _label(12, 'p', 0.55, 0.60, 30.0, user='human'),
                           _label(13, 'q', 0.45, 0.61, 90.0, user='human')])
    det_of = {10: ('p', 0), 11: ('p', 3)}
    panos, stats = epc.server_panos(labels, det_of, {'p': run})
    by_id = {p.pano_id: p for p in panos}
    assert sorted(by_id) == ['p', 'q']
    assert (by_id['p'].lat, by_id['p'].lng) == (LAT0, LNG0)     # the run's position
    dets = {i: c for i, _x, _y, c in by_id['p'].detections}
    assert dets == {0: 0.81, 3: 0.62, epc.HUMAN_DET_BASE: epc.HUMAN_CONFIDENCE}
    assert stats == {'run_position': 1, 'inverted': 1, 'unplaceable': 0,
                     'human_labels': 2, 'ai_labels': 2}


def test_every_label_lands_in_a_cluster_and_only_ai_ones_are_scored():
    run = fs.SlimPano('p', LAT0, LNG0, 0.0, None, None, '2025-06', 'gsv',
                      [(0, 0.5, 0.62, 0.9)])
    sky = dict(_label(11, 'p', 0.5, 0.62, 0.0, user='human'), pano_y=round(0.40 * H))
    labels = pd.DataFrame([_label(10, 'p', 0.5, 0.62, 0.0), sky])  # sky: unplaceable
    panos, _ = epc.server_panos(labels, {10: ('p', 0)}, {'p': run})
    params = fs.FuseParams(min_confidence=0.0, floor=0.0, mask_rig=False,
                           apply_pose=fs.POSE_OFF)
    sites, _frame, _stats = fs.fuse(panos, params)
    clusters, n_singletons = epc.clusters_from_server_sites(sites, panos)
    assert n_singletons == 1
    assert sum(c.n_labels for c in clusters) == 2
    assert sorted(m for c in clusters for m in c.members) == [('p', 0)]


def test_precision_by_cluster_size_buckets_unplaceable_first():
    big = epc.Cluster(0, [('p', 0), ('q', 0), ('r', 0)], 3)
    lone = epc.Cluster(1, [('s', 0)], 1)
    det_of = {1: ('p', 0), 2: ('q', 0), 3: ('r', 0), 4: ('s', 0), 5: ('t', 0)}
    det_pos = {k: (0.0, 0.0) for k in det_of.values() if k != ('t', 0)}   # t: unplaceable
    verdicts = {('p', 0): True, ('q', 0): False, ('s', 0): False, ('t', 0): True}
    conf = {k: 0.8 for k in det_of.values()}
    rows = {r['bucket']: r for r in epc.size_precision([big, lone], det_of, det_pos,
                                                        verdicts, conf)}
    assert [(r['n'], r['judged'], r['t'], r['f']) for r in rows.values()] == [
        (1, 1, 1, 0), (1, 1, 0, 1), (0, 0, 0, 0), (3, 2, 1, 1)]


def test_wilson_interval():
    lo, hi = epc.wilson(21, 25)
    assert (round(lo, 2), round(hi, 2)) == (0.65, 0.94)   # scipy binomtest's Wilson
    assert epc.wilson(0, 0) is None
    assert epc.precision_ci_text(0, 0) == 'n/a'
