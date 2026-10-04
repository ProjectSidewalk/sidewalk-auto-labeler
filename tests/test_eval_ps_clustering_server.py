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
SERVER_PARAMS = fs.FuseParams(min_confidence=0.0, floor=0.0, mask_rig=False,
                              apply_pose=fs.POSE_OFF)


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
    # detection 5 was never submitted: no server label, so it must not enter the arm
    run = fs.SlimPano('p', LAT0, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81), (3, 0.70, 0.58, 0.62), (5, 0.2, 0.6, 0.9)],
                      decode='gaussian', border='keep')
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0),
                           _label(11, 'p', 0.70, 0.58, 30.0),
                           _label(12, 'p', 0.55, 0.60, 30.0, user='human'),
                           _label(13, 'q', 0.45, 0.61, 90.0, user='human')])
    det_of = {10: ('p', 0), 11: ('p', 3)}
    panos, stats = epc.server_panos(labels, det_of, {'p': run}, 'ai',
                                    decode='gaussian', border='keep')
    by_id = {p.pano_id: p for p in panos}
    assert sorted(by_id) == ['p', 'q']
    assert geo.haversine_m(by_id['p'].lat, by_id['p'].lng, LAT0, LNG0) < 0.05
    dets = {i: c for i, _x, _y, c in by_id['p'].detections}
    assert dets == {0: 0.81, 3: 0.62, epc.HUMAN_DET_BASE: epc.HUMAN_CONFIDENCE}
    assert 5 not in dets                      # only what the server holds
    # the run's decode/border pass through; a human-only pano takes the run's single one
    assert [(p.decode, p.border) for p in panos] == [('gaussian', 'keep')] * 2
    assert stats.pop('label_of') == {('p', 0): 10, ('p', 3): 11,
                                     ('p', epc.HUMAN_DET_BASE): 12,
                                     ('q', epc.HUMAN_DET_BASE): 13}
    assert stats == {'inverted': 2, 'run_position': 0, 'unplaceable': 0,
                     'unplaceable_labels': 0, 'unplaceable_label_ids': [],
                     'human_labels': 2, 'ai_labels': 2, 'ai_duplicate': 0,
                     'ai_unmapped': 0, 'inverted_far': 0}


def test_run_decode_reaches_the_fuse_guard():
    # A gaussian run pano beside an argmax one must trip fuse's mixed-decode guard
    # (#111); that works only because server_panos copies the run pano's decode.
    run = fs.SlimPano('p', LAT0, LNG0, 0.0, None, None, '2025-06', 'gsv',
                      [(0, 0.5, 0.62, 0.9)], decode='gaussian')
    labels = pd.DataFrame([_label(10, 'p', 0.5, 0.62, 0.0),
                           _label(11, 'q', 0.5, 0.62, 0.0, user='human')])
    panos, _ = epc.server_panos(labels, {10: ('p', 0)}, {'p': run}, 'ai')
    assert [p.decode for p in panos] == ['gaussian', 'argmax']
    with pytest.raises(ValueError, match='mixes peak decodes'):
        fs.fuse(panos, SERVER_PARAMS)


def test_position_is_the_servers_not_the_runs():
    # The run's block puts the camera ~11 m north of where the server placed the labels
    # from (a repositioned pano): the arm uses the server's, and flags the disagreement.
    run = fs.SlimPano('p', LAT0 + 0.0001, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81)])
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0)])
    panos, stats = epc.server_panos(labels, {10: ('p', 0)}, {'p': run}, 'ai')
    assert geo.haversine_m(panos[0].lat, panos[0].lng, LAT0, LNG0) < 0.05
    assert stats['inverted_far'] == 1
    # offline (invert=False) the run's block IS the server's position: no inversion
    panos, stats = epc.server_panos(labels, {10: ('p', 0)}, {'p': run}, 'ai', invert=False)
    assert (panos[0].lat, panos[0].lng) == (run.lat, run.lng)
    assert (stats['run_position'], stats['inverted']) == (1, 0)


def test_run_position_only_when_nothing_inverts():
    # a label near the horizon is beyond the inversion range: fall back to the run block
    run = fs.SlimPano('p', LAT0, LNG0, 0.0, None, None, '2025-06', 'gsv',
                      [(0, 0.5, 0.515, 0.7)])
    panos, stats = epc.server_panos(pd.DataFrame([_label(10, 'p', 0.5, 0.515, 0.0)]),
                                    {10: ('p', 0)}, {'p': run}, 'ai')
    assert (panos[0].lat, panos[0].lng) == (LAT0, LNG0)
    assert (stats['run_position'], stats['inverted']) == (1, 0)


def test_ai_account_label_without_a_detection_refuses():
    # A band label submitted from another file is the AI account's but maps to no stored
    # detection; read as human it would seed sites at confidence 1.0.
    run = fs.SlimPano('p', LAT0, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81)])
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0),
                           _label(11, 'p', 0.70, 0.58, 30.0)])      # AI, unmapped
    with pytest.raises(ValueError, match='maps to no stored detection'):
        epc.server_panos(labels, {10: ('p', 0)}, {'p': run}, 'ai')


def test_ai_user_keeps_unmapped_ai_labels_as_ai():
    # #56 (--ai-user): on a run rebuilt from the pano store a deployed AI label the run
    # does not reproduce pixel-exactly is still an AI label -- at the tier, in its own
    # index band, never a human label seeding sites at 1.0, and never refused
    run = fs.SlimPano('p', LAT0, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81)])
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0),
                           _label(11, 'p', 0.70, 0.58, 30.0),          # AI, unmapped
                           _label(14, 'p', 0.40, 0.62, 30.0),          # AI, same pixel as 10
                           _label(12, 'p', 0.55, 0.60, 30.0, user='human')])
    det_of = {10: ('p', 0), 14: ('p', 0)}
    panos, stats = epc.server_panos(labels, det_of, {'p': run}, 'ai',
                                    unmapped_confidence=0.55)
    dets = {i: c for i, _x, _y, c in panos[0].detections}
    assert dets == {0: 0.81, epc.DUPLICATE_DET_BASE: 0.81, epc.AI_UNMAPPED_BASE: 0.55,
                    epc.HUMAN_DET_BASE: epc.HUMAN_CONFIDENCE}
    assert (stats['ai_labels'], stats['ai_duplicate'], stats['ai_unmapped'],
            stats['human_labels']) == (2, 1, 1, 1)
    # one to one: every label id once, under four distinct keys
    assert sorted(stats['label_of'].values()) == [10, 11, 12, 14]
    assert stats['label_of'][('p', epc.AI_UNMAPPED_BASE)] == 11
    sites, _frame, _stats = fs.fuse(panos, SERVER_PARAMS)
    clusters, _n = epc.clusters_from_server_sites(sites, panos, (), stats['label_of'])
    ids = [lab for c in clusters for lab in c.label_ids]
    assert sorted(ids) == [10, 11, 12, 14]                     # each label exactly once
    assert sorted(m for c in clusters for m in c.members) == [('p', 0)]   # scored once


def test_two_labels_on_one_pixel_are_two_entries():
    # A re-submitted campaign puts two live labels on one stored detection. Each must be
    # exactly one emitted detection: none dropped, none scored twice.
    run = fs.SlimPano('p', LAT0, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81)])
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0),
                           _label(11, 'p', 0.40, 0.62, 30.0)])
    panos, stats = epc.server_panos(labels, {10: ('p', 0), 11: ('p', 0)}, {'p': run},
                                    'ai')
    assert sorted(i for i, *_ in panos[0].detections) == [0, epc.DUPLICATE_DET_BASE]
    assert stats['label_of'] == {('p', 0): 10, ('p', epc.DUPLICATE_DET_BASE): 11}
    assert stats['ai_duplicate'] == 1
    sites, _frame, _stats = fs.fuse(panos, SERVER_PARAMS)
    clusters, _n = epc.clusters_from_server_sites(sites, panos, (), stats['label_of'])
    assert sorted(lab for c in clusters for lab in c.label_ids) == [10, 11]
    assert sorted(m for c in clusters for m in c.members) == [('p', 0)]


def test_every_label_lands_in_a_cluster_and_only_ai_ones_are_scored():
    run = fs.SlimPano('p', LAT0, LNG0, 0.0, None, None, '2025-06', 'gsv',
                      [(0, 0.5, 0.62, 0.9)])
    sky = dict(_label(11, 'p', 0.5, 0.62, 0.0, user='human'), pano_y=round(0.40 * H))
    # sky: unplaceable by the raycast; z: no run block and nothing to invert
    far = _label(12, 'z', 0.5, 0.515, 0.0, user='human')
    labels = pd.DataFrame([_label(10, 'p', 0.5, 0.62, 0.0), sky, far])
    panos, stats = epc.server_panos(labels, {10: ('p', 0)}, {'p': run}, 'ai')
    assert stats['unplaceable_label_ids'] == [12]
    sites, _frame, _stats = fs.fuse(panos, SERVER_PARAMS)
    clusters, n_singletons = epc.clusters_from_server_sites(
        sites, panos, stats['unplaceable_label_ids'], stats['label_of'])
    assert n_singletons == 1
    assert sum(c.n_labels for c in clusters) == 3
    assert sorted(lab for c in clusters for lab in c.label_ids) == [10, 11, 12]
    assert sorted(m for c in clusters for m in c.members) == [('p', 0)]


def test_precision_by_cluster_size_buckets_unplaceable_first():
    big = epc.Cluster(0, [('p', 0), ('q', 0), ('r', 0)], 3)
    lone = epc.Cluster(1, [('s', 0)], 1)
    det_of = {1: ('p', 0), 2: ('q', 0), 3: ('r', 0), 4: ('s', 0), 5: ('t', 0),
              6: ('u', 0)}
    # t: unplaceable; u: placeable, but no cluster holds it (never "cluster of 1")
    det_pos = {k: (0.0, 0.0) for k in det_of.values() if k != ('t', 0)}
    verdicts = {('p', 0): True, ('q', 0): False, ('s', 0): False, ('t', 0): True}
    conf = {k: 0.8 for k in det_of.values()}
    ys = {k: 0.6 for k in det_of.values()}
    ys[('t', 0)] = 0.52
    rows = {r['bucket']: r for r in epc.size_precision([big, lone], det_of, det_pos,
                                                        verdicts, conf, ys)}
    assert list(rows) == list(epc.SIZE_BUCKETS)
    assert [(r['n'], r['judged'], r['t'], r['f']) for r in rows.values()] == [
        (1, 1, 1, 0), (1, 0, 0, 0), (1, 1, 0, 1), (0, 0, 0, 0), (3, 2, 1, 1)]
    assert rows['unplaceable']['median_y'] == 0.52
    assert rows['cluster of 2']['median_y'] is None


def test_wilson_interval():
    lo, hi = epc.wilson(21, 25)
    assert (round(lo, 2), round(hi, 2)) == (0.65, 0.94)   # scipy binomtest's Wilson
    assert epc.wilson(0, 0) is None
    assert epc.precision_ci_text(0, 0) == 'n/a'


def test_human_votes_and_label_verdicts_ignore_the_ai_validator():
    assert epc.human_votes([{'validation': 'Agree', 'validator_type': 'Human'},
                            {'validation': 'Disagree', 'validator_type': 'AI'},
                            {'validation': 'Unsure', 'validator_type': 'Human'}]) == (1, 0, 1)
    assert epc.human_votes(None) == (0, 0, 0)
    labels = pd.DataFrame([{'label_id': 1, 'human_agree': 2, 'human_disagree': 0, 'human_unsure': 0},
                           {'label_id': 2, 'human_agree': 0, 'human_disagree': 1, 'human_unsure': 0},
                           {'label_id': 3, 'human_agree': 1, 'human_disagree': 1, 'human_unsure': 0},
                           {'label_id': 4, 'human_agree': 0, 'human_disagree': 0, 'human_unsure': 0}])
    assert epc.label_verdicts(labels) == {1: True, 2: False, 3: None}


def test_validation_precision_by_cluster_size():
    clusters = [epc.Cluster(0, [], 2, label_ids=[1, 2]),      # one true, one false
                epc.Cluster(1, [], 1, label_ids=[3]),         # validated true
                epc.Cluster(2, [], 1, label_ids=[4]),         # no vote
                epc.Cluster(3, [], 3, label_ids=[5, 6, 7])]   # all false
    verdicts = {1: True, 2: False, 3: True, 5: False, 6: False, 7: False}
    rows = {r['bucket']: r for r in epc.validation_precision(clusters, verdicts)}
    assert (rows['cluster of 1']['clusters'], rows['cluster of 1']['n'],
            rows['cluster of 1']['any_false']) == (2, 1, 0)
    assert (rows['cluster of 2']['n'], rows['cluster of 2']['any_false'],
            rows['cluster of 2']['all_false'], rows['cluster of 2']['labels_validated'],
            rows['cluster of 2']['labels_false']) == (1, 1, 0, 2, 1)
    assert (rows['cluster of 3+']['any_false'], rows['cluster of 3+']['all_false']) == (1, 1)


def test_near_cluster_rate_counts_neighbours_within_r():
    frame = geo.LocalFrame(LAT0, LNG0)
    # three clusters at 0, 4 and 30 m north of the origin, positioned by their labels
    server_pos = {k: frame.to_latlng(0.0, n) for k, n in ((1, 0.0), (2, 4.0), (3, 30.0))}
    clusters = [epc.Cluster(k, [], 1, label_ids=[k]) for k in (1, 2, 3)]
    r = epc.near_cluster_rate(clusters, server_pos, frame, radii=(5.0, 12.5, 40.0))
    assert r['frame'] == 'server' and r['n_pos'] == 3
    assert [round(r['near'][x], 3) for x in (5.0, 12.5, 40.0)] == [0.667, 0.667, 1.0]
    assert r['per_1000_labels'] == 1000.0
    # a run-only cluster (no label ids) sits at its raycast position instead
    ray = [epc.Cluster(9, [], 1, e=0.0, n=100.0)]
    assert epc.near_cluster_rate(clusters + ray, server_pos, frame)['frame'] == 'mixed'
def test_ai_labels_set_the_position_when_the_pano_has_any():
    # Two human labels inserted from a stale pano position ~11 m north (Laurens) would
    # out-vote the one AI label in a median; the AI label's inversion wins.
    stale = (LAT0 + 0.0001, LNG0)
    run = fs.SlimPano('p', LAT0, LNG0, 30.0, None, None, '2025-06', 'gsv',
                      [(0, 0.40, 0.62, 0.81)])
    labels = pd.DataFrame([_label(10, 'p', 0.40, 0.62, 30.0),
                           _label(11, 'p', 0.70, 0.58, 30.0, stale, user='h'),
                           _label(12, 'p', 0.20, 0.60, 30.0, stale, user='h')])
    panos, stats = epc.server_panos(labels, {10: ('p', 0)}, {'p': run}, 'ai')
    assert geo.haversine_m(panos[0].lat, panos[0].lng, LAT0, LNG0) < 0.05
    assert stats['inverted_far'] == 0
