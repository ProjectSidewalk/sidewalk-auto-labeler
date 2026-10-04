"""The #56 additions to the clustering scorers, on a toy city: `--no-gt`, `--ai-user` (and
its refusals), the offline check's human set, and inventory_clustering's server placement.

Needs pandas/scipy/haversine/shapely (analysis-only, not in requirements-test.txt), so the
file skips where they are missing (CI)."""
import json

import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')
pytest.importorskip('shapely')

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_ps_clustering as epc  # noqa: E402
import inventory_clustering as ic  # noqa: E402

LAT0, LNG0 = 45.63, -122.60
W, H = 4096, 2048
AI, HUMAN = 'ai-account', 'a-human'


def _label_feature(label_id, user, pano_id, x, y, heading=30.0):
    """A rawLabels feature at the server's flat raycast from the toy camera."""
    probe = fs.SlimPano(pano_id, LAT0, LNG0, heading, None, None, None, 'gsv', [])
    g = geo.detection_ground_point(fs.pano_pose(probe, fs.POSE_OFF), x, y,
                                   camera_height=epc.SERVER_CAMERA_HEIGHT_M,
                                   max_range_m=1e9, apply_pose=False)
    props = {'label_id': label_id, 'user_id': user, 'pano_id': pano_id, 'region_id': 1,
             'label_type': 'CurbRamp', 'pano_x': round(x * W), 'pano_y': round(y * H),
             'pano_width': W, 'pano_height': H, 'camera_heading': heading,
             'pano_source': 'gsv', 'image_capture_date': '2024-06',
             'time_created': f'2025-09-18T15:13:{label_id:02d}-07:00',
             'validations': [{'validation': 'Disagree', 'validator_type': 'Human'}]
             if label_id == 2 else [], 'correct': None}
    return {'type': 'Feature', 'properties': props,
            'geometry': {'type': 'Point', 'coordinates': [g.lng, g.lat]}}


def _toy_city(tmp_path):
    """One pano with two stored detections; on the server: label 1 (AI, mapped to
    detection 0), label 2 (AI, at a pixel the run does not hold: 'unmapped'), label 3
    (human). The deployed clusters are {1, 2} and {3}."""
    rec = {'pano': {'panorama_id': 'a', 'lat': LAT0, 'lng': LNG0, 'camera_heading': 30.0,
                    'width': W, 'height': H, 'source': 'gsv', 'capture_date': '2024-06'},
           'detections': [{'x_normalized': 0.40, 'y_normalized': 0.62, 'confidence': 0.81},
                          {'x_normalized': 0.60, 'y_normalized': 0.60, 'confidence': 0.70}]}
    (tmp_path / 'results.jsonl').write_text(json.dumps(rec) + '\n', encoding='utf-8')
    feats = [_label_feature(1, AI, 'a', 0.40, 0.62),
             _label_feature(2, AI, 'a', 0.41, 0.63),
             _label_feature(3, HUMAN, 'a', 0.20, 0.60)]
    labels = tmp_path / 'raw_labels.geojson'
    labels.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}),
                      encoding='utf-8')
    clusters = tmp_path / 'clusters.geojson'
    clusters.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [LNG0, LAT0]},
         'properties': {'label_cluster_id': k, 'label_ids': ids, 'region_id': 1}}
        for k, ids in ((1, [1, 2]), (2, [3]))]}), encoding='utf-8')
    return ['vancouver', '--run-dir', str(tmp_path), '--labels', str(labels),
            '--clusters', str(clusters), '--out', str(tmp_path / 'out'), '--no-gt']


def test_no_gt_with_ai_user_scores_every_label_once(tmp_path):
    epc.main(_toy_city(tmp_path) + ['--ai-user', AI])
    report = (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')
    assert 'GT: none (--no-gt)' in report
    assert '1 of them the AI account\'s, kept as AI by --ai-user; 1 human' in report
    assert '1 unmapped AI (--ai-user, at 0.55)' in report
    line = next(x for x in report.splitlines() if 'in exactly one fusion_server' in x)
    assert not line.startswith('- warning') and '3 label ids, 3 distinct, of 3' in line
    # the human vote on label 2 reaches validation_precision.csv
    assert (tmp_path / 'out' / 'validation_precision.csv').exists()


def test_unmapped_ai_without_ai_user_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='map to no stored detection'):
        epc.main(_toy_city(tmp_path))


def test_ai_user_must_be_the_mapped_account(tmp_path):
    with pytest.raises(SystemExit, match='is not the account whose labels map'):
        epc.main(_toy_city(tmp_path) + ['--ai-user', HUMAN])


def test_ai_user_with_offline_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match='--ai-user names the live AI account'):
        epc.main(['vancouver', '--run-dir', str(tmp_path), '--offline', '--no-gt',
                  '--ai-user', AI])


def test_offline_check_humans_are_the_other_accounts(tmp_path, monkeypatch):
    # An unmapped AI label is not human: the offline check's human set is picked by
    # account, so label 2 must not be in it (it was, when humans were "not in det_of").
    streets = tmp_path / 'streets.geojson'
    streets.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'street_edge_id': 1, 'region_id': 1,
                                           'status': 'open'},
         'geometry': {'type': 'LineString',
                      'coordinates': [[LNG0 - 0.001, LAT0], [LNG0 + 0.001, LAT0]]}}]}),
        encoding='utf-8')
    seen = {}

    class Stop(Exception):
        pass

    def fake_check(ai, det_of, results_path, streets, min_confidence, deployed, humans,
                   clustered):
        seen['humans'] = sorted(humans.label_id)
        seen['ai'] = sorted(ai.label_id)
        raise Stop

    monkeypatch.setattr(epc, 'offline_check', fake_check)
    with pytest.raises(Stop):
        epc.main(_toy_city(tmp_path) + ['--ai-user', AI, '--offline-check',
                                        '--streets', str(streets)])
    assert seen == {'humans': [3], 'ai': [1, 2]}


def test_server_placed_uses_the_labels_server_positions():
    frame = geo.LocalFrame(LAT0, LNG0)
    server_pos = {1: frame.to_latlng(0.0, 10.0), 2: frame.to_latlng(4.0, 10.0),
                  3: frame.to_latlng(-6.0, 0.0)}
    # raycast positions (c.e / c.n, members) are deliberately elsewhere: server_placed
    # must ignore them
    clusters = [epc.Cluster(0, [('a', 0)], 2, e=100.0, n=100.0, label_ids=[1, 2]),
                epc.Cluster(1, [], 1, label_ids=[3]),
                epc.Cluster(2, [], 1, label_ids=[99])]           # not in the pull
    placed = ic.server_placed(clusters, server_pos, frame)
    (c0, p0), (c1, p1), (c2, p2) = placed
    assert c0 == pytest.approx((2.0, 10.0), abs=1e-6) and len(p0) == 2
    assert c1 == pytest.approx((-6.0, 0.0), abs=1e-6)
    assert c2 is None and p2 == []


def test_streets_for_a_server_label_city_refuses_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(epc, 'fetch', lambda *a, **k: pytest.fail('must not fetch'))
    with pytest.raises(SystemExit, match='frozen'):
        ic.streets_for('vancouver', tmp_path)
    path = tmp_path / 'runs' / 'vancouver' / 'ps_clustering_eval' / 'streets.geojson'
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    assert ic.streets_for('vancouver', tmp_path) == path


def test_bare_score_leaves_vancouver_out():
    assert 'vancouver' in ic.CITIES and 'vancouver' not in ic.DEFAULT_CITIES


def test_inventory_score_city_runs_both_server_panos_calls(tmp_path, monkeypatch):
    # #107 re-review M5: inventory_clustering's server_panos calls (server-label arms and
    # the synthesized fusion_server+attach) broke silently once in a merge; run them
    # end to end on the toy city.
    city = tmp_path / 'runs' / 'vancouver'
    (city / 'provenance_gate').mkdir(parents=True)
    (city / 'ps_clustering_eval').mkdir()
    _toy_city(city)
    cfg = dict(ic.SERVER_LABELS['vancouver'], ai_user=AI)
    (city / 'raw_labels.geojson').replace(city / cfg['labels'])
    (city / 'clusters.geojson').replace(city / cfg['clusters'])
    (city / 'ps_clustering_eval' / 'streets.geojson').write_text(json.dumps({
        'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': {'street_edge_id': 1, 'region_id': 1,
                                               'status': 'open'},
             'geometry': {'type': 'LineString',
                          'coordinates': [[LNG0 - 0.001, LAT0], [LNG0 + 0.001, LAT0]]}}]}),
        encoding='utf-8')
    monkeypatch.setattr(ic, 'REPO_ROOT', tmp_path)
    monkeypatch.setitem(ic.SERVER_LABELS, 'vancouver', cfg)
    frame = geo.LocalFrame(LAT0, LNG0)
    inventory = [(0, *frame.to_latlng(0.0, 6.0), None), (1, *frame.to_latlng(-8.0, 3.0), None)]
    monkeypatch.setattr(ic.ioracle, 'load_inventory', lambda c: (inventory, {}))
    rows = ic.score_city('vancouver', tiers=(0.55,), frames=(2.6,), radii=(5.0,))
    sets = {(r['label_set'], r['arm'], r['placement']) for r in rows}
    assert ('all', 'fusion_server', 'server') in sets
    assert ('mapped', 'fusion_server+attach', 'raycast') in sets
    assert ('synthesized', 'fusion_server+attach', 'raycast') in sets
    report = (city / 'inventory_clustering' / 'report.md').read_text(encoding='utf-8')
    assert 'Same label set' in report and '1 unmapped AI (at 0.55)' in report


def test_raycast_placed_places_a_shared_detection_twin():
    # labels 1 and 2 share stored detection ('a', 0); fusion_server makes 2 a non-member
    # (DUPLICATE_DET_BASE), so member placement would drop it -- label placement does not
    det_of = {1: ('a', 0), 2: ('a', 0), 3: ('a', 1)}
    det_pos = {('a', 0): (1.0, 2.0)}                    # detection 1 is not raycastable
    clusters = [epc.Cluster(0, [('a', 0)], 1, label_ids=[1]),
                epc.Cluster(1, [], 1, label_ids=[2]),
                epc.Cluster(2, [], 1, label_ids=[3])]
    (c0, _), (c1, _), (c2, _) = ic.raycast_placed(clusters, det_of, det_pos)
    assert c0 == c1 == (1.0, 2.0) and c2 is None
