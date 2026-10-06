"""clustering_metrics: eval_ps_clustering's pure scoring, which CI can run (#133).

No importorskip here on purpose: this module needs no pandas, scipy or haversine, so these
tests run under requirements-test.txt. The originals in test_eval_ps_clustering_server.py
stay where they are (they skip in CI) and prove the re-export through `epc.<name>`."""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import geo
import fuse_sites as fs
import eval_sites as es
import clustering_metrics as cm

REPO_ROOT = Path(__file__).resolve().parents[1]
LAT0, LNG0 = 37.54, -77.43
W, H = 8192, 4096
ENDPOINT = 'https://ps.example/ai/submitLabelsOnPano'


def _label(label_id, pano_id, x, y, heading, cam=(LAT0, LNG0), user='ai'):
    """A rawLabels row (a dict) whose lat/lng is the server's flat raycast from `cam`."""
    probe = fs.SlimPano(pano_id, cam[0], cam[1], heading, None, None, None, 'gsv', [])
    g = geo.detection_ground_point(fs.pano_pose(probe, fs.POSE_OFF), x, y,
                                   camera_height=cm.SERVER_CAMERA_HEIGHT_M,
                                   max_range_m=1e9, apply_pose=False)
    return {'label_id': label_id, 'user_id': user, 'pano_id': pano_id, 'region_id': 1,
            'lat': g.lat, 'lng': g.lng, 'pano_x': round(x * W), 'pano_y': round(y * H),
            'label_type': 'CurbRamp', 'pano_width': W, 'pano_height': H,
            'camera_heading': heading, 'pano_source': 'gsv',
            'image_capture_date': '2025-06'}


# ------------------------------------------------------- ported from the server tests

def test_inversion_recovers_the_camera_from_its_labels():
    cam = (LAT0 + 0.0003, LNG0 - 0.0002)
    rows = [_label(1, 'p', 0.40, 0.62, 30.0, cam), _label(2, 'p', 0.70, 0.58, 30.0, cam)]
    lat, lng, n = cm.invert_camera_position(rows)
    assert n == 2
    assert geo.haversine_m(lat, lng, *cam) < 0.05   # pixel rounding only


def test_labels_beyond_the_inversion_range_are_not_used():
    assert cm.invert_camera_position([_label(1, 'p', 0.5, 0.515, 0.0)]) is None


def test_precision_by_cluster_size_buckets_unplaceable_first():
    big = cm.Cluster(0, [('p', 0), ('q', 0), ('r', 0)], 3)
    lone = cm.Cluster(1, [('s', 0)], 1)
    det_of = {1: ('p', 0), 2: ('q', 0), 3: ('r', 0), 4: ('s', 0), 5: ('t', 0),
              6: ('u', 0)}
    det_pos = {k: (0.0, 0.0) for k in det_of.values() if k != ('t', 0)}
    verdicts = {('p', 0): True, ('q', 0): False, ('s', 0): False, ('t', 0): True}
    conf = {k: 0.8 for k in det_of.values()}
    ys = {k: 0.6 for k in det_of.values()}
    ys[('t', 0)] = 0.52
    rows = {r['bucket']: r for r in cm.size_precision([big, lone], det_of, det_pos,
                                                       verdicts, conf, ys)}
    assert list(rows) == list(cm.SIZE_BUCKETS)
    assert [(r['n'], r['judged'], r['t'], r['f']) for r in rows.values()] == [
        (1, 1, 1, 0), (1, 0, 0, 0), (1, 1, 0, 1), (0, 0, 0, 0), (3, 2, 1, 1)]
    assert rows['unplaceable']['median_y'] == 0.52
    assert rows['cluster of 2']['median_y'] is None


def test_wilson_interval():
    lo, hi = cm.wilson(21, 25)
    assert (round(lo, 2), round(hi, 2)) == (0.65, 0.94)   # scipy binomtest's Wilson
    assert cm.wilson(0, 0) is None
    assert cm.precision_ci_text(0, 0) == 'n/a'


def test_human_votes_ignore_the_ai_validator():
    assert cm.human_votes([{'validation': 'Agree', 'validator_type': 'Human'},
                           {'validation': 'Disagree', 'validator_type': 'AI'},
                           {'validation': 'Unsure', 'validator_type': 'Human'}]) == (1, 0, 1)
    assert cm.human_votes(None) == (0, 0, 0)


def test_validation_precision_by_cluster_size():
    clusters = [cm.Cluster(0, [], 2, label_ids=[1, 2]),      # one true, one false
                cm.Cluster(1, [], 1, label_ids=[3]),         # validated true
                cm.Cluster(2, [], 1, label_ids=[4]),         # no vote
                cm.Cluster(3, [], 3, label_ids=[5, 6, 7])]   # all false
    verdicts = {1: True, 2: False, 3: True, 5: False, 6: False, 7: False}
    rows = {r['bucket']: r for r in cm.validation_precision(clusters, verdicts)}
    assert (rows['cluster of 1']['clusters'], rows['cluster of 1']['n'],
            rows['cluster of 1']['any_false']) == (2, 1, 0)
    assert (rows['cluster of 2']['n'], rows['cluster of 2']['any_false'],
            rows['cluster of 2']['all_false'], rows['cluster of 2']['labels_validated'],
            rows['cluster of 2']['labels_false']) == (1, 1, 0, 2, 1)
    assert (rows['cluster of 3+']['any_false'], rows['cluster of 3+']['all_false']) == (1, 1)


def test_near_cluster_rate_counts_neighbours_within_r():
    frame = geo.LocalFrame(LAT0, LNG0)
    server_pos = {k: frame.to_latlng(0.0, n) for k, n in ((1, 0.0), (2, 4.0), (3, 30.0))}
    clusters = [cm.Cluster(k, [], 1, label_ids=[k]) for k in (1, 2, 3)]
    r = cm.near_cluster_rate(clusters, server_pos, frame, radii=(5.0, 12.5, 40.0))
    assert r['frame'] == 'server' and r['n_pos'] == 3
    assert [round(r['near'][x], 3) for x in (5.0, 12.5, 40.0)] == [0.667, 0.667, 1.0]
    assert r['per_1000_labels'] == 1000.0
    ray = [cm.Cluster(9, [], 1, e=0.0, n=100.0)]
    assert cm.near_cluster_rate(clusters + ray, server_pos, frame)['frame'] == 'mixed'


# ------------------------------------------------------------------- new: the scorer

def test_score_on_a_toy_city():
    # two pool ramps 20 m apart (neither self-detected); A matches ramp 0, B is a fragment
    # of it 2 m away, C matches ramp 1, D sits 100 m from both and never enters `near`
    pts = [es.GTPoint('g0', 'missed', 0.0, 0.0, True),
           es.GTPoint('g1', 'missed', 20.0, 0.0, True)]
    ramps = es.merge_gt_points(pts, 2.5)
    gt = (ramps, [r for r in ramps if r.in_pool], pts, {})
    clusters = [cm.Cluster(0, [('a', 0)], 2, e=0.5, n=0.0),
                cm.Cluster(1, [('b', 0)], 1, e=2.0, n=0.0),
                cm.Cluster(2, [('c', 0)], 3, e=20.0, n=3.0),
                cm.Cluster(3, [('d', 0)], 1, e=100.0, n=100.0),
                cm.Cluster(4, [], 1)]                               # unplaced
    r = cm.score(clusters, gt, {}, radius_m=5.0)
    assert (r['n_clusters'], r['n_placed'], r['n_labels']) == (5, 4, 8)
    assert (r['covered'], r['n_pool'], r['coverage']) == (2, 2, 1.0)
    assert r['frag'][3.0] == {'ramps': 2, 'with_extra': 1, 'extra': 1}
    assert r['frag'][5.0] == {'ramps': 2, 'with_extra': 1, 'extra': 1}
    assert r['precision'] is None and r['same_pano_pairs'] == 0
    # the prefilter is a superset of what can match: D, 2 x reach from every ramp, is out
    reach = 5.0
    far = [(c.e, c.n) for c in clusters if c.e is not None]
    hit = cm.within_reach(far, [(rp.e, rp.n) for rp in ramps], reach)
    assert hit == {0, 1, 2}
    assert all(min(math.hypot(x - rp.e, y - rp.n) for rp in ramps) >= 2 * reach
               for i, (x, y) in enumerate(far) if i not in hit)


def _brute_within(points, centres, r):
    return {i for i, (x, y) in enumerate(points)
            if any((x - cx) ** 2 + (y - cy) ** 2 <= r * r for cx, cy in centres)}


def _brute_neighbour(points, r):
    return {i for i, (x, y) in enumerate(points)
            if any(j != i and (x - u) ** 2 + (y - v) ** 2 <= r * r
                   for j, (u, v) in enumerate(points))}


def test_grid_search_equals_brute_force():
    rng = random.Random(133)
    for r in (2.5, 5.0, 7.5, 12.5):
        pts = [(rng.uniform(-60, 60), rng.uniform(-60, 60)) for _ in range(300)]
        # points exactly on grid-cell boundaries and pairs exactly r apart
        size = r * cm._CELL_PAD
        pts += [(k * size, -k * size) for k in range(-3, 4)]
        pts += [(k * r, 0.0) for k in range(-3, 4)] + [(0.0, k * r) for k in range(-3, 4)]
        pts += [(-17.0, -17.0), (-17.0, -17.0)]                  # coincident
        centres = [(rng.uniform(-60, 60), rng.uniform(-60, 60)) for _ in range(40)]
        centres += [(0.0, 0.0), (size, size)]
        assert cm.within_reach(pts, centres, r) == _brute_within(pts, centres, r)
        assert cm.with_neighbour(pts, r) == _brute_neighbour(pts, r)
    assert cm.within_reach([], [(0.0, 0.0)], 5.0) == set()
    assert cm.with_neighbour([(0.0, 0.0)], 5.0) == set()


def test_module_needs_no_pandas_scipy_or_haversine():
    code = ('import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + "/scripts"]; '
            'import clustering_metrics; '
            'print(sorted(m for m in ("pandas", "scipy", "haversine") if m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code, str(REPO_ROOT)], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == '[]'


# ------------------------------------------------- new: live position as of a pull

def _results(path, positions):
    path.write_text(''.join(json.dumps({'pano': {'panorama_id': pid, 'lat': ll[0],
                                                 'lng': ll[1]}}) + '\n'
                            for pid, ll in positions.items()), encoding='utf-8')


def _record(results, at, n_lines):
    """A submission record shaped as position_check.campaigns_for reads it (one full
    campaign on ENDPOINT), as in tests/test_position_check.py."""
    (results.parent / (results.name + '.submission.json')).write_text(json.dumps(
        {'input_file': results.name, 'total_lines': n_lines, 'endpoints': {ENDPOINT: {
            'submitted_lines': n_lines, 'labels_submitted': 3, 'min_confidence': 0.55,
            'last_submission_utc': at}}}), encoding='utf-8')


def test_live_position_as_of_a_pull(tmp_path):
    sfm = {'a': (LAT0, LNG0), 'b': (LAT0 + 0.001, LNG0)}
    raw = {'a': (LAT0 + 0.00004, LNG0)}          # 'a' resent ~4.4 m north on 09-24
    _results(tmp_path / 'results.jsonl', sfm)
    _record(tmp_path / 'results.jsonl', '2026-09-05T01:39:04Z', 2)
    _results(tmp_path / 'results.fix.jsonl', raw)
    _record(tmp_path / 'results.fix.jsonl', '2026-09-24T13:30:56Z', 1)

    live, problems = cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-21T13:34:28+00:00')
    assert problems == [] and live == sfm                  # the older campaign only
    assert cm.fallback_moved(['a', 'b'], sfm, live) == ([], [])

    live, problems = cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-28T17:09:50+00:00')
    assert problems == [] and live == {'a': raw['a'], 'b': sfm['b']}
    assert cm.fallback_moved(['a', 'b'], sfm, live) == (['a'], [])

    # before any campaign nothing is live, so nothing can be told
    live, _ = cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-01T00:00:00Z')
    assert live == {} and cm.fallback_moved(['a'], sfm, live) == ([], ['a'])
    # another endpoint holds nothing
    assert cm.live_positions_as_of(tmp_path, 'https://other/x', '2026-10-01T00:00:00Z') \
        == ({}, [])


def test_a_band_sent_after_the_pull_does_not_hide_its_base(tmp_path):
    # #139 review S1. send_to_ps.write_submission_record moves the BASE entry's
    # last_submission_utc to the band's time, so a base first sent 09-05 with a band on
    # 09-22 reads "last 09-22". A 09-10 pull saw the base live: it must be cut on
    # first_submission_utc, not dropped (which left every pano "unknown").
    sfm = {'a': (LAT0, LNG0), 'b': (LAT0 + 0.001, LNG0)}
    results = tmp_path / 'results.band.jsonl'
    _results(results, sfm)
    (tmp_path / 'results.band.jsonl.submission.json').write_text(json.dumps(
        {'input_file': results.name, 'total_lines': 2, 'endpoints': {ENDPOINT: {
            'submitted_lines': 2, 'labels_submitted': 5, 'min_confidence': 0.3,
            'first_submission_utc': '2026-09-05T01:05:03Z',
            'last_submission_utc': '2026-09-22T19:03:45Z',          # bumped by the band
            'bands': {'0.3-0.55': {'submitted_lines': 2, 'labels_submitted': 2,
                                   'first_submission_utc': '2026-09-22T19:03:45Z',
                                   'last_submission_utc': '2026-09-22T19:03:45Z'}}}}}),
        encoding='utf-8')
    live, problems = cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-10T00:00:00Z')
    assert problems == [] and live == sfm
    assert cm.fallback_moved(['a', 'b'], sfm, live) == ([], [])
    # before the base began, nothing was live
    assert cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-01T00:00:00Z') == ({}, [])


def test_a_record_without_first_submission_falls_back_to_last(tmp_path):
    # _record writes no first_submission_utc (records from before the field)
    _results(tmp_path / 'results.jsonl', {'a': (LAT0, LNG0)})
    _record(tmp_path / 'results.jsonl', '2026-09-05T01:39:04Z', 1)
    assert cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-05T01:00:00Z') == ({}, [])
    assert cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-05T02:00:00Z')[0] \
        == {'a': (LAT0, LNG0)}


def test_unreadable_timestamps_are_problems_never_errors(tmp_path):
    assert cm._utc('not a time') is None and cm._utc(None) is None
    _results(tmp_path / 'results.jsonl', {'a': (LAT0, LNG0)})
    _record(tmp_path / 'results.jsonl', 'garbled', 1)
    live, problems = cm.live_positions_as_of(tmp_path, ENDPOINT, '2026-09-21T00:00:00Z')
    assert live == {} and len(problems) == 1 and 'garbled' in problems[0]
    live, problems = cm.live_positions_as_of(tmp_path, ENDPOINT, 'whenever')
    assert live == {} and 'whenever' in problems[-1]
