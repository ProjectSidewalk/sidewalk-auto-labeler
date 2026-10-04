"""server_agree_check: the validations tally and the human-vs-AI cluster reads, on a
synthetic city. Offline; no feed is fetched."""
import csv
import json

import server_agree_check as sac

AI = sac.AI_USER
HUMAN = 'human-1'


def _lab(lid, user, lng, lat, pano='p1', validations=(), street=1, x=100, y=100):
    return {'label_id': lid, 'user_id': user, 'pano_id': pano, 'lng': lng, 'lat': lat,
            'street_edge_id': street, 'validations': list(validations),
            'pano_x': x, 'pano_y': y, 'pano_width': 1000, 'pano_height': 500}


def _vote(user, v, kind='Human'):
    return {'user_id': user, 'validation': v, 'validator_type': kind}


def _m(deg_lng):
    """degrees of longitude at lat 0 for about `deg_lng` metres."""
    return deg_lng / 111320.0


def test_validations_tally_excludes_ai_validator_and_counts_unsure_as_validated():
    labels = [
        _lab(1, AI, 0, 0, validations=[_vote(HUMAN, 'Agree')]),
        _lab(2, AI, 0, 0, validations=[_vote(HUMAN, 'Disagree')]),
        _lab(3, AI, 0, 0, validations=[_vote(HUMAN, 'Unsure')]),
        _lab(4, AI, 0, 0, validations=[_vote('ps-ai', 'Agree', kind='AI')]),
        _lab(5, AI, 0, 0),
    ]
    _, r = sac.validations_section(labels, AI)
    assert (r['agreed'], r['disagreed'], r['validated']) == (1, 1, 3)
    assert r['voters'] == {HUMAN: 3}


def test_human_section_reads_both_directions(tmp_path):
    # Two ramps. Ramp A: AI labels it from 3 panos, the human labels it (same cluster).
    # Ramp B: AI has a single-view cluster 6 m from the human's label, human voted Agree.
    # Ramp C: human only, no AI within 10 m -> a miss.
    cr = [
        _lab(10, AI, 0, 0, pano='p1', validations=[_vote(HUMAN, 'Agree')]),
        _lab(11, AI, _m(1), 0, pano='p2'),
        _lab(12, AI, _m(2), 0, pano='p3'),
        _lab(20, HUMAN, _m(1), 0, pano='p1', x=110, y=100),        # ramp A, same pano as 10
        _lab(30, AI, _m(100), 0, pano='p4', validations=[_vote(HUMAN, 'Agree')]),
        _lab(31, HUMAN, _m(106), 0, pano='p5'),                    # ramp B, 6 m off
        _lab(40, HUMAN, _m(300), 0, pano='p6'),                    # ramp C, AI miss
        _lab(60, 'human-2', _m(100), 0, pano='p7'),                # a third user, in cluster 2
    ]
    ncr = [_lab(50, HUMAN, _m(100), _m(3), pano='p4')]              # NoCurbRamp near ramp B
    clusters = [
        {'label_cluster_id': 1, 'street_edge_id': 1, 'lng': _m(1), 'lat': 0,
         'label_ids': [10, 11, 12, 20], 'users': [AI, HUMAN]},
        {'label_cluster_id': 2, 'street_edge_id': 1, 'lng': _m(100), 'lat': 0,
         'label_ids': [30, 60], 'users': [AI, 'human-2']},
    ]
    streets = [{'street_edge_id': 1, 'user_ids': [AI, HUMAN], 'audit_count': 1},
               {'street_edge_id': 2, 'user_ids': [AI], 'audit_count': 0}]
    assert sac.pick_auditor(cr, AI)[0] == HUMAN          # most CurbRamp labels, by rule
    lines, r = sac.human_section(cr, ncr, clusters, streets, HUMAN, AI, tmp_path, city='t')
    text = '\n'.join(lines)
    assert r['n_human'] == 3 and r['in_cluster'] == 1 and r['misses'] == 1
    assert r['completed'] == 1 and r['clusters_completed'] == 2
    # human -> AI, 7.5 m: 2/3 one-to-one vs clusters (A, and B at 6 m), any-cluster 2/3
    assert r['obs'][('o2o_cl', 7.5)] == 2 and r['obs'][('any', 7.5)] == 2
    assert r['obs'][('o2o_lab', 5.0)] == 1      # A only: B's one AI view is 6 m off
    assert 'In a server cluster that also holds an AI label (no radius): 1/3' in text
    # AI -> human on the completed street: cluster 1 holds the label; cluster 2 within 7.5 m
    assert '| holds a label of the human\'s | 1/2 |' in text
    assert '| … or human CurbRamp within 7.5 m | 2/2 |' in text
    assert '| … or human CurbRamp within 5 m | 1/2 |' in text
    # by views: 3+ views confirmed 1/1; cluster 2 is ONE AI view (the third user's label
    # in it is not an AI view)
    assert '| 3+ | 1 | 1 |' in text and '| 1 | 1 | 1 |' in text
    assert r['views'] == {1: 3, 2: 1}
    # pano frame: label 20 shares pano p1 with AI label 10, 0.01 of the width apart
    assert '1 of 3 of the human\'s labels are on a pano the AI labelled' in text
    assert '1 of those 1 match an AI label on that same pano one-to-one' in text
    # own vote x nearby: 10 (agree, near), 30 (agree, near at 6 m); 11, 12 (none, near)
    assert '| agree | 2 | 0 |' in text and '| none | 2 | 0 |' in text
    with open(tmp_path / 'misses.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['label_uid'] for r in rows] == ['t:40'] and rows[0]['street_completed'] == 'True'
    # the nearest AI cluster is 200 m away, outside the 3x3 cell block: still found exactly
    assert abs(float(rows[0]['nearest_ai_cluster_m']) - 199.8) < 0.5
    assert rows[0]['nearest_ai_cluster_uid'] == 't:2'


def test_cluster_precision_puts_back_labels_the_server_left_out():
    # The server clusters only label 1 (agreed). Label 2 (disagreed) sits 3 m away and was
    # left out, as the server does with labels marked incorrect; label 3 is far away.
    labels = [_lab(1, AI, 0, 0, validations=[_vote(HUMAN, 'Agree')]),
              _lab(2, AI, _m(3), 0, validations=[_vote(HUMAN, 'Disagree')]),
              _lab(3, AI, _m(500), 0, validations=[_vote(HUMAN, 'Disagree')])]
    clusters = [{'label_cluster_id': 9, 'label_ids': [1], 'users': [AI]}]
    served, regrouped, loose = sac.cluster_statuses(clusters, labels, AI)
    assert served == {9: 'agreed'} and loose == 2
    assert sorted(regrouped) == ['disagreed', 'unvalidated']   # {1, 2} ties; {3} alone


def test_label_tier_joins_on_send_to_ps_pixel_rounding(tmp_path):
    rec = {'pano': {'panorama_id': 'p1', 'width': 1000, 'height': 500},
           'detections': [{'x_normalized': 0.1004, 'y_normalized': 0.5, 'confidence': 0.6},
                          {'x_normalized': 0.3, 'y_normalized': 0.5, 'confidence': 0.4}]}
    f = tmp_path / 'results.jsonl'
    f.write_text(json.dumps(rec) + '\n', encoding='utf-8')
    keys, files = sac.load_tiers([f])
    assert sac.label_tier({'pano_id': 'p1', 'pano_x': 100, 'pano_y': 250}, keys) == '>= 0.55'
    assert sac.label_tier({'pano_id': 'p1', 'pano_x': 300, 'pano_y': 250}, keys) == '0.30-0.55'
    assert sac.label_tier({'pano_id': 'p1', 'pano_x': 301, 'pano_y': 250}, keys) == 'not joined'
