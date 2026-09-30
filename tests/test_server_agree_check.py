"""server_agree_check: the validations tally and the human-vs-AI cluster reads, on a
synthetic city. Offline; no feed is fetched."""
import csv

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
    ]
    ncr = [_lab(50, HUMAN, _m(100), _m(3), pano='p4')]              # NoCurbRamp near ramp B
    clusters = [
        {'label_cluster_id': 1, 'street_edge_id': 1, 'lng': _m(1), 'lat': 0,
         'label_ids': [10, 11, 12, 20], 'users': [AI, HUMAN]},
        {'label_cluster_id': 2, 'street_edge_id': 1, 'lng': _m(100), 'lat': 0,
         'label_ids': [30], 'users': [AI]},
    ]
    streets = [{'street_edge_id': 1, 'user_ids': [AI, HUMAN], 'audit_count': 1},
               {'street_edge_id': 2, 'user_ids': [AI], 'audit_count': 0}]
    lines, r = sac.human_section(cr, ncr, clusters, streets, HUMAN, AI, tmp_path)
    text = '\n'.join(lines)
    assert r['n_human'] == 3 and r['in_cluster'] == 1 and r['misses'] == 1
    assert r['completed'] == 1 and r['clusters_completed'] == 2
    # human -> AI: 1/3 in-cluster, 2/3 within 7.5 m (ramp B at 6 m), 2/3 within 10 m
    assert '| in a server cluster that also holds an AI label | 1/3 |' in text
    assert '| … or an AI cluster within 7.5 m | 2/3 |' in text
    assert '| … or an AI cluster within 10 m | 2/3 |' in text
    # AI -> human on the completed street: cluster 1 holds the label; cluster 2 within 7.5 m
    assert '| holds a label of the human\'s | 1/2 |' in text
    assert '| … or human CurbRamp within 7.5 m | 2/2 |' in text
    assert '| … or human CurbRamp within 5 m | 1/2 |' in text
    # by views: 3+ views confirmed 1/1, 1 view confirmed 1/1
    assert '| 3+ | 1 | 1 |' in text and '| 1 | 1 | 1 |' in text
    # pano frame: label 20 shares pano p1 with AI label 10, 10 units apart -> within radius
    assert '1 of 3 of the human\'s labels are on a pano the AI labelled' in text
    assert 'in 1 of those 1 an AI label sits within' in text
    # own vote x nearby: 10 (agree, near), 30 (agree, near at 6 m); 11, 12 (none, near)
    assert '| agree | 2 | 0 |' in text and '| none | 2 | 0 |' in text
    with open(tmp_path / 'misses.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['label_id'] for r in rows] == ['40'] and rows[0]['street_completed'] == 'True'
