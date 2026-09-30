"""cluster_review_score.py + inventory_clustering.assignment_metrics (RampNet#224): the
pre-registered assignment metrics on a toy unit with one split, one merge, one not_ramp
cluster, one unsure label and one uncovered point; the paired table; the decision rule at
its boundaries; the self-consistency check; the "no GT yet" path and a scored fixture.
Needs pandas/scipy/haversine (the #56 arms' analysis deps), not shapely."""
import csv
import hashlib
import json

import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')

import pandas as pd  # noqa: E402

import cluster_review_score as crs  # noqa: E402
import inventory_clustering as ic  # noqa: E402

# U1: r1 = {1, 2}, r2 = {3, 4}, r3 = {7}; 5 not_ramp; 6 unsure; one sure + one unsure
# uncovered point. U2 is incomplete and must be ignored.
U1 = {'labels': {'1': 'r1', '2': 'r1', '3': 'r2', '4': 'r2', '5': 'not_ramp', '6': 'unsure',
                 '7': 'r3'},
      'ramps': {'r1': {}, 'r2': {}, 'r3': {}},
      'uncovered': [{'lat': 0, 'lng': 0, 'unsure': False}, {'lat': 0, 'lng': 0, 'unsure': True}],
      'complete': True}
U2 = {'labels': {'9': 'r1'}, 'ramps': {'r1': {}}, 'uncovered': [], 'complete': False}
UNITS = {'u1': U1, 'u2': U2}
KEYS = {'u1': [str(k) for k in range(1, 8)], 'u2': ['9']}
# arm A: r1 split over c0/c1; c2 merges r2 and r3; c3 is all not_ramp; c4 holds only the
# unsure label (so it does not touch the unit); label 9 sits in an incomplete unit
ARM_A = ic.arm_index([[1], [2], [3, 4, 7], [5], [6], [9]])
ARM_B = ic.arm_index([[1, 2], [3, 4], [7], [5, 6]])


def test_assignment_metrics_exact_numbers():
    m = ic.assignment_metrics(ARM_A, UNITS, KEYS)
    assert m['units'] == 1
    assert (m['ramps'], m['covered'], m['split']) == (3, 3, 1)
    assert m['split_rate'] == pytest.approx(1 / 3)
    assert (m['merge_k'], m['merge_strict_k'], m['merge_n']) == (1, 0, 1)
    assert (m['clusters'], m['invalid_clusters']) == (4, 1)
    assert (m['labels_held'], m['not_ramp_held']) == (6, 1)
    assert m['uncovered_sure'] == 1 and m['coverage'] == pytest.approx(3 / 4)
    assert m['per_ramp'] == {('u1', 'r1'): 2, ('u1', 'r2'): 1, ('u1', 'r3'): 1}
    b = ic.assignment_metrics(ARM_B, UNITS, KEYS)
    assert (b['split'], b['merge_k'], b['merge_n']) == (0, 0, 2) and b['merge_rate'] == 0.0
    assert (b['clusters'], b['invalid_clusters']) == (4, 1)   # {5, 6}: only not_ramp in-window


def test_labels_the_arm_does_not_hold_are_not_covered():
    arm = ic.arm_index([[1, 2], [3, 4], [5]])                  # label 7 absent (e.g. human)
    m = ic.assignment_metrics(arm, UNITS, KEYS)
    assert (m['ramps'], m['covered']) == (3, 2)
    assert m['coverage'] == pytest.approx(2 / 4)


def test_paired_ramp_table():
    a = ic.assignment_metrics(ARM_A, UNITS, KEYS)['per_ramp']
    b = ic.assignment_metrics(ARM_B, UNITS, KEYS)['per_ramp']
    t = ic.paired_ramp_table(a, b)
    assert (t['fixed'], t['broken'], t['both'], t['neither']) == (1, 0, 0, 2)
    t = ic.paired_ramp_table(b, a)
    assert (t['fixed'], t['broken']) == (0, 1)
    assert ic.paired_ramp_table({('u', 'r1'): 2}, {('u', 'r1'): 0})['only_a'] == 1


def _arms(split, merge, cov):
    base = {'split_rate': 0.20, 'merge_rate': 0.02, 'coverage': 0.90}
    return {ic.REVIEW_BASE: base,
            ic.REVIEW_ARM: {'split_rate': split, 'merge_rate': merge, 'coverage': cov}}


def test_decision_rule_boundaries():
    assert ic.assignment_verdict(_arms(0.15, 0.02, 0.90))[0] == ic.PASS        # exactly -0.05
    assert ic.assignment_verdict(_arms(0.151, 0.02, 0.90))[0] == ic.NOT_ESTABLISHED
    assert ic.assignment_verdict(_arms(0.10, 0.03, 0.90))[0] == ic.PASS        # merge +0.01
    assert ic.assignment_verdict(_arms(0.10, 0.031, 0.90))[0] == ic.NOT_ESTABLISHED
    assert ic.assignment_verdict(_arms(0.10, 0.02, 0.89))[0] == ic.PASS        # coverage -0.01
    assert ic.assignment_verdict(_arms(0.10, 0.02, 0.889))[0] == ic.NOT_ESTABLISHED
    assert ic.assignment_verdict(_arms(None, 0.02, 0.9))[0] == ic.NO_GT
    assert ic.assignment_verdict({})[0] == ic.NO_GT


def test_self_consistency_of_a_derived_assignment():
    corners = [{'corner_id': 'u1', 'labels': [{'key': str(k), 'lat': 0.0, 'lng': 0.0}
                                              for k in range(1, 8)]}]
    derived = crs.derived_assignment(ARM_A, corners)
    m = ic.assignment_metrics(ARM_A, derived, {'u1': KEYS['u1']})
    assert m['split'] == 0 and m['merge_k'] == 0 and m['covered'] == 5


def _bundle(tmp_path, keys, assignment=None):
    run = tmp_path / 'run'
    (run / 'provenance_gate').mkdir(parents=True)
    labels_file = run / 'provenance_gate' / 'raw_labels.geojson'
    labels_file.write_text('{"features": []}', encoding='utf-8')
    (run / 'results.jsonl').write_text('', encoding='utf-8')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    sha = hashlib.sha256(labels_file.read_bytes()).hexdigest()
    (bundle / 'snapshot.json').write_text(json.dumps({
        'city': 'vancouver', 'labels': {'sha256': sha, 'n_features': len(keys),
                                        'fetched_at': 'x'},
        'seed_arms': {'deployed': {'sha256': 'd'}}}), encoding='utf-8')
    corner = {'corner_id': 'u1', 'type': 'signalised', 'n_labels': len(keys), 'pilot': True,
              'labels': [{'key': str(k), 'lat': 45.6, 'lng': -122.6} for k in keys]}
    (bundle / 'corners.jsonl').write_text(json.dumps(corner) + '\n', encoding='utf-8')
    if assignment is not None:
        (bundle / 'assignments.json').write_text(json.dumps(dict(assignment, snapshot_sha256=sha)),
                                                 encoding='utf-8')
    df = pd.DataFrame({'label_id': list(keys), 'lat': [45.6] * len(keys),
                       'lng': [-122.6] * len(keys)})
    return run, bundle, df


def test_no_gt_yet_report(tmp_path, monkeypatch):
    run, bundle, df = _bundle(tmp_path, [1, 2])
    toy = {name: [[1], [2]] for name in crs.ARMS}
    monkeypatch.setattr(ic, 'server_arms', lambda *a, **k: (df, toy, {}))
    out = tmp_path / 'out'
    assert crs.main(['vancouver', '--run-dir', str(run), '--bundle', str(bundle),
                     '--out', str(out)]) == 0
    text = (out / 'report.md').read_text(encoding='utf-8')
    assert 'NO GT YET' in text and 'self-consistency' in text and '-- OK' in text
    with open(out / 'arms.csv', newline='', encoding='utf-8') as f:
        assert list(csv.reader(f)) == [crs.ARM_FIELDS]


def test_scores_a_fixture_assignment(tmp_path, monkeypatch):
    a = {'schema': crs.SCHEMA, 'rubric_version': 1, 'seed_arm': 'deployed',
         'corners': {'u1': dict(U1, stratum={'type': 'signalised'})}}
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), a)
    toy = {name: [[1, 2], [3, 4, 7], [5, 6]] for name in crs.ARMS}   # merge only
    toy[ic.REVIEW_BASE] = [[1], [2], [3, 4, 7], [5], [6]]           # split + merge
    monkeypatch.setattr(ic, 'server_arms', lambda *k, **kw: (df, toy, {}))
    out = tmp_path / 'out'
    assert crs.main(['vancouver', '--run-dir', str(run), '--bundle', str(bundle),
                     '--out', str(out)]) == 0
    text = (out / 'report.md').read_text(encoding='utf-8')
    assert '**PASS**' in text          # split 1/3 -> 0, merge 1/1 -> 1/1, coverage 3/4 both
    with open(out / 'ramps.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert {(r['ramp'], r[ic.REVIEW_BASE], r[ic.REVIEW_ARM]) for r in rows} == \
        {('r1', '2', '1'), ('r2', '1', '1'), ('r3', '1', '1')}


def test_refuses_a_file_on_another_snapshot(tmp_path, monkeypatch):
    a = {'schema': crs.SCHEMA, 'rubric_version': 1, 'corners': {'u1': U1}}
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), a)
    p = bundle / 'assignments.json'
    p.write_text(json.dumps(dict(json.loads(p.read_text()), snapshot_sha256='0' * 64)),
                 encoding='utf-8')
    monkeypatch.setattr(ic, 'server_arms',
                        lambda *k, **kw: (df, {n: [[1]] for n in crs.ARMS}, {}))
    with pytest.raises(SystemExit, match='cannot be scored'):
        crs.main(['vancouver', '--run-dir', str(run), '--bundle', str(bundle),
                  '--out', str(tmp_path / 'out')])
