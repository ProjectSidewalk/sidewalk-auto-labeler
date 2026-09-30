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


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path, keys, assignment=None, human=(), inventory=False):
    run = tmp_path / 'run'
    (run / 'provenance_gate').mkdir(parents=True)
    (run / 'ps_clustering_eval').mkdir(parents=True)
    labels_file = run / 'provenance_gate' / 'raw_labels.geojson'
    labels_file.write_text('{"features": []}', encoding='utf-8')
    (run / 'ps_clustering_eval' / 'clusters.geojson').write_text('{"features": [1]}',
                                                                 encoding='utf-8')
    (run / 'results.jsonl').write_text('', encoding='utf-8')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    sha = _sha(labels_file)
    (bundle / 'snapshot.json').write_text(json.dumps({
        'city': 'vancouver', 'labels': {'sha256': sha, 'n_features': len(keys),
                                        'fetched_at': 'x'},
        'seed_arms': {'deployed': {'sha256': _sha(run / 'ps_clustering_eval' / 'clusters.geojson')},
                      'fusion': {'results_sha256': _sha(run / 'results.jsonl')}}}),
        encoding='utf-8')
    corner = {'corner_id': 'u1', 'type': 'signalised', 'n_labels': len(keys), 'pilot': True,
              'centre': {'lat': 45.6, 'lng': -122.6},
              'labels': [{'key': str(k), 'lat': 45.6, 'lng': -122.6,
                          'user_kind': 'human' if k in human else 'ai'} for k in keys]}
    if inventory:
        corner['inventory'] = [{'lat': 45.6, 'lng': -122.6, 'unit_id': 'CR1'}]
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


def _fixture(**kw):
    return dict({'schema': crs.SCHEMA, 'rubric_version': 1, 'seed_arm': 'deployed',
                 'corners': {'u1': dict(U1, stratum={'type': 'signalised'})}}, **kw)


def _run(tmp_path, monkeypatch, toy, df, run, bundle, *extra):
    monkeypatch.setattr(ic, 'server_arms', lambda *k, **kw: (df, toy, {}))
    return crs.main(['vancouver', '--run-dir', str(run), '--bundle', str(bundle),
                     '--out', str(tmp_path / 'out')] + list(extra))


def test_refuses_arm_inputs_other_than_the_bundles(tmp_path, monkeypatch):
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), _fixture())
    (run / 'results.jsonl').write_text('{"changed": 1}\n', encoding='utf-8')
    toy = {n: [[1, 2], [3, 4, 7], [5, 6]] for n in crs.ARMS}
    with pytest.raises(SystemExit, match='results.jsonl sha256'):
        _run(tmp_path, monkeypatch, toy, df, run, bundle)
    (run / 'ps_clustering_eval' / 'clusters.geojson').write_text('{}', encoding='utf-8')
    with pytest.raises(SystemExit, match='deployed clusters sha256'):
        _run(tmp_path, monkeypatch, toy, df, run, bundle)
    assert _run(tmp_path, monkeypatch, toy, df, run, bundle, '--allow-arm-mismatch') == 0
    text = (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')
    assert 'ARM INPUT MISMATCH' in text


@pytest.mark.parametrize('mutate, needle', [
    (lambda a: a['corners']['u1']['labels'].update({'3': 'r3 '}), 'label value'),
    (lambda a: a['corners']['u1']['labels'].update({'3': 'ramp'}), 'label value'),
    (lambda a: a.update(rubric_version=2), 'rubric_version'),
])
def test_check_assignments_rejects_bad_values_and_foreign_rubrics(tmp_path, monkeypatch,
                                                                  mutate, needle):
    a = json.loads(json.dumps(_fixture()))
    mutate(a)
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), a)
    with pytest.raises(SystemExit, match=needle):
        _run(tmp_path, monkeypatch, {n: [[1]] for n in crs.ARMS}, df, run, bundle)


def test_every_arm_scores_the_same_labels():
    # arm P holds neither the human label 8 nor label 7; arm F holds everything. With human
    # labels out of the key set and unheld labels completed as singletons, coverage is equal
    unit = dict(U1, labels=dict(U1['labels'], **{'8': 'r3'}))
    keys = {'u1': [str(k) for k in range(1, 8)]}            # 8 is human: excluded
    p_arm, n_p = ic.complete_arm(ic.arm_index([[1], [2], [3, 4], [5], [6]]), keys['u1'])
    f_arm, n_f = ic.complete_arm(ic.arm_index([[1, 2], [3, 4], [7, 8], [5, 6]]), keys['u1'])
    assert (n_p, n_f) == (1, 0) and p_arm['7'] == {('singleton', '7')}
    mp = ic.assignment_metrics(p_arm, {'u1': unit}, keys)
    mf = ic.assignment_metrics(f_arm, {'u1': unit}, keys)
    assert mp['coverage'] == mf['coverage'] == pytest.approx(3 / 4)
    assert (mp['covered'], mf['covered']) == (3, 3)
    assert (mp['split'], mf['split']) == (1, 0)


def test_human_labels_and_singletons_through_main(tmp_path, monkeypatch):
    unit = dict(U1, labels=dict(U1['labels'], **{'8': 'r3'}), stratum={'type': 'signalised'})
    run, bundle, df = _bundle(tmp_path, list(range(1, 9)), _fixture(corners={'u1': unit}),
                              human=(8,))
    toy = {n: [[1, 2], [3, 4], [7, 8], [5, 6]] for n in crs.ARMS}
    toy[ic.REVIEW_BASE] = [[1], [2], [3, 4], [5], [6]]       # no human, no label 7
    assert _run(tmp_path, monkeypatch, toy, df, run, bundle) == 0
    text = (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')
    assert '1 human label(s) excluded' in text and f'{ic.REVIEW_BASE} 1' in text
    with open(tmp_path / 'out' / 'arms.csv', newline='', encoding='utf-8') as f:
        cov = {r['arm']: r['coverage'] for r in csv.DictReader(f) if r['stratum'] == 'all'}
    assert len(set(cov.values())) == 1                      # coverage is arm-independent


def test_calibration_drops_units_edited_after_the_inventory_reveal(tmp_path, monkeypatch):
    unit = dict(U1, stratum={'type': 'signalised'}, inventory_seen=True,
                edited_after_inventory=True)
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), _fixture(corners={'u1': unit}),
                              inventory=True)
    toy = {n: [[1, 2], [3, 4, 7], [5, 6]] for n in crs.ARMS}
    assert _run(tmp_path, monkeypatch, toy, df, run, bundle) == 0
    text = (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')
    assert '1 were edited after the reveal and are DROPPED' in text
    assert 'no reviewed unit carries inventory points' in text


# --- can't judge (RampNet protocol, Amendment 3) -------------------------------------------

def test_cant_judge_units_count_in_no_metric():
    cj = dict(U1, complete=False, cant_judge=True, cant_judge_reason='under a bridge deck')
    m = ic.assignment_metrics(ARM_A, {'u1': cj, 'u2': U2}, KEYS)
    assert m['units'] == 0 and m['ramps'] == 0
    # defensive: even a file that (invalidly) says both is not scored
    both = dict(U1, cant_judge=True)
    assert ic.assignment_metrics(ARM_A, {'u1': both}, KEYS)['units'] == 0


@pytest.mark.parametrize('fields, needle', [
    ({'cant_judge': 'yes'}, 'not a boolean'),
    ({'cant_judge': True, 'cant_judge_reason': 'x'}, 'cant_judge and complete'),
    ({'cant_judge': True, 'complete': False, 'cant_judge_reason': ' '}, 'without a cant_judge_reason'),
])
def test_check_assignments_cant_judge_invariants(tmp_path, monkeypatch, fields, needle):
    a = json.loads(json.dumps(_fixture()))
    a['corners']['u1'].update(fields)
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), a)
    with pytest.raises(SystemExit, match=needle):
        _run(tmp_path, monkeypatch, {n: [[1]] for n in crs.ARMS}, df, run, bundle)


def test_cant_judge_listed_with_its_reason(tmp_path, monkeypatch):
    a = json.loads(json.dumps(_fixture()))
    a['corners']['u1'].update(complete=False, cant_judge=True,
                              cant_judge_reason='tree canopy hides the corner')
    run, bundle, df = _bundle(tmp_path, list(range(1, 8)), a)
    assert _run(tmp_path, monkeypatch, {n: [[1, 2], [3, 4, 7], [5, 6]] for n in crs.ARMS},
                df, run, bundle) == 0
    text = (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')
    assert "0 complete, 1 can't judge" in text
    assert "## Can't judge" in text and 'signalised 1' in text
    assert '| u1 | signalised | tree canopy hides the corner |' in text
