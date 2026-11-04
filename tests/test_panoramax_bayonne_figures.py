"""The #162 review's bootstrap fixes in scripts/panoramax_bayonne_figures.py: the
sequence-cluster CI (S1) and the capture-age cut stratified by same-sequence mates (S2)."""
import csv
import random
from types import SimpleNamespace as N

import panoramax_bayonne_figures as bf


def _mean(rows):
    return {'v': sum(r['v'] for r in rows) / len(rows)}


def test_sequence_bootstrap_resamples_whole_sequences():
    # 20 sites, each one view, from 2 sequences whose views sit at 0 and 1. Resampling
    # sites gives a tight CI around 0.5; resampling the 2 sequences can draw one twice.
    rows = [{'site_id': i, 'q': 'ab'[i % 2], 'v': float(i % 2)} for i in range(20)]
    _, site_ci = bf._boot(rows, _mean, n=200)
    _, seq_ci = bf._boot(rows, _mean, n=200, cluster=lambda r: r['q'])
    assert seq_ci['v'] == (0.0, 1.0)
    assert 0.0 < site_ci['v'][0] < site_ci['v'][1] < 1.0


def test_site_bootstrap_unchanged_by_the_cluster_argument():
    # the committed site-bootstrap CIs must not move: default clusters are site_id
    rows = [{'site_id': i // 3, 'v': random.Random(i).random()} for i in range(30)]
    assert bf._boot(rows, _mean) == bf._boot(rows, _mean, cluster=lambda r: r['site_id'])


def test_seq_key_falls_back_to_the_pano():
    key = bf._seq_key({'p1': N(sequence_id='s'), 'p2': N(sequence_id=None)})
    assert key({'pano_id': 'p1'}) == 's' and key({'pano_id': 'p2'}) == 'p2'


def test_site_sequence_is_the_modal_sequence():
    by_id = {'p1': N(sequence_id='b'), 'p2': N(sequence_id='a'), 'p3': N(sequence_id='b')}
    assert bf._site_sequence(N(views=[N(pano_id=p) for p in by_id]), by_id) == 'b'
    assert bf._site_sequence(N(views=[N(pano_id=p) for p in ('p1', 'p2')]), by_id) == 'a'


def test_cuts_split_capture_age_within_seq_mates(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, 'DATA', tmp_path)
    rows, by_id = [], {}
    for i in range(12):
        pid = f'p{i}'
        by_id[pid] = N(sequence_id=f's{i % 3}')
        rows.append({'site_id': i // 3, 'pano_id': pid, 'n_views': 3, 'site_n_sequences': 2,
                     'delta_months': 0 if i < 8 else 20, 'chi2': 1.0 + i,
                     'dist_m': 1.0, 'along_m': 0.5, 'cross_m': 0.5,
                     'seq_mates': 'all_same' if i < 3 else 'none' if i % 2 else 'some'})
    bf._cuts_data({bf.CITY: (None, by_id, None, None, rows, None)})
    with open(tmp_path / 'cuts.csv', newline='', encoding='utf-8') as f:
        out = {(r['dimension'], r['bucket']): r for r in csv.DictReader(f)}
    assert out[('n_views', '3')]['n_views'] == '12'
    assert out[('delta_months', '0')]['n_views'] == '8'
    assert out[('delta_months', '19-36')]['n_views'] == '4'
    # all-same-sequence views are all newest-month here, as in Bayonne
    assert ('delta_months|seq_mates=all_same', 'newest') in out
    assert ('delta_months|seq_mates=all_same', 'older') not in out
    none_new = [r for r in rows if r['seq_mates'] == 'none' and r['delta_months'] == 0]
    got = out[('delta_months|seq_mates=none', 'newest')]
    assert int(got['n_views']) == len(none_new)
    assert float(got['chi2_dof']) == round(sum(r['chi2'] for r in none_new)
                                           / (2 * len(none_new)), 4)
    assert float(got['chi2_dof_seq_lo']) <= float(got['chi2_dof']) <= float(got['chi2_dof_seq_hi'])
