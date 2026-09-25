"""mapillary_height: per-rig camera height (issue #53).

Synthetic geometry only: panos placed in a local ENU frame whose detections are the exact
projections of planted ramps at each rig's TRUE height, so the bearing-only instrument
must recover those heights whatever height the association runs at.
"""
import json
import math

import pytest

import geo
import fuse_sites as fs
import mapillary_height as mh

BASE = (37.54, -77.43)
FRAME = geo.LocalFrame(*BASE)
RAMPS = [(0.0, 0.0), (16.0, 0.0), (0.0, 16.0), (16.0, 16.0)]
# (pano id, east, north, rig, true height)
CAMERAS = [('a1', 8.0, -5.0, 'r1', 1.8), ('a2', -5.0, 8.0, 'r1', 1.8),
           ('a3', 8.0, 21.0, 'r1', 1.8), ('b1', 21.0, 8.0, 'r2', 2.5),
           ('b2', 8.0, 8.0, 'r2', 2.5), ('b3', -5.0, -5.0, 'r2', 2.5)]


def synthetic_panos(source='mapillary'):
    panos = []
    for pid, pe, pn, rig, h in CAMERAS:
        dets = []
        for te, tn in RAMPS:
            d = math.hypot(te - pe, tn - pn)
            if d > 22.0:
                continue
            bearing = math.degrees(math.atan2(te - pe, tn - pn))
            dets.append((len(dets), 0.5 + geo.norm_deg(bearing) / 360.0,
                         0.5 + math.atan(h / d) / math.pi, 0.9))
        lat, lng = FRAME.to_latlng(pe, pn)
        panos.append(fs.SlimPano(pid, lat, lng, 0.0, None, None, '2025-08', source, dets,
                                 sequence_id=f'seq-{rig}'))
    return panos


RIG_OF = {pid: rig for pid, _e, _n, rig, _h in CAMERAS}


def test_rig_key_normalises_case_make_and_firmware():
    assert mh.rig_key('GoPro', 'GoPro Fusion FS1.04.01.80.00') == 'gopro/fusion'
    assert mh.rig_key('GoPro', 'Fusion') == 'gopro/fusion'
    assert mh.rig_key('Trimble', 'Trimble mx7') == mh.rig_key('Trimble', 'Trimble MX7')
    assert mh.rig_key('none', 'none') == mh.rig_key(None, None) == 'unknown'
    assert [mh.speed_class(v) for v in (None, 1.0, 5.0, 12.0)] \
        == ['unknown', 'walk', 'mixed', 'drive']


def test_bearing_fixed_point_recovers_each_rigs_height():
    sweep = mh.sweep_implied(synthetic_panos(), mh.production_params(),
                             heights=(1.4, 1.8, 2.2, 2.6, 3.0))
    rows = mh.instrument_a(sweep, RIG_OF.get, n_boot=20)
    assert rows['r1']['h_star'] == pytest.approx(1.8, abs=0.1)
    assert rows['r2']['h_star'] == pytest.approx(2.5, abs=0.1)
    assert abs(rows['r1']['slope']) < 0.1     # exact geometry: nothing follows the assoc.
    assert rows['r1']['ci_lo'] <= rows['r1']['h_star'] <= rows['r1']['ci_hi']


def test_fixed_point_is_undefined_when_the_instrument_only_follows():
    assert mh.fixed_point([1.0, 2.0, 3.0], [1.1, 2.1, 3.1])[0] is None
    h, a, b = mh.fixed_point([1.0, 2.0, 3.0], [1.5, 1.75, 2.0])
    assert (a, b) == (pytest.approx(1.25), pytest.approx(0.25))
    assert h == pytest.approx(1.25 / 0.75)


def _a(h, slope=0.2, lo=None, hi=None, n=500, sites=200):
    return {'h_star': h, 'slope': slope, 'n_panos': n, 'n_sites': sites,
            'ci_lo': h - 0.05 if lo is None else lo, 'ci_hi': h + 0.05 if hi is None else hi,
            'boot_sd': 0.03}


def _b(h, views=1000):
    return {'h_scale': h, 'h_scale_lo': h - 0.1, 'h_scale_hi': h + 0.1, 'n_views': views,
            'h_scale_with_offset': h}


def test_decision_rule_clauses():
    assert mh.decide_group(_a(2.0), _b(2.1))[0] == pytest.approx(2.05)
    assert 'support' not in mh.decide_group(_a(2.0, n=99), _b(2.1))[1]
    assert 'identifiable' not in mh.decide_group(_a(2.0, slope=0.61), _b(2.1))[1]
    assert 'agreement' not in mh.decide_group(_a(2.0), _b(2.3))[1]
    # material: the CI must exclude 2.6 AND the effect exceed 0.2 m
    assert 'material' not in mh.decide_group(_a(2.45), _b(2.45))[1]
    assert 'material' not in mh.decide_group(_a(2.0, hi=2.7), _b(2.0))[1]
    assert mh.decide_group(None, None)[0] is None


def test_table_resolves_every_sequence_and_keeps_failures_at_default():
    groups_of = {'p1': {'rig': 'r1', 'sequence': 's1'}, 'p2': {'rig': 'r1', 'sequence': 's2'},
                 'p3': {'rig': 'r2', 'sequence': 's3'}}
    table, notes = mh.build_table(groups_of, {'r1': _a(1.9), 'r2': _a(2.5, slope=0.9)},
                                  {'r1': _b(2.0), 'r2': _b(2.5)}, {}, {})
    assert set(table['sequences']) == {'s1', 's2', 's3'}
    assert all(k in table['groups'] for k in table['sequences'].values())
    r1, r2 = table['groups']['r1'], table['groups']['r2']
    assert r1['applied'] and r1['height_m'] == pytest.approx(1.95)
    assert not r2['applied'] and r2['height_m'] == geo.DEFAULT_CAMERA_HEIGHT_M
    assert 'identifiable' in r2['reason'] and notes['r1']['grain'] == 'rig'


def test_disagreeing_sequences_go_per_sequence_and_small_ones_inherit():
    groups_of = {f'p{i}': {'rig': 'r1', 'sequence': s} for i, s in enumerate('abc')}
    a_seq = {'a': _a(1.5, n=30, sites=20), 'b': _a(2.5, n=30, sites=20),
             'c': _a(2.0, n=5, sites=3)}
    views = {'a': 100, 'b': 100, 'c': 10}
    boot = {'a': _a(1.5, n=150, sites=60), 'b': _a(2.5, n=20, sites=10)}
    table, notes = mh.build_table(groups_of, {'r1': _a(2.0)}, {'r1': _b(2.0)}, a_seq, views,
                                  a_seq_boot=boot, b_seq={'a': _b(1.55), 'b': _b(2.5)})
    assert notes['r1']['grain'] == 'sequence' and table['grain'] == 'sequence'
    assert table['sequences'] == {'a': 'r1@a', 'b': 'r1', 'c': 'r1'}
    assert table['groups']['r1@a']['height_m'] == pytest.approx(1.525)


def _record(p, source, meta_only_sequence=False):
    pano = {'panorama_id': p.pano_id, 'lat': p.lat, 'lng': p.lng,
            'camera_heading': 0.0, 'camera_pitch': None, 'camera_roll': None,
            'capture_date': p.capture_date, 'source': source,
            'sequence_id': None if meta_only_sequence else p.sequence_id,
            'source_metadata': {'sequence': p.sequence_id, 'make': 'GoPro',
                                'model': 'GoPro Max'}}
    return {'detections': [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                           for _i, x, y, c in p.detections], 'pano': pano}


def _write_run(tmp_path, source='mapillary', meta_only=()):
    run = tmp_path / 'run'
    run.mkdir()
    with open(run / 'results.jsonl', 'w', encoding='utf-8') as f:
        for p in synthetic_panos(source):
            f.write(json.dumps(_record(p, source, p.pano_id in meta_only)) + '\n')
    return run


def _table(r1_applied=True):
    groups = {'r1': {'height_m': 1.8 if r1_applied else 2.6, 'sigma_m': 0.1,
                     'applied': r1_applied},
              'r2': {'height_m': 2.6, 'sigma_m': None, 'applied': False}}
    return {'schema': 1, 'source': 'mapillary', 'default_m': 2.6, 'grain': 'rig',
            'groups': groups, 'sequences': {'seq-r1': 'r1', 'seq-r2': 'r2'}}


def test_load_results_applies_the_table_and_refuses_a_mismatch(tmp_path):
    run = _write_run(tmp_path)
    path = mh.write_table(run, _table(), recommended=False)
    panos, _ = fs.load_results(run / 'results.jsonl', read_heights=False, height_table=path)
    heights = {p.pano_id: p.camera_height_m for p in panos}
    assert heights['a1'] == 1.8 and heights['b1'] is None
    counts = fs.camera_height_counts(panos, fs.FuseParams(camera_height_m=geo.PER_RIG))
    assert (counts['applied'], counts['fallback']) == (3, 3)
    with open(run / 'results.jsonl', 'a', encoding='utf-8') as f:
        f.write('\n')                                   # the file changed after measuring
    with pytest.raises(ValueError, match='different results.jsonl'):
        fs.load_results(run / 'results.jsonl', read_heights=False, height_table=path)


def test_load_results_refuses_a_gsv_run(tmp_path):
    run = _write_run(tmp_path, source='launch')
    path = mh.write_table(run, _table(), recommended=False)
    with pytest.raises(ValueError, match='crowdsourced'):
        fs.load_results(run / 'results.jsonl', read_heights=False, height_table=path)


def _sites_json(panos, params):
    sites, frame, _ = fs.fuse(panos, params)
    return [json.dumps(fs.site_to_json(s, frame)) for s in sites]


def test_default_fuse_is_byte_identical_with_or_without_a_table(tmp_path):
    run = _write_run(tmp_path)
    plain, _ = fs.load_results(run / 'results.jsonl', read_heights=False)
    path = mh.write_table(run, _table(), recommended=False)
    tabled, _ = fs.load_results(run / 'results.jsonl', read_heights=False, height_table=path)
    base = _sites_json(plain, fs.FuseParams())
    assert _sites_json(tabled, fs.FuseParams()) == base          # 2.6 ignores the table
    assert _sites_json(tabled, fs.FuseParams(camera_height_m=geo.PER_RIG)) != base
    path = mh.write_table(run, _table(r1_applied=False), recommended=False)
    none, _ = fs.load_results(run / 'results.jsonl', read_heights=False, height_table=path)
    assert _sites_json(none, fs.FuseParams(camera_height_m=geo.PER_RIG)) == base


def test_gate_needs_three_changed_cities_and_no_recall_loss():
    def rows(p90, med, rec):
        return [{'arm': 'off', 'p90_gt_to_site_m': 2.0, 'median_gt_to_site_m': 1.0,
                 'recall_off_pool_2p5m': 0.9},
                {'arm': 'per-rig', 'p90_gt_to_site_m': p90, 'median_gt_to_site_m': med,
                 'recall_off_pool_2p5m': rec}]
    info = {'panos': 100, 'panos_changed': 50}
    good = {c: (rows(1.9, 0.9, 0.9), info) for c in 'xyz'}
    ok, clauses, _ = mh.gate_verdict(good, {})
    assert ok and all(clauses.values())
    bad = dict(good, w=(rows(2.0, 1.0, 0.85), {'panos': 100, 'panos_changed': 0}))
    ok, clauses, _ = mh.gate_verdict(bad, {})
    assert not ok and not clauses['iii'] and clauses['ii']
    two = {c: good[c] for c in 'xy'}
    assert not mh.gate_verdict(two, {})[1]['ii']


# --- review fixes: bootstrap multiplicity, instrument B, the gate, key symmetry, eval_sites

def test_bootstrap_draw_keeps_multiplicity():
    sites = [(10, [('p', 1.0)]), (11, [('p', 3.0), ('q', 2.0)]), (12, [('r', 9.0)])]
    full = {'p': 2, 'q': 1, 'r': 1}
    # every site once: exactly the point estimate
    assert mh.resampled_median(sites, [0, 1, 2], full) \
        == mh._group_median([e for _s, es_ in sites for e in es_])
    # site 12 drawn three times outweighs the rest: r's 9.0 carries weight 3 of 5
    assert mh.resampled_median(sites, [2, 2, 2, 1], full) == 9.0
    # ...whereas collapsing repeats (the pre-review bug) would give p, q, r one vote each
    assert mh._group_median([e for i in {2, 1} for e in sites[i][1]]) == 3.0
    assert mh.weighted_median([(1.0, 1), (2.0, 1), (3.0, 1), (4.0, 1)]) == 2.5


def _uniform_panos(h):
    """The synthetic scene with every camera at true height h (one rig)."""
    global CAMERAS
    saved = CAMERAS
    CAMERAS = [(pid, e, n, 'r1', h) for pid, e, n, _r, _h in saved]
    try:
        return synthetic_panos()
    finally:
        CAMERAS = saved


def test_instrument_b_reads_a_lower_true_height_below_the_default():
    sites, _frame = mh.site_views(_uniform_panos(2.1), mh.production_params())
    assert sites, 'the scene must yield sites with >= 3 views'
    rows = mh.instrument_b(sites, lambda _pid: 'r1', min_rows=1)
    # ranges run 2.6/2.1 long; the null-corrected scale identity must see most of it
    assert rows['r1']['h_scale'] == pytest.approx(2.1, abs=0.2)
    assert rows['r1']['k_net'] > 1.05


def _gt_for(panos):
    verdicts, bundle = {}, {}
    for p in panos:
        ops = [(x, y, c) for _i, x, y, c in p.detections if c >= mh.BENCHMARK_CONFIDENCE]
        bundle[p.pano_id] = ops
        verdicts[p.pano_id] = {'dets': [True] * len(ops), 'missed': [], 'no_missed': True}
    return verdicts, bundle


def test_height_gate_scores_the_true_heights_better():
    panos = synthetic_panos()
    for p in panos:            # the table knows each rig's true height
        p.camera_height_m = 1.8 if RIG_OF[p.pano_id] == 'r1' else 2.5
    verdicts, bundle = _gt_for(panos)
    rows, info = mh.height_gate(verdicts, bundle, panos)
    r = {row['arm']: row for row in rows}
    assert info['panos_changed'] == 6 and info['sites_scored'] > 0
    assert r['per-rig']['median_gt_to_site_m'] < r['off']['median_gt_to_site_m']
    assert r['per-rig']['median_gt_to_site_m'] < 0.05      # exact geometry, right height


def test_gate_clauses_i_and_iv_fail_on_their_own():
    def rows(p90):
        return [{'arm': 'off', 'p90_gt_to_site_m': 2.0, 'median_gt_to_site_m': 1.0,
                 'recall_off_pool_2p5m': 0.9},
                {'arm': 'per-rig', 'p90_gt_to_site_m': p90, 'median_gt_to_site_m': 0.9,
                 'recall_off_pool_2p5m': 0.9}]
    info = {'panos': 100, 'panos_changed': 50}
    worse = {c: (rows(2.2 if c == 'x' else 1.9), info) for c in 'xyz'}
    ok, clauses, _ = mh.gate_verdict(worse, {})
    assert not ok and not clauses['i'] and clauses['ii'] and clauses['iv']
    steeper = {c: {'off': {'all_slope': -0.1},
                   'per-rig': {'all_slope': -0.2 if c == 'y' else -0.05}} for c in 'xyz'}
    ok, clauses, _ = mh.gate_verdict({c: (rows(1.9), info) for c in 'xyz'}, steeper)
    assert not ok and not clauses['iv'] and clauses['i']


def test_table_key_falls_back_to_source_metadata_sequence(tmp_path):
    run = _write_run(tmp_path, meta_only={'a1'})
    groups = mh.pano_groups(run / 'results.jsonl')
    assert groups['a1']['sequence'] == 'seq-r1'
    path = mh.write_table(run, _table(), recommended=False)
    panos, _ = fs.load_results(run / 'results.jsonl', read_heights=False, height_table=path)
    assert {p.pano_id: p.camera_height_m for p in panos}['a1'] == 1.8


def test_a_sequence_spanning_two_rigs_goes_to_its_majority():
    groups_of = {'p1': {'rig': 'r1', 'sequence': 's'}, 'p2': {'rig': 'r2', 'sequence': 's'},
                 'p3': {'rig': 'r2', 'sequence': 's'}}
    table, _ = mh.build_table(groups_of, {}, {}, {}, {})
    assert table['sequences'] == {'s': 'r2'} and table['grain'] == 'rig'


def _eval_setup(tmp_path, monkeypatch, with_table):
    import eval_sites as es
    run = _write_run(tmp_path)
    bench = tmp_path / 'bench' / 'syn'
    bench.mkdir(parents=True)
    panos = synthetic_panos()
    verdicts, bundle = _gt_for(panos)
    (bench / 'verdicts.json').write_text(json.dumps({'panos': verdicts}), encoding='utf-8')
    with open(bench / 'records.jsonl', 'w', encoding='utf-8') as f:
        for p in panos:
            f.write(json.dumps({'pano': {'panorama_id': p.pano_id}, 'detections': [
                {'x_normalized': x, 'y_normalized': y, 'confidence': c}
                for x, y, c in bundle[p.pano_id]]}) + '\n')
    if with_table:
        mh.write_table(run, _table(), recommended=False)
    monkeypatch.setattr('sys.argv', ['eval_sites.py', 'syn', '--benchmark-root',
                                     str(tmp_path / 'bench'), '--run-dir', str(run),
                                     '--camera-height-m', 'per-rig',
                                     '--out', str(tmp_path / 'out'), '--radius-sweep'])
    return es, run


def test_eval_sites_per_rig_exits_cleanly_without_a_table(tmp_path, monkeypatch):
    es, _run = _eval_setup(tmp_path, monkeypatch, with_table=False)
    with pytest.raises(SystemExit, match='camera-height table'):
        es.main()


def test_eval_sites_per_rig_runs_with_a_table_and_refuses_a_stale_one(tmp_path, monkeypatch):
    es, run = _eval_setup(tmp_path, monkeypatch, with_table=True)
    es.main()
    assert (tmp_path / 'out').exists()
    with open(run / 'results.jsonl', 'a', encoding='utf-8') as f:
        f.write('\n')
    with pytest.raises(SystemExit, match='different results.jsonl'):
        es.main()


def test_fuse_sites_cli_exits_cleanly_without_a_table(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    monkeypatch.setattr('sys.argv', ['fuse_sites.py', str(run), '--camera-height-m',
                                     'per-rig', '--out', str(tmp_path / 's.jsonl')])
    with pytest.raises(SystemExit, match='camera-height table'):
        fs.main()
