"""peak_anchor (RampNet#158 step 3): the pre-registered peak-anchored target rule, and the
floor re-inference's instrument check."""
import floor_infer_archive as fia
import peak_anchor as pa


def test_pano_distance_wraps_the_seam():
    assert pa.pano_distance(0.999, 0.5, 0.001, 0.5) < 0.0021
    assert abs(pa.pano_distance(0.5, 0.5, 0.5, 0.5 + 0.022 * 2) - 0.022) < 1e-9


def test_rule_emits_the_strongest_subthreshold_peak_in_the_window():
    a = (0.5, 0.6)
    peaks = [(0.505, 0.6, 0.3), (0.51, 0.6, 0.4), (0.7, 0.6, 0.9)]   # 0.9 is far away
    status, p = pa.anchor_peak(a, peaks)
    assert status == 'emit' and p == (0.51, 0.6, 0.4)


def test_rule_withholds_where_the_model_already_fires_or_is_silent():
    a = (0.5, 0.6)
    assert pa.anchor_peak(a, [(0.505, 0.6, 0.3), (0.51, 0.6, 0.7)])[0] == \
        'operational_in_window'
    assert pa.anchor_peak(a, [(0.6, 0.6, 0.4)])[0] == 'no_peak'      # outside 0.022
    assert pa.anchor_peak(a, [(0.5, 0.6, 0.05)])[0] == 'no_peak'     # below the floor


def test_build_falls_back_to_the_flat_anchor_and_marks_emit():
    src = [{'city': 'c', 'site_id': '1', 'pano_id': 't', 'proj_x': '0.5', 'proj_y': '0.6'}]
    peaks = {'c': {'t': [(0.3, 0.6, 0.3)]}}
    flat = pa.build(src, peaks)[0]
    assert (flat['status'], flat['emit'], flat['anchor_via']) == ('no_peak', False, 'flat')
    armed = pa.build(src, peaks, anchors={('c', 1, 't'): (0.301, 0.6)})[0]
    assert (armed['status'], armed['x'], armed['anchor_via']) == ('emit', 0.3, 'arm')
    fell_back = pa.build(src, peaks, anchors={('c', 1, 't'): None})[0]
    assert fell_back['anchor_via'] == 'flat'


def _d(x, y, c):
    return {'x_normalized': x, 'y_normalized': y, 'confidence': c}


def test_instrument_check_matches_operational_sets_within_one_cell():
    old = [_d(0.5, 0.6, 0.8), _d(0.2, 0.55, 0.3)]
    assert fia.reproduces(old, [_d(0.5 + 1 / 1024, 0.6, 0.7), _d(0.9, 0.5, 0.2)])
    assert not fia.reproduces(old, [_d(0.5 + 3 / 1024, 0.6, 0.7)])       # moved
    assert not fia.reproduces(old, [_d(0.5, 0.6, 0.7), _d(0.1, 0.6, 0.6)])  # extra op
    assert fia.reproduces([_d(0.9995, 0.6, 0.8)], [_d(0.0, 0.6, 0.8)])      # seam


# ---- review of sidewalk-auto-labeler#112 (2026-09-29): three planted faults stayed green ----

def test_rule_counts_a_peak_at_exactly_the_benchmark_threshold_as_operational():
    """0.55 itself is operational (>=), as in every benchmark bundle; 0.1 itself is kept."""
    a = (0.5, 0.6)
    assert pa.anchor_peak(a, [(0.505, 0.6, 0.55)])[0] == 'operational_in_window'
    assert pa.anchor_peak(a, [(0.505, 0.6, 0.5499)])[0] == 'emit'
    assert pa.anchor_peak(a, [(0.505, 0.6, 0.1)]) == ('emit', (0.505, 0.6, 0.1))


def test_instrument_check_tolerance_is_exactly_one_cell():
    old = [_d(0.5, 0.6, 0.8)]
    assert fia.reproduces(old, [_d(0.5 + 1 / 1024, 0.6 + 1 / 512, 0.8)])
    assert not fia.reproduces(old, [_d(0.5 + 2 / 1024, 0.6, 0.8)])       # two cells in x
    assert not fia.reproduces(old, [_d(0.5, 0.6 + 2 / 512, 0.8)])        # two cells in y


def _write_jsonl(path, recs):
    import json
    path.write_text(''.join(json.dumps(r) + '\n' for r in recs), encoding='utf-8')


def _gate(tmp_path, n, n_ok):
    """cmd_check over n panos of which n_ok reproduce."""
    import argparse
    pinned, floor = [], []
    for i in range(n):
        pano = {'panorama_id': f'p{i}'}
        pinned.append({'pano': pano, 'detections': [_d(0.5, 0.6, 0.8)]})
        x = 0.5 if i < n_ok else 0.7
        floor.append({'pano': pano, 'detections': [_d(x, 0.6, 0.8)]})
    _write_jsonl(tmp_path / 'r.jsonl', pinned)
    _write_jsonl(tmp_path / 'f.jsonl', floor)
    (tmp_path / 'ids.txt').write_text(''.join(f'p{i}\n' for i in range(n)), encoding='utf-8')
    args = argparse.Namespace(results=tmp_path / 'r.jsonl', ids=tmp_path / 'ids.txt',
                              out=tmp_path / 'f.jsonl', check_out=None)
    return fia.cmd_check(args)


def test_gate_passes_at_95_percent_and_fails_below(tmp_path):
    assert _gate(tmp_path, 20, 19)['passes'] is True           # 0.95 exactly
    assert _gate(tmp_path, 20, 18)['passes'] is False          # 0.90
    assert _gate(tmp_path, 10, 7)['passes'] is False           # bend's 0.70


def test_input_manifest_checks_hashes_and_sizes_against_rampnet():
    floor = {'a': {'floor_reinfer': {'image_size': [4096, 2048]}},
             'b': {'floor_reinfer': {'image_size': [4096, 2048]}}}
    index = {'a': {'sha256': 'x' * 64, 'bytes': '10'}, 'b': {'sha256': 'y' * 64, 'bytes': '11'}}
    man = {'digest': 'd', 'panos': {'a': {'sha256': 'x' * 64, 'width': 4096, 'height': 2048},
                                    'b': {'sha256': 'z' * 64, 'width': 4096, 'height': 2048}}}
    out = fia.input_manifest(['a', 'b'], floor, index=index, imagery_manifest=man)
    assert out['images']['a'] == {'sha256': 'x' * 64, 'bytes': 10, 'image_size': [4096, 2048]}
    assert out['rampnet_check']['sha256_and_size_equal'] == 1
    assert out['rampnet_check']['differ'] == ['b']
