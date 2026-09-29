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
