"""depth_standin.py's invariant check (#47): the reindex may move only the spread and the
two new stand-in columns. Offline; plain dicts shaped like index.csv rows."""
import depth_standin as ds


def _row(pid, **kw):
    base = {'panorama_id': pid, 'camera_height_m': '1.9000', 'ground_tilt_deg': '1.200',
            'degenerate': '0', 'height_spread_m': '0.3000', 'sha256': 'ab'}
    return {**base, **kw}


def test_invariant_allows_only_the_moved_columns():
    old = {'A': _row('A')}
    new = {'A': _row('A', height_spread_m='0.1000', n_standin_planes='1',
                     standin_pixel_share='0.0400')}
    assert ds.invariant_violations(old, new) == []


def test_invariant_catches_a_moved_height_and_a_lost_row():
    old = {'A': _row('A'), 'B': _row('B')}
    new = {'A': _row('A', camera_height_m='2.5000')}
    got = ds.invariant_violations(old, new)
    assert ('B', '<row>', 'present', 'absent') in got
    assert ('A', 'camera_height_m', '1.9000', '2.5000') in got


def test_summarize_counts_standins_and_changes():
    rows = [{'pano_id': 'A', 'year': '2024', 'spread_old': 0.30, 'spread_new': 0.10,
             'n_standin': 1, 'standin_share': 0.04},
            {'pano_id': 'B', 'year': '2024', 'spread_old': 0.20, 'spread_new': 0.20,
             'n_standin': 0, 'standin_share': 0.0}]
    s = ds.summarize(rows)
    assert (s['n_measured'], s['n_with_standin'], s['n_changed_ge_0p05']) == (2, 1, 1)
    assert (s['n_narrowed'], s['n_widened']) == (1, 0)


def _pano(pid, old, new):
    return {'pano_id': pid, 'year': '2024', 'spread_old': old, 'spread_new': new,
            'n_standin': 1, 'standin_share': 0.01}


def test_a_change_of_exactly_the_bar_counts():
    """0.1234 - 0.0734 is 0.04999... in floats; the index holds 4 decimals."""
    assert abs(0.1234 - 0.0734) < ds.CHANGE_M
    assert ds.summarize([_pano('A', 0.1234, 0.0734)])['n_changed_ge_0p05'] == 1


def test_sigma_changed_is_the_union_above_the_floor():
    f = ds.SIGMA_FLOOR_SPREAD_M
    rows = [_pano('A', f + 0.1, f - 0.1),      # crosses down: counted
            _pano('B', f - 0.1, f + 0.1),      # crosses up: counted
            _pano('C', f + 0.2, f + 0.1),      # both above, moved: counted
            _pano('D', f - 0.2, f - 0.1),      # both floored: sigma unchanged
            _pano('E', f + 0.1, f + 0.1)]      # above, unchanged
    s = ds.summarize(rows)
    assert s['n_sigma_changed'] == 3
    assert (s['n_old_above_sigma_floor'], s['n_new_above_sigma_floor']) == (3, 3)


def test_pano_rows_skips_a_row_the_invariant_already_reported():
    """A lost row is an invariant failure (exit 1), not a KeyError."""
    new = {'A': _row('A', height_spread_m='0.1000', n_standin_planes='1',
                     standin_pixel_share='0.0400')}
    assert ds.pano_rows({}, new, {}) == []
