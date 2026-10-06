"""run_census (issue #57): the tier, rig-mask and nadir-band tallies on two records."""
import run_census as rc


def _rec(model, date, pitch, roll, dets):
    return {'pano': {'camera_make': 'GoPro', 'camera_model': model, 'width': 5760,
                     'height': 2880, 'capture_date': date, 'camera_pitch': pitch,
                     'camera_roll': roll, 'copyright': 'Ville', 'license': 'etalab-2.0',
                     'panoramax_instance': 'panoramax.ign.fr'},
            'detections': [{'x_normalized': 0.5, 'y_normalized': y, 'confidence': c}
                           for y, c in dets]}


def test_census_counts_tiers_rig_and_band():
    c = rc.census([_rec('MAX', '2024-05', None, None, [(0.6, 0.9), (0.85, 0.4)]),
                   _rec('MAX', '2026-01', 0.0, 0.0, [(0.79, 0.2)])], band_y=0.8)
    assert c['panos'] == 2
    assert c['pose'] == {'absent': 1, 'zeros': 1}
    assert c['years'] == {'2024': 1, '2026': 1}
    op, bench = c['detections'][0.30], c['detections'][0.55]
    assert (op['detections'], op['panos_with'], op['in_band'], op['on_rig']) == (2, 1, 1, 1)
    assert (bench['detections'], bench['in_band'], bench['on_rig']) == (1, 0, 0)


def _ladybug(date, dets):
    rec = _rec('Ladybug', date, None, None, dets)
    rec['pano'].update(camera_make='Point Grey', width=8192, height=4096)
    return rec


def test_rig_detections_split_by_rig_and_year():
    recs = [_rec('MAX', '2024-05', None, None, [(0.6, 0.9), (0.6, 0.4)]),
            _rec('MAX', '2024-07', None, None, []),
            _ladybug('2020-06', [(0.6, 0.35)]),
            _ladybug('2022-03', [(0.6, 0.7), (0.62, 0.6), (0.7, 0.2)])]
    c = rc.census(recs, band_y=0.8)
    # The pre-existing counters are untouched by the new table.
    assert c['panos'] == 4
    assert c['rigs'] == {('GoPro', 'MAX', '5760x2880'): 2,
                         ('Point Grey', 'Ladybug', '8192x4096'): 2}
    assert c['years'] == {'2024': 2, '2020': 1, '2022': 1}
    assert c['detections'][0.30]['detections'] == 5
    assert c['detections'][0.55]['detections'] == 3
    rows = rc.rig_detection_rows(c)
    assert len(rc.RIG_DETECTION_HEADER) == len(rows[0]) == 13
    by_key = {tuple(r[:4]): r[4:] for r in rows}
    # panos, then (detections, per pano, panos with, share) at 0.30 and at 0.55
    assert by_key[('GoPro', 'MAX', '5760x2880', '2024')] == [2, 2, 1.0, 1, 0.5, 1, 0.5, 1, 0.5]
    assert by_key[('Point Grey', 'Ladybug', '8192x4096', '2020')] == [1, 1, 1.0, 1, 1.0,
                                                                      0, 0.0, 0, 0.0]
    assert by_key[('Point Grey', 'Ladybug', '8192x4096', '2022')] == [1, 2, 2.0, 1, 1.0,
                                                                      2, 2.0, 1, 1.0]
    assert rows[0][:4] == ['GoPro', 'MAX', '5760x2880', '2024']   # largest group first
