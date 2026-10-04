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
