"""Pure-logic checks for scripts/crosswalk_fp_filters.py (no data files, no network)."""
import math

import crosswalk_fp_filters as cf


def test_angles_wrap_at_the_seam():
    assert cf.wrap180(181.0) == -179.0
    assert round(cf.circular_mean_deg([179.0, -179.0]), 6) == -180.0
    assert cf.circular_mean_deg([]) is None
    # image azimuth 0 is the centre column; the seam is +/-180
    assert cf.image_azimuth(512) == 0.0 and cf.image_azimuth(0) == -180.0


def test_heatmap_position_and_dip():
    assert cf.heatmap_pos(7906.25, 2203.125, 8000, 4000) == (1012, 282)
    assert cf.heatmap_pos(7999.0, 2000.0, 8000, 4000)[0] == 0   # wraps, never 1024
    assert cf.dip_deg(256) == 0.0 and math.isclose(cf.dip_deg(384), 45.0)


def test_beside_band_is_symmetric_and_bounded():
    assert cf.in_band(35.0, 90.0, 30, 40, 45) and cf.in_band(35.0, -90.0, 30, 40, 45)
    assert cf.in_band(35.0, 135.0, 30, 40, 45) and not cf.in_band(35.0, 136.0, 30, 40, 45)
    assert not cf.in_band(41.0, 90.0, 30, 40, 45)        # outside the dip band
    assert cf.in_band(35.0, 0.0, 30, 40, 90)             # H = 90 is a pure dip band


def test_masks_count_sequences_or_panos_per_camera_model():
    labs = [dict(camera_model='m', sequence_id=s, pano_id=s + '1', col=11, row=276) for s in 'abc']
    labs.append(dict(camera_model='other', sequence_id='d', pano_id='d1', col=11, row=276))
    assert cf.rig_cells(labs, 8, 3) == {('m', 1, 34)}
    assert [cf.in_rig_cell(x, {('m', 1, 34)}, 8) for x in labs] == [True, True, True, False]
    # within-sequence recurrence counts panos of the label's own sequence only
    assert cf.within_sequence_recurrence(labs, 8, 2) == [False] * 4


def test_stratified_estimator_weights_and_missing_votes():
    units = [dict(stratum='k', y=1), dict(stratum='k', y=0), dict(stratum='k', y=None),
             dict(stratum='d', y=0), dict(stratum='d', y=1)]
    pop = {'k': 100, 'd': 10}
    assert cf.stratum_weights(units, pop) == {'k': 50.0, 'd': 5.0}
    p, lost, tot = cf.weighted_stats(units, [False] * 5, pop)
    assert math.isclose(p, (50 + 5) / (100 + 10)) and lost == 0 and tot == 55
    # removing the kept stratum's Agree loses 50 of 55 estimated true positives
    p, lost, tot = cf.weighted_stats(units, [True, False, False, False, False], pop)
    assert lost == 50 and math.isclose(p, 5 / 60)
    assert cf.unweighted_stats(units, [False, True, False, False, False]) == (2 / 3, 0, 2, 1, 2)
