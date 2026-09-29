"""The POST HOC step-3 diagnostics (RampNet#158, review of sidewalk-auto-labeler#112):
geometry helpers of step3_exclusion_diag and step3_gate_diag's displacement tally."""
import step3_exclusion_diag as ed
import step3_gate_diag as gd


def test_degrees_uses_the_benchmark_geometry():
    assert abs(ed.degrees(0.5, 0.5, 0.522, 0.5) - 7.92) < 1e-9       # the 0.022 window
    assert abs(ed.degrees(0.999, 0.5, 0.001, 0.5) - 0.72) < 1e-9     # across the seam


def test_cells_is_chebyshev_on_the_heatmap_grid():
    assert abs(ed.cells(0.5, 0.5, 0.5 + 10 / 1024, 0.5 + 3 / 512) - 10) < 1e-9
    assert abs(ed.cells(0.5, 0.5, 0.5 + 1 / 1024, 0.5 + 11 / 512) - 11) < 1e-9
    assert abs(ed.cells(0.9995, 0.5, 0.0005, 0.5) - 1.024) < 1e-9     # seam


def _rec(pid, dets):
    return {'pano': {'panorama_id': pid},
            'detections': [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                           for x, y, c in dets]}


def test_displacement_separates_same_cell_one_cell_and_moved():
    pinned = {'p': _rec('p', [(0.5, 0.6, 0.8), (0.2, 0.6, 0.7), (0.8, 0.6, 0.9),
                              (0.3, 0.6, 0.2)])}                     # 0.2 is not operational
    floor = {'p': _rec('p', [(0.5, 0.6, 0.8), (0.2 + 1 / 1024, 0.6, 0.65),
                             (0.8 + 3 / 1024, 0.6, 0.9)])}
    rows = gd.displacement(pinned, floor)
    assert [(r['exact'], r['one_cell']) for r in rows] == [(True, True), (False, True),
                                                           (False, False)]
    assert abs(rows[1]['dconf'] - 0.05) < 1e-9
