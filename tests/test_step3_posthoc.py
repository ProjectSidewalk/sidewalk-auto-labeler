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


def _heatmap_fixture(tmp_path, ids=('a', 'b')):
    """A committed dir, a cache dir holding one fake heatmap per id, and a manifest."""
    import hashlib
    import json
    committed, cache = tmp_path / 'data', tmp_path / 'cache'
    committed.mkdir()
    cache.mkdir()
    files = {}
    for pid in ids:
        p = cache / f'{pid}.z3.npy'
        p.write_bytes(pid.encode() * 8)
        files[p.name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest = committed / 'manifest.json'
    manifest.write_text(json.dumps({'files': files}), encoding='utf-8')
    return committed, cache, manifest


def test_committed_heatmap_guard_refuses_to_blank_or_replace_the_hm_columns(tmp_path):
    import pytest
    committed, cache, manifest = _heatmap_fixture(tmp_path)
    out = [committed / 'step3' / 'posthoc_exclusion.csv', None]

    def run(outputs, hm_dir, ids=('a', 'b')):
        ed.require_committed_heatmaps(outputs, hm_dir, set(ids), committed, manifest)

    with pytest.raises(SystemExit, match='--heatmap'):        # no heatmaps: would blank
        run(out, None)
    run(out, cache)                                           # exactly the committed set
    run([tmp_path / 'scratch.csv'], None)                     # exploration: no guard
    run(out, None, ids=())                                    # no GSV rows: nothing to blank
    with pytest.raises(SystemExit, match='differ'):           # a different pano set
        run(out, cache, ids=('a',))
    (cache / 'b.z3.npy').write_bytes(b're-fetched, not equal')
    with pytest.raises(SystemExit, match='not equal'):        # a re-fetch that drifted
        run(out, cache)
    (cache / 'b.z3.npy').unlink()
    with pytest.raises(SystemExit, match='missing'):          # a file gone
        run(out, cache)


def test_committed_heatmap_manifest_lists_the_20_gsv_target_panos():
    import json
    man = json.loads(ed.HEATMAP_MANIFEST.read_text(encoding='utf-8'))
    assert man['published'] is False
    assert len(man['files']) == 20
    assert all(len(v['sha256']) == 64 and v['shape'] == [512, 1024]
               for v in man['files'].values())
    rows = (ed.HEATMAP_MANIFEST.parent / 'posthoc_exclusion.csv').read_text(
        encoding='utf-8').splitlines()
    import csv
    gsv = {r['pano_id'] for r in csv.DictReader(rows) if r['hm_reproduces'] != ''}
    assert {f'{p}.z3.npy' for p in gsv} == set(man['files'])


def test_software_json_must_record_every_package_including_numpy(tmp_path):
    import json
    import pytest
    import floor_infer_archive as fia
    assert 'numpy' in fia.SOFTWARE_PACKAGES
    full = {k: '1' for k in fia.SOFTWARE_PACKAGES}
    p = tmp_path / 'sw.json'
    p.write_text(json.dumps(full), encoding='utf-8')
    assert fia.read_software_json(p) == full
    p.write_text(json.dumps({k: v for k, v in full.items() if k != 'numpy'}), encoding='utf-8')
    with pytest.raises(SystemExit, match='numpy'):
        fia.read_software_json(p)
