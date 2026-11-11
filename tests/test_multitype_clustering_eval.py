"""eval_ps_clustering --label-type (#168 item 4): the server pulls, the offline labels and the
AI-label pixel map read one type's detections; CurbRamp is unchanged. Needs pandas/scipy/
haversine (analysis-only), so it skips where they are missing (CI)."""
import json

import pytest

pytest.importorskip('pandas')
pytest.importorskip('scipy')
pytest.importorskip('haversine')

import eval_ps_clustering as epc  # noqa: E402
from test_multitype_fusion import (_legacy_lines, _two_type, _write,  # noqa: E402,F401
                                   crosswalk_spec)


def _sized(line):
    """A fixture line with the pano dimensions send_to_ps's pixel transform needs."""
    rec = json.loads(line)
    rec['pano'].update(width=4096, height=2048)
    return json.dumps(rec)


def _lines():
    return [_sized(line) for line in _legacy_lines()]


def test_eval_ps_clustering_label_type(tmp_path, crosswalk_spec):
    """The API pulls name the type, and the offline labels / pixel map read only that type's
    detections, keeping each one's index in the full stored list."""
    assert epc.api_labels('CurbRamp') == epc.API_LABELS
    assert epc.api_clusters('CurbRamp') == epc.API_CLUSTERS
    assert 'labelType=Crosswalk' in epc.api_labels('Crosswalk')
    assert 'labelType=Crosswalk' in epc.api_clusters('Crosswalk')

    legacy = _write(tmp_path / 'a.jsonl', _lines())
    two = _write(tmp_path / 'b.jsonl', [_two_type(line) for line in _lines()])
    a, a_of = epc.synthesize_labels(legacy, 0.55)
    b, b_of = epc.synthesize_labels(two, 0.55)
    assert a.equals(b) and a_of == b_of                  # CurbRamp: unchanged
    c, c_of = epc.synthesize_labels(two, 0.55, label_type='Crosswalk')
    assert set(c.label_type) == {'Crosswalk'} and {i for _, i in c_of.values()} == {2}
    assert len(c) == len(_lines())
    assert epc.synthesize_rig_count(two, 0.0, 'Crosswalk') == 0     # its spec: no rig mask
    m_a = epc.label_to_detection(legacy, a)
    m_b = epc.label_to_detection(two, b)
    assert m_a == m_b
    m_c, _amb, _dup = epc.label_to_detection(two, c, label_type='Crosswalk')
    assert m_c == c_of
    args = epc.build_parser().parse_args(['city'])
    assert args.label_type == 'CurbRamp'
    assert epc.default_out(args, tmp_path).name == 'ps_clustering_eval'
    args = epc.build_parser().parse_args(['city', '--label-type', 'Crosswalk'])
    assert epc.default_out(args, tmp_path).name == 'ps_clustering_eval.Crosswalk'


def test_eval_ps_clustering_run_refuses_a_type_without_its_server_threshold(tmp_path,
                                                                           crosswalk_spec):
    """The PS threshold arms are CurbRamp's cut; another type is refused before any I/O."""
    args = epc.build_parser().parse_args(['city', '--label-type', 'Crosswalk',
                                          '--run-dir', str(tmp_path)])
    with pytest.raises(SystemExit, match='PS_THRESHOLD_KM'):
        epc.run(args)
