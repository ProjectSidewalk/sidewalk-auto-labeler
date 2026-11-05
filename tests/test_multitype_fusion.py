"""Per-type fusion (issue #168 item 4): FuseParams per label type, fuse_sites --label-type,
and the --label-type plumbing in eval_sites / eval_ps_clustering. A CurbRamp fuse must be
unchanged: same params, same sites.jsonl bytes, same output path. Torch-free, no network."""
import json

import pytest

import fuse_sites as fs
from detectors import registry as reg
from test_fuse_sites import _demo_city, _record_line


@pytest.fixture
def crosswalk_spec(monkeypatch):
    """A second registered type with its own fusion knobs (none exists yet, #168)."""
    monkeypatch.setitem(reg.SPECS, 'Crosswalk', reg.DetectorSpec(
        'Crosswalk', 'nowhere', 'Nothing', 0.2, 0.5, 0.6, False,
        {'max_match_m': 12.0, 'gate_chi2': 13.8}))


def _legacy_lines():
    return [_record_line(p) for p in _demo_city()]


def _two_type(line):
    """The same pano as a {CurbRamp, Crosswalk} line: CurbRamp's detections keep their
    indices, then one Crosswalk detection (straight ahead, 0.8) is appended."""
    rec = json.loads(line)
    rec['detections'] = ([{**d, 'label_type': 'CurbRamp'} for d in rec['detections']]
                         + [{'x_normalized': 0.5, 'y_normalized': 0.6, 'confidence': 0.8,
                             'label_type': 'Crosswalk'}])
    rec.update(label_type=None, label_types=['Crosswalk', 'CurbRamp'],
               models={'CurbRamp': {}, 'Crosswalk': {}})
    return json.dumps(rec)


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


# --- FuseParams per type ------------------------------------------------------------------

def test_curb_ramp_params_are_the_defaults():
    """CurbRamp's spec values are FuseParams' defaults and its overrides are empty, so every
    bare FuseParams() an analysis script builds is still exactly CurbRamp's."""
    assert reg.spec_for('CurbRamp').fuse_overrides == {}
    assert fs.fuse_params_for('CurbRamp') == fs.FuseParams()
    assert fs.FuseParams().label_type == 'CurbRamp'
    # a None override is "flag not given": the type's value stands
    assert fs.fuse_params_for('CurbRamp', gate_chi2=None, floor=None) == fs.FuseParams()
    assert fs.fuse_params_for('CurbRamp', max_match_m=6.0).max_match_m == 6.0
    # label_type is the last field, so asdict order (sites_meta.json) only gains a key
    assert list(fs.asdict(fs.FuseParams()))[-1] == 'label_type'


def test_unknown_label_type_refuses():
    with pytest.raises(reg.UnknownLabelTypeError, match='known: CurbRamp'):
        fs.FuseParams(label_type='Crosswalk')
    with pytest.raises(reg.UnknownLabelTypeError):
        fs.fuse_params_for('Crosswalk')


def test_spec_then_overrides_then_flags(crosswalk_spec):
    p = fs.fuse_params_for('Crosswalk')
    assert (p.label_type, p.floor, p.min_confidence, p.mask_rig) == ('Crosswalk', 0.2, 0.5,
                                                                      False)
    assert (p.max_match_m, p.gate_chi2) == (12.0, 13.8)
    assert p.residual_per_dof_max == fs.FuseParams.residual_per_dof_max   # not overridden
    q = fs.fuse_params_for('Crosswalk', max_match_m=9.0, min_confidence=0.7)
    assert (q.max_match_m, q.min_confidence, q.gate_chi2) == (9.0, 0.7, 13.8)


def test_flag_defaults_leave_the_type_in_charge():
    args = fs.build_parser().parse_args(['runs/x'])
    for flag in ('floor', 'min_confidence', 'max_range_m', 'gate_chi2', 'max_match_m',
                 'residual_per_dof_max', 'sigma_scale'):
        assert getattr(args, flag) is None, flag


# --- the CLI: output paths and equivalence ------------------------------------------------

def test_curb_ramp_cli_fuse_is_unchanged(tmp_path):
    """The default fuse writes sites.jsonl, byte-identical to fusing with FuseParams() at a
    fixed height, and sites_meta.json records the type."""
    run = tmp_path / 'run'
    src = _write(run / 'results.jsonl', _legacy_lines())
    fs.main([str(run), '--camera-height-m', '2.6'])
    panos, _ = fs.load_results(src, read_heights=False)
    sites, frame, stats = fs.fuse(panos, fs.FuseParams())
    fs.write_sites(sites, frame, stats, fs.FuseParams(), tmp_path / 'ref.jsonl',
                   tmp_path / 'ref_meta.json')
    assert (run / 'sites.jsonl').read_bytes() == (tmp_path / 'ref.jsonl').read_bytes()
    meta = json.loads((run / 'sites_meta.json').read_text(encoding='utf-8'))
    assert meta['params']['label_type'] == 'CurbRamp'
    assert meta['params']['max_match_m'] == fs.FuseParams.max_match_m


def test_two_type_file_fuses_each_type_to_its_own_files(tmp_path, crosswalk_spec):
    """A CurbRamp fuse of a two-type file is byte-identical to the legacy file's; a Crosswalk
    fuse writes sites.Crosswalk.jsonl + sites.Crosswalk_meta.json under the type's params and
    never touches sites.jsonl."""
    legacy = tmp_path / 'legacy'
    two = tmp_path / 'two'
    _write(legacy / 'results.jsonl', _legacy_lines())
    _write(two / 'results.jsonl', [_two_type(line) for line in _legacy_lines()])
    for run in (legacy, two):
        fs.main([str(run), '--camera-height-m', '2.6'])
    assert (two / 'sites.jsonl').read_bytes() == (legacy / 'sites.jsonl').read_bytes()

    before = (two / 'sites.jsonl').read_bytes()
    fs.main([str(two), '--label-type', 'Crosswalk', '--camera-height-m', '2.6'])
    assert (two / 'sites.jsonl').read_bytes() == before
    meta = json.loads((two / 'sites.Crosswalk_meta.json').read_text(encoding='utf-8'))
    assert meta['params']['label_type'] == 'Crosswalk'
    assert (meta['params']['max_match_m'], meta['params']['min_confidence']) == (12.0, 0.5)
    members = [m for line in (two / 'sites.Crosswalk.jsonl').read_text(
        encoding='utf-8').splitlines() for m in json.loads(line)['members']]
    # every member is the appended Crosswalk detection (index 2), never a CurbRamp one
    assert members and {m['det_index'] for m in members} == {2}


# --- eval_sites / eval_ps_clustering plumbing ---------------------------------------------

def test_eval_sites_label_type_reaches_the_loader(tmp_path, crosswalk_spec):
    """eval_sites' loader keeps refusing a multi-type file for CurbRamp (its benchmark contract),
    and with --label-type loads only that type's detections."""
    import eval_sites
    bench = tmp_path / 'bench' / 'city'
    bench.mkdir(parents=True)
    (bench / 'verdicts.json').write_text(json.dumps({'panos': {}}), encoding='utf-8')
    (bench / 'records.jsonl').write_text('', encoding='utf-8')
    path = _write(tmp_path / 'results.jsonl', [_two_type(line) for line in _legacy_lines()])
    with pytest.raises(ValueError):
        eval_sites.load_city_at_height('city', tmp_path / 'bench', tmp_path, 2.6,
                                       results_path=path)
    _v, _b, panos, _h, _a = eval_sites.load_city_at_height(
        'city', tmp_path / 'bench', tmp_path, 2.6, results_path=path, label_type='Crosswalk')
    assert {d[0] for p in panos for d in p.detections} == {2}
    assert eval_sites.default_out_dir(tmp_path, 'CurbRamp').name == 'fusion_eval'
    assert eval_sites.default_out_dir(tmp_path, 'Crosswalk').name == 'fusion_eval.Crosswalk'


def test_every_spec_is_a_server_label_type():
    """A spec's label_type goes out verbatim as the submission's labelType (PS's enum)."""
    assert len(reg.PS_LABEL_TYPES) == 9 and 'CurbRamp' in reg.PS_LABEL_TYPES
    for name, spec in reg.SPECS.items():
        assert name == spec.label_type and name in reg.PS_LABEL_TYPES
