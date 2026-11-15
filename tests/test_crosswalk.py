"""The Crosswalk detector spec (RampNet#274 crosswalk_s4 in the labeler): its thresholds, the
decode floor, checkpoint provenance, the vendored network, and how Crosswalk runs describe
themselves in records, manifests and submission records. Torch-free: the detector itself is
never constructed here (scripts/crosswalk_parity.py is its GPU acceptance test)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

import detectors
from detectors import checkpoints as ck
from detectors import decode as dec
from detectors import registry as reg

REPO = Path(__file__).resolve().parents[1]
S4_SHA = 'ebad379b5f7c1645dc39ccb820cd18500109723ae5b1e5706c1e4521b2ab42b4'


# --- the spec --------------------------------------------------------------------------

def test_crosswalk_spec_values():
    s = reg.spec_for('Crosswalk')
    assert (s.module, s.cls) == ('detectors.crosswalk', 'CrosswalkDetector')
    assert (s.storage_floor, s.operational_confidence, s.benchmark_confidence) == (0.05, 0.23, 0.23)
    assert s.mask_rig is True and s.border == 'keep'
    assert reg.type_frame('Crosswalk') == {'detection_decode': 'argmax',
                                           'detection_border': 'keep',
                                           'detection_storage_floor': 0.05}


def test_border_field_defaults_for_seven_argument_specs():
    """Every 7-argument DetectorSpec (CurbRamp's, the tests' stand-ins) keeps working."""
    assert reg.spec_for('CurbRamp').border == 'exclude'
    assert reg.DetectorSpec('X', 'm', 'C', 0.1, 0.3, 0.5, False).border == 'exclude'
    with pytest.raises(ValueError, match='per run'):
        reg.type_frame('CurbRamp')


# --- decode floor -----------------------------------------------------------------------

def test_decode_floor_keyword():
    """0.05 finds a 0.07 peak; the default (CurbRamp's 0.1) does not."""
    h = np.zeros((512, 1024), dtype=np.float32)
    h[200, 300] = 0.07
    h[300, 700] = 0.5
    assert [round(c, 2) for _, _, c in dec.detections_from_heatmap(h)] == [0.5]
    assert [round(c, 2) for _, _, c in dec.detections_from_heatmap(h, floor=0.05)] == [0.5, 0.07]
    assert len(dec.detections_from_heatmap(h, border='wrap', floor=0.05)) == 2
    assert len(dec.detections_from_heatmap(h, border='wrap')) == 1


# --- checkpoint provenance --------------------------------------------------------------

def test_known_checkpoint_provenance():
    p = ck.provenance_for_checkpoint(S4_SHA)
    assert p == {'model_id': 'rampnet-crosswalk@ebad379b5f7c',
                 'model_training_date': '11-09-2026',
                 'api_version': detectors.API_VERSION,
                 'model_repo': 'unpublished:RampNet#274/crosswalk_s4',
                 'model_revision': f'sha256:{S4_SHA}'}
    assert set(p) == set(reg.PROVENANCE_KEYS)


def test_unknown_checkpoint_refuses_unless_allowed():
    other = 'f' * 64
    with pytest.raises(detectors.ModelProvenanceError, match='KNOWN_CHECKPOINTS'):
        ck.provenance_for_checkpoint(other)
    p = ck.provenance_for_checkpoint(other, allow_unknown=True)
    assert p['model_training_date'] is None
    assert p['model_repo'] == ck.UNKNOWN_CHECKPOINT_REPO
    assert p['model_id'] == 'rampnet-crosswalk@ffffffffffff'
    with pytest.raises(detectors.ModelProvenanceError, match='not a sha256'):
        ck.provenance_for_checkpoint('abc', allow_unknown=True)


def test_checkpoint_of_another_type_refuses(monkeypatch):
    monkeypatch.setitem(ck.KNOWN_CHECKPOINTS, 'a' * 64,
                        {**ck.KNOWN_CHECKPOINTS[S4_SHA], 'label_type': 'NoCurbRamp'})
    with pytest.raises(detectors.ModelProvenanceError, match='not a Crosswalk one'):
        ck.provenance_for_checkpoint('a' * 64, allow_unknown=True)


def test_sha256_hex():
    assert ck.sha256_hex(b'') == hashlib.sha256(b'').hexdigest()


def test_vendored_keypoint_model_is_rampnets_file_verbatim():
    """detectors/rampnet_keypoint.py must stay byte-identical to RampNet's rampnet/model.py at
    RAMPNET_KEYPOINT_COMMIT. To update: copy the file verbatim and change both constants."""
    data = (REPO / 'detectors' / 'rampnet_keypoint.py').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(data).hexdigest() == ck.RAMPNET_KEYPOINT_SHA256


# --- load_detectors: per-type kwargs ------------------------------------------------------

def test_load_detectors_passes_type_kwargs_to_that_type_only(monkeypatch):
    import sys
    import types
    built = {}

    def fake(name):
        class Fake:
            def __init__(self, **kw):
                built[name] = kw
        return Fake

    mod = types.ModuleType('fake_det_mod_xw')
    mod.A, mod.B = fake('A'), fake('B')
    monkeypatch.setitem(sys.modules, 'fake_det_mod_xw', mod)
    for t in ('A', 'B'):
        monkeypatch.setitem(reg.SPECS, t, reg.SPECS['CurbRamp']._replace(
            label_type=t, module='fake_det_mod_xw', cls=t))
    reg.load_detectors(['B'], type_kwargs={'B': {'checkpoint': 'x.pth'}}, batch_size=1)
    assert built == {'B': {'batch_size': 1, 'checkpoint': 'x.pth'}}
    built.clear()
    reg.load_detectors(['A', 'B'], type_kwargs={'A': {'decode': 'gaussian'}}, batch_size=1)
    assert built['A']['decode'] == 'gaussian' and 'decode' not in built['B']
    assert built['A']['inference_lock'] is built['B']['inference_lock']
    with pytest.raises(ValueError, match='not among the requested types'):
        reg.load_detectors(['A'], type_kwargs={'B': {}})


def test_crosswalk_module_is_not_imported_by_the_registry():
    import subprocess
    import sys
    code = ("import sys; from detectors import registry; registry.spec_for('Crosswalk'); "
            "bad = [m for m in ('torch', 'detectors.crosswalk', 'detectors.rampnet_keypoint') "
            "if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True)
