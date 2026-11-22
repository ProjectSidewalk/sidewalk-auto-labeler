"""The Crosswalk detector: RampNet#274's crosswalk_s4, loaded from a local checkpoint.

The network is the curb-ramp pano model (RampNet ``rampnet/model.py`` ``KeypointModel``,
vendored verbatim as ``detectors/rampnet_keypoint.py``) retrained to put a heatmap peak at
the centre of each crosswalk. Everything but the weights is CurbRampDetector's: the same
2048x4096 bilinear resize, ImageNet normalisation, fp32 forward, inference lock and batching.
So this is a subclass that changes two things:

- **Weights and provenance** (``_load_model``): the checkpoint file is read once, its bytes
  hashed (detectors.checkpoints.provenance_for_checkpoint; an unknown sha256 refuses unless
  --allow-unknown-model-revision) and the SAME bytes deserialized with ``weights_only=True``,
  then loaded strictly into a freshly built KeypointModel. Nothing is fetched from any hub:
  the backbone is built with ``pretrained_backbone=False`` and every weight comes from the
  checkpoint.
- **A fixed frame**: argmax placement, the spec's border rule (``keep``, RampNet's
  exclude_border=False) and storage floor (0.05), the rule the 0.23 operating point was chosen
  under. The constructor takes no decode or border argument, so a CLI flag cannot reach them.

The checkpoint is a raw state_dict (no ``module.`` / ``model.`` prefix, no EMA). Strict
loading means any key mismatch raises instead of leaving layers randomly initialised.

Example::

    det = CrosswalkDetector(checkpoint='final_step_1000.pth')
    det.provenance['model_id']        # 'rampnet-crosswalk@ebad379b5f7c'
    det.detect(pil_image)             # [(x, y, confidence), ...] down to 0.05
"""
import io
from pathlib import Path

import torch

from detectors import DECODE_ARGMAX
from detectors.batching import DEFAULT_BATCH_WAIT_S
from detectors.checkpoints import provenance_for_checkpoint, sha256_hex
from detectors.curb_ramp import CurbRampDetector
from detectors.registry import CROSSWALK, spec_for


def load_state_dict_bytes(data):
    """Deserialize a raw state_dict from checkpoint bytes, tensors only (weights_only)."""
    state_dict = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f'checkpoint is a {type(state_dict).__name__}, not a raw state_dict')
    return state_dict


class CrosswalkDetector(CurbRampDetector):
    """Crosswalk centres in one equirectangular pano; see the module docstring."""

    LABEL_TYPE = CROSSWALK

    def __init__(self, checkpoint, allow_unknown_revision=False, batch_size=1,
                 batch_wait_s=DEFAULT_BATCH_WAIT_S, inference_lock=None):
        if not checkpoint:
            raise ValueError('CrosswalkDetector needs a checkpoint path (--crosswalk-checkpoint)')
        self.checkpoint = Path(checkpoint)
        super().__init__(allow_unknown_revision=allow_unknown_revision, batch_size=batch_size,
                         batch_wait_s=batch_wait_s, decode=DECODE_ARGMAX,
                         border=spec_for(CROSSWALK).border, inference_lock=inference_lock)

    def _load_model(self, allow_unknown_revision):
        from detectors.rampnet_keypoint import KeypointModel
        data = self.checkpoint.read_bytes()
        provenance = provenance_for_checkpoint(sha256_hex(data), allow_unknown_revision,
                                               label_type=CROSSWALK)
        state_dict = load_state_dict_bytes(data)
        del data
        model = KeypointModel(pretrained_backbone=False)
        model.load_state_dict(state_dict, strict=True)
        return model, provenance
