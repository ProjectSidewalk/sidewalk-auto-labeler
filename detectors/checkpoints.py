"""Provenance of detectors loaded from a local checkpoint file, not the Hugging Face hub.

CurbRamp's weights come from the hub, and its provenance is the hub revision SHA
(detectors.KNOWN_REVISIONS). A detector that is not published anywhere -- today Crosswalk,
RampNet#274's crosswalk_s4 -- is loaded from a ``.pth`` file instead, so its identity is the
sha256 of the file's bytes. The rule is the same as for the hub: provenance is RESOLVED from
what was loaded, never declared, and an unknown checkpoint refuses to start (issue #39's
reasoning: PS rows are permanent and insert-only, so a guessed training date can never be
corrected afterwards).

The detector reads the file ONCE, hashes those bytes and deserializes the same bytes
(detectors/crosswalk.py), so the hash cannot describe a different file than the one loaded.

This module imports no torch (the test suite runs torch-free).

Example::

    >>> sha = 'ebad379b5f7c1645dc39ccb820cd18500109723ae5b1e5706c1e4521b2ab42b4'
    >>> p = provenance_for_checkpoint(sha)
    >>> p['model_id'], p['model_training_date'], p['model_revision'][:19]
    ('rampnet-crosswalk@ebad379b5f7c', '11-09-2026', 'sha256:ebad379b5f7c')
"""
import hashlib
import re

from detectors import API_VERSION, ModelProvenanceError, ps_training_date

#: model_id prefix per label type: ``<prefix>@<first 12 hex of the sha256>``. Every row PS
#: already holds starts ``rampnet-``, which this keeps (Jon's call; PR description).
MODEL_ID_PREFIX = {'Crosswalk': 'rampnet-crosswalk'}

#: model_repo written when an unknown checkpoint runs under --allow-unknown-model-revision.
UNKNOWN_CHECKPOINT_REPO = 'unpublished:unknown-checkpoint'

# Training date (and name) per checkpoint sha256. HOW TO EXTEND IT for a new checkpoint
# (s7, a retrain): hash the file (`sha256sum`, or the refusal message prints it), add a row
# with the date the checkpoint was TRAINED and the run it came from. If the weights are ever
# pushed to Hugging Face, the hub route (KNOWN_REVISIONS) replaces this row.
KNOWN_CHECKPOINTS = {
    'ebad379b5f7c1645dc39ccb820cd18500109723ae5b1e5706c1e4521b2ab42b4': {
        'name': 'crosswalk_s4',
        'label_type': 'Crosswalk',
        'training_date': '2026-11-09',
        'model_repo': 'unpublished:RampNet#274/crosswalk_s4',
        'note': 'crosswalk_s4 seed 1, final_step_1000.pth, klone job 43075001; adopted under '
                'RampNet#274. 360,360,378 bytes. Copies: klone /gscratch/makelab/jonf/'
                'crosswalk274/v4_s1/, lab share /projects/makeabilitylab/RampNet/archives/'
                'stage1_crosswalk_detector_274/checkpoints/crosswalk_s4/'},
}

# The network the checkpoint is loaded into: RampNet's rampnet/model.py, vendored verbatim as
# detectors/rampnet_keypoint.py (the detectors/rampnet_subcell.py precedent). The commit is the
# last one to touch the file (identical at RampNet main on 2026-11-15); the hash is of its
# bytes with LF line endings. tests/test_crosswalk.py checks it; to update, copy the file
# verbatim from RampNet and change both constants together.
RAMPNET_KEYPOINT_COMMIT = 'ed9ec436c1d6f3b1657ec150f3d4ddee011f9e7a'
RAMPNET_KEYPOINT_SHA256 = '5d2c4b0f2ec7326439b4f2662c0075a28e2cf60465b45893c5f3518c7b3952af'

_SHA256_RE = re.compile(r'[0-9a-f]{64}')


def sha256_hex(data: bytes) -> str:
    """The sha256 of ``data`` as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def provenance_for_checkpoint(sha256, allow_unknown=False, label_type='Crosswalk') -> dict:
    """The five registry.PROVENANCE_KEYS for a checkpoint identified by its sha256.

    ``model_revision`` is ``sha256:<64 hex>``, ``model_id`` ``<prefix>@<first 12 hex>``,
    ``model_training_date`` PS's MM-DD-YYYY. Raises ModelProvenanceError when ``sha256`` is
    malformed, when the checkpoint is in KNOWN_CHECKPOINTS under another label type (a curb
    ramp checkpoint passed as a crosswalk one), or when it is unknown and ``allow_unknown`` is
    False. With ``allow_unknown`` an unknown checkpoint gets a null training date (which
    send_to_ps.py refuses) and model_repo UNKNOWN_CHECKPOINT_REPO, never a guess.
    """
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ModelProvenanceError(f'not a sha256 hex digest: {sha256!r}')
    if label_type not in MODEL_ID_PREFIX:
        raise ModelProvenanceError(f'no local-checkpoint detector for label type {label_type!r}')
    known = KNOWN_CHECKPOINTS.get(sha256)
    if known is not None and known['label_type'] != label_type:
        raise ModelProvenanceError(
            f"checkpoint {sha256[:12]}... is {known['name']}, a {known['label_type']} model, "
            f"not a {label_type} one")
    if known is None and not allow_unknown:
        raise ModelProvenanceError(
            f"The {label_type} checkpoint has sha256 {sha256}, which is not in "
            f"detectors.checkpoints.KNOWN_CHECKPOINTS, so its training date is unknown.\n"
            f"   If this is a new checkpoint, add a row for it with the date it was trained "
            f"(see the comment there).\n"
            f"   --allow-unknown-model-revision runs anyway with model_training_date null "
            f"(recorded in the manifest); send_to_ps.py will refuse that file.")
    return {
        'model_id': f'{MODEL_ID_PREFIX[label_type]}@{sha256[:12]}',
        'model_training_date': ps_training_date(known['training_date']) if known else None,
        'api_version': API_VERSION,
        'model_repo': known['model_repo'] if known else UNKNOWN_CHECKPOINT_REPO,
        'model_revision': f'sha256:{sha256}',
    }
