"""The detection-confidence contract shared across the pipeline (issue #27).

Two distinct thresholds govern detections, and they are deliberately not the same:

- DETECTION_STORAGE_FLOOR: results.jsonl stores every heatmap peak at or above this
  floor (capped at MAX_PEAKS_PER_PANO per pano, highest-intensity first) — including
  candidates well below the operational threshold. Sub-threshold evidence is the raw
  material for multi-view consensus fusion, and discarding it at extraction is
  irreversible: the only way to get it back is re-running inference over a whole
  city. Storing low costs only disk, and the per-pano cap bounds that wherever the
  heatmap's noise floor turns out to sit.

- OPERATIONAL_CONFIDENCE: the decision threshold (the pre-#27 ``threshold_abs``).
  Everything that ACTS on detections must filter here: PS submission
  (send_to_ps.py --min-confidence), production fusion (scripts/fuse_sites.py), and
  any analysis that treats a detection as a believed ramp. A record whose stored
  peaks all fall below it is still a valid "checked, nothing found" pano.
  0.30 since 2026-09-21 (issue #20), replacing 0.55: RampNet's threshold sweep
  (RampNet docs/operating_point.md, GT-completeness corrected, seven US splits pooled)
  gives P 0.919 / R 0.796 / F1 0.853 at 0.30 against 0.964 / 0.722 / 0.826 at 0.55 —
  +7.4 recall points for -4.5 precision, 1.86 -> 2.23 detections per pano; 27% of the
  extra "false positives" were ramps the GT had missed. Clovis is the binding split
  (corrected P 0.883). The policy is deliberately recall-first: a false positive is
  cheap to validate on the server, a false negative is never seen again.

- BENCHMARK_CONFIDENCE: the threshold every judged RampNet benchmark bundle was
  exported (scripts/export_benchmark.py) and reviewed at. Verdicts exist only for
  detections at or above it, so anything that JOINS a run to a bundle — the drift
  gate in scripts/eval_sites.py / scripts/mined_precision.py, the export strata, the
  tilt and thinning studies whose numbers are committed — filters here, never at the
  policy threshold above. The two are decoupled on purpose: moving the operating
  point must not silently re-key nine cities of ground truth. Changing THIS value
  means re-exporting and re-judging every bundle.

A third filter is geometric rather than confidence-based — see NADIR_MASK_DEG.

The module also owns MODEL PROVENANCE (issues #39/#6) — which weights produced a record —
because it has the same audience: see KNOWN_REVISIONS and provenance_from_snapshot_dir.

This module imports no torch so main.py, send_to_ps.py, and scripts/ can read the
contract without pulling in the model stack (the test suite runs torch-free); the
detector itself lives in detectors.curb_ramp.
"""
import re
from datetime import date
from pathlib import PurePath

DETECTION_STORAGE_FLOOR = 0.1
MAX_PEAKS_PER_PANO = 50
OPERATIONAL_CONFIDENCE = 0.30
BENCHMARK_CONFIDENCE = 0.55

# --- The nadir (camera-rig) mask -----------------------------------------------------
#
# "Nadir" is the direction straight down from the camera (the opposite of zenith) — in 360
# photography, the bottom of the sphere, which is where the tripod or vehicle appears. It is
# the region panoramic rigs traditionally cover with a "nadir patch" or logo.
#
# In an equirectangular pano the vertical axis IS the dip angle: y_normalized 0.5 is the
# horizon and 1.0 is the nadir. Straight down is the vehicle the camera is mounted on, so a
# detection steep enough lands on the rig itself, never on the street.
#
# NOTE this rule uses only the pixel's position in the image. It says nothing about where
# the label sits on the street network, so a legitimate mid-block ramp — a school crossing,
# a mid-block crosswalk, a driveway cut — is untouched by it. That distinction matters: the
# same Laurens validations show the rig's false positives also sit far from intersections,
# but that is a CONSEQUENCE of the rig artifact (it lands wherever the car drove, while real
# ramps cluster at corners), not an independent signal. Measured 2026-09-22: after this mask,
# a ">30 m from an intersection" rule would have cut 3 validated-true labels for every 1
# false, so it is deliberately NOT implemented.
#
# Measured on Laurens prod, 2026-09-22, from human validations of live AI labels. Of the 158
# labels in this region, 50 have been judged and **every one is false** (0 true; 95% CI on
# precision [0.000, 0.071]). Counted from the database; a /v3/api/rawLabels pull the same day
# reported 48, for reasons never established (the feed itself was then measured to be
# real-time, agreeing with the database row-for-row during live validation), so prefer the
# database when a count has to be exact. They are not scattered: the 158 fall on 156 panos
# across 19 sequences at just seven discrete y values, every one of them GoPro Max — a roof
# rack, fixed in the rig's own frame, re-detected pano after pano. At 2.6 m camera height
# those dips are 1.5-2.1 m of ground range, i.e. on the vehicle.
#
# The separation is exact and has held as the judged set grew from 379 to 536 labels: the
# steepest validated TRUE label is at 46.1 deg, the shallowest validated FALSE one below the
# horizon-ish band is at 51.7 deg, and nothing lies between. 49 deg sits mid-gap. It is
# expressed as an ANGLE, not a range, on purpose: range needs a camera height, which is
# per-pano and currently a constant known to be too high (issue #40), whereas the dip is read
# straight off the pixel and cannot drift.
#
# This mask is NOT a general false-positive filter, and must not be sold as one. With it
# applied, the band's remaining errors are ordinary model mistakes — driveway cuts and the
# like — running at 0.828 precision (77/93 judged) against 0.972 for the >=0.55 tier.
#
# This mask was invisible at the old 0.55 operating point (0 of 708 Laurens labels, 2 of
# 9,526 Richmond) — dropping to 0.30 is what surfaced it.
NADIR_MASK_DEG = 49.0
NADIR_MASK_Y = 0.5 + NADIR_MASK_DEG / 180.0


def on_camera_rig(y_normalized: float) -> bool:
    """Is this detection steep enough to be on the camera vehicle rather than the street?

    The single definition of the rule, so the submission path and any analysis that adopts
    it cannot drift apart. Callers reconstructing a campaign that shipped BEFORE the mask
    existed must not apply it — what is already live is what is already live.
    """
    return y_normalized > NADIR_MASK_Y


# --- Model provenance (issues #39, #6) -----------------------------------------------
#
# Every JSONL line and every manifest run entry says which weights produced it, and Project
# Sidewalk stores that permanently, one `label_ai_info` row per AI label. It used to be three
# hand-maintained literals in main.py that nothing tied to the model actually loaded, so a
# retrained checkpoint would have kept reporting the old training date: provenance that
# looks authoritative while being wrong, with no signal that anything is stale.
#
# Now it is RESOLVED, not declared: detectors.curb_ramp.CurbRampDetector reads the Hugging
# Face revision SHA of the snapshot it actually loaded, and everything else follows from that
# SHA through the pure functions below (torch-free, so the test suite covers them).

MODEL_REPO = "projectsidewalk/rampnet-model"

# The version of the record format Project Sidewalk's /ai/submitLabelsOnPano reads. This one
# IS declared rather than resolved, legitimately: it is a fact about this repo's output
# contract, not about the weights, so it changes only with a change here.
API_VERSION = "1.0.0"

# Training date per Hugging Face revision of MODEL_REPO. The HF model card carries no
# training-date field (and adding one upstream would only move the hand-maintained literal),
# so the date is recorded here, keyed on the one thing that cannot drift: the commit SHA.
#
# Seeded 2026-09-23 from `HfApi().list_repo_commits(MODEL_REPO)`: every revision that carries
# weights. All of them hold the ICCV 2025 paper weights, so all map to one date. That was
# checked, not assumed: model.safetensors has the same LFS sha256 (b4122c254fde...) on every
# revision from fa8e2dc through the v1.0-paper tag (1078bcd), and the 2026-07-24 re-export
# (606a119, main) serializes the same 380 tensors under a `model.` key prefix — compared
# value-for-value, all 380 identical. Revisions before 2ccf69f carry no modeling code and
# cannot be loaded, but are listed anyway: the table answers "what date are these weights",
# not "can you load it". The initial commit (bbf1ad4, .gitattributes only) has no weights.
#
# HOW TO EXTEND IT when a retrained checkpoint is pushed (RampNet#158): push to the hub, copy
# the new commit SHA (the refusal message prints it; so does `HfApi().list_repo_commits`),
# and add a row with the date the checkpoint was TRAINED, not pushed. A metadata-only commit
# (a README edit) keeps the weights, so it keeps the old date; say so in its note.
#
# WHY AN UNKNOWN SHA REFUSES: the alternative is a default, and a default date on a new
# model is precisely the stale-but-authoritative failure this table exists to end. PS rows
# are permanent and insert-only (SidewalkWebpage#5382), so a wrong date cannot be corrected
# afterwards. `main.py --allow-unknown-model-revision` runs anyway with the date recorded as
# null (and the override recorded in the manifest) — and send_to_ps.py refuses such a file,
# since PS requires the date.
_PAPER_WEIGHTS = "2025-08-21"
KNOWN_REVISIONS = {
    "606a11956743f7eb328d9207769034752f6191f4": {
        "training_date": _PAPER_WEIGHTS,
        "note": "HF main since 2026-07-24: transformers-compatible re-export (RampNet#19); "
                "paper weights, value-identical"},
    "1078bcd6771d63bd845d9fd36042904473e20b07": {
        "training_date": _PAPER_WEIGHTS, "note": "v1.0-paper tag (ICCV 2025 release)"},
    "3cdeabf8cd5571e64c22af65e05c3d12257b6643": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "fbe8e8dbca2995f13463bb2c5ae6cb1e8ca999d5": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "d3f93ddc25e9e9074a1c8dcb0049ab564001a390": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "e941a1f550c0ab0db017cb8d1bc6189350baef6b": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "2b652e5ad4db1b88959977e8d7b6b3c881195efd": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "ad39a5107e2d9121681176e90eb1f8fc82c9a850": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "4e5d7bc12cbf72cb80192a770b672aaf3c72a633": {
        "training_date": _PAPER_WEIGHTS, "note": "README edit; paper weights"},
    "58fa1b8b589179e683b83085f904c399b08e13f3": {
        "training_date": _PAPER_WEIGHTS, "note": "modeling.py fix; paper weights"},
    "2ccf69ff97a191aaa31082e100c94cfd712649fd": {
        "training_date": _PAPER_WEIGHTS, "note": "modeling.py added; paper weights"},
    "4c227f0c4aaf474de82cc8b0ce5717eb02158feb": {
        "training_date": _PAPER_WEIGHTS, "note": "config.json edit; no modeling code yet"},
    "fa8e2dc96da04df668e60131bf72dd2384dec294": {
        "training_date": _PAPER_WEIGHTS, "note": "first weights push; no modeling code yet"},
}

_SHA_RE = re.compile(r"[0-9a-f]{40}")


class ModelProvenanceError(RuntimeError):
    """The loaded model's provenance cannot be stated truthfully, so detection must not
    start: either no revision SHA could be resolved at all, or the SHA is not in
    KNOWN_REVISIONS. Records written anyway would become permanent, unverifiable PS rows."""


def is_revision(value) -> bool:
    """A full 40-hex git commit SHA — the only form KNOWN_REVISIONS is keyed on."""
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def revision_from_snapshot_path(path):
    """The SHA in a hub-cache path, `.../models--org--name/snapshots/<sha>[/file]`, or None.

    The fallback for when transformers did not set ``config._commit_hash``: the hub cache
    names each snapshot directory after the commit it holds, so the directory the weights
    were read from IS the revision. A model loaded from a plain local folder has no
    `snapshots/<sha>` component and so no resolvable revision.
    """
    if not path:
        return None
    parts = PurePath(str(path)).parts
    for i, part in enumerate(parts[:-1]):
        if part == "snapshots" and is_revision(parts[i + 1]):
            return parts[i + 1]
    return None


def model_id_for(revision: str) -> str:
    """`rampnet-model@<first 12 hex>`: the version-bearing model_id.

    The prefix is the id every existing consumer already matches on (PS's `label_ai_info`
    queries, the clustering-eval join), so `LIKE 'rampnet-model%'` still finds every row;
    the suffix makes two checkpoints distinguishable in the database forever. 12 hex is
    git's usual short-SHA length and far past collision range for one repo. PS stores
    model_id as unbounded TEXT (SidewalkWebpage evolution 286), so no width limit applies.
    """
    return f"{MODEL_REPO.split('/')[-1]}@{revision[:12]}"


def ps_training_date(iso_date: str) -> str:
    """A KNOWN_REVISIONS ISO date as the MM-DD-YYYY string Project Sidewalk parses.

    PS reads `model_training_date` with DateTimeFormatter.ofPattern("MM-dd-yyyy")
    (ExploreService.submitAiLabelData) into a NOT NULL timestamp column, so the wire format
    is the server's; the table stays ISO because that is the unambiguous one to edit.
    """
    return date.fromisoformat(iso_date).strftime("%m-%d-%Y")


def provenance_for_revision(revision, allow_unknown: bool = False) -> dict:
    """The provenance block written into every JSONL line and manifest run entry.

    Keys: model_repo, model_revision (40 hex), model_id, model_training_date (MM-DD-YYYY,
    PS's format) and api_version. Raises ModelProvenanceError for an unresolvable revision
    always, and for one missing from KNOWN_REVISIONS unless ``allow_unknown`` — in which
    case model_training_date is None, never a guess.
    """
    if not is_revision(revision):
        raise ModelProvenanceError(
            f"Could not resolve the Hugging Face revision of the loaded {MODEL_REPO} "
            f"(got {revision!r}). Detection refuses to run without it: every record would "
            f"carry provenance nobody could verify. Load the model through the hub cache.")
    known = KNOWN_REVISIONS.get(revision)
    if known is None and not allow_unknown:
        raise ModelProvenanceError(
            f"The loaded {MODEL_REPO} is revision {revision}, which is not in "
            f"detectors.KNOWN_REVISIONS, so its training date is unknown.\n"
            f"   If this is a new checkpoint, add a row for {revision} to KNOWN_REVISIONS in "
            f"detectors/__init__.py with the date it was trained (see the comment there).\n"
            f"   --allow-unknown-model-revision runs anyway with model_training_date null "
            f"(recorded in the manifest); send_to_ps.py will refuse that file.")
    return {
        "model_repo": MODEL_REPO,
        "model_revision": revision,
        "model_id": model_id_for(revision),
        "model_training_date": ps_training_date(known["training_date"]) if known else None,
        "api_version": API_VERSION,
    }


def provenance_from_snapshot_dir(snapshot_dir, commit_hash=None, allow_unknown: bool = False) -> dict:
    """Resolve a loaded model's provenance block.

    ``commit_hash`` is transformers' ``config._commit_hash`` and wins when it is a valid SHA;
    otherwise the revision is read off the `snapshots/<sha>` directory the weights came from
    (``snapshot_dir`` may also be a file inside it). See ``provenance_for_revision`` for the
    refusals.

    Example::

        >>> provenance_from_snapshot_dir(
        ...     "hub/models--projectsidewalk--rampnet-model/snapshots/"
        ...     "606a11956743f7eb328d9207769034752f6191f4/config.json")["model_id"]
        'rampnet-model@606a11956743'
    """
    revision = commit_hash if is_revision(commit_hash) else revision_from_snapshot_path(snapshot_dir)
    return provenance_for_revision(revision, allow_unknown=allow_unknown)


def load_with_offline_fallback(loader, repo_id: str = MODEL_REPO, **kwargs):
    """``loader(repo_id, **kwargs)``, retried once with ``local_files_only=True`` on OSError.

    Hyak compute nodes have no internet egress. transformers raises OSError when it cannot
    reach the hub, even though the cached snapshot would do, so the retry loads exactly what
    is in the local cache — and the provenance is then resolved from that snapshot like any
    other load, so an offline run is still attributed to the weights it really used.
    ``loader`` is ``AutoModel.from_pretrained`` in production and a stub in the torch-free
    tests.
    """
    try:
        return loader(repo_id, **kwargs)
    except OSError as online_error:
        if kwargs.get("local_files_only"):
            raise
        print(f"-> Could not reach the Hugging Face hub ({online_error}); "
              f"loading {repo_id} from the local cache.")
        return loader(repo_id, **dict(kwargs, local_files_only=True))
