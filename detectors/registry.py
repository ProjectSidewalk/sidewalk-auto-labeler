"""Which detectors exist, and how a results.jsonl record says which ran (issue #168).

The pipeline was written for one label type, CurbRamp. A second detector (a crosswalk or
NoCurbRamp model, Planning#36) needs its own thresholds -- a crosswalk heatmap's noise floor
and operating point are not RampNet's -- so each type's thresholds live in its
``DetectorSpec`` rather than in module constants. The CurbRamp spec REFERENCES the constants
in ``detectors/__init__.py``; they keep their names and values, so every existing import
keeps working and a CurbRamp-only run behaves exactly as before. The second spec is
Crosswalk (RampNet#274's crosswalk_s4, loaded from a local checkpoint: detectors/crosswalk.py).

Readers never look at a record's type keys directly. They go through the three helpers
below, which also read every line written before #168 (no ``label_types``, no per-detection
``label_type``, no ``models``) as the CurbRamp-only line it is:

- ``record_types(record)``: the detectors that RAN on the pano -- so a pano with zero
  detections still says what it was checked for.
- ``detection_type(record, detection)``: one detection's type.
- ``provenance_for(record, label_type)``: the model that produced that type's detections.

The writer rule (main.build_output_line): a run whose type set is exactly {CurbRamp} writes
the pre-#168 format with no new key; any other set writes per-detection ``label_type``,
``label_types``, ``models`` and a line-level ``label_type`` that is null unless one type ran.

This module imports no torch (the test suite runs torch-free); ``load_detectors`` imports a
detector's module only when that type is asked for.

Example::

    >>> rec = {'detections': [{'x_normalized': .5, 'y_normalized': .6, 'confidence': .9}],
    ...        'label_type': 'CurbRamp', 'model_id': 'rampnet-model@abc'}
    >>> record_types(rec), detection_type(rec, rec['detections'][0])
    (['CurbRamp'], 'CurbRamp')
    >>> spec_for('CurbRamp').operational_confidence
    0.3
"""
import importlib
from collections import namedtuple
from types import MappingProxyType

from detectors import (BENCHMARK_CONFIDENCE, BORDER_EXCLUDE, BORDER_KEEP, DECODE_ARGMAX,
                       DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE)

CURB_RAMP = 'CurbRamp'
CROSSWALK = 'Crosswalk'

# The server's LabelType enum (SidewalkWebpage app/models/label/LabelType.scala, read
# 2026-11-04), in its canonical order. A DetectorSpec's label_type is sent verbatim as the
# submission's labelType, so it must be one of these; tests pin every SPECS key to it.
PS_LABEL_TYPES = ('CurbRamp', 'NoCurbRamp', 'Obstacle', 'SurfaceProblem', 'Crosswalk',
                  'Signal', 'NoSidewalk', 'Occlusion', 'Other')

# The keys a record carries per model (main.build_output_line writes them at line level for
# a CurbRamp-only run, and inside models[T] for every type otherwise).
PROVENANCE_KEYS = ('model_id', 'model_training_date', 'api_version', 'model_repo',
                   'model_revision')

# The keys that say how a type's peaks were found (its "frame"). For CurbRamp they are the
# line-level / manifest top-level keys and per-run choices (--decode, --border); for every
# other type they ride in models[T] and are fixed by its spec (see type_frame).
FRAME_KEYS = ('detection_decode', 'detection_border', 'detection_storage_floor')

DetectorSpec = namedtuple('DetectorSpec', [
    'label_type',              # PS label type string, e.g. 'CurbRamp'
    'module',                  # import path of the detector class
    'cls',                     # class name; constructed with load_detectors' **kw
    'storage_floor',           # results.jsonl keeps peaks down to here
    'operational_confidence',  # everything that ACTS on a detection filters here
    'benchmark_confidence',    # the tier this type's judged bundles were exported at
    'mask_rig',                # whether detectors.on_camera_rig applies to this type
    'border',                  # peak border rule (detectors.BORDERS): fixed for every type
                               # but CurbRamp, whose --border flag chooses it per run
    'min_depression_deg',      # range gate (detectors.beyond_range_gate, #192): send only
                               # detections at least this many degrees below the image
                               # horizon; None = ungated
    'fuse_overrides',          # FuseParams field -> value for this type's fusion (#168
                               # item 4; fuse_sites.fuse_params_for). A crosswalk is a region
                               # centred mid-road and a signal a pole, so association radii
                               # tuned for a point on the curb line need not fit them.
                               # Read-only; empty for CurbRamp, so its fusion is unchanged.
], defaults=(BORDER_EXCLUDE, None, MappingProxyType({})))

SPECS = {
    CURB_RAMP: DetectorSpec(CURB_RAMP, 'detectors.curb_ramp', 'CurbRampDetector',
                            DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE,
                            BENCHMARK_CONFIDENCE, True),
    # crosswalk_s4 (RampNet#274; detectors/checkpoints.py): the curb-ramp network retrained on
    # crosswalk centres. Its thresholds are RampNet's s4 recipe: peaks stored down to 0.05
    # (the frozen extractor's floor), the 0.23 operating point Jon's 440 s4 cards were drawn
    # at, and exclude_border=False (`keep`), the rule that threshold was chosen under.
    # benchmark_confidence = 0.23 binds no judged bundle yet: no labeler-format Crosswalk
    # bundle exists, so nothing joins a run to verdicts at this tier today.
    # The 49-degree rig mask removed 7 of 72 rated false alarms and 0 of 359 crosswalks.
    # Range gate 5.625 deg (#192, Jon 2026-11-17): PS's placement is bounded near 23.85 m, so
    # shallower detections land far short of their target. The cut sits on a 2.8125-deg cell
    # edge (row 272/512). Juneau: 4,400 -> 3,525 labels downtown, 60 -> 54 in the expansion.
    # CurbRamp stays ungated (None) for now.
    CROSSWALK: DetectorSpec(CROSSWALK, 'detectors.crosswalk', 'CrosswalkDetector',
                            0.05, 0.23, 0.23, True, BORDER_KEEP, 5.625),
}


class UnknownLabelTypeError(ValueError):
    """A label type with no DetectorSpec."""


class UntypedDetectionError(ValueError):
    """A detection with no label_type on a line that ran more than one detector."""


def spec_for(label_type):
    """The DetectorSpec of ``label_type``; raises UnknownLabelTypeError listing SPECS."""
    try:
        return SPECS[label_type]
    except KeyError:
        raise UnknownLabelTypeError(
            f'unknown label type {label_type!r}; known: {", ".join(sorted(SPECS))}') from None


def parse_types(value):
    """A comma list of label types -> the sorted, de-duplicated list; raises
    UnknownLabelTypeError on an unknown or empty entry. ``'CurbRamp'`` -> ``['CurbRamp']``."""
    names = [v.strip() for v in str(value).split(',')]
    if not names or any(not n for n in names):
        raise UnknownLabelTypeError(
            f'empty label type in {value!r}; known: {", ".join(sorted(SPECS))}')
    for n in names:
        spec_for(n)
    return sorted(set(names))


def is_legacy_type_set(label_types):
    """True when the type set is exactly {CurbRamp}: the run writes the pre-#168 format."""
    return sorted(set(label_types)) == [CURB_RAMP]


def type_frame(label_type):
    """The frame (FRAME_KEYS) a non-CurbRamp type's peaks are found under, fixed by its spec:
    argmax placement, the spec's border rule and storage floor. CurbRamp has no fixed frame
    (its decode and border are per-run flags), so asking for it raises ValueError.

        >>> type_frame('Crosswalk')
        {'detection_decode': 'argmax', 'detection_border': 'keep', 'detection_storage_floor': 0.05}
    """
    if label_type == CURB_RAMP:
        raise ValueError("CurbRamp's frame is chosen per run (--decode, --border), not by its spec")
    spec = spec_for(label_type)
    return {'detection_decode': DECODE_ARGMAX, 'detection_border': spec.border,
            'detection_storage_floor': spec.storage_floor}


def load_detectors(label_types, type_kwargs=None, **kw):
    """``{type: detector}`` for each requested type, importing each spec's module lazily.

    ``**kw`` goes to every detector's constructor (allow_unknown_revision, batch_size,
    batch_wait_s); ``type_kwargs[T]`` goes to type T's only (CurbRamp's decode / border,
    Crosswalk's checkpoint). A ``type_kwargs`` key that is not a requested type raises
    ValueError rather than being dropped. Each detector exposes ``.detect(image)``,
    ``.provenance``, ``.stats()`` and ``.close()``.
    """
    types = parse_types(','.join(label_types))
    type_kwargs = dict(type_kwargs or {})
    stray = sorted(set(type_kwargs) - set(types))
    if stray:
        raise ValueError(f'type_kwargs for {stray}, which are not among the requested types {types}')
    if len(types) > 1 and 'inference_lock' not in kw:
        # Each detector serializes its own forward passes; two detectors on 50 fetch threads
        # would still overlap on the device. One shared lock serializes them all. A single
        # detector keeps its own lock, so a CurbRamp-only run is constructed exactly as before.
        import threading
        kw = {**kw, 'inference_lock': threading.Lock()}
    out = {}
    for t in types:
        spec = spec_for(t)
        cls = getattr(importlib.import_module(spec.module), spec.cls)
        out[t] = cls(**kw, **type_kwargs.get(t, {}))
    return out


def record_types(record):
    """The label types whose detectors ran on this record's pano, as stored (main.py writes
    them sorted).

    ``record['label_types']`` when present (a #168 multi-type line), else the line's
    ``label_type`` (every pre-#168 line says 'CurbRamp'), else ['CurbRamp'].
    """
    if 'label_types' in record:
        return list(record['label_types'])
    return [record.get('label_type') or CURB_RAMP]


def detection_type(record, detection):
    """One detection's label type: its own ``label_type``, else the line's, else 'CurbRamp'.

    On a line that ran more than one detector an untyped detection is a malformed record,
    not a CurbRamp one, so it RAISES UntypedDetectionError instead of falling through.
    """
    t = detection.get('label_type')
    if t:
        return t
    if len(record_types(record)) > 1:
        raise UntypedDetectionError(
            f'detection without label_type on a line that ran {record_types(record)}: '
            f'{detection!r}')
    return record.get('label_type') or CURB_RAMP


def detections_of_type(record, label_type):
    """The record's detections whose detection_type is ``label_type``, in stored order."""
    return [d for d in record.get('detections', ())
            if detection_type(record, d) == label_type]


def provenance_for(record, label_type):
    """The model provenance of ``label_type``'s detections on this record.

    ``record['models'][label_type]`` when the line has a models map, else the line-level
    model_* keys (a pre-#168 or CurbRamp-only line); only PROVENANCE_KEYS, never a type's
    decode/border markers. A multi-type line whose models map lacks the type raises KeyError
    rather than reporting another type's model.
    """
    src = record['models'][label_type] if 'models' in record else record
    return {k: src.get(k) for k in PROVENANCE_KEYS}


def type_markers(record, label_type):
    """(detection_decode, detection_border) that ``label_type``'s detections on this record were
    found under. The line-level markers describe the CurbRamp detector (absent = argmax /
    exclude); any other type's live in its ``models[T]`` entry (main._multitype_line)."""
    from detectors import record_border, record_decode
    if label_type == CURB_RAMP or 'models' not in record:
        return record_decode(record), record_border(record)
    m = record['models'][label_type]
    return (m.get('detection_decode', DECODE_ARGMAX),
            m.get('detection_border', BORDER_EXCLUDE))


def require_curb_ramp_only(record, what):
    """Raise ValueError unless ``record`` ran exactly CurbRamp. For the analysis scripts that
    have not been taught to filter by type (#168): refuse a multi-type file loudly instead of
    scoring another type's detections as curb ramps."""
    types = record_types(record)
    if not is_legacy_type_set(types):
        raise ValueError(
            f'{what}: record ran label types {types}; this tool scores CurbRamp-only '
            f'results (#168: not yet taught to filter by type)')


def require_curb_ramp_file(path, what):
    """require_curb_ramp_only over every record of a results file (one json parse per line).
    Returns None; raises ValueError naming the first record that ran another type set."""
    import json
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                require_curb_ramp_only(json.loads(line), f'{what} ({path}, line {n})')
