"""Which detectors exist, and how a results.jsonl record says which ran (issue #168).

The pipeline was written for one label type, CurbRamp. A second detector (a crosswalk or
NoCurbRamp model, Planning#36) needs its own thresholds -- a crosswalk heatmap's noise floor
and operating point are not RampNet's -- so each type's thresholds live in its
``DetectorSpec`` rather than in module constants. The CurbRamp spec REFERENCES the constants
in ``detectors/__init__.py``; they keep their names and values, so every existing import
keeps working and a CurbRamp-only run behaves exactly as before.

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

from detectors import (BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR,
                       OPERATIONAL_CONFIDENCE)

CURB_RAMP = 'CurbRamp'

# The keys a record carries per model (main.build_output_line writes them at line level for
# a CurbRamp-only run, and inside models[T] for every type otherwise).
PROVENANCE_KEYS = ('model_id', 'model_training_date', 'api_version', 'model_repo',
                   'model_revision')

DetectorSpec = namedtuple('DetectorSpec', [
    'label_type',              # PS label type string, e.g. 'CurbRamp'
    'module',                  # import path of the detector class
    'cls',                     # class name; constructed with load_detectors' **kw
    'storage_floor',           # results.jsonl keeps peaks down to here
    'operational_confidence',  # everything that ACTS on a detection filters here
    'benchmark_confidence',    # the tier this type's judged bundles were exported at
    'mask_rig',                # whether detectors.on_camera_rig applies to this type
])

SPECS = {
    CURB_RAMP: DetectorSpec(CURB_RAMP, 'detectors.curb_ramp', 'CurbRampDetector',
                            DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE,
                            BENCHMARK_CONFIDENCE, True),
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


def load_detectors(label_types, **kw):
    """``{type: detector}`` for each requested type, importing each spec's module lazily.

    ``**kw`` goes to every detector's constructor (today: CurbRampDetector's
    allow_unknown_revision / batch_size / batch_wait_s / decode / border). Each detector
    exposes ``.detect(image)``, ``.provenance``, ``.stats()`` and ``.close()``.
    """
    out = {}
    for t in parse_types(','.join(label_types)):
        spec = spec_for(t)
        cls = getattr(importlib.import_module(spec.module), spec.cls)
        out[t] = cls(**kw)
    return out


def record_types(record):
    """The label types whose detectors ran on this record's pano, sorted.

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
    model_* keys (a pre-#168 or CurbRamp-only line). A multi-type line whose models map lacks
    the type raises KeyError rather than reporting another type's model.
    """
    if 'models' in record:
        return dict(record['models'][label_type])
    return {k: record.get(k) for k in PROVENANCE_KEYS}


def require_curb_ramp_only(record, what):
    """Raise ValueError unless ``record`` ran exactly CurbRamp. For the analysis scripts that
    have not been taught to filter by type (#168): refuse a multi-type file loudly instead of
    scoring another type's detections as curb ramps."""
    types = record_types(record)
    if not is_legacy_type_set(types):
        raise ValueError(
            f'{what}: record ran label types {types}; this tool scores CurbRamp-only '
            f'results (#168: not yet taught to filter by type)')
