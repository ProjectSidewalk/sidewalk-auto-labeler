"""Heatmap -> detections: the peak decode (issue #111), torch-free.

RampNet's 512x1024 heatmap is an exact bilinear 8x upsample of a 64x128 "coarse" map (the
head is conv3x3 -> ReLU -> Upsample(x8, bilinear) -> conv1x1, and the 1x1 conv commutes
with the upsample). A bilinear surface peaks only at its coarse sample points, so an
integer argmax lands on hi-res pixel 3 or 4 mod 8 and every stored position is quantized to
one coarse cell (8 heatmap px, 2.8 deg). docs/heatmap-grid.md measures that on every run.

Two decodes, selected per run (``--decode`` on main.py / reinfer.py / detect_from_store.py,
bound in manifest.json):

- ``argmax`` (the default, and what every live label and published table used): the pixel
  ``peak_local_max`` returns. Bit-identical to the extractor this repo has always run
  (pinned in tests/test_decode.py against the pre-#111 code on stored heatmaps).
- ``gaussian``: RampNet's sub-cell rule (RampNet#221). The SAME peaks with the SAME scores,
  in the same order; only (x, y) move, by the log-parabola vertex of each peak's 3x3 coarse
  neighbourhood (at most half a coarse cell per axis, after re-anchoring a peak whose argmax
  a diagonal neighbour pulled into the next cell). On RampNet's manual_gold it lowers the
  mean distance to independent box centres from 5.08 to 4.35 heatmap px.

The gaussian rule is RampNet's own code, carried verbatim in ``detectors/rampnet_subcell.py``
(RampNet ``rampnet/subcell.py`` at RAMPNET_SUBCELL_COMMIT; tests/test_decode.py hashes it),
the way RampNet's Hub exporter ships it as ``rampnet_subcell.py``. It is vendored rather
than taken from the Hub because the published model package does not yet include it.

The coarse map is recovered from the heatmap by least squares against the bilinear operator,
which is exact only for the RAW, unclipped, single-pass head output (what
``CurbRampDetector.heatmap`` returns). A heatmap that is not an exact x8 upsample (clipped,
TTA-combined, or from another input size) is refused with ValueError rather than decoded
from a least-squares guess -- RampNet's guard (``upsample_residual`` > ``UPSAMPLE_RTOL``).

Example::

    >>> import numpy as np
    >>> from detectors import rampnet_subcell as sc
    >>> yy, xx = np.mgrid[0:64, 0:128]
    >>> coarse = 0.9 * np.exp(-((yy - 30.3) ** 2 + (xx - 60.8) ** 2) / (2 * 1.25 ** 2))
    >>> h = sc.upsample(coarse)                       # what the head emits
    >>> [round(v, 4) for v in detections_from_heatmap(h)[0]]   # col 491 = 8*61+3: on the grid
    [0.4795, 0.4766, 0.8475]
    >>> [round(v, 4) for v in detections_from_heatmap(h, 'gaussian')[0]]   # (8*60.8+3.5)/1024
    [0.4784, 0.4803, 0.8475]
"""
import numpy as np

from detectors import DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO
from detectors import rampnet_subcell as sc

ARGMAX = 'argmax'
GAUSSIAN = 'gaussian'
#: The decodes a run may use. ``argmax`` is the default everywhere (#111: the default does
#: not change in the PR that adds ``gaussian``; that rollout is a per-city decision).
DECODES = (ARGMAX, GAUSSIAN)
DEFAULT_DECODE = ARGMAX
#: The JSONL record key that marks a non-default decode. Written ONLY on lines whose decode
#: is not argmax, so argmax runs stay byte-identical and every line written before #111
#: (no key) reads as argmax. See record_decode().
RECORD_DECODE_KEY = 'detection_decode'

#: Provenance of the vendored ``detectors/rampnet_subcell.py``: RampNet's ``rampnet/subcell.py``
#: at this commit (the last one to touch it; identical at RampNet main 459ea9e, 2026-10-02),
#: byte for byte. tests/test_decode.py checks the hash; to update, copy the file verbatim
#: from RampNet and change both constants together.
RAMPNET_SUBCELL_COMMIT = 'cf7aecdbce53f72028663ab21a73fc0354255865'
RAMPNET_SUBCELL_SHA256 = 'b0712dfe98fc6012ddfd22f149dc1ece7917d74b21a10276ff83dec0e9a2ad8e'

# The peak finder every decode shares (unchanged since #27): peaks on clip(h, 0, 1) down to
# the storage floor, at most MAX_PEAKS_PER_PANO, skimage's default exclude_border (peaks
# within min_distance of the edge are dropped -- RampNet#132 calls that a defect; the
# labeler keeps it, because changing it moves stored detections).
MIN_DISTANCE = 10


def _peaks(heatmap):
    from skimage.feature import peak_local_max
    return peak_local_max(np.clip(heatmap, 0, 1), min_distance=MIN_DISTANCE,
                          threshold_abs=DETECTION_STORAGE_FLOOR,
                          num_peaks=MAX_PEAKS_PER_PANO)


def check_exact_upsample(heatmap, coarse=None):
    """Raise ValueError unless ``heatmap`` is an exact x8 bilinear upsample (RampNet's
    UPSAMPLE_RTOL guard); returns the recovered coarse map."""
    if coarse is None:
        coarse = sc.coarse_from_heatmap(heatmap)
    res = sc.upsample_residual(heatmap, coarse)
    if res > sc.UPSAMPLE_RTOL:
        raise ValueError(
            f'heatmap {np.shape(heatmap)} is not an exact x{sc.FACTOR} bilinear upsample '
            f'(relative residual {res:.2e} > {sc.UPSAMPLE_RTOL:g}); the gaussian decode needs '
            'the raw, unclipped, single-pass head output at the model input size')
    return coarse


def detections_from_heatmap(heatmap, decode=DEFAULT_DECODE, coarse=None):
    """Heatmap peaks -> normalized ``[(x, y, confidence), ...]``, highest first.

    Peaks are stored down to the storage floor; the operational threshold is applied by
    consumers, not here (see detectors/__init__.py). num_peaks keeps the highest-intensity
    peaks, so the >= OPERATIONAL_CONFIDENCE set is unaffected by the lower floor. This is
    per image whether or not the forward pass was batched.

    ``decode`` picks where each peak is placed (module docstring); the peaks, their order and
    their confidences (the raw heatmap value at the argmax pixel) are the same under every
    decode. ``coarse`` may pass the 64x128 map when the caller already has it.
    """
    if decode not in DECODES:
        raise ValueError(f'unknown decode {decode!r}; known: {", ".join(DECODES)}')
    peaks = _peaks(heatmap)
    if decode == ARGMAX or not len(peaks):
        return [(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c]))
                for r, c in peaks]
    coarse = check_exact_upsample(heatmap, coarse)
    xy = sc.refine_peaks(heatmap, peaks, method=decode, coarse=coarse)
    return [(float(x), float(y), float(heatmap[r][c]))
            for (x, y), (r, c) in zip(xy, peaks)]


def detections_both(heatmap):
    """``{'argmax': [...], 'gaussian': [...]}`` from ONE heatmap: the paired comparison #111
    needs (one forward pass, two decodes -- never two inference runs, whose near-tied cells
    can flip between passes). Element k of both lists is the same peak."""
    return {ARGMAX: detections_from_heatmap(heatmap, ARGMAX),
            GAUSSIAN: detections_from_heatmap(heatmap, GAUSSIAN)}


def record_decode(record):
    """The decode a results.jsonl record was written under: its RECORD_DECODE_KEY, or
    ``argmax`` when absent (every line before #111, and every argmax line since).

        >>> record_decode({'detections': []}), record_decode({'detection_decode': 'gaussian'})
        ('argmax', 'gaussian')
    """
    return record.get(RECORD_DECODE_KEY, ARGMAX)
