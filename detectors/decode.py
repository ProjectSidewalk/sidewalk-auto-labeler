"""Heatmap -> detections: the peak decode (issue #111), torch-free.

RampNet's 512x1024 heatmap is an exact bilinear 8x upsample of a 64x128 "coarse" map (the
head is conv3x3 -> ReLU -> Upsample(x8, bilinear) -> conv1x1, and the 1x1 conv commutes
with the upsample). A bilinear surface peaks only at its coarse sample points, so an
integer argmax lands on hi-res pixel 3 or 4 mod 8 and every stored position is quantized to
one coarse cell (8 heatmap px, 2.8 deg). docs/heatmap-grid.md measures that on every run.

Two decodes, selected per run (``--decode`` on main.py and reinfer.py, bound in
manifest.json; detect_from_store.py stays argmax, because a store run exists to reproduce the
live argmax labels):

- ``argmax`` (the default, and what every live label and published table used): the pixel
  ``peak_local_max`` returns. Bit-identical to the extractor this repo has always run
  (pinned in tests/test_decode.py against the pre-#111 code on stored heatmaps).
- ``gaussian``: RampNet's sub-cell rule (RampNet#221). The SAME peaks with the SAME scores,
  in the same order; only (x, y) move, by the log-parabola vertex of each peak's 3x3 coarse
  neighbourhood. That is at most half a coarse cell from the coarse centre, but a peak whose
  argmax a diagonal neighbour pulled into the next cell is re-anchored first, so the move
  from the argmax pixel is not bounded by half a cell: over the 14,817 committed peaks
  (docs/figures/heatmap-grid/data/decode/) the median is 1.5 heatmap px per axis, the p99 3.5
  px, and 3 peaks moved more than 4.5 px (max 8.7 px). On RampNet's manual_gold it lowers the
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

# The decode names, the record marker and the file-level checks are stdlib-only and live in
# detectors/__init__.py beside the rest of the detection contract; re-exported here.
from detectors import (DECODE_ARGMAX as ARGMAX, DECODE_GAUSSIAN as GAUSSIAN,  # noqa: F401
                       DECODES, DEFAULT_DECODE, DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO,
                       RECORD_DECODE_KEY, record_decode,
                       BORDER_EXCLUDE as EXCLUDE, BORDER_KEEP as KEEP,
                       BORDER_WRAP as WRAP, BORDERS,
                       DEFAULT_BORDER, PEAK_MIN_DISTANCE, RECORD_BORDER_KEY,
                       border_band_edge, record_border)
from detectors import rampnet_subcell as sc

#: Provenance of the vendored ``detectors/rampnet_subcell.py``: RampNet's ``rampnet/subcell.py``
#: at this commit (the last one to touch it; identical at RampNet main 459ea9e, 2026-10-02),
#: byte for byte. tests/test_decode.py checks the hash; to update, copy the file verbatim
#: from RampNet and change both constants together.
RAMPNET_SUBCELL_COMMIT = 'cf7aecdbce53f72028663ab21a73fc0354255865'
RAMPNET_SUBCELL_SHA256 = 'b0712dfe98fc6012ddfd22f149dc1ece7917d74b21a10276ff83dec0e9a2ad8e'

# The peak finder every decode shares (unchanged since #27): peaks on clip(h, 0, 1) down to
# the storage floor, at most MAX_PEAKS_PER_PANO, at least MIN_DISTANCE apart. Under the
# default border rule (`exclude`, skimage's default exclude_border=True) every peak within
# MIN_DISTANCE of an edge is also dropped -- including along the 360-degree seam, a 20-column
# band that on the exact x8 upsample holds coarse columns 0 and 127 (5.6 degrees), blind on
# every pano. RampNet#132 calls that a defect; for the labeler it is
# #130, and `--border keep` (RampNet's rule: exclude_border=False, no NMS across the seam)
# is the opt-in fix; `--border wrap` (this repo's rule) adds NMS across the seam, by running
# the same finder on the heatmap as a cylinder (_cylinder_peaks). The default stays
# `exclude`, because changing it adds stored detections that the live labels do not have
# (docs/seam-band-130.md measures how many; section 9 estimates `wrap`).
MIN_DISTANCE = PEAK_MIN_DISTANCE    # 10; detectors.border_band_edge measures the band with it


def _peaks(heatmap, border=DEFAULT_BORDER):
    """(row, col) peaks of clip(heatmap, 0, 1), highest first, under a border rule.

    ``exclude``: skimage's default, peaks within MIN_DISTANCE of any edge dropped (every live
    label). ``keep``: ``exclude_border=False``, exactly RampNet's detect_peaks since RampNet#132
    -- edge peaks kept, NMS NOT wrapped across the seam (a seam-straddling ramp can give one
    peak in column 0-9 and another in column 1014-1023, as it does in RampNet). Two
    differences from RampNet predate #130 and hold under both rules: this finder keeps at
    most MAX_PEAKS_PER_PANO (50) peaks (RampNet has no cap; no Laurens pano reached it), and
    detections_from_heatmap scores a peak by the RAW heatmap value where RampNet's
    detect_peaks(clip=True) reports the clipped one (they differ only above 1.0).

    ``wrap`` (this repo's rule, #130 follow-up): the same finder on the heatmap as a
    cylinder, so a ramp straddling the 360-degree seam gives one peak, not two. It is
    implemented directly, with skimage's semantics everywhere except across the seam
    (``_cylinder_peaks``): a candidate is a pixel of clip(h, 0, 1) above the storage floor that
    equals the maximum of its (2d+1)-square window, where the window wraps in x and is clamped
    in y (``mode=('nearest', 'wrap')``; zenith and nadir are not neighbours); candidates are
    taken in descending clipped value, ties in row-major order (skimage's stable sort), and
    one is rejected when an accepted peak lies at wrapped Chebyshev distance < d (skimage's
    ensure_spacing keeps a point at exactly d); the cap keeps the first 50.
    tests/test_border_wrap.py checks it against an independent brute-force cylinder on random
    maps with clipped plateaus and multi-way ties.

    Relation to ``keep``. A pixel whose window does not reach the seam (columns d .. W-d-1)
    has the same window under both rules, and two such pixels the same distance. Spacing only
    ever separates EQUAL candidates (two candidates within d are in each other's window), so
    when no exact ties are involved -- in practice, no plateau clipped at 1.0 -- the peaks off
    the band are ``keep``'s, ``wrap`` only removes seam-band peaks, and as sets exclude <=
    wrap <= keep (while the cap does not bind). Exact ties can chain through the spacing
    pass, so on clipped maps both relations can fail (exclude <= keep fails there too, from
    the band edge); that is the cylinder's answer, not an artefact.

    Ties. Of a strictly unequal straddling pair the weaker half is not a local maximum at all.
    Of an exactly tied pair (in practice: both halves clipped to 1.0), the survivor is the one
    first in row-major order -- the earlier row; on the same row, the left-edge half (column
    0-9). Because the order is on CLIPPED values, the survivor of a clipped pair can be the
    half with the lower raw score (what detections_from_heatmap reports).

        >>> from detectors import rampnet_subcell as sc
        >>> yy, xx = np.mgrid[0:64, 0:128]
        >>> g = lambda cx, a: a * np.exp(-((yy - 25) ** 2 + (xx - cx) ** 2) / 3.125)
        >>> h = sc.upsample(g(0, 0.8) + g(127, 0.75))       # one ramp across the seam
        >>> len(_peaks(h, 'keep')), [int(c) for _, c in _peaks(h, 'wrap')]
        (2, [0])
    """
    if border not in BORDERS:
        raise ValueError(f'unknown border rule {border!r}; known: {", ".join(BORDERS)}')
    if border == WRAP:
        return _cylinder_peaks(np.clip(heatmap, 0, 1))
    from skimage.feature import peak_local_max
    return peak_local_max(np.clip(heatmap, 0, 1), min_distance=MIN_DISTANCE,
                          threshold_abs=DETECTION_STORAGE_FLOOR,
                          num_peaks=MAX_PEAKS_PER_PANO,
                          exclude_border=(border == EXCLUDE))


def _cylinder_peaks(h, d=MIN_DISTANCE, floor=DETECTION_STORAGE_FLOOR, cap=MAX_PEAKS_PER_PANO):
    """``peak_local_max(h, min_distance=d, threshold_abs=floor, num_peaks=cap,
    exclude_border=False)`` with the x axis cyclic (the ``wrap`` rule; ``_peaks``).

    Step for step what skimage 0.26 does, with the two seam-blind operations replaced:
    ``_get_peak_mask`` (maximum filter, mode 'nearest' -> ('nearest', 'wrap'); a constant
    image has no peak; then ``> floor``), ``_get_high_intensity_peaks`` (row-major
    candidates, stable descending sort) and ``ensure_spacing`` (greedy, reject at Chebyshev
    distance < d -> wrapped Chebyshev distance < d). Returns an (n, 2) int array of
    (row, col), highest first.

        >>> h = np.zeros((40, 100)); h[20, 99] = 0.6; h[20, 3] = 0.5; h[5, 50] = 0.4
        >>> _cylinder_peaks(h).tolist()        # (20, 3) is 4 columns from (20, 99) on the cylinder
        [[20, 99], [5, 50]]
    """
    from scipy import ndimage as ndi
    size = 2 * d + 1
    is_max = h == ndi.maximum_filter(h, size=size, mode=('nearest', 'wrap'))
    if np.all(is_max):                       # skimage: no peak for a trivial image
        return np.zeros((0, 2), dtype=np.intp)
    is_max &= h > floor
    coord = np.argwhere(is_max)              # row-major, as np.nonzero
    coord = coord[np.argsort(-h[is_max], kind='stable')]
    width = h.shape[1]
    kept = []
    for r, c in coord:
        if kept:
            k = np.asarray(kept)
            dx = np.abs(k[:, 1] - c)
            dx = np.minimum(dx, width - dx)
            if np.any(np.maximum(np.abs(k[:, 0] - r), dx) < d):
                continue
        kept.append((r, c))
        if len(kept) >= cap:
            break
    return np.asarray(kept, dtype=np.intp).reshape(-1, 2)


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


def detections_from_heatmap(heatmap, decode=DEFAULT_DECODE, coarse=None, border=DEFAULT_BORDER):
    """Heatmap peaks -> normalized ``[(x, y, confidence), ...]``, highest first.

    Peaks are stored down to the storage floor; the operational threshold is applied by
    consumers, not here (see detectors/__init__.py). num_peaks keeps the highest-intensity
    peaks, so the >= OPERATIONAL_CONFIDENCE set is unaffected by the lower floor. This is
    per image whether or not the forward pass was batched.

    ``decode`` picks where each peak is placed (module docstring); the peaks, their order and
    their confidences (the raw heatmap value at the argmax pixel) are the same under every
    decode. ``coarse`` may pass the 64x128 map when the caller already has it.

    ``border`` picks which peaks are found at all (#130; ``_peaks``): ``exclude`` (default,
    every live label) drops peaks within MIN_DISTANCE of an edge, ``keep`` returns them, and
    ``wrap`` returns them with NMS across the seam. Under ``keep`` an edge peak's gaussian
    decode is RampNet's own: the 3x3 coarse neighbourhood is NaN off the map and that axis
    gets no sub-cell offset (rampnet_subcell._axis), so a peak in coarse column 0 stays at
    the column-0 centre in x (pinned in tests/test_border.py). Under ``wrap`` the coarse
    neighbourhood is cyclic in x too (``refine_peaks(wrap_x=True)``), so a seam peak gets a
    sub-cell offset and may land either side of the seam; refine_peaks reduces x modulo the
    width, so it stays in [0, 1). That is a choice inside the double opt-in (wrap +
    gaussian), reversible by passing wrap_x=False. It also reaches a few peaks just off the
    band: a peak in fine columns 10-15 / 1008-1013 (coarse column 1 / 126) whose coarse climb
    steps into column 0 / 127 decodes with the cyclic neighbourhood there, so its gaussian x
    can differ from ``keep``'s (the peak, its argmax pixel and its score do not).

    Collisions (gaussian only). Several peaks that peak_local_max returns on one clipped
    plateau (a flat top above 1, at least MIN_DISTANCE apart) all climb to the same coarse
    maximum and would decode to the IDENTICAL position: duplicate labels on the server. The
    first (highest-scoring) keeps its gaussian position; every later one that lands exactly on
    an earlier position keeps its argmax pixel instead, so positions stay distinct and no
    peak is dropped. Measured rate: 0 of the 14,817 committed peaks (the highest confidence
    there is 1.04); the case is pinned on a synthetic map in tests/test_decode.py.
    """
    if decode not in DECODES:
        raise ValueError(f'unknown decode {decode!r}; known: {", ".join(DECODES)}')
    peaks = _peaks(heatmap, border)
    if decode == ARGMAX or not len(peaks):
        return [(float(c / heatmap.shape[1]), float(r / heatmap.shape[0]), float(heatmap[r][c]))
                for r, c in peaks]
    coarse = check_exact_upsample(heatmap, coarse)
    xy = sc.refine_peaks(heatmap, peaks, method=decode, coarse=coarse,
                         wrap_x=(border == WRAP))
    out, seen = [], set()
    for (x, y), (r, c) in zip(xy, peaks):
        x, y = float(x), float(y)
        if (x, y) in seen:      # a collision (docstring): fall back to this peak's argmax
            x, y = float(c / heatmap.shape[1]), float(r / heatmap.shape[0])
        seen.add((x, y))
        out.append((x, y, float(heatmap[r][c])))
    return out


def detections_both(heatmap, border=DEFAULT_BORDER):
    """``{'argmax': [...], 'gaussian': [...]}`` from ONE heatmap: the paired comparison #111
    needs (one forward pass, two decodes -- never two inference runs, whose near-tied cells
    can flip between passes). Element k of both lists is the same peak."""
    return {ARGMAX: detections_from_heatmap(heatmap, ARGMAX, border=border),
            GAUSSIAN: detections_from_heatmap(heatmap, GAUSSIAN, border=border)}
