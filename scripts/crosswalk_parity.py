#!/usr/bin/env python3
"""Parity check: does the labeler's CrosswalkDetector reproduce RampNet's stored crosswalk_s4
predictions? The acceptance test for the local-checkpoint loader (GPU; not in CI).

RampNet's s4 recipe (scripts/analysis/stage1_crosswalk_sdot_274/score_274.py `infer`) ran
the same network on pre-encoded 4096x2048 JPEGs: PIL bilinear resize to 2048x4096, ImageNet
normalisation, fp32, ``peak_local_max(clip(h, 0, 1), min_distance=10, threshold_abs=0.05,
exclude_border=False)`` with no peak cap, confidence = the raw heatmap value at the peak,
rounded to 6 decimals. This script runs ``CrosswalkDetector.detect`` on those exact JPEGs
and compares peak for peak.

The pass rule, fixed before the first run (plan crosswalk-labeler-spec-2026-11-15, item 8):

- Every stored peak (>= 0.05) has a labeler peak at the same heatmap pixel with
  |confidence difference| <= TOL (5e-3), and vice versa.
- A near-tie swap or a one-pixel move counts as a match only when the two values are within
  TOL: a peak left unmatched at its own pixel may pair with an unmatched peak of the other
  side within the NMS window (Chebyshev distance <= 10 heatmap px) and within TOL. Those
  pairs are counted separately ("moved").
- An unmatched peak within TOL of 0.05 (the floor) is excluded from the mismatch count
  ("excused"): a value that moves by less than TOL may cross the floor, so the peak exists
  on one side only. Proximity to 0.23 excuses nothing: a peak there is far above the floor
  and must exist on both sides. (Tightened after the first runs, review F2 on PR #188: the
  first version also excused peaks within TOL of 0.23. Its `excused` count was 0 on test,
  val and b', so no recorded verdict depended on it.)
- A pano where RampNet stored more than 50 peaks (the labeler's cap, MAX_PEAKS_PER_PANO) is
  compared on RampNet's first 50, which is exactly what peak_local_max(num_peaks=50) keeps
  provided the stored list is in peak_local_max's order (descending clipped confidence); the
  script refuses a capped pano whose list is not (``first_peaks``).
- A run that compares no pano does not pass.
- Reported, not gated beyond the above: max |difference| over matched pairs, and the
  operational (>= 0.23) count on each side.

The tolerance is not zero because the stored predictions came from an A40 on torch 2.6.0
and this run is on whatever GPU and torch it finds.

Before anything else the prediction directory's meta.json sha256 must equal the checkpoint's.

Usage (the Seattle test split, 515 panos; val and bprime the same way)::

    python scripts/crosswalk_parity.py \\
        --checkpoint <archive>/checkpoints/crosswalk_s4/final_step_1000.pth \\
        --preds <archive>/work/s4_preds/score_preds_v4/crosswalk_s4/test.jsonl \\
        --images <archive>/stage2_root/images/test --out <scratch>/parity_test

Writes ``summary.json`` and ``per_pano.csv`` to ``--out`` and exits 0 on PASS, 1 on FAIL.

``--native`` (REPORT ONLY, never the gate) fetches each pano through the labeler's own GSV
path instead (zoom 3 = 4096x2048, handed to the detector without a JPEG round trip), so the
numbers show how far the 0.23 count moves on the pixels the labeler actually feeds the
detector. Needs network; panos that no longer resolve are listed under ``not_fetched``.
Each compared pano also gets ``pixel_mad``: the mean absolute difference (0-255, over RGB)
between the live fetch and RampNet's stored JPEG, which says whether the two are the same
capture at all.

``--native --jpeg-quality 92`` (also REPORT ONLY) re-encodes each fetched pano as a JPEG at
that quality before detection, the way RampNet's inputs were stored (their quantization
tables are Pillow's quality 92, 4:2:0 chroma, which is Pillow's default subsampling), and
adds ``pixel_mad_jpeg`` for the re-encoded image. Compared with a plain ``--native`` run, it
separates what the JPEG round trip does from what different source pixels do.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detectors import MAX_PEAKS_PER_PANO, PEAK_HEATMAP_SHAPE, PEAK_MIN_DISTANCE  # noqa: E402

TOL = 5e-3
FLOOR = 0.05
OPERATING_POINT = 0.23


def to_pixel(x, y):
    """A normalized (x, y) -> its (row, col) on the 512x1024 heatmap (exact for argmax)."""
    h, w = PEAK_HEATMAP_SHAPE
    return round(y * h), round(x * w)


def near_floor(c):
    """True for a value within TOL of the storage floor: the only peaks that may exist on one
    side only (see the pass rule)."""
    return abs(c - FLOOR) <= TOL


def first_peaks(stored, cap=MAX_PEAKS_PER_PANO):
    """RampNet's first ``cap`` stored peaks: what peak_local_max(num_peaks=cap) keeps, PROVIDED
    the list is in peak_local_max's order, descending clip(confidence, 0, 1) (values above 1.0
    tie there, so only the clipped order is checked). Raises ValueError otherwise, rather than
    comparing an arbitrary 50.

        >>> first_peaks([(0, 0, 1.2), (0, 0, 1.4), (0, 0, 0.5)], cap=2)
        [(0, 0, 1.2), (0, 0, 1.4)]
    """
    clipped = [min(max(c, 0.0), 1.0) for _, _, c in stored]
    if any(a < b for a, b in zip(clipped, clipped[1:])):
        raise ValueError('stored peaks are not in descending (clipped) confidence order, so '
                         'their first %d are not what the capped finder keeps' % cap)
    return list(stored[:cap])


def jpeg_round_trip(image, quality):
    """``image`` encoded as a JPEG at ``quality`` with Pillow's defaults (4:2:0 chroma) and
    decoded again, as RGB: how RampNet's stored inputs were made."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    image.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert('RGB')


def pixel_mad(image, path):
    """Mean |image - the JPEG at path| over every RGB value (0-255), or None when the stored
    file is missing or a different size."""
    import numpy as np
    from PIL import Image
    if not Path(path).exists():
        return None
    with Image.open(path) as im:
        if im.size != image.size:
            return None
        ref = np.asarray(im.convert('RGB'), dtype=np.int16)
    return round(float(np.abs(np.asarray(image, dtype=np.int16) - ref).mean()), 3)


def compare(stored, labeler):
    """Match two peak lists [(x, y, c), ...] under the pass rule; returns a dict of counts and
    the matched differences. Pure, so it can be unit-tested without a GPU."""
    a = [(to_pixel(x, y), c) for x, y, c in stored]
    b = [(to_pixel(x, y), c) for x, y, c in labeler]
    out = {'exact': 0, 'moved': 0, 'diffs': [], 'bad_value': 0,
           'unmatched_stored': [], 'unmatched_labeler': [], 'excused': 0}
    b_at = {}
    for j, (px, _) in enumerate(b):
        b_at.setdefault(px, []).append(j)
    used_b, left_a = set(), []
    for i, (px, c) in enumerate(a):
        j = next((j for j in b_at.get(px, ()) if j not in used_b), None)
        if j is None:
            left_a.append(i)
            continue
        used_b.add(j)
        d = abs(c - b[j][1])
        out['diffs'].append(d)
        if d <= TOL:
            out['exact'] += 1
        else:
            out['bad_value'] += 1
    left_b = [j for j in range(len(b)) if j not in used_b]
    for i in left_a:     # a near-tie swap or a one-pixel move, within TOL only
        (r, col), c = a[i]
        cand = [j for j in left_b
                if max(abs(b[j][0][0] - r), abs(b[j][0][1] - col)) <= PEAK_MIN_DISTANCE
                and abs(b[j][1] - c) <= TOL]
        if cand:
            j = min(cand, key=lambda j: abs(b[j][1] - c))
            left_b.remove(j)
            out['moved'] += 1
            out['diffs'].append(abs(c - b[j][1]))
        else:
            out['unmatched_stored'].append(c)
    out['unmatched_labeler'] = [b[j][1] for j in left_b]
    for side in ('unmatched_stored', 'unmatched_labeler'):
        out['excused'] += sum(near_floor(c) for c in out[side])
        out[side] = [c for c in out[side] if not near_floor(c)]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--preds', required=True, help="RampNet's <split>.jsonl (meta.json beside it)")
    ap.add_argument('--images', required=True, help='directory of <pano_id>.jpg inputs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, help='first N panos only (a smoke test; not a gate)')
    ap.add_argument('--native', action='store_true',
                    help="REPORT ONLY, never a gate: fetch each pano through the labeler's own "
                         "GSV path (sources.gsv.fetch_pano_by_id: zoom 3, no JPEG round trip) "
                         "instead of reading --images, and compare with the stored predictions. "
                         "Needs network.")
    ap.add_argument('--jpeg-quality', type=int, metavar='Q',
                    help='with --native only, REPORT ONLY: re-encode each fetched pano as a JPEG '
                         'at quality Q (RampNet stored its inputs at 92) before detection')
    a = ap.parse_args(argv)
    if a.jpeg_quality is not None and not a.native:
        ap.error('--jpeg-quality re-encodes the live fetch; it needs --native')

    preds = Path(a.preds)
    meta = json.loads((preds.parent / 'meta.json').read_text(encoding='utf-8'))
    digest = sha256(a.checkpoint)
    if meta.get('sha256') != digest:
        raise SystemExit(f"{preds.parent / 'meta.json'} records sha256 {meta.get('sha256')}, "
                         f"but {a.checkpoint} is {digest}: not the same model")
    for k, want in (('score_floor', FLOOR), ('min_distance', PEAK_MIN_DISTANCE),
                    ('tta', False), ('fp16', False)):
        if meta.get(k) != want:
            raise SystemExit(f'meta.json {k} = {meta.get(k)!r}, expected {want!r}')
    rows = [json.loads(line) for line in preds.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    partial = a.limit is not None and a.limit < len(rows)
    if partial:
        rows = rows[:a.limit]

    import torch
    from detectors.crosswalk import CrosswalkDetector
    det = CrosswalkDetector(checkpoint=a.checkpoint)
    print(f'-> {det.provenance["model_id"]} on {det.DEVICE} '
          f'({torch.cuda.get_device_name(0) if det.DEVICE.type == "cuda" else "cpu"}), '
          f'torch {torch.__version__}; {len(rows)} panos', flush=True)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pano, t0 = [], time.time()
    tot = {'stored_peaks': 0, 'labeler_peaks': 0, 'exact': 0, 'moved': 0, 'bad_value': 0,
           'unmatched_stored': 0, 'unmatched_labeler': 0, 'excused': 0,
           'stored_op': 0, 'labeler_op': 0, 'op_count_mismatch_panos': 0}
    max_diff, capped, not_fetched = 0.0, [], []
    for n, (row, image) in enumerate(_images(rows, a), 1):
        pid = row['pano_id']
        if isinstance(image, str):          # --native: the fetch failed or was skipped
            not_fetched.append({'pano_id': pid, 'reason': image})
            continue
        stored = [tuple(p) for p in row['preds']]
        if len(stored) > MAX_PEAKS_PER_PANO:
            capped.append({'pano_id': pid, 'stored_peaks': len(stored)})
            try:
                stored = first_peaks(stored)
            except ValueError as e:
                raise SystemExit(f'{pid}: {e}')
        extra = {}
        if a.native:     # how far the live fetch is from the stored JPEG, before and after
            ref = Path(a.images) / f'{pid}.jpg'
            extra['pixel_mad'] = pixel_mad(image, ref)
            if a.jpeg_quality is not None:
                image = jpeg_round_trip(image, a.jpeg_quality)
                extra['pixel_mad_jpeg'] = pixel_mad(image, ref)
        labeler = det.detect(image)
        r = compare(stored, labeler)
        s_op = sum(c >= OPERATING_POINT for _, _, c in stored)
        l_op = sum(c >= OPERATING_POINT for _, _, c in labeler)
        md = max(r['diffs'], default=0.0)
        max_diff = max(max_diff, md)
        for k, v in (('stored_peaks', len(stored)), ('labeler_peaks', len(labeler)),
                     ('exact', r['exact']), ('moved', r['moved']), ('bad_value', r['bad_value']),
                     ('unmatched_stored', len(r['unmatched_stored'])),
                     ('unmatched_labeler', len(r['unmatched_labeler'])),
                     ('excused', r['excused']), ('stored_op', s_op), ('labeler_op', l_op),
                     ('op_count_mismatch_panos', int(s_op != l_op))):
            tot[k] += v
        per_pano.append({'pano_id': pid, 'stored': len(stored), 'labeler': len(labeler),
                         'exact': r['exact'], 'moved': r['moved'], 'bad_value': r['bad_value'],
                         'unmatched_stored': len(r['unmatched_stored']),
                         'unmatched_labeler': len(r['unmatched_labeler']),
                         'excused': r['excused'], 'stored_op': s_op, 'labeler_op': l_op,
                         'max_abs_diff': f'{md:.3g}', **extra})
        if n % 50 == 0:
            print(f'   {n}/{len(rows)} ({(time.time() - t0) / n:.2f} s/pano)', flush=True)

    passed = (bool(per_pano) and tot['bad_value'] == 0 and tot['unmatched_stored'] == 0
              and tot['unmatched_labeler'] == 0)
    # A --limit run is a smoke test: it can fail, but it never passes the gate. A --native run
    # is a report on different pixels (no JPEG round trip), never a gate.
    if a.native:
        verdict = 'NATIVE-REPORT'
    else:
        verdict = 'FAIL' if not passed else ('SMOKE-ONLY' if partial else 'PASS')
    summary = {'verdict': verdict, 'panos': len(rows),
               'panos_compared': len(rows) - len(not_fetched), **tot,
               'max_abs_diff': max_diff, 'tolerance': TOL, 'capped_panos': capped,
               'not_fetched': not_fetched,
               'checkpoint_sha256': digest, 'preds': str(preds),
               'images': (str(a.images) if not a.native else 'labeler GSV fetch (zoom 3)'
                          + (f' -> JPEG q{a.jpeg_quality}' if a.jpeg_quality is not None
                             else '')),
               'stored_meta': meta, 'model_id': det.provenance['model_id'],
               'device': str(det.DEVICE), 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name(0) if det.DEVICE.type == 'cuda' else None,
               'seconds': round(time.time() - t0, 1)}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    with open(out_dir / 'per_pano.csv', 'w', newline='', encoding='utf-8') as f:
        if per_pano:     # empty when no pano was compared (e.g. every --native fetch failed)
            w = csv.DictWriter(f, fieldnames=list(per_pano[0]))
            w.writeheader()
            w.writerows(per_pano)
    print(json.dumps({k: v for k, v in summary.items() if k != 'stored_meta'}, indent=1))
    return 0 if verdict in ('PASS', 'NATIVE-REPORT') else 1


def _images(rows, a, workers=8):
    """(row, PIL RGB image) per row, in order: the stored JPEG, or under --native the pano as
    the labeler's GSV source fetches it (a str reason instead of an image when that fails)."""
    from PIL import Image
    if not a.native:
        for row in rows:
            with Image.open(Path(a.images) / f"{row['pano_id']}.jpg") as im:
                yield row, im.convert('RGB')
        return
    from concurrent.futures import ThreadPoolExecutor
    from shapely.geometry import box
    from sources import gsv
    world = box(-180, -90, 180, 90)

    def fetch(row):
        got = gsv.fetch_pano_by_id(row['pano_id'], world)
        if got.get('status') != 'success':
            return f"{got.get('status')}: {got.get('reason')}"
        return got['image'].convert('RGB')

    # A bounded window of in-flight fetches: each image is ~25 MB decoded, and detection is
    # slower than eight fetchers, so an unbounded pool.map would hold the whole split in RAM.
    from collections import deque
    with ThreadPoolExecutor(workers) as pool:
        pending, todo = deque(), iter(rows)
        for row in todo:
            pending.append((row, pool.submit(fetch, row)))
            if len(pending) >= 2 * workers:
                break
        while pending:
            row, fut = pending.popleft()
            nxt = next(todo, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(fetch, nxt)))
            try:
                image = fut.result()
            except Exception as e:      # a network error is a reason, not a crash
                image = f'error: {e}'
            yield row, image


if __name__ == '__main__':
    sys.exit(main())
