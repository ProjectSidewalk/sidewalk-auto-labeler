"""Provenance gate: does a rebuilt run reproduce the AI labels a city already has? (#56)

When a city's `results.jsonl` was not kept, `scripts/detect_from_store.py` rebuilds it from
the pano store. Before anything is computed on the rebuilt run -- a benchmark, a clustering
evaluation -- it has to be shown to be usable in place of the run the deployed labels came
from. This joins the city's live AI CurbRamp labels to the run's stored detections on the
pixel key `send_to_ps.py` writes, `(pano_id, round(x * W), round(y * H))` (x wraps at the
seam), at BENCHMARK_CONFIDENCE (0.55): the tier the 2025 submissions were made at.

    # pull the labels once (cached; --refresh re-pulls) and see which account is the AI's
    python scripts/provenance_gate.py vancouver \\
        --server https://sidewalk-vancouver.cs.washington.edu --fetch-only

    # after the store run: the gate (Arm S only)
    python scripts/provenance_gate.py vancouver

    # the optional pipeline-identity control (Arm Z): draw the ids, re-detect them through
    # the 2025 path (zoom-3 GSV pixels), then gate with the control file
    python scripts/provenance_gate.py vancouver --draw-control          # 200 ids, seed 56
    python scripts/reinfer.py runs/vancouver \\
        --ids runs/vancouver/provenance_gate/control_ids.txt --out runs/vancouver/control_zoom3.jsonl
    python scripts/provenance_gate.py vancouver --control runs/vancouver/control_zoom3.jsonl

**Pre-registered reading** (`verdict()`). The first rule (PASS iff >= 0.98 of joinable
labels match within +/-1 px) was committed before any Vancouver number existed. It was
AMENDED after review (PR #96), still before any Vancouver number existed, because the store
run's pixels are a PIL-bilinear downsample of pano-tools' native JPEG where the 2025 run's
were Google's zoom-3 JPEG: a different low-pass filter and a different JPEG, so a peak that
moves one heatmap cell (W/1024 px) is a foreseeable outcome of resampling alone, and +/-1 px
would have read it as a different pipeline. The amended rule has two arms and two floors:

  * Arm S, store usability (always run). A label matches when a >= 0.55 detection on the
    same pano lies within +/-(W/1024 + 1) px in x and +/-(H/512 + 1) px in y -- one heatmap
    cell plus the rounding pixel, W and H the pano's stored native size. S passes iff
    >= PASS_SHARE (0.98) of joinable labels match. The share within +/-1 px is reported
    beside it as `exact_share` and never gates.
  * Arm Z, pipeline identity (optional control; --control <results.jsonl>). A file made
    through the 2025 path -- zoom-3 GSV pixels via panorama.fetch_panorama, i.e.
    `scripts/reinfer.py --ids` -- on a seeded subset of labeled panos that still resolve
    (CONTROL_SIZE = 200, CONTROL_SEED = 56). The same join at +/-1 px on the control's
    panos must reach >= PASS_SHARE. Threshold flips between the arms (labels matched under
    S but not Z, and vice versa, on the control's panos) are reported.
  * Coverage floor (UNDETERMINED otherwise). No pano may be pending -- every selected id is
    processed or cached-skipped -- AND joinable labels must be >= COVERAGE_FLOOR (0.95) of
    the AI labels whose pano has a store JPEG. The metadata-404 set
    (`metadata_404_null_field_or_no_file`: a null pano_data field or no server-side file,
    most often a null camera_pitch) is reported by count and as a share, so a non-random
    hole in the denominator is visible.
  * Precision (P). >= 0.55 detections on labeled panos that no label claims (under Arm S's
    tolerance) must be <= EXTRA_DETECTION_BOUND (0.02) x joinable labels, else STOP: the
    2025 run shipped every >= 0.55 detection on those panos. Soft-deleted AI labels are a
    known source of such detections (a retired label's detection is still in the run, its
    label is no longer in the pull), so a STOP here is a signal to investigate, not a
    verdict on the store.

PASS only if the coverage floor holds, S passes, P passes, and -- when a control is given --
Z passes; STOP otherwise, naming the failing arm(s). UNDETERMINED (coverage floor not met,
or nothing joinable in an arm that runs) takes precedence over STOP. Exit 0 / 1 / 2.

**The coarse-cell rule (#111; amended 2026-09-30, the default since).** Both tolerances above
assumed a detection is stable to the heatmap pixel. It is not: RampNet's head upsamples a
stride-32 map 8x bilinearly, so every stored detection sits on residue 3 or 4 of an 8-cell
block, and where two neighbouring coarse cells are near-tied a small input change moves the
argmax 7-8 heatmap cells -- the same ramp, decoded at the other cell (the Vancouver gate
under the rule above: 7,339 of 7,679 far misses at exactly 7 cells). So Arms S and Z now
both match within +/-1 coarse cell, +/-(8 W/1024 + 1) px in x and +/-(8 H/512 + 1) px in y
(Chebyshev, seam-wrapped), and the report splits every match by how far its detection sits
(same heatmap cell / grid neighbour / off-grid / **adjacent-coarse-cell flip** / further).
PASS_SHARE, the coverage floor and the precision bound are unchanged. The amendment was made
AFTER the Vancouver gate had run, so a Vancouver report under it is exploratory; `--rule
pixel-96` reproduces the rule above exactly (PR #108's report).

What the report carries besides the verdict, all diagnostics that never move it:
  - per pano, whether all / some / none of its AI labels matched;
  - for every unmatched label, the nearest stored detection at ANY confidence (pixel
    distance and confidence), histogrammed in heatmap cells; rows in unmatched.csv (keyed
    by `label_uid` = `<city>:<label_id>`, since a label_id is per city);
  - threshold flips: a detection within tolerance but below 0.55;
  - labels whose pano disagrees with the run on width/height, and the run's own count of
    store JPEGs whose native size differs from the server's (manifest run entries);
  - duplicate label keys: PS is insert-only, so a label can be live twice on the same
    pixel, and the join is many-to-one (both copies match one detection);
  - labels whose pano is not in the run, by why: not selected, skipped (with the reason
    detect_from_store recorded), or not yet processed.

Writes runs/<city>/provenance_gate/{report.md, unmatched.csv} (git-tracked; control_ids.txt
too) beside the label pull `raw_labels.geojson` (not tracked; report.md records its url,
fetch time, sha256 and feature count). Stdlib only, except that `--server` fetches through
eval_ps_clustering.fetch, which needs that script's analysis dependencies.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402  (stdlib-only)
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

# ---- Pre-registered (issue #56; amended after the PR #96 review, before any Vancouver ----
# ---- number existed). Do not tune these after seeing a city's numbers.               ----
PASS_SHARE = 0.98             # Arms S and Z: share of joinable AI labels that must match
TOLERANCE_PX = 1              # Arm Z (and exact_share): per axis, native pixels
STORE_SLACK_PX = 1            # Arm S: +/-(W/1024 + 1) x, +/-(H/512 + 1) y -- a cell + rounding
COVERAGE_FLOOR = 0.95         # joinable / AI labels whose pano has a store JPEG
EXTRA_DETECTION_BOUND = 0.02  # unclaimed tier detections on labeled panos / joinable labels
CONTROL_SIZE = 200            # Arm Z: labeled panos drawn for the zoom-3 control...
CONTROL_SEED = 56             # ...with this seed
TIER = BENCHMARK_CONFIDENCE
# -------------------------------------------------------------------------------------------

# ---- Amended again for #111 (2026-09-30), AFTER the Vancouver gate had run under the rule ----
# ---- above: see "The coarse-cell rule" in the module docstring. Both arms now match       ----
# ---- within one coarse heatmap cell; the #96 tolerances stay available as RULE_PIXEL_96.  ----
COARSE_CELL = geo.HEATMAP_COARSE_CELL_PX   # heatmap px per coarse (stride-32) cell
COARSE_CELLS = 1              # Arms S and Z: +/-1 coarse cell (+ the rounding pixel), Chebyshev
RULE_COARSE_CELL = 'coarse-cell'   # the default since #111
RULE_PIXEL_96 = 'pixel-96'         # the #96 rule the Vancouver gate (PR #108) ran under
# -------------------------------------------------------------------------------------------

HEATMAP_WIDTH = 1024    # detector heatmap columns: a detection's x is a multiple of W/1024
HEATMAP_HEIGHT = 512    # ...and rows

# detect_from_store's skip reasons (duplicated, not imported: that module pulls in requests
# and PIL, and this one stays stdlib; tests/test_provenance_gate.py pins them equal).
SKIP_NO_JPEG = 'jpg_missing'
SKIP_METADATA_UNSERVED = 'metadata_404_null_field_or_no_file'

API_LABELS = '/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson'
CONTROL_IDS_FILE = 'control_ids.txt'
# Distances in native px or in heatmap cells ('cell' = W/1024 px); first bucket that fits.
DISTANCE_BUCKETS = ((2, '<= 2 px'), (4, '<= 4 px'), ('cell', '<= 1 heatmap cell'),
                    ('2cell', '<= 2 heatmap cells'), ('coarse', '<= 1 coarse cell (8 heatmap cells)'),
                    ('2coarse', '<= 2 coarse cells'), (math.inf, '> 2 coarse cells'))
# How far (Chebyshev, heatmap cells, rounded) a matched detection sits from its label (#111):
# geo.CELL_SHIFT_CLASSES, named for the report.
MATCH_CLASSES = {'same_cell': 'same heatmap cell (0)',
                 'grid_neighbour': 'grid neighbour (1: residue 3 <-> 4)',
                 'off_grid': 'off-grid shift (2-6)', 'flip': 'adjacent-coarse-cell flip (7-8)',
                 'beyond': 'further (> 8)'}
UNMATCHED_FIELDS = ['label_uid', 'label_id', 'pano_id', 'pano_x', 'pano_y', 'run_width',
                    'run_height', 'label_width', 'label_height', 'reason',
                    'nearest_px', 'nearest_confidence', 'nearest_at_tier_px']
ARM_NAMES = {'S': 'Arm S (store usability)', 'Z': 'Arm Z (pipeline identity)',
             'P': 'precision (unclaimed detections)'}


def verdict(store_matched, joinable, labels_with_jpeg, pending, extra_detections,
            control_matched=None, control_joinable=None):
    """(verdict, detail) under the pre-registered, amended rule (module docstring).

    `store_matched` / `joinable`: Arm S's matches and joinable labels. `labels_with_jpeg`:
    AI labels whose pano has a store JPEG; `pending`: selected panos neither processed nor
    cached-skipped. `extra_detections`: tier detections on labeled panos that no label
    claims. `control_matched` / `control_joinable`: Arm Z, or None without a control.

    `detail` holds each share (None where undefined), `failing` (the arms that STOP it) and
    `reason` (why UNDETERMINED, else None).

    Example:
        >>> verdict(98, 100, 100, 0, 2)[0]
        'PASS'
        >>> verdict(98, 100, 100, 0, 3)[1]['failing']
        ['P']
        >>> verdict(98, 100, 106, 0, 0)[0]          # joinable 0.943 of labels with a JPEG
        'UNDETERMINED'
        >>> verdict(99, 100, 100, 0, 0, control_matched=190, control_joinable=200)[1]['failing']
        ['Z']
    """
    detail = {
        'store_share': store_matched / joinable if joinable else None,
        'coverage_share': joinable / labels_with_jpeg if labels_with_jpeg else None,
        'extra_ratio': extra_detections / joinable if joinable else None,
        'control_share': (control_matched / control_joinable
                          if control_joinable else None),
        'pending': pending, 'failing': [], 'reason': None,
    }
    if pending:
        detail['reason'] = f'{pending} selected pano(s) still pending'
    elif not joinable:
        detail['reason'] = 'no joinable label'
    elif detail['coverage_share'] < COVERAGE_FLOOR:
        detail['reason'] = (f'joinable labels are {detail["coverage_share"]:.4f} of the AI labels '
                            f'whose pano has a store JPEG (floor {COVERAGE_FLOOR})')
    elif control_joinable is not None and not control_joinable:
        detail['reason'] = 'the control holds no joinable label'
    if detail['reason']:
        return 'UNDETERMINED', detail
    if detail['store_share'] < PASS_SHARE:
        detail['failing'].append('S')
    if control_joinable is not None and detail['control_share'] < PASS_SHARE:
        detail['failing'].append('Z')
    if detail['extra_ratio'] > EXTRA_DETECTION_BOUND:
        detail['failing'].append('P')
    return ('STOP' if detail['failing'] else 'PASS'), detail


# ------------------------------------------------------------------------------ inputs

def load_ai_labels(path, ai_user=None):
    """(labels, ai_user, users): the CurbRamp labels of the AI account from a rawLabels
    geojson. Without `ai_user` the account holding the most CurbRamp labels is taken, and
    `users` (a Counter of every account) is returned so the report shows the choice."""
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    rows, users = [], Counter()
    for ft in feats:
        q = ft.get('properties') or {}
        if q.get('label_type', 'CurbRamp') != 'CurbRamp':
            continue
        users[str(q.get('user_id'))] += 1
        rows.append(q)
    if not rows:
        raise SystemExit(f'{path}: no CurbRamp labels')
    chosen = ai_user if ai_user is not None else users.most_common(1)[0][0]
    labels = [{'label_id': q['label_id'],
               'pano_id': q.get('pano_id') or q.get('gsv_panorama_id'),
               'pano_x': int(q['pano_x']), 'pano_y': int(q['pano_y']),
               'pano_width': q.get('pano_width'), 'pano_height': q.get('pano_height')}
              for q in rows if str(q.get('user_id')) == chosen]
    return labels, chosen, users


def load_run(results_path):
    """{pano_id: (W, H, [(px, py, confidence), ...])} from results.jsonl, detections keyed
    by the pixel send_to_ps.py computes."""
    run = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            w, h = int(p['width']), int(p['height'])
            dets = [(round(d['x_normalized'] * w), round(d['y_normalized'] * h),
                     d['confidence']) for d in rec.get('detections', [])]
            run[p['panorama_id']] = (w, h, dets)
    return run


def load_selection(run_dir):
    """(selected ids or None, {pano_id: skip reason}) from detect_from_store's files."""
    ids_path = run_dir / 'store_ids.txt'
    selected = None
    if ids_path.exists():
        selected = {l.strip() for l in ids_path.read_text(encoding='utf-8').splitlines()
                    if l.strip()}
    skips = {}
    skip_path = run_dir / 'store_skipped.jsonl'
    if skip_path.exists():
        for line in skip_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                skips[row['pano_id']] = row['reason']
    return selected, skips


def native_size_mismatches(run_dir):
    """The store runs' own count of JPEGs whose native size differs from the server's
    width/height (manifest run entries, phase 'store'), or None without a manifest."""
    path = run_dir / 'manifest.json'
    if not path.exists():
        return None
    runs = json.loads(path.read_text(encoding='utf-8')).get('runs') or []
    return sum(r.get('native_size_mismatch', 0) for r in runs if r.get('phase') == 'store')


# ------------------------------------------------------------------------------- join

def store_tolerance(w, h):
    """Arm S under the #96 rule: one heatmap cell plus the rounding pixel, per axis."""
    return w / HEATMAP_WIDTH + STORE_SLACK_PX, h / HEATMAP_HEIGHT + STORE_SLACK_PX


def exact_tolerance(w, h):
    """Arm Z under the #96 rule (and exact_share under both): +/-1 px per axis."""
    return TOLERANCE_PX, TOLERANCE_PX


def coarse_tolerance(w, h):
    """Arms S and Z under the #111 rule: +/-COARSE_CELLS coarse cells plus the rounding
    pixel, per axis (a box, so Chebyshev), x wrapping at the seam.

    Example:
        >>> coarse_tolerance(16384, 8192)        # 8 heatmap cells of 16 px, + 1
        (129.0, 129.0)
    """
    return (COARSE_CELLS * COARSE_CELL * w / HEATMAP_WIDTH + STORE_SLACK_PX,
            COARSE_CELLS * COARSE_CELL * h / HEATMAP_HEIGHT + STORE_SLACK_PX)


# rule -> (Arm S tolerance, Arm Z tolerance)
RULES = {RULE_COARSE_CELL: (coarse_tolerance, coarse_tolerance),
         RULE_PIXEL_96: (store_tolerance, exact_tolerance)}




def _dx(a, b, w):
    d = abs(a - b) % w
    return min(d, w - d)


def _dist(label, det, w):
    return math.hypot(_dx(label[0], det[0], w), label[1] - det[1])


def _within(pt, det, w, tol):
    return _dx(pt[0], det[0], w) <= tol[0] and abs(pt[1] - det[1]) <= tol[1]


def join(labels, run, selected=None, skips=None, tolerance=store_tolerance):
    """Score every AI label against the run under `tolerance` ((W, H) -> (tx, ty)).
    Returns a dict of tallies and rows; see the module docstring for what each diagnostic
    means. `matched_ids` is the set of matched label_ids (for the arm comparison);
    `within_1px` counts labels with a tier detection within +/-1 px whatever the arm."""
    skips = skips or {}
    out = {'joinable': 0, 'matched': 0, 'within_1px': 0, 'on_pixel': 0, 'sub_tier': 0,
           'dims_differ': 0, 'not_in_run': Counter(), 'unmatched': [], 'buckets': Counter(),
           'panos': Counter(), 'unlabeled_dets': {'labeled_panos': 0, 'unlabeled_panos': 0},
           'matched_ids': set(), 'match_classes': Counter(), 'shared_claims': 0}
    claims = defaultdict(set)    # (pano, detection index) -> distinct label pixels claiming it
    keys = Counter((lab['pano_id'], lab['pano_x'], lab['pano_y']) for lab in labels)
    out['duplicate_keys'] = sum(1 for n in keys.values() if n > 1)
    out['duplicate_labels'] = sum(n for n in keys.values() if n > 1)
    by_pano = defaultdict(list)
    for lab in labels:
        by_pano[lab['pano_id']].append(lab)

    used = defaultdict(set)      # pano -> indices of tier detections claimed by some label
    for pid, labs in by_pano.items():
        if pid not in run:
            if selected is not None and pid not in selected:
                why = 'not selected'
            elif pid in skips:
                why = f'skipped: {skips[pid]}'
            else:
                why = 'not processed yet (pending or failed)'
            out['not_in_run'][why] += len(labs)
            continue
        w, h, dets = run[pid]
        tol = tolerance(w, h)
        cell = w / HEATMAP_WIDTH
        n_ok = 0
        for lab in labs:
            out['joinable'] += 1
            pt = (lab['pano_x'], lab['pano_y'])
            dims_differ = (lab.get('pano_width') not in (None, w)
                           or lab.get('pano_height') not in (None, h))
            out['dims_differ'] += dims_differ
            tier = [(i, d) for i, d in enumerate(dets) if d[2] >= TIER]
            out['within_1px'] += any(_within(pt, d, w, exact_tolerance(w, h)) for _, d in tier)
            hit = None
            for i, d in tier:
                if _within(pt, d, w, tol) and (hit is None or _dist(pt, d, w) < _dist(pt, dets[hit], w)):
                    hit = i
            if hit is not None:
                n_ok += 1
                out['matched'] += 1
                out['matched_ids'].add(lab['label_id'])
                out['on_pixel'] += dets[hit][:2] == pt
                out['match_classes'][geo.cell_shift_class(
                    geo.heatmap_cell_distance(pt, dets[hit], w, h))] += 1
                claims[(pid, hit)].add(pt)
                used[pid].add(hit)
                continue
            near = min(dets, key=lambda d: _dist(pt, d, w), default=None)
            near_tier = min((d for _, d in tier), key=lambda d: _dist(pt, d, w), default=None)
            dist = None if near is None else _dist(pt, near, w)
            # No tier detection is within tolerance; if any detection is, it sits below the
            # tier: the label's peak is there, its confidence is not (a threshold flip).
            sub_tier = any(_within(pt, d, w, tol) for d in dets)
            out['sub_tier'] += sub_tier
            if dist is None:
                out['buckets']['no detection on the pano'] += 1
            else:
                for bound, name in DISTANCE_BUCKETS:
                    limit = {'cell': cell, '2cell': 2 * cell, 'coarse': COARSE_CELL * cell,
                             '2coarse': 2 * COARSE_CELL * cell}.get(bound, bound)
                    if dist <= limit:
                        out['buckets'][name] += 1
                        break
            reason = ('dims differ' if dims_differ else
                      'no detection on the pano' if near is None else
                      'below tier within tolerance' if sub_tier else
                      'no detection within tolerance')
            out['unmatched'].append({
                'label_id': lab['label_id'], 'pano_id': pid, 'pano_x': pt[0], 'pano_y': pt[1],
                'run_width': w, 'run_height': h, 'label_width': lab.get('pano_width'),
                'label_height': lab.get('pano_height'), 'reason': reason,
                'nearest_px': None if dist is None else round(dist, 2),
                'nearest_confidence': None if near is None else round(near[2], 6),
                'nearest_at_tier_px': None if near_tier is None else round(_dist(pt, near_tier, w), 2)})
        out['panos']['all matched' if n_ok == len(labs) else 'none matched' if n_ok == 0
                     else 'partly matched'] += 1

    # A detection claimed by labels at two different pixels: with a coarse-cell tolerance two
    # ramps within ~2.8 deg can share one detection (the join is nearest-within-tolerance,
    # not one-to-one). Duplicate label keys (same pixel) are not counted here.
    out['shared_claims'] = sum(1 for pts in claims.values() if len(pts) > 1)
    for pid, (w, h, dets) in run.items():
        n = sum(1 for i, d in enumerate(dets) if d[2] >= TIER and i not in used[pid])
        out['unlabeled_dets']['labeled_panos' if pid in by_pano else 'unlabeled_panos'] += n
    return out


def coverage(labels, run, selected=None, skips=None):
    """The coverage floor's inputs. `pending`: selected panos (else, without a selection,
    labeled panos) neither in the run nor cached-skipped. `labels_with_jpeg`: AI labels
    whose pano was not skipped `jpg_missing` (a label on a pano never selected counts: its
    JPEG was never looked for). The metadata-404 set, by panos and by labels on them."""
    skips = skips or {}
    labeled = {lab['pano_id'] for lab in labels}
    universe = selected if selected is not None else labeled
    pending = sorted(p for p in universe if p not in run and p not in skips)
    no_jpeg = {p for p, r in skips.items() if r == SKIP_NO_JPEG}
    unserved = {p for p, r in skips.items() if r == SKIP_METADATA_UNSERVED and p in universe}
    return {'pending': len(pending), 'pending_examples': pending[:5],
            'n_selected': len(universe), 'labels': len(labels),
            'labels_with_jpeg': sum(1 for lab in labels if lab['pano_id'] not in no_jpeg),
            'unserved_panos': len(unserved),
            'unserved_labeled_panos': len(unserved & labeled),
            'unserved_labels': sum(1 for lab in labels if lab['pano_id'] in unserved)}


def compare_arms(store_res, control_res, labels, control_run):
    """Labels on the control's panos: matched under S (store run) and/or Z (control)."""
    on_control = [lab for lab in labels if lab['pano_id'] in control_run]
    s, z = store_res['matched_ids'], control_res['matched_ids']
    out = Counter()
    for lab in on_control:
        lid = lab['label_id']
        out['both' if lid in s and lid in z else 'S only' if lid in s
            else 'Z only' if lid in z else 'neither'] += 1
    return out


def draw_control(labels, run, n=CONTROL_SIZE, seed=CONTROL_SEED):
    """`n` AI-labeled pano ids that are in the store run, drawn with random.Random(seed)
    from the SORTED candidates, so the same run and labels always give the same draw."""
    pool = sorted({lab['pano_id'] for lab in labels} & set(run))
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


# ------------------------------------------------------------------------------ report

def pull_record(path):
    """One report line for the label pull: url, fetch time, sha256, feature count."""
    side = path.parent / (path.name + '.source.json')
    meta = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    n = len(json.loads(path.read_text(encoding='utf-8'))['features'])
    return (f"`{path.name}`: {n:,} features, sha256 `{digest}`, fetched "
            f"{meta.get('fetched_at', '(unrecorded)')} from {meta.get('url', '(supplied on the command line)')}")


def _verdict_for(res, cov, control=None):
    return verdict(res['matched'], res['joinable'], cov['labels_with_jpeg'], cov['pending'],
                   res['unlabeled_dets']['labeled_panos'],
                   None if control is None else control['matched'],
                   None if control is None else control['joinable'])


def rule_lines(rule):
    """The report's statement of the Arm S / Arm Z tolerances under `rule`."""
    if rule == RULE_PIXEL_96:
        return [
            '## Rule (pre-registered; amended after the PR #96 review, before any Vancouver number)',
            '',
            f'- **Arm S, store usability:** a label matches when a stored detection >= {TIER} lies '
            f'within +/-(W/{HEATMAP_WIDTH} + {STORE_SLACK_PX}) px in x and +/-(H/{HEATMAP_HEIGHT} + '
            f'{STORE_SLACK_PX}) px in y of it (key `round(x*W), round(y*H)`, x wrapping at the seam). '
            f'Passes iff >= {PASS_SHARE} of joinable labels match. `exact_share` (+/-{TOLERANCE_PX} px) '
            f'is reported, never gated.',
            f'- **Arm Z, pipeline identity (with --control):** the same join at +/-{TOLERANCE_PX} px on a '
            f'zoom-3 control file ({CONTROL_SIZE} seeded labeled panos, seed {CONTROL_SEED}, via '
            f'`reinfer.py --ids`); passes iff >= {PASS_SHARE}.']
    return [
        f'## Rule (`{RULE_COARSE_CELL}`: amended for #111 on 2026-09-30, after the Vancouver gate '
        f'had run under `{RULE_PIXEL_96}`)', '',
        f'- **Arms S and Z:** a label matches when a stored detection >= {TIER} lies within '
        f'+/-{COARSE_CELLS} coarse heatmap cell -- +/-({COARSE_CELL}W/{HEATMAP_WIDTH} + '
        f'{STORE_SLACK_PX}) px in x and +/-({COARSE_CELL}H/{HEATMAP_HEIGHT} + {STORE_SLACK_PX}) px '
        f"in y, Chebyshev, x wrapping at the seam (key `round(x*W), round(y*H)`). RampNet's "
        f'heatmap is a bilinear 8x upsample of a stride-32 map, so a detection can only land on '
        f'residue 3 or 4 of each 8-cell block, and a near-tie between two coarse cells moves it by '
        f'7-8 heatmap cells on a small input change (the **adjacent-coarse-cell flip**, counted '
        f'below). Arm S runs on the store run; Arm Z on a zoom-3 control file ({CONTROL_SIZE} '
        f'seeded labeled panos, seed {CONTROL_SEED}, via `reinfer.py --ids`). Each passes iff '
        f'>= {PASS_SHARE}. `exact_share` (+/-{TOLERANCE_PX} px) is reported, never gated. '
        f'`--rule {RULE_PIXEL_96}` reproduces the previous rule.']


def class_table(res):
    """Matched labels by the Chebyshev distance (heatmap cells) of their detection."""
    n = res['matched']
    rows = ['| matched detection sits | labels | share of matched |', '|---|---:|---:|']
    for key, name in MATCH_CLASSES.items():
        k = res['match_classes'][key]
        rows.append(f'| {name} | {k:,} | {k / n:.4f} |' if n else f'| {name} | 0 | n/a |')
    return rows


def render(city, res, cov, ai_user, users, pull_line, results_path, n_run,
           control=None, control_path=None, arms=None, size_mismatch=None,
           rule=RULE_COARSE_CELL, exploratory=None):
    v, d = _verdict_for(res, cov, control)
    pct = lambda n, t: f'{n:,} ({100 * n / t:.2f}%)' if t else f'{n:,}'  # noqa: E731
    fmt = lambda x: 'n/a' if x is None else f'{x:.4f}'  # noqa: E731
    head = f'**Verdict: {v}**'
    if v == 'UNDETERMINED':
        head += f' -- {d["reason"]}.'
    elif v == 'STOP':
        head += ' -- failing: ' + ', '.join(ARM_NAMES[a] for a in d['failing']) + '.'
    lines = [
        f'# {city}: provenance gate (#56)', '', head, '',
        *([f'> **Exploratory:** {exploratory}', ''] if exploratory else []),
        *rule_lines(rule),
        f'- **Coverage floor:** UNDETERMINED unless no selected pano is pending and joinable '
        f'labels are >= {COVERAGE_FLOOR} of the AI labels whose pano has a store JPEG.',
        f'- **Precision (P):** detections >= {TIER} on labeled panos that no label claims must be '
        f'<= {EXTRA_DETECTION_BOUND} x joinable labels, else STOP. Soft-deleted AI labels are a '
        f'known source of such detections: a STOP here is a signal to investigate, not a verdict '
        f'on the store.',
        '- PASS only if every arm that runs passes; UNDETERMINED takes precedence over STOP. '
        'STOP means nothing downstream runs.', '',
        '| check | value | rule | result |', '|---|---:|---|---|',
        f'| Arm S share | {fmt(d["store_share"])} | >= {PASS_SHARE} | '
        f'{"n/a" if d["store_share"] is None else "fail" if "S" in d["failing"] else "pass"} |',
        f'| exact_share (+/-{TOLERANCE_PX} px, not gated) | '
        f'{fmt(res["within_1px"] / res["joinable"] if res["joinable"] else None)} | -- | -- |',
        f'| Arm Z share | {fmt(d["control_share"])} | >= {PASS_SHARE} | '
        f'{"not run" if control is None else "n/a" if d["control_share"] is None else "fail" if "Z" in d["failing"] else "pass"} |',
        f'| coverage (joinable / labels with a JPEG) | {fmt(d["coverage_share"])} | >= {COVERAGE_FLOOR} | '
        f'{"n/a" if d["coverage_share"] is None else "fail" if d["coverage_share"] < COVERAGE_FLOOR else "pass"} |',
        f'| pending selected panos | {cov["pending"]:,} | 0 | {"fail" if cov["pending"] else "pass"} |',
        f'| unclaimed tier detections / joinable | {fmt(d["extra_ratio"])} | <= {EXTRA_DETECTION_BOUND} | '
        f'{"n/a" if d["extra_ratio"] is None else "fail" if "P" in d["failing"] else "pass"} |', '',
        '## Inputs', '',
        f'- Labels: {pull_line}',
        f'- AI account: `{ai_user}` ({users[ai_user]:,} CurbRamp labels); every account: ' +
        ', '.join(f'`{u}` {c:,}' for u, c in users.most_common()),
        f'- Run: `{results_path.as_posix()}`, {n_run:,} panos, sha256 '
        f'`{hashlib.sha256(results_path.read_bytes()).hexdigest()}`',
    ]
    if control_path is not None:
        lines.append(f'- Control: `{control_path.as_posix()}`, sha256 '
                     f'`{hashlib.sha256(control_path.read_bytes()).hexdigest()}`')
    lines += [
        f'- Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}', '',
        '## Coverage', '',
        f'- Selected panos: {cov["n_selected"]:,}; pending (neither processed nor cached-skipped): '
        f'{cov["pending"]:,}' + (f', e.g. {", ".join(cov["pending_examples"])}' if cov['pending'] else ''),
        f'- AI labels: {cov["labels"]:,}; on a pano with a store JPEG: {cov["labels_with_jpeg"]:,}; '
        f'joinable: {res["joinable"]:,}',
        f'- Metadata 404 (`{SKIP_METADATA_UNSERVED}`: a null pano_data field -- most often '
        f'camera_pitch -- or no server-side file): {pct(cov["unserved_panos"], cov["n_selected"])} '
        f'of selected panos ({cov["unserved_labeled_panos"]:,} of them labeled), carrying '
        f'{pct(cov["unserved_labels"], cov["labels"])} of the AI labels',
        f'- Store JPEGs whose native size differs from the server\'s width/height (manifest): '
        f'{"(no manifest)" if size_mismatch is None else f"{size_mismatch:,}"}', '',
        '## Arm S join', '',
        f'- Joinable labels (pano in the run): {res["joinable"]:,}',
        f'- Matched under Arm S: {pct(res["matched"], res["joinable"])}; of those exactly on the '
        f'pixel: {res["on_pixel"]:,}',
        f'- Within +/-{TOLERANCE_PX} px at >= {TIER} (exact_share): {pct(res["within_1px"], res["joinable"])}',
        f'- Unmatched: {len(res["unmatched"]):,}; of those, a stored detection BELOW {TIER} sits '
        f'within tolerance (a threshold flip): {res["sub_tier"]:,}',
        f'- Tier detections claimed by labels at two or more different pixels (the join is '
        f'nearest-within-tolerance, not one-to-one): {res["shared_claims"]:,}',
        f'- Labels whose pano width/height differ from the run\'s: {res["dims_differ"]:,}',
        f'- Duplicate label keys (same pano and pixel; PS is insert-only, and the join is '
        f'many-to-one): {res["duplicate_keys"]:,} key(s) carrying {res["duplicate_labels"]:,} labels', '',
        '| panos with AI labels | count |', '|---|---:|',
        *[f'| {k} | {res["panos"][k]:,} |' for k in ('all matched', 'partly matched', 'none matched')],
        '', '### Matched labels: where the detection sits (heatmap cells, Chebyshev; #111)', '',
        *class_table(res), '',
        'A flip (7-8 cells) is the same ramp decoded at the neighbouring coarse cell of a '
        'near-tied pair; residue 3 <-> 4 (1 cell) is the same coarse cell.',
        '', '### Unmatched labels: nearest stored detection (any confidence)', '',
        '| distance | labels |', '|---|---:|',
        *[f'| {name} | {res["buckets"][name]:,} |'
          for name in [n for _, n in DISTANCE_BUCKETS] + ['no detection on the pano']],
        '', f'A heatmap cell is W/{HEATMAP_WIDTH} px (16 px on a 16384-wide pano).', '',
        f'## Detections >= {TIER} that no label claims (under Arm S)', '',
        f'- on panos carrying AI labels (gated by P): {res["unlabeled_dets"]["labeled_panos"]:,}',
        f'- on panos carrying none (e.g. the sampled empty stratum; not gated): '
        f'{res["unlabeled_dets"]["unlabeled_panos"]:,}', '',
    ]
    if control is not None:
        lines += [
            '## Arm Z (control)', '',
            f'- Joinable labels on the control\'s panos: {control["joinable"]:,}; matched '
            f'under Arm Z: {pct(control["matched"], control["joinable"])}; within '
            f'+/-{TOLERANCE_PX} px: {pct(control["within_1px"], control["joinable"])}',
            f'- Unmatched: {len(control["unmatched"]):,}; threshold flips within tolerance: '
            f'{control["sub_tier"]:,}', '',
            *class_table(control), '',
            '| on the control\'s panos | labels |', '|---|---:|',
            *[f'| {k} | {arms[k]:,} |' for k in ('both', 'S only', 'Z only', 'neither')],
            '', '`S only`: matched under Arm S on the store run but not under Arm Z on the '
            'control; `Z only` the reverse.', '',
        ]
    lines += [
        '## AI labels whose pano is not in the run', '',
        '| why | labels |', '|---|---:|',
        *[f'| {k} | {c:,} |' for k, c in sorted(res['not_in_run'].items())],
        *([] if res['not_in_run'] else ['| (none) | 0 |']), '',
    ]
    return '\n'.join(lines)


def write_unmatched(path, city, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=UNMATCHED_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r['label_id']):
            w.writerow({'label_uid': f"{city}:{r['label_id']}", **r})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', help='run name: reads runs/<city>/results.jsonl')
    ap.add_argument('--run-dir', type=Path, help='default runs/<city>')
    ap.add_argument('--server', help='PS server to pull /v3/api/rawLabels from (cached)')
    ap.add_argument('--refresh', action='store_true', help='re-pull the labels over the cache')
    ap.add_argument('--labels', type=Path, help='a rawLabels geojson instead of the pull')
    ap.add_argument('--ai-user', help="the AI account's user_id (default: the account with "
                                      "the most CurbRamp labels, as reported)")
    ap.add_argument('--out', type=Path, help='default runs/<city>/provenance_gate')
    ap.add_argument('--fetch-only', action='store_true',
                    help='pull the labels, print the accounts, and stop')
    ap.add_argument('--control', type=Path, metavar='RESULTS',
                    help='Arm Z: a results.jsonl made through the 2025 path (zoom-3 GSV pixels; '
                         'scripts/reinfer.py --ids <control_ids.txt> --out RESULTS)')
    ap.add_argument('--draw-control', type=int, nargs='?', const=CONTROL_SIZE, metavar='N',
                    help=f'write <out>/{CONTROL_IDS_FILE}: N (default {CONTROL_SIZE}) AI-labeled '
                         f'panos of the run, seed --control-seed, then stop')
    ap.add_argument('--control-seed', type=int, default=CONTROL_SEED)
    ap.add_argument('--rule', choices=sorted(RULES), default=RULE_COARSE_CELL,
                    help=f'match tolerances: {RULE_COARSE_CELL} (the default since #111: both arms '
                         f'+/-1 coarse heatmap cell) or {RULE_PIXEL_96} (the rule the Vancouver '
                         f'gate ran under in PR #108: Arm S one heatmap cell, Arm Z +/-1 px)')
    ap.add_argument('--exploratory', metavar='NOTE',
                    help='mark the report exploratory with this note (a check re-run after the '
                         'decision it would have gated was taken)')
    args = ap.parse_args(argv)
    tol_s, tol_z = RULES[args.rule]

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    out = args.out or run_dir / 'provenance_gate'
    out.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels or out / 'raw_labels.geojson'
    if args.server:
        from eval_ps_clustering import fetch   # analysis deps; only for the pull
        fetch(args.server.rstrip('/') + API_LABELS, labels_path, args.refresh)
    elif args.refresh:
        raise SystemExit('--refresh needs --server')
    if not labels_path.exists():
        raise SystemExit(f'{labels_path} does not exist: pass --server to pull it, or --labels')

    labels, ai_user, users = load_ai_labels(labels_path, args.ai_user)
    if args.fetch_only:
        print(f'{sum(users.values()):,} CurbRamp labels; accounts: ' +
              ', '.join(f'{u} {c:,}' for u, c in users.most_common()))
        print(f'AI account (most labels unless --ai-user): {ai_user}, {len(labels):,} labels on '
              f'{len({l["pano_id"] for l in labels}):,} panos')
        return 0

    results_path = run_dir / 'results.jsonl'
    if not results_path.exists():
        raise SystemExit(f'{results_path} does not exist')
    run = load_run(results_path)

    if args.draw_control is not None:
        ids_path = out / CONTROL_IDS_FILE
        if ids_path.exists():
            raise SystemExit(f'{ids_path} exists; the control draw is made once. Move it aside '
                             f'to draw again.')
        if args.draw_control < 1:
            raise SystemExit('--draw-control needs at least 1 pano')
        ids = draw_control(labels, run, args.draw_control, args.control_seed)
        ids_path.write_text(''.join(f'{i}\n' for i in ids), encoding='utf-8', newline='\n')
        print(f'-> {len(ids)} control pano ids (seed {args.control_seed}) -> {ids_path}\n'
              f'   next: python scripts/reinfer.py {run_dir.as_posix()} --ids {ids_path.as_posix()} '
              f'--out {(run_dir / "control_zoom3.jsonl").as_posix()}')
        return 0

    selected, skips = load_selection(run_dir)
    res = join(labels, run, selected, skips, tolerance=tol_s)
    cov = coverage(labels, run, selected, skips)
    control = arms = None
    if args.control is not None:
        if not args.control.exists():
            raise SystemExit(f'{args.control} does not exist')
        control_run = load_run(args.control)
        control_labels = [lab for lab in labels if lab['pano_id'] in control_run]
        control = join(control_labels, control_run, tolerance=tol_z)
        arms = compare_arms(res, control, labels, control_run)
    report = render(args.city, res, cov, ai_user, users, pull_record(labels_path), results_path,
                    len(run), control, args.control, arms, native_size_mismatches(run_dir),
                    rule=args.rule, exploratory=args.exploratory)
    (out / 'report.md').write_text(report, encoding='utf-8')
    write_unmatched(out / 'unmatched.csv', args.city, res['unmatched'])
    v, d = _verdict_for(res, cov, control)
    tail = (d['reason'] if v == 'UNDETERMINED' else
            'failing: ' + ', '.join(ARM_NAMES[a] for a in d['failing']) if v == 'STOP' else
            f'S {d["store_share"]:.4f}' + ('' if control is None else f', Z {d["control_share"]:.4f}'))
    print(f'{args.city}: {v} [{args.rule}] ({tail}) -> {out / "report.md"}')
    flip = 'flip'
    print(f'  Arm S: {res["match_classes"][flip]:,} of {res["matched"]:,} matches are '
          f'adjacent-coarse-cell flips' + ('' if control is None else
                                           f'; Arm Z: {control["match_classes"][flip]:,} of '
                                           f'{control["matched"]:,}'))
    return {'PASS': 0, 'STOP': 1}.get(v, 2)


if __name__ == '__main__':
    sys.exit(main())
