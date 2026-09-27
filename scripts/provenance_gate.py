"""Provenance gate: does a rebuilt run reproduce the AI labels a city already has? (#56)

When a city's `results.jsonl` was not kept, `scripts/detect_from_store.py` rebuilds it from
the pano store. Before anything is computed on the rebuilt run -- a benchmark, a clustering
evaluation -- it has to be shown to be the run the deployed labels came from. This joins
the city's live AI CurbRamp labels to the run's stored detections on the pixel key
`send_to_ps.py` writes, `(pano_id, round(x * W), round(y * H))`, with a +/-1 px tolerance
on each axis (x wraps at the seam), at BENCHMARK_CONFIDENCE (0.55): the tier the 2025
submissions were made at.

    # pull the labels once (cached; --refresh re-pulls) and see which account is the AI's
    python scripts/provenance_gate.py vancouver \\
        --server https://sidewalk-vancouver.cs.washington.edu --fetch-only

    # after the run: the gate
    python scripts/provenance_gate.py vancouver

**Pre-registered reading** (`verdict()`, committed before any Vancouver number existed):
PASS iff at least PASS_SHARE = 0.98 of the joinable AI labels (those whose pano is in the
run) have a detection within +/-1 px at >= 0.55. Anything less is STOP: the 2025 pipeline
differed from this one somewhere, and nothing downstream runs until it is found. No
joinable label at all is UNDETERMINED. Exit 0 / 1 / 2 respectively.

What the report carries besides the verdict, all diagnostics that never move it:
  - per pano, whether all / some / none of its AI labels matched;
  - for every unmatched label, the nearest stored detection at ANY confidence (pixel
    distance and confidence) -- a near-miss one heatmap cell away (W / 1024 px) reads as
    resampling, a match below 0.55 as a threshold flip; histogram in report.md, rows in
    unmatched.csv (keyed by `label_uid` = `<city>:<label_id>`, since a label_id is per city);
  - labels whose pano disagrees with the run on width/height;
  - detections >= 0.55 with no label, split by whether the pano carries AI labels;
  - labels whose pano is not in the run, by why: not selected, skipped (and the reason
    detect_from_store recorded: no JPEG in the store, metadata 404, ...), or not yet
    processed.

Writes runs/<city>/provenance_gate/{report.md, unmatched.csv} (git-tracked) beside the
label pull `raw_labels.geojson` (not tracked; report.md records its url, fetch time,
sha256 and feature count). Stdlib only, except that `--server` fetches through
eval_ps_clustering.fetch, which needs that script's analysis dependencies.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

# ---- Pre-registered (issue #56). Do not tune these after seeing a city's numbers. ----
PASS_SHARE = 0.98       # share of joinable AI labels that must match
TOLERANCE_PX = 1        # per axis, in the pano's native pixels
TIER = BENCHMARK_CONFIDENCE
# -------------------------------------------------------------------------------------

API_LABELS = '/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson'
HEATMAP_WIDTH = 1024    # detector heatmap columns: a detection's x is a multiple of W/1024
DISTANCE_BUCKETS = ((2, '<= 2 px'), (4, '<= 4 px'), ('cell', '<= 1 heatmap cell'),
                    ('2cell', '<= 2 heatmap cells'), (64, '<= 64 px'), (math.inf, '> 64 px'))
UNMATCHED_FIELDS = ['label_uid', 'label_id', 'pano_id', 'pano_x', 'pano_y', 'run_width',
                    'run_height', 'label_width', 'label_height', 'reason',
                    'nearest_px', 'nearest_confidence', 'nearest_at_tier_px']


def verdict(n_matched, n_joinable):
    """(verdict, share): 'PASS' iff n_matched / n_joinable >= PASS_SHARE, else 'STOP';
    'UNDETERMINED' (share None) with nothing to join. Pre-registered for #56.

    Example:
        >>> verdict(98, 100)
        ('PASS', 0.98)
        >>> verdict(97, 100)
        ('STOP', 0.97)
    """
    if n_joinable == 0:
        return 'UNDETERMINED', None
    share = n_matched / n_joinable
    return ('PASS' if share >= PASS_SHARE else 'STOP'), share


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


# ------------------------------------------------------------------------------- join

def _dx(a, b, w):
    d = abs(a - b) % w
    return min(d, w - d)


def _dist(label, det, w):
    return math.hypot(_dx(label[0], det[0], w), label[1] - det[1])


def join(labels, run, selected=None, skips=None):
    """Score every AI label against the run. Returns a dict of tallies and rows; see the
    module docstring for what each diagnostic means."""
    skips = skips or {}
    out = {'joinable': 0, 'matched': 0, 'exact': 0, 'sub_tier': 0, 'dims_differ': 0,
           'not_in_run': Counter(), 'unmatched': [], 'buckets': Counter(),
           'panos': Counter(), 'unlabeled_dets': {'labeled_panos': 0, 'unlabeled_panos': 0}}
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
        cell = w / HEATMAP_WIDTH
        n_ok = 0
        for lab in labs:
            out['joinable'] += 1
            pt = (lab['pano_x'], lab['pano_y'])
            dims_differ = (lab.get('pano_width') not in (None, w)
                           or lab.get('pano_height') not in (None, h))
            out['dims_differ'] += dims_differ
            hit = None
            for i, d in enumerate(dets):
                if d[2] >= TIER and _dx(pt[0], d[0], w) <= TOLERANCE_PX \
                        and abs(pt[1] - d[1]) <= TOLERANCE_PX:
                    if hit is None or _dist(pt, d, w) < _dist(pt, dets[hit], w):
                        hit = i
            if hit is not None:
                n_ok += 1
                out['matched'] += 1
                out['exact'] += dets[hit][:2] == pt
                used[pid].add(hit)
                continue
            near = min(dets, key=lambda d: _dist(pt, d, w), default=None)
            near_tier = min((d for d in dets if d[2] >= TIER), key=lambda d: _dist(pt, d, w),
                            default=None)
            dist = None if near is None else _dist(pt, near, w)
            # The nearest detection is within tolerance, so (no tier detection being) it
            # must sit below the tier: the label's peak is there, its confidence is not.
            sub_tier = (near is not None and _dx(pt[0], near[0], w) <= TOLERANCE_PX
                        and abs(pt[1] - near[1]) <= TOLERANCE_PX)
            out['sub_tier'] += sub_tier
            if dist is None:
                out['buckets']['no detection on the pano'] += 1
            else:
                for bound, name in DISTANCE_BUCKETS:
                    limit = cell if bound == 'cell' else 2 * cell if bound == '2cell' else bound
                    if dist <= limit:
                        out['buckets'][name] += 1
                        break
            reason = ('dims differ' if dims_differ else
                      'no detection on the pano' if near is None else
                      'below tier at the pixel' if sub_tier else
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

    for pid, (w, h, dets) in run.items():
        n = sum(1 for i, d in enumerate(dets) if d[2] >= TIER and i not in used[pid])
        out['unlabeled_dets']['labeled_panos' if pid in by_pano else 'unlabeled_panos'] += n
    return out


# ------------------------------------------------------------------------------ report

def pull_record(path):
    """One report line for the label pull: url, fetch time, sha256, feature count."""
    side = path.parent / (path.name + '.source.json')
    meta = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    n = len(json.loads(path.read_text(encoding='utf-8'))['features'])
    return (f"`{path.name}`: {n:,} features, sha256 `{digest}`, fetched "
            f"{meta.get('fetched_at', '(unrecorded)')} from {meta.get('url', '(supplied on the command line)')}")


def render(city, res, ai_user, users, pull_line, results_path, n_run):
    v, share = verdict(res['matched'], res['joinable'])
    pct = lambda n, d: f'{n:,} ({100 * n / d:.2f}%)' if d else f'{n:,}'  # noqa: E731
    lines = [
        f'# {city}: provenance gate (#56)', '',
        f'**Verdict: {v}**' + ('' if share is None else
                                f' -- {res["matched"]:,} of {res["joinable"]:,} joinable AI labels '
                                f'match ({share:.4f}; PASS needs >= {PASS_SHARE}).'), '',
        f'Rule (pre-registered): a label matches when a stored detection with confidence >= '
        f'{TIER} lies within +/-{TOLERANCE_PX} px of it on each axis of its pano '
        f'(key `round(x*W), round(y*H)`, x wrapping at the seam); PASS iff >= {PASS_SHARE} of '
        f'labels whose pano is in the run match. STOP means nothing downstream runs.', '',
        '## Inputs', '',
        f'- Labels: {pull_line}',
        f'- AI account: `{ai_user}` ({users[ai_user]:,} CurbRamp labels); every account: ' +
        ', '.join(f'`{u}` {c:,}' for u, c in users.most_common()),
        f'- Run: `{results_path.as_posix()}`, {n_run:,} panos, sha256 '
        f'`{hashlib.sha256(results_path.read_bytes()).hexdigest()}`',
        f'- Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}', '',
        '## Join', '',
        f'- Joinable labels (pano in the run): {res["joinable"]:,}',
        f'- Matched within +/-{TOLERANCE_PX} px at >= {TIER}: {pct(res["matched"], res["joinable"])}, '
        f'of which exactly on the pixel: {res["exact"]:,}',
        f'- Unmatched: {len(res["unmatched"]):,}; of those, a stored detection BELOW {TIER} sits '
        f'within tolerance (a threshold flip): {res["sub_tier"]:,}',
        f'- Labels whose pano width/height differ from the run\'s: {res["dims_differ"]:,}', '',
        '| panos with AI labels | count |', '|---|---:|',
        *[f'| {k} | {res["panos"][k]:,} |' for k in ('all matched', 'partly matched', 'none matched')],
        '', '### Unmatched labels: nearest stored detection (any confidence)', '',
        '| distance | labels |', '|---|---:|',
        *[f'| {name} | {res["buckets"][name]:,} |'
          for name in [n for _, n in DISTANCE_BUCKETS] + ['no detection on the pano']],
        '', f'A heatmap cell is W/{HEATMAP_WIDTH} px (16 px on a 16384-wide pano): a label one '
        f'cell off points at resampling, not at a different model or peak rule.', '',
        f'## Detections >= {TIER} with no label', '',
        f'- on panos carrying AI labels: {res["unlabeled_dets"]["labeled_panos"]:,}',
        f'- on panos carrying none (e.g. the sampled empty stratum): '
        f'{res["unlabeled_dets"]["unlabeled_panos"]:,}', '',
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
    args = ap.parse_args(argv)

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
    selected, skips = load_selection(run_dir)
    res = join(labels, run, selected, skips)
    report = render(args.city, res, ai_user, users, pull_record(labels_path), results_path, len(run))
    (out / 'report.md').write_text(report, encoding='utf-8')
    write_unmatched(out / 'unmatched.csv', args.city, res['unmatched'])
    v, share = verdict(res['matched'], res['joinable'])
    print(f'{args.city}: {v}' + ('' if share is None else f' ({share:.4f} of '
          f'{res["joinable"]:,} joinable labels match)') + f' -> {out / "report.md"}')
    return {'PASS': 0, 'STOP': 1}.get(v, 2)


if __name__ == '__main__':
    sys.exit(main())
