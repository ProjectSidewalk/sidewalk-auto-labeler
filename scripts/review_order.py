"""Review order: the AI labels live on one server, ordered for human review by the validator's score.

Issue #168 item 5 ("review-order export, no gate"). sidewalk-validator-ai scores every live
AI label with its CurbRamp validator (p_correct, p_incorrect). This script turns those scores
into ONE ordering of the server's AI labels for a person to review, most-likely-wrong first.
It is an ordering only: nothing is withheld, deleted or sent anywhere, and no submission
record is written (where a pointer to the list should live is decision D1 in the PR).

THE UNIT IS THE SERVER LABEL, PER ENDPOINT. A city's records can describe the same live
labels more than once: Richmond's results.band.jsonl record copies the 0.55 core that
results.jsonl's record also holds, and results.posfix3seq.raw.jsonl resent 72 panos whose
old labels were soft-deleted (position_check.live_positions). A list per results file would
list the core twice and put soft-deleted labels in the queue under the new label's id. So:
  - one list per endpoint, over every campaign of every run dir given (sibling run dirs such
    as runs/juneau + runs/juneau_expansion share an endpoint);
  - one row per AI-authored row of the ranker's labels.csv whose (label type, pano_id,
    pano_x, pano_y) a campaign on that endpoint shipped. `campaigns` lists every campaign
    that shipped the key; `campaign` is the one that CREATED the server label (attribute():
    the earliest that shipped the key and was recorded no earlier than the label's
    time_created, less CLOCK_SLACK_S; `attribution` says how it was decided) and
    `confidence` the detector confidence that campaign sent;
  - shipped keys with no server row are NOT queue rows (nothing to review on the server):
    they are counted per campaign and listed in not_on_server.csv (soft-deleted,
    overwritten, or sent after the rawLabels pull);
  - ranker rows that join no shipped key are listed in unmatched.csv (a label sent from a
    file in another run dir, the wrong --city, or a human label at an AI pixel that the
    author filter did not catch).

INPUT CONTRACT (--ranker-run RUN; the only input mode). RUN is a validator-ai
`analysis/ai_labels_eval.py` run made with `select --all-ai`, then `crops`, then `score`:
  labels.csv                    label_uid, city, label_id, pano_id, author, pano_x, pano_y, ...
  selection.json                all_ai, labels_csv_sha256, label_type, selected_on, cities
  manifest.csv                  label_uid, pano_id, status (the crops that were cut)
  predictions.csv               label_uid, p_correct, p_incorrect [, rank_score]
  predictions_provenance.json   manifest_sha256, n, model_sha256, checkpoint, scored
  rawlabels/cities.json         optional: fetch_all.py's listing, each city's pull url
Refused (exit 2) unless all of these hold: selection.json says all_ai (without --all-ai,
labels.csv holds only the voted labels, a few percent of a city); labels.csv hashes to
selection.json's labels_csv_sha256; manifest.csv hashes to the provenance's
manifest_sha256; predictions.csv has the provenance's `n` rows; every predictions label_uid
is in manifest.csv and in labels.csv, and the manifest's pano_id for it equals labels.csv's;
label_uid is unique and equals `city:label_id`; pano_x/pano_y are integral; an input
rank_score is finite. The server the rawLabels pull came from must be the --endpoint's host
(labels.csv does not say, and a city's test and prod servers share every pixel key, so a
prod pull run against the test endpoint would otherwise match ~100%): it is read from
rawlabels/cities.json when the run has one, else --ranker-server must name it, and when both
are present they must agree. --city (the validator's city id, e.g. richmond-va) is required
when labels.csv holds more than one city. Rows whose `author` is not `ai` are dropped before
the join. A labels row with no prediction is listed after every ranked row, with rank_status
`crop_failed` (manifest status != ok), `not_cropped` (no manifest row: crops stopped early
or ran with --limit) or `not_scored`.

JOIN KEY: (label type, pano_id as a string, pano_x, pano_y), the pixel the labeler sent
(detectors.ps_pixel, via send_to_ps.transform_record). It is the same on every server a file
went to. `label_uid` (= city:label_id) is the row's identity and is never used to join: a
label_id is per city AND per server, and changes on every overwrite (SidewalkWebpage#5707).
The label type of a ranker row is selection.json's `label_type` (validator-ai ranks
CurbRamp only today); shipped keys of any other type are counted as `other_type` per
campaign, never as "not on server".

ORDER. Ranked rows by `rank_score` descending (ties by label_uid), then the unscored rows
by detector confidence ascending (ties by pano_id, pano_x, pano_y, label_uid);
`review_rank` = 1..N. `rank_score` is the input's `rank_score` column when predictions.csv
or labels.csv carries one, else validator-ai's formula, computed here:
log(p_incorrect) - log(p_correct), both clipped to [1e-45, 1] (ai_labels_eval.rank_score;
float32 p_incorrect saturates at 1.0, the log-odds keeps those apart). summary.json records
which (`rank_score_source`). The validator's own review packets order its `band` tier first
(validator-ai #80); this list does not (decision D2).

REPLAY. Every campaign on the endpoint is replayed exactly as it was sent
(coverage_check.replay_campaigns: base entries, bands, per-type campaigns of a multi-type
file, partial campaigns over their sidecar's lines), and the replay is checked against the
records' label counts. Refused (exit 2): a record problem (position_check.campaigns_for), a
replay that does not add up, a results file that no longer hashes to its record's sha256,
or zero list rows (wrong city, wrong server, or a stale pull).

OUTPUTS (default <first run dir>/review_order/<endpoint host>/):
  review_list.csv     the queue (local; per-label server ids, re-made on every run)
  not_on_server.csv   shipped keys with no server row (local; only when non-empty)
  unmatched.csv       ranker rows that join no shipped key (local; only when non-empty)
  summary.json        provenance, per-campaign counts, the records' state when ranked (tracked)
  report.md           the same, readable (tracked)
A run that exits 2 writes nothing. Warnings (exit 0): fewer than MIN_MATCH_WARN of the
ranker type's shipped keys on the server; fewer than MIN_RANKED_WARN of the list rows with
a prediction (a partial crops/score run); list rows created after every campaign that
shipped their pixel; a campaign sent after the ranker run was selected (validator-ai's
selected_on is when `select` ran, so the pull itself is no later than that).

STALENESS. summary.json keeps each record's sha256 and every campaign's submitted_lines,
labels_submitted and last_submission_utc for the endpoint, and the list's sha256. `--check`
compares them with the records, the results files and the list now: exit 0 fresh, 1 stale
(a send, a band, a resumed partial campaign, a gap-fill append, an edited list), 2 could
not tell. A list file that is absent (it is gitignored) is said, not counted as stale. An
overwrite run on the server changes no local file, so re-run after every campaign AND
every overwrite.

Usage:
    python scripts/review_order.py runs/richmond \\
        --endpoint https://sidewalk-richmond.cs.washington.edu/ai/submitLabelsOnPano \\
        --ranker-run ~/Data/ai-labels-2026-11 --ranker-server sidewalk-richmond.cs.washington.edu \\
        --city richmond-va
    # sibling run dirs on one endpoint
    python scripts/review_order.py runs/juneau runs/juneau_expansion --endpoint <prod url> \\
        --ranker-run RUN --ranker-server <prod host> --city juneau-ak
    # is the list still current?
    python scripts/review_order.py runs/richmond --endpoint <url> --check
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
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
# Root first, forced: scripts/position_check.py is a shim with the root module's name (see
# coverage_check.py, which does the same).
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = str(REPO_ROOT / 'scripts')
if SCRIPTS not in sys.path:
    sys.path.append(SCRIPTS)

import coverage_check  # noqa: E402
import position_check  # noqa: E402
import send_to_ps  # noqa: E402

RANKER_FILES = ('labels.csv', 'selection.json', 'manifest.csv', 'predictions.csv',
                'predictions_provenance.json')
LABEL_COLUMNS = ('label_uid', 'city', 'label_id', 'pano_id', 'author', 'pano_x', 'pano_y')
PREDICTION_COLUMNS = ('label_uid', 'p_correct', 'p_incorrect')
MANIFEST_COLUMNS = ('label_uid', 'pano_id')
PASS_THROUGH = ('tier', 'n_votes', 'clean')     # labels.csv columns copied to the list when present
DEFAULT_LABEL_TYPE = 'CurbRamp'
RANK_CLIP = 1e-45                               # ai_labels_eval.rank_score's clip floor
MIN_MATCH_WARN = 0.90                           # a guess, not derived from data (decision D5)
MIN_RANKED_WARN = 0.90                          # share of list rows with a prediction; a guess too
CLOCK_SLACK_S = 120     # server time_created vs the record's last_submission_utc (whole seconds,
#                         written at the end of a send by another machine's clock)
ORDER = 'score'
UNSCORED = ('crop_failed', 'not_cropped', 'not_scored')   # rank_status of a row with no prediction

LIST_FILE, NOT_ON_SERVER_FILE, UNMATCHED_FILE = 'review_list.csv', 'not_on_server.csv', 'unmatched.csv'
SUMMARY_FILE, REPORT_FILE = 'summary.json', 'report.md'
LIST_COLUMNS = ['review_rank', 'rank_status', 'label_uid', 'city', 'label_id', 'label_type',
                'time_created', 'campaign', 'attribution', 'campaigns', 'pano_id', 'pano_x',
                'pano_y', 'confidence',
                'rank_score', 'p_correct', 'p_incorrect', 'verdict', 'key_rows']
NOT_ON_SERVER_COLUMNS = ['label_type', 'pano_id', 'pano_x', 'pano_y', 'confidence', 'campaign',
                         'campaigns']
UNMATCHED_COLUMNS = ['label_uid', 'city', 'label_id', 'label_type', 'pano_id', 'pano_x', 'pano_y',
                     'rank_status', 'rank_score', 'p_correct', 'p_incorrect']
EXIT_OK, EXIT_STALE, EXIT_UNDETERMINED = 0, 1, 2


class UsageError(Exception):
    """Bad arguments or inputs that cannot be trusted; exit 2, nothing written."""


# ----------------------------------------------------------------------------------- ranking

def rank_score(p_correct, p_incorrect):
    """validator-ai's ranking score for "incorrect": log(p_incorrect) - log(p_correct), both
    clipped to [1e-45, 1] (analysis/ai_labels_eval.py rank_score, validator-ai 750b071).

    float32 p_incorrect saturates at exactly 1.0 for the most confident Disagrees, while
    p_correct still has resolution there, so the log-odds keeps their order.

    >>> [round(rank_score(c, i), 1) for c, i in [(0.5, 0.5), (1e-6, 1.0), (1e-3, 1.0)]]
    [0.0, 13.8, 6.9]
    """
    pc = min(max(float(p_correct), RANK_CLIP), 1.0)
    pi = min(max(float(p_incorrect), RANK_CLIP), 1.0)
    return math.log(pi) - math.log(pc)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path, required):
    """(rows, header) of a CSV; UsageError naming the first missing required column."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in required if c not in header]
        if missing:
            raise UsageError(f'{path.name} has no {missing[0]!r} column (columns: {", ".join(header)})')
        return list(reader), header


def _pixel(value, what):
    """An integral pixel from a CSV cell: '1234' and '1234.0' read 1234; '1234.5' refuses."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise UsageError(f'{what} is not a number ({value!r})') from None
    if not math.isfinite(f) or f != int(f):
        raise UsageError(f'{what} is not an integral pixel ({value!r}); the join is on the exact '
                         f'pixel the labeler sent')
    return int(f)


def _probability(value, what):
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise UsageError(f'{what} is not a number ({value!r})') from None
    if not (math.isfinite(f) and 0.0 <= f <= 1.0):
        raise UsageError(f'{what} is not a probability ({value!r})')
    return f


def _unique(rows, name):
    seen = Counter(r['label_uid'] for r in rows)
    dup = sorted(u for u, k in seen.items() if k > 1)
    if dup:
        raise UsageError(f'{name}: label_uid is not unique ({len(dup)} repeated, first {dup[0]!r})')


def read_ranker_run(run, city=None):
    """Read and check one validator-ai ranker run: {rows, label_type, city, provenance}.

    `rows` are the AI-authored labels.csv rows of `city`, each {label_uid, city, label_id,
    label_type, pano_id (str), time_created, pano_x, pano_y (int), rank_status ('ranked' or
    one of UNSCORED), rank_score (float or None), p_correct, p_incorrect (the input text),
    + PASS_THROUGH columns}. UsageError on any broken link in the chain the module docstring
    lists.

    Usage (illustrative path):
        r = read_ranker_run(Path('~/Data/ai-labels-2026-11').expanduser(), 'richmond-va')
        r['rows'][0]['label_uid']                 # 'richmond-va:<label_id>'
        r['provenance']['rank_score_source']      # 'computed' unless the CSV carries rank_score
    """
    run = Path(run)
    if not run.is_dir():
        raise UsageError(f'--ranker-run {run} is not a directory')
    missing = [n for n in RANKER_FILES if not (run / n).is_file()]
    if missing:
        raise UsageError(f'--ranker-run {run} has no {", ".join(missing)} (a select --all-ai, '
                         f'crops, score run of validator-ai analysis/ai_labels_eval.py is needed)')
    try:
        selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
        provenance = json.loads((run / 'predictions_provenance.json').read_text(encoding='utf-8'))
    except ValueError as exc:
        raise UsageError(f'--ranker-run: unreadable JSON ({exc})') from None
    if selection.get('all_ai') is not True:
        raise UsageError('selection.json says all_ai is not true: without `select --all-ai`, '
                         'labels.csv holds only the human-voted labels, not every live AI label')
    labels_sha = _sha256(run / 'labels.csv')
    if labels_sha != selection.get('labels_csv_sha256'):
        raise UsageError('labels.csv does not hash to selection.json labels_csv_sha256: it is not '
                         'the file that selection describes')
    manifest_sha = _sha256(run / 'manifest.csv')
    if manifest_sha != provenance.get('manifest_sha256'):
        raise UsageError('manifest.csv does not hash to predictions_provenance.json '
                         'manifest_sha256: the predictions were scored on another crop set')

    labels, label_header = _read_csv(run / 'labels.csv', LABEL_COLUMNS)
    predictions, pred_header = _read_csv(run / 'predictions.csv', PREDICTION_COLUMNS)
    manifest, _ = _read_csv(run / 'manifest.csv', MANIFEST_COLUMNS)
    if provenance.get('n') is not None and int(provenance['n']) != len(predictions):
        raise UsageError(f"predictions.csv has {len(predictions):,} rows and "
                         f"predictions_provenance.json says {provenance['n']}: a truncated or "
                         f"swapped predictions file")
    _unique(labels, 'labels.csv')
    _unique(predictions, 'predictions.csv')
    _unique(manifest, 'manifest.csv')
    for r in labels:
        if r['label_uid'] != f"{r['city']}:{r['label_id']}":
            raise UsageError(f"labels.csv: label_uid {r['label_uid']!r} is not city:label_id "
                             f"({r['city']}:{r['label_id']})")

    label_pano = {r['label_uid']: r['pano_id'] for r in labels}
    manifest_pano = {r['label_uid']: r['pano_id'] for r in manifest}
    manifest_status = {r['label_uid']: r.get('status', 'ok') for r in manifest}
    for p in predictions:
        uid = p['label_uid']
        if uid not in manifest_pano:
            raise UsageError(f'predictions.csv: {uid} is not in manifest.csv')
        if uid not in label_pano:
            raise UsageError(f'predictions.csv: {uid} is not in labels.csv')
        if manifest_pano[uid] != label_pano[uid]:
            raise UsageError(f'{uid}: manifest.csv pano_id {manifest_pano[uid]!r} is not labels.csv '
                             f'pano_id {label_pano[uid]!r}')

    cities = sorted({r['city'] for r in labels})
    if city is None:
        if len(cities) != 1:
            raise UsageError(f'labels.csv holds {len(cities)} cities ({", ".join(cities)}): name one '
                             f'with --city (a label_id is per city, never a key on its own)')
        city = cities[0]
    elif city not in cities:
        raise UsageError(f'--city {city} is not in labels.csv (cities: {", ".join(cities) or "none"})')

    label_type = selection.get('label_type') or DEFAULT_LABEL_TYPE
    score_column = 'rank_score' in pred_header or 'rank_score' in label_header
    by_uid = {p['label_uid']: p for p in predictions}
    rows, humans = [], 0
    for r in labels:
        if r['city'] != city:
            continue
        if r['author'] != 'ai':
            humans += 1
            continue
        uid = r['label_uid']
        # No prediction: the crop failed (manifest status != ok), was never attempted (no
        # manifest row: crops stopped early, --limit), or was cut but not scored.
        status = ('not_cropped' if uid not in manifest_status else
                  'crop_failed' if manifest_status[uid] != 'ok' else 'not_scored')
        row = {'label_uid': uid, 'city': city, 'label_id': r['label_id'], 'label_type': label_type,
               'pano_id': str(r['pano_id']), 'time_created': r.get('time_created') or '',
               'pano_x': _pixel(r['pano_x'], f'labels.csv {uid} pano_x'),
               'pano_y': _pixel(r['pano_y'], f'labels.csv {uid} pano_y'),
               'rank_status': status, 'rank_score': None, 'rank_score_text': '',
               'p_correct': '', 'p_incorrect': '', 'verdict': ''}
        row.update({c: r[c] for c in PASS_THROUGH if c in label_header})
        p = by_uid.get(uid)
        if p is not None:
            pc = _probability(p['p_correct'], f'predictions.csv {uid} p_correct')
            pi = _probability(p['p_incorrect'], f'predictions.csv {uid} p_incorrect')
            if score_column:
                text = p.get('rank_score') or r.get('rank_score') or ''
                try:
                    score = float(text)
                except ValueError:
                    raise UsageError(f'{uid}: rank_score {text!r} is not a number') from None
                if not math.isfinite(score):
                    raise UsageError(f'{uid}: rank_score {text!r} is not finite')
            else:
                score = rank_score(pc, pi)
                text = repr(score)
            row.update(rank_status='ranked', rank_score=score, rank_score_text=text,
                       p_correct=p['p_correct'], p_incorrect=p['p_incorrect'],
                       verdict='incorrect' if pi > pc else 'correct')
        rows.append(row)

    created = [t for t in (_utc(r.get('time_created')) for r in labels if r['city'] == city) if t]
    checkpoint = provenance.get('checkpoint')
    if checkpoint:   # validator-ai writes an absolute path on the scoring machine; keep the name
        checkpoint = str(checkpoint).replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
    statuses = Counter(r['rank_status'] for r in rows)
    return {'rows': rows, 'label_type': label_type, 'city': city, 'provenance': {
        'run': run.name, 'city': city, 'label_type': label_type,
        'selected_on': selection.get('selected_on'), 'all_ai': True,
        'newest_label_created': max(created).isoformat(timespec='seconds') if created else None,
        **pull_listing(run, city),
        'labels_sha256': labels_sha, 'manifest_sha256': manifest_sha,
        'predictions_sha256': _sha256(run / 'predictions.csv'),
        'model_sha256': provenance.get('model_sha256'), 'checkpoint': checkpoint,
        'scored': provenance.get('scored'),
        'rank_score_source': 'input' if score_column else 'computed',
        'ai_rows': len(rows), 'human_rows_dropped': humans,
        'ranked_rows': statuses.get('ranked', 0), **{s: statuses.get(s, 0) for s in UNSCORED}}}


def pull_listing(run, city):
    """{server, server_from, pull_fetched_on} from `<run>/rawlabels/cities.json`, which
    validator-ai's fetch_all.py writes beside the pulls: each city's `url` is the server its
    rawLabels CSV was fetched from, so it is evidence of the pull's server rather than an
    assertion. All None when the listing is absent or does not name `city`."""
    path = Path(run) / 'rawlabels' / 'cities.json'
    none = {'server': None, 'server_from': None, 'pull_fetched_on': None}
    if not path.is_file():
        return none
    try:
        listing = json.loads(path.read_text(encoding='utf-8'))
        hit = [c for c in listing.get('cities') or [] if c.get('city_id') == city]
    except (ValueError, AttributeError, TypeError) as exc:
        raise UsageError(f'rawlabels/cities.json is unreadable ({exc})') from None
    host = urlparse(str(hit[0].get('url') or '')).netloc.lower() if hit else ''
    if not host:
        return none
    return {'server': host, 'server_from': 'rawlabels/cities.json',
            'pull_fetched_on': listing.get('fetched_on')}


# ------------------------------------------------------------------------------- shipped set

def shipped_labels(run_dirs, endpoint):
    """Every label the campaigns on `endpoint` shipped, over all `run_dirs`:
    ({(label_type, pano_id, pano_x, pano_y): {campaign: {at, confidence, labels}}},
    [campaign summaries], [problems]). Campaign names are prefixed with the run dir's name
    (`richmond/results.band.jsonl (band 0.3-0.55)`). A results file that no longer hashes to
    its record's sha256 is a problem: the replay would not describe what was sent."""
    shipped = defaultdict(dict)
    summaries, problems = [], []
    for run_dir in run_dirs:
        def visit(c, _n, record, label_type, sent, _d=run_dir.name):
            name = f"{_d}/{c['campaign']}"
            pid = str(record['pano']['panorama_id'])
            for s in sent:
                hit = shipped[(label_type, pid, int(s['pano_x']), int(s['pano_y']))].setdefault(
                    name, {'at': c['at'], 'confidence': s['confidence'], 'labels': 0})
                hit['labels'] += 1
                hit['confidence'] = max(hit['confidence'], s['confidence'])

        try:
            parts, found = coverage_check.replay_campaigns(run_dir, endpoint, visit)
        except ValueError as exc:      # an unreadable record
            problems.append(str(exc))
            continue
        problems += found
        checked = set()
        for s in parts:
            s['campaign'] = f"{run_dir.name}/{s['campaign']}"
            summaries.append(s)
            if s['record'] in checked:
                continue
            checked.add(s['record'])
            record = send_to_ps.load_submission_record(run_dir / s['record'])
            if send_to_ps.hash_and_count(Path(s['path']))[0] != record.get('sha256'):
                problems.append(f"{run_dir.name}/{s['record']}: {Path(s['path']).name} no longer "
                                f"hashes to the record's sha256, so its replay may not be what was sent")
    return dict(shipped), summaries, problems


def record_snapshot(run_dirs, endpoint):
    """What the records say about `endpoint` now, for the staleness check:
    {'<run dir>/<record>': {sha256, submitted_lines, labels_submitted, last_submission_utc,
    [bands|types: {key: {submitted_lines, labels_submitted, last_submission_utc}}]}}."""
    def counts(state):
        return {k: state.get(k) for k in ('submitted_lines', 'labels_submitted', 'last_submission_utc')}

    snap = {}
    for run_dir in run_dirs:
        for rec_path in sorted(Path(run_dir).glob('*.submission.json')):
            record = send_to_ps.load_submission_record(rec_path)
            state = record['endpoints'].get(endpoint) or {}
            if int(state.get('submitted_lines') or 0) <= 0:
                continue
            entry = {'sha256': record.get('sha256'), **counts(state)}
            for part in ('bands', 'types'):
                if state.get(part):
                    entry[part] = {k: counts(v) for k, v in sorted(state[part].items())}
            snap[f'{Path(run_dir).name}/{rec_path.name}'] = entry
    return snap


def stale_reasons(then, now):
    """Why a summary's record snapshot (`then`) no longer describes the records (`now`)."""
    reasons = []
    for key in sorted(set(then) | set(now)):
        if key not in now:
            reasons.append(f'{key}: no longer holds a campaign on this endpoint')
        elif key not in then:
            reasons.append(f'{key}: a campaign on this endpoint the list does not cover')
        elif then[key] != now[key]:
            changed = sorted(k for k in set(then[key]) | set(now[key])
                             if then[key].get(k) != now[key].get(k))
            reasons.append(f'{key}: {", ".join(changed)} changed since the list was made')
    return reasons


def _utc(text):
    try:
        t = datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------------------- the list

def build_list(ranked, shipped, ranker_type):
    """Join ranker rows to shipped keys and order them: (list rows, not_on_server rows,
    unmatched rows, per-campaign counts {campaign: {...}}).

    Example (one shipped key, one ranked row on it, one ranker row elsewhere):
        >>> shipped = {('CurbRamp', 'P', 10, 20): {'r/a.jsonl': {'at': 't', 'confidence': 0.9,
        ...                                                      'labels': 1}}}
        >>> rows = [{'label_uid': 'x:1', 'label_type': 'CurbRamp', 'pano_id': 'P', 'pano_x': 10,
        ...          'pano_y': 20, 'rank_status': 'ranked', 'rank_score': 2.0},
        ...         {'label_uid': 'x:2', 'label_type': 'CurbRamp', 'pano_id': 'Q', 'pano_x': 1,
        ...          'pano_y': 1, 'rank_status': 'ranked', 'rank_score': 5.0}]
        >>> listed, gone, unmatched, _ = build_list(rows, shipped, 'CurbRamp')
        >>> [r['label_uid'] for r in listed], [r['label_uid'] for r in unmatched], gone
        (['x:1'], ['x:2'], [])
    """
    by_key = defaultdict(list)
    for row in ranked:
        by_key[(row['label_type'], row['pano_id'], row['pano_x'], row['pano_y'])].append(row)

    listed, unmatched = [], []
    for key, rows in by_key.items():
        hit = shipped.get(key)
        for row in rows:
            if hit is None:
                unmatched.append(row)
                continue
            name, how = attribute(hit, row.get('time_created'))
            listed.append({**row, 'campaign': name, 'attribution': how,
                           'campaigns': ';'.join(sorted(hit)),
                           'confidence': hit[name]['confidence'], 'key_rows': len(rows)})
    ranked_rows = sorted((r for r in listed if r['rank_status'] == 'ranked'),
                         key=lambda r: (-r['rank_score'], r['label_uid']))
    unscored = sorted((r for r in listed if r['rank_status'] != 'ranked'),
                      key=lambda r: (r['confidence'], r['pano_id'], r['pano_x'], r['pano_y'],
                                     r['label_uid']))
    listed = ranked_rows + unscored
    for i, row in enumerate(listed, 1):
        row['review_rank'] = i
    unmatched.sort(key=lambda r: r['label_uid'])

    gone = []
    per = defaultdict(lambda: {'keys': 0, 'on_server': 0, 'not_on_server': 0, 'other_type': 0,
                               'list_rows_attributed': 0})
    for key, campaigns in shipped.items():
        on_server = key in by_key
        for name in campaigns:
            c = per[name]
            c['keys'] += 1
            if key[0] != ranker_type:
                c['other_type'] += 1
            elif on_server:
                c['on_server'] += 1
            else:
                c['not_on_server'] += 1
        if key[0] == ranker_type and not on_server:
            name = _newest(campaigns)   # no server label, so no time_created to go by
            gone.append({'label_type': key[0], 'pano_id': key[1], 'pano_x': key[2],
                         'pano_y': key[3], 'confidence': campaigns[name]['confidence'],
                         'campaign': name, 'campaigns': ';'.join(sorted(campaigns))})
    for row in listed:
        per[row['campaign']]['list_rows_attributed'] += 1
    gone.sort(key=lambda r: (r['pano_id'], r['pano_x'], r['pano_y']))
    return listed, gone, unmatched, dict(per)


def _newest(campaigns):
    return max(campaigns, key=lambda n: (campaigns[n]['at'] or '', n))


def attribute(campaigns, time_created):
    """(campaign, how): which of the campaigns that shipped a key created this server label.

    The EARLIEST campaign whose last_submission_utc is not before the label's time_created
    (less CLOCK_SLACK_S): a send stamps its record at the end, after every label it created.
    Not "newest": a band send moves the base entry's timestamp forward, and a derived record
    (reinfer.write_derived_record) carries a copy of an older campaign, so the newest record
    to list a key is often not the send that created its label (review F1). A campaign with
    no timestamp cannot be ruled out and sorts after the timed ones. `how` is
    'time_created', 'campaign_time_unknown' (only untimed campaigns fit), 'newest_no_time'
    (the label has no time_created) or 'newest_no_fit' (created after every campaign that
    shipped the key: the records are out of date, or the pull is from another server).

        >>> c = {'a': {'at': '2026-09-05T01:39:04Z'}, 'b': {'at': '2026-09-22T19:03:45Z'}}
        >>> attribute(c, '2026-09-05T01:20:00.000Z'), attribute(c, '2026-09-22T18:59:00Z')
        (('a', 'time_created'), ('b', 'time_created'))
        >>> attribute(c, '2026-10-01T00:00:00Z'), attribute(c, '')
        (('b', 'newest_no_fit'), ('b', 'newest_no_time'))
    """
    t = _utc(time_created) if time_created else None
    if t is None:
        return _newest(campaigns), 'newest_no_time'
    floor = t.timestamp() - CLOCK_SLACK_S
    fits = []
    for name, c in campaigns.items():
        at = _utc(c['at']) if c['at'] else None
        if at is None or at.timestamp() >= floor:
            fits.append((at is None, at.timestamp() if at else 0.0, name))
    if not fits:
        return _newest(campaigns), 'newest_no_fit'
    untimed, _, name = min(fits)
    return name, 'campaign_time_unknown' if untimed else 'time_created'


# ----------------------------------------------------------------------------------- outputs

def _csv_rows(rows, columns):
    out = []
    for r in rows:
        row = {c: r.get(c, '') for c in columns}
        if 'rank_score' in columns:
            row['rank_score'] = r.get('rank_score_text', '')
        out.append(row)
    return out


def render_report(s):
    """report.md from summary.json's dict."""
    c = s['counts']
    rk = {k: ('unknown' if v is None else v) for k, v in s['ranker'].items()}
    fetched = '' if rk['pull_fetched_on'] == 'unknown' else f", listing fetched {rk['pull_fetched_on']}"
    pulled = f"rawLabels from `{rk['server']}` (per {rk['server_from']}{fetched})"
    L = [f"# Review order: {s['endpoint_host']}", '',
         f"Made {s['generated_at']} by `scripts/review_order.py` (issue #168 item 5) for "
         f"`{s['endpoint']}`, over run dir(s) {', '.join(f'`{d}`' for d in s['run_dirs'])}. "
         f"An ordering for human review only: nothing was withheld, deleted or sent.", '',
         '## Ranker input', '',
         f"validator-ai run `{rk['run']}`, city `{rk['city']}`, label type `{rk['label_type']}`, "
         f"{pulled}; selected {rk['selected_on']} (the pull is no later; its newest label was "
         f"created {rk['newest_label_created']}); scored {rk['scored']}, checkpoint "
         f"`{rk['checkpoint']}`. labels.csv `{rk['labels_sha256']}`, predictions.csv "
         f"`{rk['predictions_sha256']}`, model `{rk['model_sha256']}`. rank_score: "
         f"{rk['rank_score_source']}. {rk['ai_rows']:,} AI rows: {rk['ranked_rows']:,} with a "
         f"prediction, {rk['crop_failed']:,} crop failed, {rk['not_cropped']:,} never cropped, "
         f"{rk['not_scored']:,} cut but not scored; {rk['human_rows_dropped']:,} human rows dropped.",
         '']
    if s['warnings']:
        L += ['## Warnings', ''] + [f'- {w}' for w in s['warnings']] + ['']
    how = ', '.join(f'{k} {v:,}' for k, v in c['attribution'].items())
    L += ['## The list', '',
          f"`{s['list']['file']}` (sha256 `{s['list']['sha256']}`, local, not tracked): "
          f"**{c['list_rows']:,}** rows, {c['ranked']:,} ranked by score, then "
          f"{c['unscored']:,} with no prediction (by detector confidence, ascending). "
          f"{c['rows_in_several_campaigns']:,} rows were shipped by more than one campaign; "
          f"{c['rows_sharing_a_key']:,} rows share their pixel with another server label. "
          f"Campaign attribution: {how}.", '',
          f"Shipped keys of type `{rk['label_type']}`: {c['shipped_keys']:,}, of which "
          f"{c['shipped_keys_on_server']:,} have a server row ({c['match_rate']:.1%}) and "
          f"{c['not_on_server']:,} do not (`{NOT_ON_SERVER_FILE}`). Ranker rows that join no "
          f"shipped key: {c['unmatched_rows']:,} (`{UNMATCHED_FILE}`). Shipped keys of types the "
          f"ranker does not cover: "
          f"{', '.join(f'{t} {n:,}' for t, n in c['other_type_keys'].items()) or 'none'}.", '',
          '## Campaigns', '',
          '| campaign | sent range | rig masked | labels (record) | labels (replayed) | keys | '
          'on server | not on server | other type | rows attributed |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in s['campaigns']:
        hi = '∞' if p['select_max'] is None else f"{p['select_max']:g}"
        rec = p['recorded_labels']
        L.append(f"| `{p['campaign']}`{' (partial)' if p['partial'] else ''} | "
                 f"[{p['select_min']:g}, {hi}) | {'yes' if p['rig_masked'] else 'no'} | "
                 f"{rec:,} | {p['replayed_labels']:,} | {p['keys']:,} | {p['on_server']:,} | "
                 f"{p['not_on_server']:,} | {p['other_type']:,} | {p['list_rows_attributed']:,} |"
                 if isinstance(rec, int) else
                 f"| `{p['campaign']}` | | | {rec} | {p['replayed_labels']:,} | | | | | |")
    L += ['', 'A base entry\'s recorded labels include its complete bands; the replayed column is '
          'per campaign. "Rows attributed" counts the list rows whose label that campaign created: '
          'the earliest campaign that shipped the key and was recorded no earlier than the label\'s '
          '`time_created`.', '',
          '## Reading this', '',
          '- Re-run after every campaign and after any overwrite run on the server: label ids change '
          '(SidewalkWebpage#5707), and `--check` cannot see an overwrite.',
          '- `not on server` keys were shipped but have no AI row in the ranker\'s rawLabels pull: '
          'soft-deleted, overwritten, or sent after the pull.', '']
    return '\n'.join(L)


def write_outputs(out, summary, listed, gone, unmatched, pass_through):
    out.mkdir(parents=True, exist_ok=True)
    columns = LIST_COLUMNS + list(pass_through)
    coverage_check.write_csv(out / LIST_FILE, _csv_rows(listed, columns), columns)
    coverage_check.write_csv(out / NOT_ON_SERVER_FILE, _csv_rows(gone, NOT_ON_SERVER_COLUMNS),
                             NOT_ON_SERVER_COLUMNS)
    coverage_check.write_csv(out / UNMATCHED_FILE, _csv_rows(unmatched, UNMATCHED_COLUMNS),
                             UNMATCHED_COLUMNS)
    summary['list'] = {'file': LIST_FILE, 'sha256': _sha256(out / LIST_FILE)}
    with open(out / SUMMARY_FILE, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    (out / REPORT_FILE).write_text(render_report(summary), encoding='utf-8', newline='\n')


# -------------------------------------------------------------------------------------- main

def _host(text):
    """'https://PS.example.org/' or 'ps.example.org' -> 'ps.example.org'."""
    text = (text or '').strip().lower()
    return urlparse(text).netloc or text.split('/')[0]


def _inputs(args):
    endpoint = send_to_ps.canonical_endpoint(args.endpoint)
    parts = urlparse(endpoint)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise UsageError(f'--endpoint must be an http(s) URL, got {args.endpoint!r}')
    run_dirs = []
    for d in args.run_dirs:
        p = Path(d)
        if not p.is_dir():
            raise UsageError(f'{d} is not a run directory')
        run_dirs.append(p.resolve())
    if len({p.name for p in run_dirs}) != len(run_dirs):
        raise UsageError('two run dirs share a name; campaign names would collide')
    # `host:port` is not a valid Windows folder name (a local dev server; review F6)
    folder = parts.netloc.replace(':', '_')
    out = Path(args.out) if args.out else run_dirs[0] / 'review_order' / folder
    return endpoint, parts.netloc, run_dirs, out


def results_hash_problems(run_dirs, endpoint):
    """Results files (of records holding campaigns on `endpoint`) that no longer hash to their
    record's sha256: a replay of them would not be what was sent."""
    out = []
    for run_dir in run_dirs:
        for rec_path in sorted(Path(run_dir).glob('*.submission.json')):
            record = send_to_ps.load_submission_record(rec_path)
            if int((record['endpoints'].get(endpoint) or {}).get('submitted_lines') or 0) <= 0:
                continue
            results = rec_path.with_name(record.get('input_file') or rec_path.name[:-len('.submission.json')])
            if results.exists() and send_to_ps.hash_and_count(results)[0] != record.get('sha256'):
                out.append(f'{Path(run_dir).name}/{results.name} no longer hashes to its record\'s '
                           f'sha256, so its replay may not be what was sent')
    return out


def check(endpoint, run_dirs, out):
    """--check: is out/summary.json still a description of the records, the results files and
    the list? 0 fresh / 1 stale / 2 cannot tell. The list CSV is gitignored, so on another
    machine it is absent: said, not counted as stale (the summary still holds)."""
    path = out / SUMMARY_FILE
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
        then = summary['records']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'review_order: no readable {path} ({exc})', file=sys.stderr)
        return EXIT_UNDETERMINED
    if summary.get('endpoint') != endpoint:
        print(f"review_order: {path} is for {summary.get('endpoint')}, not {endpoint}", file=sys.stderr)
        return EXIT_UNDETERMINED
    reasons = stale_reasons(then, record_snapshot(run_dirs, endpoint))
    reasons += results_hash_problems(run_dirs, endpoint)
    listing = out / (summary.get('list') or {}).get('file', LIST_FILE)
    note = ''
    if not listing.exists():
        note = f'; the list ({listing.name}) is not on this machine: re-run to rebuild it'
    elif _sha256(listing) != (summary.get('list') or {}).get('sha256'):
        reasons.append(f'{listing.name} no longer hashes to the sha256 summary.json recorded')
    if reasons:
        print(f'stale: {path}')
        for r in reasons:
            print(f'  {r}')
        return EXIT_STALE
    print(f"fresh: {path} (made {summary.get('generated_at')}; an overwrite run on the server "
          f"would not show here){note}")
    return EXIT_OK


def run(endpoint, host, run_dirs, out, args):
    ranker = read_ranker_run(args.ranker_run, args.city)
    prov = ranker['provenance']
    flag = _host(args.ranker_server) if args.ranker_server else None
    if prov['server'] and flag and flag != prov['server']:
        raise UsageError(f"--ranker-server {args.ranker_server!r} disagrees with the ranker run's "
                         f"rawlabels/cities.json, which says {prov['city']} was pulled from "
                         f"{prov['server']}")
    server = prov['server'] or flag
    if not server:
        raise UsageError('the ranker run has no rawlabels/cities.json naming the server its pull came '
                         'from: give --ranker-server')
    if server != host:
        raise UsageError(f'the ranker\'s rawLabels pull came from {server}, not the --endpoint host '
                         f'{host}: the pull must come from the server the list is for')
    prov.update(server=server, server_from=prov['server_from'] or '--ranker-server')
    shipped, campaigns, problems = shipped_labels(run_dirs, endpoint)
    if problems:
        for p in problems:
            print(f'problem: {p}', file=sys.stderr)
        print('review_order: exit 2, nothing written', file=sys.stderr)
        return EXIT_UNDETERMINED

    label_type = ranker['label_type']
    listed, gone, unmatched, per = build_list(ranker['rows'], shipped, label_type)
    if not listed:
        print(f"review_order: no ranker row joins a label shipped to {host} (wrong --city, wrong "
              f"server, or a pull older than every campaign); exit 2, nothing written", file=sys.stderr)
        return EXIT_UNDETERMINED

    typed = [k for k in shipped if k[0] == label_type]
    on_server = len(typed) - len(gone)
    other = Counter(k[0] for k in shipped if k[0] != label_type)
    match = on_server / len(typed) if typed else 0.0
    ranked = sum(r['rank_status'] == 'ranked' for r in listed)
    statuses = Counter(r['rank_status'] for r in listed if r['rank_status'] != 'ranked')
    attribution = Counter(r['attribution'] for r in listed)
    warnings = []
    if match < MIN_MATCH_WARN:
        warnings.append(f'only {on_server:,} of {len(typed):,} shipped {label_type} keys ({match:.1%}) '
                        f'have a server row in the ranker input, under {MIN_MATCH_WARN:.0%}: check '
                        f'--city, the server, and the pull date')
    if ranked / len(listed) < MIN_RANKED_WARN:
        warnings.append(f'only {ranked:,} of {len(listed):,} list rows ({ranked / len(listed):.1%}) have '
                        f'a prediction, under {MIN_RANKED_WARN:.0%} (' +
                        ', '.join(f'{s} {statuses.get(s, 0):,}' for s in UNSCORED) +
                        '): a partial crops or score run orders most of the list by detector '
                        'confidence instead')
    if attribution.get('newest_no_fit'):
        warnings.append(f"{attribution['newest_no_fit']:,} list row(s) were created after every "
                        f"campaign that shipped their pixel to this endpoint (attributed to the newest): "
                        f"records out of date, or a pull from another server")
    for t, n in sorted(other.items()):
        warnings.append(f'no ranker input covers type {t}: its {n:,} shipped key(s) are not in the list')
    selected = _utc(prov['selected_on'])
    for c in campaigns:
        at = _utc(c['at'])
        if selected and at and at > selected:
            warnings.append(f"{c['campaign']} was sent ({c['at']}) after the ranker run was selected "
                            f"({prov['selected_on']}); the rawLabels pull is no later than that, so "
                            f"labels this campaign created cannot be in the list")

    for c in campaigns:
        c.update(per.get(c['campaign'], {'keys': 0, 'on_server': 0, 'not_on_server': 0,
                                         'other_type': 0, 'list_rows_attributed': 0}))
    counts = {'list_rows': len(listed), 'ranked': ranked, 'unscored': len(listed) - ranked,
              'unscored_by_status': {s: statuses.get(s, 0) for s in UNSCORED},
              'unmatched_rows': len(unmatched), 'shipped_keys': len(typed),
              'shipped_keys_on_server': on_server, 'not_on_server': len(gone),
              'match_rate': round(match, 6), 'other_type_keys': dict(sorted(other.items())),
              'rows_in_several_campaigns': sum(';' in r['campaigns'] for r in listed),
              'rows_sharing_a_key': sum(r['key_rows'] > 1 for r in listed),
              'attribution': dict(sorted(attribution.items()))}
    pass_through = [c for c in PASS_THROUGH if listed and c in listed[0]]
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'tool': 'scripts/review_order.py', 'issue': '#168 item 5', 'order': ORDER,
        'endpoint': endpoint, 'endpoint_host': host,
        'run_dirs': [position_check.repo_relative(d) for d in run_dirs],
        'ranker': prov,
        'counts': counts, 'warnings': warnings,
        'campaigns': [{k: c[k] for k in ('campaign', 'record', 'band', 'label_type', 'select_min',
                                         'select_max', 'rig_masked', 'partial', 'lines',
                                         'recorded_labels', 'replayed_labels', 'at', 'keys',
                                         'on_server', 'not_on_server', 'other_type',
                                         'list_rows_attributed')} for c in campaigns],
        'records': record_snapshot(run_dirs, endpoint),
    }
    write_outputs(out, summary, listed, gone, unmatched, pass_through)

    print(f"{host}: {counts['list_rows']:,} list rows ({ranked:,} ranked, "
          f"{counts['unscored']:,} unscored); {on_server:,}/{len(typed):,} shipped "
          f"{label_type} keys on the server ({match:.1%}); {len(gone):,} not on server; "
          f"{len(unmatched):,} unmatched ranker rows; attribution " +
          ', '.join(f'{k} {v:,}' for k, v in counts['attribution'].items()))
    for c in campaigns:
        print(f"  {c['campaign']}: {c['list_rows_attributed']:,} rows attributed")
    for w in warnings:
        print(f'warning: {w}', file=sys.stderr)
    print(f'list: {out / LIST_FILE}\nreport: {out / REPORT_FILE}')
    return EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dirs', nargs='+', help='run dir(s) whose submission records hold campaigns '
                                                'on --endpoint (sibling dirs of one city allowed)')
    ap.add_argument('--endpoint', required=True,
                    help='the submit URL the records are keyed by, e.g. '
                         'https://sidewalk-richmond.cs.washington.edu/ai/submitLabelsOnPano')
    ap.add_argument('--ranker-run', help='a validator-ai ai_labels_eval.py run dir (select --all-ai, '
                                         'crops, score)')
    ap.add_argument('--ranker-server', help='host the ranker run\'s rawLabels pull came from; needed '
                                            'when the run has no rawlabels/cities.json, else must '
                                            'agree with it; must be the --endpoint host')
    ap.add_argument('--city', help='validator city id (e.g. richmond-va); required when labels.csv '
                                   'holds several cities')
    ap.add_argument('--out', help='output dir (default <first run dir>/review_order/<endpoint host>)')
    ap.add_argument('--check', action='store_true',
                    help='only report whether the existing summary.json is still current '
                         '(exit 0 fresh, 1 stale, 2 cannot tell)')
    args = ap.parse_args(argv)
    try:
        endpoint, host, run_dirs, out = _inputs(args)
        if args.check:
            return check(endpoint, run_dirs, out)
        if not args.ranker_run:
            raise UsageError('--ranker-run is required (unless --check)')
        return run(endpoint, host, run_dirs, out, args)
    except (UsageError, ValueError) as exc:   # ValueError: an unreadable submission record
        print(f'review_order: {exc}', file=sys.stderr)
    except OSError as exc:                    # e.g. an output folder that cannot be made
        print(f'review_order: {type(exc).__name__}: {exc}', file=sys.stderr)
    return EXIT_UNDETERMINED


if __name__ == '__main__':
    sys.exit(main())
