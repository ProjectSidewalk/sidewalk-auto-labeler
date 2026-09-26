"""Post-submission coverage check: does every pano we put labels on have its image on the server?

Issue #46 (re-scoped, parts 1 + 2). Project Sidewalk serves a Mapillary label's crop and
panorama from its own backup copy of the image; a label whose pano was never backed up
renders as nothing. This asks the server, read-only, whether each pano the labeler SENT
labels to is backed up, and lists the archived fallback files for any that are not.

How backup status is observable (all public GETs, nothing written anywhere):
  - /labels/all          one feature per label with `has_backup` and `ai_generated`, but
                         no pano id. `has_backup` is pano_data.has_backup with NULL read as
                         false; it is set true only by server code that opened the file (the
                         crop job, serving the backup image, the imagery check). So TRUE
                         means "a server process found the file" and FALSE means
                         "unconfirmed", never "confirmed absent".
  - /v3/api/rawLabels    label_id -> pano_id (+ camera_pitch), per label type. The join on
                         label_id gives {pano_id -> has_backup} for the live AI labels.
  - /adminapi/panos      every pano row with `has_labels` ("ever had a label", soft-deleted
                         included); it has no has_backup. It separates `retired` (labels
                         landed, then were soft-deleted) from `never_landed`.
  - /backupImage/<id>/metadata   200 iff the file is on disk AND the pano row has a
                         camera_pitch (and width/height/lat/lng/heading). A 404 on a pano
                         whose pitch is null is therefore UNCONFIRMED, not missing; only a
                         404 with a non-null pitch is decisive. Probed sequentially, with
                         spacing and backoff, only for panos the flag leaves unconfirmed;
                         redirects are not followed and a 200 must carry a JSON object.
  The aggregate on the admin health page (labeled / backed_up / no_backup / unchecked) is
  the human cross-check; it needs an admin session, so this script cannot read it.

The EXPECTED set comes from the submission records and their sidecars, never from
results.jsonl alone (the selection rule): every campaign every `*.submission.json` in the
run directory holds for this server (position_check.campaigns_for -- base entries and
bands, partial campaigns restricted to their sidecar's lines), each replayed through
send_to_ps.transform_record at exactly the range it sent ([select_min, select_max), nadir
mask iff the record's `rig_masked`). A pano is expected if any campaign sent it at least
one label. There is deliberately no --min-confidence: the record is the truth. The replay
is checked against the record: a band's recorded label count must equal its replay, and a
base entry's must equal its own replay plus the replays of the bands that cover the file
(send_to_ps.write_submission_record adds a band to the base total when it completes). A
mismatch means the inferred range or mask is wrong, and is a problem (exit 2).

Classification of each expected pano, in order:
  retired           no live AI label on the server, but the pano row has `has_labels` (it
                    once held one: e.g. Laurens' soft-deleted rig labels) -- reported,
                    never a gap: nothing renders for it
  never_landed      no live AI label AND the server has never held a label on it
                    (`has_labels` false, or no pano row): labels the records say were sent
                    never arrived -- a problem (exit 2), never folded into `retired`
  covered           some live AI label on it reads has_backup: true
  covered_metadata  /backupImage/<id>/metadata answered 200 with a JSON object
  missing           the metadata probe 404'd AND the pano has a camera_pitch (decisive)
  unconfirmed       everything else (404 with a null pitch, not probed, over --max-confirm)

Exit status: 0 no missing and no unconfirmed; 1 some missing (or, with --strict, some
unconfirmed); 2 could not determine (a record problem or replay mismatch, a failed or empty
pull, a failed probe, more panos to confirm than --max-confirm, a never_landed pano, a live
AI label rawLabels cannot place, an expected pano absent from the pano table, a usage error,
or any unexpected error). Exit 1 means only "missing", never "the check broke".

Outputs (default runs/<city>/coverage/): report.md, plus missing.csv (missing, unconfirmed
and never_landed rows) and fallback.csv (archived files that could be placed, with
--archive) when they are non-empty -- all git-tracked. The three API pulls are cached there,
with a <file>.source.json fetch record each, and are NOT tracked; the report carries each
pull's url, fetch time, sha256 and feature count. Every run pulls FRESH by default -- the
point of a re-run is usually that the scraper has had a night, and a silently reused pull
would report yesterday's state. The set is fetched in full before any file is replaced, so
a failed GET leaves the old cache whole. --reuse-pulls re-reads the cache instead (offline
re-runs of the report), and refuses a cache whose file no longer hashes to its fetch record
or whose pulls were fetched more than PULL_SET_MAX_SPREAD_S apart. A run that stops before
classifying still rewrites report.md as an exit-2 report (and removes the CSVs), so a stale
"Exit 0" never survives a failed check; a usage error stops before anything is written.

Usage:
    python scripts/coverage_check.py runs/richmond/results.jsonl \\
        --server https://sidewalk-richmond.cs.washington.edu
    python scripts/coverage_check.py runs/laurens/results.raw.jsonl \\
        --server https://sidewalk-laurens.cs.washington.edu --archive D:/archive/laurens/panos
    # regenerate the report offline from the cached pulls
    python scripts/coverage_check.py runs/richmond/results.jsonl \\
        --server https://sidewalk-richmond.cs.washington.edu --reuse-pulls

Out of scope: placing archived files into the server's store (SidewalkWebpage#4865), the
pano-only pose push that would make the metadata probe decisive (PR #74's manual step),
and GSV cities (their imagery is served by Google, not from a backup copy).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
# Root first, forced: scripts/position_check.py is a shim with the root module's name, and
# running this file puts scripts/ at sys.path[0], where the shim would shadow the module.
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import position_check  # noqa: E402
import send_to_ps  # noqa: E402

SUBMIT_PATH = '/ai/submitLabelsOnPano'
API_PANOS = '/adminapi/panos'
API_LABELS_ALL = '/labels/all'
API_RAW_LABELS = '/v3/api/rawLabels?labelType={label_type}&filetype=geojson'
API_METADATA = '/backupImage/{pano_id}/metadata'

DEFAULT_MAX_CONFIRM = 500      # above this many probes, stop and say so rather than fan out
PROBE_SPACING_S = 0.2          # between sequential metadata probes
PROBE_BACKOFF_S = (2.0, 8.0)   # waits before retry 1 and 2 on a 5xx, a 429 or a connection error
PROBE_RETRY_AFTER_CAP_S = 60.0  # most a 429's Retry-After is honoured for
PULL_SET_MAX_SPREAD_S = 600    # --reuse-pulls refuses a cached set fetched further apart than this
PULL_TIMEOUT_S = 300
PROBE_TIMEOUT_S = 30

STATUSES = ('covered', 'covered_metadata', 'missing', 'unconfirmed', 'retired', 'never_landed')
GAP_STATUSES = ('missing', 'unconfirmed', 'never_landed')   # the rows missing.csv lists
EXIT_OK, EXIT_GAPS, EXIT_UNDETERMINED = 0, 1, 2


class PullError(Exception):
    """A server pull that cannot be used (HTTP error, not JSON, zero rows, a cache that
    cannot be trusted)."""


class UsageError(Exception):
    """Bad arguments or inputs; reported before anything is written, exit 2."""


# ------------------------------------------------------------------------------ server I/O

def fetch_json(url):
    """GET `url` and return the raw body bytes. The single network entry point for pulls
    (tests monkeypatch it). Raises PullError on anything but a 200."""
    try:
        r = requests.get(url, timeout=PULL_TIMEOUT_S)
    except requests.RequestException as exc:
        raise PullError(f'{url}: {exc}') from exc
    if r.status_code != 200:
        raise PullError(f'{url}: HTTP {r.status_code}')
    return r.content


def count_rows(payload):
    """Rows in a pull: features of a FeatureCollection, or items of a JSON list."""
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get('features'), list):
        return len(payload['features'])
    raise PullError('neither a GeoJSON FeatureCollection nor a JSON list')


def source_record_path(dest):
    return dest.with_name(dest.name + '.source.json')


def _rows_of(url, body):
    """Row count of a pull body; PullError when it is not JSON or holds zero rows (a zero-row
    200 would otherwise read as "the city has no labels")."""
    try:
        n = count_rows(json.loads(body.decode('utf-8')))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PullError(f'{url}: not JSON ({exc})') from exc
    except PullError as exc:
        raise PullError(f'{url}: {exc}') from exc
    if n == 0:
        raise PullError(f'{url}: zero rows -- refusing to cache an empty pull')
    return n


def _cached(url, dest):
    """(fetch record, body) of a cached pull of `url` at `dest`, or None when there is none.
    PullError when the file no longer hashes to its recorded sha256 (edited or truncated
    since it was pulled): the report would otherwise print a hash it did not read."""
    side = source_record_path(dest)
    if not (dest.exists() and side.exists()):
        return None
    try:
        meta = json.loads(side.read_text(encoding='utf-8'))
    except ValueError:
        return None
    if not isinstance(meta, dict) or meta.get('url') != url:
        return None
    body = dest.read_bytes()
    if hashlib.sha256(body).hexdigest() != meta.get('sha256'):
        raise PullError(f'cached {dest.name} no longer matches the sha256 recorded when it was '
                        f'pulled ({meta.get("fetched_at")}); re-run without --reuse-pulls')
    return {**meta, 'reused': True}, body


def _fetch_times(metas):
    try:
        return [datetime.fromisoformat(m['fetched_at']) for m in metas]
    except (KeyError, TypeError, ValueError) as exc:
        raise PullError(f'a cached fetch record has no usable fetched_at ({exc}); re-run '
                        f'without --reuse-pulls') from exc


def pull_set(specs, out, reuse=False, max_spread_s=PULL_SET_MAX_SPREAD_S):
    """Pull every (name, url) of `specs` into `out`: {name: (fetch record, body bytes)}, a
    fetch record being {url, fetched_at, n_features, sha256, reused}.

    FRESH by default. Every body is fetched and validated before any cached file is
    touched, then each is written temp-then-rename with its `<name>.source.json`, so a GET
    that fails part-way leaves the previous cache whole rather than half old, half new.
    With `reuse`, a complete cache is read back instead -- refused (PullError) when a file
    no longer matches its recorded sha256 or the set was fetched more than `max_spread_s`
    apart (a mixed-age set would join the labels of one day to the backup flags of
    another); an incomplete cache is pulled fresh, in full."""
    if reuse:
        cached = [_cached(url, out / name) for name, url in specs]
        if all(c is not None for c in cached):
            times = _fetch_times([meta for meta, _ in cached])
            spread = (max(times) - min(times)).total_seconds()
            if spread > max_spread_s:
                raise PullError(f'the cached pulls were fetched {spread:,.0f} s apart (more than '
                                f'{max_spread_s:,} s): a mixed-age set; re-run without --reuse-pulls')
            now = datetime.now(timezone.utc)
            for (name, _), (meta, _), t in zip(specs, cached, times):
                print(f'reusing cached {name} (pulled {meta["fetched_at"]}, '
                      f'{(now - t).total_seconds() / 3600:.1f} h ago)', file=sys.stderr)
            return {name: c for (name, _), c in zip(specs, cached)}
        print('--reuse-pulls: the cache does not hold every pull; pulling the whole set fresh',
              file=sys.stderr)
    fetched = []
    for name, url in specs:
        body = fetch_json(url)
        fetched.append((name, {'url': url,
                               'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                               'n_features': _rows_of(url, body),
                               'sha256': hashlib.sha256(body).hexdigest()}, body))
    out.mkdir(parents=True, exist_ok=True)
    for name, _, body in fetched:                       # stage the whole set...
        (out / (name + '.part')).write_bytes(body)
    result = {}
    for name, meta, body in fetched:                    # ...then commit it together
        dest = out / name
        (out / (name + '.part')).replace(dest)
        source_record_path(dest).write_text(json.dumps(meta, indent=1), encoding='utf-8')
        result[name] = ({**meta, 'reused': False}, body)
    return result


def _retry_after_s(response, default):
    """A 429's Retry-After in seconds (the delta-seconds form), capped; `default` otherwise."""
    try:
        return min(max(float(response.headers.get('Retry-After')), 0.0), PROBE_RETRY_AFTER_CAP_S)
    except (AttributeError, TypeError, ValueError):
        return default


def probe_metadata(server, pano_id, session=None):
    """Answer of GET <server>/backupImage/<pano_id>/metadata: 200 only for a 200 whose body
    is a JSON object, the string '200-not-json' for any other 200 (a login or error page
    served with 200 must not read as "the file is there"), else the HTTP status, or None
    when no answer came back. Redirects are NOT followed: a 3xx comes back as itself and
    counts as a failed probe. Retries a 5xx, a 429 (honouring Retry-After, capped at
    PROBE_RETRY_AFTER_CAP_S) or a connection error after PROBE_BACKOFF_S; the caller spaces
    calls. (Tests monkeypatch this, or pass a fake `session`.)"""
    url = server + API_METADATA.format(pano_id=pano_id)
    get = (session or requests).get
    status = None
    for attempt in range(len(PROBE_BACKOFF_S) + 1):
        response = None
        try:
            response = get(url, timeout=PROBE_TIMEOUT_S, allow_redirects=False)
            status = response.status_code
        except requests.RequestException:
            status = None
        if status == 200:
            try:
                return 200 if isinstance(response.json(), dict) else '200-not-json'
            except ValueError:
                return '200-not-json'
        if status is not None and status < 500 and status != 429:
            return status
        if attempt < len(PROBE_BACKOFF_S):
            wait = PROBE_BACKOFF_S[attempt]
            time.sleep(_retry_after_s(response, wait) if status == 429 else wait)
    return status


# --------------------------------------------------------------------------- expected set

def expected_panos(run_dir, endpoint):
    """What the records say this server was sent: ({pano_id: {labels, label_types,
    campaigns}}, [campaign summaries], [problems]). Each campaign is replayed through
    send_to_ps.transform_record at its own [select_min, select_max) and rig mask, and the
    replayed counts are checked against the records (label_count_problems)."""
    campaigns, problems = position_check.campaigns_for(run_dir, endpoint)
    if not campaigns and not problems:
        problems.append(f'no submission record in {run_dir} holds any line for {endpoint}')
    expected = defaultdict(lambda: {'labels': 0, 'label_types': set(), 'campaigns': set()})
    summaries = []
    for c in campaigns:
        if c['select_min'] is None:
            problems.append(f"{c['campaign']}: band key cannot be parsed, so what it sent is unknown")
            continue
        panos, labels = set(), 0
        with open(c['path'], encoding='utf-8') as f:
            for n, line in enumerate(f, 1):
                if not line.strip() or (c['line_numbers'] is not None and n not in c['line_numbers']):
                    continue
                record = json.loads(line)
                sent = send_to_ps.transform_record(record, c['select_min'], c['select_max'],
                                                   mask_rig=c['rig_masked'])['labels']
                if not sent:
                    continue
                pid = str(record['pano']['panorama_id'])
                entry = expected[pid]
                entry['labels'] += len(sent)
                entry['label_types'].add(record.get('label_type') or 'CurbRamp')
                entry['campaigns'].add(c['campaign'])
                panos.add(pid)
                labels += len(sent)
        summaries.append({'campaign': c['campaign'], 'select_min': c['select_min'],
                          'select_max': c['select_max'], 'rig_masked': c['rig_masked'],
                          'lines': c['lines'], 'recorded_labels': c['labels'],
                          'replayed_labels': labels, 'panos': len(panos),
                          'partial': c['line_numbers'] is not None,
                          'band': c['band'], 'record': c['record']})
    by_record = defaultdict(list)
    for summary in summaries:
        by_record[summary['record']].append(summary)
    for parts in by_record.values():
        problems += label_count_problems(parts)
    return dict(expected), summaries, problems


def label_count_problems(parts):
    """Check one record's replayed label counts against what it recorded; `parts` are the
    expected_panos summaries of one record's campaigns for this server.

    A band records exactly what it sent. A base entry's count is its own campaign plus every
    band that covers the file (send_to_ps.write_submission_record adds a band's count to the
    base total, and moves the base min_confidence down to the band's floor, when the band
    completes), so it is checked against its replay plus the complete bands' replays. This
    turns the inferred [select_min, select_max) and mask into a check: e.g. a base campaign
    resumed after its band completed (a #59 append, sent at the lowered floor) no longer
    adds up."""
    problems = []
    bands = [p for p in parts if p['band'] is not None]
    for p in parts:
        expect = p['replayed_labels']
        if p['band'] is None:
            expect += sum(b['replayed_labels'] for b in bands if not b['partial'])
        recorded = p['recorded_labels']
        if not isinstance(recorded, int) or isinstance(recorded, bool):
            problems.append(f"{p['campaign']}: the record holds no label count ({recorded!r}) "
                            f"to check the replay against")
        elif recorded != expect:
            what = 'its replay' if p['band'] is not None else 'its replay plus its complete bands'
            problems.append(f"{p['campaign']}: the record says {recorded:,} label(s), {what} "
                            f"gives {expect:,} -- what it sent is not what the replay infers, so "
                            f"the expected set cannot be trusted")
    return problems


# ---------------------------------------------------------------------------- server state

def server_ai_panos(labels_all, raw_by_type):
    """Join /labels/all (has_backup, ai_generated; no pano id) to rawLabels (pano id) on
    label_id: ({pano_id: {ai_labels, backed, pitch}}, unjoinable_ai_label_count,
    {pano_id: backed} over ALL labels of those types, crowd included). Only labels of the
    pulled types are considered; a live AI label of one of them that rawLabels does not list
    cannot be placed on a pano, and is counted rather than dropped."""
    raw = {}
    for feats in raw_by_type.values():
        for ft in feats:
            q = ft['properties']
            raw[int(q['label_id'])] = (str(q['pano_id']), q.get('camera_pitch'))
    panos, unjoinable, any_label = {}, 0, {}
    for ft in labels_all:
        q = ft['properties']
        if q.get('label_type') not in raw_by_type:
            continue
        hit = raw.get(int(q['label_id']))
        if hit is not None:
            any_label[hit[0]] = any_label.get(hit[0], False) or q.get('has_backup') is True
        if not q.get('ai_generated'):
            continue
        if hit is None:
            unjoinable += 1
            continue
        pid, pitch = hit
        p = panos.setdefault(pid, {'ai_labels': 0, 'backed': False, 'pitch': None})
        p['ai_labels'] += 1
        p['backed'] = p['backed'] or q.get('has_backup') is True
        if pitch is not None:
            p['pitch'] = pitch
    return panos, unjoinable, any_label


def classify(expected, server, probe, max_confirm=DEFAULT_MAX_CONFIRM, confirm=True,
             spacing_s=PROBE_SPACING_S, ever_labelled=None):
    """({pano_id: row}, [problems]). `server` is server_ai_panos()'s map; `probe(pano_id)`
    returns probe_metadata()'s answer. has_backup false is only ever "unconfirmed".
    `ever_labelled` is the set of pano ids whose row has `has_labels`: an expected pano with
    no live AI label is `retired` only when it is in it, else `never_landed` (a problem).
    None (no pano table to ask) reads every such pano as retired."""
    rows, queue, problems = {}, [], []
    for pid in sorted(expected):
        s = server.get(pid)
        row = {'pano_id': pid, 'sent_labels': expected[pid]['labels'],
               'live_ai_labels': s['ai_labels'] if s else 0,
               'camera_pitch': s['pitch'] if s else None, 'probe': '', 'note': ''}
        if s is None:
            landed = ever_labelled is None or pid in ever_labelled
            row['status'] = 'retired' if landed else 'never_landed'
        elif s['backed']:
            row['status'] = 'covered'
        else:
            row['status'] = 'unconfirmed'
            queue.append(pid)
        rows[pid] = row
    never = sum(r['status'] == 'never_landed' for r in rows.values())
    if never:
        problems.append(f'{never:,} expected pano(s) never_landed: the records say labels were '
                        f'sent, but the server has never held a label on them (`has_labels` '
                        f'false, or no pano row)')
    if not confirm:
        for pid in queue:
            rows[pid]['note'] = 'not probed (--no-confirm)'
        return rows, problems
    if len(queue) > max_confirm:
        problems.append(f'{len(queue)} pano(s) need a metadata probe, more than --max-confirm '
                        f'{max_confirm}; the first {max_confirm} were probed and the rest are '
                        f'reported unconfirmed')
    failed = 0
    for i, pid in enumerate(queue):
        row = rows[pid]
        if i >= max_confirm:
            row['note'] = 'not probed (over --max-confirm)'
            continue
        if i:
            time.sleep(spacing_s)
        status = probe(pid)
        row['probe'] = '' if status is None else str(status)
        if status == 200:
            row['status'] = 'covered_metadata'
        elif status == 404 and row['camera_pitch'] is not None:
            row['status'] = 'missing'
        elif status == 404:
            row['note'] = '404 with a null camera_pitch: the probe cannot see the file'
        else:
            failed += 1
            row['note'] = 'probe failed'
    if failed:
        problems.append(f'{failed} metadata probe(s) got no usable answer (not a JSON 200 or a 404)')
    return rows, problems


# --------------------------------------------------------------------------------- archive

def load_archive(arg):
    """({pano_id: {filename, bytes, sha256}}, decayed_ids, index_path) from an index.csv, or
    a directory holding one (a `panos/` dir's manifest sits beside it, as
    export_benchmark.manifest_dir writes it). UsageError when there is none."""
    path = Path(arg)
    if path.is_dir():
        base = path.parent if path.name == 'panos' and not (path / 'index.csv').exists() else path
        path = base / 'index.csv'
    if not path.exists():
        raise UsageError(f'--archive: no index.csv at {path}')
    with open(path, newline='', encoding='utf-8') as f:
        index = {row['panorama_id']: row for row in csv.DictReader(f)}
    decayed_path = path.with_name('decayed.txt')
    decayed = set()
    if decayed_path.exists():
        decayed = {ln.strip() for ln in decayed_path.read_text(encoding='utf-8').splitlines() if ln.strip()}
    return index, decayed, path


def archive_state(pid, index, decayed):
    """archived (in index.csv), decayed (the source said it was gone before it could be
    archived), or not_archived."""
    if pid in index:
        return 'archived'
    return 'decayed' if pid in decayed else 'not_archived'


# ---------------------------------------------------------------------------------- report

def exit_code(counts, problems, strict=False):
    if problems:
        return EXIT_UNDETERMINED
    if counts.get('missing') or (strict and counts.get('unconfirmed')):
        return EXIT_GAPS
    return EXIT_OK


def write_csv(path, rows, fields):
    """Write rows, or remove a stale file when there are none (an old missing.csv beside a
    clean report would contradict it)."""
    if not rows:
        path.unlink(missing_ok=True)
        return False
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    return True


def _range(c):
    hi = '' if c['select_max'] is None else f", {c['select_max']:g})"
    return f"[{c['select_min']:g}{hi or ', ∞)'}"


def _count(v):
    return f'{v:,}' if isinstance(v, int) and not isinstance(v, bool) else str(v)


def _header(ctx):
    return [f"# Post-submission coverage check: {ctx['city']}", '',
            f"Server `{ctx['server']}` (records matched on `{ctx['endpoint']}`), run "
            f"`{ctx['run_dir']}`, checked {ctx['generated_at']} by `scripts/coverage_check.py` "
            f"(issue #46). Read-only: GET requests only.", '']


def render_failure_report(ctx):
    """report.md for a check that stopped before classifying: exit 2 and why, so a failed
    re-run never leaves an earlier "Exit 0" standing."""
    return '\n'.join(_header(ctx) + [
        '**Exit 2**: coverage could not be determined -- the check stopped before classifying '
        'any pano, and no earlier result stands.', '', '## Problems', ''] +
        [f'- {p}' for p in ctx['problems']] + [''])


def render_report(ctx):
    """report.md as a string. `ctx` is what run() collected."""
    c, L = ctx['counts'], []
    n_exp = len(ctx['rows'])
    L += _header(ctx) + [
        f"**Exit {ctx['exit']}**: " + {0: 'every expected pano with a live AI label is backed up.',
                                      1: 'some expected panos are missing their backup image.',
                                      2: 'coverage could not be fully determined (see problems).'}[ctx['exit']],
        '']
    if ctx['problems']:
        L += ['## Problems', ''] + [f'- {p}' for p in ctx['problems']] + ['']
    L += ['## Server pulls', '', '| file | url | fetched (UTC) | rows | sha256 |',
          '|---|---|---|---:|---|']
    for name, m in ctx['pulls']:
        L.append(f"| `{name}` | {m['url']} | {m['fetched_at']}{' (cached)' if m.get('reused') else ''} "
                 f"| {m['n_features']:,} | `{m['sha256']}` |")
    L += ['', 'The pulls are cached beside this report and not tracked. Every run pulls fresh '
          'unless `--reuse-pulls` is given; a reused set is marked (cached).', '',
          '## Expected set (campaigns unioned)', '',
          'Each campaign replayed through `send_to_ps.transform_record` at the range it sent; '
          '`rig_masked` absent in a record means the campaign predates the nadir mask. The '
          'recorded label count of a base entry includes its bands once they cover the file, so only '
          'the replayed column is per campaign. Every recorded count is checked: a band must equal '
          'its replay, a base entry its replay plus its complete bands.', '',
          '| campaign | sent range | rig masked | lines | labels (record) | labels (replayed) | panos with labels |',
          '|---|---|---|---:|---:|---:|---:|']
    for s in ctx['campaigns']:
        part = ' (partial, sidecar lines)' if s['partial'] else ''
        L.append(f"| `{s['campaign']}`{part} | {_range(s)} | {'yes' if s['rig_masked'] else 'no'} "
                 f"| {s['lines']:,} | {_count(s['recorded_labels'])} | {s['replayed_labels']:,} "
                 f"| {s['panos']:,} |")
    L += ['', f'Distinct expected panos: **{n_exp:,}**.', '',
          '## Classification', '', '| status | panos | meaning |', '|---|---:|---|']
    meaning = {'covered': 'a live AI label on it reads `has_backup: true`',
               'covered_metadata': '`/backupImage/<id>/metadata` answered 200 with a JSON object',
               'missing': 'metadata 404 with a non-null `camera_pitch` (decisive)',
               'unconfirmed': 'flag false and not settled by the probe (e.g. 404 with a null pitch)',
               'retired': 'no live AI label left, but the pano once held one: nothing renders, not a gap',
               'never_landed': 'no label ever on the server (`has_labels` false or no pano row): '
                               'the labels never arrived (a problem)'}
    for st in STATUSES:
        L.append(f'| {st} | {c.get(st, 0):,} | {meaning[st]} |')
    live = n_exp - c.get('retired', 0) - c.get('never_landed', 0)
    ok = c.get('covered', 0) + c.get('covered_metadata', 0)
    L += ['', f'**{ok:,} of {live:,}** expected panos with a live AI label are confirmed backed up.', '']
    if ctx.get('archive'):
        a = ctx['archive']
        L += ['## Archive cross-reference', '', f"Index `{a['index']}`.", '',
              '| gap panos | archived | decayed | not archived |', '|---:|---:|---:|---:|',
              f"| {a['gaps']:,} | {a['archived']:,} | {a['decayed']:,} | {a['not_archived']:,} |", '',
              '`fallback.csv` lists the archived files for gap panos. Placing them on the '
              'server is out of scope (SidewalkWebpage#4865).', '']
    x = ctx['cross']
    L += ['## Server cross-checks', '',
          f"- Live AI labels of the checked type(s) on the server: {x['server_ai_labels']:,} on "
          f"{x['server_ai_panos']:,} panos.",
          f"- Server AI panos NOT in the expected set: {x['server_ai_not_expected']:,} "
          f"(labels sent from files with no record here, or from another run).",
          f"- Live AI labels that rawLabels does not list (unjoinable, no pano id; a problem "
          f"when non-zero): {x['unjoinable']:,}.",
          f"- `/adminapi/panos`: {x['pano_rows']:,} rows, {x['has_labels']:,} with `has_labels` "
          f"(ever labelled, soft-deleted included); {x['has_labels_not_expected']:,} of those are "
          f"not expected (crowd-only, retired-only or other); {x['has_labels_not_expected_typed']:,} "
          f"of them carry a live label of the checked type(s), and "
          f"{x['has_labels_not_expected_backed']:,} of those read `has_backup: true`.",
          f"- Expected panos absent from the server's pano table (a problem when non-zero): "
          f"{x['expected_not_in_panos']:,}.",
          '', '## Reading this', '',
          '- `has_backup: true` is a cache of a past disk check by the server; a file removed '
          'later would still read true. `false` means unconfirmed (NULL is read as false), never '
          'confirmed absent.',
          '- `/backupImage/<id>/metadata` also 404s when the pano row has no `camera_pitch`, so '
          'it can confirm only panos whose pose has been pushed.',
          '- Human cross-checks: the admin health page\'s backup aggregate (admin session), and '
          'the panorama scraper\'s `pano_id_log.csv` (`downloaded=0` rows) on the server host.',
          '']
    return '\n'.join(L)


# -------------------------------------------------------------------------------------- main

def check_inputs(args):
    """Validate arguments before anything is written: {run_dir, server, endpoint, out,
    archive}. UsageError on anything wrong (exit 2, never 1)."""
    if not args.results.is_file():
        raise UsageError(f'{args.results} does not exist or is not a file')
    server = args.server.rstrip('/')
    if not server.startswith(('http://', 'https://')):
        raise UsageError(f'--server must be an http(s) site root, got {args.server!r}')
    if args.max_confirm < 0:
        raise UsageError('--max-confirm must be >= 0')
    run_dir = args.results.resolve().parent
    return {'run_dir': run_dir, 'server': server,
            'endpoint': send_to_ps.canonical_endpoint(server + SUBMIT_PATH),
            'out': args.out or run_dir / 'coverage',
            'archive': load_archive(args.archive) if args.archive else None}


def _report_ctx(run_dir, server, endpoint):
    return {'city': run_dir.name, 'server': server, 'endpoint': endpoint,
            'run_dir': position_check.repo_relative(run_dir),
            'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}


def run(args, run_dir, server, endpoint, out, archive):
    """The check proper; returns the exit code. PullError and anything unexpected propagate
    to main(), which turns them into an exit-2 report."""
    try:
        expected, campaigns, problems = expected_panos(run_dir, endpoint)
    except ValueError as exc:   # an unreadable record
        expected, campaigns, problems = {}, [], [str(exc)]
    label_types = sorted({t for e in expected.values() for t in e['label_types']}) or ['CurbRamp']

    specs = [('panos.json', server + API_PANOS), ('labels_all.geojson', server + API_LABELS_ALL)] + [
        (f'raw_labels_{t}.geojson', server + API_RAW_LABELS.format(label_type=t)) for t in label_types]
    fetched = pull_set(specs, out, reuse=args.reuse_pulls)
    pulls = [(name, fetched[name][0]) for name, _ in specs]
    payloads = {name: json.loads(body.decode('utf-8')) for name, (_, body) in fetched.items()}

    pano_rows = payloads['panos.json']
    pano_ids = {str(p['pano_id']) for p in pano_rows}
    has_labels = {str(p['pano_id']) for p in pano_rows if p.get('has_labels')}
    server_panos, unjoinable, any_label = server_ai_panos(
        payloads['labels_all.geojson']['features'],
        {t: payloads[f'raw_labels_{t}.geojson']['features'] for t in label_types})
    session = requests.Session()
    rows, probe_problems = classify(expected, server_panos,
                                    lambda pid: probe_metadata(server, pid, session),
                                    args.max_confirm, not args.no_confirm,
                                    ever_labelled=has_labels)
    problems += probe_problems
    counts = Counter(r['status'] for r in rows.values())

    gaps = [r for r in rows.values() if r['status'] in GAP_STATUSES]
    archive_summary = None
    if archive is not None:
        index, decayed, index_path = archive
        for r in gaps:
            r['archive'] = archive_state(r['pano_id'], index, decayed)
        fallback = [{'pano_id': r['pano_id'], 'status': r['status'], **index[r['pano_id']]}
                    for r in gaps if r['archive'] == 'archived']
        write_csv(out / 'fallback.csv', fallback,
                  ['pano_id', 'status', 'filename', 'bytes', 'sha256'])
        states = Counter(r['archive'] for r in gaps)
        archive_summary = {'index': index_path.as_posix(), 'gaps': len(gaps), **{
            k: states.get(k, 0) for k in ('archived', 'decayed', 'not_archived')}}
    write_csv(out / 'missing.csv', gaps, ['pano_id', 'status', 'sent_labels', 'live_ai_labels',
                                          'camera_pitch', 'probe', 'note', 'archive'])

    extra = has_labels - set(expected)
    not_in_panos = set(expected) - pano_ids
    if unjoinable:
        problems.append(f'{unjoinable:,} live AI label(s) of the checked type(s) are not listed by '
                        f'rawLabels, so their panos are unknown and a gap could hide behind them')
    if not_in_panos:
        problems.append(f"{len(not_in_panos):,} expected pano(s) have no row in the server's pano "
                        f"table")
    cross = {'server_ai_labels': sum(p['ai_labels'] for p in server_panos.values()),
             'server_ai_panos': len(server_panos),
             'server_ai_not_expected': len(set(server_panos) - set(expected)),
             'unjoinable': unjoinable, 'pano_rows': len(pano_rows), 'has_labels': len(has_labels),
             'has_labels_not_expected': len(extra),
             'has_labels_not_expected_typed': sum(1 for p in extra if p in any_label),
             'has_labels_not_expected_backed': sum(1 for p in extra if any_label.get(p)),
             'expected_not_in_panos': len(not_in_panos)}
    code = exit_code(counts, problems, args.strict)
    ctx = {**_report_ctx(run_dir, server, endpoint),
           'exit': code, 'problems': problems, 'pulls': pulls, 'campaigns': campaigns,
           'rows': rows, 'counts': counts, 'archive': archive_summary, 'cross': cross}
    (out / 'report.md').write_text(render_report(ctx), encoding='utf-8', newline='\n')

    live = len(rows) - counts.get('retired', 0) - counts.get('never_landed', 0)
    print(f"{run_dir.name}: {len(rows):,} expected panos; " +
          ', '.join(f'{s} {counts.get(s, 0):,}' for s in STATUSES) +
          f"; {counts.get('covered', 0) + counts.get('covered_metadata', 0):,}/{live:,} live confirmed")
    for p in problems:
        print(f'problem: {p}', file=sys.stderr)
    print(f'report: {out / "report.md"}  (exit {code})')
    return code


def write_failure(inputs, problem):
    """Replace report.md with an exit-2 report and drop the CSVs, which would otherwise
    describe an earlier check. Best effort: a failure here is printed, not raised."""
    out = inputs['out']
    try:
        out.mkdir(parents=True, exist_ok=True)
        ctx = {**_report_ctx(inputs['run_dir'], inputs['server'], inputs['endpoint']),
               'problems': [problem]}
        (out / 'report.md').write_text(render_failure_report(ctx), encoding='utf-8', newline='\n')
        for name in ('missing.csv', 'fallback.csv'):
            (out / name).unlink(missing_ok=True)
        print(f'report: {out / "report.md"}  (exit {EXIT_UNDETERMINED})')
    except OSError as exc:
        print(f'could not write the exit-2 report: {exc}', file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('results', type=Path,
                    help='a results file of the run; every *.submission.json in its directory is read')
    ap.add_argument('--server', required=True, help='site root, e.g. https://sidewalk-richmond.cs.washington.edu')
    ap.add_argument('--out', type=Path, help='output dir (default: <run dir>/coverage)')
    ap.add_argument('--reuse-pulls', action='store_true',
                    help='re-read the cached server pulls instead of pulling fresh (offline '
                         're-runs); refused when a cached file changed or the set is of mixed age')
    ap.add_argument('--archive', help='an archive index.csv, or the directory holding it (or its panos/)')
    ap.add_argument('--max-confirm', type=int, default=DEFAULT_MAX_CONFIRM,
                    help=f'most metadata probes to make (default {DEFAULT_MAX_CONFIRM})')
    ap.add_argument('--no-confirm', action='store_true', help='make no metadata probes')
    ap.add_argument('--strict', action='store_true', help='exit 1 on unconfirmed panos too')
    args = ap.parse_args(argv)     # argparse's own usage errors exit 2 as well

    try:
        inputs = check_inputs(args)
    except UsageError as exc:
        print(f'coverage_check: {exc}', file=sys.stderr)
        return EXIT_UNDETERMINED
    try:
        return run(args, **inputs)
    except PullError as exc:
        print(f'pull failed: {exc}', file=sys.stderr)
        write_failure(inputs, f'pull failed: {exc}')
    except Exception as exc:       # a traceback must never read as exit 1 ("missing")
        traceback.print_exc()
        write_failure(inputs, f'unexpected error: {type(exc).__name__}: {exc}')
    return EXIT_UNDETERMINED


if __name__ == '__main__':
    sys.exit(main())
