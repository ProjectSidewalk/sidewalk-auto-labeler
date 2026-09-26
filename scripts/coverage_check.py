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
                         included) -- informational only; it has no has_backup.
  - /backupImage/<id>/metadata   200 iff the file is on disk AND the pano row has a
                         camera_pitch (and width/height/lat/lng/heading). A 404 on a pano
                         whose pitch is null is therefore UNCONFIRMED, not missing; only a
                         404 with a non-null pitch is decisive. Probed sequentially, with
                         spacing and backoff, only for panos the flag leaves unconfirmed.
  The aggregate on the admin health page (labeled / backed_up / no_backup / unchecked) is
  the human cross-check; it needs an admin session, so this script cannot read it.

The EXPECTED set comes from the submission records and their sidecars, never from
results.jsonl alone (the selection rule): every campaign every `*.submission.json` in the
run directory holds for this server (position_check.campaigns_for -- base entries and
bands, partial campaigns restricted to their sidecar's lines), each replayed through
send_to_ps.transform_record at exactly the range it sent ([select_min, select_max), nadir
mask iff the record's `rig_masked`). A pano is expected if any campaign sent it at least
one label. There is deliberately no --min-confidence: the record is the truth.

Classification of each expected pano, in order:
  retired           no live AI label on the server (e.g. Laurens' soft-deleted rig labels)
                    -- reported, never a gap: nothing renders for it
  covered           some live AI label on it reads has_backup: true
  covered_metadata  /backupImage/<id>/metadata answered 200
  missing           the metadata probe 404'd AND the pano has a camera_pitch (decisive)
  unconfirmed       everything else (404 with a null pitch, not probed, over --max-confirm)

Exit status: 0 no missing and no unconfirmed; 1 some missing (or, with --strict, some
unconfirmed); 2 could not determine (a record problem, a failed or empty pull, a failed
probe, or more panos to confirm than --max-confirm).

Outputs (default runs/<city>/coverage/): report.md, plus missing.csv (missing +
unconfirmed rows) and fallback.csv (archived files that could be placed, with --archive)
when they are non-empty -- all git-tracked. The three API pulls are cached there, with a
<file>.source.json fetch record each, and are NOT tracked; the report carries each pull's
url, fetch time, sha256 and feature count. A cached pull is reused (its age is printed);
--refresh re-pulls.

Usage:
    python scripts/coverage_check.py runs/richmond/results.jsonl \\
        --server https://sidewalk-richmond.cs.washington.edu
    python scripts/coverage_check.py runs/laurens/results.raw.jsonl \\
        --server https://sidewalk-laurens.cs.washington.edu --archive D:/archive/laurens/panos

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
PROBE_BACKOFF_S = (2.0, 8.0)   # waits before retry 1 and 2 on a 5xx or a connection error
PULL_TIMEOUT_S = 300
PROBE_TIMEOUT_S = 30

STATUSES = ('covered', 'covered_metadata', 'missing', 'unconfirmed', 'retired')
EXIT_OK, EXIT_GAPS, EXIT_UNDETERMINED = 0, 1, 2


class PullError(Exception):
    """A server pull that cannot be used (HTTP error, not JSON, zero rows)."""


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


def pull(url, dest, refresh=False):
    """Download `url` to `dest` unless it is cached there; return its fetch record
    {url, fetched_at, n_features, sha256, reused}. Written binary, temp-then-rename; a body
    that is not JSON or has zero rows is refused rather than cached (a zero-row 200 would
    otherwise read as "the city has no labels")."""
    side = source_record_path(dest)
    if dest.exists() and side.exists() and not refresh:
        meta = json.loads(side.read_text(encoding='utf-8'))
        if meta.get('url') == url:
            meta['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
            meta['reused'] = True
            print(f'reusing cached {dest.name} (pulled {meta.get("fetched_at")}); --refresh '
                  f're-pulls it', file=sys.stderr)
            return meta
    body = fetch_json(url)
    try:
        n = count_rows(json.loads(body.decode('utf-8')))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PullError(f'{url}: not JSON ({exc})') from exc
    except PullError as exc:
        raise PullError(f'{url}: {exc}') from exc
    if n == 0:
        raise PullError(f'{url}: zero rows -- refusing to cache an empty pull')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    tmp.write_bytes(body)
    tmp.replace(dest)
    meta = {'url': url, 'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'n_features': n, 'sha256': hashlib.sha256(body).hexdigest()}
    side.write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return {**meta, 'reused': False}


def probe_metadata(server, pano_id, session=None):
    """HTTP status of GET <server>/backupImage/<pano_id>/metadata, or None when no answer
    came back. Retries a 5xx or a connection error after PROBE_BACKOFF_S; the caller spaces
    calls. (Tests monkeypatch this.)"""
    url = server + API_METADATA.format(pano_id=pano_id)
    get = (session or requests).get
    status = None
    for wait in (0.0,) + PROBE_BACKOFF_S:
        if wait:
            time.sleep(wait)
        try:
            status = get(url, timeout=PROBE_TIMEOUT_S).status_code
        except requests.RequestException:
            status = None
            continue
        if status < 500:
            return status
    return status


# --------------------------------------------------------------------------- expected set

def expected_panos(run_dir, endpoint):
    """What the records say this server was sent: ({pano_id: {labels, label_types,
    campaigns}}, [campaign summaries], [problems]). Each campaign is replayed through
    send_to_ps.transform_record at its own [select_min, select_max) and rig mask."""
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
                          'partial': c['line_numbers'] is not None})
    return dict(expected), summaries, problems


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
             spacing_s=PROBE_SPACING_S):
    """({pano_id: row}, [problems]). `server` is server_ai_panos()'s map; `probe(pano_id)`
    returns an HTTP status or None. has_backup false is only ever "unconfirmed"."""
    rows, queue, problems = {}, [], []
    for pid in sorted(expected):
        s = server.get(pid)
        row = {'pano_id': pid, 'sent_labels': expected[pid]['labels'],
               'live_ai_labels': s['ai_labels'] if s else 0,
               'camera_pitch': s['pitch'] if s else None, 'probe': '', 'note': ''}
        if s is None:
            row['status'] = 'retired'
        elif s['backed']:
            row['status'] = 'covered'
        else:
            row['status'] = 'unconfirmed'
            queue.append(pid)
        rows[pid] = row
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
        problems.append(f'{failed} metadata probe(s) got no usable answer (not 200/404)')
    return rows, problems


# --------------------------------------------------------------------------------- archive

def load_archive(arg):
    """({pano_id: {filename, bytes, sha256}}, decayed_ids, index_path) from an index.csv, or
    a directory holding one (a `panos/` dir's manifest sits beside it, as
    export_benchmark.manifest_dir writes it)."""
    path = Path(arg)
    if path.is_dir():
        base = path.parent if path.name == 'panos' and not (path / 'index.csv').exists() else path
        path = base / 'index.csv'
    if not path.exists():
        raise SystemExit(f'--archive: no index.csv at {path}')
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


def render_report(ctx):
    """report.md as a string. `ctx` is what main() collected."""
    c, L = ctx['counts'], []
    n_exp = len(ctx['rows'])
    L += [f"# Post-submission coverage check: {ctx['city']}", '',
          f"Server `{ctx['server']}` (records matched on `{ctx['endpoint']}`), run "
          f"`{ctx['run_dir']}`, checked {ctx['generated_at']} by `scripts/coverage_check.py` "
          f"(issue #46). Read-only: GET requests only.", '',
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
    L += ['', 'The pulls are cached beside this report and not tracked; `--refresh` re-pulls.', '',
          '## Expected set (campaigns unioned)', '',
          'Each campaign replayed through `send_to_ps.transform_record` at the range it sent; '
          '`rig_masked` absent in a record means the campaign predates the nadir mask. The '
          'recorded label count of a base entry includes its bands once they cover the file, so only '
          'the replayed column is per campaign.', '',
          '| campaign | sent range | rig masked | lines | labels (record) | labels (replayed) | panos with labels |',
          '|---|---|---|---:|---:|---:|---:|']
    for s in ctx['campaigns']:
        part = ' (partial, sidecar lines)' if s['partial'] else ''
        L.append(f"| `{s['campaign']}`{part} | {_range(s)} | {'yes' if s['rig_masked'] else 'no'} "
                 f"| {s['lines']:,} | {s['recorded_labels']} | {s['replayed_labels']:,} | {s['panos']:,} |")
    L += ['', f'Distinct expected panos: **{n_exp:,}**.', '',
          '## Classification', '', '| status | panos | meaning |', '|---|---:|---|']
    meaning = {'covered': 'a live AI label on it reads `has_backup: true`',
               'covered_metadata': '`/backupImage/<id>/metadata` answered 200',
               'missing': 'metadata 404 with a non-null `camera_pitch` (decisive)',
               'unconfirmed': 'flag false and not settled by the probe (e.g. 404 with a null pitch)',
               'retired': 'no live AI label on the server: nothing renders, not a gap'}
    for st in STATUSES:
        L.append(f'| {st} | {c.get(st, 0):,} | {meaning[st]} |')
    live = n_exp - c.get('retired', 0)
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
    L += ['## Server cross-checks (informational)', '',
          f"- Live AI labels of the checked type(s) on the server: {x['server_ai_labels']:,} on "
          f"{x['server_ai_panos']:,} panos.",
          f"- Server AI panos NOT in the expected set: {x['server_ai_not_expected']:,} "
          f"(labels sent from files with no record here, or from another run).",
          f"- Live AI labels that rawLabels does not list (unjoinable, no pano id): "
          f"{x['unjoinable']:,}.",
          f"- `/adminapi/panos`: {x['pano_rows']:,} rows, {x['has_labels']:,} with `has_labels` "
          f"(ever labelled, soft-deleted included); {x['has_labels_not_expected']:,} of those are "
          f"not expected (crowd-only, retired-only or other); {x['has_labels_not_expected_typed']:,} "
          f"of them carry a live label of the checked type(s), and "
          f"{x['has_labels_not_expected_backed']:,} of those read `has_backup: true`.",
          f"- Expected panos absent from the server's pano table: {x['expected_not_in_panos']:,}.",
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

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('results', type=Path,
                    help='a results file of the run; every *.submission.json in its directory is read')
    ap.add_argument('--server', required=True, help='site root, e.g. https://sidewalk-richmond.cs.washington.edu')
    ap.add_argument('--out', type=Path, help='output dir (default: <run dir>/coverage)')
    ap.add_argument('--refresh', action='store_true', help='re-pull the cached server state')
    ap.add_argument('--archive', help='an archive index.csv, or the directory holding it (or its panos/)')
    ap.add_argument('--max-confirm', type=int, default=DEFAULT_MAX_CONFIRM,
                    help=f'most metadata probes to make (default {DEFAULT_MAX_CONFIRM})')
    ap.add_argument('--no-confirm', action='store_true', help='make no metadata probes')
    ap.add_argument('--strict', action='store_true', help='exit 1 on unconfirmed panos too')
    args = ap.parse_args(argv)

    run_dir = args.results.resolve().parent
    server = args.server.rstrip('/')
    endpoint = send_to_ps.canonical_endpoint(server + SUBMIT_PATH)
    out = args.out or run_dir / 'coverage'
    out.mkdir(parents=True, exist_ok=True)
    if not args.results.exists():
        raise SystemExit(f'{args.results} does not exist')

    try:
        expected, campaigns, problems = expected_panos(run_dir, endpoint)
    except ValueError as exc:   # an unreadable record
        expected, campaigns, problems = {}, [], [str(exc)]
    label_types = sorted({t for e in expected.values() for t in e['label_types']}) or ['CurbRamp']

    pulls, payloads = [], {}
    try:
        for name, path in [('panos.json', API_PANOS), ('labels_all.geojson', API_LABELS_ALL)] + [
                (f'raw_labels_{t}.geojson', API_RAW_LABELS.format(label_type=t)) for t in label_types]:
            meta = pull(server + path, out / name, args.refresh)
            pulls.append((name, meta))
            payloads[name] = json.loads((out / name).read_bytes().decode('utf-8'))
    except PullError as exc:
        print(f'pull failed: {exc}', file=sys.stderr)
        return EXIT_UNDETERMINED

    server_panos, unjoinable, any_label = server_ai_panos(
        payloads['labels_all.geojson']['features'],
        {t: payloads[f'raw_labels_{t}.geojson']['features'] for t in label_types})
    session = requests.Session()
    rows, probe_problems = classify(expected, server_panos,
                                    lambda pid: probe_metadata(server, pid, session),
                                    args.max_confirm, not args.no_confirm)
    problems += probe_problems
    counts = Counter(r['status'] for r in rows.values())

    gaps = [r for r in rows.values() if r['status'] in ('missing', 'unconfirmed')]
    archive = None
    if args.archive:
        index, decayed, index_path = load_archive(args.archive)
        for r in gaps:
            r['archive'] = archive_state(r['pano_id'], index, decayed)
        fallback = [{'pano_id': r['pano_id'], 'status': r['status'], **index[r['pano_id']]}
                    for r in gaps if r['archive'] == 'archived']
        write_csv(out / 'fallback.csv', fallback,
                  ['pano_id', 'status', 'filename', 'bytes', 'sha256'])
        states = Counter(r['archive'] for r in gaps)
        archive = {'index': index_path.as_posix(), 'gaps': len(gaps), **{
            k: states.get(k, 0) for k in ('archived', 'decayed', 'not_archived')}}
    write_csv(out / 'missing.csv', gaps, ['pano_id', 'status', 'sent_labels', 'live_ai_labels',
                                          'camera_pitch', 'probe', 'note', 'archive'])

    pano_rows = payloads['panos.json']
    has_labels = {str(p['pano_id']) for p in pano_rows if p.get('has_labels')}
    extra = has_labels - set(expected)
    cross = {'server_ai_labels': sum(p['ai_labels'] for p in server_panos.values()),
             'server_ai_panos': len(server_panos),
             'server_ai_not_expected': len(set(server_panos) - set(expected)),
             'unjoinable': unjoinable, 'pano_rows': len(pano_rows), 'has_labels': len(has_labels),
             'has_labels_not_expected': len(extra),
             'has_labels_not_expected_typed': sum(1 for p in extra if p in any_label),
             'has_labels_not_expected_backed': sum(1 for p in extra if any_label.get(p)),
             'expected_not_in_panos': len(set(expected) - {str(p['pano_id']) for p in pano_rows})}
    code = exit_code(counts, problems, args.strict)
    ctx = {'city': run_dir.name, 'server': server, 'endpoint': endpoint,
           'run_dir': position_check.repo_relative(run_dir),
           'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'exit': code, 'problems': problems, 'pulls': pulls, 'campaigns': campaigns,
           'rows': rows, 'counts': counts, 'archive': archive, 'cross': cross}
    (out / 'report.md').write_text(render_report(ctx), encoding='utf-8', newline='\n')

    live = len(rows) - counts.get('retired', 0)
    print(f"{run_dir.name}: {len(rows):,} expected panos; " +
          ', '.join(f'{s} {counts.get(s, 0):,}' for s in STATUSES) +
          f"; {counts.get('covered', 0) + counts.get('covered_metadata', 0):,}/{live:,} live confirmed")
    for p in problems:
        print(f'problem: {p}', file=sys.stderr)
    print(f'report: {out / "report.md"}  (exit {code})')
    return code


if __name__ == '__main__':
    sys.exit(main())
