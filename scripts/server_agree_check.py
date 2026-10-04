"""Server-side read of a city's AI curb-ramp labels: human validations, and one human
auditor's own labels against the AI's, both directions.

This is the quick read that sits beside `agree_rate.py`. That tool compares a *run's raw
detections* with crowd labels in the pano frame and needs the run directory. This one
reads only what the server serves, so it works on a city where the AI labels are already
deployed and a person has been validating them and auditing streets (Laurens, issue
RampNet#158). Four read-only GETs, frozen into the output dir with url, fetch time,
sha256 and feature count; every number is relative to that pull:

  /v3/api/rawLabels?labelType=CurbRamp&filetype=geojson      -> raw_labels_CurbRamp.geojson
  /v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson    -> raw_labels_NoCurbRamp.geojson
  /v3/api/labelClusters?labelType=CurbRamp&filetype=geojson  -> label_clusters_CurbRamp.geojson
  /v3/api/streets?filetype=geojson                           -> streets.geojson

Two sections:

1. **Validations of the AI's CurbRamp labels.** Per label, the majority of HUMAN Agree vs
   Disagree votes (PS's own AI validator is excluded, as in agree_rate.human_status).
   Precision = agreed / (agreed + disagreed) with a Wilson interval. Reported with how
   many votes each label has and who cast them, because on these servers it is mostly one
   person.

2. **One human's labels vs the AI's** (`--human auto`, or `--human <user_id>`). `auto`
   picks the auditor by rule: the human user with the most CurbRamp labels in the
   snapshot (ties broken on user_id), so a replicator re-derives the same account from the
   pull and no full id needs to be written down. The AI labels a ramp from every pano that
   sees it; a human auditor labels it once. So the comparison is in world space, at
   several radii, against the server's *clusters* (its own 7.5 m complete linkage) and
   against the raw AI labels.
   - human -> AI: share of the human's CurbRamp labels the AI also has. The HEADLINE is
     one-to-one against AI clusters (agree_rate's matcher: greedy by distance, d <= r, a
     cluster credits at most one label); one-to-one against raw AI labels is the lenient
     bracket (a ramp has several AI views), and "any cluster within r" is the coverage
     line (one cluster may credit several labels, e.g. paired corner ramps). Each comes
     with a displaced-label chance floor: every human label moved 25 m in a seeded random
     direction (agree_rate.CHANCE_SHIFT_M / CHANCE_SEED). Labels with no AI cluster within
     10 m are written to misses.csv: candidate AI misses.
   - AI -> human: only clusters on streets the human has COMPLETED an audit of
     (`audit_count >= 1` and the human in the street's `user_ids`), split by how many AI
     views the cluster holds. The human's NoCurbRamp labels nearby are the informative
     disagreement.
   - The human's own validation vote on each AI label crossed with whether the human
     placed a label nearby. This is what says whether "no human label nearby" is a false
     positive or a coverage gap: a label the human voted Agree on and did not re-label is
     the latter.
   - Pano frame, for reference only: the human's labels whose pano the AI also labelled,
     matched to AI labels on that same pano with agree_rate's pano matcher (one-to-one,
     strictly within RampNet's radius), plus the any-label reading beside it.

Section 1 also gives precision per server CLUSTER (majority of its AI labels' human
statuses), because a label-level rate counts a ramp seen from k panos k times, and, with
`--results`, per confidence tier: each AI label is joined back to the run's stored
detection on (pano_id, pano_x, pano_y), the pixel rounding send_to_ps.py used.

Every per-city serial in an output (label, cluster, street) is written as
`<city>:<id>`, never bare: these ids restart at 1 in every city.

Caveats the report carries: validated labels reach validators through PS's validation
queue, not by random sampling, so section 1 is the precision of the labels that were
shown, not of a random sample; an audit is one pass from one pano, so AI -> human on
completed streets is a lower bound on coverage, never a precision read; ramps at an
intersection are attributed to one street edge and may be labelled from another; the
human's placement and the cluster centroid carry independent errors, so the 5 m row is
tight and the 10 m row loose.

Stdlib only (plus agree_rate's fetch/snapshot/matcher helpers).

Usage:
    python scripts/server_agree_check.py laurens --human auto --results runs/laurens/results.raw.jsonl
    python scripts/server_agree_check.py laurens --human <user_id>   # a named account
    python scripts/server_agree_check.py richmond          # validations only
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
from agree_rate import fetch, snapshot_row, human_status  # noqa: E402
from agree_rate import (Pt, match_one_to_one, match_pano, pano_distance,  # noqa: E402
                        PANO_RADIUS, CHANCE_SHIFT_M, CHANCE_SEED)
from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE  # noqa: E402
from eval_sites import wilson  # noqa: E402

AI_USER = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4'   # the auto-labeler's account, every city
RADII_M = (5.0, 7.5, 10.0)
HEADLINE_RADIUS_M = 7.5
MISS_RADIUS_M = 10.0
CHANCE_RANGE_SEEDS = range(1, 21)    # the floor's spread; the table itself uses CHANCE_SEED
API = {
    'CurbRamp': '/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson',
    'NoCurbRamp': '/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson',
    'clusters': '/v3/api/labelClusters?labelType=CurbRamp&filetype=geojson',
    'streets': '/v3/api/streets?filetype=geojson',
}
FILES = {
    'CurbRamp': 'raw_labels_CurbRamp.geojson',
    'NoCurbRamp': 'raw_labels_NoCurbRamp.geojson',
    'clusters': 'label_clusters_CurbRamp.geojson',
    'streets': 'streets.geojson',
}
TIERS = (f'>= {BENCHMARK_CONFIDENCE:.2f}',
         f'{OPERATIONAL_CONFIDENCE:.2f}-{BENCHMARK_CONFIDENCE:.2f}',
         f'< {OPERATIONAL_CONFIDENCE:.2f}', 'ambiguous', 'not joined')


def uid(city, i):
    """`<city>:<id>` for a per-city serial (label, cluster, street); '' for None."""
    return '' if i is None or i == '' else f'{city}:{i}'


def rate(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {(k / n) if n else float("nan"):.3f} [{lo:.3f}, {hi:.3f}]'


# ----------------------------------------------------------------------------- geometry

def haversine_m(lng1, lat1, lng2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


class Grid:
    """Exact nearest-point lookup over (lng, lat) points, bucketed at 0.001 deg.

    Searches square rings of cells outward until no unsearched cell can hold a closer
    point: after rings 0..k every unsearched point is at least k cell widths away, and a
    cell is never narrower than its longitude width at the data's highest latitude. So
    the answer is the true nearest at any distance, not only within the 3x3 block.
    """

    def __init__(self, points):
        self.cells = collections.defaultdict(list)
        for p in points:
            self.cells[self._key(p['lng'], p['lat'])].append(p)
        keys = list(self.cells)
        self.kx = (min((k[0] for k in keys), default=0), max((k[0] for k in keys), default=0))
        self.ky = (min((k[1] for k in keys), default=0), max((k[1] for k in keys), default=0))
        self.max_lat = max((abs(p['lat']) for p in points), default=0.0)

    @staticmethod
    def _cell_m(lat):
        """A lower bound on a cell's width in metres up to latitude |lat|: 0.001 deg of
        latitude is ~111 m, of longitude ~111 m * cos(lat); 0.99 is margin for the
        haversine sphere against the flat-cell bound."""
        return 0.99 * 111.19 * math.cos(math.radians(min(89.0, abs(lat) + 0.01)))

    @staticmethod
    def _key(lng, lat):
        return (int(math.floor(lng * 1000)), int(math.floor(lat * 1000)))

    def nearest(self, lng, lat):
        """(distance_m, point) of the nearest point, or (inf, None) when empty."""
        if not self.cells:
            return float('inf'), None
        kx, ky = self._key(lng, lat)
        last = max(kx - self.kx[0], self.kx[1] - kx, ky - self.ky[0], self.ky[1] - ky, 0)
        cell_m = self._cell_m(max(self.max_lat, abs(lat)))
        best, best_p = float('inf'), None
        for k in range(last + 1):
            for dx in range(-k, k + 1):
                for dy in ((-k, k) if abs(dx) != k else range(-k, k + 1)):
                    for p in self.cells.get((kx + dx, ky + dy), ()):
                        d = haversine_m(lng, lat, p['lng'], p['lat'])
                        if d < best:
                            best, best_p = d, p
            if best <= k * cell_m:
                break
        return best, best_p


def displaced(pts, shift_m=CHANCE_SHIFT_M, seed=CHANCE_SEED):
    """Every point moved shift_m in a seeded random direction: the same draws, in the
    same order, as agree_rate.chance_floor, so the one-to-one floor here is its number."""
    rng = random.Random(seed)
    out = []
    for p in pts:
        a = rng.uniform(0.0, 2.0 * math.pi)
        out.append(Pt(p.id, p.e + shift_m * math.cos(a), p.n + shift_m * math.sin(a)))
    return out


def any_hits(pts, sites, radius_m):
    """Indices of pts with ANY site within radius_m (d <= r): coverage, not one-to-one."""
    grid = geo.GridIndex(radius_m)
    for s in sites:
        grid.add(s.e, s.n, s)
    return {i for i, p in enumerate(pts)
            if any(math.hypot(p.e - s.e, p.n - s.n) <= radius_m for s in grid.near(p.e, p.n))}


# ----------------------------------------------------------------------------- inputs

def load_features(path):
    """Features with a finite position, as (properties + lng/lat) dicts; count of dropped."""
    feats = json.loads(path.read_text(encoding='utf-8'))['features']
    out, dropped = [], 0
    for ft in feats:
        coords = (ft.get('geometry') or {}).get('coordinates')
        if not coords or coords[0] is None or coords[1] is None:
            dropped += 1
            continue
        lng, lat = float(coords[0]), float(coords[1])
        if not (math.isfinite(lng) and math.isfinite(lat)) or abs(lng) > 360:
            dropped += 1
            continue
        q = dict(ft['properties'])
        q['lng'], q['lat'] = lng, lat
        out.append(q)
    return out, dropped


UUID_RE = re.compile(rb'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
REDACTED_SUFFIX = '-redacted'


AS_SERVED = 'as_served'   # subdir of the output dir: the pulls exactly as served (untracked)
REDACTION_NOTE = (f'human user ids cut to 8 hex chars + "{REDACTED_SUFFIX}" '
                  f'(scripts/server_agree_check.py; the pull as served stays in {AS_SERVED}/, '
                  'untracked)')


def human_ids(raw, ai_user):
    """Full user ids in `raw` (bytes) other than the AI account's."""
    ai = ai_user.encode()
    return {m for m in UUID_RE.findall(raw) if m != ai}


def git_ignored(path):
    """True only when git says `path` is ignored. Anything else (tracked or trackable, not
    a repository, no git) reads False, so the redaction guard fails safe."""
    try:
        r = subprocess.run(['git', '-C', str(path.parent), 'check-ignore', '-q', str(path)],
                           capture_output=True)
    except OSError:
        return False
    return r.returncode == 0


def publish_redacted(raw_path, dest, ai_user):
    """Write `dest` as `raw_path` (a pull as served) with every human user id cut to its
    first 8 hex characters + '-redacted' (the AI account's id, already in this file, is
    kept), and its sidecar as the pull's, plus `sha256_as_served`.

    This is what lets a pull sit at a tracked path: every number here depends only on which
    labels share an account, never on the full id, so the report re-renders identically,
    and its 8-character ids are the ones it always printed. Refuses, writing nothing, if
    two ids would collide. The served file's sha256 stays in the sidecar beside its url and
    fetch time, so the redacted copy is tied to what the server sent.
    """
    side = raw_path.parent / (raw_path.name + '.source.json')
    meta = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}
    raw = raw_path.read_bytes()
    short = collections.Counter(i[:8] for i in human_ids(raw, ai_user))
    if any(v > 1 for v in short.values()):
        sys.exit(f'{raw_path.name}: two user ids share an 8-character prefix; not redacting, '
                 f'and not writing {dest}')
    ai = ai_user.encode()
    out = UUID_RE.sub(lambda m: m.group(0) if m.group(0) == ai
                      else m.group(0)[:8] + REDACTED_SUFFIX.encode(), raw)
    meta['sha256_as_served'] = hashlib.sha256(raw).hexdigest()
    meta['redacted'] = REDACTION_NOTE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    tmp.write_bytes(out)
    tmp.replace(dest)
    (dest.parent / (dest.name + '.source.json')).write_text(
        json.dumps(meta, indent=1) + '\n', encoding='utf-8', newline='\n')


def check_redacted(path, ai_user):
    """Refuse a pull that holds full human user ids at a path git would track."""
    if git_ignored(path):
        return
    if human_ids(path.read_bytes(), ai_user):
        sys.exit(f'{path}: full human user ids at a path git would track; refusing to use it. '
                 f'Move it into {path.parent / AS_SERVED} (untracked) and re-run: the script '
                 'then writes the redacted copy here.')


def materialize(out, key, url, ai_user, refresh=False):
    """The pull for `key`, at its tracked path `out/<file>`, redacted.

    The body as served is fetched into `out/as_served/` only (cached there like any pull),
    and what lands at `out/<file>` is always publish_redacted's copy, so redaction is not
    a choice. A file already at `out/<file>` is reused (the frozen snapshot) but checked:
    one with full human ids at a path git does not ignore is refused (a pull written by an
    older version of this script; Richmond's and Vancouver's local caches are git-ignored
    and read as they are).
    """
    dest = out / FILES[key]
    if refresh or not dest.exists():
        raw = out / AS_SERVED / FILES[key]
        fetch(url, raw, refresh=refresh)
        publish_redacted(raw, dest, ai_user)
    check_redacted(dest, ai_user)
    return dest


def own_vote(label, user):
    """The one user's own vote on a label: 'agree' / 'disagree' / 'none'."""
    a = d = 0
    for v in label.get('validations') or ():
        if v.get('validator_type') != 'Human' or v.get('user_id') != user:
            continue
        a += v.get('validation') == 'Agree'
        d += v.get('validation') == 'Disagree'
    return 'agree' if a > d else 'disagree' if d > a else 'none'


def pick_auditor(cr, ai_user):
    """(user_id, CurbRamp label counts by human user). The rule: the human user with the
    most CurbRamp labels in the snapshot, ties broken on user_id; None when there is none."""
    counts = collections.Counter(q['user_id'] for q in cr if q['user_id'] != ai_user)
    if not counts:
        return None, counts
    return sorted(counts.items(), key=lambda t: (-t[1], t[0]))[0][0], counts


def tier_of(c):
    if c >= BENCHMARK_CONFIDENCE:
        return TIERS[0]
    if c >= OPERATIONAL_CONFIDENCE:
        return TIERS[1]
    return TIERS[2]


def load_tiers(paths):
    """({(pano_id, pano_x, pano_y): set of tiers}, [(file name, sha256)]) over the run's
    stored detections, keyed by send_to_ps.transform_record's pixel rounding."""
    keys = collections.defaultdict(set)
    files = []
    for path in paths:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for line in f:
                h.update(line)
                if not line.strip():
                    continue
                rec = json.loads(line)
                p = rec['pano']
                w, ht = p['width'], p['height']
                for d in rec.get('detections') or ():
                    keys[(p['panorama_id'], round(d['x_normalized'] * w),
                          round(d['y_normalized'] * ht))].add(tier_of(d['confidence']))
        files.append((Path(path).name, h.hexdigest()))
    return keys, files


def label_tier(q, keys):
    t = keys.get((q.get('pano_id'), q.get('pano_x'), q.get('pano_y')))
    if not t:
        return TIERS[4]
    return next(iter(t)) if len(t) == 1 else TIERS[3]


# ----------------------------------------------------------------------------- sections

def _majority(statuses):
    s = collections.Counter(statuses)
    a, d = s['agreed'], s['disagreed']
    return 'agreed' if a > d else 'disagreed' if d > a else 'unvalidated'


def cluster_statuses(clusters, ai_labels, ai_user, link_m=7.5):
    """Per-cluster human status, two ways: ({server cluster id: status}, [status per
    re-attached group], n AI labels outside every server cluster).

    A cluster's status is the majority of its AI labels' human statuses (a tie is
    unvalidated). The server's clusters as served are NOT a fair unit for precision: the
    server leaves labels already marked incorrect out of its clustering (measured on all
    three pulls: every AI label outside a cluster has `correct` false), so a ramp the
    validator rejected mostly has no cluster at all and per-cluster precision reads high.
    The second reading puts those labels back: each unclustered AI label is joined to
    every AI label within link_m (the server's CurbRamp threshold), served clusters taken
    as given. That join is single linkage (one loose label can bridge two served
    clusters) where the server's own clustering is complete linkage, and it ignores the
    server's per-region split, so it approximates what the server would have built had it
    kept them.
    """
    st = {q['label_id']: human_status(q.get('validations'), ai_user) for q in ai_labels}
    served, parent = {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in clusters:
        members = [i for i in c.get('label_ids') or () if i in st]
        if not members:
            continue
        served[c['label_cluster_id']] = _majority(st[i] for i in members)
        for i in members:
            parent.setdefault(i, members[0])
        parent.setdefault(members[0], members[0])
    loose = [q for q in ai_labels if q['label_id'] not in parent]
    for q in loose:
        parent[q['label_id']] = q['label_id']
    if loose and ai_labels:
        frame = geo.LocalFrame(sum(q['lat'] for q in ai_labels) / len(ai_labels),
                               sum(q['lng'] for q in ai_labels) / len(ai_labels))
        grid = geo.GridIndex(link_m)
        enu = {}
        for q in ai_labels:
            enu[q['label_id']] = frame.to_enu(q['lat'], q['lng'])
            grid.add(*enu[q['label_id']], q['label_id'])
        for q in loose:
            e, n = enu[q['label_id']]
            for j in grid.near(e, n):
                if math.hypot(e - enu[j][0], n - enu[j][1]) <= link_m:
                    ra, rb = find(q['label_id']), find(j)
                    if ra != rb:
                        parent[ra] = rb
    groups = collections.defaultdict(list)
    for i in st:
        groups[find(i)].append(st[i])
    return served, [_majority(v) for v in groups.values()], len(loose)


def validations_section(ai_labels, ai_user, clusters=(), tiers=None):
    st = collections.Counter()
    votes = collections.Counter()
    voters = collections.Counter()
    n_ai_validator = 0
    n_validated = 0   # labels with at least one human vote, Unsure included
    for q in ai_labels:
        st[human_status(q.get('validations'), ai_user)] += 1
        n = 0
        for v in q.get('validations') or ():
            if v.get('validator_type') == 'AI':
                n_ai_validator += 1
                continue
            if v.get('validator_type') != 'Human':
                continue
            n += 1
            voters[v.get('user_id')] += 1
        votes[min(n, 3)] += 1
        n_validated += n > 0
    ag, dg = st['agreed'], st['disagreed']
    lo, hi = wilson(ag, ag + dg)
    L = ['## 1. Human validations of the AI\'s CurbRamp labels', '',
         'Per label, the majority of human Agree vs Disagree votes; PS\'s own AI validator is '
         'excluded, so this is people judging the model, not a model judging a model. '
         'Precision = agreed / (agreed + disagreed). **Not a random sample:** labels reach '
         'validators through PS\'s validation queue, so this is the precision of the labels '
         'that were shown, and it speaks for the rest only as far as the queue is '
         'representative of them.', '',
         '| AI labels | human-validated | agreed | disagreed | tie / unsure only | precision [95% Wilson] |',
         '|---:|---:|---:|---:|---:|---:|',
         f'| {len(ai_labels)} | {n_validated} | {ag} | {dg} | {n_validated - ag - dg} | '
         f'{(ag / (ag + dg)) if ag + dg else float("nan"):.3f} [{lo:.3f}, {hi:.3f}] |', '',
         f'- Human votes on AI labels: {sum(voters.values())} in all. Per label: 0 votes '
         f'{votes[0]}, 1 vote {votes[1]}, 2 votes {votes[2]}, 3+ votes {votes[3]}.',
         f'- AI-validator votes on AI labels: {n_ai_validator}.',
         f'- Human validators: {len(voters)}. Top: '
         + ', '.join(f'`{u[:8]}…` {n}' for u, n in voters.most_common(5))
         + '. **When one account cast most of the votes, this is one rater\'s precision read, '
           'not a crowd\'s.**']
    served, regrouped, n_loose = cluster_statuses(clusters, ai_labels, ai_user)
    ca = sum(1 for v in regrouped if v == 'agreed')
    cd = sum(1 for v in regrouped if v == 'disagreed')
    sa = sum(1 for v in served.values() if v == 'agreed')
    sd = sum(1 for v in served.values() if v == 'disagreed')
    in_cl = {i for c in clusters for i in c.get('label_ids') or ()}
    loose_dis = sum(1 for q in ai_labels if q['label_id'] not in in_cl
                    and human_status(q.get('validations'), ai_user) == 'disagreed')
    if served:
        L.append(f'- **Per cluster** (the label rate counts a ramp seen from k panos k times; '
                 f'a cluster\'s status is the majority of its AI labels\' human statuses). The '
                 f'server leaves labels already marked incorrect out of its clustering: '
                 f'{n_loose} AI labels sit outside every cluster, {loose_dis} of them '
                 f'disagreed. So its {len(served)} clusters as served read high: agreed {sa}, '
                 f'disagreed {sd}, precision {rate(sa, sa + sd)}. **With those labels put back** '
                 f'(each joined to any AI label within 7.5 m, a single-linkage approximation '
                 f"of the server's 7.5 m complete linkage; {len(regrouped)} groups): agreed {ca}, "
                 f'disagreed {cd}, neither {len(regrouped) - ca - cd}, precision '
                 f'**{rate(ca, ca + cd)}**.')
        seen = collections.Counter(i for c in clusters for i in set(c.get('label_ids') or ()))
        shared = sum(1 for v in seen.values() if v > 1)
        if shared:
            L.append(f'- {shared} label ids appear in more than one served cluster; the put-back '
                     'unions clusters that share a label, so they count as one group there.')
    L.append('')
    by_tier = None
    if tiers is not None:
        keys, files = tiers
        by_tier = {t: collections.Counter() for t in TIERS}
        for q in ai_labels:
            by_tier[label_tier(q, keys)][human_status(q.get('validations'), ai_user)] += 1
        L += ['### 1b. By confidence tier', '',
              'Each AI label joined to the run\'s stored detection on (pano_id, pano_x, pano_y), '
              'the pixel rounding send_to_ps.py used; `ambiguous` = the key carries two tiers '
              'across the files given. Run files: '
              + ', '.join(f'`{n}` (sha256 `{h[:12]}…`)' for n, h in files) + '.', '',
              '| tier | AI labels | agreed | disagreed | precision [95% Wilson] |',
              '|---|---:|---:|---:|---:|']
        for t in TIERS:
            c = by_tier[t]
            n = sum(c.values())
            if not n:
                continue
            a, d = c['agreed'], c['disagreed']
            L.append(f'| {t} | {n} | {a} | {d} | {rate(a, a + d)} |')
        L.append('')
    else:
        L += ['Confidence tier: not split (no `--results` run file given), so the precision '
              'above pools every live AI label, whatever tier it was sent at.', '']
    return L, {'agreed': ag, 'disagreed': dg, 'validated': n_validated,
               'precision_lo': lo, 'precision_hi': hi, 'voters': voters,
               'cluster_agreed': ca, 'cluster_disagreed': cd, 'served_agreed': sa,
               'served_disagreed': sd, 'by_tier': by_tier}


def human_section(cr, ncr, clusters, streets, human, ai_user, out_dir, city='city', rule=''):
    ai_labels = [q for q in cr if q['user_id'] == ai_user]
    ai_ids = {q['label_id'] for q in ai_labels}
    h_cr = [q for q in cr if q['user_id'] == human]
    h_ncr = [q for q in ncr if q['user_id'] == human]
    h_ids = {q['label_id'] for q in h_cr}
    touched = {s['street_edge_id'] for s in streets if human in (s.get('user_ids') or ())}
    done = {s['street_edge_id'] for s in streets
            if human in (s.get('user_ids') or ()) and (s.get('audit_count') or 0) >= 1}
    ai_cl = [c for c in clusters if ai_user in (c.get('users') or ())]
    L = [f'## 2. Human `{human[:8]}…` vs the AI', '']
    if rule:
        L += [rule, '']
    L += [f'- Streets: {len(streets)} in the city; {len(touched)} carry a label of the human\'s; '
          f'**{len(done)} completed** (audit_count ≥ 1 and the human among its users). The audit '
          'is treated as incomplete unless those two numbers are the whole city.',
          f'- The human\'s labels: **{len(h_cr)} CurbRamp**, {len(h_ncr)} NoCurbRamp. '
          f'AI CurbRamp labels: {len(ai_labels)}, in {len(ai_cl)} server clusters '
          f'({sum(1 for c in ai_cl if human in c["users"])} of which also hold a label of the human\'s).',
          '']
    if not h_cr:
        L.append('The human has no CurbRamp label on this server; nothing to compare.')
        return L, {}

    # ---- human -> AI
    cluster_of = {}
    for c in clusters:
        for i in c.get('label_ids') or ():
            if i in h_ids:
                cluster_of[i] = c
    ai_grid = Grid(ai_cl)
    rows = []
    for q in h_cr:
        c = cluster_of.get(q['label_id'])
        in_cl = bool(c and ai_user in c['users'])
        d, near = ai_grid.nearest(q['lng'], q['lat'])
        rows.append((q, in_cl, d, near))
    n = len(h_cr)
    in_cluster = sum(1 for _, ic, _, _ in rows if ic)

    frame = geo.LocalFrame(sum(q['lat'] for q in h_cr) / n, sum(q['lng'] for q in h_cr) / n)

    def pts(items, key):
        return [Pt(it[key], *frame.to_enu(it['lat'], it['lng'])) for it in items]

    h_pts = pts(h_cr, 'label_id')
    targets = {'o2o_cl': pts(ai_cl, 'label_cluster_id'), 'o2o_lab': pts(ai_labels, 'label_id')}

    def count(key, ramps, r, observed):
        if key != 'any':
            return len(match_one_to_one(ramps, targets[key], r))
        if observed:   # membership of the human's own label counts as a hit, as before
            return sum(1 for _, ic, d, _ in rows if ic or d <= r)
        return len(any_hits(ramps, targets['o2o_cl'], r))

    keys_ = ('o2o_cl', 'o2o_lab', 'any')
    moved = displaced(h_pts)
    obs = {(k, r): count(k, h_pts, r, True) for k in keys_ for r in RADII_M}
    floor = {(k, r): count(k, moved, r, False) for k in keys_ for r in RADII_M}
    spread = {}
    for k in keys_:
        vals = [count(k, displaced(h_pts, seed=s), HEADLINE_RADIUS_M, False)
                for s in CHANCE_RANGE_SEEDS]
        spread[k] = (min(vals), max(vals))
    names = {'o2o_cl': 'one-to-one vs AI clusters', 'o2o_lab': 'one-to-one vs raw AI labels',
             'any': 'any AI cluster (holds the label, or within r)'}
    L += ['### 2a. Human → AI: share of the human\'s CurbRamp labels the AI also has', '',
          'Three matchers, because they disagree and the truth sits between the two one-to-one '
          'rows. **Headline: one-to-one vs AI clusters** (agree_rate\'s matcher: greedy by '
          'distance, d ≤ r, a cluster credits at most one label). It is strict where the '
          'server\'s 7.5 m complete linkage merged two ramps at a corner into one cluster. '
          'One-to-one vs raw AI labels is lenient the other way, since a ramp has several AI '
          'views. `any` lets one cluster credit several labels (paired corner ramps), so it is '
          'coverage, not agreement. **Chance** = the same matcher after every human label is '
          f'moved {CHANCE_SHIFT_M:g} m in a random direction (seed {CHANCE_SEED}, as '
          'agree_rate.chance_floor): what street-bound density alone would match.', '',
          '| reading | ' + ' | '.join(f'{r:g} m' for r in RADII_M) + ' |',
          '|---|' + '---:|' * len(RADII_M)]
    for k in keys_:
        label = f'**{names[k]}** (headline)' if k == 'o2o_cl' else names[k]
        L.append(f'| {label} | ' + ' | '.join(rate(obs[(k, r)], n) for r in RADII_M) + ' |')
    for k in keys_:
        L.append(f'| chance: {names[k]} | '
                 + ' | '.join(f'{floor[(k, r)]}/{n} = {floor[(k, r)] / n:.3f}' for r in RADII_M)
                 + ' |')
    L += ['', f'- Chance at {HEADLINE_RADIUS_M:g} m over seeds {CHANCE_RANGE_SEEDS.start}-'
          f'{CHANCE_RANGE_SEEDS.stop - 1}: '
          + '; '.join(f'{names[k]} {a}-{b} of {n} ({a / n:.2f}-{b / n:.2f})'
                      for k, (a, b) in spread.items()) + '.',
          f'- In a server cluster that also holds an AI label (no radius): {rate(in_cluster, n)}.']
    misses = [(q, d, near) for q, ic, d, near in rows if not ic and d > MISS_RADIUS_M]
    L += ['', f'**{len(misses)} of the human\'s CurbRamp labels have no AI cluster within '
          f'{MISS_RADIUS_M:g} m** (candidate AI misses; `misses.csv`; two labels can be one '
          'location): ' + ', '.join(f'`{uid(city, q["label_id"])}`' for q, _, _ in misses) + '.', '']
    with open(out_dir / 'misses.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['label_uid', 'pano_id', 'lat', 'lng', 'street_uid', 'street_completed',
                    'nearest_ai_cluster_m', 'nearest_ai_cluster_uid', 'pano_url'])
        for q, d, near in sorted(misses, key=lambda t: t[0]['label_id']):
            w.writerow([uid(city, q['label_id']), q.get('pano_id'), f'{q["lat"]:.6f}',
                        f'{q["lng"]:.6f}', uid(city, q.get('street_edge_id')),
                        q.get('street_edge_id') in done,
                        f'{d:.1f}' if math.isfinite(d) else '',
                        uid(city, near['label_cluster_id']) if near else '', q.get('pano_url', '')])

    # ---- pano frame (reference), agree_rate's matcher: one-to-one, strictly within
    def norm(q):
        if not (q.get('pano_width') and q.get('pano_height')):
            return None
        if q.get('pano_x') is None or q.get('pano_y') is None:
            return None
        return (q['pano_x'] / q['pano_width'], q['pano_y'] / q['pano_height'])

    ai_by_pano = collections.defaultdict(list)
    for q in ai_labels:
        if norm(q):
            ai_by_pano[q['pano_id']].append(norm(q))
    h_by_pano = collections.defaultdict(list)
    for q in h_cr:
        if q['pano_id'] in ai_by_pano and norm(q):
            h_by_pano[q['pano_id']].append(norm(q))
    same = sum(len(v) for v in h_by_pano.values())
    hit = sum(len(match_pano(v, ai_by_pano[pid], PANO_RADIUS)) for pid, v in h_by_pano.items())
    hit_any = sum(1 for pid, v in h_by_pano.items() for a in v
                  if any(pano_distance(*a, *b) < PANO_RADIUS for b in ai_by_pano[pid]))
    L += ['Pano frame, for reference: '
          f'{same} of {n} of the human\'s labels are on a pano the AI labelled at all; '
          f'{hit} of those {same} match an AI label on that same pano one-to-one, strictly '
          f'within RampNet\'s match radius ({PANO_RADIUS} of the pano width; agree_rate\'s '
          f'matcher), and {hit_any} have any AI label within it. The gap to the world-frame '
          'rows is the AI finding the ramp from a different pano.', '']

    # ---- AI -> human, completed streets
    sub = [c for c in ai_cl if c.get('street_edge_id') in done]
    h_grid, n_grid = Grid(h_cr), Grid(h_ncr)
    per = []
    for c in sub:
        holds = any(i in h_ids for i in c.get('label_ids') or ())
        n_ai = sum(1 for i in c.get('label_ids') or () if i in ai_ids)
        d_cr, _ = h_grid.nearest(c['lng'], c['lat'])
        d_ncr, _ = n_grid.nearest(c['lng'], c['lat'])
        per.append((c, holds, n_ai, d_cr, d_ncr))
    L += ['### 2b. AI → human: AI clusters on the streets the human COMPLETED', '',
          f'{len(sub)} clusters holding an AI label sit on the {len(done)} completed streets. '
          'A cluster is confirmed when it holds the human\'s label or the human has a CurbRamp '
          'label within r. **This is a lower bound on coverage, not a false-positive rate** '
          '(see 2c): an audit is one pass from one pano, and a corner ramp is attributed to one '
          'street edge but may be labelled from another.', '',
          '| test | clusters | share [95% Wilson] | human NoCurbRamp within r instead |',
          '|---|---:|---:|---:|']
    m = len(sub)
    if m:
        k = sum(1 for _, h, _, _, _ in per if h)
        lo, hi = wilson(k, m)
        L.append(f'| holds a label of the human\'s | {k}/{m} | {k / m:.3f} [{lo:.3f}, {hi:.3f}] | |')
        for r in RADII_M:
            k = sum(1 for _, h, _, d, _ in per if h or d <= r)
            nk = sum(1 for _, h, _, d, dn in per if not h and d > r and dn <= r)
            lo, hi = wilson(k, m)
            L.append(f'| … or human CurbRamp within {r:g} m | {k}/{m} | {k / m:.3f} [{lo:.3f}, {hi:.3f}] | {nk} |')
        L += ['', 'By AI views in the cluster (AI labels only; confirmed = holds the label or '
              'human CurbRamp within 7.5 m):', '',
              '| AI views | clusters | confirmed | share [95% Wilson] |', '|---|---:|---:|---:|']
        for name, pred in (('1', lambda v: v == 1), ('2', lambda v: v == 2), ('3+', lambda v: v >= 3)):
            s = [(h, d) for _, h, v, d, _ in per if pred(v)]
            k = sum(1 for h, d in s if h or d <= 7.5)
            lo, hi = wilson(k, len(s))
            L.append(f'| {name} | {len(s)} | {k} | {(k / len(s)) if s else float("nan"):.3f} [{lo:.3f}, {hi:.3f}] |')
        L.append('')
    with open(out_dir / 'clusters_completed_streets.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['cluster_uid', 'street_uid', 'lat', 'lng', 'n_ai_labels',
                    'holds_human_label', 'nearest_human_curbramp_m', 'nearest_human_nocurbramp_m',
                    'human_vote_majority'])
        for c, holds, n_ai, d_cr, d_ncr in sorted(per, key=lambda t: t[0]['label_cluster_id']):
            members = set(c.get('label_ids') or ())
            votes = collections.Counter(own_vote(q, human) for q in ai_labels
                                        if q['label_id'] in members)
            maj = max(votes, key=votes.get) if votes else 'none'
            w.writerow([uid(city, c['label_cluster_id']), uid(city, c.get('street_edge_id')),
                        f'{c["lat"]:.6f}', f'{c["lng"]:.6f}', n_ai, holds,
                        f'{d_cr:.1f}' if math.isfinite(d_cr) else '',
                        f'{d_ncr:.1f}' if math.isfinite(d_ncr) else '', maj])

    # ---- own vote x nearby label
    tab = collections.Counter()
    ai_on_done = [q for q in ai_labels if q.get('street_edge_id') in done]
    for q in ai_on_done:
        d, _ = h_grid.nearest(q['lng'], q['lat'])
        tab[(own_vote(q, human), d <= 7.5)] += 1
    L += ['### 2c. The human\'s own validation vote × whether the human labelled nearby', '',
          f'Over the {len(ai_on_done)} AI labels on completed streets (label level, so a ramp '
          'seen from k panos counts k times). An Agree vote with no label of the human\'s within '
          '7.5 m is a ramp the human accepted when validating and did not place when auditing: '
          'a coverage or attribution gap, not a false positive.', '',
          '| own vote | human CurbRamp within 7.5 m | none within 7.5 m |', '|---|---:|---:|']
    for v in ('agree', 'disagree', 'none'):
        L.append(f'| {v} | {tab[(v, True)]} | {tab[(v, False)]} |')
    L.append('')
    return L, {'n_human': n, 'in_cluster': in_cluster, 'misses': len(misses),
               'completed': len(done), 'clusters_completed': m, 'obs': obs, 'floor': floor,
               'views': {c['label_cluster_id']: v for c, _, v, _, _ in per}}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', help='PS city slug, e.g. laurens (host sidewalk-<city>.cs.washington.edu)')
    ap.add_argument('--human', help='`auto` (the human user with the most CurbRamp labels in the '
                    'snapshot) or a user_id: the auditor to compare against the AI')
    ap.add_argument('--ai', default=AI_USER, help='the auto-labeler\'s user_id')
    ap.add_argument('--results', type=Path, action='append',
                    help='run file(s) the live labels were sent from, to split section 1 by '
                         'confidence tier (repeatable; read only)')
    ap.add_argument('--out', type=Path, help='output dir (default runs/<city>/server_agree)')
    ap.add_argument('--base-url', help='override https://sidewalk-<city>.cs.washington.edu')
    ap.add_argument('--refresh', action='store_true',
                    help='re-pull the four feeds (as served into as_served/, untracked; the '
                         'output dir gets the copies with human user ids redacted)')
    args = ap.parse_args()

    out = args.out or (REPO_ROOT / 'runs' / args.city / 'server_agree')
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url or f'https://sidewalk-{args.city}.cs.washington.edu'
    paths = {key: materialize(out, key, base + rel, args.ai, refresh=args.refresh)
             for key, rel in API.items()}

    cr, d1 = load_features(paths['CurbRamp'])
    ncr, d2 = load_features(paths['NoCurbRamp'])
    clusters, d3 = load_features(paths['clusters'])
    streets = json.loads(paths['streets'].read_text(encoding='utf-8'))['features']
    streets = [s['properties'] for s in streets]
    ai_labels = [q for q in cr if q['user_id'] == args.ai]
    by_user = collections.Counter(q['user_id'] for q in cr)
    sources = collections.Counter(q.get('pano_source') for q in cr)

    L = [f'# {args.city}: server-side read of the AI\'s curb-ramp labels', '',
         f'Tool: `scripts/server_agree_check.py`. Every number is relative to this pull.', '',
         '## Snapshot', '',
         '| file | url | fetched (UTC) | sha256 | features |', '|---|---|---|---|---:|']
    L += [f'| `{a}` | {b} | {c} | `{d}` | {e} |' for a, b, c, d, e in
          (snapshot_row(paths[k]) for k in ('CurbRamp', 'NoCurbRamp', 'clusters', 'streets'))]
    for k in ('CurbRamp', 'NoCurbRamp', 'clusters', 'streets'):
        side = paths[k].parent / (paths[k].name + '.source.json')
        meta = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}
        if meta.get('sha256_as_served'):
            L.append(f'- `{paths[k].name}` is a redacted copy (human user ids cut to 8 '
                     f'characters; no number depends on the full id); as served, sha256 '
                     f'`{meta["sha256_as_served"]}`.')
    L += ['', f'- CurbRamp labels: {len(cr)} ({len(ai_labels)} by the AI `{args.ai[:8]}…`, '
          f'{len(cr) - len(ai_labels)} by {len(by_user) - (1 if args.ai in by_user else 0)} human users); '
          f'pano_source: {dict(sources)}; dropped for a null/corrupt position: '
          f'{d1} CurbRamp, {d2} NoCurbRamp, {d3} clusters.', '']
    tiers = load_tiers(args.results) if args.results else None
    sec, _ = validations_section(ai_labels, args.ai, clusters, tiers)
    L += sec
    if args.human:
        rule, human = '', args.human
        if human == 'auto':
            human, counts = pick_auditor(cr, args.ai)
            if human is None:
                sys.exit('--human auto: no human CurbRamp label in the snapshot')
            ranked = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
            others = ', '.join(f'`{u[:8]}…` {k}' for u, k in ranked[1:4])
            rule = (f'Auditor chosen by rule (`--human auto`): the human user with the most '
                    f'CurbRamp labels in this snapshot, `{human[:8]}…`, with {counts[human]} of '
                    f'the {sum(counts.values())} human CurbRamp labels'
                    + (f' (next: {others})' if others else '') + '. The rule re-derives the '
                    'same account from the pull, so no full id is recorded.')
        sec, _ = human_section(cr, ncr, clusters, streets, human, args.ai, out,
                               city=args.city, rule=rule)
        L += sec
    (out / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    try:
        print('\n'.join(L))
    except UnicodeEncodeError:  # a cp1252 console; the file on disk is UTF-8 regardless
        print('\n'.join(L).encode('ascii', 'replace').decode())
    print(f'\nwrote {out / "report.md"}', file=sys.stderr)


if __name__ == '__main__':
    main()
