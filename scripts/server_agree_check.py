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

2. **One human's labels vs the AI's** (`--human <user_id>`). The AI labels a ramp from
   every pano that sees it; a human auditor labels it once. So the comparison is against
   the server's *clusters* (its own 7.5 m single linkage), in world space, at several
   radii, and a cluster is "confirmed" when it holds the human's label or the human has a
   CurbRamp label within r.
   - human -> AI: share of the human's CurbRamp labels with an AI cluster (all of them).
     Labels with no AI cluster within 10 m are written to misses.csv: candidate AI misses.
   - AI -> human: only clusters on streets the human has COMPLETED an audit of
     (`audit_count >= 1` and the human in the street's `user_ids`), split by how many AI
     views the cluster holds. The human's NoCurbRamp labels nearby are the informative
     disagreement.
   - The human's own validation vote on each AI label crossed with whether the human
     placed a label nearby. This is what says whether "no human label nearby" is a false
     positive or a coverage gap: a label the human voted Agree on and did not re-label is
     the latter.
   - Pano frame, for reference only: the human's labels whose pano the AI also labelled,
     and whether an AI label sits within RampNet's match radius in that same pano.

Caveats the report carries: an audit is one pass from one pano, so AI -> human on
completed streets is a lower bound on coverage, never a precision read; ramps at an
intersection are attributed to one street edge and may be labelled from another; the
human's placement and the cluster centroid carry independent errors, so the 5 m row is
tight and the 10 m row loose.

Stdlib only (plus agree_rate's fetch/snapshot helpers).

Usage:
    python scripts/server_agree_check.py laurens --human 549187e0-82c9-4014-a48d-31f18083d575
    python scripts/server_agree_check.py richmond          # validations only
"""
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agree_rate import fetch, snapshot_row, human_status  # noqa: E402
from agree_rate import PANO_SCALE_X, PANO_SCALE_Y, PANO_RADIUS  # noqa: E402
from eval_sites import wilson  # noqa: E402

AI_USER = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4'   # the auto-labeler's account, every city
RADII_M = (5.0, 7.5, 10.0)
MISS_RADIUS_M = 10.0
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


# ----------------------------------------------------------------------------- geometry

def haversine_m(lng1, lat1, lng2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def pano_dist(a, b):
    """Distance between two labels in the same pano, in RampNet's matcher units
    (x scaled to 1024, y to 512, x cyclic). None when either lacks a pano size."""
    if not (a.get('pano_width') and a.get('pano_height')
            and b.get('pano_width') and b.get('pano_height')):
        return None
    if None in (a.get('pano_x'), a.get('pano_y'), b.get('pano_x'), b.get('pano_y')):
        return None
    xa, ya = a['pano_x'] / a['pano_width'] * PANO_SCALE_X, a['pano_y'] / a['pano_height'] * PANO_SCALE_Y
    xb, yb = b['pano_x'] / b['pano_width'] * PANO_SCALE_X, b['pano_y'] / b['pano_height'] * PANO_SCALE_Y
    dx = abs(xa - xb)
    dx = min(dx, PANO_SCALE_X - dx)
    return math.hypot(dx, ya - yb)


class Grid:
    """Nearest-point lookup over (lng, lat) points, bucketed at ~0.001 deg (~100 m)."""

    def __init__(self, points):
        self.cells = collections.defaultdict(list)
        for p in points:
            self.cells[self._key(p['lng'], p['lat'])].append(p)

    @staticmethod
    def _key(lng, lat):
        return (int(math.floor(lng * 1000)), int(math.floor(lat * 1000)))

    def nearest(self, lng, lat):
        """(distance_m, point) of the nearest point, or (inf, None)."""
        kx, ky = self._key(lng, lat)
        best, best_p = float('inf'), None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for p in self.cells.get((kx + dx, ky + dy), ()):
                    d = haversine_m(lng, lat, p['lng'], p['lat'])
                    if d < best:
                        best, best_p = d, p
        return best, best_p


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


def own_vote(label, user):
    """The one user's own vote on a label: 'agree' / 'disagree' / 'none'."""
    a = d = 0
    for v in label.get('validations') or ():
        if v.get('validator_type') != 'Human' or v.get('user_id') != user:
            continue
        a += v.get('validation') == 'Agree'
        d += v.get('validation') == 'Disagree'
    return 'agree' if a > d else 'disagree' if d > a else 'none'


# ----------------------------------------------------------------------------- sections

def validations_section(ai_labels, ai_user):
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
         'Precision = agreed / (agreed + disagreed).', '',
         '| AI labels | human-validated | agreed | disagreed | tie / unsure only | precision [95% Wilson] |',
         '|---:|---:|---:|---:|---:|---:|',
         f'| {len(ai_labels)} | {n_validated} | {ag} | {dg} | {n_validated - ag - dg} | '
         f'{(ag / (ag + dg)) if ag + dg else float("nan"):.3f} [{lo:.3f}, {hi:.3f}] |', '',
         f'- Human votes per AI label: 0 votes {votes[0]}, 1 vote {votes[1]}, 2 votes '
         f'{votes[2]}, 3+ votes {votes[3]}.',
         f'- AI-validator votes on AI labels: {n_ai_validator}.',
         f'- Human validators: {len(voters)}. Top: '
         + ', '.join(f'`{u[:8]}…` {n}' for u, n in voters.most_common(5))
         + '. **When one account cast most of the votes, this is one rater\'s precision read, '
           'not a crowd\'s.**', '']
    return L, {'agreed': ag, 'disagreed': dg, 'validated': n_validated,
               'precision_lo': lo, 'precision_hi': hi, 'voters': voters}


def human_section(cr, ncr, clusters, streets, human, ai_user, out_dir):
    ai_labels = [q for q in cr if q['user_id'] == ai_user]
    h_cr = [q for q in cr if q['user_id'] == human]
    h_ncr = [q for q in ncr if q['user_id'] == human]
    h_ids = {q['label_id'] for q in h_cr}
    touched = {s['street_edge_id'] for s in streets if human in (s.get('user_ids') or ())}
    done = {s['street_edge_id'] for s in streets
            if human in (s.get('user_ids') or ()) and (s.get('audit_count') or 0) >= 1}
    ai_cl = [c for c in clusters if ai_user in (c.get('users') or ())]
    L = [f'## 2. Human `{human[:8]}…` vs the AI', '',
         f'- Streets: {len(streets)} in the city; {len(touched)} carry a label of the human\'s; '
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
    L += ['### 2a. Human → AI: share of the human\'s CurbRamp labels the AI also has', '',
          '| test | labels | share [95% Wilson] |', '|---|---:|---:|']
    lo, hi = wilson(in_cluster, n)
    L.append(f'| in a server cluster that also holds an AI label | {in_cluster}/{n} | '
             f'{in_cluster / n:.3f} [{lo:.3f}, {hi:.3f}] |')
    for r in RADII_M:
        k = sum(1 for _, ic, d, _ in rows if ic or d <= r)
        lo, hi = wilson(k, n)
        L.append(f'| … or an AI cluster within {r:g} m | {k}/{n} | {k / n:.3f} [{lo:.3f}, {hi:.3f}] |')
    misses = [(q, d, near) for q, ic, d, near in rows if not ic and d > MISS_RADIUS_M]
    L += ['', f'**{len(misses)} of the human\'s ramps have no AI cluster within {MISS_RADIUS_M:g} m** '
          '(candidate AI misses; `misses.csv`): '
          + ', '.join(f'`{q["label_id"]}`' for q, _, _ in misses) + '.', '']
    with open(out_dir / 'misses.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['label_id', 'pano_id', 'lat', 'lng', 'street_edge_id', 'street_completed',
                    'nearest_ai_cluster_m', 'nearest_ai_cluster_id', 'pano_url'])
        for q, d, near in sorted(misses, key=lambda t: t[0]['label_id']):
            w.writerow([q['label_id'], q.get('pano_id'), f'{q["lat"]:.6f}', f'{q["lng"]:.6f}',
                        q.get('street_edge_id'), q.get('street_edge_id') in done,
                        f'{d:.1f}' if math.isfinite(d) else '',
                        near['label_cluster_id'] if near else '', q.get('pano_url', '')])

    # ---- pano frame (reference)
    ai_by_pano = collections.defaultdict(list)
    for q in ai_labels:
        ai_by_pano[q['pano_id']].append(q)
    same = [q for q in h_cr if q['pano_id'] in ai_by_pano]
    hit = sum(1 for q in same if any((d := pano_dist(q, g)) is not None
                                     and d <= PANO_RADIUS * PANO_SCALE_X
                                     for g in ai_by_pano[q['pano_id']]))
    L += ['Pano frame, for reference: '
          f'{len(same)} of {n} of the human\'s labels are on a pano the AI labelled at all; '
          f'in {hit} of those {len(same)} an AI label sits within RampNet\'s match radius '
          f'({PANO_RADIUS} × {PANO_SCALE_X}) in that same pano. The gap to the world-frame '
          'rows is the AI finding the ramp from a different pano.', '']

    # ---- AI -> human, completed streets
    sub = [c for c in ai_cl if c.get('street_edge_id') in done]
    h_grid, n_grid = Grid(h_cr), Grid(h_ncr)
    per = []
    for c in sub:
        holds = any(i in h_ids for i in c.get('label_ids') or ())
        n_ai = sum(1 for i in c.get('label_ids') or () if i not in h_ids)
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
        L += ['', 'By AI views in the cluster (confirmed = holds the label or human CurbRamp within 7.5 m):', '',
              '| AI views | clusters | confirmed | share [95% Wilson] |', '|---|---:|---:|---:|']
        for name, pred in (('1', lambda v: v == 1), ('2', lambda v: v == 2), ('3+', lambda v: v >= 3)):
            s = [(h, d) for _, h, v, d, _ in per if pred(v)]
            k = sum(1 for h, d in s if h or d <= 7.5)
            lo, hi = wilson(k, len(s))
            L.append(f'| {name} | {len(s)} | {k} | {(k / len(s)) if s else float("nan"):.3f} [{lo:.3f}, {hi:.3f}] |')
        L.append('')
    with open(out_dir / 'clusters_completed_streets.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['label_cluster_id', 'street_edge_id', 'lat', 'lng', 'n_ai_labels',
                    'holds_human_label', 'nearest_human_curbramp_m', 'nearest_human_nocurbramp_m',
                    'human_vote_majority'])
        for c, holds, n_ai, d_cr, d_ncr in sorted(per, key=lambda t: t[0]['label_cluster_id']):
            votes = collections.Counter(own_vote(q, human) for q in ai_labels
                                        if q['label_id'] in set(c.get('label_ids') or ()))
            maj = max(votes, key=votes.get) if votes else 'none'
            w.writerow([c['label_cluster_id'], c.get('street_edge_id'), f'{c["lat"]:.6f}',
                        f'{c["lng"]:.6f}', n_ai, holds,
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
               'completed': len(done), 'clusters_completed': m}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', help='PS city slug, e.g. laurens (host sidewalk-<city>.cs.washington.edu)')
    ap.add_argument('--human', help='user_id of the human auditor to compare against the AI')
    ap.add_argument('--ai', default=AI_USER, help='the auto-labeler\'s user_id')
    ap.add_argument('--out', type=Path, help='output dir (default runs/<city>/server_agree)')
    ap.add_argument('--base-url', help='override https://sidewalk-<city>.cs.washington.edu')
    ap.add_argument('--refresh', action='store_true', help='re-pull the four feeds')
    args = ap.parse_args()

    out = args.out or (REPO_ROOT / 'runs' / args.city / 'server_agree')
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url or f'https://sidewalk-{args.city}.cs.washington.edu'
    paths = {}
    for key, rel in API.items():
        paths[key] = out / FILES[key]
        fetch(base + rel, paths[key], refresh=args.refresh)

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
    L += ['', f'- CurbRamp labels: {len(cr)} ({len(ai_labels)} by the AI `{args.ai[:8]}…`, '
          f'{len(cr) - len(ai_labels)} by {len(by_user) - (1 if args.ai in by_user else 0)} human users); '
          f'pano_source: {dict(sources)}; dropped for a null/corrupt position: '
          f'{d1} CurbRamp, {d2} NoCurbRamp, {d3} clusters.', '']
    sec, _ = validations_section(ai_labels, args.ai)
    L += sec
    if args.human:
        sec, _ = human_section(cr, ncr, clusters, streets, args.human, args.ai, out)
        L += sec
    (out / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    try:
        print('\n'.join(L))
    except UnicodeEncodeError:  # a cp1252 console; the file on disk is UTF-8 regardless
        print('\n'.join(L).encode('ascii', 'replace').decode())
    print(f'\nwrote {out / "report.md"}', file=sys.stderr)


if __name__ == '__main__':
    main()
