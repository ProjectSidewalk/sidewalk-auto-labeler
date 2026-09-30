"""Export a corner-level cluster-review bundle for RampNet (RampNet#224).

Samples review units (OSM intersections typed signalised / arterial / residential, and
mid-block windows) under the rule pre-registered in RampNet's
docs/cluster_review_protocol.md, lists every label of the frozen server pull in each 30 m
window with its seed group under two arms (`deployed`, the clusters as served, and
`fusion`, fusion_server+attach built exactly as inventory_clustering.score_server_arms
builds it), and writes into the bundle dir:

  snapshot.json      provenance of every input (url, fetched_at, sha256, counts)
  corners.jsonl      one line per unit (schema in the protocol)
  aerial/            one 512 px north-up Esri World Imagery mosaic per unit (+-35 m)
  crops/             one 45 deg / 512 px crop per label (model-resolution sampling), cut
                     from the PS pano store on makelab2 (or --local-panos)
  crops_missing.csv  labels no crop could be cut for
  export_runs.json   wall-clock of every step of every invocation
  report.md          counts per stratum, no-label share, labels per unit, spacing
                     rejections, provenance, timing, and the reconcile verdict

Resumable: an existing corners.jsonl is never re-sampled (a changed label pull is
refused), existing crops and aerials are never re-made, tiles are cached. The run ends
with a reconcile: every label has a crop or is listed missing, every unit has an aerial.

Network: one Overpass GET (cached in <work-dir>/osm.json), Esri tile GETs (cached in
<work-dir>/tiles/), makelab2 over SSH for crops (read-only; stages under
~/.sidewalk_explorer/ as site_explorer.py does). Nothing is sent to any PS server; the
run dir is read in place and never written. Needs pandas/scipy/haversine (the #56 arms);
no shapely.

    python scripts/export_cluster_review.py vancouver \
        --run-dir ../sal-vancouver/runs/vancouver \
        --bundle ../RampNet/benchmark/vancouver/cluster_review
    # --no-crops / --skip-aerial finish everything else; re-run without them to fill in
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402

SCHEMA_SNAPSHOT = 'rampnet.cluster_review.snapshot/1'
RULE_VERSION = 2          # 2: grade-separated windows excluded (protocol, Amendment 3)
SEED = 224
PER_STRATUM = 20
EMPTY_SHARE = 0.15
WINDOW_M = 30.0
SPACING_M = 60.0
MERGE_NODES_M = 25.0
SIGNAL_RADIUS_M = 30.0
MIDBLOCK_MIN_M = 60.0
MIDBLOCK_STEP_M = 20.0
ELIGIBLE_STREET_M = 20.0
STRATA = ('signalised', 'arterial', 'residential', 'mid_block')
PREFIX = {'signalised': 'sig', 'arterial': 'art', 'residential': 'res', 'mid_block': 'mid'}
RESIDENTIAL = frozenset({'residential', 'living_street', 'unclassified'})
PILOT_QUOTA = {'signalised': 8, 'arterial': 7, 'residential': 8, 'mid_block': 7}
DOUBLE_RATE_SHARE = 0.2
AERIAL_HALF_M = 35.0
AERIAL_PX = 512
AERIAL_ZOOM = 20
CROP_FOV_DEG = 45
CROP_PX = 512
DRAFT_WIDTH = 4096       # JPEG draft decode never below the model's width
# the auto-labeler's street filter (position_check.STREET_HIGHWAY_RE), copied so this
# module imports without position_check's dependencies; a test pins the two equal
STREET_HIGHWAY_RE = ('^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|'
                     'living_street)(_link)?$')
OVERPASS_ENDPOINTS = ('https://overpass-api.de/api/interpreter',
                      'https://overpass.kumi.systems/api/interpreter')
USER_AGENT = 'sidewalk-auto-labeler export_cluster_review (RampNet#224)'
# rule 6b (rule_version 2): a candidate is excluded when its window touches (a) ANY OSM way
# over the ground -- bridge/covered not "no", or layer >= 1: an overhead structure hides the
# ground on the aerial, the reviewer's check -- or (b) a STREET under it -- tunnel not "no",
# or layer <= -1: a corner nobody stands at. A subway or pipe underground is not (b).
GRADE_RULE = ('exclude a candidate whose 30 m window touches any OSM way tagged bridge!=no, '
              'covered!=no or layer>=1, or a street way (STREET_HIGHWAY_RE) tagged tunnel!=no '
              'or layer<=-1')
# where each city's inputs live inside its run dir (the #56 layout)
CITY_INPUTS = {
    'vancouver': {'streets': 'ps_clustering_eval/streets.geojson',
                  'inventory': 'inventory_oracle', 'store': '/projects/makeabilitylab/'
                  'sidewalk_panos/Panoramas/vancouver-wa', 'sharded': True},
}


def round_half_up(x):
    return int(math.floor(x + 0.5))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha1_hex(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def safe_name(key):
    """A file name for an id: ':' is not allowed on Windows."""
    return str(key).replace(':', '_').replace('/', '_')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------------------- OSM

def overpass_query(bbox, pad_deg=0.003):
    min_lng, min_lat, max_lng, max_lat = bbox
    b = f'({min_lat - pad_deg},{min_lng - pad_deg},{max_lat + pad_deg},{max_lng + pad_deg})'
    return (f'[out:json][timeout:180];(way["highway"~"{STREET_HIGHWAY_RE}"]{b};'
            f'node["highway"="traffic_signals"]{b};node["crossing"="traffic_signals"]{b};'
            # rule 6b (a): any way over the ground. (b)'s streets below grade need no clause:
            # the street ways above already come back with their tunnel / layer tags.
            f'way["bridge"]["bridge"!="no"]{b};way["covered"]["covered"!="no"]{b};'
            f'way["layer"~"^[+]?[1-9]"]{b};'
            f');out geom;')


def validate_osm(payload):
    """Reason a payload is unusable, or None (position_check.validate_osm_payload's rule:
    Overpass answers a timeout with HTTP 200 and a `remark`, so that is refused)."""
    if not isinstance(payload, dict) or 'elements' not in payload:
        return 'no elements key in the response'
    if payload.get('remark'):
        return f"Overpass remark: {payload['remark']}"
    if not any(e.get('type') == 'way' for e in payload['elements']):
        return 'zero streets returned'
    return None


def load_or_fetch_osm(path, query):
    """(payload, cached). GET only; a cache is reused when it answers the same query."""
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if validate_osm(payload) is None and (payload.get('_query') or {}).get('query') == query:
            return payload, True
        print(f'-> {path}: stale or invalid cache; refetching')
    last = None
    for endpoint in OVERPASS_ENDPOINTS:
        url = endpoint + '?' + urllib.parse.urlencode({'data': query})
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
                payload = json.loads(r.read().decode('utf-8'))
            problem = validate_osm(payload)
            if problem is None:
                payload['_query'] = {'query': query, 'endpoint': endpoint,
                                     'fetched_at': utc_now()}
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.json.tmp')
                tmp.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
                os.replace(tmp, path)
                return payload, False
            last = f'{endpoint}: {problem}'
        except Exception as exc:  # next mirror
            last = f'{endpoint}: {exc}'
        time.sleep(2)
    raise SystemExit(f'Overpass query failed on every endpoint: {last}')


def street_ways(payload):
    rx = re.compile(STREET_HIGHWAY_RE)
    return [e for e in payload.get('elements', ())
            if e.get('type') == 'way' and e.get('nodes') and e.get('geometry')
            and rx.match((e.get('tags') or {}).get('highway', ''))]


def _on(tags, key):
    v = tags.get(key)
    return v is not None and str(v).strip().lower() != 'no'


def _layer(tags):
    """OSM layer as an int (0 when absent or unparseable; '1;2' reads its first value)."""
    v = tags.get('layer')
    if v is None:
        return 0
    try:
        return int(round(float(str(v).split(';')[0].strip())))
    except ValueError:
        return 0


def grade_reason(tags, is_street):
    """Why a way makes a window grade-separated under rule 6b, or None."""
    if _on(tags, 'bridge'):
        return 'bridge'
    if _on(tags, 'covered'):
        return 'covered'
    if _layer(tags) >= 1:
        return 'layer_above'
    if is_street and _on(tags, 'tunnel'):
        return 'street_tunnel'
    if is_street and _layer(tags) <= -1:
        return 'street_below'
    return None


def grade_separated_lines(payload):
    """[(reason, [(lat, lng), ...])] for every way rule 6b excludes a window over."""
    rx = re.compile(STREET_HIGHWAY_RE)
    out = []
    for e in payload.get('elements', ()):
        if e.get('type') != 'way' or len(e.get('geometry') or ()) < 2:
            continue
        tags = e.get('tags') or {}
        r = grade_reason(tags, bool(rx.match(tags.get('highway', ''))))
        if r:
            out.append((r, [(p['lat'], p['lon']) for p in e['geometry']]))
    return out


def signal_points(payload):
    """[(node_id, lat, lng)] of every traffic-signal node (highway= or crossing=)."""
    out = []
    for e in payload.get('elements', ()):
        t = e.get('tags') or {}
        if e.get('type') == 'node' and (t.get('highway') == 'traffic_signals'
                                        or t.get('crossing') == 'traffic_signals'):
            out.append((e['id'], e['lat'], e['lon']))
    return out


def intersection_nodes(ways):
    """{node_id: {'lat', 'lng', 'legs', 'highways'}} for nodes with >= 3 legs. A way
    passing through a node contributes 2 legs, a way ending there 1."""
    legs, hws, pos = {}, {}, {}
    for w in ways:
        hw = w['tags']['highway']
        ids, geom = w['nodes'], w['geometry']
        last = len(ids) - 1
        for i, (nid, g) in enumerate(zip(ids, geom)):
            pos[nid] = (g['lat'], g['lon'])
            legs[nid] = legs.get(nid, 0) + (1 if i in (0, last) else 2)
            hws.setdefault(nid, set()).add(hw)
    return {nid: {'lat': pos[nid][0], 'lng': pos[nid][1], 'legs': n,
                  'highways': sorted(hws[nid])}
            for nid, n in legs.items() if n >= 3}


def merge_nodes(nodes, radius_m=MERGE_NODES_M):
    """Single-linkage groups of intersection nodes within radius_m: [{'node_ids',
    'lat', 'lng' (centroid), 'highways'}], in ascending first-node-id order."""
    ids = sorted(nodes)
    if not ids:
        return []
    fr = geo.LocalFrame(sum(nodes[i]['lat'] for i in ids) / len(ids),
                        sum(nodes[i]['lng'] for i in ids) / len(ids))
    xy = {i: fr.to_enu(nodes[i]['lat'], nodes[i]['lng']) for i in ids}
    grid = geo.GridIndex(radius_m)
    for i in ids:
        grid.add(*xy[i], i)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in ids:
        e, n = xy[i]
        for j in grid.near(e, n):
            if j != i and math.hypot(xy[j][0] - e, xy[j][1] - n) <= radius_m:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(groups):
        g = groups[root]
        out.append({'node_ids': g,
                    'lat': sum(nodes[i]['lat'] for i in g) / len(g),
                    'lng': sum(nodes[i]['lng'] for i in g) / len(g),
                    'highways': sorted(set().union(*(nodes[i]['highways'] for i in g)))})
    return out


def classify(unit, signal_ids, signals_near):
    """signalised / residential / arterial (protocol rule 4)."""
    if signals_near or any(i in signal_ids for i in unit['node_ids']):
        return 'signalised'
    if set(unit['highways']) <= RESIDENTIAL:
        return 'residential'
    return 'arterial'


class PointIndex:
    """Points in a LocalFrame with a within-radius query (GridIndex underneath)."""

    def __init__(self, fr, pts, cell_m):
        self.fr, self.grid, self.xy = fr, geo.GridIndex(cell_m), []
        for k, (lat, lng) in enumerate(pts):
            e, n = fr.to_enu(lat, lng)
            self.xy.append((e, n))
            self.grid.add(e, n, k)

    def within(self, lat, lng, r):
        e, n = self.fr.to_enu(lat, lng)
        return [k for k in self.grid.near(e, n)
                if math.hypot(self.xy[k][0] - e, self.xy[k][1] - n) <= r]


def midblock_candidates(ways, node_pts, fr, min_m=MIDBLOCK_MIN_M, step_m=MIDBLOCK_STEP_M):
    """[(lat, lng, highway)] every step_m of arc length along each street way, kept when
    more than min_m from every intersection node (node_pts: [(lat, lng)])."""
    idx = PointIndex(fr, node_pts, min_m)
    out = []
    for w in ways:
        pts = [fr.to_enu(g['lat'], g['lon']) for g in w['geometry']]
        carry = 0.0     # distance along the way to the next candidate
        for (e0, n0), (e1, n1) in zip(pts, pts[1:]):
            seg = math.hypot(e1 - e0, n1 - n0)
            if seg == 0:
                continue
            t = carry
            while t <= seg:
                e, n = e0 + (e1 - e0) * t / seg, n0 + (n1 - n0) * t / seg
                lat, lng = fr.to_latlng(e, n)
                if not idx.within(lat, lng, min_m):
                    out.append((lat, lng, w['tags']['highway']))
                t += step_m
            carry = t - seg
    return out


# ------------------------------------------------------------------- eligibility

def point_in_ring(lng, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat) and lng < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def in_area(lat, lng, geom):
    """Point in a (Multi)Polygon, holes respected (no shapely)."""
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    for poly in polys:
        if point_in_ring(lng, lat, poly[0]) and not any(point_in_ring(lng, lat, h)
                                                        for h in poly[1:]):
            return True
    return False


class SegmentIndex:
    """Street segments (ENU) with a nearest-distance-within-r query."""

    def __init__(self, fr, lines, cell_m):
        self.fr, self.cell, self.cells, self.segs = fr, cell_m, {}, []
        for line in lines:
            pts = [fr.to_enu(lat, lng) for lat, lng in line]
            for a, b in zip(pts, pts[1:]):
                k = len(self.segs)
                self.segs.append((a, b))
                x0, x1 = sorted((a[0], b[0]))
                y0, y1 = sorted((a[1], b[1]))
                for cx in range(math.floor(x0 / cell_m), math.floor(x1 / cell_m) + 1):
                    for cy in range(math.floor(y0 / cell_m), math.floor(y1 / cell_m) + 1):
                        self.cells.setdefault((cx, cy), []).append(k)

    def near(self, lat, lng, r):
        e, n = self.fr.to_enu(lat, lng)
        kx, ky = math.floor(e / self.cell), math.floor(n / self.cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in self.cells.get((kx + dx, ky + dy), ()):
                    (ax, ay), (bx, by) = self.segs[k]
                    vx, vy = bx - ax, by - ay
                    L = vx * vx + vy * vy
                    t = 0.0 if L == 0 else max(0.0, min(1.0, ((e - ax) * vx + (n - ay) * vy) / L))
                    if math.hypot(ax + t * vx - e, ay + t * vy - n) <= r:
                        return True
        return False


def open_ps_streets(path):
    """[[(lat, lng), ...]] for every OPEN street of a /v3/api/streets pull."""
    feats = json.loads(Path(path).read_text(encoding='utf-8'))['features']
    out = []
    for f in feats:
        if (f.get('properties') or {}).get('status', 'open') != 'open':
            continue
        g = f['geometry']
        lines = g['coordinates'] if g['type'] == 'MultiLineString' else [g['coordinates']]
        out += [[(c[1], c[0]) for c in line] for line in lines]
    return out


# ---------------------------------------------------------------------- candidates

def build_candidates(payload, fr, area_geom=None, street_index=None,
                     eligible_m=ELIGIBLE_STREET_M, grade_rule=True):
    """(candidates, stats): every eligible unit before sampling, each
    {'type', 'lat', 'lng', 'node_ids', 'highways'}; deterministic order. With
    grade_rule (rule 6b, rule_version 2) a candidate whose window touches a
    grade-separated way is excluded and counted as `grade_separated`."""
    ways = street_ways(payload)
    nodes = intersection_nodes(ways)
    units = merge_nodes(nodes)
    sigs = signal_points(payload)
    sig_ids = {s[0] for s in sigs}
    sig_idx = PointIndex(fr, [(s[1], s[2]) for s in sigs], SIGNAL_RADIUS_M)
    cands = []
    for u in units:
        near = sig_idx.within(u['lat'], u['lng'], SIGNAL_RADIUS_M)
        cands.append({'type': classify(u, sig_ids, near), 'lat': u['lat'], 'lng': u['lng'],
                      'node_ids': u['node_ids'], 'highways': u['highways']})
    node_pts = [(v['lat'], v['lng']) for v in nodes.values()]
    for lat, lng, hw in midblock_candidates(ways, node_pts, fr):
        cands.append({'type': 'mid_block', 'lat': lat, 'lng': lng, 'node_ids': [],
                      'highways': [hw]})
    stats = {'street_ways': len(ways), 'intersection_nodes': len(nodes),
             'intersection_units': len(units), 'signal_nodes': len(sigs),
             'midblock_points': sum(1 for c in cands if c['type'] == 'mid_block'),
             'outside_area': 0, 'off_ps_street': 0, 'grade_separated': 0,
             'grade_separated_by_stratum': {st: 0 for st in STRATA},
             'grade_separated_ways': {}}
    grade = None
    if grade_rule:
        lines = grade_separated_lines(payload)
        for r, _ in lines:
            stats['grade_separated_ways'][r] = stats['grade_separated_ways'].get(r, 0) + 1
        grade = SegmentIndex(fr, [ln for _, ln in lines], WINDOW_M)
    kept = []
    for c in cands:
        if area_geom is not None and not in_area(c['lat'], c['lng'], area_geom):
            stats['outside_area'] += 1
            continue
        if street_index is not None and not street_index.near(c['lat'], c['lng'], eligible_m):
            stats['off_ps_street'] += 1
            continue
        if grade is not None and grade.near(c['lat'], c['lng'], WINDOW_M):
            stats['grade_separated'] += 1
            stats['grade_separated_by_stratum'][c['type']] += 1
            continue
        kept.append(c)
    stats['eligible'] = {s: sum(1 for c in kept if c['type'] == s) for s in STRATA}
    return kept, stats


def count_labels(cands, label_index, window_m=WINDOW_M):
    """Attach 'label_idx' (indices into the label list) to every candidate."""
    for c in cands:
        c['label_idx'] = sorted(label_index.within(c['lat'], c['lng'], window_m))
    return cands


# ------------------------------------------------------------------------ sampling

def draw(cands, city, per_stratum=PER_STRATUM, empty_share=EMPTY_SHARE, seed=SEED,
         spacing_m=SPACING_M):
    """(units, stats): the pre-registered draw (protocol rule 8). One Random(seed)
    shuffles each stratum in STRATA order; labelled units first, then no-label ones;
    a unit is accepted only >= spacing_m from every unit already accepted (any stratum)."""
    rng = random.Random(seed)
    index = geo.LatLngSpacingIndex(spacing_m)
    n_empty = round_half_up(per_stratum * empty_share)
    units, stats = [], {}
    for stratum in STRATA:
        pool = sorted((c for c in cands if c['type'] == stratum),
                      key=lambda c: (round(c['lat'], 7), round(c['lng'], 7),
                                     tuple(c['node_ids'])))
        rng.shuffle(pool)
        st = {'candidates': len(pool), 'candidates_empty': sum(1 for c in pool
                                                               if not c['label_idx'])}
        serial = 0
        for want_labels, k in ((True, per_stratum - n_empty), (False, n_empty)):
            picked, rejected = 0, 0
            for c in pool:
                if picked >= k:
                    break
                if bool(c['label_idx']) != want_labels:
                    continue
                if not index.far_enough(c['lat'], c['lng']):
                    rejected += 1
                    continue
                index.add(c['lat'], c['lng'])
                serial += 1
                picked += 1
                units.append(dict(c, corner_id=f'{city}:{PREFIX[stratum]}:{serial:06d}',
                                  has_labels=want_labels))
            tag = 'labelled' if want_labels else 'empty'
            st[f'{tag}_target'], st[f'{tag}_drawn'] = k, picked
            st[f'{tag}_spacing_rejected'] = rejected
        stats[stratum] = st
    return units, stats


def mark_pilot(units, quota=PILOT_QUOTA, empty_share=EMPTY_SHARE,
               double_share=DOUBLE_RATE_SHARE):
    """Set pilot / double_rate / rater_b_seed in place (protocol rules 9-10)."""
    for u in units:
        u['pilot'], u['double_rate'], u['rater_b_seed'] = False, False, None
    for stratum, q in quota.items():
        su = sorted((u for u in units if u['type'] == stratum),
                    key=lambda u: sha1_hex(u['corner_id']))
        want_e = round_half_up(q * empty_share)
        empties = [u for u in su if not u['has_labels']][:want_e]
        labs = [u for u in su if u['has_labels']][:q - len(empties)]
        for u in empties + labs:
            u['pilot'] = True
    pilot = sorted((u for u in units if u['pilot']), key=lambda u: sha1_hex(u['corner_id']))
    for i, u in enumerate(pilot):
        u['double_rate'] = True
        u['rater_b_seed'] = 'fusion' if i % 2 == 0 else 'deployed'
    rest = sorted((u for u in units if not u['pilot']), key=lambda u: sha1_hex(u['corner_id']))
    for i, u in enumerate(rest[:round_half_up(double_share * len(rest))]):
        u['double_rate'] = True
        u['rater_b_seed'] = 'fusion' if i % 2 == 0 else 'deployed'
    return units


# ------------------------------------------------------------------- label entries

def seed_groups(deployed, fusion):
    """({label_id: [deployed group ids, ascending]}, {label_id: fusion group id}).
    deployed: [(label_cluster_id, [label_id, ...])]; fusion: [[label_id, ...]]."""
    dep = {}
    for cid, lids in sorted(deployed, key=lambda t: t[0]):
        for lab in lids:
            dep.setdefault(int(lab), []).append(f'd{cid}')
    fus = {}
    for k, lids in enumerate(fusion):
        for lab in lids:
            fus.setdefault(int(lab), f'f{k}')
    return dep, fus


def _num(v):
    """A plain int for a present value, None for None / NaN (pandas' missing ints)."""
    if v is None or v != v:
        return None
    return int(v)


def _json_default(o):
    return o.item() if hasattr(o, 'item') else str(o)


def label_entry(row, dep, fus, camera, crop_dir='crops'):
    """One corners.jsonl label from a load_labels row (dict)."""
    lid = int(row['label_id'])
    key = str(lid)
    w, h = _num(row['pano_width']), _num(row['pano_height'])
    d = dep.get(lid, [])
    out = {'key': key, 'label_id': lid, 'pano_id': row['pano_id'],
           'user_kind': row['user_kind'], 'pano_x': int(row['pano_x']),
           'pano_y': int(row['pano_y']), 'pano_width': w, 'pano_height': h,
           'x': round(row['pano_x'] / w, 6) if w else None,
           'y': round(row['pano_y'] / h, 6) if h else None,
           'lat': row['lat'], 'lng': row['lng'], 'camera': camera,
           'capture_date': row.get('image_capture_date'),
           'crop': f'{crop_dir}/{safe_name(key)}.jpg',
           'seed_group': {'deployed': d[0] if d else None, 'fusion': fus.get(lid)}}
    if len(d) > 1:
        out['seed_group_all'] = {'deployed': d}
    return out


# ---------------------------------------------------------------------- aerials

def world_px(lat, lng, zoom=AERIAL_ZOOM):
    n = 2 ** zoom * 256
    return ((lng + 180) / 360 * n,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def world_to_latlng(x, y, zoom=AERIAL_ZOOM):
    n = 2 ** zoom * 256
    lng = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lng


def aerial_window(lat, lng, half_m=AERIAL_HALF_M, zoom=AERIAL_ZOOM):
    """({'x0','y0','x1','y1'} integer world px, bbox dict) of the +-half_m square."""
    fr = geo.LocalFrame(lat, lng)
    x0, y0 = world_px(*fr.to_latlng(-half_m, half_m), zoom)
    x1, y1 = world_px(*fr.to_latlng(half_m, -half_m), zoom)
    w = {'x0': int(math.floor(x0)), 'y0': int(math.floor(y0)),
         'x1': int(math.ceil(x1)), 'y1': int(math.ceil(y1))}
    north, west = world_to_latlng(w['x0'], w['y0'], zoom)
    south, east = world_to_latlng(w['x1'], w['y1'], zoom)
    return w, {'south': south, 'west': west, 'north': north, 'east': east}


def make_aerial(win, out_path, tile_dir, seed_tile_dir=None, zoom=AERIAL_ZOOM, px=AERIAL_PX):
    """Mosaic Esri tiles over win, resize to px x px, save JPEG. Returns tiles fetched."""
    import requests
    from PIL import Image
    import split_figures as sf
    tx0, ty0 = win['x0'] // 256, win['y0'] // 256
    tx1, ty1 = (win['x1'] - 1) // 256, (win['y1'] - 1) // 256
    mosaic = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    fetched = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            path = Path(tile_dir) / f'{zoom}_{tx}_{ty}.jpg'
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                seed = Path(seed_tile_dir) / path.name if seed_tile_dir else None
                if seed is not None and seed.exists():
                    shutil.copyfile(seed, path)
                else:
                    r = requests.get(sf.TILE_URL.format(z=zoom, x=tx, y=ty), timeout=30,
                                     headers={'User-Agent': USER_AGENT})
                    r.raise_for_status()
                    if not r.headers.get('Content-Type', '').startswith('image/'):
                        raise RuntimeError(f'tile {tx},{ty}: not an image')
                    path.write_bytes(r.content)
                    fetched += 1
            with Image.open(path) as t:
                mosaic.paste(t.convert('RGB'), ((tx - tx0) * 256, (ty - ty0) * 256))
    crop = mosaic.crop((win['x0'] - tx0 * 256, win['y0'] - ty0 * 256,
                        win['x1'] - tx0 * 256, win['y1'] - ty0 * 256))
    crop.resize((px, px), Image.LANCZOS).save(out_path, 'JPEG', quality=85)
    return fetched


# ------------------------------------------------------------------------- crops

def crop_items(corners, crops_dir, sharded, fov_deg=CROP_FOV_DEG, px=CROP_PX):
    """Crop-worker items for every label whose crop is not on disk yet (one per key)."""
    items, seen = [], set()
    for c in corners:
        for lab in c['labels']:
            name = Path(lab['crop']).name
            if name in seen or (Path(crops_dir) / name).exists() or lab['x'] is None:
                continue
            seen.add(name)
            it = {'pano_id': lab['pano_id'], 'x': lab['x'], 'y': lab['y'],
                  'fov_deg': fov_deg, 'px': px, 'name': name}
            if sharded:
                it['path'] = f"{lab['pano_id'][:2]}/{lab['pano_id']}.jpg"
            items.append(it)
    return items


def read_missing(path):
    if not Path(path).exists():
        return {}
    with open(path, newline='', encoding='utf-8') as f:
        return {r['key']: r for r in csv.DictReader(f)}


def write_missing(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['key', 'pano_id', 'reason'])
        w.writeheader()
        for k in sorted(rows, key=lambda k: (len(k), k)):
            w.writerow({'key': k, 'pano_id': rows[k]['pano_id'], 'reason': rows[k]['reason']})


# ------------------------------------------------------------------------- report

def reconcile(corners, bundle):
    """(problems, counts): every label has a crop or a crops_missing.csv row, every unit
    an aerial; files with no label behind them are listed too."""
    missing = read_missing(bundle / 'crops_missing.csv')
    keys, problems = set(), []
    n_crop = n_missing = 0
    for c in corners:
        if not (bundle / c['aerial']['file']).exists():
            problems.append(f"{c['corner_id']}: no aerial")
        for lab in c['labels']:
            keys.add(Path(lab['crop']).name)
            if (bundle / lab['crop']).exists():
                n_crop += 1
            elif lab['key'] in missing:
                n_missing += 1
            else:
                problems.append(f"label {lab['key']}: no crop and not in crops_missing.csv")
    crop_dir = bundle / 'crops'
    extra = sorted(p.name for p in crop_dir.glob('*.jpg') if p.name not in keys) \
        if crop_dir.exists() else []
    if extra:
        problems.append(f'{len(extra)} crop file(s) with no label, e.g. {extra[:3]}')
    return problems, {'crops': n_crop, 'crops_missing': n_missing,
                      'aerials': sum(1 for c in corners
                                     if (bundle / c['aerial']['file']).exists())}


def quantiles(xs, ps=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] for p in ps}


def write_report(bundle, snapshot, corners, draw_stats, cand_stats, runs, problems, counts,
                 copy_to=None):
    lines = [f"# {snapshot['city']}: cluster-review bundle (RampNet#224)", '',
             f"Exported by `{snapshot['exporter']}`; sampling rule v{RULE_VERSION} "
             '(RampNet docs/cluster_review_protocol.md). No review has been done: this bundle '
             'holds no assignments.', '', '## Provenance', '']
    lab = snapshot['labels']
    lines.append(f"- labels: `{lab['path_in_run']}`, {lab['n_features']} features, sha256 "
                 f"`{lab['sha256']}`, fetched {lab['fetched_at']} from {lab['url']}")
    dep = snapshot['seed_arms']['deployed']
    lines.append(f"- deployed seed: `{dep['path_in_run']}`, {dep['n_features']} clusters, sha256 "
                 f"`{dep['sha256']}`, fetched {dep['fetched_at']} from {dep['url']}")
    fu = snapshot['seed_arms']['fusion']
    lines.append(f"- fusion seed: {fu['definition']}; {fu['n_clusters']} clusters; results.jsonl "
                 f"sha256 `{fu['results_sha256']}`; state.pkl cross-check: {fu.get('cross_check')}")
    o = snapshot['osm']
    lines.append(f"- OSM: {o['n_ways']} street ways, {o['n_signal_nodes']} signal nodes, fetched "
                 f"{o['fetched_at']} from {o['endpoint']}, payload sha256 `{o['sha256']}`")
    s = snapshot['ps_streets']
    lines.append(f"- PS streets (eligibility): `{s['path_in_run']}`, sha256 `{s['sha256']}`, "
                 f"fetched {s['fetched_at']}")
    if snapshot.get('inventory'):
        inv = snapshot['inventory']
        lines.append(f"- inventory: {inv['n_kept']} kept points, sha256 `{inv['sha256']}`, "
                     f"fetched {inv['fetched_at']}")
    a = snapshot['aerial']
    lines.append(f"- aerial: {a['source']} z{a['zoom']}, +-{a['half_m']:g} m at {a['px']} px; "
                 f"{a['attribution']}")
    cr = snapshot['crops']
    lines.append(f"- crops: {cr['fov_deg']} deg / {cr['px']} px, {cr['resolution']}, from "
                 f"{cr['store']}")
    lines += ['', '## Candidates', '', '| step | count |', '|---|---:|']
    for k in ('street_ways', 'intersection_nodes', 'intersection_units', 'signal_nodes',
              'midblock_points', 'outside_area', 'off_ps_street', 'grade_separated'):
        lines.append(f'| {k} | {cand_stats.get(k)} |')
    if 'grade_separated' in cand_stats:
        gs, by = cand_stats['grade_separated'], cand_stats.get('grade_separated_by_stratum') or {}
        elig = cand_stats.get('eligible') or {}
        pre = gs + sum(elig.values())
        per = ', '.join(f'{st} {by.get(st, 0)}/{by.get(st, 0) + elig.get(st, 0)}' for st in STRATA)
        share = f' ({gs / pre:.1%})' if pre else ''
        lines += ['', f'Rule 6b ({GRADE_RULE}) removed {gs} of {pre} otherwise eligible '
                  f'candidates{share}; by stratum {per}; structures '
                  f"{cand_stats.get('grade_separated_ways')}. The study's scope is corners at grade."]
    lines += ['', '| stratum | eligible | of them no-label | labelled drawn / target | '
              'no-label drawn / target | spacing rejections (labelled, no-label) |',
              '|---|---:|---:|---|---|---|']
    for st in STRATA:
        d = draw_stats.get(st, {})
        lines.append(f"| {st} | {d.get('candidates')} | {d.get('candidates_empty')} | "
                     f"{d.get('labelled_drawn')} / {d.get('labelled_target')} | "
                     f"{d.get('empty_drawn')} / {d.get('empty_target')} | "
                     f"{d.get('labelled_spacing_rejected')}, {d.get('empty_spacing_rejected')} |")
    lines += ['', '## Units', '', '| stratum | units | no-label | pilot | double-rated | '
              'labels | human labels |', '|---|---:|---:|---:|---:|---:|---:|']
    for st in STRATA + ('all',):
        us = [c for c in corners if st == 'all' or c['type'] == st]
        lines.append(f"| {st} | {len(us)} | {sum(1 for c in us if not c['has_labels'])} | "
                     f"{sum(1 for c in us if c['pilot'])} | "
                     f"{sum(1 for c in us if c['double_rate'])} | "
                     f"{sum(c['n_labels'] for c in us)} | "
                     f"{sum(1 for c in us for lab in c['labels'] if lab['user_kind'] == 'human')} |")
    n = len(corners)
    lines.append('')
    lines.append(f"- no-label share: {sum(1 for c in corners if not c['has_labels'])}/{n}"
                 + (f" = {sum(1 for c in corners if not c['has_labels']) / n:.3f}" if n else ''))
    for tag, us in (('all units', corners), ('labelled units', [c for c in corners if c['has_labels']]),
                    ('pilot units', [c for c in corners if c['pilot']])):
        q = quantiles([c['n_labels'] for c in us])
        lines.append(f'- labels per unit ({tag}): ' + ', '.join(
            f'p{int(p * 100)} {v}' for p, v in q.items()))
    for arm in ('deployed', 'fusion'):
        g = [len({lab['seed_group'][arm] for lab in c['labels'] if lab['seed_group'][arm]})
             for c in corners if c['has_labels']]
        nulls = sum(1 for c in corners for lab in c['labels'] if not lab['seed_group'][arm])
        lines.append(f'- {arm} seed groups per labelled unit: '
                     + ', '.join(f'p{int(p * 100)} {v}' for p, v in quantiles(g).items())
                     + f'; labels with no {arm} group: {nulls}')
    multi = sum(1 for c in corners for lab in c['labels'] if 'seed_group_all' in lab)
    lines.append(f'- labels held by two or more deployed clusters: {multi}')
    rb = {k: sum(1 for c in corners if c['rater_b_seed'] == k) for k in ('fusion', 'deployed')}
    lines.append(f"- rater_b_seed: fusion {rb['fusion']}, deployed {rb['deployed']}")
    lines += ['', '## Files and reconcile', '',
              f"- crops on disk {counts['crops']}, listed missing {counts['crops_missing']}, "
              f"aerials {counts['aerials']} / {n}",
              f"- STATUS: {'OK -- every label has a crop or a missing row, every unit an aerial' if not problems else 'INCOMPLETE -- ' + str(len(problems)) + ' problem(s)'}"]
    lines += [f'  - {p}' for p in problems[:20]]
    lines += ['', '## Timing (wall clock, every invocation)', '',
              '| started (UTC) | step | seconds | note |', '|---|---|---:|---|']
    for r in runs:
        for st in r['steps']:
            lines.append(f"| {r['started_at']} | {st['step']} | {st['seconds']:.1f} | "
                         f"{st.get('note', '')} |")
    text = '\n'.join(lines) + '\n'
    (bundle / 'report.md').write_text(text, encoding='utf-8')
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        (copy_to / 'report.md').write_text(text, encoding='utf-8')
    return text


# ---------------------------------------------------------------------------- main

def git_sha():
    try:
        return subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return 'unknown'


def source_json(path):
    side = Path(str(path) + '.source.json')
    return json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}


def load_inventory_points(run_dir, city):
    """[(lat, lng, unit_id)] of the kept inventory ramps and the record, read in place."""
    import inventory_oracle as ioracle
    cfg = CITY_INPUTS.get(city, {})
    if not cfg.get('inventory'):
        return None, None
    d = Path(run_dir) / cfg['inventory']
    geo_path, rec_path = d / 'inventory.geojson', d / 'inventory.json'
    if not geo_path.exists() or not rec_path.exists():
        return None, None
    record = json.loads(rec_path.read_text(encoding='utf-8'))
    ioracle.check_cache(geo_path, record)
    feats = json.loads(geo_path.read_text(encoding='utf-8'))['features']
    pts = [(f['geometry']['coordinates'][1], f['geometry']['coordinates'][0],
            (f.get('properties') or {}).get('UNITID'))
           for f in ioracle.kept(feats, ioracle.INVENTORIES[city])]
    return pts, record


def build_bundle(args, run_dir, bundle, work, steps):
    """Sample and write snapshot.json + corners.jsonl (the first invocation only)."""
    import eval_ps_clustering as epc
    import inventory_clustering as ic
    cfg = ic.SERVER_LABELS[args.city]
    inputs = CITY_INPUTS[args.city]
    area = json.loads((run_dir / 'area.geojson').read_text(encoding='utf-8'))

    t = time.time()
    labels_path = run_dir / cfg['labels']
    labels, arms, st = ic.server_arms(run_dir, cfg, frame=args.frame, thresholds_m=(7.5,))
    server_clusters = epc.load_server_clusters(run_dir / cfg['clusters'])
    dep, fus = seed_groups([(sc['id'], sc['label_ids']) for sc in server_clusters],
                           arms['fusion_server+attach'])
    cross = 'no state.pkl'
    state_pkl = run_dir / 'split_figures' / 'state.pkl'
    if state_pkl.exists():
        import pickle
        state = pickle.loads(state_pkl.read_bytes())

        def part(cl):
            return sorted(tuple(sorted(int(x) for x in c)) for c in cl)
        ok = all(part(arms[m]) == part([c['label_ids'] for c in state['arms'][s]['clusters']])
                 for m, s in (('deployed', 'deployed'), ('ps @ 7.5 m', 'ps@7.5'),
                              ('fusion_server+attach', 'fusion+attach')))
        cross = 'identical partitions (deployed, ps @ 7.5 m, fusion+attach)' if ok \
            else 'MISMATCH against state.pkl'
        if not ok:
            raise SystemExit('the rebuilt arms do not match split_figures/state.pkl')
    steps.append({'step': 'arms', 'seconds': time.time() - t,
                  'note': f"{len(labels)} labels; fusion+attach {len(arms['fusion_server+attach'])} clusters"})

    t = time.time()
    query = overpass_query(__import__('position_check').area_bbox(area))
    payload, cached = load_or_fetch_osm(work / 'osm.json', query)
    steps.append({'step': 'overpass', 'seconds': time.time() - t,
                  'note': 'cached' if cached else 'GET'})

    t = time.time()
    rows = labels.to_dict('records')
    for r in rows:
        r['user_kind'] = 'ai' if str(r['user_id']) == cfg['ai_user'] else 'human'
    fr = geo.LocalFrame(float(labels.lat.mean()), float(labels.lng.mean()))
    label_index = PointIndex(fr, [(r['lat'], r['lng']) for r in rows], WINDOW_M)
    streets_path = run_dir / inputs['streets']
    street_index = SegmentIndex(fr, open_ps_streets(streets_path), ELIGIBLE_STREET_M)
    cands, cand_stats = build_candidates(payload, fr, area, street_index)
    count_labels(cands, label_index)
    units, draw_stats = draw(cands, args.city, args.per_stratum, EMPTY_SHARE, args.seed)
    mark_pilot(units)
    steps.append({'step': 'sample', 'seconds': time.time() - t,
                  'note': f'{len(cands)} eligible candidates -> {len(units)} units'})

    inv_pts, inv_rec = load_inventory_points(run_dir, args.city)
    inv_index = PointIndex(fr, [(p[0], p[1]) for p in inv_pts], WINDOW_M) if inv_pts else None
    run_pos = st['run_pos']
    by_pano = {}
    for r in rows:
        by_pano.setdefault(r['pano_id'], []).append(r)
    cams = {}

    def camera(r):
        pid = r['pano_id']
        if pid not in cams:
            if pid in run_pos:
                lat, lng, _h = run_pos[pid]
                cams[pid] = (lat, lng, 'run')
            else:
                inv = epc.invert_camera_position(by_pano[pid])
                cams[pid] = (inv[0], inv[1], 'inverted') if inv else (None, None, 'none')
        lat, lng, src = cams[pid]
        return {'lat': lat, 'lng': lng, 'heading_deg': r['camera_heading'], 'source': src}

    corners = []
    for u in units:
        win, bbox = aerial_window(u['lat'], u['lng'])
        labs = [label_entry(rows[k], dep, fus, camera(rows[k])) for k in u['label_idx']]
        labs.sort(key=lambda lab: lab['label_id'])
        line = {'corner_id': u['corner_id'], 'city': args.city, 'type': u['type'],
                'centre': {'lat': u['lat'], 'lng': u['lng']}, 'window_m': WINDOW_M,
                'node_ids': u['node_ids'], 'highways': u['highways'],
                'has_labels': u['has_labels'], 'n_labels': len(labs), 'pilot': u['pilot'],
                'double_rate': u['double_rate'], 'rater_b_seed': u['rater_b_seed'],
                'aerial': {'file': f"aerial/{safe_name(u['corner_id'])}.jpg", 'px': AERIAL_PX,
                           'north_up': True, 'zoom': AERIAL_ZOOM, 'world_px': win,
                           'bbox': bbox, 'source': 'Esri World Imagery'},
                'labels': labs}
        if inv_index is not None:
            line['inventory'] = [{'lat': inv_pts[k][0], 'lng': inv_pts[k][1],
                                  'unit_id': inv_pts[k][2]}
                                 for k in sorted(inv_index.within(u['lat'], u['lng'], WINDOW_M))]
        corners.append(line)

    import split_figures as sf
    lab_src, cl_src = source_json(labels_path), source_json(run_dir / cfg['clusters'])
    st_src = source_json(streets_path)
    snapshot = {
        'schema': SCHEMA_SNAPSHOT, 'city': args.city,
        'labels': {'kind': 'server_pull', 'path_in_run': cfg['labels'],
                   'url': lab_src.get('url'), 'fetched_at': lab_src.get('fetched_at'),
                   'sha256': sha256_file(labels_path), 'n_features': st['n_labels'],
                   'ai_user_id': cfg['ai_user'], 'tier': cfg['tier'],
                   'n_ai': st['n_ai'], 'n_human': st['n_labels'] - st['n_ai']},
        'seed_arms': {
            'deployed': {'path_in_run': cfg['clusters'], 'url': cl_src.get('url'),
                         'fetched_at': cl_src.get('fetched_at'),
                         'sha256': sha256_file(run_dir / cfg['clusters']),
                         'n_features': len(server_clusters)},
            'fusion': {'definition': f'fusion_server+attach, {args.frame} frame, '
                                     'inventory_clustering.server_arms (= score_server_arms)',
                       'results_sha256': sha256_file(run_dir / 'results.jsonl'),
                       'n_clusters': len(arms['fusion_server+attach']),
                       'n_attached': st['n_attached'], 'height': st['height'],
                       'cross_check': cross}},
        'osm': {'query': query, 'endpoint': payload['_query'].get('endpoint'),
                'fetched_at': payload['_query'].get('fetched_at'),
                'sha256': sha256_file(work / 'osm.json'),
                'n_ways': cand_stats['street_ways'], 'n_signal_nodes': cand_stats['signal_nodes']},
        'ps_streets': {'path_in_run': inputs['streets'], 'url': st_src.get('url'),
                       'fetched_at': st_src.get('fetched_at'),
                       'sha256': sha256_file(streets_path)},
        'aerial': {'source': 'Esri World Imagery', 'url_template': sf.TILE_URL,
                   'zoom': AERIAL_ZOOM, 'half_m': AERIAL_HALF_M, 'px': AERIAL_PX,
                   'attribution': sf.ATTRIBUTION},
        'crops': {'fov_deg': CROP_FOV_DEG, 'px': CROP_PX,
                  'resolution': 'model (4096x2048-equivalent): native crop resampled to '
                                f'{CROP_PX} px; JPEG draft decode never below {DRAFT_WIDTH} px',
                  'store': ('makelab2 ' + inputs['store']) if inputs.get('store') else None},
        'sampling': {'seed': args.seed, 'per_stratum': args.per_stratum,
                     'empty_share': EMPTY_SHARE, 'window_m': WINDOW_M, 'spacing_m': SPACING_M,
                     'merge_nodes_m': MERGE_NODES_M, 'signal_radius_m': SIGNAL_RADIUS_M,
                     'midblock_min_m': MIDBLOCK_MIN_M, 'midblock_step_m': MIDBLOCK_STEP_M,
                     'eligible_street_m': ELIGIBLE_STREET_M, 'pilot_quota': PILOT_QUOTA,
                     'double_rate_share': DOUBLE_RATE_SHARE, 'rule_version': RULE_VERSION,
                     'grade_separation': {'rule': GRADE_RULE, 'window_m': WINDOW_M},
                     'candidates': cand_stats, 'draw': draw_stats},
        'exported_at': utc_now(),
        'exporter': f'sidewalk-auto-labeler scripts/export_cluster_review.py@{git_sha()}'}
    if inv_rec:
        snapshot['inventory'] = {'path_in_run': inputs['inventory'] + '/inventory.geojson',
                                 'url': inv_rec.get('url'), 'sha256': inv_rec.get('sha256'),
                                 'fetched_at': inv_rec.get('fetched_utc'),
                                 'n_kept': len(inv_pts)}
    bundle.mkdir(parents=True, exist_ok=True)
    tmp = bundle / 'corners.jsonl.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        for c in corners:
            f.write(json.dumps(c, separators=(',', ':'), default=_json_default) + '\n')
    (bundle / 'snapshot.json').write_text(json.dumps(snapshot, indent=1,
                                                     default=_json_default) + '\n',
                                          encoding='utf-8', newline='\n')
    os.replace(tmp, bundle / 'corners.jsonl')
    return snapshot, corners


def load_corners(bundle):
    with open(bundle / 'corners.jsonl', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city', choices=sorted(CITY_INPUTS))
    ap.add_argument('--run-dir', type=Path, required=True,
                    help='run directory holding the frozen pulls and results.jsonl (read only)')
    ap.add_argument('--bundle', type=Path, required=True,
                    help='output dir, e.g. ../RampNet/benchmark/<city>/cluster_review')
    ap.add_argument('--work-dir', type=Path, default=None,
                    help='osm.json + tiles/ cache and the report copy '
                         '(default runs/<city>/cluster_review in this checkout)')
    ap.add_argument('--per-stratum', type=int, default=PER_STRATUM)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--frame', default='auto', help='fuse frame of the fusion seed')
    ap.add_argument('--no-crops', action='store_true', help='skip the crop step')
    ap.add_argument('--skip-aerial', action='store_true', help='skip the aerial step')
    ap.add_argument('--local-panos', type=Path, default=None,
                    help='cut crops locally from this pano dir instead of makelab2')
    ap.add_argument('--flat', action='store_true',
                    help='the pano dir is flat <id>.jpg (default for the city: see CITY_INPUTS)')
    ap.add_argument('--host', default='makelab2')
    ap.add_argument('--helper', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=5400)
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    bundle = args.bundle.resolve()
    work = (args.work_dir or REPO_ROOT / 'runs' / args.city / 'cluster_review').resolve()
    if work == run_dir or run_dir in work.parents:
        raise SystemExit('--work-dir must not be inside the run dir (it is read only)')
    work.mkdir(parents=True, exist_ok=True)
    runs_path = bundle / 'export_runs.json'
    runs = json.loads(runs_path.read_text(encoding='utf-8')) if runs_path.exists() else []
    this = {'started_at': utc_now(), 'argv': sys.argv[1:] if argv is None else list(argv),
            'steps': []}
    steps = this['steps']
    inputs = CITY_INPUTS[args.city]

    if (bundle / 'corners.jsonl').exists():
        snapshot = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
        import inventory_clustering as ic
        cur = sha256_file(run_dir / ic.SERVER_LABELS[args.city]['labels'])
        if cur != snapshot['labels']['sha256']:
            raise SystemExit(f"{run_dir}: the label pull's sha256 {cur[:12]} is not the "
                             f"bundle's {snapshot['labels']['sha256'][:12]}; refusing to mix")
        corners = load_corners(bundle)
        print(f'reusing {len(corners)} units from {bundle / "corners.jsonl"} (never re-sampled)')
    else:
        snapshot, corners = build_bundle(args, run_dir, bundle, work, steps)
        print(f'sampled {len(corners)} units -> {bundle / "corners.jsonl"}')

    if not args.skip_aerial:
        t = time.time()
        (bundle / 'aerial').mkdir(parents=True, exist_ok=True)
        fetched = made = 0
        seed_tiles = run_dir / 'split_figures' / 'tiles'
        for c in corners:
            out = bundle / c['aerial']['file']
            if out.exists():
                continue
            fetched += make_aerial(c['aerial']['world_px'], out, work / 'tiles',
                                   seed_tiles if seed_tiles.exists() else None)
            made += 1
        steps.append({'step': 'aerial', 'seconds': time.time() - t,
                      'note': f'{made} made, {fetched} tiles fetched'})
        print(f'aerials: {made} made, {fetched} tiles fetched')

    if not args.no_crops:
        t = time.time()
        crops_dir = bundle / 'crops'
        crops_dir.mkdir(parents=True, exist_ok=True)
        sharded = inputs.get('sharded', False) and not args.flat
        items = crop_items(corners, crops_dir, sharded)
        missing_rows = read_missing(bundle / 'crops_missing.csv')
        items = [it for it in items if Path(it['name']).stem not in
                 {safe_name(k) for k in missing_rows}]
        key_of = {Path(lab['crop']).name: lab for c in corners for lab in c['labels']}
        note = 'nothing to cut'
        if items:
            import site_explorer as se
            job = {'items': items, 'workers': args.workers, 'draft_width': DRAFT_WIDTH}
            if args.local_panos:
                job['panos_root'] = str(args.local_panos)
                missing = se.fetch_crops_local(job, crops_dir, args.local_panos)
                where = f'local {args.local_panos}'
            else:
                job['panos_root'] = inputs['store']
                missing = se.fetch_crops_remote(job, crops_dir, args.helper or se.DEFAULT_HELPER,
                                                args.host, se.DEFAULT_REMOTE_ROOT,
                                                se.DEFAULT_REMOTE_HOME, args.timeout)
                where = f"{args.host}:{inputs['store']}"
            for name in missing:
                lab = key_of[name]
                missing_rows[lab['key']] = {'pano_id': lab['pano_id'],
                                            'reason': f'no readable pano jpeg in {where}'}
            write_missing(bundle / 'crops_missing.csv', missing_rows)
            note = f'{len(items)} requested, {len(missing)} missing, from {where}'
        elif not (bundle / 'crops_missing.csv').exists():
            write_missing(bundle / 'crops_missing.csv', missing_rows)
        steps.append({'step': 'crops', 'seconds': time.time() - t, 'note': note})
        print(f'crops: {note}')

    problems, counts = reconcile(corners, bundle)
    runs.append(this)
    runs_path.write_text(json.dumps(runs, indent=1) + '\n', encoding='utf-8', newline='\n')
    snapshot_stats = snapshot['sampling']
    write_report(bundle, snapshot, corners, snapshot_stats.get('draw', {}),
                 snapshot_stats.get('candidates', {}), runs, problems, counts, copy_to=work)
    print(f"reconcile: {counts}; {'OK' if not problems else f'{len(problems)} problem(s)'}")
    for p in problems[:10]:
        print('  ' + p)
    print(f'wrote {bundle / "report.md"} (copy in {work})')
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
