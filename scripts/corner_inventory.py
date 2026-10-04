"""Corner-level ramp inventory: present / absent / unobservable per OSM corner (RampNet#238).

Protocol (written before any number): docs/corner-inventory.md. Four subcommands:

  build              every eligible #224 intersection unit (no sampling) and mid-block point,
                     its legs and corners, panos / operational fused sites / deployed clusters /
                     inventory points per corner, and the state under both presence arms and
                     three observability rules -> <out>/corners_full.jsonl (+ units.csv,
                     corners.csv, corners_224.jsonl / .csv for the 80 #224 units, build.json)
  score              the tables (city sentence, corner recall, absence precision, inventory
                     gaps) and the decision rule -> <out>/report.md, counts.csv, gaps.csv
  score-assignments  a #224 rater's assignments.json against corners_224.jsonl
  census             GSV history for the panos near a stratified sample of 200 units: one
                     metadata request per pano, no image -> <out>/census/

Inputs are read in place and never written; results.jsonl and the OSM payload are asserted
by sha256. The OSM / eligibility functions are imported from export_cluster_review.py (the
#224 exporter), so a unit here is the unit #224 samples from. No pandas: stdlib + geo.

    python scripts/corner_inventory.py build \\
        --run-dir ../sal-vancouver/runs/vancouver \\
        --osm ../sal-cluster-review/runs/vancouver/cluster_review/osm.json \\
        --units224 ../RampNet/benchmark/vancouver/cluster_review/corners.jsonl \\
        --out runs/vancouver/corner_inventory
    python scripts/corner_inventory.py score --out runs/vancouver/corner_inventory
    python scripts/corner_inventory.py census --out runs/vancouver/corner_inventory
    python scripts/corner_inventory.py score-assignments \\
        --assignments ../RampNet/benchmark/vancouver/cluster_review/assignments.json \\
        --out runs/vancouver/corner_inventory
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import export_cluster_review as ecr  # noqa: E402

SCHEMA = 'sidewalk-auto-labeler.corner_inventory/1'
RESULTS_SHA256 = {'vancouver': '7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28'}
OSM_SHA256 = {'vancouver': '58ce83ffa4b8eea199d7017b53fe125079e2cfa2f3d5181f9e22d61bc0c2a030'}
# the exporter's snapshot.json `eligible` counts (rule v2); the build must reproduce them
EXPECTED_ELIGIBLE = {'vancouver': {'signalised': 284, 'arterial': 1497, 'residential': 2648,
                                   'mid_block': 16912}}

LEG_PROBE_M = 20.0
LEG_STUB_MIN_M = 5.0
LEG_MERGE_DEG = 30.0
CORNER_OFFSET_M = 12.0
WIDE_SECTOR_DEG = 150.0
OBS_NEAR_M = 15.0
INTERSECTION_STRATA = ('signalised', 'arterial', 'residential')
ARMS = ('fusion', 'deployed')
VARIANTS = ('primary', 'ge2', 'le15')
STATES = ('present', 'absent', 'unobservable')
INV_CLASSES = ('Available', 'NA_noramp', 'NA_typed', 'RMV', 'Expired/Removed', 'other')
INV_ATTRS = ('INSTDATE', 'PROJNAME', 'DEFECT', 'CORNER', 'OWNER')
CENSUS_SEED = 238
CENSUS_N = {'signalised': 67, 'arterial': 67, 'residential': 66}
CENSUS_SLEEP_S = 0.25
DECISION_THRESHOLD = 0.90


# ------------------------------------------------------------------------- helpers

def sha256_file(path):
    return ecr.sha256_file(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def git_sha(path=None):
    try:
        args = ['git', '-C', str(REPO_ROOT), 'log', '-1', '--format=%H']
        if path:
            args += ['--', str(path)]
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # pragma: no cover
        return 'unknown'


def wilson(k, n, z=1.96):
    """Wilson score interval for k of n; None when n == 0.

    >>> [round(x, 3) for x in wilson(9, 10)]
    [0.596, 0.982]
    """
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def r6(x):
    return None if x is None else round(float(x), 6)


def r2(x):
    return None if x is None else round(float(x), 2)


def bearing_deg(de, dn):
    """Compass bearing (deg, clockwise from north, [0, 360)) of an ENU offset."""
    return math.degrees(math.atan2(de, dn)) % 360.0


def ang_diff(a, b):
    """Smallest absolute difference between two bearings (deg)."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def circ_mean(bs):
    s = sum(math.sin(math.radians(b)) for b in bs)
    c = sum(math.cos(math.radians(b)) for b in bs)
    return math.degrees(math.atan2(s, c)) % 360.0


def write_text_lf(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def write_csv_lf(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fields})


def write_jsonl_lf(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, separators=(',', ':'), sort_keys=True) + '\n')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------- legs / corners

def street_graph(ways):
    """(adjacency {node: set(neighbour)}, position {node: (lat, lng)}) over street ways."""
    adj, pos = {}, {}
    for w in ways:
        ids, geom = w['nodes'], w['geometry']
        for nid, g in zip(ids, geom):
            pos[nid] = (g['lat'], g['lon'])
            adj.setdefault(nid, set())
        for a, b in zip(ids, ids[1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return adj, pos


def leg_bearings(node_ids, centre_en, adj, pos_en, probe_m=LEG_PROBE_M,
                 stub_min_m=LEG_STUB_MIN_M):
    """Raw leg bearings (deg) of a unit: every edge from a unit node to a node outside
    the unit, walked outward through degree-2 nodes until it leaves a probe_m circle about
    centre_en (bearing to the interpolated exit point), or stops inside it at a dead end
    or another intersection (bearing to the stop, kept when >= stub_min_m out)."""
    unit = set(node_ids)
    ce, cn = centre_en
    out = []

    def dist(nid):
        e, n = pos_en[nid]
        return math.hypot(e - ce, n - cn)
    for u in sorted(unit):
        for v0 in sorted(adj.get(u, ())):
            if v0 in unit:
                continue
            prev, cur = u, v0
            seen = {u}
            while True:
                if dist(cur) >= probe_m:
                    (pe, pn), (qe, qn) = pos_en[prev], pos_en[cur]
                    d0, d1 = math.hypot(pe - ce, pn - cn), math.hypot(qe - ce, qn - cn)
                    t = 1.0 if d1 == d0 else max(0.0, min(1.0, (probe_m - d0) / (d1 - d0)))
                    xe, xn = pe + (qe - pe) * t, pn + (qn - pn) * t
                    out.append(bearing_deg(xe - ce, xn - cn))
                    break
                nbrs = adj.get(cur, set()) - {prev}
                if len(adj.get(cur, ())) != 2 or cur in seen or not nbrs or cur in unit:
                    if dist(cur) >= stub_min_m:
                        e, n = pos_en[cur]
                        out.append(bearing_deg(e - ce, n - cn))
                    break
                seen.add(cur)
                prev, cur = cur, next(iter(nbrs))
    return out


def merge_bearings(bearings, merge_deg=LEG_MERGE_DEG):
    """Circular single-linkage merge of bearings closer than merge_deg; each group is one
    leg at its circular mean. Returns the leg bearings, ascending."""
    if not bearings:
        return []
    bs = sorted(b % 360.0 for b in bearings)
    n = len(bs)
    if n == 1:
        return [round(bs[0], 6)]
    gaps = [((bs[(i + 1) % n] - bs[i]) % 360.0, i) for i in range(n)]
    breaks = [i for g, i in gaps if g >= merge_deg]
    if not breaks:                       # everything chains together: one leg
        return [round(circ_mean(bs), 6) % 360.0]
    legs = []
    start = (breaks[-1] + 1) % n
    group = []
    for k in range(n):
        i = (start + k) % n
        group.append(bs[i])
        if i in breaks:
            legs.append(circ_mean(group))
            group = []
    return sorted(round(b, 6) % 360.0 for b in legs)


def corner_sectors(legs):
    """[(start_deg, width_deg)] clockwise from each leg to the next; one 360 deg sector
    when there are fewer than two legs."""
    if len(legs) < 2:
        return [(0.0, 360.0)]
    out = []
    for i, b in enumerate(legs):
        nxt = legs[(i + 1) % len(legs)]
        out.append((b, (nxt - b) % 360.0 or 360.0))
    return out


def sector_index(bearing, sectors):
    """Index of the sector holding a bearing (a bearing on a leg goes to the sector it
    starts)."""
    for i, (s, w) in enumerate(sectors):
        if (bearing - s) % 360.0 < w:
            return i
    return len(sectors) - 1


def corner_point(centre_en, start, width, offset_m=CORNER_OFFSET_M):
    b = math.radians((start + width / 2.0) % 360.0)
    return centre_en[0] + offset_m * math.sin(b), centre_en[1] + offset_m * math.cos(b)


# ----------------------------------------------------------------------- inventory

def inv_class(props):
    s = props.get('STATUS')
    if s == 'Available':
        return 'Available'
    if s == 'NA':
        return 'NA_typed' if props.get('RAMPTYPE') else 'NA_noramp'
    if s in ('RMV', 'Expired/Removed'):
        return s
    return 'other'


def _date(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d')
    return str(v)


def inv_record(props):
    out = {'unit_id': props.get('UNITID'), 'class': inv_class(props),
           'status': props.get('STATUS'), 'ramptype': props.get('RAMPTYPE')}
    for a in INV_ATTRS:
        v = props.get(a)
        out[a.lower()] = _date(v) if a == 'INSTDATE' else v
    return out


# ------------------------------------------------------------------------- states

def state(present, observed):
    if present:
        return 'present'
    return 'absent' if observed else 'unobservable'


def observed_variants(n25, n15):
    return {'primary': n25 >= 1, 'ge2': n25 >= 2, 'le15': n15 >= 1}


class Points:
    """ENU points with payloads and a within-radius query."""

    def __init__(self, fr, items, cell_m):
        """items: [(lat, lng, payload)]"""
        self.grid, self.xy, self.payload = geo.GridIndex(cell_m), [], []
        self.fr = fr
        for k, (lat, lng, p) in enumerate(items):
            e, n = fr.to_enu(lat, lng)
            self.xy.append((e, n))
            self.payload.append(p)
            self.grid.add(e, n, k)

    def within_en(self, e, n, r):
        return sorted(k for k in self.grid.near(e, n)
                      if math.hypot(self.xy[k][0] - e, self.xy[k][1] - n) <= r)


def describe_unit(unit, centre_en, legs, idx, obs_m, window_m=ecr.WINDOW_M,
                  corners=True):
    """The record of one unit: unit-level and (when corners) corner-level counts and
    states. idx: {'panos', 'sites', 'clusters', 'inventory'} Points objects."""
    ce, cn = centre_en
    pano_r = obs_m + (CORNER_OFFSET_M if corners else 0.0)
    near_panos = idx['panos'].within_en(ce, cn, pano_r + 1e-9)
    sites = idx['sites'].within_en(ce, cn, window_m)
    clus = idx['clusters'].within_en(ce, cn, window_m)
    inv = idx['inventory'].within_en(ce, cn, window_m)

    def dists(ks, e, n):
        return [math.hypot(idx['panos'].xy[k][0] - e, idx['panos'].xy[k][1] - n) for k in ks]
    du = dists(near_panos, ce, cn)
    p25 = [k for k, d in zip(near_panos, du) if d <= obs_m]
    rec = dict(unit)
    rec.update({'legs': legs, 'n_legs': len(legs),
                'n_panos_25': len(p25),
                'n_panos_15': sum(1 for d in du if d <= OBS_NEAR_M),
                'nearest_pano_m': r2(min(du)) if du else None,
                'pano_ids_25': sorted(idx['panos'].payload[k] for k in p25),
                'site_ids': [idx['sites'].payload[k]['site_id'] for k in sites],
                'cluster_ids': [idx['clusters'].payload[k] for k in clus],
                'inventory': [idx['inventory'].payload[k] for k in inv]})
    rec['inv_counts'] = {c: sum(1 for x in rec['inventory'] if x['class'] == c)
                         for c in INV_CLASSES}
    ov = observed_variants(rec['n_panos_25'], rec['n_panos_15'])
    rec['state'] = {f'{a}/{v}': state(bool(rec['site_ids'] if a == 'fusion'
                                           else rec['cluster_ids']), ov[v])
                    for a in ARMS for v in VARIANTS}
    if not corners:
        return rec
    secs = corner_sectors(legs)

    def by_sector(points, ks):
        out = [[] for _ in secs]
        for k in ks:
            e, n = points.xy[k]
            out[sector_index(bearing_deg(e - ce, n - cn), secs)].append(k)
        return out
    s_sites, s_clus, s_inv = by_sector(idx['sites'], sites), by_sector(idx['clusters'], clus), \
        by_sector(idx['inventory'], inv)
    rec['corners'] = []
    for i, (start, width) in enumerate(secs):
        pe, pn = corner_point(centre_en, start, width)
        dc = dists(near_panos, pe, pn)
        k25 = [k for k, d in zip(near_panos, dc) if d <= obs_m]
        lat, lng = idx['panos'].fr.to_latlng(pe, pn)
        c = {'corner': i, 'start_deg': r2(start), 'width_deg': r2(width),
             'wide': width > WIDE_SECTOR_DEG, 'lat': r6(lat), 'lng': r6(lng),
             'n_panos_25': len(k25), 'n_panos_15': sum(1 for d in dc if d <= OBS_NEAR_M),
             'nearest_pano_m': r2(min(dc)) if dc else None,
             'pano_ids_25': sorted(idx['panos'].payload[k] for k in k25),
             'site_ids': [idx['sites'].payload[k]['site_id'] for k in s_sites[i]],
             'site_pano_ids': sorted({p for k in s_sites[i]
                                      for p in idx['sites'].payload[k]['op_panos']}),
             'cluster_ids': [idx['clusters'].payload[k] for k in s_clus[i]],
             'inventory': [idx['inventory'].payload[k] for k in s_inv[i]]}
        c['inv_counts'] = {cl: sum(1 for x in c['inventory'] if x['class'] == cl)
                           for cl in INV_CLASSES}
        ov = observed_variants(c['n_panos_25'], c['n_panos_15'])
        c['state'] = {f'{a}/{v}': state(bool(c['site_ids'] if a == 'fusion'
                                             else c['cluster_ids']), ov[v])
                      for a in ARMS for v in VARIANTS}
        rec['corners'].append(c)
    return rec


# -------------------------------------------------------------------------- loading

def load_sites(run_dir):
    meta = json.loads((run_dir / 'sites_meta.json').read_text(encoding='utf-8'))
    thr = meta['params']['min_confidence']
    out = []
    with open(run_dir / 'sites.jsonl', encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)
            if s.get('n_operational', 0) < 1:
                continue
            ops = sorted({m['pano_id'] for m in s['members'] if m['confidence'] >= thr})
            out.append((s['lat'], s['lng'], {'site_id': s['site_id'], 'op_panos': ops}))
    return out, meta


def load_panos(results_path):
    out, seen = [], set()
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)['pano']
            if p['panorama_id'] in seen:
                continue
            seen.add(p['panorama_id'])
            out.append((p['lat'], p['lng'], p['panorama_id']))
    return out


def load_clusters(path):
    feats = json.loads(Path(path).read_text(encoding='utf-8'))['features']
    return [(f['geometry']['coordinates'][1], f['geometry']['coordinates'][0],
             f['properties']['label_cluster_id']) for f in feats]


def label_frame(raw_labels_path):
    """The exporter's LocalFrame: mean position of the CurbRamp labels load_labels keeps."""
    feats = json.loads(Path(raw_labels_path).read_text(encoding='utf-8'))['features']
    lats, lngs = [], []
    for f in feats:
        lng, lat = f['geometry']['coordinates']
        if lng is None or lat is None or float(lng) > 360 or math.isnan(float(lng)):
            continue
        if (f['properties'] or {}).get('label_type') != 'CurbRamp':
            continue
        lats.append(float(lat))
        lngs.append(float(lng))
    return geo.LocalFrame(math.fsum(lats) / len(lats), math.fsum(lngs) / len(lngs)), len(lats)


def unit_key(c, mid_serial=None):
    """vancouver:<sig|art|res>:n<lowest node id>; mid-block points (which can coincide
    where OSM ways overlap) by their serial in build_candidates order: vancouver:mid:m<n>."""
    if c['type'] == 'mid_block':
        return f'vancouver:mid:m{mid_serial:06d}'
    return f"vancouver:{ecr.PREFIX[c['type']]}:n{min(c['node_ids'])}"


def assign_keys(cands):
    keys, serial = [], 0
    for c in cands:
        if c['type'] == 'mid_block':
            serial += 1
            keys.append(unit_key(c, serial))
        else:
            keys.append(unit_key(c))
    if len(set(keys)) != len(keys):
        raise SystemExit('duplicate unit keys')
    return keys


# ---------------------------------------------------------------------------- build

def cmd_build(args):
    t0 = time.time()
    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    city = args.city
    inputs = {}

    def rec_input(name, path):
        inputs[name] = {'path': str(path), 'sha256': sha256_file(path)}
        return inputs[name]['sha256']
    if rec_input('results', run_dir / 'results.jsonl') != RESULTS_SHA256[city]:
        raise SystemExit('results.jsonl sha256 mismatch')
    osm_path = Path(args.osm)
    if not osm_path.exists():
        raise SystemExit(f'{osm_path}: OSM payload missing (re-fetch deliberately with the '
                         'exporter; this tool never fetches)')
    if rec_input('osm', osm_path) != OSM_SHA256[city]:
        raise SystemExit('osm.json sha256 mismatch')
    for name, rel in (('sites', 'sites.jsonl'), ('sites_meta', 'sites_meta.json'),
                      ('clusters', 'ps_clustering_eval/clusters.geojson'),
                      ('raw_labels', 'provenance_gate/raw_labels.geojson'),
                      ('streets', 'ps_clustering_eval/streets.geojson'),
                      ('area', 'area.geojson'),
                      ('inventory', 'inventory_oracle/inventory.geojson'),
                      ('inventory_record', 'inventory_oracle/inventory.json'),
                      ('store_selection', 'store_selection.json'),
                      ('store_sampled_ids', 'store_sampled_ids.txt')):
        if (run_dir / rel).exists():
            rec_input(name, run_dir / rel)
    inv_rec = json.loads((run_dir / 'inventory_oracle/inventory.json').read_text(encoding='utf-8'))
    if inv_rec.get('sha256') != inputs['inventory']['sha256']:
        raise SystemExit('inventory.geojson does not match inventory.json')
    if args.units224:
        rec_input('units224', Path(args.units224))
    steps = []

    t = time.time()
    payload = json.loads(osm_path.read_text(encoding='utf-8'))
    problem = ecr.validate_osm(payload)
    if problem:
        raise SystemExit(f'OSM payload unusable: {problem}')
    fr, n_lab = label_frame(run_dir / 'provenance_gate/raw_labels.geojson')
    area = json.loads((run_dir / 'area.geojson').read_text(encoding='utf-8'))
    street_index = ecr.SegmentIndex(fr, ecr.open_ps_streets(run_dir / 'ps_clustering_eval/'
                                                            'streets.geojson'),
                                    ecr.ELIGIBLE_STREET_M)
    cands, cand_stats = ecr.build_candidates(payload, fr, area, street_index)
    if cand_stats['eligible'] != EXPECTED_ELIGIBLE[city]:
        raise SystemExit(f"eligible counts {cand_stats['eligible']} != exporter's "
                         f'{EXPECTED_ELIGIBLE[city]}')
    ways = ecr.street_ways(payload)
    adj, pos = street_graph(ways)
    pos_en = {k: fr.to_enu(*v) for k, v in pos.items()}
    steps.append({'step': 'units', 'seconds': round(time.time() - t, 1)})

    t = time.time()
    meta_sites, meta = load_sites(run_dir)
    obs_m = float(meta['params']['max_range_m'])
    panos = load_panos(run_dir / 'results.jsonl')
    clusters = load_clusters(run_dir / 'ps_clustering_eval/clusters.geojson')
    inv_feats = json.loads((run_dir / 'inventory_oracle/inventory.geojson')
                           .read_text(encoding='utf-8'))['features']
    inv_items = [(f['geometry']['coordinates'][1], f['geometry']['coordinates'][0],
                  inv_record(f.get('properties') or {})) for f in inv_feats]
    cell = obs_m + CORNER_OFFSET_M
    idx = {'panos': Points(fr, panos, cell), 'sites': Points(fr, meta_sites, ecr.WINDOW_M),
           'clusters': Points(fr, clusters, ecr.WINDOW_M),
           'inventory': Points(fr, inv_items, ecr.WINDOW_M)}
    steps.append({'step': 'load', 'seconds': round(time.time() - t, 1)})

    t = time.time()
    records = []
    keys = assign_keys(cands)
    for c, key in zip(cands, keys):
        centre = fr.to_enu(c['lat'], c['lng'])
        base = {'unit': key, 'type': c['type'], 'lat': c['lat'], 'lng': c['lng'],
                'node_ids': c['node_ids'], 'highways': c['highways']}
        if c['type'] == 'mid_block':
            records.append(describe_unit(base, centre, [], idx, obs_m, corners=False))
        else:
            legs = merge_bearings(leg_bearings(c['node_ids'], centre, adj, pos_en))
            records.append(describe_unit(base, centre, legs, idx, obs_m))
    steps.append({'step': 'describe', 'seconds': round(time.time() - t, 1),
                  'note': f'{len(records)} units'})

    write_jsonl_lf(out / 'corners_full.jsonl', records)
    write_flat(out, records)
    u224 = []
    if args.units224:
        elig = {tuple(sorted(c['node_ids'])) for c in cands if c['type'] != 'mid_block'}
        mid = Points(fr, [(c['lat'], c['lng'], k) for c, k in zip(cands, keys)
                          if c['type'] == 'mid_block'], 5.0)
        for line in read_jsonl(args.units224):
            centre = fr.to_enu(line['centre']['lat'], line['centre']['lng'])
            base = {'unit': line['corner_id'], 'type': line['type'],
                    'lat': line['centre']['lat'], 'lng': line['centre']['lng'],
                    'node_ids': line['node_ids'], 'highways': line['highways'],
                    'has_labels': line['has_labels']}
            if line['type'] == 'mid_block':
                hit = mid.within_en(centre[0], centre[1], 0.5)
                base['matches_build'] = mid.payload[hit[0]] if hit else None
                legs = []
                r = describe_unit(base, centre, legs, idx, obs_m, corners=False)
            else:
                base['matches_build'] = unit_key(line) \
                    if tuple(sorted(line['node_ids'])) in elig else None
                legs = merge_bearings(leg_bearings(line['node_ids'], centre, adj, pos_en))
                r = describe_unit(base, centre, legs, idx, obs_m)
            u224.append(r)
        write_jsonl_lf(out / 'corners_224.jsonl', u224)
        write_flat(out, u224, prefix='corners_224_')
    build = {'schema': SCHEMA, 'city': city, 'built_at': utc_now(), 'inputs': inputs,
             'exporter': {'file': 'scripts/export_cluster_review.py',
                          'git_sha': git_sha('scripts/export_cluster_review.py')},
             'tool_git_sha': git_sha(),
             'frame': {'lat0': fr.lat0, 'lng0': fr.lng0, 'n_labels': n_lab},
             'params': {'leg_probe_m': LEG_PROBE_M, 'leg_stub_min_m': LEG_STUB_MIN_M,
                        'leg_merge_deg': LEG_MERGE_DEG, 'corner_offset_m': CORNER_OFFSET_M,
                        'wide_sector_deg': WIDE_SECTOR_DEG, 'obs_m': obs_m,
                        'obs_near_m': OBS_NEAR_M, 'window_m': ecr.WINDOW_M,
                        'min_confidence': meta['params']['min_confidence']},
             'candidates': cand_stats,
             'counts': {'panos': len(panos), 'operational_sites': len(meta_sites),
                        'clusters': len(clusters), 'inventory_points': len(inv_items),
                        'units': len(records), 'units224': len(u224)},
             'steps': steps, 'seconds': round(time.time() - t0, 1)}
    (out / 'build.json').write_text(json.dumps(build, indent=1) + '\n', encoding='utf-8',
                                    newline='\n')
    print(f"built {len(records)} units ({build['seconds']} s) -> {out}")
    return build


def write_flat(out, records, prefix=''):
    """units.csv and corners.csv: one row per unit / corner, lists as ';'-joined."""
    ufields = ['unit', 'type', 'lat', 'lng', 'n_legs', 'legs', 'n_panos_25', 'n_panos_15',
               'nearest_pano_m', 'n_sites', 'n_clusters'] + \
        [f'inv_{c}' for c in INV_CLASSES] + [f'state_{a}_{v}' for a in ARMS for v in VARIANTS]
    cfields = ['unit', 'type', 'corner', 'start_deg', 'width_deg', 'wide', 'lat', 'lng',
               'n_panos_25', 'n_panos_15', 'nearest_pano_m', 'n_sites', 'n_clusters',
               'site_ids', 'site_pano_ids'] + [f'inv_{c}' for c in INV_CLASSES] + \
        [f'inv_{a.lower()}' for a in INV_ATTRS] + \
        [f'state_{a}_{v}' for a in ARMS for v in VARIANTS]
    urows, crows = [], []
    for r in records:
        u = {k: r.get(k) for k in ufields}
        u['lat'], u['lng'] = r6(r['lat']), r6(r['lng'])
        u['legs'] = ';'.join(f'{b:.1f}' for b in r['legs'])
        u['n_sites'], u['n_clusters'] = len(r['site_ids']), len(r['cluster_ids'])
        for c in INV_CLASSES:
            u[f'inv_{c}'] = r['inv_counts'][c]
        for a in ARMS:
            for v in VARIANTS:
                u[f'state_{a}_{v}'] = r['state'][f'{a}/{v}']
        urows.append(u)
        for c in r.get('corners', ()):
            row = {k: c.get(k) for k in cfields}
            row.update({'unit': r['unit'], 'type': r['type'], 'wide': int(c['wide']),
                        'n_sites': len(c['site_ids']), 'n_clusters': len(c['cluster_ids']),
                        'site_ids': ';'.join(str(s) for s in c['site_ids']),
                        'site_pano_ids': ';'.join(c['site_pano_ids'])})
            for cl in INV_CLASSES:
                row[f'inv_{cl}'] = c['inv_counts'][cl]
            for a in INV_ATTRS:
                vals = sorted({str(x[a.lower()]) for x in c['inventory']
                               if x[a.lower()] not in (None, '')})
                row[f'inv_{a.lower()}'] = ';'.join(vals)
            for a in ARMS:
                for v in VARIANTS:
                    row[f'state_{a}_{v}'] = c['state'][f'{a}/{v}']
            crows.append(row)
    write_csv_lf(Path(out) / f'{prefix}units.csv', ufields, urows)
    write_csv_lf(Path(out) / f'{prefix}corners.csv', cfields, crows)


# ---------------------------------------------------------------------------- score

def items_for(records, level, strata, wide=None):
    """The scored items: unit records, or corner records carrying their unit's type."""
    out = []
    for r in records:
        if r['type'] not in strata:
            continue
        if level == 'unit':
            out.append(r)
        else:
            for c in r.get('corners', ()):
                if wide is None or c['wide'] == wide:
                    out.append(c)
    return out


def tally(items, key):
    """Counts of the states under one 'arm/variant' key, and the inventory reads."""
    n = len(items)
    st = {s: sum(1 for x in items if x['state'][key] == s) for s in STATES}
    avail = [x for x in items if x['inv_counts']['Available'] > 0]
    rec = {s: sum(1 for x in avail if x['state'][key] == s) for s in STATES}
    absent = [x for x in items if x['state'][key] == 'absent']
    clean = sum(1 for x in absent if not any(x['inv_counts'].values()))
    false_abs = sum(1 for x in absent if x['inv_counts']['Available'] > 0)
    gaps = sum(1 for x in items if x['state'][key] == 'present'
               and not any(x['inv_counts'].values()))
    present_unobs = sum(1 for x in items if x['state'][key] == 'present'
                        and x['n_panos_25'] == 0)
    return {'n': n, **{f'n_{s}': st[s] for s in STATES}, 'n_avail': len(avail),
            **{f'avail_{s}': rec[s] for s in STATES}, 'n_absent': len(absent),
            'absent_clean': clean, 'absent_false': false_abs,
            'absent_rmvna_only': len(absent) - clean - false_abs, 'present_no_inventory': gaps,
            'present_not_observed': present_unobs}


def fmt_share(k, n):
    if n == 0:
        return 'n/a'
    lo, hi = wilson(k, n)
    return f'{k / n:.3f} [{lo:.3f}, {hi:.3f}]'


def score_rows(records):
    """Every (level, stratum, wide, arm, variant) tally as a flat row."""
    rows = []
    groups = [(s, (s,)) for s in INTERSECTION_STRATA] + [('intersections',
                                                          INTERSECTION_STRATA)]
    for level in ('unit', 'corner'):
        for name, strata in groups:
            for wide_name, wide in ((('all', None), ('non_wide', False))
                                    if level == 'corner' else (('all', None),)):
                items = items_for(records, level, strata, wide)
                for a in ARMS:
                    for v in VARIANTS:
                        rows.append({'level': level, 'stratum': name, 'sectors': wide_name,
                                     'arm': a, 'variant': v, **tally(items, f'{a}/{v}')})
    mids = items_for(records, 'unit', ('mid_block',))
    for a in ARMS:
        for v in VARIANTS:
            rows.append({'level': 'unit', 'stratum': 'mid_block', 'sectors': 'all', 'arm': a,
                         'variant': v, **tally(mids, f'{a}/{v}')})
    return rows


COUNT_FIELDS = ['level', 'stratum', 'sectors', 'arm', 'variant', 'n', 'n_present', 'n_absent',
                'n_unobservable', 'n_avail', 'avail_present', 'avail_absent',
                'avail_unobservable', 'absent_clean', 'absent_false', 'absent_rmvna_only',
                'present_no_inventory', 'present_not_observed']


def decision(rows):
    r = next(x for x in rows if x['level'] == 'unit' and x['stratum'] == 'intersections'
             and x['arm'] == 'fusion' and x['variant'] == 'primary')
    n, k = r['n_absent'], r['absent_clean']
    if n == 0:
        return {'n_absent': 0, 'clean': 0, 'precision': None, 'ci': None,
                'outcome': 'UNDEFINED: no unit is called absent'}
    p, ci = k / n, wilson(k, n)
    outcome = ('PASS: experiment 2 is unblocked' if p >= DECISION_THRESHOLD else
               'FAIL: triage the false absences (detector miss vs observability) first')
    return {'n_absent': n, 'clean': k, 'precision': round(p, 4),
            'ci': [round(ci[0], 4), round(ci[1], 4)], 'outcome': outcome,
            'no_available_read': round(1 - r['absent_false'] / n, 4)}


def render_tables(rows):
    L = []

    def pick(level, stratum, arm, variant, sectors='all'):
        return next(x for x in rows if x['level'] == level and x['stratum'] == stratum
                    and x['arm'] == arm and x['variant'] == variant
                    and x['sectors'] == sectors)
    strata = INTERSECTION_STRATA + ('intersections',)
    for level, sectors in (('unit', 'all'), ('corner', 'all'), ('corner', 'non_wide')):
        L += [f'### City sentence, {level} level' + (' (sectors <= 150 deg only)'
                                                      if sectors == 'non_wide' else ''), '',
              '| stratum | arm | n | present | absent | unobservable |', '|---|---|---:|---|---|---|']
        for s in strata + (('mid_block',) if level == 'unit' else ()):
            for a in ARMS:
                x = pick(level, s, a, 'primary', sectors)
                L.append(f"| {s} | {a} | {x['n']} | {fmt_share(x['n_present'], x['n'])} | "
                         f"{fmt_share(x['n_absent'], x['n'])} | "
                         f"{fmt_share(x['n_unobservable'], x['n'])} |")
        L.append('')
    for level in ('unit', 'corner'):
        L += [f'### Recall against `Available` inventory, {level} level', '',
              '| stratum | arm | with Available | present | absent | unobservable |',
              '|---|---|---:|---|---|---|']
        for s in strata:
            for a in ARMS:
                x = pick(level, s, a, 'primary')
                L.append(f"| {s} | {a} | {x['n_avail']} | {fmt_share(x['avail_present'], x['n_avail'])} | "
                         f"{fmt_share(x['avail_absent'], x['n_avail'])} | "
                         f"{fmt_share(x['avail_unobservable'], x['n_avail'])} |")
        L.append('')
    for level in ('unit', 'corner'):
        L += [f'### Absence precision against the inventory, {level} level', '',
              '| stratum | arm | absent | clean (no point) | false absence (Available) | '
              'RMV/NA only | no-Available read |', '|---|---|---:|---|---|---|---|']
        for s in strata:
            for a in ARMS:
                x = pick(level, s, a, 'primary')
                n = x['n_absent']
                L.append(f"| {s} | {a} | {n} | {fmt_share(x['absent_clean'], n)} | "
                         f"{fmt_share(x['absent_false'], n)} | "
                         f"{fmt_share(x['absent_rmvna_only'], n)} | "
                         f"{fmt_share(n - x['absent_false'], n)} |")
        L.append('')
    L += ['### Observability sensitivity (intersections pooled, fusion arm)', '',
          '| level | variant | present | absent | unobservable | absent clean | false absence |',
          '|---|---|---|---|---|---|---|']
    for level in ('unit', 'corner'):
        for v in VARIANTS:
            x = pick(level, 'intersections', 'fusion', v)
            L.append(f"| {level} | {v} | {fmt_share(x['n_present'], x['n'])} | "
                     f"{fmt_share(x['n_absent'], x['n'])} | "
                     f"{fmt_share(x['n_unobservable'], x['n'])} | "
                     f"{fmt_share(x['absent_clean'], x['n_absent'])} | "
                     f"{fmt_share(x['absent_false'], x['n_absent'])} |")
    L += ['', '### Inventory gaps (present, no inventory point of any status)', '',
          '| level | stratum | fusion | deployed | present not observed (fusion) |',
          '|---|---|---:|---:|---:|']
    for level in ('unit', 'corner'):
        for s in strata:
            f_, d_ = pick(level, s, 'fusion', 'primary'), pick(level, s, 'deployed', 'primary')
            L.append(f"| {level} | {s} | {f_['present_no_inventory']} | "
                     f"{d_['present_no_inventory']} | {f_['present_not_observed']} |")
    return L


def selection_diagnostics(records, build, tier=0.55):
    """How the run's pano set was chosen, and what that does to the states.

    A store-built run (detect_from_store.py) holds the panos that carry a deployed label
    plus a small seeded sample of unlabeled store panos (store_sampled_ids.txt). The
    observability test is therefore conditioned on a label having been made in the pano."""
    inputs = build['inputs']
    if 'store_sampled_ids' not in inputs:
        return None
    sampled = set(Path(inputs['store_sampled_ids']['path']).read_text(encoding='utf-8')
                  .split())
    n_runs, n_det = 0, 0
    run_ids = set()
    with open(inputs['results']['path'], encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            run_ids.add(d['pano']['panorama_id'])
            n_runs += 1
            if any(x['confidence'] >= tier for x in d.get('detections') or ()):
                n_det += 1
    ints = [r for r in records if r['type'] in INTERSECTION_STRATA]
    near_sampled = {p for r in ints for p in r['pano_ids_25'] if p in sampled}
    cross = {}
    for r in ints:
        k = (r['state']['fusion/primary'],
             'sampled-unlabeled pano within 25 m' if any(p in sampled for p in r['pano_ids_25'])
             else 'labeled panos only' if r['pano_ids_25'] else 'no pano within 25 m')
        cross[k] = cross.get(k, 0) + 1
    unobs = [r['nearest_pano_m'] for r in ints if r['state']['fusion/primary'] == 'unobservable'
             and r['nearest_pano_m'] is not None]
    return {'run_panos': n_runs, 'run_panos_with_det_ge_tier': n_det, 'tier': tier,
            'sampled_unlabeled_in_selection': len(sampled),
            'sampled_unlabeled_in_run': len(sampled & run_ids),
            'sampled_unlabeled_within_25m_of_a_unit': len(near_sampled),
            'unit_state_by_pano_kind': {f'{a} | {b}': n for (a, b), n in sorted(cross.items())},
            'unobservable_units_nearest_pano_m': quant(unobs),
            'unobservable_units_with_no_pano_within_37m': sum(
                1 for r in ints if r['state']['fusion/primary'] == 'unobservable'
                and r['nearest_pano_m'] is None)}


def false_absence_rows(records, key='fusion/primary'):
    """Units and corners called absent that hold an `Available` inventory point: the
    triage list of the decision rule (detector miss vs observability)."""
    out = []
    for r in records:
        if r['type'] not in INTERSECTION_STRATA:
            continue
        items = [('', r)] + [(c['corner'], c) for c in r['corners']]
        for corner, x in items:
            if x['state'][key] == 'absent' and x['inv_counts']['Available'] > 0:
                out.append({'level': 'unit' if corner == '' else 'corner', 'unit': r['unit'],
                            'type': r['type'], 'corner': corner,
                            'lat': r6(x['lat']), 'lng': r6(x['lng']),
                            'n_panos_25': x['n_panos_25'], 'nearest_pano_m': x['nearest_pano_m'],
                            'pano_ids_25': ';'.join(x['pano_ids_25']),
                            'inventory_unit_ids': ';'.join(str(i['unit_id']) for i in x['inventory']
                                                           if i['class'] == 'Available'),
                            'deployed_present': int(bool(x['cluster_ids']))})
    return out


def gap_rows(records):
    out = []
    for r in records:
        if r['type'] not in INTERSECTION_STRATA:
            continue
        for c in r['corners']:
            if c['state']['fusion/primary'] == 'present' and not any(c['inv_counts'].values()):
                out.append({'unit': r['unit'], 'type': r['type'], 'corner': c['corner'],
                            'lat': c['lat'], 'lng': c['lng'], 'n_sites': len(c['site_ids']),
                            'site_ids': ';'.join(str(s) for s in c['site_ids']),
                            'pano_ids': ';'.join(c['site_pano_ids']),
                            'deployed_present': int(bool(c['cluster_ids'])),
                            'unit_has_inventory': int(any(r['inv_counts'].values()))})
    return out


def cmd_score(args):
    t0 = time.time()
    out = Path(args.out)
    records = read_jsonl(out / 'corners_full.jsonl')
    build = json.loads((out / 'build.json').read_text(encoding='utf-8'))
    rows = score_rows(records)
    write_csv_lf(out / 'counts.csv', COUNT_FIELDS, rows)
    gaps = gap_rows(records)
    write_csv_lf(out / 'gaps.csv', ['unit', 'type', 'corner', 'lat', 'lng', 'n_sites',
                                    'site_ids', 'pano_ids', 'deployed_present',
                                    'unit_has_inventory'], gaps)
    fa = false_absence_rows(records)
    write_csv_lf(out / 'false_absences.csv', ['level', 'unit', 'type', 'corner', 'lat', 'lng',
                                              'n_panos_25', 'nearest_pano_m', 'pano_ids_25',
                                              'inventory_unit_ids', 'deployed_present'], fa)
    dec = decision(rows)
    ints = [r for r in records if r['type'] in INTERSECTION_STRATA]
    n_corners = sum(len(r['corners']) for r in ints)
    leg_hist = {}
    for r in ints:
        leg_hist[r['n_legs']] = leg_hist.get(r['n_legs'], 0) + 1
    L = [f"# {build['city']}: corner inventory (RampNet#238)", '',
         f"Built {build['built_at']} by `scripts/corner_inventory.py@{build['tool_git_sha']}`; "
         f"exporter `export_cluster_review.py@{build['exporter']['git_sha']}`. Protocol: "
         'docs/corner-inventory.md.', '', '## Inputs', '']
    for k, v in build['inputs'].items():
        L.append(f"- {k}: `{v['path']}` sha256 `{v['sha256']}`")
    L += ['', '## Units', '',
          f"- intersection units {len(ints)} ({', '.join(f'{s} {sum(1 for r in ints if r['type'] == s)}' for s in INTERSECTION_STRATA)}), "
          f"corners {n_corners} (wide > 150 deg: "
          f"{sum(1 for r in ints for c in r['corners'] if c['wide'])}); mid-block points "
          f"{sum(1 for r in records if r['type'] == 'mid_block')}",
          f"- legs per intersection unit: {dict(sorted(leg_hist.items()))}",
          f"- panos {build['counts']['panos']}, operational sites "
          f"{build['counts']['operational_sites']}, deployed clusters "
          f"{build['counts']['clusters']}, inventory points {build['counts']['inventory_points']}",
          '', '## Decision', '',
          f"Absence precision, clean read, unit level, fusion arm, primary observability, "
          f"intersections pooled: {dec['clean']}/{dec['n_absent']} = {dec['precision']} "
          f"(Wilson 95% {dec['ci']}); threshold {DECISION_THRESHOLD}. **{dec['outcome']}**. "
          f"Second read (absent with no `Available` point, not the rule): "
          f"{dec.get('no_available_read')}.", '', '## Tables', ''] + render_tables(rows)
    L += ['', '### What the inventory holds at absent units / corners (fusion, primary, '
          'intersections pooled)', '',
          '| level | absent | ' + ' | '.join(f'with {c}' for c in INV_CLASSES) + ' | none |',
          '|---|---:|' + '---:|' * (len(INV_CLASSES) + 1)]
    for level in ('unit', 'corner'):
        ab = [x for x in items_for(records, level, INTERSECTION_STRATA)
              if x['state']['fusion/primary'] == 'absent']
        L.append(f'| {level} | {len(ab)} | ' + ' | '.join(
            str(sum(1 for x in ab if x['inv_counts'][c] > 0)) for c in INV_CLASSES) +
            f" | {sum(1 for x in ab if not any(x['inv_counts'].values()))} |")
    diag = selection_diagnostics(records, build)
    if diag:
        L += ['', '### How the pano set was selected (store-built run)', '',
              f"- run panos {diag['run_panos']}; with >= 1 detection >= {diag['tier']}: "
              f"{diag['run_panos_with_det_ge_tier']}",
              f"- seeded unlabeled store sample: {diag['sampled_unlabeled_in_selection']} "
              f"selected, {diag['sampled_unlabeled_in_run']} in the run, "
              f"{diag['sampled_unlabeled_within_25m_of_a_unit']} within 25 m of an "
              'intersection unit centre',
              f"- unobservable units: nearest pano (m) {diag['unobservable_units_nearest_pano_m']}"
              f", none within 37 m: {diag['unobservable_units_with_no_pano_within_37m']}", '',
              '| unit state (fusion, primary) | panos within 25 m | units |', '|---|---|---:|']
        for k, n in diag['unit_state_by_pano_kind'].items():
            a, b = k.split(' | ')
            L.append(f'| {a} | {b} | {n} |')
        dec['selection'] = diag
    L += ['', f"Inventory-gap corners (fusion, primary): {len(gaps)} rows in gaps.csv.", '',
          f"Build {build['seconds']} s; score {round(time.time() - t0, 1)} s.", '']
    write_text_lf(out / 'report.md', '\n'.join(L))
    (out / 'decision.json').write_text(json.dumps(dec, indent=1) + '\n', encoding='utf-8',
                                       newline='\n')
    print('\n'.join(L[-40:]))
    print(json.dumps(dec))
    return dec


# ---------------------------------------------------------------- score-assignments

def rater_state(ramps, uncovered):
    return 'present' if ramps or any(not u.get('unsure') for u in uncovered) else 'absent'


def score_assignments(units, assignments, fr=None):
    """Compare our states (corners_224 records) with a rater's assignments.json.

    Returns (rows, summary): one row per scored unit and per corner, and a confusion
    {arm/variant: {level: {our_state: {rater_state: n}}}}."""
    by_id = {u['unit']: u for u in units}
    rows = []
    conf = {}
    excluded = {'not_in_units': [], 'incomplete': [], 'cant_judge': []}
    for cid in sorted(assignments.get('corners', {})):
        a = assignments['corners'][cid]
        if cid not in by_id:
            excluded['not_in_units'].append(cid)
            continue
        if a.get('cant_judge'):
            excluded['cant_judge'].append(cid)
            continue
        if not a.get('complete'):
            excluded['incomplete'].append(cid)
            continue
        u = by_id[cid]
        frame = fr or geo.LocalFrame(u['lat'], u['lng'])
        ce, cn = frame.to_enu(u['lat'], u['lng'])
        ramps = list((a.get('ramps') or {}).values())
        unc = list(a.get('uncovered') or [])
        n_unsure = sum(1 for x in unc if x.get('unsure'))
        rs = rater_state(ramps, unc)
        rows.append({'unit': cid, 'type': u['type'], 'corner': '', 'rater': rs,
                     'n_ramps': len(ramps), 'n_uncovered': len(unc) - n_unsure,
                     'n_uncovered_unsure': n_unsure,
                     **{f'ours_{k.replace("/", "_")}': v for k, v in u['state'].items()}})
        if u.get('corners'):
            secs = [(c['start_deg'], c['width_deg']) for c in u['corners']]
            per = [[[], []] for _ in secs]
            for kind, pts in ((0, ramps), (1, unc)):
                for p in pts:
                    e, n = frame.to_enu(p['lat'], p['lng'])
                    if math.hypot(e - ce, n - cn) > ecr.WINDOW_M:
                        continue
                    per[sector_index(bearing_deg(e - ce, n - cn), secs)][kind].append(p)
            for c, (rp, up) in zip(u['corners'], per):
                rows.append({'unit': cid, 'type': u['type'], 'corner': c['corner'],
                             'rater': rater_state(rp, up), 'n_ramps': len(rp),
                             'n_uncovered': sum(1 for x in up if not x.get('unsure')),
                             'n_uncovered_unsure': sum(1 for x in up if x.get('unsure')),
                             **{f'ours_{k.replace("/", "_")}': v
                                for k, v in c['state'].items()}})
    for r in rows:
        level = 'unit' if r['corner'] == '' else 'corner'
        for a in ARMS:
            for v in VARIANTS:
                k = f'{a}/{v}'
                d = conf.setdefault(k, {}).setdefault(level, {})
                ours = r[f'ours_{a}_{v}']
                d.setdefault(ours, {}).setdefault(r['rater'], 0)
                d[ours][r['rater']] += 1
    summary = {'confusion': conf, 'excluded': excluded}
    reads = {}
    for k, levels in conf.items():
        for level, m in levels.items():
            ab = m.get('absent', {})
            n_ab = sum(ab.values())
            pres = {o: m.get(o, {}).get('present', 0) for o in STATES}
            n_rp = sum(pres.values())
            reads[f'{k}/{level}'] = {
                'absence_precision': [ab.get('absent', 0), n_ab],
                'recall_vs_rater': [pres['present'], n_rp],
                'rater_present_called_unobservable': [pres['unobservable'], n_rp]}
    summary['reads'] = reads
    return rows, summary


def cmd_score_assignments(args):
    out = Path(args.out)
    units = read_jsonl(Path(args.units) if args.units else out / 'corners_224.jsonl')
    assignments = json.loads(Path(args.assignments).read_text(encoding='utf-8'))
    if assignments.get('schema') != 'rampnet.cluster_review/1':
        raise SystemExit(f"unexpected schema {assignments.get('schema')!r}")
    rows, summary = score_assignments(units, assignments)
    dest = Path(args.dest) if args.dest else out / 'assignments_score'
    fields = ['unit', 'type', 'corner', 'rater', 'n_ramps', 'n_uncovered',
              'n_uncovered_unsure'] + [f'ours_{a}_{v}' for a in ARMS for v in VARIANTS]
    write_csv_lf(dest / 'rows.csv', fields, rows)
    summary['assignments'] = {'path': str(args.assignments),
                              'sha256': sha256_file(args.assignments),
                              'rater': assignments.get('rater'), 'role': assignments.get('role')}
    write_text_lf(dest / 'summary.json', json.dumps(summary, indent=1, sort_keys=True) + '\n')
    for k in ('fusion/primary/unit', 'fusion/primary/corner', 'deployed/primary/unit'):
        if k in summary['reads']:
            print(k, summary['reads'][k])
    return rows, summary


# --------------------------------------------------------------------------- census

def census_sample(records, n_by=CENSUS_N, seed=CENSUS_SEED):
    rng = random.Random(seed)
    out = []
    for s in INTERSECTION_STRATA:
        pool = sorted((r for r in records if r['type'] == s), key=lambda r: r['unit'])
        out += rng.sample(pool, min(n_by[s], len(pool)))
    return out


def fetch_raw(pano_id, api_mod, sleep=time.sleep, attempts=None):
    """(raw response or None, status, error, attempts used). status: ok / not_found /
    empty / error. The retry policy of sources/gsv.py (METADATA_ATTEMPTS, same backoff);
    a pano the endpoint reports as not found (status code 2) is not retried."""
    import sources.gsv as gsv
    attempts = attempts or gsv.METADATA_ATTEMPTS
    kind, err = 'empty', None
    for k in range(attempts):
        try:
            raw = api_mod.find_panorama_by_id(pano_id, download_depth=False)
            try:
                code = raw[1][0][0][0]
            except (TypeError, IndexError, KeyError):
                code = None
            if code == 2:
                return raw, 'not_found', None, k + 1
            if code == 1:
                return raw, 'ok', None, k + 1
            kind, err = 'empty', f'unexpected status code {code!r}'
        except Exception as exc:  # network / parse
            kind, err = 'error', f'{type(exc).__name__}: {exc}'
        if k < attempts - 1:
            sleep(2 * (k + 1) + random.uniform(0, 1))
    return None, kind, err, attempts


def parse_history(raw):
    """(capture 'YYYY-MM' or None, [(pano_id, 'YYYY-MM')] historical) of a raw response."""
    from streetlevel.streetview.parse import parse_panorama_id_response
    m = parse_panorama_id_response(raw)
    if m is None:
        return None, []
    cur = f'{m.date.year}-{m.date.month:02d}' if m.date else None
    hist = [(h.id, f'{h.date.year}-{h.date.month:02d}') for h in (m.historical or [])
            if h.date is not None]
    return cur, hist


def cmd_census(args):
    out = Path(args.out)
    cdir = out / 'census'
    cache = cdir / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(out / 'corners_full.jsonl')
    sample = census_sample(records)
    panos = sorted({p for r in sample for p in r['pano_ids_25']})
    from streetlevel.streetview import api
    t0 = time.time()
    log = []
    fetched = 0
    stop_err = None
    consecutive_err = 0
    for i, pid in enumerate(panos):
        path = cache / f'{pid}.json'
        if path.exists():
            continue
        raw, status, err, n_att = fetch_raw(pid, api)
        fetched += 1
        path.write_text(json.dumps({'pano_id': pid, 'fetched_at': utc_now(), 'status': status,
                                    'error': err, 'attempts': n_att, 'raw': raw},
                                   separators=(',', ':')), encoding='utf-8', newline='\n')
        consecutive_err = consecutive_err + 1 if status == 'error' else 0
        if consecutive_err >= 5:
            stop_err = err
            break
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(panos)} {time.time() - t0:.0f} s')
        time.sleep(CENSUS_SLEEP_S)
    wall = time.time() - t0
    info = {}
    for pid in panos:
        path = cache / f'{pid}.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        cur, hist = (None, [])
        if d['status'] == 'ok':
            cur, hist = parse_history(d['raw'])
        info[pid] = {'status': d['status'], 'error': d['error'], 'current': cur, 'hist': hist}
    h = hashlib.sha256()
    for pid in sorted(info):
        h.update((cache / f'{pid}.json').read_bytes())
    build = json.loads((out / 'build.json').read_text(encoding='utf-8'))
    run_dates = {}
    with open(build['inputs']['results']['path'], encoding='utf-8') as f:
        for line in f:
            if line.strip():
                p = json.loads(line)['pano']
                run_dates[p['panorama_id']] = p.get('capture_date')
    summary = census_summary(sample, info, records, run_dates)
    summary.update({'sample_seed': CENSUS_SEED, 'sample_n': CENSUS_N, 'n_panos': len(panos),
                    'fetched_this_invocation': fetched, 'wall_clock_s': round(wall, 1),
                    'stopped_on_error': stop_err, 'cache_sha256': h.hexdigest(),
                    'sleep_s': CENSUS_SLEEP_S, 'finished_at': utc_now()})
    prev = cdir / 'census.json'
    if prev.exists():
        old = json.loads(prev.read_text(encoding='utf-8'))
        summary['wall_clock_history_s'] = old.get('wall_clock_history_s', [old.get('wall_clock_s')])
    summary.setdefault('wall_clock_history_s', [])
    summary['wall_clock_history_s'] = [w for w in summary['wall_clock_history_s'] if w] + \
        ([round(wall, 1)] if fetched else [])
    write_text_lf(prev, json.dumps(summary, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_unit'}, indent=1))
    return summary


def quant(xs, ps=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {str(p): xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] for p in ps}


def census_summary(sample, info, records, run_dates=None):
    """Captures per unit / corner (distinct year-months over the run's own capture dates +
    every GSV-served current and historical pano), earliest year, span, and the
    historical-pano count (ids not already in the run), with the full-pass extrapolation.
    `n_captures_gsv` counts only what the endpoint served (a pano it no longer serves
    contributes its run capture date to `n_captures` and nothing else)."""
    run_dates = run_dates or {}
    status = {}
    for v in info.values():
        status[v['status']] = status.get(v['status'], 0) + 1
    current_ids = set(info) | set(run_dates)
    hist_ids = {}
    for pid, v in info.items():
        for hid, ym in v['hist']:
            if hid not in current_ids:
                hist_ids[hid] = ym
    per_unit, per_corner = [], []

    def caps(pids, gsv_only=False):
        yms = set()
        hs = set()
        for p in pids:
            if run_dates.get(p) and not gsv_only:
                yms.add(run_dates[p])
            v = info.get(p)
            if not v or v['status'] != 'ok':
                continue
            if v['current']:
                yms.add(v['current'])
            for hid, ym in v['hist']:
                yms.add(ym)
                if hid not in current_ids:
                    hs.add(hid)
        return yms, hs
    for r in sample:
        yms, hs = caps(r['pano_ids_25'])
        years = sorted(int(y[:4]) for y in yms)
        per_unit.append({'unit': r['unit'], 'type': r['type'], 'n_panos': len(r['pano_ids_25']),
                         'n_captures': len(yms), 'n_hist': len(hs),
                         'n_captures_gsv': len(caps(r['pano_ids_25'], gsv_only=True)[0]),
                         'earliest': years[0] if years else None,
                         'span_years': (years[-1] - years[0]) if years else None})
        for c in r['corners']:
            cy, _ = caps(c['pano_ids_25'])
            cyears = sorted(int(y[:4]) for y in cy)
            per_corner.append({'n_captures': len(cy),
                               'earliest': cyears[0] if cyears else None,
                               'span_years': (cyears[-1] - cyears[0]) if cyears else None})
    ratio = len(hist_ids) / len(info) if info else None      # per QUERIED pano
    all_near = {p for r in records if r['type'] in INTERSECTION_STRATA for p in r['pano_ids_25']}
    by_stratum = {}
    for s in INTERSECTION_STRATA:
        us = [u for u in per_unit if u['type'] == s]
        by_stratum[s] = {'units': len(us),
                         'captures_per_unit': quant([u['n_captures'] for u in us]),
                         'earliest': quant([u['earliest'] for u in us if u['earliest']]),
                         'span_years': quant([u['span_years'] for u in us
                                              if u['span_years'] is not None])}
    return {'status': status, 'n_hist_distinct': len(hist_ids),
            'hist_per_queried_pano': None if ratio is None else round(ratio, 3),
            'panos_near_any_unit': len(all_near),
            # extrapolation 1: distinct historical ids per queried pano x every run pano
            # within 25 m of an eligible intersection unit. Low if neighbouring sampled panos
            # share history less than the city does (they are spread out), high otherwise.
            'extrapolated_hist_full_pass_by_pano': None if ratio is None
            else round(ratio * len(all_near)),
            # extrapolation 2: mean distinct historical ids per sampled unit, per stratum, x
            # eligible units in the stratum. An upper bound: neighbouring windows share panos.
            'extrapolated_hist_full_pass_by_unit': round(sum(
                (sum(u['n_hist'] for u in per_unit if u['type'] == st)
                 / max(1, sum(1 for u in per_unit if u['type'] == st)))
                * sum(1 for r in records if r['type'] == st) for st in INTERSECTION_STRATA)),
            'captures_per_unit': quant([u['n_captures'] for u in per_unit]),
            'captures_per_unit_gsv_only': quant([u['n_captures_gsv'] for u in per_unit]),
            'captures_per_corner': quant([c['n_captures'] for c in per_corner]),
            'earliest_year_unit': quant([u['earliest'] for u in per_unit if u['earliest']]),
            'span_years_unit': quant([u['span_years'] for u in per_unit
                                      if u['span_years'] is not None]),
            'units_with_no_pano': sum(1 for u in per_unit if u['n_panos'] == 0),
            'units_with_panos_none_served': sum(1 for u in per_unit if u['n_panos'] > 0
                                                and u['n_captures_gsv'] == 0),
            'by_stratum': by_stratum, 'per_unit': per_unit}


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--city', default='vancouver')
    b.add_argument('--run-dir', required=True)
    b.add_argument('--osm', required=True)
    b.add_argument('--units224')
    b.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('--out', required=True)
    c = sub.add_parser('census')
    c.add_argument('--out', required=True)
    a = sub.add_parser('score-assignments')
    a.add_argument('--assignments', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--units', help='corners_224.jsonl (default <out>/corners_224.jsonl)')
    a.add_argument('--dest', help='output dir (default <out>/assignments_score)')
    args = ap.parse_args(argv)
    return {'build': cmd_build, 'score': cmd_score, 'census': cmd_census,
            'score-assignments': cmd_score_assignments}[args.cmd](args)


if __name__ == '__main__':
    main()
