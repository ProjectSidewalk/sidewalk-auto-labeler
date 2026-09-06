"""Score Project Sidewalk's production label clustering against RampNet ground truth.

Companion to docs/ps-clustering-eval.md (the pre-registered protocol) and to
SidewalkWebpage#4706. The labeler keeps every per-view AI label on purpose, so the
server's clustering is what turns labels into ramps; this measures how well it does
that, and where any shortfall comes from (threshold, placement or merge criterion).

Inputs (all read as data, nothing imported from SidewalkWebpage unless --ps-script):
  - /v3/api/rawLabels?labelType=CurbRamp&filetype=geojson      -> raw_labels.geojson
  - /v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
                                                               -> clusters.geojson
    (both downloaded into --out when --server is given and the files are absent)
  - runs/<city>/results.jsonl (maps each AI label back to its stored detection by
    pano id + pixel position; the same rounding send_to_ps.py used)
  - ../RampNet/benchmark/<city>/{verdicts.json,records.jsonl} via eval_sites.py

Arms (every one a partition of the same AI labels, scored by one scorer in one frame):
  deployed        the server's clusters as served
  ps_repro        SidewalkWebpage/scripts/label_clustering.py cluster(), verbatim
                  (--ps-script); exists to prove the harness reproduces the server
  ps @ t          the PS algorithm (complete linkage + same-(user,pano) cannot-link),
                  re-implemented vectorized, per region, threshold sweep
  ps_citywide     ...the same at 7.5 m over the whole city (region-boundary effect)
  ps_raycast @ t  the PS algorithm on the labeler's raycast positions (isolates
                  placement from algorithm)
  fusion          fuse_sites.py's ray-aware associator (the labeler's reference)

Every cluster is placed at the mean of its members' raycast positions, so scores
depend only on who was grouped with whom. Needs pandas, scipy and `haversine`
(the package the PS script uses; pip install haversine).

Usage:
    python scripts/eval_ps_clustering.py richmond \
        --server https://sidewalk-richmond.cs.washington.edu \
        --ps-script /path/to/SidewalkWebpage/scripts/label_clustering.py
"""
import argparse
import csv
import importlib.util
import json
import math
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402

try:
    from haversine import haversine_vector
except ImportError as exc:  # pragma: no cover
    raise SystemExit('pip install haversine (the package the PS script uses)') from exc

PS_THRESHOLD_KM = 0.0075   # label_clustering.py THRESHOLDS['CurbRamp']
API_LABELS = '/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson'
API_CLUSTERS = ('/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true'
                '&filetype=geojson')


# ----------------------------------------------------------------------------- inputs

def fetch(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 (https, fixed host)
        dest.write_bytes(r.read())


def load_labels(path):
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    rows = []
    for ft in feats:
        q = ft['properties']
        lng, lat = ft['geometry']['coordinates']
        sev = q.get('severity')
        rows.append({'label_id': q['label_id'], 'user_id': q['user_id'],
                     'pano_id': q['pano_id'], 'region_id': q['region_id'],
                     'lat': float(lat), 'lng': float(lng),
                     'pano_x': q['pano_x'], 'pano_y': q['pano_y'],
                     'severity': float('nan') if sev is None else float(sev),
                     'label_type': q['label_type']})
    return pd.DataFrame(rows)


def load_server_clusters(path):
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    out = []
    for ft in feats:
        q = ft['properties']
        lng, lat = ft['geometry']['coordinates']
        out.append({'id': q['label_cluster_id'], 'label_ids': list(q['label_ids']),
                    'lat': lat, 'lng': lng})
    return out


def label_to_detection(results_path, labels):
    """{label_id: (pano_id, det_index)} by pano id + the pixel rounding of send_to_ps."""
    by_key = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            w, h = p['width'], p['height']
            for i, d in enumerate(rec.get('detections', [])):
                key = (p['panorama_id'], round(d['x_normalized'] * w),
                       round(d['y_normalized'] * h))
                if key in by_key:
                    raise SystemExit(f'pixel-key collision in results.jsonl: {key}')
                by_key[key] = i
    mapping = {}
    for row in labels.itertuples(index=False):
        i = by_key.get((row.pano_id, row.pano_x, row.pano_y))
        if i is not None:
            mapping[row.label_id] = (row.pano_id, i)
    return mapping


# --------------------------------------------------------------------------- clusters

@dataclass
class Cluster:
    id: int
    members: list                    # [(pano_id, det_index)] — AI labels only
    n_labels: int                    # every label, mapped or not (human labels too)
    e: float | None = None
    n: float | None = None
    server_latlng: tuple | None = None
    label_ids: list = field(default_factory=list)


def place(clusters, det_pos):
    for c in clusters:
        pts = [det_pos[m] for m in c.members if m in det_pos]
        if pts:
            c.e = sum(p[0] for p in pts) / len(pts)
            c.n = sum(p[1] for p in pts) / len(pts)
    return clusters


def clusters_from_assignment(labels, assignment, det_of):
    """labels: DataFrame; assignment: array of cluster ids aligned with labels rows."""
    groups = {}
    for lid, cid in zip(labels['label_id'].to_numpy(), assignment):
        groups.setdefault(int(cid), []).append(int(lid))
    out = []
    for k, (_cid, lids) in enumerate(sorted(groups.items())):
        out.append(Cluster(k, [det_of[lab] for lab in lids if lab in det_of], len(lids),
                           label_ids=lids))
    return out


def clusters_from_server(server_clusters, det_of):
    out = []
    for k, sc in enumerate(server_clusters):
        out.append(Cluster(k, [det_of[lab] for lab in sc['label_ids'] if lab in det_of],
                           len(sc['label_ids']), server_latlng=(sc['lat'], sc['lng']),
                           label_ids=list(sc['label_ids'])))
    return out


def clusters_from_sites(sites, refit_position):
    out = []
    for s in sites:
        members = [(d.pano_id, d.det_index) for d, _ in s.members if d.operational]
        if not members:
            continue
        c = Cluster(s.id, members, len(members))
        if refit_position:
            c.e, c.n = s.e, s.n
        out.append(c)
    return out


# ---------------------------------------------------------------- the PS algorithm

def ps_linkage(sub):
    """Complete-linkage tree over label_clustering.py's custom_dist for one group:
    haversine km between lat/lng, float max for two labels from the same (user, pano)."""
    pts = sub[['lat', 'lng']].to_numpy(dtype=float)
    dist = np.asarray(haversine_vector(pts, pts, comb=True), dtype=float)
    users = sub['user_id'].to_numpy()
    panos = sub['pano_id'].to_numpy()
    same = (users[:, None] == users[None, :]) & (panos[:, None] == panos[None, :])
    dist = np.where(same, sys.float_info.max, dist)
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='complete')


def ps_partition(labels, thresholds_km, per_region=True):
    """{threshold_km: cluster-id array aligned with labels rows} for one linkage per
    group, cut at every threshold (fcluster cuts the same tree the script builds)."""
    out = {t: np.zeros(len(labels), dtype=int) for t in thresholds_km}
    offset = {t: 0 for t in thresholds_km}
    if per_region:
        groups = [idx for _, idx in sorted(labels.groupby('region_id').indices.items())]
    else:
        groups = [np.arange(len(labels))]
    for idx in groups:
        sub = labels.iloc[idx]
        if len(sub) == 1:
            for t in thresholds_km:
                out[t][idx] = offset[t] + 1
                offset[t] += 1
            continue
        tree = ps_linkage(sub)
        for t in thresholds_km:
            cl = fcluster(tree, t=t, criterion='distance')
            out[t][idx] = cl + offset[t]
            offset[t] += int(cl.max())
    return out


def ps_verbatim(labels, ps_script):
    """Run the SidewalkWebpage script's own cluster() per region, exactly as the
    server invokes it (one region at a time, CurbRamp threshold)."""
    spec = importlib.util.spec_from_file_location('ps_label_clustering', ps_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assignment = np.zeros(len(labels), dtype=int)
    offset = 0
    for _, idx in sorted(labels.groupby('region_id').indices.items()):
        sub = labels.iloc[idx].copy()
        sub['coords'] = sub.apply(lambda r: (r.lat, r.lng), axis=1)
        if len(sub) > 1:
            _, labeled = mod.cluster(sub, 'CurbRamp', mod.THRESHOLDS)
            cl = labeled['cluster'].to_numpy(dtype=int)
        else:
            cl = np.array([1])
        assignment[idx] = cl + offset
        offset += int(cl.max())
    return assignment, mod.THRESHOLDS['CurbRamp']


def partition_agreement(a, b):
    """How much of partition a (list of Cluster) is reproduced by b: fraction of a's
    clusters whose label set appears verbatim in b, plus labels in disagreeing ones."""
    sets_b = {frozenset(c.label_ids) for c in b}
    same = [c for c in a if frozenset(c.label_ids) in sets_b]
    off_labels = sum(c.n_labels for c in a if frozenset(c.label_ids) not in sets_b)
    return len(same), len(a), off_labels


# ------------------------------------------------------------------------- scoring

def score(clusters, gt, det_pos, radius_m=5.0, frag_radii=(3.0, 5.0)):
    ramps, pool, points, op_verdicts = gt
    placed = [c for c in clusters if c.e is not None]
    matched = es.match_one_to_one(pool, placed, radius_m)
    matched_ids = {c.id for c in matched.values()}

    buckets = {'self_detected': 0, 'recovered_other_view': 0, 'unmatched': 0}
    for i, ramp in enumerate(pool):
        if ramp.self_detected:
            buckets['self_detected'] += 1
        elif i in matched:
            buckets['recovered_other_view'] += 1
        else:
            buckets['unmatched'] += 1

    tp = fp = unsure = 0
    for c in clusters:
        vs = [op_verdicts[m] for m in c.members if m in op_verdicts]
        if not vs:
            continue
        if any(v is True or v == 'duplicate' for v in vs):
            tp += 1
        elif any(v is False for v in vs):
            fp += 1
        else:
            unsure += 1

    frag = {}
    for rf in frag_radii:
        n_with = n_extra = 0
        for i, _c in matched.items():
            ramp = pool[i]
            extra = sum(1 for o in placed if o.id not in matched_ids
                        and math.hypot(o.e - ramp.e, o.n - ramp.n) <= rf)
            n_extra += extra
            n_with += extra > 0
        frag[rf] = {'ramps': len(matched), 'with_extra': n_with, 'extra': n_extra}

    # dual-ramp separation, as eval_sites (f)
    ramp_of_point = {id(pt): ri for ri, ramp in enumerate(ramps) for pt in ramp.points}
    pool_index = {id(ramp): i for i, ramp in enumerate(pool)}
    by_pano = {}
    for pt in points:
        by_pano.setdefault(pt.pano_id, []).append(pt)
    dual = {'pairs': 0, 'both': 0, 'one': 0, 'neither': 0}
    for pid in sorted(by_pano):
        pts = by_pano[pid]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if math.hypot(pts[i].e - pts[j].e, pts[i].n - pts[j].n) >= 5.0:
                    continue
                dual['pairs'] += 1
                hits = sum(1 for pt in (pts[i], pts[j])
                           if pool_index.get(id(ramps[ramp_of_point[id(pt)]])) in matched)
                dual['both' if hits == 2 else 'one' if hits == 1 else 'neither'] += 1

    # coherence: the cluster holding the self-detection vs its own GT ramp
    pos_key = {}
    for m, (e, n) in det_pos.items():
        pos_key[(m[0], round(e, 6), round(n, 6))] = m
    cluster_of = {m: c for c in clusters for m in c.members}
    offsets, missing = [], 0
    for ramp in pool:
        for pt in ramp.points:
            if pt.kind != 'det':
                continue
            m = pos_key.get((pt.pano_id, round(pt.e, 6), round(pt.n, 6)))
            c = cluster_of.get(m)
            if c is None or c.e is None:
                missing += 1
                continue
            offsets.append(math.hypot(c.e - ramp.e, c.n - ramp.n))
    offsets.sort()

    def pct(q):
        return offsets[min(len(offsets) - 1, int(q * len(offsets)))] if offsets else None

    same_pano_pairs = 0
    for c in clusters:
        seen = {}
        for m in c.members:
            seen[m[0]] = seen.get(m[0], 0) + 1
        same_pano_pairs += sum(k * (k - 1) // 2 for k in seen.values())

    n_pool = len(pool)
    recalled = buckets['self_detected'] + buckets['recovered_other_view']
    n_labels = sum(c.n_labels for c in clusters)
    return {
        'n_clusters': len(clusters), 'n_placed': len(placed), 'n_labels': n_labels,
        'labels_per_cluster': n_labels / len(clusters) if clusters else None,
        'precision': tp / (tp + fp) if tp + fp else None,
        'precision_ci': es.wilson(tp, tp + fp), 'tp': tp, 'fp': fp, 'unsure': unsure,
        'recall': recalled / n_pool if n_pool else None,
        'recall_ci': es.wilson(recalled, n_pool), 'buckets': buckets,
        'frag': frag, 'dual': dual,
        'coherence': {'n': len(offsets), 'median': pct(0.5), 'p90': pct(0.9),
                      'over_5m': sum(1 for o in offsets if o > 5.0), 'missing': missing},
        'same_pano_pairs': same_pano_pairs,
    }


# --------------------------------------------------------------------------- report

def fmt(v, nd=3):
    return 'n/a' if v is None else f'{v:.{nd}f}'


def frac(f):
    return f"{f['with_extra'] / f['ramps']:.2f}" if f['ramps'] else 'n/a'


def row_line(name, r):
    fr3, fr5, d, co = r['frag'][3.0], r['frag'][5.0], r['dual'], r['coherence']
    return (f"| {name} | {r['n_clusters']} | {r['n_labels']} | "
            f"{fmt(r['labels_per_cluster'], 2)} | {fmt(r['precision'])} "
            f"({r['tp']}/{r['fp']}) | {fmt(r['recall'])} | "
            f"{r['buckets']['self_detected']}/{r['buckets']['recovered_other_view']}/"
            f"{r['buckets']['unmatched']} | {frac(fr3)} ({fr3['extra']}) | "
            f"{frac(fr5)} ({fr5['extra']}) | {d['both']}/{d['one']}/{d['neither']} | "
            f"{fmt(co['median'], 2)} / {fmt(co['p90'], 2)} / {co['over_5m']} |")


TABLE_HEADER = (
    "| arm | clusters | labels | labels/cluster | precision (TP/FP) | recall | "
    "self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither "
    "| coherence med / p90 / >5 m |\n"
    "|---|---:|---:|---:|---|---|---|---|---|---|---|")


def csv_row(name, r):
    fr3, fr5, d, co = r['frag'][3.0], r['frag'][5.0], r['dual'], r['coherence']
    return {'arm': name, 'n_clusters': r['n_clusters'], 'n_placed': r['n_placed'],
            'n_labels': r['n_labels'], 'labels_per_cluster': r['labels_per_cluster'],
            'precision': r['precision'], 'tp': r['tp'], 'fp': r['fp'],
            'unsure': r['unsure'], 'recall': r['recall'],
            'self_detected': r['buckets']['self_detected'],
            'other_view': r['buckets']['recovered_other_view'],
            'unmatched': r['buckets']['unmatched'],
            'frag3_ramps': fr3['ramps'], 'frag3_with_extra': fr3['with_extra'],
            'frag3_extra': fr3['extra'], 'frag5_with_extra': fr5['with_extra'],
            'frag5_extra': fr5['extra'], 'dual_pairs': d['pairs'], 'dual_both': d['both'],
            'dual_one': d['one'], 'dual_neither': d['neither'],
            'coh_n': co['n'], 'coh_median': co['median'], 'coh_p90': co['p90'],
            'coh_over_5m': co['over_5m'], 'same_pano_pairs': r['same_pano_pairs']}


def quantile(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/ps_clustering_eval)')
    ap.add_argument('--server', default=None,
                    help='PS server URL; downloads the two geojson files if absent')
    ap.add_argument('--labels', type=Path, default=None)
    ap.add_argument('--clusters', type=Path, default=None)
    ap.add_argument('--ps-script', type=Path, default=None,
                    help='SidewalkWebpage/scripts/label_clustering.py for the verbatim arm')
    ap.add_argument('--thresholds-m', type=float, nargs='*',
                    default=[2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--radius-sweep', type=float, nargs='*', default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument('--gt-merge-m', type=float, default=2.5)
    ap.add_argument('--camera-height-m', type=float, default=geo.DEFAULT_CAMERA_HEIGHT_M,
                    help='raycast camera height for every placement (GT, clusters, '
                         'fusion); the #101/#158 sensitivity knob')
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    out = args.out or run_dir / 'ps_clustering_eval'
    out.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels or out / 'raw_labels.geojson'
    clusters_path = args.clusters or out / 'clusters.geojson'
    if args.server:
        fetch(args.server.rstrip('/') + API_LABELS, labels_path)
        fetch(args.server.rstrip('/') + API_CLUSTERS, clusters_path)

    labels = load_labels(labels_path)
    labels = labels[labels.label_type == 'CurbRamp'].reset_index(drop=True)
    server_clusters = load_server_clusters(clusters_path)
    det_of = label_to_detection(run_dir / 'results.jsonl', labels)
    ai = labels[labels.label_id.isin(det_of)].reset_index(drop=True)
    lines = [f'# {args.city}: PS label clustering vs RampNet GT',
             '',
             f'labels: {len(labels)} CurbRamp on the server, {len(ai)} map to stored '
             f'detections (AI), {len(labels) - len(ai)} do not (human); '
             f'{len(server_clusters)} server clusters over '
             f'{sum(len(c["label_ids"]) for c in server_clusters)} labels']

    # world frame, raycast positions, GT — one code path with eval_sites
    params = fs.FuseParams(camera_height_m=args.camera_height_m)
    lines.append(f'raycast camera height {args.camera_height_m:g} m')
    verdict_panos, bundle_ops, run_panos = es.load_city_files(
        args.city, args.benchmark_root, run_dir)
    dets, frame, drops = fs.project(run_panos, params)
    det_pos = {(d.pano_id, d.det_index): (d.e, d.n) for d in dets}
    points, op_verdicts, counts, warnings = es.build_gt(
        verdict_panos, bundle_ops, {p.pano_id: p for p in run_panos}, params, frame)
    ramps = es.merge_gt_points(points, args.gt_merge_m)
    pool = [r for r in ramps if r.in_pool]
    gt = (ramps, pool, points, op_verdicts)
    lines.append(f"GT: {counts['judged']} judged panos -> {counts['placeable']} placeable "
                 f"points -> {len(ramps)} ramps ({len(points) - len(ramps)} cross-pano "
                 f"merges), {len(pool)} in the recall pool; raycast placed {len(dets)} of "
                 f"{sum(len(p.detections) for p in run_panos)} detections "
                 f"(drops {drops})")
    for w in warnings:
        lines.append(f'warning: {w}')

    results = {}

    # arm: deployed
    deployed = place(clusters_from_server(server_clusters, det_of), det_pos)
    results['deployed'] = score(deployed, gt, det_pos, args.match_radius_m)

    # descriptive: server centroid vs raycast centroid; per-label placement offset
    offs = []
    for c in deployed:
        if c.e is not None and c.server_latlng:
            lat, lng = frame.to_latlng(c.e, c.n)
            offs.append(geo.haversine_m(lat, lng, *c.server_latlng))
    offs.sort()
    lab_offs = []
    for row in ai.itertuples(index=False):
        m = det_of[row.label_id]
        if m in det_pos:
            lat, lng = frame.to_latlng(*det_pos[m])
            lab_offs.append(geo.haversine_m(lat, lng, row.lat, row.lng))
    lab_offs.sort()
    sizes = {}
    for c in deployed:
        sizes[c.n_labels] = sizes.get(c.n_labels, 0) + 1

    # arm: ps_repro (verbatim script) — on the labels the server actually clustered
    checks = []
    clustered_ids = {lab for c in server_clusters for lab in c['label_ids']}
    on_server = labels[labels.label_id.isin(clustered_ids)].reset_index(drop=True)
    if args.ps_script:
        assign, t_km = ps_verbatim(on_server, args.ps_script)
        repro = place(clusters_from_assignment(on_server, assign, det_of), det_pos)
        results['ps_repro'] = score(repro, gt, det_pos, args.match_radius_m)
        k, n, off = partition_agreement(deployed, repro)
        checks.append(f'ps_repro reproduces deployed: {k}/{n} clusters identical '
                      f'({off} labels in clusters that differ); script threshold {t_km} km')
        vec = ps_partition(on_server, [PS_THRESHOLD_KM])[PS_THRESHOLD_KM]
        vec_clusters = clusters_from_assignment(on_server, vec, det_of)
        k2, n2, off2 = partition_agreement(repro, vec_clusters)
        checks.append(f'vectorized PS distance reproduces the script: {k2}/{n2} clusters '
                      f'identical ({off2} labels differ)')

    # arms: ps @ t (server positions, per region) and ps_citywide @ 7.5
    t_kms = [t / 1000.0 for t in args.thresholds_m]
    parts = ps_partition(ai, t_kms, per_region=True)
    for t_m, t_km in zip(args.thresholds_m, t_kms):
        cl = place(clusters_from_assignment(ai, parts[t_km], det_of), det_pos)
        results[f'ps @ {t_m:g} m'] = score(cl, gt, det_pos, args.match_radius_m)
    city = ps_partition(ai, [PS_THRESHOLD_KM], per_region=False)[PS_THRESHOLD_KM]
    results['ps_citywide @ 7.5 m'] = score(
        place(clusters_from_assignment(ai, city, det_of), det_pos), gt, det_pos,
        args.match_radius_m)

    # arms: ps_placeable @ t — the PS algorithm on server positions, restricted to the
    # labels the raycast can place: exactly the label set ps_raycast and fusion use, so
    # against ps_raycast the only difference is the positions
    placeable = ai[[det_of[lab] in det_pos for lab in ai.label_id]].reset_index(drop=True)
    parts_pl = ps_partition(placeable, t_kms, per_region=True)
    for t_m, t_km in zip(args.thresholds_m, t_kms):
        cl = place(clusters_from_assignment(placeable, parts_pl[t_km], det_of), det_pos)
        results[f'ps_placeable @ {t_m:g} m'] = score(cl, gt, det_pos, args.match_radius_m)

    # arms: ps_raycast @ t — same algorithm, labeler raycast positions
    ray = ai.copy()
    keep = []
    for i, row in enumerate(ai.itertuples(index=False)):
        m = det_of[row.label_id]
        if m in det_pos:
            lat, lng = frame.to_latlng(*det_pos[m])
            ray.loc[i, 'lat'] = lat
            ray.loc[i, 'lng'] = lng
            keep.append(i)
    ray = ray.iloc[keep].reset_index(drop=True)
    parts_ray = ps_partition(ray, t_kms, per_region=True)
    for t_m, t_km in zip(args.thresholds_m, t_kms):
        cl = place(clusters_from_assignment(ray, parts_ray[t_km], det_of), det_pos)
        results[f'ps_raycast @ {t_m:g} m'] = score(cl, gt, det_pos, args.match_radius_m)

    # arm: fusion (mean-of-members placement, and refit position as the tie-back)
    sites, _frame2, _stats = fs.fuse(run_panos, params)
    results['fusion'] = score(place(clusters_from_sites(sites, False), det_pos), gt,
                              det_pos, args.match_radius_m)
    results['fusion_refit'] = score(clusters_from_sites(sites, True), gt, det_pos,
                                    args.match_radius_m)

    # mechanism: same-ramp scatter. Over fusion sites with >= 3 placeable members, the
    # largest pairwise distance among the members under server positions vs raycast
    # positions — complete linkage at t keeps a group together only if this is <= t.
    server_ll = {det_of[r.label_id]: (r.lat, r.lng) for r in ai.itertuples(index=False)}
    spread = {'server': [], 'raycast': []}
    for s in sites:
        mem = [(d.pano_id, d.det_index) for d, _ in s.members
               if d.operational and (d.pano_id, d.det_index) in det_pos]
        if len(mem) < 3:
            continue
        for key, pts in (('server', [server_ll[m] for m in mem]),
                         ('raycast', [frame.to_latlng(*det_pos[m]) for m in mem])):
            spread[key].append(max(geo.haversine_m(*pts[i], *pts[j])
                                   for i in range(len(pts))
                                   for j in range(i + 1, len(pts))))
    for v in spread.values():
        v.sort()

    # radius sweep for the two arms that matter most
    sweep = []
    for r_m in args.radius_sweep:
        sweep.append((r_m,
                      score(deployed, gt, det_pos, r_m),
                      score(place(clusters_from_sites(sites, False), det_pos), gt,
                            det_pos, r_m)))

    # ---- report
    lines += ['', '## Validation checks', '']
    lines += [f'- {c}' for c in checks] or ['- ps_repro skipped (no --ps-script)']
    fr = results['fusion_refit']
    lines.append(f"- fusion_refit vs runs/{args.city}/fusion_eval/report.md: precision "
                 f"{fmt(fr['precision'])}, recall {fmt(fr['recall'])}, dual "
                 f"{fr['dual']['both']}/{fr['dual']['one']}/{fr['dual']['neither']}")
    bad = {k: v['same_pano_pairs'] for k, v in results.items() if v['same_pano_pairs']}
    lines.append('- same-pano pairs inside one cluster (must be 0 under the cannot-link): '
                 + (', '.join(f'{k} {v}' for k, v in bad.items()) if bad
                    else '0 in every arm'))
    lines += ['', f'## Arms (match radius {args.match_radius_m:g} m, GT merge '
              f'{args.gt_merge_m:g} m)', '', TABLE_HEADER]
    lines += [row_line(k, v) for k, v in results.items()]
    lines += ['', "frag = share of matched GT ramps with at least one extra cluster within "
              "r that is nobody's match (total extras in parentheses); coherence = "
              "distance from a self-detected GT ramp to the centroid of the cluster "
              "holding that label."]
    lines += ['', '## Deployed clusters, descriptive', '',
              '- cluster size histogram (labels -> clusters): '
              + ', '.join(f'{k}: {v}' for k, v in sorted(sizes.items())),
              f'- server centroid vs raycast centroid, same members (n={len(offs)}): '
              f'median {fmt(quantile(offs, .5), 1)} m, p90 {fmt(quantile(offs, .9), 1)} m',
              f'- per-label server lat/lng vs labeler raycast (n={len(lab_offs)}): '
              f'median {fmt(quantile(lab_offs, .5), 1)} m, p90 '
              f'{fmt(quantile(lab_offs, .9), 1)} m, over 5 m '
              f'{sum(1 for o in lab_offs if o > 5) / len(lab_offs):.2f}']
    lines += ['', '## Same-ramp scatter (mechanism)', '',
              f'Largest pairwise member distance over {len(spread["server"])} fusion '
              f'sites with >= 3 placeable members (a complete-linkage cut at t keeps the '
              f'group only if this is <= t):', '',
              '| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |',
              '|---|---:|---:|---:|---:|---:|']
    for key, v in spread.items():
        lines.append(f'| {key} | {fmt(quantile(v, .5), 1)} m | {fmt(quantile(v, .9), 1)} m '
                     f'| {sum(1 for x in v if x > 7.5) / len(v):.2f} '
                     f'| {sum(1 for x in v if x > 10) / len(v):.2f} '
                     f'| {sum(1 for x in v if x > 15) / len(v):.2f} |')
    missing = {k: v['coherence']['missing'] for k, v in results.items()
               if v['coherence']['missing']}
    lines += ['', '- self-detections whose cluster could not be located (should be 0): '
              + (', '.join(f'{k} {v}' for k, v in missing.items()) if missing
                 else '0 in every arm')]
    lines += ['', '## Match-radius sweep (deployed vs fusion)', '',
              '| radius m | deployed recall | deployed frag 5 m | deployed dual both | '
              'fusion recall | fusion frag 5 m | fusion dual both |',
              '|---:|---|---|---|---|---|---|']
    for r_m, a, b in sweep:
        fa, fb = a['frag'][5.0], b['frag'][5.0]
        lines.append(f"| {r_m:g} | {fmt(a['recall'])} | {fa['with_extra']}/{fa['ramps']} | "
                     f"{a['dual']['both']}/{a['dual']['pairs']} | {fmt(b['recall'])} | "
                     f"{fb['with_extra']}/{fb['ramps']} | "
                     f"{b['dual']['both']}/{b['dual']['pairs']} |")

    report = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    with open(out / 'arms.csv', 'w', newline='', encoding='utf-8') as f:
        rows = [csv_row(k, v) for k, v in results.items()]
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(report)
    print(f'wrote {out / "report.md"} and arms.csv')


if __name__ == '__main__':
    main()
