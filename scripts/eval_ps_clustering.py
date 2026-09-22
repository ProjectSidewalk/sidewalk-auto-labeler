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
  ps_placeable @ t ...the same, restricted to the labels the raycast can place, so
                  it is exactly the label set ps_raycast and fusion use
  ps_raycast @ t  the PS algorithm on the labeler's raycast positions (isolates
                  placement from algorithm)
  fusion          fuse_sites.py's ray-aware associator (the labeler's reference)

The headline metric is `coverage` (a cluster of this arm within the match radius
of a pool GT ramp). `recall (union)` is eval_sites' definition, kept only for the
tie-back: it counts a self-detected ramp as recovered whether or not any cluster
landed on it, which pins ~83% of it constant across arms.

Every cluster is placed at the mean of its members' raycast positions, so scores
depend only on who was grouped with whom. Needs pandas, scipy and `haversine`
(the package the PS script uses); none of them is a pipeline dependency, so
install them alongside requirements.txt: pip install pandas scipy haversine.

Usage:
    python scripts/eval_ps_clustering.py richmond \
        --server https://sidewalk-richmond.cs.washington.edu \
        --ps-script /path/to/SidewalkWebpage/scripts/label_clustering.py
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

try:  # analysis-only dependencies, deliberately not in requirements.txt
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from haversine import haversine_vector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f'{exc.name} is missing. This analysis tool needs three packages the '
        'pipeline does not: pip install pandas scipy haversine '
        '(haversine is the package label_clustering.py itself uses).') from exc

PS_THRESHOLD_KM = 0.0075   # label_clustering.py THRESHOLDS['CurbRamp']
API_LABELS = '/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson'
API_CLUSTERS = ('/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true'
                '&filetype=geojson')
# A dense N x N float64 matrix plus its bool mask and condensed copy; the citywide
# arm is the only path that builds one over every label in the city.
CITYWIDE_MAX_LABELS = 20000


# ----------------------------------------------------------------------------- inputs

def fetch(url, dest, refresh=False):
    """Download url to dest unless it is already there, and record its provenance.

    The deployed partition is regenerated whenever the server re-clusters, so a
    cached pull has to say *when* it was taken, and reusing one has to be visible:
    a cache hit prints the pull's age, and --refresh re-pulls over it. Written
    temp-then-rename, and a body that is not a non-empty FeatureCollection is
    refused rather than cached (a zero-feature 200 would otherwise score as "the
    city has no labels").
    """
    if dest.exists() and not refresh:
        age = pull_age_days(dest)
        how_old = 'age unknown' if age is None else f'pulled {age:.1f} days ago'
        print(f'reusing cached {dest.name} ({how_old}); --refresh re-pulls it',
              file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 (https, fixed host)
        tmp.write_bytes(r.read())
    try:
        n = len(json.loads(tmp.read_text(encoding='utf-8'))['features'])
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f'{url}: not a GeoJSON FeatureCollection ({exc})') from exc
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f'{url}: zero features — refusing to cache an empty pull')
    tmp.replace(dest)
    (dest.parent / (dest.name + '.source.json')).write_text(
        json.dumps({'url': url, 'n_features': n,
                    'fetched_at': datetime.now(timezone.utc)
                    .isoformat(timespec='seconds')}, indent=1), encoding='utf-8')


def source_meta(path):
    """The .source.json fetch record beside path, or {} when there is none."""
    side = path.parent / (path.name + '.source.json')
    if not side.exists():
        return {}
    try:
        return json.loads(side.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def pull_age_days(path):
    """How long ago this file was pulled, per its fetch record; None if unrecorded."""
    when = source_meta(path).get('fetched_at')
    if not when:
        return None
    try:
        stamp = datetime.fromisoformat(when)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0


def provenance(path, n_features):
    """One report line per input file: where it came from, when, and its sha256."""
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = source_meta(path)
    age = pull_age_days(path)
    how_old = '' if age is None else f' ({age:.1f} days old at run time)'
    when = (meta.get('fetched_at', '') + how_old) or (
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .isoformat(timespec='seconds') + ' (file mtime; not a recorded fetch)')
    where = meta.get('url') or '(supplied on the command line)'
    return f'- `{path.name}`: {n_features} features, sha256 `{h}`, {when}, from {where}'


def load_labels(path):
    """(DataFrame, n_dropped). Mirrors label_clustering.clean_label_data: rows whose
    lng is null or > 360 are dropped, because the server drops them before
    clustering (corrupt values of order 1e14 have been observed upstream)."""
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    rows, dropped = [], 0
    for ft in feats:
        q = ft['properties']
        lng, lat = ft['geometry']['coordinates']
        lng = float('nan') if lng is None else float(lng)
        lat = float('nan') if lat is None else float(lat)
        if math.isnan(lng) or lng > 360:
            dropped += 1
            continue
        sev = q.get('severity')
        rows.append({'label_id': q['label_id'], 'user_id': q['user_id'],
                     'pano_id': q['pano_id'], 'region_id': q['region_id'],
                     'lat': lat, 'lng': lng,
                     'pano_x': q['pano_x'], 'pano_y': q['pano_y'],
                     # severity is unused here; the verbatim PS cluster() reads it.
                     'severity': float('nan') if sev is None else float(sev),
                     'label_type': q['label_type']})
    return pd.DataFrame(rows), dropped


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
    """({label_id: (pano_id, det_index)}, n_ambiguous_pixel_keys, n_duplicate_labels).

    Keyed by pano id + the pixel rounding send_to_ps.py used. Both directions can
    be many-to-one and both are reported rather than collapsed silently:

    - two stored *detections* that round to the same pixel make that key
      ambiguous, so the key is dropped (no label is attributed to the wrong
      detection) and counted;
    - two *labels* at one pixel — a re-submitted campaign, which is what laurens
      got on -test (SidewalkWebpage#5382) — both map to one detection. The
      same-(user, pano) cannot-link then forces them into different clusters,
      where they read as a fragment, so the `same_pano_pairs` tripwire cannot
      catch them. The count is reported instead.
    """
    by_key, ambiguous = {}, set()
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
                    ambiguous.add(key)
                by_key[key] = i
    for key in ambiguous:
        by_key.pop(key, None)
    mapping = {}
    for row in labels.itertuples(index=False):
        i = by_key.get((row.pano_id, row.pano_x, row.pano_y))
        if i is not None:
            mapping[row.label_id] = (row.pano_id, i)
    n_duplicate_labels = len(mapping) - len(set(mapping.values()))
    return mapping, len(ambiguous), n_duplicate_labels


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


def region_groups(labels):
    """Positional index arrays, one per region. groupby drops null keys, so a label
    with no region_id would silently never be assigned — refuse instead."""
    missing = int(labels['region_id'].isna().sum())
    if missing:
        raise SystemExit(f'{missing} labels have a null region_id; per-region '
                         'clustering cannot place them, and silently lumping them '
                         'together would fabricate one giant cluster')
    return [idx for _, idx in sorted(labels.groupby('region_id').indices.items())]


def ps_partition(labels, thresholds_km, per_region=True):
    """{threshold_km: cluster-id array aligned with labels rows} for one linkage per
    group, cut at every threshold (fcluster cuts the same tree the script builds)."""
    # -1, not 0: every row must be assigned, and an unassigned row has to be loud
    # rather than collapsing into a fabricated cluster 0.
    out = {t: np.full(len(labels), -1, dtype=int) for t in thresholds_km}
    offset = {t: 0 for t in thresholds_km}
    groups = region_groups(labels) if per_region else [np.arange(len(labels))]
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
    for t, arr in out.items():
        if (arr < 0).any():
            raise SystemExit(f'{int((arr < 0).sum())} labels were never assigned a '
                             f'cluster at {t} km — the grouping dropped rows')
    return out


def dense_matrix_gb(n):
    """Peak memory of ps_linkage over n labels: float64 matrix + bool mask +
    condensed copy."""
    return (n * n * 8 * 1.5 + n * n) / 1024 ** 3


def ps_verbatim(labels, ps_script):
    """Run the SidewalkWebpage script's own cluster() per region, exactly as the
    server invokes it (one region at a time, CurbRamp threshold)."""
    spec = importlib.util.spec_from_file_location('ps_label_clustering', ps_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assignment = np.full(len(labels), -1, dtype=int)
    offset = 0
    for idx in region_groups(labels):
        sub = labels.iloc[idx].copy()
        sub['coords'] = sub.apply(lambda r: (r.lat, r.lng), axis=1)
        if len(sub) > 1:
            _, labeled = mod.cluster(sub, 'CurbRamp', mod.THRESHOLDS)
            cl = labeled['cluster'].to_numpy(dtype=int)
        else:
            cl = np.array([1])
        assignment[idx] = cl + offset
        offset += int(cl.max())
    if (assignment < 0).any():
        raise SystemExit(f'{int((assignment < 0).sum())} labels were never assigned '
                         'a cluster by the verbatim script')
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
    # A cluster is "somebody's match" if it matches ANY GT ramp, not only a pool
    # one: a ramp on a pano whose missed-check was not confirmed is still a real
    # ramp, and a cluster sitting on it is not a fragment.
    matched_all = es.match_one_to_one(ramps, placed, radius_m)
    matched_ids = {c.id for c in matched_all.values()}

    # Two recall-shaped numbers, deliberately both reported:
    #  - coverage: a cluster OF THIS ARM is within radius_m. This is what RQ2a
    #    asks and the only one of the two that responds to the partition.
    #  - recall: eval_sites' union recall, which counts a self-detected ramp as
    #    recovered whether or not any cluster landed on it. ~83% of Richmond's
    #    pool is self-detected, so it is nearly constant across arms; keep it
    #    only to tie back to fusion_eval/report.md.
    buckets = {'self_detected': 0, 'recovered_other_view': 0, 'unmatched': 0}
    self_detected_without_cluster = 0
    for i, ramp in enumerate(pool):
        if ramp.self_detected:
            buckets['self_detected'] += 1
            if i not in matched:
                self_detected_without_cluster += 1
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

    # dual-ramp separation, as eval_sites (f). Matched against every GT ramp for
    # the same reason as the fragment count above, so a pair involving a non-pool
    # ramp can still be scored as kept apart.
    ramp_of_point = {id(pt): ri for ri, ramp in enumerate(ramps) for pt in ramp.points}
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
                           if ramp_of_point[id(pt)] in matched_all)
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
    covered = len(matched)
    n_labels = sum(c.n_labels for c in clusters)
    return {
        'n_clusters': len(clusters), 'n_placed': len(placed), 'n_labels': n_labels,
        'labels_per_cluster': n_labels / len(clusters) if clusters else None,
        'precision': tp / (tp + fp) if tp + fp else None,
        'precision_ci': es.wilson(tp, tp + fp), 'tp': tp, 'fp': fp, 'unsure': unsure,
        'coverage': covered / n_pool if n_pool else None,
        'coverage_ci': es.wilson(covered, n_pool), 'covered': covered,
        'n_pool': n_pool,
        'self_detected_without_cluster': self_detected_without_cluster,
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
    return (f"| {name} | {r['n_clusters']} | {r['n_placed']} | {r['n_labels']} | "
            f"{fmt(r['labels_per_cluster'], 2)} | {fmt(r['precision'])} "
            f"({r['tp']}/{r['fp']}) | {fmt(r['coverage'])} "
            f"({r['covered']}/{r['n_pool']}) | "
            f"{r['self_detected_without_cluster']} | {fmt(r['recall'])} | "
            f"{r['buckets']['self_detected']}/{r['buckets']['recovered_other_view']}/"
            f"{r['buckets']['unmatched']} | {frac(fr3)} ({fr3['extra']}) | "
            f"{frac(fr5)} ({fr5['extra']}) | {d['both']}/{d['one']}/{d['neither']} | "
            f"{fmt(co['median'], 2)} / {fmt(co['p90'], 2)} / {co['over_5m']} |")


TABLE_HEADER = (
    "| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | "
    "coverage | no cluster | recall (union) | self/other/unmatched | "
    "frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither "
    "| coherence med / p90 / >5 m |\n"
    "|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|")

def table_legend(results):
    """The legend under the arms table. The self-detected / pool counts are read
    from this run (they are frame-dependent: Richmond is 210/253 at 2.6 m and
    210/260 at 2.341 m), never hardcoded."""
    any_arm = next(iter(results.values()))
    n_self = any_arm['buckets']['self_detected']
    n_pool = any_arm['n_pool']
    share = f'{n_self / n_pool:.0%}' if n_pool else 'most'
    return (
        "**coverage** = pool GT ramps with a cluster of this arm within the match "
        "radius, matched one-to-one — the metric RQ2a asks for, and the only "
        "recall-shaped one that responds to the partition. **no cluster** = ramps "
        "counted as recalled by the union metric although no cluster is within the "
        "radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = "
        "`eval_sites`' definition, which counts a self-detected ramp as recovered "
        f"whether or not any cluster landed on it; {n_self} of this run's {n_pool} "
        f"pool ramps are self-detected, so {share} of it is constant across arms "
        "and it is kept only to tie back to `fusion_eval/report.md`. **frag** = "
        "share of covered GT ramps with at least one extra cluster within r that is "
        "not the one-to-one match of any GT ramp (total extras in parentheses). "
        "**coherence** = distance from a self-detected GT ramp to the centroid of "
        "the cluster holding that label.")


def csv_row(name, r):
    fr3, fr5, d, co = r['frag'][3.0], r['frag'][5.0], r['dual'], r['coherence']
    return {'arm': name, 'n_clusters': r['n_clusters'], 'n_placed': r['n_placed'],
            'n_labels': r['n_labels'], 'labels_per_cluster': r['labels_per_cluster'],
            'precision': r['precision'], 'tp': r['tp'], 'fp': r['fp'],
            'unsure': r['unsure'], 'coverage': r['coverage'],
            'covered': r['covered'], 'n_pool': r['n_pool'],
            'self_detected_without_cluster': r['self_detected_without_cluster'],
            'recall_union': r['recall'],
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
    ap.add_argument('--refresh', action='store_true',
                    help='with --server, re-pull the two geojson over the cached copies '
                         '(the server re-clusters nightly, so a cached pull goes stale)')
    ap.add_argument('--labels', type=Path, default=None)
    ap.add_argument('--clusters', type=Path, default=None)
    ap.add_argument('--ps-script', type=Path, default=None,
                    help='SidewalkWebpage/scripts/label_clustering.py for the verbatim arm')
    ap.add_argument('--thresholds-m', type=float, nargs='+',
                    default=[2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--radius-sweep', type=float, nargs='*', default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument('--gt-merge-m', type=float, default=2.5)
    ap.add_argument('--camera-height-m', type=float, default=geo.DEFAULT_CAMERA_HEIGHT_M,
                    help='raycast camera height for every placement (GT, clusters, '
                         'fusion); the #101/#158 sensitivity knob')
    ap.add_argument('--min-confidence', type=float, default=BENCHMARK_CONFIDENCE,
                    help='the tier the fusion arm runs at. Defaults to the BENCHMARK tier '
                         f'({BENCHMARK_CONFIDENCE}), not the operating point: this script '
                         'scores what the SERVER holds, and every city clustered so far went '
                         'live at 0.55. Set it to whatever a city was actually submitted at '
                         '(its submission record says) before comparing arms.')
    ap.add_argument('--citywide-max-labels', type=int, default=CITYWIDE_MAX_LABELS,
                    help='skip the ps_citywide arm above this many labels (it builds '
                         'a dense N x N distance matrix)')
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    # The scoring frame is part of the result, so a non-default height gets its own
    # directory instead of silently overwriting the default one.
    default_out = 'ps_clustering_eval' if (
        args.camera_height_m == geo.DEFAULT_CAMERA_HEIGHT_M
    ) else f'ps_clustering_eval_h{args.camera_height_m:.2f}'
    out = args.out or run_dir / default_out
    out.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels or out / 'raw_labels.geojson'
    clusters_path = args.clusters or out / 'clusters.geojson'
    if args.server:
        fetch(args.server.rstrip('/') + API_LABELS, labels_path, args.refresh)
        fetch(args.server.rstrip('/') + API_CLUSTERS, clusters_path, args.refresh)
    elif args.refresh:
        raise SystemExit('--refresh needs --server (there is nothing to re-pull from)')

    labels, n_bad_lng = load_labels(labels_path)
    labels = labels[labels.label_type == 'CurbRamp'].reset_index(drop=True)
    server_clusters = load_server_clusters(clusters_path)
    det_of, n_ambiguous, n_dup_labels = label_to_detection(
        run_dir / 'results.jsonl', labels)
    ai = labels[labels.label_id.isin(det_of)].reset_index(drop=True)
    # "AI label" is inferred from the pixel match; make sure that inference picks out
    # exactly one submitting account, otherwise a human label at the same pixel as a
    # detection would be scored as an AI label.
    ai_users = sorted({str(u) for u in ai.user_id})
    if len(ai_users) != 1:
        raise SystemExit('labels that map to stored detections span '
                         f'{len(ai_users)} user_ids ({", ".join(ai_users[:5])}); '
                         'the pixel map cannot be read as "the AI user\'s labels"')
    ai_user = ai_users[0]
    unmapped_ai = int((labels.user_id.astype(str) == ai_user).sum()) - len(ai)
    by_user = labels.user_id.astype(str).value_counts()
    user_breakdown = ', '.join(
        f'{u}{" (AI)" if u == ai_user else ""} {int(c)}'
        for u, c in by_user.items())
    lines = [f'# {args.city}: PS label clustering vs RampNet GT',
             '',
             f'labels: {len(labels)} CurbRamp on the server, {len(ai)} map to stored '
             f'detections (AI), {len(labels) - len(ai)} do not (human); '
             f'{len(server_clusters)} server clusters over '
             f'{sum(len(c["label_ids"]) for c in server_clusters)} labels']

    # world frame, raycast positions, GT — one code path with eval_sites
    params = fs.FuseParams(camera_height_m=args.camera_height_m,
                           min_confidence=args.min_confidence)
    lines.append(f'raycast camera height {args.camera_height_m:g} m; '
                 f'fusion arm at --min-confidence {args.min_confidence:g}')
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
    if len(ai) <= args.citywide_max_labels:
        city = ps_partition(ai, [PS_THRESHOLD_KM], per_region=False)[PS_THRESHOLD_KM]
        results['ps_citywide @ 7.5 m'] = score(
            place(clusters_from_assignment(ai, city, det_of), det_pos), gt, det_pos,
            args.match_radius_m)
    else:
        checks.append(
            f'ps_citywide skipped: {len(ai)} labels exceed --citywide-max-labels '
            f'{args.citywide_max_labels}; one dense matrix over them would need about '
            f'{dense_matrix_gb(len(ai)):.1f} GB (per-region arms are unaffected)')

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
    #
    # The fusion arm is built from the RUN's detections and the ps_* arms from the
    # SERVER's labels. They coincide only when every placeable operational detection
    # was submitted; a gap-filled pano or a partial campaign breaks that, so a member
    # with no server label is skipped and counted rather than raising a KeyError here
    # (and the equality is reported as a validation check below).
    server_ll = {det_of[r.label_id]: (r.lat, r.lng) for r in ai.itertuples(index=False)}
    n_fusion_members = n_fusion_unsubmitted = 0
    spread = {'server': [], 'raycast': []}
    for s in sites:
        mem = [(d.pano_id, d.det_index) for d, _ in s.members
               if d.operational and (d.pano_id, d.det_index) in det_pos]
        n_fusion_members += len(mem)
        n_fusion_unsubmitted += sum(1 for m in mem if m not in server_ll)
        mem = [m for m in mem if m in server_ll]
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
    lines += ['', '## Data provenance', '',
              provenance(labels_path, len(labels)),
              provenance(clusters_path, len(server_clusters)),
              f'- labels by account: {user_breakdown}',
              f'- {n_bad_lng} labels dropped before clustering (null lng or lng > 360), '
              'matching label_clustering.clean_label_data',
              f'- {n_ambiguous} ambiguous pixel keys in results.jsonl (two stored '
              'detections round to one pixel; those keys are left unmapped)',
              f'- {n_dup_labels} server labels share a pixel with another label and so '
              'map to the same stored detection (a re-submitted campaign does this)']

    lines += ['', '## Validation checks', '']
    if not args.ps_script:
        checks.insert(0, 'ps_repro skipped (no --ps-script)')
    lines += [f'- {c}' for c in checks]

    def check_line(text, ok):
        return f'- {"" if ok else "warning: "}{text}'

    lines.append(check_line(
        'every label that maps to a stored detection belongs to one account '
        f'({ai_user}); {unmapped_ai} of that account\'s labels did not map '
        '(should be 0)', unmapped_ai == 0))
    ps_pl = results.get(f'ps_placeable @ {args.thresholds_m[0]:g} m')
    lines.append(check_line(
        'fusion arm vs ps_* arms cover the same labels: '
        f"{results['fusion']['n_labels']} fusion members vs "
        f"{ps_pl['n_labels'] if ps_pl else 'n/a'} placeable server labels; "
        f'{n_fusion_unsubmitted} of {n_fusion_members} placeable operational '
        'detections have no label on the server (should be 0; they are excluded '
        'from the scatter below)',
        n_fusion_unsubmitted == 0
        and (ps_pl is None or ps_pl['n_labels'] == results['fusion']['n_labels'])))
    fr = results['fusion_refit']
    # The published fusion_eval numbers were produced in the labeler's default frame,
    # so this only reproduces them when this run is scored at that height; at any other
    # --camera-height-m the world-space columns are expected to differ.
    same_frame = args.camera_height_m == geo.DEFAULT_CAMERA_HEIGHT_M
    lines.append(
        f"- fusion_refit at {args.camera_height_m:g} m vs runs/{args.city}/fusion_eval/"
        f"report.md (published in the {geo.DEFAULT_CAMERA_HEIGHT_M:g} m frame): precision "
        f"{fmt(fr['precision'])}, recall (union) {fmt(fr['recall'])}, dual "
        f"{fr['dual']['both']}/{fr['dual']['one']}/{fr['dual']['neither']}"
        + ('' if same_frame else ' — a different frame, so the world-space figures are '
                                 'expected to differ; precision is the frame-free part'))
    bad = {k: v['same_pano_pairs'] for k, v in results.items() if v['same_pano_pairs']}
    lines.append('- same-pano pairs inside one cluster (must be 0 under the cannot-link): '
                 + (', '.join(f'{k} {v}' for k, v in bad.items()) if bad
                    else '0 in every arm'))
    lines += ['', f'## Arms (match radius {args.match_radius_m:g} m, GT merge '
              f'{args.gt_merge_m:g} m)', '', TABLE_HEADER]
    lines += [row_line(k, v) for k, v in results.items()]
    lines += ['', table_legend(results)]
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
              '| radius m | deployed coverage | deployed frag 5 m | deployed dual both | '
              'fusion coverage | fusion frag 5 m | fusion dual both |',
              '|---:|---|---|---|---|---|---|']
    for r_m, a, b in sweep:
        fa, fb = a['frag'][5.0], b['frag'][5.0]
        lines.append(f"| {r_m:g} | {fmt(a['coverage'])} | {fa['with_extra']}/{fa['ramps']} "
                     f"| {a['dual']['both']}/{a['dual']['pairs']} | {fmt(b['coverage'])} | "
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
