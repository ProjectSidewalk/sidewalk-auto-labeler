"""Score label clustering against city curb-ramp inventories (issue #106, Part 2).

RampNet GT judges de-clustered panos, so it never says which labels from different panos are
one ramp. Bend and Gainesville publish one point per curb ramp (fetched for #79 by
scripts/inventory_oracle.py), which answers exactly that. This scores the server's PS rule
and fusion against them under the rule pre-registered on #106 before any score existed
(docs/ps-clustering-eval.md, "City-inventory scoring"; the comment is linked there):

  - labels synthesized from the run as eval_ps_clustering.py --offline does (one per stored
    detection >= tier, rig-masked, at the server's own positions via ps_placement);
  - a VISIBLE POOL of inventory ramps within POOL_RADIUS_M of a processed pano;
  - every cluster of every arm placed at the mean of its members' labeler raycast positions
    in the frame under test (`auto` primary, 2.6 m secondary);
  - covered / split / merge at r = 3, 5, 8 m, read at r = 5 m, tier 0.30, auto frame.

Arms: `ps @ t` (per region where the city has a server, else citywide), `ps_citywide @ 7.5 m`,
`fusion`, `fusion_server+attach` (cluster count only), and `ps @ 7.5 m (server centroid)`,
the sensitivity row that places the PS clusters at the mean of their members' server
positions instead.

Vancouver (issue #56) is scored under the same metrics but from the SERVER's own labels
(SERVER_LABELS): the frozen label pull and the deployed clusters give `deployed`, `ps @ t`
on the server's positions, `fusion_server` and `fusion_server+attach`, each placed at the
mean of its labels' server positions (the server's frame; no run needed) with a raycast
sensitivity row per fuse frame. Those are the confirmatory rows of the #56 pre-registration;
the synthesized arms above, run on the rebuilt store run, are exploratory there and say so
in the `scope` column. The verdict command still reads Bend and Gainesville only.

Network: one GET of the Gainesville server's /v3/api/streets (cached under
runs/gainesville/inventory_clustering/streets.geojson), for regions; Vancouver's streets
are the pull eval_ps_clustering.py made (copied beside the report). Nothing is written to
any server; every server is read-only. Needs pandas/scipy/haversine/shapely, as
eval_ps_clustering.py does (analysis-only, not in requirements.txt).

Usage:
    python scripts/inventory_clustering.py score bend gainesville
    python scripts/inventory_clustering.py score vancouver      # #56, descriptive
    python scripts/inventory_clustering.py verdict
"""
import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_ps_clustering as epc  # noqa: E402  (exits with an install hint if deps missing)
import inventory_oracle as ioracle  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OUT_NAME = 'inventory_clustering'
POOLED = REPO_ROOT / 'runs' / '_pooled' / OUT_NAME
SERVERS = {'gainesville': 'https://sidewalk-gainesville.cs.washington.edu',
           'vancouver': 'https://sidewalk-vancouver.cs.washington.edu'}
CITIES = ('bend', 'gainesville', 'vancouver')
# #56: a city whose AI labels are live and whose run was rebuilt from the pano store. The
# confirmatory arms come from the server's labels (paths under runs/<city>/); the label
# pull is the provenance gate's frozen one, the clusters the eval_ps_clustering pull.
SERVER_LABELS = {
    'vancouver': {'labels': 'provenance_gate/raw_labels.geojson',
                  'clusters': 'ps_clustering_eval/clusters.geojson',
                  'ai_user': '51b0b927-3c8a-45b2-93de-bd878d1e5cf4',
                  'tier': 0.55},   # the tier the deployed labels went live at
}
FRAMES_BY_CITY = {'vancouver': ('auto', 2.6, geo.PER_PANO)}   # per-pano: store-bound depth
DECIDING_CITY = 'gainesville'   # not a RampNet training city
GUARD_CITY = 'bend'             # a training city, with the larger inventory

POOL_RADIUS_M = 20.0
RADII_M = (3.0, 5.0, 8.0)
PRIMARY_RADIUS_M = 5.0
TIERS = (0.30, 0.55)
PRIMARY_TIER = 0.30
FRAMES = ('auto', 2.6)
PRIMARY_FRAME = 'auto'
THRESHOLDS_M = (7.5, 10.0, 12.5, 15.0)
PS_ARM = 'ps @ 7.5 m'
FUSION_ARM = 'fusion'
# the pre-registered decision rule (absolute shares)
RULE_SPLIT_DROP = 0.05       # fusion's split lower by at least this
RULE_MERGE_RISE = 0.01       # fusion's merge not higher by more than this
RULE_COVERED_DROP = 0.01     # fusion's covered not lower by more than this

PASS = 'PASS'
NOT_ESTABLISHED = 'NOT ESTABLISHED'
FRAME_DEPENDENT = 'NOT ESTABLISHED (frame-dependent)'


# ------------------------------------------------------------------------- metrics

def visible_pool(inv_xy, pano_xy, radius_m=POOL_RADIUS_M):
    """Boolean mask over inventory ramps: within radius_m of at least one pano position.

    Example:
        >>> visible_pool(np.array([[0.0, 5.0], [0.0, 40.0]]), np.array([[0.0, 0.0]])).tolist()
        [True, False]
    """
    inv_xy = np.asarray(inv_xy, dtype=float).reshape(-1, 2)
    pano_xy = np.asarray(pano_xy, dtype=float).reshape(-1, 2)
    if not len(inv_xy) or not len(pano_xy):
        return np.zeros(len(inv_xy), dtype=bool)
    d, _ = cKDTree(pano_xy).query(inv_xy, k=1, distance_upper_bound=radius_m)
    return np.isfinite(d)


def nearest_within(tree, xy, r):
    """Index of each point's nearest inventory ramp within r, -1 when there is none."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, dtype=int)
    d, idx = tree.query(xy, k=1, distance_upper_bound=r)
    return np.where(np.isfinite(d), idx, -1)


def inventory_metrics(clusters, inv_xy, pool, r):
    """The pre-registered metrics for one partition at match radius r.

    clusters: [(centroid or None, [member (e, n), ...])] -- the centroid is the mean of the
    placeable members' positions (None: no placeable member, counted but not placed).
    inv_xy: (N, 2) inventory positions in the same frame; pool: (N,) bool visible mask.

    covered = pool ramps with >= 1 assigned cluster / pool; split = covered pool ramps with
    >= 2 assigned clusters / covered; extra = (clusters on covered pool ramps - covered) /
    covered; merge = clusters with >= 2 assigned members whose members' ramps include >= 2
    distinct ramps each holding >= 2 of them, over clusters with >= 2 assigned members.
    Assignment is to the nearest ramp of the WHOLE inventory within r (pool or not).
    """
    inv_xy = np.asarray(inv_xy, dtype=float).reshape(-1, 2)
    pool = np.asarray(pool, dtype=bool)
    tree = cKDTree(inv_xy)
    placed = [c for c, _m in clusters if c is not None]
    assign = nearest_within(tree, placed, r)
    counts = np.bincount(assign[assign >= 0], minlength=len(inv_xy))
    n_pool = int(pool.sum())
    covered_mask = pool & (counts >= 1)
    covered = int(covered_mask.sum())
    split = int((pool & (counts >= 2)).sum())
    on_covered = int(counts[covered_mask].sum())
    merge_n = merge_k = 0
    for _c, members in clusters:
        if len(members) < 2:
            continue
        m = nearest_within(tree, members, r)
        m = m[m >= 0]
        if len(m) < 2:
            continue
        merge_n += 1
        _ids, per = np.unique(m, return_counts=True)
        if int((per >= 2).sum()) >= 2:
            merge_k += 1
    return {'n_clusters': len(clusters), 'n_placed': len(placed), 'pool': n_pool,
            'covered': covered, 'split': split, 'on_covered': on_covered,
            'merge_k': merge_k, 'merge_n': merge_n,
            'covered_rate': covered / n_pool if n_pool else None,
            'split_rate': split / covered if covered else None,
            'extra_per_covered': (on_covered - covered) / covered if covered else None,
            'merge_rate': merge_k / merge_n if merge_n else None,
            'clusters_per_covered': len(clusters) / covered if covered else None}


# ------------------------------------------------ cluster-review GT (RampNet#224)

NOT_RAMP, UNSURE = 'not_ramp', 'unsure'
REVIEW_ARM, REVIEW_BASE = 'fusion_server+attach', PS_ARM   # the pre-registered comparison
NO_GT = 'NO GT YET'


def _is_ramp(v):
    return isinstance(v, str) and len(v) > 1 and v[0] == 'r' and v[1:].isdigit()


def _wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def arm_index(clusters):
    """{label key (str): set of cluster indices} from [[label_id, ...], ...]. A label in
    two clusters (a stale deployed pull) maps to both."""
    out = {}
    for k, lids in enumerate(clusters):
        for lab in lids:
            out.setdefault(str(lab), set()).add(k)
    return out


def assignment_metrics(arm_of, units, unit_keys=None):
    """The pre-registered cluster-review metrics (RampNet docs/cluster_review_protocol.md,
    "Metrics") for one arm over reviewed units. Pure, numpy-free.

    arm_of: {label key: set of cluster ids} (arm_index); units: {corner_id: assignment
    unit} -- only `complete` ones count; unit_keys: {corner_id: the unit's label keys}
    (default: the keys the unit's assignment names). Within a unit only its own labels are
    read; unsure labels are dropped from everything; not_ramp labels count for validity.

    per GT ramp: clusters(r) = distinct clusters holding its labels; covered >= 1, split >= 2.
    per cluster touching the unit: its in-window ramp-assigned labels span >= 2 ramps ->
    merge (strict: >= 2 ramps with >= 2 labels each), over clusters with >= 2 of them.
    validity: clusters whose in-window labels are all not_ramp / clusters touching; not_ramp
    labels / non-unsure labels the arm holds. coverage: covered / (ramps + sure uncovered).
    Returns counts, rates (Wilson CIs) and `per_ramp` {(corner_id, ramp): n_clusters}."""
    m = dict(units=0, ramps=0, covered=0, split=0, merge_k=0, merge_strict_k=0, merge_n=0,
             clusters=0, invalid_clusters=0, labels_held=0, not_ramp_held=0,
             uncovered_sure=0)
    per_ramp = {}
    for cid, u in sorted(units.items()):
        if not u.get('complete'):
            continue
        m['units'] += 1
        a = u.get('labels') or {}
        keys = [k for k in (unit_keys or {}).get(cid, a) if a.get(k) != UNSURE and k in a]
        ramps = sorted({a[k] for k in keys if _is_ramp(a[k])})
        for r in ramps:
            cl = set()
            for k in keys:
                if a[k] == r:
                    cl |= arm_of.get(k, set())
            per_ramp[(cid, r)] = len(cl)
            m['ramps'] += 1
            m['covered'] += len(cl) >= 1
            m['split'] += len(cl) >= 2
        m['uncovered_sure'] += sum(1 for p in u.get('uncovered') or [] if not p.get('unsure'))
        touching = {}
        for k in keys:
            if arm_of.get(k):
                m['labels_held'] += 1
                m['not_ramp_held'] += a[k] == NOT_RAMP
            for c in arm_of.get(k, ()):
                touching.setdefault(c, []).append(a[k])
        for c, vals in touching.items():
            m['clusters'] += 1
            m['invalid_clusters'] += all(v == NOT_RAMP for v in vals)
            rv = [v for v in vals if _is_ramp(v)]
            if len(rv) >= 2:
                m['merge_n'] += 1
                per = {}
                for v in rv:
                    per[v] = per.get(v, 0) + 1
                m['merge_k'] += len(per) >= 2
                m['merge_strict_k'] += sum(1 for n in per.values() if n >= 2) >= 2
    pop = m['ramps'] + m['uncovered_sure']
    m.update(split_rate=m['split'] / m['covered'] if m['covered'] else None,
             split_ci=_wilson(m['split'], m['covered']),
             merge_rate=m['merge_k'] / m['merge_n'] if m['merge_n'] else None,
             merge_ci=_wilson(m['merge_k'], m['merge_n']),
             merge_strict_rate=m['merge_strict_k'] / m['merge_n'] if m['merge_n'] else None,
             invalid_cluster_rate=m['invalid_clusters'] / m['clusters'] if m['clusters'] else None,
             not_ramp_label_rate=m['not_ramp_held'] / m['labels_held'] if m['labels_held'] else None,
             coverage=m['covered'] / pop if pop else None,
             coverage_ci=_wilson(m['covered'], pop), per_ramp=per_ramp)
    return m


def paired_ramp_table(per_ramp_a, per_ramp_b):
    """Arms A and B on the same GT ramps: fixed = A splits (>= 2 clusters) and B holds it
    in one; broken = the reverse; both / neither over ramps both cover; plus the ramps
    only one arm covers. (A = the baseline, e.g. ps @ 7.5 m, as the #56 table.)"""
    t = dict(fixed=0, broken=0, both=0, neither=0, only_a=0, only_b=0, neither_covers=0)
    for key in set(per_ramp_a) | set(per_ramp_b):
        a, b = per_ramp_a.get(key, 0), per_ramp_b.get(key, 0)
        if a and b:
            t['fixed'] += a >= 2 and b == 1
            t['broken'] += a == 1 and b >= 2
            t['both'] += a >= 2 and b >= 2
            t['neither'] += a == 1 and b == 1
        elif a:
            t['only_a'] += 1
        elif b:
            t['only_b'] += 1
        else:
            t['neither_covers'] += 1
    return t


def assignment_verdict(by_arm, arm=REVIEW_ARM, base=REVIEW_BASE):
    """The pre-registered decision rule on assignment metrics: `arm` vs `base` on the same
    units -- split lower by >= RULE_SPLIT_DROP, merge not higher by > RULE_MERGE_RISE,
    coverage not lower by > RULE_COVERED_DROP -> PASS, else NOT ESTABLISHED; NO GT YET when
    either rate is undefined (nothing reviewed)."""
    a, b = by_arm.get(arm), by_arm.get(base)
    need = ('split_rate', 'merge_rate', 'coverage')
    if a is None or b is None or any(a[k] is None or b[k] is None for k in need):
        return NO_GT, ['no complete reviewed unit gives every rate for both arms']
    d_split = a['split_rate'] - b['split_rate']
    d_merge = a['merge_rate'] - b['merge_rate']
    d_cov = a['coverage'] - b['coverage']
    ok = (d_split <= -RULE_SPLIT_DROP + 1e-12 and d_merge <= RULE_MERGE_RISE + 1e-12
          and d_cov >= -RULE_COVERED_DROP - 1e-12)
    text = (f'{base} -> {arm}: split {b["split_rate"]:.3f} -> {a["split_rate"]:.3f} '
            f'({d_split:+.3f}), merge {b["merge_rate"]:.3f} -> {a["merge_rate"]:.3f} '
            f'({d_merge:+.3f}), coverage {b["coverage"]:.3f} -> {a["coverage"]:.3f} '
            f'({d_cov:+.3f})')
    return (PASS if ok else NOT_ESTABLISHED), [text]


def verdict(rows):
    """Apply the pre-registered decision rule to arms.csv rows (dicts with city, tier,
    frame, arm, r, covered_rate, split_rate, merge_rate). Returns (verdict, reasons).

    PASS iff, at r = 5 m and tier 0.30, in BOTH cities in the auto frame: fusion's split
    is lower than ps @ 7.5 m's by >= 5 points, its merge is not higher by > 1 point and its
    covered is not lower by > 1 point -- and in the 2.6 m frame no city reverses any of the
    three (split not lower at all, merge higher by > 1 point, covered lower by > 1 point).
    A reversal at 2.6 m makes it NOT ESTABLISHED (frame-dependent); anything else is NOT
    ESTABLISHED.
    """
    def get(city, frame, arm):
        for r in rows:
            if (r['city'] == city and str(r['frame']) == str(frame) and r['arm'] == arm
                    and float(r['tier']) == PRIMARY_TIER
                    and float(r['r']) == PRIMARY_RADIUS_M):
                return r
        return None

    def f(v):
        return None if v in (None, '') else float(v)

    reasons, primary_ok, reversed_ = [], True, False
    for city in (DECIDING_CITY, GUARD_CITY):
        for frame in FRAMES:
            ps, fu = get(city, frame, PS_ARM), get(city, frame, FUSION_ARM)
            if ps is None or fu is None:
                reasons.append(f'{city} {frame}: missing rows')
                primary_ok = False
                continue
            d_split = f(fu['split_rate']) - f(ps['split_rate'])
            d_merge = f(fu['merge_rate']) - f(ps['merge_rate'])
            d_cov = f(fu['covered_rate']) - f(ps['covered_rate'])
            text = (f'{city} {frame}: split {f(ps["split_rate"]):.3f} -> '
                    f'{f(fu["split_rate"]):.3f} ({d_split:+.3f}), merge '
                    f'{f(ps["merge_rate"]):.3f} -> {f(fu["merge_rate"]):.3f} ({d_merge:+.3f}), '
                    f'covered {f(ps["covered_rate"]):.3f} -> {f(fu["covered_rate"]):.3f} '
                    f'({d_cov:+.3f})')
            if str(frame) == str(PRIMARY_FRAME):
                ok = (d_split <= -RULE_SPLIT_DROP and d_merge <= RULE_MERGE_RISE
                      and d_cov >= -RULE_COVERED_DROP)
                primary_ok &= ok
                reasons.append(text + (' -- meets the rule' if ok else ' -- does NOT meet it'))
            else:
                rev = d_split >= 0 or d_merge > RULE_MERGE_RISE or d_cov < -RULE_COVERED_DROP
                reversed_ |= rev
                reasons.append(text + (' -- REVERSES' if rev else ' -- no reversal'))
    if primary_ok and not reversed_:
        return PASS, reasons
    if primary_ok and reversed_:
        return FRAME_DEPENDENT, reasons
    return NOT_ESTABLISHED, reasons


# ---------------------------------------------------------------------------- scoring

def streets_for(city, out):
    if city not in SERVERS:
        return None
    path = out / 'streets.geojson'
    epc.fetch(SERVERS[city] + epc.API_STREETS, path)
    return path


def placed_clusters(clusters, det_pos):
    """[(centroid or None, member positions)] from eval_ps_clustering Clusters."""
    out = []
    for c in clusters:
        pts = [det_pos[m] for m in c.members if m in det_pos]
        cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) \
            if pts else None
        out.append((cen, pts))
    return out


def server_placed(clusters, server_pos, fr):
    """[(centroid or None, member positions)] with every position a label's SERVER
    lat/lng (projected into fr): the arm as the server holds it, no run involved."""
    out = []
    for c in clusters:
        pts = [fr.to_enu(*server_pos[lab]) for lab in c.label_ids if lab in server_pos]
        cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) \
            if pts else None
        out.append((cen, pts))
    return out


def score_server_arms(city, cfg, inventory, results_path, frames, radii, rows, lines):
    """The #56 confirmatory arms, from the server's own labels: `deployed` (the clusters as
    served), `ps @ t` and `ps_citywide @ 7.5 m` (the PS rule re-run on the server's
    positions, AI labels, per the labels' own region), and per fuse frame `fusion_server`
    and `fusion_server+attach`. Every cluster is placed at the mean of its labels' server
    positions (`placement: server`); `deployed`, `ps @ 7.5 m` and `fusion_server` also get a
    `placement: raycast` sensitivity row per frame (the Part 2 common frame, over the
    members the rebuilt run maps). The visible pool is the run's panos, as for every arm."""
    run_dir = REPO_ROOT / 'runs' / city
    labels_path, clusters_path = run_dir / cfg['labels'], run_dir / cfg['clusters']
    labels, _bad = epc.load_labels(labels_path)
    labels = labels[labels.label_type == 'CurbRamp'].reset_index(drop=True)
    server_clusters = epc.load_server_clusters(clusters_path)
    det_of, _amb, _dup = epc.label_to_detection(results_path, labels)
    ai = labels[labels.user_id.astype(str) == cfg['ai_user']].reset_index(drop=True)
    server_pos = {int(r.label_id): (r.lat, r.lng) for r in labels.itertuples(index=False)}
    lines += ['', '## Server-label arms (#56 confirmatory)', '',
              epc.provenance(labels_path, len(labels)),
              epc.provenance(clusters_path, len(server_clusters)),
              f"- {len(ai)} labels of the AI account, {sum(1 for lab in ai.label_id if lab in det_of)} "
              'of them map pixel-exactly to a stored detection of the rebuilt run (the rest '
              'are placed only at their server position); '
              f'{len(labels) - len(ai)} human labels are in `deployed` and `fusion_server` '
              'but not in `ps @ t`, as in eval_ps_clustering.py']
    t_kms = [t / 1000.0 for t in THRESHOLDS_M]
    parts = epc.ps_partition(ai, t_kms, per_region=True)
    city_part = epc.ps_partition(ai, [epc.PS_THRESHOLD_KM], per_region=False)[epc.PS_THRESHOLD_KM]
    fixed = {'deployed': epc.clusters_from_server(server_clusters, det_of)}
    for t_m, t_km in zip(THRESHOLDS_M, t_kms):
        fixed[f'ps @ {t_m:g} m'] = epc.clusters_from_assignment(ai, parts[t_km], det_of)
    fixed['ps_citywide @ 7.5 m'] = epc.clusters_from_assignment(ai, city_part, det_of)
    tier = cfg['tier']
    for frame in frames:
        try:
            panos, _skip, height, auto = fs.load_at_height(results_path, frame)
        except ValueError as e:
            raise SystemExit(str(e))
        params = fs.FuseParams(camera_height_m=height, min_confidence=tier,
                               mask_rig=False, apply_pose=fs.POSE_OFF)
        dets, fr, _drops = fs.project(panos, params)
        det_pos = {(d.pano_id, d.det_index): (d.e, d.n) for d in dets}
        inv_xy = np.array([fr.to_enu(lat, lng) for _k, lat, lng, _p in inventory])
        pool = visible_pool(inv_xy, np.array([fr.to_enu(p.lat, p.lng) for p in panos]))
        run_by_id = {p.pano_id: p for p in panos}
        srv_panos, st = epc.server_panos(labels, det_of, run_by_id, ai_user=cfg['ai_user'],
                                         unmapped_confidence=tier)
        srv_sites, srv_frame, _s3 = fs.fuse(srv_panos, replace(params, min_confidence=0.0,
                                                               floor=0.0))
        srv_cl, _n1 = epc.clusters_from_server_sites(srv_sites, srv_panos, st['label_of'])
        att, attached = epc.attach_unplaceable(srv_sites, srv_panos, srv_frame,
                                               label_of=st['label_of'])
        arms = dict(fixed)
        arms['fusion_server'] = srv_cl
        arms['fusion_server+attach'] = att
        table = []
        for name, cl in arms.items():
            n_labels = sum(c.n_labels for c in cl)
            variants = [('server', server_placed(cl, server_pos, fr))]
            if name in ('deployed', PS_ARM, 'fusion_server'):
                variants.append(('raycast', placed_clusters(epc.place(cl, det_pos), det_pos)))
            for placement, placed in variants:
                if placement == 'server' and name in fixed and frame != frames[0]:
                    continue      # frame-free: written once, under the first frame
                by_r = {}
                for r in radii:
                    m = inventory_metrics(placed, inv_xy, pool, r)
                    by_r[r] = {'city': city, 'tier': tier,
                               'frame': '-' if (placement == 'server' and name in fixed)
                               else str(frame), 'arm': name, 'r': r, 'n_labels': n_labels,
                               'placement': placement, 'scope': 'confirmatory', **m}
                    rows.append(by_r[r])
                table.append((by_r[PRIMARY_RADIUS_M], by_r))
        lines += ['', f'### {fs.frame_label(frame)} frame (fusion_server association; '
                  'server-placed rows of the fixed arms are written once)', '',
                  f'- visible pool: {int(pool.sum())} of {len(inventory)} inventory ramps '
                  f'within {POOL_RADIUS_M:g} m of one of {len(panos)} panos',
                  f"- fusion_server input: {st['ai_labels']} mapped AI + {st['ai_unmapped']} "
                  f"unmapped AI (at {tier:g}) + {st['human_labels']} human labels on "
                  f"{len(srv_panos)} panos ({st['inverted']} positioned by inversion); "
                  f'{len(attached)} labels attached by bearing']
        lines += fs.height_resolution_lines(frame, panos, params, auto)
        lines += ['', '| arm | placement | clusters | labels | placed | covered r3 / r5 / r8 '
                  '| split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |',
                  '|---|---|---:|---:|---:|---|---|---:|---|---:|']
        for p, by_r in table:
            lines.append(
                f"| {p['arm']} | {p['placement']} | {p['n_clusters']} | {p['n_labels']} | "
                f"{p['n_placed']} | "
                + ' / '.join(f"{by_r[r]['covered_rate']:.3f}" for r in radii) + ' | '
                f"{p['split_rate']:.3f} ({p['split']}/{p['covered']}) | "
                f"{p['extra_per_covered']:.3f} | {p['merge_rate']:.3f} "
                f"({p['merge_k']}/{p['merge_n']}) | {p['clusters_per_covered']:.2f} |")


def server_arms(run_dir, cfg, frame=PRIMARY_FRAME, thresholds_m=THRESHOLDS_M):
    """(labels, arms, stats): every #56 server-label arm as {name: [[label_id, ...], ...]}.

    The same construction as score_server_arms (RampNet#224 reuses it verbatim, so the
    cluster-review GT scores exactly the partitions #56 scored): `deployed` as served,
    `ps @ t` per region on the AI labels' server positions, `fusion_server` and
    `fusion_server+attach` from a fuse of the server's labels in `frame`. It is duplicated
    here rather than factored out of score_server_arms so that function, and the committed
    report it writes, stay byte-for-byte what they were. `run_dir` is a run directory
    holding cfg['labels'], cfg['clusters'] and results.jsonl (read in place; nothing is
    written). `labels` is the CurbRamp DataFrame the arms were built from."""
    run_dir = Path(run_dir)
    results_path = run_dir / 'results.jsonl'
    labels, _bad = epc.load_labels(run_dir / cfg['labels'])
    labels = labels[labels.label_type == 'CurbRamp'].reset_index(drop=True)
    server_clusters = epc.load_server_clusters(run_dir / cfg['clusters'])
    det_of, _amb, _dup = epc.label_to_detection(results_path, labels)
    ai = labels[labels.user_id.astype(str) == cfg['ai_user']].reset_index(drop=True)
    arms = {'deployed': [list(sc['label_ids']) for sc in server_clusters]}
    t_kms = [t / 1000.0 for t in thresholds_m]
    parts = epc.ps_partition(ai, t_kms, per_region=True)
    for t_m, t_km in zip(thresholds_m, t_kms):
        arms[f'ps @ {t_m:g} m'] = [c.label_ids for c in
                                   epc.clusters_from_assignment(ai, parts[t_km], det_of)]
    try:
        panos, _skip, height, _auto = fs.load_at_height(results_path, frame)
    except ValueError as e:
        raise SystemExit(str(e))
    params = fs.FuseParams(camera_height_m=height, min_confidence=cfg['tier'],
                           mask_rig=False, apply_pose=fs.POSE_OFF)
    srv_panos, st = epc.server_panos(labels, det_of, {p.pano_id: p for p in panos},
                                     ai_user=cfg['ai_user'], unmapped_confidence=cfg['tier'])
    srv_sites, srv_frame, _s3 = fs.fuse(srv_panos, replace(params, min_confidence=0.0,
                                                           floor=0.0))
    srv_cl, _n1 = epc.clusters_from_server_sites(srv_sites, srv_panos, st['label_of'])
    att, attached = epc.attach_unplaceable(srv_sites, srv_panos, srv_frame,
                                           label_of=st['label_of'])
    arms['fusion_server'] = [c.label_ids for c in srv_cl]
    arms['fusion_server+attach'] = [c.label_ids for c in att]
    stats = {'n_labels': len(labels), 'n_ai': len(ai), 'n_mapped': len(det_of),
             'n_run_panos': len(panos), 'n_attached': len(attached),
             'run_pos': {p.pano_id: (p.lat, p.lng, p.camera_heading) for p in panos},
             'server_panos': {k: v for k, v in st.items() if k != 'label_of'},
             'frame': str(frame), 'height': str(height)}
    return labels, arms, stats


def score_city(city, tiers=TIERS, frames=None, radii=RADII_M):
    frames = tuple(frames) if frames else FRAMES_BY_CITY.get(city, FRAMES)
    run_dir = REPO_ROOT / 'runs' / city
    out = run_dir / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / 'results.jsonl'
    inventory, record = ioracle.load_inventory(city)
    streets_path = streets_for(city, out)
    streets = epc.load_streets(streets_path) if streets_path else []
    server_cfg = SERVER_LABELS.get(city)
    lines = [f'# {city}: label clustering vs the city curb-ramp inventory (#106 Part 2)', '',
             f"- inventory: {len(inventory)} kept ramps ({record.get('keep', {}).get('where_equivalent', '')}) "
             f"of {record.get('layer_total')}, `{record.get('file')}` sha256 "
             f"`{record.get('sha256')}`, fetched {record.get('fetched_utc')} from "
             f"{record.get('url')}",
             f'- results `{results_path.name}` sha256 `{fs.file_sha256(results_path)}`',
             ('- regions: nearest street of the server\'s street network (the server\'s '
              'insert rule); ' + epc.streets_provenance(streets_path, streets)[2:])
             if streets else '- regions: none (no server), so per-region == citywide']
    rows = []
    if server_cfg:
        score_server_arms(city, server_cfg, inventory, results_path, frames, radii, rows,
                          lines)
        lines += ['', '## Synthesized arms from the rebuilt run (#56: EXPLORATORY here, '
                  'never the deployed labels\' fusion)']
    for tier in tiers:
        labels, det_of = epc.synthesize_labels(results_path, tier, mask_rig=True)
        if streets:
            labels['region_id'] = epc.assign_regions(labels.lat, labels.lng, streets)[0]
        t_kms = [t / 1000.0 for t in THRESHOLDS_M]
        parts = epc.ps_partition(labels, t_kms, per_region=True)
        city_part = epc.ps_partition(labels, [epc.PS_THRESHOLD_KM],
                                     per_region=False)[epc.PS_THRESHOLD_KM]
        ps_clusters = {f'ps @ {t_m:g} m': epc.clusters_from_assignment(labels, parts[t_km], det_of)
                       for t_m, t_km in zip(THRESHOLDS_M, t_kms)}
        ps_clusters['ps_citywide @ 7.5 m'] = epc.clusters_from_assignment(labels, city_part,
                                                                          det_of)
        server_ll = {det_of[r.label_id]: (r.lat, r.lng) for r in labels.itertuples(index=False)}
        for frame in frames:
            try:
                panos, _skip, height, auto = fs.load_at_height(results_path, frame)
            except ValueError as e:
                raise SystemExit(str(e))
            params = fs.FuseParams(camera_height_m=height, min_confidence=tier,
                                   mask_rig=True, apply_pose=fs.POSE_OFF)
            dets, fr, _drops = fs.project(panos, params)
            det_pos = {(d.pano_id, d.det_index): (d.e, d.n) for d in dets}
            inv_xy = np.array([fr.to_enu(lat, lng) for _k, lat, lng, _p in inventory])
            pano_xy = np.array([fr.to_enu(p.lat, p.lng) for p in panos])
            pool = visible_pool(inv_xy, pano_xy)
            arms = {name: placed_clusters(cl, det_pos) for name, cl in ps_clusters.items()}
            # sensitivity: the PS clusters at their members' SERVER positions
            srv = []
            for c in ps_clusters[PS_ARM]:
                pts = [fr.to_enu(*server_ll[m]) for m in c.members if m in server_ll]
                ray = [det_pos[m] for m in c.members if m in det_pos]
                srv.append(((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
                            if pts else None, ray))
            arms[f'{PS_ARM} (server centroid)'] = srv
            sites, _f2, _s2 = fs.fuse(panos, params)
            fusion_clusters = epc.clusters_from_sites(sites, False)
            arms[FUSION_ARM] = placed_clusters(fusion_clusters, det_pos)
            run_by_id = {p.pano_id: p for p in panos}
            srv_panos, _st = epc.server_panos(labels, det_of, run_by_id)
            srv_sites, srv_frame, _s3 = fs.fuse(srv_panos, replace(params, min_confidence=0.0,
                                                                   floor=0.0))
            att, attached = epc.attach_unplaceable(srv_sites, srv_panos, srv_frame)
            arms['fusion_server+attach'] = placed_clusters(epc.place(att, det_pos), det_pos)
            n_labels = {name: sum(len(c.members) for c in cl) for name, cl in ps_clusters.items()}
            n_labels[f'{PS_ARM} (server centroid)'] = n_labels[PS_ARM]
            n_labels[FUSION_ARM] = sum(len(c.members) for c in fusion_clusters)
            n_labels['fusion_server+attach'] = sum(c.n_labels for c in att)
            for name, cl in arms.items():
                for r in radii:
                    m = inventory_metrics(cl, inv_xy, pool, r)
                    rows.append({'city': city, 'tier': tier, 'frame': str(frame), 'arm': name,
                                 'r': r, 'n_labels': n_labels[name],
                                 **({'placement': 'raycast' if name != f'{PS_ARM} (server centroid)'
                                     else 'server', 'scope': 'exploratory'} if server_cfg
                                    else {}), **m})
            lines += ['', f'## tier {tier:g}, {fs.frame_label(frame)} frame', '',
                      f'- visible pool: {int(pool.sum())} of {len(inventory)} inventory ramps '
                      f'within {POOL_RADIUS_M:g} m of one of {len(panos)} panos '
                      f'({len(inventory) - int(pool.sum())} excluded)',
                      f'- {len(labels)} labels synthesized; raycast placed {len(dets)} '
                      'detections at or above the storage floor']
            lines += fs.height_resolution_lines(frame, panos, params, auto)
            lines.append(f'- fusion_server+attach: {len(attached)} labels attached by bearing')
            lines += ['', '| arm | clusters | labels | placed | covered r3 / r5 / r8 | '
                      'split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |',
                      '|---|---:|---:|---:|---|---|---:|---|---:|']
            for name in arms:
                by_r = {row['r']: row for row in rows if row['tier'] == tier
                        and row['frame'] == str(frame) and row['arm'] == name
                        and row.get('scope', 'exploratory') == 'exploratory'}
                p = by_r[PRIMARY_RADIUS_M]
                lines.append(
                    f"| {name} | {p['n_clusters']} | {p['n_labels']} | {p['n_placed']} | "
                    + ' / '.join(f"{by_r[r]['covered_rate']:.3f}" for r in radii) + ' | '
                    f"{p['split_rate']:.3f} ({p['split']}/{p['covered']}) | "
                    f"{p['extra_per_covered']:.3f} | {p['merge_rate']:.3f} "
                    f"({p['merge_k']}/{p['merge_n']}) | {p['clusters_per_covered']:.2f} |")
    lines += ['', 'covered = pool ramps with a cluster assigned (nearest ramp within r) / '
              'pool; split = covered ramps with >= 2 clusters / covered; merge = clusters '
              'whose members fall on >= 2 distinct ramps with >= 2 members each / clusters '
              'with >= 2 assigned members; clusters/covered = all clusters / covered pool '
              'ramps. Every cluster is placed at the mean of its members\' raycast positions '
              'in the frame named, except the `(server centroid)` row. Rule and read: '
              'docs/ps-clustering-eval.md, "City-inventory scoring".']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with open(out / 'arms.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(lines))
    print(f'wrote {out / "report.md"} and arms.csv')
    return rows


def read_rows(city):
    path = REPO_ROOT / 'runs' / city / OUT_NAME / 'arms.csv'
    if not path.exists():
        raise SystemExit(f'{path} is missing: run `inventory_clustering.py score {city}` first')
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def cmd_verdict():
    rows = read_rows(DECIDING_CITY) + read_rows(GUARD_CITY)
    v, reasons = verdict(rows)
    POOLED.mkdir(parents=True, exist_ok=True)
    lines = ['# City-inventory clustering score: verdict (#106 Part 2)', '',
             'Pre-registered rule (docs/ps-clustering-eval.md, "City-inventory scoring"): '
             '`fusion` vs `ps @ 7.5 m` at r = 5 m, tier 0.30, common `auto` frame, in both '
             'cities; the 2.6 m frame must not reverse it.', '', f'**{v}**', '']
    lines += [f'- {r}' for r in reasons]
    lines += ['', '| city | tier | frame | arm | covered r5 | split r5 | merge r5 | '
              'clusters |', '|---|---|---|---|---|---|---|---:|']
    for r in rows:
        if float(r['r']) == PRIMARY_RADIUS_M:
            lines.append(f"| {r['city']} | {float(r['tier']):g} | {r['frame']} | {r['arm']} | "
                         f"{float(r['covered_rate']):.3f} | {float(r['split_rate']):.3f} | "
                         f"{float(r['merge_rate']):.3f} | {r['n_clusters']} |")
    (POOLED / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (POOLED / 'verdict.json').write_text(json.dumps(
        {'verdict': v, 'reasons': reasons, 'rule': {
            'radius_m': PRIMARY_RADIUS_M, 'tier': PRIMARY_TIER, 'frame': PRIMARY_FRAME,
            'split_drop': RULE_SPLIT_DROP, 'merge_rise': RULE_MERGE_RISE,
            'covered_drop': RULE_COVERED_DROP}}, indent=1) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('cities', nargs='*', default=list(CITIES))
    s.add_argument('--tier', type=float, nargs='+', default=list(TIERS))
    s.add_argument('--radius', type=float, nargs='+', default=list(RADII_M))
    s.add_argument('--frame', type=fs.fuse_camera_height_arg, nargs='+', default=None,
                   help=f'fuse frames (default {FRAMES}, or FRAMES_BY_CITY for a city named '
                        'there)')
    sub.add_parser('verdict')
    args = ap.parse_args(argv)
    if args.cmd == 'score':
        if PRIMARY_RADIUS_M not in args.radius:
            raise SystemExit(f'--radius must include the primary {PRIMARY_RADIUS_M:g} m')
        for city in args.cities:
            if city not in CITIES:
                raise SystemExit(f'{city}: no inventory; choose from {CITIES}')
            score_city(city, args.tier, args.frame, args.radius)
    else:
        cmd_verdict()


if __name__ == '__main__':
    main()
