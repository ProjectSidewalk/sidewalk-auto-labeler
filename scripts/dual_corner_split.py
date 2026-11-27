"""Dual-corner split: split fused sites that likely hold two curb ramps, then emit one
best-view label per sub-site, scored at SERVER placement against city inventories.

Builds on scripts/best_view_emission.py (BVE): its `best` arm (one label per fused site,
the member with the largest dip) placed labels much better than a random member but lost
inventory coverage, mostly where fusion holds the two ramps of a dual corner in one site.
This adds a GT-free split rule and asks whether it recovers that coverage. The rule reads
only fusion's output; inventories are used for scoring and (on Bend only) for tuning.

The split rule family (docs/overnight/dual_corner_split.md, section 2):
  For an operational site with >= 2 operational members, the candidate partition is the
  two-part split of its members' raycast ground points that minimizes the summed GLS
  chi-square of the two parts, searched over cuts perpendicular to N_DIRECTIONS directions
  (bipartition()). Part K holds the site's best-view member; part O is the other. The
  site is split iff
    dchi2 = chi2(site) - chi2(K) - chi2(O) >= SPLIT_DCHI2       (two positions fit better),
    |server label of K - server label of O| >= SPLIT_LABEL_SEP_M (the two labels would land
                                                                  apart, not on one ramp),
    server range of O's label <= SPLIT_MAX_RANGE_M              (the new label is a near,
                                                                  well-placed view).
  Each part then emits best_view_emission.best_key over its own members.
  FROZEN (the pre-registered Bend objective's pick): all three thresholds off, i.e. every
  site with >= 2 operational members is split at its minimum-chi2 cut. EXPLORATORY `knee`
  setting (amendment A1): dchi2 >= 2, label separation >= 2 m, new label's range <= 14 m.

Arms (pre-registered, section 1 of the doc):
  all                 every label (today)
  best                BVE's one label per operational site
  best+split          one best-view label per sub-site after the rule
  best+random_split   control: the same number of sites split, chosen at random among sites
                      with >= 2 operational members, members divided at random
                      (random.Random(192); seeds 0-19 as sensitivity)
  split+randmember    reference for the placement clause: the rule's sub-sites, one random
                      member each

Usage:
    python scripts/dual_corner_split.py tune bend --runs-root runs \\
        --inventory bend=<scratch>/inv/bend/inventory.geojson         # Bend only
    python scripts/dual_corner_split.py score bend gainesville --runs-root runs \\
        --inventory bend=<scratch>/inv/bend/inventory.geojson \\
        --inventory gainesville=<scratch>/inv/gainesville/inventory.geojson
    python scripts/dual_corner_split.py verdict

No network, no GPU. numpy, scipy and shapely (as BVE).
"""
import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):  # root ends up first (scripts/position_check.py is a shim)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import best_view_emission as bve  # noqa: E402
import fuse_sites as fs  # noqa: E402
from best_view_emission import ic  # noqa: E402  (inventory_clustering, pandas-free import)
from agree_rate import match_one_to_one  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OUT_DIR = REPO_ROOT / 'docs' / 'overnight' / 'data' / 'dual_corner_split'
CITIES = ('bend', 'gainesville')
TUNING_CITY, CONFIRM_CITY = 'bend', 'gainesville'
SEED = bve.SEED                       # 192
SENSITIVITY_SEEDS = tuple(range(20))
N_BOOT = bve.N_BOOT
MATCH_R_M = bve.MATCH_R_M             # 5 m
OVERSPLIT_R_M = 8.0
CLUSTER_R_M = 10.0                    # coverage bootstrap: ramp -> nearest site within this
N_DIRECTIONS = 36

# FROZEN 2026-11-27 from the Bend tuning grid (doc section 2), before Gainesville was loaded.
# The pre-registered objective selected the cell with no thresholds: every site with >= 2
# operational members is split at its minimum-chi2 cut.
SPLIT_DCHI2 = 0.0
SPLIT_LABEL_SEP_M = 0.0
SPLIT_MAX_RANGE_M = math.inf
# EXPLORATORY (amendment A1, 2026-11-27, before Gainesville was loaded): the fewest-label
# Bend cell within 1.0 pt of `all` (p90 gain >= 1.0 m). Reported, never gates.
KNEE_DCHI2 = 2.0
KNEE_LABEL_SEP_M = 2.0
KNEE_MAX_RANGE_M = 14.0
SETTINGS = {'rule': (SPLIT_DCHI2, SPLIT_LABEL_SEP_M, SPLIT_MAX_RANGE_M),
            'knee': (KNEE_DCHI2, KNEE_LABEL_SEP_M, KNEE_MAX_RANGE_M)}

# The tuning grid (pre-declared family; doc section 2).
GRID_DCHI2 = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0)
GRID_SEP_M = (0.0, 1.5, 2.0, 2.5, 3.0, 4.0)
GRID_RANGE_M = (8.0, 10.0, bve.TAIL_START_M, 14.0, math.inf)
TUNE_P90_MIN_M = 1.0
TUNE_COVER_MARGIN = 0.005

RULE_COVERED_DROP = 0.010
RULE_P90_MARGIN_M = 0.5

ALL, BEST, SPLIT, RSPLIT, SPLIT_RM = ('all', 'best', 'best+split', 'best+random_split',
                                      'split+randmember')
ARMS = (ALL, BEST, SPLIT, RSPLIT, SPLIT_RM)
PASS, FAIL, UNDETERMINED = 'PASS', 'FAIL', 'UNDETERMINED'


# ------------------------------------------------------------------------ the partition

def _info(cov):
    a, b, c = cov
    det = a * c - b * b
    return (c / det, -b / det, a / det)


def gls(points):
    """(e, n, chi2) of the information-form fit of [(e, n, cov)], fuse_sites.Site's algebra.

    Example:
        >>> e, n, c = gls([(0.0, 0.0, (1.0, 0.0, 1.0)), (2.0, 0.0, (1.0, 0.0, 1.0))])
        >>> round(e, 6), round(n, 6), round(c, 6)
        (1.0, 0.0, 2.0)
    """
    lam = np.zeros(3)
    ee = en = s = 0.0
    for e, n, cov in points:
        w = _info(cov)
        lam += w
        ee += w[0] * e + w[1] * n
        en += w[1] * e + w[2] * n
        s += w[0] * e * e + 2 * w[1] * e * n + w[2] * n * n
    ci = _info(tuple(float(v) for v in lam))
    pe = ci[0] * ee + ci[1] * en
    pn = ci[1] * ee + ci[2] * en
    return float(pe), float(pn), float(max(0.0, s - (ee * pe + en * pn)))


def _chi2_prefix(w, e, n):
    """chi2 of every prefix of the (already ordered) points, vectorized. w: (k, 3)."""
    lam = np.cumsum(w, axis=0)
    eta_e = np.cumsum(w[:, 0] * e + w[:, 1] * n)
    eta_n = np.cumsum(w[:, 1] * e + w[:, 2] * n)
    s = np.cumsum(w[:, 0] * e * e + 2 * w[:, 1] * e * n + w[:, 2] * n * n)
    a, b, c = lam[:, 0], lam[:, 1], lam[:, 2]
    det = a * c - b * b
    pe = (c * eta_e - b * eta_n) / det
    pn = (-b * eta_e + a * eta_n) / det
    return np.maximum(0.0, s - (eta_e * pe + eta_n * pn))


def bipartition(points, n_dirs=N_DIRECTIONS):
    """The two-part split of points [(e, n, cov)] (len >= 2) minimizing chi2(A) + chi2(B)
    over cuts perpendicular to n_dirs directions in [0, 180) deg. Returns
    (dchi2, part_a, part_b) with parts as sorted index tuples; part_a holds index 0's side
    canonicalized to contain the smallest index. Deterministic: ties go to the smaller
    direction index, then the smaller cut.

    Example:
        >>> I = (0.25, 0.0, 0.25)
        >>> pts = [(0, 0, I), (0.2, 0, I), (6, 0, I), (6.2, 0, I)]
        >>> d, a, b = bipartition(pts)
        >>> a, b, round(d, 2)
        ((0, 1), (2, 3), 144.0)
    """
    k = len(points)
    xy = np.array([(p[0], p[1]) for p in points], dtype=float)
    w = np.array([_info(p[2]) for p in points], dtype=float)
    chi0 = float(_chi2_prefix(w, xy[:, 0], xy[:, 1])[-1])
    best = None
    for j in range(n_dirs):
        th = math.pi * j / n_dirs
        order = np.argsort(xy @ np.array([math.cos(th), math.sin(th)]), kind='stable')
        fw = _chi2_prefix(w[order], xy[order, 0], xy[order, 1])
        rev = order[::-1]
        bw = _chi2_prefix(w[rev], xy[rev, 0], xy[rev, 1])[::-1]
        tot = fw[:-1] + bw[1:]                 # cut after position c: [0..c] | [c+1..]
        c = int(np.argmin(tot))
        if best is None or tot[c] < best[0] - 1e-9:
            best = (float(tot[c]), order[:c + 1], order[c + 1:])
    tot, a, b = best
    a, b = tuple(sorted(int(i) for i in a)), tuple(sorted(int(i) for i in b))
    if b[0] < a[0]:
        a, b = b, a
    return chi0 - tot, a, b


# ---------------------------------------------------------------------------- the rule

class SitePlan:
    """One operational site's split candidate, with every quantity the rule reads."""
    __slots__ = ('site', 'members', 'cands', 'best', 'dchi2', 'keep', 'other', 'k_keep',
                 'k_other', 'label_sep', 'other_range', 'pos_keep', 'pos_other')

    def __init__(self, site, labels):
        self.site = site
        self.members = sorted((d for d, _r in site.members if d.operational),
                              key=lambda d: (d.pano_id, d.det_index))
        self.cands = [(d.pano_id, d.det_index, d.y, d.conf) for d in self.members]
        self.best = bve.best_key(self.cands)
        self.dchi2 = None
        if len(self.members) < 2:
            return
        pts = [(d.e, d.n, d.cov) for d in self.members]
        self.dchi2, a, b = bipartition(pts)
        bi = next(i for i, c in enumerate(self.cands) if (c[0], c[1]) == self.best)
        self.keep, self.other = (a, b) if bi in a else (b, a)
        self.k_keep = bve.best_key([self.cands[i] for i in self.keep])
        self.k_other = bve.best_key([self.cands[i] for i in self.other])
        lk, lo = labels[self.k_keep], labels[self.k_other]
        self.label_sep = math.hypot(lk.e - lo.e, lk.n - lo.n)
        self.other_range = lo.range_m
        self.pos_keep = gls([pts[i] for i in self.keep])[:2]
        self.pos_other = gls([pts[i] for i in self.other])[:2]


def plan_sites(sites, labels):
    """[SitePlan] over operational sites in id order. labels must carry ENU (e, n)."""
    return [SitePlan(s, labels) for s in bve.operational_sites(sites)]


def rule_splits(plan, dchi2_min, sep_min, range_max):
    """The site ids the rule splits.

    Example:
        >>> from types import SimpleNamespace as N
        >>> p = [N(site=N(id=3), dchi2=5.0, label_sep=3.0, other_range=9.0),
        ...      N(site=N(id=4), dchi2=None, label_sep=None, other_range=None)]
        >>> sorted(rule_splits(p, 4.0, 2.5, 11.77)), sorted(rule_splits(p, 6.0, 2.5, 11.77))
        ([3], [])
    """
    return {p.site.id for p in plan if p.dchi2 is not None and p.dchi2 >= dchi2_min
            and p.label_sep >= sep_min and p.other_range <= range_max}


def random_split(plan, n_split, seed=SEED):
    """{site id: (part_a, part_b)} for the control: n_split sites drawn uniformly among
    sites with >= 2 operational members (sorted by id), each divided by independent fair
    coin flips per member, redrawn until both parts are non-empty.

    Example:
        >>> from types import SimpleNamespace as N
        >>> p = [N(site=N(id=i), members=[0] * 3) for i in range(5)]
        >>> out = random_split(p, 2, seed=1)
        >>> len(out), all(len(a) and len(b) and len(a) + len(b) == 3 for a, b in out.values())
        (2, True)
    """
    rng = random.Random(seed)
    elig = [p for p in plan if len(p.members) >= 2]
    if n_split > len(elig):
        raise ValueError(f'cannot split {n_split} of {len(elig)} eligible sites')
    chosen = sorted(rng.sample(range(len(elig)), n_split))
    out = {}
    for j in chosen:
        p = elig[j]
        while True:
            flips = [rng.random() < 0.5 for _ in p.members]
            if 0 < sum(flips) < len(flips):
                break
        out[p.site.id] = (tuple(i for i, f in enumerate(flips) if f),
                          tuple(i for i, f in enumerate(flips) if not f))
    return out


def emit(plan, parts_by_site):
    """[(sub-site id, key)]: one best-view key per part of a split site, the site's best
    key otherwise. Sub-site id = 2 * site id + part (0 for an unsplit site)."""
    out = []
    for p in plan:
        parts = parts_by_site.get(p.site.id)
        if parts is None:
            out.append((2 * p.site.id, p.best))
            continue
        for j, part in enumerate(parts):
            out.append((2 * p.site.id + j, bve.best_key([p.cands[i] for i in part])))
    return out


def rule_parts(plan, split_ids):
    return {p.site.id: (p.keep, p.other) for p in plan if p.site.id in split_ids}


# ------------------------------------------------------------------------------ metrics

def covered_vector(inv_xy, pool, pts, r=MATCH_R_M):
    """Per-ramp covered indicator (BVE's M3): each label to its nearest kept ramp within r;
    a pool ramp is covered when >= 1 label is assigned to it."""
    tree = cKDTree(inv_xy)
    asg = ic.nearest_within(tree, pts, r)
    counts = np.bincount(asg[asg >= 0], minlength=len(inv_xy))
    return np.asarray(pool, dtype=bool) & (counts >= 1)


def ramp_clusters(inv_xy, pool, site_xy, r=CLUSTER_R_M):
    """Cluster id per POOL ramp for the bootstrap: its nearest operational site within r,
    else a singleton of its own. Returns an int array aligned with np.flatnonzero(pool).

    Example:
        >>> ramp_clusters(np.array([[0, 0], [1, 0], [50, 0]]), np.array([1, 1, 1], bool),
        ...               np.array([[0.5, 0.0]])).tolist()
        [0, 0, 1]
    """
    idx = np.flatnonzero(pool)
    d, j = cKDTree(site_xy).query(np.asarray(inv_xy)[idx], k=1)
    out = np.where(d <= r, j, -1)
    nxt = len(site_xy)
    for i in np.flatnonzero(out < 0):
        out[i] = nxt
        nxt += 1
    _u, inv = np.unique(out, return_inverse=True)
    return inv


def cluster_bootstrap_diff(x, y, clusters, n_boot=N_BOOT, seed=SEED):
    """(point, lo, hi): mean(x) - mean(y) over pool ramps, 95% percentile CI resampling
    clusters with replacement (ratio estimator: sum of differences / ramps drawn).

    Example:
        >>> x = np.array([1, 1, 0, 0]); y = np.array([0, 0, 0, 0])
        >>> p, lo, hi = cluster_bootstrap_diff(x, y, np.array([0, 0, 1, 1]), 200, 1)
        >>> p, lo <= p <= hi
        (0.5, True)
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    k = int(clusters.max()) + 1
    sd = np.bincount(clusters, weights=d, minlength=k)
    sn = np.bincount(clusters, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        c = np.bincount(rng.integers(0, k, k), minlength=k)
        reps[b] = (c @ sd) / (c @ sn)
    return float(d.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


class _P:
    __slots__ = ('id', 'e', 'n')

    def __init__(self, id_, e, n):
        self.id, self.e, self.n = id_, e, n


def subsite_positions(plan, split_ids):
    """{sub-site id: (e, n)}: an unsplit site keeps its fused position; a split site's
    parts get the information-form refit of their members."""
    out = {}
    for p in plan:
        if p.site.id in split_ids:
            out[2 * p.site.id] = p.pos_keep
            out[2 * p.site.id + 1] = p.pos_other
        else:
            out[2 * p.site.id] = (p.site.e, p.site.n)
    return out


def randmember(plan, parts, seed=SEED):
    """{sub-site id: key}: one uniformly random member per sub-site (random.Random(seed),
    sub-sites in (site id, part) order) -- the split+randmember reference."""
    rng = random.Random(seed)
    out = {}
    for p in plan:
        for j, part in enumerate(parts.get(p.site.id) or [tuple(range(len(p.cands)))]):
            c = p.cands[part[rng.randrange(len(part))]]
            out[2 * p.site.id + j] = (c[0], c[1])
    return out


def split_m1(inv_xy, pool, plan, split_ids, labels, seed=SEED):
    """Clause (ii): pool ramps matched one-to-one to the rule's sub-site positions within
    5 m; per pair the distance to best+split's label and to split+randmember's label.
    Returns (best distances, random-member distances, random-member keys)."""
    parts = rule_parts(plan, split_ids)
    pos = subsite_positions(plan, split_ids)
    idx = np.flatnonzero(pool)
    ramps = [_P(int(i), float(inv_xy[i, 0]), float(inv_xy[i, 1])) for i in idx]
    subs = [_P(sid, e, n) for sid, (e, n) in sorted(pos.items())]
    m = match_one_to_one(ramps, subs, MATCH_R_M)
    best = dict(emit(plan, parts))
    rand = randmember(plan, parts, seed)
    db, dr = [], []
    for gi, s in sorted(m.items(), key=lambda kv: kv[1].id):
        r = ramps[gi]
        for out, keyset in ((db, best), (dr, rand)):
            lab = labels[keyset[s.id]]
            out.append(math.hypot(lab.e - r.e, lab.n - r.n))
    return db, dr, rand


def oversplit(inv_xy, plan, split_ids, r=OVERSPLIT_R_M):
    """{0, 1, '2+'} counts of kept inventory ramps within r of each split site's ORIGINAL
    fused position."""
    tree = cKDTree(inv_xy)
    out = {'0': 0, '1': 0, '2+': 0}
    for p in plan:
        if p.site.id in split_ids:
            k = len(tree.query_ball_point([p.site.e, p.site.n], r))
            out['0' if k == 0 else '1' if k == 1 else '2+'] += 1
    return out


def ps_cluster_metrics(pts, pano_ids, inv_xy, pool):
    """BVE's approximate server clustering (complete linkage 7.5 m, citywide)."""
    cid = bve.ps_clusters(pts, pano_ids)
    groups = {}
    for j, c in enumerate(cid):
        groups.setdefault(int(c), []).append(tuple(pts[j]))
    clusters = [((sum(p[0] for p in g) / len(g), sum(p[1] for p in g) / len(g)), g)
                for _c, g in sorted(groups.items())]
    cm = ic.inventory_metrics(clusters, inv_xy, pool, MATCH_R_M)
    lk, ln = bve.loose_merge(clusters, inv_xy)
    return {'ps_clusters': cm['n_clusters'], 'ps_covered_r5': cm['covered_rate'],
            'ps_split_r5': cm['split_rate'], 'ps_merge_loose_r5': lk / ln if ln else None}


# -------------------------------------------------------------------------------- frames

def load_frame(city, tier, frame_req, results_path, inv_ll, labels):
    """Fuse exactly as BVE's score_frame does; returns the pieces every scorer reads."""
    panos, _skip, height, _auto = fs.load_at_height(results_path, frame_req)
    params = fs.FuseParams(camera_height_m=height, min_confidence=tier, mask_rig=True,
                           apply_pose=fs.POSE_OFF)
    sites, frame, stats = fs.fuse(panos, params)
    for lab in labels.values():
        lab.e, lab.n = frame.to_enu(lab.lat, lab.lng)
    inv_xy = np.array([frame.to_enu(lat, lng) for lat, lng in inv_ll])
    pool = ic.visible_pool(inv_xy, np.array([frame.to_enu(p.lat, p.lng) for p in panos]))
    owner = bve.site_of(sites)
    missing = [k for k in owner if k not in labels]
    if missing:
        raise ValueError(f'{len(missing)} site members have no label (e.g. {missing[0]})')
    plan = plan_sites(sites, labels)
    unplaced = sorted(k for k in labels if k not in owner)
    return {'sites': sites, 'plan': plan, 'inv_xy': inv_xy, 'pool': pool, 'labels': labels,
            'unplaced': unplaced, 'height': height, 'panos': len(panos), 'stats': stats}


def pts_of(labels, keys):
    return np.array([(labels[k].e, labels[k].n) for k in keys], dtype=float).reshape(-1, 2)


def tune(fr, base):
    """Every grid cell on one frame: labels, splits, covered_r5, clause-(ii) p90 gain."""
    labels, inv_xy, pool, plan = fr['labels'], fr['inv_xy'], fr['pool'], fr['plan']
    cov_all = covered_vector(inv_xy, pool, pts_of(labels, list(labels)))
    cov_best = covered_vector(inv_xy, pool, pts_of(labels, [p.best for p in plan]))
    rows = []
    for t, d, r in itertools.product(GRID_DCHI2, GRID_SEP_M, GRID_RANGE_M):
        ids = rule_splits(plan, t, d, r)
        keys = [k for _s, k in emit(plan, rule_parts(plan, ids))]
        cov = covered_vector(inv_xy, pool, pts_of(labels, keys))
        db, dr, _rk = split_m1(inv_xy, pool, plan, ids, labels)
        rows.append({**base, 'dchi2_min': t, 'label_sep_min_m': d,
                     'other_range_max_m': r if math.isfinite(r) else 'inf',
                     'n_split': len(ids), 'n_labels': len(keys),
                     'covered_r5': cov.sum() / pool.sum(),
                     'covered_r5_all': cov_all.sum() / pool.sum(),
                     'covered_r5_best': cov_best.sum() / pool.sum(),
                     'p90_best_split': bve.pct(db, 0.9), 'p90_split_randmember': bve.pct(dr, 0.9),
                     'p90_gain': bve.pct(dr, 0.9) - bve.pct(db, 0.9)})
    return rows


def select(rows):
    """The pre-registered tuning objective (doc section 1): among cells with p90_gain >=
    TUNE_P90_MIN_M, the fewest labels with covered_r5 >= covered_r5_all - 0.005; if none,
    the highest covered_r5 (ties: fewer labels). Ties on labels: fewer splits, then the
    grid order (stricter thresholds first is NOT assumed: the first row in grid order).

    Example:
        >>> r = [dict(p90_gain=1.2, covered_r5=0.90, covered_r5_all=0.91, n_labels=10, n_split=1),
        ...      dict(p90_gain=1.2, covered_r5=0.906, covered_r5_all=0.91, n_labels=20, n_split=5),
        ...      dict(p90_gain=0.8, covered_r5=0.909, covered_r5_all=0.91, n_labels=12, n_split=2)]
        >>> select(r)['n_labels']
        20
    """
    ok = [r for r in rows if float(r['p90_gain']) >= TUNE_P90_MIN_M]
    hit = [r for r in ok if float(r['covered_r5'])
           >= float(r['covered_r5_all']) - TUNE_COVER_MARGIN - 1e-12]
    if hit:
        return min(hit, key=lambda r: (int(r['n_labels']), int(r['n_split'])))
    return max(ok, key=lambda r: (float(r['covered_r5']), -int(r['n_labels'])))


def score_frame(fr, base, setting='rule'):
    """Arm rows, control-seed rows, the bootstrap row, over-splitting and survivorship
    for one split setting ('rule' = the frozen rule; 'knee' = exploratory amendment A1)."""
    labels, inv_xy, pool, plan = fr['labels'], fr['inv_xy'], fr['pool'], fr['plan']
    t, d, r = SETTINGS[setting]
    base = {**base, 'setting': setting}
    ids = rule_splits(plan, t, d, r)
    rparts = random_split(plan, len(ids), SEED)
    rule_keys = emit(plan, rule_parts(plan, ids))
    keys = {ALL: sorted(labels), BEST: [p.best for p in plan],
            SPLIT: [k for _s, k in rule_keys],
            RSPLIT: [k for _s, k in emit(plan, rparts)]}
    db, dr, rk = split_m1(inv_xy, pool, plan, ids, labels)
    keys[SPLIT_RM] = [k for _s, k in sorted(rk.items())]
    covs = {a: {rr: covered_vector(inv_xy, pool, pts_of(labels, k), rr)
                for rr in bve.RADII_M} for a, k in keys.items()}
    npool = int(pool.sum())
    pidx = np.flatnonzero(pool)
    site_xy = np.array([(p.site.e, p.site.n) for p in plan])
    clus = ramp_clusters(inv_xy, pool, site_xy)
    # BVE's own M1 (best vs random member on the ORIGINAL sites), for reference
    pairs = bve.anchored_pairs(inv_xy, pool, fr['sites'])
    sel_b = {s: [k] for s, k in bve.select_best(fr['sites']).items()}
    sel_r = {s: [k] for s, k in bve.select_random(fr['sites'], SEED).items()}
    m1b = bve.anchored_distances(pairs, inv_xy, sel_b, labels)[0]
    m1r = bve.anchored_distances(pairs, inv_xy, sel_r, labels)[0]
    rows = []
    for arm, k in keys.items():
        if setting != 'rule' and arm in (ALL, BEST):
            continue                      # identical to the rule's rows
        pts = pts_of(labels, k)
        m5 = ic.inventory_metrics([((e, n), [(e, n)]) for e, n in pts], inv_xy, pool,
                                  MATCH_R_M)
        assert m5['covered'] == int(covs[arm][MATCH_R_M].sum())
        rows.append({**base, 'arm': arm, 'n_labels': len(k), 'pool': npool,
                     **{f'covered_r{rr:g}': covs[arm][rr].sum() / npool for rr in bve.RADII_M},
                     'labels_per_covered_r5': m5['on_covered'] / m5['covered']
                     if m5['covered'] else None,
                     'multi_label_share_r5': m5['split_rate'],
                     'unassigned_r5': bve.unassigned(inv_xy, pts),
                     **ps_cluster_metrics(pts, [kk[0] for kk in k], inv_xy, pool)})
    boot = {**base, 'n_split': len(ids), 'n_subsite_pairs': len(db), 'n_boot': N_BOOT,
            'seed': SEED, 'p90_best_split': bve.pct(db, 0.9),
            'p90_split_randmember': bve.pct(dr, 0.9),
            'p50_best_split': bve.pct(db, 0.5), 'p50_split_randmember': bve.pct(dr, 0.5)}
    boot['p90_gain'] = boot['p90_split_randmember'] - boot['p90_best_split']
    ci = bve.bootstrap_diff(db, dr)
    boot['p90_gain_lo'], boot['p90_gain_hi'] = ci[0.9]
    boot['bve_m1_pairs'] = len(m1b)
    boot['bve_p90_gain'] = bve.pct(m1r, 0.9) - bve.pct(m1b, 0.9)
    for other in (ALL, BEST, RSPLIT):
        p, lo, hi = cluster_bootstrap_diff(covs[SPLIT][MATCH_R_M][pidx],
                                           covs[other][MATCH_R_M][pidx], clus)
        tag = other.replace('+', '_')
        boot[f'cov_diff_vs_{tag}'], boot[f'cov_diff_vs_{tag}_lo'], \
            boot[f'cov_diff_vs_{tag}_hi'] = p, lo, hi
    boot['n_clusters'] = int(clus.max()) + 1
    seed_rows = []
    for s in SENSITIVITY_SEEDS:
        kk = [k for _s, k in emit(plan, random_split(plan, len(ids), s))]
        c = covered_vector(inv_xy, pool, pts_of(labels, kk))
        seed_rows.append({**base, 'seed': s, 'n_labels': len(kk),
                          'covered_r5': c.sum() / npool})
    multi_ids = {p.site.id for p in plan if len(p.members) >= 2}
    over = []
    for name, sids in (('rule', ids), ('random_split', set(rparts)),
                       ('all_multi_member_sites', multi_ids)):
        o = oversplit(inv_xy, plan, sids)
        n = sum(o.values())
        over.append({**base, 'set': name, 'sites': n, 'ramps8_0': o['0'], 'ramps8_1': o['1'],
                     'ramps8_2plus': o['2+'], 'share_one_ramp': o['1'] / n if n else None})
    surv = {**base, 'height': str(fr['height']), 'panos': fr['panos'],
            'labels': len(labels), 'labels_unplaced': len(fr['unplaced']),
            'operational_sites': len(plan), 'multi_member_sites': len(multi_ids),
            'split_sites': len(ids), 'subsites': len(rule_keys),
            'pool': npool, 'subsite_pairs': len(db), 'bve_m1_pairs': len(m1b)}
    return rows, seed_rows, boot, over, surv


# ------------------------------------------------------------------------------ verdict

def verdict(rows, boots, city=CONFIRM_CITY, tier=0.30, frame='auto', setting='rule'):
    """(PASS|FAIL|UNDETERMINED, reasons) for one city/tier/frame from the CSV rows."""
    def same(r):
        return (r['city'] == city and abs(float(r['tier']) - tier) < 1e-9
                and r['frame'] == frame)

    def get(arm):
        for r in rows:
            if same(r) and r['arm'] == arm and (arm in (ALL, BEST) or r['setting'] == setting):
                return r
        return None
    b = next((r for r in boots if same(r) and r['setting'] == setting), None)
    a, s = get(ALL), get(SPLIT)
    if a is None or s is None or b is None:
        return UNDETERMINED, [f'{city}: not scored']
    c1 = float(s['covered_r5']) >= float(a['covered_r5']) - RULE_COVERED_DROP - 1e-12
    c2 = float(b['p90_gain']) >= RULE_P90_MARGIN_M - 1e-12
    c3 = float(b['cov_diff_vs_best_random_split_lo']) > 0
    reasons = [
        f'(i) covered_r5 best+split {float(s["covered_r5"]):.4f} vs all '
        f'{float(a["covered_r5"]):.4f} ({float(s["covered_r5"]) - float(a["covered_r5"]):+.4f},'
        f' need >= -0.010): {"ok" if c1 else "FAILS"}',
        f'(ii) p90 split+randmember - best+split = {float(b["p90_gain"]):+.2f} m '
        f'[{float(b["p90_gain_lo"]):+.2f}, {float(b["p90_gain_hi"]):+.2f}] (need >= +0.50): '
        f'{"ok" if c2 else "FAILS"}',
        f'(iii) covered_r5 best+split - best+random_split = '
        f'{float(b["cov_diff_vs_best_random_split"]):+.4f} '
        f'[{float(b["cov_diff_vs_best_random_split_lo"]):+.4f}, '
        f'{float(b["cov_diff_vs_best_random_split_hi"]):+.4f}] (CI must exclude 0): '
        f'{"ok" if c3 else "FAILS"}']
    return (PASS if c1 and c2 and c3 else FAIL), reasons


# ---------------------------------------------------------------------------------- CLI

def _inventory(args, city):
    paths = dict(s.split('=', 1) for s in args.inventory)
    inv_ll, prov = bve.load_points(city, args.runs_root, paths.get(city))
    results = Path(args.runs_root) / city / 'results.jsonl'
    prov['results_sha256'] = fs.file_sha256(results)
    f = Path(prov['file']).resolve()
    if REPO_ROOT not in f.parents:          # never record a machine-specific absolute path
        prov['file'] = '<external>/' + '/'.join(f.parts[-3:])
    print(f'{city}: inventory {prov["status"]} ({prov["kept"]} kept), results sha256 '
          f'{prov["results_sha256"][:12]}', file=sys.stderr)
    return inv_ll, prov, results


def cmd_tune(args):
    if args.city != TUNING_CITY:
        raise SystemExit('tuning reads Bend only (pre-registration)')
    inv_ll, prov, results = _inventory(args, args.city)
    rows = []
    for tier in args.tier:
        labels = bve.synthesize(results, tier)
        for frame in args.frame:
            fr = load_frame(args.city, tier, frame, results, inv_ll, labels)
            rows += tune(fr, {'city': args.city, 'tier': tier,
                              'frame': fs.frame_label(frame)})
    out = Path(args.out)
    bve.write_csv(out / 'bend_tuning_grid.csv', rows)
    prim = [r for r in rows if r['tier'] == 0.30 and r['frame'] == 'auto']
    sel = select(prim)
    print('SELECTED (tier 0.30, auto):', json.dumps(sel, default=str), file=sys.stderr)
    (out / 'bend_tuning_inventory.json').write_text(json.dumps(prov, indent=1) + '\n',
                                                    encoding='utf-8')


def cmd_score(args):
    out = Path(args.out)
    for city in args.cities:
        inv_ll, prov, results = _inventory(args, city)
        prov['role'] = 'tuning' if city == TUNING_CITY else 'confirmatory'
        rows, seeds, boots, overs, survs = [], [], [], [], []
        for tier in args.tier:
            labels = bve.synthesize(results, tier)
            for frame in args.frame:
                fr = load_frame(city, tier, frame, results, inv_ll, labels)
                base = {'city': city, 'tier': tier, 'frame': fs.frame_label(frame)}
                for setting in SETTINGS:
                    r, s, b, o, sv = score_frame(fr, base, setting)
                    rows += r
                    seeds += s
                    boots.append(b)
                    overs += o
                    survs.append(sv)
                    print(f'  tier {tier:g} {base["frame"]} {setting}: split '
                          f'{b["n_split"]}, p90 gain {b["p90_gain"]:+.2f}, cov vs all '
                          f'{b["cov_diff_vs_all"]:+.4f}, vs random_split '
                          f'{b["cov_diff_vs_best_random_split"]:+.4f} '
                          f'[{b["cov_diff_vs_best_random_split_lo"]:+.4f}, '
                          f'{b["cov_diff_vs_best_random_split_hi"]:+.4f}]', file=sys.stderr)
        bve.write_csv(out / f'{city}_arms.csv', rows)
        bve.write_csv(out / f'{city}_control_seeds.csv', seeds)
        bve.write_csv(out / f'{city}_bootstrap.csv', boots)
        bve.write_csv(out / f'{city}_oversplit.csv', overs)
        bve.write_csv(out / f'{city}_survivorship.csv', survs)
        (out / f'{city}_inventory.json').write_text(json.dumps(prov, indent=1) + '\n',
                                                    encoding='utf-8')
    print(f'wrote {out}', file=sys.stderr)


def cmd_verdict(args):
    out = Path(args.out)
    rows, boots = [], []
    for city in CITIES:
        if (out / f'{city}_arms.csv').exists():
            rows += bve.read_csv(out / f'{city}_arms.csv')
            boots += bve.read_csv(out / f'{city}_bootstrap.csv')
    v, reasons = verdict(rows, boots)
    print(f'{CONFIRM_CITY} (confirmatory, tier 0.30, auto, frozen rule): {v}')
    for r in reasons:
        print('  -', r)
    for setting in SETTINGS:
        for city, tier, frame in itertools.product(CITIES, bve.TIERS, ('auto', '2.6 m')):
            if (setting, city, tier, frame) == ('rule', CONFIRM_CITY, 0.30, 'auto'):
                continue
            vv, rr = verdict(rows, boots, city, tier, frame, setting)
            print(f'{city} tier {tier:g} {frame} setting {setting} (reported, does not '
                  f'gate): {vv}')
            for r in rr:
                print('  -', r)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('tune', 'score'):
        s = sub.add_parser(name)
        if name == 'tune':
            s.add_argument('city', choices=CITIES)
        else:
            s.add_argument('cities', nargs='+', choices=CITIES)
        s.add_argument('--runs-root', default=str(REPO_ROOT / 'runs'))
        s.add_argument('--inventory', action='append', default=[], metavar='CITY=PATH')
        # tune defaults to the frame the objective reads (tier 0.30, auto)
        s.add_argument('--tier', type=float, nargs='+',
                       default=[bve.PRIMARY_TIER] if name == 'tune' else list(bve.TIERS))
        s.add_argument('--frame', type=fs.fuse_camera_height_arg, nargs='+',
                       default=[bve.PRIMARY_FRAME] if name == 'tune' else list(bve.FRAMES))
        s.add_argument('--out', default=str(OUT_DIR))
    v = sub.add_parser('verdict')
    v.add_argument('--out', default=str(OUT_DIR))
    args = ap.parse_args(argv)
    {'tune': cmd_tune, 'score': cmd_score, 'verdict': cmd_verdict}[args.cmd](args)


if __name__ == '__main__':
    main()
