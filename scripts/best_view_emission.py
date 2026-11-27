"""Best-view emission: one label per fused site, scored at SERVER placement against city
curb-ramp inventories.

Today every stored detection >= the tier ships as its own Project Sidewalk label. The server
places each one once from its pixel (ps_placement: 2.341 m camera, range capped near
23.85 m, no pose; issue #192), then clusters them. This asks whether emitting ONE label per
fused site, the member that sees the ramp at the largest dip, places labels closer to the
city inventory without losing coverage. No arm moves a label: an arm only chooses which of
today's labels to emit, and every label is scored where the server would put it.

Arms (pre-registered in docs/overnight/best_view_emission.md, section 1):
  all       every label (today's behaviour)
  best      one label per operational fused site: its operational member with the largest
            y_normalized (dip); ties -> higher confidence -> (pano_id, det_index)
  random    control: one uniformly random operational member per site (random.Random(seed),
            sites in id order; seed 192 for the rule, 0-19 reported as a sensitivity check)
  gate20    every label, except one whose server range is > 20 m while its site has an
            operational member at <= 20 m
  best+u / random+u   sensitivity: best / random plus the labels fusion could not place
  gate_tail EXPLORATORY (amendment 1): every label, except one in the server's linear range
            tail (server range > ~11.77 m, depression < 11.25 deg) while its site has a
            member outside the tail

Metrics: M1, the anchored one-to-one distance (pool ramps matched one-to-one to fused site
positions within 5 m; for each pair, the distance from the ramp to the label the arm emits
for that site, uncapped; paired bootstrap over sites); M2, own one-to-one at 5 m; M3,
coverage (inventory_clustering's nearest-ramp assignment at 3, 5 and 8 m); the workload
(labels emitted, labels per covered ramp); and an approximate re-run of the server's
complete-linkage clustering on each arm's labels.

Inputs are read, never written: <runs-root>/<city>/results.jsonl (+ depth/index.csv for the
`auto` frame) and the inventory snapshot. By default that is
<runs-root>/<city>/inventory_oracle/inventory.geojson, verified against the sha256 in
inventory.json. `--inventory city=PATH` takes another file: a GeoJSON FeatureCollection or a
RampNet `data/inventories/*.jsonl.gz` (one row of properties per ramp, with `lat`/`lon`).
Either is filtered to the run's area and to inventory_oracle.kept(), and is labelled a
SUBSTITUTE unless its sha256 matches the record. No network.

Needs numpy, scipy and shapely (no pandas: inventory_clustering's metric functions are
imported with a stub standing in for eval_ps_clustering, which they do not call).

Usage:
    python scripts/best_view_emission.py score bend gainesville \\
        --runs-root runs
    python scripts/best_view_emission.py score bend --runs-root <runs> \\
        --inventory bend=../RampNet/data/inventories/bend-or-2026-07-31.jsonl.gz
    python scripts/best_view_emission.py verdict      # reads the CSVs score wrote
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Root inserted last so it ends up first: scripts/position_check.py is a shim that
# would otherwise shadow the root position_check module.
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402,F401
import ps_placement  # noqa: E402
import fuse_sites as fs  # noqa: E402
import inventory_oracle as ioracle  # noqa: E402
from agree_rate import match_one_to_one  # noqa: E402
from detectors import on_camera_rig, ps_pixel  # noqa: E402
from detectors import registry  # noqa: E402

try:
    import inventory_clustering as ic  # noqa: E402
except SystemExit:
    # eval_ps_clustering exits when pandas/haversine are missing. inventory_clustering only
    # calls it inside score functions this script never uses; its metric functions
    # (visible_pool, inventory_metrics) need numpy and scipy alone.
    sys.modules['eval_ps_clustering'] = types.ModuleType('eval_ps_clustering')
    try:
        import inventory_clustering as ic  # noqa: E402
    finally:
        # Keep `ic` but leave nothing behind: a later importer must get the real modules
        # (or the real error), not an empty stub or an inventory_clustering bound to it.
        sys.modules.pop('eval_ps_clustering', None)
        sys.modules.pop('inventory_clustering', None)

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

OUT_DIR = REPO_ROOT / 'docs' / 'overnight' / 'data' / 'best_view_emission'
CITIES = ('bend', 'gainesville')
TIERS = (0.30, 0.55)
PRIMARY_TIER = 0.30
FRAMES = (fs.HEIGHT_AUTO, 2.6)
PRIMARY_FRAME = fs.HEIGHT_AUTO
MATCH_R_M = 5.0
RADII_M = (3.0, 5.0, 8.0)
GATE_M = 20.0
SEED = 192
SENSITIVITY_SEEDS = tuple(range(20))
N_BOOT = 2000
PS_THRESHOLD_M = 7.5          # label_clustering.py, CurbRamp
BLOCK_MARGIN_M = 0.5
RULE_P90_MARGIN_M = 0.5
RULE_COVERED_DROP = 0.010

ALL, BEST, RANDOM, GATE = 'all', 'best', 'random', 'gate20'
BEST_U, RANDOM_U = 'best+u', 'random+u'
# EXPLORATORY (amendment 1, 2026-11-26, added after the Bend smoke run showed gate20 drops
# nothing): drop a label in the server's linear range tail (depression < BLEND_DEG, i.e.
# server range > TAIL_START_M) when its site has a member outside the tail.
GATE_TAIL = 'gate_tail'
TAIL_START_M = ps_placement.estimate_distance_m(ps_placement.BLEND_DEG)   # ~11.77 m
ARMS = (ALL, BEST, RANDOM, GATE, BEST_U, RANDOM_U, GATE_TAIL)
CHANCE_SHIFT_M = 25.0
ONE_PER_SITE = (BEST, RANDOM)

PASS, FAIL, UNDETERMINED = 'PASS', 'FAIL', 'UNDETERMINED'


# ------------------------------------------------------------------------------ labels

@dataclass
class Label:
    """One synthesized label at the server's position."""
    key: tuple            # (pano_id, det_index): the stored detection it ships
    lat: float
    lng: float
    range_m: float        # the server's estimated distance (ps_placement)
    y: float              # y_normalized: the dip (0.5 horizon, 1.0 nadir)
    conf: float
    e: float = 0.0
    n: float = 0.0
    id: int = 0           # serial within one arm; match_one_to_one reads .id


def synthesize(results_path, tier, label_type=registry.CURB_RAMP):
    """{(pano_id, det_index): Label}: one label per stored detection >= tier, rig-masked,
    at detectors.ps_pixel, placed by ps_placement. The same rule as
    eval_ps_clustering.synthesize_labels (which needs pandas), without the DataFrame.

    Example:
        A record at (47.0, -122.0), heading 0, 4096 x 2048, one detection at x 0.5,
        y 0.6, confidence 0.7 -> one label due north at ~7.20 m (pano_y 1229 is 18.02 deg
        down; 2.341 / tan 18.02).
    """
    mask = registry.spec_for(label_type).mask_rig
    out = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            if p.get('lat') is None or p.get('lng') is None \
                    or p.get('camera_heading') is None:
                continue
            w, h = p['width'], p['height']
            for i, d in enumerate(rec.get('detections', [])):
                if registry.detection_type(rec, d) != label_type:
                    continue
                if d['confidence'] < tier:
                    continue
                if mask and on_camera_rig(d['y_normalized']):
                    continue
                px, py = ps_pixel(d['x_normalized'], d['y_normalized'], w, h)
                lat, lng = ps_placement.label_latlng(p['lat'], p['lng'], px, py, w, h,
                                                     p['camera_heading'])
                rng, _brg = ps_placement.label_offset(px, py, w, h, p['camera_heading'])
                out[(p['panorama_id'], i)] = Label((p['panorama_id'], i), lat, lng, rng,
                                                   d['y_normalized'], d['confidence'])
    return out


# -------------------------------------------------------------------------------- arms

def candidates(site):
    """The site's operational members as (pano_id, det_index, y, conf), sorted by key."""
    return sorted((d.pano_id, d.det_index, d.y, d.conf)
                  for d, _in_refit in site.members if d.operational)


def operational_sites(sites):
    return sorted((s for s in sites if s.n_operational > 0), key=lambda s: s.id)


def best_key(cands):
    """The member with the largest dip; ties -> higher confidence -> smaller key.

    Example:
        >>> best_key([('a', 0, 0.60, 0.9), ('b', 1, 0.70, 0.4), ('c', 2, 0.70, 0.5)])
        ('c', 2)
    """
    pid, idx, _y, _c = min(cands, key=lambda c: (-c[2], -c[3], c[0], c[1]))
    return (pid, idx)


def select_best(sites):
    """{site id: emitted key} for the `best` arm."""
    return {s.id: best_key(candidates(s)) for s in operational_sites(sites)}


def select_random(sites, seed=SEED):
    """{site id: emitted key} for the `random` control: one uniform draw per site,
    random.Random(seed), sites in id order, candidates in key order (so the draw depends
    only on the seed and the fused sites, which fuse_sites makes deterministically)."""
    rng = random.Random(seed)
    out = {}
    for s in operational_sites(sites):
        c = candidates(s)
        pid, idx, _y, _c = c[rng.randrange(len(c))]
        out[s.id] = (pid, idx)
    return out


def site_of(sites):
    """{label key: site id} over every operational member."""
    return {(c[0], c[1]): s.id for s in operational_sites(sites) for c in candidates(s)}


def gate_keep(labels, sites, gate_m=GATE_M):
    """Keys the `gate20` arm emits: every label, except one whose server range is
    > gate_m while its site holds an operational member at <= gate_m. Labels in no site
    are kept."""
    near_site = set()
    for s in operational_sites(sites):
        if any(labels[(c[0], c[1])].range_m <= gate_m for c in candidates(s)):
            near_site.add(s.id)
    owner = site_of(sites)
    return [k for k, lab in labels.items()
            if not (lab.range_m > gate_m and owner.get(k) in near_site)]


def build_arms(labels, sites, seed=SEED):
    """({arm: [keys]}, {arm: {site id: [keys]}}, unplaced keys)."""
    owner = site_of(sites)
    missing = [k for k in owner if k not in labels]
    if missing:
        raise ValueError(f'{len(missing)} operational site members have no label '
                         f'(e.g. {missing[0]}): fusion and synthesis disagree on the tier '
                         'or the rig mask')
    unplaced = sorted(k for k in labels if k not in owner)
    best, rand = select_best(sites), select_random(sites, seed)
    arms = {ALL: sorted(labels), BEST: sorted(best.values()), RANDOM: sorted(rand.values()),
            GATE: sorted(gate_keep(labels, sites)),
            GATE_TAIL: sorted(gate_keep(labels, sites, TAIL_START_M))}
    arms[BEST_U] = sorted(arms[BEST] + unplaced)
    arms[RANDOM_U] = sorted(arms[RANDOM] + unplaced)
    per_site = {}
    for arm, keys in arms.items():
        d = {}
        for k in keys:
            if k in owner:
                d.setdefault(owner[k], []).append(k)
        per_site[arm] = d
    return arms, per_site, unplaced


# ----------------------------------------------------------------------------- metrics

def pct(values, q):
    """Nearest-rank percentile, eval_sites._pct's convention (None on empty).

    Example:
        >>> pct([1.0, 2.0, 3.0, 4.0, 5.0], 0.9)
        5.0
    """
    if len(values) == 0:
        return None
    v = np.sort(np.asarray(values, dtype=float))
    return float(v[min(len(v) - 1, int(round((len(v) - 1) * q)))])


@dataclass
class _P:
    id: int
    e: float
    n: float


def anchored_pairs(inv_xy, pool, sites, r=MATCH_R_M):
    """[(inventory index, site id)]: pool ramps matched one-to-one (greedy ascending
    distance, agree_rate.match_one_to_one) to the operational sites' fused positions
    within r. The pairs depend on the fuse only, so every arm is scored on the same ones."""
    idx = np.flatnonzero(pool)
    ramps = [_P(int(i), float(inv_xy[i, 0]), float(inv_xy[i, 1])) for i in idx]
    ops = [_P(s.id, s.e, s.n) for s in operational_sites(sites)]
    m = match_one_to_one(ramps, ops, r)
    return sorted((ramps[gi].id, site.id) for gi, site in m.items())


def anchored_distances(pairs, inv_xy, per_site, labels):
    """(per-pair distances of the one emitted label, or None if the arm emits several or
    none for some pair; label-weighted distances over every emitted label)."""
    one, every, n_none = [], [], 0
    single = True
    for i, sid in pairs:
        keys = per_site.get(sid, [])
        if not keys:
            n_none += 1
            single = False
            continue
        ds = [math.hypot(labels[k].e - inv_xy[i, 0], labels[k].n - inv_xy[i, 1]) for k in keys]
        every += ds
        if len(ds) == 1:
            one.append(ds[0])
        else:
            single = False
    return (one if single else None), every, n_none


def own_one_to_one(inv_xy, pool, pts, r=MATCH_R_M):
    """Distances of a greedy one-to-one match between pool ramps and the arm's labels."""
    idx = np.flatnonzero(pool)
    ramps = [_P(int(i), float(inv_xy[i, 0]), float(inv_xy[i, 1])) for i in idx]
    labs = [_P(k, float(e), float(n)) for k, (e, n) in enumerate(pts)]
    m = match_one_to_one(ramps, labs, r)
    return [math.hypot(ramps[gi].e - s.e, ramps[gi].n - s.n) for gi, s in m.items()]


def unassigned(inv_xy, pts, r=MATCH_R_M):
    """How many labels have no kept inventory ramp within r."""
    if not len(pts):
        return 0
    d, _i = cKDTree(inv_xy).query(np.asarray(pts, dtype=float), k=1,
                                  distance_upper_bound=r)
    return int((~np.isfinite(d)).sum())


def ps_clusters(pts, pano_ids, t_m=PS_THRESHOLD_M):
    """Cluster ids: label_clustering.py's complete linkage cut at t_m with the same-pano
    cannot-link, blocked into single-linkage components at t_m + BLOCK_MARGIN_M (exact for
    complete linkage; eval_ps_clustering.blocked_components). APPROXIMATE vs the server:
    ENU distances instead of haversine (mm at city scale) and one region (citywide).

    Example:
        >>> ps_clusters(np.array([[0.0, 0.0], [3.0, 0.0], [30.0, 0.0]]), ['a', 'b', 'c']).tolist()
        [0, 0, 1]
        >>> ps_clusters(np.array([[0.0, 0.0], [3.0, 0.0]]), ['a', 'a']).tolist()
        [0, 1]
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = len(pts)
    out = np.full(n, -1, dtype=int)
    if not n:
        return out
    pairs = cKDTree(pts).query_pairs(t_m + BLOCK_MARGIN_M, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
    _nc, comp = connected_components(graph, directed=False)
    order = np.argsort(comp, kind='stable')
    cuts = np.flatnonzero(np.diff(comp[order])) + 1
    _u, codes = np.unique(np.asarray(pano_ids, dtype=object).astype(str), return_inverse=True)
    codes = codes.astype(float)
    nxt = 0
    for block in np.split(order, cuts):
        if len(block) == 1:
            out[block] = nxt
            nxt += 1
            continue
        d = pdist(pts[block])
        same = pdist(codes[block][:, None], 'cityblock') == 0   # same pano: cannot-link
        d = np.where(same, 1e12, d)
        cl = fcluster(linkage(d, method='complete'), t=t_m, criterion='distance')
        # renumber in order of first appearance so ids are deterministic
        remap = {}
        for j, c in zip(block, cl):
            if c not in remap:
                remap[c] = nxt
                nxt += 1
            out[j] = remap[c]
    return out


def loose_merge(clusters, inv_xy, r=MATCH_R_M):
    """EXPLORATORY (amendment 2): (k, n) -- n clusters with >= 2 members assigned to an
    inventory ramp (nearest within r), k of them spanning >= 2 distinct ramps. Unlike
    inventory_metrics' strict merge (>= 2 ramps holding >= 2 members EACH), this sees a
    cluster of two single labels on two ramps, the only merge a one-label-per-site arm
    can make.

    Example:
        >>> loose_merge([((0, 0), [(0, 0), (6, 0)])], np.array([[0.0, 0.0], [6.0, 0.0]]))
        (1, 1)
    """
    tree = cKDTree(inv_xy)
    k = n = 0
    for _c, members in clusters:
        if len(members) < 2:
            continue
        d, i = tree.query(np.asarray(members, dtype=float), k=1, distance_upper_bound=r)
        i = i[np.isfinite(d)]
        if len(i) < 2:
            continue
        n += 1
        k += len(set(i.tolist())) >= 2
    return k, n


def bootstrap_diff(a, b, n_boot=N_BOOT, seed=SEED, qs=(0.5, 0.9)):
    """{q: (lo, hi)} 95% percentile CIs of pct(b, q) - pct(a, q), resampling the PAIRS
    (a[i], b[i]) together."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    out = {q: [] for q in qs}
    ranks = {q: min(n - 1, int(round((n - 1) * q))) for q in qs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sa, sb = np.sort(a[idx]), np.sort(b[idx])
        for q in qs:
            out[q].append(sb[ranks[q]] - sa[ranks[q]])
    return {q: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for q, v in out.items()}


def coverage_loss(inv_xy, pool, labels, arms, sites, pairs, r=MATCH_R_M):
    """EXPLORATORY (amendment 1): where `best` loses coverage against `all`. Pool ramps
    covered (nearest-ramp assignment within r) by `all` but not by `best`, split by what
    covered them under `all`: only labels fusion could not place (`unplaced_only`); labels
    of a site that is paired one-to-one with ANOTHER ramp (`site_paired_elsewhere`, e.g.
    the second ramp of a dual corner fusion holds in one site); labels of a site with no
    pair (`site_unpaired`)."""
    tree = cKDTree(inv_xy)
    owner = site_of(sites)
    paired = {sid for _i, sid in pairs}

    def assign(keys):
        if not keys:
            return np.zeros(0, dtype=int)
        pts = np.array([(labels[k].e, labels[k].n) for k in keys])
        d, i = tree.query(pts, k=1, distance_upper_bound=r)
        return np.where(np.isfinite(d), i, -1)

    a_keys, b_keys = arms[ALL], arms[BEST]
    a_asg, b_asg = assign(a_keys), assign(b_keys)
    covered_b = set(int(i) for i in b_asg if i >= 0)
    by_ramp = {}
    for k, i in zip(a_keys, a_asg):
        if i >= 0 and pool[i] and int(i) not in covered_b:
            by_ramp.setdefault(int(i), []).append(k)
    out = {'lost_ramps': len(by_ramp), 'lost_unplaced_only': 0,
           'lost_site_paired_elsewhere': 0, 'lost_site_unpaired': 0,
           'lost_single_label': sum(1 for v in by_ramp.values() if len(v) == 1)}
    for _i, keys in by_ramp.items():
        sids = {owner.get(k) for k in keys} - {None}
        if not sids:
            out['lost_unplaced_only'] += 1
        elif sids & paired:
            out['lost_site_paired_elsewhere'] += 1
        else:
            out['lost_site_unpaired'] += 1
    return out


def chance_coverage(inv_xy, pool, labels, keys, shift_m=CHANCE_SHIFT_M, seed=SEED,
                    r=MATCH_R_M):
    """EXPLORATORY (amendment 1): covered at r after moving every label shift_m in a
    random direction (seeded) -- the coverage a label cloud of this density earns by
    chance in this inventory."""
    rng = np.random.default_rng(seed)
    pts = np.array([(labels[k].e, labels[k].n) for k in keys]).reshape(-1, 2)
    th = rng.uniform(0, 2 * math.pi, len(pts))
    moved = pts + shift_m * np.column_stack([np.cos(th), np.sin(th)])
    m = ic.inventory_metrics([((e, n), [(e, n)]) for e, n in moved], inv_xy, pool, r)
    return m['covered_rate']


# ----------------------------------------------------------------------------- inventory

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_points(city, runs_root, path=None):
    """([(lat, lng)], provenance dict). Default: the inventory_oracle snapshot, which must
    match the sha256 its inventory.json records."""
    entry = ioracle.INVENTORIES[city]
    rec_path = Path(runs_root) / city / 'inventory_oracle' / 'inventory.json'
    record = json.loads(rec_path.read_text(encoding='utf-8')) if rec_path.exists() else {}
    if path is None:
        path = rec_path.with_name('inventory.geojson')
        if not path.exists():
            raise FileNotFoundError(
                f'{path} is missing (untracked; never pulled on this machine). Needs '
                f'`python scripts/inventory_oracle.py fetch {city} --out <dir>` (network), or '
                '--inventory city=PATH with a local substitute')
    path = Path(path)
    digest = sha256_file(path)
    if path.suffix == '.gz' or path.name.endswith('.jsonl.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            feats = [{'type': 'Feature', 'properties': row,
                      'geometry': {'type': 'Point', 'coordinates': [row['lon'], row['lat']]}}
                     for row in map(json.loads, f)
                     if row.get('lat') is not None and row.get('lon') is not None]
    else:
        feats = json.loads(path.read_text(encoding='utf-8'))['features']
    area = ioracle.load_area(Path(runs_root) / city)
    inside = ioracle.in_area(feats, area)
    keep = ioracle.kept(inside, entry)
    verified = bool(record) and digest == record.get('sha256')
    prov = {'city': city, 'file': str(path), 'sha256': digest,
            'status': 'verified' if verified else 'SUBSTITUTE',
            'record_sha256': record.get('sha256'), 'record_fetched_utc': record.get('fetched_utc'),
            'record_in_area': record.get('in_area'), 'record_kept': record.get('kept'),
            'features': len(feats), 'in_area': len(inside), 'kept': len(keep)}
    pts = [(f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]) for f in keep]
    return pts, prov


# ------------------------------------------------------------------------------- score

def score_frame(city, tier, frame_req, results_path, inv_ll, labels_all):
    """Rows (one per arm), seed rows, a bootstrap row and a survivorship row."""
    panos, _skip, height, _auto = fs.load_at_height(results_path, frame_req)
    params = fs.FuseParams(camera_height_m=height, min_confidence=tier, mask_rig=True,
                           apply_pose=fs.POSE_OFF)
    sites, frame, stats = fs.fuse(panos, params)
    labels = labels_all
    for lab in labels.values():
        lab.e, lab.n = frame.to_enu(lab.lat, lab.lng)
    inv_xy = np.array([frame.to_enu(lat, lng) for lat, lng in inv_ll])
    pool = ic.visible_pool(inv_xy, np.array([frame.to_enu(p.lat, p.lng) for p in panos]))
    arms, per_site, unplaced = build_arms(labels, sites)
    pairs = anchored_pairs(inv_xy, pool, sites)
    fl = fs.frame_label(frame_req)
    base = {'city': city, 'tier': tier, 'frame': fl}
    rows, m1 = [], {}
    for arm, keys in arms.items():
        pts = np.array([(labels[k].e, labels[k].n) for k in keys]).reshape(-1, 2)
        one, every, n_none = anchored_distances(pairs, inv_xy, per_site[arm], labels)
        if one is not None:
            m1[arm] = one
        own = own_one_to_one(inv_xy, pool, pts)
        cov = {r: ic.inventory_metrics([((e, n), [(e, n)]) for e, n in pts], inv_xy, pool, r)
               for r in RADII_M}
        c5 = cov[MATCH_R_M]
        cid = ps_clusters(pts, [k[0] for k in keys])
        groups = {}
        for j, c in enumerate(cid):
            groups.setdefault(int(c), []).append(tuple(pts[j]))
        clusters = [((sum(p[0] for p in g) / len(g), sum(p[1] for p in g) / len(g)), g)
                    for _c, g in sorted(groups.items())]
        cm = ic.inventory_metrics(clusters, inv_xy, pool, MATCH_R_M)
        lk, ln = loose_merge(clusters, inv_xy)
        rows.append({
            **base, 'arm': arm, 'n_labels': len(keys), 'pool': c5['pool'],
            'm1_pairs': len(pairs), 'm1_pairs_without_label': n_none,
            'm1_one_p50': pct(one, 0.5) if one is not None else None,
            'm1_one_p90': pct(one, 0.9) if one is not None else None,
            'm1_one_mean': float(np.mean(one)) if one else None,
            'm1_labelw_n': len(every), 'm1_labelw_p50': pct(every, 0.5),
            'm1_labelw_p90': pct(every, 0.9),
            'm2_matched': len(own), 'm2_coverage': len(own) / c5['pool'] if c5['pool'] else None,
            'm2_p50': pct(own, 0.5), 'm2_p90': pct(own, 0.9),
            **{f'covered_r{r:g}': cov[r]['covered_rate'] for r in RADII_M},
            'labels_per_covered_r5': c5['on_covered'] / c5['covered'] if c5['covered'] else None,
            'multi_label_share_r5': c5['split_rate'],
            'unassigned_r5': unassigned(inv_xy, pts),
            'ps_clusters': cm['n_clusters'], 'ps_covered_r5': cm['covered_rate'],
            'ps_split_r5': cm['split_rate'], 'ps_merge_r5': cm['merge_rate'],
            'ps_merge_k': cm['merge_k'], 'ps_merge_n': cm['merge_n'],
            'ps_merge_loose_r5': lk / ln if ln else None, 'ps_merge_loose_k': lk,
            'ps_merge_loose_n': ln})
    # the control over seeds (sensitivity)
    seed_rows = []
    for seed in SENSITIVITY_SEEDS:
        rand = select_random(sites, seed)
        ps_r = {sid: [k] for sid, k in rand.items()}
        one, _every, _n = anchored_distances(pairs, inv_xy, ps_r, labels)
        pts = [(labels[k].e, labels[k].n) for k in rand.values()]
        c5 = ic.inventory_metrics([(p, [p]) for p in pts], inv_xy, pool, MATCH_R_M)
        seed_rows.append({**base, 'seed': seed, 'm1_one_p50': pct(one, 0.5),
                          'm1_one_p90': pct(one, 0.9), 'covered_r5': c5['covered_rate']})
    ci = bootstrap_diff(m1[BEST], m1[RANDOM])
    boot = {**base, 'n_pairs': len(m1[BEST]), 'n_boot': N_BOOT, 'seed': SEED,
            'best_p50': pct(m1[BEST], 0.5), 'random_p50': pct(m1[RANDOM], 0.5),
            'best_p90': pct(m1[BEST], 0.9), 'random_p90': pct(m1[RANDOM], 0.9),
            'diff_p50': pct(m1[RANDOM], 0.5) - pct(m1[BEST], 0.5),
            'diff_p50_lo': ci[0.5][0], 'diff_p50_hi': ci[0.5][1],
            'diff_p90': pct(m1[RANDOM], 0.9) - pct(m1[BEST], 0.9),
            'diff_p90_lo': ci[0.9][0], 'diff_p90_hi': ci[0.9][1],
            'pairs_same_label': sum(1 for a, b in zip(m1[BEST], m1[RANDOM]) if a == b)}
    ops = operational_sites(sites)
    diag = {**base, **coverage_loss(inv_xy, pool, labels, arms, sites, pairs),
            **{f'chance_covered_r5_{a}': chance_coverage(inv_xy, pool, labels, arms[a])
               for a in (ALL, BEST, RANDOM)}}
    surv = {**base, 'height': str(height), 'panos': len(panos), 'labels': len(labels),
            'labels_in_sites': len(labels) - len(unplaced), 'labels_unplaced': len(unplaced),
            # fusion's own drop counters: every projected detection >= the storage floor,
            # NOT the unplaced labels at this tier (those are labels_unplaced)
            'fusion_drops_out_of_range_or_horizon_all_floor':
                stats['drops']['out_of_range'] + stats['drops']['horizon'],
            'operational_sites': len(ops),
            'single_member_sites': sum(1 for s in ops if len(candidates(s)) == 1),
            'gate20_dropped': len(labels) - len(arms[GATE]),
            'gate_tail_dropped': len(labels) - len(arms[GATE_TAIL]),
            'labels_in_tail': sum(1 for lab in labels.values() if lab.range_m > TAIL_START_M),
            'labels_range_gt20': sum(1 for lab in labels.values() if lab.range_m > GATE_M),
            'pool': int(pool.sum()), 'inventory_kept': len(inv_ll), 'm1_pairs': len(pairs)}
    return rows, seed_rows, boot, surv, diag


def verdict(rows, statuses):
    """Apply the pre-registered rule. rows: arm rows; statuses: {city: 'verified' | ...}.

    PASS iff in both cities (tier 0.30, auto): p90_M1(random) - p90_M1(best) >= 0.5 m and
    covered_r5(best) >= covered_r5(all) - 0.010. UNDETERMINED if either city is missing or
    was scored on a substitute inventory, as the pre-registration says, even when the other
    city fails on its verified snapshot; otherwise FAIL if a city misses either clause."""
    reasons, failed_verified, undetermined = [], False, False
    for city in CITIES:
        def get(arm):
            for r in rows:
                if (r['city'] == city and float(r['tier']) == PRIMARY_TIER
                        and r['frame'] == fs.frame_label(PRIMARY_FRAME) and r['arm'] == arm):
                    return r
            return None
        b, rnd, a = get(BEST), get(RANDOM), get(ALL)
        if b is None or rnd is None or a is None:
            reasons.append(f'{city}: not scored')
            undetermined = True
            continue
        d90 = float(rnd['m1_one_p90']) - float(b['m1_one_p90'])
        dcov = float(b['covered_r5']) - float(a['covered_r5'])
        ok = d90 >= RULE_P90_MARGIN_M - 1e-12 and dcov >= -RULE_COVERED_DROP - 1e-12
        tag = '' if statuses.get(city) == 'verified' else ' [SUBSTITUTE inventory: does not count]'
        reasons.append(f'{city}: p90 random {float(rnd["m1_one_p90"]):.2f} - best '
                       f'{float(b["m1_one_p90"]):.2f} = {d90:+.2f} m (need >= +0.50); '
                       f'covered r5 best {float(b["covered_r5"]):.3f} vs all '
                       f'{float(a["covered_r5"]):.3f} ({dcov:+.3f}, need >= -0.010) -> '
                       f'{"meets" if ok else "does NOT meet"} the rule{tag}')
        if statuses.get(city) != 'verified':
            undetermined = True
        elif not ok:
            failed_verified = True
    if undetermined:
        return UNDETERMINED, reasons
    return (FAIL if failed_verified else PASS), reasons


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def cmd_score(args):
    out = Path(args.out)
    inv_paths = dict(s.split('=', 1) for s in args.inventory)
    for city in args.cities:
        results_path = Path(args.runs_root) / city / 'results.jsonl'
        inv_ll, prov = load_points(city, args.runs_root, inv_paths.get(city))
        prov['results_sha256'] = fs.file_sha256(results_path)
        print(f'{city}: inventory {prov["status"]} ({prov["kept"]} kept; record '
              f'{prov["record_kept"]}), results sha256 {prov["results_sha256"][:12]}',
              file=sys.stderr)
        rows, seeds, boots, survs, diags = [], [], [], [], []
        for tier in args.tier:
            labels = synthesize(results_path, tier)
            for frame in args.frame:
                r, s, b, sv, dg = score_frame(city, tier, frame, results_path, inv_ll, labels)
                diags.append(dg)
                rows += r
                seeds += s
                boots.append(b)
                survs.append(sv)
                print(f'  tier {tier:g} {fs.frame_label(frame)}: best p90 {b["best_p90"]:.2f} '
                      f'random p90 {b["random_p90"]:.2f} diff {b["diff_p90"]:+.2f} '
                      f'[{b["diff_p90_lo"]:+.2f}, {b["diff_p90_hi"]:+.2f}]', file=sys.stderr)
        write_csv(out / f'{city}_arms.csv', rows)
        write_csv(out / f'{city}_seeds.csv', seeds)
        write_csv(out / f'{city}_bootstrap.csv', boots)
        write_csv(out / f'{city}_survivorship.csv', survs)
        write_csv(out / f'{city}_diagnostics.csv', diags)
        (out / f'{city}_inventory.json').write_text(json.dumps(prov, indent=1) + '\n',
                                                    encoding='utf-8', newline='\n')
    print(f'wrote {out}', file=sys.stderr)


def cmd_verdict(args):
    out = Path(args.out)
    rows, statuses = [], {}
    for city in CITIES:
        p = out / f'{city}_arms.csv'
        if p.exists():
            rows += read_csv(p)
            statuses[city] = json.loads((out / f'{city}_inventory.json').read_text())['status']
    v, reasons = verdict(rows, statuses)
    print(v)
    for r in reasons:
        print('-', r)
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('cities', nargs='+', choices=CITIES)
    s.add_argument('--runs-root', default=str(REPO_ROOT / 'runs'))
    s.add_argument('--inventory', action='append', default=[], metavar='CITY=PATH',
                   help='a substitute inventory file (GeoJSON or RampNet jsonl.gz)')
    s.add_argument('--tier', type=float, nargs='+', default=list(TIERS))
    s.add_argument('--frame', type=fs.fuse_camera_height_arg, nargs='+', default=list(FRAMES))
    s.add_argument('--out', default=str(OUT_DIR))
    v = sub.add_parser('verdict')
    v.add_argument('--out', default=str(OUT_DIR))
    args = ap.parse_args(argv)
    if args.cmd == 'score':
        cmd_score(args)
    else:
        cmd_verdict(args)


if __name__ == '__main__':
    main()
