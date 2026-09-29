"""Sidewalk polygons from aerial imagery (Tile2Net) scored against the street-level run (#104).

A STUDY, not production. Tile2Net (VIDA-NYU/tile2net, BSD-3; Hosseini et al., "Mapping the
walk", CEUS 2023) segments orthorectified aerial tiles into sidewalk / crosswalk / road /
footpath polygons in WGS 84. It runs in its own environment on the GPU host; this script
reads its polygons (converted to GeoJSON, see docs/aerial-sidewalk-study.md) and asks the
four questions of #104, under the rules pre-registered on the issue before any GT-joined
number existed:

  gt         Q1  mask quality: distance from every RampNet GT ramp (verdict-true detections
                 and missed-ramp clicks on judged benchmark panos, tier 0.55, raycast at the
                 production `auto` height) and every Bend inventory ramp (#79, image-free)
                 to the nearest sidewalk-or-crosswalk polygon.
  project    Q2  projection: sidewalk/crosswalk polygon edges projected into each judged pano
                 (geo.ground_point_to_pano at `auto` and 2.6 m) vs the reviewer's box centre
                 (Paterson; Bend has no boxes, so its verdict-true peaks and missed clicks),
                 in 1024x512 heatmap px. Also renders a 20-pano gallery per city (untracked;
                 pixels are cut from ../RampNet/benchmark/<city>/panos).
  anchors    Q3  corner anchors (where a crosswalk polygon meets a sidewalk polygon, plus
                 the network's crosswalk endpoints) used to snap or merge clusters of the
                 `fusion` arm and the server-rule `ps @ 7.5 m` offline arm (#106), scored
                 with eval_ps_clustering.score on the same GT pool.
  precision  Q4  operational detections on judged panos: share whose ground point is > 2 m
                 from every walkable polygon, by verdict, with a pano-cluster bootstrap.
  verdict        the pre-registered reading of the four (committed before the run).
  figures        docs/figures/aerial-sidewalks/*.png (matplotlib).

Inputs are read as data from --run-root (default this checkout's runs/): results.jsonl,
area.geojson, the depth index `auto` reads, inventory_oracle/, ps_streets.geojson. RampNet
GT comes from --benchmark-root (default ../RampNet/benchmark). Aerial inputs live in THIS
checkout's runs/<city>/aerial/ (polygons.geojson + network.geojson, untracked, and
tile2net.json, tracked, which records the tile source, zoom, Tile2Net commit and sha256s);
every output goes to runs/<city>/aerial/ and runs/_summary/aerial/. No network.

Bend is a RampNet TRAINING city: every Bend number built on GT or detections says so.

Needs numpy + shapely; `anchors` also pandas/scipy/haversine (through
eval_ps_clustering), `project --gallery` Pillow, `figures` matplotlib. None of it is in
requirements.txt: an analysis tool, not the pipeline.

Usage:
    python scripts/aerial_sidewalks.py gt bend paterson --run-root ../sidewalk-auto-labeler/runs
    python scripts/aerial_sidewalks.py project bend paterson --run-root ... [--gallery]
    python scripts/aerial_sidewalks.py anchors bend paterson --run-root ...
    python scripts/aerial_sidewalks.py precision bend paterson --run-root ...
    python scripts/aerial_sidewalks.py verdict
    python scripts/aerial_sidewalks.py figures
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE, on_camera_rig  # noqa: E402

OUT_NAME = 'aerial'
SUMMARY = REPO_ROOT / 'runs' / '_summary' / OUT_NAME
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'aerial-sidewalks'
CITIES = ('bend', 'paterson')
TRAINING_CITIES = {'bend'}          # RampNet Stage-1 training city: say so beside its numbers
INVENTORY_CITY = 'bend'             # the only one of the two with a city curb-ramp inventory

# Tile2Net classes (the `f_type` column of its polygon output)
SIDEWALK, CROSSWALK, ROAD, FOOTPATH = 'sidewalk', 'crosswalk', 'road', 'footpath'
SURFACE = (SIDEWALK, CROSSWALK)               # Q1/Q2's "sidewalk or crosswalk"
WALKABLE = (SIDEWALK, CROSSWALK, FOOTPATH)    # Q4's "every walkable polygon"

# ---- the pre-registered constants (#104 pre-registration comment) ----
Q1_DISTANCES_M = (1.0, 2.0, 3.0)
Q1_USABLE_SHARE = 0.90          # >= this share of inventory ramps ...
Q1_USABLE_WITHIN_M = 2.0        # ... within this distance of a sidewalk/crosswalk polygon
CHANCE_SHIFT_M = 10.0           # displaced-point chance floor (descriptive, never gated)
CHANCE_SEED = 104
Q2_HEIGHTS = ('auto', 2.6)
Q2_SAMPLE_M = 0.2               # polygon-boundary sampling step before projection
HEATMAP_W, HEATMAP_H = 1024, 512
GALLERY_PANOS = 20
GALLERY_SEED = 104
ANCHOR_RADIUS_M = 3.0           # r_a: a cluster snaps to the nearest anchor within this
ANCHOR_TOUCH_M = 1.0            # a crosswalk "meets" a sidewalk where they are within this
ANCHOR_MERGE_M = 2.0            # anchors closer than this are one anchor
Q3_FRAG_CUT = 0.03              # frag 5 m must fall by >= this (absolute) ...
Q3_COVERAGE_TOL = 0.01          # ... coverage may not fall by more than this ...
Q3_DUAL_TOL = 0.01              # ... nor dual-ramp separation (both / pairs)
Q3_BASES = ('fusion', 'ps @ 7.5 m')
Q3_ARMS = ('anchored', 'merge-at-anchor')
OFF_SURFACE_M = 2.0
Q4_TRUE_ON_MIN = 0.90           # (i) >= this share of verdict-true detections on surface
Q4_GAP_MIN = 0.15               # (ii) False - True off-surface share >= this, CI above 0
Q4_MIN_FALSE = 30               # pooled False detections below this -> underpowered
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 104

MATCH_RADIUS_M = 5.0            # eval_ps_clustering's defaults, so the base rows reproduce
GT_MERGE_M = 2.5                # runs/<city>/ps_clustering_eval_offline_auto_t0.55/arms.csv


def training_note(city):
    return ' (Bend is a RampNet training city)' if city in TRAINING_CITIES else ''


# ============================================================ geometry helpers (tested)

def enu_array(frame, lng, lat):
    """Vectorized geo.LocalFrame.to_enu: arrays of lng/lat -> (east, north) arrays."""
    lng = np.asarray(lng, dtype=float)
    lat = np.asarray(lat, dtype=float)
    e = ((lng - frame.lng0 + 180.0) % 360.0 - 180.0) * frame._m_per_deg_lng
    n = (lat - frame.lat0) * geo.METERS_PER_DEG_LAT
    return e, n


def to_frame(geom, frame):
    """A WGS84 shapely geometry -> the same geometry in the run's ENU metres."""
    import shapely

    def f(coords):
        e, n = enu_array(frame, coords[:, 0], coords[:, 1])
        return np.column_stack([e, n])
    return shapely.transform(geom, f)


class Surface:
    """Polygons of some Tile2Net classes, in ENU metres, with a nearest-distance query.

    Example:
        >>> from shapely.geometry import box
        >>> s = Surface([box(0, 0, 10, 2)])
        >>> [round(float(d), 3) for d in s.distance([(5, 1), (5, 5)])]
        [0.0, 3.0]
    """

    def __init__(self, polygons):
        import shapely
        self.polygons = [p for p in polygons if p is not None and not p.is_empty]
        self.tree = shapely.STRtree(self.polygons) if self.polygons else None

    def distance(self, points):
        """Distance (m) from each (e, n) to the nearest polygon; 0 inside. inf if none."""
        import shapely
        pts = shapely.points(np.asarray(points, dtype=float).reshape(-1, 2))
        if self.tree is None:
            return np.full(len(pts), np.inf)
        idx, dist = self.tree.query_nearest(pts, return_distance=True, all_matches=False)
        out = np.full(len(pts), np.inf)
        out[idx[0]] = dist
        return out

    def near(self, e, n, radius_m):
        """Polygons intersecting the disk of radius_m about (e, n)."""
        import shapely
        if self.tree is None:
            return []
        disk = shapely.points(e, n).buffer(radius_m)
        return [self.polygons[i] for i in self.tree.query(disk, predicate='intersects')]


def crosswalk_anchors(crosswalks, sidewalks, touch_m=ANCHOR_TOUCH_M, merge_m=ANCHOR_MERGE_M,
                      extra_points=()):
    """Corner anchors: where a crosswalk polygon meets a sidewalk polygon.

    For each crosswalk, the part of its boundary within `touch_m` of any sidewalk polygon is
    split into connected pieces, and each piece's centroid is one anchor (a crosswalk that
    spans a street touches the sidewalk at both ends, so it yields two). `extra_points`
    (e.g. the pedestrian network's crosswalk endpoints) join the candidates, and candidates
    closer than `merge_m` are merged greedily into their mean. All in ENU metres.

    Example:
        >>> from shapely.geometry import box
        >>> cw = [box(0, 0, 3, 12)]                       # a crosswalk across a 12 m street
        >>> sw = [box(-10, -3, 13, 0), box(-10, 12, 13, 15)]  # the two sidewalks it meets
        >>> sorted((round(e, 1), round(n, 1)) for e, n in crosswalk_anchors(cw, sw))
        [(1.5, 0.2), (1.5, 11.8)]
    """
    import shapely
    from shapely.ops import unary_union
    cands = [tuple(p) for p in extra_points]
    tree = shapely.STRtree(sidewalks) if sidewalks else None
    for cw in crosswalks:
        if cw is None or cw.is_empty or tree is None:
            continue
        near = [sidewalks[i] for i in tree.query(cw.buffer(touch_m), predicate='intersects')]
        if not near:
            continue
        # clip first: a Tile2Net sidewalk polygon can be a whole connected block network
        local = cw.buffer(2 * touch_m).envelope
        zone = unary_union([s.intersection(local) for s in near]).buffer(touch_m)
        touch = cw.boundary.intersection(zone)
        if touch.is_empty:
            continue
        pieces = getattr(touch, 'geoms', [touch])
        merged = shapely.line_merge(unary_union([g for g in pieces if g.length > 0])) \
            if any(g.length > 0 for g in pieces) else None
        if merged is None or merged.is_empty:
            continue
        # connected pieces: group line parts that touch (a boundary ring cut by a zone can
        # leave a piece split at the ring's start vertex)
        parts = list(getattr(merged, 'geoms', [merged]))
        groups = []
        for part in parts:
            hit = [g for g in groups if any(part.distance(q) < 1e-6 for q in g)]
            new = [part]
            for g in hit:
                new += g
                groups.remove(g)
            groups.append(new)
        for g in groups:
            c = unary_union(g).centroid
            cands.append((c.x, c.y))
    return merge_points(cands, merge_m)


def merge_points(points, merge_m):
    """Greedy merge: each point joins the first kept cluster within merge_m of its mean.

    Example:
        >>> merge_points([(0, 0), (1, 0), (10, 0)], 2.0)
        [(0.5, 0.0), (10.0, 0.0)]
    """
    clusters = []
    for p in points:
        for c in clusters:
            me, mn = c[0] / c[2], c[1] / c[2]
            if math.hypot(p[0] - me, p[1] - mn) <= merge_m:
                c[0] += p[0]
                c[1] += p[1]
                c[2] += 1
                break
        else:
            clusters.append([p[0], p[1], 1])
    return [(c[0] / c[2], c[1] / c[2]) for c in clusters]


def sample_boundary(polygons, step_m=Q2_SAMPLE_M):
    """Points every step_m along every ring of every polygon, as a list of (e, n) runs,
    one run per ring (so a caller can draw each ring as a polyline)."""
    runs = []
    for poly in polygons:
        for g in getattr(poly, 'geoms', [poly]):
            for ring in [g.exterior, *g.interiors]:
                length = ring.length
                if length <= 0:
                    continue
                k = max(2, int(math.ceil(length / step_m)) + 1)
                pts = [ring.interpolate(d) for d in np.linspace(0.0, length, k)]
                runs.append([(p.x, p.y) for p in pts])
    return runs


def heatmap_offset(x_ref, y_ref, x, y):
    """(dx, dy) in 1024x512 heatmap px from a reference to a point, x wrapped at the seam.

    Example:
        >>> heatmap_offset(0.999, 0.5, 0.001, 0.5)
        (2.048, 0.0)
    """
    dx = ((x - x_ref + 0.5) % 1.0 - 0.5) * HEATMAP_W
    return round(dx, 6), round((y - y_ref) * HEATMAP_H, 6)


def bootstrap_gap(groups, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Pano-cluster bootstrap of (off share among False) - (off share among True).

    `groups` is a list of strata (cities); each stratum is a list of panos; each pano is a
    list of (verdict_true: bool, off: bool). Panos are resampled with replacement within
    each stratum. Returns (point, lo, hi) -- the 2.5/97.5 percentiles -- or Nones when a
    side is empty.

    Example:
        >>> panos = [[(True, False), (False, True)]] * 20
        >>> bootstrap_gap([panos], n=200)
        (1.0, 1.0, 1.0)
    """
    def gap(sample):
        t = f = t_off = f_off = 0
        for pano in sample:
            for is_true, off in pano:
                if is_true:
                    t += 1
                    t_off += off
                else:
                    f += 1
                    f_off += off
        if not t or not f:
            return None
        return f_off / f - t_off / t

    point = gap([p for g in groups for p in g])
    if point is None:
        return None, None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = []
        for g in groups:
            sample += [g[rng.randrange(len(g))] for _ in range(len(g))]
        v = gap(sample)
        if v is not None:
            draws.append(v)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return round(point, 6), round(lo, 6), round(hi, 6)


# ============================================================ the pre-registered reading

def q1_verdict(inv_row):
    """Q1: USABLE iff >= 0.90 of the Bend inventory ramps lie within 2 m of a sidewalk or
    crosswalk polygon (inside counts as 0 m). GT shares are reported, never gated."""
    if inv_row is None:
        return 'NOT MEASURED', {'why': 'no inventory row'}
    share = inv_row['share_within_2m']
    return ('USABLE' if share >= Q1_USABLE_SHARE else 'NOT USABLE'), {
        'inventory_share_within_2m': share, 'bar': Q1_USABLE_SHARE,
        'n': inv_row['n']}


def q3_verdict(rows):
    """Q3, per (base, arm): PASS iff in EVERY city frag 5 m falls by >= 0.03 absolute and
    coverage and dual-ramp separation each fall by no more than 0.01. The headline is the
    `fusion` base; the `ps @ 7.5 m` base is judged by the same gate and reported beside it.
    `rows`: dicts with city, base, arm, frag5, coverage, dual (the arm's values) and the
    base's under base_frag5 / base_coverage / base_dual."""
    out = {}
    for base in Q3_BASES:
        for arm in Q3_ARMS:
            cells = [r for r in rows if r['base'] == base and r['arm'] == arm]
            cities = sorted({r['city'] for r in cells})
            ok, why = set(cities) == set(CITIES), []
            if not ok:
                why.append(f'scored in {cities}, the rule needs {list(CITIES)}')
            for r in cells:
                d_frag = r['frag5'] - r['base_frag5']
                d_cov = r['coverage'] - r['base_coverage']
                d_dual = (r['dual'] - r['base_dual']) if r['dual'] is not None \
                    and r['base_dual'] is not None else 0.0
                if d_frag > -Q3_FRAG_CUT:
                    ok = False
                    why.append(f"{r['city']}: frag 5 m {d_frag:+.3f} (needs <= -{Q3_FRAG_CUT})")
                if d_cov < -Q3_COVERAGE_TOL:
                    ok = False
                    why.append(f"{r['city']}: coverage {d_cov:+.3f}")
                if d_dual < -Q3_DUAL_TOL:
                    ok = False
                    why.append(f"{r['city']}: dual {d_dual:+.3f}")
            out[f'{base} / {arm}'] = {'verdict': 'PASS' if ok else 'FAIL',
                                      'cities': cities, 'why': why}
    head = [out[f'fusion / {a}']['verdict'] for a in Q3_ARMS]
    label = 'YES' if 'PASS' in head else 'NO'
    return label, out


def q4_verdict(city_rows, pooled):
    """Q4: FILTER iff (i) >= 0.90 of verdict-true detections are on surface in every city
    AND (ii) the pooled False - True off-surface share is >= 0.15 with its bootstrap 95% CI
    above 0. With fewer than 30 pooled False detections the answer is NOT ESTABLISHED
    (underpowered); otherwise failing either clause is NOT A FILTER."""
    i_ok = all(r['true_on_share'] is not None and r['true_on_share'] >= Q4_TRUE_ON_MIN
               for r in city_rows)
    gap, lo = pooled['gap'], pooled['gap_lo']
    ii_ok = gap is not None and gap >= Q4_GAP_MIN and lo is not None and lo > 0
    detail = {'i_true_on_surface_every_city': i_ok, 'ii_gap': gap, 'ii_ci': [lo,
              pooled['gap_hi']], 'ii_holds': ii_ok, 'n_false_pooled': pooled['n_false']}
    if pooled['n_false'] < Q4_MIN_FALSE:
        return 'NOT ESTABLISHED (underpowered)', detail
    return ('FILTER' if i_ok and ii_ok else 'NOT A FILTER'), detail


# ============================================================ inputs

def out_dir(city):
    d = REPO_ROOT / 'runs' / city / OUT_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_aerial(city):
    """(polygons by class as WGS84 shapely lists, network crosswalk endpoints (lng, lat),
    coverage polygon WGS84, record) from runs/<city>/aerial/."""
    from shapely.geometry import box, shape
    d = REPO_ROOT / 'runs' / city / OUT_NAME
    rec_path = d / 'tile2net.json'
    if not rec_path.exists():
        raise SystemExit(f'{city}: no {rec_path} -- run Tile2Net first (docs/aerial-sidewalk-study.md)')
    record = json.loads(rec_path.read_text(encoding='utf-8'))
    poly_path = d / 'polygons.geojson'
    if not poly_path.exists():
        raise SystemExit(f'{city}: no {poly_path}')
    want = record.get('polygons_geojson_sha256')
    if want and sha256_file(poly_path) != want:
        raise SystemExit(f'{city}: {poly_path.name} does not match the sha256 in tile2net.json')
    by_class = {}
    for ft in json.loads(poly_path.read_text(encoding='utf-8'))['features']:
        by_class.setdefault(ft['properties']['f_type'], []).append(shape(ft['geometry']))
    ends = []
    net_path = d / 'network.geojson'
    if net_path.exists():
        for ft in json.loads(net_path.read_text(encoding='utf-8'))['features']:
            if ft['properties'].get('f_type') != CROSSWALK:
                continue
            g = shape(ft['geometry'])
            for line in getattr(g, 'geoms', [g]):
                cs = list(line.coords)
                ends += [cs[0][:2], cs[-1][:2]]
    w, s, e, n = record['bbox_wgs84']
    return by_class, ends, box(w, s, e, n), record


@dataclass
class City:
    name: str
    run_panos: list
    by_id: dict
    params: object
    frame: object
    height: object
    auto: object
    verdict_panos: dict
    bundle_ops: dict
    surfaces: dict          # class tuple -> Surface (ENU)
    classes_enu: dict       # f_type -> [polygons ENU]
    net_ends_enu: list
    coverage_enu: object
    record: dict


def load_city(city, args, height='auto'):
    import fuse_sites as fs
    import eval_sites as es
    run_dir = args.run_root / city
    verdict_panos, bundle_ops, run_panos, h, auto = es.load_city_at_height(
        city, args.benchmark_root, run_dir, height)
    params = fs.FuseParams(camera_height_m=h, min_confidence=BENCHMARK_CONFIDENCE,
                           mask_rig=True, apply_pose=fs.POSE_OFF)
    frame = geo.LocalFrame(sum(p.lat for p in run_panos) / len(run_panos),
                           sum(p.lng for p in run_panos) / len(run_panos))
    by_class, ends, cov, record = load_aerial(city)
    classes_enu = {k: [to_frame(g, frame) for g in v] for k, v in by_class.items()}
    surfaces = {}
    for key in (SURFACE, WALKABLE, (SIDEWALK,)):
        surfaces[key] = Surface([g for c in key for g in classes_enu.get(c, [])])
    ee, nn = enu_array(frame, [p[0] for p in ends], [p[1] for p in ends])
    return City(city, run_panos, {p.pano_id: p for p in run_panos}, params, frame, h, auto,
                verdict_panos, bundle_ops, surfaces, classes_enu, list(zip(ee, nn)),
                to_frame(cov, frame), record)


def gt_ramps(c):
    """(ramps, points, op_verdicts, counts) at the city's params, as eval_ps_clustering."""
    import eval_sites as es
    points, op_verdicts, counts, _w = es.build_gt(c.verdict_panos, c.bundle_ops, c.by_id,
                                                  c.params, c.frame)
    return es.merge_gt_points(points, GT_MERGE_M), points, op_verdicts, counts


def in_coverage(c, pts):
    import shapely
    from shapely.prepared import prep
    pc = prep(c.coverage_enu)
    return np.array([pc.contains(shapely.points(e, n)) for e, n in pts], dtype=bool)


def chance_points(pts, shift_m=CHANCE_SHIFT_M, seed=CHANCE_SEED):
    rng = random.Random(seed)
    out = []
    for e, n in pts:
        a = rng.uniform(0, 2 * math.pi)
        out.append((e + shift_m * math.sin(a), n + shift_m * math.cos(a)))
    return out


def _q(xs, q):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def provenance_lines(c):
    r = c.record
    return [f"- Tile2Net {r.get('tile2net_version')} commit `{r.get('tile2net_commit')}`; "
            f"tiles: {r.get('tile_source')} at zoom {r.get('zoom')} "
            f"({r.get('n_tiles')} tiles); polygons.geojson sha256 "
            f"`{r.get('polygons_geojson_sha256')}`",
            f"- camera height `auto` -> {c.height}" + (
                f" ({c.auto.get('resolved')}; per capture year: "
                + ', '.join(f"{y} {a['height_m']:g} m" for y, a in
                            sorted((c.auto.get('assignment') or {}).items())) + ')'
                if isinstance(c.auto, dict) else '')]


def distance_summary(label, city, dists, chance=None):
    d = np.asarray(dists, dtype=float)
    row = {'city': city, 'set': label, 'n': int(len(d))}
    for k in Q1_DISTANCES_M:
        row[f'share_within_{k:g}m'] = round(float(np.mean(d <= k)), 4) if len(d) else None
    row['share_inside'] = round(float(np.mean(d == 0)), 4) if len(d) else None
    rest = [x for x in d if x > Q1_DISTANCES_M[0]]
    row['beyond_1m_p50'] = _q(rest, 0.5)
    row['beyond_1m_p90'] = _q(rest, 0.9)
    row['p50'] = _q(list(d), 0.5)
    row['p90'] = _q(list(d), 0.9)
    if chance is not None:
        ch = np.asarray(chance, dtype=float)
        row['chance_within_2m'] = round(float(np.mean(ch <= 2.0)), 4) if len(ch) else None
    return row


# ============================================================ Q1: gt

def cmd_gt(args):
    rows_all = []
    for city in args.cities:
        c = load_city(city, args)
        ramps, _pts, _ov, counts = gt_ramps(c)
        pts = [(r.e, r.n) for r in ramps]
        cov = in_coverage(c, pts)
        surf = c.surfaces[SURFACE]
        d_sc = surf.distance(pts)
        d_walk = c.surfaces[WALKABLE].distance(pts)
        d_sw = c.surfaces[(SIDEWALK,)].distance(pts)
        chance = surf.distance(chance_points(pts))
        per = []
        for k, r in enumerate(ramps):
            kinds = sorted({p.kind for p in r.points})
            lat, lng = c.frame.to_latlng(r.e, r.n)
            years = sorted({(c.by_id[p.pano_id].capture_date or '')[:4] or 'undated'
                            for p in r.points})
            per.append({'city': city, 'ramp': k, 'kinds': '+'.join(kinds),
                        'capture_year': years[-1],
                        'n_points': len(r.points), 'lat': round(lat, 7), 'lng': round(lng, 7),
                        'in_coverage': bool(cov[k]), 'dist_surface_m': round(float(d_sc[k]), 3),
                        'dist_walkable_m': round(float(d_walk[k]), 3),
                        'dist_sidewalk_m': round(float(d_sw[k]), 3),
                        'dist_chance_m': round(float(chance[k]), 3)})
        write_csv(out_dir(city) / 'gt_ramps.csv', per)
        keep = [i for i in range(len(pts)) if cov[i]]
        summary = [distance_summary('gt_all', city, d_sc[keep], chance[keep])]
        for kind in ('det', 'missed'):
            idx = [i for i in keep if kind in per[i]['kinds'].split('+')]
            summary.append(distance_summary(f'gt_{kind}', city, d_sc[idx], chance[idx]))
        # by the capture year of the ramp's newest judged pano: the aerial imagery has one
        # date per city, the panos do not, so a year gap is a candidate for disagreement
        for year in sorted({per[i]['capture_year'] for i in keep}):
            idx = [i for i in keep if per[i]['capture_year'] == year]
            summary.append(distance_summary(f'gt_year_{year}', city, d_sc[idx], chance[idx]))
        summary.append(distance_summary('gt_all_walkable', city, d_walk[keep]))
        summary.append(distance_summary('gt_all_sidewalk_only', city, d_sw[keep]))
        for s in summary:
            s['n_outside_coverage'] = int(len(pts) - len(keep))
        if city == INVENTORY_CITY:
            summary += inventory_rows(c, args)
        rows_all += summary
        write_csv(out_dir(city) / 'q1.csv', summary)
        write_report_q1(c, summary, counts)
        print(f'{city}: {len(ramps)} GT ramps ({len(keep)} in coverage){training_note(city)}')
    write_csv(SUMMARY / 'q1.csv', rows_all)


def inventory_rows(c, args):
    import inventory_oracle as ioracle
    # ioracle.load_inventory reads this checkout's runs/; the snapshot lives under --run-root
    src = args.run_root / c.name / ioracle.OUT_NAME
    record = json.loads((src / 'inventory.json').read_text(encoding='utf-8'))
    ioracle.check_cache(src / 'inventory.geojson', record)
    feats = json.loads((src / 'inventory.geojson').read_text(encoding='utf-8'))['features']
    pts = [(k, f['geometry']['coordinates'][1], f['geometry']['coordinates'][0], None)
           for k, f in enumerate(ioracle.kept(feats, ioracle.INVENTORIES[c.name]))]
    area = ioracle.load_area(args.run_root / c.name)
    from shapely.geometry import Point
    from shapely.prepared import prep
    pa = prep(area)
    inside = [(lat, lng) for _k, lat, lng, _p in pts if pa.covers(Point(lng, lat))]
    ee, nn = enu_array(c.frame, [p[1] for p in inside], [p[0] for p in inside])
    xy = list(zip(ee, nn))
    cov = in_coverage(c, xy)
    xy = [p for p, k in zip(xy, cov) if k]
    surf = c.surfaces[SURFACE]
    d = surf.distance(xy)
    ch = surf.distance(chance_points(xy))
    # the visible pool (#106): inventory ramps within 20 m of a processed pano
    pe, pn = enu_array(c.frame, [p.lng for p in c.run_panos], [p.lat for p in c.run_panos])
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([pe, pn]))
    dist_pano, _ = tree.query(np.asarray(xy))
    vis = dist_pano <= 20.0
    rows = [distance_summary('inventory_all', c.name, d, ch),
            distance_summary('inventory_visible_pool', c.name, d[vis], ch[vis])]
    hist = np.histogram(np.minimum(d, 20.0), bins=np.arange(0, 20.5, 0.5))[0]
    write_csv(out_dir(c.name) / 'inventory_distance_hist.csv',
              [{'bin_lo_m': round(0.5 * i, 1), 'count': int(v)} for i, v in enumerate(hist)])
    for r in rows:
        r['n_outside_coverage'] = int(len(inside) - len(xy))
    return rows


def write_report_q1(c, summary, counts):
    lines = [f'# {c.name}: Q1 aerial mask quality (#104){training_note(c.name)}', '']
    lines += provenance_lines(c)
    lines += [f"- GT: {counts['judged']} judged panos -> {counts['placeable']} placeable "
              f"points (tier {BENCHMARK_CONFIDENCE}), merged at {GT_MERGE_M} m", '',
              '| set | n | inside | <= 1 m | <= 2 m | <= 3 m | p50 | p90 | beyond 1 m p50 / p90 '
              '| chance <= 2 m | outside coverage |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for s in summary:
        lines.append(
            f"| {s['set']} | {s['n']} | {s['share_inside']} | {s['share_within_1m']} | "
            f"{s['share_within_2m']} | {s['share_within_3m']} | {s['p50']} | {s['p90']} | "
            f"{s['beyond_1m_p50']} / {s['beyond_1m_p90']} | {s.get('chance_within_2m', '')} "
            f"| {s['n_outside_coverage']} |")
    lines += ['', 'Distances are to the nearest sidewalk or crosswalk polygon (0 = inside) '
              'unless the set says walkable (adds footpath) or sidewalk_only. `chance` = the '
              f'same points displaced {CHANCE_SHIFT_M:g} m in a seeded random direction, a '
              'floor for how much of the city the polygons cover (descriptive, never gated).']
    (out_dir(c.name) / 'report_q1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ============================================================ Q2: project

def load_boxes(city, benchmark_root):
    import reprojection_residual as rr
    return rr.load_boxes(benchmark_root / city)


def reference_marks(c, boxes):
    """[(pano_id, ref_kind, key, x, y)] per judged pano: box centre where a box exists,
    else the verdict-true peak ('peak') or the missed click ('missed')."""
    import eval_sites as es
    import reprojection_residual as rr
    counts, warnings, out = es.gt_counts(), [], []
    for pid, entry, _rp, ops, _pool in es.judged_gt_panos(
            c.verdict_panos, c.bundle_ops, c.by_id, counts, warnings):
        for k, (v, (_i, x, y, _c)) in enumerate(zip(entry['dets'], ops)):
            if v is not True:
                continue
            b = rr._box_for(boxes, pid, f'det:{k}', x, y, warnings)
            out.append((pid, 'box' if b else 'peak', f'det:{k}', *(b or (x, y))))
        for j, m in enumerate(entry.get('missed', ())):
            if m.get('unsure'):
                continue
            b = rr._box_for(boxes, pid, f'missed:{j}', m['x'], m['y'], warnings)
            out.append((pid, 'box' if b else 'missed', f'missed:{j}', *(b or (m['x'], m['y']))))
    return out


def project_runs(c, pano, height, polygons):
    """Project sampled polygon boundaries into a pano. Returns a list of runs, each a list
    of (x_norm, y_norm) or None where a sample falls outside the raycast's reach."""
    import fuse_sites as fs
    pose = fs.pano_pose(pano, fs.POSE_OFF)
    ce, cn = c.frame.to_enu(pano.lat, pano.lng)
    reach = geo.DEFAULT_MAX_RANGE_M + 0.5       # cheap pre-filter; the projection decides
    out = []
    for run in sample_boundary(polygons):
        proj = []
        for e, n in run:
            if math.hypot(e - ce, n - cn) > reach:
                proj.append(None)
                continue
            lat, lng = c.frame.to_latlng(e, n)
            p = geo.ground_point_to_pano(pose, lat, lng, camera_height=height)
            proj.append(None if p is None else (p.x_norm, p.y_norm))
        out.append(proj)
    return out


def cmd_project(args):
    import fuse_sites as fs
    rows_all, summ_all = [], []
    for city in args.cities:
        c = load_city(city, args)
        boxes = load_boxes(city, args.benchmark_root)
        marks = reference_marks(c, boxes)
        by_pano = {}
        for m in marks:
            by_pano.setdefault(m[0], []).append(m)
        rows = []
        heights = {'auto': c.height, '2.6': 2.6}
        cache = {}
        from shapely.geometry import Point
        reach = c.coverage_enu.buffer(-geo.DEFAULT_MAX_RANGE_M)   # the whole 25 m disk mapped
        outside = [pid for pid in by_pano
                   if not reach.contains(Point(*c.frame.to_enu(c.by_id[pid].lat,
                                                                c.by_id[pid].lng)))]
        for pid in outside:
            del by_pano[pid]
        for pid in sorted(by_pano):
            pano = c.by_id[pid]
            ce, cn = c.frame.to_enu(pano.lat, pano.lng)
            polys = c.surfaces[SURFACE].near(ce, cn, geo.DEFAULT_MAX_RANGE_M + 1.0)
            pose = fs.pano_pose(pano, fs.POSE_OFF)
            for hname, h in heights.items():
                runs = project_runs(c, pano, h, polys)
                cache[(pid, hname)] = runs
                samples = np.array([p for r in runs for p in r if p is not None]) \
                    if runs else np.zeros((0, 2))
                for _pid, kind, key, x, y in by_pano[pid]:
                    row = {'city': city, 'pano_id': pid, 'ref_kind': kind, 'ref': key,
                           'height': hname, 'x': round(x, 6), 'y': round(y, 6)}
                    if len(samples):
                        dx = ((samples[:, 0] - x + 0.5) % 1.0 - 0.5) * HEATMAP_W
                        dy = (samples[:, 1] - y) * HEATMAP_H
                        dd = np.hypot(dx, dy)
                        k = int(np.argmin(dd))
                        row.update(px=round(float(dd[k]), 3), dx_px=round(float(dx[k]), 3),
                                   dy_px=round(float(dy[k]), 3))
                    else:
                        row.update(px=None, dx_px=None, dy_px=None)
                    g = geo.detection_ground_point(pose, x, y, camera_height=h)
                    if g is None:
                        row.update(world_dist_m=None, on_surface=None)
                    else:
                        e, n = c.frame.to_enu(g.lat, g.lng)
                        d = float(c.surfaces[SURFACE].distance([(e, n)])[0])
                        row.update(world_dist_m=round(d, 3), on_surface=d == 0.0,
                                   range_m=round(g.range_m, 2))
                    rows.append(row)
        write_csv(out_dir(city) / 'projection.csv', rows,
                  ['city', 'pano_id', 'ref_kind', 'ref', 'height', 'x', 'y', 'px', 'dx_px',
                   'dy_px', 'world_dist_m', 'on_surface', 'range_m'])
        summ = []
        for hname in heights:
            for kind in ('all', 'box', 'peak', 'missed'):
                sel = [r for r in rows if r['height'] == hname
                       and (kind == 'all' or r['ref_kind'] == kind)]
                if not sel:
                    continue
                px = [r['px'] for r in sel if r['px'] is not None]
                summ.append({'city': city, 'height': hname, 'ref_kind': kind, 'n': len(sel),
                             'n_with_edge': len(px), 'px_p50': _q(px, 0.5),
                             'px_p90': _q(px, 0.9),
                             'dx_px_p50': _q([r['dx_px'] for r in sel if r['dx_px'] is not None], .5),
                             'dy_px_p50': _q([r['dy_px'] for r in sel if r['dy_px'] is not None], .5),
                             'share_px_le_5': round(sum(p <= 5 for p in px) / len(px), 4) if px else None,
                             'world_dist_p50': _q([r['world_dist_m'] for r in sel], 0.5),
                             'on_surface_share': round(
                                 sum(1 for r in sel if r['on_surface']) /
                                 max(1, sum(1 for r in sel if r['on_surface'] is not None)), 4)})
        write_csv(out_dir(city) / 'q2.csv', summ)
        summ_all += summ
        rows_all += rows
        write_report_q2(c, summ, len(by_pano), len(outside))
        if args.gallery:
            render_gallery(c, by_pano, cache, args)
        print(f'{city}: {len(rows)} reference rows on {len(by_pano)} panos{training_note(city)}')
    write_csv(SUMMARY / 'q2.csv', summ_all)


def write_report_q2(c, summ, n_panos, n_outside=0):
    lines = [f'# {c.name}: Q2 projected sidewalk edges vs reviewer marks (#104)'
             f'{training_note(c.name)}', '']
    lines += provenance_lines(c)
    lines += [f'- {n_panos} judged panos with at least one reference mark and their whole '
              f'25 m disk inside the Tile2Net coverage ({n_outside} more left out)', '',
              '| height | reference | n | with an edge in range | px p50 | px p90 | '
              'dx p50 | dy p50 | share <= 5 px | world dist p50 (m) | ref on surface |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summ:
        lines.append(f"| {s['height']} | {s['ref_kind']} | {s['n']} | {s['n_with_edge']} | "
                     f"{s['px_p50']} | {s['px_p90']} | {s['dx_px_p50']} | {s['dy_px_p50']} | "
                     f"{s['share_px_le_5']} | {s['world_dist_p50']} | {s['on_surface_share']} |")
    lines += ['', 'px = distance in 1024x512 heatmap pixels (the grid RampNet predicts on; '
              '1 px = 0.35 deg) from the reference to the nearest projected sidewalk/crosswalk '
              'boundary sample (every 0.2 m, within the 25 m raycast cap); dx/dy = that '
              'nearest sample minus the reference (dy > 0 = the edge projects lower, i.e. '
              'nearer). It mixes mask error with projection error (height, heading, GPS) '
              'by construction. `box` = the reviewer\'s extent-box centre (Paterson only), '
              '`peak` = a verdict-true detection with no box, `missed` = a missed-ramp click. '
              'world dist = the reference raycast at that height, metres to the nearest '
              'polygon (0 inside).']
    (out_dir(c.name) / 'report_q2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def render_gallery(c, by_pano, cache, args):
    """20 seeded panos per city: the lower half of the equirect, with the projected
    sidewalk/crosswalk edges (auto solid, 2.6 m dashed) and the reference marks. Untracked
    (pixels) -> runs/<city>/aerial/gallery/index.html."""
    import base64
    import io
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    pano_dir = args.benchmark_root / c.name / 'panos'
    ids = sorted(by_pano)
    random.Random(GALLERY_SEED).shuffle(ids)
    ids = ids[:GALLERY_PANOS]
    W, y0, y1 = 2048, 0.47, 0.68
    H = int(round(W / 2 * (y1 - y0) * 2))
    cards = []
    for pid in sorted(ids):
        path = pano_dir / f'{pid}.jpg'
        if not path.exists():
            continue
        im = Image.open(path).convert('RGB')
        iw, ih = im.size
        crop = im.crop((0, int(y0 * ih), iw, int(y1 * ih))).resize((W, H))
        buf = io.BytesIO()
        crop.save(buf, 'JPEG', quality=78)
        src = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

        def pts(run):
            segs, cur = [], []
            for p in run:
                if p is None:
                    if len(cur) > 1:
                        segs.append(cur)
                    cur = []
                    continue
                x, y = p[0] * W, (p[1] - y0) / (y1 - y0) * H
                if cur and abs(x - cur[-1][0]) > W / 2:      # seam wrap
                    if len(cur) > 1:
                        segs.append(cur)
                    cur = []
                cur.append((x, y))
            if len(cur) > 1:
                segs.append(cur)
            return segs
        svg = [f'<svg viewBox="0 0 {W} {H}" class="ov" aria-hidden="true">']
        for hname, style in (('2.6', 'stroke:#ffd23f;stroke-dasharray:6 5'),
                             ('auto', 'stroke:#2cf5a0')):
            for run in cache.get((pid, hname), []):
                for seg in pts(run):
                    d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in seg)
                    svg.append(f'<polyline points="{d}" style="fill:none;stroke-width:2;{style}"/>')
        for _pid, kind, key, x, y in by_pano[pid]:
            cx, cy = x * W, (y - y0) / (y1 - y0) * H
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" style="fill:none;'
                       f'stroke:#ff3b6b;stroke-width:3"><title>{kind} {key}</title></circle>')
        svg.append('</svg>')
        cards.append(f'<figure><div class="wrap"><img src="{src}" alt="Lower half of pano '
                     f'{pid} with projected sidewalk edges">{"".join(svg)}</div>'
                     f'<figcaption>{pid} — {len(by_pano[pid])} reference marks</figcaption>'
                     '</figure>')
    html = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<title>Aerial edges {c.name}</title><style>',
            ':root{--bg:#fff;--fg:#111}@media (prefers-color-scheme:dark){:root{--bg:#121212;'
            '--fg:#eee}}body{background:var(--bg);color:var(--fg);font:14px system-ui;'
            'margin:16px}.wrap{position:relative}.wrap img{width:100%;display:block}'
            '.ov{position:absolute;inset:0;width:100%;height:100%}figure{margin:0 0 24px}',
            '</style></head><body>',
            f'<h1>{c.name}: Tile2Net sidewalk/crosswalk edges projected into judged panos</h1>',
            f'<p>Equirect rows {y0}-{y1} of each pano. Green = polygon edges at the `auto` '
            f'height ({c.height}), yellow dashed = 2.6 m; red rings = reviewer marks (box '
            f'centre, else verdict-true peak or missed click). Seeded sample of '
            f'{len(cards)} panos (seed {GALLERY_SEED}).{training_note(c.name)}</p>',
            *cards, '</body></html>']
    g = out_dir(c.name) / 'gallery'
    g.mkdir(exist_ok=True)
    (g / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'  gallery -> {g / "index.html"} ({len(cards)} panos)')


# ============================================================ Q3: anchors

def anchor_clusters(clusters, anchors, radius_m=ANCHOR_RADIUS_M):
    """(anchored, merged) copies of placed clusters: `anchored` moves each cluster to the
    nearest anchor within radius_m (else leaves it); `merged` additionally unites every
    cluster that snapped to the same anchor into one (members pooled, placed at the anchor).

    Example:
        >>> from types import SimpleNamespace as N
        >>> cs = [N(id=0, members=[('a', 0)], n_labels=1, e=0.5, n=0.0, label_ids=[]),
        ...       N(id=1, members=[('b', 0)], n_labels=1, e=-0.5, n=0.0, label_ids=[]),
        ...       N(id=2, members=[('c', 0)], n_labels=1, e=20.0, n=0.0, label_ids=[])]
        >>> a, m = anchor_clusters(cs, [(0.0, 0.0)])
        >>> [(c.e, c.n) for c in a], len(m), sorted(len(c.members) for c in m)
        ([(0.0, 0.0), (0.0, 0.0), (20.0, 0.0)], 2, [1, 2])
    """
    import copy
    from scipy.spatial import cKDTree
    placed = [c for c in clusters if c.e is not None]
    tree = cKDTree(np.asarray(anchors)) if len(anchors) else None
    anchored, groups, merged = [], {}, []
    for c in clusters:
        a = copy.copy(c)
        if c.e is not None and tree is not None:
            d, k = tree.query([c.e, c.n])
            if d <= radius_m:
                a.e, a.n = float(anchors[k][0]), float(anchors[k][1])
                groups.setdefault(int(k), []).append(a)
        anchored.append(a)
    snapped = {id(a) for g in groups.values() for a in g}
    next_id = max((c.id for c in clusters), default=-1) + 1
    for a in anchored:
        if id(a) not in snapped:
            merged.append(a)
    for k in sorted(groups):
        g = groups[k]
        if len(g) == 1:
            merged.append(g[0])
            continue
        m = copy.copy(g[0])
        m.id = next_id
        next_id += 1
        m.members = [x for c in g for x in c.members]
        m.n_labels = sum(c.n_labels for c in g)
        m.label_ids = [x for c in g for x in getattr(c, 'label_ids', [])]
        merged.append(m)
    del placed
    return anchored, merged


def city_anchors(c):
    cw = c.classes_enu.get(CROSSWALK, [])
    sw = c.classes_enu.get(SIDEWALK, [])
    poly_only = crosswalk_anchors(cw, sw)
    both = crosswalk_anchors(cw, sw, extra_points=c.net_ends_enu)
    return poly_only, both


def q3_row(city, base, arm, r, b):
    def dual(x):
        d = x['dual']
        return d['both'] / d['pairs'] if d['pairs'] else None
    fr, bfr = r['frag'][5.0], b['frag'][5.0]
    return {'city': city, 'base': base, 'arm': arm, 'n_clusters': r['n_clusters'],
            'coverage': round(r['coverage'], 4), 'covered': r['covered'], 'n_pool': r['n_pool'],
            'frag5': round(fr['with_extra'] / fr['ramps'], 4) if fr['ramps'] else None,
            'frag5_extra': fr['extra'], 'frag3': round(
                r['frag'][3.0]['with_extra'] / r['frag'][3.0]['ramps'], 4)
            if r['frag'][3.0]['ramps'] else None,
            'dual': round(dual(r), 4) if dual(r) is not None else None,
            'dual_pairs': r['dual']['pairs'], 'dual_both': r['dual']['both'],
            'precision': round(r['precision'], 4) if r['precision'] is not None else None,
            'tp': r['tp'], 'fp': r['fp'], 'same_pano_pairs': r['same_pano_pairs'],
            'base_frag5': round(bfr['with_extra'] / bfr['ramps'], 4) if bfr['ramps'] else None,
            'base_coverage': round(b['coverage'], 4),
            'base_dual': round(dual(b), 4) if dual(b) is not None else None}


def cmd_anchors(args):
    import fuse_sites as fs
    import eval_ps_clustering as epc
    rows_all, anchor_stats, sens_all = [], [], []
    for city in args.cities:
        c = load_city(city, args)
        run_dir = args.run_root / city
        results_path = run_dir / 'results.jsonl'
        dets, frame, _drops = fs.project(c.run_panos, c.params)
        if (frame.lat0, frame.lng0) != (c.frame.lat0, c.frame.lng0):
            raise SystemExit('frame mismatch')
        det_pos = {(d.pano_id, d.det_index): (d.e, d.n) for d in dets}
        ramps, points, op_verdicts, _counts = gt_ramps(c)
        pool = [r for r in ramps if r.in_pool]
        gt = (ramps, pool, points, op_verdicts)
        # base arms, built exactly as eval_ps_clustering --offline builds them
        labels, det_of = epc.synthesize_labels(results_path, BENCHMARK_CONFIDENCE, mask_rig=True)
        streets_path = run_dir / 'ps_streets.geojson'
        streets = epc.load_streets(streets_path) if streets_path.exists() else []
        if streets:
            reg, _ties = epc.assign_regions(labels.lat, labels.lng, streets)
            labels['region_id'] = reg
        t_km = epc.PS_THRESHOLD_KM
        part = epc.ps_partition(labels, [t_km], per_region=True)[t_km]
        bases = {
            'ps @ 7.5 m': epc.place(epc.clusters_from_assignment(labels, part, det_of), det_pos),
        }
        sites, _f2, _s = fs.fuse(c.run_panos, c.params)
        bases['fusion'] = epc.place(epc.clusters_from_sites(sites, False), det_pos)
        poly_only, anchors = city_anchors(c)
        # how many GT ramps have an anchor within r_a: the ceiling on what snapping can do
        if anchors:
            from scipy.spatial import cKDTree
            at = cKDTree(np.asarray(anchors))
            dd, _ = at.query(np.asarray([[r.e, r.n] for r in pool]))
            near_share = float(np.mean(dd <= ANCHOR_RADIUS_M))
            dd_all = dd
        else:
            near_share, dd_all = 0.0, []
        anchor_stats.append({'city': city, 'n_anchors': len(anchors),
                             'n_anchors_polygon_only': len(poly_only),
                             'n_network_endpoints': len(c.net_ends_enu),
                             'pool_ramps': len(pool),
                             'pool_share_anchor_within_3m': round(near_share, 4),
                             'pool_anchor_dist_p50': _q(list(dd_all), 0.5)})
        rows = []
        for base in Q3_BASES:
            b = epc.score(bases[base], gt, det_pos, MATCH_RADIUS_M)
            rows.append(q3_row(city, base, 'base', b, b))
            anchored, merged = anchor_clusters(bases[base], anchors)
            for arm, cl in (('anchored', anchored), ('merge-at-anchor', merged)):
                rows.append(q3_row(city, base, arm, epc.score(cl, gt, det_pos, MATCH_RADIUS_M), b))
        write_csv(out_dir(city) / 'q3.csv', rows)
        rows_all += rows
        # sensitivity, never gated: anchors from polygon contacts alone (Bend's Tile2Net
        # network step failed, so there this is the same set as the headline)
        sens = []
        for base in Q3_BASES:
            b = epc.score(bases[base], gt, det_pos, MATCH_RADIUS_M)
            anchored, merged = anchor_clusters(bases[base], poly_only)
            for arm, cl in (('anchored', anchored), ('merge-at-anchor', merged)):
                sens.append(q3_row(city, base, arm, epc.score(cl, gt, det_pos, MATCH_RADIUS_M), b))
        write_csv(out_dir(city) / 'q3_polygon_anchors.csv', sens)
        sens_all += sens
        write_report_q3(c, rows, anchor_stats[-1])
        print(f'{city}: {len(anchors)} anchors; {near_share:.2f} of pool ramps have one '
              f'within {ANCHOR_RADIUS_M:g} m{training_note(city)}')
    write_csv(SUMMARY / 'q3.csv', rows_all)
    write_csv(SUMMARY / 'q3_anchors.csv', anchor_stats)
    write_csv(SUMMARY / 'q3_polygon_anchors.csv', sens_all)


def write_report_q3(c, rows, st):
    lines = [f'# {c.name}: Q3 corner anchors vs cluster fragmentation (#104)'
             f'{training_note(c.name)}', '']
    lines += provenance_lines(c)
    lines += [f"- {st['n_anchors']} anchors ({st['n_anchors_polygon_only']} from polygon "
              f"contacts alone; {st['n_network_endpoints']} network crosswalk endpoints "
              f"joined, merged within {ANCHOR_MERGE_M:g} m); {st['pool_share_anchor_within_3m']}"
              f" of the {st['pool_ramps']} GT pool ramps have an anchor within "
              f"{ANCHOR_RADIUS_M:g} m (median distance {st['pool_anchor_dist_p50']} m)",
              f'- labels synthesized at tier {BENCHMARK_CONFIDENCE} as '
              '`eval_ps_clustering.py --offline` does; the `base` rows must reproduce '
              "that script's committed `ps @ 7.5 m` and `fusion` rows", '',
              '| base | arm | clusters | coverage | frag 5 m (extra) | frag 3 m | dual (both/pairs) '
              '| precision (TP/FP) | same-pano pairs |',
              '|---|---|---:|---:|---|---:|---|---|---:|']
    for r in rows:
        lines.append(f"| {r['base']} | {r['arm']} | {r['n_clusters']} | {r['coverage']} "
                     f"({r['covered']}/{r['n_pool']}) | {r['frag5']} ({r['frag5_extra']}) | "
                     f"{r['frag3']} | {r['dual']} ({r['dual_both']}/{r['dual_pairs']}) | "
                     f"{r['precision']} ({r['tp']}/{r['fp']}) | {r['same_pano_pairs']} |")
    lines += ['', 'Gate (pre-registered on #104): an arm passes when, in BOTH cities, frag 5 m '
              f'falls by >= {Q3_FRAG_CUT} absolute against its base while coverage and dual-ramp '
              f'separation fall by no more than {Q3_COVERAGE_TOL} / {Q3_DUAL_TOL}. '
              '`merge-at-anchor` can put two detections from one pano in one cluster (the '
              'fusion cannot-link); the same-pano column counts that.']
    (out_dir(c.name) / 'report_q3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ============================================================ Q4: precision

def cmd_precision(args):
    import eval_sites as es
    import fuse_sites as fs
    city_rows, strata = [], []
    for city in args.cities:
        c = load_city(city, args)
        counts, warnings, rows = es.gt_counts(), [], []
        panos = []
        walk = c.surfaces[WALKABLE]
        from shapely.geometry import Point
        from shapely.prepared import prep
        covered = prep(c.coverage_enu)
        for pid, entry, rp, ops, _pool in es.judged_gt_panos(
                c.verdict_panos, c.bundle_ops, c.by_id, counts, warnings):
            verdict_of = {i: v for v, (i, *_r) in zip(entry['dets'], ops)}
            pose = fs.pano_pose(rp, fs.POSE_OFF)
            pano_items = []
            for i, x, y, conf in rp.detections:
                if conf < OPERATIONAL_CONFIDENCE or on_camera_rig(y):
                    continue
                v = verdict_of.get(i, 'unjudged') if conf >= BENCHMARK_CONFIDENCE else 'band'
                g = geo.detection_ground_point(pose, x, y, camera_height=c.params.camera_height_m)
                row = {'city': city, 'pano_id': pid, 'det_index': i,
                       'confidence': round(conf, 4),
                       'verdict': {True: 'true', False: 'false'}.get(v, str(v)),
                       'range_m': None, 'dist_walkable_m': None, 'off_surface': None}
                if g is not None and not covered.contains(
                        Point(*c.frame.to_enu(g.lat, g.lng))):
                    row['verdict'] += ':outside_coverage'
                elif g is not None:
                    e, n = c.frame.to_enu(g.lat, g.lng)
                    d = float(walk.distance([(e, n)])[0])
                    row.update(range_m=round(g.range_m, 2), dist_walkable_m=round(d, 3),
                               off_surface=d > OFF_SURFACE_M)
                    if v in (True, 'duplicate', False):
                        pano_items.append((v is not False, d > OFF_SURFACE_M))
                rows.append(row)
            if pano_items:
                panos.append(pano_items)
        write_csv(out_dir(city) / 'precision_detections.csv', rows)
        strata.append(panos)

        def share(sel):
            placed = [r for r in sel if r['off_surface'] is not None]
            return (round(sum(1 for r in placed if r['off_surface']) / len(placed), 4)
                    if placed else None), len(placed), len(sel) - len(placed)
        t_rows = [r for r in rows if r['verdict'] in ('true', 'duplicate')]
        f_rows = [r for r in rows if r['verdict'] == 'false']
        b_rows = [r for r in rows if r['verdict'] == 'band']
        t_off, n_t, u_t = share(t_rows)
        f_off, n_f, u_f = share(f_rows)
        b_off, n_b, u_b = share(b_rows)
        gap, lo, hi = bootstrap_gap([panos])
        city_rows.append({'city': city, 'n_true': n_t, 'n_false': n_f, 'n_band': n_b,
                          'n_outside_coverage': sum(1 for r in rows
                                                    if r['verdict'].endswith('outside_coverage')),
                          'unplaceable_true': u_t, 'unplaceable_false': u_f,
                          'unplaceable_band': u_b,
                          'true_off_share': t_off,
                          'true_on_share': round(1 - t_off, 4) if t_off is not None else None,
                          'false_off_share': f_off, 'band_off_share': b_off,
                          'gap': gap, 'gap_lo': lo, 'gap_hi': hi})
        print(f'{city}: {n_t} true, {n_f} false, {n_b} band detections placed'
              f'{training_note(city)}')
    gap, lo, hi = bootstrap_gap(strata)
    pooled = {'city': 'pooled', 'n_true': sum(r['n_true'] for r in city_rows),
              'n_false': sum(r['n_false'] for r in city_rows), 'gap': gap, 'gap_lo': lo,
              'gap_hi': hi}
    write_csv(SUMMARY / 'q4.csv', city_rows + [pooled],
              ['city', 'n_true', 'n_false', 'n_band', 'n_outside_coverage', 'unplaceable_true', 'unplaceable_false',
               'unplaceable_band', 'true_off_share', 'true_on_share', 'false_off_share',
               'band_off_share', 'gap', 'gap_lo', 'gap_hi'])
    for r in city_rows:
        write_csv(out_dir(r['city']) / 'q4.csv', [r])


# ============================================================ verdict

def _num(v):
    if v in (None, '', 'None'):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def cmd_verdict(args):
    out = {}
    q1 = [{k: _num(v) for k, v in r.items()} for r in read_csv(SUMMARY / 'q1.csv')]
    inv = next((r for r in q1 if r['city'] == INVENTORY_CITY and r['set'] == 'inventory_all'),
               None)
    label, detail = q1_verdict(inv)
    detail['gt_share_within_2m'] = {r['city']: r['share_within_2m'] for r in q1
                                    if r['set'] == 'gt_all'}
    out['Q1'] = {'answer': label, **detail}
    q2 = read_csv(SUMMARY / 'q2.csv')
    out['Q2'] = {'answer': 'DESCRIPTIVE (no gate pre-registered)',
                 'px_p50': {f"{r['city']} @ {r['height']}": _num(r['px_p50'])
                            for r in q2 if r['ref_kind'] == 'all'}}
    q3 = [{k: _num(v) for k, v in r.items()} for r in read_csv(SUMMARY / 'q3.csv')
          if r['arm'] != 'base']
    label, detail = q3_verdict(q3)
    out['Q3'] = {'answer': label, 'arms': detail}
    q4 = [{k: _num(v) for k, v in r.items()} for r in read_csv(SUMMARY / 'q4.csv')]
    pooled = next(r for r in q4 if r['city'] == 'pooled')
    pooled['n_false'] = int(pooled['n_false'])
    label, detail = q4_verdict([r for r in q4 if r['city'] != 'pooled'], pooled)
    out['Q4'] = {'answer': label, **detail}
    out['note'] = 'Bend is a RampNet training city; its GT and detection numbers carry that.'
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / 'verdict.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=1))


# ============================================================ figures

def cmd_figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {'bend': '#3b6fb6', 'paterson': '#d9822b'}
    # Q1: cumulative share within d
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for city in CITIES:
        p = REPO_ROOT / 'runs' / city / OUT_NAME / 'gt_ramps.csv'
        if not p.exists():
            continue
        rows = [r for r in read_csv(p) if r['in_coverage'] == 'True']
        d = np.sort([float(r['dist_surface_m']) for r in rows])
        ch = np.sort([float(r['dist_chance_m']) for r in rows])
        y = np.arange(1, len(d) + 1) / len(d)
        lbl = city + (' (training city)' if city in TRAINING_CITIES else '')
        ax.step(d, y, where='post', color=colors[city], label=f'{lbl} GT ramps')
        ax.step(ch, y, where='post', color=colors[city], ls=':', label=f'{city} displaced 10 m')
    hist = REPO_ROOT / 'runs' / INVENTORY_CITY / OUT_NAME / 'inventory_distance_hist.csv'
    if hist.exists():
        h = read_csv(hist)
        cnt = np.array([int(r['count']) for r in h])
        lo = np.array([float(r['bin_lo_m']) for r in h])
        ax.step(lo, np.cumsum(cnt) / cnt.sum(), where='post', color='#2a9d5c',
                label='Bend inventory ramps (0.5 m bins)')
    ax.axvline(Q1_USABLE_WITHIN_M, color='#888', lw=0.8)
    ax.axhline(Q1_USABLE_SHARE, color='#888', lw=0.8)
    ax.set_xlim(0, 10)
    ax.set_xlabel('distance to nearest sidewalk/crosswalk polygon (m)')
    ax.set_ylabel('cumulative share')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Q1: ramps vs Tile2Net sidewalk/crosswalk polygons')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'q1_distance_cdf.png', dpi=140)
    plt.close(fig)
    # Q2: px histogram
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for city in CITIES:
        p = REPO_ROOT / 'runs' / city / OUT_NAME / 'projection.csv'
        if not p.exists():
            continue
        for hname, ls in (('auto', '-'), ('2.6', '--')):
            px = np.sort([float(r['px']) for r in read_csv(p)
                          if r['height'] == hname and r['px']])
            if len(px):
                ax.step(px, np.arange(1, len(px) + 1) / len(px), where='post', ls=ls,
                        color=colors[city], label=f'{city} @ {hname}')
    ax.set_xlim(0, 40)
    ax.set_xlabel('reference mark to nearest projected edge (1024x512 heatmap px)')
    ax.set_ylabel('cumulative share')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Q2: projected sidewalk edges vs reviewer marks')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'q2_edge_px_cdf.png', dpi=140)
    plt.close(fig)
    # Q3: frag 5 m by arm
    p = SUMMARY / 'q3.csv'
    if p.exists():
        rows = read_csv(p)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
        for ax, base in zip(axes, Q3_BASES):
            arms = ['base', *Q3_ARMS]
            x = np.arange(len(arms))
            for k, city in enumerate(CITIES):
                vals = [next((float(r['frag5']) for r in rows if r['city'] == city
                              and r['base'] == base and r['arm'] == a), np.nan) for a in arms]
                ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=colors[city], label=city)
            ax.set_xticks(x, arms, fontsize=8)
            ax.set_title(base, fontsize=9)
        axes[0].set_ylabel('frag 5 m (share of covered GT ramps)')
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle('Q3: corner anchors vs fragmentation', fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'q3_frag5.png', dpi=140)
        plt.close(fig)
    # Q4
    p = SUMMARY / 'q4.csv'
    if p.exists():
        rows = [r for r in read_csv(p) if r['city'] != 'pooled']
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        x = np.arange(len(rows))
        for k, (col, lbl, colr) in enumerate((('true_off_share', 'verdict true', '#3b6fb6'),
                                              ('false_off_share', 'verdict false', '#c0392b'),
                                              ('band_off_share', '0.30-0.55 (unjudged)',
                                               '#999999'))):
            vals = [float(r[col]) if r[col] else np.nan for r in rows]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=colr, label=lbl)
        ax.set_xticks(x, [r['city'] for r in rows])
        ax.set_ylabel(f'share > {OFF_SURFACE_M:g} m from any walkable polygon')
        ax.legend(fontsize=8, frameon=False)
        ax.set_title('Q4: off-surface detections by verdict')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'q4_off_surface.png', dpi=140)
        plt.close(fig)
    print(f'figures -> {FIG_DIR}')


# ============================================================ CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('gt', 'project', 'anchors', 'precision'):
        p = sub.add_parser(name)
        p.add_argument('cities', nargs='*', default=list(CITIES))
        p.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs',
                       help='where runs/<city>/{results.jsonl,depth,...} are read from (as '
                            'data); outputs always go to this checkout\'s runs/')
        p.add_argument('--benchmark-root', type=Path,
                       default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
        if name == 'project':
            p.add_argument('--gallery', action='store_true',
                           help='also render the 20-pano HTML gallery (untracked)')
    sub.add_parser('verdict')
    sub.add_parser('figures')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {'gt': cmd_gt, 'project': cmd_project, 'anchors': cmd_anchors,
     'precision': cmd_precision, 'verdict': cmd_verdict, 'figures': cmd_figures}[args.cmd](args)


if __name__ == '__main__':
    main()
