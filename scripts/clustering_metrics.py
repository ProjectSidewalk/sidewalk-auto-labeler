"""The pure scoring of eval_ps_clustering.py: clusters, the scorer, the report rows, the
GT-free fragmentation proxy, precision by cluster size, and the camera-position inversion.

Split out of scripts/eval_ps_clustering.py (#133, from the #105 review's N6) so CI covers
it: nothing here needs pandas, scipy or `haversine` -- the analysis-only packages that
make eval_ps_clustering's tests skip in CI. Allowed imports are the stdlib, numpy, and
the pipeline's own geo / fuse_sites / eval_sites (plus position_check, loaded by path
when live_positions_as_of needs it). eval_ps_clustering re-exports every name below, so
`epc.score`, `epc.Cluster`, ... keep working for every caller.

The two spatial prefilters that used scipy's cKDTree (score's "which clusters are within
reach of a GT ramp" and near_cluster_rate's "which clusters have a neighbour within r")
are a plain grid bucketing here, with cKDTree's own test (squared distance <= r * r), so
they return the same sets; tests/test_clustering_metrics.py checks both against a brute
force, and tests/test_eval_ps_clustering_server.py against cKDTree where scipy is present.

Example:
    >>> c = Cluster(0, [('p', 0)], 1)
    >>> place([c], {('p', 0): (3.0, 4.0)})[0].e
    3.0
"""
import importlib.util
import math
import sys
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


# ------------------------------------------------------------- grid neighbour search

# A grid cell is a hair wider than the search radius, so two points within r of each
# other are never more than one cell apart even after the division rounds.
_CELL_PAD = 1.0 + 1e-6


def _grid(points, size):
    """{(cell_x, cell_y): [index, ...]} for (x, y) points on a square grid of `size`."""
    cells = {}
    for i, (x, y) in enumerate(points):
        cells.setdefault((math.floor(x / size), math.floor(y / size)), []).append(i)
    return cells


def _cell_size(r):
    return r * _CELL_PAD if r > 0 else 1.0


def within_reach(points, centres, r):
    """Indices of `points` ((x, y) pairs) lying within r of at least one of `centres`.

    The test is cKDTree's own (dx * dx + dy * dy <= r * r), so this returns exactly the set
    scipy's `cKDTree(points).query_ball_point(centres, r)` unions to, without scipy.

    Example:
        >>> sorted(within_reach([(0.0, 0.0), (5.0, 0.0), (9.0, 0.0)], [(0.0, 0.0)], 5.0))
        [0, 1]
    """
    size, r2 = _cell_size(r), r * r
    cells = _grid(points, size)
    hit = set()
    for cx, cy in centres:
        gx, gy = math.floor(cx / size), math.floor(cy / size)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for i in cells.get((gx + ox, gy + oy), ()):
                    if i in hit:
                        continue
                    dx, dy = points[i][0] - cx, points[i][1] - cy
                    if dx * dx + dy * dy <= r2:
                        hit.add(i)
    return hit


def with_neighbour(points, r):
    """Indices of `points` that have ANOTHER point within r (squared distance <= r * r):
    the union of both ends of scipy's `cKDTree(points).query_pairs(r)`, without scipy.

    Example:
        >>> sorted(with_neighbour([(0.0, 0.0), (4.0, 0.0), (30.0, 0.0)], 5.0))
        [0, 1]
    """
    size, r2 = _cell_size(r), r * r
    cells = _grid(points, size)
    out = set()
    for (gx, gy), members in cells.items():
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                others = cells.get((gx + ox, gy + oy))
                if not others:
                    continue
                for i in members:
                    if i in out:
                        continue
                    xi, yi = points[i]
                    for j in others:
                        if j == i:
                            continue
                        dx, dy = points[j][0] - xi, points[j][1] - yi
                        if dx * dx + dy * dy <= r2:
                            out.add(i)
                            break
    return out


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


def partition_agreement(a, b):
    """How much of partition a (list of Cluster) is reproduced by b: fraction of a's
    clusters whose label set appears verbatim in b, plus labels in disagreeing ones."""
    sets_b = {frozenset(c.label_ids) for c in b}
    same = [c for c in a if frozenset(c.label_ids) in sets_b]
    off_labels = sum(c.n_labels for c in a if frozenset(c.label_ids) not in sets_b)
    return len(same), len(a), off_labels


# ------------------------------------------------- fusion on what the server holds

# The server's own placement height (SidewalkWebpage#4819): its label lat/lng are the
# flat raycast at this height, so inverting that raycast recovers the camera position.
SERVER_CAMERA_HEIGHT_M = 2.341219672825709
# Only labels this close are inverted: nearer the horizon the server's bounded tail
# departs from the flat raycast, and one such label would move a camera metres.
INVERT_MAX_RANGE_M = 15.0
# det_index for a human label. Stored detections are capped at 50 per pano, so this
# never collides with a real index, and det_pos never holds it (scored by n_labels only).
HUMAN_DET_BASE = 10000
HUMAN_CONFIDENCE = 1.0
# det_index for a second (third, ...) AI label at a pixel another AI label already took:
# a re-submitted campaign (Vancouver: 1,058 pairs) puts two live labels on one stored
# detection. Both are on the server, so both are fused (the same-pano cannot-link keeps
# them apart, as it does on the server), but only the first carries the run's det_index:
# the scorer keys on (pano_id, det_index), so the second is counted in n_labels and never
# scored twice. >= HUMAN_DET_BASE, so clusters_from_server_sites leaves it out of members.
DUPLICATE_DET_BASE = 20000
# det_index for an AI-account label that maps to no stored detection, admitted only when
# the caller passes `unmapped_confidence` (#56, `--ai-user`: Vancouver's deployed labels
# were placed on heatmap plateaus the store-rebuilt run resolves differently, so 22% of
# them have no pixel-exact twin). The server holds its confidence in label_ai_info, which
# the rawLabels feed does not carry, so such a label gets the tier: the lowest value the
# server can hold. Never placeable (no det_pos), like a human label, but counted as AI.
# Above DUPLICATE_DET_BASE + any per-pano duplicate count, so the two never collide.
AI_UNMAPPED_BASE = 30000


def invert_camera_position(rows, server_height=SERVER_CAMERA_HEIGHT_M):
    """(lat, lng, n_used) of one pano's camera, from its labels' server positions, or
    None when no label is close enough to invert.

    Each server position is the flat raycast from the camera, so the camera sits at the
    label minus that raycast's offset (the offset depends only on heading, pixel and
    height). The per-label estimates are combined by median.

    Example:
        A label straight ahead (x = 0.5, heading 0) and 20 degrees down at 2.34 m lies
        ~6.4 m north of the camera, so the camera comes back ~6.4 m south of the label.
    """
    lats, lngs = [], []
    for r in rows:
        if not r['pano_width'] or not r['pano_height'] or r['camera_heading'] is None:
            continue
        probe = fs.SlimPano(r['pano_id'], r['lat'], r['lng'], r['camera_heading'],
                            None, None, None, r['pano_source'] or '', [])
        g = geo.detection_ground_point(
            fs.pano_pose(probe, fs.POSE_OFF), r['pano_x'] / r['pano_width'],
            r['pano_y'] / r['pano_height'], camera_height=server_height,
            max_range_m=INVERT_MAX_RANGE_M, errors=geo.error_model_for(probe.source),
            apply_pose=False)
        if g is None:
            continue
        f = geo.LocalFrame(r['lat'], r['lng'])
        ge, gn = f.to_enu(g.lat, g.lng)
        lat, lng = f.to_latlng(-ge, -gn)
        lats.append(lat)
        lngs.append(lng)
    if not lats:
        return None
    return float(np.median(lats)), float(np.median(lngs)), len(lats)


def clusters_from_server_sites(sites, panos=(), unpositioned=(), label_of=None):
    """Clusters from a fusion_server fuse: every member counts toward n_labels, but only
    AI members (a real det_index) are placeable and scoreable, as in the other arms.

    A server has to put every label somewhere, but the raycast drops the ones it cannot
    place (beyond the range cap, or at/above the horizon), so no site holds them. Each
    label of `panos` that no site holds becomes a singleton cluster, and the number of
    them is the second return value. Each label id in `unpositioned` (labels on panos
    server_panos could not position at all) is one more singleton with no AI member, so
    every live label is in exactly one cluster; those are not in the second return value.
    `label_of` (server_panos' stats['label_of']) fills each cluster's label_ids."""
    label_of = label_of or {}
    out, held = [], set()
    for s in sites:
        keys = [(d.pano_id, d.det_index) for d, _ in s.members]
        held.update(keys)
        out.append(Cluster(s.id, [k for k in keys if k[1] < HUMAN_DET_BASE], len(keys),
                           label_ids=[label_of[k] for k in keys if k in label_of]))
    n_singletons = 0
    for p in panos:
        for i, *_rest in p.detections:
            key = (p.pano_id, i)
            if key not in held:
                out.append(Cluster(len(out), [key] if i < HUMAN_DET_BASE else [], 1,
                                   label_ids=[label_of[key]] if key in label_of else []))
                n_singletons += 1
    for lab in unpositioned:
        out.append(Cluster(len(out), [], 1, label_ids=[lab]))
    return out, n_singletons


# How far (m) a camera position may sit from the run's pano block before it counts as
# "somewhere else": the inverted_far check (an inverted pano) and the run-block fallback
# check (a pano positioned from the block, against its live position as of the pull).
LIVE_POSITION_TOL_M = 1.0


def _position_check():
    """The repo-root position_check module. Loaded by path, because scripts/ holds a shim
    of the same name that `import position_check` would find first from a script."""
    mod = sys.modules.get('position_check')
    if mod is not None and hasattr(mod, 'campaigns_for'):
        return mod
    spec = importlib.util.spec_from_file_location('position_check',
                                                  REPO_ROOT / 'position_check.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['position_check'] = mod
    spec.loader.exec_module(mod)
    return mod


def _utc(stamp):
    """An ISO 8601 timestamp ('...Z' or '+00:00') as an aware UTC datetime."""
    t = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def live_positions_as_of(directory, endpoint, as_of):
    """Where each pano sat on `endpoint` at time `as_of`, per the submission records in
    `directory`: ({panorama_id: (lat, lng)}, [problems]).

    position_check.live_positions' rule -- newest campaign wins, per pano -- restricted to
    the campaigns whose `last_submission_utc` is at or before `as_of` (a pull's
    `fetched_at`), so it describes the server the pull saw, not the server today. A
    campaign with no timestamp counts as before any time (live_positions sorts it oldest).
    Two campaigns with the same timestamp that disagree on a pano are a problem, as there.
    position_check.campaigns_for's own problems (a record whose results file or partial
    sidecar is missing) are passed through unfiltered: they carry no timestamp, so the
    caller has to treat any problem as "cannot tell".

    Example:
        A pano sent at SfM on 09-05 and re-sent at raw GPS on 09-24 is at SfM as of a
        09-21 pull, and at raw as of a 09-28 one.
    """
    pc = _position_check()
    campaigns, problems = pc.campaigns_for(directory, endpoint)
    cutoff = _utc(as_of)
    kept = [c for c in campaigns if not c['at'] or _utc(c['at']) <= cutoff]
    live, at, tied = {}, {}, set()
    for c in sorted(kept, key=lambda c: _utc(c['at']) if c['at']
                    else datetime.min.replace(tzinfo=timezone.utc)):
        lines = c['line_numbers']
        for pid, ll in pc.pano_positions_by_id(c['path'], lines).items():
            if pid in live and at[pid] == c['at'] and not pc.same_position(live[pid], ll):
                tied.add(pid)
            elif pid in live and at[pid] != c['at']:
                tied.discard(pid)       # strictly newer: an earlier tie is superseded
            live[pid], at[pid] = ll, c['at']
    if tied:
        problems = problems + [f'{len(tied)} pano(s) were sent at different coordinates by '
                               'campaigns recorded at the same time']
    return live, problems


def fallback_moved(pano_ids, run_positions, live, tol_m=LIVE_POSITION_TOL_M):
    """(moved, unknown): of the panos positioned from the run's block (`pano_ids`), those
    whose live position (`live`, from live_positions_as_of) is more than tol_m from that
    block (`run_positions`: {pano_id: (lat, lng)}), and those no campaign had sent as of
    the pull, whose live position this cannot tell.

    Example:
        >>> fallback_moved(['a', 'b'], {'a': (0.0, 0.0), 'b': (0.0, 0.0)},
        ...                {'a': (0.0, 0.0001)})
        (['a'], ['b'])
    """
    moved, unknown = [], []
    for pid in pano_ids:
        if pid not in live:
            unknown.append(pid)
        elif geo.haversine_m(*live[pid], *run_positions[pid]) > tol_m:
            moved.append(pid)
    return moved, unknown


# ------------------------------------------------------- human votes (#56 (c))

def human_votes(validations):
    """(agree, disagree, unsure) counts over a rawLabels row's `validations`, HUMAN
    validators only. PS's own AI validator also votes there (`validator_type: AI`) and
    dominates the feed's `correct` in some cities, so it is left out, as agree_rate.py
    does; `correct` itself is kept on the row and reported apart.

    Example:
        >>> human_votes([{'validation': 'Agree', 'validator_type': 'Human'},
        ...              {'validation': 'Disagree', 'validator_type': 'AI'}])
        (1, 0, 0)
    """
    a = d = u = 0
    for v in validations or ():
        if v.get('validator_type') != 'Human':
            continue
        kind = v.get('validation')
        a += kind == 'Agree'
        d += kind == 'Disagree'
        u += kind == 'Unsure'
    return a, d, u



# ------------------------------------------------------------------------- scoring

def score(clusters, gt, det_pos, radius_m=5.0, frag_radii=(3.0, 5.0)):
    ramps, pool, points, op_verdicts = gt
    placed = [c for c in clusters if c.e is not None]
    # Only clusters within reach of some GT ramp can match or count as a fragment, so the
    # O(ramps x clusters) loops below run over those alone (identical results; a city of
    # 100k clusters otherwise costs minutes per arm).
    reach = max(radius_m, *frag_radii)
    near = placed
    if placed and ramps:
        hit = within_reach([(c.e, c.n) for c in placed], [(r.e, r.n) for r in ramps], reach)
        near = [placed[i] for i in sorted(hit)]
    matched = es.match_one_to_one(pool, near, radius_m)
    # A cluster is "somebody's match" if it matches ANY GT ramp, not only a pool
    # one: a ramp on a pano whose missed-check was not confirmed is still a real
    # ramp, and a cluster sitting on it is not a fragment.
    matched_all = es.match_one_to_one(ramps, near, radius_m)
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
            extra = sum(1 for o in near if o.id not in matched_ids
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
            'coh_over_5m': co['over_5m'], 'same_pano_pairs': r['same_pano_pairs'],
            **size_columns(r.get('size')), **near_columns(r.get('near')),
            **near_columns(r.get('near_placed'), 'nearp')}


def near_columns(near, prefix='near'):
    """arms.csv columns for near_cluster_rate: near5 / near7_5 / near12_5 (share of placed
    clusters with another cluster of the arm within r), near_n, near_frame,
    per_1000_labels; the same under `nearp` for the placeable-member subset."""
    if not near:
        return {}
    out = {f'{prefix}{r:g}'.replace('.', '_'): near['near'].get(r) for r in NEAR_RADII_M}
    out.update({f'{prefix}_n': near['n_pos'], f'{prefix}_frame': near['frame']})
    if prefix == 'near':
        out['per_1000_labels'] = near['per_1000_labels']
    return out


SIZE_KEYS = {'unplaceable': 'sz_unpl', 'unclustered': 'sz_uncl', 'cluster of 1': 'sz_1',
             'cluster of 2': 'sz_2',
             'cluster of 3+': 'sz_3p'}


def size_columns(rows):
    """arms.csv columns for size_precision rows: n / judged / t / f per bucket, so a pooled
    table can sum numerators and denominators across cities."""
    out = {}
    for row in rows or ():
        k = SIZE_KEYS[row['bucket']]
        out.update({f'{k}_n': row['n'], f'{k}_judged': row['judged'],
                    f'{k}_t': row['t'], f'{k}_f': row['f']})
    return out


SIZE_BUCKETS = ('unplaceable', 'unclustered', 'cluster of 1', 'cluster of 2',
                'cluster of 3+')


def size_precision(clusters, det_of, det_pos, verdicts, conf, ys=None):
    """Rows (one per SIZE_BUCKETS entry) of GT precision by the size of the cluster
    holding each AI label. A label the raycast cannot place (not in det_pos) is
    `unplaceable` whatever partition holds it; a placeable one that no cluster of the
    partition holds is `unclustered` (the server's partition can leave a label out), never
    counted as a cluster of 1. `verdicts` is build_gt's (pano_id, det_index) -> verdict
    map; only True and False count toward precision. `ys` (key -> y_normalized) gives each
    row's median_y, where 0.5 is the horizon.

    Example:
        A label alone in its cluster, judged False, lands in 'cluster of 1' with
        t=0, f=1; one on a pano nobody judged adds to n but not to `judged`.
    """
    size_of = {}
    for c in clusters:
        for m in c.members:
            size_of[m] = c.n_labels
    keys = {b: [] for b in SIZE_BUCKETS}
    for key in det_of.values():
        if key not in det_pos:
            keys['unplaceable'].append(key)
        else:
            s = size_of.get(key)
            keys['unclustered' if s is None else 'cluster of 1' if s == 1
                 else 'cluster of 2' if s == 2 else 'cluster of 3+'].append(key)
    rows = []
    for b in SIZE_BUCKETS:
        ks = keys[b]
        v = [verdicts[k] for k in ks if k in verdicts]
        cs = sorted(conf[k] for k in ks if k in conf)
        yv = sorted(ys[k] for k in ks if ys and k in ys)
        rows.append({'bucket': b, 'n': len(ks), 'judged': len(v),
                     't': sum(x is True for x in v), 'f': sum(x is False for x in v),
                     'median_conf': quantile(cs, .5), 'median_y': quantile(yv, .5)})
    return rows


# --------------------------------------------- GT-free metrics (issue #56, Vancouver)

NEAR_RADII_M = (5.0, 7.5, 12.5)


def cluster_positions(clusters, server_pos, frame):
    """(xy array, frame name) for near_cluster_rate: a cluster sits at the mean of its
    labels' SERVER positions (`server_pos`: {label_id: (lat, lng)}, projected into
    `frame`) when it carries label_ids with a position -- the server's own frame, which
    needs no run -- else at its raycast position (c.e, c.n), which the `fusion` arms
    have. Clusters with neither are left out. The name says which frame was used
    ('server', 'raycast' or 'mixed')."""
    xy, kinds = [], set()
    for c in clusters:
        pts = [server_pos[lab] for lab in c.label_ids if lab in server_pos]
        if pts:
            e_n = [frame.to_enu(lat, lng) for lat, lng in pts]
            xy.append((sum(p[0] for p in e_n) / len(e_n), sum(p[1] for p in e_n) / len(e_n)))
            kinds.add('server')
        elif c.e is not None:
            xy.append((c.e, c.n))
            kinds.add('raycast')
    name = kinds.pop() if len(kinds) == 1 else 'mixed' if kinds else 'none'
    return np.array(xy).reshape(-1, 2), name


def near_cluster_rate(clusters, server_pos, frame, radii=NEAR_RADII_M):
    """The GT-free fragmentation proxy pre-registered for Vancouver (#56 metric (b)):
    the share of an arm's placed clusters with ANOTHER cluster of the same arm within r,
    for each r, plus clusters per 1,000 labels. Two clusters of one ramp read as a
    near pair; so do two real ramps of one corner, which is why it is a proxy and why
    the same arms are also scored against a city inventory.

    Returns {'near': {r: share}, 'n_pos': placed clusters, 'frame': cluster_positions'
    frame name, 'per_1000_labels': clusters per 1,000 labels of the arm}.

    Example:
        Three clusters at 0, 4 and 30 m along a line: 2 of 3 have a neighbour within
        5 m (share 0.667), none has one within 5 m beyond those two.
    """
    xy, frame_name = cluster_positions(clusters, server_pos, frame)
    near = {}
    if len(xy) >= 2:
        pts = [(float(x), float(y)) for x, y in xy]
        for r in radii:
            near[r] = len(with_neighbour(pts, r)) / len(xy)
    else:
        near = {r: None for r in radii}
    n_labels = sum(c.n_labels for c in clusters)
    return {'near': near, 'n_pos': int(len(xy)), 'frame': frame_name,
            'per_1000_labels': 1000.0 * len(clusters) / n_labels if n_labels else None}


VAL_BUCKETS = ('cluster of 1', 'cluster of 2', 'cluster of 3+')


def validation_precision(clusters, verdicts):
    """#56 metric (c): validation-based precision by cluster size, from HUMAN votes.

    `verdicts` is label_verdicts(): label_id -> True / False / None. For each size
    bucket (labels in the cluster): clusters holding >= 1 validated label (`n`), of them
    the clusters holding >= 1 label the humans voted FALSE (`any_false`), and the labels
    behind them (`labels_validated`, `labels_false`). Only clusters with label_ids count
    (every server-label arm has them; the run-only `fusion` arms do not).

    Example:
        A cluster of two labels, one voted True and one False, is one `any_false`
        cluster of 1 validated in 'cluster of 2', with 2 validated labels, 1 false.
    """
    rows = {b: {'bucket': b, 'clusters': 0, 'n': 0, 'any_false': 0, 'all_false': 0,
                'labels_validated': 0, 'labels_false': 0} for b in VAL_BUCKETS}
    for c in clusters:
        if not c.label_ids:
            continue
        b = ('cluster of 1' if c.n_labels == 1 else 'cluster of 2' if c.n_labels == 2
             else 'cluster of 3+')
        row = rows[b]
        row['clusters'] += 1
        vs = [verdicts[lab] for lab in c.label_ids if lab in verdicts]
        if not vs:
            continue
        row['n'] += 1
        row['labels_validated'] += len(vs)
        row['labels_false'] += sum(v is False for v in vs)
        row['any_false'] += any(v is False for v in vs)
        row['all_false'] += all(v is False for v in vs)
    return [rows[b] for b in VAL_BUCKETS]


def wilson(t, n, z=1.96):
    """Wilson score interval for t successes in n trials; None when n == 0.

    Example:
        >>> [round(x, 2) for x in wilson(27, 27)]
        [0.88, 1.0]
    """
    if n == 0:
        return None
    p = t / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def precision_ci_text(t, f):
    ci = wilson(t, t + f)
    if ci is None:
        return 'n/a'
    return f'{t / (t + f):.3f} [{ci[0]:.2f}, {ci[1]:.2f}]'


def quantile(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
