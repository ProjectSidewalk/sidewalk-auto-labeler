"""Bearing-only placement of multi-view sites vs the flat-raycast GLS (overnight study).

Production places a fused site by a GLS refit of each view's flat-ground raycast at an
assumed camera height (2.6 m for Mapillary/Panoramax, which serve no depth; #53/#89), so a
site's position scales with a height nobody has validated. A site seen from >= 2 camera
positions can instead be placed by intersecting its horizontal rays: camera position plus
world bearing, heading + (x - 0.5) * 360, which no camera height enters. This script
compares the two on ONE site set and scores them where the height cannot bias the
comparison: the azimuth at which each arm's leave-pano-out position lands in a judged
benchmark pano, against the reviewer's box centre (or the confirmed peak).

Arms (association frozen from fuse_sites.fuse at 2.6 m, tier 0.55, apply_pose off,
production rig mask; a view = an operational refit member; sites with >= 3 views):
  A  GLS of the views' flat 2.6 m raycasts (reprojection_residual.solve_views)
  B  IRLS weighted least-squares intersection of the views' horizontal rays
     (bearing_solve); weights 1 / (r^2 sigma_theta^2 + sigma_gps^2) from the source's
     geo.ErrorModel, r the range to the current estimate; nothing reads a height.

Outputs per city (docs/overnight/data/bearing_refit/<city>/): refs.csv (one row per
GT reference: leave-pano-out azimuth and row error per arm, parallax, solvability),
gtfree.csv (one row per held-out view: GT-free azimuth residual per arm), stress.csv (one
row per site: A shift under a frozen re-raycast at 2.0 / 3.2 m, A and B shifts under
re-association at those heights), keeps.csv (site counts by spread threshold). The
`summarize` step reads those, applies the pre-registered rules
(docs/overnight/bearing_refit.md) with seeded site-cluster bootstraps, and writes
summary CSVs + a markdown fragment.

No network, no GPU. Deterministic (fixed seed, canonical sort inside fuse_sites).

Usage:
    PY=python
    RUNS=runs
    BENCH=../RampNet/benchmark
    nice $PY scripts/bearing_refit.py run richmond --run-root $RUNS --benchmark-root $BENCH
    nice $PY scripts/bearing_refit.py run laurens  --run-root $RUNS --benchmark-root $BENCH
    nice $PY scripts/bearing_refit.py run bayonne  --run-root $RUNS --benchmark-root $BENCH
    nice $PY scripts/bearing_refit.py run paterson --run-root $RUNS --benchmark-root $BENCH
    $PY scripts/bearing_refit.py summarize      # -> summary_*.csv + summary.md
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import reprojection_residual as rr  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

# city -> (benchmark split, results file). laurens' Mapillary run is runs/laurens.
CITIES = {'richmond': ('richmond', 'results.jsonl'),
          'laurens': ('laurens_mapillary', 'results.jsonl'),
          'bayonne': ('bayonne', 'results.jsonl'),
          'paterson': ('paterson', 'results.jsonl')}
OPEN_CITIES = ('richmond', 'laurens', 'bayonne')
H0 = 2.6
STRESS_HEIGHTS = (2.0, 3.2)
MIN_VIEWS = 3
SPREADS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
PRIMARY_SPREAD = 10.0
BASELINE_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 1e9))
PARALLAX_BINS = ((10.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 90.01))
B_RANGE = fs.IMPLIED_RANGE_M          # (1, 30) m, the --implied-height pairing bounds
IRLS_ITERS = 5
SEED = 20261126
N_BOOT = 2000
RULE1_MIN_DIFF_DEG = 0.25
RULE2_MAX_RATIO = 0.25
OUT_ROOT = REPO_ROOT / 'docs' / 'overnight' / 'data' / 'bearing_refit'


# --- Bearing-only geometry (pure; unit-tested) -------------------------------------------

def spread_deg(bearings):
    """Ray-angle spread of a view set: the largest pairwise intersection angle, folded to
    [0, 90] deg (two rays 170 deg apart constrain a point as badly as two 10 deg apart).

    >>> round(spread_deg([0.0, 30.0, 170.0]), 6)
    40.0
    >>> spread_deg([45.0])
    0.0
    """
    best = 0.0
    for i in range(len(bearings)):
        for j in range(i + 1, len(bearings)):
            a = abs(math.sin(math.radians(bearings[i] - bearings[j])))
            best = max(best, math.degrees(math.asin(min(1.0, a))))
    return best


def bearing_solve(rays, iters=IRLS_ITERS):
    """Weighted LS intersection of horizontal rays. `rays` = [(ce, cn, bearing_deg,
    sigma_theta2, sigma_gps2)]. Minimises sum w_i (n_i . (X - c_i))^2, n_i the ray's unit
    normal, with IRLS weights 1 / (r_i^2 sigma_theta2 + sigma_gps2) and r_i the range along
    ray i to the current estimate (unweighted start). Returns ((e, n), [r_i]) or None when
    the normal matrix is singular (parallel rays).

    >>> (e, n), r = bearing_solve([(0, 0, 45.0, 1e-4, 1.0), (10, 0, 315.0, 1e-4, 1.0)])
    >>> round(e, 6), round(n, 6), [round(x, 6) for x in r]
    (5.0, 5.0, [7.071068, 7.071068])
    """
    w = [1.0] * len(rays)
    sol = None
    for it in range(iters + 1):
        a11 = a12 = a22 = b1 = b2 = 0.0
        for wi, (ce, cn, b, _s2t, _s2g) in zip(w, rays):
            br = math.radians(b)
            nx, ny = math.cos(br), -math.sin(br)        # normal to u = (sin b, cos b)
            d = nx * ce + ny * cn
            a11 += wi * nx * nx
            a12 += wi * nx * ny
            a22 += wi * ny * ny
            b1 += wi * nx * d
            b2 += wi * ny * d
        det = a11 * a22 - a12 * a12
        tr = a11 + a22
        if tr <= 0 or det <= 1e-10 * tr * tr:
            return None
        xe = (a22 * b1 - a12 * b2) / det
        xn = (a11 * b2 - a12 * b1) / det
        ranges = []
        for ce, cn, b, _s2t, _s2g in rays:
            br = math.radians(b)
            ranges.append(math.sin(br) * (xe - ce) + math.cos(br) * (xn - cn))
        sol = ((xe, xn), ranges)
        if it < iters:
            w = [1.0 / (max(r, 1.0) ** 2 * s2t + s2g)
                 for r, (_ce, _cn, _b, s2t, s2g) in zip(ranges, rays)]
    return sol


def b_solution(rays, min_spread=0.0, range_bounds=B_RANGE):
    """(position, spread) when B can place this view set, else (None, spread): >= 2 rays,
    spread >= min_spread, a non-singular solve, and every range inside range_bounds."""
    sp = spread_deg([r[2] for r in rays])
    if len(rays) < 2 or sp < min_spread:
        return None, sp
    sol = bearing_solve(rays)
    if sol is None:
        return None, sp
    pos, ranges = sol
    lo, hi = range_bounds
    if not all(lo < r < hi for r in ranges):
        return None, sp
    return pos, sp


def azimuth_error(cam, pos, ref_bearing_deg):
    """Wrapped (predicted - reference) bearing in deg, from camera `cam` to `pos`.

    >>> round(azimuth_error((0, 0), (1, 1), 40.0), 6)
    5.0
    """
    pred = math.degrees(math.atan2(pos[0] - cam[0], pos[1] - cam[1]))
    return geo.norm_deg(pred - ref_bearing_deg)


def row_error_px(cam, pos, ref_y, camera_height=H0):
    """(reference - predicted) row in heatmap px at an assumed camera height (flat)."""
    d = math.hypot(pos[0] - cam[0], pos[1] - cam[1])
    y_pred = 0.5 + math.atan2(camera_height, d) / math.pi
    return (ref_y - y_pred) * rr.HEATMAP_H


# --- Run-side machinery ------------------------------------------------------------------

def fuse_params(h):
    return fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=True,
                         camera_height_m=h, apply_pose=fs.POSE_OFF)


class Ctx:
    """Per-run lookups: camera ENU, poses, ray tuples."""

    def __init__(self, panos, frame):
        self.frame = frame
        self.by_id = {p.pano_id: p for p in panos}
        self._cam, self._pose = {}, {}

    def cam(self, pid):
        if pid not in self._cam:
            p = self.by_id[pid]
            self._cam[pid] = self.frame.to_enu(p.lat, p.lng)
        return self._cam[pid]

    def pose(self, pid):
        if pid not in self._pose:
            self._pose[pid] = fs.pano_pose(self.by_id[pid], fs.POSE_OFF)
        return self._pose[pid]

    def ray(self, v):
        """B's ray for a View: camera, the height-free bearing, the error-model terms."""
        p = self.by_id[v.pano_id]
        err = geo.error_model_for(p.source)
        s2t = (err.sigma_peak_px * geo.RAD_PER_HEATMAP_PX) ** 2 + err.sigma_heading_rad ** 2
        ce, cn = self.cam(v.pano_id)
        bearing = (p.camera_heading + (v.x - 0.5) * 360.0) % 360.0
        return (ce, cn, bearing, s2t, err.sigma_gps_m ** 2)

    def reraycast(self, views, h):
        """The same views as rr.View objects raycast at height h (uncapped)."""
        out = []
        for v in views:
            p = self.by_id[v.pano_id]
            err = geo.error_model_for(p.source)
            g = geo.detection_ground_point(self.pose(v.pano_id), v.x, v.y, camera_height=h,
                                           max_range_m=math.inf, errors=err,
                                           apply_pose=False)
            if g is None:
                return None
            e, n = self.frame.to_enu(g.lat, g.lng)
            out.append(rr.View(v.pano_id, v.det_index, v.x, v.y, v.conf, e, n,
                               g.cov_en(err.sigma_gps_m), g.range_m, g.bearing_deg,
                               v.source, v.capture_date))
        return out


def _r(v, nd=4):
    return None if v is None else round(v, nd)


def ref_rows(city, split_data, panos, sites, ctx):
    """One row per GT reference mapped to a site with >= MIN_VIEWS views."""
    verdicts, bundle_ops, boxes = split_data
    counts = es.gt_counts()
    counts.update(refs_det=0, det_no_site=0, det_small_site=0, refs_missed=0,
                  missed_unplaceable=0, missed_no_site=0, missed_small_site=0)
    warnings = []
    site_of = {}
    for sv in sites:
        for v in sv.views:
            site_of[(v.pano_id, v.det_index)] = sv
    op_sites = [sv for sv in sites if sv.views]
    grid = geo.GridIndex(rr.MATCH_RADIUS_M)
    for sv in op_sites:
        grid.add(sv.e, sv.n, sv)
    es.require_bundle_frame(ctx.by_id.values(), city)
    rows = []

    def emit(kind, pid, ref_x, ref_y, sv, member):
        pano = ctx.by_id[pid]
        cam = ctx.cam(pid)
        ref_b = (pano.camera_heading + (ref_x - 0.5) * 360.0) % 360.0
        use = [v for v in sv.views if v.pano_id != pid] if member else list(sv.views)
        pos_a = rr.solve_views(use)
        pos_b, sp = b_solution([ctx.ray(v) for v in use])
        rows.append({
            'city': city, 'pano_id': pid, 'site_id': sv.site_id, 'ref_kind': kind,
            'member': member, 'n_views_site': len(sv.views), 'n_views_used': len(use),
            'spread_deg': _r(sp, 3), 'b_ok': pos_b is not None,
            **geometry_cols(use, ctx),
            'az_a': _r(azimuth_error(cam, pos_a, ref_b)),
            'az_b': None if pos_b is None else _r(azimuth_error(cam, pos_b, ref_b)),
            'row_a_px': _r(row_error_px(cam, pos_a, ref_y)),
            'row_b_px': None if pos_b is None else _r(row_error_px(cam, pos_b, ref_y)),
            'range_a_m': _r(math.hypot(pos_a[0] - cam[0], pos_a[1] - cam[1]), 3),
            'range_b_m': None if pos_b is None
            else _r(math.hypot(pos_b[0] - cam[0], pos_b[1] - cam[1]), 3),
            'ab_dist_m': None if pos_b is None
            else _r(math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]), 3)})

    for pid, entry, run_pano, ops, _pool in es.judged_gt_panos(
            verdicts, bundle_ops, ctx.by_id, counts, warnings):
        used = set()
        for k, (verdict, (stored_i, x, y, _c)) in enumerate(zip(entry['dets'], ops)):
            if verdict is not True:
                continue
            counts['refs_det'] += 1
            sv = site_of.get((pid, stored_i))
            if sv is None:
                counts['det_no_site'] += 1
                continue
            used.add(sv.site_id)
            if len(sv.views) < MIN_VIEWS:
                counts['det_small_site'] += 1
                continue
            box = rr._box_for(boxes, pid, f'det:{k}', x, y, warnings)
            if box is not None:
                emit('box', pid, box[0], box[1], sv, True)
            else:
                emit('peak', pid, x, y, sv, True)
        # Missed marks: rr's world-space match at 2.6 m (A's frame), one to one per pano.
        pose = ctx.pose(pid)
        errors = geo.error_model_for(run_pano.source)
        cands = []
        for j, mark in enumerate(entry.get('missed', ())):
            if mark.get('unsure'):
                continue
            counts['refs_missed'] += 1
            box = rr._box_for(boxes, pid, f'missed:{j}', mark['x'], mark['y'], warnings)
            rx, ry = box if box is not None else (mark['x'], mark['y'])
            g = geo.detection_ground_point(pose, mark['x'], mark['y'], camera_height=H0,
                                           max_range_m=geo.DEFAULT_MAX_RANGE_M,
                                           errors=errors, apply_pose=False)
            if g is None:
                counts['missed_unplaceable'] += 1
                continue
            me, mn = ctx.frame.to_enu(g.lat, g.lng)
            found = False
            for sv in grid.near(me, mn):
                if sv.site_id in used or any(v.pano_id == pid for v in sv.views):
                    continue
                d = math.hypot(sv.e - me, sv.n - mn)
                if d <= rr.MATCH_RADIUS_M:
                    cands.append((d, j, sv.site_id, sv, 'missed_box' if box else 'missed',
                                  rx, ry))
                    found = True
            if not found:
                counts['missed_no_site'] += 1
        cands.sort(key=lambda c: c[:3])
        done = set()
        for d, j, sid, sv, kind, rx, ry in cands:
            if j in done or sid in used:
                continue
            done.add(j)
            used.add(sid)
            if len(sv.views) < MIN_VIEWS:
                counts['missed_small_site'] += 1
                continue
            emit(kind, pid, rx, ry, sv, False)
        counts['missed_no_site'] += len({c[1] for c in cands} - done)
    return rows, counts, warnings


def gtfree_rows(city, sites, ctx):
    """One row per (site with >= MIN_VIEWS views, held-out view): azimuth residual of the
    held-out view's own detection against each arm's position solved without it."""
    rows = []
    for sv in sites:
        if len(sv.views) < MIN_VIEWS:
            continue
        rays = [ctx.ray(v) for v in sv.views]
        for i, v in enumerate(sv.views):
            others = sv.views[:i] + sv.views[i + 1:]
            pos_a = rr.solve_views(others)
            pos_b, sp = b_solution(rays[:i] + rays[i + 1:])
            cam = ctx.cam(v.pano_id)
            b_i = rays[i][2]
            rows.append({'city': city, 'site_id': sv.site_id, 'pano_id': v.pano_id,
                         'n_views': len(sv.views), 'spread_deg': _r(sp, 3),
                         'b_ok': pos_b is not None, **geometry_cols(others, ctx),
                         'az_a': _r(azimuth_error(cam, pos_a, b_i)),
                         'az_b': None if pos_b is None
                         else _r(azimuth_error(cam, pos_b, b_i))})
    return rows


def geometry_cols(views, ctx):
    """Exploratory (added after the pre-registered tables were read): the camera baseline
    of a view set (max pairwise camera distance, m) and its number of capture sequences
    (None for sources without one). B's range comes from the baseline, so camera-position
    noise comparable to it is what B cannot absorb."""
    cams = [ctx.cam(v.pano_id) for v in views]
    base = max((math.hypot(a[0] - b[0], a[1] - b[1])
                for i, a in enumerate(cams) for b in cams[i + 1:]), default=0.0)
    seqs = {ctx.by_id[v.pano_id].sequence_id for v in views}
    return {'baseline_m': _r(base, 3),
            'n_sequences': None if None in seqs else len(seqs)}


def member_keys(sv):
    return {(v.pano_id, v.det_index) for v in sv.views}


def stress_rows(city, panos, sites, ctx):
    """Per site with >= MIN_VIEWS views: A frozen at each wrong height; A and B on the
    site matched by view overlap after re-association at that height."""
    base = {}
    for sv in sites:
        if len(sv.views) < MIN_VIEWS:
            continue
        pos_b, sp = b_solution([ctx.ray(v) for v in sv.views])
        base[sv.site_id] = {'sv': sv, 'a': rr.solve_views(sv.views), 'b': pos_b,
                            'spread': sp}
    rows = {sid: {'city': city, 'site_id': sid, 'n_views': len(d['sv'].views),
                  'spread_deg': _r(d['spread'], 3), 'b_ok': d['b'] is not None}
            for sid, d in base.items()}
    for h in STRESS_HEIGHTS:
        tag = f'{h:g}'
        # A, association frozen: re-raycast the same views at h.
        for sid, d in base.items():
            views_h = ctx.reraycast(d['sv'].views, h)
            pos = None if views_h is None else rr.solve_views(views_h)
            rows[sid][f'a_frozen_shift_{tag}'] = None if pos is None else _r(
                math.hypot(pos[0] - d['a'][0], pos[1] - d['a'][1]), 4)
        # Re-association at h, in the SAME frame (fuse's frame is the pano centroid, which
        # does not depend on h, but check).
        sites_h, frame_h, _ = fs.fuse(panos, fuse_params(h))
        if (frame_h.lat0, frame_h.lng0) != (ctx.frame.lat0, ctx.frame.lng0):
            raise RuntimeError('re-association frame differs from the 2.6 m frame')
        svs_h = [rr.views_from_site(s) for s in sites_h]
        owner = {}
        for k, svh in enumerate(svs_h):
            for key in member_keys(svh):
                owner[key] = k
        for sid, d in base.items():
            tally = {}
            for key in member_keys(d['sv']):
                if key in owner:
                    tally[owner[key]] = tally.get(owner[key], 0) + 1
            best = max(tally.items(), key=lambda kv: (kv[1], -kv[0]), default=None)
            r = rows[sid]
            if best is None or best[1] < 2:
                r[f'match_{tag}'] = 'lost'
                r[f'a_reassoc_shift_{tag}'] = r[f'b_reassoc_shift_{tag}'] = None
                continue
            svh = svs_h[best[0]]
            r[f'match_{tag}'] = 'same' if member_keys(svh) == member_keys(d['sv']) \
                else 'changed'
            pa = rr.solve_views(svh.views)
            r[f'a_reassoc_shift_{tag}'] = _r(math.hypot(pa[0] - d['a'][0],
                                                        pa[1] - d['a'][1]), 4)
            pb, _ = b_solution([ctx.ray(v) for v in svh.views])
            r[f'b_reassoc_ok_{tag}'] = pb is not None
            r[f'b_reassoc_shift_{tag}'] = None if pb is None or d['b'] is None else _r(
                math.hypot(pb[0] - d['b'][0], pb[1] - d['b'][1]), 4)
    return list(rows.values())


def keep_rows(city, sites, ctx):
    multi = [sv for sv in sites if len(sv.views) >= MIN_VIEWS]
    out = []
    sols = [b_solution([ctx.ray(v) for v in sv.views]) for sv in multi]
    for thr in SPREADS:
        n_ok = sum(1 for pos, sp in sols if pos is not None and sp >= thr)
        n_spread = sum(1 for _pos, sp in sols if sp >= thr)
        out.append({'city': city, 'spread_min_deg': thr, 'sites_3plus': len(multi),
                    'sites_spread_ok': n_spread, 'sites_b_solvable': n_ok})
    return out


def run_city(city, run_root, benchmark_root):
    split, results = CITIES[city]
    run_dir = run_root / city
    panos, skipped = fs.load_results(run_dir / results, read_heights=False)
    print(f'{city}: {len(panos)} panos ({skipped} without position/heading)', flush=True)
    sites_raw, frame, stats = fs.fuse(panos, fuse_params(H0))
    sites = [rr.views_from_site(s) for s in sites_raw]
    ctx = Ctx(panos, frame)
    print(f'{city}: {len(sites)} sites, '
          f'{sum(1 for s in sites if len(s.views) >= MIN_VIEWS)} with >= {MIN_VIEWS} views',
          flush=True)
    split_data = rr.load_split(benchmark_root, split)
    if split_data is None:
        raise SystemExit(f'{city}: no benchmark split at {benchmark_root / split}')
    refs, counts, warnings = ref_rows(city, split_data, panos, sites, ctx)
    print(f'{city}: {len(refs)} GT references; counts {counts}; {len(warnings)} warnings',
          flush=True)
    gtfree = gtfree_rows(city, sites, ctx)
    keeps = keep_rows(city, sites, ctx)
    stress = stress_rows(city, panos, sites, ctx)
    out = OUT_ROOT / city
    rr.write_csv(out / 'refs.csv', refs)
    rr.write_csv(out / 'gtfree.csv', gtfree)
    rr.write_csv(out / 'stress.csv', stress)
    rr.write_csv(out / 'keeps.csv', keeps)
    rr.write_csv(out / 'counts.csv', [{'city': city, 'split': split, 'results': results,
                                       'panos': len(panos), 'sites': len(sites),
                                       **counts, 'warnings': len(warnings)}])
    print(f'{city}: wrote {out}', flush=True)


# --- Summaries and the pre-registered rules ----------------------------------------------

def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fnum(v):
    return None if v in (None, '', 'None') else float(v)


def truthy(v):
    return v in ('True', 'true', '1', True)


def boot_median_diff(rows, ka, kb, seed=SEED, n_boot=N_BOOT):
    """(D, lo, hi): D = median|a| - median|b| over rows, with a 95% percentile bootstrap
    resampling clusters (site_id) with replacement."""
    if not rows:
        return None, None, None
    a = np.array([abs(r[ka]) for r in rows])
    b = np.array([abs(r[kb]) for r in rows])
    d = float(np.median(a) - np.median(b))
    clusters = {}
    for i, r in enumerate(rows):
        clusters.setdefault(r['site_id'], []).append(i)
    idx = [np.array(v) for _k, v in sorted(clusters.items())]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for t in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[p] for p in pick])
        stats[t] = np.median(a[sel]) - np.median(b[sel])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return d, float(lo), float(hi)


def med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def p90(vals):
    vals = [v for v in vals if v is not None]
    return float(np.percentile(vals, 90)) if vals else None


def load_units(path, thr):
    rows = []
    for r in read_csv(path):
        rr_ = dict(r)
        for k in ('az_a', 'az_b', 'row_a_px', 'row_b_px', 'spread_deg', 'ab_dist_m'):
            if k in rr_:
                rr_[k] = fnum(rr_[k])
        rr_['b_ok'] = truthy(r['b_ok'])
        rr_['site_id'] = int(r['site_id'])
        rows.append(rr_)
    unit = [r for r in rows if r['b_ok'] and r['spread_deg'] >= thr]
    lost = [r for r in rows if not (r['b_ok'] and r['spread_deg'] >= thr)]
    return rows, unit, lost


def fmt(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def summarize():
    cities = [c for c in CITIES if (OUT_ROOT / c / 'refs.csv').exists()]
    md = []
    head_rows, gtfree_out, bins_out, keeps_out, stress_out, surv_out, sub_out = \
        [], [], [], [], [], [], []
    for city in cities:
        allrefs, unit, lost = load_units(OUT_ROOT / city / 'refs.csv', PRIMARY_SPREAD)
        member_unit = [r for r in unit if truthy(r['member'])]
        member_lost = [r for r in lost if truthy(r['member'])]
        d, lo, hi = boot_median_diff(member_unit, 'az_a', 'az_b')
        passes = d is not None and d >= RULE1_MIN_DIFF_DEG and lo > 0
        head_rows.append({
            'city': city, 'n_refs': len(member_unit),
            'n_sites': len({r['site_id'] for r in member_unit}),
            'n_box': sum(1 for r in member_unit if r['ref_kind'] == 'box'),
            'med_az_a': med([abs(r['az_a']) for r in member_unit]),
            'med_az_b': med([abs(r['az_b']) for r in member_unit]),
            'p90_az_a': p90([abs(r['az_a']) for r in member_unit]),
            'p90_az_b': p90([abs(r['az_b']) for r in member_unit]),
            'D_deg': d, 'D_lo': lo, 'D_hi': hi, 'rule1_pass': passes,
            'med_row_a_px': med([abs(r['row_a_px']) for r in member_unit]),
            'med_row_b_px': med([abs(r['row_b_px']) for r in member_unit]),
            'med_ab_dist_m': med([r['ab_dist_m'] for r in member_unit])})
        # Subsets: box-only member refs; missed refs (A-frame selection); other thresholds.
        for label, subset in (
                ('box_member', [r for r in member_unit if r['ref_kind'] == 'box']),
                ('peak_member', [r for r in member_unit if r['ref_kind'] == 'peak']),
                ('missed', [r for r in unit if not truthy(r['member'])])):
            sd, slo, shi = boot_median_diff(subset, 'az_a', 'az_b')
            sub_out.append({'city': city, 'subset': label, 'spread_min_deg': PRIMARY_SPREAD,
                            'n': len(subset),
                            'med_az_a': med([abs(r['az_a']) for r in subset]),
                            'med_az_b': med([abs(r['az_b']) for r in subset]),
                            'D_deg': sd, 'D_lo': slo, 'D_hi': shi})
        for thr in SPREADS:
            _all, u2, _l2 = load_units(OUT_ROOT / city / 'refs.csv', thr)
            u2 = [r for r in u2 if truthy(r['member'])]
            sd, slo, shi = boot_median_diff(u2, 'az_a', 'az_b')
            sub_out.append({'city': city, 'subset': 'member', 'spread_min_deg': thr,
                            'n': len(u2), 'med_az_a': med([abs(r['az_a']) for r in u2]),
                            'med_az_b': med([abs(r['az_b']) for r in u2]),
                            'D_deg': sd, 'D_lo': slo, 'D_hi': shi})
        for lo_b, hi_b in PARALLAX_BINS:
            sub = [r for r in member_unit if lo_b <= r['spread_deg'] < hi_b]
            bins_out.append({'city': city, 'parallax_bin': f'[{lo_b:g},{min(hi_b, 90):g})',
                             'n': len(sub), 'med_az_a': med([abs(r['az_a']) for r in sub]),
                             'med_az_b': med([abs(r['az_b']) for r in sub])})
        surv_out.append({
            'city': city, 'member_refs_total': len([r for r in allrefs
                                                    if truthy(r['member'])]),
            'unit': len(member_unit), 'b_unsolvable': len(member_lost),
            'unsolvable_low_spread': sum(1 for r in member_lost
                                         if r['spread_deg'] < PRIMARY_SPREAD),
            'unsolvable_bad_range': sum(1 for r in member_lost
                                        if r['spread_deg'] >= PRIMARY_SPREAD),
            'med_az_a_on_unsolvable': med([abs(r['az_a']) for r in member_lost]),
            'p90_az_a_on_unsolvable': p90([abs(r['az_a']) for r in member_lost]),
            'med_az_a_on_unit': med([abs(r['az_a']) for r in member_unit])})
        # GT-free LOO.
        _g, gunit, glost = load_units(OUT_ROOT / city / 'gtfree.csv', PRIMARY_SPREAD)
        gd, glo, ghi = boot_median_diff(gunit, 'az_a', 'az_b')
        gtfree_out.append({'city': city, 'n_views': len(gunit),
                           'n_sites': len({r['site_id'] for r in gunit}),
                           'n_unsolvable': len(glost),
                           'med_az_a': med([abs(r['az_a']) for r in gunit]),
                           'med_az_b': med([abs(r['az_b']) for r in gunit]),
                           'p90_az_a': p90([abs(r['az_a']) for r in gunit]),
                           'p90_az_b': p90([abs(r['az_b']) for r in gunit]),
                           'D_deg': gd, 'D_lo': glo, 'D_hi': ghi,
                           'med_az_a_on_unsolvable': med([abs(r['az_a']) for r in glost])})
        keeps_out += read_csv(OUT_ROOT / city / 'keeps.csv')
        # Stress.
        srows = read_csv(OUT_ROOT / city / 'stress.csv')
        for h in STRESS_HEIGHTS:
            tag = f'{h:g}'
            base_ok = [r for r in srows if truthy(r['b_ok'])
                       and fnum(r['spread_deg']) >= PRIMARY_SPREAD]
            both = [r for r in base_ok if fnum(r.get(f'b_reassoc_shift_{tag}')) is not None
                    and fnum(r.get(f'a_frozen_shift_{tag}')) is not None]
            ma = med([fnum(r[f'a_frozen_shift_{tag}']) for r in both])
            mb = med([fnum(r[f'b_reassoc_shift_{tag}']) for r in both])
            stress_out.append({
                'city': city, 'height_m': h, 'sites_b_solvable_2.6': len(base_ok),
                'sites_both_defined': len(both),
                'lost_match': sum(1 for r in base_ok if r.get(f'match_{tag}') == 'lost'),
                'changed_membership': sum(1 for r in base_ok
                                          if r.get(f'match_{tag}') == 'changed'),
                'b_unsolvable_after': sum(1 for r in base_ok
                                          if r.get(f'match_{tag}') not in ('lost', None)
                                          and not truthy(r.get(f'b_reassoc_ok_{tag}'))),
                'med_shift_a_frozen_m': ma,
                'p90_shift_a_frozen_m': p90([fnum(r[f'a_frozen_shift_{tag}']) for r in both]),
                'med_shift_a_reassoc_m': med([fnum(r[f'a_reassoc_shift_{tag}'])
                                              for r in both]),
                'med_shift_b_reassoc_m': mb,
                'p90_shift_b_reassoc_m': p90([fnum(r[f'b_reassoc_shift_{tag}'])
                                              for r in both]),
                'share_b_zero': (sum(1 for r in both
                                     if fnum(r[f'b_reassoc_shift_{tag}']) < 1e-6)
                                 / len(both)) if both else None,
                'ratio_b_over_a': None if not ma else mb / ma,
                'rule2_pass': bool(ma) and mb < RULE2_MAX_RATIO * ma})
    # EXPLORATORY (post hoc, after the pre-registered tables): by camera baseline and by
    # number of capture sequences, GT-free units and member refs at the primary spread.
    explore = []
    for city in cities:
        for kind, fname in (('gtfree', 'gtfree.csv'), ('refs_member', 'refs.csv')):
            _a, unit, _l = load_units(OUT_ROOT / city / fname, PRIMARY_SPREAD)
            if kind == 'refs_member':
                unit = [r for r in unit if truthy(r['member'])]
            groups = [(f'baseline [{lo:g},{hi:g}) m',
                       [r for r in unit if lo <= fnum(r['baseline_m']) < hi])
                      for lo, hi in BASELINE_BINS]
            groups += [('1 sequence', [r for r in unit if r['n_sequences'] == '1']),
                       ('>= 2 sequences', [r for r in unit if r['n_sequences']
                                           not in ('', '1', 'None')])]
            groups.append(('all', unit))
            for label, sub in groups:
                sd, slo, shi = boot_median_diff(sub, 'az_a', 'az_b')
                explore.append({
                    'city': city, 'units': kind, 'group': label, 'n': len(sub),
                    'med_az_a': med([abs(r['az_a']) for r in sub]),
                    'med_az_b': med([abs(r['az_b']) for r in sub]),
                    'share_b_better': (sum(1 for r in sub if abs(r['az_b']) < abs(r['az_a']))
                                       / len(sub)) if sub else None,
                    'D_deg': sd, 'D_lo': slo, 'D_hi': shi})
    out = OUT_ROOT
    rr.write_csv(out / 'summary_explore.csv', explore)
    rr.write_csv(out / 'summary_headline.csv', head_rows)
    rr.write_csv(out / 'summary_subsets.csv', sub_out)
    rr.write_csv(out / 'summary_parallax.csv', bins_out)
    rr.write_csv(out / 'summary_survivorship.csv', surv_out)
    rr.write_csv(out / 'summary_gtfree.csv', gtfree_out)
    rr.write_csv(out / 'summary_keeps.csv', keeps_out)
    rr.write_csv(out / 'summary_stress.csv', stress_out)

    votes = sum(1 for r in head_rows if r['city'] in OPEN_CITIES and r['rule1_pass'])
    open_stress = [r for r in stress_out if r['city'] in OPEN_CITIES]
    robust = bool(open_stress) and all(r['rule2_pass'] for r in open_stress) \
        and len(open_stress) == 2 * len(OPEN_CITIES)
    md.append(f'Rule 1: B beats A in {votes} of {len(OPEN_CITIES)} open cities -> '
              f'{"PASS" if votes >= 2 else "FAIL"}')
    md.append(f'Rule 2: height-robust in {sum(r["rule2_pass"] for r in open_stress)} of '
              f'{len(open_stress)} open (city, height) cells -> '
              f'{"PASS" if robust else "FAIL"}')
    md.append('')
    md.append('| city | refs (sites, box) | med abs az A | med abs az B | D = A - B [95% CI] | '
              'p90 A / B | rule 1 | med row A / B px | med A-B dist m |')
    md.append('|---|---|---:|---:|---|---|---|---|---:|')
    for r in head_rows:
        md.append(f"| {r['city']} | {r['n_refs']} ({r['n_sites']}, {r['n_box']}) | "
                  f"{fmt(r['med_az_a'])} | {fmt(r['med_az_b'])} | {fmt(r['D_deg'])} "
                  f"[{fmt(r['D_lo'])}, {fmt(r['D_hi'])}] | {fmt(r['p90_az_a'])} / "
                  f"{fmt(r['p90_az_b'])} | {'pass' if r['rule1_pass'] else 'fail'} | "
                  f"{fmt(r['med_row_a_px'], 1)} / {fmt(r['med_row_b_px'], 1)} | "
                  f"{fmt(r['med_ab_dist_m'])} |")
    md.append('')
    md.append('| city | h m | sites | lost | changed | B unsolvable | med shift A frozen | '
              'med shift A re-assoc | med shift B re-assoc | B = 0 share | B/A | rule 2 |')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for r in stress_out:
        md.append(f"| {r['city']} | {r['height_m']:g} | {r['sites_both_defined']} | "
                  f"{r['lost_match']} | {r['changed_membership']} | "
                  f"{r['b_unsolvable_after']} | {fmt(r['med_shift_a_frozen_m'])} | "
                  f"{fmt(r['med_shift_a_reassoc_m'])} | {fmt(r['med_shift_b_reassoc_m'], 3)} |"
                  f" {fmt(r['share_b_zero'])} | {fmt(r['ratio_b_over_a'], 3)} | "
                  f"{'pass' if r['rule2_pass'] else 'fail'} |")
    md.append('')
    md.append('| city | GT-free views (sites) | med abs az A | med abs az B | D [95% CI] | '
              'p90 A / B | unsolvable views (A med) |')
    md.append('|---|---|---:|---:|---|---|---|')
    for r in gtfree_out:
        md.append(f"| {r['city']} | {r['n_views']} ({r['n_sites']}) | {fmt(r['med_az_a'])} |"
                  f" {fmt(r['med_az_b'])} | {fmt(r['D_deg'])} [{fmt(r['D_lo'])}, "
                  f"{fmt(r['D_hi'])}] | {fmt(r['p90_az_a'])} / {fmt(r['p90_az_b'])} | "
                  f"{r['n_unsolvable']} ({fmt(r['med_az_a_on_unsolvable'])}) |")
    md.append('')
    md.append('| city | subset | spread >= | n | med A | med B | D [95% CI] |')
    md.append('|---|---|---:|---:|---:|---:|---|')
    for r in sub_out:
        md.append(f"| {r['city']} | {r['subset']} | {r['spread_min_deg']:g} | {r['n']} | "
                  f"{fmt(r['med_az_a'])} | {fmt(r['med_az_b'])} | {fmt(r['D_deg'])} "
                  f"[{fmt(r['D_lo'])}, {fmt(r['D_hi'])}] |")
    md.append('')
    md.append('| city | parallax bin deg | n | med A | med B |')
    md.append('|---|---|---:|---:|---:|')
    for r in bins_out:
        md.append(f"| {r['city']} | {r['parallax_bin']} | {r['n']} | {fmt(r['med_az_a'])} | "
                  f"{fmt(r['med_az_b'])} |")
    md.append('')
    md.append('| city | member refs | unit | B unsolvable (low spread / bad range) | '
              'A med az on unsolvable (p90) | A med az on unit |')
    md.append('|---|---:|---:|---|---|---:|')
    for r in surv_out:
        md.append(f"| {r['city']} | {r['member_refs_total']} | {r['unit']} | "
                  f"{r['b_unsolvable']} ({r['unsolvable_low_spread']} / "
                  f"{r['unsolvable_bad_range']}) | {fmt(r['med_az_a_on_unsolvable'])} "
                  f"({fmt(r['p90_az_a_on_unsolvable'])}) | {fmt(r['med_az_a_on_unit'])} |")
    md.append('')
    md.append('| city | spread >= | sites >= 3 views | spread ok | B solvable |')
    md.append('|---|---:|---:|---:|---:|')
    for r in keeps_out:
        md.append(f"| {r['city']} | {float(r['spread_min_deg']):g} | {r['sites_3plus']} | "
                  f"{r['sites_spread_ok']} | {r['sites_b_solvable']} |")
    md.append('')
    md.append('EXPLORATORY (post hoc):')
    md.append('')
    md.append('| city | units | group | n | med A | med B | share B better | D [95% CI] |')
    md.append('|---|---|---|---:|---:|---:|---:|---|')
    for r in explore:
        md.append(f"| {r['city']} | {r['units']} | {r['group']} | {r['n']} | "
                  f"{fmt(r['med_az_a'])} | {fmt(r['med_az_b'])} | "
                  f"{fmt(r['share_b_better'])} | {fmt(r['D_deg'])} "
                  f"[{fmt(r['D_lo'])}, {fmt(r['D_hi'])}] |")
    text = '\n'.join(md) + '\n'
    (out / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='one city: refs/gtfree/stress/keeps CSVs')
    r.add_argument('city', choices=sorted(CITIES))
    r.add_argument('--run-root', type=Path, required=True)
    r.add_argument('--benchmark-root', type=Path, required=True)
    sub.add_parser('summarize', help='apply the pre-registered rules to the CSVs')
    args = ap.parse_args()
    if args.cmd == 'run':
        run_city(args.city, args.run_root, args.benchmark_root)
    else:
        summarize()


if __name__ == '__main__':
    main()
