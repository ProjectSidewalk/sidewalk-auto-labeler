"""Leave-one-view-out reprojection residual over fused multi-view sites (issue #36).

A fused site is a GLS triangulation of every view that saw one ramp, so each view
makes a testable prediction about the others: drop it, re-solve the site from the
rest, and ask where that position lands in the dropped pano. The distance between
that prediction and what the pano actually emitted is the reprojection residual. Two
variants, which answer different questions:

GT-FREE (every multi-view site, every city, no labelling)
    For every site with >= MIN_VIEWS operational refit members (so at least two
    remain when one is held out) and each member i, subtract i's information from
    the site's accumulators (they are additive, so this is a subtraction, not a
    re-fuse), solve the held-out position, and compare with the view's own
    detection: in heatmap pixels (1024 x 512, x and y separately) after projecting
    with geo.ground_point_to_pano under the height the site was fused at, and in
    metres on the ground (the view's own raycast point minus the held-out position,
    split along and across that view's ray). It measures SELF-CONSISTENCY of the
    geometry and the detector. It cannot see a bias every view shares - a
    common translation, or a scale error that moves all views of a site the same
    way - because the held-out position inherits it. Its residuals are also
    truncated by the association that built the site (chi-square gate, 8 m cap):
    a view that disagreed by more was never a member, so the tails are floors.

    The scale instrument. If every range is k times the truth (a wrong camera
    height does exactly that: d = h / tan(depression)), then with s = 1 - 1/k
    each member's point is X + s r_j u_j (X the true ramp, r_j the measured
    range, u_j the unit ray) and the held-out GLS solution is
    X + s Lambda_{-i}^{-1} sum_j W_j r_j u_j. So the residual vector is EXACTLY

        member_i - heldout_i = s * g_i,
        g_i = r_i u_i - Lambda_{-i}^{-1} sum_{j != i} W_j r_j u_j,

    where g_i is computed from data alone. Least squares of residual on g (through
    the origin) estimates s, hence the implied range scale k = 1 / (1 - s),
    corrected for the viewing geometry - which the naive slope of along-ray
    residual on range (the RampNet#101 instrument, also reported) is not: it is
    attenuated by however much the other views' rays point the same way. With
    capture year as the group, the same identity is linear in one s per year, so
    a joint fit separates rigs that share sites (fit_scale).

GT-ANCHORED (judged benchmark panos only)
    For every judged pano (eval_sites.judged_gt_panos, the benchmark's own gate)
    that saw a ramp belonging to a site, project the site into that pano and
    compare with a reviewer reference pixel, both from the FULL site and from the
    site with that pano's own view LEFT OUT (so the reference pixel never
    contributed to the position it is scored against). References, reported
    separately because only some are independent of the detector:

      box     centre of the reviewer's extent box (RampNet boxes.json, where the
              split has one) for a verdict-true detection or a missed mark:
              drawn by hand, independent of our geometry and of the peak
      missed  a reviewer's missed-ramp click with no box: independent. The pano
              is not a refit member of the site it is matched to (the model did
              not detect it there at the benchmark tier), so full == left-out
      peak    a verdict-true detection with no box: the reviewer confirmed a ramp
              is there, but the pixel is the model's own peak - left-out on these
              is the GT-free residual restricted to reviewer-confirmed views

    A missed mark is matched to a site in world space (raycast under the same
    height model, nearest operational site within MATCH_RADIUS_M = eval_sites'
    default, one to one per pano), so its residual is truncated there too. Metres
    for GT rows are the reference pixel raycast from the judged pano minus the
    site position, so they share that pano's height model; the pixel residual is
    the independent number.

Nothing here changes a production path: it reads results.jsonl + sites.jsonl (or
re-fuses in memory through fuse_sites.fuse) and RampNet's benchmark as data, and
writes CSVs and a report. Sites are fused at BENCHMARK_CONFIDENCE with
mask_rig=False, the parameters every sites.jsonl on disk was written with and the
ones eval_sites pins, so GT joins stay keyed to the bundles.

Usage:
    python scripts/reprojection_residual.py paterson
    python scripts/reprojection_residual.py paterson bend --camera-height-m 2.6 per-pano
    python scripts/reprojection_residual.py paterson --run-root D:/Git/sidewalk-auto-labeler/runs
    # every city at once, plus the aggregate tables under runs/_summary/reprojection/
    python scripts/reprojection_residual.py bend paterson gainesville sao_paulo \\
        richmond clovis laurens laurens_gsv annapolis morgantown \\
        --camera-height-m 2.6 per-pano --refuse --publish docs/figures/reprojection-residual/data

Findings: docs/reprojection-residual.md.
"""
import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

MIN_VIEWS = 3                    # operational refit views, so >= 2 survive a hold-out
MATCH_RADIUS_M = 5.0             # eval_sites' default match radius
HEATMAP_W, HEATMAP_H = 1024, 512
RANGE_EDGES = (8.0, 12.0, 18.0, 25.0)
# A split name that differs from the run directory's. laurens is the dual-source city:
# its Mapillary run is runs/laurens, and the GSV run already carries the split's name.
DEFAULT_SPLITS = {'laurens': 'laurens_mapillary'}
MIN_GROUP_ROWS = 30              # a capture year with fewer member rows joins 'other'


# --- Views and the information-form algebra ------------------------------------------

@dataclass
class View:
    """One operational refit member of a site, in the run's local ENU frame."""
    pano_id: str
    det_index: int
    x: float
    y: float
    conf: float
    e: float
    n: float
    cov: tuple               # sym2 ENU covariance, exactly what fusion weighted it by
    range_m: float
    bearing_deg: float
    source: str
    capture_date: str | None
    sigma_along_m: float = 0.0    # the error model's own 1-sigmas (null_scale reads them)
    sigma_cross_m: float = 0.0

    @property
    def unit(self):
        b = math.radians(self.bearing_deg)
        return math.sin(b), math.cos(b)


@dataclass
class SiteViews:
    site_id: int
    views: list                        # operational refit members
    e: float                           # the full fused position
    n: float
    pano_ids: set = field(default_factory=set)   # every member pano, incl. support


def views_from_site(site):
    """SiteViews from an in-memory fuse_sites.Site."""
    views = [View(d.pano_id, d.det_index, d.x, d.y, d.conf, d.e, d.n, d.cov,
                  d.ground.range_m, d.ground.bearing_deg, d.source, d.capture_date,
                  d.ground.sigma_along_m, d.ground.sigma_cross_m)
             for d, in_refit in site.members if in_refit and d.operational]
    return SiteViews(site.id, views, site.e, site.n, set(site.pano_ids))


def views_from_json(rec, frame, min_confidence, sigma_scale=1.0):
    """SiteViews from one sites.jsonl line, rebuilding each member's covariance from
    its stored (range, bearing, sigma_along, sigma_cross) exactly as fuse_sites.project
    does. The stored fields are rounded (1e-7 deg, 1 mm), which moves positions by
    centimetres; load_sites_json reports the worst full-site discrepancy so a reader can
    see it is negligible."""
    s2 = sigma_scale ** 2
    views = []
    for m in rec['members']:
        if not m['in_refit'] or m['confidence'] < min_confidence:
            continue
        g = geo.GroundEstimate(m['lat'], m['lng'], m['range_m'], m['bearing_deg'],
                               m['sigma_along_m'], m['sigma_cross_m'])
        cov = g.cov_en(geo.error_model_for(m['source']).sigma_gps_m)
        if s2 != 1.0:
            cov = (cov[0] * s2, cov[1] * s2, cov[2] * s2)
        e, n = frame.to_enu(m['lat'], m['lng'])
        views.append(View(m['pano_id'], m['det_index'], m['x_normalized'],
                          m['y_normalized'], m['confidence'], e, n, cov, m['range_m'],
                          m['bearing_deg'], m['source'], m['capture_date'],
                          m['sigma_along_m'], m['sigma_cross_m']))
    e, n = frame.to_enu(rec['lat'], rec['lng'])
    return SiteViews(rec['site_id'], views, e, n, {m['pano_id'] for m in rec['members']})


def _info(v):
    w = geo.sym2_inv(v.cov)
    return w, (w[0] * v.e + w[1] * v.n, w[1] * v.e + w[2] * v.n)


def _solve(lam, eta):
    inv = geo.sym2_inv(lam)
    return (inv[0] * eta[0] + inv[1] * eta[1], inv[1] * eta[0] + inv[2] * eta[1])


def _mul(m, v):
    return (m[0] * v[0] + m[1] * v[1], m[1] * v[0] + m[2] * v[1])


def accumulate(views):
    """(Lambda, eta) summed over views: fuse_sites.Site's accumulators."""
    lam, eta = (0.0, 0.0, 0.0), (0.0, 0.0)
    for v in views:
        w, we = _info(v)
        lam = geo.sym2_add(lam, w)
        eta = (eta[0] + we[0], eta[1] + we[1])
    return lam, eta


def solve_views(views):
    """The GLS position of a set of views - what fusion would place them at."""
    return _solve(*accumulate(views))


def leave_one_out(views):
    """[(held-out position, Lambda_{-i})] for each view: the site solved without it.

    Subtracts the view's information from the totals rather than re-summing the rest,
    which is the point of keeping the site in information form."""
    lam, eta = accumulate(views)
    out = []
    for v in views:
        w, we = _info(v)
        lam_i = (lam[0] - w[0], lam[1] - w[1], lam[2] - w[2])
        eta_i = (eta[0] - we[0], eta[1] - we[1])
        out.append((_solve(lam_i, eta_i), lam_i))
    return out


def scale_design(views, groups, group_names):
    """For each view i, the columns D_i[g] (2-vectors) of the range-scale identity

        member_i - heldout_i = sum_g s_g * D_i[g],
        D_i[g] = [g_i == g] r_i u_i - Lambda_{-i}^{-1} sum_{j != i, g_j == g} W_j r_j u_j

    which holds exactly when group g's ranges are all k_g times the truth
    (s_g = 1 - 1/k_g). With one group it is the pooled g_i of the module docstring.
    `groups` gives each view's group name; returns [[D_i[g] for g in group_names]]."""
    idx = {g: k for k, g in enumerate(group_names)}
    lam, _ = accumulate(views)
    wr = []
    for v in views:
        w, _ = _info(v)
        ue, un = v.unit
        wr.append((w, _mul(w, (v.range_m * ue, v.range_m * un))))
    totals = [(0.0, 0.0) for _ in group_names]
    for (w, t), g in zip(wr, groups):
        k = idx[g]
        totals[k] = (totals[k][0] + t[0], totals[k][1] + t[1])
    out = []
    for i, v in enumerate(views):
        w, t = wr[i]
        lam_i = (lam[0] - w[0], lam[1] - w[1], lam[2] - w[2])
        inv = geo.sym2_inv(lam_i)
        gi = idx[groups[i]]
        ue, un = v.unit
        cols = []
        for k in range(len(group_names)):
            tot = totals[k]
            if k == gi:
                tot = (tot[0] - t[0], tot[1] - t[1])
            pe, pn = _mul(inv, tot)
            own = (v.range_m * ue, v.range_m * un) if k == gi else (0.0, 0.0)
            cols.append((own[0] - pe, own[1] - pn))
        out.append(cols)
    return out


def along_cross(de, dn, bearing_deg):
    """Split an ENU vector along a ray (positive = away from the camera) and across it
    (positive = to the camera's right, i.e. clockwise)."""
    b = math.radians(bearing_deg)
    ue, un = math.sin(b), math.cos(b)
    return de * ue + dn * un, de * un - dn * ue


def pixel_residual(x_obs, y_obs, x_pred, y_pred):
    """(dx, dy) observed minus predicted, in heatmap pixels, x wrapped across the seam.
    Positive dy = the observation sits lower in the pano (nearer) than predicted."""
    dx = ((x_obs - x_pred + 0.5) % 1.0 - 0.5) * HEATMAP_W
    return dx, (y_obs - y_pred) * HEATMAP_H


def project_into(pose, frame, e, n, camera_height):
    """ground_point_to_pano of an ENU point, uncapped in range: a held-out position a
    little past 25 m is still a prediction worth scoring, and dropping it would bias
    the far bucket toward agreement."""
    lat, lng = frame.to_latlng(e, n)
    return geo.ground_point_to_pano(pose, lat, lng, camera_height=camera_height,
                                     max_range_m=math.inf)


# --- GT-free rows ------------------------------------------------------------------------

def _year(capture_date):
    return (capture_date or '')[:4] or 'unknown'


def gtfree_rows(city, height_label, sites, frame, by_id, camera_height):
    """One row per (site with >= MIN_VIEWS views, held-out view). Each row also carries
    the pooled scale regressor (g_along, g_cross) and the site's id so the fit can
    cluster by site."""
    rows = []
    for sv in sites:
        views = sv.views
        if len(views) < MIN_VIEWS:
            continue
        loo = leave_one_out(views)
        design = scale_design(views, ['all'] * len(views), ['all'])
        newest = max((fs._months(v.capture_date) for v in views
                      if fs._months(v.capture_date) is not None), default=None)
        for v, ((he, hn), _), cols in zip(views, loo, design):
            de, dn = v.e - he, v.n - hn
            along, cross = along_cross(de, dn, v.bearing_deg)
            g_along, g_cross = along_cross(cols[0][0], cols[0][1], v.bearing_deg)
            pose = geo.pano_pose(by_id[v.pano_id].pose_fields())
            h_used, _ = geo.camera_height_for(pose, camera_height=camera_height)
            proj = project_into(pose, frame, he, hn, camera_height)
            if proj is None:
                dx = dy = x_pred = y_pred = None
            else:
                x_pred, y_pred = proj.x_norm, proj.y_norm
                dx, dy = pixel_residual(v.x, v.y, x_pred, y_pred)
            months = fs._months(v.capture_date)
            rows.append({
                'city': city, 'height_model': height_label, 'site_id': sv.site_id,
                'pano_id': v.pano_id, 'det_index': v.det_index,
                'n_views': len(views), 'source': v.source,
                'capture_date': v.capture_date, 'capture_year': _year(v.capture_date),
                'delta_months': None if months is None or newest is None
                else newest - months,
                'camera_height_m': round(h_used, 3),
                'confidence': v.conf, 'range_m': round(v.range_m, 3),
                'heldout_range_m': None if proj is None else round(proj.range_m, 3),
                'x_obs': v.x, 'y_obs': v.y,
                'x_pred': None if x_pred is None else round(x_pred, 6),
                'y_pred': None if y_pred is None else round(y_pred, 6),
                'dx_px': _r(dx), 'dy_px': _r(dy),
                'px': None if dx is None else _r(math.hypot(dx, dy)),
                'along_m': _r(along), 'cross_m': _r(cross),
                'dist_m': _r(math.hypot(de, dn)),
                'res_e': de, 'res_n': dn,
                'g_along': _r(g_along), 'g_cross': _r(g_cross),
                'g_e': cols[0][0], 'g_n': cols[0][1],
            })
    return rows


def _r(v, nd=4):
    return None if v is None else round(v, nd)


# --- The scale fit -------------------------------------------------------------------

def _gauss_solve(a, b):
    """Solve a small dense linear system (partial pivoting); None if singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _inverse(a):
    n = len(a)
    cols = [_gauss_solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    if any(c is None for c in cols):
        return None
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit_scale(site_blocks, group_names):
    """Least-squares s_g from the identity in scale_design, with a cluster-robust
    (by site) sandwich standard error - residuals of one site's views share its
    held-out solution, so they are not independent.

    site_blocks: [[(D_i, y_i)]] per site, D_i = [2-vector per group], y_i the residual
    2-vector. Returns {group: (s, se)}, or {} if the system is singular."""
    k = len(group_names)
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for block in site_blocks:
        for d, y in block:
            for p in range(k):
                b[p] += d[p][0] * y[0] + d[p][1] * y[1]
                for q in range(k):
                    a[p][q] += d[p][0] * d[q][0] + d[p][1] * d[q][1]
    s = _gauss_solve(a, b)
    ainv = _inverse(a)
    if s is None or ainv is None:
        return {}
    meat = [[0.0] * k for _ in range(k)]
    for block in site_blocks:
        score = [0.0] * k
        for d, y in block:
            fe = y[0] - sum(s[p] * d[p][0] for p in range(k))
            fn = y[1] - sum(s[p] * d[p][1] for p in range(k))
            for p in range(k):
                score[p] += d[p][0] * fe + d[p][1] * fn
        for p in range(k):
            for q in range(k):
                meat[p][q] += score[p] * score[q]
    cov = [[sum(ainv[p][i] * meat[i][j] * ainv[j][q] for i in range(k) for j in range(k))
            for q in range(k)] for p in range(k)]
    return {g: (s[p], math.sqrt(max(cov[p][p], 0.0))) for p, g in enumerate(group_names)}


def null_scale(sites, seed=0):
    """The pooled scale estimator's answer when there is NO scale error, under the
    error model's own noise: every view of every scored site is re-drawn to look at the
    site's fused position exactly, then perturbed by N(0, sigma_along) along its ray,
    N(0, sigma_cross) across it and an isotropic GPS shift that moves camera and point
    together (so it changes no range), and the same fit is run with the same weights.

    Why: the regressor g_i contains the view's own measured range, whose noise also
    sits in the residual - errors in variables - and the association-free null says how
    much of a measured s that alone produces. The naive along~range slope is subject to
    the same artefact, far more strongly (a view whose range reads long is, by that
    fact, both farther and beyond the consensus), so its null is returned too.

    Returns (null s, null naive slope). Deterministic for a given seed."""
    import random
    rng = random.Random(seed)
    blocks, xs, ys, cl = [], [], [], []
    for sv in sites:
        if len(sv.views) < MIN_VIEWS:
            continue
        sim = []
        for v in sv.views:
            ue, un = v.unit
            ce, cn = v.e - v.range_m * ue, v.n - v.range_m * un
            te, tn = sv.e - ce, sv.n - cn
            r0 = math.hypot(te, tn)
            if r0 < 1e-6:
                continue
            u0 = (te / r0, tn / r0)
            a = rng.gauss(0.0, v.sigma_along_m)
            c = rng.gauss(0.0, v.sigma_cross_m)
            gps = geo.error_model_for(v.source).sigma_gps_m
            ge, gn = rng.gauss(0.0, gps), rng.gauss(0.0, gps)
            pe = sv.e + a * u0[0] + c * u0[1] + ge
            pn = sv.n + a * u0[1] - c * u0[0] + gn
            de, dn = pe - (ce + ge), pn - (cn + gn)
            sim.append(View(v.pano_id, v.det_index, v.x, v.y, v.conf, pe, pn, v.cov,
                            math.hypot(de, dn), math.degrees(math.atan2(de, dn)) % 360,
                            v.source, v.capture_date))
        if len(sim) < MIN_VIEWS:
            continue
        loo = leave_one_out(sim)
        design = scale_design(sim, ['all'] * len(sim), ['all'])
        blocks.append([(cols, (v.e - he, v.n - hn))
                       for v, ((he, hn), _), cols in zip(sim, loo, design)])
        for v, ((he, hn), _) in zip(sim, loo):
            xs.append(v.range_m)
            ys.append(along_cross(v.e - he, v.n - hn, v.bearing_deg)[0])
            cl.append(sv.site_id)
    fit = fit_scale(blocks, ['all'])
    return fit.get('all', (None, None))[0], ols_slope(xs, ys, cl)[0]


def ols_slope(xs, ys, clusters):
    """Slope of y on x with an intercept, and its cluster-robust standard error."""
    n = len(xs)
    if n < 3:
        return None, None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None, None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    scores = {}
    for x, y, c in zip(xs, ys, clusters):
        scores[c] = scores.get(c, 0.0) + (x - mx) * (y - icpt - slope * x)
    se = math.sqrt(sum(v * v for v in scores.values())) / sxx
    return slope, se, icpt


def scale_table(city, height_label, sites, rows):
    """Range-slope rows for one city and height model: the naive along~range slope,
    the pooled geometry-corrected scale, and a joint per-capture-year scale fit."""
    out = []
    if not rows:
        return out
    xs = [r['range_m'] for r in rows]
    ys = [r['along_m'] for r in rows]
    cl = [r['site_id'] for r in rows]
    slope, slope_se, icpt = ols_slope(xs, ys, cl)
    pooled_blocks = {}
    for r in rows:
        pooled_blocks.setdefault(r['site_id'], []).append(
            ([(r['g_e'], r['g_n'])], (r['res_e'], r['res_n'])))
    pooled = fit_scale(list(pooled_blocks.values()), ['all'])
    s, se = pooled.get('all', (None, None))
    out.append(_scale_row(city, height_label, 'all', len(rows), len(pooled_blocks),
                          slope, slope_se, icpt, s, se, null=null_scale(sites)))

    # Joint per-year fit: a site mixing rigs contributes to every year it holds.
    counts = {}
    for sv in sites:
        if len(sv.views) >= MIN_VIEWS:
            for v in sv.views:
                counts[_year(v.capture_date)] = counts.get(_year(v.capture_date), 0) + 1
    names = sorted(g for g, c in counts.items() if c >= MIN_GROUP_ROWS)
    if len(counts) > len(names):
        names.append('other')
    if len(names) < 2:
        return out
    blocks = []
    per_group_rows = {g: [] for g in names}
    row_by_key = {(r['site_id'], r['pano_id']): r for r in rows}
    for sv in sites:
        if len(sv.views) < MIN_VIEWS:
            continue
        groups = [_year(v.capture_date) if _year(v.capture_date) in names else 'other'
                  for v in sv.views]
        design = scale_design(sv.views, groups, names)
        block = []
        for v, g, cols in zip(sv.views, groups, design):
            r = row_by_key[(sv.site_id, v.pano_id)]
            block.append((cols, (r['res_e'], r['res_n'])))
            per_group_rows[g].append(r)
        blocks.append(block)
    joint = fit_scale(blocks, names)
    for g in names:
        grp = per_group_rows[g]
        if not grp or g not in joint:
            continue
        gs, gse, gi = ols_slope([r['range_m'] for r in grp], [r['along_m'] for r in grp],
                                [r['site_id'] for r in grp])
        s_g, se_g = joint[g]
        out.append(_scale_row(city, height_label, g, len(grp),
                              len({r['site_id'] for r in grp}), gs, gse, gi, s_g, se_g))
    return out


def _scale_row(city, height_label, group, n_rows, n_sites, slope, slope_se, icpt, s, se,
               null=(None, None)):
    k = None if s is None or s >= 1 else 1.0 / (1.0 - s)
    return {'city': city, 'height_model': height_label, 'capture_year': group,
            'n_views': n_rows, 'n_sites': n_sites,
            'naive_slope': _r(slope), 'naive_slope_se': _r(slope_se),
            'naive_intercept_m': _r(icpt),
            'scale_s': _r(s), 'scale_s_se': _r(se),
            'implied_range_scale_k': _r(k),
            'k_lo95': _r(None if s is None or s + 1.96 * se >= 1
                         else 1.0 / (1.0 - (s - 1.96 * se))),
            'k_hi95': _r(None if s is None or s + 1.96 * se >= 1
                         else 1.0 / (1.0 - (s + 1.96 * se))),
            'null_naive_slope': _r(null[1]), 'null_scale_s': _r(null[0]),
            # s less its errors-in-variables null, as a range scale
            'k_null_corrected': _r(None if s is None or null[0] is None
                                   else 1.0 / (1.0 - (s - null[0])))}


# --- Summaries --------------------------------------------------------------------------

def pct(values, p):
    """Nearest-rank percentile (the convention fuse_sites._percentiles uses)."""
    v = sorted(values)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, math.ceil(len(v) * p / 100) - 1))]


def _range_bucket(r):
    lo = 0.0
    for hi in RANGE_EDGES:
        if r < hi:
            return f'{lo:g}-{hi:g}'
        lo = hi
    return f'>={RANGE_EDGES[-1]:g}'


def _views_bucket(n):
    return '3' if n == 3 else '4-5' if n <= 5 else '6+'


def summarize_rows(rows, keys):
    """Distribution row: medians and p90s of |residual|, plus the signed along median.
    The denominator is always n_views (held-out views), stated in the column."""
    px = [r['px'] for r in rows if r['px'] is not None]
    adx = [abs(r['dx_px']) for r in rows if r['dx_px'] is not None]
    ady = [abs(r['dy_px']) for r in rows if r['dy_px'] is not None]
    dist = [r['dist_m'] for r in rows]
    along = [r['along_m'] for r in rows]
    aal = [abs(a) for a in along]
    acr = [abs(r['cross_m']) for r in rows]
    return {**keys, 'n_views': len(rows), 'n_sites': len({r['site_id'] for r in rows}),
            'n_projected': len(px),
            'px_p50': _r(pct(px, 50), 3), 'px_p90': _r(pct(px, 90), 3),
            'abs_dx_px_p50': _r(pct(adx, 50), 3), 'abs_dx_px_p90': _r(pct(adx, 90), 3),
            'abs_dy_px_p50': _r(pct(ady, 50), 3), 'abs_dy_px_p90': _r(pct(ady, 90), 3),
            'dist_m_p50': _r(pct(dist, 50), 3), 'dist_m_p90': _r(pct(dist, 90), 3),
            'along_m_median': _r(pct(along, 50), 3),
            'abs_along_m_p50': _r(pct(aal, 50), 3), 'abs_along_m_p90': _r(pct(aal, 90), 3),
            'abs_cross_m_p50': _r(pct(acr, 50), 3), 'abs_cross_m_p90': _r(pct(acr, 90), 3)}


def _bucket_order(val):
    """Numeric order for range / capture-delta / view-count buckets ('8-12' after
    '0-8', not after '18-25'); anything else sorts after them by name."""
    head = val.lstrip('>=').split('-')[0].rstrip('+')
    try:
        return (0, float(head), val)
    except ValueError:
        return (1, 0.0, val)


def breakdowns(rows):
    """The by-range / by-source / by-capture-delta / by-views strata the issue asks for."""
    out = []
    for dim, fn in (('range_m', lambda r: _range_bucket(r['range_m'])),
                    ('source', lambda r: r['source'] or 'unknown'),
                    ('delta_months', lambda r: es._vintage_bucket(r['delta_months'])),
                    ('n_views', lambda r: _views_bucket(r['n_views']))):
        groups = {}
        for r in rows:
            groups.setdefault(fn(r), []).append(r)
        for val in sorted(groups, key=_bucket_order):
            g = groups[val]
            out.append(summarize_rows(g, {'city': g[0]['city'],
                                          'height_model': g[0]['height_model'],
                                          'dimension': dim, 'bucket': val}))
    return out


# --- GT-anchored ----------------------------------------------------------------------

def load_boxes(split_dir):
    """{pano_id: {'det:k' | 'missed:j': (cx, cy, point)}} for boxed entries, or {}."""
    path = split_dir / 'boxes.json'
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for pid, entries in data.get('panos', {}).items():
        for key, b in entries.items():
            if b.get('status') == 'boxed' and b.get('cx') is not None:
                out.setdefault(pid, {})[key] = (b['cx'], b['cy'], b.get('point') or {})
    return out


BOX_POINT_TOL = 2e-3     # box 'point' vs the mark it was drawn for (normalized units)


def _box_for(boxes, pid, key, x, y, warnings):
    """The reviewer's box centre for a mark, or None. A box whose recorded anchor point
    is not the mark it is keyed to is refused (index drift between files)."""
    b = boxes.get(pid, {}).get(key)
    if b is None:
        return None
    cx, cy, point = b
    if point and (abs(point.get('x', x) - x) > BOX_POINT_TOL
                  or abs(point.get('y', y) - y) > BOX_POINT_TOL):
        warnings.append(f'{pid} {key}: box anchor does not match the mark; box ignored')
        return None
    return cx, cy


def _gt_measure(pose, frame, ref_x, ref_y, e, n, camera_height, errors, params):
    """(px residuals, metres) of a site position (e, n) against a reference pixel."""
    proj = project_into(pose, frame, e, n, camera_height)
    out = {'dx_px': None, 'dy_px': None, 'px': None,
           'along_m': None, 'cross_m': None, 'dist_m': None, 'pred_range_m': None}
    if proj is not None:
        dx, dy = pixel_residual(ref_x, ref_y, proj.x_norm, proj.y_norm)
        out.update(dx_px=_r(dx), dy_px=_r(dy), px=_r(math.hypot(dx, dy)),
                   pred_range_m=_r(proj.range_m, 3))
    g = geo.detection_ground_point(pose, ref_x, ref_y, camera_height=camera_height,
                                   max_range_m=math.inf, errors=errors,
                                   apply_pose=params.apply_pose)
    if g is not None:
        re_, rn = frame.to_enu(g.lat, g.lng)
        de, dn = re_ - e, rn - n
        a, c = along_cross(de, dn, g.bearing_deg)
        out.update(along_m=_r(a), cross_m=_r(c), dist_m=_r(math.hypot(de, dn)))
    return out


def gt_anchored_rows(city, height_label, verdict_panos, bundle_ops, boxes, run_panos,
                     sites_views, frame, params):
    """One row per reviewer reference that maps to an operational site. Returns
    (rows, counts, warnings)."""
    by_id = {p.pano_id: p for p in run_panos}
    counts = es.gt_counts()
    counts['gt_panos'] = len(verdict_panos)
    for unused in ('unplaceable', 'placeable', 'unsure_missed'):   # build_gt's, not ours
        counts.pop(unused)
    counts.update(refs_det=0, refs_missed=0, missed_unplaceable=0, missed_no_site=0,
                  det_no_site=0)
    warnings = []
    site_of = {}
    for sv in sites_views:
        for v in sv.views:
            site_of[(v.pano_id, v.det_index)] = sv
    op_sites = [sv for sv in sites_views if sv.views]
    grid = geo.GridIndex(MATCH_RADIUS_M)
    for sv in op_sites:
        grid.add(sv.e, sv.n, sv)
    rows = []
    for pid, entry, run_pano, ops, _in_pool in es.judged_gt_panos(
            verdict_panos, bundle_ops, by_id, counts, warnings):
        pose = geo.pano_pose(run_pano.pose_fields())
        errors = geo.error_model_for(run_pano.source)

        def emit(kind, ref_x, ref_y, sv, own_view, peak=None):
            others = [v for v in sv.views if v.pano_id != pid]
            full = _gt_measure(pose, frame, ref_x, ref_y, sv.e, sv.n,
                               params.camera_height_m, errors, params)
            if own_view is None:
                loo = full           # the pano never contributed to this position
            elif others:
                le, ln = solve_views(others)
                loo = _gt_measure(pose, frame, ref_x, ref_y, le, ln,
                                  params.camera_height_m, errors, params)
            else:
                loo = None
            row = {'city': city, 'height_model': height_label, 'pano_id': pid,
                   'ref_kind': kind, 'ref_x': round(ref_x, 6), 'ref_y': round(ref_y, 6),
                   'site_id': sv.site_id, 'n_views_full': len(sv.views),
                   'n_views_loo': len(others), 'pano_in_site': own_view is not None,
                   'source': run_pano.source,
                   'capture_year': _year(run_pano.capture_date)}
            # For a boxed detection: how far the reviewer's box centre sits from the
            # model's own peak - the floor under that row's residual that no geometry
            # can remove (peak placement within the ramp, not position error).
            if peak is not None:
                pdx, pdy = pixel_residual(ref_x, ref_y, peak[0], peak[1])
                row.update(ref_minus_peak_dx_px=_r(pdx), ref_minus_peak_dy_px=_r(pdy))
            else:
                row.update(ref_minus_peak_dx_px=None, ref_minus_peak_dy_px=None)
            for k, v in full.items():
                row[f'full_{k}'] = v
            for k in full:
                row[f'loo_{k}'] = None if loo is None else loo[k]
            rows.append(row)

        used = set()
        for k, (verdict, (stored_i, x, y, _c)) in enumerate(zip(entry['dets'], ops)):
            if verdict is not True:
                continue
            counts['refs_det'] += 1
            sv = site_of.get((pid, stored_i))
            if sv is None:
                counts['det_no_site'] += 1   # unplaceable (horizon / beyond 25 m)
                continue
            used.add(sv.site_id)
            box = _box_for(boxes, pid, f'det:{k}', x, y, warnings)
            if box is not None:
                emit('box', box[0], box[1], sv, (pid, stored_i), peak=(x, y))
            else:
                emit('peak', x, y, sv, (pid, stored_i))

        # Missed marks: world-space match to an operational site the pano is not a
        # refit member of, nearest first, one to one within the pano.
        cands = []
        for j, mark in enumerate(entry.get('missed', ())):
            if mark.get('unsure'):
                continue
            counts['refs_missed'] += 1
            box = _box_for(boxes, pid, f'missed:{j}', mark['x'], mark['y'], warnings)
            rx, ry = box if box is not None else (mark['x'], mark['y'])
            g = geo.detection_ground_point(
                pose, mark['x'], mark['y'], camera_height=params.camera_height_m,
                max_range_m=params.max_range_m, errors=errors,
                apply_pose=params.apply_pose)
            if g is None:
                counts['missed_unplaceable'] += 1
                continue
            me, mn = frame.to_enu(g.lat, g.lng)
            found = False
            for sv in grid.near(me, mn):
                if sv.site_id in used or any(v.pano_id == pid for v in sv.views):
                    continue
                d = math.hypot(sv.e - me, sv.n - mn)
                if d <= MATCH_RADIUS_M:
                    cands.append((d, j, sv.site_id, sv, 'box' if box else 'missed', rx, ry))
                    found = True
            if not found:
                counts['missed_no_site'] += 1
        cands.sort(key=lambda c: c[:3])
        done_marks = set()
        for d, j, sid, sv, kind, rx, ry in cands:
            if j in done_marks or sid in used:
                continue
            done_marks.add(j)
            used.add(sid)
            emit(kind, rx, ry, sv, None)
        matched = {c[1] for c in cands}
        counts['missed_no_site'] += len(matched - done_marks)   # lost the 1:1 contest
    return rows, counts, warnings


def summarize_gt(rows, keys, prefix):
    px = [r[f'{prefix}_px'] for r in rows if r[f'{prefix}_px'] is not None]
    ady = [abs(r[f'{prefix}_dy_px']) for r in rows if r[f'{prefix}_dy_px'] is not None]
    adx = [abs(r[f'{prefix}_dx_px']) for r in rows if r[f'{prefix}_dx_px'] is not None]
    dist = [r[f'{prefix}_dist_m'] for r in rows if r[f'{prefix}_dist_m'] is not None]
    dy = [r[f'{prefix}_dy_px'] for r in rows if r[f'{prefix}_dy_px'] is not None]
    along = [r[f'{prefix}_along_m'] for r in rows if r[f'{prefix}_along_m'] is not None]
    offs = [math.hypot(r['ref_minus_peak_dx_px'], r['ref_minus_peak_dy_px'])
            for r in rows if r.get('ref_minus_peak_dx_px') is not None]
    return {**keys, 'variant': prefix, 'n_refs': len(px),
            # boxed detections only: the box centre's own distance from the model's
            # peak, a floor under those rows that no position estimate can remove
            'n_box_vs_peak': len(offs), 'box_vs_peak_px_p50': _r(pct(offs, 50), 3),
            'px_p50': _r(pct(px, 50), 3), 'px_p90': _r(pct(px, 90), 3),
            'abs_dx_px_p50': _r(pct(adx, 50), 3), 'abs_dy_px_p50': _r(pct(ady, 50), 3),
            'dy_px_median': _r(pct(dy, 50), 3),
            'dist_m_p50': _r(pct(dist, 50), 3), 'dist_m_p90': _r(pct(dist, 90), 3),
            'along_m_median': _r(pct(along, 50), 3)}


def gt_summary(rows, city, height_label):
    """Per ref-kind group ('independent' = box + missed, and 'peak'), full vs left-out.
    Left-out rows need at least one other view; the n_refs column is the denominator."""
    out = []
    groups = {'independent': [r for r in rows if r['ref_kind'] in ('box', 'missed')],
              'box': [r for r in rows if r['ref_kind'] == 'box'],
              'missed': [r for r in rows if r['ref_kind'] == 'missed'],
              'peak': [r for r in rows if r['ref_kind'] == 'peak']}
    for name, grp in groups.items():
        keys = {'city': city, 'height_model': height_label, 'ref_group': name}
        out.append(summarize_gt(grp, keys, 'full'))
        out.append(summarize_gt([r for r in grp if r['loo_px'] is not None or
                                 r['loo_dist_m'] is not None], keys, 'loo'))
    return out


# --- Loading ---------------------------------------------------------------------------

def load_sites_json(run_dir):
    """(SiteViews list, frame, params-dict, worst full-site rebuild error in m) from a
    run's sites.jsonl + sites_meta.json, or None if either is missing."""
    sites_path, meta_path = run_dir / 'sites.jsonl', run_dir / 'sites_meta.json'
    if not sites_path.exists() or not meta_path.exists():
        return None
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    origin = meta['frame_origin']
    frame = geo.LocalFrame(origin['lat0'], origin['lng0'])
    params = meta['params']
    out, worst = [], 0.0
    with open(sites_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            sv = views_from_json(json.loads(line), frame, params['min_confidence'],
                                 params.get('sigma_scale', 1.0))
            if len(sv.views) >= 2:
                pe, pn = solve_views(sv.views)
                worst = max(worst, math.hypot(pe - sv.e, pn - sv.n))
            out.append(sv)
    return out, frame, params, worst


def meta_panos(run_dir):
    with open(run_dir / 'sites_meta.json', encoding='utf-8') as f:
        return json.load(f).get('n_panos')


def height_label(h):
    return geo.PER_PANO if h == geo.PER_PANO else f'{float(h):g}m'


def sites_for(run_dir, panos, camera_height, force_refuse):
    """Sites to score under a camera-height model: sites.jsonl when it was fused with
    this model at the benchmark tier (every one on disk is: 2.6 m, 0.55), else an
    in-memory re-fuse with those same parameters. Returns (sites, frame, params,
    provenance string)."""
    params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                           camera_height_m=camera_height)
    if not force_refuse:
        disk = load_sites_json(run_dir)
        if disk is not None:
            svs, frame, p, worst = disk
            same = (p.get('camera_height_m') == camera_height
                    and p.get('min_confidence') == BENCHMARK_CONFIDENCE
                    and not p.get('mask_rig', False) and not p.get('apply_pose'))
            if same:
                prov = (f'sites.jsonl on disk (fused at {p["camera_height_m"]} m, tier '
                        f'{p["min_confidence"]}); worst full-site rebuild error '
                        f'{worst * 100:.1f} cm')
                if meta_panos(run_dir) != len(panos):
                    # results.jsonl grew (or shrank) after the sites were fused: they
                    # still score, but not the same pano set a re-fuse would.
                    prov += (f'; STALE: fused over {meta_panos(run_dir)} panos, '
                             f'results.jsonl now has {len(panos)} (use --refuse)')
                return svs, frame, params, prov
    sites, frame, _ = fs.fuse(panos, params)
    return ([views_from_site(s) for s in sites], frame, params,
            f're-fused in memory (fuse_sites.fuse, height {camera_height}, tier '
            f'{BENCHMARK_CONFIDENCE}, mask_rig off)')


def load_split(benchmark_root, split):
    d = benchmark_root / split
    if not (d / 'verdicts.json').exists() or not (d / 'records.jsonl').exists():
        return None
    with open(d / 'verdicts.json', encoding='utf-8') as f:
        verdicts = json.load(f)['panos']
    bundle_ops = {}
    with open(d / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bundle_ops[rec['pano']['panorama_id']] = \
                    [(x['x_normalized'], x['y_normalized'], x['confidence'])
                     for x in rec['detections']]
    return verdicts, bundle_ops, load_boxes(d)


# --- Output ------------------------------------------------------------------------------

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    cols = list(rows[0])
    for r in rows[1:]:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


PER_VIEW_DROP = ('res_e', 'res_n', 'g_e', 'g_n')   # frame-internal; kept out of the CSV


def _fmt(v, nd=2):
    return '—' if v is None else f'{v:.{nd}f}'


def city_report(city, provenance, summaries, slopes, gt_rows_summary, gt_counts):
    lines = [f'# Reprojection residual: {city}', '',
             'Generated by scripts/reprojection_residual.py (issue #36). Residual = '
             'observed minus held-out prediction; px are RampNet heatmap pixels '
             '(1024x512, 0.35 deg each).', '']
    for label, prov in provenance.items():
        lines.append(f'- **{label}**: {prov}')
    lines += ['', '## GT-free (leave one view out, sites with >= 3 operational views)', '',
              '| height | views | sites | px p50 | px p90 | m p50 | m p90 | along median m |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['height_model']} | {s['n_views']} | {s['n_sites']} | "
                     f"{_fmt(s['px_p50'])} | {_fmt(s['px_p90'])} | {_fmt(s['dist_m_p50'])} | "
                     f"{_fmt(s['dist_m_p90'])} | {_fmt(s['along_m_median'])} |")
    lines += ['', '## Range scale', '',
              '| height | capture year | views | naive slope | scale s (±1.96 se) | '
              'implied k |', '|---|---|---:|---:|---:|---:|']
    for r in slopes:
        lines.append(f"| {r['height_model']} | {r['capture_year']} | {r['n_views']} | "
                     f"{_fmt(r['naive_slope'], 3)} | {_fmt(r['scale_s'], 3)} "
                     f"(±{_fmt(None if r['scale_s_se'] is None else 1.96 * r['scale_s_se'], 3)})"
                     f" | {_fmt(r['implied_range_scale_k'], 3)} |")
    if gt_rows_summary:
        lines += ['', '## GT-anchored', '',
                  '| height | refs | variant | n | px p50 | px p90 | m p50 | m p90 |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for r in gt_rows_summary:
            lines.append(f"| {r['height_model']} | {r['ref_group']} | {r['variant']} | "
                         f"{r['n_refs']} | {_fmt(r['px_p50'])} | {_fmt(r['px_p90'])} | "
                         f"{_fmt(r['dist_m_p50'])} | {_fmt(r['dist_m_p90'])} |")
        for label, c in gt_counts.items():
            lines.append('')
            lines.append(f'{label}: ' + ', '.join(f'{k} {v}' for k, v in c.items()))
    return '\n'.join(lines) + '\n'


def run_city(city, run_dir, heights, benchmark_root, split, force_refuse=False):
    """Everything for one run. Returns a dict of row lists + report text."""
    need_heights = any(h == geo.PER_PANO for h in heights)
    panos, _skipped = fs.load_results(run_dir / 'results.jsonl', read_heights=need_heights)
    by_id = {p.pano_id: p for p in panos}
    split_data = load_split(benchmark_root, split) if benchmark_root else None
    result = {'views': [], 'summary': [], 'breakdown': [], 'slopes': [], 'gt': [],
              'gt_summary': [], 'gt_counts': {}, 'provenance': {}, 'warnings': []}
    for h in heights:
        label = height_label(h)
        if h == geo.PER_PANO and not any(p.camera_height_m is not None for p in panos):
            result['provenance'][label] = 'skipped: no pano in this run has a measured height'
            continue
        sites, frame, params, prov = sites_for(run_dir, panos, h, force_refuse)
        if h == geo.PER_PANO:
            measured = sum(1 for p in panos if p.camera_height_m is not None)
            prov += f'; {measured}/{len(panos)} panos measured, rest at 2.6 m'
        result['provenance'][label] = prov
        rows = gtfree_rows(city, label, sites, frame, by_id, h)
        result['views'] += rows
        result['summary'].append(summarize_rows(rows, {'city': city, 'height_model': label}))
        result['breakdown'] += breakdowns(rows)
        result['slopes'] += scale_table(city, label, sites, rows)
        if split_data is not None:
            verdicts, bundle_ops, boxes = split_data
            gt_rows, counts, warnings = gt_anchored_rows(
                city, label, verdicts, bundle_ops, boxes, panos, sites, frame, params)
            result['gt'] += gt_rows
            result['gt_summary'] += gt_summary(gt_rows, city, label)
            result['gt_counts'][f'{label} GT counts ({split})'] = counts
            result['warnings'] += warnings
    result['report'] = city_report(city, result['provenance'], result['summary'],
                                   result['slopes'], result['gt_summary'],
                                   result['gt_counts'])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cities', nargs='+', help='run names under --run-root')
    ap.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs',
                    help='where runs/<city>/results.jsonl (+ sites.jsonl) are READ from')
    ap.add_argument('--out-root', type=Path, default=REPO_ROOT / 'runs',
                    help='outputs go to <out-root>/<city>/reprojection/ and '
                         '<out-root>/_summary/reprojection/')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--split', action='append', default=[],
                    help='CITY=SPLIT when the benchmark split name differs from the run '
                         f'name (built in: {DEFAULT_SPLITS})')
    ap.add_argument('--camera-height-m', type=fs.camera_height_arg, nargs='+',
                    default=[geo.DEFAULT_CAMERA_HEIGHT_M],
                    help='one or more height models: meters, or "per-pano" (#40)')
    ap.add_argument('--refuse', action='store_true',
                    help='always re-fuse in memory instead of reading sites.jsonl')
    ap.add_argument('--publish', type=Path, default=None,
                    help='also copy the aggregate CSVs here (the committed copies)')
    args = ap.parse_args()

    splits = dict(DEFAULT_SPLITS)
    for s in args.split:
        c, _, sp = s.partition('=')
        splits[c] = sp
    agg = {'summary': [], 'breakdown': [], 'slopes': [], 'gt_summary': []}
    all_gt = []
    for city in args.cities:
        run_dir = args.run_root / city
        if not (run_dir / 'results.jsonl').exists():
            print(f'{city}: no results.jsonl under {run_dir}; skipped')
            continue
        print(f'{city}: scoring ...', flush=True)
        r = run_city(city, run_dir, args.camera_height_m, args.benchmark_root,
                     splits.get(city, city), args.refuse)
        out = args.out_root / city / 'reprojection'
        write_csv(out / 'views.csv', [{k: v for k, v in row.items()
                                       if k not in PER_VIEW_DROP} for row in r['views']])
        write_csv(out / 'gt_anchored.csv', r['gt'])
        write_csv(out / 'breakdown.csv', r['breakdown'])
        (out / 'report.md').write_text(r['report'], encoding='utf-8')
        if r['warnings']:
            (out / 'warnings.txt').write_text('\n'.join(r['warnings']) + '\n',
                                              encoding='utf-8')
        print(r['report'])
        for k in agg:
            agg[k] += r[k]
        all_gt += r['gt']
    # Pooled GT-anchored rows (city 'ALL'): rows are keyed by (city, pano_id, ref), never
    # by pano or site id alone, so pooling is a plain concatenation.
    for label in dict.fromkeys(row['height_model'] for row in all_gt):
        agg['gt_summary'] += gt_summary([row for row in all_gt
                                         if row['height_model'] == label], 'ALL', label)
    # ...and every model pooled over only the cities per-pano covers (the GSV runs with
    # measured heights), so the two height models are compared on the same references.
    pp_cities = {row['city'] for row in all_gt if row['height_model'] == geo.PER_PANO}
    if pp_cities:
        for label in dict.fromkeys(row['height_model'] for row in all_gt):
            agg['gt_summary'] += gt_summary(
                [row for row in all_gt if row['height_model'] == label
                 and row['city'] in pp_cities], 'ALL_PER_PANO_CITIES', label)
    summ = args.out_root / '_summary' / 'reprojection'
    names = {'summary': 'gtfree_summary.csv', 'breakdown': 'gtfree_breakdown.csv',
             'slopes': 'range_slope.csv', 'gt_summary': 'gt_anchored_summary.csv'}
    for k, name in names.items():
        write_csv(summ / name, agg[k])
        if args.publish:
            args.publish.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(summ / name, args.publish / name)
    print(f'wrote {summ}' + (f' and {args.publish}' if args.publish else ''))


if __name__ == '__main__':
    main()
