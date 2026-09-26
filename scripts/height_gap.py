"""Why the two #53 camera-height instruments disagree on Richmond's GoPro Max (issue #87).

Instrument A (bearing-only fixed point) reads Richmond's GoPro Max at 1.983 m; instrument
B (the null-corrected leave-one-out scale identity, read at the 2.6 m association only)
reads 2.377 m. G = B(2.6) - h*_A = 0.394 m. This script tests the three candidate
explanations under decision rules pre-registered on #87 before any number was produced
(see the RULE_* constants and the verdict_* functions -- they are the plan's, verbatim):

  candidate 3  association pulls B toward the height it ran at (B was never a fixed
               point). Tested by sweeping B over the #53 association heights (E1) and by
               a simulation on Richmond's real view graph at a known height (E2).
  candidate 1  a constant vertical peak offset that B absorbs as scale. Measured from
               the reviewers' boxes (E3a) and simulated (E3b).
  candidate 2  A mixes mountings. Per-sequence fixed points of both instruments (E4).
  E5           instrument C (GT along-ray residual on range) at several heights.
  E6           second cases: Laurens (tall GoPro Max, A non-identifiable) and Paterson
               (GSV, with a depth-measured height) -- reported only.

Subcommands:
  sweep     A and B at every association height in mapillary_height.SWEEP_HEIGHTS, per
            group: A's fixed point h*_A (bootstrap CI), B(h) = h * (1 - (s - s_null)) at
            each h with the null averaged over NULL_SEEDS seeds (its SD quoted), the line
            B(h) = a_B + b_B h and its fixed point h*_B = a_B / (1 - b_B) with a CI from
            N_DRAWS parametric draws of each height's cluster-robust SE.
  simulate  keep a run's panos, headings and confidences; plant its 2.6 m fused sites as
            the true ramps; resynthesize every member detection's (x, y) with
            geo.ground_point_to_pano at a known camera height per rig (the named group at
            --true-height; Pulsar 2.6 m, unknown 2.65 m, anything else 2.6 m), add the
            Mapillary error model's noise scaled by --noise-scale and an optional
            constant dip offset (--offset-px, heatmap px; + = the peak sits LOWER than
            the ramp), and run the same A and B sweep.
  gt        E3a (box-minus-peak vertical offset per rig, bootstrap CI by pano) and E5
            (instrument C for the group's references at each --heights value plus h*_B).
  verdict   applies the pre-registered rules to the CSVs above and writes report.md.

Everything is offline CPU (no GPU, no network, no server writes) and changes nothing in
production: camera_heights.json, groups.csv, results.jsonl and sites.jsonl are only read.

Usage:
    python scripts/height_gap.py sweep richmond --group gopro/max --sequences
    python scripts/height_gap.py sweep richmond laurens clovis morgantown annapolis
    python scripts/height_gap.py simulate richmond --group gopro/max \\
        --true-height 1.8 2.0 2.2 2.4 --noise-scale 0 0.5 1 --seeds 2
    python scripts/height_gap.py gt richmond --group gopro/max --heights 1.98 2.2 2.38 2.6
    python scripts/height_gap.py verdict richmond

Findings: docs/mapillary-camera-height.md section 7.
"""
import argparse
import csv
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
import reprojection_residual as rr  # noqa: E402
import mapillary_height as mh  # noqa: E402
from detectors import on_camera_rig  # noqa: E402

SWEEP_HEIGHTS = mh.SWEEP_HEIGHTS
H0 = geo.DEFAULT_CAMERA_HEIGHT_M
NULL_SEEDS = 10           # the errors-in-variables null is averaged over this many seeds
N_DRAWS = 500             # parametric draws for h*_B's CI
DRAW_SEED = 87
N_BOOT_OFFSET = 2000      # bootstrap draws (by pano) for the box-minus-peak CI
SIM_TRUE_HEIGHTS = {'nctech/istar pulsar': 2.6, 'unknown': 2.65}   # the other Richmond rigs
SIM_DEFAULT_HEIGHT = 2.6

# --- The pre-registered rules (#87 plan, "Decision rules (fixed now)") -----------------
RULE3_MAX_FIXED_GAP_M = 0.25   # |h*_B - h*_A| <= this on real data (the #53 rule-3 bar) ...
RULE3_MIN_SLOPE = 0.3          # ... with b_B >= this
RULE3_SIM_TRUE_M = 2.0         # E2 at this h_true:
RULE3_SIM_MIN_BIAS_M = 0.25    #   B(2.6) - h_true >= this (>= 2/3 of G, right sign) ...
RULE3_SIM_FIXED_TOL_M = 0.10   #   ... while |h*_A - h_true| and |h*_B - h_true| <= this
RULE3_NOISES = (0.5, 1.0)      #   at both noise scales
RULE3_REJECT_SLOPE = 0.15      # rejected if b_B < this, or E2's B(2.6) within
RULE3_REJECT_RECOVER_M = 0.10  # this of h_true
RULE1_CONFIRM_MOVE_M = 0.20    # an eps inside the box-minus-peak CI moves the fixed-point
RULE1_REJECT_MOVE_M = 0.10     # gap >= 0.20 m (observed direction); |eps| <= 4 px < 0.10 m
RULE1_MAX_EPS_PX = 4.0
RULE2_CONFIRM_MEDIAN_M = 0.10  # median within-sequence gap <= this ...
RULE2_CONFIRM_CLOSURE = 0.75   # ... and the mixture arm closes >= 75% of G
RULE2_REJECT_MEDIAN_M = 0.25   # rejected: median within-sequence gap >= this, pooled sign
E5_MAX_SE = 0.05               # E5 decisive only if the C slope SE < this
E5_PRED_SLOPE = {'A (2.0 m rig)': 0.23, 'B (2.38 m rig)': 0.09}   # at 2.6 m, from the plan
EXPLAINED_AGREE_M = 0.25       # corrected fixed points within this on real data
EXPLAINED_REPRODUCE_M = 0.10   # and the confirmed mechanisms reproduce G within this


def _rd(v, nd=4):
    return None if v is None else round(v, nd)


def _tag(h):
    return f'{h:g}'.replace('.', 'p')


# --- Loading and groups -------------------------------------------------------------------

def load_run(run_dir, results='results.jsonl', group_by='rig'):
    """(panos, groups_of) for one results file. groups_of[pid] has 'rig', 'sequence'
    (Mapillary) or 'year' (any source). group_by='year' also reads measured heights."""
    jsonl = Path(run_dir) / results
    panos, _ = fs.load_results(jsonl, read_heights=(group_by == 'year'))
    if group_by == 'year':
        groups_of = {p.pano_id: {'year': (p.capture_date or '')[:4] or 'unknown',
                                 'rig': 'all', 'sequence': 'none'} for p in panos}
    else:
        groups_of = mh.pano_groups(jsonl)
    return panos, groups_of


# --- Instrument B at one association height, null averaged over seeds -----------------

def _b_names(sites, key_of, min_rows):
    """instrument_b's own grouping: groups under min_rows held-out views pool as 'other'."""
    counts = {}
    for sv in sites:
        for v in sv.views:
            counts[key_of(v.pano_id)] = counts.get(key_of(v.pano_id), 0) + 1
    names = sorted((g for g, c in counts.items() if c >= min_rows), key=str)
    if len(counts) > len(names):
        names.append('other')
    named = set(names)
    return names, (lambda pid: key_of(pid) if key_of(pid) in named else 'other')


def b_at(sites, key_of, h0, null_seeds=range(NULL_SEEDS), min_rows=rr.MIN_GROUP_ROWS):
    """Instrument B at association height h0 with the null averaged over `null_seeds`.

    mapillary_height.instrument_b gives the fit, the first seed's null and the offset fit;
    the remaining seeds refit the null only. Returns {group: row} with h_b = h0 * (1 - (s -
    mean null)) and null_sd_m = h0 * SD(null s) -- the null's own contribution to B."""
    seeds = list(null_seeds)
    rows = mh.instrument_b(sites, key_of, min_rows=min_rows, seed=seeds[0], h0=h0)
    if not rows:
        return {}
    names, kof = _b_names(sites, key_of, min_rows)
    nulls = {g: [r['null_s']] for g, r in rows.items()}
    for seed in seeds[1:]:
        rng = random.Random(seed)
        fit = rr.fit_scale(mh._scale_blocks(sites, kof, names, h0=h0,
                                            simulate=lambda sv: mh._null_views(sv, rng)),
                           names)
        for g in nulls:
            nulls[g].append(fit.get(g, (0.0, None))[0])
    out = {}
    for g, r in rows.items():
        s0 = statistics.fmean(nulls[g])
        sd = statistics.stdev(nulls[g]) if len(nulls[g]) > 1 else None
        net = r['scale_s'] - s0
        out[g] = {'group': g, 'n_views': r['n_views'], 'scale_s': r['scale_s'],
                  'scale_s_se': r['scale_s_se'], 'null_s_mean': s0, 'null_s_sd': sd,
                  'h_b': h0 * (1.0 - net), 'h_b_se': h0 * r['scale_s_se'],
                  'null_sd_m': None if sd is None else h0 * sd,
                  'h_b_raw': h0 * (1.0 - r['scale_s']),
                  'h_b_offset': (None if r['offset_fit_s'] is None
                                 else h0 * (1.0 - (r['offset_fit_s'] - s0)))}
    return out


# --- The sweep ------------------------------------------------------------------------------

def sweep(panos, keyings, heights=SWEEP_HEIGHTS, null_seeds=range(NULL_SEEDS),
          params=None, log=None, min_rows=rr.MIN_GROUP_ROWS):
    """Associate the run at every height once and read both instruments.

    keyings: {name: key_of(pano_id)} -- instrument B is fitted jointly per keying.
    Returns {'a': {h: [(site_id, pano_id, implied)]}, 'b': {name: {h: {group: row}}}}."""
    params = params or mh.production_params()
    by_id = {p.pano_id: p for p in panos}
    a, b = {}, {name: {} for name in keyings}
    for h in heights:
        t0 = time.time()
        sites, frame, _ = fs.fuse(panos, replace(params, camera_height_m=h))
        recs = []
        for site in sites:
            if site.n_operational < 2:
                continue
            for pid, vals in fs.implied_heights([site], frame, by_id).items():
                recs.extend((site.id, pid, v) for v in vals)
        a[h] = recs
        svs = [sv for sv in (rr.views_from_site(s) for s in sites)
               if len(sv.views) >= rr.MIN_VIEWS]
        for name, key_of in keyings.items():
            b[name][h] = b_at(svs, key_of, h, null_seeds, min_rows=min_rows)
        if log:
            log(f'  h={h:g}: {len(sites)} sites, {len(svs)} with >= {rr.MIN_VIEWS} views, '
                f'{len(recs)} implied heights ({time.time() - t0:.1f} s)')
    return {'a': a, 'b': b}


def b_fixed_point(b_by_h, group, n_draws=N_DRAWS, seed=DRAW_SEED):
    """(h*_B, a_B, b_B, ci_lo, ci_hi, undefined draws) of the line through (h, B(h)).

    The CI draws s at each height from N(s, cluster-robust SE) independently (the plan's
    choice; heights share data, so this treats them as independent) and keeps the mean
    null fixed."""
    pts = [(h, rows[group]) for h, rows in sorted(b_by_h.items()) if group in rows]
    if len(pts) < 3:
        return None, None, None, None, None, None
    hs = [h for h, _ in pts]
    h_star, a, b = mh.fixed_point(hs, [r['h_b'] for _, r in pts])
    rng = random.Random(seed)
    draws, undefined = [], 0
    for _ in range(n_draws):
        vals = [h * (1.0 - (rng.gauss(r['scale_s'], r['scale_s_se'] or 0.0) - r['null_s_mean']))
                for h, r in pts]
        hd, _, _ = mh.fixed_point(hs, vals)
        if hd is None:
            undefined += 1
        else:
            draws.append(hd)
    lo = hi = None
    if draws:
        draws.sort()
        lo = draws[int(0.025 * (len(draws) - 1))]
        hi = draws[int(math.ceil(0.975 * (len(draws) - 1)))]
    return h_star, a, b, lo, hi, undefined


def summarize(sw, keying, key_of, city, results, grouping, n_boot=mh.N_BOOT,
              boot_keys=None, groups=None, n_draws=N_DRAWS, extra=None):
    """One row per group: A's fixed point (with bootstrap CI for boot_keys), B at every
    height, B's fixed point with its CI, and the two gaps. `groups` limits the rows."""
    a_rows = mh.instrument_a(sw['a'], key_of, n_boot=n_boot, boot_keys=boot_keys)
    b_by_h = sw['b'][keying]
    names = set(a_rows)
    for rows in b_by_h.values():
        names |= set(rows)
    out = []
    for g in sorted(names, key=str):
        if groups is not None and g not in groups:
            continue
        ra = a_rows.get(g) or {}
        hb, ab, bb, lo, hi, und = b_fixed_point(b_by_h, g, n_draws=n_draws)
        off_pts = [(h, rows[g]['h_b_offset']) for h, rows in sorted(b_by_h.items())
                   if g in rows and rows[g]['h_b_offset'] is not None]
        hb_off = mh.fixed_point([h for h, _ in off_pts], [v for _, v in off_pts])[0] \
            if len(off_pts) >= 3 else None
        b26 = b_by_h.get(H0, {}).get(g)
        ha = ra.get('h_star')
        row = {'city': city, 'results': results, 'grouping': grouping, 'group': g,
               'a_panos': ra.get('n_panos'), 'a_sites': ra.get('n_sites'),
               'h_star_a': _rd(ha), 'a_intercept': _rd(ra.get('a')),
               'a_slope': _rd(ra.get('slope')), 'h_star_a_lo': _rd(ra.get('ci_lo')),
               'h_star_a_hi': _rd(ra.get('ci_hi')),
               'b_views_2p6': b26['n_views'] if b26 else None,
               'b_2p6': _rd(b26['h_b']) if b26 else None,
               'b_2p6_lo': _rd(b26['h_b'] - 1.96 * b26['h_b_se']) if b26 else None,
               'b_2p6_hi': _rd(b26['h_b'] + 1.96 * b26['h_b_se']) if b26 else None,
               'b_2p6_null_sd_m': _rd(b26['null_sd_m']) if b26 else None,
               'b_2p6_offset': _rd(b26['h_b_offset']) if b26 else None,
               'h_star_b': _rd(hb), 'b_intercept': _rd(ab), 'b_slope': _rd(bb),
               'h_star_b_lo': _rd(lo), 'h_star_b_hi': _rd(hi), 'b_draws_undefined': und,
               'h_star_b_offset': _rd(hb_off),
               'gap_raw': _rd(b26['h_b'] - ha) if b26 and ha is not None else None,
               'gap_fixed': _rd(hb - ha) if hb is not None and ha is not None else None,
               'null_sd_m_max': _rd(max((rows[g]['null_sd_m'] or 0.0)
                                        for rows in b_by_h.values() if g in rows))
               if any(g in rows for rows in b_by_h.values()) else None}
        for h in sorted(sw['a']):
            row[f'implied_at_{h:g}'] = _rd(ra.get(f'implied_at_{h:g}'))
        for h, rows in sorted(b_by_h.items()):
            row[f'b_at_{h:g}'] = _rd(rows[g]['h_b']) if g in rows else None
            row[f'b_views_at_{h:g}'] = rows[g]['n_views'] if g in rows else None
        if extra:
            row.update(extra.get(g, {}))
        out.append(row)
    return out


def quarter_support(row):
    """Rule 1 at a quarter of the counts (#53's sequence-qualifying test)."""
    f = mh.SEQ_SUPPORT_FRACTION
    return (row.get('a_panos') or 0) >= mh.RULE_MIN_PANOS_A * f \
        and (row.get('a_sites') or 0) >= math.ceil(mh.RULE_MIN_SITES_A * f) \
        and (row.get('b_views_2p6') or 0) >= mh.RULE_MIN_VIEWS_B * f


def mixture_arm(seq_rows, pooled):
    """E4's mixture arm: the held-out-view-weighted mean of per-sequence h*_A against the
    pooled B(2.6). Returns {subset: (mean, closure of G)} for qualifying and all sequences
    with a defined h*_A and B views."""
    g = pooled['b_2p6'] - pooled['h_star_a']
    out = {}
    for name, rows in (('qualifying', [r for r in seq_rows if r['quarter_support']]),
                       ('all', seq_rows)):
        use = [(r['h_star_a'], r['b_views_2p6']) for r in rows
               if r['h_star_a'] is not None and r['b_views_2p6']]
        if not use:
            out[name] = (None, None, 0)
            continue
        m = sum(h * w for h, w in use) / sum(w for _, w in use)
        out[name] = (m, (m - pooled['h_star_a']) / g if g else None, len(use))
    return out


def run_sweep(args):
    """The `sweep` subcommand."""
    cities = args.cities
    single = len(cities) == 1
    all_rows, seq_rows = [], []
    for city in cities:
        run_dir = args.run_root / city
        for results in args.results:
            if not (run_dir / results).exists():
                print(f'{city}: no {results}; skipped')
                continue
            t0 = time.time()
            panos, groups_of = load_run(run_dir, results, args.group_by)
            print(f'{city}/{results}: {len(panos)} panos; sweeping {SWEEP_HEIGHTS} with '
                  f'{args.null_seeds} null seeds ...', flush=True)
            kind = args.group_by
            keyings = {kind: (lambda pid, k=kind: groups_of[pid][k])}
            if args.sequences:
                grp = args.group

                def seq_key(pid):
                    g = groups_of[pid]
                    return f"seq:{g['sequence']}" if g['rig'] == grp else g['rig']
                keyings['sequence'] = seq_key
            if kind == 'year':
                keyings['all'] = lambda _pid: 'all'
            sw = sweep(panos, keyings, null_seeds=range(args.null_seeds), log=print)
            extra = None
            if kind == 'year':
                extra = {}
                per = {}
                for p in panos:
                    if p.camera_height_m is not None:
                        per.setdefault(groups_of[p.pano_id]['year'], []).append(
                            p.camera_height_m)
                        per.setdefault('all', []).append(p.camera_height_m)
                for g, v in per.items():
                    extra[g] = {'depth_median_m': _rd(statistics.median(v)),
                                'depth_panos': len(v)}
            rows = summarize(sw, kind, keyings[kind], city, results, kind,
                             n_boot=args.n_boot, extra=extra)
            if kind == 'year':
                rows += summarize(sw, 'all', keyings['all'], city, results, 'all',
                                  n_boot=args.n_boot, extra=extra)
            all_rows += rows
            if args.sequences:
                srows = summarize(sw, 'sequence', keyings['sequence'], city, results,
                                  'sequence', boot_keys=set(),
                                  groups=None)
                srows = [r for r in srows if str(r['group']).startswith('seq:')]
                for r in srows:
                    r['group'] = r['group'][4:]
                    r['rig'] = args.group
                    r['quarter_support'] = quarter_support(r)
                pooled = next(r for r in rows if r['group'] == args.group)
                mix = mixture_arm(srows, pooled)
                for name, (m, closure, n) in mix.items():
                    print(f'  mixture arm ({name}, {n} sequences): view-weighted h*_A '
                          f'{_fmt(m)} vs pooled h*_A {pooled["h_star_a"]} and B(2.6) '
                          f'{pooled["b_2p6"]}: closes {_fmt(closure, ".0%")} of G')
                seq_rows += srows
            for r in rows:
                print(f"  {r['group']}: h*_A {r['h_star_a']} (slope {r['a_slope']}), "
                      f"B(2.6) {r['b_2p6']} (null SD {r['b_2p6_null_sd_m']} m), "
                      f"h*_B {r['h_star_b']} [{r['h_star_b_lo']}, {r['h_star_b_hi']}] "
                      f"(b_B {r['b_slope']}); gap raw {r['gap_raw']}, fixed {r['gap_fixed']}")
            print(f'{city}/{results}: {time.time() - t0:.0f} s', flush=True)
    if args.out:
        out = args.out
    elif args.group_by == 'year':
        out = args.run_root / '_summary' / 'camera_height' / f'gap_{"_".join(cities)}_year.csv'
    elif single:
        out = args.run_root / cities[0] / 'camera_height' / 'gap' / 'sweep.csv'
    else:
        out = args.run_root / '_summary' / 'camera_height' / 'gap_sweep.csv'
    rr.write_csv(out, all_rows)
    print(f'wrote {out}')
    if seq_rows:
        sp = out.parent / 'sequences.csv'
        rr.write_csv(sp, seq_rows)
        print(f'wrote {sp}')


def _fmt(v, spec='.3f'):
    return 'n/a' if v is None else format(v, spec)


# --- Simulation on the real view graph -------------------------------------------------

def planted_sites(panos, params=None):
    """{(pano_id, det_index): (lat, lng)} -- every member of the run's 2.6 m fuse mapped to
    its site's fused position (the planted "true" ramp), plus the fuse's stats."""
    params = params or mh.production_params(camera_height_m=H0)
    sites, frame, stats = fs.fuse(panos, params)
    truth = {}
    for s in sites:
        lat, lng = frame.to_latlng(s.e, s.n)
        for d, _in_refit in s.members:
            truth[(d.pano_id, d.det_index)] = (lat, lng)
    return truth, stats


def resynthesize(panos, truth, true_height_of, noise_scale=0.0, offset_px=0.0, seed=0,
                 errors=geo.MAPILLARY_ERRORS):
    """Synthetic panos: same ids, headings, confidences and (pano, det) keys, detections
    re-projected from their planted site at the pano's true camera height.

    Noise (the error model's sigmas times noise_scale): per pano, a GPS shift of the
    stored position (sigma_gps), a heading error (sigma_heading), a rig tilt -- pitch and
    roll each N(0, sigma_pitch) -- entering the dip as pitch*cos(phi) - roll*sin(phi), and
    a true-height jitter (sigma_height); per detection, peak jitter of sigma_peak heatmap
    px on each axis. offset_px is a constant dip offset in heatmap px (+ = the peak sits
    lower than the ramp, i.e. it raycasts nearer). A detection with no planted site, or
    whose site does not project (at the camera), is dropped.

    Returns (panos, counts)."""
    rng = random.Random(seed)
    ns = noise_scale
    rad_px = geo.RAD_PER_HEATMAP_PX
    counts = {'dets_in': 0, 'dets_out': 0, 'no_site': 0, 'unprojectable': 0, 'on_rig': 0}
    out = []
    for p in panos:
        g = (lambda s: rng.gauss(0.0, s * ns) if ns else 0.0)
        h = true_height_of(p) + g(errors.sigma_height_m)
        ge, gn = g(errors.sigma_gps_m), g(errors.sigma_gps_m)
        dpsi = g(errors.sigma_heading_rad)
        pitch, roll = g(errors.sigma_pitch_rad), g(errors.sigma_pitch_rad)
        pose = geo.pano_pose(p.pose_fields(camera_pitch=None, camera_roll=None))
        dets = []
        for i, _x, _y, conf in p.detections:
            counts['dets_in'] += 1
            ll = truth.get((p.pano_id, i))
            if ll is None:
                counts['no_site'] += 1
                continue
            proj = geo.ground_point_to_pano(pose, *ll, camera_height=h, max_range_m=math.inf)
            if proj is None:
                counts['unprojectable'] += 1
                continue
            phi = (proj.x_norm - 0.5) * 2.0 * math.pi
            dphi = -dpsi + g(errors.sigma_peak_px * rad_px)
            ddip = pitch * math.cos(phi) - roll * math.sin(phi) \
                + g(errors.sigma_peak_px * rad_px) + offset_px * rad_px
            x = (proj.x_norm + dphi / (2.0 * math.pi)) % 1.0
            y = proj.y_norm + ddip / math.pi
            if on_camera_rig(y):
                counts['on_rig'] += 1           # kept: fusion's own mask drops it
            dets.append((i, x, y, conf))
            counts['dets_out'] += 1
        lat, lng = p.lat, p.lng
        if ge or gn:
            lat, lng = geo.LocalFrame(p.lat, p.lng).to_latlng(ge, gn)
        out.append(replace(p, lat=lat, lng=lng, detections=dets))
    return out, counts


def sim_height_of(groups_of, group, h_true):
    """True height per pano for a simulation: the named group at h_true, the other
    Richmond rigs at their measured #53 values, anything else at 2.6 m."""
    def of(p):
        rig = groups_of.get(p.pano_id, {}).get('rig')
        if rig == group:
            return h_true
        return SIM_TRUE_HEIGHTS.get(rig, SIM_DEFAULT_HEIGHT)
    return of


def simulate_one(panos, truth, groups_of, group, h_true, noise, offset_px, seed,
                 null_seeds=range(NULL_SEEDS), n_draws=N_DRAWS, heights=SWEEP_HEIGHTS):
    """One simulation configuration: resynthesize, sweep, summarize the named group."""
    syn, counts = resynthesize(panos, truth, sim_height_of(groups_of, group, h_true),
                               noise, offset_px, seed)
    key_of = lambda pid: groups_of[pid]['rig']   # noqa: E731
    sw = sweep(syn, {'rig': key_of}, heights=heights, null_seeds=null_seeds)
    rows = summarize(sw, 'rig', key_of, '', '', 'rig', n_boot=0, groups={group},
                     n_draws=n_draws)
    r = rows[0] if rows else {}
    keep = ('a_panos', 'a_sites', 'h_star_a', 'a_slope', 'b_views_2p6', 'b_2p6',
            'b_2p6_null_sd_m', 'b_2p6_offset', 'h_star_b', 'b_slope', 'h_star_b_lo',
            'h_star_b_hi', 'gap_raw', 'gap_fixed')
    return {'group': group, 'h_true': h_true, 'noise_scale': noise, 'offset_px': offset_px,
            'seed': seed, **{k: r.get(k) for k in keep},
            'a_err': _rd(r['h_star_a'] - h_true) if r.get('h_star_a') is not None else None,
            'b_2p6_err': _rd(r['b_2p6'] - h_true) if r.get('b_2p6') is not None else None,
            'b_fixed_err': _rd(r['h_star_b'] - h_true) if r.get('h_star_b') is not None
            else None,
            **{f'b_at_{h:g}': r.get(f'b_at_{h:g}') for h in heights},
            **{f'implied_at_{h:g}': r.get(f'implied_at_{h:g}') for h in heights},
            **{f'n_{k}': v for k, v in counts.items()}}


SIM_KEY = ('group', 'h_true', 'noise_scale', 'offset_px', 'seed')


def _read_csv(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _key(row):
    return tuple(str(float(row[k])) if k != 'group' else row[k] for k in SIM_KEY)


def run_simulate(args):
    """The `simulate` subcommand: rows are merged into simulate.csv by configuration, so
    E2 and E3b land in one file."""
    city = args.cities[0]
    run_dir = args.run_root / city
    panos, groups_of = load_run(run_dir, args.results[0])
    truth, stats = planted_sites(panos)
    print(f'{city}: {len(panos)} panos, {stats["n_sites"]} planted sites '
          f'({stats["n_projected"]} projected detections)', flush=True)
    out = args.out or run_dir / 'camera_height' / 'gap' / 'simulate.csv'
    rows = {_key(r): r for r in _read_csv(out)}
    for h_true in args.true_height:
        for noise in args.noise_scale:
            for off in args.offset_px:
                for seed in range(args.seeds):
                    t0 = time.time()
                    r = simulate_one(panos, truth, groups_of, args.group, h_true, noise, off,
                                     seed, null_seeds=range(args.null_seeds))
                    r = {'city': city, 'results': args.results[0], **r}
                    rows[_key(r)] = r
                    print(f"  h_true {h_true} noise {noise} eps {off:+g} px seed {seed}: "
                          f"h*_A {r['h_star_a']} (slope {r['a_slope']}), B(2.6) "
                          f"{r['b_2p6']}, h*_B {r['h_star_b']} (b_B {r['b_slope']}); "
                          f"{time.time() - t0:.0f} s", flush=True)
                    rr.write_csv(out, list(rows.values()))
    print(f'wrote {out}')


# --- GT: E3a (box minus peak) and E5 (instrument C) ------------------------------------

def median_ci(values, clusters, n_boot=N_BOOT_OFFSET, seed=DRAW_SEED):
    """(median, lo, hi): a 95% bootstrap CI resampling clusters (panos) with replacement."""
    if not values:
        return None, None, None
    by = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [v for _ in keys for v in by[keys[rng.randrange(len(keys))]]]
        draws.append(statistics.median(pick))
    draws.sort()
    return (statistics.median(values), draws[int(0.025 * (len(draws) - 1))],
            draws[int(math.ceil(0.975 * (len(draws) - 1)))])


def zero_crossing(points):
    """Height where C's slope crosses zero: weighted (1/SE^2) line through (h, slope),
    with a delta-method SE that treats the heights' slopes as independent (they share
    references, so the SE is optimistic). points = [(h, slope, se)]."""
    pts = [(h, s, se) for h, s, se in points if s is not None and se]
    if len(pts) < 2:
        return None, None, None
    w = [1.0 / se ** 2 for _, _, se in pts]
    sw = sum(w)
    mh_ = sum(wi * h for wi, (h, _, _) in zip(w, pts)) / sw
    ms = sum(wi * s for wi, (_, s, _) in zip(w, pts)) / sw
    sxx = sum(wi * (h - mh_) ** 2 for wi, (h, _, _) in zip(w, pts))
    if sxx <= 0:
        return None, None, None
    b = sum(wi * (h - mh_) * (s - ms) for wi, (h, s, _) in zip(w, pts)) / sxx
    a = ms - b * mh_
    if b == 0:
        return None, None, b
    h0 = -a / b
    # var(a + b h0) at the crossing, divided by b^2
    var_fit = 1.0 / sw + (h0 - mh_) ** 2 / sxx
    return h0, math.sqrt(var_fit) / abs(b), b


def run_gt(args):
    """The `gt` subcommand."""
    city = args.cities[0]
    run_dir = args.run_root / city
    panos, groups_of = load_run(run_dir, args.results[0])
    panos = mh.flat(panos)
    split = mh.SPLITS.get(city, city)
    split_data = rr.load_split(args.benchmark_root, split)
    if split_data is None:
        raise SystemExit(f'{city}: no benchmark split {split} under {args.benchmark_root}')
    verdicts, bundle_ops, boxes = split_data
    out_dir = args.out.parent if args.out else run_dir / 'camera_height' / 'gap'
    heights = list(args.heights)
    swp = _read_csv(out_dir / 'sweep.csv')
    hb = next((float(r['h_star_b']) for r in swp
               if r['group'] == args.group and r['h_star_b'] not in ('', None)), None)
    if hb is not None and all(abs(hb - h) > 1e-3 for h in heights):
        heights.append(round(hb, 3))
    rig_of = lambda pid: groups_of.get(pid, {}).get('rig')   # noqa: E731
    c_rows, offset_rows = [], []
    for h in heights:
        params = mh.gate_params(h)
        sites, frame, _ = fs.fuse(panos, params)
        svs = [rr.views_from_site(s) for s in sites]
        rows, _counts, warnings = rr.gt_anchored_rows(city, rr.height_label(h), verdicts,
                                                      bundle_ops, boxes, panos, svs, frame,
                                                      params)
        if abs(h - H0) < 1e-9:
            by_rig = {}
            for r in rows:
                if r['ref_kind'] == 'box' and r['ref_minus_peak_dy_px'] is not None:
                    by_rig.setdefault(rig_of(r['pano_id']), []).append(r)
            by_rig['all'] = [r for rs in by_rig.values() for r in rs]
            for rig in sorted(by_rig, key=str):
                rs = by_rig[rig]
                med, lo, hi = median_ci([r['ref_minus_peak_dy_px'] for r in rs],
                                        [r['pano_id'] for r in rs])
                offset_rows.append({
                    'city': city, 'rig': rig, 'n_boxes': len(rs),
                    'n_panos': len({r['pano_id'] for r in rs}),
                    'box_minus_peak_dy_px_median': _rd(med, 3),
                    'ci_lo': _rd(lo, 3), 'ci_hi': _rd(hi, 3),
                    'box_minus_peak_dy_px_mean': _rd(statistics.fmean(
                        r['ref_minus_peak_dy_px'] for r in rs), 3),
                    # the simulation's eps (peak minus ramp) is the negation
                    'eps_px_median': _rd(-med, 3) if med is not None else None,
                    'eps_px_lo': _rd(-hi, 3) if hi is not None else None,
                    'eps_px_hi': _rd(-lo, 3) if lo is not None else None})
        mine = [r for r in rows if rig_of(r['pano_id']) == args.group]
        for name, kinds in (('all', ('box', 'missed', 'peak')), ('independent', ('box', 'missed'))):
            use = [r for r in mine if r['ref_kind'] in kinds and r['loo_along_m'] is not None
                   and r['loo_pred_range_m'] is not None]
            slope, se, icpt = rr.ols_slope([r['loo_pred_range_m'] for r in use],
                                           [r['loo_along_m'] for r in use],
                                           [r['site_id'] for r in use])
            c_rows.append({'city': city, 'group': args.group, 'height_m': h, 'refs': name,
                           'n': len(use), 'n_sites': len({r['site_id'] for r in use}),
                           'slope': _rd(slope), 'slope_se': _rd(se),
                           'intercept_m': _rd(icpt)})
        print(f'  h={h:g}: {len(mine)} {args.group} reference rows; C slope (all) '
              f"{c_rows[-2]['slope']} +/- {c_rows[-2]['slope_se']} (n {c_rows[-2]['n']})",
              flush=True)
    for name in ('all', 'independent'):
        pts = [(r['height_m'], r['slope'], r['slope_se']) for r in c_rows if r['refs'] == name]
        hz, hz_se, dsdh = zero_crossing(pts)
        c_rows.append({'city': city, 'group': args.group, 'height_m': 'zero_crossing',
                       'refs': name, 'n': None, 'n_sites': None, 'slope': _rd(dsdh),
                       'slope_se': None, 'intercept_m': None,
                       'zero_crossing_m': _rd(hz), 'zero_crossing_se_m': _rd(hz_se)})
    rr.write_csv(out_dir / 'instrument_c.csv', c_rows)
    rr.write_csv(out_dir / 'gt_offset.csv', offset_rows)
    for r in offset_rows:
        print(f"  box-minus-peak dy {r['rig']}: {r['box_minus_peak_dy_px_median']} px "
              f"[{r['ci_lo']}, {r['ci_hi']}] (n {r['n_boxes']})")
    print(f'wrote {out_dir / "instrument_c.csv"} and gt_offset.csv')


# --- The verdict (pre-registered rules) ---------------------------------------------------

def verdict_candidate3(real, sim):
    """Candidate 3 (association pulls B toward the association height).

    real: {'h_star_a', 'h_star_b', 'b_slope'} for the group on real data.
    sim: {noise: {'b_2p6', 'h_star_a', 'h_star_b'}} at h_true = RULE3_SIM_TRUE_M, seeds
    averaged. Returns (outcome, share, reasons); outcome in confirmed/rejected/partial.
    The rejection's "E2 recovers h_true" is read at every RULE3_NOISES level (the plan's
    risks: a conclusion must hold at 0.5 and 1.0, else it is noise-dependent)."""
    h = RULE3_SIM_TRUE_M
    gap = abs(real['h_star_b'] - real['h_star_a']) \
        if real.get('h_star_b') is not None and real.get('h_star_a') is not None else None
    slope = real.get('b_slope')
    real_ok = gap is not None and slope is not None \
        and gap <= RULE3_MAX_FIXED_GAP_M and slope >= RULE3_MIN_SLOPE
    sims = [sim.get(n) for n in RULE3_NOISES]
    sim_ok = all(s is not None and s['b_2p6'] - h >= RULE3_SIM_MIN_BIAS_M
                 and abs(s['h_star_a'] - h) <= RULE3_SIM_FIXED_TOL_M
                 and s['h_star_b'] is not None
                 and abs(s['h_star_b'] - h) <= RULE3_SIM_FIXED_TOL_M for s in sims)
    recovers = all(s is not None and abs(s['b_2p6'] - h) <= RULE3_REJECT_RECOVER_M
                   for s in sims)
    reasons = [f'real: |h*_B - h*_A| = {_fmt(gap)} m (<= {RULE3_MAX_FIXED_GAP_M}), '
               f'b_B = {_fmt(slope)} (>= {RULE3_MIN_SLOPE}) -> {"yes" if real_ok else "no"}']
    for n, s in zip(RULE3_NOISES, sims):
        if s is None:
            reasons.append(f'E2 noise {n:g}: missing')
            continue
        reasons.append(f"E2 noise {n:g}: B(2.6) - h_true = {_fmt(s['b_2p6'] - h, '+.3f')}, "
                       f"h*_A - h_true = {_fmt(s['h_star_a'] - h, '+.3f')}, h*_B - h_true = "
                       f"{_fmt(None if s['h_star_b'] is None else s['h_star_b'] - h, '+.3f')}")
    if real_ok and sim_ok:
        return 'confirmed', None, reasons
    if (slope is not None and slope < RULE3_REJECT_SLOPE) or recovers:
        return 'rejected', None, reasons
    return 'partial', None, reasons


def candidate3_share(real, g):
    """The plan's partial share: 1 - |h*_B - h*_A| / G."""
    if real.get('h_star_b') is None or real.get('h_star_a') is None or not g:
        return None
    return 1.0 - abs(real['h_star_b'] - real['h_star_a']) / g


def _interp(xs, ys, x):
    """Piecewise-linear interpolation, x clamped to [min(xs), max(xs)]."""
    pts = sorted(zip(xs, ys))
    x = min(max(x, pts[0][0]), pts[-1][0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def verdict_candidate1(eps_ci, gap_by_eps, direction=1.0):
    """Candidate 1 (a constant vertical peak offset).

    eps_ci: (lo, hi) of the GoPro Max peak-minus-box offset in heatmap px (the negated
    box-minus-peak CI, the simulation's eps convention). gap_by_eps: {eps: fixed-point
    gap h*_B - h*_A} from E3b. direction: the sign of the observed gap (+1: B above A).
    The move at an eps inside the CI is read at the CI's ends (and the grid points
    between) by linear interpolation over the grid -- the gap is monotone in eps to
    first order, so the ends bound it. Returns (outcome, reasons)."""
    xs = sorted(gap_by_eps)
    if 0.0 not in gap_by_eps or len(xs) < 2:
        return 'untested', ['E3b grid missing eps = 0']
    base = gap_by_eps[0.0]
    move = {e: gap_by_eps[e] - base for e in xs}
    lo, hi = eps_ci
    inside = [e for e in xs if lo <= e <= hi] + [lo, hi]
    in_ci = max(direction * (_interp(xs, [gap_by_eps[e] for e in xs], e) - base)
                for e in inside)
    within4 = [abs(move[e]) for e in xs if abs(e) <= RULE1_MAX_EPS_PX]
    reasons = [f'eps CI [{lo:+.2f}, {hi:+.2f}] px; largest move of the fixed-point gap '
               f'inside it in the observed direction: {in_ci:+.3f} m (confirm >= '
               f'{RULE1_CONFIRM_MOVE_M})',
               f'largest |move| over |eps| <= {RULE1_MAX_EPS_PX:g} px: {max(within4):.3f} m '
               f'(reject < {RULE1_REJECT_MOVE_M})',
               'moves by eps: ' + ', '.join(f'{e:+g}: {move[e]:+.3f}' for e in xs)]
    if in_ci >= RULE1_CONFIRM_MOVE_M:
        return 'confirmed', reasons
    if max(within4) < RULE1_REJECT_MOVE_M:
        return 'rejected', reasons
    return 'partial', reasons


def verdict_candidate2(seq_gaps, closure, pooled_sign=1.0):
    """Candidate 2 (A mixes mountings). seq_gaps: within-sequence gaps h*_B - h*_A for
    sequences meeting quarter support; closure: the mixture arm's share of G.
    Returns (outcome, median, reasons)."""
    if not seq_gaps:
        return 'untested', None, ['no sequence meets quarter support on both instruments']
    med = statistics.median(seq_gaps)
    reasons = [f'median within-sequence gap {med:+.3f} m over {len(seq_gaps)} sequences '
               f'({sum(1 for g in seq_gaps if g * pooled_sign > 0)} with the pooled sign); '
               f'mixture arm closes {_fmt(closure, ".0%")} of G']
    if abs(med) <= RULE2_CONFIRM_MEDIAN_M and closure is not None \
            and closure >= RULE2_CONFIRM_CLOSURE:
        return 'confirmed', med, reasons
    if med * pooled_sign >= RULE2_REJECT_MEDIAN_M:
        return 'rejected', med, reasons
    return 'partial', med, reasons


def verdict_e5(slope, se):
    """E5: decisive only if the C slope's SE at 2.6 m is < E5_MAX_SE; then it favours the
    candidate height whose predicted slope is nearer. Returns (decisive, favours, text)."""
    if slope is None or se is None:
        return False, None, 'no instrument C slope at 2.6 m'
    fav = min(E5_PRED_SLOPE, key=lambda k: abs(slope - E5_PRED_SLOPE[k]))
    decisive = se < E5_MAX_SE
    return decisive, fav if decisive else None, (
        f'C slope at 2.6 m {slope:+.3f} +/- {se:.3f} (decisive needs SE < {E5_MAX_SE}); '
        f'predicted ' + ', '.join(f'{k} {v:+.2f}' for k, v in E5_PRED_SLOPE.items())
        + (f'; nearer {fav}' if decisive else '; reported only'))


def verdict_explained(real, c3, c1, sim_raw_gaps, g, offset_corr=None):
    """'Explained' = the corrected instruments (both fixed points, offset-corrected when
    candidate 1 confirms: offset_corr = (dA, dB) to subtract) agree within 0.25 m on real
    data AND the confirmed mechanisms reproduce G within 0.10 m in simulation
    (sim_raw_gaps: the simulated raw gap B(2.6) - h*_A per RULE3_NOISES level, for the
    confirmed configuration). Returns (label, residual, reasons)."""
    confirmed = [n for n, o in (('3', c3), ('1', c1)) if o == 'confirmed']
    ha, hb = real.get('h_star_a'), real.get('h_star_b')
    if offset_corr and c1 == 'confirmed' and ha is not None and hb is not None:
        ha, hb = ha - offset_corr[0], hb - offset_corr[1]
    resid = None if ha is None or hb is None else hb - ha
    agree = resid is not None and abs(resid) <= EXPLAINED_AGREE_M
    repro = bool(confirmed) and bool(sim_raw_gaps) and all(
        v is not None and abs(v - g) <= EXPLAINED_REPRODUCE_M for v in sim_raw_gaps)
    reasons = [f'confirmed mechanisms: {", ".join("candidate " + c for c in confirmed) or "none"}',
               f'corrected fixed points: h*_A {_fmt(ha)}, h*_B {_fmt(hb)}, residual '
               f'{_fmt(resid, "+.3f")} m (explained needs |.| <= {EXPLAINED_AGREE_M})',
               'simulated raw gap vs G = ' + _fmt(g) + ': '
               + (', '.join(_fmt(v, '+.3f') for v in sim_raw_gaps) or 'n/a')
               + f' (needs within {EXPLAINED_REPRODUCE_M})']
    if confirmed and agree and repro:
        return 'explained', resid, reasons
    return 'partially explained', resid, reasons


def _num(v):
    try:
        return None if v in (None, '', 'None') else float(v)
    except (TypeError, ValueError):
        return v


def _numrows(rows):
    return [{k: _num(v) if k not in ('city', 'results', 'grouping', 'group', 'rig', 'refs')
             else v for k, v in r.items()} for r in rows]


def sim_mean(rows, group, h_true, noise, offset=0.0):
    """Seed-averaged simulation summary for one configuration, or None."""
    rs = [r for r in rows if r['group'] == group and abs(r['h_true'] - h_true) < 1e-9
          and abs(r['noise_scale'] - noise) < 1e-9 and abs(r['offset_px'] - offset) < 1e-9]
    if not rs:
        return None
    out = {'n_seeds': len(rs)}
    for k in ('h_star_a', 'a_slope', 'b_2p6', 'h_star_b', 'b_slope', 'gap_raw', 'gap_fixed',
              'b_2p6_null_sd_m', 'n_on_rig', 'n_no_site', 'n_dets_out'):
        vals = [r[k] for r in rs if isinstance(r.get(k), float)]
        out[k] = statistics.fmean(vals) if len(vals) == len(rs) else None
        out[k + '_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
    return out


def run_verdict(args):
    """The `verdict` subcommand: apply the rules, write report.md."""
    city = args.cities[0]
    group = args.group
    gap_dir = args.run_root / city / 'camera_height' / 'gap'
    sweep_rows = _numrows(_read_csv(gap_dir / 'sweep.csv'))
    seq_rows = _numrows(_read_csv(gap_dir / 'sequences.csv'))
    sim_rows = _numrows(_read_csv(gap_dir / 'simulate.csv'))
    off_rows = _numrows(_read_csv(gap_dir / 'gt_offset.csv'))
    c_rows = _numrows(_read_csv(gap_dir / 'instrument_c.csv'))
    summ = args.run_root / '_summary' / 'camera_height'
    five = _numrows(_read_csv(summ / 'gap_sweep.csv'))
    real = next(r for r in sweep_rows if r['group'] == group and r['grouping'] == 'rig')
    g = real['b_2p6'] - real['h_star_a']
    L = []   # report lines
    L += [f'# Height gap, {city} {group} (issue #87)', '',
          'Generated by `scripts/height_gap.py verdict` from the CSVs beside this file; the '
          'rules are the pre-registered ones in the script (RULE_* constants).', '',
          f'G = B(2.6) - h*_A = {real["b_2p6"]:.3f} - {real["h_star_a"]:.3f} = **{g:.3f} m**.', '']

    # E1
    L += ['## E1: B swept over association heights', '',
          '| group | h*_A (95% CI) | A slope | B(2.6) | null SD at 2.6 (m) | b_B | h*_B (95% CI) '
          '| h*_B − h*_A | B(2.6) − h*_A |', '|---|---|---:|---:|---:|---:|---|---:|---:|']
    for r in sweep_rows:
        L.append(f"| {r['group']} | {_fmt(r['h_star_a'])} ({_fmt(r['h_star_a_lo'])}–"
                 f"{_fmt(r['h_star_a_hi'])}) | {_fmt(r['a_slope'])} | {_fmt(r['b_2p6'])} | "
                 f"{_fmt(r['b_2p6_null_sd_m'])} | {_fmt(r['b_slope'])} | {_fmt(r['h_star_b'])} "
                 f"({_fmt(r['h_star_b_lo'])}–{_fmt(r['h_star_b_hi'])}) | "
                 f"{_fmt(r['gap_fixed'], '+.3f')} | {_fmt(r['gap_raw'], '+.3f')} |")
    hs = [h for h in SWEEP_HEIGHTS if f'b_at_{h:g}' in real]
    L += ['', f'B(h) for {group}: ' + ', '.join(f"{h:g} m → {_fmt(real[f'b_at_{h:g}'])}"
                                                 for h in hs), '']
    if five:
        L += ['### Every rig group, five cities (rule 3 with h*_B)', '',
              '| city / group | h*_A | A slope | B(2.6) | b_B | h*_B | |h*_B − h*_A| ≤ 0.25 | '
              '|B(2.6) − h*_A| ≤ 0.25 (#53) |', '|---|---:|---:|---:|---:|---:|---|---|']
        for r in five:
            if r['h_star_a'] is None and r['h_star_b'] is None:
                continue
            fx = r['gap_fixed']
            rw = r['gap_raw']
            L.append(f"| {r['city']} / {r['group']} | {_fmt(r['h_star_a'])} | "
                     f"{_fmt(r['a_slope'])} | {_fmt(r['b_2p6'])} | {_fmt(r['b_slope'])} | "
                     f"{_fmt(r['h_star_b'])} | "
                     f"{'n/a' if fx is None else ('pass' if abs(fx) <= 0.25 else 'fail')} "
                     f"({_fmt(fx, '+.3f')}) | "
                     f"{'n/a' if rw is None else ('pass' if abs(rw) <= 0.25 else 'fail')} "
                     f"({_fmt(rw, '+.3f')}) |")
        L.append('')

    # E2
    L += ['## E2: simulation on the real view graph', '',
          '| h_true | noise | seeds | h*_A (slope) | B(2.6) | h*_B (b_B) | B(2.6) − h_true | '
          'h*_B − h_true | h*_A − h_true |', '|---:|---:|---:|---|---:|---|---:|---:|---:|']
    configs = sorted({(r['h_true'], r['noise_scale']) for r in sim_rows
                      if r['group'] == group and r['offset_px'] == 0.0})
    sims = {}
    for ht, n in configs:
        m = sim_mean(sim_rows, group, ht, n)
        sims[ht, n] = m
        L.append(f"| {ht:g} | {n:g} | {m['n_seeds']} | {_fmt(m['h_star_a'])} "
                 f"({_fmt(m['a_slope'])}) | {_fmt(m['b_2p6'])} | {_fmt(m['h_star_b'])} "
                 f"({_fmt(m['b_slope'])}) | {_fmt(m['b_2p6'] - ht if m['b_2p6'] is not None else None, '+.3f')} | "
                 f"{_fmt(m['h_star_b'] - ht if m['h_star_b'] is not None else None, '+.3f')} | "
                 f"{_fmt(m['h_star_a'] - ht if m['h_star_a'] is not None else None, '+.3f')} |")
    L.append('')
    sim3 = {n: sims.get((RULE3_SIM_TRUE_M, n)) for n in RULE3_NOISES}
    c3, _share, c3_reasons = verdict_candidate3(real, {n: s for n, s in sim3.items() if s})
    share = candidate3_share(real, g)

    # E3
    L += ['## E3: vertical peak offset', '', '### E3a: box minus peak (benchmark boxes)', '',
          '| rig | boxes | panos | median dy px (95% CI) | eps = peak − box (95% CI) |',
          '|---|---:|---:|---|---|']
    for r in off_rows:
        L.append(f"| {r['rig']} | {int(r['n_boxes'])} | {int(r['n_panos'])} | "
                 f"{_fmt(r['box_minus_peak_dy_px_median'], '+.2f')} ({_fmt(r['ci_lo'], '+.2f')}"
                 f" to {_fmt(r['ci_hi'], '+.2f')}) | {_fmt(r['eps_px_median'], '+.2f')} "
                 f"({_fmt(r['eps_px_lo'], '+.2f')} to {_fmt(r['eps_px_hi'], '+.2f')}) |")
    off = next((r for r in off_rows if r['rig'] == group), None)
    e3b = {}
    for r in sim_rows:
        if r['group'] == group and r['h_true'] == RULE3_SIM_TRUE_M and r['noise_scale'] == 0.5:
            e3b.setdefault(r['offset_px'], [])
    for e in sorted(e3b):
        e3b[e] = sim_mean(sim_rows, group, RULE3_SIM_TRUE_M, 0.5, e)
    L += ['', f'### E3b: simulation at h_true {RULE3_SIM_TRUE_M:g} m, noise 0.5', '',
          '| eps px | h*_A | h*_B | h*_B − h*_A | B(2.6) − h*_A |', '|---:|---:|---:|---:|---:|']
    for e, m in sorted(e3b.items()):
        L.append(f"| {e:+g} | {_fmt(m['h_star_a'])} | {_fmt(m['h_star_b'])} | "
                 f"{_fmt(m['gap_fixed'], '+.3f')} | {_fmt(m['gap_raw'], '+.3f')} |")
    L.append('')
    if off and len(e3b) >= 2:
        c1, c1_reasons = verdict_candidate1(
            (off['eps_px_lo'], off['eps_px_hi']),
            {e: m['gap_fixed'] for e, m in e3b.items() if m['gap_fixed'] is not None},
            direction=1.0 if g > 0 else -1.0)
    else:
        c1, c1_reasons = 'untested', ['E3a or E3b missing']

    # E4
    q = [r for r in seq_rows if r.get('quarter_support') in (True, 'True', 1.0)]
    for r in seq_rows:
        r['quarter_support'] = r.get('quarter_support') in (True, 'True', 1.0)
    gaps = [r['h_star_b'] - r['h_star_a'] for r in q
            if r['h_star_b'] is not None and r['h_star_a'] is not None]
    gaps_raw = [r['b_2p6'] - r['h_star_a'] for r in q
                if r['b_2p6'] is not None and r['h_star_a'] is not None]
    mix = mixture_arm(seq_rows, real) if seq_rows else {}
    L += ['## E4: per sequence (quarter support)', '',
          '| sequence | A panos | A sites | B views | h*_A (slope) | B(2.6) | h*_B (b_B) | '
          'h*_B − h*_A | B(2.6) − h*_A |', '|---|---:|---:|---:|---|---:|---|---:|---:|']
    for r in sorted(q, key=lambda r: -(r['b_views_2p6'] or 0)):
        L.append(f"| {r['group']} | {int(r['a_panos'] or 0)} | {int(r['a_sites'] or 0)} | "
                 f"{int(r['b_views_2p6'] or 0)} | {_fmt(r['h_star_a'])} ({_fmt(r['a_slope'])})"
                 f" | {_fmt(r['b_2p6'])} | {_fmt(r['h_star_b'])} ({_fmt(r['b_slope'])}) | "
                 f"{_fmt(r['gap_fixed'], '+.3f')} | {_fmt(r['gap_raw'], '+.3f')} |")
    L.append('')
    if gaps_raw:
        L.append(f'Median within-sequence raw gap B(2.6) − h*_A: '
                 f'{statistics.median(gaps_raw):+.3f} m over {len(gaps_raw)} sequences.')
    for name, (m, closure, n) in mix.items():
        L.append(f'Mixture arm ({name} sequences, n = {n}): view-weighted mean of h*_A '
                 f'{_fmt(m)} m against pooled h*_A {real["h_star_a"]:.3f} and B(2.6) '
                 f'{real["b_2p6"]:.3f}: closes {_fmt(closure, ".0%")} of G.')
    L.append('')
    c2, c2_med, c2_reasons = verdict_candidate2(
        gaps, mix.get('qualifying', (None, None, 0))[1], 1.0 if g > 0 else -1.0)

    # E5
    L += ['## E5: instrument C on the group\'s references', '',
          '| height | refs | n | sites | slope (SE) |', '|---:|---|---:|---:|---|']
    zc = {}
    for r in c_rows:
        if r['height_m'] == 'zero_crossing':
            zc[r['refs']] = r
            continue
        L.append(f"| {r['height_m']:g} | {r['refs']} | {int(r['n'])} | {int(r['n_sites'])} | "
                 f"{_fmt(r['slope'], '+.3f')} ({_fmt(r['slope_se'], '.3f')}) |")
    for name, r in zc.items():
        L.append(f"\nZero crossing ({name} refs): {_fmt(r.get('zero_crossing_m'))} m "
                 f"(SE {_fmt(r.get('zero_crossing_se_m'))}, d slope / dh {_fmt(r['slope'], '+.3f')})")
    c26 = next((r for r in c_rows if r['refs'] == 'all' and r['height_m'] == H0), None)
    e5_dec, e5_fav, e5_text = verdict_e5(c26 and c26['slope'], c26 and c26['slope_se'])
    L.append('')

    # E6
    lau = _numrows(_read_csv(args.run_root / 'laurens' / 'camera_height' / 'gap' / 'sweep.csv'))
    lau_sim = _numrows(_read_csv(args.run_root / 'laurens' / 'camera_height' / 'gap'
                                 / 'simulate.csv'))
    pat = _numrows(_read_csv(summ / 'gap_paterson_year.csv'))
    if lau or lau_sim or pat:
        L += ['## E6: second cases (report only)', '']
    if lau:
        L += ['| Laurens file / group | h*_A (slope) | B(2.6) | h*_B (b_B) |', '|---|---|---:|---|']
        L += [f"| {r['results']} / {r['group']} | {_fmt(r['h_star_a'])} ({_fmt(r['a_slope'])}) | "
              f"{_fmt(r['b_2p6'])} | {_fmt(r['h_star_b'])} ({_fmt(r['b_slope'])}) |" for r in lau]
        L.append('')
    if lau_sim:
        L += ['| Laurens sim h_true | noise | h*_A (A slope) | B(2.6) | h*_B (b_B) |',
              '|---:|---:|---|---:|---|']
        for ht, n in sorted({(r['h_true'], r['noise_scale']) for r in lau_sim}):
            m = sim_mean(lau_sim, lau_sim[0]['group'], ht, n)
            L.append(f"| {ht:g} | {n:g} | {_fmt(m['h_star_a'])} ({_fmt(m['a_slope'])}) | "
                     f"{_fmt(m['b_2p6'])} | {_fmt(m['h_star_b'])} ({_fmt(m['b_slope'])}) |")
        L.append('')
    if pat:
        L += ['| Paterson year | h*_A (slope) | B(2.6) | h*_B (b_B) | depth median |',
              '|---|---|---:|---|---:|']
        L += [f"| {r['group']} | {_fmt(r['h_star_a'])} ({_fmt(r['a_slope'])}) | "
              f"{_fmt(r['b_2p6'])} | {_fmt(r['h_star_b'])} ({_fmt(r['b_slope'])}) | "
              f"{_fmt(r.get('depth_median_m'))} |" for r in pat]
        L.append('')

    # Verdict
    offset_corr = None
    sim_gaps = []
    if c3 == 'confirmed' or c1 == 'confirmed':
        for n in RULE3_NOISES:
            s = sim3.get(n)
            if c1 == 'confirmed' and off:
                s = sim_mean(sim_rows, group, RULE3_SIM_TRUE_M, n,
                             min(e3b, key=lambda e: abs(e - off['eps_px_median'])))
            sim_gaps.append(None if not s else s['gap_raw'])
        if c1 == 'confirmed' and off:
            e = min(e3b, key=lambda e: abs(e - off['eps_px_median']))
            m0, me = e3b[0.0], e3b[e]
            offset_corr = (me['h_star_a'] - m0['h_star_a'], me['h_star_b'] - m0['h_star_b'])
    label, resid, ex_reasons = verdict_explained(real, c3, c1, sim_gaps, g, offset_corr)
    L += ['## Verdict (pre-registered rules)', '',
          f'- **Candidate 3 (association pull): {c3}**'
          + (f' (share = 1 − |h*_B − h*_A| / G = {share:.2f})' if c3 == 'partial'
             and share is not None else '')]
    L += [f'  - {x}' for x in c3_reasons]
    L += [f'- **Candidate 1 (vertical peak offset): {c1}**'] + [f'  - {x}' for x in c1_reasons]
    L += [f'- **Candidate 2 (mixed mountings): {c2}**'] + [f'  - {x}' for x in c2_reasons]
    L += [f'- **E5: {"decisive" if e5_dec else "not decisive"}** — {e5_text}']
    L += [f'- **Overall: {label}**'] + [f'  - {x}' for x in ex_reasons]
    if label != 'explained':
        L.append(f'  - residual quoted: {_fmt(resid, "+.3f")} m between the '
                 'fixed points; #87 stays open on it.')
    L.append('')
    out = gap_dir / 'report.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    print('\n'.join(L[-(len(c3_reasons) + len(c1_reasons) + len(c2_reasons) + 12):]))
    print(f'wrote {out}')


# --- CLI ----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=('sweep', 'simulate', 'gt', 'verdict'))
    ap.add_argument('cities', nargs='+')
    ap.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--results', nargs='+', default=['results.jsonl'],
                    help='results file(s) in the run dir (sweep takes several)')
    ap.add_argument('--group', default='gopro/max', help='the rig group under study')
    ap.add_argument('--group-by', choices=('rig', 'year'), default='rig',
                    help='sweep grouping; year for a GSV run (with depth medians)')
    ap.add_argument('--sequences', action='store_true',
                    help="sweep: also fit --group's sequences as their own groups (E4)")
    ap.add_argument('--null-seeds', type=int, default=NULL_SEEDS)
    ap.add_argument('--n-boot', type=int, default=mh.N_BOOT)
    ap.add_argument('--true-height', type=float, nargs='+', default=[2.0])
    ap.add_argument('--noise-scale', type=float, nargs='+', default=[0.5])
    ap.add_argument('--offset-px', type=float, nargs='+', default=[0.0])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--heights', type=float, nargs='+', default=[1.98, 2.2, 2.38, 2.6])
    ap.add_argument('--out', type=Path, default=None, help='output CSV path override')
    args = ap.parse_args(argv)
    {'sweep': run_sweep, 'simulate': run_simulate, 'gt': run_gt,
     'verdict': run_verdict}[args.command](args)


if __name__ == '__main__':
    main()
