"""Per-rig camera height for crowdsourced imagery (issue #53).

Mapillary serves no depth, so every Mapillary raycast uses geo.DEFAULT_CAMERA_HEIGHT_M
(2.6 m) whatever carried the camera: a car-roof rig, a bicycle helmet or a hand-held
pole. The only height evidence is the imagery's own geometry, and the repo already has
two independent instruments for it; this script runs both per capture-rig group, applies
a decision rule fixed before any number was produced, and writes the result as a table
that fusion can opt into (`fuse_sites.py --camera-height-m per-rig`).

Groupings (each reported; the table's grain is the rig class):
    rig       normalised (make, model) from pano.source_metadata: case-folded, the make
              dropped from the model, firmware tokens dropped (rig_key), so
              'GoPro Fusion FS1.04.01.80.00' and 'Fusion' are one class
    creator   the Mapillary contributor
    sequence  the capture sequence
    speed     rig class split by the sequence's median speed between consecutive frames:
              walk < 2.5 m/s, mixed 2.5-7, drive > 7, unknown without timestamps

Instrument A -- bearing-only fixed point (fuse_sites.implied_heights, #68). Two views of
one site fix the ramp from bearings alone, so each view implies a height range*tan(dip)
independent of the height model -- but WHICH pairs exist depends on the association
height. So the whole run is associated at each height in SWEEP_HEIGHTS, the group's median
implied height is fitted as implied = a + b*h, and the fixed point is h* = a / (1 - b).
The slope b is the instrument's own blind spot: at b = 1 the implied height merely follows
the association height and says nothing. 95% CI by bootstrap over sites.

Instrument B -- the leave-one-out scale identity (reprojection_residual.fit_scale, #76).
Fused at 2.6 m, each held-out view's residual is s*g_i with g_i from data alone; the joint
fit takes one s per group (groups sharing sites are separated), minus the same fit on the
error model's own simulated noise (the errors-in-variables null, here per group), gives the
range scale k = 1/(1 - (s - s_null)) and h_B = 2.6 / k. The scale-plus-dip-offset fit
(range_offset_levers) is reported beside it, since #76 found an offset moves k.

Instrument B as a fixed point (#89). Read at one association height, B is pulled toward
that height (#87, docs section 7). So B is read at every height in SWEEP_HEIGHTS_B, with
the null averaged over NULL_SEEDS seeds (b_at), and two estimators of its fixed point are
always reported (b_estimates): the line B(h) = a_B + b_B h and a_B / (1 - b_B), with the
amplification 1/(1 - b_B) beside it, and the local crossing of B(h) = h between the
bracketing swept heights. Which one rule 3 reads is decided per city by the estimator
gate: `--validate` re-synthesizes the city's real view graph with every pano at a known
height (height_gap.resynthesize), sweeps B with a noise-matched null, and rule V (the
VALID_* constants, pre-registered on #89) keeps an estimator only if it recovers the
planted height within 0.10 m in every cell. Neither valid -> rule 3 reads b_unvalidated.

Instrument C -- GT-anchored (the five judged splits): the slope of the along-ray residual
of reviewer references (reprojection_residual.gt_anchored_rows, left-out variant) on the
predicted range. A correct height flattens it.

Decision rule (pre-registered on #53, fixed before the numbers): see decide_group and the
RULE_* constants. The table then faces the production gate (height_gate / gate_verdict) on
the common site set, as eval_sites --pose-precondition does. Whatever the gate says, the
default height stays 2.6 m; `recommended` in the table records the verdict.

Fusion parameters. Instruments A and B run at the production tier (FuseParams defaults,
pose explicitly off): the table feeds production fusion. Instrument C and the gate run at
the benchmark tier with mask_rig off, the parameters every judged bundle is keyed to
(detectors.BENCHMARK_CONFIDENCE; eval_sites pins the same). Every raycast here is flat:
panos are handed to the GT code with pitch/roll stripped, because
reprojection_residual.gt_anchored_rows passes FuseParams.apply_pose -- a non-empty string,
so truthy -- straight to geo.detection_ground_point, which would rotate any posed pano.

No network, no GPU, no writes outside the output directories.

Usage:
    # the estimator gate first (#89; simulation, ~15 min on 10 workers, resumable)
    python scripts/mapillary_height.py --validate richmond laurens clovis morgantown annapolis
    # then all five Mapillary runs, the gate, and the tables (the #53 measurement)
    python scripts/mapillary_height.py richmond laurens clovis morgantown annapolis
    python scripts/mapillary_height.py richmond --run-root D:/Git/sidewalk-auto-labeler/runs
    # one run, no gate (the table is written with recommended: false)
    python scripts/mapillary_height.py richmond --no-gate
    # then, opt-in:
    python scripts/fuse_sites.py runs/richmond --camera-height-m per-rig --out /tmp/s.jsonl

Findings: docs/mapillary-camera-height.md.
"""
import argparse
import csv
import datetime
import json
import math
import random
import re
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
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

SWEEP_HEIGHTS = (1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0)
# Instrument B's sweep (#89): #53's heights plus two above 3.0 m, so a rig anywhere in
# SANE_HEIGHT_M is bracketed rather than extrapolated. A keeps SWEEP_HEIGHTS.
SWEEP_HEIGHTS_B = (1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.4, 3.8)
NULL_SEEDS = 10           # B's errors-in-variables null is averaged over this many seeds
N_DRAWS = 500             # parametric draws for h*_B's CI
DRAW_SEED = 87
SUPPORT_AT_M = 2.6        # instrument A's support counts are taken at this association
N_BOOT = 200
BOOT_SEED = 53
WALK_MPS, DRIVE_MPS = 2.5, 7.0
SPEED_MAX_GAP_S = 60.0    # consecutive frames further apart than this are a pause
SANE_HEIGHT_M = (1.0, 3.5)  # outside: reported as suspect whatever the instruments say
# Split names that differ from the run directory's (reprojection_residual's convention).
SPLITS = dict(rr.DEFAULT_SPLITS)

# --- The pre-registered rule (#53 plan section 3; fixed before any number) ------------
RULE_MIN_PANOS_A = 100    # 1. support: panos with an implied height (instrument A) ...
RULE_MIN_SITES_A = 50     # ... sites contributing them ...
RULE_MIN_VIEWS_B = 300    # ... and held-out views (instrument B)
RULE_MAX_SLOPE = 0.6      # 2. identifiable: d(implied)/d(association) <= this
RULE_MAX_DISAGREE_M = 0.25  # 3. agreement: |h*_A - h_B| <= this; h_g = their mean
RULE_MIN_EFFECT_M = 0.2   # 4. material: CI on h*_A excludes 2.6 and |h_g - 2.6| > this
SEQ_IQR_M = 0.4           # a rig class whose qualifying sequences' h* IQR exceeds this ...
SEQ_SUPPORT_FRACTION = 0.25  # ... (qualifying = rule 1 at a quarter of the counts) goes
                             # per sequence; sequences failing rule 1 inherit the rig value
# The production gate, per-rig vs off (2.6 m) on the common site set:
GATE_CHANGED_FRAC = 0.10  # a city "changes" when the table moves >= 10% of its panos
GATE_P90_TOL_M = 0.1      # (i) p90 GT-to-site no worse by more than this in a changed city
GATE_MIN_MEDIAN_WINS = 3  # (ii) median better in at least this many changed cities
GATE_MAX_RECALL_DROP = 0.010  # (iii) off-pool recall@2.5 m drop, any city
# (iv) instrument C's |slope| not larger in any changed city.

# --- The estimator gate (#89 plan section 2; fixed before any number) -----------------
# Each h*_B estimator is validated per city on the city's real view graph, every group
# planted at the same h_true (validation_cell). Rule V: VALID iff over every cell the
# |mean error over seeds| <= VALID_MAX_ERR_M and no cell is undefined or extrapolated.
VALID_TRUE_HEIGHTS = (1.8, 2.2, 2.6, 3.0)
VALID_NOISES = (0.5, 1.0)
VALID_SEEDS = 2
VALID_MAX_ERR_M = 0.10
VALID_BUDGET_S = 15 * 60  # one real sweep longer than this -> 1 seed per cell
EST_LINE, EST_LOCAL = 'line', 'local'
B_UNVALIDATED = 'b_unvalidated'

ARM_OFF, ARM_RIG = 'off', 'per-rig'


# --- Groups ---------------------------------------------------------------------------

_FIRMWARE = re.compile(r'^[a-z]{0,3}\d+(\.\d+){2,}$')


def rig_key(make, model):
    """Normalised 'make/model' rig class: case-folded, the make's words dropped from the
    model, firmware tokens (FS1.04.01.80.00) dropped; 'unknown' for a missing/'none' make.

    Example:
        >>> rig_key('GoPro', 'GoPro Fusion FS1.04.01.80.00'), rig_key('GoPro', 'Fusion')
        ('gopro/fusion', 'gopro/fusion')
        >>> rig_key('Trimble', 'Trimble mx7'), rig_key('none', 'none')
        ('trimble/mx7', 'unknown')
    """
    make = (make or '').strip().casefold()
    if make in ('', 'none', 'null', 'unknown'):
        return 'unknown'
    make_words = set(make.split())
    words = [w for w in (model or '').strip().casefold().split()
             if w not in make_words and not _FIRMWARE.match(w)]
    return f"{make.split()[0]}/{' '.join(words) or '?'}"


def speed_class(mps):
    if mps is None:
        return 'unknown'
    return 'walk' if mps < WALK_MPS else 'drive' if mps > DRIVE_MPS else 'mixed'


def sequence_speeds(frames):
    """{sequence: median m/s between consecutive kept frames} from (seq, t_ms, lat, lng);
    frames without a timestamp are ignored, gaps over SPEED_MAX_GAP_S are pauses."""
    by_seq = {}
    for seq, t, lat, lng in frames:
        if seq is not None and t is not None:
            by_seq.setdefault(seq, []).append((t, lat, lng))
    out = {}
    for seq, fr in by_seq.items():
        fr.sort()
        v = [geo.haversine_m(a[1], a[2], b[1], b[2]) / ((b[0] - a[0]) / 1000.0)
             for a, b in zip(fr, fr[1:]) if 0 < (b[0] - a[0]) / 1000.0 <= SPEED_MAX_GAP_S]
        if v:
            out[seq] = statistics.median(v)
    return out


def pano_groups(jsonl):
    """{pano_id: {'rig', 'creator', 'sequence', 'speed'}} read from the raw records
    (load_results drops source_metadata). 'speed' is '<rig>|<class>'."""
    raw, frames = {}, []
    with open(jsonl, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)['pano']
            m = p.get('source_metadata') or {}
            creator = m.get('creator')
            if isinstance(creator, dict):
                creator = creator.get('username')
            seq = p.get('sequence_id') or m.get('sequence')
            raw[p['panorama_id']] = (rig_key(m.get('make'), m.get('model')),
                                     creator or 'unknown', seq)
            frames.append((seq, m.get('captured_at'), p.get('lat'), p.get('lng')))
    speeds = sequence_speeds(f for f in frames if f[2] is not None)
    return {pid: {'rig': rig, 'creator': creator, 'sequence': seq or 'none',
                  'speed': f'{rig}|{speed_class(speeds.get(seq))}'}
            for pid, (rig, creator, seq) in raw.items()}


# --- Instrument A -----------------------------------------------------------------------

def production_params(**kw):
    """Instruments A and B: production fusion, pose explicitly off."""
    return fs.FuseParams(apply_pose=fs.POSE_OFF, **kw)


def sweep_implied(panos, params, heights=SWEEP_HEIGHTS):
    """{h: [(site_id, pano_id, implied height)]}: the run associated at each height and
    every operational pair triangulated (fuse_sites.implied_heights, site by site)."""
    by_id = {p.pano_id: p for p in panos}
    out = {}
    for h in heights:
        sites, frame, _ = fs.fuse(panos, replace(params, camera_height_m=h))
        recs = []
        for site in sites:
            if site.n_operational < 2:
                continue
            for pid, vals in fs.implied_heights([site], frame, by_id).items():
                recs.extend((site.id, pid, v) for v in vals)
        out[h] = recs
    return out


def _fit_line(xs, ys):
    """(a, b) of y = a + b x by least squares, or (None, None) with < 3 points."""
    if len(xs) < 3:
        return None, None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def fixed_point(heights, medians):
    """(h*, a, b) of the line through (association height, median implied height);
    h* = a / (1 - b), None when b >= 1 (no fixed point: the instrument only follows)."""
    pts = [(h, m) for h, m in zip(heights, medians) if m is not None]
    a, b = _fit_line([h for h, _ in pts], [m for _, m in pts])
    if a is None or b >= 1:
        return None, a, b
    return a / (1 - b), a, b


def _group_median(entries):
    """Median over panos of each pano's median implied height; entries = [(pano, h)]."""
    per = {}
    for pid, v in entries:
        per.setdefault(pid, []).append(v)
    return statistics.median(statistics.median(v) for v in per.values()) if per else None


def weighted_median(pairs):
    """Median of [(value, weight)] with positive weights: the value where cumulative
    weight crosses half the total, averaging the two neighbours on an exact tie -- so
    with every weight 1 it equals statistics.median."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    cum = 0.0
    for i, (v, w) in enumerate(pairs):
        cum += w
        if abs(cum - total / 2) < 1e-9 * total and i + 1 < len(pairs):
            return (v + pairs[i + 1][0]) / 2
        if cum > total / 2:
            return v
    return pairs[-1][0]


def resampled_median(site_list, picks, full_counts):
    """Group median for one bootstrap draw, keeping draw multiplicity (#53 review).

    `site_list` is [(site_id, [(pano, v)])], `picks` the drawn site indices (with
    replacement) and `full_counts` {pano: its number of entries over ALL sites}. Each pano's
    median weights every value by how many times its site was drawn; the group median then
    weights each pano by (drawn entries, with multiplicity) / (its full entry count), whose
    expectation is 1. With every site drawn exactly once, both weights are 1 and this is
    the point estimate (_group_median of all entries)."""
    mult = {}
    for i in picks:
        mult[i] = mult.get(i, 0) + 1
    per = {}
    for i, m in mult.items():
        for pid, v in site_list[i][1]:
            per.setdefault(pid, []).append((v, m))
    if not per:
        return None
    return weighted_median([(weighted_median(vs), sum(w for _, w in vs) / full_counts[pid])
                            for pid, vs in per.items()])


def instrument_a(sweep, key_of, n_boot=N_BOOT, seed=BOOT_SEED, boot_keys=None):
    """Per group: medians over the sweep, the fit, h*, and (for boot_keys, default all) a
    bootstrap 95% CI resampling sites independently at each association height.

    key_of: pano_id -> group key. Returns {key: row dict}."""
    heights = sorted(sweep)
    by_group = {}           # key -> h -> site -> [(pano, v)]
    for h in heights:
        for sid, pid, v in sweep[h]:
            by_group.setdefault(key_of(pid), {}).setdefault(h, {}) \
                .setdefault(sid, []).append((pid, v))
    rng = random.Random(seed)
    out = {}
    for key in sorted(by_group, key=str):
        per_h = by_group[key]
        meds = [_group_median([e for es_ in per_h.get(h, {}).values() for e in es_])
                for h in heights]
        h_star, a, b = fixed_point(heights, meds)
        sup = per_h.get(SUPPORT_AT_M, {})
        row = {'group': key, 'h_star': h_star, 'a': a, 'slope': b,
               'n_panos': len({pid for es_ in sup.values() for pid, _ in es_}),
               'n_sites': len(sup), 'ci_lo': None, 'ci_hi': None, 'boot_sd': None,
               'boot_undefined': None}
        for h, m in zip(heights, meds):
            row[f'implied_at_{h:g}'] = m
        if h_star is not None and (boot_keys is None or key in boot_keys) and n_boot:
            draws, undefined = [], 0
            site_lists = {h: list(per_h.get(h, {}).items()) for h in heights}
            full_counts = {h: {} for h in heights}   # entries per pano over all sites
            for h in heights:
                for es_ in per_h.get(h, {}).values():
                    for pid, _v in es_:
                        full_counts[h][pid] = full_counts[h].get(pid, 0) + 1
            for _ in range(n_boot):
                bm = []
                for h in heights:
                    sl = site_lists[h]
                    if not sl:
                        bm.append(None)
                        continue
                    picks = [rng.randrange(len(sl)) for _ in sl]
                    bm.append(resampled_median(sl, picks, full_counts[h]))
                hs, _, _ = fixed_point(heights, bm)
                if hs is None:
                    undefined += 1
                else:
                    draws.append(hs)
            if draws:
                draws.sort()
                row['ci_lo'] = draws[int(0.025 * (len(draws) - 1))]
                row['ci_hi'] = draws[int(math.ceil(0.975 * (len(draws) - 1)))]
                row['boot_sd'] = statistics.pstdev(draws)
            row['boot_undefined'] = undefined
        out[key] = row
    return out


# --- Instrument B -----------------------------------------------------------------------

def site_views(panos, params):
    """(SiteViews with >= rr.MIN_VIEWS views, frame) fused under params."""
    sites, frame, _ = fs.fuse(panos, params)
    svs = [rr.views_from_site(s) for s in sites]
    return [sv for sv in svs if len(sv.views) >= rr.MIN_VIEWS], frame


def _null_views(sv, rng, noise_scale=1.0):
    """One site's views re-drawn under the error model's own noise and NO scale error
    (reprojection_residual.null_study's simulation, verbatim in effect).

    noise_scale multiplies every sigma drawn (along, cross, GPS) -- #89's noise-matched
    null: a simulation that injects noise at scale k must be corrected by a null drawn at
    k, not at the model's full sigmas. At 0 every view is re-drawn exactly at the site
    (zero residual, so the null fits s = 0 and corrects nothing); the draws are the same
    standard normals whatever the scale, so at 0.5 every displacement is exactly half."""
    sim = []
    for v in sv.views:
        ue, un = v.unit
        ce, cn = v.e - v.range_m * ue, v.n - v.range_m * un
        te, tn = sv.e - ce, sv.n - cn
        r0 = math.hypot(te, tn)
        if r0 < 1e-6:
            continue
        u0 = (te / r0, tn / r0)
        a = rng.gauss(0.0, v.sigma_along_m * noise_scale)
        c = rng.gauss(0.0, v.sigma_cross_m * noise_scale)
        gps = geo.error_model_for(v.source).sigma_gps_m * noise_scale
        ge, gn = rng.gauss(0.0, gps), rng.gauss(0.0, gps)
        pe = sv.e + a * u0[0] + c * u0[1] + ge
        pn = sv.n + a * u0[1] - c * u0[0] + gn
        de, dn = pe - (ce + ge), pn - (cn + gn)
        sim.append(rr.View(v.pano_id, v.det_index, v.x, v.y, v.conf, pe, pn, v.cov,
                           math.hypot(de, dn), math.degrees(math.atan2(de, dn)) % 360,
                           v.source, v.capture_date))
    return sim


def _scale_blocks(sites, key_of, names, offset=False, simulate=None,
                  h0=geo.DEFAULT_CAMERA_HEIGHT_M):
    """fit_scale blocks for a joint per-group scale fit (plus a pooled dip-offset column
    when offset=True). key_of maps a view's pano to a name in `names`. h0 is the height
    the sites were associated (fused) at, which the dip-offset lever needs (#87)."""
    blocks = []
    for sv in sites:
        views = simulate(sv) if simulate else sv.views
        if len(views) < rr.MIN_VIEWS:
            continue
        loo = rr.leave_one_out(views)
        groups = [key_of(v.pano_id) for v in views]
        if offset:
            idx = {g: k for k, g in enumerate(names)}
            levers = []
            for v, g, (sc, off) in zip(views, groups, rr.range_offset_levers(
                    views, [h0] * len(views))):
                cols = [(0.0, 0.0)] * len(names)
                cols[idx[g]] = sc
                levers.append(cols + [off])
            design = rr.lever_design(views, levers)
        else:
            design = rr.scale_design(views, groups, names)
        blocks.append([(cols, (v.e - he, v.n - hn))
                       for v, ((he, hn), _), cols in zip(views, loo, design)])
    return blocks


def instrument_b(sites, key_of, min_rows=rr.MIN_GROUP_ROWS, seed=0,
                 h0=geo.DEFAULT_CAMERA_HEIGHT_M, noise_scale=1.0):
    """Joint per-group range scale, null-corrected per group. Groups with fewer than
    min_rows held-out views are fitted together as 'other'. Returns {key: row}.

    h0 is the association height `sites` were fused at (default 2.6 m, the #53
    measurement); h_B = h0 * (1 - (s - s_null)) is written against it, so a sweep over
    association heights (scripts/height_gap.py, #87) reads B(h) at each. One null seed;
    b_at averages NULL_SEEDS. noise_scale scales the null's sigmas (_null_views)."""
    counts = {}
    for sv in sites:
        for v in sv.views:
            counts[key_of(v.pano_id)] = counts.get(key_of(v.pano_id), 0) + 1
    names = sorted((g for g, c in counts.items() if c >= min_rows), key=str)
    if len(counts) > len(names):
        names.append('other')
    if not names:
        return {}
    named = set(names)

    def kof(pid):
        g = key_of(pid)
        return g if g in named else 'other'

    fit = rr.fit_scale(_scale_blocks(sites, kof, names), names)
    rng = random.Random(seed)
    null = rr.fit_scale(_scale_blocks(sites, kof, names,
                                      simulate=lambda sv: _null_views(sv, rng, noise_scale)),
                        names)
    off = rr.fit_scale(_scale_blocks(sites, kof, names, offset=True, h0=h0),
                       names + ['__dip_offset__'])
    out = {}
    for g in names:
        if g not in fit:
            continue
        s, se = fit[g]
        s0 = null.get(g, (0.0, None))[0]
        net = s - s0
        row = {'group': g, 'n_views': sum(c for k, c in counts.items()
                                          if (k if k in named else 'other') == g),
               'scale_s': s, 'scale_s_se': se, 'null_s': s0,
               'k_net': None if net >= 1 else 1.0 / (1.0 - net),
               'h_scale': h0 * (1.0 - net),                 # = h0 / k_net
               'h_scale_lo': h0 * (1.0 - (net + 1.96 * se)),
               'h_scale_hi': h0 * (1.0 - (net - 1.96 * se)),
               'offset_fit_s': None, 'h_scale_with_offset': None}
        if g in off:
            row['offset_fit_s'] = off[g][0]
            row['h_scale_with_offset'] = h0 * (1.0 - (off[g][0] - s0))
        out[g] = row
    return out


def pooled_scale(sites):
    """(k_net, s, s_null) of the pooled scale fit -- the #76 tie-back number."""
    fit = rr.fit_scale(_scale_blocks(sites, lambda _p: 'all', ['all']), ['all'])
    if not fit:
        return None, None, None
    s = fit['all'][0]
    s0 = rr.null_scale(sites)[0]
    return 1.0 / (1.0 - (s - s0)), s, s0


# --- Instrument B as a fixed point (#87 helpers, moved here by #89) ---------------------

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


def b_at(sites, key_of, h0, null_seeds=range(NULL_SEEDS), min_rows=rr.MIN_GROUP_ROWS,
         noise_scale=1.0):
    """Instrument B at association height h0 with the null averaged over `null_seeds`.

    instrument_b gives the fit, the first seed's null and the offset fit; the remaining
    seeds refit the null only. noise_scale scales every null draw (#89: 1.0 on real data;
    a simulation passes the noise it injected). Returns {group: row} with h_b = h0 * (1 -
    (s - mean null)) and null_sd_m = h0 * SD(null s) -- the null's own contribution."""
    seeds = list(null_seeds)
    rows = instrument_b(sites, key_of, min_rows=min_rows, seed=seeds[0], h0=h0,
                        noise_scale=noise_scale)
    if not rows:
        return {}
    names, kof = _b_names(sites, key_of, min_rows)
    nulls = {g: [r['null_s']] for g, r in rows.items()}
    for seed in seeds[1:]:
        rng = random.Random(seed)
        fit = rr.fit_scale(_scale_blocks(sites, kof, names, h0=h0,
                                         simulate=lambda sv: _null_views(sv, rng,
                                                                         noise_scale)),
                           names)
        for g in nulls:
            if g not in fit:
                # a zero here would be averaged in silently as "no null bias" (#90 review)
                raise SystemExit(f'b_at: null seed {seed} at h0 {h0:g} fitted no scale for '
                                 f'group {g!r}, which the first seed did fit')
            nulls[g].append(fit[g][0])
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


def sweep_ab(panos, keyings, heights=SWEEP_HEIGHTS_B, a_heights=SWEEP_HEIGHTS,
             null_seeds=range(NULL_SEEDS), params=None, log=None,
             min_rows=rr.MIN_GROUP_ROWS, noise_scale=1.0):
    """Associate the run once at every height and read both instruments.

    B (b_at, per keying: {name: key_of(pano_id)}, fitted jointly per keying) is read at
    every height in `heights`; A's implied heights only at those also in `a_heights`
    (A keeps #53's sweep, so its numbers reproduce exactly; pass () to skip A).
    Returns {'a': {h: [(site_id, pano_id, implied)]}, 'b': {name: {h: {group: row}}}}."""
    params = params or production_params()
    by_id = {p.pano_id: p for p in panos}
    a_set = set(a_heights)
    a, b = {}, {name: {} for name in keyings}
    for h in sorted(set(heights) | a_set):
        t0 = time.time()
        sites, frame, _ = fs.fuse(panos, replace(params, camera_height_m=h))
        recs = []
        if h in a_set:
            for site in sites:
                if site.n_operational < 2:
                    continue
                for pid, vals in fs.implied_heights([site], frame, by_id).items():
                    recs.extend((site.id, pid, v) for v in vals)
            a[h] = recs
        if h in heights:
            svs = [sv for sv in (rr.views_from_site(s) for s in sites)
                   if len(sv.views) >= rr.MIN_VIEWS]
            for name, key_of in keyings.items():
                b[name][h] = b_at(svs, key_of, h, null_seeds, min_rows=min_rows,
                                  noise_scale=noise_scale)
        if log:
            log(f'  h={h:g}: {len(sites)} sites, {len(recs)} implied heights '
                f'({time.time() - t0:.1f} s)')
    return {'a': a, 'b': b}


def local_crossing(heights, values):
    """(h, extrapolated) where B(h) crosses the identity B = h, read locally.

    The line fit's fixed point a_B / (1 - b_B) amplifies any bias of the line near the
    crossing by 1/(1 - b_B) (about 3.4 at b_B = 0.7), and B(h) is concave, so a global
    line through the sweep's centroid overshoots a crossing near the top of the sweep.
    This reads the crossing off the two sweep heights that bracket it instead: the first
    adjacent pair where d = B(h) - h goes from >= 0 to < 0 (a stable fixed point),
    linearly interpolated. With no bracket inside the sweep it extrapolates the end
    segment on the side the crossing lies (the top pair when d > 0 everywhere, the bottom
    pair when d < 0 everywhere) and flags it; None when that segment's slope is >= 1 (B
    only follows, no crossing) or fewer than two heights have a value.

    Example: B = 1.548, 1.721, 1.870, 2.033, 2.197 at 1.4, 1.6, 1.8, 2.0, 2.3 m crosses
    between 2.0 (d +0.033) and 2.3 (d -0.103): 2.0 + 0.3 * 0.033 / 0.136 = 2.073."""
    pts = sorted((h, v) for h, v in zip(heights, values) if v is not None)
    if len(pts) < 2:
        return None, None
    d = [(h, v - h) for h, v in pts]
    for h, dv in d:
        if dv == 0.0:
            return h, False
    for (h0, d0), (h1, d1) in zip(d, d[1:]):
        if d0 > 0 > d1:
            return h0 + (h1 - h0) * d0 / (d0 - d1), False
    (h0, d0), (h1, d1) = (d[-2], d[-1]) if all(dv > 0 for _, dv in d) else (d[0], d[1])
    if d1 - d0 >= 0:          # B's local slope >= 1: no crossing on this segment's line
        return None, True
    return h0 + (h1 - h0) * d0 / (d0 - d1), True


def b_fixed_point(b_by_h, group, n_draws=N_DRAWS, seed=DRAW_SEED):
    """(h*_B, a_B, b_B, ci_lo, ci_hi, undefined draws) of the line through (h, B(h)).

    The CI draws s at each height from N(s, cluster-robust SE) independently (the plan's
    choice) and keeps the mean null fixed. The heights share data -- the same views are
    re-associated at every height -- so their errors are positively correlated, and a
    common-mode shift is exactly what 1/(1 - b_B) amplifies: the CI is too narrow and must
    be quoted with that caveat (#90 review). It also describes the line fit only, not
    local_crossing's estimate (b_estimates draws both)."""
    pts = [(h, rows[group]) for h, rows in sorted(b_by_h.items()) if group in rows]
    if len(pts) < 3:
        return None, None, None, None, None, None
    hs = [h for h, _ in pts]
    h_star, a, b = fixed_point(hs, [r['h_b'] for _, r in pts])
    rng = random.Random(seed)
    draws, undefined = [], 0
    for _ in range(n_draws):
        vals = [h * (1.0 - (rng.gauss(r['scale_s'], r['scale_s_se'] or 0.0) - r['null_s_mean']))
                for h, r in pts]
        hd, _, _ = fixed_point(hs, vals)
        if hd is None:
            undefined += 1
        else:
            draws.append(hd)
    lo, hi = _ci(draws)
    return h_star, a, b, lo, hi, undefined


def _ci(draws):
    """(2.5%, 97.5%) order statistics of `draws`, (None, None) when empty."""
    if not draws:
        return None, None
    draws = sorted(draws)
    return (draws[int(0.025 * (len(draws) - 1))],
            draws[int(math.ceil(0.975 * (len(draws) - 1)))])


def b_estimates(b_by_h, group, n_draws=N_DRAWS, seed=DRAW_SEED):
    """Both #89 estimators of instrument B's fixed point for one group, always reported.

    b_by_h: {h: {group: b_at row}} over the sweep. Returns a dict:
      h_b_line, a_b, b_b     the line B(h) = a_B + b_B h over every swept height and its
                             fixed point a_B / (1 - b_B); None when b_B >= 1 (undefined)
      amplification          1 / (1 - b_B): how much any bias of the line near the crossing
                             is magnified in h_b_line
      line_extrapolated      h_b_line outside the swept heights
      h_b_local, local_extrapolated   local_crossing over the same B(h)
      *_ci_lo / *_ci_hi      parametric draws of each height's s (b_fixed_point's scheme);
                             TOO NARROW -- the heights share views, so the common mode is
                             understated. line_draws_undefined counts draws with b_B >= 1.
      h_at_2p6, n_views, null_sd_m, h_at_2p6_se, h_at_2p6_offset   B read at 2.6 m (#53's
                             single reading) and its support
    With fewer than three heights holding the group, the line is None."""
    pts = [(h, rows[group]) for h, rows in sorted(b_by_h.items()) if group in rows]
    hs = [h for h, _ in pts]
    out = {'h_b_line': None, 'a_b': None, 'b_b': None, 'amplification': None,
           'line_extrapolated': None, 'line_ci_lo': None, 'line_ci_hi': None,
           'line_draws_undefined': None, 'h_b_local': None, 'local_extrapolated': None,
           'local_ci_lo': None, 'local_ci_hi': None}
    r26 = b_by_h.get(geo.DEFAULT_CAMERA_HEIGHT_M, {}).get(group)
    out.update({'h_at_2p6': r26['h_b'] if r26 else None,
                'h_at_2p6_se': r26['h_b_se'] if r26 else None,
                'h_at_2p6_offset': r26['h_b_offset'] if r26 else None,
                'n_views': r26['n_views'] if r26 else 0,
                'null_sd_m': r26['null_sd_m'] if r26 else None})
    for h, r in pts:
        out[f'b_at_{h:g}'] = r['h_b']
    if len(pts) >= 2:
        out['h_b_local'], out['local_extrapolated'] = local_crossing(
            hs, [r['h_b'] for _, r in pts])
    if len(pts) >= 3:
        h_star, a, b = fixed_point(hs, [r['h_b'] for _, r in pts])
        out.update({'h_b_line': h_star, 'a_b': a, 'b_b': b,
                    'amplification': None if b is None or b >= 1 else 1.0 / (1.0 - b),
                    'line_extrapolated': None if h_star is None
                    else not (min(hs) <= h_star <= max(hs))})
    if n_draws and len(pts) >= 2:
        rng = random.Random(seed)
        line, local, undefined = [], [], 0
        for _ in range(n_draws):
            vals = [h * (1.0 - (rng.gauss(r['scale_s'], r['scale_s_se'] or 0.0)
                                - r['null_s_mean'])) for h, r in pts]
            if len(pts) >= 3:
                hd = fixed_point(hs, vals)[0]
                if hd is None:
                    undefined += 1
                else:
                    line.append(hd)
            hl = local_crossing(hs, vals)[0]
            if hl is not None:
                local.append(hl)
        out['line_ci_lo'], out['line_ci_hi'] = _ci(line)
        out['local_ci_lo'], out['local_ci_hi'] = _ci(local)
        out['line_draws_undefined'] = undefined if len(pts) >= 3 else None
    return out


def estimator_value(est, estimator):
    """(h*_B, extrapolated) of the named estimator from a b_estimates dict."""
    if estimator == EST_LINE:
        return est['h_b_line'], est['line_extrapolated']
    if estimator == EST_LOCAL:
        return est['h_b_local'], est['local_extrapolated']
    raise ValueError(estimator)


def b_rows_from_sweep(b_by_h, estimator, validated, n_draws=N_DRAWS):
    """Instrument B rows for decide_group / the table from one keying's sweep.

    Each row carries both estimators (b_estimates) and the SELECTED one as h_star (None
    when the city validated neither: estimator None, validated False -> rule 3 reads
    b_unvalidated). #53's single reading B(2.6) stays as h_scale / h_at_2p6, now with the
    null averaged over NULL_SEEDS seeds."""
    groups = set()
    for rows in b_by_h.values():
        groups |= set(rows)
    out = {}
    for g in sorted(groups, key=str):
        est = b_estimates(b_by_h, g, n_draws=n_draws)
        h_star, extrap = (estimator_value(est, estimator) if validated
                          else (None, None))
        r26 = b_by_h.get(geo.DEFAULT_CAMERA_HEIGHT_M, {}).get(g)
        se = est['h_at_2p6_se']
        out[g] = {**est, 'group': g, 'estimator': estimator if validated else None,
                  'validated': bool(validated), 'h_star': h_star, 'extrapolated': extrap,
                  'h_scale': est['h_at_2p6'],
                  'h_scale_lo': None if se is None else est['h_at_2p6'] - 1.96 * se,
                  'h_scale_hi': None if se is None else est['h_at_2p6'] + 1.96 * se,
                  'h_scale_with_offset': est['h_at_2p6_offset'],
                  'null_s': r26['null_s_mean'] if r26 else None,
                  'k_net': None if not r26 or r26['h_b'] <= 0
                  else geo.DEFAULT_CAMERA_HEIGHT_M / r26['h_b']}
    return out


# --- The estimator gate (#89): validation on the real view graph -----------------------

def validation_groups(b_by_h_real):
    """Rig classes a city's estimators are validated on: those meeting rule 1's B clause
    (>= RULE_MIN_VIEWS_B held-out views at 2.6 m) on the real data ('other' excluded)."""
    rows = b_by_h_real.get(geo.DEFAULT_CAMERA_HEIGHT_M, {})
    return sorted(g for g, r in rows.items()
                  if g != 'other' and r['n_views'] >= RULE_MIN_VIEWS_B)


def validation_cells(seed_rows):
    """Cell means over seeds: one row per (city, group, h_true, noise) with, per
    estimator, the mean error and whether any seed was undefined or extrapolated."""
    cells = {}
    for r in seed_rows:
        cells.setdefault((r['city'], r['group'], r['h_true'], r['noise_scale']), []).append(r)
    out = []
    for (city, group, ht, ns), rs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        row = {'city': city, 'group': group, 'h_true': ht, 'noise_scale': ns,
               'seeds': len(rs), 'real_sweep_s': rs[0].get('real_sweep_s')}
        for est in (EST_LINE, EST_LOCAL):
            vals = [r[f'h_b_{est}'] for r in rs]
            undefined = any(v is None for v in vals)
            row[f'{est}_undefined'] = undefined
            row[f'{est}_extrapolated'] = any(bool(r[f'{est}_extrapolated']) for r in rs
                                             if r[f'h_b_{est}'] is not None)
            row[f'{est}_mean_err'] = None if undefined else statistics.fmean(
                v - ht for v in vals)
            row[f'{est}_errs'] = ' '.join('none' if v is None else f'{v - ht:+.4f}'
                                          for v in vals)
        row['b_b_mean'] = (statistics.fmean(r['b_b'] for r in rs)
                           if all(r['b_b'] is not None for r in rs) else None)
        out.append(row)
    return out


def rule_v(cells, estimator):
    """Rule V for one city's cells and one estimator: (valid, max |mean error|, reasons).

    VALID iff every cell has a defined, non-extrapolated value on every seed and the
    largest |mean error over seeds| is <= VALID_MAX_ERR_M (0.10 exactly passes). No
    cells at all is not valid (nothing was validated)."""
    if not cells:
        return False, None, ['no validation cells']
    reasons, errs = [], []
    for c in cells:
        tag = f"{c['group']} h_true {float(c['h_true']):g} noise {float(c['noise_scale']):g}"
        if _truthy(c[f'{estimator}_undefined']):
            reasons.append(f'{tag}: undefined')
            continue
        if _truthy(c[f'{estimator}_extrapolated']):
            reasons.append(f'{tag}: extrapolated')
        errs.append(abs(float(c[f'{estimator}_mean_err'])))
    worst = max(errs) if errs else None
    if worst is not None and worst > VALID_MAX_ERR_M + 1e-12:
        reasons.append(f'max |mean error| {worst:.3f} m > {VALID_MAX_ERR_M}')
    return not reasons, worst, reasons


def _truthy(v):
    return v in (True, 'True', 'true', 1, '1')


def select_estimator(valid_line, valid_local):
    """The plan's selection: both valid -> the line; one -> that one; neither -> None."""
    if valid_line:
        return EST_LINE
    if valid_local:
        return EST_LOCAL
    return None


def city_selection(cells):
    """{'estimator', 'validated', 'line': {...}, 'local': {...}} for one city's cells."""
    out = {}
    for est in (EST_LINE, EST_LOCAL):
        ok, worst, reasons = rule_v(cells, est)
        out[est] = {'valid': ok, 'max_abs_mean_err_m': None if worst is None
                    else round(worst, 4), 'reasons': reasons}
    sel = select_estimator(out[EST_LINE]['valid'], out[EST_LOCAL]['valid'])
    out.update({'estimator': sel, 'validated': sel is not None,
                'groups': sorted({c['group'] for c in cells}),
                'seeds_per_cell': min(int(float(c['seeds'])) for c in cells) if cells else 0,
                'real_sweep_s': cells[0].get('real_sweep_s') if cells else None})
    return out


_VALIDATION_CACHE = {}


def _validation_inputs(run_dir):
    """(panos, groups_of, truth) for a run, cached per worker process."""
    key = str(run_dir)
    if key not in _VALIDATION_CACHE:
        import height_gap as hg
        jsonl = Path(run_dir) / 'results.jsonl'
        panos, _ = fs.load_results(jsonl, read_heights=False)
        groups_of = pano_groups(jsonl)
        truth, _stats = hg.planted_sites(panos)
        _VALIDATION_CACHE.clear()          # one run per worker at a time
        _VALIDATION_CACHE[key] = (panos, groups_of, truth)
    return _VALIDATION_CACHE[key]


def validation_cell(city, run_dir, groups, h_true, noise, seed, real_sweep_s=None,
                    heights=SWEEP_HEIGHTS_B, null_seeds=range(NULL_SEEDS),
                    min_rows=rr.MIN_GROUP_ROWS):
    """One cell-seed of the estimator gate: the run re-synthesized with EVERY pano at
    h_true (height_gap.resynthesize, noise scaled, seeded), B swept over `heights` with a
    noise-matched null, both estimators read for each group in `groups`. Module level
    so a ProcessPoolExecutor can run it. Returns a list of rows."""
    import height_gap as hg
    t0 = time.time()
    panos, groups_of, truth = _validation_inputs(run_dir)
    syn, counts = hg.resynthesize(panos, truth, lambda _p: h_true, noise, 0.0, seed)
    key_of = lambda pid: groups_of[pid]['rig']   # noqa: E731
    sw = sweep_ab(syn, {'rig': key_of}, heights=heights, a_heights=(),
                  null_seeds=null_seeds, noise_scale=noise, min_rows=min_rows)
    rows = []
    for g in groups:
        est = b_estimates(sw['b']['rig'], g, n_draws=0)
        rows.append({'city': city, 'group': g, 'h_true': h_true, 'noise_scale': noise,
                     'seed': seed, 'h_b_line': est['h_b_line'], 'b_b': est['b_b'],
                     'amplification': est['amplification'],
                     'line_extrapolated': est['line_extrapolated'],
                     'h_b_local': est['h_b_local'],
                     'local_extrapolated': est['local_extrapolated'],
                     'h_at_2p6': est['h_at_2p6'], 'n_views': est['n_views'],
                     **{f'b_at_{h:g}': est.get(f'b_at_{h:g}') for h in heights},
                     'n_dets_out': counts['dets_out'], 'real_sweep_s': real_sweep_s,
                     'cell_s': round(time.time() - t0, 1)})
    return rows


def time_real_sweep(run_dir, log=print):
    """The budget clock (#89 plan section 2): one real B sweep of the city -- every
    height in SWEEP_HEIGHTS_B, the rig keying, NULL_SEEDS null seeds, i.e. the work one
    validation cell repeats on the real data. Returns (seconds, groups to validate).
    Prints only support counts: no h*_B is read or shown."""
    jsonl = Path(run_dir) / 'results.jsonl'
    panos, _ = fs.load_results(jsonl, read_heights=False)
    groups_of = pano_groups(jsonl)
    t0 = time.time()
    sw = sweep_ab(panos, {'rig': lambda pid: groups_of[pid]['rig']}, a_heights=())
    secs = time.time() - t0
    groups = validation_groups(sw['b']['rig'])
    views = {g: r['n_views'] for g, r in sw['b']['rig'][geo.DEFAULT_CAMERA_HEIGHT_M].items()}
    log(f'{Path(run_dir).name}: real sweep {secs:.0f} s; B views at 2.6 m {views}; '
        f'validating {groups}')
    return secs, groups


# --- The decision rule ------------------------------------------------------------------

def decide_group(a, b, support_scale=1.0):
    """Apply rules 1-4 to one group's instrument A row `a` and B row `b` (either None).
    Returns (height_m or None, passed [rule names], reason).

    Since #89, B's value is its fixed point h*_B under the city's validated estimator
    (b['h_star'], from b_rows_from_sweep), not #53's single reading B(2.6). A B row with
    validated False (no estimator passed rule V in the city) fails rule 3 as
    'b_unvalidated'. A row without the key (#53's shape) reads h_scale, as #53 did."""
    passed, fails = [], []
    n_views = b['n_views'] if b else 0
    if a and a['n_panos'] >= RULE_MIN_PANOS_A * support_scale \
            and a['n_sites'] >= math.ceil(RULE_MIN_SITES_A * support_scale) \
            and n_views >= RULE_MIN_VIEWS_B * support_scale:
        passed.append('support')
    else:
        fails.append(f"support (A panos {a['n_panos'] if a else 0}, sites "
                     f"{a['n_sites'] if a else 0}; B views {n_views})")
    if a and a['slope'] is not None and a['slope'] <= RULE_MAX_SLOPE \
            and a['h_star'] is not None:
        passed.append('identifiable')
    else:
        fails.append('identifiable (slope '
                     f"{'n/a' if not a or a['slope'] is None else round(a['slope'], 3)})")
    h_a = a['h_star'] if a else None
    h_b = _b_value(b)
    unvalidated = bool(b) and 'validated' in b and not b['validated']
    if unvalidated:
        fails.append(f'agreement ({B_UNVALIDATED}: no h*_B estimator passed rule V in '
                     f"this city; line {_f(b.get('h_b_line'), '.3f')}, local "
                     f"{_f(b.get('h_b_local'), '.3f')})")
    elif h_a is not None and h_b is not None and abs(h_a - h_b) <= RULE_MAX_DISAGREE_M:
        passed.append('agreement')
    else:
        fails.append('agreement (A '
                     f"{'n/a' if h_a is None else round(h_a, 3)}, B "
                     f"{'n/a' if h_b is None else round(h_b, 3)})")
    h_g = (h_a + h_b) / 2 if h_a is not None and h_b is not None else None
    h0 = geo.DEFAULT_CAMERA_HEIGHT_M
    ci_ok = a and a['ci_lo'] is not None and not (a['ci_lo'] <= h0 <= a['ci_hi'])
    if ci_ok and h_g is not None and abs(h_g - h0) > RULE_MIN_EFFECT_M:
        passed.append('material')
    else:
        ci = ('n/a' if not a or a['ci_lo'] is None
              else f"{a['ci_lo']:.2f}-{a['ci_hi']:.2f}")
        fails.append(f"material (CI {ci}, h_g {'n/a' if h_g is None else round(h_g, 3)})")
    if fails:
        return None, passed, 'fails ' + '; '.join(fails)
    return h_g, passed, 'passes all four rules'


def _b_value(b):
    """B's value for rule 3: the selected fixed point (#89 rows), else B(2.6) (#53 rows)."""
    if not b:
        return None
    return b['h_star'] if 'h_star' in b else b['h_scale']


def _iqr(values):
    if len(values) < 2:
        return None
    q = statistics.quantiles(values, n=4, method='inclusive')
    return q[2] - q[0]


def build_table(groups_of, a_rig, b_rig, a_seq, b_seq_views, a_seq_boot=None,
                b_seq=None):
    """The camera_heights.json body (without the hash/time/recommended fields).

    groups_of: {pano_id: group dict}; a_rig/b_rig: instrument rows by rig class; a_seq:
    instrument A point rows by sequence; b_seq_views: {sequence: held-out views}; a_seq_boot
    / b_seq: full A (with CI) and B rows for the sequences of a rig class that went to
    sequence grain (as returned by instrument_a / instrument_b with sequence keys).
    Returns (table, per-rig grain notes)."""
    h0 = geo.DEFAULT_CAMERA_HEIGHT_M
    # A sequence belongs to the rig class most of its panos report (ties: first by name),
    # so a sequence spanning two rig keys resolves deterministically.
    votes = {}
    for g in groups_of.values():
        v = votes.setdefault(g['sequence'], {})
        v[g['rig']] = v.get(g['rig'], 0) + 1
    seqs_by_rig = {}
    for seq, v in votes.items():
        rig = min(v, key=lambda r: (-v[r], str(r)))
        seqs_by_rig.setdefault(rig, set()).add(seq)
    groups, sequences, notes = {}, {}, {}
    for rig in sorted(seqs_by_rig):
        a, b = a_rig.get(rig), b_rig.get(rig)
        h, passed, reason = decide_group(a, b)
        groups[rig] = _group_entry(a, b, h, passed, reason, 'rig')
        # Do this rig's sequences disagree? Only sequences meeting rule 1 at a quarter.
        qual = []
        for seq in sorted(seqs_by_rig[rig]):
            sa = a_seq.get(seq)
            if sa and sa['h_star'] is not None \
                    and sa['n_panos'] >= RULE_MIN_PANOS_A * SEQ_SUPPORT_FRACTION \
                    and sa['n_sites'] >= math.ceil(RULE_MIN_SITES_A * SEQ_SUPPORT_FRACTION) \
                    and b_seq_views.get(seq, 0) >= RULE_MIN_VIEWS_B * SEQ_SUPPORT_FRACTION:
                qual.append(sa['h_star'])
        iqr = _iqr(qual)
        per_seq = iqr is not None and iqr > SEQ_IQR_M
        notes[rig] = {'qualifying_sequences': len(qual), 'sequence_iqr_m': iqr,
                      'grain': 'sequence' if per_seq else 'rig'}
        groups[rig]['sequence_iqr_m'] = _rd(iqr)
        groups[rig]['qualifying_sequences'] = len(qual)
        for seq in sorted(seqs_by_rig[rig], key=str):
            sequences[seq] = rig
        if not per_seq:
            continue
        for seq in sorted(seqs_by_rig[rig]):
            sa = (a_seq_boot or {}).get(seq)
            sb = (b_seq or {}).get(seq)
            h_s, p_s, r_s = decide_group(sa, sb)
            if 'support' not in p_s:
                continue                         # inherits the rig-class value
            key = f'{rig}@{seq}'
            groups[key] = _group_entry(sa, sb, h_s, p_s, r_s, 'sequence')
            sequences[seq] = key
    # The table's grain is what it resolves to: 'sequence' only if a per-sequence group was
    # actually written (a rig class can go per-sequence with every sequence inheriting).
    grain = 'sequence' if any(g['grain'] == 'sequence' for g in groups.values()) else 'rig'
    return {'schema': 1, 'source': 'mapillary', 'default_m': h0, 'grain': grain,
            'groups': groups, 'sequences': sequences}, notes


def _group_entry(a, b, h, passed, reason, grain):
    h_a = a['h_star'] if a else None
    h_b = _b_value(b)
    sigma = None
    if h is not None:
        sd = a['boot_sd'] or 0.0
        sigma = math.sqrt(sd ** 2 + ((h_a - h_b) / 2) ** 2)
    suspect = h_a is not None and not (SANE_HEIGHT_M[0] <= h_a <= SANE_HEIGHT_M[1])
    entry = _entry_core(a, b, h, sigma, passed, reason, grain, suspect)
    if b and 'h_star' in b:
        entry['method'] = ('mean(bearing fixed point, scale-identity fixed point '
                           f"[{b['estimator'] or 'none validated'}])")
        entry['instrument_b'] = {
            'estimator': b['estimator'], 'h_star': _rd(b['h_star']),
            'amplification': _rd(b['amplification']), 'extrapolated': b['extrapolated'],
            'validated': b['validated'], 'h_at_2p6': _rd(b['h_at_2p6']),
            'h_line': _rd(b['h_b_line']), 'h_line_ci': _ci_pair(b, 'line'),
            'h_local': _rd(b['h_b_local']), 'h_local_ci': _ci_pair(b, 'local'),
            'local_extrapolated': b['local_extrapolated'],
            'ci_note': 'too narrow: the association heights share views'}
    return entry


def _ci_pair(b, est):
    lo, hi = b.get(f'{est}_ci_lo'), b.get(f'{est}_ci_hi')
    return None if lo is None else [_rd(lo), _rd(hi)]


def _entry_core(a, b, h, sigma, passed, reason, grain, suspect):
    h0 = geo.DEFAULT_CAMERA_HEIGHT_M
    h_a = a['h_star'] if a else None
    return {'height_m': h0 if h is None else round(h, 3),
            'sigma_m': None if sigma is None else round(sigma, 3),
            'applied': h is not None, 'grain': grain,
            'n_panos': a['n_panos'] if a else 0, 'n_sites': a['n_sites'] if a else 0,
            'n_views_b': b['n_views'] if b else 0,
            'h_bearing': _rd(h_a), 'h_bearing_ci': None if not a or a['ci_lo'] is None
            else [_rd(a['ci_lo']), _rd(a['ci_hi'])],
            'h_scale': _rd(b['h_scale']) if b else None, 'h_scale_ci': None if not b
            else [_rd(b['h_scale_lo']), _rd(b['h_scale_hi'])],
            'h_scale_with_offset': _rd(b['h_scale_with_offset']) if b else None,
            'slope': _rd(a['slope']) if a else None,
            'method': 'mean(bearing fixed point, null-corrected scale identity)',
            'passed': passed, 'reason': reason + ('; SUSPECT: bearing height outside '
                                                  f'{SANE_HEIGHT_M[0]}-{SANE_HEIGHT_M[1]} m'
                                                  if suspect else '')}


def _rd(v, nd=3):
    return None if v is None else round(v, nd)


# --- Instrument C and the production gate ---------------------------------------------

def gate_params(camera_height):
    """The benchmark tier, as eval_sites pins it, pose off."""
    return fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False,
                         camera_height_m=camera_height, apply_pose=fs.POSE_OFF)


def flat(panos):
    """Copies without pitch/roll, so every GT raycast is flat whatever apply_pose the
    borrowed code passes (see the module docstring)."""
    return [replace(p, camera_pitch=None, camera_roll=None) for p in panos]


def instrument_c(city, panos, split_data, camera_height):
    """Along-ray residual of reviewer references on predicted range (left-out variant,
    every reference kind), cluster-robust by site. Returns a row dict."""
    verdicts, bundle_ops, boxes = split_data
    params = gate_params(camera_height)
    sites, frame, _ = fs.fuse(panos, params)
    svs = [rr.views_from_site(s) for s in sites]
    label = ARM_RIG if camera_height == geo.PER_RIG else ARM_OFF
    rows, _counts, _w = rr.gt_anchored_rows(city, label, verdicts, bundle_ops, boxes,
                                           panos, svs, frame, params)
    out = {'city': city, 'arm': label}
    for name, kinds in (('all', ('box', 'missed', 'peak')), ('independent', ('box', 'missed'))):
        use = [r for r in rows if r['ref_kind'] in kinds and r['loo_along_m'] is not None
               and r['loo_pred_range_m'] is not None]
        slope, se, icpt = rr.ols_slope([r['loo_pred_range_m'] for r in use],
                                       [r['loo_along_m'] for r in use],
                                       [r['site_id'] for r in use])
        out.update({f'{name}_n': len(use), f'{name}_slope': slope, f'{name}_slope_se': se,
                    f'{name}_intercept_m': icpt})
    return out


def height_gate(verdict_panos, bundle_ops, panos, gt_merge_m=2.5):
    """per-rig vs off on ONE site set and ONE GT set -- eval_sites.pose_precondition's
    design with height arms instead of pose arms. Association is frozen from the `off`
    (2.6 m) fuse; a site is scored only if both arms place every operational member at
    the production cap, and each arm refits it from its own raycasts; a GT mark is used
    only if both arms place it; distances over the pool ramps both arms match within
    5 m. Recall on the OFF pool guards against survivorship. `panos` must carry the
    table's heights (fuse_sites.apply_height_table) and no pitch/roll.

    Returns (rows, info), one row per arm."""
    arms = (ARM_OFF, ARM_RIG)
    params = {ARM_OFF: gate_params(geo.DEFAULT_CAMERA_HEIGHT_M),
              ARM_RIG: gate_params(geo.PER_RIG)}
    by_id = {p.pano_id: p for p in panos}
    sites, frame, _ = fs.fuse(panos, params[ARM_OFF])
    poses = {}

    def place(arm, pid, x, y):
        pose = poses.get(pid)
        if pose is None:
            pose = poses[pid] = fs.pano_pose(by_id[pid], fs.POSE_OFF)
        return geo.detection_ground_point(
            pose, x, y, camera_height=params[arm].camera_height_m,
            max_range_m=params[arm].max_range_m,
            errors=geo.error_model_for(by_id[pid].source), apply_pose=False)

    op_sites = [s for s in sites if s.n_operational > 0]
    kept, site_pos = [], {arm: [] for arm in arms}
    for site in op_sites:
        members = [d for d, _ in site.members if d.operational]
        placed = {}
        for arm in arms:
            gs = [place(arm, d.pano_id, d.x, d.y) for d in members]
            if any(g is None for g in gs):
                break
            placed[arm] = gs
        else:
            kept.append(site)
            for arm in arms:
                lam, eta_e, eta_n = (0.0, 0.0, 0.0), 0.0, 0.0
                for d, g in zip(members, placed[arm]):
                    e, n = frame.to_enu(g.lat, g.lng)
                    w = geo.sym2_inv(g.cov_en(geo.error_model_for(d.source).sigma_gps_m))
                    lam = geo.sym2_add(lam, w)
                    eta_e += w[0] * e + w[1] * n
                    eta_n += w[1] * e + w[2] * n
                inv = geo.sym2_inv(lam)
                site_pos[arm].append(es._Placed(site.id, inv[0] * eta_e + inv[1] * eta_n,
                                                inv[1] * eta_e + inv[2] * eta_n))

    counts, warnings = es.gt_counts(), []
    marks = []
    for pid, entry, _run_pano, ops, in_pool in es.judged_gt_panos(
            verdict_panos, bundle_ops, by_id, counts, warnings):
        for verdict, (_i, x, y, _c) in zip(entry['dets'], ops):
            if verdict is True:
                marks.append((pid, x, y, 'det', in_pool))
        for mark in entry.get('missed', ()):
            if not mark.get('unsure'):
                marks.append((pid, mark['x'], mark['y'], 'missed', in_pool))
    enu = []
    for pid, x, y, _k, _p in marks:
        row = {}
        for arm in arms:
            g = place(arm, pid, x, y)
            row[arm] = None if g is None else frame.to_enu(g.lat, g.lng)
        enu.append(row)
    unplaceable = {arm: sum(1 for r in enu if r[arm] is None) for arm in arms}

    def ramps_from(ids):
        pts = [es.GTPoint(marks[i][0], marks[i][3], *enu[i][ARM_OFF], marks[i][4])
               for i in ids]
        index_of = {id(pt): i for pt, i in zip(pts, ids)}
        return [(r, [index_of[id(pt)] for pt in r.points])
                for r in es.merge_gt_points(pts, gt_merge_m) if r.in_pool]

    def place_ramps(arm, ramps):
        out = []
        for k, (_r, ids) in enumerate(ramps):
            got = [enu[i][arm] for i in ids if enu[i][arm] is not None]
            out.append(None if not got else es._Placed(
                k, sum(e for e, _ in got) / len(got), sum(n for _, n in got) / len(got)))
        return out

    common_ids = [i for i, r in enumerate(enu) if all(v is not None for v in r.values())]
    pool = ramps_from(common_ids)
    off_pool = ramps_from([i for i, r in enumerate(enu) if r[ARM_OFF] is not None])
    matched, off_recall = {}, {}
    for arm in arms:
        placed = place_ramps(arm, pool)
        for radius in es.PRECONDITION_RADII:
            hits = es.match_one_to_one(placed, site_pos[arm], radius)
            matched[arm, radius] = {i: math.hypot(placed[i].e - s.e, placed[i].n - s.n)
                                    for i, s in hits.items()}
        placed_off = place_ramps(arm, off_pool)
        present = [p for p in placed_off if p is not None]
        for radius in es.PRECONDITION_RADII:
            hits = es.match_one_to_one(present, site_pos[arm], radius)
            hit_ids = {present[i].id for i in hits}
            recalled = sum(1 for k, (r, _ids) in enumerate(off_pool)
                           if placed_off[k] is not None and (r.self_detected or k in hit_ids))
            off_recall[arm, radius] = recalled / len(off_pool) if off_pool else None
    common = set(range(len(pool)))
    for arm in arms:
        common &= set(matched[arm, 5.0])
    applied = sum(1 for p in panos if p.camera_height_m is not None)
    info = {'op_sites': len(op_sites), 'sites_scored': len(kept), 'gt_marks': len(marks),
            'gt_marks_dropped': len(marks) - len(common_ids), 'pool_ramps': len(pool),
            'off_pool_ramps': len(off_pool), 'common_matched_5m': len(common),
            'panos': len(panos), 'panos_changed': applied, 'warnings': warnings}
    rows = []
    for arm in arms:
        dists = [matched[arm, 5.0][i] for i in sorted(common)]
        row = {'arm': arm, 'sites_scored': len(kept), 'pool_ramps': len(pool),
               'n_common_matched': len(common),
               'median_gt_to_site_m': es._pct(dists, 0.5),
               'p90_gt_to_site_m': es._pct(dists, 0.9),
               'mean_gt_to_site_m': sum(dists) / len(dists) if dists else None}
        for radius in es.PRECONDITION_RADII:
            tag = f'{radius:g}'.replace('.', 'p')
            hit = matched[arm, radius]
            row[f'world_recall_{tag}m'] = (sum(1 for i, (r, _) in enumerate(pool)
                                               if r.self_detected or i in hit) / len(pool)
                                           if pool else None)
            row[f'recall_off_pool_{tag}m'] = off_recall[arm, radius]
        row['off_pool_ramps'] = len(off_pool)
        row['gt_marks_unplaceable'] = unplaceable[arm]
        rows.append(row)
    return rows, info


def gate_verdict(gate_by_city, c_by_city):
    """Apply the pre-registered production gate. gate_by_city = {city: (rows, info)},
    c_by_city = {city: {arm: instrument C row}}. Returns (passes, clauses, lines)."""
    changed = [c for c, (_rows, info) in gate_by_city.items()
               if info['panos'] and info['panos_changed'] / info['panos'] >= GATE_CHANGED_FRAC]
    p90_fail, wins, rec_fail, c_fail, lines = [], [], [], [], []
    for city, (rows, info) in gate_by_city.items():
        r = {row['arm']: row for row in rows}
        off, rig = r[ARM_OFF], r[ARM_RIG]
        d90 = _diff(rig['p90_gt_to_site_m'], off['p90_gt_to_site_m'])
        dmed = _diff(rig['median_gt_to_site_m'], off['median_gt_to_site_m'])
        drec = _diff(rig['recall_off_pool_2p5m'], off['recall_off_pool_2p5m'])
        c = c_by_city.get(city, {})
        c_off = (c.get(ARM_OFF) or {}).get('all_slope')
        c_rig = (c.get(ARM_RIG) or {}).get('all_slope')
        is_changed = city in changed
        if is_changed and d90 is not None and d90 > GATE_P90_TOL_M:
            p90_fail.append(city)
        if is_changed and dmed is not None and dmed < 0:
            wins.append(city)
        if drec is not None and drec < -GATE_MAX_RECALL_DROP:
            rec_fail.append(city)
        if is_changed and c_off is not None and c_rig is not None and abs(c_rig) > abs(c_off):
            c_fail.append(city)
        lines.append(f"{city}: {info['panos_changed']}/{info['panos']} panos changed "
                     f"({'changed' if is_changed else 'unchanged'}); p90 rig-off "
                     f"{_f(d90, '+.3f')} m, median {_f(dmed, '+.3f')} m, off-pool R@2.5 "
                     f"{_f(None if drec is None else 100 * drec, '+.1f')} pt, C slope "
                     f"{_f(c_off, '+.4f')} -> {_f(c_rig, '+.4f')}")
    clauses = {'i': not p90_fail, 'ii': len(wins) >= GATE_MIN_MEDIAN_WINS,
               'iii': not rec_fail, 'iv': not c_fail}
    lines += [f"changed cities (>= {GATE_CHANGED_FRAC:.0%} of panos): "
              f"{', '.join(changed) or 'none'}",
              f"(i) p90 worse by > {GATE_P90_TOL_M} m in a changed city: "
              f"{', '.join(p90_fail) or 'none'} -> {_pf(clauses['i'])}",
              f"(ii) median better in {len(wins)} changed cities (needs "
              f"{GATE_MIN_MEDIAN_WINS}): {', '.join(wins) or 'none'} -> {_pf(clauses['ii'])}",
              f"(iii) off-pool recall@2.5 m down > {100 * GATE_MAX_RECALL_DROP:.1f} pt in: "
              f"{', '.join(rec_fail) or 'none'} -> {_pf(clauses['iii'])}",
              f"(iv) instrument C |slope| larger in a changed city: "
              f"{', '.join(c_fail) or 'none'} -> {_pf(clauses['iv'])}"]
    return all(clauses.values()), clauses, lines


def _diff(a, b):
    return None if a is None or b is None else a - b


def _f(v, spec):
    return 'n/a' if v is None else format(v, spec)


def _pf(ok):
    return 'PASS' if ok else 'FAIL'


# --- Per-city driver ------------------------------------------------------------------

def measure_city(city, run_dir, selection, n_boot=N_BOOT, log=print):
    """Instruments A and B over every grouping, the rule, and the table body.

    selection: the city's estimator selection (city_selection over its
    estimator_validation.csv): B's value for rule 3 is the selected estimator's h*_B, or
    b_unvalidated when neither passed rule V. Returns a dict with rows per grouping, the
    table and notes."""
    t_start = time.time()
    jsonl = run_dir / 'results.jsonl'
    panos, _ = fs.load_results(jsonl, read_heights=False)
    other = {p.source for p in panos if p.source not in geo.CROWDSOURCED_SOURCES}
    if other:
        raise ValueError(f'{city}: per-rig heights are for crowdsourced runs; holds {other}')
    groups_of = pano_groups(jsonl)
    params = production_params()
    kinds = ('rig', 'creator', 'speed', 'sequence')
    b_kinds = ('rig', 'creator', 'speed')
    log(f'{city}: {len(panos)} panos; A at {SWEEP_HEIGHTS}, B at {SWEEP_HEIGHTS_B} '
        f'({NULL_SEEDS} null seeds) ...')
    sw = sweep_ab(panos, {k: (lambda pid, k=k: groups_of[pid][k]) for k in b_kinds},
                  heights=SWEEP_HEIGHTS_B, a_heights=SWEEP_HEIGHTS, params=params, log=log)
    a_rows = {}
    for kind in kinds:
        a_rows[kind] = instrument_a(
            sw['a'], lambda pid, k=kind: groups_of[pid][k], n_boot=n_boot,
            boot_keys=set() if kind == 'sequence' else None)
    est, ok = selection['estimator'], selection['validated']
    b_rows = {kind: b_rows_from_sweep(sw['b'][kind], est, ok) for kind in b_kinds}
    sites, _frame = site_views(panos, replace(params, camera_height_m=geo.DEFAULT_CAMERA_HEIGHT_M))
    seq_views = {}
    for sv in sites:
        for v in sv.views:
            s = groups_of[v.pano_id]['sequence']
            seq_views[s] = seq_views.get(s, 0) + 1
    pooled = pooled_scale(sites)
    bench_sites, _ = site_views(panos, gate_params(geo.DEFAULT_CAMERA_HEIGHT_M))
    pooled_bench = pooled_scale(bench_sites)
    table, notes = build_table(groups_of, a_rows['rig'], b_rows['rig'], a_rows['sequence'],
                               seq_views)
    seq_rigs = [rig for rig, n in notes.items() if n['grain'] == 'sequence']
    if seq_rigs:
        # Sequence grain: the rig's sequences that meet rule 1 become their own groups,
        # so they need the full instruments (CI for A, a joint B sweep).
        cand = {g['sequence'] for g in groups_of.values() if g['rig'] in seq_rigs}
        a_seq_boot = instrument_a(sw['a'], lambda pid: groups_of[pid]['sequence'],
                                  n_boot=n_boot, boot_keys=cand)

        def seq_or_rig(pid):
            g = groups_of[pid]
            return g['sequence'] if g['sequence'] in cand else g['rig']
        log(f'{city}: sequence grain for {seq_rigs}; B sweep per sequence ...')
        sw_seq = sweep_ab(panos, {'seq': seq_or_rig}, heights=SWEEP_HEIGHTS_B, a_heights=(),
                          params=params, log=log)
        b_seq = b_rows_from_sweep(sw_seq['b']['seq'], est, ok)
        table, notes = build_table(groups_of, a_rows['rig'], b_rows['rig'],
                                   a_rows['sequence'], seq_views, a_seq_boot, b_seq)
        a_rows['sequence'] = {**a_rows['sequence'], **{k: v for k, v in a_seq_boot.items()
                                                       if k in cand}}
        b_rows['sequence'] = b_seq
    table['estimator_validation'] = validation_block(selection)
    panos_by_group = {}
    for g in groups_of.values():
        for kind in kinds:
            panos_by_group[kind, g[kind]] = panos_by_group.get((kind, g[kind]), 0) + 1
    return {'city': city, 'panos': panos, 'groups_of': groups_of, 'a': a_rows, 'b': b_rows,
            'table': table, 'notes': notes, 'pooled_k': pooled, 'pooled_k_bench': pooled_bench,
            'panos_by_group': panos_by_group, 'seq_views': seq_views,
            'seconds': round(time.time() - t_start)}


def validation_block(selection):
    """The camera_heights.json `estimator_validation` block (#89)."""
    return {'estimator': selection['estimator'], 'validated': selection['validated'],
            'rule': (f'rule V: over h_true {list(VALID_TRUE_HEIGHTS)} x noise '
                     f'{list(VALID_NOISES)}, max |mean error over seeds| <= '
                     f'{VALID_MAX_ERR_M} m and no cell undefined or extrapolated; both '
                     'valid -> line, one -> that one, neither -> b_unvalidated'),
            'groups': selection['groups'], 'seeds_per_cell': selection['seeds_per_cell'],
            'real_sweep_s': selection['real_sweep_s'],
            'line': selection[EST_LINE], 'local': selection[EST_LOCAL]}


def group_rows(m):
    """Flat CSV rows (one per grouping x group) joining A and B."""
    rows = []
    for kind in ('rig', 'creator', 'speed', 'sequence'):
        a, b = m['a'].get(kind, {}), m['b'].get(kind, {})
        for key in sorted(set(a) | set(b), key=str):
            ra, rb = a.get(key) or {}, b.get(key) or {}
            row = {'city': m['city'], 'grouping': kind, 'group': key,
                   'panos': m['panos_by_group'].get((kind, key)),
                   'a_panos': ra.get('n_panos'), 'a_sites': ra.get('n_sites'),
                   'h_bearing': _rd(ra.get('h_star')), 'slope': _rd(ra.get('slope')),
                   'h_bearing_lo': _rd(ra.get('ci_lo')), 'h_bearing_hi': _rd(ra.get('ci_hi')),
                   'boot_undefined': ra.get('boot_undefined'),
                   'b_views': rb.get('n_views') if rb else
                   (m['seq_views'].get(key) if kind == 'sequence' else None),
                   'h_scale': _rd(rb.get('h_scale')), 'h_scale_lo': _rd(rb.get('h_scale_lo')),
                   'h_scale_hi': _rd(rb.get('h_scale_hi')), 'k_net': _rd(rb.get('k_net')),
                   'null_s': _rd(rb.get('null_s'), 4),
                   'null_sd_m': _rd(rb.get('null_sd_m'), 4),
                   'h_scale_with_offset': _rd(rb.get('h_scale_with_offset')),
                   'estimator': rb.get('estimator'), 'b_validated': rb.get('validated'),
                   'h_star_b': _rd(rb.get('h_star')),
                   'h_b_line': _rd(rb.get('h_b_line')), 'b_slope_b': _rd(rb.get('b_b')),
                   'amplification': _rd(rb.get('amplification')),
                   'line_extrapolated': rb.get('line_extrapolated'),
                   'h_b_line_lo': _rd(rb.get('line_ci_lo')),
                   'h_b_line_hi': _rd(rb.get('line_ci_hi')),
                   'line_draws_undefined': rb.get('line_draws_undefined'),
                   'h_b_local': _rd(rb.get('h_b_local')),
                   'local_extrapolated': rb.get('local_extrapolated'),
                   'h_b_local_lo': _rd(rb.get('local_ci_lo')),
                   'h_b_local_hi': _rd(rb.get('local_ci_hi'))}
            for h in SWEEP_HEIGHTS:
                row[f'implied_at_{h:g}'] = _rd(ra.get(f'implied_at_{h:g}'))
            for h in SWEEP_HEIGHTS_B:
                row[f'b_at_{h:g}'] = _rd(rb.get(f'b_at_{h:g}'))
            rows.append(row)
    return rows


def write_csv(path, rows):
    rr.write_csv(path, rows)


def write_table(run_dir, table, recommended, gate_summary=None):
    """Write runs/<name>/camera_heights.json bound to results.jsonl by sha256."""
    body = dict(table)
    body['results_sha256'] = fs.file_sha256(run_dir / 'results.jsonl')
    body['generated_at'] = datetime.datetime.now(datetime.timezone.utc) \
        .replace(microsecond=0).isoformat()
    body['recommended'] = bool(recommended)
    if gate_summary is not None:
        body['gate'] = gate_summary
    body['generated_by'] = 'scripts/mapillary_height.py (issues #53, #89)'
    order = ['schema', 'source', 'default_m', 'grain', 'recommended', 'gate',
             'estimator_validation', 'results_sha256', 'generated_at', 'generated_by',
             'groups', 'sequences']
    body = {k: body[k] for k in order if k in body}
    path = run_dir / fs.HEIGHT_TABLE_NAME
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(body, f, indent=1, sort_keys=False)
        f.write('\n')
    return path


# --- The validate step (#89) ------------------------------------------------------------

VALIDATION_CSV = 'estimator_validation.csv'               # cell means (what rule V reads)
VALIDATION_SEEDS_CSV = 'estimator_validation_seeds.csv'   # one row per cell-seed-group


def _read_rows(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _num(v):
    """A CSV cell back to None / bool / float (strings otherwise)."""
    if v in (None, '', 'None'):
        return None
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return float(v)
    except ValueError:
        return v


def _typed(rows):
    return [{k: v if k in ('city', 'group') else _num(v) for k, v in r.items()} for r in rows]


def _seed_key(r):
    return (str(r['city']), str(r['group']), float(r['h_true']), float(r['noise_scale']),
            int(float(r['seed'])))


def run_validation(cities, run_root, workers, log=print):
    """The estimator gate: for each city, time one real sweep (the budget rule), then run
    the grid VALID_TRUE_HEIGHTS x VALID_NOISES x seeds in a process pool. Resumable: the
    per-seed rows are written to runs/<city>/camera_height/estimator_validation_seeds.csv
    as they finish and a re-run skips them (and reuses the recorded sweep time)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    plan, done = [], {}
    for city in cities:
        run_dir = run_root / city
        prior = _typed(_read_rows(run_dir / 'camera_height' / VALIDATION_SEEDS_CSV))
        done[city] = {_seed_key(r): r for r in prior}
        if prior and prior[0].get('real_sweep_s') is not None:
            secs = prior[0]['real_sweep_s']
            groups = sorted({r['group'] for r in prior})
            log(f'{city}: reusing the recorded real sweep time {secs:.0f} s; groups {groups}')
        else:
            secs, groups = time_real_sweep(run_dir, log=log)
            secs = round(secs, 1)
        seeds = VALID_SEEDS if secs <= VALID_BUDGET_S else 1
        log(f'{city}: {seeds} seed(s) per cell (budget rule: real sweep {secs:.0f} s vs '
            f'{VALID_BUDGET_S} s)', flush=True)
        for ht in VALID_TRUE_HEIGHTS:
            for ns in VALID_NOISES:
                for seed in range(seeds):
                    if groups and all((city, g, ht, ns, seed) in done[city] for g in groups):
                        continue
                    plan.append((secs, city, run_dir, groups, ht, ns, seed))
    plan.sort(key=lambda t: -t[0])               # the slowest cities first
    log(f'{len(plan)} cell-seeds to run on {workers} workers', flush=True)

    def flush(city):
        rows = sorted(done[city].values(), key=lambda r: tuple(map(str, _seed_key(r))))
        out = run_root / city / 'camera_height'
        write_csv(out / VALIDATION_SEEDS_CSV, rows)
        write_csv(out / VALIDATION_CSV, validation_cells(rows))
    if plan:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(validation_cell, city, run_dir, groups, ht, ns, seed, secs):
                    (city, ht, ns, seed) for secs, city, run_dir, groups, ht, ns, seed in plan}
            for k, fut in enumerate(as_completed(futs), 1):
                city, ht, ns, seed = futs[fut]
                rows = fut.result()
                for r in rows:
                    done[city][_seed_key(r)] = r
                flush(city)
                log(f'  [{k}/{len(plan)}] {city} h_true {ht:g} noise {ns:g} seed {seed}: '
                    f"{rows[0]['cell_s'] if rows else 0:.0f} s", flush=True)
    for city in cities:
        flush(city)


def load_selection(run_dir):
    """The city's estimator selection from its estimator_validation.csv, or SystemExit."""
    path = Path(run_dir) / 'camera_height' / VALIDATION_CSV
    rows = _read_rows(path)
    if not rows:
        raise SystemExit(f'{path} is missing or empty: run `mapillary_height.py --validate '
                         f'{Path(run_dir).name}` first (#89: B needs a validated estimator)')
    return city_selection(_typed(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cities', nargs='+', help='Mapillary run names under --run-root')
    ap.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--no-gate', action='store_true',
                    help='measure and write tables only (recommended: false)')
    ap.add_argument('--validate', action='store_true',
                    help='run the #89 estimator gate (simulation) and write '
                         'camera_height/estimator_validation.csv; nothing else')
    ap.add_argument('--workers', type=int, default=None,
                    help='--validate: worker processes (default: CPU count)')
    args = ap.parse_args()

    if args.validate:
        import os
        run_validation(args.cities, args.run_root, args.workers or os.cpu_count() or 1)
        return

    measured, k_rows, v_rows, v_lines = {}, [], [], []
    for city in args.cities:
        run_dir = args.run_root / city
        selection = load_selection(run_dir)
        v_rows += _read_rows(run_dir / 'camera_height' / VALIDATION_CSV)
        v_lines.append(f"{city}: estimator {selection['estimator'] or B_UNVALIDATED} "
                       f"(groups {selection['groups']}, {selection['seeds_per_cell']} seed(s) "
                       f"per cell, real sweep {_f(selection['real_sweep_s'], '.0f')} s)")
        for est in (EST_LINE, EST_LOCAL):
            s = selection[est]
            v_lines.append(f"  {est}: {'VALID' if s['valid'] else 'not valid'}; max |mean "
                           f"error| {_f(s['max_abs_mean_err_m'], '.3f')} m"
                           + (f"; {'; '.join(s['reasons'])}" if s['reasons'] else ''))
        print('\n'.join(v_lines[-3:]), flush=True)
        m = measure_city(city, run_dir, selection, n_boot=args.n_boot)
        measured[city] = m
        out = run_dir / 'camera_height'
        write_csv(out / 'groups.csv', group_rows(m))
        write_table(run_dir, m['table'], recommended=False)
        k, s, s0 = m['pooled_k']
        kb, sb, s0b = m['pooled_k_bench']
        k_rows.append({'city': city, 'k_net_production': _rd(k, 4), 's_production': _rd(s, 4),
                       's_null_production': _rd(s0, 4), 'k_net_benchmark': _rd(kb, 4),
                       's_benchmark': _rd(sb, 4), 's_null_benchmark': _rd(s0b, 4)})
        print(f"{city}: measured in {m['seconds']} s; pooled k_net {k:.3f} at the production "
              f'tier, {kb:.3f} at the benchmark tier (#76 tie-back)')
        for rig, n in m['notes'].items():
            print(f"  {rig}: grain {n['grain']} ({n['qualifying_sequences']} qualifying "
                  f"sequences, h* IQR {_f(n['sequence_iqr_m'], '.3f')} m)")
        for key, g in m['table']['groups'].items():
            ib = g.get('instrument_b') or {}
            print(f"  {key}: {g['height_m']} m (applied {g['applied']}); A {g['h_bearing']} "
                  f"{g['h_bearing_ci']} slope {g['slope']}; B(2.6) {g['h_scale']}; h*_B line "
                  f"{ib.get('h_line')} {ib.get('h_line_ci')} (x{ib.get('amplification')}), "
                  f"local {ib.get('h_local')}"
                  f"{' (extrapolated)' if ib.get('local_extrapolated') else ''} "
                  f"{ib.get('h_local_ci')} [CIs too narrow: heights share views]; "
                  f"{g['reason']}", flush=True)
    summ = args.run_root / '_summary' / 'camera_height'
    write_csv(summ / 'k_net.csv', k_rows)
    write_csv(summ / VALIDATION_CSV, v_rows)
    (summ / 'estimator_validation.txt').write_text('\n'.join(v_lines) + '\n', encoding='utf-8')
    if args.no_gate:
        return

    gate, cslopes = {}, {}
    for city, m in measured.items():
        run_dir = args.run_root / city
        split = SPLITS.get(city, city)
        split_data = rr.load_split(args.benchmark_root, split)
        if split_data is None:
            print(f'{city}: no benchmark split {split}; gate skipped for it')
            continue
        panos, _ = fs.load_results(run_dir / 'results.jsonl', read_heights=False,
                                   height_table=run_dir / fs.HEIGHT_TABLE_NAME)
        panos = flat(panos)
        print(f'{city}: gate + instrument C ...', flush=True)
        rows, info = height_gate(split_data[0], split_data[1], panos)
        gate[city] = (rows, info)
        cslopes[city] = {arm: instrument_c(city, panos, split_data, h)
                         for arm, h in ((ARM_OFF, geo.DEFAULT_CAMERA_HEIGHT_M),
                                        (ARM_RIG, geo.PER_RIG))}
    passes, clauses, lines = gate_verdict(gate, cslopes)
    write_csv(summ / 'gate.csv', [{'city': c, **row, **{f'info_{k}': v for k, v in info.items()
                                                        if k != 'warnings'}}
                                  for c, (rows, info) in gate.items() for row in rows])
    write_csv(summ / 'instrument_c.csv', [r for c in cslopes.values() for r in c.values()])
    write_csv(summ / 'groups.csv', [r for m in measured.values() for r in group_rows(m)])
    (summ / 'gate.txt').write_text('\n'.join(lines) + f'\nVERDICT: {_pf(passes)}\n',
                                   encoding='utf-8')
    print('\n'.join(lines))
    print(f'VERDICT: {_pf(passes)} -> recommended: {passes}')
    for city, m in measured.items():
        write_table(args.run_root / city, m['table'], recommended=passes,
                    gate_summary={'passes': passes, 'clauses': clauses})
    print(f'wrote {summ} and the tables')


if __name__ == '__main__':
    main()
