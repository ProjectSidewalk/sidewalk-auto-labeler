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
    # all five Mapillary runs, the gate, and the tables (the #53 measurement)
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


def _null_views(sv, rng):
    """One site's views re-drawn under the error model's own noise and NO scale error
    (reprojection_residual.null_study's simulation, verbatim in effect)."""
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
        sim.append(rr.View(v.pano_id, v.det_index, v.x, v.y, v.conf, pe, pn, v.cov,
                           math.hypot(de, dn), math.degrees(math.atan2(de, dn)) % 360,
                           v.source, v.capture_date))
    return sim


def _scale_blocks(sites, key_of, names, offset=False, simulate=None):
    """fit_scale blocks for a joint per-group scale fit (plus a pooled dip-offset column
    when offset=True). key_of maps a view's pano to a name in `names`."""
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
                    views, [geo.DEFAULT_CAMERA_HEIGHT_M] * len(views))):
                cols = [(0.0, 0.0)] * len(names)
                cols[idx[g]] = sc
                levers.append(cols + [off])
            design = rr.lever_design(views, levers)
        else:
            design = rr.scale_design(views, groups, names)
        blocks.append([(cols, (v.e - he, v.n - hn))
                       for v, ((he, hn), _), cols in zip(views, loo, design)])
    return blocks


def instrument_b(sites, key_of, min_rows=rr.MIN_GROUP_ROWS, seed=0):
    """Joint per-group range scale at 2.6 m, null-corrected per group. Groups with fewer
    than min_rows held-out views are fitted together as 'other'. Returns {key: row}."""
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
                                      simulate=lambda sv: _null_views(sv, rng)), names)
    off = rr.fit_scale(_scale_blocks(sites, kof, names, offset=True),
                       names + ['__dip_offset__'])
    h0 = geo.DEFAULT_CAMERA_HEIGHT_M
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
               'h_scale': h0 * (1.0 - net),                 # = 2.6 / k_net
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


# --- The decision rule ------------------------------------------------------------------

def decide_group(a, b, support_scale=1.0):
    """Apply rules 1-4 to one group's instrument A row `a` and B row `b` (either None).
    Returns (height_m or None, passed [rule names], reason)."""
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
    h_b = b['h_scale'] if b else None
    if h_a is not None and h_b is not None and abs(h_a - h_b) <= RULE_MAX_DISAGREE_M:
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
    h0 = geo.DEFAULT_CAMERA_HEIGHT_M
    h_a = a['h_star'] if a else None
    h_b = b['h_scale'] if b else None
    sigma = None
    if h is not None:
        sd = a['boot_sd'] or 0.0
        sigma = math.sqrt(sd ** 2 + ((h_a - h_b) / 2) ** 2)
    suspect = h_a is not None and not (SANE_HEIGHT_M[0] <= h_a <= SANE_HEIGHT_M[1])
    return {'height_m': h0 if h is None else round(h, 3),
            'sigma_m': None if sigma is None else round(sigma, 3),
            'applied': h is not None, 'grain': grain,
            'n_panos': a['n_panos'] if a else 0, 'n_sites': a['n_sites'] if a else 0,
            'n_views_b': b['n_views'] if b else 0,
            'h_bearing': _rd(h_a), 'h_bearing_ci': None if not a or a['ci_lo'] is None
            else [_rd(a['ci_lo']), _rd(a['ci_hi'])],
            'h_scale': _rd(h_b), 'h_scale_ci': None if not b
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

def measure_city(city, run_dir, n_boot=N_BOOT, log=print):
    """Instruments A and B over every grouping, the rule, and the table body.
    Returns a dict with rows per grouping, the table and notes."""
    jsonl = run_dir / 'results.jsonl'
    panos, _ = fs.load_results(jsonl, read_heights=False)
    other = {p.source for p in panos if p.source not in geo.CROWDSOURCED_SOURCES}
    if other:
        raise ValueError(f'{city}: per-rig heights are for crowdsourced runs; holds {other}')
    groups_of = pano_groups(jsonl)
    params = production_params()
    log(f'{city}: {len(panos)} panos; sweeping association heights {SWEEP_HEIGHTS} ...')
    sweep = sweep_implied(panos, params)
    kinds = ('rig', 'creator', 'speed', 'sequence')
    a_rows = {}
    for kind in kinds:
        a_rows[kind] = instrument_a(
            sweep, lambda pid, k=kind: groups_of[pid][k], n_boot=n_boot,
            boot_keys=set() if kind == 'sequence' else None)
    log(f'{city}: instrument B ...')
    sites, _frame = site_views(panos, replace(params, camera_height_m=geo.DEFAULT_CAMERA_HEIGHT_M))
    b_rows = {kind: instrument_b(sites, lambda pid, k=kind: groups_of[pid][k])
              for kind in ('rig', 'creator', 'speed')}
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
        # so they need the full instruments (CI for A, a joint fit for B).
        cand = {g['sequence'] for g in groups_of.values() if g['rig'] in seq_rigs}
        a_seq_boot = instrument_a(sweep, lambda pid: groups_of[pid]['sequence'],
                                  n_boot=n_boot, boot_keys=cand)

        def seq_or_rig(pid):
            g = groups_of[pid]
            return g['sequence'] if g['sequence'] in cand else g['rig']
        b_seq = instrument_b(sites, seq_or_rig)
        table, notes = build_table(groups_of, a_rows['rig'], b_rows['rig'],
                                   a_rows['sequence'], seq_views, a_seq_boot, b_seq)
        a_rows['sequence'] = {**a_rows['sequence'], **{k: v for k, v in a_seq_boot.items()
                                                       if k in cand}}
        b_rows['sequence'] = b_seq
    panos_by_group = {}
    for g in groups_of.values():
        for kind in kinds:
            panos_by_group[kind, g[kind]] = panos_by_group.get((kind, g[kind]), 0) + 1
    return {'city': city, 'panos': panos, 'groups_of': groups_of, 'a': a_rows, 'b': b_rows,
            'table': table, 'notes': notes, 'pooled_k': pooled, 'pooled_k_bench': pooled_bench,
            'panos_by_group': panos_by_group, 'seq_views': seq_views}


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
                   'h_scale_with_offset': _rd(rb.get('h_scale_with_offset'))}
            for h in SWEEP_HEIGHTS:
                row[f'implied_at_{h:g}'] = _rd(ra.get(f'implied_at_{h:g}'))
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
    body['generated_by'] = 'scripts/mapillary_height.py (issue #53)'
    order = ['schema', 'source', 'default_m', 'grain', 'recommended', 'gate', 'results_sha256',
             'generated_at', 'generated_by', 'groups', 'sequences']
    body = {k: body[k] for k in order if k in body}
    path = run_dir / fs.HEIGHT_TABLE_NAME
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(body, f, indent=1, sort_keys=False)
        f.write('\n')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cities', nargs='+', help='Mapillary run names under --run-root')
    ap.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--no-gate', action='store_true',
                    help='measure and write tables only (recommended: false)')
    args = ap.parse_args()

    measured, k_rows = {}, []
    for city in args.cities:
        run_dir = args.run_root / city
        m = measure_city(city, run_dir, n_boot=args.n_boot)
        measured[city] = m
        out = run_dir / 'camera_height'
        write_csv(out / 'groups.csv', group_rows(m))
        write_table(run_dir, m['table'], recommended=False)
        k, s, s0 = m['pooled_k']
        kb, sb, s0b = m['pooled_k_bench']
        k_rows.append({'city': city, 'k_net_production': _rd(k, 4), 's_production': _rd(s, 4),
                       's_null_production': _rd(s0, 4), 'k_net_benchmark': _rd(kb, 4),
                       's_benchmark': _rd(sb, 4), 's_null_benchmark': _rd(s0b, 4)})
        print(f'{city}: pooled k_net {k:.3f} at the production tier, {kb:.3f} at the '
              f'benchmark tier (#76 tie-back)')
        for rig, n in m['notes'].items():
            print(f"  {rig}: grain {n['grain']} ({n['qualifying_sequences']} qualifying "
                  f"sequences, h* IQR {_f(n['sequence_iqr_m'], '.3f')} m)")
        for key, g in m['table']['groups'].items():
            print(f"  {key}: {g['height_m']} m (applied {g['applied']}); A {g['h_bearing']} "
                  f"{g['h_bearing_ci']} slope {g['slope']}; B {g['h_scale']}; {g['reason']}")
    summ = args.run_root / '_summary' / 'camera_height'
    write_csv(summ / 'k_net.csv', k_rows)
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
