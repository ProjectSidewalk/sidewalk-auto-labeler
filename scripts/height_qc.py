"""Which per-pano GSV depth heights to believe: the pre-registered QC tests (issue #44).

`depth.classify_height` already rejects the payload-level fallbacks (degenerate,
synthetic_ground, implausible). This script asks the next question: of the heights whose
status is `measured`, which individual values should a raycast believe? It runs the four
tests pre-registered on #44 (the plan comment, 2026-09-25) against the bearing-only
triangulation instrument (`fuse_sites.implied_heights`, #40), and prints a verdict for
each under the rule fixed before any number was run.

The instrument, and its trap. Per pano, the implied height is the median height its own
rays imply when paired with another pano's rays in one fused site. That is independent of
any height model *per pair*, but which pairs exist depends on the association height, so
everything is computed twice -- associated `per-pano` and at a fixed 2.3 m -- and a verdict
counts only if it holds under both. The per-pano association here is the RAW measured
height (no QC rule applied): the tests are what decide the rule, so the rule must not
shape the pairs it is judged on.

Definitions (from the plan):
  k_city   self-consistent depth scale from docs/camera-height-study.md
  resid    implied - k_city * depth
  vintage  (city, capture year); its median is taken over every measured pano of that
           year. For T1 a vintage qualifies with >= 300 measured panos that have an
           implied height.

Tests and verdict rules, verbatim in substance from the plan:
  T1  within each qualifying vintage, bin panos by depth - vintage_median (<= -0.40,
      (-0.40, -0.15], (-0.15, +0.15), [+0.15, +0.40), >= +0.40 m), median implied per bin,
      and a Theil-Sen slope of implied on depth (2,000-pano deterministic subsample).
      Per city, the pano-weighted mean of the vintage slopes. Slope >= 0.6 in >= 3 of 4
      cities -> `believe the pano`; <= 0.3 in >= 3 of 4 -> `believe the vintage`; else
      `undecided`.
  T2  measured panos with depth < 1.5 m, split by tilt < 6 / >= 6 deg: median
      implied/depth. A group above 1.30 is not a measurement (-> fallback); within 1.30
      it stays measured.
  T3  gates tilt >= 6, spread >= 0.30, pixel share < 0.15, planes < 60, sky > 0.6,
      |depth - vintage_median| >= 0.40: median |resid|/implied flagged vs unflagged, and
      the flag rate over measured panos. A gate ships iff flagged/unflagged >= 2 AND the
      flag rate < 5%. Pixel share is EXPECTED TO FAIL (#44 showed it points the wrong
      way) -- recorded here before running.
  T4  |resid| by height_spread_m quartile: if its p68 does not rise monotonically across
      the quartiles, spread * SIGMA_PER_P10_P90 is not a per-pano sigma, and a constant
      p68 of |resid| over kept panos replaces it (floored at the error model's sigma).

Interpretation choices the plan left open, fixed here BEFORE the numbers ran:
  - T2, T3 and T4 are judged on the four cities POOLED (the relative residual and the
    implied/depth ratio are scale-free; T4's |resid| is in meters on every city alike);
    per-city numbers are reported beside them. T1's 3-of-4 is per city, as written.
  - "Holds under both" means: T1's verdict is the same label under both associations
    (else undecided); a T2 group is routed to fallback only if above 1.30 under both;
    a T3 gate ships only if it passes under both; T4 keeps the spread sigma only if the
    p68 rises monotonically under both, and the replacement constant is the LARGER of
    the two pooled p68s (the conservative sigma).
  - T4's "kept panos" are the measured panos that the gates adopted by T2 and T3 leave
    measured.
  - Sensitivity: 0.40 m, 6 deg, 1.30 and 2x are each re-run at x0.75 and x1.25, one at a
    time; the report says whether any verdict moves.

No GPU, no network. Reads runs/<city>/{results.jsonl, depth/index.csv}; writes
runs/<city>/height_qc/{report.md, bins.csv, gates.csv} per city and the cross-city
verdicts to runs/_pooled/height_qc/ (the same three, plus decision.csv: the default-height
decision table for docs/camera-height-study.md).

Usage:
    python scripts/height_qc.py                       # the four harvested GSV cities
    python scripts/height_qc.py paterson bend --runs-root D:/Git/sidewalk-auto-labeler/runs
"""
import argparse
import contextlib
import csv
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import depth as depthlib  # noqa: E402
import fuse_sites as fs  # noqa: E402

CITIES = ('bend', 'paterson', 'gainesville', 'sao_paulo')
# Self-consistent depth scale per city (docs/camera-height-study.md, "self-consistent
# scale" column). Fixed inputs to the pre-registered tests, not re-fitted here.
K_CITY = {'bend': 1.06, 'paterson': 1.08, 'gainesville': 1.10, 'sao_paulo': 1.16}
ASSOCIATIONS = (geo.PER_PANO, 2.3)

VINTAGE_MIN_PANOS = 300
PRE_QC_P10_P90_PER_SIGMA = 2.563   # p90-p10 of a normal, in sigmas (the pre-#44 geo rule)
THEIL_SEN_SAMPLE = 2000
SEED = 44


@dataclass(frozen=True)
class Thresholds:
    """The pre-registered thresholds; the four the sensitivity pass perturbs come first."""
    dev_m: float = 0.40          # T1 outer bin edge, T3 vintage-deviation gate
    tilt_deg: float = 6.0        # T2 split, T3 tilt gate
    ratio_max: float = 1.30      # T2
    gate_factor: float = 2.0     # T3
    dev_inner_m: float = 0.15    # T1 inner bin edge
    low_m: float = 1.5           # T2 low tail
    slope_believe: float = 0.6   # T1
    slope_noise: float = 0.3     # T1
    spread_m: float = 0.30       # T3 gates below
    pixel_share: float = 0.15
    planes: int = 60
    sky: float = 0.6
    flag_rate_max: float = 0.05  # T3


SENSITIVITY_KNOBS = ('dev_m', 'tilt_deg', 'ratio_max', 'gate_factor')


@dataclass
class Row:
    """One measured pano, with its implied height under each association (None = no pair)."""
    city: str
    pano_id: str
    year: str
    depth: float
    tilt: float | None
    spread: float | None
    pixel_share: float | None
    planes: int | None
    sky: float | None
    vintage_median: float
    implied: dict          # {association: implied height or None}


# --- data ------------------------------------------------------------------------------

def load_index_features(path):
    """{pano_id: row dict} of depth/index.csv, for the features the pano block lacks."""
    feats = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            feats[r['panorama_id']] = r
    return feats


def _f(value, cast=float):
    return None if value in (None, '') else cast(value)


@contextlib.contextmanager
def raw_heights():
    """Associate per-pano on the RAW measured height, with no QC rule applied.

    The rule is what these tests decide, so it must not shape the pairs they are judged
    on. geo.camera_height_for consults depth.believe_height (when it exists) for PER_PANO;
    this swaps in a pass-through for the duration, so the baseline is reproducible after
    the rule has shipped. The sigma is the pre-#44 one (the p90-p10 ground-plane spread
    over 2.563, floored by geo), so the baseline associations are exactly those the
    tests were run on.
    """
    original = getattr(depthlib, 'believe_height', None)
    if original is None:
        yield
        return
    depthlib.believe_height = lambda h, spread, tilt, **_: (
        h, (spread or 0.0) / PRE_QC_P10_P90_PER_SIGMA, depthlib.MEASURED)
    try:
        yield
    finally:
        depthlib.believe_height = original


def implied_by_pano(panos, camera_height):
    """{pano_id: median implied height} with the association run at `camera_height`."""
    params = fs.FuseParams(camera_height_m=camera_height)
    with raw_heights():
        sites, frame, _ = fs.fuse(panos, params)
    by_id = {p.pano_id: p for p in panos}
    return {pid: fs._median(hs)
            for pid, hs in fs.implied_heights(sites, frame, by_id).items()}


def load_city(city, runs_root):
    """(rows, unmeasured, n_panos, n_implied) for one city.

    `rows` is every measured pano, with its features; `unmeasured` is (year, that year's
    vintage median or None, {association: implied}) for the rest, which the default-height
    decision table needs (they raycast at the fallback under every option)."""
    run = Path(runs_root) / city
    panos, _ = fs.load_results(run / 'results.jsonl', run / 'depth' / 'index.csv')
    feats = load_index_features(run / 'depth' / 'index.csv')
    implied = {a: implied_by_pano(panos, a) for a in ASSOCIATIONS}

    by_year = {}
    for p in panos:
        if p.camera_height_m is not None:
            by_year.setdefault((p.capture_date or '????')[:4], []).append(p.camera_height_m)
    vmed = {y: fs._median(v) for y, v in by_year.items()}

    rows, unmeasured = [], []
    for p in panos:
        year = (p.capture_date or '????')[:4]
        if p.camera_height_m is None:
            unmeasured.append((year, vmed.get(year),
                               {a: implied[a].get(p.pano_id) for a in ASSOCIATIONS}))
            continue
        r = feats.get(p.pano_id, {})
        rows.append(Row(
            city=city, pano_id=p.pano_id, year=year, depth=p.camera_height_m,
            tilt=_f(r.get('ground_tilt_deg')), spread=p.camera_height_spread_m,
            pixel_share=_f(r.get('ground_pixel_share')), planes=_f(r.get('n_planes'), int),
            sky=_f(r.get('sky_fraction')), vintage_median=vmed[year],
            implied={a: implied[a].get(p.pano_id) for a in ASSOCIATIONS}))
    return rows, unmeasured, len(panos), {a: len(implied[a]) for a in ASSOCIATIONS}


# --- statistics (stdlib) ---------------------------------------------------------------

def median(values):
    v = sorted(values)
    return fs._median(v) if v else None


def quantile(values, q):
    v = sorted(values)
    if not v:
        return None
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def theil_sen(xs, ys, sample=THEIL_SEN_SAMPLE, seed=SEED):
    """Median of pairwise slopes; subsampled deterministically above `sample` points."""
    pts = list(zip(xs, ys))
    if len(pts) > sample:
        pts = random.Random(seed).sample(pts, sample)
    slopes = []
    for i in range(len(pts)):
        xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            dx = pts[j][0] - xi
            if dx != 0:
                slopes.append((pts[j][1] - yi) / dx)
    return median(slopes)


def monotonic_rising(values):
    return all(b > a for a, b in zip(values, values[1:]))


# --- the tests -------------------------------------------------------------------------

def dev_bin(dev, t):
    if dev <= -t.dev_m:
        return f'<= -{t.dev_m:.2f}'
    if dev <= -t.dev_inner_m:
        return f'(-{t.dev_m:.2f}, -{t.dev_inner_m:.2f}]'
    if dev < t.dev_inner_m:
        return f'(-{t.dev_inner_m:.2f}, +{t.dev_inner_m:.2f})'
    if dev < t.dev_m:
        return f'[+{t.dev_inner_m:.2f}, +{t.dev_m:.2f})'
    return f'>= +{t.dev_m:.2f}'


_SLOPES = {}


def t1(rows, assoc, t):
    """Per city: ({vintage: (n, slope)}, weighted slope), plus the bin table rows."""
    per_city, bins = {}, []
    for city in sorted({r.city for r in rows}):
        vint = {}
        for r in rows:
            if r.city == city and r.implied[assoc] is not None:
                vint.setdefault(r.year, []).append(r)
        slopes = {}
        for year, rs in sorted(vint.items()):
            if len(rs) < VINTAGE_MIN_PANOS:
                continue
            key = (city, assoc, year, len(rs))
            if key not in _SLOPES:      # thresholds never move it; sensitivity re-asks
                _SLOPES[key] = theil_sen([r.depth for r in rs],
                                         [r.implied[assoc] for r in rs])
            slope = _SLOPES[key]
            slopes[year] = (len(rs), slope)
            groups = {}
            for r in rs:
                groups.setdefault(dev_bin(r.depth - r.vintage_median, t), []).append(r)
            order = [dev_bin(d, t) for d in (-9, -(t.dev_m + t.dev_inner_m) / 2, 0,
                                             (t.dev_m + t.dev_inner_m) / 2, 9)]
            for b in order:
                g = groups.get(b, [])
                bins.append({'test': 'T1', 'city': city, 'association': assoc,
                             'vintage': year, 'bin': b, 'n': len(g),
                             'median_depth': _r(median(r.depth for r in g)),
                             'median_implied': _r(median(r.implied[assoc] for r in g)),
                             'vintage_median_depth': _r(rs[0].vintage_median),
                             'p68_abs_resid': ''})
        n = sum(n for n, _ in slopes.values())
        weighted = sum(n * s for n, s in slopes.values()) / n if n else None
        per_city[city] = (slopes, weighted)
    return per_city, bins


def t1_verdict(per_city, t):
    slopes = [w for _, w in per_city.values() if w is not None]
    if sum(s >= t.slope_believe for s in slopes) >= 3:
        return 'believe the pano'
    if sum(s <= t.slope_noise for s in slopes) >= 3:
        return 'believe the vintage'
    return 'undecided'


def rel_resid(r, assoc):
    h = r.implied[assoc]
    return abs(h - K_CITY[r.city] * r.depth) / h


def t2(rows, assoc, t):
    """{group: (n, median implied/depth)} for the low tail, split by tilt."""
    groups = {f'tilt < {t.tilt_deg:g}': [], f'tilt >= {t.tilt_deg:g}': []}
    for r in rows:
        if r.depth < t.low_m and r.implied[assoc] is not None and r.tilt is not None:
            key = f'tilt < {t.tilt_deg:g}' if r.tilt < t.tilt_deg else f'tilt >= {t.tilt_deg:g}'
            groups[key].append(r.implied[assoc] / r.depth)
    return {g: (len(v), median(v)) for g, v in groups.items()}


def gate_defs(t):
    """(name, predicate over a Row -> True/False/None when the feature is missing)."""
    def pick(attr, test):
        return lambda r: None if getattr(r, attr) is None else test(getattr(r, attr))
    return (
        (f'ground_tilt_deg >= {t.tilt_deg:g}', pick('tilt', lambda v: v >= t.tilt_deg)),
        (f'height_spread_m >= {t.spread_m:g}', pick('spread', lambda v: v >= t.spread_m)),
        (f'ground_pixel_share < {t.pixel_share:g}',
         pick('pixel_share', lambda v: v < t.pixel_share)),
        (f'depth_planes < {t.planes:g}', pick('planes', lambda v: v < t.planes)),
        (f'sky_fraction > {t.sky:g}', pick('sky', lambda v: v > t.sky)),
        (f'|depth - vintage_median| >= {t.dev_m:g}',
         lambda r: abs(r.depth - r.vintage_median) >= t.dev_m),
    )


def t3(rows, assoc, t):
    """[gate result dict] over `rows` (pooled or one city)."""
    out = []
    for name, pred in gate_defs(t):
        flags = [(pred(r), r) for r in rows]
        known = [(f, r) for f, r in flags if f is not None]
        n_flag = sum(1 for f, _ in known if f)
        fl = [rel_resid(r, assoc) for f, r in known if f and r.implied[assoc] is not None]
        un = [rel_resid(r, assoc) for f, r in known if not f and r.implied[assoc] is not None]
        mf, mu = median(fl), median(un)
        ratio = mf / mu if mf is not None and mu else None
        rate = n_flag / len(known) if known else None
        out.append({'gate': name, 'association': assoc, 'n_measured': len(known),
                    'n_flagged': n_flag, 'flag_rate': _r(rate, 4),
                    'n_flagged_implied': len(fl), 'n_unflagged_implied': len(un),
                    'med_rel_resid_flagged': _r(mf, 4),
                    'med_rel_resid_unflagged': _r(mu, 4), 'ratio': _r(ratio, 3),
                    'pass': bool(ratio is not None and ratio >= t.gate_factor
                                 and rate < t.flag_rate_max)})
    return out


def t4(rows, assoc, kept):
    """(quartile rows, rising?, p68 over kept) of |resid| by spread quartile."""
    rs = [r for r in rows if r.implied[assoc] is not None and r.spread is not None]
    if len(rs) < 4:
        return [], None, None
    rs.sort(key=lambda r: r.spread)
    q = []
    for i in range(4):
        part = rs[i * len(rs) // 4:(i + 1) * len(rs) // 4]
        res = [abs(r.implied[assoc] - K_CITY[r.city] * r.depth) for r in part]
        q.append({'quartile': i + 1, 'n': len(part),
                  'spread_lo': part[0].spread, 'spread_hi': part[-1].spread,
                  'p68': quantile(res, 0.68)})
    kept_res = [abs(r.implied[assoc] - K_CITY[r.city] * r.depth) for r in rs if kept(r)]
    return q, monotonic_rising([x['p68'] for x in q]), quantile(kept_res, 0.68)


def _r(v, nd=3):
    return '' if v is None else round(v, nd)


# --- verdicts --------------------------------------------------------------------------

def verdicts(rows, t):
    """All four verdicts for pooled `rows` under thresholds `t`, plus the raw numbers."""
    res = {'t1': {}, 't2': {}, 't3': {}, 't4': {}}
    t1_labels = {}
    for a in ASSOCIATIONS:
        per_city, _ = t1(rows, a, t)
        res['t1'][a] = per_city
        t1_labels[a] = t1_verdict(per_city, t)
    labels = set(t1_labels.values())
    res['t1_verdict'] = labels.pop() if len(labels) == 1 else 'undecided'
    res['t1_labels'] = t1_labels

    for a in ASSOCIATIONS:
        res['t2'][a] = t2(rows, a, t)
    groups = list(res['t2'][ASSOCIATIONS[0]])
    res['t2_fallback'] = [g for g in groups
                          if all(res['t2'][a][g][1] is not None
                                 and res['t2'][a][g][1] > t.ratio_max for a in ASSOCIATIONS)]

    for a in ASSOCIATIONS:
        res['t3'][a] = t3(rows, a, t)
    names = [g['gate'] for g in res['t3'][ASSOCIATIONS[0]]]
    res['t3_ship'] = [n for i, n in enumerate(names)
                      if all(res['t3'][a][i]['pass'] for a in ASSOCIATIONS)]

    kept = kept_predicate(res, t)
    rising = {}
    p68 = {}
    for a in ASSOCIATIONS:
        q, up, k = t4(rows, a, kept)
        res['t4'][a] = q
        rising[a], p68[a] = up, k
    res['t4_keep_spread'] = all(rising.values())
    res['t4_rising'] = rising
    res['t4_p68_kept'] = p68
    res['t4_sigma'] = max(v for v in p68.values() if v is not None)
    return res


def kept_predicate(res, t):
    """Row -> still measured after the T2 routes and T3 gates this result adopts."""
    preds = dict(gate_defs(t))
    gates = [preds[n] for n in res['t3_ship']]
    low_groups = res['t2_fallback']

    def kept(r):
        if r.depth < t.low_m and r.tilt is not None:
            key = f'tilt < {t.tilt_deg:g}' if r.tilt < t.tilt_deg else f'tilt >= {t.tilt_deg:g}'
            if key in low_groups:
                return False
        return not any(g(r) for g in gates)
    return kept


def summary_line(res):
    return (f"T1 {res['t1_verdict']}; T2 fallback {res['t2_fallback'] or 'none'}; "
            f"T3 ship {res['t3_ship'] or 'none'}; "
            f"T4 {'keep spread sigma' if res['t4_keep_spread'] else 'constant sigma'}")


def sensitivity(rows, base):
    """[(knob, factor, summary, moved?)] re-running every verdict with one knob scaled."""
    out = []
    ref = _verdict_key(base)
    for knob in SENSITIVITY_KNOBS:
        for factor in (0.75, 1.25):
            t = replace(Thresholds(), **{knob: getattr(Thresholds(), knob) * factor})
            res = verdicts(rows, t)
            out.append((knob, factor, getattr(t, knob), summary_line(res),
                        _verdict_key(res) != ref))
    return out


def _verdict_key(res):
    # Gate names embed their threshold, so compare which KIND of gate ships, not its text.
    return (res['t1_verdict'], tuple(g.split(' ')[0] for g in res['t2_fallback']),
            tuple(g.split(' ')[0] for g in res['t3_ship']), res['t4_keep_spread'])


# --- the default-height decision table (docs/camera-height-study.md options a/b/c) ------

LOW_VINTAGE_M = 2.1   # a vintage whose median depth is below this is the low (2025-26) rig
OPTION_B_SCALE = 1.08
OPTION_C = (2.1, 2.0, 2.5)   # depth below 2.1 m -> 2.0 m, else 2.5 m


def option_height(option, row, kept):
    """Camera height option a/b/c assigns a pano; `row` None = unmeasured."""
    default = geo.DEFAULT_CAMERA_HEIGHT_M
    if option == 'a' or row is None or not kept(row):
        return default
    if option == 'b':
        return OPTION_B_SCALE * row.depth
    cut, low, high = OPTION_C
    return low if row.depth < cut else high


def decision_table(per_city, kept):
    """[dict] per (association, city, option): signed and absolute median range error
    (H / implied - 1) on the low-vintage rig and on the rest, and the share of panos whose
    height the option changes from the 2.6 m default."""
    out = []
    for a in ASSOCIATIONS:
        for city, (rows, unmeasured, n_panos) in per_city.items():
            for option in ('a', 'b', 'c'):
                err = {'low': [], 'other': []}
                changed = 0
                for r in rows:
                    h = option_height(option, r, kept)
                    changed += h != geo.DEFAULT_CAMERA_HEIGHT_M
                    if r.implied[a] is not None:
                        rig = 'low' if r.vintage_median < LOW_VINTAGE_M else 'other'
                        err[rig].append(h / r.implied[a] - 1)
                for year, vm, imp in unmeasured:
                    if imp[a] is not None:
                        rig = 'low' if vm is not None and vm < LOW_VINTAGE_M else 'other'
                        err[rig].append(geo.DEFAULT_CAMERA_HEIGHT_M / imp[a] - 1)
                row = {'association': a, 'city': city, 'option': option,
                       'share_changed': _r(changed / n_panos, 4)}
                for rig in ('low', 'other'):
                    row[f'n_{rig}'] = len(err[rig])
                    row[f'median_err_{rig}'] = _r(median(err[rig]), 4)
                    row[f'median_abs_err_{rig}'] = _r(median(abs(e) for e in err[rig]), 4)
                out.append(row)
    return out


def decision_markdown(table, t1_verdict):
    lines = ['## Default-height decision table (rule applied; the default is NOT changed)',
             '',
             'Range error = H / implied − 1 per pano (positive = ranges run long), median '
             'signed / median absolute. "Low rig" = vintages whose median depth is below '
             f'{LOW_VINTAGE_M} m. Options: (a) 2.6 m everywhere; (b) {OPTION_B_SCALE} × the '
             f'depth height where the QC rule keeps it, else 2.6; (c) depth < {OPTION_C[0]} '
             f'→ {OPTION_C[1]} m, else {OPTION_C[2]} m where kept, else 2.6. Unmeasured and '
             f'rejected panos raycast at 2.6 m under every option. T1 said: '
             f'**{t1_verdict}**.', '']
    for a in ASSOCIATIONS:
        lines += [f'Associated at {a}:', '',
                  '| city | option | low rig (n) | low rig err | other (n) | other err | '
                  'panos changed |', '|---|---|---:|---:|---:|---:|---:|']
        for r in table:
            if r['association'] != a:
                continue
            lines.append(
                f"| {r['city']} | {r['option']} | {r['n_low']} | "
                f"{_pct(r['median_err_low'])} / {_pct(r['median_abs_err_low'])} | "
                f"{r['n_other']} | {_pct(r['median_err_other'])} / "
                f"{_pct(r['median_abs_err_other'])} | {_pct(r['share_changed'])} |")
        lines.append('')
    return '\n'.join(lines)


def _pct(v):
    return '—' if v in (None, '') else f'{100 * v:+.1f}%'


# --- output ----------------------------------------------------------------------------

def fmt(v, nd=3):
    return '—' if v is None or v == '' else f'{v:.{nd}f}'


def city_report(city, rows, n_panos, n_implied, t):
    lines = [f'# Height QC tests (#44): {city}', '',
             f'{n_panos} panos, {len(rows)} with a measured depth height; implied height '
             f'for ' + ', '.join(f'{n_implied[a]} (assoc {a})' for a in ASSOCIATIONS)
             + f'. k = {K_CITY[city]}.', '']
    bins_rows = []
    lines += ['## T1: Theil–Sen slope of implied on depth, per vintage', '',
              '| vintage | ' + ' | '.join(f'n ({a}) | slope ({a})' for a in ASSOCIATIONS)
              + ' |', '|---|' + '---:|---:|' * len(ASSOCIATIONS)]
    t1s = {}
    for a in ASSOCIATIONS:
        per_city, b = t1(rows, a, t)
        t1s[a] = per_city[city] if city in per_city else ({}, None)
        bins_rows += b
    years = sorted(set().union(*(set(v[0]) for v in t1s.values())))
    for y in years:
        cells = []
        for a in ASSOCIATIONS:
            n, s = t1s[a][0].get(y, (None, None))
            cells += [str(n) if n else '—', fmt(s)]
        lines.append(f'| {y} | ' + ' | '.join(cells) + ' |')
    lines.append('| **pano-weighted** | ' + ' | '.join(
        f'{sum(n for n, _ in t1s[a][0].values())} | {fmt(t1s[a][1])}'
        for a in ASSOCIATIONS) + ' |')
    lines += ['', 'Median implied height per deviation bin: bins.csv (test T1).', '',
              '## T2: the low tail (depth < 1.5 m), median implied/depth', '',
              '| group | ' + ' | '.join(f'n ({a}) | ratio ({a})' for a in ASSOCIATIONS)
              + ' |', '|---|' + '---:|---:|' * len(ASSOCIATIONS)]
    t2s = {a: t2(rows, a, t) for a in ASSOCIATIONS}
    for g in t2s[ASSOCIATIONS[0]]:
        lines.append(f'| {g} | ' + ' | '.join(
            f'{t2s[a][g][0]} | {fmt(t2s[a][g][1])}' for a in ASSOCIATIONS) + ' |')
    gates = []
    lines += ['', '## T3: candidate gates (median |resid|/implied, flagged vs unflagged)', '',
              '| gate | assoc | flag rate | flagged | unflagged | ratio | pass |',
              '|---|---|---:|---:|---:|---:|---|']
    for a in ASSOCIATIONS:
        for g in t3(rows, a, t):
            gates.append({'city': city, **g})
            lines.append(f"| {g['gate']} | {a} | {fmt(g['flag_rate'], 4)} | "
                         f"{fmt(g['med_rel_resid_flagged'], 4)} | "
                         f"{fmt(g['med_rel_resid_unflagged'], 4)} | {fmt(g['ratio'])} | "
                         f"{'yes' if g['pass'] else 'no'} |")
    lines += ['', '## T4: p68 of |resid| by height_spread_m quartile', '',
              '| quartile | ' + ' | '.join(f'spread range ({a}) | p68 ({a})'
                                            for a in ASSOCIATIONS) + ' |',
              '|---|' + '---|---:|' * len(ASSOCIATIONS)]
    t4s = {a: t4(rows, a, lambda r: True)[0] for a in ASSOCIATIONS}
    for i in range(4):
        cells = []
        for a in ASSOCIATIONS:
            q = t4s[a][i] if i < len(t4s[a]) else None
            cells += ([f"{q['spread_lo']:.3f}–{q['spread_hi']:.3f}", fmt(q['p68'])]
                      if q else ['—', '—'])
            if q:
                bins_rows.append({'test': 'T4', 'city': city, 'association': a,
                                  'vintage': 'all', 'bin': f"Q{q['quartile']} "
                                  f"{q['spread_lo']:.3f}-{q['spread_hi']:.3f}",
                                  'n': q['n'], 'median_depth': '', 'median_implied': '',
                                  'vintage_median_depth': '',
                                  'p68_abs_resid': _r(q['p68'], 4)})
        lines.append(f'| Q{i + 1} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Per-city numbers are context; the verdicts are read on the pooled '
              'cities (runs/_pooled/height_qc/report.md).']
    return '\n'.join(lines) + '\n', bins_rows, gates


def pooled_report(cities, res, sens, t):
    lines = [f'# Height QC verdicts (#44), pooled over {", ".join(cities)}', '',
             'Pre-registered on '
             '[#44](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/44); '
             'scripts/height_qc.py has the rules verbatim and the interpretation choices '
             'fixed before the run.', '', '## Verdicts', '',
             '| test | verdict |', '|---|---|',
             f"| T1 | **{res['t1_verdict']}** (per-pano association: "
             f"{res['t1_labels'][ASSOCIATIONS[0]]}; 2.3 m: "
             f"{res['t1_labels'][ASSOCIATIONS[1]]}) |",
             f"| T2 | fallback: {', '.join(res['t2_fallback']) or 'none'} |",
             f"| T3 | ships: {', '.join(res['t3_ship']) or 'none'} |",
             f"| T4 | {'keep spread × SIGMA_PER_P10_P90' if res['t4_keep_spread'] else 'constant sigma'}"
             f" (p68 rising: per-pano {res['t4_rising'][ASSOCIATIONS[0]]}, 2.3 m "
             f"{res['t4_rising'][ASSOCIATIONS[1]]}; kept-pano p68 "
             + ', '.join(f"{fmt(res['t4_p68_kept'][a])} ({a})" for a in ASSOCIATIONS)
             + ') |', '',
             '## T1: pano-weighted slope per city', '',
             '| city | ' + ' | '.join(f'slope ({a})' for a in ASSOCIATIONS) + ' |',
             '|---|' + '---:|' * len(ASSOCIATIONS)]
    for c in cities:
        lines.append(f'| {c} | ' + ' | '.join(
            fmt(res['t1'][a].get(c, ({}, None))[1]) for a in ASSOCIATIONS) + ' |')
    lines += ['', '## T2: low tail, pooled', '',
              '| group | ' + ' | '.join(f'n ({a}) | implied/depth ({a})'
                                         for a in ASSOCIATIONS) + ' |',
              '|---|' + '---:|---:|' * len(ASSOCIATIONS)]
    for g in res['t2'][ASSOCIATIONS[0]]:
        lines.append(f'| {g} | ' + ' | '.join(
            f"{res['t2'][a][g][0]} | {fmt(res['t2'][a][g][1])}" for a in ASSOCIATIONS)
            + ' |')
    lines += ['', '## T3: gates, pooled', '',
              '| gate | ' + ' | '.join(f'rate ({a}) | ratio ({a})' for a in ASSOCIATIONS)
              + ' | ships |', '|---|' + '---:|---:|' * len(ASSOCIATIONS) + '---|']
    for i, g in enumerate(res['t3'][ASSOCIATIONS[0]]):
        cells = []
        for a in ASSOCIATIONS:
            ga = res['t3'][a][i]
            cells += [fmt(ga['flag_rate'], 4), fmt(ga['ratio'])]
        lines.append(f"| {g['gate']} | " + ' | '.join(cells)
                     + f" | {'yes' if g['gate'] in res['t3_ship'] else 'no'} |")
    lines += ['', '## T4: p68 of |resid| (m) by spread quartile, pooled', '',
              '| quartile | ' + ' | '.join(f'spread ({a}) | p68 ({a})' for a in ASSOCIATIONS)
              + ' |', '|---|' + '---|---:|' * len(ASSOCIATIONS)]
    for i in range(4):
        cells = []
        for a in ASSOCIATIONS:
            q = res['t4'][a][i]
            cells += [f"{q['spread_lo']:.3f}–{q['spread_hi']:.3f}", fmt(q['p68'])]
        lines.append(f'| Q{i + 1} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Sensitivity (one threshold at a time, ×0.75 and ×1.25)', '',
              '| knob | value | verdicts | moved? |', '|---|---:|---|---|']
    for knob, factor, value, text, moved in sens:
        lines.append(f'| {knob} ×{factor} | {value:g} | {text} | '
                     f"{'**yes**' if moved else 'no'} |")
    return '\n'.join(lines) + '\n'


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cities', nargs='*', default=list(CITIES))
    ap.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if any(c not in K_CITY for c in args.cities):
        ap.error(f'k_city is only defined for {", ".join(K_CITY)}')

    t = Thresholds()
    all_rows, all_bins, all_gates, per_city = [], [], [], {}
    for city in args.cities:
        print(f'{city}: loading and fusing twice ...', file=sys.stderr)
        rows, unmeasured, n_panos, n_implied = load_city(city, args.runs_root)
        per_city[city] = (rows, unmeasured, n_panos)
        text, bins_rows, gates = city_report(city, rows, n_panos, n_implied, t)
        out = args.runs_root / city / 'height_qc'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.md').write_text(text, encoding='utf-8')
        write_csv(out / 'bins.csv', bins_rows)
        write_csv(out / 'gates.csv', gates)
        print(text)
        all_rows += rows
        all_bins += bins_rows
        all_gates += gates

    res = verdicts(all_rows, t)
    sens = sensitivity(all_rows, res)
    table = decision_table(per_city, kept_predicate(res, t))
    text = (pooled_report(args.cities, res, sens, t) + '\n'
            + decision_markdown(table, res['t1_verdict']))
    out = args.runs_root / '_pooled' / 'height_qc'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.md').write_text(text, encoding='utf-8')
    write_csv(out / 'bins.csv', all_bins)
    write_csv(out / 'gates.csv', [{'city': 'pooled', **g}
                                  for a in ASSOCIATIONS for g in res['t3'][a]])
    write_csv(out / 'decision.csv', table)
    print(text)


if __name__ == '__main__':
    main()
