"""Does RampNet's recall on open-imagery rigs depend on pano tilt? (overnight `recall_by_tilt`)

CPU only, no network. Reads judged RampNet bundles (records.jsonl + verdicts.json) as data,
builds one row per GT ramp (verdict-true detection at >= 0.55, or a confident missed mark),
attaches the pano's pose in RampNet's #113 convention (pitch > 0 nose down, roll > 0 left
side up) and per-ramp covariates, and applies the decision rule pre-registered in
docs/overnight/recall_by_tilt.md section 0:

1. recall(|T| Q1) - recall(|T| Q4) >= 0.10 with a pano-cluster bootstrap CI excluding 0;
2. the |T| coefficient of detected ~ |T| + rig + gravity-dip bucket stays below 0 (cluster
   bootstrap);
3. a pose shuffle within rig x city (200 draws) reaches that gap in < 10 draws.

T is the elevation of the gravity horizon above the image's middle row at the ramp's column
(first order: T = pitch cos b + roll sin b), computed exactly with geo._world_ray's rotation.

Pose sources (no imputation; a pano without one is left out of every tilt test):
- Mapillary: geo.mapillary_pitch_roll(computed_rotation); pitch113 = -pitch, roll113 = roll.
- Lyon: RampNet's image-based estimate (analysis_out/pano_tilt/lyon.json, status ok).
- Bayonne: Panoramax panoflat pers:pitch/pers:roll (non-0/0 only); pitch113 = -pers:pitch.
- GSV: record camera_pitch / wrapped camera_roll.

Usage:
    python scripts/recall_by_tilt.py \
        --benchmark-root ../RampNet/benchmark \
        --lyon-tilt ../RampNet/analysis_out/pano_tilt/lyon.json \
        --out docs/overnight/data/recall_by_tilt
    python scripts/recall_by_tilt.py ... --quick     # 200 bootstrap draws, 20 shuffles (smoke)

Deterministic: every random draw comes from numpy Generators seeded from --seed.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE, registry  # noqa: E402

OPEN_SPLITS = ['laurens_mapillary', 'bayonne', 'juneau', 'lyon', 'richmond', 'clovis',
               'morgantown', 'annapolis']
HELD_OUT = ['budapest_district5']
GSV_SPLITS = ['laurens_gsv', 'paterson', 'bend', 'gainesville', 'sao_paulo']
PANORAMAX_SPLITS = {'bayonne', 'lyon'}

DIP_EDGES = [4.0, 6.0, 10.0, 20.0]          # buckets: <4, 4-6, 6-10, 10-20, >=20
DIP_LABELS = ['<4', '4-6', '6-10', '10-20', '>=20']
DIP_REF = '10-20'
SIGNED_EDGES = [-3.0, -1.0, 1.0, 3.0]
SIGNED_LABELS = ['<=-3', '-3..-1', '-1..1', '1..3', '>=3']
RIG_REF = 'gopro_max'
ABS_T_CAP = 20.0
COEF_SCALE = 5.0                            # coefficient is per 5 degrees
GAP_BAR = 0.10
SHUFFLE_FAIL_DRAWS = 10


# --------------------------------------------------------------------------------------
# Loading

def rig_class(split, pano):
    if split in GSV_SPLITS:
        return 'gsv'
    make = str(pano.get('camera_make') or '').lower()
    model = str(pano.get('camera_model') or '').lower()
    if model in ('', 'none'):
        return 'unknown'
    if 'max2' in model:
        return 'other_360'
    if 'gopro' in make and 'max' in model:
        return 'gopro_max'
    if 'fusion' in model:
        return 'gopro_fusion'
    if 'mx7' in model:
        return 'trimble_mx7'
    if 'pulsar' in model:
        return 'istar_pulsar'
    if 'ladybug' in model:
        return 'ladybug'
    return 'other_360'


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def pose_113(split, pano, lyon_tilt):
    """(pitch113, roll113, source) or (None, None, reason)."""
    if split in GSV_SPLITS:
        p, r = pano.get('camera_pitch'), pano.get('camera_roll')
        if p is None or r is None:
            return None, None, 'gsv_missing'
        return float(p), wrap180(r), 'gsv_record'
    if split == 'lyon':
        row = lyon_tilt.get(pano['panorama_id'])
        if row is None:
            return None, None, 'lyon_no_row'
        if row['status'] != 'ok':
            return None, None, 'lyon_declined'
        return float(row['pitch_deg']), float(row['roll_deg']), 'rampnet_image_estimate'
    sm = pano.get('source_metadata') or {}
    if split in PANORAMAX_SPLITS:
        pp, pr = sm.get('pers:pitch'), sm.get('pers:roll')
        if pp is None or pr is None:
            return None, None, 'panoramax_absent'
        if float(pp) == 0.0 and float(pr) == 0.0:
            return None, None, 'panoramax_zero'
        sem = {d.get('key'): d.get('value') for d in sm.get('semantics') or []
               if isinstance(d, dict)}
        if not str(sem.get('computed_by[pitch]', '')).startswith('panoflat'):
            return None, None, 'panoramax_not_panoflat'
        return -float(pp), float(pr), 'panoramax_panoflat'
    status = geo.mapillary_pose_status(sm.get('computed_rotation'))
    if status != geo.POSE_OK:
        return None, None, f'mapillary_{status}'
    p, r = geo.mapillary_pitch_roll(sm['computed_rotation'])
    return -p, r, 'mapillary_computed_rotation'


def load_lyon_tilt(path):
    """{pano_id: row} from RampNet's Lyon tilt estimate. A missing file RAISES: silently
    reading Lyon as unposed would change every pooled number and overwrite the tracked
    outputs with them."""
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f'--lyon-tilt {path}: not found. It is RampNet analysis_out/pano_tilt/lyon.json '
            '(the Lyon pose source); pass its path explicitly.')
    with open(path, encoding='utf-8') as f:
        return {r['pano_id']: r for r in json.load(f)['rows']}


def load_split(split, bench_root, lyon_tilt, exclude_top):
    """Rows (one per GT ramp) and pano rows (one per pool pano) for one split."""
    recs = {}
    records_path = bench_root / split / 'records.jsonl'
    with open(records_path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                # eval_sites.load_benchmark's check: a CurbRamp score needs a CurbRamp bundle
                types = sorted(registry.record_types(r))
                if types != [registry.CURB_RAMP]:
                    raise ValueError(f'{records_path} line {lineno}: the bundle record ran '
                                     f'{types}, but this scores {registry.CURB_RAMP}')
                recs[r['pano']['panorama_id']] = r
    with open(bench_root / split / 'verdicts.json', encoding='utf-8') as f:
        verdicts = json.load(f)['panos']
    units, panos = [], []
    for pid in sorted(verdicts):
        e = verdicts[pid]
        if exclude_top and e.get('group') == 'top':
            continue
        rec = recs.get(pid)
        if rec is None or len(rec['detections']) != len(e['dets']):
            continue
        if any(d is None for d in e['dets']):
            continue
        checked = (e['no_missed'] or e['missed']) if 'no_missed' in e else True
        if not checked:
            continue
        pano = rec['pano']
        p113, r113, psrc = pose_113(split, pano, lyon_tilt)
        rig = rig_class(split, pano)
        year = str(pano.get('capture_date') or '')[:4] or 'unknown'
        prow = dict(split=split, pano_id=pid, rig=rig, year=year, group=e.get('group'),
                    posed=p113 is not None, pose_source=psrc,
                    pitch113=p113, roll113=r113)
        panos.append(prow)
        pts = []
        for d, v in zip(rec['detections'], e['dets']):
            if v is True:
                assert d['confidence'] >= BENCHMARK_CONFIDENCE, (split, pid)
                pts.append(('det', 1, d['x_normalized'], d['y_normalized']))
        for m in e.get('missed') or []:
            if not m.get('unsure'):
                pts.append(('missed', 0, float(m['x']), float(m['y'])))
        for kind, det, x, y in pts:
            units.append(dict(prow, kind=kind, detected=det, x=float(x), y=float(y)))
    return units, panos


# --------------------------------------------------------------------------------------
# Geometry (vectorized copy of geo._world_ray, checked against it on every unit)

def gravity_elev_deg(x, y, pitch113, roll113):
    """Gravity elevation (deg) of normalized pano points under a #113 pose (arrays)."""
    phi = (np.asarray(x) - 0.5) * 2.0 * np.pi
    theta = (0.5 - np.asarray(y)) * np.pi
    alpha = np.radians(-np.asarray(pitch113))    # geo: pitch > 0 raises the view axis
    rho = -np.radians(np.asarray(roll113))       # geo: PS roll in, geometric rotation out
    ca, sa, cr, sr = np.cos(alpha), np.sin(alpha), np.cos(rho), np.sin(rho)
    # heading 0: f=(1,0,0) r=(0,1,0) u=(0,0,1) in (north, east, up); only the up component
    # of the final vector is needed.
    f_up = sa                     # after pitch: f = ca*f0 + sa*u0
    u_up_p = ca                   # after pitch: u = -sa*f0 + ca*u0
    r_up = sr * u_up_p            # after roll: r = cr*r0 + sr*u
    u_up = cr * u_up_p            # after roll: u = -sr*r0 + cr*u
    ct = np.cos(theta)
    v_up = ct * np.cos(phi) * f_up + ct * np.sin(phi) * r_up + np.sin(theta) * u_up
    return np.degrees(np.arcsin(np.clip(v_up, -1.0, 1.0)))


def check_against_geo(units):
    worst = 0.0
    for u in units:
        if not u['posed']:
            continue
        pose = geo.Pose(0.0, 0.0, 0.0, -u['pitch113'], u['roll113'], True, '')
        elev, _ = geo._world_ray(pose, (u['x'] - 0.5) * 2 * math.pi, (0.5 - u['y']) * math.pi)
        mine = float(gravity_elev_deg(u['x'], u['y'], u['pitch113'], u['roll113']))
        worst = max(worst, abs(math.degrees(elev) - mine))
    return worst


def add_covariates(units):
    for u in units:
        u['img_dip'] = (u['y'] - 0.5) * 180.0
        b = (u['x'] - 0.5) * 2.0 * math.pi
        if u['posed']:
            p, r = u['pitch113'], u['roll113']
            u['T'] = -float(gravity_elev_deg(u['x'], 0.5, p, r))
            u['T1'] = p * math.cos(b) + r * math.sin(b)
            u['grav_dip'] = -float(gravity_elev_deg(u['x'], u['y'], p, r))
            u['tilt'] = math.degrees(math.acos(math.cos(math.radians(p)) *
                                               math.cos(math.radians(r))))
            u['inplane'] = -p * math.sin(b) + r * math.cos(b)
        else:
            u['T'] = u['T1'] = u['grav_dip'] = u['tilt'] = u['inplane'] = None


def dip_bucket(d):
    for edge, lab in zip(DIP_EDGES, DIP_LABELS):
        if d < edge:
            return lab
    return DIP_LABELS[-1]


def signed_bucket(t):
    if t <= SIGNED_EDGES[0]:
        return SIGNED_LABELS[0]
    if t < SIGNED_EDGES[1]:
        return SIGNED_LABELS[1]
    if t < SIGNED_EDGES[2]:
        return SIGNED_LABELS[2]
    if t < SIGNED_EDGES[3]:
        return SIGNED_LABELS[3]
    return SIGNED_LABELS[4]


# --------------------------------------------------------------------------------------
# Statistics

class Clusters:
    """Pano-cluster bootstrap weights, panos resampled within each split."""

    def __init__(self, units, seed):
        keys = sorted({(u['split'], u['pano_id']) for u in units})
        self.index = {k: i for i, k in enumerate(keys)}
        self.unit_pano = np.array([self.index[(u['split'], u['pano_id'])] for u in units])
        by_split = defaultdict(list)
        for k, i in self.index.items():
            by_split[k[0]].append(i)
        self.by_split = [np.array(v) for _, v in sorted(by_split.items())]
        self.n_panos = len(keys)
        self.rng = np.random.default_rng(seed)

    def draw(self):
        w = np.zeros(self.n_panos)
        for ids in self.by_split:
            np.add.at(w, self.rng.choice(ids, size=len(ids), replace=True), 1.0)
        return w[self.unit_pano]


def wrecall(y, mask, w=None):
    w = np.ones_like(y, dtype=float) if w is None else w
    den = np.sum(w * mask)
    return float(np.sum(w * mask * y) / den) if den > 0 else float('nan')


def quartile_edges(values):
    return [float(q) for q in np.quantile(values, [0.25, 0.5, 0.75])]


def quartile_of(values, edges):
    return np.searchsorted(np.asarray(edges), values, side='right')   # 0..3


def gap_q1_q4(y, q, w=None):
    return wrecall(y, q == 0, w) - wrecall(y, q == 3, w)


def logistic_fit(X, y, w=None, ridge=1e-4, iters=50):
    """Weighted logistic regression by IRLS (ridge on all but the intercept)."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(k)
    pen = np.full(k, ridge)
    pen[0] = 0.0
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = w * mu * (1 - mu)
        grad = X.T @ (w * (y - mu)) - pen * beta
        H = (X * s[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def design(units, tilt_key, rigs, dip_key='grav_dip', city_fe=False, within=False,
           bearing_term=False):
    """Design matrix: intercept, tilt term(s), rig dummies, dip dummies, [city dummies],
    [|cos b|, the post hoc front/back-vs-side bearing term]."""
    cols = ['intercept']
    x_t = np.array([min(abs(u[tilt_key]), ABS_T_CAP) / COEF_SCALE for u in units])
    blocks = [np.ones(len(units))]
    if within:
        pm = defaultdict(list)
        for u, v in zip(units, x_t):
            pm[(u['split'], u['pano_id'])].append(v)
        mean = {k: float(np.mean(v)) for k, v in pm.items()}
        between = np.array([mean[(u['split'], u['pano_id'])] for u in units])
        blocks += [x_t - between, between]
        cols += [f'{tilt_key}_within', f'{tilt_key}_between']
    else:
        blocks.append(x_t)
        cols.append(tilt_key)
    for rig in rigs:
        if rig == RIG_REF:
            continue
        blocks.append(np.array([1.0 if u['rig'] == rig else 0.0 for u in units]))
        cols.append(f'rig={rig}')
    for lab in DIP_LABELS:
        if lab == DIP_REF:
            continue
        blocks.append(np.array([1.0 if dip_bucket(u[dip_key]) == lab else 0.0 for u in units]))
        cols.append(f'dip={lab}')
    if bearing_term:
        blocks.append(np.array([abs(math.cos((u['x'] - 0.5) * 2 * math.pi)) for u in units]))
        cols.append('abs_cos_bearing')
    if city_fe:
        cities = sorted({u['split'] for u in units})
        for c in cities[1:]:
            blocks.append(np.array([1.0 if u['split'] == c else 0.0 for u in units]))
            cols.append(f'city={c}')
    return np.column_stack(blocks), cols


def boot_ci(samples):
    a = np.asarray(samples)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


# --------------------------------------------------------------------------------------
# Tables

def recall_rows(units, key_fn, label, extra=None):
    """[(group..., n, detected, recall)] for units grouped by key_fn."""
    agg = defaultdict(lambda: [0, 0, set()])
    for u in units:
        k = key_fn(u)
        if k is None:
            continue
        a = agg[k]
        a[0] += 1
        a[1] += u['detected']
        a[2].add((u['split'], u['pano_id']))
    rows = []
    for k in sorted(agg, key=lambda z: tuple(str(t) for t in (z if isinstance(z, tuple) else (z,)))):
        n, d, ps = agg[k]
        kk = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(label, kk), n_gt=n, detected=d, panos=len(ps),
                         recall=round(d / n, 4) if n else None))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def fmt(v, nd=3):
    return 'nan' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f'{v:.{nd}f}'


# --------------------------------------------------------------------------------------
# The three tests

def bucket_test(units, key, n_boot, seed, edges=None):
    vals = np.array([abs(u[key]) for u in units])
    y = np.array([u['detected'] for u in units], dtype=float)
    edges = edges or quartile_edges(vals)
    q = quartile_of(vals, edges)
    obs = gap_q1_q4(y, q)
    cl = Clusters(units, seed)
    boots = [gap_q1_q4(y, q, cl.draw()) for _ in range(n_boot)]
    per_q = []
    for i in range(4):
        m = q == i
        lo = 0.0 if i == 0 else edges[i - 1]
        hi = edges[i] if i < 3 else float(vals.max())
        per_q.append(dict(quartile=f'Q{i + 1}', lo_deg=round(lo, 3), hi_deg=round(hi, 3),
                          n_gt=int(m.sum()), detected=int(y[m].sum()),
                          recall=round(float(y[m].mean()), 4) if m.any() else None))
    return dict(edges=edges, gap=obs, ci=boot_ci(boots), per_quartile=per_q)


def stratified_gap(units, key, edges, n_boot, seed):
    """POST HOC: Q1 - Q4 gap within each split (pooled edges), combined with
    Mantel-Haenszel-style weights n1*n4/(n1+n4); pano-cluster bootstrap CI."""
    vals = np.array([abs(u[key]) for u in units])
    y = np.array([u['detected'] for u in units], dtype=float)
    q = quartile_of(vals, edges)
    split = np.array([u['split'] for u in units])
    names = sorted(set(split))

    def stat(w):
        num = den = 0.0
        for s_ in names:
            m = split == s_
            n1, n4 = np.sum(w * m * (q == 0)), np.sum(w * m * (q == 3))
            if n1 <= 0 or n4 <= 0:
                continue
            g = wrecall(y, m & (q == 0), w) - wrecall(y, m & (q == 3), w)
            wt = n1 * n4 / (n1 + n4)
            num += wt * g
            den += wt
        return num / den if den else float('nan')
    cl = Clusters(units, seed)
    return dict(gap=stat(np.ones(len(units))), ci=boot_ci([stat(cl.draw()) for _ in range(n_boot)]))


def logistic_test(units, key, n_boot, seed, **kw):
    rigs = sorted({u['rig'] for u in units})
    X, cols = design(units, key, rigs, **kw)
    y = np.array([u['detected'] for u in units], dtype=float)
    beta = logistic_fit(X, y)
    cl = Clusters(units, seed)
    bs = np.array([logistic_fit(X, y, cl.draw()) for _ in range(n_boot)])
    out = []
    for j, c in enumerate(cols):
        lo, hi = boot_ci(bs[:, j])
        out.append(dict(term=c, coef=round(float(beta[j]), 4), ci_lo=round(lo, 4),
                        ci_hi=round(hi, 4)))
    return out


def shuffle_test(units, key, edges, n_draws, seed, obs_gap):
    """Shuffle (pitch, roll) among posed panos within rig x city; recompute the gap and the
    logistic |T| coefficient (point fit) per draw."""
    rng = np.random.default_rng(seed)
    pano_keys = sorted({(u['split'], u['pano_id']) for u in units})
    pose = {}
    stratum = defaultdict(list)
    for u in units:
        k = (u['split'], u['pano_id'])
        pose[k] = (u['pitch113'], u['roll113'])
    rig_of = {(u['split'], u['pano_id']): u['rig'] for u in units}
    for k in pano_keys:
        stratum[(rig_of[k], k[0])].append(k)
    x = np.array([u['x'] for u in units])
    yy = np.array([u['y'] for u in units])
    det = np.array([u['detected'] for u in units], dtype=float)
    upk = [(u['split'], u['pano_id']) for u in units]
    rigs = sorted({u['rig'] for u in units})
    rows = []
    for draw in range(n_draws):
        new = {}
        for keys in stratum.values():
            perm = rng.permutation(len(keys))
            for k, j in zip(keys, perm):
                new[k] = pose[keys[j]]
        p = np.array([new[k][0] for k in upk])
        r = np.array([new[k][1] for k in upk])
        if key == 'T':
            val = np.abs(-gravity_elev_deg(x, 0.5, p, r))
        else:
            val = np.degrees(np.arccos(np.cos(np.radians(p)) * np.cos(np.radians(r))))
        q = quartile_of(val, edges)
        gap = gap_q1_q4(det, q)
        gdip = -gravity_elev_deg(x, yy, p, r)
        shuffled = [dict(u, **{key: float(v), 'grav_dip': float(g)})
                    for u, v, g in zip(units, val, gdip)]
        X, cols = design(shuffled, key, rigs)
        beta = logistic_fit(X, det)
        rows.append(dict(draw=draw, gap=round(gap, 5), coef=round(float(beta[1]), 5)))
    gaps = np.array([r['gap'] for r in rows])
    return dict(rows=rows, n_reach_bar=int(np.sum(gaps >= GAP_BAR)),
                perm_p=float((1 + np.sum(gaps >= obs_gap)) / (1 + len(gaps))),
                gap_median=float(np.median(gaps)), gap_q95=float(np.quantile(gaps, 0.95)))


def horizon_check(units, flip=False):
    """GT points above the horizon: image frame vs gravity frame (convention check)."""
    img = sum(1 for u in units if u['img_dip'] < 0)
    s = -1.0 if flip else 1.0
    grav = 0
    for u in units:
        if u['posed']:
            d = -float(gravity_elev_deg(u['x'], u['y'], s * u['pitch113'], s * u['roll113']))
        else:
            d = u['img_dip']
        grav += d < 0
    return img, grav


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--lyon-tilt', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'analysis_out' / 'pano_tilt' / 'lyon.json')
    ap.add_argument('--out', type=Path, default=REPO_ROOT / 'docs/overnight/data/recall_by_tilt')
    ap.add_argument('--seed', type=int, default=20261126)
    ap.add_argument('--quick', action='store_true', help='smoke run: few draws')
    args = ap.parse_args()
    n_boot_gap, n_boot_log, n_shuffle = (200, 100, 20) if args.quick else (2000, 1000, 200)
    args.out.mkdir(parents=True, exist_ok=True)
    lyon_tilt = load_lyon_tilt(args.lyon_tilt)

    summary = {'seed': args.seed, 'n_boot_gap': n_boot_gap, 'n_boot_logistic': n_boot_log,
               'n_shuffle': n_shuffle, 'populations': {}}
    all_splits = OPEN_SPLITS + HELD_OUT + GSV_SPLITS
    pop_units = {}
    for label, exclude_top in (('unbiased', True), ('all_panos', False)):
        units = []
        for s in all_splits:
            u, _ = load_split(s, args.benchmark_root, lyon_tilt, exclude_top)
            units += u
        add_covariates(units)
        pop_units[label] = units
    units_ub = pop_units['unbiased']
    summary['geo_check_max_abs_deg'] = check_against_geo(units_ub)

    # Units CSV (unbiased subset; `group` lets anyone rebuild all-panos except top)
    write_csv(args.out / 'units_unbiased.csv', [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in u.items()
         if k not in ('posed',)} | {'posed': int(u['posed'])} for u in units_ub])

    # Coverage
    cov = []
    for s in all_splits:
        us = [u for u in units_ub if u['split'] == s]
        srcs = Counter(u['pose_source'] for u in us)
        posed = [u for u in us if u['posed']]
        unposed = [u for u in us if not u['posed']]
        cov.append(dict(split=s, gt=len(us), posed_gt=len(posed),
                        posed_share=round(len(posed) / len(us), 3) if us else None,
                        recall_posed=round(np.mean([u['detected'] for u in posed]), 4) if posed else None,
                        recall_unposed=round(np.mean([u['detected'] for u in unposed]), 4) if unposed else None,
                        recall_all=round(np.mean([u['detected'] for u in us]), 4) if us else None,
                        pose_sources=';'.join(f'{k}:{v}' for k, v in sorted(srcs.items()))))
    write_csv(args.out / 'coverage.csv', cov)

    # Convention checks
    hc = {}
    for s in all_splits:
        us = [u for u in units_ub if u['split'] == s]
        img, grav = horizon_check(us)
        _, flip = horizon_check(us, flip=True)
        hc[s] = dict(image=img, gravity=grav, flipped=flip)
    # Lyon reproduction of lyon_review_159's 87 -> 2 (declined read level, same pool)
    hc_rows = [dict(split=k, **v) for k, v in hc.items()]
    write_csv(args.out / 'horizon_check.csv', hc_rows)
    summary['horizon_check'] = hc

    def tilt_tables(units, tag):
        posed = [u for u in units if u['posed']]
        return posed

    results = {}
    for pop in ('unbiased', 'all_panos'):
        units = pop_units[pop]
        open_posed = [u for u in units if u['split'] in OPEN_SPLITS and u['posed']]
        gsv_posed = [u for u in units if u['split'] in GSV_SPLITS and u['posed']]
        bud_posed = [u for u in units if u['split'] in HELD_OUT and u['posed']]
        res = {}
        for key in ('T', 'tilt'):
            bt = bucket_test(open_posed, key, n_boot_gap, args.seed)
            lt = logistic_test(open_posed, key, n_boot_log, args.seed + 1)
            res[key] = dict(bucket=bt, logistic=lt)
            if pop == 'unbiased':
                sh = shuffle_test(open_posed, key, bt['edges'], n_shuffle, args.seed + 2,
                                  bt['gap'])
                res[key]['shuffle'] = {k: v for k, v in sh.items() if k != 'rows'}
                write_csv(args.out / f'shuffle_{key}.csv', sh['rows'])
                # GSV contrast and budapest, descriptive
                res[key]['gsv_bucket'] = bucket_test(gsv_posed, key, n_boot_gap, args.seed)
                res[key]['budapest_bucket'] = bucket_test(bud_posed, key, n_boot_gap, args.seed)
        if pop == 'unbiased':
            # Sensitivities on |T|
            res['T_sens'] = {
                'city_fe': logistic_test(open_posed, 'T', n_boot_log, args.seed + 3, city_fe=True),
                'image_dip': logistic_test(open_posed, 'T', n_boot_log, args.seed + 4,
                                           dip_key='img_dip'),
                'first_order_T1': bucket_test(open_posed, 'T1', n_boot_gap, args.seed),
                'within_pano': logistic_test(open_posed, 'T', n_boot_log, args.seed + 5,
                                             within=True),
                # POST HOC (added after the pre-registered results were read; amendment in
                # the report): bearing confound of the within-pano term, and a within-split gap.
                'posthoc_within_pano_bearing': logistic_test(
                    open_posed, 'T', n_boot_log, args.seed + 6, within=True, bearing_term=True),
                'posthoc_stratified_gap_T': stratified_gap(
                    open_posed, 'T', res['T']['bucket']['edges'], n_boot_gap, args.seed + 7),
                'posthoc_stratified_gap_tilt': stratified_gap(
                    open_posed, 'tilt', res['tilt']['bucket']['edges'], n_boot_gap, args.seed + 8),
            }
        results[pop] = res

    # Verdict
    r = results['unbiased']

    def verdict_for(key):
        b, lg, sh = r[key]['bucket'], r[key]['logistic'], r[key]['shuffle']
        coef = lg[1]
        c1 = b['gap'] >= GAP_BAR and b['ci'][0] > 0
        c2 = coef['ci_hi'] < 0
        c3 = sh['n_reach_bar'] < SHUFFLE_FAIL_DRAWS
        return dict(gap=b['gap'], gap_ci=b['ci'], coef=coef, shuffle_reach=sh['n_reach_bar'],
                    perm_p=sh['perm_p'], c1_bucket_gap=c1, c2_logistic=c2, c3_shuffle_clean=c3,
                    passes=bool(c1 and c2 and c3))
    v_T, v_tilt = verdict_for('T'), verdict_for('tilt')
    if v_T['passes']:
        verdict = 'ESTABLISHED'
    elif v_tilt['passes']:
        verdict = 'SECONDARY ONLY (|tilt| passes, |T| does not): not established'
    else:
        verdict = 'NOT ESTABLISHED'
    summary['verdict'] = verdict
    summary['verdict_T'] = v_T
    summary['verdict_tilt'] = v_tilt
    summary['results'] = results

    # Descriptive tables (unbiased)
    open_u = [u for u in units_ub if u['split'] in OPEN_SPLITS]
    open_posed = [u for u in open_u if u['posed']]
    eT = r['T']['bucket']['edges']
    et = r['tilt']['bucket']['edges']
    qlab = lambda v, e: f'Q{int(quartile_of(np.array([abs(v)]), e)[0]) + 1}'
    desc = {}
    desc['absT_by_split'] = recall_rows(open_posed, lambda u: (u['split'], qlab(u['T'], eT)),
                                        ['split', 'absT_q'])
    desc['absT_by_rig'] = recall_rows(open_posed, lambda u: (u['rig'], qlab(u['T'], eT)),
                                      ['rig', 'absT_q'])
    desc['tilt_by_rig'] = recall_rows(open_posed, lambda u: (u['rig'], qlab(u['tilt'], et)),
                                      ['rig', 'tilt_q'])
    desc['signedT_pooled'] = recall_rows(open_posed, lambda u: signed_bucket(u['T']), ['signed_T'])
    desc['signedT_by_rig'] = recall_rows(open_posed, lambda u: (u['rig'], signed_bucket(u['T'])),
                                         ['rig', 'signed_T'])
    desc['inplane_pooled'] = recall_rows(open_posed, lambda u: signed_bucket(u['inplane']),
                                         ['inplane_rot'])
    desc['dip_by_rig'] = recall_rows(
        open_u, lambda u: (u['rig'], dip_bucket(u['grav_dip'] if u['posed'] else u['img_dip'])),
        ['rig', 'dip'])
    desc['rig_city'] = recall_rows(open_u + [u for u in units_ub if u['split'] in HELD_OUT + GSV_SPLITS],
                                   lambda u: (u['split'], u['rig']), ['split', 'rig'])
    desc['posthoc_absT_by_rig_split'] = recall_rows(
        open_posed, lambda u: (u['rig'], u['split'], qlab(u['T'], eT)), ['rig', 'split', 'absT_q'])
    desc['posthoc_tilt_by_rig_split'] = recall_rows(
        open_posed, lambda u: (u['rig'], u['split'], qlab(u['tilt'], et)), ['rig', 'split', 'tilt_q'])
    desc['year_by_rig'] = recall_rows(open_u, lambda u: (u['rig'], u['year']), ['rig', 'year'])
    for name, rows in desc.items():
        write_csv(args.out / f'{name}.csv', rows)

    # Rig recall standardized to the pooled open dip distribution
    dip_of = lambda u: dip_bucket(u['grav_dip'] if u['posed'] else u['img_dip'])
    pooled = Counter(dip_of(u) for u in open_u)
    tot = sum(pooled.values())
    std_rows = []
    for rig in sorted({u['rig'] for u in open_u}):
        us = [u for u in open_u if u['rig'] == rig]
        cell = defaultdict(lambda: [0, 0])
        for u in us:
            c = cell[dip_of(u)]
            c[0] += 1
            c[1] += u['detected']
        covered = [b for b in DIP_LABELS if cell[b][0] >= 5]
        wsum = sum(pooled[b] for b in covered)
        std = sum(pooled[b] / wsum * cell[b][1] / cell[b][0] for b in covered) if wsum else None
        std_rows.append(dict(rig=rig, n_gt=len(us), crude_recall=round(np.mean([u['detected'] for u in us]), 4),
                             dip_standardized_recall=round(std, 4) if std is not None else None,
                             dip_buckets_used=';'.join(covered),
                             share_near_ge10=round(sum(1 for u in us if dip_of(u) in ('10-20', '>=20')) / len(us), 3)))
    write_csv(args.out / 'rig_dip_standardized.csv', std_rows)

    # Covariate distributions (open, posed)
    dist = []
    for s in OPEN_SPLITS + HELD_OUT + GSV_SPLITS:
        us = [u for u in units_ub if u['split'] == s and u['posed']]
        if not us:
            continue
        a = np.array([abs(u['T']) for u in us])
        t = np.array([u['tilt'] for u in us])
        dist.append(dict(split=s, posed_gt=len(us), absT_median=round(float(np.median(a)), 2),
                         absT_p90=round(float(np.quantile(a, 0.9)), 2),
                         tilt_median=round(float(np.median(t)), 2),
                         tilt_p90=round(float(np.quantile(t, 0.9)), 2)))
    write_csv(args.out / 'covariate_distribution.csv', dist)

    # Logistic tables
    lrows = []
    for pop in ('unbiased', 'all_panos'):
        for key in ('T', 'tilt'):
            for row in results[pop][key]['logistic']:
                lrows.append(dict(population=pop, model=f'{key}+rig+gravdip', **row))
    for name in ('city_fe', 'image_dip', 'within_pano', 'posthoc_within_pano_bearing'):
        for row in r['T_sens'][name]:
            lrows.append(dict(population='unbiased', model=f'T sens: {name}', **row))
    write_csv(args.out / 'logistic.csv', lrows)

    qrows = []
    for pop in ('unbiased', 'all_panos'):
        for key in ('T', 'tilt'):
            b = results[pop][key]['bucket']
            for row in b['per_quartile']:
                qrows.append(dict(population=pop, pool='open', covariate=key, **row))
            if pop == 'unbiased':
                for pool in ('gsv_bucket', 'budapest_bucket'):
                    for row in results[pop][key][pool]['per_quartile']:
                        qrows.append(dict(population=pop, pool=pool.replace('_bucket', ''),
                                          covariate=key, **row))
    for row in r['T_sens']['first_order_T1']['per_quartile']:
        qrows.append(dict(population='unbiased', pool='open', covariate='T1_first_order', **row))
    write_csv(args.out / 'quartiles.csv', qrows)

    with open(args.out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=float)

    # Console
    print(f"geo check max |diff| = {summary['geo_check_max_abs_deg']:.2e} deg")
    for s, v in hc.items():
        print(f'horizon {s:20s} image {v["image"]:4d}  gravity {v["gravity"]:4d}  flipped {v["flipped"]:4d}')
    for row in cov:
        print(row)
    for key, v in (('|T|', v_T), ('|tilt|', v_tilt)):
        print(f"{key}: gap {v['gap']:+.3f} CI [{v['gap_ci'][0]:+.3f}, {v['gap_ci'][1]:+.3f}]  "
              f"coef {v['coef']['coef']:+.3f} [{v['coef']['ci_lo']:+.3f}, {v['coef']['ci_hi']:+.3f}]  "
              f"shuffle reach {v['shuffle_reach']}/{n_shuffle} perm p {v['perm_p']:.3f}  "
              f"-> c1 {v['c1_bucket_gap']} c2 {v['c2_logistic']} c3 {v['c3_shuffle_clean']}")
    print('VERDICT:', verdict)


if __name__ == '__main__':
    main()
