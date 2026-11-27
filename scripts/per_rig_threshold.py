"""Does a per-rig confidence threshold beat one global threshold on open imagery?
(overnight `per_rig_threshold`)

CPU only, no network. Leave-one-city-out over eight judged open-imagery RampNet bundles:
for each held-out city, choose thresholds on the other seven by maximizing
U = TP - lambda * FP (ties -> lowest threshold), then score the held-out city. Arms:

- A0 production: global 0.30 (detectors.OPERATIONAL_CONFIDENCE);
- A1 global-procedure: one threshold chosen by the same objective (the fair control);
- A2 per-rig: one threshold per eligible rig class (camera make/model), plus `other`;
- A3 shuffled-rig control: A2 fit after permuting rig labels within each training city.

The pre-registered rule, units and semantics are in docs/overnight/per_rig_threshold.md
section 0. In short: predictions are RampNet's own low-floor sweep cache
(../RampNet/analysis_out/op_cache/<split>.json; Lyon: the production run's stored peaks,
../RampNet/analysis_out/lyon_review_159/run_peaks.json), cut to production's 0.10 storage
floor, its exclude_border rule and the 49-degree rig mask; GT and matching mirror
rampnet.detection_eval.build_ground_truth / score_pano (re-implemented here, checked against
each cache's embedded GT). A sub-0.55 peak is a TP only if it matches a missed-ramp click.

Usage:
    python scripts/per_rig_threshold.py \
        --rampnet-root ../RampNet \
        --out docs/overnight/data/per_rig_threshold
    python scripts/per_rig_threshold.py ... --quick   # 200 bootstrap draws, 20 shuffles (smoke)

Deterministic: every random draw comes from numpy Generators seeded from --seed.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Root first, scripts/ after it: scripts/position_check.py is a shim that must not
# shadow the root position_check module (code review of PR 197).
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
sys.path.remove(str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT))

from detectors import (BENCHMARK_CONFIDENCE, DETECTION_STORAGE_FLOOR,  # noqa: E402
                       OPERATIONAL_CONFIDENCE, on_camera_rig)
from recall_by_tilt import rig_class  # noqa: E402

CITIES = ['richmond', 'clovis', 'morgantown', 'annapolis', 'laurens_mapillary', 'juneau',
          'bayonne', 'lyon']
PANORAMAX = {'bayonne', 'lyon'}
LYON_RESULTS_SHA256 = '60e688b28b62f85e71f51c603dc1b6f4ff020348220c6cc7955891dcfef061ef'

# RampNet's matching geometry (rampnet/detection_eval.py).
SCALE_X, SCALE_Y = 1024, 512
RADIUS_NORMALIZED = 0.022
RADIUS_SQ = (RADIUS_NORMALIZED * SCALE_X) ** 2

# Production peak finder: skimage exclude_border = min_distance = 10 on the 1024 x 512 heatmap.
HEAT_W, HEAT_H, BORDER_PX = 1024, 512, 10

GRID = np.round(np.arange(0.10, 0.9001, 0.01), 2)
LAMBDA = 1.0
PRECISION_FLOOR = 0.85
GUARD_SLACK = 0.03
MIN_RIG_GT = 100
MIN_CITIES = 2
MIN_CITY_GT = 20
OTHER = 'other'
SHUFFLE_FAIL_DRAWS = 10
PARITY_MIN = 0.95


# --------------------------------------------------------------------------------------
# Pure logic (tested)

def fold(dx, period):
    """Shortest signed cyclic difference (rampnet.geometry.fold)."""
    dx = dx % period
    return dx - period if dx > period / 2 else dx


def dist_sq(ax, ay, bx, by):
    dx = fold((ax - bx) * SCALE_X, SCALE_X)
    dy = (ay - by) * SCALE_Y
    return dx * dx + dy * dy


def build_ground_truth(detections, verdicts, missed):
    """(gt_points, ignore_points), rampnet.detection_eval.build_ground_truth semantics."""
    if len(detections) != len(verdicts):
        raise ValueError('detections and verdicts misaligned')
    gt, ign = [], []
    for d, v in zip(detections, verdicts):
        if v is True or v == 'true':
            gt.append((float(d['x_normalized']), float(d['y_normalized'])))
        elif v == 'unsure':
            ign.append((float(d['x_normalized']), float(d['y_normalized'])))
    for m in missed or []:
        pt = (float(m['x']), float(m['y']))
        (ign if m.get('unsure') else gt).append(pt)
    return gt, ign


def classify(preds, gt, ign):
    """Per-prediction outcome ('tp' / 'fp' / 'ignore') in descending-confidence order.

    rampnet score_pano semantics: greedy, each prediction claims the nearest UNCLAIMED GT
    strictly inside the radius; an unmatched one inside the radius of an ignore point is
    ignored, otherwise it is a FP (a redundant second hit included). Because the matching is
    greedy in confidence order, the outcomes of the predictions kept at any threshold are a
    prefix of this list, so one pass serves every threshold.

    Returns [(conf, outcome)] sorted by descending confidence.
    """
    order = sorted(preds, key=lambda p: p[2], reverse=True)
    claimed = [False] * len(gt)
    out = []
    for x, y, c in order:
        best, best_d = -1, RADIUS_SQ
        for k, (gx, gy) in enumerate(gt):
            d = dist_sq(x, y, gx, gy)
            if d < RADIUS_SQ and not claimed[k] and d < best_d:
                best, best_d = k, d
        if best >= 0:
            claimed[best] = True
            out.append((c, 'tp'))
        elif any(dist_sq(x, y, ix, iy) < RADIUS_SQ for ix, iy in ign):
            out.append((c, 'ignore'))
        else:
            out.append((c, 'fp'))
    return out


def counts_on_grid(outcomes, grid=GRID):
    """TP(t), FP(t) arrays over the threshold grid (kept iff conf >= t)."""
    tp = np.zeros(len(grid), dtype=np.int64)
    fp = np.zeros(len(grid), dtype=np.int64)
    for c, o in outcomes:
        kept = c >= grid - 1e-12
        if o == 'tp':
            tp += kept
        elif o == 'fp':
            fp += kept
    return tp, fp


def in_border(x, y):
    """True when production's exclude_border would have dropped this peak."""
    col, row = int(round(x * HEAT_W)), int(round(y * HEAT_H))
    return (col < BORDER_PX or col > HEAT_W - 1 - BORDER_PX
            or row < BORDER_PX or row > HEAT_H - 1 - BORDER_PX)


def best_threshold_index(tp, fp, lam):
    """argmax of tp - lam * fp; ties -> lowest threshold (the first index)."""
    u = np.asarray(tp, dtype=float) - lam * np.asarray(fp, dtype=float)
    return int(np.argmax(u))


def eligible_rigs(gt_by_rig_city, min_rig_gt=MIN_RIG_GT, min_cities=MIN_CITIES,
                  min_city_gt=MIN_CITY_GT):
    """Rigs that get their own threshold: >= min_rig_gt GT ramps in training, and >=
    min_cities training cities each holding >= min_city_gt of them.

    gt_by_rig_city: {rig: {city: gt_count}}."""
    out = set()
    for rig, by_city in gt_by_rig_city.items():
        if rig == OTHER:
            continue
        if (sum(by_city.values()) >= min_rig_gt
                and sum(v >= min_city_gt for v in by_city.values()) >= min_cities):
            out.add(rig)
    return out


def fit_per_rig(tp, fp, ngt, rig, city, weights, lam, global_idx, rig_names,
                min_rig_gt=MIN_RIG_GT, min_cities=MIN_CITIES, min_city_gt=MIN_CITY_GT):
    """Per-rig threshold indices on one training set.

    tp, fp: (n_panos, n_grid); ngt: (n_panos,); rig: (n_panos,) int codes into rig_names;
    city: (n_panos,) city labels; weights: (n_panos,) resample counts (1 = as observed).
    Returns ({rig_name: grid index}, other_index, eligible set). A rig not in the map takes
    other_index."""
    gbrc = defaultdict(lambda: defaultdict(float))
    for r in np.unique(rig):
        m = rig == r
        for c in np.unique(city[m]):
            mc = m & (city == c)
            gbrc[rig_names[r]][c] = float((ngt[mc] * weights[mc]).sum())
    elig = eligible_rigs(gbrc, min_rig_gt, min_cities, min_city_gt)
    out = {}
    other_mask = np.ones(len(rig), dtype=bool)
    for r in np.unique(rig):
        name = rig_names[r]
        if name in elig:
            m = rig == r
            other_mask &= ~m
            w = weights[m][:, None]
            out[name] = best_threshold_index((tp[m] * w).sum(0), (fp[m] * w).sum(0), lam)
    if (ngt[other_mask] * weights[other_mask]).sum() >= min_rig_gt:
        w = weights[other_mask][:, None]
        other_idx = best_threshold_index((tp[other_mask] * w).sum(0),
                                         (fp[other_mask] * w).sum(0), lam)
    else:
        other_idx = global_idx
    return out, other_idx, elig


def shuffle_within(labels, groups, rng):
    """Permute labels among the members of each group (in place on a copy)."""
    out = labels.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = labels[rng.permutation(idx)]
    return out


def score(tp, fp, idx_per_pano, weights):
    """(TP, FP) summed over panos, each at its own threshold index."""
    rows = np.arange(len(idx_per_pano))
    return (float((tp[rows, idx_per_pano] * weights).sum()),
            float((fp[rows, idx_per_pano] * weights).sum()))


# --------------------------------------------------------------------------------------
# Loading

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_city(rampnet_root, split):
    """Pano rows for one split + survivorship + parity numbers."""
    bdir = rampnet_root / 'benchmark' / split
    recs = {}
    with open(bdir / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r['pano']['panorama_id']] = r
    with open(bdir / 'verdicts.json', encoding='utf-8') as f:
        verdicts = json.load(f)['panos']
    cache_gt = {}
    if split == 'lyon':
        rp = json.load(open(rampnet_root / 'analysis_out' / 'lyon_review_159' / 'run_peaks.json',
                            encoding='utf-8'))
        if rp['results_jsonl_sha256'] != LYON_RESULTS_SHA256:
            raise SystemExit('lyon run_peaks.json is pinned to another results.jsonl')
        preds_raw = {pid: [(d['x_normalized'], d['y_normalized'], d['confidence']) for d in v]
                     for pid, v in rp['panos'].items()}
        source = 'run_peaks'
    else:
        cache = json.load(open(rampnet_root / 'analysis_out' / 'op_cache' / f'{split}.json',
                               encoding='utf-8'))
        if cache['meta'].get('tta') or abs(cache['meta']['radius_normalized'] - RADIUS_NORMALIZED) > 1e-12:
            raise SystemExit(f'{split}: unexpected op_cache meta {cache["meta"]}')
        preds_raw = {p['pano']: [tuple(t) for t in p['preds']] for p in cache['panos']}
        cache_gt = {p['pano']: p['gt'] for p in cache['panos']}
        source = 'op_cache'

    surv = Counter()
    rows = []
    parity_n = parity_hit = 0
    gt_mismatch = 0
    for pid in sorted(verdicts):
        e = verdicts[pid]
        surv['bundle_panos'] += 1
        rec = recs.get(pid)
        if rec is None or len(rec['detections']) != len(e['dets']):
            surv['drop_misaligned'] += 1
            continue
        if any(d is None for d in e['dets']):
            surv['drop_none_verdict'] += 1
            continue
        if not (e.get('no_missed') or e.get('missed')):
            surv['drop_unconfirmed_fn'] += 1
            continue
        if pid not in preds_raw:
            surv['drop_no_predictions'] += 1
            continue
        gt, ign = build_ground_truth(rec['detections'], e['dets'], e.get('missed') or [])
        if pid in cache_gt:
            cg = cache_gt[pid]
            same = (sorted(map(tuple, cg['gt_points'])) == sorted(gt)
                    and sorted(map(tuple, cg['ignore_points'])) == sorted(ign))
            gt_mismatch += not same
        preds = []
        for x, y, c in preds_raw[pid]:
            surv['peaks_cache'] += 1
            if c < DETECTION_STORAGE_FLOOR:
                surv['peaks_below_floor'] += 1
                continue
            if in_border(x, y):
                surv['peaks_border'] += 1
                continue
            if on_camera_rig(y):
                surv['peaks_rig_mask'] += 1
                continue
            preds.append((float(x), float(y), float(c)))
        surv['peaks_kept'] += len(preds)
        # Parity: each record detection (>= 0.55) has a kept peak >= 0.55 within 0.5 R.
        for d in rec['detections']:
            if on_camera_rig(d['y_normalized']):
                continue
            parity_n += 1
            parity_hit += any(c >= BENCHMARK_CONFIDENCE and
                              dist_sq(x, y, d['x_normalized'], d['y_normalized']) <= 0.25 * RADIUS_SQ
                              for x, y, c in preds)
        outcomes = classify(preds, gt, ign)
        tp, fp = counts_on_grid(outcomes)
        pano = rec['pano']
        rows.append(dict(city=split, pano_id=pid, rig=rig_class(split, pano),
                         source='panoramax' if split in PANORAMAX else 'mapillary',
                         group=e.get('group'), n_gt=len(gt), tp=tp, fp=fp))
        surv['panos_used'] += 1
        surv['gt_ramps'] += len(gt)
    parity = dict(split=split, pred_source=source, record_dets=parity_n, reproduced=parity_hit,
                  frac=parity_hit / parity_n if parity_n else 1.0,
                  passes=(parity_hit / parity_n if parity_n else 1.0) >= PARITY_MIN,
                  gt_mismatch_vs_cache=gt_mismatch)
    return rows, dict(surv, split=split), parity


# --------------------------------------------------------------------------------------
# The leave-one-city-out procedure

class Data:
    def __init__(self, rows, rig_key='rig'):
        self.rows = rows
        self.tp = np.stack([r['tp'] for r in rows]).astype(float)
        self.fp = np.stack([r['fp'] for r in rows]).astype(float)
        self.ngt = np.array([r['n_gt'] for r in rows], dtype=float)
        self.city = np.array([r['city'] for r in rows])
        keys = [rig_key_of(r, rig_key) for r in rows]
        self.rig_names = sorted(set(keys))
        code = {k: i for i, k in enumerate(self.rig_names)}
        self.rig = np.array([code[k] for k in keys])
        self.cities = [c for c in CITIES if c in set(self.city)]


def rig_key_of(row, rig_key):
    if rig_key == 'rig':
        return row['rig']
    if rig_key == 'rig_source':
        return f"{row['rig']}@{row['source']}"
    raise ValueError(rig_key)


IDX_030 = int(np.flatnonzero(np.isclose(GRID, OPERATIONAL_CONFIDENCE))[0])


def loco(data, weights, lam, rig_labels=None, elig_kw=None, want_detail=False):
    """Run the leave-one-city-out procedure once.

    weights: (n_panos,) resample counts. rig_labels: training-side rig codes (the shuffle
    control passes permuted ones; held-out panos always use data.rig). Returns per-city
    {arm: (TP, FP, GT)} and, if want_detail, the chosen thresholds."""
    elig_kw = elig_kw or {}
    train_rig = data.rig if rig_labels is None else rig_labels
    res, detail = {}, {}
    for held in data.cities:
        te = data.city == held
        tr = ~te
        w_tr = weights[tr][:, None]
        g_idx = best_threshold_index((data.tp[tr] * w_tr).sum(0), (data.fp[tr] * w_tr).sum(0), lam)
        rig_idx, other_idx, elig = fit_per_rig(
            data.tp[tr], data.fp[tr], data.ngt[tr], train_rig[tr], data.city[tr], weights[tr],
            lam, g_idx, data.rig_names, **elig_kw)
        w_te = weights[te]
        tp_te, fp_te = data.tp[te], data.fp[te]
        gt_te = float((data.ngt[te] * w_te).sum())
        n = int(te.sum())
        per_pano = np.array([rig_idx.get(data.rig_names[r], other_idx) for r in data.rig[te]])
        arms = {
            'A0': score(tp_te, fp_te, np.full(n, IDX_030), w_te),
            'A1': score(tp_te, fp_te, np.full(n, g_idx), w_te),
            'A2': score(tp_te, fp_te, per_pano, w_te),
        }
        res[held] = {a: (t, f, gt_te) for a, (t, f) in arms.items()}
        if want_detail:
            o_idx = best_threshold_index((tp_te * w_te[:, None]).sum(0),
                                         (fp_te * w_te[:, None]).sum(0), lam)
            res[held]['oracle'] = score(tp_te, fp_te, np.full(n, o_idx), w_te) + (gt_te,)
            detail[held] = dict(global_t=float(GRID[g_idx]), other_t=float(GRID[other_idx]),
                                oracle_t=float(GRID[o_idx]),
                                rig_t={k: float(GRID[v]) for k, v in sorted(rig_idx.items())},
                                eligible=sorted(elig),
                                held_rigs=dict(Counter(data.rig_names[r] for r in data.rig[te])))
    return res, detail


def utility(res, arm, lam, cities=None, equal_weight=False):
    """Pooled utility per GT ramp (x100), or the mean over cities of each city's."""
    cities = cities or list(res)
    if equal_weight:
        vals = [(res[c][arm][0] - lam * res[c][arm][1]) / res[c][arm][2] for c in cities]
        return 100 * float(np.mean(vals))
    num = sum(res[c][arm][0] - lam * res[c][arm][1] for c in cities)
    den = sum(res[c][arm][2] for c in cities)
    return 100 * num / den


def d_util(res, a, b, lam, equal_weight=False):
    return utility(res, a, lam, equal_weight=equal_weight) - utility(res, b, lam, equal_weight=equal_weight)


def prec(t, f):
    return t / (t + f) if t + f else float('nan')


def guard(res):
    """Pre-registered precision guard: list of (city, P_A1, P_A2, ok)."""
    out = []
    for c, arms in res.items():
        p1, p2 = prec(*arms['A1'][:2]), prec(*arms['A2'][:2])
        ok = (p2 >= PRECISION_FLOOR) if p1 >= PRECISION_FLOOR else (p2 >= p1 - GUARD_SLACK)
        out.append((c, p1, p2, bool(ok)))
    return out


def bootstrap(data, lam, n_boot, seed, elig_kw=None, equal_weight=False):
    """Cluster bootstrap by pano within city; the whole procedure re-runs per draw."""
    rng = np.random.default_rng(seed)
    out = {'A2-A1': [], 'A1-A0': [], 'A2-A0': []}
    idx_by_city = {c: np.flatnonzero(data.city == c) for c in data.cities}
    for _ in range(n_boot):
        w = np.zeros(len(data.rows))
        for c, idx in idx_by_city.items():
            w[idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
        res, _ = loco(data, w, lam, elig_kw=elig_kw)
        out['A2-A1'].append(d_util(res, 'A2', 'A1', lam, equal_weight))
        out['A1-A0'].append(d_util(res, 'A1', 'A0', lam, equal_weight))
        out['A2-A0'].append(d_util(res, 'A2', 'A0', lam, equal_weight))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def shuffle_control(data, lam, n_draws, seed, elig_kw=None):
    rng = np.random.default_rng(seed)
    ones = np.ones(len(data.rows))
    vals = []
    for _ in range(n_draws):
        lab = shuffle_within(data.rig, data.city, rng)
        res, _ = loco(data, ones, lam, rig_labels=lab, elig_kw=elig_kw)
        vals.append(d_util(res, 'A2', 'A1', lam))
    return np.array(vals)


# --------------------------------------------------------------------------------------
# Output

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})


def per_city_rows(res, lam, label):
    rows = []
    for c, arms in res.items():
        for a, (t, f, g) in arms.items():
            rows.append(dict(analysis=label, city=c, arm=a, tp=round(t), fp=round(f), gt=round(g),
                             precision=prec(t, f), recall=t / g if g else float('nan'),
                             u_per_100gt=100 * (t - lam * f) / g if g else float('nan')))
    return rows


def run_analysis(data, lam, label, n_boot, seed, elig_kw=None, equal_weight=False,
                 shuffles=0):
    ones = np.ones(len(data.rows))
    res, detail = loco(data, ones, lam, elig_kw=elig_kw, want_detail=True)
    pt = {k: d_util(res, a, b, lam, equal_weight)
          for k, (a, b) in {'A2-A1': ('A2', 'A1'), 'A1-A0': ('A1', 'A0'),
                            'A2-A0': ('A2', 'A0')}.items()}
    ci = bootstrap(data, lam, n_boot, seed, elig_kw, equal_weight)
    summ = dict(analysis=label, lam=lam, equal_weight=equal_weight,
                u_A0=utility(res, 'A0', lam, equal_weight=equal_weight),
                u_A1=utility(res, 'A1', lam, equal_weight=equal_weight),
                u_A2=utility(res, 'A2', lam, equal_weight=equal_weight),
                u_oracle=utility(res, 'oracle', lam, equal_weight=equal_weight))
    for k in pt:
        summ[f'd_{k}'] = pt[k]
        summ[f'd_{k}_lo'], summ[f'd_{k}_hi'] = ci[k]
    g = guard(res)
    summ['guard_ok'] = all(ok for *_, ok in g)
    if shuffles:
        sh = shuffle_control(data, lam, shuffles, seed + 1, elig_kw)
        summ['shuffle_median'] = float(np.median(sh))
        summ['shuffle_p95'] = float(np.percentile(sh, 95))
        summ['shuffle_reaching'] = int((sh >= pt['A2-A1'] - 1e-12).sum())
        summ['shuffle_draws'] = shuffles
    return res, detail, summ, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rampnet-root', type=Path, default=REPO_ROOT.parent / 'RampNet')
    ap.add_argument('--out', type=Path, default=REPO_ROOT / 'docs/overnight/data/per_rig_threshold')
    ap.add_argument('--seed', type=int, default=20261127)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--n-boot-sens', type=int, default=500)
    ap.add_argument('--shuffles', type=int, default=200)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.n_boot, args.n_boot_sens, args.shuffles = 200, 100, 20
    args.out.mkdir(parents=True, exist_ok=True)

    rows, survs, parities = [], [], []
    for split in CITIES:
        r, s, p = load_city(args.rampnet_root, split)
        rows += r
        survs.append(s)
        parities.append(p)
        print(f'{split}: {s["panos_used"]} panos, {s["gt_ramps"]} GT; parity {p["reproduced"]}/'
              f'{p["record_dets"]} ({p["frac"]:.3f}); GT mismatches vs cache {p["gt_mismatch_vs_cache"]}',
              flush=True)
    keys = sorted({k for s in survs for k in s if k != 'split'})
    write_csv(args.out / 'survivorship.csv', [dict(split=s['split'], **{k: s.get(k, 0) for k in keys})
                                              for s in survs])
    write_csv(args.out / 'parity.csv', parities)

    # Rig sizes per city.
    rs = defaultdict(lambda: [0, 0])
    for r in rows:
        rs[(r['city'], r['rig'])][0] += 1
        rs[(r['city'], r['rig'])][1] += r['n_gt']
    write_csv(args.out / 'rig_sizes.csv', [dict(city=c, rig=g, panos=v[0], gt=v[1])
                                           for (c, g), v in sorted(rs.items())])

    # Per-rig, per-city descriptive curve (full data, not LOCO): P/R at a few thresholds.
    data = Data(rows)
    curve = []
    for rig in data.rig_names:
        for c in data.cities + ['ALL']:
            m = (data.rig == data.rig_names.index(rig)) & ((data.city == c) if c != 'ALL' else True)
            if not m.any():
                continue
            tp, fp, g = data.tp[m].sum(0), data.fp[m].sum(0), data.ngt[m].sum()
            b = best_threshold_index(tp, fp, LAMBDA)
            for kind, i in [(f't{t:.2f}', int(np.flatnonzero(np.isclose(GRID, t))[0]))
                            for t in (0.10, 0.20, 0.30, 0.40, 0.55)] + [('best_lam1', b)]:
                curve.append(dict(rig=rig, city=c, kind=kind, threshold=float(GRID[i]),
                                  panos=int(m.sum()), gt=int(g), tp=int(tp[i]), fp=int(fp[i]),
                                  precision=prec(tp[i], fp[i]),
                                  recall=tp[i] / g if g else float('nan')))
    write_csv(args.out / 'rig_curves.csv', curve)

    summaries, all_city_rows, thresholds, guards = [], [], [], []

    def record(label, res, detail, summ, g, lam):
        summaries.append(summ)
        all_city_rows.extend(per_city_rows(res, lam, label))
        for c, d in detail.items():
            for rig, t in d['rig_t'].items():
                thresholds.append(dict(analysis=label, held_out=c, key=rig, threshold=t))
            thresholds.append(dict(analysis=label, held_out=c, key='other', threshold=d['other_t']))
            thresholds.append(dict(analysis=label, held_out=c, key='GLOBAL(A1)', threshold=d['global_t']))
            thresholds.append(dict(analysis=label, held_out=c, key='ORACLE(held-out)', threshold=d['oracle_t']))
        for c, p1, p2, ok in g:
            guards.append(dict(analysis=label, city=c, precision_A1=p1, precision_A2=p2, ok=ok))
        print(f'[{label}] dU(A2-A1) {summ["d_A2-A1"]:+.2f} ({summ["d_A2-A1_lo"]:+.2f}, '
              f'{summ["d_A2-A1_hi"]:+.2f}); dU(A1-A0) {summ["d_A1-A0"]:+.2f}; guard {summ["guard_ok"]}'
              + (f'; shuffles reaching {summ["shuffle_reaching"]}/{summ["shuffle_draws"]}'
                 if 'shuffle_reaching' in summ else ''), flush=True)

    # PRIMARY
    res, detail, summ, g = run_analysis(data, LAMBDA, 'primary', args.n_boot, args.seed,
                                        shuffles=args.shuffles)
    record('primary', res, detail, summ, g, LAMBDA)
    primary_detail = detail
    sh_pass = summ['shuffle_reaching'] < SHUFFLE_FAIL_DRAWS
    ci_pass = summ['d_A2-A1_lo'] > 0
    verdict = dict(ci_excludes_0_and_positive=ci_pass, beats_shuffle=sh_pass,
                   precision_guard=summ['guard_ok'],
                   passes=bool(ci_pass and sh_pass and summ['guard_ok']))

    # Sensitivities (non-gating)
    for lam in (0.5, 2.0):
        r_, d_, s_, g_ = run_analysis(data, lam, f'lambda_{lam:g}', args.n_boot_sens, args.seed,
                                      shuffles=args.shuffles // 4)
        record(f'lambda_{lam:g}', r_, d_, s_, g_, lam)
    r_, d_, s_, g_ = run_analysis(data, LAMBDA, 'one_city_rigs', args.n_boot_sens, args.seed,
                                  elig_kw=dict(min_cities=1), shuffles=args.shuffles // 4)
    record('one_city_rigs', r_, d_, s_, g_, LAMBDA)
    data_rs = Data(rows, rig_key='rig_source')
    r_, d_, s_, g_ = run_analysis(data_rs, LAMBDA, 'rig_x_source', args.n_boot_sens, args.seed,
                                  shuffles=args.shuffles // 4)
    record('rig_x_source', r_, d_, s_, g_, LAMBDA)
    r_, d_, s_, g_ = run_analysis(data, LAMBDA, 'equal_city_weight', args.n_boot_sens, args.seed,
                                  equal_weight=True)
    record('equal_city_weight', r_, d_, s_, g_, LAMBDA)
    data_nt = Data([r for r in rows if r['group'] != 'top'])
    r_, d_, s_, g_ = run_analysis(data_nt, LAMBDA, 'exclude_top', args.n_boot_sens, args.seed,
                                  shuffles=args.shuffles // 4)
    record('exclude_top', r_, d_, s_, g_, LAMBDA)

    write_csv(args.out / 'summary.csv', summaries)
    write_csv(args.out / 'per_city.csv', all_city_rows)
    write_csv(args.out / 'thresholds.csv', thresholds)
    write_csv(args.out / 'guard.csv', guards)
    with open(args.out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(dict(verdict=verdict, primary=summaries[0], primary_detail=primary_detail,
                       config=dict(grid=[float(GRID[0]), float(GRID[-1]), 0.01], lam=LAMBDA,
                                   precision_floor=PRECISION_FLOOR, guard_slack=GUARD_SLACK,
                                   min_rig_gt=MIN_RIG_GT, min_cities=MIN_CITIES,
                                   min_city_gt=MIN_CITY_GT, seed=args.seed, n_boot=args.n_boot,
                                   shuffles=args.shuffles,
                                   lyon_results_sha256=LYON_RESULTS_SHA256)),
                  f, indent=1, sort_keys=True)
        f.write('\n')
    print('VERDICT', verdict)


if __name__ == '__main__':
    main()
