#!/usr/bin/env python3
"""GT-free geometric filters for AI Crosswalk false positives (Juneau pilot, RampNet#298).

Can a simple geometric rule, decided without looking at votes, drop the capture-car-roof and
brick-sidewalk false positives from the 3,579 AI Crosswalk labels sent to juneau-test without
losing many real crosswalks? Four arms are tuned on the NON-random votes and confirmed once on
the seeded random sample (79 voted labels), weighted back to all 3,579:

    A  rig-cell mask: per camera model, heatmap cells whose sent labels come from >= S sequences
    B  steep-dip cut: drop labels more than D degrees below the horizon
    C  dip x beside-the-vehicle band: D_lo <= dip <= D_hi and | |rel_az| - 90 | <= H
    D  A* union (the better of B*, C*)

plus a random-removal control of the same size. Pre-registration, results and caveats:
docs/overnight/crosswalk_fp_filters.md. No network, no GPU, stdlib only.

Usage (from the repo root; the data dirs are read-only):
    python scripts/crosswalk_fp_filters.py \
        --validation-dir runs/juneau_crosswalk/validation \
        --results runs/juneau/results.jsonl \
        --out docs/overnight/data/crosswalk_fp_filters

``--results`` is the CurbRamp run over the same 4,999 panos; only its pano blocks are read
(sequence, capture time, position, heading) to get each sequence's camera-to-travel offset.

Outputs (small CSV/JSON): offsets.csv, tuning_grid.csv, arms.csv, survivorship.csv,
remaining_fp.csv, rig_cells.csv, summary.json.

Example (the beside-the-vehicle band on toy values):
    >>> in_band(dip=32.0, rel_az=-95.0, d_lo=25, d_hi=40, h=45)
    True
    >>> in_band(dip=32.0, rel_az=170.0, d_lo=25, d_hi=40, h=45)
    False
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Root must come first: scripts/position_check.py would shadow the root module of that name.
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
from detectors import on_camera_rig  # noqa: E402

HM_W, HM_H = geo.HEATMAP_WIDTH, geo.HEATMAP_HEIGHT   # 1024 x 512
MIN_STEP_M = 2.0          # consecutive frames closer than this give no travel bearing
MIN_PAIRS = 3             # fewer valid pairs in a sequence -> fall back to the rig offset
TUNE_MAX_AGREE_LOSS = 0.05
BOOT_REPS = 10_000
BOOT_SEED = 20261127
CONTROL_DRAWS = 5_000
CONTROL_SEED = 20261127

GRID_A = [(c, s) for c in (8, 16) for s in (3, 4, 5, 6, 8, 10)]
GRID_B = [20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 45.0]
GRID_C = [(lo, hi, h) for lo in (20.0, 25.0, 27.5, 30.0) for hi in (35.0, 40.0, 45.0, 49.0)
          for h in (30.0, 45.0, 60.0, 90.0)]
SCORED_CSV = 'validator_r3_all_labels_scored.csv'
SAMPLE_JSON = 'random_sample_2026-11-17.json'


# --- geometry ---------------------------------------------------------------------------

def wrap180(a):
    """Fold degrees into [-180, 180).

        >>> wrap180(190.0), wrap180(-180.0)
        (-170.0, -180.0)
    """
    return (a + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles):
    """Circular mean of angles in degrees, in [-180, 180); None for an empty list.

        >>> round(circular_mean_deg([170.0, -170.0]), 6)
        -180.0
    """
    if not angles:
        return None
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return wrap180(math.degrees(math.atan2(s, c)))


def initial_bearing_deg(lat1, lng1, lat2, lng2):
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def heatmap_pos(pano_x, pano_y, width, height):
    """(col, row) on the 1024 x 512 heatmap, rounded; col wraps at the seam.

        >>> heatmap_pos(7906.25, 2203.125, 8000, 4000)
        (1012, 282)
    """
    return int(round(pano_x / width * HM_W)) % HM_W, int(round(pano_y / height * HM_H))


def dip_deg(row):
    """Degrees below the image horizon of a heatmap row (negative above it)."""
    return (row / HM_H - 0.5) * 180.0


def image_azimuth(col):
    """Degrees from the image centre column, [-180, 180)."""
    return wrap180((col / HM_W - 0.5) * 360.0)


def in_band(dip, rel_az, d_lo, d_hi, h):
    """Arm C: is a label in the dip band AND within h degrees of straight abeam (either side)?"""
    return d_lo <= dip <= d_hi and abs(abs(wrap180(rel_az)) - 90.0) <= h


def rig_cells(labels, cell, min_seqs):
    """Arm A's mask: the set of (camera_model, cx, cy) whose labels come from >= min_seqs
    distinct sequences of that camera model. Labels are dicts with camera_model, sequence_id,
    col, row.

        >>> L = [dict(camera_model='m', sequence_id=s, col=1012, row=283) for s in 'abc']
        >>> sorted(rig_cells(L, 8, 3)), rig_cells(L, 8, 4)
        ([('m', 126, 35)], set())
    """
    seqs = defaultdict(set)
    for lab in labels:
        seqs[(lab['camera_model'], lab['col'] // cell, lab['row'] // cell)].add(lab['sequence_id'])
    return {k for k, v in seqs.items() if len(v) >= min_seqs}


def within_sequence_recurrence(labels, cell, k):
    """Amendment 1 (exploratory): flag a label whose cell (col // cell, row // cell) holds
    labels from >= k distinct panos of its OWN sequence (a fixture riding along).

        >>> L = [dict(sequence_id='s', pano_id=p, col=11, row=276) for p in 'abc']
        >>> within_sequence_recurrence(L, 8, 3), within_sequence_recurrence(L, 8, 4)
        ([True, True, True], [False, False, False])
    """
    panos = defaultdict(set)
    for lab in labels:
        panos[(lab['sequence_id'], lab['col'] // cell, lab['row'] // cell)].add(lab['pano_id'])
    return [len(panos[(lab['sequence_id'], lab['col'] // cell, lab['row'] // cell)]) >= k
            for lab in labels]


def in_rig_cell(lab, mask, cell):
    return (lab['camera_model'], lab['col'] // cell, lab['row'] // cell) in mask


# --- estimators -------------------------------------------------------------------------

def stratum_weights(units, pop):
    """w_h = N_h / n_h over the units that hold a vote (y is not None)."""
    n = Counter(u['stratum'] for u in units if u['y'] is not None)
    return {h: pop[h] / n[h] for h in n if n[h]}


def weighted_stats(units, removed, pop):
    """(precision of kept, est. TP lost, est. TP total) for the stratified sample.

    ``units``: dicts with stratum and y (1 Agree, 0 Disagree, None missing); ``removed``: a
    parallel list of bools.

        >>> U = [dict(stratum='k', y=1), dict(stratum='k', y=0), dict(stratum='d', y=0)]
        >>> p, lost, tot = weighted_stats(U, [False, True, False], {'k': 10, 'd': 10})
        >>> round(p, 4), lost, tot
        (0.3333, 0.0, 5.0)
    """
    w = stratum_weights(units, pop)
    kept_w = kept_tp = lost = tot = 0.0
    for u, r in zip(units, removed):
        if u['y'] is None:
            continue
        wi = w[u['stratum']]
        tot += wi * u['y']
        if r:
            lost += wi * u['y']
        else:
            kept_w += wi
            kept_tp += wi * u['y']
    return (kept_tp / kept_w if kept_w else float('nan')), lost, tot


def unweighted_stats(units, removed):
    """(kept precision, agree removed, agree total, disagree removed, disagree total)."""
    kept = [u['y'] for u, r in zip(units, removed) if not r and u['y'] is not None]
    ar = sum(1 for u, r in zip(units, removed) if r and u['y'] == 1)
    dr = sum(1 for u, r in zip(units, removed) if r and u['y'] == 0)
    at = sum(1 for u in units if u['y'] == 1)
    dt = sum(1 for u in units if u['y'] == 0)
    return (sum(kept) / len(kept) if kept else float('nan')), ar, at, dr, dt


def cluster_bootstrap(units, removed_by_arm, pop, reps=BOOT_REPS, seed=BOOT_SEED):
    """Per arm, lists of replicate (delta precision vs no filter, TP-lost share), resampling
    sequences with replacement and re-deriving N_h / n_h per replicate."""
    rng = random.Random(seed)
    by_seq = defaultdict(list)
    for i, u in enumerate(units):
        if u['y'] is not None:
            by_seq[u['sequence_id']].append(i)
    seqs = sorted(by_seq)
    out = {a: {'delta': [], 'lost_share': [], 'p': []} for a in removed_by_arm}
    done = 0
    while done < reps:
        idx = [i for _ in seqs for i in by_seq[rng.choice(seqs)]]
        sub = [units[i] for i in idx]
        if len({u['stratum'] for u in sub}) < len(pop):
            continue
        p0, _, _ = weighted_stats(sub, [False] * len(sub), pop)
        for a, rem in removed_by_arm.items():
            p, lost, tot = weighted_stats(sub, [rem[i] for i in idx], pop)
            out[a]['p'].append(p)
            out[a]['delta'].append(p - p0)
            out[a]['lost_share'].append(lost / tot if tot else float('nan'))
        done += 1
    return out


def pct(xs, q):
    """Percentile by the nearest-rank-on-sorted rule (q in [0, 100])."""
    s = sorted(x for x in xs if not math.isnan(x))
    if not s:
        return float('nan')
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def random_control(n_pop, n_remove, sample_idx, units, pop, draws=CONTROL_DRAWS, seed=CONTROL_SEED):
    """Delta precision when n_remove of the n_pop labels are removed uniformly at random."""
    rng = random.Random(seed)
    p0, _, _ = weighted_stats(units, [False] * len(units), pop)
    deltas = []
    for _ in range(draws):
        gone = set(rng.sample(range(n_pop), n_remove))
        p, _, _ = weighted_stats(units, [i in gone for i in sample_idx], pop)
        deltas.append(p - p0)
    return deltas


# --- loading ----------------------------------------------------------------------------

def sequence_offsets(results_path):
    """Camera-heading-minus-travel offset per sequence and per camera model (degrees)."""
    seqs = defaultdict(list)
    with open(results_path) as fh:
        for line in fh:
            p = json.loads(line)['pano']
            seqs[p['sequence_id']].append((p['source_metadata']['captured_at'], p['panorama_id'],
                                           p['lat'], p['lng'], p['camera_heading'],
                                           p['camera_model']))
    per_seq, per_rig, rows = {}, defaultdict(list), []
    for s, ps in seqs.items():
        ps.sort()
        diffs = []
        for a, b in zip(ps, ps[1:]):
            if geo.haversine_m(a[2], a[3], b[2], b[3]) < MIN_STEP_M:
                continue
            diffs.append(wrap180(a[4] - initial_bearing_deg(a[2], a[3], b[2], b[3])))
        per_rig[ps[0][5]].extend(diffs)
        within30 = sum(1 for d in diffs if abs(wrap180(d - circular_mean_deg(diffs))) <= 30) if diffs else 0
        rows.append(dict(sequence_id=s, camera_model=ps[0][5], n_panos=len(ps), n_pairs=len(diffs),
                         offset_deg=round(circular_mean_deg(diffs), 2) if diffs else '',
                         share_within_30=round(within30 / len(diffs), 3) if diffs else ''))
        if len(diffs) >= MIN_PAIRS:
            per_seq[s] = circular_mean_deg(diffs)
    rig = {m: circular_mean_deg(d) for m, d in per_rig.items()}
    return per_seq, rig, rows


def load_labels(validation_dir, per_seq, rig_off):
    rows = list(csv.DictReader(open(Path(validation_dir) / SCORED_CSV)))
    sample = json.load(open(Path(validation_dir) / SAMPLE_JSON))
    stratum = {f'juneau-ak:{i}': 'kept' for i in sample['sample_kept']}
    stratum.update({f'juneau-ak:{i}': 'dropped' for i in sample['sample_dropped']})
    pop = {'kept': sample['kept_pop'], 'dropped': sample['dropped_pop']}
    labels = []
    for r in rows:
        col, row = heatmap_pos(float(r['pano_x']), float(r['pano_y']), float(r['width']),
                               float(r['height']))
        if on_camera_rig(row / HM_H):
            raise SystemExit(f"{r['label_uid']} is below the nadir mask; was the file rig-masked?")
        seq = r['sequence_id']
        off_src = 'sequence' if seq in per_seq else 'rig'
        off = per_seq.get(seq, rig_off[r['camera_model']])
        vote = r['validation_result']
        labels.append(dict(
            label_uid=r['label_uid'], pano_id=r['pano_id'], sequence_id=seq, run=r['run'],
            camera_model=r['camera_model'], col=col, row=row, dip=dip_deg(row),
            az_img=image_azimuth(col), rel_az=wrap180(image_azimuth(col) + off), offset_src=off_src,
            confidence=float(r['confidence']), vote=vote, fp_class=r['fp_class'],
            first_batch=r['first_batch'], stratum=stratum.get(r['label_uid']),
            in_sample=r['label_uid'] in stratum))
    assert len({lab['label_uid'] for lab in labels}) == len(labels), 'label_uid not unique'
    assert sum(lab['in_sample'] for lab in labels) == len(stratum), 'sample label missing'
    return labels, pop


def y_of(vote, unsure_as_disagree=False):
    if vote == 'Agree':
        return 1
    if vote == 'Disagree' or (vote == 'Unsure' and unsure_as_disagree):
        return 0
    return None


# --- arms -------------------------------------------------------------------------------

def arm_removed(labels, arm, params):
    if arm == 'A':
        c, s = params
        mask = rig_cells(labels, c, s)
        return [in_rig_cell(lab, mask, c) for lab in labels]
    if arm == 'B':
        return [lab['dip'] > params for lab in labels]
    if arm == 'C':
        lo, hi, h = params
        return [in_band(lab['dip'], lab['rel_az'], lo, hi, h) for lab in labels]
    raise ValueError(arm)


def tune(labels, tune_idx, arm, grid, tuning_rows):
    units = [dict(y=y_of(labels[i]['vote'])) for i in tune_idx]
    best = None
    for k, params in enumerate(grid):
        rem = arm_removed(labels, arm, params)
        p, ar, at, dr, dt = unweighted_stats(units, [rem[i] for i in tune_idx])
        n_rem = sum(rem)
        eligible = ar <= TUNE_MAX_AGREE_LOSS * at
        tuning_rows.append(dict(arm=arm, params=json.dumps(params), removed_all=n_rem,
                                tune_precision=round(p, 4), tune_agree_removed=ar,
                                tune_agree_total=at, tune_disagree_removed=dr,
                                tune_disagree_total=dt, eligible=eligible))
        key = (eligible, p, -n_rem, -k)
        if best is None or key > best[0]:
            best = (key, params, rem)
    return best[1], best[2], best[0][0]


# --- main -------------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--validation-dir', required=True)
    ap.add_argument('--results', required=True, help='runs/juneau/results.jsonl (pano blocks)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot-reps', type=int, default=BOOT_REPS)
    ap.add_argument('--control-draws', type=int, default=CONTROL_DRAWS)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    per_seq, rig_off, off_rows = sequence_offsets(a.results)
    write_csv(out / 'offsets.csv', sorted(off_rows, key=lambda r: (r['camera_model'], r['sequence_id'])))
    labels, pop = load_labels(a.validation_dir, per_seq, rig_off)
    n = len(labels)

    tune_idx = [i for i, lab in enumerate(labels) if not lab['in_sample'] and y_of(lab['vote']) is not None]
    sample_idx = [i for i, lab in enumerate(labels) if lab['in_sample']]

    tuning_rows = []
    chosen = {}
    for arm, grid in (('A', GRID_A), ('B', GRID_B), ('C', GRID_C)):
        params, rem, ok = tune(labels, tune_idx, arm, grid, tuning_rows)
        chosen[arm] = dict(params=params, removed=rem, tuning_eligible=ok)
    write_csv(out / 'tuning_grid.csv', tuning_rows)

    def tune_p(rem):
        units = [dict(y=y_of(labels[i]['vote'])) for i in tune_idx]
        return unweighted_stats(units, [rem[i] for i in tune_idx])[0]
    xb, xc = chosen['B'], chosen['C']
    kb, kc = (tune_p(xb['removed']), -sum(xb['removed'])), (tune_p(xc['removed']), -sum(xc['removed']))
    x_arm = 'B' if kb >= kc else 'C'
    chosen['D'] = dict(params={'A': chosen['A']['params'], x_arm: chosen[x_arm]['params']},
                       removed=[p or q for p, q in zip(chosen['A']['removed'], chosen[x_arm]['removed'])],
                       tuning_eligible=None)
    chosen['none'] = dict(params=None, removed=[False] * n, tuning_eligible=None)

    # A's mask cells, for the record
    c, s = chosen['A']['params']
    cell_rows = []
    seqs = defaultdict(set)
    cnt = Counter()
    for lab in labels:
        k = (lab['camera_model'], lab['col'] // c, lab['row'] // c)
        seqs[k].add(lab['sequence_id'])
        cnt[k] += 1
    for k in sorted(rig_cells(labels, c, s)):
        cm, cx, cy = k
        cell_rows.append(dict(camera_model=cm, cell_px=c, cx=cx, cy=cy,
                              col_range=f'{cx * c}-{cx * c + c - 1}',
                              dip_mid=round(dip_deg(cy * c + c / 2), 2),
                              az_img_mid=round(image_azimuth(cx * c + c / 2), 1),
                              n_labels=cnt[k], n_sequences=len(seqs[k])))
    write_csv(out / 'rig_cells.csv', cell_rows)

    summary = {'n_labels': n, 'population_strata': pop, 'rig_offsets_deg': rig_off,
               'n_seq_offsets': len(per_seq), 'tuning_n': len(tune_idx), 'x_arm_for_D': x_arm,
               'chosen': {k: v['params'] for k, v in chosen.items()}, 'sensitivities': {}}
    arm_rows, surv_rows = [], []
    arms = ['none', 'A', 'B', 'C', 'D']
    for unsure_mode in (False, True):
        units = []
        for i in sample_idx:
            lab = labels[i]
            units.append(dict(stratum=lab['stratum'], y=y_of(lab['vote'], unsure_mode),
                              sequence_id=lab['sequence_id']))
        w = stratum_weights(units, pop)
        removed_by_arm = {k: [chosen[k]['removed'][i] for i in sample_idx] for k in arms}
        boot = cluster_bootstrap(units, removed_by_arm, pop, reps=a.boot_reps)
        p0, _, tot0 = weighted_stats(units, removed_by_arm['none'], pop)
        tag = 'unsure_as_disagree' if unsure_mode else 'primary'
        n_strat = Counter(u['stratum'] for u in units if u['y'] is not None)
        summary['sensitivities'][tag] = {'weights': w, 'n_voted_by_stratum': dict(n_strat),
                                         'precision_none': p0, 'tp_total_est': tot0}
        # the unfiltered precision's CI: the same resampling scheme, on the precision itself
        none_ps = []
        rng = random.Random(BOOT_SEED)
        by_seq = defaultdict(list)
        for j, u in enumerate(units):
            if u['y'] is not None:
                by_seq[u['sequence_id']].append(j)
        sq = sorted(by_seq)
        while len(none_ps) < a.boot_reps:
            sub = [units[j] for _ in sq for j in by_seq[rng.choice(sq)]]
            if len({u['stratum'] for u in sub}) < len(pop):
                continue
            none_ps.append(weighted_stats(sub, [False] * len(sub), pop)[0])
        summary['sensitivities'][tag]['precision_none_ci'] = [pct(none_ps, 2.5), pct(none_ps, 97.5)]

        tune_units = [dict(y=y_of(labels[i]['vote'], unsure_mode)) for i in tune_idx]
        for k in arms:
            rem_all = chosen[k]['removed']
            n_rem = sum(rem_all)
            p, lost, tot = weighted_stats(units, removed_by_arm[k], pop)
            delta = p - p0
            d_lo, d_hi = pct(boot[k]['delta'], 2.5), pct(boot[k]['delta'], 97.5)
            ls_lo, ls_hi = pct(boot[k]['lost_share'], 2.5), pct(boot[k]['lost_share'], 97.5)
            if k == 'none':
                ctrl_p = float('nan')
            else:
                ctrl = random_control(n, n_rem, sample_idx, units, pop, draws=a.control_draws)
                ctrl_p = (sum(1 for d in ctrl if d >= delta - 1e-12) / len(ctrl))
            lost_share = lost / tot if tot else float('nan')
            crit = (d_lo > 0, ctrl_p < 0.05, lost_share <= 0.05)
            tp, tar, tat, tdr, tdt = unweighted_stats(tune_units, [rem_all[i] for i in tune_idx])
            s_rem = [u for u, r in zip(units, removed_by_arm[k]) if r]
            arm_rows.append(dict(
                mode=tag, arm=k, params=json.dumps(chosen[k]['params']),
                tuning_eligible=chosen[k]['tuning_eligible'],
                removed=n_rem, removed_share=round(n_rem / n, 4),
                precision=round(p, 4), precision_lo=round(pct(boot[k]['p'], 2.5), 4),
                precision_hi=round(pct(boot[k]['p'], 97.5), 4), delta=round(delta, 4), delta_lo=round(d_lo, 4),
                delta_hi=round(d_hi, 4), tp_lost_est=round(lost, 1),
                tp_lost_share=round(lost_share, 4), tp_lost_share_lo=round(ls_lo, 4),
                tp_lost_share_hi=round(ls_hi, 4), control_p=round(ctrl_p, 4),
                sample_removed=len(s_rem), sample_removed_agree=sum(1 for u in s_rem if u['y'] == 1),
                sample_removed_disagree=sum(1 for u in s_rem if u['y'] == 0),
                crit_ci=crit[0], crit_control=crit[1], crit_loss=crit[2],
                recommended=(k != 'none' and all(crit)),
                tune_precision=round(tp, 4), tune_agree_removed=f'{tar}/{tat}',
                tune_disagree_removed=f'{tdr}/{tdt}'))

    # survivorship (primary): what each arm removes, by rig, run, stratum, vote set
    for k in arms[1:]:
        rem = chosen[k]['removed']
        groups = Counter()
        for lab, r in zip(labels, rem):
            vset = ('random_sample' if lab['in_sample'] else
                    'tuning' if y_of(lab['vote']) is not None else
                    'unsure' if lab['vote'] == 'Unsure' else 'unvoted')
            for g in (('camera_model', lab['camera_model']), ('run', lab['run']),
                      ('vote_set', vset), ('offset_src', lab['offset_src']),
                      ('sample_stratum', lab['stratum'] or 'not_sampled')):
                groups[(g, 'total')] += 1
                if r:
                    groups[(g, 'removed')] += 1
        for (g, kind), v in sorted(groups.items()):
            if kind == 'total':
                surv_rows.append(dict(arm=k, group=g[0], value=g[1], total=v,
                                      removed=groups[(g, 'removed')]))

    # exploratory: fp_class of every voted Disagree, all votes, removed vs left, per arm
    fp_rows = []
    for k in arms:
        rem = chosen[k]['removed']
        c_left, c_rem = Counter(), Counter()
        for lab, r in zip(labels, rem):
            if lab['vote'] != 'Disagree':
                continue
            (c_rem if r else c_left)[lab['fp_class'] or '(none)'] += 1
        agree_rem = sum(1 for lab, r in zip(labels, rem) if r and lab['vote'] == 'Agree')
        for cls in sorted(set(c_left) | set(c_rem), key=lambda x: (-(c_left[x] + c_rem[x]), x)):
            fp_rows.append(dict(arm=k, fp_class=cls, disagree_left=c_left[cls],
                                disagree_removed=c_rem[cls]))
        fp_rows.append(dict(arm=k, fp_class='(Agree votes, all)', disagree_left='',
                            disagree_removed=agree_rem))
    # Amendment 1 (exploratory, post hoc): within-sequence recurrence, all votes, no gate
    ws_rows = []
    for c in (8, 16):
        for k in (3, 4, 5, 6, 8, 10):
            rem = within_sequence_recurrence(labels, c, k)
            row = dict(cell_px=c, min_panos=k, removed=sum(rem))
            for name, sel in (('all_votes', lambda lab: True),
                              ('tuning', lambda lab: not lab['in_sample']),
                              ('random_sample', lambda lab: lab['in_sample'])):
                vv = [(lab, r) for lab, r in zip(labels, rem) if sel(lab) and lab['vote']]
                row[f'{name}_agree_removed'] = sum(1 for lab, r in vv if r and lab['vote'] == 'Agree')
                row[f'{name}_disagree_removed'] = sum(1 for lab, r in vv if r and lab['vote'] == 'Disagree')
            row['roof_removed_all_votes'] = sum(1 for lab, r in zip(labels, rem)
                                                if r and lab['fp_class'] == 'capture-car roof')
            row['gopro_removed'] = sum(1 for lab, r in zip(labels, rem)
                                       if r and lab['camera_model'] == 'GoPro Max')
            ws_rows.append(row)
    write_csv(out / 'amendment1_within_sequence.csv', ws_rows)
    write_csv(out / 'arms.csv', arm_rows)
    write_csv(out / 'survivorship.csv', surv_rows)
    write_csv(out / 'remaining_fp.csv', fp_rows)
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, sort_keys=True, default=str)
    for r in arm_rows:
        print({k: r[k] for k in ('mode', 'arm', 'params', 'removed', 'precision', 'delta', 'delta_lo',
                                 'delta_hi', 'tp_lost_share', 'control_p', 'recommended',
                                 'tune_precision', 'tune_agree_removed', 'tune_disagree_removed')})
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
