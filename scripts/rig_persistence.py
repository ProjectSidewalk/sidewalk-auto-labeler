#!/usr/bin/env python3
"""Rig persistence: find camera-rig artifacts by sequence persistence, without a hand-set angle.

A camera rig (roof rack, mount, burned-in logo band) is fixed in the camera frame, so a false
detection on it recurs at nearly the same image position frame after frame along one capture
sequence; a real ramp moves through the image as the camera passes. This script flags a
detection >= 0.30 when members in >= K distinct non-stationary frames of its sequence (and
>= F of the sequence's frames) sit within T heatmap px of it (Chebyshev, seam-wrapped in x),
tunes (T, K, F) on Laurens only, freezes them, and scores every other city. The nadir mask is
deliberately NOT applied (it is what the rule is compared with). Pre-registration, results and
caveats: docs/overnight/rig_persistence.md.

No network, no GPU, stdlib only (imports eval_sites / detectors / geo from this repo).

Usage:
    python scripts/rig_persistence.py run \
        --runs-root runs \
        --benchmark-root ../RampNet/benchmark \
        --out docs/overnight/data/rig_persistence

    # a subset of cities (laurens is always loaded: it sets the frozen parameters)
    python scripts/rig_persistence.py run --cities laurens bayonne ...

Outputs (all small CSV/JSON): tuning_grid.csv, frozen.json, city_summary.csv, y_hist.csv,
cluster_spans.csv, null.csv, gt_check.csv, examples.csv, summary.json.

Example of the core rule on toy data (frames 0..4 of one sequence, a fixed spot):
    >>> m = [Member('s', g, 512.0, 400.0, 0.4, 'p%d' % g, 0) for g in range(5)]
    >>> support_counts(m, m, 8)
    [4, 4, 4, 4, 4]
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict, namedtuple
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Root inserted last so it ends up first: scripts/position_check.py is a shim that
# would otherwise shadow the root position_check module.
for p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import geo  # noqa: E402
from detectors import (BENCHMARK_CONFIDENCE, BORDER_EXCLUDE, DECODE_ARGMAX,  # noqa: E402
                       OPERATIONAL_CONFIDENCE, NADIR_MASK_DEG, on_camera_rig,
                       record_border, record_decode, registry)

HM_W, HM_H = geo.HEATMAP_WIDTH, geo.HEATMAP_HEIGHT   # 1024 x 512
STATIONARY_M = 2.0
SUPPORT_FLOOR_E1 = 0.10
GRID_T = (4, 8, 16)
GRID_K = (3, 4, 5, 6, 8, 10)
GRID_F = (0.0, 0.05, 0.10, 0.20)
BAR_A = 0.80
BAR_B = 0.002
BAR_C = 0.10
NULL_DRAWS = 20
NULL_SEED = 20261126
Y_BIN = 0.025
CITIES = ('laurens', 'richmond', 'juneau', 'clovis', 'morgantown', 'annapolis', 'bayonne', 'lyon')
SPLITS = {'laurens_mapillary': 'laurens', 'richmond': 'richmond', 'juneau': 'juneau',
          'clovis': 'clovis', 'morgantown': 'morgantown', 'annapolis': 'annapolis',
          'bayonne': 'bayonne', 'lyon': 'lyon'}

Member = namedtuple('Member', 'seq group col row conf pid idx')


# --- loading ----------------------------------------------------------------------------

class RunPano:
    """Minimal stand-in for fuse_sites' pano: eval_sites.judged_gt_panos reads .detections."""
    __slots__ = ('pid', 'seq', 't', 'rank', 'lat', 'lng', 'detections')

    def __init__(self, pid, seq, t, rank, lat, lng, detections):
        self.pid, self.seq, self.t, self.rank = pid, seq, t, rank
        self.lat, self.lng, self.detections = lat, lng, detections


def _capture_time(pano):
    sm = pano.get('source_metadata') or {}
    if sm.get('captured_at') is not None:
        return float(sm['captured_at']) / 1000.0
    dt = sm.get('datetime')
    if dt:
        try:
            return datetime.fromisoformat(dt.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return None


def load_run(path):
    """Stream results.jsonl -> ({pid: RunPano}, counts). Refuses a record that did not run
    exactly CurbRamp (#168; detection indices stay the stored ones) and non-default
    decode/border."""
    panos, counts = {}, Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                registry.require_curb_ramp_only(rec, f'rig_persistence ({path})')
            except ValueError as e:
                raise SystemExit(str(e)) from None
            if record_decode(rec) != DECODE_ARGMAX or record_border(rec) != BORDER_EXCLUDE:
                raise SystemExit(f'{path}: non-default decode/border record; refused')
            p = rec['pano']
            sm = p.get('source_metadata') or {}
            counts['panos'] += 1
            seq = p.get('sequence_id')
            meta_seq = sm.get('sequence') or sm.get('collection')
            if meta_seq is not None and seq is not None and meta_seq != seq:
                counts['seq_mismatch_dropped'] += 1
                continue
            seq = seq or meta_seq
            t = _capture_time(p)
            if seq is None or t is None:
                counts['no_seq_or_time_dropped'] += 1
                continue
            rank = sm.get('geovisio:rank_in_collection')
            dets = [(i, d['x_normalized'], d['y_normalized'], d['confidence'])
                    for i, d in enumerate(rec.get('detections') or [])]
            pid = p['panorama_id']
            if pid in panos:
                counts['duplicate_pid_dropped'] += 1
                continue
            panos[pid] = RunPano(pid, seq, t, rank if rank is not None else -1,
                                 float(p['lat']), float(p['lng']), dets)
    return panos, counts


def frame_groups(panos):
    """Stationary collapse. Returns ({pid: (seq, group)}, {seq: n_groups},
    {(seq, group): cumulative along-track metres of the group's kept frame})."""
    by_seq = defaultdict(list)
    for p in panos.values():
        by_seq[p.seq].append(p)
    assign, n_frames, along = {}, {}, {}
    for seq, ps in by_seq.items():
        ps.sort(key=lambda p: (p.t, p.rank, p.pid))
        g, kept, cum = -1, None, 0.0
        for p in ps:
            if kept is not None:
                d = geo.haversine_m(kept.lat, kept.lng, p.lat, p.lng)
                if d < STATIONARY_M:
                    assign[p.pid] = (seq, g)
                    continue
                cum += d
            g += 1
            kept = p
            along[(seq, g)] = cum
            assign[p.pid] = (seq, g)
        n_frames[seq] = g + 1
    return assign, n_frames, along


def members_from(panos, assign, min_conf):
    out = []
    for p in panos.values():
        seq, g = assign[p.pid]
        for i, x, y, c in p.detections:
            if c >= min_conf:
                out.append(Member(seq, g, x * HM_W, y * HM_H, c, p.pid, i))
    return out


# --- the rule ---------------------------------------------------------------------------

def _dx(a, b):
    d = abs(a - b) % HM_W
    return min(d, HM_W - d)


def support_counts(targets, supporters, T, positions=None, window=None):
    """For each target, the number of distinct frames (!= its own) of its sequence holding a
    supporter within T heatmap px (Chebyshev, x wrapped); with `window`, only frames whose
    group index is within `window` of the target's count (Amendment A). `positions`
    optionally overrides (col, row) per member, keyed by (pano_id, detection index)."""
    def pos(m):
        return positions[(m.pid, m.idx)] if positions is not None else (m.col, m.row)

    nbx = int(math.ceil(HM_W / T))
    bins = defaultdict(list)
    for s in supporters:
        c, r = pos(s)
        bins[(s.seq, int(c // T) % nbx, int(r // T))].append((s.group, c, r))
    eps = 1e-9
    out = []
    for t in targets:
        c, r = pos(t)
        bx, by = int(c // T) % nbx, int(r // T)
        groups = set()
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for g, sc, sr in bins.get((t.seq, (bx + ox) % nbx, by + oy), ()):
                    if g != t.group and _dx(c, sc) <= T + eps and abs(r - sr) <= T + eps \
                            and (window is None or abs(g - t.group) <= window):
                        groups.add(g)
        out.append(len(groups))
    return out


def flags_from(targets, support, n_frames, K, F):
    return [(1 + s) >= K and (1 + s) / n_frames[t.seq] >= F for t, s in zip(targets, support)]


def below(m):
    return on_camera_rig(m.row / HM_H)


BANDS = ('a_above_horizon', 'b_dip_0_18', 'c_dip_18_49', 'd_below_mask')


def band(m):
    """Descriptive dip band of a member (reporting only, never part of the rule)."""
    y = m.row / HM_H
    if y < 0.5:
        return BANDS[0]
    if y < 0.6:
        return BANDS[1]
    return BANDS[3] if on_camera_rig(y) else BANDS[2]


# --- reporting helpers -------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def clusters(targets, flags, T):
    """Connected components of flagged targets of one sequence linked within T."""
    idx = [i for i, f in enumerate(flags) if f]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    by_seq = defaultdict(list)
    for i in idx:
        by_seq[targets[i].seq].append(i)
    for seq, ii in by_seq.items():
        for a_pos, a in enumerate(ii):
            ma = targets[a]
            for b in ii[a_pos + 1:]:
                mb = targets[b]
                if _dx(ma.col, mb.col) <= T + 1e-9 and abs(ma.row - mb.row) <= T + 1e-9:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
    comps = defaultdict(list)
    for i in idx:
        comps[find(i)].append(i)
    return list(comps.values())


def quantiles(vals, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    if not vals:
        return [None] * len(qs)
    s = sorted(vals)
    return [s[min(len(s) - 1, int(round(q * (len(s) - 1))))] for q in qs]


def write_csv(path, rows, header):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        for r in rows:
            w.writerow([f'{v:.6g}' if isinstance(v, float) else v for v in r])


# --- main ---------------------------------------------------------------------------------

class FrozenRule:
    """The pre-registered rule: sequence-wide support among members >= 0.30, K and fraction F."""
    name, subdir = 'frozen', ''
    keys = ('T', 'K', 'F')

    def grid(self):
        return [{'T': T, 'K': K, 'F': F} for T in GRID_T for K in GRID_K for F in GRID_F]

    def tie(self, r):
        return (-r['K'], -r['F'], r['T'])

    def flags(self, d, prm, positions=None, supporters=None):
        sup = support_counts(d['targets'], supporters or d['targets'], prm['T'], positions)
        return flags_from(d['targets'], sup, d['n_frames'], prm['K'], prm['F'])

    def null_pool(self, d):
        return d['targets']


class LocalRule:
    """Amendment A (exploratory): support from stored detections >= 0.10 within +/-W frames."""
    name, subdir = 'local', 'amendment_a'
    keys = ('T', 'W', 'K')

    def grid(self):
        return [{'T': T, 'W': W, 'K': K} for T in (4, 8) for W in (5, 10) for K in (3, 4, 5, 6, 8)]

    def tie(self, r):
        return (-r['K'], r['T'], r['W'])

    def flags(self, d, prm, positions=None, supporters=None):
        sup = support_counts(d['targets'], d['floor_sup'], prm['T'], positions, window=prm['W'])
        return [s >= prm['K'] for s in sup]

    def null_pool(self, d):
        return d['floor_sup']


def tune(rule, laurens):
    targets = laurens['targets']
    nb = sum(below(m) for m in targets)
    rows = []
    for prm in rule.grid():
        fl = rule.flags(laurens, prm)
        fb = sum(f for f, m in zip(fl, targets) if below(m))
        fa = sum(f for f, m in zip(fl, targets) if not below(m))
        rows.append(dict(prm, below_n=nb, below_flagged=fb,
                         below_recall=fb / nb if nb else float('nan'),
                         above_n=len(targets) - nb, above_flagged=fa,
                         eligible=nb > 0 and fb / nb >= BAR_A))
    elig = [r for r in rows if r['eligible']]
    if elig:
        pick = min(elig, key=lambda r: (r['above_flagged'],) + rule.tie(r))
    else:
        pick = min(rows, key=lambda r: (-r['below_recall'],) + rule.tie(r))
    return rows, pick, bool(elig)


def load_city(city, runs_root):
    path = runs_root / city / 'results.jsonl'
    panos, counts = load_run(path)
    assign, n_frames, along = frame_groups(panos)
    targets = members_from(panos, assign, OPERATIONAL_CONFIDENCE)
    floor_sup = members_from(panos, assign, SUPPORT_FLOOR_E1)
    min_stored = min((c for p in panos.values() for _, _, _, c in p.detections), default=None)
    print(f'[{city}] {counts["panos"]} panos, {len(n_frames)} sequences, '
          f'{sum(n_frames.values())} frames, {len(targets)} members >= 0.30; '
          f'min stored conf {min_stored}', file=sys.stderr, flush=True)
    return {'city': city, 'panos': panos, 'counts': counts, 'assign': assign,
            'n_frames': n_frames, 'along': along, 'targets': targets,
            'floor_sup': floor_sup, 'min_stored': min_stored}


def score_city(rule, d, prm):
    fl = rule.flags(d, prm)
    d['flags'] = {(m.pid, m.idx): f for m, f in zip(d['targets'], fl)}
    d['flag_list'] = fl
    # E1 (frozen rule only): support from every stored detection >= 0.10
    d['flags_e1'] = rule.flags(d, prm, supporters=d['floor_sup']) \
        if isinstance(rule, FrozenRule) else [False] * len(fl)
    return fl


def null_draws(rule, d, prm):
    pool = rule.null_pool(d)
    counts = []
    for k in range(NULL_DRAWS):
        rng = random.Random(NULL_SEED + k)
        pos = [(m.col, m.row) for m in pool]
        rng.shuffle(pos)
        positions = {(m.pid, m.idx): p for m, p in zip(pool, pos)}
        counts.append(sum(rule.flags(d, prm, positions)))
    return counts


def gt_check(split, city_data, benchmark_root):
    import eval_sites as es
    vpath = benchmark_root / split / 'verdicts.json'
    exploratory = False
    if not vpath.exists():
        backups = sorted((benchmark_root / split / 'verdicts_backups').glob('verdicts_*.json'))
        if not backups:
            return None
        vpath, exploratory = backups[-1], True
    with open(vpath, encoding='utf-8') as f:
        verdict_panos = json.load(f)['panos']
    if exploratory:
        bundle_ops = {}
        with open(benchmark_root / split / 'records.jsonl', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    # the type check es.load_benchmark makes on the verdicts.json branch
                    registry.require_curb_ramp_only(rec, f'rig_persistence gt_check ({split})')
                    bundle_ops[rec['pano']['panorama_id']] = \
                        [(x['x_normalized'], x['y_normalized'], x['confidence'])
                         for x in rec['detections']]
    else:
        verdict_panos, bundle_ops = es.load_benchmark(split, benchmark_root)
    counts, warnings = es.gt_counts(), []
    tally = Counter()
    rows = []
    for pid, entry, run_pano, ops, in_pool in es.judged_gt_panos(
            verdict_panos, bundle_ops, city_data['panos'], counts, warnings):
        for (i, x, y, c), v in zip(ops, entry['dets']):
            if v is True:
                lab = 'true'
            elif v is False:
                lab = 'false'
            else:
                continue
            f = city_data['flags'][(pid, i)]
            zone = 'below' if on_camera_rig(y) else 'above'
            tally[(lab, zone, 'n')] += 1
            tally[(lab, zone, 'flagged')] += int(f)
            if f:
                rows.append((split, pid, x, y, c, lab, zone))
    return {'split': split, 'verdicts_file': str(vpath.name), 'exploratory': exploratory,
            'counts': dict(counts), 'n_warnings': len(warnings), 'tally': tally,
            'flagged_rows': rows}


def run_rule(rule, data, cities, out, benchmark_root):
    """Tune `rule` on Laurens, freeze it, score every loaded city, write outputs."""
    out = out / rule.subdir if rule.subdir else out
    out.mkdir(parents=True, exist_ok=True)
    grid, pick, any_elig = tune(rule, data['laurens'])
    write_csv(out / 'tuning_grid.csv', [[g[k] for k in grid[0]] for g in grid], list(grid[0]))
    prm = {k: pick[k] for k in rule.keys}
    T = prm['T']
    frozen = {'rule': rule.name, 'params': prm, 'any_cell_eligible': any_elig,
              'laurens_below_recall': pick['below_recall'],
              'laurens_above_flagged': pick['above_flagged'],
              'stationary_m': STATIONARY_M, 'member_min_conf': OPERATIONAL_CONFIDENCE,
              'nadir_mask_deg_for_reporting_only': NADIR_MASK_DEG}
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=1) + '\n')
    print(f'[{rule.name}] FROZEN {prm} (eligible cells exist: {any_elig})', file=sys.stderr)

    summary_rows, yhist_rows, span_rows, null_rows, example_rows = [], [], [], [], []
    band_rows, band_example_rows = [], []
    summary = {'frozen': frozen, 'cities': {}}
    for city in cities:
        d = data[city]
        fl = score_city(rule, d, prm)
        tg = d['targets']
        z = defaultdict(int)
        for m, f, f1 in zip(tg, fl, d['flags_e1']):
            zone = 'below' if below(m) else 'above'
            tiers = ['t030'] + (['t055'] if m.conf >= BENCHMARK_CONFIDENCE else [])
            for tier in tiers:
                z[(tier, zone, 'n')] += 1
                z[(tier, zone, 'flag')] += int(f)
                z[(tier, zone, 'flag_e1')] += int(f1)
        nseq_frames = d['n_frames']
        row = [city, d['counts']['panos'], len(nseq_frames), sum(nseq_frames.values()),
               d['min_stored']]
        for tier in ('t030', 't055'):
            for zone in ('below', 'above'):
                row += [z[(tier, zone, 'n')], z[(tier, zone, 'flag')], z[(tier, zone, 'flag_e1')]]
        summary_rows.append(row)

        # y histogram of flagged members (>= 0.30)
        h = Counter(int((m.row / HM_H) // Y_BIN) for m, f in zip(tg, fl) if f)
        for b in sorted(h):
            yhist_rows.append([city, round(b * Y_BIN, 4), round((b + 1) * Y_BIN, 4), h[b]])

        # cluster spans
        comps = clusters(tg, fl, T)
        spans, metres = [], []
        for comp in comps:
            gs = sorted({tg[i].group for i in comp})
            seq = tg[comp[0]].seq
            spans.append(len(gs))
            metres.append(d['along'][(seq, gs[-1])] - d['along'][(seq, gs[0])])
            ys = [tg[i].row / HM_H for i in comp]
            span_rows.append([city, seq, len(comp), len(gs), nseq_frames[seq],
                              round(metres[-1], 1), round(min(ys), 4), round(max(ys), 4),
                              'below' if all(on_camera_rig(y) for y in ys) else
                              ('above' if not any(on_camera_rig(y) for y in ys) else 'mixed')])

        # null
        nc = null_draws(rule, d, prm)
        real = sum(fl)
        mean_null = sum(nc) / len(nc)
        null_rows.append([city, real, mean_null, min(nc), max(nc),
                          (mean_null / real) if real else None,
                          ('n/a' if real == 0 else ('PASS' if mean_null <= BAR_C * real else 'FAIL'))])

        # examples: above-mask flags first, highest confidence, one per pano
        fl_m = sorted([m for m, f in zip(tg, fl) if f],
                      key=lambda m: (below(m), -m.conf, m.pid))
        seen = set()
        for m in fl_m:
            if m.pid in seen:
                continue
            seen.add(m.pid)
            example_rows.append([city, m.pid, m.seq, round(m.col / HM_W, 6), round(m.row / HM_H, 6),
                                 round(m.conf, 4), round((m.row / HM_H - 0.5) * 180, 2),
                                 'below' if below(m) else 'above'])
            if len(seen) >= 10:
                break
        # per dip band: flags, sequences, the most common coarse cell and how many sequences
        # share it (a fixture fixed in the camera frame recurs across sequences of one rig)
        by_band = defaultdict(list)
        for m, f in zip(tg, fl):
            if f:
                by_band[band(m)].append(m)
        for b in BANDS:
            ms = by_band.get(b, [])
            n_all = sum(1 for m in tg if band(m) == b)
            cells = Counter((int(m.col) // 8, int(m.row) // 8) for m in ms)
            top = cells.most_common(1)
            if top:
                (cx, cy), cn = top[0]
                in_cell = [m for m in ms if (int(m.col) // 8, int(m.row) // 8) == (cx, cy)]
                cseq = len({m.seq for m in in_cell})
                for m in sorted(in_cell, key=lambda m: (-m.conf, m.pid))[:3]:
                    band_example_rows.append([city, b, m.pid, m.seq, round(m.col / HM_W, 6),
                                              round(m.row / HM_H, 6), round(m.conf, 4)])
                band_rows.append([city, b, n_all, len(ms), len({m.seq for m in ms}),
                                  round((cx * 8 + 4) / HM_W, 4), round((cy * 8 + 4) / HM_H, 4),
                                  cn, cseq])
            else:
                band_rows.append([city, b, n_all, 0, 0, None, None, 0, 0])
        summary['cities'][city] = {
            'load_counts': dict(d['counts']), 'flagged': real, 'null_mean': mean_null,
            'clusters': len(comps), 'cluster_frames_q': quantiles(spans),
            'cluster_metres_q': quantiles(metres),
            'zones': {f'{k[0]}_{k[1]}_{k[2]}': v for k, v in z.items()}}
        print(f'[{city}] flagged {real} (null mean {mean_null:.1f}); clusters {len(comps)}',
              file=sys.stderr, flush=True)

    hdr = ['city', 'panos', 'sequences', 'frames', 'min_stored_conf']
    for tier in ('t030', 't055'):
        for zone in ('below', 'above'):
            hdr += [f'{tier}_{zone}_n', f'{tier}_{zone}_flag', f'{tier}_{zone}_flag_e1']
    write_csv(out / 'city_summary.csv', summary_rows, hdr)
    write_csv(out / 'y_hist.csv', yhist_rows, ['city', 'y_lo', 'y_hi', 'flagged'])
    write_csv(out / 'cluster_spans.csv', span_rows,
              ['city', 'sequence', 'members', 'frames', 'seq_frames', 'along_m', 'y_min',
               'y_max', 'zone'])
    write_csv(out / 'null.csv', null_rows,
              ['city', 'real_flagged', 'null_mean', 'null_min', 'null_max', 'null_over_real',
               'check_c'])
    write_csv(out / 'band_summary.csv', band_rows,
              ['city', 'band', 'members', 'flagged', 'flagged_sequences', 'top_cell_x',
               'top_cell_y', 'top_cell_flags', 'top_cell_sequences'])
    write_csv(out / 'band_examples.csv', band_example_rows,
              ['city', 'band', 'pano_id', 'sequence', 'x', 'y', 'conf'])
    write_csv(out / 'examples.csv', example_rows,
              ['city', 'pano_id', 'sequence', 'x', 'y', 'conf', 'dip_deg', 'zone'])

    # (b) ground truth
    gt_rows, flagged_gt, pooled = [], [], Counter()
    for split, city in SPLITS.items():
        if city not in data or not (benchmark_root / split).exists():
            continue
        res = gt_check(split, data[city], benchmark_root)
        if res is None:
            continue
        t = res['tally']
        gt_rows.append([split, res['verdicts_file'], res['exploratory'], res['counts']['judged'],
                        res['counts']['skipped'], res['counts']['partial'],
                        t[('true', 'above', 'n')] + t[('true', 'below', 'n')],
                        t[('true', 'above', 'flagged')] + t[('true', 'below', 'flagged')],
                        t[('false', 'below', 'n')], t[('false', 'below', 'flagged')],
                        t[('false', 'above', 'n')], t[('false', 'above', 'flagged')]])
        flagged_gt += res['flagged_rows']
        if not res['exploratory']:
            for k, v in t.items():
                pooled[k] += v
    write_csv(out / 'gt_check.csv', gt_rows,
              ['split', 'verdicts_file', 'exploratory', 'judged_panos', 'skipped_panos',
               'partial_panos', 'true_n', 'true_flagged', 'false_below_n', 'false_below_flagged',
               'false_above_n', 'false_above_flagged'])
    write_csv(out / 'gt_flagged.csv', flagged_gt,
              ['split', 'pano_id', 'x', 'y', 'conf', 'verdict', 'zone'])
    tn = pooled[('true', 'above', 'n')] + pooled[('true', 'below', 'n')]
    tf = pooled[('true', 'above', 'flagged')] + pooled[('true', 'below', 'flagged')]
    lnr = data['laurens']
    lb = sum(1 for m in lnr['targets'] if below(m))
    lbf = sum(1 for m, f in zip(lnr['targets'], lnr['flag_list']) if f and below(m))
    summary['checks'] = {
        'a': {'laurens_below_n': lb, 'flagged': lbf, 'recall': lbf / lb if lb else None,
              'ci95': wilson(lbf, lb), 'bar': BAR_A,
              'verdict': 'PASS' if lb and lbf / lb >= BAR_A else 'FAIL'},
        'b': {'true_n': tn, 'true_flagged': tf, 'rate': tf / tn if tn else None,
              'ci95': wilson(tf, tn), 'bar': BAR_B,
              'verdict': 'PASS' if tn and tf / tn <= BAR_B else 'FAIL'},
        'c': {r[0]: r[-1] for r in null_rows}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print(rule.name, json.dumps(summary['checks'], indent=1, default=str))
    return summary




def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--runs-root', type=Path, required=True)
    r.add_argument('--benchmark-root', type=Path, required=True)
    r.add_argument('--out', type=Path, default=REPO_ROOT / 'docs/overnight/data/rig_persistence')
    r.add_argument('--cities', nargs='+', default=list(CITIES))
    r.add_argument('--rule', choices=('frozen', 'local', 'both'), default='both',
                   help='frozen = the pre-registered rule; local = Amendment A (exploratory)')
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    cities = ['laurens'] + [c for c in a.cities if c != 'laurens']

    data = {c: load_city(c, a.runs_root) for c in cities}
    rules = {'frozen': [FrozenRule()], 'local': [LocalRule()],
             'both': [FrozenRule(), LocalRule()]}[a.rule]
    for rule in rules:
        run_rule(rule, data, cities, out, a.benchmark_root)


if __name__ == '__main__':
    main()
