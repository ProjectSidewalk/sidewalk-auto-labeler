#!/usr/bin/env python3
"""label_frame_beta.py — how far a human Project Sidewalk label's pano_y sits from the
image's own frame, as a fraction (beta) of the rig's tilt (issue #113).

GSV's equirectangulars are in the capture rig's frame (sidewalk-panorama-tools#158). A
RampNet detection's row is read off those pixels, so it is in the image's frame. A human
label's pano_y is whatever PS's click mapping wrote. If the labelling viewer showed a
levelled view and the mapping ignored the tilt, the two differ by beta * T(b):

    el_detection - el_human = beta * T(b) + c
    T(b) = pitch cos b + roll sin b        b = (x - 0.5) * 360,  el = (0.5 - y) * 180

with streetlevel's pitch and roll (pitch > 0 = nose DOWN, roll > 0 = left side up) and b
the bearing from the centre column. beta = 0: one frame. beta = 1: the label is off by the
whole tilt. The intercept c absorbs where on a ramp a person clicks against where the
detector peaks; it is not a tilt effect.

THIS BETA IS NOT A PLACEMENT COEFFICIENT. It is about the viewer. How much of the tilt
leaks into the ground raycast is a different question with a different answer (#116).

Pairing: each label with the nearest detection >= the tier on the SAME pano, inside a
window that is narrow in bearing and wide in elevation, so a shift of several degrees is
not cut off. A wrong pairing (another ramp at that bearing) does not depend on the sign of
T, so it adds noise and widens the SE; it does not bias the slope. The window is reported
at three widths because it still matters: a narrow one truncates large shifts.

Two modes, one fit:

  run   a city this repo has detected. Labels are pulled once from the server's v3 API
        into runs/<city>/label_frame_beta/ (read-only GET; reused on re-runs, --refresh
        re-pulls); detections and pose come from results.jsonl.
  pool  sidewalk-panorama-tools' vouched label pool and its pose scan, plus a detections
        file made by running the detector over those store panos. Reported per label era.

    python scripts/label_frame_beta.py run gainesville \
        --server https://sidewalk-gainesville.cs.washington.edu
    python scripts/label_frame_beta.py pool --pool <tilt-jm-pool.csv.gz> \
        --pose <tilt-pose-jm.csv.gz> --detections <pool_detections.jsonl>

Stdlib only, no network outside `run`'s one GET. Writes report.md, fits.csv, bins.csv and
pairs.csv (keyed on label_uid = <city>:<label_id>; a label_id alone is per-city).
"""
import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import agree_rate as ar  # noqa: E402  (fetch / snapshot_row: the frozen-pull helpers)
from detectors import (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       on_camera_rig)

WINDOW_BEARING_DEG = 3.0
WINDOWS_ELEVATION_DEG = (4.0, 6.0, 10.0)
HEADLINE_WINDOW_DEG = 6.0
TIERS = (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE)
BIN_EDGES_DEG = (-99.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 99.0)
MIN_BIN = 10
MIN_FIT = 30
# rawlabels.era in sidewalk-panorama-tools: PS release boundaries, not viewer changes
ERA_MID_FROM = '2021-01-01'
ERA_POST179_FROM = '2023-03-29'


# ------------------------------------------------------------------------- geometry

def wrap180(deg):
    """An angle in (-180, 180]. GSV camera_roll is stored unwrapped (359.4 means -0.6)."""
    a = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def tilt_terms(pitch_deg, roll_deg, x_norm):
    """(pitch cos b, roll sin b) in degrees at a normalized pano x."""
    b = math.radians((x_norm - 0.5) * 360.0)
    return pitch_deg * math.cos(b), wrap180(roll_deg) * math.sin(b)


def xml_pitch_roll(pano_yaw_deg, tilt_yaw_deg, tilt_pitch_deg):
    """streetlevel-sign (pitch, roll) from the legacy XML's projection properties.

    The convention sidewalk-panorama-tools fitted on 3,594 Seattle panos carrying both
    records (#158): pitch = -m cos(dir), roll = -m sin(dir), m = tilt_pitch_deg,
    dir = tilt_yaw_deg - pano_yaw_deg.
    """
    d = math.radians(tilt_yaw_deg - pano_yaw_deg)
    return -tilt_pitch_deg * math.cos(d), -tilt_pitch_deg * math.sin(d)


def era_of(time_created):
    day = (time_created or '')[:10]
    if not day:
        return 'unknown'
    if day < ERA_MID_FROM:
        return 'legacy'
    return 'mid' if day < ERA_POST179_FROM else 'post179'


# --------------------------------------------------------------------------- inputs

@dataclass
class Label:
    uid: str            # <city>:<label_id>
    key: tuple          # (city, pano_id)
    x: float
    y: float
    width: int
    height: int
    era: str
    vouched: bool       # run: a human-Agree majority; pool: always (that is the pool)


@dataclass
class Pano:
    pitch: float
    roll: float
    width: int
    height: int
    pose_source: str
    detections: list    # [(x, y, confidence)], rig-masked


def _masked(dets):
    return [(x, y, c) for x, y, c in dets if not on_camera_rig(y)]


def load_run_panos(results_path, city):
    out = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            p = r['pano']
            if p.get('camera_pitch') is None or p.get('camera_roll') is None:
                continue
            out[(city, p['panorama_id'])] = Pano(
                p['camera_pitch'], p['camera_roll'], int(p['width']), int(p['height']),
                'results.jsonl',
                _masked([(d['x_normalized'], d['y_normalized'], d['confidence'])
                         for d in r['detections']]))
    return out


def load_api_labels(path, city, exclude_users=()):
    """(labels, counts) from a rawLabels geojson. GSV labels only, since the tilt model
    is GSV's; labels by an excluded user (an AI account) are dropped and counted."""
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    out, counts = [], {'features': len(feats), 'excluded_user': 0, 'not_gsv': 0,
                       'no_pixel': 0}
    for ft in feats:
        q = ft['properties']
        if str(q.get('user_id')) in exclude_users:
            counts['excluded_user'] += 1
            continue
        if q.get('pano_source') not in (None, 'gsv'):
            counts['not_gsv'] += 1
            continue
        w, h = q.get('pano_width'), q.get('pano_height')
        if not (w and h) or q.get('pano_x') is None or q.get('pano_y') is None:
            counts['no_pixel'] += 1
            continue
        out.append(Label(
            uid=f"{city}:{q['label_id']}", key=(city, q['pano_id']),
            x=q['pano_x'] / w, y=q['pano_y'] / h, width=int(w), height=int(h),
            era=era_of(q.get('time_created')),
            vouched=ar.human_status(q.get('validations'), str(q.get('user_id'))) == 'agreed'))
    uids = [lab.uid for lab in out]
    if len(set(uids)) != len(uids):
        raise SystemExit(f'{path}: duplicate label_uid — the feed repeats a label')
    return out, counts


def _num(value):
    return float(value) if value not in (None, '') else None


def load_pool(pool_path, pose_path, detections_path, label_type):
    """(labels, panos, counts) for sidewalk-panorama-tools' vouched pool.

    A store JPEG with an .xml beside it is a 2019-22 stitch and is posed by that XML,
    never by a later record that may describe a re-render (#158's rule); otherwise the
    pose is the modern one. A label whose stored pano size differs from the JPEG's was
    made on other pixels and is dropped.
    """
    dets = {}
    with open(detections_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if 'error' not in r:
                dets[(r['city'], r['pano_id'])] = (int(r['native_w']), int(r['native_h']),
                                                   [tuple(d) for d in r['detections']])
    panos = {}
    counts = {'no_pose_row': 0, 'no_pose': 0, 'not_detected': 0, 'dims_differ': 0,
              'other_type_or_source': 0}
    with gzip.open(pose_path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = (r['city'], r['pano_id'])
            if key not in dets:
                continue
            xml = [_num(r[k]) for k in ('xml_pano_yaw_deg', 'xml_tilt_yaw_deg',
                                        'xml_tilt_pitch_deg')]
            if r['xml_present'] == '1' and None not in xml:
                pitch, roll = xml_pitch_roll(*xml)
                source = 'xml'
            elif r['npz_present'] == '1' and _num(r['pitch_deg']) is not None \
                    and _num(r['roll_deg']) is not None:
                pitch, roll, source = _num(r['pitch_deg']), _num(r['roll_deg']), 'npz'
            else:
                continue
            w, h, d = dets[key]
            panos[key] = Pano(pitch, roll, w, h, source, _masked(d))
    posed_rows = set()
    with gzip.open(pose_path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            posed_rows.add((r['city'], r['pano_id']))
    labels = []
    with gzip.open(pool_path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['label_type'] != label_type or r['pano_source'] != 'gsv':
                counts['other_type_or_source'] += 1
                continue
            key = (r['city'], r['pano_id'])
            w, h = _num(r['pano_width']), _num(r['pano_height'])
            if key not in posed_rows:
                counts['no_pose_row'] += 1
            elif key not in dets:
                counts['not_detected'] += 1
            elif key not in panos:
                counts['no_pose'] += 1
            elif not (w and h) or (int(w), int(h)) != (panos[key].width, panos[key].height):
                counts['dims_differ'] += 1
            else:
                labels.append(Label(uid=r['label_uid'], key=key,
                                    x=float(r['pano_x']) / w, y=float(r['pano_y']) / h,
                                    width=int(w), height=int(h), era=r['era'], vouched=True))
    uids = [lab.uid for lab in labels]
    if len(set(uids)) != len(uids):
        raise SystemExit(f'{pool_path}: duplicate label_uid')
    return labels, panos, counts


# -------------------------------------------------------------------------- pairing

@dataclass
class Pair:
    uid: str
    key: tuple
    era: str
    vouched: bool
    pose_source: str
    t_pitch: float      # pitch cos b, deg
    t_roll: float       # roll sin b, deg
    diff: float         # el_detection - el_human, deg
    confidence: float

    @property
    def t(self):
        return self.t_pitch + self.t_roll


def pair_labels(labels, panos, tier, window_elevation_deg,
                window_bearing_deg=WINDOW_BEARING_DEG):
    """([Pair], counts). One pair per label at most; a detection may serve two labels
    (two people marking one ramp are two observations of the same offset)."""
    out = []
    counts = {'labels': len(labels), 'pano_not_available': 0, 'dims_differ': 0,
              'no_detection_in_window': 0}
    for lab in labels:
        pano = panos.get(lab.key)
        if pano is None:
            counts['pano_not_available'] += 1
            continue
        if (pano.width, pano.height) != (lab.width, lab.height):
            counts['dims_differ'] += 1
            continue
        best = None
        for x, y, c in pano.detections:
            if c < tier:
                continue
            dx = abs(x - lab.x) % 1.0
            dx = min(dx, 1.0 - dx) * 360.0
            dy = (y - lab.y) * 180.0
            if dx <= window_bearing_deg and abs(dy) <= window_elevation_deg:
                d = math.hypot(dx, dy)
                if best is None or d < best[0]:
                    best = (d, y, c)
        if best is None:
            counts['no_detection_in_window'] += 1
            continue
        tp, tr = tilt_terms(pano.pitch, pano.roll, lab.x)
        out.append(Pair(lab.uid, lab.key, lab.era, lab.vouched, pano.pose_source,
                        tp, tr, (lab.y - best[1]) * 180.0, best[2]))
    return out, counts


# ------------------------------------------------------------------------------ fit

def _solve(a, b):
    """x with a x = b, by Gauss-Jordan with partial pivoting (k <= 3 here)."""
    n = len(a)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12:
            raise ValueError('singular design')
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [vr - f * vi for vr, vi in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols_clustered(X, y, clusters):
    """(coefficients, standard errors), errors clustered by `clusters` (CR0 sandwich).

    Labels on one pano share its pose, so their residuals are not independent.
    """
    k = len(X[0])
    xtx = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    beta = _solve(xtx, [sum(r[i] * yi for r, yi in zip(X, y)) for i in range(k)])
    inv = [_solve(xtx, [1.0 if i == j else 0.0 for i in range(k)]) for j in range(k)]
    inv = [[inv[j][i] for j in range(k)] for i in range(k)]
    score = {}
    for r, yi, c in zip(X, y, clusters):
        e = yi - sum(bi * ri for bi, ri in zip(beta, r))
        s = score.setdefault(c, [0.0] * k)
        for i in range(k):
            s[i] += r[i] * e
    meat = [[sum(s[i] * s[j] for s in score.values()) for j in range(k)] for i in range(k)]
    left = [[sum(inv[i][a] * meat[a][j] for a in range(k)) for j in range(k)]
            for i in range(k)]
    var = [sum(left[i][a] * inv[a][i] for a in range(k)) for i in range(k)]
    return beta, [math.sqrt(max(v, 0.0)) for v in var]


def _pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def _median(values):
    v = sorted(values)
    n = len(v)
    return (v[n // 2] + v[(n - 1) // 2]) / 2.0


def fit(pairs):
    """One row of fits.csv, or None below MIN_FIT pairs."""
    if len(pairs) < MIN_FIT:
        return None
    y = [p.diff for p in pairs]
    cl = [p.key for p in pairs]
    try:
        b1, s1 = ols_clustered([[1.0, p.t] for p in pairs], y, cl)
        b2, s2 = ols_clustered([[1.0, p.t_pitch, p.t_roll] for p in pairs], y, cl)
    except ValueError:
        return None
    abs_t = [abs(p.t) for p in pairs]
    return {'pairs': len(pairs), 'panos': len(set(cl)),
            'beta': b1[1], 'beta_se': s1[1], 'intercept': b1[0],
            'beta_pitch': b2[1], 'beta_pitch_se': s2[1],
            'beta_roll': b2[2], 'beta_roll_se': s2[2],
            'abs_t_p50': _pct(abs_t, 0.5), 'abs_t_p90': _pct(abs_t, 0.9),
            'abs_t_max': max(abs_t),
            'n_abs_t_ge_3': sum(1 for t in abs_t if t >= 3.0)}


def bins(pairs):
    out = []
    for lo, hi in zip(BIN_EDGES_DEG, BIN_EDGES_DEG[1:]):
        sel = [p for p in pairs if lo <= p.t < hi]
        if len(sel) >= MIN_BIN:
            out.append({'t_from': lo, 't_to': hi, 'n': len(sel),
                        'median_t': _median([p.t for p in sel]),
                        'median_diff': _median([p.diff for p in sel])})
    return out


def groups_of(pairs, mode):
    """[(name, pairs)]: the strata one report reads side by side."""
    out = [('all', pairs)]
    if mode == 'run':
        out.append(('human-agreed', [p for p in pairs if p.vouched]))
    for era in ('legacy', 'mid', 'post179'):
        sel = [p for p in pairs if p.era == era]
        if sel and len(sel) != len(pairs):
            out.append((f'era {era}', sel))
    for src in sorted({p.pose_source for p in pairs}):
        sel = [p for p in pairs if p.pose_source == src]
        if len(sel) != len(pairs):
            out.append((f'pose {src}', sel))
    out.append(('abs T >= 3 deg', [p for p in pairs if abs(p.t) >= 3.0]))
    return out


# --------------------------------------------------------------------------- output

def _f(v, nd=3):
    return '' if v is None else f'{v:.{nd}f}'


def analyse(labels, panos, mode):
    fits, bin_rows, headline_pairs, pair_counts = [], [], None, {}
    for tier in TIERS:
        for win in WINDOWS_ELEVATION_DEG:
            pairs, counts = pair_labels(labels, panos, tier, win)
            pair_counts[(tier, win)] = counts
            head = tier == BENCHMARK_CONFIDENCE and win == HEADLINE_WINDOW_DEG
            if head:
                headline_pairs = pairs
            for name, sel in groups_of(pairs, mode):
                row = fit(sel)
                if row is not None:
                    fits.append({'tier': tier, 'window_deg': win, 'group': name, **row})
                if head:
                    for b in bins(sel):
                        bin_rows.append({'group': name, **b})
    return fits, bin_rows, headline_pairs or [], pair_counts


def write_outputs(out_dir, title, inputs, load_counts, labels, panos, mode):
    out_dir.mkdir(parents=True, exist_ok=True)
    fits, bin_rows, pairs, pair_counts = analyse(labels, panos, mode)

    fit_cols = ['tier', 'window_deg', 'group', 'pairs', 'panos', 'beta', 'beta_se',
                'intercept', 'beta_pitch', 'beta_pitch_se', 'beta_roll', 'beta_roll_se',
                'abs_t_p50', 'abs_t_p90', 'abs_t_max', 'n_abs_t_ge_3']
    with open(out_dir / 'fits.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fit_cols)
        w.writeheader()
        for r in fits:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    with open(out_dir / 'bins.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['group', 't_from', 't_to', 'n', 'median_t',
                                          'median_diff'])
        w.writeheader()
        for r in bin_rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    with open(out_dir / 'pairs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label_uid', 'city', 'pano_id', 'era', 'vouched', 'pose_source',
                    't_pitch_deg', 't_roll_deg', 't_deg', 'diff_deg', 'confidence'])
        for p in sorted(pairs, key=lambda p: p.uid):
            w.writerow([p.uid, p.key[0], p.key[1], p.era, int(p.vouched), p.pose_source,
                        f'{p.t_pitch:.4f}', f'{p.t_roll:.4f}', f'{p.t:.4f}',
                        f'{p.diff:.4f}', f'{p.confidence:.4f}'])

    L = [f'# {title}: label-frame beta (issue #113)', '',
         'How far a human label\'s `pano_y` sits from the image\'s own frame, as a fraction '
         'of the rig tilt at the label\'s bearing. beta = 0: one frame; beta = 1: off by '
         'the whole tilt. **This is not a placement coefficient** (that is #116).', '',
         '## Inputs', '']
    L += [f'- {line}' for line in inputs]
    L += [f'- labels kept: {len(labels)}; dropped while loading: '
          + ', '.join(f'{k} {v}' for k, v in load_counts.items() if k != 'features'),
          f'- panos with a pose and detections: {len(panos)}', '']
    head = [r for r in fits if r['tier'] == BENCHMARK_CONFIDENCE
            and r['window_deg'] == HEADLINE_WINDOW_DEG]
    L += [f'## Headline (tier {BENCHMARK_CONFIDENCE}, window {WINDOW_BEARING_DEG:g} x '
          f'{HEADLINE_WINDOW_DEG:g} deg)', '',
          '| group | pairs | panos | beta (SE) | beta pitch (SE) | beta roll (SE) | '
          'intercept | abs T p50 / p90 / max | pairs at abs T >= 3 |',
          '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in head:
        L.append(f"| {r['group']} | {r['pairs']} | {r['panos']} | {_f(r['beta'])} "
                 f"({_f(r['beta_se'])}) | {_f(r['beta_pitch'])} ({_f(r['beta_pitch_se'])}) | "
                 f"{_f(r['beta_roll'])} ({_f(r['beta_roll_se'])}) | {_f(r['intercept'], 2)} | "
                 f"{_f(r['abs_t_p50'], 2)} / {_f(r['abs_t_p90'], 2)} / "
                 f"{_f(r['abs_t_max'], 2)} | {r['n_abs_t_ge_3']} |")
    c = pair_counts[(BENCHMARK_CONFIDENCE, HEADLINE_WINDOW_DEG)]
    L += ['', f"Of {c['labels']} labels: {c['pano_not_available']} on a pano with no pose "
          f"or detections, {c['dims_differ']} whose stored pano size differs from the "
          f"image's, {c['no_detection_in_window']} with no detection in the window.", '',
          '## Every tier and window (group `all`)', '',
          '| tier | window (deg) | pairs | beta (SE) | beta pitch (SE) | beta roll (SE) |',
          '|---|---:|---:|---:|---:|---:|']
    for r in fits:
        if r['group'] == 'all':
            L.append(f"| {r['tier']} | {r['window_deg']:g} | {r['pairs']} | {_f(r['beta'])} "
                     f"({_f(r['beta_se'])}) | {_f(r['beta_pitch'])} "
                     f"({_f(r['beta_pitch_se'])}) | {_f(r['beta_roll'])} "
                     f"({_f(r['beta_roll_se'])}) |")
    L += ['', 'A wider window admits more wrong pairings (noise, not bias) and cuts off '
          'fewer large shifts, so beta rising with the window is truncation at the narrow '
          'end.', '', '## Median offset by signed tilt (headline pairs, group `all`)', '',
          '| T from | T to | n | median T | median (el_detection - el_human) |',
          '|---:|---:|---:|---:|---:|']
    for b in bin_rows:
        if b['group'] == 'all':
            L.append(f"| {b['t_from']:g} | {b['t_to']:g} | {b['n']} | "
                     f"{_f(b['median_t'], 2)} | {_f(b['median_diff'], 2)} |")
    L += ['', '## Reading it', '',
          '- Standard errors are clustered by pano: labels on one pano share its pose.',
          '- The intercept is where a person clicks on a ramp against where the detector '
          'peaks. It is not a tilt effect.',
          '- Detections are rig-masked, as production ships them.',
          '- `pairs.csv` holds the headline pairs, keyed on `label_uid`.', '']
    (out_dir / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    return fits


# ------------------------------------------------------------------------------ cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run', help='a city this repo has detected')
    r.add_argument('city')
    r.add_argument('--server', required=True, help='the PS instance, e.g. https://sidewalk-<city>...')
    r.add_argument('--run-dir', type=Path, help='read results.jsonl from here (read-only)')
    r.add_argument('--label-type', default='CurbRamp')
    r.add_argument('--exclude-user', action='append', default=[],
                   help='a user_id to drop (an AI account); repeatable')
    r.add_argument('--refresh', action='store_true', help='re-pull the labels')
    r.add_argument('--out', type=Path)
    p = sub.add_parser('pool', help="sidewalk-panorama-tools' vouched pool")
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--pose', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--label-type', default='CurbRamp')
    p.add_argument('--out', type=Path)
    args = ap.parse_args(argv)

    if args.mode == 'run':
        out = args.out or REPO / 'runs' / args.city / 'label_frame_beta'
        results = (args.run_dir or REPO / 'runs' / args.city) / 'results.jsonl'
        if not results.exists():
            sys.exit(f'{results}: no such file')
        pull = out / f'raw_labels_{args.label_type}.geojson'
        ar.fetch(args.server.rstrip('/') + ar.API_LABELS.format(args.label_type), pull,
                 refresh=args.refresh)
        name, url, fetched, sha, n = ar.snapshot_row(pull)
        labels, counts = load_api_labels(pull, args.city, set(args.exclude_user))
        panos = load_run_panos(results, args.city)
        inputs = [f'labels: `{name}`, {n} features, fetched {fetched} from {url}, '
                  f'sha256 `{sha}`',
                  f'run: `results.jsonl` sha256 `{ar.sha256_file(results)}`']
        title = args.city
    else:
        out = args.out or REPO / 'runs' / '_pooled' / 'label_frame_beta'
        labels, panos, counts = load_pool(args.pool, args.pose, args.detections,
                                          args.label_type)
        inputs = [f'{what}: `{path.name}` sha256 `{ar.sha256_file(path)}`'
                  for what, path in (('pool', args.pool), ('pose scan', args.pose),
                                     ('detections', args.detections))]
        title = f'vouched pool ({args.label_type})'
    fits = write_outputs(out, title, inputs, counts, labels, panos, args.mode)
    for row in fits:
        if row['tier'] == BENCHMARK_CONFIDENCE and row['window_deg'] == HEADLINE_WINDOW_DEG:
            print(f"{row['group']:>16}: pairs {row['pairs']:6d}  beta {row['beta']:.3f} "
                  f"(se {row['beta_se']:.3f})")
    print(f'wrote {out / "report.md"}')


if __name__ == '__main__':
    main()
