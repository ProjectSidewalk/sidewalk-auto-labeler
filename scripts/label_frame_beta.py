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
window narrow in bearing (3 deg) and wider in elevation. A wrong pairing (another ramp at
that bearing) does not depend on the sign of T, so it adds noise, not bias. The window
itself IS a bias: centred on diff = 0 it cuts off the largest shifts and pulls beta toward
0, more the narrower it is. So every fit is also run with the window centred on the
predicted shift T (beta = 1), which truncates the other way; the two bracket beta, and
they meet by 10 deg on the vouched pool. The headline is the 10 deg zero-centred window.

Pose error attenuates beta too (errors in the regressor). Where a pano carries two poses
(a legacy XML and a modern npz), pool mode fits the SAME pairs under each: a gap between
them belongs to the pose record, not to the label's era.

Two modes, one fit:

  run   a city this repo has detected. Labels are pulled once from the server's v3 API
        into runs/<city>/label_frame_beta/ (read-only GET; reused on re-runs, --refresh
        re-pulls); detections and pose come from results.jsonl. Refuses when labels sit
        on their own detections (an AI account's labels) unless --exclude-user names it
        or --allow-self-pairs says it was checked.
  pool  sidewalk-panorama-tools' vouched label pool and its pose scan, plus a detections
        file made by running the detector over those store panos. Reported per label era.

The pool result is rebuilt end to end with three commands (the inputs are pinned in
POOL_INPUTS; download them from the URLs there and the sha256 is checked):

    python scripts/label_frame_beta.py pool-ids --pool <tilt-jm-pool.csv.gz> \
        --pose <tilt-pose-jm.csv.gz> --out runs/_pooled/label_frame_beta/pool_ids.txt
    python scripts/label_frame_beta.py pool-detect --ids .../pool_ids.txt \
        --store /projects/makeabilitylab/sidewalk_panos/Panoramas \
        --out .../pool_detections.jsonl          # GPU; ~2.8 h on makelab2's A40
    python scripts/label_frame_beta.py pool --pool <...> --pose <...> \
        --detections .../pool_detections.jsonl

    python scripts/label_frame_beta.py run gainesville \
        --server https://sidewalk-gainesville.cs.washington.edu

Stdlib only outside `pool-detect` (which loads the detector), no network outside `run`'s
one GET. Writes report.md, fits.csv, bins.csv and pairs.csv (keyed on
label_uid = <city>:<label_id>; a label_id alone is per-city).
"""
import argparse
import csv
import dataclasses
import gzip
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import agree_rate as ar  # noqa: E402  (fetch / snapshot_row: the frozen-pull helpers)
from detectors import (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       on_camera_rig)

WINDOW_BEARING_DEG = 3.0
WINDOWS_ELEVATION_DEG = (4.0, 6.0, 10.0, 15.0)
HEADLINE_WINDOW_DEG = 10.0
CENTRES = ('zero', 'tilt')      # the elevation window centred on diff = 0, or on diff = T
TIERS = (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE)
BIN_EDGES_DEG = (-99.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 99.0)
MIN_BIN = 10
MIN_FIT = 30
# rawlabels.era in sidewalk-panorama-tools: PS release boundaries, not viewer changes
ERA_MID_FROM = '2021-01-01'
ERA_POST179_FROM = '2023-03-29'
# run mode: a label within this many pano pixels of a detection on both axes was placed
# from it (an AI label), and more than this share of such labels refuses the run
SELF_PAIR_PX = 1.0
SELF_PAIR_MAX_SHARE = 0.01

_PT = ('https://raw.githubusercontent.com/ProjectSidewalk/sidewalk-panorama-tools/'
       '2d0fe60f5b86547381d048698714b42029901c7d/reports/data/')
# The pool inputs the committed report was made from, by file name: where to get them and
# what they must hash to. A file under another name, or with another hash, is reported
# as unpinned rather than refused.
POOL_INPUTS = {
    '2026-09-29-tilt-jm-pool.csv.gz': (
        _PT + '2026-09-29-tilt-jm-pool.csv.gz',
        'c048471769b3835b422b5df5c130561503fe0a6969fa3ff469bf402c8b774dcd'),
    '2026-09-29-tilt-pose-jm.csv.gz': (
        _PT + '2026-09-29-tilt-pose-jm.csv.gz',
        '2c4c2ab0c75ac7b05ea2d960223eafb282fbcecbcf08bf155499921ab5f38362'),
}


# ------------------------------------------------------------------------- geometry

def wrap180(deg):
    """An angle in (-180, 180]. GSV camera_roll is stored unwrapped (359.4 means -0.6)."""
    a = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def tilt_terms(pitch_deg, roll_deg, x_norm):
    """(pitch cos b, roll sin b) in degrees at a normalized pano x."""
    b = math.radians((x_norm - 0.5) * 360.0)
    return wrap180(pitch_deg) * math.cos(b), wrap180(roll_deg) * math.sin(b)


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
    alt_pose: tuple = None  # (pitch, roll) of a second record: the npz of an XML-posed pano


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


def self_pairs(labels, panos, px=SELF_PAIR_PX):
    """Labels sitting within `px` pano pixels of a stored detection on both axes.

    A human click lands there by chance a handful of times in a city; an AI label is
    placed from a detection's pixel, so every one of them does. Such a label paired with
    its own detection reads diff = 0 at any T and drags beta toward 0.
    """
    n = 0
    for lab in labels:
        pano = panos.get(lab.key)
        if pano is None:
            continue
        tx, ty = px / lab.width, px / lab.height
        for x, y, _ in pano.detections:
            dx = abs(x - lab.x) % 1.0
            if min(dx, 1.0 - dx) <= tx and abs(y - lab.y) <= ty:
                n += 1
                break
    return n


def _num(value):
    return float(value) if value not in (None, '') else None


def pose_from_scan_row(r):
    """(pitch, roll, source) from one row of pano-tools' pose scan, or None.

    A store JPEG with an .xml beside it is a 2019-22 stitch and is posed by that XML,
    never by a later record that may describe a re-render (#158's rule); otherwise the
    modern pose is used.
    """
    xml = [_num(r[k]) for k in ('xml_pano_yaw_deg', 'xml_tilt_yaw_deg', 'xml_tilt_pitch_deg')]
    if r.get('xml_present') == '1' and None not in xml:
        pitch, roll = xml_pitch_roll(*xml)
        return pitch, roll, 'xml'
    return npz_pose_of_scan_row(r)


def npz_pose_of_scan_row(r):
    """(pitch, roll, 'npz') from the scan row's modern record, or None."""
    if r.get('npz_present') == '1' and _num(r['pitch_deg']) is not None \
            and _num(r['roll_deg']) is not None:
        return _num(r['pitch_deg']), _num(r['roll_deg']), 'npz'
    return None


@dataclass
class ScanRow:
    has_jpg: bool
    width: float
    height: float
    pose: tuple         # (pitch, roll, source) or None
    alt_pose: tuple     # the npz (pitch, roll) when `pose` is the XML's, else None


def read_scan(pose_path):
    scan = {}
    with gzip.open(pose_path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pose = pose_from_scan_row(r)
            npz = npz_pose_of_scan_row(r) if pose and pose[2] == 'xml' else None
            scan[(r['city'], r['pano_id'])] = ScanRow(
                r.get('jpg_present') == '1', _num(r['jpg_width']), _num(r['jpg_height']),
                pose, npz[:2] if npz else None)
    return scan


def pool_rows(pool_path, scan, label_type, counts):
    """Yield (row, key, w, h) for every pool label with a scan row, a JPEG, a pose and a
    stored pano size matching the JPEG's; count every other label by its first reason."""
    with gzip.open(pool_path, 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['label_type'] != label_type or r['pano_source'] != 'gsv':
                counts['other_type_or_source'] += 1
                continue
            key = (r['city'], r['pano_id'])
            w, h = _num(r['pano_width']), _num(r['pano_height'])
            s = scan.get(key)
            if s is None:
                counts['no_pose_row'] += 1
            elif not s.has_jpg:
                counts['no_jpg'] += 1
            elif s.pose is None:
                counts['no_pose'] += 1
            elif not (w and h) or (w, h) != (s.width, s.height):
                counts['dims_differ'] += 1
            else:
                yield r, key, w, h


def _pool_counts():
    return {'other_type_or_source': 0, 'no_pose_row': 0, 'no_jpg': 0, 'no_pose': 0,
            'dims_differ': 0, 'not_detected': 0}


def pool_ids(pool_path, pose_path, label_type):
    """The sorted (city, pano_id) list the detector has to run over: every pano holding a
    pool label that load_pool would keep, given a detection."""
    scan = read_scan(pose_path)
    return sorted({key for _, key, _, _ in pool_rows(pool_path, scan, label_type,
                                                     _pool_counts())})


def load_pool(pool_path, pose_path, detections_path, label_type):
    """(labels, panos, counts) for sidewalk-panorama-tools' vouched pool.

    A label is dropped, and counted by the first reason that applies, when its pano has
    no row in the pose scan, no JPEG on the store, no pose (pose_from_scan_row), a JPEG
    whose size differs from the label's stored pano size (the label was made on other
    pixels), or no line in the detections file.
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
    scan = read_scan(pose_path)
    panos = {}
    for key, s in scan.items():
        if key in dets and s.has_jpg and s.pose is not None:
            dw, dh, d = dets[key]
            if (s.width, s.height) != (float(dw), float(dh)):
                raise SystemExit(f'{key}: the detections file saw a {dw}x{dh} JPEG but the '
                                 f'pose scan recorded {s.width}x{s.height} -- different pixels')
            panos[key] = Pano(s.pose[0], s.pose[1], dw, dh, s.pose[2], _masked(d), s.alt_pose)
    counts = _pool_counts()
    labels = []
    for r, key, w, h in pool_rows(pool_path, scan, label_type, counts):
        if key not in panos:
            counts['not_detected'] += 1
            continue
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
    alt_t: tuple = None  # (t_pitch, t_roll) under the pano's alt_pose

    @property
    def t(self):
        return self.t_pitch + self.t_roll


def pair_labels(labels, panos, tier, window_elevation_deg,
                window_bearing_deg=WINDOW_BEARING_DEG, centre='zero'):
    """([Pair], counts). One pair per label at most; a detection may serve two labels
    (two people marking one ramp are two observations of the same offset).

    centre='zero' takes detections with |diff| <= window, 'tilt' those with
    |diff - T| <= window, T under the pano's primary pose.
    """
    if centre not in CENTRES:
        raise ValueError(f'centre must be one of {CENTRES}')
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
        tp, tr = tilt_terms(pano.pitch, pano.roll, lab.x)
        mid = tp + tr if centre == 'tilt' else 0.0
        best = None
        for x, y, c in pano.detections:
            if c < tier:
                continue
            dx = abs(x - lab.x) % 1.0
            dx = min(dx, 1.0 - dx) * 360.0
            off = (lab.y - y) * 180.0 - mid
            if dx <= window_bearing_deg and abs(off) <= window_elevation_deg:
                d = math.hypot(dx, off)
                if best is None or d < best[0]:
                    best = (d, y, c)
        if best is None:
            counts['no_detection_in_window'] += 1
            continue
        alt = tilt_terms(*pano.alt_pose, lab.x) if pano.alt_pose else None
        out.append(Pair(lab.uid, lab.key, lab.era, lab.vouched, pano.pose_source,
                        tp, tr, (lab.y - best[1]) * 180.0, best[2], alt))
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


def under_alt_pose(pairs):
    """The pairs that carry a second pose, re-expressed under it (same detections and
    labels, so the same diffs; only T changes)."""
    return [dataclasses.replace(p, t_pitch=p.alt_t[0], t_roll=p.alt_t[1], pose_source='npz')
            for p in pairs if p.alt_t is not None]


def groups_of(pairs, mode, centre='zero'):
    """[(name, pairs)]: the strata one report reads side by side.

    The same-pairs pose comparison is only formed for the zero-centred window, where the
    pairing does not depend on the pose, so both arms really are the same pairs.
    """
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
    if centre == 'zero':
        both = [p for p in pairs if p.alt_t is not None]
        if both:
            out.append(('same pairs, xml pose', both))
            out.append(('same pairs, npz pose', under_alt_pose(both)))
    out.append(('abs T >= 3 deg', [p for p in pairs if abs(p.t) >= 3.0]))
    return out


# --------------------------------------------------------------------------- output

def _f(v, nd=3):
    return '' if v is None else f'{v:.{nd}f}'


def analyse(labels, panos, mode):
    fits, bin_rows, headline_pairs, pair_counts = [], [], None, {}
    for centre in CENTRES:
        for tier in TIERS:
            for win in WINDOWS_ELEVATION_DEG:
                pairs, counts = pair_labels(labels, panos, tier, win, centre=centre)
                pair_counts[(centre, tier, win)] = counts
                head = (centre == 'zero' and tier == BENCHMARK_CONFIDENCE
                        and win == HEADLINE_WINDOW_DEG)
                if head:
                    headline_pairs = pairs
                for name, sel in groups_of(pairs, mode, centre):
                    row = fit(sel)
                    if row is not None:
                        fits.append({'centre': centre, 'tier': tier, 'window_deg': win,
                                     'group': name, **row})
                    if head:
                        for b in bins(sel):
                            bin_rows.append({'group': name, **b})
    return fits, bin_rows, headline_pairs or [], pair_counts


def _row(fits, centre, tier, win, group):
    for r in fits:
        if (r['centre'], r['tier'], r['window_deg'], r['group']) == (centre, tier, win, group):
            return r
    return None


def _cell(r):
    return '' if r is None else f"{_f(r['beta'])} ({_f(r['beta_se'])})"


def write_outputs(out_dir, title, inputs, load_counts, labels, panos, mode,
                  provenance=(), notes=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    fits, bin_rows, pairs, pair_counts = analyse(labels, panos, mode)

    fit_cols = ['centre', 'tier', 'window_deg', 'group', 'pairs', 'panos', 'beta', 'beta_se',
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

    tier = BENCHMARK_CONFIDENCE
    L = [f'# {title}: label-frame beta (issue #113)', '',
         'How far a human label\'s `pano_y` sits from the image\'s own frame, as a fraction '
         'of the rig tilt at the label\'s bearing. beta = 0: one frame; beta = 1: off by '
         'the whole tilt. **This is not a placement coefficient** (that is #116).', '',
         'Generated by `scripts/label_frame_beta.py`; the command is at the top of that '
         'file.', '', '## Inputs', '']
    L += [f'- {line}' for line in inputs]
    n_det = sum(1 for p in panos.values() if p.detections)
    L += [f'- labels kept: {len(labels)}; dropped while loading: '
          + ', '.join(f'{k} {v}' for k, v in load_counts.items() if k != 'features'),
          f'- panos with a pose: {len(panos)} ({n_det} with at least one detection '
          f'outside the rig mask)', '']
    if provenance:
        L += ['## Detector provenance', ''] + [f'- {line}' for line in provenance] + ['']
    head = [r for r in fits if r['centre'] == 'zero' and r['tier'] == tier
            and r['window_deg'] == HEADLINE_WINDOW_DEG]
    L += [f'## Headline (tier {tier}, window {WINDOW_BEARING_DEG:g} x '
          f'{HEADLINE_WINDOW_DEG:g} deg, centred on 0)', '',
          '| group | pairs | panos | beta (SE) | beta pitch (SE) | beta roll (SE) | '
          'intercept | abs T p50 / p90 / max | pairs at abs T >= 3 |',
          '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in head:
        L.append(f"| {r['group']} | {r['pairs']} | {r['panos']} | {_f(r['beta'])} "
                 f"({_f(r['beta_se'])}) | {_f(r['beta_pitch'])} ({_f(r['beta_pitch_se'])}) | "
                 f"{_f(r['beta_roll'])} ({_f(r['beta_roll_se'])}) | {_f(r['intercept'], 2)} | "
                 f"{_f(r['abs_t_p50'], 2)} / {_f(r['abs_t_p90'], 2)} / "
                 f"{_f(r['abs_t_max'], 2)} | {r['n_abs_t_ge_3']} |")
    c = pair_counts[('zero', tier, HEADLINE_WINDOW_DEG)]
    L += ['', f"Of {c['labels']} labels: {c['pano_not_available']} on a pano with no pose "
          f"or detections, {c['dims_differ']} whose stored pano size differs from the "
          f"image's, {c['no_detection_in_window']} with no detection in the window.", '']

    groups = [g for g, _ in groups_of(pairs, mode, 'zero')
              if not g.startswith('same pairs')]
    L += ['## The window brackets beta', '',
          'Centred on 0, the elevation window cuts off the largest shifts and biases beta '
          'toward 0; centred on the predicted shift T it cuts off the other side and biases '
          'it toward 1. Where the two columns meet, the window no longer matters. '
          f'Tier {tier}.', '',
          '| group | ' + ' | '.join(f'{w:g} deg: on 0 / on T' for w in WINDOWS_ELEVATION_DEG)
          + ' |', '|---|' + '---:|' * len(WINDOWS_ELEVATION_DEG)]
    for g in groups:
        cells = [f"{_cell(_row(fits, 'zero', tier, w, g))} / "
                 f"{_cell(_row(fits, 'tilt', tier, w, g))}" for w in WINDOWS_ELEVATION_DEG]
        L.append(f'| {g} | ' + ' | '.join(cells) + ' |')
    L.append('')

    if any(r['group'].startswith('same pairs') for r in fits):
        L += ['## Same pairs, two poses', '',
              'Panos that carry both a legacy XML pose and a modern npz pose, fitted on the '
              'SAME label-detection pairs under each (window centred on 0, so the pairing '
              'does not depend on the pose). The diffs are identical; only T differs. Where '
              'the npz pose reads higher, it predicts the offset better, so a lower beta '
              'in the old eras belongs to the pose record, not to when the label was made '
              '(era and pose source are nearly the same variable in this pool). Whether '
              'that is error in the XML pose or the viewer having used a pose closer to '
              f'the npz one is not separated here. Tier {tier}.', '',
              '| window (deg) | pairs | panos | beta, xml pose (SE) | beta, npz pose (SE) | '
              'pitch xml / npz | roll xml / npz |', '|---:|---:|---:|---:|---:|---|---|']
        for w in WINDOWS_ELEVATION_DEG:
            a = _row(fits, 'zero', tier, w, 'same pairs, xml pose')
            b = _row(fits, 'zero', tier, w, 'same pairs, npz pose')
            if a and b:
                L.append(f"| {w:g} | {a['pairs']} | {a['panos']} | {_cell(a)} | {_cell(b)} | "
                         f"{_f(a['beta_pitch'])} / {_f(b['beta_pitch'])} | "
                         f"{_f(a['beta_roll'])} / {_f(b['beta_roll'])} |")
        L.append('')

    L += ['## Every tier and window (group `all`)', '',
          '| centre | tier | window (deg) | pairs | beta (SE) | beta pitch (SE) | '
          'beta roll (SE) |', '|---|---|---:|---:|---:|---:|---:|']
    for r in fits:
        if r['group'] == 'all':
            L.append(f"| {r['centre']} | {r['tier']} | {r['window_deg']:g} | {r['pairs']} | "
                     f"{_f(r['beta'])} ({_f(r['beta_se'])}) | {_f(r['beta_pitch'])} "
                     f"({_f(r['beta_pitch_se'])}) | {_f(r['beta_roll'])} "
                     f"({_f(r['beta_roll_se'])}) |")
    L += ['', '## Median offset by signed tilt (headline pairs, group `all`)', '',
          '| T from | T to | n | median T | median (el_detection - el_human) |',
          '|---:|---:|---:|---:|---:|']
    for b in bin_rows:
        if b['group'] == 'all':
            L.append(f"| {b['t_from']:g} | {b['t_to']:g} | {b['n']} | "
                     f"{_f(b['median_t'], 2)} | {_f(b['median_diff'], 2)} |")
    L += ['', '## Reading it', '',
          '- Standard errors are clustered by pano: labels on one pano share its pose.',
          '- Every beta here is a lower bound on the viewer\'s true coefficient: any error '
          'in the pose (the regressor) attenuates the slope, and no pose here is free of '
          'error.',
          '- A wrong pairing (another ramp at that bearing) is noise, not bias; the window '
          'is a bias, which is what the bracket table is for.',
          '- The intercept is where a person clicks on a ramp against where the detector '
          'peaks. It is not a tilt effect.',
          '- Detections are rig-masked, as production ships them.',
          '- `pairs.csv` holds the headline pairs, keyed on `label_uid`.']
    L += [f'- {n}' for n in notes]
    L.append('')
    (out_dir / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    return fits


def pinned_input_line(what, path):
    sha = ar.sha256_file(path)
    pin = POOL_INPUTS.get(path.name)
    if pin and pin[1] == sha:
        where = f'pinned: {pin[0]}'
    elif pin:
        where = f'**differs from the pinned file** ({pin[0]}, sha256 `{pin[1]}`)'
    else:
        where = 'unpinned'
    return f'{what}: `{path.name}` sha256 `{sha}` ({where})'


def provenance_lines(detections_path, ids_path=None):
    """Lines for the report from the detections file's <name>.meta.json, if present."""
    meta_path = detections_path.with_name(detections_path.name + '.meta.json')
    if not meta_path.exists():
        return [f'no `{meta_path.name}` beside the detections file: model unknown']
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    p = meta.get('provenance', {})
    out = [f"model: `{p.get('model_repo')}` revision `{p.get('model_revision')}` "
           f"(`{p.get('model_id')}`, trained {p.get('model_training_date')})",
           f"store: `{meta.get('store')}`",
           f"labeler code: commit `{meta.get('labeler_commit', 'not recorded')}`"
           + (f" ({meta['labeler_commit_source']})" if meta.get('labeler_commit_source')
              else '')]
    if meta.get('ids_sha256'):
        line = f"pano ids: sha256 `{meta['ids_sha256']}`"
        if ids_path is not None and ids_path.exists():
            ok = ar.sha256_file(ids_path) == meta['ids_sha256']
            line += (f' = `{ids_path.name}`' if ok
                     else f' **differs from `{ids_path.name}`**')
        out.append(line)
    if meta.get('started_utc'):
        out.append(f"run: {meta['started_utc']} to {meta.get('finished_utc', '?')}")
    return out


# ------------------------------------------------------------------------ detection

def _git_commit():
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'detectors',
                                'panorama.py', 'scripts/label_frame_beta.py'], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def detect_pool(ids_path, store, out_path, workers=6):
    """Run the detector over the store JPEGs of every (city, pano_id) in `ids_path`.

    Pixels only, from `<store>/<city>/<id[:2]>/<id>.jpg`: the pose comes from the pose
    scan, so no metadata is fetched and nothing is written into a run directory. Same
    preprocessing (panorama.normalize_image) and detector as production. Resumable: ids
    already in the output are skipped. A missing or corrupt JPEG is written as an
    `error` line and never stops the pass.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    import panorama
    from detectors.curb_ramp import CurbRampDetector

    Image.MAX_IMAGE_PIXELS = None   # 16384x8192 store JPEGs trip PIL's bomb guard
    todo = [tuple(line.rstrip('\n').split('\t'))
            for line in open(ids_path, encoding='utf-8') if line.strip()]
    done = set()
    if out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                done.add((r['city'], r['pano_id']))
    todo = [t for t in todo if t not in done]
    print(f'{len(done)} done, {len(todo)} to do', flush=True)

    det = CurbRampDetector()
    meta_path = out_path.with_name(out_path.name + '.meta.json')
    meta = {'provenance': det.provenance, 'store': str(store),
            'labeler_commit': _git_commit(), 'ids_sha256': ar.sha256_file(ids_path),
            'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        if done and (old.get('provenance') != meta['provenance']
                     or old.get('ids_sha256') != meta['ids_sha256']):
            sys.exit(f'{out_path}: resuming under another model or id list than '
                     f'{meta_path.name} records -- start a new output')
        meta['started_utc'] = old.get('started_utc', meta['started_utc'])
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    lock = threading.Lock()
    n = [0, 0]
    t0 = time.time()

    def one(item):
        city, pid = item
        rec = {'city': city, 'pano_id': pid}
        try:
            with Image.open(Path(store) / city / pid[:2] / f'{pid}.jpg') as im:
                rec['native_w'], rec['native_h'] = im.size
                image = panorama.normalize_image(im.convert('RGB'))
            rec['detections'] = [[round(x, 6), round(y, 6), round(c, 6)]
                                 for x, y, c in det.detect(image)]
        except Exception as e:   # noqa: BLE001 -- recorded on the line, never fatal
            rec['error'] = f'{type(e).__name__}: {e}'
        with lock:
            out.write(json.dumps(rec) + '\n')
            out.flush()
            n[0] += 1
            n[1] += 'error' in rec
            if n[0] % 200 == 0:
                print(f'{n[0]}/{len(todo)}  errors {n[1]}  '
                      f'{n[0] / (time.time() - t0):.2f} panos/s', flush=True)

    with open(out_path, 'a', encoding='utf-8') as out, ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    det.close()
    meta['finished_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'finished: {n[0]} written, {n[1]} errors, {time.time() - t0:.0f} s', flush=True)


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
    r.add_argument('--allow-self-pairs', action='store_true',
                   help='run although labels sit on their own detections (checked by hand)')
    r.add_argument('--refresh', action='store_true', help='re-pull the labels')
    r.add_argument('--out', type=Path)
    i = sub.add_parser('pool-ids', help='the pano ids pool-detect has to run over')
    i.add_argument('--pool', type=Path, required=True)
    i.add_argument('--pose', type=Path, required=True)
    i.add_argument('--label-type', default='CurbRamp')
    i.add_argument('--out', type=Path, required=True)
    d = sub.add_parser('pool-detect', help='run the detector over the store panos (GPU)')
    d.add_argument('--ids', type=Path, required=True)
    d.add_argument('--store', type=Path, required=True)
    d.add_argument('--out', type=Path, required=True)
    d.add_argument('--workers', type=int, default=6)
    p = sub.add_parser('pool', help="sidewalk-panorama-tools' vouched pool")
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--pose', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--ids', type=Path, help='the pool-ids file, checked against the '
                   'detections meta (default: pool_ids.txt beside the detections)')
    p.add_argument('--label-type', default='CurbRamp')
    p.add_argument('--out', type=Path)
    args = ap.parse_args(argv)

    if args.mode == 'pool-ids':
        ids = pool_ids(args.pool, args.pose, args.label_type)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(f'{c}\t{pid}\n' for c, pid in ids)
        print(f'{len(ids)} panos -> {args.out} (sha256 {ar.sha256_file(args.out)})')
        return
    if args.mode == 'pool-detect':
        detect_pool(args.ids, args.store, args.out, args.workers)
        return

    notes, provenance = [], []
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
        n_self = self_pairs(labels, panos)
        share = n_self / max(len(labels), 1)
        if share > SELF_PAIR_MAX_SHARE and not args.allow_self_pairs:
            sys.exit(f'{n_self} of {len(labels)} labels ({share:.1%}) sit within '
                     f'{SELF_PAIR_PX:g} px of a stored detection: an AI account\'s labels '
                     f'pair with their own detections at diff 0. Name it with '
                     f'--exclude-user, or pass --allow-self-pairs after checking.')
        inputs = [f'labels: `{name}`, {n} features, fetched {fetched} from {url}, '
                  f'sha256 `{sha}`',
                  f'run: `results.jsonl` sha256 `{ar.sha256_file(results)}`',
                  f"excluded users: {', '.join(args.exclude_user) or 'none'}"]
        notes.append(f'{n_self} of {len(labels)} labels sit within {SELF_PAIR_PX:g} px of a '
                     f'stored detection on both axes (an AI label would; the run refuses '
                     f'above {SELF_PAIR_MAX_SHARE:.0%}).')
        title = args.city
    else:
        out = args.out or REPO / 'runs' / '_pooled' / 'label_frame_beta'
        labels, panos, counts = load_pool(args.pool, args.pose, args.detections,
                                          args.label_type)
        inputs = [pinned_input_line('pool', args.pool), pinned_input_line('pose scan', args.pose),
                  f'detections: `{args.detections.name}` sha256 '
                  f'`{ar.sha256_file(args.detections)}`']
        provenance = provenance_lines(
            args.detections, args.ids or args.detections.with_name('pool_ids.txt'))
        title = f'vouched pool ({args.label_type})'
    fits = write_outputs(out, title, inputs, counts, labels, panos, args.mode,
                         provenance, notes)
    for row in fits:
        if (row['centre'], row['tier'], row['window_deg']) == \
                ('zero', BENCHMARK_CONFIDENCE, HEADLINE_WINDOW_DEG):
            print(f"{row['group']:>22}: pairs {row['pairs']:6d}  beta {row['beta']:.3f} "
                  f"(se {row['beta_se']:.3f})")
    print(f'wrote {out / "report.md"}')


if __name__ == '__main__':
    main()
