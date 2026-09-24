"""Mapillary rig-tilt study (issue #42): parse OpenSfM's ``computed_rotation`` into a
camera pose, lock its convention against independent evidence, and measure what
applying it does to the ground raycast.

Every Mapillary record already carries the rotation in ``pano.source_metadata``, so
nothing here touches the network or the GPU: the inputs are ``runs/<city>/results.jsonl``,
RampNet's benchmark bundles (verdicts + native-res panos) and nothing else.

Conventions (all verified in ``tests/test_mapillary_tilt.py`` and by the ``stats``
subcommand's compass identity):

- ``computed_rotation`` is an axis-angle (Rodrigues) vector for the WORLD->CAMERA
  rotation, world = topocentric ENU (x east, y north, z up), camera = OpenCV
  (x right, y down, z forward). This is OpenSfM's documented convention; Mapillary's
  own ``computed_compass_angle`` equals the yaw of that matrix to 1e-11 degrees on
  every record of every run, which pins the frame assignment independently of us.
- The equirectangular pixel -> camera-frame bearing is OpenSfM's spherical model,
  which coincides with geo.py's: column x=0.5 is camera-forward, row y=0.5 is the
  camera-frame horizon, phi=(x-0.5)*2pi, theta=(0.5-y)*pi.
- ``opensfm_pose`` decomposes the matrix into (heading, pitch, roll) in exactly the
  convention ``geo._world_ray`` composes (yaw -> pitch about right, +up -> roll about
  forward, + lowers the camera's right axis), so ``geo.detection_ground_point(...,
  apply_pose=True)`` reproduces the direct R^T*bearing raycast to machine precision.
  That roll sign is Project Sidewalk's (``MapillaryViewer.extractPitchRoll``: positive =
  camera rolled clockwise as seen by the photographer). Until the #42 wiring the code
  used the opposite sign and carried PS's as a separate ``roll_ps_deg``; the two were
  unified by flipping ``geo._world_ray``, so every roll computed here now has PS's sign
  -- and the roll columns of the per-pano and summary CSVs committed with PR #50
  (``tilt_summary.csv``, ``tilt_by_rig.csv``, ``tilt_by_sequence.csv``,
  ``verticality.csv``) have the other. ``ablation`` and ``eval`` are sign-free (every
  convention is a product of signs with the same angles) and reproduce cell for cell.

Subcommands. Per-panorama CSVs go to ``runs/<city>/tilt/`` and the aggregated ones to
``runs/_summary/tilt/`` (unless --out is given); the aggregated ones are also committed
under ``docs/figures/mapillary-tilt/data/``, which ``figures`` falls back to:

    python scripts/mapillary_tilt.py stats richmond clovis morgantown annapolis laurens
    python scripts/mapillary_tilt.py displacement richmond ... # flat vs corrected point
    python scripts/mapillary_tilt.py grade richmond ...        # rig tilt, or road grade?
    python scripts/mapillary_tilt.py horizon richmond ...      # GT marks above the horizon
    python scripts/mapillary_tilt.py ablation richmond ...     # multi-view sign lock
    python scripts/mapillary_tilt.py eval richmond ...         # world P/R vs RampNet GT
    python scripts/mapillary_tilt.py rectify richmond ...      # pixel-level sign lock
    python scripts/mapillary_tilt.py examples richmond ...     # before/after image strips
    python scripts/mapillary_tilt.py figures                   # docs/figures/mapillary-tilt
    python scripts/mapillary_tilt.py pose <pano_id> [--run richmond]

Two measurement traps, handled in ``cmd_ablation`` and ``cmd_eval`` and documented there:
every convention must be scored on the SAME sites, and the GT recall denominator moves
with the convention because GT marks are raycast under the pose being tested.

Needs numpy + Pillow for ``rectify`` and matplotlib for ``figures``; everything else
is stdlib on top of geo.py / fuse_sites.py / eval_sites.py.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same order tests/conftest.py forces, for the same reason: the repo root must come
# FIRST, because scripts/position_check.py is a shim sharing the root module's name.
for p in (str(REPO_ROOT / 'scripts'), str(REPO_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

# Benchmark split name -> run directory name, where they differ.
BENCHMARK_OF = {'laurens': 'laurens_mapillary'}
DEFAULT_CITIES = ['richmond', 'clovis', 'morgantown', 'annapolis', 'laurens']
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'mapillary-tilt'

# Sign conventions tested everywhere: multipliers on (pitch, roll) relative to the
# documented decomposition. 'off' is the flat raycast production runs today.
CONVENTIONS = [('off', None), ('documented', (1, 1)), ('pitch-flipped', (-1, 1)),
               ('roll-flipped', (1, -1)), ('both-flipped', (-1, -1)),
               ('pitch-only', (1, 0)), ('roll-only', (0, 1)),
               ('road-relative', 'road')]


def pose_angles(pose, signs):
    """(camera_pitch, camera_roll) to feed geo.pano_pose under a convention. None =
    production's null pose. 'road' subtracts the road grade along the direction of
    travel (from the sequence's SfM altitude profile) so the tilt is measured
    against the local ground plane rather than gravity; where no grade is available
    it falls back to the documented gravity-relative angles.

    That fallback is not free: it is the gravity-relative convention, which section
    5.3 of the report measures as the WRONG one for a vehicle rig on a grade. It
    fires on the frames with no usable sequence neighbour -- 0.3% (morgantown,
    laurens) to 7.3% (richmond) of panos; ``grade`` reports the rate per city as
    ``frac_with_grade``."""
    if signs is None:
        return None, None
    if signs == 'road':
        g = pose.get('grade_deg')
        if g is None:
            return pose['pitch_deg'], pose['roll_deg']
        # The production formula (fuse_sites' `road` mode applies the same one).
        return geo.road_relative_pitch_roll(pose['pitch_deg'], pose['roll_deg'],
                                            pose['heading_deg'], g,
                                            pose['travel_bearing_deg'])
    return signs[0] * pose['pitch_deg'], signs[1] * pose['roll_deg']
# Open-ended on purpose: the top bucket used to stop at 90 deg, so the handful of
# panos with a failed reconstruction (clovis reaches 170 deg) fell into a label no
# consumer listed and vanished from every by-bucket table without a count.
TILT_BUCKETS = [(0, 1.5), (1.5, 3), (3, 5), (5, 10), (10, math.inf)]


# --- Rotation math ------------------------------------------------------------------

# rotation_matrix and opensfm_pose moved to geo.py with the #42 production wiring (one
# decomposition in the tree, shared with sources/mapillary.py); re-exported here because
# every subcommand and tests/test_mapillary_tilt.py use them by these names.
rotation_matrix = geo.rotation_matrix
opensfm_pose = geo.opensfm_pose


def matrix_from_pose(heading_deg, pitch_deg, roll_deg):
    """Inverse of opensfm_pose: the world->camera matrix (rows right/down/forward in
    ENU) that geo._world_ray's yaw->pitch->roll composition describes."""
    psi, alpha, rho = (math.radians(a) for a in (heading_deg, pitch_deg, roll_deg))
    rho = -rho   # PS-sign roll in; the rotation below lifts the right axis for rho > 0
    # geo._world_ray works in (north, east, up); build there, then reorder to ENU.
    f = (math.cos(psi), math.sin(psi), 0.0)
    r = (-math.sin(psi), math.cos(psi), 0.0)
    u = (0.0, 0.0, 1.0)
    ca, sa = math.cos(alpha), math.sin(alpha)
    f, u = tuple(ca * fi + sa * ui for fi, ui in zip(f, u)), \
        tuple(-sa * fi + ca * ui for fi, ui in zip(f, u))
    cr, sr = math.cos(rho), math.sin(rho)
    r, u = tuple(cr * ri + sr * ui for ri, ui in zip(r, u)), \
        tuple(-sr * ri + cr * ui for ri, ui in zip(r, u))
    enu = lambda v: [v[1], v[0], v[2]]  # noqa: E731
    return [enu(r), [-c for c in enu(u)], enu(f)]


def world_direction(R, x_norm, y_norm):
    """(elevation_rad, bearing_rad) of an equirect pixel straight from the matrix:
    camera bearing (OpenSfM spherical model) rotated by R^T into ENU."""
    phi = (x_norm - 0.5) * 2.0 * math.pi
    theta = (0.5 - y_norm) * math.pi
    d = (math.cos(theta) * math.sin(phi), -math.sin(theta), math.cos(theta) * math.cos(phi))
    e = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    n = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    u = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    return math.asin(max(-1.0, min(1.0, u))), math.atan2(e, n)


# --- Loading ------------------------------------------------------------------------

def load_run(city, floor=None):
    """SlimPanos (pitch/roll left None, exactly as production) plus a per-pano dict of
    the parsed pose and provenance. Records without a rotation are counted."""
    path = REPO_ROOT / 'runs' / city / 'results.jsonl'
    panos, poses, no_rotation = [], {}, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec['pano']
            sm = p.get('source_metadata') or {}
            rvec = sm.get('computed_rotation')
            if rvec is None:
                no_rotation += 1
                continue
            pose = opensfm_pose(rvec)
            pose.update({
                'pano_id': p['panorama_id'], 'make': p.get('camera_make'),
                'model': p.get('camera_model'), 'sequence': p.get('sequence_id'),
                'capture_date': p.get('capture_date'),
                'camera_heading': p['camera_heading'],
                'compass_angle': sm.get('compass_angle'),
                'computed_compass_angle': sm.get('computed_compass_angle'),
                'rvec': rvec, 'width': p.get('width'), 'height': p.get('height'),
                'captured_at': sm.get('captured_at'),
                'computed_altitude': sm.get('computed_altitude'),
                'lat': p['lat'], 'lng': p['lng'],
                'n_operational': sum(1 for d in rec.get('detections', [])
                                     if d['confidence'] >= BENCHMARK_CONFIDENCE),
            })
            poses[p['panorama_id']] = pose
            panos.append(fs.SlimPano(
                pano_id=p['panorama_id'], lat=p['lat'], lng=p['lng'],
                camera_heading=p['camera_heading'], camera_pitch=None, camera_roll=None,
                capture_date=p.get('capture_date'), source=p.get('source') or '',
                detections=[(i, d['x_normalized'], d['y_normalized'], d['confidence'])
                            for i, d in enumerate(rec.get('detections', []))
                            if floor is None or d['confidence'] >= floor]))
    add_sequence_grade(poses)
    return panos, poses, no_rotation


def add_sequence_grade(poses):
    """Per pano: the road grade along the direction of travel (from the sequence's
    SfM altitude profile) and the camera's gravity-relative pitch along that same
    direction, so rig tilt can be separated from road grade. Fields stay None when
    the sequence gives no usable neighbour.

    The grade itself is fuse_sites.sequence_grades -- production's, since the #42
    wiring -- so what this study measured is what fusion's `road` mode applies."""
    by_seq = defaultdict(list)
    for pose in poses.values():
        pose.update({'grade_deg': None, 'travel_pitch_deg': None, 'travel_bearing_deg': None,
                     'pitch_rel_road_deg': None, 'lag1_pitch_diff': None, 'lag1_roll_diff': None})
        if pose['captured_at'] is not None:
            by_seq[pose['sequence']].append(pose)
    for ps in by_seq.values():
        ps.sort(key=lambda q: q['captured_at'])
        for i, q in enumerate(ps[:-1]):
            # lag-1 differences of the tilt signal itself (smoothness test)
            nxt = ps[i + 1]
            if (nxt['captured_at'] - q['captured_at']) / 1000.0 <= fs.GRADE_MAX_GAP_S:
                q['lag1_pitch_diff'] = nxt['pitch_deg'] - q['pitch_deg']
                q['lag1_roll_diff'] = nxt['roll_deg'] - q['roll_deg']
    grades = fs.sequence_grades(
        (pid, q['sequence'], q['captured_at'], q['lat'], q['lng'], q['computed_altitude'])
        for pid, q in poses.items())
    for pid, (grade, bearing) in grades.items():
        q = poses[pid]
        phi = math.radians(geo.norm_deg(bearing - q['heading_deg']))
        # PS-sign roll: + lowers the right axis, so it enters with a minus.
        travel_pitch = q['pitch_deg'] * math.cos(phi) - q['roll_deg'] * math.sin(phi)
        q.update({'grade_deg': grade, 'travel_bearing_deg': bearing,
                  'travel_pitch_deg': travel_pitch,
                  'pitch_rel_road_deg': travel_pitch - grade})


def with_convention(panos, poses, signs):
    """Copies of the SlimPanos carrying pitch/roll under a sign convention (None =
    production's null pose)."""
    if signs is None:
        return panos
    out = []
    for p in panos:
        cp, cr = pose_angles(poses[p.pano_id], signs)
        out.append(replace(p, camera_pitch=cp, camera_roll=cr))
    return out


def out_dir_for(city, out):
    d = (out / city) if out else REPO_ROOT / 'runs' / city / 'tilt'
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round((len(v) - 1) * q)))]


def bucket_label(lo, hi):
    return f'{lo}+' if hi == math.inf else f'{lo}-{hi}'


BUCKET_LABELS = [bucket_label(lo, hi) for lo, hi in TILT_BUCKETS]


def bucket_of(tilt):
    for lo, hi in TILT_BUCKETS:
        if lo <= tilt < hi:
            return bucket_label(lo, hi)
    return BUCKET_LABELS[-1]     # unreachable: the last bucket is open-ended


# --- stats ----------------------------------------------------------------------------

def cmd_stats(args):
    summary_rows, rig_rows, seq_rows = [], [], []
    for city in args.cities:
        panos, poses, no_rot = load_run(city)
        out = out_dir_for(city, args.out)
        rows = []
        for p in panos:
            pose = poses[p.pano_id]
            cd = None
            if pose['compass_angle'] is not None and pose['computed_compass_angle'] is not None:
                cd = abs(geo.norm_deg(pose['compass_angle'] - pose['computed_compass_angle']))
            rows.append({k: pose[k] for k in ('pano_id', 'make', 'model', 'sequence',
                                               'capture_date', 'captured_at', 'heading_deg',
                                               'pitch_deg', 'roll_deg', 'tilt_deg',
                                               'n_operational', 'width', 'height',
                                               'computed_altitude', 'grade_deg', 'travel_pitch_deg',
                                               'pitch_rel_road_deg', 'lag1_pitch_diff',
                                               'lag1_roll_diff')}
                        | {'compass_disagreement_deg': cd,
                           'yaw_vs_computed_compass_deg':
                               abs(geo.norm_deg(pose['heading_deg'] - pose['computed_compass_angle']))
                               if pose['computed_compass_angle'] is not None else None})
        write_csv(out / 'poses.csv', rows)

        if not rows:
            print(f'{city}: no record carries computed_rotation — nothing to report')
            continue
        tilts = [r['tilt_deg'] for r in rows]
        tilts_w = [r['tilt_deg'] for r in rows for _ in range(r['n_operational'])]
        yaws = [r['yaw_vs_computed_compass_deg'] for r in rows
                if r['yaw_vs_computed_compass_deg'] is not None]
        if not yaws:
            print(f'{city}: no record carries computed_compass_angle — the compass '
                  f'identity cannot be checked on this run')
        yaw_err = max(yaws) if yaws else None
        cds = [r['compass_disagreement_deg'] for r in rows if r['compass_disagreement_deg'] is not None]
        summary_rows.append({
            'city': city, 'n_panos': len(rows), 'n_without_rotation': no_rot,
            'n_sequences': len({r['sequence'] for r in rows}),
            'yaw_vs_computed_compass_max_deg': yaw_err,
            'tilt_p50': pct(tilts, .5), 'tilt_p90': pct(tilts, .9), 'tilt_p99': pct(tilts, .99),
            'tilt_max': max(tilts),
            'frac_tilt_gt_1p5': sum(t > 1.5 for t in tilts) / len(tilts),
            'frac_tilt_gt_3': sum(t > 3 for t in tilts) / len(tilts),
            'frac_tilt_gt_5': sum(t > 5 for t in tilts) / len(tilts),
            'frac_tilt_gt_10': sum(t > 10 for t in tilts) / len(tilts),
            'tilt_p50_label_weighted': pct(tilts_w, .5),
            'tilt_p90_label_weighted': pct(tilts_w, .9),
            'pitch_p05': pct([r['pitch_deg'] for r in rows], .05),
            'pitch_p50': pct([r['pitch_deg'] for r in rows], .5),
            'pitch_p95': pct([r['pitch_deg'] for r in rows], .95),
            'roll_p05': pct([r['roll_deg'] for r in rows], .05),
            'roll_p50': pct([r['roll_deg'] for r in rows], .5),
            'roll_p95': pct([r['roll_deg'] for r in rows], .95),
            'exif_compass_disagreement_p50': pct(cds, .5),
            'frac_exif_compass_off_gt_10deg': sum(c > 10 for c in cds) / len(cds) if cds else None,
        })
        # per rig (make/model)
        by_rig = defaultdict(list)
        for r in rows:
            by_rig[(r['make'], r['model'])].append(r)
        for (make, model), rs in sorted(by_rig.items(), key=lambda kv: -len(kv[1])):
            t = [r['tilt_deg'] for r in rs]
            rig_rows.append({'city': city, 'make': make, 'model': model, 'n_panos': len(rs),
                             'tilt_p50': pct(t, .5), 'tilt_p90': pct(t, .9), 'tilt_max': max(t),
                             'pitch_p50': pct([r['pitch_deg'] for r in rs], .5),
                             'roll_p50': pct([r['roll_deg'] for r in rs], .5),
                             'n_sequences': len({r['sequence'] for r in rs})})
        # per sequence: within-sequence spread vs the sequence mean
        by_seq = defaultdict(list)
        for r in rows:
            by_seq[r['sequence']].append(r)
        within_p, within_r, means_p, means_r = [], [], [], []
        for seq, rs in by_seq.items():
            if len(rs) < 5:
                continue
            ps_ = [r['pitch_deg'] for r in rs]
            rs_ = [r['roll_deg'] for r in rs]
            sd_p, sd_r = statistics.pstdev(ps_), statistics.pstdev(rs_)
            within_p.append(sd_p)
            within_r.append(sd_r)
            means_p.append(statistics.mean(ps_))
            means_r.append(statistics.mean(rs_))
            seq_rows.append({'city': city, 'sequence': seq, 'n_panos': len(rs),
                             'make': rs[0]['make'], 'model': rs[0]['model'],
                             'pitch_mean': statistics.mean(ps_), 'pitch_sd': sd_p,
                             'roll_mean': statistics.mean(rs_), 'roll_sd': sd_r,
                             'tilt_p50': pct([r['tilt_deg'] for r in rs], .5)})
        lag_p = [r['lag1_pitch_diff'] for r in rows if r['lag1_pitch_diff'] is not None]
        lag_r = [r['lag1_roll_diff'] for r in rows if r['lag1_roll_diff'] is not None]
        rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v)) if v else None  # noqa: E731
        summary_rows[-1].update({
            # white noise would give lag-1 RMS = sqrt(2) * SD; a smooth signal much less
            'lag1_rms_pitch': rms(lag_p), 'lag1_rms_roll': rms(lag_r),
            'lag1_over_sqrt2_sd_pitch': (rms(lag_p) / (math.sqrt(2) * pct(within_p, .5))
                                          if lag_p and within_p and pct(within_p, .5) else None),
            'n_lag1_pairs': len(lag_p),
            'seq_within_sd_pitch_p50': pct(within_p, .5),
            'seq_within_sd_roll_p50': pct(within_r, .5),
            'seq_between_sd_pitch': statistics.pstdev(means_p) if len(means_p) > 1 else None,
            'seq_between_sd_roll': statistics.pstdev(means_r) if len(means_r) > 1 else None,
            'n_sequences_ge5': len(within_p),
        })
        print(f"{city}: {len(rows)} panos ({no_rot} without rotation), yaw-vs-compass max "
              f"{'n/a' if yaw_err is None else format(yaw_err, '.1e')} deg, "
              f"tilt p50/p90/p99 {pct(tilts, .5):.2f}/{pct(tilts, .9):.2f}/"
              f"{pct(tilts, .99):.2f} deg, {summary_rows[-1]['frac_tilt_gt_1p5']:.0%} above 1.5 deg")

    agg = out_dir_for('_summary', args.out)
    write_csv(agg / 'tilt_summary.csv', summary_rows)
    write_csv(agg / 'tilt_by_rig.csv', rig_rows)
    write_csv(agg / 'tilt_by_sequence.csv', seq_rows)
    print(f'wrote {agg}')


# --- displacement: what the tilt does to each detection's ground point ---------------

def cmd_displacement(args):
    rows_all = []
    for city in args.cities:
        panos, poses, _ = load_run(city, floor=BENCHMARK_CONFIDENCE)
        rows = []
        for p in panos:
            pose = poses[p.pano_id]
            flat = geo.pano_pose({'lat': p.lat, 'lng': p.lng, 'camera_heading': p.camera_heading,
                                  'camera_pitch': None, 'camera_roll': None, 'source': p.source})
            tilted = geo.pano_pose({'lat': p.lat, 'lng': p.lng, 'camera_heading': p.camera_heading,
                                    'camera_pitch': pose['pitch_deg'],
                                    'camera_roll': pose['roll_deg'], 'source': p.source})
            errors = geo.error_model_for(p.source)
            for i, x, y, conf in p.detections:
                g0 = geo.detection_ground_point(flat, x, y, errors=errors, apply_pose=False)
                g1 = geo.detection_ground_point(tilted, x, y, errors=errors, apply_pose=True)
                row = {'city': city, 'pano_id': p.pano_id, 'det_index': i, 'confidence': conf,
                       'tilt_deg': pose['tilt_deg'], 'tilt_bucket': bucket_of(pose['tilt_deg']),
                       'range_flat_m': None if g0 is None else g0.range_m,
                       'range_tilt_m': None if g1 is None else g1.range_m,
                       'status': ('both' if g0 and g1 else 'flat_only' if g0
                                  else 'tilt_only' if g1 else 'neither')}
                if g0 and g1:
                    e0, n0 = geo.LocalFrame(p.lat, p.lng).to_enu(g0.lat, g0.lng)
                    e1, n1 = geo.LocalFrame(p.lat, p.lng).to_enu(g1.lat, g1.lng)
                    row['displacement_m'] = math.hypot(e1 - e0, n1 - n0)
                    row['range_ratio'] = g1.range_m / g0.range_m
                rows.append(row)
        write_csv(out_dir_for(city, args.out) / 'displacement.csv', rows)
        both = [r for r in rows if r['status'] == 'both']
        print(f"{city}: {len(rows)} operational detections; placeable flat {sum(r['status'] in ('both','flat_only') for r in rows)}, "
              f"tilted {sum(r['status'] in ('both','tilt_only') for r in rows)}; displacement p50 "
              f"{pct([r['displacement_m'] for r in both], .5):.2f} p90 {pct([r['displacement_m'] for r in both], .9):.2f} m")
        rows_all.extend(rows)
    # aggregate by city x tilt bucket
    agg = defaultdict(list)
    for r in rows_all:
        agg[(r['city'], r['tilt_bucket'])].append(r)
    out_rows = []
    for (city, b), rs in sorted(agg.items()):
        both = [r for r in rs if r['status'] == 'both']
        out_rows.append({'city': city, 'tilt_bucket': b, 'n_dets': len(rs),
                         'n_both': len(both),
                         'n_flat_only': sum(r['status'] == 'flat_only' for r in rs),
                         'n_tilt_only': sum(r['status'] == 'tilt_only' for r in rs),
                         'displacement_p50': pct([r['displacement_m'] for r in both], .5),
                         'displacement_p90': pct([r['displacement_m'] for r in both], .9),
                         'range_ratio_p10': pct([r['range_ratio'] for r in both], .1),
                         'range_ratio_p50': pct([r['range_ratio'] for r in both], .5),
                         'range_ratio_p90': pct([r['range_ratio'] for r in both], .9)})
    write_csv(out_dir_for('_summary', args.out) / 'displacement_by_bucket.csv', out_rows)


# --- ablation: multi-view sign lock ----------------------------------------------------

def cmd_ablation(args):
    """Freeze association from the pose-off fuse (as fuse_sites.pose_ablation_report
    does), then re-project every operational member under each sign convention and
    measure within-site pairwise member distance — raw and range-normalized — overall
    and by the pair's larger tilt.

    Every convention is scored on the SAME sites: a site is kept only if every
    convention places every one of its operational members. Without that, a
    convention that pushes members above the horizon silently drops whole sites and
    is measured on an easier subset than the others (up to 4.3% of morgantown's
    pairs); the pair count would then differ per row while reading as shared.

    The raycast runs with max_range_m=inf, deliberately: the 25 m production cap
    would drop a member (and with it the whole site) the moment a correction
    lengthens its ray, which is the same survivorship problem one step further in
    — capping costs 40-50% of the pairs, asymmetrically by convention. The price is
    that the tail of this distribution contains pairs at ranges production would
    never emit, so the raw mean and p90 here are pessimistic about any correction
    that lengthens rays. Median, mean, p90 and the range-normalized mean are all
    reported; they do not always agree, and the report says which one each claim
    rests on."""
    # Pinned to the benchmark tier, NOT the operating point: every number this study
    # published (docs/mapillary-tilt-study.md, PR #50) was produced at 0.55, and
    # FuseParams.min_confidence defaults to OPERATIONAL_CONFIDENCE, which moved to 0.30
    # with issue #20. Taking the default would silently re-key the committed CSVs.
    # mask_rig=False for the same reason as the tier: these CSVs are committed.
    params = fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE, mask_rig=False)
    all_rows = []
    for city in args.cities:
        panos, poses, _ = load_run(city)
        sites, frame, _ = fs.fuse(panos, params)
        groups = [[d for d, _ in s.members if d.operational] for s in sites]
        groups = [g for g in groups if len(g) >= 2]
        by_id = {p.pano_id: p for p in panos}

        def project(ms, signs):
            """ENU + provenance for every member, or None if any is unplaceable."""
            pts = []
            for d in ms:
                p = by_id[d.pano_id]
                pose = poses[p.pano_id]
                cp, cr = pose_angles(pose, signs)
                pp = geo.pano_pose({'lat': p.lat, 'lng': p.lng, 'camera_heading': p.camera_heading,
                                    'camera_pitch': cp, 'camera_roll': cr, 'source': p.source})
                g = geo.detection_ground_point(pp, d.x, d.y, max_range_m=math.inf,
                                               errors=geo.error_model_for(p.source),
                                               apply_pose=signs is not None)
                if g is None:
                    return None
                pts.append((frame.to_enu(g.lat, g.lng), g.range_m, pose['tilt_deg'],
                            pose['grade_deg'], pose['pitch_rel_road_deg']))
            return pts

        n_groups = len(groups)
        groups = [g for g in groups
                  if all(project(g, signs) is not None for _, signs in CONVENTIONS)]
        if not groups:
            print(f'{city}: no site survives every convention — nothing to compare')
            continue
        print(f'{city}: {len(groups)} of {n_groups} multi-member sites placeable under '
              f'every convention ({n_groups - len(groups)} dropped from all rows)')
        for name, signs in CONVENTIONS:
            dists, norm, by_bucket = [], [], defaultdict(list)
            by_grade, by_rel = defaultdict(list), defaultdict(list)
            for ms in groups:
                pts = project(ms, signs)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        (a, ra, ta, ga, rla), (b, rb, tb, gb, rlb) = pts[i], pts[j]
                        dd = math.hypot(a[0] - b[0], a[1] - b[1])
                        dists.append(dd)
                        norm.append(dd / (0.5 * (ra + rb)))
                        by_bucket[bucket_of(max(ta, tb))].append(dd)
                        if ga is not None and gb is not None:
                            by_grade[bucket_of(max(abs(ga), abs(gb)))].append(dd)
                            by_rel[bucket_of(max(abs(rla), abs(rlb)))].append(dd)
            row = {'city': city, 'convention': name, 'n_sites': len(groups),
                   'n_sites_before_intersection': n_groups,
                   'n_pairs': len(dists),
                   'mean_pair_m': statistics.mean(dists), 'median_pair_m': pct(dists, .5),
                   'p90_pair_m': pct(dists, .9),
                   'mean_pair_over_range': statistics.mean(norm)}
            for lo, hi in TILT_BUCKETS:
                b = bucket_label(lo, hi)
                row[f'mean_pair_m_tilt_{b}'] = statistics.mean(by_bucket[b]) if by_bucket[b] else None
                row[f'n_pairs_tilt_{b}'] = len(by_bucket[b])
                row[f'median_pair_m_grade_{b}'] = pct(by_grade[b], .5) if by_grade[b] else None
                row[f'n_pairs_grade_{b}'] = len(by_grade[b])
                row[f'median_pair_m_relroad_{b}'] = pct(by_rel[b], .5) if by_rel[b] else None
                row[f'n_pairs_relroad_{b}'] = len(by_rel[b])
            all_rows.append(row)
            print(f"{city:>11} {name:>14}  mean {row['mean_pair_m']:.3f}  median {row['median_pair_m']:.3f}  "
                  f"p90 {row['p90_pair_m']:.3f}  norm {row['mean_pair_over_range']:.4f}  pairs {len(dists)}")
        write_csv(out_dir_for(city, args.out) / 'ablation.csv', [r for r in all_rows if r['city'] == city])
    write_csv(out_dir_for('_summary', args.out) / 'ablation.csv', all_rows)


# --- eval: RampNet GT in world space ---------------------------------------------------

def gt_matched_distances(verdict_panos, bundle_ops, run_panos, params, prefused, radius_m=5.0):
    sites, frame, _ = prefused
    op_sites = [s for s in sites if s.n_operational > 0]
    run_by_id = {p.pano_id: p for p in run_panos}
    points, _, counts, _ = es.build_gt(verdict_panos, bundle_ops, run_by_id, params, frame)
    ramps = es.merge_gt_points(points, 2.5)
    pool = [r for r in ramps if r.in_pool]
    matched = es.match_one_to_one(pool, op_sites, radius_m)
    return [math.hypot(pool[i].e - s.e, pool[i].n - s.n) for i, s in matched.items()], counts


def load_gt_files(bench, benchmark_root):
    """verdicts + the frozen bundle's operational detections.

    ``eval_sites.load_city_files`` also parses the whole ``results.jsonl`` into
    SlimPanos, which ``load_run`` re-reads anyway (it needs ``source_metadata``
    that load_results discards) — a second 107 MB parse per city, thrown away.
    """
    with open(benchmark_root / bench / 'verdicts.json', encoding='utf-8') as f:
        verdicts = json.load(f)
    bundle_ops = {}
    with open(benchmark_root / bench / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bundle_ops[rec['pano']['panorama_id']] = \
                    [(d['x_normalized'], d['y_normalized'], d['confidence'])
                     for d in rec['detections']]
    return verdicts['panos'], bundle_ops


def cmd_eval(args):
    """World P/R per convention.

    Caveat carried into every row: GT marks are raycast under the pose being
    tested, so a convention that pushes a mark above the horizon or past the 25 m
    cap removes it from the recall DENOMINATOR as well as the numerator.
    ``pool_ramps`` therefore moves between rows and ``world_recall_*`` is not a
    like-for-like comparison. ``recall_vs_off_pool_*`` re-expresses the same
    numerator over one fixed denominator — the flat ('off') raycast's pool — which
    is comparable across rows at the cost of being able to exceed 1.
    """
    rows = []
    for city in args.cities:
        bench = BENCHMARK_OF.get(city, city)
        verdict_panos, bundle_ops = load_gt_files(bench, args.benchmark_root)
        panos, poses, _ = load_run(city)
        for name, signs in CONVENTIONS:
            if name in ('pitch-only', 'roll-only'):
                continue
            ps_ = with_convention(panos, poses, signs)
            # Benchmark tier, for the same reason as `ablation` above: this arm is scored
            # against GT bundles that were exported and judged at 0.55.
            # 'gravity' = rotate by the angles as injected: with_convention has already
            # put this convention's (possibly road-relative) pitch/roll on each pano.
            params = replace(fs.FuseParams(min_confidence=BENCHMARK_CONFIDENCE,
                                           mask_rig=False),
                             apply_pose=fs.POSE_OFF if signs is None else fs.POSE_GRAVITY)
            prefused = fs.fuse(ps_, params)
            r5 = es.evaluate_city(verdict_panos, bundle_ops, ps_, params, match_radius_m=5.0,
                                  prefused=prefused)
            r25 = es.evaluate_city(verdict_panos, bundle_ops, ps_, params, match_radius_m=2.5,
                                   prefused=prefused)
            dists, counts = gt_matched_distances(verdict_panos, bundle_ops, ps_, params, prefused)
            row = {'city': city, 'convention': name,
                   'n_sites': r5['fuse']['n_sites'], 'n_multi_pano_sites': r5['fuse']['n_multi_pano_sites'],
                   'gt_placeable': counts['placeable'], 'gt_unplaceable': counts['unplaceable'],
                   'pool_ramps': r5['n_pool_ramps'],
                   'world_recall_5m': r5['world_recall'], 'own_view_recall_5m': r5['own_view_recall'],
                   'precision_5m': r5['precision']['value'],
                   'tp_5m': r5['precision']['tp'], 'fp_5m': r5['precision']['fp'],
                   'unmatched_5m': r5['buckets']['unmatched'],
                   'world_recall_2p5m': r25['world_recall'], 'unmatched_2p5m': r25['buckets']['unmatched'],
                   # world_recall's own numerator: pool - unmatched would also count
                   # the sub-threshold-only matches world_recall deliberately excludes.
                   'matched_5m': round(r5['world_recall'] * r5['n_pool_ramps']),
                   'matched_2p5m': round(r25['world_recall'] * r25['n_pool_ramps']),
                   'gt_ramps': r5['counts']['gt_ramps'],
                   'match_dist_p50': pct(dists, .5), 'match_dist_p90': pct(dists, .9),
                   'n_matched': len(dists)}
            rows.append(row)
            print(f"{city:>11} {name:>14}  R5 {row['world_recall_5m']:.3f}  R2.5 {row['world_recall_2p5m']:.3f}  "
                  f"P {row['precision_5m']:.3f}  match p50/p90 {row['match_dist_p50']:.2f}/{row['match_dist_p90']:.2f} m  "
                  f"pool {row['pool_ramps']}  GT unplaceable {counts['unplaceable']}  sites {row['n_sites']}")
        # Fixed denominator: the flat raycast's pool, so the rows are comparable.
        pool_off = next(r['pool_ramps'] for r in rows
                        if r['city'] == city and r['convention'] == 'off')
        for r in (r for r in rows if r['city'] == city):
            r['pool_ramps_off'] = pool_off
            r['recall_vs_off_pool_5m'] = r['matched_5m'] / pool_off
            r['recall_vs_off_pool_2p5m'] = r['matched_2p5m'] / pool_off
            print(f"{city:>11} {r['convention']:>14}  recall vs the off pool ({pool_off}): "
                  f"5 m {r['recall_vs_off_pool_5m']:.3f}  2.5 m {r['recall_vs_off_pool_2p5m']:.3f}")
        write_csv(out_dir_for(city, args.out) / 'gt_eval.csv', [r for r in rows if r['city'] == city])
    write_csv(out_dir_for('_summary', args.out) / 'gt_eval.csv', rows)


# --- rectify: pixel-level sign lock ------------------------------------------------------

def rectify(img, R, heading_deg, out_w):
    """Re-render an equirect so world-up is image-up, keeping the camera heading at the
    center column. img: HxW(x3) float array; R: world->camera matrix."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    out_h = out_w // 2
    xo = (np.arange(out_w) + 0.5) / out_w
    yo = (np.arange(out_h) + 0.5) / out_h
    phi = (xo - 0.5) * 2 * np.pi + math.radians(heading_deg)
    theta = (0.5 - yo) * np.pi
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    e = ct * np.sin(phi)[None, :]
    n = ct * np.cos(phi)[None, :]
    u = np.broadcast_to(st, e.shape)
    Rm = np.asarray(R)
    cx = Rm[0, 0] * e + Rm[0, 1] * n + Rm[0, 2] * u
    cy = Rm[1, 0] * e + Rm[1, 1] * n + Rm[1, 2] * u
    cz = Rm[2, 0] * e + Rm[2, 1] * n + Rm[2, 2] * u
    lon = np.arctan2(cx, cz)
    lat = np.arctan2(-cy, np.hypot(cx, cz))
    H, W = img.shape[:2]
    xs = (lon / (2 * np.pi) + 0.5) * W - 0.5
    ys = (0.5 - lat / np.pi) * H - 0.5
    xs = np.mod(xs, W)          # longitude genuinely wraps
    # ...latitude does not: ys only leaves [0, H-1] within half a pixel of a pole,
    # where mode='wrap' would sample the opposite pole. Clamp instead. (No effect on
    # `verticality`, which crops to elevation -20..+35 long before the poles.)
    ys = np.clip(ys, 0, H - 1)
    if img.ndim == 2:
        return map_coordinates(img, [ys, xs], order=1, mode='wrap')
    return np.stack([map_coordinates(img[..., c], [ys, xs], order=1, mode='wrap')
                     for c in range(img.shape[2])], axis=-1)


def verticality(gray, elev_lo_deg=-20.0, elev_hi_deg=35.0, tol_deg=4.0, top_q=0.9):
    """Fraction of strong-gradient pixels in the building band whose edge is vertical.
    In a gravity-aligned equirect every world-vertical line is a straight pixel
    column, so this rises when the image is level and falls when it is tilted."""
    import numpy as np
    from scipy.ndimage import gaussian_filter, sobel
    H = gray.shape[0]
    g = gaussian_filter(gray, 1.0)
    gx, gy = sobel(g, axis=1), sobel(g, axis=0)
    r0 = int((0.5 - elev_hi_deg / 180.0) * H)
    r1 = int((0.5 - elev_lo_deg / 180.0) * H)
    gx, gy = gx[r0:r1], gy[r0:r1]
    mag = np.hypot(gx, gy)
    strong = mag > np.quantile(mag, top_q)
    # a vertical edge has a horizontal gradient: |gy| small relative to |g|
    vertical = np.abs(gy) < mag * math.sin(math.radians(tol_deg))
    return float(np.mean(vertical[strong]))


def load_pano_gray(path, width):
    """Decode a benchmark pano at reduced size (JPEG DCT scaling) -> float gray array."""
    import numpy as np
    from PIL import Image
    im = Image.open(path)
    im.draft('L', (width, width // 2))
    im = im.convert('L').resize((width, width // 2), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def cmd_rectify(args):
    import numpy as np
    rows = []
    for city in args.cities:
        bench = BENCHMARK_OF.get(city, city)
        pano_dir = args.benchmark_root / bench / 'panos'
        _, poses, _ = load_run(city)
        files = sorted(pano_dir.glob('*.jpg'))
        if args.limit:
            files = files[:args.limit]
        done = 0
        for f in files:
            pid = f.stem
            pose = poses.get(pid)
            if pose is None:
                continue
            gray = load_pano_gray(f, args.width)
            base = verticality(gray)
            row = {'city': city, 'pano_id': pid, 'tilt_deg': pose['tilt_deg'],
                   'tilt_bucket': bucket_of(pose['tilt_deg']), 'pitch_deg': pose['pitch_deg'],
                   'roll_deg': pose['roll_deg'], 'make': pose['make'], 'model': pose['model'],
                   'v_original': base}
            for name, signs in CONVENTIONS:
                if signs is None or signs == 'road':
                    continue
                R = matrix_from_pose(pose['heading_deg'], signs[0] * pose['pitch_deg'],
                                     signs[1] * pose['roll_deg'])
                rect = rectify(gray, R, pose['heading_deg'], args.width)
                row[f'v_{name}'] = verticality(rect)
            rows.append(row)
            done += 1
            if done % 25 == 0:
                print(f'{city}: {done}/{len(files)}', flush=True)
        write_csv(out_dir_for(city, args.out) / 'verticality.csv',
                  [r for r in rows if r['city'] == city])
        cr = [r for r in rows if r['city'] == city]
        tilted = [r for r in cr if r['tilt_deg'] >= 3.0]
        for name, signs in CONVENTIONS:
            if signs is None or signs == 'road':
                continue
            wins = sum(r[f'v_{name}'] > r['v_original'] for r in tilted)
            print(f"{city:>11} {name:>14}: verticality gain>0 on {wins}/{len(tilted)} panos with tilt>=3 deg; "
                  f"mean gain {np.mean([r[f'v_{name}'] - r['v_original'] for r in tilted]):+.4f}")
    write_csv(out_dir_for('_summary', args.out) / 'verticality.csv', rows)


def cmd_examples(args):
    """Render original / rectified(documented) / rectified(both-flipped) strips for the
    most tilted benchmark panos of each city — the image evidence for the report."""
    import numpy as np
    from PIL import Image, ImageDraw
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for city in args.cities:
        bench = BENCHMARK_OF.get(city, city)
        pano_dir = args.benchmark_root / bench / 'panos'
        _, poses, _ = load_run(city)
        cands = [(poses[f.stem]['tilt_deg'], f) for f in pano_dir.glob('*.jpg') if f.stem in poses]
        cands.sort(reverse=True)
        picks = [f for _, f in cands[:args.limit or 3]]
        if args.pano:
            picks = [pano_dir / f'{p}.jpg' for p in args.pano if (pano_dir / f'{p}.jpg').exists()]
        for f in picks:
            pose = poses[f.stem]
            im = Image.open(f)
            im.draft('RGB', (args.width, args.width // 2))
            im = im.convert('RGB').resize((args.width, args.width // 2), Image.BILINEAR)
            rgb = np.asarray(im, dtype=np.float32)
            panels = [('original (as served)', rgb)]
            for name, signs in (('documented', (1, 1)), ('both-flipped', (-1, -1))):
                R = matrix_from_pose(pose['heading_deg'], signs[0] * pose['pitch_deg'],
                                     signs[1] * pose['roll_deg'])
                panels.append((f'rectified, {name} convention', rectify(rgb, R, pose['heading_deg'], args.width)))
            # crop to the band that matters (elev -30..+45) for a compact strip
            H = args.width // 2
            r0, r1 = int((0.5 - 45 / 180) * H), int((0.5 + 30 / 180) * H)
            strip_h = r1 - r0
            canvas = Image.new('RGB', (args.width, (strip_h + 28) * len(panels)), 'white')
            draw = ImageDraw.Draw(canvas)
            for k, (label, arr) in enumerate(panels):
                tile = Image.fromarray(np.clip(arr[r0:r1], 0, 255).astype(np.uint8))
                y0 = k * (strip_h + 28)
                draw.text((8, y0 + 6), f'{label}   pano {f.stem}  tilt {pose["tilt_deg"]:.1f} deg  '
                          f'pitch {pose["pitch_deg"]:+.1f}  roll {pose["roll_deg"]:+.1f}  ({pose["make"]} {pose["model"]})',
                          fill='black')
                canvas.paste(tile, (0, y0 + 28))
                # horizon guide: the rectified world horizon is the row at elev 0
                yh = y0 + 28 + int((0.5 * H) - r0)
                draw.line([(0, yh), (args.width, yh)], fill=(255, 60, 60), width=1)
            out = FIG_DIR / f'example_{city}_{f.stem}.jpg'
            canvas.save(out, quality=82)
            print('wrote', out)


# --- grade: is the tilt the rig, or the road? ---------------------------------------------

def cmd_grade(args):
    """Regress the camera's gravity-relative pitch along the direction of travel on
    the road grade from the sequence's SfM altitude profile. Slope ~1: the camera
    rides level on a vehicle and its 'tilt' is the road, so a flat-ground raycast in
    the camera frame is already right. Slope ~0: the tilt is the rig's own, and the
    gravity correction is the right one."""
    rows = []
    for city in args.cities:
        _, poses, _ = load_run(city)
        pairs = [(q['grade_deg'], q['travel_pitch_deg'], q['pitch_rel_road_deg'])
                 for q in poses.values() if q['grade_deg'] is not None and abs(q['grade_deg']) < 20]
        n = len(pairs)
        if n < 10:
            print(f'{city}: {n} usable frames — no altitude profile')
            continue
        g = [p[0] for p in pairs]
        t = [p[1] for p in pairs]
        mg, mt = statistics.mean(g), statistics.mean(t)
        sxy = sum((a - mg) * (b - mt) for a, b in zip(g, t))
        sxx = sum((a - mg) ** 2 for a in g)
        syy = sum((b - mt) ** 2 for b in t)
        slope = sxy / sxx if sxx else None
        corr = sxy / math.sqrt(sxx * syy) if sxx and syy else None
        rel = [abs(p[2]) for p in pairs]
        trav = [abs(p[1]) for p in pairs]
        # n_frames/n_panos is exactly the road-relative convention's coverage: the
        # rest fall back to the gravity-relative angles (see pose_angles).
        row = {'city': city, 'n_frames': n, 'n_panos': len(poses),
               'frac_with_grade': n / len(poses),
               'slope_travel_pitch_on_grade': slope, 'corr': corr,
               'grade_abs_p50': pct([abs(x) for x in g], .5), 'grade_abs_p90': pct([abs(x) for x in g], .9),
               'travel_pitch_abs_p50': pct(trav, .5), 'travel_pitch_abs_p90': pct(trav, .9),
               'pitch_rel_road_abs_p50': pct(rel, .5), 'pitch_rel_road_abs_p90': pct(rel, .9),
               'frac_rel_road_within_1p5': sum(x < 1.5 for x in rel) / n,
               'frac_travel_pitch_within_1p5': sum(x < 1.5 for x in trav) / n}
        rows.append(row)
        print(f"{city:>11}: {n} frames ({row['frac_with_grade']:.1%} of panos; the rest fall "
              f"back to gravity-relative); travel-pitch vs grade slope {slope:.2f} (r={corr:.2f}); "
              f"|grade| p50 {row['grade_abs_p50']:.2f}; |pitch along travel| p50 {row['travel_pitch_abs_p50']:.2f} "
              f"-> relative to road p50 {row['pitch_rel_road_abs_p50']:.2f} deg")
        write_csv(out_dir_for(city, args.out) / 'grade.csv',
                  [{'pano_id': q['pano_id'], 'sequence': q['sequence'], 'make': q['make'], 'model': q['model'],
                    'grade_deg': q['grade_deg'], 'travel_bearing_deg': q['travel_bearing_deg'],
                    'travel_pitch_deg': q['travel_pitch_deg'], 'pitch_rel_road_deg': q['pitch_rel_road_deg'],
                    'pitch_deg': q['pitch_deg'], 'roll_deg': q['roll_deg'], 'tilt_deg': q['tilt_deg']}
                   for q in poses.values() if q['grade_deg'] is not None])
    write_csv(out_dir_for('_summary', args.out) / 'grade.csv', rows)


# --- horizon: reviewer marks that end up above the horizon ---------------------------------

def cmd_horizon(args):
    """A curb ramp is on the ground, so a reviewer's mark raycast above the horizon
    is a pose error by definition. Count them per convention — no association, no
    matching, just the marks and the rotation.

    The marks come through ``eval_sites.judged_gt_panos``, the same guard
    ``build_gt`` uses, so a pano whose stored detections drifted from the frozen
    bundle is skipped rather than silently pairing a verdict with a peak the
    reviewer never saw — which would have read as a clean '0 above the horizon'.
    """
    rows = []
    for city in args.cities:
        bench = BENCHMARK_OF.get(city, city)
        verdicts, bundle_ops = load_gt_files(bench, args.benchmark_root)
        panos, poses, _ = load_run(city)
        by_id = {p.pano_id: p for p in panos}
        counts, warnings = es.gt_counts(), []
        counts['gt_panos'] = len(verdicts)
        marks = []   # (pano, x, y)
        for pid, entry, p, ops, _in_pool in es.judged_gt_panos(
                verdicts, bundle_ops, by_id, counts, warnings):
            for v, (_i, x, y, _c) in zip(entry['dets'], ops):
                if v is True:
                    marks.append((p, x, y))
            for m in entry.get('missed', ()):
                if not m.get('unsure'):
                    marks.append((p, m['x'], m['y']))
        if warnings:
            print(f'{city}: {len(warnings)} GT panos skipped by the bundle guard; '
                  f'first: {warnings[0]}')
        for name, signs in CONVENTIONS:
            above = beyond = 0
            for p, x, y in marks:
                pose = poses[p.pano_id]
                cp, cr = pose_angles(pose, signs)
                pp = geo.pano_pose({'lat': p.lat, 'lng': p.lng, 'camera_heading': p.camera_heading,
                                    'camera_pitch': cp, 'camera_roll': cr, 'source': p.source})
                g = geo.detection_ground_point(pp, x, y, max_range_m=math.inf,
                                               apply_pose=signs is not None)
                if g is None:
                    above += 1
                elif g.range_m > geo.DEFAULT_MAX_RANGE_M:
                    beyond += 1
            rows.append({'city': city, 'convention': name, 'n_marks': len(marks),
                         'n_gt_panos_judged': counts['judged'],
                         'n_gt_panos_skipped': counts['skipped'],
                         'above_horizon': above, 'beyond_25m': beyond,
                         'frac_above_horizon': above / len(marks) if marks else None})
            print(f"{city:>11} {name:>14}: {above:4d} of {len(marks)} GT marks above the horizon, "
                  f"{beyond} beyond 25 m")
        write_csv(out_dir_for(city, args.out) / 'horizon.csv',
                  [r for r in rows if r['city'] == city])
    write_csv(out_dir_for('_summary', args.out) / 'horizon.csv', rows)


# --- pose: one pano ----------------------------------------------------------------------

def cmd_pose(args):
    _, poses, _ = load_run(args.run)
    for pid in args.pano_id:
        pose = poses.get(pid)
        if pose is None:
            print(f'{pid}: not in runs/{args.run}')
            continue
        print(json.dumps({k: pose[k] for k in ('pano_id', 'make', 'model', 'sequence', 'rvec',
                                                'camera_heading', 'heading_deg', 'pitch_deg',
                                                'roll_deg', 'tilt_deg')}, indent=2))


# --- figures -------------------------------------------------------------------------------

def _read_csv(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ('', None):
                r[k] = None
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def cmd_figures(args):
    """Redraw every figure.

    The aggregated CSVs are looked up in the run tree first and then in
    ``docs/figures/mapillary-tilt/data/``, the copies committed beside the report —
    so figures 3-7 and 9 redraw from a fresh clone with no run data at all.
    Figures 1, 2 and 8 need the per-panorama ``poses.csv`` / ``grade.csv`` (up to
    73k rows per city, deliberately not committed); without them they are skipped
    with a message rather than crashing. Re-run ``stats`` and ``grade`` to get them.

    ``--out`` is an INPUT root here, unlike every other subcommand: it says where to read
    the CSVs from. The PNGs always go to ``docs/figures/mapillary-tilt/``, so redrawing
    from an experimental run tree overwrites the committed figures -- intended, since they
    are meant to track the committed CSVs, but worth knowing before pointing it somewhere
    odd.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    root = args.out or REPO_ROOT / 'runs'
    summ = root / '_summary' if args.out else root / '_summary' / 'tilt'
    cities = args.cities
    colors = {'richmond': '#1f77b4', 'clovis': '#ff7f0e', 'morgantown': '#2ca02c',
              'annapolis': '#d62728', 'laurens': '#9467bd'}

    def tilt_dir(city):
        return (root / city) if args.out else root / city / 'tilt'

    def summary_csv(name):
        """Aggregated CSV from the run tree, else the copy committed under data/."""
        p = summ / name
        return _read_csv(p if p.exists() else FIG_DIR / 'data' / name)

    def city_csvs(name):
        """[(city, rows)] for the cities that have this per-panorama CSV, or []."""
        out = []
        for city in cities:
            p = tilt_dir(city) / name
            if p.exists():
                out.append((city, _read_csv(p)))
        missing = [c for c in cities if not (tilt_dir(c) / name).exists()]
        if missing:
            print(f'skipping a figure: no {name} for {", ".join(missing)} '
                  f'(per-panorama CSVs are not committed — re-run stats/grade)')
        return [] if missing else out

    poses_by_city = dict(city_csvs('poses.csv'))

    # Fig 1: tilt CDF per city
    if poses_by_city:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for city in cities:
            t = np.sort([r['tilt_deg'] for r in poses_by_city[city]])
            ax.plot(t, np.linspace(0, 1, len(t)), label=f'{city} (n={len(t):,})',
                    color=colors.get(city))
        ax.axvline(1.5, color='k', ls='--', lw=1)
        ax.text(1.6, 0.05, 'σ_pitch assumed today (1.5°)', fontsize=8)
        ax.set_xscale('log')
        ax.set_xlim(0.1, 60)
        ax.set_xlabel('rig tilt: angle between camera-up and world-up (degrees, log scale)')
        ax.set_ylabel('fraction of panoramas')
        ax.set_title('Mapillary rig tilt from computed_rotation, all processed panoramas')
        ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig1_tilt_cdf.png', dpi=150)

    # Fig 2: pitch vs roll by rig, one panel per city
    if poses_by_city:
        fig, axes = plt.subplots(1, len(cities), figsize=(3.2 * len(cities), 3.4), sharex=True, sharey=True)
        for ax, city in zip(np.atleast_1d(axes), cities):
            rows = poses_by_city[city]
            rng = np.random.default_rng(0)
            idx = rng.choice(len(rows), size=min(4000, len(rows)), replace=False)
            rigs = defaultdict(list)
            for i in idx:
                r = rows[i]
                rigs[f"{r['make']} {r['model']}"[:28]].append((r['pitch_deg'], r['roll_deg']))
            for k, (rig, pts) in enumerate(sorted(rigs.items(), key=lambda kv: -len(kv[1]))):
                p = np.array(pts)
                ax.scatter(p[:, 0], p[:, 1], s=3, alpha=.35, label=f'{rig} ({len(pts)})')
            ax.set_title(city, fontsize=10)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.axhline(0, color='k', lw=.5)
            ax.axvline(0, color='k', lw=.5)
            ax.set_xlabel('pitch (deg)')
            ax.legend(fontsize=6, loc='lower left')
        np.atleast_1d(axes)[0].set_ylabel('roll (deg, PS sign; PR #50 plotted the opposite)')
        fig.suptitle('Pitch and roll by rig (random 4,000 panos per city)', fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig2_pitch_roll_by_rig.png', dpi=150)

    # Fig 3: sequence consistency — within-sequence SD vs the between-sequence spread
    seq = summary_csv('tilt_by_sequence.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key in zip(axes, ('pitch', 'roll')):
        for city in cities:
            rs = [r for r in seq if r['city'] == city]
            sd = np.sort([r[f'{key}_sd'] for r in rs])
            ax.plot(sd, np.linspace(0, 1, len(sd)), label=f'{city} ({len(sd)} seqs)', color=colors.get(city))
        ax.set_xscale('log')
        ax.set_xlim(0.05, 30)
        ax.set_xlabel(f'within-sequence SD of {key} (deg, log)')
        ax.set_ylabel('fraction of sequences (>=5 panos)')
        ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=7)
    fig.suptitle('Tilt is a per-frame quantity: it varies within a sequence, not just between rigs', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_sequence_consistency.png', dpi=150)

    # Fig 4: ground-point displacement vs tilt bucket
    disp = summary_csv('displacement_by_bucket.csv')
    buckets = BUCKET_LABELS
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.8 / len(cities)
    for k, city in enumerate(cities):
        rs = {r['tilt_bucket']: r for r in disp if r['city'] == city}
        vals = [rs[b]['displacement_p50'] if b in rs and rs[b]['displacement_p50'] is not None else 0 for b in buckets]
        p90 = [rs[b]['displacement_p90'] if b in rs and rs[b]['displacement_p90'] is not None else 0 for b in buckets]
        x = np.arange(len(buckets)) + (k - len(cities) / 2 + .5) * w
        ax.bar(x, vals, w, color=colors.get(city), label=city)
        ax.scatter(x, p90, color=colors.get(city), marker='_', s=120)
    ax.set_xticks(np.arange(len(buckets)))
    ax.set_xticklabels([f'{b}°' for b in buckets])
    ax.set_xlabel('rig tilt bucket (degrees)')
    ax.set_ylabel('ground-point displacement, tilt applied vs flat (m)')
    ax.set_title('How far the flat raycast is from the tilt-corrected one (bars = median, ticks = p90)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_displacement.png', dpi=150)

    # Fig 5: multi-view ablation — mean pairwise member distance per convention
    abl = summary_csv('ablation.csv')
    names = [n for n, _ in CONVENTIONS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    w = 0.8 / len(cities)
    for k, city in enumerate(cities):
        rs = {r['convention']: r for r in abl if r['city'] == city}
        base = rs['off']['median_pair_m']
        vals = [rs[n]['median_pair_m'] / base for n in names]
        ax.bar(np.arange(len(names)) + (k - len(cities) / 2 + .5) * w, vals, w,
               color=colors.get(city), label=city)
    ax.axhline(1.0, color='k', lw=.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('median within-site pairwise distance, relative to pose-off')
    ax.set_title('Every sign flip loosens every city; the gravity correction splits by regime', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3, axis='y')
    ax = axes[1]
    for city in cities:
        rs = {r['convention']: r for r in abl if r['city'] == city}
        off = [rs['off'][f'mean_pair_m_tilt_{b}'] for b in buckets]
        for conv, ls in (('documented', '-'), ('road-relative', '--')):
            on = [rs[conv][f'mean_pair_m_tilt_{b}'] for b in buckets]
            rel = [None if (o is None or a is None or o == 0) else a / o for o, a in zip(off, on)]
            ax.plot(range(len(buckets)), [np.nan if v is None else v for v in rel], marker='o',
                    ls=ls, color=colors.get(city), label=f'{city} ({conv})' if conv == 'documented' else None)
    ax.axhline(1.0, color='k', lw=.8)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f'{b}°' for b in buckets])
    ax.set_xlabel("pair's larger rig tilt (gravity-relative)")
    ax.set_ylabel('mean pairwise distance relative to pose-off')
    ax.set_title('solid = gravity-relative (documented), dashed = road-relative', fontsize=9)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_ablation.png', dpi=150)

    # Fig 6: verticality gain by convention, panos with tilt >= 3 deg
    vert = summary_csv('verticality.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    conv = ['documented', 'pitch-flipped', 'roll-flipped', 'both-flipped', 'pitch-only', 'roll-only']
    data = []
    for n in conv:
        data.append([r[f'v_{n}'] - r['v_original'] for r in vert if r['tilt_deg'] >= 3.0])
    ax.boxplot(data, showfliers=False, widths=.6)
    ax.set_xticks(range(1, len(conv) + 1))
    ax.set_xticklabels(conv, rotation=20, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=.8)
    ax.set_ylabel('vertical-edge fraction: rectified minus original')
    ax.set_title(f'Pixel-level sign lock on {len(data[0])} benchmark panos with tilt >= 3° '
                 '(no detector, no GT, no association involved)', fontsize=9)
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_verticality.png', dpi=150)

    # Fig 7: GT eval — world recall at 2.5 m and matched-distance p90 per convention
    ev = summary_csv('gt_eval.csv')
    conv = ['off', 'documented', 'road-relative', 'pitch-flipped', 'roll-flipped', 'both-flipped']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(axes, ('world_recall_2p5m', 'match_dist_p90', 'gt_unplaceable'),
                              ('world recall, 2.5 m match radius', 'GT-to-site distance p90 (m)',
                               'GT marks that raycast above the horizon')):
        w = 0.8 / len(cities)
        for k, city in enumerate(cities):
            rs = {r['convention']: r for r in ev if r['city'] == city}
            vals = [rs[n][key] if n in rs and rs[n][key] is not None else 0 for n in conv]
            ax.bar(np.arange(len(conv)) + (k - len(cities) / 2 + .5) * w, vals, w,
                   color=colors.get(city), label=city)
        ax.set_xticks(np.arange(len(conv)))
        ax.set_xticklabels(conv, rotation=25, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3, axis='y')
    axes[0].set_ylim(0.6, 1.0)
    axes[0].legend(fontsize=7)
    fig.suptitle('Against RampNet ground truth (reviewer marks raycast with the same pose model)', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig7_gt_eval.png', dpi=150)

    # Fig 8: camera pitch along travel vs road grade — rig tilt or road?
    grade_by_city = dict(city_csvs('grade.csv'))
    if grade_by_city:
        fig, axes = plt.subplots(1, len(cities), figsize=(3.2 * len(cities), 3.4), sharex=True, sharey=True)
        gsum = {r['city']: r for r in summary_csv('grade.csv')}
        for ax, city in zip(np.atleast_1d(axes), cities):
            rows = grade_by_city[city]
            g = np.array([r['grade_deg'] for r in rows])
            t = np.array([r['travel_pitch_deg'] for r in rows])
            keep = np.abs(g) < 20
            ax.hexbin(g[keep], t[keep], gridsize=45, extent=(-12, 12, -12, 12), bins='log', cmap='viridis')
            ax.plot([-12, 12], [-12, 12], color='w', lw=1, ls='--')
            sl = gsum[city]['slope_travel_pitch_on_grade']
            ax.set_title(f"{city}: slope {sl:.2f}, r={gsum[city]['corr']:.2f}", fontsize=9)
            ax.set_xlabel('road grade along travel (deg)')
        np.atleast_1d(axes)[0].set_ylabel('camera pitch along travel (deg, gravity-relative)')
        fig.suptitle('Slope 1 (dashed) = the camera rides level on the vehicle and the "tilt" is the road; '
                     'slope 0 = the rig itself is tilted', fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'fig8_grade_mechanism.png', dpi=150)

    # Fig 9: the two regimes — median pair distance by |grade| and by |pitch relative to the road|
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, xlabel in zip(axes, ('grade', 'relroad'),
                               ("pair's larger |road grade| (deg)",
                                "pair's larger |camera pitch relative to the road| (deg)")):
        for city in cities:
            rs = {r['convention']: r for r in abl if r['city'] == city}
            for conv, ls, mk in (('documented', '-', 'o'), ('road-relative', '--', 's')):
                vals = []
                for b in buckets:
                    o, a = rs['off'][f'median_pair_m_{key}_{b}'], rs[conv][f'median_pair_m_{key}_{b}']
                    n = rs['off'][f'n_pairs_{key}_{b}']
                    vals.append(np.nan if (o is None or a is None or not o or (n or 0) < 100) else a / o)
                ax.plot(range(len(buckets)), vals, ls=ls, marker=mk, color=colors.get(city),
                        label=f'{city}' if conv == 'documented' else None)
        ax.axhline(1.0, color='k', lw=.8)
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels([f'{b}°' for b in buckets])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('median pairwise distance relative to pose-off')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle('solid = gravity-relative correction, dashed = road-relative; below 1 = tighter multi-view agreement '
                 '(buckets with >=100 pairs)', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig9_regimes.png', dpi=150)
    print('wrote figures to', FIG_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    OUT_WRITE = 'write the CSVs under <out>/<city>/ instead of runs/<city>/tilt/'
    OUT_READ = ('READ the CSVs from <out>/<city>/ and <out>/_summary/ instead of the run '
                'tree; the PNGs always go to docs/figures/mapillary-tilt/')
    OUT_UNUSED = 'unused by this subcommand; the strips always go to docs/figures/mapillary-tilt/'

    def common(p, cities=True, out_help=OUT_WRITE):
        if cities:
            p.add_argument('cities', nargs='*', default=DEFAULT_CITIES)
        p.add_argument('--out', type=Path, default=None, help=out_help)
        p.add_argument('--benchmark-root', type=Path,
                       default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    for name, fn in (('stats', cmd_stats), ('displacement', cmd_displacement),
                     ('ablation', cmd_ablation), ('eval', cmd_eval), ('grade', cmd_grade),
                     ('horizon', cmd_horizon)):
        p = sub.add_parser(name)
        common(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser('rectify')
    common(p)
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--limit', type=int, default=0)
    p.set_defaults(fn=cmd_rectify)
    p = sub.add_parser('examples')
    common(p, out_help=OUT_UNUSED)
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--pano', nargs='*', default=None)
    p.set_defaults(fn=cmd_examples)
    p = sub.add_parser('figures')
    common(p, out_help=OUT_READ)
    p.set_defaults(fn=cmd_figures)
    p = sub.add_parser('pose')
    p.add_argument('pano_id', nargs='+')
    p.add_argument('--run', default='richmond')
    p.set_defaults(fn=cmd_pose)
    args = ap.parse_args()
    if getattr(args, 'cities', None) == []:
        args.cities = DEFAULT_CITIES
    args.fn(args)


if __name__ == '__main__':
    main()
