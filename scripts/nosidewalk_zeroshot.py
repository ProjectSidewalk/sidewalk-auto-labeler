"""Zero-training NoSidewalk per block face from street-level panoramas (a pre-registered
study; docs/overnight/nosidewalk_zeroshot.md holds the pre-registration and results).

A STUDY, not production: nothing here is wired into main.py or send_to_ps.py.

Question. Can an off-the-shelf semantic segmenter (Mask2Former, Mapillary Vistas), with no
training at all, decide whether a block face (city, street_edge_id, street_side) has a
sidewalk, scored against SDOT's sidewalk inventory as joined to Project Sidewalk's Seattle
street edges by the #5222 study (Planning reference/sidewalk-presence-sdot-study)?

Method v0, per pano: two lateral perspective views (90 deg FOV, 1024 px, pitched
VIEW_PITCH_DEG), one per side of the street, aimed along the outward normal of the PS
street edge's local tangent (left = tangent - 90 deg, right = tangent + 90 deg; "left" is
left of the edge's digitized direction, the #5222 / label_point.street_side convention).
Each view is segmented with Mask2Former-Vistas (the recipe of scripts/footway_segmentation.py
`segment`, reused as is). A fixed ground band -- flat ground CAMERA_HEIGHT_M below the
camera, BAND_LATERAL_M from the camera along the face normal, at most BAND_ALONG_M along
the street and inside the view's frustum -- is sampled on a uniform ground grid, so every
sample stands for the same ground area. Occluder classes (vehicles, people, riders) are
UNOBSERVED, not absent. A pano-side is `observed` when at least OBSERVED_MIN_FRAC of its
band is unoccluded; its score is the walkway share (Sidewalk + Pedestrian Area + Curb Cut)
of the observed band. A face's score pools its observed pano-sides with the
pre-registered POOL_PRIMARY statistic; a face with fewer than MIN_OBSERVED_PANOS observed
pano-sides is `unobserved`.

Subcommands, in order (where each runs is noted; GSV and segmentation run on klone):

    # Mac: seeded stratified face sample (SDOT verdict used only to stratify)
    python scripts/nosidewalk_zeroshot.py sample --sp-face <planning>/.../results/sp_face.csv.gz \
        --streets <streets.geojson from /v3/api/streets?filetype=geojson>
    # klone CPU job: GSV coverage -> mid-block candidates -> metadata -> selection -> views
    python scripts/nosidewalk_zeroshot.py gsv-views --work <work>
    # Mac: Mapillary arm (token stays on the Mac; panos are rendered to views, then deleted)
    python scripts/nosidewalk_zeroshot.py mly-views --work <work>
    # klone GPU job: segmentation, reusing footway_segmentation.py's recipe unchanged
    python scripts/footway_segmentation.py segment --in <work>/views --out <work>/labels
    # klone: band histograms per view (cheap, CPU)
    python scripts/nosidewalk_zeroshot.py bands --work <work>
    # Mac: OSM sidewalk=* tags for the sampled ways (Overpass, one query)
    python scripts/nosidewalk_zeroshot.py osm-tags
    # Mac: pooling, metrics, controls, decision rule, slices, gallery
    python scripts/nosidewalk_zeroshot.py score --sp-face <sp_face.csv.gz>

Small outputs go to docs/overnight/data/nosidewalk_zeroshot/ (DATA_DIR). Everything is
seeded (SEED) and deterministic given the inputs; GSV/Mapillary coverage drifts, so the
selected panos are recorded with ids and capture dates.

Stdlib + numpy (+ Pillow for views/bands); streetlevel for GSV, requests for Mapillary and
Overpass; torch + transformers only for the segment step (footway_segmentation.py).
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo root ahead of scripts/ (scripts/position_check.py is a shim sharing a root name).
for p in (str(REPO_ROOT / 'scripts'), str(REPO_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import geo  # noqa: E402

DATA_DIR = REPO_ROOT / 'docs' / 'overnight' / 'data' / 'nosidewalk_zeroshot'

# --- pre-registered constants (docs/overnight/nosidewalk_zeroshot.md, section 1) --------
SEED = 20261127
CITY = 'seattle'
ELIGIBLE_WAY_TYPES = ('residential', 'tertiary', 'secondary', 'primary')
MIN_EDGE_LEN_M = 40.0          # leaves >= 16 m of mid-block interior
END_BUFFER_M = 12.0            # mid-block: a pano's foot > 12 m from either end
N_DISCORDANT_STREETS = 90      # both faces sampled -> 180 faces
N_ABSENT_FACES = 210           # absent faces on non-discordant streets
N_PRESENT_FACES = 210          # present faces on non-discordant streets
DEV_FRACTION = 0.15            # of sampled streets, per stratum
STRATA = ('D', 'A', 'P')

MAX_PANOS_PER_STREET = 5
MIN_PANO_SPACING_M = 8.0
MAX_CROSS_TRACK_M = 10.0       # a candidate pano within 10 m of the edge centerline
TANGENT_HALF_CHORD_M = 10.0    # edge tangent = chord over +-10 m around the pano's foot

VIEW_SIZE = 1024
VIEW_FOV_DEG = 90.0
VIEW_PITCH_DEG = -30.0
CAMERA_HEIGHT_M = 2.5
BAND_LATERAL_M = (3.0, 15.0)
BAND_NEAR_FAR_SPLIT_M = 8.0
BAND_ALONG_M = 15.0
BAND_STEP_M = 0.25
OBSERVED_MIN_FRAC = 0.5
MIN_OBSERVED_PANOS = 2
POOL_PRIMARY = 'q75'
POOLS = ('q75', 'max', 'median', 'mean')
TARGET_RECALL = 0.67
N_BOOT = 2000

WALKWAY_CLASSES = ('Sidewalk', 'Pedestrian Area', 'Curb Cut')
VEGETATION_OCCLUDER = ('Vegetation',)   # amendment A1: a secondary variant only
CURB_CLASSES = ('Curb',)
OCCLUDER_CLASSES = (
    'Car', 'Truck', 'Bus', 'Other Vehicle', 'Trailer', 'Caravan', 'Motorcycle', 'Bicycle',
    'On Rails', 'Wheeled Slow', 'Boat', 'Person', 'Bicyclist', 'Motorcyclist',
    'Other Rider', 'Car Mount', 'Ego Vehicle', 'Bird', 'Ground Animal')

# Decision rule (advisors' rule, kept verbatim in the doc).
CONTINUE_PRECISION = 0.80
CONTINUE_DISC_AUC = 0.80
SWAP_COLLAPSED_MAX = 0.55
STOP_DISC_AUC = 0.70
STOP_NULL_MARGIN = 0.05
STOP_PRECISION = 0.70
STOP_COVERAGE = 0.50

SIDES = ('left', 'right')


# --- small io helpers ---------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def open_text(path):
    path = str(path)
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else \
        open(path, encoding='utf-8')


def read_csv(path):
    with open_text(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n', encoding='utf-8')


def face_uid(edge_id, side):
    """Faces are keyed by city as well: PS ids are per city schema."""
    return f'{CITY}:{int(edge_id)}:{side}'


# --- edge geometry ------------------------------------------------------------------------

def bearing_deg(de, dn):
    """Compass bearing (clockwise from north) of an (east, north) vector."""
    return math.degrees(math.atan2(de, dn)) % 360.0


class EdgeGeom:
    """A PS street edge as a polyline in a local metric frame, in its digitized direction.

    Example:
        >>> g = EdgeGeom([(-122.30, 47.60), (-122.30, 47.601)])   # runs due north
        >>> s, cross, tangent = g.project(47.6005, -122.30001)     # a bit west = left
        >>> round(tangent), cross > 0
        (0, True)
    """

    def __init__(self, coords):
        lats = [c[1] for c in coords]
        lngs = [c[0] for c in coords]
        self.frame = geo.LocalFrame(sum(lats) / len(lats), sum(lngs) / len(lngs))
        self.xy = np.array([self.frame.to_enu(lat, lng) for lng, lat in coords], dtype=float)
        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])

    def point_at(self, s):
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.cum, s, side='right') - 1)
        i = min(max(i, 0), len(self.seg_len) - 1)
        t = 0.0 if self.seg_len[i] == 0 else (s - self.cum[i]) / self.seg_len[i]
        return self.xy[i] + t * (self.xy[i + 1] - self.xy[i])

    def tangent_at(self, s, half=TANGENT_HALF_CHORD_M):
        a = self.point_at(s - half)
        b = self.point_at(s + half)
        if np.allclose(a, b):
            i = min(max(int(np.searchsorted(self.cum, s) - 1), 0), len(self.seg_len) - 1)
            a, b = self.xy[i], self.xy[i + 1]
        return bearing_deg(b[0] - a[0], b[1] - a[1])

    def project(self, lat, lng):
        """(s_m along the edge, signed cross-track m (+ = left), tangent bearing deg)."""
        p = np.array(self.frame.to_enu(lat, lng), dtype=float)
        best = None
        for i in range(len(self.seg_len)):
            a, b = self.xy[i], self.xy[i + 1]
            d = b - a
            L2 = float(d @ d)
            t = 0.0 if L2 == 0 else min(max(float((p - a) @ d) / L2, 0.0), 1.0)
            foot = a + t * d
            dist = float(np.hypot(*(p - foot)))
            if best is None or dist < best[0] - 1e-9:
                best = (dist, i, t, foot, d)
        dist, i, t, foot, d = best
        z = d[0] * (p - foot)[1] - d[1] * (p - foot)[0]     # > 0: p is left of the direction
        s = float(self.cum[i] + t * self.seg_len[i])
        return s, (dist if z >= 0 else -dist), self.tangent_at(s)


def face_bearing(tangent_deg, side):
    """Outward normal of a face: left of the digitized direction is tangent - 90."""
    if side not in SIDES:
        raise ValueError(side)
    return (tangent_deg + (-90.0 if side == 'left' else 90.0)) % 360.0


def view_heading(face_bearing_deg, pano_heading_deg):
    """Tile heading relative to the pano's centre column (positive = clockwise/right), as
    footway_segmentation.tile_rotation expects; GSV and Mapillary both put the compass
    heading at the centre column."""
    return (face_bearing_deg - pano_heading_deg) % 360.0


def select_panos(cands, length_m, max_n=MAX_PANOS_PER_STREET, spacing=MIN_PANO_SPACING_M,
                 end_buffer=END_BUFFER_M, max_cross=MAX_CROSS_TRACK_M):
    """Mid-block candidates, newest capture first, greedily spaced along the edge.

    `cands`: dicts with pano_id, s_m, cross_m, date ('YYYY-MM' or sortable string).

        >>> c = [dict(pano_id='a', s_m=20, cross_m=1, date='2019-05'),
        ...      dict(pano_id='b', s_m=24, cross_m=1, date='2024-07'),
        ...      dict(pano_id='c', s_m=5, cross_m=1, date='2025-01')]
        >>> [x['pano_id'] for x in select_panos(c, 60)]
        ['b']
    """
    ok = [c for c in cands
          if end_buffer < c['s_m'] < length_m - end_buffer and abs(c['cross_m']) <= max_cross]
    ok.sort(key=lambda c: (c['date'] or '', c['pano_id']), reverse=True)
    out = []
    for c in ok:
        if all(abs(c['s_m'] - o['s_m']) >= spacing for o in out):
            out.append(c)
        if len(out) >= max_n:
            break
    return out


# --- view and band geometry ---------------------------------------------------------------

def band_ground_points(lateral=BAND_LATERAL_M, along=BAND_ALONG_M, step=BAND_STEP_M):
    """(a_m along the street, d_m lateral) of a uniform ground grid; equal area each."""
    d = np.arange(lateral[0], lateral[1] + 1e-9, step)
    a = np.arange(-along, along + 1e-9, step)
    A, D = np.meshgrid(a, d)
    return A.ravel(), D.ravel()


def band_pixels(pitch_deg=VIEW_PITCH_DEG, fov_deg=VIEW_FOV_DEG, size=VIEW_SIZE,
                cam_h=CAMERA_HEIGHT_M, lateral=BAND_LATERAL_M, along=BAND_ALONG_M,
                step=BAND_STEP_M):
    """Pixels of a lateral view that see the ground band, one per in-frustum ground sample.

    Returns dict(row, col, d, a) arrays. The view looks along the face normal (heading 0
    in its own frame), pitched by pitch_deg; ground is flat, cam_h below the camera.
    Frame and rotation are footway_segmentation's (X right, Y up, Z forward)."""
    from footway_segmentation import tile_rotation
    a, d = band_ground_points(lateral, along, step)
    v = np.stack([a, np.full_like(a, -cam_h), d], axis=-1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    cam = v @ tile_rotation(0.0, pitch_deg)          # world -> camera (R^T v)
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    cz = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = f * cam[:, 0] / cz + size / 2
        w = size / 2 - f * cam[:, 1] / cz
    col = np.floor(np.where(cz > 0, u, -1)).astype(np.int64)
    row = np.floor(np.where(cz > 0, w, -1)).astype(np.int64)
    ok = (cz > 0) & (col >= 0) & (col < size) & (row >= 0) & (row < size)
    return {'row': row[ok], 'col': col[ok], 'd': d[ok], 'a': a[ok]}


def class_ids(id2label, names):
    have = {v: int(k) for k, v in id2label.items()}
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError(f'classes not in the model vocabulary: {missing}')
    return [have[n] for n in names]


def view_score(hist, id2label, min_obs_frac=OBSERVED_MIN_FRAC, occluders=OCCLUDER_CLASSES):
    """Per pano-side numbers from a band class histogram (list indexed by class id)."""
    h = np.asarray(hist, dtype=float)
    n_band = float(h.sum())
    occ = float(h[class_ids(id2label, occluders)].sum())
    walk = float(h[class_ids(id2label, WALKWAY_CLASSES)].sum())
    curb = float(h[class_ids(id2label, CURB_CLASSES)].sum())
    sidewalk_only = float(h[class_ids(id2label, ('Sidewalk',))].sum())
    n_obs = n_band - occ
    obs_frac = n_obs / n_band if n_band else 0.0
    return {
        'n_band': int(n_band), 'occluded_frac': (occ / n_band) if n_band else None,
        'observed': bool(n_band > 0 and obs_frac >= min_obs_frac),
        'walk_share': walk / n_obs if n_obs > 0 else None,
        'sidewalk_share': sidewalk_only / n_obs if n_obs > 0 else None,
        'curb_share': curb / n_obs if n_obs > 0 else None,
        'walkcurb_share': (walk + curb) / n_obs if n_obs > 0 else None,
    }


# --- pooling, metrics, decision -------------------------------------------------------------

def pool(values, how=POOL_PRIMARY, min_n=MIN_OBSERVED_PANOS):
    """Pool observed pano-side shares into a face score, or None if unobserved.

        >>> pool([0.0, 0.0, 0.6, 0.1])        # one clear view through a gap counts
        0.225
        >>> pool([0.4]) is None
        True
    """
    vals = [v for v in values if v is not None]
    if len(vals) < min_n:
        return None
    if how == 'q75':
        return float(np.percentile(vals, 75))
    if how == 'max':
        return float(max(vals))
    if how == 'median':
        return float(np.median(vals))
    if how == 'mean':
        return float(np.mean(vals))
    raise ValueError(how)


def weighted_auc(scores, is_absent, weights=None):
    """AUC for SDOT-absent (positive) vs present where a LOWER score means absent:
    P(score_absent < score_present), ties 1/2, each pair weighted by w_i * w_j.

        >>> weighted_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
        1.0
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_absent, dtype=bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    sa, wa, sp, wp = s[y], w[y], s[~y], w[~y]
    if len(sa) == 0 or len(sp) == 0:
        return None
    lt = (sa[:, None] < sp[None, :]).astype(float) + 0.5 * (sa[:, None] == sp[None, :])
    ww = wa[:, None] * wp[None, :]
    return float((lt * ww).sum() / ww.sum())


def threshold_for_recall(scores, is_absent, weights=None, target=TARGET_RECALL):
    """Smallest tau such that the (weighted) recall of `absent` (score <= tau) >= target."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_absent, dtype=bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    tot = w[y].sum()
    if tot == 0:
        return None
    for tau in np.unique(s[y]):
        if w[y & (s <= tau)].sum() / tot >= target - 1e-12:
            return float(tau)
    return float(s[y].max())


def precision_recall(pred_absent, is_absent, weights=None):
    p = np.asarray(pred_absent, dtype=bool)
    y = np.asarray(is_absent, dtype=bool)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    tp = w[p & y].sum()
    prec = tp / w[p].sum() if w[p].sum() > 0 else None
    rec = tp / w[y].sum() if w[y].sum() > 0 else None
    return (None if prec is None else float(prec)), (None if rec is None else float(rec))


def decide(precision, disc_auc, swap_disc_auc, beats_prior, coverage):
    """The advisors' rule. Returns (verdict, {clause: bool}).

        >>> decide(0.85, 0.85, 0.2, True, 0.9)[0]
        'CONTINUE'
        >>> decide(0.85, 0.65, 0.35, True, 0.9)[0]
        'STOP'
        >>> decide(0.75, 0.85, 0.2, True, 0.9)[0]
        'INCONCLUSIVE'
    """
    def ok(x):
        return x is not None
    stop = {
        'disc_auc_below_0.70': ok(disc_auc) and disc_auc < STOP_DISC_AUC,
        'disc_auc_within_0.05_of_swap_null': (ok(disc_auc) and ok(swap_disc_auc)
                                              and abs(disc_auc - swap_disc_auc) <= STOP_NULL_MARGIN),
        'precision_below_0.70': ok(precision) and precision < STOP_PRECISION,
        'coverage_below_0.50': ok(coverage) and coverage < STOP_COVERAGE,
    }
    cont = {
        'precision_ge_0.80': ok(precision) and precision >= CONTINUE_PRECISION,
        'disc_auc_ge_0.80': ok(disc_auc) and disc_auc >= CONTINUE_DISC_AUC,
        'beats_waytype_prior': bool(beats_prior),
        'swap_null_collapses': ok(swap_disc_auc) and swap_disc_auc <= SWAP_COLLAPSED_MAX,
    }
    if any(stop.values()):
        verdict = 'STOP'
    elif all(cont.values()):
        verdict = 'CONTINUE'
    else:
        verdict = 'INCONCLUSIVE'
    return verdict, {**{f'stop:{k}': v for k, v in stop.items()},
                     **{f'continue:{k}': v for k, v in cont.items()}}


# --- sample -----------------------------------------------------------------------------------

def load_streets(path):
    g = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for ft in g['features']:
        p = ft['properties']
        if ft['geometry']['type'] != 'LineString':
            continue
        out[int(p['street_edge_id'])] = {
            'coords': [tuple(c) for c in ft['geometry']['coordinates']],
            'status': p.get('status'), 'way_type': p.get('way_type'),
            'osm_way_id': p.get('osm_way_id'), 'osm_ways': p.get('osm_ways') or [],
            'region_name': p.get('region_name')}
    return out


def eligible_faces(sp_rows, streets):
    """In-scope #5222 faces on eligible way types, open edges >= MIN_EDGE_LEN_M long."""
    out = []
    for r in sp_rows:
        eid = int(r['street_edge_id'])
        st = streets.get(eid)
        if (r['in_scope'] != 't' or r['truth2'] not in ('none', 'present')
                or r['way_type'] not in ELIGIBLE_WAY_TYPES or st is None
                or st['status'] != 'open' or float(r['len_m']) < MIN_EDGE_LEN_M):
            continue
        out.append(r)
    return out


def stratify(faces):
    """{stratum: [street ids (D) or face rows (A, P)]}; D = both faces eligible, verdicts differ."""
    by_edge = defaultdict(dict)
    for r in faces:
        by_edge[int(r['street_edge_id'])][r['ps_side']] = r
    disc = sorted(e for e, f in by_edge.items()
                  if len(f) == 2 and f['left']['truth2'] != f['right']['truth2'])
    dset = set(disc)
    a = [r for r in faces if int(r['street_edge_id']) not in dset and r['truth2'] == 'none']
    p = [r for r in faces if int(r['street_edge_id']) not in dset and r['truth2'] == 'present']
    return by_edge, disc, a, p


def draw_sample(faces, seed=SEED, n_d=N_DISCORDANT_STREETS, n_a=N_ABSENT_FACES,
                n_p=N_PRESENT_FACES, dev_fraction=DEV_FRACTION):
    by_edge, disc, a, p = stratify(faces)
    rng = random.Random(seed)
    key = lambda r: (int(r['street_edge_id']), r['ps_side'])  # noqa: E731
    d_streets = sorted(rng.sample(disc, min(n_d, len(disc))))
    a_s = sorted(rng.sample(sorted(a, key=key), min(n_a, len(a))), key=key)
    p_s = sorted(rng.sample(sorted(p, key=key), min(n_p, len(p))), key=key)
    pop = {'D': 2 * len(disc), 'A': len(a), 'P': len(p)}
    smp = {'D': 2 * len(d_streets), 'A': len(a_s), 'P': len(p_s)}
    rows = []
    for e in d_streets:
        for side in SIDES:
            rows.append(('D', by_edge[e][side]))
    rows += [('A', r) for r in a_s] + [('P', r) for r in p_s]
    # dev split: seeded, by street, within stratum
    dev = set()
    for h in STRATA:
        streets_h = sorted({int(r['street_edge_id']) for s, r in rows if s == h})
        k = round(dev_fraction * len(streets_h))
        dev |= set(rng.sample(streets_h, k))
    out = []
    for h, r in rows:
        eid = int(r['street_edge_id'])
        out.append({'face_uid': face_uid(eid, r['ps_side']), 'city': CITY,
                    'street_edge_id': eid, 'street_side': r['ps_side'], 'stratum': h,
                    'weight': pop[h] / smp[h], 'split': 'dev' if eid in dev else 'test',
                    'truth2': r['truth2'], 'truth3': r['truth3'], 'way_type': r['way_type'],
                    'len_m': round(float(r['len_m']), 2)})
    return out, pop, smp


def cmd_sample(args):
    sp = read_csv(args.sp_face)
    streets = load_streets(args.streets)
    faces = eligible_faces(sp, streets)
    sample, pop, smp = draw_sample(faces)
    uids = [r['face_uid'] for r in sample]
    assert len(uids) == len(set(uids)), 'face_uid must be unique'
    out = Path(args.out)
    write_csv(out / 'face_sample.csv', sample)
    edges = sorted({r['street_edge_id'] for r in sample})
    write_json(out / 'edges.geojson', {
        'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'geometry': {'type': 'LineString',
                                             'coordinates': streets[e]['coords']},
             'properties': {'street_edge_id': e, 'way_type': streets[e]['way_type'],
                            'osm_way_id': streets[e]['osm_way_id'],
                            'osm_ways': streets[e]['osm_ways'],
                            'region_name': streets[e]['region_name']}} for e in edges]})
    # way-type prior from eligible faces on streets NOT in the test split
    test_streets = {r['street_edge_id'] for r in sample if r['split'] == 'test'}
    cnt = defaultdict(lambda: [0, 0])
    for r in faces:
        if int(r['street_edge_id']) in test_streets:
            continue
        cnt[r['way_type']][0] += r['truth2'] == 'none'
        cnt[r['way_type']][1] += 1
    prior = {k: {'absent': v[0], 'n': v[1], 'p_absent': v[0] / v[1]} for k, v in cnt.items()}
    write_json(out / 'population.json', {
        'seed': SEED, 'n_eligible_faces': len(faces), 'population_by_stratum': pop,
        'sample_by_stratum': smp, 'waytype_prior': prior,
        'n_sampled_streets': len(edges),
        'split_counts': {f'{k[0]}:{k[1]}': v for k, v in
                         sorted(Counter((r['stratum'], r['split']) for r in sample).items())},
        'inputs': {'sp_face_sha256': sha256_file(args.sp_face),
                   'streets_sha256': sha256_file(args.streets)}})
    print(f'eligible faces {len(faces)}; population {pop}; sample {smp}; '
          f'streets {len(edges)}')


# --- GSV views (klone) ---------------------------------------------------------------------------

def load_edges_geojson(path):
    g = json.loads(Path(path).read_text(encoding='utf-8'))
    return {int(ft['properties']['street_edge_id']): ft for ft in g['features']}


def tiles_for_edge(coords, zoom, buffer_m=30.0):
    import sources.gsv  # noqa: F401  (stubs pyexiv2 before streetlevel loads)
    from streetlevel.geo import wgs84_to_tile_coord
    lats = [c[1] for c in coords]
    lngs = [c[0] for c in coords]
    dlat = buffer_m / 111320.0
    dlng = buffer_m / (111320.0 * math.cos(math.radians(sum(lats) / len(lats))))
    x0, y0 = wgs84_to_tile_coord(max(lats) + dlat, min(lngs) - dlng, zoom)
    x1, y1 = wgs84_to_tile_coord(min(lats) - dlat, max(lngs) + dlng, zoom)
    return {(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)
            for y in range(min(y0, y1), max(y0, y1) + 1)}


GRID_CELL_DEG = 0.0005     # ~55 m north-south: a coarse prefilter before exact projection


def grid_index(points, cell=GRID_CELL_DEG):
    """{(i, j): [id]} for {id: (lat, lng, ...)}."""
    idx = defaultdict(list)
    for pid, v in points.items():
        idx[(int(math.floor(v[0] / cell)), int(math.floor(v[1] / cell)))].append(pid)
    return idx


def ids_near_edge(idx, coords, buffer_m=30.0, cell=GRID_CELL_DEG):
    """Ids in grid cells overlapping the edge's bbox grown by buffer_m (a superset)."""
    lats = [c[1] for c in coords]
    lngs = [c[0] for c in coords]
    dlat = buffer_m / 111320.0
    dlng = buffer_m / (111320.0 * math.cos(math.radians(sum(lats) / len(lats))))
    out = []
    for i in range(int(math.floor((min(lats) - dlat) / cell)),
                   int(math.floor((max(lats) + dlat) / cell)) + 1):
        for j in range(int(math.floor((min(lngs) - dlng) / cell)),
                       int(math.floor((max(lngs) + dlng) / cell)) + 1):
            out += idx.get((i, j), [])
    return out


def render_views(img_rgb, specs):
    """{name: uint8 view} for [(name, heading_rel_deg)] from a (H, W, 3) equirect array,
    with footway_segmentation's tile geometry and bilinear sampler."""
    from footway_segmentation import tile_pixel_to_equirect, sample_bilinear
    out = {}
    for name, heading in specs:
        x, y = tile_pixel_to_equirect(heading, VIEW_PITCH_DEG, VIEW_SIZE, VIEW_FOV_DEG)
        out[name] = np.clip(np.rint(sample_bilinear(img_rgb, x, y)), 0, 255).astype(np.uint8)
    return out


def view_rows_for_pano(edge_id, geom, pano):
    """Both faces' view specs for one selected pano (a dict with lat, lng, heading_deg)."""
    s, cross, tangent = geom.project(pano['lat'], pano['lng'])
    rows = []
    for side in SIDES:
        fb = face_bearing(tangent, side)
        rows.append({'view_id': f"{pano['source']}_{pano['pano_id']}_{edge_id}_{side}",
                     'source': pano['source'], 'pano_id': pano['pano_id'],
                     'street_edge_id': edge_id, 'street_side': side,
                     'face_uid': face_uid(edge_id, side), 's_m': round(s, 2),
                     'cross_m': round(cross, 2), 'tangent_deg': round(tangent, 2),
                     'face_bearing_deg': round(fb, 2),
                     'view_heading_deg': round(view_heading(fb, pano['heading_deg']), 3),
                     'date': pano['date'], 'pano_heading_deg': round(pano['heading_deg'], 3),
                     'pano_lat': pano['lat'], 'pano_lng': pano['lng']})
    return rows


def cmd_gsv_views(args):
    from PIL import Image
    import sources.gsv as gsv          # first: it stubs pyexiv2 where glibc is too old
    from streetlevel import streetview
    from panorama import fetch_panorama
    work = Path(args.work)
    edges = load_edges_geojson(args.edges)
    geoms = {e: EdgeGeom(ft['geometry']['coordinates']) for e, ft in edges.items()}
    # 1. coverage tiles (cached)
    cache = work / 'gsv_coverage.json'
    cov = json.loads(cache.read_text()) if cache.exists() else {}
    need = set()
    for e, ft in edges.items():
        need |= tiles_for_edge(ft['geometry']['coordinates'], gsv.COVERAGE_TILE_ZOOM)
    todo = sorted(t for t in need if f'{t[0]},{t[1]}' not in cov)
    print(f'gsv-views: {len(edges)} edges, {len(need)} coverage tiles, {len(todo)} to fetch',
          flush=True)

    def tile(t):
        for attempt in range(3):
            try:
                return t, [(p.id, p.lat, p.lon) for p in streetview.get_coverage_tile(*t)]
            except Exception:
                time.sleep(2 * (attempt + 1))
        return t, None
    with ThreadPoolExecutor(args.workers) as ex:
        for t, panos in ex.map(tile, todo):
            if panos is not None:
                cov[f'{t[0]},{t[1]}'] = panos
    cache.write_text(json.dumps(cov))
    failed_tiles = len([t for t in need if f'{t[0]},{t[1]}' not in cov])
    # 2. mid-block candidates per edge
    allp = {}
    for v in cov.values():
        for pid, lat, lng in v:
            allp[pid] = (lat, lng)
    cands = defaultdict(list)
    gidx = grid_index(allp)
    for e, g in geoms.items():
        for pid in ids_near_edge(gidx, edges[e]['geometry']['coordinates']):
            lat, lng = allp[pid]
            s, cross, _ = g.project(lat, lng)
            if END_BUFFER_M < s < g.length - END_BUFFER_M and abs(cross) <= MAX_CROSS_TRACK_M:
                cands[e].append({'pano_id': pid, 'lat': lat, 'lng': lng, 's_m': s,
                                 'cross_m': cross})
    # 3. metadata for candidates (cached): date, heading, source, depth camera height
    mcache = work / 'gsv_meta.json'
    meta = json.loads(mcache.read_text()) if mcache.exists() else {}
    need_ids = sorted({c['pano_id'] for v in cands.values() for c in v} - set(meta))
    print(f'gsv-views: {sum(len(v) for v in cands.values())} candidates, '
          f'{len(need_ids)} metadata to fetch', flush=True)

    def md(pid):
        m, depth_fields = gsv.fetch_metadata_with_retry(pid)
        if m is None:
            return pid, None
        problem = gsv._metadata_problem(m)
        return pid, {'date': f'{m.date.year}-{m.date.month:02d}' if m.date else None,
                     'heading_deg': math.degrees(m.heading), 'pitch_deg': math.degrees(m.pitch),
                     'roll_deg': math.degrees(m.roll), 'source': m.source, 'problem': problem,
                     'camera_height_m': (depth_fields or {}).get('camera_height_m')}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (pid, m) in enumerate(ex.map(md, need_ids)):
            if m is not None:
                meta[pid] = m
            if (i + 1) % 500 == 0:
                mcache.write_text(json.dumps(meta))
                print(f'  metadata {i + 1}/{len(need_ids)}', flush=True)
    mcache.write_text(json.dumps(meta))
    # 4. selection
    selected = []
    for e in sorted(cands):
        usable = [dict(c, date=meta[c['pano_id']]['date']) for c in cands[e]
                  if c['pano_id'] in meta and not meta[c['pano_id']]['problem']]
        for c in select_panos(usable, geoms[e].length):
            m = meta[c['pano_id']]
            selected.append({'street_edge_id': e, 'source': 'gsv', 'pano_id': c['pano_id'],
                             'lat': c['lat'], 'lng': c['lng'], 'date': m['date'],
                             'heading_deg': m['heading_deg'], 'pitch_deg': m['pitch_deg'],
                             'roll_deg': m['roll_deg'], 'gsv_source': m['source'],
                             'camera_height_m': m['camera_height_m'],
                             'n_candidates': len(usable)})
    # 5. download and render (resumable: a pano whose two views exist is skipped)
    vdir = work / 'views' / 'gsv'
    vdir.mkdir(parents=True, exist_ok=True)
    view_rows = []
    jobs = []
    for p in selected:
        rows = view_rows_for_pano(p['street_edge_id'], geoms[p['street_edge_id']], p)
        view_rows += rows
        if not all((vdir / f"{r['view_id']}.jpg").exists() for r in rows):
            jobs.append((p, rows))
    print(f'gsv-views: {len(selected)} selected panos, {len(jobs)} to download', flush=True)
    by_pano = defaultdict(list)
    for p, rows in jobs:
        by_pano[p['pano_id']].append(rows)

    def dl(pid):
        m, _ = gsv.fetch_metadata_with_retry(pid)
        if m is None:
            return pid, 'metadata'
        img = fetch_panorama(m)
        if img is None:
            return pid, 'image'
        arr = np.asarray(img.convert('RGB'))
        specs = [(r['view_id'], r['view_heading_deg']) for rows in by_pano[pid] for r in rows]
        for name, v in render_views(arr, specs).items():
            Image.fromarray(v).save(vdir / f'{name}.jpg', quality=95)
        return pid, None
    failures = Counter()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(dl, pid) for pid in sorted(by_pano)]
        for i, fu in enumerate(as_completed(futs)):
            pid, err = fu.result()
            if err:
                failures[err] += 1
            if (i + 1) % 200 == 0:
                print(f'  downloaded {i + 1}/{len(futs)}', flush=True)
    have = [r for r in view_rows if (vdir / f"{r['view_id']}.jpg").exists()]
    write_csv(work / 'gsv_panos.csv', selected)
    write_csv(work / 'views_gsv.csv', have)
    write_json(work / 'gsv_views_summary.json', {
        'edges': len(edges), 'coverage_tiles': len(need), 'failed_tiles': failed_tiles,
        'candidates': sum(len(v) for v in cands.values()), 'selected_panos': len(selected),
        'views': len(have), 'download_failures': dict(failures)})
    print(f'gsv-views: done, {len(have)} views; failures {dict(failures)}')


# --- Mapillary views (Mac: the token never leaves it) -----------------------------------------

def cmd_mly_views(args):
    import requests
    from PIL import Image
    from dotenv import load_dotenv
    import sources.mapillary as mly
    load_dotenv(args.env_file)
    mly.prepare()
    work = Path(args.work)
    edges = load_edges_geojson(args.edges)
    geoms = {e: EdgeGeom(ft['geometry']['coordinates']) for e, ft in edges.items()}
    need = set()
    for ft in edges.values():
        need |= tiles_for_edge(ft['geometry']['coordinates'], mly.COVERAGE_TILE_ZOOM)
    allp = {}
    failed_tiles = 0
    for t in sorted(need):
        for attempt in range(3):
            try:
                r = requests.get(mly.TILE_URL.format(z=mly.COVERAGE_TILE_ZOOM, x=t[0], y=t[1]),
                                 params={'access_token': os.environ[mly.TOKEN_ENV_VAR]},
                                 timeout=60)
                if r.status_code == 404:
                    break
                r.raise_for_status()
                layer = mly._decode_tile(r.content).get('image') or {}
                ext = layer.get('extent', 4096)
                for ftr in layer.get('features', []):
                    pr = ftr.get('properties', {})
                    if not pr.get('is_pano'):
                        continue
                    lon, lat = mly.tile_point_to_lonlat(*ftr['geometry']['coordinates'], t[0],
                                                        t[1], mly.COVERAGE_TILE_ZOOM, ext)
                    allp[str(pr['id'])] = (lat, lon, pr.get('captured_at', 0))
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            failed_tiles += 1
    print(f'mly-views: {len(need)} z14 tiles, {len(allp)} 360 images, '
          f'{failed_tiles} failed tiles', flush=True)
    # Pre-selection on the tile position and capture time (cheap; Seattle has millions of
    # 360 images): the select_panos rule with a looser corridor and up to 3x the final
    # count, then Graph metadata for the SfM position and compass, then the final rule.
    pre = defaultdict(list)
    gidx = grid_index(allp)
    for e, g in geoms.items():
        cands = []
        for pid in ids_near_edge(gidx, edges[e]['geometry']['coordinates']):
            lat, lng, cap = allp[pid]
            s, cross, _ = g.project(lat, lng)
            cands.append({'pano_id': pid, 's_m': s, 'cross_m': cross,
                          'date': f'{int(cap or 0):015d}'})
        pre[e] = [c['pano_id'] for c in select_panos(
            cands, g.length, max_n=3 * MAX_PANOS_PER_STREET, end_buffer=END_BUFFER_M - 3,
            max_cross=MAX_CROSS_TRACK_M + 5)]
    ids = sorted({p for v in pre.values() for p in v})
    print(f'mly-views: {len(ids)} images near sampled edges; fetching Graph metadata', flush=True)
    meta = {}

    def md(pid):
        m, gone = mly.fetch_image_metadata(pid)
        return pid, m
    with ThreadPoolExecutor(8) as ex:
        for pid, m in ex.map(md, ids):
            if m is not None:
                meta[pid] = m
    selected = []
    for e in sorted(pre):
        usable = []
        for pid in pre[e]:
            m = meta.get(pid)
            if not m:
                continue
            coords = mly.position_coordinates(m)
            heading = mly._compass_angle(m)
            w, h = m.get('width'), m.get('height')
            if not coords or heading is None or not w or not h or abs(w / h - 2) > 0.02:
                continue
            s, cross, _ = geoms[e].project(coords[1], coords[0])
            date = time.strftime('%Y-%m', time.gmtime(m['captured_at'] / 1000)) \
                if m.get('captured_at') else None
            usable.append({'pano_id': pid, 'lat': coords[1], 'lng': coords[0], 's_m': s,
                           'cross_m': cross, 'date': date, 'heading_deg': float(heading),
                           'make': m.get('make'), 'model': m.get('model')})
        for c in select_panos(usable, geoms[e].length):
            selected.append(dict(c, street_edge_id=e, source='mly', n_candidates=len(usable)))
    vdir = work / 'views' / 'mly'
    vdir.mkdir(parents=True, exist_ok=True)
    view_rows, failures = [], Counter()
    for i, p in enumerate(selected):
        rows = view_rows_for_pano(p['street_edge_id'], geoms[p['street_edge_id']], p)
        if not all((vdir / f"{r['view_id']}.jpg").exists() for r in rows):
            url = (meta[p['pano_id']] or {}).get('thumb_original_url')
            img = mly._download_image(url)[0] if url else None    # 4096x2048 RGB or None
            if img is None:
                failures['image'] += 1
                continue
            specs = [(r['view_id'], r['view_heading_deg']) for r in rows]
            for name, v in render_views(np.asarray(img), specs).items():
                Image.fromarray(v).save(vdir / f'{name}.jpg', quality=95)
        view_rows += rows
        if (i + 1) % 50 == 0:
            print(f'  mly {i + 1}/{len(selected)}', flush=True)
    write_csv(work / 'mly_panos.csv', selected)
    write_csv(work / 'views_mly.csv', [r for r in view_rows
                                       if (vdir / f"{r['view_id']}.jpg").exists()])
    write_json(work / 'mly_views_summary.json', {
        'tiles': len(need), 'failed_tiles': failed_tiles, 'images_360_in_tiles': len(allp),
        'near_edges': len(ids), 'metadata': len(meta), 'selected_panos': len(selected),
        'edges_with_panos': len({p['street_edge_id'] for p in selected}),
        'failures': dict(failures)})
    print(f'mly-views: {len(selected)} panos on '
          f"{len({p['street_edge_id'] for p in selected})} edges; failures {dict(failures)}")


# --- bands (klone, after segmentation) -----------------------------------------------------------

def cmd_bands(args):
    from PIL import Image
    work = Path(args.work)
    seg_manifest = json.loads((work / 'labels' / 'segment_manifest.json').read_text())
    id2label = {int(k): v for k, v in seg_manifest['id2label'].items()}
    n_cls = max(id2label) + 1
    bp = band_pixels()
    near = bp['d'] < BAND_NEAR_FAR_SPLIT_M
    out = work / 'view_bands.jsonl'
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for png in sorted((work / 'labels').rglob('*.png')):
            lab = np.asarray(Image.open(png))
            v = lab[bp['row'], bp['col']].astype(np.int64)
            f.write(json.dumps({
                'view_id': png.stem, 'source': png.parent.name,
                'all': np.bincount(v, minlength=n_cls).tolist(),
                'near': np.bincount(v[near], minlength=n_cls).tolist(),
                'far': np.bincount(v[~near], minlength=n_cls).tolist()}) + '\n')
            n += 1
    write_json(work / 'bands_meta.json', {
        'id2label': {str(k): v for k, v in id2label.items()}, 'n_views': n,
        'band_samples': int(len(bp['row'])), 'near_samples': int(near.sum()),
        'model_id': seg_manifest.get('model_id'),
        'model_revision': seg_manifest.get('model_revision'),
        'segment_device': seg_manifest.get('device')})
    print(f'bands: {n} views, {len(bp["row"])} band samples per view -> {out}')


# --- OSM sidewalk tags (Mac) ----------------------------------------------------------------------

OVERPASS_URLS = ['https://overpass-api.de/api/interpreter',
                 'https://overpass.private.coffee/api/interpreter']


def osm_side_value(tags, osm_side):
    """'present' / 'absent' / None for one side ('left'/'right' of the OSM way)."""
    v = tags.get(f'sidewalk:{osm_side}') or tags.get('sidewalk:both')
    if v:
        return 'present' if v in ('yes', 'separate') else ('absent' if v == 'no' else None)
    s = tags.get('sidewalk')
    if s is None:
        return None
    if s in ('no', 'none'):
        return 'absent'
    if s in ('both', 'separate', 'yes'):
        return 'present'
    if s in ('left', 'right'):
        return 'present' if s == osm_side else 'absent'
    return None


def cmd_osm_tags(args):
    import requests
    edges = load_edges_geojson(args.edges)
    ways = sorted({int(ft['properties']['osm_way_id']) for ft in edges.values()
                   if ft['properties'].get('osm_way_id')})
    q = f'[out:json][timeout:120];way(id:{",".join(map(str, ways))});out tags;'
    for attempt, url in enumerate(OVERPASS_URLS * 2):
        try:
            r = requests.post(url, data={'data': q}, timeout=180,
                              headers={'User-Agent': 'nosidewalk-zeroshot-study/0.1'})
            r.raise_for_status()
            elements = r.json()['elements']
            break
        except Exception as e:
            print(f'osm-tags: {url} failed ({e}); retrying', flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit('osm-tags: every Overpass endpoint failed')
    tags = {int(el['id']): el.get('tags', {}) for el in elements}
    missing = [w for w in ways if w not in tags]
    if missing:   # a partial answer is refused, never silently read as "untagged"
        raise SystemExit(f'osm-tags: {len(missing)} ways missing from the Overpass answer')
    rows = []
    for e, ft in sorted(edges.items()):
        wid = ft['properties'].get('osm_way_id')
        same = next((w.get('same_direction', True) for w in ft['properties'].get('osm_ways', [])
                     if w.get('osm_way_id') == wid), True)
        t = tags.get(int(wid), {}) if wid else {}
        for side in SIDES:
            osm_side = side if same else ('right' if side == 'left' else 'left')
            rows.append({'face_uid': face_uid(e, side), 'osm_way_id': wid,
                         'same_direction': same, 'osm_sidewalk': osm_side_value(t, osm_side),
                         'raw': json.dumps({k: v for k, v in t.items()
                                            if k.startswith('sidewalk')}, sort_keys=True)})
    write_csv(Path(args.out) / 'osm_sidewalk_tags.csv', rows)
    print(f'osm-tags: {len(ways)} ways, {sum(1 for r in rows if r["osm_sidewalk"])} '
          f'of {len(rows)} faces tagged')


# --- score (Mac) -------------------------------------------------------------------------------------

def face_scores_from_views(views, bands, id2label, pool_how=POOL_PRIMARY, band='all',
                           share='walk_share', occluders=OCCLUDER_CLASSES):
    """{face_uid: dict(score, n_views, n_observed, occluded_mean, ...)} for one source."""
    per_face = defaultdict(list)
    for v in views:
        b = bands.get(v['view_id'])
        if b is None:
            continue
        per_face[v['face_uid']].append((v, view_score(b[band], id2label,
                                                      occluders=occluders)))
    out = {}
    for fu, items in per_face.items():
        obs = [s[share] for _, s in items if s['observed']]
        occ = [s['occluded_frac'] for _, s in items if s['occluded_frac'] is not None]
        dates = sorted((v['date'] or '') for v, s in items if s['observed'])
        out[fu] = {'score': pool(obs, pool_how), 'n_views': len(items), 'n_observed': len(obs),
                   'occluded_mean': float(np.mean(occ)) if occ else None,
                   'newest_observed_date': dates[-1] if dates else None,
                   'shares': obs}
    return out


def other_face(uid):
    city, e, side = uid.split(':')
    return f'{city}:{e}:{"right" if side == "left" else "left"}'


def _metric_block(rows, score_key, tau, weighted=True):
    """AUC, discordant AUC, precision/recall at tau over rows with a non-null score."""
    rr = [r for r in rows if r.get(score_key) is not None]
    if not rr:
        return {}
    s = [r[score_key] for r in rr]
    y = [r['truth2'] == 'none' for r in rr]
    w = [r['weight'] if weighted else 1.0 for r in rr]
    d = [r for r in rr if r['stratum'] == 'D']
    out = {'n': len(rr), 'auc': weighted_auc(s, y, w),
           'disc_auc': weighted_auc([r[score_key] for r in d],
                                    [r['truth2'] == 'none' for r in d]) if d else None,
           'n_disc': len(d)}
    if tau is not None:
        prec, rec = precision_recall([x <= tau for x in s], y, w)
        out.update(precision=prec, recall=rec)
    return out


def bootstrap(rows, fn, n_boot=N_BOOT, seed=SEED):
    """Stratified cluster bootstrap: resample streets with replacement within each stratum."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r['stratum']][r['street_edge_id']].append(r)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        smp = []
        for h, streets in by.items():
            keys = sorted(streets)
            for k in rng.integers(0, len(keys), len(keys)):
                smp += streets[keys[k]]
        reps.append(fn(smp))
    return reps


def ci(reps, key):
    v = [r[key] for r in reps if r.get(key) is not None]
    if len(v) < 10:
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def cmd_score(args):  # noqa: C901  (one linear report)
    dd = Path(args.data)
    work = Path(args.work)
    sample = read_csv(dd / 'face_sample.csv')
    for r in sample:
        r['weight'] = float(r['weight'])
        r['street_edge_id'] = int(r['street_edge_id'])
    pop = json.loads((dd / 'population.json').read_text())
    bmeta = json.loads((work / 'bands_meta.json').read_text())
    id2label = {int(k): v for k, v in bmeta['id2label'].items()}
    bands = {}
    bands_path = work / 'view_bands.jsonl'
    if not bands_path.exists():
        bands_path = work / 'view_bands.jsonl.gz'     # the committed copy
    with open_text(bands_path) as f:
        for line in f:
            b = json.loads(line)
            bands[b['view_id']] = b
    views = {src: read_csv(work / f'views_{src}.csv') for src in ('gsv', 'mly')
             if (work / f'views_{src}.csv').exists()}
    sp = {face_uid(r['street_edge_id'], r['ps_side']): r for r in read_csv(args.sp_face)}
    osm = {r['face_uid']: r for r in read_csv(dd / 'osm_sidewalk_tags.csv')} \
        if (dd / 'osm_sidewalk_tags.csv').exists() else {}
    prior = {k: v['p_absent'] for k, v in pop['waytype_prior'].items()}

    variants = {}
    for src, vv in views.items():
        for how in POOLS:
            variants[(src, how, 'all', 'walk_share')] = face_scores_from_views(
                vv, bands, id2label, how)
        for share in ('sidewalk_share', 'curb_share', 'walkcurb_share'):
            variants[(src, POOL_PRIMARY, 'all', share)] = face_scores_from_views(
                vv, bands, id2label, POOL_PRIMARY, share=share)
        for band in ('near', 'far'):
            variants[(src, POOL_PRIMARY, band, 'walk_share')] = face_scores_from_views(
                vv, bands, id2label, POOL_PRIMARY, band=band)
        # amendment A1 (pre-score): Vegetation also treated as an occluder
        variants[(src, POOL_PRIMARY, 'all', 'walk_share_vegocc')] = face_scores_from_views(
            vv, bands, id2label, POOL_PRIMARY,
            occluders=OCCLUDER_CLASSES + VEGETATION_OCCLUDER)
    prim = variants[('gsv', POOL_PRIMARY, 'all', 'walk_share')]

    # attach scores to faces; side-swap and street-mean nulls use the other face's views
    rows = []
    for r in sample:
        uid = r['face_uid']
        f = prim.get(uid, {})
        o = prim.get(other_face(uid), {})
        spr = sp[uid]
        row = dict(r)
        row.update(score=f.get('score'), n_views=f.get('n_views', 0),
                   n_observed=f.get('n_observed', 0), occluded_mean=f.get('occluded_mean'),
                   newest_observed_date=f.get('newest_observed_date'),
                   swap_score=o.get('score'),
                   street_mean_score=(None if f.get('score') is None or o.get('score') is None
                                      else (f['score'] + o['score']) / 2),
                   prior_score=-prior[r['way_type']],     # lower = absent, like the method
                   ps_m1=spr['m1'], ps_m2_k2=spr['m2_k2'], ps_ns=int(spr['ns']),
                   ps_audits=int(spr['audits']), arterial=spr['arterial_label'],
                   osm_sidewalk=(osm.get(uid) or {}).get('osm_sidewalk') or None,
                   sdot_mid_lat=spr['mid_lat'], sdot_mid_lng=spr['mid_lng'])
        for key, var in variants.items():
            name = '_'.join(key)
            row[f'v_{name}'] = (var.get(uid) or {}).get('score')
        rows.append(row)
    dev = [r for r in rows if r['split'] == 'dev']
    test = [r for r in rows if r['split'] == 'test']

    # threshold on dev (weighted), primary score
    dev_obs = [r for r in dev if r['score'] is not None]
    tau = threshold_for_recall([r['score'] for r in dev_obs],
                               [r['truth2'] == 'none' for r in dev_obs],
                               [r['weight'] for r in dev_obs])
    test_obs = [r for r in test if r['score'] is not None]
    coverage = len(test_obs) / len(test)
    coverage_w = sum(r['weight'] for r in test_obs) / sum(r['weight'] for r in test)

    def primary(rr):
        obs = [r for r in rr if r['score'] is not None]
        m = _metric_block(obs, 'score', tau)
        mu = _metric_block(obs, 'score', tau, weighted=False)
        sw = _metric_block(obs, 'swap_score', None)
        sm = _metric_block(obs, 'street_mean_score', None)
        pr = _metric_block(obs, 'prior_score', None)
        out = {'auc_w': m.get('auc'), 'auc_u': mu.get('auc'), 'disc_auc': m.get('disc_auc'),
               'precision_w': m.get('precision'), 'recall_w': m.get('recall'),
               'precision_u': mu.get('precision'), 'recall_u': mu.get('recall'),
               'swap_disc_auc': sw.get('disc_auc'), 'swap_auc_w': sw.get('auc'),
               'street_mean_disc_auc': sm.get('disc_auc'),
               'prior_auc_w': pr.get('auc'),
               'coverage': (len(obs) / len(rr)) if rr else None}
        if out['auc_w'] is not None and out['prior_auc_w'] is not None:
            out['auc_minus_prior'] = out['auc_w'] - out['prior_auc_w']
        if out['disc_auc'] is not None and out['swap_disc_auc'] is not None:
            out['disc_minus_swap'] = out['disc_auc'] - out['swap_disc_auc']
        return out
    point = primary(test)
    reps = bootstrap(test, primary, args.n_boot)
    cis = {k: ci(reps, k) for k in point}
    beats_prior = cis.get('auc_minus_prior') is not None and cis['auc_minus_prior'][0] > 0
    verdict, clauses = decide(point['precision_w'], point['disc_auc'], point['swap_disc_auc'],
                              beats_prior, coverage)

    # oracle (test-optimal) precision at recall >= 0.67, secondary
    tau_test = threshold_for_recall([r['score'] for r in test_obs],
                                    [r['truth2'] == 'none' for r in test_obs],
                                    [r['weight'] for r in test_obs])
    oracle = _metric_block(test_obs, 'score', tau_test)

    # secondary variants on the test set
    var_metrics = {}
    for key in variants:
        name = 'v_' + '_'.join(key)
        obs_dev = [r for r in dev if r[name] is not None]
        t = threshold_for_recall([r[name] for r in obs_dev],
                                 [r['truth2'] == 'none' for r in obs_dev],
                                 [r['weight'] for r in obs_dev]) if obs_dev else None
        m = _metric_block([r for r in test if r[name] is not None], name, t)
        m['coverage'] = sum(r[name] is not None for r in test) / len(test)
        m['tau_dev'] = t
        var_metrics['/'.join(key)] = m

    # PS crowd rule and OSM on the same faces (context)
    def rule_block(rr, pred_key, absent_value='none'):
        rr = [r for r in rr if r.get(pred_key) is not None]
        prec, rec = precision_recall([r[pred_key] == absent_value for r in rr],
                                     [r['truth2'] == 'none' for r in rr],
                                     [r['weight'] for r in rr])
        return {'n': len(rr), 'precision_w': prec, 'recall_w': rec}
    context = {
        'ps_m1_on_observed_test': rule_block(test_obs, 'ps_m1'),
        'ps_m2_k2_on_observed_test': rule_block(test_obs, 'ps_m2_k2'),
        'ps_m1_on_all_test': rule_block(test, 'ps_m1'),
        'osm_on_all_test': dict(rule_block(test, 'osm_sidewalk', 'absent'),
                                coverage=sum(r['osm_sidewalk'] is not None for r in test)
                                / len(test)),
        'osm_on_observed_test': rule_block(test_obs, 'osm_sidewalk', 'absent'),
    }

    # exploratory slices (test, observed)
    def slice_by(keyfn):
        groups = defaultdict(list)
        for r in test:
            groups[keyfn(r)].append(r)
        out = {}
        for g, rr in sorted(groups.items(), key=lambda kv: str(kv[0])):
            obs = [r for r in rr if r['score'] is not None]
            m = _metric_block(obs, 'score', tau) if obs else {}
            m.update(n_faces=len(rr), coverage=len(obs) / len(rr),
                     n_absent=sum(r['truth2'] == 'none' for r in obs),
                     mean_share_absent=_mean([r['score'] for r in obs if r['truth2'] == 'none']),
                     mean_share_present=_mean([r['score'] for r in obs
                                               if r['truth2'] == 'present']))
            out[str(g)] = m
        return out

    def setback(r):
        try:
            e = edges[r['street_edge_id']]
            _, cross, _ = e.project(float(r['sdot_mid_lat']), float(r['sdot_mid_lng']))
            a = abs(cross)
            return '<8m' if a < 8 else ('8-12m' if a < 12 else '>=12m')
        except (ValueError, TypeError):
            return 'n/a'
    edges = {e: EdgeGeom(ft['geometry']['coordinates'])
             for e, ft in load_edges_geojson(dd / 'edges.geojson').items()}
    for r in rows:
        r['sdot_setback'] = setback(r)
    slices = {
        'way_type': slice_by(lambda r: r['way_type']),
        'truth3': slice_by(lambda r: r['truth3']),
        'stratum': slice_by(lambda r: r['stratum']),
        'sdot_unit_offset_from_centerline': slice_by(lambda r: r['sdot_setback']),
        'capture_year': slice_by(lambda r: (r['newest_observed_date'] or 'none')[:4]),
        'occlusion': slice_by(lambda r: 'n/a' if r['occluded_mean'] is None else
                              ('<0.2' if r['occluded_mean'] < 0.2 else
                               ('0.2-0.4' if r['occluded_mean'] < 0.4 else '>=0.4'))),
        'arterial': slice_by(lambda r: r['arterial']),
    }

    # Mapillary arm (secondary): coverage and metrics on faces it observes
    mly_out = None
    if ('mly', POOL_PRIMARY, 'all', 'walk_share') in variants:
        mv = variants[('mly', POOL_PRIMARY, 'all', 'walk_share')]
        for r in rows:
            r['mly_score'] = (mv.get(r['face_uid']) or {}).get('score')
            r['mly_n_views'] = (mv.get(r['face_uid']) or {}).get('n_views', 0)
        allobs = [r for r in rows if r['mly_score'] is not None]
        both = [r for r in allobs if r['score'] is not None]
        mly_out = {
            'faces_with_any_mly_view': sum(r['mly_n_views'] > 0 for r in rows),
            'faces_observed_all_splits': len(allobs), 'n_sample': len(rows),
            'coverage': len(allobs) / len(rows),
            'metrics_all_splits_at_gsv_tau': _metric_block(allobs, 'mly_score', tau),
            'gsv_on_same_faces': _metric_block(both, 'score', tau),
            'mly_on_same_faces': _metric_block(both, 'mly_score', tau),
            'n_absent_observed': sum(r['truth2'] == 'none' for r in allobs)}

    # gallery: worst false-absent / false-present on the test set
    gsv_panos = defaultdict(list)
    for v in views.get('gsv', []):
        gsv_panos[v['face_uid']].append(v)
    rng = random.Random(SEED)

    def gallery(rr, reverse, kind):
        rr = sorted(rr, key=lambda r: (r['score'], r['face_uid']), reverse=reverse)[:20]
        rng.shuffle(rr)   # seeded order, so a reader doesn't anchor on rank
        out = []
        for r in sorted(rr, key=lambda r: r['face_uid']):
            for v in gsv_panos[r['face_uid']]:
                out.append({'kind': kind, 'face_uid': r['face_uid'], 'truth2': r['truth2'],
                            'truth3': r['truth3'], 'way_type': r['way_type'],
                            'face_score': round(r['score'], 4), 'pano_id': v['pano_id'],
                            'date': v['date'], 'face_bearing_deg': v['face_bearing_deg'],
                            'gsv_url': ('https://www.google.com/maps/@?api=1&map_action=pano'
                                        f"&pano={v['pano_id']}&heading="
                                        f"{float(v['face_bearing_deg']):.0f}&pitch=-20"),
                            'view_id': v['view_id']})
        return out
    fa = [r for r in test_obs if r['truth2'] == 'present' and r['score'] <= tau]
    fp = [r for r in test_obs if r['truth2'] == 'none' and r['score'] > tau]
    gal = gallery(fa, False, 'false_absent') + gallery(fp, True, 'false_present')

    metrics = {
        'pre_registered': {'seed': SEED, 'pool': POOL_PRIMARY, 'target_recall': TARGET_RECALL,
                           'n_boot': args.n_boot, 'observed_min_frac': OBSERVED_MIN_FRAC,
                           'min_observed_panos': MIN_OBSERVED_PANOS},
        'tau_dev': tau, 'n_dev': len(dev), 'n_dev_observed': len(dev_obs), 'n_test': len(test),
        'n_test_observed': len(test_obs),
        'n_test_disc_observed': sum(r['stratum'] == 'D' for r in test_obs),
        'coverage': coverage, 'coverage_weighted': coverage_w,
        'primary': point, 'ci95': cis, 'beats_prior': beats_prior,
        'verdict': verdict, 'clauses': clauses,
        'oracle_test_tau': tau_test, 'oracle_precision_w': oracle.get('precision'),
        'oracle_recall_w': oracle.get('recall'),
        'variants': var_metrics, 'context': context, 'slices': slices, 'mapillary': mly_out,
        'gallery_faces': {'false_absent': len(fa), 'false_present': len(fp)},
    }
    write_json(dd / 'metrics.json', metrics)
    keep = ['face_uid', 'city', 'street_edge_id', 'street_side', 'stratum', 'weight', 'split',
            'truth2', 'truth3', 'way_type', 'len_m', 'score', 'n_views', 'n_observed',
            'occluded_mean', 'newest_observed_date', 'swap_score', 'street_mean_score',
            'prior_score', 'ps_m1', 'ps_m2_k2', 'ps_ns', 'osm_sidewalk', 'sdot_setback',
            'mly_score', 'mly_n_views'] + sorted(k for k in rows[0] if k.startswith('v_'))
    write_csv(dd / 'face_scores.csv', [{k: _fmt(r.get(k)) for k in keep} for r in rows], keep)
    write_csv(dd / 'gallery.csv', gal)
    print(json.dumps({'verdict': verdict, 'tau_dev': tau, 'coverage': coverage,
                      'primary': point, 'ci95': cis, 'clauses': clauses}, indent=1))


def _mean(v):
    return float(np.mean(v)) if v else None


def _fmt(v):
    return round(v, 5) if isinstance(v, float) else v


# --- manifest ---------------------------------------------------------------------------------------

def cmd_manifest(args):
    """sha256 of every input and output this study used (run where the files are)."""
    files = {}
    for p in args.files:
        for q in sorted(Path(p).rglob('*')) if Path(p).is_dir() else [Path(p)]:
            if q.is_file():
                files[str(q)] = {'sha256': sha256_file(q), 'bytes': q.stat().st_size}
    write_json(args.out, {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'files': files})
    print(f'manifest: {len(files)} files -> {args.out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--sp-face', required=True)
    s.add_argument('--streets', required=True)
    s.add_argument('--out', default=str(DATA_DIR))
    s = sub.add_parser('gsv-views')
    s.add_argument('--work', required=True)
    s.add_argument('--edges', default=str(DATA_DIR / 'edges.geojson'))
    s.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('mly-views')
    s.add_argument('--work', required=True)
    s.add_argument('--edges', default=str(DATA_DIR / 'edges.geojson'))
    s.add_argument('--env-file', default=str(REPO_ROOT / '.env'))
    s = sub.add_parser('bands')
    s.add_argument('--work', required=True)
    s = sub.add_parser('osm-tags')
    s.add_argument('--edges', default=str(DATA_DIR / 'edges.geojson'))
    s.add_argument('--out', default=str(DATA_DIR))
    s = sub.add_parser('score')
    s.add_argument('--work', default=str(DATA_DIR))
    s.add_argument('--data', default=str(DATA_DIR))
    s.add_argument('--sp-face', required=True)
    s.add_argument('--n-boot', type=int, default=N_BOOT)
    s = sub.add_parser('manifest')
    s.add_argument('--out', required=True)
    s.add_argument('files', nargs='+')
    args = ap.parse_args(argv)
    {'sample': cmd_sample, 'gsv-views': cmd_gsv_views, 'mly-views': cmd_mly_views,
     'bands': cmd_bands, 'osm-tags': cmd_osm_tags, 'score': cmd_score,
     'manifest': cmd_manifest}[args.cmd](args)


if __name__ == '__main__':
    main()
