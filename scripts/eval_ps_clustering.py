"""Score Project Sidewalk's production label clustering against RampNet ground truth.

Companion to docs/ps-clustering-eval.md (the pre-registered protocol) and to
SidewalkWebpage#4706. The labeler keeps every per-view AI label on purpose, so the
server's clustering is what turns labels into ramps; this measures how well it does
that, and where any shortfall comes from (threshold, placement or merge criterion).

Inputs (all read as data, nothing imported from SidewalkWebpage unless --ps-script):
  - /v3/api/rawLabels?labelType=CurbRamp&filetype=geojson      -> raw_labels.geojson
  - /v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
                                                               -> clusters.geojson
    (both downloaded into --out when --server is given and the files are absent)
  - runs/<city>/results.jsonl (maps each AI label back to its stored detection by
    pano id + pixel position; the same rounding send_to_ps.py used)
  - ../RampNet/benchmark/<city>/{verdicts.json,records.jsonl} via eval_sites.py

Arms (every one a partition of the same AI labels, scored by one scorer in one frame):
  deployed        the server's clusters as served
  ps_repro        SidewalkWebpage/scripts/label_clustering.py cluster(), verbatim
                  (--ps-script); exists to prove the harness reproduces the server
  ps @ t          the PS algorithm (complete linkage + same-(user,pano) cannot-link),
                  re-implemented vectorized, per region, threshold sweep
  ps_citywide     ...the same at 7.5 m over the whole city (region-boundary effect)
  ps_placeable @ t ...the same, restricted to the labels the raycast can place, so
                  it is exactly the label set ps_raycast and fusion use
  ps_raycast @ t  the PS algorithm on the labeler's raycast positions (isolates
                  placement from algorithm)
  fusion          fuse_sites.py's ray-aware associator (the labeler's reference)
  fusion_server   the same associator fed only what the server holds: every live label
                  (AI and human) at its stored pixel, AI confidence (label_ai_info), and
                  the pano position (pano_data; inverted from the labels for a pano only
                  humans labeled). The partition the server would compute if its
                  clustering were fusion; it touches no SidewalkWebpage code
  fusion_server+attach  ...plus one pre-declared rule for the labels the raycast cannot
                  place: join the site on the label's bearing ray (attach_unplaceable)

Offline mode (--offline, issue #106) scores a city with no server, or a run no server
holds: one label per stored detection >= --min-confidence (rig-masked) is synthesized
and placed with the server's own estimator (ps_placement.py, exact to the server's
parity fixture); regions come from the nearest street of /v3/api/streets when --server
names one (the server's own rule), else every label is in one region. `deployed` and
`ps_repro` do not exist there; every other arm runs unchanged. --offline-check, run
against a live server, measures how well that reproduces the server's positions and
partition (docs/ps-clustering-eval.md, "Beyond Richmond").

The headline metric is `coverage` (a cluster of this arm within the match radius
of a pool GT ramp). `recall (union)` is eval_sites' definition, kept only for the
tie-back: it counts a self-detected ramp as recovered whether or not any cluster
landed on it, which pins ~83% of it constant across arms.

Every cluster is placed at the mean of its members' raycast positions, so scores
depend only on who was grouped with whom. Needs pandas, scipy and `haversine`
(the package the PS script uses); none of them is a pipeline dependency, so
install them alongside requirements.txt: pip install pandas scipy haversine.

Usage:
    python scripts/eval_ps_clustering.py richmond \
        --server https://sidewalk-richmond.cs.washington.edu \
        --ps-script /path/to/SidewalkWebpage/scripts/label_clustering.py
    # another frame (#56): metres, per-pano, auto or per-rig, resolved exactly as
    # fuse_sites.py does; writes ps_clustering_eval_<mode>/ (a number: _h<val>)
    python scripts/eval_ps_clustering.py richmond --camera-height-m per-pano \
        --labels runs/richmond/ps_clustering_eval/raw_labels.geojson \
        --clusters runs/richmond/ps_clustering_eval/clusters.geojson
    # offline (no server labels): -> runs/<city>/ps_clustering_eval_offline_t0.55/
    python scripts/eval_ps_clustering.py bend --offline
    python scripts/eval_ps_clustering.py laurens --split laurens_mapillary \
        --results runs/laurens/results.raw.jsonl --offline --min-confidence 0.3 \
        --server https://sidewalk-laurens.cs.washington.edu    # streets, for regions only
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import ps_placement  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
from detectors import (BENCHMARK_CONFIDENCE, BORDER_EXCLUDE,  # noqa: E402
                       DECODE_ARGMAX, on_camera_rig)

try:  # analysis-only dependencies, deliberately not in requirements.txt
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import squareform
    from haversine import haversine_vector
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f'{exc.name} is missing. This analysis tool needs three packages the '
        'pipeline does not: pip install pandas scipy haversine '
        '(haversine is the package label_clustering.py itself uses).') from exc

PS_THRESHOLD_KM = 0.0075   # label_clustering.py THRESHOLDS['CurbRamp']
API_LABELS = '/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson'
API_CLUSTERS = ('/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true'
                '&filetype=geojson')


# ----------------------------------------------------------------------------- inputs

def fetch(url, dest, refresh=False):
    """Download url to dest unless it is already there, and record its provenance.

    The deployed partition is regenerated whenever the server re-clusters, so a
    cached pull has to say *when* it was taken, and reusing one has to be visible:
    a cache hit prints the pull's age, and --refresh re-pulls over it. Written
    temp-then-rename, and a body that is not a non-empty FeatureCollection is
    refused rather than cached (a zero-feature 200 would otherwise score as "the
    city has no labels").
    """
    if dest.exists() and not refresh:
        age = pull_age_days(dest)
        how_old = 'age unknown' if age is None else f'pulled {age:.1f} days ago'
        print(f'reusing cached {dest.name} ({how_old}); --refresh re-pulls it',
              file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 (https, fixed host)
        tmp.write_bytes(r.read())
    try:
        n = len(json.loads(tmp.read_text(encoding='utf-8'))['features'])
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f'{url}: not a GeoJSON FeatureCollection ({exc})') from exc
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f'{url}: zero features — refusing to cache an empty pull')
    tmp.replace(dest)
    (dest.parent / (dest.name + '.source.json')).write_text(
        json.dumps({'url': url, 'n_features': n,
                    'fetched_at': datetime.now(timezone.utc)
                    .isoformat(timespec='seconds')}, indent=1), encoding='utf-8',
        newline='\n')


def source_meta(path):
    """The .source.json fetch record beside path, or {} when there is none."""
    side = path.parent / (path.name + '.source.json')
    if not side.exists():
        return {}
    try:
        return json.loads(side.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def pull_age_days(path):
    """How long ago this file was pulled, per its fetch record; None if unrecorded."""
    when = source_meta(path).get('fetched_at')
    if not when:
        return None
    try:
        stamp = datetime.fromisoformat(when)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0


def provenance(path, n_features):
    """One report line per input file: where it came from, when, and its sha256."""
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = source_meta(path)
    age = pull_age_days(path)
    how_old = '' if age is None else f' ({age:.1f} days old at run time)'
    when = (meta.get('fetched_at', '') + how_old) or (
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .isoformat(timespec='seconds') + ' (file mtime; not a recorded fetch)')
    where = meta.get('url') or '(supplied on the command line)'
    return f'- `{path.name}`: {n_features} features, sha256 `{h}`, {when}, from {where}'


def load_labels(path):
    """(DataFrame, n_dropped). Mirrors label_clustering.clean_label_data: rows whose
    lng is null or > 360 are dropped, because the server drops them before
    clustering (corrupt values of order 1e14 have been observed upstream)."""
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    rows, dropped = [], 0
    for ft in feats:
        q = ft['properties']
        lng, lat = ft['geometry']['coordinates']
        lng = float('nan') if lng is None else float(lng)
        lat = float('nan') if lat is None else float(lat)
        if math.isnan(lng) or lng > 360:
            dropped += 1
            continue
        sev = q.get('severity')
        rows.append({'label_id': q['label_id'], 'user_id': q['user_id'],
                     'pano_id': q['pano_id'], 'region_id': q['region_id'],
                     'lat': lat, 'lng': lng,
                     'pano_x': q['pano_x'], 'pano_y': q['pano_y'],
                     # severity is unused here; the verbatim PS cluster() reads it.
                     'severity': float('nan') if sev is None else float(sev),
                     'label_type': q['label_type'],
                     # read only by the fusion_server arm (server_panos)
                     'pano_width': q.get('pano_width'), 'pano_height': q.get('pano_height'),
                     'camera_heading': q.get('camera_heading'),
                     'pano_source': q.get('pano_source'),
                     'image_capture_date': q.get('image_capture_date')})
    return pd.DataFrame(rows), dropped


def load_server_clusters(path):
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    out = []
    for ft in feats:
        q = ft['properties']
        lng, lat = ft['geometry']['coordinates']
        out.append({'id': q['label_cluster_id'], 'label_ids': list(q['label_ids']),
                    'lat': lat, 'lng': lng})
    return out


def label_to_detection(results_path, labels):
    """({label_id: (pano_id, det_index)}, n_ambiguous_pixel_keys, n_duplicate_labels).

    Keyed by pano id + the pixel rounding send_to_ps.py used. Both directions can
    be many-to-one and both are reported rather than collapsed silently:

    - two stored *detections* that round to the same pixel make that key
      ambiguous, so the key is dropped (no label is attributed to the wrong
      detection) and counted;
    - two *labels* at one pixel — a re-submitted campaign, which is what laurens
      got on -test (SidewalkWebpage#5382) — both map to one detection. The
      same-(user, pano) cannot-link then forces them into different clusters,
      where they read as a fragment, so the `same_pano_pairs` tripwire cannot
      catch them. The count is reported instead.
    """
    by_key, ambiguous = {}, set()
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            w, h = p['width'], p['height']
            for i, d in enumerate(rec.get('detections', [])):
                key = (p['panorama_id'], round(d['x_normalized'] * w),
                       round(d['y_normalized'] * h))
                if key in by_key:
                    ambiguous.add(key)
                by_key[key] = i
    for key in ambiguous:
        by_key.pop(key, None)
    mapping = {}
    for row in labels.itertuples(index=False):
        i = by_key.get((row.pano_id, row.pano_x, row.pano_y))
        if i is not None:
            mapping[row.label_id] = (row.pano_id, i)
    n_duplicate_labels = len(mapping) - len(set(mapping.values()))
    return mapping, len(ambiguous), n_duplicate_labels


# --------------------------------------------------------------------------- clusters

@dataclass
class Cluster:
    id: int
    members: list                    # [(pano_id, det_index)] — AI labels only
    n_labels: int                    # every label, mapped or not (human labels too)
    e: float | None = None
    n: float | None = None
    server_latlng: tuple | None = None
    label_ids: list = field(default_factory=list)


def place(clusters, det_pos):
    for c in clusters:
        pts = [det_pos[m] for m in c.members if m in det_pos]
        if pts:
            c.e = sum(p[0] for p in pts) / len(pts)
            c.n = sum(p[1] for p in pts) / len(pts)
    return clusters


def clusters_from_assignment(labels, assignment, det_of):
    """labels: DataFrame; assignment: array of cluster ids aligned with labels rows."""
    groups = {}
    for lid, cid in zip(labels['label_id'].to_numpy(), assignment):
        groups.setdefault(int(cid), []).append(int(lid))
    out = []
    for k, (_cid, lids) in enumerate(sorted(groups.items())):
        out.append(Cluster(k, [det_of[lab] for lab in lids if lab in det_of], len(lids),
                           label_ids=lids))
    return out


def clusters_from_server(server_clusters, det_of):
    out = []
    for k, sc in enumerate(server_clusters):
        out.append(Cluster(k, [det_of[lab] for lab in sc['label_ids'] if lab in det_of],
                           len(sc['label_ids']), server_latlng=(sc['lat'], sc['lng']),
                           label_ids=list(sc['label_ids'])))
    return out


def clusters_from_sites(sites, refit_position):
    out = []
    for s in sites:
        members = [(d.pano_id, d.det_index) for d, _ in s.members if d.operational]
        if not members:
            continue
        c = Cluster(s.id, members, len(members))
        if refit_position:
            c.e, c.n = s.e, s.n
        out.append(c)
    return out


# ------------------------------------------------- fusion on what the server holds

# The server's own placement height (SidewalkWebpage#4819): its label lat/lng are the
# flat raycast at this height, so inverting that raycast recovers the camera position.
SERVER_CAMERA_HEIGHT_M = 2.341219672825709
# Only labels this close are inverted: nearer the horizon the server's bounded tail
# departs from the flat raycast, and one such label would move a camera metres.
INVERT_MAX_RANGE_M = 15.0
# det_index for a human label. Stored detections are capped at 50 per pano, so this
# never collides with a real index, and det_pos never holds it (scored by n_labels only).
HUMAN_DET_BASE = 10000
HUMAN_CONFIDENCE = 1.0
# det_index for a second (third, ...) AI label at a pixel another AI label already took:
# a re-submitted campaign (Vancouver: 1,058 pairs) puts two live labels on one stored
# detection. Both are on the server, so both are fused (the same-pano cannot-link keeps
# them apart, as it does on the server), but only the first carries the run's det_index:
# the scorer keys on (pano_id, det_index), so the second is counted in n_labels and never
# scored twice. >= HUMAN_DET_BASE, so clusters_from_server_sites leaves it out of members.
DUPLICATE_DET_BASE = 20000


def invert_camera_position(rows, server_height=SERVER_CAMERA_HEIGHT_M):
    """(lat, lng, n_used) of one pano's camera, from its labels' server positions, or
    None when no label is close enough to invert.

    Each server position is the flat raycast from the camera, so the camera sits at the
    label minus that raycast's offset (the offset depends only on heading, pixel and
    height). The per-label estimates are combined by median.

    Example:
        A label straight ahead (x = 0.5, heading 0) and 20 degrees down at 2.34 m lies
        ~6.4 m north of the camera, so the camera comes back ~6.4 m south of the label.
    """
    lats, lngs = [], []
    for r in rows:
        if not r['pano_width'] or not r['pano_height'] or r['camera_heading'] is None:
            continue
        probe = fs.SlimPano(r['pano_id'], r['lat'], r['lng'], r['camera_heading'],
                            None, None, None, r['pano_source'] or '', [])
        g = geo.detection_ground_point(
            fs.pano_pose(probe, fs.POSE_OFF), r['pano_x'] / r['pano_width'],
            r['pano_y'] / r['pano_height'], camera_height=server_height,
            max_range_m=INVERT_MAX_RANGE_M, errors=geo.error_model_for(probe.source),
            apply_pose=False)
        if g is None:
            continue
        f = geo.LocalFrame(r['lat'], r['lng'])
        ge, gn = f.to_enu(g.lat, g.lng)
        lat, lng = f.to_latlng(-ge, -gn)
        lats.append(lat)
        lngs.append(lng)
    if not lats:
        return None
    return float(np.median(lats)), float(np.median(lngs)), len(lats)


def server_panos(labels, det_of, run_by_id, ai_user, decode=DECODE_ARGMAX,
                 border=BORDER_EXCLUDE, invert=True):
    """SlimPanos built from the server's labels, for fuse_sites to associate: the
    partition the server would compute if its clustering were fusion. What each field
    comes from, and why the server has it:

    - detections: every label on the pano, at its stored pixel. A label is AI when its
      user_id is `ai_user` (the account that submitted the run): it keeps the run's
      det_index (so the scorer places it) and its confidence, which the server stores in
      label_ai_info. Every other account's label is human: HUMAN_DET_BASE+k at
      HUMAN_CONFIDENCE (they seed sites first). A second AI label on a detection
      another label already took gets DUPLICATE_DET_BASE+j at the same confidence, so
      every label is exactly one detection. An AI-account label that maps to no
      stored detection raises ValueError: it came from some other file (a band from
      results.band.jsonl, a re-inferred campaign), and read as human it would enter at
      confidence 1.0 and seed sites.
    - heading / source / capture date: the label row.
    - camera position: inverted from the pano's own labels (invert_camera_position),
      which is server-held whichever file the labels were submitted from; the run's pano
      block only when no label is close enough to invert. The run's block is NOT
      pano_data's once a pano has been repositioned (Richmond's posfix3seq panos are live
      at raw GPS while results.jsonl holds SfM), so it is the fallback, not the source.
      The AI account's labels are inverted when the pano has any (the placement check
      validates exactly those), and the other accounts' only when it has none: a human
      label keeps the lat/lng it was inserted at, which on Laurens is a pano position the
      server no longer holds (human-only inversion sits a median 8.7 m from the live block
      on 70 panos, AI-only 0.000 m), so mixed into the median it moved 57 of 695 panos.
      With invert=False (offline: the synthesized labels were placed FROM the run's pano
      block, so that block is the server's position by construction) the run's block is
      used wherever there is one, and inversion only for a pano with no run pano --
      inverting there would add only the inversion's own error (p90 0.24-0.33 m).
    - camera height fields, decode, border: copied from the run pano when there is one,
      so this arm raycasts in the same frame as every other labeler arm and fuse's
      mixed-decode / mixed-border guards see the run's real values (#111, #130); a pano
      only humans labeled takes the run's single `decode` / `border`.

    Returns (panos, stats). stats counts panos by position source (`inverted`,
    `run_position`, `unplaceable`: no run pano and nothing to invert, left out), the
    labels on unplaceable panos (`unplaceable_labels`, their ids in
    `unplaceable_label_ids`), the AI and human labels, the AI labels that shared a
    detection (`ai_duplicate`), and `inverted_far`: inverted panos more than 1 m from the
    run's block, i.e. panos whose live position is not the run's (a repositioned
    sequence). stats['label_of'] maps every emitted (pano_id, det_index) to its label_id,
    one to one: a key is never reused, so no label is dropped or counted twice.

    Example:
        A run pano with detections 0 and 3 and a third detection 5 that was never
        submitted, plus one human label: the built pano holds 0, 3 and HUMAN_DET_BASE,
        never 5 -- the run's detections are read only for confidences.
    """
    stats = {'inverted': 0, 'run_position': 0, 'unplaceable': 0, 'unplaceable_labels': 0,
             'human_labels': 0, 'ai_labels': 0, 'ai_duplicate': 0, 'inverted_far': 0,
             'label_of': {}, 'unplaceable_label_ids': []}
    panos = []
    for pano_id, grp in labels.groupby('pano_id', sort=True):
        rows = grp.to_dict('records')
        run = run_by_id.get(pano_id)
        dets, k, j, used = [], 0, 0, set()
        for r in rows:
            if not r['pano_width'] or not r['pano_height']:
                continue
            x, y = r['pano_x'] / r['pano_width'], r['pano_y'] / r['pano_height']
            if str(r['user_id']) == str(ai_user):
                ai = det_of.get(r['label_id'])
                if ai is None:
                    raise ValueError(
                        f"AI-account label {r['label_id']} on pano {pano_id} maps to no "
                        'stored detection in the run file; it was submitted from another '
                        'file, so fusion_server cannot give it a confidence')
                conf = next((c for i, _x, _y, c in run.detections if i == ai[1]),
                            None) if run else None
                index = ai[1]
                if index in used:
                    index = DUPLICATE_DET_BASE + j
                    j += 1
                    stats['ai_duplicate'] += 1
                used.add(index)
                dets.append((index, x, y, HUMAN_CONFIDENCE if conf is None else conf))
                stats['ai_labels'] += 1
            else:
                dets.append((HUMAN_DET_BASE + k, x, y, HUMAN_CONFIDENCE))
                k += 1
                stats['human_labels'] += 1
            stats['label_of'][(pano_id, dets[-1][0])] = r['label_id']
        head = rows[0]
        ai_rows = [r for r in rows if str(r['user_id']) == str(ai_user)]
        inv = (invert_camera_position(ai_rows or rows) if invert or run is None
               else None)
        if inv is not None:
            lat, lng, _n = inv
            stats['inverted'] += 1
            if run is not None and geo.haversine_m(lat, lng, run.lat, run.lng) > 1.0:
                stats['inverted_far'] += 1
        elif run is not None:
            lat, lng = run.lat, run.lng
            stats['run_position'] += 1
        else:
            stats['unplaceable'] += 1
            stats['unplaceable_labels'] += len(rows)
            stats['unplaceable_label_ids'] += [r['label_id'] for r in rows]
            for i, *_rest in dets:
                del stats['label_of'][(pano_id, i)]
            continue
        p = fs.SlimPano(pano_id, lat, lng, head['camera_heading'], None, None,
                        head['image_capture_date'], head['pano_source'] or '', dets,
                        decode=run.decode if run is not None else decode,
                        border=run.border if run is not None else border)
        if run is not None:
            p.camera_height_m = run.camera_height_m
            p.camera_height_spread_m = run.camera_height_spread_m
            p.ground_tilt_deg = run.ground_tilt_deg
            p.camera_height_vintage_m = run.camera_height_vintage_m
            p.height_group = run.height_group
            p.height_table = run.height_table
        panos.append(p)
    return panos, stats


def inversion_check(labels, run_by_id, exclude_user=None):
    """Distances (m) between the inverted and the run camera position over panos in
    both. With exclude_user (the AI account), only the other accounts' labels are
    inverted: the accuracy of inversion from human labels, which is all a pano only
    humans labeled has."""
    out = []
    for pano_id, grp in labels.groupby('pano_id', sort=True):
        run = run_by_id.get(pano_id)
        if run is None:
            continue
        rows = [r for r in grp.to_dict('records')
                if exclude_user is None or str(r['user_id']) != str(exclude_user)]
        inv = invert_camera_position(rows)
        if inv is not None:
            out.append(geo.haversine_m(inv[0], inv[1], run.lat, run.lng))
    return sorted(out)


def clusters_from_server_sites(sites, panos=(), unpositioned=(), label_of=None):
    """Clusters from a fusion_server fuse: every member counts toward n_labels, but only
    AI members (a real det_index) are placeable and scoreable, as in the other arms.

    A server has to put every label somewhere, but the raycast drops the ones it cannot
    place (beyond the range cap, or at/above the horizon), so no site holds them. Each
    label of `panos` that no site holds becomes a singleton cluster, and the number of
    them is the second return value. Each label id in `unpositioned` (labels on panos
    server_panos could not position at all) is one more singleton with no AI member, so
    every live label is in exactly one cluster; those are not in the second return value.
    `label_of` (server_panos' stats['label_of']) fills each cluster's label_ids."""
    label_of = label_of or {}
    out, held = [], set()
    for s in sites:
        keys = [(d.pano_id, d.det_index) for d, _ in s.members]
        held.update(keys)
        out.append(Cluster(s.id, [k for k in keys if k[1] < HUMAN_DET_BASE], len(keys),
                           label_ids=[label_of[k] for k in keys if k in label_of]))
    n_singletons = 0
    for p in panos:
        for i, *_rest in p.detections:
            key = (p.pano_id, i)
            if key not in held:
                out.append(Cluster(len(out), [key] if i < HUMAN_DET_BASE else [], 1,
                                   label_ids=[label_of[key]] if key in label_of else []))
                n_singletons += 1
    for lab in unpositioned:
        out.append(Cluster(len(out), [], 1, label_ids=[lab]))
    return out, n_singletons


# ---------------------------------------------------------------- the PS algorithm

def ps_linkage(sub):
    """Complete-linkage tree over label_clustering.py's custom_dist for one group:
    haversine km between lat/lng, float max for two labels from the same (user, pano)."""
    pts = sub[['lat', 'lng']].to_numpy(dtype=float)
    dist = np.asarray(haversine_vector(pts, pts, comb=True), dtype=float)
    users = sub['user_id'].to_numpy()
    panos = sub['pano_id'].to_numpy()
    same = (users[:, None] == users[None, :]) & (panos[:, None] == panos[None, :])
    dist = np.where(same, sys.float_info.max, dist)
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method='complete')


def region_groups(labels):
    """Positional index arrays, one per region. groupby drops null keys, so a label
    with no region_id would silently never be assigned — refuse instead."""
    missing = int(labels['region_id'].isna().sum())
    if missing:
        raise SystemExit(f'{missing} labels have a null region_id; per-region '
                         'clustering cannot place them, and silently lumping them '
                         'together would fabricate one giant cluster')
    return [idx for _, idx in sorted(labels.groupby('region_id').indices.items())]


# The blocked partition (issue #106) cuts the city into single-linkage components at the
# widest threshold plus this margin before running complete linkage per component. The
# components are found on a flat equirectangular projection and the cut is made on the
# script's own haversine distances; the two disagree by well under 1e-3 relative at city
# scale (centimetres at 15 m), so 0.5 m guarantees no pair within any threshold is ever
# separated by the blocking.
BLOCK_MARGIN_M = 0.5


def blocked_components(lat, lng, radius_m):
    """Connected components of the graph joining every pair of points within radius_m
    (flat projection about the points' mean). Returns a label array aligned with lat.

    Why it is exact for complete linkage: every pair in two different components is more
    than radius_m apart, so any two clusters drawn from different components are too, and
    complete linkage never merges them below radius_m. Every flat cluster at a cut
    t <= radius_m therefore lies inside one component, and the merges below t inside a
    component happen in the same order as over the whole set.

    Example:
        >>> blocked_components(np.array([0.0, 0.0, 0.001]), np.array([0.0, 0.00005, 0.0]),
        ...                    10.0).tolist()
        [0, 0, 1]
    """
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    if len(lat) == 0:
        return np.zeros(0, dtype=int)
    lat0 = math.radians(float(lat.mean()))
    xy = np.column_stack([np.radians(lng) * math.cos(lat0) * geo.EARTH_RADIUS_M,
                          np.radians(lat) * geo.EARTH_RADIUS_M])
    pairs = cKDTree(xy).query_pairs(radius_m, output_type='ndarray')
    n = len(lat)
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
    _n, comp = connected_components(graph, directed=False)
    return comp


def ps_partition(labels, thresholds_km, per_region=True, blocked=True, stats=None):
    """{threshold_km: cluster-id array aligned with labels rows} for one linkage per
    group, cut at every threshold (fcluster cuts the same tree the script builds).

    blocked=True (the default since #106) runs the linkage per single-linkage component
    at the widest threshold + BLOCK_MARGIN_M (blocked_components) instead of over the
    whole group: the same partition at every threshold, without an N x N matrix over a
    whole city. blocked=False is the dense path, kept for the equality test. `stats`, if
    given, receives the largest block size."""
    # -1, not 0: every row must be assigned, and an unassigned row has to be loud
    # rather than collapsing into a fabricated cluster 0.
    out = {t: np.full(len(labels), -1, dtype=int) for t in thresholds_km}
    offset = {t: 0 for t in thresholds_km}
    groups = region_groups(labels) if per_region else [np.arange(len(labels))]
    if blocked:
        radius_m = max(thresholds_km) * 1000.0 + BLOCK_MARGIN_M
        lat_all = labels['lat'].to_numpy(dtype=float)
        lng_all = labels['lng'].to_numpy(dtype=float)
        blocks = []
        for idx in groups:
            comp = blocked_components(lat_all[idx], lng_all[idx], radius_m)
            order = np.argsort(comp, kind='stable')
            cuts = np.flatnonzero(np.diff(comp[order])) + 1
            blocks += [np.asarray(idx)[part] for part in np.split(order, cuts)]
        groups = blocks
    if stats is not None:
        stats['largest_block'] = max((len(g) for g in groups), default=0)
        stats['n_blocks'] = len(groups)
    for idx in groups:
        sub = labels.iloc[idx]
        if len(sub) == 1:
            for t in thresholds_km:
                out[t][idx] = offset[t] + 1
                offset[t] += 1
            continue
        tree = ps_linkage(sub)
        for t in thresholds_km:
            cl = fcluster(tree, t=t, criterion='distance')
            out[t][idx] = cl + offset[t]
            offset[t] += int(cl.max())
    for t, arr in out.items():
        if (arr < 0).any():
            raise SystemExit(f'{int((arr < 0).sum())} labels were never assigned a '
                             f'cluster at {t} km — the grouping dropped rows')
    return out


def ps_verbatim(labels, ps_script):
    """Run the SidewalkWebpage script's own cluster() per region, exactly as the
    server invokes it (one region at a time, CurbRamp threshold)."""
    spec = importlib.util.spec_from_file_location('ps_label_clustering', ps_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assignment = np.full(len(labels), -1, dtype=int)
    offset = 0
    for idx in region_groups(labels):
        sub = labels.iloc[idx].copy()
        sub['coords'] = sub.apply(lambda r: (r.lat, r.lng), axis=1)
        if len(sub) > 1:
            _, labeled = mod.cluster(sub, 'CurbRamp', mod.THRESHOLDS)
            cl = labeled['cluster'].to_numpy(dtype=int)
        else:
            cl = np.array([1])
        assignment[idx] = cl + offset
        offset += int(cl.max())
    if (assignment < 0).any():
        raise SystemExit(f'{int((assignment < 0).sum())} labels were never assigned '
                         'a cluster by the verbatim script')
    return assignment, mod.THRESHOLDS['CurbRamp']


def partition_agreement(a, b):
    """How much of partition a (list of Cluster) is reproduced by b: fraction of a's
    clusters whose label set appears verbatim in b, plus labels in disagreeing ones."""
    sets_b = {frozenset(c.label_ids) for c in b}
    same = [c for c in a if frozenset(c.label_ids) in sets_b]
    off_labels = sum(c.n_labels for c in a if frozenset(c.label_ids) not in sets_b)
    return len(same), len(a), off_labels


# ------------------------------------------------------------------------- scoring

def score(clusters, gt, det_pos, radius_m=5.0, frag_radii=(3.0, 5.0)):
    ramps, pool, points, op_verdicts = gt
    placed = [c for c in clusters if c.e is not None]
    # Only clusters within reach of some GT ramp can match or count as a fragment, so the
    # O(ramps x clusters) loops below run over those alone (identical results; a city of
    # 100k clusters otherwise costs minutes per arm).
    reach = max(radius_m, *frag_radii)
    near = placed
    if placed and ramps:
        tree = cKDTree(np.array([[c.e, c.n] for c in placed]))
        hit = set()
        for lst in tree.query_ball_point(np.array([[r.e, r.n] for r in ramps]), reach):
            hit.update(lst)
        near = [placed[i] for i in sorted(hit)]
    matched = es.match_one_to_one(pool, near, radius_m)
    # A cluster is "somebody's match" if it matches ANY GT ramp, not only a pool
    # one: a ramp on a pano whose missed-check was not confirmed is still a real
    # ramp, and a cluster sitting on it is not a fragment.
    matched_all = es.match_one_to_one(ramps, near, radius_m)
    matched_ids = {c.id for c in matched_all.values()}

    # Two recall-shaped numbers, deliberately both reported:
    #  - coverage: a cluster OF THIS ARM is within radius_m. This is what RQ2a
    #    asks and the only one of the two that responds to the partition.
    #  - recall: eval_sites' union recall, which counts a self-detected ramp as
    #    recovered whether or not any cluster landed on it. ~83% of Richmond's
    #    pool is self-detected, so it is nearly constant across arms; keep it
    #    only to tie back to fusion_eval/report.md.
    buckets = {'self_detected': 0, 'recovered_other_view': 0, 'unmatched': 0}
    self_detected_without_cluster = 0
    for i, ramp in enumerate(pool):
        if ramp.self_detected:
            buckets['self_detected'] += 1
            if i not in matched:
                self_detected_without_cluster += 1
        elif i in matched:
            buckets['recovered_other_view'] += 1
        else:
            buckets['unmatched'] += 1

    tp = fp = unsure = 0
    for c in clusters:
        vs = [op_verdicts[m] for m in c.members if m in op_verdicts]
        if not vs:
            continue
        if any(v is True or v == 'duplicate' for v in vs):
            tp += 1
        elif any(v is False for v in vs):
            fp += 1
        else:
            unsure += 1

    frag = {}
    for rf in frag_radii:
        n_with = n_extra = 0
        for i, _c in matched.items():
            ramp = pool[i]
            extra = sum(1 for o in near if o.id not in matched_ids
                        and math.hypot(o.e - ramp.e, o.n - ramp.n) <= rf)
            n_extra += extra
            n_with += extra > 0
        frag[rf] = {'ramps': len(matched), 'with_extra': n_with, 'extra': n_extra}

    # dual-ramp separation, as eval_sites (f). Matched against every GT ramp for
    # the same reason as the fragment count above, so a pair involving a non-pool
    # ramp can still be scored as kept apart.
    ramp_of_point = {id(pt): ri for ri, ramp in enumerate(ramps) for pt in ramp.points}
    by_pano = {}
    for pt in points:
        by_pano.setdefault(pt.pano_id, []).append(pt)
    dual = {'pairs': 0, 'both': 0, 'one': 0, 'neither': 0}
    for pid in sorted(by_pano):
        pts = by_pano[pid]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if math.hypot(pts[i].e - pts[j].e, pts[i].n - pts[j].n) >= 5.0:
                    continue
                dual['pairs'] += 1
                hits = sum(1 for pt in (pts[i], pts[j])
                           if ramp_of_point[id(pt)] in matched_all)
                dual['both' if hits == 2 else 'one' if hits == 1 else 'neither'] += 1

    # coherence: the cluster holding the self-detection vs its own GT ramp
    pos_key = {}
    for m, (e, n) in det_pos.items():
        pos_key[(m[0], round(e, 6), round(n, 6))] = m
    cluster_of = {m: c for c in clusters for m in c.members}
    offsets, missing = [], 0
    for ramp in pool:
        for pt in ramp.points:
            if pt.kind != 'det':
                continue
            m = pos_key.get((pt.pano_id, round(pt.e, 6), round(pt.n, 6)))
            c = cluster_of.get(m)
            if c is None or c.e is None:
                missing += 1
                continue
            offsets.append(math.hypot(c.e - ramp.e, c.n - ramp.n))
    offsets.sort()

    def pct(q):
        return offsets[min(len(offsets) - 1, int(q * len(offsets)))] if offsets else None

    same_pano_pairs = 0
    for c in clusters:
        seen = {}
        for m in c.members:
            seen[m[0]] = seen.get(m[0], 0) + 1
        same_pano_pairs += sum(k * (k - 1) // 2 for k in seen.values())

    n_pool = len(pool)
    recalled = buckets['self_detected'] + buckets['recovered_other_view']
    covered = len(matched)
    n_labels = sum(c.n_labels for c in clusters)
    return {
        'n_clusters': len(clusters), 'n_placed': len(placed), 'n_labels': n_labels,
        'labels_per_cluster': n_labels / len(clusters) if clusters else None,
        'precision': tp / (tp + fp) if tp + fp else None,
        'precision_ci': es.wilson(tp, tp + fp), 'tp': tp, 'fp': fp, 'unsure': unsure,
        'coverage': covered / n_pool if n_pool else None,
        'coverage_ci': es.wilson(covered, n_pool), 'covered': covered,
        'n_pool': n_pool,
        'self_detected_without_cluster': self_detected_without_cluster,
        'recall': recalled / n_pool if n_pool else None,
        'recall_ci': es.wilson(recalled, n_pool), 'buckets': buckets,
        'frag': frag, 'dual': dual,
        'coherence': {'n': len(offsets), 'median': pct(0.5), 'p90': pct(0.9),
                      'over_5m': sum(1 for o in offsets if o > 5.0), 'missing': missing},
        'same_pano_pairs': same_pano_pairs,
    }


# --------------------------------------------------------------------------- report

def fmt(v, nd=3):
    return 'n/a' if v is None else f'{v:.{nd}f}'


def frac(f):
    return f"{f['with_extra'] / f['ramps']:.2f}" if f['ramps'] else 'n/a'


def row_line(name, r):
    fr3, fr5, d, co = r['frag'][3.0], r['frag'][5.0], r['dual'], r['coherence']
    return (f"| {name} | {r['n_clusters']} | {r['n_placed']} | {r['n_labels']} | "
            f"{fmt(r['labels_per_cluster'], 2)} | {fmt(r['precision'])} "
            f"({r['tp']}/{r['fp']}) | {fmt(r['coverage'])} "
            f"({r['covered']}/{r['n_pool']}) | "
            f"{r['self_detected_without_cluster']} | {fmt(r['recall'])} | "
            f"{r['buckets']['self_detected']}/{r['buckets']['recovered_other_view']}/"
            f"{r['buckets']['unmatched']} | {frac(fr3)} ({fr3['extra']}) | "
            f"{frac(fr5)} ({fr5['extra']}) | {d['both']}/{d['one']}/{d['neither']} | "
            f"{fmt(co['median'], 2)} / {fmt(co['p90'], 2)} / {co['over_5m']} |")


TABLE_HEADER = (
    "| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | "
    "coverage | no cluster | recall (union) | self/other/unmatched | "
    "frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither "
    "| coherence med / p90 / >5 m |\n"
    "|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|")

def table_legend(results):
    """The legend under the arms table. The self-detected / pool counts are read
    from this run (they are frame-dependent: Richmond is 210/253 at 2.6 m and
    210/260 at 2.341 m), never hardcoded."""
    any_arm = next(iter(results.values()))
    n_self = any_arm['buckets']['self_detected']
    n_pool = any_arm['n_pool']
    share = f'{n_self / n_pool:.0%}' if n_pool else 'most'
    return (
        "**coverage** = pool GT ramps with a cluster of this arm within the match "
        "radius, matched one-to-one — the metric RQ2a asks for, and the only "
        "recall-shaped one that responds to the partition. **no cluster** = ramps "
        "counted as recalled by the union metric although no cluster is within the "
        "radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = "
        "`eval_sites`' definition, which counts a self-detected ramp as recovered "
        f"whether or not any cluster landed on it; {n_self} of this run's {n_pool} "
        f"pool ramps are self-detected, so {share} of it is constant across arms "
        "and it is kept only to tie back to `fusion_eval/report.md`. **frag** = "
        "share of covered GT ramps with at least one extra cluster within r that is "
        "not the one-to-one match of any GT ramp (total extras in parentheses). "
        "**coherence** = distance from a self-detected GT ramp to the centroid of "
        "the cluster holding that label.")


def csv_row(name, r):
    fr3, fr5, d, co = r['frag'][3.0], r['frag'][5.0], r['dual'], r['coherence']
    return {'arm': name, 'n_clusters': r['n_clusters'], 'n_placed': r['n_placed'],
            'n_labels': r['n_labels'], 'labels_per_cluster': r['labels_per_cluster'],
            'precision': r['precision'], 'tp': r['tp'], 'fp': r['fp'],
            'unsure': r['unsure'], 'coverage': r['coverage'],
            'covered': r['covered'], 'n_pool': r['n_pool'],
            'self_detected_without_cluster': r['self_detected_without_cluster'],
            'recall_union': r['recall'],
            'self_detected': r['buckets']['self_detected'],
            'other_view': r['buckets']['recovered_other_view'],
            'unmatched': r['buckets']['unmatched'],
            'frag3_ramps': fr3['ramps'], 'frag3_with_extra': fr3['with_extra'],
            'frag3_extra': fr3['extra'], 'frag5_with_extra': fr5['with_extra'],
            'frag5_extra': fr5['extra'], 'dual_pairs': d['pairs'], 'dual_both': d['both'],
            'dual_one': d['one'], 'dual_neither': d['neither'],
            'coh_n': co['n'], 'coh_median': co['median'], 'coh_p90': co['p90'],
            'coh_over_5m': co['over_5m'], 'same_pano_pairs': r['same_pano_pairs'],
            **size_columns(r.get('size'))}


SIZE_KEYS = {'unplaceable': 'sz_unpl', 'unclustered': 'sz_uncl', 'cluster of 1': 'sz_1',
             'cluster of 2': 'sz_2',
             'cluster of 3+': 'sz_3p'}


def size_columns(rows):
    """arms.csv columns for size_precision rows: n / judged / t / f per bucket, so a pooled
    table can sum numerators and denominators across cities."""
    out = {}
    for row in rows or ():
        k = SIZE_KEYS[row['bucket']]
        out.update({f'{k}_n': row['n'], f'{k}_judged': row['judged'],
                    f'{k}_t': row['t'], f'{k}_f': row['f']})
    return out


SIZE_BUCKETS = ('unplaceable', 'unclustered', 'cluster of 1', 'cluster of 2',
                'cluster of 3+')


def size_precision(clusters, det_of, det_pos, verdicts, conf, ys=None):
    """Rows (one per SIZE_BUCKETS entry) of GT precision by the size of the cluster
    holding each AI label. A label the raycast cannot place (not in det_pos) is
    `unplaceable` whatever partition holds it; a placeable one that no cluster of the
    partition holds is `unclustered` (the server's partition can leave a label out), never
    counted as a cluster of 1. `verdicts` is build_gt's (pano_id, det_index) -> verdict
    map; only True and False count toward precision. `ys` (key -> y_normalized) gives each
    row's median_y, where 0.5 is the horizon.

    Example:
        A label alone in its cluster, judged False, lands in 'cluster of 1' with
        t=0, f=1; one on a pano nobody judged adds to n but not to `judged`.
    """
    size_of = {}
    for c in clusters:
        for m in c.members:
            size_of[m] = c.n_labels
    keys = {b: [] for b in SIZE_BUCKETS}
    for key in det_of.values():
        if key not in det_pos:
            keys['unplaceable'].append(key)
        else:
            s = size_of.get(key)
            keys['unclustered' if s is None else 'cluster of 1' if s == 1
                 else 'cluster of 2' if s == 2 else 'cluster of 3+'].append(key)
    rows = []
    for b in SIZE_BUCKETS:
        ks = keys[b]
        v = [verdicts[k] for k in ks if k in verdicts]
        cs = sorted(conf[k] for k in ks if k in conf)
        yv = sorted(ys[k] for k in ks if ys and k in ys)
        rows.append({'bucket': b, 'n': len(ks), 'judged': len(v),
                     't': sum(x is True for x in v), 'f': sum(x is False for x in v),
                     'median_conf': quantile(cs, .5), 'median_y': quantile(yv, .5)})
    return rows


def wilson(t, n, z=1.96):
    """Wilson score interval for t successes in n trials; None when n == 0.

    Example:
        >>> [round(x, 2) for x in wilson(27, 27)]
        [0.88, 1.0]
    """
    if n == 0:
        return None
    p = t / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def precision_ci_text(t, f):
    ci = wilson(t, t + f)
    if ci is None:
        return 'n/a'
    return f'{t / (t + f):.3f} [{ci[0]:.2f}, {ci[1]:.2f}]'


def quantile(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


# --------------------------------------------------------- offline mode (issue #106)

API_STREETS = '/v3/api/streets?filetype=geojson'
# The label account offline-synthesized labels are attributed to. One account, so the
# same-(user, pano) cannot-link is exactly "same pano", as it is for the live AI user.
OFFLINE_USER = 'ai'
# Region id for every label of a city whose streets are unknown (no street feed), and
# the one region every label of a city with no server (and so no regions) shares.
NO_REGION = -1
CITY_REGION = 0
# How far (m) a pano may have moved since its labels were inserted before the offline
# check leaves its labels out: the server placed each label ONCE, from the pano position
# the campaign sent, and a later repositioning campaign moves the pano row, not the labels.
MOVED_PANO_M = 0.5
# The placement check fails when ANY live AI label -- moved panos included -- re-places
# more than this far from where the server put it. The moved-pano exclusion is decided
# by inverting the same estimator the check validates, so a placement error that is
# consistent within a pano (a heading offset, a systematic distance error) would be
# excluded as "moved" rather than failed; gating on every label closes that hole.
PLACEMENT_TOL_M = 0.5


def synthesize_labels(results_path, min_confidence, mask_rig=True):
    """(labels DataFrame, det_of) for a city with no server, or no server holding this
    run: one label per stored detection with confidence >= min_confidence, exactly as
    send_to_ps.transform_record would send it (pixel = round(x * width), round(y * height);
    rig detections dropped when mask_rig), placed where the server would place it
    (ps_placement.label_latlng at the server's 2.341 m).

    Records without a position or heading are skipped, as fuse_sites.load_results skips
    them. label_id is a per-run serial (unique only within this run; never pool on it),
    region_id is CITY_REGION until assign_regions sets it, and det_of maps each label_id to
    its (pano_id, det_index) directly -- no pixel lookup, so two detections that round to
    one pixel both keep their labels (the live path has to drop such keys).

    Example:
        One record at (47.0, -122.0), heading 0, width 4096, with one detection at
        x = 0.5, y = 0.6 and confidence 0.7 gives one label at pano_x 2048, pano_y 1229,
        due north of the camera, ~7.20 m out (18.02 degrees down at 2.341 m:
        2.341 / tan 18.02).
    """
    rows, det_of = [], {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            if p.get('lat') is None or p.get('lng') is None \
                    or p.get('camera_heading') is None:
                continue
            w, h = p['width'], p['height']
            for i, d in enumerate(rec.get('detections', [])):
                if d['confidence'] < min_confidence:
                    continue
                if mask_rig and on_camera_rig(d['y_normalized']):
                    continue
                px, py = round(d['x_normalized'] * w), round(d['y_normalized'] * h)
                lat, lng = ps_placement.label_latlng(p['lat'], p['lng'], px, py, w, h,
                                                     p['camera_heading'])
                lid = len(rows) + 1
                det_of[lid] = (p['panorama_id'], i)
                rows.append({'label_id': lid, 'user_id': OFFLINE_USER,
                             'pano_id': p['panorama_id'], 'region_id': CITY_REGION,
                             'lat': lat, 'lng': lng, 'pano_x': px, 'pano_y': py,
                             'severity': float('nan'), 'label_type': 'CurbRamp',
                             'pano_width': w, 'pano_height': h,
                             'camera_heading': p['camera_heading'],
                             'pano_source': p.get('source'),
                             'image_capture_date': p.get('capture_date'),
                             'confidence': d['confidence']})
    cols = ['label_id', 'user_id', 'pano_id', 'region_id', 'lat', 'lng', 'pano_x',
            'pano_y', 'severity', 'label_type', 'pano_width', 'pano_height',
            'camera_heading', 'pano_source', 'image_capture_date', 'confidence']
    return pd.DataFrame(rows, columns=cols), det_of


def load_streets(path):
    """[(street_edge_id, region_id, shapely LineString)] for the OPEN streets of a
    /v3/api/streets geojson.

    The server snaps an AI label to its nearest OPEN street only
    (LabelTable.getStreetEdgeIdClosestToLatLng queries streetEdgeTable.streetsWithTutorial,
    i.e. status == Open), while /v3/api/streets returns every street (open, no_imagery,
    closed/disabled) -- on Richmond only 704 of 16,365. Keeping them all put 1.4% of
    Richmond's live labels in the wrong region. The server's set also includes the
    tutorial street, which the API omits; no real label lies near it, so that difference
    is negligible. A pull without a `status` property cannot be filtered and is refused
    (re-pull it).
    """
    from shapely.geometry import shape
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    feats = [ft for ft in feats if ft.get('geometry')]
    if any('status' not in (ft.get('properties') or {}) for ft in feats):
        raise SystemExit(f'{path}: street features without a `status` property; cannot '
                         'keep open streets only (the server rule). Re-pull with --refresh.')
    return [(int(ft['properties']['street_edge_id']), int(ft['properties']['region_id']),
             shape(ft['geometry'])) for ft in feats
            if ft['properties']['status'] == 'open']


def streets_provenance(path, streets):
    """provenance() for a street pull, saying how many OPEN streets load_streets kept."""
    with open(path, encoding='utf-8') as f:
        n_all = len(json.load(f)['features'])
    return (provenance(path, n_all)
            + f'; {len(streets)} open streets kept (the server snaps to open streets only)')


def assign_regions(lat, lng, streets):
    """(region id array, n_ties): each point takes the region of its NEAREST street edge,
    which is how the server assigns an AI label's region at insert
    (ExploreService.submitAiLabelData: LabelTable.getStreetEdgeIdClosestToLatLng on the
    label's own lat/lng, then that street's region). Distances are taken on an
    equirectangular projection about the points' mean (the server uses
    ST_DistanceSphere; the two agree to well under a centimetre over label-to-street
    distances). A point equidistant from streets in different regions takes the lowest
    street_edge_id, and how many did is returned (should be ~0). With no streets every
    point gets NO_REGION.

    Example:
        >>> from shapely.geometry import LineString
        >>> st = [(1, 7, LineString([(0, 0), (0.001, 0)])),
        ...       (2, 9, LineString([(0, 0.001), (0.001, 0.001)]))]
        >>> assign_regions([0.0002, 0.0009], [0.0005, 0.0005], st)[0].tolist()
        [7, 9]
    """
    import shapely
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    out = np.full(len(lat), NO_REGION, dtype=int)
    if not streets or not len(lat):
        return out, 0
    k = math.cos(math.radians(float(lat.mean())))

    def proj(geom):
        return shapely.transform(geom, lambda c: np.column_stack([c[:, 0] * k, c[:, 1]]))

    lines = [proj(g) for _sid, _rid, g in streets]
    tree = shapely.STRtree(lines)
    pts = shapely.points(lng * k, lat)
    pt_idx, line_idx = tree.query_nearest(pts, all_matches=True)
    best = {}
    for i, j in zip(pt_idx.tolist(), line_idx.tolist()):
        sid, rid, _g = streets[j]
        best.setdefault(i, []).append((sid, rid))
    ties = 0
    for i, cands in best.items():
        cands.sort()
        out[i] = cands[0][1]
        ties += len({rid for _sid, rid in cands}) > 1
    return out, ties


def camera_from_label(lat, lng, pano_x, pano_y, width, height, camera_heading):
    """The camera position a server label implies: its lat/lng walked back along the
    server's own bearing by the server's own distance (the exact inverse of
    ps_placement.label_latlng to well under a millimetre at label ranges)."""
    dist, bearing = ps_placement.label_offset(pano_x, pano_y, width, height, camera_heading)
    return ps_placement.destination(lat, lng, dist, bearing + 180.0)


def offline_check(ai, det_of, results_path, streets, min_confidence, deployed,
                  humans=None, clustered=None, threshold_km=PS_THRESHOLD_KM):
    """Validate the offline server arm against a city that HAS live labels (issue #106).

    (a) placement: every live AI label is re-placed by ps_placement from the results
        file's pano block and its stored pixel, and compared with the server's lat/lng.
        A pano whose position the server's labels imply (camera_from_label, median over
        the pano's labels) sits more than MOVED_PANO_M from the file's is left out and
        counted: its labels were inserted from another pano position (a repositioning
        campaign after the file was written). That exclusion is self-referential (it
        inverts the estimator under test), so the check's pass/fail (`placement_ok`) is
        taken over ALL labels: max re-placement error <= PLACEMENT_TOL_M.
    (b) partition: the labels synthesized offline from the file (at min_confidence,
        unmasked), joined to the live AI labels on (pano_id, pano_x, pano_y) and given
        regions by nearest street (assign_regions), are clustered with the PS rule at
        7.5 m; the
        deployed clusters whose labels are all AI and all joined are then looked up in
        that partition by label set (partition_agreement). That is the gate. Offline
        there are no human labels, and complete linkage lets a nearby human label change
        how AI labels group, so (b+) repeats it with the live `humans` added at their live
        positions and regions, against every deployed cluster whose labels are all present:
        where (b) falls short and (b+) does not, the gap is the human labels, not the
        offline placement or regions. Both partitions are computed over the labels the
        server has clustered (`clustered`, label ids): a label inserted after the last
        nightly clustering is in no deployed cluster, and complete linkage would let it
        move others.

    Returns a dict of the measured numbers (the report formats them).
    """
    by_key = {}
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                p = json.loads(line)['pano']
                by_key[p['panorama_id']] = p
    offs, cams = {}, {}
    for r in ai.itertuples(index=False):
        p = by_key.get(r.pano_id)
        if p is None or p.get('lat') is None or p.get('camera_heading') is None:
            continue
        lat, lng = ps_placement.label_latlng(p['lat'], p['lng'], r.pano_x, r.pano_y,
                                             p['width'], p['height'], p['camera_heading'])
        offs[r.label_id] = geo.haversine_m(lat, lng, r.lat, r.lng)
        cams.setdefault(r.pano_id, []).append(camera_from_label(
            r.lat, r.lng, r.pano_x, r.pano_y, p['width'], p['height'],
            p['camera_heading']))
    moved = set()
    for pid, cs in cams.items():
        lat = float(np.median([c[0] for c in cs]))
        lng = float(np.median([c[1] for c in cs]))
        if geo.haversine_m(lat, lng, by_key[pid]['lat'], by_key[pid]['lng']) > MOVED_PANO_M:
            moved.add(pid)
    pano_of = dict(zip(ai.label_id, ai.pano_id))
    kept = sorted(o for lid, o in offs.items() if pano_of[lid] not in moved)
    every = sorted(offs.values())
    res = {'n_ai': len(ai), 'n_placed': len(offs), 'n_moved_panos': len(moved),
           'n_moved_labels': len(offs) - len(kept), 'n_kept': len(kept),
           'median': quantile(kept, .5), 'p90': quantile(kept, .9),
           'max': kept[-1] if kept else None,
           'over_0_5': sum(1 for o in kept if o > 0.5),
           'median_all': quantile(every, .5), 'p90_all': quantile(every, .9),
           'over_0_5_all': sum(1 for o in every if o > 0.5),
           'max_all': every[-1] if every else None}
    res['placement_ok'] = bool(every) and every[-1] <= PLACEMENT_TOL_M

    syn, _syn_det = synthesize_labels(results_path, min_confidence, mask_rig=False)
    live_key = {(r.pano_id, r.pano_x, r.pano_y): r.label_id
                for r in ai.itertuples(index=False)}
    syn['live_id'] = [live_key.get(k) for k in zip(syn.pano_id, syn.pano_x, syn.pano_y)]
    res['n_synth'] = len(syn)
    res['synth_not_live'] = int(syn.live_id.isna().sum())
    res['live_not_synth'] = len(ai) - int(syn.live_id.notna().sum())
    joined = syn[syn.live_id.notna() & ~syn.pano_id.isin(moved)].copy()
    joined['label_id'] = joined['live_id'].astype(int)
    joined = joined.drop_duplicates('label_id').reset_index(drop=True)
    res['unclustered'] = 0
    if clustered is not None:
        res['unclustered'] = int((~joined.label_id.isin(clustered)).sum())
        joined = joined[joined.label_id.isin(clustered)].reset_index(drop=True)
        if humans is not None:
            humans = humans[humans.label_id.isin(clustered)].reset_index(drop=True)
    reg, n_ties = assign_regions(joined.lat, joined.lng, streets)
    live_region = dict(zip(ai.label_id, ai.region_id))
    res['region_agree'] = int(sum(int(live_region[lid]) == int(g)
                                  for lid, g in zip(joined.label_id, reg)))
    res['n_joined'] = len(joined)
    res['region_ties'] = n_ties
    joined['region_id'] = reg
    part = ps_partition(joined, [threshold_km])[threshold_km]
    offline = clusters_from_assignment(joined, part, det_of)
    joined_ids = set(joined.label_id)
    ai_ids = set(ai.label_id)
    eligible = [c for c in deployed if c.label_ids
                and all(lab in ai_ids and lab in joined_ids for lab in c.label_ids)]
    k, n, off = partition_agreement(eligible, offline)
    res.update({'identical': k, 'eligible': n, 'off_labels': off,
                'n_deployed': len(deployed)})
    if humans is not None and len(humans):
        both = pd.concat([joined[list(humans.columns.intersection(joined.columns))],
                          humans[list(humans.columns.intersection(joined.columns))]],
                         ignore_index=True)
        part2 = ps_partition(both, [threshold_km])[threshold_km]
        offline2 = clusters_from_assignment(both, part2, det_of)
        present = set(both.label_id)
        eligible2 = [c for c in deployed if c.label_ids
                     and all(lab in present for lab in c.label_ids)]
        k2, n2, off2 = partition_agreement(eligible2, offline2)
        res.update({'identical_h': k2, 'eligible_h': n2, 'off_labels_h': off2,
                    'n_humans': len(humans)})
    return res


# Unplaceable labels (no raycast position: at/above the horizon or beyond the 25 m cap)
# get one pre-declared association rule, fixed before any result (#106) and never tuned on
# GT: attach to a placed site of the same fuse that lies ON the label's bearing ray.
# 15 m: where the server's bounded tail departs from the flat raycast (INVERT_MAX_RANGE_M);
#   nearer than that a label would have been placeable, so a site there is another ramp.
# 60 m: about where a curb ramp stops being resolvable in a 4096 px equirectangular.
# 3 m:  the lateral same-ramp scatter of a multi-view site.
ATTACH_PERP_M = 3.0
ATTACH_MIN_RANGE_M = 15.0
ATTACH_MAX_RANGE_M = 60.0


def attach_unplaceable(sites, panos, frame, perp_m=ATTACH_PERP_M,
                       min_range_m=ATTACH_MIN_RANGE_M, max_range_m=ATTACH_MAX_RANGE_M,
                       mask_rig=False, unpositioned=(), label_of=None):
    """(clusters, attached) for the `fusion_server+attach` arm.

    Every label of `panos` that no site holds (clusters_from_server_sites makes these
    singletons) is tested against the placed sites of the same fuse (`sites`, positions
    in `frame`): along the label's bearing from its camera -- the server's own heading,
    camera_heading - 180 + x * 360, which is the labeler's azimuth -- a site qualifies
    when it lies between min_range_m and max_range_m ahead and within perp_m of the ray,
    and does not already hold a label from the same pano (fusion's cannot-link). The label
    joins the qualifying site nearest along the ray as a non-refit member (the site does
    not move); otherwise it stays a singleton. Labels are processed in descending
    confidence (then pano id, index), so the cannot-link is deterministic.

    Only labels the raycast dropped for the horizon or the range cap are candidates: with
    mask_rig (the fuse's FuseParams.mask_rig), a label in the rig band was dropped as
    `on_rig` -- it is on the camera vehicle, not the street -- and stays a singleton.

    `unpositioned` and `label_of` are clusters_from_server_sites' (server_panos' stats):
    the labels on panos with no position stay one singleton each, and every cluster
    carries its label ids, so this arm holds every live label exactly once too.

    Returns the clusters (sites with their attachments, then the remaining singletons,
    in clusters_from_server_sites' layout) and {(pano_id, det_index): site id}.

    Example:
        A label looking due north from (0, 0) attaches to a site at (1.0, 30.0) (30 m
        ahead, 1 m off the ray), not to one at (0, 10.0) (too near) or (5.0, 30.0) (5 m off).
    """
    held = {(d.pano_id, d.det_index) for s in sites for d, _ in s.members}
    site_panos = {s.id: set(s.pano_ids) for s in sites}
    xy = np.array([[s.e, s.n] for s in sites]) if sites else np.zeros((0, 2))
    tree = cKDTree(xy) if sites else None
    loose = []
    for p in panos:
        for i, x, y, conf in p.detections:
            if (p.pano_id, i) not in held:
                loose.append((-conf, p.pano_id, i, x, p, y))
    loose.sort(key=lambda t: t[:3])
    attached = {}
    for _negc, pid, i, x, p, y in loose:
        if tree is None:
            break
        if mask_rig and on_camera_rig(y):
            continue
        ce, cn = frame.to_enu(p.lat, p.lng)
        b = math.radians(p.camera_heading - 180.0 + x * 360.0)
        ue, un = math.sin(b), math.cos(b)
        best = None
        for j in tree.query_ball_point([ce, cn], max_range_m + perp_m):
            de, dn = xy[j, 0] - ce, xy[j, 1] - cn
            along = de * ue + dn * un
            perp = abs(de * un - dn * ue)
            if not (min_range_m <= along <= max_range_m and perp <= perp_m):
                continue
            sid = sites[j].id
            if pid in site_panos[sid]:
                continue
            if best is None or (along, sid) < best:
                best = (along, sid)
        if best is not None:
            attached[(pid, i)] = best[1]
            site_panos[best[1]].add(pid)
    extra = {}
    for key, sid in attached.items():
        extra.setdefault(sid, []).append(key)
    label_of = label_of or {}
    out = []
    for s in sites:
        keys = [(d.pano_id, d.det_index) for d, _ in s.members] + extra.get(s.id, [])
        out.append(Cluster(s.id, [k for k in keys if k[1] < HUMAN_DET_BASE], len(keys),
                           label_ids=[label_of[k] for k in keys if k in label_of]))
    for _negc, pid, i, _x, _p, _y in sorted(loose, key=lambda t: (t[1], t[2])):
        if (pid, i) not in attached:
            out.append(Cluster(len(out), [(pid, i)] if i < HUMAN_DET_BASE else [], 1,
                               label_ids=[label_of[(pid, i)]] if (pid, i) in label_of
                               else []))
    for lab in unpositioned:
        out.append(Cluster(len(out), [], 1, label_ids=[lab]))
    return out, attached


# ----------------------------------------------------------------------------- main

# Bumped by hand whenever a change moves any number the reports print; the pooled driver
# (clustering_eval_pooled.py) re-runs a cell whose report records another version.
SCORER_VERSION = '106.3'


def repo_relative(path):
    """`path` as POSIX relative to the repo root, so a committed report names no machine's
    checkout; a path outside the repo prints as its file name."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def input_stamp(streets_path, verdicts_path):
    """The report-head line stamping the street pull and the RampNet verdicts by sha256
    (`none` when absent), read back by clustering_eval_pooled.cell_current."""
    def sha(p):
        return fs.file_sha256(p) if p and Path(p).exists() else 'none'
    return f'inputs: streets sha256 `{sha(streets_path)}`; verdicts sha256 `{sha(verdicts_path)}`'


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city')
    ap.add_argument('--split', default=None,
                    help='RampNet benchmark split, when it is not named like the city '
                         '(laurens -> laurens_mapillary)')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--results', type=Path, default=None,
                    help="the run's results file (default <run-dir>/results.jsonl; Laurens "
                         'went live from results.raw.jsonl)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/ps_clustering_eval, or '
                         'ps_clustering_eval_offline<frame>_t<tier> with --offline)')
    ap.add_argument('--offline', action='store_true',
                    help='no server labels: synthesize one label per stored detection >= '
                         '--min-confidence (rig-masked) and place it with the server\'s own '
                         'estimator (ps_placement). --server then only supplies the street '
                         'network, for regions')
    ap.add_argument('--offline-check', action='store_true',
                    help='(live mode) validate the offline arm against this live server: '
                         'per-label placement and the 7.5 m partition')
    ap.add_argument('--streets', type=Path, default=None,
                    help="a /v3/api/streets geojson, for each label's region (its nearest "
                         "street's, as the server assigns it; default: pulled with --server "
                         'into the output dir when --offline or --offline-check needs it)')
    ap.add_argument('--mask-rig', action='store_true',
                    help='(live mode) the fusion arms drop camera-rig detections too: for a '
                         "city whose live AI labels are rig-masked (Laurens: its rig labels "
                         'were soft-deleted), so fusion scores the label set the server holds. '
                         'Offline mode always masks')
    ap.add_argument('--server', default=None,
                    help='PS server URL; downloads the geojson files if absent (GET only)')
    ap.add_argument('--refresh', action='store_true',
                    help='with --server, re-pull the geojson over the cached copies '
                         '(the server re-clusters nightly, so a cached pull goes stale)')
    ap.add_argument('--labels', type=Path, default=None)
    ap.add_argument('--clusters', type=Path, default=None)
    ap.add_argument('--ps-script', type=Path, default=None,
                    help='SidewalkWebpage/scripts/label_clustering.py for the verbatim arm')
    ap.add_argument('--thresholds-m', type=float, nargs='+',
                    default=[2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    ap.add_argument('--match-radius-m', type=float, default=5.0)
    ap.add_argument('--radius-sweep', type=float, nargs='*', default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument('--gt-merge-m', type=float, default=2.5)
    ap.add_argument('--camera-height-m', type=fs.fuse_camera_height_arg,
                    default=geo.DEFAULT_CAMERA_HEIGHT_M,
                    help='raycast camera height for every labeler placement (detections, '
                         "GT, ps_raycast, fusion; arms on the server's own positions are "
                         "untouched): metres (default 2.6; the server's own frame is "
                         '2.341219672825709), or "per-pano" (GSV depth-measured heights, '
                         'from the pano block else runs/<city>/depth/index.csv; unmeasured '
                         "panos fall back to 2.6), \"auto\" (fuse_sites' per-rig GSV rule, "
                         "#79) or \"per-rig\" (the run's camera_heights.json, #53), resolved "
                         'exactly as fuse_sites.py does (#56). The report states the '
                         'resolution; a non-default frame writes its own output dir')
    ap.add_argument('--min-confidence', type=float, default=BENCHMARK_CONFIDENCE,
                    help='the tier the fusion arm runs at. Defaults to the BENCHMARK tier '
                         f'({BENCHMARK_CONFIDENCE}), not the operating point: this script '
                         'scores what the SERVER holds, and every city clustered so far went '
                         'live at 0.55. Set it to whatever a city was actually submitted at '
                         '(its submission record says) before comparing arms. With --offline '
                         'it is also the tier the labels are synthesized at.')
    return ap


def default_out(args, run_dir):
    """The output dir a run writes to when --out is not given: the scoring frame and, for
    an offline run, the tier are part of the result, so each gets its own directory."""
    suffix = fs.frame_suffix(args.camera_height_m)
    if args.offline:
        return run_dir / f'ps_clustering_eval_offline{suffix}_t{args.min_confidence:g}'
    return run_dir / ('ps_clustering_eval' + suffix)


def run(args):
    """Score every arm for one city, write report.md + arms.csv, and return
    {'results': {arm: score}, 'out': Path, 'attach': {...}, 'meta': {...}} for the pooled
    driver (scripts/clustering_eval_pooled.py)."""
    if args.offline and (args.labels or args.clusters or args.ps_script or args.offline_check):
        raise SystemExit('--offline synthesizes the labels: --labels, --clusters, '
                         '--ps-script and --offline-check need a live server')
    split = args.split or args.city
    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    results_path = args.results or run_dir / 'results.jsonl'
    out = args.out or default_out(args, run_dir)
    out.mkdir(parents=True, exist_ok=True)
    server = args.server.rstrip('/') if args.server else None
    if args.refresh and not server:
        raise SystemExit('--refresh needs --server (there is nothing to re-pull from)')

    streets_path = args.streets
    if streets_path is None and server and (args.offline or args.offline_check):
        streets_path = out / 'streets.geojson'
        fetch(server + API_STREETS, streets_path, args.refresh)
    if args.offline_check and streets_path is None:
        raise SystemExit('--offline-check needs the street network: --streets, or --server '
                         'to pull it')
    streets = load_streets(streets_path) if streets_path else []
    results_sha = fs.file_sha256(results_path)

    if args.offline:
        labels, det_of = synthesize_labels(results_path, args.min_confidence, mask_rig=True)
        n_rig = synthesize_rig_count(results_path, args.min_confidence)
        n_ties = 0
        if streets:
            reg, n_ties = assign_regions(labels.lat, labels.lng, streets)
            labels['region_id'] = reg
        ai = labels
        # every synthesized label is the AI account's and maps to its detection
        ai_user, unmapped_ai = OFFLINE_USER, 0
        server_clusters = None
        n_bad_lng = n_ambiguous = n_dup_labels = 0
        pix_map, n_ambiguous, _dup = label_to_detection(results_path, labels)
        n_pix_same = sum(1 for lid, m in pix_map.items() if det_of.get(lid) == m)
        lines = [f'# {args.city}: PS label clustering vs RampNet GT (offline)',
                 '',
                 f'mode: offline -- {len(labels)} labels synthesized from '
                 f'`{results_path.name}`, one per stored detection >= '
                 f'{args.min_confidence:g} ({n_rig} on the camera rig left out), each placed '
                 "where the server would place it (ps_placement: the server's estimator at "
                 f'{ps_placement.CAMERA_HEIGHT_M:.3f} m); no server labels or clusters, so '
                 '`deployed` and `ps_repro` do not exist here']
    else:
        labels_path = args.labels or out / 'raw_labels.geojson'
        clusters_path = args.clusters or out / 'clusters.geojson'
        if server:
            fetch(server + API_LABELS, labels_path, args.refresh)
            fetch(server + API_CLUSTERS, clusters_path, args.refresh)
        labels, n_bad_lng = load_labels(labels_path)
        labels = labels[labels.label_type == 'CurbRamp'].reset_index(drop=True)
        server_clusters = load_server_clusters(clusters_path)
        det_of, n_ambiguous, n_dup_labels = label_to_detection(results_path, labels)
        ai = labels[labels.label_id.isin(det_of)].reset_index(drop=True)
        # "AI label" is inferred from the pixel match; make sure that inference picks out
        # exactly one submitting account, otherwise a human label at the same pixel as a
        # detection would be scored as an AI label.
        ai_users = sorted({str(u) for u in ai.user_id})
        if len(ai_users) != 1:
            raise SystemExit('labels that map to stored detections span '
                             f'{len(ai_users)} user_ids ({", ".join(ai_users[:5])}); '
                             'the pixel map cannot be read as "the AI user\'s labels"')
        ai_user = ai_users[0]
        unmapped_ai = int((labels.user_id.astype(str) == ai_user).sum()) - len(ai)
        by_user = labels.user_id.astype(str).value_counts()
        user_breakdown = ', '.join(
            f'{u}{" (AI)" if u == ai_user else ""} {int(c)}'
            for u, c in by_user.items())
        lines = [f'# {args.city}: PS label clustering vs RampNet GT',
                 '',
                 f'labels: {len(labels)} CurbRamp on the server, {len(ai)} map to stored '
                 f'detections (AI), {len(labels) - len(ai)} do not (human); '
                 f'{len(server_clusters)} server clusters over '
                 f'{sum(len(c["label_ids"]) for c in server_clusters)} labels']
    lines.append(f'scorer {SCORER_VERSION}; results `{results_path.name}` sha256 '
                 f'`{results_sha}`' + (f'; benchmark split `{split}`' if split != args.city
                                       else ''))
    # the other two inputs a number can move with; clustering_eval_pooled.cell_current
    # checks both, so a --refresh'ed street pull or an updated benchmark re-runs the cell
    lines.append(input_stamp(streets_path, args.benchmark_root / split / 'verdicts.json'))

    # world frame, raycast positions, GT — one code path with eval_sites
    # mask_rig: live, this arm is compared against what the SERVER holds, and Richmond's
    # labels were submitted before the nadir mask existed, so it is off (--mask-rig for a
    # city whose live labels are masked, e.g. Laurens); offline, every
    # arm scores the same rig-masked label set a run submitted today would ship.
    # apply_pose=OFF likewise: the server placed those labels with a flat raycast, and the
    # committed report was scored flat -- FuseParams' `auto` default would rotate Mapillary.
    # The camera height resolves through fs.load_at_height, the resolver fuse_sites and
    # eval_sites share (#56): per-pano / auto / per-rig mean the same thing everywhere.
    try:
        verdict_panos, bundle_ops, run_panos, height, auto = es.load_city_at_height(
            split, args.benchmark_root, run_dir, args.camera_height_m,
            results_path=results_path)
    except ValueError as e:        # no per-rig table, or one measured on another file
        raise SystemExit(str(e))
    params = fs.FuseParams(camera_height_m=height,
                           min_confidence=args.min_confidence,
                           mask_rig=args.offline or args.mask_rig, apply_pose=fs.POSE_OFF)
    lines.append(f'raycast camera height {fs.frame_label(args.camera_height_m)}; '
                 f'fusion arm at --min-confidence {args.min_confidence:g}')
    lines += fs.height_resolution_lines(args.camera_height_m, run_panos, params, auto)
    dets, frame, drops = fs.project(run_panos, params)
    det_pos = {(d.pano_id, d.det_index): (d.e, d.n) for d in dets}
    points, op_verdicts, counts, warnings = es.build_gt(
        verdict_panos, bundle_ops, {p.pano_id: p for p in run_panos}, params, frame)
    ramps = es.merge_gt_points(points, args.gt_merge_m)
    pool = [r for r in ramps if r.in_pool]
    gt = (ramps, pool, points, op_verdicts)
    lines.append(f"GT: {counts['judged']} judged panos -> {counts['placeable']} placeable "
                 f"points -> {len(ramps)} ramps ({len(points) - len(ramps)} cross-pano "
                 f"merges), {len(pool)} in the recall pool; raycast placed {len(dets)} of "
                 f"{sum(len(p.detections) for p in run_panos)} detections "
                 f"(drops {drops})")
    for w in warnings:
        lines.append(f'warning: {w}')

    results, partitions = {}, {}

    def add_arm(name, clusters):
        partitions[name] = clusters
        results[name] = score(clusters, gt, det_pos, args.match_radius_m)

    checks = []
    deployed = None
    if not args.offline:
        # arm: deployed
        deployed = place(clusters_from_server(server_clusters, det_of), det_pos)
        add_arm('deployed', deployed)

        # descriptive: server centroid vs raycast centroid; per-label placement offset
        offs = []
        for c in deployed:
            if c.e is not None and c.server_latlng:
                lat, lng = frame.to_latlng(c.e, c.n)
                offs.append(geo.haversine_m(lat, lng, *c.server_latlng))
        offs.sort()
        lab_offs = []
        for row in ai.itertuples(index=False):
            m = det_of[row.label_id]
            if m in det_pos:
                lat, lng = frame.to_latlng(*det_pos[m])
                lab_offs.append(geo.haversine_m(lat, lng, row.lat, row.lng))
        lab_offs.sort()
        sizes = {}
        for c in deployed:
            sizes[c.n_labels] = sizes.get(c.n_labels, 0) + 1

        # arm: ps_repro (verbatim script) — on the labels the server actually clustered
        clustered_ids = {lab for c in server_clusters for lab in c['label_ids']}
        on_server = labels[labels.label_id.isin(clustered_ids)].reset_index(drop=True)
        if args.ps_script:
            assign, t_km = ps_verbatim(on_server, args.ps_script)
            repro = place(clusters_from_assignment(on_server, assign, det_of), det_pos)
            add_arm('ps_repro', repro)
            k, n, off = partition_agreement(deployed, repro)
            checks.append(f'ps_repro reproduces deployed: {k}/{n} clusters identical '
                          f'({off} labels in clusters that differ); script threshold '
                          f'{t_km} km')
            vec = ps_partition(on_server, [PS_THRESHOLD_KM])[PS_THRESHOLD_KM]
            vec_clusters = clusters_from_assignment(on_server, vec, det_of)
            k2, n2, off2 = partition_agreement(repro, vec_clusters)
            checks.append(f'vectorized PS distance reproduces the script: {k2}/{n2} '
                          f'clusters identical ({off2} labels differ)')

    # arms: ps @ t (server positions, per region) and ps_citywide @ 7.5
    t_kms = [t / 1000.0 for t in args.thresholds_m]
    block_stats = {}
    parts = ps_partition(ai, t_kms, per_region=True, stats=block_stats)
    for t_m, t_km in zip(args.thresholds_m, t_kms):
        add_arm(f'ps @ {t_m:g} m',
                place(clusters_from_assignment(ai, parts[t_km], det_of), det_pos))
    # Blocked (#106), so it runs at any city size: no N x N matrix over the whole city.
    city = ps_partition(ai, [PS_THRESHOLD_KM], per_region=False)[PS_THRESHOLD_KM]
    add_arm('ps_citywide @ 7.5 m',
            place(clusters_from_assignment(ai, city, det_of), det_pos))

    # arms: ps_placeable @ t — the PS algorithm on server positions, restricted to the
    # labels the raycast can place: exactly the label set ps_raycast and fusion use, so
    # against ps_raycast the only difference is the positions
    placeable = ai[[det_of[lab] in det_pos for lab in ai.label_id]].reset_index(drop=True)
    parts_pl = ps_partition(placeable, t_kms, per_region=True)
    for t_m, t_km in zip(args.thresholds_m, t_kms):
        add_arm(f'ps_placeable @ {t_m:g} m',
                place(clusters_from_assignment(placeable, parts_pl[t_km], det_of), det_pos))

    # arms: ps_raycast @ t — same algorithm, labeler raycast positions
    keep, ray_lat, ray_lng = [], [], []
    for i, row in enumerate(ai.itertuples(index=False)):
        m = det_of[row.label_id]
        if m in det_pos:
            lat, lng = frame.to_latlng(*det_pos[m])
            keep.append(i)
            ray_lat.append(lat)
            ray_lng.append(lng)
    ray = ai.iloc[keep].copy().reset_index(drop=True)
    ray['lat'] = ray_lat
    ray['lng'] = ray_lng
    parts_ray = ps_partition(ray, t_kms, per_region=True)
    for t_m, t_km in zip(args.thresholds_m, t_kms):
        add_arm(f'ps_raycast @ {t_m:g} m',
                place(clusters_from_assignment(ray, parts_ray[t_km], det_of), det_pos))

    # arm: fusion (mean-of-members placement, and refit position as the tie-back)
    sites, _frame2, _stats = fs.fuse(run_panos, params)
    add_arm('fusion', place(clusters_from_sites(sites, False), det_pos))
    add_arm('fusion_refit', clusters_from_sites(sites, True))

    # arm: fusion_server — the same associator on the server's labels (AI and human)
    # instead of the run's detections: what the server would compute if its clustering
    # were fusion. Every label on the server is live, so all of them are operational.
    # Offline, "the server's labels" are the synthesized ones.
    run_by_id = {p.pano_id: p for p in run_panos}
    if unmapped_ai:
        raise SystemExit(
            f'{unmapped_ai} labels of the AI account ({ai_user}) map to no stored detection '
            f'in {run_dir / "results.jsonl"}: they were submitted from another file (a band, '
            'a re-inferred campaign), so fusion_server cannot give them a confidence, and '
            'read as human they would seed sites at 1.0. Score a pull that predates them, '
            'or point --run-dir at the file they came from.')
    srv_panos, srv_stats = server_panos(
        labels, det_of, run_by_id, ai_user,
        decode=fs.single_decode(Counter(p.decode for p in run_panos), 'results.jsonl'),
        border=fs.single_border(Counter(p.border for p in run_panos), 'results.jsonl'),
        invert=not args.offline)
    srv_params = replace(params, min_confidence=0.0, floor=0.0)
    srv_sites, srv_frame, _s3 = fs.fuse(srv_panos, srv_params)
    srv_clusters, srv_singletons = clusters_from_server_sites(
        srv_sites, srv_panos, srv_stats['unplaceable_label_ids'], srv_stats['label_of'])
    add_arm('fusion_server', place(srv_clusters, det_pos))
    inv_err = inversion_check(labels, run_by_id)
    inv_err_human = inversion_check(labels, run_by_id, exclude_user=ai_user)

    # arm: fusion_server+attach — the same sites, plus the pre-declared bearing rule for
    # the labels the raycast cannot place (attach_unplaceable). Their positions are
    # unknown to the scorer, so coverage, frag, dual and coherence cannot move; cluster
    # counts, size buckets and cluster-level precision can (attaching a judged-true
    # singleton to a site that is already TP removes one TP cluster).
    att_clusters, attached = attach_unplaceable(
        srv_sites, srv_panos, srv_frame, mask_rig=srv_params.mask_rig,
        unpositioned=srv_stats['unplaceable_label_ids'], label_of=srv_stats['label_of'])
    add_arm('fusion_server+attach', place(att_clusters, det_pos))
    site_true = {}
    for s in srv_sites:
        site_true[s.id] = any(op_verdicts.get((d.pano_id, d.det_index)) is True
                              for d, _ in s.members)
    judged_att = [(k, sid) for k, sid in attached.items() if k in op_verdicts]
    attach_stats = {
        'unplaceable': srv_singletons, 'attached': len(attached),
        'clusters_server': len(srv_clusters), 'clusters_attach': len(att_clusters),
        'judged_unplaceable': sum(1 for p in srv_panos for i, *_r in p.detections
                                  if (p.pano_id, i) in op_verdicts
                                  and (p.pano_id, i) not in det_pos),
        'judged_attached': len(judged_att),
        'judged_attached_true': sum(1 for k, _s in judged_att if op_verdicts[k] is True),
        'judged_attached_to_true_site': sum(1 for _k, sid in judged_att if site_true[sid]),
    }

    # mechanism: same-ramp scatter. Over fusion sites with >= 3 placeable members, the
    # largest pairwise distance among the members under server positions vs raycast
    # positions — complete linkage at t keeps a group together only if this is <= t.
    #
    # The fusion arm is built from the RUN's detections and the ps_* arms from the
    # SERVER's labels. They coincide only when every placeable operational detection
    # was submitted; a gap-filled pano or a partial campaign breaks that, so a member
    # with no server label is skipped and counted rather than raising a KeyError here
    # (and the equality is reported as a validation check below).
    server_ll = {det_of[r.label_id]: (r.lat, r.lng) for r in ai.itertuples(index=False)}
    n_fusion_members = n_fusion_unsubmitted = 0
    spread = {'server': [], 'raycast': []}
    for s in sites:
        mem = [(d.pano_id, d.det_index) for d, _ in s.members
               if d.operational and (d.pano_id, d.det_index) in det_pos]
        n_fusion_members += len(mem)
        n_fusion_unsubmitted += sum(1 for m in mem if m not in server_ll)
        mem = [m for m in mem if m in server_ll]
        if len(mem) < 3:
            continue
        for key, pts in (('server', [server_ll[m] for m in mem]),
                         ('raycast', [frame.to_latlng(*det_pos[m]) for m in mem])):
            spread[key].append(max(geo.haversine_m(*pts[i], *pts[j])
                                   for i in range(len(pts))
                                   for j in range(i + 1, len(pts))))
    for v in spread.values():
        v.sort()

    # radius sweep for the two arms that matter most (offline: ps @ 7.5 m stands in for
    # the deployed clustering, which it reproduces where a server exists)
    ref_name = 'deployed' if not args.offline else f'ps @ {PS_THRESHOLD_KM * 1000:g} m'
    ref_clusters = partitions.get(ref_name)
    sweep = []
    if ref_clusters is not None:
        for r_m in args.radius_sweep:
            sweep.append((r_m,
                          score(ref_clusters, gt, det_pos, r_m),
                          score(place(clusters_from_sites(sites, False), det_pos), gt,
                                det_pos, r_m)))

    run_conf = {(p.pano_id, i): c for p in run_panos for i, _x, _y, c in p.detections}
    run_y = {(p.pano_id, i): y for p in run_panos for i, _x, y, _c in p.detections}
    for name, cl in partitions.items():
        results[name]['size'] = size_precision(cl, det_of, det_pos, gt[3], run_conf, run_y)

    ocheck = None
    if args.offline_check:
        humans = labels[~labels.label_id.isin(det_of)].reset_index(drop=True)
        ocheck = offline_check(ai, det_of, results_path, streets, args.min_confidence,
                               deployed, humans=humans, clustered=clustered_ids)

    # ---- report
    lines += ['', '## Data provenance', '']
    if args.offline:
        lines += [f'- results file `{repo_relative(results_path)}`: sha256 `{results_sha}`',
                  f'- {n_ambiguous} ambiguous pixel keys in `{results_path.name}` (two '
                  'stored detections round to one pixel); offline labels map to their '
                  f'detection directly, and the pixel-key map agrees on {n_pix_same} of '
                  f'{len(labels)}']
    else:
        lines += [provenance(labels_path, len(labels)),
                  provenance(clusters_path, len(server_clusters)),
                  f'- labels by account: {user_breakdown}',
                  f'- {n_bad_lng} labels dropped before clustering (null lng or lng > 360), '
                  'matching label_clustering.clean_label_data',
                  f'- {n_ambiguous} ambiguous pixel keys in {results_path.name} (two stored '
                  'detections round to one pixel; those keys are left unmapped)',
                  f'- {n_dup_labels} server labels share a pixel with another label and so '
                  'map to the same stored detection (a re-submitted campaign does this)']
    if streets_path:
        lines.append(streets_provenance(streets_path, streets))
    if args.offline:
        if streets:
            lines.append('- regions: every synthesized label takes the region of the street '
                         'nearest its server position, as the server assigns it at insert; '
                         f'{n_ties} labels were equidistant from streets in two regions '
                         '(lowest street_edge_id taken)')
        else:
            lines.append('- regions: none (no server), so every label is in one region and '
                         'per-region == citywide')
    lines.append(f"- PS partitions are blocked (single-linkage components at the widest "
                 f"threshold + {BLOCK_MARGIN_M:g} m): {block_stats.get('n_blocks')} blocks, "
                 f"largest {block_stats.get('largest_block')} labels")

    lines += ['', '## Validation checks', '']
    if not args.offline and not args.ps_script:
        checks.insert(0, 'ps_repro skipped (no --ps-script)')
    lines += [f'- {c}' for c in checks]

    def check_line(text, ok):
        return f'- {"" if ok else "warning: "}{text}'

    if not args.offline:
        lines.append(check_line(
            'every label that maps to a stored detection belongs to one account '
            f'({ai_user}); {unmapped_ai} of that account\'s labels did not map '
            '(should be 0)', unmapped_ai == 0))
    ps_pl = results.get(f'ps_placeable @ {args.thresholds_m[0]:g} m')
    lines.append(check_line(
        'fusion arm vs ps_* arms cover the same labels: '
        f"{results['fusion']['n_labels']} fusion members vs "
        f"{ps_pl['n_labels'] if ps_pl else 'n/a'} placeable server labels; "
        f'{n_fusion_unsubmitted} of {n_fusion_members} placeable operational '
        'detections have no label on the server (should be 0; they are excluded '
        'from the scatter below)',
        n_fusion_unsubmitted == 0
        and (ps_pl is None or ps_pl['n_labels'] == results['fusion']['n_labels'])))
    fr = results['fusion_refit']
    # The published fusion_eval numbers were produced in the labeler's default frame,
    # so this only reproduces them when this run is scored at that height; at any other
    # --camera-height-m the world-space columns are expected to differ. Judged on what the
    # height RESOLVED to: per-pano/per-rig with no pano measured (every Mapillary city) or
    # auto -> 2.6 is the published frame, whatever the mode is called.
    resolved = fs.resolved_height_counts(run_panos, params, auto)
    same_frame = (params.camera_height_m == geo.DEFAULT_CAMERA_HEIGHT_M
                  or resolved.get('measured', resolved.get('applied')) == 0)
    lines.append(
        f"- fusion_refit at {fs.frame_label(args.camera_height_m)} vs "
        f"runs/{args.city}/fusion_eval/"
        f"report.md (published in the {geo.DEFAULT_CAMERA_HEIGHT_M:g} m frame): precision "
        f"{fmt(fr['precision'])}, recall (union) {fmt(fr['recall'])}, dual "
        f"{fr['dual']['both']}/{fr['dual']['one']}/{fr['dual']['neither']}"
        + ('' if same_frame else ' — a different frame, so the world-space figures are '
                                 'expected to differ; precision is the frame-free part'))
    fus_sets = {frozenset(c.members) for c in clusters_from_sites(sites, False)}
    srv_sets = [frozenset(c.members) for c in srv_clusters[:len(srv_sites)]
                if c.members]
    same_as_fusion = sum(1 for s in srv_sets if s in fus_sets)
    lines.append(
        f"- fusion_server input: {srv_stats['ai_labels']} AI (by account; "
        f"{srv_stats['ai_duplicate']} share a detection with another AI label) + "
        f"{srv_stats['human_labels']} human labels on "
        f"{len(srv_panos) + srv_stats['unplaceable']} panos ({srv_stats['inverted']} "
        f"positioned by inverting their labels, {srv_stats['run_position']} from the run's "
        + ("pano block (offline: the block the labels were placed from), " if args.offline
           else f"pano block (no label within {INVERT_MAX_RANGE_M:g} m to invert), ")
        + f"{srv_stats['unplaceable']} with neither, whose {srv_stats['unplaceable_labels']} "
        f'labels are singleton clusters); {srv_singletons} labels the raycast cannot place '
        '(range cap, horizon) are singleton clusters; '
        f'{same_as_fusion} of its {len(srv_sets)} clusters with AI members are, member '
        f"for member, a cluster of the `fusion` arm")
    srv_ids = sorted(lab for c in srv_clusters for lab in c.label_ids)
    lines.append(check_line(
        f'every server label is in exactly one fusion_server cluster: {len(srv_ids)} label '
        f'ids, {len(set(srv_ids))} distinct, of {len(labels)} labels',
        srv_ids == sorted(labels.label_id)))
    lines.append(check_line(
        f"inverted camera positions more than 1 m from the run's pano block (a pano live "
        f"somewhere other than results.jsonl says, e.g. repositioned): "
        f"{srv_stats['inverted_far']} of {srv_stats['inverted']} (should be 0 for a pull "
        'taken before any reposition)', srv_stats['inverted_far'] == 0))
    lines.append(
        f"- camera-position inversion vs the run's position, over {len(inv_err)} panos in "
        f'both (all labels, mostly AI): median {fmt(quantile(inv_err, .5), 3)} m, p90 '
        f'{fmt(quantile(inv_err, .9), 2)} m; from human labels only, over '
        f'{len(inv_err_human)} panos: median {fmt(quantile(inv_err_human, .5), 3)} m, p90 '
        f'{fmt(quantile(inv_err_human, .9), 2)} m, max '
        f'{fmt(inv_err_human[-1] if inv_err_human else None, 2)} m')
    bad = {k: v['same_pano_pairs'] for k, v in results.items() if v['same_pano_pairs']}
    lines.append('- same-pano pairs inside one cluster (must be 0 under the cannot-link): '
                 + (', '.join(f'{k} {v}' for k, v in bad.items()) if bad
                    else '0 in every arm'))
    lines += ['', f'## Arms (match radius {args.match_radius_m:g} m, GT merge '
              f'{args.gt_merge_m:g} m)', '', TABLE_HEADER]
    lines += [row_line(k, v) for k, v in results.items()]
    lines += ['', table_legend(results)]
    if not args.offline:
        lines += ['', '## Deployed clusters, descriptive', '',
                  '- cluster size histogram (labels -> clusters): '
                  + ', '.join(f'{k}: {v}' for k, v in sorted(sizes.items())),
                  f'- server centroid vs raycast centroid, same members (n={len(offs)}): '
                  f'median {fmt(quantile(offs, .5), 1)} m, p90 '
                  f'{fmt(quantile(offs, .9), 1)} m',
                  f'- per-label server lat/lng vs labeler raycast (n={len(lab_offs)}): '
                  f'median {fmt(quantile(lab_offs, .5), 1)} m, p90 '
                  f'{fmt(quantile(lab_offs, .9), 1)} m, over 5 m '
                  f'{sum(1 for o in lab_offs if o > 5) / len(lab_offs):.2f}']
    lines += ['', '## Same-ramp scatter (mechanism)', '',
              f'Largest pairwise member distance over {len(spread["server"])} fusion '
              f'sites with >= 3 placeable members (a complete-linkage cut at t keeps the '
              f'group only if this is <= t):', '',
              '| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |',
              '|---|---:|---:|---:|---:|---:|']
    for key, v in spread.items():
        if not v:
            lines.append(f'| {key} | n/a | n/a | n/a | n/a | n/a |')
            continue
        lines.append(f'| {key} | {fmt(quantile(v, .5), 1)} m | {fmt(quantile(v, .9), 1)} m '
                     f'| {sum(1 for x in v if x > 7.5) / len(v):.2f} '
                     f'| {sum(1 for x in v if x > 10) / len(v):.2f} '
                     f'| {sum(1 for x in v if x > 15) / len(v):.2f} |')
    missing = {k: v['coherence']['missing'] for k, v in results.items()
               if v['coherence']['missing']}
    lines += ['', '- self-detections whose cluster could not be located (should be 0): '
              + (', '.join(f'{k} {v}' for k, v in missing.items()) if missing
                 else '0 in every arm')]
    ref_label = ref_name if not args.offline else f'{ref_name} (offline stand-in for deployed)'
    lines += ['', f'## Match-radius sweep ({ref_name} vs fusion)', '',
              f'| radius m | {ref_label} coverage | {ref_name} frag 5 m | {ref_name} dual '
              'both | fusion coverage | fusion frag 5 m | fusion dual both |'
              if args.offline else
              '| radius m | deployed coverage | deployed frag 5 m | deployed dual both | '
              'fusion coverage | fusion frag 5 m | fusion dual both |',
              '|---:|---|---|---|---|---|---|']
    for r_m, a, b in sweep:
        fa, fb = a['frag'][5.0], b['frag'][5.0]
        lines.append(f"| {r_m:g} | {fmt(a['coverage'])} | {fa['with_extra']}/{fa['ramps']} "
                     f"| {a['dual']['both']}/{a['dual']['pairs']} | {fmt(b['coverage'])} | "
                     f"{fb['with_extra']}/{fb['ramps']} | "
                     f"{b['dual']['both']}/{b['dual']['pairs']} |")

    lines += ['', '## Precision by cluster size', '',
              'Is a small cluster a false positive? Each AI label is bucketed by the size '
              '(labels) of the cluster holding it, or as `unplaceable` when the raycast '
              'cannot place it (beyond the range cap, or at/above the horizon); fusion '
              'cannot associate those, so they are the singletons of `fusion_server`, and '
              f'the same bucket is split out of `{ref_name}` for comparison. Precision is '
              'T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), '
              'with a Wilson 95% interval. Median y is the stored detection\'s '
              'y_normalized (0.5 = the horizon). `unclustered` (a placeable label no '
              'cluster of that partition holds) is shown only when non-empty.', '',
              '| partition | bucket | AI labels | median conf | median y | judged | '
              'precision [95% CI] | T | F | neither |',
              '|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for name in (ref_name, 'fusion_server', 'fusion_server+attach'):
        for row in results[name]['size']:
            if row['bucket'] == 'unclustered' and not row['n']:
                continue
            lines.append(f"| {name} | {row['bucket']} | {row['n']} | "
                         f"{fmt(row['median_conf'], 2)} | {fmt(row['median_y'], 3)} | "
                         f"{row['judged']} | "
                         f"{precision_ci_text(row['t'], row['f'])} | {row['t']} | "
                         f"{row['f']} | {row['judged'] - row['t'] - row['f']} |")

    a = attach_stats
    lines += ['', '## Unplaceable labels: attach by bearing (`fusion_server+attach`)', '',
              'One rule, fixed before any result and not tuned on GT (issue #106): a label '
              'the raycast cannot place joins the placed `fusion_server` site nearest along '
              f'its bearing ray, if one lies {ATTACH_MIN_RANGE_M:g}-{ATTACH_MAX_RANGE_M:g} m '
              f'ahead and within {ATTACH_PERP_M:g} m of the ray and holds no label from the '
              'same pano; it does not move the site. Otherwise it stays a singleton.', '',
              f"- unplaceable labels: {a['unplaceable']}; attached {a['attached']} "
              f"({a['attached'] / a['unplaceable']:.2f})" if a['unplaceable'] else
              '- unplaceable labels: 0',
              f"- clusters: {a['clusters_server']} (`fusion_server`) -> "
              f"{a['clusters_attach']} (`fusion_server+attach`)",
              f"- sanity (not a metric): {a['judged_unplaceable']} unplaceable labels are on "
              f"judged panos; {a['judged_attached']} of them attached "
              f"({a['judged_attached_true']} judged true), "
              f"{a['judged_attached_to_true_site']} to a site holding a verdict-true member "
              '(de-clustered benchmark panos rarely see each other, so most sites hold no '
              'judged member at all)']

    if ocheck is not None:
        o = ocheck
        lines += ['', '## Offline server arm vs this server (`--offline-check`)', '',
                  "Validates the offline mode used for cities without a server: the labels "
                  f"it synthesizes from `{results_path.name}` and places with the server's "
                  'estimator (ps_placement), against the labels this server actually holds.',
                  '',
                  f"- (a) placement, {o['n_placed']} of {o['n_ai']} AI labels re-placed from "
                  f"the file: {o['n_moved_labels']} labels on {o['n_moved_panos']} panos "
                  f'left out because the position their labels imply (inverting the '
                  f"server's estimator, median per pano) is > {MOVED_PANO_M:g} m from the "
                  f"file's; over the other {o['n_kept']}: median {fmt(o['median'], 6)} m, "
                  f"p90 {fmt(o['p90'], 6)} m, max {fmt(o['max'], 6)} m, "
                  f"{o['over_0_5']} over 0.5 m (all labels: median "
                  f"{fmt(o['median_all'], 6)} m, p90 {fmt(o['p90_all'], 6)} m, "
                  f"{o['over_0_5_all']} over 0.5 m, max {fmt(o['max_all'], 6)} m)",
                  f"- (a) placement check: **{'PASS' if o['placement_ok'] else 'FAIL'}** "
                  f"(max error over ALL labels <= {PLACEMENT_TOL_M:g} m). Gated on all "
                  'labels because the moved-pano exclusion above is self-referential: it '
                  'inverts the estimator being validated, so an error consistent within a '
                  'pano would be excluded, not failed',
                  f"- (b) labels: {o['n_synth']} synthesized at {args.min_confidence:g} "
                  f"(unmasked); {o['n_synth'] - o['synth_not_live']} match a live AI label "
                  f"by pano and pixel, {o['synth_not_live']} do not (soft-deleted, or never "
                  f"sent); {o['live_not_synth']} live AI labels have no synthesized twin; "
                  f"{o['unclustered']} matched labels are in no deployed cluster (the "
                  'server has not clustered them) and are left out of the partitions',
                  f"- (b) regions by nearest street to the offline position: "
                  f"{o['region_agree']} of {o['n_joined']} equal the label's live region_id "
                  f"({o['region_ties']} equidistant ties)",
                  f"- (b) partition: of the {o['eligible']} deployed clusters whose labels "
                  f"are all AI and all synthesized (of {o['n_deployed']}), {o['identical']} "
                  f"are, label for label, a cluster of the offline `ps @ 7.5 m` "
                  f"({o['identical'] / o['eligible']:.3f}; {o['off_labels']} labels in the "
                  'others)' if o['eligible'] else '- (b) partition: no eligible clusters']
        if 'identical_h' in o:
            lines.append(
                f"- (b+) the same with the {o['n_humans']} live human labels added at their "
                f"live positions and regions: {o['identical_h']} of the {o['eligible_h']} "
                'deployed clusters whose labels are all present are reproduced '
                f"({o['identical_h'] / o['eligible_h']:.3f}; {o['off_labels_h']} labels in "
                'the others)' if o['eligible_h'] else '- (b+) no eligible clusters')

    report = '\n'.join(lines) + '\n'
    # LF on every platform, so a regeneration is byte-comparable with the committed copy
    (out / 'report.md').write_text(report, encoding='utf-8', newline='\n')
    with open(out / 'arms.csv', 'w', newline='', encoding='utf-8') as f:
        rows = [csv_row(k, v) for k, v in results.items()]
        w = csv.DictWriter(f, list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(report)
    print(f'wrote {out / "report.md"} and arms.csv')
    return {'results': results, 'out': out, 'attach': attach_stats, 'offline_check': ocheck,
            'meta': {'city': args.city, 'split': split, 'offline': args.offline,
                     'results_sha256': results_sha, 'n_labels': len(labels),
                     'n_ai': len(ai), 'scorer': SCORER_VERSION}}


def synthesize_rig_count(results_path, min_confidence):
    """How many stored detections >= min_confidence the nadir mask drops (report line)."""
    n = 0
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec['pano'].get('lat') is None or rec['pano'].get('lng') is None \
                        or rec['pano'].get('camera_heading') is None:
                    continue
                n += sum(1 for d in rec.get('detections', [])
                         if d['confidence'] >= min_confidence
                         and on_camera_rig(d['y_normalized']))
    return n


def main(argv=None):
    out = run(build_parser().parse_args(argv))
    oc = out.get('offline_check') if isinstance(out, dict) else None
    if oc is not None and not oc['placement_ok']:
        print(f"offline check FAILED: a live AI label re-places {oc['max_all']:.3f} m from "
              f'the server (> {PLACEMENT_TOL_M:g} m)', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
