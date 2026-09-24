"""AI-vs-crowd agree rate: RampNet detections vs a PS city's crowd curb-ramp labels.

Goal 2 of issue #31. Gainesville is the first city where this comparison is possible by
construction: the run's boundary is the deployment's opened area, the crowd audited it
Apr-Aug 2026, and ~2/3 of the run's GSV imagery is from 2026 — so AI and crowd looked at
the same streets, largely in the same pictures. Nothing is submitted anywhere: the crowd
corpus is the baseline, and an AI label on the server would contaminate it (#21).

The crowd baseline keeps moving, so it is FROZEN: the three read-only GETs below are
pulled once into the output dir and every number is relative to that snapshot, which the
report heads with url, fetch time, sha256 and feature count.

  /v3/api/rawLabels?labelType=CurbRamp&filetype=geojson     -> raw_labels_CurbRamp.geojson
  /v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson   -> raw_labels_NoCurbRamp.geojson
  /v3/api/regions?filetype=geojson                          -> regions.geojson

Two matching frames, reported side by side, because they answer different questions:

- PANO frame (the headline). A crowd label carries its pano id and pixel. When that pano
  is one the run processed, crowd mark and AI detection are compared in the SAME image,
  in normalized pixel space, with RampNet's own matcher geometry (anisotropic 1024x512
  scaling, x wraps at the seam, radius 0.022 in x units — rampnet.detection_eval's
  PANO_RADIUS_NORMALIZED, mirrored here rather than imported). No camera height, no
  raycast, no RampNet#101 scale error and no PS placement error enter it. Its limit is
  coverage: it scores only crowd labels whose pano the run processed.
- WORLD frame (a bound). Every crowd label — including those on panos the run never saw —
  is placed where PS placed it (the feed's lat/lng, from the viewer's estimator,
  SidewalkWebpage#4766) and matched one-to-one against fused AI sites (fuse_sites.fuse,
  the labeler's own flat-ground raycast). The two placements carry different, independent
  error models, so disagreement here mixes geometry with detection: it is a bound on the
  agree rate, never the headline. The raw union of operational raycasts (any single view
  within radius) is reported beside it as the loosest bound.

Directions:
- crowd -> AI: share of crowd CurbRamp labels that an AI detection (pano) / site (world)
  agrees with, with the misses split by HUMAN validation. The feed's `correct` field is
  also reported, but separately and labelled: on this deployment ~99.7% of labels carry a
  vote from PS's own AI validator (`validator_type: AI`), so `correct` is mostly another
  model's opinion, not ground truth.
- AI -> crowd: share of operational AI sites in FULLY audited regions with a crowd
  CurbRamp label within radius (world), and the share sitting on a crowd NoCurbRamp label
  (the informative disagreement). A pano-frame version is reported as a lower bound only:
  an auditor labels a ramp once, from one of the several panos that see it.

Regions: the run's footprint is the set of PS regions the area polygon covers. Regions
whose `completion_rate` in the snapshot is below --full-completion are "partial": their
crowd labels still count for crowd -> AI, but an AI site there may simply sit on an
unaudited street, so partial regions are kept out of AI -> crowd and reported apart.

Every AI figure is given at OPERATIONAL_CONFIDENCE (what production ships) with the
BENCHMARK tier beside it, and every world figure at the production camera height with
geo.PER_PANO (#40) as an ablation. Keys: a PS label_id is per-city, so every per-label row
carries `label_uid` = "<city>:<label_id>".

Deliberately stdlib + shapely (both in requirements-test.txt) and NOT pandas/scipy, so the
offline tests run in CI; the fetch/provenance helpers are therefore small copies of
eval_ps_clustering's rather than an import of that pandas-bound module.

Usage:
    python scripts/agree_rate.py gainesville \
        --server https://sidewalk-gainesville.cs.washington.edu
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402
from detectors import (BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       on_camera_rig)

# rampnet.detection_eval: PANO_SCALE_X/Y and PANO_RADIUS_NORMALIZED. Mirrored, not
# imported (RampNet is a sibling repo, read as data only), so a change there has to be
# carried here by hand — the report prints the values it used.
PANO_SCALE_X = 1024
PANO_SCALE_Y = 512
PANO_RADIUS = 0.022
PANO_RADII = (0.011, 0.022, 0.033, 0.05)
WORLD_RADII = (2.5, 5.0, 7.5)
WORLD_RADIUS = 5.0
LABEL_TYPES = ('CurbRamp', 'NoCurbRamp')
API_LABELS = '/v3/api/rawLabels?labelType={}&filetype=geojson'
API_REGIONS = '/v3/api/regions?filetype=geojson'
# A region counts as part of the run's footprint when this share of its area lies
# inside the run polygon. The boundary was dissolved from the opened regions, so real
# members sit at ~1.0 and neighbours at ~0.0; the threshold only has to split them.
FOOTPRINT_OVERLAP = 0.5
VINTAGE_BUCKETS = ('same pano', 'newer than ours', 'same month', '1-18 mo older',
                   '19-36 mo older', '>36 mo older', 'no run pano nearby')
# A crowd label whose pano is neither processed nor in any processed pano's history is
# dated against the nearest run pano; farther than this, there is none to compare to.
NEAREST_PANO_M = 30.0


# ----------------------------------------------------------------------------- inputs

def fetch(url, dest, refresh=False):
    """GET url into dest once, and record where and when it came from.

    A copy of eval_ps_clustering.fetch (that module imports pandas at import time, which
    the CI test environment does not have). The pull IS the frozen snapshot, so a cache
    hit is reused and says so, a body that is not a non-empty FeatureCollection is refused
    rather than cached (a zero-feature 200 would read as "the crowd labeled nothing"),
    and the write is temp-then-rename.
    """
    if dest.exists() and not refresh:
        print(f'reusing cached {dest.name} (the frozen snapshot; --refresh re-pulls it)',
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
                    .isoformat(timespec='seconds')}, indent=1), encoding='utf-8')


def snapshot_row(path):
    """(name, url, fetched_at, sha256, n_features) — the tuple that identifies a pull."""
    side = path.parent / (path.name + '.source.json')
    meta = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}
    n = len(json.loads(path.read_text(encoding='utf-8'))['features'])
    return (path.name, meta.get('url', '(supplied locally; no fetch record)'),
            meta.get('fetched_at', 'unrecorded'),
            hashlib.sha256(path.read_bytes()).hexdigest(), n)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CrowdLabel:
    uid: str                  # "<city>:<label_id>" — label_id alone is per-city
    label_id: int
    label_type: str
    user_id: str
    pano_id: str
    region_id: int | None
    lat: float
    lng: float
    x: float | None           # pano_x / pano_width, None if the feed lacks dimensions
    y: float | None
    capture_date: str | None
    human: str                # 'agreed' | 'disagreed' | 'unvalidated' (see human_status)
    ps_correct: bool | None   # the feed's `correct` (includes PS's AI validator)
    has_ai_vote: bool
    e: float = 0.0
    n: float = 0.0
    id: int = 0               # position in its list; match_one_to_one needs an .id


def human_status(validations, label_user):
    """Majority of HUMAN validations by someone other than the labeler.

    'agreed' / 'disagreed' on a strict majority of Agree vs Disagree votes; everything
    else — no human vote, only Unsure votes, or a tie — is 'unvalidated', because none
    of those is a human verdict. PS's AI validator votes are excluded on purpose: they
    are another model's opinion, and letting them in would make "crowd agrees with AI"
    partly "AI agrees with AI".
    """
    agree = disagree = 0
    for v in validations or ():
        if v.get('validator_type') != 'Human' or v.get('user_id') == label_user:
            continue
        if v.get('validation') == 'Agree':
            agree += 1
        elif v.get('validation') == 'Disagree':
            disagree += 1
    if agree > disagree:
        return 'agreed'
    if disagree > agree:
        return 'disagreed'
    return 'unvalidated'


def load_crowd(path, city):
    """(labels, n_dropped) from a rawLabels geojson. Rows with a null or corrupt
    position (lng > 360 has been seen upstream) are dropped and counted, as
    eval_ps_clustering.load_labels does; a missing pano size only loses the pano frame."""
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    out, dropped = [], 0
    for ft in feats:
        q = ft['properties']
        lng, lat = ft['geometry']['coordinates']
        if lng is None or lat is None or not math.isfinite(lng) or abs(lng) > 360:
            dropped += 1
            continue
        w, h = q.get('pano_width'), q.get('pano_height')
        px, py = q.get('pano_x'), q.get('pano_y')
        ok = bool(w and h) and px is not None and py is not None
        out.append(CrowdLabel(
            uid=f"{city}:{q['label_id']}", label_id=int(q['label_id']),
            label_type=q['label_type'], user_id=str(q['user_id']), pano_id=q['pano_id'],
            region_id=q.get('region_id'), lat=float(lat), lng=float(lng),
            x=(px / w) if ok else None, y=(py / h) if ok else None,
            capture_date=q.get('image_capture_date'),
            human=human_status(q.get('validations'), str(q['user_id'])),
            ps_correct=q.get('correct'),
            has_ai_vote=any(v.get('validator_type') == 'AI'
                            for v in q.get('validations') or ()),
            id=len(out)))
    uids = [lab.uid for lab in out]
    if len(set(uids)) != len(uids):
        raise SystemExit(f'{path.name}: duplicate label_uid — the feed repeats a label')
    return out, dropped


@dataclass
class Region:
    region_id: int
    name: str
    completion: float
    overlap: float            # share of the region's area inside the run polygon
    geom: object = field(repr=False, default=None)


def load_regions(path, area_geom):
    """{region_id: Region} for every region in the feed, with its overlap with the run."""
    from shapely.geometry import shape
    with open(path, encoding='utf-8') as f:
        feats = json.load(f)['features']
    out = {}
    for ft in feats:
        q = ft['properties']
        g = shape(ft['geometry'])
        ov = g.intersection(area_geom).area / g.area if g.area > 0 else 0.0
        out[int(q['region_id'])] = Region(int(q['region_id']), str(q.get('name')),
                                          float(q.get('completion_rate') or 0.0), ov, g)
    return out


def region_class(region_id, regions, full_completion):
    """'full' | 'partial' | 'outside' for a region id, the bucketing every table uses.

    Outside = not in the run's footprint (a region the deployment opened after the run,
    or one the run polygon does not cover), or an id the region feed does not know.
    """
    r = regions.get(region_id)
    if r is None or r.overlap < FOOTPRINT_OVERLAP:
        return 'outside'
    return 'full' if r.completion >= full_completion else 'partial'


class RegionLocator:
    """Point-in-region for AI sites, which carry no region id. Footprint regions only,
    so a site is attributed to the same region set the crowd labels are bucketed by."""

    def __init__(self, regions):
        from shapely.prepared import prep
        self.items = [(rid, prep(r.geom), r.geom.bounds) for rid, r in regions.items()
                      if r.overlap >= FOOTPRINT_OVERLAP]

    def locate(self, lat, lng):
        from shapely.geometry import Point
        pt = None
        for rid, pg, (x0, y0, x1, y1) in self.items:
            if x0 <= lng <= x1 and y0 <= lat <= y1:
                pt = pt or Point(lng, lat)
                if pg.contains(pt):
                    return rid
        return None


def scan_run(results_path):
    """One light pass over results.jsonl for what fuse_sites.load_results discards:
    {pano_id: (capture_date, width, height, record_ordinal)}, the history map
    {older_pano_id: [(run_pano_id, run_capture_date)]}, the stored detection pixels
    (the contamination check), and the number of records.

    record_ordinal counts RECORDS (non-blank lines), not raw line numbers, and a pano id
    that repeats keeps its FIRST ordinal — so it is directly comparable with the record
    counts manifest.json keeps per phase (see gap_fill_ids)."""
    run, history, pixels = {}, {}, set()
    n_records = 0
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec['pano']
            pid = p['panorama_id']
            if pid not in run:
                run[pid] = (p.get('capture_date'), p.get('width'), p.get('height'),
                            n_records)
            n_records += 1
            for h in p.get('history') or ():
                history.setdefault(h['pano_id'], []).append((pid, p.get('capture_date')))
            w, h_ = p.get('width'), p.get('height')
            if w and h_:
                for d in rec.get('detections', ()):
                    pixels.add((pid, round(d['x_normalized'] * w),
                                round(d['y_normalized'] * h_)))
    return run, history, pixels, n_records


def gap_fill_ids(run_meta, n_records, n_gap):
    """Pano ids whose record came from the gap-fill phase.

    main.py appends gap-fill records to the end of results.jsonl, and the manifest counts
    them per phase (`processed` of the `phase: gap_fill` entries), so they are the last
    n_gap RECORDS. The cut is taken in record ordinals on both sides — mixing a raw line
    index with a unique-pano count (the earlier version) drifts by one per blank or
    repeated line.
    """
    first = n_records - n_gap
    return {pid for pid, meta in run_meta.items() if meta[3] >= first}


# ------------------------------------------------------------------------ pano frame

def pano_distance(ax, ay, bx, by):
    """Distance between two normalized pano points in normalized-x units, with
    RampNet's geometry: x scaled by 1024 and y by 512 (equal angle per unit on a 2:1
    equirect), x cyclic at the seam."""
    dx = abs(ax - bx) % 1.0
    dx = min(dx, 1.0 - dx) * PANO_SCALE_X
    dy = (ay - by) * PANO_SCALE_Y
    return math.hypot(dx, dy) / PANO_SCALE_X


def match_pano(a_pts, b_pts, radius):
    """Greedy ascending-distance one-to-one match of two point lists on ONE pano.

    Returns {a_index: (b_index, distance)}. Strictly-within-radius, as RampNet's
    greedy_match (`dist_sq < radius_sq`); ties broken on (distance, a, b) so the result
    never depends on input order beyond the indices themselves.
    """
    pairs = []
    for i, (ax, ay) in enumerate(a_pts):
        for j, (bx, by) in enumerate(b_pts):
            d = pano_distance(ax, ay, bx, by)
            if d < radius:
                pairs.append((d, i, j))
    pairs.sort()
    out, used = {}, set()
    for d, i, j in pairs:
        if i in out or j in used:
            continue
        out[i] = (j, d)
        used.add(j)
    return out


def ai_points(slim_pano, tier):
    """Operational detections of one pano at `tier`, rig-masked as production ships."""
    return [(x, y) for _i, x, y, c in slim_pano.detections
            if c >= tier and not on_camera_rig(y)]


def pano_frame(crowd, run_by_id, tier, radius):
    """Per-pano matching of crowd labels to AI detections on the same image.

    Returns (crowd_hit {label.id: distance}, ai_total, ai_hit) where ai_* count the
    operational detections on crowd-labelled panos (the pano-frame AI -> crowd lower
    bound). All crowd labels on a pano are matched together, one-to-one, so two crowd
    marks can never both claim one detection.
    """
    by_pano = {}
    for lab in crowd:
        if lab.pano_id in run_by_id and lab.x is not None:
            by_pano.setdefault(lab.pano_id, []).append(lab)
    hit, ai_total, ai_hit = {}, 0, 0
    for pid in sorted(by_pano):
        labs = by_pano[pid]
        ai = ai_points(run_by_id[pid], tier)
        m = match_pano([(lab.x, lab.y) for lab in labs], ai, radius)
        for i, (_j, d) in m.items():
            hit[labs[i].id] = d
        ai_total += len(ai)
        ai_hit += len(m)
    return hit, ai_total, ai_hit


def pano_frame_any(crowd, run_by_id, tier, radius):
    """{label.id: nearest distance} for crowd labels with ANY operational detection on
    the same pano strictly within radius — coverage, not one-to-one.

    The headline matcher (pano_frame) is one-to-one, as RampNet's is, so when two crowd
    marks sit under one AI peak only one of them can agree. This is the other reading:
    the gap between the two is the count of "shadowed" labels, a detection nearby that
    another crowd mark on the same pano claimed.
    """
    out = {}
    for lab in crowd:
        p = run_by_id.get(lab.pano_id)
        if p is None or lab.x is None:
            continue
        ds = [pano_distance(x, y, lab.x, lab.y) for x, y in ai_points(p, tier)]
        ds = [d for d in ds if d < radius]
        if ds:
            out[lab.id] = min(ds)
    return out


# ----------------------------------------------------------------------- world frame

@dataclass
class Pt:
    id: int
    e: float
    n: float


def match_one_to_one(ramps, sites, radius_m):
    """eval_sites.match_one_to_one, grid-accelerated. Same pairs (d <= radius), same
    sort key (d, ramp index, site id), same greedy pass — tests assert the outputs are
    identical. The original is O(n*m), minutes on a city's thousands of labels x sites
    at every tier/height/radius; the grid makes it near-linear."""
    grid = geo.GridIndex(radius_m)
    for s in sites:
        grid.add(s.e, s.n, s)
    pairs = []
    for gi, ramp in enumerate(ramps):
        for site in grid.near(ramp.e, ramp.n):
            d = math.hypot(ramp.e - site.e, ramp.n - site.n)
            if d <= radius_m:
                pairs.append((d, gi, site.id, site))
    pairs.sort(key=lambda p: p[:3])
    matched, used = {}, set()
    for d, gi, sid, site in pairs:
        if gi in matched or sid in used:
            continue
        matched[gi] = site
        used.add(sid)
    return matched


def any_within(points, others, radius_m):
    """{point index: nearest distance} for points with ANY of `others` within radius —
    coverage, not one-to-one (the raw-union bound, and the NoCurbRamp overlap)."""
    grid = geo.GridIndex(radius_m)
    for o in others:
        grid.add(o.e, o.n, o)
    out = {}
    for i, p in enumerate(points):
        best = None
        for o in grid.near(p.e, p.n):
            d = math.hypot(p.e - o.e, p.n - o.n)
            if d <= radius_m and (best is None or d < best):
                best = d
        if best is not None:
            out[i] = best
    return out


def world_config(run_panos, tier, height):
    """(operational sites as Pt, operational projected detections as Pt, frame, stats,
    the raw fuse_sites.Site list) for one tier x camera height, through fuse_sites' own
    code path — so every world number is in exactly the frame production fuses in."""
    params = fs.FuseParams(camera_height_m=height, min_confidence=tier)
    sites, frame, stats = fs.fuse(run_panos, params)
    dets, _frame, _drops = fs.project(run_panos, params)
    op_sites = [Pt(s.id, s.e, s.n) for s in sites if s.n_operational > 0]
    op_dets = [Pt(k, d.e, d.n) for k, d in enumerate(dets) if d.operational]
    return op_sites, op_dets, frame, stats, sites


# --------------------------------------------------------------------------- vintage

def vintage_stratum(lab, run_meta, history, nearest_capture):
    """(stratum, months) placing a crowd label's imagery against the run's.

    Same pano -> 'same pano'. Otherwise the reference run pano is the one whose GSV
    history lists the crowd pano (same spot, exact), else the nearest run pano within
    NEAREST_PANO_M of the label; months = ours - theirs, so positive = crowd is older.
    """
    if lab.pano_id in run_meta:
        return 'same pano', 0
    theirs = fs._months(lab.capture_date)
    owners = history.get(lab.pano_id)
    # Several run panos can list the same old pano; the newest of them is the run's
    # current imagery at that spot, which is what the crowd pano is being compared to.
    ours = max((m for m in (fs._months(d) for _pid, d in owners) if m is not None),
               default=None) if owners else nearest_capture
    if ours is None or theirs is None:
        return 'no run pano nearby', None
    delta = ours - theirs
    if delta < 0:
        return 'newer than ours', delta
    if delta == 0:
        return 'same month', 0
    if delta <= 18:
        return '1-18 mo older', delta
    if delta <= 36:
        return '19-36 mo older', delta
    return '>36 mo older', delta


# ------------------------------------------------------------------ interpretation

CHANCE_SHIFT_M = 25.0
CHANCE_SEED = 31


def chance_floor(crowd_pts, sites, radius_m, shift_m=CHANCE_SHIFT_M, seed=CHANCE_SEED):
    """One-to-one agree count after moving every crowd label shift_m in a random
    (seeded, so reproducible) direction.

    The world frame matches against thousands of AI sites, so some agreement is
    coincidence: a point dropped anywhere on a street often has a site within 5 m.
    Displacing the labels keeps their density and street-bound layout but breaks any
    real correspondence, so what still matches estimates that coincidence. It is a
    rough floor, not an exact null: a displaced label can land on a different real
    ramp (corners repeat along a street).
    """
    import random
    rng = random.Random(seed)
    moved = []
    for p in crowd_pts:
        a = rng.uniform(0.0, 2.0 * math.pi)
        moved.append(Pt(p.id, p.e + shift_m * math.cos(a), p.n + shift_m * math.sin(a)))
    return len(match_one_to_one(moved, sites, radius_m))


def pixel_alignment(crowd, run_by_id, tier, radius=PANO_RADIUS):
    """(n, median dx deg, median dy deg, p10-p90 dx, p10-p90 dy) over crowd labels whose
    nearest operational detection on the same pano is within radius.

    The pano frame silently assumes PS's pano_x/pano_y and the labeler's normalized
    x/y are the same image coordinates (send_to_ps.py's contract). A mirrored axis, a
    heading offset or a y measured from the horizon would not raise; it would just
    depress the agree rate. So the report prints the signed offsets: aligned frames
    centre on zero with a spread of a few degrees.
    """
    dxs, dys = [], []
    for lab in crowd:
        p = run_by_id.get(lab.pano_id)
        if p is None or lab.x is None:
            continue
        best = None
        for x, y in ai_points(p, tier):
            d = pano_distance(x, y, lab.x, lab.y)
            if d < radius and (best is None or d < best[0]):
                best = (d, ((x - lab.x + 0.5) % 1.0 - 0.5) * 360.0, (y - lab.y) * 180.0)
        if best:
            dxs.append(best[1])
            dys.append(best[2])
    if not dxs:
        return 0, None, None, None, None
    dxs.sort()
    dys.sort()

    def q(xs, f):
        return xs[min(len(xs) - 1, int(f * len(xs)))]
    return (len(dxs), q(dxs, .5), q(dys, .5), (q(dxs, .1), q(dxs, .9)),
            (q(dys, .1), q(dys, .9)))


def site_views(site, tier):
    """(distinct panos with an operational member, max member confidence)."""
    ops = [d for d, _ in site.members if d.conf >= tier]
    return len({d.pano_id for d in ops}), max((d.conf for d in ops), default=0.0)


def unmatched_breakdown(w, crowd_pts, tier, radius):
    """Why an operational site in a fully audited region has no crowd label.

    Buckets each full-region site by the distance to its nearest crowd CurbRamp label
    (one-to-one matched / another site took the label within radius / nearest 5-10 m /
    10-20 m / > 20 m) and, within each, by how many panos saw it and whether it clears
    the benchmark tier. A site with no crowd label within 20 m, seen from several panos
    at high confidence, is a ramp the crowd did not label, not a placement quibble.
    """
    raw = {s.id: s for s in w['raw_sites']}
    order = (f'matched one-to-one (<= {radius:g} m)',
             f'label within {radius:g} m taken by another site',
             f'nearest label {radius:g}-10 m', 'nearest label 10-20 m',
             'no label within 20 m')
    far = any_within(w['full_sites'], crowd_pts, 20.0)
    near10 = any_within(w['full_sites'], crowd_pts, 10.0)
    near_r = any_within(w['full_sites'], crowd_pts, radius)
    out = {}
    for i, s in enumerate(w['full_sites']):
        if s.id in w['site_one'][radius]:
            b = f'matched one-to-one (<= {radius:g} m)'
        elif i in near_r:
            b = f'label within {radius:g} m taken by another site'
        elif i in near10:
            b = f'nearest label {radius:g}-10 m'
        elif i in far:
            b = 'nearest label 10-20 m'
        else:
            b = 'no label within 20 m'
        n_views, conf = site_views(raw[s.id], tier)
        row = out.setdefault(b, {'sites': 0, 'multi_view': 0, 'benchmark_tier': 0})
        row['sites'] += 1
        row['multi_view'] += n_views >= 2
        row['benchmark_tier'] += conf >= BENCHMARK_CONFIDENCE
    return {k: out[k] for k in order if k in out}


def skipped_ramp_estimate(n_crowd_labels, crowd_recall_k, n_pool, n_unmatched):
    """How many of the unmatched AI sites the crowd's recall can account for.

    If the crowd's CurbRamp labels in the fully audited regions cover a share R of the
    real ramps there (R = crowd recall against the GT pool), and each label is one ramp,
    the regions hold about n_crowd_labels / R ramps, so about n_crowd_labels * (1/R - 1)
    were skipped. Each skipped ramp can explain at most one unmatched AI site, so
    skipped / n_unmatched is an UPPER bound on the share of unmatched sites that are
    crowd omissions (a skipped ramp the AI also missed explains none).

    Returns {'recall', 'skipped', 'share', 'skipped_lo', 'skipped_hi', 'share_lo',
    'share_hi'}; lo/hi come from the recall's Wilson 95% interval (a HIGH recall means
    FEW skipped ramps, so the bounds swap). None when recall or a denominator is 0.
    """
    if not n_pool or not crowd_recall_k or not n_unmatched:
        return None
    lo, hi = es.wilson(crowd_recall_k, n_pool)
    r = crowd_recall_k / n_pool

    def skipped(rec):
        return n_crowd_labels * (1.0 / rec - 1.0)
    out = {'recall': r, 'skipped': skipped(r), 'skipped_lo': skipped(hi),
           'skipped_hi': skipped(lo)}
    for k in ('', '_lo', '_hi'):
        out['share' + k] = out['skipped' + k] / n_unmatched
    return out


def gt_adjudication(city, benchmark_root, run_panos, w, crowd, regions, full_completion,
                    radius=WORLD_RADIUS):
    """Who is right when AI and crowd disagree, per RampNet's judged benchmark panos.

    World frame, benchmark tier (the tier every bundle was judged at), production camera
    height; GT built exactly as eval_sites builds it. Two readings:
      - GT recall pool ramps in fully audited regions: covered by the crowd (any crowd
        CurbRamp label within radius), by AI (any operational site within radius), both,
        or neither. The crowd's share is its recall against the same GT the AI is scored
        on, which is what an AI -> crowd rate has to be read against.
      - AI sites with a judged operational member: precision (eval_sites' rule: TP if any
        judged member is true/duplicate, FP if every decided one is false) split by
        whether a crowd label is within radius. The split says whether the AI sites the
        crowd did not label are real.
    Returns None when the benchmark split is absent. The GT is RampNet-anchored (built
    during a RampNet review), which favours the AI; the report says so.
    """
    vpath = benchmark_root / city / 'verdicts.json'
    if not vpath.exists():
        return None
    # eval_sites.load_city_files, minus its re-read of the whole results.jsonl.
    verdict_panos = json.loads(vpath.read_text(encoding='utf-8'))['panos']
    bundle_ops = {}
    with open(benchmark_root / city / 'records.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bundle_ops[rec['pano']['panorama_id']] = [
                    (d['x_normalized'], d['y_normalized'], d['confidence'])
                    for d in rec['detections']]
    frame = w['frame']
    params = fs.FuseParams(camera_height_m=geo.DEFAULT_CAMERA_HEIGHT_M,
                           min_confidence=BENCHMARK_CONFIDENCE)
    points, op_verdicts, counts, warnings = es.build_gt(
        verdict_panos, bundle_ops, {p.pano_id: p for p in run_panos}, params, frame)
    ramps = es.merge_gt_points(points, 2.5)
    loc = RegionLocator(regions)
    pool = [Pt(k, r.e, r.n) for k, r in enumerate(ramps) if r.in_pool
            and region_class(loc.locate(*frame.to_latlng(r.e, r.n)), regions,
                             full_completion) == 'full']
    crowd_pts = [Pt(lab.id, *frame.to_enu(lab.lat, lab.lng)) for lab in crowd]
    by_crowd = any_within(pool, crowd_pts, radius)
    by_ai = any_within(pool, w['sites'], radius)
    cover = {'both': 0, 'crowd only': 0, 'AI only': 0, 'neither': 0}
    for i in range(len(pool)):
        c, a = i in by_crowd, i in by_ai
        cover['both' if c and a else 'crowd only' if c else 'AI only' if a else 'neither'] += 1

    # Full-region sites only: elsewhere "no crowd label" can just mean "not audited".
    site_pts = {s.id: s for s in w['full_sites']}
    crowd_near = any_within(w['full_sites'], crowd_pts, radius)
    near_ids = {w['full_sites'][i].id for i in crowd_near}
    prec = {True: [0, 0], False: [0, 0]}      # crowd near? -> [tp, fp]
    for s in w['raw_sites']:
        if s.id not in site_pts:
            continue
        vs = [op_verdicts[(d.pano_id, d.det_index)] for d, _ in s.members
              if d.operational and (d.pano_id, d.det_index) in op_verdicts]
        if not vs:
            continue
        key = s.id in near_ids
        if any(v is True or v == 'duplicate' for v in vs):
            prec[key][0] += 1
        elif any(v is False for v in vs):
            prec[key][1] += 1
    return {'counts': counts, 'n_warnings': len(warnings), 'pool': len(pool),
            'cover': cover, 'crowd_recall': len(by_crowd), 'ai_recall': len(by_ai),
            'prec': prec}


# ---------------------------------------------------------------------------- report

def rate(k, n):
    """'0.812 (k/n) [lo, hi]' — every rate prints its numerator, denominator and a
    Wilson 95% interval, so no number in the report travels without its base."""
    if n == 0:
        return 'n/a (0/0)'
    lo, hi = es.wilson(k, n)
    return f'{k / n:.3f} ({k}/{n}) [{lo:.3f}, {hi:.3f}]'


def tier_name(t):
    return f'{"operational" if t == OPERATIONAL_CONFIDENCE else "benchmark"} {t:g}'


def height_name(h):
    return 'per-pano' if h == geo.PER_PANO else f'{h:g} m'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city')
    ap.add_argument('--server', default=None,
                    help='PS server URL; pulls the three geojson if they are absent')
    ap.add_argument('--refresh', action='store_true',
                    help='re-pull over the cached snapshot (a NEW snapshot: every number '
                         'moves with it)')
    ap.add_argument('--run-dir', type=Path, default=None,
                    help='read-only run inputs (default runs/<city>)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output dir (default runs/<city>/agree_rate in THIS checkout)')
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark',
                    help='RampNet benchmark splits (read as data) for the GT adjudication '
                         'section; skipped when the city has none')
    ap.add_argument('--full-completion', type=float, default=0.999,
                    help='region completion_rate at or above which a region counts as '
                         'fully audited')
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    out = args.out or REPO_ROOT / 'runs' / args.city / 'agree_rate'
    out.mkdir(parents=True, exist_ok=True)
    paths = {t: out / f'raw_labels_{t}.geojson' for t in LABEL_TYPES}
    regions_path = out / 'regions.geojson'
    if args.server:
        base = args.server.rstrip('/')
        for t in LABEL_TYPES:
            fetch(base + API_LABELS.format(t), paths[t], args.refresh)
        fetch(base + API_REGIONS, regions_path, args.refresh)
    elif args.refresh:
        raise SystemExit('--refresh needs --server')
    for p in (*paths.values(), regions_path):
        if not p.exists():
            raise SystemExit(f'{p} missing: pass --server to pull the snapshot')

    from shapely.geometry import shape
    area = shape(json.loads((run_dir / 'area.geojson').read_text(encoding='utf-8')))
    regions = load_regions(regions_path, area)
    curb, curb_dropped = load_crowd(paths['CurbRamp'], args.city)
    nocurb, nocurb_dropped = load_crowd(paths['NoCurbRamp'], args.city)
    for lab in curb + nocurb:
        lab.region_class = region_class(lab.region_id, regions, args.full_completion)
    curb_all = curb
    curb = [lab for lab in curb_all if lab.region_class != 'outside']
    for k, lab in enumerate(curb):
        lab.id = k
    nocurb = [lab for lab in nocurb if lab.region_class != 'outside']
    for k, lab in enumerate(nocurb):
        lab.id = k

    results_path = run_dir / 'results.jsonl'
    run_meta, history, stored_pixels, n_records = scan_run(results_path)
    depth_index = results_path.parent / 'depth' / 'index.csv'
    if not depth_index.exists():
        print(f'WARNING: {depth_index} is missing (a local artifact, not in git). The '
              'per-pano ablation then has only the heights stored in #40-era pano blocks '
              'and falls back to the default for every other pano; the report prints how '
              'many panos had a measured height.', file=sys.stderr)
    run_panos, n_unplaceable = fs.load_results(results_path, read_heights=True)
    run_by_id = {p.pano_id: p for p in run_panos}
    n_measured = sum(1 for p in run_panos if p.camera_height_m is not None)
    print(f'per-pano heights: {n_measured} of {len(run_panos)} panos have a measured '
          f'height (depth index {"present" if depth_index.exists() else "MISSING"})',
          file=sys.stderr)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8')) \
        if (run_dir / 'manifest.json').exists() else {}
    n_gap = sum(r.get('processed', 0) for r in manifest.get('runs', ())
                if r.get('phase') == 'gap_fill')
    gap_ids = gap_fill_ids(run_meta, n_records, n_gap)

    # Contamination guard: a crowd label at exactly a stored detection's pixel is what an
    # AI submission would look like (eval_ps_clustering's label->detection key).
    contaminated = [lab for lab in curb_all if lab.pano_id in run_meta and lab.x is not None
                    and (lab.pano_id,
                         round(lab.x * run_meta[lab.pano_id][1]),
                         round(lab.y * run_meta[lab.pano_id][2])) in stored_pixels]

    tiers = (OPERATIONAL_CONFIDENCE, BENCHMARK_CONFIDENCE)
    heights = (geo.DEFAULT_CAMERA_HEIGHT_M, geo.PER_PANO)
    full_ids = sorted(rid for rid, r in regions.items()
                      if region_class(rid, regions, args.full_completion) == 'full')
    partial_ids = sorted(rid for rid, r in regions.items()
                         if region_class(rid, regions, args.full_completion) == 'partial')

    # ---- pano frame
    on_run = [lab for lab in curb if lab.pano_id in run_by_id and lab.x is not None]
    pano = {}   # (tier, radius) -> (hit dict, ai_total, ai_hit)
    for t in tiers:
        for r in sorted(set(PANO_RADII) | {PANO_RADIUS}):
            pano[(t, r)] = pano_frame(curb, run_by_id, t, r)
    pano_any = {(t, r): pano_frame_any(curb, run_by_id, t, r)
                for t in tiers for r in sorted(set(PANO_RADII) | {PANO_RADIUS})}

    # ---- world frame
    world = {}  # (tier, height) -> dict
    for t in tiers:
        for h in heights:
            print(f'fusing at {tier_name(t)}, height {height_name(h)} ...', file=sys.stderr)
            op_sites, op_dets, frame, stats, raw_sites = world_config(run_panos, t, h)
            for lab in curb + nocurb:
                lab.e, lab.n = frame.to_enu(lab.lat, lab.lng)
            curb_pts = [Pt(lab.id, lab.e, lab.n) for lab in curb]
            nocurb_pts = [Pt(lab.id, lab.e, lab.n) for lab in nocurb]
            loc = RegionLocator(regions)
            site_region = {}
            for s in op_sites:
                site_region[s.id] = loc.locate(*frame.to_latlng(s.e, s.n))
            full_sites = [s for s in op_sites
                          if region_class(site_region[s.id], regions,
                                          args.full_completion) == 'full']
            w = {'stats': stats, 'n_sites': len(op_sites), 'site_region': site_region,
                 'sites': op_sites, 'raw_sites': raw_sites, 'frame': frame,
                 'full_sites': full_sites, 'crowd_site': {},
                 'crowd_union': {}, 'site_one': {}, 'site_any': {}, 'site_ncr': {},
                 'site_ncr_only': {}}
            for r in WORLD_RADII:
                m = match_one_to_one(curb_pts, op_sites, r)
                w['crowd_site'][r] = {curb[i].id: site.id for i, site in m.items()}
                w['crowd_union'][r] = any_within(curb_pts, op_dets, r)
                ms = match_one_to_one(full_sites, curb_pts, r)
                w['site_one'][r] = {full_sites[i].id for i in ms}
                near_curb = any_within(full_sites, curb_pts, r)
                near_ncr = any_within(full_sites, nocurb_pts, r)
                w['site_any'][r] = {full_sites[i].id for i in near_curb}
                w['site_ncr'][r] = {full_sites[i].id for i in near_ncr}
                w['site_ncr_only'][r] = {full_sites[i].id for i in near_ncr
                                         if i not in near_curb}
            world[(t, h)] = w

    # ---- vintage (reference capture for labels on panos the run did not process)
    ref_frame = geo.LocalFrame(sum(p.lat for p in run_panos) / len(run_panos),
                               sum(p.lng for p in run_panos) / len(run_panos))
    pano_grid = geo.GridIndex(NEAREST_PANO_M)
    for p in run_panos:
        e, n = ref_frame.to_enu(p.lat, p.lng)
        pano_grid.add(e, n, (e, n, p))
    strata = {}
    for lab in curb:
        e, n = ref_frame.to_enu(lab.lat, lab.lng)
        best = None
        for pe, pn, p in pano_grid.near(e, n):
            d = math.hypot(pe - e, pn - n)
            if d <= NEAREST_PANO_M and (best is None or d < best[0]):
                best = (d, p)
        near = fs._months(best[1].capture_date) if best else None
        strata[lab.id] = vintage_stratum(lab, run_meta, history, near)

    # ================================================================== report
    H = geo.DEFAULT_CAMERA_HEIGHT_M
    OP, BM = OPERATIONAL_CONFIDENCE, BENCHMARK_CONFIDENCE
    snap = [snapshot_row(p) for p in (paths['CurbRamp'], paths['NoCurbRamp'], regions_path)]
    L = [f'# {args.city}: AI vs crowd curb-ramp agree rate (issue #31, goal 2)', '',
         '## Snapshot (every number below is relative to this pull)', '',
         '| file | url | fetched (UTC) | sha256 | features |', '|---|---|---|---|---:|']
    L += [f'| `{a}` | {b} | {c} | `{d}` | {e} |' for a, b, c, d, e in snap]
    L += ['', f'- run: `results.jsonl` sha256 `{sha256_file(results_path)}`, '
          f'{n_records} records / {len(run_meta)} distinct processed panos '
          f'({n_records - n_gap} main pass + {n_gap} gap-fill records, per manifest.json); '
          f'{n_unplaceable} without position/heading',
          f'- per-pano camera heights (the `{geo.PER_PANO}` ablation): **{n_measured} of '
          f'{len(run_panos)}** panos have a measured height; the rest use the '
          f'{geo.DEFAULT_CAMERA_HEIGHT_M:g} m default. Blocks from before #40 read it from '
          '`depth/index.csv` beside results.jsonl, a local artifact that is not in git'
          + ('' if depth_index.exists() else ' — **MISSING for this run, so the per-pano '
             'rows below are mostly the default height**'),
          f'- matcher: pano frame = RampNet geometry (x*{PANO_SCALE_X}, y*{PANO_SCALE_Y}, '
          f'x wraps), one-to-one, strictly within {PANO_RADIUS} x-units '
          f'(= {PANO_RADIUS * 360:.1f} deg); world frame = eval_sites.match_one_to_one '
          f'semantics, {WORLD_RADIUS:g} m headline',
          f'- AI tiers: operational {OP:g} (what production ships, rig-masked) and '
          f'benchmark {BM:g}; world camera height {H:g} m (production default) with '
          f'`{geo.PER_PANO}` (#40) as the ablation']

    n_out = len(curb_all) - len(curb)
    by_cls = {c: sum(1 for lab in curb if lab.region_class == c) for c in ('full', 'partial')}
    L += ['', '## Scope', '',
          f'- footprint: {len(full_ids) + len(partial_ids)} PS regions cover the run polygon '
          f'(overlap >= {FOOTPRINT_OVERLAP}); **{len(full_ids)} fully audited** '
          f'(completion_rate >= {args.full_completion}) and **{len(partial_ids)} partial**: '
          + ', '.join(f'{rid} ({regions[rid].completion:.3f})' for rid in partial_ids),
          f'- crowd CurbRamp labels: {len(curb_all)} in the pull ({curb_dropped} dropped for '
          f'a bad position), {n_out} outside the footprint (regions opened after the run), '
          f'**{len(curb)} in scope**: {by_cls["full"]} in full regions, {by_cls["partial"]} '
          f'in partial ones',
          f'- crowd NoCurbRamp labels in scope: {len(nocurb)} ({nocurb_dropped} dropped)',
          f'- **pano-frame coverage: {rate(len(on_run), len(curb))}** of in-scope crowd '
          f'CurbRamp labels sit on a pano the run processed '
          f'({sum(1 for lab in on_run if lab.pano_id in gap_ids)} of them on '
          f'{len({lab.pano_id for lab in on_run if lab.pano_id in gap_ids})} distinct '
          'gap-fill panos, i.e. panos the tile scan never returned); '
          f'{sum(1 for lab in curb if lab.x is None)} lack pano dimensions',
          f'- labellers: {len({lab.user_id for lab in curb})} accounts; the largest holds '
          f'{max(sum(1 for lab in curb if lab.user_id == u) for u in {lab.user_id for lab in curb})} '
          'labels',
          f'- validation: {sum(1 for lab in curb if lab.human != "unvalidated")} in-scope '
          f'labels carry a human majority verdict; {sum(lab.has_ai_vote for lab in curb)} '
          'carry a vote from PS\'s own AI validator, which is why `correct` is reported '
          'apart and never used as ground truth',
          f'- contamination check: {len(contaminated)} crowd labels sit exactly on a stored '
          'detection pixel (an AI submission would; must be 0)']
    n_al, mdx, mdy, (dx10, dx90), (dy10, dy90) = pixel_alignment(on_run, run_by_id, OP)         if on_run else (0, None, None, (None, None), (None, None))
    if n_al:
        L.append(f'- pixel-frame alignment (PS pano_x/pano_y vs stored x/y): over {n_al} '
                 f'crowd labels with a detection within {PANO_RADIUS}, signed offset median '
                 f'dx {mdx:+.2f} deg (p10-p90 {dx10:+.1f} to {dx90:+.1f}), dy {mdy:+.2f} deg '
                 f'({dy10:+.1f} to {dy90:+.1f}) — centred on zero, so the two coordinate '
                 'systems agree')

    # --- pano frame headline
    def pano_rate(labs, t, r):
        hit = pano[(t, r)][0]
        return rate(sum(1 for lab in labs if lab.id in hit), len(labs))

    groups = [('full regions', [lab for lab in on_run if lab.region_class == 'full']),
              ('partial regions', [lab for lab in on_run if lab.region_class == 'partial']),
              ('all in scope', on_run)]
    L += ['', '## Pano frame, crowd -> AI (the headline)', '',
          'Statistic: share of crowd CurbRamp labels on a run-processed pano with an AI '
          'detection on the SAME pano within the radius, one-to-one. Denominator: crowd '
          'labels on processed panos. Frame-free: no raycast, no camera height, no PS '
          'placement.', '',
          f'| regions | {tier_name(OP)} | {tier_name(BM)} |', '|---|---|---|']
    L += [f'| {g} | {pano_rate(labs, OP, PANO_RADIUS)} | {pano_rate(labs, BM, PANO_RADIUS)} |'
          for g, labs in groups]

    def any_rate(labs, t, r):
        hit = pano_any[(t, r)]
        return rate(sum(1 for lab in labs if lab.id in hit), len(labs))

    def shadowed(labs, t, r):
        """Labels with a detection within r that another crowd mark claimed."""
        one, anyd = pano[(t, r)][0], pano_any[(t, r)]
        return sum(1 for lab in labs if lab.id in anyd and lab.id not in one)

    L += ['', 'The one-to-one matcher is a choice, and it moves the number. When two crowd '
          'marks sit under one AI peak, only one of them can agree. **any detection** '
          'drops the one-to-one constraint: a crowd label agrees when ANY operational '
          'detection on its pano is within the radius. **shadowed** counts the labels '
          'that have a detection within the radius but lose it to another crowd mark on '
          'the same pano. Both are all in scope, radius '
          f'{PANO_RADIUS}.', '',
          '| tier | one-to-one | any detection | shadowed |', '|---|---|---|---:|']
    L += [f'| {tier_name(t)} | {pano_rate(on_run, t, PANO_RADIUS)} | '
          f'{any_rate(on_run, t, PANO_RADIUS)} | {shadowed(on_run, t, PANO_RADIUS)} |'
          for t in tiers]
    L += ['', f'Radius sweep (all in scope, {tier_name(OP)} / {tier_name(BM)}):', '',
          '| radius (x-units) | degrees | operational | operational, any detection | '
          'benchmark |', '|---:|---:|---|---|---|']
    L += [f'| {r:g} | {r * 360:.1f} | {pano_rate(on_run, OP, r)} | '
          f'{any_rate(on_run, OP, r)} | {pano_rate(on_run, BM, r)} |'
          for r in PANO_RADII]

    L += ['', '### Pano-frame crowd -> AI by the label\'s validation', '',
          'Human majority (PS AI-validator votes excluded), then the feed\'s `correct` '
          '(which does include them). Rates are pano-frame agree rates at '
          f'{tier_name(OP)}, radius {PANO_RADIUS}.', '',
          '| bucket | agree rate |', '|---|---|']
    for b in ('agreed', 'disagreed', 'unvalidated'):
        L.append(f'| human: {b} | {pano_rate([lab for lab in on_run if lab.human == b], OP, PANO_RADIUS)} |')
    for b, name in ((True, 'true'), (False, 'false'), (None, 'null')):
        L.append(f'| feed `correct` = {name} | '
                 f'{pano_rate([lab for lab in on_run if lab.ps_correct is b], OP, PANO_RADIUS)} |')

    # --- pano frame AI -> crowd lower bound
    L += ['', '### Pano frame, AI -> crowd (lower bound only)', '',
          'Share of operational detections on crowd-labelled processed panos that a crowd '
          'label on the same pano claims. A LOWER bound: auditors label each ramp once, '
          'from one of the several panos that see it, so a correct detection of a ramp the '
          'auditor marked from a neighbouring pano counts as unclaimed here.', '']
    for t in tiers:
        _hit, tot, ah = pano[(t, PANO_RADIUS)]
        L.append(f'- {tier_name(t)}: {rate(ah, tot)}')

    # --- world frame
    def world_rate(labs, t, h, r, kind='crowd_site'):
        hit = world[(t, h)][kind][r]
        return rate(sum(1 for lab in labs if lab.id in hit), len(labs))

    full_labs = [lab for lab in curb if lab.region_class == 'full']
    L += ['', '## World frame, crowd -> AI (a bound, not the headline)', '',
          'Crowd labels at PS\'s own lat/lng (viewer estimator, SidewalkWebpage#4766) vs '
          'fused AI sites from the labeler\'s flat-ground raycast. Two independent '
          'placement errors (and RampNet#101\'s range scale error) sit between them, so '
          'a miss here can be geometry rather than detection. **sites** = one-to-one '
          'against operational fused sites; **union** = any single operational raycast '
          'within the radius (loosest bound). Denominator: all in-scope crowd CurbRamp '
          'labels, including those on panos the run never processed.', '',
          '| tier | height | radius | sites (all in scope) | sites (full regions) | '
          'union (all in scope) |', '|---|---|---:|---|---|---|']
    for t in tiers:
        for h in heights:
            for r in WORLD_RADII:
                L.append(f'| {tier_name(t)} | {height_name(h)} | {r:g} m | '
                         f'{world_rate(curb, t, h, r)} | {world_rate(full_labs, t, h, r)} | '
                         f'{world_rate(curb, t, h, r, "crowd_union")} |')

    L.append('')
    for t in tiers:
        k = chance_floor([Pt(lab.id, lab.e, lab.n) for lab in curb], world[(t, H)]['sites'],
                         WORLD_RADIUS)
        L.append(f'- chance floor, {tier_name(t)}, {H:g} m, {WORLD_RADIUS:g} m: every crowd '
                 f'label displaced {CHANCE_SHIFT_M:g} m in a random direction (seed '
                 f'{CHANCE_SEED}) still matches a site {rate(k, len(curb))} of the time — the '
                 'share of the world-frame rate that site density alone would produce')

    # pano vs world on the same labels
    hit_p = pano[(OP, PANO_RADIUS)][0]
    hit_w = world[(OP, H)]['crowd_site'][WORLD_RADIUS]
    both = sum(1 for lab in on_run if lab.id in hit_p and lab.id in hit_w)
    p_only = sum(1 for lab in on_run if lab.id in hit_p and lab.id not in hit_w)
    w_only = sum(1 for lab in on_run if lab.id not in hit_p and lab.id in hit_w)
    neither = len(on_run) - both - p_only - w_only
    L += ['', '### The two frames on the same labels', '',
          f'Crowd labels on processed panos ({len(on_run)}), {tier_name(OP)}, pano radius '
          f'{PANO_RADIUS}, world {WORLD_RADIUS:g} m at {H:g} m:', '',
          f'- agree in both frames: {both}',
          f'- pano frame only (the world frame loses them: placement, or beyond the 25 m '
          f'raycast envelope): {p_only}',
          f'- world frame only (the pano missed, a site from another view is near): {w_only}',
          f'- neither: {neither}',
          f'- in-run labels: pano {rate(both + p_only, len(on_run))} vs world '
          f'{rate(both + w_only, len(on_run))}']
    off_run = [lab for lab in curb if lab.pano_id not in run_by_id]
    L.append(f'- labels on panos the run did NOT process ({len(off_run)}): world '
             f'{world_rate(off_run, OP, H, WORLD_RADIUS)} — scorable only in the world frame')

    # --- AI -> crowd world
    L += ['', '## World frame, AI -> crowd (fully audited regions only)', '',
          'Denominator: operational fused sites located in a fully audited region. '
          '**one-to-one** = matched to a distinct crowd CurbRamp label; **any** = a crowd '
          'CurbRamp label within radius (duplicate AI sites on one ramp count); '
          '**on NoCurbRamp** = a crowd NoCurbRamp label within radius (the auditor looked '
          'and said there is no ramp).', '',
          '| tier | height | radius | one-to-one | any | on NoCurbRamp | '
          'on NoCurbRamp, no CurbRamp near |', '|---|---|---:|---|---|---|---|']
    for t in tiers:
        for h in heights:
            w = world[(t, h)]
            n_full = len(w['full_sites'])
            for r in WORLD_RADII:
                L.append(f'| {tier_name(t)} | {height_name(h)} | {r:g} m | '
                         f'{rate(len(w["site_one"][r]), n_full)} | '
                         f'{rate(len(w["site_any"][r]), n_full)} | '
                         f'{rate(len(w["site_ncr"][r]), n_full)} | '
                         f'{rate(len(w["site_ncr_only"][r]), n_full)} |')

    # --- why AI sites go unlabelled
    wo = world[(OP, H)]
    curb_pts_all = [Pt(lab.id, lab.e, lab.n) for lab in curb]
    ub = unmatched_breakdown(wo, curb_pts_all, OP, WORLD_RADIUS)
    n_full_sites = len(wo['full_sites'])
    L += ['', f'### Where the unlabelled AI sites are ({tier_name(OP)}, {H:g} m, '
          f'{WORLD_RADIUS:g} m; {n_full_sites} full-region sites)', '',
          '| nearest crowd CurbRamp label | sites | share | seen from >= 2 panos | '
          f'max member >= {BM:g} |', '|---|---:|---:|---|---|']
    for b, row in ub.items():
        L.append(f"| {b} | {row['sites']} | {row['sites'] / n_full_sites:.3f} | "
                 f"{rate(row['multi_view'], row['sites'])} | "
                 f"{rate(row['benchmark_tier'], row['sites'])} |")

    # --- GT adjudication
    wb = world[(BM, H)]
    adj = gt_adjudication(args.city, args.benchmark_root, run_panos, wb, curb, regions,
                          args.full_completion)
    summary_extra = []
    L += ['', '## Who is right when they disagree: RampNet GT adjudication', '']
    if adj is None:
        L.append(f'- skipped: no RampNet benchmark split at {args.benchmark_root / args.city}')
    else:
        c = adj['cover']
        (tp_n, fp_n), (tp_f, fp_f) = adj['prec'][True], adj['prec'][False]
        L += [f'World frame, {tier_name(BM)} (the tier the bundle was judged at), '
              f'{H:g} m, {WORLD_RADIUS:g} m; GT built as eval_sites builds it '
              f"({adj['counts']['judged']} judged panos, {adj['n_warnings']} skipped with a "
              'warning). The GT was made during a RampNet review, so it is RampNet-anchored '
              'and favours the AI; read the crowd column as a lower bound.', '',
              f"- GT recall-pool ramps in fully audited regions: {adj['pool']}",
              f"- covered by a crowd CurbRamp label: {rate(adj['crowd_recall'], adj['pool'])}",
              f"- covered by an operational AI site: {rate(adj['ai_recall'], adj['pool'])}",
              f"- both {c['both']}, crowd only {c['crowd only']}, AI only {c['AI only']}, "
              f"neither {c['neither']}",
              f'- precision of full-region AI sites WITH a crowd label within '
              f'{WORLD_RADIUS:g} m: {rate(tp_n, tp_n + fp_n)}',
              f'- precision of full-region AI sites with NO crowd label within '
              f'{WORLD_RADIUS:g} m: {rate(tp_f, tp_f + fp_f)}. Measured on '
              f'{tier_name(BM)} sites (the tier the bundle was judged at): the precision of '
              f'the extra sites the {tier_name(OP)} tier adds is NOT measured here']
        n_unm = n_full_sites - len(wo['site_one'][WORLD_RADIUS])
        n_far = ub.get('no label within 20 m', {'sites': 0})['sites']
        n_full_labs = sum(1 for lab in curb if lab.region_class == 'full')
        sk = skipped_ramp_estimate(n_full_labs, adj['crowd_recall'], adj['pool'], n_unm)
        L += ['', f'### How much of the {tier_name(OP)} AI -> crowd gap crowd omissions '
              'can explain', '',
              f'- unmatched full-region AI sites ({tier_name(OP)}, {H:g} m, '
              f'{WORLD_RADIUS:g} m one-to-one): {n_unm} of {n_full_sites}']
        if sk:
            L += [f"- crowd recall {sk['recall']:.3f} over {n_full_labs} full-region crowd "
                  f"labels implies about **{sk['skipped']:.0f}** skipped ramps "
                  f"({sk['skipped_lo']:.0f}-{sk['skipped_hi']:.0f} over the recall's 95% "
                  'CI), assuming one label per ramp',
                  f"- so crowd omissions can explain **at most {sk['share']:.0%}** of the "
                  f"unmatched sites ({sk['share_lo']:.0%}-{sk['share_hi']:.0%}); each "
                  'skipped ramp accounts for at most one site, and one the AI also missed '
                  'accounts for none']
        L += [f'- unmatched sites with no crowd label within 20 m: {n_far} '
              f'({n_far / n_unm:.0%} of the unmatched)',
              '- the rest of the gap is fragmentation, placement beyond '
              f'{WORLD_RADIUS:g} m, and low-confidence single-view sites that nobody '
              f'judged. Precision at {tier_name(OP)} is unmeasured']
        summary_extra = [
            {'frame': 'world', 'direction': 'AI->crowd unmatched', 'tier': OP,
             'height': height_name(H), 'radius': WORLD_RADIUS, 'scope': 'full regions',
             'k': n_unm, 'n': n_full_sites},
            {'frame': 'world', 'direction': 'AI->crowd unmatched, no label within 20 m',
             'tier': OP, 'height': height_name(H), 'radius': WORLD_RADIUS,
             'scope': 'full regions', 'k': n_far, 'n': n_unm}]
        if sk:
            summary_extra.append(
                {'frame': 'world', 'direction': 'implied skipped ramps / unmatched sites',
                 'tier': OP, 'height': height_name(H), 'radius': WORLD_RADIUS,
                 'scope': 'full regions', 'k': round(sk['skipped']), 'n': n_unm})

    # --- vintage
    L += ['', '## Vintage: how much of the crowd -> AI miss rate is imagery age', '',
          'Each in-scope crowd label is placed against the run\'s imagery: **same pano** '
          '(scored in both frames), else against the run pano whose GSV history lists the '
          f'crowd pano, else the nearest run pano within {NEAREST_PANO_M:g} m; months = '
          'ours minus theirs. The WORLD rate is used for every stratum so they are '
          f'comparable ({tier_name(OP)}, {H:g} m, {WORLD_RADIUS:g} m, sites one-to-one); '
          'the pano-frame rate is given where it exists. **excess misses** = misses '
          'beyond what the same-pano stratum\'s world rate predicts for that many labels '
          '- the part of the gap a vintage difference could explain (an upper bound on it: '
          'any other difference between the strata lands here too).', '']
    L += ['| stratum | labels | history-linked | pano-frame agree | world agree | '
          'excess misses |', '|---|---:|---:|---|---|---:|']
    sw = world[(OP, H)]['crowd_site'][WORLD_RADIUS]
    on_run_ids = {lab.id for lab in on_run}
    same = [lab for lab in curb if strata[lab.id][0] == 'same pano']
    base = sum(1 for lab in same if lab.id in sw) / len(same) if same else 0.0
    total_excess = 0.0
    vint_rows = []
    for st in VINTAGE_BUCKETS:
        labs = [lab for lab in curb if strata[lab.id][0] == st]
        if not labs:
            continue
        k = sum(1 for lab in labs if lab.id in sw)
        excess = 0.0 if st == 'same pano' else (base * len(labs) - k)
        total_excess += excess
        linked = 0 if st == 'same pano' else sum(1 for lab in labs if lab.pano_id in history)
        prate = pano_rate([lab for lab in labs if lab.id in on_run_ids], OP, PANO_RADIUS) \
            if st == 'same pano' else '-'
        L.append(f'| {st} | {len(labs)} | {linked} | {prate} | {rate(k, len(labs))} | '
                 f'{excess:+.1f} |')
        vint_rows.append({'stratum': st, 'labels': len(labs), 'history_linked': linked,
                          'world_agree': k, 'world_rate': round(k / len(labs), 4),
                          'excess_misses': round(excess, 2)})
    n_miss = len(curb) - sum(1 for lab in curb if lab.id in sw)
    L += ['', (f'- world-frame misses overall: {n_miss} of {len(curb)}; excess over the '
               f'same-pano rate: {total_excess:.1f} ({total_excess / n_miss:.0%} of the '
               'misses)') if n_miss else '- no world-frame misses']

    # --- per region
    L += ['', '## Per region', '',
          f'{tier_name(OP)}; pano radius {PANO_RADIUS}; world {H:g} m, {WORLD_RADIUS:g} m. '
          '**AI -> crowd** = operational sites located in the region matched one-to-one to '
          'a crowd CurbRamp label. Partial regions are listed for completeness; their '
          'AI -> crowd is biased low (unaudited streets) and never pooled.', '',
          '| region | class | completion | crowd labels | on run panos | pano agree | '
          'world agree | AI sites | AI -> crowd |', '|---|---|---:|---:|---:|---|---|---:|---|']
    site_ids_by_region = {}
    for sid, rid in wo['site_region'].items():
        site_ids_by_region.setdefault(rid, set()).add(sid)
    region_rows = []
    for rid in full_ids + partial_ids:
        labs = [lab for lab in curb if lab.region_id == rid]
        onr = [lab for lab in labs if lab.id in on_run_ids]
        sids = site_ids_by_region.get(rid, set())
        k_p = sum(1 for lab in onr if lab.id in hit_p)
        k_w = sum(1 for lab in labs if lab.id in hit_w)
        region_sites = [s for s in wo['sites'] if s.id in sids]
        k_a = len(match_one_to_one(region_sites, curb_pts_all, WORLD_RADIUS))
        cls = region_class(rid, regions, args.full_completion)
        L.append(f'| {rid} | {cls} | {regions[rid].completion:.3f} | {len(labs)} | '
                 f'{len(onr)} | {rate(k_p, len(onr))} | {rate(k_w, len(labs))} | '
                 f'{len(sids)} | {rate(k_a, len(sids))} |')
        region_rows.append({'region_id': rid, 'class': cls,
                            'completion_rate': round(regions[rid].completion, 4),
                            'crowd_labels': len(labs), 'on_run_panos': len(onr),
                            'pano_agree': k_p, 'world_agree': k_w, 'ai_sites': len(sids),
                            'ai_sites_matched': k_a})

    # ================================================================== CSVs
    def write_csv(name, rows):
        with open(out / name, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    label_rows = []
    for lab in curb:
        st, months = strata[lab.id]
        row = {'label_uid': lab.uid, 'label_id': lab.label_id, 'region_id': lab.region_id,
               'region_class': lab.region_class, 'pano_id': lab.pano_id,
               'pano_in_run': lab.id in on_run_ids, 'capture_date': lab.capture_date,
               'vintage': st, 'months_ours_minus_theirs': months,
               'human': lab.human, 'feed_correct': lab.ps_correct}
        for t in tiers:
            hit = pano[(t, PANO_RADIUS)][0]
            row[f'pano_agree_{t:g}'] = (lab.id in hit) if lab.id in on_run_ids else None
            for h in heights:
                row[f'world_agree_{t:g}_{height_name(h).replace(" ", "")}'] = \
                    lab.id in world[(t, h)]['crowd_site'][WORLD_RADIUS]
        label_rows.append(row)
    uids = [r['label_uid'] for r in label_rows]
    if len(set(uids)) != len(uids):
        raise SystemExit('label_uid is not unique across labels.csv')
    write_csv('labels.csv', label_rows)
    write_csv('regions.csv', region_rows)
    write_csv('vintage.csv', vint_rows)
    summary = []
    for t in tiers:
        for r in PANO_RADII:
            hit, tot, ah = pano[(t, r)]
            summary.append({'frame': 'pano', 'direction': 'crowd->AI', 'tier': t,
                            'height': '', 'radius': r, 'scope': 'all in scope',
                            'k': sum(1 for lab in on_run if lab.id in hit),
                            'n': len(on_run)})
            summary.append({'frame': 'pano', 'direction': 'crowd->AI any detection',
                            'tier': t, 'height': '', 'radius': r,
                            'scope': 'all in scope',
                            'k': sum(1 for lab in on_run if lab.id in pano_any[(t, r)]),
                            'n': len(on_run)})
            summary.append({'frame': 'pano', 'direction': 'AI->crowd (lower bound)',
                            'tier': t, 'height': '', 'radius': r,
                            'scope': 'crowd-labelled panos', 'k': ah, 'n': tot})
        for h in heights:
            w = world[(t, h)]
            for r in WORLD_RADII:
                for scope, labs in (('all in scope', curb), ('full regions', full_labs)):
                    summary.append({'frame': 'world', 'direction': 'crowd->AI sites',
                                    'tier': t, 'height': height_name(h), 'radius': r,
                                    'scope': scope, 'n': len(labs),
                                    'k': sum(1 for lab in labs
                                             if lab.id in w['crowd_site'][r])})
                summary.append({'frame': 'world', 'direction': 'crowd->AI union',
                                'tier': t, 'height': height_name(h), 'radius': r,
                                'scope': 'all in scope', 'n': len(curb),
                                'k': sum(1 for lab in curb if lab.id in w['crowd_union'][r])})
                for key, name in (('site_one', 'AI->crowd one-to-one'),
                                  ('site_any', 'AI->crowd any'),
                                  ('site_ncr', 'AI on NoCurbRamp'),
                                  ('site_ncr_only', 'AI on NoCurbRamp, no CurbRamp')):
                    summary.append({'frame': 'world', 'direction': name, 'tier': t,
                                    'height': height_name(h), 'radius': r,
                                    'scope': 'full regions', 'k': len(w[key][r]),
                                    'n': len(w['full_sites'])})
    summary += summary_extra
    for row in summary:
        row['rate'] = round(row['k'] / row['n'], 4) if row['n'] else None
    write_csv('summary.csv', summary)

    report = '\n'.join(L) + '\n'
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
