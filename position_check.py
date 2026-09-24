#!/usr/bin/env python
"""Check a run's pano positions against the street network before anything is submitted.

Why this exists (SidewalkWebpage#5361): in Laurens, IA the Mapillary auto-labels sat
8-10 m west of every intersection. Nothing in the raycast or the server's transform was
wrong; the *pano positions* were. Mapillary serves two positions per image — the raw GPS
fix (`geometry`) and an SfM-corrected one (`computed_geometry`) — and the source submits
the SfM one. SfM alignment to GPS is one similarity transform per reconstruction, so a
whole sequence can drift several metres in one direction, and every label placed from
that pano inherits the drift 1:1. Raw GPS had no such bias there. Which field is better
is a per-city, per-sequence question, so it has to be measured, not assumed.

The measurement needs no ground truth and no second imagery source: a car drives on the
street, so a pano's perpendicular distance to the nearest OpenStreetMap centerline is a
read on position error. It is a coarse one. A camera is legitimately metres from a
centerline (lane, parking, one-way geometry), so the metric FLOORS: GSV, whose positions
are not in question, measures 1.75 m median against the Laurens centerlines where
Mapillary SfM measures 3.73 m and raw GPS 2.59 m. So this check catches gross block drift
and says so plainly when a difference is below what it can see (issue #62).

The verdict per sequence, on the field it actually submitted:
  - OFF THE STREET: median unsigned cross-track distance above GROSS_OFF_STREET_M
    (--threshold), or most of it beyond MAX_SNAP_M of any street while the other field
    is not (beyond_snap).
  - FLAGGED: off the street AND the other field fixes it: pano by pano, against the same
    street, it is closer by more than RESOLUTION_FLOOR_M (the paired metric, median of
    cross_sfm - cross_raw) and no more than MAX_IQR_RATIO times as scattered.
  - BOTH_OFF: off the street and no such fix. Reported, never gated.
  - UNDECIDABLE: the fields differ by RESOLUTION_FLOOR_M or less. Reported with the
    caveat that the reference cannot adjudicate it, whichever sign it has.
The #60 signed per-axis bias stays in every row as the off-street picture a human reads;
it no longer decides anything, because a lane offset cancels in it over a sequence driven
both ways (Richmond's jKtaJMek7wQl5AOH28qdcm read as "raw fixes it" by bias while raw was
3.3 m worse per pano). There is no per-sequence automatic field choice: the whole-run
field (main.py --mapillary-position) is a measured, per-city decision made once.

Repositioning a file that is already live is a whole-city decision, not a per-file one:
PS upserts the pano row, so it moves every live label on the moved panos. The check
reports the live campaigns from `<file>.submission.json`; scripts/reposition.py and
send_to_ps.py refuse to move live panos without --reposition-live-city.

This runs automatically at the end of every main.py run (--no-position-check skips it),
and send_to_ps.py refuses a Mapillary file whose check is missing, stale (another file
hash, or another verdict RULE) or flagged (--ignore-position-check overrides). The module
lives at the repo root so both stages import it; scripts/position_check.py is a shim for
the manual commands below.

Usage:
    python scripts/position_check.py runs/laurens                 # console + position_check.json
    python scripts/position_check.py runs/laurens --report        # + position_report.html
    python scripts/position_check.py runs/laurens --report \\
        --labels runs/laurens/ps_labels.geojson \\                 # PS v3 rawLabels feed
        --reference runs/laurens_gsv                              # independent run, same area

Exit status 1 when any sequence with >= --min-sequence panos is flagged, so a deploy
script can gate on it. Streets come from Overpass once and are cached beside the run
(osm_streets.json; a partial or errored answer is never cached). Stdlib only, like geo.py.

`--results FILE` checks another results file against the run's area and streets — the
way to confirm a scripts/reposition.py output before it is submitted, without swapping
it into results.jsonl (a repositioned file is a submission artifact, not a resumable run).
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
from detectors import OPERATIONAL_CONFIDENCE, on_camera_rig  # noqa: E402

OVERPASS_ENDPOINTS = (
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
)
# Streets a survey car drives. Service roads and paths are excluded: a pano on a
# driveway or alley would otherwise be measured against the wrong line.
STREET_HIGHWAY_RE = '^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street)(_link)?$'
MAX_SNAP_M = 30.0        # a pano farther than this from any street is not scored
AXIS_TOLERANCE_DEG = 25  # a segment within this of N-S (E-W) counts for the signed east (north) offset
GRID_M = 40.0
DEFAULT_MIN_SEQUENCE = 20
MIN_AXIS_SAMPLES = 10        # panos a sequence needs on one street family before its signed bias is
                             # reported, and paired / cross-track samples before it can be judged at all

# The verdict (issue #62). The reference is OSM centerlines, and a camera is legitimately
# metres from one (lane position, parking, one-way geometry): GSV, whose positioning is not
# in question, measures 1.75 m median cross-track against the same Laurens centerlines
# that Mapillary SfM measures 3.73 m against. That floor bounds what this check can see.
GROSS_OFF_STREET_M = 5.0     # a sequence is OFF THE STREET when the submitted field's median
                             # unsigned cross-track distance exceeds this: ~3x the GSV floor, and
                             # well under the 8-10 m block drift of SidewalkWebpage#5361. Chosen
                             # from the 4/5/6 m table in the #62 PR, not from first principles.
RESOLUTION_FLOOR_M = 2.0     # the other field only FIXES a sequence when, pano by pano on the
                             # same street, it is closer by more than this (paired median). Below
                             # it the difference is inside the reference's own floor: reported as
                             # undecidable, never acted on.
# Names the verdict rule a position_check.json was written under. send_to_ps.py treats a
# check from any other rule as stale and main.py re-runs it, so a gate can never pass (or
# refuse) a file on a rule the code no longer applies.
RULE = 'paired-unsigned-v1 (issue #62)'
MAX_IQR_RATIO = 1.5          # ...and when its cross-track spread is not more than this times the
                             # submitted field's: jitter is per-pano error too, so a field that is
                             # closer at the median but twice as scattered is not a fix.
HIST_HALF_WIDTH_M = 25   # signed-offset histograms cover [-25, 25) in 1 m bins

# Position fields per source. 'submitted' is always pano.lat/lng — whatever the run
# actually wrote; the others are re-read from source_metadata.
MAPILLARY_FIELDS = {
    'sfm': ('SfM computed_geometry', 'computed_geometry'),
    'raw': ('raw GPS geometry', 'geometry'),
}


# --------------------------------------------------------------------------- geometry
class StreetIndex:
    """Nearest-segment queries over OSM street centerlines, in frame metres.

    Segments are bucketed into GRID_M cells over their bbox padded by MAX_SNAP_M, so a
    query looks at exactly one cell and still sees every segment within MAX_SNAP_M.

    Signed offsets need a fixed world direction per street family, or a lane offset on
    the way out and the way back would not cancel. The two families are the street
    grid's own axes: theta0 is the dominant segment bearing (length-weighted, mod 90,
    folded into [-45, 45)), family 'a' runs along theta0 (north-south for a cardinal
    grid) with positive offsets toward theta0+90 (east), family 'b' runs along
    theta0+90 with positive offsets toward theta0 (north). Richmond's grid sits ~30 deg
    off north, which is why the axes are measured rather than assumed.
    """

    def __init__(self, segments):
        # segments: list of (ax, ay, bx, by, name, street_key)
        self.segments = segments
        self.cells = defaultdict(list)
        sx = sy = 0.0
        for i, (ax, ay, bx, by, *_rest) in enumerate(segments):
            x0, x1 = sorted((ax, bx))
            y0, y1 = sorted((ay, by))
            for cx in range(int((x0 - MAX_SNAP_M) // GRID_M), int((x1 + MAX_SNAP_M) // GRID_M) + 1):
                for cy in range(int((y0 - MAX_SNAP_M) // GRID_M), int((y1 + MAX_SNAP_M) // GRID_M) + 1):
                    self.cells[(cx, cy)].append(i)
            length = math.hypot(bx - ax, by - ay)
            bearing4 = 4.0 * math.atan2(bx - ax, by - ay)   # mod-90 circular mean
            sx += length * math.cos(bearing4)
            sy += length * math.sin(bearing4)
        theta0 = math.degrees(math.atan2(sy, sx)) / 4.0 if (sx or sy) else 0.0
        self.theta0 = ((theta0 + 45.0) % 90.0) - 45.0
        t = math.radians(self.theta0)
        self.normal = {'a': (math.cos(t), -math.sin(t)),   # bearing theta0 + 90: east-ish
                       'b': (math.sin(t), math.cos(t))}    # bearing theta0: north-ish

    def axes(self):
        """Human labels for the two families, cardinal when the grid is within 15 deg of it."""
        t = self.theta0
        if abs(t) < 15.0:
            return {'theta0_deg': round(t, 1),
                    'a': {'streets': 'N-S streets', 'pos': 'east', 'neg': 'west'},
                    'b': {'streets': 'E-W streets', 'pos': 'north', 'neg': 'south'}}
        b = lambda deg: f'{round(deg % 360):d}°'  # noqa: E731
        return {'theta0_deg': round(t, 1),
                'a': {'streets': f'streets bearing {b(t)}', 'pos': f'toward {b(t + 90)}', 'neg': f'toward {b(t + 270)}'},
                'b': {'streets': f'streets bearing {b(t + 90)}', 'pos': f'toward {b(t)}', 'neg': f'toward {b(t + 180)}'}}

    def nearest(self, x, y, max_d=MAX_SNAP_M):
        """(distance, foot_x, foot_y, segment_index) of the nearest segment, or None."""
        best = None
        for i in self.cells.get((int(x // GRID_M), int(y // GRID_M)), ()):
            ax, ay, bx, by, *_rest = self.segments[i]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(x - px, y - py)
            if d <= max_d and (best is None or d < best[0]):
                best = (d, px, py, i)
        return best

    def axis_of(self, i):
        """'a', 'b' or None (diagonal to the grid) for a segment, by its bearing."""
        ax, ay, bx, by, *_rest = self.segments[i]
        rel = (math.degrees(math.atan2(bx - ax, by - ay)) - self.theta0) % 180.0
        if rel < AXIS_TOLERANCE_DEG or rel > 180 - AXIS_TOLERANCE_DEG:
            return 'a'
        if abs(rel - 90.0) < AXIS_TOLERANCE_DEG:
            return 'b'
        return None


def measure_point(index, x, y):
    """Offsets of one point from the street network, or None when no street is within reach.

    cross:      unsigned perpendicular distance to the nearest street of any orientation
    a:          signed offset across the street when it belongs to family 'a' (N-S on a
                cardinal grid; positive = east of it), else None
    b:          signed offset across a family-'b' street (E-W; positive = north), else None
    street:     the street's OSM name ('' when unnamed)
    street_key: what "the same street" means for the paired metric — the name, or the way
                id for an unnamed way, so a named street split into several ways still pairs
    """
    hit = index.nearest(x, y)
    if hit is None:
        return None
    d, px, py, i = hit
    axis = index.axis_of(i)
    out = {'cross': d, 'a': None, 'b': None, 'street': index.segments[i][4],
           'street_key': index.segments[i][5]}
    if axis:
        nx, ny = index.normal[axis]
        out[axis] = (x - px) * nx + (y - py) * ny
    return out


# --------------------------------------------------------------------------- OSM
def area_bbox(area_geojson):
    coords = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            coords.append(c)
        else:
            for x in c:
                walk(x)
    walk(area_geojson['coordinates'])
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lngs), min(lats), max(lngs), max(lats)


def count_ways(payload):
    return sum(1 for e in payload.get('elements', []) if e.get('type') == 'way')


def validate_osm_payload(payload):
    """Reason a payload is unusable, or None. Overpass answers a query timeout or memory
    exhaustion with HTTP 200, a top-level `remark` and whatever elements it had produced
    so far — possibly none, possibly half the streets — so a payload has to be checked
    for that before it is trusted, and above all before it is cached."""
    if not isinstance(payload, dict) or 'elements' not in payload:
        return 'no elements key in the response'
    if payload.get('remark'):
        return f"Overpass remark: {payload['remark']}"
    if not count_ways(payload):
        return 'zero streets returned'
    return None


def fetch_osm_streets(bbox, pad_deg=0.003):
    """Overpass query for drivable streets in bbox (padded so edge panos still snap).
    Only a complete answer with at least one way is returned; anything else falls
    through to the next mirror."""
    min_lng, min_lat, max_lng, max_lat = bbox
    query = (f'[out:json][timeout:90];'
             f'way["highway"~"{STREET_HIGHWAY_RE}"]'
             f'({min_lat - pad_deg},{min_lng - pad_deg},{max_lat + pad_deg},{max_lng + pad_deg});'
             f'out geom;')
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            data = urllib.parse.urlencode({'data': query}).encode()
            req = urllib.request.Request(endpoint, data=data,
                                         headers={'User-Agent': 'sidewalk-auto-labeler position_check'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
            problem = validate_osm_payload(payload)
            if problem is None:
                return payload
            last_error = f'{endpoint}: {problem}'
        except Exception as exc:  # try the next mirror
            last_error = f'{endpoint}: {exc}'
        time.sleep(2)
    raise SystemExit(f'Overpass query failed on every endpoint: {last_error}\n'
                     f'   (zero streets for a real area means the bbox is wrong — check area.geojson)')


def load_or_fetch_streets(run_dir, area_geojson, osm_path=None):
    """(payload, path, cached). A cached file is re-validated and its query metadata
    compared with the current highway filter, so a bad or stale cache is refetched
    rather than silently reused; the cache is written only after the payload validates."""
    path = Path(osm_path) if osm_path else run_dir / 'osm_streets.json'
    bbox = area_bbox(area_geojson)
    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                payload = json.load(f)
            problem = validate_osm_payload(payload)
        except ValueError as exc:  # truncated or otherwise unparseable: same as a bad answer
            payload, problem = {}, f'unreadable JSON ({exc})'
        stale = (payload.get('_query') or {}).get('highway') not in (None, STREET_HIGHWAY_RE)
        if problem is None and not stale:
            return payload, path, True
        if osm_path:  # an explicit --osm file is the user's; refuse rather than overwrite it
            raise SystemExit(f'{path}: {problem or "built with a different highway filter"}')
        print(f"-> {path}: {problem or 'built with a different highway filter'}; refetching")
    payload = fetch_osm_streets(bbox)
    payload['_query'] = {'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         'bbox': bbox, 'highway': STREET_HIGHWAY_RE}
    tmp = path.with_suffix('.json.tmp')   # atomic: a Ctrl-C mid-write must not leave a half file
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))
    os.replace(tmp, path)
    return payload, path, False


def street_segments(osm_payload, frame):
    segments = []
    for way in osm_payload.get('elements', []):
        if way.get('type') != 'way' or not way.get('geometry'):
            continue
        name = (way.get('tags') or {}).get('name') or ''
        key = name or f"way/{way.get('id')}"
        pts = [frame.to_enu(p['lat'], p['lon']) for p in way['geometry']]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            segments.append((ax, ay, bx, by, name, key))
    return segments


# --------------------------------------------------------------------------- run loading
def load_run(run_dir, results_path=None):
    """(manifest, area, records). `results_path` swaps in another results file — e.g. a
    scripts/reposition.py output — while the area and streets stay the run's own."""
    run_dir = Path(run_dir)
    with open(run_dir / 'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(run_dir / 'area.geojson', encoding='utf-8') as f:
        area = json.load(f)
    records = []
    with open(Path(results_path) if results_path else run_dir / 'results.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return manifest, area, records


def check_path_for(results_path):
    """Where the check for a results file lives: position_check.json beside results.jsonl,
    <stem>.position_check.json beside anything else (a reposition.py output), so a
    --results check never overwrites the run's own verdict. Single home for the rule —
    main(), send_to_ps.py's gate and reposition.py all read it from here."""
    results_path = Path(results_path)
    prefix = '' if results_path.name == 'results.jsonl' else results_path.stem + '.'
    return results_path.parent / f'{prefix}position_check.json'


def report_path_for(results_path):
    return check_path_for(results_path).with_name(check_path_for(results_path).name.replace(
        'position_check.json', 'position_report.html'))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_check(results_path):
    """(check dict, None) for the check beside a results file, or (None, reason) when it is
    missing or unreadable. Staleness is the caller's call (compare results_sha256)."""
    path = check_path_for(results_path)
    if not path.exists():
        return None, f'no position check beside {results_path.as_posix()} ({path.name} missing)'
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f), None
    except ValueError as exc:
        return None, f'{path} is unreadable ({exc})'


def submission_record_for(results_path):
    """Where send_to_ps.py keeps a results file's submission record (<file>.submission.json).
    Duplicated here rather than imported: send_to_ps pulls in requests, and this module is
    stdlib-only so main.py and the scripts can run it anywhere."""
    results_path = Path(results_path)
    return results_path.with_name(results_path.name + '.submission.json')


def live_campaigns(results_path):
    """[{endpoint, submitted_lines, labels_submitted}] for every endpoint the submission
    record beside `results_path` says holds at least one line of it; [] when there is no
    record. An unreadable record raises ValueError: it is the memory of what is live on a
    server, so "can't read it" must not decay into "nothing is live" (as in send_to_ps.py)."""
    path = submission_record_for(results_path)
    if not path.exists():
        return []
    try:
        with open(path, encoding='utf-8') as f:
            record = json.load(f)
    except ValueError as exc:
        raise ValueError(f'{path} is unreadable ({exc}): it records what is live on a server, so it '
                         f'has to be repaired, not ignored') from exc
    out = []
    for endpoint, state in sorted((record.get('endpoints') or {}).items()):
        lines = int(state.get('submitted_lines') or 0)
        if lines > 0:
            out.append({'endpoint': endpoint, 'submitted_lines': lines,
                        'labels_submitted': int(state.get('labels_submitted') or 0)})
    return out


def repo_relative(path):
    """`path` relative to the repo root when it lies inside it (what the tracked JSON and
    report record — never an absolute local path), else as given."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def pano_positions(pano):
    """{field: (lat, lng)} for one pano block. 'submitted' is what the run wrote."""
    out = {'submitted': (pano['lat'], pano['lng'])}
    if pano.get('source') == 'mapillary':
        meta = pano.get('source_metadata') or {}
        for field, (_label, key) in MAPILLARY_FIELDS.items():
            coords = (meta.get(key) or {}).get('coordinates')
            if coords:
                out[field] = (coords[1], coords[0])
    return out


def field_labels(source):
    labels = {'submitted': 'submitted position (pano lat/lng)'}
    if source == 'mapillary':
        labels.update({k: v[0] for k, v in MAPILLARY_FIELDS.items()})
    return labels


# --------------------------------------------------------------------------- statistics
def quantiles(values, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    v = sorted(values)
    if not v:
        return [None] * len(qs)
    return [round(v[min(len(v) - 1, int(len(v) * q))], 2) for q in qs]


def summarize_offsets(rows):
    """Distribution summary for a list of measure_point() results (None entries dropped)."""
    rows = [r for r in rows if r]
    cross = [r['cross'] for r in rows]

    def block(vals, signed=False):
        if not vals:
            return {'n': 0}
        q = quantiles(vals)
        out = {'n': len(vals), 'p5': q[0], 'q25': q[1], 'median': q[2], 'q75': q[3], 'p95': q[4]}
        if signed:
            out['neg_share'] = round(sum(v < 0 for v in vals) / len(vals), 3)
            out['bins'] = histogram(vals)
        return out

    return {'n_scored': len(rows), 'cross_track': block(cross),
            'across_a': block([r['a'] for r in rows if r['a'] is not None], signed=True),
            'across_b': block([r['b'] for r in rows if r['b'] is not None], signed=True)}


def histogram(values, half=HIST_HALF_WIDTH_M):
    bins = [0] * (2 * half)
    for v in values:
        i = int(math.floor(v + half))
        if 0 <= i < len(bins):
            bins[i] += 1
    return bins


def _median(values):
    return round(statistics.median(values), 2) if values else None


def _iqr(values):
    """q75 - q25 on the same quantile rule as summarize_offsets, or None when empty."""
    if not values:
        return None
    q25, q75 = quantiles(values, (0.25, 0.75))
    return round(q75 - q25, 2)


def paired_verdict(paired_median, paired_n, floor_m=RESOLUTION_FLOOR_M):
    """Which field the paired metric says is closer to the street, or 'undecidable'.

    `paired_median` is median(cross_sfm - cross_raw) over panos where both fields snap to
    the same street, so positive means raw is closer. Lane offset and the OSM geometry's
    own error are identical for both fields on one pano and cancel in the difference, which
    is why this is the statistic and the signed bias is not (a lane offset cancels over a
    sequence only when its two driving directions are balanced; per pano it never has to).
    A difference inside the reference's floor is 'undecidable': the check cannot see it,
    whichever way the sign points. None when there are too few pairs to say anything.
    """
    if paired_median is None or paired_n < MIN_AXIS_SAMPLES:
        return None
    if paired_median > floor_m:
        return 'raw'
    if paired_median < -floor_m:
        return 'sfm'
    return 'undecidable'


def summarize_paired(diffs):
    """{n, median_m, raw_closer_share} of a list of cross_sfm - cross_raw differences."""
    if not diffs:
        return {'n': 0, 'median_m': None, 'raw_closer_share': None}
    return {'n': len(diffs), 'median_m': _median(diffs),
            'raw_closer_share': round(sum(d > 0 for d in diffs) / len(diffs), 3)}


# --------------------------------------------------------------------------- core check
def check_run(manifest, area, records, osm_payload, threshold_m=GROSS_OFF_STREET_M,
              min_sequence=DEFAULT_MIN_SEQUENCE, floor_m=RESOLUTION_FLOOR_M,
              max_iqr_ratio=MAX_IQR_RATIO):
    """The whole measurement, as one JSON-serializable dict (see position_check.json).

    The verdict per sequence (issue #62), always on the field it actually submitted:
      off_street  its median unsigned cross-track distance exceeds `threshold_m`
                  (GROSS_OFF_STREET_M), or most of it is beyond MAX_SNAP_M of any street
                  while the other field is not (beyond_snap)
      fixable     the other field is closer pano by pano by more than `floor_m` (paired
                  metric) and no more than `max_iqr_ratio` times as scattered; under
                  beyond_snap, the other field lands within MAX_SNAP_M - floor_m
      flagged     off_street and fixable: the only thing that gates a submission
      both_off    off_street and not fixable: reported, never gated
      undecidable the paired difference is inside the floor: reported with the caveat
                  that this reference cannot adjudicate it, whichever sign it has
    The signed per-axis bias from #60 stays in every row (`bias_m`) as the off-street
    report a human eyeballs; it no longer decides anything.
    """
    source = manifest.get('imagery_source', 'gsv')
    min_lng, min_lat, max_lng, max_lat = area_bbox(area)
    frame = geo.LocalFrame((min_lat + max_lat) / 2, (min_lng + max_lng) / 2)
    index = StreetIndex(street_segments(osm_payload, frame))
    labels = field_labels(source)

    per_field_rows = defaultdict(list)   # field -> [measure_point(...)]
    per_seq = defaultdict(lambda: {'n': 0, 'shift': [], 'cross': defaultdict(list),
                                   'a': defaultdict(list), 'b': defaultdict(list),
                                   'votes': defaultdict(int), 'paired': []})
    unsnapped = 0
    votes = defaultdict(int)   # run-wide: which re-readable field equals the submitted one
    pooled_paired = []
    for rec in records:
        pano = rec['pano']
        positions = pano_positions(pano)
        enu = {f: frame.to_enu(*ll) for f, ll in positions.items()}
        measured = {f: measure_point(index, *enu[f]) for f in enu}
        if measured['submitted'] is None:
            unsnapped += 1
        for f, m in measured.items():
            per_field_rows[f].append(m)
        seq = pano.get('sequence_id')
        if seq and 'sfm' in enu and 'raw' in enu:
            s = per_seq[seq]
            s['n'] += 1
            s['shift'].append((enu['sfm'][0] - enu['raw'][0], enu['sfm'][1] - enu['raw'][1]))
            for f in MAPILLARY_FIELDS:
                if positions[f] == positions['submitted']:
                    votes[f] += 1
                    s['votes'][f] += 1
            for f, m in measured.items():
                if m is not None:
                    s['cross'][f].append(m['cross'])
                    for axis in ('a', 'b'):
                        if m[axis] is not None:
                            s[axis][f].append(m[axis])
            # The paired metric compares the two fields on the SAME pano against the SAME
            # street; a pano whose fields snap to different streets (near an intersection,
            # or one of them drifted onto a parallel street) says nothing about either.
            ms, mr = measured.get('sfm'), measured.get('raw')
            if ms and mr and ms['street_key'] == mr['street_key']:
                diff = ms['cross'] - mr['cross']
                s['paired'].append(diff)
                pooled_paired.append(diff)

    fields = {f: {'label': labels.get(f, f), **summarize_offsets(rows)} for f, rows in per_field_rows.items()}

    # Which field did the run submit? Compared coordinate by coordinate, not read from the
    # manifest, so old runs work - and PER SEQUENCE, because a reposition.py output is mixed
    # by design (flagged sequences on one field, the rest on the other) and the check has
    # to be able to confirm its own fix. The run-wide majority is kept for the report.
    submitted_field = max(votes, key=votes.get) if votes else None

    all_fields = ('submitted', *MAPILLARY_FIELDS)
    sequences = []
    flagged = []
    both_off = []
    for seq, s in per_seq.items():
        de = _median([v[0] for v in s['shift']])
        dn = _median([v[1] for v in s['shift']])
        mags = sorted(math.hypot(*v) for v in s['shift'])
        cross_med = {f: _median(s['cross'][f]) for f in all_fields}
        cross_iqr = {f: _iqr(s['cross'][f]) for f in all_fields}
        # Signed bias per street family: the #60 statistic, kept as a report. A lane
        # offset cancels over a sequence driven both ways and a block shift does not,
        # which makes it a good picture of drift and a poor judge of it (Richmond's
        # jKtaJMek7wQl5AOH28qdcm read as "raw fixes it" by bias while raw was 3.3 m
        # worse per pano).
        bias = {}
        for f in all_fields:
            across_a = _median(s['a'][f]) if len(s['a'][f]) >= MIN_AXIS_SAMPLES else None
            across_b = _median(s['b'][f]) if len(s['b'][f]) >= MIN_AXIS_SAMPLES else None
            worst = max((abs(v) for v in (across_a, across_b) if v is not None), default=None)
            bias[f] = {'a': across_a, 'b': across_b, 'max_abs': worst}
        # Panos beyond MAX_SNAP_M of any street measure nothing, so a sequence that
        # drifted 40 m would otherwise have no cross-track at all and read as clean - the
        # worst failure passing while an 8 m one flags. Count them per field instead.
        unsnapped_by_field = {f: s['n'] - len(s['cross'][f]) for f in all_fields}
        chosen = (max(s['votes'], key=s['votes'].get) if s['votes'] else None) or submitted_field or 'sfm'
        alt = next(f for f in MAPILLARY_FIELDS if f != chosen)

        paired = summarize_paired(s['paired'])
        verdict = paired_verdict(paired['median_m'], paired['n'], floor_m)
        # How much closer the alternative is, pano by pano (positive = alt closer).
        gain = None
        if verdict is not None:
            gain = paired['median_m'] if alt == 'raw' else -paired['median_m']

        enough = s['n'] >= min_sequence
        sub_cross = cross_med['submitted'] if len(s['cross']['submitted']) >= MIN_AXIS_SAMPLES else None
        off_gross = enough and sub_cross is not None and sub_cross > threshold_m
        beyond_snap = enough and (unsnapped_by_field['submitted'] - unsnapped_by_field[alt]
                                  >= max(MIN_AXIS_SAMPLES, s['n'] // 2))
        off = off_gross or beyond_snap
        sub_iqr, alt_iqr = cross_iqr['submitted'], cross_iqr[alt]
        if beyond_snap:
            # Most of the submitted positions measure nothing, so there is no paired
            # difference to read; an alternative that lands back within reach of the
            # street, by more than the floor, is the fix.
            fixable = cross_med[alt] is not None and cross_med[alt] <= MAX_SNAP_M - floor_m
        else:
            fixable = (off_gross and gain is not None and gain > floor_m
                       and sub_iqr is not None and alt_iqr is not None
                       and alt_iqr <= max_iqr_ratio * sub_iqr)
        row = {
            'sequence_id': seq, 'n': s['n'],
            'submitted_field': chosen if s['votes'] else None,
            'sfm_minus_raw_east_m': de, 'sfm_minus_raw_north_m': dn,
            'shift_p90_m': round(mags[int(len(mags) * 0.9)], 2) if mags else None,
            'bias_m': bias,
            'cross_track_median_m': cross_med,
            'cross_track_iqr_m': cross_iqr,
            'paired_median_m': paired['median_m'],     # median(cross_sfm - cross_raw); + = raw closer
            'paired_n': paired['n'],
            'raw_closer_share': paired['raw_closer_share'],
            'paired_verdict': verdict,                 # 'raw' | 'sfm' | 'undecidable' | None
            'undecidable': verdict == 'undecidable',
            'not_near_a_street': unsnapped_by_field,
            'recommended': alt if fixable else (chosen if sub_cross is not None else None),
            'off_street': bool(off),
            'flagged': bool(fixable),
            'both_off': bool(off and not fixable),
            'beyond_snap': bool(beyond_snap),
        }
        sequences.append(row)
        if fixable:
            flagged.append(seq)
        elif off:
            both_off.append(seq)
    sequences.sort(key=lambda r: -r['n'])

    drift = None
    if per_seq:
        all_mags = [math.hypot(*v) for s in per_seq.values() for v in s['shift']]
        drift = {'n': len(all_mags), 'median_m': _median(all_mags),
                 'share_over_5m': round(sum(m > 5 for m in all_mags) / len(all_mags), 3),
                 'share_over_10m': round(sum(m > 10 for m in all_mags) / len(all_mags), 3)}

    paired_run = None
    if per_seq:
        judged = [r for r in sequences if r['n'] >= min_sequence and r['paired_verdict']]
        n_undecidable = sum(r['paired_verdict'] == 'undecidable' for r in judged)
        paired_run = {**summarize_paired(pooled_paired),
                      'sequences_judged': len(judged),
                      'raw_better': sum(r['paired_verdict'] == 'raw' for r in judged),
                      'sfm_better': sum(r['paired_verdict'] == 'sfm' for r in judged),
                      'undecidable': n_undecidable,
                      'raw_closer_by_sign': sum(r['paired_median_m'] > 0 for r in judged),
                      'share_below_floor': round(n_undecidable / len(judged), 3) if judged else None}

    return {
        'run_name': manifest.get('run_name'),
        'imagery_source': source,
        'checked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'rule': RULE,
        'threshold_m': threshold_m,          # GROSS_OFF_STREET_M: the off-street cross-track bar
        'resolution_floor_m': floor_m,
        'max_iqr_ratio': max_iqr_ratio,
        'min_sequence': min_sequence,
        'max_snap_m': MAX_SNAP_M,
        'panos': len(records),
        'panos_not_near_a_street': unsnapped,
        'submitted_field': submitted_field,
        'axes': index.axes(),
        'fields': fields,
        'sfm_vs_raw_drift': drift,
        'paired': paired_run,
        'sequences': sequences,
        'flagged_sequences': flagged,
        'both_off_sequences': both_off,
        'frame': {'lat0': frame.lat0, 'lng0': frame.lng0},
    }


# --------------------------------------------------------------------------- report data
def operational_points(records, frame):
    """Every operational detection raycast to the ground (flat, no pose), in frame metres,
    with the pano's per-field position vectors so the report can move them."""
    out = []
    for rec in records:
        pano = rec['pano']
        try:
            pose = geo.pano_pose(pano)
        except (KeyError, TypeError, ValueError):
            continue
        positions = pano_positions(pano)
        base = frame.to_enu(pano['lat'], pano['lng'])
        vectors = {f: (frame.to_enu(*ll)[0] - base[0], frame.to_enu(*ll)[1] - base[1]) for f, ll in positions.items()}
        for i, det in enumerate(rec.get('detections') or []):
            if det.get('confidence', 0) < OPERATIONAL_CONFIDENCE:
                continue
            if on_camera_rig(det['y_normalized']):
                # Report-only, but it is the near field that this plot exists to make
                # legible, and a rig detection raycasts to ~1.5 m from the camera - a dense
                # ghost cluster right where the reader is trying to judge a 3 m offset.
                continue
            est = geo.detection_ground_point(pose, det['x_normalized'], det['y_normalized'], apply_pose=False)
            if est is None:
                continue
            e, n = frame.to_enu(est.lat, est.lng)
            out.append({'id': f"{pano['panorama_id']}#{i}", 'pano': pano['panorama_id'],
                        'seq': pano.get('sequence_id'), 'pos': (e, n), 'vectors': vectors})
    return out


def ps_label_points(feed_path, records, frame):
    """Points from a PS v3 rawLabels GeoJSON, joined to the run's panos by pano_id."""
    with open(feed_path, encoding='utf-8') as f:
        feed = json.load(f)
    by_id = {rec['pano']['panorama_id']: rec['pano'] for rec in records}
    out, missing = [], 0
    for feat in feed.get('features', []):
        props = feat.get('properties') or {}
        pano = by_id.get(str(props.get('pano_id')))
        if pano is None:
            missing += 1
            continue
        lng, lat = feat['geometry']['coordinates']
        positions = pano_positions(pano)
        base = frame.to_enu(pano['lat'], pano['lng'])
        vectors = {f: (frame.to_enu(*ll)[0] - base[0], frame.to_enu(*ll)[1] - base[1]) for f, ll in positions.items()}
        out.append({'id': str(props.get('label_id')), 'pano': pano['panorama_id'],
                    'seq': pano.get('sequence_id'), 'pos': frame.to_enu(lat, lng), 'vectors': vectors})
    if missing:
        print(f'   {missing} labels in the feed belong to panos not in this run (ignored)')
    return out


def densest_window(points, width_m=800.0, height_m=400.0, cell_m=100.0):
    """The width x height box (frame metres) holding the most points, on a cell grid."""
    if not points:
        return (-width_m / 2, -height_m / 2, width_m / 2, height_m / 2)
    counts = defaultdict(int)
    for e, n in points:
        counts[(int(e // cell_m), int(n // cell_m))] += 1
    cw, ch = int(width_m // cell_m), int(height_m // cell_m)
    # Every window that contains a populated cell is anchored within (cw, ch) cells
    # south-west of it, so those anchors are the whole candidate set — including ones
    # whose own cell is empty, which anchoring on populated cells alone would miss.
    anchors = {(cx - i, cy - j) for (cx, cy) in counts for i in range(cw) for j in range(ch)}
    best, best_cell = -1, (0, 0)
    for (cx, cy) in sorted(anchors):
        total = sum(counts.get((cx + i, cy + j), 0) for i in range(cw) for j in range(ch))
        if total > best:
            best, best_cell = total, (cx, cy)
    x0, y0 = best_cell[0] * cell_m, best_cell[1] * cell_m
    return (x0, y0, x0 + width_m, y0 + height_m)


def _in_window(p, w, pad=20.0):
    return w[0] - pad <= p[0] <= w[2] + pad and w[1] - pad <= p[1] <= w[3] + pad


def _segment_hits_window(a, b, w, pad=20.0):
    """Does segment a-b intersect the padded window? Liang-Barsky clip, so a long straight
    that crosses the map with both endpoints outside it is still drawn."""
    x0, y0, x1, y1 = w[0] - pad, w[1] - pad, w[2] + pad, w[3] + pad
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def build_report_data(result, manifest, area, records, osm_payload, labels_path=None,
                      reference_dir=None, window=None):
    frame = geo.LocalFrame(result['frame']['lat0'], result['frame']['lng0'])
    source = result['imagery_source']
    r = lambda v: round(v, 1)  # noqa: E731

    if labels_path:
        points = ps_label_points(labels_path, records, frame)
        points_kind = 'ps_labels'
    else:
        points = operational_points(records, frame)
        points_kind = 'raycast'

    pano_pts = []
    for rec in records:
        pano = rec['pano']
        pos = {f: frame.to_enu(*ll) for f, ll in pano_positions(pano).items()}
        pano_pts.append({'id': pano['panorama_id'], 'seq': pano.get('sequence_id'),
                         'hd': round(float(pano.get('camera_heading') or 0), 1), 'pos': pos})

    if window:
        lng0, lat0, lng1, lat1 = window
        w = (*frame.to_enu(lat0, lng0), *frame.to_enu(lat1, lng1))
    else:
        w = densest_window([p['pos'] for p in points] or [p['pos']['submitted'] for p in pano_pts])

    seg_rows = []
    index = StreetIndex(street_segments(osm_payload, frame))
    for i, (ax, ay, bx, by, name, _key) in enumerate(index.segments):
        if _segment_hits_window((ax, ay), (bx, by), w, 60):
            seg_rows.append({'a': [r(ax), r(ay)], 'b': [r(bx), r(by)], 'axis': index.axis_of(i), 'name': name})

    fields_in_points = list(field_labels(source).keys())
    points_rows = []
    for p in points:
        if not _in_window(p['pos'], w):
            continue
        row = {'id': p['id'], 'pano': p['pano'], 'seq': p['seq'], 'pos': {}}
        for f in fields_in_points:
            v = p['vectors'].get(f)
            if v is not None:
                row['pos'][f] = [r(p['pos'][0] + v[0]), r(p['pos'][1] + v[1])]
        points_rows.append(row)

    pano_rows = [{'id': p['id'], 'seq': p['seq'], 'hd': p['hd'],
                  'pos': {f: [r(e), r(n)] for f, (e, n) in p['pos'].items()}}
                 for p in pano_pts if _in_window(p['pos']['submitted'], w)]

    # Per-street table inside the window, for the submitted field vs. the others.
    streets = defaultdict(lambda: defaultdict(list))
    for p in points_rows:
        for f, pos in p['pos'].items():
            m = measure_point(index, *pos)
            if m and m['a'] is not None:
                streets[m['street'] or '(unnamed)'][f].append(m['a'])

    reference = None
    if reference_dir:
        ref_manifest, _ref_area, ref_records = load_run(reference_dir)
        ref_panos = [frame.to_enu(rec['pano']['lat'], rec['pano']['lng']) for rec in ref_records]
        sites_path = Path(reference_dir) / 'sites.jsonl'
        if sites_path.exists():
            ref_sites = []
            with open(sites_path, encoding='utf-8') as f:
                for line in f:
                    s = json.loads(line)
                    if s.get('n_operational', 0) > 0:
                        ref_sites.append(frame.to_enu(s['lat'], s['lng']))
            ref_kind = 'fused sites'
        else:
            ref_sites = [p['pos'] for p in operational_points(ref_records, frame)]
            ref_kind = 'raycast detections'
        ref_rows = [measure_point(index, *p) for p in ref_sites]
        ref_streets = defaultdict(list)
        for m in ref_rows:
            if m and m['a'] is not None:
                ref_streets[m['street'] or '(unnamed)'].append(m['a'])
        reference = {
            'run_name': ref_manifest.get('run_name'), 'source': ref_manifest.get('imagery_source', 'gsv'),
            'kind': ref_kind,
            'panos': [[r(e), r(n)] for e, n in ref_panos if _in_window((e, n), w)],
            'sites': [[r(e), r(n)] for e, n in ref_sites if _in_window((e, n), w)],
            'pano_offsets': summarize_offsets([measure_point(index, *p) for p in ref_panos]),
            'site_offsets': summarize_offsets(ref_rows),
            'streets': {k: v for k, v in ref_streets.items()},
        }

    all_points_rows = {f: [] for f in fields_in_points}
    for p in points:
        for f in fields_in_points:
            v = p['vectors'].get(f)
            if v is not None:
                all_points_rows[f].append(measure_point(index, p['pos'][0] + v[0], p['pos'][1] + v[1]))
    point_offsets = {f: summarize_offsets(rows) for f, rows in all_points_rows.items() if rows}

    def street_summary(vals):
        return {'n': len(vals), 'median': _median(vals),
                'neg_share': round(sum(v < 0 for v in vals) / len(vals), 2)} if vals else None

    return {
        'check': result,
        'axes': result['axes'],
        'points_kind': points_kind,
        'points_total': len(points),
        'point_offsets': point_offsets,
        'window': {'e0': r(w[0]), 'n0': r(w[1]), 'e1': r(w[2]), 'n1': r(w[3]),
                   'sw': frame.to_latlng(w[0], w[1]), 'ne': frame.to_latlng(w[2], w[3])},
        'segments': seg_rows,
        'points': points_rows,
        'panos': pano_rows,
        'streets': sorted(
            [{'name': k, **{f: street_summary(v.get(f, [])) for f in fields_in_points},
              'reference': street_summary(reference['streets'].get(k, [])) if reference else None}
             for k, v in streets.items()],
            key=lambda row: -(row['submitted'] or {'n': 0})['n']),
        'reference': reference,
        'field_labels': field_labels(source),
    }


def write_report(data, out_path):
    template_path = Path(__file__).with_name('position_report_template.html')
    template = template_path.read_text(encoding='utf-8')
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    out_path.write_text(template.replace('/*__DATA__*/', payload), encoding='utf-8')


# --------------------------------------------------------------------------- CLI
def _s(v):
    return '-' if v is None else (f'{v:.1f}' if isinstance(v, float) else str(v))


def print_summary(result):
    print(f"\n--- Position check: {result['run_name']} ({result['imagery_source']}) ---")
    print(f"panos {result['panos']}, not within {MAX_SNAP_M:.0f} m of any street: "
          f"{result['panos_not_near_a_street']}")
    if result['submitted_field']:
        print(f"submitted position field: {result['submitted_field']}")
    ax = result['axes']
    print(f"street grid bearing {ax['theta0_deg']} deg; axis a = across {ax['a']['streets']} (+{ax['a']['pos']}), "
          f"axis b = across {ax['b']['streets']} (+{ax['b']['pos']})")
    print(f"{'field':40s} {'a median':>9s} {'a IQR':>14s} {'a neg':>6s} {'b median':>9s} {'cross-track':>11s}")
    for f, s in result['fields'].items():
        a, b, ct = s['across_a'], s['across_b'], s['cross_track']
        iqr = f"{_s(a.get('q25'))} .. {_s(a.get('q75'))}" if a.get('n') else '-'
        print(f"{s['label']:40s} {_s(a.get('median')):>9s} {iqr:>14s} {_s(a.get('neg_share')):>6s} "
              f"{_s(b.get('median')):>9s} {_s(ct.get('median')):>11s}")
    floor = result['resolution_floor_m']
    d = result['sfm_vs_raw_drift']
    if d:
        print(f"SfM moves a pano {d['median_m']} m from its GPS fix (median); "
              f"{d['share_over_5m']:.0%} > 5 m, {d['share_over_10m']:.0%} > 10 m")
        p = result.get('paired') or {}
        if p.get('n'):
            print(f"paired per pano (cross_sfm - cross_raw, same street): median {p['median_m']} m over "
                  f"{p['n']} panos, raw closer on {p['raw_closer_share']:.0%}; of {p['sequences_judged']} "
                  f"sequences: raw better {p['raw_better']}, SfM better {p['sfm_better']}, "
                  f"undecidable (|paired| <= {floor} m) {p['undecidable']}")
        print(f"\n{'sequence':26s} {'n':>5s} {'cross sfm':>9s} {'IQR':>5s} {'cross raw':>9s} {'IQR':>5s} "
              f"{'paired':>7s} {'raw<':>5s} {'bias sfm':>8s} {'bias raw':>8s} {'rec':>4s}  verdict")
        for s in result['sequences']:
            if s['n'] < result['min_sequence']:
                continue
            b, c, q = s['bias_m'], s['cross_track_median_m'], s['cross_track_iqr_m']
            share = s['raw_closer_share']
            verdict = ('FLAG' if s['flagged'] else 'both off' if s['both_off']
                       else 'undecidable' if s['undecidable'] else '')
            print(f"{s['sequence_id']:26s} {s['n']:5d} {_s(c['sfm']):>9s} {_s(q['sfm']):>5s} "
                  f"{_s(c['raw']):>9s} {_s(q['raw']):>5s} {_s(s['paired_median_m']):>7s} "
                  f"{('-' if share is None else f'{share:.0%}'):>5s} {_s(b['sfm']['max_abs']):>8s} "
                  f"{_s(b['raw']['max_abs']):>8s} {_s(s['recommended']):>4s}  {verdict}"
                  f"{' (beyond snap)' if s.get('beyond_snap') else ''}")
        print(f"(cross = median unsigned distance to the nearest street, IQR its spread; paired = median of "
              f"cross_sfm - cross_raw per pano on the same street, + = raw closer; raw< = share of panos raw "
              f"is closer on; bias = the #60 signed offset, a report only; 'beyond snap' = most of the "
              f"sequence is > {MAX_SNAP_M:.0f} m from any street on the submitted field)")
    n_flag, n_both = len(result['flagged_sequences']), len(result['both_off_sequences'])
    if n_both:
        print(f"\n{n_both} sequence(s) sit more than {result['threshold_m']} m from the street on the submitted "
              f"field where the other field is not closer by more than {floor} m (or is more scattered): off "
              f"in both fields (wide one-way streets driven once, or OSM geometry) - reported, not gated")
    undecidable = [s for s in result['sequences'] if s['undecidable'] and s['n'] >= result['min_sequence']]
    if undecidable:
        print(f"{len(undecidable)} sequence(s) differ between the fields by {floor} m or less per pano: the OSM "
              f"reference cannot adjudicate that (GSV itself measures ~1.75 m against it), so no field "
              f"choice is implied")
    live = result.get('live_campaigns') or []
    if live:
        where = '; '.join(f"{c['endpoint']} with {c['labels_submitted']} labels on {c['submitted_lines']} "
                          f"lines" for c in live)
        print(f"\nThis file is already submitted: {where}. {n_flag} sequence(s) would improve by switching "
              f"field, but repositioning moves panos that carry live labels - a whole-city decision "
              f"(scripts/reposition.py refuses without --reposition-live-city).")
    if n_flag:
        results = result.get('results_path') or f"runs/{result['run_name']}/results.jsonl"
        print(f"\n!! {n_flag} sequence(s) sit more than {result['threshold_m']} m from the street on the "
              f"submitted field and the other field is closer pano by pano by more than {floor} m. "
              f"Run: python scripts/reposition.py {results} --from-check")
    else:
        print('\nOK: no sequence is grossly off the street on the submitted field where the other field would fix it')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='runs/<name> with manifest.json, area.geojson, results.jsonl')
    ap.add_argument('--threshold', type=float, default=GROSS_OFF_STREET_M, metavar='M',
                    help='a sequence is off the street when the median unsigned distance of its submitted '
                         'positions from the nearest street exceeds this (default %(default)s; the metric '
                         'floors near 1.75 m on a well-positioned source, so do not go near that)')
    ap.add_argument('--min-sequence', type=int, default=DEFAULT_MIN_SEQUENCE, metavar='N',
                    help=f'only sequences with at least N panos can be flagged (default %(default)s; '
                         f'a verdict needs {MIN_AXIS_SAMPLES} measured panos, so values below '
                         f'{MIN_AXIS_SAMPLES} are raised to it)')
    ap.add_argument('--results', metavar='FILE',
                    help='check this results file instead of runs/<name>/results.jsonl (e.g. a '
                         'scripts/reposition.py output) against the same area and streets; '
                         'position_check.json / the report are written beside it')
    ap.add_argument('--osm', metavar='FILE', help='use this Overpass JSON instead of runs/<name>/osm_streets.json')
    ap.add_argument('--report', action='store_true',
                    help='also write position_report.html beside position_check.json (i.e. beside the results file)')
    ap.add_argument('--labels', metavar='GEOJSON',
                    help="report: place the server's own labels (v3 rawLabels feed) instead of raycast detections")
    ap.add_argument('--reference', metavar='RUN_DIR',
                    help='report: another run over the same area (e.g. a GSV run) drawn as an independent layer')
    ap.add_argument('--window', metavar='LNG0,LAT0,LNG1,LAT1',
                    help='report: map window (default: the densest 800 x 400 m box)')
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(errors='replace')

    run_dir = Path(args.run_dir)
    results_path = Path(args.results) if args.results else run_dir / 'results.jsonl'
    if args.results and results_path.name == 'results.jsonl' and results_path.resolve() != (run_dir / 'results.jsonl').resolve():
        sys.exit(f'{results_path} is another run\'s results.jsonl: check that run directly '
                 f'(python scripts/position_check.py {results_path.parent.as_posix()}), or it would be scored '
                 f'against {run_dir}\'s area and streets and overwrite that run\'s position_check.json')
    window = tuple(float(v) for v in args.window.split(',')) if args.window else None
    result = run_check(run_dir, results_path, threshold_m=args.threshold, min_sequence=args.min_sequence,
                       osm_path=args.osm, report=args.report, labels_path=args.labels,
                       reference_dir=args.reference, window=window)
    return 1 if result['flagged_sequences'] else 0


def run_check(run_dir, results_path=None, threshold_m=GROSS_OFF_STREET_M, min_sequence=DEFAULT_MIN_SEQUENCE,
              osm_path=None, report=True, labels_path=None, reference_dir=None, window=None):
    """The whole check as one call — what main.py runs at the end of every run and what
    the CLI wraps: load the run (or `results_path` against its area/streets), score it,
    write position_check.json (+ the report) beside the results file, print the summary,
    and return the result dict. Raises on a missing run file or an Overpass failure;
    main.py catches and records that rather than failing a run whose detections are
    already on disk."""
    run_dir = Path(run_dir)
    results_path = Path(results_path) if results_path else run_dir / 'results.jsonl'
    if min_sequence < MIN_AXIS_SAMPLES:
        print(f"-> --min-sequence {min_sequence} raised to {MIN_AXIS_SAMPLES}: a verdict needs that many "
              f"measured panos, so nothing smaller can be flagged")
        min_sequence = MIN_AXIS_SAMPLES
    manifest, area, records = load_run(run_dir, results_path)
    osm_payload, osm_file, cached = load_or_fetch_streets(run_dir, area, osm_path)
    print(f"-> streets: {count_ways(osm_payload)} OSM ways from {osm_file}{' (cached)' if cached else ' (fetched)'}")
    print(f"-> panos: {len(records)} from {results_path}")

    result = check_run(manifest, area, records, osm_payload, threshold_m, min_sequence)
    result['results_path'] = repo_relative(results_path)
    # send_to_ps.py's gate compares this with the file it is about to submit, so a check
    # cannot vouch for a results file that changed after it ran.
    result['results_sha256'] = file_sha256(results_path)
    # What is already live from this very file, read from its submission record: the
    # report's "N sequences would improve; this file is already submitted to X with M
    # labels" sentence (issue #62). As of checked_at; the record is the authority.
    try:
        result['live_campaigns'] = live_campaigns(results_path)
    except ValueError as exc:   # report it; the check itself is still valid
        result['live_campaigns'], result['live_campaigns_error'] = [], str(exc)
        print(f'!! {exc}')
    # Outputs sit beside the results file they describe: a --results check of a
    # reposition.py output must not overwrite the run's own verdict.
    out_json = check_path_for(results_path)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)
    print_summary(result)
    print(f"-> wrote {out_json}")

    if report:
        data = build_report_data(result, manifest, area, records, osm_payload,
                                 labels_path=labels_path, reference_dir=reference_dir, window=window)
        out_html = report_path_for(results_path)
        write_report(data, out_html)
        print(f"-> wrote {out_html} ({out_html.stat().st_size // 1024} KB, self-contained)")
    return result


if __name__ == '__main__':
    sys.exit(main())
