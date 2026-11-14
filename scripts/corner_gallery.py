"""Corner present / absent rating gallery for the Vancouver corner inventory (RampNet#243).

The position-selected run (RampNet#241, amendment A8 of docs/corner-inventory.md) left two
questions that only a person looking at the imagery can answer:

  1. what do the city's `NA` inventory points with no `RAMPTYPE` mean? 944 of the 990 absent
     units that are not clean hold only such points;
  2. are the 35 false absences at the #241 target units real detector misses, or inventory /
     geometry artifacts?

This tool builds one rating bundle of three parts, drawn with a recorded seed:

  false_absence  every unit called absent (fusion arm, primary observability) at a #241 target
                 unit that holds an `Available` inventory point (`false_absences.csv`): 35
  na_noramp      a seeded random sample of the absent units whose ONLY inventory points are
                 `NA` with no `RAMPTYPE` (944 in the merged build): 40
  clean          a seeded random sample of the absent units with no inventory point at all
                 (333): 20, the control

and renders a review page where each unit is rated **per corner**: present / absent / can't
tell. The page is a sibling of RampNet's #224 cluster-review gallery
(`scripts/cluster_review_gallery.py` on RampNet branch cluster-review-224) and reuses its
mechanics: one unit per screen, a north-up aerial with the 30 m window, crops cut on makelab2
by the same crop worker (`site_explorer.CROP_WORKER`, via `export_cluster_review`'s aerial
and crop helpers), state autosaved in the browser keyed by rater + item-list sha256, the
same prefill rule (local work wins only on units worked in this browser), the same timing
rule (1 s ticks while visible and active within 60 s), review notes, and Export to a
per-rater file. The city inventory is shown only after a unit is completed, and the verdicts
given before that reveal are frozen as `blind`; the scorer reads `blind`.

A second mode builds the pre-registered precision gallery of issue #169 from a seeded sample
of the rule's OWN proposals (scripts/corner_labels.py `sample`), not from these three parts:

  build --items-from <sample.csv>   one unit per sampled intersection, showing ONLY its
          sampled corners; the #243 population checks are bypassed; snapshot.json records the
          sample's, sample.json's, the proposals' and their build.json's sha256. Same views,
          crops, aerials, rubric, blind freeze and Export. The reveal is unchanged (the city
          inventory); the rule's proposal (chosen pano and pixel) is kept in items.jsonl and
          is not shown. Scored by `corner_labels.py score --sample --bundle`, not by
          corner_gallery_score.py.

Subcommands:

  aerial-dates  look up each aerial's capture date in Esri's imagery metadata -> <bundle>/aerial_dates.json
  build   sample, write <bundle>/items.jsonl + snapshot.json + report.md, make the aerials
          (Esri tiles, cached) and the crops (makelab2 PS store, or --local-panos)
  render  write <bundle>/gallery/<rater>/index.html; Export downloads verdicts__<rater>.json

Inputs are read in place and never written. The merged build's corners_full.jsonl is
untracked (tens of MB); its sha256 is recorded in snapshot.json, and the population sizes
are checked against the committed numbers of amendment A8 (35 / 944 / 333).

    python scripts/corner_gallery.py build \\
        --corners-full ../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl \\
        --bundle runs/vancouver/corner_gallery243
    python scripts/corner_gallery.py render --bundle runs/vancouver/corner_gallery243 --rater jonf
    # open runs/vancouver/corner_gallery243/gallery/jonf/index.html, rate, Export, and save the
    # download as runs/vancouver/corner_gallery243/verdicts__jonf.json
    python scripts/corner_gallery_score.py runs/vancouver/corner_gallery243

    # #169's precision gallery over the pilot build's sample
    python scripts/corner_gallery.py build --items-from \
        runs/vancouver/corner_labels169/na_discount_sw/sample.csv \
        --corners-full ../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl \
        --bundle runs/vancouver/corner_gallery169
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402

ITEMS_SCHEMA = 'sidewalk-auto-labeler.corner_gallery.items/1'
SNAPSHOT_SCHEMA = 'sidewalk-auto-labeler.corner_gallery.snapshot/1'
VERDICTS_SCHEMA = 'sidewalk-auto-labeler.corner_gallery/1'
CITY = 'vancouver'
SEED = 243
N_NA = 40
N_CLEAN = 20
PARTS = ('false_absence', 'na_noramp', 'clean')
ARM_STATE = 'fusion/primary'           # the decision rule's arm and observability variant
INTERSECTIONS = ('signalised', 'arterial', 'residential')
# Population sizes stated in amendment A8 (docs/corner-inventory.md, "Results"); the build
# refuses an input that does not reproduce them.
EXPECTED_POPULATION = {'false_absence': 35, 'na_noramp': 944, 'clean': 333}
VIEWS_PER_CORNER = 3
MIN_VIEW_M = 3.0                       # a pano nearer than this looks almost straight down
MAX_VIEW_M = 40.0
CAMERA_HEIGHT_M = 2.5                  # the fixed height of the #241 fusion (amendment A8)
CROP_FOV_DEG = 60                      # square crops: fov_deg x fov_deg, CROP_PX x CROP_PX
CROP_PX = 448
CROP_ASPECT = 1.0                      # width:height; > 1 cuts landscape crops (#169, 2026-11-06)
WINDOW_M = 30.0
STORE = '/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa'
SAMPLE_PART = 'rule_sample'            # the one part of a --items-from bundle (#169)
CROPS_NEEDED = 'CROPS_NEEDED.md'
CROP_JOB_FILE = 'crop_job.json'

#: the rubric; travels inside every verdicts file (see RUBRIC_VERSION). A bundle keeps the
#: version it was built with (snapshot.json `rubric_version` / `rubric`): gallery 243 and the
#: #169 pilot bundle are v1, the #169 split-wide bundle is v2. render shows the bundle's text.
RUBRIC_V1 = """\
Rubric v1 (RampNet#243). Rate every corner of the unit from the crops and the aerial.

The unit is an OSM intersection. Its legs (white lines on the aerial) split it into corners,
numbered clockwise from north. A corner is the sector between two adjacent legs; its marker
on the aerial and the orange ring in each crop is the corner point, 12 m from the centre
along the sector's bisector. A ramp can sit several metres from that point; judge the whole
corner of the sector (the curb return and the crossings that land on it), not the point.

present   at least one curb ramp serving pedestrians at this corner is visible: a cut in the
          curb with a sloped walking surface down to street level, of any type (perpendicular,
          parallel, diagonal at the apex, blended / depressed corner). A diagonal ramp counts
          for the corner whose sector holds it.
absent    the corner is visible well enough to judge and has no curb ramp. Say which, when
          you can: `curb_no_ramp` (a sidewalk reaches the corner and the curb there has no
          cut), or `no_sidewalk` (no sidewalk reaches the corner, so there is nothing to ramp
          from: a lawn, gravel or a shoulder to the street edge, with or without a curb).
          A driveway apron is not a curb ramp.
cant_tell the corner cannot be judged: hidden by vehicles, vegetation or shadow; out of
          frame; too far or blurred; under construction; or the sector is not a real street
          corner (an OSM artifact, a parking lot). Say why in the unit note.

Use the newest crop that shows the corner clearly. If crops of different years disagree
(a ramp appears in a later year), rate what the newest clear crop shows and say so in the
note. The city inventory is shown only after the unit is completed; the verdicts given
before that are kept as `blind` and are what is scored. Changing a verdict after the
reveal is allowed, is recorded (`edited_after_inventory`), and is never scored as blind.
"""

RUBRIC_V2 = RUBRIC_V1.replace(
    'Rubric v1 (RampNet#243).', 'Rubric v2 (RampNet#243; #169 revision of 2026-11-06).') + """
Added in v2 (#169). DASHED white lines on the aerial are phantom legs: where a street ends at
a T, the stem's line continued straight across the through street. So the far side of a T has
TWO corners, one at each place a crossing of the through street lands (where one of the stem's
two sidewalks, carried straight across, meets the far curb). Each needs its own ramp: rate a
corner present only for a ramp serving its own landing, not one at the other landing. Solid
lines are real streets, as before. Crops may be landscape (wider than tall) and there may be
more of them per corner; the orange ring is still the corner point.
"""

RUBRICS = {1: RUBRIC_V1, 2: RUBRIC_V2}
RUBRIC_VERSION = 2                     # what a new bundle gets; see rubric_for()
RUBRIC = RUBRICS[RUBRIC_VERSION]


# ----------------------------------------------------------------------------- helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(text):
    """A unit key as a file-name stem (':' is not allowed on Windows)."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', text)


def rater_file_name(rater):
    """``verdicts__<rater>.json``; every rater has a named file (single rater today)."""
    if not rater or not re.fullmatch(r'[A-Za-z0-9_.-]+', rater):
        raise ValueError(f'rater name {rater!r}: letters, digits, "_", ".", "-" only')
    return f'verdicts__{rater}.json'


def write_text_lf(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def git_sha():
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return None


def git_dirty():
    """True when tracked files differ from HEAD (so tool_git_sha alone does not pin the code);
    None when git is unavailable."""
    import subprocess
    try:
        out = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return None


def portable_path(path):
    """A path relative to the repo root, '/'-separated, so snapshot.json does not record one
    machine's absolute layout; absolute only when it is on another drive.

    >>> portable_path(REPO_ROOT / 'runs' / 'x.json')
    'runs/x.json'
    """
    p = Path(path).resolve()
    try:
        return os.path.relpath(p, REPO_ROOT).replace(os.sep, '/')
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------------------- sampling

def only_na_noramp(counts):
    """True when every inventory point is `NA` with no `RAMPTYPE` (and there is one)."""
    return counts.get('NA_noramp', 0) > 0 and \
        sum(v for k, v in counts.items() if k != 'NA_noramp') == 0


def populations(records, targets):
    """{part: sorted unit keys}: the three populations, intersections only, among units the
    fusion arm calls absent under primary observability (the decision rule's read)."""
    out = {p: [] for p in PARTS}
    for r in records:
        if r['type'] not in INTERSECTIONS or r['state'][ARM_STATE] != 'absent':
            continue
        c = r['inv_counts']
        if c.get('Available', 0) > 0:
            if r['unit'] in targets:
                out['false_absence'].append(r['unit'])
        elif only_na_noramp(c):
            out['na_noramp'].append(r['unit'])
        elif sum(c.values()) == 0:
            out['clean'].append(r['unit'])
    return {p: sorted(v) for p, v in out.items()}


def draw(pops, seed=SEED, n_na=N_NA, n_clean=N_CLEAN):
    """{part: unit keys}. Every false absence; then, from ONE random.Random(seed), n_na of the
    na_noramp population and n_clean of the clean one, each drawn from its sorted list with
    rng.sample. Same inputs and seed -> the same lists."""
    rng = random.Random(seed)
    return {'false_absence': list(pops['false_absence']),
            'na_noramp': sorted(rng.sample(pops['na_noramp'], min(n_na, len(pops['na_noramp'])))),
            'clean': sorted(rng.sample(pops['clean'], min(n_clean, len(pops['clean']))))}


def review_order(units):
    """Units in sha1(unit key) order, so a unit's slot says nothing about its part."""
    return sorted(units, key=lambda u: hashlib.sha1(u.encode()).hexdigest())


# ---------------------------------------------------------------------------- geometry

def pano_view(cam_lat, cam_lng, heading_deg, lat, lng, camera_height_m=CAMERA_HEIGHT_M):
    """Where a ground point appears in an equirectangular pano.

    x follows the store's convention, bearing = heading + (x - 0.5) * 360 (fuse_sites,
    thinning_experiment); y puts a ground point at camera_height_m below a level camera,
    y = 0.5 + atan(h / d) / 180 (pitch and roll ignored, as the flat-pose fusion does).

    >>> v = pano_view(45.0, -122.0, 90.0, 45.0, -121.9999)   # ~7.9 m due east
    >>> round(v['x'], 3), round(v['dist_m'], 1), round(v['y'], 3)
    (0.5, 7.9, 0.598)
    """
    fr = geo.LocalFrame(cam_lat, cam_lng)
    e, n = fr.to_enu(lat, lng)
    d = math.hypot(e, n)
    bearing = math.degrees(math.atan2(e, n)) % 360.0
    x = ((bearing - heading_deg) / 360.0 + 0.5) % 1.0
    y = 0.5 + math.degrees(math.atan2(camera_height_m, max(d, 0.01))) / 180.0
    return {'dist_m': round(d, 2), 'bearing_deg': round(bearing, 2), 'x': round(x, 6),
            'y': round(min(y, 0.999), 6)}


def _distance_order(cands, min_m):
    """Nearest first among those >= min_m away, then the nearer ones, farthest of them first
    (a pano almost on top of the corner point looks straight down)."""
    far = sorted((c for c in cands if c['dist_m'] >= min_m), key=lambda c: (c['dist_m'], c['pano_id']))
    near = sorted((c for c in cands if c['dist_m'] < min_m), key=lambda c: (-c['dist_m'], c['pano_id']))
    return far + near


def newest_date(cands):
    """The newest capture_date among cands ('' when none is dated). Dates are 'YYYY-MM'
    strings, so string order is time order."""
    return max((c.get('capture_date') or '' for c in cands), default='')


def choose_views(cands, k=VIEWS_PER_CORNER, min_m=MIN_VIEW_M, max_m=MAX_VIEW_M):
    """The k views to show for a corner, capture-date aware (RampNet#243 review, B1).

    Among the candidates within max_m of the corner point, slot 1 goes to the newest capture
    date: of the panos carrying that date, the first in distance order (nearest >= min_m,
    else the farthest of the nearer ones). The other k - 1 slots are filled in distance order
    from the rest. So the newest imagery in the pool is always shown, and the rubric's "use
    the newest crop that shows the corner clearly" is always possible. The returned list is in
    distance order; a view's capture date is on its caption. cands: dicts with dist_m, pano_id
    and capture_date.

    >>> c = [{'pano_id': 'a', 'dist_m': 5, 'capture_date': '2014-08'},
    ...      {'pano_id': 'b', 'dist_m': 6, 'capture_date': '2014-08'},
    ...      {'pano_id': 'c', 'dist_m': 7, 'capture_date': '2014-08'},
    ...      {'pano_id': 'd', 'dist_m': 30, 'capture_date': '2021-12'}]
    >>> [v['pano_id'] for v in choose_views(c)]
    ['a', 'b', 'd']
    """
    ok = _distance_order([c for c in cands if c['dist_m'] <= max_m], min_m)
    if not ok:
        return []
    newest = newest_date(ok)
    first = next(c for c in ok if (c.get('capture_date') or '') == newest)
    rest = [c for c in ok if c is not first][:k - 1]
    chosen = {id(c) for c in rest} | {id(first)}
    return [c for c in ok if id(c) in chosen]


def crop_geometry(aspect=CROP_ASPECT, fov_deg=CROP_FOV_DEG, px=CROP_PX):
    """None for the square default (the job items are then exactly the pre-#169 ones), else
    {'aspect', 'fov_h_deg', 'fov_v_deg', 'px_w', 'px_h'}: fov_deg wide and fov_deg / aspect
    tall on the equirect, resized to px x px / aspect.

    >>> crop_geometry() is None
    True
    >>> crop_geometry(2, 120, 896)
    {'aspect': 2.0, 'fov_h_deg': 120.0, 'fov_v_deg': 60.0, 'px_w': 896, 'px_h': 448}
    """
    aspect, fov_deg, px = float(aspect), float(fov_deg), int(px)
    if aspect <= 0:
        raise SystemExit(f'--crop-aspect {aspect}: must be > 0')
    if aspect == 1.0:
        return None
    return {'aspect': aspect, 'fov_h_deg': fov_deg, 'fov_v_deg': round(fov_deg / aspect, 6),
            'px_w': px, 'px_h': int(round(px / aspect))}


def bundle_params(args, bundle):
    """(views_per_corner, crop geometry or None, rubric_version) for a build. A flag left
    unset takes the value the bundle's existing snapshot.json recorded, so re-running a build
    (e.g. to cut its crops) never silently changes the views or the rubric; with no snapshot
    it takes the module default."""
    snap_path = Path(bundle) / 'snapshot.json'
    snap = json.loads(snap_path.read_text(encoding='utf-8')) if snap_path.exists() else {}
    views = snap.get('views') or {}
    k = getattr(args, 'views_per_corner', None)
    if k is None:
        k = views.get('per_corner', VIEWS_PER_CORNER)
    if k < 1:
        raise SystemExit(f'--views-per-corner {k}: must be >= 1')
    aspect = getattr(args, 'crop_aspect', None)
    fov = getattr(args, 'crop_fov_deg', None)
    px = getattr(args, 'crop_px', None)
    if aspect is None and fov is None and px is None and views:
        geom = views.get('crop_geometry')
    else:
        dflt = views.get('crop_geometry') or {}
        geom = crop_geometry(aspect if aspect is not None else dflt.get('aspect', CROP_ASPECT),
                             fov if fov is not None else dflt.get('fov_h_deg', CROP_FOV_DEG),
                             px if px is not None else dflt.get('px_w', CROP_PX))
    rv = getattr(args, 'rubric_version', None)
    if rv is None:
        rv = snap.get('rubric_version', RUBRIC_VERSION)
    if rv not in RUBRICS:
        raise SystemExit(f'rubric version {rv}: known {sorted(RUBRICS)}')
    return k, geom, rv


def views_block(k, geom, rule):
    """snapshot.json's `views`; the square default writes exactly the pre-#169 keys."""
    out = {'per_corner': k, 'min_m': MIN_VIEW_M, 'max_m': MAX_VIEW_M, 'rule': rule,
           'camera_height_m': CAMERA_HEIGHT_M, 'crop_fov_deg': CROP_FOV_DEG,
           'crop_px': CROP_PX, 'store': f'makelab2 {STORE}'}
    if geom:
        out.update({'crop_fov_deg': geom['fov_h_deg'], 'crop_px': geom['px_w'],
                    'crop_geometry': geom})
    return out


def rubric_for(snapshot):
    """(version, text) a bundle is rated under: its snapshot's, which must be a known version's
    exact text (an edited rubric needs a new version)."""
    v = snapshot.get('rubric_version', 1)
    text = snapshot.get('rubric', RUBRICS.get(v))
    if RUBRICS.get(v) != text:
        raise SystemExit(f'snapshot.json rubric v{v} is not the text of RUBRICS[{v}]: an edited '
                         'rubric needs a new version')
    return v, text


# ----------------------------------------------------------------------------- loading

def load_pano_index(results_paths, wanted):
    """{pano_id: {lat, lng, heading, capture_date, run}} for the wanted ids; the first run
    that holds an id wins (the build refuses a pano shared between runs)."""
    out = {}
    for name, path in results_paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)['pano']
                pid = p['panorama_id']
                if pid in wanted and pid not in out and p.get('camera_heading') is not None:
                    out[pid] = {'lat': p['lat'], 'lng': p['lng'],
                                'heading': float(p['camera_heading']),
                                'capture_date': p.get('capture_date'), 'run': name}
    return out


def load_inventory(path):
    """{UNITID: [(lat, lng, STATUS, RAMPTYPE)]} for every inventory point. UNITID is NOT
    unique in the Vancouver pull (111 ids repeat), so a point is resolved by position:
    see inventory_position."""
    import corner_inventory as ci
    feats = json.loads(Path(path).read_text(encoding='utf-8'))['features']
    out = {}
    for f in feats:
        props = f.get('properties') or {}
        lng, lat = f['geometry']['coordinates'][:2]
        out.setdefault(props.get('UNITID'), []).append((lat, lng, ci.inv_class(props)))
    return out


def inventory_position(inv, uid, cls, lat0, lng0, window_m=WINDOW_M):
    """(lat, lng) of the point with this UNITID and class nearest (lat0, lng0), which must be
    inside the window (+1 m); (None, None) when there is none. Two points with this id and
    class inside the window more than 1 m apart are ambiguous, and raise ValueError rather
    than silently picking one twin (none in the Vancouver bundle: the repeated ids there have
    their other copy 10-19 km away)."""
    hits = []
    for lat, lng, c in inv.get(uid, []):
        if c != cls:
            continue
        d = geo.haversine_m(lat0, lng0, lat, lng)
        if d <= window_m + 1.0:
            hits.append((d, lat, lng))
    if not hits:
        return (None, None)
    hits.sort()
    for _d, lat, lng in hits[1:]:
        if geo.haversine_m(hits[0][1], hits[0][2], lat, lng) > 1.0:
            raise ValueError(f'{uid} ({cls}): {len(hits)} points within {window_m + 1:g} m of '
                             f'({lat0}, {lng0}); the position join is ambiguous')
    return (hits[0][1], hits[0][2])


# ------------------------------------------------------------------------------- items

def build_item(rec, part, panos, inv_pos, k=VIEWS_PER_CORNER, geom=None):
    """One review unit: corners with their views and (hidden until reveal) inventory.

    k views per corner (choose_views). geom: crop_geometry() of a landscape bundle, stored on
    the unit as `crop_geometry` (absent for square crops). A --split-wide corner table's
    `phantom_legs` and per-corner `phantom_start` / `phantom_end` are carried when present;
    an item from an older table has none of these keys, exactly as before."""
    import export_cluster_review as ecr
    unit = rec['unit']
    cand_ids = set(rec.get('pano_ids_25') or [])
    for c in rec['corners']:
        cand_ids.update(c.get('pano_ids_25') or [])
    win, bbox = ecr.aerial_window(rec['lat'], rec['lng'])
    corners = []
    for c in rec['corners']:
        cands = []
        for pid in sorted(cand_ids):
            p = panos.get(pid)
            if p is None:
                continue
            v = pano_view(p['lat'], p['lng'], p['heading'], c['lat'], c['lng'])
            v.update({'pano_id': pid, 'capture_date': p['capture_date'], 'run': p['run'],
                      'cam': {'lat': round(p['lat'], 7), 'lng': round(p['lng'], 7)}})
            cands.append(v)
        views = choose_views(cands, k)
        pool = [v for v in cands if v['dist_m'] <= MAX_VIEW_M]
        for v in views:
            v['crop'] = f"crops/{safe_name(unit)}_c{c['corner']}_{v['pano_id']}.jpg"
        corners.append({
            **({'phantom_start': c['phantom_start'], 'phantom_end': c['phantom_end']}
               if 'phantom_start' in c else {}),
            'corner': c['corner'], 'start_deg': c['start_deg'], 'width_deg': c['width_deg'],
            'wide': c['wide'], 'lat': c['lat'], 'lng': c['lng'],
            'n_panos_25': c['n_panos_25'], 'views': views,
            # the newest capture date among every candidate within MAX_VIEW_M, shown or not;
            # choose_views always shows one pano of this date (B1 of the review)
            'newest_available': newest_date(pool) or None, 'n_candidates': len(pool),
            'inv_counts': c['inv_counts'],
            'inventory': [dict(x, **dict(zip(('lat', 'lng'), inventory_position(
                inv_pos, x['unit_id'], x['class'], rec['lat'], rec['lng']))))
                for x in c['inventory']]})
    extra = {}
    if 'phantom_legs' in rec:
        extra['phantom_legs'] = rec['phantom_legs']
    if geom:
        extra['crop_geometry'] = geom
    return {**extra, 'unit': unit, 'city': CITY, 'part': part, 'type': rec['type'],
            'centre': {'lat': rec['lat'], 'lng': rec['lng']}, 'window_m': WINDOW_M,
            'legs': rec['legs'], 'n_legs': rec['n_legs'], 'node_ids': rec['node_ids'],
            'highways': rec.get('highways'), 'state': rec['state'][ARM_STATE],
            'n_panos_25': rec['n_panos_25'], 'nearest_pano_m': rec['nearest_pano_m'],
            'inv_counts': rec['inv_counts'],
            'aerial': {'file': f'aerial/{safe_name(unit)}.jpg', 'px': ecr.AERIAL_PX,
                       'north_up': True, 'zoom': ecr.AERIAL_ZOOM, 'world_px': win, 'bbox': bbox,
                       'source': 'Esri World Imagery'},
            'corners': corners}


def crop_jobs(items, bundle):
    """Crop-worker items for every view whose crop is not on disk yet, by run."""
    out = {}
    for it in items:
        for c in it['corners']:
            for v in c['views']:
                name = Path(v['crop']).name
                if (Path(bundle) / v['crop']).exists():
                    continue
                job = {'pano_id': v['pano_id'], 'x': v['x'], 'y': v['y'],
                       'fov_deg': CROP_FOV_DEG, 'px': CROP_PX, 'name': name,
                       'path': f"{v['pano_id'][:2]}/{v['pano_id']}.jpg"}
                g = it.get('crop_geometry')
                if g:                   # landscape (#169): the worker's optional keys
                    job.update({'fov_deg': g['fov_h_deg'], 'px': g['px_w'],
                                'fov_h_deg': g['fov_h_deg'], 'px_w': g['px_w'],
                                'px_h': g['px_h']})
                out.setdefault(v['run'], []).append(job)
    return out


IMAGES_MANIFEST = 'images.sha256'


def shown_images(items):
    """Every image the rater sees, as bundle-relative paths: each unit's aerial and every
    view's crop, sorted."""
    out = set()
    for it in items:
        out.add(it['aerial']['file'])
        for c in it['corners']:
            out.update(v['crop'] for v in c['views'])
    return sorted(out)


def write_image_manifest(bundle, items):
    """Write <bundle>/images.sha256 (``sha256sum`` format, so ``sha256sum -c images.sha256``
    works from the bundle dir) over the images on disk; return the snapshot block. The aerials
    and crops are git-ignored and Esri imagery changes over time, so this is what proves a
    rebuild or a second rater's copy shows the same pixels (review S4)."""
    rows = [(sha256_file(Path(bundle) / f), f) for f in shown_images(items)
            if (Path(bundle) / f).exists()]
    text = ''.join(f'{h}  {f}\n' for h, f in rows)
    write_text_lf(Path(bundle) / IMAGES_MANIFEST, text)
    return {'manifest': IMAGES_MANIFEST,
            'manifest_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'n_aerial': sum(1 for _h, f in rows if f.startswith('aerial/')),
            'n_crop': sum(1 for _h, f in rows if f.startswith('crops/'))}


def check_images(bundle, snapshot, items):
    """Every reason the images on disk are not the ones the manifest recorded (empty = ok):
    the manifest itself altered, an image the items reference but the manifest lacks, a
    listed image missing or with another sha256."""
    bundle = Path(bundle)
    meta = snapshot.get('images')
    if not meta:
        return ['snapshot.json has no image manifest (built before review S4): rebuild']
    mpath = bundle / meta['manifest']
    if not mpath.exists():
        return [f'{mpath} is missing']
    text = mpath.read_text(encoding='utf-8')
    problems = []
    if hashlib.sha256(text.encode()).hexdigest() != meta['manifest_sha256']:
        problems.append(f'{meta["manifest"]} is not the one snapshot.json recorded')
    listed = {}
    for line in text.splitlines():
        h, sep, f = line.partition('  ')
        if not sep or not re.fullmatch(r'[0-9a-f]{64}', h):
            problems.append(f'{meta["manifest"]}: malformed line {line[:60]!r}')
            continue
        listed[f] = h
    for f in shown_images(items):
        if f not in listed:
            problems.append(f'{f}: shown to the rater but not in the manifest')
    for f, h in sorted(listed.items()):
        p = bundle / f
        if not p.exists():
            problems.append(f'{f}: missing')
        elif sha256_file(p) != h:
            problems.append(f'{f}: sha256 differs from the manifest')
    return problems


def write_report(bundle, snapshot, items, missing):
    parts = snapshot['draw']
    lines = [
        '# Corner present/absent gallery (RampNet#243): bundle report', '',
        f"Built {snapshot['built_at']} by `scripts/corner_gallery.py` at "
        f"`{snapshot['tool_git_sha']}`. Seed **{snapshot['seed']}**. Items sha256 "
        f"`{snapshot['items_sha256']}`.", '',
        '| part | population | drawn | corners | views |', '|---|---:|---:|---:|---:|']
    for p in PARTS:
        its = [i for i in items if i['part'] == p]
        lines.append(f"| {p} | {snapshot['populations'][p]} | {len(parts[p])} | "
                     f"{sum(len(i['corners']) for i in its)} | "
                     f"{sum(len(c['views']) for i in its for c in i['corners'])} |")
    nc = sum(len(i['corners']) for i in items)
    no_view = [(i['unit'], c['corner']) for i in items for c in i['corners'] if not c['views']]
    lines += ['', f'{len(items)} units, {nc} corners. Corners with no pano within '
              f'{MAX_VIEW_M:g} m: {len(no_view)}' + (f' ({no_view})' if no_view else '') + '.',
              f'Crops that could not be cut: {len(missing)}' +
              (f" (e.g. {sorted(missing)[:3]})" if missing else '') + '.', '',
              '## Sampling', '',
              f"- Populations (fusion arm, primary observability, intersections): "
              f"`false_absence` = absent, >= 1 `Available` point, a #241 target unit; "
              f"`na_noramp` = absent, every inventory point `NA` with no `RAMPTYPE`; "
              f"`clean` = absent, no inventory point. Sizes match amendment A8 "
              f"(35 / 944 / 333).",
              f"- Draw: every false absence; then one `random.Random({snapshot['seed']})`, "
              f"`rng.sample` of {snapshot['n_na']} from the sorted `na_noramp` keys, then "
              f"{snapshot['n_clean']} from the sorted `clean` keys.",
              '- Review order: sha1 of the unit key, so a unit\'s slot says nothing about '
              'its part. The part is not shown to the rater.',
              f"- Views: up to {VIEWS_PER_CORNER} panos per corner from the candidates within "
              f"{MAX_VIEW_M:g} m of the corner point (taken from the panos within 25 m of the "
              f"unit centre or the corner point). One slot always goes to the newest capture "
              f"date among the candidates; the rest are nearest first among those >= "
              f"{MIN_VIEW_M:g} m away (nearer ones only to top up). Crops are "
              f"{CROP_FOV_DEG} deg / {CROP_PX} px squares of the equirect pano centred on the "
              f"corner point, projected with a level camera at {CAMERA_HEIGHT_M} m.", '',
              '## Inputs', '', '| input | sha256 |', '|---|---|']
    for k, v in snapshot['inputs'].items():
        lines.append(f"| {k} | `{v['sha256']}` |")
    lines += ['', '## Units', '', '| order | unit | part | type | corners |',
              '|---:|---|---|---|---:|']
    for n, i in enumerate(items, 1):
        lines.append(f"| {n} | `{i['unit']}` | {i['part']} | {i['type']} | {len(i['corners'])} |")
    write_text_lf(Path(bundle) / 'report.md', '\n'.join(lines) + '\n')


def cmd_build(args):
    if getattr(args, 'items_from', None):
        return cmd_build_sample(args)
    if args.targets is None or args.false_absences is None:
        raise SystemExit('--targets and --false-absences are needed without --items-from')
    t0 = time.time()
    bundle = Path(args.bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    build = json.loads(Path(args.build_json).read_text(encoding='utf-8'))
    inputs = {}

    def rec_input(name, path, want=None):
        h = sha256_file(path)
        if want and h != want:
            raise SystemExit(f'{path}: sha256 {h[:12]} is not the recorded {want[:12]}')
        inputs[name] = {'path': portable_path(path), 'sha256': h}

    def recorded(name):
        p = Path(build['inputs'][name]['path'])
        if not p.exists():
            raise SystemExit(f'{p} (from build.json) does not exist on this machine')
        return p, build['inputs'][name]['sha256']

    rec_input('build_json', args.build_json)
    rec_input('corners_full', args.corners_full)
    rec_input('targets', args.targets)
    rec_input('false_absences', args.false_absences)
    inv_path, inv_sha = recorded('inventory')
    rec_input('inventory', inv_path, inv_sha)
    results = []
    for name in ['results'] + sorted(k for k in build['inputs']
                                     if k.startswith('extra:') and k.endswith(':results')):
        p, h = recorded(name)
        rec_input(name, p, h)
        results.append((name, p))

    records = read_jsonl(args.corners_full)
    by_unit = {r['unit']: r for r in records}
    with open(args.targets, newline='', encoding='utf-8') as f:
        targets = {row['unit'] for row in csv.DictReader(f)}
    pops = populations(records, targets)
    sizes = {p: len(v) for p, v in pops.items()}
    if sizes != EXPECTED_POPULATION and not args.allow_population_mismatch:
        raise SystemExit(f'population sizes {sizes} != amendment A8 {EXPECTED_POPULATION}')
    with open(args.false_absences, newline='', encoding='utf-8') as f:
        fa_csv = {row['unit'] for row in csv.DictReader(f) if row['level'] == 'unit'}
    if not set(pops['false_absence']) <= fa_csv:
        raise SystemExit('a false absence is not in false_absences.csv')
    drawn = draw(pops, args.seed, args.n_na, args.n_clean)
    part_of = {u: p for p, us in drawn.items() for u in us}
    order = review_order(list(part_of))

    wanted = set()
    for u in order:
        r = by_unit[u]
        wanted.update(r.get('pano_ids_25') or [])
        for c in r['corners']:
            wanted.update(c.get('pano_ids_25') or [])
    panos = load_pano_index(results, wanted)
    inv_pos = load_inventory(inv_path)
    k, geom, rv = bundle_params(args, bundle)
    items = [build_item(by_unit[u], part_of[u], panos, inv_pos, k, geom) for u in order]

    items_path = bundle / 'items.jsonl'
    if items_path.exists() and not args.rebuild:
        old = read_jsonl(items_path)
        if [i['unit'] for i in old] != [i['unit'] for i in items]:
            raise SystemExit(f'{items_path} exists with a different unit list; refusing to '
                             're-sample (pass --rebuild to replace it deliberately)')
    with open(items_path, 'w', encoding='utf-8', newline='\n') as f:
        for it in items:
            f.write(json.dumps(it, separators=(',', ':'), sort_keys=True) + '\n')
    import split_figures as sf
    snapshot = {
        'schema': SNAPSHOT_SCHEMA, 'city': CITY, 'issue': 'RampNet#243',
        'built_at': utc_now(), 'tool_git_sha': git_sha(), 'tool_git_dirty': git_dirty(),
        # random.sample's draw is stable across CPython 3 releases for a given seed, but
        # record the interpreter anyway so a different draw can be traced
        'python': platform.python_version(), 'seed': args.seed,
        'n_na': args.n_na, 'n_clean': args.n_clean, 'arm_state': ARM_STATE,
        'populations': sizes, 'draw': drawn, 'review_order': order,
        'items_sha256': sha256_file(items_path), 'inputs': inputs,
        'views': views_block(k, geom, 'one slot for the newest capture date among the '
                             'candidates within max_m (nearest such pano), the rest nearest '
                             'first among those >= min_m, nearer ones only to top up '
                             '(choose_views; RampNet#243 B1)'),
        'aerial': {'source': 'Esri World Imagery', 'url_template': sf.TILE_URL,
                   'attribution': sf.ATTRIBUTION},
        'rubric_version': rv, 'rubric': RUBRICS[rv]}

    missing = make_images(args, bundle, items)
    snapshot['crops_missing'] = sorted(missing)
    snapshot['images'] = write_image_manifest(bundle, items)
    snapshot['seconds'] = round(time.time() - t0, 1)
    write_text_lf(bundle / 'snapshot.json', json.dumps(snapshot, indent=1) + '\n')
    write_report(bundle, snapshot, items, missing)
    print(f"{len(items)} units ({', '.join(f'{p} {len(drawn[p])}' for p in PARTS)}), "
          f"{sum(len(i['corners']) for i in items)} corners; {len(missing)} crops missing")
    print(f'wrote {items_path}, snapshot.json, report.md')
    return 0


def make_images(args, bundle, items):
    """Make the aerials (Esri tiles, cached) and the crops (makelab2's PS store through the
    crop worker, or --local-panos); return the bundle-relative crops still missing."""
    import export_cluster_review as ecr
    if not args.skip_aerial:
        (bundle / 'aerial').mkdir(exist_ok=True)
        seed_tiles = Path(args.seed_tiles) if args.seed_tiles else None
        made = fetched = 0
        for it in items:
            out = bundle / it['aerial']['file']
            if out.exists():
                continue
            fetched += ecr.make_aerial(it['aerial']['world_px'], out, bundle / 'cache' / 'tiles',
                                       seed_tiles if seed_tiles and seed_tiles.exists() else None)
            made += 1
        print(f'aerials: {made} made, {fetched} tiles fetched')

    missing = set()
    if not args.no_crops:
        import site_explorer as se
        (bundle / 'crops').mkdir(exist_ok=True)
        for run, jobs in sorted(crop_jobs(items, bundle).items()):
            job = {'items': jobs, 'workers': args.workers, 'draft_width': 4096}
            if args.local_panos:
                job['panos_root'] = str(args.local_panos)
                miss = se.fetch_crops_local(job, bundle / 'crops', args.local_panos)
            else:
                job['panos_root'] = STORE
                miss = se.fetch_crops_remote(job, bundle / 'crops', args.helper or se.DEFAULT_HELPER,
                                             args.host, se.DEFAULT_REMOTE_ROOT,
                                             se.DEFAULT_REMOTE_HOME, args.timeout)
            print(f'crops [{run}]: {len(jobs)} requested, {len(miss)} missing')
    return {v['crop'] for it in items for c in it['corners'] for v in c['views']
            if not (bundle / v['crop']).exists()}


# ------------------------------------------------------------ the #169 sample bundle

def read_sample_keys(path):
    """[(unit, corner)] from a corner_labels.py sample.csv (columns unit, corner), in file
    order; a repeated key is refused."""
    with open(path, newline='', encoding='utf-8') as f:
        keys = [(row['unit'], int(row['corner'])) for row in csv.DictReader(f)]
    if len(set(keys)) != len(keys):
        raise SystemExit(f'{path}: a corner is listed twice')
    if not keys:
        raise SystemExit(f'{path}: no corners')
    return keys


def corner_key(unit, corner):
    """The corner key corner_labels.py uses: '<unit>#<corner>'."""
    return f'{unit}#{corner}'


def sample_item(rec, corner_ids, panos, inv_pos, proposals, k=VIEWS_PER_CORNER, geom=None):
    """One review unit of a --items-from bundle: build_item over the WHOLE unit (so the view
    pool is the one gallery 243 would use), then only the sampled corners are kept. Each kept
    corner carries the rule's `proposal` (chosen pano, pixel, range; never shown to the rater)
    and whether that pano is among the views shown (`proposal.shown`)."""
    it = build_item(rec, SAMPLE_PART, panos, inv_pos, k, geom)
    it['n_corners_unit'] = len(it['corners'])
    kept = []
    for c in it['corners']:
        if c['corner'] not in corner_ids:
            continue
        pr = proposals[corner_key(rec['unit'], c['corner'])]
        c['proposal'] = {'key': pr['key'], 'pano_id': pr['pano']['pano_id'],
                         'run': pr['pano'].get('run'), 'x_normalized': pr['x_normalized'],
                         'y_normalized': pr['y_normalized'], 'range_m': pr['range_m'],
                         'shown': pr['pano']['pano_id'] in {v['pano_id'] for v in c['views']}}
        kept.append(c)
    if {c['corner'] for c in kept} != set(corner_ids):
        raise SystemExit(f"{rec['unit']}: sampled corner(s) "
                         f"{sorted(set(corner_ids) - {c['corner'] for c in kept})} not in the unit")
    it['corners'] = kept
    return it


def write_crops_needed(bundle, items, missing, args):
    """Write CROPS_NEEDED.md and crop_job.json when crops are missing (remove a stale one when
    none are). The crops are cut from the PS pano store on makelab2, which this machine cannot
    write to on its own (a Claude session must not); the file says what Jon runs."""
    bundle = Path(bundle)
    need = bundle / CROPS_NEEDED
    job_path = bundle / CROP_JOB_FILE
    if not missing:
        for f in (need, job_path):
            if f.exists():
                f.unlink()
        return None
    jobs = [j for js in crop_jobs(items, bundle).values() for j in js]
    job = {'items': sorted(jobs, key=lambda j: j['name']), 'workers': 8, 'draft_width': 4096,
           'panos_root': STORE}
    write_text_lf(job_path, json.dumps(job, indent=1, sort_keys=True) + '\n')
    panos = sorted({j['pano_id'] for j in jobs})
    b = portable_path(bundle)
    sample = portable_path(args.items_from)
    corners = portable_path(args.corners_full)
    snap_views = {}
    if (bundle / 'snapshot.json').exists():
        snap_views = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8')).get(
            'views') or {}
    g = (items[0].get('crop_geometry') if items else None) or {}
    flags = ''
    if snap_views.get('per_corner', VIEWS_PER_CORNER) != VIEWS_PER_CORNER:
        flags += f" --views-per-corner {snap_views['per_corner']}"
    if g:
        flags += f" --crop-aspect {g['aspect']:g} --crop-fov-deg {g['fov_h_deg']:g} --crop-px {g['px_w']}"
    if flags:
        corners += ' \\\n   ' + flags
    text = f"""# Crops needed: {len(missing)} of {sum(len(c['views']) for i in items for c in i['corners'])}

The crops of this bundle could not be cut here: they come from {len(panos)} panos of the Project
Sidewalk pano store on makelab2 (`{STORE}`), and no local copy of those panos was found. The
gallery must not be rated until the crops exist (`render` refuses while `images.sha256` lacks
them). Everything else (items, snapshot, aerials, report) is built.

## Option A, one command (recommended)

From this worktree on the Windows desktop, with the makelab2 helper working
(`D:\\Git\\dotfiles\\wsl-ssh.ps1 makelab2 check`) and the analysis extras installed
(`pip install pandas scipy haversine`; `split_figures` needs them), re-run the same build
without `--no-crops`.
It keeps `items.jsonl` (same unit list), cuts the crops on makelab2 with the crop worker
(`site_explorer.CROP_WORKER`, staged under `/homes/gws/jonf/.sidewalk_explorer/<hash>` and
removed afterwards), pulls them back as one tarball, then rewrites `images.sha256`,
`snapshot.json` and `report.md` and deletes this file:

```bash
python scripts/corner_gallery.py build --items-from {sample} \\
    --corners-full {corners} \\
    --bundle {b}
python scripts/corner_gallery.py render --bundle {b} --rater jonf
```

## Option B, by hand on makelab2

`{CROP_JOB_FILE}` here is the crop worker's job (`panos_root` = the store, each item's `path` is
`<id[:2]>/<id>.jpg`). On makelab2:

```bash
mkdir -p ~/crops169 && cd ~/crops169
# copy crop_job.json here, and the CROP_WORKER string of scripts/site_explorer.py as crop.py
/projects/makeabilitylab/sidewalk-auto-labeler/.venv/bin/python crop.py crop_job.json out
tar -czf crops169.tar.gz -C out .
```

Copy `crops169.tar.gz` back, unpack it into `{b}/crops/`, then run the two commands of
option A (the build finds the crops on disk, cuts nothing, and writes the manifest).
"""
    write_text_lf(need, text)
    return need


def cmd_build_sample(args):
    """build --items-from: the rating bundle over a corner_labels.py sample (issue #169)."""
    t0 = time.time()
    bundle = Path(args.bundle)
    sample_csv = Path(args.items_from)
    sample_json = sample_csv.with_suffix('.json')
    proposals_path = Path(args.proposals) if args.proposals else sample_csv.parent / 'proposals.jsonl'
    labels_build = proposals_path.parent / 'build.json'
    for f in (sample_json, proposals_path, labels_build):
        if not f.exists():
            raise SystemExit(f'{f} does not exist (the sample needs its sample.json, the '
                             f'proposals it was drawn from and their build.json)')
    sj = json.loads(sample_json.read_text(encoding='utf-8'))
    lb = json.loads(labels_build.read_text(encoding='utf-8'))
    s_sha, p_sha = sha256_file(sample_csv), sha256_file(proposals_path)
    if sj['sample']['sha256'] != s_sha:
        raise SystemExit(f'{sample_csv}: sha256 {s_sha[:12]} is not the one {sample_json.name} '
                         f'recorded ({sj["sample"]["sha256"][:12]})')
    if sj['proposals']['sha256'] != p_sha:
        raise SystemExit(f'{proposals_path}: not the proposals the sample was drawn from')
    if lb['outputs']['proposals']['sha256'] != p_sha:
        raise SystemExit(f'{proposals_path}: not the proposals {labels_build} records')
    cf_sha = sha256_file(args.corners_full)
    if lb['inputs']['corners_full']['sha256'] != cf_sha:
        raise SystemExit(f'{args.corners_full}: not the corners_full the proposals were built on')
    build = json.loads(Path(args.build_json).read_text(encoding='utf-8'))
    inputs = {}

    def rec_input(name, path, want=None):
        h = sha256_file(path)
        if want and h != want:
            raise SystemExit(f'{path}: sha256 {h[:12]} is not the recorded {want[:12]}')
        inputs[name] = {'path': portable_path(path), 'sha256': h}

    def recorded(name):
        q = Path(build['inputs'][name]['path'])
        if not q.exists():
            raise SystemExit(f'{q} (from build.json) does not exist on this machine')
        return q, build['inputs'][name]['sha256']

    rec_input('sample', sample_csv)
    rec_input('sample_json', sample_json)
    rec_input('proposals', proposals_path)
    rec_input('labels_build_json', labels_build)
    rec_input('build_json', args.build_json)
    rec_input('corners_full', args.corners_full)
    inv_path, inv_sha = recorded('inventory')
    rec_input('inventory', inv_path, inv_sha)
    results = []
    for name in ['results'] + sorted(k for k in build['inputs']
                                     if k.startswith('extra:') and k.endswith(':results')):
        q, h = recorded(name)
        rec_input(name, q, h)
        results.append((name, q))

    keys = read_sample_keys(sample_csv)
    proposals = {}
    with open(proposals_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pr = json.loads(line)
                proposals[pr['key']] = pr
    lost = [corner_key(u, c) for u, c in keys if corner_key(u, c) not in proposals]
    if lost:
        raise SystemExit(f'{len(lost)} sampled corner(s) are not proposals, e.g. {lost[:3]}')
    by_unit_corners = {}
    for u, c in keys:
        by_unit_corners.setdefault(u, set()).add(c)
    records = {r['unit']: r for r in read_jsonl(args.corners_full) if r['unit'] in by_unit_corners}
    missing_units = sorted(set(by_unit_corners) - set(records))
    if missing_units:
        raise SystemExit(f'sampled unit(s) not in corners_full: {missing_units[:3]}')
    # a sequential sample (#183) is rated in looks, so it is served in sample order
    order = (list(dict.fromkeys(u for u, _ in keys)) if sj.get('design')
             else review_order(list(by_unit_corners)))
    wanted = set()
    for u in order:
        wanted.update(records[u].get('pano_ids_25') or [])
        for c in records[u]['corners']:
            wanted.update(c.get('pano_ids_25') or [])
    panos = load_pano_index(results, wanted)
    inv_pos = load_inventory(inv_path)
    k, geom, rv = bundle_params(args, bundle)
    items = [sample_item(records[u], by_unit_corners[u], panos, inv_pos, proposals, k, geom)
             for u in order]

    bundle.mkdir(parents=True, exist_ok=True)
    items_path = bundle / 'items.jsonl'
    old_snap = bundle / 'snapshot.json'
    built_on = ((json.loads(old_snap.read_text(encoding='utf-8')).get('sample') or {})
                .get('json_sha256') if old_snap.exists() else None)
    if built_on not in (None, inputs['sample_json']['sha256']) and not args.rebuild:
        raise SystemExit(f'{sample_json} is not the one this bundle was built from (a look '
                         'schedule is fixed before rating); pass --rebuild to replace the '
                         'bundle deliberately')
    if items_path.exists() and not args.rebuild:
        old = read_jsonl(items_path)
        if [(i['unit'], [c['corner'] for c in i['corners']]) for i in old] != \
                [(i['unit'], [c['corner'] for c in i['corners']]) for i in items]:
            raise SystemExit(f'{items_path} exists with a different corner list; refusing to '
                             're-sample (pass --rebuild to replace it deliberately)')
    with open(items_path, 'w', encoding='utf-8', newline='\n') as f:
        for it in items:
            f.write(json.dumps(it, separators=(',', ':'), sort_keys=True) + '\n')
    import split_figures as sf
    params = lb.get('params') or {}
    snapshot = {
        'schema': SNAPSHOT_SCHEMA, 'city': CITY, 'issue': 'sidewalk-auto-labeler#169',
        'mode': 'sample', 'built_at': utc_now(), 'tool_git_sha': git_sha(),
        'tool_git_dirty': git_dirty(), 'python': platform.python_version(),
        'seed': sj.get('seed'), 'arm_state': ARM_STATE,
        'sample': {'path': portable_path(sample_csv), 'sha256': s_sha,
                   'json_sha256': inputs['sample_json']['sha256'], 'seed': sj.get('seed'),
                   'n': len(keys), 'population': sj.get('population'),
                   'rule': sj.get('rule')},
        'proposals': {'path': portable_path(proposals_path), 'sha256': p_sha,
                      'build_json_sha256': inputs['labels_build_json']['sha256'],
                      'tool_git_sha': (lb.get('tool') or {}).get('git_sha'),
                      'na_read': params.get('na_read'),
                      'na_read_settled': params.get('na_read_settled'),
                      'sidewalk_rule': 'applied' if isinstance(lb.get('sidewalk_rule'), dict)
                      else lb.get('sidewalk_rule')},
        'populations': {SAMPLE_PART: sj.get('population')},
        'draw': {SAMPLE_PART: [corner_key(u, c) for u, c in keys]},
        'review_order': order, 'items_sha256': sha256_file(items_path), 'inputs': inputs,
        'shown_corners': 'only the sampled corners of each unit',
        'reveal': 'the city inventory, as in gallery 243; the rule proposal (pano, pixel) is '
                  'in items.jsonl and never shown',
        'views': views_block(k, geom, 'as gallery 243 (choose_views over the whole unit\'s '
                                      'pano pool)'),
        'aerial': {'source': 'Esri World Imagery', 'url_template': sf.TILE_URL,
                   'attribution': sf.ATTRIBUTION},
        'rubric_version': rv, 'rubric': RUBRICS[rv],
        'scorer': 'scripts/corner_labels.py score --proposals <proposals> --sample <sample.csv> '
                  '--bundle <this bundle> --verdicts <bundle>/verdicts__<rater>.json'}
    if sj.get('design'):
        snapshot['sample'].update({'n': sj['n'], 'reserve': sj['reserve'], 'drawn': len(keys),
                                   'design': sj['design']})
        snapshot['review_order_rule'] = 'sample order (sequential sample, #183)'
    missing = make_images(args, bundle, items)
    snapshot['crops_missing'] = sorted(missing)
    snapshot['images'] = write_image_manifest(bundle, items)
    snapshot['seconds'] = round(time.time() - t0, 1)
    write_text_lf(bundle / 'snapshot.json', json.dumps(snapshot, indent=1) + '\n')
    need = write_crops_needed(bundle, items, missing, args)
    write_sample_report(bundle, snapshot, items, missing)
    nc = sum(len(i['corners']) for i in items)
    print(f'{len(items)} units, {nc} sampled corners; {len(missing)} crops missing')
    if need:
        print(f'crops are missing: see {need}')
    print(f'wrote {items_path}, snapshot.json, report.md')
    return 0


def write_sample_report(bundle, snapshot, items, missing):
    sm, pr = snapshot['sample'], snapshot['proposals']
    corners = [c for i in items for c in i['corners']]
    no_view = [c['proposal']['key'] for c in corners if not c['views']]
    by_type = {}
    for i in items:
        by_type[i['type']] = by_type.get(i['type'], 0) + len(i['corners'])
    lines = [
        '# Corner precision gallery (#169): bundle report', '',
        f"Built {snapshot['built_at']} by `scripts/corner_gallery.py build --items-from` at "
        f"`{snapshot['tool_git_sha']}`. Items sha256 `{snapshot['items_sha256']}`.", '',
        f"- Sample: `{sm['path']}` (sha256 `{sm['sha256']}`), {sm.get('drawn', sm['n'])} corners "
        f"drawn with seed {sm['seed']} from {sm['population']} proposals ({sm['rule']}).",
        f"- Proposals: `{pr['path']}` (sha256 `{pr['sha256']}`), built at "
        f"`{pr['tool_git_sha']}`, NA read `{pr['na_read']}` (settled: {pr['na_read_settled']}), "
        f"sidewalk rule {pr['sidewalk_rule']}.",
        f"- {len(items)} units, {len(corners)} corners; by stratum "
        f"{dict(sorted(by_type.items()))}. Only the sampled corners of a unit are shown and rated.",
        f"- Views: up to {snapshot['views']['per_corner']} per corner, chosen as in gallery 243"
        + (f" (newest-date slot, then distance order); landscape crops "
           f"{snapshot['views']['crop_geometry']['fov_h_deg']:g} x "
           f"{snapshot['views']['crop_geometry']['fov_v_deg']:g} deg at "
           f"{snapshot['views']['crop_geometry']['px_w']} x "
           f"{snapshot['views']['crop_geometry']['px_h']} px"
           if snapshot['views'].get('crop_geometry') else '') + ". The pano that "
        f"carries the rule's label is among them at {sum(c['proposal']['shown'] for c in corners)}"
        f" of {len(corners)} corners. Corners with no view: {len(no_view)}"
        + (f' ({no_view})' if no_view else '') + '.',
        f"- Crops missing: {len(missing)}" + (f' (see `{CROPS_NEEDED}`)' if missing else '') + '.',
        (f"- Rubric: v{snapshot['rubric_version']}"
         + (' (gallery 243 v1 plus the phantom-leg / two-landing paragraph)'
            if snapshot['rubric_version'] == 2 else ', gallery 243 unchanged')
         + '. Reveal: the city inventory, after completion; the blind verdicts are scored.'),
        '- Score: `python scripts/corner_labels.py score --proposals <proposals> --sample '
        '<sample.csv> --bundle <this bundle> --verdicts <bundle>/verdicts__<rater>.json`.', '',
        '## Inputs', '', '| input | sha256 |', '|---|---|']
    for k, v in snapshot['inputs'].items():
        lines.append(f"| {k} | `{v['sha256']}` |")
    split = any('phantom_legs' in i for i in items)
    if split:
        ph = [c for i in items for c in i['corners']
              if c.get('phantom_start') or c.get('phantom_end')]
        lines[lines.index('## Inputs') - 1:lines.index('## Inputs') - 1] = [
            f"- Split-wide corner table: {sum(1 for i in items if i.get('phantom_legs'))} units "
            f"carry phantom legs; {len(ph)} of {len(corners)} sampled corners are bounded by "
            f"one (the far side of a T); {sum(1 for c in corners if c['wide'])} are still "
            'wider than 150 deg.']
    lines += ['', '## Units', '', '| order | unit | type | sampled corners | of |'
              + (' phantom legs |' if split else ''),
              '|---:|---|---|---|---:|' + ('---|' if split else '')]
    for n, i in enumerate(items, 1):
        lines.append(f"| {n} | `{i['unit']}` | {i['type']} | "
                     f"{', '.join(str(c['corner']) for c in i['corners'])} | {i['n_corners_unit']} |"
                     + ((' ' + (', '.join(f'{b:.0f}' for b in i.get('phantom_legs') or ()) or '-')
                         + ' |') if split else ''))
    if sm.get('design'):
        lines += ['', '## Rating in looks (#183)', '',
                  f"A sequential sample: {sm['n']} decided corners plus a reserve of "
                  f"{sm['reserve']}, served in sample order, with looks at "
                  f"{', '.join(str(b['n']) for b in sm['design']['looks'])} decided corners "
                  '(boundaries in `snapshot.json` `sample.design`).', '',
                  '- Commit `sample.csv`, `sample.json` and this bundle before the first rating.',
                  '- Start at unit 1 and move with Next or "Next to do". Never skip ahead, and '
                  "do not press Prev on unit 1: it wraps to the last unit, the reserve's tail.",
                  '- At each look, export, commit the verdicts file and run the score command '
                  'above. It prints PASS or FAIL at the first look that settles, or how many '
                  'more decided corners the next look needs. Stop at PASS or FAIL.']
    write_text_lf(Path(bundle) / 'report.md', '\n'.join(lines) + '\n')


# ------------------------------------------------------------------------------ render

REVEAL_DIR = 'reveal'


def reveal_file(unit):
    """The unit's reveal script, relative to the gallery dir."""
    return f'{REVEAL_DIR}/{safe_name(unit)}.js'


def reveal_script(it):
    """The city inventory of one unit as a classic script the page inserts only when the unit
    is completed (review S3): ``revealInventory(<unit>, {<corner key>: [points]});``. A
    <script> tag, not fetch(), because fetch() of a file:// URL is refused by Chrome."""
    data = {str(c['corner']): c['inventory'] for c in it['corners']}
    return (f'revealInventory({json.dumps(it["unit"])}, '
            f'{json.dumps(data, sort_keys=True, separators=(",", ":"))});\n')


def viewer_unit(it, rel, aerial_dates=None):
    """The per-unit payload the page needs; paths relative to the gallery dir. The part is
    deliberately left out: the rater never sees which population a unit came from. So is the
    city inventory (status, RAMPTYPE, INSTDATE, positions): it lives in reveal/<unit>.js and is
    loaded only at completion (review S3)."""
    extra = {}
    if it.get('phantom_legs'):
        extra['phantom_legs'] = it['phantom_legs']
    if it.get('crop_geometry'):
        extra['crop_aspect'] = it['crop_geometry']['aspect']
    return {**extra, 'id': it['unit'], 'type': it['type'], 'centre': it['centre'],
            'window_m': it['window_m'], 'legs': it['legs'], 'reveal': reveal_file(it['unit']),
            'aerial': dict(it['aerial'], file=rel + it['aerial']['file'],
                           dates=(aerial_dates or {}).get(it['unit'], [])),
            'corners': [{**({'phantom': [c['phantom_start'], c['phantom_end']]}
                            if c.get('phantom_start') or c.get('phantom_end') else {}),
                         'k': str(c['corner']), 'corner': c['corner'],
                         'start_deg': c['start_deg'], 'width_deg': c['width_deg'],
                         'wide': c['wide'], 'lat': c['lat'], 'lng': c['lng'],
                         'views': [{'pano_id': v['pano_id'], 'date': v['capture_date'],
                                    'dist_m': v['dist_m'], 'cam': v['cam'],
                                    'heading': v['bearing_deg'],
                                    'crop': rel + v['crop']} for v in c['views']]}
                        for c in it['corners']]}


def load_prefill(bundle, rater, items_sha):
    """(verdicts or None, message, sha256 of the file prefilled from or None): a file made on
    another item list is not loaded. The sha256 travels in every export as
    ``prefilled_from_sha256``, so the scorer can tell an export from a stale page (review SF1)."""
    path = Path(bundle) / rater_file_name(rater)
    if not path.exists():
        return None, None, None
    raw = path.read_bytes()
    v = json.loads(raw.decode('utf-8'))
    if v.get('items_sha256') != items_sha:
        return None, (f'NOT prefilled from {path}: made on items {str(v.get("items_sha256"))[:12]},'
                      f' the bundle is {items_sha[:12]}'), None
    return (v, f'prefilled {len(v.get("units") or {})} units from {path} for revision',
            hashlib.sha256(raw).hexdigest())


def build_html(units, items_sha, rater, initial, attribution, prefilled_from=None,
               rubric_version=RUBRIC_VERSION, rubric=RUBRIC):
    from corner_gallery_page import HTML_TEMPLATE, KEYS_JS, PANO_LINKS_JS, STATE_BOOTSTRAP_JS
    return (HTML_TEMPLATE
            .replace('__STATE_BOOTSTRAP__', STATE_BOOTSTRAP_JS)
            .replace('__KEYS__', KEYS_JS)
            .replace('__PANO_LINKS__', PANO_LINKS_JS)
            .replace('__UNITS__', json.dumps(units))
            .replace('__ITEMS_SHA__', json.dumps(items_sha))
            .replace('__CITY__', json.dumps(CITY))
            .replace('__RATER__', json.dumps(rater))
            .replace('__RUBRIC_V__', str(rubric_version))
            .replace('__RUBRIC_JSON__', json.dumps(rubric))
            .replace('__RUBRIC_HTML__', rubric.replace('&', '&amp;').replace('<', '&lt;'))
            .replace('__SCHEMA__', json.dumps(VERDICTS_SCHEMA))
            .replace('__INITIAL__', json.dumps(initial))
            .replace('__PREFILLED_FROM__', json.dumps(prefilled_from))
            .replace('__ATTRIBUTION__', json.dumps(attribution))
            .replace('__PS_HOST__', json.dumps(PS_HOST))
            .replace('__FILE_NAME__', json.dumps(rater_file_name(rater))))


def cmd_render(args):
    bundle = Path(args.bundle)
    snapshot = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
    items_sha = sha256_file(bundle / 'items.jsonl')
    if items_sha != snapshot['items_sha256']:
        raise SystemExit('items.jsonl does not match snapshot.json')
    items = read_jsonl(bundle / 'items.jsonl')
    problems = check_images(bundle, snapshot, items)
    if problems:
        print('\n'.join(problems[:20]))
        if not args.allow_image_drift:
            raise SystemExit(f'{len(problems)} image problem(s): the rater would not see the '
                             f'recorded images (pass --allow-image-drift to render anyway)')
    out = Path(args.out) if args.out else bundle / 'gallery' / args.rater
    out.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(bundle.resolve(), out.resolve()).replace(os.sep, '/') + '/'
    initial, msg, prefilled_from = load_prefill(bundle, args.rater, items_sha)
    if msg:
        print(msg)
    adates = load_aerial_dates(bundle, items_sha)
    rv, rubric = rubric_for(snapshot)
    html = build_html([viewer_unit(i, rel, adates) for i in items], items_sha, args.rater, initial,
                      snapshot['aerial']['attribution'], prefilled_from, rv, rubric)
    (out / REVEAL_DIR).mkdir(exist_ok=True)
    for i in items:
        write_text_lf(out / reveal_file(i['unit']), reveal_script(i))
    write_text_lf(out / 'index.html', html)
    print(f"{len(items)} units, {sum(len(i['corners']) for i in items)} corners")
    print(f"Gallery: {out / 'index.html'}")
    print(f'Open it, rate, Export, and save the download as {bundle / rater_file_name(args.rater)}')
    return 0


# The Project Sidewalk deployment for CITY: its Explore page shows a pano by id, from PS's own
# cached copy when Google has expired it (the enlarged crop links there). Derived from CITY so
# a second city's bundle cannot link to Vancouver's server (review N7).
PS_HOST = f'https://sidewalk-{CITY}.cs.washington.edu'
ESRI_IDENTIFY = ('https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/'
                 'identify')
AERIAL_DATES_FILE = 'aerial_dates.json'


def esri_metadata_at(lat, lng, zoom, timeout=30):
    """The Esri World Imagery metadata record that is drawn at `zoom` at one point, or None.

    The service returns one "<res> Resolution Metadata" layer per resolution band; the one
    whose MinMapLevel..MaxMapLevel covers the zoom is what the tile showed. At zoom 20 that is
    usually the 15 cm layer, whose date can differ from the coarser layers' by months.
    """
    import urllib.parse
    import urllib.request
    d = 0.0005
    q = urllib.parse.urlencode({
        'geometry': f'{lng},{lat}', 'geometryType': 'esriGeometryPoint', 'sr': 4326,
        'layers': 'all', 'tolerance': 1, 'returnGeometry': 'false', 'f': 'json',
        'mapExtent': f'{lng - d},{lat - d},{lng + d},{lat + d}', 'imageDisplay': '512,512,96'})
    with urllib.request.urlopen(f'{ESRI_IDENTIFY}?{q}', timeout=timeout) as r:
        res = json.loads(r.read().decode('utf-8'))['results']
    for r in res:
        a = r['attributes']
        if not r['layerName'].endswith('Resolution Metadata'):
            continue
        try:
            lo, hi = int(a['MinMapLevel']), int(a['MaxMapLevel'])
        except (KeyError, TypeError, ValueError):
            continue
        date = a.get('SRC_DATE2')
        if lo <= zoom <= hi and date and date != 'Null':
            m, dd, y = (int(x) for x in date.split('/'))
            return {'date': f'{y:04d}-{m:02d}-{dd:02d}', 'layer': r['layerName'],
                    'release': a.get('ReleaseName'), 'source': a.get('SRC_DESC') or a.get('NICE_DESC')}
    return None


def cmd_aerial_dates(args):
    """Capture date(s) of every aerial: Esri's metadata at the window centre and its four
    corners, since one 70 m window can straddle two mosaic patches. The metadata describes
    Esri's CURRENT release; the tiles were fetched at build time (snapshot.json), so a
    republish between the two would make the date wrong for the pinned image."""
    bundle = Path(args.bundle)
    items = read_jsonl(bundle / 'items.jsonl')
    out = {}
    for n, it in enumerate(items, 1):
        bb, z = it['aerial']['bbox'], it['aerial']['zoom']
        pts = [(it['centre']['lat'], it['centre']['lng'])] + [
            (la, lo) for la in (bb['south'], bb['north']) for lo in (bb['west'], bb['east'])]
        recs = []
        for la, lo in pts:
            recs.append(esri_metadata_at(la, lo, z))
            time.sleep(args.pause)
        dates = sorted({r['date'] for r in recs if r})
        out[it['unit']] = {'centre': recs[0], 'dates': dates,
                           'n_points': len(pts), 'n_found': sum(1 for r in recs if r)}
        if n % 10 == 0:
            print(f'{n}/{len(items)}')
    doc = {'service': ESRI_IDENTIFY, 'queried_at': utc_now(),
           'items_sha256': sha256_file(bundle / 'items.jsonl'),
           'note': "Esri's metadata for its current release, not frozen with the tiles; the "
                   'tiles were fetched at build time. dates = distinct capture dates over the '
                   'window centre and its four corners.',
           'units': out}
    write_text_lf(bundle / AERIAL_DATES_FILE, json.dumps(doc, indent=1, sort_keys=True) + '\n')
    nd = sum(1 for v in out.values() if not v['dates'])
    nm = sum(1 for v in out.values() if len(v['dates']) > 1)
    print(f'{len(out)} units: {nd} with no date, {nm} spanning more than one date -> '
          f'{bundle / AERIAL_DATES_FILE}')
    return 0


def load_aerial_dates(bundle, items_sha):
    """{unit: [dates]} from aerial_dates.json, or {} when absent or made on another item list."""
    path = Path(bundle) / AERIAL_DATES_FILE
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('items_sha256') != items_sha:
        print(f'NOT using {path}: made on another items.jsonl')
        return {}
    return {u: v['dates'] for u, v in doc['units'].items()}


def cmd_check(args):
    bundle = Path(args.bundle)
    snapshot = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
    items_sha = sha256_file(bundle / 'items.jsonl')
    problems = [] if items_sha == snapshot['items_sha256'] else \
        [f'items.jsonl sha256 {items_sha[:12]} is not the recorded {snapshot["items_sha256"][:12]}']
    problems += check_images(bundle, snapshot, read_jsonl(bundle / 'items.jsonl'))
    for p in problems:
        print(p)
    meta = snapshot.get('images') or {}
    print(f"{'OK' if not problems else 'FAILED'}: items.jsonl and "
          f"{meta.get('n_aerial', 0)} aerials + {meta.get('n_crop', 0)} crops against "
          f"{meta.get('manifest', IMAGES_MANIFEST)}")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='sample and write the bundle')
    b.add_argument('--corners-full', type=Path, required=True,
                   help="the merged #241 build's corners_full.jsonl (untracked)")
    b.add_argument('--build-json', type=Path,
                   default=REPO_ROOT / 'runs/vancouver/corner_inventory_posdetect241/build.json')
    b.add_argument('--targets', type=Path,
                   default=REPO_ROOT / 'runs/vancouver/corner_posdetect241/select/units.csv')
    b.add_argument('--false-absences', type=Path,
                   default=REPO_ROOT / 'runs/vancouver/corner_inventory_posdetect241/'
                                       'false_absences.csv')
    b.add_argument('--items-from', type=Path, default=None,
                   help="#169: a corner_labels.py sample.csv; rate those corners instead of "
                        "drawing gallery 243's three parts (sample.json, proposals.jsonl and "
                        "build.json must sit beside it)")
    b.add_argument('--proposals', type=Path, default=None,
                   help='with --items-from: the proposals (default: beside the sample)')
    b.add_argument('--bundle', type=Path, required=True)
    b.add_argument('--seed', type=int, default=SEED)
    b.add_argument('--n-na', type=int, default=N_NA)
    b.add_argument('--n-clean', type=int, default=N_CLEAN)
    b.add_argument('--rebuild', action='store_true', help='replace an existing items.jsonl')
    b.add_argument('--allow-population-mismatch', action='store_true')
    b.add_argument('--skip-aerial', action='store_true')
    b.add_argument('--seed-tiles', default=None,
                   help='an existing Esri tile cache to copy from before fetching')
    b.add_argument('--no-crops', action='store_true')
    b.add_argument('--local-panos', type=Path, default=None,
                   help='cut crops locally from this sharded pano dir instead of makelab2')
    b.add_argument('--host', default='makelab2')
    b.add_argument('--helper', default=None)
    b.add_argument('--workers', type=int, default=8)
    b.add_argument('--timeout', type=int, default=3600)
    b.add_argument('--views-per-corner', type=int, default=None,
                   help=f'crops per corner (default: the bundle snapshot\'s, else '
                        f'{VIEWS_PER_CORNER}); slot 1 is the newest capture date, the rest '
                        'distance order')
    b.add_argument('--crop-aspect', type=float, default=None,
                   help='crop width:height (default 1, square). 2 = landscape: '
                        '--crop-fov-deg wide x half that tall')
    b.add_argument('--crop-fov-deg', type=float, default=None,
                   help=f'horizontal field of view of a crop (default {CROP_FOV_DEG})')
    b.add_argument('--crop-px', type=int, default=None,
                   help=f'crop width in pixels (default {CROP_PX}); height = width / aspect')
    b.add_argument('--rubric-version', type=int, default=None, choices=sorted(RUBRICS),
                   help=f'default: the bundle snapshot\'s, else {RUBRIC_VERSION}')
    r = sub.add_parser('render', help='write the review page for one rater')
    r.add_argument('--bundle', type=Path, required=True)
    r.add_argument('--rater', required=True, help='exports verdicts__<rater>.json')
    r.add_argument('--out', type=Path, default=None, help='default <bundle>/gallery/<rater>')
    r.add_argument('--allow-image-drift', action='store_true',
                   help='render even if the images differ from images.sha256')
    a = sub.add_parser('aerial-dates', help="look up each aerial's capture date (Esri metadata)")
    a.add_argument('--bundle', type=Path, required=True)
    a.add_argument('--pause', type=float, default=0.1, help='seconds between requests')
    k = sub.add_parser('check', help='verify items.jsonl and every shown image against the '
                                     'recorded sha256s')
    k.add_argument('--bundle', type=Path, required=True)
    args = ap.parse_args(argv)
    return {'build': cmd_build, 'render': cmd_render, 'check': cmd_check,
            'aerial-dates': cmd_aerial_dates}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
