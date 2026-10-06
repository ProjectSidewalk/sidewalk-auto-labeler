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

Two subcommands:

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
"""
import argparse
import csv
import hashlib
import json
import math
import os
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
CROP_FOV_DEG = 60
CROP_PX = 448
WINDOW_M = 30.0
STORE = '/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa'

#: the rubric; travels inside every verdicts file (see RUBRIC_VERSION)
RUBRIC_VERSION = 1
RUBRIC = """\
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


def choose_views(cands, k=VIEWS_PER_CORNER, min_m=MIN_VIEW_M, max_m=MAX_VIEW_M):
    """The k views to show for a corner: nearest first among those >= min_m away, topped up
    with nearer ones only when fewer than k qualify. cands: dicts with dist_m and pano_id."""
    ok = sorted((c for c in cands if c['dist_m'] <= max_m),
                key=lambda c: (c['dist_m'], c['pano_id']))
    far = [c for c in ok if c['dist_m'] >= min_m]
    near = [c for c in ok if c['dist_m'] < min_m]
    near.sort(key=lambda c: (-c['dist_m'], c['pano_id']))
    return (far + near)[:k]


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
    inside the window (+1 m); (None, None) when there is none."""
    best = None
    for lat, lng, c in inv.get(uid, []):
        if c != cls:
            continue
        d = geo.haversine_m(lat0, lng0, lat, lng)
        if d <= window_m + 1.0 and (best is None or d < best[0]):
            best = (d, lat, lng)
    return (best[1], best[2]) if best else (None, None)


# ------------------------------------------------------------------------------- items

def build_item(rec, part, panos, inv_pos):
    """One review unit: corners with their views and (hidden until reveal) inventory."""
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
        views = choose_views(cands)
        for v in views:
            v['crop'] = f"crops/{safe_name(unit)}_c{c['corner']}_{v['pano_id']}.jpg"
        corners.append({
            'corner': c['corner'], 'start_deg': c['start_deg'], 'width_deg': c['width_deg'],
            'wide': c['wide'], 'lat': c['lat'], 'lng': c['lng'],
            'n_panos_25': c['n_panos_25'], 'views': views,
            'inv_counts': c['inv_counts'],
            'inventory': [dict(x, **dict(zip(('lat', 'lng'), inventory_position(
                inv_pos, x['unit_id'], x['class'], rec['lat'], rec['lng']))))
                for x in c['inventory']]})
    return {'unit': unit, 'city': CITY, 'part': part, 'type': rec['type'],
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
                out.setdefault(v['run'], []).append(
                    {'pano_id': v['pano_id'], 'x': v['x'], 'y': v['y'], 'fov_deg': CROP_FOV_DEG,
                     'px': CROP_PX, 'name': name,
                     'path': f"{v['pano_id'][:2]}/{v['pano_id']}.jpg"})
    return out


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
              f"- Views: up to {VIEWS_PER_CORNER} panos per corner, nearest first among "
              f"those >= {MIN_VIEW_M:g} m from the corner point (nearer ones only to top up), "
              f"from the panos within 25 m of the unit centre or the corner point. Crops are "
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
    t0 = time.time()
    bundle = Path(args.bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    build = json.loads(Path(args.build_json).read_text(encoding='utf-8'))
    inputs = {}

    def rec_input(name, path, want=None):
        h = sha256_file(path)
        if want and h != want:
            raise SystemExit(f'{path}: sha256 {h[:12]} is not the recorded {want[:12]}')
        inputs[name] = {'path': str(Path(path).resolve()), 'sha256': h}

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
    items = [build_item(by_unit[u], part_of[u], panos, inv_pos) for u in order]

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
        'built_at': utc_now(), 'tool_git_sha': git_sha(), 'seed': args.seed,
        'n_na': args.n_na, 'n_clean': args.n_clean, 'arm_state': ARM_STATE,
        'populations': sizes, 'draw': drawn, 'review_order': order,
        'items_sha256': sha256_file(items_path), 'inputs': inputs,
        'views': {'per_corner': VIEWS_PER_CORNER, 'min_m': MIN_VIEW_M, 'max_m': MAX_VIEW_M,
                  'camera_height_m': CAMERA_HEIGHT_M, 'crop_fov_deg': CROP_FOV_DEG,
                  'crop_px': CROP_PX, 'store': f'makelab2 {STORE}'},
        'aerial': {'source': 'Esri World Imagery', 'url_template': sf.TILE_URL,
                   'attribution': sf.ATTRIBUTION},
        'rubric_version': RUBRIC_VERSION, 'rubric': RUBRIC}

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
            missing |= set(miss)
    missing = {v['crop'] for it in items for c in it['corners'] for v in c['views']
               if not (bundle / v['crop']).exists()}
    snapshot['crops_missing'] = sorted(missing)
    snapshot['seconds'] = round(time.time() - t0, 1)
    write_text_lf(bundle / 'snapshot.json', json.dumps(snapshot, indent=1) + '\n')
    write_report(bundle, snapshot, items, missing)
    print(f"{len(items)} units ({', '.join(f'{p} {len(drawn[p])}' for p in PARTS)}), "
          f"{sum(len(i['corners']) for i in items)} corners; {len(missing)} crops missing")
    print(f'wrote {items_path}, snapshot.json, report.md')
    return 0


# ------------------------------------------------------------------------------ render

def viewer_unit(it, rel):
    """The per-unit payload the page needs; paths relative to the gallery dir. The part is
    deliberately left out: the rater never sees which population a unit came from."""
    return {'id': it['unit'], 'type': it['type'], 'centre': it['centre'],
            'window_m': it['window_m'], 'legs': it['legs'],
            'aerial': dict(it['aerial'], file=rel + it['aerial']['file']),
            'corners': [{'k': str(c['corner']), 'corner': c['corner'],
                         'start_deg': c['start_deg'], 'width_deg': c['width_deg'],
                         'wide': c['wide'], 'lat': c['lat'], 'lng': c['lng'],
                         'inventory': c['inventory'],
                         'views': [{'pano_id': v['pano_id'], 'date': v['capture_date'],
                                    'dist_m': v['dist_m'], 'cam': v['cam'],
                                    'crop': rel + v['crop']} for v in c['views']]}
                        for c in it['corners']]}


def load_prefill(bundle, rater, items_sha):
    """(verdicts or None, message): a file made on another item list is not loaded."""
    path = Path(bundle) / rater_file_name(rater)
    if not path.exists():
        return None, None
    v = json.loads(path.read_text(encoding='utf-8'))
    if v.get('items_sha256') != items_sha:
        return None, (f'NOT prefilled from {path}: made on items {str(v.get("items_sha256"))[:12]},'
                      f' the bundle is {items_sha[:12]}')
    return v, f'prefilled {len(v.get("units") or {})} units from {path} for revision'


def build_html(units, items_sha, rater, initial, attribution):
    from corner_gallery_page import HTML_TEMPLATE, STATE_BOOTSTRAP_JS
    return (HTML_TEMPLATE
            .replace('__STATE_BOOTSTRAP__', STATE_BOOTSTRAP_JS)
            .replace('__UNITS__', json.dumps(units))
            .replace('__ITEMS_SHA__', json.dumps(items_sha))
            .replace('__CITY__', json.dumps(CITY))
            .replace('__RATER__', json.dumps(rater))
            .replace('__RUBRIC_V__', str(RUBRIC_VERSION))
            .replace('__RUBRIC_JSON__', json.dumps(RUBRIC))
            .replace('__RUBRIC_HTML__', RUBRIC.replace('&', '&amp;').replace('<', '&lt;'))
            .replace('__SCHEMA__', json.dumps(VERDICTS_SCHEMA))
            .replace('__INITIAL__', json.dumps(initial))
            .replace('__ATTRIBUTION__', json.dumps(attribution))
            .replace('__FILE_NAME__', json.dumps(rater_file_name(rater))))


def cmd_render(args):
    bundle = Path(args.bundle)
    snapshot = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
    items_sha = sha256_file(bundle / 'items.jsonl')
    if items_sha != snapshot['items_sha256']:
        raise SystemExit('items.jsonl does not match snapshot.json')
    items = read_jsonl(bundle / 'items.jsonl')
    out = Path(args.out) if args.out else bundle / 'gallery' / args.rater
    out.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(bundle.resolve(), out.resolve()).replace(os.sep, '/') + '/'
    initial, msg = load_prefill(bundle, args.rater, items_sha)
    if msg:
        print(msg)
    html = build_html([viewer_unit(i, rel) for i in items], items_sha, args.rater, initial,
                      snapshot['aerial']['attribution'])
    write_text_lf(out / 'index.html', html)
    print(f"{len(items)} units, {sum(len(i['corners']) for i in items)} corners")
    print(f"Gallery: {out / 'index.html'}")
    print(f'Open it, rate, Export, and save the download as {bundle / rater_file_name(args.rater)}')
    return 0


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
    r = sub.add_parser('render', help='write the review page for one rater')
    r.add_argument('--bundle', type=Path, required=True)
    r.add_argument('--rater', required=True, help='exports verdicts__<rater>.json')
    r.add_argument('--out', type=Path, default=None, help='default <bundle>/gallery/<rater>')
    args = ap.parse_args(argv)
    return {'build': cmd_build, 'render': cmd_render}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
