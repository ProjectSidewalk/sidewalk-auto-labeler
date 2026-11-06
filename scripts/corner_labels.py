"""NoCurbRamp label proposals from the corner inventory: one label per absent corner (issue #169).

The corner table of RampNet#238 / #241 (scripts/corner_inventory.py, merged build) says, per
OSM intersection corner, whether a fused ramp site sits in it (present), the corner was seen
from a pano within 25 m and has none (absent), or nothing saw it (unobservable). This tool
turns the absent corners into NoCurbRamp labels in the record format send_to_ps.py submits.
There is no model: the "detector" is the rule below, so the registry's NoCurbRamp spec has no
module and every proposal carries confidence 1.0 (detectors/registry.py).

Subcommands:

  build   corners_full.jsonl + the run dirs -> <out>/proposals.jsonl, build.json, counts.csv,
          unlabeled.csv. One proposal per corner that is absent under --arm/--variant and not
          excluded by the inventory under --na-read; the pano with the clearest view of the
          corner point is chosen (choose_pano) and the point is projected into it with
          geo.ground_point_to_pano.
  rank    join validator-ai's ranked CSV (sidewalk-validator-ai#69) on PS's pixel key ->
          proposals.ranked.jsonl + rank.json. Orders a lead's review; drops nothing.
  export  proposals -> a results.jsonl with label_type NoCurbRamp that send_to_ps.py accepts:
          one line per pano, one detection per corner, the pano block copied from the run.
  score   the pre-registered corner-level rule against blind gallery ratings -> score JSON.
  sample  the seeded >= 100-corner sample a future gallery rates -> sample.csv + sample.json.

The pre-registered rule (issue #169, stated before any label was built): corner-level absence
precision against blind ratings >= 0.90 with Wilson lower bound >= 0.85, on a seeded sample of
>= 100 absent corners drawn under the primary NA read; `cant_tell` corners leave the
denominator and their share is reported. The primary NA read is NOT settled yet (gallery 243,
RampNet#243, is half rated), so --na-read has no default. Write-up: docs/corner-labels-169.md.

Inputs are read in place and never written; their sha256 goes into build.json, and for
Vancouver the build refuses inputs that differ from the ones #241 / gallery 243 were built on
(PINNED_SHA256). Stdlib + the repo's own modules (shapely only behind --sidewalk-presence).

    python scripts/corner_labels.py build --na-read discount \\
        --corners ../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl \\
        --run-dir ../../sal-vancouver/runs/vancouver \\
        --extra-run ../posdetect241/runs/vancouver_posdetect241 \\
        --extra-run ../posdetect241/runs/vancouver_posdetect241_gsv \\
        --out runs/vancouver/corner_labels169/na_discount
    python scripts/corner_labels.py export --dir runs/vancouver/corner_labels169/na_discount
    python send_to_ps.py runs/vancouver/corner_labels169/na_discount/results.jsonl --dry-run
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import corner_inventory as ci  # noqa: E402
import corner_gallery as cg  # noqa: E402
from detectors import API_VERSION, ps_pixel, ps_training_date  # noqa: E402
from detectors import registry  # noqa: E402

PROPOSAL_SCHEMA = 'sidewalk-auto-labeler.corner_labels.proposal/1'
BUILD_SCHEMA = 'sidewalk-auto-labeler.corner_labels.build/1'
LABEL_TYPE = registry.NO_CURB_RAMP
#: the rule's version and the date it was written; model_training_date of every label
RULE_VERSION = 1
RULE_VERSION_DATE = '2026-11-05'
RULE_REPO = 'ProjectSidewalk/sidewalk-auto-labeler'
CONFIDENCE = 1.0
NA_READS = ('discount', 'count')
STRATA = ci.INTERSECTION_STRATA
TABLE_OBS_M = 25.0          # the corner table's pano_ids_25 radius: the candidate pool
MAX_RANGE_M = 25.0
MIN_VIEW_M = cg.MIN_VIEW_M  # 3 m: nearer, the pano looks almost straight down
TOP_K = cg.VIEWS_PER_CORNER
CAMERA_HEIGHT_M = cg.CAMERA_HEIGHT_M   # 2.5 m, the fixed height of the #241 fusion
N_RUNNER_UPS = 4
SITE_SEARCH_M = 50.0
SIDEWALK_RADIUS_M = 10.0
#: the view test's half-angle about camera_heading; None = no view test (the default since
#: 2026-11-05; the first build used 90, the forward hemisphere). --hemisphere-deg opts in.
HEMISPHERE_DEG = None
RULE_PRECISION = 0.90
RULE_WILSON_LO = 0.85
RULE_MIN_N = 100
SAMPLE_SEED = 169
SAMPLE_N = 100

#: Inputs #241's merged corner build and gallery 243 were made from (runs/vancouver/
#: corner_inventory_posdetect241/build.json `inputs`; corners_full from runs/vancouver/
#: corner_gallery243/snapshot.json). A build on other bytes is refused unless --allow-unpinned.
PINNED_SHA256 = {'vancouver': {
    'corners_full': 'ec20606098d0e8b34113b4b328e57d3737b7669d01a37f1d6d722faab4681f8d',
    'results': '7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28',
    'sites': '9d1fc51ac2867d6247bdf2e0b1c6b5c9fccbbf9a566771f0222bb2728055c914',
    'sites_meta': '5a72734079b30710343b8dda2e391145be0721114509015dc963d69e7d873678',
    'extra:vancouver_posdetect241:results':
        '69f46b26110df05d1ef52a4431c4b230ceb157941b212fd3d553fd64bb86bf0e',
    'extra:vancouver_posdetect241:sites':
        'd38c6fa08f9774de842c77c72ad51346daebb918270c90b319b662bd8534320b',
    'extra:vancouver_posdetect241:sites_meta':
        '5408f770ebe84943faf1048a9b8978fe0700997e0ea57b699110c2a5b6e626b9',
    'extra:vancouver_posdetect241_gsv:results':
        '953716d1d7f32f2680c101fb9a4d7ab0933939459704fbc62375042768ca57b2',
    'extra:vancouver_posdetect241_gsv:sites':
        'df0dec5c6c9456248a7ee1aee7c4e3e7c4c50643780a5105dd7451b3e7e7614d',
    'extra:vancouver_posdetect241_gsv:sites_meta':
        'ddae3b0e42d10dd89441cd3e6fff8925ad2e1cc42fb4e3cc388c67c9534a321d',
}}

r6, r2 = ci.r6, ci.r2


def corner_key(unit, corner):
    """'vancouver:res:n47150161#2': the one key a corner goes by in every output here."""
    return f'{unit}#{corner}'


# ------------------------------------------------------------------------ the rule

def inventory_excludes(inv_counts, na_read):
    """Whether the city inventory forbids a NoCurbRamp label at a corner called absent.

    na_read 'count' (corner-inventory read (a), "as written"): ANY inventory point excludes
    the corner, `NA` points included -- the city has recorded something there, so a ramp may
    be there. Only corners with no inventory point at all are labeled.

    na_read 'discount' (read (b) of amendment A8): `NA` points with no `RAMPTYPE`
    (class NA_noramp) are read as the city's own record of a corner without a ramp and do
    not exclude; every other class does (Available, NA_typed -- an `NA` with a ramp type --,
    RMV, Expired/Removed, other). This is exactly read (b), "no inventory point of any status,
    not counting NA points with no RAMPTYPE", the read the issue quotes at 0.965.

    Which of the two is primary is NOT settled: gallery 243 (RampNet#243) decides it.

    >>> inventory_excludes({'Available': 0, 'NA_noramp': 2}, 'discount')
    False
    >>> inventory_excludes({'Available': 0, 'NA_noramp': 2}, 'count')
    True
    >>> inventory_excludes({'NA_typed': 1}, 'discount')
    True
    """
    if na_read not in NA_READS:
        raise ValueError(f'na_read must be one of {NA_READS}, got {na_read!r}')
    if na_read == 'count':
        return any(v > 0 for v in inv_counts.values())
    return any(v > 0 for k, v in inv_counts.items() if k != 'NA_noramp')


def inventory_status(inv_counts):
    """'none', 'na_noramp_only' or 'other' (any other class present)."""
    held = {k for k, v in inv_counts.items() if v > 0}
    if not held:
        return 'none'
    return 'na_noramp_only' if held == {'NA_noramp'} else 'other'


def candidate_views(lat, lng, pano_ids, panos, max_range_m=MAX_RANGE_M,
                    min_view_m=MIN_VIEW_M, hemisphere_deg=HEMISPHERE_DEG):
    """(candidates, rejected) for one corner point among the panos that observe it.

    A candidate is a pano within max_range_m of the point and at least min_view_m from it, in
    any direction: a GSV equirect sees behind the car at the same resolution as ahead, and PS's
    pixel key wraps x modulo the width, so a point behind the camera is as labelable as one
    ahead (Jon Froehlich, 2026-11-05, reversing the first build's forward-hemisphere test).
    hemisphere_deg (opt-in, default None = off) restores a view test: the bearing from the
    camera to the point must lie within hemisphere_deg of the camera heading (90 is the old
    forward hemisphere, |x - 0.5| <= 0.25 in the pano). Distance and bearing are measured in a
    LocalFrame at the camera, as geo.ground_point_to_pano measures them, so a candidate is
    always projectable. rejected counts {no_pose, too_far, too_near, behind}; `behind` stays 0
    with the test off. panos: {pano_id: {lat, lng, heading, capture_date, ...}}.
    """
    cands, rejected = [], {'no_pose': 0, 'too_far': 0, 'too_near': 0, 'behind': 0}
    for pid in sorted(set(pano_ids)):
        p = panos.get(pid)
        if p is None or p.get('heading') is None:
            rejected['no_pose'] += 1
            continue
        e, n = geo.LocalFrame(p['lat'], p['lng']).to_enu(lat, lng)
        d = math.hypot(e, n)
        bearing = math.degrees(math.atan2(e, n)) % 360.0
        off = abs(geo.norm_deg(bearing - p['heading']))
        if d > max_range_m:
            rejected['too_far'] += 1
        elif d < min_view_m:
            rejected['too_near'] += 1
        elif hemisphere_deg is not None and off > hemisphere_deg:
            rejected['behind'] += 1
        else:
            cands.append({'pano_id': pid, 'dist_m': d, 'bearing_deg': bearing,
                          'off_axis_deg': off, 'capture_date': p.get('capture_date')})
    return cands, rejected


def choose_pano(cands, top_k=TOP_K):
    """(chosen, runner-ups) among candidate_views' candidates: "the pano with the clearest
    view". Candidates are ranked nearest first (corner_gallery._distance_order); among the
    top_k nearest, the newest capture date wins (corner_gallery.newest_date), nearest first
    within it. So a recent pano a little farther away beats an old one at the corner, but
    the label never goes on a pano far down the ranking. corner_gallery.choose_views applies
    the newest-date slot to the WHOLE 40 m pool, because a rater sees three crops; a label
    sits on one pano, so here it is confined to the top_k. Runner-ups are the rest of the
    ranking, nearest first. (None, []) when there is no candidate.

    >>> c = [{'pano_id': 'a', 'dist_m': 5, 'capture_date': '2014-08'},
    ...      {'pano_id': 'b', 'dist_m': 8, 'capture_date': '2021-12'},
    ...      {'pano_id': 'c', 'dist_m': 20, 'capture_date': '2023-05'}]
    >>> choose_pano(c)[0]['pano_id'], choose_pano(c, top_k=1)[0]['pano_id']
    ('c', 'a')
    """
    order = cg._distance_order(cands, 0.0)
    if not order:
        return None, []
    top = order[:max(1, top_k)]
    newest = cg.newest_date(top)
    chosen = next(c for c in top if (c.get('capture_date') or '') == newest)
    return chosen, [c for c in order if c is not chosen]


def project(pano, lat, lng, camera_height_m=CAMERA_HEIGHT_M, max_range_m=MAX_RANGE_M):
    """geo.ground_point_to_pano for a pano-index entry (flat pose, fixed height), or None."""
    pose = geo.pano_pose({'lat': pano['lat'], 'lng': pano['lng'],
                          'camera_heading': pano['heading'], 'camera_pitch': None,
                          'camera_roll': None, 'source': pano.get('source') or ''})
    return geo.ground_point_to_pano(pose, lat, lng, camera_height=camera_height_m,
                                    max_range_m=max_range_m)


# ------------------------------------------------------------------------- loading

def load_pano_index(results_paths, wanted=None):
    """{pano_id: {lat, lng, heading, capture_date, width, height, source, run}}; the first line
    of a pano wins. corner_gallery.load_pano_index without width/height, which the PS pixel
    key needs, hence this copy. results_paths: [(run name, path)]."""
    out = {}
    for name, path in results_paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)['pano']
                pid = p['panorama_id']
                if (wanted is not None and pid not in wanted) or pid in out:
                    continue
                h = p.get('camera_heading')
                out[pid] = {'lat': p['lat'], 'lng': p['lng'],
                            'heading': None if h is None else geo.norm_deg(float(h)) % 360.0,
                            'capture_date': p.get('capture_date'), 'width': p.get('width'),
                            'height': p.get('height'), 'source': p.get('source'), 'run': name}
    return out


def load_pano_blocks(results_paths, wanted):
    """{(run, pano_id): the full pano block} for each wanted (run, pano_id), from the first
    line holding that pano in THAT run's results.jsonl. results_paths: {run name: path}."""
    out = {}
    by_run = {}
    for run, pid in wanted:
        by_run.setdefault(run, set()).add(pid)
    for run, pids in sorted(by_run.items(), key=lambda t: str(t[0])):
        if run not in results_paths:
            raise SystemExit(f'proposals name run {run!r}, which build.json does not record '
                             f'(runs: {sorted(results_paths)})')
        with open(results_paths[run], encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec['pano']['panorama_id']
                if pid in pids and (run, pid) not in out:
                    out[(run, pid)] = rec['pano']
    return out


def run_results_paths(build):
    """{run name: input record} from build.json: the base run is the `results` role (named
    after its run dir, as build names it in the pano index), each extra
    `extra:<name>:results`."""
    out = {}
    for role, rec in build['inputs'].items():
        if role == 'results':
            out[Path(rec['path']).resolve().parent.name] = rec
        elif role.startswith('extra:') and role.endswith(':results'):
            out[role[len('extra:'):-len(':results')]] = rec
    return out


class SiteIndex:
    """Nearest operational fused site to a point, within SITE_SEARCH_M."""

    def __init__(self, sites, search_m=SITE_SEARCH_M):
        self.search_m = search_m
        lats = [s[0] for s in sites] or [0.0]
        lngs = [s[1] for s in sites] or [0.0]
        self.frame = geo.LocalFrame(sum(lats) / len(lats), sum(lngs) / len(lngs))
        self.grid = geo.GridIndex(search_m)
        for lat, lng, payload in sites:
            e, n = self.frame.to_enu(lat, lng)
            self.grid.add(e, n, (lat, lng, payload['site_id']))

    def nearest(self, lat, lng):
        """{'site_id', 'dist_m'} or None when no site is within search_m."""
        e, n = self.frame.to_enu(lat, lng)
        best = None
        for slat, slng, sid in self.grid.near(e, n):
            d = geo.haversine_m(lat, lng, slat, slng)
            if d <= self.search_m and (best is None or (d, str(sid)) < best):
                best = (d, str(sid))
        return None if best is None else {'site_id': best[1], 'dist_m': r2(best[0])}


class SidewalkPresence:
    """Guide rule 3, "no label where no sidewalk reaches the corner": a corner has a sidewalk
    when any feature of the given GeoJSON (sidewalk lines or polygons, e.g. a city's
    sidewalk-presence layer, SidewalkWebpage#5279) lies within radius_m of its corner point.
    shapely is imported only here, and must be >= 2 (requirements.txt pins it): STRtree.query
    then returns integer indices into the tree's geometries, which has_sidewalk relies on
    (shapely 1.x returned the geometries themselves). Contract: docs/corner-labels-169.md.

    keep ({field: {values}}, from --sidewalk-keep) keeps only the features whose property
    holds one of the values, e.g. {'STATUS': {'Available', 'Pathway', 'Private'}} for a city
    layer that also digitizes street sides WITHOUT a sidewalk (Vancouver WA's STATUS 'NA').
    n_features counts the kept features, n_features_in_file all of them."""

    def __init__(self, path, radius_m=SIDEWALK_RADIUS_M, keep=None):
        from shapely.geometry import Point, shape
        from shapely.ops import transform
        from shapely.strtree import STRtree
        import shapely
        if int(shapely.__version__.split('.')[0]) < 2:
            raise SystemExit(f'--sidewalk-presence needs shapely >= 2, found {shapely.__version__}')
        feats = json.loads(Path(path).read_text(encoding='utf-8'))['features']
        self.n_features_in_file = len(feats)
        if keep:
            feats = [f for f in feats if all((f.get('properties') or {}).get(k) in vals
                                             for k, vals in keep.items())]
        geoms = [shape(f['geometry']) for f in feats if f.get('geometry')]
        if not geoms:
            raise SystemExit(f'{path}: no sidewalk geometry'
                             + (f' after --sidewalk-keep {keep}' if keep else ''))
        c = geoms[0].representative_point()
        self.frame = geo.LocalFrame(c.y, c.x)
        self.geoms = [transform(lambda xs, ys: _enu_coords(self.frame, xs, ys), g)
                      for g in geoms]
        self.tree = STRtree(self.geoms)
        self.radius_m = radius_m
        self._Point = Point
        self.n_features = len(geoms)

    def has_sidewalk(self, lat, lng):
        e, n = self.frame.to_enu(lat, lng)
        pt = self._Point(e, n)
        for i in self.tree.query(pt.buffer(self.radius_m)):
            if self.geoms[int(i)].distance(pt) <= self.radius_m:
                return True
        return False


def parse_keep(specs):
    """--sidewalk-keep 'FIELD=V1,V2' (repeatable) -> {FIELD: {V1, V2}}; None when none given.

    >>> sorted(parse_keep(['STATUS=Available,Pathway'])['STATUS'])
    ['Available', 'Pathway']
    """
    if not specs:
        return None
    keep = {}
    for spec in specs:
        field, sep, vals = spec.partition('=')
        if not sep or not field or not vals:
            raise SystemExit(f'--sidewalk-keep {spec!r}: expected FIELD=V1,V2')
        keep.setdefault(field, set()).update(v for v in vals.split(',') if v)
    return keep


def _enu_coords(frame, xs, ys):
    """shapely.ops.transform callback: lng/lat sequences -> east/north sequences."""
    es, ns = [], []
    for x, y in zip(xs, ys):
        e, n = frame.to_enu(y, x)
        es.append(e)
        ns.append(n)
    return es, ns


def check_pins(city, inputs, allow_unpinned):
    """Compare every input's sha256 with PINNED_SHA256[city]; a mismatch exits unless
    allow_unpinned. Returns 'pinned', 'unpinned city' or 'overridden: <roles>'."""
    pins = PINNED_SHA256.get(city)
    if not pins:
        return 'unpinned city'
    bad = sorted(role for role, rec in inputs.items()
                 if role in pins and rec['sha256'] != pins[role])
    missing = sorted(set(pins) - set(inputs))
    if bad or missing:
        msg = f'inputs differ from the pinned {city} build: changed {bad}, missing {missing}'
        if not allow_unpinned:
            raise SystemExit(msg + ' (--allow-unpinned to build anyway; recorded)')
        print(f'WARNING (--allow-unpinned): {msg}')
        return f'overridden: changed {bad}, missing {missing}'
    return 'pinned'


# --------------------------------------------------------------------------- build

def build_proposals(units, panos, sites, args, sidewalks=None):
    """(proposals, unlabeled rows, counts by stratum) over the corner table's records.

    Every intersection corner is counted; one that is absent under args.arm/args.variant
    either becomes a proposal or lands in `unlabeled` with its reason: inventory (excluded by
    the NA read), no_sidewalk (only with a sidewalk layer), no_candidate (no pano passes
    candidate_views) or projection_dropped (geo.ground_point_to_pano returned None)."""
    key = f'{args.arm}/{args.variant}'
    fields = ('corners', 'absent', 'excluded_inventory', 'dropped_no_sidewalk', 'no_candidate',
              'dropped_projection', 'proposed')
    counts = {s: dict.fromkeys(fields, 0) for s in STRATA}
    proposals, unlabeled = [], []
    for rec in units:
        if rec['type'] not in STRATA:
            continue
        cnt = counts[rec['type']]
        for c in rec.get('corners') or ():
            cnt['corners'] += 1
            if c['state'][key] != 'absent':
                continue
            cnt['absent'] += 1
            row = {'unit': rec['unit'], 'corner': c['corner'], 'stratum': rec['type'],
                   'inventory_status': inventory_status(c['inv_counts'])}
            if inventory_excludes(c['inv_counts'], args.na_read):
                cnt['excluded_inventory'] += 1
                unlabeled.append(dict(row, reason='inventory'))
                continue
            if sidewalks is not None and not sidewalks.has_sidewalk(c['lat'], c['lng']):
                cnt['dropped_no_sidewalk'] += 1
                unlabeled.append(dict(row, reason='no_sidewalk'))
                continue
            cands, rejected = candidate_views(c['lat'], c['lng'], c.get('pano_ids_25') or (),
                                              panos, args.max_range_m, args.min_view_m,
                                              getattr(args, 'hemisphere_deg', HEMISPHERE_DEG))
            chosen, runners = choose_pano(cands, args.top_k)
            if chosen is None:
                cnt['no_candidate'] += 1
                unlabeled.append(dict(row, reason='no_candidate'))
                continue
            pano = panos[chosen['pano_id']]
            proj = project(pano, c['lat'], c['lng'], args.camera_height_m, args.max_range_m)
            if proj is None:
                cnt['dropped_projection'] += 1
                unlabeled.append(dict(row, reason='projection_dropped'))
                continue
            cnt['proposed'] += 1
            proposals.append(proposal_record(rec, c, pano, chosen, runners, cands, rejected,
                                             proj, sites.nearest(c['lat'], c['lng']), args))
    proposals.sort(key=lambda p: (p['unit'], p['corner']))
    unlabeled.sort(key=lambda r: (r['unit'], r['corner']))
    return proposals, unlabeled, counts


def proposal_record(rec, c, pano, chosen, runners, cands, rejected, proj, site, args):
    """One proposal line: the corner, the chosen pano and pixel, and the evidence."""
    return {
        'schema': PROPOSAL_SCHEMA, 'key': corner_key(rec['unit'], c['corner']),
        'unit': rec['unit'], 'corner': c['corner'], 'stratum': rec['type'],
        'lat': r6(c['lat']), 'lng': r6(c['lng']), 'label_type': LABEL_TYPE,
        'pano': {'pano_id': chosen['pano_id'], 'run': pano['run'], 'lat': r6(pano['lat']),
                 'lng': r6(pano['lng']), 'heading_deg': r2(pano['heading']),
                 'capture_date': pano.get('capture_date'), 'width': pano.get('width'),
                 'height': pano.get('height')},
        'x_normalized': r6(proj.x_norm), 'y_normalized': r6(proj.y_norm),
        'range_m': r2(proj.range_m), 'off_axis_deg': r2(chosen['off_axis_deg']),
        'rule': {'version': RULE_VERSION, 'arm': args.arm, 'variant': args.variant,
                 'na_read': args.na_read, 'camera_height_m': args.camera_height_m},
        'source': {
            'observing_panos': sorted(c.get('pano_ids_25') or ()),
            'n_panos_25': c.get('n_panos_25'), 'n_panos_15': c.get('n_panos_15'),
            'nearest_pano_m': c.get('nearest_pano_m'),
            'nearest_site': site,
            'inventory': {'status': inventory_status(c['inv_counts']),
                          'counts': c['inv_counts'],
                          'points': sorted(([p.get('unit_id'), p.get('class')]
                                            for p in c.get('inventory') or ()),
                                           key=lambda x: (str(x[0] or ''), str(x[1] or '')))},
            'states': c['state'],
            'sector': {'start_deg': c.get('start_deg'), 'width_deg': c.get('width_deg'),
                       'wide': c.get('wide')},
            'n_candidates': len(cands), 'rejected': rejected,
            'runner_ups': [{'pano_id': x['pano_id'], 'dist_m': r2(x['dist_m']),
                            'capture_date': x.get('capture_date'),
                            'off_axis_deg': r2(x['off_axis_deg'])}
                           for x in runners[:N_RUNNER_UPS]],
            'chosen_dist_m': r2(chosen['dist_m']),
        },
    }


COUNT_FIELDS = ['stratum', 'corners', 'absent', 'excluded_inventory', 'dropped_no_sidewalk',
                'no_candidate', 'dropped_projection', 'proposed']


def count_rows(counts):
    rows = [dict(stratum=s, **counts[s]) for s in STRATA]
    rows.append(dict(stratum='intersections',
                     **{k: sum(counts[s][k] for s in STRATA) for k in counts[STRATA[0]]}))
    return rows


def tool_version():
    """{'file', 'git_sha': the repo's HEAD commit, 'dirty': uncommitted edits to tracked files}.

    HEAD, not this file's last commit: the labels' model_id names the code that built them,
    and that includes geo.py, corner_inventory.py, corner_gallery.py and the registry, so the
    whole tree has to be at that commit. Untracked files and everything under runs/ are
    ignored: those are outputs, not code, and a rebuild overwrites the tracked outputs of the
    previous one (build.json, counts.csv, ...), which must not read as an edit to the code."""
    import subprocess
    try:
        st = subprocess.run(['git', '-C', str(REPO_ROOT), 'status', '--porcelain',
                             '--untracked-files=no', '--', '.', ':(exclude)runs'],
                            capture_output=True, text=True,
                            timeout=30).stdout
        dirty = bool(st.strip())
    except Exception:  # pragma: no cover
        dirty = None
    return {'file': 'scripts/corner_labels.py', 'git_sha': ci.git_sha(), 'dirty': dirty}


def file_record(path):
    """{path, sha256, lines, bytes} of an output file."""
    data = Path(path).read_bytes()
    return {'path': Path(path).name, 'sha256': hashlib.sha256(data).hexdigest(),
            'lines': data.count(b'\n'), 'bytes': len(data)}


def cmd_build(args):
    t0 = time.time()
    if args.max_range_m > TABLE_OBS_M:
        raise SystemExit(f'--max-range-m {args.max_range_m} > {TABLE_OBS_M}: the corner table '
                         f'lists only the panos within {TABLE_OBS_M:g} m (pano_ids_25)')
    if args.hemisphere_deg is not None and not 0.0 < args.hemisphere_deg <= 180.0:
        raise SystemExit(f'--hemisphere-deg {args.hemisphere_deg}: must be in (0, 180]')
    run_dir = Path(args.run_dir)
    inputs = {'corners_full': Path(args.corners), 'results': run_dir / 'results.jsonl',
              'sites': run_dir / 'sites.jsonl', 'sites_meta': run_dir / 'sites_meta.json'}
    for extra in args.extra_run or ():
        name = Path(extra).resolve().name
        for f in ('results', 'sites', 'sites_meta'):
            inputs[f'extra:{name}:{f}'] = Path(extra) / (f + ('.json' if f == 'sites_meta'
                                                              else '.jsonl'))
    if args.sidewalk_presence:
        inputs['sidewalk_presence'] = Path(args.sidewalk_presence)
    for role, p in inputs.items():
        if not p.exists():
            raise SystemExit(f'{role}: {p} does not exist')
    in_rec = {role: {'path': str(p), 'sha256': ci.sha256_file(p)} for role, p in inputs.items()}
    pins = check_pins(args.city, in_rec, args.allow_unpinned)

    units = ci.read_jsonl(inputs['corners_full'])
    sites, base_meta = ci.load_sites(run_dir)
    base_ids = {p[2] for p in ci.load_panos(inputs['results'])}
    extra_recs = []
    results_paths = [(run_dir.resolve().name, inputs['results'])]
    for extra in args.extra_run or ():
        _panos, esites, _dep, erec = ci.load_extra_run(extra, base_meta, base_ids)
        sites += esites
        extra_recs.append(erec)
        results_paths.append((erec['name'], Path(extra) / 'results.jsonl'))
    wanted = {pid for u in units for c in u.get('corners') or () for pid in
              c.get('pano_ids_25') or ()}
    panos = load_pano_index(results_paths, wanted)
    keep = parse_keep(getattr(args, 'sidewalk_keep', None))
    if keep and not args.sidewalk_presence:
        raise SystemExit('--sidewalk-keep needs --sidewalk-presence')
    sidewalks = (SidewalkPresence(args.sidewalk_presence, args.sidewalk_radius_m, keep)
                 if args.sidewalk_presence else None)

    proposals, unlabeled, counts = build_proposals(units, panos, SiteIndex(sites), args,
                                                   sidewalks)
    out = Path(args.out)
    ci.write_jsonl_lf(out / 'proposals.jsonl', proposals)
    ci.write_csv_lf(out / 'unlabeled.csv',
                    ['unit', 'corner', 'stratum', 'inventory_status', 'reason'], unlabeled)
    ci.write_csv_lf(out / 'counts.csv', COUNT_FIELDS, count_rows(counts))
    total = count_rows(counts)[-1]
    build = {
        'schema': BUILD_SCHEMA, 'city': args.city, 'issue': 'sidewalk-auto-labeler#169',
        'built_at': ci.utc_now(), 'tool': tool_version(), 'python': platform.python_version(),
        'label_type': LABEL_TYPE, 'rule_version': RULE_VERSION,
        'rule_version_date': RULE_VERSION_DATE,
        'params': {'arm': args.arm, 'variant': args.variant, 'na_read': args.na_read,
                   'na_read_settled': False, 'max_range_m': args.max_range_m,
                   'min_view_m': args.min_view_m, 'top_k': args.top_k,
                   'camera_height_m': args.camera_height_m, 'hemisphere_deg': args.hemisphere_deg,
                   'site_search_m': SITE_SEARCH_M, 'n_runner_ups': N_RUNNER_UPS},
        'sidewalk_rule': ({'applied': True, 'radius_m': args.sidewalk_radius_m,
                           'path': str(inputs['sidewalk_presence']),
                           'sha256': in_rec['sidewalk_presence']['sha256'],
                           'keep': ({k: sorted(v) for k, v in keep.items()} if keep else None),
                           'n_features_in_file': sidewalks.n_features_in_file,
                           'n_features': sidewalks.n_features}
                          if sidewalks else 'not applied'),
        'per_direction_arm': 'not built (no crossing-line source in the run dirs)',
        'inputs': in_rec, 'pins': pins, 'extra_runs': extra_recs,
        'counts': {'units': len(units), 'panos_indexed': len(panos), 'operational_sites': len(sites),
                   **{k: total[k] for k in COUNT_FIELDS[1:]}},
        'outputs': {'proposals': file_record(out / 'proposals.jsonl'),
                    'unlabeled': file_record(out / 'unlabeled.csv'),
                    'counts': file_record(out / 'counts.csv')},
        'seconds': round(time.time() - t0, 1),
    }
    ci.write_text_lf(out / 'build.json', json.dumps(build, indent=1) + '\n')
    print(f"{total['corners']} corners, {total['absent']} absent ({key_of(args)}); "
          f"{total['excluded_inventory']} excluded by inventory (NA read {args.na_read}), "
          f"{total['dropped_no_sidewalk']} no sidewalk, {total['no_candidate']} no candidate "
          f"pano, {total['dropped_projection']} projection dropped -> {total['proposed']} "
          f"proposals in {out}")


def key_of(args):
    return f'{args.arm}/{args.variant}'


# ---------------------------------------------------------------------------- rank

def read_ranked_csv(path, pano_col='pano_id', x_col='pano_x', y_col='pano_y',
                    score_col='score'):
    """{(pano_id, pano_x, pano_y): score} from validator-ai's ranked CSV.

    The minimal contract this tool assumes until sidewalk-validator-ai#69 settles its columns:
    one row per label, the pano id, the label's PIXEL position as Project Sidewalk stores it
    (integers; detectors.ps_pixel of the normalized point), and a score where higher means
    more likely a real NoCurbRamp. Column names are flags. Every #69 change lands here only.
    A key repeated with a different score is refused; with the same score it is one row."""
    out = {}
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        missing = [c for c in (pano_col, x_col, y_col, score_col)
                   if c not in (reader.fieldnames or ())]
        if missing:
            raise SystemExit(f'{path}: missing column(s) {missing}; has {reader.fieldnames}')
        for n, row in enumerate(reader, 2):
            k = (row[pano_col], int(round(float(row[x_col]))), int(round(float(row[y_col]))))
            s = float(row[score_col])
            if k in out and out[k] != s:
                raise SystemExit(f'{path} line {n}: {k} repeated with scores {out[k]} and {s}')
            out[k] = s
    return out


def pixel_key(p):
    """(pano_id, pano_x, pano_y) of a proposal exactly as PS would store its label."""
    x, y = ps_pixel(p['x_normalized'], p['y_normalized'], p['pano']['width'],
                    p['pano']['height'])
    return (p['pano']['pano_id'], x, y)


def rank_proposals(proposals, scores):
    """Proposals with `rank` (1 = highest score; ties by key) and `rank_score`, both null when
    the CSV has no row for the proposal's pixel key. Order and membership are unchanged."""
    matched = sorted(((scores[pixel_key(p)], p['key']) for p in proposals
                      if pixel_key(p) in scores), key=lambda t: (-t[0], t[1]))
    rank = {k: i for i, (_s, k) in enumerate(matched, 1)}
    out = []
    for p in proposals:
        s = scores.get(pixel_key(p))
        out.append(dict(p, rank=rank.get(p['key']), rank_score=s))
    return out


def cmd_rank(args):
    d = Path(args.dir)
    build = load_build(d)
    src = d / 'proposals.jsonl'
    check_file(src, build['outputs']['proposals'], 'proposals.jsonl')
    proposals = ci.read_jsonl(src)
    scores = read_ranked_csv(args.ranked_csv, args.pano_col, args.x_col, args.y_col,
                             args.score_col)
    ranked = rank_proposals(proposals, scores)
    ci.write_jsonl_lf(d / 'proposals.ranked.jsonl', ranked)
    n_matched = sum(1 for p in ranked if p['rank'] is not None)
    rec = {'schema': 'sidewalk-auto-labeler.corner_labels.rank/1', 'ranked_at': ci.utc_now(),
           'proposals_sha256': build['outputs']['proposals']['sha256'],
           'ranked_csv': {'path': str(args.ranked_csv), 'sha256': ci.sha256_file(args.ranked_csv),
                          'rows': len(scores),
                          'columns': [args.pano_col, args.x_col, args.y_col, args.score_col]},
           'matched': n_matched, 'unmatched': len(ranked) - n_matched,
           'output': file_record(d / 'proposals.ranked.jsonl')}
    ci.write_text_lf(d / 'rank.json', json.dumps(rec, indent=1) + '\n')
    print(f'{n_matched} of {len(ranked)} proposals matched a ranked row; '
          f'{len(ranked) - n_matched} unmatched (rank null); nothing dropped')


# -------------------------------------------------------------------------- export

def load_build(d):
    p = Path(d) / 'build.json'
    if not p.exists():
        raise SystemExit(f'{p} not found: export/rank need the build.json written beside the '
                         f'proposals')
    b = json.loads(p.read_text(encoding='utf-8'))
    if b.get('schema') != BUILD_SCHEMA:
        raise SystemExit(f'{p}: schema {b.get("schema")!r}, expected {BUILD_SCHEMA!r}')
    return b


def check_file(path, rec, what):
    """Refuse a file whose sha256 is not the one recorded for it."""
    if not Path(path).exists():
        raise SystemExit(f'{what}: {path} not found. export re-reads and re-hashes the input '
                         f'paths build.json recorded (absolute, on the build machine); run '
                         f'it where those files are')
    got = ci.sha256_file(path)
    if got != rec['sha256']:
        raise SystemExit(f'{what}: {path} has sha256 {got}, but {rec["sha256"]} is recorded; '
                         f'rebuild instead of editing outputs')


def rule_provenance(build):
    """The `models[NoCurbRamp]` block: the rule's version, never a model's.

    model_id is `corner-rule@<12 hex>` of the tool commit that built the proposals;
    model_training_date is RULE_VERSION_DATE in PS's MM-DD-YYYY (send_to_ps refuses null)."""
    sha = build['tool']['git_sha']
    if not sha or not all(ch in '0123456789abcdef' for ch in sha) or len(sha) != 40:
        raise SystemExit(f'build.json tool git_sha {sha!r} is not a commit; build from a '
                         f'committed tool')
    return {'model_id': f'corner-rule@{sha[:12]}',
            'model_training_date': ps_training_date(build.get('rule_version_date',
                                                              RULE_VERSION_DATE)),
            'api_version': API_VERSION, 'model_repo': RULE_REPO, 'model_revision': sha}


def export_lines(proposals, blocks, provenance):
    """results.jsonl records: one per pano, one NoCurbRamp detection per corner on it.

    Written by main.build_output_line (the #168 writer: per-detection label_type, sorted
    label_types, models[NoCurbRamp], line-level label_type), then each detection gets the
    corner's `corner` key, `rank` and the proposal's `source` evidence block, which
    send_to_ps.transform_record does not send. Lines are ordered by their best rank (unranked
    last), then pano id; detections by corner key."""
    import main
    by_pano = {}
    for p in proposals:
        by_pano.setdefault(p['pano']['pano_id'], []).append(p)
    lines = []
    for pid, ps in by_pano.items():
        ps.sort(key=lambda p: p['key'])
        runs = {p['pano'].get('run') for p in ps}
        if len(runs) != 1:
            raise SystemExit(f'pano {pid} is chosen from several runs {sorted(map(str, runs))}')
        result = {'pano': blocks[(runs.pop(), pid)], 'label_types': [LABEL_TYPE],
                  'detections': [(p['x_normalized'], p['y_normalized'], CONFIDENCE, LABEL_TYPE)
                                 for p in ps]}
        line = main.build_output_line(result, {LABEL_TYPE: provenance})
        for d, p in zip(line['detections'], ps):
            d['corner'] = p['key']
            d['rank'] = p.get('rank')
            d['source'] = p['source']
        best = min((p['rank'] for p in ps if p.get('rank') is not None), default=None)
        lines.append(((best is None, best or 0, pid), line))
    lines.sort(key=lambda t: t[0])
    return [line for _k, line in lines]


def cmd_export(args):
    d = Path(args.dir)
    build = load_build(d)
    src = Path(args.proposals) if args.proposals else d / 'proposals.jsonl'
    if src.resolve().parent != d.resolve():
        raise SystemExit(f'{src} is not beside {d / "build.json"}')
    if src.name == 'proposals.jsonl':
        check_file(src, build['outputs']['proposals'], 'proposals.jsonl')
    else:
        rp = d / 'rank.json'
        if not rp.exists():
            raise SystemExit(f'{src}: no rank.json beside it to tie it to build.json')
        rank = json.loads(rp.read_text(encoding='utf-8'))
        if rank['proposals_sha256'] != build['outputs']['proposals']['sha256']:
            raise SystemExit(f'{rp} ranks other proposals than build.json records')
        check_file(src, rank['output'], src.name)
    if build['tool'].get('dirty') and not args.allow_dirty_build:
        raise SystemExit('build.json says the tool had uncommitted edits, so corner-rule@<sha> '
                         'would not name the code that built it (--allow-dirty-build)')
    proposals = ci.read_jsonl(src)
    runs = run_results_paths(build)
    wanted = {(p['pano'].get('run'), p['pano']['pano_id']) for p in proposals}
    for run in sorted({r for r, _ in wanted}, key=str):
        if run not in runs:
            raise SystemExit(f'proposals name run {run!r}, which build.json does not record '
                             f'(runs: {sorted(runs)})')
        check_file(runs[run]['path'], runs[run], f'results of run {run}')
    blocks = load_pano_blocks({r: rec['path'] for r, rec in runs.items()}, wanted)
    lost = sorted(wanted - set(blocks), key=str)
    if lost:
        raise SystemExit(f'{len(lost)} chosen pano(s) not in the runs, e.g. {lost[:3]}')
    lines = export_lines(proposals, blocks, rule_provenance(build))
    out = Path(args.out) if args.out else d / 'results.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    rec = {'schema': 'sidewalk-auto-labeler.corner_labels.export/1', 'exported_at': ci.utc_now(),
           'proposals': {'path': src.name, 'sha256': ci.sha256_file(src)},
           'provenance': rule_provenance(build), 'lines': len(lines),
           'labels': sum(len(x['detections']) for x in lines),
           'output': {'path': str(out), 'sha256': ci.sha256_file(out)}}
    ci.write_text_lf(d / 'export.json', json.dumps(rec, indent=1) + '\n')
    print(f"{rec['labels']} NoCurbRamp labels on {rec['lines']} panos -> {out}")


# --------------------------------------------------------------------------- score

def score(proposals, verdicts, sample_keys=None):
    """The pre-registered corner-level rule over the proposals that carry a blind verdict.

    precision = absent / (absent + present) over corners rated blind; `cant_tell` leaves the
    denominator and its share of all rated corners is reported. A secondary `label_precision`
    also counts `no_sidewalk` absences as wrong (guide rule 3: no label there). The verdict is
    PASS / FAIL only when the scored corners are a sample drawn for this rule (sample_keys
    given) and n = absent + present >= RULE_MIN_N; otherwise NOT SCORED.
    """
    units = verdicts.get('units') or {}
    keys = None if sample_keys is None else set(sample_keys)
    have = {p['key'] for p in proposals}
    not_in = [] if keys is None else sorted(keys - have)
    tally = {'present': 0, 'absent': 0, 'cant_tell': 0}
    kinds = {'curb_no_ramp': 0, 'no_sidewalk': 0, 'unspecified': 0}
    by_stratum, rated = {}, []
    for p in proposals:
        if keys is not None and p['key'] not in keys:
            continue
        blind = (units.get(p['unit']) or {}).get('blind')
        e = (blind or {}).get(str(p['corner']))
        v = (e or {}).get('verdict')
        if v not in tally:
            continue
        tally[v] += 1
        s = by_stratum.setdefault(p['stratum'], {'present': 0, 'absent': 0, 'cant_tell': 0})
        s[v] += 1
        if v == 'absent':
            kinds[e.get('absent_kind') or 'unspecified'] += 1
        rated.append(p['key'])
    n_rated = sum(tally.values())
    n = tally['absent'] + tally['present']
    prec = tally['absent'] / n if n else None
    ci95 = ci.wilson(tally['absent'], n)
    ok_label = tally['absent'] - kinds['no_sidewalk']
    lab = ci.wilson(ok_label, n)
    drawn = sample_keys is not None
    if not drawn or n < RULE_MIN_N:
        why = []
        if n < RULE_MIN_N:
            why.append(f'n = {n} < {RULE_MIN_N}')
        if not drawn:
            why.append('sample not drawn for this rule')
        verdict = f"NOT SCORED ({'; '.join(why)})"
    else:
        passed = prec >= RULE_PRECISION and ci95[0] >= RULE_WILSON_LO
        verdict = (f"{'PASS' if passed else 'FAIL'}: precision {prec:.3f} "
                   f"(rule >= {RULE_PRECISION}), Wilson lower {ci95[0]:.3f} "
                   f"(rule >= {RULE_WILSON_LO}), n = {n}")
    return {
        'n_proposals': len(proposals) if keys is None else len(keys & have),
        'sample_keys_not_in_proposals': not_in,
        'n_rated': n_rated, 'n': n, **tally,
        'cant_tell_share': round(tally['cant_tell'] / n_rated, 4) if n_rated else None,
        'precision': None if prec is None else round(prec, 4),
        'wilson95': None if ci95 is None else [round(ci95[0], 4), round(ci95[1], 4)],
        'absent_kinds': kinds,
        'label_precision': None if not n else round(ok_label / n, 4),
        'label_wilson95': None if lab is None else [round(lab[0], 4), round(lab[1], 4)],
        'by_stratum': dict(sorted(by_stratum.items())),
        'sample_drawn_for_rule': drawn, 'verdict': verdict, 'rated_keys': sorted(rated),
    }


def read_sample(path):
    with open(path, encoding='utf-8', newline='') as f:
        return [corner_key(r['unit'], int(r['corner'])) for r in csv.DictReader(f)]


def cmd_score(args):
    proposals = ci.read_jsonl(args.proposals)
    verdicts = json.loads(Path(args.verdicts).read_text(encoding='utf-8'))
    sample = read_sample(args.sample) if args.sample else None
    res = score(proposals, verdicts, sample)
    rec = {'schema': 'sidewalk-auto-labeler.corner_labels.score/1', 'issue': 'sidewalk-auto-labeler#169',
           'rule': {'precision_min': RULE_PRECISION, 'wilson_lower_min': RULE_WILSON_LO,
                    'min_n': RULE_MIN_N, 'cant_tell': 'excluded from the denominator',
                    'read': 'blind verdicts only'},
           'proposals': {'path': Path(args.proposals).name,
                         'sha256': ci.sha256_file(args.proposals)},
           'verdicts': {'path': Path(args.verdicts).name, 'sha256': ci.sha256_file(args.verdicts),
                        'rater': verdicts.get('rater'), 'items_sha256': verdicts.get('items_sha256'),
                        'rubric_version': verdicts.get('rubric_version')},
           'sample': None if not args.sample else {'path': Path(args.sample).name,
                                                   'sha256': ci.sha256_file(args.sample)},
           **res}
    text = json.dumps(rec, indent=1) + '\n'
    if args.out:
        ci.write_text_lf(args.out, text)
    if res['sample_keys_not_in_proposals']:
        print(f"WARNING: {len(res['sample_keys_not_in_proposals'])} sample key(s) are not in "
              f"the proposals, e.g. {res['sample_keys_not_in_proposals'][:3]}")
    print(f"{res['n_rated']} rated corners: {res['absent']} absent, {res['present']} present, "
          f"{res['cant_tell']} can't tell; precision {res['precision']} {res['wilson95']}; "
          f"{res['verdict']}")


# -------------------------------------------------------------------------- sample

def draw_sample(proposals, n=SAMPLE_N, seed=SAMPLE_SEED):
    """n corner keys drawn without replacement from the proposals, seeded, in draw order."""
    keys = sorted({p['key'] for p in proposals})
    if len(keys) < n:
        raise SystemExit(f'only {len(keys)} proposals; cannot draw {n}')
    return random.Random(seed).sample(keys, n)


def cmd_sample(args):
    proposals = ci.read_jsonl(args.proposals)
    keys = draw_sample(proposals, args.n, args.seed)
    out = Path(args.out)
    rows = [{'unit': k.rsplit('#', 1)[0], 'corner': k.rsplit('#', 1)[1]} for k in keys]
    ci.write_csv_lf(out, ['unit', 'corner'], rows)
    rec = {'schema': 'sidewalk-auto-labeler.corner_labels.sample/1', 'drawn_at': ci.utc_now(),
           'seed': args.seed, 'n': args.n, 'population': len({p['key'] for p in proposals}),
           'proposals': {'path': Path(args.proposals).name,
                         'sha256': ci.sha256_file(args.proposals)},
           'rule': 'random.Random(seed).sample(sorted proposal keys, n)',
           'sample': {'path': out.name, 'sha256': ci.sha256_file(out)}}
    ci.write_text_lf(out.with_suffix('.json'), json.dumps(rec, indent=1) + '\n')
    print(f'{args.n} of {rec["population"]} proposals drawn with seed {args.seed} -> {out}')


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='absent corners -> NoCurbRamp proposals')
    b.add_argument('--city', default='vancouver')
    b.add_argument('--corners', required=True, help="corner_inventory's corners_full.jsonl")
    b.add_argument('--run-dir', required=True, help='the base run (results.jsonl, sites.jsonl)')
    b.add_argument('--extra-run', action='append',
                   help='a run merged into the corner build (corner_inventory build --extra-run)')
    b.add_argument('--out', required=True)
    b.add_argument('--arm', default='fusion', choices=ci.ARMS)
    b.add_argument('--variant', default='primary', choices=ci.VARIANTS)
    b.add_argument('--na-read', required=True, choices=NA_READS,
                   help='count: any inventory point (NA included) excludes a corner; discount: '
                        'NA points with no RAMPTYPE do not. REQUIRED: the primary read is not '
                        'settled until gallery 243 (RampNet#243) is scored')
    b.add_argument('--max-range-m', type=float, default=MAX_RANGE_M,
                   help=f'farthest pano that can carry the label (<= {TABLE_OBS_M:g}, the '
                        f"table's observation radius)")
    b.add_argument('--min-view-m', type=float, default=MIN_VIEW_M,
                   help='nearer panos look almost straight down at the corner and are skipped')
    b.add_argument('--top-k', type=int, default=TOP_K,
                   help='the newest capture among the k nearest candidates carries the label')
    b.add_argument('--camera-height-m', type=float, default=CAMERA_HEIGHT_M,
                   help='camera height of the projection. 2.5 m is the fixed height the #241 '
                        'fusion, and so the corner table, was built with; keep it so the label '
                        'sits where the table measured the corner')
    b.add_argument('--hemisphere-deg', type=float, default=HEMISPHERE_DEG,
                   help='opt-in view test: the corner point must lie within this many degrees '
                   'of camera_heading (90 = forward hemisphere). Default: off, any direction')
    b.add_argument('--sidewalk-presence', help='GeoJSON of sidewalk geometry; corners with none '
                   'within --sidewalk-radius-m are not labeled (guide rule 3)')
    b.add_argument('--sidewalk-radius-m', type=float, default=SIDEWALK_RADIUS_M)
    b.add_argument('--sidewalk-keep', action='append', metavar='FIELD=V1,V2',
                   help='keep only sidewalk features whose FIELD is one of the values '
                   '(repeatable; recorded in build.json). Vancouver WA: '
                   'STATUS=Available,Pathway,Private')
    b.add_argument('--allow-unpinned', action='store_true',
                   help='build on inputs whose sha256 differs from PINNED_SHA256 (recorded)')
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser('rank', help="join validator-ai's ranked CSV (#69); drops nothing")
    r.add_argument('--dir', required=True, help='a build dir')
    r.add_argument('--ranked-csv', required=True)
    r.add_argument('--pano-col', default='pano_id')
    r.add_argument('--x-col', default='pano_x')
    r.add_argument('--y-col', default='pano_y')
    r.add_argument('--score-col', default='score')
    r.set_defaults(fn=cmd_rank)

    e = sub.add_parser('export', help='proposals -> results.jsonl (label_type NoCurbRamp)')
    e.add_argument('--dir', required=True, help='a build dir (build.json + proposals)')
    e.add_argument('--proposals', help='proposals.jsonl (default) or proposals.ranked.jsonl')
    e.add_argument('--out', help='default <dir>/results.jsonl')
    e.add_argument('--allow-dirty-build', action='store_true')
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser('score', help='the pre-registered corner rule against blind ratings')
    s.add_argument('--proposals', required=True)
    s.add_argument('--verdicts', required=True, help='a gallery verdicts__<rater>.json')
    s.add_argument('--sample', help="sample.csv: score only the sample drawn for this rule")
    s.add_argument('--out', help='write the score JSON here')
    s.set_defaults(fn=cmd_score)

    m = sub.add_parser('sample', help='draw the seeded corner sample for a future gallery')
    m.add_argument('--proposals', required=True)
    m.add_argument('--n', type=int, default=SAMPLE_N)
    m.add_argument('--seed', type=int, default=SAMPLE_SEED)
    m.add_argument('--out', required=True, help='sample.csv (sample.json is written beside it)')
    m.set_defaults(fn=cmd_sample)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
