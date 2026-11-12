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
          With --stratum / --reserve it writes instead ONE fixed seeded order over the whole
          (stratum-filtered) population, with the rated sample, its reserve and the sequential
          stop rule recorded (the #169 test-server pilot; issue #183's design).
  pilot-score  the sequential stop rule over a lead's live-site validations of a pilot order.

export takes --stratum (repeatable) and --only-corners <csv> [--only-first N] to export a
pilot subset; both are recorded in the export record with the counts kept and dropped.

The pre-registered rule (issue #169, stated before any label was built): corner-level absence
precision against blind ratings >= 0.90 with Wilson lower bound >= 0.85, on a seeded sample of
>= 100 absent corners drawn under the primary NA read; `cant_tell` corners leave the
denominator and their share is reported. --na-read has no default. Gallery 243 (RampNet#243)
settled the primary read as `discount` on 2026-11-05 (NA_READ_SETTLEMENT); a build records that
only when it is given --na-read-settled. Write-up: docs/corner-labels-169.md.

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
from collections import Counter
from pathlib import Path
from statistics import NormalDist

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
#: The test-server pilot's sequential rating (issue #183's design, adopted for the #169 pilot):
#: looks at 40, 60 and 100 decided labels, the one-sided alpha of the rule's Wilson 95% lower
#: bound (0.025) split evenly across the looks (Bonferroni), and the rule's own bar.
#: Where the label goes (2026-11-11, after the test-server smoke send: Jon found all three labels
#: a couple of metres inland of the curb). The corner point sits CORNER_OFFSET_M (12 m) from the
#: unit centre for every street type; the fused ramp sites of a residential unit sit at a median
#: of about 10 m. 'site-median' moves ONLY the label point along the corner's bisector to the
#: unit's own median centre-to-site distance (or its stratum's, from the same corner table),
#: less LABEL_INSET_M toward the road, clamped to [stratum p10, the corner offset]. The corner
#: decision and the pano choice stay keyed on the corner point. 'corner' is the old behaviour.
LABEL_POINT_RULES = ('site-median', 'corner')
LABEL_POINT_DEFAULT = 'site-median'
LABEL_INSET_M = 0.8
PILOT_LOOKS = (40, 60, 100)
PILOT_ALPHA_ONE_SIDED = 0.025
#: live-site validation results -> the rubric's terms (agree = the NoCurbRamp label is right)
PILOT_VERDICTS = {'agree': 'right', 'disagree': 'wrong', 'unsure': 'cant_tell'}

#: How the primary NA read was settled: gallery 243 (RampNet#243) scored blind, 95 of 95 units
#: (sidewalk-auto-labeler PR #180, branch corner-gallery-243-verdicts). The 40 sampled units
#: whose only inventory points are `NA` with no `RAMPTYPE` were rated absent at every corner,
#: 39/39 decided (1 undetermined), Wilson 95% [0.910, 1.000]; the corners holding those points
#: 99/99. So such points are the city's record of no ramp: read (b) of amendment A8, `discount`.
#: `build --na-read-settled` copies this block into build.json; score_json_sha256 is the git
#: blob of score/score.json on that branch.
NA_READ_SETTLEMENT = {
    'primary': 'discount',
    'settled_on': '2026-11-05',
    'evidence': 'sidewalk-auto-labeler PR #180 (branch corner-gallery-243-verdicts): '
                'runs/vancouver/corner_gallery243/score/report.md and score/score.json',
    'pr': 'ProjectSidewalk/sidewalk-auto-labeler#180',
    'score_json': 'runs/vancouver/corner_gallery243/score/score.json',
    'score_json_sha256': '447c6bb5db0fd2e2d18d7f01c8812f313db28d25cd23c64da14d23e873fb12bc',
    'verdicts': 'runs/vancouver/corner_gallery243/verdicts__jonf.json',
    'verdicts_sha256': '30a5008d125f143d6cc536e9787676310fc474b9c4cc127f59a419a16e2e80bd',
    'na_noramp_units_absent': '39/39 decided (1 undetermined), Wilson 95% [0.910, 1.000]',
    'na_noramp_corners_absent': '99/99',
}

#: Inputs #241's merged corner build and gallery 243 were made from (runs/vancouver/
#: corner_inventory_posdetect241/build.json `inputs`; corners_full from runs/vancouver/
#: corner_gallery243/snapshot.json). A build on other bytes is refused unless --allow-unpinned.
#: A role may pin several files (a tuple): corners_full is either the #241 build or the
#: --split-wide rebuild of the same inputs (runs/vancouver/corner_inventory169_split/, the
#: #169 revision of 2026-11-06, docs/corner-inventory.md amendment of that date).
CORNERS_FULL_241 = 'ec20606098d0e8b34113b4b328e57d3737b7669d01a37f1d6d722faab4681f8d'
CORNERS_FULL_SPLIT = 'b50da32e1bb0b86f5d93fcd0de0cead1be620d8ac39f309d3abd2461f87ce9fc'
PINNED_SHA256 = {'vancouver': {
    'corners_full': (CORNERS_FULL_241, CORNERS_FULL_SPLIT),
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

    Gallery 243 (RampNet#243) settled `discount` as primary on 2026-11-05 (NA_READ_SETTLEMENT).

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


def quantile(xs, q):
    """Linear-interpolation quantile (numpy's default) of a non-empty sequence.

    >>> quantile([1, 2, 3, 4], 0.5), quantile([10], 0.1)
    (2.5, 10.0)
    """
    s = sorted(float(x) for x in xs)
    i = q * (len(s) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


class LabelPlacement:
    """The 'site-median' label point: where on the corner's bisector a NoCurbRamp label goes.

    The corner point (corner_inventory.corner_point) is a fixed CORNER_OFFSET_M from the unit
    centre for every street type, which on residential streets lands behind the curb. The
    fused ramp sites the corner table assigns to a unit's corners (corners[].site_ids, resolved
    in the SAME merged site set the table was built on: the base run's operational sites plus
    each --extra-run's, ids prefixed `<run>:`) say where ramps actually sit. Distance from the
    unit centre along the bisector (start_deg + width_deg / 2):

      (a) 'unit': the unit has sites at its corners -> the median of its centre-to-site
          distances;
      (b) 'stratum': it has none -> the median over every site at every corner of every unit
          of its stratum, computed from the same corner table;
      (c) 'corner': its stratum has no site at all -> the corner offset itself (old point);
      build_proposals adds 'corner_range_fallback' when the moved point is past the chosen
      pano's projection range: the corner point is used, so no corner is ever dropped by it;

    minus inset_m (toward the road: a missing ramp belongs at the curb edge, a ramp site's
    centre sits slightly behind it), then clamped to [the stratum's p10, max_m] so a label is
    never farther out than the old corner point and never nearer the centre than 9 in 10 of
    its stratum's ramps. Nothing is per city: every number comes from the corner table given.
    """

    def __init__(self, units, site_pos, inset_m=LABEL_INSET_M, max_m=None):
        self.inset_m = float(inset_m)
        self.max_m = float(ci.CORNER_OFFSET_M if max_m is None else max_m)
        self.unit_d = {}
        by_stratum = {s: [] for s in STRATA}
        missing = set()
        for rec in units:
            if rec['type'] not in STRATA:
                continue
            fr = geo.LocalFrame(rec['lat'], rec['lng'])
            ds = []
            for c in rec.get('corners') or ():
                for sid in c.get('site_ids') or ():
                    if sid not in site_pos:
                        missing.add(sid)
                        continue
                    e, n = fr.to_enu(*site_pos[sid])
                    ds.append(math.hypot(e, n))
            self.unit_d[rec['unit']] = ds
            by_stratum[rec['type']].extend(ds)
        if missing:
            raise SystemExit(f'{len(missing)} corner site id(s) are not in the loaded sites, e.g. '
                             f'{sorted(map(str, missing))[:3]}: the corner table was built on '
                             f'other runs than --run-dir / --extra-run')
        self.stats = {}
        for s, ds in by_stratum.items():
            self.stats[s] = ({'n_sites': 0, 'median_m': None, 'p10_m': None} if not ds else
                             {'n_sites': len(ds), 'median_m': r2(quantile(ds, 0.5)),
                              'p10_m': r2(quantile(ds, 0.1)), 'p25_m': r2(quantile(ds, 0.25)),
                              'p75_m': r2(quantile(ds, 0.75)), 'p90_m': r2(quantile(ds, 0.9))})
        self._exact = {s: (quantile(ds, 0.5), quantile(ds, 0.1)) if ds else None
                       for s, ds in by_stratum.items()}

    def record(self):
        """build.json's `label_point` block."""
        return {'rule': 'site-median', 'inset_m': self.inset_m, 'max_m': self.max_m,
                'min': 'stratum p10', 'stratum_stats': self.stats,
                'cases': {'unit': "median distance from the unit centre to the fused sites at "
                                  "the unit's corners",
                          'stratum': "the stratum's median (unit has no site at its corners)",
                          'corner': 'the corner offset (stratum has no site at all)',
                          'corner_range_fallback': "the corner point: the moved point is past "
                                                   "the chosen pano's projection range"},
                'sites': "corners[].site_ids, resolved in the merged operational sites of "
                         "--run-dir and each --extra-run (extra ids prefixed '<run>:')"}

    def place(self, rec, c):
        """{'lat', 'lng', 'dist_m', 'case', 'base_m', 'n_unit_sites', 'clamped',
        'bearing_deg'} of the label point of corner c of unit rec."""
        ds = self.unit_d.get(rec['unit']) or []
        exact = self._exact.get(rec['type'])
        if ds:
            case, base = 'unit', quantile(ds, 0.5)
        elif exact is not None:
            case, base = 'stratum', exact[0]
        else:
            case, base = 'corner', self.max_m
        clamped = None
        if case == 'corner':
            d = self.max_m
        else:
            d = base - self.inset_m
            lo = min(exact[1], self.max_m)
            if d > self.max_m:
                d, clamped = self.max_m, 'max'
            elif d < lo:
                d, clamped = lo, 'min'
        b = (float(c['start_deg']) + float(c['width_deg']) / 2.0) % 360.0
        if case == 'corner':        # exactly the old point, as the corner table stored it
            lat, lng = c['lat'], c['lng']
            d = math.hypot(*geo.LocalFrame(rec['lat'], rec['lng']).to_enu(lat, lng))
        else:
            lat, lng = geo.LocalFrame(rec['lat'], rec['lng']).to_latlng(
                d * math.sin(math.radians(b)), d * math.cos(math.radians(b)))
        return {'lat': lat, 'lng': lng, 'dist_m': d, 'case': case, 'base_m': base,
                'n_unit_sites': len(ds), 'clamped': clamped, 'bearing_deg': b}


def check_pins(city, inputs, allow_unpinned):
    """Compare every input's sha256 with PINNED_SHA256[city]; a mismatch exits unless
    allow_unpinned. Returns 'pinned', 'unpinned city' or 'overridden: <roles>'."""
    pins = PINNED_SHA256.get(city)
    if not pins:
        return 'unpinned city'
    def allowed(pin):
        return pin if isinstance(pin, (tuple, list, set, frozenset)) else (pin,)
    bad = sorted(role for role, rec in inputs.items()
                 if role in pins and rec['sha256'] not in allowed(pins[role]))
    missing = sorted(set(pins) - set(inputs))
    if bad or missing:
        msg = f'inputs differ from the pinned {city} build: changed {bad}, missing {missing}'
        if not allow_unpinned:
            raise SystemExit(msg + ' (--allow-unpinned to build anyway; recorded)')
        print(f'WARNING (--allow-unpinned): {msg}')
        return f'overridden: changed {bad}, missing {missing}'
    return 'pinned'


# --------------------------------------------------------------------------- build

def build_proposals(units, panos, sites, args, sidewalks=None, placement=None):
    """(proposals, unlabeled rows, counts by stratum) over the corner table's records.

    placement (a LabelPlacement, or None for the old corner point) moves only the point that
    is projected into the chosen pano: the absent state, the inventory and sidewalk tests and
    the pano choice all stay keyed on the corner point, so the corner set and the panos are
    the same whichever label point is used.

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
            lp = None if placement is None else placement.place(rec, c)
            plat, plng = (c['lat'], c['lng']) if lp is None else (lp['lat'], lp['lng'])
            proj = project(pano, plat, plng, args.camera_height_m, args.max_range_m)
            if proj is None and lp is not None:
                # The label point fell past the chosen pano's range (a pano ~25 m off the
                # corner, the point moved away from it): keep the corner point, which the pano
                # choice already proved projectable, so the corner set never depends on it.
                proj = project(pano, c['lat'], c['lng'], args.camera_height_m, args.max_range_m)
                fr = geo.LocalFrame(rec['lat'], rec['lng'])
                lp = dict(lp, case='corner_range_fallback', lat=c['lat'], lng=c['lng'],
                          dist_m=math.hypot(*fr.to_enu(c['lat'], c['lng'])), clamped=None)
            if proj is None:
                cnt['dropped_projection'] += 1
                unlabeled.append(dict(row, reason='projection_dropped'))
                continue
            cnt['proposed'] += 1
            proposals.append(proposal_record(rec, c, pano, chosen, runners, cands, rejected,
                                             proj, sites.nearest(c['lat'], c['lng']), args, lp))
    proposals.sort(key=lambda p: (p['unit'], p['corner']))
    unlabeled.sort(key=lambda r: (r['unit'], r['corner']))
    return proposals, unlabeled, counts


def proposal_record(rec, c, pano, chosen, runners, cands, rejected, proj, site, args,
                    lp=None):
    """One proposal line: the corner, the chosen pano and pixel, and the evidence. lat/lng
    are always the corner point (the corner's identity); with a label point lp (LabelPlacement)
    the pixel and range_m are the label point's, and `label_point` records it."""
    out = {
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
            'sector': sector_record(c),
            'n_candidates': len(cands), 'rejected': rejected,
            'runner_ups': [{'pano_id': x['pano_id'], 'dist_m': r2(x['dist_m']),
                            'capture_date': x.get('capture_date'),
                            'off_axis_deg': r2(x['off_axis_deg'])}
                           for x in runners[:N_RUNNER_UPS]],
            'chosen_dist_m': r2(chosen['dist_m']),
        },
    }
    if lp is not None:
        fr = geo.LocalFrame(c['lat'], c['lng'])
        out['label_point'] = {'lat': r6(lp['lat']), 'lng': r6(lp['lng']),
                              'dist_m': r2(lp['dist_m']), 'case': lp['case'],
                              'base_m': r2(lp['base_m']), 'n_unit_sites': lp['n_unit_sites'],
                              'clamped': lp['clamped'], 'bearing_deg': r2(lp['bearing_deg']),
                              'shift_m': r2(math.hypot(*fr.to_enu(lp['lat'], lp['lng'])))}
    return out


def sector_record(c):
    """A proposal's `source.sector`: start, width, wide, and -- from a --split-wide inventory
    only -- which bound is a phantom leg (so a pre-split build's proposals are unchanged)."""
    out = {'start_deg': c.get('start_deg'), 'width_deg': c.get('width_deg'),
           'wide': c.get('wide')}
    if 'phantom_start' in c:
        out['phantom_start'] = c['phantom_start']
        out['phantom_end'] = c['phantom_end']
    return out


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
    rule = getattr(args, 'label_point', 'corner')
    placement = None
    if rule == 'site-median':
        placement = LabelPlacement(units, {s[2]['site_id']: (s[0], s[1]) for s in sites},
                                   args.label_inset_m)
    elif rule != 'corner':
        raise SystemExit(f'--label-point {rule!r}: one of {LABEL_POINT_RULES}')

    proposals, unlabeled, counts = build_proposals(units, panos, SiteIndex(sites), args,
                                                   sidewalks, placement)
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
                   **na_read_params(args), 'max_range_m': args.max_range_m,
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
        'label_point': (label_point_summary(placement, proposals) if placement is not None
                        else {'rule': 'corner', 'offset_m': ci.CORNER_OFFSET_M,
                              'note': 'the corner point itself (the behaviour of every build '
                                      'before 2026-11-11)'}),
        'label_point_default': LABEL_POINT_DEFAULT,
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


def label_point_summary(placement, proposals):
    """build.json's `label_point`: the rule and its numbers, and how the proposals used it."""
    rec = placement.record()
    lps = [p['label_point'] for p in proposals]
    rec['proposals'] = {
        'by_case': dict(sorted(Counter(lp['case'] for lp in lps).items())),
        'clamped': dict(sorted(Counter(str(lp['clamped']) for lp in lps).items())),
        'shift_m_p10_median_p90': (None if not lps else
                                   [r2(quantile([lp['shift_m'] for lp in lps], q))
                                    for q in (0.1, 0.5, 0.9)]),
        'range_under_min_view': sum(1 for p in proposals if p['range_m'] < MIN_VIEW_M)}
    return rec


def na_read_params(args):
    """build.json's NA-read params. Without --na-read-settled: {'na_read_settled': False}, as
    every build before 2026-11-05 recorded. With it: the settlement (NA_READ_SETTLEMENT), the
    primary read, and whether this build's read is that one (a `count` build stays buildable,
    as the comparison, and says it is not primary)."""
    if not getattr(args, 'na_read_settled', False):
        return {'na_read_settled': False}
    return {'na_read_settled': True, 'na_read_primary': NA_READ_SETTLEMENT['primary'],
            'na_read_is_primary': args.na_read == NA_READ_SETTLEMENT['primary'],
            'na_read_settlement': dict(NA_READ_SETTLEMENT)}


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


def read_corner_order(path):
    """Corner keys of a sample or pilot CSV (columns unit, corner), in its order: by the
    `order` column when there is one, else file order. A repeated key is refused."""
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if rows and 'order' in rows[0]:
        rows.sort(key=lambda r: int(r['order']))
    keys = [corner_key(r['unit'], int(r['corner'])) for r in rows]
    dup = sorted(k for k, n in Counter(keys).items() if n > 1)
    if dup:
        raise SystemExit(f'{path}: {len(dup)} corner key(s) repeated, e.g. {dup[:3]}')
    return keys


def read_pano_list(path):
    """Pano ids, one per line (blank lines and #-comments ignored)."""
    out = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            out.add(line)
    return out


def order_subset(order, props, skip_first=0, exclude_panos=()):
    """(kept keys, dropped [(position, key, pano_id)]) of a pilot order: positions after the
    first skip_first, without corners whose chosen pano is in exclude_panos (e.g. the panos a
    smoke send already labeled: the server refuses a second AI label of a type on a pano).
    Positions stay the full order's. A key that is not a proposal is refused.

    >>> ps = {'a#0': {'pano': {'pano_id': 'P'}}, 'b#0': {'pano': {'pano_id': 'Q'}},
    ...       'c#0': {'pano': {'pano_id': 'R'}}}
    >>> order_subset(['a#0', 'b#0', 'c#0'], ps, 1, {'R'})
    (['b#0'], [(3, 'c#0', 'R')])
    """
    missing = [k for k in order[skip_first:] if k not in props]
    if missing:
        raise SystemExit(f'{len(missing)} order key(s) are not in the proposals, e.g. '
                         f'{missing[:3]}')
    kept, dropped = [], []
    for i, k in enumerate(order[skip_first:], skip_first + 1):
        pid = props[k]['pano']['pano_id']
        if pid in exclude_panos:
            dropped.append((i, k, pid))
        else:
            kept.append(k)
    return kept, dropped


def select_proposals(proposals, strata=None, only=None):
    """(kept proposals, record) for export's pilot filters, in the proposals' own order.

    strata keeps the proposals of those strata (each must be one of STRATA). only is a list of
    corner keys (a pilot draw); every one must be a proposal, and, with strata, of a kept
    stratum, so nothing asked for is dropped silently. The record holds the filters and the
    counts kept and dropped, for the export record.

    >>> ps = [{'key': 'a#0', 'stratum': 'residential'}, {'key': 'b#0', 'stratum': 'arterial'}]
    >>> [p['key'] for p in select_proposals(ps, strata=['residential'])[0]]
    ['a#0']
    """
    rec = {'strata': None if not strata else sorted(set(strata)), 'n_proposals': len(proposals)}
    kept = list(proposals)
    if strata:
        unknown = sorted(set(strata) - set(STRATA))
        if unknown:
            raise SystemExit(f'--stratum {unknown}: not a stratum (one of {list(STRATA)})')
        kept = [p for p in kept if p['stratum'] in set(strata)]
        rec['dropped_by_stratum'] = len(proposals) - len(kept)
    if only is not None:
        by_key = {p['key']: p for p in proposals}
        missing = [k for k in only if k not in by_key]
        if missing:
            raise SystemExit(f'{len(missing)} --only-corners key(s) are not in the proposals, '
                             f'e.g. {missing[:3]}')
        if strata:
            off = [k for k in only if by_key[k]['stratum'] not in set(strata)]
            if off:
                raise SystemExit(f'{len(off)} --only-corners key(s) are outside --stratum '
                                 f'{rec["strata"]}, e.g. {off[:3]}')
        want, before = set(only), len(kept)
        kept = [p for p in kept if p['key'] in want]
        rec['dropped_by_only_corners'] = before - len(kept)
    rec['kept'] = len(kept)
    rec['kept_by_stratum'] = dict(sorted(Counter(p['stratum'] for p in kept).items()))
    return kept, rec


def export_record_path(d, out):
    """Where export writes its record: <dir>/export.json for the default <dir>/results.jsonl
    (as before), else beside the output as <stem>.export.json, so two exports of one build
    (e.g. a pilot subset and its whole stratum) each keep their own."""
    out = Path(out)
    if out.resolve() == (Path(d) / 'results.jsonl').resolve():
        return Path(d) / 'export.json'
    return out.with_name(out.stem + '.export.json')


def export_lines(proposals, blocks, provenance, order=None):
    """results.jsonl records: one per pano, one NoCurbRamp detection per corner on it.

    Written by main.build_output_line (the #168 writer: per-detection label_type, sorted
    label_types, models[NoCurbRamp], line-level label_type), then each detection gets the
    corner's `corner` key, `rank` and the proposal's `source` evidence block, which
    send_to_ps.transform_record does not send. Lines are ordered by their best rank (unranked
    last), then pano id; detections by corner key. With `order` ({corner key: position}, a
    pilot draw), each detection also gets `pilot_order` and lines are ordered by their best
    position instead, so a capped send (--limit) takes the head of the draw."""
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
            if order is not None:
                d['pilot_order'] = order[p['key']]
            d['source'] = p['source']
        if order is not None:
            best = min(order[p['key']] for p in ps)
        else:
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
    strata = getattr(args, 'stratum', None)
    only_path = getattr(args, 'only_corners', None)
    only_first = getattr(args, 'only_first', None)
    if only_first is not None and not only_path:
        raise SystemExit('--only-first needs --only-corners')
    if only_first is not None and only_first < 1:
        raise SystemExit('--only-first must be at least 1')
    skip_first = getattr(args, 'skip_first', 0) or 0
    exclude_path = getattr(args, 'exclude_panos', None)
    if (skip_first or exclude_path) and not only_path:
        raise SystemExit('--skip-first / --exclude-panos need --only-corners')
    selection, order, order_drops = None, None, None
    if strata or only_path:
        only = None
        if only_path:
            ordered, order_drops = order_subset(
                read_corner_order(only_path), {p['key']: p for p in proposals}, skip_first,
                read_pano_list(exclude_path) if exclude_path else ())
            if only_first is not None:
                if only_first > len(ordered):
                    raise SystemExit(f'--only-first {only_first}: {only_path} has only '
                                     f'{len(ordered)} corners')
                ordered = ordered[:only_first]
            only = ordered
        proposals, selection = select_proposals(proposals, strata, only)
        if only_path:
            # positions in the whole draw (1 = first), so a subset keeps the draw's numbering
            full = read_corner_order(only_path)
            pos = {k: i for i, k in enumerate(full, 1)}
            order = {p['key']: pos[p['key']] for p in proposals}
            selection['only_corners'] = {'path': Path(only_path).name,
                                         'sha256': ci.sha256_file(only_path),
                                         'rows': len(full), 'first': only_first,
                                         'skip_first': skip_first,
                                         'positions': [min(order.values()),
                                                       max(order.values())]}
            if exclude_path:
                selection['exclude_panos'] = {
                    'path': Path(exclude_path).name, 'sha256': ci.sha256_file(exclude_path),
                    'panos': sorted(read_pano_list(exclude_path)),
                    'dropped_corners': order_drops}
        if not proposals:
            raise SystemExit('the filters keep no proposal')
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
    lines = export_lines(proposals, blocks, rule_provenance(build), order)
    out = Path(args.out) if args.out else d / 'results.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    data = out.read_bytes()
    rec = {'schema': 'sidewalk-auto-labeler.corner_labels.export/1', 'exported_at': ci.utc_now(),
           'proposals': {'path': src.name, 'sha256': ci.sha256_file(src)},
           'provenance': rule_provenance(build), 'lines': len(lines),
           'labels': sum(len(x['detections']) for x in lines),
           'output': {'path': str(out), 'sha256': hashlib.sha256(data).hexdigest(),
                      'lines': data.count(b'\n'), 'bytes': len(data)}}
    if selection is not None:
        rec['selection'] = selection
    rec_path = export_record_path(d, out)
    ci.write_text_lf(rec_path, json.dumps(rec, indent=1) + '\n')
    print(f"{rec['labels']} NoCurbRamp labels on {rec['lines']} panos -> {out} "
          f"(record {rec_path.name})")


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


def check_bundle(bundle, verdicts, sample_path, proposals_path):
    """Every reason a precision-gallery export cannot be scored against this sample (empty =
    ok). The bundle is one `corner_gallery.py build --items-from` wrote: its snapshot must
    record this sample's and these proposals' sha256, items.jsonl must be the one it recorded,
    the verdicts file must be valid for those items and the bundle's rubric (the gallery 243
    validator, corner_gallery_score.validate), and every sampled corner must be an item corner."""
    import corner_gallery_score as cgs
    bundle = Path(bundle)
    snap = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
    problems = []
    if snap.get('mode') != 'sample':
        problems.append(f'{bundle}: not a --items-from bundle (snapshot mode {snap.get("mode")!r})')
    s_sha = ci.sha256_file(sample_path)
    if (snap.get('sample') or {}).get('sha256') != s_sha:
        problems.append(f'{Path(sample_path).name} (sha256 {s_sha[:12]}) is not the sample the '
                        f'bundle was built from')
    p_sha = ci.sha256_file(proposals_path)
    if (snap.get('proposals') or {}).get('sha256') != p_sha:
        problems.append(f'{Path(proposals_path).name} (sha256 {p_sha[:12]}) is not the proposals '
                        f'the bundle was built from')
    items_sha = ci.sha256_file(bundle / 'items.jsonl')
    if items_sha != snap.get('items_sha256'):
        problems.append('items.jsonl is not the one snapshot.json recorded')
    items = ci.read_jsonl(bundle / 'items.jsonl')
    problems += cgs.validate(verdicts, items, items_sha, snap.get('rubric'),
                             snap.get('rubric_version', 1))
    have = {corner_key(i['unit'], c['corner']) for i in items for c in i['corners']}
    lost = [k for k in read_sample(sample_path) if k not in have]
    if lost:
        problems.append(f'{len(lost)} sampled corner(s) are not in the bundle, e.g. {lost[:3]}')
    return problems


def cmd_score(args):
    proposals = ci.read_jsonl(args.proposals)
    verdicts = json.loads(Path(args.verdicts).read_text(encoding='utf-8'))
    sample = read_sample(args.sample) if args.sample else None
    bundle_rec = None
    if getattr(args, 'bundle', None):
        if not args.sample:
            raise SystemExit('--bundle needs --sample (the sample the bundle was built from)')
        problems = check_bundle(args.bundle, verdicts, args.sample, args.proposals)
        if problems:
            print('\n'.join(problems[:20]))
            raise SystemExit(f'{len(problems)} problem(s): the verdicts cannot be scored against '
                             f'this sample')
        snap = json.loads((Path(args.bundle) / 'snapshot.json').read_text(encoding='utf-8'))
        bundle_rec = {'path': str(args.bundle), 'items_sha256': snap['items_sha256'],
                      'tool_git_sha': snap.get('tool_git_sha')}
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
           'bundle': bundle_rec,
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


def pilot_order(proposals, strata=None, seed=SAMPLE_SEED):
    """ONE fixed seeded order over every proposal of the given strata (all when None): a full
    permutation, random.Random(seed).sample(sorted keys, len(keys)). Every prefix of it is a
    seeded random draw, so the rated sample, its reserve and a send subset are all prefixes of
    one list and can never disagree. (draw_sample's sample(keys, n) is NOT the prefix of a
    longer draw: random.sample picks its algorithm from n.)"""
    if strata:
        unknown = sorted(set(strata) - set(STRATA))
        if unknown:
            raise SystemExit(f'--stratum {unknown}: not a stratum (one of {list(STRATA)})')
    keys = sorted(p['key'] for p in proposals if not strata or p['stratum'] in set(strata))
    return random.Random(seed).sample(keys, len(keys))


def rule_passes(right, n, z):
    """The #169 bar at one look: precision >= RULE_PRECISION and Wilson lower bound (at z)
    >= RULE_WILSON_LO, over n decided labels of which `right` are right."""
    if n == 0:
        return False
    return right / n >= RULE_PRECISION and ci.wilson(right, n, z)[0] >= RULE_WILSON_LO


def stop_rule(looks=PILOT_LOOKS, alpha=PILOT_ALPHA_ONE_SIDED):
    """The pre-registered sequential stop rule and its arithmetic, as sample.json records it.

    Ratings are read in the draw's order, can't-tell skipped (absorbed by the reserve). At each
    look (a count of DECIDED labels) the rule's bar is applied with the Wilson bound at the
    Bonferroni z, alpha / len(looks) one-sided per look, so three looks together keep the
    one-sided 0.025 of the rule's Wilson 95% lower bound (conservative). PASS stops at the
    first look that clears the bar. FAIL stops as soon as the wrong count exceeds what the last
    look could still pass with. Reaching the last look without a pass is FAIL.

    >>> r = stop_rule()
    >>> [(x['look'], x['pass_if_wrong_at_most']) for x in r['looks']], r['fail_once_wrong_exceeds']
    ([(40, 0), (60, 2), (100, 6)], 6)
    """
    looks = tuple(sorted(looks))
    z = NormalDist().inv_cdf(1.0 - alpha / len(looks))
    rows = []
    for n in looks:
        ok = [f for f in range(n + 1) if rule_passes(n - f, n, z)]
        worst = max(ok) if ok else None
        lo_all = ci.wilson(n, n, z)[0]
        rows.append({'look': n, 'pass_if_wrong_at_most': worst,
                     'wilson_lower_all_right': round(lo_all, 4),
                     'wilson_lower_at_worst_pass': (None if worst is None else
                                                    round(ci.wilson(n - worst, n, z)[0], 4)),
                     'unadjusted_z196_pass_if_wrong_at_most':
                         max((f for f in range(n + 1) if rule_passes(n - f, n, 1.96)),
                             default=None)})
    final = rows[-1]['pass_if_wrong_at_most']
    return {'looks': rows, 'z': z, 'alpha_one_sided_total': alpha,
            'alpha_one_sided_per_look': round(alpha / len(looks), 6),
            'correction': 'Bonferroni across looks', 'precision_min': RULE_PRECISION,
            'wilson_lower_min': RULE_WILSON_LO,
            'fail_once_wrong_exceeds': final,
            'reads': 'decided = agree + disagree in the draw order; unsure skipped; the first '
                     'validation by the lead counts; a rating after an unrated earlier label '
                     'is refused (no skipping ahead)'}


def sequential_read(order_keys, verdicts, rule=None):
    """Apply stop_rule to a lead's verdicts ({corner key: 'agree'|'disagree'|'unsure'}) over
    the draw order. Returns {'status': PASS|FAIL|CONTINUE, 'decided', 'right', 'wrong',
    'unsure', 'stopped_at_order', 'look'}. Refuses a rated label that follows an unrated one
    before the stop: the reserve absorbs can't-tells, never a skipped label.

    >>> keys = [f'k{i}' for i in range(50)]
    >>> sequential_read(keys, {k: 'agree' for k in keys[:41]})['status']
    'PASS'
    """
    rule = rule or stop_rule()
    looks = {r['look'] for r in rule['looks']}
    last = max(looks)
    z = rule['z']
    bad = sorted(set(verdicts.values()) - set(PILOT_VERDICTS))
    if bad:
        raise SystemExit(f'unknown verdict(s) {bad}; expected {sorted(PILOT_VERDICTS)}')
    t = {'decided': 0, 'right': 0, 'wrong': 0, 'unsure': 0}
    for i, k in enumerate(order_keys, 1):
        v = verdicts.get(k)
        if v is None:
            later = [x for x in order_keys[i:] if x in verdicts]
            if later:
                raise SystemExit(f'{k} (order {i}) is unrated but {len(later)} later label(s) '
                                 f'are rated, e.g. {later[:3]}: the draw order cannot be '
                                 f'skipped')
            return dict(t, status='CONTINUE', stopped_at_order=None, look=None)
        if v == 'unsure':
            t['unsure'] += 1
            continue
        t['decided'] += 1
        t['right' if v == 'agree' else 'wrong'] += 1
        thr = rule['fail_once_wrong_exceeds']     # None: no outcome passes at the last look
        if thr is not None and t['wrong'] > thr:
            return dict(t, status='FAIL', stopped_at_order=i, look=None)
        if t['decided'] in looks:
            if rule_passes(t['right'], t['decided'], z):
                return dict(t, status='PASS', stopped_at_order=i, look=t['decided'])
            if t['decided'] == last:
                return dict(t, status='FAIL', stopped_at_order=i, look=last)
    return dict(t, status='CONTINUE', stopped_at_order=None, look=None)


def cmd_sample(args):
    proposals = ci.read_jsonl(args.proposals)
    if args.stratum or args.reserve is not None:
        return cmd_pilot_sample(args, proposals)
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


def pilot_roles(n_total, n, reserve, send=None):
    """The role of each position 1..n_total of a pilot order: 'rate' (1..n), 'reserve'
    (the next `reserve`), 'send' (sent but not planned for rating, up to `send`), 'rest'."""
    roles = []
    for i in range(1, n_total + 1):
        if i <= n:
            roles.append('rate')
        elif i <= n + reserve:
            roles.append('reserve')
        elif send is not None and i <= send:
            roles.append('send')
        else:
            roles.append('rest')
    return roles


def cmd_pilot_sample(args, proposals):
    """sample --stratum/--reserve: one seeded order over the whole filtered population, with
    the rated n, the reserve, an optional send prefix and the stop rule recorded."""
    reserve = args.reserve or 0
    if reserve < 0:
        raise SystemExit('--reserve must be >= 0')
    keys = pilot_order(proposals, args.stratum, args.seed)
    if len(keys) < args.n + reserve:
        raise SystemExit(f'only {len(keys)} proposals in {args.stratum}; cannot rate '
                         f'{args.n} + reserve {reserve}')
    if args.send is not None and not args.n + reserve <= args.send <= len(keys):
        raise SystemExit(f'--send {args.send} must cover n + reserve ({args.n + reserve}) and '
                         f'be at most the population ({len(keys)}): the rated labels must be '
                         f'among the sent ones')
    looks = tuple(int(x) for x in args.looks.split(',')) if args.looks else PILOT_LOOKS
    if max(looks) != args.n:
        raise SystemExit(f'the last look ({max(looks)}) must equal --n ({args.n})')
    rule = stop_rule(looks)
    if rule['fail_once_wrong_exceeds'] is None:
        print(f'WARNING: no outcome can pass at the last look ({args.n} decided): the bar '
              f'needs a larger n')
    by_key = {p['key']: p for p in proposals}
    roles = pilot_roles(len(keys), args.n, reserve, args.send)
    out = Path(args.out)
    rows = [{'order': i, 'unit': k.rsplit('#', 1)[0], 'corner': k.rsplit('#', 1)[1],
             'stratum': by_key[k]['stratum'], 'role': r}
            for i, (k, r) in enumerate(zip(keys, roles), 1)]
    ci.write_csv_lf(out, ['order', 'unit', 'corner', 'stratum', 'role'], rows)
    rec = {'schema': 'sidewalk-auto-labeler.corner_labels.sample/2', 'drawn_at': ci.utc_now(),
           'seed': args.seed, 'strata': sorted(args.stratum) if args.stratum else None,
           'population': len(keys), 'n': args.n, 'reserve': reserve, 'send': args.send,
           'roles': dict(Counter(roles)),
           'proposals': {'path': Path(args.proposals).name,
                         'sha256': ci.sha256_file(args.proposals)},
           'rule': 'random.Random(seed).sample(sorted keys of the strata, len(keys)): one full '
                   'order; rate in it, skip can\'t-tell, stop by stop_rule',
           'stop_rule': rule,
           'sample': {'path': out.name, 'sha256': ci.sha256_file(out)}}
    ci.write_text_lf(out.with_suffix('.json'), json.dumps(rec, indent=1) + '\n')
    print(f'seeded order of {len(keys)} proposals ({rec["strata"] or "all strata"}), seed '
          f'{args.seed}: rate {args.n}, reserve {reserve}, send {args.send} -> {out}')


def read_validations(path):
    """{(pano_id, pano_x, pano_y): verdict} from a CSV with columns pano_id, pano_x, pano_y,
    validation (PS's Agree / Disagree / Unsure, any case). A pixel key repeated with another
    verdict is refused: the SELECT must keep one validation per label (the lead's first)."""
    out = {}
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        need = ['pano_id', 'pano_x', 'pano_y', 'validation']
        missing = [c for c in need if c not in (reader.fieldnames or ())]
        if missing:
            raise SystemExit(f'{path}: missing column(s) {missing}; has {reader.fieldnames}')
        for n, row in enumerate(reader, 2):
            k = (row['pano_id'], int(row['pano_x']), int(row['pano_y']))
            v = row['validation'].strip().lower()
            if k in out and out[k] != v:
                raise SystemExit(f'{path} line {n}: {k} has verdicts {out[k]} and {v}')
            out[k] = v
    return out


def pilot_label_ids(order_keys, proposals, rows):
    """[(order, corner key, label_id)] for the pilot labels the server holds, in draw order.

    rows: [{'label_id', 'pano_id', 'pano_x', 'pano_y'}] from the read-only SELECT of the
    campaign's labels; joined on PS's pixel key (pixel_key). A pixel key held by two server
    labels is refused (a duplicate insert); corners with no server label are left out."""
    by_pixel = {}
    for r in rows:
        k = (r['pano_id'], int(r['pano_x']), int(r['pano_y']))
        if k in by_pixel and by_pixel[k] != int(r['label_id']):
            raise SystemExit(f'two server labels share the pixel key {k}: '
                             f'{by_pixel[k]} and {r["label_id"]}')
        by_pixel[k] = int(r['label_id'])
    props = {p['key']: p for p in proposals}
    out = []
    for i, k in enumerate(order_keys, 1):
        lid = by_pixel.get(pixel_key(props[k]))
        if lid is not None:
            out.append((i, k, lid))
    return out


def cmd_pilot_links(args):
    proposals = ci.read_jsonl(args.proposals)
    order, _drops = order_subset(read_corner_order(args.pilot), {p['key']: p for p in proposals},
                                 args.skip_first, read_pano_list(args.exclude_panos)
                                 if args.exclude_panos else ())
    with open(args.labels, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    got = pilot_label_ids(order[:args.first], proposals, rows)
    if args.out:
        ci.write_csv_lf(args.out, ['order', 'corner', 'label_id'],
                        [{'order': i, 'corner': k, 'label_id': lid} for i, k, lid in got])
    ids = ','.join(str(lid) for _i, _k, lid in got)
    print(f'{len(got)} of the first {args.first} pilot corners have a server label '
          f'({len(rows)} rows read). In draw order (city {args.city}):')
    print(f'{args.server.rstrip("/")}/gallery?labelIds={ids}')


def cmd_pilot_score(args):
    rec_s = json.loads(Path(args.pilot).with_suffix('.json').read_text(encoding='utf-8'))
    proposals = ci.read_jsonl(args.proposals)
    drawn = read_corner_order(args.pilot)
    if not set(drawn) <= {p['key'] for p in proposals}:
        raise SystemExit(f'{args.proposals} lacks corners of the pilot order')
    if ci.sha256_file(args.proposals) != rec_s['proposals']['sha256'] and \
            not args.relabeled_proposals:
        raise SystemExit(f'{args.proposals} is not the proposals the pilot order was drawn from '
                         f'(a rebuild of the same corners with another label point: '
                         f'--relabeled-proposals)')
    if ci.sha256_file(args.pilot) != rec_s['sample']['sha256']:
        raise SystemExit(f'{args.pilot} is not the order its .json recorded')
    order, _drops = order_subset(drawn, {p['key']: p for p in proposals}, args.skip_first,
                                 read_pano_list(args.exclude_panos)
                                 if args.exclude_panos else ())
    by_pixel = {}
    for p in proposals:
        k = pixel_key(p)
        if k in by_pixel:
            raise SystemExit(f'two proposals share the PS pixel key {k}')
        by_pixel[k] = p['key']
    vals = read_validations(args.validations)
    verdicts = {by_pixel[k]: v for k, v in vals.items() if k in by_pixel}
    unmatched = len(vals) - len(verdicts)
    rule = rec_s['stop_rule']
    res = sequential_read(order, verdicts, rule)
    out = {'schema': 'sidewalk-auto-labeler.corner_labels.pilot_score/1', 'read_at': ci.utc_now(),
           'pilot': {'path': Path(args.pilot).name, 'sha256': rec_s['sample']['sha256'],
                     'skip_first': args.skip_first, 'exclude_panos': args.exclude_panos,
                     'proposals_sha256': ci.sha256_file(args.proposals)},
           'validations': {'path': Path(args.validations).name,
                           'sha256': ci.sha256_file(args.validations), 'rows': len(vals),
                           'not_pilot_labels': unmatched},
           'stop_rule': rule, **res}
    if args.out:
        ci.write_text_lf(args.out, json.dumps(out, indent=1) + '\n')
    print(f"{res['status']}: {res['decided']} decided ({res['right']} right, {res['wrong']} "
          f"wrong), {res['unsure']} unsure; stopped at order {res['stopped_at_order']}, look "
          f"{res['look']}; {unmatched} validation row(s) not on a pilot label")


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
                        'NA points with no RAMPTYPE do not. REQUIRED, no default; gallery 243 '
                        'settled discount as primary (see --na-read-settled)')
    b.add_argument('--na-read-settled', action='store_true',
                   help='record in build.json that the primary NA read is settled (discount, '
                        'gallery 243, PR #180; NA_READ_SETTLEMENT) and whether --na-read is it')
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
    b.add_argument('--label-point', choices=LABEL_POINT_RULES, default=LABEL_POINT_DEFAULT,
                   help="where the label goes on the corner bisector. site-median (default since "
                        "2026-11-11): the unit's median centre-to-site distance, else its "
                        "stratum's, less --label-inset-m, clamped to [stratum p10, corner "
                        "offset]. corner: the corner point itself (every earlier build). The "
                        "corner decision and pano choice use the corner point either way")
    b.add_argument('--label-inset-m', type=float, default=LABEL_INSET_M,
                   help='site-median: metres subtracted toward the road (default %(default)s)')
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
    e.add_argument('--stratum', action='append', choices=STRATA,
                   help='export only proposals of this stratum (repeatable; recorded)')
    e.add_argument('--only-corners', metavar='CSV',
                   help='export only these corners (a pilot order or sample CSV with unit, '
                        'corner; ordered by its `order` column). A key that is not a proposal '
                        'is refused. Lines then follow the draw order (detections get '
                        '`pilot_order`)')
    e.add_argument('--only-first', type=int, metavar='N',
                   help='with --only-corners: only its first N corners (the send prefix), '
                        'counted after --skip-first and --exclude-panos')
    e.add_argument('--skip-first', type=int, default=0, metavar='K',
                   help='with --only-corners: start after its first K positions (e.g. the '
                        'corners a smoke send already put on the server)')
    e.add_argument('--exclude-panos', metavar='TXT',
                   help="with --only-corners: drop corners whose chosen pano is listed (one id "
                        "per line), e.g. a smoke send's panos; recorded with the dropped "
                        "positions")
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser('score', help='the pre-registered corner rule against blind ratings')
    s.add_argument('--proposals', required=True)
    s.add_argument('--verdicts', required=True, help='a gallery verdicts__<rater>.json')
    s.add_argument('--sample', help="sample.csv: score only the sample drawn for this rule")
    s.add_argument('--bundle', help='the precision gallery built from --sample (corner_gallery.py '
                                    'build --items-from): check the verdicts against it first')
    s.add_argument('--out', help='write the score JSON here')
    s.set_defaults(fn=cmd_score)

    m = sub.add_parser('sample', help='draw the seeded corner sample for a future gallery')
    m.add_argument('--proposals', required=True)
    m.add_argument('--n', type=int, default=SAMPLE_N)
    m.add_argument('--seed', type=int, default=SAMPLE_SEED)
    m.add_argument('--out', required=True, help='sample.csv (sample.json is written beside it)')
    m.add_argument('--stratum', action='append', choices=STRATA,
                   help='pilot mode: draw only from this stratum (repeatable). With --stratum or '
                        '--reserve the output is ONE seeded order over the whole filtered '
                        'population (columns order, unit, corner, stratum, role)')
    m.add_argument('--reserve', type=int, default=None,
                   help='pilot mode: reserve positions after the n rated ones, which absorb '
                        "can't-tell ratings")
    m.add_argument('--send', type=int, default=None,
                   help='pilot mode: the send prefix (role `send`); must cover n + reserve')
    m.add_argument('--looks', default=None,
                   help=f'pilot mode: comma list of decided counts at which the stop rule looks '
                        f'(default {",".join(map(str, PILOT_LOOKS))}; the last must equal --n)')
    m.set_defaults(fn=cmd_sample)

    q = sub.add_parser('pilot-score', help="the sequential stop rule over a lead's validations")
    q.add_argument('--pilot', required=True, help='the pilot order CSV (sample --stratum ...)')
    q.add_argument('--proposals', required=True, help='the proposals it was drawn from')
    q.add_argument('--validations', required=True,
                   help="CSV pano_id,pano_x,pano_y,validation: the lead's first validation of "
                        "each label (the read-only SELECT in docs/corner-labels-169-pilot.md)")
    q.add_argument('--out', help='write the read JSON here')
    q.add_argument('--skip-first', type=int, default=0,
                   help='the rated order starts after this many positions (the smoke labels)')
    q.add_argument('--exclude-panos', help='drop corners on these panos from the order (as export)')
    q.add_argument('--relabeled-proposals', action='store_true',
                   help='--proposals is a rebuild of the drawn corners with another label point '
                        '(same keys; the pixel key changes, so pass the rebuild that was sent)')
    q.set_defaults(fn=cmd_pilot_score)

    k = sub.add_parser('pilot-links', help="the pilot's server label ids in draw order, as a "
                                           'Gallery review-list URL')
    k.add_argument('--pilot', required=True, help='the pilot order CSV')
    k.add_argument('--proposals', required=True)
    k.add_argument('--labels', required=True,
                   help='CSV label_id,pano_id,pano_x,pano_y of the campaign (read-only SELECT)')
    k.add_argument('--first', type=int, default=120,
                   help='how many positions of the order to link (default 120 = rate + reserve)')
    k.add_argument('--server', required=True, help='e.g. https://sidewalk-<city>-test.cs.washington.edu')
    k.add_argument('--city', default='vancouver', help='label ids are per city; printed with them')
    k.add_argument('--out', help='write order,corner,label_id here')
    k.add_argument('--skip-first', type=int, default=0,
                   help='skip the first K positions (the smoke labels)')
    k.add_argument('--exclude-panos', help='drop corners on these panos (as export)')
    k.set_defaults(fn=cmd_pilot_links)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
