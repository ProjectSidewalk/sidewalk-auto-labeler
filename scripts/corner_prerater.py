"""VLM pre-rater for corner galleries, and the inventory stand-in question (issue #185).

Plan: corner-prerater-absence-2026-11-15.md (Claude Fable 5.2, reviewed by Claude Opus 5.5).
Write-up: docs/corner-prerater-185.md. Only Stage 0 is approved and implemented here.

Subcommands:

  gate     Stage 0, the stop gate for the stand-in question. $0, local, no model call.
           Would the #169 NoCurbRamp rule still clear its 0.90 bar if the city inventory were
           not there to remove corners? Adding the 283 corners the inventory removes back to
           the 3,070 proposals of the split table gives

               P_free = (3,070 * p + 283 * (1 - v)) / 3,353

           with p the rated absence precision of the proposals (gallery 169, split table,
           96 of 97 decided blind ratings) and v the share of inventory-removed corners that
           show a ramp (gallery 243, the fusion-absent corners holding an `Available` point,
           5 of 45 decided). The gate is fixed by the plan: the worst case (p at its Wilson
           lower bound, v at its Wilson upper bound) must be >= 0.90. Also applies the
           over-merge filter (below) and reports v with and without it, and writes the
           sensitivity readings listed in the write-up. Output: <out>/gate.json.
  figures  redraw docs/figures/corner-prerater-185/ from the committed gate.json.

  export   Stage 1 (approved 2026-11-15): the 381 judged corners of galleries 243 and 169 ->
           <out>/stage1/requests.jsonl + requests.json, with each corner's aerial rendered as
           the page draws it (stage1/aerial/, git-ignored, regenerable). The paid calls are
           made by scripts/corner_vlm.py; logic in scripts/corner_prerater_s1.py.
  score    Stage 1 answers -> <out>/stage1/score.json: present units caught, flag rate on
           the blind-absent corners, per arm and per gallery, and the pre-registered kill.
           Stage 1b and Stage 2 are not approved and not implemented.

--check (on `gate`) recomputes everything and compares it with the committed gate.json,
ignoring only the build time and the tool's git sha. Exit 1 on any difference.

The over-merge filter (plan, Stage 0): an `Available` inventory point in an absent corner that
lies nearer to a fused site of a present *adjacent* corner of the same unit (the sectors either
side of it) than to its own corner point is reassigned to that neighbour. A corner whose every
`Available` point is reassigned, and which holds no other excluding inventory class, leaves the
inventory-removed set. Amendment A1 of docs/corner-inventory.md is why: merged legs move sector
bounds, so a ramp the detector found can sit in one sector while the city's point for the same
ramp falls in the next.

Every corner key carries its table: `split:vancouver:res:n47150161#2` or
`old:vancouver:...`, because 2,650 of 4,429 units have different corners in the two tables.

    python scripts/corner_prerater.py gate --out runs/vancouver/corner_prerater185
    python scripts/corner_prerater.py gate --out runs/vancouver/corner_prerater185 --check
    python scripts/corner_prerater.py figures --out runs/vancouver/corner_prerater185

`gate` needs shapely >= 2 for the sidewalk sensitivity (the conda env has it); pass
`--sidewalks none` to skip that one reading on a machine without it (`--check` then fails, by
design, because the committed file has it).
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import corner_inventory as ci  # noqa: E402
import corner_labels as cl  # noqa: E402

SCHEMA = 'sidewalk-auto-labeler.corner_prerater.gate/1'
ISSUE = 'sidewalk-auto-labeler#185'
ARM_KEY = 'fusion/primary'
STRATA = cl.STRATA
NA_READ = 'discount'
GATE_BAR = 0.90
#: the plan's rounded inputs; the recomputed values must round to these (3 places)
PLAN_P = (0.990, 0.944, 0.998)
PLAN_V = (5, 45, 0.048, 0.235)
PLAN_COUNTS = {'proposed': 3070, 'excluded_inventory': 283}

# Defaults are relative to the repo root of a worktree under D:\Git\labeler-wt\.
D_SPLIT = '../corner169/runs/vancouver/corner_inventory169_split/corners_full.jsonl'
D_OLD = '../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl'
D_RUN = '../../sal-vancouver/runs/vancouver'
D_EXTRA = ['../posdetect241/runs/vancouver_posdetect241',
           '../posdetect241/runs/vancouver_posdetect241_gsv']
D_G243 = '../gallery243/runs/vancouver/corner_gallery243'
D_G169 = 'runs/vancouver/corner_gallery169_split'
D_LABELS = 'runs/vancouver/corner_labels169_split/na_discount_sw'
D_SIDEWALKS = '../corner169/runs/vancouver/corner_labels169/sidewalks/cov_sidewalks.geojson'
D_FIG = 'docs/figures/corner-prerater-185'

#: sha256 of every input, as the #169 builds and gallery 243 recorded them
PINS = {
    'split_corners': 'b50da32e1bb0b86f5d93fcd0de0cead1be620d8ac39f309d3abd2461f87ce9fc',
    'old_corners': 'ec20606098d0e8b34113b4b328e57d3737b7669d01a37f1d6d722faab4681f8d',
    'sites': '9d1fc51ac2867d6247bdf2e0b1c6b5c9fccbbf9a566771f0222bb2728055c914',
    'extra:vancouver_posdetect241:sites':
        'd38c6fa08f9774de842c77c72ad51346daebb918270c90b319b662bd8534320b',
    'extra:vancouver_posdetect241_gsv:sites':
        'df0dec5c6c9456248a7ee1aee7c4e3e7c4c50643780a5105dd7451b3e7e7614d',
    'inventory': 'c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87',
    'g243_items': 'bbed3d6ddaf1835a66da3afb274c3f338cc80512d4af7c4567c96c1bf47a1459',
    # the complete export (95 of 95 units), branch corner-gallery-243-verdicts; the copy on
    # corner-labels-169 is the partial export of 50 units and is NOT what PR #180 scored
    'g243_verdicts': '30a5008d125f143d6cc536e9787676310fc474b9c4cc127f59a419a16e2e80bd',
    'g169_items': 'e7648a487a92ef508d488e40d8425de333f0d3719d2412a72a80138e15e26910',
    'g169_verdicts': 'da3f01147df9f502584d84c5b4b479c5eedc37ee0d7dce99d8f3e16fe88d0a64',
    'g169_sample': '14911083669d3fa9793754b926e85cd7d1a0348218d82b44acf1f91c0eaf04ce',
    'sidewalks': '11503ed4cd7def77749c8443143f2bce73e556e8543fc17d1981a514f7bf0eff',
}


# ---------------------------------------------------------------------------- the math

def p_free(n_prop, p, n_excl, v):
    """Absence precision of the rule with the inventory removed.

    The n_prop proposals keep precision p; the n_excl corners the inventory removed come
    back, and a share v of them shows a ramp, so they are right at 1 - v.

    >>> round(p_free(3070, 0.944, 283, 0.235), 4)
    0.9289
    """
    return (n_prop * p + n_excl * (1.0 - v)) / (n_prop + n_excl)


def breakeven_v(n_prop, p, n_excl, bar=GATE_BAR):
    """The v at which p_free equals bar. Above 1: the bar holds whatever the removed corners
    show; below 0: it fails even if none shows a ramp.

    >>> round(breakeven_v(3070, 0.944, 283), 3)
    0.577
    """
    return 1.0 - (bar * (n_prop + n_excl) - n_prop * p) / n_excl


def wilson(k, n):
    """Wilson 95% interval as a list of two floats (corner_inventory.wilson)."""
    lo, hi = ci.wilson(k, n)
    return [lo, hi]


def rate_block(k, n):
    """{k, n, share, wilson95} for k of n (shares unrounded; rounding is for display)."""
    return {'k': k, 'n': n, 'share': k / n if n else None, 'wilson95': wilson(k, n)}


# ------------------------------------------------------------------ the over-merge filter

def adjacent_present(corners, i, key=ARM_KEY):
    """The corners in the sectors either side of corner i that are present under key."""
    n = len(corners)
    if n < 2:
        return []
    idx = sorted({(i - 1) % n, (i + 1) % n} - {i})
    return [corners[j] for j in idx if corners[j]['state'][key] == 'present']


def reassign(corner_en, available_en, neighbour_sites_en, other_excluding):
    """Over-merge filter for one absent corner (all positions in one local metric frame).

    Each `Available` point that lies strictly nearer to some neighbour site than to its own
    corner point moves to that neighbour. Returns (n_moved, leaves) where leaves says the
    corner drops out of the inventory-removed set: every Available point moved and no other
    excluding class holds it there.

    >>> reassign((0, 0), [(9, 0)], [(10, 0)], False)
    (1, True)
    >>> reassign((0, 0), [(4, 0)], [(10, 0)], False)
    (0, False)
    >>> reassign((0, 0), [(9, 0)], [(10, 0)], True)
    (1, False)
    """
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    moved = 0
    for pt in available_en:
        if neighbour_sites_en and min(d(pt, s) for s in neighbour_sites_en) < d(pt, corner_en):
            moved += 1
    leaves = bool(available_en) and moved == len(available_en) and not other_excluding
    return moved, leaves


def excluded(c):
    """The #169 rule's inventory exclusion under the settled `discount` read."""
    return cl.inventory_excludes(c['inv_counts'], NA_READ)


def pool(records, key=ARM_KEY):
    """(unit record, corner) for every intersection corner absent under key that the
    inventory excludes under the discount read: the corners P_free adds back."""
    for rec in records:
        if rec['type'] not in STRATA:
            continue
        for c in rec.get('corners') or ():
            if c['state'][key] == 'absent' and excluded(c):
                yield rec, c


class InventoryIndex:
    """Positions of the city inventory points. corners_full.jsonl stores each point's record
    but not its position, and UNITID is not unique in the file (42 points have none, some ids
    repeat), so a point is found by its id among the features within the unit window, and the
    match must be unique."""

    def __init__(self, path):
        feats = json.loads(Path(path).read_text(encoding='utf-8'))['features']
        self.by_id = {}
        for f in feats:
            lng, lat = f['geometry']['coordinates'][:2]
            rec = ci.inv_record(f.get('properties') or {})
            self.by_id.setdefault(rec['unit_id'], []).append((lat, lng, rec))

    def locate(self, point, centre_lat, centre_lng, window_m):
        hits = [(lat, lng) for lat, lng, rec in self.by_id.get(point['unit_id'], ())
                if rec == point and
                geo.haversine_m(lat, lng, centre_lat, centre_lng) <= window_m + 0.5]
        if len(hits) != 1:
            raise SystemExit(f"inventory point {point['unit_id']}: {len(hits)} matches within "
                             f'{window_m} m of ({centre_lat}, {centre_lng}); expected 1')
        return hits[0]


def load_site_positions(run_dir, extra_dirs):
    """{site_id: (lat, lng)} with extra runs' ids prefixed as corner_inventory does."""
    out = {}
    for d, prefix in [(Path(run_dir), '')] + [(Path(e), Path(e).resolve().name + ':')
                                             for e in extra_dirs]:
        with open(d / 'sites.jsonl', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    out[f"{prefix}{s['site_id']}" if prefix else s['site_id']] = (s['lat'],
                                                                                  s['lng'])
    return out


def filter_corner(rec, c, sites, inv, window_m):
    """(n_available, n_moved, leaves) for one pool corner under the over-merge filter."""
    fr = geo.LocalFrame(rec['lat'], rec['lng'])
    avail = [p for p in c['inventory'] if p['class'] == 'Available']
    avail_en = [fr.to_enu(*inv.locate(p, rec['lat'], rec['lng'], window_m)) for p in avail]
    nb_en = [fr.to_enu(*sites[s]) for x in adjacent_present(rec['corners'], c['corner'])
             for s in x['site_ids']]
    other = any(v > 0 for k, v in c['inv_counts'].items()
                if k not in ('Available', 'NA_noramp'))
    moved, leaves = reassign(fr.to_enu(c['lat'], c['lng']), avail_en, nb_en, other)
    return len(avail), moved, leaves


# ------------------------------------------------------------------------- the ratings

def blind_tally(verdicts, keys):
    """{absent, present, cant_tell, unrated} over (unit, corner) keys from the blind reads."""
    t = {'absent': 0, 'present': 0, 'cant_tell': 0, 'unrated': 0}
    units = verdicts.get('units') or {}
    for unit, corner in keys:
        v = (((units.get(unit) or {}).get('blind') or {}).get(str(corner)) or {}).get('verdict')
        t[v if v in t else 'unrated'] += 1
    return t


def judged_pool(items, old_by_unit):
    """Gallery 243 corners that hold an `Available` point (the v population), each with its
    old-table record. Every one must be absent under ARM_KEY in the old table."""
    out = []
    for it in items:
        rec = old_by_unit[it['unit']]
        for c in it['corners']:
            if c['inv_counts']['Available'] < 1:
                continue
            oc = rec['corners'][c['corner']]
            if abs(oc['lat'] - c['lat']) > 1e-6 or abs(oc['lng'] - c['lng']) > 1e-6:
                raise SystemExit(f"{it['unit']}#{c['corner']}: gallery and old table disagree")
            if oc['state'][ARM_KEY] != 'absent':
                raise SystemExit(f"{it['unit']}#{c['corner']}: not absent in the old table")
            out.append((rec, oc))
    return out


# ------------------------------------------------------------------------------ gate

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned(name, path):
    got = sha256_file(path)
    if got != PINS[name]:
        raise SystemExit(f'{path}: sha256 {got} is not the pinned {PINS[name]} ({name})')
    return {'path': str(path).replace('\\', '/'), 'sha256': got}


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def window_of(corners_path):
    """The unit window (m) the corner table was built with, from its build.json."""
    return float(json.loads((Path(corners_path).parent / 'build.json').read_text(
        encoding='utf-8'))['params']['window_m'])


def r4(x):
    return None if x is None else round(x, 4)


def compute(args):
    inputs = {}
    split_path, old_path = Path(args.split_corners), Path(args.old_corners)
    inputs['split_corners'] = pinned('split_corners', split_path)
    inputs['old_corners'] = pinned('old_corners', old_path)
    run_dir = Path(args.run_dir)
    inputs['sites'] = pinned('sites', run_dir / 'sites.jsonl')
    for e in args.extra_run:
        name = f'extra:{Path(e).resolve().name}:sites'
        inputs[name] = pinned(name, Path(e) / 'sites.jsonl')
    inputs['inventory'] = pinned('inventory', run_dir / 'inventory_oracle/inventory.geojson')
    g243, g169, labels = Path(args.gallery243), Path(args.gallery169), Path(args.labels)
    inputs['g243_items'] = pinned('g243_items', g243 / 'items.jsonl')
    inputs['g243_verdicts'] = pinned('g243_verdicts', g243 / 'verdicts__jonf.json')
    inputs['g169_items'] = pinned('g169_items', g169 / 'items.jsonl')
    inputs['g169_verdicts'] = pinned('g169_verdicts', g169 / 'verdicts__jonf.json')
    inputs['g169_sample'] = pinned('g169_sample', labels / 'sample.csv')
    counts_path = labels / 'counts.csv'
    inputs['labels_counts'] = {'path': str(counts_path).replace('\\', '/'),
                               'sha256': sha256_file(counts_path)}

    split = read_jsonl(split_path)
    window_m, old_window = window_of(split_path), window_of(old_path)
    with open(counts_path, encoding='utf-8', newline='') as f:
        tot = {r['stratum']: r for r in csv.DictReader(f)}['intersections']
    n_prop, n_excl_csv = int(tot['proposed']), int(tot['excluded_inventory'])

    # -- N_excl: recomputed from the split table, checked against the build's counts.csv
    split_pool = list(pool(split))
    n_excl = len(split_pool)
    if n_excl != n_excl_csv or n_prop != PLAN_COUNTS['proposed'] \
            or n_excl != PLAN_COUNTS['excluded_inventory']:
        raise SystemExit(f'counts differ: pool {n_excl}, counts.csv {n_excl_csv} excluded / '
                         f'{n_prop} proposed, plan {PLAN_COUNTS}')
    unlabeled = labels / 'unlabeled.csv'
    with open(unlabeled, encoding='utf-8', newline='') as f:
        csv_keys = {(r['unit'], int(r['corner'])) for r in csv.DictReader(f)
                    if r['reason'] == 'inventory'}
    if csv_keys != {(rec['unit'], c['corner']) for rec, c in split_pool}:
        raise SystemExit('the recomputed pool differs from unlabeled.csv reason=inventory')
    inputs['labels_unlabeled'] = {'path': str(unlabeled).replace('\\', '/'),
                                  'sha256': sha256_file(unlabeled)}
    classes = {'available': 0, 'no_available': 0}
    for _, c in split_pool:
        classes['available' if c['inv_counts']['Available'] > 0 else 'no_available'] += 1

    # -- p: gallery 169 split, blind reads of the sample
    with open(labels / 'sample.csv', encoding='utf-8', newline='') as f:
        sample = [(r['unit'], int(r['corner'])) for r in csv.DictReader(f)]
    t169 = blind_tally(json.loads((g169 / 'verdicts__jonf.json').read_text(encoding='utf-8')),
                       sample)
    n169 = t169['absent'] + t169['present']
    p_blk = rate_block(t169['absent'], n169)

    # -- v: gallery 243, fusion-absent corners holding an Available point, blind reads
    items = read_jsonl(g243 / 'items.jsonl')
    old_by_unit = {r['unit']: r for r in read_jsonl(old_path)}
    v243 = json.loads((g243 / 'verdicts__jonf.json').read_text(encoding='utf-8'))
    judged = judged_pool(items, old_by_unit)
    keys = [(rec['unit'], c['corner']) for rec, c in judged]
    t243 = blind_tally(v243, keys)
    v_blk = rate_block(t243['present'], t243['present'] + t243['absent'])
    # the same population read as the #169 rule reads it: any excluding class, not only
    # Available (a sensitivity on the population definition)
    excl_keys = [(it['unit'], c['corner']) for it in items for c in it['corners']
                 if old_by_unit[it['unit']]['corners'][c['corner']]['state'][ARM_KEY] == 'absent'
                 and excluded(c)]
    t243x = blind_tally(v243, excl_keys)

    # -- over-merge filter
    sites = load_site_positions(run_dir, args.extra_run)
    inv = InventoryIndex(run_dir / 'inventory_oracle/inventory.geojson')
    j_left = []
    j_leaves = 0
    for rec, c in judged:
        _, _, leaves = filter_corner(rec, c, sites, inv, old_window)
        j_leaves += leaves
        if not leaves:
            j_left.append((rec['unit'], c['corner']))
    t243f = blind_tally(v243, j_left)
    vf_blk = rate_block(t243f['present'], t243f['present'] + t243f['absent'])
    s_moved_any = s_leaves = s_adj = 0
    leaves_keys = []
    for rec, c in split_pool:
        if c['inv_counts']['Available'] < 1:
            continue
        s_adj += bool(adjacent_present(rec['corners'], c['corner']))
        _, moved, leaves = filter_corner(rec, c, sites, inv, window_m)
        s_moved_any += moved > 0
        if leaves:
            s_leaves += 1
            leaves_keys.append(f"split:{rec['unit']}#{c['corner']}")
    judged_units_with_present = sum(
        1 for u in {k[0] for k in keys}
        if any(x['state'][ARM_KEY] == 'present' for x in old_by_unit[u]['corners']))

    # -- sidewalk: how many of the removed corners the sidewalk rule would keep
    sw = None
    if args.sidewalks != 'none':
        inputs['sidewalks'] = pinned('sidewalks', Path(args.sidewalks))
        rule = json.loads((labels / 'build.json').read_text(encoding='utf-8'))['sidewalk_rule']
        layer = cl.SidewalkPresence(args.sidewalks, rule['radius_m'],
                                 {k: set(v) for k, v in rule['keep'].items()})
        with_sw = sum(1 for _, c in split_pool if layer.has_sidewalk(c['lat'], c['lng']))
        kept_absent = sum(1 for rec in split if rec['type'] in STRATA
                          for c in rec.get('corners') or ()
                          if c['state'][ARM_KEY] == 'absent' and not excluded(c)
                          and layer.has_sidewalk(c['lat'], c['lng']))
        if kept_absent != n_prop:
            raise SystemExit(f'sidewalk layer recount {kept_absent} != {n_prop} proposals')
        sw = {'radius_m': rule['radius_m'], 'keep': rule['keep'],
              'excluded_with_sidewalk': with_sw,
              'proposals_recounted': kept_absent}

    # -- the gate and the readings
    p_pt, (p_lo, p_hi) = p_blk['share'], p_blk['wilson95']
    v_pt, (v_lo, v_hi) = v_blk['share'], v_blk['wilson95']
    for got, want in ((p_pt, PLAN_P[0]), (p_lo, PLAN_P[1]), (p_hi, PLAN_P[2]),
                      (v_lo, PLAN_V[2]), (v_hi, PLAN_V[3])):
        if round(got, 3) != want:
            raise SystemExit(f'recomputed {got:.4f} does not round to the plan value {want}')
    if (v_blk['k'], v_blk['n']) != PLAN_V[:2]:
        raise SystemExit(f"v is {v_blk['k']}/{v_blk['n']}, plan {PLAN_V[:2]}")
    worst = p_free(n_prop, p_lo, n_excl, v_hi)
    point = p_free(n_prop, p_pt, n_excl, v_pt)
    verdict = 'PASS' if worst >= GATE_BAR else 'FAIL'

    def reading(name, p, v, n_p=n_prop, n_e=n_excl, extra=None):
        out = {'reading': name, 'p': r4(p), 'v': r4(v), 'n_prop': n_p, 'n_excl': n_e,
               'p_free': r4(p_free(n_p, p, n_e, v)),
               'clears_bar': p_free(n_p, p, n_e, v) >= GATE_BAR}
        if extra:
            out.update(extra)
        return out

    p_ct = rate_block(t169['absent'], t169['absent'] + t169['present'] + t169['cant_tell'])
    v_ct = rate_block(t243['present'] + t243['cant_tell'],
                      t243['present'] + t243['absent'] + t243['cant_tell'])
    vf_hi = vf_blk['wilson95'][1]
    readings = [
        reading('gate: worst case (p Wilson lower, v Wilson upper)', p_lo, v_hi),
        reading('point estimates', p_pt, v_pt),
        reading('p point, v = 1 (every removed corner a ramp)', p_pt, 1.0),
        reading(f"can't tell read against the rule (p = {p_ct['k']}/{p_ct['n']}, "
                f"v = {v_ct['k']}/{v_ct['n']}), Wilson worst",
                p_ct['wilson95'][0], v_ct['wilson95'][1]),
        reading('Available-less removed corners at v = 1, rest at v Wilson upper', p_lo,
                (classes['available'] * v_hi + classes['no_available']) / n_excl),
        reading('over-merge filter: reassigned corners counted at p, rest at filtered v '
                'Wilson upper', p_lo, vf_hi, n_p=n_prop + s_leaves, n_e=n_excl - s_leaves),
    ]
    if sw:
        readings.append(reading('only removed corners the sidewalk rule keeps (worst case)',
                                p_lo, v_hi, n_e=sw['excluded_with_sidewalk']))
    return {
        'schema': SCHEMA, 'issue': ISSUE,
        'plan': 'corner-prerater-absence-2026-11-15.md, Stage 0',
        'gate': {'rule': 'P_free(p Wilson lower, v Wilson upper) >= 0.90', 'bar': GATE_BAR,
                 'p_free_worst': r4(worst), 'p_free_point': r4(point), 'verdict': verdict},
        'counts': {'proposed': n_prop, 'excluded_inventory': n_excl,
                   'excluded_by_class': classes, 'table': 'split',
                   'arm': ARM_KEY, 'na_read': NA_READ},
        'p': {'source': 'gallery 169 split, blind, sample.csv', 'tally': t169,
              **{k: (r4(v) if not isinstance(v, list) else [r4(x) for x in v])
                 for k, v in p_blk.items()}},
        'v': {'source': 'gallery 243, fusion-absent corners with an Available point, blind',
              'tally': t243,
              **{k: (r4(v) if not isinstance(v, list) else [r4(x) for x in v])
                 for k, v in v_blk.items()},
              'any_excluding_class_tally': t243x},
        'over_merge_filter': {
            'rule': 'an Available point strictly nearer to a fused site of a present adjacent '
                    'corner than to its own corner point is reassigned; a corner leaves when '
                    'all its Available points do and no other excluding class holds it',
            'judged': {'corners': len(judged), 'units': len({k[0] for k in keys}),
                       'units_with_a_present_corner': judged_units_with_present,
                       'leave': j_leaves, 'tally_after': t243f,
                       'v_after': r4(vf_blk['share']),
                       'v_after_wilson95': [r4(x) for x in vf_blk['wilson95']]},
            'split_pool': {'with_available': classes['available'],
                           'adjacent_present': s_adj, 'any_point_moved': s_moved_any,
                           'leave': s_leaves, 'leave_keys': leaves_keys},
        },
        'cant_tell_against': {'p_wilson_lower': r4(p_ct['wilson95'][0]),
                              'v_wilson_upper': r4(v_ct['wilson95'][1])},
        'breakeven_v': {'at_p_lower': r4(breakeven_v(n_prop, p_lo, n_excl)),
                        'at_p_point': r4(breakeven_v(n_prop, p_pt, n_excl))},
        'sidewalk': sw,
        'readings': readings,
        'inputs': inputs,
    }


VOLATILE = ('built_at', 'tool_git_sha')


def cmd_gate(args):
    out = Path(args.out)
    res = compute(args)
    path = out / 'gate.json'
    if args.check:
        old = json.loads(path.read_text(encoding='utf-8'))
        a = {k: v for k, v in old.items() if k not in VOLATILE}
        b = {k: v for k, v in json.loads(json.dumps(res)).items() if k not in VOLATILE}
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            print(f'CHECK FAILED: {path} differs in {diff}')
            return 1
        print(f"CHECK OK: {path} reproduces ({res['gate']['verdict']}, worst case "
              f"{res['gate']['p_free_worst']})")
        return 0
    res['built_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    res['tool_git_sha'] = git_sha()
    out.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(res, indent=1) + '\n').encode('utf-8'))
    g = res['gate']
    print(f"Stage 0 gate: {g['verdict']}: worst case P_free {g['p_free_worst']} "
          f"(bar {GATE_BAR}), point {g['p_free_point']}")
    for r in res['readings']:
        print(f"  {r['p_free']:.4f} {'ok ' if r['clears_bar'] else 'LOW'} {r['reading']}")
    print(f'wrote {path}')
    return 0


# --------------------------------------------------------------------------- figures

def figure_rows(gate):
    """Curves of P_free over v in [0, 1] at three values of p, from gate.json."""
    n_p, n_e = gate['counts']['proposed'], gate['counts']['excluded_inventory']
    ps = [('p point', gate['p']['share']), ('p Wilson lower', gate['p']['wilson95'][0])]
    ps.append(("p Wilson lower, can't tell as present",
               gate['cant_tell_against']['p_wilson_lower']))
    rows = []
    for name, p in ps:
        for i in range(101):
            v = i / 100
            rows.append({'curve': name, 'p': p, 'v': v, 'p_free': round(p_free(n_p, p, n_e, v), 5)})
    return rows


def cmd_figures(args):
    gate = json.loads((Path(args.out) / 'gate.json').read_text(encoding='utf-8'))
    fig_dir = Path(args.fig_dir)
    (fig_dir / 'data').mkdir(parents=True, exist_ok=True)
    rows = figure_rows(gate)
    with open(fig_dir / 'data' / 'stage0_pfree.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, ['curve', 'p', 'v', 'p_free'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {'p point': ('#2a78d6', '-'), 'p Wilson lower': ('#eb6834', '-'),
              "p Wilson lower, can't tell as present": ('#7a7a7a', '--')}
    for name, (col, ls) in styles.items():
        xs = [r['v'] for r in rows if r['curve'] == name]
        ys = [r['p_free'] for r in rows if r['curve'] == name]
        p = next(r['p'] for r in rows if r['curve'] == name)
        ax.plot(xs, ys, color=col, ls=ls, lw=1.8, label=f'{name} (p = {p:.3f})')
    v = gate['v']
    ax.axvspan(v['wilson95'][0], v['wilson95'][1], color='#999999', alpha=0.15, lw=0,
               label=f"v Wilson 95% ({v['k']} of {v['n']} judged)")
    ax.axvline(v['share'], color='#999999', lw=1)
    ax.axhline(GATE_BAR, color='black', lw=1, ls=':')
    ax.text(0.99, GATE_BAR + 0.002, 'bar 0.90', ha='right', va='bottom', fontsize=9)
    wv, wp = v['wilson95'][1], gate['gate']['p_free_worst']
    ax.plot([wv], [wp], 'o', color='#eb6834', ms=6)
    ax.annotate(f'gate (worst case) {wp:.3f}', (wv, wp), xytext=(wv + 0.04, wp + 0.012),
                fontsize=9, arrowprops={'arrowstyle': '-', 'lw': 0.8})
    ax.set_xlim(0, 1)
    ax.set_ylim(0.84, 1.0)
    ax.set_xlabel('v: share of the 283 inventory-removed corners that show a ramp')
    ax.set_ylabel('absence precision without the inventory')
    ax.set_title('Stage 0: the #169 rule without the inventory (Vancouver, split table)',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    path = fig_dir / 'stage0_pfree.png'
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')
    print(f"wrote {fig_dir / 'data' / 'stage0_pfree.csv'}")
    return 0


# ---------------------------------------------------------------------------- stage 1

def cmd_export(args):
    import corner_prerater_s1 as s1
    st = Path(args.out) / 'stage1'
    st.mkdir(parents=True, exist_ok=True)
    reqs = s1.build_requests(REPO_ROOT, st, render=not args.no_render)
    path = st / 'requests.jsonl'
    s1.write_jsonl(path, reqs)
    counts = {}
    for q in reqs:
        k = (q['gallery'], q['truth']['blind'])
        counts[f'{k[0]}:{k[1]}'] = counts.get(f'{k[0]}:{k[1]}', 0) + 1
    meta = {'schema': 'sidewalk-auto-labeler.corner_prerater.requests/1', 'issue': ISSUE,
            'plan': 'corner-prerater-absence-2026-11-15.md, Stage 1',
            'requests_sha256': s1.sha256_file(path), 'prompt_sha256': s1.prompt_sha256(),
            'prompt_texts': s1.PROMPT_TEXTS, 'response_schema': s1.RESPONSE_SCHEMA,
            'models': s1.MODELS, 'arms': {k: list(v) for k, v in s1.ARMS.items()},
            'kill_arms': list(s1.KILL_ARMS), 'n_corners': len(reqs),
            'n_crops': sum(len(q['crops']) for q in reqs), 'blind_counts': counts,
            'smoke_keys': s1.smoke_keys(reqs),
            'galleries': {g: {'bundle': v[0], 'items_sha256': v[1], 'verdicts_sha256': v[2],
                              'rubric_version': v[3]} for g, v in s1.GALLERIES.items()},
            'tool_git_sha': git_sha()}
    (st / 'requests.json').write_bytes((json.dumps(meta, indent=1) + '\n').encode('utf-8'))
    print(f"wrote {path}: {len(reqs)} corners, {meta['n_crops']} crops, {counts}")
    return 0


def cmd_score(args):
    import corner_prerater_s1 as s1
    st = Path(args.out) / 'stage1'
    reqs = s1.read_jsonl(st / 'requests.jsonl')
    resp = {}
    for arm in s1.ARMS:
        p = st / 'responses' / f'{arm}.jsonl'
        resp[arm] = s1.read_jsonl(p) if p.exists() else []
    scores, per_gallery, kinds = {}, {}, {}
    for arm in s1.ARMS:
        if not resp[arm]:
            continue
        ans = s1.arm_answers(reqs, resp[arm], arm)
        scores[arm] = s1.score_arm(reqs, ans)
        per_gallery[arm] = {g: s1.score_arm(reqs, ans, g) for g in s1.GALLERIES}
        kinds[arm] = s1.kind_agreement(reqs, resp[arm], arm)
    verdict, per_arm = s1.kill_verdict(scores)
    out = {'schema': 'sidewalk-auto-labeler.corner_prerater.stage1_score/1', 'issue': ISSUE,
           'kill_rule': {'arms': list(s1.KILL_ARMS), 'fail_if_units_caught_le': s1.KILL_MAX_UNITS,
                         'fail_if_flag_rate_gt': s1.KILL_MAX_FLAG,
                         'kill_if': 'every arm fails'},
           'kill_verdict': verdict, 'per_arm': per_arm, 'scores': scores,
           'per_gallery': per_gallery, 'absent_kind_agreement': kinds,
           'replicate_change': (s1.replicate_change(reqs, resp['argon_multi_r1'],
                                                    resp['argon_multi_r2'])
                                if resp['argon_multi_r1'] and resp['argon_multi_r2'] else None),
           'requests_sha256': s1.sha256_file(st / 'requests.jsonl'),
           'responses_sha256': {a: s1.sha256_file(st / 'responses' / f'{a}.jsonl')
                                for a in s1.ARMS if resp[a]}}
    (st / 'score.json').write_bytes((json.dumps(out, indent=1) + '\n').encode('utf-8'))
    for arm, s in scores.items():
        p = s['present_blind']
        fr = s['flag_rate']
        print(f"{arm:16s} called {s['called']:3d}/{s['corners']}  units {p['units_caught']}/"
              f"{p['units']}  corners {p['corners_caught']}/{p['corners']}  flag "
              f"{'-' if fr is None else f'{fr:.3f}'} ({s['flagged']}/{s['absent_called']})  "
              f"{per_arm.get(arm, '')}")
    print(f'kill: {verdict}')
    return 0


# ------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gate', help='Stage 0: the stop gate for the stand-in question')
    g.add_argument('--out', default='runs/vancouver/corner_prerater185')
    g.add_argument('--split-corners', default=D_SPLIT)
    g.add_argument('--old-corners', default=D_OLD)
    g.add_argument('--run-dir', default=D_RUN)
    g.add_argument('--extra-run', action='append', default=None)
    g.add_argument('--gallery243', default=D_G243)
    g.add_argument('--gallery169', default=D_G169)
    g.add_argument('--labels', default=D_LABELS)
    g.add_argument('--sidewalks', default=D_SIDEWALKS, help="path, or 'none' to skip")
    g.add_argument('--check', action='store_true',
                   help='recompute and compare with the committed gate.json')
    f = sub.add_parser('figures', help='redraw the Stage 0 figure from gate.json')
    f.add_argument('--out', default='runs/vancouver/corner_prerater185')
    f.add_argument('--fig-dir', default=D_FIG)
    e = sub.add_parser('export', help='Stage 1: write the requests for the 381 judged corners')
    e.add_argument('--out', default='runs/vancouver/corner_prerater185')
    e.add_argument('--no-render', action='store_true', help='skip re-rendering the aerials')
    s_ = sub.add_parser('score', help='Stage 1: score the answers and apply the kill rule')
    s_.add_argument('--out', default='runs/vancouver/corner_prerater185')
    args = ap.parse_args(argv)
    if args.cmd == 'export':
        return cmd_export(args)
    if args.cmd == 'score':
        return cmd_score(args)
    if args.cmd == 'gate':
        if args.extra_run is None:
            args.extra_run = list(D_EXTRA)
        return cmd_gate(args)
    return cmd_figures(args)


if __name__ == '__main__':
    sys.exit(main())
