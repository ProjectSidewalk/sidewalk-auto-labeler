"""Position-selected panos at the corner inventory's unobservable units (RampNet#241).

The #238 corner inventory (scripts/corner_inventory.py, docs/corner-inventory.md) read
Vancouver from a run whose panos were chosen by where labels already existed, so 1,453
intersection units had no pano within 25 m and the absence read was a floor. This tool
chooses panos at those units by POSITION, hands them to the shipped detector, and compares
the re-scored inventory with the original. Subcommands:

  fetch-panos  one GET of the city server's /adminapi/panos (every PS pano, with its
               lat/lng; the list sidewalk-panorama-tools downloads from), cached under
               <out>/cache/ (untracked); url, fetch time and sha256 in
               <out>/select/ps_panos_fetch.json
  select       every PS-store pano within --radius-m (default: the build's obs_m, 25 m)
               of an unobservable unit's centre that is not already in the run and has a
               JPEG in the store -> <out>/select/store_ids.txt (input of
               detect_from_store.py --ids), units.csv, selection.json. With --store-skips
               (the new run's store_skipped.jsonl, after its metadata pass), also the units
               left with no usable store pano -> gsv_area.geojson (25 m circles), the area
               main.py scans for current GSV coverage there.
  compare      the original build against the build with the new run(s) merged in
               (`corner_inventory.py build --extra-run`): old -> new unit states, the
               decision rule read under three arms, the same reads on the units this pass
               moved, a nearest-pano-only sensitivity read, and the capture dates of the
               added panos -> <out>/compare/ (deterministic: no timestamps)

No pandas; stdlib + geo + corner_inventory.

    python scripts/corner_posdetect.py fetch-panos \\
        --server https://sidewalk-vancouver.cs.washington.edu --out runs/vancouver/corner_posdetect241
    python scripts/corner_posdetect.py select --build runs/vancouver/corner_inventory \\
        --store-ids runs/vancouver/corner_posdetect241/cache/store_jpg_ids.txt \\
        --out runs/vancouver/corner_posdetect241
    python scripts/corner_posdetect.py compare --old runs/vancouver/corner_inventory \\
        --new runs/vancouver/corner_inventory_posdetect241 \\
        --out runs/vancouver/corner_posdetect241
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import corner_inventory as ci  # noqa: E402

SCHEMA = 'sidewalk-auto-labeler.corner_posdetect/1'
PANOS_API = '/adminapi/panos'
PANOS_CACHE = 'cache/adminapi_panos.json'
CIRCLE_VERTICES = 32
KEY = 'fusion/primary'


# ------------------------------------------------------------------------- helpers

def unobservable_units(records, key=KEY):
    """The intersection units whose unit-level state is unobservable, in record order."""
    return [r for r in records if r['type'] in ci.INTERSECTION_STRATA
            and r['state'][key] == 'unobservable']


def build_frame(build):
    return geo.LocalFrame(build['frame']['lat0'], build['frame']['lng0'])


def run_pano_ids(results_path):
    return {p[2] for p in ci.load_panos(results_path)}


def near_items(fr, items, centres, radius_m):
    """{centre index: [(distance_m, item)] ascending} for items within radius_m.

    items: [(lat, lng, payload)], centres: [(lat, lng)]."""
    grid = geo.GridIndex(radius_m)
    xy = []
    for k, (lat, lng, _) in enumerate(items):
        e, n = fr.to_enu(lat, lng)
        xy.append((e, n))
        grid.add(e, n, k)
    out = {}
    for i, (lat, lng) in enumerate(centres):
        e, n = fr.to_enu(lat, lng)
        hits = []
        for k in grid.near(e, n):
            d = math.hypot(xy[k][0] - e, xy[k][1] - n)
            if d <= radius_m:
                hits.append((round(d, 2), items[k][2]))
        out[i] = sorted(hits, key=lambda x: (x[0], str(x[1])))
    return out


def circle(fr, lat, lng, radius_m, n=CIRCLE_VERTICES):
    """A closed GeoJSON ring approximating a radius_m circle about (lat, lng)."""
    e0, n0 = fr.to_enu(lat, lng)
    ring = []
    for k in range(n):
        a = 2 * math.pi * k / n
        la, lo = fr.to_latlng(e0 + radius_m * math.sin(a), n0 + radius_m * math.cos(a))
        ring.append([round(lo, 7), round(la, 7)])
    ring.append(ring[0])
    return ring


def read_skips(paths):
    """{pano_id: reason} from detect_from_store's store_skipped.jsonl file(s)."""
    out = {}
    for p in paths or ():
        for line in Path(p).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                out[row['pano_id']] = row['reason']
    return out


# ----------------------------------------------------------------------- fetch-panos

def cmd_fetch_panos(args):
    import requests
    out = Path(args.out)
    url = args.server.rstrip('/') + PANOS_API
    t0 = datetime.now(timezone.utc)
    r = requests.get(url, timeout=(30, 900))
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, list) or not body or 'pano_id' not in body[0]:
        raise SystemExit(f'{url}: not a pano list')
    path = out / PANOS_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    rec = {'url': url, 'fetched_at': t0.isoformat(timespec='seconds'),
           'sha256': ci.sha256_file(path), 'n_panos': len(body), 'path': PANOS_CACHE}
    ci.write_text_lf(out / 'select' / 'ps_panos_fetch.json', json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec))
    return rec


# ---------------------------------------------------------------------------- select

def select(records, build, ps_panos, store_ids, run_ids, radius_m, skips=None):
    """(unit rows, selected store ids) for the unobservable units of a build.

    A store pano is selected when it lies within radius_m of an unobservable unit's centre
    (by its /adminapi/panos position), is not in the run, and has a JPEG in the store.
    Every such pano is selected, not only the nearest: the observability rule counts any
    pano within 25 m. With `skips` ({pano_id: reason} from the new run's metadata pass),
    each unit row also says whether a usable store pano is left; units with none go to
    current GSV coverage (`source` 'gsv')."""
    fr = build_frame(build)
    units = unobservable_units(records)
    items = [(p['lat'], p['lng'], p['pano_id']) for p in ps_panos
             if p.get('lat') is not None and p.get('lng') is not None]
    near = near_items(fr, items, [(u['lat'], u['lng']) for u in units], radius_m)
    store_ids, run_ids = set(store_ids), set(run_ids)
    rows, chosen = [], set()
    for i, u in enumerate(units):
        hits = near[i]
        in_run = [p for _, p in hits if p in run_ids]
        no_jpg = [p for _, p in hits if p not in run_ids and p not in store_ids]
        sel = [(d, p) for d, p in hits if p not in run_ids and p in store_ids]
        chosen.update(p for _, p in sel)
        row = {'unit': u['unit'], 'type': u['type'], 'lat': ci.r6(u['lat']),
               'lng': ci.r6(u['lng']), 'n_ps_panos': len(hits), 'n_in_run': len(in_run),
               'n_no_jpg': len(no_jpg), 'n_selected': len(sel),
               'nearest_selected_m': sel[0][0] if sel else None,
               'nearest_selected_id': sel[0][1] if sel else None}
        if skips is not None:
            usable = [(d, p) for d, p in sel if p not in skips]
            row['n_usable'] = len(usable)
            row['source'] = 'store' if usable else 'gsv'
        rows.append(row)
    return rows, sorted(chosen)


def cmd_select(args):
    out = Path(args.out)
    bdir = Path(args.build)
    build = json.loads((bdir / 'build.json').read_text(encoding='utf-8'))
    records = ci.read_jsonl(bdir / 'corners_full.jsonl')
    fetch = json.loads((out / 'select' / 'ps_panos_fetch.json').read_text(encoding='utf-8'))
    ps_path = out / fetch['path']
    if ci.sha256_file(ps_path) != fetch['sha256']:
        raise SystemExit(f'{ps_path}: sha256 differs from ps_panos_fetch.json')
    ps_panos = json.loads(ps_path.read_text(encoding='utf-8'))
    store_ids = Path(args.store_ids).read_text(encoding='utf-8').split()
    results = build['inputs']['results']['path']
    run_ids = run_pano_ids(results)
    radius = args.radius_m if args.radius_m is not None else build['params']['obs_m']
    skips = read_skips(args.store_skips) if args.store_skips else None
    rows, chosen = select(records, build, ps_panos, store_ids, run_ids, radius, skips)
    sdir = out / 'select'
    fields = ['unit', 'type', 'lat', 'lng', 'n_ps_panos', 'n_in_run', 'n_no_jpg', 'n_selected',
              'nearest_selected_m', 'nearest_selected_id'] + \
        (['n_usable', 'source'] if skips is not None else [])
    ci.write_csv_lf(sdir / 'units.csv', fields, rows)
    ci.write_text_lf(sdir / 'store_ids.txt', ''.join(f'{p}\n' for p in chosen))
    by_type = {}
    for r in rows:
        d = by_type.setdefault(r['type'], {'units': 0, 'with_selected': 0, 'selected': 0})
        d['units'] += 1
        d['with_selected'] += r['n_selected'] > 0
        d['selected'] += r['n_selected']
    sel = {'schema': SCHEMA, 'radius_m': radius,
           'inputs': {'build': {'path': bdir.as_posix(),
                                'corners_full_sha256': ci.sha256_file(bdir / 'corners_full.jsonl')},
                      'results': {'sha256': build['inputs']['results']['sha256']},
                      'ps_panos': {'url': fetch['url'], 'fetched_at': fetch['fetched_at'],
                                   'sha256': fetch['sha256'], 'n': len(ps_panos)},
                      'store_ids': {'path': Path(args.store_ids).as_posix(),
                                    'sha256': ci.sha256_file(args.store_ids),
                                    'n': len(store_ids)}},
           'units': len(rows), 'by_type': dict(sorted(by_type.items())),
           'units_with_selected': sum(1 for r in rows if r['n_selected']),
           'units_no_ps_pano_within_radius': sum(1 for r in rows if r['n_ps_panos'] == 0),
           'units_only_no_jpg': sum(1 for r in rows if r['n_ps_panos'] and not r['n_selected']
                                    and r['n_no_jpg'] and not r['n_in_run']),
           'units_with_run_pano_by_ps_position': sum(1 for r in rows if r['n_in_run']),
           'selected_ids': len(chosen),
           'selected_ids_sha256': hashlib.sha256(''.join(f'{p}\n' for p in chosen)
                                                    .encode('utf-8')).hexdigest()}
    if skips is not None:
        sel['store_skips'] = {'paths': [Path(p).as_posix() for p in args.store_skips],
                              'sha256': [ci.sha256_file(p) for p in args.store_skips],
                              'n_selected_skipped': sum(1 for p in chosen if p in skips),
                              'reasons': dict(sorted(
                                  _count(skips[p] for p in chosen if p in skips).items()))}
        gsv_units = [r for r in rows if r['source'] == 'gsv']
        fr = build_frame(build)
        feats = [{'type': 'Feature', 'properties': {'unit': r['unit']},
                  'geometry': {'type': 'Polygon',
                               'coordinates': [circle(fr, r['lat'], r['lng'], radius)]}}
                 for r in gsv_units]
        sel['units_for_gsv'] = [r['unit'] for r in gsv_units]
        if feats:
            ci.write_text_lf(sdir / 'gsv_area.geojson',
                             json.dumps({'type': 'FeatureCollection', 'features': feats},
                                        separators=(',', ':')) + '\n')
    ci.write_text_lf(sdir / 'selection.json', json.dumps(sel, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in sel.items() if k != 'inputs'}, indent=1))
    return sel


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------- compare

def bucket(inv_counts, ignore=()):
    """'clean' (no inventory point, ignoring classes in `ignore`), 'false' (an Available
    point) or 'rmvna' (only other classes)."""
    if inv_counts.get('Available', 0) > 0:
        return 'false'
    if not any(v for k, v in inv_counts.items() if k not in ignore):
        return 'clean'
    return 'rmvna'


READS = (
    # name, arm key, classes ignored for the clean read
    ('a_fusion_clean_as_written', 'fusion/primary', ()),
    ('b_fusion_clean_ignoring_NA_noramp', 'fusion/primary', ('NA_noramp',)),
    ('c_deployed_clean', 'deployed/primary', ()),
)


def absence_reads(units, states=None):
    """Rows of the absence reads over `units` (unit records). `states` overrides the
    record states: {unit: {arm_key: state}}."""
    rows = []
    for name, key, ignore in READS:
        ab = [u for u in units
              if (states[u['unit']][key] if states else u['state'][key]) == 'absent']
        k = sum(1 for u in ab if bucket(u['inv_counts'], ignore) == 'clean')
        no_av = sum(1 for u in ab if u['inv_counts'].get('Available', 0) == 0)
        rows.append({'read': name, 'n_absent': len(ab), 'clean': k,
                     **_share('clean', k, len(ab)),
                     'no_available': no_av, **_share('no_available', no_av, len(ab))})
    return rows


def _share(prefix, k, n):
    if n == 0:
        return {f'{prefix}_p': None, f'{prefix}_lo': None, f'{prefix}_hi': None}
    lo, hi = ci.wilson(k, n)
    return {f'{prefix}_p': round(k / n, 4), f'{prefix}_lo': round(lo, 4),
            f'{prefix}_hi': round(hi, 4)}


def extra_sites(build):
    """{site_id: op_panos} of the operational sites of the build's extra runs (ids
    prefixed as corner_inventory.load_extra_run prefixes them)."""
    out = {}
    for k, v in build['inputs'].items():
        if k.startswith('extra:') and k.endswith(':sites'):
            name = k.split(':')[1]
            meta = json.loads(Path(build['inputs'][f'extra:{name}:sites_meta']['path'])
                              .read_text(encoding='utf-8'))
            thr = meta['params']['min_confidence']
            for s in ci.read_jsonl(v['path']):
                if s.get('n_operational', 0) >= 1:
                    out[f"{name}:{s['site_id']}"] = sorted(
                        {m['pano_id'] for m in s['members'] if m['confidence'] >= thr})
    return out


def extra_panos(build):
    """[pano block] of the build's extra runs, first-seen order."""
    out, seen = [], set()
    for k, v in build['inputs'].items():
        if k.startswith('extra:') and k.endswith(':results'):
            for d in ci.read_jsonl(v['path']):
                p = d['pano']
                if p['panorama_id'] not in seen:
                    seen.add(p['panorama_id'])
                    out.append(p)
    return out


def nearest_only_states(new_units, target, added_ids, sites_extra, fr, obs_m):
    """Fusion-arm unit states for the `target` units when, of the added panos, only the
    one nearest the unit centre is kept (a sensitivity read of 'all panos within 25 m').
    A base-run site in the window still counts; an added site counts only if the nearest
    added pano is among its operational panos."""
    out = {}
    for u in new_units:
        if u['unit'] not in target:
            continue
        added = [p for p in u['pano_ids_25'] if p in added_ids]
        base = [p for p in u['pano_ids_25'] if p not in added_ids]
        if not added:
            out[u['unit']] = {KEY: u['state'][KEY]}
            continue
        e0, n0 = fr.to_enu(u['lat'], u['lng'])
        near = min(added, key=lambda p: (math.hypot(*(a - b for a, b in zip(
            fr.to_enu(*added_ids[p]), (e0, n0)))), p))
        present = any((s not in sites_extra) or (near in sites_extra[s])
                      for s in u['site_ids'])
        observed = bool(base) or math.hypot(*(a - b for a, b in zip(
            fr.to_enu(*added_ids[near]), (e0, n0)))) <= obs_m
        out[u['unit']] = {KEY: ci.state(present, observed)}
    return out


def transitions(old, new, key):
    """{(stratum, old_state, new_state): n} over intersection units."""
    o = {r['unit']: r for r in old if r['type'] in ci.INTERSECTION_STRATA}
    out = {}
    for r in new:
        if r['type'] not in ci.INTERSECTION_STRATA:
            continue
        k = (r['type'], o[r['unit']]['state'][key], r['state'][key])
        out[k] = out.get(k, 0) + 1
    return out


def year_hist(dates):
    out = {}
    for d in dates:
        y = d[:4] if d else 'undated'
        out[y] = out.get(y, 0) + 1
    return dict(sorted(out.items()))


def compare(old, new, old_build, new_build, base_dates):
    """Everything `compare` writes, as one dict (no timestamps, no absolute paths)."""
    if [r['unit'] for r in old] != [r['unit'] for r in new]:
        raise SystemExit('old and new builds hold different units')
    fr = build_frame(new_build)
    obs_m = new_build['params']['obs_m']
    old_by = {r['unit']: r for r in old}
    ints_new = [r for r in new if r['type'] in ci.INTERSECTION_STRATA]
    target = {r['unit'] for r in unobservable_units(old)}
    added = extra_panos(new_build)
    added_ids = {p['panorama_id']: (p['lat'], p['lng']) for p in added}
    sites_x = extra_sites(new_build)
    res = {'schema': SCHEMA, 'target_units': len(target), 'added_panos': len(added),
           'added_operational_sites': len(sites_x),
           'extra_runs': new_build.get('extra_runs', [])}
    res['transitions'] = {a: [{'stratum': s, 'old': o, 'new': n_, 'units': c}
                              for (s, o, n_), c in sorted(transitions(old, new,
                                                                      f'{a}/primary').items())]
                          for a in ci.ARMS}
    res['decision_rule'] = ci.decision(ci.score_rows(new))
    res['decision_rule_old'] = ci.decision(ci.score_rows(old))
    groups = [('intersections', ci.INTERSECTION_STRATA)] + [(s, (s,))
                                                           for s in ci.INTERSECTION_STRATA]
    reads = []
    for subset in ('all', 'old_absent', 'moved'):
        for gname, strata in groups:
            for build_name, recs in (('old', old), ('new', new)):
                if subset == 'moved' and build_name == 'old':
                    continue
                units = [r for r in recs if r['type'] in strata]
                if subset == 'old_absent':
                    units = [r for r in units if old_by[r['unit']]['state'][KEY] == 'absent']
                if subset == 'moved':
                    units = [r for r in units if r['unit'] in target]
                for row in absence_reads(units):
                    reads.append({'subset': subset, 'stratum': gname, 'build': build_name,
                                  **row})
    # nearest-pano-only sensitivity, fusion arm, the target units
    near_states = nearest_only_states(ints_new, target, added_ids, sites_x, fr, obs_m)
    tgt_units = [r for r in ints_new if r['unit'] in target]
    for gname, strata in groups:
        units = [r for r in tgt_units if r['type'] in strata]
        st = {u['unit']: {**u['state'], **near_states[u['unit']]} for u in units}
        row = absence_reads(units, st)[0]
        reads.append({'subset': 'moved_nearest_only', 'stratum': gname, 'build': 'new',
                      **row})
    res['reads'] = reads
    near_counts = {}
    for u in tgt_units:
        s = near_states[u['unit']][KEY]
        near_counts[s] = near_counts.get(s, 0) + 1
    res['target_states_nearest_only'] = dict(sorted(near_counts.items()))
    res['target_states'] = {a: dict(sorted(_count(r['state'][f'{a}/primary']
                                                   for r in tgt_units).items()))
                            for a in ci.ARMS}
    res['capture_years_added'] = year_hist(p.get('capture_date') for p in added)
    res['capture_years_base'] = year_hist(base_dates)
    dated = sorted(p['capture_date'] for p in added if p.get('capture_date'))
    res['capture_date_added_quantiles'] = ci.quant(dated)
    # added panos that sit within 25 m of a target unit centre by their record position
    res['target_units_with_added_pano_within_obs_m'] = sum(
        1 for r in tgt_units if any(p in added_ids for p in r['pano_ids_25']))
    return res


def render(res):
    L = ['# Position-selected panos at the unobservable units (RampNet#241)', '',
         'Written by `scripts/corner_posdetect.py compare`. Protocol and caveats: '
         'docs/corner-inventory.md, amendment A8.', '',
         f"- target units (unobservable in the #238 build, fusion arm, unit level): "
         f"{res['target_units']}",
         f"- added panos: {res['added_panos']}; their operational fused sites: "
         f"{res['added_operational_sites']}",
         f"- target units with an added pano within 25 m (record position): "
         f"{res['target_units_with_added_pano_within_obs_m']}",
         f"- target units now (fusion): {res['target_states']['fusion']}; "
         f"(deployed, emulated for added panos): {res['target_states']['deployed']}",
         f"- target units now, nearest added pano only (fusion): "
         f"{res['target_states_nearest_only']}", '']
    d, d0 = res['decision_rule'], res['decision_rule_old']
    L += ['## Decision rule (as pre-registered: clean read, unit level, fusion, primary, '
          'intersections pooled)', '',
          f"- before: {d0['clean']}/{d0['n_absent']} = {d0['precision']} {d0['ci']}",
          f"- after: {d['clean']}/{d['n_absent']} = {d['precision']} {d['ci']} -> "
          f"**{d['outcome']}**", '']
    for a in ci.ARMS:
        L += [f'## Unit state transitions, {a} arm (intersections)', '',
              '| stratum | old | new | units |', '|---|---|---|---:|']
        for t in res['transitions'][a]:
            L.append(f"| {t['stratum']} | {t['old']} | {t['new']} | {t['units']} |")
        L.append('')
    L += ['## Absence reads', '',
          'a = clean read, fusion arm (the rule as written); b = the same, not counting the '
          "city's `NA` points with no `RAMPTYPE`; c = clean read, deployed arm (emulated for "
          'added panos). `no Available` = absent with no `Available` point.', '',
          '| subset | stratum | build | read | absent | clean | share [95% CI] | '
          'no Available | share [95% CI] |', '|---|---|---|---|---:|---:|---|---:|---|']

    def fmt(r, p):
        if r[f'{p}_p'] is None:
            return 'n/a'
        return f"{r[f'{p}_p']:.3f} [{r[f'{p}_lo']:.3f}, {r[f'{p}_hi']:.3f}]"
    for r in res['reads']:
        L.append(f"| {r['subset']} | {r['stratum']} | {r['build']} | {r['read']} | "
                 f"{r['n_absent']} | {r['clean']} | {fmt(r, 'clean')} | {r['no_available']} | "
                 f"{fmt(r, 'no_available')} |")
    L += ['', '## Capture years', '', '| year | added panos | #56 run panos |',
          '|---|---:|---:|']
    years = sorted(set(res['capture_years_added']) | set(res['capture_years_base']))
    for y in years:
        L.append(f"| {y} | {res['capture_years_added'].get(y, 0)} | "
                 f"{res['capture_years_base'].get(y, 0)} |")
    L += ['', f"Added capture dates, quantiles: {res['capture_date_added_quantiles']}", '']
    return '\n'.join(L)


def cmd_compare(args):
    old_dir, new_dir, out = Path(args.old), Path(args.new), Path(args.out) / 'compare'
    old = ci.read_jsonl(old_dir / 'corners_full.jsonl')
    new = ci.read_jsonl(new_dir / 'corners_full.jsonl')
    ob = json.loads((old_dir / 'build.json').read_text(encoding='utf-8'))
    nb = json.loads((new_dir / 'build.json').read_text(encoding='utf-8'))
    if not nb.get('extra_runs'):
        raise SystemExit(f'{new_dir}: build.json has no extra_runs')
    if ob['inputs']['results']['sha256'] != nb['inputs']['results']['sha256']:
        raise SystemExit('old and new builds use different base runs')
    base_dates = [p.get('capture_date') for p in
                  (json.loads(line)['pano'] for line in
                   open(nb['inputs']['results']['path'], encoding='utf-8') if line.strip())]
    res = compare(old, new, ob, nb, base_dates)
    res['inputs'] = {'old_corners_full_sha256': ci.sha256_file(old_dir / 'corners_full.jsonl'),
                     'new_corners_full_sha256': ci.sha256_file(new_dir / 'corners_full.jsonl'),
                     **{k: v['sha256'] for k, v in nb['inputs'].items()
                        if k.startswith('extra:')}}
    ci.write_text_lf(out / 'compare.json', json.dumps(res, indent=1, sort_keys=True) + '\n')
    rows = [{'arm': a, **t} for a in ci.ARMS for t in res['transitions'][a]]
    ci.write_csv_lf(out / 'transitions.csv', ['arm', 'stratum', 'old', 'new', 'units'], rows)
    fields = ['subset', 'stratum', 'build', 'read', 'n_absent', 'clean', 'clean_p', 'clean_lo',
              'clean_hi', 'no_available', 'no_available_p', 'no_available_lo',
              'no_available_hi']
    ci.write_csv_lf(out / 'reads.csv', fields, res['reads'])
    ci.write_text_lf(out / 'report.md', render(res))
    print(render(res))
    return res


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch-panos')
    f.add_argument('--server', required=True)
    f.add_argument('--out', required=True)
    s = sub.add_parser('select')
    s.add_argument('--build', required=True, help='corner_inventory.py build output dir')
    s.add_argument('--store-ids', required=True,
                   help='ids with a JPEG in the store, one per line '
                        '(detect_from_store.store_ids)')
    s.add_argument('--out', required=True)
    s.add_argument('--radius-m', type=float, default=None,
                   help="default: the build's obs_m (25 m)")
    s.add_argument('--store-skips', action='append', default=[],
                   help="the new run's store_skipped.jsonl, after its metadata pass; "
                        'writes gsv_area.geojson for the units left with no usable store pano')
    c = sub.add_parser('compare')
    c.add_argument('--old', required=True, help='the original build dir')
    c.add_argument('--new', required=True, help='the build dir with --extra-run')
    c.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    return {'fetch-panos': cmd_fetch_panos, 'select': cmd_select,
            'compare': cmd_compare}[args.cmd](args)


if __name__ == '__main__':
    main()
