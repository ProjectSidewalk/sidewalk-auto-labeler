"""Score label clustering against city curb-ramp inventories (issue #106, Part 2).

RampNet GT judges de-clustered panos, so it never says which labels from different panos are
one ramp. Bend and Gainesville publish one point per curb ramp (fetched for #79 by
scripts/inventory_oracle.py), which answers exactly that. This scores the server's PS rule
and fusion against them under the rule pre-registered on #106 before any score existed
(docs/ps-clustering-eval.md, "City-inventory scoring"; the comment is linked there):

  - labels synthesized from the run as eval_ps_clustering.py --offline does (one per stored
    detection >= tier, rig-masked, at the server's own positions via ps_placement);
  - a VISIBLE POOL of inventory ramps within POOL_RADIUS_M of a processed pano;
  - every cluster of every arm placed at the mean of its members' labeler raycast positions
    in the frame under test (`auto` primary, 2.6 m secondary);
  - covered / split / merge at r = 3, 5, 8 m, read at r = 5 m, tier 0.30, auto frame.

Arms: `ps @ t` (per region where the city has a server, else citywide), `ps_citywide @ 7.5 m`,
`fusion`, `fusion_server+attach` (cluster count only), and `ps @ 7.5 m (server centroid)`,
the sensitivity row that places the PS clusters at the mean of their members' server
positions instead.

Network: one GET of the Gainesville server's /v3/api/streets (cached under
runs/gainesville/inventory_clustering/streets.geojson), for regions. Nothing is written to
any server; Gainesville is read-only. Needs pandas/scipy/haversine/shapely, as
eval_ps_clustering.py does (analysis-only, not in requirements.txt).

Usage:
    python scripts/inventory_clustering.py score bend gainesville
    python scripts/inventory_clustering.py verdict
"""
import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import fuse_sites as fs  # noqa: E402
import eval_ps_clustering as epc  # noqa: E402  (exits with an install hint if deps missing)
import inventory_oracle as ioracle  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OUT_NAME = 'inventory_clustering'
POOLED = REPO_ROOT / 'runs' / '_pooled' / OUT_NAME
SERVERS = {'gainesville': 'https://sidewalk-gainesville.cs.washington.edu'}
CITIES = ('bend', 'gainesville')
DECIDING_CITY = 'gainesville'   # not a RampNet training city
GUARD_CITY = 'bend'             # a training city, with the larger inventory

POOL_RADIUS_M = 20.0
RADII_M = (3.0, 5.0, 8.0)
PRIMARY_RADIUS_M = 5.0
TIERS = (0.30, 0.55)
PRIMARY_TIER = 0.30
FRAMES = ('auto', 2.6)
PRIMARY_FRAME = 'auto'
THRESHOLDS_M = (7.5, 10.0, 12.5, 15.0)
PS_ARM = 'ps @ 7.5 m'
FUSION_ARM = 'fusion'
# the pre-registered decision rule (absolute shares)
RULE_SPLIT_DROP = 0.05       # fusion's split lower by at least this
RULE_MERGE_RISE = 0.01       # fusion's merge not higher by more than this
RULE_COVERED_DROP = 0.01     # fusion's covered not lower by more than this

PASS = 'PASS'
NOT_ESTABLISHED = 'NOT ESTABLISHED'
FRAME_DEPENDENT = 'NOT ESTABLISHED (frame-dependent)'


# ------------------------------------------------------------------------- metrics

def visible_pool(inv_xy, pano_xy, radius_m=POOL_RADIUS_M):
    """Boolean mask over inventory ramps: within radius_m of at least one pano position.

    Example:
        >>> visible_pool(np.array([[0.0, 5.0], [0.0, 40.0]]), np.array([[0.0, 0.0]])).tolist()
        [True, False]
    """
    inv_xy = np.asarray(inv_xy, dtype=float).reshape(-1, 2)
    pano_xy = np.asarray(pano_xy, dtype=float).reshape(-1, 2)
    if not len(inv_xy) or not len(pano_xy):
        return np.zeros(len(inv_xy), dtype=bool)
    d, _ = cKDTree(pano_xy).query(inv_xy, k=1, distance_upper_bound=radius_m)
    return np.isfinite(d)


def nearest_within(tree, xy, r):
    """Index of each point's nearest inventory ramp within r, -1 when there is none."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, dtype=int)
    d, idx = tree.query(xy, k=1, distance_upper_bound=r)
    return np.where(np.isfinite(d), idx, -1)


def inventory_metrics(clusters, inv_xy, pool, r):
    """The pre-registered metrics for one partition at match radius r.

    clusters: [(centroid or None, [member (e, n), ...])] -- the centroid is the mean of the
    placeable members' positions (None: no placeable member, counted but not placed).
    inv_xy: (N, 2) inventory positions in the same frame; pool: (N,) bool visible mask.

    covered = pool ramps with >= 1 assigned cluster / pool; split = covered pool ramps with
    >= 2 assigned clusters / covered; extra = (clusters on covered pool ramps - covered) /
    covered; merge = clusters with >= 2 assigned members whose members' ramps include >= 2
    distinct ramps each holding >= 2 of them, over clusters with >= 2 assigned members.
    Assignment is to the nearest ramp of the WHOLE inventory within r (pool or not).
    """
    inv_xy = np.asarray(inv_xy, dtype=float).reshape(-1, 2)
    pool = np.asarray(pool, dtype=bool)
    tree = cKDTree(inv_xy)
    placed = [c for c, _m in clusters if c is not None]
    assign = nearest_within(tree, placed, r)
    counts = np.bincount(assign[assign >= 0], minlength=len(inv_xy))
    n_pool = int(pool.sum())
    covered_mask = pool & (counts >= 1)
    covered = int(covered_mask.sum())
    split = int((pool & (counts >= 2)).sum())
    on_covered = int(counts[covered_mask].sum())
    merge_n = merge_k = 0
    for _c, members in clusters:
        if len(members) < 2:
            continue
        m = nearest_within(tree, members, r)
        m = m[m >= 0]
        if len(m) < 2:
            continue
        merge_n += 1
        _ids, per = np.unique(m, return_counts=True)
        if int((per >= 2).sum()) >= 2:
            merge_k += 1
    return {'n_clusters': len(clusters), 'n_placed': len(placed), 'pool': n_pool,
            'covered': covered, 'split': split, 'on_covered': on_covered,
            'merge_k': merge_k, 'merge_n': merge_n,
            'covered_rate': covered / n_pool if n_pool else None,
            'split_rate': split / covered if covered else None,
            'extra_per_covered': (on_covered - covered) / covered if covered else None,
            'merge_rate': merge_k / merge_n if merge_n else None,
            'clusters_per_covered': len(clusters) / covered if covered else None}


def verdict(rows):
    """Apply the pre-registered decision rule to arms.csv rows (dicts with city, tier,
    frame, arm, r, covered_rate, split_rate, merge_rate). Returns (verdict, reasons).

    PASS iff, at r = 5 m and tier 0.30, in BOTH cities in the auto frame: fusion's split
    is lower than ps @ 7.5 m's by >= 5 points, its merge is not higher by > 1 point and its
    covered is not lower by > 1 point -- and in the 2.6 m frame no city reverses any of the
    three (split not lower at all, merge higher by > 1 point, covered lower by > 1 point).
    A reversal at 2.6 m makes it NOT ESTABLISHED (frame-dependent); anything else is NOT
    ESTABLISHED.
    """
    def get(city, frame, arm):
        for r in rows:
            if (r['city'] == city and str(r['frame']) == str(frame) and r['arm'] == arm
                    and float(r['tier']) == PRIMARY_TIER
                    and float(r['r']) == PRIMARY_RADIUS_M):
                return r
        return None

    def f(v):
        return None if v in (None, '') else float(v)

    reasons, primary_ok, reversed_ = [], True, False
    for city in (DECIDING_CITY, GUARD_CITY):
        for frame in FRAMES:
            ps, fu = get(city, frame, PS_ARM), get(city, frame, FUSION_ARM)
            if ps is None or fu is None:
                reasons.append(f'{city} {frame}: missing rows')
                primary_ok = False
                continue
            d_split = f(fu['split_rate']) - f(ps['split_rate'])
            d_merge = f(fu['merge_rate']) - f(ps['merge_rate'])
            d_cov = f(fu['covered_rate']) - f(ps['covered_rate'])
            text = (f'{city} {frame}: split {f(ps["split_rate"]):.3f} -> '
                    f'{f(fu["split_rate"]):.3f} ({d_split:+.3f}), merge '
                    f'{f(ps["merge_rate"]):.3f} -> {f(fu["merge_rate"]):.3f} ({d_merge:+.3f}), '
                    f'covered {f(ps["covered_rate"]):.3f} -> {f(fu["covered_rate"]):.3f} '
                    f'({d_cov:+.3f})')
            if str(frame) == str(PRIMARY_FRAME):
                ok = (d_split <= -RULE_SPLIT_DROP and d_merge <= RULE_MERGE_RISE
                      and d_cov >= -RULE_COVERED_DROP)
                primary_ok &= ok
                reasons.append(text + (' -- meets the rule' if ok else ' -- does NOT meet it'))
            else:
                rev = d_split >= 0 or d_merge > RULE_MERGE_RISE or d_cov < -RULE_COVERED_DROP
                reversed_ |= rev
                reasons.append(text + (' -- REVERSES' if rev else ' -- no reversal'))
    if primary_ok and not reversed_:
        return PASS, reasons
    if primary_ok and reversed_:
        return FRAME_DEPENDENT, reasons
    return NOT_ESTABLISHED, reasons


# ---------------------------------------------------------------------------- scoring

def streets_for(city, out):
    if city not in SERVERS:
        return None
    path = out / 'streets.geojson'
    epc.fetch(SERVERS[city] + epc.API_STREETS, path)
    return path


def placed_clusters(clusters, det_pos):
    """[(centroid or None, member positions)] from eval_ps_clustering Clusters."""
    out = []
    for c in clusters:
        pts = [det_pos[m] for m in c.members if m in det_pos]
        cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) \
            if pts else None
        out.append((cen, pts))
    return out


def score_city(city, tiers=TIERS, frames=FRAMES, radii=RADII_M):
    run_dir = REPO_ROOT / 'runs' / city
    out = run_dir / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / 'results.jsonl'
    inventory, record = ioracle.load_inventory(city)
    streets_path = streets_for(city, out)
    streets = epc.load_streets(streets_path) if streets_path else []
    lines = [f'# {city}: label clustering vs the city curb-ramp inventory (#106 Part 2)', '',
             f"- inventory: {len(inventory)} kept ramps ({record.get('keep', {}).get('where_equivalent', '')}) "
             f"of {record.get('layer_total')}, `{record.get('file')}` sha256 "
             f"`{record.get('sha256')}`, fetched {record.get('fetched_utc')} from "
             f"{record.get('url')}",
             f'- results `{results_path.name}` sha256 `{fs.file_sha256(results_path)}`',
             ('- regions: nearest street of the server\'s street network (the server\'s '
              'insert rule); ' + epc.streets_provenance(streets_path, streets)[2:])
             if streets else '- regions: none (no server), so per-region == citywide']
    rows = []
    for tier in tiers:
        labels, det_of = epc.synthesize_labels(results_path, tier, mask_rig=True)
        if streets:
            labels['region_id'] = epc.assign_regions(labels.lat, labels.lng, streets)[0]
        t_kms = [t / 1000.0 for t in THRESHOLDS_M]
        parts = epc.ps_partition(labels, t_kms, per_region=True)
        city_part = epc.ps_partition(labels, [epc.PS_THRESHOLD_KM],
                                     per_region=False)[epc.PS_THRESHOLD_KM]
        ps_clusters = {f'ps @ {t_m:g} m': epc.clusters_from_assignment(labels, parts[t_km], det_of)
                       for t_m, t_km in zip(THRESHOLDS_M, t_kms)}
        ps_clusters['ps_citywide @ 7.5 m'] = epc.clusters_from_assignment(labels, city_part,
                                                                          det_of)
        server_ll = {det_of[r.label_id]: (r.lat, r.lng) for r in labels.itertuples(index=False)}
        for frame in frames:
            try:
                panos, _skip, height, auto = fs.load_at_height(results_path, frame)
            except ValueError as e:
                raise SystemExit(str(e))
            params = fs.FuseParams(camera_height_m=height, min_confidence=tier,
                                   mask_rig=True, apply_pose=fs.POSE_OFF)
            dets, fr, _drops = fs.project(panos, params)
            det_pos = {(d.pano_id, d.det_index): (d.e, d.n) for d in dets}
            inv_xy = np.array([fr.to_enu(lat, lng) for _k, lat, lng, _p in inventory])
            pano_xy = np.array([fr.to_enu(p.lat, p.lng) for p in panos])
            pool = visible_pool(inv_xy, pano_xy)
            arms = {name: placed_clusters(cl, det_pos) for name, cl in ps_clusters.items()}
            # sensitivity: the PS clusters at their members' SERVER positions
            srv = []
            for c in ps_clusters[PS_ARM]:
                pts = [fr.to_enu(*server_ll[m]) for m in c.members if m in server_ll]
                ray = [det_pos[m] for m in c.members if m in det_pos]
                srv.append(((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
                            if pts else None, ray))
            arms[f'{PS_ARM} (server centroid)'] = srv
            sites, _f2, _s2 = fs.fuse(panos, params)
            fusion_clusters = epc.clusters_from_sites(sites, False)
            arms[FUSION_ARM] = placed_clusters(fusion_clusters, det_pos)
            run_by_id = {p.pano_id: p for p in panos}
            srv_panos, srv_st = epc.server_panos(
                labels, det_of, run_by_id, epc.OFFLINE_USER,
                decode=fs.single_decode(Counter(p.decode for p in panos), results_path.name),
                border=fs.single_border(Counter(p.border for p in panos), results_path.name))
            srv_sites, srv_frame, _s3 = fs.fuse(srv_panos, replace(params, min_confidence=0.0,
                                                                   floor=0.0))
            att, attached = epc.attach_unplaceable(
                srv_sites, srv_panos, srv_frame, mask_rig=params.mask_rig,
                unpositioned=srv_st['unplaceable_label_ids'], label_of=srv_st['label_of'])
            arms['fusion_server+attach'] = placed_clusters(epc.place(att, det_pos), det_pos)
            n_labels = {name: sum(len(c.members) for c in cl) for name, cl in ps_clusters.items()}
            n_labels[f'{PS_ARM} (server centroid)'] = n_labels[PS_ARM]
            n_labels[FUSION_ARM] = sum(len(c.members) for c in fusion_clusters)
            n_labels['fusion_server+attach'] = sum(c.n_labels for c in att)
            for name, cl in arms.items():
                for r in radii:
                    m = inventory_metrics(cl, inv_xy, pool, r)
                    rows.append({'city': city, 'tier': tier, 'frame': str(frame), 'arm': name,
                                 'r': r, 'n_labels': n_labels[name], **m})
            lines += ['', f'## tier {tier:g}, {fs.frame_label(frame)} frame', '',
                      f'- visible pool: {int(pool.sum())} of {len(inventory)} inventory ramps '
                      f'within {POOL_RADIUS_M:g} m of one of {len(panos)} panos '
                      f'({len(inventory) - int(pool.sum())} excluded)',
                      f'- {len(labels)} labels synthesized; raycast placed {len(dets)} '
                      'detections at or above the storage floor']
            lines += fs.height_resolution_lines(frame, panos, params, auto)
            lines.append(f'- fusion_server+attach: {len(attached)} labels attached by bearing')
            lines += ['', '| arm | clusters | labels | placed | covered r3 / r5 / r8 | '
                      'split r5 | extra/covered r5 | merge r5 (k/n) | clusters/covered r5 |',
                      '|---|---:|---:|---:|---|---|---:|---|---:|']
            for name in arms:
                by_r = {row['r']: row for row in rows if row['tier'] == tier
                        and row['frame'] == str(frame) and row['arm'] == name}
                p = by_r[PRIMARY_RADIUS_M]
                lines.append(
                    f"| {name} | {p['n_clusters']} | {p['n_labels']} | {p['n_placed']} | "
                    + ' / '.join(f"{by_r[r]['covered_rate']:.3f}" for r in radii) + ' | '
                    f"{p['split_rate']:.3f} ({p['split']}/{p['covered']}) | "
                    f"{p['extra_per_covered']:.3f} | {p['merge_rate']:.3f} "
                    f"({p['merge_k']}/{p['merge_n']}) | {p['clusters_per_covered']:.2f} |")
    lines += ['', 'covered = pool ramps with a cluster assigned (nearest ramp within r) / '
              'pool; split = covered ramps with >= 2 clusters / covered; merge = clusters '
              'whose members fall on >= 2 distinct ramps with >= 2 members each / clusters '
              'with >= 2 assigned members; clusters/covered = all clusters / covered pool '
              'ramps. Every cluster is placed at the mean of its members\' raycast positions '
              'in the frame named, except the `(server centroid)` row. Rule and read: '
              'docs/ps-clustering-eval.md, "City-inventory scoring".']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    with open(out / 'arms.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(lines))
    print(f'wrote {out / "report.md"} and arms.csv')
    return rows


def read_rows(city):
    path = REPO_ROOT / 'runs' / city / OUT_NAME / 'arms.csv'
    if not path.exists():
        raise SystemExit(f'{path} is missing: run `inventory_clustering.py score {city}` first')
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def cmd_verdict():
    rows = read_rows(DECIDING_CITY) + read_rows(GUARD_CITY)
    v, reasons = verdict(rows)
    POOLED.mkdir(parents=True, exist_ok=True)
    lines = ['# City-inventory clustering score: verdict (#106 Part 2)', '',
             'Pre-registered rule (docs/ps-clustering-eval.md, "City-inventory scoring"): '
             '`fusion` vs `ps @ 7.5 m` at r = 5 m, tier 0.30, common `auto` frame, in both '
             'cities; the 2.6 m frame must not reverse it.', '', f'**{v}**', '']
    lines += [f'- {r}' for r in reasons]
    lines += ['', '| city | tier | frame | arm | covered r5 | split r5 | merge r5 | '
              'clusters |', '|---|---|---|---|---|---|---|---:|']
    for r in rows:
        if float(r['r']) == PRIMARY_RADIUS_M:
            lines.append(f"| {r['city']} | {float(r['tier']):g} | {r['frame']} | {r['arm']} | "
                         f"{float(r['covered_rate']):.3f} | {float(r['split_rate']):.3f} | "
                         f"{float(r['merge_rate']):.3f} | {r['n_clusters']} |")
    (POOLED / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8',
                                      newline='\n')
    (POOLED / 'verdict.json').write_text(json.dumps(
        {'verdict': v, 'reasons': reasons, 'rule': {
            'radius_m': PRIMARY_RADIUS_M, 'tier': PRIMARY_TIER, 'frame': PRIMARY_FRAME,
            'split_drop': RULE_SPLIT_DROP, 'merge_rise': RULE_MERGE_RISE,
            'covered_drop': RULE_COVERED_DROP}}, indent=1) + '\n', encoding='utf-8',
        newline='\n')
    print('\n'.join(lines))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('cities', nargs='*', default=list(CITIES))
    s.add_argument('--tier', type=float, nargs='+', default=list(TIERS))
    s.add_argument('--radius', type=float, nargs='+', default=list(RADII_M))
    s.add_argument('--frame', type=fs.fuse_camera_height_arg, nargs='+', default=list(FRAMES))
    sub.add_parser('verdict')
    args = ap.parse_args(argv)
    if args.cmd == 'score':
        if PRIMARY_RADIUS_M not in args.radius:
            raise SystemExit(f'--radius must include the primary {PRIMARY_RADIUS_M:g} m')
        for city in args.cities:
            if city not in CITIES:
                raise SystemExit(f'{city}: no inventory; choose from {CITIES}')
            score_city(city, args.tier, args.frame, args.radius)
    else:
        cmd_verdict()


if __name__ == '__main__':
    main()
