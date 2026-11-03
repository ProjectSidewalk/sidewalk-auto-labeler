"""Draw inventory ramps that one Vancouver clustering arm splits and another does not (#56).

Rebuilds the three partitions of the server's own labels exactly as
inventory_clustering.score_server_arms does (auto frame): `deployed` (the clusters as
served), `ps @ 7.5 m` (the server rule re-run) and `fusion_server+attach`. Every cluster is
placed at the mean of its labels' SERVER positions -- the frame in which every cluster is
placed, so no arm is flattered by dropping unplaceable clusters -- and assigned to the
nearest inventory ramp within 5 m. The run prints each arm's split rate, which must equal
the `placement: server` rows of runs/vancouver/inventory_clustering/report.md.

Visible-pool ramps fall into four groups (fix: ps splits, fusion does not; break: the
reverse; stale: deployed splits, ps does not; both), and --n ramps per group are drawn at
random (--seed) as three panels on Esri World Imagery (network: server.arcgisonline.com
only; tiles cached). The partition state and tiles are cached under the untracked
runs/vancouver/split_figures/; figures and groups.csv go to docs/figures/vancouver-splits/.

    python scripts/split_figures.py            # 3 per group, seed 56
    python scripts/split_figures.py --rebuild  # recompute the partitions (~minutes)
"""
import argparse
import csv
import math
import pickle
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fuse_sites as fs  # noqa: E402
import eval_ps_clustering as epc  # noqa: E402
import inventory_oracle as ioracle  # noqa: E402
import inventory_clustering as ic  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

CITY = 'vancouver'
CACHE = REPO_ROOT / 'runs' / CITY / 'split_figures'
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'vancouver-splits'
RADIUS_M = ic.PRIMARY_RADIUS_M
ZOOM = 20
HALF_M = 22.0
ARMS = [('deployed', 'deployed (live server)'),
        ('ps@7.5', 'server rule re-run (ps @ 7.5 m)'),
        ('fusion+attach', 'fusion + bearing attach')]
ATTRIBUTION = 'Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community'
TILE_URL = ('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/'
            'tile/{z}/{y}/{x}')


def build_state():
    """Partitions, server-position centroids and 5 m ramp assignments for every arm."""
    cfg = ic.SERVER_LABELS[CITY]
    run = REPO_ROOT / 'runs' / CITY
    results = run / 'results.jsonl'
    labels, _ = epc.load_labels(run / cfg['labels'])
    labels = labels[labels.label_type == 'CurbRamp'].reset_index(drop=True)
    det_of, _, _ = epc.label_to_detection(results, labels)
    ai = labels[labels.user_id.astype(str) == cfg['ai_user']].reset_index(drop=True)
    inventory, _ = ioracle.load_inventory(CITY)

    arms = {'deployed': epc.clusters_from_server(
        epc.load_server_clusters(run / cfg['clusters']), det_of)}
    part = epc.ps_partition(ai, [epc.PS_THRESHOLD_KM], per_region=True)[epc.PS_THRESHOLD_KM]
    arms['ps@7.5'] = epc.clusters_from_assignment(ai, part, det_of)
    panos, _s, height, _auto = fs.load_at_height(results, 'auto')
    params = fs.FuseParams(camera_height_m=height, min_confidence=cfg['tier'],
                           mask_rig=False, apply_pose=fs.POSE_OFF)
    _dets, fr, _ = fs.project(panos, params)
    srv_panos, st = epc.server_panos(
        labels, det_of, {p.pano_id: p for p in panos}, cfg['ai_user'],
        decode=fs.single_decode(Counter(p.decode for p in panos), results.name),
        border=fs.single_border(Counter(p.border for p in panos), results.name),
        unmapped_confidence=cfg['tier'])
    sites, sfr, _ = fs.fuse(srv_panos, replace(params, min_confidence=0.0, floor=0.0))
    arms['fusion+attach'], _att = epc.attach_unplaceable(
        sites, srv_panos, sfr, mask_rig=params.mask_rig,
        unpositioned=st['unplaceable_label_ids'], label_of=st['label_of'])

    inv_xy = np.array([fr.to_enu(lat, lng) for _k, lat, lng, _p in inventory])
    pool = ic.visible_pool(inv_xy, np.array([fr.to_enu(p.lat, p.lng) for p in panos]))
    tree = cKDTree(inv_xy)
    lab_xy = {int(r.label_id): fr.to_enu(r.lat, r.lng) for r in labels.itertuples(index=False)}
    state = {'fr': fr, 'inv_xy': inv_xy, 'pool': pool, 'lab_xy': lab_xy, 'arms': {}}
    for name, cl in arms.items():
        out = []
        for c in cl:
            pts = [lab_xy[lab] for lab in c.label_ids if lab in lab_xy]
            out.append({'label_ids': [int(lab) for lab in c.label_ids],
                        'cen': tuple(np.mean(pts, axis=0)) if pts else None, 'ramp': -1})
        placed = [c for c in out if c['cen'] is not None]
        idx = ic.nearest_within(tree, [c['cen'] for c in placed], RADIUS_M)
        for c, k in zip(placed, idx):
            c['ramp'] = int(k)
        counts = np.bincount([c['ramp'] for c in out if c['ramp'] >= 0],
                             minlength=len(inv_xy))
        state['arms'][name] = {'clusters': out, 'counts': counts}
    return state


def load_state(rebuild):
    path = CACHE / 'state.pkl'
    if path.exists() and not rebuild:
        return pickle.loads(path.read_bytes())
    state = build_state()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(state))
    return state


def tile(x, y):
    import requests
    from PIL import Image
    path = CACHE / 'tiles' / f'{ZOOM}_{x}_{y}.jpg'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(TILE_URL.format(z=ZOOM, x=x, y=y), timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    return Image.open(path).convert('RGB')


def world_px(lat, lng):
    n = 2 ** ZOOM * 256
    return ((lng + 180) / 360 * n,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def basemap(fr, cx, cy):
    from PIL import Image
    x0, y0 = world_px(*fr.to_latlng(cx - HALF_M, cy + HALF_M))
    x1, y1 = world_px(*fr.to_latlng(cx + HALF_M, cy - HALF_M))
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    mosaic = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mosaic.paste(tile(tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    return mosaic.crop((int(x0 - tx0 * 256), int(y0 - ty0 * 256),
                        int(x1 - tx0 * 256), int(y1 - ty0 * 256)))


def draw(state, ramp, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fr, inv, lab_xy = state['fr'], state['inv_xy'], state['lab_xy']
    cx, cy = inv[ramp]
    img = basemap(fr, cx, cy)
    in_win = {lab for lab, (e, n) in lab_xy.items()
              if abs(e - cx) < HALF_M and abs(n - cy) < HALF_M}
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
    for ax, (arm, arm_title) in zip(axes, ARMS):
        a = state['arms'][arm]
        ax.imshow(img, extent=(cx - HALF_M, cx + HALF_M, cy - HALF_M, cy + HALF_M))
        shown = [c for c in a['clusters'] if c['cen'] is not None
                 and any(lab in in_win for lab in c['label_ids'])]
        for j, c in enumerate(shown):
            col = plt.get_cmap('tab10')(j % 10)
            pts = np.array([lab_xy[lab] for lab in c['label_ids'] if lab in lab_xy])
            for p in pts:
                ax.plot([p[0], c['cen'][0]], [p[1], c['cen'][1]], color=col, lw=0.8, alpha=0.8)
            ax.scatter(pts[:, 0], pts[:, 1], s=14, color=col, edgecolor='k', lw=0.3, zorder=3)
            focal = c['ramp'] == ramp
            ax.scatter([c['cen'][0]], [c['cen'][1]], s=260 if focal else 120, marker='*',
                       color=col, edgecolor='white' if focal else 'k',
                       lw=1.4 if focal else 0.5, zorder=5)
        vis = (np.abs(inv[:, 0] - cx) < HALF_M) & (np.abs(inv[:, 1] - cy) < HALF_M)
        ax.scatter(inv[vis, 0], inv[vis, 1], s=70, marker='s', facecolor='none',
                   edgecolor='white', lw=1.6, zorder=4)
        ax.add_patch(Circle((cx, cy), RADIUS_M, fill=False, ls='--', ec='yellow', lw=1.2,
                            zorder=4))
        n = int(a['counts'][ramp])
        ax.set_title(f'{arm_title}\n{n} cluster{"" if n == 1 else "s"} on this ramp'
                     + ('  (SPLIT)' if n >= 2 else ''), fontsize=10,
                     color='firebrick' if n >= 2 else 'darkgreen')
        ax.set_xlim(cx - HALF_M, cx + HALF_M)
        ax.set_ylim(cy - HALF_M, cy + HALF_M)
        ax.set_xticks([])
        ax.set_yticks([])
    lat, lng = fr.to_latlng(cx, cy)
    fig.suptitle(f'{title}  --  inventory ramp at {lat:.6f}, {lng:.6f}', fontsize=12, y=0.985)
    fig.text(0.5, 0.028, 'white squares = city inventory ramps   yellow dashed = 5 m match '
             'radius around the ramp scored   dots = AI labels at their server position, '
             'coloured by cluster   star = cluster centre (big star = matched to this ramp)',
             ha='center', fontsize=8.5)
    fig.text(0.5, 0.006, ATTRIBUTION, ha='center', fontsize=7.5, color='0.35')
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return lat, lng


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=3, help='ramps drawn per group')
    ap.add_argument('--seed', type=int, default=56)
    ap.add_argument('--rebuild', action='store_true', help='recompute the cached partitions')
    args = ap.parse_args(argv)
    state = load_state(args.rebuild)
    pool = state['pool']
    cnt = {a: state['arms'][a]['counts'] for a in state['arms']}
    for arm, _t in ARMS:
        split = int((pool & (cnt[arm] >= 2)).sum())
        covered = int((pool & (cnt[arm] >= 1)).sum())
        print(f'{arm}: split {split}/{covered} = {split / covered:.3f} (server placement)')
    groups = {
        'fix': ('Server rule splits, fusion does not',
                (cnt['ps@7.5'] >= 2) & (cnt['fusion+attach'] == 1)),
        'break': ('Fusion splits, server rule does not',
                  (cnt['fusion+attach'] >= 2) & (cnt['ps@7.5'] == 1)),
        'stale': ('Deployed splits, fresh server rule does not',
                  (cnt['deployed'] >= 2) & (cnt['ps@7.5'] == 1)),
        'both': ('Server rule and fusion both split',
                 (cnt['ps@7.5'] >= 2) & (cnt['fusion+attach'] >= 2)),
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for tag, (title, mask) in groups.items():
        ids = np.where(pool & mask)[0].tolist()
        print(f'{tag}: {len(ids)} ramps -- {title}')
        for i, ramp in enumerate(rng.sample(ids, min(args.n, len(ids)))):
            path = FIG_DIR / f'{tag}_{i}.png'
            lat, lng = draw(state, ramp, title, path)
            rows.append({'group': tag, 'group_ramps': len(ids), 'figure': path.name,
                         'inventory_index': ramp, 'lat': f'{lat:.6f}', 'lng': f'{lng:.6f}',
                         **{f'clusters_{a}': int(cnt[a][ramp]) for a, _t in ARMS}})
    with open(FIG_DIR / 'groups.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} figures + groups.csv to {FIG_DIR}')


if __name__ == '__main__':
    main()
