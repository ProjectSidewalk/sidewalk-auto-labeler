"""Figures and examples for the first Panoramax run (issue #57; docs/panoramax-bayonne.md).

Three subcommands, so that what needs the run files, what needs the network and what
needs neither stay apart:

    data      run files -> docs/figures/panoramax-bayonne/data/fig*.csv
              Needs runs/<city>/results.jsonl for bayonne and the five Mapillary
              comparators (plus runs/bayonne/osm_streets.json). No GPU, no network.
              Bootstraps resample SITES (or pose groups, or panos for rates) with seed 57.
    examples  network -> data/examples/ (downscaled JPEGs + a CSV of what is drawn).
              Fetches 8 Bayonne panos and one cropped GoPro MAX2 upload from Panoramax.
    figures   committed data/ only -> fig1..fig7 as PNG (200 dpi) + SVG. No run files,
              no network; byte-reproducible for a fixed matplotlib (no timestamps, fixed
              SVG hash salt).

Usage:
    python scripts/panoramax_bayonne_figures.py data --run-root D:/Git/sidewalk-auto-labeler/runs
    python scripts/panoramax_bayonne_figures.py examples
    python scripts/panoramax_bayonne_figures.py figures

Example selection rules (fixed before drawing): the contact sheet is the 4 panos with
the highest single detection confidence plus 4 panos drawn uniformly from the whole run
with random.Random(57); the plan-view site is the 0.55-tier site with >= 5 operational
views whose median leave-one-out residual is closest to the run's pooled median.
"""
import argparse
import csv
import io
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO_ROOT / 'docs' / 'figures' / 'panoramax-bayonne'
DATA = OUT / 'data'
EXAMPLES = DATA / 'examples'
CITY = 'bayonne'
COMPARATORS = ('richmond', 'clovis', 'laurens', 'annapolis', 'morgantown')
HEIGHT = 2.6
SEED = 57
N_BOOT = 1000
N_BOOT_GPS = 200
RANGE_BINS = ('0-8', '8-12', '12-18', '18-25')
ANCHOR = 0.269                       # Richmond's pooled chi2/dof at 0.55 (the amended rule)
BAND = (0.5 * ANCHOR, 2.0 * ANCHOR)  # [0.13, 0.54]
M_ANCHOR = 1.95
M_BAND = (1.30, 2.93)               # 1.5x either side of Richmond's 1.95 m, as posted
NADIR_MASK_Y = 0.5 + 49.0 / 180.0    # detectors.NADIR_MASK_DEG
LOGO_BAND_Y = 0.791                  # measured on two municipal panos (doc section 1)
MAX2_EXAMPLE = '030f0568-14d4-4ae9-b46c-0eda95a98f10'

# --- palette (dataviz reference palette, light mode; validated all-pairs for slots 1-3)
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SEQ = ('#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b')


# ============================================================================ data
def _pct(values, p):
    v = sorted(values)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, math.ceil(len(v) * p / 100) - 1))]


def _stats(rows):
    """chi2/dof (pooled), its median form, and the median metre residual of view rows."""
    if not rows:
        return None
    import reprojection_residual as rr
    chi = [r['chi2'] for r in rows]
    return {'chi2_dof': sum(chi) / (2.0 * len(chi)),
            'chi2_dof_median': _pct(chi, 50) / rr.CHI2_2_MEDIAN,
            'm_p50': _pct([r['dist_m'] for r in rows], 50),
            'along_p50': _pct([abs(r['along_m']) for r in rows], 50),
            'cross_p50': _pct([abs(r['cross_m']) for r in rows], 50)}


def _boot(rows, fn, n=N_BOOT, seed=SEED):
    """(point, {key: (lo, hi)}) of fn(rows), 95% percentile CI over sites resampled."""
    by_site = {}
    for r in rows:
        by_site.setdefault(r['site_id'], []).append(r)
    sites = list(by_site)
    point = fn(rows)
    if point is None:
        return None, {}
    rng = random.Random(seed)
    draws = {k: [] for k in point}
    for _ in range(n):
        sample = [r for s in rng.choices(sites, k=len(sites)) for r in by_site[s]]
        val = fn(sample)
        if val is None:
            continue
        for k in point:
            draws[k].append(val[k])
    return point, {k: (_pct(v, 2.5), _pct(v, 97.5)) for k, v in draws.items()}


def _load_city(run_root, city, tier):
    import fuse_sites as fs
    import reprojection_residual as rr
    path = run_root / city / 'results.jsonl'
    panos, _ = fs.load_results(path)
    by_id = {p.pano_id: p for p in panos}
    sites, frame, _, _ = rr.sites_for(Path('.'), panos, HEIGHT, True, tier)
    rigs = rr.load_rigs(path)
    rows = rr.gtfree_rows(city, '2.6m', sites, frame, by_id, HEIGHT, rigs)
    return panos, by_id, sites, frame, rows, rigs


def _rig_sites(sites, rigs, rig):
    """Sites re-solved from only the views of one rig (>= MIN_VIEWS of them): the
    rig-matched population a leave-one-out calibration can be run on."""
    import reprojection_residual as rr
    from dataclasses import replace
    out = []
    for sv in sites:
        views = [v for v in sv.views if rigs.get(v.pano_id) == rig]
        if len(views) >= rr.MIN_VIEWS:
            out.append(replace(sv, views=views))
    return out


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _calibrate_gps(site_list, target=1.0, lo=0.0, hi=8.0, iters=16):
    import reprojection_residual as rr
    f = lambda m: rr.pooled_chi2_dof(site_list, sigma_gps_m=m)  # noqa: E731
    if f(lo) <= target:
        return lo
    if f(hi) > target:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def _write(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(DATA / name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f'wrote {DATA / name} ({len(rows)} rows)')


DATA_PARTS = ('inputs', 'fig1', 'fig2a', 'fig2b', 'fig3', 'pairs', 'fig4', 'fig5', 'fig6',
              'fig7b')


def cmd_data(args):
    import fuse_sites as fs
    import reprojection_residual as rr
    import geo
    from detectors import BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE
    run_root = args.run_root
    cities = (CITY,) + COMPARATORS
    parts = set(args.only or DATA_PARTS)
    loaded = {c: _load_city(run_root, c, BENCHMARK_CONFIDENCE) for c in cities}

    if 'inputs' in parts:
        rows = [{'file': f'runs/{c}/results.jsonl',
                 'sha256': _sha256(run_root / c / 'results.jsonl'),
                 'used_for': 'the measurement' if c == CITY else 'comparator'} for c in cities]
        rows.append({'file': 'runs/bayonne/osm_streets.json',
                     'sha256': _sha256(run_root / CITY / 'osm_streets.json'),
                     'used_for': 'Figure 6 cross-track and streets (Overpass pull cached by the run)'})
        _write('inputs.csv', rows)

    # fig1: the verdict, rig-matched
    fig1 = []
    for c in cities:
        rows = loaded[c][4]
        groups = [('all', rows)]
        rigs = sorted({r['camera_model'] for r in rows})
        if len(rigs) > 1:
            groups += [(rig, [r for r in rows if r['camera_model'] == rig]) for rig in rigs]
        for rig, grp in groups:
            if len(grp) < 30:
                continue
            point, ci = _boot(grp, _stats)
            label_rig = rig if rig != 'all' else ('+'.join(rigs) if len(rigs) > 1 else rigs[0])
            fig1.append({'run': c, 'subset': rig, 'rig': label_rig,
                         'gopro_max': int(label_rig.lower().replace('gopro ', '') == 'max'),
                         'n_views': len(grp), 'n_sites': len({r['site_id'] for r in grp}),
                         **{k: point[k] for k in ('chi2_dof', 'chi2_dof_median', 'm_p50')},
                         **{f'{k}_lo': ci[k][0] for k in ('chi2_dof', 'chi2_dof_median', 'm_p50')},
                         **{f'{k}_hi': ci[k][1] for k in ('chi2_dof', 'chi2_dof_median', 'm_p50')}})
    if 'fig1' in parts:
        _write('fig1_verdict.csv', fig1)

    # fig2a: residual by range bin -- Bayonne, Richmond (all views) and Richmond's GoPro
    # Max views RE-SOLVED on their own (_rig_sites: each held out against GoPro Max mates
    # only, as Bayonne's are), the rig-matched reference
    fig2a = []
    for c, subset in ((CITY, 'all'), ('richmond', 'all'), ('richmond', 'GoPro Max')):
        if subset == 'all':
            rows = loaded[c][4]
        else:
            _, by_id, sites, frame, _, rigs = loaded[c]
            rows = rr.gtfree_rows(c, '2.6m', _rig_sites(sites, rigs, subset), frame, by_id,
                                  HEIGHT, rigs)
        for b in RANGE_BINS:
            grp = [r for r in rows if rr._range_bucket(r['range_m']) == b]
            point, ci = _boot(grp, _stats)
            fig2a.append({'run': c, 'subset': subset, 'range_bin': b, 'n_views': len(grp),
                          **{k: point[k] for k in point},
                          **{f'{k}_lo': ci[k][0] for k in ci},
                          **{f'{k}_hi': ci[k][1] for k in ci}})
    if 'fig2a' in parts:
        _write('fig2a_range.csv', fig2a)

    # fig2b: the leave-one-out-calibrated sigma_gps (chi2/dof = 1), per run, plus
    # Richmond's GoPro Max views re-solved on their own (rig-matched)
    fig2b = []
    pops = [(c, 'all') for c in cities] + [('richmond', 'GoPro Max')]
    for c, subset in (pops if 'fig2b' in parts else ()):
        sites = [s for s in loaded[c][2] if len(s.views) >= rr.MIN_VIEWS]
        if subset != 'all':
            sites = _rig_sites(sites, loaded[c][5], subset)
        rig = sorted({loaded[c][5].get(v.pano_id) or 'none' for sv in sites for v in sv.views})
        point = _calibrate_gps(sites)
        rng = random.Random(SEED)
        draws = []
        for _ in range(N_BOOT_GPS):
            v = _calibrate_gps(rng.choices(sites, k=len(sites)))
            if v is not None:
                draws.append(v)
        fig2b.append({'run': c, 'subset': subset, 'rig': '+'.join(rig),
                      'gopro_max': int(rig in (['Max'], ['GoPro Max'])),
                      'n_sites': len(sites), 'sigma_gps_chi2_1_m': point,
                      'lo': _pct(draws, 2.5), 'hi': _pct(draws, 97.5),
                      'n_boot': len(draws), 'model_sigma_gps_m': geo.MAPILLARY_ERRORS.sigma_gps_m})
        print(c, subset, 'sigma_gps', point)
    if 'fig2b' in parts:
        _write('fig2b_sigma_gps.csv', fig2b)

    # fig3: the same-sequence split
    fig3 = []
    for c in cities:
        rows = loaded[c][4]
        for b in ('all_same', 'some', 'none'):
            grp = [r for r in rows if r['seq_mates'] == b]
            point, ci = _boot(grp, _stats)
            if point is None:
                continue
            fig3.append({'run': c, 'seq_mates': b, 'n_views': len(grp),
                         'share_of_views': len(grp) / len(rows),
                         'chi2_dof': point['chi2_dof'], 'chi2_dof_lo': ci['chi2_dof'][0],
                         'chi2_dof_hi': ci['chi2_dof'][1],
                         'chi2_dof_median': point['chi2_dof_median'], 'm_p50': point['m_p50'],
                         'views_per_site_p50': statistics.median(r['n_views'] for r in grp)})
        sites = [s for s in loaded[c][2] if len(s.views) >= rr.MIN_VIEWS]
        fig3.append({'run': c, 'seq_mates': 'site_makeup', 'n_views': len(rows),
                     'site_views_p50': _pct([len(s.views) for s in sites], 50),
                     'site_sequences_p50': _pct([len({loaded[c][1][v.pano_id].sequence_id
                                                      for v in s.views}) for s in sites], 50)})
    if 'fig3' in parts:
        _write('fig3_seqsplit.csv', fig3)

    # share of same-site view PAIRS that come from one capture sequence (sites >= 3 views)
    if 'pairs' in parts:
        share = []
        for c in cities:
            by_id = loaded[c][1]
            same = tot = 0
            for sv in loaded[c][2]:
                if len(sv.views) < rr.MIN_VIEWS:
                    continue
                q = [by_id[v.pano_id].sequence_id for v in sv.views]
                for i in range(len(q)):
                    for j in range(i + 1, len(q)):
                        tot += 1
                        same += q[i] is not None and q[i] == q[j]
            share.append({'run': c, 'pairs': tot, 'same_sequence_pairs': same,
                          'same_sequence_share': same / tot if tot else None})
        _write('fig3_pair_share.csv', share)

    # fig4: pose ablation, same-site set at the 25 m cap, both tiers
    panos = loaded[CITY][0]
    fig4 = []
    for tier in ((BENCHMARK_CONFIDENCE, OPERATIONAL_CONFIDENCE) if 'fig4' in parts else ()):
        params = fs.FuseParams(min_confidence=tier, camera_height_m=HEIGHT)
        groups, by_id, frame = fs.posed_groups(panos, params)
        for label, keep in fs.SAME_SITE_SUBSETS:
            kept, cand, per = fs.same_site_spreads(groups, by_id, frame, params,
                                                   fs.POSE_CONVENTIONS, keep)
            for name, _ in fs.POSE_CONVENTIONS:
                glist = per[name]
                flat = [x for g in glist for x in g]
                rng = random.Random(SEED)
                med, p90 = [], []
                for _ in range(N_BOOT):
                    idx = [rng.randrange(len(glist)) for _ in glist]
                    d = [x for i in idx for x in glist[i]]
                    med.append(statistics.median_low(d))
                    p90.append(_pct(d, 90))
                fig4.append({'tier': tier, 'subset': label, 'convention': name.strip(),
                             'groups_kept': kept, 'groups_candidate': cand,
                             'pairs': len(flat), 'max_range_m': params.max_range_m,
                             'median_m': statistics.median_low(flat),
                             'median_lo': _pct(med, 2.5), 'median_hi': _pct(med, 97.5),
                             'p90_m': _pct(flat, 90), 'p90_lo': _pct(p90, 2.5),
                             'p90_hi': _pct(p90, 97.5)})
    if 'fig4' in parts:
        _write('fig4_pose.csv', fig4)

    # fig5: detections per pano, both tiers (normal-approximation CI over panos)
    fig5 = []
    for c in (cities if 'fig5' in parts else ()):
        counts = {BENCHMARK_CONFIDENCE: [], OPERATIONAL_CONFIDENCE: []}
        sub_floor = False
        with open(run_root / c / 'results.jsonl', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                dets = json.loads(line).get('detections') or []
                sub_floor = sub_floor or any(d['confidence'] < BENCHMARK_CONFIDENCE for d in dets)
                for t in counts:
                    counts[t].append(sum(d['confidence'] >= t for d in dets))
        for t, v in counts.items():
            mean = sum(v) / len(v)
            se = statistics.pstdev(v) / math.sqrt(len(v))
            fig5.append({'run': c, 'tier': t, 'panos': len(v), 'detections': sum(v),
                         'per_pano': mean, 'lo': mean - 1.96 * se, 'hi': mean + 1.96 * se,
                         'stores_sub_055': int(sub_floor)})
    if 'fig5' in parts:
        _write('fig5_detections.csv', fig5)
    if 'fig6' in parts:
        _fig6_data(run_root, panos)
    if 'fig7b' in parts:
        _fig7b_data(loaded)


def _fig6_data(run_root, panos):
    import geo
    import position_check as pc
    # fig6: pano positions with cross-track to OSM, and the streets
    run_dir = run_root / CITY
    with open(run_dir / 'osm_streets.json', encoding='utf-8') as f:
        osm = json.load(f)
    lat0 = statistics.median(p.lat for p in panos)
    lng0 = statistics.median(p.lng for p in panos)
    frame = geo.LocalFrame(lat0, lng0)
    index = pc.StreetIndex(pc.street_segments(osm, frame))
    pos = []
    for p in sorted(panos, key=lambda p: p.pano_id):
        m = pc.measure_point(index, *frame.to_enu(p.lat, p.lng))
        pos.append({'lat': round(p.lat, 6), 'lng': round(p.lng, 6),
                    'cross_track_m': None if m is None else round(m['cross'], 2)})
    _write('fig6_positions.csv', pos)
    # streets clipped to the area: ways with a vertex within ~100 m of the polygon
    from shapely.geometry import Point, shape
    with open(run_dir / 'area.geojson', encoding='utf-8') as f:
        area = json.load(f)
    near = shape(area.get('geometry', area)).buffer(0.0012)
    streets = []
    for way in osm.get('elements', []):
        pts = way.get('geometry') or []
        if not any(near.contains(Point(pt['lon'], pt['lat'])) for pt in pts):
            continue
        for i, pt in enumerate(pts):
            streets.append({'way': way.get('id'), 'i': i, 'lat': round(pt['lat'], 5),
                            'lng': round(pt['lon'], 5)})
    _write('fig6_streets.csv', streets)


def _fig7b_data(loaded):
    import geo
    import reprojection_residual as rr
    # fig7b: one multi-view site in plan view
    _, by_id, sites, frame, rows, _ = loaded[CITY]
    pooled = _pct([r['dist_m'] for r in rows], 50)
    per_site = {}
    for r in rows:
        per_site.setdefault(r['site_id'], []).append(r['dist_m'])
    cands = [s for s in sites if len(s.views) >= 5]
    site = min(cands, key=lambda s: (abs(statistics.median(per_site[s.site_id]) - pooled),
                                     s.site_id))
    lam, eta = rr.accumulate(site.views)
    cov = geo.sym2_inv(lam)
    out = [{'kind': 'site', 'site_id': site.site_id, 'e': site.e, 'n': site.n,
            'cov_ee': cov[0], 'cov_en': cov[1], 'cov_nn': cov[2],
            'site_median_loo_m': statistics.median(per_site[site.site_id]),
            'run_median_loo_m': pooled}]
    for v in site.views:
        p = by_id[v.pano_id]
        ce, cn = frame.to_enu(p.lat, p.lng)
        out.append({'kind': 'view', 'site_id': site.site_id, 'pano_id': v.pano_id,
                    'sequence_id': p.sequence_id, 'cam_e': ce, 'cam_n': cn,
                    'e': v.e, 'n': v.n, 'cov_ee': v.cov[0], 'cov_en': v.cov[1],
                    'cov_nn': v.cov[2], 'range_m': v.range_m, 'confidence': v.conf})
    _write('fig7b_site.csv', out)


# ============================================================================ examples
def _fetch_image(url):
    import requests
    from PIL import Image
    from sources import panoramax as px
    return Image.open(io.BytesIO(requests.get(url, headers=px._headers(),
                                              timeout=180).content)).convert('RGB')


def cmd_skips(args):
    """data/skips.csv: why each cached skip of the run was skipped (network: one STAC item
    per id, plus the first 512 KB of the image for the served size). main.py logs no
    per-pano reasons, so this reconstructs them from today's metadata."""
    import requests
    from PIL import Image
    from sources import panoramax as px
    run = args.run_root / CITY
    done = {l.strip() for l in open(run / 'already_processed.txt', encoding='utf-8') if l.strip()}
    got = set()
    with open(run / 'results.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                got.add(json.loads(line)['pano']['panorama_id'])
    rows = []
    for pid in sorted(done - got):
        item, gone = px.fetch_item(pid)
        row = {'pano_id': pid, 'camera_model': None, 'declared_w': None, 'declared_h': None,
               'served_w': None, 'served_h': None, 'reason': None}
        if gone or item is None:
            row['reason'] = 'item gone' if gone else 'item unavailable today'
            rows.append(row)
            continue
        props = item['properties']
        orient = props.get('pers:interior_orientation') or {}
        dims = orient.get('sensor_array_dimensions') or [None, None]
        row.update(camera_model=orient.get('camera_model'), declared_w=dims[0],
                   declared_h=dims[1])
        if props.get('view:azimuth') is None:
            row['reason'] = 'no view:azimuth'
            rows.append(row)
            continue
        with requests.get(px.image_url(item), headers=px._headers(), timeout=180,
                          stream=True) as r:
            head = r.raw.read(512 * 1024)
        try:
            w, h = Image.open(io.BytesIO(head)).size
        except Exception:
            w = h = None
        row.update(served_w=w, served_h=h)
        if w and h and abs(w - 2 * h) > 0.02 * w:
            row['reason'] = 'served image not 2:1 (vertically cropped)'
        else:
            row['reason'] = 'unexplained today: metadata and image fine (see #127)'
        rows.append(row)
        print(pid, row['reason'])
    _write('skips.csv', rows)


def cmd_examples(args):
    """Thumbnails for Figures 7a and 7c (network; an existing thumbnail is reused). 7c is
    the first cropped-MAX2 skip in pano-id order (data/skips.csv)."""
    from PIL import Image
    from detectors import BENCHMARK_CONFIDENCE
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    recs = []
    with open(args.run_root / CITY / 'results.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r['pano']['panorama_id'])
    top = sorted(recs, key=lambda r: (-max((d['confidence'] for d in r['detections']),
                                           default=0.0), r['pano']['panorama_id']))[:4]
    top_ids = {r['pano']['panorama_id'] for r in top}
    pool = [r for r in recs if r['pano']['panorama_id'] not in top_ids]
    rand = random.Random(SEED).sample(pool, 4)
    rows = []
    for sel, group in (('top', top), ('random', rand)):
        for r in group:
            pid = r['pano']['panorama_id']
            thumb = EXAMPLES / f'{pid}.jpg'
            if not thumb.exists():
                _fetch_image(r['pano']['source_metadata']['hd_url']).resize(
                    (768, 384), Image.LANCZOS).save(thumb, quality=80)
            dets = [d for d in r['detections'] if d['confidence'] >= BENCHMARK_CONFIDENCE]
            rows.append({'selection': sel, 'pano_id': pid,
                         'capture_date': r['pano']['capture_date'],
                         'width': r['pano']['width'], 'height': r['pano']['height'],
                         'producer': r['pano'].get('copyright'),
                         'license': r['pano'].get('license'),
                         'detections': json.dumps([[round(d['x_normalized'], 5),
                                                    round(d['y_normalized'], 5),
                                                    round(d['confidence'], 3)] for d in dets])})
    _write('examples/contact_sheet.csv', rows)
    skips = [r for r in _read('skips.csv') if str(r['reason']).startswith('served image not 2:1')]
    pid = sorted(r['pano_id'] for r in skips)[0]
    from sources import panoramax as px
    item, _ = px.fetch_item(pid)
    url = px.image_url(item)
    thumb = EXAMPLES / f'max2_{pid}.jpg'
    img = None
    if not thumb.exists():
        img = _fetch_image(url)
        img.resize((768, round(768 * img.size[1] / img.size[0])), Image.LANCZOS).save(
            thumb, quality=80)
    row = next(r for r in skips if r['pano_id'] == pid)
    producers = item.get('providers') or []
    _write('examples/max2_skip.csv', [{
        'pano_id': pid, 'rule': 'first cropped-MAX2 skip in pano-id order (data/skips.csv)',
        'camera_model': row['camera_model'], 'declared_w': int(row['declared_w']),
        'declared_h': int(row['declared_h']), 'served_w': int(row['served_w']),
        'served_h': int(row['served_h']),
        'producer': item['properties'].get('geovisio:producer')
        or next((p['name'] for p in producers if p.get('name')), None),
        'license': item['properties'].get('license'), 'url': url}])


# Figure 7d: crops around detections, from the native-resolution RampNet bundle (no network).
# Privacy rule, stated before looking: a crop whose centred subject is a person at close
# range is skipped and the next detection in rank is taken; each skip is recorded here (after
# a visual check of the candidate crops) and in crops.csv.
CROP_DEG = (20.0, 15.0)
CROP_PX = (240, 180)
PRIVACY_SKIPS = {}   # {(pano_id, detection index in the bundle record): reason}


def cmd_crops(args):
    import geo
    from PIL import Image
    bundle = args.bundle
    recs = [json.loads(l) for l in open(bundle / 'records.jsonl', encoding='utf-8') if l.strip()]
    dets = []
    for r in recs:
        for i, d in enumerate(r['detections']):
            if d['confidence'] >= 0.55:
                dets.append((r, i, d))
    dets.sort(key=lambda t: (-t[2]['confidence'], t[0]['pano']['panorama_id'], t[1]))
    shuffled = dets[:]
    random.Random(SEED).shuffle(shuffled)
    out, used = [], set()
    out_dir = EXAMPLES / 'crops'
    out_dir.mkdir(parents=True, exist_ok=True)
    for sel, order, n in (('top', dets, 6), ('random', shuffled, 6)):
        taken = 0
        for rank, (r, i, d) in enumerate(order):
            if taken == n:
                break
            p = r['pano']
            key = (p['panorama_id'], i)
            if key in used:
                continue
            used.add(key)
            row = {'selection': sel, 'rank': rank, 'pano_id': p['panorama_id'], 'det_index': i,
                   'x': round(d['x_normalized'], 5), 'y': round(d['y_normalized'], 5),
                   'confidence': round(d['confidence'], 3), 'capture_date': p['capture_date'],
                   'producer': p.get('copyright'), 'license': p.get('license'),
                   'privacy_skip': PRIVACY_SKIPS.get(key, '')}
            g = geo.detection_ground_point(geo.pano_pose(p), d['x_normalized'],
                                           d['y_normalized'], camera_height=HEIGHT,
                                           max_range_m=math.inf, apply_pose=False)
            row['range_m'] = None if g is None else round(g.range_m, 1)
            out.append(row)
            if row['privacy_skip']:
                continue
            img = Image.open(bundle / 'panos' / f"{p['panorama_id']}.jpg").convert('RGB')
            w, h = img.size
            cw, ch = round(w * CROP_DEG[0] / 360), round(h * CROP_DEG[1] / 180)
            cx, cy = d['x_normalized'] * w, d['y_normalized'] * h
            x0, y0 = round(cx - cw / 2), max(0, min(h - ch, round(cy - ch / 2)))
            crop = Image.new('RGB', (cw, ch))
            left = x0 % w               # seam-wrapped: at most two pieces
            first = min(cw, w - left)
            crop.paste(img.crop((left, y0, left + first, y0 + ch)), (0, 0))
            if first < cw:
                crop.paste(img.crop((0, y0, cw - first, y0 + ch)), (first, 0))
            row['crop_cx'] = round((cx - x0) / cw, 4)
            row['crop_cy'] = round((cy - y0) / ch, 4)
            crop.resize(CROP_PX, Image.LANCZOS).save(
                out_dir / f"{p['panorama_id']}_{i}.jpg", quality=82)
            taken += 1
    _write('examples/crops.csv', out)


# ============================================================================ figures
def _read(name):
    with open(DATA / name, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ('', None) else None
            except ValueError:
                pass
    return rows


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'svg.hashsalt': 'panoramax-bayonne-57', 'svg.fonttype': 'none',
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 2})
    return plt


def _save(fig, name, svg=True, svg_dpi=72, quantize=False, jpeg=False):
    """PNG (200 dpi) + SVG, with no timestamps. The SVG goes through a buffer and is
    written with '\n' line endings, so it is byte-identical across platforms (the repo's
    .gitattributes keeps *.svg as LF). Photo figures pass svg=False: an SVG would embed the
    photos a second time. Rasterized artists (the 28k-point map) go in at svg_dpi."""
    OUT.mkdir(parents=True, exist_ok=True)
    if jpeg:
        # photo figures: a 256-colour palette posterises the photos, so these go out as a
        # quality-92 JPEG (4:4:4) rendered from the same 200 dpi raster; PIL's encoder is
        # deterministic for fixed input and settings
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                    metadata={'Software': None})
        Image.open(io.BytesIO(buf.getvalue())).convert('RGB').save(
            OUT / f'{name}.jpg', quality=92, subsampling=0)
        print(f'wrote {OUT / name}.jpg')
        return
    if quantize:
        # photo/dense figures: an adaptive 256-colour palette (median cut, Floyd-Steinberg),
        # deterministic, a third of the size of a truecolour PNG
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                    metadata={'Software': None})
        img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
        img.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                     dither=Image.Dither.FLOYDSTEINBERG).save(OUT / f'{name}.png', optimize=True)
    else:
        fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight',
                    metadata={'Software': None})
    if svg:
        buf = io.StringIO()
        fig.savefig(buf, format='svg', bbox_inches='tight', dpi=svg_dpi,
                    metadata={'Date': None, 'Creator': None})
        with open(OUT / f'{name}.svg', 'w', encoding='utf-8', newline='\n') as f:
            f.write(buf.getvalue())
    print(f'wrote {OUT / name}.png' + ('/.svg' if svg else ''))


GOPRO_MAX_RUNS = ('laurens', 'morgantown')   # all-GoPro-Max Mapillary runs
CREDIT_MUNI = ('Imagery: sig_bayonne via Panoramax (panoramax.ign.fr), '
               'Licence Ouverte / Etalab 2.0')
CREDIT_OSM = 'Streets: © OpenStreetMap contributors (ODbL)'

RUN_LABEL = {'bayonne': 'Bayonne (Panoramax)', 'richmond': 'Richmond', 'clovis': 'Clovis',
             'laurens': 'Laurens', 'annapolis': 'Annapolis', 'morgantown': 'Morgantown'}


def _color(row):
    if row['run'] == CITY:
        return BLUE
    return ORANGE if row.get('gopro_max') == 1.0 else MUTED


def fig1(plt):
    rows = [r for r in _read('fig1_verdict.csv')
            if r['subset'] == 'all' or (r['run'] == 'richmond'
                                        and r['subset'] in ('GoPro Max', 'iSTAR Pulsar'))]
    order = sorted(rows, key=lambda r: -r['chi2_dof'])
    labels = []
    for r in order:
        sub = '' if r['subset'] == 'all' else f" · {r['subset']} views"
        rig = ('GoPro Max' if r['rig'] == 'Max' else r['rig']) if r['subset'] == 'all' else ''
        labels.append(f"{RUN_LABEL[r['run']]}{sub}" + (f"  [{rig}]" if rig else ''))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True,
                               gridspec_kw={'width_ratios': [1.25, 1]})
    y = list(range(len(order)))[::-1]
    a.axvspan(*BAND, color='#cde2fb', alpha=0.6, lw=0)
    a.axvline(ANCHOR, color=INK2, lw=1)
    a.text(ANCHOR, len(order) - 0.45, f' Richmond anchor {ANCHOR}', color=INK2, fontsize=8.5)
    a.text(BAND[0], -0.75, f' amended ADEQUATE band [{BAND[0]:.2f}, {BAND[1]:.2f}]',
           color=INK2, fontsize=8.5)
    a.set_ylim(-1.1, len(order) - 0.2)
    for yy, r in zip(y, order):
        c = _color(r)
        a.plot([r['chi2_dof_lo'], r['chi2_dof_hi']], [yy + 0.12] * 2, color=c, lw=1.6)
        a.plot(r['chi2_dof'], yy + 0.12, 'o', ms=8, color=c, mec=SURFACE, mew=1.5)
        a.plot([r['chi2_dof_median_lo'], r['chi2_dof_median_hi']], [yy - 0.14] * 2, color=c,
               lw=1, alpha=0.8)
        a.plot(r['chi2_dof_median'], yy - 0.14, 'D', ms=6, mfc=SURFACE, mec=c, mew=1.6)
    bay = next(r for r in order if r['run'] == CITY)
    a.annotate(f"{bay['chi2_dof']:.3f} (median form {bay['chi2_dof_median']:.3f})",
               (bay['chi2_dof_hi'], y[order.index(bay)] + 0.12), xytext=(6, 0),
               textcoords='offset points', va='center', fontsize=9, color=INK)
    a.set_yticks(y, labels)
    a.set_xlabel('leave-one-out chi²/dof under MAPILLARY_ERRORS (2 dof per held-out view)')
    a.set_title('chi²/dof: ● pooled, ◇ median form (95% CI)')
    a.set_xlim(0, 0.95)
    b.axvspan(*M_BAND, color='#cde2fb', alpha=0.6, lw=0)
    b.axvline(M_ANCHOR, color=INK2, lw=1)
    for yy, r in zip(y, order):
        c = _color(r)
        b.plot([r['m_p50_lo'], r['m_p50_hi']], [yy] * 2, color=c, lw=1.6)
        b.plot(r['m_p50'], yy, 'o', ms=8, color=c, mec=SURFACE, mew=1.5)
    b.text(M_BAND[0], -0.75, f' metre clause [{M_BAND[0]:.2f}, {M_BAND[1]:.2f}] m',
           color=INK2, fontsize=8.5)
    b.set_xlabel('median leave-one-out residual (m)')
    b.set_title('median residual, m (95% CI)')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', color=BLUE, ms=8, label='Bayonne (GoPro Max)'),
               Line2D([], [], marker='o', ls='', color=ORANGE, ms=8,
                      label='Mapillary, GoPro Max views'),
               Line2D([], [], marker='o', ls='', color=MUTED, ms=8,
                      label='Mapillary, other rigs / mixed')]
    b.legend(handles=handles, loc='lower right', fontsize=8.5, bbox_to_anchor=(1.0, 0.08))
    fig.suptitle('Figure 1. Is Bayonne outside the Richmond-anchored band? Marginally: 8% '
                 'over the bar, 1.16x Richmond\'s GoPro Max views (CIs just separate)',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig1_verdict')


def fig2(plt):
    rows = _read('fig2a_range.csv')
    gps = _read('fig2b_sigma_gps.csv')
    fig = plt.figure(figsize=(12, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.45)
    x = list(range(len(RANGE_BINS)))
    series = ((CITY, 'all', BLUE, 'Bayonne'), ('richmond', 'all', MUTED, 'Richmond, all'),
              ('richmond', 'GoPro Max', ORANGE, 'Richmond, GoPro Max (re-solved)'))
    for k, (key, title) in enumerate((('m_p50', 'total |residual|'),
                                      ('along_p50', '|along-ray|'),
                                      ('cross_p50', '|cross-ray|'))):
        ax = fig.add_subplot(gs[0, k])
        for run, subset, c, label in series:
            rr_ = [r for r in rows if r['run'] == run and r['subset'] == subset]
            ax.fill_between(x, [r[f'{key}_lo'] for r in rr_], [r[f'{key}_hi'] for r in rr_],
                            color=c, alpha=0.15, lw=0)
            ax.plot(x, [r[key] for r in rr_], '-o', color=c, ms=6, mec=SURFACE, mew=1.2,
                    label=label)
        bay = [r[key] for r in rows if r['run'] == CITY]
        ric = [r[key] for r in rows if r['run'] == 'richmond' and r['subset'] == 'all']
        gm = [r[key] for r in rows if r['run'] == 'richmond' and r['subset'] == 'GoPro Max']
        ax.text(0.02, 0.98, 'Bayonne minus Richmond all: '
                + ' / '.join(f'{b - r:+.1f}' for b, r in zip(bay, ric)) + ' m' + chr(10)
                + 'minus Richmond GoPro Max: '
                + ' / '.join(f'{b - r:+.1f}' for b, r in zip(bay, gm)) + ' m',
                transform=ax.transAxes, va='top', fontsize=7, color=INK)
        if key == 'm_p50':
            ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, (max(r[f'{key}_hi'] for r in rows) or 1) * 1.3)
        ax.set_xticks(x, [f'{b} m' for b in RANGE_BINS])
        ax.set_title(title)
        if k == 0:
            ax.set_ylabel('median, m (95% site-bootstrap band)')
        ax.set_xlabel('range of the held-out view')
    ax = fig.add_subplot(gs[1, :])
    gps = sorted(gps, key=lambda r: r['sigma_gps_chi2_1_m'])
    for i, r in enumerate(gps):
        c = BLUE if r['run'] == CITY else ORANGE if r['gopro_max'] == 1.0 else MUTED
        ax.plot([r['lo'], r['hi']], [i, i], color=c, lw=1.6)
        ax.plot(r['sigma_gps_chi2_1_m'], i, 'o', ms=8, color=c, mec=SURFACE, mew=1.5)
        ax.text(r['hi'] + 0.05, i, f"{r['sigma_gps_chi2_1_m']:.2f} m", va='center',
                fontsize=8.5, color=INK)
    ax.axvline(gps[0]['model_sigma_gps_m'], color=INK2, lw=1)
    ax.text(gps[0]['model_sigma_gps_m'], len(gps) - 0.5,
            ' MAPILLARY_ERRORS sigma_gps = 3 m', color=INK2, fontsize=8.5)
    labels = []
    for r in gps:
        rig = 'GoPro Max' if r['gopro_max'] == 1.0 else r['rig'].replace('+', ' + ')
        sub = ', GoPro Max views only' if r['subset'] != 'all' else ''
        labels.append(f"{RUN_LABEL[r['run']]}{sub}  [{rig}]")
    ax.set_yticks(range(len(gps)), labels, fontsize=8.5)
    ax.set_xlim(0, 3.6)
    ax.set_xlabel('sigma_gps that brings leave-one-out chi²/dof to 1, every other sigma '
                  'fixed (m; 95% CI, 200 site resamples). Orange: GoPro Max populations')
    ax.set_title('Calibrated between-view position scatter (sites re-solved per population)')
    fig.suptitle('Figure 2. Tilt or position? Rig-matched, the excess is a flat 0.6-1.0 m across '
                 'the ray; scatter 2x Richmond\'s, 1.2x its and Laurens\' GoPro Max views',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold')
    _save(fig, 'fig2_position')


def fig3(plt):
    rows = [r for r in _read('fig3_seqsplit.csv') if r['seq_mates'] in ('all_same', 'none')]
    makeup = {r['run']: r for r in _read('fig3_seqsplit.csv') if r['seq_mates'] == 'site_makeup'}
    runs = [CITY] + list(COMPARATORS)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for i, run in enumerate(runs[::-1]):
        pts = {r['seq_mates']: r for r in rows if r['run'] == run}
        s_, n = pts['all_same'], pts['none']
        c = BLUE if run == CITY else INK2
        ax.plot([s_['chi2_dof_lo'], s_['chi2_dof_hi']], [i + 0.15] * 2, color=c, lw=1.2)
        ax.plot([n['chi2_dof_lo'], n['chi2_dof_hi']], [i - 0.15] * 2, color=c, lw=1.2,
                alpha=0.6)
        ax.plot(s_['chi2_dof'], i + 0.15, 'o', ms=9, color=c, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(n['chi2_dof'], i - 0.15, 's', ms=8, mfc=SURFACE, mec=c, mew=2, zorder=3)
        overlap = s_['chi2_dof_hi'] >= n['chi2_dof_lo'] and n['chi2_dof_hi'] >= s_['chi2_dof_lo']
        verdict = ('opposite' if s_['chi2_dof'] > n['chi2_dof']
                   else 'predicted, CIs overlap' if overlap else 'predicted')
        ax.text(0.99, i, f"n {int(s_['n_views'])} / {int(n['n_views'])}: {verdict}",
                transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8.5,
                color=INK if verdict == 'opposite' else INK2,
                fontweight='bold' if verdict == 'opposite' else 'normal')
    ax.set_yticks(range(len(runs)),
                  [f"{RUN_LABEL[r]}  ({int(makeup[r]['site_views_p50'])} views / "
                   f"{int(makeup[r]['site_sequences_p50'])} seq. per site)" for r in runs[::-1]],
                  fontsize=8.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=INK2, ms=9,
                              label='every site-mate from the held-out view\'s sequence'),
                       Line2D([], [], marker='s', ls='', mfc=SURFACE, mec=INK2, mew=2, ms=8,
                              label='no site-mate from its sequence')],
              loc='upper center', fontsize=8.5, bbox_to_anchor=(0.45, -0.13), ncol=2)
    ax.set_xlabel('leave-one-out chi²/dof of the held-out views (95% site-bootstrap CI). '
                  'In Richmond and Annapolis the two groups also differ in site size.')
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_title('Figure 3. Does same-sequence error cancel in the leave-one-out? Not '
                 'established: Richmond, the anchor, reads the opposite way', fontsize=11)
    fig.tight_layout()
    _save(fig, 'fig3_seqsplit')


def fig4(plt):
    rows = [r for r in _read('fig4_pose.csv') if r['subset'] == 'real-tilt members only']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, tier in zip(axes, (0.55, 0.3)):
        rr_ = [r for r in rows if abs(r['tier'] - tier) < 1e-9]
        names = [r['convention'] for r in rr_]
        for i, r in enumerate(rr_[::-1]):
            c = BLUE if r['convention'].startswith('off') else MUTED
            ax.plot([r['median_lo'], r['median_hi']], [i + 0.12] * 2, color=c, lw=1.4)
            ax.plot(r['median_m'], i + 0.12, 'o', ms=8, color=c, mec=SURFACE, mew=1.5)
            ax.plot([r['p90_lo'], r['p90_hi']], [i - 0.14] * 2, color=c, lw=1, alpha=0.8)
            ax.plot(r['p90_m'], i - 0.14, 'D', ms=6, mfc=SURFACE, mec=c, mew=1.6)
        ax.set_yticks(range(len(names)), names[::-1])
        r0 = rr_[0]
        ax.set_title(f"tier {tier}: {int(r0['groups_kept'])} of {int(r0['groups_candidate'])} "
                     f"groups, {int(r0['pairs'])} pairs, all within {r0['max_range_m']:g} m")
        ax.set_xlabel('within-site pairwise member distance (m): ● median, ◇ p90 (95% CI)')
        ax.set_xlim(0, 15)
    fig.suptitle('Figure 4. Should pers:pitch/roll be applied? No: every sign convention '
                 'loosens the spread of real-tilt members (same sites, 25 m cap)',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig4_pose')


def fig5(plt):
    rows = _read('fig5_detections.csv')
    hi_t = [r for r in rows if abs(r['tier'] - 0.55) < 1e-9]
    lo_t = {r['run']: r for r in rows if abs(r['tier'] - 0.3) < 1e-9}
    hi_t.sort(key=lambda r: r['per_pano'])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, r in enumerate(hi_t):
        c = BLUE if r['run'] == CITY else ORANGE if r['run'] in GOPRO_MAX_RUNS else MUTED
        l = lo_t[r['run']]
        ax.plot([r['per_pano'], l['per_pano']], [i, i], color=GRID, lw=3, zorder=1)
        ax.plot(r['per_pano'], i, 'o', ms=9, color=c, mec=SURFACE, mew=1.5, zorder=3)
        if l['stores_sub_055']:
            ax.plot(l['per_pano'], i, 's', ms=8, mfc=SURFACE, mec=c, mew=2, zorder=3)
        ax.text(max(r['per_pano'], l['per_pano']) + 0.03, i,
                f"{r['per_pano']:.3f}" + ('' if l['stores_sub_055'] else
                                          '  (pre-floor run: 0.30 tier = 0.55)'),
                va='center', fontsize=8.5, color=INK)
    ax.set_yticks(range(len(hi_t)), [RUN_LABEL[r['run']] for r in hi_t])
    ax.set_xlabel('detections per processed pano: ● tier 0.55, □ tier 0.30 (95% normal-'
                  'approximation CI over panos, narrower than the markers). Orange: GoPro Max runs')
    ax.set_xlim(0, 1.45)
    ax.set_title('Figure 5. Is Bayonne\'s detection rate anomalous? No: it sits inside the '
                 'Mapillary range at 0.55', fontsize=11.5)
    fig.tight_layout()
    _save(fig, 'fig5_detections')


def fig6(plt):
    years = _read('census/years.csv')
    rigs = _read('census/rigs.csv')
    pose = _read('census/pose.csv')
    pos = _read('fig6_positions.csv')
    streets = _read('fig6_streets.csv')
    with open(REPO_ROOT / 'runs' / CITY / 'area.geojson', encoding='utf-8') as f:
        area = json.load(f)
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.9], hspace=0.6, wspace=0.25)
    for k, (rows, key, title) in enumerate((
            (years, 'capture_year', 'capture year'),
            (rigs, None, 'rig (camera model, image size)'),
            (pose, 'pose_group', 'pers:pitch / pers:roll'))):
        ax = fig.add_subplot(gs[k, 0])
        labels = ([str(int(r[key])) if isinstance(r[key], float) else str(r[key]) for r in rows]
                  if key else [f"{r['camera_model'] or r['camera_make']} {r['dimensions']}"
                               for r in rows])
        vals = [r['share'] * 100 for r in rows]
        ax.barh(range(len(rows))[::-1], vals, color=BLUE, height=0.6)
        for i, v in zip(range(len(rows))[::-1], vals):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8.5, color=INK)
        ax.set_yticks(range(len(rows))[::-1], labels, fontsize=8.5)
        ax.set_xlim(0, 115)
        ax.set_title(f'{chr(65 + k)}. {title}')
        ax.grid(axis='y', visible=False)
    ax = fig.add_subplot(gs[:, 1])
    coslat = math.cos(math.radians(43.49))
    ways = {}
    for s in streets:
        ways.setdefault(s['way'], []).append((s['lng'], s['lat']))
    for pts in ways.values():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color='#c3c2b7', lw=0.4, zorder=1,
                rasterized=True)
    geom = area.get('geometry', area)
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    for poly in polys:
        ring = poly[0]
        ax.plot([p[0] for p in ring], [p[1] for p in ring], color=INK2, lw=1, zorder=2)
    bins = [(0, 1, '#86b6ef'), (1, 2, '#3987e5'), (2, 5, '#1c5cab'), (5, 30.01, '#0d366b')]
    far = [p for p in pos if p['cross_track_m'] is None]
    ax.scatter([p['lng'] for p in far], [p['lat'] for p in far], s=0.6, color=ORANGE,
               lw=0, zorder=3, rasterized=True, label=f'> 30 m from any street ({len(far)})')
    for lo, hi, c in bins:
        sel = [p for p in pos if p['cross_track_m'] is not None and lo <= p['cross_track_m'] < hi]
        ax.scatter([p['lng'] for p in sel], [p['lat'] for p in sel], s=0.6, color=c, lw=0,
                   zorder=3, rasterized=True, label=f'{lo:g}-{min(hi, 30):g} m ({len(sel)})')
    ring = polys[0][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    ax.set_xlim(min(xs) - 0.003, max(xs) + 0.003)
    ax.set_ylim(min(ys) - 0.003, max(ys) + 0.003)
    ax.set_aspect(1 / coslat)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    leg = ax.legend(title='cross-track to OSM centerline', loc='upper left', fontsize=8,
                    title_fontsize=8.5, markerscale=12, bbox_to_anchor=(1.0, 1.0))
    leg.get_title().set_color(INK2)
    x0, y0 = min(xs) + 0.002, min(ys) + 0.002
    km = 1.0 / (111.32 * coslat)   # 1 km in degrees of longitude
    ax.plot([x0, x0 + km], [y0, y0], color=INK, lw=2.5, solid_capstyle='butt')
    ax.text(x0 + km / 2, y0 + 0.0012, '1 km', ha='center', fontsize=8.5, color=INK)
    ax.set_title(f'D. {len(pos):,} thinned panos (10 m) over the commune polygon')
    ax.text(1.0, -0.02, CREDIT_OSM, transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=INK2)
    fig.suptitle('Figure 6. What did Bayonne\'s run cover? 2024-26 GoPro Max panos from one '
                 'municipal account, 42% without pose, on the street at a 1.54 m median',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold')
    _save(fig, 'fig6_census', svg=False, quantize=True)


def _ellipse(ax, e, n, cov, color, lw=1.2, alpha=1.0, nsig=1.0):
    vals, vecs = _eig2(cov)
    t = [i * 2 * math.pi / 72 for i in range(73)]
    xs, ys = [], []
    for a in t:
        u = nsig * math.sqrt(vals[0]) * math.cos(a)
        v = nsig * math.sqrt(vals[1]) * math.sin(a)
        xs.append(e + u * vecs[0][0] + v * vecs[1][0])
        ys.append(n + u * vecs[0][1] + v * vecs[1][1])
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha)


def _eig2(cov):
    a, b, c = cov
    tr, det = a + c, a * c - b * b
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    if abs(b) > 1e-12:
        v1 = (l1 - c, b)
    else:
        v1 = (1.0, 0.0) if a >= c else (0.0, 1.0)
    nv = math.hypot(*v1)
    v1 = (v1[0] / nv, v1[1] / nv)
    return (l1, l2), (v1, (-v1[1], v1[0]))


def fig7(plt):
    from PIL import Image
    sheet = _read('examples/contact_sheet.csv')
    fig, axes = plt.subplots(4, 2, figsize=(12, 7.0),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.28})
    for ax, r in zip(axes.T.flat, sheet):
        img = Image.open(EXAMPLES / f"{r['pano_id']}.jpg")
        ax.imshow(img, extent=(0, 1, 1, 0), aspect='auto')
        ax.axhline(NADIR_MASK_Y, color=ORANGE, lw=1.2)
        ax.axhspan(LOGO_BAND_Y, 1.0, color=AQUA, alpha=0.35, lw=0)
        for x, y, c in json.loads(r['detections']):
            ax.plot(x, y, 'o', ms=11, mfc='none', mec='#ffffff', mew=2.6)
            ax.plot(x, y, 'o', ms=11, mfc='none', mec=BLUE, mew=1.6)
            ax.text(x + 0.012, y - 0.03, f'{c:.2f}', color='#ffffff', fontsize=7.5,
                    bbox=dict(facecolor=INK, alpha=0.6, lw=0, pad=1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{r['selection']} · {r['pano_id'][:8]} · {r['capture_date']} · "
                     f"© {r['producer']} ({r['license']})", fontsize=8, fontweight='normal')
    fig.text(0.01, -0.035, 'Orange line: nadir mask (dip 49°, y 0.772; nothing below it '
             'ships). Aqua shading: the municipal logo band (y 0.791 to the bottom).\nBlue circles: '
             'detections ≥ 0.55, labelled with their stored confidence (a heatmap peak value, '
             'which can exceed 1).', fontsize=8.5, color=INK2)
    fig.suptitle('Figure 7a. What do Bayonne panos and their 0.55 detections look like? '
                 '4 highest-confidence + 4 random (seed 57)', x=0.01, ha='left',
                 fontsize=11.5, fontweight='bold')
    fig.text(0.01, -0.075, 'Imagery via Panoramax: producer and licence per panel '
             '(etalab-2.0 = Licence Ouverte / Etalab 2.0; CC-BY-SA-4.0).', fontsize=8.5,
             color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    _save(fig, 'fig7a_contact_sheet', svg=False, jpeg=True)

    site = _read('fig7b_site.csv')
    s = next(r for r in site if r['kind'] == 'site')
    views = [r for r in site if r['kind'] == 'view']
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    seqs = sorted({v['sequence_id'] for v in views})
    for v in views:
        k = seqs.index(v['sequence_id'])
        c = (ORANGE, AQUA, '#4a3aa7', MUTED)[min(k, 3)]
        mk = ('^', 'v', '<', '>')[min(k, 3)]   # marker repeats the sequence (not colour alone)
        ax.plot([v['cam_e'], v['e']], [v['cam_n'], v['n']], color=c, lw=1, alpha=0.7)
        ax.plot(v['cam_e'], v['cam_n'], mk, ms=9, color=c, mec=SURFACE, mew=1.2)
        ax.plot(v['e'], v['n'], 'o', ms=7, color=c, mec=SURFACE, mew=1.2)
        _ellipse(ax, v['e'], v['n'], (v['cov_ee'], v['cov_en'], v['cov_nn']), c, lw=0.8,
                 alpha=0.6)
    ax.plot(s['e'], s['n'], 'X', ms=12, color=INK, mec=SURFACE, mew=1.5)
    _ellipse(ax, s['e'], s['n'], (s['cov_ee'], s['cov_en'], s['cov_nn']), INK, lw=1.6)
    ax.set_aspect('equal')
    ax.set_xlabel('east (m, local frame)')
    ax.set_ylabel('north (m)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='^', ls='', color=MUTED, ms=9, label='camera'),
                       Line2D([], [], marker='o', ls='', color=MUTED, ms=7,
                              label='ground point (1σ ellipse, model)'),
                       Line2D([], [], marker='X', ls='', color=INK, ms=11,
                              label='fused site (1σ)')],
              loc='best', fontsize=8.5, title=f'{len(seqs)} sequences (colour)',
              title_fontsize=8.5)
    ax.set_title(f"Figure 7b. What does ground-point scatter look like? Site "
                 f"{int(s['site_id'])}: {len(views)} views, median leave-one-out "
                 f"{s['site_median_loo_m']:.2f} m\n(run median {s['run_median_loo_m']:.2f} m; "
                 f"chosen by rule: >= 5 views, median closest to the run's)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, 'fig7b_site_plan')

    m = _read('examples/max2_skip.csv')[0]
    img = Image.open(EXAMPLES / f"max2_{m['pano_id']}.jpg")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sw, sh = m['served_w'], m['served_h']
    dw, dh = m['declared_w'], m['declared_h']
    ax.imshow(img, extent=(0, sw, sh, 0))
    ax.plot([0, dw, dw, 0, 0], [0, 0, dh, dh, 0], color=ORANGE, lw=1.6)
    ax.text(dw / 2, dh - 80, f'declared {int(dw)}×{int(dh)} (2:1, a full 360×180)',
            ha='center', color=ORANGE, fontsize=9)
    ax.text(dw / 2, sh + 160, f'served {int(sw)}×{int(sh)} ({sw / sh:.2f}:1) — skipped: '
            f'not a full equirect', ha='center', color=INK, fontsize=9)
    ax.set_xlim(-50, dw + 50)
    ax.set_ylim(dh + 50, -50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"Figure 7c. Why were 104 GoPro MAX2 uploads skipped? The served image "
                 f"is cropped vertically ({m['pano_id'][:8]}; first such skip by id)",
                 fontsize=11)
    ax.text(0, dh + 260, f"Imagery: © {m['producer']} via Panoramax "
            f"(panoramax.openstreetmap.fr), {m['license']}", fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, 'fig7c_max2_skip', svg=False, jpeg=True)
    fig7d(plt)


def fig7d(plt):
    from PIL import Image
    rows = [r for r in _read('examples/crops.csv') if not r['privacy_skip']]
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.32})
    for ax, r in zip(axes.flat, rows):
        img = Image.open(EXAMPLES / 'crops' / f"{r['pano_id']}_{int(r['det_index'])}.jpg")
        ax.imshow(img, extent=(0, 1, 1, 0), aspect='auto')
        ax.plot(r['crop_cx'], r['crop_cy'], '+', ms=14, mew=1.5, color='#ffffff')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        rng = '—' if r['range_m'] is None else f"{r['range_m']:.1f} m"
        ax.set_title(f"{r['selection']} · conf {r['confidence']:.2f} · {rng} · "
                     f"{r['capture_date']}", fontsize=8, fontweight='normal')
    skipped = [r for r in _read('examples/crops.csv') if r['privacy_skip']]
    fig.text(0.01, 0.005, 'Crops 20°×15° centred on detections ≥ 0.55 (white +) from the '
             'native-resolution RampNet bundle; range is the flat raycast at 2.6 m.\n'
             'Privacy rule: a crop whose centred subject is a person at close range is skipped '
             f'and the next in rank taken ({len(skipped)} skipped). The bundle carries no '
             '0.30-0.55 detections.\n' + CREDIT_MUNI + '.', fontsize=8, color=INK2)
    fig.suptitle('Figure 7d. What did the detector fire on? 6 top + 6 random bundle detections '
                 '(seed 57), zoomed to 20° x 15°',
                 x=0.01, ha='left', fontsize=11.5, fontweight='bold')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)
    _save(fig, 'fig7d_crops', svg=False, jpeg=True)


def cmd_figures(_args):
    plt = _style()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f(plt)
        plt.close('all')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('data', 'skips', 'examples'):
        p = sub.add_parser(name)
        p.add_argument('--run-root', type=Path, default=REPO_ROOT / 'runs')
        if name == 'data':
            p.add_argument('--only', nargs='+', choices=DATA_PARTS,
                           help='regenerate only these outputs (default: all)')
    p = sub.add_parser('crops')
    p.add_argument('--bundle', type=Path, default=REPO_ROOT.parent / 'bayonne-bundle')
    sub.add_parser('figures')
    args = ap.parse_args()
    {'data': cmd_data, 'skips': cmd_skips, 'examples': cmd_examples, 'crops': cmd_crops,
     'figures': cmd_figures}[args.cmd](args)


if __name__ == '__main__':
    main()
