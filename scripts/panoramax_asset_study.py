"""Does detecting on Panoramax's SD derivative cost curb-ramp detections? (issue #161)

Panoramax's maintainers asked whether the full-resolution `hd` original is really needed,
since the `sd` derivative (2048 px wide) is much lighter to serve. This study answers it on
a seeded, rig-stratified sample of the Lyon run, with the production detector, and applies
a rule pre-registered in docs/panoramax-asset-study.md before any arm was scored.

Arms (every image is resized to 4096x2048 BILINEAR, exactly as sources/panoramax.py does):

- `hd`     the archived original (the bytes the Lyon run detected on). The sanity gate
           requires this arm to reproduce the run's stored detections.
- `sd`     the real `sd` asset, fetched at 1 picture/s through the source's own limiter.
- `sd_sim` the archived original resized to 2048 wide (LANCZOS) and back up: resampling
           alone, without SD's JPEG re-encode. Exploratory, outside the rule.

Subcommands, in order (only `run` needs a GPU; only `sample`'s inputs, `fetch-sd` and
`search-check` touch the network):

    sample        draw the sample from the run's results.jsonl -> sample.csv + ids.txt
    search-check  /search items vs /pictures items build the same pano block (20 ids)
    fetch-sd      download the sample's SD assets (1/s) -> <data>/sd/<id>.jpg + sd_fetch.csv
    run           detect on every arm -> detections.jsonl (resumable)
    score         gate + metrics + rule -> gate.csv, per_pano.csv, per_rig.csv, summary.csv,
                  verdict.json
    figures       redraw docs/figures/panoramax-asset/ from the committed CSVs only
    contact-sheet the seeded SD/HD disagreement sheet (needs the local pixels)

Committed outputs live in runs/_summary/panoramax_asset/; pixels never leave <data>.
The matcher and the rule are torch-free and tested (tests/test_panoramax_politeness.py).
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from detectors import (DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE,  # noqa: E402
                       BENCHMARK_CONFIDENCE, on_camera_rig)

OUT = REPO / 'runs' / '_summary' / 'panoramax_asset'
FIG = REPO / 'docs' / 'figures' / 'panoramax-asset'

SEED = 161
N_DETECTION = 500        # panos with >= 1 stored detection at 0.30 (rig-masked)
N_EMPTY = 100            # panos with none
MIN_PER_RIG = 30         # detection-bearing panos per rig that has this many
TIERS = (OPERATIONAL_CONFIDENCE, BENCHMARK_CONFIDENCE)
ARMS = ('hd', 'sd', 'sd_sim')
TARGET = (4096, 2048)
SIM_WIDTH = 2048         # the SD derivative's fixed width

# RampNet's matcher geometry, copied from scripts/agree_rate.py (no import: that module
# pulls shapely and its own data model): x scaled by 1024, y by 512, x cyclic at the seam,
# strictly within 0.022 normalized-x units (~22.5 heatmap px).
SCALE_X, SCALE_Y = 1024, 512
RADIUS = 0.022

# Pre-registered rule (docs/panoramax-asset-study.md, committed before scoring).
POOLED_BAR = 0.95
RIG_BAR = 0.90
RIG_MIN_PANOS = 30
BOOTSTRAP = 2000

# Sanity gate (pre-registered): the hd arm must reproduce the stored run.
GATE_PANO_SHARE = 0.99   # panos whose >= 0.30 peak set is identical (straddlers aside)
GATE_STRADDLE = 0.005    # a peak this close to 0.30 in either file may cross it
GATE_MAX_DCONF = 0.01    # max |confidence difference| over identically placed peaks


# ============================================================================ shared

def rig_of(pano):
    make = pano.get('camera_make') or 'unknown'
    model = pano.get('camera_model') or 'unknown'
    return f'{make} {model}'


def masked(dets, tier):
    """(x, y, c) at or above `tier`, rig-masked as production ships (detectors.on_camera_rig)."""
    return [d for d in dets if d[2] >= tier and not on_camera_rig(d[1])]


def pano_distance(ax, ay, bx, by):
    """Normalized-x distance with RampNet's geometry (seam wrap; see agree_rate.py)."""
    dx = abs(ax - bx) % 1.0
    dx = min(dx, 1.0 - dx) * SCALE_X
    dy = (ay - by) * SCALE_Y
    return math.hypot(dx, dy) / SCALE_X


def match(a_pts, b_pts, radius=RADIUS):
    """Greedy ascending-distance one-to-one match -> {a_index: (b_index, distance)}.
    Strictly within radius; ties broken on (distance, a, b), as agree_rate.match_pano."""
    pairs = []
    for i, a in enumerate(a_pts):
        for j, b in enumerate(b_pts):
            d = pano_distance(a[0], a[1], b[0], b[1])
            if d < radius:
                pairs.append((d, i, j))
    pairs.sort()
    out, used = {}, set()
    for d, i, j in pairs:
        if i in out or j in used:
            continue
        out[i] = (j, d)
        used.add(j)
    return out


def offset_class(a, b):
    """Where a matched pair sits on the stride-8 heatmap grid, from the Chebyshev offset in
    1024x512 heatmap px: 'same_cell' (<= 1 px: the 3/4-mod-8 flip), 'adjacent_cell' (7-9 px:
    one coarse cell over), 'other'."""
    dx = abs(a[0] - b[0]) % 1.0
    dx = min(dx, 1.0 - dx) * SCALE_X
    dy = abs(a[1] - b[1]) * SCALE_Y
    cheb = max(round(dx), round(dy))
    if cheb <= 1:
        return 'same_cell'
    if 7 <= cheb <= 9:
        return 'adjacent_cell'
    return 'other'


def compare_pano(hd, arm, tier):
    """Counts for one pano at one tier. `hd`/`arm` are full (x, y, c) lists (storage floor).

    strict: HD >= tier vs arm >= tier, one-to-one -> recall (matched / hd_n) and precision
    (matched / arm_n). lenient: HD >= tier matched one-to-one by any arm peak >= the storage
    floor -> recall only (a peak SD still sees, just below the tier)."""
    h = masked(hd, tier)
    a = masked(arm, tier)
    m = match(h, a)
    a_floor = masked(arm, DETECTION_STORAGE_FLOOR)
    m_len = match(h, a_floor)
    deltas = [a[j][2] - h[i][2] for i, (j, _) in m.items()]
    classes = Counter(offset_class(h[i], a[j]) for i, (j, _) in m.items())
    return {'hd_n': len(h), 'arm_n': len(a), 'matched': len(m), 'matched_lenient': len(m_len),
            'deltas': deltas, 'classes': classes}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def ratio(num, den):
    return num / den if den else float('nan')


def bootstrap_ci(rows, num_key, den_key, n_boot=BOOTSTRAP, seed=SEED):
    """Percentile 95% CI of sum(num)/sum(den) under a pano-level (cluster) bootstrap:
    detections on one pano are not independent, so panos are what gets resampled."""
    if not rows:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    nums = [r[num_key] for r in rows]
    dens = [r[den_key] for r in rows]
    n = len(rows)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(dens[i] for i in idx)
        if d:
            stats.append(sum(nums[i] for i in idx) / d)
    if not stats:
        return (float('nan'), float('nan'))
    stats.sort()
    return (stats[int(0.025 * (len(stats) - 1))], stats[int(0.975 * (len(stats) - 1))])


def apply_rule(pooled, rigs):
    """The pre-registered default rule. `pooled` = {'recall', 'precision'} (point
    estimates of strict recall-of-HD and precision-vs-HD at 0.30, sd arm); `rigs` =
    [{'rig', 'panos', 'recall_hi', 'precision_hi'}] with 95% upper bounds. Returns
    (default, reasons): 'sd' iff pooled both >= POOLED_BAR and no rig with >= RIG_MIN_PANOS
    sampled panos has an upper bound below RIG_BAR on either."""
    reasons = []
    for key in ('recall', 'precision'):
        if not pooled[key] >= POOLED_BAR:
            reasons.append(f'pooled {key} {pooled[key]:.3f} < {POOLED_BAR}')
    for r in rigs:
        if r['panos'] < RIG_MIN_PANOS:
            continue
        for key in ('recall', 'precision'):
            hi = r[f'{key}_hi']
            if hi < RIG_BAR:   # NaN (no detections) never fails a rig
                reasons.append(f"{r['rig']}: {key} upper bound {hi:.3f} < {RIG_BAR}")
    return ('sd' if not reasons else 'hd'), reasons


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bufsize), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})
    print(f'wrote {path} ({len(rows)} rows)')


def _fmt(v):
    if isinstance(v, float):
        return '' if math.isnan(v) else f'{v:.6g}'
    return '' if v is None else v


def stored_dets(record):
    return [(d['x_normalized'], d['y_normalized'], d['confidence'])
            for d in record['detections']]


# ============================================================================ sample

def allocate(pools, total, minimum):
    """Seats per stratum: `minimum` for every stratum with that many available, the rest
    proportional to pool size by largest remainder, never above a pool's size."""
    alloc = {k: min(minimum, n) if n >= minimum else 0 for k, n in pools.items()}
    left = total - sum(alloc.values())
    while left > 0:
        room = {k: pools[k] - alloc[k] for k in pools if pools[k] > alloc[k]}
        if not room:
            break
        base = sum(room.values())
        quota = {k: left * room[k] / base for k in room}
        add = {k: min(room[k], int(quota[k])) for k in room}
        if sum(add.values()) == 0:   # remainders only: one seat to the largest
            order = sorted(room, key=lambda k: (-(quota[k] - int(quota[k])), -room[k], k))
            add = {order[0]: 1}
        for k, v in add.items():
            alloc[k] += v
        left = total - sum(alloc.values())
    return alloc


def cmd_sample(args):
    results = Path(args.results)
    digest = sha256_file(results)
    print(f'{results}: sha256 {digest}' + ('' if digest.startswith('60e688b2') else
                                           '  (NOT the 5 m Lyon file 60e688b2...)'))
    det_pool, empty_pool = defaultdict(list), defaultdict(list)
    info = {}
    with open(results, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            p = r['pano']
            rig = rig_of(p)
            dets = stored_dets(r)
            sensor = ((p.get('source_metadata') or {}).get('pers:interior_orientation')
                      or {}).get('sensor_array_dimensions') or []
            info[p['panorama_id']] = (rig, p['width'], p['height'], sensor,
                                      len(masked(dets, OPERATIONAL_CONFIDENCE)),
                                      p.get('copyright'), p.get('license'))
            (det_pool if masked(dets, OPERATIONAL_CONFIDENCE) else empty_pool)[rig].append(
                p['panorama_id'])
    rng = random.Random(SEED)
    chosen = []
    for stratum, pools, total, minimum in (('detection', det_pool, N_DETECTION, MIN_PER_RIG),
                                           ('empty', empty_pool, N_EMPTY, 0)):
        alloc = allocate({k: len(v) for k, v in pools.items()}, total, minimum)
        for rig in sorted(pools):
            ids = sorted(pools[rig])
            rng.shuffle(ids)
            chosen += [(pid, stratum) for pid in ids[:alloc[rig]]]
    rows = []
    for pid, stratum in chosen:
        rig, w, h, sensor, n30, producer, lic = info[pid]
        rows.append({'panorama_id': pid, 'rig': rig, 'stratum': stratum, 'record_width': w,
                     'record_height': h,
                     'sensor_width': sensor[0] if len(sensor) == 2 else '',
                     'sensor_height': sensor[1] if len(sensor) == 2 else '',
                     'stored_n_030': n30, 'producer': producer, 'license': lic})
    rows.sort(key=lambda r: r['panorama_id'])
    write_csv(OUT / 'sample.csv', rows, list(rows[0]))
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    (data / 'ids.txt').write_text(''.join(f"{r['panorama_id']}.jpg\n" for r in rows),
                                  encoding='utf-8', newline='\n')
    wanted = {r['panorama_id'] for r in rows}
    with open(results, encoding='utf-8') as f, \
            open(data / 'stored.jsonl', 'w', encoding='utf-8', newline='\n') as out:
        for line in f:
            if json.loads(line)['pano']['panorama_id'] in wanted:
                out.write(line)
    pools = {'detection': {k: len(v) for k, v in det_pool.items()},
             'empty': {k: len(v) for k, v in empty_pool.items()}}
    (OUT / 'sample_meta.json').write_text(json.dumps(
        {'results_sha256': digest, 'records': len(info), 'seed': SEED,
         'n_detection': N_DETECTION, 'n_empty': N_EMPTY, 'min_per_rig': MIN_PER_RIG,
         'pools': pools, 'drawn': dict(sorted(Counter(f"{r['stratum']}|{r['rig']}"
                                                      for r in rows).items()))},
        indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    print(Counter((r['stratum'], r['rig']) for r in rows))


# ============================================================================ network

def cmd_search_check(args):
    """Do /search items build the same pano block as /pictures items? (the batching
    precondition). Fetches args.n sampled ids both ways, through the source's limiter."""
    from sources import panoramax
    rows = read_csv(OUT / 'sample.csv')
    ids = sorted(r['panorama_id'] for r in rows)
    random.Random(SEED).shuffle(ids)
    ids = ids[:args.n]
    found = panoramax.fetch_items(ids) or {}
    out = []
    for pid in ids:
        single, gone = panoramax._fetch_item_by_id(pid)
        batch = found.get(pid)
        row = {'panorama_id': pid, 'in_search': batch is not None,
               'pictures_ok': single is not None, 'identical_block': '', 'differing_keys': ''}
        if batch is not None and single is not None:
            a = panoramax.build_pano_record(pid, 0, 0, batch, 1, 1)
            b = panoramax.build_pano_record(pid, 0, 0, single, 1, 1)
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            diff += sorted(f'source_metadata.{k}' for k in
                           set(a['source_metadata']) | set(b['source_metadata'])
                           if a['source_metadata'].get(k) != b['source_metadata'].get(k))
            row['identical_block'] = a == b
            row['differing_keys'] = ';'.join(diff)
        out.append(row)
    write_csv(OUT / 'search_check.csv', out, list(out[0]))
    same = sum(1 for r in out if r['identical_block'] is True)
    print(f'{same}/{len(out)} identical pano blocks; {sum(r["in_search"] for r in out)} in '
          f'the one /search answer')


def cmd_fetch_sd(args):
    """SD assets for the sample, at the source's default pacing (1 picture/s, 1 metadata/s):
    item metadata by /search in chunks of 100, then one SD GET per picture. Resumable."""
    from sources import panoramax
    from sources.mapillary import body_is_complete
    rows = read_csv(OUT / 'sample.csv')
    sd_dir = Path(args.data) / 'sd'
    sd_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.data) / 'sd_fetch.csv'
    done = {r['panorama_id']: r for r in read_csv(log_path)} if log_path.exists() else {}
    todo = [r['panorama_id'] for r in rows
            if not (done.get(r['panorama_id'], {}).get('status') == 'ok'
                    and (sd_dir / f"{r['panorama_id']}.jpg").exists())]
    items = {}
    for i in range(0, len(todo), panoramax.SEARCH_CHUNK):
        chunk = todo[i:i + panoramax.SEARCH_CHUNK]
        items.update(panoramax.fetch_items(chunk) or {})
    t0 = time.time()
    from PIL import Image
    for k, pid in enumerate(todo):
        item = items.get(pid)
        if item is None:
            item, _ = panoramax._fetch_item_by_id(pid)
        row = {'panorama_id': pid, 'status': '', 'sd_url': '', 'bytes': '', 'sha256': '',
               'sd_width': '', 'sd_height': '', 'live_sensor': ''}
        if item is None:
            row['status'] = 'no_item'
        else:
            sensor = ((item.get('properties') or {}).get('pers:interior_orientation')
                      or {}).get('sensor_array_dimensions') or []
            row['live_sensor'] = 'x'.join(map(str, sensor))
            url = panoramax.image_url(item, asset='sd')
            row['sd_url'] = url or ''
            if not url:
                row['status'] = 'no_sd_asset'
            else:
                try:
                    resp = panoramax._get(url, 'picture', timeout=180)
                    resp.raise_for_status()
                    if not body_is_complete(resp):
                        raise ValueError('short body')
                    payload = resp.content
                    size = Image.open(BytesIO(payload)).size
                    (sd_dir / f'{pid}.jpg').write_bytes(payload)
                    row.update(status='ok', bytes=len(payload),
                               sha256=hashlib.sha256(payload).hexdigest(),
                               sd_width=size[0], sd_height=size[1])
                except Exception as e:
                    row['status'] = f'error: {e}'[:200]
        done[pid] = row
        if (k + 1) % 50 == 0:
            print(f'  {k + 1}/{len(todo)} ({(time.time() - t0) / (k + 1):.2f} s/picture)')
    write_csv(log_path, [done[p] for p in sorted(done)],
              ['panorama_id', 'status', 'sd_url', 'bytes', 'sha256', 'sd_width', 'sd_height',
               'live_sensor'])
    print(Counter(r['status'].split(':')[0] for r in done.values()))
    print(json.dumps(panoramax.run_record(), indent=1))


# ============================================================================ run

def load_arm(path, arm):
    """The 4096x2048 RGB image the detector sees for one arm, built exactly as
    sources/panoramax._download_image does (decode, RGB, BILINEAR resize)."""
    from PIL import Image
    image = Image.open(path).convert('RGB')
    native = image.size
    if arm == 'sd_sim':
        h = round(SIM_WIDTH * image.size[1] / image.size[0])
        image = image.resize((SIM_WIDTH, h), Image.LANCZOS)
    if image.size != TARGET:
        image = image.resize(TARGET, Image.BILINEAR)
    return image, native


def cmd_run(args):
    import torch
    from detectors.curb_ramp import CurbRampDetector
    data = Path(args.data)
    rows = read_csv(OUT / 'sample.csv')
    out_path = data / 'detections.jsonl'
    done = set()
    if out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            done = {(json.loads(l)['panorama_id'], json.loads(l)['arm']) for l in f if l.strip()}
    det = CurbRampDetector(batch_size=1)
    print(f"model {det.provenance['model_id']} revision {det.provenance['model_revision']} "
          f"decode {det.decode} border {det.border} device {det.DEVICE}")
    t0, n = time.time(), 0
    with open(out_path, 'a', encoding='utf-8', newline='\n') as out:
        for r in rows:
            pid = r['panorama_id']
            for arm in ARMS:
                if (pid, arm) in done:
                    continue
                src = data / ('sd' if arm == 'sd' else 'hd') / f'{pid}.jpg'
                if not src.exists():
                    continue
                image, native = load_arm(src, arm)
                t = time.time()
                dets = det.detect(image)
                dt = time.time() - t
                out.write(json.dumps({'panorama_id': pid, 'arm': arm,
                                      'native_size': list(native), 'forward_s': round(dt, 3),
                                      'detections': [list(d) for d in dets]}) + '\n')
                out.flush()
                n += 1
                if dt > 10 and n > 3:
                    print(f'WARNING: forward took {dt:.1f} s (paging?)')
                if n % 100 == 0:
                    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
                    print(f'  {n} forwards, {(time.time() - t0) / n:.2f} s each, '
                          f'peak {peak:.0f} MiB')
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    meta = {'model_id': det.provenance['model_id'],
            'model_revision': det.provenance['model_revision'], 'decode': det.decode,
            'border': det.border, 'device': str(det.DEVICE),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'torch': torch.__version__, 'peak_allocated_mib': round(peak),
            'forwards_this_pass': n, 'wall_s_this_pass': round(time.time() - t0)}
    (data / 'run_meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(meta, indent=1))
    det.close()


# ============================================================================ score

def gate_pano(stored, hd):
    """(identical, max_dconf) for the sanity gate: is the hd arm's rig-masked >= 0.30 peak
    set the stored one, peaks within GATE_STRADDLE of 0.30 in either list aside?"""
    s = {(x, y): c for x, y, c in stored}
    h = {(x, y): c for x, y, c in hd}
    def core(d, other):
        return {k for k, c in d.items() if c >= OPERATIONAL_CONFIDENCE
                and not on_camera_rig(k[1])
                and abs(c - OPERATIONAL_CONFIDENCE) > GATE_STRADDLE
                and abs(other.get(k, c) - OPERATIONAL_CONFIDENCE) > GATE_STRADDLE}
    same = core(s, h) == core(h, s)
    common = set(s) & set(h)
    dmax = max((abs(s[k] - h[k]) for k in common if max(s[k], h[k]) >= OPERATIONAL_CONFIDENCE),
               default=0.0)
    return same, dmax


def summarize(rows, arm, tier):
    """Pooled metrics for per-pano rows of one (arm, tier)."""
    hd_n = sum(r['hd_n'] for r in rows)
    arm_n = sum(r['arm_n'] for r in rows)
    m = sum(r['matched'] for r in rows)
    ml = sum(r['matched_lenient'] for r in rows)
    deltas = [d for r in rows for d in r['deltas']]
    cls = Counter()
    for r in rows:
        cls.update(r['classes'])
    rec_ci = bootstrap_ci(rows, 'matched', 'hd_n')
    pre_ci = bootstrap_ci(rows, 'matched', 'arm_n')
    len_ci = bootstrap_ci(rows, 'matched_lenient', 'hd_n')
    q = statistics.quantiles([abs(d) for d in deltas], n=10) if len(deltas) >= 10 else [float('nan')] * 9
    return {'arm': arm, 'tier': tier, 'panos': len(rows), 'hd_detections': hd_n,
            'arm_detections': arm_n, 'matched': m,
            'recall': ratio(m, hd_n), 'recall_lo': rec_ci[0], 'recall_hi': rec_ci[1],
            'precision': ratio(m, arm_n), 'precision_lo': pre_ci[0],
            'precision_hi': pre_ci[1],
            'recall_lenient': ratio(ml, hd_n), 'recall_lenient_lo': len_ci[0],
            'recall_lenient_hi': len_ci[1],
            'dconf_median': statistics.median(deltas) if deltas else float('nan'),
            'abs_dconf_p90': q[8],
            'same_cell_share': ratio(cls['same_cell'], m),
            'adjacent_cell_share': ratio(cls['adjacent_cell'], m),
            'other_offset_share': ratio(cls['other'], m)}


def cmd_score(args):
    data = Path(args.data)
    sample = {r['panorama_id']: r for r in read_csv(OUT / 'sample.csv')}
    stored = {}
    with open(data / 'stored.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            stored[r['pano']['panorama_id']] = stored_dets(r)
    arms = defaultdict(dict)
    native = defaultdict(dict)
    with open(data / 'detections.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            arms[r['panorama_id']][r['arm']] = [tuple(d) for d in r['detections']]
            native[r['panorama_id']][r['arm']] = r['native_size']
    # committed copy of the detections so score/figures replicate without a GPU
    with open(OUT / 'detections.jsonl', 'w', encoding='utf-8', newline='\n') as out:
        for pid in sorted(arms):
            for arm in ARMS:
                if arm in arms[pid]:
                    out.write(json.dumps({'panorama_id': pid, 'arm': arm,
                                          'native_size': native[pid][arm],
                                          'detections': [list(d) for d in arms[pid][arm]]})
                              + '\n')
    complete = [p for p in sorted(sample) if all(a in arms[p] for a in ARMS)]
    missing = sorted(set(sample) - set(complete))

    # sanity gate
    gate_rows = []
    for pid in sorted(sample):
        if 'hd' not in arms[pid]:
            continue
        same, dmax = gate_pano(stored[pid], arms[pid]['hd'])
        gate_rows.append({'panorama_id': pid, 'identical_030': same, 'max_dconf': dmax})
    write_csv(OUT / 'gate.csv', gate_rows, ['panorama_id', 'identical_030', 'max_dconf'])
    share = ratio(sum(r['identical_030'] for r in gate_rows), len(gate_rows))
    dmax = max((r['max_dconf'] for r in gate_rows), default=float('nan'))
    gate_pass = share >= GATE_PANO_SHARE and dmax <= GATE_MAX_DCONF
    print(f'sanity gate: {share:.4f} of {len(gate_rows)} panos identical at 0.30 '
          f'(bar {GATE_PANO_SHARE}), max |dconf| {dmax:.2e} (bar {GATE_MAX_DCONF}) -> '
          f'{"PASS" if gate_pass else "FAIL"}')

    # frame facts
    frame = Counter()
    for pid in complete:
        s = sample[pid]
        hd_native = tuple(native[pid]['hd'])
        sensor = (int(s['sensor_width']), int(s['sensor_height'])) if s['sensor_width'] else None
        frame['panos'] += 1
        frame['sensor_present'] += sensor is not None
        frame['sensor_equals_archive'] += sensor == hd_native
        frame['record_equals_archive'] += (int(s['record_width']), int(s['record_height'])) == hd_native
        sd_native = tuple(native[pid]['sd'])
        frame['sd_width_2048'] += sd_native[0] == 2048
        frame['sd_2to1'] += sd_native[0] == 2 * sd_native[1]

    per_pano, per_rig, summary = [], [], []
    for arm in ('sd', 'sd_sim'):
        for tier in TIERS:
            rows = []
            for pid in complete:
                c = compare_pano(arms[pid]['hd'], arms[pid][arm], tier)
                c.update(panorama_id=pid, rig=sample[pid]['rig'],
                         stratum=sample[pid]['stratum'])
                rows.append(c)
                per_pano.append({'panorama_id': pid, 'rig': c['rig'], 'stratum': c['stratum'],
                                 'arm': arm, 'tier': tier, 'hd_n': c['hd_n'],
                                 'arm_n': c['arm_n'], 'matched': c['matched'],
                                 'matched_lenient': c['matched_lenient'],
                                 'dconf_mean': statistics.mean(c['deltas']) if c['deltas'] else None,
                                 'same_cell': c['classes']['same_cell'],
                                 'adjacent_cell': c['classes']['adjacent_cell'],
                                 'other_offset': c['classes']['other']})
            summary.append(summarize(rows, arm, tier))
            by_rig = defaultdict(list)
            for c in rows:
                by_rig[c['rig']].append(c)
            for rig in sorted(by_rig):
                s = summarize(by_rig[rig], arm, tier)
                s['rig'] = rig
                per_rig.append(s)
    fields = list(summary[0])
    write_csv(OUT / 'summary.csv', summary, fields)
    write_csv(OUT / 'per_rig.csv', per_rig, ['rig'] + fields)
    write_csv(OUT / 'per_pano.csv', per_pano, list(per_pano[0]))

    pooled = next(s for s in summary if s['arm'] == 'sd' and s['tier'] == OPERATIONAL_CONFIDENCE)
    rigs = [s for s in per_rig if s['arm'] == 'sd' and s['tier'] == OPERATIONAL_CONFIDENCE]
    default, reasons = apply_rule(pooled, rigs)
    if not gate_pass:
        default, reasons = 'hd', ['sanity gate failed: the hd arm is not production']
    verdict = {'gate_pass': gate_pass, 'gate_identical_share': share, 'gate_max_dconf': dmax,
               'panos_scored': len(complete), 'panos_missing_an_arm': missing,
               'frame': dict(frame), 'pooled_recall_030': pooled['recall'],
               'pooled_recall_030_lo': pooled['recall_lo'],
               'pooled_precision_030': pooled['precision'],
               'pooled_precision_030_lo': pooled['precision_lo'],
               'default': default, 'reasons': reasons}
    (OUT / 'verdict.json').write_text(json.dumps(verdict, indent=2, sort_keys=True) + '\n',
                                      encoding='utf-8', newline='\n')
    print(json.dumps(verdict, indent=1))


# ============================================================================ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--results', required=True, help="the Lyon run's results.jsonl")
    s.add_argument('--data', required=True, help='local scratch dir (pixels live here)')
    s = sub.add_parser('search-check')
    s.add_argument('--n', type=int, default=20)
    for name in ('fetch-sd', 'run', 'score', 'contact-sheet'):
        s = sub.add_parser(name)
        s.add_argument('--data', required=True)
    sub.add_parser('figures')
    args = ap.parse_args(argv)
    if args.cmd == 'figures':
        import panoramax_asset_figures
        return panoramax_asset_figures.main()
    if args.cmd == 'contact-sheet':
        import panoramax_asset_figures
        return panoramax_asset_figures.contact_sheet(Path(args.data))
    {'sample': cmd_sample, 'search-check': cmd_search_check, 'fetch-sd': cmd_fetch_sd,
     'run': cmd_run, 'score': cmd_score}[args.cmd](args)


if __name__ == '__main__':
    main()
