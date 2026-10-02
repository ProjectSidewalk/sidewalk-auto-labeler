"""The sub-cell decode in this repo's detector path, measured (issue #111, decode half).

RampNet#221 measured a sub-cell ("gaussian") decode of RampNet's heatmap peaks against human
box centres and shipped it in RampNet. ``detectors.decode`` carries the same rule here as an
opt-in. This script measures what it does through THIS repo's detector, in three steps:

    # 1. GPU (makelab2 A40). ONE forward pass per pano, BOTH decodes from that one heatmap
    #    (never two inference runs: near-tied coarse cells flip between passes). Writes one
    #    JSONL line per pano: the labeler's detections under argmax and gaussian (storage floor
    #    0.1, top 50, exclude_border as production), RampNet's own extraction under both
    #    (rampnet_subcell.detect_peaks at 0.30, clip, exclude_border=False -- the #221
    #    protocol), and the recovered 64x128 coarse map's sha256 (map saved with --coarse-dir).
    python scripts/subcell_decode.py detect --panos-dir <bundle>/panos --ids ids.txt \\
        --out decode_<split>.jsonl --coarse-dir coarse/<split>
    #    ...for an archived run, --results gives the ids AND the pano blocks, and
    #    --source-resize hands the detector a 4096x2048 image exactly as sources/ do:
    python scripts/subcell_decode.py detect --panos-dir <archive>/laurens/panos \\
        --results <archive>/laurens/results.jsonl --source-resize --out decode_laurens.jsonl

The CPU steps (``residual``, ``sigma``, ``world``, ``figures``) read those files; see their
own help.
"""
import argparse
import hashlib
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detectors import decode as dec  # noqa: E402
from detectors import rampnet_subcell as sc  # noqa: E402

#: RampNet#221's extraction floor (the #79 recommended operating point), used for the
#: reproduction of its residual table.
RN_FLOOR = 0.30
#: What every source module hands the detector (panorama.TARGET_SIZE, sources/mapillary.py).
SOURCE_SIZE = (4096, 2048)


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def results_ids(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(str(json.loads(line)['pano']['panorama_id']))
    return out


def _rn(heatmap, decode):
    """RampNet#221's extraction of one heatmap: (row, col, score) rows at 0.30."""
    rcs = sc.detect_peaks(heatmap, RN_FLOOR, decode=decode, clip=True)
    return [[float(r), float(c), float(s)] for r, c, s in rcs]


def pano_line(pid, heatmap, image_size, coarse_dir=None):
    """Everything the CPU steps need from one heatmap (see the module docstring)."""
    import numpy as np
    coarse = sc.coarse_from_heatmap(heatmap)
    resid = sc.upsample_residual(heatmap, coarse)
    c32 = coarse.astype(np.float32)
    if coarse_dir is not None:
        np.save(Path(coarse_dir) / f'{pid}_coarse.npy', c32)
    both = dec.detections_both(heatmap)
    return {'pano_id': pid, 'image_size': list(image_size),
            'upsample_residual': float(f'{resid:.3e}'),
            'coarse_sha256': hashlib.sha256(c32.tobytes()).hexdigest(),
            'argmax': [list(d) for d in both[dec.ARGMAX]],
            'gaussian': [list(d) for d in both[dec.GAUSSIAN]],
            'rn_argmax': _rn(heatmap, dec.ARGMAX),
            'rn_gaussian': _rn(heatmap, dec.GAUSSIAN)}


def cmd_detect(args):
    from PIL import Image
    from detectors.curb_ramp import CurbRampDetector
    import floor_infer_archive as fia
    import heatmap_grid as hg
    Image.MAX_IMAGE_PIXELS = None
    ids = read_ids(args.ids) if args.ids else results_ids(args.results)
    if args.limit:
        ids = ids[:args.limit]
    done = set()
    if args.out.exists():
        with open(args.out, encoding='utf-8') as f:
            done = {json.loads(line)['pano_id'] for line in f if line.strip()}
    todo = [p for p in ids if p not in done]
    if args.coarse_dir:
        args.coarse_dir.mkdir(parents=True, exist_ok=True)
    det = CurbRampDetector()          # decode is applied below, both ways, per heatmap

    def work(pid):
        path = args.panos_dir / f'{pid}.jpg'
        img = Image.open(path).convert('RGB')
        size = img.size
        if args.source_resize and img.size != SOURCE_SIZE:
            img = img.resize(SOURCE_SIZE, Image.BILINEAR)
        if args.perturb:
            img = hg._resample(img)     # PR #120's store-JPEG stand-in: 0.75x + JPEG q90
        h = det.heatmap(img)
        return pano_line(pid, h, size, args.coarse_dir)

    started = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    t0 = time.perf_counter()
    n = 0
    print(f'{len(ids)} ids, {len(done)} already in {args.out}, {len(todo)} to run', flush=True)
    with open(args.out, 'a', encoding='utf-8', newline='\n') as f, \
            ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(work, todo):
            f.write(json.dumps(line) + '\n')
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f'  {n}/{len(todo)}  {time.perf_counter() - t0:.0f} s', flush=True)
    wall = time.perf_counter() - t0
    meta_path = args.out.with_name(args.out.name + '.meta.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {
        'passes': []}
    meta.update({'model': det.provenance, 'software': fia.software_versions(),
                 'rampnet_subcell_commit': dec.RAMPNET_SUBCELL_COMMIT,
                 'panos_dir': str(args.panos_dir),
                 'ids_from': str(args.ids or args.results),
                 'source_resize': list(SOURCE_SIZE) if args.source_resize else None,
                 'perturb': ({'scale': hg.RESAMPLE_SCALE, 'jpeg_quality': 90}
                             if args.perturb else None),
                 'storage_floor': dec.DETECTION_STORAGE_FLOOR,
                 'max_peaks': dec.MAX_PEAKS_PER_PANO, 'rn_floor': RN_FLOOR})
    stats = det.stats()
    meta['passes'].append({'started': started, 'host': socket.getfqdn(), 'panos': n,
                           'wall_s': round(wall, 1),
                           'forward_s': stats.get('forward_seconds'),
                           'workers': args.workers})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n', encoding='utf-8',
                         newline='\n')
    print(f'done: {n} panos in {wall:.0f} s -> {args.out} (+ {meta_path.name})', flush=True)


# --------------------------------------------------------------------------------------- #
# CPU: shared readers and the paired residual protocol (RampNet#221's, ported)
# --------------------------------------------------------------------------------------- #
HM_W, HM_H = 1024, 512
DEG_PER_PX = 360.0 / HM_W          # 0.3516 deg; the same in y (180 / 512)
RADIUS = 0.022                     # the benchmark's match radius, normalized
FLOOR = 0.30                       # RampNet#221's pair floor
SEED = 111
N_REPS = 2000
BOX_SPLITS = ('annapolis', 'paterson', 'richmond', 'sao_paulo')
BUNDLE_SPLITS = ('manual_gold',) + BOX_SPLITS
ND = 4                             # decimals written to committed CSVs
DATA_DIR = REPO_ROOT / 'docs' / 'figures' / 'heatmap-grid' / 'data'
#: RampNet's committed #221 report (analysis_out/subcell_decode_221/results.json at RampNet
#: main 459ea9e): the per-split point estimates this repo's extraction must reproduce.
RAMPNET_RESULTS = Path('analysis_out') / 'subcell_decode_221' / 'results.json'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_decode_file(path):
    """{pano_id: line} from a ``detect`` output."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec['pano_id']] = rec
    return out


def bundle_ids(rampnet_root, split):
    with open(Path(rampnet_root) / 'benchmark' / split / 'records.jsonl', encoding='utf-8') as f:
        return [json.loads(line)['pano']['panorama_id'] for line in f if line.strip()]


def ground_truth(rampnet_root, split):
    """``{pid: [(x, y, scored, w_px, h_px), ...]}`` in normalized coordinates, as RampNet#221
    reads it: manual_gold = YOLO box centres (every point scored); the box splits = reviewer
    boxes with status ``boxed`` (scored) plus ``cant`` entries kept as match decoys (not
    scored). Box sizes are on the 512x1024 heatmap grid (None for a decoy)."""
    root = Path(rampnet_root)
    out = {}
    if split == 'manual_gold':
        ids = set(bundle_ids(root, split))
        for txt in sorted((root / 'manual_labels').glob('*.txt')):
            pid = txt.stem
            if pid not in ids:
                continue
            pts = []
            for line in txt.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    _, cx, cy, w, h = line.split()
                    pts.append((float(cx), float(cy), True, float(w) * HM_W, float(h) * HM_H))
            out[pid] = pts
        return out
    with open(root / 'benchmark' / split / 'boxes.json', encoding='utf-8') as f:
        panos = json.load(f)['panos']
    for pid, entries in panos.items():
        pts = []
        for _, e in sorted(entries.items()):
            if e.get('status') == 'boxed':
                pts.append((e['cx'], e['cy'], True, e['w'] * HM_W, e['h'] * HM_H))
            else:
                pts.append((e['point']['x'], e['point']['y'], False, None, None))
        out[pid] = pts
    return out


def fold(dx, period):
    return (dx + period / 2) % period - period / 2


def greedy_match(pred, gts, radius_sq):
    """RampNet's rampnet.metrics.greedy_match on the 1024x512 grid with x wrapped: each
    prediction, in the order given, claims the nearest unclaimed GT strictly within radius.
    Returns the claimed GT index (or -1) per prediction."""
    claimed = [False] * len(gts)
    out = []
    for px, py in pred:
        best, best_d = -1, radius_sq
        for k, (gx, gy) in enumerate(gts):
            dx = fold((px - gx) * HM_W, HM_W)
            d = dx * dx + ((py - gy) * HM_H) ** 2
            if d < best_d and not claimed[k]:
                best, best_d = k, d
        if best >= 0:
            claimed[best] = True
        out.append(best)
    return out


def peaks_of(rec, path):
    """``[(conf, x_argmax, y_argmax, x_gauss, y_gauss)]`` at >= FLOOR, highest first, for one
    pano. ``path`` 'rampnet' = RampNet#221's extraction (detect_peaks, exclude_border=False),
    'labeler' = this repo's detections_from_heatmap (storage floor, top 50, exclude_border)."""
    if path == 'rampnet':
        # detect_peaks(clip=True) reports the CLIPPED score; RampNet#221 sorted by the raw
        # one, so take the raw score from the labeler's list where it holds the same pixel
        # (all but the 10-px border band), which keeps peaks above 1 in RampNet's order.
        raw = {(round(a[0] * HM_W), round(a[1] * HM_H)): a[2] for a in rec['argmax']}
        rows = [(raw.get((round(a[1]), round(a[0])), a[2]),
                 a[1] / HM_W, a[0] / HM_H, g[1] / HM_W, g[0] / HM_H)
                for a, g in zip(rec['rn_argmax'], rec['rn_gaussian'])]
    else:
        rows = [(a[2], a[0], a[1], g[0], g[1]) for a, g in zip(rec['argmax'], rec['gaussian'])]
    # sorted by score, stable: RampNet#221 sorts its stored peaks the same way
    return sorted((r for r in rows if r[0] >= FLOOR), key=lambda r: -r[0])


def build_pairs(dets, gts):
    """Rows (pid, conf, xa, ya, xg, yg, gx, gy, gw, gh) for scored pairs, matched ONCE on the
    argmax positions (so both decodes are scored on the same pairs)."""
    rsq = (RADIUS * HM_W) ** 2
    rows = []
    for pid in sorted(gts):
        pts = gts[pid]
        peaks = dets.get(pid, [])
        claim = greedy_match([(p[1], p[2]) for p in peaks], [(g[0], g[1]) for g in pts], rsq)
        for p, k in zip(peaks, claim):
            if k >= 0 and pts[k][2]:
                g = pts[k]
                rows.append((pid, p[0], p[1], p[2], p[3], p[4], g[0], g[1], g[3], g[4]))
    return rows


def great_circle_deg(x1, y1, x2, y2):
    import numpy as np
    lon1, lon2 = x1 * 2 * np.pi, x2 * 2 * np.pi
    lat1, lat2 = (0.5 - y1) * np.pi, (0.5 - y2) * np.pi
    s = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def residuals(rows, decode):
    """(dx, dy, gc): GT minus detection in heatmap px (x wrapped) and great-circle degrees."""
    import numpy as np
    k = 4 if decode == dec.GAUSSIAN else 2
    x = np.array([r[k] for r in rows])
    y = np.array([r[k + 1] for r in rows])
    gx = np.array([r[6] for r in rows])
    gy = np.array([r[7] for r in rows])
    dx = fold((gx - x) * HM_W, HM_W)
    dy = (gy - y) * HM_H
    return dx, dy, great_circle_deg(x, y, gx, gy)


def robust_sd(v):
    import numpy as np
    v = np.asarray(v)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def stats(dx, dy, gc):
    import numpy as np
    e = np.hypot(dx, dy)
    return {'mean_px': float(e.mean()), 'median_px': float(np.median(e)),
            'mean_deg': float(gc.mean()), 'median_deg': float(np.median(gc)),
            'bias_x_px': float(dx.mean()), 'bias_y_px': float(dy.mean()),
            'sd_x_px': float(dx.std()), 'sd_y_px': float(dy.std()),
            'robust_sd_x_px': robust_sd(dx), 'robust_sd_y_px': robust_sd(dy)}


def paired_boot(pids, a, b, rng, n_reps=N_REPS):
    """Pano-cluster bootstrap of b - a in mean px, SD x and SD y (RampNet#221's
    paired_boot, without its degree column). a, b = (dx, dy, gc)."""
    import numpy as np
    _, idx = np.unique(pids, return_inverse=True)
    groups = [np.flatnonzero(idx == u) for u in range(idx.max() + 1)]

    def f(ix, r):
        dx, dy, _ = (v[ix] for v in r)
        return np.array([np.hypot(dx, dy).mean(), dx.std(), dy.std()])

    allix = np.arange(len(pids))
    obs = f(allix, b) - f(allix, a)
    draws = np.empty((n_reps, 3))
    for k in range(n_reps):
        pick = rng.integers(0, len(groups), len(groups))
        ix = np.concatenate([groups[p] for p in pick])
        draws[k] = f(ix, b) - f(ix, a)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {nm: (float(o), float(l), float(h))
            for nm, o, l, h in zip(('d_mean_px', 'd_sd_x_px', 'd_sd_y_px'), obs, lo, hi)}


def size_split(rows, decode):
    """The part of the residual that grows with the box (#111: 'what part of sigma is the
    reviewer's box'). Per axis, OLS of the squared residual (bias removed) on the box's
    squared extent along that axis; the intercept is the variance a zero-size box would
    leave, i.e. the part not tied to the ramp's extent (model + point noise). Rows without a
    box size (decoys never pair) are skipped."""
    import numpy as np
    dx, dy, _ = residuals(rows, decode)
    w = np.array([r[8] for r in rows], dtype=float)
    h = np.array([r[9] for r in rows], dtype=float)
    out = {}
    for ax, d, s in (('x', dx, w), ('y', dy, h)):
        r2 = (d - d.mean()) ** 2
        s2 = s ** 2
        slope, icpt = np.polyfit(s2, r2, 1)
        out[f'size_intercept_var_{ax}'] = float(icpt)
        out[f'size_slope_{ax}'] = float(slope)
        out[f'box_extent_{ax}_median_px'] = float(np.median(s))
    return out


def rampnet_published(rampnet_root):
    """{split: {decode: {mean_px, median_px, mean_deg, sd_x_px, sd_y_px}, 'pairs': n}} from
    RampNet's committed #221 report, or {} if it is not under rampnet_root."""
    p = Path(rampnet_root) / RAMPNET_RESULTS
    if not p.exists():
        return {}
    rep = json.loads(p.read_text(encoding='utf-8'))
    out = {}
    for split, s in rep.get('splits', {}).items():
        if 'methods' not in s:
            continue
        out[split] = {'pairs': s.get('pairs'),
                      **{m: s['methods'][m] for m in (dec.ARGMAX, dec.GAUSSIAN)}}
        out[split]['d_mean_px'] = s['vs_argmax'][dec.GAUSSIAN]['d_mean_px']
    return out


def _r(v, nd=ND):
    return None if v is None else round(float(v), nd)


def cmd_residual(args):
    """Per split and pooled, both extraction paths, both decodes: the paired residual to the
    box centres, RampNet's published numbers beside the reproduction, the sub-cell agreement
    between the labeler's gaussian positions and RampNet's on the same peaks, and the
    measured sigma per axis."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    published = rampnet_published(args.rampnet_root)
    out_rows, sigma_rows, agree_rows, inputs = [], [], [], []
    pooled = {'rampnet': {'boxes4': [], 'all5': []}, 'labeler': {'boxes4': [], 'all5': []}}
    for split in BUNDLE_SPLITS:
        path = args.decode_dir / f'decode_{split}.jsonl'
        if not path.exists():
            print(f'{split}: no {path}, skipped', file=sys.stderr)
            continue
        recs = load_decode_file(path)
        inputs.append({'input': f'decode_{split}.jsonl', 'panos': len(recs),
                       'sha256': sha256_file(path)})
        gts = ground_truth(args.rampnet_root, split)
        missing = sorted(set(bundle_ids(args.rampnet_root, split)) - set(recs))
        if missing:
            raise SystemExit(f'{split}: {len(missing)} bundle panos missing from {path}')
        agree_rows.append(agreement(split, recs))
        for xpath in ('rampnet', 'labeler'):
            dets = {pid: peaks_of(r, xpath) for pid, r in recs.items()}
            rows = build_pairs(dets, gts)
            tagged = [(f'{split}/{r[0]}',) + r[1:] for r in rows]
            pooled[xpath]['all5'] += tagged
            if split in BOX_SPLITS:
                pooled[xpath]['boxes4'] += tagged
            out_rows += summarize(split, xpath, rows, rng, published.get(split)
                                  if xpath == 'rampnet' else None)
            sigma_rows += sigma_summary(split, xpath, rows)
    for xpath in ('rampnet', 'labeler'):
        for name, rows in pooled[xpath].items():
            if rows:
                key = f'pooled:{name}'
                out_rows += summarize(key, xpath, rows, rng,
                                      published.get(key) if xpath == 'rampnet' else None)
                sigma_rows += sigma_summary(key, xpath, rows)
    write_rows(args.out / 'decode_residual.csv', out_rows)
    write_rows(args.out / 'decode_sigma.csv', sigma_rows)
    write_rows(args.out / 'decode_agreement.csv', agree_rows)
    write_rows(args.out / 'decode_inputs.csv', inputs)
    for r in out_rows:
        pub = ('' if r.get('published_mean_px') is None else
               f"  (RampNet: {r['published_mean_px']:.3f}, n {r['published_pairs']})")
        print(f"{r['split']:<16} {r['path']:<8} {r['decode']:<8} pairs {r['pairs']:>5}  "
              f"mean {r['mean_px']:.3f} px{pub}"
              + (f"  d {r['d_mean_px']:+.3f} [{r['d_mean_px_lo']:+.3f}, "
                 f"{r['d_mean_px_hi']:+.3f}]" if r['decode'] == dec.GAUSSIAN else ''))


def summarize(split, xpath, rows, rng, published):
    if not rows:
        return []
    import numpy as np
    pids = np.array([r[0] for r in rows])
    res = {m: residuals(rows, m) for m in (dec.ARGMAX, dec.GAUSSIAN)}
    boot = paired_boot(pids, res[dec.ARGMAX], res[dec.GAUSSIAN], rng)
    out = []
    for m in (dec.ARGMAX, dec.GAUSSIAN):
        s = stats(*res[m])
        row = {'split': split, 'path': xpath, 'decode': m, 'pairs': len(rows),
               'panos': len(set(pids)), **{k: _r(v) for k, v in s.items()}}
        if m == dec.GAUSSIAN:
            for k, (o, lo, hi) in boot.items():
                row[k] = _r(o)
                row[f'{k}_lo'] = _r(lo)
                row[f'{k}_hi'] = _r(hi)
        else:
            for k in boot:
                row[k] = row[f'{k}_lo'] = row[f'{k}_hi'] = None
        pub = (published or {}).get(m)
        row['published_pairs'] = (published or {}).get('pairs')
        row['published_mean_px'] = _r(pub['mean_px']) if pub else None
        row['published_sd_x_px'] = _r(pub['sd_x_px']) if pub else None
        row['published_sd_y_px'] = _r(pub['sd_y_px']) if pub else None
        out.append(row)
    return out


def sigma_summary(split, xpath, rows):
    """The measured sigma_peak_px candidates, per axis and decode (module docs): SD and
    robust SD of the residual with the bias removed (upper bounds: they hold the reviewer's
    box and the model-vs-box concept offset too), and the box-size intercept."""
    out = []
    for m in (dec.ARGMAX, dec.GAUSSIAN):
        dx, dy, _ = residuals(rows, m)
        sz = size_split(rows, m)
        sd_x, sd_y = float(dx.std()), float(dy.std())
        out.append({'split': split, 'path': xpath, 'decode': m, 'pairs': len(rows),
                    'sd_x_px': _r(sd_x), 'sd_y_px': _r(sd_y),
                    'sd_rms_px': _r(((sd_x ** 2 + sd_y ** 2) / 2) ** 0.5),
                    'robust_sd_x_px': _r(robust_sd(dx)), 'robust_sd_y_px': _r(robust_sd(dy)),
                    'robust_sd_rms_px': _r(((robust_sd(dx) ** 2 + robust_sd(dy) ** 2) / 2) ** 0.5),
                    **{k: _r(v) for k, v in sz.items()},
                    'size_free_sd_x_px': _r(max(sz['size_intercept_var_x'], 0.0) ** 0.5),
                    'size_free_sd_y_px': _r(max(sz['size_intercept_var_y'], 0.0) ** 0.5)})
    return out


def agreement(split, recs):
    """The labeler's gaussian positions against RampNet's detect_peaks on the same heatmap:
    for every labeler peak >= FLOOR whose argmax pixel RampNet's extraction also returned,
    the largest position difference; plus the peaks only one path returned (the 10-px
    border band the labeler's exclude_border drops, and its top-50 cap)."""
    worst, same, only_lab, only_rn = 0.0, 0, 0, 0
    for r in recs.values():
        rn = {(round(a[1]), round(a[0])): (g[1] / HM_W, g[0] / HM_H)
              for a, g in zip(r['rn_argmax'], r['rn_gaussian']) if a[2] >= FLOOR}
        lab = {(round(a[0] * HM_W), round(a[1] * HM_H)): (g[0], g[1])
               for a, g in zip(r['argmax'], r['gaussian']) if a[2] >= FLOOR}
        for k, (x, y) in lab.items():
            if k in rn:
                same += 1
                rx, ry = rn[k]
                worst = max(worst, abs(fold((x - rx) * HM_W, HM_W)), abs((y - ry) * HM_H))
            else:
                only_lab += 1
        only_rn += sum(1 for k in rn if k not in lab)
    return {'split': split, 'peaks_both_paths': same, 'only_labeler': only_lab,
            'only_rampnet': only_rn, 'max_gaussian_diff_px': float(f'{worst:.3g}')}


def write_rows(path, rows):
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')


# --------------------------------------------------------------------------------------- #
# CPU: world space -- both decodes from the same forward pass, fused and scored
# --------------------------------------------------------------------------------------- #
#: How far (heatmap px, Chebyshev, x wrapped) a re-detected >= 0.55 peak may sit from the
#: judged bundle detection it stands in for: one coarse cell plus the rounding pixel, the
#: provenance gate's coarse-cell rule (docs/heatmap-grid.md section 3). The archive panos
#: are not the exact pixels the run saw (GSV: native resolution, not zoom-3 tiles), so a
#: near-tied peak can flip a cell; anything further is not the detection the reviewer judged.
REKEY_TOL_PX = 9.0
FRAG_M = 5.0
WORLD_HEIGHTS = ('2.6', 'auto')


def arm_files(results_path, decode_path, work_dir):
    """Write the two arms as ordinary results files: the run's own records (pano blocks
    untouched) with the detections replaced by ONE forward pass's argmax and gaussian
    peaks. Records the pass did not cover are left out of both. Returns {decode: path} and
    the count left out."""
    recs = load_decode_file(decode_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {m: work_dir / f'results.{m}.jsonl' for m in (dec.ARGMAX, dec.GAUSSIAN)}
    left_out = 0
    with open(results_path, encoding='utf-8') as f, \
            open(paths[dec.ARGMAX], 'w', encoding='utf-8', newline='\n') as fa, \
            open(paths[dec.GAUSSIAN], 'w', encoding='utf-8', newline='\n') as fg:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            d = recs.get(str(rec['pano']['panorama_id']))
            if d is None:
                left_out += 1
                continue
            for m, fo in ((dec.ARGMAX, fa), (dec.GAUSSIAN, fg)):
                out = dict(rec)
                out['detections'] = [{'x_normalized': x, 'y_normalized': y, 'confidence': c}
                                     for x, y, c in d[m]]
                out.pop(dec.RECORD_DECODE_KEY, None)
                if m != dec.ARGMAX:
                    out[dec.RECORD_DECODE_KEY] = m
                fo.write(json.dumps(out) + '\n')
    return paths, left_out


def rekey_bundle(verdict_panos, bundle_ops, panos_a, panos_g):
    """Re-key a judged bundle to one forward pass's peaks (module docs, REKEY_TOL_PX).

    For each judged pano, the bundle's >= 0.55 detections are paired one-to-one (closest
    first) with the pass's argmax >= 0.55 peaks. The pano is kept only if every one pairs
    and none is left over; its verdicts are then re-ordered onto the pass's peaks, and the
    bundle is re-keyed to the pass's argmax positions (arm A) and to the gaussian positions
    of the SAME peaks (arm G). Same verdicts, same panos, same peaks in both arms.
    Returns (verdicts, ops_a, ops_g, counts)."""
    from detectors import BENCHMARK_CONFIDENCE
    by_a = {p.pano_id: p for p in panos_a}
    by_g = {p.pano_id: p for p in panos_g}
    verdicts, ops_a, ops_g = {}, {}, {}
    counts = {'judged_panos': len(verdict_panos), 'kept': 0, 'missing': 0, 'count_differs': 0,
              'moved_beyond_tol': 0, 'max_shift_px': 0.0}
    for pid, entry in verdict_panos.items():
        pa, pg = by_a.get(pid), by_g.get(pid)
        old = bundle_ops.get(pid)
        if pa is None or pg is None or old is None:
            counts['missing'] += 1
            continue
        new = [(i, x, y, c) for i, x, y, c in pa.detections if c >= BENCHMARK_CONFIDENCE]
        if len(new) != len(old) or len(entry['dets']) != len(old):
            counts['count_differs'] += 1
            continue
        cands = sorted(
            (max(abs(fold((nx - ox) * HM_W, HM_W)), abs((ny - oy) * HM_H)), a, b)
            for a, (ox, oy, _oc) in enumerate(old) for b, (_i, nx, ny, _c) in enumerate(new))
        used_a, used_b, pair = set(), set(), {}
        for dist, a, b in cands:
            if dist > REKEY_TOL_PX:
                break
            if a in used_a or b in used_b:
                continue
            used_a.add(a)
            used_b.add(b)
            pair[b] = (a, dist)
        if len(pair) != len(new):
            counts['moved_beyond_tol'] += 1
            continue
        counts['kept'] += 1
        counts['max_shift_px'] = max([counts['max_shift_px']] + [d for _, d in pair.values()])
        e = dict(entry)
        e['dets'] = [entry['dets'][pair[b][0]] for b in range(len(new))]
        verdicts[pid] = e
        g_by_i = {i: (x, y, c) for i, x, y, c in pg.detections}
        ops_a[pid] = [(x, y, c) for _i, x, y, c in new]
        ops_g[pid] = [g_by_i[i] for i, _x, _y, _c in new]
    return verdicts, ops_a, ops_g, counts


def loo_rows(views, panos_by_id, frame, camera_height):
    """Leave-one-view-out residual of each view: metres to the site solved without it, and
    pixels between the view's detection and that position projected into its pano."""
    import math
    import geo
    import reprojection_residual as rr
    out = []
    for v, ((he, hn), _lam) in zip(views, rr.leave_one_out(views)):
        pose = geo.pano_pose(panos_by_id[v.pano_id].pose_fields())
        proj = rr.project_into(pose, frame, he, hn, camera_height)
        px = None
        if proj is not None:
            dx, dy = rr.pixel_residual(v.x, v.y, proj.x_norm, proj.y_norm)
            px = math.hypot(dx, dy)
        out.append((math.hypot(v.e - he, v.n - hn), px))
    return out


def chi2_per_dof(views):
    import geo
    import reprojection_residual as rr
    pe, pn = rr.solve_views(views)
    chi2 = sum(geo.sym2_quadform(geo.sym2_inv(v.cov), v.e - pe, v.n - pn) for v in views)
    return chi2 / (2 * len(views) - 2)


def site_metrics(sites, panos, frame, params, ramps_pool):
    """GT-free summaries of one fuse, plus the GT fragmentation read."""
    import math
    import numpy as np
    import geo
    import reprojection_residual as rr
    by_id = {p.pano_id: p for p in panos}
    op = [s for s in sites if s.n_operational > 0]
    rpd = [s.residual_per_dof() for s in op if s.n_refit >= 2]
    loo_m, loo_px = [], []
    for s in op:
        views = rr.views_from_site(s).views
        if len(views) >= rr.MIN_VIEWS:
            for m, px in loo_rows(views, by_id, frame, params.camera_height_m):
                loo_m.append(m)
                if px is not None:
                    loo_px.append(px)
    grid = geo.GridIndex(FRAG_M)
    for s in op:
        grid.add(s.e, s.n, s)

    def near(e, n, skip=None):
        return [t for t in grid.near(e, n) if t is not skip
                and math.hypot(t.e - e, t.n - n) < FRAG_M]

    frag = sum(1 for s in op if near(s.e, s.n, s))
    gt_frag = sum(1 for r in ramps_pool if len(near(r.e, r.n)) >= 2)
    pct = lambda v, q: _r(np.percentile(v, q)) if v else None  # noqa: E731
    return {'n_operational_sites': len(op),
            'n_multi_pano_sites': sum(1 for s in op if len(s.pano_ids) >= 2),
            'chi2_dof_median': pct(rpd, 50), 'chi2_dof_mean': _r(np.mean(rpd)) if rpd else None,
            'loo_views': len(loo_m), 'loo_m_median': pct(loo_m, 50), 'loo_m_p90': pct(loo_m, 90),
            'loo_px_median': pct(loo_px, 50), 'loo_px_p90': pct(loo_px, 90),
            'frag5_sites': frag, 'frag5_share': _r(frag / len(op)) if op else None,
            'gt_frag5_ramps': gt_frag}


def frozen_pair(sites_a, panos_a, panos_g, frame, params, rng):
    """THE paired test: association frozen from the argmax fuse, every member re-placed at
    its gaussian position (same pano, same peak), the leave-one-out residual and chi2/dof
    recomputed. Sites whose members all re-project are compared view by view; the CI is a
    site-cluster bootstrap of the difference in means (gaussian minus argmax)."""
    import numpy as np
    import fuse_sites as fs
    import reprojection_residual as rr
    dets_g, frame_g, _ = fs.project(panos_g, params)
    assert (frame_g.lat0, frame_g.lng0) == (frame.lat0, frame.lng0)
    g_by_key = {(d.pano_id, d.det_index): d for d in dets_g}
    by_a = {p.pano_id: p for p in panos_a}
    by_g = {p.pano_id: p for p in panos_g}
    rows, skipped = [], 0
    for s in sites_a:
        if s.n_operational == 0:
            continue
        va = rr.views_from_site(s).views
        if len(va) < rr.MIN_VIEWS:
            continue
        twin = [g_by_key.get((v.pano_id, v.det_index)) for v in va]
        if any(t is None for t in twin):
            skipped += 1
            continue
        vg = [rr.View(d.pano_id, d.det_index, d.x, d.y, d.conf, d.e, d.n, d.cov,
                      d.ground.range_m, d.ground.bearing_deg, d.source, d.capture_date)
              for d in twin]
        la = loo_rows(va, by_a, frame, params.camera_height_m)
        lg = loo_rows(vg, by_g, frame, params.camera_height_m)
        ca, cg = chi2_per_dof(va), chi2_per_dof(vg)
        for (ma, pa), (mg, pg) in zip(la, lg):
            rows.append((s.id, ma, mg, pa, pg, ca, cg))
    if not rows:
        return {'sites': 0, 'views': 0, 'sites_skipped': skipped}
    sid = np.array([r[0] for r in rows])
    cols = {k: np.array([r[i] if r[i] is not None else np.nan for r in rows], dtype=float)
            for i, k in enumerate(('sid', 'm_a', 'm_g', 'px_a', 'px_g', 'chi_a', 'chi_g'))}
    ok = ~np.isnan(cols['px_a']) & ~np.isnan(cols['px_g'])
    _, idx = np.unique(sid, return_inverse=True)
    groups = [np.flatnonzero(idx == u) for u in range(idx.max() + 1)]
    # one chi2/dof per site: take each site's first row
    first = np.array([g[0] for g in groups])

    def stat(ix):
        sites_ix = np.intersect1d(ix, first)
        pix = ix[ok[ix]]
        return np.array([cols['m_g'][ix].mean() - cols['m_a'][ix].mean(),
                         cols['px_g'][pix].mean() - cols['px_a'][pix].mean(),
                         cols['chi_g'][sites_ix].mean() - cols['chi_a'][sites_ix].mean()])

    allix = np.arange(len(rows))
    obs = stat(allix)
    draws = np.empty((N_REPS, 3))
    for k in range(N_REPS):
        pick = rng.integers(0, len(groups), len(groups))
        draws[k] = stat(np.concatenate([groups[p] for p in pick]))
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    out = {'sites': len(groups), 'views': len(rows), 'sites_skipped': skipped,
           'views_projected': int(ok.sum())}
    for name, key in (('loo_m', 'm'), ('loo_px', 'px'), ('chi2_dof', 'chi')):
        sel = first if key == 'chi' else (np.flatnonzero(ok) if key == 'px' else allix)
        a, g = cols[f'{key}_a'][sel], cols[f'{key}_g'][sel]
        out[f'{name}_mean_argmax'] = _r(a.mean())
        out[f'{name}_mean_gaussian'] = _r(g.mean())
        out[f'{name}_median_argmax'] = _r(np.median(a))
        out[f'{name}_median_gaussian'] = _r(np.median(g))
        out[f'{name}_share_closer_gaussian'] = _r((g < a).mean())
    for j, name in enumerate(('d_loo_m_mean', 'd_loo_px_mean', 'd_chi2_dof_mean')):
        out[name], out[f'{name}_lo'], out[f'{name}_hi'] = _r(obs[j]), _r(lo[j]), _r(hi[j])
    return out


def cmd_world(args):
    import numpy as np
    import eval_sites as es
    import fuse_sites as fs
    rng = np.random.default_rng(SEED)
    paths, left_out = arm_files(args.results, args.decode_file, args.work_dir)
    depth_index = args.depth_index or (args.results.parent / 'depth' / 'index.csv')
    verdict_panos, bundle_ops = es.load_benchmark(args.split, args.benchmark_root)
    world_rows, pair_rows = [], []
    print(f'{args.city}: {left_out} record(s) of {args.results.name} not in the pass')
    for h in args.heights:
        height = fs.fuse_camera_height_arg(h)
        loaded = {}
        for m, p in paths.items():
            panos, _sk, resolved, auto = fs.load_at_height(
                p, height, depth_index=depth_index if depth_index.exists() else None)
            loaded[m] = (panos, resolved, auto)
        (panos_a, res_a, auto_a), (panos_g, res_g, _) = loaded[dec.ARGMAX], loaded[dec.GAUSSIAN]
        assert res_a == res_g
        verdicts, ops_a, ops_g, rk = rekey_bundle(verdict_panos, bundle_ops, panos_a, panos_g)
        params = fs.FuseParams(min_confidence=dec_benchmark(), mask_rig=False,
                               camera_height_m=res_a, apply_pose=fs.POSE_OFF,
                               sigma_peak_px=args.sigma_peak_px)
        fused = {}
        for m, panos, ops in ((dec.ARGMAX, panos_a, ops_a), (dec.GAUSSIAN, panos_g, ops_g)):
            sites, frame, fstats = fs.fuse(panos, params)
            fused[m] = (sites, frame)
            r = es.evaluate_city(verdicts, ops, panos, params, prefused=(sites, frame, fstats))
            points, _ov, _c, _w = es.build_gt(verdicts, ops, {p.pano_id: p for p in panos},
                                              params, frame)
            pool = [x for x in es.merge_gt_points(points, 2.5) if x.in_pool]
            p, b = r['precision'], r['buckets']
            row = {'city': args.city, 'split': args.split, 'height': h,
                   'height_resolved': str(auto_a.get('resolved')) if auto_a else str(res_a),
                   'decode': m, 'sigma_peak_px': params.sigma_peak_px,
                   'gt_panos_kept': rk['kept'], 'gt_panos_judged': rk['judged_panos'],
                   'n_pool': r['n_pool_ramps'], 'world_recall': _r(r['world_recall']),
                   'recovered_other_view': b['recovered_other_view'],
                   'union_lift': _r(b['recovered_other_view'] / r['n_pool_ramps'])
                   if r['n_pool_ramps'] else None,
                   'tp': p['tp'], 'fp': p['fp'], 'world_precision': _r(p['value']),
                   'n_sites': r['fuse']['n_sites'],
                   'chi2_gate_rejections': r['fuse']['rejections']['chi2_gate'],
                   'residual_rejections': r['fuse']['rejections']['residual'],
                   **site_metrics(sites, panos, frame, params, pool)}
            world_rows.append(row)
            print(f"  h={h:<4} {m:<8} R {row['world_recall']}  P {row['world_precision']}  "
                  f"sites {row['n_operational_sites']}  chi2/dof med {row['chi2_dof_median']}  "
                  f"LOO px med {row['loo_px_median']}  frag5 {row['frag5_share']}", flush=True)
        pr = frozen_pair(fused[dec.ARGMAX][0], panos_a, panos_g, fused[dec.ARGMAX][1],
                         params, rng)
        pair_rows.append({'city': args.city, 'height': h, **pr, **{f'rekey_{k}': _r(v) if
                          isinstance(v, float) else v for k, v in rk.items()},
                          'decode_file_sha256': sha256_file(args.decode_file),
                          'results_sha256': sha256_file(args.results)})
        print(f"  h={h:<4} frozen association: {pr.get('sites')} sites / {pr.get('views')} "
              f"views; d LOO px {pr.get('d_loo_px_mean')} [{pr.get('d_loo_px_mean_lo')}, "
              f"{pr.get('d_loo_px_mean_hi')}]  d chi2/dof {pr.get('d_chi2_dof_mean')} "
              f"[{pr.get('d_chi2_dof_mean_lo')}, {pr.get('d_chi2_dof_mean_hi')}]", flush=True)
    write_rows(args.out / f'decode_world_{args.city}.csv', world_rows)
    write_rows(args.out / f'decode_world_pair_{args.city}.csv', pair_rows)


def dec_benchmark():
    from detectors import BENCHMARK_CONFIDENCE
    return BENCHMARK_CONFIDENCE


# --------------------------------------------------------------------------------------- #
# CPU: PR #120's ten-cell table with a measured sigma_peak_px
# --------------------------------------------------------------------------------------- #
def cmd_sigma_table(args):
    import heatmap_grid as hg
    import fuse_sites as fs
    rows = []
    for city in args.cities:
        for h in args.heights:
            height = fs.fuse_camera_height_arg(h)
            for sigma in [hg.SIGMAS[0]] + args.sigma:
                row = hg.score(city, args.benchmark_root, args.runs_root / city, height, sigma)
                rows.append({k: _r(v) if isinstance(v, float) else v for k, v in row.items()})
                print(f"{city:<12} h={h:<5} sigma={sigma:<5} R {row['world_recall']:.4f}  "
                      f"P {row['world_precision']:.4f}  sites {row['n_sites']:,}  gate-rej "
                      f"{row['chi2_gate_rejections']:,}  resid-rej {row['residual_rejections']:,}",
                      flush=True)
    write_rows(args.out / 'decode_sigma_table.csv', rows)
    verdicts = []
    for sigma in args.sigma:
        adopt, cmp = hg.sigma_verdict(rows, baseline=hg.SIGMAS[0], candidate=_r(sigma))
        for c in cmp:
            verdicts.append({'candidate_sigma_px': sigma,
                             **{k: _r(v) if isinstance(v, float) else v for k, v in c.items()}})
        print(f"sigma {sigma}: {('all ' + str(len(cmp)) + ' cells within the #111 rule') if adopt else 'fails the #111 rule in ' + str(sum(not c['pass'] for c in cmp)) + ' cell(s)'}")
    write_rows(args.out / 'decode_sigma_table_verdict.csv', verdicts)


# --------------------------------------------------------------------------------------- #
# CPU: reproduction under a perturbation, both decodes (the #111 reproduction rules)
# --------------------------------------------------------------------------------------- #
#: Pairing window for one peak seen in two passes: 1.5 coarse cells, Chebyshev, on the
#: argmax positions (wide enough for the 7-8 px flip, narrow enough that neighbouring ramps,
#: which peak_local_max keeps >= 10 px apart, rarely pair across).
PAIR_TOL_PX = 12.0
SHIFT_EDGES = (0.5, 1.5, 2.5, 4.5, 6.5, 8.5, PAIR_TOL_PX + 0.5)


def cheb_px(a, b):
    return max(abs(fold((a[0] - b[0]) * HM_W, HM_W)), abs((a[1] - b[1]) * HM_H))


def stability_pairs(orig, pert, floor=FLOOR):
    """One-to-one pairs of a pano's peaks between two passes (argmax positions, closest
    first, within PAIR_TOL_PX), each with its shift under argmax and under gaussian."""
    a = [(i, d) for i, d in enumerate(orig['argmax']) if d[2] >= floor]
    b = [(j, d) for j, d in enumerate(pert['argmax']) if d[2] >= floor]
    cands = sorted((cheb_px(da, db), i, j) for i, da in a for j, db in b)
    used_i, used_j, out = set(), set(), []
    for dist, i, j in cands:
        if dist > PAIR_TOL_PX:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((dist, cheb_px(orig['gaussian'][i], pert['gaussian'][j]),
                    min(orig['argmax'][i][2], pert['argmax'][j][2])))
    return out, len(a) - len(used_i), len(b) - len(used_j)


def cmd_stability(args):
    """How far the SAME peak moves between a pano and its perturbed copy (0.75x + JPEG q90,
    the store-run stand-in), under each decode. The argmax moves in whole grid steps (0, 1
    or 7-8 px: a near-tie flip); the gaussian decode should move a flipped peak only by the
    distance between the two cells' sub-cell estimates. The output restates the provenance
    gate's distance classes for a gaussian campaign."""
    import numpy as np
    rows, hist_rows = [], []
    for split in args.splits:
        orig = load_decode_file(args.decode_dir / f'decode_{split}.jsonl')
        pert = load_decode_file(args.decode_dir / f'decode_{split}_perturbed.jsonl')
        pairs, un_o, un_p = [], 0, 0
        for pid in sorted(set(orig) & set(pert)):
            p, uo, up = stability_pairs(orig[pid], pert[pid])
            pairs += p
            un_o += uo
            un_p += up
        a = np.array([p[0] for p in pairs])
        g = np.array([p[1] for p in pairs])
        flip = a >= 6.5
        rows.append({'split': split, 'panos': len(set(orig) & set(pert)), 'pairs': len(pairs),
                     'unpaired_orig': un_o, 'unpaired_perturbed': un_p,
                     'argmax_same_px': int((a < 0.5).sum()),
                     'argmax_flip_7_8': int(flip.sum()),
                     'argmax_flip_share': _r(flip.mean()),
                     'gaussian_median_shift_px': _r(np.median(g)),
                     'gaussian_p90_shift_px': _r(np.percentile(g, 90)),
                     'gaussian_p99_shift_px': _r(np.percentile(g, 99)),
                     'gaussian_max_shift_px': _r(g.max()),
                     'gaussian_shift_ge_6_5_share': _r((g >= 6.5).mean()),
                     'gaussian_median_shift_on_argmax_flips': _r(np.median(g[flip]))
                     if flip.any() else None,
                     'gaussian_p90_shift_on_argmax_flips': _r(np.percentile(g[flip], 90))
                     if flip.any() else None})
        lo = 0.0
        for hi in SHIFT_EDGES:
            hist_rows.append({'split': split, 'shift_lo_px': lo, 'shift_hi_px': hi,
                              'argmax': int(((a >= lo) & (a < hi)).sum()),
                              'gaussian': int(((g >= lo) & (g < hi)).sum())})
            lo = hi
        r = rows[-1]
        print(f"{split:<12} pairs {r['pairs']:>4}  argmax flips {r['argmax_flip_7_8']} "
              f"({r['argmax_flip_share']})  gaussian shift median {r['gaussian_median_shift_px']} "
              f"p99 {r['gaussian_p99_shift_px']} max {r['gaussian_max_shift_px']}; on the flips "
              f"median {r['gaussian_median_shift_on_argmax_flips']}")
    write_rows(args.out / 'decode_stability.csv', rows)
    write_rows(args.out / 'decode_stability_hist.csv', hist_rows)


# --------------------------------------------------------------------------------------- #
# Figures (docs/heatmap-grid.md section 4), from the committed CSVs only: no GPU, no network
# --------------------------------------------------------------------------------------- #
FIG_DIR = REPO_ROOT / 'docs' / 'figures' / 'heatmap-grid'
SPLIT_LABEL = {'manual_gold': 'manual_gold (GSV, independent)', 'annapolis': 'annapolis (Mapillary)',
               'paterson': 'paterson (GSV)', 'richmond': 'richmond (Mapillary)',
               'sao_paulo': 'sao_paulo (GSV)', 'pooled:boxes4': 'pooled, 4 box splits',
               'pooled:all5': 'pooled, all 5'}


def _csv(name):
    import csv
    with open(DATA_DIR / name, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _f(v):
    return None if v in (None, '') else float(v)


def fig_residual(plt):
    """Q: does the gaussian decode, run through THIS repo's detector, land nearer the box
    centres than argmax, and does it reproduce RampNet's numbers?"""
    import heatmap_grid as hg
    rows = _csv('decode_residual.csv')
    splits = [s for s in SPLIT_LABEL if any(r['split'] == s for r in rows)]
    get = {(r['split'], r['path'], r['decode']): r for r in rows}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True,
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    y = list(range(len(splits)))[::-1]
    for yi, s in zip(y, splits):
        a, g = get[(s, 'labeler', 'argmax')], get[(s, 'labeler', 'gaussian')]
        ax1.plot([_f(a['mean_px']), _f(g['mean_px'])], [yi, yi], color=hg.C_GRID, lw=2.5,
                 zorder=1, solid_capstyle='round')
        ax1.scatter(_f(a['mean_px']), yi, s=48, color=hg.C_ORANGE, zorder=3,
                    edgecolor='#fcfcfb', linewidth=1.5)
        ax1.scatter(_f(g['mean_px']), yi, s=48, color=hg.C_BLUE, zorder=3,
                    edgecolor='#fcfcfb', linewidth=1.5)
        ra, rg = get.get((s, 'rampnet', 'argmax')), get.get((s, 'rampnet', 'gaussian'))
        for r, c in ((ra, hg.C_ORANGE), (rg, hg.C_BLUE)):
            if r and r.get('published_mean_px'):
                ax1.scatter(_f(r['published_mean_px']), yi + 0.28, marker='v', s=26,
                            facecolor='none', edgecolor=c, linewidth=1.1, zorder=3)
        d = (_f(g['d_mean_px']), _f(g['d_mean_px_lo']), _f(g['d_mean_px_hi']))
        ax2.plot([d[1], d[2]], [yi, yi], color=hg.C_BLUE, lw=2, solid_capstyle='round')
        ax2.scatter(d[0], yi, s=40, color=hg.C_BLUE, zorder=3, edgecolor='#fcfcfb')
        ax2.text(max(d[2], 0.02) + 0.04, yi, f'{d[0]:+.2f} px  (n {int(g["pairs"]):,})',
                 va='center', fontsize=7.5, color=hg.C_INK2)
    ax1.set_yticks(y)
    ax1.set_yticklabels([SPLIT_LABEL[s] for s in splits], fontsize=8)
    ax1.set_xlabel('mean distance to the box centre (heatmap px; 1 px = 0.35 deg)',
                   color=hg.C_INK2, fontsize=8.5)
    ax1.set_title('Argmax (orange) and gaussian (blue), labeler detector\n'
                  'open triangles: RampNet#221 published values', fontsize=9.5,
                  color=hg.C_INK, loc='left')
    hg._style(ax1, grid_axis='x')
    ax2.axvline(0, color=hg.C_INK2, lw=0.8)
    ax2.set_xlabel('gaussian minus argmax, paired (px), 95% pano-cluster CI',
                   color=hg.C_INK2, fontsize=8.5)
    ax2.set_title('Change per pair', fontsize=9.5, color=hg.C_INK, loc='left')
    lo = min(_f(get[(s, 'labeler', 'gaussian')]['d_mean_px_lo']) for s in splits)
    ax2.set_xlim(min(lo, -1.2) - 0.1, 1.0)
    hg._style(ax2, grid_axis='x')
    fig.suptitle('The sub-cell decode moves detections toward the box centres on all five '
                 'splits; the CI excludes zero on four (same peaks, same scores)', fontsize=11,
                 color=hg.C_INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_stability(plt):
    """Q: under a small input change, does the same peak keep its position? (the
    reproduction rule for a gaussian campaign)"""
    import heatmap_grid as hg
    rows = _csv('decode_stability_hist.csv')
    bins = []
    for r in rows:
        k = (_f(r['shift_lo_px']), _f(r['shift_hi_px']))
        if k not in bins:
            bins.append(k)
    tot = {m: sum(int(r[m]) for r in rows) for m in ('argmax', 'gaussian')}
    share = {m: [sum(int(r[m]) for r in rows if (_f(r['shift_lo_px']), _f(r['shift_hi_px'])) == b)
                 / tot[m] for b in bins] for m in ('argmax', 'gaussian')}
    labels = [f'{lo:g}-{hi:g}' if lo else f'<{hi:g}' for lo, hi in bins]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = range(len(bins))
    wdt = 0.38
    for off, m, c in ((-wdt / 2 - 0.01, 'argmax', hg.C_ORANGE), (wdt / 2 + 0.01, 'gaussian', hg.C_BLUE)):
        ax.bar([i + off for i in x], [100 * v for v in share[m]], width=wdt, color=c,
               label=f'{m} (n {tot[m]:,} paired peaks)')
        for i, v in zip(x, share[m]):
            if v >= 0.005:
                ax.text(i + off, 100 * v + 1, f'{100 * v:.0f}', ha='center', fontsize=7,
                        color=hg.C_INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('how far the same peak moved, original vs 0.75x + JPEG q90 copy '
                  '(heatmap px, Chebyshev)', color=hg.C_INK2, fontsize=8.5)
    ax.set_ylabel('% of paired peaks', color=hg.C_INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    hg._style(ax)
    fig.suptitle('Argmax moves in whole grid steps (0, 1 or a 7-8 px flip); '
                 'the gaussian position moves continuously', fontsize=10.5, color=hg.C_INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_world(plt):
    """Q: does the decode tighten multi-view fusion? Frozen association, paired by view."""
    import heatmap_grid as hg
    rows = []
    for p in sorted(DATA_DIR.glob('decode_world_pair_*.csv')):
        rows += _csv(p.name)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.3), sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, key, base_key, title in (
            (axes[0], 'd_loo_m_mean', 'loo_m_mean_argmax', 'leave-one-out residual (m)'),
            (axes[1], 'd_loo_px_mean', 'loo_px_mean_argmax', 'leave-one-out residual (px)'),
            (axes[2], 'd_chi2_dof_mean', 'chi2_dof_mean_argmax', 'site chi2 / dof')):
        for yi, r in zip(y, rows):
            o, lo, hi = _f(r[key]), _f(r[f'{key}_lo']), _f(r[f'{key}_hi'])
            ax.plot([lo, hi], [yi, yi], color=hg.C_BLUE, lw=2, solid_capstyle='round')
            ax.scatter(o, yi, s=40, color=hg.C_BLUE, zorder=3, edgecolor='#fcfcfb')
            ax.text(o, yi - 0.3, f'{o:+.3f} (argmax mean {_f(r[base_key]):.2f})', fontsize=7,
                    color=hg.C_INK2, ha='center')
        ax.axvline(0, color=hg.C_INK2, lw=0.8)
        ax.set_ylim(-0.7, len(rows) - 0.5)
        ax.set_title(title, fontsize=9.5, color=hg.C_INK, loc='left')
        ax.set_xlabel('gaussian minus argmax, 95% site-cluster CI', color=hg.C_INK2,
                      fontsize=8.5)
        hg._style(ax, grid_axis='x')
    axes[0].set_yticks(y)
    city = {'laurens': 'laurens (Mapillary)', 'laurens_gsv': 'laurens_gsv (GSV)'}
    axes[0].set_yticklabels([f"{city.get(r['city'], r['city'])} @ {r['height']}{'' if r['height'] == 'auto' else ' m'}\n({int(r['sites'])} sites, "
                             f"{int(r['views'])} views)" for r in rows], fontsize=8)
    fig.suptitle('Frozen association: each member re-placed at its gaussian position (same '
                 'forward pass). Left of zero = views agree better', fontsize=10.5,
                 color=hg.C_INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def cmd_figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                         'svg.hashsalt': 'decode111'})
    for name, fn, need in (('decode_residual', fig_residual, 'decode_residual.csv'),
                           ('decode_stability', fig_stability, 'decode_stability_hist.csv'),
                           ('decode_world', fig_world, 'decode_world_pair_laurens_gsv.csv')):
        if need and not (DATA_DIR / need).exists():
            print(f'no data/{need}: skipped {name}', file=sys.stderr)
            continue
        fig = fn(plt)
        fig.savefig(FIG_DIR / f'{name}.png', dpi=args.dpi, metadata={'Software': None})
        svg = FIG_DIR / f'{name}.svg'
        fig.savefig(svg, metadata={'Date': None, 'Creator': None})
        svg.write_bytes(svg.read_bytes().replace(b'\r\n', b'\n'))   # LF on every platform
        plt.close(fig)
        print(f'wrote {FIG_DIR / name}.png/.svg')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('detect', help='GPU: one forward pass per pano, both decodes')
    p.add_argument('--panos-dir', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--ids', type=Path, help='one pano id per line')
    g.add_argument('--results', type=Path, help="a run's results.jsonl (ids in its order)")
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--coarse-dir', type=Path, default=None)
    p.add_argument('--source-resize', action='store_true',
                   help='resize to 4096x2048 (PIL bilinear) first, as sources/ do')
    p.add_argument('--perturb', action='store_true',
                   help="resample each image 0.75x + JPEG q90 first (heatmap_grid._resample, "
                        "PR #120's perturbation): the stability arm")
    p.add_argument('--workers', type=int, default=4, help='image-decode threads')
    p.add_argument('--limit', type=int, default=0)

    rampnet_default = REPO_ROOT.parent / 'RampNet'
    p = sub.add_parser('residual', help='CPU: paired residual to box centres, both paths')
    p.add_argument('--decode-dir', type=Path, required=True,
                   help='directory holding decode_<split>.jsonl from `detect`')
    p.add_argument('--rampnet-root', type=Path, default=rampnet_default,
                   help='RampNet at main: benchmark/, manual_labels/ and '
                        'analysis_out/subcell_decode_221/results.json')
    p.add_argument('--out', type=Path, default=DATA_DIR)

    p = sub.add_parser('world', help='CPU: both decodes of one pass, fused and scored')
    p.add_argument('city', help='name used in the output file (e.g. laurens_gsv)')
    p.add_argument('--split', required=True, help='RampNet benchmark split')
    p.add_argument('--results', type=Path, required=True,
                   help="the run's results.jsonl (pano blocks)")
    p.add_argument('--decode-file', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True,
                   help='where the two arm results files are written (not committed)')
    p.add_argument('--depth-index', type=Path, default=None)
    p.add_argument('--benchmark-root', type=Path, default=rampnet_default / 'benchmark')
    p.add_argument('--heights', nargs='+', default=list(WORLD_HEIGHTS))
    p.add_argument('--sigma-peak-px', type=float, default=None)
    p.add_argument('--out', type=Path, default=DATA_DIR)

    p = sub.add_parser('sigma-table', help="CPU: PR #120's ten cells at a measured sigma")
    p.add_argument('cities', nargs='+')
    p.add_argument('--sigma', type=float, nargs='+', required=True)
    p.add_argument('--runs-root', type=Path, default=REPO_ROOT / 'runs')
    p.add_argument('--benchmark-root', type=Path, default=rampnet_default / 'benchmark')
    p.add_argument('--heights', nargs='+', default=['2.6', 'auto'])
    p.add_argument('--out', type=Path, default=DATA_DIR)

    p = sub.add_parser('stability', help='CPU: the same peak in a pano and its perturbed '
                                         'copy, both decodes')
    p.add_argument('splits', nargs='+')
    p.add_argument('--decode-dir', type=Path, required=True,
                   help='holding decode_<split>.jsonl and decode_<split>_perturbed.jsonl')
    p.add_argument('--out', type=Path, default=DATA_DIR)

    p = sub.add_parser('figures', help='redraw the decode figures from the committed CSVs')
    p.add_argument('--dpi', type=int, default=200)

    args = ap.parse_args(argv)
    {'detect': cmd_detect, 'residual': cmd_residual, 'world': cmd_world,
     'sigma-table': cmd_sigma_table, 'stability': cmd_stability,
     'figures': cmd_figures}[args.cmd](args)


if __name__ == '__main__':
    main()
