"""What the seam band costs (issue #130): exclude vs keep from the SAME heatmaps, CPU only.

The peak finder's default border rule (`exclude`, skimage's exclude_border=True) drops every
peak within 10 heatmap px of the 512x1024 heatmap's edge; the left and right edges are the
360-degree seam. `keep` (RampNet's rule) returns them. This script measures, on both Laurens
arms (Mapillary `laurens`, GSV `laurens_gsv`), what `keep` adds, from the 64x128 coarse maps
the #111 detect pass saved for every pano (the model's head output is exactly their bilinear
x8 upsample, so a coarse map IS the heatmap). No GPU, no network.

Steps, in order (docs/seam-band-130.md has the exact commands and results):

    # makelab2 (where the coarse maps are): the instrument check, then both rules per pano
    python scripts/seam_band_130.py check laurens --results <run>/results.jsonl \\
        --coarse-dir /homes/gws/jonf/decode111/coarse/laurens \\
        --decode-file /homes/gws/jonf/decode111/decode_laurens.jsonl
    python scripts/seam_band_130.py peaks laurens --results <run>/results.jsonl \\
        --coarse-dir /homes/gws/jonf/decode111/coarse/laurens
    # anywhere with the run's results.jsonl and RampNet's benchmark bundle
    python scripts/seam_band_130.py world laurens --results runs/laurens/results.jsonl \\
        --peaks runs/laurens/seam_band_130/peaks.jsonl --benchmark-root ../RampNet/benchmark
    python scripts/seam_band_130.py summary        # both arms + pooled -> summary.json
    python scripts/seam_band_130.py verify         # re-derive summary.json, byte for byte

- ``check``: rebuild each heatmap from its coarse map, run the production extractor
  (``exclude``, argmax) and compare with the run's stored detections at the storage floor,
  on exact pixel keys (round(x*1024), round(y*512)) and scores to 1e-6. Writes
  ``<arm>_check.csv`` (committed: pano_id, coarse_sha256, reproduces, ...). Only panos that
  reproduce are counted by the later steps.
- ``peaks``: both rules, both decodes, from each reproducing pano's heatmap. "Gained" = a
  peak present under keep and absent under exclude at the same pixel key (argmax; the
  gaussian row is the same peak's sub-cell position). Writes ``peaks.jsonl`` (untracked),
  ``<arm>_gained.csv`` and ``<arm>_counts.json`` (committed).
- ``world``: the run's own records, with every gained peak appended to the pano's stored
  detections (so the two files differ ONLY by the band), fused with fuse_sites under
  identical parameters; each gained peak >= 0.30 is then a lost VIEW (it joined a site that
  was already operational), a PROMOTION (its site existed only as sub-threshold support) or
  a NEW site, and on panos of the RampNet bundle it is matched to the reviewer's marks.
  Writes ``<arm>_world.csv`` and ``<arm>_world.json`` (committed).
- ``summary`` / ``verify``: every table in the doc, from the committed files alone.

Every committed file is written with newline='' and rounded floats, so it is
byte-reproducible.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / 'scripts', REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from detectors import (BORDER_EXCLUDE, BORDER_KEEP, DECODE_ARGMAX,  # noqa: E402
                       DECODE_GAUSSIAN, DETECTION_STORAGE_FLOOR, MAX_PEAKS_PER_PANO,
                       PEAK_MIN_DISTANCE, RECORD_BORDER_KEY, border_band_edge)

ARMS = ('laurens', 'laurens_gsv')
#: The RampNet benchmark split each arm's judged bundle lives under.
BUNDLE_OF = {'laurens': 'laurens_mapillary', 'laurens_gsv': 'laurens_gsv'}
DATA_DIR = REPO_ROOT / 'docs' / 'figures' / 'seam-band-130' / 'data'
SUMMARY = DATA_DIR / 'summary.json'
HM_H, HM_W = 512, 1024
#: Thresholds the summary reports: the storage floor, the operating point, the benchmark tier.
THRESHOLDS = (DETECTION_STORAGE_FLOOR, 0.30, 0.55)
OPERATING = 0.30
#: Score tolerance for the instrument check (float32 rebuild noise is ~1e-7).
SCORE_TOL = 1e-6
#: The instrument gate: share of panos whose stored detections the rebuild reproduces.
GATE = 0.99
#: RampNet's benchmark match radius, normalized x units (scripts/agree_rate.PANO_RADIUS).
GT_RADIUS = 0.022
#: World-step fusion: the camera height the committed Laurens sites_meta.json files used
#: (runs/laurens{,_gsv}/sites_meta.json: camera_height_m 2.6), at the operating point.
WORLD_HEIGHT_M = 2.6
SEAM_EDGES = ('left', 'right')
#: Columns in the seam band: PEAK_MIN_DISTANCE on each side of the 1024-column heatmap.
SEAM_COLUMNS = 2 * PEAK_MIN_DISTANCE


def rnd(v, nd=7):
    """The one rounding every committed float and every threshold comparison uses."""
    return round(float(v), nd)


def key(x, y):
    """The heatmap pixel a normalized detection came from (exact for argmax)."""
    return (round(x * HM_W), round(y * HM_H))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_results(path):
    """{pano_id: record}, in file order."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec['pano']['panorama_id'])] = rec
    return out


def load_coarse(coarse_dir, pid):
    """(float32 coarse map as saved by subcell_decode.py detect, its sha256)."""
    import numpy as np
    c = np.load(Path(coarse_dir) / f'{pid}_coarse.npy')
    c = np.ascontiguousarray(c, dtype=np.float32)
    return c, hashlib.sha256(c.tobytes()).hexdigest()


def heatmap_from_coarse(c32):
    """The 512x1024 heatmap the head emitted, rebuilt by the exact bilinear operator (the
    same rebuild tests/test_decode.py uses on the stored fixture)."""
    import numpy as np
    from detectors import rampnet_subcell as sc
    return sc.upsample(c32.astype(np.float64)).astype(np.float32)


def stored_detections(rec):
    return [(d['x_normalized'], d['y_normalized'], d['confidence'])
            for d in rec.get('detections', [])]


def same_detections(a, b, tol=SCORE_TOL):
    """Equal pixel-key multisets, and every paired score within tol (paired after sorting
    by key, then score)."""
    ka = sorted((key(x, y), c) for x, y, c in a)
    kb = sorted((key(x, y), c) for x, y, c in b)
    if [k for k, _ in ka] != [k for k, _ in kb]:
        return False
    return all(abs(ca - cb) <= tol for (_, ca), (_, cb) in zip(ka, kb))


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def dump_json(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + '\n'


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(dump_json(obj))


def work_dir(arm, results):
    return Path(results).parent / 'seam_band_130'


# --------------------------------------------------------------------------------------- #
# a. check: does the rebuilt heatmap reproduce the stored detections?
# --------------------------------------------------------------------------------------- #

def check_pano(c32, stored, floor):
    """(reproduces, rebuilt exclude-argmax detections) for one pano."""
    from detectors import decode as dec
    h = heatmap_from_coarse(c32)
    rebuilt = dec.detections_from_heatmap(h, DECODE_ARGMAX, border=BORDER_EXCLUDE)
    want = [d for d in stored if d[2] >= floor]
    got = [d for d in rebuilt if d[2] >= floor]
    return same_detections(want, got), rebuilt


def diagnose(stored, rebuilt):
    """Why a pano does not reproduce (diagnostic columns only; `reproduces` decides):
    whether the pixel-key multisets agree among peaks >= 0.30 and >= 0.55 (scores ignored),
    and the largest score difference over pixel keys present on both sides."""
    def keys(ds, t):
        return sorted(key(x, y) for x, y, c in ds if c >= t)
    sa = {key(x, y): c for x, y, c in stored}
    sb = {key(x, y): c for x, y, c in rebuilt}
    shared = [abs(sa[k] - sb[k]) for k in sa.keys() & sb.keys()]
    return (int(keys(stored, 0.30) == keys(rebuilt, 0.30)),
            int(keys(stored, 0.55) == keys(rebuilt, 0.55)),
            rnd(max(shared), 6) if shared else '')


def cmd_check(args):
    t0 = time.perf_counter()
    recs = read_results(args.results)
    manifest = Path(args.results).parent / 'manifest.json'
    floor = DETECTION_STORAGE_FLOOR
    if manifest.exists():
        floor = json.loads(manifest.read_text(encoding='utf-8')).get(
            'detection_storage_floor', 0.55)
    decode_ref = {}
    if args.decode_file:
        with open(args.decode_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    decode_ref[str(d['pano_id'])] = d
    rows, n_ok, n_ref_ok, n_ref, n_sha_ok = [], 0, 0, 0, 0
    for pid, rec in recs.items():
        c32, sha = load_coarse(args.coarse_dir, pid)
        stored = stored_detections(rec)
        ok, rebuilt = check_pano(c32, stored, floor)
        n_ok += ok
        ref = decode_ref.get(pid)
        ref_ok = ''
        if ref is not None:
            n_ref += 1
            n_sha_ok += ref.get('coarse_sha256') == sha
            # the #111 pass's own exclude-argmax peaks, from the ORIGINAL float32 heatmap
            ref_ok = int(same_detections([tuple(d) for d in ref['argmax']], rebuilt))
            n_ref_ok += ref_ok
        rows.append([pid, sha, int(ok), len([d for d in stored if d[2] >= floor]),
                     len([d for d in rebuilt if d[2] >= floor]), ref_ok,
                     *diagnose(stored, rebuilt)])
    rows.sort()
    out = Path(args.data_dir) / f'{args.arm}_check.csv'
    write_csv(out, ['pano_id', 'coarse_sha256', 'reproduces', 'n_stored', 'n_rebuilt',
                    'rebuild_matches_detect_pass', 'keys_match_030', 'keys_match_055',
                    'max_dscore_shared'], rows)
    n = len(rows)
    report = {'arm': args.arm, 'panos': n, 'reproduce': n_ok,
              'rate': rnd(n_ok / n, 6) if n else None, 'gate': GATE,
              'passes_gate': bool(n and n_ok / n >= GATE), 'storage_floor': floor,
              'detect_pass_panos': n_ref, 'detect_pass_coarse_sha_match': n_sha_ok,
              'rebuild_matches_detect_pass': n_ref_ok,
              'results_sha256': sha256_file(args.results),
              'check_csv_sha256': sha256_file(out),
              'wall_s': round(time.perf_counter() - t0, 1)}
    wd = work_dir(args.arm, args.results)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / 'check.json').write_text(dump_json(report), encoding='utf-8', newline='')
    print(dump_json(report), end='')
    return report


# --------------------------------------------------------------------------------------- #
# b. peaks: exclude vs keep, both decodes, from one heatmap
# --------------------------------------------------------------------------------------- #

def both_rules(h, coarse=None):
    """{border: {decode: [(x, y, c), ...]}} from ONE heatmap."""
    import numpy as np
    from detectors import decode as dec
    c = None if coarse is None else np.asarray(coarse, dtype=np.float64)
    return {b: {m: dec.detections_from_heatmap(h, m, coarse=c, border=b)
                for m in (DECODE_ARGMAX, DECODE_GAUSSIAN)}
            for b in (BORDER_EXCLUDE, BORDER_KEEP)}


def straddle_flags(dets):
    """For each detection, whether it is one of a seam-straddling pair: two peaks with
    wrapped |dx| <= MIN_DISTANCE and |dy| <= MIN_DISTANCE heatmap px. Unwrapped, the peak
    finder never returns two peaks that close (its spacing is Chebyshev min_distance), so
    every such pair straddles the seam. Returns (flags, pairs as index tuples)."""
    keys = [key(x, y) for x, y, _ in dets]
    flags, pairs = [False] * len(dets), []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dx = abs(keys[i][0] - keys[j][0]) % HM_W
            dx = min(dx, HM_W - dx)
            if dx <= PEAK_MIN_DISTANCE and abs(keys[i][1] - keys[j][1]) <= PEAK_MIN_DISTANCE:
                flags[i] = flags[j] = True
                pairs.append((i, j))
    return flags, pairs


def gained_lost(rules):
    """(gained indices into keep's lists, lost count): keep vs exclude argmax pixel keys
    as multisets."""
    ex = Counter(key(x, y) for x, y, _ in rules[BORDER_EXCLUDE][DECODE_ARGMAX])
    seen = Counter()
    gained = []
    for i, (x, y, _) in enumerate(rules[BORDER_KEEP][DECODE_ARGMAX]):
        k = key(x, y)
        seen[k] += 1
        if seen[k] > ex[k]:
            gained.append(i)
    kp = Counter(key(x, y) for x, y, _ in rules[BORDER_KEEP][DECODE_ARGMAX])
    lost = sum((ex - kp).values())
    return gained, lost


def pano_rows(pid, rules):
    """Gained-peak CSV rows for one pano (both decodes), and its straddle pairs as
    (score_a, score_b)."""
    keep_a = rules[BORDER_KEEP][DECODE_ARGMAX]
    keep_g = rules[BORDER_KEEP][DECODE_GAUSSIAN]
    gained, lost = gained_lost(rules)
    flags, pairs = straddle_flags(keep_a)
    rows = []
    for i in gained:
        x, y, c = keep_a[i]
        edge = border_band_edge(x, y)
        for m, (gx, gy, _) in ((DECODE_ARGMAX, keep_a[i]), (DECODE_GAUSSIAN, keep_g[i])):
            rows.append([pid, m, rnd(gx), rnd(gy), rnd(c), edge or 'none', int(flags[i])])
    straddles = [(rnd(keep_a[i][2]), rnd(keep_a[j][2])) for i, j in pairs]
    return rows, lost, straddles


def tier_counts(dets):
    return {str(t): sum(1 for _, _, c in dets if rnd(c) >= t) for t in THRESHOLDS}


GAINED_HEADER = ['pano_id', 'decode', 'x', 'y', 'score', 'edge', 'straddle_pair']


def cmd_peaks(args):
    t0 = time.perf_counter()
    recs = read_results(args.results)
    check = {r['pano_id']: r for r in read_csv(Path(args.data_dir) / f'{args.arm}_check.csv')}
    wd = work_dir(args.arm, args.results)
    wd.mkdir(parents=True, exist_ok=True)
    peaks_path = wd / 'peaks.jsonl'
    rows, straddles = [], []
    exclude_counts = Counter()
    keep_counts = Counter()
    lost_total, capped, n_used = 0, 0, 0
    with open(peaks_path, 'w', newline='', encoding='utf-8') as fp:
        for pid in sorted(recs):
            row = check.get(pid)
            if row is None or row['reproduces'] != '1':
                continue
            c32, sha = load_coarse(args.coarse_dir, pid)
            if sha != row['coarse_sha256']:
                raise SystemExit(f'{pid}: coarse map changed since check ({sha[:12]} vs '
                                 f'{row["coarse_sha256"][:12]})')
            h = heatmap_from_coarse(c32)
            rules = both_rules(h, c32)
            n_used += 1
            prow, lost, strad = pano_rows(pid, rules)
            rows.extend(prow)
            straddles.extend(strad)
            lost_total += lost
            capped += len(rules[BORDER_KEEP][DECODE_ARGMAX]) >= MAX_PEAKS_PER_PANO
            exclude_counts.update(tier_counts(rules[BORDER_EXCLUDE][DECODE_ARGMAX]))
            keep_counts.update(tier_counts(rules[BORDER_KEEP][DECODE_ARGMAX]))
            fp.write(json.dumps({'pano_id': pid, 'coarse_sha256': sha,
                                 **{b: {m: [[rnd(v) for v in d] for d in ds]
                                        for m, ds in r.items()}
                                    for b, r in rules.items()}}) + '\n')
    gained_csv = Path(args.data_dir) / f'{args.arm}_gained.csv'
    write_csv(gained_csv, GAINED_HEADER, rows)
    counts = {'arm': args.arm, 'panos_counted': n_used,
              'exclude_peaks': {k: exclude_counts[k] for k in map(str, THRESHOLDS)},
              'keep_peaks': {k: keep_counts[k] for k in map(str, THRESHOLDS)},
              'lost_peaks': lost_total, 'panos_at_peak_cap': capped,
              'straddle_pairs': sorted([list(s) for s in straddles]),
              'peaks_jsonl_sha256': sha256_file(peaks_path)}
    write_json(Path(args.data_dir) / f'{args.arm}_counts.json', counts)
    if lost_total:
        print(f'WARNING: {lost_total} exclude peak(s) absent under keep (num_peaks cap)')
    print(dump_json({**counts, 'straddle_pairs': len(straddles),
                     'wall_s': round(time.perf_counter() - t0, 1)}), end='')
    return counts


# --------------------------------------------------------------------------------------- #
# d. world: lost ramps or lost views?
# --------------------------------------------------------------------------------------- #

def read_gained(arm, data_dir=DATA_DIR, decode=DECODE_ARGMAX):
    return [r for r in read_csv(Path(data_dir) / f'{arm}_gained.csv') if r['decode'] == decode]


def write_keep_file(results, gained_by_pano, out):
    """The run's own records with each pano's gained peaks APPENDED to its stored detections
    (so every stored detection keeps its det_index and the two files differ only by the
    band), and detection_border: keep on every record. Pano blocks untouched."""
    n_added = 0
    with open(results, encoding='utf-8') as f, \
            open(out, 'w', newline='\n', encoding='utf-8') as fo:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = str(rec['pano']['panorama_id'])
            add = gained_by_pano.get(pid, [])
            rec['detections'] = list(rec.get('detections', [])) + [
                {'x_normalized': x, 'y_normalized': y, 'confidence': c} for x, y, c in add]
            n_added += len(add)
            rec[RECORD_BORDER_KEY] = BORDER_KEEP
            fo.write(json.dumps(rec) + '\n')
    return n_added


def site_index(sites):
    """{(pano_id, det_index): site} over every member of every site."""
    return {(str(d.pano_id), d.det_index): s for s in sites for d, _ in s.members}


def classify(gained_keys, sites_ex, sites_kp):
    """{(pano_id, det_index): class} for each gained detection that fused.

    'view': its keep site holds stored members, at least one of whose exclude-run sites was
    already operational (a lost view of a ramp the run had). 'promoted': stored members
    exist but none of their exclude sites was operational (the gained peak lifts
    sub-threshold support to an operational site). 'new': its keep site holds no stored
    member (it seeded a site the exclude run does not have). A gained detection that did not
    fuse (dropped by projection: horizon, range, rig mask) is absent and reads as
    'not_projected'.
    """
    ex_idx, kp_idx = site_index(sites_ex), site_index(sites_kp)
    gained = set(gained_keys)
    out = {}
    for k in gained_keys:
        site = kp_idx.get(k)
        if site is None:
            continue
        stored = [(str(d.pano_id), d.det_index) for d, _ in site.members
                  if (str(d.pano_id), d.det_index) not in gained]
        if not stored:
            out[k] = 'new'
            continue
        ex_sites = {ex_idx[m].id: ex_idx[m] for m in stored if m in ex_idx}
        out[k] = 'view' if any(s.n_operational for s in ex_sites.values()) else 'promoted'
    return out


def gt_verdicts(gained_pts, entry, bundle_dets):
    """Each gained point on one bundle pano: 'tp' (matches a reviewer-marked missed ramp),
    'dup' (lands on a judged-true detection: a second peak on a ramp already found), 'fp'
    (neither, on a pano whose misses were reviewed), or 'unjudged' (misses not reviewed).
    One-to-one greedy matching within GT_RADIUS, seam-wrapped (agree_rate.match_pano)."""
    import agree_rate as ar
    out = ['unjudged'] * len(gained_pts)
    in_pool = (entry.get('no_missed') or bool(entry.get('missed'))) \
        if 'no_missed' in entry else True
    missed = [(m['x'], m['y']) for m in entry.get('missed') or []]
    tp = ar.match_pano(gained_pts, missed, GT_RADIUS)
    trues = [(x, y) for (x, y, _), v in zip(bundle_dets, entry.get('dets') or []) if v]
    dup = ar.match_pano(gained_pts, trues, GT_RADIUS)
    for i in range(len(gained_pts)):
        if i in tp:
            out[i] = 'tp'
        elif i in dup:
            out[i] = 'dup'
        elif in_pool:
            out[i] = 'fp'
    return out


WORLD_HEADER = ['pano_id', 'x', 'y', 'score', 'edge', 'world_class', 'bundle_pano', 'gt']


def cmd_world(args):
    import eval_sites as es
    import fuse_sites as fs
    t0 = time.perf_counter()
    results = Path(args.results)
    wd = work_dir(args.arm, results)
    wd.mkdir(parents=True, exist_ok=True)
    counts = json.loads((Path(args.data_dir) / f'{args.arm}_counts.json').read_text(encoding='utf-8'))
    if args.peaks and Path(args.peaks).exists() \
            and sha256_file(args.peaks) != counts['peaks_jsonl_sha256']:
        raise SystemExit(f'{args.peaks} is not the peaks.jsonl {args.arm}_counts.json records')
    gained = read_gained(args.arm, args.data_dir)
    by_pano = {}
    for r in gained:
        by_pano.setdefault(r['pano_id'], []).append(
            (float(r['x']), float(r['y']), float(r['score'])))
    keep_path = wd / 'results.keep.jsonl'
    n_added = write_keep_file(results, by_pano, keep_path)

    params = fs.FuseParams(min_confidence=OPERATING, camera_height_m=WORLD_HEIGHT_M)
    panos_ex, _, _, _ = fs.load_at_height(results, WORLD_HEIGHT_M)
    panos_kp, _, _, _ = fs.load_at_height(keep_path, WORLD_HEIGHT_M)
    sites_ex, _, stats_ex = fs.fuse(panos_ex, params)
    sites_kp, _, stats_kp = fs.fuse(panos_kp, params)

    # the gained detections' det_index in the keep file: after the stored ones, in order
    n_stored = {pid: len(rec.get('detections', [])) for pid, rec in read_results(results).items()}
    gkeys, gpts = [], []
    for pid in sorted(by_pano):
        for j, pt in enumerate(by_pano[pid]):
            gkeys.append((pid, n_stored[pid] + j))
            gpts.append(pt)
    cls = classify([k for k, (_, _, c) in zip(gkeys, gpts) if c >= OPERATING],
                   sites_ex, sites_kp)

    verdicts, bundle_ops = {}, {}
    vpath = Path(args.benchmark_root) / BUNDLE_OF[args.arm] / 'verdicts.json'
    rpath = vpath.with_name('records.jsonl')
    if vpath.exists():
        verdicts, bundle_ops = es.load_benchmark(BUNDLE_OF[args.arm], Path(args.benchmark_root))
    rows = []
    per_pano_gt = {}
    for pid in sorted(by_pano):
        if pid in verdicts:
            pts = [(x, y) for x, y, c in by_pano[pid]]
            per_pano_gt[pid] = gt_verdicts(pts, verdicts[pid], bundle_ops.get(pid, []))
    for (pid, idx), (x, y, c) in zip(gkeys, gpts):
        if c < OPERATING:
            continue
        j = idx - n_stored[pid]
        rows.append([pid, rnd(x), rnd(y), rnd(c), border_band_edge(x, y) or 'none',
                     cls.get((pid, idx), 'not_projected'), int(pid in verdicts),
                     per_pano_gt[pid][j] if pid in per_pano_gt else ''])
    write_csv(Path(args.data_dir) / f'{args.arm}_world.csv', WORLD_HEADER, rows)
    info = {'arm': args.arm,
            'fuse_params': {k: v for k, v in sorted(vars(params).items())},
            'gained_peaks_added': n_added,
            'operational_sites': {'exclude': stats_ex['n_operational_sites'],
                                  'keep': stats_kp['n_operational_sites']},
            'sites': {'exclude': stats_ex['n_sites'], 'keep': stats_kp['n_sites']},
            'results_sha256': sha256_file(results),
            'bundle': ({'split': BUNDLE_OF[args.arm], 'verdicts_sha256': sha256_file(vpath),
                        'records_sha256': sha256_file(rpath), 'panos': len(verdicts)}
                       if vpath.exists() else None)}
    write_json(Path(args.data_dir) / f'{args.arm}_world.json', info)
    print(dump_json({**info, 'wall_s': round(time.perf_counter() - t0, 1)}), end='')
    return info


# --------------------------------------------------------------------------------------- #
# c. summary (and e. verify): every table, from the committed files alone
# --------------------------------------------------------------------------------------- #

def wilson(k, n, z=1.96):
    """Wilson score interval (the form eval_sites.wilson and RampNet use)."""
    import math
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def share(k, n):
    lo, hi = wilson(k, n)
    return {'k': k, 'n': n, 'rate': rnd(k / n, 6) if n else None,
            'ci95': [rnd(lo, 6), rnd(hi, 6)]}


def arm_summary(check_rows, gained_rows, counts, world_rows, world_info):
    """The summary block of one arm (or of several, pooled: pass concatenated rows and
    summed counts)."""
    out = {'panos': len(check_rows),
           'panos_reproduce': sum(1 for r in check_rows if r['reproduces'] == '1'),
           'panos_counted': counts['panos_counted'],
           'lost_peaks': counts['lost_peaks'],
           'panos_at_peak_cap': counts['panos_at_peak_cap']}
    out['reproduce_rate'] = share(out['panos_reproduce'], out['panos'])
    tiers = {}
    arg = [r for r in gained_rows if r['decode'] == DECODE_ARGMAX]
    for t in THRESHOLDS:
        g = [r for r in arg if float(r['score']) >= t]
        seam = [r for r in g if r['edge'] in SEAM_EDGES]
        ex, kp = counts['exclude_peaks'][str(t)], counts['keep_peaks'][str(t)]
        tiers[str(t)] = {
            'exclude_peaks': ex, 'keep_peaks': kp, 'gained': len(g),
            # the under-report rate: of all the peaks keep finds, the share exclude dropped
            # (the issue's "3 of 150"); a proportion, so it carries a Wilson interval
            'gained_share_of_keep': share(len(g), kp),
            # ...and as a ratio to what exclude stored (no interval: not a proportion)
            'gained_per_exclude_peak': rnd(len(g) / ex, 6) if ex else None,
            'seam_gained': len(seam),
            'seam_left': sum(1 for r in seam if r['edge'] == 'left'),
            'seam_right': sum(1 for r in seam if r['edge'] == 'right'),
            'top_bottom_gained': sum(1 for r in g if r['edge'] in ('top', 'bottom')),
            'seam_share_of_keep': share(len(seam), kp),
            'panos_with_seam_gain': len({r['pano_id'] for r in seam}),
            'straddle_pairs': sum(1 for a, b in counts['straddle_pairs'] if a >= t and b >= t),
            'gained_in_straddle_pairs': sum(1 for r in g if r['straddle_pair'] == '1'),
        }
    out['tiers'] = tiers
    out['geometric_expectation'] = rnd(SEAM_COLUMNS / HM_W, 6)
    if world_info is not None:
        w = {}
        for edges, name in ((SEAM_EDGES, 'seam'), (('top', 'bottom'), 'top_bottom')):
            rows = [r for r in world_rows if r['edge'] in edges]
            w[name] = {'peaks': len(rows),
                       **{c: sum(1 for r in rows if r['world_class'] == c)
                          for c in ('view', 'promoted', 'new', 'not_projected')}}
        seam_rows = [r for r in world_rows if r['edge'] in SEAM_EDGES]
        on_bundle = [r for r in seam_rows if r['bundle_pano'] == '1']
        gt = Counter(r['gt'] for r in on_bundle)
        judged = gt['tp'] + gt['fp'] + gt['dup']
        w['gt'] = {'seam_peaks_on_bundle_panos': len(on_bundle),
                   'tp': gt['tp'], 'dup': gt['dup'], 'fp': gt['fp'],
                   'unjudged': gt['unjudged'],
                   # quoted only when there are at least 10 judged peaks
                   'precision': share(gt['tp'], gt['tp'] + gt['fp'])
                   if judged >= 10 else None}
        w['operational_sites'] = world_info['operational_sites']
        w['operational_site_change'] = (world_info['operational_sites']['keep']
                                        - world_info['operational_sites']['exclude'])
        out['world'] = w
    return out


def sum_counts(cs):
    out = {'panos_counted': 0, 'lost_peaks': 0, 'panos_at_peak_cap': 0, 'straddle_pairs': [],
           'exclude_peaks': Counter(), 'keep_peaks': Counter()}
    for c in cs:
        for k in ('panos_counted', 'lost_peaks', 'panos_at_peak_cap'):
            out[k] += c[k]
        out['straddle_pairs'] += c['straddle_pairs']
        out['exclude_peaks'].update(c['exclude_peaks'])
        out['keep_peaks'].update(c['keep_peaks'])
    return out


def sum_world(ws):
    if any(w is None for w in ws):
        return None
    return {'operational_sites': {k: sum(w['operational_sites'][k] for w in ws)
                                  for k in ('exclude', 'keep')}}


def build_summary(data_dir=DATA_DIR, arms=ARMS):
    data_dir = Path(data_dir)
    per = {}
    parts = []
    for arm in arms:
        check = read_csv(data_dir / f'{arm}_check.csv')
        gained = read_csv(data_dir / f'{arm}_gained.csv')
        counts = json.loads((data_dir / f'{arm}_counts.json').read_text(encoding='utf-8'))
        wpath = data_dir / f'{arm}_world.json'
        winfo = json.loads(wpath.read_text(encoding='utf-8')) if wpath.exists() else None
        world = read_csv(data_dir / f'{arm}_world.csv') if winfo is not None else []
        per[arm] = arm_summary(check, gained, counts, world, winfo)
        parts.append((check, gained, counts, world, winfo))
    pooled = arm_summary([r for p in parts for r in p[0]], [r for p in parts for r in p[1]],
                         sum_counts([p[2] for p in parts]), [r for p in parts for r in p[3]],
                         sum_world([p[4] for p in parts]))
    inputs = {f.name: sha256_file(f) for f in sorted(data_dir.glob('*'))
              if f.name != SUMMARY.name and f.suffix in ('.csv', '.json')}
    return {'issue': 130, 'arms': per, 'pooled': pooled, 'inputs_sha256': inputs,
            'thresholds': list(THRESHOLDS), 'operating_point': OPERATING,
            'gt_radius': GT_RADIUS, 'world_camera_height_m': WORLD_HEIGHT_M}


def cmd_summary(args):
    s = build_summary(args.data_dir)
    out = Path(args.data_dir) / SUMMARY.name
    with open(out, 'w', newline='', encoding='utf-8') as f:
        f.write(dump_json(s))
    print(render_tables(s))
    print(f'wrote {out}')


def render_tables(s):
    """The doc's tables, as markdown, from a summary dict."""
    buf = io.StringIO()
    cols = list(s['arms']) + ['pooled']
    blocks = {**s['arms'], 'pooled': s['pooled']}
    buf.write('| arm | panos | reproduce | tier | exclude peaks | keep peaks | gained | '
              'gained / keep [95% CI] | seam (L/R) | top/bottom | panos w/ seam gain | '
              'straddle pairs |\n|' + '---|' * 12 + '\n')
    for a in cols:
        b = blocks[a]
        for t, r in b['tiers'].items():
            g = r['gained_share_of_keep']
            buf.write(f"| {a} | {b['panos']} | {b['panos_reproduce']} | {t} | "
                      f"{r['exclude_peaks']} | {r['keep_peaks']} | {r['gained']} | "
                      f"{100 * (g['rate'] or 0):.2f}% [{100 * g['ci95'][0]:.2f}, "
                      f"{100 * g['ci95'][1]:.2f}] | {r['seam_gained']} ({r['seam_left']}/"
                      f"{r['seam_right']}) | {r['top_bottom_gained']} | "
                      f"{r['panos_with_seam_gain']} | {r['straddle_pairs']} |\n")
    if all('world' in blocks[a] for a in cols):
        buf.write('\n| arm | seam peaks >= 0.30 | lost view | promoted | new site | '
                  'not projected | operational sites exclude -> keep |\n|' + '---|' * 7 + '\n')
        for a in cols:
            w = blocks[a]['world']
            buf.write(f"| {a} | {w['seam']['peaks']} | {w['seam']['view']} | "
                      f"{w['seam']['promoted']} | {w['seam']['new']} | "
                      f"{w['seam']['not_projected']} | {w['operational_sites']['exclude']} -> "
                      f"{w['operational_sites']['keep']} ({w['operational_site_change']:+d}) |\n")
        buf.write('\n| arm | seam peaks on bundle panos | tp | dup | fp | unjudged | '
                  'precision |\n|' + '---|' * 7 + '\n')
        for a in cols:
            g = blocks[a]['world']['gt']
            p = g['precision']
            ptxt = ('not quoted (< 10 judged)' if p is None else
                    f"{p['rate']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]")
            buf.write(f"| {a} | {g['seam_peaks_on_bundle_panos']} | {g['tp']} | {g['dup']} | "
                      f"{g['fp']} | {g['unjudged']} | {ptxt} |\n")
    return buf.getvalue()


def cmd_verify(args):
    out = Path(args.data_dir) / SUMMARY.name
    want = out.read_bytes()
    got = dump_json(build_summary(args.data_dir)).encode('utf-8')
    if got != want:
        raise SystemExit(f'{out} does NOT re-derive from the committed CSV/JSON files')
    print(render_tables(json.loads(want)))
    print(f'OK: {out} re-derives byte for byte from {len(json.loads(want)["inputs_sha256"])} '
          f'committed files')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('check', 'peaks'):
        p = sub.add_parser(name)
        p.add_argument('arm', choices=ARMS)
        p.add_argument('--results', type=Path, required=True)
        p.add_argument('--coarse-dir', type=Path, required=True)
        if name == 'check':
            p.add_argument('--decode-file', type=Path, default=None,
                           help="the #111 detect pass's decode_<arm>.jsonl: also checks the "
                                "rebuild against that pass's own peaks and coarse sha256")
    p = sub.add_parser('world')
    p.add_argument('arm', choices=ARMS)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--peaks', type=Path, default=None)
    p.add_argument('--benchmark-root', type=Path, default=REPO_ROOT.parent / 'RampNet' /
                   'benchmark')
    for name in ('summary', 'verify'):
        sub.add_parser(name)
    for p in sub.choices.values():
        p.add_argument('--data-dir', type=Path, default=DATA_DIR,
                       help='where the committed CSV/JSON files are read and written '
                            '(default docs/figures/seam-band-130/data)')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {'check': cmd_check, 'peaks': cmd_peaks, 'world': cmd_world,
            'summary': cmd_summary, 'verify': cmd_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
