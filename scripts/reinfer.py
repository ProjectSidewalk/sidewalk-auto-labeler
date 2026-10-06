"""Re-run inference over an existing run's panos, by id, at the current storage floor.

Why this exists (issue #20): runs made before the storage floor (#28) hold only peaks at
or above 0.55, so when the operating point moved to 0.30 there was nothing stored to
submit for the 0.30-0.55 band. A fresh `main.py` run would re-scan and re-thin against
today's coverage and pick a *different* pano set; this fetches exactly the panos the
run already has, through the same fetch -> detect -> record path as main.py, and writes
a sibling file. The pano block is rebuilt from fresh metadata (positions included), so
the run's `--mapillary-position` binding is honoured from the manifest.

    python scripts/reinfer.py runs/richmond                 # -> runs/richmond/results.f01.jsonl
    python scripts/reinfer.py runs/richmond --verify        # compare, no network, no GPU
    python scripts/reinfer.py runs/richmond --verify --band-floor 0.30 --write-band-file

    # issue #56: re-infer exactly the listed ids through the GSV path (zoom-3 pixels via
    # panorama.fetch_panorama) into a control file for scripts/provenance_gate.py --control
    python scripts/reinfer.py runs/vancouver \\
        --ids runs/vancouver/provenance_gate/control_ids.txt --out runs/vancouver/control_zoom3.jsonl

Output order follows the old file where it can: futures are consumed in submission order,
so a single clean pass preserves it, but a pano that failed and is retried on a later pass
lands at the end of that pass instead.

Resumable: `<out>.processed` caches the ids already written (and the deterministic skips -
a pano the source no longer serves).

`--verify` compares the new file to the old one, per pano, with no network and no GPU. A
pano REPRODUCES when both of these hold:

  * its detections at the tier the server actually holds (the submission record's
    `min_confidence`, not the benchmark constant) land on exactly the same pixel keys PS
    stores - `round(x*W)`, `round(y*H)`; and
  * its pano block still describes the same image and the same place - width, height,
    lat, lng and camera heading unchanged. Dimensions decide the pixel key, and position
    is what every label inherits, so a band whose panos moved would arrive in a different
    frame from the base labels already live.

**Coarse-cell agreement is a diagnostic, never a reproduction** (#111). RampNet's heatmap is
a bilinear 8x upsample of a stride-32 map, so a peak can only land on residue 3 or 4 of each
8-cell block, and when two neighbouring coarse cells are near-tied a small input change (a
different JPEG of the same pano) moves the argmax 7-8 heatmap cells: the same ramp, decoded
at the other cell. `--verify` therefore also reports, for every pano whose pixel keys
differ (and whose width/height did not change), how its old and new keys pair up one-to-one
within +/-1 coarse cell (Chebyshev, seam-wrapped): how many pairs sit in the same heatmap
cell, one cell apart (residue 3 <-> 4), 2-6 apart (off the grid), 7-8 apart (**the flip**),
and how many keys found no partner. That tells an operator WHY a pano failed. It does not
change the decision: a pano still reproduces only on exact pixel keys, because
`--write-band-file` asserts that the band file's labels at the server's tier ARE the live
ones, and a label that moved one coarse cell is a different pixel on the server -- shipping
the new record would insert its band beside a live label that no longer matches the file.
A pano that agrees only within a cell is carried over from the old file like any other.

`--write-band-file` then writes the file a band may actually ship from: the NEW record for
every pano that reproduced, and the OLD record for every pano that did not. The old records
hold nothing below the tier the server has, so their band is empty and `send_to_ps.py` skips
them without a POST. That gives the file one property, which is asserted before it is left
on disk: **its labels at and above the server's tier are exactly the ones already live**. A
derived `<band file>.submission.json` therefore describes a real campaign, and the ordinary
band guard applies with no override.

**The peak decode** (#111). A re-inference uses the run's own decode (manifest.json
`detection_decode`, argmax when absent) unless `--decode` says otherwise, and an existing
`--out` file is bound to the decode its records carry. `--verify` refuses to compare files
written under different decodes (every sub-cell position differs from its argmax pixel, so
nothing would reproduce and the coarse-cell diagnostic would describe the decode, not the
model) unless `--allow-mixed-decode` asks for that comparison as a diagnostic; and
`--write-band-file` refuses a mix outright, because a band file holding gaussian records over
an argmax base would ship its band in a second frame.

**The border rule** (#130), the same way. A re-inference uses the run's own rule (manifest.json
`detection_border`, `exclude` when absent) unless `--border` says otherwise; an existing
`--out` is bound to the rule its records carry; `--verify` refuses an `exclude`/`keep` pair
unless `--allow-mixed-border` asks for it as a diagnostic, and `--write-band-file` refuses it
outright -- a `keep` record shipped beside `exclude` live labels would add its seam-band
labels in the band. Under the diagnostic, `--verify` says how many differing pixel keys lie
in the 10-px border band (seam, zenith, nadir), i.e. are the border rule rather than the model.
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geo  # noqa: E402
from detectors import (BENCHMARK_CONFIDENCE, BORDERS, DECODES, DEFAULT_BORDER,  # noqa: E402
                       DEFAULT_DECODE, border_band_edge, borders_in_file, decodes_in_file,
                       on_camera_rig, single_border, single_decode)
from detectors.batching import BATCH_SIZE_HELP, report_detector  # noqa: E402

# The pano fields a band must not change: the first two set the pixel key PS stores, the
# rest are what every label on that pano inherits (SidewalkWebpage#5361).
PANO_INVARIANTS = ('width', 'height', 'lat', 'lng', 'camera_heading')


def read_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pixel_set(record, floor):
    """The pixel keys PS stores for this record, as a MULTISET.

    A multiset rather than a set because two detections can round to the same pixel: if
    re-inference adds or drops one of a co-located pair, a set comparison sees no change,
    `verify` calls the pano exact, and the disagreement only surfaces later as a label-count
    mismatch when the derived record is written. Counting them makes `verify` the place that
    catches it, which is the place that can carry the pano over.
    """
    w, h = record['pano']['width'], record['pano']['height']
    return Counter((round(d['x_normalized'] * w), round(d['y_normalized'] * h))
                   for d in record['detections'] if d['confidence'] >= floor)


def pano_drift(old, new):
    """Which of PANO_INVARIANTS differ between two records' pano blocks, as (field, old, new)."""
    return [(f, old['pano'].get(f), new['pano'].get(f))
            for f in PANO_INVARIANTS if old['pano'].get(f) != new['pano'].get(f)]


def cell_agreement(old_set, new_set, width, height):
    """Pair two pixel-key multisets one-to-one within +/-1 coarse heatmap cell (#111).

    Greedy on distance (closest pairs first; ties by key), Chebyshev in heatmap cells with
    x wrapping at the seam, tolerance one coarse cell plus the rounding pixel. Returns
    (pairs by geo.CELL_SHIFT_CLASSES key, unpaired old keys, unpaired new keys). A
    DIAGNOSTIC: nothing here decides whether a pano reproduces (see the module docstring).

    Example (a flip: the same ramp 7 cells over, on a 16384-wide pano):
        >>> from collections import Counter
        >>> cell_agreement(Counter({(1000, 4000): 1}), Counter({(1112, 4000): 1}), 16384, 8192)
        (Counter({'flip': 1}), 0, 0)
    """
    tol = geo.HEATMAP_COARSE_CELL_PX + max(geo.HEATMAP_WIDTH / width,
                                           geo.HEATMAP_HEIGHT / height)
    olds = sorted(old_set.elements())
    news = sorted(new_set.elements())
    cands = sorted((geo.heatmap_cell_distance(a, b, width, height), i, j)
                   for i, a in enumerate(olds) for j, b in enumerate(news))
    used_old, used_new, pairs = set(), set(), Counter()
    for dist, i, j in cands:
        if dist > tol:
            break
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        pairs[geo.cell_shift_class(dist)] += 1
    return pairs, len(olds) - len(used_old), len(news) - len(used_new)


def verify(old_path, new_path, floor=BENCHMARK_CONFIDENCE, band_floor=None):
    """Compare the new file to the old one at `floor`.

    Returns (summary, mismatches, carry_over) where `carry_over` is the set of pano ids
    that did NOT reproduce - the ids a band file must take from the old file. A pano
    reproduces only on exact pixel keys; summary['coarse_cell'] is the #111 diagnostic over
    the panos whose keys differ (cell_agreement), and never moves a pano into or out of
    carry_over.
    """
    new_by_id = {r['pano']['panorama_id']: r for r in read_records(new_path)}
    summary = {'old_panos': 0, 'new_panos': len(new_by_id), 'missing': 0, 'exact': 0,
               'mismatch': 0, 'pano_drift': 0, 'old_labels': 0}
    if band_floor is not None:
        summary['band_labels'] = 0
    cells = {'panos': 0, 'panos_agree_within_cell': 0, 'pairs': Counter(),
             'unpaired_old': 0, 'unpaired_new': 0}
    band = {'panos_only_in_band': 0, 'keys_old': 0, 'keys_new': 0}
    mismatches = []
    carry_over = set()
    for old in read_records(old_path):
        pid = old['pano']['panorama_id']
        summary['old_panos'] += 1
        old_set = pixel_set(old, floor)
        summary['old_labels'] += sum(old_set.values())
        new = new_by_id.get(pid)
        if new is None:
            # Not carried over: a band file cannot hold a record that does not exist. The
            # caller refuses outright rather than silently shipping a shorter file.
            summary['missing'] += 1
            continue
        drift = pano_drift(old, new)
        new_set = pixel_set(new, floor)
        if new_set == old_set and not drift:
            summary['exact'] += 1
            if band_floor is not None:
                # The rig mask has to be applied here too, or this number is not the one
                # the band would actually ship — and it is the number an operator reads to
                # decide whether to ship at all. (Richmond: 3,448 raw vs 3,436 delivered.)
                summary['band_labels'] += sum(
                    1 for d in new['detections']
                    if band_floor <= d['confidence'] < floor
                    and not on_camera_rig(d['y_normalized']))
            continue
        carry_over.add(pid)
        if drift:
            summary['pano_drift'] += 1
            mismatches.append((pid, [f"{f}: {o!r} -> {n!r}" for f, o, n in drift], []))
        if new_set != old_set:
            summary['mismatch'] += 1
            mismatches.append((pid, sorted(old_set - new_set), sorted(new_set - old_set)))
            w, h = old['pano']['width'], old['pano']['height']
            # #130 diagnostic: differing keys inside the 10-px border band are what the
            # border rule changes; a pano whose every differing key is there differs by
            # the rule alone.
            only_old, only_new = old_set - new_set, new_set - old_set
            in_band = lambda k: border_band_edge(k[0] / w, k[1] / h) is not None  # noqa: E731
            n_old = sum(n for k, n in only_old.items() if in_band(k))
            n_new = sum(n for k, n in only_new.items() if in_band(k))
            band['keys_old'] += n_old
            band['keys_new'] += n_new
            band['panos_only_in_band'] += (n_old + n_new) == sum((only_old + only_new).values())
            if (new['pano']['width'], new['pano']['height']) == (w, h):
                pairs, lone_old, lone_new = cell_agreement(old_set, new_set, w, h)
                cells['panos'] += 1
                cells['panos_agree_within_cell'] += not (lone_old or lone_new)
                cells['pairs'].update(pairs)
                cells['unpaired_old'] += lone_old
                cells['unpaired_new'] += lone_new
    cells['pairs'] = {k: cells['pairs'][k] for k, _, _ in geo.CELL_SHIFT_CLASSES}
    summary['coarse_cell'] = cells
    summary['border_band'] = band
    return summary, mismatches, carry_over


def server_tier(record, default=BENCHMARK_CONFIDENCE):
    """The threshold the server actually holds, per the old file's submission record.

    `verify` has to compare at this tier, not at the benchmark constant: the question it
    answers is "are the labels already live the ones this file describes?", and only the
    record knows what was sent. They coincide today (every legacy campaign went at 0.55),
    which is exactly why comparing at the constant looked correct.
    """
    tiers = {state.get('min_confidence') for state in (record.get('endpoints') or {}).values()
             if state.get('min_confidence') is not None}
    if not tiers:
        return default
    if len(tiers) > 1:
        raise SystemExit(f"the submission record lists more than one threshold ({sorted(tiers)}); "
                         f"a band needs one tier to sit on. Resolve it by hand.")
    return tiers.pop()


def write_band_file(old_path, new_path, band_path, carry_over, tier):
    """Write the file a band may ship from, then prove the property that makes it safe.

    One record per line in the old file's order: the new record where the pano reproduced,
    the old record where it did not. Returns the number of records carried over.
    """
    new_by_id = {r['pano']['panorama_id']: r for r in read_records(new_path)}
    carried = 0
    tmp_path = band_path.with_name(band_path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='\n') as f:
        for old in read_records(old_path):
            pid = old['pano']['panorama_id']
            if pid in carry_over:
                record, carried = old, carried + 1
            else:
                record = new_by_id[pid]
            f.write(json.dumps(record) + '\n')
    tmp_path.replace(band_path)

    # The invariant, checked rather than trusted: re-read what was actually written and
    # confirm its labels at or above the server's tier are exactly the old file's. If this
    # ever fails the file must not survive - shipping a band from it would insert labels
    # the server already holds, and PS cannot retire them (SidewalkWebpage#5382).
    def keys(path):
        """Every (pano, pixel) key in the file at `tier`, with multiplicity."""
        counts = Counter()
        for record in read_records(path):
            pid = record['pano']['panorama_id']
            for px, n in pixel_set(record, tier).items():
                counts[(pid, px)] += n
        return counts

    old_keys, band_keys = keys(old_path), keys(band_path)
    if band_keys != old_keys:
        band_path.unlink()
        raise SystemExit(
            f"{band_path.name} does not reproduce {old_path.name} at {tier} "
            f"({sum((old_keys - band_keys).values())} missing, "
            f"{sum((band_keys - old_keys).values())} extra); "
            f"refusing to leave it on disk.")
    return carried


def write_derived_record(old_path, band_path, tier):
    """Derive the band file's submission record from the old file's.

    Honest only because of the invariant `write_band_file` just proved: the band file's
    labels at `tier` ARE the ones live on every endpoint the old record names, so those
    campaigns describe this file too. Every endpoint must be complete on the old file - a
    band only ever sits on top of a finished campaign.
    """
    import send_to_ps

    old_record = send_to_ps.load_submission_record(send_to_ps.submission_record_path(old_path))
    if not old_record.get('endpoints'):
        raise SystemExit(f"{old_path.name} has no submission record, so there is no campaign "
                         f"for a band to sit on. Submit it normally first.")
    old_lines = old_record.get('total_lines')
    digest, total_lines, total_bytes = send_to_ps.hash_and_count(band_path)
    # This has to equal what the recorded campaign ACTUALLY sent, which depends on whether
    # that campaign ran with the nadir mask - the record says so in `rig_masked` (absent
    # means it predates the mask). Both counts are taken once, above the loop: each is a
    # property of the file, not of the endpoint, and the file is large enough that
    # re-reading it per endpoint would be a second full pass for nothing.
    counts = {masked: send_to_ps.count_band_labels_in_file(band_path, tier, None,
                                                           mask_rig=masked)
              for masked in (False, True)}
    endpoints = {}
    for url, state in old_record['endpoints'].items():
        sent = state.get('submitted_lines', 0)
        if not isinstance(old_lines, int) or sent < old_lines:
            raise SystemExit(f"{url} holds {sent} of {old_lines} line(s) of {old_path.name}; a "
                             f"band only sits on a COMPLETE campaign. Finish it first.")
        recounted = counts[bool(state.get('rig_masked', False))]
        if recounted != state.get('labels_submitted'):
            raise SystemExit(f"{band_path.name} holds {recounted} label(s) at {tier} but "
                             f"{url} was recorded with {state.get('labels_submitted')}. The "
                             f"band file does not describe that campaign; refusing.")
        endpoints[url] = dict(state, submitted_lines=total_lines, labels_submitted=recounted)
    record_path = send_to_ps.submission_record_path(band_path)
    record = {
        "input_file": band_path.name,
        "sha256": digest,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        # Provenance: this record was not written by a campaign, it was derived from one.
        "derived_from": {"input_file": old_path.name,
                         "sha256": old_record.get('sha256'),
                         "total_lines": old_lines,
                         "tier": tier},
        "endpoints": endpoints,
    }
    # The band file's border rule (#130), as send_to_ps writes it: only when it is not plain
    # exclude, so a derived record over a keep campaign cannot read back as exclude to the
    # city-wide guard (#131 review M5; the decode twin is #129's N4).
    border = single_border(borders_in_file(band_path), band_path.name)
    if border != DEFAULT_BORDER:
        record["detection_border"] = border
    tmp_path = record_path.with_name(record_path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    tmp_path.replace(record_path)
    return record_path, endpoints


def read_id_list(path):
    """Pano ids, one per line; blank lines and `#` comments ignored; order kept, deduped."""
    ids = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            ids.append(line)
    return list(dict.fromkeys(ids))


def select_todo(run_dir, done, ids=None):
    """(pano_id, lat, lng) to re-infer: every record of the run not yet `done`, or with
    `ids` exactly those ids, in the list's order, positioned from the run's own records.
    An id the run has no record for is refused: without a record there is no lat/lng to
    build the pano block from, and a silent drop would shrink a pre-registered draw."""
    if ids is None:
        return [(r['pano']['panorama_id'], r['pano']['lat'], r['pano']['lng'])
                for r in read_records(run_dir / 'results.jsonl')
                if r['pano']['panorama_id'] not in done]
    positions = {r['pano']['panorama_id']: (r['pano']['lat'], r['pano']['lng'])
                 for r in read_records(run_dir / 'results.jsonl')}
    unknown = [pid for pid in ids if pid not in positions]
    if unknown:
        raise SystemExit(f"{len(unknown)} id(s) of --ids have no record in "
                         f"{run_dir / 'results.jsonl'}, e.g. {unknown[:3]}")
    return [(pid, *positions[pid]) for pid in ids if pid not in done]


def run_decode(manifest):
    """The decode a run was detected with (absent key = argmax, every pre-#111 run)."""
    return manifest.get('detection_decode', DEFAULT_DECODE)


def resolve_decode(manifest, out_path, decode=None):
    """The decode a re-inference into ``out_path`` uses: ``decode`` if given, else the run's.
    Refuses (SystemExit) when ``out_path`` already holds records of another decode -- a
    resume must not put two frames in one file."""
    decode = decode or run_decode(manifest)
    if Path(out_path).exists():
        try:
            have = single_decode(decodes_in_file(out_path), Path(out_path).name)
        except ValueError as e:
            raise SystemExit(str(e))
        if Path(out_path).stat().st_size and have != decode:
            raise SystemExit(f"{Path(out_path).name} already holds '{have}' records; this "
                             f"re-inference would append '{decode}' ones (issue #111). Use "
                             f"another --out, or --decode {have}.")
    return decode


def run_border(manifest):
    """The border rule a run was detected with (absent key = exclude, every pre-#130 run)."""
    return manifest.get('detection_border', DEFAULT_BORDER)


def resolve_border(manifest, out_path, border=None):
    """The border rule a re-inference into ``out_path`` uses: ``border`` if given, else the
    run's. Refuses (SystemExit) when ``out_path`` already holds records of the other rule."""
    border = border or run_border(manifest)
    if Path(out_path).exists():
        try:
            have = single_border(borders_in_file(out_path), Path(out_path).name)
        except ValueError as e:
            raise SystemExit(str(e))
        if Path(out_path).stat().st_size and have != border:
            raise SystemExit(f"{Path(out_path).name} already holds '{have}' records; this "
                             f"re-inference would append '{border}' ones (issue #130). Use "
                             f"another --out, or --border {have}.")
    return border


def reinfer(run_dir, out_path, workers, limit, ids=None, batch_size=1, decode=None,
            border=None):
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    decode = resolve_decode(manifest, out_path, decode)
    border = resolve_border(manifest, out_path, border)
    if border != run_border(manifest):
        print(f"NOTE: {run_dir.name} was detected with the '{run_border(manifest)}' border "
              f"rule; this re-inference uses '{border}' (#130). --verify will refuse the pair "
              f"without --allow-mixed-border.")
    if decode != run_decode(manifest):
        print(f"NOTE: {run_dir.name} was detected with the '{run_decode(manifest)}' decode; "
              f"this re-inference uses '{decode}' (#111). --verify will refuse the pair "
              f"without --allow-mixed-decode.")
    source_name = manifest.get('imagery_source') or manifest.get('source') or 'gsv'
    if ids is not None and source_name != 'gsv':
        raise SystemExit(f"--ids re-runs panos through the GSV path; {run_dir.name} is a "
                         f"'{source_name}' run.")

    import main  # noqa: E402  (loads torch lazily below, like main.py does)
    from sources import get_source
    from detectors.curb_ramp import CurbRampDetector

    source = get_source(source_name)
    if source_name == 'mapillary':
        source.POSITION_FIELD = manifest.get('mapillary_position', 'sfm')
    source.prepare()
    from detectors import ModelProvenanceError
    try:
        # refuses an unknown revision (issue #39); batch_size > 1 batches forwards (#2)
        main.curb_ramp_detector = CurbRampDetector(batch_size=batch_size, decode=decode,
                                                   border=border)
    except ModelProvenanceError as e:
        raise SystemExit(str(e))
    provenance = main.curb_ramp_detector.provenance
    # A re-inference exists to reproduce the run's own detections, so it should use the
    # weights the run was made with. It writes a separate file and --verify catches any
    # pixel that moved, so a different revision is warned about rather than refused.
    bound = manifest.get('model_revision')
    if bound and bound != provenance['model_revision']:
        print(f"WARNING: {run_dir.name} was made with revision {bound[:12]}; re-inferring "
              f"with {provenance['model_revision'][:12]}. --verify will show what moved.")
    print(f"-> Model: {provenance['model_id']} (trained {provenance['model_training_date']}), "
          f"decode {decode}, border {border}")

    cache_path = Path(f"{out_path}.processed")
    done = set()
    if cache_path.exists():
        done = {l.strip() for l in cache_path.read_text(encoding='utf-8').splitlines() if l.strip()}
    todo = select_todo(run_dir, done, ids)
    if limit is not None:
        todo = todo[:limit]
    print(f"-> {len(done)} already done, {len(todo)} to re-infer from {source_name} "
          f"(storage floor {main.DETECTION_STORAGE_FLOOR}) -> {out_path}")

    counts = {'success': 0, 'skipped': 0, 'failed': 0}
    # Futures are consumed in submission order so the output keeps the input's line order;
    # downloads still overlap (the pool runs ahead) and the GPU serialises inference.
    t_pass = time.perf_counter()
    try:
        with open(cache_path, 'a', encoding='utf-8') as f_cache, \
             open(out_path, 'a', encoding='utf-8') as f_out, \
             ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(main.process_pano, source, pid, lat, lon)
                       for pid, lat, lon in todo]
            try:
                from tqdm import tqdm
                it = tqdm(futures, desc="Re-inferring")
            except ImportError:
                it = futures
            for future in it:
                counts[main.handle_result(future.result(), f_cache, f_out, provenance)] += 1
        print(f"-> Re-inference: {counts['success']} written, {counts['skipped']} skipped "
              f"(no longer served / unusable), {counts['failed']} failed (retry by re-running).")
        report_detector(main.curb_ramp_detector, time.perf_counter() - t_pass, counts['success'])
    finally:  # also on Ctrl-C or an exception out of the pass
        getattr(main.curb_ramp_detector, "close", lambda: None)()


def main_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir', type=Path, help='runs/<name>, holding results.jsonl + manifest.json')
    ap.add_argument('--out', type=Path, default=None,
                    help='output file (default <run_dir>/results.f01.jsonl)')
    ap.add_argument('--verify', action='store_true',
                    help='compare --out against results.jsonl at the tier the server holds '
                         'and exit 1 on any mismatch; no network, no GPU')
    ap.add_argument('--band-floor', type=float, default=None,
                    help='with --verify: also count labels in [band-floor, tier) on the '
                         'panos that reproduced, i.e. what a band campaign would ship')
    ap.add_argument('--write-band-file', nargs='?', const=True, default=None, metavar='PATH',
                    help='with --verify: write the file a band may ship from (default '
                         '<run_dir>/results.band.jsonl) plus its derived submission record. '
                         'Panos that did not reproduce are taken from results.jsonl, so their '
                         'band is empty and they are never POSTed.')
    ap.add_argument('--workers', type=int, default=None,
                    help='fetch threads (default main.PROCESSING_CONCURRENCY)')
    ap.add_argument('--batch-size', type=int, default=1, help=BATCH_SIZE_HELP)
    ap.add_argument('--limit', type=int, default=None, help='re-infer at most this many panos')
    ap.add_argument('--decode', choices=DECODES, default=None,
                    help="peak decode for the re-inference (#111; default: the run's own, "
                         "manifest.json detection_decode, argmax when absent)")
    ap.add_argument('--allow-mixed-decode', action='store_true',
                    help='with --verify: compare files written under different decodes, as a '
                         'diagnostic only (never with --write-band-file)')
    ap.add_argument('--border', choices=BORDERS, default=None,
                    help="peak border rule for the re-inference (#130: exclude, keep = RampNet's, "
                         "or wrap = keep plus NMS across the seam; default: the run's own, "
                         "manifest.json detection_border, exclude when absent)")
    ap.add_argument('--allow-mixed-border', action='store_true',
                    help='with --verify: compare an exclude file with a keep file, as a '
                         'diagnostic only (never with --write-band-file)')
    ap.add_argument('--ids', type=Path, default=None, metavar='FILE',
                    help='re-infer exactly these pano ids (one per line, each with a record in '
                         'results.jsonl) through the GSV path into --out, which is required. '
                         'Issue #56: the provenance gate\'s zoom-3 control arm '
                         '(provenance_gate.py --draw-control writes the file)')
    args = ap.parse_args(argv)

    if args.batch_size < 1:
        ap.error('--batch-size must be >= 1')
    if args.ids is not None:
        if args.verify:
            ap.error('--ids re-infers; it does not combine with --verify')
        if args.out is None:
            ap.error('--ids needs an explicit --out: its file is a control, not the run\'s '
                     'results.f01.jsonl')
    out_path = args.out or (args.run_dir / 'results.f01.jsonl')
    if args.write_band_file is not None and not args.verify:
        ap.error('--write-band-file requires --verify: the band file is built from the '
                 'per-pano comparison, so there is nothing to write without it.')
    if args.verify:
        import send_to_ps
        old_path = args.run_dir / 'results.jsonl'
        tier = server_tier(send_to_ps.load_submission_record(
            send_to_ps.submission_record_path(old_path)))
        try:
            old_dec = single_decode(decodes_in_file(old_path), old_path.name)
            new_dec = single_decode(decodes_in_file(out_path), out_path.name)
        except ValueError as e:
            raise SystemExit(str(e))
        if old_dec != new_dec:
            what = (f"{old_path.name} is '{old_dec}' and {out_path.name} is '{new_dec}' "
                    f"(issue #111): the decode moves every peak, so the pair cannot reproduce")
            if args.write_band_file is not None:
                raise SystemExit(f"{what}, and a band file built from it would ship its band in "
                                 f"another frame than the live labels. Re-infer with --decode "
                                 f"{old_dec}.")
            if not args.allow_mixed_decode:
                raise SystemExit(f"{what}. --allow-mixed-decode compares them as a diagnostic.")
            print(f"WARNING (--allow-mixed-decode): {what}.")
        try:
            old_bor = single_border(borders_in_file(old_path), old_path.name)
            new_bor = single_border(borders_in_file(out_path), out_path.name)
        except ValueError as e:
            raise SystemExit(str(e))
        if old_bor != new_bor:
            what = (f"{old_path.name} is '{old_bor}' and {out_path.name} is '{new_bor}' "
                    f"(issue #130): the border rule adds or drops every seam-band peak")
            if args.write_band_file is not None:
                raise SystemExit(f"{what}, and a band file built from it would ship seam-band "
                                 f"labels the live campaign never had. Re-infer with --border "
                                 f"{old_bor}.")
            if not args.allow_mixed_border:
                raise SystemExit(f"{what}. --allow-mixed-border compares them as a diagnostic.")
            print(f"WARNING (--allow-mixed-border): {what}.")
        summary, mismatches, carry_over = verify(old_path, out_path, floor=tier,
                                                 band_floor=args.band_floor)
        summary['tier'] = tier
        print(json.dumps(summary, indent=2))
        for pid, only_old, only_new in mismatches[:10]:
            print(f"  {pid}: old-only {only_old} new-only {only_new}")
        cc = summary['coarse_cell']
        if cc['panos']:
            print(f"coarse-cell diagnostic (#111; exact pixels still decide): of {cc['panos']} "
                  f"pano(s) whose keys differ, {cc['panos_agree_within_cell']} agree within +/-1 "
                  f"coarse cell; {cc['pairs']['flip']} key(s) moved 7-8 heatmap cells (an "
                  f"adjacent-coarse-cell flip), {cc['pairs']['grid_neighbour']} moved 1 cell.")
        bb = summary['border_band']
        if bb['keys_old'] or bb['keys_new']:
            print(f"border-band diagnostic (#130; exact pixels still decide): {bb['keys_old']} "
                  f"old-only and {bb['keys_new']} new-only key(s) lie in the 10-px border band "
                  f"(the 360-degree seam band, or the zenith/nadir rows); "
                  f"{bb['panos_only_in_band']} pano(s) differ ONLY there -- the border rule, "
                  f"not the model.")
        if summary['missing']:
            print(f"{summary['missing']} pano(s) of {old_path.name} are absent from "
                  f"{out_path.name}: finish the re-inference first.")
            raise SystemExit(1)
        if args.write_band_file is None:
            if carry_over:
                print(f"{len(carry_over)} pano(s) do not reproduce the old operational set at "
                      f"{tier}: do NOT ship a band from {out_path.name}. Use --write-band-file "
                      f"to build one that excludes them.")
                raise SystemExit(1)
            return
        band_path = (args.run_dir / 'results.band.jsonl'
                     if args.write_band_file is True else Path(args.write_band_file))
        if band_path.exists():
            raise SystemExit(f"{band_path} already exists; move it aside rather than "
                             f"overwriting a file a campaign may have been recorded against.")
        carried = write_band_file(old_path, out_path, band_path, carry_over, tier)
        try:
            record_path, endpoints = write_derived_record(old_path, band_path, tier)
        except BaseException:
            # A band file with no record beside it is the one file that must NOT be shipped,
            # sitting under exactly the name the docs say to ship from — and the "already
            # exists" guard above would then block the retry that would fix it. Every exit
            # from write_derived_record is a refusal, so take the file with it.
            band_path.unlink(missing_ok=True)
            raise
        band_labels = (send_to_ps.count_band_labels_in_file(band_path, args.band_floor, tier)
                       if args.band_floor is not None else None)
        print(f"-> {band_path.name}: {summary['old_panos']} record(s), {carried} carried over "
              f"from {old_path.name}; labels at {tier} verified identical to the live set.")
        print(f"-> {record_path.name}: derived for {len(endpoints)} endpoint(s).")
        if band_labels is not None:
            print(f"-> a band [{args.band_floor}, {tier}) from this file would ship "
                  f"{band_labels} label(s).")
        return

    import main  # noqa: E402
    reinfer(args.run_dir, out_path, args.workers or main.PROCESSING_CONCURRENCY, args.limit,
            None if args.ids is None else read_id_list(args.ids), batch_size=args.batch_size,
            decode=args.decode, border=args.border)


if __name__ == '__main__':
    main_cli()
