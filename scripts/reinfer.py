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

`--write-band-file` then writes the file a band may actually ship from: the NEW record for
every pano that reproduced, and the OLD record for every pano that did not. The old records
hold nothing below the tier the server has, so their band is empty and `send_to_ps.py` skips
them without a POST. That gives the file one property, which is asserted before it is left
on disk: **its labels at and above the server's tier are exactly the ones already live**. A
derived `<band file>.submission.json` therefore describes a real campaign, and the ordinary
band guard applies with no override.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import BENCHMARK_CONFIDENCE  # noqa: E402

# The pano fields a band must not change: the first two set the pixel key PS stores, the
# rest are what every label on that pano inherits (SidewalkWebpage#5361).
PANO_INVARIANTS = ('width', 'height', 'lat', 'lng', 'camera_heading')


def read_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pixel_set(record, floor):
    w, h = record['pano']['width'], record['pano']['height']
    return {(round(d['x_normalized'] * w), round(d['y_normalized'] * h))
            for d in record['detections'] if d['confidence'] >= floor}


def pano_drift(old, new):
    """Which of PANO_INVARIANTS differ between two records' pano blocks, as (field, old, new)."""
    return [(f, old['pano'].get(f), new['pano'].get(f))
            for f in PANO_INVARIANTS if old['pano'].get(f) != new['pano'].get(f)]


def verify(old_path, new_path, floor=BENCHMARK_CONFIDENCE, band_floor=None):
    """Compare the new file to the old one at `floor`.

    Returns (summary, mismatches, carry_over) where `carry_over` is the set of pano ids
    that did NOT reproduce - the ids a band file must take from the old file.
    """
    new_by_id = {r['pano']['panorama_id']: r for r in read_records(new_path)}
    summary = {'old_panos': 0, 'new_panos': len(new_by_id), 'missing': 0, 'exact': 0,
               'mismatch': 0, 'pano_drift': 0, 'old_labels': 0}
    if band_floor is not None:
        summary['band_labels'] = 0
    mismatches = []
    carry_over = set()
    for old in read_records(old_path):
        pid = old['pano']['panorama_id']
        summary['old_panos'] += 1
        old_set = pixel_set(old, floor)
        summary['old_labels'] += len(old_set)
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
                summary['band_labels'] += sum(
                    1 for d in new['detections'] if band_floor <= d['confidence'] < floor)
            continue
        carry_over.add(pid)
        if drift:
            summary['pano_drift'] += 1
            mismatches.append((pid, [f"{f}: {o!r} -> {n!r}" for f, o, n in drift], []))
        if new_set != old_set:
            summary['mismatch'] += 1
            mismatches.append((pid, sorted(old_set - new_set), sorted(new_set - old_set)))
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
    old_keys = {(r['pano']['panorama_id'], px) for r in read_records(old_path)
                for px in pixel_set(r, tier)}
    band_keys = {(r['pano']['panorama_id'], px) for r in read_records(band_path)
                 for px in pixel_set(r, tier)}
    if band_keys != old_keys:
        band_path.unlink()
        raise SystemExit(
            f"{band_path.name} does not reproduce {old_path.name} at {tier} "
            f"({len(old_keys - band_keys)} missing, {len(band_keys - old_keys)} extra); "
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
    endpoints = {}
    for url, state in old_record['endpoints'].items():
        sent = state.get('submitted_lines', 0)
        if not isinstance(old_lines, int) or sent < old_lines:
            raise SystemExit(f"{url} holds {sent} of {old_lines} line(s) of {old_path.name}; a "
                             f"band only sits on a COMPLETE campaign. Finish it first.")
        recounted = send_to_ps.count_band_labels_in_file(band_path, tier, float('inf'))
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
    tmp_path = record_path.with_name(record_path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    tmp_path.replace(record_path)
    return record_path, endpoints


def reinfer(run_dir, out_path, workers, limit):
    import main  # noqa: E402  (loads torch lazily below, like main.py does)
    from sources import get_source
    from detectors.curb_ramp import CurbRampDetector

    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    source_name = manifest.get('imagery_source') or manifest.get('source') or 'gsv'
    source = get_source(source_name)
    if source_name == 'mapillary':
        source.POSITION_FIELD = manifest.get('mapillary_position', 'sfm')
    source.prepare()
    main.curb_ramp_detector = CurbRampDetector()

    cache_path = Path(f"{out_path}.processed")
    done = set()
    if cache_path.exists():
        done = {l.strip() for l in cache_path.read_text(encoding='utf-8').splitlines() if l.strip()}
    todo = [(r['pano']['panorama_id'], r['pano']['lat'], r['pano']['lng'])
            for r in read_records(run_dir / 'results.jsonl')
            if r['pano']['panorama_id'] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"-> {len(done)} already done, {len(todo)} to re-infer from {source_name} "
          f"(storage floor {main.DETECTION_STORAGE_FLOOR}) -> {out_path}")

    counts = {'success': 0, 'skipped': 0, 'failed': 0}
    # Futures are consumed in submission order so the output keeps the input's line order;
    # downloads still overlap (the pool runs ahead) and the GPU serialises inference.
    with open(cache_path, 'a', encoding='utf-8') as f_cache, \
         open(out_path, 'a', encoding='utf-8') as f_out, \
         ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(main.process_pano, source, pid, lat, lon) for pid, lat, lon in todo]
        try:
            from tqdm import tqdm
            it = tqdm(futures, desc="Re-inferring")
        except ImportError:
            it = futures
        for future in it:
            counts[main.handle_result(future.result(), f_cache, f_out)] += 1
    print(f"-> Re-inference: {counts['success']} written, {counts['skipped']} skipped "
          f"(no longer served / unusable), {counts['failed']} failed (retry by re-running).")


def main_cli():
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
    ap.add_argument('--limit', type=int, default=None, help='re-infer at most this many panos')
    args = ap.parse_args()

    out_path = args.out or (args.run_dir / 'results.f01.jsonl')
    if args.write_band_file is not None and not args.verify:
        ap.error('--write-band-file requires --verify: the band file is built from the '
                 'per-pano comparison, so there is nothing to write without it.')
    if args.verify:
        import send_to_ps
        old_path = args.run_dir / 'results.jsonl'
        tier = server_tier(send_to_ps.load_submission_record(
            send_to_ps.submission_record_path(old_path)))
        summary, mismatches, carry_over = verify(old_path, out_path, floor=tier,
                                                 band_floor=args.band_floor)
        summary['tier'] = tier
        print(json.dumps(summary, indent=2))
        for pid, only_old, only_new in mismatches[:10]:
            print(f"  {pid}: old-only {only_old} new-only {only_new}")
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
        record_path, endpoints = write_derived_record(old_path, band_path, tier)
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
    reinfer(args.run_dir, out_path, args.workers or main.PROCESSING_CONCURRENCY, args.limit)


if __name__ == '__main__':
    main_cli()
