"""Backfill extended Mapillary provenance into existing records/results JSONL files.

Older runs (Richmond, Clovis) were produced before the pano record carried camera
make/model and the full `source_metadata` dump (see sources.mapillary.provenance_fields).
The pano id is immutable and IS the image identity, so the metadata is recoverable at any
time: this re-fetches it (one Graph API call per pano, NO image download) and merges the
same provenance fields a fresh run would write into each line, in place.

Idempotent and resumable: a line that already has `source_metadata` is left untouched, so
a re-run only fetches what's still missing, and the cache is appended as each fetch lands
so an interrupted pass keeps its work. A shared on-disk cache (`--cache`) lets the several
copies of one city (the makelab archive, the local run, the benchmark bundle) be backfilled
from a single fetch pass — point them all at the same cache file. The rewrite goes through
a temp file and an atomic replace, so a kill mid-write can't truncate the input.

    # one city, all three copies, one fetch pass:
    python scripts/backfill_metadata.py runs/clovis/results.jsonl --cache runs/clovis/.meta_cache.jsonl
    python scripts/backfill_metadata.py ../RampNet/benchmark/clovis/records.jsonl --cache runs/clovis/.meta_cache.jsonl

Only `source: mapillary` lines are touched; GSV lines pass through unchanged.
Needs MAPILLARY_ACCESS_TOKEN (from ./.env).

`--pose` is a second, OFFLINE pass (issue #42): it fills `camera_pitch`/`camera_roll` from
the `computed_rotation` each line already carries in `source_metadata`, through the same
geo.mapillary_pitch_roll a fresh run uses -- no token, no network. Project Sidewalk's
backup-image gate refuses a pano_data row with a null camera_pitch, so every Mapillary pano
submitted before #42 needs the value. Lines without `source_metadata` need the provenance
pass above first; lines whose rotation is missing, malformed or a failed reconstruction
stay null and are counted.

    python scripts/backfill_metadata.py runs/richmond/results.jsonl --pose --dry-run   # counts only
    python scripts/backfill_metadata.py runs/richmond/results.jsonl --pose --out runs/richmond/results.pose.jsonl

A file that has a `<file>.submission.json` or any `<file>.submitted*` /
`<file>.band-*.submitted` resume sidecar beside it is under send_to_ps.py's guard, and
rewriting it changes its sha256: the recorded campaign (and any band on it) would then
refuse to resume, and its position_check.json would go stale. So --pose refuses to write
over such a file -- in place, or as the `--out` target -- unless --rewrite-submitted says
that is intended; a fresh `--out` leaves the input untouched and is the route for a
pano-only push (`send_to_ps.py <out> --min-confidence 2.0`, which submits no labels).
Blank lines are copied through as blank lines, so a sidecar's line numbers still name the
same panos after a rewrite.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

import geo  # noqa: E402
from sources import mapillary  # noqa: E402

WORKERS = 8
# send_to_ps.py's campaign record, beside the file it describes (SUBMISSION_RECORD_SUFFIX
# there; not imported, since send_to_ps pulls in the whole submission stack).
SUBMISSION_RECORD_SUFFIX = ".submission.json"
# ...and its resume sidecars, which name lines of the file BY NUMBER: `<file>.submitted`,
# the `<file>.submitted.<endpoint>` copies the test->prod move leaves, and a band's
# `<file>.band-<min>-<max>.submitted`.
SUBMITTED_SIDECAR_SUFFIX = ".submitted"
BAND_SIDECAR_INFIX = ".band-"

# Outcomes of the --pose pass, per line. The first two change no angle (POSE_ALREADY may
# still stamp a missing camera_pose_source).
POSE_ALREADY, POSE_NOT_MAPILLARY, POSE_NO_METADATA, POSE_FILLED = (
    "already_set", "not_mapillary", "no_source_metadata", "filled")


def load_cache(path):
    """{pano_id: provenance_fields} from a prior pass; {} if none. The file is
    append-only, so a later row for the same id (a --force refresh) wins."""
    cache = {}
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["id"]] = row["fields"]
    return cache


def fetch_fields(pano_id):
    """Provenance fields for one pano, or None if its metadata is unavailable."""
    meta = mapillary._fetch_image_metadata(pano_id)
    if not meta:
        return None
    return mapillary.provenance_fields(meta)


def is_mapillary(record):
    return record["pano"].get("source", "gsv") == "mapillary"


def needs_backfill(record, force):
    """True when this line should get (re)written provenance."""
    return is_mapillary(record) and (force or not record["pano"].get("source_metadata"))


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_lines(path):
    """Every physical line of a JSONL: the parsed record, or None for a blank line.

    The --pose rewrite keeps blank lines as blank lines (iter_records drops them):
    send_to_ps.py's resume sidecars number PHYSICAL lines, blanks included, so dropping one
    would shift every later line onto a different pano under --rewrite-submitted."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line) if line.strip() else None


def submission_artifacts(path):
    """Names of send_to_ps.py's campaign state beside `path`, sorted: its
    `.submission.json` record and every resume sidecar (`.submitted`, `.submitted.*`,
    `.band-*.submitted`). Empty when there is none (or the directory does not exist)."""
    path = Path(path)
    if not path.parent.is_dir():
        return []
    name = path.name
    return sorted(q.name for q in path.parent.iterdir()
                  if q.name == name + SUBMISSION_RECORD_SUFFIX
                  or q.name.startswith(name + SUBMITTED_SIDECAR_SUFFIX)
                  or (q.name.startswith(name + BAND_SIDECAR_INFIX)
                      and q.name.endswith(SUBMITTED_SIDECAR_SUFFIX)))


def fill_pose(record):
    """Fill one line's camera_pitch/camera_roll from its own source_metadata, in place.

    Returns the outcome: POSE_FILLED, or why not -- POSE_NOT_MAPILLARY, POSE_ALREADY (both
    already set; a fresh run since #42 wrote them, so they are never recomputed), POSE_NO_
    METADATA (run the provenance backfill first), or one of geo's POSE_MISSING /
    POSE_MALFORMED / POSE_OVER_TILT, which leave both null.
    """
    pano = record["pano"]
    if not is_mapillary(record):
        return POSE_NOT_MAPILLARY
    if pano.get("camera_pitch") is not None and pano.get("camera_roll") is not None:
        # A block written between the pose write and camera_pose_source (#42) has the
        # same derived angles and lacks only the provenance key.
        if pano.get("camera_pose_source") is None:
            pano["camera_pose_source"] = mapillary.POSE_SOURCE
        return POSE_ALREADY
    # Every Mapillary line carries the key, null until there are angles to attribute --
    # the shape sources/mapillary.build_pano_record writes.
    pano.setdefault("camera_pose_source", None)
    meta = pano.get("source_metadata")
    if not meta:
        return POSE_NO_METADATA
    status = geo.mapillary_pose_status(meta.get("computed_rotation"))
    if status != geo.POSE_OK:
        return status
    pano["camera_pitch"], pano["camera_roll"] = geo.mapillary_pitch_roll(meta["computed_rotation"])
    pano["camera_pose_source"] = mapillary.POSE_SOURCE
    return POSE_FILLED


def backfill_pose(src, out=None, dry_run=False):
    """The offline --pose pass over one JSONL: {outcome: line count}.

    Writes `out` (or `src` in place) through a temp file and an atomic replace, one line
    per input line in the same order, so line numbers -- which send_to_ps.py's resume
    sidecars are made of -- still mean the same panos. dry_run writes nothing.
    """
    counts = {}
    dest = out or src
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    sink = None if dry_run else open(tmp, "w", encoding="utf-8", newline="\n")
    try:
        for rec in iter_lines(src):
            if rec is None:                      # a blank line stays a line: numbering holds
                if sink:
                    sink.write("\n")
                continue
            outcome = fill_pose(rec)
            counts[outcome] = counts.get(outcome, 0) + 1
            if sink:
                sink.write(json.dumps(rec) + "\n")
    finally:
        if sink:
            sink.close()
    if not dry_run:
        tmp.replace(dest)
    return counts


def pose_main(args):
    """--pose: refuse to write over a submitted file (in place or as --out), then run."""
    if args.out is not None and args.out.resolve() == args.jsonl.resolve():
        sys.exit("--out is the input file; drop --out to rewrite in place.")
    dest = args.out or args.jsonl
    held = [] if args.dry_run else submission_artifacts(dest)
    if held and not args.rewrite_submitted:
        what = "rewriting it in place" if args.out is None else "overwriting it as --out"
        sys.exit(f"{dest} has send_to_ps.py campaign state beside it ({', '.join(held)}): "
                 f"{what} changes the sha256 send_to_ps.py's guard holds, so that campaign "
                 f"(and any band on it) would refuse to resume and its position check would "
                 f"go stale. Write a NEW file with --out instead (and submit that, e.g. "
                 f"`send_to_ps.py <out> --min-confidence 2.0` for a pano-only update), or "
                 f"pass --rewrite-submitted if overwriting it is really intended.")
    counts = backfill_pose(args.jsonl, args.out, args.dry_run)
    order = [POSE_FILLED, POSE_ALREADY, POSE_NO_METADATA, geo.POSE_MISSING,
             geo.POSE_MALFORMED, geo.POSE_OVER_TILT, POSE_NOT_MAPILLARY]
    total = sum(counts.values())
    print(f"{args.jsonl.name}: {total} lines -- "
          + ", ".join(f"{k} {counts[k]}" for k in order if counts.get(k)))
    print("  DRY RUN: nothing written." if args.dry_run
          else f"  -> {args.out or args.jsonl}")
    if counts.get(POSE_NO_METADATA):
        print(f"  {counts[POSE_NO_METADATA]} Mapillary lines carry no source_metadata: run the "
              f"provenance pass (this script without --pose) first.")
    return counts


def main():
    ap = argparse.ArgumentParser(description="Backfill Mapillary provenance into a JSONL in place.")
    ap.add_argument("jsonl", type=Path, help="A records.jsonl / results.jsonl to update in place.")
    ap.add_argument("--cache", type=Path,
                    help="Shared fetch cache (JSONL of {id, fields}); reused across a city's copies.")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch and rewrite lines that already have source_metadata "
                         "(default: skip them). Ignores any cached fields for those panos, "
                         "so this really does go back to the API.")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"Concurrent metadata fetches (default: {WORKERS}). Mapillary's "
                         "60k req/min limit leaves ample headroom for a city-scale backfill.")
    ap.add_argument("--pose", action="store_true",
                    help="Offline pass (no token, no network): fill camera_pitch/camera_roll "
                         "from each line's own source_metadata.computed_rotation (#42).")
    ap.add_argument("--out", type=Path,
                    help="--pose only: write the result here instead of rewriting the input.")
    ap.add_argument("--dry-run", action="store_true",
                    help="--pose only: count what would change; write nothing.")
    ap.add_argument("--rewrite-submitted", action="store_true",
                    help="--pose only: allow overwriting (in place, or as the --out "
                         "target) a file with a send_to_ps.py submission record or resume "
                         "sidecar beside it (see the module docstring).")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"No such file: {args.jsonl}")
    if args.pose:
        pose_main(args)
        return
    if args.out or args.dry_run or args.rewrite_submitted:
        ap.error("--out, --dry-run and --rewrite-submitted apply only to --pose.")
    # Fail fast: without a token every fetch burns its full retry/backoff budget and then
    # reports "no metadata", which looks like decay rather than a missing credential.
    mapillary.prepare()

    # --force means the cached copy is exactly what we don't trust, so ignore it.
    cache = {} if args.force else load_cache(args.cache)

    # Which mapillary panos still need a fetch (not already done, not already cached).
    need, total = [], 0
    for rec in iter_records(args.jsonl):
        total += 1
        if needs_backfill(rec, args.force) and rec["pano"]["panorama_id"] not in cache:
            need.append(rec["pano"]["panorama_id"])
    need = list(dict.fromkeys(need))

    print(f"{args.jsonl.name}: {total} lines, {len(need)} panos to fetch "
          f"({len(cache)} already cached).")

    if need:
        failures = 0
        # The cache is appended as results land, not at the end: a city-scale pass is
        # hours long, and an interrupted one must not throw away what it already fetched.
        cache_file = open(args.cache, "a", encoding="utf-8") if args.cache else None
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(fetch_fields, pid): pid for pid in need}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Fetching metadata"):
                    pid = futs[fut]
                    fields = fut.result()
                    if fields is None:
                        failures += 1
                        continue
                    cache[pid] = fields
                    if cache_file:
                        cache_file.write(json.dumps({"id": pid, "fields": fields}) + "\n")
                        cache_file.flush()
        finally:
            if cache_file:
                cache_file.close()
        if failures:
            print(f"  WARNING: {failures} panos had no metadata (skipped, will retry next run).")

    # Merge cached fields into every mapillary line, streaming into a temp file that
    # atomically replaces the original — an interrupted rewrite leaves the input intact.
    updated = skipped_gsv = missing = 0
    tmp = args.jsonl.with_suffix(args.jsonl.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as out:  # LF on every platform
        for rec in iter_records(args.jsonl):
            if not is_mapillary(rec):
                skipped_gsv += 1
            elif needs_backfill(rec, args.force):
                fields = cache.get(rec["pano"]["panorama_id"])
                if fields is None:
                    missing += 1
                else:
                    rec["pano"].update(fields)
                    updated += 1
            out.write(json.dumps(rec) + "\n")
    tmp.replace(args.jsonl)

    print(f"  updated {updated} lines"
          + (f", {skipped_gsv} gsv untouched" if skipped_gsv else "")
          + (f", {missing} still missing metadata" if missing else "")
          + f" -> {args.jsonl}")


if __name__ == "__main__":
    main()
