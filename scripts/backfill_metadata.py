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

from sources import mapillary  # noqa: E402

WORKERS = 8


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
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"No such file: {args.jsonl}")
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
