"""Backfill extended Mapillary provenance into existing records/results JSONL files.

Older runs (Richmond, Clovis) were produced before the pano record carried camera
make/model and the full `source_metadata` dump (see sources.mapillary.provenance_fields).
The pano id is immutable and IS the image identity, so the metadata is recoverable at any
time: this re-fetches it (one Graph API call per pano, NO image download) and merges the
same provenance fields a fresh run would write into each line, in place.

Idempotent and resumable: a line that already has `source_metadata` is left untouched, so
a re-run only fetches what's still missing. A shared on-disk cache (`--cache`) lets the
several copies of one city (the makelab archive, the local run, the benchmark bundle) be
backfilled from a single fetch pass — point them all at the same cache file.

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
    """{pano_id: provenance_fields} from a prior pass; {} if none."""
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


def main():
    ap = argparse.ArgumentParser(description="Backfill Mapillary provenance into a JSONL in place.")
    ap.add_argument("jsonl", type=Path, help="A records.jsonl / results.jsonl to update in place.")
    ap.add_argument("--cache", type=Path,
                    help="Shared fetch cache (JSONL of {id, fields}); reused across a city's copies.")
    ap.add_argument("--force", action="store_true",
                    help="Re-backfill lines that already have source_metadata (default: skip them).")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"Concurrent metadata fetches (default: {WORKERS}). Mapillary's "
                         "60k req/min limit leaves ample headroom for a city-scale backfill.")
    args = ap.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"No such file: {args.jsonl}")

    lines = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    cache = load_cache(args.cache)

    # Which mapillary panos still need a fetch (not already done, not already cached).
    need = []
    for rec in lines:
        p = rec["pano"]
        if p.get("source", "gsv") != "mapillary":
            continue
        if p.get("source_metadata") and not args.force:
            continue
        pid = p["panorama_id"]
        if pid not in cache:
            need.append(pid)

    print(f"{args.jsonl.name}: {len(lines)} lines, {len(need)} panos to fetch "
          f"({len(cache)} already cached).")

    if need:
        new_cache_rows = []
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(fetch_fields, pid): pid for pid in dict.fromkeys(need)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Fetching metadata"):
                pid = futs[fut]
                fields = fut.result()
                if fields is None:
                    failures += 1
                    continue
                cache[pid] = fields
                new_cache_rows.append({"id": pid, "fields": fields})
        if args.cache and new_cache_rows:
            with open(args.cache, "a", encoding="utf-8") as f:
                for row in new_cache_rows:
                    f.write(json.dumps(row) + "\n")
        if failures:
            print(f"  WARNING: {failures} panos had no metadata (skipped, will retry next run).")

    # Merge cached fields into every mapillary line, then rewrite the file atomically.
    updated = skipped_gsv = missing = 0
    for rec in lines:
        p = rec["pano"]
        if p.get("source", "gsv") != "mapillary":
            skipped_gsv += 1
            continue
        if p.get("source_metadata") and not args.force:
            continue
        fields = cache.get(p["panorama_id"])
        if fields is None:
            missing += 1
            continue
        p.update(fields)
        updated += 1

    tmp = args.jsonl.with_suffix(args.jsonl.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:  # LF on every platform
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(args.jsonl)

    print(f"  updated {updated} lines"
          + (f", {skipped_gsv} gsv untouched" if skipped_gsv else "")
          + (f", {missing} still missing metadata" if missing else "")
          + f" -> {args.jsonl}")


if __name__ == "__main__":
    main()
