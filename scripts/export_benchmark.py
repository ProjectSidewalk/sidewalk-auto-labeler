"""Fetch native-resolution panos for a validation benchmark bundle.

Reads a bundle's records.jsonl (source + pano id per validated pano) and downloads
each pano at NATIVE resolution — Mapillary `thumb_original` with no downscale, GSV at
max zoom — into <bundle>/panos/<id>.jpg. Resumable: existing files are skipped, so a
flaky run can just be re-run. records.jsonl + verdicts.json already hold the (image-free)
scoring data; these native panos are the imagery half of the HF benchmark (RampNet #21)
and the input the GT labeling tool renders (#26).

After every run (including a fully-resumed one) it reconciles the archive against the
records it was built from — so "is this the same data we processed?" is always answerable
mechanically — and writes two files next to the panos dir:
  - index.csv   : panorama_id,filename,bytes,sha256 for every archived pano (durable
                  integrity manifest; built incrementally, unchanged panos aren't re-hashed)
  - decayed.txt : record ids whose imagery is gone from the source since the run (if any)
The pano id IS the identity — same id means the same immutable source image — so the archive
matches the run exactly, minus any explicitly-listed decayed ids. Exits non-zero only on
contamination (a pano file with no matching record); decay is expected and not a failure.

    # benchmark bundle (writes panos/ next to records.jsonl):
    python scripts/export_benchmark.py D:/Git/RampNet/benchmark/richmond

    # full archive of every processed pano (e.g. onto makelab2) — point it at a run's
    # results.jsonl and an output dir (#17):
    python scripts/export_benchmark.py runs/richmond/results.jsonl --out /data/makelab2/richmond

Needs MAPILLARY_ACCESS_TOKEN (from ./.env) for Mapillary panos.
"""
import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from sources import mapillary  # noqa: E402

WORKERS = 6


def fetch_native(pano_id, source, out_path):
    """Download one pano at native resolution to out_path. Returns None on success,
    else an error string."""
    try:
        if source == "mapillary":
            meta = mapillary._fetch_image_metadata(pano_id)
            url = (meta or {}).get("thumb_original_url")
            if not url:
                return "no thumb_original_url (image gone?)"
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)          # raw native bytes, no resize
        else:  # gsv
            from streetlevel import streetview  # lazy: Mapillary-only archives don't need it
            meta = streetview.find_panorama_by_id(pano_id)
            if meta is None:
                return "metadata unavailable"
            img = streetview.get_panorama(meta, zoom=len(meta.image_sizes) - 1)  # max zoom
            img.save(out_path, "JPEG", quality=95)
        return None
    except Exception as e:
        return str(e)


def _sha256(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def reconcile(records_path, panos_dir, expected):
    """Prove the archive matches the records it was built from, and write a durable
    integrity record. Runs after every fetch (even a fully-resumed one), so the question
    "is this the same data we processed?" is always answerable mechanically.

    `expected` is the list of panorama_ids from the records (in file order; may repeat).
    Writes, next to the panos dir:
      - index.csv   : panorama_id,filename,bytes,sha256 for every archived pano
      - decayed.txt : record ids with no pano file (imagery gone since the run), if any
    index.csv is built incrementally — a pano already recorded with a matching size is not
    re-hashed on a resumed run, so only newly fetched panos pay the hashing cost.
    Returns (missing_count, extra_count); 0/0 means a clean 1:1 archive."""
    expected_ids = set(expected)
    present = {p.stem: p for p in panos_dir.glob("*.jpg")}
    present_ids = set(present)

    verified = expected_ids & present_ids          # archived and backed by a record
    missing = expected_ids - present_ids           # decayed / never fetched
    extra = present_ids - expected_ids             # files with no matching record

    bundle_dir = panos_dir.parent
    index_path = bundle_dir / "index.csv"

    prior = {}
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prior[row["panorama_id"]] = row

    rows = []
    for pid in tqdm(sorted(verified), desc="Hashing archive", unit="pano"):
        p = present[pid]
        size = p.stat().st_size
        cached = prior.get(pid)
        if cached and int(cached["bytes"]) == size and cached.get("sha256"):
            sha = cached["sha256"]                  # unchanged since last index — reuse
        else:
            sha = _sha256(p)
        rows.append((pid, p.name, size, sha))

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panorama_id", "filename", "bytes", "sha256"])
        w.writerows(rows)

    decayed_path = bundle_dir / "decayed.txt"
    if missing:
        decayed_path.write_text("".join(pid + "\n" for pid in sorted(missing)), encoding="utf-8")
    elif decayed_path.exists():
        decayed_path.unlink()                       # a resume recovered them — clear stale list

    print(f"--- Reconcile: {records_path.name} vs {panos_dir} ---")
    print(f"  records (unique panos): {len(expected_ids)}")
    print(f"  archived + verified:    {len(rows)}  -> {index_path.name}")
    print(f"  missing (decayed):      {len(missing)}" + (f"  -> {decayed_path.name}" if missing else ""))
    if extra:
        print(f"  WARNING: {len(extra)} pano file(s) not in records, e.g. {sorted(extra)[:3]}")
    if extra:
        status = f"ANOMALY — {len(extra)} file(s) not backed by a record"
    elif missing:
        status = f"OK with decay — {len(missing)} pano(s) gone from Mapillary since the run (recorded in {decayed_path.name})"
    else:
        status = "OK — archive matches records 1:1"
    print(f"  STATUS: {status}")
    return len(missing), len(extra)


def main():
    ap = argparse.ArgumentParser(description="Fetch native-res panos for a benchmark bundle or full run.")
    ap.add_argument("source", help="A bundle dir (with records.jsonl) OR a records/results .jsonl file.")
    ap.add_argument("--out", type=Path,
                    help="Output dir for panos. Default: <bundle>/panos when SOURCE is a bundle dir; "
                         "required when SOURCE is a .jsonl file (e.g. a makelab2 path).")
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_dir():
        records_path = src / "records.jsonl"
        panos_dir = args.out or (src / "panos")
    else:
        records_path = src
        if args.out is None:
            sys.exit("--out is required when SOURCE is a .jsonl file")
        panos_dir = args.out
    if not records_path.exists():
        sys.exit(f"No records file at {records_path}")
    panos_dir.mkdir(parents=True, exist_ok=True)

    expected = []   # every panorama_id in the records — the set the archive must match
    todo = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)["pano"]
            pid, source = p["panorama_id"], p.get("source", "gsv")
            expected.append(pid)
            out = panos_dir / f"{pid}.jpg"
            if not out.exists():
                todo.append((pid, source, out))

    label = src.name
    print(f"{label}: {len(todo)} panos to fetch -> {panos_dir} "
          f"({len(list(panos_dir.glob('*.jpg')))} already present)")

    if todo:
        failures = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(fetch_native, pid, source, out): pid for pid, source, out in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Fetching {label}"):
                err = fut.result()
                if err:
                    failures.append((futs[fut], err))
        print(f"Done: {len(todo) - len(failures)} fetched, {len(failures)} failed.")
        for pid, err in failures:
            print(f"  FAILED {pid}: {err}")

    # Always verify — reconcile the archive against the records and (re)write the integrity
    # index, even when nothing was fetched this pass (a resumed / already-complete run).
    # Decayed panos are expected on Mapillary (recorded in decayed.txt, not a failure);
    # only genuine contamination — files with no matching record — exits non-zero.
    _missing, extra = reconcile(records_path, panos_dir, expected)
    if extra:
        sys.exit(1)


if __name__ == "__main__":
    main()
