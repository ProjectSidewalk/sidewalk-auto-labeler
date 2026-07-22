"""Fetch native-resolution panos for a validation benchmark bundle.

Reads a bundle's records.jsonl (source + pano id per validated pano) and downloads
each pano at NATIVE resolution — Mapillary `thumb_original` with no downscale, GSV at
max zoom — into <bundle>/panos/<id>.jpg. Resumable: existing files are skipped, so a
flaky run can just be re-run. records.jsonl + verdicts.json already hold the (image-free)
scoring data; these native panos are the imagery half of the HF benchmark (RampNet #21)
and the input the GT labeling tool renders (#26).

    # benchmark bundle (writes panos/ next to records.jsonl):
    python scripts/export_benchmark.py D:/Git/RampNet/benchmark/richmond

    # full archive of every processed pano (e.g. onto makelab2) — point it at a run's
    # results.jsonl and an output dir (#17):
    python scripts/export_benchmark.py runs/richmond/results.jsonl --out /data/makelab2/richmond

Needs MAPILLARY_ACCESS_TOKEN (from ./.env) for Mapillary panos.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from streetlevel import streetview
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
            meta = streetview.find_panorama_by_id(pano_id)
            if meta is None:
                return "metadata unavailable"
            img = streetview.get_panorama(meta, zoom=len(meta.image_sizes) - 1)  # max zoom
            img.save(out_path, "JPEG", quality=95)
        return None
    except Exception as e:
        return str(e)


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

    todo = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)["pano"]
            pid, source = p["panorama_id"], p.get("source", "gsv")
            out = panos_dir / f"{pid}.jpg"
            if not out.exists():
                todo.append((pid, source, out))

    label = src.name
    print(f"{label}: {len(todo)} panos to fetch -> {panos_dir} "
          f"({len(list(panos_dir.glob('*.jpg')))} already present)")
    if not todo:
        return

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


if __name__ == "__main__":
    main()
