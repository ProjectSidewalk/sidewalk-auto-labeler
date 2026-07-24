"""Build a validation benchmark bundle: sample panos from a run, fetch them native-res.

Reads a bundle's records.jsonl (source + pano id per validated pano) and downloads
each pano at NATIVE resolution — Mapillary `thumb_original` with no downscale, GSV at
max zoom — into <bundle>/panos/<id>.jpg. Resumable: existing files are skipped, so a
flaky run can just be re-run. records.jsonl + verdicts.json already hold the (image-free)
scoring data; these native panos are the imagery half of the HF benchmark (RampNet #21)
and the input the GT labeling tool renders (#26).

With `--bundle`, it also *creates* that records.jsonl: a spatially de-clustered sample of
a run's results.jsonl (`choose_panos`), each record tagged with its sampling stratum as
`benchmark_group` (top / random / empty) so RampNet's scorer can exclude the always-included
densest panos from unbiased estimates. This is the labeler's half of the repo split — it
owns "which panos, and their pixels"; RampNet owns ground truth and scoring.

After every run (including a fully-resumed one) it reconciles the archive against the
records it was built from — so "is this the same data we processed?" is always answerable
mechanically — and writes two files next to the panos dir:
  - index.csv   : panorama_id,filename,bytes,sha256 for every archived pano (durable
                  integrity manifest; built incrementally, unchanged panos aren't re-hashed)
  - decayed.txt : record ids whose imagery is gone from the source since the run (if any)
The pano id IS the identity — same id means the same immutable source image — so the archive
matches the run exactly, minus any explicitly-listed decayed ids. Exits non-zero only on
contamination (a pano file with no matching record); decay is expected and not a failure.

    # new benchmark bundle from a finished run (samples, then fetches its panos):
    python scripts/export_benchmark.py runs/clovis/results.jsonl \
        --bundle D:/Git/RampNet/benchmark/clovis --sample 100 --empty-sample 25

    # existing bundle (writes panos/ next to records.jsonl):
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
import math
import random
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

TOP_N_BY_COUNT = 5     # the densest panos are always included (group "top")

# Two validation panos closer than this almost certainly show the same physical
# curb ramps, so the sampler keeps selected panos at least this far apart: a
# reviewer never judges the same ramps twice, and precision/recall aren't inflated
# by correlated duplicates. Tunable per city/imagery via --min-spacing; deliberately
# well above the Mapillary thinning spacing (~5 m) since that's coverage, not sampling.
DEFAULT_MIN_SPACING_M = 30


# --- Sampling: which panos become the benchmark ---------------------------------------

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _coords(record):
    """(lat, lng) for a record, or None if it carries no position."""
    p = record['pano']
    lat, lng = p.get('lat'), p.get('lng')
    return (lat, lng) if lat is not None and lng is not None else None


class _SpatialIndex:
    """Grid of accepted points for fast 'is anything within min_spacing?' checks.

    Cell size == min_spacing, so any point within range lies in one of the nine
    neighbouring cells — the acceptance test touches a handful of points, not the
    whole accepted set, so selection stays cheap on city-sized candidate pools.
    """

    def __init__(self, min_spacing):
        self.s = max(min_spacing, 1e-9)
        self.cells = {}

    def _key(self, lat, lng):
        clat = self.s / 111320.0
        clng = self.s / (111320.0 * max(0.01, math.cos(math.radians(lat))))
        return (math.floor(lat / clat), math.floor(lng / clng))

    def far_enough(self, lat, lng):
        kx, ky = self._key(lat, lng)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for plat, plng in self.cells.get((kx + dx, ky + dy), ()):
                    if _haversine_m(lat, lng, plat, plng) < self.s:
                        return False
        return True

    def add(self, lat, lng):
        self.cells.setdefault(self._key(lat, lng), []).append((lat, lng))


def _spread(candidates, k, index, min_spacing):
    """Greedily take up to k candidates (in the given order) that sit at least
    min_spacing metres from every already-accepted point in the shared `index`.

    Records without coordinates are accepted without constraint (graceful
    degradation — a coordinate-less run can't be de-clustered, but must still
    render). Returns (picked, rejected_for_spacing)."""
    picked, rejected = [], 0
    for rec in candidates:
        if len(picked) >= k:
            break
        c = _coords(rec)
        if c is None:
            picked.append(rec)
        elif min_spacing <= 0 or index.far_enough(*c):
            picked.append(rec)
            index.add(*c)
        else:
            rejected += 1
    return picked, rejected


def choose_panos(records, sample, empty_sample, seed, min_spacing=DEFAULT_MIN_SPACING_M):
    """Return [(record, group)] for a spatially de-clustered validation sample.

    Three strata, all held at least `min_spacing` metres apart — *across* strata as
    well as within — so a reviewer never judges the same physical ramps twice and
    the precision/recall estimate isn't inflated by correlated duplicates:
      - 'top':    the densest *distinct* intersections (up to TOP_N_BY_COUNT) — a
                  dense-scene stress test, excluded from unbiased scoring.
      - 'random': a spatially spread sample of panos with detections.
      - 'empty':  a spatially spread sample of zero-detection panos (for recall).

    Deterministic given `seed`. `min_spacing=0` disables spacing (the original pure
    random behaviour). Records without lat/lng are included unconstrained. The method
    is city-agnostic by design — it's meant to seed every city's ground-truth set.
    """
    rng = random.Random(seed)
    with_det = [r for r in records if r['detections']]
    without_det = [r for r in records if not r['detections']]
    index = _SpatialIndex(min_spacing)

    # top: densest first, but only one pano per distinct location.
    by_density = sorted(with_det, key=lambda r: len(r['detections']), reverse=True)
    top, _ = _spread(by_density, TOP_N_BY_COUNT, index, min_spacing)
    top_ids = {r['pano']['panorama_id'] for r in top}

    # random: spread across the remaining detection panos.
    rest = [r for r in with_det if r['pano']['panorama_id'] not in top_ids]
    rng.shuffle(rest)
    n_random = max(0, sample - len(top))
    random_sel, rej_r = _spread(rest, n_random, index, min_spacing)

    # empty: spread across the zero-detection panos.
    empties = list(without_det)
    rng.shuffle(empties)
    empty_sel, rej_e = _spread(empties, empty_sample, index, min_spacing)

    # No silent caps: if spacing (not scarcity) kept us short of a target, say so.
    if min_spacing > 0 and rej_r and len(random_sel) < n_random:
        print(f"  spatial sampling: kept {len(random_sel)}/{n_random} detection panos "
              f"at >= {min_spacing} m spacing (area too dense for more).")
    if min_spacing > 0 and rej_e and len(empty_sel) < empty_sample:
        print(f"  spatial sampling: kept {len(empty_sel)}/{empty_sample} empty panos "
              f"at >= {min_spacing} m spacing.")

    return ([(r, 'top') for r in top]
            + [(r, 'random') for r in random_sel]
            + [(r, 'empty') for r in empty_sel])


def write_bundle_records(results_path, bundle_dir, sample, empty_sample, seed, min_spacing):
    """Sample `results_path` into `bundle_dir`/records.jsonl and return that path.

    Each record keeps its Stage-1 shape plus `benchmark_group` (the sampling stratum),
    so the bundle carries its own strata — RampNet's labeler/scorer doesn't have to
    recover them from a verdicts file that doesn't exist yet for a fresh city.

    An existing records.jsonl is never re-sampled: once a reviewer has judged a bundle,
    silently changing which panos are in it would invalidate their verdicts. Delete the
    file to draw a new sample.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    records_path = bundle_dir / "records.jsonl"
    if records_path.exists():
        n = sum(1 for line in open(records_path, encoding="utf-8") if line.strip())
        print(f"{records_path} already exists ({n} records) — keeping that sample "
              f"(sampling flags ignored; delete it to re-sample).")
        return records_path

    with open(results_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    chosen = choose_panos(records, sample, empty_sample, seed, min_spacing)
    chosen.sort(key=lambda rg: rg[0]['pano']['panorama_id'])

    with open(records_path, "w", encoding="utf-8", newline="\n") as f:  # LF on every platform
        for rec, group in chosen:
            f.write(json.dumps({**rec, "benchmark_group": group}) + "\n")

    groups = {g: sum(1 for _, gg in chosen if gg == g) for g in ("top", "random", "empty")}
    meta = {"source_records": str(results_path), "source_record_count": len(records),
            "sample": sample, "empty_sample": empty_sample, "seed": seed,
            "min_spacing_m": min_spacing, "selected": len(chosen), "groups": groups}
    (bundle_dir / "sample.json").write_text(json.dumps(meta, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    print(f"Sampled {len(chosen)} of {len(records)} panos -> {records_path} "
          f"(top {groups['top']} / random {groups['random']} / empty {groups['empty']})")
    return records_path


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
    ap.add_argument("--bundle", type=Path,
                    help="Build a benchmark bundle here from SOURCE (a run's results.jsonl): "
                         "samples panos into <bundle>/records.jsonl, then fetches them into "
                         "<bundle>/panos. An existing records.jsonl is reused, never re-sampled.")
    ap.add_argument("--sample", type=int, default=100,
                    help="With --bundle: detection panos to sample (default: %(default)s).")
    ap.add_argument("--empty-sample", type=int, default=25,
                    help="With --bundle: zero-detection panos to sample (default: %(default)s).")
    ap.add_argument("--seed", type=int, default=0,
                    help="With --bundle: sampling seed (default: %(default)s).")
    ap.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_SPACING_M,
                    help="With --bundle: min metres between sampled panos "
                         "(default: %(default)s; 0 disables).")
    ap.add_argument("--records-only", action="store_true",
                    help="With --bundle: write records.jsonl and stop — no fetch, no "
                         "reconcile. For sourcing the pixels elsewhere (e.g. copying the "
                         "sampled ids out of an existing full-city archive); re-run without "
                         "it afterwards to fetch anything missing and verify.")
    args = ap.parse_args()

    src = Path(args.source)
    if args.bundle:
        if src.is_dir():
            sys.exit("--bundle takes a run's results.jsonl as SOURCE, not a directory")
        records_path = write_bundle_records(src, args.bundle, args.sample, args.empty_sample,
                                            args.seed, args.min_spacing)
        panos_dir = args.out or (args.bundle / "panos")
        if args.records_only:
            return
    elif src.is_dir():
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

    label = args.bundle.name if args.bundle else src.name
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
