"""Archive GSV's depth payload for every panorama of a finished run.

Google serves a metric depth map alongside the panorama metadata this pipeline already
fetches. It is the only absolute distance reference we have, and it is entirely outside
our control: the *JavaScript* API that once exposed depth was withdrawn in 2020 (which is
why Project Sidewalk has a regression estimator at all), and anonymous access to the
imagery tile endpoint went away in ~June 2026. The metadata endpoint used here still
serves the payload today — that is precisely the point. Capture it now, derive from it
later (labeler #41).

What it buys, measured in #40: the dominant ground plane's distance IS the camera height,
exactly, and `geo.DEFAULT_CAMERA_HEIGHT_M = 2.6` is above every value observed — ranges
run ~29-35% long at real detection points. The plane normal gives ground tilt (1-2 deg
even on levelled rigs) and the per-pixel plane index gives occlusion structure.

Cheap enough to be complete rather than a sample: the payload gzips to ~5-7 KB, so all
four GSV runs (~171k panoramas) are ~1 GB. This is a metadata-only pass — no imagery is
downloaded — so it neither needs nor touches the native-resolution archive.

    # harvest one run (writes runs/paterson/depth/, resumable — just re-run it)
    python scripts/harvest_depth.py runs/paterson

    # archive alongside the imagery on makelab2, per the usual per-city layout
    python scripts/harvest_depth.py runs/bend --out /projects/.../runs/bend/depth

    # verify an existing archive without fetching anything (never mutates the caches)
    python scripts/harvest_depth.py runs/paterson --verify

    # ...and re-hash every file rather than trusting a matching byte size (catches bit rot)
    python scripts/harvest_depth.py runs/paterson --verify --rehash

    # re-check that depth.py still agrees with streetlevel's own raster (do this after
    # any streetlevel upgrade — the payload layout is undocumented)
    python scripts/harvest_depth.py runs/paterson --check-convention

GSV only: Mapillary serves no depth, and the run's manifest is checked so a Mapillary run
is refused rather than silently producing an empty archive.

Three sidecar files next to the payloads carry state between runs. `index.csv` is the
integrity record plus the derived geometry; `no_depth.txt` and `gone.txt` are skip caches
for the two deterministic outcomes, so neither is ever re-fetched — delete one to force a
re-check. They are only ever added to: a pass that does not re-encounter an id (--verify,
--limit) must not be able to erase it, which is a mistake this script has already made
once.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import depth as depthlib  # noqa: E402
from geo import DEFAULT_CAMERA_HEIGHT_M  # noqa: E402  (stdlib-only, like depth.py)

WORKERS = 16          # metadata-sized requests; the main pipeline uses 50-100 for these
FETCH_ATTEMPTS = 3    # per panorama, with backoff, before it becomes a retryable ERROR

GONE = "GONE"         # the source says this panorama is gone — cached in gone.txt
NO_DEPTH = "NO_DEPTH"  # panorama exists but carries no depth — cached in no_depth.txt
ERROR = "ERROR"       # transient; left uncached so the next run retries it

# NO_DEPTH is a deterministic outcome, but it is inferred from a *positional* walk into an
# undocumented response (depth.blob_from_response), so "Google moved the field" and "this
# panorama has no depth" look identical. Caching the second is right; caching the first
# would silently poison a whole city, and the payload is the thing this tool exists to
# capture before it disappears. Across all four production runs (170,932 panoramas)
# NO_DEPTH fired exactly zero times, so a sudden crop of it is far likelier to be a schema
# change than a real one. Above this rate the cache is not written and the run is an
# anomaly; the minimum keeps a tiny --limit smoke test from tripping on a single miss.
NO_DEPTH_ALARM_RATE = 0.05
NO_DEPTH_ALARM_MIN = 20

INDEX_FIELDS = ["panorama_id", "filename", "bytes", "sha256", "n_planes", "degenerate",
                "camera_height_m", "ground_tilt_deg", "ground_pixel_share",
                "height_spread_m", "sky_fraction"]


def run_pano_ids(run_dir):
    """Every panorama id in a run's results.jsonl, in file order, plus its source."""
    results = run_dir / "results.jsonl"
    if not results.exists():
        sys.exit(f"No results.jsonl in {run_dir}")
    ids, source = [], None
    with open(results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pano = json.loads(line)["pano"]
            ids.append(pano["panorama_id"])
            source = source or pano.get("source")
    return ids, source


def check_gsv(run_dir):
    """Refuse a non-GSV run up front rather than fetching 50k panoramas of nothing."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        source = json.loads(manifest_path.read_text(encoding="utf-8")).get("imagery_source", "gsv")
        if source != "gsv":
            sys.exit(f"{run_dir.name} is a '{source}' run; only GSV serves depth. "
                     f"See labeler #42 for the Mapillary route.")


def fetch_depth(pano_id, out_path):
    """Fetch and store one panorama's depth payload.

    Returns None on success, else (kind, message). Writes through a .part file for the
    same reason export_benchmark.py does: the resume check is "does the file exist?", so
    a fetch that dies partway must never leave bytes at out_path.
    """
    from streetlevel.streetview import api  # lazy — --verify needs no network at all

    part = out_path.with_suffix(out_path.suffix + ".part")
    try:
        # A short retry, matching sources/gsv.fetch_metadata_with_retry in spirit. An ERROR
        # is uncached and heals on the next run anyway, so this only saves restarts during
        # a flaky patch rather than changing any outcome.
        for attempt in range(FETCH_ATTEMPTS):
            try:
                response = api.find_panorama_by_id(pano_id, download_depth=True)
                break
            except Exception:
                if attempt == FETCH_ATTEMPTS - 1:
                    raise
                time.sleep(2 ** attempt)

        try:
            code = response[1][0][0][0]
        except (IndexError, KeyError, TypeError):
            return ERROR, "unrecognized response shape"
        if code not in (1, 3):                      # streetlevel's own OK codes
            return GONE, f"no such panorama (response code {code})"

        blob = depthlib.blob_from_response(response)
        if not blob:
            return NO_DEPTH, "panorama carries no depth payload"
        depthlib.parse(blob)                        # reject a corrupt payload here, not at analysis time

        payload = {"pano_id": pano_id, "depth_b64": blob,
                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with gzip.open(part, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        if part.stat().st_size == 0:
            raise OSError("wrote 0 bytes")
        part.replace(out_path)
        return None
    except Exception as e:
        return ERROR, str(e)
    finally:
        part.unlink(missing_ok=True)


def read_payload(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return depthlib.parse(json.load(f)["depth_b64"])


def _sha256(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ids(path):
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _write_ids(path, ids):
    """Rewrite an id cache from a set, so repeated passes cannot accumulate duplicates."""
    if ids:
        path.write_text("".join(pid + "\n" for pid in sorted(ids)), encoding="utf-8")
    elif path.exists():
        path.unlink()


def load_no_depth(manifest_dir):
    return _load_ids(manifest_dir / "no_depth.txt")


def no_depth_looks_poisoned(n_new, n_attempted):
    """Is this pass's NO_DEPTH rate high enough to suspect a changed response shape?

    See NO_DEPTH_ALARM_RATE. The minimum count keeps a small --limit smoke test from
    tripping on one genuine miss.
    """
    return (n_new >= NO_DEPTH_ALARM_MIN
            and n_new / max(n_attempted, 1) > NO_DEPTH_ALARM_RATE)


def load_gone(manifest_dir):
    """Panoramas the source itself says are gone. A cached, deterministic outcome.

    This is a real skip cache, not just a report: without it `gone.txt` was written by
    reconcile but never read back, so gone panoramas were re-fetched on every run despite
    GONE being documented as never retried. Delete the file to force a re-check.
    """
    return _load_ids(manifest_dir / "gone.txt")


def reconcile(depth_dir, expected, failures=None, attempted=None, rehash=False):
    """Prove the archive matches the run it was built from, and record what's in it.

    Mirrors export_benchmark.reconcile so "is this the same data we processed?" is
    answerable the same mechanical way for depth as for imagery. index.csv additionally
    carries the derived per-panorama geometry, so city-wide camera-height statistics need
    no decompression at all.

    `attempted` is the set of ids this pass actually tried. It separates a panorama that
    *failed* from one that was simply never reached (a --limit smoke test, or an
    interrupted run) — reporting the second as a failure would be a lie, and it is the
    difference between "re-run to repair" and "re-run to continue". Passing None means
    "everything was attempted", which is the right reading for --verify.

    Rebuilt incrementally: a panorama already indexed at the same byte size is not
    re-hashed or re-parsed, so a resumed run only pays for what it actually fetched.
    `rehash` forces the full check, which is the only way to catch bit rot in a file whose
    size did not change.

    The gone list is the UNION of what is already recorded and what this pass found, minus
    anything now archived. It must never be rebuilt from this pass alone: --verify and
    --limit legitimately do not re-encounter those ids, and overwriting from an empty set
    silently deleted gone.txt and downgraded a complete archive to PARTIAL.

    Returns (gone, failed, pending, anomalies); failed/anomalies are what make a run
    untrustworthy, pending only means unfinished.
    """
    failures = failures or {}
    expected_ids = set(expected)
    present = {p.name[:-len(".json.gz")]: p for p in depth_dir.glob("*.json.gz")}

    for pid in [pid for pid, p in present.items() if p.stat().st_size == 0]:
        present.pop(pid).unlink()
        print(f"  removed empty file {pid}.json.gz — it will be re-fetched next run")
    present_ids = set(present)

    # A hard kill skips fetch_depth's `finally`, so .part files can outlive it. They match
    # neither the *.json.gz glob nor the `extra` check, so sweep them here or they
    # accumulate invisibly.
    orphans = [p for p in depth_dir.glob("*.json.gz.part")]
    for p in orphans:
        p.unlink()
    if orphans:
        print(f"  swept {len(orphans)} interrupted .part file(s)")

    no_depth = load_no_depth(depth_dir)
    verified = expected_ids & present_ids
    missing = expected_ids - present_ids - no_depth
    extra = present_ids - expected_ids
    gone = ((load_gone(depth_dir) | {pid for pid, f in failures.items() if f[0] == GONE})
            & expected_ids) - present_ids
    unresolved = missing - gone
    pending = unresolved if attempted is None else unresolved - set(attempted)
    failed = unresolved - pending

    index_path = depth_dir / "index.csv"
    prior = {}
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prior[row["panorama_id"]] = row

    rows, unreadable, corrupted = [], [], []
    for pid in tqdm(sorted(verified), desc="Indexing depth", unit="pano"):
        p = present[pid]
        size = p.stat().st_size
        cached = prior.get(pid)
        if not rehash and cached and int(cached["bytes"]) == size and cached.get("sha256"):
            rows.append([cached.get(k, "") for k in INDEX_FIELDS])
            continue
        try:
            payload = read_payload(p)
        except Exception as e:
            unreadable.append((pid, str(e)))
            continue
        digest = _sha256(p)
        # Only reachable under --rehash: without it a size match short-circuits above.
        if cached and cached.get("sha256") and cached["sha256"] != digest:
            corrupted.append(pid)
        g = depthlib.ground_plane(payload)
        rows.append([
            pid, p.name, size, digest, payload.n_planes, int(payload.degenerate),
            f"{g.camera_height_m:.4f}" if g else "",
            f"{g.tilt_deg:.3f}" if g else "",
            f"{g.pixel_share:.4f}" if g else "",
            f"{g.height_spread_m:.4f}" if g else "",
            f"{payload.sky_fraction:.4f}",
        ])

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(INDEX_FIELDS)
        w.writerows(rows)

    _write_ids(depth_dir / "gone.txt", gone)

    print(f"--- Reconcile: run records vs {depth_dir} ---")
    print(f"  run panoramas (unique):  {len(expected_ids)}")
    print(f"  archived + verified:     {len(rows)}  -> index.csv")
    print(f"  no depth served:         {len(no_depth & expected_ids)}" +
          ("  -> no_depth.txt" if no_depth else ""))
    if gone:
        print(f"  missing (source gone):   {len(gone)}  -> gone.txt")
    if failed:
        print(f"  missing (fetch failed):  {len(failed)}, e.g. {sorted(failed)[:3]}")
    if pending:
        print(f"  not yet fetched:         {len(pending)}")
    if unreadable:
        print(f"  WARNING: {len(unreadable)} unreadable file(s), e.g. {unreadable[:2]}")
    if extra:
        print(f"  WARNING: {len(extra)} depth file(s) not in the run, e.g. {sorted(extra)[:3]}")
    if corrupted:
        print(f"  WARNING: {len(corrupted)} file(s) changed on disk since they were indexed "
              f"(same size, different sha256), e.g. {sorted(corrupted)[:3]}")

    if extra or unreadable or corrupted:
        status = (f"ANOMALY — {len(extra)} unbacked file(s), {len(unreadable)} unreadable, "
                  f"{len(corrupted)} altered")
    elif failed:
        status = (f"INCOMPLETE — {len(failed)} panorama(s) missing from a failed fetch; "
                  f"re-run to repair")
    elif pending:
        status = f"PARTIAL — {len(pending)} panorama(s) not yet fetched; re-run to continue"
    elif gone or no_depth:
        status = (f"OK with gaps — {len(gone)} gone from the source, "
                  f"{len(no_depth & expected_ids)} served no depth")
    else:
        status = "OK — depth archive matches the run 1:1"
    print(f"  STATUS: {status}")
    return len(gone), len(failed), len(pending), len(extra) + len(unreadable) + len(corrupted)


def summarize(depth_dir):
    """City-wide camera-height statistics, straight from index.csv — no decompression."""
    index_path = depth_dir / "index.csv"
    if not index_path.exists():
        return
    heights, degenerate, tilts, n = [], 0, [], 0
    with open(index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            if int(row["degenerate"] or 0):
                degenerate += 1
                continue                    # a fallback reconstruction, not a measurement
            if row["camera_height_m"]:
                heights.append(float(row["camera_height_m"]))
                tilts.append(float(row["ground_tilt_deg"]))
    if not heights:
        return
    heights.sort()
    tilts.sort()

    def pct(a, q):
        return a[min(len(a) - 1, int(q * len(a)))]

    print(f"--- Camera height across {n} panoramas "
          f"({degenerate} degenerate, excluded) ---")
    print(f"  min {heights[0]:.3f}   p25 {pct(heights, .25):.3f}   "
          f"median {pct(heights, .5):.3f}   p75 {pct(heights, .75):.3f}   "
          f"max {heights[-1]:.3f}")
    print(f"  ground tilt: median {pct(tilts, .5):.2f} deg, p90 {pct(tilts, .9):.2f} deg")
    # Read the live constant rather than restating it: #40 is about to change it, and a
    # hardcoded 2.6 here would start printing a false claim the moment it lands.
    bias = 100 * (DEFAULT_CAMERA_HEIGHT_M / pct(heights, .5) - 1)
    print(f"  geo.DEFAULT_CAMERA_HEIGHT_M = {DEFAULT_CAMERA_HEIGHT_M} "
          f"{'overestimates' if bias >= 0 else 'underestimates'} range by "
          f"{abs(bias):.0f}% at the median (labeler #40)")


def check_convention(pano_ids, n_panos, n_samples=2000):
    """Re-verify depth.py against streetlevel's own raster on live panoramas.

    The payload layout is undocumented and positional, so this is the check that says
    whether a streetlevel upgrade (or a Google change) has invalidated the parser.
    Exits non-zero on any disagreement.

    Panoramas are drawn from the whole run rather than off the front of results.jsonl: the
    first few lines of a run are one tile's worth of coverage, captured by one rig on one
    day, which is exactly the sample least likely to contain an awkward reconstruction.
    Seeded, so it stays reproducible.
    """
    import random
    from streetlevel.streetview import api
    from streetlevel.streetview.depth import parse as sl_parse

    random.seed(0)
    total_bad = 0
    for pid in random.sample(pano_ids, min(n_panos, len(pano_ids))):
        response = api.find_panorama_by_id(pid, download_depth=True)
        blob = depthlib.blob_from_response(response)
        if not blob:
            print(f"  {pid}: no depth, skipped")
            continue
        ref = sl_parse(blob).data                      # streetlevel's raster
        payload = depthlib.parse(blob)
        bad = 0
        for _ in range(n_samples):
            row = random.randrange(payload.height)
            col = random.randrange(payload.width)
            mine = depthlib.depth_at(payload, (col + 0.5) / payload.width,
                                     (row + 0.5) / payload.height)
            theirs = ref[row][col]
            if theirs < 0:
                bad += mine is not None
            elif mine is None or abs(mine - theirs) / max(theirs, 1e-9) > 1e-6:
                bad += 1
        total_bad += bad
        print(f"  {pid}: {n_samples} samples, {bad} mismatch(es)")
    if total_bad:
        sys.exit(f"CONVENTION CHECK FAILED — {total_bad} mismatch(es); depth.py and "
                 f"streetlevel disagree, do not trust harvested geometry")
    print("  STATUS: OK — depth.py agrees with streetlevel exactly")


def main():
    ap = argparse.ArgumentParser(description="Archive GSV depth payloads for a finished run.")
    ap.add_argument("run", type=Path, help="A run directory, e.g. runs/paterson")
    ap.add_argument("--out", type=Path,
                    help="Where to write the archive. Default: <run>/depth")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, help="Fetch at most this many new panoramas (smoke test)")
    ap.add_argument("--verify", action="store_true",
                    help="Reconcile and summarize an existing archive; fetch nothing.")
    ap.add_argument("--rehash", action="store_true",
                    help="Re-read and re-hash every archived file instead of trusting a "
                         "matching byte size — the only way to catch silent bit rot.")
    ap.add_argument("--check-convention", type=int, nargs="?", const=3, metavar="N",
                    help="Verify depth.py against streetlevel's raster on N live panoramas.")
    args = ap.parse_args()

    run_dir = args.run
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")
    check_gsv(run_dir)
    pano_ids, source = run_pano_ids(run_dir)
    # Belt to check_gsv's braces: that reads the manifest, this the records, so a run dir
    # without a manifest is still refused. Allowlist rather than denylist — a new source
    # must not default into "GSV serves depth for this".
    if source and source != "gsv":
        sys.exit(f"This run's records are {source}; only GSV serves depth (see #42).")

    # `is not None`, not truthiness: 0 is a meaningful value for both of these and reading
    # it as "unset" turns `--limit 0` into a full 170k-panorama harvest.
    if args.limit is not None and args.limit < 0:
        sys.exit("--limit must be >= 0")
    if args.check_convention is not None:
        if args.check_convention < 1:
            sys.exit("--check-convention needs at least 1 panorama")
        print(f"--- Convention check ({args.check_convention} panoramas) ---")
        check_convention(pano_ids, args.check_convention)
        return

    depth_dir = args.out or (run_dir / "depth")
    depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run_dir.name}: {len(pano_ids)} panoramas in the run -> {depth_dir}")

    failures, attempted, poisoned = {}, None, False
    if not args.verify:
        # Both deterministic outcomes are skip caches; delete the file to force a re-check.
        skip = load_no_depth(depth_dir) | load_gone(depth_dir)
        todo = [pid for pid in dict.fromkeys(pano_ids)
                if pid not in skip and not (depth_dir / f"{pid}.json.gz").exists()]
        print(f"  {len(pano_ids) - len(todo)} already archived, known depth-less or gone; "
              f"{len(todo)} to fetch")
        if args.limit is not None:
            todo = todo[:args.limit]
            print(f"  --limit: fetching {len(todo)}")
        attempted = set(todo)

        new_no_depth = []
        if todo:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(fetch_depth, pid, depth_dir / f"{pid}.json.gz"): pid
                           for pid in todo}
                for fut in tqdm(as_completed(futures), desc="Fetching depth",
                                unit="pano", total=len(todo)):
                    result = fut.result()
                    if result is None:
                        continue
                    pid = futures[fut]
                    kind, message = result
                    failures[pid] = (kind, message)
                    if kind == NO_DEPTH:
                        new_no_depth.append(pid)

        # Deterministic outcomes are cached so they are never retried; ERRORs are
        # deliberately not, so a flaky pass self-heals on the next run (main.py idiom).
        # gone.txt is written by reconcile, which owns the union with what's already there.
        if new_no_depth:
            rate = len(new_no_depth) / max(len(attempted), 1)
            if no_depth_looks_poisoned(len(new_no_depth), len(attempted)):
                poisoned = True
                print(f"  REFUSING to cache {len(new_no_depth)} NO_DEPTH result(s) "
                      f"({100 * rate:.1f}% of this pass, threshold "
                      f"{100 * NO_DEPTH_ALARM_RATE:.0f}%). A rate this high is far more "
                      f"likely to be a changed response shape than genuinely depth-less "
                      f"panoramas — caching it would permanently skip them. Re-check "
                      f"depth.blob_from_response against a live response, then re-run.")
            else:
                _write_ids(depth_dir / "no_depth.txt",
                           load_no_depth(depth_dir) | set(new_no_depth))

        kinds = {}
        for kind, _ in failures.values():
            kinds[kind] = kinds.get(kind, 0) + 1
        if kinds:
            print(f"  outcomes: {kinds}")

    gone, failed, pending, anomalies = reconcile(depth_dir, pano_ids, failures, attempted,
                                                 rehash=args.rehash)
    summarize(depth_dir)
    # Unfinished is not untrustworthy: exit non-zero only for things a re-run won't fix
    # by itself (failed fetches, unbacked/unreadable/altered files, a suspected schema
    # change), so `--limit` smoke tests and interrupted runs don't read as errors to a
    # caller or a CI step.
    sys.exit(1 if (failed or anomalies or poisoned) else 0)


if __name__ == "__main__":
    main()
