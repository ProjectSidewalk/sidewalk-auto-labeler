"""Archive GSV's depth payload for every panorama of a finished run.

Google serves a metric depth map alongside the panorama metadata this pipeline already
fetches. It is the only absolute distance reference we have, and it is entirely outside
our control: the *JavaScript* API that once exposed depth was withdrawn in 2020 (which is
why Project Sidewalk has a regression estimator at all), and anonymous access to the
imagery tile endpoint went away in ~June 2026. The metadata endpoint used here still
serves the payload today — that is precisely the point. Capture it now, derive from it
later (labeler #41).

What it buys (#40): the dominant ground plane's distance is a per-pano camera height --
it ranks capture rigs correctly (the 2025-26 GSV rig really is lower), though it runs
6-16% short of the height the imagery's own geometry implies, so it is not a drop-in
range correction (docs/camera-height-study.md). The plane normal gives ground tilt (1-2
deg even on levelled rigs) and the per-pixel plane index gives occlusion structure.

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

    # rebuild index.csv's derived columns after a depth.py change (offline, #47)
    python scripts/harvest_depth.py runs/paterson --reindex

    # re-check that depth.py still agrees with streetlevel's own raster (do this after
    # any streetlevel upgrade — the payload layout is undocumented)
    python scripts/harvest_depth.py runs/paterson --check-convention

    # build the index from pano-tools' depth artifacts in a local pano store instead of
    # fetching (issue #56): <store>/<id[:2]>/<id>.depth.npz, read in place, nothing copied
    python scripts/harvest_depth.py runs/vancouver --from-store /projects/.../vancouver-wa

    # ...after proving the store's frame equals a live payload's on N panos (N checked; a
    # pano gone from GSV is not a check, so the draw goes on; the result lands in
    # <depth dir>/store.json, and an index pass without one says "frame unchecked")
    python scripts/harvest_depth.py runs/vancouver --from-store <store> --check-store-frame 5

A depth dir indexed from a store is bound to it (store.json) and is ALWAYS addressed with
--from-store: without the flag the script refuses it rather than reconciling it as a harvest
archive (which would empty its index) or fetching into it. A store index keeps pano-tools'
`unavailable` ledger outcome (gone OR served no depth) in `unavailable.txt`, not gone.txt.

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
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import depth as depthlib  # noqa: E402
from geo import DEFAULT_CAMERA_HEIGHT_M  # noqa: E402  (stdlib-only, like depth.py)
from sources import SOURCE_NAMES  # noqa: E402  (the registry only; source modules load lazily)

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

# height_spread_m leaves Google's stand-in planes out since #47 (depth.ground_plane);
# n_standin_planes / standin_pixel_share count them. An index written before that has
# neither column and the old spread: reconcile recomputes such rows rather than trusting
# them, and --reindex rebuilds a whole archive's index offline.

# pano-tools' depth artifacts (sidewalk-panorama-tools downloaders/gsv.py, format v3): one
# `<id>.depth.npz` beside each store JPEG, and a `depth_log.csv` ledger of outcomes
# ('saved' | 'unavailable' = the pano is gone or served no depth; never retried there).
STORE_SUFFIX = ".depth.npz"
STORE_LEDGER = "depth_log.csv"
STORE_RECORD = "store.json"          # in the depth dir: which store, + the frame check's result
NO_PLANES_FILE = "no_planes.txt"     # artifacts without the plane list (pre-v3); see NoPlanes
# pano-tools' `unavailable` means "gone OR served no depth" (its gsv.py does not tell them
# apart), which a harvest keeps in two files (gone.txt / no_depth.txt). A store index keeps
# it under its own name rather than pretending it is either.
UNAVAILABLE_FILE = "unavailable.txt"

INDEX_FIELDS = ["panorama_id", "filename", "bytes", "sha256", "n_planes", "degenerate",
                "camera_height_m", "ground_tilt_deg", "ground_pixel_share",
                "height_spread_m", "sky_fraction", "n_standin_planes",
                "standin_pixel_share"]


def run_pano_ids(run_dir):
    """Every panorama id in a run's results.jsonl, in file order, plus the set of every
    record's `source` (None left out) -- all of them, so a mixed file can be refused."""
    results = run_dir / "results.jsonl"
    if not results.exists():
        sys.exit(f"No results.jsonl in {run_dir}")
    ids, sources = [], set()
    with open(results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pano = json.loads(line)["pano"]
            ids.append(pano["panorama_id"])
            if pano.get("source") is not None:
                sources.add(pano["source"])
    return ids, sources


# Pano-block `source` values that name a non-GSV provider: every registered imagery source
# but gsv (derived, so a source added to sources.SOURCE_NAMES is refused without touching
# this file), plus infra3d, the one PS pano_source enum value no source module writes yet.
# A GSV record stores streetlevel's raw source string instead -- 'launch', 'scout',
# 'photos:...' -- an open set, so GSV cannot be an allowlist of strings: the
# `source != "gsv"` test that stood here refused every real GSV run ('launch'), --verify
# included. send_to_ps.transform_pano reads the field the same way.
NON_GSV_SOURCES = frozenset((set(SOURCE_NAMES) | {"infra3d"}) - {"gsv"})


def is_gsv_source(source):
    """Is a pano block's `source` GSV imagery? None (no records) passes; the manifest
    check (check_gsv) is the other half.

    Example:
        >>> [is_gsv_source(s) for s in ("launch", "gsv", "scout", "mapillary", None)]
        [True, True, True, False, True]
    """
    return source is None or source.lower() not in NON_GSV_SOURCES


def record_source_problem(sources):
    """Why a run's record sources rule out a depth harvest, or None if they are all GSV.

    `sources` is run_pano_ids' set. Any non-GSV provider refuses; a file that holds both
    GSV and non-GSV records is called out as mixed, since no run directory should be.

    Example:
        >>> record_source_problem({"launch", "scout"}) is None
        True
        >>> record_source_problem({"launch", "mapillary"})
        "This run's records are mixed (GSV and mapillary); only GSV serves depth (see #42)."
    """
    non_gsv = sorted(s for s in sources if not is_gsv_source(s))
    if not non_gsv:
        return None
    if len(non_gsv) < len(sources):
        return (f"This run's records are mixed (GSV and {', '.join(non_gsv)}); "
                f"only GSV serves depth (see #42).")
    return f"This run's records are {', '.join(non_gsv)}; only GSV serves depth (see #42)."


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


def reconcile(depth_dir, expected, failures=None, attempted=None, rehash=False,
              write_on_anomaly=True):
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

    An unreadable or altered file (same size, different sha256) keeps its PRIOR index row,
    old sha256 included, so the anomaly is reported on every later --rehash pass instead of
    being laundered into a clean index by recording the new digest. The index is written to
    a temporary file and moved into place (os.replace). With write_on_anomaly=False
    (--reindex) nothing is written at all unless the pass found no anomaly.

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
            reader = csv.DictReader(f)
            # An index from an older INDEX_FIELDS is missing columns, and its derived
            # columns may follow an older definition (the #47 spread), so none of its rows
            # may be reused as-is: every file is re-read. Its sha256s are still compared.
            stale_schema = not set(INDEX_FIELDS) <= set(reader.fieldnames or [])
            for row in reader:
                prior[row["panorama_id"]] = row
        if stale_schema and prior:
            print(f"  index.csv predates the current columns; recomputing all "
                  f"{len(prior)} row(s)")
            rehash = True

    rows, unreadable, corrupted = [], [], []
    for pid in tqdm(sorted(verified), desc="Indexing depth", unit="pano"):
        p = present[pid]
        size = p.stat().st_size
        cached = prior.get(pid)
        # A row carried forward from an older schema (an anomaly's, below) has a ground
        # plane but blank stand-in columns and the old spread: never reuse it as current.
        current = cached and (cached.get("n_standin_planes") or not cached.get("camera_height_m"))
        if not rehash and current and int(cached["bytes"]) == size and cached.get("sha256"):
            rows.append([cached.get(k, "") for k in INDEX_FIELDS])
            continue
        try:
            payload = read_payload(p)
        except Exception as e:
            unreadable.append((pid, str(e)))
            if cached:
                rows.append([cached.get(k, "") for k in INDEX_FIELDS])   # carried forward
            continue
        digest = _sha256(p)
        # Only reachable under --rehash: without it a size match short-circuits above.
        if cached and cached.get("sha256") and cached["sha256"] != digest:
            corrupted.append(pid)
            rows.append([cached.get(k, "") for k in INDEX_FIELDS])       # old sha kept
            continue
        g = depthlib.ground_plane(payload)
        rows.append([
            pid, p.name, size, digest, payload.n_planes, int(payload.degenerate),
            f"{g.camera_height_m:.4f}" if g else "",
            f"{g.tilt_deg:.3f}" if g else "",
            f"{g.pixel_share:.4f}" if g else "",
            f"{g.height_spread_m:.4f}" if g else "",
            f"{payload.sky_fraction:.4f}",
            g.n_standin_planes if g else "",
            f"{g.standin_pixel_share:.4f}" if g else "",
        ])

    n_anomalies = len(extra) + len(unreadable) + len(corrupted)
    written = write_on_anomaly or not n_anomalies
    if written:
        tmp = index_path.with_name(index_path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(INDEX_FIELDS)
            w.writerows(rows)
        os.replace(tmp, index_path)
        _write_ids(depth_dir / "gone.txt", gone)

    print(f"--- Reconcile: run records vs {depth_dir} ---")
    print(f"  run panoramas (unique):  {len(expected_ids)}")
    print(f"  archived + verified:     {len(rows)}  -> index.csv"
          + ("" if written else " NOT WRITTEN (anomalies; index.csv left untouched)"))
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
    return len(gone), len(failed), len(pending), n_anomalies


def reindex(depth_dir, expected):
    """Recompute every derived index.csv column from the archived files, offline (#47).

    For a change to what `depth.py` derives from a payload -- the files themselves never
    change, so nothing is fetched. Every file is re-read and re-hashed (reconcile with
    `rehash`), so a file altered since it was indexed still surfaces as an anomaly.

    Refuses, before writing anything, when index.csv is absent or names a panorama whose
    file is gone or empty (reconcile deletes a zero-byte file, and would then silently drop
    that row): a reindex must never be able to shrink the archive's record. The rebuilt
    index replaces the old one (temp file + os.replace) ONLY when the pass found no anomaly
    -- an unreadable, altered or unbacked file leaves index.csv and gone.txt untouched, so
    the old row and its recorded sha256 stay the evidence. Returns reconcile's tuple.
    """
    index_path = depth_dir / "index.csv"
    if not index_path.exists():
        sys.exit(f"--reindex: no index.csv in {depth_dir}; nothing to reindex")
    with open(index_path, newline="", encoding="utf-8") as f:
        indexed = [row["panorama_id"] for row in csv.DictReader(f)]
    def _missing(pid):
        path = depth_dir / f"{pid}.json.gz"
        return not path.exists() or path.stat().st_size == 0

    missing = [pid for pid in indexed if _missing(pid)]
    if missing:
        sys.exit(f"--reindex: REFUSING -- {len(missing)} indexed panorama(s) have no file "
                 f"(or an empty one) in {depth_dir}, e.g. {missing[:3]}; index.csv left "
                 f"untouched")
    print(f"--- Reindex: recomputing {len(indexed)} row(s) from the archived files ---")
    return reconcile(depth_dir, expected, rehash=True, write_on_anomaly=False)


def summarize(depth_dir):
    """City-wide camera-height statistics, straight from index.csv — no decompression."""
    index_path = depth_dir / "index.csv"
    if not index_path.exists():
        return
    heights, tilts, n = [], [], 0
    statuses = Counter()
    with open(index_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            h = float(row["camera_height_m"]) if row["camera_height_m"] else None
            tilt = float(row["ground_tilt_deg"]) if row["ground_tilt_deg"] else None
            status = depthlib.classify_height(h, tilt, degenerate=row["degenerate"] == "1")
            statuses[status] += 1
            if status == depthlib.MEASURED:
                heights.append(h)
                tilts.append(tilt)
    if not heights:
        return
    heights.sort()
    tilts.sort()

    def pct(a, q):
        return a[min(len(a) - 1, int(q * len(a)))]

    excluded = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())
                         if k != depthlib.MEASURED)
    print(f"--- Camera height across {n} panoramas "
          f"({statuses[depthlib.MEASURED]} measured; excluded: {excluded or 'none'}) ---")
    print(f"  min {heights[0]:.3f}   p25 {pct(heights, .25):.3f}   "
          f"median {pct(heights, .5):.3f}   p75 {pct(heights, .75):.3f}   "
          f"max {heights[-1]:.3f}")
    print(f"  ground tilt: median {pct(tilts, .5):.2f} deg, p90 {pct(tilts, .9):.2f} deg")
    # These are the depth frame's heights, which run 6-16% short of the height the
    # imagery's own geometry implies -- so a gap to the raycast constant here is NOT a
    # range bias. `fuse_sites.py --implied-height` measures that one.
    print(f"  (depth-frame heights; geo.DEFAULT_CAMERA_HEIGHT_M = {DEFAULT_CAMERA_HEIGHT_M}. "
          f"Compare against fuse_sites.py --implied-height, not this -- labeler #40)")


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


# ------------------------------------------------------------------ the pano store (#56)
# pano-tools' nightly depth phase writes one v3 artifact per pano into the Project Sidewalk
# pano store. Where it has run, there is nothing left to fetch: the artifact carries
# Google's plane list verbatim (`plane_indices` uint8 (h, w) in the payload's own column
# order, `planes_n` float32 (P, 3), `planes_d` float32 (P,)), which is exactly what
# depth.parse reads off the wire. So the store is read in place and indexed with the same
# schema as a harvest; `filename` is the artifact's path relative to the store and
# `bytes`/`sha256` are the artifact's own. Nothing is copied.
#
# Frame: pano-tools flips its *raster* (`depth`) on write (sidewalk-panorama-tools#58), and depth.py has its own
# raster mirror (#80). Neither touches the plane indices, which both sides keep in raw
# payload order (= image order) -- but that is a claim about two codebases, so
# check_store_frame proves it against live payloads before an index is trusted.

class NoPlanes(ValueError):
    """An artifact without the plane list (format < 3: pano-tools' v2 kept only the
    raster). The planes cannot be recovered from it, so it is recorded, never guessed."""


def store_npz_path(store, pano_id):
    """<store>/<id[:2]>/<id>.depth.npz (pano-tools' layout), else <store>/<id>.depth.npz."""
    store = Path(store)
    sharded = store / pano_id[:2] / f"{pano_id}{STORE_SUFFIX}"
    if sharded.exists():
        return sharded
    flat = store / f"{pano_id}{STORE_SUFFIX}"
    return flat if flat.exists() else sharded


def payload_from_npz(source):
    """A depth.DepthPayload rebuilt from a pano-tools depth artifact (a path, or any
    mapping with its fields). Raises NoPlanes for an artifact without the plane list and
    ValueError for one whose fields disagree with each other.

    float32 -> Python float is exact, as is depth.parse's struct '<f', so a payload rebuilt
    here is field-for-field identical to the parse of the wire payload it came from.

    Example:
        >>> import numpy as np
        >>> p = payload_from_npz({'plane_indices': np.zeros((2, 4), np.uint8),
        ...     'planes_n': np.zeros((1, 3), np.float32), 'planes_d': np.zeros(1, np.float32)})
        >>> (p.width, p.height, p.n_planes)
        (4, 2, 1)
    """
    import numpy as np
    if isinstance(source, (str, Path)):
        with np.load(source) as z:
            return payload_from_npz({k: z[k] for k in z.files})
    version = int(source["format_version"]) if "format_version" in source else None
    if not all(k in source for k in ("plane_indices", "planes_n", "planes_d")):
        raise NoPlanes(f"no plane list (format_version {version})")
    idx = np.ascontiguousarray(source["plane_indices"], dtype=np.uint8)
    if idx.ndim != 2:
        raise ValueError(f"plane_indices has shape {idx.shape}, expected (h, w)")
    normals = np.asarray(source["planes_n"], dtype=np.float32).reshape(-1, 3)
    dists = np.asarray(source["planes_d"], dtype=np.float32).reshape(-1)
    if len(normals) != len(dists):
        raise ValueError(f"{len(normals)} normals but {len(dists)} offsets")
    if idx.size and int(idx.max()) >= max(len(normals), 1):
        raise ValueError(f"plane index {int(idx.max())} past {len(normals)} planes")
    planes = [depthlib.Plane(nx, ny, nz, d)
              for (nx, ny, nz), d in zip(normals.tolist(), dists.tolist())]
    height, width = idx.shape
    return depthlib.DepthPayload(width, height, planes, idx.tobytes())


def load_store_ledger(store):
    """{pano_id: status} from the store's depth_log.csv (last row wins), or {}."""
    path = Path(store) / STORE_LEDGER
    status = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2 and row[0] != "pano_id":
                    status[row[0]] = row[1]
    return status


def load_store_record(depth_dir):
    """The depth dir's store record (STORE_RECORD), or None when it indexes no store."""
    path = Path(depth_dir) / STORE_RECORD
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_store_record(depth_dir, record):
    path = Path(depth_dir) / STORE_RECORD
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def bind_store(depth_dir, store):
    """A depth dir indexes ONE source: a harvest's *.json.gz or one store. Mixing them would
    give index.csv rows whose `filename` means two different things. Stores are compared
    resolved, so a trailing slash or a relative path names the same store. Returns the
    store record."""
    depth_dir.mkdir(parents=True, exist_ok=True)
    if any(depth_dir.glob("*.json.gz")):
        sys.exit(f"{depth_dir} holds harvested payloads (*.json.gz); index a store into a "
                 f"separate --out rather than mixing the two in one index.csv.")
    store = str(Path(store).resolve())
    record = load_store_record(depth_dir)
    if record is not None:
        if record.get("store") != store:
            sys.exit(f"{depth_dir} was indexed from store {record.get('store')}, not {store}; "
                     f"use another --out.")
        return record
    record = {"store": store, "artifact": f"<id[:2]>/<id>{STORE_SUFFIX}",
              "producer": "sidewalk-panorama-tools depth phase (format v3)",
              "bound_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_store_record(depth_dir, record)
    return record


def reconcile_store(depth_dir, store, expected, rehash=False):
    """index.csv for a run's panos from the store's depth artifacts, with the same schema,
    incremental re-use and --rehash semantics as `reconcile`.

    Outcomes per run pano: indexed; `unavailable` (no artifact, and the store's ledger says
    'unavailable' -- the pano was gone OR served no depth when pano-tools asked, which it
    does not tell apart; added to unavailable.txt, which like a harvest's gone.txt is only
    ever added to); `no_planes` (a pre-v3 artifact; rewritten each pass into no_planes.txt,
    since pano-tools can still re-fetch it); pending (no artifact and no ledger verdict yet
    -- the depth phase has not reached it).

    The index is only as good as the frame it was read in: without a passing
    --check-store-frame result in the store record, the pass says "frame unchecked".

    Returns (unavailable, no_planes, pending, anomalies).
    """
    store = Path(store)
    record = bind_store(depth_dir, store)
    expected_ids = list(dict.fromkeys(expected))
    ledger = load_store_ledger(store)

    index_path = depth_dir / "index.csv"
    prior = {}
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prior[row["panorama_id"]] = row

    rows, no_planes, unreadable, corrupted, absent = [], set(), [], [], set()
    for pid in tqdm(sorted(expected_ids), desc="Indexing store depth", unit="pano"):
        path = store_npz_path(store, pid)
        if not path.exists():
            absent.add(pid)
            continue
        rel = path.relative_to(store).as_posix()
        size = path.stat().st_size
        cached = prior.get(pid)
        if (not rehash and cached and cached.get("filename") == rel
                and int(cached["bytes"]) == size and cached.get("sha256")):
            rows.append([cached.get(k, "") for k in INDEX_FIELDS])
            continue
        try:
            payload = payload_from_npz(path)
        except NoPlanes:
            no_planes.add(pid)
            continue
        except Exception as e:  # noqa: BLE001 -- a bad artifact is reported, never fatal
            unreadable.append((pid, str(e)))
            continue
        digest = _sha256(path)
        # Only reachable under --rehash: without it a size match short-circuits above.
        if (cached and cached.get("sha256") and cached.get("filename") == rel
                and int(cached["bytes"]) == size and cached["sha256"] != digest):
            corrupted.append(pid)
        g = depthlib.ground_plane(payload)
        rows.append([
            pid, rel, size, digest, payload.n_planes, int(payload.degenerate),
            f"{g.camera_height_m:.4f}" if g else "",
            f"{g.tilt_deg:.3f}" if g else "",
            f"{g.pixel_share:.4f}" if g else "",
            f"{g.height_spread_m:.4f}" if g else "",
            f"{payload.sky_fraction:.4f}",
            g.n_standin_planes if g else "",
            f"{g.standin_pixel_share:.4f}" if g else "",
        ])

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(INDEX_FIELDS)
        w.writerows(rows)

    unavailable = {pid for pid in absent if ledger.get(pid) == "unavailable"}
    indexed = {r[0] for r in rows}
    gone = ((_load_ids(depth_dir / UNAVAILABLE_FILE) | unavailable) & set(expected_ids)) - indexed
    _write_ids(depth_dir / UNAVAILABLE_FILE, gone)
    _write_ids(depth_dir / NO_PLANES_FILE, no_planes)
    pending = absent - gone
    frame = record.get("frame_check")

    print(f"--- Reconcile: run records vs store {store} ---")
    print(f"  run panoramas (unique):  {len(expected_ids)}")
    print(f"  indexed from the store:  {len(rows)}  -> index.csv")
    if gone:
        print(f"  gone or no depth (ledger 'unavailable'): {len(gone)}  -> {UNAVAILABLE_FILE}")
    if no_planes:
        print(f"  artifact has no plane list (pre-v3): {len(no_planes)}  -> {NO_PLANES_FILE}")
    if pending:
        print(f"  no artifact yet:         {len(pending)}, e.g. {sorted(pending)[:3]}")
    if unreadable:
        print(f"  WARNING: {len(unreadable)} unreadable artifact(s), e.g. {unreadable[:2]}")
    if corrupted:
        print(f"  WARNING: {len(corrupted)} artifact(s) changed since indexed (same size, "
              f"different sha256), e.g. {sorted(corrupted)[:3]}")
    if unreadable or corrupted:
        status = f"ANOMALY — {len(unreadable)} unreadable, {len(corrupted)} altered"
    elif pending:
        status = f"PARTIAL — {len(pending)} panorama(s) have no artifact yet; re-run later"
    elif gone or no_planes:
        status = f"OK with gaps — {len(gone)} gone / no depth, {len(no_planes)} without planes"
    else:
        status = "OK — every run panorama indexed from the store"
    if frame is None:
        print(f"  frame unchecked: no --check-store-frame result in {STORE_RECORD}; run it "
              f"before anything reads this index")
        status += " (frame unchecked)"
    elif not frame.get("ok"):
        print(f"  WARNING: the recorded store frame check FAILED ({frame.get('checked_at')})")
        status += " (frame check FAILED)"
    else:
        print(f"  frame: checked {frame.get('checked_at')} on {frame.get('n_checked')} "
              f"panorama(s): {frame.get('identical')} identical, {frame.get('revised')} revised")
    print(f"  STATUS: {status}")
    return len(gone), len(no_planes), len(pending), len(unreadable) + len(corrupted)


# Asymmetric (x, y) image-frame points for the frame check: each is below the horizon and
# off the heading axis, so a mirrored index array reads a different plane at most of them.
FRAME_POINTS = ((0.08, 0.62), (0.21, 0.71), (0.33, 0.58), (0.41, 0.83),
                (0.62, 0.66), (0.71, 0.93), (0.86, 0.60), (0.93, 0.77))
FRAME_TOL = 1e-6
# Google revises a payload now and then, so a live fetch can differ from an artifact saved
# weeks earlier without anything being wrong with either frame (seen on 1 of the first 5
# Vancouver panos checked: 3 planes added, 98.3% of indices unchanged, ground identical).
# Such a pair still has to agree far better in the image frame than mirrored, and on the
# ground plane, to pass. Set on that one case, so they are a floor, not a calibration.
REVISED_MIN_AGREEMENT = 0.9    # share of pixels with the same plane index
REVISED_MIN_MARGIN = 0.1       # ...over the share that agree with the MIRRORED artifact


def compare_store_frame(raw, stored, raster=None, points=FRAME_POINTS, tol=FRAME_TOL):
    """Does a store-rebuilt payload say what the wire payload says, in the image frame?

    `raw` is depth.parse of the live payload, `stored` payload_from_npz of the artifact,
    `raster` the artifact's `depth` array if any. Reports:
      indices_identical   the plane-index arrays are byte-identical (a mirrored store fails)
      planes_identical    every (normal, distance) is identical
      index_agreement     share of pixels with the same plane index, and
      mirrored_agreement  ...the same against the artifact flipped left-right
      ground              depth.ground_plane agrees on height, tilt and spread within `tol`
      range_points        depth.ground_range_at agrees at every point in `points`
      discriminating      how many of those points sit on a different plane from their
                          mirror image (1 - x, y) -- 0 would mean the range check alone
                          could not see a flipped index array on this pano
      raster_mismatches   sampled pixels where the artifact's own raster disagrees with
                          depth.py's image-frame value `depth_at(payload, 1 - x, y)`
    and `match`: 'identical' (all of the above), 'revised' (Google has changed the payload
    since the artifact was saved, but the frame is right: see REVISED_MIN_*; ground and
    raster must still agree) or 'mismatch'. `ok` is match != 'mismatch'.
    """
    same_shape = (raw.width, raw.height) == (stored.width, stored.height)
    out = {"indices_identical": same_shape and raw.indices == stored.indices,
           "planes_identical": raw.planes == stored.planes,
           "index_agreement": None, "mirrored_agreement": None}
    if same_shape:
        w, n = raw.width, len(raw.indices)
        mirrored = b"".join(stored.indices[r * w:(r + 1) * w][::-1] for r in range(raw.height))
        out["index_agreement"] = sum(a == b for a, b in zip(raw.indices, stored.indices)) / n
        out["mirrored_agreement"] = sum(a == b for a, b in zip(raw.indices, mirrored)) / n
    g_raw, g_st = depthlib.ground_plane(raw), depthlib.ground_plane(stored)
    if g_raw is None or g_st is None:
        out["ground"] = g_raw is None and g_st is None
    else:
        out["ground"] = all(abs(getattr(g_raw, k) - getattr(g_st, k)) <= tol
                            for k in ("camera_height_m", "tilt_deg", "height_spread_m"))
    out["ground_height_m"] = None if g_raw is None else round(g_raw.camera_height_m, 4)

    def same(a, b):
        return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= tol)

    agree, disc = 0, 0
    for x, y in points:
        agree += same(depthlib.ground_range_at(raw, x, y), depthlib.ground_range_at(stored, x, y))
        disc += depthlib._plane_at(raw, x, y)[0] != depthlib._plane_at(raw, 1.0 - x, y)[0]
    out["range_points"] = agree == len(points)
    out["discriminating"] = disc

    out["raster_mismatches"] = None
    if raster is not None:
        import numpy as np
        r = np.asarray(raster)
        h, w = r.shape
        bad = 0
        for row in range(0, h, max(1, h // 32)):
            for col in range(0, w, max(1, w // 64)):
                mine = depthlib.depth_at(stored, 1.0 - (col + 0.5) / w, (row + 0.5) / h)
                theirs = float(r[row, col])
                if theirs < 0:
                    bad += mine is not None
                elif mine is None or abs(mine - theirs) > 1e-4 * max(theirs, 1.0):
                    bad += 1
        out["raster_mismatches"] = bad
    raster_ok = not out["raster_mismatches"]
    if (out["indices_identical"] and out["planes_identical"] and out["ground"]
            and out["range_points"] and raster_ok):
        out["match"] = "identical"
    elif (same_shape and out["ground"] and raster_ok
          and out["index_agreement"] >= REVISED_MIN_AGREEMENT
          and out["index_agreement"] - out["mirrored_agreement"] >= REVISED_MIN_MARGIN):
        out["match"] = "revised"
    else:
        out["match"] = "mismatch"
    out["ok"] = out["match"] != "mismatch"
    return out


FRAME_MAX_ATTEMPTS_PER_CHECK = 4    # live requests allowed per wanted check, + FRAME_MAX_EXTRA
FRAME_MAX_EXTRA = 10


def check_store_frame(store, pano_ids, n, seed=0, scratch=None, depth_dir=None):
    """Fetch live payloads for panos that have a v3 artifact in the store, drawing in a
    seeded order until `n` have been CHECKED, and compare each with its artifact
    (compare_store_frame). A pano whose live fetch fails (about a third of a year-old
    city's labeled panos are gone from GSV) or whose artifact has no plane list is reported
    and does not count, so the draw goes on -- up to FRAME_MAX_ATTEMPTS_PER_CHECK * n +
    FRAME_MAX_EXTRA live requests, so a GSV outage cannot turn into a request storm. One
    request per attempt, through fetch_depth.

    With `depth_dir` the result is written into its store record (bind_store), which is
    what the index pass reads to say whether the frame was checked. Exits non-zero on any
    disagreement, or when fewer than `n` could be checked. Returns the per-pano results."""
    import random
    import tempfile
    import numpy as np
    candidates = [pid for pid in sorted(set(pano_ids)) if store_npz_path(store, pid).exists()]
    if not candidates:
        sys.exit("no run panorama has an artifact in the store")
    order = random.Random(seed).sample(candidates, len(candidates))
    max_attempts = FRAME_MAX_ATTEMPTS_PER_CHECK * n + FRAME_MAX_EXTRA
    tmp = Path(scratch or tempfile.mkdtemp(prefix="store_frame_"))
    tmp.mkdir(parents=True, exist_ok=True)
    results, failed, attempts = [], 0, 0
    for pid in order:
        if len(results) >= n or attempts >= max_attempts:
            break
        with np.load(store_npz_path(store, pid)) as z:
            fields = {k: z[k] for k in z.files}
        try:
            stored = payload_from_npz(fields)
        except NoPlanes:
            print(f"  {pid}: artifact has no plane list, skipped")
            continue
        attempts += 1
        err = fetch_depth(pid, tmp / f"{pid}.json.gz")
        if err is not None:
            print(f"  {pid}: live fetch failed ({err[0]}: {err[1]}), not checked")
            continue
        res = compare_store_frame(read_payload(tmp / f"{pid}.json.gz"), stored,
                                  fields.get("depth"))
        res["pano_id"] = pid
        results.append(res)
        failed += not res["ok"]
        agree = res["index_agreement"]
        print(f"  {pid}: {res['match'].upper()}  indices {res['indices_identical']} "
              f"(agree {agree if agree is None else round(agree, 4)}, mirrored "
              f"{res['mirrored_agreement'] if agree is None else round(res['mirrored_agreement'], 4)}), "
              f"planes {res['planes_identical']}, ground {res['ground']} "
              f"(h {res['ground_height_m']}), range {res['range_points']} "
              f"({res['discriminating']}/{len(FRAME_POINTS)} points discriminate a mirror), "
              f"raster mismatches {res['raster_mismatches']}")
    revised = sum(r["match"] == "revised" for r in results)
    if depth_dir is not None:
        record = bind_store(Path(depth_dir), store)
        record["frame_check"] = {
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_requested": n, "n_checked": len(results), "live_requests": attempts,
            "seed": seed, "identical": len(results) - revised - failed, "revised": revised,
            "mismatch": failed, "ok": not failed and len(results) >= n,
            "panos": [{"pano_id": r["pano_id"], "match": r["match"],
                       "index_agreement": r["index_agreement"],
                       "mirrored_agreement": r["mirrored_agreement"]} for r in results]}
        _write_store_record(Path(depth_dir), record)
    if failed:
        sys.exit(f"STORE FRAME CHECK FAILED on {failed} of {len(results)} panorama(s): do not "
                 f"index this store until the frames are reconciled")
    if not results:
        sys.exit("STORE FRAME CHECK: nothing could be checked")
    if len(results) < n:
        sys.exit(f"STORE FRAME CHECK: only {len(results)} of {n} panorama(s) could be checked "
                 f"in {attempts} live request(s); the frame is not established")
    print(f"  STATUS: OK — {len(results)} store artifact(s) agree with the live payload in the "
          f"image frame ({len(results) - revised} identical, {revised} revised by the source "
          f"since the artifact was saved)")
    return results


def main(argv=None):
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
    ap.add_argument("--reindex", action="store_true",
                    help="Offline: recompute every derived index.csv column from the "
                         "archived files (after a depth.py change, e.g. #47's spread). "
                         "Fetches nothing; refuses if any indexed file is missing.")
    ap.add_argument("--check-convention", type=int, nargs="?", const=3, metavar="N",
                    help="Verify depth.py against streetlevel's raster on N live panoramas.")
    ap.add_argument("--from-store", type=Path, metavar="STORE",
                    help="Index pano-tools' depth artifacts (<id[:2]>/<id>.depth.npz) in this "
                         "pano store instead of fetching anything (issue #56).")
    ap.add_argument("--check-store-frame", type=int, nargs="?", const=5, metavar="N",
                    help="With --from-store: fetch N live payloads (N requests) and prove the "
                         "store's artifacts equal them in the image frame; writes no index.")
    args = ap.parse_args(argv)

    run_dir = args.run
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")
    check_gsv(run_dir)
    pano_ids, sources = run_pano_ids(run_dir)
    # Belt to check_gsv's braces: that reads the manifest, this the records, so a run dir
    # without a manifest is still refused.
    problem = record_source_problem(sources)
    if problem:
        sys.exit(problem)

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

    if args.check_store_frame is not None and args.from_store is None:
        sys.exit("--check-store-frame needs --from-store")
    depth_dir = args.out or (run_dir / "depth")
    # A store-bound depth dir is ALWAYS addressed with --from-store: the harvest path would
    # reconcile it as an archive of *.json.gz (finding none, it rewrites index.csv with only
    # its header), and a plain harvest would fetch every run pano live into it (PR #96).
    bound = load_store_record(depth_dir)
    if bound is not None and args.from_store is None:
        sys.exit(f"{depth_dir} indexes the pano store {bound.get('store')}; pass "
                 f"--from-store {bound.get('store')} (with --rehash to re-hash it), or --out "
                 f"another directory for a harvest.")
    if args.from_store is not None:
        if not args.from_store.is_dir():
            sys.exit(f"--from-store {args.from_store}: not a directory")
        # --verify is accepted and changes nothing: a --from-store pass fetches nothing and
        # always reconciles, so `--verify --rehash` means here what it means for a harvest.
        if args.check_store_frame is not None:
            if args.check_store_frame < 1:
                sys.exit("--check-store-frame needs at least 1 panorama")
            print(f"--- Store frame check ({args.check_store_frame} panoramas) ---")
            check_store_frame(args.from_store, pano_ids, args.check_store_frame,
                              depth_dir=depth_dir)
            return
        print(f"{run_dir.name}: {len(pano_ids)} panoramas in the run -> {depth_dir} "
              f"(from store {args.from_store}; nothing is fetched)")
        _, _, _, anomalies = reconcile_store(depth_dir, args.from_store, pano_ids,
                                             rehash=args.rehash)
        summarize(depth_dir)
        sys.exit(1 if anomalies else 0)

    depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run_dir.name}: {len(pano_ids)} panoramas in the run -> {depth_dir}")

    if args.reindex:
        gone, failed, pending, anomalies = reindex(depth_dir, pano_ids)
        summarize(depth_dir)
        sys.exit(1 if (failed or anomalies) else 0)

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
