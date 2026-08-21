# Handoff: Paterson, NJ — Phase 1 of a new RampNet benchmark split

**Written 2026-07-29 by the RampNet-side Claude Code session, for the agent working in
`sidewalk-auto-labeler`.** This file is untracked scratch — do not commit it. Everything
command-level below was verified against this repo's code today, not guessed.

## Mission

Produce the Phase-1 deliverable for a new RampNet benchmark city, **paterson**:

> `D:\Git\RampNet\benchmark\paterson\{records.jsonl, panos/}` via `main.py --source gsv`
> followed by `scripts/export_benchmark.py --bundle`.

The full protocol lives in **`D:\Git\RampNet\docs\adding_a_benchmark_city.md`** — read it
first. Phase 1 (this repo: detect + export bundle) is yours; phases 2–6 (GT review, scoring,
operating-point analysis, docs) happen on the RampNet side and are **not** your job. The
two-repo boundary: this repo answers *"what are the curb ramps in this city?"*; RampNet
answers *"how good is the model?"*.

## Why Paterson (decision record — put this in the run's PR/issue)

- RampNet's runbook names the most valuable next split: a **second GSV city**, because bend
  (the only GSV auto-labeler split) is also a Stage-2 training city, confounding "GSV" with
  "in-domain" in the operating-point analysis (`RampNet/docs/operating_point.md`).
- Paterson is **clean per RampNet's contamination registry** (`RampNet/docs/data_provenance.md`):
  it is neither an open-data Stage-2 city (NYC/Portland/Bend) nor one of the twelve
  Project Sidewalk crop-model cities. It appears nowhere in the RampNet repo (grep-verified
  2026-07-29).
- Bonuses: first dense-urban East-coast **GSV** fabric in the benchmark (Annapolis/Richmond
  are East coast but Mapillary), and a live Project Sidewalk deployment
  (`sidewalk-paterson.cs.washington.edu`, active Jan–Jul 2026) enabling a future
  agree-rate comparison on the exact same footprint.
- **Caveat to record verbatim in Phase 0**: purpose = "second GSV city; out-of-training but
  metro-adjacent" — Paterson sits in the NYC metro (NYC = Stage-2 training city;
  Teaneck/Oradell/Cliffside Park, all next-door Bergen County, = crop-model cities). Do not
  present it as a geographic-transfer test.
- Runner-up considered and shelved: Vancouver, WA (also clean, also PS-deployed, but
  suburban PNW — less new fabric than Paterson).

## Step 1 — Boundary geojson from the PS API

The deployment's own regions API is the boundary source (reproducible provenance, and it
matches the crowd-label footprint by construction):

- **Endpoint**: `https://sidewalk-paterson.cs.washington.edu/v3/api/regions` — anonymous,
  returns 200. (The docs URL `/v3/api-docs/regions` is login-gated and 303-redirects to
  `anonSignUp` — don't burn time on it.)
- **Verified 2026-07-29**: FeatureCollection of **68** neighborhood MultiPolygons;
  `shapely.ops.unary_union` dissolves them to a **single Polygon, ~22.5 km²**
  (bbox ≈ −74.206..−74.128, 40.889..40.942). Paterson's land area is ~22.6 km², so the
  deployment covers essentially the whole municipality — no coverage-gap caveat.
- **Format trap**: `main.py` requires the geojson file's **root to be a bare geometry
  object** — `get_geojson_hash()` hashes the root as the geometry (main.py:64) and
  `run_labeler` passes it straight to `shapely.shape()` (main.py:226). A raw
  FeatureCollection will crash. Save the dissolved geometry alone.

Recipe (this repo's `.venv` has shapely):

```python
import json, urllib.request
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

fc = json.load(urllib.request.urlopen(
    "https://sidewalk-paterson.cs.washington.edu/v3/api/regions"))
u = unary_union([shape(f["geometry"]) for f in fc["features"]])
u = u.buffer(0)  # clear any sliver interior rings from adjacent-polygon dissolve
assert u.geom_type in ("Polygon", "MultiPolygon")
# inspect u.interiors if Polygon — tiny slivers between neighborhoods should be filled,
# a genuine hole (there shouldn't be one) should not
json.dump(mapping(u), open("paterson.geojson", "w"))
```

**Commit `paterson.geojson` to this repo**, and record in the commit/PR: source endpoint,
fetch date, feature count, dissolved area. PS can redraw regions; the run manifest's
`area_hash` is tied to this snapshot, and `main.py` refuses to reuse a run dir if the
geometry changes — that refusal is a feature, don't work around it.

## Step 2 — Scan, then decide where to run detection

```powershell
.venv\Scripts\python.exe main.py paterson.geojson --scan-only
```

`--scan-only` skips the torch import and model load, and prints the GSV pano count plus a
runtime estimate. Run this first, locally — it's cheap.

**Do not launch the full detection run without checking with Jon where it should run.**
His standing rule: the desktop (RTX 3070) is for smoke tests only; real runs go on Hyak or
makelab2. But this workload is **network-heavy** (GSV tile fetches) as well as GPU-bound,
which may change the calculus — check how the previous city runs (richmond, annapolis,
morgantown, clovis, budapest_district5) were executed (git history / issues in this repo)
and follow that precedent. A `--limit 25` smoke run locally to validate the pipeline
end-to-end before the real run is cheap and worth it. Useful knobs if Google throttles:
`--processing-concurrency` (default 50). `--thin-spacing` is a no-op for GSV — ignore it.

The run is resumable: processed pano IDs are cached under `runs/paterson/`, so an
interrupted run continues where it stopped.

## Step 3 — Export the benchmark bundle

```powershell
.venv\Scripts\python.exe scripts\export_benchmark.py runs\paterson\results.jsonl `
    --bundle D:\Git\RampNet\benchmark\paterson
```

- **Defaults are the contract — do not override them.** `--sample 100 --empty-sample 25
  --seed 0` with `TOP_N_BY_COUNT = 5` yields the canonical strata **5 top / 95 random /
  25 empty = 125 panos**, spatially de-clustered. Changing any of it makes the split
  non-comparable to the existing seven.
- **The exporter's exit code is a gate.** It exits non-zero if the archive is not
  trustworthy (a pano with no matching record, a failed fetch that isn't source decay).
  Non-zero ⇒ stop and fix; do not hand a bundle past a non-zero exit.
- It writes `index.csv` (sha256 per pano) and `decayed.txt` — read `decayed.txt`, note the
  count in your report.
- If the spatial-spacing pass keeps fewer than the requested counts, it prints a
  `spatial sampling: kept N/M` warning — surface that in your report rather than letting
  the strata silently shrink.

## Step 4 — Quality checks before handing back to RampNet

1. **Camera provenance**: confirm `camera_make` / `camera_model` are populated in the
   bundle's `records.jsonl`. RampNet's `tier_of()` classifies the rig from these; missing
   provenance forces a fallback table entry on their side.
2. **Capture-date recency**: summarize the GSV capture-date distribution (year histogram)
   across the 125 sampled panos. This was the one open unknown in the city decision — old
   imagery in Paterson would be a finding worth flagging loudly.
3. **Strata counts**: the exporter prints `top X / random Y / empty Z` — confirm 5/95/25.
4. **`panos/` is git-ignored on the RampNet side and irreplaceable if GSV coverage churns.**
   Keep the local copy under `runs/` too, and note in your report that the bundle needs a
   durable archive home (RampNet issue #21 tracks HF archiving; it is currently hardcoded
   to bend + richmond and lags the repo).

## Reporting

Per the RampNet research culture (its `CLAUDE.md`): record the run **as it happens** in a
GitHub issue in this repo — boundary provenance, scan count, where the detection ran,
exporter output, decayed count, the quality-check results above. Commit and push early;
don't let the only copy of anything sit uncommitted.

## Hand back to RampNet

When `benchmark/paterson/{records.jsonl, panos/}` exists and the checks pass, report back:
pano count scanned, detection stats, strata confirmation, camera/date summary, decayed
count, and the issue link. The RampNet side picks up at Phase 2 (GT review via
`scripts/gt_gallery.py`) — reviewer expectations for Paterson: more degraded/legacy ramps
than any existing split, so the review-notes field will matter (Budapest precedent).
