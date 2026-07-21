# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A batch pipeline that finds every Google Street View (GSV) panorama inside a geographic
area, runs the RampNet curb-ramp detector on each panorama, and writes predictions to a
JSONL file. A separate script then submits those predictions to a Project Sidewalk server.

## Commands

```bash
# Set up the conda environment (CUDA build of torch; GPU strongly recommended)
conda env create -f environment.yml
conda activate sidewalk-auto-labeler

# Scope an area first: pano count + runtime estimate, no model load, nothing processed
python main.py example_geojson/bend.geojson --name bend --scan-only

# Run the labeler over an area; all per-area state goes to runs/<name>/
# (--name defaults to the geojson filename stem)
python main.py example_geojson/bend.geojson --name bend

# Render a one-pano-at-a-time viewer of sampled detections (also a validation UI:
# judge crops correct/incorrect/unsure, click the pano to mark missed ramps
# (or downgrade a mark to unsure) or affirm "no missed ramps" — required for a
# pano to count as reviewed; "unsure" abstains from both metrics — then export
# <name>_verdicts.json and save it into the run directory)
python scripts/spot_check_gallery.py runs/bend

# Score the saved verdicts: precision/recall + confidence-threshold sweep
# (finds runs/bend/bend_verdicts.json automatically)
python scripts/score_validation.py runs/bend

# Run the tests (no GPU/network/model; light deps via requirements-test.txt)
pytest

# Submit a produced JSONL file to a Project Sidewalk endpoint
python send_to_ps.py runs/bend/results.jsonl --dry-run
python send_to_ps.py runs/bend/results.jsonl --endpoint https://<server>/ai/submitLabelsOnPano
```

Tests live in `tests/` and cover the Project-Sidewalk-facing contracts: the JSONL record
format (`build_output_line`), the normalized→pixel transform (`send_to_ps.transform_record`),
validation scoring (`score_validation.collect`, both verdict schemas), gallery
sampling/geometry, and the viewer's review state machine (driven in `node`; skipped when
node is missing). They need only `requirements-test.txt` (no torch) and never touch the
network — streetlevel is monkeypatched. CI runs them (`.github/workflows/tests.yml`).
Keep the suite lean: test what PS consumes; don't add test infrastructure. There is no
linter config or build step.

## Architecture

The pipeline is two stages run by two separate entry points:

**Stage 1 — detection (`main.py`)**
1. Loads a GeoJSON file. **The file must be a bare geometry object** (e.g. a raw
   `MultiPolygon`), not a `Feature` or `FeatureCollection` — `shape()` and the SHA-256 area
   hash both consume the geometry directly. See `example_geojson/` for the expected shape.
2. Converts the area bounds to Slippy Map tiles (zoom 17) and scans them concurrently via
   `streetlevel.streetview.get_coverage_tile` to collect all pano IDs whose point falls
   inside the area polygon.
3. For each new pano, downloads the equirectangular image (`panorama.py`), runs the
   detector (`detectors/curb_ramp.py`), and appends one JSON line per **successfully
   processed** pano — even when zero detections are found (`detections: []`).

Concurrency uses plain OS threads (`concurrent.futures.ThreadPoolExecutor`) — **not gevent**.
streetlevel's sync imagery API runs an internal asyncio event loop per call (`asyncio.run` +
aiohttp), which requires one real thread per concurrent call; under gevent monkey-patching
those loops collide and every download stalls for minutes. Two pools
(`COVERAGE_API_CONCURRENCY=100` for tile scanning, `PROCESSING_CONCURRENCY=50` for per-pano
work) are the main tuning knobs. GPU inference is serialized by a lock inside
`CurbRampDetector` — download threads overlap, forward passes don't (VRAM limit).

**Run directories / resumability:** all per-area state lives in `runs/<name>/` —
`results.jsonl`, the resume cache (`already_processed.txt`), `manifest.json` (geometry hash,
model provenance, streetlevel version, per-run stats), and `area.geojson` (exact copy of the
geometry used). The JSONL and cache are appended to and flushed line-by-line, so a run is
resumable — re-running skips cached panos, and failed panos are intentionally left out of the
cache so they retry next run. A run directory is bound to one geometry: rerunning a name with
an edited geojson is refused (hash check against the manifest) instead of silently forking
state. `scripts/spot_check_gallery.py runs/<name>` renders a sampled HTML gallery of annotated
detections into `runs/<name>/spot_check/`.

**`panorama.py`** downloads the equirectangular image via `streetlevel.streetview.get_panorama`
using the pano metadata that `process_pano` already fetched (tile grid + true dimensions come
from the metadata, so nothing is probed). It clamps to a 2:1 aspect ratio and resizes to a
4096×2048 RGB PIL image. Returns `None` on any failure (caller treats as a retryable failure).
Do not fetch Google's tile URL (`streetviewpixels-pa.googleapis.com/v1/tile`) directly — it
returns 403 for anonymous callers since ~June 2026; streetlevel handles the required request
format (and must stay ≥ 0.12.10 for the same reason).

**`detectors/curb_ramp.py`** wraps the `projectsidewalk/rampnet-model` HuggingFace model
(loaded with `trust_remote_code=True`). It outputs a heatmap; `peak_local_max` extracts peaks
above `threshold_abs=0.55`. Detections are returned as **normalized** `(x, y, confidence)`
tuples in `[0, 1]`.

**Stage 2 — submission (`send_to_ps.py`)**
Reads the Stage-1 JSONL and POSTs each record to a Project Sidewalk endpoint
(`/ai/submitLabelsOnPano`). Its key job is a coordinate transform: it converts the normalized
`x_normalized`/`y_normalized` detections into **pixel** `pano_x`/`pano_y` using the pano
width/height stored in the record, renames `detections` → `labels`, and drops the original
`detections` key.

## Output format notes

- The detector emits normalized coordinates; Stage 1 stores them normalized in the JSONL.
  The normalized → pixel conversion happens only in `send_to_ps.py`. Keep these in sync if
  you change either side.
- Each JSONL line carries model provenance (`model_id`, `model_training_date`, `api_version`)
  and rich pano metadata (capture date, dimensions, camera heading/pitch/roll, source,
  historical panos, and links). Heading/pitch/roll are converted from radians to degrees on
  write.
- Indoor panoramas (sources `innerspace`, `cultural_institute`, `photos:legacy_innerspace`)
  are skipped.
