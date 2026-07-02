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

# Run the labeler over an area; output goes to <basename>.jsonl in the CWD
python main.py example_geojson/vancouver.geojson

# Submit a produced JSONL file to a Project Sidewalk endpoint
# (edit JSONL_FILE_PATH and ENDPOINT_URL at the bottom of send_to_ps.py first)
python send_to_ps.py
```

There are no tests, linter config, or build step. `pytest` is installed but no test files exist.

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

**Caching / resumability:** results are keyed by a SHA-256 hash of the GeoJSON geometry.
Per-area state lives in `cache/<area_hash>/already_processed.txt`. The JSONL output and the
cache file are appended to and flushed line-by-line, so a run is resumable — re-running skips
panos already in the cache, and failed panos are intentionally left out of the cache so they
retry next run. Editing the geometry changes the hash and starts a fresh cache/output set.

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
