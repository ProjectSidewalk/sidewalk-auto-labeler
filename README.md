# Sidewalk Auto-Labeler

Automatically detect **curb ramps** in Google Street View (GSV) imagery across an entire
city and submit the predictions to [Project Sidewalk](https://projectsidewalk.org/).

Give it a geographic area (a GeoJSON polygon), and it will:

1. Find every GSV panorama inside that area.
2. Download each panorama and run the **RampNet** curb-ramp detection model on it.
3. Write one prediction record per panorama to a `.jsonl` file.
4. (Optional second step) POST those predictions to a Project Sidewalk server.

## How this relates to RampNet

This repo is the **deployment / inference tool**. The actual machine-learning model lives
elsewhere:

- **[ProjectSidewalk/RampNet](https://github.com/ProjectSidewalk/RampNet)** — the research
  project that *trained* the model. It generated a dataset of 210,000+ annotated GSV
  panoramas (bootstrapped from government curb-ramp location data) and trained a
  panorama-level detector on a ConvNeXtV2 backbone.
- **[`projectsidewalk/rampnet-model`](https://huggingface.co/projectsidewalk/rampnet-model)** —
  the trained model, published on HuggingFace. This repo downloads and runs it; you do not
  need the RampNet training code to use this tool.

In short: *RampNet builds the model; this repo applies it at city scale and feeds the results
into Project Sidewalk.*

## How it works

The model takes a 2048×4096 RGB panorama and outputs a **heatmap** of curb-ramp likelihood.
Local-maximum peak detection (threshold 0.55) turns the heatmap into a list of discrete
`(x, y, confidence)` detections, normalized to `[0, 1]`.

```
GeoJSON area
     │
     ▼
[1] Find panoramas        main.py        → scan map tiles (zoom 17) for GSV pano IDs
     │                                      inside the polygon
     ▼
[2] Download + detect     panorama.py    → reassemble each panorama from GSV tiles
                          curb_ramp.py    → run RampNet, extract peak detections
     │
     ▼
[3] Write predictions     <area>.jsonl   → one JSON line per processed panorama
     │
     ▼
[4] Submit (optional)     send_to_ps.py  → POST each record to Project Sidewalk
```

### Components

| File | Role |
|------|------|
| `main.py` | Entry point for stages 1–3. Finds panos, runs detection, writes JSONL. |
| `panorama.py` | Reassembles a full equirectangular panorama from individual GSV tiles. |
| `detectors/curb_ramp.py` | Wraps the `projectsidewalk/rampnet-model` HuggingFace model. |
| `send_to_ps.py` | Stage 4. Reads a JSONL file and submits predictions to Project Sidewalk. |
| `example_geojson/` | Example area polygons (`bend.geojson`, `chicago.geojson`, `vancouver.geojson`). |

## Prerequisites: a Project Sidewalk city instance must exist first

> **Read this before you start.** This repo is only the *last* link in a longer chain. It
> generates curb-ramp predictions and submits them to a Project Sidewalk server — but it
> **cannot create the city** on that server. The submission endpoint (`/ai/submitLabelsOnPano`)
> only accepts labels for a city that already exists in the database, with its regions and
> street network in place.

So deploying Bend (or any city) is really **two separate projects in two separate repos**:

| | Where | What | Repo |
|---|---|---|---|
| **A. Stand up the city** *(prerequisite)* | Server-side | Create the DB schema, import neighborhood/region boundaries and the OSM street network, configure the city, deploy the server | [`SidewalkWebpage`](https://github.com/ProjectSidewalk/SidewalkWebpage) |
| **B. Generate + submit AI labels** *(this repo)* | Your machine | Detect curb ramps across GSV imagery and POST them to that server | **this repo** |

### What "stand up the city" (A) involves

This is the larger, multi-step effort — mostly QGIS data prep — documented on the PS wiki.
Roughly:

1. **Create the schema** — `make create-new-schema name=sidewalk_bend_or`.
2. **Prepare geo data in QGIS** (EPSG:4326 WGS 84):
   - Neighborhood/region boundaries → `qgis_region` table (with `region_id`).
   - OSM street network (filtered road classes, split at intersections, single-linestrings)
     → `qgis_road` table (with `road_id`).
3. **Populate tables** — `make fill-new-schema` builds `street_edge`, `region`,
   `street_edge_region`, `region_completion`, etc.
4. **Configure** the `config` table — `city_center_lat/lng`, SW/NE map boundaries,
   `open_status`, zoom — plus `cityparams.conf`, UI translations, Google Analytics ID, and
   whitelisting the server URL on the Google Maps/Street View API key.
5. **Prune streets without imagery** — `check_streets_for_imagery.py` then
   `make hide-streets-without-imagery`.
6. **Deploy** — `pg_dump` the schema and provision server infrastructure with IT.

Reference wiki pages:
- [Creating database for a new city](https://github.com/ProjectSidewalk/SidewalkWebpage/wiki/Creating-database-for-a-new-city)
- [Adding New Road Geometries to a Database](https://github.com/ProjectSidewalk/SidewalkWebpage/wiki/Adding-New-Road-Geometries-to-a-Database)
- [Considerations when Preparing for and Deploying to New Cities](https://github.com/ProjectSidewalk/SidewalkWebpage/wiki)

> The PS wiki estimates this stage takes **significant effort** (but we are trying to streamline it), dominated by QGIS data cleaning.
> If a Bend instance already exists, skip all of this and just point `ENDPOINT_URL`
> (in `send_to_ps.py`) at it.

### Keep the two areas aligned

The polygon you feed `main.py` defines **where panoramas are searched for**. For the
submitted labels to land on real streets, that search area should describe the **same Bend**
as the regions/street network in the Project Sidewalk database. Mismatched boundaries mean
detections with nowhere valid to attach.

## Setup

A CUDA-capable GPU is strongly recommended — running the detector on CPU is very slow.

```bash
conda env create -f environment.yml
conda activate sidewalk-auto-labeler
```

> **Note:** `environment.yml` is a **linux-64 conda export** and will not solve on Windows or
> macOS. On those platforms use the portable requirements file instead:
>
> ```bash
> conda create -n sidewalk-auto-labeler python=3.12
> conda activate sidewalk-auto-labeler
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # CUDA build
> pip install -r requirements.txt
> ```

The environment pins a CUDA 12.6 build of PyTorch. On first run, the RampNet model
(~hundreds of MB) is downloaded from HuggingFace and cached locally.

## Usage

### Step 0 — Scope the area first

```bash
python main.py example_geojson/bend.geojson --name bend --scan-only
```

This scans coverage tiles only (no model load, no processing) and reports how many
panoramas are in the polygon plus a rough runtime estimate — so you know whether you're
committing to minutes or days before launching. On a resumed run it also tells you how
many panos remain.

### Step 1 — Run the labeler over an area

```bash
python main.py example_geojson/bend.geojson --name bend
```

This writes all per-area state to `runs/bend/` (`--name` defaults to the GeoJSON filename
stem):

```
runs/bend/
  results.jsonl           # one prediction record per processed panorama
  already_processed.txt   # resume cache
  manifest.json           # geometry hash, model provenance, per-run stats
  area.geojson            # exact copy of the geometry used
  spot_check/             # optional visual-QA gallery (see below)
```

The run:

- Scans the map tiles covering the polygon's bounding box, keeping only panoramas whose
  location actually falls inside the polygon.
- Skips indoor panoramas.
- Writes **one line per successfully processed panorama** — including panoramas where zero
  curb ramps were found (those get an empty `detections` list). This is intentional: it
  records that the pano was inspected.

The run is **resumable and cached** (see below), so it's safe to stop and restart.

### Alternative imagery source: Mapillary

Both steps accept `--source mapillary` to run on [Mapillary](https://www.mapillary.com/)
360° imagery (CC BY-SA) instead of GSV — e.g. for cities where Project Sidewalk deploys
on Mapillary coverage:

```bash
# Client token from mapillary.com/dashboard/developers — export it, or put
# MAPILLARY_ACCESS_TOKEN=MLY|... in a ./.env file (gitignored, loaded automatically).
export MAPILLARY_ACCESS_TOKEN="MLY|..."
python main.py example_geojson/richmond.geojson --name richmond --source mapillary --scan-only
python main.py example_geojson/richmond.geojson --name richmond --source mapillary
```

Coverage comes from Mapillary's z14 vector tiles (only 360° panoramas are kept —
`is_pano`), imagery from the Graph API's full-resolution `thumb_original_url`. Because
contributors re-drive the same streets, coverage is spatially thinned to the best pano
per ~10 m grid cell (newest capture, quality-score tiebreak) before processing. A run
directory is bound to one source the same way it's bound to one geometry; use a
different `--name` per source.

### Step 2 — Spot-check and validate the detections

```bash
python scripts/spot_check_gallery.py runs/bend
```

This renders `runs/bend/spot_check/index.html`: a one-pano-at-a-time viewer (open directly
or via any static server, e.g. VS Code Live Server) showing each sampled panorama with its
detections marked by **numbered circles** and a close-up crop per detection (the numbers
match the crop captions). Navigate with ←/→; the sample always includes the densest panos
plus a few zero-detection panos; `--sample` / `--empty-sample` / `--seed` control it. Each
pano links to its live Google Street View view.

The viewer doubles as a quick **validation tool**:

- Click a crop, its numbered circle on the panorama, or press `1`–`9` to cycle a
  detection's verdict: unjudged → correct → incorrect → **unsure**. Circles start **yellow**
  (with a white halo) and turn **green**/**red**/**blue** as you judge. Use *unsure* when the
  imagery is too blurry/distant/occluded to call — it **abstains** (the scorer drops it from
  both precision and recall) instead of forcing a guess that would bias the numbers.
- Click the panorama to mark a curb ramp the model **missed** (click the magenta marker to
  downgrade it to **unsure** amber, then again to remove it). This gives
  per-pano-comprehensive ground truth — the recall signal that Project Sidewalk's validation
  workflow can't provide. An *unsure* missed mark also abstains: it confirms you scanned the
  pano but is left out of the recall denominator.
- If a pano has **no** missed ramps, say so explicitly: press `m` or click
  **"No missed ramps"**. A pano only counts as *reviewed* once every crop is judged **and**
  you've either marked a missed ramp or affirmed there are none — so paging past a pano
  can't silently count as "no missed ramps" and inflate recall.
- Verdicts autosave in the browser (localStorage). When done, **Export verdicts** downloads
  `<name>_verdicts.json` (e.g. `bend_verdicts.json`) — **save it into the run directory**
  (`runs/bend/`), where the scorer finds it automatically:

```bash
python scripts/score_validation.py runs/bend
```

prints precision and recall with 95% CIs and a confidence-threshold sweep — both overall
and on the unbiased random subset (the always-included densest panos are flagged and
excluded there). A pano whose detections were judged but whose missed-ramp check was never
confirmed still counts toward **precision** (its crop verdicts are valid) but is excluded
from **recall** — the scorer reports how many panos were excluded this way. Detections and
missed marks flagged **unsure** are abstentions: excluded from both metrics and reported as
their own counts, so the numbers stay honest about how much was too ambiguous to call.
Verdicts exported by older galleries (no `no_missed` flags) still score as before, with a
warning that their recall may be optimistic.

If you scanned every pano for missed ramps but only *clicked* when there was one to mark
(a common habit — you never pressed "No missed ramps" on the clean panos), the per-pano
gate will wrongly hold those clean panos out of recall and bias it low toward the
missed-heavy panos. Pass `--assume-scanned` to attest that every fully-judged pano was
scanned; recall then uses all of them (the scorer prints a caveat noting the attestation).
Use
the sweep to pick a per-city threshold that clears the city's
`ai-validation-min-accuracy` with margin.

> Galleries generated before July 2026 have the old red circles baked into the images —
> re-run `spot_check_gallery.py` to regenerate (existing in-browser verdicts survive;
> they're keyed by the run's area hash, not by the images).

### Step 3 — Submit predictions to Project Sidewalk

```bash
# Preview the transformed payloads without sending anything:
python send_to_ps.py runs/bend/results.jsonl --dry-run

# Submit for real:
python send_to_ps.py runs/bend/results.jsonl --endpoint https://your-ps-server/ai/submitLabelsOnPano
```

This reads each JSONL line and POSTs it to the Project Sidewalk endpoint. It also converts
the **normalized** detection coordinates from step 1 into **pixel** coordinates
(`pano_x`, `pano_y`) using the panorama dimensions stored in each record.

- **Auth:** if the server requires Project Sidewalk's internal API key, export it as
  `PS_INTERNAL_API_KEY` (or point `--api-key-env` at another variable); it is sent as an
  `Authorization: Bearer` header. If unset, no auth header is sent.
- **Resumable:** successfully submitted line numbers are recorded in a `<file>.submitted`
  sidecar, so re-running skips them instead of re-POSTing. Delete the sidecar to resubmit
  everything.
- Transient failures (connection errors, 5xx) are retried with backoff; 4xx responses are
  treated as permanent and logged.

> The detection coordinates are stored normalized (`0–1`) in the JSONL and only converted to
> pixels at submission time. If you change the coordinate handling on one side, update the
> other.

## Deploying to a new city

You already have a Bend, Oregon polygon at `example_geojson/bend.geojson`. To run a new city
from scratch:

1. **Create a GeoJSON polygon** for the city boundary. The file must be a **bare geometry
   object** (a raw `Polygon` or `MultiPolygon`) — *not* a GeoJSON `Feature` or
   `FeatureCollection`. Compare against the existing examples; tools like
   [geojson.io](https://geojson.io) export Features, so you may need to extract just the
   `geometry` portion. A quick source for city limits is
   [OSM Nominatim](https://nominatim.openstreetmap.org/): search the city with
   `polygon_geojson=1`, take the `boundary=administrative` result, and save its `geojson`
   field (the bare geometry) to a file. Ideally, though, derive the boundary from the
   target Project Sidewalk instance's own regions so the two areas match exactly (see
   [Keep the two areas aligned](#keep-the-two-areas-aligned)).
2. Save it under `example_geojson/` (or anywhere) and run
   `python main.py path/to/city.geojson --name <city>`.
3. Review `runs/<city>/results.jsonl` and the spot-check gallery
   (`python scripts/spot_check_gallery.py runs/<city>`).
4. Point `send_to_ps.py` at `runs/<city>/results.jsonl` and the target Project Sidewalk
   server, then submit.

### Tips for a large city

- **Scope it first**: `--scan-only` gives the pano count and a runtime estimate in a
  minute or two. Throughput is GPU-bound (~1–2 s/pano; inference is serialized), so a
  large city is a multi-day single-GPU run regardless of concurrency settings.
- The two concurrency settings at the top of `main.py` are the main performance knobs:
  - `COVERAGE_API_CONCURRENCY = 100` — parallel workers scanning map tiles for pano IDs.
  - `PROCESSING_CONCURRENCY = 50` — parallel workers downloading + running detection.
- For long unattended runs, launch with `nohup` (e.g. `nohup python main.py ... &`);
  `nohup.out` is git-ignored.

The per-city checklist for expanding beyond the example cities (and open questions like
threshold policy and boundary source of truth) is tracked in
[issue #13](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/13).

## Working away from the GPU machine

Only **detection** (`main.py` without `--scan-only`) needs the GPU box. Everything else
runs on a laptop (macOS/Windows/Linux) with just internet access — install via
`requirements.txt` (the plain CPU `pip install torch torchvision` is fine since the model
never runs; or skip torch entirely and install `requirements-test.txt`, which covers every
non-detection tool):

| Works on a laptop | Needs |
|---|---|
| `main.py --scan-only` (scope a new city) | network only |
| `scripts/spot_check_gallery.py` (build/regenerate a gallery) | network + the run's `results.jsonl` |
| reviewing a gallery + exporting verdicts | a browser |
| `scripts/score_validation.py` | `results.jsonl` + verdicts |
| `send_to_ps.py` | network + `results.jsonl` |
| `pytest` | nothing else |

The catch: **`results.jsonl` is git-ignored** (it's tens of MB per city) and lives only on
the machine that ran the detection — copy it over first (e.g.
`scp makelab2:.../runs/bend/results.jsonl runs/bend/`). The small irreplaceable files —
`manifest.json`, `area.geojson`, and hand-labeled `*_verdicts.json` — are tracked in git
and sync normally.

## Caching & resumability

Each run directory is bound to the **SHA-256 hash of its GeoJSON geometry**, recorded in
`runs/<name>/manifest.json`. Progress is tracked in `runs/<name>/already_processed.txt`.

- Panoramas already listed there are skipped on re-runs.
- The JSONL output and the cache are appended and flushed line-by-line, so an interrupted run
  loses almost nothing.
- **Skipped** panoramas (indoor sources, missing/incomplete metadata) are deterministic, so
  they are cached too — they won't be refetched on re-runs (and produce no JSONL line).
- **Failed** panoramas are deliberately *not* cached, so they are retried on the next run.
- Re-running a name with an **edited geometry is refused** (the hash no longer matches the
  manifest) instead of silently mixing two areas' state — use a new `--name`, or restore the
  geometry from the run's `area.geojson`. To fully re-run an unchanged area, delete its
  `runs/<name>/` directory.
- `manifest.json` also records model provenance (`model_id`, training date, `api_version`),
  the `streetlevel` version, and per-run counts (found/processed/skipped/failed) — so a
  months-old results file is self-describing.
- **Git tracks the small, irreplaceable files** in each run directory — `manifest.json`,
  `area.geojson`, and `*_verdicts.json` (hand-labeled ground truth) — while `results.jsonl`,
  the resume cache, and spot-check galleries stay local. Archive full-city `results.jsonl`
  files as release assets rather than committing them.

## Output format

Each line of `results.jsonl` is a JSON object like:

```json
{
  "detections": [
    { "x_normalized": 0.61, "y_normalized": 0.78, "confidence": 0.91 }
  ],
  "label_type": "CurbRamp",
  "model_id": "rampnet-model",
  "model_training_date": "08-21-2025",
  "api_version": "1.0.0",
  "pano": {
    "panorama_id": "…",
    "capture_date": "2021-06",
    "width": 16384,
    "height": 8192,
    "tile_width": 512,
    "tile_height": 512,
    "lat": 44.05,
    "lng": -121.31,
    "camera_heading": 123.4,
    "camera_pitch": 1.2,
    "camera_roll": 0.3,
    "copyright": "© Google",
    "source": "…",
    "history": [ { "pano_id": "…", "date": "2019-08" } ],
    "links":   [ { "target_gsv_panorama_id": "…", "yaw_deg": 90.0, "description": "…" } ]
  }
}
```

Notes:

- `detections` is empty (`[]`) when the panorama was processed but no curb ramps were found.
- Camera `heading` / `pitch` / `roll` are written in **degrees** (converted from the
  radians returned by the metadata source).
- `model_training_date` and `api_version` are currently hard-coded in `main.py`.

## Tests

A small pytest suite covers the parts whose output Project Sidewalk consumes: the JSONL
record format, the normalized→pixel transform in `send_to_ps.py`, the validation scoring
(including old- vs new-schema verdicts), the gallery's sampling/rendering geometry, and
the in-browser review state machine (run in `node`, skipped if node is absent). No test
touches the network or loads the model.

```bash
pip install -r requirements-test.txt   # light deps only, no torch needed
pytest
```

CI runs the same suite on every push (`.github/workflows/tests.yml`).

## Requirements

- Python 3.12 (provided by the conda environment).
- A CUDA GPU (recommended) or CPU.
- Internet access to Google Street View and HuggingFace.
- A running Project Sidewalk server **with the target city already set up** (only for the
  submission step) — see [Prerequisites](#prerequisites-a-project-sidewalk-city-instance-must-exist-first).

## Project layout

```
.
├── main.py                  # Stage 1–3: find panos, detect, write JSONL
├── panorama.py              # GSV panorama download (via streetlevel)
├── detectors/
│   └── curb_ramp.py         # RampNet model wrapper
├── send_to_ps.py            # Stage 4: submit predictions to Project Sidewalk
├── scripts/
│   ├── spot_check_gallery.py  # One-pano-at-a-time viewer + validation UI (visual QA)
│   ├── score_validation.py    # Precision/recall + threshold sweep from gallery verdicts
│   └── visual_check.py        # Single-pano coordinate spot check
├── tests/                   # Pytest suite (light deps only; no network, no model)
├── example_geojson/         # Example area polygons (Bend, Chicago, Vancouver)
├── environment.yml          # Conda environment (linux-64 export)
├── requirements.txt         # Portable pip requirements (Windows/macOS or non-conda)
├── requirements-test.txt    # Test/laptop deps (everything except torch)
└── runs/                    # Per-area results + resume state (git-ignored)
```
