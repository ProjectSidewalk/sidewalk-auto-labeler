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

### Step 1 — Run the labeler over an area

```bash
python main.py example_geojson/bend.geojson
```

This produces `bend.jsonl` in the current directory (named after the GeoJSON file). The run:

- Scans the map tiles covering the polygon's bounding box, keeping only panoramas whose
  location actually falls inside the polygon.
- Skips indoor panoramas.
- Writes **one line per successfully processed panorama** — including panoramas where zero
  curb ramps were found (those get an empty `detections` list). This is intentional: it
  records that the pano was inspected.

The run is **resumable and cached** (see below), so it's safe to stop and restart.

### Step 2 — Submit predictions to Project Sidewalk

```bash
# Preview the transformed payloads without sending anything:
python send_to_ps.py bend.jsonl --dry-run

# Submit for real:
python send_to_ps.py bend.jsonl --endpoint https://your-ps-server/ai/submitLabelsOnPano
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
   `geometry` portion.
2. Save it under `example_geojson/` (or anywhere) and run `python main.py path/to/city.geojson`.
3. Review the resulting `<city>.jsonl`.
4. Point `send_to_ps.py` at that file and the target Project Sidewalk server, then submit.

### Tips for a large city

- The two concurrency settings at the top of `main.py` are the main performance knobs:
  - `COVERAGE_API_CONCURRENCY = 100` — parallel workers scanning map tiles for pano IDs.
  - `PROCESSING_CONCURRENCY = 50` — parallel workers downloading + running detection.
- For long unattended runs, launch with `nohup` (e.g. `nohup python main.py ... &`);
  `nohup.out` is git-ignored.

## Caching & resumability

Each area is identified by a **SHA-256 hash of its GeoJSON geometry**. Progress is tracked in:

```
cache/<area_hash>/already_processed.txt
```

- Panoramas already listed there are skipped on re-runs.
- The JSONL output and the cache are appended and flushed line-by-line, so an interrupted run
  loses almost nothing.
- **Skipped** panoramas (indoor sources, missing/incomplete metadata) are deterministic, so
  they are cached too — they won't be refetched on re-runs (and produce no JSONL line).
- **Failed** panoramas are deliberately *not* cached, so they are retried on the next run.
- Editing the polygon changes the hash, which starts a fresh cache and output set. (To fully
  re-run an unchanged area, delete its `cache/<area_hash>/` directory.)

## Output format

Each line of `<area>.jsonl` is a JSON object like:

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
├── panorama.py              # GSV tile download + panorama reassembly
├── detectors/
│   └── curb_ramp.py         # RampNet model wrapper
├── send_to_ps.py            # Stage 4: submit predictions to Project Sidewalk
├── example_geojson/         # Example area polygons (Bend, Chicago, Vancouver)
├── environment.yml          # Conda environment (linux-64 export)
├── requirements.txt         # Portable pip requirements (Windows/macOS or non-conda)
└── cache/                   # Per-area processed-pano cache (git-ignored)
```
