# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Size budget: keep this file under 50,000 characters** (`tests/test_claude_md_size.py`
enforces it). Claude Code caps all instruction files together at 150k, and the user's global
files take about 60k of that. Commands go in [docs/commands.md](docs/commands.md); design
detail and measured findings go in [docs/architecture.md](docs/architecture.md) or a study doc.
This file keeps the rule plus a one-line pointer.

## What this is

A batch pipeline that finds every street-level panorama inside a geographic area (Google
Street View by default, or Mapillary or Panoramax), runs the RampNet curb-ramp detector on each
panorama, and writes predictions to a JSONL file. A separate script then submits those
predictions to a Project Sidewalk (PS) server. This repo is production-only:
enumerate -> thin -> detect -> submit. Ground truth, the GT gallery and P/R scoring live in
ProjectSidewalk/RampNet (`rampnet.validation`, `scripts/gt_gallery.py`, `benchmark/<city>/`).
The one hand-off to RampNet is the native-res imagery bundle, because only this repo can
fetch pixels.

## Core commands

The full reference, with every analysis and study script and its notes, is
[docs/commands.md](docs/commands.md). Read the relevant section there before you run or
change one of those scripts.

```bash
# Environment. requirements.txt is the single source of truth for pins.
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# ...on Windows/macOS install the CUDA torch build first (linux-64 gets it by default):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# conda also works: conda env create -f environment.yml && conda activate sidewalk-auto-labeler

# Scope an area first: pano count + runtime estimate, no model load, nothing processed.
python main.py example_geojson/bend.geojson --name bend --scan-only
python main.py example_geojson/bend.geojson --name bend --reuse-scan   # OFF by default (coverage churns)

# Run the labeler; all per-area state goes to runs/<name>/ (--name defaults to the file stem)
python main.py example_geojson/bend.geojson --name bend
python main.py example_geojson/richmond.geojson --name richmond --source mapillary  # needs MAPILLARY_ACCESS_TOKEN (env or ./.env)
python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax    # no token; paced 1/s (#161)
python main.py runs/paterson/area.geojson --name paterson --gap-fill-only           # GSV gap fill retrofit

# Benchmark bundle for RampNet, and a native-res archive of a whole run
python scripts/export_benchmark.py runs/clovis/results.jsonl \
    --bundle ../RampNet/benchmark/clovis --sample 100 --empty-sample 25
python scripts/export_benchmark.py runs/clovis/results.jsonl --out /path/to/archive/clovis/panos

# Archive GSV depth payloads (metadata-only pass; resumable; --verify [--rehash], --reindex)
python scripts/harvest_depth.py runs/paterson

# Multi-view fusion -> runs/<name>/sites.jsonl (defaults: --camera-height-m auto, --apply-pose auto = flat)
python scripts/fuse_sites.py runs/paterson

# Position check (main.py runs it at the end of every run) and the reposition fix
python scripts/position_check.py runs/laurens --report

# Tests (no GPU/network/model; light deps via requirements-test.txt)
pytest

# Submit to Project Sidewalk
python send_to_ps.py runs/bend/results.jsonl --dry-run
python send_to_ps.py runs/bend/results.jsonl --endpoint https://<server>/ai/submitLabelsOnPano
```

Tests live in `tests/` and cover the PS-facing contracts: the JSONL record format
(`build_output_line` + the per-source pano builders in `sources/`) and the normalized→pixel
transform and PS field mapping (`send_to_ps.transform_record`). They need only
`requirements-test.txt` (no torch) and never touch the network; streetlevel and `requests`
are monkeypatched. CI runs them (`.github/workflows/tests.yml`). Keep the suite lean: test what
PS consumes, and don't add test infrastructure. There is no linter config or build step.

### Analysis and study scripts → where to read

Each script has a section in [docs/commands.md](docs/commands.md). Its findings are in the doc
named here. Verdicts are summarised only where they set a production default.

| Script(s) | What | Doc / verdict |
|---|---|---|
| `thinning_experiment.py` | coverage vs thinning spacing | `docs/thinning-experiment.md`; 5 m stays default |
| `panoramax_asset_study.py`, `run_census.py`, `panoramax_*_figures.py` | Panoramax HD vs SD; Bayonne/Lyon census | `docs/panoramax-asset-study.md` (HD stays), `panoramax-bayonne.md`, `panoramax-lyon.md` |
| `depth_standin.py`, `harvest_depth.py --reindex` | stand-in planes out of the height spread (#47) | `docs/camera-height-study.md` addendum |
| `gsv_partial_pose.py` (+`_figures`) | fraction of GSV tilt in placement (#116) | `docs/gsv-partial-pose-study.md`; FAIL on recall → flat default, `partial` opt-in |
| `fuse_sites.py --implied-height`, `height_qc.py`, `inventory_oracle.py` | camera height (#40/#44/#79) | `docs/camera-height-study.md`, `docs/placement-oracle.md`; per-rig (d) = `auto` |
| `mapillary_height.py`, `height_gap.py` | Mapillary per-rig height (#53/#87/#89) | `docs/mapillary-camera-height.md`; applies nothing (all 2.6 m) |
| `eval_sites.py` | fusion vs RampNet GT in world space | `docs/architecture.md` (Multi-view fusion) |
| `heatmap_grid.py`, `subcell_decode.py` | 8-cell heatmap grid, sub-cell decode (#111) | `docs/heatmap-grid.md`; sigma stays 1.0, decode opt-in |
| `seam_band_130.py` | 360° seam band (#130) | `docs/seam-band-130.md`; `--border` opt-in |
| `reprojection_residual.py` | GT-free leave-one-view-out residual (#36) | `docs/reprojection-residual.md` |
| `eval_ps_clustering.py`, `clustering_metrics.py`, `clustering_eval_pooled.py`, `inventory_clustering.py`, `split_figures.py`, `ps_placement.py` | PS server clustering vs fusion (SW#4706, #106, #56) | `docs/ps-clustering-eval.md` |
| `detect_from_store.py`, `provenance_gate.py`, `reinfer.py --ids` | a run rebuilt from the PS pano store (#56) | `docs/ps-clustering-eval.md` "Step 2"; Vancouver gate STOP |
| `agree_rate.py` | AI vs crowd labels (#31) | `docs/agree-rate-gainesville.md` |
| `server_agree_check.py` | live-label validation precision, auditor vs AI | `docs/server-agree-check.md` |
| `label_frame_beta.py` | human-label vs image frame on GSV (#113) | commands.md section; lower bounds, not a placement coefficient |
| `mined_precision.py` (+`_compare`), `mined_placement_attribution.py`, `floor_infer_archive.py`, `peak_anchor.py` | precision of multi-view-mined positives (RampNet#158) | `docs/mined-precision.md` |
| `site_explorer.py` | per-site HTML cards (crops cut on makelab2) | commands.md section |
| `mapillary_tilt.py` | Mapillary rig tilt (#42) | `docs/mapillary-tilt-study.md`; road-relative withheld |
| `backfill_metadata.py --pose` | fill Mapillary pitch/roll offline | commands.md section (refuses submitted files) |
| `dem_grade.py` | DEM road grade (#51) | `docs/dem-grade-study.md`; NEGATIVE |
| `gsv_ground_plane.py` | GSV depth ground normal (#52) | `docs/gsv-ground-plane-study.md`; UNDERCUT, GSV stays flat |
| `depth_at_detection.py` | depth plane under each detection (#47) | `docs/depth-at-detection-study.md`; no FP filter |
| `aerial_sidewalks.py`, `aerial_fetch_tiles.py`, `t2n_convert.py` | Tile2Net aerial polygons (#104) | `docs/aerial-sidewalk-study.md`; NEGATIVE |
| `footway_segmentation.py` | segmenter vs depth (#47 step 2) | `docs/footway-depth-study.md` |
| `reposition.py` | rewrite flagged sequences' positions | commands.md "Position check" |
| `coverage_check.py` | post-submission backup-image coverage (#46) | commands.md section |

Analysis-only extra deps (`pandas scipy haversine`, Tile2Net's venv, Mask2Former's
torch/transformers) are **deliberately not in requirements.txt**.

## Architecture (digest)

Full text: [docs/architecture.md](docs/architecture.md). The headings below match its
sections.

### Stage 1 — detection (`main.py`)

1. Loads a GeoJSON `Polygon`/`MultiPolygon`, bare or wrapped in a `Feature`/`FeatureCollection`
   (#7). `extract_geometry` normalizes it to the bare geometry, and that is what `area.geojson`
   stores and the SHA-256 area hash covers. Non-polygonal input is refused.
2. Converts the bounds to Slippy tiles (zoom per source) and keeps tiles within
   `TILE_EDGE_BUFFER_M` = 50 m of the polygon (`tiles_intersecting`, #4). **The buffer is not
   optional:** a coverage tile returns panos up to 28 m outside its own bounds, so don't shrink
   it without re-measuring. It scans them concurrently and keeps the pano IDs inside the polygon.
   The pre-thinning list goes to `scan.json`. `--reuse-scan` loads it only when area hash,
   source, endpoint, zoom and prefilter rule all match and no tile failed. It is **off by
   default** because coverage churns (paterson ~0.3% per 4 h). The manifest records
   `scan: fresh|reused` and `scan_age_hours`.
3. For each new pano: fetch the 4096×2048 equirect, run the detector, and append one JSON line
   per **successfully processed** pano, even with `detections: []`.
4. **Gap fill** (GSV, via `fetch_pano_by_id`): link-target ids that were never processed are
   fetched by id and kept if their own position is in-area. It iterates until the view graph is
   closed, and each pass is a `phase: gap_fill` manifest entry (#32).

### Imagery sources (`sources/`)

Each source implements the interface in `sources/__init__.py`: `fetch_panos_for_tile`,
`fetch_pano` (→ JSONL `pano` block + PIL image), and `prepare` (fail fast).
`fetch_pano` returns one of two outcomes:

- **`skipped`** is deterministic: cached and never retried. Examples are indoor GSV, a non-360
  or non-2:1 image, incomplete metadata, and bytes PIL says are not an image.
- **`failure`** is retryable: left uncached. A `MemoryError` mid-decode, a short body and PIL's
  truncated-file `OSError` all count as failures (#127,
  `sources.mapillary.decode_failure_is_permanent` / `body_is_complete`), as do network/HTTP
  errors and a non-`image/*` 200.

**An image 404 is a retryable failure in both Mapillary and Panoramax.** Mapillary's signed URL
expires. Panoramax is flaky, and Jon decided (2026-10-05, reversing #137) not to read a 404 as
gone. That includes a Panoramax catalog 404. A Mapillary Graph API 400/404 is still a cached
skip.

- **gsv** (default): z17 tiles, metadata and imagery via streetlevel (see `panorama.py`).
- **mapillary**: z14 MVT `image` layer (`is_pano` filters during enumeration), then one Graph
  call per image for the signed `thumb_original_url` (downloaded in the same pass) plus the SfM
  position and compass. `thin_panos` keeps the newest pano per grid cell (`--thin-spacing`,
  default 5 m; downtown Richmond went 35k → 9k). **The centre column is the compass bearing**,
  the same as GSV and PS's panoX→heading, so images are never rotated.
  `camera_pitch`/`camera_roll` come from `computed_rotation` via `geo.mapillary_pitch_roll`
  (gravity-relative, **PS's roll sign**, null past a 45° tilt).
- **panoramax**: federated, no token. z15 `pictures` layer, then STAC items for position,
  `view:azimuth` → heading, pitch/roll, make/model, producer and licence, plus the unsigned
  `hd` asset. Thinning works as for Mapillary. `PANORAMAX_API_URL` targets one instance, and
  `prepare()` probes its STAC root, since a wrong root would otherwise read as zero coverage.
  **Politeness (#161):** every request goes through `panoramax._get(url, kind)` with per-kind
  token buckets (`sources/ratelimit.py`): picture 1/s, metadata 1/s, tile 5/s (env
  `PANORAMAX_*_PER_SECOND`, 0 = unlimited). 429/503 pauses that kind. **Limits are per
  process.** Metadata is batched 100 per `/search?ids=`. The pool is capped at 4. Lyon-size
  runs take ~48 h of downloads, and the native-res archive re-downloads everything at the same
  rate. Coordinate with the maintainers first. `--panoramax-asset sd` is opt-in (SD keeps only
  0.629 of HD's detections), keeps the sensor frame, falls back to HD per picture, and is bound
  in the manifest.

Concurrency uses plain OS threads, **not gevent**. streetlevel runs an asyncio loop per call,
and under gevent those loops collide. The knobs are `COVERAGE_API_CONCURRENCY=100` and
`PROCESSING_CONCURRENCY=50`. GPU forward passes are serialized by a lock in
`CurbRampDetector`, and preprocessing runs outside it. `--batch-size N` (#2; `main.py`,
`reinfer.py`, `detect_from_store.py`; `detectors/batching.py`) exists but **buys nothing on
the A40**: one 4096×2048 forward already saturates it, so the default stays 1. Every pass
ends with a `detector:` throughput line.

### Run directories / resumability

`runs/<name>/` holds the following:

- `results.jsonl`
- `already_processed.txt` (the resume cache, gitignored)
- `manifest.json`
- `area.geojson`
- `scan.json` (gitignored)

Both the JSONL and the cache are appended and flushed line by line. Failed panos stay out of
the cache, so they retry.

A run dir is **bound** to one value of each of the following, and a mismatch is refused (use a
new `--name`):

- geometry
- `--source`
- `--thin-spacing`, for thinning sources (#126)
- `detection_storage_floor` (legacy manifests read 0.55)
- `model_revision`
- `detection_decode` / `detection_border`
- `label_types`
- `panoramax_asset`

A pre-key manifest is bound on its first resume, with a one-time note.

### `panorama.py`

It downloads the GSV equirect via `streetlevel.streetview.get_panorama`, using metadata
`process_pano` already fetched, then clamps to 2:1 and resizes to 4096×2048 RGB. It returns
`None` on failure (retryable). **Never fetch Google's tile URL directly**: it returns 403 for
anonymous callers. streetlevel must stay ≥ 0.12.10.

### Detector and the threshold contract (`detectors/`)

`detectors/curb_ramp.py` wraps `projectsidewalk/rampnet-model` (`trust_remote_code=True`). It
outputs a heatmap, and `detectors/decode.py` extracts peaks. The thresholds and filters
(`detectors/__init__.py`, torch-free):

- **Storage floor** `DETECTION_STORAGE_FLOOR = 0.1`, top-50 per pano. `results.jsonl`
  deliberately stores sub-threshold candidates for fusion.
- **Operational** `OPERATIONAL_CONFIDENCE = 0.30` (#20): everything that *acts* on detections
  filters here. That covers `send_to_ps.py --min-confidence`, the fusion operational tier and
  `position_check.py`. Fusion also associates the sub-floor band as `in_refit: false` support,
  the one sanctioned sub-floor consumer.
- **Benchmark** `BENCHMARK_CONFIDENCE = 0.55`: every judged RampNet bundle was reviewed at
  0.55. Anything that joins a run to a bundle filters here, never at the policy value. That
  covers eval_sites, mined_precision's drift gate, export_benchmark strata, mapillary_tilt and
  thinning_experiment.
- **Nadir rig mask** `NADIR_MASK_DEG = 49.0`, `on_camera_rig(y)`. It is geometric: a
  detection steeper than 49° of dip is the vehicle. It was found on Laurens GoPro Max roof
  racks, where all 50 of 50 judged labels were false. It is applied in
  `send_to_ps.transform_record` and fusion (`FuseParams.mask_rig`, `drops['on_rig']`).
  `mask_rig=False` reconstructs pre-mask campaigns. **Deliberately not implemented:** a
  distance-from-intersection filter, which would cut 3 true labels per false one.
- **Trap:** `FuseParams.min_confidence` defaults to OPERATIONAL. Any analysis scored against a
  bundle must pass the tier explicitly. `eval_ps_clustering.py --min-confidence` defaults to
  the benchmark tier, so set it to what the city went live at.
- Detections are **normalized** `(x, y, confidence)` in `[0, 1]`. The default decode is
  argmax. `--decode gaussian` (#111) and `--border keep|wrap` (#130) are opt-in, bound per run,
  refused in mixes (`--allow-mixed-decode` / `--allow-mixed-border`), and refused by every
  bundle-joined analysis. Shipping either beside live labels is Jon's call.

**Bands.** A city that went live at 0.55 gets 0.30–0.55 as a band:
`send_to_ps.py <file> --min-confidence 0.30 --max-confidence 0.55`. It runs from its own
sidecar `<file>.band-0.3-0.55.submitted`, only on top of a campaign recorded complete at 0.55
on the unchanged file. The guard refuses each of these real failure sequences:

- an empty band (a pre-storage-floor file);
- a band re-run after a gap-fill append;
- an upward band.

A band whose sidecar is lost but whose record is complete stops with "nothing to do". Never
reach for `--ignore-submission-guard` there.

**Re-inference is not a band.** `scripts/reinfer.py` makes fresh forward passes with new
confidences (0.550011 → 0.549993 on Richmond). `--verify` compares at the tier the server
holds, on PS's pixel key and the pano geometry. `--write-band-file` builds
`results.band.jsonl`, using new records where the pano reproduced and old ones elsewhere, with
a derived `.submission.json`. **Ship from `results.band.jsonl`, never `results.f01.jsonl`.**
`--verify`'s coarse-cell diagnostic explains mismatches and never excuses them.

### Multi-view fusion (`geo.py`, `scripts/fuse_sites.py`, `scripts/eval_sites.py`)

`geo.py` (stdlib-only) is the single home for geodesy: haversine, the declustering grid,
`LocalFrame`, and the ground raycast `detection_ground_point`. The raycast intersects flat
ground at a camera height, with closed-form error (`geo.ErrorModel`), and **drops** (never
clamps) rays beyond 25 m. `sigma_peak_px` stays 1.0. That is optimistic given the 8-px coarse
cells (#111), but 2.31 failed the adoption rule. `--sigma-peak-px` opts in, and every
published table used 1.0. `fuse_sites.py` associates detections into sites:

- greedy by confidence, with a same-pano cannot-link
- chi-square gating, inverse-covariance refit and residual rejection
- output to `sites.jsonl`; it reads no manifest and never touches `send_to_ps.py`

`eval_sites.py` scores sites against RampNet verdicts in world space.
`reprojection_residual.py` is the GT-free companion. Its blind spot is a bias shared by every
view of a site.

**Defaults that encode study verdicts. Don't change them without a new pre-registered test:**

- **Camera height:** the fuse_sites CLI defaults to `--camera-height-m auto`. For GSV that is
  per-rig by capture year (#79, the inventory oracle's arm (d), `fuse_sites.gsv_rig_assignment`:
  2.0 m for a well-measured year with a median under 2.1 m, else 2.5 m). Everything else, and
  `FuseParams()` and every analysis script, uses `geo.DEFAULT_CAMERA_HEIGHT_M = 2.6`, so
  published numbers reproduce. `per-pano` (GSV depth) and `per-rig` (Mapillary, which
  currently applies nothing) are opt-in. GSV depth heights are the depth frame's, 6–16% below
  what the imagery implies.
- **Pose:** `--apply-pose auto` is **flat for every source**. GSV equirects are in the rig's
  frame, not gravity-rectified (#113). The full pose loosens multi-view agreement, and the
  partial fraction (`--apply-pose partial`, #116) failed its recall clause. Mapillary
  road-relative (`--apply-pose road`, #42) failed the shuffled-grade control, so
  `AUTO_ROAD_SOURCES = ()`, and the DEM referee (#51) was negative. gravity/road on GSV or
  Panoramax panos warns. A human PS label's `pano_y` on GSV is off from a detection's by about
  0.9 of the tilt at its bearing (#113), so AI-vs-human pixel comparisons mix two frames.
- **GSV ground plane** (#52): rotating rays into the depth payload's ground normal loosens
  agreement everywhere. Keep `apply_pose=False` with no ground-plane term. If you express a ray
  in a sloped frame, also move its origin to the foot of the perpendicular
  (`ground_frame_fields`).

### GSV depth (`depth.py`, `scripts/harvest_depth.py`)

`depth.py` (stdlib-only) parses GSV's depth payload, which is **planes plus a per-pixel plane
index, not a raster**. The dominant ground plane gives a per-pano camera height and tilt.
`sources/gsv.py` stores the following on each block:

- `camera_height_m`, non-null only when the status is `measured`
- `camera_height_spread_m`, which excludes stand-in planes since #47
- `ground_tilt_deg`
- `depth_planes`
- `camera_height_status`

**Traps that live in depth.py, not at call sites:**

1. The header `offset` is a **uint8** at byte 7.
2. The raster is **mirrored** relative to the raw index array (`_raw_column`).
3. The degenerate 2-plane 2.500 m fallback is filtered structurally (`DEGENERATE_MAX_PLANES`).
4. Google's **stand-in ground** (exactly 2.500 m, exactly vertical normal, 14% of payloads) is
   detected on the normal (`SYNTHETIC_GROUND`, `depth.is_standin`), and those panos fall back to
   2.6 m.
5. **Range queries must not snap to a pixel.** `ray_depth_at`/`ground_range_at` snap only the
   plane lookup and take the IMAGE frame (`phi = (1-x)2pi + pi/2`). `depth_at` alone takes
   streetlevel's mirrored raster (#80).

A level plane can't distinguish the two conventions, so test with tilted planes. An
`index.csv` without `n_standin_planes` is refused (`--reindex` it). `no_depth.txt`/`gone.txt`
are append-only skip caches. Mapillary serves no depth.

### Position check (`position_check.py`; `scripts/position_check.py` is a shim)

SidewalkWebpage#5361. It scores each pano's submitted position against OSM street centerlines
(one cached Overpass query; a partial answer is refused and never cached). The metric **floors
near 1.75 m**, so it gates only gross drift (#62). A Mapillary sequence is **flagged** when its
submitted field's median cross-track is > 5 m (`GROSS_OFF_STREET_M`) AND the other field is
closer by > 2 m (`RESOLUTION_FLOOR_M`) on the paired metric without being > 1.5× as scattered.
Otherwise it is `both_off` or `undecidable` and never gates. A check under another rule or
knob is stale, and main.py re-runs it. main.py runs it at the end of every run
(`--no-position-check` skips it). `send_to_ps.py` refuses a Mapillary file whose check is
missing, stale or flagged (`--ignore-position-check`). GSV and Panoramax are scored for the
record only.

**Frame consistency:** PS places a label once, at insert. Resending moved panos **duplicates**
their live labels unless those are soft-deleted first, which makes repositioning a whole-city
decision. Where a pano is live is decided by **newest campaign wins, per pano**
(`position_check.live_positions`, over every `*.submission.json` in the dir).
`send_to_ps.check_live_positions` and `reposition.py` refuse anything that would put a pano
elsewhere (`--reposition-live-city` overrides, and the reason is recorded). Confirm a
reposition output **in place** with `--results`, and never swap it into `results.jsonl`.

### Stage 2 — submission (`send_to_ps.py`)

It POSTs each record to `/ai/submitLabelsOnPano`. On the way it applies these transforms:

- **Pixels:** normalized x/y → pixel `pano_x`/`pano_y` from the record's width/height, and
  `detections` → `labels`.
- **Pano block** (`transform_pano`): `panorama_id` → `pano_id`; a raw source → the
  `pano_source` enum, with the raw string kept as `source_detail` (#23);
  `target_gsv_panorama_id` → `target_pano_id`; and guaranteed `links`/`history` arrays, so
  legacy files stay submittable.
- **`source_metadata`:** PS stores it verbatim and 400s over 64 KB, so GSV's is cut to the
  **allow-list** `PS_GSV_SOURCE_METADATA_KEYS` (uploader, uploader_icon_url, upload_date,
  elevation, country_code) plus `source_detail` inside the blob. Mapillary/Panoramax blobs go
  unchanged. `check_source_metadata_size` refuses an over-cap record.

**Submission guards. Duplicate labels on PS cannot be retired (insert-only, SW#5382):**

- Resume state is `<file>.submitted` (line numbers), plus `<file>.submission.json`. That holds
  the sha256 and byte length and, per endpoint, the line/label counts, `min_confidence` and
  timestamps. It is git-tracked, recounted from sidecar plus file on write, written
  atomically, and an unreadable record **refuses**.
- `check_resume_state` refuses each of these:
  - a changed hash;
  - a record with more lines than the sidecar;
  - sidecar lines that went to another endpoint;
  - a different `--min-confidence`.
- To move test→prod, rename the sidecar aside; never delete it. `--ignore-submission-guard`
  overrides all of it, for hand-checked cases only. Dry runs read and write nothing.
- **Pure append resumes** (#59). `append_check` proves the recorded `total_bytes` prefix
  hashes to the recorded sha256, ends on a newline and holds `total_lines`, **and** that the
  appended `panorama_id`s are disjoint from the prefix's. A record without `total_bytes` is
  migrated with `--prefix-digest <total_lines>`, never overridden.
- **Multi-type files (#168):** one POST per (line, type), per-type sidecars
  `<file>.type-<T>.submitted`, and `endpoints[ep].types[T]` in the record. Bands are refused
  on multi-type files. A single-type CurbRamp file is byte-identical to before (pinned by
  `tests/test_multitype_submit.py`).

## Output format notes

- Detections are stored **normalized**. The normalized → pixel conversion happens only in
  `send_to_ps.py`, and the pixel is always `detectors.ps_pixel` (x modulo width). Every analysis
  that rebuilds PS's `(pano_id, pano_x, pano_y)` key calls it, never `round(x * w)` (PR #152).
- `detections` means **stored candidates** down to the storage floor, not believed ramps. Any
  new consumer must filter at `OPERATIONAL_CONFIDENCE` (import it from `detectors`) unless it
  deliberately wants sub-floor candidates.
- Each line carries model provenance and rich pano metadata. That includes capture date,
  dimensions, heading/pitch/roll (radians → degrees on write), source, history and links.
- Mapillary pitch/roll are **derived** from SfM `computed_rotation`, and
  `camera_pose_source = "mapillary_computed_rotation"` says so.
- Indoor GSV sources (`innerspace`, `cultural_institute`, `photos:legacy_innerspace`) are
  skipped.
- Every source's block carries `camera_make`/`camera_model`/`camera_type` + `source_metadata`
  (`provenance_fields()`). GSV's `source_metadata` is an **explicit per-field projection**
  (`sources/gsv.SOURCE_METADATA_FIELDS`), never `vars()`/`asdict()`. Every key is always
  present, values are JSON-coerced, and a failing converter records `null`. Only the allow-list
  reaches PS.
- **Model provenance is resolved, never declared** (#39/#6). `CurbRampDetector.provenance`
  reads the HF revision SHA it loaded (retrying `local_files_only` offline), and `model_id` is
  `rampnet-model@<12 hex>`. The training date comes from `detectors.KNOWN_REVISIONS`. **An
  unknown SHA refuses to start.** After a retrain, add the SHA to that table rather than using
  `--allow-unknown-model-revision`, whose null date `send_to_ps.py` refuses.
- `pano.copyright` is an **attribution ingredient**: the contributor's bare name for Mapillary
  and Panoramax (#61, SW#5360). PS composes ©, provider and licence around it. GSV stores
  streetlevel's `© <year> Google` verbatim.

## Multiple label types (#168)

This is the plumbing for a second detector (crosswalk / NoCurbRamp, Planning#36). **No second
detector exists yet**, and CurbRamp is the only spec. Full text is in
[docs/architecture.md](docs/architecture.md#multiple-label-types-168).

- **Registry** (`detectors/registry.py`, torch-free): `DetectorSpec` (thresholds and
  `mask_rig` per type), `SPECS`, `spec_for`, `parse_types`, `load_detectors` (one shared
  inference lock when there is more than one type). A consumer reads thresholds and the rig
  mask from the spec of **that detection's** type.
- **Readers use the helpers**, never the keys: `record_types`, `detection_type` (raises on an
  untyped detection in a multi-type line), `provenance_for`, `detections_of_type`. Pre-#168
  lines read as CurbRamp.
- **Writer rule** (`main.build_output_line`): a `{CurbRamp}`-only run writes the pre-#168 line
  with **no new key** (byte-pinned in `tests/test_multitype_records.py`). Other sets write
  per-detection `label_type`, sorted `label_types`, `models: {T: provenance}`, and a line-level
  `label_type` only when exactly one type ran.
- `main.py --types` (comma list). The manifest binds `label_types` and a per-type `models`
  map.
- **Consumers:** `fuse_sites.py --label-type` (a non-CurbRamp fuse writes
  `sites.<type>.jsonl`) and `export_benchmark.py --label-type`. `agree_rate`, `eval_sites`,
  `mined_precision` and `provenance_gate` **refuse** anything but `{CurbRamp}`.
- **Not done:** per-type `FuseParams`, the ranker review-order export, and a second detector.
