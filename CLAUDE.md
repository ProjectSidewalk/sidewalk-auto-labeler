# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A batch pipeline that finds every Google Street View (GSV) panorama inside a geographic
area, runs the RampNet curb-ramp detector on each panorama, and writes predictions to a
JSONL file. A separate script then submits those predictions to a Project Sidewalk server.

## Commands

```bash
# Set up the environment (GPU strongly recommended). requirements.txt is the single
# source of truth for pins; environment.yml just pins python and defers to it.
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# ...on Windows/macOS install the CUDA torch build first (linux-64 gets it by default):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# conda also works: conda env create -f environment.yml && conda activate sidewalk-auto-labeler

# Scope an area first: pano count + runtime estimate, no model load, nothing processed
python main.py example_geojson/bend.geojson --name bend --scan-only

# Run the labeler over an area; all per-area state goes to runs/<name>/
# (--name defaults to the geojson filename stem)
python main.py example_geojson/bend.geojson --name bend

# Same pipeline on Mapillary 360 imagery (needs MAPILLARY_ACCESS_TOKEN — a client
# token from mapillary.com/dashboard/developers — in the env or in gitignored ./.env)
python main.py example_geojson/richmond.geojson --name richmond --source mapillary

# GSV runs end with a gap-fill phase (issue #32): link-target panos the run's own
# records reference but the tile scan never enumerated (coverage churn) are fetched
# by id and kept if their metadata position is in-area. --no-gap-fill skips it;
# --gap-fill-only retrofits an existing run (add --scan-only to just count targets).
python main.py runs/paterson/area.geojson --name paterson --gap-fill-only

# GROUND TRUTH / VALIDATION now lives in RampNet, not here. The GT gallery and the
# precision/recall scorer moved to ProjectSidewalk/RampNet — decoupled from sources/ and
# merged (RampNet#26/#31): `rampnet.validation` (verdict -> P/R) + `scripts/gt_gallery.py`
# (reads benchmark/<city>/{panos,records.jsonl}, no network), with the validated splits under
# RampNet's benchmark/. This repo is production-only: enumerate -> thin -> detect -> submit.
# Its one hand-off to RampNet is the native-res imagery bundle below (only this repo can
# fetch pixels).

# Build a benchmark bundle for RampNet from a finished run: samples a spatially
# de-clustered set of panos into <bundle>/records.jsonl (each tagged with its stratum
# as benchmark_group), fetches them at native resolution, then reconciles panos vs
# records and writes index.csv. Resumable; an existing records.jsonl is never re-sampled.
python scripts/export_benchmark.py runs/clovis/results.jsonl \
    --bundle ../RampNet/benchmark/clovis --sample 100 --empty-sample 25
# ...--records-only stops after records.jsonl, for when the pixels come from an existing
# full-city archive instead (copy those ids in, then re-run without it to verify).

# Archive every processed pano of a run at native resolution (same verification).
# index.csv/decayed.txt are written beside a `panos/` dir (else into --out itself), so
# per-city manifests never collide when several cities share an archive root.
python scripts/export_benchmark.py runs/clovis/results.jsonl --out /path/to/archive/clovis/panos

# Archive GSV's depth payload for every pano of a finished run (issue #41). GSV serves
# depth alongside the metadata we already fetch, so this is a metadata-only pass (no
# imagery) and gzips to ~5-7 KB/pano — ~1 GB for all four GSV runs. Resumable, with the
# same reconcile/index.csv verification as export_benchmark.py. GSV only; Mapillary runs
# are refused. Why it matters: the ground plane's distance IS the camera height, and
# geo.py's hardcoded 2.6 m is above every observed value (issue #40).
python scripts/harvest_depth.py runs/paterson
python scripts/harvest_depth.py runs/bend --out /path/to/archive/bend/depth
python scripts/harvest_depth.py runs/paterson --verify           # reconcile only, no network
python scripts/harvest_depth.py runs/paterson --verify --rehash  # ...and re-hash every file
#   rather than trusting a matching byte size — the only way to catch silent bit rot.
python scripts/harvest_depth.py runs/paterson --check-convention # re-verify depth.py vs
#   streetlevel's own raster on live panos — run this after any streetlevel upgrade, since
#   the payload layout is undocumented and positional. (The same check runs offline against
#   a synthetic payload in tests/test_depth.py, so CI catches a convention regression too.)
# no_depth.txt and gone.txt are skip caches for the two deterministic outcomes — delete one
# to force a re-check. Both are append-only: a pass that doesn't re-encounter an id
# (--verify, --limit) must never be able to erase it.

# MULTI-VIEW FUSION (issue #27, stages 2-3). Associate a finished run's detections
# into physical-ramp sites (world-space raycast + constrained clustering; no GPU,
# no network); writes runs/<name>/sites.jsonl + sites_meta.json.
python scripts/fuse_sites.py runs/paterson
# ...--pose-ablation reports within-site spread per pitch/roll sign convention
# instead (the experiment that showed GSV equirects are already gravity-rectified).

# Score fusion against RampNet GT in world space: world P/R, the union-recall
# decomposition, stage-4 promotion calibration, vintage + match-radius ablations.
# Reads ../RampNet/benchmark/<city>/{verdicts.json,records.jsonl} as data (no
# RampNet code import); writes runs/<city>/fusion_eval/{report.md,*.csv}.
python scripts/eval_sites.py paterson
python scripts/eval_sites.py paterson --vintage-ablation

# Eyeball the fusion: one HTML card per site with a crop from every member view,
# a plan view (cameras/rays/error ellipses/fused 1-sigma) and the RampNet verdict.
# Crops are cut on makelab2's native-res archive and pulled back as a tarball, then
# cached under runs/<city>/explorer/crops/ (re-renders are offline).
python scripts/site_explorer.py richmond              # 40 GT-seen sites
python scripts/site_explorer.py sao_paulo --select fp # only the false positives
python scripts/site_explorer.py richmond --select fragment  # over-split suspects
python scripts/site_explorer.py richmond --inline     # one shareable file
# ...--local-panos <dir> cuts crops locally instead (no SSH; e.g. a RampNet bundle).

# MAPILLARY RIG TILT (issue #42). OpenSfM's `computed_rotation` sits in every Mapillary
# record's source_metadata; scripts/mapillary_tilt.py parses it (convention locked four
# ways -- see docs/mapillary-tilt-study.md) and re-runs the pose ablation, the GT eval and a
# pixel-level rectification check per sign convention. No network, no GPU; stdlib except
# `rectify`/`examples` (numpy, Pillow, SciPy) and `figures` (matplotlib).
python scripts/mapillary_tilt.py stats                 # tilt distributions + compass identity
python scripts/mapillary_tilt.py ablation              # multi-view sign lock, by tilt/grade bucket
python scripts/mapillary_tilt.py eval                  # world P/R vs RampNet GT per convention
python scripts/mapillary_tilt.py pose <mapillary_id> --run richmond

# Run the tests (no GPU/network/model; light deps via requirements-test.txt)
pytest

# Submit a produced JSONL file to a Project Sidewalk endpoint
python send_to_ps.py runs/bend/results.jsonl --dry-run
python send_to_ps.py runs/bend/results.jsonl --endpoint https://<server>/ai/submitLabelsOnPano
```

Tests live in `tests/` and cover the Project-Sidewalk-facing contracts: the JSONL record
format (`build_output_line` + the per-source pano builders in `sources/`) and the
normalized→pixel transform and PS field mapping (`send_to_ps.transform_record`). They need
only `requirements-test.txt` (no torch) and never touch the network — streetlevel and
`requests` are monkeypatched. CI runs them (`.github/workflows/tests.yml`). Keep the suite
lean: test what PS consumes; don't add test infrastructure. There is no linter config or
build step. (Validation-scoring and gallery tests moved to RampNet with their tooling.)

## Architecture

The pipeline is two stages run by two separate entry points:

**Stage 1 — detection (`main.py`)**
1. Loads a GeoJSON file. **The file must be a bare geometry object** (e.g. a raw
   `MultiPolygon`), not a `Feature` or `FeatureCollection` — `shape()` and the SHA-256 area
   hash both consume the geometry directly. See `example_geojson/` for the expected shape.
2. Converts the area bounds to Slippy Map tiles (zoom per source) and scans them
   concurrently through the imagery source (`--source`, see below) to collect all pano IDs
   whose point falls inside the area polygon.
3. For each new pano, fetches the 4096×2048 equirectangular image through the source, runs
   the detector (`detectors/curb_ramp.py`), and appends one JSON line per **successfully
   processed** pano — even when zero detections are found (`detections: []`).
4. **Gap fill** (sources providing `fetch_pano_by_id`, i.e. GSV): after the main pass, any
   link-target id in results.jsonl that was never processed is fetched by id, positioned
   from its own metadata, and kept if in-area (outside = deterministic skip, cached) —
   closing the view graph for multi-view fusion (#27). Iterates until closed; each pass is
   a `phase: gap_fill` entry in the manifest's run history.

**Imagery sources (`sources/`)** — main.py is source-agnostic; each source module
(`sources/gsv.py`, `sources/mapillary.py`) implements the interface documented in
`sources/__init__.py`: `fetch_panos_for_tile` (coverage enumeration), `fetch_pano`
(metadata + image → the JSONL `pano` block + a PIL image), and `prepare` (fail fast on
misconfiguration). `fetch_pano` distinguishes deterministic `skipped` (cached, never
retried — indoor GSV panos, non-360 or non-2:1 Mapillary images, incomplete metadata)
from retryable `failure` (left uncached).
- **gsv** (default): z17 coverage tiles + metadata/imagery via streetlevel; the original
  pipeline behavior, including panorama.py below.
- **mapillary**: z14 vector coverage tiles (`mly1_public`, MVT `image` layer — carries
  `is_pano`, so 360-filtering happens during enumeration), then one Graph API call per
  image for the signed full-res `thumb_original_url` (expires — downloaded in the same
  worker pass) + SfM-computed position/compass. Needs `MAPILLARY_ACCESS_TOKEN`. Coverage
  is near-duplicate-heavy (~1 pano/1.5 m of street), so after the scan `thin_panos`
  keeps the best pano per grid cell (newest capture, quality tiebreak; `--thin-spacing`,
  default 5 m — deliberately denser than GSV since Mapillary image quality is lower,
  proximity helps detection, and PS clustering absorbs duplicate labels. Downtown
  Richmond: 35k → 9k). The
  center column of a Mapillary equirectangular is the camera's compass bearing (same
  convention as GSV and as PS's panoX→heading math), so images are never rotated;
  `computed_compass_angle` becomes `camera_heading`, pitch/roll stay null.

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
cache so they retry next run. A run directory is bound to one geometry and one imagery
source: rerunning a name with an edited geojson or a different `--source` is refused
(checked against the manifest) instead of silently forking state. The manifest also records
`detection_storage_floor`; resuming a run whose stored floor differs from the current code's
is refused for the same reason (legacy manifests read as 0.55 — those runs stored only
operational detections).

**`panorama.py`** downloads the equirectangular image via `streetlevel.streetview.get_panorama`
using the pano metadata that `process_pano` already fetched (tile grid + true dimensions come
from the metadata, so nothing is probed). It clamps to a 2:1 aspect ratio and resizes to a
4096×2048 RGB PIL image. Returns `None` on any failure (caller treats as a retryable failure).
Do not fetch Google's tile URL (`streetviewpixels-pa.googleapis.com/v1/tile`) directly — it
returns 403 for anonymous callers since ~June 2026; streetlevel handles the required request
format (and must stay ≥ 0.12.10 for the same reason).

**`detectors/curb_ramp.py`** wraps the `projectsidewalk/rampnet-model` HuggingFace model
(loaded with `trust_remote_code=True`). It outputs a heatmap; `peak_local_max` extracts peaks
down to the **storage floor** (`DETECTION_STORAGE_FLOOR=0.1`, top-50 per pano), NOT the
decision threshold. The two-threshold contract lives in `detectors/__init__.py` (torch-free,
importable everywhere): results.jsonl deliberately stores sub-threshold candidates as raw
material for multi-view fusion (#27), and everything that *acts* on detections filters at
`OPERATIONAL_CONFIDENCE=0.55` — `send_to_ps.py --min-confidence`, `export_benchmark.py`'s
strata + bundle records, `thinning_experiment.py`'s ramp sites, `fuse_sites.py`'s
operational tier (which deliberately also associates the sub-floor band, flagged
`in_refit: false` — the one sanctioned sub-floor consumer). Detections are returned as
**normalized** `(x, y, confidence)` tuples in `[0, 1]`.

**Multi-view fusion (`geo.py`, `scripts/fuse_sites.py`, `scripts/eval_sites.py`)** —
issue #27 stages 2–3, a post-processing layer between detection and submission.
`geo.py` (repo root, stdlib-only, torch/numpy-free like `detectors/__init__.py`) is the
single home for geodesy: haversine + the declustering grid (imported back by
`export_benchmark.py`), a `LocalFrame` ENU tangent plane, and the ground raycast
`detection_ground_point` — flat-ground intersection at 2.6 m camera height with
closed-form anisotropic error from the 1024×512 heatmap quantization, **dropping** (never
clamping) rays beyond 25 m. Camera pitch/roll are deliberately NOT applied: the
`--pose-ablation` experiment measured that streetlevel's GSV equirects are already
gravity-rectified (details in `geo._world_ray`'s docstring). `fuse_sites.py` associates a
run's stored detections into physical-ramp sites (descending-confidence greedy with a
same-pano cannot-link, chi-square gating, inverse-covariance refit, residual rejection;
sub-threshold detections attach as `in_refit: false` support and never move operational
positions) and writes `runs/<name>/sites.jsonl` — cluster plus members, so the
what-to-submit decision stays late; it reads no manifest and leaves `send_to_ps.py`
untouched. `eval_sites.py` scores fusion against RampNet's benchmark verdicts in world
space (semantics mirror `rampnet.validation.collect`) and produces the stage-4 promotion
calibration. Measured 2026-08-02 (5 m match radius): world recall 0.93–0.96 vs own-view
0.72–0.83, precision 0.89–0.98 across paterson/gainesville/sao_paulo/richmond/bend.

**Mapillary rig tilt (`scripts/mapillary_tilt.py`)** — issue #42. Mapillary blocks store
`camera_pitch`/`camera_roll` as null, but `source_metadata.computed_rotation` is OpenSfM's full
world→camera rotation (axis-angle; world = east/north/up, camera = right/down/forward). Its yaw equals
Mapillary's `computed_compass_angle` to 1e-11° on every record, PS's own `MapillaryViewer.extractPitchRoll`
decomposes it the same way (pitch identical; **PS's roll sign is the negative of `geo._world_ray`'s**), and
re-rendering panos with it levels them. Median tilt is ~3° (81–85% above the 1.5° sigma the error model
assumes). Measured 2026-09-04: applying the tilt relative to *gravity* tightens multi-view agreement where
the rig itself is tilted (Clovis −40%, Laurens −28%, Richmond −13%) but loosens it where the camera rides
level on a car in hilly terrain (Morgantown +28%: camera pitch tracks the road grade with slope 0.97) —
the raycast wants tilt relative to the local road, which the sequence's SfM altitude profile provides
(`road-relative` convention: never worse than flat by >1%). Full study, figures and the recommended
production change in `docs/mapillary-tilt-study.md`. Not yet wired into `sources/mapillary.py`/`geo.py`.

**GSV depth (`depth.py`, `scripts/harvest_depth.py`)** — issues #40/#41. `depth.py` (repo
root, stdlib-only like `geo.py`) parses GSV's depth payload, which is **not a raster**: it
is a list of `{normal, distance}` planes plus one plane index per pixel, and streetlevel
computes a raster from it and then discards the planes. That matters because **the dominant
ground plane's distance IS the camera height, exactly**, and its normal is the ground tilt.
Measured across four cities, camera height is per-pano (1.11–2.50 m, tracking capture
vintage), so `geo.DEFAULT_CAMERA_HEIGHT_M = 2.6` — above every observed value — runs
**29–35% long at real detection points**; correcting only the height flattens the residual
across every range bucket, i.e. the flat-ground cotangent is right and only its constant was
wrong. Four traps live in `depth.py` rather than at call sites: the header's `offset` field
is a **uint8** at byte 7 (reading it as a uint16 swallows the first plane index and makes
~0.5% of panos unparseable); the raster is **mirrored** relative to the raw index array
(`_raw_column`); Google returns a degenerate 2-plane fallback at exactly 2.500 m that must
be filtered structurally (`DEGENERATE_MAX_PLANES`), not by testing the value; and **a range
query must not snap to a pixel**. On that last one: `depth_at` snaps, because it has to
reproduce streetlevel's raster, but a detection lands at an arbitrary coordinate, and
snapping it to one of 256 rows costs up to **6.6% of the horizontal range (±1.2 m at
20–25 m)** once the near-horizon `1/(sin·cos)` amplification is applied — an alternating
sign that reads as noise. So `ray_depth_at`/`ground_range_at` snap only the *plane lookup*
(the segmentation genuinely is per-pixel) and intersect the true ray. Watch the azimuth
there: the mirror cancels in the continuous form, and because mirroring phi flips only the
ray's x component, any plane with `nx == 0` — every level ground plane — cannot tell a
correct convention from a backwards one.
`harvest_depth.py` archives the payloads before they go away: the *JavaScript* API that
exposed depth was withdrawn in 2020 and anonymous tile access in ~2026, but the metadata
endpoint used here still serves it. `no_depth.txt`/`gone.txt` are append-only skip caches
(see the command block above). GSV only; Mapillary serves no depth (its tilt is available
but unparsed — see #42).

**Stage 2 — submission (`send_to_ps.py`)**
Reads the Stage-1 JSONL and POSTs each record to a Project Sidewalk endpoint
(`/ai/submitLabelsOnPano`). Its key job is a coordinate transform: it converts the normalized
`x_normalized`/`y_normalized` detections into **pixel** `pano_x`/`pano_y` using the pano
width/height stored in the record, renames `detections` → `labels`, and drops the original
`detections` key. It also maps the pano block onto the server's `PanoSubmission` reader
(`transform_pano`): `panorama_id` → `pano_id`, raw streetlevel source strings → the
`pano_source` enum (`gsv`/`mapillary`/`infra3d`), `target_gsv_panorama_id` → `target_pano_id`,
and guarantees `links`/`history` arrays — so legacy JSONL files stay submittable unchanged.

## Output format notes

- The detector emits normalized coordinates; Stage 1 stores them normalized in the JSONL.
  The normalized → pixel conversion happens only in `send_to_ps.py`. Keep these in sync if
  you change either side.
- `detections` in results.jsonl means "stored candidates", not "believed ramps": it includes
  peaks down to the storage floor. Any new consumer must filter at `OPERATIONAL_CONFIDENCE`
  (import it from `detectors`) unless it deliberately wants the sub-floor candidates.
- Each JSONL line carries model provenance (`model_id`, `model_training_date`, `api_version`)
  and rich pano metadata (capture date, dimensions, camera heading/pitch/roll, source,
  historical panos, and links). Heading/pitch/roll are converted from radians to degrees on
  write.
- Indoor panoramas (sources `innerspace`, `cultural_institute`, `photos:legacy_innerspace`)
  are skipped.
