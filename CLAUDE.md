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

# ...or on Panoramax (federated open imagery, no token; coverage is mostly France today)
python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax

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

# Score Project Sidewalk's SERVER-SIDE label clustering against RampNet GT (SW#4706 step 1;
# protocol + findings in docs/ps-clustering-eval.md). Pulls the city's CurbRamp labels and
# the server's clusters from the v3 API, maps every AI label back to its stored detection,
# and scores the deployed partition, the PS algorithm re-run at a threshold sweep (on the
# server's positions and on the labeler's raycast), and fuse_sites.py, all with one scorer.
# The headline metric is COVERAGE (a cluster of that arm within the match radius); the
# `recall (union)` column is eval_sites' definition, which credits a self-detected ramp
# whether or not any cluster landed on it, and is kept only for the tie-back — read the
# `no cluster` column beside it. Needs THREE packages the pipeline does not
# (`pip install pandas scipy haversine`; the script says so if they are missing) —
# deliberately not in requirements.txt, since this is an analysis tool, not the pipeline.
# --ps-script points at SidewalkWebpage/scripts/label_clustering.py for the
# verbatim-reproduction check. The two API pulls are cached in the output dir and REUSED on
# a re-run (the run prints how old they are) — pass --refresh to re-pull, since the server
# re-clusters nightly. --camera-height-m sets the scoring frame (the server's own is
# 2.341219672825709) and picks the default output dir, so the two frames never overwrite
# each other. report.md/arms.csv are git-tracked like manifest.json; the two API geojson
# are not, so the report records each pull's url, fetch time, sha256 and feature count.
python scripts/eval_ps_clustering.py richmond --server https://sidewalk-richmond.cs.washington.edu

# Eyeball the fusion: one HTML card per site with a crop from every member view,
# a plan view (cameras/rays/error ellipses/fused 1-sigma) and the RampNet verdict.
# Crops are cut on makelab2's native-res archive and pulled back as a tarball, then
# cached under runs/<city>/explorer/crops/ (re-renders are offline).
python scripts/site_explorer.py richmond              # 40 GT-seen sites
python scripts/site_explorer.py sao_paulo --select fp # only the false positives
python scripts/site_explorer.py richmond --select fragment  # over-split suspects
python scripts/site_explorer.py richmond --inline     # one shareable file
# ...--local-panos <dir> cuts crops locally instead (no SSH; e.g. a RampNet bundle).

# POSITION CHECK (SidewalkWebpage#5361) — a STANDARD part of the pipeline, not a step to
# remember: main.py runs it at the end of every run (--no-position-check skips it, e.g. no
# internet egress) and send_to_ps.py REFUSES a Mapillary file whose position_check.json is
# missing, stale (results_sha256 mismatch) or flagged (--ignore-position-check overrides).
# The manual commands below are for re-checks and for confirming a reposition output. Scores
# every pano's position against OSM street centerlines (one cached Overpass query, no GPU,
# no imagery, no second source needed). Mapillary serves two positions per image and the
# run submits one of them (--mapillary-position, default sfm = computed_geometry); SfM
# sequences can sit 8-10 m off the street as a block while raw GPS (geometry) does not, and
# every label inherits its pano's error 1:1. Exit 1 = some sequence is off the street on the
# submitted field (>3 m median signed offset, or most of it beyond 30 m of any street) AND
# the other field moves it >= 2 m closer — a swap forces a new submission campaign, so it has
# to buy something. The submitted field is judged per sequence from the coordinates, so a
# mixed (repositioned) file is checked correctly. Writes position_check.json + (--report) a
# self-contained position_report.html, both git-tracked beside manifest.json. A partial
# Overpass answer (HTTP 200 + `remark`) is refused and never cached.
python scripts/position_check.py runs/laurens --report
python scripts/position_check.py runs/laurens --report --labels <ps_v3_rawLabels.geojson> \
    --reference runs/laurens_gsv     # optional: the server's own placements; a second run over
                                     # the same area as an independent layer (Laurens only)
# ...then fix a flagged run WITHOUT re-detecting: rewrite the flagged sequences' pano lat/lng
# from the recommended field into a new file (new hash -> fresh submission campaign).
python scripts/reposition.py runs/laurens/results.jsonl --from-check
python scripts/reposition.py runs/laurens/results.jsonl --field raw   # whole file, one field
# ...and confirm the output IN PLACE — never swap it into results.jsonl (main.py's field
# binding cannot see inside the file, so a resume would append the manifest's field to a
# mixed file). Outputs land beside it as results.check.position_check.json / _report.html.
python scripts/position_check.py runs/laurens --results runs/laurens/results.check.jsonl

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
- **panoramax**: the federated open imagery commons (IGN + OSM France; CC BY-SA / Etalab),
  no token. z15 vector tiles from the federation catalog (`pictures` layer carries `type`,
  so 360-filtering happens during enumeration), then one STAC item request per picture
  (`/api/pictures/{id}`) for position, `view:azimuth` → `camera_heading`, pitch/roll,
  camera make/model, producer + license, and the `hd` asset — the original upload on the
  picture's home instance, unsigned. Same thinning as Mapillary (newest per cell,
  pixel-density tiebreak). `PANORAMAX_API_URL` targets one instance instead of the
  federation, and `prepare()` probes that root's STAC landing page so a mistyped one fails
  fast (the tile endpoint answers 204 for an empty tile and 404 for a bad path, so a wrong
  root would otherwise read as a legitimate zero-coverage scan). Records carry `license` and `panoramax_instance`; `source_metadata` is the
  STAC properties (EXIF included) minus the viewer's tile descriptors.

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

**Position check (`position_check.py`, repo root; `scripts/position_check.py` is a shim)** —
SidewalkWebpage#5361. Stdlib-only like `geo.py`. Every pano's submitted position is scored
against OpenStreetMap street centerlines (one Overpass query, cached beside the run in
`osm_streets.json`; a partial answer — HTTP 200 + `remark` — is refused and never cached).
For Mapillary runs both positions are scored and each sequence gets a verdict: **flagged**
when the median signed offset on the submitted field exceeds 3 m (or most of the sequence
is beyond 30 m of any street) AND the other field moves it ≥ 2 m closer; **both_off** when
it is off but a switch would not buy that (wide one-way streets driven once). The submitted
field is voted per sequence from the coordinates, so a mixed reposition output is judged
correctly. It is wired into both stages: `main.py` ends every run with
`position_check.run_check` (non-fatal; the verdict summary lands in `manifest.json` under
`position_check`, the full `position_check.json` + `position_report.html` beside
`results.jsonl` and are git-tracked), and `send_to_ps.py`'s `check_position_state` refuses a
Mapillary file whose check is missing, stale (`results_sha256` ≠ the file) or flagged —
`--ignore-position-check` overrides, dry runs are exempt. `scripts/reposition.py` rewrites
flagged sequences' pano lat/lng from the other field into a new file (a *submission
artifact*, never swapped into `results.jsonl`: the run-dir field binding cannot see inside
the file), which is confirmed with `--results` and submitted from where it is. What it
cannot catch: a bias both fields share, drift under 3 m, and anything on a source with one
position (GSV/Panoramax are scored for the record but never gated).

**Stage 2 — submission (`send_to_ps.py`)**
Reads the Stage-1 JSONL and POSTs each record to a Project Sidewalk endpoint
(`/ai/submitLabelsOnPano`). Its key job is a coordinate transform: it converts the normalized
`x_normalized`/`y_normalized` detections into **pixel** `pano_x`/`pano_y` using the pano
width/height stored in the record, renames `detections` → `labels`, and drops the original
`detections` key. It also maps the pano block onto the server's `PanoSubmission` reader
(`transform_pano`): `panorama_id` → `pano_id`, raw streetlevel source strings → the
`pano_source` enum (`gsv`/`mapillary`/`infra3d`), `target_gsv_panorama_id` → `target_pano_id`,
and guarantees `links`/`history` arrays — so legacy JSONL files stay submittable unchanged.

Resume state is a `<file>.submitted` sidecar of **line numbers**, which silently stops
describing the campaign if the JSONL is edited, if the sidecar is lost (deleted, or a run
moved to a second machine), or if the same file is pointed at a second server — the first
two re-POST records that are already live and duplicate a whole city's labels; the third
skips the staged lines on production so they never reach it. So each campaign also writes
`<file>.submission.json`: the JSONL's sha256 and, **per endpoint**, line/label counts and
timestamps — the one submission artifact small and stable enough to commit
(`runs/*/*.submission.json` is git-tracked like `manifest.json`). Both counts are recounted
from the sidecar and the file when the record is written (also after Ctrl-C), never from
the run's own tallies, so record and sidecar cannot drift; the write is atomic, and an
existing-but-unreadable record (merge conflict, truncated write) **refuses** rather than
reading as "nothing sent". Before POSTing anything, `check_resume_state` refuses when the
file's hash changed, when the record says more lines went to this endpoint than the
sidecar holds, when the sidecar holds *more* lines than this endpoint is recorded to have
while another endpoint has a count (they went there), or when `--min-confidence` differs
from the one this endpoint was submitted at — the test→prod move is "rename the sidecar
aside", never delete. `--ignore-submission-guard` overrides all of it, for a case checked
by hand. Dry runs read none of it and write nothing. A sidecar with no record beside it
(a campaign begun before the record existed) is unprotected: its lines are attributed to
whichever endpoint runs next, so backfill the record by hand first.

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
