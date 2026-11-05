# Architecture and contracts (full reference)

The full design notes for the pipeline, moved verbatim from `CLAUDE.md`, which now keeps a
condensed digest of the rules with pointers here. When a detail below changes, edit it here
and keep the one-line rule in `CLAUDE.md` in step. "The command block above" and similar
now mean [commands.md](commands.md).

## Architecture

The pipeline is two stages run by two separate entry points:

**Stage 1 — detection (`main.py`)**
1. Loads a GeoJSON file — a `Polygon`/`MultiPolygon`, bare or wrapped in a `Feature` or
   `FeatureCollection` (issue #7). `extract_geometry` normalizes it to the bare geometry,
   dissolving a multi-feature collection into one `MultiPolygon`; that bare geometry is what
   `area.geojson` stores and what the SHA-256 area hash covers, so a wrapped and an unwrapped
   copy of the same polygon bind to the same run (the manifest's `input_geojson_type` says
   which it came in as). Anything non-polygonal is refused.
2. Converts the area bounds to Slippy Map tiles (zoom per source), keeps only the tiles
   within `TILE_EDGE_BUFFER_M` = 50 m of the polygon (`tiles_intersecting`, issue #4 —
   on the committed concave/multi-part areas 29–56% of candidate tiles are empty corners),
   and scans them concurrently through the imagery source (`--source`, see below) to
   collect all pano IDs whose point falls inside the area polygon. The buffer is not
   optional: **a coverage tile returns panos lying outside its own bounds** (a live GSV
   probe measured out-of-tile hits up to 28 m past the edge, and some panos returned only
   by the neighbouring tile), so a tile that merely touches the area would lose in-area
   panos near its edge. Don't shrink it without re-measuring. The pano list
   (pre-thinning) is saved to `scan.json`; `--reuse-scan` loads it instead of rescanning
   when area hash, source, source endpoint (Panoramax's `PANORAMAX_API_URL`), zoom and
   tile prefilter rule all match and no tile failed. It is **off by default** because coverage churns (paterson ~0.3% per
   4 h — the reason gap fill exists): a resume that silently reused an old scan would miss
   new panos without saying so. Each manifest run entry records `scan: fresh|reused` and
   `scan_age_hours`.
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
retried — indoor GSV panos, non-360 or non-2:1 Mapillary images, incomplete metadata,
image bytes that arrived but do not decode — only PIL's "these bytes are not an image"
errors count, so a `MemoryError` mid-decode, a body shorter than its `Content-Length`, and
PIL's truncated-file `OSError` all stay retryable (#127; both rules shared by Mapillary and
Panoramax via `sources.mapillary.decode_failure_is_permanent` / `body_is_complete`)) from retryable `failure` (left
uncached — network/HTTP errors, and a 200 whose Content-Type is not `image/*`). An image **404
is a retryable `failure` in both sources**, for different reasons: Mapillary's
`thumb_original_url` is signed and expires, so a 404 there is a lapsed signature (retried
within the call, then left uncached); Panoramax's `hd` URL is plain, but its infrastructure is
young and flaky, so a 404 is NOT read as the pixels being gone (Jon, 2026-10-05, reversing
#137). Panoramax also treats a **catalog** (STAC item) 404 the same way — Mapillary's Graph
API 400/404 is still a cached skip. A Panoramax 404 costs one request per pano (not retried
within the call), so a picture that really is gone is re-requested on every resume, forever:
cheap, and deliberate. Each has its own reason (`Image asset HTTP 404`, `Picture not found on
Panoramax (HTTP 404; ...)`) so a run log can count them.
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
  `computed_compass_angle` becomes `camera_heading`; `camera_pitch`/`camera_roll` are parsed
  from `computed_rotation` by `geo.mapillary_pitch_roll` (gravity-relative, **PS's roll
  sign**, null past a 45° tilt = failed reconstruction; issue #42). `copyright` is
  the contributor's bare username (see the attribution note below); the constant CC BY-SA
  4.0 licence rides in the record's own `license` field.
- **panoramax**: the federated open imagery commons (IGN + OSM France; CC BY-SA / Etalab),
  no token. z15 vector tiles from the federation catalog (`pictures` layer carries `type`,
  so 360-filtering happens during enumeration), then one STAC item request per picture
  (`/api/pictures/{id}`) for position, `view:azimuth` → `camera_heading`, pitch/roll,
  camera make/model, producer + license, and the `hd` asset — the original upload on the
  picture's home instance, unsigned. Same thinning as Mapillary (newest per cell,
  pixel-density tiebreak). `PANORAMAX_API_URL` targets one instance instead of the
  federation, and `prepare()` probes that root's STAC landing page so a mistyped one fails
  fast (the tile endpoint answers 204 for an empty tile and 404 for a bad path, so a wrong
  root would otherwise read as a legitimate zero-coverage scan). Records carry `license` and `panoramax_instance`, and `copyright` is the
  producer's bare name (see the attribution note below); `source_metadata` is the
  STAC properties (EXIF included) minus the viewer's tile descriptors.
  **Politeness (issue #161)** -- Panoramax's maintainers asked for fewer full-resolution
  downloads and about 1/s. Every request goes through `panoramax._get(url, kind)` and a
  per-kind token bucket (`sources/ratelimit.py`): `picture` 1/s, `metadata` 1/s, `tile` 5/s
  (env `PANORAMAX_PICTURES_PER_SECOND` / `_METADATA_` / `_TILES_`, 0 = unlimited for a
  private mirror; recorded per run entry as `panoramax_rate_limits` with per-kind request
  counts). A 429/503 pauses the whole kind for min(Retry-After, 300) s. **Limits are per
  process**: two runs (or a run plus an archive) against the same instance get 1/s EACH.
  Metadata is batched: main.py calls the optional `queue_hint` hook with the processing
  order, and a miss fetches that picture plus the next <= 99 in one `/search?ids=` request
  (20 of 20 live items built identical pano blocks to `/pictures/{id}`'s, 2026-10-28,
  `runs/_summary/panoramax_asset/search_check.csv`); an id the search leaves out falls back
  to the per-id GET, never read as gone. `MAX_PROCESSING_CONCURRENCY = 4` caps the pool.
  **The arithmetic:** at 1 picture/s a Lyon-size run (174k panos) is ~48 h of downloads and
  Bayonne (50k) ~14 h; `--scan-only` prints that download floor beside its usual 1.5 s/pano
  pipeline estimate (Lyon ~73 h), and the run is bounded by the larger. `--panoramax-asset
  {hd,sd}` picks the downloaded asset. The default stays `hd` by
  docs/panoramax-asset-study.md's pre-registered rule: on 600 Lyon panos SD kept only 0.629
  of HD's 0.30 detections (precision 0.847 vs HD; bar 0.95 each; every rig failed), mostly
  because SD lowers confidence (lenient recall 0.881), so don't flip it without a model
  trained at SD resolution; under `sd` the record keeps the SENSOR frame
  (`sensor_array_dimensions`, which PS's viewer and pano store use) and a picture without
  2:1 sensor dims, without an `sd` asset, or whose SD 404s / is undecodable / is not 2:1 is
  read in HD instead (`asset_fallbacks` in the run entry); `source_metadata` gains
  `image_asset` / `image_url` / `image_dimensions_downloaded` under `sd` only, so `hd` lines
  are unchanged. The manifest binds `panoramax_asset` (`main.bind_source_settings`, via the
  source's `manifest_bindings()`; a pre-#161 run with records is `hd`): a resume under the
  other asset is refused. **The native-res archive re-downloads every pano** --
  `export_benchmark.py` always fetches HD (`asset='hd'`), now at the same 1/s, so archiving
  Lyon is another ~48 h and another 174k full-resolution pictures; coordinate with the
  Panoramax maintainers before any run or archive that size.

Concurrency uses plain OS threads (`concurrent.futures.ThreadPoolExecutor`) — **not gevent**.
streetlevel's sync imagery API runs an internal asyncio event loop per call (`asyncio.run` +
aiohttp), which requires one real thread per concurrent call; under gevent monkey-patching
those loops collide and every download stalls for minutes. Two pools
(`COVERAGE_API_CONCURRENCY=100` for tile scanning, `PROCESSING_CONCURRENCY=50` for per-pano
work) are the main tuning knobs. GPU inference is serialized by a lock inside
`CurbRampDetector` — download threads overlap, forward passes don't (VRAM limit).
Preprocessing (resize/normalize) runs in the calling thread, outside the lock. `--batch-size N`
(issue #2; `main.py`, `scripts/reinfer.py`, `scripts/detect_from_store.py`, default 1 = the
unbatched path) instead hands each thread's tensor to one consumer thread
(`detectors/batching.py`, torch-free) that runs a single forward over up to N images, waiting
at most 0.1 s for a batch to fill; a failed batched forward (e.g. CUDA OOM) fails only that
batch's panos, as retryable failures. Each queued image is a ~100 MB 3x2048x4096 float32 tensor
on top of the decoded panos the pool already holds. Every pass ends with a `detector:` line
(forward passes, mean batch fill, seconds in forward, panos/s) — measure the gain there; it is
for the fetch-cheap re-inference paths, since `main.py` is network-bound. Batched and unbatched
heatmaps agree (the opt-in `RAMPNET_EQUIVALENCE=1|full pytest
tests/test_curb_ramp_batching_equivalence.py -s` check, CPU, needs the cached weights and
../RampNet bundle panos). **Measured on makelab2's A40 (2026-09-30, issue #2): batching buys
nothing, so the default stays 1.** Same 300 Vancouver store panos, `detect_from_store.py
--workers 16`, GPU otherwise idle, two passes per size: batch 1 / 4 / 8 ran at 1.273-1.279 /
1.275-1.279 / 1.270-1.279 panos/s, with forward time 99% of wall at every size (~0.78 s per
image) -- one 4096x2048 forward already holds the GPU at ~100% utilization, so there is no
idle time for a batch to fill. Peak VRAM was 5.8 / 22.9 / 40.1-44.6 GB: batch 4 OOMs beside
~28 GB of other jobs, and batch 8 leaves no room for anything. Host RSS ~23 GB at every size
(the 16 workers decoding 13-16k store JPEGs). Batch 1 reproduced the deployed Vancouver run
exactly; batched passes matched it at every detection >= 0.30, with confidence drift <= 6e-5
and 2 of 300 panos differing only below 0.14 (a peak crossing the 0.1 storage floor, a
plateau peak moving one heatmap row). Revisit only for a smaller input or a lighter model.
Figure: `docs/figures/batch-size/batch_size_a40.png` (redraw with its `make_figure.py`).

**Run directories / resumability:** all per-area state lives in `runs/<name>/` —
`results.jsonl`, the resume cache (`already_processed.txt`), `manifest.json` (geometry hash,
model provenance, streetlevel version, per-run stats), `area.geojson` (exact copy of the
geometry used), and `scan.json` (the last coverage scan, gitignored; see step 2). The JSONL and cache are appended to and flushed line-by-line, so a run is
resumable — re-running skips cached panos, and failed panos are intentionally left out of the
cache so they retry next run. A run directory is bound to one geometry and one imagery
source: rerunning a name with an edited geojson or a different `--source` is refused
(checked against the manifest) instead of silently forking state; and, for sources that thin
(Mapillary, Panoramax), to one `--thin-spacing` (`thin_spacing_m`, #126; `0` = disabled,
`null` = no thinning hook); a manifest from before the key is bound on its first resume
with a one-time note, since the value that made its records is not knowable from the code.
The manifest also records
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
(loaded with `trust_remote_code=True`). It outputs a heatmap; `detectors/decode.py`'s
`peak_local_max` extracts peaks (placed at the argmax pixel by default, or by the opt-in
`gaussian` sub-cell decode, #111) down to the **storage floor** (`DETECTION_STORAGE_FLOOR=0.1`, top-50 per pano), NOT the
decision threshold. The two-threshold contract lives in `detectors/__init__.py` (torch-free,
importable everywhere): results.jsonl deliberately stores sub-threshold candidates as raw
material for multi-view fusion (#27), and everything that *acts* on detections filters at
`OPERATIONAL_CONFIDENCE=0.30` (issue #20, adopted 2026-09-21 from RampNet's threshold sweep:
+7.4 recall for -4.5 precision, recall-first because a false positive is cheap to validate
and a false negative is never seen again) — `send_to_ps.py --min-confidence`, `fuse_sites.py`'s
operational tier (which deliberately also associates the sub-floor band, flagged
`in_refit: false` — the one sanctioned sub-floor consumer), `position_check.py`. The
**benchmark contract is a separate constant**, `BENCHMARK_CONFIDENCE=0.55`: every judged
RampNet bundle was exported and reviewed at 0.55, so anything that joins a run to a bundle
(`eval_sites.py`/`mined_precision.py`'s drift gate, `export_benchmark.py`'s strata,
`mapillary_tilt.py`, `thinning_experiment.py`) filters there, never at the policy value —
moving the operating point must not silently re-key nine cities of ground truth. Detections
are returned as **normalized** `(x, y, confidence)` tuples in `[0, 1]`.

A third filter is **geometric, not confidence-based**: `NADIR_MASK_DEG = 49.0` and
`on_camera_rig(y_normalized)` in `detectors/__init__.py`. In an equirectangular pano the
vertical axis *is* the dip angle (`y_normalized` 0.5 = horizon, 1.0 = straight down, the
"nadir"), and straight down is the vehicle the camera is bolted to — so a steep enough
detection is on the rig, never on the street. Found 2026-09-22 from human validations of
live Laurens labels: **158 labels on 156 panos across 19 sequences at seven discrete `y`
values, every one GoPro Max** — a roof rack, fixed in the rig's frame, re-detected pano after
pano. Of those, **50 have been judged and all 50 are false** (precision 0.000, CI to 0.071).
No label a validator marked correct sits below 46.1° of dip and no false one above 40° sits
shallower than 51.7°, a gap that has held as the judged set grew 379 → 536; 49° splits it.
It is expressed as an
angle, not a range, because range needs a camera height that is per-pano and known to be too
high (#40), while the dip is read straight off the pixel. The mask was invisible at 0.55
(0 of 708 Laurens labels, 2 of 9,526 Richmond) — **dropping to 0.30 is what surfaced it**;
it costs Richmond's band 12 of 3,448. Applied in `send_to_ps.transform_record` (so nothing
rig-borne ever ships) and in `fuse_sites`' projection (`FuseParams.mask_rig`, counted as
`drops['on_rig']`); `mask_rig=False` / `transform_record(..., mask_rig=False)` reconstructs
a campaign that shipped *before* the mask, which is why `reinfer.py`'s derived record and the
tilt/clustering analyses pin it off — what is already live is already live.
**Deliberately NOT implemented:** the same validations show these false positives also sit
far from intersections (median 55 m vs 10 m for true ramps), but that is a *consequence* of
the rig artifact — it lands wherever the car drove — not a second signal, and legitimate
mid-block ramps exist (school and mid-block crossings, driveway cuts). Measured: after the
nadir mask a ">30 m from an intersection" rule would cut 3 validated-true labels for every 1
false.

The trap that catches: **`FuseParams.min_confidence` defaults to `OPERATIONAL_CONFIDENCE`**,
so every bare `fs.FuseParams()` silently followed the operating point down to 0.30. Any
analysis scored against a bundle must therefore pass the tier explicitly —
`mapillary_tilt.py`'s `ablation` and `eval` pin `BENCHMARK_CONFIDENCE`, and
`eval_ps_clustering.py` takes `--min-confidence`, defaulting to the benchmark tier because
what it scores is *what the server holds* (set it to whatever the city's submission record
says it went live at). `site_explorer.py` deliberately keeps the default: it views the sites
production ships, with the caveat that a site built only from 0.30–0.55 members was never
adjudicated by the GT session its card overlays. Checked 2026-09-22: re-running `ablation`
and `eval` pinned reproduces PR #50's committed CSVs cell-for-cell, and the Richmond
clustering report is unaffected either way — of the cities involved only `laurens` was made
after the storage floor, so everywhere else the stored file has no sub-0.55 band for the tier
to select.

A city that went live at the old threshold gets the new labels as a **band**:
`send_to_ps.py <file> --min-confidence 0.30 --max-confidence 0.55` ships exactly
`0.30 <= c < 0.55`, only on top of a campaign the record shows complete at 0.55 on the
unchanged file, from its own sidecar (`<file>.band-0.3-0.55.submitted`), skipping records
with nothing in the band (the server already holds them as checked); the record gains
`bands` per endpoint and the endpoint's `min_confidence` drops once the band covers the
file. The guard refuses a band that could insert a label twice, and every refusal below is
a real sequence, not a hypothetical: a band over a file holding **nothing** in `[min, max)`
(a pre-storage-floor `results.jsonl` is exactly that) would POST nothing, mark every line
done, and then record the endpoint as holding a tier it was never sent; a band **re-run
after a gap-fill** (#32) appended panos would re-send the band labels of every line its
sidecar predates, so it is sent to the base campaign instead; a **lost sidecar on a band the
record shows complete** stops with "nothing to do" rather than refusing, because refusing
pushes you to `--ignore-submission-guard`, which with an empty sidecar re-POSTs the lot.
A band never runs *upward*, so a run above the recorded tier is told so rather than handed
an impossible `--min-confidence 0.55 --max-confidence 0.3`.

Runs from before the storage floor hold no band at all, and re-inference is not a substitute
for one: `scripts/reinfer.py runs/<name>` re-runs the run's own panos by id into
`results.f01.jsonl`, but those are **fresh forward passes, so the confidences are new
numbers** — on Richmond one detection sat at 0.550011 in July (shipped, live) and re-infers
at 0.549993, which a naive band would insert a second time. So `--verify` compares each pano
against the old file **at the tier the submission record says the server holds** (not the
benchmark constant) on the pixel key PS stores, *and* requires the pano block's width,
height, lat, lng and heading to be unchanged — a band whose panos moved would arrive in a
different frame from the labels already live. `--write-band-file` then builds the file a band
may actually ship from (`results.band.jsonl`): the new record where the pano reproduced, the
**old** record where it did not, so those panos have an empty band and are never POSTed. It
asserts, against what it wrote, that the file's labels at the server's tier are exactly the
live ones, and derives `results.band.jsonl.submission.json` from the old campaign — which is
what lets the ordinary band guard pass with **no override**. Ship from `results.band.jsonl`,
never from `results.f01.jsonl`.

`--verify` also prints a **coarse-cell diagnostic** (#111): for every pano whose pixel keys
differ, how old and new keys pair one-to-one within +/-1 coarse heatmap cell (same cell, 3<->4
neighbour, off-grid, and the 7-8-cell **flip** of a near-tied pair). It explains a mismatch; it
never excuses one. Band eligibility stays EXACT pixel keys on purpose: a label that moved a
cell is a different pixel on the server, so a pano that agrees only within a cell is carried
over from the old file like any other, and the band file's labels at the tier stay exactly the
live ones.

**Multi-view fusion (`geo.py`, `scripts/fuse_sites.py`, `scripts/eval_sites.py`)** —
issue #27 stages 2–3, a post-processing layer between detection and submission.
`geo.py` (repo root, stdlib-only, torch/numpy-free like `detectors/__init__.py`) is the
single home for geodesy: haversine + the declustering grid (imported back by
`export_benchmark.py`), a `LocalFrame` ENU tangent plane, and the ground raycast
`detection_ground_point` — flat-ground intersection at a camera height (2.6 m unless the
caller asks otherwise; the fuse_sites CLI's `auto` default is per-rig for GSV, #79) with
closed-form anisotropic error (`geo.ErrorModel`), **dropping** (never clamping) rays beyond
25 m. Its peak term `sigma_peak_px` defaults to **1.0 heatmap px**, which is optimistic: the
heatmap is an 8x bilinear upsample of a stride-32 map, so detections are quantized to 8-px
coarse cells (#111; `scripts/heatmap_grid.py grid`: >= 99.8% on residue 3/4 in every run), whose
uniform quantization alone is 8/sqrt(12) = 2.31 px. 2.31 failed #111's pre-registered adoption
rule in one of ten cells (Sao Paulo at 2.6 m: world recall -1.6 pts, SE 1.3; precision
unchanged everywhere; gate/residual rejections down 16-58% on GSV), so the default stayed 1.0
and `--sigma-peak-px` (fuse_sites, eval_sites; `FuseParams.sigma_peak_px`) opts in. **Every
published fusion/clustering/residual table used 1.0.** The MEASURED residual (#111 decode half,
residual to box centres, bias removed, an upper bound): 4.24 px argmax / 3.79 px gaussian on
manual_gold; at 4.24 (and 3.67) the same ten cells fail in Sao Paulo at both heights, so the
default still stays 1.0; docs/heatmap-grid.md section 4. GSV camera pitch/roll are deliberately NOT applied: the
`--pose-ablation` experiment measured that applying the full pose loosens multi-view
agreement. **That does not mean the equirects are gravity-rectified** (corrected 2026-09-29,
#113): they are in the rig's frame (sidewalk-panorama-tools#158), streetlevel and the PS pano
store serve the same pixels, and the full pose overshoots because the car rides the road, so
the local ground shares most of the tilt. What leaks into placement is a fraction of it
(roughly 0.15-0.25 of the pitch term, 0.4-0.55 of the roll term, scratch measurements on
#113). #116's pre-registered test (`scripts/gsv_partial_pose.py`,
docs/gsv-partial-pose-study.md) confirmed that fraction held out (pooled 0.18 pitch / 0.38 roll
at `auto`): it tightens re-associated sites 5-11% on the median, beats a magnitude-matched
shuffled pose and the inventories agree -- but it FAILED the recall clause (Bend, 2 of 157
ramps on the half split), so the DEFAULT stays flat; the fraction is opt-in as
`--apply-pose partial` (geo.PARTIAL_POSE_K_GSV, frozen) pending the pre-registered
confirmatory run (`gsv_partial_pose.py confirm`). Two
consequences outside fusion: a detection's `pano_y` is in the image's frame while a human PS
label's is off by about 0.9 of the tilt at its bearing (measured on 1,704 Gainesville
crowd/detection pairs, #113), so an AI-vs-human pixel comparison mixes two frames. At
Gainesville's tilts that moves `agree_rate.py`'s pano-frame rate by only 0.2-0.3 points (its
radius is 7.9 deg), but it matters wherever the tolerance is tight or the tilt large. And
streetlevel's pitch > 0 is nose DOWN, with GSV
`camera_roll` stored unwrapped (359.4 = -0.6). Details in `geo._world_ray`'s docstring. Mapillary's are available
(`--apply-pose road`) but NOT applied by default either: the #42 shuffled-grade control withheld
it (see the rig-tilt paragraph below).
`fuse_sites.py` associates a
run's stored detections into physical-ramp sites (descending-confidence greedy with a
same-pano cannot-link, chi-square gating, inverse-covariance refit, residual rejection;
sub-threshold detections attach as `in_refit: false` support and never move operational
positions) and writes `runs/<name>/sites.jsonl` — cluster plus members, so the
what-to-submit decision stays late; it reads no manifest and leaves `send_to_ps.py`
untouched. `eval_sites.py` scores fusion against RampNet's benchmark verdicts in world
space (semantics mirror `rampnet.validation.collect`) and produces the stage-4 promotion
calibration. Measured 2026-08-02 (5 m match radius): world recall 0.93–0.96 vs own-view
0.72–0.83, precision 0.89–0.98 across paterson/gainesville/sao_paulo/richmond/bend.
`scripts/reprojection_residual.py` (issue #36) is the GT-free companion: a
leave-one-view-out residual over every multi-view site, and a range-scale fit that turns
the residual into an implied range scale per capture year (the instrument that replicates
the camera-height study's rig ranking without depth or triangulation pairs). Its blind
spot is structural: a bias every view of a site shares (a common offset, a scale error on
views that all look the same way) moves the held-out position with it and is invisible;
its residuals are also truncated by association's own gate. Its `k` is an EFFECTIVE range
scale: a constant vertical peak offset reads as range error growing ~r^2 and the fit
absorbs it (the `r2fit_*` columns show k moving 0.05-0.13 when an offset term is allowed),
and it is measured at the association height, so it is pulled toward 1 (the new rig's
1.18-1.19 is a lower bound; the camera-height fixed point implies ~1.31-1.35). It also
runs RampNet#101's own estimator (full-site residual, site fixed effects, >= 4 m span;
reproduces #101's table exactly on the on-disk sites.jsonl) with its error-model null
(37-76% of the slope on GSV; the null itself is miscalibrated, see the doc). The
GT-anchored half projects sites into judged panos against reviewer box centres: quote its
PIXELS (the pixel residual never raycasts the reference); its metres are truncated for
missed marks (5 m match gate), so metres are quoted only for the `box_on_detection`
subset. All of it is benchmark tier 0.55, not the 0.30 production ships. Numbers in
docs/reprojection-residual.md.

**Mapillary rig tilt (`scripts/mapillary_tilt.py`)** — issue #42. Mapillary blocks store
`camera_pitch`/`camera_roll` as null, but `source_metadata.computed_rotation` is OpenSfM's full
world→camera rotation (axis-angle; world = east/north/up, camera = right/down/forward). Its yaw equals
Mapillary's `computed_compass_angle` to 1e-11° on every record, PS's own `MapillaryViewer.extractPitchRoll`
decomposes it the same way (pitch identical; **PS's roll sign is the negative of `geo._world_ray`'s**), and
re-rendering panos with it levels them. Median tilt is ~3° (81–85% above the 1.5° sigma the error model
assumes). Measured 2026-09-21 (every claim below names the statistic it is true of — they disagree, and
that is the finding): applying the tilt relative to *gravity* tightens the MEDIAN within-site multi-view
spread where the rig itself is tilted (Clovis −39.7%, Laurens −28.5%, Richmond −14.0%) but loosens it where
the camera rides level on a car in hilly terrain (Morgantown +23.3%: camera pitch tracks the road grade with
slope 0.97). The raycast wants tilt relative to the local road, which the sequence's SfM altitude profile
provides — and the `road-relative` convention tightens the MEDIAN in all five cities (−1.8% Annapolis,
−2.3% Morgantown, −19.2% Richmond, −28.4% Laurens, −38.1% Clovis) and the range-normalized mean in all five.
**On the RAW MEAN and p90 the same correction is worse than doing nothing** — Richmond +3.9%/+6.1%,
Morgantown +5.6%/+6.4%, Annapolis +11.3%/+17.5% — partly because the ablation runs uncapped and charges a
correction for rays production drops, partly because it really does move a minority of detections a long
way. So the wiring was gated on a **pre-registered precondition** (study §8 rec 3, rule posted on #42
before it ran): at the production 25 m cap, on ONE site set (association frozen from the flat fuse; a site
is scored only if every arm places every operational member) and ONE GT set, road-relative had to be no
worse than flat on p90 GT-to-site distance in any city (0.1 m tolerance) and better on the median in 3 of
5. **It passed the first rule everywhere** (2026-09-23, study §10, `docs/figures/mapillary-tilt/data/pose_precondition_3arm.csv`):
p90 off→road Richmond 3.42→2.74, Clovis 3.32→2.37, Morgantown 3.07→2.79, Annapolis 3.55→2.75, Laurens
3.82→2.74 m; median better in all five. The uncapped ablation's bad tail was the rays production drops.
**Then it failed the control** (second pre-registered rule, after #52 showed on GSV that a shuffled
ground normal also "improves" p90 and that p90 can improve by survivorship; study §10.5): road had to beat
`road-shuffled-within` (each frame given another frame's grade from its own sequence) by >0.1 m on p90 AND
median in 4 of 5 cities, lose ≤1.0 pt of recall@2.5 m on the OFF pool, and add ≤5% of the pool in
unplaceable GT marks. It beat the shuffle in only 3 (Clovis and Laurens: shuffled median is *better*),
and off-pool recall fell 4.0 pts (Richmond) and 2.9 pts (Annapolis). So **the road-frame default is
withheld**: `auto` resolves to off for Mapillary. Open reading (#52): pitch and grade come from the same
SfM, so subtracting may cancel shared SfM error, not road slope — a DEM grade (#51) is the independent
referee.
**What shipped (#42):** `sources/mapillary.py` writes pitch/roll (PS sign, 45° cap); **`geo._world_ray`
now takes PS's roll sign** (positive lowers the camera's right axis — the opposite of before #42, so any
roll quoted from earlier has the other sign); `fuse_sites` has `--apply-pose road` (default still flat); and
`backfill_metadata.py --pose` fills old files. The in-place backfill of submitted files and the pano-only
push to PS are deliberately manual (see the backfill command block).

**GSV depth (`depth.py`, `scripts/harvest_depth.py`)** — issues #40/#41. `depth.py` (repo
root, stdlib-only like `geo.py`) parses GSV's depth payload, which is **not a raster**: it
is a list of `{normal, distance}` planes plus one plane index per pixel, and streetlevel
computes a raster from it and then discards the planes. That matters because the dominant
ground plane's distance is a **per-pano camera height** and its normal is the ground tilt.
**But it is the depth frame's height, not the imagery's** (measured 2026-09-23,
`docs/camera-height-study.md`): bearing-only triangulation of multi-view sites implies
heights **6–16% above** it, city by city, so #40's original "ranges run 29–35% long at
2.6 m" was computed against a depth map sharing that bias and does not hold city-wide.
The rig ranking is real, though: the 2025–26 GSV rig triangulates to ~1.9–2.0 m against
~2.5 m for every earlier vintage, so on *that* rig 2.6 m does run ranges 31–35% long
(~2–4% on older ones). GT world P/R still cannot tell any height model apart
(all within ±3 pts). So `geo.PER_PANO` / `--camera-height-m per-pano` exists and is
**opt-in**. What finally moved the default was an EXTERNAL oracle, not GT: city curb-ramp
inventories (#79, `docs/placement-oracle.md`) select a per-rig constant by capture year
(2.0 m for a well-measured low-rig year, else 2.5 m), now the fuse_sites CLI's `auto`
default for GSV; analysis scripts and `FuseParams()` still default to 2.6 m.
Under per-pano a measured height passes `depth.believe_height` (#44, pre-registered tests
in `scripts/height_qc.py`), which today only FLAGS: a height ≥ 0.40 m from its vintage's
median (same run and capture year, ≥ 300 panos) is counted as `flagged_qc` in
`sites_meta.json` and still used. The gate failed per city and its 2.6 m fallback was worse
than the flagged height; T4's constant sigma worsened GT p90 placement. So both were not
adopted, and per-pano `sites.jsonl` was byte-identical to #68's until #47's follow-up
took stand-in planes out of the spread (the sigma), which moves it slightly.
The default-height decision itself (#79) is scored against the Bend and Gainesville city
curb-ramp inventories by `scripts/inventory_oracle.py`, under a pre-registered rule; see
`docs/placement-oracle.md`.
`sources/gsv.py` stores the height on every new GSV pano block (`camera_height_m`,
`camera_height_spread_m` (stand-in planes excluded since #47; older blocks include them),
`ground_tilt_deg`, `depth_planes`, `camera_height_status`) —
read from the raw response, because streetlevel's own depth parser rasterizes 131k pixels
in pure Python and throws on the uint8-offset bug below; the height is non-null only when
the status is `measured`. **Google's stand-in ground** is common and the plane-count test
misses it: 14% of harvested payloads (16% of bend's) are full reconstructions whose ground is exactly 2.500 m
with an exactly vertical normal (`SYNTHETIC_GROUND`, detected on the normal, not the
value); those panos triangulate to ≥2.6 m, which is why unmeasured panos fall back to
2.6 m. Four traps live in `depth.py` rather than at call sites: the header's `offset` field
is a **uint8** at byte 7 (reading it as a uint16 swallows the first plane index and makes
~0.3–0.5% of panos unparseable); the raster is **mirrored** relative to the raw index array
(`_raw_column`); Google returns a degenerate 2-plane fallback at exactly 2.500 m that must
be filtered structurally (`DEGENERATE_MAX_PLANES`), not by testing the value; and **a range
query must not snap to a pixel**. On that last one: `depth_at` snaps, because it has to
reproduce streetlevel's raster, but a detection lands at an arbitrary coordinate, and
snapping it to one of 256 rows costs up to **6.6% of the horizontal range (±1.2 m at
20–25 m)** once the near-horizon `1/(sin·cos)` amplification is applied — an alternating
sign that reads as noise. So `ray_depth_at`/`ground_range_at` snap only the *plane lookup*
(the segmentation genuinely is per-pixel) and intersect the true ray. Watch the azimuth
there: range queries take the IMAGE frame (raw index column == JPEG column, `phi = (1-x)2pi +
pi/2`), while `depth_at` alone takes streetlevel's raster, which is the image's MIRROR (#80: the
range queries read `(1-x, y)` until then); because mirroring phi flips only the ray's x
component, any plane with `nx == 0` — every level ground plane — cannot tell a correct
convention from a backwards one.
`harvest_depth.py` archives the payloads before they go away: the *JavaScript* API that
exposed depth was withdrawn in 2020 and anonymous tile access in ~2026, but the metadata
endpoint used here still serves it. `no_depth.txt`/`gone.txt` are append-only skip caches
(see the command block above). GSV only; Mapillary serves no depth (since #42 its tilt is
parsed from computed_rotation and written into the record, and fusion can apply it with
`--apply-pose road` or `gravity`, but it is OFF by default -- the shuffled-grade control
withheld the road-relative default; its camera HEIGHT is still the 2.6 m constant — #53).

**GSV ground plane (`scripts/gsv_ground_plane.py`)** — issue #52, a study. The same payloads'
dominant ground plane has a **normal**, and in depth.py's frame -x is camera-right (#80), **-y is
camera-forward** (x = 0.5) and +z is down; `camera_frame_normal`/`slopes` turn it into grade
(rise ahead) and cross-slope (rise to the right), and the tests pin that mapping against
`depth.ground_range_at`. Measured 2026-09-23 on the four harvested runs: **the normal is not a
per-pano road-grade measurement.** Its slope along a street does not persist between linked panos
(r <= 0.31 within one drive, ~0 across capture months; per-pano noise bound 1.0-2.4 deg), its
grade follows the rig's metadata pitch only on climbs, and the 2025-26 rig reads ~2x the median
|grade| of earlier rigs on the same streets; its cross-slope tracks the rig roll, but the study
ran in the mirrored frame (#80): every cross-sign in it is inverted, `cross-flipped` is the correctly
signed arm (verdict unchanged, see the dated correction in the study), and in the corrected frame the
plane falls LEFT on 56-67% of panos, against the crown prior -- unexplained, open. **Rotating GSV rays into the observed plane loosens
multi-view agreement in every city** (median 1.13-1.25x, 2.2-2.9x in the 4+ deg bucket, uncapped), and
so does `travel-only` -- the grade-only arm that mirrors #50's correction (1.04-1.12x) -- so keep GSV at
`apply_pose=False` with no ground-plane term. Two traps the review caught: the rig-attitude arm is one
sign pattern of #27's pose ablation, not an independent estimate; and pair buckets are cut on the arm's
own |grade|, so a steep-bucket comparison needs a MAGNITUDE-MATCHED control (shuffle within buckets) --
against one, the observed plane is as good as or slightly better than random, and both lose to flat.
Scope of the claim: the GSV estimates (0.7-2.4 deg per-pano noise, >= the median grade) cannot improve
the flat raycast; #50's road-relative *mechanism* on un-rectified Mapillary is untested, not refuted
(the Mapillary median result itself stands); the doc's section 8 asks the wiring PR for a
magnitude-matched within-sequence shuffled-grade control. The ray injection trap: expressing a ray in a sloped
plane's frame via `geo.detection_ground_point(apply_pose=True)` also needs the ray origin moved to
the foot of the perpendicular (`ground_frame_fields`), or every point of a sloped pano shifts
downhill by h*sin(slope).

**Position check (`position_check.py`, repo root; `scripts/position_check.py` is a shim)** —
SidewalkWebpage#5361. Stdlib-only like `geo.py`. Every pano's submitted position is scored
against OpenStreetMap street centerlines (one Overpass query, cached beside the run in
`osm_streets.json`; a partial answer — HTTP 200 + `remark` — is refused and never cached).
The metric is real but coarse: a camera is legitimately metres from a centerline, so it
floors near 1.75 m (Laurens: GSV 1.75 m vs Mapillary SfM 3.73 m / raw 2.59 m against the
same streets), and it gates only gross block drift (issue #62, rescoped 2026-09-22). For
Mapillary runs both positions are scored and each sequence gets a verdict on the field it
submitted: **off the street** when that field's median unsigned cross-track exceeds
`GROSS_OFF_STREET_M` = 5 m (or most of the sequence is beyond 30 m of any street);
**flagged** when it is off AND the other field is closer by > `RESOLUTION_FLOOR_M` = 2 m on
the **paired metric** — median over panos where both fields snap to the same street of
`cross_sfm − cross_raw`, where lane offset and OSM error cancel — without being more than
`MAX_IQR_RATIO` = 1.5× as scattered; **both_off** when off but not fixable (reported, never
gated); **undecidable** when the fields differ by ≤ 2 m (the reference cannot see it,
whichever sign it has). The #60 signed bias stays in each row as a report only: a lane
offset cancels in it, which is how Richmond's `jKtaJMek7wQl5AOH28qdcm` was flagged toward a
raw field 3.3 m *worse* per pano. The 5 m bar was chosen from a 4/5/6 m table over every
Mapillary run (in the #62 PR); the check records its verdict `rule` and its knobs
(`threshold_m`, `resolution_floor_m`, `max_iqr_ratio`, `min_sequence`), and a check from
another rule — or with any knob off its constant, e.g. `--threshold 100`, which would
otherwise be a zero-flag verdict under the right rule — is stale to the gate
(`position_check.rule_mismatches`) and re-run by main.py. Under `beyond_snap` the other
field is only a fix if it is itself on the street (median cross-track ≤ 5 m); otherwise the
verdict is both_off. The submitted field is voted per
sequence from the coordinates, so a mixed reposition output is judged correctly. **Frame
consistency:** repositioning a city that already carries live labels is a whole-city
decision, never a per-file one — PS computes a label's lat/lng once, at insert, and a
resubmission never moves a stored label, so resending moved panos duplicates their live labels
unless those are soft-deleted in the database first (the Richmond 3-sequence fix, 2026-09-24,
did exactly that: 143 retired, 143 resubmitted at raw). Where a pano is live is decided by
**newest campaign wins, per pano** (`position_check.live_positions`): over every
`*.submission.json` in the directory — the file's own included, bands as campaigns of their
own, each over the lines its sidecar says it sent — the campaign with the latest
`last_submission_utc` that sent the pano holds its live position. Records cannot say a pano
was superseded, so without it Richmond's older `results.jsonl`/`results.band.jsonl` records
(SfM) and `results.posfix3seq.raw.jsonl` (raw) would refuse every file. `send_to_ps.py`'s
`check_live_positions` refuses a file with any pano elsewhere than its live position — there
is no exemption for the file's own campaign, so a later band from `results.band.jsonl` is
refused for the 71 panos it would put back at SfM — and `reposition.py` refuses an output
that would (and never overwrites a file that has its own record). A missing campaign file, a
partial campaign without its sidecar, or a same-timestamp disagreement also refuses.
`--reposition-live-city` overrides both, and the reason lands in the submission record; the check and report print the live campaigns beside the verdict. It is wired into both stages: `main.py` ends every run with
`position_check.run_check` (non-fatal; the verdict summary lands in `manifest.json` under
`position_check`, the full `position_check.json` + `position_report.html` beside
`results.jsonl` and are git-tracked), and `send_to_ps.py`'s `check_position_state` refuses a
Mapillary file whose check is missing, stale (`results_sha256` ≠ the file) or flagged —
`--ignore-position-check` overrides, dry runs are exempt. `scripts/reposition.py` rewrites
flagged sequences' pano lat/lng from the other field into a new file (a *submission
artifact*, never swapped into `results.jsonl`: the run-dir field binding cannot see inside
the file), which is confirmed with `--results` and submitted from where it is. What it
cannot catch: a bias both fields share, anything under the ~2 m floor, and anything on a
source with one position (GSV/Panoramax are scored for the record but never gated).

**Stage 2 — submission (`send_to_ps.py`)**
Reads the Stage-1 JSONL and POSTs each record to a Project Sidewalk endpoint
(`/ai/submitLabelsOnPano`). Its key job is a coordinate transform: it converts the normalized
`x_normalized`/`y_normalized` detections into **pixel** `pano_x`/`pano_y` using the pano
width/height stored in the record, renames `detections` → `labels`, and drops the original
`detections` key. It also maps the pano block onto the server's `PanoSubmission` reader
(`transform_pano`): `panorama_id` → `pano_id`, raw streetlevel source strings → the
`pano_source` enum (`gsv`/`mapillary`/`infra3d`) with the raw string kept as `source_detail`
(issue #23; filled from `source` for records that predate it), `target_gsv_panorama_id` → `target_pano_id`,
and guarantees `links`/`history` arrays — so legacy JSONL files stay submittable unchanged.
PS **stores `source_metadata` verbatim** (`pano_data.source_metadata`, jsonb) and 400s the
whole record when it is over 64 KB (`ExploreFormats.scala` `maxSourceMetadataBytes`), so a
GSV pano's blob is cut to an **allow-list** (`send_to_ps.PS_GSV_SOURCE_METADATA_KEYS`:
uploader, uploader_icon_url, upload_date, elevation, country_code) plus `source_detail` —
inside the blob, because PS does not read the top-level `source_detail`. Every key is
present, `null` for legacy records. Places, artworks, neighbors, street names, address and
building levels stay JSONL-only. Mapillary/Panoramax blobs go unchanged.
`check_source_metadata_size` refuses an over-cap record before its POST (not sidecar'd).

Resume state is a `<file>.submitted` sidecar of **line numbers**, which silently stops
describing the campaign if the JSONL is edited, if the sidecar is lost (deleted, or a run
moved to a second machine), or if the same file is pointed at a second server — the first
two re-POST records that are already live and duplicate a whole city's labels; the third
skips the staged lines on production so they never reach it. So each campaign also writes
`<file>.submission.json`: the JSONL's sha256 and byte length and, **per endpoint**,
line/label counts and timestamps — the one submission artifact small and stable enough to commit
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
The one changed hash that **resumes** instead is a **pure append** (#59): a gap-fill (#32)
adds panos to the end of an already-submitted `results.jsonl`, and every recorded line keeps
the number the sidecar holds for it. `append_check` proves that byte-for-byte — the record's
`total_bytes` prefix must still hash to the recorded `sha256`, end on a newline, and hold the
recorded `total_lines` — **and then proves the appended lines are new panos**: their
`panorama_id`s must be disjoint from the prefix's, since a gap-fill only fetches ids the run
never processed. New bytes are not new panos — a doubled file, or a re-run after the
gitignored `already_processed.txt` was lost, appends panos that are already live, and PS is
insert-only (SidewalkWebpage#5382), so the duplicate labels cannot be retired. When it does
pass, the run prints how many new lines it found and rewrites the record with the new hash
and length. Everything else still refuses: a shorter or same-size file, an edited prefix, a
prefix ending mid-line, a repeated `panorama_id`, or a record from before `total_bytes`
existed. That last one is migrated, not overridden — `send_to_ps.py <file> --prefix-digest
<total_lines>` prints the recorded prefix's sha256 and byte length, and if the hash matches,
adding `"total_bytes"` to the record by hand lets the normal guard do the rest (prefer that to
`--ignore-submission-guard`, which also silences the lost-sidecar and wrong-endpoint checks).
In practice that path is unlikely to fire: every record committed so far is Mapillary, and
gap-fill is GSV-only (`fetch_pano_by_id`).

**Several label types in one file (#168).** PS takes one `labelType` per submission, so a
file whose lines ran several detectors (`registry.record_types` > 1 anywhere in the file) is
sent as **one POST per (line, type)**: `transform_record(..., label_type=T)` keeps T's
detections, names T, carries `models[T]` as the provenance and applies T's spec (operational
threshold as the default, rig mask only if `DetectorSpec.mask_rig`); `split_record` yields the
pairs. `transform_record` without a type **raises** on a multi-type line. Progress is one
**per-type sidecar**, `<file>.type-<T>.submitted` (plain line numbers), never
`<file>.submitted`; a line with only some types sent is half-sent and resumes with only the
missing types. The record gains `endpoints[ep].types[T]` = {submitted_lines,
labels_submitted, min_confidence, rig_masked, timestamps}, the shape of a band entry; the
endpoint entry then counts lines ANY type was sent for (`submitted_lines`), sums the labels,
and records the lowest type's `min_confidence` plus `min_confidence_from`: `flag` when an
explicit `--min-confidence` set every type, `per_type_default` when each type used its own.
`check_resume_state` holds each type to the single-type checks against its `types[T]` entry,
and refuses a whole-line sidecar beside a multi-type file or a record whose shape does not
match the file. `position_check.campaigns_for` yields one campaign per type, so a half-sent
pano counts as live; `coverage_check` replays per type; `backfill_metadata`'s guard sees
`.type-*.submitted`; every `models[T]` must be dated. **Bands are refused** on a multi-type
file (`send_to_ps.py --max-confidence`, `reinfer.py --write-band-file`, and `reinfer.py
--verify`, which re-runs only CurbRamp): they exist for the legacy CurbRamp cities. A
single-type file whose lines carry `label_types` (a run of one non-CurbRamp type) is sent
through the typed transform too, so its payload matches `split_record`. A file whose lines
ran DIFFERENT type sets is handled (a line counts done once every type it ran is sent) but
is not produced today: `main.py` binds one type set per run directory. **A single-type
CurbRamp file is unchanged**: same payloads, same `<file>.submitted` bytes, same record
bytes, pinned against fixtures captured from main (`tests/test_multitype_submit.py`,
`tests/fixtures/multitype_submit/`, un-ignored in `.gitignore` by `!tests/fixtures/**`).

## Output format notes

- The detector emits normalized coordinates; Stage 1 stores them normalized in the JSONL.
  The normalized → pixel conversion happens only in `send_to_ps.py`. Keep these in sync if
  you change either side. The pixel itself is `detectors.ps_pixel` (x modulo width -- a
  `--border wrap` + gaussian decode can store x just below 1.0; y never wraps and stays below
  height), and every analysis that rebuilds PS's `(pano_id, pano_x, pano_y)` key to join
  server labels (reinfer, agree_rate, eval_ps_clustering, server_agree_check,
  provenance_gate) calls it rather than re-deriving `round(x * w)` (PR #152).
- `detections` in results.jsonl means "stored candidates", not "believed ramps": it includes
  peaks down to the storage floor. Any new consumer must filter at `OPERATIONAL_CONFIDENCE`
  (import it from `detectors`) unless it deliberately wants the sub-floor candidates.
- Each JSONL line carries model provenance (`model_id`, `model_training_date`, `api_version`,
  plus `model_repo` and the full 40-hex `model_revision`) and rich pano metadata (capture date, dimensions, camera heading/pitch/roll, source,
  historical panos, and links). Heading/pitch/roll are converted from radians to degrees on
  write.
- Mapillary `camera_pitch`/`camera_roll` are **derived**, not measured: decomposed from the
  SfM `computed_rotation` (gravity-relative, PS's roll sign), and `camera_pose_source`
  (`"mapillary_computed_rotation"`, null when the angles are null) says so on every record.
  PS has no column for it; it lives in the JSONL for any consumer that might otherwise treat
  the angles as sensor readings. The conversion is exact (matches PS's own viewer to 1e-15);
  the SfM tilt's absolute error is unmeasured.
- Indoor panoramas (sources `innerspace`, `cultural_institute`, `photos:legacy_innerspace`)
  are skipped.
- Every source's pano block carries the same provenance keys, from its own
  `provenance_fields()`: `camera_make`/`camera_model`/`camera_type` + `source_metadata`
  (issue #23 added GSV's). GSV has no make/model (`null`); its analogue is `source_detail`
  (the raw streetlevel source, which survives `transform_pano`'s enum coercion) and
  `uploader`. GSV's `source_metadata` is an **explicit per-field projection**
  (`sources/gsv.SOURCE_METADATA_FIELDS`: uploader, uploader_icon_url, upload_date,
  elevation, country_code, street_names, address, building_level(s), places, artworks,
  neighbor ids) — never `vars()`/`asdict()` of the streetlevel object, which holds numpy
  arrays and recursive panos. Every named key is always present (`null` when unset, `[]` for
  the list-defaulted `building_levels`/`neighbors`); a new streetlevel attribute is ignored
  until named there; values are coerced to JSON-native types and a converter that throws
  (or yields something `json.dumps` cannot write) records `null` rather than failing the
  pano. All of it stays in the JSONL; only the allow-list above reaches PS. GSV runs from
  before #23 have no `elevation` (#52).
- **Model provenance is resolved, never declared** (issues #39/#6). `CurbRampDetector` reads
  the Hugging Face revision SHA of the snapshot it loaded (`config._commit_hash`, else the hub
  cache's `snapshots/<sha>` dir; loading retries `local_files_only` when the hub is
  unreachable, e.g. Hyak compute nodes) and exposes `.provenance`; `main.py` and
  `scripts/reinfer.py` write that dict, and there are no provenance literals left in `main.py`.
  `model_id` is `rampnet-model@<12 hex>` (PS stores it as TEXT, so no width limit; the prefix
  keeps `rampnet-model` string matches working). The training date is a fact about a SHA:
  `detectors.KNOWN_REVISIONS`, seeded with every hub revision that carries weights (all the
  paper weights → 2025-08-21, emitted as PS's `MM-DD-YYYY`). **An unknown SHA refuses to
  start** — after a retrain, add the new SHA to the table (instructions beside it) rather than
  reaching for `--allow-unknown-model-revision`, which writes a null date that
  `send_to_ps.py` refuses (and a run made that way refuses to resume once its SHA gains a
  table row: its undated lines can't be repaired, so re-run under a fresh `--name`). A run directory is bound to one `model_revision` like it is to one
  geometry (a mismatch is refused); pre-#39 manifests resume with a one-time note and are
  bound on that resume, unless their recorded training date differs. `--scan-only` loads no
  model and binds nothing.
- `pano.copyright` is an **attribution ingredient, not a rendered attribution** (issue #61,
  SidewalkWebpage#5360). For Mapillary and Panoramax it is the contributor's *bare* name
  (creator username / `geovisio:producer`), `null` when the source names nobody — PS's
  `ImageryAttribution` composes the ©, the provider (from `source`) and the licence around
  it wherever it shows its own copy of the imagery, so wrapping them in here rendered a
  doubled credit on every crop. PS renders the licence from `license` for Panoramax, whose
  instances differ, and from `source` for Mapillary, which is uniformly CC BY-SA 4.0 — the
  Mapillary `license` key is kept as record provenance. GSV is the exception: streetlevel's
  `copyright_message` (`© 2025 Google`) is the provider's own string, stored and shown
  verbatim.

## Multiple label types (#168)

The plumbing for a second detector (crosswalk / NoCurbRamp, Planning#36). **No second
detector exists yet**; the only spec is CurbRamp.

- **Registry** (`detectors/registry.py`, torch-free): `DetectorSpec(label_type, module, cls,
  storage_floor, operational_confidence, benchmark_confidence, mask_rig)`, `SPECS`,
  `spec_for(t)`, `parse_types`, `load_detectors(types, **kw)` (lazy import per spec). The
  CurbRamp spec references the constants in `detectors/__init__.py`, which keep their names and
  values; a new type brings its own thresholds (a crosswalk heatmap's floor and operating point
  are not RampNet's). Wherever a consumer acts on a detection, it reads the thresholds and the
  rig mask from the spec of THAT detection's type.
- **Readers go through the helpers**, never the keys: `record_types(rec)` (the detectors that
  RAN on the pano; `label_types`, else `[label_type or 'CurbRamp']`), `detection_type(rec, d)`
  (the detection's own `label_type`, else the line's, else CurbRamp, and it RAISES on an
  untyped detection in a line that ran more than one type), `provenance_for(rec, t)`
  (`models[t]`, else the line-level `model_*`), `detections_of_type`. Every pre-#168 line reads
  as CurbRamp.
- **Writer rule** (`main.build_output_line`): a run whose type set is exactly `{CurbRamp}`
  writes the pre-#168 line with **no new key** (byte-pinned: `PRE_130_LINE_SHA256` holds
  through the registry path, `tests/test_multitype_records.py`). Any other set writes
  per-detection `label_type`; `label_types` (sorted: what ran, so an empty line still says what
  it was checked for); `models: {T: provenance}`; line-level `label_type` = the type iff one
  ran, else `null` (never derived from the detections); line-level `model_*` = CurbRamp's when
  CurbRamp ran, else `label_types[0]`'s. `detection_decode` / `detection_border` at line level
  describe the CurbRamp detector; any other type's ride in `models[T]`. `API_VERSION` stays
  `1.0.0`.
- **`main.py --types CurbRamp`** (comma list; an unknown type is a parser error listing
  SPECS). `main.detectors` is the loaded `{type: detector}`; `main.curb_ramp_detector` stays the
  CurbRamp entry, which is what `scripts/reinfer.py` and `scripts/detect_from_store.py` set.
  With more than one type, `load_detectors` hands every detector ONE shared `inference_lock`,
  so forward passes never overlap on the device. One detector keeps its own lock, so a
  CurbRamp-only run is constructed as before. The CurbRamp-only writer raises if it is handed
  a tagged or multi-type result.
- **Binding**: the manifest records `label_types`. It is bound like `detection_decode`:
  - a legacy manifest with records reads `['CurbRamp']` without a rewrite;
  - it is bound for `--gap-fill-only`, and never for `--scan-only`;
  - a mismatch is refused (new `--name`, never a resume).

  A non-CurbRamp set also binds every type's model in a `models` map beside the top-level model
  block. `bind_model` runs on each `models[T]`, so the revision, training-date and
  unknown-revision refusals all apply per type. Each run entry gains `models` too.
  `detect_from_store.bind_manifest` binds CurbRamp or refuses.
- **Consumers**:
  - `fuse_sites.py --label-type` (default CurbRamp). Per-type params (item 4):
    `fs.fuse_params_for(T, **flags)` = the spec's floor / operational threshold / rig mask,
    then `DetectorSpec.fuse_overrides` (an empty read-only map for CurbRamp), then every flag
    that is not None. So `--floor`, `--min-confidence`, `--max-range-m`, `--gate-chi2`,
    `--max-match-m`, `--residual-per-dof-max` and `--sigma-scale` DEFAULT TO None (= the
    type's value); `fuse_params_for('CurbRamp') == FuseParams()`. `FuseParams.label_type`
    (last field) records the type in `sites_meta.json`'s params; `sites.jsonl` lines carry no
    type and a CurbRamp fuse is byte-identical (tests/test_multitype_fusion.py). Each
    SlimPano's decode/border are that type's (`registry.type_markers`). A non-CurbRamp fuse
    writes `sites.<type>.jsonl` + `sites.<type>_meta.json`. A file that never ran the type is
    refused. `registry.PS_LABEL_TYPES` is the server's `LabelType` enum; every SPECS key must
    be one (pinned by a test), since the spec's name goes out verbatim as `labelType`.
  - `eval_sites.py --label-type T` loads T's detections (`load_city_at_height(...,
    label_type=)`) and fuses with `fuse_params_for(T, ...)`; CurbRamp keeps the CurbRamp-only
    refusal, another type only needs to have run. Default report dir `fusion_eval.<T>`. A type
    whose `benchmark_confidence` is not 0.55 is refused (TODO: `judged_gt_panos` and the
    sub-floor test still read `BENCHMARK_CONFIDENCE`).
  - `eval_ps_clustering.py --label-type T`: the two API pulls (`api_labels(T)`,
    `api_clusters(T)`), the label filter, `label_to_detection`, `synthesize_labels` (rig mask
    per spec), the loaders, the fusion arm's `fuse_params_for` and `ps_verbatim` are T's;
    the default dir gains `.<T>`. `run()` refuses any T but CurbRamp for now:
    `PS_THRESHOLD_KM` / `DIAG_SPAN_M` are the server's CurbRamp cut (TODO: the type's
    `label_clustering.py` threshold).
  - `export_benchmark.py --label-type`: strata at the type's benchmark tier. A multi-type
    record becomes a true single-type record: `label_types`, `models`, `model_*` and the
    markers are all that type's.
  - `position_check`: its report points read typed detections.
  - `agree_rate`, `mined_precision` (through eval_sites' loader, at its CurbRamp default) and
    `provenance_gate` **refuse** a file whose records ran anything but `{CurbRamp}`; eval_sites
    does so when scoring CurbRamp. The check runs inside `fs.load_results`' own pass
    (`require_types`), so there is no second scan.
- **NOT done**: the ranker review-order export (item 5; it consumes
  sidewalk-validator-ai#69's `analysis/ai_labels_eval.py` CSV, not yet merged there) and a
  second detector. (Submission, one POST per line and type with per-type sidecars, has since
  shipped; see Stage 2 above.)
