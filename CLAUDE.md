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
# are refused. Why it matters: the ground plane's distance is a per-pano camera height
# (issue #40) -- though in the depth frame, 6-16% short of the imagery's (see below).
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
# --apply-pose {auto,off,gravity,road} (issue #42). The DEFAULT is `auto`, which today is FLAT
# for every source: road-relative (pitch/roll minus the sequence's SfM road grade) passed the
# first #42 rule but FAILED the pre-registered shuffled-grade control (study section 10.5), so
# the road-frame default is withheld (fuse_sites.AUTO_ROAD_SOURCES = ()); GSV is flat on
# evidence (#52). `road` is opt-in. The flag takes an explicit value (`--apply-pose road`;
# a bare `--apply-pose` is an error), and gravity/road on a run holding GSV or Panoramax
# panos warns on stderr (GSV has no grade; Panoramax's convention is unmeasured).
# A Mapillary run from before #42 needs no
# rewrite: load_results derives the pose from source_metadata. sites_meta.json's `pose`
# block counts flat / gravity / road_relative / gravity_fallback panos -- the fallback (no
# usable sequence neighbour; 0.1% Morgantown to 6.9% Richmond) is the convention the study
# showed wrong for a car on a slope, so watch its rate on a new city.
python scripts/fuse_sites.py runs/richmond                      # = --apply-pose auto -> flat
python scripts/fuse_sites.py runs/richmond --apply-pose road    # opt-in, withheld as default

# CAMERA HEIGHT (issue #40). Every raycast uses geo.DEFAULT_CAMERA_HEIGHT_M = 2.6 unless
# asked otherwise; `per-pano` uses GSV's depth-measured height (pano block since #40, else
# the harvested runs/<name>/depth/index.csv) and is OPT-IN on evidence -- see the GSV depth
# section below and docs/camera-height-study.md. --implied-height is the instrument:
# bearing-only triangulation of multi-view sites, the height the imagery itself implies,
# by capture year. Associate near the answer (it drifts toward the association height).
python scripts/fuse_sites.py runs/paterson --implied-height --camera-height-m per-pano
python scripts/fuse_sites.py runs/paterson --camera-height-m per-pano   # opt-in fuse
python scripts/eval_sites.py paterson --camera-height-m per-pano --out /tmp/eval_pp

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

# Precision of positives mined from multi-view consensus (RampNet#158 step 1 /
# RampNet#102): for each site with >=3 operational panos and each judged benchmark pano
# nearby that is NOT one of its members (membership is the only test a real miner can
# apply — it has no verdicts), project the site into the pano (geo.ground_point_to_pano)
# and ask the reviewer's GT what is there. TWO denominators are reported side by side and
# they read the pre-registered rule differently, so never quote one alone: `hard-only`
# = tp/(tp+fp) is the rate at which mined targets are misses the model does not already
# make; `all-mined` additionally counts `already_detected` (the pano did detect the ramp,
# into a different site) as correct, because a miner cannot filter those out and the
# labels it ships for them are right. A verdict-FALSE detection at the site counts as a
# false positive under both (the reviewer looked there and said no). No GPU, no network;
# writes runs/<city>/mined_precision/{report.md,candidates.csv} — one CSV row per
# candidate, always carrying the nearest GT point and its distance (`within_match` says
# whether it adjudicated), which is how the localization hypothesis gets tested.
# --radius may not exceed the 25 m ground-raycast range: past it no GT mark can be
# placed, so a candidate could only ever be counted false (refused, not silently wrong).
# --camera-height is the #101 range-anchoring knob. GSV's 2.2 m is the median measured
# from GSV depth payloads (#40/#41) -- a depth-frame height, 6-16% below what the imagery
# implies (docs/camera-height-study.md); Mapillary serves no depth, so richmond has NO
# measured height — and no constant helps there (a sweep is flat at 0.31-0.32 over
# 2.0-2.6 m and worse below), which is itself the finding.
python scripts/mined_precision.py richmond
python scripts/mined_precision.py paterson --camera-height 2.2 --radius 10 15 20
# Name several cities to ALSO get the pooled headline (runs/_pooled/mined_precision).
# The RampNet#158 decision numbers are pooled, so these two commands are what
# regenerates them — the PR-body table comes from exactly these. `site_id` is a
# per-run serial, so never pool by concatenating candidates.csv and grouping on it;
# the city column is there because (city, site_id) is the key, as with a PS label_id.
# --camera-height takes one value for all, or one per city in the order named.
python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo
python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo \
    --camera-height 2.6 2.2 2.2 2.2 2.2   # richmond has no measured height; GSV does

# Eyeball the fusion: one HTML card per site with a crop from every member view,
# a plan view (cameras/rays/error ellipses/fused 1-sigma) and the RampNet verdict.
# Crops are cut on makelab2's native-res archive and pulled back as a tarball, then
# cached under runs/<city>/explorer/crops/ (re-renders are offline).
python scripts/site_explorer.py richmond              # 40 GT-seen sites
python scripts/site_explorer.py sao_paulo --select fp # only the false positives
python scripts/site_explorer.py richmond --select fragment  # over-split suspects
python scripts/site_explorer.py richmond --inline     # one shareable file
# ...--local-panos <dir> cuts crops locally instead (no SSH; e.g. a RampNet bundle).

# MAPILLARY RIG TILT (issue #42) -- the STUDY behind the production wiring. OpenSfM's
# `computed_rotation` sits in every Mapillary record's source_metadata; the decomposition
# (convention locked four ways) now lives in geo.py (geo.opensfm_pose /
# mapillary_pitch_roll) and the road grade in fuse_sites.sequence_grades, and this script
# imports both, so what it measures is what production does. Full write-up + the wiring's
# precondition in docs/mapillary-tilt-study.md. No network, no GPU; stdlib on top of
# geo.py/fuse_sites.py/eval_sites.py except `rectify`/`examples` (numpy, Pillow, SciPy) and
# `figures` (matplotlib).
# Eleven subcommands. All but `pose` take a city list (default: all five Mapillary runs)
# and --out; `pose` takes pano ids and --run. --out redirects the CSVs only: `figures`
# and `examples` always write into docs/figures/mapillary-tilt/.
python scripts/mapillary_tilt.py stats                 # tilt distributions + compass identity
python scripts/mapillary_tilt.py displacement          # flat vs tilt-corrected ground point
python scripts/mapillary_tilt.py grade                 # is the tilt the rig's or the road's?
python scripts/mapillary_tilt.py horizon               # GT marks raycast above the horizon
python scripts/mapillary_tilt.py ablation              # multi-view sign lock, by tilt/grade bucket
python scripts/mapillary_tilt.py eval                  # world P/R vs RampNet GT per convention
python scripts/mapillary_tilt.py precondition          # #42 gate: p90/median GT-to-site per
#   arm (off/gravity/road + road-shuffled-within/-across grade controls), one site set + one GT
#   set, 25 m cap, through PRODUCTION's loader, plus off-pool recall and unplaceable-mark counts
#   (survivorship); applies BOTH pre-registered rules and prints the verdict that sets `auto`
python scripts/mapillary_tilt.py rectify               # pixel-level sign lock (vertical edges)
python scripts/mapillary_tilt.py examples              # the annotated before/after strips
python scripts/mapillary_tilt.py figures               # redraw docs/figures/mapillary-tilt/
python scripts/mapillary_tilt.py pose <mapillary_id> --run richmond
# Two traps the study had to design around, both worth knowing before reusing the numbers:
# every convention must be scored on the SAME sites (an unplaceable member silently drops a
# whole site, so the rows are otherwise different subsets), and the GT recall DENOMINATOR
# moves with the convention (GT marks are raycast under the pose being tested) -- hence
# `recall_vs_off_pool_*` in gt_eval.csv. The ablation runs uncapped, so its raw mean/p90 are
# pessimistic about any correction that lengthens rays; median and mean/range are the
# centre. See sections 4.4/4.5/6 of the study. `ablation` and `eval` pin apply_pose=off and
# BENCHMARK_CONFIDENCE and reproduce PR #50's committed CSVs cell for cell (re-checked after
# the #42 roll-sign flip); the per-pano ROLL columns of `stats`/`rectify` now have PS's sign,
# the opposite of the committed tilt_summary/tilt_by_rig/tilt_by_sequence/verticality CSVs.

# POSE BACKFILL (issue #42). Offline, no token: fill camera_pitch/camera_roll from each
# line's own source_metadata (PS's /backupImage gate 404s on a null camera_pitch). It
# REFUSES to write over a file with a .submission.json or any .submitted* / .band-*.submitted
# sidecar beside it, in place OR as the --out target (that breaks send_to_ps.py's sha256
# guard for the live campaign and stales its position check) -- write a new file and push
# it pano-only instead; --min-confidence 2.0 submits zero labels. A pano-only push still
# UPSERTS each pano row's lat/lng, so every pano must be pushed from the file whose
# positions are live for it: Richmond's 72 posfix3seq panos are live at raw GPS, so they go
# out from results.posfix3seq.raw.pose.jsonl and are EXCLUDED from results.pose.jsonl's
# push (exact steps in PR #74's body).
python scripts/backfill_metadata.py runs/richmond/results.jsonl --pose --dry-run
python scripts/backfill_metadata.py runs/richmond/results.jsonl --pose --out runs/richmond/results.pose.jsonl

# GSV GROUND PLANE (issue #52) -- a STUDY, not production: the depth payload's dominant ground
# plane NORMAL per pano, decomposed into along-travel grade + cross-slope, and the direct test of
# #50's road-relative raycast claim (off / ground-normal / shuffled-normal, same site set, plus an
# exploratory rig-attitude arm and, on review, travel-only / magnitude-matched shuffles / cross-sign
# arms on their own `*_review` site set). Verdict UNDERCUT by the pre-registered rule (verdict() in the
# script, committed before the run) -- meaning the GSV estimates cannot improve the flat raycast, NOT
# that #50's mechanism is refuted (it is untested); write-up in docs/gsv-ground-plane-study.md. No network, no
# GPU. Inputs are read in place via --run-root (runs/<city>/{results.jsonl,depth/}); outputs go
# to runs/<city>/ground_plane/ + runs/_summary/ground_plane/, committed copies under
# docs/figures/gsv-ground-plane/data/. `planes` must run first (~10 min, multiprocess).
python scripts/gsv_ground_plane.py planes --run-root runs    # step 1 (+ frame checks)
python scripts/gsv_ground_plane.py chain --run-root runs     # grade persistence along links
python scripts/gsv_ground_plane.py ablation --run-root runs  # frozen-association spread
python scripts/gsv_ground_plane.py eval --run-root runs      # world P/R vs RampNet GT, 25 m cap
python scripts/gsv_ground_plane.py crossslope --run-root runs  # slope at ramp bearings (step 4)
python scripts/gsv_ground_plane.py verdict                   # the pre-registered reading
python scripts/gsv_ground_plane.py figures

# POSITION CHECK (SidewalkWebpage#5361) — a STANDARD part of the pipeline, not a step to
# remember: main.py runs it at the end of every run (--no-position-check skips it, e.g. no
# internet egress) and send_to_ps.py REFUSES a Mapillary file whose position_check.json is
# missing, stale (results_sha256 mismatch, or written under another verdict `rule`) or
# flagged (--ignore-position-check overrides). The manual commands below are for re-checks
# and for confirming a reposition output. Scores every pano's position against OSM street
# centerlines (one cached Overpass query, no GPU, no imagery, no second source needed).
# Mapillary serves two positions per image and the run submits one of them
# (--mapillary-position, default sfm = computed_geometry, a whole-run per-city choice — there
# is deliberately no per-sequence `auto`); SfM sequences can sit 8-10 m off the street as a
# block, and every label inherits its pano's error 1:1. The metric FLOORS near 1.75 m (the
# Laurens GSV control), so it gates only GROSS drift (issue #62): exit 1 = some sequence's
# submitted field sits > 5 m from the street (median unsigned cross-track, or most of it
# beyond 30 m of any street) AND the other field is closer PANO BY PANO on the same street by
# > 2 m (the paired metric) without being > 1.5x as scattered (IQR). Fields differing by
# <= 2 m are reported `undecidable` and never gate. The submitted field is judged per sequence
# from the coordinates, so a mixed (repositioned) file is checked correctly. Writes
# position_check.json + (--report) a self-contained position_report.html, both git-tracked
# beside manifest.json. A partial Overpass answer (HTTP 200 + `remark`) is refused and never
# cached.
python scripts/position_check.py runs/laurens --report
python scripts/position_check.py runs/laurens --report --labels <ps_v3_rawLabels.geojson> \
    --reference runs/laurens_gsv     # optional: the server's own placements; a second run over
                                     # the same area as an independent layer (Laurens only)
# ...then fix a flagged run WITHOUT re-detecting: rewrite the flagged sequences' pano lat/lng
# from the recommended field into a new file (new hash -> fresh submission campaign). An
# output that would put any pano elsewhere than its live position (newest campaign wins, per
# pano, over every .submission.json in the dir) is REFUSED, and so is sending it: PS places a
# label once, at insert, so resending moved panos DUPLICATES their live labels unless those
# are soft-deleted first — a whole-city decision, overridden only with --reposition-live-city.
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
retried — indoor GSV panos, non-360 or non-2:1 Mapillary images, incomplete metadata,
image bytes that arrived but do not decode — only PIL's "these bytes are not an image"
errors count, so a `MemoryError` mid-decode stays retryable) from retryable `failure` (left
uncached — network/HTTP errors, and a 200 whose Content-Type is not `image/*`). An image **404 differs by source on purpose**: Mapillary's
`thumb_original_url` is signed and expires, so a 404 there is transient (`failure`);
Panoramax's `hd` URL is plain, so a 404 there means the pixels are gone (`skipped`).
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

**Multi-view fusion (`geo.py`, `scripts/fuse_sites.py`, `scripts/eval_sites.py`)** —
issue #27 stages 2–3, a post-processing layer between detection and submission.
`geo.py` (repo root, stdlib-only, torch/numpy-free like `detectors/__init__.py`) is the
single home for geodesy: haversine + the declustering grid (imported back by
`export_benchmark.py`), a `LocalFrame` ENU tangent plane, and the ground raycast
`detection_ground_point` — flat-ground intersection at 2.6 m camera height with
closed-form anisotropic error from the 1024×512 heatmap quantization, **dropping** (never
clamping) rays beyond 25 m. GSV camera pitch/roll are deliberately NOT applied: the
`--pose-ablation` experiment measured that streetlevel's GSV equirects are already
gravity-rectified (details in `geo._world_ray`'s docstring). Mapillary's are available
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
**opt-in**; the default stays 2.6 m and fused output is byte-identical to before #40.
`sources/gsv.py` stores the height on every new GSV pano block (`camera_height_m`,
`camera_height_spread_m`, `ground_tilt_deg`, `depth_planes`, `camera_height_status`) —
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
there: the mirror cancels in the continuous form, and because mirroring phi flips only the
ray's x component, any plane with `nx == 0` — every level ground plane — cannot tell a
correct convention from a backwards one.
`harvest_depth.py` archives the payloads before they go away: the *JavaScript* API that
exposed depth was withdrawn in 2020 and anonymous tile access in ~2026, but the metadata
endpoint used here still serves it. `no_depth.txt`/`gone.txt` are append-only skip caches
(see the command block above). GSV only; Mapillary serves no depth (since #42 its tilt is
parsed from computed_rotation and written into the record, and fusion can apply it with
`--apply-pose road` or `gravity`, but it is OFF by default -- the shuffled-grade control
withheld the road-relative default; its camera HEIGHT is still the 2.6 m constant — #53).

**GSV ground plane (`scripts/gsv_ground_plane.py`)** — issue #52, a study. The same payloads'
dominant ground plane has a **normal**, and in depth.py's frame +x is camera-right, **-y is
camera-forward** (x = 0.5) and +z is down; `camera_frame_normal`/`slopes` turn it into grade
(rise ahead) and cross-slope (rise to the right), and the tests pin that mapping against
`depth.ground_range_at`. Measured 2026-09-23 on the four harvested runs: **the normal is not a
per-pano road-grade measurement.** Its slope along a street does not persist between linked panos
(r <= 0.31 within one drive, ~0 across capture months; per-pano noise bound 1.0-2.4 deg), its
grade follows the rig's metadata pitch only on climbs, and the 2025-26 rig reads ~2x the median
|grade| of earlier rigs on the same streets; its cross-slope does carry the road crown (falls right
on 56-67% of panos) and tracks the rig roll (but its left/right sign is NOT settled: the measured
cross-slope loses to its own flip, and a detection-side split says one plane is the wrong model across
a crowned street, not that the frame is mirrored). **Rotating GSV rays into the observed plane loosens
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
`pano_source` enum (`gsv`/`mapillary`/`infra3d`), `target_gsv_panorama_id` → `target_pano_id`,
and guarantees `links`/`history` arrays — so legacy JSONL files stay submittable unchanged.

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

## Output format notes

- The detector emits normalized coordinates; Stage 1 stores them normalized in the JSONL.
  The normalized → pixel conversion happens only in `send_to_ps.py`. Keep these in sync if
  you change either side.
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
