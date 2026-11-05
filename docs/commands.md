# Command reference

Every command for this repo's pipeline, archival and analysis scripts, with the notes that
go with each. Moved verbatim from `CLAUDE.md` (which keeps only the core commands) so the
instruction file stays inside Claude Code's size budget. "Above"/"below" in a note refers
to this file; "the GSV depth section" and similar point to [architecture.md](architecture.md).

## Contents

- [Environment setup](#environment-setup)
- [Scope an area: scan-only and reuse-scan](#scope-an-area-scan-only-and-reuse-scan)
- [Thinning spacing experiment](#thinning-spacing-experiment)
- [Run the labeler (GSV, Mapillary, Panoramax)](#run-the-labeler-gsv-mapillary-panoramax)
- [Panoramax asset study (HD vs SD)](#panoramax-asset-study-hd-vs-sd)
- [Panoramax measured: Bayonne and the error model](#panoramax-measured-bayonne-and-the-error-model)
- [Lyon (Panoramax training city)](#lyon-panoramax-training-city)
- [Gap fill](#gap-fill)
- [Ground truth lives in RampNet; benchmark bundles and native-res archives](#ground-truth-lives-in-rampnet-benchmark-bundles-and-native-res-archives)
- [GSV depth harvest and stand-in planes](#gsv-depth-harvest-and-stand-in-planes)
- [Multi-view fusion and pose modes](#multi-view-fusion-and-pose-modes)
- [GSV partial pose study and confirmatory run](#gsv-partial-pose-study-and-confirmatory-run)
- [Camera height: auto, per-pano, inventory oracle, Mapillary per-rig](#camera-height-auto-per-pano-inventory-oracle-mapillary-per-rig)
- [Fusion scored against RampNet GT](#fusion-scored-against-rampnet-gt)
- [Heatmap grid and sub-cell decode](#heatmap-grid-and-sub-cell-decode)
- [Seam band (--border keep / wrap)](#seam-band---border-keep--wrap)
- [Reprojection residual](#reprojection-residual)
- [PS server-side clustering eval (Richmond, beyond Richmond, Vancouver)](#ps-server-side-clustering-eval-richmond-beyond-richmond-vancouver)
- [Runs rebuilt from the pano store; provenance gate](#runs-rebuilt-from-the-pano-store-provenance-gate)
- [AI-vs-crowd agree rate](#ai-vs-crowd-agree-rate)
- [Server-side read (validation precision, auditor agreement)](#server-side-read-validation-precision-auditor-agreement)
- [Label-frame beta](#label-frame-beta)
- [Mined-positive precision](#mined-positive-precision)
- [Site explorer](#site-explorer)
- [Mapillary rig tilt study](#mapillary-rig-tilt-study)
- [Pose backfill](#pose-backfill)
- [DEM road grade](#dem-road-grade)
- [GSV ground plane study](#gsv-ground-plane-study)
- [Depth at the detection study](#depth-at-the-detection-study)
- [Aerial sidewalks (Tile2Net) study](#aerial-sidewalks-tile2net-study)
- [Footway: segmenter vs depth study](#footway-segmenter-vs-depth-study)
- [Position check and reposition](#position-check-and-reposition)
- [Post-submission coverage check](#post-submission-coverage-check)
- [Tests](#tests)
- [Submit to Project Sidewalk](#submit-to-project-sidewalk)

## Environment setup

```bash
# Set up the environment (GPU strongly recommended). requirements.txt is the single
# source of truth for pins; environment.yml just pins python and defers to it.
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# ...on Windows/macOS install the CUDA torch build first (linux-64 gets it by default):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# conda also works: conda env create -f environment.yml && conda activate sidewalk-auto-labeler
```

## Scope an area: scan-only and reuse-scan

```bash
# Scope an area first: pano count + runtime estimate, no model load, nothing processed
python main.py example_geojson/bend.geojson --name bend --scan-only
# ...every tile pass is saved to runs/<name>/scan.json; --reuse-scan (OFF by default —
# coverage churns, so a reused scan misses newer panos) lets the follow-up run skip it
python main.py example_geojson/bend.geojson --name bend --reuse-scan
```

## Thinning spacing experiment

```bash
# THINNING SPACING (docs/thinning-experiment.md): an UN-thinned run (--thin-spacing 0) of a
# small Mapillary/Panoramax area, then coverage vs spacing / proximity / newest-vs-random
# from the run's own scan.json (offline). Measured: 5 m stays the default; 10 m is a budget
# fallback that loses well-seen sites (10-18 pts on Panoramax, 3-9 on Mapillary).
python scripts/thinning_experiment.py runs/thinexp_bayonne [--min-confidence 0.3]
python scripts/thinning_experiment.py figures   # redraw docs/figures/thinning-experiment/ from the CSVs
# Lyon (#148, docs/thinning-experiment-lyon.md): three boxes thinexp_lyon_{a,b,c}; `subset` = offline
# densify cost of a thinned run, `crosscheck` = a box vs the canonical run; figures --out docs/figures/thinning-experiment-lyon
python scripts/thinning_experiment.py subset runs/lyon --from 10 --to 5 --panos-per-second 2.505 --out docs/figures/thinning-experiment-lyon/data
```

## Run the labeler (GSV, Mapillary, Panoramax)

```bash
# Run the labeler over an area; all per-area state goes to runs/<name>/
# (--name defaults to the geojson filename stem)
python main.py example_geojson/bend.geojson --name bend

# Same pipeline on Mapillary 360 imagery (needs MAPILLARY_ACCESS_TOKEN — a client
# token from mapillary.com/dashboard/developers — in the env or in gitignored ./.env)
python main.py example_geojson/richmond.geojson --name richmond --source mapillary

# ...or on Panoramax (federated open imagery, no token; coverage is mostly France today)
python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax
# ...paced at 1 picture/s + 1 metadata request/s (#161; per process), pool capped at 4,
# metadata batched 100 per /search. --panoramax-asset sd (opt-in: it loses ~37% of HD's
# 0.30 detections, docs/panoramax-asset-study.md) reads the 2048-px derivative but records
# the sensor frame; bound in the manifest. Talk to the maintainers before a big run.
PANORAMAX_PICTURES_PER_SECOND=1 python main.py example_geojson/bayonne.geojson     --name bayonne_sd --source panoramax --panoramax-asset sd
```

## Panoramax asset study (HD vs SD)

```bash
# The asset study behind the default (docs/panoramax-asset-study.md; GPU only for `run`):
python scripts/panoramax_asset_study.py sample --results <lyon results.jsonl> --data <dir>
python scripts/panoramax_asset_study.py fetch-sd --data <dir>     # 600 SD assets at 1/s
python scripts/panoramax_asset_study.py run --data <dir>          # hd / sd / sd_sim arms
python scripts/panoramax_asset_study.py score --data <dir> && python scripts/panoramax_asset_study.py figures
```

## Panoramax measured: Bayonne and the error model

```bash
# PANORAMAX MEASURED (issue #57 part 1; docs/panoramax-bayonne.md, corrected after two PR #125
# reviews). Bayonne ran in full at --thin-spacing 10 (rule: 10 m if the scan-only estimate > 16 h;
# it printed 21.1 h at 5 m; the manifest did not record the spacing at the time; since #126 every
# main-pass run entry records `thin_spacing_m` + `panos_before_thinning` and the run dir is bound
# to the spacing (Bayonne's manifest predates it and was BACKFILLED by hand to 10 m, marked
# `thin_spacing_backfilled`: run.log + a re-thin of scan.json reproduce the cache exactly)): 73,161 in-area
# pictures -> 28,634 thinned -> 28,524 processed, 0 failed, 6.7 h on the A40. 104 of the 110
# skips are GoPro MAX2 uploads whose `hd` image is a vertically CROPPED equirect (declared
# 7680x3840, served 7680x2940) -- correctly skipped; 3 more are unexplained (#127: truncated body
# or transient 404; both are retryable now, and those 3 were removed from the local cache). 0.135 detections per pano at 0.55 -- INSIDE the Mapillary range (Clovis
# 0.123 ... Richmond 1.048), worth GT, not anomalous. The municipal rig burns a white logo band
# from y 0.791 (dip 52.4 deg), inside the nadir mask (2 of 10,011 stored detections there).
# ERROR MODEL: the leave-one-out residual gained a normalized form (`chi2` = r'S^-1 r, S = the
# view's cov_en + the held-out solution's covariance; `seq_mates` / `pose_group` / rig
# breakdowns; `--fit-sigma-pitch TARGET` solves sigma_pitch and sigma_gps for a chi2/dof
# target). MAPILLARY_ERRORS reads 0.18-0.43 on the five Mapillary runs, so the rule was
# re-anchored on Richmond (0.269) before Bayonne was read. The amendment's stated mechanism
# (shared same-sequence error cancels) is NOT ESTABLISHED: Richmond reads the opposite way.
# What is measured: its 3 m sigma_gps is ~2.7x Richmond's leave-one-out-calibrated between-view
# scatter (1.11 m). Bayonne, MEASURED ON THE 5 m RUN (#147; doc s2 keeps a 10 m vs 5 m table):
# 0.556 [0.536, 0.576] (median form 0.559), m p50 3.44 -> still TOO TIGHT under the amended rule
# on the point estimate, but only 3% over the 0.538 bar and the CI now INCLUDES it (10 m: 0.580
# [0.539, 0.618], 8% over). Rig confound stands: Mapillary GoPro Max populations read 0.18-0.50
# (Richmond's GoPro Max views 0.499, Laurens 0.431, Morgantown 0.177; Bayonne 1.11x Richmond's,
# NOT distinguishable from it: site-bootstrap CIs just separate, but those 1,301 views are 14
# capture sequences and a sequence bootstrap overlaps -- Richmond [0.454, 0.559], Bayonne [0.534,
# 0.578]; panoramax_bayonne_figures.py writes sequence CIs beside the fig1/fig2a/fig2b/cuts site
# CIs only, as *_seq_lo/_hi (seq_lo/seq_hi in fig2b), #162 review).
# The site-size confound mostly went away: sites grew 3 -> 4 views, chi2/dof fell only 0.024, and
# within the run there is no trend by site size (0.549 / 0.535 / 0.579). The excess is ACROSS the
# ray (vs Richmond's GoPro Max views re-solved alone: along-ray +0.2..-0.5 m, cross-ray +0.6-0.8 m),
# not pitch: no sigma_pitch <= 15 deg fixes it. Calibrated sigma_gps: Bayonne 2.10 m vs Richmond
# 1.11, Richmond GoPro Max 1.81, Laurens 1.74, Morgantown 0.80. NOTHING adopted -- error_model_for(
# 'panoramax') stays MAPILLARY_ERRORS; GT (eval_sites) or a second Panoramax city decides. Pose:
# pers:pitch/roll LOOSENS the same-site spread in every sign convention (real-tilt p90 7.56 ->
# 9.85-13.20 m at 0.55, 225 groups), so Panoramax stays flat. GT: the RampNet bundle (125 panos,
# sampled from the 10 m run, reconcile 1:1) awaits review.
# DENSIFIED TO 5 m (#147, 2026-10-06; doc s1.1): manifest rebound 10 -> 5 by hand
# (`thin_spacing_changed`), resumed on Hyak with --reuse-scan --thin-spacing 5 -> 50,528 thinned,
# 21,885 new processed / 12 MAX2 skips / 0 failed; runs/bayonne now holds 50,409 panos (0.148 per
# pano at 0.55, census in data/census_5m/). The error model and Figures 1-6/7b are the 5 m run;
# skips.csv, Figures 7a/7c/7d and the bundle stay on the 10 m run (redrawing them needs network).
# The 10 m readings (data/ at 7a7a1a6, data/census/, skips, examples, bundle) read $R10, a run root
# whose bayonne/results.jsonl is the 10 m BYTE PREFIX (`head -n 28524 runs/bayonne/results.jsonl`,
# sha256 4f38ff52...; recipe in doc s5): pointed at runs/bayonne they read the 5 m file instead.
# Times in the doc are pre-#161 (unpaced); at today's 1 picture/s the archive of the 21,885
# added panos is ~6 h and a 5 m run from scratch ~14 h (doc s5, "Re-running it today").
python scripts/run_census.py $R10/bayonne --out docs/figures/panoramax-bayonne/data/census --band-y 0.79
python scripts/run_census.py runs/bayonne --out docs/figures/panoramax-bayonne/data/census_5m --band-y 0.79
python scripts/reprojection_residual.py bayonne richmond --camera-height-m 2.6 --refuse \
    --fit-sigma-pitch 0.269 --benchmark-root /nonexistent   # GT-free; --min-confidence 0.3 too
python scripts/panoramax_bayonne_figures.py data && python scripts/panoramax_bayonne_figures.py figures
#   data: run files -> data/fig*.csv (site bootstraps, seed 57; ~1.5-2 h (two sigma_gps
#   bootstraps, site and sequence), no GPU/network, byte-reproducible; --only for parts,
#   refused on ANY partial run (`inputs` or not, `cuts` too) whose run files differ from
#   data/inputs.csv; --allow-changed-inputs overrides);
#   skips / examples: network, on $R10 (skips also needs $R10's rebuilt 10 m cache, doc s5);
#   crops: from the RampNet bundle; figures: committed data only,
#   byte-reproducible (PNG + LF SVG)
```

## Lyon (Panoramax training city)

```bash
# LYON (RampNet#159 training-only Panoramax city; docs/panoramax-lyon.md). Slice 1 ran on makelab2's
# A40 from a separate worktree (~/sal-lyon): --reuse-scan --thin-spacing 10 --limit 40000, i.e. the
# first 40,000 of the 10 m set's sorted UUIDv4 ids (a uniform subset); runs/lyon was bound to 10 m for the
# slices (5 m since the densify below), and --limit continued past what was processed. run_census.py's
# rig_detections.csv gives detections per pano per (make, model, dimensions, capture year) at both
# tiers -- the per-rig diagnostic Bayonne lacked. No GT bundle (train-or-evaluate rule), no submission.
# runs/lyon/results.jsonl is now the 5 m run; slice 1 is its first 39,975 lines (10 m: 86,889), so the
# slice-1 and 10 m tables regenerate from a head -n prefix saved as results.jsonl (doc section 8)
head -n 39975 runs/lyon/results.jsonl > /tmp/lyon_slice1/results.jsonl   # mkdir -p first; sha256 802230ea...
python scripts/run_census.py runs/lyon --results /tmp/lyon_slice1/results.jsonl --out docs/figures/panoramax-lyon/data/census/lyon --band-y 0.79
python scripts/panoramax_lyon_figures.py figures    # committed CSVs only; census-scan needs scan.json
# DENSIFIED TO 5 m 2026-10-07 (#148 decision, #159; Hyak job 41504145): runs/lyon is now BOUND to 5 m,
# 174,405 processed of 174,525 (120 skips, all but one no view:azimuth); sections 1-7 of the doc stay
# the 10 m evidence, section 8 + data/census/lyon_5m/ + data/full_5m/ are the 5 m run.
```

## Gap fill

```bash
# GSV runs end with a gap-fill phase (issue #32): link-target panos the run's own
# records reference but the tile scan never enumerated (coverage churn) are fetched
# by id and kept if their metadata position is in-area. --no-gap-fill skips it;
# --gap-fill-only retrofits an existing run (add --scan-only to just count targets).
python main.py runs/paterson/area.geojson --name paterson --gap-fill-only
```

## Ground truth lives in RampNet; benchmark bundles and native-res archives

```bash
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
```

## GSV depth harvest and stand-in planes

```bash
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
python scripts/harvest_depth.py runs/paterson --reindex    # offline: recompute every derived
#   index.csv column from the archived files after a depth.py change; refuses if an indexed
#   file is missing or empty, and replaces index.csv only when the pass finds no anomaly.
#   (reconcile also recomputes any row from an index with older columns, and never launders
#   an anomaly: an unreadable or altered file keeps its prior row and recorded sha256.)
# STAND-INS OUT OF THE SPREAD (#47 follow-up). 98% of measured GSV payloads carry a small secondary
# stand-in plane (normal exactly vertical, 2.500 m; median 1.2% of the image), and
# depth.ground_plane's height_spread_m -- the per-pano sigma under --camera-height-m per-pano --
# used to take it in. It is now computed over measured planes only (depth.is_standin); the
# dominant plane, camera_height_m and every status are unchanged (asserted column by column).
# index.csv gains n_standin_planes / standin_pixel_share. Pooled spread p50 0.105 -> 0.074 m.
# T4 (height_qc) re-read: still not rising, so the stand-ins were not why the spread fails to
# predict the residual; every #44 verdict label is unchanged. auto and 2.6 m fuses are
# byte-identical; per-pano sites move slightly (paterson 13,135 -> 13,146; world P/R unchanged).
# GSV pano blocks written before this change carry the OLD camera_height_spread_m under the
# same key. An index.csv WITHOUT n_standin_planes is REFUSED by fuse_sites.load_depth_index
# (so every fuse that reads heights, auto included) and height_qc: `harvest_depth.py <run>
# --reindex` it (offline, minutes). sites_meta.json's camera_heights records
# spread_definition when per-pano heights came from an index. Addendum in
# docs/camera-height-study.md.
python scripts/depth_standin.py snapshot    # copy each depth/index.csv aside BEFORE --reindex
python scripts/depth_standin.py measure     # old vs new -> runs/_summary/depth_standin/ + docs copy
# no_depth.txt and gone.txt are skip caches for the two deterministic outcomes — delete one
# to force a re-check. Both are append-only: a pass that doesn't re-encounter an id
# (--verify, --limit) must never be able to erase it.
```

## Multi-view fusion and pose modes

```bash
# MULTI-VIEW FUSION (issue #27, stages 2-3). Associate a finished run's detections
# into physical-ramp sites (world-space raycast + constrained clustering; no GPU,
# no network); writes runs/<name>/sites.jsonl + sites_meta.json.
python scripts/fuse_sites.py runs/paterson
# ...--pose-ablation reports within-site spread per pitch/roll sign convention
# instead (the experiment behind GSV's flat default: the FULL pose loosens every city. It
# does NOT show the equirects are gravity-rectified -- they are rig-frame, #113).
# --apply-pose {auto,off,gravity,road,partial} (issues #42, #116). The DEFAULT is `auto`, which today is FLAT
# for every source: road-relative (pitch/roll minus the sequence's SfM road grade) passed the
# first #42 rule but FAILED the pre-registered shuffled-grade control (study section 10.5), so
# the road-frame default is withheld (fuse_sites.AUTO_ROAD_SOURCES = ()); GSV is flat on
# evidence (#52). `road` is opt-in. The flag takes an explicit value (`--apply-pose road`;
# a bare `--apply-pose` is an error), and gravity/road on a run holding GSV or Panoramax
# panos warns on stderr (GSV has no grade; Panoramax's convention is unmeasured).
# `partial` (#116 follow-up; OPT-IN, GSV only) feeds the ray the frozen leaked fraction of the
# stored pose: geo.partial_pitch_roll with geo.PARTIAL_POSE_K_GSV = (0.183 pitch, 0.382 roll),
# #116's pooled `auto` fit, i.e. (-0.183 x pitch, +0.382 x roll). A GSV pano missing either
# angle, every Mapillary/Panoramax pano, and every store-built GSV block (source_detail
# `ps_store`: the PS row's pose convention is unverified) raycasts flat, with one stderr
# warning per kind; sites_meta.json's `pose` block counts `partial`, `store_unverified_flat`
# and records `partial_coefficients`. eval_sites.py --apply-pose takes it too
# (fs.POSE_MODES) and prints the same warnings. `auto` is
# NOT changed: that waits on #116's pre-registered confirmatory run (below).
# A Mapillary run from before #42 needs no
# rewrite: load_results derives the pose from source_metadata. sites_meta.json's `pose`
# block counts flat / gravity / road_relative / gravity_fallback panos -- the fallback (no
# usable sequence neighbour; 0.1% Morgantown to 6.9% Richmond) is the convention the study
# showed wrong for a car on a slope, so watch its rate on a new city.
python scripts/fuse_sites.py runs/richmond                      # = --apply-pose auto -> flat
python scripts/fuse_sites.py runs/richmond --apply-pose road    # opt-in, withheld as default
```

## GSV partial pose study and confirmatory run

```bash
# GSV PARTIAL POSE (#116, a study; docs/gsv-partial-pose-study.md): a fraction of the stored
# tilt, fit on a seeded half (seed 116), scored on the other half against off / full / mirror
# and a |tilt|-bucket shuffled control, re-associated per arm, with the inventory referee and
# GT survivorship. VERDICT: FAIL on the recall clause only -> GSV's DEFAULT stays flat; the
# fraction is wired as opt-in `--apply-pose partial` (above). The study's arm_pose IS
# geo.partial_pitch_roll, so the study measures what fusion applies.
python scripts/gsv_partial_pose.py fit && python scripts/gsv_partial_pose.py score \
    --benchmark-root ../RampNet/benchmark && python scripts/gsv_partial_pose.py verdict
python scripts/gsv_partial_pose.py consistency paterson --benchmark-root ../RampNet/benchmark
#   production's `partial` beside the study arms on the TEST half (reported; gates nothing)
# CONFIRMATORY RUN (pre-registered on #116, 2026-09-30; constants and rule frozen there):
# the first GSV benchmark city with verdicts #116 never saw (Vancouver when its GT lands),
# whole run, arms off / partial / partial-shuffled / mirror; clauses (i), (iii), (iv) as
# #116, and (ii) sized to the pool: FAIL iff lost - gained >= k*(n), the smallest k with
# P(Binomial(n, 0.01) >= k) <= 0.05 (n = off-pool ramps at 2.5 m; below n = 50 a FAIL
# stands and a pass becomes INCONCLUSIVE; `loss-bar` prints the table; its power and its size
# under churn are in the doc addendum). A #116 train city (any case), a path-like name, or
# --seed != 116 is refused without --exploratory. A STORE-BUILT city (detect_from_store.py;
# Vancouver) must first pass the store-pose gate: PS-row pitch/roll vs streetlevel's on a
# seeded sample of the same ids (metadata only; network), one sign mapping agreeing within
# 0.1 deg on >= 95% of >= 50 panos -> the mapping is applied; fail or no network refuses.
# `--apply-pose partial` itself raycasts `ps_store` panos flat (warned, counted): the gate's
# mapping is applied in confirm's memory only and is NOT persisted, so a later default-switch
# PR must persist or apply it first.
python scripts/gsv_partial_pose.py confirm vancouver --benchmark-root ../RampNet/benchmark
python scripts/gsv_partial_pose.py confirm laurens_gsv --exploratory \
    --benchmark-root ../RampNet/benchmark     # the dry run; outputs labelled EXPLORATORY
python scripts/gsv_partial_pose_figures.py [--refresh]   # addendum figures 5-7, 9 (PNG 200 dpi +
#   SVG, byte-reproducible) from docs/figures/gsv-partial-pose/data/addendum_*; --refresh
#   recomputes that data (site examples re-fuse runs/paterson; simulations seed 116; no network)
```

## Camera height: auto, per-pano, inventory oracle, Mapillary per-rig

```bash
# CAMERA HEIGHT (issues #40, #79). fuse_sites.py DEFAULTS to `--camera-height-m auto` (#79):
# GSV panos get a per-rig height by capture year from the run's own depth-measured heights
# (fuse_sites.gsv_rig_assignment: a year whose median is < 2.1 m -> 2.0 m, else 2.5 m, but
# only with >= 50 measured panos that are >= 50% of the year -- else 2.5 m); Mapillary,
# Panoramax, undated GSV panos, and a GSV run with no year that meets that minimum (none
# measured, or a partly harvested depth index) stay at 2.6 m. sites_meta.json's
# `camera_heights` records the resolution and the year table. ONLY that CLI's fuse default
# moved (its --pose-ablation / --implied-height still resolve `auto` to 2.6 m):
# FuseParams() and every analysis script (eval_sites, mined_precision, agree_rate,
# reprojection_residual, ...) still default to geo.DEFAULT_CAMERA_HEIGHT_M = 2.6, so
# published numbers reproduce; `fuse_sites.py --camera-height-m 2.6` reproduces a pre-#79
# fuse. `per-pano` uses GSV's depth-measured height (pano block since #40, else the
# harvested runs/<name>/depth/index.csv) and is OPT-IN on evidence -- see the GSV depth
# section below and docs/camera-height-study.md. --implied-height is the instrument:
# bearing-only triangulation of multi-view sites, the height the imagery itself implies,
# by capture year. Associate near the answer (it drifts toward the association height).
python scripts/fuse_sites.py runs/paterson --implied-height --camera-height-m per-pano
python scripts/fuse_sites.py runs/paterson                              # auto (per-rig GSV)
python scripts/fuse_sites.py runs/paterson --camera-height-m 2.6        # the pre-#79 default
python scripts/fuse_sites.py runs/paterson --camera-height-m per-pano   # opt-in fuse
python scripts/eval_sites.py paterson --camera-height-m per-pano --out /tmp/eval_pp
# Which measured heights per-pano believes (#44): the pre-registered QC tests T1-T4 over
# the four harvested GSV cities, both association heights; writes runs/<city>/height_qc/
# and the cross-city verdicts + default-height decision table to runs/_pooled/height_qc/.
python scripts/height_qc.py
# PLACEMENT ORACLE (issue #79). Bend and Gainesville publish per-corner curb-ramp
# inventories (one point per ramp, at the ramp); `scripts/inventory_oracle.py` fetches them
# into runs/<city>/inventory_oracle/ (geojson untracked; inventory.json + report + CSVs
# tracked, sha256 in the record), re-solves Bend's and Gainesville's sites (the only GSV
# cities with an inventory; Paterson and Sao Paulo have none) under 2.6 m / per-pano x1.08 /
# per-rig with the association frozen from the 2.6 m fuse, and scores site-to-inventory
# distance one-to-one. Gainesville (64% 2026 rig) decides, Bend (84% 2024) guards the old
# rig. The verdict rule is pre-registered on #79 (inventory_oracle.verdict);
# docs/placement-oracle.md. VERDICT (amended 2026-09-26, posted on #79 before scoring): rule 4
# read one-sided, and (d) -- per-rig, but a vintage keeps its depth-median height only with
# >= 50 measured panos that are >= 50% of it (else 2.5 m) -- replaces (c), which gave thinly
# measured old vintages 2.0 m. (d) PASSES and is selected, and is now fuse_sites.py's
# default (`auto`); the oracle scores the production function itself.
# Network only in `fetch` (the registry's three ArcGIS hosts -- Vancouver's is the generic
# services.arcgis.com, shared by every ArcGIS Online org, but the layer path is fixed by the
# registry); a cached pull is reused (refused if
# area.geojson's bbox changed), --refresh re-pulls it. `score --pool-anchor frame --out
# <dir>` is the rule-3 anchoring sensitivity, never read by verdict.
python scripts/inventory_oracle.py fetch bend gainesville vancouver
python scripts/inventory_oracle.py score bend gainesville     # ~100 s for both, no GPU/network
python scripts/inventory_oracle.py verdict
# Mapillary has no depth: `per-rig` (issue #53) reads runs/<name>/camera_heights.json, a
# per-rig-class height measured by scripts/mapillary_height.py (bearing fixed point + #76's
# scale identity, a pre-registered rule, then a production gate). OPT-IN, and since #89 it
# applies NOTHING: B read as a fixed point has no validated estimator in any city (rule V:
# at the model's full noise both the line and the local crossing read +0.25-0.53 m high at a
# 1.8 m rig), so rule 3 reads b_unvalidated everywhere, #53's Annapolis 2.376 m is withdrawn,
# every group is 2.6 m and every table says recommended: false -- docs/mapillary-camera-height.md s8.
# #89: B is read as a fixed point; --validate FIRST (simulation on each city's real view graph,
# rule V picks the line or the local-crossing estimator per city; ~15 min on 10 workers, resumable).
python scripts/mapillary_height.py --validate richmond laurens clovis morgantown annapolis
# ...--exploratory adds the #98 review arms (B at h_true, oracle-clean B, local slope, noise-0
# cells: estimator_validation_exploratory.csv); rule V never reads them. docs s8.3.
python scripts/mapillary_height.py richmond laurens clovis morgantown annapolis
python scripts/fuse_sites.py runs/annapolis --camera-height-m per-rig --out /tmp/s.jsonl
# #87: B read at one association height is pulled toward it; as a fixed point it meets A (doc s7).
# Its line fixed point amplifies bias by 1/(1-b_B); #89: neither estimator passes rule V in any city (docs s8).
# `simulate` now draws the null at the injected noise (-> simulate_matched.csv); #87's committed
# simulate.csv was full-sigma at every noise scale and reproduces with --null-unmatched.
python scripts/height_gap.py sweep richmond --group gopro/max --sequences  # + simulate/gt/verdict
```

## Fusion scored against RampNet GT

```bash
# Score fusion against RampNet GT in world space: world P/R, the union-recall
# decomposition, stage-4 promotion calibration, vintage + match-radius ablations.
# Reads ../RampNet/benchmark/<city>/{verdicts.json,records.jsonl} as data (no
# RampNet code import); writes runs/<city>/fusion_eval/{report.md,*.csv}.
python scripts/eval_sites.py paterson
python scripts/eval_sites.py paterson --vintage-ablation
```

## Heatmap grid and sub-cell decode

```bash
# HEATMAP GRID (issue #111; docs/heatmap-grid.md). RampNet's heatmap is an 8x bilinear upsample
# of a stride-32 map, so every detection sits on residue 3/4 of an 8-cell block. `grid` is the
# census per run and tier; `sigma` re-fuses at the benchmark tier with sigma_peak_px 1.0 vs 2.31
# (one coarse cell's uniform quantization) at 2.6 m and auto and applies the pre-registered rule
# (verdict: KEEP 1.0). Writes docs/figures/heatmap-grid/data/. No GPU, no network.
python scripts/heatmap_grid.py grid paterson bend gainesville sao_paulo richmond
# ...the rare off-grid peak is a clipped plateau (#151 q2; doc section 5): decode.py finds peaks on
# clip(heatmap, 0, 1), so a peak > 1.0 is a flat top and its raster-first pixel wins (skimage 0.26's
# stable sort; scikit-image is pinned to 0.26.0 in requirements.txt since PR 158, and the tie order
# is also pinned in tests). `offgrid` is the census (prints only; --out writes the committed CSVs);
# `offgrid-heatmaps` (GPU) checks it on real heatmaps of the local decode panos (46/46). The one
# off-grid peak below 1.0 (Paterson, 0.57) is not a clip: it is consistent with (reachable by,
# not reproduced) a float32 near-tie of its two knots (`offgrid-tie`, GPU, fetches the pano once);
# `offgrid-tie-backend` shows the off-grid counts are this GPU's numeric backend, not the model's.
# Both write the committed data dir by default and reproduce only on the stack their meta.json
# records (they print a WARNING if this machine differs): elsewhere pass --out <scratch>.
python scripts/heatmap_grid.py offgrid --results <runs...> --labels <raw_labels.geojson> --labels-user <ai user_id> --unmatched runs/vancouver/provenance_gate/unmatched.csv
python scripts/heatmap_grid.py offgrid-heatmaps
python scripts/heatmap_grid.py offgrid-tie
python scripts/heatmap_grid.py offgrid-tie-backend
python scripts/heatmap_grid.py sigma paterson bend gainesville sao_paulo richmond
python scripts/eval_sites.py paterson --sigma-peak-px 2.31 --out /tmp/eval_s231   # one cell
# SUB-CELL DECODE (#111 decode half, docs/heatmap-grid.md section 4). OPT-IN, default argmax:
# `main.py --decode gaussian` / `reinfer.py --decode` place each peak by RampNet#221's rule
# (detectors/decode.py; detectors/rampnet_subcell.py is RampNet's subcell.py VERBATIM, hash-pinned
# by tests/test_decode.py -- vendored because the Hub package does not ship it yet). Same peaks,
# same scores; only (x, y) move: median 1.5 heatmap px per axis, p99 3.5, max 8.7 measured (a
# re-anchored peak can move about one coarse cell). A run is BOUND to its decode (manifest
# `detection_decode`), gaussian records carry "detection_decode": "gaussian" (argmax records and
# submission records are byte-identical to before; manifests and sites_meta gain the key), and
# fuse_sites / reinfer --verify /
# --write-band-file / send_to_ps.py refuse a mix (--allow-mixed-decode, recorded); eval_sites and
# provenance_gate (and agree_rate, site_explorer, gsv_partial_pose, mapillary_tilt) refuse a
# non-argmax run (bundles and live labels are argmax), so a gaussian city has no provenance
# gate yet. A mixed send under --allow-mixed-decode records the full mix. A gaussian
# campaign beside live argmax labels is a whole-city frame change and Jon's call.
# The measurement: one forward pass, both decodes (GPU `detect`), then CPU steps.
python scripts/subcell_decode.py residual --rampnet-root ../RampNet   # reads the committed decode/ outputs
python scripts/subcell_decode.py sigma-table paterson bend gainesville sao_paulo richmond --sigma 4.24 3.67
python scripts/subcell_decode.py world laurens_gsv --split laurens_gsv --results runs/laurens_gsv/results.jsonl     --decode-file docs/figures/heatmap-grid/data/decode/decode_laurens_gsv.jsonl.gz --work-dir /tmp/w
python scripts/subcell_decode.py figures
```

## Seam band (--border keep / wrap)

```bash
# SEAM BAND (#130; docs/seam-band-130.md). OPT-IN, default exclude: the peak finder drops every
# peak within 10 heatmap px of the heatmap edge (skimage's exclude_border default), so the 20
# columns at the 360-degree seam -- coarse columns 0 and 127 of the exact x8 upsample, 5.6 deg of
# azimuth -- never yield a detection. `main.py --border keep`
# / `reinfer.py --border keep` use RampNet's rule instead (exclude_border=False, NO NMS across the
# seam, so a straddling ramp can give two peaks). `--border wrap` (#130 follow-up) is keep plus NMS
# wrapped across the seam (skimage's finder re-implemented on a cylinder, decode._cylinder_peaks;
# tested against a brute-force cylinder incl. clipped ties), this repo's own rule: one peak per
# straddling ramp; bound and guarded like keep, a frame of its own (keep/wrap mixes are refused too);
# estimated on the committed Laurens data to remove 3 of keep's 15 gained peaks at 0.30 and the
# one duplicate site (seam-band-130.md section 9). Bound exactly like the decode: manifest
# `detection_border`, keep records carry "detection_border": "keep" (exclude records and
# submission records byte-identical), fuse_sites / reinfer --verify / send_to_ps.py refuse a mix
# (--allow-mixed-border, recorded; send_to_ps also reads sibling runs/*/ campaigns on the same
# endpoint, so a re-run under a new --name is caught), --write-band-file refuses it outright, and eval_sites (via
# es.require_bundle_frame: agree_rate, site_explorer, gsv_partial_pose, mapillary_tilt) and
# provenance_gate refuse a keep run (bundles and live labels are exclude). detect_from_store.py
# stays exclude. RULE: a NEW city may use --border keep from its first run; an EXISTING city only
# under a new --name, and shipping it beside live exclude labels is Jon's call.
# The measurement (Laurens, both arms; CPU): `check`/`peaks` on makelab2 where the #111 coarse maps
# live, then `world` anywhere; `verify` re-derives summary.json from the committed data, no network.
python scripts/seam_band_130.py check laurens --results runs/laurens/results.jsonl --coarse-dir /homes/gws/jonf/decode111/coarse/laurens
python scripts/seam_band_130.py peaks laurens --results runs/laurens/results.jsonl --coarse-dir /homes/gws/jonf/decode111/coarse/laurens
python scripts/seam_band_130.py world laurens --results runs/laurens/results.jsonl --peaks runs/laurens/seam_band_130/peaks.run.jsonl
python scripts/seam_band_130.py summary && python scripts/seam_band_130.py verify
```

## Reprojection residual

```bash
# Leave-one-view-out REPROJECTION RESIDUAL (issue #36; findings in
# docs/reprojection-residual.md). GT-free: for every site with >= 3 operational views,
# drop each view, re-solve the site from the rest (an information-form subtraction) and
# compare with what that view emitted, in heatmap px and metres; plus a range-scale fit
# (member - held-out = s*g exactly under a range scale k, s = 1 - 1/k), pooled and per
# capture year. GT-anchored: project the full and the leave-that-pano-out site into each
# judged benchmark pano and compare with the reviewer's box centre / missed click. No GPU,
# no network; reads runs + ../RampNet/benchmark as data, writes
# <out-root>/<city>/reprojection/ and <out-root>/_summary/reprojection/. Reads sites.jsonl
# when it matches the model (it says STALE if results.jsonl grew since); --refuse re-fuses
# in memory, which is what the published tables use. A regression check for any new city,
# source or rig: it needs no ground truth.
python scripts/reprojection_residual.py paterson --camera-height-m 2.6 per-pano --refuse
python scripts/reprojection_residual.py bend paterson gainesville sao_paulo richmond     clovis laurens laurens_gsv annapolis morgantown --camera-height-m 2.6 per-pano     --refuse --publish docs/figures/reprojection-residual/data
```

## PS server-side clustering eval (Richmond, beyond Richmond, Vancouver)

```bash
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
# The pure scoring (`score`, `near_cluster_rate`, `size_precision`, `wilson`, the inversion,
# the report rows) lives in `scripts/clustering_metrics.py`, pandas/scipy-free, so CI covers
# it (#133); `eval_ps_clustering.py` re-exports every name.
# --ps-script points at SidewalkWebpage/scripts/label_clustering.py for the
# verbatim-reproduction check. The two API pulls are cached in the output dir and REUSED on
# a re-run (the run prints how old they are) — pass --refresh to re-pull, since the server
# re-clusters nightly. --camera-height-m sets the scoring frame (the server's own is
# 2.341219672825709) and picks the default output dir, so the two frames never overwrite
# each other. report.md/arms.csv are git-tracked like manifest.json; the two API geojson
# are not, so the report records each pull's url, fetch time, sha256 and feature count.
# It also takes per-pano / auto / per-rig (#56), resolved by fuse_sites.load_at_height -- the one
# resolver fuse_sites, eval_sites and mined_precision share -- into ps_clustering_eval_<mode>/;
# the report says how many panos fell back to 2.6 m (Richmond, Mapillary: all of them).
# The committed Richmond reports are PINNED to the cached 2026-09-21 pull (--labels/--clusters):
# a fresh pull now holds ~3.4k band labels from results.band.jsonl and the 72 posfix3seq panos'
# raw-GPS labels, so against results.jsonl it refuses (unmapped AI labels) -- correct behaviour.
python scripts/eval_ps_clustering.py richmond --server https://sidewalk-richmond.cs.washington.edu
# BEYOND RICHMOND (#106; docs/ps-clustering-eval.md "Beyond Richmond" + "City-inventory
# scoring"). --offline scores a city with NO server: one label per stored detection >= the
# tier (rig-masked) at send_to_ps's integer pixel, placed by ps_placement.py -- the server's
# own estimator ported exactly (SW's 59-case parity fixture, tests/fixtures/). Regions are the
# NEAREST STREET's (the server's insert rule; --server only GETs /v3/api/streets), not
# point-in-region-polygon (6% wrong on Richmond). --offline-check (live mode) proves it:
# Richmond/Laurens placement exact to 1e-6 m (gated on ALL labels); all-AI 7.5 m partition
# Richmond 0.996 / Laurens 0.945 (1.000 / 1.000 with live humans). Regions snap to OPEN
# streets only, as the server does (/v3/api/streets returns all). --results reads
# another file (Laurens is live from results.raw.jsonl), --split names the benchmark split,
# --mask-rig for a live city whose rig labels were soft-deleted. The PS partition is BLOCKED
# (exact: single-linkage components at max t + 0.5 m), so ps_citywide runs at any size.
# fusion_server+attach = one pre-declared bearing rule for unplaceable labels (15-60 m along
# the ray, 3 m perpendicular; never tune it on GT). Pooled (10 cities; Budapest excluded --
# its local file is partial): the server rule fragments ~2x fusion (frag 5 m 0.24 vs 0.12) at
# level coverage (0.872 vs 0.876); a 12.5 m cut gets most of the way to fusion's fragmentation
# (0.15 vs 0.12) but costs 3.8 pts coverage -- no constant fixes it. Unplaceable labels are real ramps (0.96 precision) and
# 93% attach. Part 2 (city inventories, pre-registered on #106): NOT ESTABLISHED -- fusion's
# split advantage in Gainesville holds at `auto` and reverses at 2.6 m (Gainesville alone:
# it holds at 2.6 m in Bend and in the Part 1 pool); Bend misses the 5-pt split bar.
# Five runs (bend clovis morgantown annapolis richmond) predate the storage floor: their 0.30
# tier IS their 0.55 tier. The pooled driver is resumable (results, streets and verdicts
# sha256 + SCORER_VERSION; bump SCORER_VERSION by hand when a number can move), pulls each
# city's streets once, and REFUSES (exit 1, pooled outputs untouched) when any cell is
# missing or stale, unless --allow-partial.
python scripts/eval_ps_clustering.py bend --offline                      # -> ..._offline_t0.55/
python scripts/eval_ps_clustering.py laurens --split laurens_mapillary --results \
    runs/laurens/results.raw.jsonl --min-confidence 0.3 --mask-rig --offline-check \
    --server https://sidewalk-laurens.cs.washington.edu
python scripts/clustering_eval_pooled.py            # every (city, tier, frame) cell, then pool
python scripts/clustering_eval_pooled.py --pool-only   # -> runs/_pooled/ps_clustering_eval/
python scripts/inventory_clustering.py score bend gainesville && python scripts/inventory_clustering.py verdict

# CLUSTER-REVIEW GT (RampNet#224; protocol RampNet docs/cluster_review_protocol.md, rubric
# RUBRICS.md §6 -- those, the gallery, rampnet/cluster_review.py and the bundle live ONLY on
# RampNet branch cluster-review-224 for now, linked at e869be2 (PR #280) in
# docs/ps-clustering-eval.md; TODO: repoint to RampNet main once it merges).
# export_cluster_review.py samples OSM intersection / mid-block units (seed 224, 20 per stratum, 15% no-label, >= 60 m apart, 30 m windows by SERVER position; sampling rule v2
# drops any window touching a bridge / covered / layer>=1 way or a street below grade --
# `grade_separated`, protocol rule 6b) and writes a
# bundle into RampNet: snapshot.json + corners.jsonl (each label with its `deployed` and
# `fusion_server+attach` seed group -- inventory_clustering.server_arms, whose fusion arms come
# from fuse_server_labels, the helper score_server_arms also calls; checked against
# split_figures/state.pkl), Esri aerials and
# 45 deg / 512 px crops cut on makelab2 from the sharded PS store (site_explorer's worker; item
# `path` + job `draft_width`). Resumable; never re-samples corners.jsonl; reconciles crops and
# aerials. The run dir's inputs are read in place (the work dir, default
# runs/<city>/cluster_review, may sit inside it). cluster_review_score.py reads assignments.json AS DATA
# (no RampNet import), rebuilds every #56 arm, applies inventory_clustering.assignment_metrics +
# the pre-registered rule (fusion_server+attach vs ps @ 7.5 m) and the inventory calibration; with
# no assignments.json it writes a "NO GT YET" report and exits 0. Every arm scores the SAME labels:
# human labels out, a label an arm does not hold = its singleton. A `cant_judge` unit (reason
# required) counts in no metric and is listed with its reason. It refuses deployed clusters /
# results.jsonl whose sha256 differs from snapshot.json (--allow-arm-mismatch, recorded). Nobody but a human reviewer in
# RampNet's cluster_review_gallery.py writes an assignments file. The decision is headed
# "pre-registered" only for rater A's full pass. Anything else is REFUSED before an arm-level
# number exists (RampNet protocol Amendment 5, S1: one rater who knows the hypothesis must not
# see arm tables mid-study) -- a pilot or partial pass always, a role-b retest until the bundle's
# assignments.json is A's full pass; --allow-partial-descriptive overrides, headed DESCRIPTIVE
# and recorded. The report names the labeler commit it ran at (and any dirty tracked .py).
# The frozen inputs (label pull, deployed clusters, results.jsonl, streets, osm.json, inventory)
# are untracked; their sha256s and locations are in the doc's "Frozen inputs" table.
# --run-dir: a run dir holding the Vancouver run and those inputs, e.g. runs/vancouver
python scripts/export_cluster_review.py vancouver --run-dir runs/vancouver \
    --bundle ../RampNet/benchmark/vancouver/cluster_review [--no-crops] [--skip-aerial]
python scripts/cluster_review_score.py vancouver --run-dir runs/vancouver \
    --bundle ../RampNet/benchmark/vancouver/cluster_review    # -> runs/vancouver/cluster_review/score/
# CORNER INVENTORY (RampNet#238; protocol + results docs/corner-inventory.md). Every eligible
# #224 unit (no sampling; OSM functions imported from export_cluster_review.py, eligible counts
# asserted against the exporter's), legs walked out to 20 m; oneway same-name carriageways
# (within 90 deg) and *_link slip lanes (within 60 deg) merge by OSM tags, then any legs within
# 30 deg chain (angle alone left divided-road medians as corners -- review B1), corners =
# sectors between legs; state per unit / corner = present (operational fused site, or deployed
# cluster) > absent (a run pano within 25 m of the corner point) > unobservable. CAUTION: a
# store-built run holds the LABELED panos (+ a 300-pano unlabeled sample), so "observed" is
# conditioned on a label having been made nearby -- read every absence number with that.
# `census` makes one streetlevel metadata request per pano (no image), cached with fetch time.
python scripts/corner_inventory.py build --run-dir ../sal-vancouver/runs/vancouver \
    --osm ../sal-cluster-review/runs/vancouver/cluster_review/osm.json \
    --units224 ../RampNet/benchmark/vancouver/cluster_review/corners.jsonl \
    --out runs/vancouver/corner_inventory
python scripts/corner_inventory.py score --out runs/vancouver/corner_inventory
python scripts/corner_inventory.py census --out runs/vancouver/corner_inventory   # network
python scripts/corner_inventory.py score-assignments --out runs/vancouver/corner_inventory \
    --assignments ../RampNet/benchmark/vancouver/cluster_review/assignments.json
python scripts/corner_inventory.py figures   # docs/figures/corner-inventory/ from committed
#   counts.csv + decision.json only (matplotlib); byte-reproducible, re-run leaves git clean
# Vancouver split examples (#56): ramps one arm splits and another does not, drawn on Esri
# aerial tiles at SERVER placement (every cluster placed) -> docs/figures/vancouver-splits/
python scripts/split_figures.py            # --rebuild recomputes the cached partitions
# VANCOUVER (#56, scored under the amended scope pre-registered on #56; docs "Step 2"). No
# RampNet GT and a run rebuilt from the pano store that the gate STOPPED (heatmap plateaus,
# #111), so the confirmatory arms are the server's own labels only: deployed, ps@t on server
# positions, fusion_server, +attach. The CONFIRMATORY fragmentation test is metric (b), the
# near-cluster proxy (share of clusters with another cluster within 5/7.5/12.5 m; SERVER
# frame for arms with label ids), and it came out against Part 1: fusion_server read MORE
# near pairs than ps @ 7.5 m, not half. The inventory split is DESCRIPTIVE there and
# frame-dependent (each arm reads best in the frame it clusters in; the `label_set: mapped`
# rows of inventory_clustering/report.md compare the frames on one label set) -- never quote
# it as the confirmatory answer. `--no-gt` skips the GT join; `--ai-user` names the AI account
# so the 22% of its labels the rebuilt run does not reproduce pixel-exactly still count as AI
# (never placeable; they enter fusion_server at the tier; an error with --offline). Without it
# such labels are refused (#105). Other GT-free output: partition agreement (ps_repro +
# --offline-check), the deployed-partition diagnostics, validation-based precision by cluster
# size from HUMAN votes (validation_precision.csv). Inputs are FROZEN: the gate's
# raw_labels.geojson, clusters + streets pulled once into ps_clustering_eval/; pass them by
# path and never --server/--refresh (a missing pull is a stop, not a re-pull). No --mask-rig:
# the rig labels are live. `--offline` at 0.55 / 0.30 and the fusion arms at auto / 2.6 /
# per-pano are EXPLORATORY (rebuilt-run detections). A bare `inventory_clustering.py score`
# scores Bend + Gainesville only; Vancouver only when named, and it reads
# ps_clustering_eval/streets.geojson (refuses if absent). Replication block in the doc.
V=runs/vancouver; P=$V/ps_clustering_eval
python scripts/eval_ps_clustering.py vancouver --no-gt --ai-user 51b0b927-3c8a-45b2-93de-bd878d1e5cf4     --labels $V/provenance_gate/raw_labels.geojson --clusters $P/clusters.geojson     --streets $P/streets.geojson --offline-check --ps-script <SW 0062ed0>/scripts/label_clustering.py
#   ...the same with --camera-height-m auto / per-pano -> ps_clustering_eval_{auto,per-pano}/
python scripts/eval_ps_clustering.py vancouver --offline --no-gt --min-confidence 0.3     --streets $P/streets.geojson      # exploratory; and at 0.55
python scripts/inventory_clustering.py score vancouver                # descriptive, no verdict
```

## Runs rebuilt from the pano store; provenance gate

```bash
# A RUN REBUILT FROM THE PANO STORE (issue #56; runbook in docs/ps-clustering-eval.md, "Step 2").
# For a city whose results.jsonl was not kept and whose panos have partly left GSV (Vancouver).
# detect_from_store.py reads pixels from the PS pano store (<id[:2]>/<id>.jpg or flat) and the
# pano block from the server's /backupImage/<id>/metadata (sequential, spaced, 429 honoured,
# cached per id in store_metadata/; source_detail "ps_store", no links/history/depth), and runs
# them through main.py's detect/record/resume path. The id list (labels' panos + a seeded
# unlabeled sample) is frozen in store_ids.txt on first use; store_selection.json (which binds
# the resolved store, layout and SERVER, checked on every call) and store_sampled_ids.txt are
# git-tracked. --metadata-only needs no model. The metadata 404 is cached as
# `metadata_404_null_field_or_no_file` -- NOT "no backup": the server 404s on a null pano_data
# field (usually camera_pitch) as well as on a missing file. jpg_missing and that 404 are
# poison-guarded (> 5% of a pass, min 20: not cached; --accept-skip-rate after a hand check).
# A store-built run is fenced off both ways: main.py refuses a manifest with `pixels`, the
# runner refuses a run dir main.py wrote, and send_to_ps.py refuses the file (its labels are the
# live ones; --allow-store-file overrides).
# provenance_gate.py then checks that the rebuilt run can stand in for the deployed one. The
# match rule is in COARSE heatmap cells since #111 (2026-09-30; `--rule coarse-cell`, the
# default): Arm S (always, the store run) and Arm Z (optional, --control: a zoom-3 control on
# 200 seeded labeled panos via --draw-control then reinfer.py --ids) each match a label when a
# >= 0.55 detection lies within +/-1 coarse cell = +/-(8W/1024+1) x, +/-(8H/512+1) y
# (Chebyshev, seam-wrapped), >= 0.98 of joinable labels (exact_share at +/-1 px reported, not
# gated). Why cells: RampNet's heatmap is an 8x bilinear upsample of a stride-32 map, so every
# detection sits on residue 3/4 of an 8-cell block and a near-tie flips the argmax 7-8 cells on
# a small input change; the report counts matches by distance class and names that flip.
# `--rule pixel-96` reproduces the rule PR #108's Vancouver report ran under (Arm S
# +/-(W/1024+1) px, Arm Z +/-1 px; amended after the PR #96 review, before any Vancouver
# number); PR #108's committed report is reproduced by `provenance_gate.py vancouver --control
# runs/vancouver/control_zoom3.jsonl --rule pixel-96` -- without the flag the same path is
# overwritten with the coarse-cell reading. --control refuses a control pano not in
# control_ids.txt (--control-ids), and the report says how many drawn panos the control holds
# (Vancouver: 141 of 200; the other 59 are no longer served by id, so Z is conditioned on
# survival). UNDETERMINED unless nothing is pending and joinable >= 0.95 of labels with a store
# JPEG; STOP if unclaimed tier detections on labeled panos exceed 0.02 x joinable (unchanged).
# Vancouver under the coarse-cell rule (exploratory, #111; the #56 decision stands): still STOP
# -- S 0.930 (11.3% of its matches are flips; 4,268 of 4,477 misses are sub-0.55 at the spot),
# Z 0.987 passes, P 0.029 fails; docs/heatmap-grid.md. Both checks now say WHY a miss did not
# reproduce (#111 item 3; diagnostics, never a rule): the gate's unmatched.csv carries
# `miss_class` (below_tier_in_tolerance / tier_flip / below_tier_flip / beyond / ...; pixel-96
# exposes the tier classes, coarse-cell cannot), and reinfer.py --verify classes each
# non-reproducing pano (threshold / flip / off_grid / jitter / new_or_lost / border_band_only /
# pano_drift; --mismatch-csv PATH for the rows). harvest_depth.py --from-store indexes
# pano-tools' v3 .depth.npz in place (same index.csv schema; --check-store-frame N draws until
# N panos are checked against live payloads in the image frame and records the result in
# depth/store.json -- a mirrored index array fails, a payload Google has revised since reads
# `revised`; the index pass says "frame unchecked" without it). A store-bound depth dir is
# ALWAYS addressed with --from-store (--verify --rehash too): without it the call is refused.
# pano-tools' `unavailable` goes to unavailable.txt. depth_at_detection/gsv_ground_plane read
# *.json.gz payloads, not the index, so they do not see a store index yet.
python scripts/provenance_gate.py vancouver --server https://sidewalk-vancouver.cs.washington.edu --fetch-only
python scripts/detect_from_store.py --run-dir runs/vancouver --store <store> --server <server> \
    --labels runs/vancouver/provenance_gate/raw_labels.geojson --labels-user <ai user_id> \
    --sample-unlabeled 300 --seed 56 [--metadata-only] [--batch-size 4]
# ...--batch-size N (reinfer.py too; issue #2) runs N panos per forward pass; the pass ends
# with a `detector:` line (mean batch fill, seconds in forward, panos/s) to measure it by.
python scripts/provenance_gate.py vancouver
python scripts/provenance_gate.py vancouver --draw-control     # optional Arm Z, then:
python scripts/reinfer.py runs/vancouver --ids runs/vancouver/provenance_gate/control_ids.txt \
    --out runs/vancouver/control_zoom3.jsonl
python scripts/provenance_gate.py vancouver --control runs/vancouver/control_zoom3.jsonl --rule pixel-96
python scripts/harvest_depth.py runs/vancouver --from-store <store> [--check-store-frame 5]
```

## AI-vs-crowd agree rate

```bash
# AI-vs-crowd AGREE RATE (issue #31 goal 2; write-up in docs/agree-rate-gainesville.md).
# Compares the run's detections with the city's CROWD CurbRamp labels — read-only, nothing
# is submitted (an AI label on the server would contaminate the very baseline). Three GETs
# (rawLabels CurbRamp + NoCurbRamp, regions) are pulled ONCE into runs/<city>/agree_rate/
# and are the frozen snapshot: reused on every re-run, --refresh makes a NEW snapshot and
# every number moves with it. Two frames: PANO (the headline — crowd pixel vs detection on
# the SAME pano, RampNet's matcher geometry: x*1024/y*512, seam wrap, radius 0.022; no
# camera height, no placement error) and WORLD (a bound — PS's lat/lng vs fused sites, two
# independent placement errors, with a displaced-label chance floor printed beside it).
# Every AI figure at OPERATIONAL (0.30) + BENCHMARK (0.55); world at 2.6 m + per-pano.
# Regions come from the snapshot's completion_rate: partial regions count for crowd -> AI
# but never for AI -> crowd. Validation buckets use HUMAN votes only — the feed's `correct`
# is dominated by PS's own AI validator and is reported apart. Per-label rows key on
# `label_uid` (<city>:<label_id>). Stdlib + shapely (no pandas), so its tests run in CI.
# With ../RampNet/benchmark/<city> present it also adjudicates the disagreements against
# RampNet GT. --run-dir reads the run in place (read-only); outputs always go to THIS
# checkout's runs/<city>/agree_rate/. report.md + CSVs are git-tracked, the geojson not.
# Quote the pano headline WITH its matcher: one-to-one moves it ~6 pts vs any-detection
# (both are printed, plus the "shadowed" count). The per-pano ablation needs the untracked
# depth/index.csv beside results.jsonl; the report prints how many panos had a height.
python scripts/agree_rate.py gainesville --server https://sidewalk-gainesville.cs.washington.edu
```

## Server-side read (validation precision, auditor agreement)

```bash
# SERVER-SIDE READ (PR #119; write-up in docs/server-agree-check.md). Server feeds only, no run
# dir: (1) human-validation precision of the live AI CurbRamp labels, per label AND per cluster
# (the served clusters omit labels already marked incorrect, so labels are put back within 7.5 m
# first, approximating the server's complete linkage; never quote the served-cluster rate), and
# by tier with --results (joined on send_to_ps's pixel); (2) one auditor (--human auto = the human with the most CurbRamp
# labels) vs the AI. Headline = one-to-one vs AI clusters, quoted with its chance floor; any-
# cluster is coverage, never the headline. Validations come through PS's queue, not a random
# sample. Pulls cached in runs/<city>/server_agree/ (Laurens's tracked); as-served pulls go
# only to the untracked as_served/ subdir, the tracked copy is always redacted, and an unredacted
# pull at a tracked path is refused.
python scripts/server_agree_check.py laurens --human auto --results runs/laurens/results.raw.jsonl
```

## Label-frame beta

```bash
# LABEL-FRAME BETA (issue #113). GSV equirects are rig-frame, so a detection's row is in the
# image's frame while a human PS label's pano_y is off by beta x T(b), T = pitch cos b +
# roll sin b (streetlevel's sign). Fits beta from labels paired with the nearest detection on
# the SAME pano (3 deg of bearing, 4/6/10/15 deg of elevation; pano-clustered SEs; rig-masked).
# The elevation window is a BIAS, not just noise: centred on diff 0 it truncates large shifts
# and reads low (6 deg: pool 0.863), so every fit also runs centred on T (reads high) and the
# two bracket beta; they meet by 10 deg, which is the headline. `run` pulls the city's labels
# once (read-only GET, cached, --refresh re-pulls) and reads pose + detections from
# results.jsonl; it REFUSES when > 1% of labels sit within 1 px of a stored detection (an AI
# account's labels pair with themselves at diff 0) unless --exclude-user names it. `pool` reads
# sidewalk-panorama-tools' vouched pool + pose scan (pinned by commit + sha256 in POOL_INPUTS)
# and a detections file made over those store panos; it is rebuilt end to end by `pool-ids`
# (the sorted pano list, reproduces pool_ids.txt byte for byte) -> `pool-detect` (GPU, ~2.8 h
# on the A40; writes model + commit + ids sha256 into .meta.json) -> `pool`. pool_ids.txt, the
# detections file, its meta, the original runner and its log are tracked under
# runs/_pooled/label_frame_beta/; the API pull and the pano-tools CSVs are not.
# Results (10 deg): Gainesville 0.951 (SE 0.022); pool 0.902, legacy 0.881 / mid 0.909 /
# post179 0.943. The era gap is the POSE RECORD's, not the label era's: on the 3,518 pairs whose
# pano has both an XML and an npz pose, beta is 0.884 under XML and 0.953 under npz. Pose error
# attenuates beta, so all of these are lower bounds. NOT a placement coefficient (#116).
python scripts/label_frame_beta.py run gainesville --server https://sidewalk-gainesville.cs.washington.edu
python scripts/label_frame_beta.py pool-ids --pool <tilt-jm-pool.csv.gz> --pose <tilt-pose-jm.csv.gz> \
    --out runs/_pooled/label_frame_beta/pool_ids.txt
python scripts/label_frame_beta.py pool-detect --ids runs/_pooled/label_frame_beta/pool_ids.txt \
    --store /projects/makeabilitylab/sidewalk_panos/Panoramas --out <pool_detections.jsonl>
python scripts/label_frame_beta.py pool --pool <tilt-jm-pool.csv.gz> --pose <tilt-pose-jm.csv.gz> \
    --detections runs/_pooled/label_frame_beta/pool_detections.jsonl   # -> runs/_pooled/label_frame_beta/
```

## Mined-positive precision

```bash
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
# writes runs/<city>/mined_precision/{report.md,candidates.csv} (at 2.6 m; any other
# height writes mined_precision_<frame>/, e.g. _auto or _h2.20, like eval_ps_clustering;
# a pooled run over mixed heights names each, _h2.60+h2.20+...; a --placement run adds
# _placed-<label or placement file stem>) — one CSV row per
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
# --camera-height takes one value for all, or one per city in the order named; a value may
# also be per-pano / auto (#56, the shared resolver), and the report records the mode.
python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo
python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo \
    --camera-height 2.6 2.2 2.2 2.2 2.2   # richmond has no measured height; GSV does
# STEP 2 (RampNet#158, 2026-09-29; docs/mined-precision.md): the same check under
# per-rig/per-pano and auto heights on inputs frozen to step 1 (the 2026-09-21 gap fills
# are cut off; hashes in docs/figures/mined-precision/data/inputs.json). GSV's point
# estimates move from drop into visibility on both denominators, NOT decisively (each CI
# crosses a band edge); richmond cannot move (per-rig applies 2.6 m, #89).
# mined_precision_compare.py lays several --out dirs side by side (per city, range band,
# pooled, pooled GSV) from their candidates.csv, with mined_precision's own tallies.
python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo \
    --camera-height per-rig per-pano per-pano per-pano per-pano --out /tmp/mp/perrig_perpano
python scripts/mined_precision_compare.py --cities richmond paterson bend gainesville \
    sao_paulo --mapillary richmond --arm a=/tmp/mp/h2.6 --arm b=/tmp/mp/perrig_perpano
# PHASE 2 (image-based placement; docs/mined-precision.md). --emit-sources writes, per
# candidate, the source view the pre-registered SOURCE RULE picks (nearest member camera;
# nothing GT-derived). RampNet's scripts/analysis/mined_placement_158.py runs a #48
# placement arm on those pairs; --placement FILE then adjudicates the placed pixel on the
# SAME candidates (paired), and --paired-base in the compare tool gives fixed/broken counts.
# Under the pre-registered 5 m world match, roma_local moves 9 richmond false positives
# into already_detected (9 / 0) - but that does NOT show it places the SITE's ramp: the
# post hoc attribution below finds 6 of the 9 land on a detection of ANOTHER multi-pano
# site 5.4-11 m away (>= 2 demonstrably a different ramp), and on GSV it turns 7 true hard
# positives into already_detected. Nothing helps on GSV. See docs/mined-precision.md.
python scripts/mined_precision.py richmond paterson bend gainesville sao_paulo \
    --camera-height per-rig per-pano per-pano per-pano per-pano --out /tmp/mp/roma_local \
    --placement docs/figures/mined-precision/data/placement/roma_local.jsonl
# POST HOC (after review): which fused site does each changed candidate's adjudicating
# detection belong to? own site (impossible by construction) / another multi-pano site /
# singleton / none, plus a strict all-mined sensitivity; --verify refuses unless every
# recomputed bucket equals the committed run's.
python scripts/mined_placement_attribution.py richmond paterson bend gainesville sao_paulo \
    --runs-root $FROZEN --camera-height per-rig per-pano per-pano per-pano per-pano \
    --arm roma_local=docs/figures/mined-precision/data/placement/roma_local.jsonl \
    --verify docs/figures/mined-precision/data/frozen --mapillary richmond
# STEP 3 (peak-anchored targets; docs/mined-precision.md). floor_infer_archive.py re-runs
# RampNet at the 0.1 storage floor on a pinned run's ARCHIVED panos (pano blocks untouched)
# and gates it (--check: >= 95% of panos reproduce their >= 0.55 detections; bend FAILED,
# 0.70). peak_anchor.py turns an anchor into a --placement file: emit the strongest
# [0.1, 0.55) peak within the 0.022 benchmark radius, or not at all (emit: false leaves both
# denominators). --own-site is a secondary read (other_site = closer to another fused site).
python scripts/floor_infer_archive.py --results $A/richmond/results.jsonl --panos $A/richmond/panos \
    --ids docs/figures/mined-precision/data/step3/richmond_ids.txt --out richmond.floor.jsonl
python scripts/peak_anchor.py --sources-root docs/figures/mined-precision/data/frozen/perrig_perpano \
    --cities richmond paterson gainesville sao_paulo --runs-root $FROZEN \
    --peaks richmond=docs/figures/mined-precision/data/step3/richmond.floor.jsonl --out /tmp/peak_flat.jsonl
```

## Site explorer

```bash
# Eyeball the fusion: one HTML card per site with a crop from every member view,
# a plan view (cameras/rays/error ellipses/fused 1-sigma) and the RampNet verdict.
# Crops are cut on makelab2's native-res archive and pulled back as a tarball, then
# cached under runs/<city>/explorer/crops/ (re-renders are offline).
python scripts/site_explorer.py richmond              # 40 GT-seen sites
python scripts/site_explorer.py sao_paulo --select fp # only the false positives
python scripts/site_explorer.py richmond --select fragment  # over-split suspects
python scripts/site_explorer.py richmond --inline     # one shareable file
# ...--local-panos <dir> cuts crops locally instead (no SSH; e.g. a RampNet bundle).
```

## Mapillary rig tilt study

```bash
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
```

## Pose backfill

```bash
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
```

## DEM road grade

```bash
# DEM ROAD GRADE (issue #51) -- the independent arm of the #42 shuffled-grade control. USGS 3DEP
# float32 GeoTIFF tiles (network: elevation.nationalmap.gov only; tiles cached under
# runs/<city>/dem/tiles/, hashed in the tracked tiles.json) -> runs/<city>/dem/grades.csv (SfM,
# SfM-fitted, DEM 2-point, DEM-fitted grade per pano; untracked, sha256 in the tracked report.md).
# `fuse_sites.py --grade-source {sfm,sfm-smoothed,dem}` (default sfm; only matters under
# --apply-pose road) reads it. VERDICT: NEGATIVE -- the pre-registered rule (eval_sites.dem_verdict)
# fails (i)+(ii), so AUTO_ROAD_SOURCES stays (); DEM and SfM grades agree (r 0.80-0.83 in hilly
# cities) and land within 0.23 m in fusion. grades.csv is BOUND to its results file by grades.json
# (sha256): fusing another file (e.g. laurens results.raw.jsonl) or a stale/truncated CSV refuses. The service returns SQUARE degree-pixels whatever size
# is asked, so the grid is square in degrees and every tile's georeference is checked.
python scripts/dem_grade.py richmond clovis morgantown annapolis laurens   # --verify re-hashes
python scripts/mapillary_tilt.py precondition --grade-source dem           # eight arms + verdict
```

## GSV ground plane study

```bash
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
```

## Depth at the detection study

```bash
# DEPTH AT THE DETECTION (issue #47 step 1) -- a STUDY, not production: what GSV's depth model puts
# under every stored detection's pixel (ground / another floor plane / horizontal non-floor / wall /
# no plane), the height of a secondary floor plane above the LOCAL road (the first different floor
# plane down the same payload column; the dominant-plane offset is extrapolation, never a height),
# and the plane's range against the flat raycast (a third range instrument, #40/#101). IMAGE frame:
# the lookup is depth._plane_at + the exact ray (depth._direction_continuous) -- NEVER depth.depth_at
# (raster frame, snapped ray). Only `measured` payloads enter the headline; anything joined to GT is
# tier 0.55 through eval_sites.judged_gt_panos. Pre-registered verdict() (rules (i) surface on True
# detections, (ii) free FP signal, flagged underpowered) -- docs/depth-at-detection-study.md.
# Measured 2026-09-26: "no plane" under a detection NEVER happens (0 of 114,932); (i) SUPPORTED
# (0.919 of True on surface) but (ii) NOT SUPPORTED (False sits on surface as often; not underpowered),
# so depth is no FP filter. 10-27% of "floor" hits are an EXACTLY level 2.5 m secondary plane --
# Google's stand-in again, now as a secondary plane. Set aside the way the registration sets stand-ins
# aside (out of the denominator) True surface is 0.930, still SUPPORTED; counted as failures, 0.782 --
# quote (i) with that qualifier. The stand-ins read ~0.13-0.15 m above the local road, but that is the
# road's TILT carried out to the hit (were the road level, the stand-in would sit ~0.14 m BELOW it), so
# no ramp plane measured here sits ~0.15 m up (real planes: ~0.03 m). offset_local always includes the
# reference's tilt extrapolated over the distance from where the column walk met it to the hit;
# level_floor.csv splits it. Follow-up (shipped): height_spread_m (#44's per-pano sigma) now leaves
# those secondary stand-ins out (depth.is_standin) -- see the stand-in command block above.
python scripts/depth_at_detection.py measure       # ~53k panos, multiprocess; --limit N smoke-tests
#   (to detections.limit.csv, never over the full file); --summaries-only refuses a row-count mismatch
python scripts/depth_at_detection.py gt            # reads ../RampNet/benchmark
python scripts/depth_at_detection.py verdict
python scripts/depth_at_detection.py figures       # also copies aggregates to docs/figures/depth-at-detection/data/
```

## Aerial sidewalks (Tile2Net) study

```bash
# AERIAL SIDEWALKS (issue #104) -- a STUDY, not production: Tile2Net (VIDA-NYU/tile2net, BSD-3;
# Hosseini et al., CEUS 2023) sidewalk/crosswalk/road/footpath polygons from aerial tiles, scored
# against the run under rules pre-registered on #104 (q1/q3/q4_verdict in the script); write-up
# in docs/aerial-sidewalk-study.md. Tile2Net runs in its OWN venv on makelab2 (never a pipeline
# dependency) with `--deterministic` -- without it cudnn benchmark probing peaks at ~34 GB VRAM on
# the shared A40 (1.4 GB with it). Paterson uses Tile2Net's `nj` source (2020 orthos); Oregon is
# NOT supported upstream (and OSIP's cert expired 2026-09-26), so Bend's tiles are the City of
# Bend 2019 orthos fetched by scripts/aerial_fetch_tiles.py (stdlib, TLS verified) and fed through
# `tile2net generate --input <dir>/z/x/y.png`. Outputs archived on makelab2 under
# /projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/aerial/; the GeoJSON exports
# (polygons.geojson, network.geojson) are copied untracked into runs/<city>/aerial/, bound by
# sha256 in the tracked tile2net.json (source, date, URL, terms, zoom, commit, checkpoint sha256s,
# Tile2Net's tile grid, runtime, tile count). load_aerial REFUSES a recorded network.geojson that is
# missing or altered (without it Paterson's Q3 anchors silently drop 5,815 -> 3,853).
# scripts/t2n_convert.py is the GeoParquet -> GeoJSON converter (runs in the tile2net venv; its
# 7-decimal rounding is what polygons_geojson_sha256 binds).
# --run-root reads results.jsonl / depth index / inventory_oracle / ps_streets.geojson from another
# checkout as data; outputs always go to THIS checkout's runs/<city>/aerial/ + runs/_summary/aerial/.
# Needs numpy + shapely (+ pandas/scipy/haversine for `anchors`, Pillow for --gallery, matplotlib
# for `figures`). Bend is a RampNet TRAINING city: every Bend GT/detection number says so.
python scripts/aerial_sidewalks.py tiles bend --tiles-csv <copy of archive tiles.csv>  # tile audit
python scripts/aerial_sidewalks.py gt bend paterson --run-root <runs>        # Q1 mask vs GT + Bend inventory
python scripts/aerial_sidewalks.py project bend paterson --run-root <runs> --gallery   # Q2 (+ untracked gallery)
python scripts/aerial_sidewalks.py anchors bend paterson --run-root <runs>   # Q3 corner anchors vs frag
python scripts/aerial_sidewalks.py precision bend paterson --run-root <runs> # Q4 off-surface by verdict
python scripts/aerial_sidewalks.py verdict && python scripts/aerial_sidewalks.py figures
# VERDICT (2026-09-28): Q1 NOT USABLE (Bend inventory 0.703 within 2 m vs the 0.90 bar -- polygons
# are MISSING, not misplaced; Paterson GT 0.913); Q2 median 3.4 px (Paterson, auto) against a 5.2 px
# displaced-mark chance floor; Q3 NO (snapping merges nothing, merging at crosswalk anchors fuses
# dual ramps); Q4 NOT ESTABLISHED (17 False pooled < 30; Bend's mask misses 30% of true ramps).
# Bend's network step crashed (centerline TooFewRidgesError) after writing polygons.
# REVIEW FIXES (PR #109): Q2's nearest-edge metric favours LOWER heights for any point (denser
# edges), so `auto` beating 2.6 m is mostly that: of Paterson's 1.17 px lead only 0.34 px survives
# the displaced-mark chance row, and none at the box centres. It is NOT evidence for #79's heights;
# compare heights on `p50_minus_displaced` in q2.csv. Q1's missing Bend polygons are mostly NEWER
# than the 2019 flight: 55% of ramps > 2 m away were installed 2020+ (inventory InstallDate), the
# 25 Tile2Net inputs with no polygon at all hold 278 ramps (274 installed after 2019), and none of
# the 1,565 blank tiles (tile_audit.json) is under a ramp. Pre-2019 ramps still read 0.859 < 0.90.
```

## Footway: segmenter vs depth study

```bash
# FOOTWAY: SEGMENTER vs DEPTH (issue #47 step 2) -- a STUDY, not production; docs/footway-depth-study.md.
# 416 judged GSV benchmark panos with a `measured` payload; Mask2Former Swin-L Mapillary Vistas (pinned
# revision; processor resize OFF; torch + transformers deliberately NOT in requirements.txt, only `segment`
# needs them) on 16 PERSPECTIVE TILES per pano (90 deg, 1024 px, 8 headings x pitch 0/-35), voted back onto
# a 1024x512 image-frame grid. Depth classes, the plane lookup and offset_local are IMPORTED from
# depth_at_detection.py. Tiles + label maps stay in the untracked runs/_pooled/footway/work/ (sha256 in
# masks_manifest.json). Measured 2026-09-30, corrected on review of #124 the same day: below the horizon
# depth puts a floor almost EVERYWHERE (98% within 25 m), so P(WALK|ROAD | depth surface) 0.90 is a BASE
# RATE (marginal 0.88; a 180-deg-rotated / mirrored depth null gives 0.89) -- always quote (A) beside its
# null. Signal lives in the walls: P(STRUCTURE | depth wall) 0.66 vs 0.42/0.47 null; kappa 0.25 vs 0.17/0.19.
# "Depth draws the ground through objects" holds against the scrambled-geometry NULL (95% of OBJECT pixels
# on a floor vs 94% null: no object-shaped holes; do not argue it from a lift over the nadir-heavy marginal).
# The segmenter class at the peak pixel is NO FP filter (False 2/42 vs True 43/770). GSV depth shows no
# ~0.15 m curb step, at most a few cm (0.022 m above the LOCAL reference plane; 0.042 m road-referenced,
# exploratory). Trap: the 2048 px direct arm's Curb Cut loss was RESOLUTION; at 4096 px the direct equirect
# keeps Curb Cut but labels the nadir FILL under the car SKY -- a city/rig-dependent failure (-80..-70 deg:
# Bend 0.000, Paterson 0.110, Sao Paulo 0.097, Gainesville 0.247). Tile (or full-res + mask the nadir).
# Inputs not in git (benchmark JPEGs, depth payloads, results.jsonl) live in the makelab2 run archive;
# sample.csv / inputs.json hold their sha256. `check-numbers` re-reads every quoted number (exit 1 on drift).
python scripts/footway_segmentation.py sample --run-root <runs> --benchmark-root ../RampNet/benchmark
python scripts/footway_segmentation.py tiles --benchmark-root ../RampNet/benchmark --workers 8
python scripts/footway_segmentation.py tiles --benchmark-root ../RampNet/benchmark --direct-width 4096
python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/tiles     --out runs/_pooled/footway/work/tile_labels --fp16 --batch-size 2          # GPU; ~19 min on a 3070
python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/direct     --out runs/_pooled/footway/work/direct_labels --target 1024x512 --fp16 --batch-size 1
python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/direct4096     --out runs/_pooled/footway/work/direct4096_labels --target 1024x512 --fp16 --batch-size 1   # 5.7 GiB
python scripts/footway_segmentation.py stitch  --run-root <runs> --benchmark-root ../RampNet/benchmark
python scripts/footway_segmentation.py compare --run-root <runs> --benchmark-root ../RampNet/benchmark
python scripts/footway_segmentation.py examples --run-root <runs> --benchmark-root ../RampNet/benchmark  # example panels
python scripts/footway_segmentation.py check-numbers   # every quoted number vs its committed file; exit 1 on drift
python scripts/footway_segmentation.py figures   # COMMITTED files only: byte-reproducible figures + data/numbers.csv
#   (every quoted number re-read from its committed file and checked) -- no GPU, network or work/ needed
```

## Position check and reposition

```bash
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
```

## Post-submission coverage check

```bash
# POST-SUBMISSION COVERAGE CHECK (issue #46). Read-only (GET only): does every pano we sent
# labels to have its backup image on the server? The EXPECTED set comes from the submission
# records + sidecars (position_check.campaigns_for, each campaign replayed through
# transform_record at the range it sent and its recorded rig mask), never from results.jsonl
# alone, and every record in the run dir is unioned (Richmond = three files, 4,613 panos).
# The range a BASE entry sent is not its recorded min_confidence once it has bands: when a
# band completes, send_to_ps.write_submission_record moves the base min_confidence DOWN to
# the band's floor and adds the band's labels to the base labels_submitted, so the base is
# replayed from the highest band ceiling (a partial band has moved nothing: the recorded
# value stands). Anyone changing write_submission_record must keep that rule in step. Every
# replay is checked against its record (band = its replay; base = its replay + its complete
# bands) and a mismatch is exit 2. Backup state is `has_backup` from /labels/all joined to
# rawLabels on label_id; `false` means UNCONFIRMED (NULL reads as false), never absent.
# /backupImage/<id>/metadata also 404s when the pano row has a null camera_pitch, so its
# probe (sequential, spaced, no redirects, JSON 200 required, 429 honoured, --max-confirm) is
# decisive only for pose-pushed panos: 404 + pitch = missing, 404 + null pitch = unconfirmed.
# No live AI label left = `retired` (never a gap) only when the pano row has `has_labels`;
# otherwise `never_landed`, which is exit 2, as are unjoinable AI labels and expected panos
# with no pano row. Exit 0 clean, 1 missing (or unconfirmed with --strict) and nothing else,
# 2 undetermined (incl. usage errors and crashes; a run that stops before classifying still
# rewrites report.md as exit 2). --archive <index.csv|dir> lists the archived fallback files
# (fallback.csv); placing them is out of scope (SW#4865). Pulls are FRESH every run (a re-run
# after the scraper's night must not read yesterday's flags), staged so a failed GET leaves
# the old set whole, and cached in runs/<city>/coverage/ (untracked); --reuse-pulls re-reads
# them for an offline re-render, refusing an edited file or a set fetched > 10 min apart.
# report.md + CSVs tracked.
python scripts/coverage_check.py runs/richmond/results.jsonl --server https://sidewalk-richmond.cs.washington.edu
python scripts/coverage_check.py runs/laurens/results.raw.jsonl --server https://sidewalk-laurens.cs.washington.edu
```

## Tests

```bash
# Run the tests (no GPU/network/model; light deps via requirements-test.txt)
pytest
```

## Submit to Project Sidewalk

```bash
# Submit a produced JSONL file to a Project Sidewalk endpoint
python send_to_ps.py runs/bend/results.jsonl --dry-run
python send_to_ps.py runs/bend/results.jsonl --endpoint https://<server>/ai/submitLabelsOnPano
```
