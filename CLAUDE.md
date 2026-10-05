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
# ...every tile pass is saved to runs/<name>/scan.json; --reuse-scan (OFF by default —
# coverage churns, so a reused scan misses newer panos) lets the follow-up run skip it
python main.py example_geojson/bend.geojson --name bend --reuse-scan

# Run the labeler over an area; all per-area state goes to runs/<name>/
# (--name defaults to the geojson filename stem)
python main.py example_geojson/bend.geojson --name bend

# Same pipeline on Mapillary 360 imagery (needs MAPILLARY_ACCESS_TOKEN — a client
# token from mapillary.com/dashboard/developers — in the env or in gitignored ./.env)
python main.py example_geojson/richmond.geojson --name richmond --source mapillary

# ...or on Panoramax (federated open imagery, no token; coverage is mostly France today)
python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax
# PANORAMAX MEASURED (issue #57 part 1; docs/panoramax-bayonne.md, corrected after two PR #125
# reviews). Bayonne ran in full at --thin-spacing 10 (rule: 10 m if the scan-only estimate > 16 h;
# it printed 21.1 h at 5 m; the manifest does not record the spacing -- #126): 73,161 in-area
# pictures -> 28,634 thinned -> 28,524 processed, 0 failed, 6.7 h on the A40. 104 of the 110
# skips are GoPro MAX2 uploads whose `hd` image is a vertically CROPPED equirect (declared
# 7680x3840, served 7680x2940) -- correctly skipped; 3 more are unexplained (#127: truncated body
# or transient 404). 0.135 detections per pano at 0.55 -- INSIDE the Mapillary range (Clovis
# 0.123 ... Richmond 1.048), worth GT, not anomalous. The municipal rig burns a white logo band
# from y 0.791 (dip 52.4 deg), inside the nadir mask (2 of 10,011 stored detections there).
# ERROR MODEL: the leave-one-out residual gained a normalized form (`chi2` = r'S^-1 r, S = the
# view's cov_en + the held-out solution's covariance; `seq_mates` / `pose_group` / rig
# breakdowns; `--fit-sigma-pitch TARGET` solves sigma_pitch and sigma_gps for a chi2/dof
# target). MAPILLARY_ERRORS reads 0.18-0.43 on the five Mapillary runs, so the rule was
# re-anchored on Richmond (0.269) before Bayonne was read. The amendment's stated mechanism
# (shared same-sequence error cancels) is NOT ESTABLISHED: Richmond reads the opposite way.
# What is measured: its 3 m sigma_gps is ~2.7x Richmond's leave-one-out-calibrated between-view
# scatter (1.11 m). Bayonne: 0.580 [0.539, 0.618] (median form 0.582), m p50 3.66 -> TOO TIGHT
# under the amended rule, but MARGINAL (bar 0.538) and confounded: Mapillary GoPro Max
# populations read 0.18-0.50 (Richmond's GoPro Max views 0.499, Laurens 0.431, Morgantown 0.177),
# and Bayonne sites are small (3 views / 2 sequences vs Richmond's 7 / 4, 10 m thinning), so it
# cannot be separated from a single-consumer-rig or site-size effect. The excess is ACROSS the ray
# (vs Richmond's GoPro Max views re-solved alone: along-ray +0.5..-0.5 m, cross-ray +0.6-1.0 m), not
# pitch: no sigma_pitch <= 15 deg fixes it. Calibrated sigma_gps: Bayonne 2.16 m vs Richmond 1.11,
# Richmond GoPro Max 1.81, Laurens 1.74, Morgantown 0.80. NOTHING adopted -- error_model_for(
# 'panoramax') stays MAPILLARY_ERRORS until GT (eval_sites) and a 5 m run. Pose: pers:pitch/roll
# LOOSENS the same-site spread in every sign convention (real-tilt p90 7.58 -> 9.45-11.27 m at
# 0.55), so Panoramax stays flat. GT: the RampNet bundle (125 panos, reconcile 1:1) awaits review.
python scripts/run_census.py runs/bayonne --out docs/figures/panoramax-bayonne/data/census --band-y 0.79
python scripts/reprojection_residual.py bayonne richmond --camera-height-m 2.6 --refuse \
    --fit-sigma-pitch 0.269 --benchmark-root /nonexistent   # GT-free; --min-confidence 0.3 too
python scripts/panoramax_bayonne_figures.py data && python scripts/panoramax_bayonne_figures.py figures
#   data: run files -> data/fig*.csv (site bootstraps, seed 57; ~45 min, no GPU/network; --only
#   for parts); skips / examples: network; crops: from the RampNet bundle; figures: committed
#   data only, byte-reproducible (PNG + LF SVG)

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

# Score fusion against RampNet GT in world space: world P/R, the union-recall
# decomposition, stage-4 promotion calibration, vintage + match-radius ablations.
# Reads ../RampNet/benchmark/<city>/{verdicts.json,records.jsonl} as data (no
# RampNet code import); writes runs/<city>/fusion_eval/{report.md,*.csv}.
python scripts/eval_sites.py paterson
python scripts/eval_sites.py paterson --vintage-ablation

# HEATMAP GRID (issue #111; docs/heatmap-grid.md). RampNet's heatmap is an 8x bilinear upsample
# of a stride-32 map, so every detection sits on residue 3/4 of an 8-cell block. `grid` is the
# census per run and tier; `sigma` re-fuses at the benchmark tier with sigma_peak_px 1.0 vs 2.31
# (one coarse cell's uniform quantization) at 2.6 m and auto and applies the pre-registered rule
# (verdict: KEEP 1.0). Writes docs/figures/heatmap-grid/data/. No GPU, no network.
python scripts/heatmap_grid.py grid paterson bend gainesville sao_paulo richmond
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
# SEAM BAND (#130; docs/seam-band-130.md). OPT-IN, default exclude: the peak finder drops every
# peak within 10 heatmap px of the heatmap edge (skimage's exclude_border default), so the 20
# columns at the 360-degree seam -- coarse columns 0 and 127 of the exact x8 upsample, 5.6 deg of
# azimuth -- never yield a detection. `main.py --border keep`
# / `reinfer.py --border keep` use RampNet's rule instead (exclude_border=False, NO NMS across the
# seam, so a straddling ramp can give two peaks). `--border wrap` (#130 follow-up) is keep plus NMS
# wrapped across the seam (circular pad by 10 px), this repo's own rule: one peak per straddling
# ramp; bound and guarded like keep, a frame of its own (keep/wrap mixes are refused too);
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
# Z 0.987 passes, P 0.029 fails; docs/heatmap-grid.md. harvest_depth.py --from-store indexes
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
