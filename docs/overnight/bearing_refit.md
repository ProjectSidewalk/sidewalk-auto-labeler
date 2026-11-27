# Bearing-only site placement vs the flat raycast (overnight, `bearing_refit`)

## Pre-registration (written 2026-11-26 before any real-data scoring; not edited after)

**Question.** Open imagery (Mapillary, Panoramax) has no depth and no validated camera-height
estimator (#53/#89), so production places a fused site by a GLS refit of flat-ground raycasts at
an assumed 2.6 m. Does placing multi-view sites by bearing-only triangulation (azimuths only,
camera-height-free horizontally) beat that position, and is it robust to a wrong height?

**Site set (one for every arm).** Association frozen from `fuse_sites.fuse` at 2.6 m, tier
`BENCHMARK_CONFIDENCE` 0.55, `apply_pose=off`, `mask_rig=True` (production's rig mask). A *view*
is an operational (>= 0.55) refit member. Sites with >= 3 views.

**Arms.**
- **A** (production-style): GLS of the views' flat 2.6 m raycasts with fusion's own covariances
  (`reprojection_residual.solve_views`).
- **B** (bearing-only): weighted least-squares intersection of the views' horizontal rays
  (camera position + world bearing `heading + (x - 0.5) * 360`), minimising the weighted sum of
  squared perpendicular distances. Weights by IRLS: w_i = 1 / (r_i^2 sigma_theta^2 + sigma_gps^2),
  sigma_theta^2 = (sigma_peak_px * rad/px)^2 + sigma_heading^2 from the source's `geo.ErrorModel`,
  r_i = range along ray i to the current B estimate; unweighted start, 5 iterations. Nothing in B
  reads a camera height. B is **solvable** for a view set when its ray-angle spread (max over view
  pairs of asin|sin(b_i - b_j)|, deg) is >= the spread threshold and every range r_i lies in
  (1, 30) m (`fuse_sites.IMPLIED_RANGE_M`). **Primary spread threshold: 10 deg.** Keeps are
  reported at 0 / 5 / 10 / 15 / 20 / 30 deg.

**Headline metric (pano frame, height-free).** References are verdict-true detections in judged
benchmark panos (`eval_sites.judged_gt_panos`) whose detection is a view of a site in the set:
the reference pixel is the reviewer's box centre where `boxes.json` has one, else the model's own
peak. For each, the site is re-solved **without that pano** (leave-pano-out; >= 2 views remain) by
each arm and projected into the pano; **azimuth error** = wrapped difference between the bearing
from the pano's camera to the solved position and the reference column's bearing (deg). Unit set:
references whose leave-pano-out view set is B-solvable at the threshold; A is scored on exactly
those.

**Decision rule 1 (B beats A).** Per city, D = median|az_A| - median|az_B| over the unit set,
with a 95% percentile site-cluster bootstrap (2000 resamples, clusters = site, seed 20261126).
B beats A in a city when D >= 0.25 deg AND the CI's lower bound > 0. **Overall: B beats A iff it
does in >= 2 of the 3 open-imagery cities** (richmond, laurens = `laurens_mapillary` split,
bayonne). paterson (GSV, `results.jsonl`) is reported as a reference and does not vote.

**Decision rule 2 (height robustness).** Wrong heights 2.0 and 3.2 m. A's shift: the same frozen
sites, every view re-raycast at the wrong height (uncapped, so no view drops), GLS re-solved;
shift = |A(h) - A(2.6)|. B's shift: the run **re-associated** at the wrong height, each 2.6 m site
matched to the wrong-height site sharing most of its views (>= 2 shared), B solved on that site's
views; shift = |B(h) - B(2.6)| (B on frozen sites moves exactly 0 by construction, so only
re-association can move it). Per (city, height): robust when median B shift < 0.25 x median A
shift, over sites with both shifts defined. **Overall: height-robust iff it holds in all six
open-imagery (city, height) cells.** Sites whose match is lost or not B-solvable are counted.

**Secondary (reported, no vote).**
- GT-free leave-one-view-out azimuth residual (deg) for A vs B on the same (site, held-out view)
  units (B-solvable remainder), with the same bootstrap.
- Row (elevation) error in heatmap px at an assumed 2.6 m for both arms (needs a height: not the
  headline).
- Results binned by the leave-out set's parallax (spread): [10,15), [15,30), [30,60), [60,90].
- Box-only subset (independent of the detector).
- Missed-mark references (the pano is not a site member, matched to a site within 5 m of the
  mark's 2.6 m raycast, one to one): this selection is made in A's frame and favours A; secondary.
- Survivorship: references whose leave-pano-out set is NOT B-solvable, and A's azimuth error there.
- Wrong-height A under re-association too, for completeness.

**Amendment A1 (2026-11-26, after the per-city runs printed only their reference COUNTS, before
any azimuth/shift number was read).** The brief names Bayonne as the 10 m-thinned run, but the
`runs/bayonne/results.jsonl` in the archive is the **5 m run** (50,409 panos; the worktree's
manifest records a 2026-10-06 `thin_spacing_m: 5` pass on top of the 10 m one, and
docs/panoramax-bayonne.md says runs/bayonne now holds it). The 10 m subset cannot be recovered
offline (scan.json is not here, and re-thinning the 5 m set is not the original 10 m set), so
Bayonne is scored on the 5 m run. Consequence: denser sites (more views, more parallax), which
favours B relative to what a 10 m run would show. Rules unchanged.

---

## Verdict

- **Rule 1 (B beats A on leave-pano-out azimuth error): FAIL, 1 of 3 open cities.**
  Richmond passes (D = +0.81 deg [0.36, 1.40]). Laurens points the same way but its CI
  crosses 0 (+1.30 [-0.52, 2.56], n = 76). **Bayonne goes the other way: B is worse**
  (-5.67 [-8.70, -0.25], n = 53). Paterson (GSV, no vote) points B's way, but its CI just
  touches 0 (+0.44 [-0.01, 1.20]).
- **Rule 2 (height robustness): FAIL, 3 of 6 open (city, height) cells.** Richmond passes
  at both heights and Laurens at 3.2 m. Laurens at 2.0 m (B/A 0.45) and Bayonne at both
  heights (0.33, 0.77) fail. On a fixed view set B moves exactly 0 under a height change,
  because no height enters it. All of B's shift comes from **re-association**: at the
  wrong height the frozen site's membership changes in 68-97% of sites. So what fails is the
  association's height dependence, transmitted through B, not B's geometry. A's frozen shift
  is 1.1-1.6 m (median) for a 0.6 m height error.
- **Reading.** Where camera positions are mutually consistent (Mapillary SfM in Richmond and
  Laurens, GSV), bearing-only triangulation agrees better with held-out views and with
  reviewer boxes, by about 0.4-1.3 deg of azimuth. Its median ranges match A's (B/A range ratio
  0.99-1.07), so the gain is the sub-metre cross-ray part. The GT-free leave-one-view-out
  says the same with tight CIs (Richmond +1.08 [0.96, 1.21], Laurens +2.24 [1.10, 3.00],
  Paterson +0.88 [0.82, 0.93]). **On Panoramax (raw GPS, 4 m catalog accuracy, integer-degree
  headings) B is worse on every metric**, GT-free included (-1.01 [-1.48, -0.53]). So bearing
  triangulation is not a source-agnostic fix for the unknown height. It should not replace
  A for open imagery as a rule, and certainly not for Panoramax.

## Method notes

- One site set per city: `fuse_sites.fuse` at 2.6 m, tier 0.55, `apply_pose=off`,
  `mask_rig=True`. Views are operational refit members (`reprojection_residual.views_from_site`),
  and sites need >= 3 views. A = `reprojection_residual.solve_views` (fusion's GLS). B = the
  IRLS ray intersection in `scripts/bearing_refit.py` (`bearing_solve`). Its bearing is
  `heading + (x - 0.5) * 360`, exactly the flat raycast's bearing, so the two arms share every
  bearing and differ only in how positions are combined.
- Reference matching for member refs needs no distance gate: the reference is a
  verdict-true detection of the judged pano, and that pano's own view is left out of both
  arms. Missed-mark refs reuse reprojection_residual's 5 m world match at 2.6 m (A's frame).
  They are secondary.
- Azimuth error is the bearing from the judged pano's camera to the solved position, minus
  the reference column's bearing (flat pose: a pixel column is an azimuth). Both arms use the
  same camera position, so a judged pano's own position or heading error enters them equally.
- Bootstrap: 2000 resamples of sites (clusters), seed 20261126, percentile 95% CI of
  median|az_A| - median|az_B|.
- Paterson read `results.jsonl` (34,427 panos; the gap-fill file was not added). Laurens read
  `results.jsonl` (4,495 panos; no bundle drift warnings). Bayonne read the 5 m run (Amendment A1).

## Results

All tables below are generated by `summarize` (copied from
`docs/overnight/data/bearing_refit/summary.md`). The order is: headline; stress; GT-free
leave-one-view-out; subsets and spread thresholds; parallax bins; survivorship; site keeps by
spread threshold; then the exploratory baseline/sequence table. Azimuths are |deg|, rows are
heatmap px (1024 x 512), shifts are m.

Rule 1: B beats A in 1 of 3 open cities -> FAIL
Rule 2: height-robust in 3 of 6 open (city, height) cells -> FAIL

| city | refs (sites, box) | med abs az A | med abs az B | D = A - B [95% CI] | p90 A / B | rule 1 | med row A / B px | med A-B dist m |
|---|---|---:|---:|---|---|---|---|---:|
| richmond | 183 (183, 175) | 3.29 | 2.48 | 0.81 [0.36, 1.40] | 11.17 / 11.67 | pass | 4.2 / 4.9 | 0.80 |
| laurens | 76 (70, 0) | 4.48 | 3.19 | 1.30 [-0.52, 2.56] | 19.20 / 21.91 | fail | 5.1 / 6.8 | 1.52 |
| bayonne | 53 (53, 0) | 9.57 | 15.24 | -5.67 [-8.70, -0.25] | 27.59 / 27.06 | fail | 7.6 / 9.7 | 1.37 |
| paterson | 194 (194, 64) | 2.41 | 1.97 | 0.44 [-0.01, 1.20] | 8.56 / 6.70 | fail | 3.2 / 3.7 | 0.69 |

| city | h m | sites | lost | changed | B unsolvable | med shift A frozen | med shift A re-assoc | med shift B re-assoc | B = 0 share | B/A | rule 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| richmond | 2 | 896 | 5 | 642 | 24 | 1.41 | 1.50 | 0.164 | 0.31 | 0.116 | pass |
| richmond | 3.2 | 867 | 47 | 767 | 11 | 1.22 | 1.40 | 0.272 | 0.13 | 0.223 | pass |
| laurens | 2 | 86 | 1 | 83 | 20 | 1.58 | 2.21 | 0.717 | 0.27 | 0.454 | fail |
| laurens | 3.2 | 105 | 1 | 71 | 1 | 1.42 | 1.57 | 0.174 | 0.33 | 0.122 | pass |
| bayonne | 2 | 602 | 2 | 421 | 77 | 1.32 | 1.79 | 0.439 | 0.43 | 0.332 | fail |
| bayonne | 3.2 | 588 | 54 | 455 | 39 | 1.19 | 1.67 | 0.919 | 0.29 | 0.773 | fail |
| paterson | 2 | 4489 | 58 | 3158 | 4 | 1.12 | 1.26 | 0.133 | 0.30 | 0.119 | pass |
| paterson | 3.2 | 3240 | 1299 | 3043 | 12 | 0.95 | 1.50 | 0.292 | 0.06 | 0.308 | fail |

| city | GT-free views (sites) | med abs az A | med abs az B | D [95% CI] | p90 A / B | unsolvable views (A med) |
|---|---|---:|---:|---|---|---|
| richmond | 7078 (941) | 3.13 | 2.05 | 1.08 [0.96, 1.21] | 12.11 / 11.16 | 157 (6.10) |
| laurens | 556 (109) | 5.23 | 2.98 | 2.24 [1.10, 3.00] | 19.62 / 20.39 | 29 (11.01) |
| bayonne | 3054 (767) | 8.85 | 9.86 | -1.01 [-1.48, -0.53] | 29.19 / 34.58 | 572 (12.47) |
| paterson | 19717 (4552) | 2.41 | 1.53 | 0.88 [0.82, 0.93] | 8.26 / 6.39 | 241 (4.75) |

| city | subset | spread >= | n | med A | med B | D [95% CI] |
|---|---|---:|---:|---:|---:|---|
| richmond | box_member | 10 | 175 | 3.29 | 2.48 | 0.81 [0.31, 1.40] |
| richmond | peak_member | 10 | 8 | 4.01 | 1.96 | 2.05 [-1.23, 5.34] |
| richmond | missed | 10 | 21 | 6.17 | 6.81 | -0.64 [-8.37, 2.97] |
| richmond | member | 0 | 184 | 3.34 | 2.48 | 0.86 [0.35, 1.39] |
| richmond | member | 5 | 184 | 3.34 | 2.48 | 0.86 [0.35, 1.39] |
| richmond | member | 10 | 183 | 3.29 | 2.48 | 0.81 [0.36, 1.40] |
| richmond | member | 15 | 182 | 3.25 | 2.47 | 0.77 [0.32, 1.32] |
| richmond | member | 20 | 181 | 3.21 | 2.47 | 0.74 [0.32, 1.32] |
| richmond | member | 30 | 181 | 3.21 | 2.47 | 0.74 [0.32, 1.32] |
| laurens | box_member | 10 | 0 | — | — | — [—, —] |
| laurens | peak_member | 10 | 76 | 4.48 | 3.19 | 1.30 [-0.52, 2.56] |
| laurens | missed | 10 | 35 | 4.85 | 3.78 | 1.07 [-2.24, 3.79] |
| laurens | member | 0 | 78 | 4.50 | 3.19 | 1.31 [-0.47, 2.70] |
| laurens | member | 5 | 77 | 4.49 | 3.35 | 1.14 [-0.67, 2.50] |
| laurens | member | 10 | 76 | 4.48 | 3.19 | 1.30 [-0.52, 2.56] |
| laurens | member | 15 | 75 | 4.47 | 3.02 | 1.45 [-0.47, 2.61] |
| laurens | member | 20 | 75 | 4.47 | 3.02 | 1.45 [-0.47, 2.61] |
| laurens | member | 30 | 73 | 4.47 | 3.35 | 1.12 [-0.70, 2.38] |
| bayonne | box_member | 10 | 0 | — | — | — [—, —] |
| bayonne | peak_member | 10 | 53 | 9.57 | 15.24 | -5.67 [-8.70, -0.25] |
| bayonne | missed | 10 | 24 | 12.02 | 9.36 | 2.67 [-3.76, 5.62] |
| bayonne | member | 0 | 54 | 8.84 | 15.09 | -6.26 [-8.60, -0.57] |
| bayonne | member | 5 | 54 | 8.84 | 15.09 | -6.26 [-8.60, -0.57] |
| bayonne | member | 10 | 53 | 9.57 | 15.24 | -5.67 [-8.70, -0.25] |
| bayonne | member | 15 | 53 | 9.57 | 15.24 | -5.67 [-8.70, -0.25] |
| bayonne | member | 20 | 52 | 9.84 | 15.09 | -5.26 [-8.16, -0.28] |
| bayonne | member | 30 | 48 | 8.84 | 15.09 | -6.26 [-8.51, -0.29] |
| paterson | box_member | 10 | 64 | 3.05 | 2.28 | 0.77 [-0.71, 1.66] |
| paterson | peak_member | 10 | 130 | 2.35 | 1.70 | 0.65 [0.02, 1.18] |
| paterson | missed | 10 | 19 | 2.52 | 1.98 | 0.54 [-1.72, 1.92] |
| paterson | member | 0 | 196 | 2.41 | 2.01 | 0.40 [-0.01, 1.18] |
| paterson | member | 5 | 195 | 2.41 | 2.00 | 0.41 [-0.01, 1.20] |
| paterson | member | 10 | 194 | 2.41 | 1.97 | 0.44 [-0.01, 1.20] |
| paterson | member | 15 | 192 | 2.41 | 1.92 | 0.49 [0.04, 1.16] |
| paterson | member | 20 | 192 | 2.41 | 1.92 | 0.49 [0.04, 1.16] |
| paterson | member | 30 | 191 | 2.41 | 1.90 | 0.50 [0.02, 1.21] |

| city | parallax bin deg | n | med A | med B |
|---|---|---:|---:|---:|
| richmond | [10,15) | 1 | 15.82 | 25.59 |
| richmond | [15,30) | 1 | 9.01 | 5.25 |
| richmond | [30,60) | 14 | 3.56 | 2.32 |
| richmond | [60,90) | 167 | 3.18 | 2.48 |
| laurens | [10,15) | 1 | 7.23 | 40.62 |
| laurens | [15,30) | 2 | 9.72 | 0.75 |
| laurens | [30,60) | 12 | 4.97 | 1.59 |
| laurens | [60,90) | 61 | 4.19 | 3.71 |
| bayonne | [10,15) | 0 | — | — |
| bayonne | [15,30) | 5 | 10.10 | 21.01 |
| bayonne | [30,60) | 9 | 7.17 | 16.47 |
| bayonne | [60,90) | 39 | 12.87 | 14.95 |
| paterson | [10,15) | 2 | 3.94 | 7.47 |
| paterson | [15,30) | 1 | 5.27 | 5.44 |
| paterson | [30,60) | 33 | 3.57 | 2.46 |
| paterson | [60,90) | 158 | 2.35 | 1.64 |

| city | member refs | unit | B unsolvable (low spread / bad range) | A med az on unsolvable (p90) | A med az on unit |
|---|---:|---:|---|---|---:|
| richmond | 190 | 183 | 7 (1 / 6) | 8.49 (15.02) | 3.29 |
| laurens | 80 | 76 | 4 (2 / 2) | 8.93 (15.46) | 4.48 |
| bayonne | 62 | 53 | 9 (2 / 7) | 4.84 (20.40) | 9.57 |
| paterson | 196 | 194 | 2 (2 / 0) | 9.78 (15.05) | 2.41 |

| city | spread >= | sites >= 3 views | spread ok | B solvable |
|---|---:|---:|---:|---:|
| richmond | 0 | 945 | 945 | 926 |
| richmond | 5 | 945 | 943 | 925 |
| richmond | 10 | 945 | 942 | 925 |
| richmond | 15 | 945 | 939 | 923 |
| richmond | 20 | 945 | 933 | 917 |
| richmond | 30 | 945 | 915 | 901 |
| laurens | 0 | 109 | 109 | 107 |
| laurens | 5 | 109 | 109 | 107 |
| laurens | 10 | 109 | 109 | 107 |
| laurens | 15 | 109 | 109 | 107 |
| laurens | 20 | 109 | 109 | 107 |
| laurens | 30 | 109 | 107 | 105 |
| bayonne | 0 | 793 | 793 | 684 |
| bayonne | 5 | 793 | 793 | 684 |
| bayonne | 10 | 793 | 783 | 681 |
| bayonne | 15 | 793 | 774 | 678 |
| bayonne | 20 | 793 | 763 | 669 |
| bayonne | 30 | 793 | 724 | 641 |
| paterson | 0 | 4552 | 4552 | 4551 |
| paterson | 5 | 4552 | 4552 | 4551 |
| paterson | 10 | 4552 | 4552 | 4551 |
| paterson | 15 | 4552 | 4548 | 4548 |
| paterson | 20 | 4552 | 4544 | 4544 |
| paterson | 30 | 4552 | 4531 | 4531 |

EXPLORATORY (post hoc):

| city | units | group | n | med A | med B | share B better | D [95% CI] |
|---|---|---|---:|---:|---:|---:|---|
| richmond | gtfree | baseline [0,5) m | 43 | 3.76 | 1.91 | 0.70 | 1.85 [0.46, 3.44] |
| richmond | gtfree | baseline [5,10) m | 179 | 5.92 | 3.31 | 0.58 | 2.61 [0.97, 3.56] |
| richmond | gtfree | baseline [10,20) m | 849 | 4.10 | 2.40 | 0.62 | 1.70 [1.19, 2.29] |
| richmond | gtfree | baseline [20,1e+09) m | 6007 | 2.99 | 1.99 | 0.62 | 1.00 [0.86, 1.14] |
| richmond | gtfree | 1 sequence | 380 | 6.15 | 2.56 | 0.71 | 3.59 [2.88, 4.56] |
| richmond | gtfree | >= 2 sequences | 6698 | 3.04 | 2.02 | 0.61 | 1.02 [0.88, 1.14] |
| richmond | gtfree | all | 7078 | 3.13 | 2.05 | 0.62 | 1.08 [0.96, 1.21] |
| richmond | refs_member | baseline [0,5) m | 1 | 9.01 | 5.25 | 1.00 | 3.77 [3.77, 3.77] |
| richmond | refs_member | baseline [5,10) m | 3 | 2.94 | 2.46 | 0.67 | 0.47 [-9.77, 0.47] |
| richmond | refs_member | baseline [10,20) m | 27 | 4.13 | 3.97 | 0.56 | 0.16 [-3.62, 3.45] |
| richmond | refs_member | baseline [20,1e+09) m | 152 | 3.17 | 2.42 | 0.59 | 0.75 [0.26, 1.45] |
| richmond | refs_member | 1 sequence | 7 | 9.01 | 7.75 | 0.57 | 1.27 [-2.80, 5.86] |
| richmond | refs_member | >= 2 sequences | 176 | 3.12 | 2.46 | 0.59 | 0.66 [0.26, 1.29] |
| richmond | refs_member | all | 183 | 3.29 | 2.48 | 0.59 | 0.81 [0.36, 1.40] |
| laurens | gtfree | baseline [0,5) m | 7 | 5.58 | 7.92 | 0.43 | -2.34 [-11.99, 7.83] |
| laurens | gtfree | baseline [5,10) m | 46 | 6.43 | 1.81 | 0.72 | 4.62 [1.40, 9.78] |
| laurens | gtfree | baseline [10,20) m | 155 | 5.70 | 4.53 | 0.57 | 1.17 [-0.64, 3.26] |
| laurens | gtfree | baseline [20,1e+09) m | 348 | 4.93 | 2.91 | 0.66 | 2.02 [0.95, 2.90] |
| laurens | gtfree | 1 sequence | 56 | 5.71 | 2.76 | 0.61 | 2.96 [0.37, 4.88] |
| laurens | gtfree | >= 2 sequences | 500 | 5.20 | 3.07 | 0.64 | 2.14 [0.98, 2.97] |
| laurens | gtfree | all | 556 | 5.23 | 2.98 | 0.64 | 2.24 [1.10, 3.00] |
| laurens | refs_member | baseline [0,5) m | 0 | — | — | — | — [—, —] |
| laurens | refs_member | baseline [5,10) m | 4 | 4.38 | 0.59 | 0.75 | 3.79 [-4.41, 4.47] |
| laurens | refs_member | baseline [10,20) m | 24 | 5.08 | 2.57 | 0.58 | 2.51 [-0.90, 4.62] |
| laurens | refs_member | baseline [20,1e+09) m | 48 | 4.25 | 3.71 | 0.60 | 0.54 [-1.35, 2.15] |
| laurens | refs_member | 1 sequence | 6 | 7.59 | 4.96 | 0.67 | 2.62 [-25.94, 14.29] |
| laurens | refs_member | >= 2 sequences | 70 | 4.38 | 2.98 | 0.60 | 1.40 [-0.70, 2.72] |
| laurens | refs_member | all | 76 | 4.48 | 3.19 | 0.61 | 1.30 [-0.52, 2.56] |
| bayonne | gtfree | baseline [0,5) m | 111 | 12.30 | 14.07 | 0.50 | -1.76 [-8.33, 2.85] |
| bayonne | gtfree | baseline [5,10) m | 268 | 9.01 | 10.27 | 0.41 | -1.25 [-4.53, 0.25] |
| bayonne | gtfree | baseline [10,20) m | 1032 | 9.64 | 10.65 | 0.43 | -1.02 [-2.10, -0.02] |
| bayonne | gtfree | baseline [20,1e+09) m | 1643 | 8.06 | 9.30 | 0.46 | -1.24 [-1.74, -0.68] |
| bayonne | gtfree | 1 sequence | 536 | 8.72 | 8.70 | 0.46 | 0.03 [-1.40, 1.34] |
| bayonne | gtfree | >= 2 sequences | 2518 | 8.89 | 9.99 | 0.44 | -1.10 [-1.67, -0.71] |
| bayonne | gtfree | all | 3054 | 8.85 | 9.86 | 0.45 | -1.01 [-1.48, -0.53] |
| bayonne | refs_member | baseline [0,5) m | 3 | 7.54 | 59.19 | 0.33 | -51.66 [-140.53, 0.20] |
| bayonne | refs_member | baseline [5,10) m | 4 | 17.96 | 14.81 | 0.75 | 3.16 [-1.15, 43.32] |
| bayonne | refs_member | baseline [10,20) m | 14 | 10.83 | 15.67 | 0.50 | -4.83 [-9.59, 4.84] |
| bayonne | refs_member | baseline [20,1e+09) m | 32 | 7.77 | 13.16 | 0.47 | -5.39 [-9.53, 0.63] |
| bayonne | refs_member | 1 sequence | 7 | 7.54 | 8.03 | 0.71 | -0.49 [-9.42, 3.58] |
| bayonne | refs_member | >= 2 sequences | 46 | 12.20 | 16.64 | 0.46 | -4.44 [-9.32, -0.57] |
| bayonne | refs_member | all | 53 | 9.57 | 15.24 | 0.49 | -5.67 [-8.70, -0.25] |
| paterson | gtfree | baseline [0,5) m | 21 | 4.01 | 1.54 | 0.62 | 2.46 [-2.83, 5.15] |
| paterson | gtfree | baseline [5,10) m | 838 | 3.66 | 1.99 | 0.64 | 1.67 [1.34, 2.11] |
| paterson | gtfree | baseline [10,20) m | 5634 | 2.92 | 1.59 | 0.65 | 1.34 [1.22, 1.45] |
| paterson | gtfree | baseline [20,1e+09) m | 13224 | 2.19 | 1.49 | 0.61 | 0.70 [0.64, 0.76] |
| paterson | gtfree | 1 sequence | 0 | — | — | — | — [—, —] |
| paterson | gtfree | >= 2 sequences | 0 | — | — | — | — [—, —] |
| paterson | gtfree | all | 19717 | 2.41 | 1.53 | 0.62 | 0.88 [0.82, 0.93] |
| paterson | refs_member | baseline [0,5) m | 1 | 0.44 | 6.37 | 0.00 | -5.93 [-5.93, -5.93] |
| paterson | refs_member | baseline [5,10) m | 4 | 9.12 | 5.52 | 0.75 | 3.60 [-1.13, 8.34] |
| paterson | refs_member | baseline [10,20) m | 67 | 2.42 | 2.19 | 0.55 | 0.23 [-0.73, 1.32] |
| paterson | refs_member | baseline [20,1e+09) m | 122 | 2.35 | 1.64 | 0.63 | 0.70 [0.09, 1.64] |
| paterson | refs_member | 1 sequence | 0 | — | — | — | — [—, —] |
| paterson | refs_member | >= 2 sequences | 0 | — | — | — | — [—, —] |
| paterson | refs_member | all | 194 | 2.41 | 1.97 | 0.60 | 0.44 [-0.01, 1.20] |

Signed medians on the headline unit set (computed from refs.csv): azimuth A / B Richmond
-0.69 / -0.31, Laurens +0.14 / +0.51, Bayonne +1.24 / +4.54, Paterson +0.09 / +0.01 deg. B/A
range ratio 1.001 / 1.069 / 0.994 / 0.987.

### Survivorship

- **Sites.** B can solve 98% of the >= 3-view sites in Richmond, Laurens and Paterson
  (926/945, 107/109, 4551/4552). In Bayonne it solves 86% (684/793). Failures are mostly
  rays that cross behind a camera or beyond 30 m, not low spread.
- **Member refs.** B cannot solve 7/190 (Richmond), 4/80 (Laurens), 9/62 (Bayonne) and 2/196
  (Paterson). A does badly on those too: median 8.5-9.8 deg against 2.4-4.5 deg on the unit
  set. Bayonne is the exception, at 4.8 deg.
- **What neither arm sees.** True detections whose site has fewer than 3 views
  (`det_small_site`: 20 / 17 / 33 / 36) or no site (`det_no_site`, unplaceable at 25 m:
  27 / 0 / 17 / 38) are outside the comparison entirely. With two views, B is exactly
  determined and leaves nothing to hold out.
- **Parallax is not the binding limit here.** 74-91% of headline refs have a leave-out spread
  >= 60 deg, and the 10 deg threshold drops at most 1-2 refs per city. Sites that reach 3
  operational views have been seen from around the ramp. The low-parallax case (a ramp far
  ahead on a straight drive) mostly never reaches 3 views at 0.55, so this test cannot see it.

### Exploratory (post hoc, not part of the rules)

The table at the end, by camera baseline and by number of capture sequences, tests one
explanation for Bayonne, that its baselines are too short for raw-GPS noise. **It does not
support it.** B loses at every baseline bin, including >= 20 m (GT-free -1.24 [-1.74, -0.68]).
Among single-sequence GT-free units it ties A (+0.03 [-1.40, 1.34]); it loses where views
come from >= 2 sequences (-1.10 [-1.67, -0.71]). The remaining hypothesis, not tested here:
**camera-position offsets between Panoramax sequences** (raw GPS, separate captures). These
break ray intersection, while A's GLS of raycast points degrades more gently. Mapillary's SfM
aligns sequences, and GSV positions are consistent. Bayonne's signed azimuth bias of B
(+4.5 deg) also hints at a systematic component. The integer-degree headings alone (sigma
about 0.3 deg) cannot explain a 5 deg gap.

## Blind spots

- **Shared heading bias.** A heading offset common to the views of a site rotates B's rays
  (and A's) together. When the judged pano shares it (same sequence or rig), its own azimuth
  shares it too and the error cancels in the pano frame. The metric cannot see such a bias.
- **Azimuth is B's own currency.** The headline scores only the cross-ray direction. A's
  extra information is range, the height-dependent part, and the azimuth metric cannot
  credit it. The row error (which needs a height) is smaller for A at 2.6 m in every city,
  but that comparison is in A's frame by construction.
- **Low parallax along one straight drive** is mostly absent from the >= 3-view site set (see
  survivorship). The result says nothing about 2-view or far-ahead sites.
- **Association is A's.** Sites are formed in the 2.6 m flat frame. Rule 2 shows B inherits
  that height dependence through membership.
- **Small GT samples for the open cities**: 76 peak-only refs in Laurens, 53 in Bayonne. Neither
  split has boxes, so their references are the model's own (reviewer-confirmed) peaks, which
  are not independent of the detector. Richmond's are 175/183 reviewer boxes. Paterson's are
  64/194 boxes.
- Bayonne is the 5 m run, not the 10 m one the brief named (Amendment A1).
- B's weights use the error model's sigma_heading/sigma_gps, which are guesses for
  Mapillary/Panoramax (#57). An unweighted or differently weighted B was not tried, by design
  (pre-registered).

## Replication

```bash
# from the repo root; runs/ must hold the archived results.jsonl files (gitignored)
PY=python
RUNS=runs
BENCH=../RampNet/benchmark
for c in richmond laurens bayonne paterson; do
  nice $PY scripts/bearing_refit.py run $c --run-root $RUNS --benchmark-root $BENCH
done                                   # ~2-4 min per city, no GPU/network
$PY scripts/bearing_refit.py summarize # -> docs/overnight/data/bearing_refit/summary*.{csv,md}
$PY -m pytest -q tests/test_bearing_refit.py && $PY -m doctest scripts/bearing_refit.py
```

Inputs: `results.jsonl` of each run (Bayonne sha256 per docs/panoramax-bayonne.md:
cb03768c..., 50,409 lines), RampNet splits `richmond`, `laurens_mapillary`, `bayonne`,
`paterson` (`verdicts.json`, `records.jsonl`, `boxes.json` where present). Outputs (3.7 MB):
`docs/overnight/data/bearing_refit/<city>/{refs,gtfree,stress,keeps,counts}.csv` and
`summary_*.csv`.
