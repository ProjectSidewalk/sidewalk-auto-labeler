# gainesville: AI vs crowd curb-ramp agree rate (issue #31, goal 2)

## Snapshot (every number below is relative to this pull)

| file | url | fetched (UTC) | sha256 | features |
|---|---|---|---|---:|
| `raw_labels_CurbRamp.geojson` | https://sidewalk-gainesville.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-24T05:28:36+00:00 | `a3cd657e8e028be59174d85fa344a237c95924642529f18067bc9c9ff6c279de` | 4498 |
| `raw_labels_NoCurbRamp.geojson` | https://sidewalk-gainesville.cs.washington.edu/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson | 2026-09-24T05:28:36+00:00 | `a2c6c4fd08ea152aa3eba5a1c4e1176eb34fbf7a8ef19df63f5efeb506b4895e` | 59 |
| `regions.geojson` | https://sidewalk-gainesville.cs.washington.edu/v3/api/regions?filetype=geojson | 2026-09-24T05:28:10+00:00 | `25ce88180ef1995f0279ff8f09d0d3bd1a1580ecadb42e173d65fec76d0064cb` | 57 |

- run: `results.jsonl` sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`, 37435 records / 37435 distinct processed panos (35204 main pass + 2231 gap-fill records, per manifest.json); 0 without position/heading
- per-pano camera heights (the `per-pano` ablation): **30458 of 37435** panos have a measured height; the rest use the 2.6 m default. Blocks from before #40 read it from `depth/index.csv` beside results.jsonl, a local artifact that is not in git
- matcher: pano frame = RampNet geometry (x*1024, y*512, x wraps), one-to-one, strictly within 0.022 x-units (= 7.9 deg); world frame = eval_sites.match_one_to_one semantics, 5 m headline
- AI tiers: operational 0.3 (what production ships, rig-masked) and benchmark 0.55; world camera height 2.6 m (production default) with `per-pano` (#40) as the ablation

## Scope

- footprint: 40 PS regions cover the run polygon (overlap >= 0.5); **37 fully audited** (completion_rate >= 0.999) and **3 partial**: 7 (0.907), 30 (0.372), 59 (0.740)
- crowd CurbRamp labels: 4498 in the pull (0 dropped for a bad position), 44 outside the footprint (regions opened after the run), **4454 in scope**: 4253 in full regions, 201 in partial ones
- crowd NoCurbRamp labels in scope: 59 (0 dropped)
- **pano-frame coverage: 0.663 (2952/4454) [0.649, 0.677]** of in-scope crowd CurbRamp labels sit on a pano the run processed (114 of them on 55 distinct gap-fill panos, i.e. panos the tile scan never returned); 0 lack pano dimensions
- labellers: 6 accounts; the largest holds 4247 labels
- validation: 263 in-scope labels carry a human majority verdict; 4439 carry a vote from PS's own AI validator, which is why `correct` is reported apart and never used as ground truth
- contamination check: 0 crowd labels sit exactly on a stored detection pixel (an AI submission would; must be 0)
- pixel-frame alignment (PS pano_x/pano_y vs stored x/y): over 2325 crowd labels with a detection within 0.022, signed offset median dx -0.15 deg (p10-p90 -2.7 to +2.1), dy +0.15 deg (-1.9 to +2.2) — centred on zero, so the two coordinate systems agree

## Pano frame, crowd -> AI (the headline)

Statistic: share of crowd CurbRamp labels on a run-processed pano with an AI detection on the SAME pano within the radius, one-to-one. Denominator: crowd labels on processed panos. Frame-free: no raycast, no camera height, no PS placement.

| regions | operational 0.3 | benchmark 0.55 |
|---|---|---|
| full regions | 0.725 (2058/2838) [0.708, 0.741] | 0.604 (1714/2838) [0.586, 0.622] |
| partial regions | 0.789 (90/114) [0.706, 0.854] | 0.702 (80/114) [0.612, 0.778] |
| all in scope | 0.728 (2148/2952) [0.711, 0.743] | 0.608 (1794/2952) [0.590, 0.625] |

The one-to-one matcher is a choice, and it moves the number. When two crowd marks sit under one AI peak, only one of them can agree. **any detection** drops the one-to-one constraint: a crowd label agrees when ANY operational detection on its pano is within the radius. **shadowed** counts the labels that have a detection within the radius but lose it to another crowd mark on the same pano. Both are all in scope, radius 0.022.

| tier | one-to-one | any detection | shadowed |
|---|---|---|---:|
| operational 0.3 | 0.728 (2148/2952) [0.711, 0.743] | 0.788 (2325/2952) [0.772, 0.802] | 177 |
| benchmark 0.55 | 0.608 (1794/2952) [0.590, 0.625] | 0.655 (1934/2952) [0.638, 0.672] | 140 |

Radius sweep (all in scope, operational 0.3 / benchmark 0.55):

| radius (x-units) | degrees | operational | operational, any detection | benchmark |
|---:|---:|---|---|---|
| 0.011 | 4.0 | 0.643 (1898/2952) [0.625, 0.660] | 0.692 (2043/2952) [0.675, 0.708] | 0.547 (1616/2952) [0.529, 0.565] |
| 0.022 | 7.9 | 0.728 (2148/2952) [0.711, 0.743] | 0.788 (2325/2952) [0.772, 0.802] | 0.608 (1794/2952) [0.590, 0.625] |
| 0.033 | 11.9 | 0.752 (2220/2952) [0.736, 0.767] | 0.814 (2404/2952) [0.800, 0.828] | 0.624 (1841/2952) [0.606, 0.641] |
| 0.05 | 18.0 | 0.770 (2273/2952) [0.754, 0.785] | 0.834 (2463/2952) [0.821, 0.847] | 0.637 (1879/2952) [0.619, 0.654] |

### Pano-frame crowd -> AI by the label's validation

Human majority (PS AI-validator votes excluded), then the feed's `correct` (which does include them). Rates are pano-frame agree rates at operational 0.3, radius 0.022.

| bucket | agree rate |
|---|---|
| human: agreed | 0.705 (124/176) [0.633, 0.767] |
| human: disagreed | 0.000 (0/3) [0.000, 0.562] |
| human: unvalidated | 0.730 (2024/2773) [0.713, 0.746] |
| feed `correct` = true | 0.765 (1983/2591) [0.749, 0.781] |
| feed `correct` = false | 0.242 (16/66) [0.155, 0.358] |
| feed `correct` = null | 0.505 (149/295) [0.448, 0.562] |

### Pano frame, AI -> crowd (lower bound only)

Share of operational detections on crowd-labelled processed panos that a crowd label on the same pano claims. A LOWER bound: auditors label each ramp once, from one of the several panos that see it, so a correct detection of a ramp the auditor marked from a neighbouring pano counts as unclaimed here.

- operational 0.3: 0.493 (2148/4360) [0.478, 0.508]
- benchmark 0.55: 0.562 (1794/3192) [0.545, 0.579]

## World frame, crowd -> AI (a bound, not the headline)

Crowd labels at PS's own lat/lng (viewer estimator, SidewalkWebpage#4766) vs fused AI sites from the labeler's flat-ground raycast. Two independent placement errors (and RampNet#101's range scale error) sit between them, so a miss here can be geometry rather than detection. **sites** = one-to-one against operational fused sites; **union** = any single operational raycast within the radius (loosest bound). Denominator: all in-scope crowd CurbRamp labels, including those on panos the run never processed.

| tier | height | radius | sites (all in scope) | sites (full regions) | union (all in scope) |
|---|---|---:|---|---|---|
| operational 0.3 | 2.6 m | 2.5 m | 0.659 (2933/4454) [0.644, 0.672] | 0.658 (2800/4253) [0.644, 0.672] | 0.862 (3839/4454) [0.851, 0.872] |
| operational 0.3 | 2.6 m | 5 m | 0.793 (3532/4454) [0.781, 0.805] | 0.792 (3370/4253) [0.780, 0.804] | 0.941 (4193/4454) [0.934, 0.948] |
| operational 0.3 | 2.6 m | 7.5 m | 0.848 (3775/4454) [0.837, 0.858] | 0.846 (3599/4253) [0.835, 0.857] | 0.961 (4279/4454) [0.955, 0.966] |
| operational 0.3 | per-pano | 2.5 m | 0.624 (2781/4454) [0.610, 0.638] | 0.625 (2658/4253) [0.610, 0.639] | 0.850 (3786/4454) [0.839, 0.860] |
| operational 0.3 | per-pano | 5 m | 0.784 (3493/4454) [0.772, 0.796] | 0.783 (3328/4253) [0.770, 0.795] | 0.945 (4211/4454) [0.938, 0.952] |
| operational 0.3 | per-pano | 7.5 m | 0.819 (3649/4454) [0.808, 0.830] | 0.817 (3474/4253) [0.805, 0.828] | 0.960 (4278/4454) [0.954, 0.966] |
| benchmark 0.55 | 2.6 m | 2.5 m | 0.587 (2616/4454) [0.573, 0.602] | 0.586 (2491/4253) [0.571, 0.600] | 0.774 (3447/4454) [0.761, 0.786] |
| benchmark 0.55 | 2.6 m | 5 m | 0.717 (3194/4454) [0.704, 0.730] | 0.715 (3042/4253) [0.702, 0.729] | 0.883 (3934/4454) [0.873, 0.892] |
| benchmark 0.55 | 2.6 m | 7.5 m | 0.767 (3416/4454) [0.754, 0.779] | 0.765 (3255/4253) [0.752, 0.778] | 0.914 (4069/4454) [0.905, 0.921] |
| benchmark 0.55 | per-pano | 2.5 m | 0.568 (2531/4454) [0.554, 0.583] | 0.569 (2419/4253) [0.554, 0.584] | 0.782 (3482/4454) [0.769, 0.794] |
| benchmark 0.55 | per-pano | 5 m | 0.705 (3141/4454) [0.692, 0.718] | 0.703 (2989/4253) [0.689, 0.716] | 0.890 (3965/4454) [0.881, 0.899] |
| benchmark 0.55 | per-pano | 7.5 m | 0.734 (3270/4454) [0.721, 0.747] | 0.731 (3108/4253) [0.717, 0.744] | 0.919 (4092/4454) [0.910, 0.926] |

- chance floor, operational 0.3, 2.6 m, 5 m: every crowd label displaced 25 m in a random direction (seed 31) still matches a site 0.085 (378/4454) [0.077, 0.093] of the time — the share of the world-frame rate that site density alone would produce
- chance floor, benchmark 0.55, 2.6 m, 5 m: every crowd label displaced 25 m in a random direction (seed 31) still matches a site 0.063 (279/4454) [0.056, 0.070] of the time — the share of the world-frame rate that site density alone would produce

### The two frames on the same labels

Crowd labels on processed panos (2952), operational 0.3, pano radius 0.022, world 5 m at 2.6 m:

- agree in both frames: 1900
- pano frame only (the world frame loses them: placement, or beyond the 25 m raycast envelope): 248
- world frame only (the pano missed, a site from another view is near): 513
- neither: 291
- in-run labels: pano 0.728 (2148/2952) [0.711, 0.743] vs world 0.817 (2413/2952) [0.803, 0.831]
- labels on panos the run did NOT process (1502): world 0.745 (1119/1502) [0.722, 0.766] — scorable only in the world frame

## World frame, AI -> crowd (fully audited regions only)

Denominator: operational fused sites located in a fully audited region. **one-to-one** = matched to a distinct crowd CurbRamp label; **any** = a crowd CurbRamp label within radius (duplicate AI sites on one ramp count); **on NoCurbRamp** = a crowd NoCurbRamp label within radius (the auditor looked and said there is no ramp).

| tier | height | radius | one-to-one | any | on NoCurbRamp | on NoCurbRamp, no CurbRamp near |
|---|---|---:|---|---|---|---|
| operational 0.3 | 2.6 m | 2.5 m | 0.332 (2747/8264) [0.322, 0.343] | 0.352 (2910/8264) [0.342, 0.362] | 0.003 (25/8264) [0.002, 0.004] | 0.003 (25/8264) [0.002, 0.004] |
| operational 0.3 | 2.6 m | 5 m | 0.399 (3298/8264) [0.389, 0.410] | 0.504 (4167/8264) [0.493, 0.515] | 0.005 (42/8264) [0.004, 0.007] | 0.004 (36/8264) [0.003, 0.006] |
| operational 0.3 | 2.6 m | 7.5 m | 0.426 (3520/8264) [0.415, 0.437] | 0.621 (5130/8264) [0.610, 0.631] | 0.007 (56/8264) [0.005, 0.009] | 0.005 (42/8264) [0.004, 0.007] |
| operational 0.3 | per-pano | 2.5 m | 0.364 (2609/7168) [0.353, 0.375] | 0.387 (2774/7168) [0.376, 0.398] | 0.004 (26/7168) [0.002, 0.005] | 0.004 (26/7168) [0.002, 0.005] |
| operational 0.3 | per-pano | 5 m | 0.454 (3252/7168) [0.442, 0.465] | 0.532 (3813/7168) [0.520, 0.543] | 0.007 (47/7168) [0.005, 0.009] | 0.006 (42/7168) [0.004, 0.008] |
| operational 0.3 | per-pano | 7.5 m | 0.473 (3394/7168) [0.462, 0.485] | 0.606 (4347/7168) [0.595, 0.618] | 0.008 (60/7168) [0.007, 0.011] | 0.006 (43/7168) [0.004, 0.008] |
| benchmark 0.55 | 2.6 m | 2.5 m | 0.423 (2450/5788) [0.411, 0.436] | 0.439 (2541/5788) [0.426, 0.452] | 0.003 (15/5788) [0.002, 0.004] | 0.003 (15/5788) [0.002, 0.004] |
| benchmark 0.55 | 2.6 m | 5 m | 0.516 (2985/5788) [0.503, 0.529] | 0.614 (3551/5788) [0.601, 0.626] | 0.005 (27/5788) [0.003, 0.007] | 0.004 (21/5788) [0.002, 0.006] |
| benchmark 0.55 | 2.6 m | 7.5 m | 0.551 (3191/5788) [0.538, 0.564] | 0.740 (4282/5788) [0.728, 0.751] | 0.006 (34/5788) [0.004, 0.008] | 0.004 (22/5788) [0.003, 0.006] |
| benchmark 0.55 | per-pano | 2.5 m | 0.501 (2376/4744) [0.487, 0.515] | 0.517 (2452/4744) [0.503, 0.531] | 0.003 (13/4744) [0.002, 0.005] | 0.003 (13/4744) [0.002, 0.005] |
| benchmark 0.55 | per-pano | 5 m | 0.616 (2922/4744) [0.602, 0.630] | 0.680 (3224/4744) [0.666, 0.693] | 0.005 (24/4744) [0.003, 0.008] | 0.004 (19/4744) [0.003, 0.006] |
| benchmark 0.55 | per-pano | 7.5 m | 0.641 (3041/4744) [0.627, 0.655] | 0.741 (3517/4744) [0.729, 0.754] | 0.007 (35/4744) [0.005, 0.010] | 0.004 (20/4744) [0.003, 0.007] |

### Where the unlabelled AI sites are (operational 0.3, 2.6 m, 5 m; 8264 full-region sites)

| nearest crowd CurbRamp label | sites | share | seen from >= 2 panos | max member >= 0.55 |
|---|---:|---:|---|---|
| matched one-to-one (<= 5 m) | 3298 | 0.399 | 0.839 (2766/3298) [0.826, 0.851] | 0.895 (2951/3298) [0.884, 0.905] |
| label within 5 m taken by another site | 869 | 0.105 | 0.541 (470/869) [0.508, 0.574] | 0.770 (669/869) [0.741, 0.797] |
| nearest label 5-10 m | 1514 | 0.183 | 0.225 (340/1514) [0.204, 0.246] | 0.671 (1016/1514) [0.647, 0.694] |
| nearest label 10-20 m | 705 | 0.085 | 0.330 (233/705) [0.297, 0.366] | 0.423 (298/705) [0.387, 0.459] |
| no label within 20 m | 1878 | 0.227 | 0.332 (624/1878) [0.311, 0.354] | 0.454 (853/1878) [0.432, 0.477] |

## Who is right when they disagree: RampNet GT adjudication

World frame, benchmark 0.55 (the tier the bundle was judged at), 2.6 m, 5 m; GT built as eval_sites builds it (125 judged panos, 0 skipped with a warning). The GT was made during a RampNet review, so it is RampNet-anchored and favours the AI; read the crowd column as a lower bound.

- GT recall-pool ramps in fully audited regions: 195
- covered by a crowd CurbRamp label: 0.749 (146/195) [0.683, 0.804]
- covered by an operational AI site: 0.923 (180/195) [0.877, 0.953]
- both 138, crowd only 8, AI only 42, neither 7
- precision of full-region AI sites WITH a crowd label within 5 m: 0.992 (118/119) [0.954, 0.999]
- precision of full-region AI sites with NO crowd label within 5 m: 0.837 (36/43) [0.700, 0.919]. Measured on benchmark 0.55 sites (the tier the bundle was judged at): the precision of the extra sites the operational 0.3 tier adds is NOT measured here

### How much of the operational 0.3 AI -> crowd gap crowd omissions can explain

- unmatched full-region AI sites (operational 0.3, 2.6 m, 5 m one-to-one): 4966 of 8264
- crowd recall 0.749 over 4253 full-region crowd labels implies about **1427** skipped ramps (1034-1970 over the recall's 95% CI), assuming one label per ramp
- so crowd omissions can explain **at most 29%** of the unmatched sites (21%-40%); each skipped ramp accounts for at most one site, and one the AI also missed accounts for none
- unmatched sites with no crowd label within 20 m: 1878 (38% of the unmatched)
- the rest of the gap is fragmentation, placement beyond 5 m, and low-confidence single-view sites that nobody judged. Precision at operational 0.3 is unmeasured

## Vintage: how much of the crowd -> AI miss rate is imagery age

Each in-scope crowd label is placed against the run's imagery: **same pano** (scored in both frames), else against the run pano whose GSV history lists the crowd pano, else the nearest run pano within 30 m; months = ours minus theirs. The WORLD rate is used for every stratum so they are comparable (operational 0.3, 2.6 m, 5 m, sites one-to-one); the pano-frame rate is given where it exists. **excess misses** = misses beyond what the same-pano stratum's world rate predicts for that many labels - the part of the gap a vintage difference could explain (an upper bound on it: any other difference between the strata lands here too).

| stratum | labels | history-linked | pano-frame agree | world agree | excess misses |
|---|---:|---:|---|---|---:|
| same pano | 2952 | 0 | 0.728 (2148/2952) [0.711, 0.743] | 0.817 (2413/2952) [0.803, 0.831] | +0.0 |
| newer than ours | 348 | 43 | - | 0.716 (249/348) [0.666, 0.760] | +35.5 |
| same month | 912 | 0 | - | 0.787 (718/912) [0.760, 0.813] | +27.5 |
| 1-18 mo older | 166 | 47 | - | 0.699 (116/166) [0.625, 0.763] | +19.7 |
| 19-36 mo older | 13 | 5 | - | 0.615 (8/13) [0.355, 0.823] | +2.6 |
| >36 mo older | 42 | 18 | - | 0.667 (28/42) [0.516, 0.790] | +6.3 |
| no run pano nearby | 21 | 0 | - | 0.000 (0/21) [0.000, 0.155] | +17.2 |

- world-frame misses overall: 922 of 4454; excess over the same-pano rate: 108.8 (12% of the misses)

## Per region

operational 0.3; pano radius 0.022; world 2.6 m, 5 m. **AI -> crowd** = operational sites located in the region matched one-to-one to a crowd CurbRamp label. Partial regions are listed for completeness; their AI -> crowd is biased low (unaudited streets) and never pooled.

| region | class | completion | crowd labels | on run panos | pano agree | world agree | AI sites | AI -> crowd |
|---|---|---:|---:|---:|---|---|---:|---|
| 5 | full | 1.000 | 98 | 74 | 0.797 (59/74) [0.692, 0.873] | 0.867 (85/98) [0.786, 0.921] | 188 | 0.426 (80/188) [0.357, 0.497] |
| 6 | full | 1.000 | 118 | 54 | 0.704 (38/54) [0.572, 0.809] | 0.797 (94/118) [0.715, 0.859] | 196 | 0.423 (83/196) [0.356, 0.493] |
| 8 | full | 1.000 | 197 | 41 | 0.756 (31/41) [0.607, 0.862] | 0.822 (162/197) [0.763, 0.869] | 376 | 0.420 (158/376) [0.371, 0.471] |
| 14 | full | 1.000 | 239 | 224 | 0.679 (152/224) [0.615, 0.736] | 0.686 (164/239) [0.625, 0.742] | 275 | 0.553 (152/275) [0.494, 0.610] |
| 16 | full | 1.000 | 158 | 122 | 0.770 (94/122) [0.688, 0.836] | 0.829 (131/158) [0.763, 0.880] | 289 | 0.415 (120/289) [0.360, 0.473] |
| 20 | full | 1.000 | 80 | 52 | 0.750 (39/52) [0.618, 0.848] | 0.750 (60/80) [0.645, 0.832] | 191 | 0.330 (63/191) [0.267, 0.399] |
| 21 | full | 1.000 | 17 | 5 | 0.400 (2/5) [0.118, 0.769] | 0.765 (13/17) [0.527, 0.904] | 20 | 0.650 (13/20) [0.433, 0.819] |
| 22 | full | 1.000 | 42 | 26 | 0.769 (20/26) [0.579, 0.890] | 0.786 (33/42) [0.641, 0.883] | 107 | 0.280 (30/107) [0.204, 0.372] |
| 23 | full | 1.000 | 28 | 16 | 0.625 (10/16) [0.386, 0.815] | 0.571 (16/28) [0.391, 0.735] | 43 | 0.442 (19/43) [0.304, 0.589] |
| 24 | full | 1.000 | 54 | 17 | 0.706 (12/17) [0.469, 0.867] | 0.667 (36/54) [0.534, 0.778] | 78 | 0.410 (32/78) [0.308, 0.521] |
| 25 | full | 1.000 | 189 | 148 | 0.730 (108/148) [0.653, 0.795] | 0.735 (139/189) [0.668, 0.793] | 356 | 0.379 (135/356) [0.330, 0.431] |
| 26 | full | 1.000 | 155 | 154 | 0.779 (120/154) [0.707, 0.837] | 0.826 (128/155) [0.758, 0.877] | 261 | 0.448 (117/261) [0.389, 0.509] |
| 27 | full | 1.000 | 241 | 232 | 0.655 (152/232) [0.592, 0.713] | 0.809 (195/241) [0.755, 0.854] | 527 | 0.347 (183/527) [0.308, 0.389] |
| 28 | full | 1.000 | 173 | 168 | 0.792 (133/168) [0.724, 0.846] | 0.763 (132/173) [0.694, 0.820] | 309 | 0.417 (129/309) [0.364, 0.473] |
| 29 | full | 1.000 | 41 | 35 | 0.600 (21/35) [0.436, 0.744] | 0.780 (32/41) [0.633, 0.880] | 66 | 0.455 (30/66) [0.340, 0.574] |
| 31 | full | 1.000 | 38 | 34 | 0.676 (23/34) [0.508, 0.809] | 0.763 (29/38) [0.608, 0.870] | 90 | 0.344 (31/90) [0.254, 0.447] |
| 32 | full | 1.000 | 214 | 166 | 0.705 (117/166) [0.631, 0.769] | 0.593 (127/214) [0.527, 0.657] | 387 | 0.341 (132/387) [0.296, 0.390] |
| 33 | full | 1.000 | 29 | 14 | 0.429 (6/14) [0.214, 0.674] | 0.586 (17/29) [0.407, 0.745] | 113 | 0.248 (28/113) [0.177, 0.335] |
| 35 | full | 1.000 | 580 | 346 | 0.769 (266/346) [0.722, 0.810] | 0.871 (505/580) [0.841, 0.896] | 1326 | 0.394 (522/1326) [0.368, 0.420] |
| 38 | full | 1.000 | 45 | 26 | 0.731 (19/26) [0.539, 0.863] | 0.844 (38/45) [0.712, 0.923] | 65 | 0.462 (30/65) [0.346, 0.581] |
| 39 | full | 1.000 | 2 | 2 | 0.500 (1/2) [0.095, 0.905] | 0.500 (1/2) [0.095, 0.905] | 35 | 0.257 (9/35) [0.142, 0.421] |
| 40 | full | 1.000 | 359 | 322 | 0.717 (231/322) [0.666, 0.764] | 0.850 (305/359) [0.809, 0.883] | 785 | 0.433 (340/785) [0.399, 0.468] |
| 42 | full | 1.000 | 101 | 48 | 0.729 (35/48) [0.590, 0.834] | 0.703 (71/101) [0.608, 0.783] | 227 | 0.339 (77/227) [0.281, 0.403] |
| 43 | full | 1.000 | 181 | 86 | 0.814 (70/86) [0.719, 0.882] | 0.906 (164/181) [0.855, 0.941] | 310 | 0.465 (144/310) [0.410, 0.520] |
| 44 | full | 1.000 | 117 | 66 | 0.742 (49/66) [0.626, 0.833] | 0.803 (94/117) [0.722, 0.865] | 222 | 0.414 (92/222) [0.352, 0.480] |
| 46 | full | 1.000 | 95 | 51 | 0.588 (30/51) [0.452, 0.712] | 0.811 (77/95) [0.720, 0.877] | 191 | 0.372 (71/191) [0.306, 0.442] |
| 47 | full | 1.000 | 82 | 44 | 0.659 (29/44) [0.511, 0.781] | 0.817 (67/82) [0.720, 0.886] | 198 | 0.364 (72/198) [0.300, 0.433] |
| 48 | full | 1.000 | 162 | 77 | 0.649 (50/77) [0.538, 0.747] | 0.877 (142/162) [0.817, 0.919] | 328 | 0.418 (137/328) [0.366, 0.472] |
| 49 | full | 1.000 | 78 | 22 | 0.864 (19/22) [0.667, 0.953] | 0.885 (69/78) [0.795, 0.938] | 189 | 0.302 (57/189) [0.241, 0.370] |
| 50 | full | 1.000 | 91 | 65 | 0.662 (43/65) [0.540, 0.765] | 0.791 (72/91) [0.697, 0.862] | 136 | 0.456 (62/136) [0.375, 0.540] |
| 54 | full | 1.000 | 28 | 16 | 0.875 (14/16) [0.640, 0.965] | 0.929 (26/28) [0.774, 0.980] | 58 | 0.414 (24/58) [0.296, 0.542] |
| 55 | full | 1.000 | 6 | 4 | 1.000 (4/4) [0.510, 1.000] | 1.000 (6/6) [0.610, 1.000] | 12 | 0.500 (6/12) [0.254, 0.746] |
| 56 | full | 1.000 | 32 | 5 | 0.800 (4/5) [0.376, 0.964] | 0.719 (23/32) [0.546, 0.844] | 45 | 0.511 (23/45) [0.370, 0.650] |
| 57 | full | 1.000 | 12 | 6 | 1.000 (6/6) [0.610, 1.000] | 0.583 (7/12) [0.320, 0.807] | 12 | 0.417 (5/12) [0.193, 0.680] |
| 58 | full | 1.000 | 6 | 2 | 0.500 (1/2) [0.095, 0.905] | 0.667 (4/6) [0.300, 0.903] | 14 | 0.357 (5/14) [0.163, 0.612] |
| 93 | full | 1.000 | 46 | 27 | 0.778 (21/27) [0.592, 0.894] | 0.609 (28/46) [0.465, 0.736] | 42 | 0.452 (19/42) [0.312, 0.601] |
| 150 | full | 1.000 | 119 | 41 | 0.707 (29/41) [0.555, 0.824] | 0.655 (78/119) [0.566, 0.735] | 197 | 0.365 (72/197) [0.301, 0.435] |
| 7 | partial | 0.907 | 81 | 39 | 0.718 (28/39) [0.562, 0.835] | 0.778 (63/81) [0.676, 0.855] | 190 | 0.337 (64/190) [0.273, 0.407] |
| 30 | partial | 0.372 | 26 | 16 | 0.938 (15/16) [0.717, 0.989] | 0.731 (19/26) [0.539, 0.863] | 89 | 0.157 (14/89) [0.096, 0.247] |
| 59 | partial | 0.740 | 94 | 59 | 0.797 (47/59) [0.677, 0.880] | 0.851 (80/94) [0.765, 0.909] | 213 | 0.329 (70/213) [0.269, 0.394] |
