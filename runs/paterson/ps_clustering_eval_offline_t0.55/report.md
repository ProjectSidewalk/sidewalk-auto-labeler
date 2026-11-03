# paterson: PS label clustering vs RampNet GT (offline)

mode: offline -- 29754 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (2 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
inputs: streets sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`; verdicts sha256 `5e43c54389eb9ea1d8daf0a075e562e27eb6b484dfcb3397cb6f438619365dad`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 125 judged panos -> 304 placeable points -> 304 ramps (0 cross-pano merges), 304 in the recall pool; raycast placed 37062 of 46312 detections (drops {'below_floor': 0, 'on_rig': 14, 'horizon': 129, 'out_of_range': 9107})

## Data provenance

- results file `runs/paterson/results.jsonl`: sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 29754 of 29754
- `ps_streets.geojson`: 6462 features, sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`, 2026-09-28T17:22:43+00:00 (7.5 days old at run time), from https://sidewalk-paterson.cs.washington.edu/v3/api/streets?filetype=geojson; 5306 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2239 blocks, largest 73 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 25766 fusion members vs 25766 placeable server labels; 0 of 25766 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/paterson/fusion_eval/report.md (published in the 2.6 m frame): precision 0.975, recall (union) 0.957, dual 77/16/0
- fusion_server input: 29754 AI (by account; 0 share a detection with another AI label) + 0 human labels on 10941 panos (0 positioned by inverting their labels, 10941 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 3988 labels the raycast cannot place (range cap, horizon) are singleton clusters; 8872 of its 8872 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 29754 label ids, 29754 distinct, of 29754 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 8356 panos: median 0.010 m, p90 0.24 m; from all labels (the #105 method), over 8356 panos: median 0.010 m, p90 0.24 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 15465 | 13460 | 29754 | 1.92 | 0.978 (271/6) | 0.964 (293/304) | 5 | 0.980 | 232/66/6 | 0.55 (229) | 0.80 (406) | 86/7/0 | 1.03 / 2.79 / 3 |
| ps @ 5 m | 10289 | 9335 | 29754 | 2.89 | 0.978 (271/6) | 0.924 (281/304) | 10 | 0.957 | 232/59/13 | 0.28 (86) | 0.47 (154) | 78/14/1 | 1.42 / 3.79 / 10 |
| ps @ 7.5 m | 8383 | 7961 | 29754 | 3.55 | 0.978 (271/6) | 0.911 (277/304) | 12 | 0.951 | 232/57/15 | 0.14 (41) | 0.29 (86) | 74/17/2 | 1.48 / 4.02 / 14 |
| ps @ 10 m | 7706 | 7542 | 29754 | 3.86 | 0.978 (271/6) | 0.905 (275/304) | 13 | 0.947 | 232/56/16 | 0.12 (32) | 0.25 (70) | 72/19/2 | 1.50 / 4.02 / 15 |
| ps @ 12.5 m | 7484 | 7375 | 29754 | 3.98 | 0.978 (271/6) | 0.885 (269/304) | 17 | 0.941 | 232/54/18 | 0.12 (32) | 0.23 (62) | 69/22/2 | 1.54 / 4.15 / 19 |
| ps @ 15 m | 7338 | 7256 | 29754 | 4.05 | 0.978 (271/6) | 0.878 (267/304) | 17 | 0.934 | 232/52/20 | 0.11 (30) | 0.22 (59) | 67/24/2 | 1.55 / 4.18 / 20 |
| ps_citywide @ 7.5 m | 7984 | 7587 | 29754 | 3.73 | 0.978 (271/6) | 0.905 (275/304) | 14 | 0.951 | 232/57/15 | 0.08 (24) | 0.20 (60) | 73/18/2 | 1.50 / 4.03 / 14 |
| ps_placeable @ 2.5 m | 13236 | 13236 | 25766 | 1.95 | 0.975 (233/6) | 0.964 (293/304) | 5 | 0.980 | 232/66/6 | 0.54 (222) | 0.79 (390) | 86/7/0 | 1.05 / 2.79 / 3 |
| ps_placeable @ 5 m | 9075 | 9075 | 25766 | 2.84 | 0.975 (233/6) | 0.924 (281/304) | 10 | 0.957 | 232/59/13 | 0.27 (82) | 0.46 (149) | 78/14/1 | 1.43 / 3.79 / 9 |
| ps_placeable @ 7.5 m | 7761 | 7761 | 25766 | 3.32 | 0.975 (233/6) | 0.914 (278/304) | 11 | 0.951 | 232/57/15 | 0.14 (40) | 0.28 (83) | 74/18/1 | 1.49 / 3.93 / 12 |
| ps_placeable @ 10 m | 7438 | 7438 | 25766 | 3.46 | 0.975 (233/6) | 0.911 (277/304) | 12 | 0.951 | 232/57/15 | 0.11 (31) | 0.24 (67) | 73/19/1 | 1.51 / 3.95 / 14 |
| ps_placeable @ 12.5 m | 7270 | 7270 | 25766 | 3.54 | 0.975 (233/6) | 0.895 (272/304) | 15 | 0.944 | 232/55/17 | 0.11 (31) | 0.22 (60) | 71/21/1 | 1.55 / 4.02 / 17 |
| ps_placeable @ 15 m | 7138 | 7138 | 25766 | 3.61 | 0.975 (233/6) | 0.885 (269/304) | 16 | 0.938 | 232/53/19 | 0.12 (31) | 0.22 (59) | 68/24/1 | 1.58 / 4.11 / 19 |
| ps_raycast @ 2.5 m | 15093 | 15093 | 25766 | 1.71 | 0.975 (233/6) | 0.977 (297/304) | 1 | 0.980 | 232/66/6 | 0.41 (150) | 0.75 (389) | 86/7/0 | 0.47 / 1.06 / 0 |
| ps_raycast @ 5 m | 10572 | 10572 | 25766 | 2.44 | 0.975 (233/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.14 (43) | 0.37 (120) | 80/13/0 | 1.01 / 2.13 / 0 |
| ps_raycast @ 7.5 m | 8599 | 8599 | 25766 | 3.00 | 0.975 (233/6) | 0.944 (287/304) | 5 | 0.961 | 232/60/12 | 0.12 (35) | 0.26 (78) | 78/15/0 | 1.30 / 2.63 / 0 |
| ps_raycast @ 10 m | 7693 | 7693 | 25766 | 3.35 | 0.975 (233/6) | 0.924 (281/304) | 8 | 0.951 | 232/57/15 | 0.12 (33) | 0.21 (59) | 74/19/0 | 1.34 / 3.21 / 3 |
| ps_raycast @ 12.5 m | 7339 | 7339 | 25766 | 3.51 | 0.975 (233/6) | 0.918 (279/304) | 11 | 0.954 | 232/58/14 | 0.13 (35) | 0.20 (57) | 72/21/0 | 1.34 / 3.65 / 5 |
| ps_raycast @ 15 m | 7167 | 7167 | 25766 | 3.60 | 0.975 (233/6) | 0.901 (274/304) | 14 | 0.947 | 232/56/16 | 0.11 (29) | 0.19 (53) | 68/25/0 | 1.36 / 3.89 / 7 |
| fusion | 8872 | 8872 | 25766 | 2.90 | 0.975 (233/6) | 0.947 (288/304) | 4 | 0.961 | 232/60/12 | 0.04 (11) | 0.15 (47) | 78/15/0 | 1.32 / 2.68 / 0 |
| fusion_refit | 8872 | 8872 | 25766 | 2.90 | 0.975 (233/6) | 0.944 (287/304) | 4 | 0.957 | 232/59/13 | 0.03 (10) | 0.16 (50) | 77/16/0 | 1.11 / 3.21 / 1 |
| fusion_server | 12860 | 8872 | 29754 | 2.31 | 0.978 (271/6) | 0.947 (288/304) | 4 | 0.961 | 232/60/12 | 0.04 (11) | 0.15 (47) | 78/15/0 | 1.32 / 2.68 / 0 |
| fusion_server+attach | 8956 | 8872 | 29754 | 3.32 | 0.978 (271/6) | 0.947 (288/304) | 4 | 0.961 | 232/60/12 | 0.04 (11) | 0.15 (47) | 78/15/0 | 1.32 / 2.68 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 232 of this run's 304 pool ramps are self-detected, so 76% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 4564 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.0 m | 7.9 m | 0.12 | 0.04 | 0.00 |
| raycast | 4.1 m | 6.9 m | 0.06 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.691 | 90/210 | 46/93 | 0.766 | 53/233 | 51/93 |
| 5 | 0.911 | 81/277 | 74/93 | 0.947 | 43/288 | 78/93 |
| 7.5 | 0.964 | 73/293 | 84/93 | 0.974 | 41/296 | 85/93 |
| 10 | 0.974 | 73/296 | 85/93 | 0.984 | 40/299 | 88/93 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ps @ 2.5 m | 15465 | 29754 | 519.8 | server | 15465 | 0.877 | 0.951 | 0.979 | 13460 | 0.870 | 0.933 | 0.971 |
| ps @ 5 m | 10289 | 29754 | 345.8 | server | 10289 | 0.627 | 0.840 | 0.945 | 9335 | 0.638 | 0.799 | 0.922 |
| ps @ 7.5 m | 8383 | 29754 | 281.7 | server | 8383 | 0.532 | 0.702 | 0.902 | 7961 | 0.548 | 0.678 | 0.877 |
| ps @ 10 m | 7706 | 29754 | 259.0 | server | 7706 | 0.545 | 0.647 | 0.873 | 7542 | 0.548 | 0.639 | 0.861 |
| ps @ 12.5 m | 7484 | 29754 | 251.5 | server | 7484 | 0.542 | 0.639 | 0.862 | 7375 | 0.543 | 0.632 | 0.855 |
| ps @ 15 m | 7338 | 29754 | 246.6 | server | 7338 | 0.530 | 0.630 | 0.857 | 7256 | 0.530 | 0.623 | 0.852 |
| ps_citywide @ 7.5 m | 7984 | 29754 | 268.3 | server | 7984 | 0.481 | 0.663 | 0.888 | 7587 | 0.499 | 0.636 | 0.859 |
| ps_placeable @ 2.5 m | 13236 | 25766 | 513.7 | server | 13236 | 0.864 | 0.930 | 0.971 | 13236 | 0.864 | 0.930 | 0.971 |
| ps_placeable @ 5 m | 9075 | 25766 | 352.2 | server | 9075 | 0.619 | 0.779 | 0.913 | 9075 | 0.619 | 0.779 | 0.913 |
| ps_placeable @ 7.5 m | 7761 | 25766 | 301.2 | server | 7761 | 0.548 | 0.652 | 0.860 | 7761 | 0.548 | 0.652 | 0.860 |
| ps_placeable @ 10 m | 7438 | 25766 | 288.7 | server | 7438 | 0.553 | 0.624 | 0.842 | 7438 | 0.553 | 0.624 | 0.842 |
| ps_placeable @ 12.5 m | 7270 | 25766 | 282.2 | server | 7270 | 0.545 | 0.617 | 0.830 | 7270 | 0.545 | 0.617 | 0.830 |
| ps_placeable @ 15 m | 7138 | 25766 | 277.0 | server | 7138 | 0.530 | 0.606 | 0.825 | 7138 | 0.530 | 0.606 | 0.825 |
| ps_raycast @ 2.5 m | 15093 | 25766 | 585.8 | server | 15093 | 0.914 | 0.946 | 0.979 | 15093 | 0.914 | 0.946 | 0.979 |
| ps_raycast @ 5 m | 10572 | 25766 | 410.3 | server | 10572 | 0.805 | 0.858 | 0.949 | 10572 | 0.805 | 0.858 | 0.949 |
| ps_raycast @ 7.5 m | 8599 | 25766 | 333.7 | server | 8599 | 0.689 | 0.756 | 0.908 | 8599 | 0.689 | 0.756 | 0.908 |
| ps_raycast @ 10 m | 7693 | 25766 | 298.6 | server | 7693 | 0.602 | 0.669 | 0.868 | 7693 | 0.602 | 0.669 | 0.868 |
| ps_raycast @ 12.5 m | 7339 | 25766 | 284.8 | server | 7339 | 0.565 | 0.624 | 0.843 | 7339 | 0.565 | 0.624 | 0.843 |
| ps_raycast @ 15 m | 7167 | 25766 | 278.2 | server | 7167 | 0.547 | 0.607 | 0.832 | 7167 | 0.547 | 0.607 | 0.832 |
| fusion | 8872 | 25766 | 344.3 | raycast | 8872 | 0.481 | 0.686 | 0.869 | 8872 | 0.481 | 0.686 | 0.869 |
| fusion_refit | 8872 | 25766 | 344.3 | raycast | 8872 | 0.504 | 0.694 | 0.876 | 8872 | 0.504 | 0.694 | 0.876 |
| fusion_server | 12860 | 29754 | 432.2 | server | 12860 | 0.818 | 0.898 | 0.965 | 8872 | 0.707 | 0.767 | 0.914 |
| fusion_server+attach | 8956 | 29754 | 301.0 | server | 8956 | 0.702 | 0.769 | 0.924 | 8872 | 0.703 | 0.767 | 0.923 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 3988 | 0.80 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 938 | 0.77 | 0.553 | 7 | 0.714 [0.36, 0.92] | 5 | 2 | 0 |
| ps @ 7.5 m | cluster of 2 | 2326 | 0.84 | 0.555 | 22 | 0.818 [0.61, 0.93] | 18 | 4 | 0 |
| ps @ 7.5 m | cluster of 3+ | 22502 | 0.88 | 0.568 | 217 | 1.000 [0.98, 1.00] | 209 | 0 | 8 |
| fusion_server | unplaceable | 3988 | 0.80 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server | cluster of 1 | 2873 | 0.84 | 0.553 | 21 | 0.810 [0.60, 0.92] | 17 | 4 | 0 |
| fusion_server | cluster of 2 | 2870 | 0.85 | 0.568 | 21 | 0.905 [0.71, 0.97] | 19 | 2 | 0 |
| fusion_server | cluster of 3+ | 20023 | 0.88 | 0.568 | 204 | 1.000 [0.98, 1.00] | 196 | 0 | 8 |
| fusion_server+attach | unplaceable | 3988 | 0.80 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 2415 | 0.85 | 0.553 | 18 | 0.833 [0.61, 0.94] | 15 | 3 | 0 |
| fusion_server+attach | cluster of 2 | 2362 | 0.84 | 0.568 | 18 | 0.833 [0.61, 0.94] | 15 | 3 | 0 |
| fusion_server+attach | cluster of 3+ | 20989 | 0.88 | 0.568 | 210 | 1.000 [0.98, 1.00] | 202 | 0 | 8 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 3988; attached 3904 (0.98)
- clusters: 12860 (`fusion_server`) -> 8956 (`fusion_server+attach`)
- sanity (not a metric): 38 unplaceable labels are on judged panos; 37 of them attached (37 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
