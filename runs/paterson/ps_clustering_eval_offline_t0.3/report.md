# paterson: PS label clustering vs RampNet GT (offline)

mode: offline -- 35107 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (3 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
inputs: streets sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`; verdicts sha256 `5e43c54389eb9ea1d8daf0a075e562e27eb6b484dfcb3397cb6f438619365dad`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 125 judged panos -> 304 placeable points -> 304 ramps (0 cross-pano merges), 304 in the recall pool; raycast placed 37062 of 46312 detections (drops {'below_floor': 0, 'on_rig': 14, 'horizon': 129, 'out_of_range': 9107})

## Data provenance

- results file `runs/paterson/results.jsonl`: sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 35107 of 35107
- `ps_streets.geojson`: 6462 features, sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`, 2026-09-28T17:22:43+00:00 (7.5 days old at run time), from https://sidewalk-paterson.cs.washington.edu/v3/api/streets?filetype=geojson; 5306 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2617 blocks, largest 93 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 29285 fusion members vs 29285 placeable server labels; 0 of 29285 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/paterson/fusion_eval/report.md (published in the 2.6 m frame): precision 0.975, recall (union) 0.961, dual 79/14/0
- fusion_server input: 35107 AI (by account; 0 share a detection with another AI label) + 0 human labels on 12819 panos (0 positioned by inverting their labels, 12819 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 5822 labels the raycast cannot place (range cap, horizon) are singleton clusters; 10137 of its 10137 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 35107 label ids, 35107 distinct, of 35107 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 9188 panos: median 0.010 m, p90 0.25 m; from all labels (the #105 method), over 9188 panos: median 0.010 m, p90 0.25 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 18927 | 15790 | 35107 | 1.85 | 0.978 (271/6) | 0.964 (293/304) | 5 | 0.980 | 232/66/6 | 0.58 (267) | 0.84 (476) | 86/7/0 | 0.98 / 2.85 / 4 |
| ps @ 5 m | 12825 | 11190 | 35107 | 2.74 | 0.978 (271/6) | 0.944 (287/304) | 8 | 0.970 | 232/63/9 | 0.35 (117) | 0.56 (202) | 82/11/0 | 1.40 / 3.79 / 10 |
| ps @ 7.5 m | 10295 | 9490 | 35107 | 3.41 | 0.978 (271/6) | 0.934 (284/304) | 10 | 0.967 | 232/62/10 | 0.20 (62) | 0.36 (118) | 79/13/1 | 1.52 / 4.00 / 13 |
| ps @ 10 m | 9250 | 8854 | 35107 | 3.80 | 0.978 (271/6) | 0.928 (282/304) | 11 | 0.964 | 232/61/11 | 0.16 (45) | 0.30 (91) | 78/14/1 | 1.52 / 4.01 / 13 |
| ps @ 12.5 m | 8827 | 8580 | 35107 | 3.98 | 0.978 (271/6) | 0.905 (275/304) | 15 | 0.954 | 232/58/14 | 0.15 (40) | 0.25 (73) | 74/18/1 | 1.54 / 4.18 / 18 |
| ps @ 15 m | 8564 | 8398 | 35107 | 4.10 | 0.978 (271/6) | 0.901 (274/304) | 15 | 0.951 | 232/57/15 | 0.13 (35) | 0.24 (68) | 73/19/1 | 1.54 / 4.32 / 19 |
| ps_citywide @ 7.5 m | 9880 | 9124 | 35107 | 3.55 | 0.978 (271/6) | 0.924 (281/304) | 12 | 0.964 | 232/61/11 | 0.16 (47) | 0.29 (94) | 77/15/1 | 1.52 / 4.00 / 13 |
| ps_placeable @ 2.5 m | 15507 | 15507 | 29285 | 1.89 | 0.975 (233/6) | 0.964 (293/304) | 5 | 0.980 | 232/66/6 | 0.56 (259) | 0.83 (454) | 86/7/0 | 1.05 / 2.85 / 4 |
| ps_placeable @ 5 m | 10794 | 10794 | 29285 | 2.71 | 0.975 (233/6) | 0.944 (287/304) | 8 | 0.970 | 232/63/9 | 0.32 (106) | 0.54 (193) | 82/11/0 | 1.45 / 3.75 / 9 |
| ps_placeable @ 7.5 m | 9131 | 9131 | 29285 | 3.21 | 0.975 (233/6) | 0.938 (285/304) | 9 | 0.967 | 232/62/10 | 0.18 (56) | 0.34 (110) | 79/14/0 | 1.52 / 3.79 / 11 |
| ps_placeable @ 10 m | 8616 | 8616 | 29285 | 3.40 | 0.975 (233/6) | 0.934 (284/304) | 10 | 0.967 | 232/62/10 | 0.13 (38) | 0.27 (81) | 79/14/0 | 1.53 / 3.98 / 12 |
| ps_placeable @ 12.5 m | 8355 | 8355 | 29285 | 3.51 | 0.975 (233/6) | 0.911 (277/304) | 14 | 0.957 | 232/59/13 | 0.14 (38) | 0.24 (70) | 76/17/0 | 1.58 / 4.02 / 16 |
| ps_placeable @ 15 m | 8170 | 8170 | 29285 | 3.58 | 0.975 (233/6) | 0.905 (275/304) | 15 | 0.954 | 232/58/14 | 0.13 (35) | 0.23 (66) | 74/19/0 | 1.58 / 4.32 / 18 |
| ps_raycast @ 2.5 m | 17273 | 17273 | 29285 | 1.70 | 0.975 (233/6) | 0.977 (297/304) | 1 | 0.980 | 232/66/6 | 0.43 (163) | 0.80 (419) | 86/7/0 | 0.49 / 1.06 / 0 |
| ps_raycast @ 5 m | 12165 | 12165 | 29285 | 2.41 | 0.975 (233/6) | 0.961 (292/304) | 2 | 0.967 | 232/62/10 | 0.13 (38) | 0.40 (136) | 82/11/0 | 0.97 / 2.02 / 0 |
| ps_raycast @ 7.5 m | 9853 | 9853 | 29285 | 2.97 | 0.975 (233/6) | 0.947 (288/304) | 5 | 0.964 | 232/61/11 | 0.11 (31) | 0.28 (83) | 80/13/0 | 1.30 / 2.63 / 1 |
| ps_raycast @ 10 m | 8796 | 8796 | 29285 | 3.33 | 0.975 (233/6) | 0.928 (282/304) | 9 | 0.957 | 232/59/13 | 0.10 (28) | 0.21 (61) | 77/16/0 | 1.34 / 3.29 / 5 |
| ps_raycast @ 12.5 m | 8373 | 8373 | 29285 | 3.50 | 0.975 (233/6) | 0.924 (281/304) | 11 | 0.961 | 232/60/12 | 0.11 (30) | 0.21 (60) | 76/17/0 | 1.36 / 3.58 / 7 |
| ps_raycast @ 15 m | 8145 | 8145 | 29285 | 3.60 | 0.975 (233/6) | 0.901 (274/304) | 14 | 0.947 | 232/56/16 | 0.08 (23) | 0.20 (55) | 70/23/0 | 1.37 / 3.81 / 9 |
| fusion | 10137 | 10137 | 29285 | 2.89 | 0.975 (233/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.04 (12) | 0.17 (53) | 80/13/0 | 1.32 / 2.84 / 0 |
| fusion_refit | 10137 | 10137 | 29285 | 2.89 | 0.975 (233/6) | 0.951 (289/304) | 3 | 0.961 | 232/60/12 | 0.03 (10) | 0.18 (56) | 79/14/0 | 1.13 / 3.20 / 1 |
| fusion_server | 15959 | 10137 | 35107 | 2.20 | 0.978 (271/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.04 (12) | 0.17 (53) | 80/13/0 | 1.32 / 2.84 / 0 |
| fusion_server+attach | 10272 | 10137 | 35107 | 3.42 | 0.978 (271/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.04 (12) | 0.17 (53) | 80/13/0 | 1.32 / 2.84 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 232 of this run's 304 pool ramps are self-detected, so 76% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 4980 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.5 m | 9.3 m | 0.19 | 0.07 | 0.00 |
| raycast | 4.3 m | 7.1 m | 0.07 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.720 | 111/219 | 53/93 | 0.753 | 63/229 | 47/93 |
| 5 | 0.934 | 103/284 | 79/93 | 0.954 | 48/290 | 80/93 |
| 7.5 | 0.977 | 97/297 | 87/93 | 0.984 | 44/299 | 88/93 |
| 10 | 0.984 | 97/299 | 88/93 | 0.987 | 44/300 | 89/93 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ps @ 2.5 m | 18927 | 35107 | 539.1 | server | 18927 | 0.848 | 0.936 | 0.966 | 15790 | 0.845 | 0.920 | 0.956 |
| ps @ 5 m | 12825 | 35107 | 365.3 | server | 12825 | 0.601 | 0.840 | 0.938 | 11190 | 0.622 | 0.803 | 0.915 |
| ps @ 7.5 m | 10295 | 35107 | 293.2 | server | 10295 | 0.501 | 0.706 | 0.904 | 9490 | 0.528 | 0.687 | 0.879 |
| ps @ 10 m | 9250 | 35107 | 263.5 | server | 9250 | 0.510 | 0.646 | 0.877 | 8854 | 0.523 | 0.645 | 0.860 |
| ps @ 12.5 m | 8827 | 35107 | 251.4 | server | 8827 | 0.506 | 0.638 | 0.860 | 8580 | 0.513 | 0.636 | 0.850 |
| ps @ 15 m | 8564 | 35107 | 243.9 | server | 8564 | 0.499 | 0.633 | 0.852 | 8398 | 0.503 | 0.629 | 0.845 |
| ps_citywide @ 7.5 m | 9880 | 35107 | 281.4 | server | 9880 | 0.457 | 0.677 | 0.896 | 9124 | 0.485 | 0.657 | 0.867 |
| ps_placeable @ 2.5 m | 15507 | 29285 | 529.5 | server | 15507 | 0.839 | 0.916 | 0.955 | 15507 | 0.839 | 0.916 | 0.955 |
| ps_placeable @ 5 m | 10794 | 29285 | 368.6 | server | 10794 | 0.598 | 0.785 | 0.906 | 10794 | 0.598 | 0.785 | 0.906 |
| ps_placeable @ 7.5 m | 9131 | 29285 | 311.8 | server | 9131 | 0.527 | 0.656 | 0.859 | 9131 | 0.527 | 0.656 | 0.859 |
| ps_placeable @ 10 m | 8616 | 29285 | 294.2 | server | 8616 | 0.532 | 0.621 | 0.838 | 8616 | 0.532 | 0.621 | 0.838 |
| ps_placeable @ 12.5 m | 8355 | 29285 | 285.3 | server | 8355 | 0.525 | 0.613 | 0.821 | 8355 | 0.525 | 0.613 | 0.821 |
| ps_placeable @ 15 m | 8170 | 29285 | 279.0 | server | 8170 | 0.512 | 0.605 | 0.815 | 8170 | 0.512 | 0.605 | 0.815 |
| ps_raycast @ 2.5 m | 17273 | 29285 | 589.8 | server | 17273 | 0.889 | 0.931 | 0.963 | 17273 | 0.889 | 0.931 | 0.963 |
| ps_raycast @ 5 m | 12165 | 29285 | 415.4 | server | 12165 | 0.782 | 0.849 | 0.930 | 12165 | 0.782 | 0.849 | 0.930 |
| ps_raycast @ 7.5 m | 9853 | 29285 | 336.5 | server | 9853 | 0.663 | 0.749 | 0.890 | 9853 | 0.663 | 0.749 | 0.890 |
| ps_raycast @ 10 m | 8796 | 29285 | 300.4 | server | 8796 | 0.580 | 0.669 | 0.851 | 8796 | 0.580 | 0.669 | 0.851 |
| ps_raycast @ 12.5 m | 8373 | 29285 | 285.9 | server | 8373 | 0.544 | 0.627 | 0.828 | 8373 | 0.544 | 0.627 | 0.828 |
| ps_raycast @ 15 m | 8145 | 29285 | 278.1 | server | 8145 | 0.526 | 0.609 | 0.818 | 8145 | 0.526 | 0.609 | 0.818 |
| fusion | 10137 | 29285 | 346.1 | raycast | 10137 | 0.475 | 0.686 | 0.862 | 10137 | 0.475 | 0.686 | 0.862 |
| fusion_refit | 10137 | 29285 | 346.1 | raycast | 10137 | 0.496 | 0.694 | 0.866 | 10137 | 0.496 | 0.694 | 0.866 |
| fusion_server | 15959 | 35107 | 454.6 | server | 15959 | 0.801 | 0.893 | 0.952 | 10137 | 0.684 | 0.759 | 0.895 |
| fusion_server+attach | 10272 | 35107 | 292.6 | server | 10272 | 0.668 | 0.759 | 0.906 | 10137 | 0.669 | 0.759 | 0.906 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 5822 | 0.71 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 1555 | 0.50 | 0.553 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| ps @ 7.5 m | cluster of 2 | 2666 | 0.79 | 0.555 | 17 | 0.824 [0.59, 0.94] | 14 | 3 | 0 |
| ps @ 7.5 m | cluster of 3+ | 25064 | 0.87 | 0.568 | 226 | 0.986 [0.96, 1.00] | 215 | 3 | 8 |
| fusion_server | unplaceable | 5822 | 0.71 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server | cluster of 1 | 3580 | 0.76 | 0.553 | 18 | 0.889 [0.67, 0.97] | 16 | 2 | 0 |
| fusion_server | cluster of 2 | 3154 | 0.80 | 0.555 | 18 | 0.833 [0.61, 0.94] | 15 | 3 | 0 |
| fusion_server | cluster of 3+ | 22551 | 0.87 | 0.568 | 210 | 0.995 [0.97, 1.00] | 201 | 1 | 8 |
| fusion_server+attach | unplaceable | 5822 | 0.71 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 2954 | 0.75 | 0.553 | 16 | 0.875 [0.64, 0.97] | 14 | 2 | 0 |
| fusion_server+attach | cluster of 2 | 2501 | 0.78 | 0.555 | 10 | 0.800 [0.49, 0.94] | 8 | 2 | 0 |
| fusion_server+attach | cluster of 3+ | 23830 | 0.87 | 0.568 | 220 | 0.991 [0.97, 1.00] | 210 | 2 | 8 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 5822; attached 5687 (0.98)
- clusters: 15959 (`fusion_server`) -> 10272 (`fusion_server+attach`)
- sanity (not a metric): 38 unplaceable labels are on judged panos; 37 of them attached (37 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
