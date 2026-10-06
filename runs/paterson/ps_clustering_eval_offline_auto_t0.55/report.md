# paterson: PS label clustering vs RampNet GT (offline)

mode: offline -- 29754 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (2 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
inputs: streets sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`; verdicts sha256 `5e43c54389eb9ea1d8daf0a075e562e27eb6b484dfcb3397cb6f438619365dad`
raycast camera height auto; fusion arm at --min-confidence 0.55
- camera height mode `auto`: auto -> gsv-per-rig
- 34687 of 34687 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 118 of 136 measured -> 2.5 m
  - 2008: median 2.30 m, 98 of 151 measured -> 2.5 m
  - 2012: median 2.33 m, 1 of 414 measured -> 2.5 m
  - 2013: median 2.29 m, 1 of 88 measured -> 2.5 m
  - 2014: median 2.08 m, 12 of 278 measured -> 2.5 m
  - 2015: median 2.45 m, 3 of 155 measured -> 2.5 m
  - 2016: median 2.42 m, 1 of 93 measured -> 2.5 m
  - 2017: median 2.38 m, 17 of 111 measured -> 2.5 m
  - 2018: median 2.35 m, 1523 of 1646 measured -> 2.5 m
  - 2019: median 2.37 m, 1263 of 1509 measured -> 2.5 m
  - 2020: median 2.35 m, 2933 of 3102 measured -> 2.5 m
  - 2021: median 2.35 m, 1658 of 1794 measured -> 2.5 m
  - 2022: median 2.35 m, 29 of 30 measured -> 2.5 m
  - 2023: median 2.14 m, 11 of 35 measured -> 2.5 m
  - 2024: median 2.37 m, 7408 of 8403 measured -> 2.5 m
  - 2025: median 1.86 m, 15155 of 16615 measured -> 2 m
  - 2026: median 1.74 m, 94 of 127 measured -> 2 m
GT: 125 judged panos -> 323 placeable points -> 323 ramps (0 cross-pano merges), 323 in the recall pool; raycast placed 37062 of 46312 detections (drops {'below_floor': 0, 'on_rig': 14, 'horizon': 129, 'out_of_range': 9107})

## Data provenance

- results file `runs/paterson/results.jsonl`: sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 29754 of 29754
- `ps_streets.geojson`: 6462 features, sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`, 2026-09-28T17:22:43+00:00 (7.5 days old at run time), from https://sidewalk-paterson.cs.washington.edu/v3/api/streets?filetype=geojson; 5306 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2239 blocks, largest 73 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 25766 fusion members vs 25766 placeable server labels; 0 of 25766 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at auto vs runs/paterson/fusion_eval/report.md (published in the 2.6 m frame): precision 0.975, recall (union) 0.957, dual 86/13/1 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 29754 AI (by account; 0 share a detection with another AI label) + 0 human labels on 10941 panos (0 positioned by inverting their labels, 10941 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 3988 labels the raycast cannot place (range cap, horizon) are singleton clusters; 7170 of its 7171 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 29754 label ids, 29754 distinct, of 29754 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 8356 panos: median 0.010 m, p90 0.24 m; from all labels (the #105 method), over 8356 panos: median 0.010 m, p90 0.24 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 15465 | 13460 | 29754 | 1.92 | 0.978 (271/6) | 0.991 (320/323) | 0 | 0.991 | 232/88/3 | 0.67 (355) | 0.84 (507) | 99/1/0 | 0.61 / 1.71 / 0 |
| ps @ 5 m | 10289 | 9335 | 29754 | 2.89 | 0.978 (271/6) | 0.966 (312/323) | 2 | 0.972 | 232/82/9 | 0.33 (115) | 0.50 (185) | 90/10/0 | 0.85 / 2.03 / 3 |
| ps @ 7.5 m | 8383 | 7961 | 29754 | 3.55 | 0.978 (271/6) | 0.944 (305/323) | 3 | 0.954 | 232/76/15 | 0.18 (57) | 0.30 (96) | 85/14/1 | 0.94 / 2.17 / 4 |
| ps @ 10 m | 7706 | 7542 | 29754 | 3.86 | 0.978 (271/6) | 0.941 (304/323) | 3 | 0.950 | 232/75/16 | 0.15 (48) | 0.24 (73) | 85/14/1 | 0.95 / 2.25 / 4 |
| ps @ 12.5 m | 7484 | 7375 | 29754 | 3.98 | 0.978 (271/6) | 0.929 (300/323) | 5 | 0.944 | 232/73/18 | 0.14 (42) | 0.21 (64) | 82/17/1 | 0.96 / 2.56 / 8 |
| ps @ 15 m | 7338 | 7256 | 29754 | 4.05 | 0.978 (271/6) | 0.923 (298/323) | 5 | 0.938 | 232/71/20 | 0.13 (39) | 0.20 (59) | 80/19/1 | 0.96 / 2.77 / 9 |
| ps_citywide @ 7.5 m | 7984 | 7587 | 29754 | 3.73 | 0.978 (271/6) | 0.935 (302/323) | 5 | 0.950 | 232/75/16 | 0.11 (35) | 0.22 (70) | 82/17/1 | 0.94 / 2.30 / 4 |
| ps_placeable @ 2.5 m | 13236 | 13236 | 25766 | 1.95 | 0.975 (233/6) | 0.991 (320/323) | 0 | 0.991 | 232/88/3 | 0.66 (338) | 0.83 (486) | 99/1/0 | 0.61 / 1.71 / 0 |
| ps_placeable @ 5 m | 9075 | 9075 | 25766 | 2.84 | 0.975 (233/6) | 0.966 (312/323) | 2 | 0.972 | 232/82/9 | 0.32 (109) | 0.48 (176) | 90/10/0 | 0.85 / 2.03 / 3 |
| ps_placeable @ 7.5 m | 7761 | 7761 | 25766 | 3.32 | 0.975 (233/6) | 0.944 (305/323) | 3 | 0.954 | 232/76/15 | 0.17 (54) | 0.28 (89) | 85/14/1 | 0.94 / 2.17 / 4 |
| ps_placeable @ 10 m | 7438 | 7438 | 25766 | 3.46 | 0.975 (233/6) | 0.941 (304/323) | 3 | 0.950 | 232/75/16 | 0.15 (48) | 0.23 (70) | 85/14/1 | 0.95 / 2.25 / 4 |
| ps_placeable @ 12.5 m | 7270 | 7270 | 25766 | 3.54 | 0.975 (233/6) | 0.932 (301/323) | 4 | 0.944 | 232/73/18 | 0.14 (41) | 0.20 (60) | 84/15/1 | 0.96 / 2.47 / 7 |
| ps_placeable @ 15 m | 7138 | 7138 | 25766 | 3.61 | 0.975 (233/6) | 0.923 (298/323) | 5 | 0.938 | 232/71/20 | 0.14 (41) | 0.20 (59) | 81/18/1 | 0.96 / 2.77 / 9 |
| ps_raycast @ 2.5 m | 11523 | 11523 | 25766 | 2.24 | 0.975 (233/6) | 0.988 (319/323) | 0 | 0.988 | 232/87/4 | 0.41 (151) | 0.66 (307) | 98/2/0 | 0.46 / 0.95 / 0 |
| ps_raycast @ 5 m | 8200 | 8200 | 25766 | 3.14 | 0.975 (233/6) | 0.966 (312/323) | 0 | 0.966 | 232/80/11 | 0.15 (47) | 0.30 (99) | 91/9/0 | 0.76 / 1.81 / 0 |
| ps_raycast @ 7.5 m | 7468 | 7468 | 25766 | 3.45 | 0.975 (233/6) | 0.944 (305/323) | 4 | 0.957 | 232/77/14 | 0.14 (45) | 0.21 (67) | 85/14/1 | 0.88 / 2.04 / 0 |
| ps_raycast @ 10 m | 7310 | 7310 | 25766 | 3.52 | 0.975 (233/6) | 0.938 (303/323) | 4 | 0.950 | 232/75/16 | 0.14 (43) | 0.20 (61) | 83/16/1 | 0.93 / 2.16 / 1 |
| ps_raycast @ 12.5 m | 7212 | 7212 | 25766 | 3.57 | 0.975 (233/6) | 0.926 (299/323) | 7 | 0.947 | 232/74/17 | 0.12 (38) | 0.18 (56) | 81/18/1 | 0.94 / 2.40 / 3 |
| ps_raycast @ 15 m | 7071 | 7071 | 25766 | 3.64 | 0.975 (233/6) | 0.913 (295/323) | 8 | 0.938 | 232/71/20 | 0.12 (34) | 0.17 (49) | 78/21/1 | 0.95 / 2.67 / 5 |
| fusion | 7172 | 7172 | 25766 | 3.59 | 0.975 (233/6) | 0.944 (305/323) | 3 | 0.954 | 232/76/15 | 0.07 (20) | 0.13 (39) | 85/14/1 | 0.95 / 2.22 / 0 |
| fusion_refit | 7172 | 7172 | 25766 | 3.59 | 0.975 (233/6) | 0.947 (306/323) | 3 | 0.957 | 232/77/14 | 0.07 (20) | 0.13 (39) | 86/13/1 | 0.84 / 2.52 / 0 |
| fusion_server | 11159 | 7171 | 29754 | 2.67 | 0.978 (271/6) | 0.944 (305/323) | 3 | 0.954 | 232/76/15 | 0.07 (20) | 0.13 (39) | 85/14/1 | 0.95 / 2.22 / 0 |
| fusion_server+attach | 7257 | 7171 | 29754 | 4.10 | 0.978 (271/6) | 0.944 (305/323) | 3 | 0.954 | 232/76/15 | 0.07 (20) | 0.13 (39) | 85/14/1 | 0.95 / 2.22 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 232 of this run's 323 pool ramps are self-detected, so 72% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 4952 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.2 m | 7.4 m | 0.09 | 0.02 | 0.00 |
| raycast | 3.2 m | 5.4 m | 0.01 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.845 | 96/273 | 70/100 | 0.830 | 49/268 | 68/100 |
| 5 | 0.944 | 91/305 | 85/100 | 0.944 | 39/305 | 85/100 |
| 7.5 | 0.972 | 89/314 | 90/100 | 0.972 | 35/314 | 90/100 |
| 10 | 0.972 | 89/314 | 90/100 | 0.972 | 35/314 | 90/100 |

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
| ps_raycast @ 2.5 m | 11523 | 25766 | 447.2 | server | 11523 | 0.821 | 0.889 | 0.957 | 11523 | 0.821 | 0.889 | 0.957 |
| ps_raycast @ 5 m | 8200 | 25766 | 318.2 | server | 8200 | 0.615 | 0.714 | 0.884 | 8200 | 0.615 | 0.714 | 0.884 |
| ps_raycast @ 7.5 m | 7468 | 25766 | 289.8 | server | 7468 | 0.567 | 0.632 | 0.845 | 7468 | 0.567 | 0.632 | 0.845 |
| ps_raycast @ 10 m | 7310 | 25766 | 283.7 | server | 7310 | 0.564 | 0.614 | 0.833 | 7310 | 0.564 | 0.614 | 0.833 |
| ps_raycast @ 12.5 m | 7212 | 25766 | 279.9 | server | 7212 | 0.556 | 0.608 | 0.827 | 7212 | 0.556 | 0.608 | 0.827 |
| ps_raycast @ 15 m | 7071 | 25766 | 274.4 | server | 7071 | 0.538 | 0.598 | 0.823 | 7071 | 0.538 | 0.598 | 0.823 |
| fusion | 7172 | 25766 | 278.4 | raycast | 7172 | 0.516 | 0.590 | 0.800 | 7172 | 0.516 | 0.590 | 0.800 |
| fusion_refit | 7172 | 25766 | 278.4 | raycast | 7172 | 0.517 | 0.591 | 0.799 | 7172 | 0.517 | 0.591 | 0.799 |
| fusion_server | 11159 | 29754 | 375.0 | server | 11159 | 0.750 | 0.852 | 0.948 | 7171 | 0.521 | 0.593 | 0.825 |
| fusion_server+attach | 7257 | 29754 | 243.9 | server | 7257 | 0.521 | 0.600 | 0.852 | 7171 | 0.518 | 0.593 | 0.848 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 3988 | 0.80 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 938 | 0.77 | 0.553 | 7 | 0.714 [0.36, 0.92] | 5 | 2 | 0 |
| ps @ 7.5 m | cluster of 2 | 2326 | 0.84 | 0.555 | 22 | 0.818 [0.61, 0.93] | 18 | 4 | 0 |
| ps @ 7.5 m | cluster of 3+ | 22502 | 0.88 | 0.568 | 217 | 1.000 [0.98, 1.00] | 209 | 0 | 8 |
| fusion_server | unplaceable | 3988 | 0.80 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server | cluster of 1 | 1242 | 0.75 | 0.568 | 11 | 0.636 [0.35, 0.85] | 7 | 4 | 0 |
| fusion_server | cluster of 2 | 1952 | 0.81 | 0.568 | 17 | 0.857 [0.60, 0.96] | 12 | 2 | 3 |
| fusion_server | cluster of 3+ | 22572 | 0.88 | 0.555 | 218 | 1.000 [0.98, 1.00] | 213 | 0 | 5 |
| fusion_server+attach | unplaceable | 3988 | 0.80 | 0.523 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 909 | 0.74 | 0.568 | 8 | 0.625 [0.31, 0.86] | 5 | 3 | 0 |
| fusion_server+attach | cluster of 2 | 1512 | 0.79 | 0.570 | 17 | 0.800 [0.55, 0.93] | 12 | 3 | 2 |
| fusion_server+attach | cluster of 3+ | 23345 | 0.88 | 0.568 | 221 | 1.000 [0.98, 1.00] | 215 | 0 | 6 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 3988; attached 3902 (0.98)
- clusters: 11159 (`fusion_server`) -> 7257 (`fusion_server+attach`)
- sanity (not a metric): 38 unplaceable labels are on judged panos; 37 of them attached (37 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
