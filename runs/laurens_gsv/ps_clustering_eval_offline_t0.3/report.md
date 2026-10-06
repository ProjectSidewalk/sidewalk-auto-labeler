# laurens_gsv: PS label clustering vs RampNet GT (offline)

mode: offline -- 865 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (0 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `83f49aaecac1b37cf05f6687ac350694681832229f33107b42142437f43338e7`
inputs: streets sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`; verdicts sha256 `0f4608abcd6d380d388d668c4b2e48ebece2b058f8f934ab3b889d12ae9974e3`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 86 judged panos -> 195 placeable points -> 190 ramps (5 cross-pano merges), 190 in the recall pool; raycast placed 1541 of 1656 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 0, 'out_of_range': 115})

## Data provenance

- results file `runs/laurens_gsv/results.jsonl`: sha256 `83f49aaecac1b37cf05f6687ac350694681832229f33107b42142437f43338e7`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 865 of 865
- `ps_streets.geojson`: 169 features, sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`, 2026-09-28T17:22:15+00:00 (7.5 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson; 168 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 77 blocks, largest 39 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 825 fusion members vs 825 placeable server labels; 0 of 825 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/laurens_gsv/fusion_eval/report.md (published in the 2.6 m frame): precision 0.931, recall (union) 0.942, dual 21/6/0
- fusion_server input: 865 AI (by account; 0 share a detection with another AI label) + 0 human labels on 351 panos (0 positioned by inverting their labels, 351 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 40 labels the raycast cannot place (range cap, horizon) are singleton clusters; 264 of its 264 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 865 label ids, 865 distinct, of 865 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 293 panos: median 0.011 m, p90 0.20 m; from all labels (the #105 method), over 293 panos: median 0.011 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 494 | 469 | 865 | 1.75 | 0.933 (111/8) | 0.979 (186/190) | 0 | 0.979 | 107/79/4 | 0.59 (147) | 0.76 (231) | 25/2/0 | 0.67 / 2.07 / 1 |
| ps @ 5 m | 341 | 328 | 865 | 2.54 | 0.933 (111/8) | 0.963 (183/190) | 1 | 0.968 | 107/77/6 | 0.27 (50) | 0.42 (89) | 23/4/0 | 0.79 / 2.57 / 1 |
| ps @ 7.5 m | 272 | 267 | 865 | 3.18 | 0.933 (111/8) | 0.921 (175/190) | 1 | 0.926 | 107/69/14 | 0.10 (17) | 0.19 (34) | 21/6/0 | 1.00 / 3.29 / 1 |
| ps @ 10 m | 250 | 249 | 865 | 3.46 | 0.932 (110/8) | 0.884 (168/190) | 2 | 0.895 | 107/63/20 | 0.07 (11) | 0.12 (20) | 19/8/0 | 1.11 / 3.47 / 2 |
| ps @ 12.5 m | 246 | 245 | 865 | 3.52 | 0.940 (110/7) | 0.879 (167/190) | 2 | 0.889 | 107/62/21 | 0.06 (10) | 0.11 (19) | 19/8/0 | 1.14 / 3.47 / 2 |
| ps @ 15 m | 242 | 241 | 865 | 3.57 | 0.940 (110/7) | 0.868 (165/190) | 2 | 0.879 | 107/60/23 | 0.06 (10) | 0.12 (19) | 18/9/0 | 1.14 / 3.47 / 2 |
| ps_citywide @ 7.5 m | 272 | 267 | 865 | 3.18 | 0.933 (111/8) | 0.921 (175/190) | 1 | 0.926 | 107/69/14 | 0.10 (17) | 0.19 (34) | 21/6/0 | 1.00 / 3.29 / 1 |
| ps_placeable @ 2.5 m | 467 | 467 | 825 | 1.77 | 0.931 (108/8) | 0.979 (186/190) | 0 | 0.979 | 107/79/4 | 0.59 (146) | 0.76 (229) | 25/2/0 | 0.68 / 2.09 / 1 |
| ps_placeable @ 5 m | 323 | 323 | 825 | 2.55 | 0.931 (108/8) | 0.958 (182/190) | 1 | 0.963 | 107/76/7 | 0.25 (47) | 0.41 (85) | 23/4/0 | 0.81 / 2.66 / 1 |
| ps_placeable @ 7.5 m | 263 | 263 | 825 | 3.14 | 0.931 (108/8) | 0.921 (175/190) | 1 | 0.926 | 107/69/14 | 0.09 (15) | 0.17 (30) | 21/6/0 | 0.90 / 3.31 / 1 |
| ps_placeable @ 10 m | 249 | 249 | 825 | 3.31 | 0.930 (107/8) | 0.884 (168/190) | 2 | 0.895 | 107/63/20 | 0.07 (11) | 0.12 (20) | 19/8/0 | 1.00 / 3.47 / 2 |
| ps_placeable @ 12.5 m | 245 | 245 | 825 | 3.37 | 0.939 (107/7) | 0.874 (166/190) | 3 | 0.889 | 107/62/21 | 0.07 (11) | 0.12 (20) | 19/8/0 | 1.01 / 3.47 / 2 |
| ps_placeable @ 15 m | 241 | 241 | 825 | 3.42 | 0.939 (107/7) | 0.863 (164/190) | 3 | 0.879 | 107/60/23 | 0.07 (11) | 0.12 (20) | 18/9/0 | 1.01 / 3.47 / 2 |
| ps_raycast @ 2.5 m | 411 | 411 | 825 | 2.01 | 0.931 (108/8) | 0.984 (187/190) | 0 | 0.984 | 107/80/3 | 0.30 (67) | 0.58 (148) | 26/1/0 | 0.46 / 1.06 / 0 |
| ps_raycast @ 5 m | 285 | 285 | 825 | 2.89 | 0.931 (108/8) | 0.937 (178/190) | 2 | 0.947 | 107/73/10 | 0.05 (9) | 0.16 (29) | 21/6/0 | 0.80 / 2.07 / 0 |
| ps_raycast @ 7.5 m | 251 | 251 | 825 | 3.29 | 0.931 (108/8) | 0.911 (173/190) | 2 | 0.921 | 107/68/15 | 0.03 (5) | 0.06 (10) | 19/8/0 | 0.90 / 2.45 / 0 |
| ps_raycast @ 10 m | 241 | 241 | 825 | 3.42 | 0.939 (108/7) | 0.905 (172/190) | 2 | 0.916 | 107/67/16 | 0.03 (5) | 0.06 (10) | 19/8/0 | 0.97 / 2.66 / 0 |
| ps_raycast @ 12.5 m | 238 | 238 | 825 | 3.47 | 0.939 (108/7) | 0.905 (172/190) | 2 | 0.916 | 107/67/16 | 0.03 (6) | 0.06 (10) | 19/8/0 | 1.00 / 2.73 / 0 |
| ps_raycast @ 15 m | 234 | 234 | 825 | 3.53 | 0.939 (108/7) | 0.895 (170/190) | 2 | 0.905 | 107/65/18 | 0.04 (6) | 0.06 (10) | 19/8/0 | 1.00 / 2.76 / 0 |
| fusion | 264 | 264 | 825 | 3.12 | 0.931 (108/8) | 0.937 (178/190) | 1 | 0.942 | 107/72/11 | 0.04 (8) | 0.09 (16) | 21/6/0 | 0.88 / 2.15 / 0 |
| fusion_refit | 264 | 264 | 825 | 3.12 | 0.931 (108/8) | 0.937 (178/190) | 1 | 0.942 | 107/72/11 | 0.04 (7) | 0.09 (16) | 21/6/0 | 0.73 / 2.61 / 0 |
| fusion_server | 304 | 264 | 865 | 2.85 | 0.933 (111/8) | 0.937 (178/190) | 1 | 0.942 | 107/72/11 | 0.04 (8) | 0.09 (16) | 21/6/0 | 0.88 / 2.15 / 0 |
| fusion_server+attach | 265 | 264 | 865 | 3.26 | 0.932 (110/8) | 0.937 (178/190) | 1 | 0.942 | 107/72/11 | 0.04 (8) | 0.09 (16) | 21/6/0 | 0.88 / 2.15 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 107 of this run's 190 pool ramps are self-detected, so 56% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 150 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.8 m | 8.4 m | 0.18 | 0.04 | 0.00 |
| raycast | 3.4 m | 6.0 m | 0.03 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.784 | 35/149 | 12/27 | 0.842 | 16/160 | 17/27 |
| 5 | 0.921 | 33/175 | 21/27 | 0.937 | 16/178 | 21/27 |
| 7.5 | 0.932 | 32/177 | 22/27 | 0.942 | 16/179 | 22/27 |
| 10 | 0.932 | 32/177 | 22/27 | 0.942 | 16/179 | 22/27 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ps @ 2.5 m | 494 | 865 | 571.1 | server | 494 | 0.844 | 0.915 | 0.937 | 469 | 0.844 | 0.908 | 0.936 |
| ps @ 5 m | 341 | 865 | 394.2 | server | 341 | 0.548 | 0.792 | 0.886 | 328 | 0.555 | 0.777 | 0.875 |
| ps @ 7.5 m | 272 | 865 | 314.5 | server | 272 | 0.324 | 0.647 | 0.831 | 267 | 0.330 | 0.640 | 0.824 |
| ps @ 10 m | 250 | 865 | 289.0 | server | 250 | 0.316 | 0.580 | 0.804 | 249 | 0.317 | 0.582 | 0.807 |
| ps @ 12.5 m | 246 | 865 | 284.4 | server | 246 | 0.305 | 0.581 | 0.793 | 245 | 0.306 | 0.584 | 0.796 |
| ps @ 15 m | 242 | 865 | 279.8 | server | 242 | 0.310 | 0.566 | 0.785 | 241 | 0.311 | 0.568 | 0.788 |
| ps_citywide @ 7.5 m | 272 | 865 | 314.5 | server | 272 | 0.324 | 0.647 | 0.831 | 267 | 0.330 | 0.640 | 0.824 |
| ps_placeable @ 2.5 m | 467 | 825 | 566.1 | server | 467 | 0.844 | 0.908 | 0.936 | 467 | 0.844 | 0.908 | 0.936 |
| ps_placeable @ 5 m | 323 | 825 | 391.5 | server | 323 | 0.542 | 0.762 | 0.873 | 323 | 0.542 | 0.762 | 0.873 |
| ps_placeable @ 7.5 m | 263 | 825 | 318.8 | server | 263 | 0.338 | 0.620 | 0.814 | 263 | 0.338 | 0.620 | 0.814 |
| ps_placeable @ 10 m | 249 | 825 | 301.8 | server | 249 | 0.337 | 0.578 | 0.791 | 249 | 0.337 | 0.578 | 0.791 |
| ps_placeable @ 12.5 m | 245 | 825 | 297.0 | server | 245 | 0.327 | 0.580 | 0.780 | 245 | 0.327 | 0.580 | 0.780 |
| ps_placeable @ 15 m | 241 | 825 | 292.1 | server | 241 | 0.332 | 0.564 | 0.768 | 241 | 0.332 | 0.564 | 0.768 |
| ps_raycast @ 2.5 m | 411 | 825 | 498.2 | server | 411 | 0.764 | 0.873 | 0.920 | 411 | 0.764 | 0.873 | 0.920 |
| ps_raycast @ 5 m | 285 | 825 | 345.5 | server | 285 | 0.509 | 0.698 | 0.842 | 285 | 0.509 | 0.698 | 0.842 |
| ps_raycast @ 7.5 m | 251 | 825 | 304.2 | server | 251 | 0.402 | 0.578 | 0.805 | 251 | 0.402 | 0.578 | 0.805 |
| ps_raycast @ 10 m | 241 | 825 | 292.1 | server | 241 | 0.382 | 0.552 | 0.784 | 241 | 0.382 | 0.552 | 0.784 |
| ps_raycast @ 12.5 m | 238 | 825 | 288.5 | server | 238 | 0.378 | 0.546 | 0.773 | 238 | 0.378 | 0.546 | 0.773 |
| ps_raycast @ 15 m | 234 | 825 | 283.6 | server | 234 | 0.359 | 0.543 | 0.769 | 234 | 0.359 | 0.543 | 0.769 |
| fusion | 264 | 825 | 320.0 | raycast | 264 | 0.356 | 0.621 | 0.803 | 264 | 0.356 | 0.621 | 0.803 |
| fusion_refit | 264 | 825 | 320.0 | raycast | 264 | 0.352 | 0.621 | 0.807 | 264 | 0.352 | 0.621 | 0.807 |
| fusion_server | 304 | 865 | 351.4 | server | 304 | 0.526 | 0.727 | 0.868 | 264 | 0.424 | 0.636 | 0.837 |
| fusion_server+attach | 265 | 865 | 306.4 | server | 265 | 0.400 | 0.649 | 0.845 | 264 | 0.402 | 0.652 | 0.848 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 40 | 0.49 | 0.523 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 47 | 0.44 | 0.553 | 10 | 0.600 [0.31, 0.83] | 6 | 4 | 0 |
| ps @ 7.5 m | cluster of 2 | 114 | 0.51 | 0.555 | 17 | 0.938 [0.72, 0.99] | 15 | 1 | 1 |
| ps @ 7.5 m | cluster of 3+ | 664 | 0.63 | 0.570 | 91 | 0.967 [0.91, 0.99] | 87 | 3 | 1 |
| fusion_server | unplaceable | 40 | 0.49 | 0.523 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| fusion_server | cluster of 1 | 75 | 0.46 | 0.555 | 10 | 0.600 [0.31, 0.83] | 6 | 4 | 0 |
| fusion_server | cluster of 2 | 78 | 0.46 | 0.568 | 15 | 0.857 [0.60, 0.96] | 12 | 2 | 1 |
| fusion_server | cluster of 3+ | 672 | 0.63 | 0.568 | 93 | 0.978 [0.92, 0.99] | 90 | 2 | 1 |
| fusion_server+attach | unplaceable | 40 | 0.49 | 0.523 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 72 | 0.46 | 0.555 | 10 | 0.600 [0.31, 0.83] | 6 | 4 | 0 |
| fusion_server+attach | cluster of 2 | 73 | 0.46 | 0.568 | 15 | 0.857 [0.60, 0.96] | 12 | 2 | 1 |
| fusion_server+attach | cluster of 3+ | 680 | 0.63 | 0.568 | 93 | 0.978 [0.92, 0.99] | 90 | 2 | 1 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 40; attached 39 (0.97)
- clusters: 304 (`fusion_server`) -> 265 (`fusion_server+attach`)
- sanity (not a metric): 3 unplaceable labels are on judged panos; 3 of them attached (3 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
