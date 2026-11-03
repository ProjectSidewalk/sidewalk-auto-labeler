# bend: PS label clustering vs RampNet GT (offline)

mode: offline -- 51529 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (14 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153`
inputs: streets sha256 `none`; verdicts sha256 `9d9835db27f66c1904fcfd4fc87b06f26638a3c033ef10936387db40feb84b05`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 110 judged panos -> 297 placeable points -> 296 ramps (1 cross-pano merges), 296 in the recall pool; raycast placed 49701 of 51543 detections (drops {'below_floor': 0, 'on_rig': 14, 'horizon': 3, 'out_of_range': 1825})

## Data provenance

- results file `runs/bend/results.jsonl`: sha256 `1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 51529 of 51529
- regions: none (no server), so every label is in one region and per-region == citywide
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 3999 blocks, largest 69 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 49701 fusion members vs 49701 placeable server labels; 0 of 49701 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/bend/fusion_eval/report.md (published in the 2.6 m frame): precision 0.957, recall (union) 0.956, dual 25/6/0
- fusion_server input: 51529 AI (by account; 0 share a detection with another AI label) + 0 human labels on 20611 panos (0 positioned by inverting their labels, 20611 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 1828 labels the raycast cannot place (range cap, horizon) are singleton clusters; 14187 of its 14187 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 51529 label ids, 51529 distinct, of 51529 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 17883 panos: median 0.010 m, p90 0.20 m; from all labels (the #105 method), over 17883 panos: median 0.010 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 24372 | 23290 | 51529 | 2.11 | 0.957 (247/11) | 0.966 (286/296) | 0 | 0.966 | 242/44/10 | 0.51 (195) | 0.64 (308) | 27/4/0 | 0.59 / 2.23 / 0 |
| ps @ 5 m | 17093 | 16594 | 51529 | 3.01 | 0.957 (247/11) | 0.953 (282/296) | 2 | 0.959 | 242/42/12 | 0.22 (72) | 0.36 (134) | 24/7/0 | 0.93 / 2.38 / 1 |
| ps @ 7.5 m | 14650 | 14476 | 51529 | 3.52 | 0.957 (247/11) | 0.946 (280/296) | 3 | 0.956 | 242/41/13 | 0.11 (32) | 0.23 (71) | 24/7/0 | 0.98 / 2.56 / 1 |
| ps @ 10 m | 13915 | 13838 | 51529 | 3.70 | 0.957 (247/11) | 0.939 (278/296) | 3 | 0.949 | 242/39/15 | 0.07 (20) | 0.17 (48) | 23/8/0 | 0.99 / 2.59 / 1 |
| ps @ 12.5 m | 13633 | 13585 | 51529 | 3.78 | 0.957 (247/11) | 0.932 (276/296) | 4 | 0.946 | 242/38/16 | 0.06 (17) | 0.14 (42) | 22/8/1 | 1.02 / 2.74 / 2 |
| ps @ 15 m | 13423 | 13389 | 51529 | 3.84 | 0.957 (247/11) | 0.929 (275/296) | 4 | 0.943 | 242/37/17 | 0.06 (16) | 0.14 (40) | 21/9/1 | 1.03 / 2.74 / 2 |
| ps_citywide @ 7.5 m | 14650 | 14476 | 51529 | 3.52 | 0.957 (247/11) | 0.946 (280/296) | 3 | 0.956 | 242/41/13 | 0.11 (32) | 0.23 (71) | 24/7/0 | 0.98 / 2.56 / 1 |
| ps_placeable @ 2.5 m | 23193 | 23193 | 49701 | 2.14 | 0.957 (242/11) | 0.966 (286/296) | 0 | 0.966 | 242/44/10 | 0.50 (192) | 0.64 (305) | 27/4/0 | 0.59 / 2.23 / 0 |
| ps_placeable @ 5 m | 16429 | 16429 | 49701 | 3.03 | 0.957 (242/11) | 0.953 (282/296) | 2 | 0.959 | 242/42/12 | 0.20 (66) | 0.34 (124) | 24/7/0 | 0.93 / 2.41 / 1 |
| ps_placeable @ 7.5 m | 14326 | 14326 | 49701 | 3.47 | 0.957 (242/11) | 0.946 (280/296) | 3 | 0.956 | 242/41/13 | 0.09 (28) | 0.20 (63) | 24/7/0 | 0.99 / 2.56 / 2 |
| ps_placeable @ 10 m | 13726 | 13726 | 49701 | 3.62 | 0.957 (242/11) | 0.939 (278/296) | 3 | 0.949 | 242/39/15 | 0.06 (18) | 0.15 (44) | 23/8/0 | 1.00 / 2.59 / 2 |
| ps_placeable @ 12.5 m | 13487 | 13487 | 49701 | 3.69 | 0.957 (242/11) | 0.932 (276/296) | 4 | 0.946 | 242/38/16 | 0.06 (16) | 0.13 (39) | 22/8/1 | 1.03 / 2.74 / 3 |
| ps_placeable @ 15 m | 13294 | 13294 | 49701 | 3.74 | 0.957 (242/11) | 0.929 (275/296) | 4 | 0.943 | 242/37/17 | 0.06 (16) | 0.13 (39) | 21/9/1 | 1.04 / 2.74 / 3 |
| ps_raycast @ 2.5 m | 22496 | 22496 | 49701 | 2.21 | 0.957 (243/11) | 0.963 (285/296) | 0 | 0.963 | 242/43/11 | 0.34 (105) | 0.65 (279) | 26/5/0 | 0.47 / 1.07 / 0 |
| ps_raycast @ 5 m | 15728 | 15728 | 49701 | 3.16 | 0.957 (242/11) | 0.956 (283/296) | 1 | 0.959 | 242/42/12 | 0.04 (11) | 0.23 (68) | 26/5/0 | 0.89 / 2.14 / 0 |
| ps_raycast @ 7.5 m | 13848 | 13848 | 49701 | 3.59 | 0.957 (242/11) | 0.949 (281/296) | 2 | 0.956 | 242/41/13 | 0.03 (8) | 0.12 (35) | 25/6/0 | 0.96 / 2.56 / 1 |
| ps_raycast @ 10 m | 13384 | 13384 | 49701 | 3.71 | 0.957 (242/11) | 0.939 (278/296) | 2 | 0.946 | 242/38/16 | 0.03 (8) | 0.11 (32) | 24/7/0 | 0.98 / 2.63 / 1 |
| ps_raycast @ 12.5 m | 13222 | 13222 | 49701 | 3.76 | 0.957 (242/11) | 0.936 (277/296) | 2 | 0.943 | 242/37/17 | 0.03 (9) | 0.11 (30) | 24/6/1 | 0.97 / 2.66 / 1 |
| ps_raycast @ 15 m | 13056 | 13056 | 49701 | 3.81 | 0.957 (242/11) | 0.932 (276/296) | 3 | 0.943 | 242/37/17 | 0.03 (8) | 0.11 (29) | 23/7/1 | 1.00 / 2.74 / 2 |
| fusion | 14187 | 14187 | 49701 | 3.50 | 0.957 (242/11) | 0.953 (282/296) | 2 | 0.959 | 242/42/12 | 0.05 (15) | 0.14 (42) | 26/5/0 | 0.98 / 2.56 / 1 |
| fusion_refit | 14187 | 14187 | 49701 | 3.50 | 0.957 (242/11) | 0.949 (281/296) | 2 | 0.956 | 242/41/13 | 0.05 (15) | 0.16 (45) | 25/6/0 | 0.72 / 2.70 / 1 |
| fusion_server | 16015 | 14187 | 51529 | 3.22 | 0.957 (247/11) | 0.953 (282/296) | 2 | 0.959 | 242/42/12 | 0.05 (15) | 0.14 (42) | 26/5/0 | 0.98 / 2.56 / 1 |
| fusion_server+attach | 14223 | 14187 | 51529 | 3.62 | 0.957 (247/11) | 0.953 (282/296) | 2 | 0.959 | 242/42/12 | 0.05 (15) | 0.14 (42) | 26/5/0 | 0.98 / 2.56 / 1 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 242 of this run's 296 pool ramps are self-detected, so 82% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 9708 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 3.9 m | 7.7 m | 0.11 | 0.02 | 0.00 |
| raycast | 3.3 m | 5.9 m | 0.02 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.855 | 58/253 | 21/31 | 0.841 | 38/249 | 21/31 |
| 5 | 0.946 | 63/280 | 24/31 | 0.953 | 40/282 | 26/31 |
| 7.5 | 0.959 | 61/284 | 26/31 | 0.959 | 39/284 | 26/31 |
| 10 | 0.963 | 61/285 | 26/31 | 0.963 | 39/285 | 26/31 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ps @ 2.5 m | 24372 | 51529 | 473.0 | server | 24372 | 0.765 | 0.880 | 0.949 | 23290 | 0.754 | 0.868 | 0.944 |
| ps @ 5 m | 17093 | 51529 | 331.7 | server | 17093 | 0.402 | 0.720 | 0.895 | 16594 | 0.397 | 0.703 | 0.884 |
| ps @ 7.5 m | 14650 | 51529 | 284.3 | server | 14650 | 0.273 | 0.595 | 0.852 | 14476 | 0.274 | 0.591 | 0.846 |
| ps @ 10 m | 13915 | 51529 | 270.0 | server | 13915 | 0.267 | 0.556 | 0.830 | 13838 | 0.268 | 0.556 | 0.828 |
| ps @ 12.5 m | 13633 | 51529 | 264.6 | server | 13633 | 0.267 | 0.552 | 0.819 | 13585 | 0.267 | 0.551 | 0.818 |
| ps @ 15 m | 13423 | 51529 | 260.5 | server | 13423 | 0.270 | 0.550 | 0.814 | 13389 | 0.270 | 0.549 | 0.813 |
| ps_citywide @ 7.5 m | 14650 | 51529 | 284.3 | server | 14650 | 0.273 | 0.595 | 0.852 | 14476 | 0.274 | 0.591 | 0.846 |
| ps_placeable @ 2.5 m | 23193 | 49701 | 466.7 | server | 23193 | 0.752 | 0.867 | 0.944 | 23193 | 0.752 | 0.867 | 0.944 |
| ps_placeable @ 5 m | 16429 | 49701 | 330.6 | server | 16429 | 0.391 | 0.695 | 0.881 | 16429 | 0.391 | 0.695 | 0.881 |
| ps_placeable @ 7.5 m | 14326 | 49701 | 288.2 | server | 14326 | 0.273 | 0.582 | 0.839 | 14326 | 0.273 | 0.582 | 0.839 |
| ps_placeable @ 10 m | 13726 | 49701 | 276.2 | server | 13726 | 0.268 | 0.550 | 0.820 | 13726 | 0.268 | 0.550 | 0.820 |
| ps_placeable @ 12.5 m | 13487 | 49701 | 271.4 | server | 13487 | 0.268 | 0.547 | 0.810 | 13487 | 0.268 | 0.547 | 0.810 |
| ps_placeable @ 15 m | 13294 | 49701 | 267.5 | server | 13294 | 0.271 | 0.545 | 0.805 | 13294 | 0.271 | 0.545 | 0.805 |
| ps_raycast @ 2.5 m | 22496 | 49701 | 452.6 | server | 22496 | 0.761 | 0.853 | 0.938 | 22496 | 0.761 | 0.853 | 0.938 |
| ps_raycast @ 5 m | 15728 | 49701 | 316.5 | server | 15728 | 0.479 | 0.673 | 0.868 | 15728 | 0.479 | 0.673 | 0.868 |
| ps_raycast @ 7.5 m | 13848 | 49701 | 278.6 | server | 13848 | 0.333 | 0.570 | 0.825 | 13848 | 0.333 | 0.570 | 0.825 |
| ps_raycast @ 10 m | 13384 | 49701 | 269.3 | server | 13384 | 0.298 | 0.545 | 0.809 | 13384 | 0.298 | 0.545 | 0.809 |
| ps_raycast @ 12.5 m | 13222 | 49701 | 266.0 | server | 13222 | 0.296 | 0.540 | 0.802 | 13222 | 0.296 | 0.540 | 0.802 |
| ps_raycast @ 15 m | 13056 | 49701 | 262.7 | server | 13056 | 0.297 | 0.540 | 0.799 | 13056 | 0.297 | 0.540 | 0.799 |
| fusion | 14187 | 49701 | 285.4 | raycast | 14187 | 0.288 | 0.547 | 0.786 | 14187 | 0.288 | 0.547 | 0.786 |
| fusion_refit | 14187 | 49701 | 285.4 | raycast | 14187 | 0.283 | 0.555 | 0.794 | 14187 | 0.283 | 0.555 | 0.794 |
| fusion_server | 16015 | 51529 | 310.8 | server | 16015 | 0.437 | 0.674 | 0.874 | 14187 | 0.335 | 0.590 | 0.836 |
| fusion_server+attach | 14223 | 51529 | 276.0 | server | 14223 | 0.333 | 0.590 | 0.842 | 14187 | 0.332 | 0.590 | 0.841 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1828 | 0.74 | 0.523 | 5 | 1.000 [0.57, 1.00] | 5 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 2036 | 0.71 | 0.568 | 20 | 0.737 [0.51, 0.88] | 14 | 5 | 1 |
| ps @ 7.5 m | cluster of 2 | 4349 | 0.82 | 0.570 | 28 | 0.889 [0.72, 0.96] | 24 | 3 | 1 |
| ps @ 7.5 m | cluster of 3+ | 43316 | 0.88 | 0.568 | 212 | 0.986 [0.96, 1.00] | 205 | 3 | 4 |
| fusion_server | unplaceable | 1828 | 0.74 | 0.523 | 5 | 1.000 [0.57, 1.00] | 5 | 0 | 0 |
| fusion_server | cluster of 1 | 2533 | 0.73 | 0.568 | 22 | 0.650 [0.43, 0.82] | 13 | 7 | 2 |
| fusion_server | cluster of 2 | 3892 | 0.82 | 0.584 | 23 | 0.955 [0.78, 0.99] | 21 | 1 | 1 |
| fusion_server | cluster of 3+ | 43276 | 0.88 | 0.568 | 215 | 0.986 [0.96, 1.00] | 209 | 3 | 3 |
| fusion_server+attach | unplaceable | 1828 | 0.74 | 0.523 | 5 | 1.000 [0.57, 1.00] | 5 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 2459 | 0.73 | 0.568 | 21 | 0.684 [0.46, 0.85] | 13 | 6 | 2 |
| fusion_server+attach | cluster of 2 | 3759 | 0.82 | 0.584 | 22 | 0.905 [0.71, 0.97] | 19 | 2 | 1 |
| fusion_server+attach | cluster of 3+ | 43483 | 0.88 | 0.568 | 217 | 0.986 [0.96, 1.00] | 211 | 3 | 3 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1828; attached 1792 (0.98)
- clusters: 16015 (`fusion_server`) -> 14223 (`fusion_server+attach`)
- sanity (not a metric): 5 unplaceable labels are on judged panos; 5 of them attached (5 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
