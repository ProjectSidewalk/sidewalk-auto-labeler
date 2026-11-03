# laurens_gsv: PS label clustering vs RampNet GT (offline)

mode: offline -- 473 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (0 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `83f49aaecac1b37cf05f6687ac350694681832229f33107b42142437f43338e7`
inputs: streets sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`; verdicts sha256 `0f4608abcd6d380d388d668c4b2e48ebece2b058f8f934ab3b889d12ae9974e3`
raycast camera height auto; fusion arm at --min-confidence 0.55
- camera height mode `auto`: auto -> gsv-per-rig
- 2137 of 2137 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2008: median 1.97 m, 1 of 1 measured -> 2.5 m
  - 2021: median 2.41 m, 4 of 7 measured -> 2.5 m
  - 2024: median 2.40 m, 1412 of 2128 measured -> 2.5 m
  - 2025: median 2.06 m, 1 of 1 measured -> 2.5 m
GT: 86 judged panos -> 196 placeable points -> 193 ramps (3 cross-pano merges), 193 in the recall pool; raycast placed 1541 of 1656 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 0, 'out_of_range': 115})

## Data provenance

- results file `runs/laurens_gsv/results.jsonl`: sha256 `83f49aaecac1b37cf05f6687ac350694681832229f33107b42142437f43338e7`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 473 of 473
- `ps_streets.geojson`: 169 features, sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`, 2026-09-28T17:22:15+00:00 (7.5 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson; 168 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 59 blocks, largest 26 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 459 fusion members vs 459 placeable server labels; 0 of 459 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at auto vs runs/laurens_gsv/fusion_eval/report.md (published in the 2.6 m frame): precision 0.931, recall (union) 0.798, dual 20/11/2 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 473 AI (by account; 0 share a detection with another AI label) + 0 human labels on 230 panos (0 positioned by inverting their labels, 230 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 14 labels the raycast cannot place (range cap, horizon) are singleton clusters; 183 of its 183 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 473 label ids, 473 distinct, of 473 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 196 panos: median 0.011 m, p90 0.25 m; from all labels (the #105 method), over 196 panos: median 0.011 m, p90 0.25 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 292 | 282 | 473 | 1.62 | 0.933 (111/8) | 0.850 (164/193) | 0 | 0.850 | 108/56/29 | 0.33 (64) | 0.50 (109) | 25/6/2 | 0.41 / 1.48 / 0 |
| ps @ 5 m | 206 | 202 | 473 | 2.30 | 0.933 (111/8) | 0.808 (156/193) | 2 | 0.819 | 108/50/35 | 0.06 (10) | 0.15 (25) | 20/11/2 | 0.66 / 1.80 / 0 |
| ps @ 7.5 m | 174 | 171 | 473 | 2.72 | 0.933 (111/8) | 0.751 (145/193) | 4 | 0.772 | 108/41/44 | 0.01 (2) | 0.04 (6) | 17/14/2 | 0.80 / 2.70 / 1 |
| ps @ 10 m | 167 | 165 | 473 | 2.83 | 0.932 (110/8) | 0.720 (139/193) | 6 | 0.751 | 108/37/48 | 0.01 (2) | 0.03 (4) | 16/15/2 | 0.87 / 2.82 / 2 |
| ps @ 12.5 m | 165 | 163 | 473 | 2.87 | 0.940 (110/7) | 0.715 (138/193) | 6 | 0.746 | 108/36/49 | 0.01 (2) | 0.03 (4) | 16/15/2 | 0.90 / 2.92 / 1 |
| ps @ 15 m | 160 | 158 | 473 | 2.96 | 0.940 (110/7) | 0.694 (134/193) | 8 | 0.736 | 108/34/51 | 0.01 (2) | 0.03 (4) | 16/15/2 | 0.95 / 3.19 / 2 |
| ps_citywide @ 7.5 m | 174 | 171 | 473 | 2.72 | 0.933 (111/8) | 0.751 (145/193) | 4 | 0.772 | 108/41/44 | 0.01 (2) | 0.04 (6) | 17/14/2 | 0.80 / 2.70 / 1 |
| ps_placeable @ 2.5 m | 282 | 282 | 459 | 1.63 | 0.931 (108/8) | 0.850 (164/193) | 0 | 0.850 | 108/56/29 | 0.33 (64) | 0.50 (109) | 25/6/2 | 0.41 / 1.48 / 0 |
| ps_placeable @ 5 m | 199 | 199 | 459 | 2.31 | 0.931 (108/8) | 0.808 (156/193) | 2 | 0.819 | 108/50/35 | 0.06 (9) | 0.15 (23) | 20/11/2 | 0.66 / 1.89 / 0 |
| ps_placeable @ 7.5 m | 171 | 171 | 459 | 2.68 | 0.931 (108/8) | 0.751 (145/193) | 4 | 0.772 | 108/41/44 | 0.01 (2) | 0.04 (6) | 17/14/2 | 0.80 / 2.70 / 1 |
| ps_placeable @ 10 m | 165 | 165 | 459 | 2.78 | 0.930 (107/8) | 0.720 (139/193) | 6 | 0.751 | 108/37/48 | 0.01 (2) | 0.03 (4) | 16/15/2 | 0.87 / 2.82 / 2 |
| ps_placeable @ 12.5 m | 163 | 163 | 459 | 2.82 | 0.939 (107/7) | 0.715 (138/193) | 6 | 0.746 | 108/36/49 | 0.01 (2) | 0.03 (4) | 16/15/2 | 0.90 / 2.92 / 1 |
| ps_placeable @ 15 m | 158 | 158 | 459 | 2.91 | 0.939 (107/7) | 0.694 (134/193) | 8 | 0.736 | 108/34/51 | 0.01 (2) | 0.03 (4) | 16/15/2 | 0.95 / 3.19 / 2 |
| ps_raycast @ 2.5 m | 255 | 255 | 459 | 1.80 | 0.932 (109/8) | 0.845 (163/193) | 0 | 0.845 | 108/55/30 | 0.16 (27) | 0.39 (72) | 23/8/2 | 0.35 / 0.96 / 0 |
| ps_raycast @ 5 m | 189 | 189 | 459 | 2.43 | 0.931 (108/8) | 0.777 (150/193) | 3 | 0.793 | 108/45/40 | 0.02 (3) | 0.08 (12) | 20/11/2 | 0.66 / 1.73 / 0 |
| ps_raycast @ 7.5 m | 168 | 168 | 459 | 2.73 | 0.939 (108/7) | 0.741 (143/193) | 6 | 0.772 | 108/41/44 | 0.01 (2) | 0.03 (4) | 15/16/2 | 0.85 / 2.38 / 0 |
| ps_raycast @ 10 m | 165 | 165 | 459 | 2.78 | 0.939 (108/7) | 0.725 (140/193) | 6 | 0.756 | 108/38/47 | 0.01 (2) | 0.03 (4) | 15/16/2 | 0.87 / 2.39 / 0 |
| ps_raycast @ 12.5 m | 164 | 164 | 459 | 2.80 | 0.939 (107/7) | 0.720 (139/193) | 6 | 0.751 | 108/37/48 | 0.01 (2) | 0.03 (4) | 15/16/2 | 0.90 / 2.70 / 0 |
| ps_raycast @ 15 m | 159 | 159 | 459 | 2.89 | 0.939 (107/7) | 0.694 (134/193) | 8 | 0.736 | 108/34/51 | 0.01 (2) | 0.03 (4) | 15/16/2 | 0.99 / 3.02 / 1 |
| fusion | 183 | 183 | 459 | 2.51 | 0.931 (108/8) | 0.782 (151/193) | 3 | 0.798 | 108/46/39 | 0.03 (4) | 0.03 (4) | 20/11/2 | 0.73 / 1.80 / 0 |
| fusion_refit | 183 | 183 | 459 | 2.51 | 0.931 (108/8) | 0.782 (151/193) | 3 | 0.798 | 108/46/39 | 0.03 (4) | 0.03 (4) | 20/11/2 | 0.61 / 2.00 / 0 |
| fusion_server | 197 | 183 | 473 | 2.40 | 0.933 (111/8) | 0.782 (151/193) | 3 | 0.798 | 108/46/39 | 0.03 (4) | 0.03 (4) | 20/11/2 | 0.73 / 1.80 / 0 |
| fusion_server+attach | 185 | 183 | 473 | 2.56 | 0.932 (110/8) | 0.782 (151/193) | 3 | 0.798 | 108/46/39 | 0.03 (4) | 0.03 (4) | 20/11/2 | 0.73 / 1.80 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 108 of this run's 193 pool ramps are self-detected, so 56% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 85 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 3.9 m | 6.8 m | 0.06 | 0.00 | 0.00 |
| raycast | 3.1 m | 4.9 m | 0.00 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.663 | 7/128 | 13/33 | 0.741 | 4/143 | 17/33 |
| 5 | 0.751 | 6/145 | 17/33 | 0.782 | 4/151 | 20/33 |
| 7.5 | 0.762 | 6/147 | 17/33 | 0.793 | 4/153 | 20/33 |
| 10 | 0.767 | 6/148 | 18/33 | 0.798 | 4/154 | 21/33 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ps @ 2.5 m | 292 | 473 | 617.3 | server | 292 | 0.750 | 0.860 | 0.935 | 282 | 0.745 | 0.855 | 0.929 |
| ps @ 5 m | 206 | 473 | 435.5 | server | 206 | 0.374 | 0.641 | 0.879 | 202 | 0.381 | 0.634 | 0.866 |
| ps @ 7.5 m | 174 | 473 | 367.9 | server | 174 | 0.218 | 0.437 | 0.805 | 171 | 0.222 | 0.433 | 0.801 |
| ps @ 10 m | 167 | 473 | 353.1 | server | 167 | 0.228 | 0.389 | 0.772 | 165 | 0.230 | 0.394 | 0.770 |
| ps @ 12.5 m | 165 | 473 | 348.8 | server | 165 | 0.230 | 0.394 | 0.758 | 163 | 0.233 | 0.399 | 0.755 |
| ps @ 15 m | 160 | 473 | 338.3 | server | 160 | 0.237 | 0.406 | 0.756 | 158 | 0.241 | 0.411 | 0.753 |
| ps_citywide @ 7.5 m | 174 | 473 | 367.9 | server | 174 | 0.218 | 0.437 | 0.805 | 171 | 0.222 | 0.433 | 0.801 |
| ps_placeable @ 2.5 m | 282 | 459 | 614.4 | server | 282 | 0.745 | 0.855 | 0.929 | 282 | 0.745 | 0.855 | 0.929 |
| ps_placeable @ 5 m | 199 | 459 | 433.6 | server | 199 | 0.377 | 0.618 | 0.859 | 199 | 0.377 | 0.618 | 0.859 |
| ps_placeable @ 7.5 m | 171 | 459 | 372.5 | server | 171 | 0.234 | 0.427 | 0.801 | 171 | 0.234 | 0.427 | 0.801 |
| ps_placeable @ 10 m | 165 | 459 | 359.5 | server | 165 | 0.242 | 0.388 | 0.770 | 165 | 0.242 | 0.388 | 0.770 |
| ps_placeable @ 12.5 m | 163 | 459 | 355.1 | server | 163 | 0.245 | 0.393 | 0.755 | 163 | 0.245 | 0.393 | 0.755 |
| ps_placeable @ 15 m | 158 | 459 | 344.2 | server | 158 | 0.253 | 0.405 | 0.753 | 158 | 0.253 | 0.405 | 0.753 |
| ps_raycast @ 2.5 m | 255 | 459 | 555.6 | server | 255 | 0.647 | 0.804 | 0.902 | 255 | 0.647 | 0.804 | 0.902 |
| ps_raycast @ 5 m | 189 | 459 | 411.8 | server | 189 | 0.344 | 0.582 | 0.836 | 189 | 0.344 | 0.582 | 0.836 |
| ps_raycast @ 7.5 m | 168 | 459 | 366.0 | server | 168 | 0.250 | 0.435 | 0.774 | 168 | 0.250 | 0.435 | 0.774 |
| ps_raycast @ 10 m | 165 | 459 | 359.5 | server | 165 | 0.242 | 0.412 | 0.764 | 165 | 0.242 | 0.412 | 0.764 |
| ps_raycast @ 12.5 m | 164 | 459 | 357.3 | server | 164 | 0.244 | 0.402 | 0.756 | 164 | 0.244 | 0.402 | 0.756 |
| ps_raycast @ 15 m | 159 | 459 | 346.4 | server | 159 | 0.252 | 0.415 | 0.748 | 159 | 0.252 | 0.415 | 0.748 |
| fusion | 183 | 459 | 398.7 | raycast | 183 | 0.262 | 0.497 | 0.770 | 183 | 0.262 | 0.497 | 0.770 |
| fusion_refit | 183 | 459 | 398.7 | raycast | 183 | 0.273 | 0.497 | 0.770 | 183 | 0.273 | 0.497 | 0.770 |
| fusion_server | 197 | 473 | 416.5 | server | 197 | 0.381 | 0.599 | 0.863 | 183 | 0.311 | 0.541 | 0.831 |
| fusion_server+attach | 185 | 473 | 391.1 | server | 185 | 0.303 | 0.535 | 0.832 | 183 | 0.306 | 0.541 | 0.831 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 14 | 0.66 | 0.523 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 40 | 0.69 | 0.568 | 21 | 0.700 [0.48, 0.85] | 14 | 6 | 1 |
| ps @ 7.5 m | cluster of 2 | 71 | 0.70 | 0.568 | 24 | 0.958 [0.80, 0.99] | 23 | 1 | 0 |
| ps @ 7.5 m | cluster of 3+ | 348 | 0.79 | 0.570 | 73 | 0.986 [0.93, 1.00] | 71 | 1 | 1 |
| fusion_server | unplaceable | 14 | 0.66 | 0.523 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| fusion_server | cluster of 1 | 61 | 0.70 | 0.570 | 26 | 0.760 [0.57, 0.89] | 19 | 6 | 1 |
| fusion_server | cluster of 2 | 74 | 0.69 | 0.568 | 22 | 0.955 [0.78, 0.99] | 21 | 1 | 0 |
| fusion_server | cluster of 3+ | 324 | 0.79 | 0.570 | 70 | 0.986 [0.92, 1.00] | 68 | 1 | 1 |
| fusion_server+attach | unplaceable | 14 | 0.66 | 0.523 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 59 | 0.70 | 0.570 | 25 | 0.750 [0.55, 0.88] | 18 | 6 | 1 |
| fusion_server+attach | cluster of 2 | 72 | 0.70 | 0.555 | 22 | 0.955 [0.78, 0.99] | 21 | 1 | 0 |
| fusion_server+attach | cluster of 3+ | 328 | 0.79 | 0.568 | 71 | 0.986 [0.92, 1.00] | 69 | 1 | 1 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 14; attached 12 (0.86)
- clusters: 197 (`fusion_server`) -> 185 (`fusion_server+attach`)
- sanity (not a metric): 3 unplaceable labels are on judged panos; 2 of them attached (2 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
