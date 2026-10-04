# clovis: PS label clustering vs RampNet GT (offline)

mode: offline -- 8961 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (0 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `f6a896f19f4c7036186b201bbd2a1bfa4d6a20f34c14f5586a72da936475c15d`
inputs: streets sha256 `none`; verdicts sha256 `2097b14f28b650a0b7fa06398ee1dfd893c0cac7d96549e2c2c08df17c3fd718`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 125 judged panos -> 174 placeable points -> 174 ramps (0 cross-pano merges), 174 in the recall pool; raycast placed 8626 of 8961 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 5, 'out_of_range': 330})

## Data provenance

- results file `runs/clovis/results.jsonl`: sha256 `f6a896f19f4c7036186b201bbd2a1bfa4d6a20f34c14f5586a72da936475c15d`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 8961 of 8961
- regions: none (no server), so every label is in one region and per-region == citywide
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 1415 blocks, largest 77 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 8626 fusion members vs 8626 placeable server labels; 0 of 8626 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/clovis/fusion_eval/report.md (published in the 2.6 m frame): precision 0.923, recall (union) 0.943, dual 2/0/0
- fusion_server input: 8961 AI (by account; 0 share a detection with another AI label) + 0 human labels on 6092 panos (0 positioned by inverting their labels, 6092 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 335 labels the raycast cannot place (range cap, horizon) are singleton clusters; 2484 of its 2498 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 8961 label ids, 8961 distinct, of 8961 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 5379 panos in both (all labels, mostly AI): median 0.012 m, p90 0.19 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 4518 | 4398 | 8961 | 1.98 | 0.914 (139/13) | 0.948 (165/174) | 1 | 0.954 | 132/34/8 | 0.48 (118) | 0.73 (231) | 2/0/0 | 0.67 / 2.49 / 1 |
| ps @ 5 m | 3210 | 3150 | 8961 | 2.79 | 0.914 (139/13) | 0.937 (163/174) | 1 | 0.943 | 132/32/10 | 0.19 (33) | 0.38 (81) | 2/0/0 | 1.08 / 3.25 / 2 |
| ps @ 7.5 m | 2686 | 2648 | 8961 | 3.34 | 0.914 (139/13) | 0.925 (161/174) | 2 | 0.937 | 132/31/11 | 0.04 (6) | 0.15 (28) | 2/0/0 | 1.34 / 3.41 / 3 |
| ps @ 10 m | 2460 | 2428 | 8961 | 3.64 | 0.914 (139/13) | 0.914 (159/174) | 4 | 0.937 | 132/31/11 | 0.03 (5) | 0.08 (13) | 2/0/0 | 1.38 / 3.65 / 5 |
| ps @ 12.5 m | 2354 | 2325 | 8961 | 3.81 | 0.914 (139/13) | 0.920 (160/174) | 3 | 0.937 | 132/31/11 | 0.03 (4) | 0.05 (8) | 2/0/0 | 1.39 / 3.70 / 4 |
| ps @ 15 m | 2274 | 2248 | 8961 | 3.94 | 0.914 (139/13) | 0.902 (157/174) | 5 | 0.931 | 132/30/12 | 0.02 (3) | 0.04 (7) | 2/0/0 | 1.46 / 3.78 / 6 |
| ps_citywide @ 7.5 m | 2686 | 2648 | 8961 | 3.34 | 0.914 (139/13) | 0.925 (161/174) | 2 | 0.937 | 132/31/11 | 0.04 (6) | 0.15 (28) | 2/0/0 | 1.34 / 3.41 / 3 |
| ps_placeable @ 2.5 m | 4381 | 4381 | 8626 | 1.97 | 0.923 (132/11) | 0.948 (165/174) | 1 | 0.954 | 132/34/8 | 0.48 (118) | 0.72 (229) | 2/0/0 | 0.67 / 2.49 / 1 |
| ps_placeable @ 5 m | 3132 | 3132 | 8626 | 2.75 | 0.923 (132/11) | 0.937 (163/174) | 1 | 0.943 | 132/32/10 | 0.18 (32) | 0.37 (79) | 2/0/0 | 1.08 / 3.25 / 2 |
| ps_placeable @ 7.5 m | 2634 | 2634 | 8626 | 3.27 | 0.923 (132/11) | 0.925 (161/174) | 2 | 0.937 | 132/31/11 | 0.03 (5) | 0.14 (27) | 2/0/0 | 1.34 / 3.59 / 3 |
| ps_placeable @ 10 m | 2423 | 2423 | 8626 | 3.56 | 0.923 (132/11) | 0.914 (159/174) | 4 | 0.937 | 132/31/11 | 0.03 (4) | 0.07 (11) | 2/0/0 | 1.38 / 3.70 / 5 |
| ps_placeable @ 12.5 m | 2317 | 2317 | 8626 | 3.72 | 0.923 (132/11) | 0.920 (160/174) | 3 | 0.937 | 132/31/11 | 0.02 (3) | 0.04 (6) | 2/0/0 | 1.39 / 3.78 / 4 |
| ps_placeable @ 15 m | 2237 | 2237 | 8626 | 3.86 | 0.923 (132/11) | 0.902 (157/174) | 5 | 0.931 | 132/30/12 | 0.01 (2) | 0.03 (5) | 2/0/0 | 1.46 / 3.84 / 6 |
| ps_raycast @ 2.5 m | 4719 | 4719 | 8626 | 1.83 | 0.923 (132/11) | 0.960 (167/174) | 0 | 0.960 | 132/35/7 | 0.39 (89) | 0.71 (224) | 2/0/0 | 0.46 / 1.01 / 0 |
| ps_raycast @ 5 m | 3358 | 3358 | 8626 | 2.57 | 0.923 (132/11) | 0.948 (165/174) | 0 | 0.948 | 132/33/9 | 0.05 (10) | 0.33 (65) | 2/0/0 | 0.92 / 2.17 / 0 |
| ps_raycast @ 7.5 m | 2783 | 2783 | 8626 | 3.10 | 0.923 (132/11) | 0.931 (162/174) | 1 | 0.937 | 132/31/11 | 0.04 (6) | 0.15 (24) | 2/0/0 | 1.38 / 2.81 / 1 |
| ps_raycast @ 10 m | 2484 | 2484 | 8626 | 3.47 | 0.923 (132/11) | 0.920 (160/174) | 3 | 0.937 | 132/31/11 | 0.01 (1) | 0.03 (5) | 2/0/0 | 1.30 / 3.41 / 3 |
| ps_raycast @ 12.5 m | 2343 | 2343 | 8626 | 3.68 | 0.923 (132/11) | 0.914 (159/174) | 4 | 0.937 | 132/31/11 | 0.01 (1) | 0.01 (2) | 2/0/0 | 1.34 / 3.41 / 4 |
| ps_raycast @ 15 m | 2251 | 2251 | 8626 | 3.83 | 0.923 (132/11) | 0.902 (157/174) | 5 | 0.931 | 132/30/12 | 0.01 (1) | 0.01 (2) | 2/0/0 | 1.34 / 3.41 / 5 |
| fusion | 2495 | 2495 | 8626 | 3.46 | 0.923 (132/11) | 0.908 (158/174) | 5 | 0.937 | 132/31/11 | 0.01 (2) | 0.02 (3) | 2/0/0 | 1.37 / 3.59 / 7 |
| fusion_refit | 2495 | 2495 | 8626 | 3.46 | 0.923 (132/11) | 0.914 (159/174) | 5 | 0.943 | 132/32/10 | 0.01 (2) | 0.03 (4) | 2/0/0 | 1.32 / 4.26 / 8 |
| fusion_server | 2833 | 2498 | 8961 | 3.16 | 0.914 (139/13) | 0.908 (158/174) | 5 | 0.937 | 132/31/11 | 0.01 (2) | 0.03 (4) | 2/0/0 | 1.37 / 3.93 / 7 |
| fusion_server+attach | 2564 | 2498 | 8961 | 3.49 | 0.914 (139/13) | 0.908 (158/174) | 5 | 0.937 | 132/31/11 | 0.01 (2) | 0.03 (4) | 2/0/0 | 1.37 / 3.93 / 7 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 132 of this run's 174 pool ramps are self-detected, so 76% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 1135 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 5.0 m | 9.1 m | 0.21 | 0.06 | 0.00 |
| raycast | 5.7 m | 9.3 m | 0.27 | 0.06 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.724 | 22/126 | 1/2 | 0.684 | 8/119 | 1/2 |
| 5 | 0.925 | 24/161 | 2/2 | 0.908 | 3/158 | 2/2 |
| 7.5 | 0.971 | 24/169 | 2/2 | 0.971 | 3/169 | 2/2 |
| 10 | 0.971 | 24/169 | 2/2 | 0.971 | 3/169 | 2/2 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 335 | 0.74 | 0.523 | 10 | 0.778 [0.45, 0.94] | 7 | 2 | 1 |
| ps @ 7.5 m | cluster of 1 | 941 | 0.67 | 0.568 | 13 | 0.700 [0.40, 0.89] | 7 | 3 | 3 |
| ps @ 7.5 m | cluster of 2 | 940 | 0.71 | 0.568 | 18 | 0.846 [0.58, 0.96] | 11 | 2 | 5 |
| ps @ 7.5 m | cluster of 3+ | 6745 | 0.81 | 0.568 | 123 | 0.950 [0.90, 0.98] | 114 | 6 | 3 |
| fusion_server | unplaceable | 335 | 0.74 | 0.523 | 10 | 0.778 [0.45, 0.94] | 7 | 2 | 1 |
| fusion_server | cluster of 1 | 936 | 0.66 | 0.568 | 17 | 0.733 [0.48, 0.89] | 11 | 4 | 2 |
| fusion_server | cluster of 2 | 856 | 0.70 | 0.568 | 15 | 0.909 [0.62, 0.98] | 10 | 1 | 4 |
| fusion_server | cluster of 3+ | 6834 | 0.81 | 0.568 | 122 | 0.949 [0.89, 0.98] | 111 | 6 | 5 |
| fusion_server+attach | unplaceable | 335 | 0.74 | 0.523 | 10 | 0.778 [0.45, 0.94] | 7 | 2 | 1 |
| fusion_server+attach | cluster of 1 | 914 | 0.66 | 0.568 | 16 | 0.714 [0.45, 0.88] | 10 | 4 | 2 |
| fusion_server+attach | cluster of 2 | 832 | 0.70 | 0.568 | 15 | 0.909 [0.62, 0.98] | 10 | 1 | 4 |
| fusion_server+attach | cluster of 3+ | 6880 | 0.81 | 0.570 | 123 | 0.949 [0.89, 0.98] | 112 | 6 | 5 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 335; attached 269 (0.80)
- clusters: 2833 (`fusion_server`) -> 2564 (`fusion_server+attach`)
- sanity (not a metric): 10 unplaceable labels are on judged panos; 8 of them attached (6 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
