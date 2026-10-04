# gainesville: PS label clustering vs RampNet GT (offline)

mode: offline -- 15573 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (1 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
inputs: streets sha256 `39b778bb4a85bb8a3fa42f89adfff9ffc2b0615be403534b3bab3dd03b602ed3`; verdicts sha256 `776ddb2ca1a5a72fb9b27965ae8e5727d6fecc8bf4cc56b2ac94543c6cfad1e4`
raycast camera height auto; fusion arm at --min-confidence 0.55
- camera height mode `auto`: auto -> gsv-per-rig
- 37435 of 37435 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 153 of 299 measured -> 2.5 m
  - 2008: median 2.42 m, 9 of 14 measured -> 2.5 m
  - 2011: median 2.12 m, 136 of 358 measured -> 2.5 m
  - 2014: median 2.29 m, 44 of 108 measured -> 2.5 m
  - 2015: median 1.94 m, 225 of 606 measured -> 2.5 m
  - 2016: median 2.13 m, 110 of 549 measured -> 2.5 m
  - 2017: median 2.24 m, 17 of 96 measured -> 2.5 m
  - 2018: median 2.07 m, 160 of 1059 measured -> 2.5 m
  - 2019: median 2.14 m, 64 of 304 measured -> 2.5 m
  - 2021: median 2.30 m, 334 of 368 measured -> 2.5 m
  - 2022: median 2.28 m, 2224 of 2592 measured -> 2.5 m
  - 2023: median 2.30 m, 1619 of 2140 measured -> 2.5 m
  - 2024: median 2.31 m, 2286 of 3679 measured -> 2.5 m
  - 2025: median 2.29 m, 1127 of 1497 measured -> 2.5 m
  - 2026: median 1.76 m, 21950 of 23766 measured -> 2 m
GT: 125 judged panos -> 230 placeable points -> 230 ramps (0 cross-pano merges), 230 in the recall pool; raycast placed 37056 of 43871 detections (drops {'below_floor': 0, 'on_rig': 35, 'horizon': 23, 'out_of_range': 6757})

## Data provenance

- results file `runs/gainesville/results.jsonl`: sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 15573 of 15573
- `ps_streets.geojson`: 27448 features, sha256 `39b778bb4a85bb8a3fa42f89adfff9ffc2b0615be403534b3bab3dd03b602ed3`, 2026-09-28T17:25:58+00:00 (6.1 days old at run time), from https://sidewalk-gainesville.cs.washington.edu/v3/api/streets?filetype=geojson; 6480 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2154 blocks, largest 61 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 14154 fusion members vs 14154 placeable server labels; 0 of 14154 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at auto vs runs/gainesville/fusion_eval/report.md (published in the 2.6 m frame): precision 0.956, recall (union) 0.943, dual 13/2/0 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 15573 AI (by account; 0 share a detection with another AI label) + 0 human labels on 8430 panos (0 positioned by inverting their labels, 8430 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 1419 labels the raycast cannot place (range cap, horizon) are singleton clusters; 4729 of its 4729 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 15573 label ids, 15573 distinct, of 15573 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 6419 panos in both (all labels, mostly AI): median 0.012 m, p90 0.33 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 8901 | 8278 | 15573 | 1.75 | 0.955 (191/9) | 0.948 (218/230) | 0 | 0.948 | 172/46/12 | 0.52 (154) | 0.72 (274) | 14/1/0 | 0.50 / 1.22 / 0 |
| ps @ 5 m | 6065 | 5780 | 15573 | 2.57 | 0.955 (190/9) | 0.943 (217/230) | 0 | 0.943 | 172/45/13 | 0.21 (49) | 0.42 (111) | 14/1/0 | 0.86 / 1.89 / 0 |
| ps @ 7.5 m | 5091 | 4948 | 15573 | 3.06 | 0.955 (190/9) | 0.939 (216/230) | 1 | 0.943 | 172/45/13 | 0.09 (21) | 0.20 (49) | 14/1/0 | 0.98 / 2.14 / 1 |
| ps @ 10 m | 4737 | 4648 | 15573 | 3.29 | 0.955 (190/9) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.09 (19) | 0.19 (42) | 13/2/0 | 1.03 / 2.24 / 1 |
| ps @ 12.5 m | 4562 | 4502 | 15573 | 3.41 | 0.955 (190/9) | 0.930 (214/230) | 3 | 0.943 | 172/45/13 | 0.07 (15) | 0.14 (33) | 13/2/0 | 1.03 / 2.27 / 2 |
| ps @ 15 m | 4449 | 4395 | 15573 | 3.50 | 0.955 (190/9) | 0.913 (210/230) | 5 | 0.935 | 172/43/15 | 0.07 (14) | 0.13 (30) | 13/2/0 | 1.05 / 2.71 / 4 |
| ps_citywide @ 7.5 m | 4955 | 4822 | 15573 | 3.14 | 0.955 (190/9) | 0.939 (216/230) | 1 | 0.943 | 172/45/13 | 0.08 (18) | 0.18 (43) | 14/1/0 | 0.98 / 2.14 / 1 |
| ps_placeable @ 2.5 m | 8191 | 8191 | 14154 | 1.73 | 0.956 (174/8) | 0.948 (218/230) | 0 | 0.948 | 172/46/12 | 0.51 (149) | 0.72 (268) | 14/1/0 | 0.48 / 1.22 / 0 |
| ps_placeable @ 5 m | 5693 | 5693 | 14154 | 2.49 | 0.956 (173/8) | 0.943 (217/230) | 0 | 0.943 | 172/45/13 | 0.20 (48) | 0.41 (108) | 14/1/0 | 0.87 / 1.89 / 0 |
| ps_placeable @ 7.5 m | 4870 | 4870 | 14154 | 2.91 | 0.956 (173/8) | 0.943 (217/230) | 0 | 0.943 | 172/45/13 | 0.08 (19) | 0.19 (45) | 14/1/0 | 0.98 / 2.20 / 0 |
| ps_placeable @ 10 m | 4591 | 4591 | 14154 | 3.08 | 0.956 (173/8) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.08 (17) | 0.17 (38) | 13/2/0 | 1.03 / 2.27 / 1 |
| ps_placeable @ 12.5 m | 4449 | 4449 | 14154 | 3.18 | 0.956 (173/8) | 0.930 (214/230) | 3 | 0.943 | 172/45/13 | 0.07 (15) | 0.14 (32) | 13/2/0 | 1.03 / 2.29 / 2 |
| ps_placeable @ 15 m | 4333 | 4333 | 14154 | 3.27 | 0.956 (173/8) | 0.913 (210/230) | 5 | 0.935 | 172/43/15 | 0.06 (12) | 0.11 (26) | 13/2/0 | 1.07 / 2.85 / 5 |
| ps_raycast @ 2.5 m | 7248 | 7248 | 14154 | 1.95 | 0.956 (174/8) | 0.943 (217/230) | 0 | 0.943 | 172/45/13 | 0.34 (86) | 0.62 (202) | 14/1/0 | 0.59 / 1.04 / 0 |
| ps_raycast @ 5 m | 5236 | 5236 | 14154 | 2.70 | 0.956 (173/8) | 0.943 (217/230) | 0 | 0.943 | 172/45/13 | 0.09 (20) | 0.26 (61) | 14/1/0 | 0.92 / 1.80 / 0 |
| ps_raycast @ 7.5 m | 4676 | 4676 | 14154 | 3.03 | 0.956 (173/8) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.07 (16) | 0.15 (35) | 13/2/0 | 0.97 / 2.08 / 1 |
| ps_raycast @ 10 m | 4506 | 4506 | 14154 | 3.14 | 0.956 (173/8) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.07 (15) | 0.14 (32) | 13/2/0 | 0.98 / 2.14 / 1 |
| ps_raycast @ 12.5 m | 4398 | 4398 | 14154 | 3.22 | 0.956 (173/8) | 0.926 (213/230) | 3 | 0.939 | 172/44/14 | 0.06 (13) | 0.13 (30) | 13/2/0 | 1.02 / 2.27 / 2 |
| ps_raycast @ 15 m | 4309 | 4309 | 14154 | 3.28 | 0.956 (173/8) | 0.909 (209/230) | 6 | 0.935 | 172/43/15 | 0.06 (13) | 0.11 (26) | 13/2/0 | 1.09 / 2.71 / 5 |
| fusion | 4729 | 4729 | 14154 | 2.99 | 0.956 (173/8) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.07 (17) | 0.15 (37) | 13/2/0 | 0.94 / 1.96 / 0 |
| fusion_refit | 4729 | 4729 | 14154 | 2.99 | 0.956 (173/8) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.07 (17) | 0.16 (39) | 13/2/0 | 0.83 / 2.05 / 0 |
| fusion_server | 6148 | 4729 | 15573 | 2.53 | 0.955 (190/9) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.07 (17) | 0.15 (37) | 13/2/0 | 0.94 / 1.96 / 0 |
| fusion_server+attach | 4812 | 4729 | 15573 | 3.24 | 0.955 (190/9) | 0.935 (215/230) | 2 | 0.943 | 172/45/13 | 0.07 (17) | 0.15 (37) | 13/2/0 | 0.94 / 1.96 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 172 of this run's 230 pool ramps are self-detected, so 75% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 2472 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.2 m | 7.5 m | 0.10 | 0.03 | 0.00 |
| raycast | 3.3 m | 5.8 m | 0.02 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.813 | 42/187 | 11/15 | 0.826 | 31/190 | 11/15 |
| 5 | 0.939 | 44/216 | 14/15 | 0.935 | 32/215 | 13/15 |
| 7.5 | 0.961 | 44/221 | 15/15 | 0.952 | 32/219 | 14/15 |
| 10 | 0.970 | 43/223 | 15/15 | 0.965 | 32/222 | 15/15 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1419 | 0.74 | 0.523 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| ps @ 7.5 m | cluster of 1 | 1208 | 0.70 | 0.555 | 14 | 0.727 [0.43, 0.90] | 8 | 3 | 3 |
| ps @ 7.5 m | cluster of 2 | 1893 | 0.77 | 0.555 | 30 | 0.893 [0.73, 0.96] | 25 | 3 | 2 |
| ps @ 7.5 m | cluster of 3+ | 11053 | 0.82 | 0.555 | 143 | 0.986 [0.95, 1.00] | 139 | 2 | 2 |
| fusion_server | unplaceable | 1419 | 0.74 | 0.523 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| fusion_server | cluster of 1 | 1408 | 0.69 | 0.555 | 21 | 0.750 [0.51, 0.90] | 12 | 4 | 5 |
| fusion_server | cluster of 2 | 1698 | 0.77 | 0.568 | 29 | 0.893 [0.73, 0.96] | 25 | 3 | 1 |
| fusion_server | cluster of 3+ | 11048 | 0.83 | 0.555 | 137 | 0.993 [0.96, 1.00] | 135 | 1 | 1 |
| fusion_server+attach | unplaceable | 1419 | 0.74 | 0.523 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| fusion_server+attach | cluster of 1 | 1284 | 0.69 | 0.555 | 19 | 0.733 [0.48, 0.89] | 11 | 4 | 4 |
| fusion_server+attach | cluster of 2 | 1555 | 0.76 | 0.568 | 29 | 0.889 [0.72, 0.96] | 24 | 3 | 2 |
| fusion_server+attach | cluster of 3+ | 11315 | 0.83 | 0.555 | 139 | 0.993 [0.96, 1.00] | 137 | 1 | 1 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1419; attached 1336 (0.94)
- clusters: 6148 (`fusion_server`) -> 4812 (`fusion_server+attach`)
- sanity (not a metric): 18 unplaceable labels are on judged panos; 17 of them attached (17 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
