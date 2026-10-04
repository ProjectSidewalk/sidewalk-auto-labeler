# gainesville: PS label clustering vs RampNet GT (offline)

mode: offline -- 15573 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (1 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
inputs: streets sha256 `39b778bb4a85bb8a3fa42f89adfff9ffc2b0615be403534b3bab3dd03b602ed3`; verdicts sha256 `776ddb2ca1a5a72fb9b27965ae8e5727d6fecc8bf4cc56b2ac94543c6cfad1e4`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 125 judged panos -> 219 placeable points -> 219 ramps (0 cross-pano merges), 219 in the recall pool; raycast placed 37056 of 43871 detections (drops {'below_floor': 0, 'on_rig': 35, 'horizon': 23, 'out_of_range': 6757})

## Data provenance

- results file `runs/gainesville/results.jsonl`: sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 15573 of 15573
- `ps_streets.geojson`: 27448 features, sha256 `39b778bb4a85bb8a3fa42f89adfff9ffc2b0615be403534b3bab3dd03b602ed3`, 2026-09-28T17:25:58+00:00 (6.1 days old at run time), from https://sidewalk-gainesville.cs.washington.edu/v3/api/streets?filetype=geojson; 6480 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2154 blocks, largest 61 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 14154 fusion members vs 14154 placeable server labels; 0 of 14154 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/gainesville/fusion_eval/report.md (published in the 2.6 m frame): precision 0.956, recall (union) 0.927, dual 6/3/0
- fusion_server input: 15573 AI (by account; 0 share a detection with another AI label) + 0 human labels on 8430 panos (0 positioned by inverting their labels, 8430 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 1419 labels the raycast cannot place (range cap, horizon) are singleton clusters; 6252 of its 6252 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 15573 label ids, 15573 distinct, of 15573 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 6419 panos in both (all labels, mostly AI): median 0.012 m, p90 0.33 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 8901 | 8278 | 15573 | 1.75 | 0.955 (191/9) | 0.927 (203/219) | 1 | 0.932 | 172/32/15 | 0.30 (77) | 0.57 (170) | 7/2/0 | 1.09 / 2.78 / 1 |
| ps @ 5 m | 6065 | 5780 | 15573 | 2.57 | 0.955 (190/9) | 0.890 (195/219) | 8 | 0.927 | 172/31/16 | 0.14 (34) | 0.32 (81) | 6/3/0 | 1.66 / 4.30 / 9 |
| ps @ 7.5 m | 5091 | 4948 | 15573 | 3.06 | 0.955 (190/9) | 0.877 (192/219) | 11 | 0.927 | 172/31/16 | 0.08 (18) | 0.19 (43) | 5/4/0 | 1.82 / 4.90 / 13 |
| ps @ 10 m | 4737 | 4648 | 15573 | 3.29 | 0.955 (190/9) | 0.872 (191/219) | 12 | 0.927 | 172/31/16 | 0.06 (13) | 0.16 (33) | 5/4/0 | 1.95 / 4.90 / 13 |
| ps @ 12.5 m | 4562 | 4502 | 15573 | 3.41 | 0.955 (190/9) | 0.863 (189/219) | 14 | 0.927 | 172/31/16 | 0.04 (8) | 0.13 (27) | 4/5/0 | 2.02 / 4.94 / 15 |
| ps @ 15 m | 4449 | 4395 | 15573 | 3.50 | 0.955 (190/9) | 0.845 (185/219) | 17 | 0.922 | 172/30/17 | 0.04 (8) | 0.12 (26) | 4/5/0 | 2.14 / 5.01 / 18 |
| ps_citywide @ 7.5 m | 4955 | 4822 | 15573 | 3.14 | 0.955 (190/9) | 0.872 (191/219) | 12 | 0.927 | 172/31/16 | 0.07 (14) | 0.18 (39) | 5/4/0 | 1.82 / 4.90 / 14 |
| ps_placeable @ 2.5 m | 8191 | 8191 | 14154 | 1.73 | 0.956 (174/8) | 0.927 (203/219) | 1 | 0.932 | 172/32/15 | 0.30 (76) | 0.56 (167) | 7/2/0 | 1.09 / 2.78 / 1 |
| ps_placeable @ 5 m | 5693 | 5693 | 14154 | 2.49 | 0.956 (173/8) | 0.890 (195/219) | 8 | 0.927 | 172/31/16 | 0.14 (34) | 0.31 (79) | 6/3/0 | 1.66 / 4.30 / 9 |
| ps_placeable @ 7.5 m | 4870 | 4870 | 14154 | 2.91 | 0.956 (173/8) | 0.877 (192/219) | 11 | 0.927 | 172/31/16 | 0.07 (15) | 0.18 (39) | 5/4/0 | 1.82 / 4.90 / 13 |
| ps_placeable @ 10 m | 4591 | 4591 | 14154 | 3.08 | 0.956 (173/8) | 0.872 (191/219) | 12 | 0.927 | 172/31/16 | 0.05 (11) | 0.14 (30) | 5/4/0 | 2.02 / 4.90 / 13 |
| ps_placeable @ 12.5 m | 4449 | 4449 | 14154 | 3.18 | 0.956 (173/8) | 0.863 (189/219) | 14 | 0.927 | 172/31/16 | 0.04 (8) | 0.13 (27) | 4/5/0 | 2.13 / 4.94 / 15 |
| ps_placeable @ 15 m | 4333 | 4333 | 14154 | 3.27 | 0.956 (173/8) | 0.840 (184/219) | 18 | 0.922 | 172/30/17 | 0.03 (7) | 0.11 (24) | 4/5/0 | 2.23 / 5.16 / 19 |
| ps_raycast @ 2.5 m | 9746 | 9746 | 14154 | 1.45 | 0.956 (174/8) | 0.945 (207/219) | 0 | 0.945 | 172/35/12 | 0.27 (64) | 0.64 (224) | 9/0/0 | 0.44 / 1.12 / 0 |
| ps_raycast @ 5 m | 6984 | 6984 | 14154 | 2.03 | 0.956 (174/8) | 0.932 (204/219) | 0 | 0.932 | 172/32/15 | 0.07 (14) | 0.31 (72) | 7/2/0 | 1.12 / 2.04 / 0 |
| ps_raycast @ 7.5 m | 5642 | 5642 | 14154 | 2.51 | 0.956 (173/8) | 0.922 (202/219) | 0 | 0.922 | 172/30/17 | 0.04 (8) | 0.11 (24) | 7/2/0 | 1.35 / 2.87 / 0 |
| ps_raycast @ 10 m | 4873 | 4873 | 14154 | 2.90 | 0.956 (173/8) | 0.890 (195/219) | 7 | 0.922 | 172/30/17 | 0.04 (7) | 0.09 (19) | 5/4/0 | 1.66 / 4.44 / 7 |
| ps_raycast @ 12.5 m | 4521 | 4521 | 14154 | 3.13 | 0.956 (173/8) | 0.872 (191/219) | 11 | 0.922 | 172/30/17 | 0.03 (6) | 0.09 (19) | 5/3/1 | 1.91 / 4.60 / 11 |
| ps_raycast @ 15 m | 4371 | 4371 | 14154 | 3.24 | 0.956 (173/8) | 0.858 (188/219) | 14 | 0.922 | 172/30/17 | 0.03 (6) | 0.10 (19) | 5/3/1 | 1.95 / 4.77 / 14 |
| fusion | 6252 | 6252 | 14154 | 2.26 | 0.956 (173/8) | 0.913 (200/219) | 3 | 0.927 | 172/31/16 | 0.03 (6) | 0.20 (44) | 6/3/0 | 1.15 / 2.52 / 3 |
| fusion_refit | 6252 | 6252 | 14154 | 2.26 | 0.956 (173/8) | 0.913 (200/219) | 3 | 0.927 | 172/31/16 | 0.03 (6) | 0.20 (46) | 6/3/0 | 0.99 / 2.93 / 4 |
| fusion_server | 7671 | 6252 | 15573 | 2.03 | 0.955 (190/9) | 0.913 (200/219) | 3 | 0.927 | 172/31/16 | 0.03 (6) | 0.20 (44) | 6/3/0 | 1.15 / 2.52 / 3 |
| fusion_server+attach | 6347 | 6252 | 15573 | 2.45 | 0.955 (190/9) | 0.913 (200/219) | 3 | 0.927 | 172/31/16 | 0.03 (6) | 0.20 (44) | 6/3/0 | 1.15 / 2.52 / 3 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 172 of this run's 219 pool ramps are self-detected, so 79% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 2056 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 3.8 m | 8.3 m | 0.13 | 0.05 | 0.00 |
| raycast | 4.4 m | 7.0 m | 0.07 | 0.01 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.603 | 29/132 | 4/9 | 0.785 | 33/172 | 5/9 |
| 5 | 0.877 | 36/192 | 5/9 | 0.913 | 40/200 | 6/9 |
| 7.5 | 0.963 | 33/211 | 9/9 | 0.954 | 40/209 | 8/9 |
| 10 | 0.977 | 33/214 | 9/9 | 0.982 | 38/215 | 9/9 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1419 | 0.74 | 0.523 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| ps @ 7.5 m | cluster of 1 | 1208 | 0.70 | 0.555 | 14 | 0.727 [0.43, 0.90] | 8 | 3 | 3 |
| ps @ 7.5 m | cluster of 2 | 1893 | 0.77 | 0.555 | 30 | 0.893 [0.73, 0.96] | 25 | 3 | 2 |
| ps @ 7.5 m | cluster of 3+ | 11053 | 0.82 | 0.555 | 143 | 0.986 [0.95, 1.00] | 139 | 2 | 2 |
| fusion_server | unplaceable | 1419 | 0.74 | 0.523 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| fusion_server | cluster of 1 | 2880 | 0.77 | 0.555 | 47 | 0.860 [0.73, 0.93] | 37 | 6 | 4 |
| fusion_server | cluster of 2 | 2632 | 0.79 | 0.555 | 34 | 0.969 [0.84, 0.99] | 31 | 1 | 2 |
| fusion_server | cluster of 3+ | 8642 | 0.82 | 0.555 | 106 | 0.990 [0.95, 1.00] | 104 | 1 | 1 |
| fusion_server+attach | unplaceable | 1419 | 0.74 | 0.523 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| fusion_server+attach | cluster of 1 | 2659 | 0.76 | 0.553 | 44 | 0.850 [0.71, 0.93] | 34 | 6 | 4 |
| fusion_server+attach | cluster of 2 | 2416 | 0.79 | 0.555 | 36 | 0.971 [0.85, 0.99] | 33 | 1 | 2 |
| fusion_server+attach | cluster of 3+ | 9079 | 0.82 | 0.555 | 107 | 0.991 [0.95, 1.00] | 105 | 1 | 1 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1419; attached 1324 (0.93)
- clusters: 7671 (`fusion_server`) -> 6347 (`fusion_server+attach`)
- sanity (not a metric): 18 unplaceable labels are on judged panos; 16 of them attached (16 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
