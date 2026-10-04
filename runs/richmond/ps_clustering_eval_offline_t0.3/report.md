# richmond: PS label clustering vs RampNet GT (offline)

mode: offline -- 9524 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (2 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
inputs: streets sha256 `02f3e0061c16ecda5d6ba5523f25bbff5999d758061cb75dd408edb732f92206`; verdicts sha256 `3721a2ac75a056fdd1e3ab45d9fff33e118d5851ae9c50a3a16e42ae2e15562d`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 124 judged panos -> 253 placeable points -> 253 ramps (0 cross-pano merges), 253 in the recall pool; raycast placed 8096 of 9526 detections (drops {'below_floor': 0, 'on_rig': 2, 'horizon': 236, 'out_of_range': 1192})

## Data provenance

- results file `runs/richmond/results.jsonl`: sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 9524 of 9524
- `ps_streets.geojson`: 16365 features, sha256 `02f3e0061c16ecda5d6ba5523f25bbff5999d758061cb75dd408edb732f92206`, 2026-09-28T17:38:33+00:00 (6.1 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/streets?filetype=geojson; 704 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 406 blocks, largest 141 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 8096 fusion members vs 8096 placeable server labels; 0 of 8096 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/richmond/fusion_eval/report.md (published in the 2.6 m frame): precision 0.959, recall (union) 0.941, dual 23/4/3
- fusion_server input: 9524 AI (by account; 0 share a detection with another AI label) + 0 human labels on 3684 panos (0 positioned by inverting their labels, 3684 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 1428 labels the raycast cannot place (range cap, horizon) are singleton clusters; 1558 of its 1569 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 9524 label ids, 9524 distinct, of 9524 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 3022 panos in both (all labels, mostly AI): median 0.010 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 4513 | 3906 | 9524 | 2.11 | 0.964 (238/9) | 0.964 (244/253) | 1 | 0.968 | 210/35/8 | 0.61 (295) | 0.80 (542) | 27/1/2 | 0.80 / 2.57 / 2 |
| ps @ 5 m | 2849 | 2564 | 9524 | 3.34 | 0.964 (238/9) | 0.937 (237/253) | 6 | 0.960 | 210/33/10 | 0.38 (113) | 0.62 (235) | 25/2/3 | 1.40 / 3.73 / 8 |
| ps @ 7.5 m | 2147 | 1985 | 9524 | 4.44 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (59) | 0.47 (130) | 23/4/3 | 1.52 / 3.98 / 11 |
| ps @ 10 m | 1771 | 1674 | 9524 | 5.38 | 0.963 (237/9) | 0.889 (225/253) | 14 | 0.945 | 210/29/14 | 0.15 (33) | 0.32 (81) | 22/5/3 | 1.80 / 4.43 / 15 |
| ps @ 12.5 m | 1583 | 1514 | 9524 | 6.02 | 0.963 (237/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.14 (31) | 0.29 (68) | 22/5/3 | 1.86 / 4.57 / 18 |
| ps @ 15 m | 1486 | 1433 | 9524 | 6.41 | 0.963 (237/9) | 0.874 (221/253) | 17 | 0.941 | 210/28/15 | 0.13 (28) | 0.26 (60) | 22/5/3 | 1.95 / 4.57 / 18 |
| ps_citywide @ 7.5 m | 2113 | 1954 | 9524 | 4.51 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (58) | 0.46 (126) | 23/4/3 | 1.69 / 4.00 / 11 |
| ps_placeable @ 2.5 m | 3841 | 3841 | 8096 | 2.11 | 0.959 (211/9) | 0.960 (243/253) | 1 | 0.964 | 210/34/9 | 0.60 (289) | 0.78 (526) | 26/2/2 | 0.81 / 2.62 / 2 |
| ps_placeable @ 5 m | 2474 | 2474 | 8096 | 3.27 | 0.959 (211/9) | 0.929 (235/253) | 7 | 0.957 | 210/32/11 | 0.34 (101) | 0.60 (221) | 25/2/3 | 1.40 / 3.68 / 9 |
| ps_placeable @ 7.5 m | 1885 | 1885 | 8096 | 4.29 | 0.959 (211/9) | 0.897 (227/253) | 13 | 0.949 | 210/30/13 | 0.20 (46) | 0.44 (113) | 23/4/3 | 1.54 / 4.06 / 14 |
| ps_placeable @ 10 m | 1588 | 1588 | 8096 | 5.10 | 0.959 (211/9) | 0.893 (226/253) | 14 | 0.949 | 210/30/13 | 0.11 (25) | 0.29 (70) | 23/4/3 | 1.80 / 4.50 / 15 |
| ps_placeable @ 12.5 m | 1459 | 1459 | 8096 | 5.55 | 0.959 (211/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.12 (26) | 0.27 (61) | 23/4/3 | 1.78 / 4.57 / 18 |
| ps_placeable @ 15 m | 1383 | 1383 | 8096 | 5.85 | 0.959 (211/9) | 0.874 (221/253) | 16 | 0.937 | 210/27/16 | 0.11 (24) | 0.24 (54) | 23/4/3 | 1.89 / 4.51 / 17 |
| ps_raycast @ 2.5 m | 3910 | 3910 | 8096 | 2.07 | 0.959 (211/9) | 0.968 (245/253) | 0 | 0.968 | 210/35/8 | 0.56 (211) | 0.81 (477) | 27/1/2 | 0.48 / 1.14 / 0 |
| ps_raycast @ 5 m | 2582 | 2582 | 8096 | 3.14 | 0.959 (211/9) | 0.957 (242/253) | 0 | 0.957 | 210/32/11 | 0.17 (45) | 0.53 (169) | 24/3/3 | 1.05 / 2.17 / 0 |
| ps_raycast @ 7.5 m | 1977 | 1977 | 8096 | 4.10 | 0.959 (211/9) | 0.949 (240/253) | 1 | 0.953 | 210/31/12 | 0.07 (18) | 0.31 (78) | 23/4/3 | 1.34 / 2.72 / 0 |
| ps_raycast @ 10 m | 1637 | 1637 | 8096 | 4.95 | 0.959 (211/9) | 0.913 (231/253) | 8 | 0.945 | 210/29/14 | 0.06 (16) | 0.22 (52) | 23/4/3 | 1.76 / 3.75 / 8 |
| ps_raycast @ 12.5 m | 1487 | 1487 | 8096 | 5.44 | 0.959 (211/9) | 0.889 (225/253) | 13 | 0.941 | 210/28/15 | 0.06 (15) | 0.19 (44) | 23/4/3 | 1.80 / 4.43 / 13 |
| ps_raycast @ 15 m | 1395 | 1395 | 8096 | 5.80 | 0.959 (211/9) | 0.874 (221/253) | 17 | 0.941 | 210/28/15 | 0.06 (15) | 0.17 (40) | 23/4/3 | 1.86 / 4.98 / 18 |
| fusion | 1569 | 1569 | 8096 | 5.16 | 0.959 (211/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.06 (14) | 0.16 (38) | 24/3/3 | 1.84 / 4.47 / 12 |
| fusion_refit | 1569 | 1569 | 8096 | 5.16 | 0.959 (211/9) | 0.897 (227/253) | 11 | 0.941 | 210/28/15 | 0.07 (17) | 0.16 (39) | 23/4/3 | 1.72 / 4.67 / 14 |
| fusion_server | 2997 | 1569 | 9524 | 3.18 | 0.964 (238/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.06 (14) | 0.16 (38) | 24/3/3 | 1.84 / 4.47 / 12 |
| fusion_server+attach | 1874 | 1569 | 9524 | 5.08 | 0.963 (237/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.06 (14) | 0.16 (38) | 24/3/3 | 1.84 / 4.47 / 12 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 210 of this run's 253 pool ramps are self-detected, so 83% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 945 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 7.4 m | 12.1 m | 0.49 | 0.24 | 0.02 |
| raycast | 7.1 m | 10.7 m | 0.45 | 0.14 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.684 | 100/173 | 15/30 | 0.680 | 43/172 | 15/30 |
| 5 | 0.913 | 108/231 | 23/30 | 0.909 | 36/230 | 24/30 |
| 7.5 | 0.957 | 108/242 | 23/30 | 0.949 | 35/240 | 24/30 |
| 10 | 0.976 | 107/247 | 25/30 | 0.968 | 35/245 | 25/30 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1428 | 0.79 | 0.523 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| ps @ 7.5 m | cluster of 1 | 326 | 0.75 | 0.555 | 11 | 0.900 [0.60, 0.98] | 9 | 1 | 1 |
| ps @ 7.5 m | cluster of 2 | 593 | 0.81 | 0.555 | 18 | 0.812 [0.57, 0.93] | 13 | 3 | 2 |
| ps @ 7.5 m | cluster of 3+ | 7177 | 0.87 | 0.568 | 205 | 0.974 [0.94, 0.99] | 188 | 5 | 12 |
| fusion_server | unplaceable | 1428 | 0.79 | 0.523 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server | cluster of 1 | 386 | 0.78 | 0.553 | 13 | 0.846 [0.58, 0.96] | 11 | 2 | 0 |
| fusion_server | cluster of 2 | 478 | 0.76 | 0.555 | 14 | 0.818 [0.52, 0.95] | 9 | 2 | 3 |
| fusion_server | cluster of 3+ | 7232 | 0.87 | 0.568 | 207 | 0.974 [0.94, 0.99] | 190 | 5 | 12 |
| fusion_server+attach | unplaceable | 1428 | 0.79 | 0.523 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server+attach | cluster of 1 | 329 | 0.77 | 0.553 | 9 | 0.778 [0.45, 0.94] | 7 | 2 | 0 |
| fusion_server+attach | cluster of 2 | 387 | 0.76 | 0.555 | 13 | 0.900 [0.60, 0.98] | 9 | 1 | 3 |
| fusion_server+attach | cluster of 3+ | 7380 | 0.87 | 0.568 | 212 | 0.970 [0.94, 0.99] | 194 | 6 | 12 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1428; attached 1123 (0.79)
- clusters: 2997 (`fusion_server`) -> 1874 (`fusion_server+attach`)
- sanity (not a metric): 33 unplaceable labels are on judged panos; 28 of them attached (24 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
