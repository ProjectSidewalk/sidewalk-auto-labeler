# richmond: PS label clustering vs RampNet GT (offline)

mode: offline -- 9524 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (2 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.1; results `results.jsonl` sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 124 judged panos -> 253 placeable points -> 253 ramps (0 cross-pano merges), 253 in the recall pool; raycast placed 8096 of 9526 detections (drops {'below_floor': 0, 'on_rig': 2, 'horizon': 236, 'out_of_range': 1192})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\richmond\results.jsonl`: sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 9524 of 9524
- `ps_streets.geojson`: 16365 features, sha256 `02f3e0061c16ecda5d6ba5523f25bbff5999d758061cb75dd408edb732f92206`, 2026-09-28T17:38:33+00:00 (0.0 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/streets?filetype=geojson
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 413 blocks, largest 141 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 8096 fusion members vs 8096 placeable server labels; 0 of 8096 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/richmond/fusion_eval/report.md (published in the 2.6 m frame): precision 0.959, recall (union) 0.941, dual 23/4/3
- fusion_server input: 9524 AI + 0 human labels on 3684 panos (3684 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 1428 labels the raycast cannot place (range cap, horizon) are singleton clusters; 1558 of its 1569 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 3022 panos in both: median 0.01 m, p90 0.20 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 4518 | 3910 | 9524 | 2.11 | 0.964 (238/9) | 0.964 (244/253) | 1 | 0.968 | 210/35/8 | 0.62 (296) | 0.80 (544) | 27/1/2 | 0.80 / 2.57 / 2 |
| ps @ 5 m | 2849 | 2566 | 9524 | 3.34 | 0.964 (238/9) | 0.937 (237/253) | 6 | 0.960 | 210/33/10 | 0.38 (114) | 0.62 (235) | 25/2/3 | 1.40 / 3.73 / 8 |
| ps @ 7.5 m | 2148 | 1988 | 9524 | 4.43 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.23 (57) | 0.47 (128) | 23/4/3 | 1.55 / 3.98 / 11 |
| ps @ 10 m | 1773 | 1677 | 9524 | 5.37 | 0.963 (237/9) | 0.893 (226/253) | 14 | 0.949 | 210/30/13 | 0.14 (31) | 0.32 (79) | 22/5/3 | 1.80 / 4.43 / 15 |
| ps @ 12.5 m | 1584 | 1516 | 9524 | 6.01 | 0.963 (237/9) | 0.881 (223/253) | 17 | 0.949 | 210/30/13 | 0.13 (28) | 0.28 (64) | 22/5/3 | 1.87 / 4.80 / 18 |
| ps @ 15 m | 1486 | 1433 | 9524 | 6.41 | 0.963 (237/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.11 (25) | 0.24 (56) | 22/5/3 | 1.95 / 4.80 / 18 |
| ps_citywide @ 7.5 m | 2113 | 1954 | 9524 | 4.51 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (58) | 0.46 (126) | 23/4/3 | 1.69 / 4.00 / 11 |
| ps_placeable @ 2.5 m | 3845 | 3845 | 8096 | 2.11 | 0.959 (211/9) | 0.960 (243/253) | 1 | 0.964 | 210/34/9 | 0.61 (290) | 0.78 (528) | 26/2/2 | 0.81 / 2.62 / 2 |
| ps_placeable @ 5 m | 2477 | 2477 | 8096 | 3.27 | 0.959 (211/9) | 0.929 (235/253) | 7 | 0.957 | 210/32/11 | 0.35 (103) | 0.61 (222) | 25/2/3 | 1.40 / 3.68 / 9 |
| ps_placeable @ 7.5 m | 1890 | 1890 | 8096 | 4.28 | 0.959 (211/9) | 0.897 (227/253) | 13 | 0.949 | 210/30/13 | 0.19 (45) | 0.44 (113) | 23/4/3 | 1.55 / 4.06 / 14 |
| ps_placeable @ 10 m | 1590 | 1590 | 8096 | 5.09 | 0.959 (211/9) | 0.893 (226/253) | 14 | 0.949 | 210/30/13 | 0.11 (24) | 0.28 (68) | 23/4/3 | 1.80 / 4.50 / 15 |
| ps_placeable @ 12.5 m | 1462 | 1462 | 8096 | 5.54 | 0.959 (211/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.11 (25) | 0.27 (60) | 23/4/3 | 1.80 / 4.80 / 18 |
| ps_placeable @ 15 m | 1384 | 1384 | 8096 | 5.85 | 0.959 (211/9) | 0.874 (221/253) | 16 | 0.937 | 210/27/16 | 0.10 (23) | 0.24 (53) | 23/4/3 | 1.93 / 4.57 / 17 |
| ps_raycast @ 2.5 m | 3913 | 3913 | 8096 | 2.07 | 0.959 (211/9) | 0.968 (245/253) | 0 | 0.968 | 210/35/8 | 0.56 (209) | 0.81 (475) | 27/1/2 | 0.50 / 1.15 / 0 |
| ps_raycast @ 5 m | 2580 | 2580 | 8096 | 3.14 | 0.959 (211/9) | 0.957 (242/253) | 0 | 0.957 | 210/32/11 | 0.15 (42) | 0.52 (163) | 24/3/3 | 1.07 / 2.17 / 0 |
| ps_raycast @ 7.5 m | 1982 | 1982 | 8096 | 4.08 | 0.959 (211/9) | 0.953 (241/253) | 1 | 0.957 | 210/32/11 | 0.07 (17) | 0.29 (73) | 23/4/3 | 1.34 / 2.82 / 0 |
| ps_raycast @ 10 m | 1639 | 1639 | 8096 | 4.94 | 0.959 (211/9) | 0.921 (233/253) | 7 | 0.949 | 210/30/13 | 0.06 (15) | 0.20 (48) | 23/4/3 | 1.76 / 3.69 / 6 |
| ps_raycast @ 12.5 m | 1490 | 1490 | 8096 | 5.43 | 0.959 (211/9) | 0.897 (227/253) | 12 | 0.945 | 210/29/14 | 0.06 (14) | 0.17 (40) | 23/4/3 | 1.80 / 4.34 / 11 |
| ps_raycast @ 15 m | 1395 | 1395 | 8096 | 5.80 | 0.959 (211/9) | 0.885 (224/253) | 16 | 0.949 | 210/30/13 | 0.06 (14) | 0.16 (36) | 23/4/3 | 1.86 / 4.80 / 16 |
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
| 2.5 | 0.688 | 102/174 | 14/30 | 0.680 | 43/172 | 15/30 |
| 5 | 0.913 | 108/231 | 23/30 | 0.909 | 36/230 | 24/30 |
| 7.5 | 0.957 | 108/242 | 23/30 | 0.949 | 35/240 | 24/30 |
| 10 | 0.976 | 107/247 | 25/30 | 0.968 | 35/245 | 25/30 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1428 | 0.79 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| ps @ 7.5 m | cluster of 1 | 326 | 0.75 | 10 | 0.889 [0.56, 0.98] | 8 | 1 | 1 |
| ps @ 7.5 m | cluster of 2 | 592 | 0.81 | 18 | 0.812 [0.57, 0.93] | 13 | 3 | 2 |
| ps @ 7.5 m | cluster of 3+ | 7178 | 0.87 | 206 | 0.974 [0.94, 0.99] | 189 | 5 | 12 |
| fusion_server | unplaceable | 1428 | 0.79 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server | cluster of 1 | 386 | 0.78 | 13 | 0.846 [0.58, 0.96] | 11 | 2 | 0 |
| fusion_server | cluster of 2 | 478 | 0.76 | 14 | 0.818 [0.52, 0.95] | 9 | 2 | 3 |
| fusion_server | cluster of 3+ | 7232 | 0.87 | 207 | 0.974 [0.94, 0.99] | 190 | 5 | 12 |
| fusion_server+attach | unplaceable | 1428 | 0.79 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server+attach | cluster of 1 | 329 | 0.77 | 9 | 0.778 [0.45, 0.94] | 7 | 2 | 0 |
| fusion_server+attach | cluster of 2 | 387 | 0.76 | 13 | 0.900 [0.60, 0.98] | 9 | 1 | 3 |
| fusion_server+attach | cluster of 3+ | 7380 | 0.87 | 212 | 0.970 [0.94, 0.99] | 194 | 6 | 12 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1428; attached 1123 (0.79)
- clusters: 2997 (`fusion_server`) -> 1874 (`fusion_server+attach`)
- sanity (not a metric): 33 unplaceable labels are on judged panos; 28 of them attached (24 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
