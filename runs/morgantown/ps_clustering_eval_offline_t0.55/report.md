# morgantown: PS label clustering vs RampNet GT (offline)

mode: offline -- 10401 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (81 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.1; results `results.jsonl` sha256 `7dbf24e03574c31b0a3011490177a5d51dc20e30badf7fdd35da443e870b98bc`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 125 judged panos -> 250 placeable points -> 250 ramps (0 cross-pano merges), 250 in the recall pool; raycast placed 9753 of 10482 detections (drops {'below_floor': 0, 'on_rig': 81, 'horizon': 31, 'out_of_range': 617})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\morgantown\results.jsonl`: sha256 `7dbf24e03574c31b0a3011490177a5d51dc20e30badf7fdd35da443e870b98bc`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 10401 of 10401
- regions: none (no server), so every label is in one region and per-region == citywide
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 708 blocks, largest 115 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 9753 fusion members vs 9753 placeable server labels; 0 of 9753 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/morgantown/fusion_eval/report.md (published in the 2.6 m frame): precision 0.979, recall (union) 0.948, dual 24/5/0
- fusion_server input: 10401 AI + 0 human labels on 5720 panos (5720 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 648 labels the raycast cannot place (range cap, horizon) are singleton clusters; 1681 of its 1683 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 4755 panos in both: median 0.01 m, p90 0.20 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 3450 | 3214 | 10401 | 3.01 | 0.980 (195/4) | 0.968 (242/250) | 1 | 0.972 | 188/55/7 | 0.48 (178) | 0.71 (328) | 28/1/0 | 0.67 / 1.97 / 3 |
| ps @ 5 m | 2236 | 2113 | 10401 | 4.65 | 0.980 (195/4) | 0.936 (234/250) | 7 | 0.964 | 188/53/9 | 0.21 (54) | 0.38 (99) | 25/4/0 | 1.01 / 3.26 / 9 |
| ps @ 7.5 m | 1832 | 1762 | 10401 | 5.68 | 0.980 (195/4) | 0.920 (230/250) | 10 | 0.960 | 188/52/10 | 0.12 (27) | 0.20 (47) | 24/4/1 | 1.14 / 3.82 / 11 |
| ps @ 10 m | 1692 | 1647 | 10401 | 6.15 | 0.980 (195/4) | 0.892 (223/250) | 13 | 0.944 | 188/48/14 | 0.09 (19) | 0.15 (34) | 22/6/1 | 1.25 / 4.18 / 14 |
| ps @ 12.5 m | 1614 | 1585 | 10401 | 6.44 | 0.980 (195/4) | 0.864 (216/250) | 15 | 0.924 | 188/43/19 | 0.07 (16) | 0.13 (30) | 21/7/1 | 1.30 / 4.52 / 16 |
| ps @ 15 m | 1570 | 1545 | 10401 | 6.62 | 0.980 (195/4) | 0.860 (215/250) | 16 | 0.924 | 188/43/19 | 0.07 (16) | 0.14 (31) | 21/7/1 | 1.30 / 4.61 / 17 |
| ps_citywide @ 7.5 m | 1832 | 1762 | 10401 | 5.68 | 0.980 (195/4) | 0.920 (230/250) | 10 | 0.960 | 188/52/10 | 0.12 (27) | 0.20 (47) | 24/4/1 | 1.14 / 3.82 / 11 |
| ps_placeable @ 2.5 m | 3178 | 3178 | 9753 | 3.07 | 0.979 (188/4) | 0.968 (242/250) | 1 | 0.972 | 188/55/7 | 0.48 (170) | 0.70 (321) | 28/1/0 | 0.74 / 2.08 / 4 |
| ps_placeable @ 5 m | 2067 | 2067 | 9753 | 4.72 | 0.979 (188/4) | 0.932 (233/250) | 7 | 0.960 | 188/52/10 | 0.19 (47) | 0.34 (89) | 25/4/0 | 1.00 / 3.26 / 9 |
| ps_placeable @ 7.5 m | 1737 | 1737 | 9753 | 5.61 | 0.979 (188/4) | 0.920 (230/250) | 10 | 0.960 | 188/52/10 | 0.10 (23) | 0.18 (42) | 25/3/1 | 1.14 / 3.85 / 11 |
| ps_placeable @ 10 m | 1631 | 1631 | 9753 | 5.98 | 0.979 (188/4) | 0.892 (223/250) | 13 | 0.944 | 188/48/14 | 0.08 (17) | 0.13 (31) | 23/5/1 | 1.22 / 4.18 / 14 |
| ps_placeable @ 12.5 m | 1572 | 1572 | 9753 | 6.20 | 0.979 (188/4) | 0.872 (218/250) | 15 | 0.932 | 188/45/17 | 0.06 (14) | 0.12 (27) | 22/6/1 | 1.23 / 4.52 / 16 |
| ps_placeable @ 15 m | 1529 | 1529 | 9753 | 6.38 | 0.979 (188/4) | 0.868 (217/250) | 16 | 0.932 | 188/45/17 | 0.06 (14) | 0.12 (28) | 22/6/1 | 1.23 / 4.61 / 17 |
| ps_raycast @ 2.5 m | 3745 | 3745 | 9753 | 2.60 | 0.979 (188/4) | 0.972 (243/250) | 0 | 0.972 | 188/55/7 | 0.51 (173) | 0.81 (403) | 28/1/0 | 0.54 / 1.06 / 0 |
| ps_raycast @ 5 m | 2420 | 2420 | 9753 | 4.03 | 0.979 (188/4) | 0.968 (242/250) | 0 | 0.968 | 188/54/8 | 0.12 (30) | 0.39 (115) | 27/2/0 | 0.76 / 1.59 / 0 |
| ps_raycast @ 7.5 m | 1941 | 1941 | 9753 | 5.02 | 0.979 (188/4) | 0.948 (237/250) | 2 | 0.956 | 188/51/11 | 0.05 (12) | 0.14 (37) | 26/3/0 | 0.91 / 2.26 / 2 |
| ps_raycast @ 10 m | 1698 | 1698 | 9753 | 5.74 | 0.979 (188/4) | 0.920 (230/250) | 7 | 0.948 | 188/49/13 | 0.04 (10) | 0.10 (23) | 24/5/0 | 1.04 / 3.06 / 8 |
| ps_raycast @ 12.5 m | 1582 | 1582 | 9753 | 6.16 | 0.979 (188/4) | 0.896 (224/250) | 10 | 0.936 | 188/46/16 | 0.04 (8) | 0.08 (18) | 24/4/1 | 1.11 / 3.82 / 11 |
| ps_raycast @ 15 m | 1528 | 1528 | 9753 | 6.38 | 0.979 (188/4) | 0.888 (222/250) | 12 | 0.936 | 188/46/16 | 0.04 (8) | 0.07 (15) | 24/4/1 | 1.15 / 4.20 / 13 |
| fusion | 1683 | 1683 | 9753 | 5.80 | 0.979 (188/4) | 0.908 (227/250) | 11 | 0.952 | 188/50/12 | 0.06 (13) | 0.11 (24) | 25/4/0 | 1.14 / 3.61 / 11 |
| fusion_refit | 1683 | 1683 | 9753 | 5.80 | 0.979 (188/4) | 0.908 (227/250) | 10 | 0.948 | 188/49/13 | 0.06 (13) | 0.11 (25) | 24/5/0 | 1.02 / 3.36 / 10 |
| fusion_server | 2331 | 1683 | 10401 | 4.46 | 0.980 (195/4) | 0.908 (227/250) | 11 | 0.952 | 188/50/12 | 0.06 (13) | 0.11 (24) | 25/4/0 | 1.14 / 3.61 / 11 |
| fusion_server+attach | 1767 | 1683 | 10401 | 5.89 | 0.980 (195/4) | 0.908 (227/250) | 11 | 0.952 | 188/50/12 | 0.06 (13) | 0.11 (24) | 25/4/0 | 1.14 / 3.61 / 11 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 188 of this run's 250 pool ramps are self-detected, so 75% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 1166 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.5 m | 9.2 m | 0.20 | 0.08 | 0.01 |
| raycast | 5.5 m | 9.7 m | 0.29 | 0.09 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.784 | 47/196 | 22/29 | 0.744 | 31/186 | 17/29 |
| 5 | 0.920 | 46/230 | 24/29 | 0.908 | 24/227 | 25/29 |
| 7.5 | 0.980 | 45/245 | 28/29 | 0.972 | 24/243 | 28/29 |
| 10 | 0.988 | 42/247 | 29/29 | 0.980 | 24/245 | 28/29 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 648 | 0.73 | 10 | 1.000 [0.65, 1.00] | 7 | 0 | 3 |
| ps @ 7.5 m | cluster of 1 | 268 | 0.64 | 4 | 0.500 [0.15, 0.85] | 2 | 2 | 0 |
| ps @ 7.5 m | cluster of 2 | 342 | 0.71 | 6 | 1.000 [0.57, 1.00] | 5 | 0 | 1 |
| ps @ 7.5 m | cluster of 3+ | 9143 | 0.83 | 188 | 0.989 [0.96, 1.00] | 181 | 2 | 5 |
| fusion_server | unplaceable | 648 | 0.73 | 10 | 1.000 [0.65, 1.00] | 7 | 0 | 3 |
| fusion_server | cluster of 1 | 345 | 0.65 | 7 | 0.667 [0.30, 0.90] | 4 | 2 | 1 |
| fusion_server | cluster of 2 | 342 | 0.70 | 3 | 1.000 [0.34, 1.00] | 2 | 0 | 1 |
| fusion_server | cluster of 3+ | 9066 | 0.83 | 188 | 0.989 [0.96, 1.00] | 182 | 2 | 4 |
| fusion_server+attach | unplaceable | 648 | 0.73 | 10 | 1.000 [0.65, 1.00] | 7 | 0 | 3 |
| fusion_server+attach | cluster of 1 | 325 | 0.64 | 5 | 0.500 [0.15, 0.85] | 2 | 2 | 1 |
| fusion_server+attach | cluster of 2 | 327 | 0.70 | 4 | 1.000 [0.44, 1.00] | 3 | 0 | 1 |
| fusion_server+attach | cluster of 3+ | 9101 | 0.83 | 189 | 0.989 [0.96, 1.00] | 183 | 2 | 4 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 648; attached 564 (0.87)
- clusters: 2331 (`fusion_server`) -> 1767 (`fusion_server+attach`)
- sanity (not a metric): 10 unplaceable labels are on judged panos; 9 of them attached (7 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
