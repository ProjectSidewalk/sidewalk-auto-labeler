# richmond: PS label clustering vs RampNet GT

labels: 9639 CurbRamp on the server, 9526 map to stored detections (AI), 113 do not (human); 2156 server clusters over 9634 labels
raycast camera height 2.6 m
GT: 124 judged panos -> 253 placeable points -> 253 ramps (0 cross-pano merges), 253 in the recall pool; raycast placed 8098 of 9526 detections (drops {'below_floor': 0, 'horizon': 236, 'out_of_range': 1192})

## Data provenance

- `raw_labels.geojson`: 9639 features, sha256 `17bde58ca3d678099195923781cf9846f5dc237dc1fd8e897417d4658087a058`, 2026-09-21T13:34:28+00:00, from https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 2156 features, sha256 `3f7ca04dfc67c32950f8a14150c54ff6c3cb2485b069c60b7554072e30641fd0`, 2026-09-21T13:34:28+00:00, from https://sidewalk-richmond.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson

## Validation checks

- ps_repro reproduces deployed: 2156/2156 clusters identical (0 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 2156/2156 clusters identical (0 labels differ)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 0 of that account's labels did not map (should be 0)
- fusion arm vs ps_* arms cover the same labels: 8098 fusion members vs 8098 placeable server labels; 0 of 8098 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit vs runs/richmond/fusion_eval/report.md: precision 0.959, recall (union) 0.941, dual 23/4/3
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| deployed | 2156 | 1993 | 9634 | 4.47 | 0.964 (238/9) | 0.917 (232/253) | 9 | 0.953 | 210/31/12 | 0.24 (60) | 0.47 (132) | 23/4/3 | 1.49 / 3.98 / 11 |
| ps_repro | 2156 | 1993 | 9634 | 4.47 | 0.964 (238/9) | 0.917 (232/253) | 9 | 0.953 | 210/31/12 | 0.24 (60) | 0.47 (132) | 23/4/3 | 1.49 / 3.98 / 11 |
| ps @ 2.5 m | 4515 | 3908 | 9526 | 2.11 | 0.964 (238/9) | 0.964 (244/253) | 1 | 0.968 | 210/35/8 | 0.61 (295) | 0.80 (542) | 27/1/2 | 0.80 / 2.57 / 2 |
| ps @ 5 m | 2850 | 2565 | 9526 | 3.34 | 0.964 (238/9) | 0.937 (237/253) | 6 | 0.960 | 210/33/10 | 0.38 (113) | 0.62 (235) | 25/2/3 | 1.40 / 3.73 / 8 |
| ps @ 7.5 m | 2149 | 1987 | 9526 | 4.43 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (59) | 0.47 (130) | 23/4/3 | 1.52 / 3.98 / 11 |
| ps @ 10 m | 1773 | 1676 | 9526 | 5.37 | 0.963 (237/9) | 0.889 (225/253) | 14 | 0.945 | 210/29/14 | 0.15 (33) | 0.32 (81) | 22/5/3 | 1.80 / 4.43 / 15 |
| ps @ 12.5 m | 1585 | 1516 | 9526 | 6.01 | 0.963 (237/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.14 (31) | 0.29 (68) | 22/5/3 | 1.86 / 4.57 / 18 |
| ps @ 15 m | 1487 | 1434 | 9526 | 6.41 | 0.963 (237/9) | 0.874 (221/253) | 17 | 0.941 | 210/28/15 | 0.13 (28) | 0.26 (60) | 22/5/3 | 1.95 / 4.57 / 18 |
| ps_citywide @ 7.5 m | 2115 | 1956 | 9526 | 4.50 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (58) | 0.46 (126) | 23/4/3 | 1.69 / 4.00 / 11 |
| ps_placeable @ 2.5 m | 3843 | 3843 | 8098 | 2.11 | 0.959 (211/9) | 0.960 (243/253) | 1 | 0.964 | 210/34/9 | 0.60 (289) | 0.78 (526) | 26/2/2 | 0.81 / 2.62 / 2 |
| ps_placeable @ 5 m | 2475 | 2475 | 8098 | 3.27 | 0.959 (211/9) | 0.929 (235/253) | 7 | 0.957 | 210/32/11 | 0.34 (101) | 0.60 (221) | 25/2/3 | 1.40 / 3.68 / 9 |
| ps_placeable @ 7.5 m | 1887 | 1887 | 8098 | 4.29 | 0.959 (211/9) | 0.897 (227/253) | 13 | 0.949 | 210/30/13 | 0.20 (46) | 0.44 (113) | 23/4/3 | 1.54 / 4.06 / 14 |
| ps_placeable @ 10 m | 1590 | 1590 | 8098 | 5.09 | 0.959 (211/9) | 0.893 (226/253) | 14 | 0.949 | 210/30/13 | 0.11 (25) | 0.29 (70) | 23/4/3 | 1.80 / 4.50 / 15 |
| ps_placeable @ 12.5 m | 1461 | 1461 | 8098 | 5.54 | 0.959 (211/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.12 (26) | 0.27 (61) | 23/4/3 | 1.78 / 4.57 / 18 |
| ps_placeable @ 15 m | 1384 | 1384 | 8098 | 5.85 | 0.959 (211/9) | 0.874 (221/253) | 16 | 0.937 | 210/27/16 | 0.11 (24) | 0.24 (54) | 23/4/3 | 1.89 / 4.51 / 17 |
| ps_raycast @ 2.5 m | 3912 | 3912 | 8098 | 2.07 | 0.959 (211/9) | 0.968 (245/253) | 0 | 0.968 | 210/35/8 | 0.56 (211) | 0.81 (477) | 27/1/2 | 0.48 / 1.14 / 0 |
| ps_raycast @ 5 m | 2584 | 2584 | 8098 | 3.13 | 0.959 (211/9) | 0.957 (242/253) | 0 | 0.957 | 210/32/11 | 0.17 (45) | 0.53 (169) | 24/3/3 | 1.05 / 2.17 / 0 |
| ps_raycast @ 7.5 m | 1979 | 1979 | 8098 | 4.09 | 0.959 (211/9) | 0.949 (240/253) | 1 | 0.953 | 210/31/12 | 0.07 (18) | 0.31 (78) | 23/4/3 | 1.34 / 2.72 / 0 |
| ps_raycast @ 10 m | 1638 | 1638 | 8098 | 4.94 | 0.959 (211/9) | 0.913 (231/253) | 8 | 0.945 | 210/29/14 | 0.06 (16) | 0.22 (52) | 23/4/3 | 1.76 / 3.75 / 8 |
| ps_raycast @ 12.5 m | 1488 | 1488 | 8098 | 5.44 | 0.959 (211/9) | 0.889 (225/253) | 13 | 0.941 | 210/28/15 | 0.06 (15) | 0.19 (44) | 23/4/3 | 1.80 / 4.43 / 13 |
| ps_raycast @ 15 m | 1396 | 1396 | 8098 | 5.80 | 0.959 (211/9) | 0.874 (221/253) | 17 | 0.941 | 210/28/15 | 0.06 (15) | 0.17 (40) | 23/4/3 | 1.86 / 4.98 / 18 |
| fusion | 1570 | 1570 | 8098 | 5.16 | 0.959 (211/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.06 (14) | 0.16 (38) | 24/3/3 | 1.84 / 4.47 / 12 |
| fusion_refit | 1570 | 1570 | 8098 | 5.16 | 0.959 (211/9) | 0.897 (227/253) | 11 | 0.941 | 210/28/15 | 0.07 (17) | 0.16 (39) | 23/4/3 | 1.72 / 4.67 / 14 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 210 of Richmond's 253 pool ramps are self-detected, so it is nearly constant across arms and is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 411, 2: 388, 3: 330, 4: 228, 5: 172, 6: 128, 7: 102, 8: 88, 9: 87, 10: 55, 11: 49, 12: 31, 13: 40, 14: 18, 15: 6, 16: 10, 17: 6, 18: 4, 19: 1, 20: 2
- server centroid vs raycast centroid, same members (n=1993): median 1.5 m, p90 4.4 m
- per-label server lat/lng vs labeler raycast (n=8098): median 1.2 m, p90 4.8 m, over 5 m 0.07

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 945 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 7.4 m | 12.1 m | 0.49 | 0.24 | 0.02 |
| raycast | 7.1 m | 10.7 m | 0.45 | 0.14 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed coverage | deployed frag 5 m | deployed dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.684 | 102/173 | 15/30 | 0.680 | 43/172 | 15/30 |
| 5 | 0.917 | 109/232 | 23/30 | 0.909 | 36/230 | 24/30 |
| 7.5 | 0.957 | 109/242 | 23/30 | 0.949 | 35/240 | 24/30 |
| 10 | 0.976 | 108/247 | 25/30 | 0.968 | 35/245 | 25/30 |
