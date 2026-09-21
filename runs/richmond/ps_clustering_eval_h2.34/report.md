# richmond: PS label clustering vs RampNet GT

labels: 9639 CurbRamp on the server, 9526 map to stored detections (AI), 113 do not (human); 2156 server clusters over 9634 labels
raycast camera height 2.34122 m
GT: 124 judged panos -> 260 placeable points -> 260 ramps (0 cross-pano merges), 260 in the recall pool; raycast placed 8098 of 9526 detections (drops {'below_floor': 0, 'horizon': 236, 'out_of_range': 1192})

## Data provenance

- `raw_labels.geojson`: 9639 features, sha256 `17bde58ca3d678099195923781cf9846f5dc237dc1fd8e897417d4658087a058`, 2026-09-21T13:34:28+00:00, from https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 2156 features, sha256 `3f7ca04dfc67c32950f8a14150c54ff6c3cb2485b069c60b7554072e30641fd0`, 2026-09-21T13:34:28+00:00, from https://sidewalk-richmond.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson

## Validation checks

- ps_repro reproduces deployed: 2156/2156 clusters identical (0 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 2156/2156 clusters identical (0 labels differ)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 0 of that account's labels did not map (should be 0)
- fusion arm vs ps_* arms cover the same labels: 8098 fusion members vs 8098 placeable server labels; 0 of 8098 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit vs runs/richmond/fusion_eval/report.md: precision 0.959, recall (union) 0.942, dual 25/5/2
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| deployed | 2156 | 1993 | 9634 | 4.47 | 0.964 (238/9) | 0.931 (242/260) | 4 | 0.946 | 210/36/14 | 0.17 (43) | 0.48 (136) | 25/4/3 | 1.39 / 2.99 / 3 |
| ps_repro | 2156 | 1993 | 9634 | 4.47 | 0.964 (238/9) | 0.931 (242/260) | 4 | 0.946 | 210/36/14 | 0.17 (43) | 0.48 (136) | 25/4/3 | 1.39 / 2.99 / 3 |
| ps @ 2.5 m | 4515 | 3908 | 9526 | 2.11 | 0.964 (238/9) | 0.962 (250/260) | 0 | 0.962 | 210/40/10 | 0.60 (260) | 0.84 (567) | 29/1/2 | 0.66 / 1.76 / 0 |
| ps @ 5 m | 2850 | 2565 | 9526 | 3.34 | 0.964 (238/9) | 0.954 (248/260) | 0 | 0.954 | 210/38/12 | 0.28 (78) | 0.62 (232) | 27/2/3 | 1.18 / 2.82 / 0 |
| ps @ 7.5 m | 2149 | 1987 | 9526 | 4.43 | 0.964 (238/9) | 0.931 (242/260) | 4 | 0.946 | 210/36/14 | 0.17 (43) | 0.46 (131) | 25/4/3 | 1.40 / 3.08 / 3 |
| ps @ 10 m | 1773 | 1676 | 9526 | 5.37 | 0.963 (237/9) | 0.912 (237/260) | 8 | 0.942 | 210/35/15 | 0.12 (29) | 0.28 (75) | 25/4/3 | 1.59 / 4.00 / 7 |
| ps @ 12.5 m | 1585 | 1516 | 9526 | 6.01 | 0.963 (237/9) | 0.904 (235/260) | 10 | 0.942 | 210/35/15 | 0.11 (27) | 0.25 (64) | 25/4/3 | 1.63 / 4.11 / 9 |
| ps @ 15 m | 1487 | 1434 | 9526 | 6.41 | 0.963 (237/9) | 0.896 (233/260) | 11 | 0.938 | 210/34/16 | 0.11 (27) | 0.21 (55) | 25/4/3 | 1.68 / 4.11 / 11 |
| ps_citywide @ 7.5 m | 2115 | 1956 | 9526 | 4.50 | 0.964 (238/9) | 0.935 (243/260) | 3 | 0.946 | 210/36/14 | 0.17 (42) | 0.45 (125) | 25/4/3 | 1.44 / 3.09 / 2 |
| ps_placeable @ 2.5 m | 3843 | 3843 | 8098 | 2.11 | 0.959 (211/9) | 0.958 (249/260) | 0 | 0.958 | 210/39/11 | 0.59 (252) | 0.84 (554) | 28/2/2 | 0.69 / 1.88 / 0 |
| ps_placeable @ 5 m | 2475 | 2475 | 8098 | 3.27 | 0.959 (211/9) | 0.946 (246/260) | 2 | 0.954 | 210/38/12 | 0.26 (70) | 0.62 (218) | 27/2/3 | 1.18 / 2.77 / 2 |
| ps_placeable @ 7.5 m | 1887 | 1887 | 8098 | 4.29 | 0.959 (211/9) | 0.931 (242/260) | 4 | 0.946 | 210/36/14 | 0.13 (33) | 0.41 (109) | 25/4/3 | 1.42 / 3.23 / 4 |
| ps_placeable @ 10 m | 1590 | 1590 | 8098 | 5.09 | 0.959 (211/9) | 0.912 (237/260) | 8 | 0.942 | 210/35/15 | 0.10 (25) | 0.25 (63) | 25/4/3 | 1.66 / 3.98 / 7 |
| ps_placeable @ 12.5 m | 1461 | 1461 | 8098 | 5.54 | 0.959 (211/9) | 0.900 (234/260) | 10 | 0.938 | 210/34/16 | 0.10 (25) | 0.23 (57) | 25/4/3 | 1.65 / 4.03 / 9 |
| ps_placeable @ 15 m | 1384 | 1384 | 8098 | 5.85 | 0.959 (211/9) | 0.892 (232/260) | 10 | 0.931 | 210/32/18 | 0.10 (25) | 0.20 (49) | 25/4/3 | 1.68 / 4.03 / 10 |
| ps_raycast @ 2.5 m | 3866 | 3866 | 8098 | 2.09 | 0.959 (211/9) | 0.962 (250/260) | 0 | 0.962 | 210/40/10 | 0.61 (234) | 0.86 (545) | 29/1/2 | 0.55 / 1.10 / 0 |
| ps_raycast @ 5 m | 2506 | 2506 | 8098 | 3.23 | 0.959 (211/9) | 0.954 (248/260) | 0 | 0.954 | 210/38/12 | 0.18 (48) | 0.55 (172) | 27/2/3 | 1.09 / 2.18 / 0 |
| ps_raycast @ 7.5 m | 1896 | 1896 | 8098 | 4.27 | 0.959 (211/9) | 0.938 (244/260) | 1 | 0.942 | 210/35/15 | 0.09 (23) | 0.30 (79) | 23/6/3 | 1.42 / 2.77 / 1 |
| ps_raycast @ 10 m | 1619 | 1619 | 8098 | 5.00 | 0.959 (211/9) | 0.919 (239/260) | 5 | 0.938 | 210/34/16 | 0.08 (19) | 0.21 (52) | 22/7/3 | 1.60 / 3.65 / 4 |
| ps_raycast @ 12.5 m | 1464 | 1464 | 8098 | 5.53 | 0.959 (211/9) | 0.896 (233/260) | 10 | 0.935 | 210/33/17 | 0.08 (19) | 0.19 (46) | 22/7/3 | 1.63 / 4.02 / 9 |
| ps_raycast @ 15 m | 1371 | 1371 | 8098 | 5.91 | 0.959 (211/9) | 0.881 (229/260) | 13 | 0.931 | 210/32/18 | 0.07 (18) | 0.17 (40) | 22/7/3 | 1.63 / 4.03 / 12 |
| fusion | 1514 | 1514 | 8098 | 5.35 | 0.959 (211/9) | 0.908 (236/260) | 8 | 0.938 | 210/34/16 | 0.08 (20) | 0.17 (41) | 26/3/3 | 1.71 / 4.32 / 12 |
| fusion_refit | 1514 | 1514 | 8098 | 5.35 | 0.959 (211/9) | 0.908 (236/260) | 9 | 0.942 | 210/35/15 | 0.08 (20) | 0.17 (44) | 25/5/2 | 1.67 / 4.64 / 12 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 210 of Richmond's 253 pool ramps are self-detected, so it is nearly constant across arms and is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 411, 2: 388, 3: 330, 4: 228, 5: 172, 6: 128, 7: 102, 8: 88, 9: 87, 10: 55, 11: 49, 12: 31, 13: 40, 14: 18, 15: 6, 16: 10, 17: 6, 18: 4, 19: 1, 20: 2
- server centroid vs raycast centroid, same members (n=1993): median 0.6 m, p90 2.6 m
- per-label server lat/lng vs labeler raycast (n=8098): median 0.0 m, p90 2.7 m, over 5 m 0.00

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 954 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 7.1 m | 11.3 m | 0.46 | 0.19 | 0.01 |
| raycast | 7.0 m | 10.5 m | 0.44 | 0.14 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed coverage | deployed frag 5 m | deployed dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.731 | 104/190 | 15/32 | 0.650 | 40/169 | 14/32 |
| 5 | 0.931 | 115/242 | 25/32 | 0.908 | 39/236 | 26/32 |
| 7.5 | 0.969 | 115/252 | 26/32 | 0.958 | 38/249 | 26/32 |
| 10 | 0.977 | 114/254 | 28/32 | 0.965 | 38/251 | 27/32 |
