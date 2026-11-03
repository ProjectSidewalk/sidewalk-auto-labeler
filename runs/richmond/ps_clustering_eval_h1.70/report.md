# richmond: PS label clustering vs RampNet GT

labels: 9631 CurbRamp on the server, 9526 map to stored detections (AI), 105 do not (human); 2156 server clusters over 9627 labels
raycast camera height 1.7 m
GT: 124 judged panos -> 289 placeable points -> 289 ramps (0 cross-pano merges), 289 in the recall pool; raycast placed 8885 of 9526 detections (drops {'below_floor': 0, 'horizon': 236, 'out_of_range': 405})

## Validation checks

- ps_repro skipped (no --ps-script)
- fusion_refit vs runs/richmond/fusion_eval/report.md: precision 0.962, recall 0.927, dual 32/20/3
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | labels | labels/cluster | precision (TP/FP) | recall | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---|---|---|---|---|---|---|
| deployed | 2156 | 9627 | 4.47 | 0.964 (238/9) | 0.927 | 229/39/21 | 0.11 (27) | 0.34 (103) | 36/14/5 | 2.37 / 4.91 / 21 |
| ps @ 2.5 m | 4515 | 9526 | 2.11 | 0.964 (238/9) | 0.965 | 229/50/10 | 0.45 (198) | 0.77 (540) | 48/4/3 | 1.30 / 3.30 / 0 |
| ps @ 5 m | 2850 | 9526 | 3.34 | 0.964 (238/9) | 0.955 | 229/47/13 | 0.19 (55) | 0.52 (202) | 42/8/5 | 2.08 / 4.32 / 12 |
| ps @ 7.5 m | 2149 | 9526 | 4.43 | 0.964 (238/9) | 0.927 | 229/39/21 | 0.11 (27) | 0.33 (98) | 35/15/5 | 2.37 / 4.94 / 22 |
| ps @ 10 m | 1773 | 9526 | 5.37 | 0.963 (237/9) | 0.924 | 229/38/22 | 0.08 (19) | 0.20 (51) | 31/19/5 | 2.49 / 5.59 / 33 |
| ps @ 12.5 m | 1585 | 9526 | 6.01 | 0.963 (237/9) | 0.920 | 229/37/23 | 0.07 (17) | 0.17 (41) | 31/19/5 | 2.57 / 5.82 / 34 |
| ps @ 15 m | 1487 | 9526 | 6.41 | 0.963 (237/9) | 0.917 | 229/36/24 | 0.06 (14) | 0.15 (36) | 32/18/5 | 2.57 / 5.82 / 36 |
| ps_citywide @ 7.5 m | 2115 | 9526 | 4.50 | 0.964 (238/9) | 0.927 | 229/39/21 | 0.09 (23) | 0.33 (95) | 35/15/5 | 2.37 / 4.91 / 20 |
| ps_placeable @ 2.5 m | 4163 | 8885 | 2.13 | 0.962 (230/9) | 0.965 | 229/50/10 | 0.44 (195) | 0.76 (528) | 48/4/3 | 1.33 / 3.30 / 0 |
| ps_placeable @ 5 m | 2654 | 8885 | 3.35 | 0.962 (230/9) | 0.955 | 229/47/13 | 0.18 (53) | 0.51 (193) | 42/8/5 | 2.08 / 4.32 / 12 |
| ps_placeable @ 7.5 m | 2003 | 8885 | 4.44 | 0.962 (230/9) | 0.924 | 229/38/22 | 0.10 (24) | 0.32 (92) | 34/16/5 | 2.40 / 5.28 / 26 |
| ps_placeable @ 10 m | 1672 | 8885 | 5.31 | 0.962 (229/9) | 0.924 | 229/38/22 | 0.08 (18) | 0.19 (49) | 32/17/6 | 2.52 / 5.75 / 35 |
| ps_placeable @ 12.5 m | 1515 | 8885 | 5.86 | 0.962 (229/9) | 0.913 | 229/35/25 | 0.07 (16) | 0.16 (39) | 31/18/6 | 2.52 / 5.82 / 36 |
| ps_placeable @ 15 m | 1430 | 8885 | 6.21 | 0.962 (229/9) | 0.910 | 229/34/26 | 0.07 (15) | 0.16 (37) | 31/18/6 | 2.56 / 5.82 / 37 |
| ps_raycast @ 2.5 m | 4978 | 8885 | 1.78 | 0.962 (230/9) | 0.969 | 229/51/9 | 0.54 (229) | 0.87 (692) | 47/6/2 | 0.48 / 1.01 / 0 |
| ps_raycast @ 5 m | 3200 | 8885 | 2.78 | 0.962 (230/9) | 0.952 | 229/46/14 | 0.19 (57) | 0.62 (265) | 42/11/2 | 1.05 / 2.25 / 0 |
| ps_raycast @ 7.5 m | 2445 | 8885 | 3.63 | 0.962 (230/9) | 0.941 | 229/43/17 | 0.07 (20) | 0.39 (132) | 40/10/5 | 1.49 / 2.89 / 0 |
| ps_raycast @ 10 m | 2010 | 8885 | 4.42 | 0.962 (230/9) | 0.924 | 229/38/22 | 0.06 (16) | 0.30 (92) | 37/13/5 | 1.77 / 4.12 / 11 |
| ps_raycast @ 12.5 m | 1766 | 8885 | 5.03 | 0.962 (229/9) | 0.920 | 229/37/23 | 0.06 (15) | 0.27 (76) | 34/15/6 | 2.19 / 5.00 / 23 |
| ps_raycast @ 15 m | 1605 | 8885 | 5.54 | 0.962 (229/9) | 0.910 | 229/34/26 | 0.06 (14) | 0.24 (63) | 33/17/5 | 2.34 / 5.55 / 33 |
| fusion | 1805 | 8885 | 4.92 | 0.962 (230/9) | 0.927 | 229/39/21 | 0.05 (14) | 0.22 (55) | 31/20/4 | 2.36 / 5.44 / 29 |
| fusion_refit | 1805 | 8885 | 4.92 | 0.962 (230/9) | 0.927 | 229/39/21 | 0.05 (14) | 0.23 (56) | 32/20/3 | 2.28 / 5.51 / 31 |

frag = share of matched GT ramps with at least one extra cluster within r that is nobody's match (total extras in parentheses); coherence = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 412, 2: 388, 3: 330, 4: 229, 5: 170, 6: 128, 7: 102, 8: 88, 9: 86, 10: 57, 11: 48, 12: 32, 13: 40, 14: 17, 15: 6, 16: 10, 17: 6, 18: 4, 19: 1, 20: 2
- server centroid vs raycast centroid, same members (n=2055): median 1.8 m, p90 3.3 m
- per-label server lat/lng vs labeler raycast (n=8885): median 2.5 m, p90 3.5 m, over 5 m 0.00

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 1123 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 7.2 m | 11.5 m | 0.46 | 0.20 | 0.01 |
| raycast | 8.4 m | 11.6 m | 0.64 | 0.26 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed recall | deployed frag 5 m | deployed dual both | fusion recall | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.855 | 77/162 | 14/55 | 0.862 | 45/155 | 14/55 |
| 5 | 0.927 | 85/249 | 36/55 | 0.927 | 54/243 | 31/55 |
| 7.5 | 0.969 | 83/279 | 45/55 | 0.965 | 44/276 | 44/55 |
| 10 | 0.976 | 83/282 | 48/55 | 0.969 | 44/279 | 46/55 |
