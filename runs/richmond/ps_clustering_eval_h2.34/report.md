# richmond: PS label clustering vs RampNet GT

labels: 9639 CurbRamp on the server, 9526 map to stored detections (AI), 113 do not (human); 2156 server clusters over 9634 labels
scorer 106.1; results `results.jsonl` sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
raycast camera height 2.34122 m; fusion arm at --min-confidence 0.55
GT: 124 judged panos -> 260 placeable points -> 260 ramps (0 cross-pano merges), 260 in the recall pool; raycast placed 8098 of 9526 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 236, 'out_of_range': 1192})

## Data provenance

- `raw_labels.geojson`: 9639 features, sha256 `17bde58ca3d678099195923781cf9846f5dc237dc1fd8e897417d4658087a058`, 2026-09-21T13:34:28+00:00 (7.2 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 2156 features, sha256 `3f7ca04dfc67c32950f8a14150c54ff6c3cb2485b069c60b7554072e30641fd0`, 2026-09-21T13:34:28+00:00 (7.2 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- labels by account: 51b0b927-3c8a-45b2-93de-bd878d1e5cf4 (AI) 9526, 549187e0-82c9-4014-a48d-31f18083d575 81, 18b26a38-24ab-402d-a64e-158fc0bb8a8a 30, 61460b3e-712d-4732-9044-924c4c1fc221 2
- 0 labels dropped before clustering (null lng or lng > 360), matching label_clustering.clean_label_data
- 0 ambiguous pixel keys in results.jsonl (two stored detections round to one pixel; those keys are left unmapped)
- 0 server labels share a pixel with another label and so map to the same stored detection (a re-submitted campaign does this)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 406 blocks, largest 141 labels

## Validation checks

- ps_repro reproduces deployed: 2156/2156 clusters identical (0 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 2156/2156 clusters identical (0 labels differ)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 0 of that account's labels did not map (should be 0)
- fusion arm vs ps_* arms cover the same labels: 8098 fusion members vs 8098 placeable server labels; 0 of 8098 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.34122 m vs runs/richmond/fusion_eval/report.md (published in the 2.6 m frame): precision 0.959, recall (union) 0.942, dual 25/5/2 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 9526 AI + 113 human labels on 3718 panos (3685 positioned from the run's pano block, 33 inverted from their labels, 3 unplaceable and left out); 1432 labels the raycast cannot place (range cap, horizon) are singleton clusters; 1462 of its 1536 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 3024 panos in both: median 0.01 m, p90 0.20 m
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
| fusion_server | 2970 | 1536 | 9636 | 3.24 | 0.964 (238/9) | 0.912 (237/260) | 7 | 0.938 | 210/34/16 | 0.09 (21) | 0.18 (46) | 26/3/3 | 1.72 / 4.22 / 12 |
| fusion_server+attach | 1843 | 1536 | 9636 | 5.23 | 0.963 (237/9) | 0.912 (237/260) | 7 | 0.938 | 210/34/16 | 0.09 (21) | 0.18 (46) | 26/3/3 | 1.72 / 4.22 / 12 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 210 of this run's 260 pool ramps are self-detected, so 81% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

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

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `deployed` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| deployed | unplaceable | 1428 | 0.79 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| deployed | cluster of 1 | 326 | 0.75 | 11 | 0.900 [0.60, 0.98] | 9 | 1 | 1 |
| deployed | cluster of 2 | 595 | 0.81 | 18 | 0.812 [0.57, 0.93] | 13 | 3 | 2 |
| deployed | cluster of 3+ | 7177 | 0.87 | 205 | 0.974 [0.94, 0.99] | 188 | 5 | 12 |
| fusion_server | unplaceable | 1428 | 0.79 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server | cluster of 1 | 337 | 0.74 | 11 | 0.900 [0.60, 0.98] | 9 | 1 | 1 |
| fusion_server | cluster of 2 | 458 | 0.76 | 17 | 0.786 [0.52, 0.92] | 11 | 3 | 3 |
| fusion_server | cluster of 3+ | 7303 | 0.87 | 206 | 0.974 [0.94, 0.99] | 190 | 5 | 11 |
| fusion_server+attach | unplaceable | 1428 | 0.79 | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server+attach | cluster of 1 | 279 | 0.73 | 8 | 0.857 [0.49, 0.97] | 6 | 1 | 1 |
| fusion_server+attach | cluster of 2 | 359 | 0.74 | 13 | 0.800 [0.49, 0.94] | 8 | 2 | 3 |
| fusion_server+attach | cluster of 3+ | 7460 | 0.87 | 213 | 0.970 [0.94, 0.99] | 196 | 6 | 11 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1432; attached 1127 (0.79)
- clusters: 2970 (`fusion_server`) -> 1843 (`fusion_server+attach`)
- sanity (not a metric): 33 unplaceable labels are on judged panos; 28 of them attached (24 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
