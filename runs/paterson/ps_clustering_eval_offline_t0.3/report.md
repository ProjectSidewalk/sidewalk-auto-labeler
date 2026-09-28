# paterson: PS label clustering vs RampNet GT (offline)

mode: offline -- 35107 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (3 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.1; results `results.jsonl` sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 125 judged panos -> 304 placeable points -> 304 ramps (0 cross-pano merges), 304 in the recall pool; raycast placed 37062 of 46312 detections (drops {'below_floor': 0, 'on_rig': 14, 'horizon': 129, 'out_of_range': 9107})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\paterson\results.jsonl`: sha256 `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 35107 of 35107
- `ps_streets.geojson`: 6462 features, sha256 `f52225848375d8d452465da7040026eccf4fc61780aac8f79430aa3ac7da4cb0`, 2026-09-28T17:22:43+00:00 (0.0 days old at run time), from https://sidewalk-paterson.cs.washington.edu/v3/api/streets?filetype=geojson
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2630 blocks, largest 93 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 29285 fusion members vs 29285 placeable server labels; 0 of 29285 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/paterson/fusion_eval/report.md (published in the 2.6 m frame): precision 0.975, recall (union) 0.961, dual 79/14/0
- fusion_server input: 35107 AI + 0 human labels on 12819 panos (12819 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 5822 labels the raycast cannot place (range cap, horizon) are singleton clusters; 10137 of its 10137 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 9188 panos in both: median 0.01 m, p90 0.25 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 18931 | 15794 | 35107 | 1.85 | 0.978 (271/6) | 0.964 (293/304) | 5 | 0.980 | 232/66/6 | 0.58 (267) | 0.84 (476) | 86/7/0 | 0.98 / 2.85 / 4 |
| ps @ 5 m | 12831 | 11194 | 35107 | 2.74 | 0.978 (271/6) | 0.944 (287/304) | 8 | 0.970 | 232/63/9 | 0.35 (117) | 0.56 (202) | 82/11/0 | 1.40 / 3.79 / 10 |
| ps @ 7.5 m | 10302 | 9495 | 35107 | 3.41 | 0.978 (271/6) | 0.934 (284/304) | 10 | 0.967 | 232/62/10 | 0.20 (62) | 0.36 (118) | 79/13/1 | 1.52 / 4.00 / 13 |
| ps @ 10 m | 9260 | 8861 | 35107 | 3.79 | 0.978 (271/6) | 0.928 (282/304) | 11 | 0.964 | 232/61/11 | 0.16 (45) | 0.30 (91) | 78/14/1 | 1.52 / 4.01 / 13 |
| ps @ 12.5 m | 8835 | 8586 | 35107 | 3.97 | 0.978 (271/6) | 0.905 (275/304) | 15 | 0.954 | 232/58/14 | 0.15 (40) | 0.25 (73) | 74/18/1 | 1.54 / 4.18 / 18 |
| ps @ 15 m | 8570 | 8402 | 35107 | 4.10 | 0.978 (271/6) | 0.901 (274/304) | 15 | 0.951 | 232/57/15 | 0.13 (35) | 0.24 (68) | 73/19/1 | 1.54 / 4.32 / 19 |
| ps_citywide @ 7.5 m | 9880 | 9124 | 35107 | 3.55 | 0.978 (271/6) | 0.924 (281/304) | 12 | 0.964 | 232/61/11 | 0.16 (47) | 0.29 (94) | 77/15/1 | 1.52 / 4.00 / 13 |
| ps_placeable @ 2.5 m | 15511 | 15511 | 29285 | 1.89 | 0.975 (233/6) | 0.964 (293/304) | 5 | 0.980 | 232/66/6 | 0.56 (259) | 0.83 (454) | 86/7/0 | 1.05 / 2.85 / 4 |
| ps_placeable @ 5 m | 10797 | 10797 | 29285 | 2.71 | 0.975 (233/6) | 0.944 (287/304) | 8 | 0.970 | 232/63/9 | 0.32 (106) | 0.54 (193) | 82/11/0 | 1.45 / 3.75 / 9 |
| ps_placeable @ 7.5 m | 9138 | 9138 | 29285 | 3.20 | 0.975 (233/6) | 0.938 (285/304) | 9 | 0.967 | 232/62/10 | 0.18 (56) | 0.34 (110) | 79/14/0 | 1.52 / 3.79 / 11 |
| ps_placeable @ 10 m | 8623 | 8623 | 29285 | 3.40 | 0.975 (233/6) | 0.934 (284/304) | 10 | 0.967 | 232/62/10 | 0.13 (38) | 0.27 (81) | 79/14/0 | 1.53 / 3.98 / 12 |
| ps_placeable @ 12.5 m | 8362 | 8362 | 29285 | 3.50 | 0.975 (233/6) | 0.911 (277/304) | 14 | 0.957 | 232/59/13 | 0.14 (38) | 0.24 (70) | 76/17/0 | 1.58 / 4.02 / 16 |
| ps_placeable @ 15 m | 8175 | 8175 | 29285 | 3.58 | 0.975 (233/6) | 0.905 (275/304) | 15 | 0.954 | 232/58/14 | 0.13 (35) | 0.23 (66) | 74/19/0 | 1.58 / 4.32 / 18 |
| ps_raycast @ 2.5 m | 17280 | 17280 | 29285 | 1.69 | 0.975 (233/6) | 0.977 (297/304) | 1 | 0.980 | 232/66/6 | 0.43 (163) | 0.80 (419) | 86/7/0 | 0.49 / 1.06 / 0 |
| ps_raycast @ 5 m | 12174 | 12174 | 29285 | 2.41 | 0.975 (233/6) | 0.961 (292/304) | 2 | 0.967 | 232/62/10 | 0.13 (38) | 0.40 (136) | 82/11/0 | 0.97 / 2.02 / 0 |
| ps_raycast @ 7.5 m | 9860 | 9860 | 29285 | 2.97 | 0.975 (233/6) | 0.947 (288/304) | 5 | 0.964 | 232/61/11 | 0.11 (31) | 0.28 (83) | 80/13/0 | 1.30 / 2.63 / 1 |
| ps_raycast @ 10 m | 8803 | 8803 | 29285 | 3.33 | 0.975 (233/6) | 0.928 (282/304) | 9 | 0.957 | 232/59/13 | 0.10 (28) | 0.21 (61) | 77/16/0 | 1.34 / 3.29 / 5 |
| ps_raycast @ 12.5 m | 8380 | 8380 | 29285 | 3.49 | 0.975 (233/6) | 0.924 (281/304) | 11 | 0.961 | 232/60/12 | 0.11 (30) | 0.21 (60) | 76/17/0 | 1.36 / 3.58 / 7 |
| ps_raycast @ 15 m | 8152 | 8152 | 29285 | 3.59 | 0.975 (233/6) | 0.901 (274/304) | 14 | 0.947 | 232/56/16 | 0.08 (23) | 0.20 (55) | 70/23/0 | 1.37 / 3.81 / 9 |
| fusion | 10137 | 10137 | 29285 | 2.89 | 0.975 (233/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.04 (12) | 0.17 (53) | 80/13/0 | 1.32 / 2.84 / 0 |
| fusion_refit | 10137 | 10137 | 29285 | 2.89 | 0.975 (233/6) | 0.951 (289/304) | 3 | 0.961 | 232/60/12 | 0.03 (10) | 0.18 (56) | 79/14/0 | 1.13 / 3.20 / 1 |
| fusion_server | 15959 | 10137 | 35107 | 2.20 | 0.978 (271/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.04 (12) | 0.17 (53) | 80/13/0 | 1.32 / 2.84 / 0 |
| fusion_server+attach | 10272 | 10137 | 35107 | 3.42 | 0.978 (271/6) | 0.954 (290/304) | 3 | 0.964 | 232/61/11 | 0.04 (12) | 0.17 (53) | 80/13/0 | 1.32 / 2.84 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 232 of this run's 304 pool ramps are self-detected, so 76% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 4980 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.5 m | 9.3 m | 0.19 | 0.07 | 0.00 |
| raycast | 4.3 m | 7.1 m | 0.07 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.720 | 111/219 | 53/93 | 0.753 | 63/229 | 47/93 |
| 5 | 0.934 | 103/284 | 79/93 | 0.954 | 48/290 | 80/93 |
| 7.5 | 0.977 | 97/297 | 87/93 | 0.984 | 44/299 | 88/93 |
| 10 | 0.984 | 97/299 | 88/93 | 0.987 | 44/300 | 89/93 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 5822 | 0.71 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 1561 | 0.50 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| ps @ 7.5 m | cluster of 2 | 2665 | 0.79 | 17 | 0.824 [0.59, 0.94] | 14 | 3 | 0 |
| ps @ 7.5 m | cluster of 3+ | 25059 | 0.87 | 226 | 0.986 [0.96, 1.00] | 215 | 3 | 8 |
| fusion_server | unplaceable | 5822 | 0.71 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server | cluster of 1 | 3580 | 0.76 | 18 | 0.889 [0.67, 0.97] | 16 | 2 | 0 |
| fusion_server | cluster of 2 | 3154 | 0.80 | 18 | 0.833 [0.61, 0.94] | 15 | 3 | 0 |
| fusion_server | cluster of 3+ | 22551 | 0.87 | 210 | 0.995 [0.97, 1.00] | 201 | 1 | 8 |
| fusion_server+attach | unplaceable | 5822 | 0.71 | 38 | 1.000 [0.91, 1.00] | 38 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 2954 | 0.75 | 16 | 0.875 [0.64, 0.97] | 14 | 2 | 0 |
| fusion_server+attach | cluster of 2 | 2501 | 0.78 | 10 | 0.800 [0.49, 0.94] | 8 | 2 | 0 |
| fusion_server+attach | cluster of 3+ | 23830 | 0.87 | 220 | 0.991 [0.97, 1.00] | 210 | 2 | 8 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 5822; attached 5687 (0.98)
- clusters: 15959 (`fusion_server`) -> 10272 (`fusion_server+attach`)
- sanity (not a metric): 38 unplaceable labels are on judged panos; 37 of them attached (37 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
