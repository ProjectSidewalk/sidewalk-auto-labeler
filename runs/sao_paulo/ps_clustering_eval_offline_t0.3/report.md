# sao_paulo: PS label clustering vs RampNet GT (offline)

mode: offline -- 27338 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (11 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.2; results `results.jsonl` sha256 `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f`
inputs: streets sha256 `559b85105f088f246c7c7f0d52acbc5a08291f988050a1a0dddd311528c13d4f`; verdicts sha256 `dbe32bcc8f8d87b3caedcc54597b6547720d98202abe149fb707dd560aa0db4c`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 125 judged panos -> 255 placeable points -> 255 ramps (0 cross-pano merges), 255 in the recall pool; raycast placed 47186 of 51908 detections (drops {'below_floor': 0, 'on_rig': 43, 'horizon': 255, 'out_of_range': 4424})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\sao_paulo\results.jsonl`: sha256 `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 27338 of 27338
- `ps_streets.geojson`: 196335 features, sha256 `559b85105f088f246c7c7f0d52acbc5a08291f988050a1a0dddd311528c13d4f`, 2026-09-28T17:29:39+00:00 (0.2 days old at run time), from https://sidewalk-sao-paulo.cs.washington.edu/v3/api/streets?filetype=geojson; 655 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2075 blocks, largest 301 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 25656 fusion members vs 25656 placeable server labels; 0 of 25656 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/sao_paulo/fusion_eval/report.md (published in the 2.6 m frame): precision 0.893, recall (union) 0.973, dual 51/7/0
- fusion_server input: 27338 AI + 0 human labels on 12044 panos (12044 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 1682 labels the raycast cannot place (range cap, horizon) are singleton clusters; 8091 of its 8091 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 9600 panos in both: median 0.01 m, p90 0.24 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 15140 | 14212 | 27338 | 1.81 | 0.893 (191/23) | 0.988 (252/255) | 0 | 0.988 | 184/68/3 | 0.70 (359) | 0.88 (626) | 58/0/0 | 0.57 / 2.45 / 0 |
| ps @ 5 m | 10572 | 10057 | 27338 | 2.59 | 0.892 (190/23) | 0.973 (248/255) | 3 | 0.984 | 184/67/4 | 0.43 (140) | 0.67 (270) | 55/2/1 | 1.03 / 2.98 / 3 |
| ps @ 7.5 m | 8586 | 8291 | 27338 | 3.18 | 0.892 (190/23) | 0.965 (246/255) | 3 | 0.976 | 184/65/6 | 0.26 (70) | 0.48 (144) | 53/4/1 | 1.04 / 3.32 / 4 |
| ps @ 10 m | 7604 | 7434 | 27338 | 3.60 | 0.892 (190/23) | 0.957 (244/255) | 4 | 0.973 | 184/64/7 | 0.20 (51) | 0.39 (108) | 51/6/1 | 1.09 / 3.35 / 4 |
| ps @ 12.5 m | 7014 | 6912 | 27338 | 3.90 | 0.892 (190/23) | 0.937 (239/255) | 5 | 0.957 | 184/60/11 | 0.14 (35) | 0.31 (83) | 49/8/1 | 1.13 / 3.58 / 6 |
| ps @ 15 m | 6633 | 6581 | 27338 | 4.12 | 0.892 (190/23) | 0.933 (238/255) | 6 | 0.957 | 184/60/11 | 0.14 (34) | 0.30 (78) | 48/9/1 | 1.15 / 3.63 / 6 |
| ps_citywide @ 7.5 m | 8586 | 8291 | 27338 | 3.18 | 0.892 (190/23) | 0.965 (246/255) | 3 | 0.976 | 184/65/6 | 0.26 (70) | 0.48 (144) | 53/4/1 | 1.04 / 3.32 / 4 |
| ps_placeable @ 2.5 m | 14135 | 14135 | 25656 | 1.82 | 0.894 (185/22) | 0.988 (252/255) | 0 | 0.988 | 184/68/3 | 0.69 (354) | 0.88 (618) | 58/0/0 | 0.58 / 2.45 / 0 |
| ps_placeable @ 5 m | 9916 | 9916 | 25656 | 2.59 | 0.893 (184/22) | 0.973 (248/255) | 3 | 0.984 | 184/67/4 | 0.41 (136) | 0.65 (260) | 55/2/1 | 1.03 / 2.98 / 3 |
| ps_placeable @ 7.5 m | 8150 | 8150 | 25656 | 3.15 | 0.893 (184/22) | 0.965 (246/255) | 3 | 0.976 | 184/65/6 | 0.24 (63) | 0.45 (132) | 53/4/1 | 1.06 / 3.35 / 4 |
| ps_placeable @ 10 m | 7320 | 7320 | 25656 | 3.50 | 0.893 (184/22) | 0.957 (244/255) | 4 | 0.973 | 184/64/7 | 0.19 (48) | 0.36 (102) | 51/6/1 | 1.12 / 3.49 / 4 |
| ps_placeable @ 12.5 m | 6796 | 6796 | 25656 | 3.78 | 0.893 (184/22) | 0.937 (239/255) | 5 | 0.957 | 184/60/11 | 0.14 (35) | 0.30 (80) | 50/7/1 | 1.18 / 3.58 / 6 |
| ps_placeable @ 15 m | 6465 | 6465 | 25656 | 3.97 | 0.893 (184/22) | 0.933 (238/255) | 6 | 0.957 | 184/60/11 | 0.14 (34) | 0.29 (76) | 49/8/1 | 1.22 / 3.67 / 6 |
| ps_raycast @ 2.5 m | 12788 | 12788 | 25656 | 2.01 | 0.894 (185/22) | 0.988 (252/255) | 0 | 0.988 | 184/68/3 | 0.52 (179) | 0.84 (430) | 58/0/0 | 0.49 / 1.06 / 0 |
| ps_raycast @ 5 m | 8955 | 8955 | 25656 | 2.86 | 0.894 (185/22) | 0.973 (248/255) | 0 | 0.973 | 184/64/7 | 0.15 (39) | 0.45 (138) | 53/5/0 | 0.91 / 1.89 / 0 |
| ps_raycast @ 7.5 m | 7492 | 7492 | 25656 | 3.42 | 0.893 (184/22) | 0.953 (243/255) | 2 | 0.961 | 184/61/10 | 0.10 (25) | 0.28 (73) | 50/8/0 | 1.08 / 2.47 / 0 |
| ps_raycast @ 10 m | 6833 | 6833 | 25656 | 3.75 | 0.893 (184/22) | 0.945 (241/255) | 3 | 0.957 | 184/60/11 | 0.10 (23) | 0.24 (61) | 50/8/0 | 1.14 / 2.83 / 1 |
| ps_raycast @ 12.5 m | 6472 | 6472 | 25656 | 3.96 | 0.893 (184/22) | 0.929 (237/255) | 6 | 0.953 | 184/59/12 | 0.09 (22) | 0.24 (60) | 47/11/0 | 1.16 / 3.09 / 4 |
| ps_raycast @ 15 m | 6219 | 6219 | 25656 | 4.13 | 0.893 (184/22) | 0.922 (235/255) | 8 | 0.953 | 184/59/12 | 0.09 (22) | 0.24 (59) | 46/12/0 | 1.17 / 3.40 / 6 |
| fusion | 8091 | 8091 | 25656 | 3.17 | 0.893 (184/22) | 0.961 (245/255) | 3 | 0.973 | 184/64/7 | 0.10 (25) | 0.29 (78) | 51/7/0 | 1.14 / 3.02 / 2 |
| fusion_refit | 8091 | 8091 | 25656 | 3.17 | 0.893 (184/22) | 0.957 (244/255) | 4 | 0.973 | 184/64/7 | 0.11 (27) | 0.30 (80) | 51/7/0 | 1.00 / 2.98 / 3 |
| fusion_server | 9773 | 8091 | 27338 | 2.80 | 0.892 (190/23) | 0.961 (245/255) | 3 | 0.973 | 184/64/7 | 0.10 (25) | 0.29 (78) | 51/7/0 | 1.14 / 3.02 / 2 |
| fusion_server+attach | 8194 | 8091 | 27338 | 3.34 | 0.892 (190/23) | 0.961 (245/255) | 3 | 0.973 | 184/64/7 | 0.10 (25) | 0.29 (78) | 51/7/0 | 1.14 / 3.02 / 2 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 184 of this run's 255 pool ramps are self-detected, so 72% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 3566 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 6.2 m | 11.4 m | 0.37 | 0.19 | 0.01 |
| raycast | 4.2 m | 6.9 m | 0.06 | 0.01 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.816 | 115/208 | 41/58 | 0.831 | 66/212 | 44/58 |
| 5 | 0.965 | 117/246 | 53/58 | 0.961 | 70/245 | 51/58 |
| 7.5 | 0.980 | 117/250 | 55/58 | 0.976 | 70/249 | 54/58 |
| 10 | 0.984 | 117/251 | 55/58 | 0.984 | 69/251 | 55/58 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1682 | 0.49 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| ps @ 7.5 m | cluster of 1 | 2350 | 0.42 | 13 | 0.500 [0.24, 0.76] | 5 | 5 | 3 |
| ps @ 7.5 m | cluster of 2 | 2976 | 0.51 | 17 | 0.786 [0.52, 0.92] | 11 | 3 | 3 |
| ps @ 7.5 m | cluster of 3+ | 20330 | 0.69 | 210 | 0.923 [0.88, 0.95] | 168 | 14 | 28 |
| fusion_server | unplaceable | 1682 | 0.49 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| fusion_server | cluster of 1 | 3294 | 0.42 | 17 | 0.462 [0.23, 0.71] | 6 | 7 | 4 |
| fusion_server | cluster of 2 | 2462 | 0.50 | 13 | 0.818 [0.52, 0.95] | 9 | 2 | 2 |
| fusion_server | cluster of 3+ | 19900 | 0.71 | 210 | 0.929 [0.88, 0.96] | 169 | 13 | 28 |
| fusion_server+attach | unplaceable | 1682 | 0.49 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| fusion_server+attach | cluster of 1 | 3066 | 0.42 | 15 | 0.364 [0.15, 0.65] | 4 | 7 | 4 |
| fusion_server+attach | cluster of 2 | 2327 | 0.49 | 10 | 0.875 [0.53, 0.98] | 7 | 1 | 2 |
| fusion_server+attach | cluster of 3+ | 20263 | 0.70 | 215 | 0.925 [0.88, 0.95] | 173 | 14 | 28 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1682; attached 1579 (0.94)
- clusters: 9773 (`fusion_server`) -> 8194 (`fusion_server+attach`)
- sanity (not a metric): 11 unplaceable labels are on judged panos; 10 of them attached (6 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
