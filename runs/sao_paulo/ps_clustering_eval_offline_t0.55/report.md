# sao_paulo: PS label clustering vs RampNet GT (offline)

mode: offline -- 16158 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (1 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.2; results `results.jsonl` sha256 `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f`
inputs: streets sha256 `559b85105f088f246c7c7f0d52acbc5a08291f988050a1a0dddd311528c13d4f`; verdicts sha256 `dbe32bcc8f8d87b3caedcc54597b6547720d98202abe149fb707dd560aa0db4c`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 125 judged panos -> 255 placeable points -> 255 ramps (0 cross-pano merges), 255 in the recall pool; raycast placed 47186 of 51908 detections (drops {'below_floor': 0, 'on_rig': 43, 'horizon': 255, 'out_of_range': 4424})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\sao_paulo\results.jsonl`: sha256 `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 16158 of 16158
- `ps_streets.geojson`: 196335 features, sha256 `559b85105f088f246c7c7f0d52acbc5a08291f988050a1a0dddd311528c13d4f`, 2026-09-28T17:29:39+00:00 (0.2 days old at run time), from https://sidewalk-sao-paulo.cs.washington.edu/v3/api/streets?filetype=geojson; 655 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 1400 blocks, largest 122 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 15507 fusion members vs 15507 placeable server labels; 0 of 15507 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/sao_paulo/fusion_eval/report.md (published in the 2.6 m frame): precision 0.893, recall (union) 0.953, dual 50/7/1
- fusion_server input: 16158 AI + 0 human labels on 8045 panos (8045 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 651 labels the raycast cannot place (range cap, horizon) are singleton clusters; 4790 of its 4790 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 6604 panos in both: median 0.01 m, p90 0.20 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 8681 | 8327 | 16158 | 1.86 | 0.893 (191/23) | 0.969 (247/255) | 0 | 0.969 | 184/63/8 | 0.60 (261) | 0.79 (429) | 54/3/1 | 0.47 / 2.04 / 0 |
| ps @ 5 m | 6026 | 5843 | 16158 | 2.68 | 0.892 (190/23) | 0.945 (241/255) | 3 | 0.957 | 184/60/11 | 0.29 (85) | 0.45 (157) | 50/5/3 | 0.92 / 2.98 / 3 |
| ps @ 7.5 m | 4923 | 4833 | 16158 | 3.28 | 0.892 (190/23) | 0.925 (236/255) | 4 | 0.941 | 184/56/15 | 0.17 (41) | 0.30 (80) | 46/9/3 | 0.99 / 3.16 / 5 |
| ps @ 10 m | 4457 | 4406 | 16158 | 3.63 | 0.892 (190/23) | 0.914 (233/255) | 6 | 0.937 | 184/55/16 | 0.11 (26) | 0.21 (54) | 45/10/3 | 1.07 / 3.18 / 8 |
| ps @ 12.5 m | 4194 | 4162 | 16158 | 3.85 | 0.892 (190/23) | 0.902 (230/255) | 7 | 0.929 | 184/53/18 | 0.10 (22) | 0.17 (39) | 43/12/3 | 1.22 / 3.77 / 10 |
| ps @ 15 m | 4020 | 3999 | 16158 | 4.02 | 0.892 (190/23) | 0.882 (225/255) | 10 | 0.922 | 184/51/20 | 0.09 (21) | 0.16 (37) | 43/11/4 | 1.28 / 4.44 / 13 |
| ps_citywide @ 7.5 m | 4923 | 4833 | 16158 | 3.28 | 0.892 (190/23) | 0.925 (236/255) | 4 | 0.941 | 184/56/15 | 0.17 (41) | 0.30 (80) | 46/9/3 | 0.99 / 3.16 / 5 |
| ps_placeable @ 2.5 m | 8296 | 8296 | 15507 | 1.87 | 0.894 (185/22) | 0.969 (247/255) | 0 | 0.969 | 184/63/8 | 0.60 (255) | 0.79 (420) | 54/3/1 | 0.47 / 2.04 / 0 |
| ps_placeable @ 5 m | 5786 | 5786 | 15507 | 2.68 | 0.893 (184/22) | 0.945 (241/255) | 3 | 0.957 | 184/60/11 | 0.27 (81) | 0.44 (151) | 50/5/3 | 0.93 / 2.98 / 3 |
| ps_placeable @ 7.5 m | 4785 | 4785 | 15507 | 3.24 | 0.893 (184/22) | 0.925 (236/255) | 4 | 0.941 | 184/56/15 | 0.15 (38) | 0.29 (77) | 46/9/3 | 1.00 / 3.16 / 5 |
| ps_placeable @ 10 m | 4363 | 4363 | 15507 | 3.55 | 0.893 (184/22) | 0.910 (232/255) | 7 | 0.937 | 184/55/16 | 0.10 (25) | 0.19 (49) | 45/10/3 | 1.10 / 3.22 / 9 |
| ps_placeable @ 12.5 m | 4128 | 4128 | 15507 | 3.76 | 0.893 (184/22) | 0.894 (228/255) | 9 | 0.929 | 184/53/18 | 0.10 (22) | 0.17 (39) | 42/13/3 | 1.26 / 3.77 / 11 |
| ps_placeable @ 15 m | 3958 | 3958 | 15507 | 3.92 | 0.893 (184/22) | 0.875 (223/255) | 12 | 0.922 | 184/51/20 | 0.09 (21) | 0.17 (37) | 42/12/4 | 1.32 / 4.44 / 14 |
| ps_raycast @ 2.5 m | 7441 | 7441 | 15507 | 2.08 | 0.894 (185/22) | 0.969 (247/255) | 0 | 0.969 | 184/63/8 | 0.41 (123) | 0.72 (285) | 54/3/1 | 0.45 / 1.02 / 0 |
| ps_raycast @ 5 m | 5237 | 5237 | 15507 | 2.96 | 0.894 (185/22) | 0.945 (241/255) | 0 | 0.945 | 184/57/14 | 0.05 (13) | 0.25 (68) | 48/8/2 | 0.87 / 1.89 / 0 |
| ps_raycast @ 7.5 m | 4480 | 4480 | 15507 | 3.46 | 0.893 (184/22) | 0.922 (235/255) | 2 | 0.929 | 184/53/18 | 0.02 (5) | 0.12 (32) | 46/10/2 | 1.00 / 2.23 / 0 |
| ps_raycast @ 10 m | 4190 | 4190 | 15507 | 3.70 | 0.893 (184/22) | 0.910 (232/255) | 5 | 0.929 | 184/53/18 | 0.02 (5) | 0.12 (29) | 46/10/2 | 1.05 / 2.76 / 3 |
| ps_raycast @ 12.5 m | 4035 | 4035 | 15507 | 3.84 | 0.893 (184/22) | 0.906 (231/255) | 6 | 0.929 | 184/53/18 | 0.02 (5) | 0.12 (29) | 45/11/2 | 1.08 / 2.84 / 4 |
| ps_raycast @ 15 m | 3895 | 3895 | 15507 | 3.98 | 0.893 (184/22) | 0.882 (225/255) | 10 | 0.922 | 184/51/20 | 0.02 (5) | 0.11 (26) | 43/13/2 | 1.15 / 3.40 / 8 |
| fusion | 4790 | 4790 | 15507 | 3.24 | 0.893 (184/22) | 0.941 (240/255) | 3 | 0.953 | 184/59/12 | 0.07 (16) | 0.16 (45) | 50/7/1 | 1.08 / 2.80 / 1 |
| fusion_refit | 4790 | 4790 | 15507 | 3.24 | 0.893 (184/22) | 0.941 (240/255) | 3 | 0.953 | 184/59/12 | 0.07 (16) | 0.17 (44) | 50/7/1 | 0.96 / 2.84 / 1 |
| fusion_server | 5441 | 4790 | 16158 | 2.97 | 0.892 (190/23) | 0.941 (240/255) | 3 | 0.953 | 184/59/12 | 0.07 (16) | 0.16 (45) | 50/7/1 | 1.08 / 2.80 / 1 |
| fusion_server+attach | 4828 | 4790 | 16158 | 3.35 | 0.892 (190/23) | 0.941 (240/255) | 3 | 0.953 | 184/59/12 | 0.07 (16) | 0.16 (45) | 50/7/1 | 1.08 / 2.80 / 1 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 184 of this run's 255 pool ramps are self-detected, so 72% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 2371 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 5.1 m | 9.6 m | 0.23 | 0.08 | 0.00 |
| raycast | 3.8 m | 6.1 m | 0.04 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.800 | 70/204 | 41/58 | 0.816 | 42/208 | 42/58 |
| 5 | 0.925 | 71/236 | 46/58 | 0.941 | 39/240 | 50/58 |
| 7.5 | 0.949 | 71/242 | 50/58 | 0.961 | 38/245 | 52/58 |
| 10 | 0.953 | 71/243 | 50/58 | 0.969 | 37/247 | 53/58 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 651 | 0.69 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| ps @ 7.5 m | cluster of 1 | 1182 | 0.67 | 19 | 0.625 [0.39, 0.82] | 10 | 6 | 3 |
| ps @ 7.5 m | cluster of 2 | 1853 | 0.73 | 26 | 0.810 [0.60, 0.92] | 17 | 4 | 5 |
| ps @ 7.5 m | cluster of 3+ | 12472 | 0.81 | 195 | 0.929 [0.88, 0.96] | 157 | 12 | 26 |
| fusion_server | unplaceable | 651 | 0.69 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| fusion_server | cluster of 1 | 1593 | 0.67 | 22 | 0.529 [0.31, 0.74] | 9 | 8 | 5 |
| fusion_server | cluster of 2 | 1652 | 0.72 | 26 | 0.810 [0.60, 0.92] | 17 | 4 | 5 |
| fusion_server | cluster of 3+ | 12262 | 0.82 | 192 | 0.940 [0.89, 0.97] | 158 | 10 | 24 |
| fusion_server+attach | unplaceable | 651 | 0.69 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| fusion_server+attach | cluster of 1 | 1527 | 0.67 | 20 | 0.467 [0.25, 0.70] | 7 | 8 | 5 |
| fusion_server+attach | cluster of 2 | 1575 | 0.72 | 22 | 0.789 [0.57, 0.91] | 15 | 4 | 3 |
| fusion_server+attach | cluster of 3+ | 12405 | 0.81 | 198 | 0.942 [0.90, 0.97] | 162 | 10 | 26 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 651; attached 613 (0.94)
- clusters: 5441 (`fusion_server`) -> 4828 (`fusion_server+attach`)
- sanity (not a metric): 11 unplaceable labels are on judged panos; 9 of them attached (5 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
