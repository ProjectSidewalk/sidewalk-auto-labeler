# sao_paulo: PS label clustering vs RampNet GT (offline)

mode: offline -- 16158 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (1 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f`
inputs: streets sha256 `559b85105f088f246c7c7f0d52acbc5a08291f988050a1a0dddd311528c13d4f`; verdicts sha256 `dbe32bcc8f8d87b3caedcc54597b6547720d98202abe149fb707dd560aa0db4c`
raycast camera height auto; fusion arm at --min-confidence 0.55
- camera height mode `auto`: auto -> gsv-per-rig
- 30034 of 30034 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2010: median 2.20 m, 2 of 92 measured -> 2.5 m
  - 2011: median 2.37 m, 28 of 83 measured -> 2.5 m
  - 2014: median 2.38 m, 36 of 815 measured -> 2.5 m
  - 2015: median 2.38 m, 3 of 9 measured -> 2.5 m
  - 2016: median 1.74 m, 64 of 108 measured -> 2 m
  - 2017: median 1.72 m, 174 of 195 measured -> 2 m
  - 2018: median 2.10 m, 152 of 172 measured -> 2.5 m
  - 2019: median 2.25 m, 131 of 146 measured -> 2.5 m
  - 2020: median 2.15 m, 83 of 90 measured -> 2.5 m
  - 2021: median 1.65 m, 876 of 992 measured -> 2 m
  - 2022: median 2.22 m, 714 of 807 measured -> 2.5 m
  - 2023: median 2.26 m, 1443 of 1636 measured -> 2.5 m
  - 2024: median 2.27 m, 7049 of 14606 measured -> 2.5 m
  - 2025: median 2.23 m, 7593 of 9963 measured -> 2.5 m
  - 2026: median 2.13 m, 253 of 320 measured -> 2.5 m
GT: 125 judged panos -> 257 placeable points -> 257 ramps (0 cross-pano merges), 257 in the recall pool; raycast placed 47186 of 51908 detections (drops {'below_floor': 0, 'on_rig': 43, 'horizon': 255, 'out_of_range': 4424})

## Data provenance

- results file `runs/sao_paulo/results.jsonl`: sha256 `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 16158 of 16158
- `ps_streets.geojson`: 196335 features, sha256 `559b85105f088f246c7c7f0d52acbc5a08291f988050a1a0dddd311528c13d4f`, 2026-09-28T17:29:39+00:00 (6.1 days old at run time), from https://sidewalk-sao-paulo.cs.washington.edu/v3/api/streets?filetype=geojson; 655 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 1400 blocks, largest 122 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 15507 fusion members vs 15507 placeable server labels; 0 of 15507 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at auto vs runs/sao_paulo/fusion_eval/report.md (published in the 2.6 m frame): precision 0.893, recall (union) 0.949, dual 51/8/1 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 16158 AI (by account; 0 share a detection with another AI label) + 0 human labels on 8045 panos (0 positioned by inverting their labels, 8045 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 651 labels the raycast cannot place (range cap, horizon) are singleton clusters; 4855 of its 4855 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 16158 label ids, 16158 distinct, of 16158 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 6604 panos in both (all labels, mostly AI): median 0.010 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 8681 | 8327 | 16158 | 1.86 | 0.893 (191/23) | 0.965 (248/257) | 0 | 0.965 | 184/64/9 | 0.60 (245) | 0.78 (420) | 56/3/1 | 0.56 / 1.61 / 0 |
| ps @ 5 m | 6026 | 5843 | 16158 | 2.68 | 0.892 (190/23) | 0.949 (244/257) | 2 | 0.957 | 184/62/11 | 0.27 (73) | 0.43 (147) | 53/5/2 | 0.91 / 2.53 / 0 |
| ps @ 7.5 m | 4923 | 4833 | 16158 | 3.28 | 0.892 (190/23) | 0.926 (238/257) | 4 | 0.942 | 184/58/15 | 0.13 (33) | 0.29 (74) | 48/10/2 | 0.99 / 2.69 / 3 |
| ps @ 10 m | 4457 | 4406 | 16158 | 3.63 | 0.892 (190/23) | 0.914 (235/257) | 6 | 0.938 | 184/57/16 | 0.08 (19) | 0.20 (50) | 47/11/2 | 1.12 / 2.69 / 5 |
| ps @ 12.5 m | 4194 | 4162 | 16158 | 3.85 | 0.892 (190/23) | 0.903 (232/257) | 7 | 0.930 | 184/55/18 | 0.06 (15) | 0.16 (38) | 45/13/2 | 1.23 / 3.36 / 7 |
| ps @ 15 m | 4020 | 3999 | 16158 | 4.02 | 0.892 (190/23) | 0.887 (228/257) | 10 | 0.926 | 184/54/19 | 0.07 (15) | 0.16 (37) | 45/12/3 | 1.26 / 3.91 / 10 |
| ps_citywide @ 7.5 m | 4923 | 4833 | 16158 | 3.28 | 0.892 (190/23) | 0.926 (238/257) | 4 | 0.942 | 184/58/15 | 0.13 (33) | 0.29 (74) | 48/10/2 | 0.99 / 2.69 / 3 |
| ps_placeable @ 2.5 m | 8296 | 8296 | 15507 | 1.87 | 0.894 (185/22) | 0.965 (248/257) | 0 | 0.965 | 184/64/9 | 0.60 (238) | 0.78 (411) | 56/3/1 | 0.57 / 1.61 / 0 |
| ps_placeable @ 5 m | 5786 | 5786 | 15507 | 2.68 | 0.893 (184/22) | 0.949 (244/257) | 2 | 0.957 | 184/62/11 | 0.25 (67) | 0.42 (141) | 53/5/2 | 0.91 / 2.53 / 0 |
| ps_placeable @ 7.5 m | 4785 | 4785 | 15507 | 3.24 | 0.893 (184/22) | 0.926 (238/257) | 4 | 0.942 | 184/58/15 | 0.12 (31) | 0.27 (71) | 48/10/2 | 0.99 / 2.69 / 3 |
| ps_placeable @ 10 m | 4363 | 4363 | 15507 | 3.55 | 0.893 (184/22) | 0.914 (235/257) | 6 | 0.938 | 184/57/16 | 0.07 (18) | 0.18 (45) | 47/11/2 | 1.16 / 2.70 / 5 |
| ps_placeable @ 12.5 m | 4128 | 4128 | 15507 | 3.76 | 0.893 (184/22) | 0.899 (231/257) | 8 | 0.930 | 184/55/18 | 0.06 (15) | 0.16 (38) | 44/14/2 | 1.26 / 3.36 / 7 |
| ps_placeable @ 15 m | 3958 | 3958 | 15507 | 3.92 | 0.893 (184/22) | 0.883 (227/257) | 11 | 0.926 | 184/54/19 | 0.07 (15) | 0.16 (37) | 44/13/3 | 1.30 / 3.91 / 10 |
| ps_raycast @ 2.5 m | 7489 | 7489 | 15507 | 2.07 | 0.894 (185/22) | 0.965 (248/257) | 0 | 0.965 | 184/64/9 | 0.42 (121) | 0.70 (297) | 56/3/1 | 0.47 / 1.02 / 0 |
| ps_raycast @ 5 m | 5224 | 5224 | 15507 | 2.97 | 0.893 (184/22) | 0.938 (241/257) | 2 | 0.946 | 184/59/14 | 0.07 (17) | 0.27 (74) | 51/7/2 | 0.93 / 1.93 / 0 |
| ps_raycast @ 7.5 m | 4454 | 4454 | 15507 | 3.48 | 0.893 (184/22) | 0.930 (239/257) | 2 | 0.938 | 184/57/16 | 0.04 (10) | 0.14 (34) | 50/8/2 | 1.04 / 2.26 / 0 |
| ps_raycast @ 10 m | 4198 | 4198 | 15507 | 3.69 | 0.893 (184/22) | 0.911 (234/257) | 6 | 0.934 | 184/56/17 | 0.05 (11) | 0.13 (30) | 48/10/2 | 1.16 / 2.61 / 3 |
| ps_raycast @ 12.5 m | 4043 | 4043 | 15507 | 3.84 | 0.893 (184/22) | 0.907 (233/257) | 7 | 0.934 | 184/56/17 | 0.05 (11) | 0.12 (29) | 47/11/2 | 1.19 / 2.70 / 4 |
| ps_raycast @ 15 m | 3896 | 3896 | 15507 | 3.98 | 0.893 (184/22) | 0.891 (229/257) | 10 | 0.930 | 184/55/18 | 0.05 (11) | 0.12 (28) | 46/12/2 | 1.26 / 3.49 / 8 |
| fusion | 4855 | 4855 | 15507 | 3.19 | 0.893 (184/22) | 0.942 (242/257) | 3 | 0.953 | 184/61/12 | 0.05 (13) | 0.17 (46) | 51/8/1 | 1.12 / 2.50 / 1 |
| fusion_refit | 4855 | 4855 | 15507 | 3.19 | 0.893 (184/22) | 0.938 (241/257) | 3 | 0.949 | 184/60/13 | 0.06 (14) | 0.18 (47) | 51/8/1 | 1.02 / 2.56 / 1 |
| fusion_server | 5506 | 4855 | 16158 | 2.93 | 0.892 (190/23) | 0.942 (242/257) | 3 | 0.953 | 184/61/12 | 0.05 (13) | 0.17 (46) | 51/8/1 | 1.12 / 2.50 / 1 |
| fusion_server+attach | 4894 | 4855 | 16158 | 3.30 | 0.892 (190/23) | 0.942 (242/257) | 3 | 0.953 | 184/61/12 | 0.05 (13) | 0.17 (46) | 51/8/1 | 1.12 / 2.50 / 1 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 184 of this run's 257 pool ramps are self-detected, so 72% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 2384 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.8 m | 9.1 m | 0.20 | 0.06 | 0.00 |
| raycast | 3.7 m | 5.9 m | 0.02 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.805 | 66/207 | 40/60 | 0.813 | 46/209 | 40/60 |
| 5 | 0.926 | 68/238 | 48/60 | 0.942 | 42/242 | 51/60 |
| 7.5 | 0.949 | 67/244 | 52/60 | 0.957 | 42/246 | 53/60 |
| 10 | 0.953 | 67/245 | 52/60 | 0.965 | 40/248 | 54/60 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 651 | 0.69 | 0.523 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| ps @ 7.5 m | cluster of 1 | 1182 | 0.67 | 0.568 | 19 | 0.625 [0.39, 0.82] | 10 | 6 | 3 |
| ps @ 7.5 m | cluster of 2 | 1853 | 0.73 | 0.568 | 26 | 0.810 [0.60, 0.92] | 17 | 4 | 5 |
| ps @ 7.5 m | cluster of 3+ | 12472 | 0.81 | 0.568 | 195 | 0.929 [0.88, 0.96] | 157 | 12 | 26 |
| fusion_server | unplaceable | 651 | 0.69 | 0.523 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| fusion_server | cluster of 1 | 1633 | 0.67 | 0.570 | 24 | 0.556 [0.34, 0.75] | 10 | 8 | 6 |
| fusion_server | cluster of 2 | 1676 | 0.73 | 0.568 | 25 | 0.800 [0.58, 0.92] | 16 | 4 | 5 |
| fusion_server | cluster of 3+ | 12198 | 0.82 | 0.568 | 191 | 0.940 [0.89, 0.97] | 158 | 10 | 23 |
| fusion_server+attach | unplaceable | 651 | 0.69 | 0.523 | 11 | 0.857 [0.49, 0.97] | 6 | 1 | 4 |
| fusion_server+attach | cluster of 1 | 1562 | 0.67 | 0.570 | 22 | 0.500 [0.28, 0.72] | 8 | 8 | 6 |
| fusion_server+attach | cluster of 2 | 1614 | 0.72 | 0.568 | 22 | 0.789 [0.57, 0.91] | 15 | 4 | 3 |
| fusion_server+attach | cluster of 3+ | 12331 | 0.81 | 0.568 | 196 | 0.942 [0.90, 0.97] | 161 | 10 | 25 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 651; attached 612 (0.94)
- clusters: 5506 (`fusion_server`) -> 4894 (`fusion_server+attach`)
- sanity (not a metric): 11 unplaceable labels are on judged panos; 9 of them attached (5 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
