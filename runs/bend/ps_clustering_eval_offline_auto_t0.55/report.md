# bend: PS label clustering vs RampNet GT (offline)

mode: offline -- 51529 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (14 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153`
inputs: streets sha256 `none`; verdicts sha256 `9d9835db27f66c1904fcfd4fc87b06f26638a3c033ef10936387db40feb84b05`
raycast camera height auto; fusion arm at --min-confidence 0.55
- camera height mode `auto`: auto -> gsv-per-rig
- 78560 of 78560 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 1.96 m, 6 of 6 measured -> 2.5 m
  - 2008: median 1.78 m, 9 of 9 measured -> 2.5 m
  - 2009: median 1.83 m, 13 of 19 measured -> 2.5 m
  - 2011: median n/a, 0 of 8 measured -> 2.5 m
  - 2012: median 2.36 m, 166 of 3603 measured -> 2.5 m
  - 2015: median 2.37 m, 77 of 206 measured -> 2.5 m
  - 2017: median 2.39 m, 72 of 740 measured -> 2.5 m
  - 2018: median 2.33 m, 264 of 1451 measured -> 2.5 m
  - 2019: median 2.34 m, 1541 of 1666 measured -> 2.5 m
  - 2021: median 2.38 m, 909 of 1007 measured -> 2.5 m
  - 2024: median 2.37 m, 59928 of 65897 measured -> 2.5 m
  - 2025: median 2.35 m, 2881 of 3948 measured -> 2.5 m
GT: 110 judged panos -> 299 placeable points -> 298 ramps (1 cross-pano merges), 298 in the recall pool; raycast placed 49701 of 51543 detections (drops {'below_floor': 0, 'on_rig': 14, 'horizon': 3, 'out_of_range': 1825})

## Data provenance

- results file `runs/bend/results.jsonl`: sha256 `1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 51529 of 51529
- regions: none (no server), so every label is in one region and per-region == citywide
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 3999 blocks, largest 69 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 49701 fusion members vs 49701 placeable server labels; 0 of 49701 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at auto vs runs/bend/fusion_eval/report.md (published in the 2.6 m frame): precision 0.957, recall (union) 0.953, dual 28/6/0 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 51529 AI (by account; 0 share a detection with another AI label) + 0 human labels on 20611 panos (0 positioned by inverting their labels, 20611 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 1828 labels the raycast cannot place (range cap, horizon) are singleton clusters; 14058 of its 14058 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 51529 label ids, 51529 distinct, of 51529 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 17883 panos in both (all labels, mostly AI): median 0.010 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 24372 | 23290 | 51529 | 2.11 | 0.957 (247/11) | 0.966 (288/298) | 0 | 0.966 | 242/46/10 | 0.49 (182) | 0.64 (298) | 30/4/0 | 0.55 / 1.74 / 0 |
| ps @ 5 m | 17093 | 16594 | 51529 | 3.01 | 0.957 (247/11) | 0.950 (283/298) | 2 | 0.956 | 242/43/13 | 0.21 (66) | 0.35 (130) | 27/7/0 | 0.83 / 2.12 / 1 |
| ps @ 7.5 m | 14650 | 14476 | 51529 | 3.52 | 0.957 (247/11) | 0.943 (281/298) | 3 | 0.953 | 242/42/14 | 0.09 (26) | 0.21 (66) | 27/7/0 | 0.91 / 2.29 / 1 |
| ps @ 10 m | 13915 | 13838 | 51529 | 3.70 | 0.957 (247/11) | 0.936 (279/298) | 3 | 0.946 | 242/40/16 | 0.06 (18) | 0.15 (44) | 26/8/0 | 0.93 / 2.34 / 1 |
| ps @ 12.5 m | 13633 | 13585 | 51529 | 3.78 | 0.957 (247/11) | 0.930 (277/298) | 4 | 0.943 | 242/39/17 | 0.06 (16) | 0.13 (38) | 25/8/1 | 0.94 / 2.55 / 2 |
| ps @ 15 m | 13423 | 13389 | 51529 | 3.84 | 0.957 (247/11) | 0.926 (276/298) | 4 | 0.940 | 242/38/18 | 0.05 (15) | 0.13 (37) | 24/9/1 | 0.97 / 2.55 / 2 |
| ps_citywide @ 7.5 m | 14650 | 14476 | 51529 | 3.52 | 0.957 (247/11) | 0.943 (281/298) | 3 | 0.953 | 242/42/14 | 0.09 (26) | 0.21 (66) | 27/7/0 | 0.91 / 2.29 / 1 |
| ps_placeable @ 2.5 m | 23193 | 23193 | 49701 | 2.14 | 0.957 (242/11) | 0.966 (288/298) | 0 | 0.966 | 242/46/10 | 0.49 (182) | 0.64 (295) | 30/4/0 | 0.55 / 1.74 / 0 |
| ps_placeable @ 5 m | 16429 | 16429 | 49701 | 3.03 | 0.957 (242/11) | 0.950 (283/298) | 2 | 0.956 | 242/43/13 | 0.18 (58) | 0.33 (119) | 27/7/0 | 0.83 / 2.12 / 1 |
| ps_placeable @ 7.5 m | 14326 | 14326 | 49701 | 3.47 | 0.957 (242/11) | 0.940 (280/298) | 4 | 0.953 | 242/42/14 | 0.07 (22) | 0.18 (59) | 26/8/0 | 0.91 / 2.29 / 1 |
| ps_placeable @ 10 m | 13726 | 13726 | 49701 | 3.62 | 0.957 (242/11) | 0.933 (278/298) | 4 | 0.946 | 242/40/16 | 0.06 (16) | 0.14 (41) | 25/9/0 | 0.93 / 2.34 / 1 |
| ps_placeable @ 12.5 m | 13487 | 13487 | 49701 | 3.69 | 0.957 (242/11) | 0.926 (276/298) | 5 | 0.943 | 242/39/17 | 0.05 (15) | 0.12 (36) | 24/9/1 | 0.94 / 2.54 / 2 |
| ps_placeable @ 15 m | 13294 | 13294 | 49701 | 3.74 | 0.957 (242/11) | 0.923 (275/298) | 5 | 0.940 | 242/38/18 | 0.05 (15) | 0.12 (36) | 23/10/1 | 0.97 / 2.54 / 2 |
| ps_raycast @ 2.5 m | 21477 | 21477 | 49701 | 2.31 | 0.957 (243/11) | 0.960 (286/298) | 0 | 0.960 | 242/44/12 | 0.35 (112) | 0.65 (269) | 29/5/0 | 0.50 / 1.09 / 0 |
| ps_raycast @ 5 m | 15124 | 15124 | 49701 | 3.29 | 0.957 (242/11) | 0.953 (284/298) | 1 | 0.956 | 242/43/13 | 0.05 (14) | 0.21 (67) | 29/5/0 | 0.79 / 1.97 / 0 |
| ps_raycast @ 7.5 m | 13691 | 13691 | 49701 | 3.63 | 0.957 (242/11) | 0.950 (283/298) | 1 | 0.953 | 242/42/14 | 0.04 (11) | 0.12 (34) | 28/6/0 | 0.83 / 2.29 / 0 |
| ps_raycast @ 10 m | 13388 | 13388 | 49701 | 3.71 | 0.957 (242/11) | 0.943 (281/298) | 2 | 0.950 | 242/41/15 | 0.04 (11) | 0.11 (31) | 29/5/0 | 0.84 / 2.33 / 1 |
| ps_raycast @ 12.5 m | 13240 | 13240 | 49701 | 3.75 | 0.957 (242/11) | 0.936 (279/298) | 2 | 0.943 | 242/39/17 | 0.04 (10) | 0.11 (31) | 28/5/1 | 0.88 / 2.36 / 1 |
| ps_raycast @ 15 m | 13070 | 13070 | 49701 | 3.80 | 0.957 (242/11) | 0.933 (278/298) | 3 | 0.943 | 242/39/17 | 0.04 (11) | 0.11 (31) | 27/6/1 | 0.91 / 2.42 / 2 |
| fusion | 14058 | 14058 | 49701 | 3.54 | 0.957 (242/11) | 0.950 (283/298) | 2 | 0.956 | 242/43/13 | 0.05 (15) | 0.14 (44) | 29/5/0 | 0.89 / 2.24 / 1 |
| fusion_refit | 14058 | 14058 | 49701 | 3.54 | 0.957 (242/11) | 0.946 (282/298) | 2 | 0.953 | 242/42/14 | 0.05 (15) | 0.15 (45) | 28/6/0 | 0.70 / 2.41 / 1 |
| fusion_server | 15886 | 14058 | 51529 | 3.24 | 0.957 (247/11) | 0.950 (283/298) | 2 | 0.956 | 242/43/13 | 0.05 (15) | 0.14 (44) | 29/5/0 | 0.89 / 2.24 / 1 |
| fusion_server+attach | 14094 | 14058 | 51529 | 3.66 | 0.957 (247/11) | 0.950 (283/298) | 2 | 0.956 | 242/43/13 | 0.05 (15) | 0.14 (44) | 29/5/0 | 0.89 / 2.24 / 1 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 242 of this run's 298 pool ramps are self-detected, so 81% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 9742 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 3.8 m | 7.4 m | 0.10 | 0.02 | 0.00 |
| raycast | 3.1 m | 5.4 m | 0.01 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.879 | 53/262 | 24/34 | 0.872 | 39/260 | 24/34 |
| 5 | 0.943 | 58/281 | 27/34 | 0.950 | 41/283 | 29/34 |
| 7.5 | 0.960 | 56/286 | 29/34 | 0.960 | 39/286 | 29/34 |
| 10 | 0.963 | 56/287 | 29/34 | 0.963 | 39/287 | 29/34 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 1828 | 0.74 | 0.523 | 5 | 1.000 [0.57, 1.00] | 5 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 2036 | 0.71 | 0.568 | 20 | 0.737 [0.51, 0.88] | 14 | 5 | 1 |
| ps @ 7.5 m | cluster of 2 | 4349 | 0.82 | 0.570 | 28 | 0.889 [0.72, 0.96] | 24 | 3 | 1 |
| ps @ 7.5 m | cluster of 3+ | 43316 | 0.88 | 0.568 | 212 | 0.986 [0.96, 1.00] | 205 | 3 | 4 |
| fusion_server | unplaceable | 1828 | 0.74 | 0.523 | 5 | 1.000 [0.57, 1.00] | 5 | 0 | 0 |
| fusion_server | cluster of 1 | 2387 | 0.72 | 0.570 | 23 | 0.667 [0.45, 0.83] | 14 | 7 | 2 |
| fusion_server | cluster of 2 | 3858 | 0.82 | 0.584 | 22 | 0.952 [0.77, 0.99] | 20 | 1 | 1 |
| fusion_server | cluster of 3+ | 43456 | 0.88 | 0.568 | 215 | 0.986 [0.96, 1.00] | 209 | 3 | 3 |
| fusion_server+attach | unplaceable | 1828 | 0.74 | 0.523 | 5 | 1.000 [0.57, 1.00] | 5 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 2319 | 0.72 | 0.570 | 22 | 0.700 [0.48, 0.85] | 14 | 6 | 2 |
| fusion_server+attach | cluster of 2 | 3710 | 0.82 | 0.584 | 22 | 0.905 [0.71, 0.97] | 19 | 2 | 1 |
| fusion_server+attach | cluster of 3+ | 43672 | 0.88 | 0.568 | 216 | 0.986 [0.96, 1.00] | 210 | 3 | 3 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1828; attached 1792 (0.98)
- clusters: 15886 (`fusion_server`) -> 14094 (`fusion_server+attach`)
- sanity (not a metric): 5 unplaceable labels are on judged panos; 5 of them attached (5 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
