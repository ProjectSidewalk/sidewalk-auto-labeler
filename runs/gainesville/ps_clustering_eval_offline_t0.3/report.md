# gainesville: PS label clustering vs RampNet GT (offline)

mode: offline -- 24072 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (10 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.1; results `results.jsonl` sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 125 judged panos -> 219 placeable points -> 219 ramps (0 cross-pano merges), 219 in the recall pool; raycast placed 37056 of 43871 detections (drops {'below_floor': 0, 'on_rig': 35, 'horizon': 23, 'out_of_range': 6757})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\gainesville\results.jsonl`: sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 24072 of 24072
- `ps_streets.geojson`: 27448 features, sha256 `39b778bb4a85bb8a3fa42f89adfff9ffc2b0615be403534b3bab3dd03b602ed3`, 2026-09-28T17:25:58+00:00 (0.0 days old at run time), from https://sidewalk-gainesville.cs.washington.edu/v3/api/streets?filetype=geojson
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 2794 blocks, largest 89 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 21176 fusion members vs 21176 placeable server labels; 0 of 21176 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/gainesville/fusion_eval/report.md (published in the 2.6 m frame): precision 0.956, recall (union) 0.968, dual 6/3/0
- fusion_server input: 24072 AI + 0 human labels on 11453 panos (11453 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 2896 labels the raycast cannot place (range cap, horizon) are singleton clusters; 8967 of its 8967 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 8352 panos in both: median 0.01 m, p90 0.33 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 14001 | 12593 | 24072 | 1.72 | 0.955 (191/9) | 0.973 (213/219) | 1 | 0.977 | 172/42/5 | 0.37 (114) | 0.67 (268) | 8/1/0 | 1.30 / 3.05 / 2 |
| ps @ 5 m | 9670 | 8936 | 24072 | 2.49 | 0.955 (190/9) | 0.932 (204/219) | 8 | 0.968 | 172/40/7 | 0.20 (53) | 0.42 (130) | 6/3/0 | 1.93 / 4.46 / 10 |
| ps @ 7.5 m | 7936 | 7548 | 24072 | 3.03 | 0.955 (190/9) | 0.913 (200/219) | 12 | 0.968 | 172/40/7 | 0.14 (29) | 0.28 (72) | 6/3/0 | 2.03 / 4.90 / 15 |
| ps @ 10 m | 7215 | 6986 | 24072 | 3.34 | 0.955 (190/9) | 0.900 (197/219) | 14 | 0.963 | 172/39/8 | 0.11 (23) | 0.24 (57) | 6/3/0 | 2.17 / 4.92 / 16 |
| ps @ 12.5 m | 6820 | 6666 | 24072 | 3.53 | 0.955 (190/9) | 0.877 (192/219) | 18 | 0.959 | 172/38/9 | 0.10 (19) | 0.23 (52) | 5/4/0 | 2.27 / 5.20 / 20 |
| ps @ 15 m | 6571 | 6448 | 24072 | 3.66 | 0.955 (190/9) | 0.868 (190/219) | 19 | 0.954 | 172/37/10 | 0.09 (17) | 0.23 (50) | 5/4/0 | 2.30 / 5.23 / 21 |
| ps_citywide @ 7.5 m | 7781 | 7405 | 24072 | 3.09 | 0.955 (190/9) | 0.904 (198/219) | 13 | 0.963 | 172/39/8 | 0.12 (24) | 0.26 (65) | 6/3/0 | 1.99 / 4.92 / 16 |
| ps_placeable @ 2.5 m | 12433 | 12433 | 21176 | 1.70 | 0.956 (174/8) | 0.973 (213/219) | 1 | 0.977 | 172/42/5 | 0.36 (111) | 0.65 (262) | 8/1/0 | 1.30 / 3.07 / 2 |
| ps_placeable @ 5 m | 8743 | 8743 | 21176 | 2.42 | 0.956 (173/8) | 0.932 (204/219) | 8 | 0.968 | 172/40/7 | 0.19 (49) | 0.39 (121) | 6/3/0 | 1.86 / 4.46 / 10 |
| ps_placeable @ 7.5 m | 7389 | 7389 | 21176 | 2.87 | 0.956 (173/8) | 0.918 (201/219) | 11 | 0.968 | 172/40/7 | 0.11 (24) | 0.24 (63) | 6/3/0 | 2.04 / 4.90 / 15 |
| ps_placeable @ 10 m | 6842 | 6842 | 21176 | 3.10 | 0.956 (173/8) | 0.904 (198/219) | 13 | 0.963 | 172/39/8 | 0.09 (18) | 0.22 (51) | 6/3/0 | 2.13 / 4.92 / 16 |
| ps_placeable @ 12.5 m | 6533 | 6533 | 21176 | 3.24 | 0.956 (173/8) | 0.881 (193/219) | 17 | 0.959 | 172/38/9 | 0.09 (17) | 0.21 (46) | 5/4/0 | 2.30 / 5.20 / 19 |
| ps_placeable @ 15 m | 6316 | 6316 | 21176 | 3.35 | 0.956 (173/8) | 0.872 (191/219) | 18 | 0.954 | 172/37/10 | 0.08 (15) | 0.20 (43) | 5/4/0 | 2.32 / 5.25 / 20 |
| ps_raycast @ 2.5 m | 14230 | 14230 | 21176 | 1.49 | 0.956 (174/8) | 0.986 (216/219) | 0 | 0.986 | 172/44/3 | 0.36 (90) | 0.76 (307) | 9/0/0 | 0.47 / 1.19 / 0 |
| ps_raycast @ 5 m | 10161 | 10161 | 21176 | 2.08 | 0.956 (174/8) | 0.977 (214/219) | 0 | 0.977 | 172/42/5 | 0.08 (19) | 0.41 (105) | 7/2/0 | 1.17 / 2.13 / 0 |
| ps_raycast @ 7.5 m | 8210 | 8210 | 21176 | 2.58 | 0.956 (173/8) | 0.968 (212/219) | 0 | 0.968 | 172/40/7 | 0.05 (10) | 0.17 (42) | 7/2/0 | 1.36 / 3.04 / 0 |
| ps_raycast @ 10 m | 7135 | 7135 | 21176 | 2.97 | 0.956 (173/8) | 0.936 (205/219) | 6 | 0.963 | 172/39/8 | 0.05 (10) | 0.14 (30) | 5/3/1 | 1.69 / 4.26 / 6 |
| ps_raycast @ 12.5 m | 6570 | 6570 | 21176 | 3.22 | 0.956 (173/8) | 0.927 (203/219) | 8 | 0.963 | 172/39/8 | 0.04 (9) | 0.13 (27) | 5/3/1 | 1.93 / 4.47 / 8 |
| ps_raycast @ 15 m | 6315 | 6315 | 21176 | 3.35 | 0.956 (173/8) | 0.886 (194/219) | 15 | 0.954 | 172/37/10 | 0.05 (10) | 0.14 (27) | 5/2/2 | 2.10 / 4.88 / 15 |
| fusion | 8967 | 8967 | 21176 | 2.36 | 0.956 (173/8) | 0.954 (209/219) | 3 | 0.968 | 172/40/7 | 0.06 (14) | 0.23 (59) | 6/3/0 | 1.23 / 2.82 / 3 |
| fusion_refit | 8967 | 8967 | 21176 | 2.36 | 0.956 (173/8) | 0.954 (209/219) | 3 | 0.968 | 172/40/7 | 0.06 (12) | 0.25 (66) | 6/3/0 | 1.18 / 3.13 / 4 |
| fusion_server | 11863 | 8967 | 24072 | 2.03 | 0.955 (190/9) | 0.954 (209/219) | 3 | 0.968 | 172/40/7 | 0.06 (14) | 0.23 (59) | 6/3/0 | 1.23 / 2.82 / 3 |
| fusion_server+attach | 9142 | 8967 | 24072 | 2.63 | 0.955 (190/9) | 0.954 (209/219) | 3 | 0.968 | 172/40/7 | 0.06 (14) | 0.23 (59) | 6/3/0 | 1.23 / 2.82 / 3 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 172 of this run's 219 pool ramps are self-detected, so 79% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 3034 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.6 m | 9.9 m | 0.23 | 0.10 | 0.01 |
| raycast | 4.5 m | 7.4 m | 0.09 | 0.01 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.612 | 49/134 | 4/9 | 0.822 | 40/180 | 5/9 |
| 5 | 0.913 | 55/200 | 6/9 | 0.954 | 48/209 | 6/9 |
| 7.5 | 0.986 | 52/216 | 9/9 | 0.986 | 48/216 | 8/9 |
| 10 | 0.995 | 51/218 | 9/9 | 1.000 | 47/219 | 9/9 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 2896 | 0.55 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| ps @ 7.5 m | cluster of 1 | 2104 | 0.43 | 4 | 0.667 [0.21, 0.94] | 2 | 1 | 1 |
| ps @ 7.5 m | cluster of 2 | 2551 | 0.56 | 20 | 0.944 [0.74, 0.99] | 17 | 1 | 2 |
| ps @ 7.5 m | cluster of 3+ | 16521 | 0.75 | 163 | 0.962 [0.92, 0.98] | 153 | 6 | 4 |
| fusion_server | unplaceable | 2896 | 0.55 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| fusion_server | cluster of 1 | 4212 | 0.52 | 32 | 0.862 [0.69, 0.95] | 25 | 4 | 3 |
| fusion_server | cluster of 2 | 3442 | 0.64 | 25 | 0.955 [0.78, 0.99] | 21 | 1 | 3 |
| fusion_server | cluster of 3+ | 13522 | 0.75 | 130 | 0.977 [0.93, 0.99] | 126 | 3 | 1 |
| fusion_server+attach | unplaceable | 2896 | 0.55 | 18 | 0.944 [0.74, 0.99] | 17 | 1 | 0 |
| fusion_server+attach | cluster of 1 | 3814 | 0.52 | 29 | 0.846 [0.66, 0.94] | 22 | 4 | 3 |
| fusion_server+attach | cluster of 2 | 3064 | 0.61 | 24 | 0.952 [0.77, 0.99] | 20 | 1 | 3 |
| fusion_server+attach | cluster of 3+ | 14298 | 0.75 | 134 | 0.977 [0.94, 0.99] | 130 | 3 | 1 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 2896; attached 2721 (0.94)
- clusters: 11863 (`fusion_server`) -> 9142 (`fusion_server+attach`)
- sanity (not a metric): 18 unplaceable labels are on judged panos; 16 of them attached (16 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
