# annapolis: PS label clustering vs RampNet GT (offline)

mode: offline -- 29188 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (0 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.1; results `results.jsonl` sha256 `f1b228f742dc95909067cc3ba9b045f1acf76617747b6e21eb9e7df58335b9dd`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 125 judged panos -> 241 placeable points -> 241 ramps (0 cross-pano merges), 241 in the recall pool; raycast placed 25927 of 29188 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 23, 'out_of_range': 3238})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\annapolis\results.jsonl`: sha256 `f1b228f742dc95909067cc3ba9b045f1acf76617747b6e21eb9e7df58335b9dd`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 29188 of 29188
- regions: none (no server), so every label is in one region and per-region == citywide
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 1283 blocks, largest 273 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 25927 fusion members vs 25927 placeable server labels; 0 of 25927 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/annapolis/fusion_eval/report.md (published in the 2.6 m frame): precision 0.980, recall (union) 0.934, dual 15/8/1
- fusion_server input: 29188 AI + 0 human labels on 14884 panos (14884 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 3261 labels the raycast cannot place (range cap, horizon) are singleton clusters; 3979 of its 4030 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 12352 panos in both: median 0.01 m, p90 0.19 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 10271 | 9715 | 29188 | 2.84 | 0.977 (217/5) | 0.963 (232/241) | 1 | 0.967 | 194/39/8 | 0.58 (196) | 0.83 (419) | 23/0/1 | 1.00 / 2.51 / 2 |
| ps @ 5 m | 5972 | 5744 | 29188 | 4.89 | 0.977 (217/5) | 0.934 (225/241) | 6 | 0.959 | 194/37/10 | 0.25 (61) | 0.55 (159) | 22/1/1 | 1.56 / 3.54 / 11 |
| ps @ 7.5 m | 4325 | 4227 | 29188 | 6.75 | 0.977 (217/5) | 0.905 (218/241) | 11 | 0.950 | 194/35/12 | 0.10 (21) | 0.25 (57) | 20/3/1 | 1.66 / 4.17 / 15 |
| ps @ 10 m | 3711 | 3648 | 29188 | 7.87 | 0.977 (217/5) | 0.892 (215/241) | 14 | 0.950 | 194/35/12 | 0.05 (10) | 0.14 (31) | 20/3/1 | 1.78 / 4.92 / 19 |
| ps @ 12.5 m | 3506 | 3457 | 29188 | 8.33 | 0.977 (217/5) | 0.880 (212/241) | 16 | 0.946 | 194/34/13 | 0.04 (8) | 0.11 (24) | 18/5/1 | 1.77 / 5.20 / 21 |
| ps @ 15 m | 3398 | 3359 | 29188 | 8.59 | 0.977 (217/5) | 0.871 (210/241) | 18 | 0.946 | 194/34/13 | 0.04 (8) | 0.11 (23) | 18/5/1 | 1.78 / 5.20 / 23 |
| ps_citywide @ 7.5 m | 4325 | 4227 | 29188 | 6.75 | 0.977 (217/5) | 0.905 (218/241) | 11 | 0.950 | 194/35/12 | 0.10 (21) | 0.25 (57) | 20/3/1 | 1.66 / 4.17 / 15 |
| ps_placeable @ 2.5 m | 9533 | 9533 | 25927 | 2.72 | 0.980 (197/4) | 0.959 (231/241) | 1 | 0.963 | 194/38/9 | 0.56 (181) | 0.82 (399) | 22/1/1 | 1.00 / 2.52 / 2 |
| ps_placeable @ 5 m | 5574 | 5574 | 25927 | 4.65 | 0.980 (197/4) | 0.929 (224/241) | 6 | 0.954 | 194/36/11 | 0.22 (55) | 0.52 (150) | 21/2/1 | 1.60 / 3.52 / 10 |
| ps_placeable @ 7.5 m | 4088 | 4088 | 25927 | 6.34 | 0.980 (197/4) | 0.905 (218/241) | 11 | 0.950 | 194/35/12 | 0.09 (20) | 0.24 (55) | 20/3/1 | 1.74 / 4.39 / 16 |
| ps_placeable @ 10 m | 3602 | 3602 | 25927 | 7.20 | 0.980 (197/4) | 0.888 (214/241) | 15 | 0.950 | 194/35/12 | 0.05 (10) | 0.13 (30) | 20/3/1 | 1.81 / 5.11 / 20 |
| ps_placeable @ 12.5 m | 3415 | 3415 | 25927 | 7.59 | 0.980 (197/4) | 0.876 (211/241) | 17 | 0.946 | 194/34/13 | 0.04 (8) | 0.10 (22) | 18/5/1 | 1.80 / 5.34 / 22 |
| ps_placeable @ 15 m | 3327 | 3327 | 25927 | 7.79 | 0.980 (197/4) | 0.867 (209/241) | 19 | 0.946 | 194/34/13 | 0.03 (6) | 0.08 (17) | 18/5/1 | 1.83 / 5.34 / 24 |
| ps_raycast @ 2.5 m | 11492 | 11492 | 25927 | 2.26 | 0.980 (197/4) | 0.971 (234/241) | 0 | 0.971 | 194/40/7 | 0.62 (213) | 0.88 (508) | 23/0/1 | 0.56 / 1.14 / 0 |
| ps_raycast @ 5 m | 7170 | 7170 | 25927 | 3.62 | 0.980 (197/4) | 0.959 (231/241) | 0 | 0.959 | 194/37/10 | 0.17 (40) | 0.58 (161) | 20/3/1 | 1.14 / 2.13 / 0 |
| ps_raycast @ 7.5 m | 5393 | 5393 | 25927 | 4.81 | 0.980 (197/4) | 0.950 (229/241) | 0 | 0.950 | 194/35/12 | 0.05 (12) | 0.21 (53) | 20/3/1 | 1.49 / 3.00 / 0 |
| ps_raycast @ 10 m | 4304 | 4304 | 25927 | 6.02 | 0.980 (197/4) | 0.921 (222/241) | 4 | 0.938 | 194/32/15 | 0.03 (7) | 0.11 (26) | 19/4/1 | 1.66 / 3.19 / 5 |
| ps_raycast @ 12.5 m | 3650 | 3650 | 25927 | 7.10 | 0.980 (197/4) | 0.900 (217/241) | 9 | 0.938 | 194/32/15 | 0.03 (7) | 0.07 (16) | 19/4/1 | 1.77 / 3.99 / 10 |
| ps_raycast @ 15 m | 3367 | 3367 | 25927 | 7.70 | 0.980 (197/4) | 0.888 (214/241) | 11 | 0.934 | 194/31/16 | 0.03 (6) | 0.07 (14) | 19/4/1 | 1.82 / 4.17 / 11 |
| fusion | 4018 | 4018 | 25927 | 6.45 | 0.980 (197/4) | 0.892 (215/241) | 10 | 0.934 | 194/31/16 | 0.03 (7) | 0.09 (19) | 15/8/1 | 1.71 / 4.09 / 12 |
| fusion_refit | 4018 | 4018 | 25927 | 6.45 | 0.980 (197/4) | 0.876 (211/241) | 14 | 0.934 | 194/31/16 | 0.04 (9) | 0.09 (21) | 15/8/1 | 1.47 / 4.76 / 17 |
| fusion_server | 7291 | 4030 | 29188 | 4.00 | 0.977 (217/5) | 0.892 (215/241) | 10 | 0.934 | 194/31/16 | 0.03 (7) | 0.08 (18) | 15/8/1 | 1.78 / 4.09 / 12 |
| fusion_server+attach | 4302 | 4030 | 29188 | 6.78 | 0.977 (217/5) | 0.892 (215/241) | 10 | 0.934 | 194/31/16 | 0.03 (7) | 0.08 (18) | 15/8/1 | 1.78 / 4.09 / 12 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 194 of this run's 241 pool ramps are self-detected, so 80% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 2727 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 5.9 m | 10.8 m | 0.33 | 0.13 | 0.01 |
| raycast | 7.6 m | 11.0 m | 0.51 | 0.20 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.676 | 49/163 | 13/24 | 0.639 | 22/154 | 8/24 |
| 5 | 0.905 | 54/218 | 20/24 | 0.892 | 19/215 | 15/24 |
| 7.5 | 0.959 | 53/231 | 21/24 | 0.963 | 17/232 | 20/24 |
| 10 | 0.975 | 53/235 | 21/24 | 0.975 | 17/235 | 20/24 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 3261 | 0.78 | 23 | 0.952 [0.77, 0.99] | 20 | 1 | 2 |
| ps @ 7.5 m | cluster of 1 | 496 | 0.66 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| ps @ 7.5 m | cluster of 2 | 809 | 0.72 | 6 | 0.667 [0.30, 0.90] | 4 | 2 | 0 |
| ps @ 7.5 m | cluster of 3+ | 24622 | 0.87 | 195 | 0.989 [0.96, 1.00] | 187 | 2 | 6 |
| fusion_server | unplaceable | 3261 | 0.78 | 23 | 0.952 [0.77, 0.99] | 20 | 1 | 2 |
| fusion_server | cluster of 1 | 889 | 0.72 | 3 | 1.000 [0.34, 1.00] | 2 | 0 | 1 |
| fusion_server | cluster of 2 | 822 | 0.74 | 8 | 0.625 [0.31, 0.86] | 5 | 3 | 0 |
| fusion_server | cluster of 3+ | 24216 | 0.87 | 193 | 0.995 [0.97, 1.00] | 187 | 1 | 5 |
| fusion_server+attach | unplaceable | 3261 | 0.78 | 23 | 0.952 [0.77, 0.99] | 20 | 1 | 2 |
| fusion_server+attach | cluster of 1 | 782 | 0.71 | 3 | 1.000 [0.34, 1.00] | 2 | 0 | 1 |
| fusion_server+attach | cluster of 2 | 752 | 0.75 | 6 | 0.500 [0.19, 0.81] | 3 | 3 | 0 |
| fusion_server+attach | cluster of 3+ | 24393 | 0.87 | 195 | 0.995 [0.97, 1.00] | 189 | 1 | 5 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 3261; attached 2989 (0.92)
- clusters: 7291 (`fusion_server`) -> 4302 (`fusion_server+attach`)
- sanity (not a metric): 23 unplaceable labels are on judged panos; 21 of them attached (19 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
