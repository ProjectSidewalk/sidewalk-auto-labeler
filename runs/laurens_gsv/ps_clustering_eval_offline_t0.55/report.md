# laurens_gsv: PS label clustering vs RampNet GT (offline)

mode: offline -- 473 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (0 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.2; results `results.jsonl` sha256 `83f49aaecac1b37cf05f6687ac350694681832229f33107b42142437f43338e7`
inputs: streets sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`; verdicts sha256 `0f4608abcd6d380d388d668c4b2e48ebece2b058f8f934ab3b889d12ae9974e3`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 86 judged panos -> 195 placeable points -> 190 ramps (5 cross-pano merges), 190 in the recall pool; raycast placed 1541 of 1656 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 0, 'out_of_range': 115})

## Data provenance

- results file `D:\Git\sidewalk-auto-labeler\runs\laurens_gsv\results.jsonl`: sha256 `83f49aaecac1b37cf05f6687ac350694681832229f33107b42142437f43338e7`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 473 of 473
- `ps_streets.geojson`: 169 features, sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`, 2026-09-28T17:22:15+00:00 (0.2 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson; 168 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 59 blocks, largest 26 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 459 fusion members vs 459 placeable server labels; 0 of 459 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/laurens_gsv/fusion_eval/report.md (published in the 2.6 m frame): precision 0.931, recall (union) 0.811, dual 15/11/1
- fusion_server input: 473 AI + 0 human labels on 230 panos (230 positioned from the run's pano block, 0 inverted from their labels, 0 unplaceable and left out); 14 labels the raycast cannot place (range cap, horizon) are singleton clusters; 186 of its 186 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 196 panos in both: median 0.01 m, p90 0.25 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 292 | 282 | 473 | 1.62 | 0.933 (111/8) | 0.853 (162/190) | 0 | 0.853 | 107/55/28 | 0.36 (68) | 0.51 (109) | 20/6/1 | 0.50 / 1.92 / 1 |
| ps @ 5 m | 206 | 202 | 473 | 2.30 | 0.933 (111/8) | 0.816 (155/190) | 1 | 0.821 | 107/49/34 | 0.08 (13) | 0.16 (25) | 15/11/1 | 0.72 / 2.18 / 1 |
| ps @ 7.5 m | 174 | 171 | 473 | 2.72 | 0.933 (111/8) | 0.763 (145/190) | 3 | 0.779 | 107/41/42 | 0.03 (4) | 0.05 (7) | 12/14/1 | 0.87 / 2.96 / 1 |
| ps @ 10 m | 167 | 165 | 473 | 2.83 | 0.932 (110/8) | 0.737 (140/190) | 5 | 0.763 | 107/38/45 | 0.02 (3) | 0.03 (4) | 11/15/1 | 0.92 / 3.04 / 2 |
| ps @ 12.5 m | 165 | 163 | 473 | 2.87 | 0.940 (110/7) | 0.732 (139/190) | 6 | 0.763 | 107/38/45 | 0.02 (3) | 0.03 (4) | 11/15/1 | 0.97 / 3.05 / 2 |
| ps @ 15 m | 160 | 158 | 473 | 2.96 | 0.940 (110/7) | 0.711 (135/190) | 8 | 0.753 | 107/36/47 | 0.02 (3) | 0.03 (4) | 11/15/1 | 1.00 / 3.35 / 3 |
| ps_citywide @ 7.5 m | 174 | 171 | 473 | 2.72 | 0.933 (111/8) | 0.763 (145/190) | 3 | 0.779 | 107/41/42 | 0.03 (4) | 0.05 (7) | 12/14/1 | 0.87 / 2.96 / 1 |
| ps_placeable @ 2.5 m | 282 | 282 | 459 | 1.63 | 0.931 (108/8) | 0.853 (162/190) | 0 | 0.853 | 107/55/28 | 0.36 (68) | 0.51 (109) | 20/6/1 | 0.50 / 1.92 / 1 |
| ps_placeable @ 5 m | 199 | 199 | 459 | 2.31 | 0.931 (108/8) | 0.811 (154/190) | 2 | 0.821 | 107/49/34 | 0.08 (13) | 0.16 (24) | 15/11/1 | 0.72 / 2.24 / 1 |
| ps_placeable @ 7.5 m | 171 | 171 | 459 | 2.68 | 0.931 (108/8) | 0.763 (145/190) | 3 | 0.779 | 107/41/42 | 0.03 (4) | 0.05 (7) | 12/14/1 | 0.87 / 2.96 / 1 |
| ps_placeable @ 10 m | 165 | 165 | 459 | 2.78 | 0.930 (107/8) | 0.737 (140/190) | 5 | 0.763 | 107/38/45 | 0.02 (3) | 0.03 (4) | 11/15/1 | 0.92 / 3.04 / 2 |
| ps_placeable @ 12.5 m | 163 | 163 | 459 | 2.82 | 0.939 (107/7) | 0.732 (139/190) | 6 | 0.763 | 107/38/45 | 0.02 (3) | 0.03 (4) | 11/15/1 | 0.97 / 3.05 / 2 |
| ps_placeable @ 15 m | 158 | 158 | 459 | 2.91 | 0.939 (107/7) | 0.711 (135/190) | 8 | 0.753 | 107/36/47 | 0.02 (3) | 0.03 (4) | 11/15/1 | 1.00 / 3.35 / 3 |
| ps_raycast @ 2.5 m | 259 | 259 | 459 | 1.77 | 0.931 (108/8) | 0.853 (162/190) | 0 | 0.853 | 107/55/28 | 0.16 (27) | 0.36 (69) | 18/8/1 | 0.32 / 1.00 / 0 |
| ps_raycast @ 5 m | 193 | 193 | 459 | 2.38 | 0.931 (108/8) | 0.805 (153/190) | 1 | 0.811 | 107/47/36 | 0.01 (2) | 0.07 (10) | 15/11/1 | 0.67 / 1.73 / 0 |
| ps_raycast @ 7.5 m | 168 | 168 | 459 | 2.73 | 0.939 (108/7) | 0.758 (144/190) | 5 | 0.784 | 107/42/41 | 0.01 (2) | 0.02 (3) | 10/16/1 | 0.83 / 2.53 / 0 |
| ps_raycast @ 10 m | 165 | 165 | 459 | 2.78 | 0.939 (108/7) | 0.742 (141/190) | 5 | 0.768 | 107/39/44 | 0.01 (2) | 0.02 (3) | 10/16/1 | 0.87 / 2.66 / 0 |
| ps_raycast @ 12.5 m | 164 | 164 | 459 | 2.80 | 0.939 (108/7) | 0.742 (141/190) | 5 | 0.768 | 107/39/44 | 0.01 (2) | 0.02 (3) | 10/16/1 | 0.88 / 2.66 / 0 |
| ps_raycast @ 15 m | 159 | 159 | 459 | 2.89 | 0.939 (107/7) | 0.721 (137/190) | 7 | 0.758 | 107/37/46 | 0.01 (2) | 0.02 (3) | 10/16/1 | 0.97 / 3.05 / 1 |
| fusion | 186 | 186 | 459 | 2.47 | 0.931 (108/8) | 0.811 (154/190) | 0 | 0.811 | 107/47/36 | 0.01 (2) | 0.03 (5) | 15/11/1 | 0.72 / 1.94 / 0 |
| fusion_refit | 186 | 186 | 459 | 2.47 | 0.931 (108/8) | 0.811 (154/190) | 0 | 0.811 | 107/47/36 | 0.01 (2) | 0.03 (5) | 15/11/1 | 0.61 / 2.24 / 0 |
| fusion_server | 200 | 186 | 473 | 2.37 | 0.933 (111/8) | 0.811 (154/190) | 0 | 0.811 | 107/47/36 | 0.01 (2) | 0.03 (5) | 15/11/1 | 0.72 / 1.94 / 0 |
| fusion_server+attach | 188 | 186 | 473 | 2.52 | 0.932 (110/8) | 0.811 (154/190) | 0 | 0.811 | 107/47/36 | 0.01 (2) | 0.03 (5) | 15/11/1 | 0.72 / 1.94 / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 107 of this run's 190 pool ramps are self-detected, so 56% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 85 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 3.9 m | 6.8 m | 0.06 | 0.00 | 0.00 |
| raycast | 3.1 m | 5.3 m | 0.00 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.632 | 9/120 | 7/27 | 0.726 | 5/138 | 13/27 |
| 5 | 0.763 | 7/145 | 12/27 | 0.811 | 5/154 | 15/27 |
| 7.5 | 0.779 | 6/148 | 13/27 | 0.821 | 5/156 | 16/27 |
| 10 | 0.779 | 6/148 | 13/27 | 0.821 | 5/156 | 16/27 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 14 | 0.66 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 40 | 0.69 | 21 | 0.700 [0.48, 0.85] | 14 | 6 | 1 |
| ps @ 7.5 m | cluster of 2 | 71 | 0.70 | 24 | 0.958 [0.80, 0.99] | 23 | 1 | 0 |
| ps @ 7.5 m | cluster of 3+ | 348 | 0.79 | 73 | 0.986 [0.93, 1.00] | 71 | 1 | 1 |
| fusion_server | unplaceable | 14 | 0.66 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| fusion_server | cluster of 1 | 65 | 0.71 | 26 | 0.760 [0.57, 0.89] | 19 | 6 | 1 |
| fusion_server | cluster of 2 | 72 | 0.69 | 22 | 0.955 [0.78, 0.99] | 21 | 1 | 0 |
| fusion_server | cluster of 3+ | 322 | 0.79 | 70 | 0.986 [0.92, 1.00] | 68 | 1 | 1 |
| fusion_server+attach | unplaceable | 14 | 0.66 | 3 | 1.000 [0.44, 1.00] | 3 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 63 | 0.70 | 25 | 0.750 [0.55, 0.88] | 18 | 6 | 1 |
| fusion_server+attach | cluster of 2 | 70 | 0.70 | 22 | 0.955 [0.78, 0.99] | 21 | 1 | 0 |
| fusion_server+attach | cluster of 3+ | 326 | 0.79 | 71 | 0.986 [0.92, 1.00] | 69 | 1 | 1 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 14; attached 12 (0.86)
- clusters: 200 (`fusion_server`) -> 188 (`fusion_server+attach`)
- sanity (not a metric): 3 unplaceable labels are on judged panos; 2 of them attached (2 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
