# laurens: PS label clustering vs RampNet GT (offline)

mode: offline -- 708 labels synthesized from `results.raw.jsonl`, one per stored detection >= 0.55 (0 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.raw.jsonl` sha256 `a7da290de5b0576cf5f54ab5bc1615a52fba766c47da3acb97e57c272860340a`; benchmark split `laurens_mapillary`
inputs: streets sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`; verdicts sha256 `0fb67c5d6da90827049c30f17ce5150ff4b24c5ae605a3f047b0b1d6255ad49e`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 94 judged panos -> 240 placeable points -> 238 ramps (2 cross-pano merges), 238 in the recall pool; raycast placed 3370 of 5727 detections (drops {'below_floor': 0, 'on_rig': 2199, 'horizon': 7, 'out_of_range': 151})

## Data provenance

- results file `runs/laurens/results.raw.jsonl`: sha256 `a7da290de5b0576cf5f54ab5bc1615a52fba766c47da3acb97e57c272860340a`
- 0 ambiguous pixel keys in `results.raw.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 708 of 708
- `ps_streets.geojson`: 169 features, sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`, 2026-09-28T17:43:29+00:00 (6.1 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson; 168 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 60 blocks, largest 44 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 696 fusion members vs 696 placeable server labels; 0 of 696 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/laurens/fusion_eval/report.md (published in the 2.6 m frame): precision 0.896, recall (union) 0.597, dual 14/25/13
- fusion_server input: 708 AI (by account; 0 share a detection with another AI label) + 0 human labels on 420 panos (0 positioned by inverting their labels, 420 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 12 labels the raycast cannot place (range cap, horizon) are singleton clusters; 300 of its 309 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 708 label ids, 708 distinct, of 708 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 375 panos in both (all labels, mostly AI): median 0.010 m, p90 0.19 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 585 | 580 | 708 | 1.21 | 0.898 (97/11) | 0.693 (165/238) | 1 | 0.697 | 97/69/72 | 0.14 (24) | 0.45 (109) | 30/14/8 | 0.00 / 1.59 / 0 |
| ps @ 5 m | 438 | 437 | 708 | 1.62 | 0.898 (97/11) | 0.634 (151/238) | 3 | 0.647 | 97/57/84 | 0.03 (4) | 0.23 (37) | 20/23/9 | 0.76 / 2.73 / 0 |
| ps @ 7.5 m | 364 | 364 | 708 | 1.95 | 0.897 (96/11) | 0.588 (140/238) | 5 | 0.609 | 97/48/93 | 0.00 (0) | 0.11 (16) | 16/27/9 | 1.69 / 3.32 / 1 |
| ps @ 10 m | 305 | 305 | 708 | 2.32 | 0.897 (96/11) | 0.529 (126/238) | 13 | 0.584 | 97/42/99 | 0.00 (0) | 0.06 (7) | 12/29/11 | 2.17 / 4.53 / 6 |
| ps @ 12.5 m | 264 | 264 | 708 | 2.68 | 0.896 (95/11) | 0.445 (106/238) | 26 | 0.555 | 97/35/106 | 0.00 (0) | 0.06 (6) | 7/33/12 | 2.89 / 6.06 / 20 |
| ps @ 15 m | 233 | 233 | 708 | 3.04 | 0.895 (94/11) | 0.403 (96/238) | 35 | 0.550 | 97/34/107 | 0.00 (0) | 0.04 (4) | 4/32/16 | 3.44 / 6.73 / 29 |
| ps_citywide @ 7.5 m | 364 | 364 | 708 | 1.95 | 0.897 (96/11) | 0.588 (140/238) | 5 | 0.609 | 97/48/93 | 0.00 (0) | 0.11 (16) | 16/27/9 | 1.69 / 3.32 / 1 |
| ps_placeable @ 2.5 m | 578 | 578 | 696 | 1.20 | 0.898 (97/11) | 0.689 (164/238) | 1 | 0.693 | 97/68/73 | 0.14 (24) | 0.45 (108) | 29/15/8 | 0.00 / 1.59 / 0 |
| ps_placeable @ 5 m | 433 | 433 | 696 | 1.61 | 0.898 (97/11) | 0.634 (151/238) | 3 | 0.647 | 97/57/84 | 0.02 (3) | 0.22 (36) | 20/23/9 | 1.04 / 2.73 / 0 |
| ps_placeable @ 7.5 m | 362 | 362 | 696 | 1.92 | 0.897 (96/11) | 0.588 (140/238) | 5 | 0.609 | 97/48/93 | 0.00 (0) | 0.11 (16) | 16/27/9 | 1.75 / 3.35 / 1 |
| ps_placeable @ 10 m | 304 | 304 | 696 | 2.29 | 0.897 (96/11) | 0.529 (126/238) | 13 | 0.584 | 97/42/99 | 0.00 (0) | 0.06 (8) | 12/29/11 | 2.17 / 4.53 / 6 |
| ps_placeable @ 12.5 m | 264 | 264 | 696 | 2.64 | 0.896 (95/11) | 0.445 (106/238) | 26 | 0.555 | 97/35/106 | 0.00 (0) | 0.07 (7) | 7/33/12 | 2.96 / 6.06 / 20 |
| ps_placeable @ 15 m | 235 | 235 | 696 | 2.96 | 0.895 (94/11) | 0.408 (97/238) | 34 | 0.550 | 97/34/107 | 0.00 (0) | 0.05 (5) | 4/32/16 | 3.44 / 6.73 / 28 |
| ps_raycast @ 2.5 m | 572 | 572 | 696 | 1.22 | 0.898 (97/11) | 0.681 (162/238) | 0 | 0.681 | 97/65/76 | 0.10 (17) | 0.44 (106) | 27/17/8 | 0.00 / 1.05 / 0 |
| ps_raycast @ 5 m | 429 | 429 | 696 | 1.62 | 0.898 (97/11) | 0.626 (149/238) | 2 | 0.634 | 97/54/87 | 0.02 (3) | 0.21 (33) | 20/24/8 | 0.68 / 2.05 / 0 |
| ps_raycast @ 7.5 m | 352 | 352 | 696 | 1.98 | 0.898 (97/11) | 0.584 (139/238) | 5 | 0.605 | 97/47/94 | 0.01 (2) | 0.06 (8) | 15/28/9 | 1.30 / 3.08 / 0 |
| ps_raycast @ 10 m | 302 | 302 | 696 | 2.30 | 0.898 (97/11) | 0.529 (126/238) | 13 | 0.584 | 97/42/99 | 0.01 (1) | 0.04 (5) | 11/29/12 | 1.85 / 4.71 / 7 |
| ps_raycast @ 12.5 m | 267 | 267 | 696 | 2.61 | 0.898 (97/11) | 0.483 (115/238) | 22 | 0.576 | 97/40/101 | 0.00 (0) | 0.02 (2) | 9/28/15 | 2.44 / 5.63 / 16 |
| ps_raycast @ 15 m | 236 | 236 | 696 | 2.95 | 0.897 (96/11) | 0.441 (105/238) | 28 | 0.559 | 97/36/105 | 0.01 (1) | 0.01 (1) | 6/30/16 | 3.08 / 6.13 / 26 |
| fusion | 307 | 307 | 696 | 2.27 | 0.896 (95/11) | 0.534 (127/238) | 14 | 0.592 | 97/44/97 | 0.02 (2) | 0.03 (4) | 14/25/13 | 1.95 / 4.71 / 9 |
| fusion_refit | 307 | 307 | 696 | 2.27 | 0.896 (95/11) | 0.538 (128/238) | 14 | 0.597 | 97/45/96 | 0.02 (2) | 0.03 (4) | 14/25/13 | 1.91 / 4.56 / 8 |
| fusion_server | 321 | 309 | 708 | 2.21 | 0.896 (95/11) | 0.534 (127/238) | 16 | 0.601 | 97/46/95 | 0.02 (2) | 0.03 (4) | 14/25/13 | 2.00 / 4.71 / 9 |
| fusion_server+attach | 311 | 309 | 708 | 2.28 | 0.896 (95/11) | 0.534 (127/238) | 16 | 0.601 | 97/46/95 | 0.02 (2) | 0.03 (4) | 14/25/13 | 2.00 / 4.71 / 9 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 97 of this run's 238 pool ramps are self-detected, so 41% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 98 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 8.1 m | 11.6 m | 0.62 | 0.21 | 0.01 |
| raycast | 7.8 m | 9.9 m | 0.53 | 0.08 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.382 | 14/91 | 5/52 | 0.315 | 2/75 | 5/52 |
| 5 | 0.588 | 16/140 | 16/52 | 0.534 | 4/127 | 14/52 |
| 7.5 | 0.693 | 12/165 | 26/52 | 0.639 | 2/152 | 24/52 |
| 10 | 0.756 | 12/180 | 33/52 | 0.706 | 1/168 | 30/52 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 12 | 0.62 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 169 | 0.70 | 0.570 | 38 | 0.730 [0.57, 0.85] | 27 | 10 | 1 |
| ps @ 7.5 m | cluster of 2 | 225 | 0.73 | 0.584 | 36 | 0.971 [0.85, 0.99] | 33 | 1 | 2 |
| ps @ 7.5 m | cluster of 3+ | 302 | 0.74 | 0.584 | 37 | 1.000 [0.91, 1.00] | 37 | 0 | 0 |
| fusion_server | unplaceable | 12 | 0.62 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 1 | 143 | 0.66 | 0.570 | 33 | 0.677 [0.50, 0.81] | 21 | 10 | 2 |
| fusion_server | cluster of 2 | 136 | 0.71 | 0.584 | 22 | 1.000 [0.85, 1.00] | 21 | 0 | 1 |
| fusion_server | cluster of 3+ | 417 | 0.76 | 0.570 | 56 | 0.982 [0.91, 1.00] | 55 | 1 | 0 |
| fusion_server+attach | unplaceable | 12 | 0.62 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 142 | 0.66 | 0.570 | 33 | 0.677 [0.50, 0.81] | 21 | 10 | 2 |
| fusion_server+attach | cluster of 2 | 131 | 0.71 | 0.584 | 20 | 1.000 [0.83, 1.00] | 19 | 0 | 1 |
| fusion_server+attach | cluster of 3+ | 423 | 0.75 | 0.570 | 58 | 0.983 [0.91, 1.00] | 57 | 1 | 0 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 12; attached 10 (0.83)
- clusters: 321 (`fusion_server`) -> 311 (`fusion_server+attach`)
- sanity (not a metric): 0 unplaceable labels are on judged panos; 0 of them attached (0 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
