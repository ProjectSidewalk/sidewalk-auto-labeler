# vancouver: PS label clustering vs RampNet GT (offline)

mode: offline -- 73478 labels synthesized from `results.jsonl`, one per stored detection >= 0.3 (22 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.3; results `results.jsonl` sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
inputs: streets sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`; verdicts sha256 `none`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: none (--no-gt), so coverage, recall, frag, dual, coherence and GT precision read n/a below; GT: 0 judged panos -> 0 placeable points -> 0 ramps (0 cross-pano merges), 0 in the recall pool; raycast placed 85466 of 92067 detections (drops {'below_floor': 0, 'on_rig': 48, 'horizon': 13, 'out_of_range': 6540})

## Data provenance

- results file `runs/vancouver/results.jsonl`: sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 73478 of 73478
- `streets.geojson`: 12567 features, sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`, 2026-09-30T01:39:49+00:00 (4.8 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/streets?filetype=geojson; 11783 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 5892 blocks, largest 165 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 69812 fusion members vs 69812 placeable server labels; 0 of 69812 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/vancouver/fusion_eval/report.md (published in the 2.6 m frame): precision n/a, recall (union) n/a, dual 0/0/0
- fusion_server input: 73478 AI (by account; 0 share a detection with another AI label) + 0 human labels on 28502 panos (0 positioned by inverting their labels, 28502 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 3666 labels the raycast cannot place (range cap, horizon) are singleton clusters; 17551 of its 17551 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 73478 label ids, 73478 distinct, of 73478 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 25368 panos in both (all labels, mostly AI): median 0.010 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 36130 | 33936 | 73478 | 2.03 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 5 m | 24605 | 23459 | 73478 | 2.99 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 7.5 m | 20205 | 19654 | 73478 | 3.64 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 10 m | 18317 | 18046 | 73478 | 4.01 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 12.5 m | 17456 | 17310 | 73478 | 4.21 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 15 m | 16996 | 16897 | 73478 | 4.32 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_citywide @ 7.5 m | 20061 | 19520 | 73478 | 3.66 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 2.5 m | 33769 | 33769 | 69812 | 2.07 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 5 m | 23112 | 23112 | 69812 | 3.02 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 7.5 m | 19294 | 19294 | 69812 | 3.62 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 10 m | 17681 | 17681 | 69812 | 3.95 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 12.5 m | 16990 | 16990 | 69812 | 4.11 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 15 m | 16618 | 16618 | 69812 | 4.20 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 2.5 m | 30505 | 30505 | 69812 | 2.29 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 5 m | 20519 | 20519 | 69812 | 3.40 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 7.5 m | 17466 | 17466 | 69812 | 4.00 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 10 m | 16582 | 16582 | 69812 | 4.21 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 12.5 m | 16277 | 16277 | 69812 | 4.29 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 15 m | 16061 | 16061 | 69812 | 4.35 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion | 17551 | 17551 | 69812 | 3.98 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_refit | 17551 | 17551 | 69812 | 3.98 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_server | 21217 | 17551 | 73478 | 3.46 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_server+attach | 17620 | 17551 | 73478 | 4.17 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 0 of this run's 0 pool ramps are self-detected, so most of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 10825 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 5.0 m | 9.8 m | 0.24 | 0.09 | 0.01 |
| raycast | 3.7 m | 6.6 m | 0.05 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (ps @ 7.5 m vs fusion)

| radius m | ps @ 7.5 m (offline stand-in for deployed) coverage | ps @ 7.5 m frag 5 m | ps @ 7.5 m dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |
| 5 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |
| 7.5 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |
| 10 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ps @ 2.5 m | 36130 | 73478 | 491.7 | server | 36130 | 0.793 | 0.883 | 0.942 | 33936 | 0.787 | 0.872 | 0.936 |
| ps @ 5 m | 24605 | 73478 | 334.9 | server | 24605 | 0.464 | 0.731 | 0.885 | 23459 | 0.462 | 0.713 | 0.873 |
| ps @ 7.5 m | 20205 | 73478 | 275.0 | server | 20205 | 0.296 | 0.578 | 0.829 | 19654 | 0.300 | 0.570 | 0.817 |
| ps @ 10 m | 18317 | 73478 | 249.3 | server | 18317 | 0.277 | 0.503 | 0.785 | 18046 | 0.279 | 0.502 | 0.779 |
| ps @ 12.5 m | 17456 | 73478 | 237.6 | server | 17456 | 0.275 | 0.485 | 0.761 | 17310 | 0.276 | 0.485 | 0.757 |
| ps @ 15 m | 16996 | 73478 | 231.3 | server | 16996 | 0.276 | 0.481 | 0.751 | 16897 | 0.277 | 0.481 | 0.749 |
| ps_citywide @ 7.5 m | 20061 | 73478 | 273.0 | server | 20061 | 0.285 | 0.571 | 0.826 | 19520 | 0.289 | 0.563 | 0.814 |
| ps_placeable @ 2.5 m | 33769 | 69812 | 483.7 | server | 33769 | 0.785 | 0.871 | 0.935 | 33769 | 0.785 | 0.871 | 0.935 |
| ps_placeable @ 5 m | 23112 | 69812 | 331.1 | server | 23112 | 0.451 | 0.703 | 0.868 | 23112 | 0.451 | 0.703 | 0.868 |
| ps_placeable @ 7.5 m | 19294 | 69812 | 276.4 | server | 19294 | 0.296 | 0.552 | 0.805 | 19294 | 0.296 | 0.552 | 0.805 |
| ps_placeable @ 10 m | 17681 | 69812 | 253.3 | server | 17681 | 0.277 | 0.482 | 0.760 | 17681 | 0.277 | 0.482 | 0.760 |
| ps_placeable @ 12.5 m | 16990 | 69812 | 243.4 | server | 16990 | 0.277 | 0.468 | 0.737 | 16990 | 0.277 | 0.468 | 0.737 |
| ps_placeable @ 15 m | 16618 | 69812 | 238.0 | server | 16618 | 0.278 | 0.465 | 0.728 | 16618 | 0.278 | 0.465 | 0.728 |
| ps_raycast @ 2.5 m | 30505 | 69812 | 437.0 | server | 30505 | 0.761 | 0.843 | 0.924 | 30505 | 0.761 | 0.843 | 0.924 |
| ps_raycast @ 5 m | 20519 | 69812 | 293.9 | server | 20519 | 0.486 | 0.631 | 0.827 | 20519 | 0.486 | 0.631 | 0.827 |
| ps_raycast @ 7.5 m | 17466 | 69812 | 250.2 | server | 17466 | 0.332 | 0.495 | 0.757 | 17466 | 0.332 | 0.495 | 0.757 |
| ps_raycast @ 10 m | 16582 | 69812 | 237.5 | server | 16582 | 0.288 | 0.451 | 0.727 | 16582 | 0.288 | 0.451 | 0.727 |
| ps_raycast @ 12.5 m | 16277 | 69812 | 233.2 | server | 16277 | 0.282 | 0.442 | 0.716 | 16277 | 0.282 | 0.442 | 0.716 |
| ps_raycast @ 15 m | 16061 | 69812 | 230.1 | server | 16061 | 0.280 | 0.440 | 0.712 | 16061 | 0.280 | 0.440 | 0.712 |
| fusion | 17551 | 69812 | 251.4 | raycast | 17551 | 0.283 | 0.485 | 0.688 | 17551 | 0.283 | 0.485 | 0.688 |
| fusion_refit | 17551 | 69812 | 251.4 | raycast | 17551 | 0.284 | 0.485 | 0.690 | 17551 | 0.284 | 0.485 | 0.690 |
| fusion_server | 21217 | 73478 | 288.8 | server | 21217 | 0.451 | 0.631 | 0.834 | 17551 | 0.319 | 0.501 | 0.758 |
| fusion_server+attach | 17620 | 73478 | 239.8 | server | 17620 | 0.316 | 0.502 | 0.769 | 17551 | 0.316 | 0.502 | 0.769 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 3666 | 0.60 | n/a | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 4055 | 0.56 | n/a | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 2 | 6474 | 0.70 | n/a | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 3+ | 59283 | 0.85 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server | unplaceable | 3666 | 0.60 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 1 | 4474 | 0.55 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 2 | 4504 | 0.69 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 3+ | 60834 | 0.85 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | unplaceable | 3666 | 0.60 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 4260 | 0.55 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 2 | 4431 | 0.68 | n/a | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 3+ | 61121 | 0.85 | n/a | 0 | n/a | 0 | 0 | 0 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 3666; attached 3597 (0.98)
- clusters: 21217 (`fusion_server`) -> 17620 (`fusion_server+attach`)
- sanity (not a metric): 0 unplaceable labels are on judged panos; 0 of them attached (0 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
