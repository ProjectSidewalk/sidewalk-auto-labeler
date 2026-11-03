# vancouver: PS label clustering vs RampNet GT (offline)

mode: offline -- 60094 labels synthesized from `results.jsonl`, one per stored detection >= 0.55 (10 on the camera rig left out), each placed where the server would place it (ps_placement: the server's estimator at 2.341 m); no server labels or clusters, so `deployed` and `ps_repro` do not exist here
scorer 106.4; results `results.jsonl` sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
inputs: streets sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`; verdicts sha256 `none`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: none (--no-gt), so coverage, recall, frag, dual, coherence and GT precision read n/a below; GT: 0 judged panos -> 0 placeable points -> 0 ramps (0 cross-pano merges), 0 in the recall pool; raycast placed 85466 of 92067 detections (drops {'below_floor': 0, 'on_rig': 48, 'horizon': 13, 'out_of_range': 6540})

## Data provenance

- results file `runs/vancouver/results.jsonl`: sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
- 0 ambiguous pixel keys in `results.jsonl` (two stored detections round to one pixel); offline labels map to their detection directly, and the pixel-key map agrees on 60094 of 60094
- `streets.geojson`: 12567 features, sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`, 2026-09-30T01:39:49+00:00 (6.1 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/streets?filetype=geojson; 11783 open streets kept (the server snaps to open streets only)
- regions: every synthesized label takes the region of the street nearest its server position, as the server assigns it at insert; 0 labels were equidistant from streets in two regions (lowest street_edge_id taken)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 5677 blocks, largest 124 labels

## Validation checks

- fusion arm vs ps_* arms cover the same labels: 57972 fusion members vs 57972 placeable server labels; 0 of 57972 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/vancouver/fusion_eval/report.md (published in the 2.6 m frame): precision n/a, recall (union) n/a, dual 0/0/0
- fusion_server input: 60094 AI (by account; 0 share a detection with another AI label) + 0 human labels on 26928 panos (0 positioned by inverting their labels, 26928 from the run's pano block (offline: the block the labels were placed from), 0 with neither, whose 0 labels are singleton clusters); 2122 labels the raycast cannot place (range cap, horizon) are singleton clusters; 14820 of its 14820 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 60094 label ids, 60094 distinct, of 60094 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 0 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (offline: the labels were placed from the run's block)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 23829 panos: median 0.010 m, p90 0.20 m; from all labels (the #105 method), over 23829 panos: median 0.010 m, p90 0.20 m; from human labels only, over 0 panos: median n/a m, p90 n/a m, max n/a m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| ps @ 2.5 m | 28690 | 27432 | 60094 | 2.09 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 5 m | 19482 | 18819 | 60094 | 3.08 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 7.5 m | 16200 | 15906 | 60094 | 3.71 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 10 m | 14930 | 14807 | 60094 | 4.03 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 12.5 m | 14415 | 14351 | 60094 | 4.17 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 15 m | 14135 | 14091 | 60094 | 4.25 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_citywide @ 7.5 m | 16065 | 15773 | 60094 | 3.74 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 2.5 m | 27322 | 27322 | 57972 | 2.12 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 5 m | 18627 | 18627 | 57972 | 3.11 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 7.5 m | 15727 | 15727 | 57972 | 3.69 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 10 m | 14630 | 14630 | 57972 | 3.96 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 12.5 m | 14198 | 14198 | 57972 | 4.08 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 15 m | 13966 | 13966 | 57972 | 4.15 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 2.5 m | 25069 | 25069 | 57972 | 2.31 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 5 m | 17033 | 17033 | 57972 | 3.40 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 7.5 m | 14698 | 14698 | 57972 | 3.94 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 10 m | 14076 | 14076 | 57972 | 4.12 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 12.5 m | 13858 | 13858 | 57972 | 4.18 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 15 m | 13695 | 13695 | 57972 | 4.23 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion | 14820 | 14820 | 57972 | 3.91 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_refit | 14820 | 14820 | 57972 | 3.91 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_server | 16942 | 14820 | 60094 | 3.55 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_server+attach | 14864 | 14820 | 60094 | 4.04 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 0 of this run's 0 pool ramps are self-detected, so most of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 9499 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 4.3 m | 8.6 m | 0.16 | 0.05 | 0.00 |
| raycast | 3.5 m | 6.1 m | 0.03 | 0.00 | 0.00 |

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
| ps @ 2.5 m | 28690 | 60094 | 477.4 | server | 28690 | 0.773 | 0.857 | 0.924 | 27432 | 0.767 | 0.846 | 0.918 |
| ps @ 5 m | 19482 | 60094 | 324.2 | server | 19482 | 0.412 | 0.662 | 0.841 | 18819 | 0.408 | 0.643 | 0.826 |
| ps @ 7.5 m | 16200 | 60094 | 269.6 | server | 16200 | 0.242 | 0.492 | 0.766 | 15906 | 0.243 | 0.485 | 0.755 |
| ps @ 10 m | 14930 | 60094 | 248.4 | server | 14930 | 0.227 | 0.421 | 0.718 | 14807 | 0.227 | 0.420 | 0.712 |
| ps @ 12.5 m | 14415 | 60094 | 239.9 | server | 14415 | 0.227 | 0.406 | 0.690 | 14351 | 0.227 | 0.405 | 0.687 |
| ps @ 15 m | 14135 | 60094 | 235.2 | server | 14135 | 0.226 | 0.403 | 0.681 | 14091 | 0.226 | 0.403 | 0.680 |
| ps_citywide @ 7.5 m | 16065 | 60094 | 267.3 | server | 16065 | 0.228 | 0.482 | 0.761 | 15773 | 0.229 | 0.474 | 0.750 |
| ps_placeable @ 2.5 m | 27322 | 57972 | 471.3 | server | 27322 | 0.765 | 0.844 | 0.918 | 27322 | 0.765 | 0.844 | 0.918 |
| ps_placeable @ 5 m | 18627 | 57972 | 321.3 | server | 18627 | 0.399 | 0.635 | 0.821 | 18627 | 0.399 | 0.635 | 0.821 |
| ps_placeable @ 7.5 m | 15727 | 57972 | 271.3 | server | 15727 | 0.239 | 0.470 | 0.745 | 15727 | 0.239 | 0.470 | 0.745 |
| ps_placeable @ 10 m | 14630 | 57972 | 252.4 | server | 14630 | 0.226 | 0.405 | 0.699 | 14630 | 0.226 | 0.405 | 0.699 |
| ps_placeable @ 12.5 m | 14198 | 57972 | 244.9 | server | 14198 | 0.226 | 0.394 | 0.673 | 14198 | 0.226 | 0.394 | 0.673 |
| ps_placeable @ 15 m | 13966 | 57972 | 240.9 | server | 13966 | 0.226 | 0.393 | 0.666 | 13966 | 0.226 | 0.393 | 0.666 |
| ps_raycast @ 2.5 m | 25069 | 57972 | 432.4 | server | 25069 | 0.742 | 0.817 | 0.906 | 25069 | 0.742 | 0.817 | 0.906 |
| ps_raycast @ 5 m | 17033 | 57972 | 293.8 | server | 17033 | 0.440 | 0.570 | 0.785 | 17033 | 0.440 | 0.570 | 0.785 |
| ps_raycast @ 7.5 m | 14698 | 57972 | 253.5 | server | 14698 | 0.278 | 0.426 | 0.705 | 14698 | 0.278 | 0.426 | 0.705 |
| ps_raycast @ 10 m | 14076 | 57972 | 242.8 | server | 14076 | 0.236 | 0.387 | 0.672 | 14076 | 0.236 | 0.387 | 0.672 |
| ps_raycast @ 12.5 m | 13858 | 57972 | 239.0 | server | 13858 | 0.230 | 0.378 | 0.659 | 13858 | 0.230 | 0.378 | 0.659 |
| ps_raycast @ 15 m | 13695 | 57972 | 236.2 | server | 13695 | 0.228 | 0.376 | 0.655 | 13695 | 0.228 | 0.376 | 0.655 |
| fusion | 14820 | 57972 | 255.6 | raycast | 14820 | 0.229 | 0.424 | 0.628 | 14820 | 0.229 | 0.424 | 0.628 |
| fusion_refit | 14820 | 57972 | 255.6 | raycast | 14820 | 0.230 | 0.425 | 0.635 | 14820 | 0.230 | 0.425 | 0.635 |
| fusion_server | 16942 | 60094 | 281.9 | server | 16942 | 0.379 | 0.553 | 0.778 | 14820 | 0.266 | 0.437 | 0.709 |
| fusion_server+attach | 14864 | 60094 | 247.3 | server | 14864 | 0.265 | 0.438 | 0.717 | 14820 | 0.265 | 0.438 | 0.717 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `ps @ 7.5 m` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ps @ 7.5 m | unplaceable | 2122 | 0.74 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 1 | 3021 | 0.71 | 0.568 | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 2 | 5156 | 0.80 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| ps @ 7.5 m | cluster of 3+ | 49795 | 0.87 | 0.568 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | unplaceable | 2122 | 0.74 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 1 | 3222 | 0.72 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 2 | 4198 | 0.80 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 3+ | 50552 | 0.87 | 0.568 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | unplaceable | 2122 | 0.74 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 3125 | 0.72 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 2 | 4093 | 0.80 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 3+ | 50754 | 0.87 | 0.568 | 0 | n/a | 0 | 0 | 0 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 2122; attached 2078 (0.98)
- clusters: 16942 (`fusion_server`) -> 14864 (`fusion_server+attach`)
- sanity (not a metric): 0 unplaceable labels are on judged panos; 0 of them attached (0 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
