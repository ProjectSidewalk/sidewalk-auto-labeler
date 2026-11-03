# vancouver: PS label clustering vs RampNet GT

labels: 64847 CurbRamp on the server, 50252 map to stored detections (AI), 14595 do not (14562 of them the AI account's, kept as AI by --ai-user; 33 human); 18684 server clusters over 64918 labels
scorer 106.4; results `results.jsonl` sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
inputs: streets sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`; verdicts sha256 `none`
raycast camera height auto; fusion arm at --min-confidence 0.55
- camera height mode `auto`: auto -> gsv-per-rig
- 28830 of 28830 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2011: median 2.28 m, 22 of 197 measured -> 2.5 m
  - 2012: median 2.41 m, 4 of 310 measured -> 2.5 m
  - 2014: median 2.34 m, 40 of 722 measured -> 2.5 m
  - 2015: median 2.38 m, 8 of 277 measured -> 2.5 m
  - 2016: median 2.41 m, 18 of 319 measured -> 2.5 m
  - 2017: median 2.43 m, 15 of 258 measured -> 2.5 m
  - 2018: median 2.24 m, 46 of 953 measured -> 2.5 m
  - 2019: median 2.32 m, 968 of 1663 measured -> 2.5 m
  - 2021: median 2.37 m, 506 of 926 measured -> 2.5 m
  - 2022: median 2.35 m, 1624 of 3434 measured -> 2.5 m
  - 2023: median 2.37 m, 4562 of 8653 measured -> 2.5 m
  - 2024: median 2.35 m, 5482 of 10621 measured -> 2.5 m
  - 2025: median 2.36 m, 277 of 497 measured -> 2.5 m
GT: none (--no-gt), so coverage, recall, frag, dual, coherence and GT precision read n/a below; GT: 0 judged panos -> 0 placeable points -> 0 ramps (0 cross-pano merges), 0 in the recall pool; raycast placed 85514 of 92067 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 13, 'out_of_range': 6540})

## Data provenance

- `raw_labels.geojson`: 64847 features, sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d`, 2026-09-28T23:01:58+00:00 (7.2 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 18684 features, sha256 `d8a1e7065e2cf97ff7da0ab50eba7e754b2ac67c4286ecbc8d528f8ff6bec1c3`, 2026-09-30T01:39:55+00:00 (6.1 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- labels by account: 51b0b927-3c8a-45b2-93de-bd878d1e5cf4 (AI) 64814, c6030d8f-9163-498b-a102-d147a27b8c44 7, f34410b6-90d2-4176-a590-f371b75ab4c5 6, de9a2eb6-52e3-4854-b488-b970c5c1c567 5, aab0b9c1-bffc-4884-9c76-365762503a95 5, fb9b61d7-c613-454e-82de-63722f6baa2a 5, 71a31933-61d9-4474-b0d0-1d41f7aba0ac 2, 964eb6f2-da36-4a4f-bb95-aa99ff6eae6f 2, 0ff4a61a-8f80-4ef9-a5f5-1c85d8917e0d 1
- 0 labels dropped before clustering (null lng or lng > 360), matching label_clustering.clean_label_data
- 0 ambiguous pixel keys in results.jsonl (two stored detections round to one pixel; those keys are left unmapped)
- 1058 server labels share a pixel with another label and so map to the same stored detection (a re-submitted campaign does this)
- `streets.geojson`: 12567 features, sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`, 2026-09-30T01:39:49+00:00 (6.1 days old at run time), from https://sidewalk-vancouver.cs.washington.edu/v3/api/streets?filetype=geojson; 11783 open streets kept (the server snaps to open streets only)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 6009 blocks, largest 127 labels

## Validation checks

- ps_repro reproduces deployed: 14300/18684 clusters identical (15431 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 17412/17414 clusters identical (7 labels differ)

## Deployed-partition diagnostics (#56 (a))

Printed when ps_repro does not reproduce every deployed cluster. All at the labels' CURRENT positions in the pull (eval_ps_clustering.deployed_diagnostics has the definitions).

- deployed clusters spanning more than 7.5 m (a fresh complete-linkage cut cannot): 276 of 18684 (1.5%)
- cluster geometries more than 1 m from their members' mean: 3665 of 18684 (19.6%); max 11.1 m
- labels in more than one deployed cluster: 154
- deployed clusters ps_repro does not reproduce: 4384; proper subsets of one ps_repro cluster: 2896; of those, created all before or all after the rest of their ps_repro cluster: 929
- verbatim script with each label in its deployed cluster's region instead of its own: 14175/18684 deployed clusters identical (0.759)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 14562 of that account's labels did not map (--ai-user: kept as AI, unplaceable, at the tier)
- warning: fusion arm vs ps_* arms cover the same labels: 57982 fusion members vs 48373 placeable server labels; 13338 of 57982 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at auto vs runs/vancouver/fusion_eval/report.md (published in the 2.6 m frame): precision n/a, recall (union) n/a, dual 0/0/0 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 50252 AI (by account; 1058 share a detection with another AI label) + 14562 unmapped AI (--ai-user, at 0.55) + 33 human labels on 28894 panos (25242 positioned by inverting their labels, 3620 from the run's pano block (no label within 15 m to invert), 32 with neither, whose 41 labels are singleton clusters); 2369 labels the raycast cannot place (range cap, horizon) are singleton clusters; 5257 of its 14364 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 64847 label ids, 64847 distinct, of 64847 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 25242 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): n/a (no submission records beside the run)
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 24910 panos: median 0.010 m, p90 0.20 m; from all labels (the #105 method), over 24910 panos: median 0.010 m, p90 0.20 m; from human labels only, over 3 panos: median 0.006 m, p90 0.02 m, max 0.02 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| deployed | 18684 | 16790 | 64918 | 3.47 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_repro | 17414 | 15889 | 64764 | 3.72 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 2.5 m | 31061 | 25870 | 64814 | 2.09 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 5 m | 21049 | 18557 | 64814 | 3.08 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 7.5 m | 17451 | 15927 | 64814 | 3.71 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 10 m | 16034 | 14928 | 64814 | 4.04 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 12.5 m | 15421 | 14528 | 64814 | 4.20 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps @ 15 m | 15080 | 14285 | 64814 | 4.30 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_citywide @ 7.5 m | 17310 | 15814 | 64814 | 3.74 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 2.5 m | 24969 | 24969 | 48373 | 1.94 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 5 m | 17677 | 17677 | 48373 | 2.74 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 7.5 m | 15148 | 15148 | 48373 | 3.19 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 10 m | 14171 | 14171 | 48373 | 3.41 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 12.5 m | 13704 | 13704 | 48373 | 3.53 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_placeable @ 15 m | 13373 | 13373 | 48373 | 3.62 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 2.5 m | 22435 | 22435 | 48373 | 2.16 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 5 m | 15949 | 15949 | 48373 | 3.03 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 7.5 m | 14274 | 14274 | 48373 | 3.39 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 10 m | 13773 | 13773 | 48373 | 3.51 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 12.5 m | 13534 | 13534 | 48373 | 3.57 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| ps_raycast @ 15 m | 13281 | 13281 | 48373 | 3.64 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion | 14869 | 14869 | 57982 | 3.90 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_refit | 14869 | 14869 | 57982 | 3.90 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_server | 18484 | 14364 | 64847 | 3.51 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |
| fusion_server+attach | 16171 | 14364 | 64847 | 4.01 | n/a (0/0) | n/a (0/0) | 0 | n/a | 0/0/0 | n/a (0) | n/a (0) | 0/0/0 | n/a / n/a / 0 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 0 of this run's 0 pool ramps are self-detected, so most of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 4295, 2: 3406, 3: 2860, 4: 2400, 5: 2090, 6: 1715, 7: 1043, 8: 536, 9: 194, 10: 82, 11: 31, 12: 24, 13: 6, 14: 1, 15: 1
- server centroid vs raycast centroid, same members (n=16790): median 1.1 m, p90 4.4 m
- per-label server lat/lng vs labeler raycast (n=48373): median 0.7 m, p90 3.9 m, over 5 m 0.00

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 7750 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 3.9 m | 7.9 m | 0.12 | 0.03 | 0.00 |
| raycast | 3.1 m | 5.6 m | 0.02 | 0.00 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed coverage | deployed frag 5 m | deployed dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |
| 5 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |
| 7.5 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |
| 10 | n/a | 0/0 | 0/0 | n/a | 0/0 | 0/0 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed | 18684 | 64918 | 287.8 | server | 18684 | 0.352 | 0.583 | 0.799 | 16790 | 0.338 | 0.557 | 0.777 |
| ps_repro | 17414 | 64764 | 268.9 | server | 17414 | 0.274 | 0.516 | 0.771 | 15889 | 0.277 | 0.503 | 0.751 |
| ps @ 2.5 m | 31061 | 64814 | 479.2 | server | 31061 | 0.784 | 0.863 | 0.926 | 25870 | 0.743 | 0.828 | 0.906 |
| ps @ 5 m | 21049 | 64814 | 324.8 | server | 21049 | 0.439 | 0.678 | 0.842 | 18557 | 0.426 | 0.643 | 0.816 |
| ps @ 7.5 m | 17451 | 64814 | 269.2 | server | 17451 | 0.274 | 0.515 | 0.769 | 15927 | 0.276 | 0.502 | 0.750 |
| ps @ 10 m | 16034 | 64814 | 247.4 | server | 16034 | 0.260 | 0.445 | 0.723 | 14928 | 0.260 | 0.443 | 0.713 |
| ps @ 12.5 m | 15421 | 64814 | 237.9 | server | 15421 | 0.262 | 0.431 | 0.693 | 14528 | 0.262 | 0.430 | 0.692 |
| ps @ 15 m | 15080 | 64814 | 232.7 | server | 15080 | 0.262 | 0.428 | 0.683 | 14285 | 0.262 | 0.428 | 0.685 |
| ps_citywide @ 7.5 m | 17310 | 64814 | 267.1 | server | 17310 | 0.261 | 0.506 | 0.765 | 15814 | 0.264 | 0.494 | 0.745 |
| ps_placeable @ 2.5 m | 24969 | 48373 | 516.2 | server | 24969 | 0.721 | 0.814 | 0.900 | 24969 | 0.721 | 0.814 | 0.900 |
| ps_placeable @ 5 m | 17677 | 48373 | 365.4 | server | 17677 | 0.360 | 0.602 | 0.798 | 17677 | 0.360 | 0.602 | 0.798 |
| ps_placeable @ 7.5 m | 15148 | 48373 | 313.1 | server | 15148 | 0.231 | 0.443 | 0.720 | 15148 | 0.231 | 0.443 | 0.720 |
| ps_placeable @ 10 m | 14171 | 48373 | 293.0 | server | 14171 | 0.223 | 0.385 | 0.671 | 14171 | 0.223 | 0.385 | 0.671 |
| ps_placeable @ 12.5 m | 13704 | 48373 | 283.3 | server | 13704 | 0.222 | 0.376 | 0.642 | 13704 | 0.222 | 0.376 | 0.642 |
| ps_placeable @ 15 m | 13373 | 48373 | 276.5 | server | 13373 | 0.221 | 0.375 | 0.632 | 13373 | 0.221 | 0.375 | 0.632 |
| ps_raycast @ 2.5 m | 22435 | 48373 | 463.8 | server | 22435 | 0.666 | 0.766 | 0.878 | 22435 | 0.666 | 0.766 | 0.878 |
| ps_raycast @ 5 m | 15949 | 48373 | 329.7 | server | 15949 | 0.347 | 0.510 | 0.747 | 15949 | 0.347 | 0.510 | 0.747 |
| ps_raycast @ 7.5 m | 14274 | 48373 | 295.1 | server | 14274 | 0.246 | 0.396 | 0.680 | 14274 | 0.246 | 0.396 | 0.680 |
| ps_raycast @ 10 m | 13773 | 48373 | 284.7 | server | 13773 | 0.230 | 0.372 | 0.650 | 13773 | 0.230 | 0.372 | 0.650 |
| ps_raycast @ 12.5 m | 13534 | 48373 | 279.8 | server | 13534 | 0.228 | 0.369 | 0.634 | 13534 | 0.228 | 0.369 | 0.634 |
| ps_raycast @ 15 m | 13281 | 48373 | 274.6 | server | 13281 | 0.226 | 0.368 | 0.626 | 13281 | 0.226 | 0.368 | 0.626 |
| fusion | 14869 | 57982 | 256.4 | raycast | 14869 | 0.235 | 0.432 | 0.650 | 14869 | 0.235 | 0.432 | 0.650 |
| fusion_refit | 14869 | 57982 | 256.4 | raycast | 14869 | 0.236 | 0.432 | 0.651 | 14869 | 0.236 | 0.432 | 0.651 |
| fusion_server | 18484 | 64847 | 285.0 | server | 18484 | 0.408 | 0.581 | 0.787 | 14674 | 0.263 | 0.434 | 0.698 |
| fusion_server+attach | 16171 | 64847 | 249.4 | server | 16171 | 0.294 | 0.471 | 0.730 | 14674 | 0.262 | 0.435 | 0.707 |

Pre-registered reading (#56 metric (b), the confirmatory test of Part 1's fragmentation claim): expected fusion_server near 5 m about half of ps @ 7.5 m; read 0.408 vs 0.274 (ratio 1.49), not the expected halving.

## Validation-based precision, human votes (#56 metric (c))

A label's verdict is the majority of its HUMAN Agree / Disagree votes (`validations` with `validator_type: Human`; a tie or Unsure-only is neither). Per cluster-size bucket: clusters holding >= 1 validated label, and the share of them holding >= 1 label voted FALSE (`any false`) or only FALSE labels (`all false`). A validator judged a label on its own pano, so a false member says the cluster holds a false label, not that the ramp is absent.

- labels with a human vote: 2912 (2714 true, 81 false, 117 tie/unsure); AI labels among them: 2911 (2713 true, 81 false) -> label-level precision 0.971 [0.96, 0.98]
- the feed's own `correct` over the AI labels (reported apart; it includes PS's AI validator where one votes): 2713 true, 81 false, 62020 null
- labels in no deployed cluster: 83, of them 81 of the 81 voted false (the server clusters only labels not marked incorrect)

| arm | bucket | clusters | with a validated label | any false | all false | validated labels | false labels |
|---|---|---:|---:|---|---|---:|---:|
| deployed | cluster of 1 | 4295 | 145 | 0.000 (0) | 0.000 (0) | 145 | 0 |
| deployed | cluster of 2 | 3406 | 270 | 0.000 (0) | 0.000 (0) | 282 | 0 |
| deployed | cluster of 3+ | 10983 | 2196 | 0.000 (0) | 0.000 (0) | 2414 | 0 |
| ps @ 7.5 m | cluster of 1 | 3655 | 169 | 0.243 (41) | 0.243 (41) | 169 | 41 |
| ps @ 7.5 m | cluster of 2 | 2944 | 243 | 0.062 (15) | 0.062 (15) | 251 | 15 |
| ps @ 7.5 m | cluster of 3+ | 10852 | 2253 | 0.011 (25) | 0.010 (23) | 2491 | 25 |
| ps_citywide @ 7.5 m | cluster of 1 | 3563 | 165 | 0.248 (41) | 0.248 (41) | 165 | 41 |
| ps_citywide @ 7.5 m | cluster of 2 | 2905 | 239 | 0.063 (15) | 0.063 (15) | 246 | 15 |
| ps_citywide @ 7.5 m | cluster of 3+ | 10842 | 2261 | 0.011 (25) | 0.010 (23) | 2500 | 25 |
| fusion_server | cluster of 1 | 6144 | 264 | 0.170 (45) | 0.170 (45) | 264 | 45 |
| fusion_server | cluster of 2 | 2278 | 194 | 0.093 (18) | 0.093 (18) | 201 | 18 |
| fusion_server | cluster of 3+ | 10062 | 2184 | 0.008 (18) | 0.007 (16) | 2447 | 18 |
| fusion_server+attach | cluster of 1 | 3708 | 162 | 0.272 (44) | 0.272 (44) | 162 | 44 |
| fusion_server+attach | cluster of 2 | 2289 | 193 | 0.088 (17) | 0.088 (17) | 200 | 17 |
| fusion_server+attach | cluster of 3+ | 10174 | 2267 | 0.009 (20) | 0.007 (16) | 2550 | 20 |
| fusion | cluster of 1 | 0 | 0 | n/a | n/a | 0 | 0 |
| fusion | cluster of 2 | 0 | 0 | n/a | n/a | 0 | 0 |
| fusion | cluster of 3+ | 0 | 0 | n/a | n/a | 0 | 0 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `deployed` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| deployed | unplaceable | 1879 | 0.73 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| deployed | unclustered | 58 | 0.65 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| deployed | cluster of 1 | 2825 | 0.69 | 0.555 | 0 | n/a | 0 | 0 | 0 |
| deployed | cluster of 2 | 4817 | 0.79 | 0.568 | 0 | n/a | 0 | 0 | 0 |
| deployed | cluster of 3+ | 40673 | 0.88 | 0.568 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | unplaceable | 1879 | 0.73 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 1 | 2587 | 0.70 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 2 | 3348 | 0.79 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 3+ | 42438 | 0.87 | 0.568 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | unplaceable | 1879 | 0.73 | 0.523 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 2512 | 0.69 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 2 | 3276 | 0.79 | 0.570 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 3+ | 42585 | 0.87 | 0.568 | 0 | n/a | 0 | 0 | 0 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 2369; attached 2313 (0.98)
- clusters: 18484 (`fusion_server`) -> 16171 (`fusion_server+attach`)
- sanity (not a metric): 0 unplaceable labels are on judged panos; 0 of them attached (0 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)

## Offline server arm vs this server (`--offline-check`)

Validates the offline mode used for cities without a server: the labels it synthesizes from `results.jsonl` and places with the server's estimator (ps_placement), against the labels this server actually holds.

- (a) placement, 64006 of 64814 AI labels re-placed from the file: 1 labels on 1 panos left out because the position their labels imply (inverting the server's estimator, median per pano) is > 0.5 m from the file's; over the other 64005: median 0.000000 m, p90 0.000000 m, max 0.157542 m, 0 over 0.5 m (all labels: median 0.000000 m, p90 0.000000 m, 1 over 0.5 m, max 0.550202 m)
- (a) placement check: **FAIL** (max error over ALL labels <= 0.5 m). Gated on all labels because the moved-pano exclusion above is self-referential: it inverts the estimator being validated, so an error consistent within a pano would be excluded, not failed
- (b) labels: 60104 synthesized at 0.55 (unmasked); 46257 match a live AI label by pano and pixel, 13847 do not (soft-deleted, or never sent); 18557 live AI labels have no synthesized twin; 48 matched labels are in no deployed cluster (the server has not clustered them) and are left out of the partitions
- (b) regions by nearest street to the offline position: 46208 of 46208 equal the label's live region_id (0 equidistant ties)
- (b) partition: of the 6956 deployed clusters whose labels are all AI and all synthesized (of 18684), 5099 are, label for label, a cluster of the offline `ps @ 7.5 m` (0.733; 4992 labels in the others)
- (b+) the same with the 31 live human labels added at their live positions and regions: 5108 of the 6965 deployed clusters whose labels are all present are reproduced (0.733; 4992 labels in the others)
