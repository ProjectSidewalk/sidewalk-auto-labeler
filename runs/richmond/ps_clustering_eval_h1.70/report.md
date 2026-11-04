# richmond: PS label clustering vs RampNet GT

labels: 9639 CurbRamp on the server, 9526 map to stored detections (AI), 113 do not (human); 2156 server clusters over 9634 labels
scorer 106.4; results `results.jsonl` sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
inputs: streets sha256 `none`; verdicts sha256 `3721a2ac75a056fdd1e3ab45d9fff33e118d5851ae9c50a3a16e42ae2e15562d`
raycast camera height 1.7 m; fusion arm at --min-confidence 0.55
GT: 124 judged panos -> 289 placeable points -> 289 ramps (0 cross-pano merges), 289 in the recall pool; raycast placed 8885 of 9526 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 236, 'out_of_range': 405})

## Data provenance

- `raw_labels.geojson`: 9639 features, sha256 `17bde58ca3d678099195923781cf9846f5dc237dc1fd8e897417d4658087a058`, 2026-09-21T13:34:28+00:00 (43.4 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 2156 features, sha256 `3f7ca04dfc67c32950f8a14150c54ff6c3cb2485b069c60b7554072e30641fd0`, 2026-09-21T13:34:28+00:00 (43.4 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- labels by account: 51b0b927-3c8a-45b2-93de-bd878d1e5cf4 (AI) 9526, 549187e0-82c9-4014-a48d-31f18083d575 81, 18b26a38-24ab-402d-a64e-158fc0bb8a8a 30, 61460b3e-712d-4732-9044-924c4c1fc221 2
- 0 labels dropped before clustering (null lng or lng > 360), matching label_clustering.clean_label_data
- 0 ambiguous pixel keys in results.jsonl (two stored detections round to one pixel; those keys are left unmapped)
- 0 server labels share a pixel with another label and so map to the same stored detection (a re-submitted campaign does this)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 406 blocks, largest 141 labels

## Validation checks

- ps_repro reproduces deployed: 2156/2156 clusters identical (0 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 2156/2156 clusters identical (0 labels differ)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 0 of that account's labels did not map (should be 0)
- fusion arm vs ps_* arms cover the same labels: 8885 fusion members vs 8885 placeable server labels; 0 of 8885 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 1.7 m vs runs/richmond/fusion_eval/report.md (published in the 2.6 m frame): precision 0.962, recall (union) 0.927, dual 32/20/3 — a different frame, so the world-space figures are expected to differ; precision is the frame-free part
- fusion_server input: 9526 AI (by account; 0 share a detection with another AI label) + 113 human labels on 3721 panos (3057 positioned by inverting their labels, 661 from the run's pano block (no label within 15 m to invert), 3 with neither, whose 3 labels are singleton clusters); 642 labels the raycast cannot place (range cap, horizon) are singleton clusters; 1674 of its 1819 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 9639 label ids, 9639 distinct, of 9639 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 3057 (should be 0 for a pull taken before any reposition)
- run-block fallback panos whose live position at the pull's fetch time differs from the run's block by more than 1 m (labels placed from a position this file does not hold): 0 of 661
- camera-position inversion vs the run's position: from the AI account's labels (the arm's rule), over 3023 panos: median 0.010 m, p90 0.20 m; from all labels (the #105 method), over 3024 panos: median 0.010 m, p90 0.20 m; from human labels only, over 12 panos: median 0.007 m, p90 0.07 m, max 0.54 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| deployed | 2156 | 2055 | 9634 | 4.47 | 0.964 (238/9) | 0.862 (249/289) | 19 | 0.927 | 229/39/21 | 0.11 (27) | 0.34 (103) | 36/14/5 | 2.37 / 4.91 / 21 |
| ps_repro | 2156 | 2055 | 9634 | 4.47 | 0.964 (238/9) | 0.862 (249/289) | 19 | 0.927 | 229/39/21 | 0.11 (27) | 0.34 (103) | 36/14/5 | 2.37 / 4.91 / 21 |
| ps @ 2.5 m | 4515 | 4193 | 9526 | 2.11 | 0.964 (238/9) | 0.965 (279/289) | 0 | 0.965 | 229/50/10 | 0.45 (198) | 0.77 (540) | 48/4/3 | 1.30 / 3.30 / 0 |
| ps @ 5 m | 2850 | 2685 | 9526 | 3.34 | 0.964 (238/9) | 0.913 (264/289) | 12 | 0.955 | 229/47/13 | 0.19 (55) | 0.52 (202) | 42/8/5 | 2.08 / 4.32 / 12 |
| ps @ 7.5 m | 2149 | 2049 | 9526 | 4.43 | 0.964 (238/9) | 0.858 (248/289) | 20 | 0.927 | 229/39/21 | 0.11 (27) | 0.33 (98) | 35/15/5 | 2.37 / 4.94 / 22 |
| ps @ 10 m | 1773 | 1717 | 9526 | 5.37 | 0.963 (237/9) | 0.806 (233/289) | 34 | 0.924 | 229/38/22 | 0.08 (19) | 0.20 (51) | 31/19/5 | 2.49 / 5.59 / 33 |
| ps @ 12.5 m | 1585 | 1546 | 9526 | 6.01 | 0.963 (237/9) | 0.799 (231/289) | 35 | 0.920 | 229/37/23 | 0.07 (17) | 0.17 (41) | 31/19/5 | 2.57 / 5.82 / 34 |
| ps @ 15 m | 1487 | 1458 | 9526 | 6.41 | 0.963 (237/9) | 0.792 (229/289) | 36 | 0.917 | 229/36/24 | 0.06 (14) | 0.15 (36) | 32/18/5 | 2.57 / 5.82 / 36 |
| ps_citywide @ 7.5 m | 2115 | 2016 | 9526 | 4.50 | 0.964 (238/9) | 0.862 (249/289) | 19 | 0.927 | 229/39/21 | 0.09 (23) | 0.33 (95) | 35/15/5 | 2.37 / 4.91 / 20 |
| ps_placeable @ 2.5 m | 4163 | 4163 | 8885 | 2.13 | 0.962 (230/9) | 0.965 (279/289) | 0 | 0.965 | 229/50/10 | 0.44 (195) | 0.76 (528) | 48/4/3 | 1.33 / 3.30 / 0 |
| ps_placeable @ 5 m | 2654 | 2654 | 8885 | 3.35 | 0.962 (230/9) | 0.913 (264/289) | 12 | 0.955 | 229/47/13 | 0.18 (53) | 0.51 (193) | 42/8/5 | 2.08 / 4.32 / 12 |
| ps_placeable @ 7.5 m | 2003 | 2003 | 8885 | 4.44 | 0.962 (230/9) | 0.844 (244/289) | 23 | 0.924 | 229/38/22 | 0.10 (24) | 0.32 (92) | 34/16/5 | 2.40 / 5.28 / 26 |
| ps_placeable @ 10 m | 1672 | 1672 | 8885 | 5.31 | 0.962 (229/9) | 0.803 (232/289) | 35 | 0.924 | 229/38/22 | 0.08 (18) | 0.19 (49) | 32/17/6 | 2.52 / 5.75 / 35 |
| ps_placeable @ 12.5 m | 1515 | 1515 | 8885 | 5.86 | 0.962 (229/9) | 0.789 (228/289) | 36 | 0.913 | 229/35/25 | 0.07 (16) | 0.16 (39) | 31/18/6 | 2.52 / 5.82 / 36 |
| ps_placeable @ 15 m | 1430 | 1430 | 8885 | 6.21 | 0.962 (229/9) | 0.782 (226/289) | 37 | 0.910 | 229/34/26 | 0.07 (15) | 0.16 (37) | 31/18/6 | 2.56 / 5.82 / 37 |
| ps_raycast @ 2.5 m | 4978 | 4978 | 8885 | 1.78 | 0.962 (230/9) | 0.969 (280/289) | 0 | 0.969 | 229/51/9 | 0.54 (229) | 0.87 (692) | 47/6/2 | 0.48 / 1.01 / 0 |
| ps_raycast @ 5 m | 3200 | 3200 | 8885 | 2.78 | 0.962 (230/9) | 0.952 (275/289) | 0 | 0.952 | 229/46/14 | 0.19 (57) | 0.62 (265) | 42/11/2 | 1.05 / 2.25 / 0 |
| ps_raycast @ 7.5 m | 2445 | 2445 | 8885 | 3.63 | 0.962 (230/9) | 0.941 (272/289) | 0 | 0.941 | 229/43/17 | 0.07 (20) | 0.39 (132) | 40/10/5 | 1.49 / 2.89 / 0 |
| ps_raycast @ 10 m | 2010 | 2010 | 8885 | 4.42 | 0.962 (230/9) | 0.893 (258/289) | 9 | 0.924 | 229/38/22 | 0.06 (16) | 0.30 (92) | 37/13/5 | 1.77 / 4.12 / 11 |
| ps_raycast @ 12.5 m | 1766 | 1766 | 8885 | 5.03 | 0.962 (229/9) | 0.834 (241/289) | 25 | 0.920 | 229/37/23 | 0.06 (15) | 0.27 (76) | 34/15/6 | 2.19 / 5.00 / 23 |
| ps_raycast @ 15 m | 1605 | 1605 | 8885 | 5.54 | 0.962 (229/9) | 0.799 (231/289) | 32 | 0.910 | 229/34/26 | 0.06 (14) | 0.24 (63) | 33/17/5 | 2.34 / 5.55 / 33 |
| fusion | 1805 | 1805 | 8885 | 4.92 | 0.962 (230/9) | 0.841 (243/289) | 25 | 0.927 | 229/39/21 | 0.05 (14) | 0.22 (55) | 31/20/4 | 2.36 / 5.44 / 29 |
| fusion_refit | 1805 | 1805 | 8885 | 4.92 | 0.962 (230/9) | 0.837 (242/289) | 26 | 0.927 | 229/39/21 | 0.05 (14) | 0.23 (56) | 32/20/3 | 2.28 / 5.51 / 31 |
| fusion_server | 2465 | 1819 | 9639 | 3.91 | 0.964 (238/9) | 0.858 (248/289) | 21 | 0.931 | 229/40/20 | 0.05 (13) | 0.21 (54) | 32/19/4 | 2.27 / 5.16 / 24 |
| fusion_server+attach | 2011 | 1819 | 9639 | 4.79 | 0.964 (238/9) | 0.858 (248/289) | 21 | 0.931 | 229/40/20 | 0.05 (13) | 0.21 (54) | 32/19/4 | 2.27 / 5.16 / 24 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 229 of this run's 289 pool ramps are self-detected, so 79% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 411, 2: 388, 3: 330, 4: 228, 5: 172, 6: 128, 7: 102, 8: 88, 9: 87, 10: 55, 11: 49, 12: 31, 13: 40, 14: 18, 15: 6, 16: 10, 17: 6, 18: 4, 19: 1, 20: 2
- server centroid vs raycast centroid, same members (n=2055): median 1.8 m, p90 3.3 m
- per-label server lat/lng vs labeler raycast (n=8885): median 2.5 m, p90 3.5 m, over 5 m 0.00

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 1123 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 7.2 m | 11.5 m | 0.46 | 0.20 | 0.01 |
| raycast | 8.4 m | 11.6 m | 0.64 | 0.26 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed coverage | deployed frag 5 m | deployed dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.561 | 77/162 | 14/55 | 0.536 | 45/155 | 14/55 |
| 5 | 0.862 | 85/249 | 36/55 | 0.841 | 54/243 | 31/55 |
| 7.5 | 0.965 | 83/279 | 45/55 | 0.955 | 44/276 | 44/55 |
| 10 | 0.976 | 83/282 | 48/55 | 0.965 | 44/279 | 46/55 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed | 2156 | 9634 | 223.8 | server | 2156 | 0.396 | 0.746 | 0.914 | 2055 | 0.406 | 0.742 | 0.908 |
| ps_repro | 2156 | 9634 | 223.8 | server | 2156 | 0.396 | 0.746 | 0.914 | 2055 | 0.406 | 0.742 | 0.908 |
| ps @ 2.5 m | 4515 | 9526 | 474.0 | server | 4515 | 0.884 | 0.954 | 0.981 | 4193 | 0.880 | 0.951 | 0.979 |
| ps @ 5 m | 2850 | 9526 | 299.2 | server | 2850 | 0.612 | 0.878 | 0.950 | 2685 | 0.619 | 0.872 | 0.947 |
| ps @ 7.5 m | 2149 | 9526 | 225.6 | server | 2149 | 0.390 | 0.744 | 0.913 | 2049 | 0.401 | 0.740 | 0.906 |
| ps @ 10 m | 1773 | 9526 | 186.1 | server | 1773 | 0.333 | 0.613 | 0.864 | 1717 | 0.337 | 0.618 | 0.858 |
| ps @ 12.5 m | 1585 | 9526 | 166.4 | server | 1585 | 0.322 | 0.560 | 0.818 | 1546 | 0.327 | 0.563 | 0.814 |
| ps @ 15 m | 1487 | 9526 | 156.1 | server | 1487 | 0.316 | 0.533 | 0.786 | 1458 | 0.321 | 0.537 | 0.786 |
| ps_citywide @ 7.5 m | 2115 | 9526 | 222.0 | server | 2115 | 0.368 | 0.734 | 0.911 | 2016 | 0.381 | 0.729 | 0.905 |
| ps_placeable @ 2.5 m | 4163 | 8885 | 468.5 | server | 4163 | 0.878 | 0.951 | 0.979 | 4163 | 0.878 | 0.951 | 0.979 |
| ps_placeable @ 5 m | 2654 | 8885 | 298.7 | server | 2654 | 0.612 | 0.868 | 0.946 | 2654 | 0.612 | 0.868 | 0.946 |
| ps_placeable @ 7.5 m | 2003 | 8885 | 225.4 | server | 2003 | 0.403 | 0.723 | 0.902 | 2003 | 0.403 | 0.723 | 0.902 |
| ps_placeable @ 10 m | 1672 | 8885 | 188.2 | server | 1672 | 0.341 | 0.605 | 0.848 | 1672 | 0.341 | 0.605 | 0.848 |
| ps_placeable @ 12.5 m | 1515 | 8885 | 170.5 | server | 1515 | 0.330 | 0.569 | 0.805 | 1515 | 0.330 | 0.569 | 0.805 |
| ps_placeable @ 15 m | 1430 | 8885 | 160.9 | server | 1430 | 0.324 | 0.548 | 0.776 | 1430 | 0.324 | 0.548 | 0.776 |
| ps_raycast @ 2.5 m | 4978 | 8885 | 560.3 | server | 4978 | 0.909 | 0.961 | 0.983 | 4978 | 0.909 | 0.961 | 0.983 |
| ps_raycast @ 5 m | 3200 | 8885 | 360.2 | server | 3200 | 0.795 | 0.914 | 0.963 | 3200 | 0.795 | 0.914 | 0.963 |
| ps_raycast @ 7.5 m | 2445 | 8885 | 275.2 | server | 2445 | 0.663 | 0.847 | 0.939 | 2445 | 0.663 | 0.847 | 0.939 |
| ps_raycast @ 10 m | 2010 | 8885 | 226.2 | server | 2010 | 0.533 | 0.769 | 0.908 | 2010 | 0.533 | 0.769 | 0.908 |
| ps_raycast @ 12.5 m | 1766 | 8885 | 198.8 | server | 1766 | 0.460 | 0.690 | 0.869 | 1766 | 0.460 | 0.690 | 0.869 |
| ps_raycast @ 15 m | 1605 | 8885 | 180.6 | server | 1605 | 0.432 | 0.635 | 0.832 | 1605 | 0.432 | 0.635 | 0.832 |
| fusion | 1805 | 8885 | 203.2 | raycast | 1805 | 0.329 | 0.579 | 0.896 | 1805 | 0.329 | 0.579 | 0.896 |
| fusion_refit | 1805 | 8885 | 203.2 | raycast | 1805 | 0.358 | 0.587 | 0.900 | 1805 | 0.358 | 0.587 | 0.900 |
| fusion_server | 2465 | 9639 | 255.7 | server | 2465 | 0.614 | 0.811 | 0.925 | 1819 | 0.438 | 0.698 | 0.882 |
| fusion_server+attach | 2011 | 9639 | 208.6 | server | 2011 | 0.473 | 0.727 | 0.902 | 1819 | 0.435 | 0.694 | 0.885 |

## Validation-based precision, human votes (#56 metric (c))

A label's verdict is the majority of its HUMAN Agree / Disagree votes (`validations` with `validator_type: Human`; a tie or Unsure-only is neither). Per cluster-size bucket: clusters holding >= 1 validated label, and the share of them holding >= 1 label voted FALSE (`any false`) or only FALSE labels (`all false`). A validator judged a label on its own pano, so a false member says the cluster holds a false label, not that the ramp is absent.

- labels with a human vote: 112 (111 true, 1 false, 0 tie/unsure); AI labels among them: 109 (108 true, 1 false) -> label-level precision 0.991 [0.95, 1.00]
- the feed's own `correct` over the AI labels (reported apart; it includes PS's AI validator where one votes): 108 true, 1 false, 9417 null
- labels in no deployed cluster: 5, of them 1 of the 1 voted false (the server clusters only labels not marked incorrect)

| arm | bucket | clusters | with a validated label | any false | all false | validated labels | false labels |
|---|---|---:|---:|---|---|---:|---:|
| deployed | cluster of 1 | 411 | 3 | 0.000 (0) | 0.000 (0) | 3 | 0 |
| deployed | cluster of 2 | 388 | 2 | 0.000 (0) | 0.000 (0) | 2 | 0 |
| deployed | cluster of 3+ | 1357 | 89 | 0.000 (0) | 0.000 (0) | 106 | 0 |
| ps @ 7.5 m | cluster of 1 | 413 | 3 | 0.000 (0) | 0.000 (0) | 3 | 0 |
| ps @ 7.5 m | cluster of 2 | 385 | 2 | 0.000 (0) | 0.000 (0) | 2 | 0 |
| ps @ 7.5 m | cluster of 3+ | 1351 | 87 | 0.011 (1) | 0.011 (1) | 104 | 1 |
| ps_citywide @ 7.5 m | cluster of 1 | 392 | 3 | 0.000 (0) | 0.000 (0) | 3 | 0 |
| ps_citywide @ 7.5 m | cluster of 2 | 372 | 2 | 0.000 (0) | 0.000 (0) | 2 | 0 |
| ps_citywide @ 7.5 m | cluster of 3+ | 1351 | 87 | 0.011 (1) | 0.011 (1) | 104 | 1 |
| fusion_server | cluster of 1 | 1071 | 9 | 0.000 (0) | 0.000 (0) | 9 | 0 |
| fusion_server | cluster of 2 | 253 | 2 | 0.000 (0) | 0.000 (0) | 2 | 0 |
| fusion_server | cluster of 3+ | 1141 | 85 | 0.012 (1) | 0.012 (1) | 101 | 1 |
| fusion_server+attach | cluster of 1 | 582 | 5 | 0.000 (0) | 0.000 (0) | 5 | 0 |
| fusion_server+attach | cluster of 2 | 252 | 2 | 0.000 (0) | 0.000 (0) | 2 | 0 |
| fusion_server+attach | cluster of 3+ | 1177 | 89 | 0.011 (1) | 0.011 (1) | 105 | 1 |
| fusion | cluster of 1 | 0 | 0 | n/a | n/a | 0 | 0 |
| fusion | cluster of 2 | 0 | 0 | n/a | n/a | 0 | 0 |
| fusion | cluster of 3+ | 0 | 0 | n/a | n/a | 0 | 0 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `deployed` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| deployed | unplaceable | 641 | 0.78 | 0.508 | 12 | 1.000 [0.68, 1.00] | 8 | 0 | 4 |
| deployed | unclustered | 1 | 0.67 | 0.555 | 0 | n/a | 0 | 0 | 0 |
| deployed | cluster of 1 | 360 | 0.76 | 0.555 | 12 | 0.909 [0.62, 0.98] | 10 | 1 | 1 |
| deployed | cluster of 2 | 668 | 0.81 | 0.555 | 19 | 0.824 [0.59, 0.94] | 14 | 3 | 2 |
| deployed | cluster of 3+ | 7856 | 0.86 | 0.568 | 224 | 0.976 [0.95, 0.99] | 205 | 5 | 14 |
| fusion_server | unplaceable | 641 | 0.78 | 0.508 | 12 | 1.000 [0.68, 1.00] | 8 | 0 | 4 |
| fusion_server | cluster of 1 | 425 | 0.71 | 0.555 | 13 | 0.818 [0.52, 0.95] | 9 | 2 | 2 |
| fusion_server | cluster of 2 | 501 | 0.76 | 0.555 | 20 | 0.778 [0.55, 0.91] | 14 | 4 | 2 |
| fusion_server | cluster of 3+ | 7959 | 0.87 | 0.568 | 222 | 0.986 [0.96, 1.00] | 206 | 3 | 13 |
| fusion_server+attach | unplaceable | 641 | 0.78 | 0.508 | 12 | 1.000 [0.68, 1.00] | 8 | 0 | 4 |
| fusion_server+attach | cluster of 1 | 390 | 0.72 | 0.555 | 11 | 0.778 [0.45, 0.94] | 7 | 2 | 2 |
| fusion_server+attach | cluster of 2 | 472 | 0.76 | 0.555 | 18 | 0.875 [0.64, 0.97] | 14 | 2 | 2 |
| fusion_server+attach | cluster of 3+ | 8023 | 0.87 | 0.568 | 226 | 0.977 [0.95, 0.99] | 208 | 5 | 13 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 642; attached 454 (0.71)
- clusters: 2465 (`fusion_server`) -> 2011 (`fusion_server+attach`)
- sanity (not a metric): 12 unplaceable labels are on judged panos; 11 of them attached (8 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)
