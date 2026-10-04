# richmond: PS label clustering vs RampNet GT

labels: 9639 CurbRamp on the server, 9526 map to stored detections (AI), 113 do not (human); 2156 server clusters over 9634 labels
scorer 106.3; results `results.jsonl` sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`
inputs: streets sha256 `02f3e0061c16ecda5d6ba5523f25bbff5999d758061cb75dd408edb732f92206`; verdicts sha256 `3721a2ac75a056fdd1e3ab45d9fff33e118d5851ae9c50a3a16e42ae2e15562d`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.55
GT: 124 judged panos -> 253 placeable points -> 253 ramps (0 cross-pano merges), 253 in the recall pool; raycast placed 8098 of 9526 detections (drops {'below_floor': 0, 'on_rig': 0, 'horizon': 236, 'out_of_range': 1192})

## Data provenance

- `raw_labels.geojson`: 9639 features, sha256 `17bde58ca3d678099195923781cf9846f5dc237dc1fd8e897417d4658087a058`, 2026-09-21T13:34:28+00:00 (13.3 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 2156 features, sha256 `3f7ca04dfc67c32950f8a14150c54ff6c3cb2485b069c60b7554072e30641fd0`, 2026-09-21T13:34:28+00:00 (13.3 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- labels by account: 51b0b927-3c8a-45b2-93de-bd878d1e5cf4 (AI) 9526, 549187e0-82c9-4014-a48d-31f18083d575 81, 18b26a38-24ab-402d-a64e-158fc0bb8a8a 30, 61460b3e-712d-4732-9044-924c4c1fc221 2
- 0 labels dropped before clustering (null lng or lng > 360), matching label_clustering.clean_label_data
- 0 ambiguous pixel keys in results.jsonl (two stored detections round to one pixel; those keys are left unmapped)
- 0 server labels share a pixel with another label and so map to the same stored detection (a re-submitted campaign does this)
- `streets.geojson`: 16365 features, sha256 `02f3e0061c16ecda5d6ba5523f25bbff5999d758061cb75dd408edb732f92206`, 2026-09-28T17:07:50+00:00 (6.1 days old at run time), from https://sidewalk-richmond.cs.washington.edu/v3/api/streets?filetype=geojson; 704 open streets kept (the server snaps to open streets only)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 406 blocks, largest 141 labels

## Validation checks

- ps_repro reproduces deployed: 2156/2156 clusters identical (0 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 2156/2156 clusters identical (0 labels differ)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 0 of that account's labels did not map (should be 0)
- fusion arm vs ps_* arms cover the same labels: 8098 fusion members vs 8098 placeable server labels; 0 of 8098 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/richmond/fusion_eval/report.md (published in the 2.6 m frame): precision 0.959, recall (union) 0.941, dual 23/4/3
- fusion_server input: 9526 AI (by account; 0 share a detection with another AI label) + 113 human labels on 3721 panos (3057 positioned by inverting their labels, 661 from the run's pano block (no label within 15 m to invert), 3 with neither, whose 3 labels are singleton clusters); 1434 labels the raycast cannot place (range cap, horizon) are singleton clusters; 1485 of its 1589 clusters with AI members are, member for member, a cluster of the `fusion` arm
- every server label is in exactly one fusion_server cluster: 9639 label ids, 9639 distinct, of 9639 labels
- inverted camera positions more than 1 m from the run's pano block (a pano live somewhere other than results.jsonl says, e.g. repositioned): 0 of 3057 (should be 0 for a pull taken before any reposition)
- camera-position inversion vs the run's position, over 3024 panos in both (all labels, mostly AI): median 0.010 m, p90 0.20 m; from human labels only, over 12 panos: median 0.007 m, p90 0.07 m, max 0.54 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| deployed | 2156 | 1993 | 9634 | 4.47 | 0.964 (238/9) | 0.917 (232/253) | 9 | 0.953 | 210/31/12 | 0.24 (60) | 0.47 (132) | 23/4/3 | 1.49 / 3.98 / 11 |
| ps_repro | 2156 | 1993 | 9634 | 4.47 | 0.964 (238/9) | 0.917 (232/253) | 9 | 0.953 | 210/31/12 | 0.24 (60) | 0.47 (132) | 23/4/3 | 1.49 / 3.98 / 11 |
| ps @ 2.5 m | 4515 | 3908 | 9526 | 2.11 | 0.964 (238/9) | 0.964 (244/253) | 1 | 0.968 | 210/35/8 | 0.61 (295) | 0.80 (542) | 27/1/2 | 0.80 / 2.57 / 2 |
| ps @ 5 m | 2850 | 2565 | 9526 | 3.34 | 0.964 (238/9) | 0.937 (237/253) | 6 | 0.960 | 210/33/10 | 0.38 (113) | 0.62 (235) | 25/2/3 | 1.40 / 3.73 / 8 |
| ps @ 7.5 m | 2149 | 1987 | 9526 | 4.43 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (59) | 0.47 (130) | 23/4/3 | 1.52 / 3.98 / 11 |
| ps @ 10 m | 1773 | 1676 | 9526 | 5.37 | 0.963 (237/9) | 0.889 (225/253) | 14 | 0.945 | 210/29/14 | 0.15 (33) | 0.32 (81) | 22/5/3 | 1.80 / 4.43 / 15 |
| ps @ 12.5 m | 1585 | 1516 | 9526 | 6.01 | 0.963 (237/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.14 (31) | 0.29 (68) | 22/5/3 | 1.86 / 4.57 / 18 |
| ps @ 15 m | 1487 | 1434 | 9526 | 6.41 | 0.963 (237/9) | 0.874 (221/253) | 17 | 0.941 | 210/28/15 | 0.13 (28) | 0.26 (60) | 22/5/3 | 1.95 / 4.57 / 18 |
| ps_citywide @ 7.5 m | 2115 | 1956 | 9526 | 4.50 | 0.964 (238/9) | 0.913 (231/253) | 9 | 0.949 | 210/30/13 | 0.24 (58) | 0.46 (126) | 23/4/3 | 1.69 / 4.00 / 11 |
| ps_placeable @ 2.5 m | 3843 | 3843 | 8098 | 2.11 | 0.959 (211/9) | 0.960 (243/253) | 1 | 0.964 | 210/34/9 | 0.60 (289) | 0.78 (526) | 26/2/2 | 0.81 / 2.62 / 2 |
| ps_placeable @ 5 m | 2475 | 2475 | 8098 | 3.27 | 0.959 (211/9) | 0.929 (235/253) | 7 | 0.957 | 210/32/11 | 0.34 (101) | 0.60 (221) | 25/2/3 | 1.40 / 3.68 / 9 |
| ps_placeable @ 7.5 m | 1887 | 1887 | 8098 | 4.29 | 0.959 (211/9) | 0.897 (227/253) | 13 | 0.949 | 210/30/13 | 0.20 (46) | 0.44 (113) | 23/4/3 | 1.54 / 4.06 / 14 |
| ps_placeable @ 10 m | 1590 | 1590 | 8098 | 5.09 | 0.959 (211/9) | 0.893 (226/253) | 14 | 0.949 | 210/30/13 | 0.11 (25) | 0.29 (70) | 23/4/3 | 1.80 / 4.50 / 15 |
| ps_placeable @ 12.5 m | 1461 | 1461 | 8098 | 5.54 | 0.959 (211/9) | 0.877 (222/253) | 17 | 0.945 | 210/29/14 | 0.12 (26) | 0.27 (61) | 23/4/3 | 1.78 / 4.57 / 18 |
| ps_placeable @ 15 m | 1384 | 1384 | 8098 | 5.85 | 0.959 (211/9) | 0.874 (221/253) | 16 | 0.937 | 210/27/16 | 0.11 (24) | 0.24 (54) | 23/4/3 | 1.89 / 4.51 / 17 |
| ps_raycast @ 2.5 m | 3912 | 3912 | 8098 | 2.07 | 0.959 (211/9) | 0.968 (245/253) | 0 | 0.968 | 210/35/8 | 0.56 (211) | 0.81 (477) | 27/1/2 | 0.48 / 1.14 / 0 |
| ps_raycast @ 5 m | 2584 | 2584 | 8098 | 3.13 | 0.959 (211/9) | 0.957 (242/253) | 0 | 0.957 | 210/32/11 | 0.17 (45) | 0.53 (169) | 24/3/3 | 1.05 / 2.17 / 0 |
| ps_raycast @ 7.5 m | 1979 | 1979 | 8098 | 4.09 | 0.959 (211/9) | 0.949 (240/253) | 1 | 0.953 | 210/31/12 | 0.07 (18) | 0.31 (78) | 23/4/3 | 1.34 / 2.72 / 0 |
| ps_raycast @ 10 m | 1638 | 1638 | 8098 | 4.94 | 0.959 (211/9) | 0.913 (231/253) | 8 | 0.945 | 210/29/14 | 0.06 (16) | 0.22 (52) | 23/4/3 | 1.76 / 3.75 / 8 |
| ps_raycast @ 12.5 m | 1488 | 1488 | 8098 | 5.44 | 0.959 (211/9) | 0.889 (225/253) | 13 | 0.941 | 210/28/15 | 0.06 (15) | 0.19 (44) | 23/4/3 | 1.80 / 4.43 / 13 |
| ps_raycast @ 15 m | 1396 | 1396 | 8098 | 5.80 | 0.959 (211/9) | 0.874 (221/253) | 17 | 0.941 | 210/28/15 | 0.06 (15) | 0.17 (40) | 23/4/3 | 1.86 / 4.98 / 18 |
| fusion | 1570 | 1570 | 8098 | 5.16 | 0.959 (211/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.06 (14) | 0.16 (38) | 24/3/3 | 1.84 / 4.47 / 12 |
| fusion_refit | 1570 | 1570 | 8098 | 5.16 | 0.959 (211/9) | 0.897 (227/253) | 11 | 0.941 | 210/28/15 | 0.07 (17) | 0.16 (39) | 23/4/3 | 1.72 / 4.67 / 14 |
| fusion_server | 3030 | 1589 | 9639 | 3.18 | 0.964 (238/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.07 (15) | 0.17 (44) | 24/3/3 | 1.84 / 4.42 / 12 |
| fusion_server+attach | 1899 | 1589 | 9639 | 5.08 | 0.963 (237/9) | 0.909 (230/253) | 9 | 0.945 | 210/29/14 | 0.07 (15) | 0.17 (44) | 24/3/3 | 1.84 / 4.42 / 12 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 210 of this run's 253 pool ramps are self-detected, so 83% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 411, 2: 388, 3: 330, 4: 228, 5: 172, 6: 128, 7: 102, 8: 88, 9: 87, 10: 55, 11: 49, 12: 31, 13: 40, 14: 18, 15: 6, 16: 10, 17: 6, 18: 4, 19: 1, 20: 2
- server centroid vs raycast centroid, same members (n=1993): median 1.5 m, p90 4.4 m
- per-label server lat/lng vs labeler raycast (n=8098): median 1.2 m, p90 4.8 m, over 5 m 0.07

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 945 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 7.4 m | 12.1 m | 0.49 | 0.24 | 0.02 |
| raycast | 7.1 m | 10.7 m | 0.45 | 0.14 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed coverage | deployed frag 5 m | deployed dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.684 | 102/173 | 15/30 | 0.680 | 43/172 | 15/30 |
| 5 | 0.917 | 109/232 | 23/30 | 0.909 | 36/230 | 24/30 |
| 7.5 | 0.957 | 109/242 | 23/30 | 0.949 | 35/240 | 24/30 |
| 10 | 0.976 | 108/247 | 25/30 | 0.968 | 35/245 | 25/30 |

## Fragmentation proxy, GT-free (#56 metric (b))

Share of an arm's placed clusters with ANOTHER cluster of the same arm within r, and clusters per 1,000 labels. Clusters that carry label ids sit at the mean of their labels' SERVER positions (the server's own frame, no run needed); the run-only `fusion` arms sit at their raycast position. Two clusters of one ramp read as a near pair, and so do two real ramps of one corner, so this is a proxy for `frag`, not the same quantity. The `all clusters` columns are the pre-registered read; `placeable-member clusters` (post hoc) drop the clusters no member of which the raycast can place -- fusion_server leaves every such label a singleton beside the site it could not join, and those singletons dominate the all-clusters read.

| arm | clusters | labels | clusters / 1,000 labels | frame | all clusters | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters | near 5 m | near 7.5 m | near 12.5 m |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed | 2156 | 9634 | 223.8 | server | 2156 | 0.396 | 0.746 | 0.914 | 1993 | 0.414 | 0.745 | 0.906 |
| ps_repro | 2156 | 9634 | 223.8 | server | 2156 | 0.396 | 0.746 | 0.914 | 1993 | 0.414 | 0.745 | 0.906 |
| ps @ 2.5 m | 4515 | 9526 | 474.0 | server | 4515 | 0.884 | 0.954 | 0.981 | 3908 | 0.872 | 0.947 | 0.977 |
| ps @ 5 m | 2850 | 9526 | 299.2 | server | 2850 | 0.612 | 0.878 | 0.950 | 2565 | 0.622 | 0.869 | 0.946 |
| ps @ 7.5 m | 2149 | 9526 | 225.6 | server | 2149 | 0.390 | 0.744 | 0.913 | 1987 | 0.409 | 0.743 | 0.904 |
| ps @ 10 m | 1773 | 9526 | 186.1 | server | 1773 | 0.333 | 0.613 | 0.864 | 1676 | 0.341 | 0.621 | 0.859 |
| ps @ 12.5 m | 1585 | 9526 | 166.4 | server | 1585 | 0.322 | 0.560 | 0.818 | 1516 | 0.328 | 0.566 | 0.817 |
| ps @ 15 m | 1487 | 9526 | 156.1 | server | 1487 | 0.316 | 0.533 | 0.786 | 1434 | 0.321 | 0.539 | 0.790 |
| ps_citywide @ 7.5 m | 2115 | 9526 | 222.0 | server | 2115 | 0.368 | 0.734 | 0.911 | 1956 | 0.389 | 0.732 | 0.903 |
| ps_placeable @ 2.5 m | 3843 | 8098 | 474.6 | server | 3843 | 0.869 | 0.946 | 0.976 | 3843 | 0.869 | 0.946 | 0.976 |
| ps_placeable @ 5 m | 2475 | 8098 | 305.6 | server | 2475 | 0.590 | 0.857 | 0.941 | 2475 | 0.590 | 0.857 | 0.941 |
| ps_placeable @ 7.5 m | 1887 | 8098 | 233.0 | server | 1887 | 0.384 | 0.707 | 0.894 | 1887 | 0.384 | 0.707 | 0.894 |
| ps_placeable @ 10 m | 1590 | 8098 | 196.3 | server | 1590 | 0.333 | 0.587 | 0.831 | 1590 | 0.333 | 0.587 | 0.831 |
| ps_placeable @ 12.5 m | 1461 | 8098 | 180.4 | server | 1461 | 0.327 | 0.551 | 0.795 | 1461 | 0.327 | 0.551 | 0.795 |
| ps_placeable @ 15 m | 1384 | 8098 | 170.9 | server | 1384 | 0.323 | 0.530 | 0.768 | 1384 | 0.323 | 0.530 | 0.768 |
| ps_raycast @ 2.5 m | 3912 | 8098 | 483.1 | server | 3912 | 0.871 | 0.950 | 0.978 | 3912 | 0.871 | 0.950 | 0.978 |
| ps_raycast @ 5 m | 2584 | 8098 | 319.1 | server | 2584 | 0.707 | 0.869 | 0.950 | 2584 | 0.707 | 0.869 | 0.950 |
| ps_raycast @ 7.5 m | 1979 | 8098 | 244.4 | server | 1979 | 0.548 | 0.757 | 0.904 | 1979 | 0.548 | 0.757 | 0.904 |
| ps_raycast @ 10 m | 1638 | 8098 | 202.3 | server | 1638 | 0.408 | 0.643 | 0.842 | 1638 | 0.408 | 0.643 | 0.842 |
| ps_raycast @ 12.5 m | 1488 | 8098 | 183.7 | server | 1488 | 0.353 | 0.578 | 0.799 | 1488 | 0.353 | 0.578 | 0.799 |
| ps_raycast @ 15 m | 1396 | 8098 | 172.4 | server | 1396 | 0.343 | 0.542 | 0.774 | 1396 | 0.343 | 0.542 | 0.774 |
| fusion | 1570 | 8098 | 193.9 | raycast | 1570 | 0.304 | 0.436 | 0.780 | 1570 | 0.304 | 0.436 | 0.780 |
| fusion_refit | 1570 | 8098 | 193.9 | raycast | 1570 | 0.327 | 0.443 | 0.785 | 1570 | 0.327 | 0.443 | 0.785 |
| fusion_server | 3030 | 9639 | 314.3 | server | 3030 | 0.761 | 0.868 | 0.950 | 1589 | 0.425 | 0.615 | 0.833 |
| fusion_server+attach | 1899 | 9639 | 197.0 | server | 1899 | 0.493 | 0.689 | 0.878 | 1589 | 0.421 | 0.621 | 0.841 |

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
| fusion_server | cluster of 1 | 1826 | 14 | 0.000 (0) | 0.000 (0) | 14 | 0 |
| fusion_server | cluster of 2 | 248 | 1 | 0.000 (0) | 0.000 (0) | 1 | 0 |
| fusion_server | cluster of 3+ | 956 | 77 | 0.013 (1) | 0.013 (1) | 97 | 1 |
| fusion_server+attach | cluster of 1 | 637 | 5 | 0.000 (0) | 0.000 (0) | 5 | 0 |
| fusion_server+attach | cluster of 2 | 224 | 2 | 0.000 (0) | 0.000 (0) | 2 | 0 |
| fusion_server+attach | cluster of 3+ | 1038 | 85 | 0.012 (1) | 0.012 (1) | 105 | 1 |
| fusion | cluster of 1 | 0 | 0 | n/a | n/a | 0 | 0 |
| fusion | cluster of 2 | 0 | 0 | n/a | n/a | 0 | 0 |
| fusion | cluster of 3+ | 0 | 0 | n/a | n/a | 0 | 0 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `deployed` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval. Median y is the stored detection's y_normalized (0.5 = the horizon). `unclustered` (a placeable label no cluster of that partition holds) is shown only when non-empty.

| partition | bucket | AI labels | median conf | median y | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| deployed | unplaceable | 1428 | 0.79 | n/a | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| deployed | unclustered | 1 | 0.67 | n/a | 0 | n/a | 0 | 0 | 0 |
| deployed | cluster of 1 | 325 | 0.75 | n/a | 11 | 0.900 [0.60, 0.98] | 9 | 1 | 1 |
| deployed | cluster of 2 | 595 | 0.81 | n/a | 18 | 0.812 [0.57, 0.93] | 13 | 3 | 2 |
| deployed | cluster of 3+ | 7177 | 0.87 | n/a | 205 | 0.974 [0.94, 0.99] | 188 | 5 | 12 |
| fusion_server | unplaceable | 1428 | 0.79 | n/a | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server | cluster of 1 | 387 | 0.78 | n/a | 13 | 0.846 [0.58, 0.96] | 11 | 2 | 0 |
| fusion_server | cluster of 2 | 490 | 0.76 | n/a | 14 | 0.818 [0.52, 0.95] | 9 | 2 | 3 |
| fusion_server | cluster of 3+ | 7221 | 0.87 | n/a | 207 | 0.974 [0.94, 0.99] | 190 | 5 | 12 |
| fusion_server+attach | unplaceable | 1428 | 0.79 | n/a | 33 | 1.000 [0.88, 1.00] | 27 | 0 | 6 |
| fusion_server+attach | cluster of 1 | 329 | 0.77 | n/a | 9 | 0.778 [0.45, 0.94] | 7 | 2 | 0 |
| fusion_server+attach | cluster of 2 | 401 | 0.76 | n/a | 13 | 0.900 [0.60, 0.98] | 9 | 1 | 3 |
| fusion_server+attach | cluster of 3+ | 7368 | 0.87 | n/a | 212 | 0.970 [0.94, 0.99] | 194 | 6 | 12 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 1434; attached 1131 (0.79)
- clusters: 3030 (`fusion_server`) -> 1899 (`fusion_server+attach`)
- sanity (not a metric): 33 unplaceable labels are on judged panos; 28 of them attached (24 judged true), 1 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)

## Offline server arm vs this server (`--offline-check`)

Validates the offline mode used for cities without a server: the labels it synthesizes from `results.jsonl` and places with the server's estimator (ps_placement), against the labels this server actually holds.

- (a) placement, 9526 of 9526 AI labels re-placed from the file: 0 labels on 0 panos left out because the position their labels imply (inverting the server's estimator, median per pano) is > 0.5 m from the file's; over the other 9526: median 0.000000 m, p90 0.000000 m, max 0.000000 m, 0 over 0.5 m (all labels: median 0.000000 m, p90 0.000000 m, 0 over 0.5 m, max 0.000000 m)
- (a) placement check: **PASS** (max error over ALL labels <= 0.5 m). Gated on all labels because the moved-pano exclusion above is self-referential: it inverts the estimator being validated, so an error consistent within a pano would be excluded, not failed
- (b) labels: 9526 synthesized at 0.55 (unmasked); 9526 match a live AI label by pano and pixel, 0 do not (soft-deleted, or never sent); 0 live AI labels have no synthesized twin; 1 matched labels are in no deployed cluster (the server has not clustered them) and are left out of the partitions
- (b) regions by nearest street to the offline position: 9525 of 9525 equal the label's live region_id (0 equidistant ties)
- (b) partition: of the 2088 deployed clusters whose labels are all AI and all synthesized (of 2156), 2080 are, label for label, a cluster of the offline `ps @ 7.5 m` (0.996; 15 labels in the others)
- (b+) the same with the 109 live human labels added at their live positions and regions: 2156 of the 2156 deployed clusters whose labels are all present are reproduced (1.000; 0 labels in the others)
