# laurens: PS label clustering vs RampNet GT

labels: 1803 CurbRamp on the server, 1575 map to stored detections (AI), 228 do not (human); 671 server clusters over 1708 labels
scorer 106.2; results `results.raw.jsonl` sha256 `a7da290de5b0576cf5f54ab5bc1615a52fba766c47da3acb97e57c272860340a`; benchmark split `laurens_mapillary`
inputs: streets sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`; verdicts sha256 `0fb67c5d6da90827049c30f17ce5150ff4b24c5ae605a3f047b0b1d6255ad49e`
raycast camera height 2.6 m; fusion arm at --min-confidence 0.3
GT: 94 judged panos -> 240 placeable points -> 238 ramps (2 cross-pano merges), 238 in the recall pool; raycast placed 3370 of 5727 detections (drops {'below_floor': 0, 'on_rig': 2199, 'horizon': 7, 'out_of_range': 151})

## Data provenance

- `raw_labels.geojson`: 1803 features, sha256 `4fea373b3a7ebe2755e9396a6b73c6ab35642cd2cbaa52759fbb607e237cdb65`, 2026-09-28T17:09:50+00:00 (0.2 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- `clusters.geojson`: 671 features, sha256 `329706bb7d21796497ee11c40162339aa63e5c66957d197a8a7d23bf82b5b838`, 2026-09-28T17:09:51+00:00 (0.2 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&includeRawLabels=true&filetype=geojson
- labels by account: 51b0b927-3c8a-45b2-93de-bd878d1e5cf4 (AI) 1575, 549187e0-82c9-4014-a48d-31f18083d575 224, 18b26a38-24ab-402d-a64e-158fc0bb8a8a 4
- 0 labels dropped before clustering (null lng or lng > 360), matching label_clustering.clean_label_data
- 0 ambiguous pixel keys in results.raw.jsonl (two stored detections round to one pixel; those keys are left unmapped)
- 0 server labels share a pixel with another label and so map to the same stored detection (a re-submitted campaign does this)
- `streets.geojson`: 169 features, sha256 `ced8bab31b6335b8c700d6cea5f014bbc851598a66ef5f6f765dff828d493ef4`, 2026-09-28T17:09:49+00:00 (0.2 days old at run time), from https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson; 168 open streets kept (the server snaps to open streets only)
- PS partitions are blocked (single-linkage components at the widest threshold + 0.5 m): 115 blocks, largest 81 labels

## Validation checks

- ps_repro reproduces deployed: 671/671 clusters identical (0 labels in clusters that differ); script threshold 0.0075 km
- vectorized PS distance reproduces the script: 671/671 clusters identical (0 labels differ)
- every label that maps to a stored detection belongs to one account (51b0b927-3c8a-45b2-93de-bd878d1e5cf4); 0 of that account's labels did not map (should be 0)
- fusion arm vs ps_* arms cover the same labels: 1540 fusion members vs 1540 placeable server labels; 0 of 1540 placeable operational detections have no label on the server (should be 0; they are excluded from the scatter below)
- fusion_refit at 2.6 m vs runs/laurens/fusion_eval/report.md (published in the 2.6 m frame): precision 0.896, recall (union) 0.782, dual 24/27/1
- fusion_server input: 1575 AI + 228 human labels on 769 panos (759 positioned from the run's pano block, 10 inverted from their labels, 0 unplaceable and left out); 39 labels the raycast cannot place (range cap, horizon) are singleton clusters; 416 of its 582 clusters with AI members are, member for member, a cluster of the `fusion` arm
- camera-position inversion (for panos only humans labeled) vs the run's position, over 685 panos in both: median 0.01 m, p90 0.33 m
- same-pano pairs inside one cluster (must be 0 under the cannot-link): 0 in every arm

## Arms (match radius 5 m, GT merge 2.5 m)

| arm | clusters | placed | labels | labels/cluster | precision (TP/FP) | coverage | no cluster | recall (union) | self/other/unmatched | frag 3 m (extra) | frag 5 m (extra) | dual both/one/neither | coherence med / p90 / >5 m |
|---|---:|---:|---:|---:|---|---|---:|---|---|---|---|---|---|
| deployed | 671 | 636 | 1708 | 2.55 | 0.960 (97/4) | 0.807 (192/238) | 3 | 0.819 | 97/98/43 | 0.05 (9) | 0.28 (58) | 32/20/0 | 1.75 / 3.71 / 3 |
| ps_repro | 671 | 636 | 1708 | 2.55 | 0.960 (97/4) | 0.807 (192/238) | 3 | 0.819 | 97/98/43 | 0.05 (9) | 0.28 (58) | 32/20/0 | 1.75 / 3.71 / 3 |
| ps @ 2.5 m | 1159 | 1146 | 1575 | 1.36 | 0.898 (97/11) | 0.887 (211/238) | 0 | 0.887 | 97/114/27 | 0.33 (87) | 0.70 (301) | 42/10/0 | 0.00 / 2.11 / 0 |
| ps @ 5 m | 817 | 811 | 1575 | 1.93 | 0.898 (97/11) | 0.845 (201/238) | 1 | 0.849 | 97/105/36 | 0.09 (21) | 0.44 (115) | 38/14/0 | 1.33 / 2.94 / 2 |
| ps @ 7.5 m | 657 | 654 | 1575 | 2.40 | 0.898 (97/11) | 0.794 (189/238) | 1 | 0.798 | 97/93/48 | 0.03 (6) | 0.26 (51) | 32/17/3 | 1.80 / 3.78 / 3 |
| ps @ 10 m | 568 | 566 | 1575 | 2.77 | 0.898 (97/11) | 0.748 (178/238) | 4 | 0.765 | 97/85/56 | 0.03 (5) | 0.15 (27) | 26/22/4 | 2.09 / 4.42 / 5 |
| ps @ 12.5 m | 503 | 502 | 1575 | 3.13 | 0.897 (96/11) | 0.702 (167/238) | 11 | 0.748 | 97/81/60 | 0.01 (2) | 0.10 (16) | 22/26/4 | 2.63 / 5.20 / 10 |
| ps @ 15 m | 465 | 464 | 1575 | 3.39 | 0.897 (96/11) | 0.660 (157/238) | 17 | 0.731 | 97/77/64 | 0.02 (3) | 0.08 (13) | 19/29/4 | 2.77 / 6.01 / 15 |
| ps_citywide @ 7.5 m | 657 | 654 | 1575 | 2.40 | 0.898 (97/11) | 0.794 (189/238) | 1 | 0.798 | 97/93/48 | 0.03 (6) | 0.26 (51) | 32/17/3 | 1.80 / 3.78 / 3 |
| ps_placeable @ 2.5 m | 1140 | 1140 | 1540 | 1.35 | 0.898 (97/11) | 0.887 (211/238) | 0 | 0.887 | 97/114/27 | 0.33 (85) | 0.69 (292) | 42/10/0 | 0.00 / 2.11 / 0 |
| ps_placeable @ 5 m | 807 | 807 | 1540 | 1.91 | 0.898 (97/11) | 0.845 (201/238) | 1 | 0.849 | 97/105/36 | 0.09 (20) | 0.43 (113) | 38/14/0 | 1.33 / 2.94 / 2 |
| ps_placeable @ 7.5 m | 644 | 644 | 1540 | 2.39 | 0.898 (97/11) | 0.786 (187/238) | 1 | 0.790 | 97/91/50 | 0.03 (5) | 0.24 (45) | 30/19/3 | 1.85 / 3.71 / 3 |
| ps_placeable @ 10 m | 566 | 566 | 1540 | 2.72 | 0.898 (97/11) | 0.748 (178/238) | 4 | 0.765 | 97/85/56 | 0.03 (6) | 0.16 (28) | 26/22/4 | 2.10 / 4.42 / 5 |
| ps_placeable @ 12.5 m | 503 | 503 | 1540 | 3.06 | 0.897 (96/11) | 0.697 (166/238) | 12 | 0.748 | 97/81/60 | 0.02 (3) | 0.10 (16) | 22/26/4 | 2.63 / 5.41 / 11 |
| ps_placeable @ 15 m | 463 | 463 | 1540 | 3.33 | 0.897 (96/11) | 0.651 (155/238) | 18 | 0.727 | 97/76/65 | 0.03 (4) | 0.08 (13) | 19/29/4 | 2.83 / 6.01 / 16 |
| ps_raycast @ 2.5 m | 1120 | 1120 | 1540 | 1.38 | 0.898 (97/11) | 0.895 (213/238) | 0 | 0.895 | 97/116/25 | 0.26 (67) | 0.68 (278) | 42/10/0 | 0.00 / 1.04 / 0 |
| ps_raycast @ 5 m | 790 | 790 | 1540 | 1.95 | 0.898 (97/11) | 0.849 (202/238) | 1 | 0.853 | 97/106/35 | 0.05 (11) | 0.35 (92) | 37/15/0 | 1.14 / 2.26 / 0 |
| ps_raycast @ 7.5 m | 636 | 636 | 1540 | 2.42 | 0.898 (97/11) | 0.794 (189/238) | 2 | 0.803 | 97/94/47 | 0.02 (3) | 0.18 (35) | 30/20/2 | 1.77 / 3.06 / 0 |
| ps_raycast @ 10 m | 547 | 547 | 1540 | 2.82 | 0.898 (97/11) | 0.735 (175/238) | 6 | 0.761 | 97/84/57 | 0.02 (3) | 0.11 (20) | 24/23/5 | 2.00 / 4.00 / 5 |
| ps_raycast @ 12.5 m | 484 | 484 | 1540 | 3.18 | 0.898 (97/11) | 0.685 (163/238) | 15 | 0.748 | 97/81/60 | 0.01 (1) | 0.08 (13) | 21/26/5 | 2.51 / 5.63 / 15 |
| ps_raycast @ 15 m | 437 | 437 | 1540 | 3.52 | 0.897 (96/11) | 0.613 (146/238) | 22 | 0.706 | 97/71/70 | 0.01 (1) | 0.08 (12) | 14/33/5 | 2.97 / 5.88 / 25 |
| fusion | 536 | 536 | 1540 | 2.87 | 0.896 (95/11) | 0.735 (175/238) | 11 | 0.782 | 97/89/52 | 0.03 (6) | 0.11 (19) | 24/25/3 | 2.48 / 5.20 / 11 |
| fusion_refit | 536 | 536 | 1540 | 2.87 | 0.896 (95/11) | 0.735 (175/238) | 11 | 0.782 | 97/89/52 | 0.02 (4) | 0.10 (17) | 24/27/1 | 2.42 / 5.47 / 12 |
| fusion_server | 664 | 582 | 1803 | 2.72 | 0.896 (95/11) | 0.769 (183/238) | 10 | 0.811 | 97/96/45 | 0.05 (9) | 0.20 (40) | 28/22/2 | 2.76 / 5.03 / 10 |
| fusion_server+attach | 632 | 582 | 1803 | 2.85 | 0.896 (95/11) | 0.769 (183/238) | 10 | 0.811 | 97/96/45 | 0.05 (9) | 0.20 (40) | 28/22/2 | 2.76 / 5.03 / 10 |

**coverage** = pool GT ramps with a cluster of this arm within the match radius, matched one-to-one — the metric RQ2a asks for, and the only recall-shaped one that responds to the partition. **no cluster** = ramps counted as recalled by the union metric although no cluster is within the radius (`eval_sites`' `self_detected_without_site`). **recall (union)** = `eval_sites`' definition, which counts a self-detected ramp as recovered whether or not any cluster landed on it; 97 of this run's 238 pool ramps are self-detected, so 41% of it is constant across arms and it is kept only to tie back to `fusion_eval/report.md`. **frag** = share of covered GT ramps with at least one extra cluster within r that is not the one-to-one match of any GT ramp (total extras in parentheses). **coherence** = distance from a self-detected GT ramp to the centroid of the cluster holding that label.

## Deployed clusters, descriptive

- cluster size histogram (labels -> clusters): 1: 211, 2: 204, 3: 107, 4: 64, 5: 38, 6: 27, 7: 9, 8: 7, 9: 2, 10: 1, 13: 1
- server centroid vs raycast centroid, same members (n=636): median 1.0 m, p90 2.6 m
- per-label server lat/lng vs labeler raycast (n=1540): median 1.1 m, p90 4.7 m, over 5 m 0.02

## Same-ramp scatter (mechanism)

Largest pairwise member distance over 235 fusion sites with >= 3 placeable members (a complete-linkage cut at t keeps the group only if this is <= t):

| positions | median | p90 | share > 7.5 m | share > 10 m | share > 15 m |
|---|---:|---:|---:|---:|---:|
| server | 8.9 m | 12.9 m | 0.73 | 0.34 | 0.03 |
| raycast | 7.7 m | 10.8 m | 0.53 | 0.15 | 0.00 |

- self-detections whose cluster could not be located (should be 0): 0 in every arm

## Match-radius sweep (deployed vs fusion)

| radius m | deployed coverage | deployed frag 5 m | deployed dual both | fusion coverage | fusion frag 5 m | fusion dual both |
|---:|---|---|---|---|---|---|
| 2.5 | 0.517 | 44/123 | 10/52 | 0.408 | 17/97 | 6/52 |
| 5 | 0.807 | 54/192 | 32/52 | 0.735 | 19/175 | 24/52 |
| 7.5 | 0.878 | 52/209 | 41/52 | 0.857 | 16/204 | 34/52 |
| 10 | 0.929 | 50/221 | 46/52 | 0.878 | 16/209 | 35/52 |

## Precision by cluster size

Is a small cluster a false positive? Each AI label is bucketed by the size (labels) of the cluster holding it, or as `unplaceable` when the raycast cannot place it (beyond the range cap, or at/above the horizon); fusion cannot associate those, so they are the singletons of `fusion_server`, and the same bucket is split out of `deployed` for comparison. Precision is T / (T + F) over labels on judged panos (RampNet verdicts, benchmark tier), with a Wilson 95% interval.

| partition | bucket | AI labels | median conf | judged | precision [95% CI] | T | F | neither |
|---|---|---:|---:|---:|---|---:|---:|---:|
| deployed | unplaceable | 35 | 0.46 | 0 | n/a | 0 | 0 | 0 |
| deployed | cluster of 1 | 274 | 0.44 | 23 | 0.500 [0.31, 0.69] | 11 | 11 | 1 |
| deployed | cluster of 2 | 355 | 0.53 | 20 | 1.000 [0.83, 1.00] | 19 | 0 | 1 |
| deployed | cluster of 3+ | 911 | 0.54 | 68 | 1.000 [0.95, 1.00] | 67 | 0 | 1 |
| fusion_server | unplaceable | 35 | 0.46 | 0 | n/a | 0 | 0 | 0 |
| fusion_server | cluster of 1 | 181 | 0.42 | 13 | 0.538 [0.29, 0.77] | 7 | 6 | 0 |
| fusion_server | cluster of 2 | 243 | 0.46 | 17 | 0.765 [0.53, 0.90] | 13 | 4 | 0 |
| fusion_server | cluster of 3+ | 1116 | 0.56 | 81 | 0.987 [0.93, 1.00] | 77 | 1 | 3 |
| fusion_server+attach | unplaceable | 35 | 0.46 | 0 | n/a | 0 | 0 | 0 |
| fusion_server+attach | cluster of 1 | 180 | 0.42 | 13 | 0.538 [0.29, 0.77] | 7 | 6 | 0 |
| fusion_server+attach | cluster of 2 | 237 | 0.46 | 16 | 0.750 [0.51, 0.90] | 12 | 4 | 0 |
| fusion_server+attach | cluster of 3+ | 1123 | 0.55 | 82 | 0.987 [0.93, 1.00] | 78 | 1 | 3 |

## Unplaceable labels: attach by bearing (`fusion_server+attach`)

One rule, fixed before any result and not tuned on GT (issue #106): a label the raycast cannot place joins the placed `fusion_server` site nearest along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no label from the same pano; it does not move the site. Otherwise it stays a singleton.

- unplaceable labels: 39; attached 32 (0.82)
- clusters: 664 (`fusion_server`) -> 632 (`fusion_server+attach`)
- sanity (not a metric): 0 unplaceable labels are on judged panos; 0 of them attached (0 judged true), 0 to a site holding a verdict-true member (de-clustered benchmark panos rarely see each other, so most sites hold no judged member at all)

## Offline server arm vs this server (`--offline-check`)

Validates the offline mode used for cities without a server: the labels it synthesizes from `results.raw.jsonl` and places with the server's estimator (ps_placement), against the labels this server actually holds.

- (a) placement, 1575 of 1575 AI labels re-placed from the file: 0 labels on 0 panos left out because the position their labels imply (inverting the server's estimator, median per pano) is > 0.5 m from the file's; over the other 1575: median 0.000000 m, p90 0.000000 m, max 0.000000 m, 0 over 0.5 m (all labels: median 0.000000 m, p90 0.000000 m, 0 over 0.5 m, max 0.000000 m)
- (a) placement check: **PASS** (max error over ALL labels <= 0.5 m). Gated on all labels because the moved-pano exclusion above is self-referential: it inverts the estimator being validated, so an error consistent within a pano would be excluded, not failed
- (b) labels: 1733 synthesized at 0.3 (unmasked); 1575 match a live AI label by pano and pixel, 158 do not (soft-deleted, or never sent); 0 live AI labels have no synthesized twin; 84 matched labels are in no deployed cluster (the server has not clustered them) and are left out of the partitions
- (b) regions by nearest street to the offline position: 1491 of 1491 equal the label's live region_id (0 equidistant ties)
- (b) partition: of the 508 deployed clusters whose labels are all AI and all synthesized (of 671), 480 are, label for label, a cluster of the offline `ps @ 7.5 m` (0.945; 71 labels in the others)
- (b+) the same with the 217 live human labels added at their live positions and regions: 671 of the 671 deployed clusters whose labels are all present are reproduced (1.000; 0 labels in the others)
