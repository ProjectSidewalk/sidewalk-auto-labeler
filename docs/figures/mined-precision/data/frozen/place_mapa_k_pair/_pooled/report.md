## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 180283 panos -> 60209 sites, 19947 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano/per-rig
excluded: 29 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- richmond: camera height mode `per-rig`
- richmond: 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one
- paterson: camera height mode `per-pano`
- paterson: 30325 of 34687 panos took a measured height; 4362 fell back to the 2.6 m constant
- paterson: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 484}
- paterson: 30325 heights read from depth/index.csv; spread definition: measured_planes_only
- bend: camera height mode `per-pano`
- bend: 65866 of 78560 panos took a measured height; 12694 fell back to the 2.6 m constant
- bend: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 969}
- bend: 65866 heights read from depth/index.csv; spread definition: measured_planes_only
- gainesville: camera height mode `per-pano`
- gainesville: 30458 of 35204 panos took a measured height; 4746 fell back to the 2.6 m constant
- gainesville: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 3112}
- gainesville: 30458 heights read from depth/index.csv; spread definition: measured_planes_only
- sao_paulo: camera height mode `per-pano`
- sao_paulo: 18601 of 22741 panos took a measured height; 4140 fell back to the 2.6 m constant
- sao_paulo: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- sao_paulo: 18601 heights read from depth/index.csv; spread definition: measured_planes_only

placement: mapa_k_pair -- 121 of 127 candidates placed by the arm, 1 fell back and 5 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.18 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.565 [0.45, 0.68] (n=69) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.45, 0.68] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.735 [0.65, 0.81] (n=113) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.65, 0.81] spans the visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 127 | 39 | 30 | 1 | 44 | 14 | 0 | 0.565 [0.45, 0.68] (n=69) | 0.735 [0.65, 0.81] (n=113) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 53 | 20 | 10 | 1 | 15 | 8 | 0 | 0.667 [0.49, 0.81] (n=30) | 0.778 [0.64, 0.87] (n=45) |
| <= 15 m | 127 | 39 | 30 | 1 | 44 | 14 | 0 | 0.565 [0.45, 0.68] (n=69) | 0.735 [0.65, 0.81] (n=113) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 33 | 14 | 5 | 0 | 6 | 8 | 0 | 0.737 [0.51, 0.88] (n=19) | 0.800 [0.61, 0.91] (n=25) |
| 8-12 m | 38 | 10 | 11 | 1 | 14 | 3 | 0 | 0.476 [0.28, 0.68] (n=21) | 0.686 [0.52, 0.81] (n=35) |
| 12-18 m | 56 | 15 | 14 | 0 | 24 | 3 | 0 | 0.517 [0.34, 0.69] (n=29) | 0.736 [0.60, 0.84] (n=53) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 45 | 19 | 11 | 1 | 12 | 3 | 0 | 0.633 [0.46, 0.78] (n=30) | 0.738 [0.59, 0.85] (n=42) |
| 4 panos | 34 | 10 | 6 | 0 | 15 | 3 | 0 | 0.625 [0.39, 0.82] (n=16) | 0.806 [0.64, 0.91] (n=31) |
| >= 5 panos | 48 | 10 | 13 | 0 | 17 | 8 | 0 | 0.435 [0.26, 0.63] (n=23) | 0.675 [0.52, 0.80] (n=40) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 78 | 22 | 23 | 1 | 25 | 8 | 0 | 0.489 [0.35, 0.63] (n=45) | 0.671 [0.56, 0.77] (n=70) |
| best member < 0.9 | 49 | 17 | 7 | 0 | 19 | 6 | 0 | 0.708 [0.51, 0.85] (n=24) | 0.837 [0.70, 0.92] (n=43) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 3895 | 0.03x |
| <= 15 m | 13458 | 0.11x |
