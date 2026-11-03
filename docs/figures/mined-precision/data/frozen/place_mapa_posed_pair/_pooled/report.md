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

placement: mapa_posed_pair -- 127 of 127 candidates placed by the arm, 0 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 1.84 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.487 [0.38, 0.60] (n=80) -> read at <= 15 m: drop this label source; the 95% CI [0.38, 0.60] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.637 [0.55, 0.72] (n=113) -> read at <= 15 m: add the visibility test before mining

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 127 | 39 | 41 | 0 | 33 | 14 | 0 | 0.487 [0.38, 0.60] (n=80) | 0.637 [0.55, 0.72] (n=113) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 53 | 19 | 11 | 0 | 15 | 8 | 0 | 0.633 [0.46, 0.78] (n=30) | 0.756 [0.61, 0.86] (n=45) |
| <= 15 m | 127 | 39 | 41 | 0 | 33 | 14 | 0 | 0.487 [0.38, 0.60] (n=80) | 0.637 [0.55, 0.72] (n=113) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 33 | 14 | 6 | 0 | 5 | 8 | 0 | 0.700 [0.48, 0.85] (n=20) | 0.760 [0.57, 0.89] (n=25) |
| 8-12 m | 38 | 9 | 14 | 0 | 12 | 3 | 0 | 0.391 [0.22, 0.59] (n=23) | 0.600 [0.44, 0.74] (n=35) |
| 12-18 m | 56 | 16 | 21 | 0 | 16 | 3 | 0 | 0.432 [0.29, 0.59] (n=37) | 0.604 [0.47, 0.72] (n=53) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 45 | 17 | 11 | 0 | 14 | 3 | 0 | 0.607 [0.42, 0.76] (n=28) | 0.738 [0.59, 0.85] (n=42) |
| 4 panos | 34 | 12 | 10 | 0 | 9 | 3 | 0 | 0.545 [0.35, 0.73] (n=22) | 0.677 [0.50, 0.81] (n=31) |
| >= 5 panos | 48 | 10 | 20 | 0 | 10 | 8 | 0 | 0.333 [0.19, 0.51] (n=30) | 0.500 [0.35, 0.65] (n=40) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 78 | 21 | 33 | 0 | 16 | 8 | 0 | 0.389 [0.27, 0.52] (n=54) | 0.529 [0.41, 0.64] (n=70) |
| best member < 0.9 | 49 | 18 | 8 | 0 | 17 | 6 | 0 | 0.692 [0.50, 0.83] (n=26) | 0.814 [0.67, 0.90] (n=43) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 3895 | 0.03x |
| <= 15 m | 13458 | 0.11x |
