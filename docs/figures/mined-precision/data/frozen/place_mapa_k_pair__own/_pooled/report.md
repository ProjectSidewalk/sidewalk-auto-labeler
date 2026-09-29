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

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 47).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.287 [0.21, 0.38] (n=108) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.319 [0.24, 0.41] (n=113) -> read at <= 15 m: drop this label source

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 127 | 31 | 77 | 1 | 5 | 14 | 0 | 0.287 [0.21, 0.38] (n=108) | 0.319 [0.24, 0.41] (n=113) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 53 | 16 | 25 | 1 | 4 | 8 | 0 | 0.390 [0.26, 0.54] (n=41) | 0.444 [0.31, 0.59] (n=45) |
| <= 15 m | 127 | 31 | 77 | 1 | 5 | 14 | 0 | 0.287 [0.21, 0.38] (n=108) | 0.319 [0.24, 0.41] (n=113) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 33 | 10 | 12 | 0 | 3 | 8 | 0 | 0.455 [0.27, 0.65] (n=22) | 0.520 [0.33, 0.70] (n=25) |
| 8-12 m | 38 | 10 | 24 | 1 | 1 | 3 | 0 | 0.294 [0.17, 0.46] (n=34) | 0.314 [0.19, 0.48] (n=35) |
| 12-18 m | 56 | 11 | 41 | 0 | 1 | 3 | 0 | 0.212 [0.12, 0.34] (n=52) | 0.226 [0.13, 0.36] (n=53) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 45 | 12 | 30 | 1 | 0 | 3 | 0 | 0.286 [0.17, 0.44] (n=42) | 0.286 [0.17, 0.44] (n=42) |
| 4 panos | 34 | 10 | 19 | 0 | 2 | 3 | 0 | 0.345 [0.20, 0.53] (n=29) | 0.387 [0.24, 0.56] (n=31) |
| >= 5 panos | 48 | 9 | 28 | 0 | 3 | 8 | 0 | 0.243 [0.13, 0.40] (n=37) | 0.300 [0.18, 0.45] (n=40) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 78 | 17 | 51 | 1 | 2 | 8 | 0 | 0.250 [0.16, 0.36] (n=68) | 0.271 [0.18, 0.39] (n=70) |
| best member < 0.9 | 49 | 14 | 26 | 0 | 3 | 6 | 0 | 0.350 [0.22, 0.50] (n=40) | 0.395 [0.26, 0.54] (n=43) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 3895 | 0.03x |
| <= 15 m | 13458 | 0.11x |
