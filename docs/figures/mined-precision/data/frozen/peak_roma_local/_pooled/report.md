## pooled over richmond, paterson, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 101723 panos -> 46004 sites, 10205 with >= 3 operational panos
GT: 499 fully judged panos, 499 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano/per-rig
excluded: 29 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- richmond: camera height mode `per-rig`
- richmond: 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one
- paterson: camera height mode `per-pano`
- paterson: 30325 of 34687 panos took a measured height; 4362 fell back to the 2.6 m constant
- paterson: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 484}
- paterson: 30325 heights read from depth/index.csv; spread definition: measured_planes_only
- gainesville: camera height mode `per-pano`
- gainesville: 30458 of 35204 panos took a measured height; 4746 fell back to the 2.6 m constant
- gainesville: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 3112}
- gainesville: 30458 heights read from depth/index.csv; spread definition: measured_planes_only
- sao_paulo: camera height mode `per-pano`
- sao_paulo: 18601 of 22741 panos took a measured height; 4140 fell back to the 2.6 m constant
- sao_paulo: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- sao_paulo: 18601 heights read from depth/index.csv; spread definition: measured_planes_only

placement: peak_roma_local -- 20 of 115 candidates placed by the arm, 0 fell back and 1 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.22 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it. 94 candidates were NOT EMITTED by the placement's target definition and are in neither denominator.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.667 [0.42, 0.85] (n=15) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.42, 0.85] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.688 [0.44, 0.86] (n=16) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.44, 0.86] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 21 | 10 | 5 | 0 | 1 | 5 | 0 | 0.667 [0.42, 0.85] (n=15) | 0.688 [0.44, 0.86] (n=16) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 11 | 7 | 2 | 0 | 1 | 1 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.800 [0.49, 0.94] (n=10) |
| <= 15 m | 21 | 10 | 5 | 0 | 1 | 5 | 0 | 0.667 [0.42, 0.85] (n=15) | 0.688 [0.44, 0.86] (n=16) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 7 | 5 | 1 | 0 | 0 | 1 | 0 | 0.833 [0.44, 0.97] (n=6) | 0.833 [0.44, 0.97] (n=6) |
| 8-12 m | 9 | 3 | 3 | 0 | 1 | 2 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.571 [0.25, 0.84] (n=7) |
| 12-18 m | 5 | 2 | 1 | 0 | 0 | 2 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.667 [0.21, 0.94] (n=3) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 4 | 2 | 0 | 0 | 1 | 1 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.44, 1.00] (n=3) |
| 4 panos | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.44, 1.00] (n=3) |
| >= 5 panos | 14 | 5 | 5 | 0 | 0 | 4 | 0 | 0.500 [0.24, 0.76] (n=10) | 0.500 [0.24, 0.76] (n=10) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 13 | 4 | 5 | 0 | 0 | 4 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.444 [0.19, 0.73] (n=9) |
| best member < 0.9 | 8 | 6 | 0 | 0 | 1 | 1 | 0 | 1.000 [0.61, 1.00] (n=6) | 1.000 [0.65, 1.00] (n=7) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 2664 | 0.04x |
| <= 15 m | 8308 | 0.13x |
