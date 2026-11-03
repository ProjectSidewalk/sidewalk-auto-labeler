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

placement: peak_mapa_k_pair -- 23 of 115 candidates placed by the arm, 0 fell back and 1 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.22 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it. 91 candidates were NOT EMITTED by the placement's target definition and are in neither denominator.

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 4).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.400 [0.22, 0.61] (n=20) -> read at <= 15 m: drop this label source; the 95% CI [0.22, 0.61] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.400 [0.22, 0.61] (n=20) -> read at <= 15 m: drop this label source; the 95% CI [0.22, 0.61] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 24 | 8 | 12 | 0 | 0 | 4 | 0 | 0.400 [0.22, 0.61] (n=20) | 0.400 [0.22, 0.61] (n=20) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 12 | 6 | 5 | 0 | 0 | 1 | 0 | 0.545 [0.28, 0.79] (n=11) | 0.545 [0.28, 0.79] (n=11) |
| <= 15 m | 24 | 8 | 12 | 0 | 0 | 4 | 0 | 0.400 [0.22, 0.61] (n=20) | 0.400 [0.22, 0.61] (n=20) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 8 | 5 | 2 | 0 | 0 | 1 | 0 | 0.714 [0.36, 0.92] (n=7) | 0.714 [0.36, 0.92] (n=7) |
| 8-12 m | 8 | 1 | 6 | 0 | 0 | 1 | 0 | 0.143 [0.03, 0.51] (n=7) | 0.143 [0.03, 0.51] (n=7) |
| 12-18 m | 8 | 2 | 4 | 0 | 0 | 2 | 0 | 0.333 [0.10, 0.70] (n=6) | 0.333 [0.10, 0.70] (n=6) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 5 | 2 | 3 | 0 | 0 | 0 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.400 [0.12, 0.77] (n=5) |
| 4 panos | 4 | 3 | 1 | 0 | 0 | 0 | 0 | 0.750 [0.30, 0.95] (n=4) | 0.750 [0.30, 0.95] (n=4) |
| >= 5 panos | 15 | 3 | 8 | 0 | 0 | 4 | 0 | 0.273 [0.10, 0.57] (n=11) | 0.273 [0.10, 0.57] (n=11) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 14 | 3 | 7 | 0 | 0 | 4 | 0 | 0.300 [0.11, 0.60] (n=10) | 0.300 [0.11, 0.60] (n=10) |
| best member < 0.9 | 10 | 5 | 5 | 0 | 0 | 0 | 0 | 0.500 [0.24, 0.76] (n=10) | 0.500 [0.24, 0.76] (n=10) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 2664 | 0.04x |
| <= 15 m | 8308 | 0.13x |
