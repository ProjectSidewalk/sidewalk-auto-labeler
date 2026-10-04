## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 189807 panos -> 62817 sites, 20584 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano/per-rig
excluded: 34 (site, pano) pairs where the pano is a member through a sub-threshold detection only

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
- gainesville: 30458 of 37435 panos took a measured height; 6977 fell back to the 2.6 m constant
- gainesville: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 3112}
- gainesville: 30458 heights read from depth/index.csv; spread definition: measured_planes_only
- sao_paulo: camera height mode `per-pano`
- sao_paulo: 18601 of 30034 panos took a measured height; 11433 fell back to the 2.6 m constant
- sao_paulo: flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- sao_paulo: 18601 heights read from depth/index.csv; spread definition: measured_planes_only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.495 [0.40, 0.59] (n=95) -> read at <= 15 m: drop this label source; the 95% CI [0.40, 0.59] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.625 [0.54, 0.70] (n=128) -> read at <= 15 m: add the visibility test before mining

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 141 | 47 | 48 | 0 | 33 | 13 | 0 | 0.495 [0.40, 0.59] (n=95) | 0.625 [0.54, 0.70] (n=128) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 60 | 23 | 15 | 0 | 16 | 6 | 0 | 0.605 [0.45, 0.74] (n=38) | 0.722 [0.59, 0.82] (n=54) |
| <= 15 m | 141 | 47 | 48 | 0 | 33 | 13 | 0 | 0.495 [0.40, 0.59] (n=95) | 0.625 [0.54, 0.70] (n=128) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 37 | 14 | 9 | 0 | 8 | 6 | 0 | 0.609 [0.41, 0.78] (n=23) | 0.710 [0.53, 0.84] (n=31) |
| 8-12 m | 43 | 15 | 14 | 0 | 11 | 3 | 0 | 0.517 [0.34, 0.69] (n=29) | 0.650 [0.50, 0.78] (n=40) |
| 12-18 m | 61 | 18 | 25 | 0 | 14 | 4 | 0 | 0.419 [0.28, 0.57] (n=43) | 0.561 [0.43, 0.68] (n=57) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 48 | 23 | 12 | 0 | 10 | 3 | 0 | 0.657 [0.49, 0.79] (n=35) | 0.733 [0.59, 0.84] (n=45) |
| 4 panos | 36 | 11 | 12 | 0 | 12 | 1 | 0 | 0.478 [0.29, 0.67] (n=23) | 0.657 [0.49, 0.79] (n=35) |
| >= 5 panos | 57 | 13 | 24 | 0 | 11 | 9 | 0 | 0.351 [0.22, 0.51] (n=37) | 0.500 [0.36, 0.64] (n=48) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 81 | 22 | 36 | 0 | 15 | 8 | 0 | 0.379 [0.27, 0.51] (n=58) | 0.507 [0.39, 0.62] (n=73) |
| best member < 0.9 | 60 | 25 | 12 | 0 | 18 | 5 | 0 | 0.676 [0.51, 0.80] (n=37) | 0.782 [0.66, 0.87] (n=55) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 4658 | 0.04x |
| <= 15 m | 15310 | 0.12x |
