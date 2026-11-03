## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

placement: mapa_posed_pair -- 59 of 59 candidates placed by the arm, 0 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.30 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.342 [0.21, 0.50] (n=38) -> read at <= 15 m: drop this label source; the 95% CI [0.21, 0.50] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.500 [0.37, 0.63] (n=50) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.37, 0.63] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 59 | 13 | 25 | 0 | 12 | 9 | 0 | 0.342 [0.21, 0.50] (n=38) | 0.500 [0.37, 0.63] (n=50) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 28 | 10 | 6 | 0 | 5 | 7 | 0 | 0.625 [0.39, 0.82] (n=16) | 0.714 [0.50, 0.86] (n=21) |
| <= 15 m | 59 | 13 | 25 | 0 | 12 | 9 | 0 | 0.342 [0.21, 0.50] (n=38) | 0.500 [0.37, 0.63] (n=50) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 20 | 7 | 3 | 0 | 3 | 7 | 0 | 0.700 [0.40, 0.89] (n=10) | 0.769 [0.50, 0.92] (n=13) |
| 8-12 m | 16 | 4 | 9 | 0 | 2 | 1 | 0 | 0.308 [0.13, 0.58] (n=13) | 0.400 [0.20, 0.64] (n=15) |
| 12-18 m | 23 | 2 | 13 | 0 | 7 | 1 | 0 | 0.133 [0.04, 0.38] (n=15) | 0.409 [0.23, 0.61] (n=22) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 3 | 5 | 0 | 2 | 1 | 0 | 0.375 [0.14, 0.69] (n=8) | 0.500 [0.24, 0.76] (n=10) |
| 4 panos | 11 | 4 | 3 | 0 | 2 | 2 | 0 | 0.571 [0.25, 0.84] (n=7) | 0.667 [0.35, 0.88] (n=9) |
| >= 5 panos | 37 | 6 | 17 | 0 | 8 | 6 | 0 | 0.261 [0.13, 0.46] (n=23) | 0.452 [0.29, 0.62] (n=31) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 44 | 6 | 23 | 0 | 8 | 7 | 0 | 0.207 [0.10, 0.38] (n=29) | 0.378 [0.24, 0.54] (n=37) |
| best member < 0.9 | 15 | 7 | 2 | 0 | 4 | 2 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.846 [0.58, 0.96] (n=13) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
