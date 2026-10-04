## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

placement: peak_roma_local -- 16 of 59 candidates placed by the arm, 0 fell back and 1 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.17 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it. 42 candidates were NOT EMITTED by the placement's target definition and are in neither denominator.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.615 [0.36, 0.82] (n=13) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.36, 0.82] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.615 [0.36, 0.82] (n=13) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.36, 0.82] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 17 | 8 | 5 | 0 | 0 | 4 | 0 | 0.615 [0.36, 0.82] (n=13) | 0.615 [0.36, 0.82] (n=13) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 10 | 7 | 2 | 0 | 0 | 1 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.778 [0.45, 0.94] (n=9) |
| <= 15 m | 17 | 8 | 5 | 0 | 0 | 4 | 0 | 0.615 [0.36, 0.82] (n=13) | 0.615 [0.36, 0.82] (n=13) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 7 | 5 | 1 | 0 | 0 | 1 | 0 | 0.833 [0.44, 0.97] (n=6) | 0.833 [0.44, 0.97] (n=6) |
| 8-12 m | 6 | 2 | 3 | 0 | 0 | 1 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.400 [0.12, 0.77] (n=5) |
| 12-18 m | 4 | 1 | 1 | 0 | 0 | 2 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.500 [0.09, 0.91] (n=2) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.21, 1.00] (n=1) | 1.000 [0.21, 1.00] (n=1) |
| 4 panos | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.44, 1.00] (n=3) |
| >= 5 panos | 13 | 4 | 5 | 0 | 0 | 4 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.444 [0.19, 0.73] (n=9) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 12 | 3 | 5 | 0 | 0 | 4 | 0 | 0.375 [0.14, 0.69] (n=8) | 0.375 [0.14, 0.69] (n=8) |
| best member < 0.9 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.57, 1.00] (n=5) | 1.000 [0.57, 1.00] (n=5) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
