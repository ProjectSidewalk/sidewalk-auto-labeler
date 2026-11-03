## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 22741 panos -> 16381 sites, 1921 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 13 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 18601 of 22741 panos took a measured height; 4140 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- 18601 heights read from depth/index.csv; spread definition: measured_planes_only

placement: roma -- 20 of 21 candidates placed by the arm, 1 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.65 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.455 [0.21, 0.72] (n=11) -> read at <= 15 m: drop this label source; the 95% CI [0.21, 0.72] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.667 [0.44, 0.84] (n=18) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.44, 0.84] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 21 | 5 | 6 | 0 | 7 | 3 | 0 | 0.455 [0.21, 0.72] (n=11) | 0.667 [0.44, 0.84] (n=18) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 7 | 1 | 3 | 0 | 2 | 1 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.500 [0.19, 0.81] (n=6) |
| <= 15 m | 21 | 5 | 6 | 0 | 7 | 3 | 0 | 0.455 [0.21, 0.72] (n=11) | 0.667 [0.44, 0.84] (n=18) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 1 | 1 | 0 | 0 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.500 [0.09, 0.91] (n=2) |
| 8-12 m | 10 | 2 | 4 | 0 | 3 | 1 | 0 | 0.333 [0.10, 0.70] (n=6) | 0.556 [0.27, 0.81] (n=9) |
| 12-18 m | 8 | 2 | 1 | 0 | 4 | 1 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.857 [0.49, 0.97] (n=7) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 12 | 3 | 3 | 0 | 4 | 2 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.700 [0.40, 0.89] (n=10) |
| 4 panos | 5 | 1 | 2 | 0 | 2 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| >= 5 panos | 4 | 1 | 1 | 0 | 1 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.667 [0.21, 0.94] (n=3) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 12 | 2 | 5 | 0 | 4 | 1 | 0 | 0.286 [0.08, 0.64] (n=7) | 0.545 [0.28, 0.79] (n=11) |
| best member < 0.9 | 9 | 3 | 1 | 0 | 3 | 2 | 0 | 0.750 [0.30, 0.95] (n=4) | 0.857 [0.49, 0.97] (n=7) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 417 | 0.03x |
| <= 15 m | 1427 | 0.12x |
