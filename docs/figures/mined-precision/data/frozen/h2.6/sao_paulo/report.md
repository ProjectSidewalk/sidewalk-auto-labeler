## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 22741 panos -> 15409 sites, 1947 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 18 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.300 [0.11, 0.60] (n=10) -> read at <= 15 m: drop this label source; the 95% CI [0.11, 0.60] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.533 [0.30, 0.75] (n=15) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.30, 0.75] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 18 | 3 | 7 | 0 | 5 | 3 | 0 | 0.300 [0.11, 0.60] (n=10) | 0.533 [0.30, 0.75] (n=15) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 9 | 1 | 4 | 0 | 2 | 2 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.429 [0.16, 0.75] (n=7) |
| <= 15 m | 18 | 3 | 7 | 0 | 5 | 3 | 0 | 0.300 [0.11, 0.60] (n=10) | 0.533 [0.30, 0.75] (n=15) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 5 | 1 | 2 | 0 | 1 | 1 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.500 [0.15, 0.85] (n=4) |
| 8-12 m | 7 | 1 | 2 | 0 | 2 | 2 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| 12-18 m | 6 | 1 | 3 | 0 | 2 | 0 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.500 [0.19, 0.81] (n=6) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 9 | 1 | 3 | 0 | 2 | 3 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.500 [0.19, 0.81] (n=6) |
| 4 panos | 5 | 2 | 1 | 0 | 2 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.800 [0.38, 0.96] (n=5) |
| >= 5 panos | 4 | 0 | 3 | 0 | 1 | 0 | 0 | 0.000 [0.00, 0.56] (n=3) | 0.250 [0.05, 0.70] (n=4) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 10 | 1 | 5 | 0 | 3 | 1 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.444 [0.19, 0.73] (n=9) |
| best member < 0.9 | 8 | 2 | 2 | 0 | 2 | 2 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.667 [0.30, 0.90] (n=6) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 437 | 0.04x |
| <= 15 m | 1382 | 0.11x |
