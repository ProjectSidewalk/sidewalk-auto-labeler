## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 22741 panos -> 16357 sites, 1920 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.2 m
excluded: 13 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.375 [0.18, 0.61] (n=16) -> read at <= 15 m: drop this label source; the 95% CI [0.18, 0.61] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.545 [0.35, 0.73] (n=22) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.35, 0.73] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 25 | 6 | 10 | 0 | 6 | 3 | 0 | 0.375 [0.18, 0.61] (n=16) | 0.545 [0.35, 0.73] (n=22) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 8 | 1 | 4 | 0 | 2 | 1 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.429 [0.16, 0.75] (n=7) |
| <= 15 m | 25 | 6 | 10 | 0 | 6 | 3 | 0 | 0.375 [0.18, 0.61] (n=16) | 0.545 [0.35, 0.73] (n=22) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 4 | 1 | 1 | 0 | 1 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.667 [0.21, 0.94] (n=3) |
| 8-12 m | 7 | 0 | 5 | 0 | 1 | 1 | 0 | 0.000 [0.00, 0.43] (n=5) | 0.167 [0.03, 0.56] (n=6) |
| 12-18 m | 14 | 5 | 4 | 0 | 4 | 1 | 0 | 0.556 [0.27, 0.81] (n=9) | 0.692 [0.42, 0.87] (n=13) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 3 | 3 | 0 | 3 | 2 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.667 [0.35, 0.88] (n=9) |
| 4 panos | 11 | 3 | 5 | 0 | 2 | 1 | 0 | 0.375 [0.14, 0.69] (n=8) | 0.500 [0.24, 0.76] (n=10) |
| >= 5 panos | 3 | 0 | 2 | 0 | 1 | 0 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.333 [0.06, 0.79] (n=3) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 15 | 4 | 6 | 0 | 4 | 1 | 0 | 0.400 [0.17, 0.69] (n=10) | 0.571 [0.33, 0.79] (n=14) |
| best member < 0.9 | 10 | 2 | 4 | 0 | 2 | 2 | 0 | 0.333 [0.10, 0.70] (n=6) | 0.500 [0.22, 0.78] (n=8) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 410 | 0.03x |
| <= 15 m | 1480 | 0.12x |
