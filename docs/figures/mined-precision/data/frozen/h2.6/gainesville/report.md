## gainesville: precision of mined positives (RampNet#158 step 1)

run: 35204 panos -> 15707 sites, 1859 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 8 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.133 [0.04, 0.38] (n=15) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.350 [0.18, 0.57] (n=20) -> read at <= 15 m: drop this label source; the 95% CI [0.18, 0.57] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 20 | 2 | 13 | 0 | 5 | 0 | 0 | 0.133 [0.04, 0.38] (n=15) | 0.350 [0.18, 0.57] (n=20) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 7 | 1 | 4 | 0 | 2 | 0 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.429 [0.16, 0.75] (n=7) |
| <= 15 m | 20 | 2 | 13 | 0 | 5 | 0 | 0 | 0.133 [0.04, 0.38] (n=15) | 0.350 [0.18, 0.57] (n=20) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 4 | 1 | 1 | 0 | 2 | 0 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.750 [0.30, 0.95] (n=4) |
| 8-12 m | 6 | 0 | 4 | 0 | 2 | 0 | 0 | 0.000 [0.00, 0.49] (n=4) | 0.333 [0.10, 0.70] (n=6) |
| 12-18 m | 10 | 1 | 8 | 0 | 1 | 0 | 0 | 0.111 [0.02, 0.44] (n=9) | 0.200 [0.06, 0.51] (n=10) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 15 | 1 | 10 | 0 | 4 | 0 | 0 | 0.091 [0.02, 0.38] (n=11) | 0.333 [0.15, 0.58] (n=15) |
| 4 panos | 4 | 1 | 2 | 0 | 1 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.500 [0.15, 0.85] (n=4) |
| >= 5 panos | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.79] (n=1) | 0.000 [0.00, 0.79] (n=1) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 11 | 1 | 8 | 0 | 2 | 0 | 0 | 0.111 [0.02, 0.44] (n=9) | 0.273 [0.10, 0.57] (n=11) |
| best member < 0.9 | 9 | 1 | 5 | 0 | 3 | 0 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.444 [0.19, 0.73] (n=9) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 499 | 0.04x |
| <= 15 m | 1652 | 0.12x |
