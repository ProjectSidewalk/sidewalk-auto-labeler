## gainesville: precision of mined positives (RampNet#158 step 1)

run: 37435 panos -> 16411 sites, 2057 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 9 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.125 [0.03, 0.36] (n=16) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.333 [0.17, 0.55] (n=21) -> read at <= 15 m: drop this label source; the 95% CI [0.17, 0.55] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 21 | 2 | 14 | 0 | 5 | 0 | 0 | 0.125 [0.03, 0.36] (n=16) | 0.333 [0.17, 0.55] (n=21) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 8 | 1 | 5 | 0 | 2 | 0 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.375 [0.14, 0.69] (n=8) |
| <= 15 m | 21 | 2 | 14 | 0 | 5 | 0 | 0 | 0.125 [0.03, 0.36] (n=16) | 0.333 [0.17, 0.55] (n=21) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 5 | 1 | 2 | 0 | 2 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| 8-12 m | 7 | 0 | 5 | 0 | 2 | 0 | 0 | 0.000 [0.00, 0.43] (n=5) | 0.286 [0.08, 0.64] (n=7) |
| 12-18 m | 9 | 1 | 7 | 0 | 1 | 0 | 0 | 0.125 [0.02, 0.47] (n=8) | 0.222 [0.06, 0.55] (n=9) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 15 | 1 | 11 | 0 | 3 | 0 | 0 | 0.083 [0.01, 0.35] (n=12) | 0.267 [0.11, 0.52] (n=15) |
| 4 panos | 5 | 1 | 2 | 0 | 2 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| >= 5 panos | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.79] (n=1) | 0.000 [0.00, 0.79] (n=1) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 11 | 1 | 8 | 0 | 2 | 0 | 0 | 0.111 [0.02, 0.44] (n=9) | 0.273 [0.10, 0.57] (n=11) |
| best member < 0.9 | 10 | 1 | 6 | 0 | 3 | 0 | 0 | 0.143 [0.03, 0.51] (n=7) | 0.400 [0.17, 0.69] (n=10) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 601 | 0.04x |
| <= 15 m | 1976 | 0.13x |
