## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 189807 panos -> 63852 sites, 19647 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 35 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.333 [0.25, 0.43] (n=105) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.510 [0.43, 0.59] (n=143) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.43, 0.59] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 158 | 35 | 70 | 1 | 38 | 15 | 0 | 0.333 [0.25, 0.43] (n=105) | 0.510 [0.43, 0.59] (n=143) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 64 | 16 | 20 | 0 | 20 | 8 | 0 | 0.444 [0.30, 0.60] (n=36) | 0.643 [0.51, 0.76] (n=56) |
| <= 15 m | 158 | 35 | 70 | 1 | 38 | 15 | 0 | 0.333 [0.25, 0.43] (n=105) | 0.510 [0.43, 0.59] (n=143) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 39 | 12 | 12 | 0 | 8 | 7 | 0 | 0.500 [0.31, 0.69] (n=24) | 0.625 [0.45, 0.77] (n=32) |
| 8-12 m | 55 | 12 | 21 | 0 | 19 | 3 | 0 | 0.364 [0.22, 0.53] (n=33) | 0.596 [0.46, 0.72] (n=52) |
| 12-18 m | 64 | 11 | 37 | 1 | 11 | 5 | 0 | 0.229 [0.13, 0.37] (n=48) | 0.373 [0.26, 0.50] (n=59) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 57 | 12 | 23 | 0 | 18 | 4 | 0 | 0.343 [0.21, 0.51] (n=35) | 0.566 [0.43, 0.69] (n=53) |
| 4 panos | 44 | 12 | 18 | 1 | 12 | 2 | 0 | 0.400 [0.25, 0.58] (n=30) | 0.571 [0.42, 0.71] (n=42) |
| >= 5 panos | 57 | 11 | 29 | 0 | 8 | 9 | 0 | 0.275 [0.16, 0.43] (n=40) | 0.396 [0.27, 0.54] (n=48) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 103 | 18 | 54 | 1 | 22 | 9 | 0 | 0.250 [0.16, 0.36] (n=72) | 0.426 [0.33, 0.53] (n=94) |
| best member < 0.9 | 55 | 17 | 16 | 0 | 16 | 6 | 0 | 0.515 [0.35, 0.67] (n=33) | 0.673 [0.53, 0.79] (n=49) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 5200 | 0.04x |
| <= 15 m | 17201 | 0.14x |
