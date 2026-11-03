## bend: precision of mined positives (RampNet#158 step 1)

run: 78560 panos -> 15130 sites, 9592 with >= 3 operational panos
GT: 110 fully judged panos, 110 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.2 m
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.778 [0.45, 0.94] (n=9) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.45, 0.94] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.833 [0.55, 0.95] (n=12) -> read at <= 15 m: build the miner; the 95% CI [0.55, 0.95] spans the visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 14 | 7 | 2 | 0 | 3 | 2 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.833 [0.55, 0.95] (n=12) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 4 | 3 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.51, 1.00] (n=4) |
| <= 15 m | 14 | 7 | 2 | 0 | 3 | 2 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.833 [0.55, 0.95] (n=12) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.44, 1.00] (n=3) |
| 8-12 m | 4 | 3 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.51, 1.00] (n=4) |
| 12-18 m | 7 | 2 | 2 | 0 | 1 | 2 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.600 [0.23, 0.88] (n=5) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.44, 1.00] (n=3) |
| 4 panos | 6 | 3 | 1 | 0 | 1 | 1 | 0 | 0.750 [0.30, 0.95] (n=4) | 0.800 [0.38, 0.96] (n=5) |
| >= 5 panos | 5 | 2 | 1 | 0 | 1 | 1 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.750 [0.30, 0.95] (n=4) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 11 | 5 | 2 | 0 | 3 | 1 | 0 | 0.714 [0.36, 0.92] (n=7) | 0.800 [0.49, 0.94] (n=10) |
| best member < 0.9 | 3 | 2 | 0 | 0 | 0 | 1 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1281 | 0.02x |
| <= 15 m | 5364 | 0.10x |
