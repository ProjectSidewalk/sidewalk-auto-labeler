## paterson: precision of mined positives (RampNet#158 step 1)

run: 34687 panos -> 14363 sites, 4565 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 4 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.350 [0.18, 0.57] (n=20) -> read at <= 15 m: drop this label source; the 95% CI [0.18, 0.57] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.629 [0.46, 0.77] (n=35) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.46, 0.77] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 35 | 7 | 13 | 0 | 15 | 0 | 0 | 0.350 [0.18, 0.57] (n=20) | 0.629 [0.46, 0.77] (n=35) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 12 | 1 | 2 | 0 | 9 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.833 [0.55, 0.95] (n=12) |
| <= 15 m | 35 | 7 | 13 | 0 | 15 | 0 | 0 | 0.350 [0.18, 0.57] (n=20) | 0.629 [0.46, 0.77] (n=35) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 4 | 1 | 2 | 0 | 1 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.500 [0.15, 0.85] (n=4) |
| 8-12 m | 17 | 2 | 5 | 0 | 10 | 0 | 0 | 0.286 [0.08, 0.64] (n=7) | 0.706 [0.47, 0.87] (n=17) |
| 12-18 m | 14 | 4 | 6 | 0 | 4 | 0 | 0 | 0.400 [0.17, 0.69] (n=10) | 0.571 [0.33, 0.79] (n=14) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 16 | 4 | 2 | 0 | 10 | 0 | 0 | 0.667 [0.30, 0.90] (n=6) | 0.875 [0.64, 0.97] (n=16) |
| 4 panos | 13 | 3 | 6 | 0 | 4 | 0 | 0 | 0.333 [0.12, 0.65] (n=9) | 0.538 [0.29, 0.77] (n=13) |
| >= 5 panos | 6 | 0 | 5 | 0 | 1 | 0 | 0 | 0.000 [0.00, 0.43] (n=5) | 0.167 [0.03, 0.56] (n=6) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 23 | 4 | 10 | 0 | 9 | 0 | 0 | 0.286 [0.12, 0.55] (n=14) | 0.565 [0.37, 0.74] (n=23) |
| best member < 0.9 | 12 | 3 | 3 | 0 | 6 | 0 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.750 [0.47, 0.91] (n=12) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1096 | 0.04x |
| <= 15 m | 3988 | 0.13x |
