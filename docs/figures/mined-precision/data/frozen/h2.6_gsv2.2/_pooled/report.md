## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 180283 panos -> 60568 sites, 19585 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.2/2.6 m
excluded: 30 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.440 [0.34, 0.54] (n=91) -> read at <= 15 m: drop this label source; the 95% CI [0.34, 0.54] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.585 [0.50, 0.67] (n=123) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.50, 0.67] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 136 | 40 | 51 | 0 | 32 | 13 | 0 | 0.440 [0.34, 0.54] (n=91) | 0.585 [0.50, 0.67] (n=123) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 57 | 24 | 14 | 0 | 13 | 6 | 0 | 0.632 [0.47, 0.77] (n=38) | 0.725 [0.59, 0.83] (n=51) |
| <= 15 m | 136 | 40 | 51 | 0 | 32 | 13 | 0 | 0.440 [0.34, 0.54] (n=91) | 0.585 [0.50, 0.67] (n=123) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 37 | 15 | 8 | 0 | 8 | 6 | 0 | 0.652 [0.45, 0.81] (n=23) | 0.742 [0.57, 0.86] (n=31) |
| 8-12 m | 37 | 12 | 13 | 0 | 10 | 2 | 0 | 0.480 [0.30, 0.67] (n=25) | 0.629 [0.46, 0.77] (n=35) |
| 12-18 m | 62 | 13 | 30 | 0 | 14 | 5 | 0 | 0.302 [0.19, 0.45] (n=43) | 0.474 [0.35, 0.60] (n=57) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 47 | 15 | 14 | 0 | 15 | 3 | 0 | 0.517 [0.34, 0.69] (n=29) | 0.682 [0.53, 0.80] (n=44) |
| 4 panos | 37 | 13 | 14 | 0 | 8 | 2 | 0 | 0.481 [0.31, 0.66] (n=27) | 0.600 [0.44, 0.74] (n=35) |
| >= 5 panos | 52 | 12 | 23 | 0 | 9 | 8 | 0 | 0.343 [0.21, 0.51] (n=35) | 0.477 [0.34, 0.62] (n=44) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 90 | 22 | 40 | 0 | 20 | 8 | 0 | 0.355 [0.25, 0.48] (n=62) | 0.512 [0.41, 0.62] (n=82) |
| best member < 0.9 | 46 | 18 | 11 | 0 | 12 | 5 | 0 | 0.621 [0.44, 0.77] (n=29) | 0.732 [0.58, 0.84] (n=41) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 4022 | 0.03x |
| <= 15 m | 13944 | 0.12x |
