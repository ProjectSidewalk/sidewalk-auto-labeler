## paterson: precision of mined positives (RampNet#158 step 1)

run: 34687 panos -> 13377 sites, 4899 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.2 m
excluded: 4 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.632 [0.41, 0.81] (n=19) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.41, 0.81] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.767 [0.59, 0.88] (n=30) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.59, 0.88] spans the visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 30 | 12 | 7 | 0 | 11 | 0 | 0 | 0.632 [0.41, 0.81] (n=19) | 0.767 [0.59, 0.88] (n=30) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 14 | 8 | 2 | 0 | 4 | 0 | 0 | 0.800 [0.49, 0.94] (n=10) | 0.857 [0.60, 0.96] (n=14) |
| <= 15 m | 30 | 12 | 7 | 0 | 11 | 0 | 0 | 0.632 [0.41, 0.81] (n=19) | 0.767 [0.59, 0.88] (n=30) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 7 | 3 | 2 | 0 | 2 | 0 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.714 [0.36, 0.92] (n=7) |
| 8-12 m | 10 | 5 | 0 | 0 | 5 | 0 | 0 | 1.000 [0.57, 1.00] (n=5) | 1.000 [0.72, 1.00] (n=10) |
| 12-18 m | 13 | 4 | 5 | 0 | 4 | 0 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.615 [0.36, 0.82] (n=13) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 15 | 5 | 3 | 0 | 7 | 0 | 0 | 0.625 [0.31, 0.86] (n=8) | 0.800 [0.55, 0.93] (n=15) |
| 4 panos | 9 | 3 | 3 | 0 | 3 | 0 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.667 [0.35, 0.88] (n=9) |
| >= 5 panos | 6 | 4 | 1 | 0 | 1 | 0 | 0 | 0.800 [0.38, 0.96] (n=5) | 0.833 [0.44, 0.97] (n=6) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 17 | 7 | 3 | 0 | 7 | 0 | 0 | 0.700 [0.40, 0.89] (n=10) | 0.824 [0.59, 0.94] (n=17) |
| best member < 0.9 | 13 | 5 | 4 | 0 | 4 | 0 | 0 | 0.556 [0.27, 0.81] (n=9) | 0.692 [0.42, 0.87] (n=13) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 866 | 0.03x |
| <= 15 m | 3173 | 0.11x |
