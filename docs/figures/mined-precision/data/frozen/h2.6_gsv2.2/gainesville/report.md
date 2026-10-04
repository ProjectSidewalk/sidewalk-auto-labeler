## gainesville: precision of mined positives (RampNet#158 step 1)

run: 35204 panos -> 14134 sites, 2229 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.2 m
excluded: 13 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.400 [0.12, 0.77] (n=5) -> read at <= 15 m: drop this label source; the 95% CI [0.12, 0.77] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.625 [0.31, 0.86] (n=8) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.31, 0.86] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 8 | 2 | 3 | 0 | 3 | 0 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.625 [0.31, 0.86] (n=8) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.44, 1.00] (n=3) |
| <= 15 m | 8 | 2 | 3 | 0 | 3 | 0 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.625 [0.31, 0.86] (n=8) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.44, 1.00] (n=3) |
| 8-12 m | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| 12-18 m | 5 | 0 | 3 | 0 | 2 | 0 | 0 | 0.000 [0.00, 0.56] (n=3) | 0.400 [0.12, 0.77] (n=5) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 7 | 2 | 3 | 0 | 2 | 0 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.571 [0.25, 0.84] (n=7) |
| 4 panos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| >= 5 panos | 1 | 0 | 0 | 0 | 1 | 0 | 0 | n/a | 1.000 [0.21, 1.00] (n=1) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 3 | 0 | 2 | 0 | 1 | 0 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.333 [0.06, 0.79] (n=3) |
| best member < 0.9 | 5 | 2 | 1 | 0 | 2 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.800 [0.38, 0.96] (n=5) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 255 | 0.02x |
| <= 15 m | 916 | 0.07x |
