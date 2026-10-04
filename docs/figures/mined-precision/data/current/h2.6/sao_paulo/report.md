## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 30034 panos -> 17318 sites, 2371 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 22 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.375 [0.18, 0.61] (n=16) -> read at <= 15 m: drop this label source; the 95% CI [0.18, 0.61] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.545 [0.35, 0.73] (n=22) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.35, 0.73] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 27 | 6 | 10 | 0 | 6 | 5 | 0 | 0.375 [0.18, 0.61] (n=16) | 0.545 [0.35, 0.73] (n=22) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 12 | 1 | 5 | 0 | 3 | 3 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.444 [0.19, 0.73] (n=9) |
| <= 15 m | 27 | 6 | 10 | 0 | 6 | 5 | 0 | 0.375 [0.18, 0.61] (n=16) | 0.545 [0.35, 0.73] (n=22) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 7 | 1 | 3 | 0 | 1 | 2 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.400 [0.12, 0.77] (n=5) |
| 8-12 m | 10 | 3 | 2 | 0 | 3 | 2 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.750 [0.41, 0.93] (n=8) |
| 12-18 m | 10 | 2 | 5 | 0 | 2 | 1 | 0 | 0.286 [0.08, 0.64] (n=7) | 0.444 [0.19, 0.73] (n=9) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 2 | 4 | 0 | 2 | 3 | 0 | 0.333 [0.10, 0.70] (n=6) | 0.500 [0.22, 0.78] (n=8) |
| 4 panos | 8 | 1 | 3 | 0 | 3 | 1 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.571 [0.25, 0.84] (n=7) |
| >= 5 panos | 8 | 3 | 3 | 0 | 1 | 1 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.571 [0.25, 0.84] (n=7) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 13 | 2 | 6 | 0 | 3 | 2 | 0 | 0.250 [0.07, 0.59] (n=8) | 0.455 [0.21, 0.72] (n=11) |
| best member < 0.9 | 14 | 4 | 4 | 0 | 3 | 3 | 0 | 0.500 [0.22, 0.78] (n=8) | 0.636 [0.35, 0.85] (n=11) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1012 | 0.06x |
| <= 15 m | 2778 | 0.17x |
