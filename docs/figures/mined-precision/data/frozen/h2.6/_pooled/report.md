## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 180283 panos -> 61239 sites, 19025 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height 2.6 m
excluded: 30 (site, pano) pairs where the pano is a member through a sub-threshold detection only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.327 [0.24, 0.42] (n=98) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.511 [0.43, 0.59] (n=135) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.43, 0.59] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 148 | 32 | 66 | 1 | 37 | 13 | 0 | 0.327 [0.24, 0.42] (n=98) | 0.511 [0.43, 0.59] (n=135) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 60 | 16 | 18 | 0 | 19 | 7 | 0 | 0.471 [0.31, 0.63] (n=34) | 0.660 [0.53, 0.77] (n=53) |
| <= 15 m | 148 | 32 | 66 | 1 | 37 | 13 | 0 | 0.327 [0.24, 0.42] (n=98) | 0.511 [0.43, 0.59] (n=135) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 36 | 12 | 10 | 0 | 8 | 6 | 0 | 0.545 [0.35, 0.73] (n=22) | 0.667 [0.49, 0.81] (n=30) |
| 8-12 m | 51 | 10 | 20 | 0 | 18 | 3 | 0 | 0.333 [0.19, 0.51] (n=30) | 0.583 [0.44, 0.71] (n=48) |
| 12-18 m | 61 | 10 | 36 | 1 | 11 | 4 | 0 | 0.217 [0.12, 0.36] (n=46) | 0.368 [0.26, 0.50] (n=57) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 55 | 11 | 21 | 0 | 19 | 4 | 0 | 0.344 [0.20, 0.52] (n=32) | 0.588 [0.45, 0.71] (n=51) |
| 4 panos | 40 | 13 | 16 | 1 | 10 | 1 | 0 | 0.448 [0.28, 0.62] (n=29) | 0.590 [0.43, 0.73] (n=39) |
| >= 5 panos | 53 | 8 | 29 | 0 | 8 | 8 | 0 | 0.216 [0.11, 0.37] (n=37) | 0.356 [0.23, 0.50] (n=45) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 100 | 17 | 53 | 1 | 22 | 8 | 0 | 0.243 [0.16, 0.35] (n=70) | 0.424 [0.33, 0.53] (n=92) |
| best member < 0.9 | 48 | 15 | 13 | 0 | 15 | 5 | 0 | 0.536 [0.36, 0.70] (n=28) | 0.698 [0.55, 0.81] (n=43) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 4523 | 0.04x |
| <= 15 m | 15481 | 0.13x |
