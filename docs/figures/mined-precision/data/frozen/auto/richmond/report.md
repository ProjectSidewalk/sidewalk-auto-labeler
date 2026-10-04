## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `auto`: auto -> 2.6 (no GSV panos)
- all 9091 panos raycast at the constant 2.6 m (no pano took a measured or per-rig height)

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.310 [0.19, 0.46] (n=42) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.431 [0.31, 0.57] (n=51) -> read at <= 15 m: drop this label source; the 95% CI [0.31, 0.57] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 59 | 13 | 29 | 0 | 9 | 8 | 0 | 0.310 [0.19, 0.46] (n=42) | 0.431 [0.31, 0.57] (n=51) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 28 | 10 | 8 | 0 | 5 | 5 | 0 | 0.556 [0.34, 0.75] (n=18) | 0.652 [0.45, 0.81] (n=23) |
| <= 15 m | 59 | 13 | 29 | 0 | 9 | 8 | 0 | 0.310 [0.19, 0.46] (n=42) | 0.431 [0.31, 0.57] (n=51) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 20 | 7 | 5 | 0 | 3 | 5 | 0 | 0.583 [0.32, 0.81] (n=12) | 0.667 [0.42, 0.85] (n=15) |
| 8-12 m | 16 | 4 | 8 | 0 | 3 | 1 | 0 | 0.333 [0.14, 0.61] (n=12) | 0.467 [0.25, 0.70] (n=15) |
| 12-18 m | 23 | 2 | 16 | 0 | 3 | 2 | 0 | 0.111 [0.03, 0.33] (n=18) | 0.238 [0.11, 0.45] (n=21) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 3 | 5 | 0 | 2 | 1 | 0 | 0.375 [0.14, 0.69] (n=8) | 0.500 [0.24, 0.76] (n=10) |
| 4 panos | 11 | 4 | 5 | 0 | 2 | 0 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.545 [0.28, 0.79] (n=11) |
| >= 5 panos | 37 | 6 | 19 | 0 | 5 | 7 | 0 | 0.240 [0.11, 0.43] (n=25) | 0.367 [0.22, 0.54] (n=30) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 44 | 6 | 27 | 0 | 5 | 6 | 0 | 0.182 [0.09, 0.34] (n=33) | 0.289 [0.17, 0.45] (n=38) |
| best member < 0.9 | 15 | 7 | 2 | 0 | 4 | 2 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.846 [0.58, 0.96] (n=13) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
