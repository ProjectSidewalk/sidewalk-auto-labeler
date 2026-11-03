## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 22741 panos -> 16381 sites, 1921 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 13 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 18601 of 22741 panos took a measured height; 4140 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- 18601 heights read from depth/index.csv; spread definition: measured_planes_only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.429 [0.21, 0.67] (n=14) -> read at <= 15 m: drop this label source; the 95% CI [0.21, 0.67] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.556 [0.34, 0.75] (n=18) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.34, 0.75] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 21 | 6 | 8 | 0 | 4 | 3 | 0 | 0.429 [0.21, 0.67] (n=14) | 0.556 [0.34, 0.75] (n=18) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 7 | 2 | 3 | 0 | 1 | 1 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.500 [0.19, 0.81] (n=6) |
| <= 15 m | 21 | 6 | 8 | 0 | 4 | 3 | 0 | 0.429 [0.21, 0.67] (n=14) | 0.556 [0.34, 0.75] (n=18) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 1 | 1 | 0 | 0 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.500 [0.09, 0.91] (n=2) |
| 8-12 m | 10 | 2 | 4 | 0 | 2 | 2 | 0 | 0.333 [0.10, 0.70] (n=6) | 0.500 [0.22, 0.78] (n=8) |
| 12-18 m | 8 | 3 | 3 | 0 | 2 | 0 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.625 [0.31, 0.86] (n=8) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 12 | 5 | 4 | 0 | 1 | 2 | 0 | 0.556 [0.27, 0.81] (n=9) | 0.600 [0.31, 0.83] (n=10) |
| 4 panos | 5 | 1 | 2 | 0 | 2 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| >= 5 panos | 4 | 0 | 2 | 0 | 1 | 1 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.333 [0.06, 0.79] (n=3) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 12 | 4 | 5 | 0 | 2 | 1 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.545 [0.28, 0.79] (n=11) |
| best member < 0.9 | 9 | 2 | 3 | 0 | 2 | 2 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.571 [0.25, 0.84] (n=7) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 417 | 0.03x |
| <= 15 m | 1427 | 0.12x |
