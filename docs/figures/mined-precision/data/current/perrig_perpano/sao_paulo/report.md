## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 30034 panos -> 18251 sites, 2384 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 18 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 18601 of 30034 panos took a measured height; 11433 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- 18601 heights read from depth/index.csv; spread definition: measured_planes_only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.478 [0.29, 0.67] (n=23) -> read at <= 15 m: drop this label source; the 95% CI [0.29, 0.67] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.625 [0.45, 0.77] (n=32) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.45, 0.77] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 35 | 11 | 12 | 0 | 9 | 3 | 0 | 0.478 [0.29, 0.67] (n=23) | 0.625 [0.45, 0.77] (n=32) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 13 | 2 | 6 | 0 | 4 | 1 | 0 | 0.250 [0.07, 0.59] (n=8) | 0.500 [0.25, 0.75] (n=12) |
| <= 15 m | 35 | 11 | 12 | 0 | 9 | 3 | 0 | 0.478 [0.29, 0.67] (n=23) | 0.625 [0.45, 0.77] (n=32) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 6 | 1 | 3 | 0 | 1 | 1 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.400 [0.12, 0.77] (n=5) |
| 8-12 m | 15 | 4 | 5 | 0 | 4 | 2 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.615 [0.36, 0.82] (n=13) |
| 12-18 m | 14 | 6 | 4 | 0 | 4 | 0 | 0 | 0.600 [0.31, 0.83] (n=10) | 0.714 [0.45, 0.88] (n=14) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 17 | 7 | 6 | 0 | 2 | 2 | 0 | 0.538 [0.29, 0.77] (n=13) | 0.600 [0.36, 0.80] (n=15) |
| 4 panos | 6 | 0 | 3 | 0 | 3 | 0 | 0 | 0.000 [0.00, 0.56] (n=3) | 0.500 [0.19, 0.81] (n=6) |
| >= 5 panos | 12 | 4 | 3 | 0 | 4 | 1 | 0 | 0.571 [0.25, 0.84] (n=7) | 0.727 [0.43, 0.90] (n=11) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 15 | 5 | 6 | 0 | 3 | 1 | 0 | 0.455 [0.21, 0.72] (n=11) | 0.571 [0.33, 0.79] (n=14) |
| best member < 0.9 | 20 | 6 | 6 | 0 | 6 | 2 | 0 | 0.500 [0.25, 0.75] (n=12) | 0.667 [0.44, 0.84] (n=18) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1104 | 0.07x |
| <= 15 m | 3031 | 0.19x |
