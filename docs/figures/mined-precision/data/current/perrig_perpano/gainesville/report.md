## gainesville: precision of mined positives (RampNet#158 step 1)

run: 37435 panos -> 15617 sites, 2484 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 12 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 30458 of 37435 panos took a measured height; 6977 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 3112}
- 30458 heights read from depth/index.csv; spread definition: measured_planes_only

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.500 [0.09, 0.91] (n=2) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.09, 0.91] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.875 [0.53, 0.98] (n=8) -> read at <= 15 m: build the miner; the 95% CI [0.53, 0.98] spans the visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 8 | 1 | 1 | 0 | 6 | 0 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.875 [0.53, 0.98] (n=8) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 5 | 1 | 0 | 0 | 4 | 0 | 0 | 1.000 [0.21, 1.00] (n=1) | 1.000 [0.57, 1.00] (n=5) |
| <= 15 m | 8 | 1 | 1 | 0 | 6 | 0 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.875 [0.53, 0.98] (n=8) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 1 | 0 | 0 | 2 | 0 | 0 | 1.000 [0.21, 1.00] (n=1) | 1.000 [0.44, 1.00] (n=3) |
| 8-12 m | 2 | 0 | 0 | 0 | 2 | 0 | 0 | n/a | 1.000 [0.34, 1.00] (n=2) |
| 12-18 m | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 0.000 [0.00, 0.79] (n=1) | 0.667 [0.21, 0.94] (n=3) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 4 | 1 | 0 | 0 | 3 | 0 | 0 | 1.000 [0.21, 1.00] (n=1) | 1.000 [0.51, 1.00] (n=4) |
| 4 panos | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 0.000 [0.00, 0.79] (n=1) | 0.667 [0.21, 0.94] (n=3) |
| >= 5 panos | 1 | 0 | 0 | 0 | 1 | 0 | 0 | n/a | 1.000 [0.21, 1.00] (n=1) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 3 | 0 | 1 | 0 | 2 | 0 | 0 | 0.000 [0.00, 0.79] (n=1) | 0.667 [0.21, 0.94] (n=3) |
| best member < 0.9 | 5 | 1 | 0 | 0 | 4 | 0 | 0 | 1.000 [0.21, 1.00] (n=1) | 1.000 [0.57, 1.00] (n=5) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 320 | 0.02x |
| <= 15 m | 1120 | 0.07x |
