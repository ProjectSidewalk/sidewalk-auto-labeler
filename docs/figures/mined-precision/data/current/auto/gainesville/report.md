## gainesville: precision of mined positives (RampNet#158 step 1)

run: 37435 panos -> 14320 sites, 2472 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 14 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `auto`: auto -> gsv-per-rig
- 37435 of 37435 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 153 of 299 measured -> 2.5 m
  - 2008: median 2.42 m, 9 of 14 measured -> 2.5 m
  - 2011: median 2.12 m, 136 of 358 measured -> 2.5 m
  - 2014: median 2.29 m, 44 of 108 measured -> 2.5 m
  - 2015: median 1.94 m, 225 of 606 measured -> 2.5 m
  - 2016: median 2.13 m, 110 of 549 measured -> 2.5 m
  - 2017: median 2.24 m, 17 of 96 measured -> 2.5 m
  - 2018: median 2.07 m, 160 of 1059 measured -> 2.5 m
  - 2019: median 2.14 m, 64 of 304 measured -> 2.5 m
  - 2021: median 2.30 m, 334 of 368 measured -> 2.5 m
  - 2022: median 2.28 m, 2224 of 2592 measured -> 2.5 m
  - 2023: median 2.30 m, 1619 of 2140 measured -> 2.5 m
  - 2024: median 2.31 m, 2286 of 3679 measured -> 2.5 m
  - 2025: median 2.29 m, 1127 of 1497 measured -> 2.5 m
  - 2026: median 1.76 m, 21950 of 23766 measured -> 2 m

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.500 [0.15, 0.85] (n=4) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.15, 0.85] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.600 [0.23, 0.88] (n=5) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.23, 0.88] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 5 | 2 | 2 | 0 | 1 | 0 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.600 [0.23, 0.88] (n=5) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |
| <= 15 m | 5 | 2 | 2 | 0 | 1 | 0 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.600 [0.23, 0.88] (n=5) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |
| 8-12 m | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |
| 12-18 m | 3 | 0 | 2 | 0 | 1 | 0 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.333 [0.06, 0.79] (n=3) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.500 [0.15, 0.85] (n=4) |
| 4 panos | 1 | 0 | 0 | 0 | 1 | 0 | 0 | n/a | 1.000 [0.21, 1.00] (n=1) |
| >= 5 panos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.000 [0.00, 0.66] (n=2) |
| best member < 0.9 | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.44, 1.00] (n=3) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 265 | 0.02x |
| <= 15 m | 927 | 0.06x |
