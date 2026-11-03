## bend: precision of mined positives (RampNet#158 step 1)

run: 78560 panos -> 14061 sites, 9743 with >= 3 operational panos
GT: 110 fully judged panos, 110 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `auto`: auto -> gsv-per-rig
- 78560 of 78560 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 1.96 m, 6 of 6 measured -> 2.5 m
  - 2008: median 1.78 m, 9 of 9 measured -> 2.5 m
  - 2009: median 1.83 m, 13 of 19 measured -> 2.5 m
  - 2011: median n/a, 0 of 8 measured -> 2.5 m
  - 2012: median 2.36 m, 166 of 3603 measured -> 2.5 m
  - 2015: median 2.37 m, 77 of 206 measured -> 2.5 m
  - 2017: median 2.39 m, 72 of 740 measured -> 2.5 m
  - 2018: median 2.33 m, 264 of 1451 measured -> 2.5 m
  - 2019: median 2.34 m, 1541 of 1666 measured -> 2.5 m
  - 2021: median 2.38 m, 909 of 1007 measured -> 2.5 m
  - 2024: median 2.37 m, 59928 of 65897 measured -> 2.5 m
  - 2025: median 2.35 m, 2881 of 3948 measured -> 2.5 m

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.636 [0.35, 0.85] (n=11) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.35, 0.85] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.692 [0.42, 0.87] (n=13) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.42, 0.87] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 15 | 7 | 4 | 1 | 2 | 2 | 0 | 0.636 [0.35, 0.85] (n=11) | 0.692 [0.42, 0.87] (n=13) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.44, 1.00] (n=3) |
| <= 15 m | 15 | 7 | 4 | 1 | 2 | 2 | 0 | 0.636 [0.35, 0.85] (n=11) | 0.692 [0.42, 0.87] (n=13) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |
| 8-12 m | 5 | 3 | 1 | 0 | 1 | 0 | 0 | 0.750 [0.30, 0.95] (n=4) | 0.800 [0.38, 0.96] (n=5) |
| 12-18 m | 8 | 2 | 3 | 1 | 1 | 2 | 0 | 0.400 [0.12, 0.77] (n=5) | 0.500 [0.19, 0.81] (n=6) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.667 [0.21, 0.94] (n=3) |
| 4 panos | 8 | 4 | 2 | 1 | 1 | 1 | 0 | 0.667 [0.30, 0.90] (n=6) | 0.714 [0.36, 0.92] (n=7) |
| >= 5 panos | 4 | 1 | 1 | 0 | 1 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.667 [0.21, 0.94] (n=3) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 11 | 5 | 3 | 1 | 2 | 1 | 0 | 0.625 [0.31, 0.86] (n=8) | 0.700 [0.40, 0.89] (n=10) |
| best member < 0.9 | 4 | 2 | 1 | 0 | 0 | 1 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.667 [0.21, 0.94] (n=3) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1255 | 0.02x |
| <= 15 m | 5264 | 0.10x |
