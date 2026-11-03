## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 22741 panos -> 15519 sites, 1955 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 16 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `auto`: auto -> gsv-per-rig
- 22741 of 22741 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2010: median 2.20 m, 2 of 92 measured -> 2.5 m
  - 2011: median 2.37 m, 28 of 83 measured -> 2.5 m
  - 2014: median 2.38 m, 36 of 93 measured -> 2.5 m
  - 2015: median 2.38 m, 3 of 9 measured -> 2.5 m
  - 2016: median 1.74 m, 64 of 108 measured -> 2 m
  - 2017: median 1.72 m, 174 of 192 measured -> 2 m
  - 2018: median 2.10 m, 152 of 165 measured -> 2.5 m
  - 2019: median 2.25 m, 131 of 145 measured -> 2.5 m
  - 2020: median 2.15 m, 83 of 90 measured -> 2.5 m
  - 2021: median 1.65 m, 876 of 988 measured -> 2 m
  - 2022: median 2.22 m, 714 of 791 measured -> 2.5 m
  - 2023: median 2.26 m, 1443 of 1604 measured -> 2.5 m
  - 2024: median 2.27 m, 7049 of 8538 measured -> 2.5 m
  - 2025: median 2.23 m, 7593 of 9543 measured -> 2.5 m
  - 2026: median 2.13 m, 253 of 300 measured -> 2.5 m

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.375 [0.14, 0.69] (n=8) -> read at <= 15 m: drop this label source; the 95% CI [0.14, 0.69] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.583 [0.32, 0.81] (n=12) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.32, 0.81] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 15 | 3 | 5 | 0 | 4 | 3 | 0 | 0.375 [0.14, 0.69] (n=8) | 0.583 [0.32, 0.81] (n=12) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 7 | 1 | 3 | 0 | 1 | 2 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.400 [0.12, 0.77] (n=5) |
| <= 15 m | 15 | 3 | 5 | 0 | 4 | 3 | 0 | 0.375 [0.14, 0.69] (n=8) | 0.583 [0.32, 0.81] (n=12) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 1 | 1 | 0 | 0 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.500 [0.09, 0.91] (n=2) |
| 8-12 m | 7 | 1 | 2 | 0 | 2 | 2 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| 12-18 m | 5 | 1 | 2 | 0 | 2 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 7 | 1 | 2 | 0 | 2 | 2 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.600 [0.23, 0.88] (n=5) |
| 4 panos | 4 | 1 | 1 | 0 | 1 | 1 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.667 [0.21, 0.94] (n=3) |
| >= 5 panos | 4 | 1 | 2 | 0 | 1 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.500 [0.15, 0.85] (n=4) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 9 | 1 | 5 | 0 | 2 | 1 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.375 [0.14, 0.69] (n=8) |
| best member < 0.9 | 6 | 2 | 0 | 0 | 2 | 2 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.51, 1.00] (n=4) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 429 | 0.04x |
| <= 15 m | 1362 | 0.11x |
