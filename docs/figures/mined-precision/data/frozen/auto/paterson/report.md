## paterson: precision of mined positives (RampNet#158 step 1)

run: 34687 panos -> 12703 sites, 4953 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 4 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `auto`: auto -> gsv-per-rig
- 34687 of 34687 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 118 of 136 measured -> 2.5 m
  - 2008: median 2.30 m, 98 of 151 measured -> 2.5 m
  - 2012: median 2.33 m, 1 of 414 measured -> 2.5 m
  - 2013: median 2.29 m, 1 of 88 measured -> 2.5 m
  - 2014: median 2.08 m, 12 of 278 measured -> 2.5 m
  - 2015: median 2.45 m, 3 of 155 measured -> 2.5 m
  - 2016: median 2.42 m, 1 of 93 measured -> 2.5 m
  - 2017: median 2.38 m, 17 of 111 measured -> 2.5 m
  - 2018: median 2.35 m, 1523 of 1646 measured -> 2.5 m
  - 2019: median 2.37 m, 1263 of 1509 measured -> 2.5 m
  - 2020: median 2.35 m, 2933 of 3102 measured -> 2.5 m
  - 2021: median 2.35 m, 1658 of 1794 measured -> 2.5 m
  - 2022: median 2.35 m, 29 of 30 measured -> 2.5 m
  - 2023: median 2.14 m, 11 of 35 measured -> 2.5 m
  - 2024: median 2.37 m, 7408 of 8403 measured -> 2.5 m
  - 2025: median 1.86 m, 15155 of 16615 measured -> 2 m
  - 2026: median 1.74 m, 94 of 127 measured -> 2 m

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.722 [0.49, 0.88] (n=18) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.49, 0.88] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.783 [0.58, 0.90] (n=23) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.58, 0.90] spans the visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 23 | 13 | 5 | 0 | 5 | 0 | 0 | 0.722 [0.49, 0.88] (n=18) | 0.783 [0.58, 0.90] (n=23) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 11 | 6 | 1 | 0 | 4 | 0 | 0 | 0.857 [0.49, 0.97] (n=7) | 0.909 [0.62, 0.98] (n=11) |
| <= 15 m | 23 | 13 | 5 | 0 | 5 | 0 | 0 | 0.722 [0.49, 0.88] (n=18) | 0.783 [0.58, 0.90] (n=23) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 5 | 2 | 1 | 0 | 2 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.800 [0.38, 0.96] (n=5) |
| 8-12 m | 7 | 4 | 1 | 0 | 2 | 0 | 0 | 0.800 [0.38, 0.96] (n=5) | 0.857 [0.49, 0.97] (n=7) |
| 12-18 m | 11 | 7 | 3 | 0 | 1 | 0 | 0 | 0.700 [0.40, 0.89] (n=10) | 0.727 [0.43, 0.90] (n=11) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 13 | 8 | 1 | 0 | 4 | 0 | 0 | 0.889 [0.56, 0.98] (n=9) | 0.923 [0.67, 0.99] (n=13) |
| 4 panos | 4 | 1 | 3 | 0 | 0 | 0 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.250 [0.05, 0.70] (n=4) |
| >= 5 panos | 6 | 4 | 1 | 0 | 1 | 0 | 0 | 0.800 [0.38, 0.96] (n=5) | 0.833 [0.44, 0.97] (n=6) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 8 | 5 | 2 | 0 | 1 | 0 | 0 | 0.714 [0.36, 0.92] (n=7) | 0.750 [0.41, 0.93] (n=8) |
| best member < 0.9 | 15 | 8 | 3 | 0 | 4 | 0 | 0 | 0.727 [0.43, 0.90] (n=11) | 0.800 [0.55, 0.93] (n=15) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 800 | 0.03x |
| <= 15 m | 2900 | 0.10x |
