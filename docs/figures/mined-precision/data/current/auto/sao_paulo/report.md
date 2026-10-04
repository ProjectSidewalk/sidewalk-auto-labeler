## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 30034 panos -> 17428 sites, 2384 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 21 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `auto`: auto -> gsv-per-rig
- 30034 of 30034 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2010: median 2.20 m, 2 of 92 measured -> 2.5 m
  - 2011: median 2.37 m, 28 of 83 measured -> 2.5 m
  - 2014: median 2.38 m, 36 of 815 measured -> 2.5 m
  - 2015: median 2.38 m, 3 of 9 measured -> 2.5 m
  - 2016: median 1.74 m, 64 of 108 measured -> 2 m
  - 2017: median 1.72 m, 174 of 195 measured -> 2 m
  - 2018: median 2.10 m, 152 of 172 measured -> 2.5 m
  - 2019: median 2.25 m, 131 of 146 measured -> 2.5 m
  - 2020: median 2.15 m, 83 of 90 measured -> 2.5 m
  - 2021: median 1.65 m, 876 of 992 measured -> 2 m
  - 2022: median 2.22 m, 714 of 807 measured -> 2.5 m
  - 2023: median 2.26 m, 1443 of 1636 measured -> 2.5 m
  - 2024: median 2.27 m, 7049 of 14606 measured -> 2.5 m
  - 2025: median 2.23 m, 7593 of 9963 measured -> 2.5 m
  - 2026: median 2.13 m, 253 of 320 measured -> 2.5 m

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.438 [0.23, 0.67] (n=16) -> read at <= 15 m: drop this label source; the 95% CI [0.23, 0.67] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.571 [0.37, 0.76] (n=21) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.37, 0.76] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 25 | 7 | 9 | 0 | 5 | 4 | 0 | 0.438 [0.23, 0.67] (n=16) | 0.571 [0.37, 0.76] (n=21) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 11 | 1 | 6 | 0 | 2 | 2 | 0 | 0.143 [0.03, 0.51] (n=7) | 0.333 [0.12, 0.65] (n=9) |
| <= 15 m | 25 | 7 | 9 | 0 | 5 | 4 | 0 | 0.438 [0.23, 0.67] (n=16) | 0.571 [0.37, 0.76] (n=21) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 6 | 1 | 4 | 0 | 0 | 1 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.200 [0.04, 0.62] (n=5) |
| 8-12 m | 10 | 3 | 2 | 0 | 3 | 2 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.750 [0.41, 0.93] (n=8) |
| 12-18 m | 9 | 3 | 3 | 0 | 2 | 1 | 0 | 0.500 [0.19, 0.81] (n=6) | 0.625 [0.31, 0.86] (n=8) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 3 | 4 | 0 | 2 | 2 | 0 | 0.429 [0.16, 0.75] (n=7) | 0.556 [0.27, 0.81] (n=9) |
| 4 panos | 8 | 1 | 4 | 0 | 2 | 1 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.429 [0.16, 0.75] (n=7) |
| >= 5 panos | 6 | 3 | 1 | 0 | 1 | 1 | 0 | 0.750 [0.30, 0.95] (n=4) | 0.800 [0.38, 0.96] (n=5) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 12 | 2 | 6 | 0 | 2 | 2 | 0 | 0.250 [0.07, 0.59] (n=8) | 0.400 [0.17, 0.69] (n=10) |
| best member < 0.9 | 13 | 5 | 3 | 0 | 3 | 2 | 0 | 0.625 [0.31, 0.86] (n=8) | 0.727 [0.43, 0.90] (n=11) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1027 | 0.06x |
| <= 15 m | 2788 | 0.17x |
