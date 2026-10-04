## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 180283 panos -> 57493 sites, 19889 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 33 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- richmond: camera height mode `auto`: auto -> 2.6 (no GSV panos)
- richmond: all 9091 panos raycast at the constant 2.6 m (no pano took a measured or per-rig height)
- paterson: camera height mode `auto`: auto -> gsv-per-rig
- paterson: 34687 of 34687 panos took a per-rig height; 0 fell back to the 2.6 m constant
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
- bend: camera height mode `auto`: auto -> gsv-per-rig
- bend: 78560 of 78560 panos took a per-rig height; 0 fell back to the 2.6 m constant
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
- gainesville: camera height mode `auto`: auto -> gsv-per-rig
- gainesville: 35204 of 35204 panos took a per-rig height; 0 fell back to the 2.6 m constant
  - 2007: median 2.15 m, 153 of 291 measured -> 2.5 m
  - 2008: median 2.42 m, 9 of 14 measured -> 2.5 m
  - 2011: median 2.12 m, 136 of 353 measured -> 2.5 m
  - 2014: median 2.29 m, 44 of 102 measured -> 2.5 m
  - 2015: median 1.94 m, 225 of 585 measured -> 2.5 m
  - 2016: median 2.13 m, 110 of 517 measured -> 2.5 m
  - 2017: median 2.24 m, 17 of 92 measured -> 2.5 m
  - 2018: median 2.07 m, 160 of 1008 measured -> 2.5 m
  - 2019: median 2.14 m, 64 of 274 measured -> 2.5 m
  - 2021: median 2.30 m, 334 of 364 measured -> 2.5 m
  - 2022: median 2.28 m, 2224 of 2538 measured -> 2.5 m
  - 2023: median 2.30 m, 1619 of 2114 measured -> 2.5 m
  - 2024: median 2.31 m, 2286 of 2563 measured -> 2.5 m
  - 2025: median 2.29 m, 1127 of 1265 measured -> 2.5 m
  - 2026: median 1.76 m, 21950 of 23124 measured -> 2 m
- sao_paulo: camera height mode `auto`: auto -> gsv-per-rig
- sao_paulo: 22741 of 22741 panos took a per-rig height; 0 fell back to the 2.6 m constant
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

- **hard-only** (new misses): 0.452 [0.35, 0.56] (n=84) -> read at <= 15 m: drop this label source; the 95% CI [0.35, 0.56] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.562 [0.47, 0.65] (n=105) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.47, 0.65] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 118 | 38 | 46 | 1 | 21 | 13 | 0 | 0.452 [0.35, 0.56] (n=84) | 0.562 [0.47, 0.65] (n=105) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 51 | 22 | 12 | 0 | 10 | 7 | 0 | 0.647 [0.48, 0.79] (n=34) | 0.727 [0.58, 0.84] (n=44) |
| <= 15 m | 118 | 38 | 46 | 1 | 21 | 13 | 0 | 0.452 [0.35, 0.56] (n=84) | 0.562 [0.47, 0.65] (n=105) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 32 | 14 | 7 | 0 | 5 | 6 | 0 | 0.667 [0.45, 0.83] (n=21) | 0.731 [0.54, 0.86] (n=26) |
| 8-12 m | 35 | 12 | 12 | 0 | 8 | 3 | 0 | 0.500 [0.31, 0.69] (n=24) | 0.625 [0.45, 0.77] (n=32) |
| 12-18 m | 51 | 12 | 27 | 1 | 8 | 4 | 0 | 0.308 [0.19, 0.46] (n=39) | 0.426 [0.30, 0.57] (n=47) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 40 | 16 | 12 | 0 | 9 | 3 | 0 | 0.571 [0.39, 0.73] (n=28) | 0.676 [0.51, 0.80] (n=37) |
| 4 panos | 27 | 10 | 11 | 1 | 4 | 2 | 0 | 0.476 [0.28, 0.68] (n=21) | 0.560 [0.37, 0.73] (n=25) |
| >= 5 panos | 51 | 12 | 23 | 0 | 8 | 8 | 0 | 0.343 [0.21, 0.51] (n=35) | 0.465 [0.33, 0.61] (n=43) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 75 | 17 | 40 | 1 | 10 | 8 | 0 | 0.298 [0.20, 0.43] (n=57) | 0.403 [0.29, 0.52] (n=67) |
| best member < 0.9 | 43 | 21 | 6 | 0 | 11 | 5 | 0 | 0.778 [0.59, 0.89] (n=27) | 0.842 [0.70, 0.93] (n=38) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 3893 | 0.03x |
| <= 15 m | 13260 | 0.11x |
