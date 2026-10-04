## pooled over richmond, paterson, bend, gainesville, sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 189807 panos -> 60082 sites, 20497 with >= 3 operational panos
GT: 609 fully judged panos, 609 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height auto
excluded: 39 (site, pano) pairs where the pano is a member through a sub-threshold detection only

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
- gainesville: 37435 of 37435 panos took a per-rig height; 0 fell back to the 2.6 m constant
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
- sao_paulo: camera height mode `auto`: auto -> gsv-per-rig
- sao_paulo: 30034 of 30034 panos took a per-rig height; 0 fell back to the 2.6 m constant
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

- **hard-only** (new misses): 0.462 [0.36, 0.56] (n=91) -> read at <= 15 m: drop this label source; the 95% CI [0.36, 0.56] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.566 [0.47, 0.65] (n=113) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.47, 0.65] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 127 | 42 | 49 | 1 | 22 | 14 | 0 | 0.462 [0.36, 0.56] (n=91) | 0.566 [0.47, 0.65] (n=113) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 55 | 22 | 15 | 0 | 11 | 7 | 0 | 0.595 [0.43, 0.74] (n=37) | 0.688 [0.55, 0.80] (n=48) |
| <= 15 m | 127 | 42 | 49 | 1 | 22 | 14 | 0 | 0.462 [0.36, 0.56] (n=91) | 0.566 [0.47, 0.65] (n=113) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 35 | 14 | 10 | 0 | 5 | 6 | 0 | 0.583 [0.39, 0.76] (n=24) | 0.655 [0.47, 0.80] (n=29) |
| 8-12 m | 38 | 14 | 12 | 0 | 9 | 3 | 0 | 0.538 [0.35, 0.71] (n=26) | 0.657 [0.49, 0.79] (n=35) |
| 12-18 m | 54 | 14 | 27 | 1 | 8 | 5 | 0 | 0.341 [0.22, 0.49] (n=41) | 0.449 [0.32, 0.59] (n=49) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 42 | 18 | 13 | 0 | 8 | 3 | 0 | 0.581 [0.41, 0.74] (n=31) | 0.667 [0.51, 0.79] (n=39) |
| 4 panos | 32 | 10 | 14 | 1 | 6 | 2 | 0 | 0.417 [0.24, 0.61] (n=24) | 0.533 [0.36, 0.70] (n=30) |
| >= 5 panos | 53 | 14 | 22 | 0 | 8 | 9 | 0 | 0.389 [0.25, 0.55] (n=36) | 0.500 [0.36, 0.64] (n=44) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 77 | 18 | 40 | 1 | 10 | 9 | 0 | 0.310 [0.21, 0.44] (n=58) | 0.412 [0.30, 0.53] (n=68) |
| best member < 0.9 | 50 | 24 | 9 | 0 | 12 | 5 | 0 | 0.727 [0.56, 0.85] (n=33) | 0.800 [0.66, 0.89] (n=45) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 4557 | 0.04x |
| <= 15 m | 14890 | 0.12x |
