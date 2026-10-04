## paterson: precision of mined positives (RampNet#158 step 1)

run: 34687 panos -> 13174 sites, 5029 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 4 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 30325 of 34687 panos took a measured height; 4362 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 484}
- 30325 heights read from depth/index.csv; spread definition: measured_planes_only

placement: mapa_k_pair -- 26 of 27 candidates placed by the arm, 1 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 1.10 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.722 [0.49, 0.88] (n=18) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.49, 0.88] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.815 [0.63, 0.92] (n=27) -> read at <= 15 m: build the miner; the 95% CI [0.63, 0.92] spans the visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 27 | 13 | 5 | 0 | 9 | 0 | 0 | 0.722 [0.49, 0.88] (n=18) | 0.815 [0.63, 0.92] (n=27) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 11 | 5 | 2 | 0 | 4 | 0 | 0 | 0.714 [0.36, 0.92] (n=7) | 0.818 [0.52, 0.95] (n=11) |
| <= 15 m | 27 | 13 | 5 | 0 | 9 | 0 | 0 | 0.722 [0.49, 0.88] (n=18) | 0.815 [0.63, 0.92] (n=27) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 6 | 3 | 2 | 0 | 1 | 0 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.667 [0.30, 0.90] (n=6) |
| 8-12 m | 6 | 3 | 0 | 0 | 3 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.61, 1.00] (n=6) |
| 12-18 m | 15 | 7 | 3 | 0 | 5 | 0 | 0 | 0.700 [0.40, 0.89] (n=10) | 0.800 [0.55, 0.93] (n=15) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 14 | 9 | 2 | 0 | 3 | 0 | 0 | 0.818 [0.52, 0.95] (n=11) | 0.857 [0.60, 0.96] (n=14) |
| 4 panos | 10 | 2 | 2 | 0 | 6 | 0 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.800 [0.49, 0.94] (n=10) |
| >= 5 panos | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.667 [0.21, 0.94] (n=3) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 11 | 8 | 1 | 0 | 2 | 0 | 0 | 0.889 [0.56, 0.98] (n=9) | 0.909 [0.62, 0.98] (n=11) |
| best member < 0.9 | 16 | 5 | 4 | 0 | 7 | 0 | 0 | 0.556 [0.27, 0.81] (n=9) | 0.750 [0.51, 0.90] (n=16) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 793 | 0.03x |
| <= 15 m | 2998 | 0.10x |
