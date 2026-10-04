## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

placement: mapa_k_pair -- 54 of 59 candidates placed by the arm, 0 fell back and 5 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.59 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.448 [0.28, 0.62] (n=29) -> read at <= 15 m: drop this label source; the 95% CI [0.28, 0.62] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.680 [0.54, 0.79] (n=50) -> read at <= 15 m: add the visibility test before mining

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 59 | 13 | 16 | 0 | 21 | 9 | 0 | 0.448 [0.28, 0.62] (n=29) | 0.680 [0.54, 0.79] (n=50) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 28 | 10 | 4 | 0 | 7 | 7 | 0 | 0.714 [0.45, 0.88] (n=14) | 0.810 [0.60, 0.92] (n=21) |
| <= 15 m | 59 | 13 | 16 | 0 | 21 | 9 | 0 | 0.448 [0.28, 0.62] (n=29) | 0.680 [0.54, 0.79] (n=50) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 20 | 7 | 2 | 0 | 4 | 7 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.846 [0.58, 0.96] (n=13) |
| 8-12 m | 16 | 4 | 6 | 0 | 5 | 1 | 0 | 0.400 [0.17, 0.69] (n=10) | 0.600 [0.36, 0.80] (n=15) |
| 12-18 m | 23 | 2 | 8 | 0 | 12 | 1 | 0 | 0.200 [0.06, 0.51] (n=10) | 0.636 [0.43, 0.80] (n=22) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 3 | 4 | 0 | 3 | 1 | 0 | 0.429 [0.16, 0.75] (n=7) | 0.600 [0.31, 0.83] (n=10) |
| 4 panos | 11 | 4 | 1 | 0 | 4 | 2 | 0 | 0.800 [0.38, 0.96] (n=5) | 0.889 [0.56, 0.98] (n=9) |
| >= 5 panos | 37 | 6 | 11 | 0 | 14 | 6 | 0 | 0.353 [0.17, 0.59] (n=17) | 0.645 [0.47, 0.79] (n=31) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 44 | 6 | 14 | 0 | 17 | 7 | 0 | 0.300 [0.15, 0.52] (n=20) | 0.622 [0.46, 0.76] (n=37) |
| best member < 0.9 | 15 | 7 | 2 | 0 | 4 | 2 | 0 | 0.778 [0.45, 0.94] (n=9) | 0.846 [0.58, 0.96] (n=13) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
