## bend: precision of mined positives (RampNet#158 step 1)

run: 78560 panos -> 14205 sites, 9742 with >= 3 operational panos
GT: 110 fully judged panos, 110 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 65866 of 78560 panos took a measured height; 12694 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 969}
- 65866 heights read from depth/index.csv; spread definition: measured_planes_only

placement: mapa_posed_pair -- 12 of 12 candidates placed by the arm, 0 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 1.01 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.700 [0.40, 0.89] (n=10) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.40, 0.89] spans the drop/visibility/build bands, so this is not decisive
- **all-mined** (correct labels): 0.727 [0.43, 0.90] (n=11) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.43, 0.90] spans the drop/visibility/build bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 12 | 7 | 3 | 0 | 1 | 1 | 0 | 0.700 [0.40, 0.89] (n=10) | 0.727 [0.43, 0.90] (n=11) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.44, 1.00] (n=3) |
| <= 15 m | 12 | 7 | 3 | 0 | 1 | 1 | 0 | 0.700 [0.40, 0.89] (n=10) | 0.727 [0.43, 0.90] (n=11) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |
| 8-12 m | 4 | 2 | 1 | 0 | 1 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.750 [0.30, 0.95] (n=4) |
| 12-18 m | 6 | 3 | 2 | 0 | 0 | 1 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.600 [0.23, 0.88] (n=5) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |
| 4 panos | 6 | 3 | 2 | 0 | 0 | 1 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.600 [0.23, 0.88] (n=5) |
| >= 5 panos | 4 | 2 | 1 | 0 | 1 | 0 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.750 [0.30, 0.95] (n=4) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 8 | 5 | 2 | 0 | 1 | 0 | 0 | 0.714 [0.36, 0.92] (n=7) | 0.750 [0.41, 0.93] (n=8) |
| best member < 0.9 | 4 | 2 | 1 | 0 | 0 | 1 | 0 | 0.667 [0.21, 0.94] (n=3) | 0.667 [0.21, 0.94] (n=3) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1231 | 0.02x |
| <= 15 m | 5150 | 0.10x |
