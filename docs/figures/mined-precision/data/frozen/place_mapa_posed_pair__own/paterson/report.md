## paterson: precision of mined positives (RampNet#158 step 1)

run: 34687 panos -> 13174 sites, 5029 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 4 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 30325 of 34687 panos took a measured height; 4362 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 484}
- 30325 heights read from depth/index.csv; spread definition: measured_planes_only

placement: mapa_posed_pair -- 27 of 27 candidates placed by the arm, 0 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 1.28 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 11).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.360 [0.20, 0.55] (n=25) -> read at <= 15 m: drop this label source; the 95% CI [0.20, 0.55] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.407 [0.25, 0.59] (n=27) -> read at <= 15 m: drop this label source; the 95% CI [0.25, 0.59] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 27 | 9 | 16 | 0 | 2 | 0 | 0 | 0.360 [0.20, 0.55] (n=25) | 0.407 [0.25, 0.59] (n=27) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 11 | 2 | 8 | 0 | 1 | 0 | 0 | 0.200 [0.06, 0.51] (n=10) | 0.273 [0.10, 0.57] (n=11) |
| <= 15 m | 27 | 9 | 16 | 0 | 2 | 0 | 0 | 0.360 [0.20, 0.55] (n=25) | 0.407 [0.25, 0.59] (n=27) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 6 | 1 | 4 | 0 | 1 | 0 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.333 [0.10, 0.70] (n=6) |
| 8-12 m | 6 | 2 | 4 | 0 | 0 | 0 | 0 | 0.333 [0.10, 0.70] (n=6) | 0.333 [0.10, 0.70] (n=6) |
| 12-18 m | 15 | 6 | 8 | 0 | 1 | 0 | 0 | 0.429 [0.21, 0.67] (n=14) | 0.467 [0.25, 0.70] (n=15) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 14 | 4 | 9 | 0 | 1 | 0 | 0 | 0.308 [0.13, 0.58] (n=13) | 0.357 [0.16, 0.61] (n=14) |
| 4 panos | 10 | 4 | 5 | 0 | 1 | 0 | 0 | 0.444 [0.19, 0.73] (n=9) | 0.500 [0.24, 0.76] (n=10) |
| >= 5 panos | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.333 [0.06, 0.79] (n=3) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 11 | 4 | 6 | 0 | 1 | 0 | 0 | 0.400 [0.17, 0.69] (n=10) | 0.455 [0.21, 0.72] (n=11) |
| best member < 0.9 | 16 | 5 | 10 | 0 | 1 | 0 | 0 | 0.333 [0.15, 0.58] (n=15) | 0.375 [0.18, 0.61] (n=16) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 793 | 0.03x |
| <= 15 m | 2998 | 0.10x |
