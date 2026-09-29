## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

placement: roma -- 48 of 59 candidates placed by the arm, 7 fell back and 4 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.35 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 16).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.250 [0.15, 0.39] (n=48) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.294 [0.19, 0.43] (n=51) -> read at <= 15 m: drop this label source

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 59 | 12 | 36 | 0 | 3 | 8 | 0 | 0.250 [0.15, 0.39] (n=48) | 0.294 [0.19, 0.43] (n=51) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 28 | 9 | 11 | 0 | 2 | 6 | 0 | 0.450 [0.26, 0.66] (n=20) | 0.500 [0.31, 0.69] (n=22) |
| <= 15 m | 59 | 12 | 36 | 0 | 3 | 8 | 0 | 0.250 [0.15, 0.39] (n=48) | 0.294 [0.19, 0.43] (n=51) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 20 | 6 | 6 | 0 | 2 | 6 | 0 | 0.500 [0.25, 0.75] (n=12) | 0.571 [0.33, 0.79] (n=14) |
| 8-12 m | 16 | 4 | 11 | 0 | 0 | 1 | 0 | 0.267 [0.11, 0.52] (n=15) | 0.267 [0.11, 0.52] (n=15) |
| 12-18 m | 23 | 2 | 19 | 0 | 1 | 1 | 0 | 0.095 [0.03, 0.29] (n=21) | 0.136 [0.05, 0.33] (n=22) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 2 | 8 | 0 | 0 | 1 | 0 | 0.200 [0.06, 0.51] (n=10) | 0.200 [0.06, 0.51] (n=10) |
| 4 panos | 11 | 4 | 6 | 0 | 0 | 1 | 0 | 0.400 [0.17, 0.69] (n=10) | 0.400 [0.17, 0.69] (n=10) |
| >= 5 panos | 37 | 6 | 22 | 0 | 3 | 6 | 0 | 0.214 [0.10, 0.40] (n=28) | 0.290 [0.16, 0.47] (n=31) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 44 | 6 | 30 | 0 | 2 | 6 | 0 | 0.167 [0.08, 0.32] (n=36) | 0.211 [0.11, 0.36] (n=38) |
| best member < 0.9 | 15 | 6 | 6 | 0 | 1 | 2 | 0 | 0.500 [0.25, 0.75] (n=12) | 0.538 [0.29, 0.77] (n=13) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
