## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

placement: peak_mapa_k_pair -- 18 of 59 candidates placed by the arm, 0 fell back and 1 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.17 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it. 40 candidates were NOT EMITTED by the placement's target definition and are in neither denominator.

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 2).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.467 [0.25, 0.70] (n=15) -> read at <= 15 m: drop this label source; the 95% CI [0.25, 0.70] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.467 [0.25, 0.70] (n=15) -> read at <= 15 m: drop this label source; the 95% CI [0.25, 0.70] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 19 | 7 | 8 | 0 | 0 | 4 | 0 | 0.467 [0.25, 0.70] (n=15) | 0.467 [0.25, 0.70] (n=15) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 11 | 6 | 4 | 0 | 0 | 1 | 0 | 0.600 [0.31, 0.83] (n=10) | 0.600 [0.31, 0.83] (n=10) |
| <= 15 m | 19 | 7 | 8 | 0 | 0 | 4 | 0 | 0.467 [0.25, 0.70] (n=15) | 0.467 [0.25, 0.70] (n=15) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 8 | 5 | 2 | 0 | 0 | 1 | 0 | 0.714 [0.36, 0.92] (n=7) | 0.714 [0.36, 0.92] (n=7) |
| 8-12 m | 6 | 1 | 4 | 0 | 0 | 1 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.200 [0.04, 0.62] (n=5) |
| 12-18 m | 5 | 1 | 2 | 0 | 0 | 2 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.333 [0.06, 0.79] (n=3) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.333 [0.06, 0.79] (n=3) |
| 4 panos | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.44, 1.00] (n=3) | 1.000 [0.44, 1.00] (n=3) |
| >= 5 panos | 13 | 3 | 6 | 0 | 0 | 4 | 0 | 0.333 [0.12, 0.65] (n=9) | 0.333 [0.12, 0.65] (n=9) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 13 | 2 | 7 | 0 | 0 | 4 | 0 | 0.222 [0.06, 0.55] (n=9) | 0.222 [0.06, 0.55] (n=9) |
| best member < 0.9 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0.833 [0.44, 0.97] (n=6) | 0.833 [0.44, 0.97] (n=6) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
