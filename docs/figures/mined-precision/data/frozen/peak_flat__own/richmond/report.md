## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

placement: peak_flat -- 16 of 59 candidates placed by the arm, 0 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 1.63 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it. 43 candidates were NOT EMITTED by the placement's target definition and are in neither denominator.

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 3).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.538 [0.29, 0.77] (n=13) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.29, 0.77] spans the drop/visibility bands, so this is not decisive
- **all-mined** (correct labels): 0.538 [0.29, 0.77] (n=13) -> read at <= 15 m: add the visibility test before mining; the 95% CI [0.29, 0.77] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 16 | 7 | 6 | 0 | 0 | 3 | 0 | 0.538 [0.29, 0.77] (n=13) | 0.538 [0.29, 0.77] (n=13) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 9 | 5 | 4 | 0 | 0 | 0 | 0 | 0.556 [0.27, 0.81] (n=9) | 0.556 [0.27, 0.81] (n=9) |
| <= 15 m | 16 | 7 | 6 | 0 | 0 | 3 | 0 | 0.538 [0.29, 0.77] (n=13) | 0.538 [0.29, 0.77] (n=13) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 5 | 3 | 2 | 0 | 0 | 0 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.600 [0.23, 0.88] (n=5) |
| 8-12 m | 6 | 3 | 2 | 0 | 0 | 1 | 0 | 0.600 [0.23, 0.88] (n=5) | 0.600 [0.23, 0.88] (n=5) |
| 12-18 m | 5 | 1 | 2 | 0 | 0 | 2 | 0 | 0.333 [0.06, 0.79] (n=3) | 0.333 [0.06, 0.79] (n=3) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0.500 [0.15, 0.85] (n=4) | 0.500 [0.15, 0.85] (n=4) |
| 4 panos | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 1.000 [0.34, 1.00] (n=2) | 1.000 [0.34, 1.00] (n=2) |
| >= 5 panos | 10 | 3 | 4 | 0 | 0 | 3 | 0 | 0.429 [0.16, 0.75] (n=7) | 0.429 [0.16, 0.75] (n=7) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 10 | 2 | 5 | 0 | 0 | 3 | 0 | 0.286 [0.08, 0.64] (n=7) | 0.286 [0.08, 0.64] (n=7) |
| best member < 0.9 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0.833 [0.44, 0.97] (n=6) | 0.833 [0.44, 0.97] (n=6) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
