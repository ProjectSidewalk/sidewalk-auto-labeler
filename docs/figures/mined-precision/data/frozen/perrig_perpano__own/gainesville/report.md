## gainesville: precision of mined positives (RampNet#158 step 1)

run: 35204 panos -> 14879 sites, 2310 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 12 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 30458 of 35204 panos took a measured height; 4746 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 3112}
- 30458 heights read from depth/index.csv; spread definition: measured_planes_only

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 5).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.125 [0.02, 0.47] (n=8) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.125 [0.02, 0.47] (n=8) -> read at <= 15 m: drop this label source

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 8 | 1 | 7 | 0 | 0 | 0 | 0 | 0.125 [0.02, 0.47] (n=8) | 0.125 [0.02, 0.47] (n=8) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 4 | 1 | 3 | 0 | 0 | 0 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.250 [0.05, 0.70] (n=4) |
| <= 15 m | 8 | 1 | 7 | 0 | 0 | 0 | 0 | 0.125 [0.02, 0.47] (n=8) | 0.125 [0.02, 0.47] (n=8) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0.500 [0.09, 0.91] (n=2) | 0.500 [0.09, 0.91] (n=2) |
| 8-12 m | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.000 [0.00, 0.66] (n=2) |
| 12-18 m | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.49] (n=4) | 0.000 [0.00, 0.49] (n=4) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 6 | 1 | 5 | 0 | 0 | 0 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.167 [0.03, 0.56] (n=6) |
| 4 panos | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.000 [0.00, 0.66] (n=2) |
| >= 5 panos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0.000 [0.00, 0.56] (n=3) | 0.000 [0.00, 0.56] (n=3) |
| best member < 0.9 | 5 | 1 | 4 | 0 | 0 | 0 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.200 [0.04, 0.62] (n=5) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 244 | 0.02x |
| <= 15 m | 872 | 0.06x |
