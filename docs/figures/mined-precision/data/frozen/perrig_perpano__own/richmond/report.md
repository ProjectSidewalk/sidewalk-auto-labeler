## richmond: precision of mined positives (RampNet#158 step 1)

run: 9091 panos -> 1570 sites, 945 with >= 3 operational panos
GT: 124 fully judged panos, 124 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-rig
excluded: 0 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-rig`
- 0 of 9091 panos took a per-rig height; 9091 fell back to the 2.6 m constant -- ALL of them, so this frame equals the constant one

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 9).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.224 [0.13, 0.36] (n=49) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.255 [0.16, 0.39] (n=51) -> read at <= 15 m: drop this label source

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 59 | 11 | 38 | 0 | 2 | 8 | 0 | 0.224 [0.13, 0.36] (n=49) | 0.255 [0.16, 0.39] (n=51) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 28 | 8 | 14 | 0 | 1 | 5 | 0 | 0.364 [0.20, 0.57] (n=22) | 0.391 [0.22, 0.59] (n=23) |
| <= 15 m | 59 | 11 | 38 | 0 | 2 | 8 | 0 | 0.224 [0.13, 0.36] (n=49) | 0.255 [0.16, 0.39] (n=51) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 20 | 6 | 8 | 0 | 1 | 5 | 0 | 0.429 [0.21, 0.67] (n=14) | 0.467 [0.25, 0.70] (n=15) |
| 8-12 m | 16 | 3 | 12 | 0 | 0 | 1 | 0 | 0.200 [0.07, 0.45] (n=15) | 0.200 [0.07, 0.45] (n=15) |
| 12-18 m | 23 | 2 | 18 | 0 | 1 | 2 | 0 | 0.100 [0.03, 0.30] (n=20) | 0.143 [0.05, 0.35] (n=21) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 11 | 2 | 8 | 0 | 0 | 1 | 0 | 0.200 [0.06, 0.51] (n=10) | 0.200 [0.06, 0.51] (n=10) |
| 4 panos | 11 | 4 | 7 | 0 | 0 | 0 | 0 | 0.364 [0.15, 0.65] (n=11) | 0.364 [0.15, 0.65] (n=11) |
| >= 5 panos | 37 | 5 | 23 | 0 | 2 | 7 | 0 | 0.179 [0.08, 0.36] (n=28) | 0.233 [0.12, 0.41] (n=30) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 44 | 5 | 31 | 0 | 2 | 6 | 0 | 0.139 [0.06, 0.29] (n=36) | 0.184 [0.09, 0.33] (n=38) |
| best member < 0.9 | 15 | 6 | 7 | 0 | 0 | 2 | 0 | 0.462 [0.23, 0.71] (n=13) | 0.462 [0.23, 0.71] (n=13) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 1210 | 0.13x |
| <= 15 m | 3011 | 0.32x |
