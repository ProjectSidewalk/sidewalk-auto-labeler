## sao_paulo: precision of mined positives (RampNet#158 step 1)

run: 22741 panos -> 16381 sites, 1921 with >= 3 operational panos
GT: 125 fully judged panos, 125 with the missed-ramp check attested; match radius 5 m; candidates within 15 m; camera height per-pano
excluded: 13 (site, pano) pairs where the pano is a member through a sub-threshold detection only

camera-height resolution (#56):
- camera height mode `per-pano`
- 18601 of 22741 panos took a measured height; 4140 fell back to the 2.6 m constant
- flagged by the #44 QC gate (kept all the same): {'flagged_qc:vintage_deviation': 922}
- 18601 heights read from depth/index.csv; spread definition: measured_planes_only

placement: mapa_posed_pair -- 21 of 21 candidates placed by the arm, 0 fell back and 0 did not reach the ground (both keep the flat projection of the site); median shift of a placed target from the site 2.09 m. Candidates, ranges and strata are the flat run's, so this is paired candidate by candidate with it.

OWN-SITE read: a right adjudication whose GT point is closer to another fused site than to the candidate's own counts as `other_site`, false under both denominators (other_site: 6).

Two denominators (see the module docstring). A miner has no verdicts, so it cannot filter `already_detected` out; those labels ship and they are correct.

- **hard-only** (new misses): 0.200 [0.07, 0.45] (n=15) -> read at <= 15 m: drop this label source
- **all-mined** (correct labels): 0.294 [0.13, 0.53] (n=17) -> read at <= 15 m: drop this label source; the 95% CI [0.13, 0.53] spans the drop/visibility bands, so this is not decisive

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| headline | 21 | 3 | 12 | 0 | 2 | 4 | 0 | 0.200 [0.07, 0.45] (n=15) | 0.294 [0.13, 0.53] (n=17) |

### cumulative by camera-to-site distance

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| <= 10 m | 7 | 0 | 4 | 0 | 2 | 1 | 0 | 0.000 [0.00, 0.49] (n=4) | 0.333 [0.10, 0.70] (n=6) |
| <= 15 m | 21 | 3 | 12 | 0 | 2 | 4 | 0 | 0.200 [0.07, 0.45] (n=15) | 0.294 [0.13, 0.53] (n=17) |

### by range bucket

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 0-8 m | 3 | 0 | 2 | 0 | 0 | 1 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.000 [0.00, 0.66] (n=2) |
| 8-12 m | 10 | 1 | 5 | 0 | 2 | 2 | 0 | 0.167 [0.03, 0.56] (n=6) | 0.375 [0.14, 0.69] (n=8) |
| 12-18 m | 8 | 2 | 5 | 0 | 0 | 1 | 0 | 0.286 [0.08, 0.64] (n=7) | 0.286 [0.08, 0.64] (n=7) |

### by site support (operational panos)

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| 3 panos | 12 | 2 | 7 | 0 | 1 | 2 | 0 | 0.222 [0.06, 0.55] (n=9) | 0.300 [0.11, 0.60] (n=10) |
| 4 panos | 5 | 1 | 3 | 0 | 1 | 0 | 0 | 0.250 [0.05, 0.70] (n=4) | 0.400 [0.12, 0.77] (n=5) |
| >= 5 panos | 4 | 0 | 2 | 0 | 0 | 2 | 0 | 0.000 [0.00, 0.66] (n=2) | 0.000 [0.00, 0.66] (n=2) |

### by best member confidence

| stratum | cand. | tp | fp | of which rej. det. | already det. | unsure | unadj. | precision hard-only [95% CI] | precision all-mined [95% CI] |
|---|--:|--:|--:|--:|--:|--:|--:|---|---|
| best member >= 0.9 | 12 | 2 | 8 | 0 | 1 | 1 | 0 | 0.200 [0.06, 0.51] (n=10) | 0.273 [0.10, 0.57] (n=11) |
| best member < 0.9 | 9 | 1 | 4 | 0 | 1 | 3 | 0 | 0.200 [0.04, 0.62] (n=5) | 0.333 [0.10, 0.70] (n=6) |

Read the confidence split with care: `best_conf` saturates (values run above 1.0), so the >= 0.9 bin holds most candidates, and it is confounded with support and range - stronger sites are seen by more panos, hence from further away, where precision falls for geometric reasons. It is not evidence that confident sites mine worse.

### mined yield over the whole run (#102 sanity check)

An upper bound: these are (strong site, non-member pano) pairs, and the `already_detected` column above shows a share of them are ramps the model already detects from that pano rather than misses.

| radius | (site, non-member pano) pairs | x operational detections |
|---|--:|--:|
| <= 10 m | 417 | 0.03x |
| <= 15 m | 1427 | 0.12x |
