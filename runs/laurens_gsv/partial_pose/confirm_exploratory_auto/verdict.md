# EXPLORATORY #116 confirmatory run: laurens_gsv @ auto

**EXPLORATORY: this city was in #116's train set, so this output proves the code path and is not a confirmation.**

Frozen constants k_pitch 0.183, k_roll 0.382 (geo.PARTIAL_POSE_K_GSV); whole run, 2137 panos, tier 0.3, 25 m cap, seed 116 (the shuffle kept its own tilt on 10 panos). Production's pose equals the study arm's on all 2137 panos.

Store-pose gate: **not_required** -- no store-built panos: every angle is streetlevel's own; `store_pose_gate.json`.

## Outcome: EXPLORATORY PASS

- (i) partial (1236 pairs): median -0.173 m vs off, -0.147 m vs partial-shuffled; p90 -0.373 vs off, -0.259 vs partial-shuffled -> ok
- (ii) partial: n = 193, lost 2, gained 3, L = -1; k*(n) = 5 -> ok
- (iii) partial: unplaceable GT marks -1 (limit 9.7) -> ok
- (iv) bend partial: inventory median -0.010 m, p90 -0.051 m vs off (limit +0.10) -> ok
- (iv) gainesville partial: inventory median -0.034 m, p90 -0.050 m vs off (limit +0.10) -> ok
- clause (i): PASS
- clause (ii): PASS
- clause (iii): PASS
- clause (iv): PASS

## Pair distance, reassoc (m)

| arm | multi_view_sites | pairs_scored | median_m | mean_m | p90_m |
|---|---|---|---|---|---|
| off | 192 | 1236 | 1.749 | 2.033 | 3.908 |
| partial | 190 | 1236 | 1.576 | 1.854 | 3.535 |
| partial-shuffled | 190 | 1236 | 1.724 | 1.977 | 3.794 |
| partial-mirror | 196 | 1236 | 1.957 | 2.294 | 4.402 |

## RampNet GT survivorship (off pool)

| arm | gt_marks | gt_marks_unplaceable | off_pool_ramps | recall_off_pool_2p5m | lost_vs_off_2p5m | gained_vs_off_2p5m |
|---|---|---|---|---|---|---|
| off | 220 | 24 | 193 | 0.907 | 0 | 0 |
| partial | 220 | 23 | 193 | 0.912 | 2 | 3 |
| partial-shuffled | 220 | 23 | 193 | 0.907 | 1 | 1 |
| partial-mirror | 220 | 23 | 193 | 0.886 | 4 | 0 |

GT panos judged: 86

## Operational members not placed inside 25 m

| arm | unplaceable | unplaceable_vs_off |
|---|---|---|
| off | 40 | 0 |
| partial | 40 | 0 |
| partial-shuffled | 40 | 0 |
| partial-mirror | 40 | 0 |

## Inventories (frozen@off, pool on off, 5 m)

| city | role | arm | n_pool | median_m | p90_m |
|---|---|---|---|---|---|
| bend | referee | off | 10866 | 0.621 | 1.358 |
| bend | referee | partial | 10866 | 0.611 | 1.307 |
| bend | referee | partial-shuffled | 10866 | 0.622 | 1.361 |
| bend | referee | partial-mirror | 10866 | 0.662 | 1.476 |
| gainesville | referee | off | 2707 | 1.120 | 2.784 |
| gainesville | referee | partial | 2707 | 1.086 | 2.733 |
| gainesville | referee | partial-shuffled | 2707 | 1.119 | 2.803 |
| gainesville | referee | partial-mirror | 2707 | 1.175 | 2.855 |

## k per capture vintage (descriptive; never scored)

| vintage | n_rows | n_panos | k_pitch | se_pitch | k_roll | se_roll | intercept |
|---|---|---|---|---|---|---|---|
| all | 1962 | 293 | 0.329 | 0.067 | 0.523 | 0.062 | -0.450 |
| 2021 | 17 | 3 |  |  |  |  |  |
| 2024 | 1945 | 290 | 0.329 | 0.068 | 0.520 | 0.062 | -0.452 |
