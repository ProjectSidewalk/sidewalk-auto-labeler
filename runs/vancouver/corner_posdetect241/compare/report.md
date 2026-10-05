# Position-selected panos at the unobservable units (RampNet#241)

Written by `scripts/corner_posdetect.py compare`. Protocol and caveats: docs/corner-inventory.md, amendment A8.

- target units (unobservable in the #238 build, fusion arm, unit level): 1453
- added panos: 10367; their operational fused sites: 277
- target units with an added pano within 25 m (record position): 1450
- target units now (fusion): {'absent': 1239, 'present': 211, 'unobservable': 3}; (deployed, emulated for added panos): {'absent': 1421, 'present': 29, 'unobservable': 3}
- target units now, nearest added pano only (fusion): {'absent': 1401, 'present': 49, 'unobservable': 3}

## Decision rule (as pre-registered: clean read, unit level, fusion, primary, intersections pooled)

- before: 46/84 = 0.5476 [0.4414, 0.6496]
- after: 333/1323 = 0.2517 [0.2291, 0.2758] -> **FAIL: triage the false absences (detector miss vs observability) first**

## Unit state transitions, fusion arm (intersections)

| stratum | old | new | units |
|---|---|---|---:|
| arterial | absent | absent | 39 |
| arterial | present | present | 1076 |
| arterial | unobservable | absent | 332 |
| arterial | unobservable | present | 50 |
| residential | absent | absent | 43 |
| residential | present | present | 1537 |
| residential | unobservable | absent | 905 |
| residential | unobservable | present | 160 |
| residential | unobservable | unobservable | 3 |
| signalised | absent | absent | 2 |
| signalised | present | present | 279 |
| signalised | unobservable | absent | 2 |
| signalised | unobservable | present | 1 |

## Unit state transitions, deployed arm (intersections)

| stratum | old | new | units |
|---|---|---|---:|
| arterial | absent | absent | 38 |
| arterial | present | present | 1077 |
| arterial | unobservable | absent | 376 |
| arterial | unobservable | present | 5 |
| arterial | unobservable | unobservable | 1 |
| residential | absent | absent | 41 |
| residential | present | present | 1541 |
| residential | unobservable | absent | 1043 |
| residential | unobservable | present | 16 |
| residential | unobservable | unobservable | 7 |
| signalised | absent | absent | 2 |
| signalised | present | present | 278 |
| signalised | unobservable | absent | 3 |
| signalised | unobservable | unobservable | 1 |

## Absence reads

a = clean read, fusion arm (the rule as written); b = the same, not counting the city's `NA` points with no `RAMPTYPE`; c = clean read, deployed arm (emulated for added panos). `no Available` = absent with no `Available` point.

| subset | stratum | build | read | absent | clean | share [95% CI] | no Available | share [95% CI] |
|---|---|---|---|---:|---:|---|---:|---|
| all | intersections | old | a_fusion_clean_as_written | 84 | 46 | 0.548 [0.441, 0.650] | 75 | 0.893 [0.809, 0.943] |
| all | intersections | old | b_fusion_clean_ignoring_NA_noramp | 84 | 75 | 0.893 [0.809, 0.943] | 75 | 0.893 [0.809, 0.943] |
| all | intersections | old | c_deployed_clean | 81 | 42 | 0.518 [0.411, 0.624] | 73 | 0.901 [0.817, 0.949] |
| all | intersections | new | a_fusion_clean_as_written | 1323 | 333 | 0.252 [0.229, 0.276] | 1279 | 0.967 [0.956, 0.975] |
| all | intersections | new | b_fusion_clean_ignoring_NA_noramp | 1323 | 1277 | 0.965 [0.954, 0.974] | 1279 | 0.967 [0.956, 0.975] |
| all | intersections | new | c_deployed_clean | 1503 | 365 | 0.243 [0.222, 0.265] | 1437 | 0.956 [0.945, 0.965] |
| all | signalised | old | a_fusion_clean_as_written | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| all | signalised | old | b_fusion_clean_ignoring_NA_noramp | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| all | signalised | old | c_deployed_clean | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| all | signalised | new | a_fusion_clean_as_written | 4 | 3 | 0.750 [0.301, 0.954] | 4 | 1.000 [0.510, 1.000] |
| all | signalised | new | b_fusion_clean_ignoring_NA_noramp | 4 | 4 | 1.000 [0.510, 1.000] | 4 | 1.000 [0.510, 1.000] |
| all | signalised | new | c_deployed_clean | 5 | 4 | 0.800 [0.376, 0.964] | 5 | 1.000 [0.566, 1.000] |
| all | arterial | old | a_fusion_clean_as_written | 39 | 29 | 0.744 [0.589, 0.854] | 33 | 0.846 [0.703, 0.927] |
| all | arterial | old | b_fusion_clean_ignoring_NA_noramp | 39 | 33 | 0.846 [0.703, 0.927] | 33 | 0.846 [0.703, 0.927] |
| all | arterial | old | c_deployed_clean | 38 | 28 | 0.737 [0.580, 0.850] | 34 | 0.895 [0.759, 0.958] |
| all | arterial | new | a_fusion_clean_as_written | 371 | 190 | 0.512 [0.461, 0.563] | 352 | 0.949 [0.921, 0.967] |
| all | arterial | new | b_fusion_clean_ignoring_NA_noramp | 371 | 351 | 0.946 [0.918, 0.965] | 352 | 0.949 [0.921, 0.967] |
| all | arterial | new | c_deployed_clean | 414 | 200 | 0.483 [0.435, 0.531] | 387 | 0.935 [0.907, 0.955] |
| all | residential | old | a_fusion_clean_as_written | 43 | 15 | 0.349 [0.224, 0.498] | 40 | 0.930 [0.814, 0.976] |
| all | residential | old | b_fusion_clean_ignoring_NA_noramp | 43 | 40 | 0.930 [0.814, 0.976] | 40 | 0.930 [0.814, 0.976] |
| all | residential | old | c_deployed_clean | 41 | 12 | 0.293 [0.176, 0.445] | 37 | 0.902 [0.774, 0.961] |
| all | residential | new | a_fusion_clean_as_written | 948 | 140 | 0.148 [0.127, 0.172] | 923 | 0.974 [0.961, 0.982] |
| all | residential | new | b_fusion_clean_ignoring_NA_noramp | 948 | 922 | 0.973 [0.960, 0.981] | 923 | 0.974 [0.961, 0.982] |
| all | residential | new | c_deployed_clean | 1084 | 161 | 0.148 [0.129, 0.171] | 1045 | 0.964 [0.951, 0.974] |
| old_absent | intersections | old | a_fusion_clean_as_written | 84 | 46 | 0.548 [0.441, 0.650] | 75 | 0.893 [0.809, 0.943] |
| old_absent | intersections | old | b_fusion_clean_ignoring_NA_noramp | 84 | 75 | 0.893 [0.809, 0.943] | 75 | 0.893 [0.809, 0.943] |
| old_absent | intersections | old | c_deployed_clean | 69 | 39 | 0.565 [0.448, 0.676] | 63 | 0.913 [0.823, 0.960] |
| old_absent | intersections | new | a_fusion_clean_as_written | 84 | 46 | 0.548 [0.441, 0.650] | 75 | 0.893 [0.809, 0.943] |
| old_absent | intersections | new | b_fusion_clean_ignoring_NA_noramp | 84 | 75 | 0.893 [0.809, 0.943] | 75 | 0.893 [0.809, 0.943] |
| old_absent | intersections | new | c_deployed_clean | 69 | 39 | 0.565 [0.448, 0.676] | 63 | 0.913 [0.823, 0.960] |
| old_absent | signalised | old | a_fusion_clean_as_written | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| old_absent | signalised | old | b_fusion_clean_ignoring_NA_noramp | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| old_absent | signalised | old | c_deployed_clean | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| old_absent | signalised | new | a_fusion_clean_as_written | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| old_absent | signalised | new | b_fusion_clean_ignoring_NA_noramp | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| old_absent | signalised | new | c_deployed_clean | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| old_absent | arterial | old | a_fusion_clean_as_written | 39 | 29 | 0.744 [0.589, 0.854] | 33 | 0.846 [0.703, 0.927] |
| old_absent | arterial | old | b_fusion_clean_ignoring_NA_noramp | 39 | 33 | 0.846 [0.703, 0.927] | 33 | 0.846 [0.703, 0.927] |
| old_absent | arterial | old | c_deployed_clean | 34 | 27 | 0.794 [0.632, 0.896] | 31 | 0.912 [0.770, 0.970] |
| old_absent | arterial | new | a_fusion_clean_as_written | 39 | 29 | 0.744 [0.589, 0.854] | 33 | 0.846 [0.703, 0.927] |
| old_absent | arterial | new | b_fusion_clean_ignoring_NA_noramp | 39 | 33 | 0.846 [0.703, 0.927] | 33 | 0.846 [0.703, 0.927] |
| old_absent | arterial | new | c_deployed_clean | 34 | 27 | 0.794 [0.632, 0.896] | 31 | 0.912 [0.770, 0.970] |
| old_absent | residential | old | a_fusion_clean_as_written | 43 | 15 | 0.349 [0.224, 0.498] | 40 | 0.930 [0.814, 0.976] |
| old_absent | residential | old | b_fusion_clean_ignoring_NA_noramp | 43 | 40 | 0.930 [0.814, 0.976] | 40 | 0.930 [0.814, 0.976] |
| old_absent | residential | old | c_deployed_clean | 33 | 10 | 0.303 [0.174, 0.473] | 30 | 0.909 [0.764, 0.969] |
| old_absent | residential | new | a_fusion_clean_as_written | 43 | 15 | 0.349 [0.224, 0.498] | 40 | 0.930 [0.814, 0.976] |
| old_absent | residential | new | b_fusion_clean_ignoring_NA_noramp | 43 | 40 | 0.930 [0.814, 0.976] | 40 | 0.930 [0.814, 0.976] |
| old_absent | residential | new | c_deployed_clean | 33 | 10 | 0.303 [0.174, 0.473] | 30 | 0.909 [0.764, 0.969] |
| moved | intersections | new | a_fusion_clean_as_written | 1239 | 287 | 0.232 [0.209, 0.256] | 1204 | 0.972 [0.961, 0.980] |
| moved | intersections | new | b_fusion_clean_ignoring_NA_noramp | 1239 | 1202 | 0.970 [0.959, 0.978] | 1204 | 0.972 [0.961, 0.980] |
| moved | intersections | new | c_deployed_clean | 1421 | 323 | 0.227 [0.206, 0.250] | 1364 | 0.960 [0.948, 0.969] |
| moved | signalised | new | a_fusion_clean_as_written | 2 | 1 | 0.500 [0.095, 0.905] | 2 | 1.000 [0.342, 1.000] |
| moved | signalised | new | b_fusion_clean_ignoring_NA_noramp | 2 | 2 | 1.000 [0.342, 1.000] | 2 | 1.000 [0.342, 1.000] |
| moved | signalised | new | c_deployed_clean | 3 | 2 | 0.667 [0.208, 0.939] | 3 | 1.000 [0.439, 1.000] |
| moved | arterial | new | a_fusion_clean_as_written | 332 | 161 | 0.485 [0.432, 0.539] | 319 | 0.961 [0.934, 0.977] |
| moved | arterial | new | b_fusion_clean_ignoring_NA_noramp | 332 | 318 | 0.958 [0.930, 0.975] | 319 | 0.961 [0.934, 0.977] |
| moved | arterial | new | c_deployed_clean | 375 | 172 | 0.459 [0.409, 0.509] | 353 | 0.941 [0.913, 0.961] |
| moved | residential | new | a_fusion_clean_as_written | 905 | 125 | 0.138 [0.117, 0.162] | 883 | 0.976 [0.964, 0.984] |
| moved | residential | new | b_fusion_clean_ignoring_NA_noramp | 905 | 882 | 0.975 [0.962, 0.983] | 883 | 0.976 [0.964, 0.984] |
| moved | residential | new | c_deployed_clean | 1043 | 149 | 0.143 [0.123, 0.165] | 1008 | 0.966 [0.954, 0.976] |
| moved_nearest_only | intersections | new | a_fusion_clean_as_written | 1401 | 317 | 0.226 [0.205, 0.249] | 1352 | 0.965 [0.954, 0.973] |
| moved_nearest_only | signalised | new | a_fusion_clean_as_written | 3 | 2 | 0.667 [0.208, 0.939] | 3 | 1.000 [0.439, 1.000] |
| moved_nearest_only | arterial | new | a_fusion_clean_as_written | 370 | 171 | 0.462 [0.412, 0.513] | 353 | 0.954 [0.928, 0.971] |
| moved_nearest_only | residential | new | a_fusion_clean_as_written | 1028 | 144 | 0.140 [0.120, 0.163] | 996 | 0.969 [0.956, 0.978] |

## Capture years

| year | added panos | #56 run panos |
|---|---:|---:|
| 2007 | 17 | 0 |
| 2011 | 176 | 197 |
| 2012 | 191 | 310 |
| 2014 | 665 | 722 |
| 2015 | 24 | 277 |
| 2016 | 102 | 319 |
| 2017 | 15 | 258 |
| 2018 | 88 | 953 |
| 2019 | 500 | 1663 |
| 2021 | 239 | 926 |
| 2022 | 2624 | 3434 |
| 2023 | 3414 | 8653 |
| 2024 | 2182 | 10621 |
| 2025 | 92 | 497 |
| 2026 | 38 | 0 |

Added capture dates, quantiles: {'0.0': '2007-08', '0.25': '2022-11', '0.5': '2023-04', '0.75': '2023-05', '1.0': '2026-07'}
