# Height QC tests (#44): gainesville

37435 panos, 30458 with a measured depth height; implied height for 9375 (assoc per-pano), 9150 (assoc 2.3). k = 1.1.

## T1: Theil–Sen slope of implied on depth, per vintage

| vintage | n (per-pano) | slope (per-pano) | n (2.3) | slope (2.3) |
|---|---:|---:|---:|---:|
| 2022 | 302 | 0.221 | 300 | 0.098 |
| 2023 | 382 | 0.362 | 377 | 0.143 |
| 2024 | 908 | 0.137 | 920 | -0.084 |
| 2025 | 348 | 0.134 | 353 | -0.028 |
| 2026 | 5486 | 0.502 | 5248 | 0.071 |
| **pano-weighted** | 7426 | 0.422 | 7198 | 0.051 |

Median implied height per deviation bin: bins.csv (test T1).

## T2: the low tail (depth < 1.5 m), median implied/depth

| group | n (per-pano) | ratio (per-pano) | n (2.3) | ratio (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 453 | 1.308 | 508 | 1.527 |
| tilt >= 6 | 6 | 1.415 | 8 | 1.755 |

## T3: candidate gates (median |resid|/implied, flagged vs unflagged)

| gate | assoc | flag rate | flagged | unflagged | ratio | pass |
|---|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | per-pano | 0.0093 | 0.1345 | 0.0919 | 1.464 | no |
| height_spread_m >= 0.3 | per-pano | 0.0233 | 0.1037 | 0.0919 | 1.128 | no |
| ground_pixel_share < 0.15 | per-pano | 0.0074 | 0.0695 | 0.0923 | 0.752 | no |
| depth_planes < 60 | per-pano | 0.0668 | 0.0943 | 0.0919 | 1.026 | no |
| sky_fraction > 0.6 | per-pano | 0.0000 | — | 0.0921 | — | no |
| |depth - vintage_median| >= 0.4 | per-pano | 0.1106 | 0.2093 | 0.0895 | 2.339 | no |
| ground_tilt_deg >= 6 | 2.3 | 0.0093 | 0.1691 | 0.0950 | 1.780 | no |
| height_spread_m >= 0.3 | 2.3 | 0.0233 | 0.1165 | 0.0949 | 1.228 | no |
| ground_pixel_share < 0.15 | 2.3 | 0.0074 | 0.0803 | 0.0952 | 0.843 | no |
| depth_planes < 60 | 2.3 | 0.0668 | 0.0975 | 0.0950 | 1.026 | no |
| sky_fraction > 0.6 | 2.3 | 0.0000 | — | 0.0951 | — | no |
| |depth - vintage_median| >= 0.4 | 2.3 | 0.1106 | 0.3264 | 0.0900 | 3.626 | no |

## T4: p68 of |resid| by height_spread_m quartile

| quartile | spread range (per-pano) | p68 (per-pano) | spread range (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.004 | 0.293 | 0.000–0.004 | 0.328 |
| Q2 | 0.004–0.044 | 0.285 | 0.004–0.044 | 0.317 |
| Q3 | 0.044–0.099 | 0.292 | 0.044–0.100 | 0.314 |
| Q4 | 0.099–1.182 | 0.283 | 0.100–1.182 | 0.297 |

Per-city numbers are context; the verdicts are read on the pooled cities (runs/_pooled/height_qc/report.md).
