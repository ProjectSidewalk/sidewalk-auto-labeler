# Height QC tests (#44): sao_paulo

30034 panos, 18601 with a measured depth height; implied height for 9334 (assoc per-pano), 9382 (assoc 2.3). k = 1.16.

## T1: Theil–Sen slope of implied on depth, per vintage

| vintage | n (per-pano) | slope (per-pano) | n (2.3) | slope (2.3) |
|---|---:|---:|---:|---:|
| 2023 | 479 | 0.576 | 489 | 0.194 |
| 2024 | 2674 | 0.495 | 2732 | 0.088 |
| 2025 | 2240 | 0.496 | 2285 | 0.069 |
| **pano-weighted** | 5393 | 0.503 | 5506 | 0.090 |

Median implied height per deviation bin: bins.csv (test T1).

## T2: the low tail (depth < 1.5 m), median implied/depth

| group | n (per-pano) | ratio (per-pano) | n (2.3) | ratio (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 16 | 1.206 | 16 | 1.774 |
| tilt >= 6 | 30 | 1.249 | 29 | 2.006 |

## T3: candidate gates (median |resid|/implied, flagged vs unflagged)

| gate | assoc | flag rate | flagged | unflagged | ratio | pass |
|---|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | per-pano | 0.1060 | 0.1267 | 0.0888 | 1.427 | no |
| height_spread_m >= 0.3 | per-pano | 0.1516 | 0.0994 | 0.0893 | 1.113 | no |
| ground_pixel_share < 0.15 | per-pano | 0.0252 | 0.0957 | 0.0904 | 1.059 | no |
| depth_planes < 60 | per-pano | 0.0133 | 0.0606 | 0.0906 | 0.669 | no |
| sky_fraction > 0.6 | per-pano | 0.0000 | — | 0.0905 | — | no |
| |depth - vintage_median| >= 0.4 | per-pano | 0.0519 | 0.1226 | 0.0891 | 1.376 | no |
| ground_tilt_deg >= 6 | 2.3 | 0.1060 | 0.1436 | 0.0918 | 1.565 | no |
| height_spread_m >= 0.3 | 2.3 | 0.1516 | 0.1129 | 0.0921 | 1.225 | no |
| ground_pixel_share < 0.15 | 2.3 | 0.0252 | 0.0878 | 0.0942 | 0.932 | no |
| depth_planes < 60 | 2.3 | 0.0133 | 0.0611 | 0.0942 | 0.648 | no |
| sky_fraction > 0.6 | 2.3 | 0.0000 | — | 0.0941 | — | no |
| |depth - vintage_median| >= 0.4 | 2.3 | 0.0519 | 0.1890 | 0.0916 | 2.064 | no |

## T4: p68 of |resid| by height_spread_m quartile

| quartile | spread range (per-pano) | p68 (per-pano) | spread range (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.053 | 0.324 | 0.000–0.054 | 0.348 |
| Q2 | 0.053–0.117 | 0.356 | 0.054–0.118 | 0.366 |
| Q3 | 0.117–0.204 | 0.336 | 0.118–0.206 | 0.350 |
| Q4 | 0.204–1.662 | 0.350 | 0.206–1.662 | 0.382 |

Per-city numbers are context; the verdicts are read on the pooled cities (runs/_pooled/height_qc/report.md).
