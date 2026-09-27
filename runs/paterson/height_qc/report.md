# Height QC tests (#44): paterson

34687 panos, 30325 with a measured depth height; implied height for 11054 (assoc per-pano), 10746 (assoc 2.3). k = 1.08.

Spread definition: `measured_planes_only` (read from depth/index.csv's header; since #47 the spread leaves Google's stand-in planes out -- docs/camera-height-study.md, 2026-09-27 addendum).

## T1: Theil–Sen slope of implied on depth, per vintage

| vintage | n (per-pano) | slope (per-pano) | n (2.3) | slope (2.3) |
|---|---:|---:|---:|---:|
| 2018 | 452 | 0.285 | 449 | 0.139 |
| 2019 | 444 | 0.568 | 446 | 0.447 |
| 2020 | 929 | 0.216 | 923 | 0.091 |
| 2021 | 575 | 0.435 | 571 | 0.337 |
| 2024 | 2516 | 0.355 | 2517 | 0.208 |
| 2025 | 5169 | 0.741 | 4887 | 0.178 |
| **pano-weighted** | 10085 | 0.551 | 9793 | 0.197 |

Median implied height per deviation bin: bins.csv (test T1).

## T2: the low tail (depth < 1.5 m), median implied/depth

| group | n (per-pano) | ratio (per-pano) | n (2.3) | ratio (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 43 | 1.247 | 37 | 1.427 |
| tilt >= 6 | 24 | 1.299 | 23 | 1.614 |

## T3: candidate gates (median |resid|/implied, flagged vs unflagged)

| gate | assoc | flag rate | flagged | unflagged | ratio | pass |
|---|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | per-pano | 0.0462 | 0.1052 | 0.0705 | 1.491 | no |
| height_spread_m >= 0.3 | per-pano | 0.0745 | 0.0947 | 0.0700 | 1.354 | no |
| ground_pixel_share < 0.15 | per-pano | 0.0163 | 0.0823 | 0.0713 | 1.154 | no |
| depth_planes < 60 | per-pano | 0.0227 | 0.0880 | 0.0711 | 1.237 | no |
| sky_fraction > 0.6 | per-pano | 0.0000 | — | 0.0715 | — | no |
| |depth - vintage_median| >= 0.4 | per-pano | 0.0162 | 0.1509 | 0.0708 | 2.132 | yes |
| ground_tilt_deg >= 6 | 2.3 | 0.0462 | 0.1171 | 0.0689 | 1.699 | no |
| height_spread_m >= 0.3 | 2.3 | 0.0745 | 0.1025 | 0.0686 | 1.494 | no |
| ground_pixel_share < 0.15 | 2.3 | 0.0163 | 0.0757 | 0.0695 | 1.088 | no |
| depth_planes < 60 | 2.3 | 0.0227 | 0.0868 | 0.0695 | 1.249 | no |
| sky_fraction > 0.6 | 2.3 | 0.0000 | — | 0.0697 | — | no |
| |depth - vintage_median| >= 0.4 | 2.3 | 0.0162 | 0.2183 | 0.0690 | 3.161 | yes |

## T4: p68 of |resid| by height_spread_m quartile

| quartile | spread range (per-pano) | p68 (per-pano) | spread range (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.026 | 0.244 | 0.000–0.026 | 0.238 |
| Q2 | 0.026–0.082 | 0.242 | 0.026–0.081 | 0.235 |
| Q3 | 0.082–0.152 | 0.234 | 0.081–0.152 | 0.235 |
| Q4 | 0.152–1.055 | 0.261 | 0.152–1.055 | 0.273 |

Per-city numbers are context; the verdicts are read on the pooled cities (runs/_pooled/height_qc/report.md).
