# Height QC tests (#44): sao_paulo

30034 panos, 18601 with a measured depth height; implied height for 9331 (assoc per-pano), 9382 (assoc 2.3). k = 1.16.

Spread definition: `measured_planes_only` (read from depth/index.csv's header; since #47 the spread leaves Google's stand-in planes out -- docs/camera-height-study.md, 2026-09-27 addendum).

## T1: Theil–Sen slope of implied on depth, per vintage

| vintage | n (per-pano) | slope (per-pano) | n (2.3) | slope (2.3) |
|---|---:|---:|---:|---:|
| 2023 | 479 | 0.583 | 489 | 0.194 |
| 2024 | 2674 | 0.496 | 2732 | 0.088 |
| 2025 | 2238 | 0.514 | 2285 | 0.069 |
| **pano-weighted** | 5391 | 0.511 | 5506 | 0.090 |

Median implied height per deviation bin: bins.csv (test T1).

## T2: the low tail (depth < 1.5 m), median implied/depth

| group | n (per-pano) | ratio (per-pano) | n (2.3) | ratio (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 16 | 1.206 | 16 | 1.774 |
| tilt >= 6 | 30 | 1.219 | 29 | 2.006 |

## T3: candidate gates (median |resid|/implied, flagged vs unflagged)

| gate | assoc | flag rate | flagged | unflagged | ratio | pass |
|---|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | per-pano | 0.1060 | 0.1280 | 0.0888 | 1.442 | no |
| height_spread_m >= 0.3 | per-pano | 0.0915 | 0.0959 | 0.0898 | 1.068 | no |
| ground_pixel_share < 0.15 | per-pano | 0.0252 | 0.0957 | 0.0904 | 1.059 | no |
| depth_planes < 60 | per-pano | 0.0133 | 0.0606 | 0.0906 | 0.669 | no |
| sky_fraction > 0.6 | per-pano | 0.0000 | — | 0.0904 | — | no |
| |depth - vintage_median| >= 0.4 | per-pano | 0.0519 | 0.1226 | 0.0891 | 1.376 | no |
| ground_tilt_deg >= 6 | 2.3 | 0.1060 | 0.1436 | 0.0918 | 1.565 | no |
| height_spread_m >= 0.3 | 2.3 | 0.0915 | 0.1094 | 0.0933 | 1.173 | no |
| ground_pixel_share < 0.15 | 2.3 | 0.0252 | 0.0878 | 0.0942 | 0.932 | no |
| depth_planes < 60 | 2.3 | 0.0133 | 0.0611 | 0.0942 | 0.648 | no |
| sky_fraction > 0.6 | 2.3 | 0.0000 | — | 0.0941 | — | no |
| |depth - vintage_median| >= 0.4 | 2.3 | 0.0519 | 0.1890 | 0.0916 | 2.064 | no |

## T4: p68 of |resid| by height_spread_m quartile

| quartile | spread range (per-pano) | p68 (per-pano) | spread range (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.023 | 0.339 | 0.000–0.023 | 0.360 |
| Q2 | 0.023–0.082 | 0.335 | 0.023–0.082 | 0.359 |
| Q3 | 0.082–0.159 | 0.343 | 0.082–0.160 | 0.357 |
| Q4 | 0.160–1.588 | 0.341 | 0.160–1.588 | 0.365 |

Per-city numbers are context; the verdicts are read on the pooled cities (runs/_pooled/height_qc/report.md).
