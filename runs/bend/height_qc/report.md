# Height QC tests (#44): bend

78560 panos, 65866 with a measured depth height; implied height for 19123 (assoc per-pano), 19082 (assoc 2.3). k = 1.06.

Spread definition: `measured_planes_only` (read from depth/index.csv's header; since #47 the spread leaves Google's stand-in planes out -- docs/camera-height-study.md, 2026-09-27 addendum).

## T1: Theil–Sen slope of implied on depth, per vintage

| vintage | n (per-pano) | slope (per-pano) | n (2.3) | slope (2.3) |
|---|---:|---:|---:|---:|
| 2019 | 358 | -0.011 | 361 | -0.093 |
| 2024 | 15669 | 0.082 | 15675 | 0.075 |
| 2025 | 503 | 0.343 | 503 | 0.192 |
| **pano-weighted** | 16530 | 0.088 | 16539 | 0.075 |

Median implied height per deviation bin: bins.csv (test T1).

## T2: the low tail (depth < 1.5 m), median implied/depth

| group | n (per-pano) | ratio (per-pano) | n (2.3) | ratio (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 1 | 1.852 | 1 | 1.852 |
| tilt >= 6 | 2 | 1.431 | 1 | 1.653 |

## T3: candidate gates (median |resid|/implied, flagged vs unflagged)

| gate | assoc | flag rate | flagged | unflagged | ratio | pass |
|---|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | per-pano | 0.0193 | 0.0821 | 0.0594 | 1.381 | no |
| height_spread_m >= 0.3 | per-pano | 0.0641 | 0.0700 | 0.0592 | 1.182 | no |
| ground_pixel_share < 0.15 | per-pano | 0.0183 | 0.0653 | 0.0595 | 1.098 | no |
| depth_planes < 60 | per-pano | 0.0142 | 0.0583 | 0.0597 | 0.977 | no |
| sky_fraction > 0.6 | per-pano | 0.0000 | — | 0.0597 | — | no |
| |depth - vintage_median| >= 0.4 | per-pano | 0.0156 | 0.1771 | 0.0590 | 3.002 | yes |
| ground_tilt_deg >= 6 | 2.3 | 0.0193 | 0.0837 | 0.0600 | 1.394 | no |
| height_spread_m >= 0.3 | 2.3 | 0.0641 | 0.0715 | 0.0598 | 1.196 | no |
| ground_pixel_share < 0.15 | 2.3 | 0.0183 | 0.0632 | 0.0601 | 1.050 | no |
| depth_planes < 60 | 2.3 | 0.0142 | 0.0612 | 0.0602 | 1.017 | no |
| sky_fraction > 0.6 | 2.3 | 0.0000 | — | 0.0602 | — | no |
| |depth - vintage_median| >= 0.4 | 2.3 | 0.0156 | 0.2071 | 0.0594 | 3.484 | yes |

## T4: p68 of |resid| by height_spread_m quartile

| quartile | spread range (per-pano) | p68 (per-pano) | spread range (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.027 | 0.219 | 0.000–0.027 | 0.223 |
| Q2 | 0.027–0.080 | 0.222 | 0.027–0.080 | 0.225 |
| Q3 | 0.080–0.146 | 0.233 | 0.080–0.146 | 0.234 |
| Q4 | 0.146–1.300 | 0.237 | 0.146–1.300 | 0.240 |

Per-city numbers are context; the verdicts are read on the pooled cities (runs/_pooled/height_qc/report.md).
