# Height QC tests (#44): bend

78560 panos, 65866 with a measured depth height; implied height for 19127 (assoc per-pano), 19082 (assoc 2.3). k = 1.06.

## T1: Theil–Sen slope of implied on depth, per vintage

| vintage | n (per-pano) | slope (per-pano) | n (2.3) | slope (2.3) |
|---|---:|---:|---:|---:|
| 2019 | 358 | -0.011 | 361 | -0.093 |
| 2024 | 15673 | 0.229 | 15675 | 0.075 |
| 2025 | 503 | 0.342 | 503 | 0.192 |
| **pano-weighted** | 16534 | 0.227 | 16539 | 0.075 |

Median implied height per deviation bin: bins.csv (test T1).

## T2: the low tail (depth < 1.5 m), median implied/depth

| group | n (per-pano) | ratio (per-pano) | n (2.3) | ratio (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 1 | 1.852 | 1 | 1.852 |
| tilt >= 6 | 2 | 1.431 | 1 | 1.653 |

## T3: candidate gates (median |resid|/implied, flagged vs unflagged)

| gate | assoc | flag rate | flagged | unflagged | ratio | pass |
|---|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | per-pano | 0.0193 | 0.0822 | 0.0595 | 1.382 | no |
| height_spread_m >= 0.3 | per-pano | 0.0715 | 0.0727 | 0.0590 | 1.232 | no |
| ground_pixel_share < 0.15 | per-pano | 0.0183 | 0.0653 | 0.0595 | 1.098 | no |
| depth_planes < 60 | per-pano | 0.0142 | 0.0583 | 0.0597 | 0.977 | no |
| sky_fraction > 0.6 | per-pano | 0.0000 | — | 0.0597 | — | no |
| |depth - vintage_median| >= 0.4 | per-pano | 0.0156 | 0.1790 | 0.0590 | 3.034 | yes |
| ground_tilt_deg >= 6 | 2.3 | 0.0193 | 0.0837 | 0.0600 | 1.394 | no |
| height_spread_m >= 0.3 | 2.3 | 0.0715 | 0.0736 | 0.0596 | 1.235 | no |
| ground_pixel_share < 0.15 | 2.3 | 0.0183 | 0.0632 | 0.0601 | 1.050 | no |
| depth_planes < 60 | 2.3 | 0.0142 | 0.0612 | 0.0602 | 1.017 | no |
| sky_fraction > 0.6 | 2.3 | 0.0000 | — | 0.0602 | — | no |
| |depth - vintage_median| >= 0.4 | 2.3 | 0.0156 | 0.2071 | 0.0594 | 3.484 | yes |

## T4: p68 of |resid| by height_spread_m quartile

| quartile | spread range (per-pano) | p68 (per-pano) | spread range (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.065 | 0.220 | 0.000–0.064 | 0.227 |
| Q2 | 0.065–0.116 | 0.222 | 0.064–0.116 | 0.223 |
| Q3 | 0.116–0.175 | 0.227 | 0.116–0.175 | 0.228 |
| Q4 | 0.175–1.300 | 0.243 | 0.175–1.300 | 0.244 |

Per-city numbers are context; the verdicts are read on the pooled cities (runs/_pooled/height_qc/report.md).
