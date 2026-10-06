# Thinning experiment — thinexp_richmond

mapillary; 1320 panos processed un-thinned (1320 in scan.json); 119 model-derived ramp sites from detections >= 0.3 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1320 | 0.55 | 119 | 119.0 | 119 | 76 | 76.0 | 76 | 86 | 86.0 |
| 2.5 | 487 | 0.2 | 110 | 91.0 | 119 | 75 | 73.1 | 76 | 67 | 65.6 |
| 5 | 363 | 0.15 | 103 | 83.2 | 119 | 73 | 70.2 | 76 | 62 | 58.9 |
| 7.5 | 284 | 0.12 | 91 | 76.5 | 119 | 69 | 66.8 | 76 | 56 | 53.4 |
| 10 | 230 | 0.1 | 85 | 72.0 | 119 | 66 | 63.6 | 76 | 53 | 47.7 |
| 15 | 169 | 0.07 | 75 | 64.3 | 119 | 62 | 58.1 | 76 | 44 | 39.5 |
| 20 | 131 | 0.05 | 68 | 58.2 | 119 | 58 | 53.4 | 76 | 42 | 34.0 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 95 | 0.147 |
| 4-8 | 452 | 0.469 |
| 8-12 | 1022 | 0.453 |
| 12-16 | 1194 | 0.379 |
| 16-20 | 1305 | 0.182 |
