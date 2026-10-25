# Thinning experiment — thinexp_lyon_a

panoramax; 4226 panos processed un-thinned (4226 in scan.json); 285 model-derived ramp sites from detections >= 0.55 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4226 | 1.76 | 285 | 285.0 | 285 | 131 | 131.0 | 131 | 176 | 176.0 |
| 2.5 | 2146 | 0.89 | 271 | 217.5 | 285 | 131 | 127.2 | 131 | 157 | 127.0 |
| 5 | 1107 | 0.46 | 246 | 164.7 | 285 | 127 | 116.0 | 131 | 137 | 96.2 |
| 7.5 | 702 | 0.29 | 217 | 136.4 | 285 | 119 | 103.8 | 131 | 102 | 78.7 |
| 10 | 485 | 0.2 | 187 | 116.2 | 285 | 107 | 93.8 | 131 | 68 | 66.9 |
| 15 | 290 | 0.12 | 127 | 94.7 | 285 | 71 | 80.1 | 131 | 24 | 52.3 |
| 20 | 186 | 0.08 | 89 | 77.2 | 285 | 54 | 68.0 | 131 | 12 | 39.3 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 912 | 0.325 |
| 4-8 | 3760 | 0.321 |
| 8-12 | 5167 | 0.214 |
| 12-16 | 5897 | 0.105 |
| 16-20 | 6315 | 0.055 |
