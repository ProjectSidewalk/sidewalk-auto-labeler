# Thinning experiment — thinexp_bayonne

panoramax; 3880 panos processed un-thinned (3880 in scan.json); 271 model-derived ramp sites from detections >= 0.3 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3880 | 1.62 | 271 | 271.0 | 271 | 95 | 95.0 | 95 | 142 | 142.0 |
| 2.5 | 3392 | 1.41 | 252 | 253.7 | 271 | 95 | 94.9 | 95 | 129 | 131.1 |
| 5 | 2455 | 1.02 | 198 | 217.3 | 271 | 94 | 93.8 | 95 | 99 | 103.5 |
| 7.5 | 1708 | 0.71 | 166 | 179.1 | 271 | 87 | 88.2 | 95 | 72 | 78.0 |
| 10 | 1238 | 0.52 | 130 | 147.0 | 271 | 77 | 80.6 | 95 | 51 | 60.2 |
| 15 | 751 | 0.31 | 99 | 105.6 | 271 | 66 | 65.0 | 95 | 30 | 36.6 |
| 20 | 514 | 0.21 | 74 | 81.7 | 271 | 54 | 54.2 | 95 | 15 | 24.8 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 438 | 0.158 |
| 4-8 | 1329 | 0.172 |
| 8-12 | 1790 | 0.134 |
| 12-16 | 2199 | 0.085 |
| 16-20 | 2365 | 0.045 |
