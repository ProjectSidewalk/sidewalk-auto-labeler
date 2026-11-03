# Thinning experiment — thinexp_richmond

mapillary; 1320 panos processed un-thinned (1320 in scan.json); 94 model-derived ramp sites from detections >= 0.55 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1320 | 0.55 | 94 | 94.0 | 94 | 57 | 57.0 | 57 | 69 | 69.0 |
| 2.5 | 487 | 0.2 | 84 | 71.7 | 94 | 56 | 55.2 | 57 | 54 | 52.0 |
| 5 | 363 | 0.15 | 77 | 65.5 | 94 | 55 | 53.2 | 57 | 51 | 47.1 |
| 7.5 | 284 | 0.12 | 69 | 60.5 | 94 | 52 | 51.2 | 57 | 48 | 42.4 |
| 10 | 230 | 0.1 | 67 | 56.8 | 94 | 53 | 48.8 | 57 | 41 | 37.9 |
| 15 | 169 | 0.07 | 56 | 50.1 | 94 | 47 | 44.4 | 57 | 35 | 32.1 |
| 20 | 131 | 0.05 | 53 | 45.3 | 94 | 46 | 41.0 | 57 | 32 | 27.4 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 78 | 0.141 |
| 4-8 | 363 | 0.474 |
| 8-12 | 847 | 0.433 |
| 12-16 | 1029 | 0.352 |
| 16-20 | 1027 | 0.206 |
