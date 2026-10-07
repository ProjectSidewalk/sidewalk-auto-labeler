# Thinning experiment — thinexp_lyon_c

panoramax; 3853 panos processed un-thinned (3941 in scan.json); 300 model-derived ramp sites from detections >= 0.55 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3941 | 1.64 | 300 | 300.0 | 300 | 123 | 123.0 | 123 | 181 | 181.0 |
| 2.5 | 2728 | 1.14 | 280 | 255.9 | 300 | 123 | 122.1 | 123 | 161 | 140.7 |
| 5 | 1518 | 0.63 | 249 | 191.9 | 300 | 121 | 112.3 | 123 | 124 | 96.2 |
| 7.5 | 978 | 0.41 | 208 | 151.7 | 300 | 110 | 100.0 | 123 | 97 | 73.5 |
| 10 | 726 | 0.3 | 174 | 129.3 | 300 | 94 | 90.8 | 123 | 64 | 61.5 |
| 15 | 444 | 0.18 | 128 | 99.3 | 300 | 73 | 75.2 | 123 | 25 | 42.8 |
| 20 | 334 | 0.14 | 94 | 84.0 | 300 | 57 | 65.4 | 123 | 8 | 33.2 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 645 | 0.175 |
| 4-8 | 2025 | 0.221 |
| 8-12 | 3030 | 0.17 |
| 12-16 | 3767 | 0.101 |
| 16-20 | 4083 | 0.04 |
