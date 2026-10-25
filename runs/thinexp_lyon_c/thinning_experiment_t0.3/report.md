# Thinning experiment — thinexp_lyon_c

panoramax; 3853 panos processed un-thinned (3941 in scan.json); 447 model-derived ramp sites from detections >= 0.3 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3941 | 1.64 | 447 | 447.0 | 447 | 209 | 209.0 | 209 | 279 | 279.0 |
| 2.5 | 2728 | 1.14 | 414 | 386.5 | 447 | 207 | 207.8 | 209 | 248 | 226.7 |
| 5 | 1518 | 0.63 | 360 | 297.4 | 447 | 201 | 191.3 | 209 | 197 | 158.8 |
| 7.5 | 978 | 0.41 | 308 | 238.3 | 447 | 183 | 169.2 | 209 | 143 | 119.0 |
| 10 | 726 | 0.3 | 260 | 203.6 | 447 | 161 | 152.4 | 209 | 107 | 98.0 |
| 15 | 444 | 0.18 | 194 | 155.2 | 447 | 123 | 123.0 | 209 | 54 | 68.0 |
| 20 | 334 | 0.14 | 146 | 130.7 | 447 | 100 | 107.3 | 209 | 24 | 53.6 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 911 | 0.225 |
| 4-8 | 2970 | 0.224 |
| 8-12 | 4389 | 0.189 |
| 12-16 | 5289 | 0.118 |
| 16-20 | 5839 | 0.065 |
