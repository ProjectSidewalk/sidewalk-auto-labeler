# Thinning experiment — thinexp_lyon_b

panoramax; 4379 panos processed un-thinned (4379 in scan.json); 444 model-derived ramp sites from detections >= 0.3 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4379 | 1.82 | 444 | 444.0 | 444 | 265 | 265.0 | 265 | 320 | 320.0 |
| 2.5 | 2463 | 1.03 | 403 | 375.1 | 444 | 261 | 261.9 | 265 | 271 | 265.2 |
| 5 | 1507 | 0.63 | 368 | 325.1 | 444 | 253 | 251.7 | 265 | 234 | 227.8 |
| 7.5 | 1038 | 0.43 | 324 | 290.8 | 444 | 229 | 238.8 | 265 | 214 | 201.0 |
| 10 | 785 | 0.33 | 315 | 267.1 | 444 | 225 | 225.8 | 265 | 186 | 177.7 |
| 15 | 500 | 0.21 | 261 | 227.2 | 444 | 192 | 200.5 | 265 | 142 | 139.3 |
| 20 | 372 | 0.15 | 233 | 200.5 | 444 | 172 | 181.7 | 265 | 112 | 114.1 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 960 | 0.449 |
| 4-8 | 3046 | 0.43 |
| 8-12 | 3923 | 0.328 |
| 12-16 | 4168 | 0.26 |
| 16-20 | 4364 | 0.187 |
