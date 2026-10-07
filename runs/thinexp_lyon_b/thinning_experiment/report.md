# Thinning experiment — thinexp_lyon_b

panoramax; 4379 panos processed un-thinned (4379 in scan.json); 314 model-derived ramp sites from detections >= 0.55 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4379 | 1.82 | 314 | 314.0 | 314 | 205 | 205.0 | 205 | 236 | 236.0 |
| 2.5 | 2463 | 1.03 | 295 | 272.5 | 314 | 205 | 203.4 | 205 | 213 | 206.1 |
| 5 | 1507 | 0.63 | 265 | 240.9 | 314 | 195 | 196.3 | 205 | 188 | 179.3 |
| 7.5 | 1038 | 0.43 | 239 | 220.2 | 314 | 180 | 188.0 | 205 | 169 | 156.2 |
| 10 | 785 | 0.33 | 233 | 203.1 | 314 | 179 | 178.0 | 205 | 144 | 136.8 |
| 15 | 500 | 0.21 | 193 | 170.6 | 314 | 150 | 153.9 | 205 | 98 | 101.3 |
| 20 | 372 | 0.15 | 172 | 150.2 | 314 | 135 | 138.1 | 205 | 66 | 80.2 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 675 | 0.4 |
| 4-8 | 2343 | 0.42 |
| 8-12 | 2910 | 0.335 |
| 12-16 | 3083 | 0.24 |
| 16-20 | 3188 | 0.152 |
