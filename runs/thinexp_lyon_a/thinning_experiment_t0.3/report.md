# Thinning experiment — thinexp_lyon_a

panoramax; 4226 panos processed un-thinned (4226 in scan.json); 404 model-derived ramp sites from detections >= 0.3 off the rig (cluster radius 7.5 m; robust = >= 3 member panos). gpu_hours at 1.5 s/pano. See the module docstring for assumptions A1-A3 and caveats.

## A2/A3 — coverage vs spacing (thin_panos vs random same-count mean)

| spacing_m | panos_kept | gpu_hours | sites_retained | sites_retained_random_mean | sites_total | robust_retained | robust_retained_random_mean | robust_total | sites_2plus_views | sites_2plus_views_random_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4226 | 1.76 | 404 | 404.0 | 404 | 203 | 203.0 | 203 | 264 | 264.0 |
| 2.5 | 2146 | 0.89 | 377 | 313.5 | 404 | 202 | 195.7 | 203 | 243 | 189.1 |
| 5 | 1107 | 0.46 | 340 | 239.6 | 404 | 197 | 173.6 | 203 | 198 | 135.4 |
| 7.5 | 702 | 0.29 | 311 | 196.9 | 404 | 187 | 153.2 | 203 | 144 | 112.1 |
| 10 | 485 | 0.2 | 272 | 167.0 | 404 | 166 | 136.1 | 203 | 106 | 94.0 |
| 15 | 290 | 0.12 | 200 | 134.2 | 404 | 118 | 116.7 | 203 | 49 | 74.7 |
| 20 | 186 | 0.08 | 142 | 110.2 | 404 | 91 | 98.4 | 203 | 26 | 58.8 |

## A1 — detection rate by camera-to-site distance

| bin_m | opportunities | detection_rate |
|---|---|---|
| 0-4 | 1177 | 0.377 |
| 4-8 | 4750 | 0.349 |
| 8-12 | 6855 | 0.243 |
| 12-16 | 8132 | 0.132 |
| 16-20 | 8861 | 0.083 |
