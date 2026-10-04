# vancouver: corner inventory (RampNet#238)

Built 2026-10-04T17:27:03+00:00 by `scripts/corner_inventory.py@8058b2fd5f842db3cdaf78803f438d1cae5568c2`; exporter `export_cluster_review.py@5b21cd9b00d9b00a2b8f9ab5345591eb599182b2`. Protocol: docs/corner-inventory.md.

## Inputs

- results: `..\..\sal-vancouver\runs\vancouver\results.jsonl` sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
- osm: `..\..\sal-cluster-review\runs\vancouver\cluster_review\osm.json` sha256 `58ce83ffa4b8eea199d7017b53fe125079e2cfa2f3d5181f9e22d61bc0c2a030`
- sites: `..\..\sal-vancouver\runs\vancouver\sites.jsonl` sha256 `9d1fc51ac2867d6247bdf2e0b1c6b5c9fccbbf9a566771f0222bb2728055c914`
- sites_meta: `..\..\sal-vancouver\runs\vancouver\sites_meta.json` sha256 `5a72734079b30710343b8dda2e391145be0721114509015dc963d69e7d873678`
- clusters: `..\..\sal-vancouver\runs\vancouver\ps_clustering_eval\clusters.geojson` sha256 `d8a1e7065e2cf97ff7da0ab50eba7e754b2ac67c4286ecbc8d528f8ff6bec1c3`
- raw_labels: `..\..\sal-vancouver\runs\vancouver\provenance_gate\raw_labels.geojson` sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d`
- streets: `..\..\sal-vancouver\runs\vancouver\ps_clustering_eval\streets.geojson` sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`
- area: `..\..\sal-vancouver\runs\vancouver\area.geojson` sha256 `c7dd888fe84fefdf9e08b46d5441282d51e05c2cc1724e46114bfdb5355437a9`
- inventory: `..\..\sal-vancouver\runs\vancouver\inventory_oracle\inventory.geojson` sha256 `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87`
- inventory_record: `..\..\sal-vancouver\runs\vancouver\inventory_oracle\inventory.json` sha256 `3f9d2917b4b39dbe535d4ffd34d2aecfc55c50916b935536b1d05f4dee1801fe`
- store_selection: `..\..\sal-vancouver\runs\vancouver\store_selection.json` sha256 `5409a89b2ef945721cedcb80bf231edf97ce81ade99614c7ce69c23019afdd9f`
- store_sampled_ids: `..\..\sal-vancouver\runs\vancouver\store_sampled_ids.txt` sha256 `4a38c466b680ebf9bb6bd85899264f9e0672a0507aeb72aa85558ca07cbb48da`
- units224: `..\..\RampNet\benchmark\vancouver\cluster_review\corners.jsonl` sha256 `4c862a4039ca65b23b0ec02d44901d1354a6dbbac0fb07877b9d9f4204c23ca3`

## Units

- intersection units 4429 (signalised 284, arterial 1497, residential 2648), corners 14901 (wide > 150 deg: 3068); mid-block points 16912
- legs per intersection unit: {1: 3, 2: 180, 3: 2770, 4: 1286, 5: 77, 6: 100, 7: 5, 8: 8}
- panos 28830, operational sites 17650, deployed clusters 18684, inventory points 17440

## Decision

Absence precision, clean read, unit level, fusion arm, primary observability, intersections pooled: 46/84 = 0.5476 (Wilson 95% [0.4414, 0.6496]); threshold 0.9. **FAIL: triage the false absences (detector miss vs observability) first**. Second read (absent with no `Available` point, not the rule): 0.8929.

## Tables

### City sentence, unit level

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 284 | 0.982 [0.959, 0.992] | 0.007 [0.002, 0.025] | 0.011 [0.004, 0.031] |
| signalised | deployed | 284 | 0.979 [0.955, 0.990] | 0.007 [0.002, 0.025] | 0.014 [0.005, 0.036] |
| arterial | fusion | 1497 | 0.719 [0.695, 0.741] | 0.026 [0.019, 0.035] | 0.255 [0.234, 0.278] |
| arterial | deployed | 1497 | 0.719 [0.696, 0.742] | 0.025 [0.019, 0.035] | 0.255 [0.234, 0.278] |
| residential | fusion | 2648 | 0.580 [0.562, 0.599] | 0.016 [0.012, 0.022] | 0.403 [0.385, 0.422] |
| residential | deployed | 2648 | 0.582 [0.563, 0.601] | 0.015 [0.011, 0.021] | 0.403 [0.384, 0.421] |
| intersections | fusion | 4429 | 0.653 [0.639, 0.667] | 0.019 [0.015, 0.023] | 0.328 [0.314, 0.342] |
| intersections | deployed | 4429 | 0.654 [0.640, 0.668] | 0.018 [0.015, 0.023] | 0.328 [0.314, 0.342] |
| mid_block | fusion | 16912 | 0.148 [0.142, 0.153] | 0.028 [0.026, 0.031] | 0.824 [0.818, 0.830] |
| mid_block | deployed | 16912 | 0.146 [0.141, 0.152] | 0.027 [0.025, 0.030] | 0.826 [0.820, 0.832] |

### City sentence, corner level

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 1254 | 0.783 [0.759, 0.805] | 0.208 [0.187, 0.231] | 0.009 [0.005, 0.016] |
| signalised | deployed | 1254 | 0.799 [0.776, 0.820] | 0.191 [0.170, 0.213] | 0.010 [0.006, 0.018] |
| arterial | fusion | 4976 | 0.578 [0.564, 0.591] | 0.179 [0.169, 0.190] | 0.243 [0.231, 0.255] |
| arterial | deployed | 4976 | 0.566 [0.552, 0.579] | 0.192 [0.181, 0.203] | 0.242 [0.231, 0.254] |
| residential | fusion | 8671 | 0.432 [0.422, 0.442] | 0.161 [0.154, 0.169] | 0.407 [0.397, 0.417] |
| residential | deployed | 8671 | 0.412 [0.402, 0.423] | 0.181 [0.173, 0.189] | 0.407 [0.397, 0.417] |
| intersections | fusion | 14901 | 0.510 [0.502, 0.518] | 0.171 [0.165, 0.177] | 0.319 [0.311, 0.326] |
| intersections | deployed | 14901 | 0.496 [0.488, 0.504] | 0.185 [0.179, 0.192] | 0.319 [0.311, 0.326] |

### City sentence, corner level (sectors <= 150 deg only)

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 1162 | 0.781 [0.756, 0.803] | 0.213 [0.191, 0.238] | 0.006 [0.003, 0.012] |
| signalised | deployed | 1162 | 0.800 [0.776, 0.822] | 0.192 [0.170, 0.216] | 0.008 [0.004, 0.015] |
| arterial | fusion | 3800 | 0.664 [0.649, 0.679] | 0.131 [0.120, 0.142] | 0.206 [0.193, 0.219] |
| arterial | deployed | 3800 | 0.656 [0.640, 0.670] | 0.139 [0.129, 0.151] | 0.205 [0.192, 0.218] |
| residential | fusion | 6871 | 0.497 [0.485, 0.509] | 0.111 [0.104, 0.119] | 0.391 [0.380, 0.403] |
| residential | deployed | 6871 | 0.479 [0.467, 0.491] | 0.130 [0.123, 0.139] | 0.391 [0.379, 0.402] |
| intersections | fusion | 11833 | 0.579 [0.570, 0.588] | 0.128 [0.122, 0.134] | 0.294 [0.286, 0.302] |
| intersections | deployed | 11833 | 0.567 [0.558, 0.576] | 0.139 [0.133, 0.146] | 0.293 [0.285, 0.302] |

### Recall against `Available` inventory, unit level

| stratum | arm | with Available | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 274 | 1.000 [0.986, 1.000] | 0.000 [0.000, 0.014] | 0.000 [0.000, 0.014] |
| signalised | deployed | 274 | 0.996 [0.980, 0.999] | 0.000 [0.000, 0.014] | 0.004 [0.001, 0.020] |
| arterial | fusion | 1020 | 0.972 [0.959, 0.980] | 0.006 [0.003, 0.013] | 0.023 [0.015, 0.034] |
| arterial | deployed | 1020 | 0.973 [0.961, 0.981] | 0.004 [0.002, 0.010] | 0.024 [0.016, 0.035] |
| residential | fusion | 1323 | 0.965 [0.954, 0.974] | 0.002 [0.001, 0.007] | 0.033 [0.024, 0.043] |
| residential | deployed | 1323 | 0.965 [0.954, 0.974] | 0.003 [0.001, 0.008] | 0.032 [0.024, 0.043] |
| intersections | fusion | 2617 | 0.971 [0.964, 0.977] | 0.003 [0.002, 0.007] | 0.025 [0.020, 0.032] |
| intersections | deployed | 2617 | 0.971 [0.964, 0.977] | 0.003 [0.002, 0.006] | 0.026 [0.020, 0.032] |

### Recall against `Available` inventory, corner level

| stratum | arm | with Available | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 953 | 0.985 [0.975, 0.991] | 0.015 [0.009, 0.025] | 0.000 [0.000, 0.004] |
| signalised | deployed | 953 | 0.979 [0.968, 0.986] | 0.019 [0.012, 0.030] | 0.002 [0.001, 0.008] |
| arterial | fusion | 2649 | 0.969 [0.961, 0.975] | 0.020 [0.016, 0.027] | 0.011 [0.008, 0.016] |
| arterial | deployed | 2649 | 0.961 [0.953, 0.967] | 0.028 [0.022, 0.035] | 0.011 [0.008, 0.016] |
| residential | fusion | 3208 | 0.940 [0.932, 0.948] | 0.035 [0.029, 0.042] | 0.025 [0.020, 0.031] |
| residential | deployed | 3208 | 0.917 [0.907, 0.926] | 0.059 [0.051, 0.068] | 0.024 [0.020, 0.030] |
| intersections | fusion | 6810 | 0.958 [0.953, 0.962] | 0.026 [0.023, 0.031] | 0.016 [0.013, 0.019] |
| intersections | deployed | 6810 | 0.943 [0.937, 0.948] | 0.041 [0.037, 0.046] | 0.016 [0.013, 0.019] |

### Absence precision against the inventory, unit level

| stratum | arm | absent | clean (no point) | false absence (Available) | RMV/NA only | no-Available read |
|---|---|---:|---|---|---|---|
| signalised | fusion | 2 | 1.000 [0.342, 1.000] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 1.000 [0.342, 1.000] |
| signalised | deployed | 2 | 1.000 [0.342, 1.000] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 1.000 [0.342, 1.000] |
| arterial | fusion | 39 | 0.744 [0.589, 0.854] | 0.154 [0.072, 0.297] | 0.103 [0.041, 0.236] | 0.846 [0.703, 0.928] |
| arterial | deployed | 38 | 0.737 [0.580, 0.850] | 0.105 [0.042, 0.241] | 0.158 [0.074, 0.304] | 0.895 [0.759, 0.958] |
| residential | fusion | 43 | 0.349 [0.224, 0.498] | 0.070 [0.024, 0.186] | 0.581 [0.433, 0.716] | 0.930 [0.814, 0.976] |
| residential | deployed | 41 | 0.293 [0.176, 0.445] | 0.098 [0.039, 0.225] | 0.610 [0.457, 0.743] | 0.902 [0.775, 0.961] |
| intersections | fusion | 84 | 0.548 [0.441, 0.650] | 0.107 [0.057, 0.191] | 0.345 [0.252, 0.452] | 0.893 [0.809, 0.943] |
| intersections | deployed | 81 | 0.519 [0.411, 0.624] | 0.099 [0.051, 0.183] | 0.383 [0.284, 0.492] | 0.901 [0.817, 0.949] |

### Absence precision against the inventory, corner level

| stratum | arm | absent | clean (no point) | false absence (Available) | RMV/NA only | no-Available read |
|---|---|---:|---|---|---|---|
| signalised | fusion | 261 | 0.908 [0.867, 0.937] | 0.054 [0.032, 0.088] | 0.038 [0.021, 0.069] | 0.946 [0.912, 0.968] |
| signalised | deployed | 239 | 0.879 [0.831, 0.914] | 0.075 [0.048, 0.116] | 0.046 [0.026, 0.081] | 0.925 [0.884, 0.952] |
| arterial | fusion | 893 | 0.732 [0.702, 0.760] | 0.060 [0.047, 0.078] | 0.207 [0.182, 0.235] | 0.940 [0.922, 0.953] |
| arterial | deployed | 955 | 0.710 [0.680, 0.738] | 0.077 [0.062, 0.096] | 0.213 [0.188, 0.240] | 0.923 [0.904, 0.938] |
| residential | fusion | 1398 | 0.532 [0.506, 0.558] | 0.080 [0.067, 0.096] | 0.388 [0.362, 0.414] | 0.920 [0.904, 0.933] |
| residential | deployed | 1568 | 0.487 [0.463, 0.512] | 0.121 [0.105, 0.138] | 0.392 [0.368, 0.417] | 0.879 [0.862, 0.895] |
| intersections | fusion | 2552 | 0.641 [0.622, 0.659] | 0.071 [0.061, 0.081] | 0.289 [0.272, 0.307] | 0.929 [0.919, 0.939] |
| intersections | deployed | 2762 | 0.598 [0.580, 0.616] | 0.102 [0.091, 0.114] | 0.300 [0.283, 0.318] | 0.898 [0.886, 0.909] |

### Observability sensitivity (intersections pooled, fusion arm)

| level | variant | present | absent | unobservable | absent clean | false absence |
|---|---|---|---|---|---|---|
| unit | primary | 0.653 [0.639, 0.667] | 0.019 [0.015, 0.023] | 0.328 [0.314, 0.342] | 0.548 [0.441, 0.650] | 0.107 [0.057, 0.191] |
| unit | ge2 | 0.653 [0.639, 0.667] | 0.002 [0.001, 0.004] | 0.345 [0.331, 0.359] | 0.700 [0.397, 0.892] | 0.100 [0.018, 0.404] |
| unit | le15 | 0.653 [0.639, 0.667] | 0.006 [0.004, 0.009] | 0.341 [0.327, 0.355] | 0.556 [0.373, 0.724] | 0.074 [0.021, 0.234] |
| corner | primary | 0.510 [0.502, 0.518] | 0.171 [0.165, 0.177] | 0.319 [0.311, 0.326] | 0.641 [0.622, 0.659] | 0.071 [0.061, 0.081] |
| corner | ge2 | 0.510 [0.502, 0.518] | 0.134 [0.129, 0.140] | 0.356 [0.348, 0.363] | 0.691 [0.671, 0.711] | 0.070 [0.060, 0.082] |
| corner | le15 | 0.510 [0.502, 0.518] | 0.133 [0.127, 0.138] | 0.357 [0.350, 0.365] | 0.650 [0.628, 0.670] | 0.065 [0.055, 0.077] |

### Inventory gaps (present, no inventory point of any status)

| level | stratum | fusion | deployed | present not observed (fusion) |
|---|---|---:|---:|---:|
| unit | signalised | 4 | 4 | 1 |
| unit | arterial | 39 | 40 | 8 |
| unit | residential | 73 | 77 | 16 |
| unit | intersections | 116 | 121 | 25 |
| corner | signalised | 36 | 63 | 3 |
| corner | arterial | 167 | 144 | 5 |
| corner | residential | 237 | 217 | 16 |
| corner | intersections | 440 | 424 | 24 |

### What the inventory holds at absent units / corners (fusion, primary, intersections pooled)

| level | absent | with Available | with NA_noramp | with NA_typed | with RMV | with Expired/Removed | with other | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unit | 84 | 9 | 32 | 0 | 0 | 0 | 0 | 46 |
| corner | 2552 | 180 | 728 | 5 | 9 | 1 | 3 | 1635 |

### How the pano set was selected (store-built run)

- run panos 28830; with >= 1 detection >= 0.55: 26930
- seeded unlabeled store sample: 300 selected, 300 in the run, 51 within 25 m of an intersection unit centre
- unobservable units: nearest pano (m) {'0.0': 25.12, '0.25': 27.5, '0.5': 31.09, '0.75': 34.09, '1.0': 36.98}, none within 37 m: 1345

| unit state (fusion, primary) | panos within 25 m | units |
|---|---|---:|
| absent | labeled panos only | 55 |
| absent | sampled-unlabeled pano within 25 m | 29 |
| present | labeled panos only | 2845 |
| present | no pano within 25 m | 25 |
| present | sampled-unlabeled pano within 25 m | 22 |
| unobservable | no pano within 25 m | 1453 |

Inventory-gap corners (fusion, primary): 440 rows in gaps.csv.

Build 17.5 s; score 2.1 s.
