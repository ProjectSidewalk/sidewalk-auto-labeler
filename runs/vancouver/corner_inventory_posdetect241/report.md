# vancouver: corner inventory (RampNet#238)

Built 2026-10-05T15:49:20+00:00 by `scripts/corner_inventory.py@9c8f4794567f69a692f66eb408c5d87431bff214`; exporter `export_cluster_review.py@5b21cd9b00d9b00a2b8f9ab5345591eb599182b2`. Protocol: docs/corner-inventory.md.

## Inputs

- results: `D:\Git\sal-vancouver\runs\vancouver\results.jsonl` sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
- osm: `D:\Git\sal-cluster-review\runs\vancouver\cluster_review\osm.json` sha256 `58ce83ffa4b8eea199d7017b53fe125079e2cfa2f3d5181f9e22d61bc0c2a030`
- sites: `D:\Git\sal-vancouver\runs\vancouver\sites.jsonl` sha256 `9d1fc51ac2867d6247bdf2e0b1c6b5c9fccbbf9a566771f0222bb2728055c914`
- sites_meta: `D:\Git\sal-vancouver\runs\vancouver\sites_meta.json` sha256 `5a72734079b30710343b8dda2e391145be0721114509015dc963d69e7d873678`
- clusters: `D:\Git\sal-vancouver\runs\vancouver\ps_clustering_eval\clusters.geojson` sha256 `d8a1e7065e2cf97ff7da0ab50eba7e754b2ac67c4286ecbc8d528f8ff6bec1c3`
- raw_labels: `D:\Git\sal-vancouver\runs\vancouver\provenance_gate\raw_labels.geojson` sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d`
- streets: `D:\Git\sal-vancouver\runs\vancouver\ps_clustering_eval\streets.geojson` sha256 `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4`
- area: `D:\Git\sal-vancouver\runs\vancouver\area.geojson` sha256 `c7dd888fe84fefdf9e08b46d5441282d51e05c2cc1724e46114bfdb5355437a9`
- inventory: `D:\Git\sal-vancouver\runs\vancouver\inventory_oracle\inventory.geojson` sha256 `c4d2497995f7b6261333c3859a668348cd62a36367593cef2dfdfa6b7e020d87`
- inventory_record: `D:\Git\sal-vancouver\runs\vancouver\inventory_oracle\inventory.json` sha256 `3f9d2917b4b39dbe535d4ffd34d2aecfc55c50916b935536b1d05f4dee1801fe`
- store_selection: `D:\Git\sal-vancouver\runs\vancouver\store_selection.json` sha256 `5409a89b2ef945721cedcb80bf231edf97ce81ade99614c7ce69c23019afdd9f`
- store_sampled_ids: `D:\Git\sal-vancouver\runs\vancouver\store_sampled_ids.txt` sha256 `4a38c466b680ebf9bb6bd85899264f9e0672a0507aeb72aa85558ca07cbb48da`
- units224: `D:\Git\RampNet\benchmark\vancouver\cluster_review\corners.jsonl` sha256 `4c862a4039ca65b23b0ec02d44901d1354a6dbbac0fb07877b9d9f4204c23ca3`
- extra:vancouver_posdetect241:results: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241\results.jsonl` sha256 `69f46b26110df05d1ef52a4431c4b230ceb157941b212fd3d553fd64bb86bf0e`
- extra:vancouver_posdetect241:sites: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241\sites.jsonl` sha256 `d38c6fa08f9774de842c77c72ad51346daebb918270c90b319b662bd8534320b`
- extra:vancouver_posdetect241:sites_meta: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241\sites_meta.json` sha256 `ce4eadec1db56a826ca85845a7594bd84e70851ac497dc896a1ce29d3d3cad0c`
- extra:vancouver_posdetect241_gsv:results: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241_gsv\results.jsonl` sha256 `953716d1d7f32f2680c101fb9a4d7ab0933939459704fbc62375042768ca57b2`
- extra:vancouver_posdetect241_gsv:sites: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241_gsv\sites.jsonl` sha256 `df0dec5c6c9456248a7ee1aee7c4e3e7c4c50643780a5105dd7451b3e7e7614d`
- extra:vancouver_posdetect241_gsv:sites_meta: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241_gsv\sites_meta.json` sha256 `3d5a1e7a04de6d972a08e87a8b5e060ac7c9811ca506aae2e7951e4da0f86099`

## Units

- intersection units 4429 (signalised 284, arterial 1497, residential 2648), corners 14391 (wide > 150 deg: 3177); mid-block points 16912
- legs per intersection unit: {1: 3, 2: 273, 3: 2778, 4: 1368, 5: 6, 6: 1}
- panos 39197, operational sites 17927, deployed clusters 18711, inventory points 17440

## Decision

Absence precision, clean read, unit level, fusion arm, primary observability, intersections pooled: 333/1323 = 0.2517 (Wilson 95% [0.2291, 0.2758]); threshold 0.9. **FAIL: triage the false absences (detector miss vs observability) first**. Second read (absent with no `Available` point, not the rule): 0.9667.

## Tables

### City sentence, unit level

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 284 | 0.986 [0.964, 0.995] | 0.014 [0.005, 0.036] | 0.000 [0.000, 0.013] |
| signalised | deployed | 284 | 0.979 [0.955, 0.990] | 0.018 [0.008, 0.041] | 0.004 [0.001, 0.020] |
| arterial | fusion | 1497 | 0.752 [0.730, 0.773] | 0.248 [0.227, 0.270] | 0.000 [0.000, 0.003] |
| arterial | deployed | 1497 | 0.723 [0.700, 0.745] | 0.277 [0.254, 0.300] | 0.001 [0.000, 0.004] |
| residential | fusion | 2648 | 0.641 [0.622, 0.659] | 0.358 [0.340, 0.376] | 0.001 [0.000, 0.003] |
| residential | deployed | 2648 | 0.588 [0.569, 0.607] | 0.409 [0.391, 0.428] | 0.003 [0.001, 0.005] |
| intersections | fusion | 4429 | 0.701 [0.687, 0.714] | 0.299 [0.285, 0.312] | 0.001 [0.000, 0.002] |
| intersections | deployed | 4429 | 0.659 [0.645, 0.672] | 0.339 [0.326, 0.353] | 0.002 [0.001, 0.004] |
| mid_block | fusion | 16912 | 0.148 [0.142, 0.153] | 0.028 [0.026, 0.031] | 0.824 [0.818, 0.830] |
| mid_block | deployed | 16912 | 0.146 [0.141, 0.152] | 0.027 [0.025, 0.030] | 0.826 [0.820, 0.832] |

### City sentence, corner level

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 1004 | 0.947 [0.932, 0.959] | 0.051 [0.039, 0.066] | 0.002 [0.001, 0.007] |
| signalised | deployed | 1004 | 0.941 [0.925, 0.954] | 0.055 [0.042, 0.071] | 0.004 [0.002, 0.010] |
| arterial | fusion | 4743 | 0.615 [0.601, 0.629] | 0.370 [0.356, 0.384] | 0.015 [0.012, 0.019] |
| arterial | deployed | 4743 | 0.590 [0.576, 0.604] | 0.394 [0.381, 0.408] | 0.015 [0.012, 0.019] |
| residential | fusion | 8644 | 0.453 [0.443, 0.464] | 0.530 [0.519, 0.540] | 0.017 [0.014, 0.020] |
| residential | deployed | 8644 | 0.415 [0.404, 0.425] | 0.568 [0.557, 0.578] | 0.017 [0.015, 0.020] |
| intersections | fusion | 14391 | 0.541 [0.533, 0.549] | 0.444 [0.436, 0.452] | 0.015 [0.013, 0.017] |
| intersections | deployed | 14391 | 0.509 [0.501, 0.517] | 0.475 [0.467, 0.483] | 0.016 [0.014, 0.018] |

### City sentence, corner level (sectors <= 150 deg only)

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 869 | 0.959 [0.943, 0.970] | 0.041 [0.030, 0.057] | 0.000 [0.000, 0.004] |
| signalised | deployed | 869 | 0.955 [0.939, 0.967] | 0.043 [0.031, 0.058] | 0.002 [0.001, 0.008] |
| arterial | fusion | 3506 | 0.719 [0.704, 0.733] | 0.271 [0.257, 0.286] | 0.010 [0.007, 0.014] |
| arterial | deployed | 3506 | 0.697 [0.682, 0.712] | 0.293 [0.278, 0.308] | 0.010 [0.007, 0.014] |
| residential | fusion | 6839 | 0.520 [0.509, 0.532] | 0.464 [0.452, 0.476] | 0.015 [0.013, 0.019] |
| residential | deployed | 6839 | 0.482 [0.470, 0.493] | 0.503 [0.491, 0.515] | 0.016 [0.013, 0.019] |
| intersections | fusion | 11214 | 0.616 [0.607, 0.625] | 0.371 [0.362, 0.380] | 0.013 [0.011, 0.015] |
| intersections | deployed | 11214 | 0.586 [0.576, 0.595] | 0.401 [0.392, 0.411] | 0.013 [0.011, 0.015] |

### Recall against `Available` inventory, unit level

| stratum | arm | with Available | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 274 | 1.000 [0.986, 1.000] | 0.000 [0.000, 0.014] | 0.000 [0.000, 0.014] |
| signalised | deployed | 274 | 0.996 [0.980, 0.999] | 0.000 [0.000, 0.014] | 0.004 [0.001, 0.020] |
| arterial | fusion | 1020 | 0.981 [0.971, 0.988] | 0.019 [0.012, 0.029] | 0.000 [0.000, 0.004] |
| arterial | deployed | 1020 | 0.974 [0.962, 0.982] | 0.026 [0.018, 0.038] | 0.000 [0.000, 0.004] |
| residential | fusion | 1323 | 0.981 [0.972, 0.987] | 0.019 [0.013, 0.028] | 0.000 [0.000, 0.003] |
| residential | deployed | 1323 | 0.970 [0.959, 0.978] | 0.029 [0.022, 0.040] | 0.001 [0.000, 0.004] |
| intersections | fusion | 2617 | 0.983 [0.978, 0.987] | 0.017 [0.013, 0.022] | 0.000 [0.000, 0.001] |
| intersections | deployed | 2617 | 0.974 [0.967, 0.979] | 0.025 [0.020, 0.032] | 0.001 [0.000, 0.003] |

### Recall against `Available` inventory, corner level

| stratum | arm | with Available | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 937 | 0.987 [0.978, 0.993] | 0.013 [0.007, 0.022] | 0.000 [0.000, 0.004] |
| signalised | deployed | 937 | 0.980 [0.969, 0.987] | 0.018 [0.011, 0.029] | 0.002 [0.001, 0.008] |
| arterial | fusion | 2636 | 0.974 [0.967, 0.979] | 0.025 [0.020, 0.032] | 0.001 [0.000, 0.003] |
| arterial | deployed | 2636 | 0.962 [0.954, 0.968] | 0.037 [0.030, 0.045] | 0.002 [0.001, 0.004] |
| residential | fusion | 3205 | 0.948 [0.940, 0.955] | 0.048 [0.041, 0.056] | 0.004 [0.002, 0.007] |
| residential | deployed | 3205 | 0.919 [0.909, 0.928] | 0.077 [0.068, 0.087] | 0.004 [0.002, 0.007] |
| intersections | fusion | 6778 | 0.964 [0.959, 0.968] | 0.034 [0.030, 0.039] | 0.002 [0.001, 0.003] |
| intersections | deployed | 6778 | 0.944 [0.938, 0.949] | 0.053 [0.048, 0.059] | 0.003 [0.002, 0.004] |

### Absence precision against the inventory, unit level

| stratum | arm | absent | clean (no point) | false absence (Available) | RMV/NA only | no-Available read |
|---|---|---:|---|---|---|---|
| signalised | fusion | 4 | 0.750 [0.301, 0.954] | 0.000 [0.000, 0.490] | 0.250 [0.046, 0.699] | 1.000 [0.510, 1.000] |
| signalised | deployed | 5 | 0.800 [0.376, 0.964] | 0.000 [0.000, 0.434] | 0.200 [0.036, 0.624] | 1.000 [0.566, 1.000] |
| arterial | fusion | 371 | 0.512 [0.461, 0.563] | 0.051 [0.033, 0.079] | 0.437 [0.387, 0.488] | 0.949 [0.921, 0.967] |
| arterial | deployed | 414 | 0.483 [0.435, 0.531] | 0.065 [0.045, 0.093] | 0.452 [0.404, 0.500] | 0.935 [0.907, 0.955] |
| residential | fusion | 948 | 0.148 [0.127, 0.172] | 0.026 [0.018, 0.039] | 0.826 [0.801, 0.849] | 0.974 [0.961, 0.982] |
| residential | deployed | 1084 | 0.149 [0.129, 0.171] | 0.036 [0.026, 0.049] | 0.815 [0.791, 0.837] | 0.964 [0.951, 0.974] |
| intersections | fusion | 1323 | 0.252 [0.229, 0.276] | 0.033 [0.025, 0.044] | 0.715 [0.690, 0.739] | 0.967 [0.956, 0.975] |
| intersections | deployed | 1503 | 0.243 [0.222, 0.265] | 0.044 [0.035, 0.055] | 0.713 [0.690, 0.736] | 0.956 [0.945, 0.965] |

### Absence precision against the inventory, corner level

| stratum | arm | absent | clean (no point) | false absence (Available) | RMV/NA only | no-Available read |
|---|---|---:|---|---|---|---|
| signalised | fusion | 51 | 0.549 [0.414, 0.677] | 0.235 [0.140, 0.368] | 0.216 [0.125, 0.346] | 0.765 [0.632, 0.860] |
| signalised | deployed | 55 | 0.473 [0.347, 0.602] | 0.309 [0.203, 0.440] | 0.218 [0.129, 0.344] | 0.691 [0.560, 0.797] |
| arterial | fusion | 1754 | 0.609 [0.586, 0.631] | 0.038 [0.030, 0.048] | 0.353 [0.331, 0.376] | 0.962 [0.952, 0.970] |
| arterial | deployed | 1871 | 0.598 [0.576, 0.620] | 0.052 [0.043, 0.063] | 0.350 [0.329, 0.372] | 0.948 [0.937, 0.957] |
| residential | fusion | 4580 | 0.393 [0.379, 0.407] | 0.034 [0.029, 0.039] | 0.574 [0.559, 0.588] | 0.966 [0.961, 0.971] |
| residential | deployed | 4908 | 0.379 [0.365, 0.392] | 0.050 [0.045, 0.057] | 0.571 [0.557, 0.585] | 0.950 [0.943, 0.955] |
| intersections | fusion | 6385 | 0.453 [0.441, 0.466] | 0.036 [0.032, 0.041] | 0.510 [0.498, 0.522] | 0.964 [0.959, 0.968] |
| intersections | deployed | 6834 | 0.439 [0.428, 0.451] | 0.053 [0.048, 0.058] | 0.508 [0.496, 0.520] | 0.947 [0.942, 0.952] |

### Observability sensitivity (intersections pooled, fusion arm)

| level | variant | present | absent | unobservable | absent clean | false absence |
|---|---|---|---|---|---|---|
| unit | primary | 0.701 [0.687, 0.714] | 0.299 [0.285, 0.312] | 0.001 [0.000, 0.002] | 0.252 [0.229, 0.276] | 0.033 [0.025, 0.044] |
| unit | ge2 | 0.701 [0.687, 0.714] | 0.282 [0.269, 0.295] | 0.017 [0.014, 0.022] | 0.235 [0.213, 0.260] | 0.028 [0.020, 0.039] |
| unit | le15 | 0.701 [0.687, 0.714] | 0.284 [0.271, 0.298] | 0.015 [0.012, 0.019] | 0.239 [0.217, 0.264] | 0.029 [0.021, 0.039] |
| corner | primary | 0.541 [0.533, 0.549] | 0.444 [0.436, 0.452] | 0.015 [0.013, 0.017] | 0.453 [0.441, 0.466] | 0.036 [0.032, 0.041] |
| corner | ge2 | 0.541 [0.533, 0.549] | 0.409 [0.401, 0.417] | 0.050 [0.046, 0.053] | 0.454 [0.442, 0.467] | 0.033 [0.029, 0.038] |
| corner | le15 | 0.541 [0.533, 0.549] | 0.396 [0.388, 0.404] | 0.063 [0.059, 0.067] | 0.426 [0.413, 0.439] | 0.031 [0.027, 0.036] |

### Inventory gaps (present, no inventory point of any status)

| level | stratum | fusion | deployed | present not observed (fusion) |
|---|---|---:|---:|---:|
| unit | signalised | 5 | 4 | 1 |
| unit | arterial | 53 | 43 | 7 |
| unit | residential | 100 | 78 | 16 |
| unit | intersections | 158 | 125 | 24 |
| corner | signalised | 19 | 21 | 3 |
| corner | arterial | 186 | 136 | 7 |
| corner | residential | 277 | 216 | 16 |
| corner | intersections | 482 | 373 | 26 |

### What the inventory holds at absent units / corners (fusion, primary, intersections pooled)

| level | absent | with Available | with NA_noramp | with NA_typed | with RMV | with Expired/Removed | with other | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unit | 1323 | 44 | 960 | 0 | 5 | 0 | 1 | 333 |
| corner | 6385 | 233 | 3239 | 12 | 14 | 1 | 10 | 2895 |

### How the pano set was selected (store-built run)

- run panos 28830; with >= 1 detection >= 0.55: 26930
- seeded unlabeled store sample: 300 selected, 300 in the run, 51 within 25 m of an intersection unit centre
- unobservable units: nearest pano (m) {'0.0': 25.25, '0.25': 25.25, '0.5': 25.25, '0.75': 25.25, '1.0': 25.25}, none within 37 m: 2

| unit state (fusion, primary) | panos within 25 m | units |
|---|---|---:|
| absent | labeled panos only | 53 |
| absent | position-selected pano within 25 m | 1244 |
| absent | sampled-unlabeled pano within 25 m | 26 |
| present | labeled panos only | 2824 |
| present | no pano within 25 m | 24 |
| present | position-selected pano within 25 m | 234 |
| present | sampled-unlabeled pano within 25 m | 21 |
| unobservable | no pano within 25 m | 3 |

| absent units (fusion, primary) observed through | units | of them only sampled panos within 25 m | clean (no point) | RMV/NA only | false (Available) |
|---|---:|---:|---:|---:|---:|
| labeled only | 53 | 0 | 36 | 10 | 7 |
| position-selected | 1244 | 0 | 289 | 920 | 35 |
| sampled-unlabeled | 26 | 26 | 8 | 16 | 2 |

Inventory-gap corners (fusion, primary): 482 rows in gaps.csv.

Build 18.3 s; score 2.8 s.
