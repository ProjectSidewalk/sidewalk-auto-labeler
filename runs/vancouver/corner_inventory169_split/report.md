# vancouver: corner inventory (RampNet#238)

Built 2026-11-06T13:50:53+00:00 by `scripts/corner_inventory.py@fff5e108c3aeeb4142191f3e4de48115ae52b647`; exporter `export_cluster_review.py@1ed70850d45d329dd547ea4363f2630512fb0024`. Protocol: docs/corner-inventory.md.

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
- extra:vancouver_posdetect241:sites_meta: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241\sites_meta.json` sha256 `5408f770ebe84943faf1048a9b8978fe0700997e0ea57b699110c2a5b6e626b9`
- extra:vancouver_posdetect241_gsv:results: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241_gsv\results.jsonl` sha256 `953716d1d7f32f2680c101fb9a4d7ab0933939459704fbc62375042768ca57b2`
- extra:vancouver_posdetect241_gsv:sites: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241_gsv\sites.jsonl` sha256 `df0dec5c6c9456248a7ee1aee7c4e3e7c4c50643780a5105dd7451b3e7e7614d`
- extra:vancouver_posdetect241_gsv:sites_meta: `D:\Git\labeler-wt\posdetect241\runs\vancouver_posdetect241_gsv\sites_meta.json` sha256 `ddae3b0e42d10dd89441cd3e6fff8925ad2e1cc42fb4e3cc388c67c9534a321d`

## Units

- intersection units 4429 (signalised 284, arterial 1497, residential 2648), corners 17063 (wide > 150 deg: 527); mid-block points 16912
- legs per intersection unit: {1: 3, 2: 273, 3: 2778, 4: 1368, 5: 6, 6: 1}; --split-wide: 2672 phantom legs at 2650 units (margin 30.0 deg)
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
| signalised | fusion | 1082 | 0.920 [0.902, 0.934] | 0.078 [0.063, 0.095] | 0.003 [0.001, 0.008] |
| signalised | deployed | 1082 | 0.911 [0.893, 0.927] | 0.084 [0.069, 0.102] | 0.005 [0.002, 0.011] |
| arterial | fusion | 5579 | 0.536 [0.523, 0.550] | 0.448 [0.435, 0.461] | 0.016 [0.013, 0.019] |
| arterial | deployed | 5579 | 0.515 [0.502, 0.528] | 0.469 [0.456, 0.483] | 0.016 [0.013, 0.019] |
| residential | fusion | 10402 | 0.383 [0.373, 0.392] | 0.599 [0.590, 0.608] | 0.018 [0.016, 0.021] |
| residential | deployed | 10402 | 0.349 [0.340, 0.358] | 0.632 [0.623, 0.641] | 0.019 [0.016, 0.021] |
| intersections | fusion | 17063 | 0.467 [0.459, 0.474] | 0.517 [0.509, 0.524] | 0.017 [0.015, 0.019] |
| intersections | deployed | 17063 | 0.439 [0.432, 0.446] | 0.544 [0.537, 0.552] | 0.017 [0.015, 0.019] |

### City sentence, corner level (sectors <= 150 deg only)

| stratum | arm | n | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 1024 | 0.922 [0.904, 0.937] | 0.076 [0.061, 0.094] | 0.002 [0.001, 0.007] |
| signalised | deployed | 1024 | 0.914 [0.895, 0.930] | 0.082 [0.067, 0.100] | 0.004 [0.002, 0.010] |
| arterial | fusion | 5176 | 0.553 [0.539, 0.566] | 0.434 [0.421, 0.448] | 0.013 [0.010, 0.016] |
| arterial | deployed | 5176 | 0.531 [0.517, 0.544] | 0.456 [0.443, 0.470] | 0.013 [0.010, 0.016] |
| residential | fusion | 10336 | 0.382 [0.373, 0.391] | 0.600 [0.591, 0.609] | 0.018 [0.016, 0.021] |
| residential | deployed | 10336 | 0.349 [0.340, 0.358] | 0.633 [0.624, 0.642] | 0.018 [0.016, 0.021] |
| intersections | fusion | 16536 | 0.469 [0.461, 0.476] | 0.516 [0.508, 0.523] | 0.015 [0.014, 0.017] |
| intersections | deployed | 16536 | 0.441 [0.433, 0.448] | 0.544 [0.536, 0.551] | 0.016 [0.014, 0.018] |

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
| signalised | fusion | 978 | 0.986 [0.976, 0.991] | 0.014 [0.009, 0.024] | 0.000 [0.000, 0.004] |
| signalised | deployed | 978 | 0.976 [0.965, 0.984] | 0.021 [0.014, 0.033] | 0.002 [0.001, 0.007] |
| arterial | fusion | 2703 | 0.971 [0.964, 0.976] | 0.028 [0.023, 0.035] | 0.001 [0.000, 0.003] |
| arterial | deployed | 2703 | 0.958 [0.950, 0.965] | 0.041 [0.034, 0.049] | 0.001 [0.001, 0.004] |
| residential | fusion | 3250 | 0.944 [0.936, 0.952] | 0.052 [0.045, 0.061] | 0.003 [0.002, 0.006] |
| residential | deployed | 3250 | 0.914 [0.904, 0.923] | 0.082 [0.073, 0.092] | 0.003 [0.002, 0.006] |
| intersections | fusion | 6931 | 0.960 [0.956, 0.965] | 0.038 [0.033, 0.042] | 0.002 [0.001, 0.003] |
| intersections | deployed | 6931 | 0.940 [0.934, 0.945] | 0.058 [0.052, 0.063] | 0.002 [0.002, 0.004] |

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
| signalised | fusion | 84 | 0.690 [0.585, 0.779] | 0.167 [0.102, 0.261] | 0.143 [0.084, 0.233] | 0.833 [0.739, 0.898] |
| signalised | deployed | 91 | 0.626 [0.524, 0.719] | 0.231 [0.156, 0.327] | 0.143 [0.085, 0.229] | 0.769 [0.673, 0.844] |
| arterial | fusion | 2498 | 0.719 [0.701, 0.736] | 0.031 [0.025, 0.038] | 0.250 [0.234, 0.268] | 0.969 [0.962, 0.975] |
| arterial | deployed | 2619 | 0.706 [0.688, 0.723] | 0.042 [0.035, 0.050] | 0.252 [0.236, 0.269] | 0.958 [0.950, 0.965] |
| residential | fusion | 6232 | 0.549 [0.537, 0.561] | 0.027 [0.024, 0.032] | 0.424 [0.411, 0.436] | 0.973 [0.968, 0.976] |
| residential | deployed | 6576 | 0.531 [0.519, 0.543] | 0.041 [0.036, 0.046] | 0.428 [0.416, 0.440] | 0.959 [0.954, 0.964] |
| intersections | fusion | 8814 | 0.599 [0.588, 0.609] | 0.030 [0.026, 0.033] | 0.372 [0.362, 0.382] | 0.970 [0.967, 0.974] |
| intersections | deployed | 9286 | 0.581 [0.571, 0.591] | 0.043 [0.039, 0.047] | 0.376 [0.366, 0.386] | 0.957 [0.953, 0.961] |

### Observability sensitivity (intersections pooled, fusion arm)

| level | variant | present | absent | unobservable | absent clean | false absence |
|---|---|---|---|---|---|---|
| unit | primary | 0.701 [0.687, 0.714] | 0.299 [0.285, 0.312] | 0.001 [0.000, 0.002] | 0.252 [0.229, 0.276] | 0.033 [0.025, 0.044] |
| unit | ge2 | 0.701 [0.687, 0.714] | 0.282 [0.269, 0.295] | 0.017 [0.014, 0.022] | 0.235 [0.213, 0.260] | 0.028 [0.020, 0.039] |
| unit | le15 | 0.701 [0.687, 0.714] | 0.284 [0.271, 0.298] | 0.015 [0.012, 0.019] | 0.239 [0.217, 0.264] | 0.029 [0.021, 0.039] |
| corner | primary | 0.467 [0.459, 0.474] | 0.517 [0.509, 0.524] | 0.017 [0.015, 0.019] | 0.599 [0.588, 0.609] | 0.030 [0.026, 0.033] |
| corner | ge2 | 0.467 [0.459, 0.474] | 0.478 [0.471, 0.486] | 0.055 [0.051, 0.058] | 0.601 [0.590, 0.611] | 0.027 [0.024, 0.031] |
| corner | le15 | 0.467 [0.459, 0.474] | 0.480 [0.472, 0.487] | 0.053 [0.050, 0.057] | 0.593 [0.582, 0.604] | 0.026 [0.023, 0.030] |

### Inventory gaps (present, no inventory point of any status)

| level | stratum | fusion | deployed | present not observed (fusion) |
|---|---|---:|---:|---:|
| unit | signalised | 5 | 4 | 1 |
| unit | arterial | 53 | 43 | 7 |
| unit | residential | 100 | 78 | 16 |
| unit | intersections | 158 | 125 | 24 |
| corner | signalised | 23 | 24 | 3 |
| corner | arterial | 205 | 154 | 6 |
| corner | residential | 307 | 237 | 10 |
| corner | intersections | 535 | 415 | 19 |

### What the inventory holds at absent units / corners (fusion, primary, intersections pooled)

| level | absent | with Available | with NA_noramp | with NA_typed | with RMV | with Expired/Removed | with other | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unit | 1323 | 44 | 960 | 0 | 5 | 0 | 1 | 333 |
| corner | 8814 | 261 | 3257 | 12 | 14 | 1 | 12 | 5276 |

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

Inventory-gap corners (fusion, primary): 535 rows in gaps.csv.

Build 17.4 s; score 3.2 s.
