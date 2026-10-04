# bend: Q1 aerial mask quality (#104) (Bend is a RampNet training city)

- Tile2Net 0.5.0 commit `c737a4e8c8b907ee3c80739673c67140ce95483d`; tiles: City of Bend 2019 orthoimagery (ArcGIS Online tile cache), fetched by scripts/aerial_fetch_tiles.py and passed with --input at zoom 19 (52480 tiles); polygons.geojson sha256 `fce788012b8c7b4d9df3146a47c018006fa359e34e5ec647c8c27ae3f8109e5a`
- camera height `auto` -> per-rig (gsv-per-rig; per capture year: 2007 2.5 m, 2008 2.5 m, 2009 2.5 m, 2011 2.5 m, 2012 2.5 m, 2015 2.5 m, 2017 2.5 m, 2018 2.5 m, 2019 2.5 m, 2021 2.5 m, 2024 2.5 m, 2025 2.5 m)
- GT: 110 judged panos -> 299 placeable points (tier 0.55), merged at 2.5 m

| set | n | inside | <= 1 m | <= 2 m | <= 3 m | p50 | p90 | beyond 1 m p50 / p90 | chance <= 2 m | outside coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| gt_all | 298 | 0.3725 | 0.604 | 0.6745 | 0.7114 | 0.371 | 17.897 | 8.338 / 44.701 | 0.3255 | 0 |
| gt_det | 242 | 0.3967 | 0.6322 | 0.6983 | 0.7231 | 0.31 | 19.081 | 10.799 / 46.598 | 0.314 | 0 |
| gt_missed | 56 | 0.2679 | 0.4821 | 0.5714 | 0.6607 | 1.168 | 9.81 | 5.367 / 20.016 | 0.375 | 0 |
| gt_year_2012 | 12 | 0.4167 | 0.6667 | 0.75 | 0.9167 | 0.225 | 2.726 | 2.726 / 11.709 | 0.25 | 0 |
| gt_year_2015 | 5 | 0.0 | 0.0 | 0.0 | 0.2 | 9.81 | 19.719 | 9.81 / 19.719 | 0.0 | 0 |
| gt_year_2017 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 30.005 | 35.287 | 30.005 / 35.287 | 0.0 | 0 |
| gt_year_2018 | 16 | 0.375 | 0.75 | 0.9375 | 0.9375 | 0.207 | 1.319 | 1.319 / 6.143 | 0.625 | 0 |
| gt_year_2019 | 13 | 0.2308 | 0.4615 | 0.6154 | 0.6154 | 1.026 | 15.264 | 8.851 / 16.731 | 0.4615 | 0 |
| gt_year_2021 | 2 | 0.5 | 0.5 | 0.5 | 0.5 | 7.69 | 7.69 | 7.69 / 7.69 | 0.0 | 0 |
| gt_year_2024 | 245 | 0.3918 | 0.6204 | 0.6776 | 0.7102 | 0.31 | 18.199 | 8.348 / 49.086 | 0.3102 | 0 |
| gt_year_2025 | 2 | 0.0 | 0.5 | 1.0 | 1.0 | 1.015 | 1.015 | 1.015 / 1.015 | 1.0 | 0 |
| gt_all_walkable | 298 | 0.3725 | 0.604 | 0.6745 | 0.7114 | 0.371 | 17.897 | 8.338 / 44.701 |  | 0 |
| gt_all_sidewalk_only | 298 | 0.3054 | 0.557 | 0.6275 | 0.6711 | 0.618 | 17.897 | 7.333 / 34.449 |  | 0 |
| inventory_all | 12504 | 0.4636 | 0.6666 | 0.7029 | 0.7258 | 0.083 | 30.919 | 12.845 / 122.244 | 0.2169 | 0 |
| inventory_visible_pool | 12065 | 0.4739 | 0.682 | 0.7194 | 0.7429 | 0.056 | 21.002 | 11.869 / 89.819 | 0.2217 | 0 |
| inventory_installed_before_imagery | 8789 | 0.5873 | 0.8277 | 0.8593 | 0.877 | 0.0 | 5.511 | 7.018 / 29.93 | 0.2628 | 0 |
| inventory_installed_imagery_year | 383 | 0.3473 | 0.5352 | 0.6005 | 0.6371 | 0.649 | 35.92 | 12.289 / 71.02 | 0.1462 | 0 |
| inventory_installed_after_imagery | 2653 | 0.098 | 0.176 | 0.225 | 0.2612 | 15.443 | 162.323 | 24.688 / 188.218 | 0.0874 | 0 |
| inventory_install_date_unknown | 679 | 0.3564 | 0.5714 | 0.6038 | 0.6333 | 0.5 | 19.918 | 11.201 / 44.957 | 0.1679 | 0 |
| inventory_in_empty_inputs | 278 | 0.0 | 0.0 | 0.0 | 0.0 | 198.909 | 359.02 | 198.909 / 359.02 | 0.0 | 0 |
| inventory_excl_empty_inputs | 12226 | 0.4742 | 0.6817 | 0.7189 | 0.7423 | 0.056 | 20.975 | 11.874 / 74.414 | 0.2218 | 0 |

Distances are to the nearest sidewalk or crosswalk polygon (0 = inside) unless the set says walkable (adds footpath) or sidewalk_only. `chance` = the same points displaced 10 m in a seeded random direction, a floor for how much of the city the polygons cover (descriptive, never gated).

`inventory_installed_*` split the inventory by its InstallDate against the imagery year (unknown = missing or the 1900-01-01 placeholder). `inventory_in_empty_inputs` = ramps inside a Tile2Net input (one stitched block of z19 tiles) that holds no polygon of any class, road included; holes.json and holes.csv break those down, with the blank-tile audit from `tiles`. All descriptive: the rule reads `inventory_all` only.
