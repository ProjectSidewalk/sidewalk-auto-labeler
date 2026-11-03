# paterson: Q1 aerial mask quality (#104)

- Tile2Net 0.5.0 commit `c737a4e8c8b907ee3c80739673c67140ce95483d`; tiles: nj (Tile2Net built-in NewJersey source) at zoom 19 (12064 tiles); polygons.geojson sha256 `0867561c7e04fa205f097f639cb113ee177c29ef88f1f1242bd3bd577c61ea45`
- camera height `auto` -> per-rig (gsv-per-rig; per capture year: 2007 2.5 m, 2008 2.5 m, 2012 2.5 m, 2013 2.5 m, 2014 2.5 m, 2015 2.5 m, 2016 2.5 m, 2017 2.5 m, 2018 2.5 m, 2019 2.5 m, 2020 2.5 m, 2021 2.5 m, 2022 2.5 m, 2023 2.5 m, 2024 2.5 m, 2025 2 m, 2026 2 m)
- GT: 125 judged panos -> 323 placeable points (tier 0.55), merged at 2.5 m

| set | n | inside | <= 1 m | <= 2 m | <= 3 m | p50 | p90 | beyond 1 m p50 / p90 | chance <= 2 m | outside coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| gt_all | 323 | 0.6533 | 0.8669 | 0.9133 | 0.9443 | 0.0 | 1.661 | 2.602 / 24.436 | 0.3963 | 0 |
| gt_det | 232 | 0.6767 | 0.8879 | 0.9267 | 0.9483 | 0.0 | 1.135 | 2.928 / 31.089 | 0.4052 | 0 |
| gt_missed | 91 | 0.5934 | 0.8132 | 0.8791 | 0.9341 | 0.0 | 2.354 | 2.459 / 24.436 | 0.3736 | 0 |
| gt_year_2007 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | None / None | 0.0 | 0 |
| gt_year_2012 | 8 | 0.5 | 0.75 | 1.0 | 1.0 | 0.292 | 1.495 | 1.495 / 1.495 | 0.875 | 0 |
| gt_year_2014 | 4 | 0.75 | 0.75 | 1.0 | 1.0 | 0.0 | 1.661 | 1.661 / 1.661 | 0.75 | 0 |
| gt_year_2018 | 12 | 0.8333 | 0.8333 | 0.9167 | 0.9167 | 0.0 | 1.686 | 3.165 / 3.165 | 0.5833 | 0 |
| gt_year_2019 | 13 | 0.4615 | 0.7692 | 0.7692 | 0.8462 | 0.159 | 22.506 | 22.506 / 34.173 | 0.2308 | 0 |
| gt_year_2020 | 27 | 0.7778 | 0.963 | 0.963 | 0.963 | 0.0 | 0.659 | 3.736 / 3.736 | 0.4074 | 0 |
| gt_year_2021 | 55 | 0.6545 | 0.8909 | 0.9273 | 0.9818 | 0.0 | 1.183 | 2.354 / 3.026 | 0.5091 | 0 |
| gt_year_2024 | 69 | 0.6087 | 0.8261 | 0.8406 | 0.8696 | 0.0 | 5.339 | 6.294 / 31.089 | 0.3623 | 0 |
| gt_year_2025 | 133 | 0.6541 | 0.8797 | 0.9398 | 0.9699 | 0.0 | 1.448 | 2.068 / 24.436 | 0.3308 | 0 |
| gt_all_walkable | 323 | 0.6533 | 0.8669 | 0.9133 | 0.9443 | 0.0 | 1.661 | 2.602 / 24.436 |  | 0 |
| gt_all_sidewalk_only | 323 | 0.5294 | 0.805 | 0.8638 | 0.9102 | 0.0 | 2.687 | 2.724 / 14.135 |  | 0 |

Distances are to the nearest sidewalk or crosswalk polygon (0 = inside) unless the set says walkable (adds footpath) or sidewalk_only. `chance` = the same points displaced 10 m in a seeded random direction, a floor for how much of the city the polygons cover (descriptive, never gated).

`inventory_installed_*` split the inventory by its InstallDate against the imagery year (unknown = missing or the 1900-01-01 placeholder). `inventory_in_empty_inputs` = ramps inside a Tile2Net input (one stitched block of z19 tiles) that holds no polygon of any class, road included; holes.json and holes.csv break those down, with the blank-tile audit from `tiles`. All descriptive: the rule reads `inventory_all` only.
