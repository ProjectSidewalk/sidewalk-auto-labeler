# paterson: Q3 corner anchors vs cluster fragmentation (#104)

- Tile2Net 0.5.0 commit `c737a4e8c8b907ee3c80739673c67140ce95483d`; tiles: nj (Tile2Net built-in NewJersey source) at zoom 19 (12064 tiles); polygons.geojson sha256 `0867561c7e04fa205f097f639cb113ee177c29ef88f1f1242bd3bd577c61ea45`
- camera height `auto` -> per-rig (gsv-per-rig; per capture year: 2007 2.5 m, 2008 2.5 m, 2012 2.5 m, 2013 2.5 m, 2014 2.5 m, 2015 2.5 m, 2016 2.5 m, 2017 2.5 m, 2018 2.5 m, 2019 2.5 m, 2020 2.5 m, 2021 2.5 m, 2022 2.5 m, 2023 2.5 m, 2024 2.5 m, 2025 2 m, 2026 2 m)
- 5815 anchors (3853 from polygon contacts alone; 4650 network crosswalk endpoints joined, merged within 2 m); 0.5387 of the 323 GT pool ramps have an anchor within 3 m (median distance 2.809 m)
- labels synthesized at tier 0.55 as `eval_ps_clustering.py --offline` does; the `base` rows must reproduce that script's committed `ps @ 7.5 m` and `fusion` rows

| base | arm | clusters | coverage | frag 5 m (extra) | frag 3 m | dual (both/pairs) | precision (TP/FP) | same-pano pairs |
|---|---|---:|---:|---|---:|---|---|---:|
| fusion | base | 7172 | 0.9443 (305/323) | 0.1279 (39) | 0.0656 | 0.85 (85/100) | 0.9749 (233/6) | 0 |
| fusion | anchored | 7172 | 0.9443 (305/323) | 0.141 (43) | 0.059 | 0.85 (85/100) | 0.9749 (233/6) | 0 |
| fusion | merge-at-anchor | 6197 | 0.7771 (251/323) | 0.0757 (19) | 0.0279 | 0.31 (31/100) | 0.972 (208/6) | 1768 |
| ps @ 7.5 m | base | 8383 | 0.9443 (305/323) | 0.2984 (96) | 0.177 | 0.85 (85/100) | 0.9783 (271/6) | 0 |
| ps @ 7.5 m | anchored | 8383 | 0.9412 (304/323) | 0.3125 (101) | 0.1711 | 0.84 (84/100) | 0.9783 (271/6) | 0 |
| ps @ 7.5 m | merge-at-anchor | 7114 | 0.805 (260/323) | 0.1423 (37) | 0.05 | 0.39 (39/100) | 0.9767 (252/6) | 1755 |

Gate (pre-registered on #104): an arm passes when, in BOTH cities, frag 5 m falls by >= 0.03 absolute against its base while coverage and dual-ramp separation fall by no more than 0.01 / 0.01. `merge-at-anchor` can put two detections from one pano in one cluster (the fusion cannot-link); the same-pano column counts that.
