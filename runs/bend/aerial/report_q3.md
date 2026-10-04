# bend: Q3 corner anchors vs cluster fragmentation (#104) (Bend is a RampNet training city)

- Tile2Net 0.5.0 commit `c737a4e8c8b907ee3c80739673c67140ce95483d`; tiles: City of Bend 2019 orthoimagery (ArcGIS Online tile cache), fetched by scripts/aerial_fetch_tiles.py and passed with --input at zoom 19 (52480 tiles); polygons.geojson sha256 `fce788012b8c7b4d9df3146a47c018006fa359e34e5ec647c8c27ae3f8109e5a`
- camera height `auto` -> per-rig (gsv-per-rig; per capture year: 2007 2.5 m, 2008 2.5 m, 2009 2.5 m, 2011 2.5 m, 2012 2.5 m, 2015 2.5 m, 2017 2.5 m, 2018 2.5 m, 2019 2.5 m, 2021 2.5 m, 2024 2.5 m, 2025 2.5 m)
- 1833 anchors (1833 from polygon contacts alone; 0 network crosswalk endpoints joined, merged within 2 m); 0.1242 of the 298 GT pool ramps have an anchor within 3 m (median distance 115.032 m)
- labels synthesized at tier 0.55 as `eval_ps_clustering.py --offline` does; the `base` rows must reproduce that script's committed `ps @ 7.5 m` and `fusion` rows

| base | arm | clusters | coverage | frag 5 m (extra) | frag 3 m | dual (both/pairs) | precision (TP/FP) | same-pano pairs |
|---|---|---:|---:|---|---:|---|---|---:|
| fusion | base | 14058 | 0.9497 (283/298) | 0.1449 (44) | 0.053 | 0.8529 (29/34) | 0.9565 (242/11) | 0 |
| fusion | anchored | 14058 | 0.9396 (280/298) | 0.1429 (44) | 0.0393 | 0.8529 (29/34) | 0.9565 (242/11) | 0 |
| fusion | merge-at-anchor | 13851 | 0.9295 (277/298) | 0.1336 (39) | 0.0397 | 0.7647 (26/34) | 0.9597 (238/10) | 495 |
| ps @ 7.5 m | base | 14650 | 0.943 (281/298) | 0.2064 (66) | 0.089 | 0.7941 (27/34) | 0.9574 (247/11) | 0 |
| ps @ 7.5 m | anchored | 14650 | 0.9396 (280/298) | 0.2036 (65) | 0.1 | 0.7941 (27/34) | 0.9574 (247/11) | 0 |
| ps @ 7.5 m | merge-at-anchor | 14354 | 0.9295 (277/298) | 0.1733 (52) | 0.0686 | 0.7059 (24/34) | 0.9605 (243/10) | 484 |

Gate (pre-registered on #104): an arm passes when, in BOTH cities, frag 5 m falls by >= 0.03 absolute against its base while coverage and dual-ramp separation fall by no more than 0.01 / 0.01. `merge-at-anchor` can put two detections from one pano in one cluster (the fusion cannot-link); the same-pano column counts that.
