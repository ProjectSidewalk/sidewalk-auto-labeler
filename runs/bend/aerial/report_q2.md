# bend: Q2 projected sidewalk edges vs reviewer marks (#104) (Bend is a RampNet training city)

- Tile2Net 0.5.0 commit `c737a4e8c8b907ee3c80739673c67140ce95483d`; tiles: City of Bend 2019 orthoimagery (ArcGIS Online tile cache), fetched by scripts/aerial_fetch_tiles.py and passed with --input at zoom 19 (52480 tiles); polygons.geojson sha256 `fce788012b8c7b4d9df3146a47c018006fa359e34e5ec647c8c27ae3f8109e5a`
- camera height `auto` -> per-rig (gsv-per-rig; per capture year: 2007 2.5 m, 2008 2.5 m, 2009 2.5 m, 2011 2.5 m, 2012 2.5 m, 2015 2.5 m, 2017 2.5 m, 2018 2.5 m, 2019 2.5 m, 2021 2.5 m, 2024 2.5 m, 2025 2.5 m)
- 96 judged panos with at least one reference mark and their whole 25 m disk inside the Tile2Net coverage (0 more left out)

| height | reference | n | with an edge in range | px p50 | px p90 | dx p50 | dy p50 | share <= 5 px | displaced px p50 / p90 | displaced share <= 5 px | p50 minus displaced | world dist p50 (m) | ref on surface |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| auto | all | 327 | 300 | 3.613 | 83.03 | -0.047 | -0.372 | 0.5867 | 9.573 / 84.008 | 0.3033 | -5.96 | 0.365 | 0.3712 |
| auto | peak | 248 | 223 | 2.992 | 80.47 | -0.011 | -0.554 | 0.6502 | 9.291 / 81.55 | 0.3094 | -6.299 | 0.282 | 0.3951 |
| auto | missed | 79 | 77 | 6.474 | 98.288 | -0.215 | 0.551 | 0.4026 | 11.095 / 93.994 | 0.2857 | -4.621 | 1.168 | 0.2679 |
| 2.6 | all | 327 | 300 | 3.674 | 82.966 | -0.052 | 0.048 | 0.56 | 9.862 / 83.705 | 0.3167 | -6.188 | 0.417 | 0.3535 |
| 2.6 | peak | 248 | 223 | 2.919 | 79.896 | -0.061 | -0.214 | 0.6188 | 9.182 / 82.631 | 0.3274 | -6.263 | 0.345 | 0.3827 |
| 2.6 | missed | 79 | 77 | 6.817 | 98.164 | -0.052 | 0.936 | 0.3896 | 11.006 / 94.118 | 0.2857 | -4.189 | 1.108 | 0.2222 |

px = distance in 1024x512 heatmap pixels (the grid RampNet predicts on; 1 px = 0.35 deg) from the reference to the nearest projected sidewalk/crosswalk boundary sample (every 0.2 m, within the 25 m raycast cap); dx/dy = that nearest sample minus the reference (dy > 0 = the edge projects lower in the pano than the mark, i.e. at a larger dip below the horizon). `with an edge in range` = marks whose pano has at least one projected edge sample; px, dx, dy and the shares are over those marks only (the rest have no sidewalk/crosswalk polygon within 25 m of the pano). `displaced` = the chance floor: the same mark moved 40 px left or right (side seeded per mark) in the same pano, measured against the same edges. Because the metric is a NEAREST-edge distance, any height that packs the projected edges closer together lowers both rows, so compare heights on `p50 minus displaced`, not on px p50 alone. It mixes mask error with projection error (height, heading, GPS) by construction. `box` = the reviewer's extent-box centre (Paterson only), `peak` = a verdict-true detection with no box, `missed` = a missed-ramp click. world dist = the reference raycast at that height, metres to the nearest polygon (0 inside).
