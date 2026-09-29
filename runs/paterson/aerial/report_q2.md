# paterson: Q2 projected sidewalk edges vs reviewer marks (#104)

- Tile2Net 0.5.0 commit `c737a4e8c8b907ee3c80739673c67140ce95483d`; tiles: nj (Tile2Net built-in NewJersey source) at zoom 19 (12064 tiles); polygons.geojson sha256 `0867561c7e04fa205f097f639cb113ee177c29ef88f1f1242bd3bd577c61ea45`
- camera height `auto` -> per-rig (gsv-per-rig; per capture year: 2007 2.5 m, 2008 2.5 m, 2012 2.5 m, 2013 2.5 m, 2014 2.5 m, 2015 2.5 m, 2016 2.5 m, 2017 2.5 m, 2018 2.5 m, 2019 2.5 m, 2020 2.5 m, 2021 2.5 m, 2022 2.5 m, 2023 2.5 m, 2024 2.5 m, 2025 2 m, 2026 2 m)
- 98 judged panos with at least one reference mark and their whole 25 m disk inside the Tile2Net coverage (0 more left out)

| height | reference | n | with an edge in range | px p50 | px p90 | dx p50 | dy p50 | share <= 5 px | world dist p50 (m) | ref on surface |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | all | 395 | 390 | 3.409 | 14.602 | -0.014 | 0.307 | 0.6333 | 0.0 | 0.6595 |
| auto | box | 109 | 109 | 3.867 | 18.943 | -0.101 | 0.237 | 0.5963 | 0.0 | 0.7609 |
| auto | peak | 184 | 181 | 3.223 | 11.803 | -0.027 | 0.015 | 0.6354 | 0.0 | 0.6538 |
| auto | missed | 102 | 100 | 3.531 | 17.605 | 0.049 | 0.575 | 0.67 | 0.0 | 0.5513 |
| 2.6 | all | 395 | 390 | 4.575 | 16.952 | -0.011 | 1.424 | 0.5231 | 0.0 | 0.6391 |
| 2.6 | box | 109 | 109 | 5.086 | 24.762 | -0.032 | 1.567 | 0.4862 | 0.0 | 0.716 |
| 2.6 | peak | 184 | 181 | 4.414 | 15.141 | -0.123 | 1.045 | 0.5249 | 0.0 | 0.6474 |
| 2.6 | missed | 102 | 100 | 4.365 | 19.064 | 0.077 | 2.125 | 0.56 | 0.0 | 0.5231 |

px = distance in 1024x512 heatmap pixels (the grid RampNet predicts on; 1 px = 0.35 deg) from the reference to the nearest projected sidewalk/crosswalk boundary sample (every 0.2 m, within the 25 m raycast cap); dx/dy = that nearest sample minus the reference (dy > 0 = the edge projects lower, i.e. nearer). It mixes mask error with projection error (height, heading, GPS) by construction. `box` = the reviewer's extent-box centre (Paterson only), `peak` = a verdict-true detection with no box, `missed` = a missed-ramp click. world dist = the reference raycast at that height, metres to the nearest polygon (0 inside).
