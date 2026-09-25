# DEM road grade: richmond (#51)

Written by `scripts/dem_grade.py`; the study is `docs/dem-grade-study.md`.

- DEM: USGS 3DEP via `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage`, 2 m posts (square 1.797e-05 deg pixels), 1946x1655 px in 4 tile(s), 14.7 MB; per-tile sha256 in `tiles.json`.
- grades.csv sha256 `063bf8d292ff3f5b3a824d9579ff3879b254cb562d19c7cd8e72d966ad3af5de` (1596004 bytes; not tracked -- regenerate with the command above and compare).
- Fit window +-20 m of path distance, >= 3 frames.
- Sampled from `results.jsonl` (sha256 `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c`, 9091 rows); `grades.json` records this and `--grade-source` refuses any other file.

## richmond

Panos 9091; out of raster 0; with a grade: sfm 8466, sfm_smoothed 8558, dem_2pt 8466, dem 8558; DEM fit fell back to the two-point value on 2469.

### DEM vs SfM agreement (degrees)

| rig | pair | n | r | slope | median abs diff | p90 abs diff |
|---|---|---:|---:|---:|---:|---:|
| all | DEM (fit) vs SfM (production) | 8466 | 0.35 | 0.40 | 0.48 | 2.99 |
| all | DEM (2-point) vs SfM (production) | 8466 | 0.30 | 0.30 | 0.46 | 2.91 |
| all | DEM (fit) vs SfM (fit) | 8558 | 0.39 | 0.40 | 0.45 | 3.07 |
| all | SfM (production) vs SfM (fit) | 8466 | 0.91 | 0.82 | 0.03 | 0.45 |
| all | DEM (2-point) vs DEM (fit) | 8466 | 0.90 | 0.77 | 0.03 | 0.62 |
| GoPro GoPro Max | DEM (fit) vs SfM (production) | 3217 | 0.30 | 0.40 | 0.80 | 4.62 |
| GoPro GoPro Max | DEM (2-point) vs SfM (production) | 3217 | 0.25 | 0.27 | 0.79 | 4.51 |
| GoPro GoPro Max | DEM (fit) vs SfM (fit) | 3297 | 0.36 | 0.41 | 0.73 | 4.71 |
| GoPro GoPro Max | SfM (production) vs SfM (fit) | 3217 | 0.89 | 0.79 | 0.10 | 0.71 |
| GoPro GoPro Max | DEM (2-point) vs DEM (fit) | 3217 | 0.88 | 0.72 | 0.13 | 1.31 |
| NCTECH LTD iSTAR Pulsar | DEM (fit) vs SfM (production) | 4289 | 0.42 | 0.41 | 0.34 | 1.61 |
| NCTECH LTD iSTAR Pulsar | DEM (2-point) vs SfM (production) | 4289 | 0.42 | 0.40 | 0.33 | 1.54 |
| NCTECH LTD iSTAR Pulsar | DEM (fit) vs SfM (fit) | 4301 | 0.43 | 0.41 | 0.32 | 1.52 |
| NCTECH LTD iSTAR Pulsar | SfM (production) vs SfM (fit) | 4289 | 0.98 | 0.95 | 0.00 | 0.25 |
| NCTECH LTD iSTAR Pulsar | DEM (2-point) vs DEM (fit) | 4289 | 0.96 | 0.93 | 0.00 | 0.23 |
| unknown | DEM (fit) vs SfM (production) | 960 | 0.42 | 0.37 | 0.48 | 2.15 |
| unknown | DEM (2-point) vs SfM (production) | 960 | 0.34 | 0.23 | 0.44 | 2.02 |
| unknown | DEM (fit) vs SfM (fit) | 960 | 0.48 | 0.36 | 0.47 | 2.16 |
| unknown | SfM (production) vs SfM (fit) | 960 | 0.87 | 0.73 | 0.08 | 0.38 |
| unknown | DEM (2-point) vs DEM (fit) | 960 | 0.87 | 0.69 | 0.08 | 0.73 |

### Frame-to-frame roughness and magnitude (degrees)

| rig | grade | lag-1 RMS | median abs grade |
|---|---|---:|---:|
| all | sfm | 2.92 | 1.06 |
| all | sfm_smoothed | 2.36 | 1.04 |
| all | dem_2pt | 2.20 | 0.89 |
| all | dem | 1.45 | 0.87 |
| GoPro GoPro Max | sfm | 4.17 | 1.48 |
| GoPro GoPro Max | sfm_smoothed | 3.33 | 1.46 |
| GoPro GoPro Max | dem_2pt | 2.36 | 1.18 |
| GoPro GoPro Max | dem | 1.22 | 1.16 |
| NCTECH LTD iSTAR Pulsar | sfm | 1.52 | 0.81 |
| NCTECH LTD iSTAR Pulsar | sfm_smoothed | 1.46 | 0.80 |
| NCTECH LTD iSTAR Pulsar | dem_2pt | 1.92 | 0.72 |
| NCTECH LTD iSTAR Pulsar | dem | 1.70 | 0.71 |
| unknown | sfm | 2.01 | 0.98 |
| unknown | sfm_smoothed | 0.76 | 0.97 |
| unknown | dem_2pt | 2.69 | 0.77 |
| unknown | dem | 1.04 | 0.77 |

### Relief ratio: per-sequence slope of SfM altitude on DEM elevation

Sequences with >= 30 frames and >= 0.5 m of DEM relief. 1.0 = the SfM altitude profile carries all of the terrain's relief.

| rig | sequences | p10 | p50 | p90 | share 0.8-1.2 | share negative |
|---|---:|---:|---:|---:|---:|---:|
| all | 61 | 0.38 | 0.98 | 1.16 | 61% | 3% |
| GoPro GoPro Max | 13 | 0.71 | 1.00 | 1.05 | 77% | 0% |
| NCTECH LTD iSTAR Pulsar | 34 | 0.40 | 0.98 | 1.17 | 62% | 0% |
| unknown | 14 | -0.15 | 0.77 | 1.17 | 43% | 14% |

### DEM (fit) vs SfM (production) by |grade| bucket

Bucketed on each source in turn, because they disagree on which frames are steep.

| bucketed on | abs grade | n | r | median abs diff | median abs DEM | median abs SfM |
|---|---|---:|---:|---:|---:|---:|
| DEM | 0-1 | 4667 | 0.17 | 0.40 | 0.42 | 0.60 |
| DEM | 1-2 | 1748 | 0.40 | 0.44 | 1.40 | 1.45 |
| DEM | 2-4 | 1324 | 0.67 | 0.62 | 2.94 | 2.79 |
| DEM | 4+ | 727 | 0.42 | 3.65 | 5.25 | 3.76 |
| SfM | 0-1 | 4072 | 0.21 | 0.34 | 0.47 | 0.42 |
| SfM | 1-2 | 2145 | 0.38 | 0.52 | 1.10 | 1.40 |
| SfM | 2-4 | 1450 | 0.65 | 0.71 | 2.38 | 2.71 |
| SfM | 4+ | 799 | 0.40 | 2.22 | 3.83 | 5.00 |

