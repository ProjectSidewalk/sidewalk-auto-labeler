# DEM road grade: annapolis (#51)

Written by `scripts/dem_grade.py`; the study is `docs/dem-grade-study.md`.

- DEM: USGS 3DEP via `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage`, 2 m posts (square 1.797e-05 deg pixels), 4076x3443 px in 9 tile(s), 60.6 MB; per-tile sha256 in `tiles.json`.
- grades.csv sha256 `4b97c0e8ce228ad4315f1dd81a2bdc474375c7545df6bcffcda696a2d358304e` (10197238 bytes; not tracked -- regenerate with the command above and compare).
- Fit window +-20 m of path distance, >= 3 frames.

## annapolis

Panos 53232; out of raster 0; with a grade: sfm 52827, sfm_smoothed 52882, dem_2pt 52827, dem 52882; DEM fit fell back to the two-point value on 3708.

### DEM vs SfM agreement (degrees)

| rig | pair | n | r | slope | median abs diff | p90 abs diff |
|---|---|---:|---:|---:|---:|---:|
| all | DEM (fit) vs SfM (production) | 52827 | 0.83 | 0.92 | 0.18 | 0.73 |
| all | DEM (2-point) vs SfM (production) | 52827 | 0.83 | 0.87 | 0.14 | 0.55 |
| all | DEM (fit) vs SfM (fit) | 52882 | 0.90 | 0.91 | 0.12 | 0.52 |
| all | SfM (production) vs SfM (fit) | 52827 | 0.92 | 0.85 | 0.08 | 0.42 |
| all | DEM (2-point) vs DEM (fit) | 52827 | 0.95 | 0.90 | 0.08 | 0.39 |
| Trimble Trimble MX7 | DEM (fit) vs SfM (production) | 15475 | 0.72 | 0.81 | 0.12 | 0.60 |
| Trimble Trimble MX7 | DEM (2-point) vs SfM (production) | 15475 | 0.72 | 0.78 | 0.10 | 0.41 |
| Trimble Trimble MX7 | DEM (fit) vs SfM (fit) | 15493 | 0.77 | 0.81 | 0.08 | 0.36 |
| Trimble Trimble MX7 | SfM (production) vs SfM (fit) | 15475 | 0.90 | 0.83 | 0.04 | 0.32 |
| Trimble Trimble MX7 | DEM (2-point) vs DEM (fit) | 15475 | 0.96 | 0.91 | 0.04 | 0.30 |
| Trimble Trimble mx7 | DEM (fit) vs SfM (production) | 37352 | 0.88 | 0.96 | 0.21 | 0.77 |
| Trimble Trimble mx7 | DEM (2-point) vs SfM (production) | 37352 | 0.88 | 0.91 | 0.16 | 0.59 |
| Trimble Trimble mx7 | DEM (fit) vs SfM (fit) | 37389 | 0.95 | 0.95 | 0.14 | 0.56 |
| Trimble Trimble mx7 | SfM (production) vs SfM (fit) | 37352 | 0.93 | 0.86 | 0.10 | 0.46 |
| Trimble Trimble mx7 | DEM (2-point) vs DEM (fit) | 37352 | 0.95 | 0.90 | 0.09 | 0.42 |

### Frame-to-frame roughness and magnitude (degrees)

| rig | grade | lag-1 RMS | median abs grade |
|---|---|---:|---:|
| all | sfm | 0.96 | 0.85 |
| all | sfm_smoothed | 0.67 | 0.82 |
| all | dem_2pt | 0.70 | 0.83 |
| all | dem | 0.54 | 0.80 |
| Trimble Trimble MX7 | sfm | 1.27 | 0.76 |
| Trimble Trimble MX7 | sfm_smoothed | 1.00 | 0.74 |
| Trimble Trimble MX7 | dem_2pt | 0.83 | 0.72 |
| Trimble Trimble MX7 | dem | 0.71 | 0.70 |
| Trimble Trimble mx7 | sfm | 0.81 | 0.90 |
| Trimble Trimble mx7 | sfm_smoothed | 0.47 | 0.86 |
| Trimble Trimble mx7 | dem_2pt | 0.63 | 0.88 |
| Trimble Trimble mx7 | dem | 0.45 | 0.84 |

### Relief ratio: per-sequence slope of SfM altitude on DEM elevation

Sequences with >= 30 frames and >= 0.5 m of DEM relief. 1.0 = the SfM altitude profile carries all of the terrain's relief.

| rig | sequences | p10 | p50 | p90 | share 0.8-1.2 | share negative |
|---|---:|---:|---:|---:|---:|---:|
| all | 452 | 0.81 | 0.99 | 1.13 | 84% | 0% |
| Trimble Trimble MX7 | 162 | 0.91 | 1.00 | 1.06 | 91% | 0% |
| Trimble Trimble mx7 | 290 | 0.79 | 0.97 | 1.17 | 80% | 0% |

### DEM (fit) vs SfM (production) by |grade| bucket

Bucketed on each source in turn, because they disagree on which frames are steep.

| bucketed on | abs grade | n | r | median abs diff | median abs DEM | median abs SfM |
|---|---|---:|---:|---:|---:|---:|
| DEM | 0-1 | 31271 | 0.52 | 0.17 | 0.48 | 0.51 |
| DEM | 1-2 | 13609 | 0.86 | 0.18 | 1.36 | 1.36 |
| DEM | 2-4 | 6841 | 0.96 | 0.21 | 2.59 | 2.58 |
| DEM | 4+ | 1106 | 0.85 | 0.30 | 4.62 | 4.54 |
| SfM | 0-1 | 29930 | 0.77 | 0.15 | 0.49 | 0.47 |
| SfM | 1-2 | 14415 | 0.86 | 0.20 | 1.29 | 1.38 |
| SfM | 2-4 | 7209 | 0.95 | 0.23 | 2.47 | 2.59 |
| SfM | 4+ | 1273 | 0.70 | 0.37 | 4.38 | 4.74 |

