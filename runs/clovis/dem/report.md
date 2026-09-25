# DEM road grade: clovis (#51)

Written by `scripts/dem_grade.py`; the study is `docs/dem-grade-study.md`.

- DEM: USGS 3DEP via `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage`, 2 m posts (square 1.797e-05 deg pixels), 7358x6780 px in 25 tile(s), 212.7 MB; per-tile sha256 in `tiles.json`.
- grades.csv sha256 `28d5c8ac5ac8764e2bfd8b8f02deb99c47155c3cc1b0451b2d2c2e275f6607b3` (13482374 bytes; not tracked -- regenerate with the command above and compare).
- Fit window +-20 m of path distance, >= 3 frames.

## clovis

Panos 72776; out of raster 0; with a grade: sfm 68812, sfm_smoothed 69818, dem_2pt 68812, dem 69818; DEM fit fell back to the two-point value on 23694.

### DEM vs SfM agreement (degrees)

| rig | pair | n | r | slope | median abs diff | p90 abs diff |
|---|---|---:|---:|---:|---:|---:|
| all | DEM (fit) vs SfM (production) | 68812 | 0.08 | 0.67 | 0.13 | 0.59 |
| all | DEM (2-point) vs SfM (production) | 68812 | 0.09 | 0.60 | 0.13 | 0.52 |
| all | DEM (fit) vs SfM (fit) | 69818 | 0.09 | 0.66 | 0.11 | 0.47 |
| all | SfM (production) vs SfM (fit) | 68812 | 0.89 | 0.80 | 0.01 | 0.19 |
| all | DEM (2-point) vs DEM (fit) | 68812 | 0.87 | 0.75 | 0.01 | 0.18 |
| GoPro Fusion | DEM (fit) vs SfM (production) | 28841 | 0.07 | 0.76 | 0.17 | 1.09 |
| GoPro Fusion | DEM (2-point) vs SfM (production) | 28841 | 0.06 | 0.56 | 0.17 | 1.03 |
| GoPro Fusion | DEM (fit) vs SfM (fit) | 29771 | 0.08 | 0.75 | 0.15 | 0.93 |
| GoPro Fusion | SfM (production) vs SfM (fit) | 28841 | 0.90 | 0.82 | 0.00 | 0.17 |
| GoPro Fusion | DEM (2-point) vs DEM (fit) | 28841 | 0.83 | 0.69 | 0.00 | 0.13 |
| GoPro GoPro Fusion FS1.04.01.80.00 | DEM (fit) vs SfM (production) | 39971 | 0.12 | 0.60 | 0.12 | 0.40 |
| GoPro GoPro Fusion FS1.04.01.80.00 | DEM (2-point) vs SfM (production) | 39971 | 0.15 | 0.64 | 0.11 | 0.34 |
| GoPro GoPro Fusion FS1.04.01.80.00 | DEM (fit) vs SfM (fit) | 40047 | 0.13 | 0.59 | 0.10 | 0.30 |
| GoPro GoPro Fusion FS1.04.01.80.00 | SfM (production) vs SfM (fit) | 39971 | 0.83 | 0.72 | 0.03 | 0.20 |
| GoPro GoPro Fusion FS1.04.01.80.00 | DEM (2-point) vs DEM (fit) | 39971 | 0.90 | 0.80 | 0.03 | 0.20 |

### Frame-to-frame roughness and magnitude (degrees)

| rig | grade | lag-1 RMS | median abs grade |
|---|---|---:|---:|
| all | sfm | 4.31 | 0.24 |
| all | sfm_smoothed | 3.68 | 0.22 |
| all | dem_2pt | 0.52 | 0.23 |
| all | dem | 0.40 | 0.21 |
| GoPro Fusion | sfm | 6.00 | 0.26 |
| GoPro Fusion | sfm_smoothed | 5.17 | 0.24 |
| GoPro Fusion | dem_2pt | 0.72 | 0.22 |
| GoPro Fusion | dem | 0.57 | 0.20 |
| GoPro GoPro Fusion FS1.04.01.80.00 | sfm | 2.63 | 0.23 |
| GoPro GoPro Fusion FS1.04.01.80.00 | sfm_smoothed | 2.09 | 0.21 |
| GoPro GoPro Fusion FS1.04.01.80.00 | dem_2pt | 0.33 | 0.23 |
| GoPro GoPro Fusion FS1.04.01.80.00 | dem | 0.23 | 0.21 |

### Relief ratio: per-sequence slope of SfM altitude on DEM elevation

Sequences with >= 30 frames and >= 0.5 m of DEM relief. 1.0 = the SfM altitude profile carries all of the terrain's relief.

| rig | sequences | p10 | p50 | p90 | share 0.8-1.2 | share negative |
|---|---:|---:|---:|---:|---:|---:|
| all | 412 | 0.60 | 0.99 | 1.43 | 53% | 2% |
| GoPro Fusion | 202 | 0.48 | 1.01 | 1.89 | 44% | 3% |
| GoPro GoPro Fusion FS1.04.01.80.00 | 210 | 0.69 | 0.99 | 1.30 | 62% | 1% |

### DEM (fit) vs SfM (production) by |grade| bucket

Bucketed on each source in turn, because they disagree on which frames are steep.

| bucketed on | abs grade | n | r | median abs diff | median abs DEM | median abs SfM |
|---|---|---:|---:|---:|---:|---:|
| DEM | 0-1 | 67511 | 0.08 | 0.13 | 0.21 | 0.24 |
| DEM | 1-2 | 1106 | 0.16 | 0.45 | 1.21 | 1.13 |
| DEM | 2-4 | 172 | 0.26 | 1.50 | 2.62 | 2.18 |
| DEM | 4+ | 23 | -0.28 | 11.97 | 11.35 | 2.01 |
| SfM | 0-1 | 63693 | 0.63 | 0.12 | 0.20 | 0.22 |
| SfM | 1-2 | 2484 | 0.37 | 0.95 | 0.41 | 1.25 |
| SfM | 2-4 | 948 | 0.15 | 2.66 | 0.26 | 2.78 |
| SfM | 4+ | 1687 | 0.09 | 11.53 | 0.27 | 11.56 |

