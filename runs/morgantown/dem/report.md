# DEM road grade: morgantown (#51)

Written by `scripts/dem_grade.py`; the study is `docs/dem-grade-study.md`.

- DEM: USGS 3DEP via `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage`, 2 m posts (square 1.797e-05 deg pixels), 5092x5619 px in 16 tile(s), 120.9 MB; per-tile sha256 in `tiles.json`.
- grades.csv sha256 `886515e1ed00245d3d9fa1caf12b7ccc194f1d00d50a1af582da5f0e440fd913` (9962045 bytes; not tracked -- regenerate with the command above and compare).
- Fit window +-20 m of path distance, >= 3 frames.
- Sampled from `results.jsonl` (sha256 `7dbf24e03574c31b0a3011490177a5d51dc20e30badf7fdd35da443e870b98bc`, 51692 rows); `grades.json` records this and `--grade-source` refuses any other file.

## morgantown

Panos 51692; out of raster 0; with a grade: sfm 51649, sfm_smoothed 51672, dem_2pt 51649, dem 51672; DEM fit fell back to the two-point value on 392.

### DEM vs SfM agreement (degrees)

| rig | pair | n | r | slope | median abs diff | p90 abs diff |
|---|---|---:|---:|---:|---:|---:|
| all | DEM (fit) vs SfM (production) | 51649 | 0.80 | 0.85 | 0.35 | 1.57 |
| all | DEM (2-point) vs SfM (production) | 51649 | 0.80 | 0.81 | 0.29 | 1.25 |
| all | DEM (fit) vs SfM (fit) | 51672 | 0.88 | 0.84 | 0.25 | 1.10 |
| all | SfM (production) vs SfM (fit) | 51649 | 0.93 | 0.84 | 0.15 | 0.97 |
| all | DEM (2-point) vs DEM (fit) | 51649 | 0.97 | 0.92 | 0.16 | 1.00 |
| GoPro GoPro Max | DEM (fit) vs SfM (production) | 51649 | 0.80 | 0.85 | 0.35 | 1.57 |
| GoPro GoPro Max | DEM (2-point) vs SfM (production) | 51649 | 0.80 | 0.81 | 0.29 | 1.25 |
| GoPro GoPro Max | DEM (fit) vs SfM (fit) | 51672 | 0.88 | 0.84 | 0.25 | 1.10 |
| GoPro GoPro Max | SfM (production) vs SfM (fit) | 51649 | 0.93 | 0.84 | 0.15 | 0.97 |
| GoPro GoPro Max | DEM (2-point) vs DEM (fit) | 51649 | 0.97 | 0.92 | 0.16 | 1.00 |

### Frame-to-frame roughness and magnitude (degrees)

| rig | grade | lag-1 RMS | median abs grade |
|---|---|---:|---:|
| all | sfm | 2.11 | 2.07 |
| all | sfm_smoothed | 0.87 | 2.04 |
| all | dem_2pt | 1.36 | 2.17 |
| all | dem | 0.89 | 2.14 |
| GoPro GoPro Max | sfm | 2.11 | 2.07 |
| GoPro GoPro Max | sfm_smoothed | 0.87 | 2.04 |
| GoPro GoPro Max | dem_2pt | 1.36 | 2.17 |
| GoPro GoPro Max | dem | 0.89 | 2.14 |

### Relief ratio: per-sequence slope of SfM altitude on DEM elevation

Sequences with >= 30 frames and >= 0.5 m of DEM relief. 1.0 = the SfM altitude profile carries all of the terrain's relief.

| rig | sequences | p10 | p50 | p90 | share 0.8-1.2 | share negative |
|---|---:|---:|---:|---:|---:|---:|
| all | 274 | 0.82 | 0.96 | 1.05 | 89% | 0% |
| GoPro GoPro Max | 274 | 0.82 | 0.96 | 1.05 | 89% | 0% |

### DEM (fit) vs SfM (production) by |grade| bucket

Bucketed on each source in turn, because they disagree on which frames are steep.

| bucketed on | abs grade | n | r | median abs diff | median abs DEM | median abs SfM |
|---|---|---:|---:|---:|---:|---:|
| DEM | 0-1 | 15143 | 0.22 | 0.26 | 0.46 | 0.53 |
| DEM | 1-2 | 9681 | 0.60 | 0.32 | 1.43 | 1.39 |
| DEM | 2-4 | 13176 | 0.81 | 0.36 | 2.88 | 2.75 |
| DEM | 4+ | 13649 | 0.86 | 0.49 | 5.69 | 5.46 |
| SfM | 0-1 | 15551 | 0.24 | 0.27 | 0.54 | 0.47 |
| SfM | 1-2 | 9679 | 0.62 | 0.32 | 1.45 | 1.44 |
| SfM | 2-4 | 13168 | 0.89 | 0.36 | 2.89 | 2.84 |
| SfM | 4+ | 13251 | 0.85 | 0.48 | 5.55 | 5.76 |

