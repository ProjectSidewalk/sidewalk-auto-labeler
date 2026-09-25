# DEM road grade: laurens (#51)

Written by `scripts/dem_grade.py`; the study is `docs/dem-grade-study.md`.

- DEM: USGS 3DEP via `https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage`, 2 m posts (square 1.797e-05 deg pixels), 1329x1048 px in 1 tile(s), 6.5 MB; per-tile sha256 in `tiles.json`.
- grades.csv sha256 `01e6151dc634191f8b04ab5c2a0b79bd0ede1c5ec7b12ca42970d5ff3fc06896` (869683 bytes; not tracked -- regenerate with the command above and compare).
- Fit window +-20 m of path distance, >= 3 frames.
- Sampled from `results.jsonl` (sha256 `16c5a348b739274bf8e7623b63551da5245547d0ca4e5a52956fd2413b4fe213`, 4495 rows); `grades.json` records this and `--grade-source` refuses any other file.

## laurens

Panos 4495; out of raster 0; with a grade: sfm 4483, sfm_smoothed 4489, dem_2pt 4483, dem 4489; DEM fit fell back to the two-point value on 479.

### DEM vs SfM agreement (degrees)

| rig | pair | n | r | slope | median abs diff | p90 abs diff |
|---|---|---:|---:|---:|---:|---:|
| all | DEM (fit) vs SfM (production) | 4483 | 0.61 | 0.97 | 0.34 | 1.09 |
| all | DEM (2-point) vs SfM (production) | 4483 | 0.59 | 0.83 | 0.33 | 1.08 |
| all | DEM (fit) vs SfM (fit) | 4489 | 0.70 | 0.95 | 0.30 | 0.89 |
| all | SfM (production) vs SfM (fit) | 4483 | 0.90 | 0.77 | 0.03 | 0.49 |
| all | DEM (2-point) vs DEM (fit) | 4483 | 0.90 | 0.79 | 0.03 | 0.44 |
| GoPro GoPro Max | DEM (fit) vs SfM (production) | 4483 | 0.61 | 0.97 | 0.34 | 1.09 |
| GoPro GoPro Max | DEM (2-point) vs SfM (production) | 4483 | 0.59 | 0.83 | 0.33 | 1.08 |
| GoPro GoPro Max | DEM (fit) vs SfM (fit) | 4489 | 0.70 | 0.95 | 0.30 | 0.89 |
| GoPro GoPro Max | SfM (production) vs SfM (fit) | 4483 | 0.90 | 0.77 | 0.03 | 0.49 |
| GoPro GoPro Max | DEM (2-point) vs DEM (fit) | 4483 | 0.90 | 0.79 | 0.03 | 0.44 |

### Frame-to-frame roughness and magnitude (degrees)

| rig | grade | lag-1 RMS | median abs grade |
|---|---|---:|---:|
| all | sfm | 0.93 | 0.62 |
| all | sfm_smoothed | 0.64 | 0.58 |
| all | dem_2pt | 0.60 | 0.47 |
| all | dem | 0.39 | 0.44 |
| GoPro GoPro Max | sfm | 0.93 | 0.62 |
| GoPro GoPro Max | sfm_smoothed | 0.64 | 0.58 |
| GoPro GoPro Max | dem_2pt | 0.60 | 0.47 |
| GoPro GoPro Max | dem | 0.39 | 0.44 |

### Relief ratio: per-sequence slope of SfM altitude on DEM elevation

Sequences with >= 30 frames and >= 0.5 m of DEM relief. 1.0 = the SfM altitude profile carries all of the terrain's relief.

| rig | sequences | p10 | p50 | p90 | share 0.8-1.2 | share negative |
|---|---:|---:|---:|---:|---:|---:|
| all | 19 | 1.14 | 1.25 | 1.54 | 37% | 0% |
| GoPro GoPro Max | 19 | 1.14 | 1.25 | 1.54 | 37% | 0% |

### DEM (fit) vs SfM (production) by |grade| bucket

Bucketed on each source in turn, because they disagree on which frames are steep.

| bucketed on | abs grade | n | r | median abs diff | median abs DEM | median abs SfM |
|---|---|---:|---:|---:|---:|---:|
| DEM | 0-1 | 3830 | 0.50 | 0.34 | 0.37 | 0.53 |
| DEM | 1-2 | 572 | 0.86 | 0.35 | 1.26 | 1.26 |
| DEM | 2-4 | 81 | 0.82 | 0.53 | 2.23 | 2.14 |
| DEM | 4+ | 0 | — | — | — | — |
| SfM | 0-1 | 3205 | 0.60 | 0.27 | 0.36 | 0.42 |
| SfM | 1-2 | 1025 | 0.78 | 0.58 | 0.79 | 1.31 |
| SfM | 2-4 | 222 | 0.79 | 1.32 | 1.18 | 2.46 |
| SfM | 4+ | 31 | 0.40 | 5.42 | 0.53 | 5.29 |

