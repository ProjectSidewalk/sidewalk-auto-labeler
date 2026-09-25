# DEM road grade: the independent arm of the shuffled-grade control (#51)

Issue: [#51](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/51). Parent:
[#42](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/42) (study:
`docs/mapillary-tilt-study.md`; this repeats the control in §10.5). Related:
[#52](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/52) (the shared-SfM-frame reading).
Written 2026-09-25. **Revised after review** (same day,
[PR #83 review](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/83#issuecomment-5835238567)):
§0, §4.2 and §5 had misread which rule cells the DEM arm fails and overstated the #52 conclusion.
They are corrected below; the verdict and every number are unchanged.

## 0. Summary

- **Verdict: #51 closes negative.** With the grade taken from USGS 3DEP instead of the SfM altitude,
  road-relative fails the pre-registered rule on clauses **(i)** and **(ii)**; (iii) and (iv) pass. So
  `fuse_sites.AUTO_ROAD_SOURCES` stays `()`, and `auto` stays flat for every source.
  `--apply-pose road --grade-source dem` is available as an opt-in. The rule was committed in
  [`5fc4569`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/commit/5fc4569)
  (`eval_sites.dem_verdict`) and pushed before any DEM arm was scored.
- **The DEM and SfM grades agree much better than the planning pilot said.**
  - The fitted DEM grade correlates with production's SfM grade at r = 0.80 in Morgantown and 0.83 in
    Annapolis (median |Δ| 0.35° and 0.18°).
  - Each sequence's SfM altitude profile carries essentially all of the terrain's relief. The
    per-sequence slope of SfM altitude on DEM elevation has a median of 0.96–0.99 in four cities (1.25 in
    Laurens, n = 19).
  - The pilot's Morgantown figures (0.14 and 0.27) were not reproduced. This sampler matches the USGS
    point service to within 4.4 cm at 30 panos across all five cities (§3.1).
  - Agreement is weak only where one grade is noise:
    - Richmond: r = 0.35, driven by the GoPro Max frames.
    - Clovis: flat, r = 0.08, but median |Δ| is only 0.13°. 1,687 frames carry an SfM grade of 4° or
      more where the DEM says 0.3°.
- **In fusion, the DEM grade behaves like the SfM grade.**
  - `road-dem` vs `road`: the median is within ±0.09 m and the p90 within ±0.23 m in every city. An
    independent grade neither rescues the control nor breaks the road frame.
  - **On the same eight-arm set the DEM arm fails six rule cells and the SfM arm four.** Both fail
    (i) in Clovis and Laurens and (ii) in Richmond and Annapolis. The DEM arm fails two more, each
    where the SfM arm passes:
    - (i) in **Annapolis**: the SfM grade beats its shuffle (+0.480 / +0.121 m on p90 / median), the DEM
      grade does not (+0.350 / +0.052 m; the median margin is under 0.1 m).
    - (ii) in **Morgantown**: off-pool recall at 2.5 m falls 1.6 points under the DEM grade against 0.8
      under the SfM grade.
  - Both road arms beat their shuffles in Richmond and Morgantown.
  - Off-pool recall at 2.5 m falls by more than 1 point against the flat raycast under `road-dem`: −4.0
    in Richmond, −1.6 in Morgantown, −2.5 in Annapolis. Gravity loses more (−5.6, −2.8, −3.3), so most of
    this cost comes with rotating rays rather than with either grade.
- **What it means for #52.** #52 suggested that road-relative wins because subtracting the SfM grade
  cancels error that the grade shares with the pitch.
  - Against a *pure* shared-error reading in the hilly cities: the SfM grade is mostly terrain slope
    (r = 0.80 / 0.83 with the DEM grade in Morgantown / Annapolis), and in Richmond and Morgantown a grade
    the SfM never saw reproduces the effect of a frame's own grade over its shuffle.
  - Not ruled out: r = 0.80–0.83 leaves about a third of the SfM grade's variance unexplained by the
    terrain, which is where any shared error would sit. Annapolis is the one pattern the shared-error
    reading predicts (the SfM grade clears its shuffle, the DEM grade does not), though its 0.07 m
    shortfall on 90 common ramps is within this instrument's noise.
  - So the DEM result contradicts a pure shared-SfM-error reading of the hilly cities' grade, and no
    more. The fusion instrument cannot resolve whether the SfM-specific residual matters.

## 1. What was built

- **`scripts/dem_grade.py`**:
  - **Fetch.** Pulls 3DEP from the ImageServer `exportImage` endpoint as uncompressed float32
    GeoTIFFs at 2 m posts. Tiles are at most 1500 px and cover each run's `area.geojson` bbox padded
    by 100 m.
  - **Cache.** Stores tiles under `runs/<city>/dem/tiles/`, with `tiles.json` recording the service,
    request, grid, per-tile sha256 and fetch time. Any response that is not the requested float32
    raster at the requested georeference is refused and never cached. `--verify` re-hashes the cache.
  - **Outputs.** Samples bilinearly at each pano and writes `runs/<city>/dem/grades.csv`,
    `runs/<city>/dem/report.md` and `runs/_summary/dem_grade.csv`. The summary is committed as
    `docs/figures/dem-grade/data/dem_grade.csv`.
- **Square degree-pixels.** The service returns square pixels in EPSG:4326 whatever size is asked for,
  and widens the extent to fit. A request with square posts in metres (non-square in degrees) therefore
  comes back on a different extent from the one requested. So the request grid uses square
  degree-pixels, and every tile's georeference is read from the TIFF and checked against the grid.
- **Four grades per pano:**
  - `grade_sfm_deg`: production's `sequence_grades`, reproduced bit for bit.
  - `grade_dem_2pt_deg`: the same call on DEM elevation.
  - `grade_dem_deg`: a least-squares line through the DEM elevation of every frame within ±20 m of path
    distance (at least 3 frames, else the two-point value). This is what `--grade-source dem` uses.
  - `grade_sfm_smoothed_deg`: the same fit on the SfM altitude.
- **`fuse_sites.py`**:
  - `FuseParams.grade_source`: `sfm` (the default), `sfm-smoothed` or `dem`; the `--grade-source` flag
    sets it.
  - `load_results(grade_source=...)` replaces each grade from `grades.csv` and keeps the bearing.
  - `pose_counts` records `grade_source` and `grade_replaced`.
  - With default flags, Richmond's `sites.jsonl` is byte-identical before and after (sha256
    `f20ac914…`).
- **`eval_sites.py`**:
  - New arms: `road-dem`; `road-dem-shuffled-within` (the same seed-42 within-sequence shuffle, applied
    to the DEM grades); and `road-sfm-smoothed` (descriptive only).
  - The verdict functions take arm names as parameters, and `dem_verdict` is the rule.
  - `mapillary_tilt.py precondition --grade-source dem` runs it.

## 2. The rule (pre-registered on the issue, committed in 5fc4569 before the run)

`road` with the DEM grade earns `AUTO_ROAD_SOURCES = ('mapillary',)` only if **all four** clauses hold.
They are scored on the eight-arm intersected set with a 25 m cap, `BENCHMARK_CONFIDENCE`, the rig mask
off, and the same constants as §10.5.

- **(i)** `road-dem` beats `road-dem-shuffled-within` by more than 0.1 m on both p90 and median
  GT-to-site distance, in at least 4 of 5 cities.
- **(ii)** Its off-pool recall at 2.5 m drops by no more than 1.0 point against `off`, in every city.
- **(iii)** Its unplaceable-mark count exceeds `off`'s by no more than 5% of the off pool, in every city.
- **(iv)** It passes the first #42 rule against `off`: p90 no worse by more than 0.1 m in any city, and
  median better in at least 3.

## 3. DEM vs SfM agreement

### 3.1 Checks on the data path

- **Sampler vs the USGS point query service (EPQS)**, 6 random panos per city. Max |Δ|:
  - Richmond 0.028 m
  - Laurens 0.009 m
  - Morgantown 0.044 m
  - Clovis 0.036 m
  - Annapolis 0.018 m
- **Fetch size and time:** 55 tiles, 415 MB, about 3.5 minutes for the five cities (Clovis alone is
  25 tiles, 213 MB). No pano falls outside the raster.
- **Checksums:** each city's `grades.csv` sha256 is in its `runs/<city>/dem/report.md`, with the
  sha256 of the `results.jsonl` it was sampled from. `grades.json` beside the CSV records both, and
  `--grade-source` refuses a CSV built from another results file (e.g. Laurens' `results.raw.jsonl`),
  a CSV that no longer matches its recorded hash, a graded pano with no row, and a non-finite cell.

### 3.2 Per city (all rigs; degrees)

| City | frames graded | DEM(fit) vs SfM r | slope | median \|Δ\| | p90 \|Δ\| | DEM(2-pt) vs SfM r | both fitted, r | lag-1 RMS SfM / SfM-fit / DEM-2pt / DEM-fit | median \|grade\| SfM / DEM | relief ratio p10 / p50 / p90 (seqs) | share 0.8–1.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| richmond | 8,466 | 0.35 | 0.40 | 0.48 | 2.99 | 0.30 | 0.39 | 2.92 / 2.36 / 2.20 / 1.45 | 1.06 / 0.87 | 0.38 / 0.98 / 1.16 (61) | 61% |
| clovis | 68,812 | 0.08 | 0.67 | 0.13 | 0.59 | 0.09 | 0.09 | 4.31 / 3.68 / 0.52 / 0.40 | 0.24 / 0.21 | 0.60 / 0.99 / 1.43 (412) | 53% |
| morgantown | 51,649 | 0.80 | 0.85 | 0.35 | 1.57 | 0.80 | 0.88 | 2.11 / 0.87 / 1.36 / 0.89 | 2.07 / 2.14 | 0.82 / 0.96 / 1.05 (274) | 89% |
| annapolis | 52,827 | 0.83 | 0.92 | 0.18 | 0.73 | 0.83 | 0.90 | 0.96 / 0.67 / 0.70 / 0.54 | 0.85 / 0.80 | 0.81 / 0.99 / 1.13 (452) | 84% |
| laurens | 4,483 | 0.61 | 0.97 | 0.34 | 1.09 | 0.59 | 0.70 | 0.93 / 0.64 / 0.60 / 0.39 | 0.62 / 0.44 | 1.14 / 1.25 / 1.54 (19) | 37% |

- **Slope** is the OLS slope of the SfM grade on the DEM grade.
- **Relief ratio** is the per-sequence OLS slope of `computed_altitude` on DEM elevation, computed for
  sequences with at least 30 frames and at least 0.5 m of DEM relief.
- Per-rig rows and the tables by |grade| bucket are in each `runs/<city>/dem/report.md`.

Readings:

- **The hilly cities agree.**
  - Morgantown and Annapolis correlate at r = 0.80–0.83, and at 0.88–0.90 once both grades are fitted.
  - Agreement rises with steepness: r = 0.86 in Morgantown's 4°+ bucket on the DEM.
  - Over whole sequences the SfM altitude profile is not compressed: 89% (Morgantown) and 84%
    (Annapolis) of sequences have a relief ratio between 0.8 and 1.2.
- **Richmond's disagreement comes from the GoPro Max frames.** Their SfM grade has a lag-1 RMS of 4.2°,
  against the DEM's 1.2°, and they correlate at only r = 0.30. The iSTAR Pulsar rig agrees better (r =
  0.42, median |Δ| 0.34°) on flatter ground.
- **Clovis's low r comes from SfM outliers on flat ground.** The median |Δ| is only 0.13°. But 1,687
  frames have an SfM grade of 4° or more (median 11.6°) where the DEM says 0.27°. Most are on the GoPro
  Fusion rig (lag-1 RMS 6.0°). These are the failed reconstructions that §5.2 of the tilt study found.
- **Limit: the relief ratio is a sequence-scale statistic.** A per-sequence regression is dominated
  by long-wavelength relief. It refutes the pilot's 0.27, but it says little about fidelity at the
  2–40 m scale the grade is taken over; the per-frame r and slope above are the evidence for that.
- **Limit: compare roughness like for like.** The lag-1 RMS column lists the production two-point SfM
  grade next to the ±20 m fitted DEM grade, and a fitted grade is smoother by construction. Compare
  two-point with two-point (Richmond 2.92° SfM vs 2.20° DEM) or fitted with fitted (2.36° vs 1.45°).
- **Smoothing mostly de-noises the SfM grade.** In Morgantown, fitting the SfM altitude cuts its lag-1
  RMS from 2.11° to 0.87° and raises its agreement with the DEM from r = 0.80 to 0.88. Fitting changes
  the DEM grade less: DEM two-point vs DEM fit correlate at r = 0.87–0.97.

### 3.3 Against the planning pilot

**What the pilot reported.** The plan's pilot was scratch code and was never committed. For
Morgantown it reported:

- DEM-vs-SfM grade correlation of r = 0.14, with a MAD of 4.0°;
- relief ratios of 0.27, and 0.46 for Richmond;
- 12% of sequences with a relief ratio between 0.8 and 1.2.

**What this pipeline measures.** None of those reproduce. Morgantown here gives r = 0.80, a relief-ratio
p50 of 0.96, and 89% of sequences between 0.8 and 1.2. This sampler is checked point by point against
EPQS in all five cities (§3.1).

**Likely cause (a hypothesis).** The pilot's fetch code is not available, so its failure cannot be
confirmed. The most likely cause is the square-pixel behaviour of `exportImage` described in §1:

- A raster sampled on the requested extent, rather than the returned one, is misregistered in
  longitude.
- That error grows with distance from the tile centre.
- So the pilot's spot check at a few panos could pass while its sequence-scale statistics failed.

**Consequence.** The pilot's prediction rested on those numbers ("the sequence's altitude profile
carries roughly a third to a half of the terrain's relief"; "pitch and grade are consistent with each
other and both differ from the ground"). The mechanism it proposed is therefore not supported. The
verdict below comes from the pre-registered rule, which does not depend on the pilot.

## 4. The eight-arm precondition

**Setup.**

- One site set and one GT set, intersected over all eight arms.
- 25 m cap, `BENCHMARK_CONFIDENCE`, rig mask off.
- Association is frozen from the `off` arm.

**Outputs.** Every column is in `docs/figures/mapillary-tilt/data/pose_precondition_dem.csv`. The
run's full output is in `docs/figures/dem-grade/data/precondition_dem.log`.

**Difference from §10.5.** This intersection is slightly tighter than the five-arm one: 802 vs 830
Richmond sites, 839 vs 866 Morgantown sites. So the SfM arms shift a little from that table. The
committed five-arm `pose_precondition.csv` is unchanged.

| City | sites scored (of op.) | common ramps / off pool | arm | median (m) | p90 (m) | R@2.5 | R@5 | R@2.5 off pool | R@5 off pool | GT marks unplaceable (Δ vs off) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| richmond | 802 (1,570) | 76 / 253 | off | 1.30 | 3.53 | 0.877 | 0.886 | 0.870 | 0.881 | 57 (+0) |
|  |  |  | gravity | 1.20 | 2.86 | 0.872 | 0.891 | 0.814 | 0.830 | 58 (+1) |
|  |  |  | road | 1.00 | 2.80 | 0.882 | 0.896 | 0.830 | 0.842 | 59 (+2) |
|  |  |  | road-shuffled-within | 1.20 | 3.15 | 0.872 | 0.886 | 0.791 | 0.802 | 60 (+3) |
|  |  |  | road-shuffled-across | 1.41 | 3.64 | 0.872 | 0.886 | 0.771 | 0.783 | 65 (+8) |
|  |  |  | road-sfm-smoothed | 1.06 | 2.77 | 0.882 | 0.896 | 0.826 | 0.838 | 61 (+4) |
|  |  |  | **road-dem** | 1.08 | 2.72 | 0.877 | 0.891 | 0.830 | 0.842 | 55 (-2) |
|  |  |  | **road-dem-shuffled-within** | 1.18 | 3.39 | 0.872 | 0.886 | 0.783 | 0.794 | 66 (+9) |
| clovis | 2,188 (2,495) | 128 / 174 | off | 1.28 | 3.28 | 0.869 | 0.929 | 0.856 | 0.914 | 21 (+0) |
|  |  |  | gravity | 0.77 | 2.46 | 0.923 | 0.946 | 0.902 | 0.925 | 14 (-7) |
|  |  |  | road | 0.85 | 2.26 | 0.911 | 0.940 | 0.891 | 0.920 | 14 (-7) |
|  |  |  | road-shuffled-within | 0.79 | 2.43 | 0.917 | 0.946 | 0.897 | 0.925 | 14 (-7) |
|  |  |  | road-shuffled-across | 0.75 | 2.33 | 0.917 | 0.946 | 0.891 | 0.920 | 16 (-5) |
|  |  |  | road-sfm-smoothed | 0.81 | 2.32 | 0.917 | 0.940 | 0.897 | 0.920 | 14 (-7) |
|  |  |  | **road-dem** | 0.84 | 2.49 | 0.923 | 0.946 | 0.908 | 0.931 | 13 (-8) |
|  |  |  | **road-dem-shuffled-within** | 0.76 | 2.28 | 0.923 | 0.946 | 0.902 | 0.925 | 14 (-7) |
| morgantown | 839 (1,733) | 74 / 250 | off | 0.92 | 2.34 | 0.825 | 0.849 | 0.808 | 0.832 | 17 (+0) |
|  |  |  | gravity | 0.84 | 2.31 | 0.835 | 0.849 | 0.780 | 0.796 | 22 (+5) |
|  |  |  | road | 0.78 | 2.28 | 0.830 | 0.844 | 0.800 | 0.828 | 21 (+4) |
|  |  |  | road-shuffled-within | 0.99 | 2.60 | 0.830 | 0.849 | 0.788 | 0.812 | 17 (+0) |
|  |  |  | road-shuffled-across | 1.16 | 2.68 | 0.825 | 0.849 | 0.732 | 0.756 | 35 (+18) |
|  |  |  | road-sfm-smoothed | 0.75 | 2.31 | 0.825 | 0.854 | 0.796 | 0.832 | 21 (+4) |
|  |  |  | **road-dem** | 0.70 | 2.20 | 0.821 | 0.854 | 0.792 | 0.836 | 20 (+3) |
|  |  |  | **road-dem-shuffled-within** | 0.94 | 2.57 | 0.825 | 0.854 | 0.788 | 0.816 | 16 (-1) |
| annapolis | 2,443 (4,018) | 90 / 241 | off | 1.32 | 3.05 | 0.872 | 0.891 | 0.846 | 0.876 | 53 (+0) |
|  |  |  | gravity | 0.94 | 2.31 | 0.882 | 0.896 | 0.813 | 0.826 | 56 (+3) |
|  |  |  | road | 0.86 | 2.17 | 0.886 | 0.900 | 0.813 | 0.838 | 50 (-3) |
|  |  |  | road-shuffled-within | 0.98 | 2.65 | 0.877 | 0.900 | 0.805 | 0.826 | 58 (+5) |
|  |  |  | road-shuffled-across | 1.04 | 2.66 | 0.882 | 0.896 | 0.788 | 0.801 | 57 (+4) |
|  |  |  | road-sfm-smoothed | 0.91 | 2.07 | 0.886 | 0.900 | 0.817 | 0.838 | 48 (-5) |
|  |  |  | **road-dem** | 0.91 | 2.32 | 0.886 | 0.900 | 0.822 | 0.842 | 51 (-2) |
|  |  |  | **road-dem-shuffled-within** | 0.96 | 2.67 | 0.877 | 0.900 | 0.805 | 0.826 | 56 (+3) |
| laurens_mapillary | 161 (186) | 115 / 235 | off | 1.79 | 3.84 | 0.529 | 0.670 | 0.511 | 0.660 | 9 (+0) |
|  |  |  | gravity | 1.43 | 2.67 | 0.617 | 0.683 | 0.600 | 0.664 | 13 (+4) |
|  |  |  | road | 1.43 | 2.61 | 0.634 | 0.683 | 0.621 | 0.664 | 12 (+3) |
|  |  |  | road-shuffled-within | 1.33 | 2.74 | 0.599 | 0.670 | 0.583 | 0.651 | 12 (+3) |
|  |  |  | road-shuffled-across | 1.40 | 2.69 | 0.626 | 0.678 | 0.609 | 0.664 | 9 (+0) |
|  |  |  | road-sfm-smoothed | 1.39 | 2.68 | 0.630 | 0.678 | 0.617 | 0.660 | 12 (+3) |
|  |  |  | **road-dem** | 1.39 | 2.72 | 0.630 | 0.683 | 0.617 | 0.664 | 11 (+2) |
|  |  |  | **road-dem-shuffled-within** | 1.42 | 2.83 | 0.599 | 0.670 | 0.579 | 0.647 | 12 (+3) |

### 4.1 The verdict, clause by clause

| City | DEM shuffle − road-dem: p90 / median | (i) | off-pool R@2.5 road-dem − off | (ii) | unplaceable road-dem − off (limit) | (iii) | road-dem − off: p90 / median | (iv) |
|---|---:|:-:|---:|:-:|---:|:-:|---:|:-:|
| richmond | +0.66 / +0.10 m | beat | −4.0 pt | **fail** | −2 (12.7) | ok | −0.81 / −0.22 m | ok |
| clovis | **−0.21 / −0.08** m | **no** | +5.2 pt | ok | −8 (8.7) | ok | −0.79 / −0.44 m | ok |
| morgantown | +0.37 / +0.25 m | beat | −1.6 pt | **fail** | +3 (12.5) | ok | −0.13 / −0.22 m | ok |
| annapolis | +0.35 / **+0.05** m | **no** | −2.5 pt | **fail** | −2 (12.1) | ok | −0.74 / −0.41 m | ok |
| laurens | +0.12 / **+0.04** m | **no** | +10.6 pt | ok | +2 (11.8) | ok | −1.12 / −0.40 m | ok |

- **(i) FAIL.** `road-dem` beats its within-sequence shuffle by more than 0.1 m on both statistics in
  only 2 of 5 cities (Richmond and Morgantown); the rule needs 4.
  - In Clovis the shuffled DEM grade is *better* on both statistics.
  - In Annapolis and Laurens the p90 margin clears 0.1 m but the median margin does not.
- **(ii) FAIL.** Off-pool recall at 2.5 m drops by more than 1.0 point in Richmond (−4.0), Morgantown
  (−1.6) and Annapolis (−2.5).
- **(iii) PASS.** No city adds unplaceable marks beyond 5% of its off pool.
- **(iv) PASS.** The p90 is no worse than `off` in any city, and the median is better in all 5.

**#51 closes negative.** `AUTO_ROAD_SOURCES` stays `()`. On this tighter set the SfM `road` arm also
fails its own control, in the same way as in §10.5: (i) passes in only 3 of 5 cities, and (ii) fails in
Richmond and Annapolis.

### 4.2 Reported, not gating

| City | road-dem − road: p90 / median / off-pool R@2.5 | road-dem − gravity: p90 / median / off-pool R@2.5 | road-sfm-smoothed − road: p90 / median / off-pool R@2.5 |
|---|---|---|---|
| richmond | −0.07 / +0.08 m / +0.0 pt | −0.14 / −0.13 m / +1.6 pt | −0.03 / +0.06 m / −0.4 pt |
| clovis | +0.22 / −0.01 m / +1.7 pt | +0.03 / +0.07 m / +0.6 pt | +0.05 / −0.04 m / +0.6 pt |
| morgantown | −0.08 / −0.09 m / −0.8 pt | −0.11 / −0.15 m / +1.2 pt | +0.03 / −0.03 m / −0.4 pt |
| annapolis | +0.15 / +0.05 m / +0.8 pt | +0.01 / −0.03 m / +0.8 pt | −0.10 / +0.05 m / +0.4 pt |
| laurens | +0.10 / −0.04 m / −0.4 pt | +0.04 / −0.04 m / +1.7 pt | +0.07 / −0.03 m / −0.4 pt |

- **At this instrument's resolution the DEM and SfM grades are interchangeable.**
  - Every `road-dem` − `road` difference is within ±0.23 m, and the sign varies from city to city.
  - `road-sfm-smoothed` behaves the same way (within ±0.10 m), so smoothing, the issue's "cheap fix",
    buys nothing measurable either.
  - #51 was motivated partly by Clovis's "2 recall points". On this pool, road trails gravity by 1.1
    points of off-pool recall in Clovis, and the DEM arm more than makes that up (+1.7 over road).
- **The DEM arm does not sit below gravity, as the pilot predicted.**
  - It beats gravity on the median in 4 of 5 cities, and on off-pool recall in all five, though the
    recall margins (+0.6 to +1.7 pt) are 1 to 4 ramps per pool.
  - So among rotated raycasts, road-relative is still the better frame. What the rule asks is whether
    any rotated raycast should be the default, and the answer is still no.

## 5. What this does and does not settle

- **Settled for the two grades tested: neither makes road-relative pass the control.**
  - The SfM grade was already mostly terrain slope in the hilly cities, and swapping in the DEM grade
    fails more rule cells (six), not fewer (four).
  - Clovis and Laurens fail (i) under both grades: there the tilt belongs to the rig, the grade is near
    zero and does not matter.
  - Richmond and Annapolis fail (ii) under both grades, and gravity pays more of that recall cost than
    either road arm, so most of it comes with the rotation itself.
  - The DEM arm additionally fails (i) in Annapolis and (ii) in Morgantown. Whether some other grade
    would fix those cells is not tested here.
- **Weakened, not settled: #52's shared-error reading.** In Richmond and Morgantown the frame's own
  grade does matter, and an independent grade shows the same effect, which contradicts a *pure*
  shared-SfM-error reading of the hilly cities' grade. About a third of the SfM grade's variance is not
  explained by the terrain, and Annapolis leans slightly the shared-error way (within noise), so a
  shared-error contribution is not ruled out.
- **Not addressed:**
  - Cross-slope. This was closed per #52 (under 0.5° at ramp bearings).
  - The instrument's weakness. It rests on 74–128 common ramps per city, and only 802 of 1,570 Richmond
    sites and 839 of 1,733 Morgantown sites survive the eight-arm intersection.
- **If the default is reopened, fix clause (ii), not the grade source.** The problem is ramps that are
  placed but land farther from any site on the off pool (§10.5, third reading).

## 6. Reproduce

```bash
python scripts/dem_grade.py richmond clovis morgantown annapolis laurens     # fetch + grades + reports
python scripts/dem_grade.py richmond clovis morgantown annapolis laurens --verify
python scripts/mapillary_tilt.py precondition --grade-source dem             # eight-arm table + verdict
python scripts/fuse_sites.py runs/morgantown --apply-pose road --grade-source dem   # opt-in fuse
```
