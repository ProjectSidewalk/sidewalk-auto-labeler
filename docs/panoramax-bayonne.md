# Bayonne: the first full Panoramax run, and whether the borrowed error model fits (#57)

Issue [#57](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57), part 1.
Part 2 of the issue was handled in PR #70.

Panoramax is the federated open imagery commons. `sources/panoramax.py` shipped in PR #49
and had only a 12-pano smoke test behind it. This document covers three things:

- the first full city run (Bayonne, France: OSM commune relation 166713, 25.8 km²);
- a ground-truth-free measurement of the fusion error model that `geo.error_model_for`
  borrows from Mapillary for this source;
- the hand-off of a benchmark bundle to RampNet for ground truth.

The reading rule was posted on #57 before any Bayonne residual was read
([pre-registration](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57#issuecomment-5921184460)).
It was amended once, also before any Bayonne residual was read, after the Mapillary
comparators showed the absolute window was mis-aimed
([amendment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57#issuecomment-5921233648)).

**Nothing in production changes here.** `error_model_for('panoramax')` still returns
`MAPILLARY_ERRORS`, and fusion's pose default is unchanged.

## 1. The run

| | |
|---|---|
| command | `main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --reuse-scan --thin-spacing 10` (lab A40, batch 1, default concurrency) |
| scan | 55 of 100 bbox z15 tiles within 50 m of the area; **73,161** in-area 360 pictures |
| thinning | 10 m: **28,634** panos (5 m would give 50,528) |
| processed / skipped / failed | **28,524** / 110 / **0** in one pass, so no drain rerun was needed |
| wall time | 6.7 h, 2026-09-30 22:57 to 2026-10-01 05:39 UTC; 1.18 panos/s (24,062 s in forward) |
| model | `rampnet-model@606a11956743` (bound on this resume; the 2026-09-04 smoke run predated #39) |
| `results.jsonl` sha256 | `4f38ff528d4445b21eb02e99427e0ba3c88dfb5c940cc4702a283182e379b4b3` (141 MB; identical on makelab2, locally and in the archive root) |

**Thinning decision.** The rule was stated before the scan. The source default of 5 m (the
density of every Mapillary run) would be used unless the scan-only estimate exceeded 16 h,
in which case 10 m. The estimate printed 21.1 h for 50,528 panos at 1.5 s/pano, so the run
used 10 m.

The A40 then ran at 0.85 s/pano, so a 5 m run would have taken about 12 h. The estimate's
constant is an RTX 3070 figure. The cost of the decision is fewer views per fused site
than the Mapillary comparators have, and the residuals below carry that caveat. Note also
that `manifest.json` does not record the thin spacing; the run log does.

**Skips (110, all cached).**

| count | cause |
|---:|---|
| 104 | GoPro **MAX2** uploads on the OSM-FR instance. The STAC item declares 7680x3840, but the `hd` image is 7680x2940: a vertically cropped equirect, not 2:1. The 2:1 check after download skips them, which is correct. Padding these from their crop metadata would recover them; that is not done here. |
| 3 | No `view:azimuth`. Correct. |
| 3 | `5376x2688` / `5760x2880` IGN panos whose metadata and pixels look fine today. The cause at run time was not recorded, because main.py logs skip counts but not per-pano reasons. A transient asset 404 is the likely reading (the source caches a 404 as permanent). To retry them, delete `139d37d3-9c50-4202-964e-c50c92712a49`, `1f8be062-1ae4-4f68-a5d4-07f3ac494b86` and `2c902fc4-669d-4b9c-9e9e-1eafb68a94f7` from `already_processed.txt` and rerun. At 0.01% of the run this was left alone. |

There were no HTTP 429s and no 5xx failures from either instance.

**Census** (`scripts/run_census.py`; data in `figures/panoramax-bayonne/data/census/`):

| | |
|---|---|
| rig | GoPro Max 99.9%: 5760x2880 for 58.0% of panos, 5376x2688 for 41.9%. The rest (39 panos) is a Kandao QooCam 3 Ultra, a Samsung phone, and 25 `GlobalExtractor.py` re-exports. |
| capture year | 2026 56.0%, 2025 13.0%, 2024 28.0%, 2023 1.9%, 2022 1.0% |
| producer | `sig_bayonne` (the municipality, IGN instance, Etalab 2.0) 95.3%; 8 other accounts. The OSM-FR instance accounts are CC BY-SA 4.0. |
| pose | absent 42.0%, real tilt reported 36.8%, reported as exactly 0/0 21.2%. This is far more posed than the 28% of the six-city sample on #57. |
| detections, tier 0.30 | 10,011 on 6,289 panos (22.1%); 0.351 per pano |
| detections, tier 0.55 | 3,861 on 2,947 panos (10.3%); **0.135 per pano** |

For comparison at 0.55: Richmond 1.05 and Annapolis 0.55 detections per pano. Both are
Mapillary runs from before the storage floor, so their 0.30 tier is their 0.55 tier.

A per-pano rate 4 to 8 times lower is the first thing ground truth has to explain. It could
be fewer ramps per frame at 10 m spacing, French lowered kerbs (*abaissés de trottoir*)
looking unlike the US ramps RampNet was trained on, or missed detections. This run cannot
tell those apart.

**The nadir logo band.** The municipal GoPro Max panos carry a solid white band (the
"Bayonne Pays Basque" logo) from y = 0.791 to the bottom of the equirect. Measured on one
5376x2688 and one 5760x2880 pano: the band starts at row 2126 / 2278, which is a dip of
52.4°.

That is entirely inside the rig mask (`NADIR_MASK_DEG` = 49°, y >= 0.772), so nothing in
the band can ship. Stored detections in the band: **2 of 10,011** at 0.30 (0.02%) and
**0 of 3,861** at 0.55. Detections on the rig at all (dip >= 49°): 9 at 0.30, 1 at 0.55.
The band costs recall within about 2 m of the camera, which 2.6 m / tan(52.4°) puts at 2.0 m.

The car body (white roof and bonnet) is visible from about y = 0.6 at the sides, above the
mask. The detector does not fire on it (rig share 0.09%). It is worth knowing for reviewers.

**Position check** (`position_check.json` / `position_report.html`, tracked): the median
cross-track to OSM centerlines is **1.54 m**, IQR 0.70 to 3.13 m. That is at the metric's
~1.75 m floor and better than Mapillary's SfM positions on the same metric in Laurens
(3.73 m).

Panoramax serves one position, so the check reports and never gates (`submitted_field:
null`, 0 flagged). 4,027 panos (14%) are more than 30 m from any OSM street of the queried
classes. The municipal capture also walks pedestrian ways and squares.

## 2. Does `MAPILLARY_ERRORS` fit Panoramax?

### Instruments

All of these are offline and need no ground truth. Each is reported at tier 0.55, with
0.30 alongside, at 2.6 m.

- **(a) Pose ablation.** `fuse_sites.py runs/bayonne --pose-ablation --min-confidence T`.
  It now also prints a same-site-set table: a group counts only if every sign convention
  places every posed member within the 25 m production cap. The table gives the median
  and p90, once over all posed members and once over real-tilt members only.
- **(b) Leave-one-out residual.** `reprojection_residual.py`, GT-free half. Bayonne has no
  RampNet split, so the GT half is skipped and nothing is faked.
- **(c) Normalized leave-one-out residual.** This was added to (b) as columns. For each
  held-out view, chi2 = rᵀ S⁻¹ r, where S = the view's own `cov_en` + Λ₋ᵢ⁻¹. It is pooled
  as chi2/(2n), with a median form beside it, and broken down by rig, pose group and range.
  `--fit-sigma-pitch TARGET` adds the `sigma_pitch` that brings pooled chi2/dof to TARGET
  with everything else fixed, and, exploratory, the `sigma_gps` that does.

Commands:

```
python scripts/fuse_sites.py runs/bayonne --pose-ablation --min-confidence 0.55
python scripts/reprojection_residual.py bayonne richmond clovis laurens annapolis morgantown \
    --camera-height-m 2.6 --refuse --fit-sigma-pitch 0.269 [--min-confidence 0.3] \
    --publish docs/figures/panoramax-bayonne/data/reprojection[_t0.3]
```

### Why the rule was amended before reading Bayonne

The comparators were computed first. On the five Mapillary runs, `MAPILLARY_ERRORS` reads
chi2/dof **0.18 to 0.43**, which is "too loose" by the absolute [0.5, 2.0] window, on the
very source the model was built for.

The 3 m `sigma_gps` dominates S. The leave-one-out residual cannot see position error that
the views of a site share: adjacent frames of one sequence share their error. With the GPS
term removed, Richmond reads 18.3 and Annapolis 8.4.

So the absolute window was mis-aimed. The chi2 clause was re-anchored on Richmond, as the
metre clause already was. **ADEQUATE** iff chi2/dof is in [0.13, 0.54] (0.5x to 2x
Richmond's 0.269) AND the median metre residual is in [1.30, 2.93] m (1.5x of Richmond's
1.95). **TOO TIGHT** iff above 0.54, and the recommended constant is the `sigma_pitch` that
brings it to 0.269. **TOO LOOSE** iff below 0.13.

A self-check of the instrument: fit on Richmond itself, at its own target, it returns
`sigma_pitch` 1.49° and `sigma_gps` 3.00 m, which are the constants it was fused with.

### Results (2.6 m; data in `figures/panoramax-bayonne/data/reprojection/`)

| run | tier | held-out views | sites | px p50 | m p50 | m p90 | abs along p50 | abs cross p50 | chi2/dof | chi2/dof (median form) | sigma_pitch to 0.269 | sigma_gps to 0.269 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **bayonne** | 0.55 | 1,039 | 281 | 31.4 | **3.66** | 6.76 | 2.25 | 1.79 | **0.580** | 0.582 | **> 15°** | **4.70 m** |
| **bayonne** | 0.30 | 3,940 | 1,017 | 29.7 | 3.70 | 6.75 | 2.37 | 1.83 | 0.584 | 0.595 | > 15° | 4.71 m |
| richmond | 0.55 | 7,235 | 945 | 11.6 | 1.95 | 5.29 | 1.46 | 0.66 | 0.269 | 0.181 | 1.49° | 3.00 m |
| clovis | 0.55 | 6,841 | 1,135 | 10.2 | 1.81 | 4.85 | 1.48 | 0.51 | 0.219 | 0.151 | 0.69° | 2.57 m |
| laurens | 0.55 | 585 | 109 | 18.1 | 2.77 | 5.73 | 2.04 | 1.16 | 0.431 | 0.381 | 5.56° | 3.97 m |
| annapolis | 0.55 | 24,228 | 2,727 | 10.6 | 1.69 | 5.14 | 1.38 | 0.49 | 0.216 | 0.146 | 0.84° | 2.50 m |
| morgantown | 0.55 | 9,084 | 1,171 | 9.6 | 1.38 | 4.41 | 1.09 | 0.44 | 0.177 | 0.098 | — | 2.21 m |

### Verdict

**TOO TIGHT** under the amended rule: Bayonne's chi2/dof is 0.580, above 0.54. The metre
clause fails independently (3.66 m is 1.88x Richmond).

The original absolute rule would have left Bayonne unclassified. Its chi2 clause passes
(0.58 is in [0.5, 2.0]) but its metre clause fails, and that rule had no verdict for the
combination. The 0.30 tier reads the same.

**The pre-registered remedy does not exist.** No `sigma_pitch` up to 15° brings chi2/dof
to 0.269. Pitch only widens the along-ray sigma, and that term grows with range, but
Bayonne's residual does not:

- metres p50 by range bin: 3.41 / 3.32 / 3.75 / 4.88 m for 0-8 / 8-12 / 12-18 / 18-25 m;
- the cross-ray part is almost flat: 1.66 / 1.71 / 1.95 / 1.95 m;
- in pixels it falls 56.6 / 32.6 / 24.1 / 17.5 px.

A residual that is roughly constant in metres and large near the camera is **camera
position error**, not tilt and not heading. Heading error would grow the cross-ray part
in proportion to range.

**Recommendation (exploratory, outside the pre-registered rule).** If Panoramax gets its
own model, widen the position term. A `sigma_gps` of about **4.7 m** (Richmond-calibrated,
every other `MAPILLARY_ERRORS` sigma unchanged) makes Bayonne read like Richmond.

This should not be adopted from this run alone, for three reasons:

- Bayonne was thinned at 10 m against Mapillary's 5 m. Sites have fewer views, and their
  views come from more passes (shared-sequence error cancels less), and both of those push
  chi2 up.
- The detections per pano are low and still unvalidated.
- The absolute position against OSM (1.54 m cross-track) is good, so the scatter is
  between captures, not a bias.

Ground truth decides it: `eval_sites.py bayonne`'s match-radius ablation once the RampNet
split exists.

### By pose group and rig (tier 0.55; `gtfree_breakdown.csv`)

| pose group | views | m p50 | chi2/dof |
|---|---:|---:|---:|
| absent | 632 | 3.31 | 0.51 |
| zeros (0/0 reported) | 188 | 3.80 | 0.60 |
| real tilt reported | 219 | 4.46 | 0.77 |

The panos that report a tilt are the worst-placed ones under a flat raycast. All held-out
views are GoPro Max, so the rig breakdown has one row. Capture-month spread inside a site
widens the residual: chi2/dof is 0.56 at 0 months, 0.61 at 1-18 and 0.73 at 19-36.

### Should `pers:pitch` / `pers:roll` be applied? No.

Pose coverage is 16,543 of 28,524 panos (58%). On the same site set at the 25 m cap,
pairwise member distance in metres:

| convention | posed members, median / p90 (0.55; 216 groups) | real-tilt members only, median / p90 (0.55; 97 groups) | real-tilt only (0.30; 249 groups) |
|---|---|---|---|
| **off (flat)** | **4.79 / 7.72** | **4.60 / 7.58** | **4.45 / 7.49** |
| +pitch +roll | 5.32 / 10.26 | 5.63 / 10.96 | 5.62 / 11.64 |
| +pitch −roll | 5.19 / 10.57 | 5.05 / 11.27 | 5.07 / 11.19 |
| −pitch +roll | 5.23 / 9.84 | 6.42 / 10.79 | 5.54 / 10.53 |
| −pitch −roll | 5.34 / 9.60 | 5.92 / 10.03 | 5.58 / 10.03 |
| +pitch, roll 0 | 5.04 / 9.70 | 4.75 / 9.45 | 4.92 / 10.06 |

Every sign convention **loosens** the spread on both median and p90, at both tiers. The
p90 loosens by 25-55%. The uncapped table that the ablation has always printed agrees
(off has the lowest mean and median).

So the reported Panoramax pose is not a usable raycast correction, whatever its sign. This
matches GSV and the withheld Mapillary road-relative default. Flat stays right for this
source, and `fuse_sites`' stderr warning for gravity/road on Panoramax panos stays.

## 3. RampNet hand-off

Bundle: **`D:/Git/labeler-wt/bayonne-bundle/`**. It sits outside every repo and is meant
to be copied to `../RampNet/benchmark/bayonne/` by a RampNet session.

```
python scripts/export_benchmark.py runs/bayonne/results.jsonl \
    --bundle D:/Git/labeler-wt/bayonne-bundle --sample 100 --empty-sample 25
```

- 125 panos, seed 0, 30 m minimum spacing, tier 0.55. Strata: `top` 5, `random` 95,
  `empty` 25.
- 147 detections >= 0.55 to judge (30 on `top`, 117 on `random`).
- All 125 are GoPro Max.
- The fetch got 125 of 125 native-resolution panos, 0 failed. The reconcile is
  **OK: archive matches records 1:1** (`index.csv` holds sha256 per pano).

**For reviewers.** The municipal panos carry a large white logo band ("Bayonne Pays
Basque") across the bottom of the equirect, from y ≈ 0.79 (dip ≈ 52°). Do not mark ramps
inside it: it covers the street within about 2 m of the camera, and the nadir mask already
drops that region from anything shipped. The white car roof and bonnet show at the sides
from about y = 0.6.

French kerb lowerings at crossings (*abaissés de trottoir*) often have no flare and no
tactile paving. Whether they count as ramps is a labelling-policy call for the RampNet
session. The low detection rate (section 1) makes it the call that matters most here.

## 4. Archive

The full native-resolution archive is on makelab2 per the archive protocol:
`/projects/makeabilitylab/sidewalk-auto-labeler/runs/bayonne/`. That directory holds
`results.jsonl`, `manifest.json` and `area.geojson` (copied and sha256-checked), plus
`panos/`, `index.csv` and `fetch.log`.

The archive runs detached in tmux session `bayonne_archive` via `bayonne_archive.sh`. When
it finishes, it writes `runs/bayonne_archive.done` with the rc and UTC time; a nonzero rc
means the archive is not yet 1:1.

At 05:56 UTC it was at 750 of 28,524 panos (1.84/s, about 4 h to go). Verify or resume
with the same command, which is resumable and reconciles on every pass:

```
cd /projects/makeabilitylab/sidewalk-auto-labeler && \
  /homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python \
  /homes/gws/jonf/sidewalk-auto-labeler-py312/scripts/export_benchmark.py \
  runs/bayonne/results.jsonl --out runs/bayonne/panos
```

The 104 cropped MAX2 panos are not in `results.jsonl`, so they are not archived.

## 5. What remains

- **Ground truth.** Review the bundle in RampNet, then `eval_sites.py bayonne` for world
  P/R and the match-radius ablation, which settles the position-error question above.
- **The detection rate.** Ground truth says whether 0.135 per pano is the city or the model.
- **`sigma_gps`.** Re-measure at 5 m thinning (a resume under a new run name), or on a
  second Panoramax city, before giving Panoramax its own `ErrorModel`.
