# Bayonne: the first full Panoramax run, and whether the borrowed error model fits (#57)

Issue [#57](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57), part 1.
Part 2 of the issue was PR #70.

Panoramax is the federated open imagery commons. `sources/panoramax.py` shipped in PR #49
and had only a 12-pano smoke test behind it. This document covers three things:

- the first full city run (Bayonne, France: OSM commune relation 166713, 25.8 km²);
- a ground-truth-free measurement of the fusion error model that `geo.error_model_for`
  borrows from Mapillary for this source;
- the hand-off of a benchmark bundle to RampNet for ground truth.

The reading rule was posted on #57 before any Bayonne residual was read
([pre-registration](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57#issuecomment-5921184460)).
It was amended once, also before any Bayonne residual was read
([amendment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57#issuecomment-5921233648)).
Two reviews of PR #125 then corrected how that amendment was explained and how its verdict
must be qualified
([correction](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57#issuecomment-5932787356)).

**Nothing in production changes here.** `error_model_for('panoramax')` still returns
`MAPILLARY_ERRORS`, and fusion's pose default is unchanged.

> **Key takeaways**
>
> 1. **TOO TIGHT under the amended rule, but marginal and confounded.** Bayonne's
>    chi²/dof is 0.580 [95% CI 0.539, 0.618] (median form 0.582 [0.522, 0.647]) against
>    a 0.538 bar.
>    - Mapillary GoPro Max populations read 0.18-0.50: Richmond's GoPro Max views 0.499
>      [0.467, 0.532] (median form 0.453), Laurens 0.431, Morgantown 0.177. Bayonne is
>      1.16x Richmond's GoPro Max views, and the CIs only just separate.
>    - Bayonne sites are also small: 3 views and 2 sequences per site at the median,
>      against Richmond's 7 and 4, because of the 10 m thinning.
>
>    So this cannot be separated from a single-consumer-rig effect or a site-size effect.
>    *Figure 1; `data/fig1_verdict.csv` `chi2_dof`, `chi2_dof_median`, `*_lo`/`*_hi`;
>    site make-up in `data/fig3_seqsplit.csv`.*
> 2. **The excess is across the ray, not along it, so it is not tilt.**
>    - Against all of Richmond, Bayonne's residual is a near-constant +1.2 to +2.2 m in
>      every range bin.
>    - Against Richmond's GoPro Max views, re-solved from GoPro Max views only
>      (rig-matched), there is no consistent along-ray excess (+0.5 / −0.2 / −0.5 /
>      +0.2 m by range bin). A **+0.6 to +1.0 m excess across the ray** remains:
>      position, with some heading error possible.
>    - The leave-one-out-calibrated `sigma_gps` is 2.16 m, against 1.11 m for Richmond
>      overall. Against the GoPro Max populations it is 1.81 m (Richmond's GoPro Max
>      views), 1.74 m (Laurens) and 0.80 m (Morgantown). All of them sit under
>      `MAPILLARY_ERRORS`' 3 m.
>    - Same confounds as takeaway 1: rig, and fewer views per site.
>
>    *Figure 2; `data/fig2a_range.csv` `m_p50`/`along_p50`/`cross_p50` by `subset`,
>    `data/fig2b_sigma_gps.csv` `sigma_gps_chi2_1_m`.*
> 3. **The amendment's mechanism is not established.** The amendment said shared
>    same-sequence error cancels in the leave-one-out. Richmond, the anchor, reads the
>    opposite way with separated CIs (0.404 with all-same-sequence mates vs 0.286 with
>    none). Four runs go the predicted way with separated CIs, and Laurens does too but
>    with overlapping CIs. *Figure 3; `data/fig3_seqsplit.csv` `chi2_dof`.*
> 4. **Do not apply `pers:pitch`/`pers:roll`.** On the same sites at the 25 m cap, every
>    sign convention loosens real-tilt members (p90 7.58 m flat vs 9.45-11.27 m at 0.55).
>    *Figure 4; `data/fig4_pose.csv` `median_m`/`p90_m`.*
> 5. **Bayonne's detection rate is inside the Mapillary range.** 0.135 per pano at 0.55,
>    against 0.123-1.048 for the five Mapillary runs. It is worth ground truth
>    (Figure 7d shows what the detector fires on), not anomalous.
>    *Figure 5; `data/fig5_detections.csv` `per_pano`.*

## 1. The run

| | |
|---|---|
| command | `main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --reuse-scan --thin-spacing 10` (lab A40, batch 1, default concurrency) |
| scan | 55 of 100 bbox z15 tiles within 50 m of the area; **73,161** in-area 360 pictures |
| thinning | 10 m: **28,634** panos (5 m would give 50,528) |
| processed / skipped / failed | **28,524** / 110 / **0** in one pass, so no drain rerun |
| wall time | 6.7 h; 1.18 panos/s (24,062 s in forward) |
| model | `rampnet-model@606a11956743` |

**Thinning decision.** The rule was stated before the scan: use the source default of
5 m (every Mapillary run's density) unless the scan-only estimate exceeded 16 h, else
10 m. The estimate printed 21.1 h for 50,528 panos at 1.5 s/pano, so the run used 10 m.

The A40 then ran at 0.85 s/pano, so 5 m would have taken about 12 h; the estimate's
constant is an RTX 3070 figure. `manifest.json` recorded neither at the time;
[#126](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/126) added
`thin_spacing_m` and `panos_before_thinning` to every run entry (the Bayonne manifest predates
it; its values are 10 m and 73,161).

**Skips (110, all cached; `data/skips.csv`).**

- **104** are GoPro **MAX2** uploads on the OSM-FR instance. Each declares 7680x3840 but
  serves a 7680x2940, vertically cropped `hd` image. The 2:1 check rejects them, which is
  correct ([Figure 7c](figures/panoramax-bayonne/fig7c_max2_skip.jpg)).
- **3** have no `view:azimuth`, which is also correct.
- **3** IGN panos (`139d37d3-…`, `1f8be062-…`, `2c902fc4-…`) have metadata and pixels
  that are fine today. main.py does not log per-pano skip reasons, and two causes fit:
  - a transient asset 404 (the source caches a 404 as permanent);
  - a body that arrived incomplete and failed to decode. `_download_image`'s bare
    `except Exception` caches any decode exception as permanent.

  [#127](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127) tracks
  this. Its
  [amendment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127#issuecomment-5934720793)
  notes that a narrowed catch alone would not cover truncation.

No HTTP 429 or 5xx failures occurred on either instance.

**Census** ([Figure 6](figures/panoramax-bayonne/fig6_census.png); `data/census/`):

- **Rig:** GoPro Max 99.9% (5760x2880 for 58.0%, 5376x2688 for 41.9%).
- **Capture year:** 2026 56.0%, 2025 13.0%, 2024 28.0%, earlier 2.9%.
- **Producer:** `sig_bayonne` (the municipality, IGN instance, Etalab 2.0) holds 95.3%.
- **Pose:** absent 42.0%, real tilt 36.8%, reported as exactly 0/0 21.2%.
- **Detections:** at tier 0.55, 3,861 on 2,947 panos (**0.135 per pano**). At 0.30,
  10,011 (0.351 per pano).

The six runs at 0.55 ([Figure 5](figures/panoramax-bayonne/fig5_detections.png)): Clovis
0.123, **Bayonne 0.135**, Laurens 0.158, Morgantown 0.203, Annapolis 0.548, Richmond 1.048.
Bayonne is inside the Mapillary range.

[Figure 7d](figures/panoramax-bayonne/fig7d_crops.jpg) zooms in on 12 bundle detections.
They are lowered kerbs at crossings, and one tactile strip. Whether French kerb lowerings
are under-detected is still a question for ground truth. Four of the five comparators
predate the storage floor, so their 0.30 tier is their 0.55 tier.

**The nadir logo band.** The municipal panos carry a solid white band from y = 0.791
(dip 52.4°), measured on one 5376x2688 and one 5760x2880 pano. That is entirely inside
the rig mask (`NADIR_MASK_DEG` = 49°, y ≥ 0.772).

- Detections in the band: **2 of 10,011** at 0.30 and **0 of 3,861** at 0.55.
- Detections on the rig (dip ≥ 49°): 9 at 0.30, 1 at 0.55.
- [Figure 7a](figures/panoramax-bayonne/fig7a_contact_sheet.jpg) shows the band and the
  mask line. One random pano carries a different, green band from another producer.

**Position check** (`runs/bayonne/position_check.json`): median cross-track to OSM
centerlines **1.54 m**, IQR 0.70-3.13 m. That is at the metric's ~1.75 m floor. Panoramax
serves one position per pano, so the check reports and never gates.

4,027 panos (14%) sit more than 30 m from any street of the queried classes (orange in
Figure 6D). What they are (paths, squares, parks) was not verified.

## 2. Does `MAPILLARY_ERRORS` fit Panoramax?

### Instruments

All of these are offline and need no ground truth. Each runs at tier 0.55, with 0.30
alongside, at a 2.6 m camera height.

- **(a) Pose ablation.** `fuse_sites.py --pose-ablation` with a same-site-set table: a
  group counts only if every sign convention places every posed member within the 25 m
  cap.
- **(b) Leave-one-out residual.** `reprojection_residual.py`, GT-free half.
- **(c) Normalized form, added to (b).** For each held-out view, chi2 = rᵀ S⁻¹ r with
  S = the view's own `cov_en` + Λ₋ᵢ⁻¹. It is pooled as chi²/(2n), with a median form
  beside it.
  - Breakdowns: by rig, pose group, range, and **site-mates' capture sequence**
    (`seq_mates`: all from the held-out view's sequence / some / none).
  - `--fit-sigma-pitch TARGET` adds the `sigma_pitch` and the `sigma_gps` that bring
    pooled chi²/dof to TARGET with everything else fixed.
  - Two synthetic recovery tests validate this fit. `tests/test_reprojection_residual.py`
    draws residuals at a known pitch sigma and a known extra position scatter, and the fit
    returns them.
  - Re-fitting Richmond at its own level returns the constants it was fused with (1.49°,
    3.00 m). That is a **round trip by construction**: it shows only that the covariance
    rebuild is consistent, not that the fit is right.

**CIs.** Every CI here is a 95% percentile bootstrap that resamples **sites** (1,000
draws, seed 57). There are three exceptions:

- the `sigma_gps` calibration uses 200 site resamples;
- Figure 4 resamples pose groups;
- Figure 5 uses a normal approximation over panos.

### The rule, and the reviews' correction to how the amendment was explained

The comparators were computed before Bayonne was read. `MAPILLARY_ERRORS` reads chi²/dof
0.18-0.43 on the five Mapillary runs, below the absolute [0.5, 2.0] window, so the chi²
clause was re-anchored on Richmond:

| reading | condition |
|---|---|
| **ADEQUATE** | chi²/dof in [0.13, 0.54] AND median metre residual in [1.30, 2.93] m |
| **TOO TIGHT** | chi²/dof > 0.54; recommend the `sigma_pitch` that brings it to 0.269 |
| **TOO LOOSE** | chi²/dof < 0.13 |

**The reason the amendment gave is not established.** It said that position error shared
by a site's views from one sequence cancels in the leave-one-out, which would leave the
3 m GPS term unseen.

Splitting held-out views by their site-mates' sequence
([Figure 3](figures/panoramax-bayonne/fig3_seqsplit.png); `data/fig3_seqsplit.csv`)
gives mixed evidence:

| run | all mates same-sequence: chi²/dof (n) | no mate same-sequence: chi²/dof (n) | reading |
|---|---:|---:|---|
| Richmond | 0.404 (299) | 0.286 (2,128) | **opposite**, CIs separate |
| Bayonne | 0.423 (189) | 0.715 (330) | predicted, CIs separate |
| Clovis | 0.205 (1,438) | 0.355 (538) | predicted, CIs separate |
| Laurens | 0.338 (25) | 0.414 (87) | predicted, **CIs overlap heavily** |
| Annapolis | 0.194 (1,900) | 0.240 (1,967) | predicted, CIs separate |
| Morgantown | 0.146 (1,237) | 0.217 (369) | predicted, CIs separate |

In two runs the groups also differ in site size, so the comparison is not a clean test of
cancellation either way (`views_per_site_p50` in the CSV):

- Richmond: median 4 views per site for all-same-sequence views against 8 for none.
- Annapolis: 5 against 8.
- Bayonne, Clovis, Laurens and Morgantown: the two groups match within one view.

The share of same-site view pairs that come from one sequence (`data/fig3_pair_share.csv`)
is 0.19 in Richmond, 0.36 in Annapolis, 0.37 in Bayonne, 0.40 in Laurens, 0.47 in
Morgantown and 0.49 in Clovis. It is below half in every run (0.19-0.49), but well
above Richmond's 0.19 elsewhere. The Richmond and Annapolis shares were measured before
the amendment and not disclosed then.

So the rule stands, but the stated reason for it is replaced by measured numbers. The
`sigma_gps` that brings chi²/dof to **1** (`data/fig2b_sigma_gps.csv`) is:

- **1.11 m** for Richmond (CI 1.05-1.16) and **2.16 m** for Bayonne (CI 2.06-2.26);
- 0.74-1.81 m across the other populations, including the GoPro Max ones: Richmond's
  GoPro Max views 1.81 m, Laurens 1.74 m, Morgantown 0.80 m.

`MAPILLARY_ERRORS`' 3 m is about 2.7x Richmond's between-view scatter, and it is still
looser than Bayonne's. The constant reads as standing for absolute position error, which
a leave-one-out cannot see, more than for between-view scatter.

### Results ([Figure 1](figures/panoramax-bayonne/fig1_verdict.png); `data/fig1_verdict.csv`, `data/reprojection/`)

| run (views) | held-out views | sites | chi²/dof [95% CI] | median form | m p50 [95% CI] |
|---|---:|---:|---|---:|---|
| **Bayonne (GoPro Max)** | 1,039 | 281 | **0.580** [0.539, 0.618] | 0.582 [0.522, 0.647] | **3.66** [3.43, 3.85] |
| Richmond, all views | 7,235 | 945 | 0.269 [0.257, 0.282] | 0.181 | 1.95 [1.89, 2.02] |
| Richmond, GoPro Max views | 1,301 | 472 | **0.499** [0.467, 0.532] | **0.453** [0.417, 0.503] | 3.13 [2.96, 3.31] |
| Richmond, iSTAR Pulsar views | 5,230 | 813 | 0.194 | 0.136 | 1.69 |
| Laurens (GoPro Max) | 585 | 109 | 0.431 [0.387, 0.474] | 0.381 | 2.77 |
| Morgantown (GoPro Max) | 9,084 | 1,171 | 0.177 [0.167, 0.187] | 0.098 | 1.38 |
| Clovis (GoPro Fusion) | 6,841 | 1,135 | 0.219 | 0.151 | 1.81 |
| Annapolis (Trimble MX7) | 24,228 | 2,727 | 0.216 | 0.146 | 1.69 |

(CIs omitted from the table are in the CSV.) At tier 0.30 Bayonne reads 0.584, m p50
3.70 (`data/reprojection_t0.3/`).

### Verdict

**TOO TIGHT under the amended rule, but marginal: 0.580 against the 0.538 bar (8% over),
with the CI's lower end at 0.539.** It is confounded twice over.

- **Rig.** Richmond's anchor is 72% iSTAR Pulsar views. Matched to rig, Bayonne is 1.16x
  Richmond's GoPro Max views (0.499; median form 0.582 vs 0.453). Those CIs just separate,
  and Bayonne is the highest point in Figure 1. Mapillary GoPro Max populations span
  0.18-0.50 (Morgantown 0.177, Laurens 0.431, Richmond GoPro Max 0.499), so the rig alone
  does not set the level.
- **Site size.** Bayonne sites have a median of 3 views and 2 sequences, against
  Richmond's 7 and 4, because of the 10 m thinning.

This cannot be separated from a single-consumer-rig effect or a site-size effect. The
metre clause fails (3.66 m against the 2.93 m bar), but Richmond's own GoPro Max views
also sit above that bar (3.13 m).

The original absolute rule would have left Bayonne unclassified: its chi² clause passes
and its metre clause fails.

**What the excess is.** The pre-registered remedy, a wider `sigma_pitch`, does not reach
0.269 at any value up to 15° (`data/reprojection/gtfree_summary.csv`). The evidence is in
[Figure 2](figures/panoramax-bayonne/fig2_position.png) and `data/fig2a_range.csv`:

- **The metre excess is roughly constant across range bins.** Against all of Richmond it
  is +2.2 / +1.6 / +1.2 / +1.8 m (0-8 / 8-12 / 12-18 / 18-25 m). A pitch error would
  grow with range.
- **Rig-matched, the excess sits across the ray.** The reference is Richmond's GoPro
  Max views re-solved from GoPro Max views only (`_rig_sites`), so each is held out
  against GoPro Max mates, as Bayonne's are. Against it, the along-ray difference is
  +0.5 / −0.2 / −0.5 / +0.2 m, while the cross-ray excess is +0.6 / +0.8 / +1.0 / +0.7 m.
  - Bins hold 128-346 views.
  - An earlier version only filtered the mixed-rig solution, so those views were held
    out against mostly iSTAR Pulsar mates. It read along +0.4 / −0.3 / −0.9 / −0.1 and
    cross +0.8 / +0.8 / +1.0 / +0.9.
  - Re-solving narrows the cross-ray gap at the two outer bins and removes the along-ray
    deficit at long range. The reading is unchanged.
  - A range-independent cross-ray offset is camera position.
  - Heading error would grow with range: the cross-ray part grows only 1.66 to 1.95 m over
    a ~16 m span, which bounds heading at about 1° or less.
- **Camera height is not unusual.** The range-scale fit gives k 1.087, inside the
  Mapillary range (Richmond 1.055, Clovis 1.091, Morgantown 1.114, Annapolis 1.132;
  `range_slope.csv`).
- Chi²/dof is also flat by range (0.60 / 0.57 / 0.60 / 0.56). This only says that the
  model's predicted variance tracks the residual's range dependence equally in every bin.
  The constant metre excess above is the evidence.

**Recommendation (exploratory, outside the pre-registered rule).** Bayonne's
between-view position scatter is about 2x Richmond's (2.2 vs 1.1 m, calibrated to
chi²/dof = 1). It is 1.2x the scatter of Richmond's GoPro Max views (1.81 m) and of
Laurens (1.74 m).

The earlier 4.7 m figure (the `sigma_gps` that brings Bayonne to Richmond's 0.269)
follows only if Richmond's ~2.7x inflation is kept on purpose, with 3 m standing for
absolute error. Used for association gating, it would loosen gates past anything
measured. Nothing is adopted. Ground truth (`eval_sites.py bayonne`'s match-radius
ablation) and a 5 m run decide it.

**By pose group** (0.55, `data/reprojection/gtfree_breakdown.csv`): chi²/dof is 0.51 for
pose absent (632 views), 0.60 for 0/0 (188) and 0.77 for real tilt (219). Panos that
report a tilt are the worst placed under a flat raycast.

### Should `pers:pitch` / `pers:roll` be applied? No ([Figure 4](figures/panoramax-bayonne/fig4_pose.png); `data/fig4_pose.csv`)

On the same site set at the 25 m cap, real-tilt members only:

| convention | 0.55: median / p90 (m) | 0.30: median / p90 (m) |
|---|---|---|
| **off (flat)** | **4.59 / 7.58** | **4.44 / 7.49** |
| +pitch +roll | 5.63 / 10.96 | 5.61 / 11.64 |
| +pitch −roll | 5.05 / 11.27 | 5.05 / 11.19 |
| −pitch +roll | 6.42 / 10.79 | 5.52 / 10.53 |
| −pitch −roll | 5.92 / 10.03 | 5.56 / 10.02 |
| +pitch, roll 0 | 4.75 / 9.45 | 4.92 / 10.06 |

These are 97 of 176 groups at 0.55 and 249 of 523 at 0.30; groups a convention cannot
place inside the cap are excluded. The text output is in
`data/pose_ablation_t{0.55,0.3}.txt`.

Every convention loosens the spread, and on p90 the CIs do not overlap the flat arm.
Flat stays right for this source, as for GSV and the withheld Mapillary road-relative
default.

A hypothesis, not tested here: the GoPro Max levels the horizon in-camera, so the images
may already be gravity-rectified. Applying `pers:pitch`/`pers:roll` on top would then be a
double correction, which would explain why every sign hurts.

[Figure 7b](figures/panoramax-bayonne/fig7b_site_plan.png) shows ground-point scatter for
one site, with position, heading and range error mixed. It plots site 183 (5 views from 4
sequences), chosen by a fixed rule: ≥ 5 views, with a median leave-one-out closest to the
run's.

## 3. RampNet hand-off

Bundle: **`D:/Git/labeler-wt/bayonne-bundle/`** (outside every repo). A RampNet session
copies it to `../RampNet/benchmark/bayonne/`.

- 125 panos (seed 0, 30 m minimum spacing, tier 0.55): `top` 5, `random` 95, `empty` 25.
- 147 detections ≥ 0.55 to judge.
- Native-resolution fetch 125/125, reconcile **OK 1:1**.
- The bundle's records carry only ≥ 0.55 detections.

**For reviewers.** Do not mark ramps inside the white logo band (y ≳ 0.79), which covers
the street within about 2 m of the camera. The car roof shows from about y = 0.6 at the
sides. Whether French lowered kerbs, often without a flare or tactile paving, count as
ramps is a labelling-policy call; Figure 7d shows typical examples.

**Attribution.** The municipal imagery is © `sig_bayonne`, published through Panoramax
(panoramax.ign.fr) under the Licence Ouverte / Etalab 2.0. Other producers are credited
per pano (`copyright`, `license` in every record). Figure 6's streets are © OpenStreetMap
contributors (ODbL).

## 4. Archive

The archive is complete on makelab2 under
`/projects/makeabilitylab/sidewalk-auto-labeler/runs/bayonne/`:

- `results.jsonl`, `manifest.json` and `area.geojson` (sha256-checked);
- `panos/` with all **28,524** native-resolution panos;
- `index.csv` and `fetch.log`.

`runs/bayonne_archive.done` reads `rc=0 2026-10-01T10:01:18Z`. The archive took 4 h 10 m
at 1.90 panos/s.

## 5. Replication

**Inputs** (`data/inputs.csv`, written by the `data` step):

| file | sha256 |
|---|---|
| `runs/bayonne/results.jsonl` | `4f38ff528d4445b21eb02e99427e0ba3c88dfb5c940cc4702a283182e379b4b3` |
| `runs/richmond/results.jsonl` | `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c` |
| `runs/clovis/results.jsonl` | `f6a896f19f4c7036186b201bbd2a1bfa4d6a20f34c14f5586a72da936475c15d` |
| `runs/laurens/results.jsonl` | `16c5a348b739274bf8e7623b63551da5245547d0ca4e5a52956fd2413b4fe213` |
| `runs/annapolis/results.jsonl` | `f1b228f742dc95909067cc3ba9b045f1acf76617747b6e21eb9e7df58335b9dd` |
| `runs/morgantown/results.jsonl` | `7dbf24e03574c31b0a3011490177a5d51dc20e30badf7fdd35da443e870b98bc` |
| `runs/bayonne/osm_streets.json` (Overpass, cached by step 2) | `60d212b73d4cd7a7757b16f3f90e189d829958bbcc025f6d774ae165a11069b7` |

**Environment**

| | |
|---|---|
| model | `projectsidewalk/rampnet-model` revision `606a11956743f7eb328d9207769034752f6191f4` |
| streetlevel | 0.12.10 (unused by this source) |
| detection | makelab2: Python 3.12.13, torch 2.14.0+cu130 (CUDA 13.0), Pillow 12.3.0, NVIDIA A40 (driver 595.80), repo at `632e819` |
| analysis | Windows: Python 3.12.13, matplotlib 3.11.2, Pillow 12.3.0, shapely |

**Steps, in order.** "Net" means it needs the network.

| # | command | needs | time |
|---|---|---|---|
| 1 | `python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --scan-only` | net | 6 s |
| 2 | `python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --reuse-scan --thin-spacing 10` (ends with the position check; Overpass) | GPU + net | 6.7 h |
| 3 | `python scripts/position_check.py runs/bayonne --report` (only to redo step 2's check) | net (cached after) | < 1 min |
| 4 | `python scripts/fuse_sites.py runs/bayonne --pose-ablation --min-confidence 0.55 > docs/figures/panoramax-bayonne/data/pose_ablation_t0.55.txt` (and `0.3` → `…_t0.3.txt`) | — | 1 s |
| 5 | `python scripts/reprojection_residual.py bayonne richmond clovis laurens annapolis morgantown --camera-height-m 2.6 --refuse --fit-sigma-pitch 0.269 --benchmark-root /nonexistent [--min-confidence 0.3] --publish docs/figures/panoramax-bayonne/data/reprojection[_t0.3]` | — | ~1.5 min per tier |
| 6 | `python scripts/run_census.py runs/bayonne --out docs/figures/panoramax-bayonne/data/census --band-y 0.79` | — | 10 s |
| 7 | `python scripts/panoramax_bayonne_figures.py data` → `data/inputs.csv`, `data/fig*.csv` (bootstraps, seed 57; `--only` regenerates parts) | — | ~45 min (the `sigma_gps` bootstrap) |
| 8 | `python scripts/panoramax_bayonne_figures.py skips` → `data/skips.csv` | net | 1 min |
| 9 | `python scripts/panoramax_bayonne_figures.py examples` → `data/examples/` (7a thumbnails; 7c, the first cropped-MAX2 skip by pano id) | net | 1 min |
| 10 | `python scripts/export_benchmark.py runs/bayonne/results.jsonl --bundle D:/Git/labeler-wt/bayonne-bundle --sample 100 --empty-sample 25` | net | 1.5 min |
| 11 | `python scripts/panoramax_bayonne_figures.py crops` → `data/examples/crops/` + `crops.csv` (from the bundle) | — | 10 s |
| 12 | `python scripts/panoramax_bayonne_figures.py figures` → `fig*.png` / `.jpg` (+ `.svg` for figures 1-5 and 7b) | — | 30 s |
| 13 | on makelab2, `bayonne_archive.sh`: `export_benchmark.py runs/bayonne/results.jsonl --out runs/bayonne/panos` | net | 4.2 h |

- **Step 2** must keep `--thin-spacing 10`. The manifest predates #126's `thin_spacing_m`; a
  resume at any other spacing is now refused once the manifest is bound.
- **Step 5's** `bayonne_report.md` copies under `data/reprojection*/` are that step's
  per-city `report.md`, copied by hand.
- **Step 12** reads only the committed `data/`. It is byte-reproducible for the listed
  versions: it was run twice and every PNG, JPEG and SVG hashed identical. SVGs are written
  with LF line endings, and `.gitattributes` keeps them LF.
- **Figure formats.** Figures 1-5 and 7b are PNG + SVG. The map (6) is PNG with a
  deterministic 256-colour palette. The photo figures (7a, 7c, 7d) are quality-92 JPEGs
  (4:4:4) rendered from the same 200 dpi raster: a palette posterised the photos, and an
  SVG would embed them a second time.

**Committed vs regenerated.**

- **Committed:**
  - `runs/bayonne/{manifest.json, area.geojson, position_check.json, position_report.html}`;
  - everything under `docs/figures/panoramax-bayonne/`: figures, `data/*.csv`, the
    `data/examples/` thumbnails and crops, and the report copies.
- **Regenerated, not committed:**
  - `results.jsonl`, `scan.json`, `osm_streets.json` and the logs (local
    `runs/bayonne/`, makelab2 and the archive root);
  - each city's `views.csv` (step 5 with `--out-root`);
  - the bundle.

### Where each number lives

| number | file | column / row |
|---|---|---|
| 73,161 / 28,634 / 28,524 / 110 / 0 | `runs/bayonne/manifest.json` (`runs[-1]`) and `run.log` | `panos_found_in_area`, `processed`, `skipped`, `failed`; 73,161 from `scan.log` |
| 6.7 h, 1.18 panos/s, 24,062 s | `run.log` | last `detector:` line; `started_at`/`finished_at` in the manifest |
| 21.1 h estimate, 50,528 | `scan.log` | — |
| 104 / 3 / 3 skips | `data/skips.csv` | `reason` |
| rig / year / pose / producer shares | `data/census/{rigs,years,pose,producers}.csv` | `share` |
| 0.135 / 0.351 per pano; 2 of 10,011, 0 of 3,861 in band; 9 / 1 on rig | `data/census/detections.csv` | `detections_per_pano`, `in_band`, `on_rig` |
| detections per pano, six runs | `data/fig5_detections.csv` | `per_pano` by `run`, `tier` |
| 1.54 m, IQR 0.70-3.13, 4,027 | `runs/bayonne/position_check.json` | `fields.submitted.cross_track`, `panos_not_near_a_street` |
| chi²/dof, median form, m p50 (+ CIs), incl. rig-matched rows | `data/fig1_verdict.csv` | `chi2_dof`, `chi2_dof_median`, `m_p50`, `*_lo`/`*_hi` by `run`, `subset` |
| 0.584 / 3.70 at 0.30 | `data/reprojection_t0.3/gtfree_summary.csv` | `bayonne` row |
| `sigma_pitch` > 15°, `sigma_gps` 4.70 m at target 0.269 | `data/reprojection/gtfree_summary.csv` | `sigma_pitch_deg_at_chi2_target`, `sigma_gps_m_at_chi2_target` |
| `sigma_gps` at chi²/dof 1 (2.16, 1.11, 1.81, 1.74, 0.80, …) | `data/fig2b_sigma_gps.csv` | `sigma_gps_chi2_1_m`, `lo`, `hi` by `run`, `subset` |
| by-range excess (total, along, cross), chi²/dof by range | `data/fig2a_range.csv` | `m_p50`, `along_p50`, `cross_p50`, `chi2_dof` by `run`, `subset`, `range_bin` |
| k 1.087 vs 1.055 / 1.091 / 1.114 / 1.132 | `data/reprojection/range_slope.csv` | `implied_range_scale_k`, `capture_year = all` |
| same-sequence split; views / sequences per site; site size per group | `data/fig3_seqsplit.csv` | `chi2_dof`, `chi2_dof_lo`/`_hi`, `n_views`, `views_per_site_p50`; `site_makeup` rows |
| same-sequence pair shares 0.19 … 0.49 | `data/fig3_pair_share.csv` | `same_sequence_share` |
| pose ablation medians / p90s | `data/fig4_pose.csv` (and `pose_ablation_t*.txt`) | `median_m`, `p90_m` by `tier`, `subset`, `convention` |
| chi²/dof by pose group | `data/reprojection/gtfree_breakdown.csv` | `dimension = pose_group`, `city = bayonne` |
| logo band y 0.791 / 52.4° | measured on two panos; drawn in Figure 7a | `LOGO_BAND_Y` in `scripts/panoramax_bayonne_figures.py` |
| Figure 7d crops, ranges, confidences | `data/examples/crops.csv` | `confidence`, `range_m`, `privacy_skip` |
| bundle strata, 147 detections | `D:/Git/labeler-wt/bayonne-bundle/sample.json`, `records.jsonl` | `groups` |
| archive 28,524, rc 0 | makelab2 `runs/bayonne/index.csv`, `runs/bayonne_archive.done` | — |

## Figures

| # | question it answers | file |
|---|---|---|
| 1 | Is Bayonne outside the Richmond-anchored band? Marginally: 8% over the bar, 1.16x Richmond's GoPro Max views (CIs just separate). | [fig1_verdict](figures/panoramax-bayonne/fig1_verdict.png) |
| 2 | Tilt or position? Rig-matched (re-solved), the excess is 0.6-1.0 m across the ray. Scatter is 2x Richmond's and 1.2x its and Laurens' GoPro Max views. | [fig2_position](figures/panoramax-bayonne/fig2_position.png) |
| 3 | Does same-sequence error cancel? Not established: the anchor reads the opposite way. | [fig3_seqsplit](figures/panoramax-bayonne/fig3_seqsplit.png) |
| 4 | Apply the reported pose? No: every sign convention loosens. | [fig4_pose](figures/panoramax-bayonne/fig4_pose.png) |
| 5 | Is the detection rate anomalous? No: it is inside the Mapillary range. | [fig5_detections](figures/panoramax-bayonne/fig5_detections.png) |
| 6 | What did the run cover? Census and map. | [fig6_census](figures/panoramax-bayonne/fig6_census.png) |
| 7a-d | What do the panos, a site, a MAX2 skip and the detections themselves look like? | [7a](figures/panoramax-bayonne/fig7a_contact_sheet.jpg), [7b](figures/panoramax-bayonne/fig7b_site_plan.png), [7c](figures/panoramax-bayonne/fig7c_max2_skip.jpg), [7d](figures/panoramax-bayonne/fig7d_crops.jpg) |

**Colour key across figures:** blue is Bayonne, orange is a GoPro Max population, and
grey is other or mixed rigs.

**Credits:**
- Figures 7a, 7c and 7d: imagery via Panoramax, with producer and licence per panel
  (municipal: © sig_bayonne, Licence Ouverte / Etalab 2.0; 7c: © lplm, CC BY-SA 4.0).
- Figure 6: © OpenStreetMap contributors (ODbL).

Alt text, in order:

1. Dot plot of chi²/dof and median residual for eight rows. Bayonne at 0.58 sits just
   above the shaded 0.13-0.54 band. Richmond's GoPro Max views (0.50) and Laurens (0.43)
   sit just inside it, Morgantown (GoPro Max) at 0.18, and other rigs near 0.2.
2. Three line panels by range: Bayonne, all of Richmond, and Richmond's GoPro Max views
   re-solved on their own. Bayonne tracks Richmond's GoPro Max along the ray but sits
   0.6-1.0 m above it across the ray. A dot plot shows calibrated `sigma_gps` from 0.74 m (Annapolis) to 2.16 m
   (Bayonne), with the GoPro Max populations at 0.80, 1.74 and 1.81 m, all left of the
   3 m line.
3. Paired dots per run: chi²/dof for held-out views whose site-mates are all from the
   same sequence vs none. Four runs are lower with same-sequence mates, Laurens too but
   with overlapping CIs, and Richmond is higher.
4. Median and p90 within-site distance for six pose conventions at two tiers. The flat
   arm is lowest everywhere.
5. Detections per pano for six runs. Bayonne (0.135) sits between Clovis (0.123) and
   Laurens (0.158).
6. Bar charts of capture year, rig and pose availability, and a map of 28,524 panos
   coloured by distance to the nearest OSM street, with a 1 km scale bar.
7. Four example figures:
   - (a) Eight panoramas with detections, the nadir-mask line and the shaded logo band.
   - (b) Plan view of five cameras, rays and ground points around a fused site.
   - (c) A MAX2 image that is shorter than its declared 2:1 frame.
   - (d) Twelve zoomed crops of detections, mostly lowered kerbs at pedestrian crossings.

## 6. What remains

- **Ground truth.** Review the bundle in RampNet, then run `eval_sites.py bayonne`.
- **Position scatter.** Re-measure at 5 m thinning, or in a second Panoramax city, before
  Panoramax gets its own `ErrorModel`.
- **Follow-ups:** [#126](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/126)
  (record and bind `--thin-spacing`) and
  [#127](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127) (the
  decode catch and truncation).
