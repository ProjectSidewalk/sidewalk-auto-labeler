# Leave-one-view-out reprojection residual (issue #36)

Measured 2026-09-23 with `scripts/reprojection_residual.py` on every local run that has
multi-view sites: four GSV cities (bend, paterson, gainesville, sao_paulo), the GSV arm of
Laurens (laurens_gsv) and five Mapillary cities (richmond, clovis, laurens, annapolis,
morgantown). Ground truth is the RampNet benchmark (`../RampNet/benchmark/<split>/`, read as
data at RampNet `8a41d17`). Every table below is generated from the CSVs committed in
[`docs/figures/reprojection-residual/data/`](figures/reprojection-residual/data/).

Revised 2026-09-24 after review. Three claims in the first version went further than the
data: that about half of RampNet#101's 11% slope is an estimator artefact (it compared two
different estimators), the placement figure in metres (most of its rows are truncated), and
the new rig's `k` (a lower bound, not a value). Each is corrected below. **Every number in
this document describes sites fused at the benchmark tier 0.55**, the tier the bundles were
judged at, not the operational tier 0.30 that production ships.

## Summary

- **The GT-free residual is small and uniform on GSV.** Hold out one view of a site with
  three or more operational views, re-solve the site from the others, and project it into
  the held-out pano. The median distance from the view's own detection is **5.5–9.8 heatmap
  px** on GSV and **9.6–18.1 px** on Mapillary (1 heatmap px = 0.35°). On the ground that is
  **1.1–1.9 m** (GSV) and **1.4–2.8 m** (Mapillary). Denominator: held-out views, 322 to
  43,288 per city.
- **Range scale, pooled per city, at 2.6 m.** The residual is exactly `s·g` under a
  range-scale error. That gives an implied range scale `k` of **1.02–1.10 on GSV** and
  **0.93–1.13 on Mapillary**. After subtracting the estimator's own null
  (errors-in-variables, simulated under the error model), GSV is 1.01–1.09.
- **Per-pano heights flatten the rig-to-rig scale gap. They do not flatten the scale
  itself.** At 2.6 m, the 2025–26 GSV rig reads **k = 1.19 (paterson 2025) and 1.18
  (gainesville 2026)**. The 2019–2024 rigs in the same cities read 1.04–1.07. That is a
  gap of 10–15 points. The 1.18–1.19 is a **lower bound**: it is measured on sites
  associated at 2.6 m, which pulls `s` toward 1 (a view that disagrees more is less likely
  to be a member). The camera-height study's fixed point for the same rig (1.98 m and
  1.92 m) implies **k ≈ 1.31–1.35**. It also rests on two city-years, and capture year is
  not the rig: bend 2025 reads 1.045 and gainesville 2025 reads 1.053. Under per-pano heights, every GSV vintage from 2019 on (2018 in
  paterson) reads **k = 0.94–1.00**, and the new rig is 0.987–0.988. So the gap closes. But the pooled
  scale moves to **0.96–0.99** in all four cities (0.95–0.98 null-corrected). Ranges now
  run 1–4% short, which is the same direction as the camera-height study's "depth heights
  run short" (see *What it cannot see*).
- **Part of RampNet#101's slope is an estimator artefact, and how much depends on the
  city.** #101's estimator regresses each member's along-ray residual from the *full* site
  on range, with site fixed effects, on sites whose ranges span ≥ 4 m. Run here on the
  on-disk `sites.jsonl` it reproduces #101's table exactly (e.g. paterson 4,433 sites,
  19,588 members, 0.1159). On simulated data with *no* scale error, the same estimator
  returns **0.040–0.050** in bend, paterson and gainesville, against observed 0.108–0.116
  (**37–45%** of it), and 0.044 of 0.058 in sao_paulo (**76%**). The leave-one-out naive
  slope this tool also reports has a null share of **39–71%** by city. Both nulls are only
  as good as the error model, which is miscalibrated (below). On Mapillary the null exceeds
  the observed slope, so no correction is possible there. The geometry-corrected `s` has a
  null of only ~0.01.
- **`k` is an effective range scale, not a camera-height error alone.** A constant
  vertical offset of the detector's peak produces a range error that grows about as r²,
  and the `s·g` fit absorbs part of it as scale. The GT shows such an offset: the
  reviewer's box centre sits a median **−1.2 px** above the model's peak (negative in all
  four boxed splits). Adding an r² term to the pooled fit moves `k` a lot: at 2.6 m, GSV
  `k` goes from 1.02–1.10 to **0.93–1.06**, with an implied constant peak offset of
  −1.0 to −2.8 px. The two terms trade off strongly, so neither fit isolates the camera
  height. Read every `k` here as scale plus detector placement.
- **GT-anchored placement, in pixels.** We project the site, *without* the judged pano's
  own view, into that pano and compare against the reviewer's box centre or missed-ramp
  click. The pixel residual never raycasts the reference. Pooled over ten splits at 2.6 m:
  **p50 11.8 px / p90 40.4 px** (n = 866 references). Of the 11.8 px, **~4.7 px** is the
  box centre's own median offset from the model's peak, a floor no position estimate can
  remove.
- **In metres, only the untruncated subset.** 443 of the 866 references are missed marks,
  matched to a site only if it lies within 5 m, which truncates their tail. The other 423
  are box centres on a detection of a member pano, which no distance gate touches. On
  those: **p50 1.86 m / p90 5.22 m** (px p50 11.2 / p90 36.9). These metres raycast the
  reference pixel under the pano's height model, so they are not independent of that
  model in the along-ray direction.

## Method

**GT-free.** Fusion keeps each site in information form: `Λ = Σ W_j`, `η = Σ W_j p_j`,
where `W_j` is the inverse of the view's anisotropic ground covariance. Dropping view *i* is
therefore a subtraction, and `p_{-i} = (Λ − W_i)⁻¹ (η − W_i p_i)`. For every site with
≥ 3 operational refit views, we compute `p_{-i}` for each view and report two residuals.
The **pixel residual** is the view's own detection minus the projection of `p_{-i}`
(`geo.ground_point_to_pano`, same height model as fusion, range uncapped). The **ground
residual** is the view's own raycast point minus `p_{-i}`, split along and across that
view's ray. Positive along means this view places the ramp farther than the others do.

**The scale identity.** Suppose every range is `k` times the truth, which is what a
wrong camera height does (`d = h / tan δ`). Let `s = 1 − 1/k`. Then

    member_i − held-out_i = s · g_i,   g_i = r_i u_i − Λ_{-i}⁻¹ Σ_{j≠i} W_j r_j u_j

holds *exactly*, and `g_i` is computed from the data. Least squares through the origin
estimates `s`, corrected for viewing geometry. The per-capture-year version is one `s` per
year in a joint fit, because sites mix rigs. Standard errors are cluster-robust by site. The
naive along-on-range slope (with intercept) is reported alongside for comparison with
RampNet#101.

**The null.** `g_i` contains view *i*'s own noisy range, and that noise also appears in the
residual. To measure how much `s` this alone produces, `null_scale` redraws every view to
look at its site's fused position exactly. It then perturbs each view by the error model's
own σ_along and σ_cross, plus a GPS shift that moves camera and point together, and reruns
the same fit. The null is only as good as the error model. For Mapillary the model carries
a 1.5° pitch budget that is known to be a guess (geo.py). That is why the Mapillary nulls
exceed the observed naive slopes: the model overstates Mapillary's range noise, so
null-corrected Mapillary `k` is a lower bound.

**GT-anchored.** A pano counts only if it passes the benchmark's own gate
(`eval_sites.judged_gt_panos`, benchmark tier 0.55, `mask_rig=False`). Each such pano
contributes one row per reference that maps to an operational site, projected from the
**full** site and from the site **with this pano's view left out**. References are
reported in two groups:

| group | reference pixel | independent of our geometry? |
|---|---|---|
| independent: `box` | centre of the reviewer's extent box (`boxes.json`: paterson, sao_paulo, richmond, annapolis) for a verdict-true detection or a missed mark | yes |
| independent: `missed` | a missed-ramp click with no box; the pano is not a refit member of the matched site, so full = left-out | yes |
| `peak` | a verdict-true detection with no box: the model's own peak, confirmed by the reviewer | no — left-out on these is the GT-free residual restricted to confirmed ramps |

A missed mark is matched to a site in world space (nearest operational site within 5 m,
eval_sites' radius, one to one per pano). This truncates the tail of its residual. GT
metres come from the reference pixel raycast from the judged pano, so they share that
pano's height model. The pixel residual is the independent number.

Sites are re-fused in memory (`--refuse`) at the benchmark tier with `mask_rig=False`,
under each height model. The `sites.jsonl` files on disk were fused at 2.6 m, and three of
them are **stale**: paterson over 34,427 panos (now 34,687), gainesville over 35,204 (now
37,435), sao_paulo over 22,741 (now 30,034). The tool flags that when it reads them. On
paterson, the disk and re-fused runs agree to 3 decimals on the pooled `s`. The in-memory
fuse is what lets 2.6 m and per-pano be compared on the same panos. Rebuilding
sites from `sites.jsonl` reproduces the stored positions to ≤ 1.2 cm.

## GT-free residual, per city

The statistic is the percentile of |residual| over held-out views. "views" is the
denominator. "along median" is signed.

| city | height | views (n) | sites | px p50 | px p90 | m p50 | m p90 | along median m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bend | 2.6m | 43288 | 9709 | 6.3 | 18.6 | 1.07 | 3.20 | 0.27 |
| bend | per-pano | 43282 | 9742 | 6.6 | 19.7 | 1.11 | 2.88 | -0.41 |
| paterson | 2.6m | 20027 | 4565 | 8.8 | 25.5 | 1.58 | 3.87 | 0.75 |
| paterson | per-pano | 23198 | 5030 | 7.3 | 21.9 | 1.30 | 3.19 | -0.57 |
| gainesville | 2.6m | 8645 | 2057 | 9.8 | 28.3 | 1.90 | 4.11 | 1.06 |
| gainesville | per-pano | 11178 | 2486 | 7.7 | 23.1 | 1.45 | 3.54 | -0.53 |
| sao_paulo | 2.6m | 12262 | 2371 | 7.4 | 22.7 | 1.24 | 3.29 | -0.00 |
| sao_paulo | per-pano | 11814 | 2389 | 8.7 | 26.6 | 1.44 | 3.33 | -0.53 |
| richmond | 2.6m | 7235 | 945 | 11.6 | 38.0 | 1.95 | 5.29 | 0.15 |
| clovis | 2.6m | 6841 | 1135 | 10.2 | 30.5 | 1.80 | 4.85 | 0.54 |
| laurens | 2.6m | 585 | 109 | 18.1 | 60.0 | 2.77 | 5.73 | -0.81 |
| laurens_gsv | 2.6m | 322 | 85 | 5.5 | 18.0 | 1.10 | 3.36 | 0.12 |
| annapolis | 2.6m | 24228 | 2727 | 10.6 | 27.8 | 1.69 | 5.14 | 0.73 |
| morgantown | 2.6m | 9084 | 1171 | 9.6 | 32.1 | 1.38 | 4.41 | 0.58 |

Per-pano only applies to runs with measured GSV heights, so the Mapillary runs and
laurens_gsv (no harvested depth) have 2.6 m rows only.

The two strongest patterns in `gtfree_breakdown.csv`:

- **Residual vs range.** Pixel residual falls with range, while metres grow. In paterson at
  2.6 m, for example, px p50 is 16.1 at 0–8 m vs 5.9 at 18–25 m, and m p50 is 0.98 vs 2.77.
  Near-field pixels are dominated by camera-position error: 1 m of GPS at 5 m range is 11°,
  or 32 px. Far-field metres are dominated by range error. The signed along median rises
  with range at 2.6 m (paterson +0.08, +0.47, +1.17, +2.28 m across the four buckets). This is the
  range-scale signature.
- **Capture-date delta and view count barely matter** next to range. For views more than
  36 months older than the site's newest view, px p50 changes by 0–2 px from the same-date
  bucket in every city. Sites with 6+ views are tighter than 3-view sites (e.g. richmond
  10.9 vs 17.0 px), as expected when more views share the held-out solution.

## Range scale: 2.6 m vs per-pano

`s` is the pooled scale fit (± 1.96 cluster-robust se). `k = 1/(1−s)` is the implied range
scale (> 1 means ranges run long). "null" is the same estimator on error-model noise with no
scale error. The naive slope is along-ray residual on range, with the null in brackets.

| city | height | views (n) | naive slope (null) | s ± 1.96 se | s null | k | k, null-corrected |
|---|---|---:|---:|---:|---:|---:|---:|
| bend | 2.6m | 43288 | 0.120 (0.047) | 0.053 ± 0.001 | 0.009 | 1.056 | 1.046 |
| bend | per-pano | 43282 | 0.074 (0.044) | -0.014 ± 0.002 | 0.007 | 0.986 | 0.979 |
| paterson | 2.6m | 20027 | 0.127 (0.050) | 0.084 ± 0.003 | 0.009 | 1.092 | 1.081 |
| paterson | per-pano | 23198 | 0.087 (0.064) | -0.026 ± 0.003 | 0.011 | 0.975 | 0.964 |
| gainesville | 2.6m | 8645 | 0.110 (0.054) | 0.092 ± 0.004 | 0.011 | 1.102 | 1.089 |
| gainesville | per-pano | 11178 | 0.120 (0.068) | -0.014 ± 0.004 | 0.011 | 0.987 | 0.976 |
| sao_paulo | 2.6m | 12262 | 0.068 (0.048) | 0.016 ± 0.003 | 0.009 | 1.017 | 1.007 |
| sao_paulo | per-pano | 11814 | 0.042 (0.046) | -0.038 ± 0.004 | 0.010 | 0.963 | 0.955 |
| richmond | 2.6m | 7235 | 0.131 (0.246) | 0.052 ± 0.007 | 0.063 | 1.055 | 0.989 |
| clovis | 2.6m | 6841 | 0.172 (0.274) | 0.083 ± 0.007 | 0.066 | 1.091 | 1.018 |
| laurens | 2.6m | 585 | 0.078 (0.191) | -0.076 ± 0.036 | 0.054 | 0.930 | 0.885 |
| laurens_gsv | 2.6m | 322 | 0.098 (0.006) | 0.029 ± 0.017 | -0.001 | 1.030 | 1.031 |
| annapolis | 2.6m | 24228 | 0.154 (0.274) | 0.117 ± 0.004 | 0.076 | 1.132 | 1.042 |
| morgantown | 2.6m | 9084 | 0.135 (0.221) | 0.102 ± 0.006 | 0.066 | 1.114 | 1.037 |

**RampNet#101's own estimator, and its null.** Full-site residuals (the view included),
site fixed effects, sites with ≥ 3 members spanning ≥ 4 m of range; the null runs this
estimator on the same simulated draws as the other nulls. "null share" is null ÷ observed.

| city | height | sites | members | #101 slope (± 1.96 se) | null | null share | LOO naive null share |
|---|---|---:|---:|---:|---:|---:|---:|
| bend | 2.6m | 9388 | 42240 | 0.108 ± 0.003 | 0.040 | 0.37 | 0.39 |
| paterson | 2.6m | 4445 | 19653 | 0.116 ± 0.005 | 0.042 | 0.37 | 0.40 |
| gainesville | 2.6m | 1965 | 8358 | 0.113 ± 0.008 | 0.050 | 0.45 | 0.49 |
| sao_paulo | 2.6m | 2265 | 11915 | 0.058 ± 0.006 | 0.044 | 0.76 | 0.71 |
| richmond | 2.6m | 890 | 7040 | 0.128 ± 0.013 | 0.249 | 1.96 | 1.89 |

The re-fused sites here are a little larger than the on-disk ones #101 used (results.jsonl
grew), which is why paterson has 4,445 sites rather than 4,433; the slope is the same to
three decimals. On the on-disk sites the estimator reproduces #101's table exactly.

**How well the null's noise matches the data.** The null perturbs each view by the error
model's own σ, so it is only as good as those σ. Leave-one-out |residual| in metres,
observed vs simulated, 2.6 m:

| city | |along| p50 / p90 observed | simulated | |cross| p50 / p90 observed | simulated |
|---|---:|---:|---:|---:|
| bend | 0.84 / 3.05 | 1.04 / 2.65 | 0.34 / 1.02 | 0.81 / 1.97 |
| paterson | 1.26 / 3.70 | 1.07 / 2.75 | 0.51 / 1.44 | 0.81 / 1.98 |
| gainesville | 1.55 / 3.94 | 1.08 / 2.79 | 0.56 / 1.63 | 0.81 / 1.98 |
| sao_paulo | 0.94 / 3.08 | 1.02 / 2.62 | 0.43 / 1.35 | 0.77 / 1.88 |

On GSV the model overstates cross-ray noise (1.4–2.4× at the median, 1.2–1.9× at p90) and understates the along-ray tail
(p90) by 13–29%. On Mapillary it overstates both (simulated |along| p50 ≈ 2.8 m vs
observed 1.1–2.0 m). So the null shares above are indicative, not exact.

**With an r² term.** The pooled fit again, with a second column for a constant error in the
detection's dip angle (`range_offset_levers`: a point moves by (r² + h²)/h per radian).
The offset is expressed as a constant dy in heatmap px, + = detections sit lower (nearer).

| city | height | k, scale only | k, with r² term | implied dy offset px (± 1.96 se) |
|---|---|---:|---:|---:|
| bend | 2.6m | 1.056 | 0.943 | −2.79 ± 0.10 |
| paterson | 2.6m | 1.092 | 1.015 | −1.64 ± 0.15 |
| gainesville | 2.6m | 1.102 | 1.055 | −0.95 ± 0.26 |
| sao_paulo | 2.6m | 1.017 | 0.926 | −2.34 ± 0.21 |
| bend | per-pano | 0.986 | 0.873 | −3.23 ± 0.11 |
| paterson | per-pano | 0.975 | 0.849 | −3.26 ± 0.14 |
| gainesville | per-pano | 0.987 | 0.858 | −3.15 ± 0.21 |
| sao_paulo | per-pano | 0.963 | 0.876 | −2.49 ± 0.24 |

`k` moves by 0.05–0.13 when the offset term is allowed, so the scale-only `k` is not a
clean camera-height measurement. The fitted offset has the opposite sign to the GT's
box-minus-peak offset (−1.2 px, i.e. peaks sit *low*), so the r² fit is not simply
recovering that offset either; the two columns are strongly collinear over the 5–25 m
range band. What survives is the *comparison* between rigs and height models, which both
fits share. The r² columns are in `range_slope.csv` (`r2fit_*`).

By capture year (joint fit, scale only; GSV vintages with ≥ 150 held-out views; the rest
are in `range_slope.csv`):

| city | capture year | views (n) at 2.6 m | k at 2.6 m [95% CI] | views (n) per-pano | k per-pano [95% CI] |
|---|---|---:|---:|---:|---:|
| bend | 2012 | 1815 | 0.944 [0.937–0.951] | 1819 | 0.943 [0.936–0.950] |
| bend | 2017 | 301 | 0.956 [0.936–0.978] | 305 | 0.957 [0.935–0.980] |
| bend | 2018 | 308 | 0.944 [0.926–0.964] | 308 | 0.939 [0.919–0.959] |
| bend | 2019 | 685 | 1.052 [1.039–1.065] | 660 | 0.974 [0.962–0.987] |
| bend | 2021 | 625 | 1.064 [1.050–1.077] | 631 | 0.997 [0.984–1.010] |
| bend | 2024 | 38485 | 1.062 [1.061–1.064] | 38489 | 0.989 [0.987–0.991] |
| bend | 2025 | 1047 | 1.045 [1.035–1.056] | 1050 | 0.979 [0.969–0.989] |
| paterson | 2012 | 293 | 0.948 [0.930–0.967] | 301 | 0.940 [0.921–0.959] |
| paterson | 2014 | 186 | 0.920 [0.898–0.942] | 185 | 0.925 [0.904–0.946] |
| paterson | 2016 | 153 | 0.937 [0.912–0.963] | 152 | 0.932 [0.907–0.958] |
| paterson | 2018 | 1013 | 1.045 [1.036–1.055] | 1045 | 0.962 [0.953–0.972] |
| paterson | 2019 | 1135 | 1.043 [1.034–1.052] | 1136 | 0.971 [0.962–0.979] |
| paterson | 2020 | 1901 | 1.055 [1.048–1.062] | 1938 | 0.969 [0.962–0.976] |
| paterson | 2021 | 1337 | 1.052 [1.043–1.060] | 1343 | 0.972 [0.963–0.980] |
| paterson | 2024 | 5969 | 1.037 [1.033–1.041] | 6063 | 0.967 [0.963–0.971] |
| paterson | **2025** | 7640 | **1.191** [1.187–1.196] | 10627 | **0.987** [0.982–0.991] |
| gainesville | 2018 | 298 | 0.934 [0.916–0.952] | 295 | 0.927 [0.910–0.945] |
| gainesville | 2021 | 160 | 1.067 [1.045–1.089] | 156 | 0.988 [0.962–1.016] |
| gainesville | 2022 | 319 | 1.074 [1.056–1.093] | 349 | 0.973 [0.956–0.991] |
| gainesville | 2023 | 502 | 1.063 [1.048–1.078] | 521 | 0.988 [0.974–1.004] |
| gainesville | 2024 | 2113 | 1.039 [1.032–1.046] | 2174 | 0.998 [0.991–1.006] |
| gainesville | 2025 | 544 | 1.053 [1.040–1.067] | 574 | 0.988 [0.974–1.002] |
| gainesville | **2026** | 4278 | **1.179** [1.172–1.186] | 6684 | **0.988** [0.982–0.993] |
| sao_paulo | 2014 | 355 | 0.955 [0.941–0.970] | 352 | 0.950 [0.935–0.964] |
| sao_paulo | 2022 | 316 | 1.017 [1.001–1.034] | 283 | 0.938 [0.920–0.956] |
| sao_paulo | 2023 | 748 | 1.016 [1.004–1.029] | 712 | 0.946 [0.934–0.958] |
| sao_paulo | 2024 | 6839 | 1.014 [1.010–1.018] | 6624 | 0.975 [0.971–0.979] |
| sao_paulo | 2025 | 3653 | 1.024 [1.018–1.030] | 3484 | 0.945 [0.940–0.951] |

What this says:

1. **The rig ranking replicates, from a different instrument.** The camera-height study
   ([docs/camera-height-study.md](camera-height-study.md)) found the 2025–26 rig
   triangulates to ~2.14 m when associated at 2.6 m (paterson), against ~2.5 m for older
   vintages. That predicts `k ≈ 2.6/2.14 = 1.21` and `≈ 1.04`. This fit reads 1.19 and
   1.04–1.06. It uses no depth and no bearing-only pairs: every view of every ≥ 3-view site,
   solved by GLS. Both instruments are associated at 2.6 m, so both are pulled toward
   k = 1: the study's fixed point for this rig (1.98 m paterson, 1.92 m gainesville)
   implies k ≈ 1.31–1.35, and 1.18–1.19 is a lower bound. The new-rig figure rests on two
   city-years (paterson 2025, gainesville 2026). A capture year is not a rig: bend 2025
   reads 1.045 and gainesville 2025 reads 1.053, like the older rigs.
2. **Per-pano heights remove the between-rig disagreement.** Paterson's views grow from
   20,027 to 23,198 and gainesville's from 8,645 to 11,178, because the new rig's views now
   agree with the old ones and associate. Held-out px p50 improves in paterson
   (8.8 → 7.3) and gainesville (9.8 → 7.7), the two cities where the new rig is roughly
   half the imagery. It does not improve in bend (6.3 → 6.6) or sao_paulo (7.4 → 8.7).
3. **Per-pano ranges run 1–4% short, not 6–16%.** Every vintage from 2019 on reads
   k < 1 under per-pano heights (0.94–1.00). That matches the study in direction (depth heights
   are low), but it is smaller. Both instruments are biased toward the association's own
   height: a view that disagrees is less likely to have been associated at all. So neither
   figure is the fixed point. The bias pushes this one toward 1 under per-pano, which
   makes 1–4% a lower bound on the shortfall.
4. **The oldest imagery reads k ≈ 0.92–0.96 under both models** (2012–2018 in bend and
   gainesville, 2012–2017 in paterson, 2014 in sao_paulo). Its heights are almost all unmeasured
   (stand-in grounds and degenerate payloads: 35 of 1,139 paterson panos from 2012–2017
   and 579 of 6,000 bend panos from 2012–2018 have one), so they fall back to 2.6 m. The study found those
   panos triangulate to ~2.9 m, which predicts `2.6/2.9 ≈ 0.90`.

## GT-anchored placement

"full" projects the whole site. "left-out" drops the judged pano's own view, so the
reference never contributed to the position. The statistic is the percentile of |residual|
over references, and "n" is the denominator. For `missed` references, full = left-out,
because the pano was never a member. Left-out `peak` rows need at least one other view,
which is why their n is smaller.

| city | height | refs | full: n | full px p50 / p90 | full m p50 / p90 | left-out: n | left-out px p50 / p90 | left-out m p50 / p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bend | 2.6m | independent | 41 | 4.5 / 15.3 | 1.42 / 2.91 | 41 | 4.5 / 15.3 | 1.42 / 2.91 |
| bend | 2.6m | peak | 243 | 4.5 / 11.4 | 0.72 / 2.70 | 230 | 6.3 / 20.2 | 1.08 / 3.35 |
| bend | per-pano | independent | 45 | 3.7 / 18.7 | 1.50 / 3.17 | 45 | 3.7 / 18.7 | 1.50 / 3.17 |
| bend | per-pano | peak | 243 | 4.9 / 12.7 | 0.82 / 2.15 | 227 | 6.9 / 20.4 | 1.16 / 2.75 |
| paterson | 2.6m | independent | 130 | 9.2 / 34.7 | 1.69 / 3.73 | 124 | 11.0 / 42.7 | 2.00 / 3.99 |
| paterson | 2.6m | peak | 156 | 5.7 / 17.4 | 1.13 / 3.31 | 145 | 8.5 / 28.7 | 1.75 / 4.18 |
| paterson | per-pano | independent | 157 | 7.2 / 23.0 | 1.28 / 3.18 | 156 | 7.6 / 28.7 | 1.40 / 3.45 |
| paterson | per-pano | peak | 160 | 6.0 / 15.2 | 0.92 / 2.64 | 154 | 8.4 / 23.9 | 1.23 / 3.02 |
| gainesville | 2.6m | independent | 30 | 15.0 / 44.6 | 1.90 / 3.74 | 30 | 15.0 / 44.6 | 1.90 / 3.74 |
| gainesville | 2.6m | peak | 172 | 6.4 / 17.1 | 0.98 / 2.93 | 135 | 13.1 / 30.6 | 1.81 / 3.82 |
| gainesville | per-pano | independent | 43 | 9.0 / 36.3 | 2.15 / 4.43 | 43 | 9.0 / 36.3 | 2.15 / 4.43 |
| gainesville | per-pano | peak | 180 | 5.6 / 17.0 | 0.86 / 2.99 | 164 | 8.9 / 29.3 | 1.32 / 3.62 |
| sao_paulo | 2.6m | independent | 130 | 7.7 / 31.0 | 1.12 / 3.10 | 126 | 9.3 / 35.3 | 1.36 / 3.63 |
| sao_paulo | 2.6m | peak | 110 | 5.5 / 15.7 | 0.99 / 2.78 | 105 | 7.6 / 26.3 | 1.38 / 3.75 |
| sao_paulo | per-pano | independent | 135 | 9.4 / 32.6 | 1.41 / 3.82 | 127 | 12.0 / 41.5 | 1.68 / 3.87 |
| sao_paulo | per-pano | peak | 110 | 7.2 / 15.0 | 1.14 / 3.33 | 104 | 9.3 / 25.6 | 1.48 / 3.71 |
| richmond | 2.6m | independent | 229 | 10.0 / 31.0 | 1.82 / 5.04 | 218 | 12.3 / 39.1 | 2.01 / 5.69 |
| clovis | 2.6m | independent | 30 | 9.2 / 23.5 | 1.66 / 3.67 | 30 | 9.2 / 23.5 | 1.66 / 3.67 |
| clovis | 2.6m | peak | 132 | 8.2 / 21.3 | 1.31 / 4.26 | 121 | 10.8 / 37.0 | 1.87 / 5.23 |
| laurens | 2.6m | independent | 72 | 17.7 / 54.1 | 2.81 / 4.42 | 72 | 17.7 / 54.1 | 2.81 / 4.42 |
| laurens | 2.6m | peak | 97 | 10.7 / 42.0 | 1.77 / 4.09 | 92 | 13.2 / 56.4 | 2.49 / 5.29 |
| laurens_gsv | 2.6m | independent | 52 | 10.0 / 37.1 | 1.27 / 2.86 | 52 | 10.0 / 37.1 | 1.27 / 2.86 |
| laurens_gsv | 2.6m | peak | 108 | 3.5 / 9.0 | 0.60 / 2.42 | 89 | 6.1 / 19.9 | 1.10 / 3.36 |
| annapolis | 2.6m | independent | 124 | 11.7 / 30.8 | 1.87 / 4.64 | 123 | 13.7 / 34.1 | 2.11 / 4.82 |
| annapolis | 2.6m | peak | 101 | 10.4 / 25.2 | 1.40 / 5.19 | 100 | 12.1 / 30.1 | 1.62 / 5.60 |
| morgantown | 2.6m | independent | 50 | 14.2 / 37.9 | 1.63 / 3.31 | 50 | 14.2 / 37.9 | 1.63 / 3.31 |
| morgantown | 2.6m | peak | 188 | 8.7 / 27.1 | 1.01 / 3.36 | 184 | 11.2 / 34.0 | 1.21 / 3.94 |
| **ALL (10 splits)** | 2.6m | independent | 888 | 10.6 / 35.6 | 1.71 / 4.19 | **866** | **11.8 / 40.4** | 1.89 / 4.41 (truncated) |
| **ALL (10 splits)** | 2.6m | box on a detection (untruncated) | 445 | 9.3 / 28.2 | 1.57 / 4.41 | **423** | 11.2 / 36.9 | **1.86 / 5.22** |
| ALL (10 splits) | 2.6m | peak | 1315 | 6.5 / 19.1 | 1.03 / 3.29 | 1209 | 9.5 / 29.3 | 1.52 / 4.18 |
| GSV-4 | 2.6m | independent | 331 | 8.8 / 33.5 | 1.48 / 3.56 | 321 | 9.8 / 39.3 | 1.65 / 3.76 |
| GSV-4 | per-pano | independent | 380 | 7.5 / 28.8 | 1.41 / 3.69 | 371 | 8.9 / 32.6 | 1.59 / 3.80 |

(The richmond `peak` row, n = 8, is omitted here: richmond's boxes cover almost every
reference. GSV-4 is `ALL_PER_PANO_CITIES` in the CSV, meaning bend, paterson, gainesville
and sao_paulo pooled.)

**Quote the pixels.** The pixel residual is the placement figure that never raycasts the
reference. Metres are given for two reasons only, and with two limits. First, 443 of the
866 independent references are missed marks (and box centres drawn on a missed mark),
which only reach a site within 5 m, so their metres are truncated; the untruncated metres
are the 423 box centres on a member pano's detection (`box_on_detection` in the CSV), p50
1.86 m / p90 5.22 m. Second, every GT metre raycasts the reference pixel under the pano's
own height model, so it inherits that model's range scale.

Reading it:

- **The box has a floor.** For boxed detections, the box centre sits a median
  **4.7 px** (n = 445 boxed detections, pooled, 2.6 m) from the model's own peak. The peak lands somewhere on
  the ramp, not at its centre. So roughly 4–5 px of every `box` residual is where on the ramp
  the reference point was taken, not placement error.
- **The left-out vs full gap is the value of the held-out pano's own view:** +1.0 px
  (independent, 11.8 vs 10.8, n = 866) and +2.4 px (peak, 9.5 vs 7.1, n = 1,209) at the
  median, pooled, on the rows that have both numbers (`full_on_loo_rows` in the CSV). The
  first version compared the left-out rows with all full rows, a different set, and read
  +1.2 / +3.0.
- **Per-pano vs 2.6 m on GT is mixed.** It helps paterson clearly (left-out independent
  p50 11.0 → 7.6 px). It hurts sao_paulo (9.3 → 12.0 px). Pooled over the four GSV cities,
  it improves 9.8 → 8.9 px, on denominators that differ (321 vs 371, because association
  changes with the height model). This agrees with the camera-height study: GT at 125
  panos per city cannot choose a height model. The GT-free fit, with 8,000–43,000 views per
  city, can see the rig effect.
- **Missed marks are matched within 5 m, so their tail is truncated.** 9–53 missed marks
  per split were unplaceable (above the horizon or beyond 25 m). Another 12–71 per split had
  no operational site within 5 m. Those ramps are recall misses, not placement errors, and
  are counted in each city's `report.md`. Laurens-Mapillary has the most (71 of 152), in
  line with its low recall.

## What each variant can and cannot see

**GT-free** sees anything that makes views of one ramp disagree: a range scale that
differs between rigs or ranges, a bad heading, a bad position, a detector that fires on
different parts of the ramp from different sides. It cannot see **a bias every view of a
site shares**. That includes a common translation (e.g. a whole Mapillary SfM sequence
shifted 8 m, SidewalkWebpage#5361), and a scale error on views that all look at the ramp
from the same direction, because the held-out solution inherits both. The pooled `k` is
identified only through viewing-geometry diversity. Its residuals are also **truncated by
association**: a view that disagreed by more than the chi-square gate, or 8 m, was never a
member. So its p90s are floors, and its `s` is pulled toward the height model the site was
fused under. Per-pano's 1–4% shortfall is therefore a lower bound. The absolute scale
still needs an external anchor, as RampNet#101 already notes.

**GT-anchored** sees accuracy against a mark our geometry never touched (box centres,
missed clicks). The judged pano's own view is left out, so the mark cannot pull the position
toward itself. It is limited by size (30–229 independent references per split, and only
four splits have boxes), by the 5 m match truncation on missed marks, and by the ~4.7 px
box-vs-peak floor. Its metres are the reference raycast under the pano's height model, so
they are not independent in the along direction. Quote the pixels, and quote metres only
for the untruncated box-on-detection subset.

**Tier.** Every site in this document is fused at the benchmark tier 0.55 with
`mask_rig=False`, so GT joins stay keyed to the bundles. Production ships the operational
tier 0.30, whose extra sites are mostly single-view and low confidence; none of these
placement numbers describe them.

## Reproduce

```
python scripts/reprojection_residual.py bend paterson gainesville sao_paulo richmond \
    clovis laurens laurens_gsv annapolis morgantown --camera-height-m 2.6 per-pano \
    --refuse --publish docs/figures/reprojection-residual/data
```

This takes ~3 minutes on one CPU. Per-run outputs (`views.csv`, `gt_anchored.csv`,
`breakdown.csv`, `report.md`) go to `runs/<city>/reprojection/`, which is gitignored. The
aggregates go to `runs/_summary/reprojection/`, and `--publish` copies them here. The null
simulation is seeded, so the tables are deterministic given the same `results.jsonl` files.
