# Bayonne: the first full Panoramax run, and whether the borrowed error model fits (#57)

Issue [#57](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/57), part 1.
Part 2 of the issue was PR #70.

> **Which run each part measures.** The run was densified from 10 m to 5 m thinning on
> 2026-10-06 ([#147](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/147),
> §1.1); `runs/bayonne/` now holds the 5 m run (50,409 panos). **Section 2, the key
> takeaways, Figures 1-6 and 7b, and `data/inputs.csv` (results sha256 `cb03768c…`) are
> the 5 m run's** (re-run 2026-10-07). The 10 m readings they replace are kept side by side
> in §2's [10 m vs 5 m table](#what-densifying-changed-10-m--5-m). §1's table, census and
> position check describe the 10 m pass; §1.1 has the 5 m ones. Three things stay on the
> 10 m run because redrawing them needs the network: the skip list (`data/skips.csv`), the
> example figures 7a, 7c and 7d, and the RampNet bundle (§3). Every 10 m pano is also in
> the 5 m run.

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
> 1. **Still TOO TIGHT under the amended rule, but weaker at 5 m: now within sampling
>    noise of the bar.** Bayonne's chi²/dof is 0.556 [95% CI 0.536, 0.576] (median form
>    0.559 [0.529, 0.588]) against a 0.538 bar. At 10 m it was 0.580 [0.539, 0.618]. The
>    point estimate is still over the bar (3%, down from 8%), so the rule as written still
>    reads TOO TIGHT. The pooled CI's lower bound moved from 0.539 to 0.536, so it now
>    includes the bar; the median form's CI included it at 10 m too. Bayonne cannot be
>    told apart from the edge of ADEQUATE.
>    - Mapillary GoPro Max populations read 0.18-0.50: Richmond's GoPro Max views 0.499
>      [0.467, 0.532] (median form 0.453), Laurens 0.431, Morgantown 0.177. Bayonne is
>      1.11x Richmond's GoPro Max views (1.16x at 10 m), and **not distinguishable from
>      them**: the CIs just separate when sites are resampled (0.536 against 0.532), but
>      Richmond's 1,301 GoPro Max views come from only 14 capture sequences, and resampling
>      sequences gives Richmond [0.454, 0.559], which overlaps Bayonne's [0.534, 0.578].
>    - **The site-size confound is weaker than it looked.** Densifying raised the median
>      site from 3 to 4 views (2 sequences either way; Richmond 7 and 4) and nearly tripled
>      the held-out views (1,039 → 3,626), but chi²/dof fell only 0.024. Within the 5 m
>      run it has no trend with site size: 0.549 [0.508, 0.588] / 0.535 [0.499, 0.569] /
>      0.579 [0.546, 0.615] for 3 / 4-5 / 6+ views. The 6+ sites have Richmond's make-up
>      (median 7 views from 4 sequences), and their CI sits above the 0.538 bar, so
>      "within sampling noise of the bar" describes the pooled number only.
>
>    The single-consumer-rig confound stands. *Figure 1; `data/fig1_verdict.csv`
>    `chi2_dof`, `chi2_dof_median`, `*_lo`/`*_hi`, sequence bootstrap `*_seq_lo`/`*_seq_hi`;
>    site make-up in `data/fig3_seqsplit.csv`; by site size in `data/cuts.csv`
>    (`dimension = n_views`).*
> 2. **The excess is across the ray, not along it, so it is not tilt. Unchanged at 5 m.**
>    - Against all of Richmond, Bayonne's residual is a near-constant +1.2 to +1.9 m in
>      every range bin.
>    - Against Richmond's GoPro Max views, re-solved from GoPro Max views only
>      (rig-matched), there is no consistent along-ray excess (+0.2 / −0.2 / −0.5 /
>      0.0 m by range bin; the along-ray CIs overlap in every bin). A **+0.6 to +0.8 m
>      excess across the ray** remains (+0.6 to +1.0 m at 10 m). Its site-bootstrap CIs
>      clear Richmond's GoPro Max CIs in all four bins (the 18-25 m bin by 0.04 m); with
>      capture sequences resampled they clear only the 8-12 and 12-18 m bins, which hold
>      most of the views. Position, with some heading error possible.
>    - The leave-one-out-calibrated `sigma_gps` is 2.10 m (2.16 m at 10 m), against 1.11 m
>      for Richmond overall. Against the GoPro Max populations it is 1.81 m (Richmond's
>      GoPro Max views), 1.74 m (Laurens) and 0.80 m (Morgantown). All of them sit under
>      `MAPILLARY_ERRORS`' 3 m. Against Richmond's GoPro Max views the gap rests on the
>      site bootstrap only: resampling capture sequences, their CI is [1.55, 2.09] (12
>      sequences) against Bayonne's [2.05, 2.16], which overlap. Against Laurens and all
>      of Richmond the CIs separate either way.
>    - Same rig confound as takeaway 1.
>
>    *Figure 2; `data/fig2a_range.csv` `m_p50`/`along_p50`/`cross_p50` by `subset`,
>    `data/fig2b_sigma_gps.csv` `sigma_gps_chi2_1_m`.*
> 3. **The amendment's mechanism is not established. Unchanged at 5 m.** The amendment
>    said shared same-sequence error cancels in the leave-one-out. Richmond, the anchor,
>    reads the opposite way with separated CIs (0.404 with all-same-sequence mates vs
>    0.286 with none). Four runs, Bayonne included (0.418 vs 0.672), go the predicted way
>    with separated CIs, and Laurens does too but with overlapping CIs.
>    *Figure 3; `data/fig3_seqsplit.csv` `chi2_dof`.*
> 4. **Do not apply `pers:pitch`/`pers:roll`. Stronger at 5 m.** On the same sites at the
>    25 m cap, every sign convention loosens real-tilt members (p90 7.56 m flat vs
>    9.85-13.20 m at 0.55, on 225 groups against 97 at 10 m). *Figure 4;
>    `data/fig4_pose.csv` `median_m`/`p90_m`.*
> 5. **Bayonne's detection rate is inside the Mapillary range.** 0.148 per pano at 0.55
>    (0.135 at 10 m), against 0.123-1.048 for the five Mapillary runs. It is worth ground
>    truth (Figure 7d shows what the detector fires on), not anomalous.
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
The run was later densified to 5 m (§1.1): the thinning experiment (PR #144) found that on
a Bayonne sub-area 10 m keeps 81% of robust sites at 0.30 against 99% at 5 m.

The A40 then ran at 0.85 s/pano, so 5 m would have taken about 12 h; the estimate's
constant is an RTX 3070 figure. `manifest.json` recorded neither at the time;
[#126](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/126) added
`thin_spacing_m` and `panos_before_thinning` to every run entry (the Bayonne manifest predates
it; its values are 10 m and 73,161). Those values were backfilled into `runs/bayonne/manifest.json`
by hand on 2026-10-05 (top-level `thin_spacing_m` binds the run; `thin_spacing_backfilled`
records the evidence): `run.log` printed `73161 → 28634 panos (best per ~10 m grid cell)`, and
re-thinning the reused `scan.json` at 10 m reproduces the 28,634 ids of
`already_processed.txt` exactly (5 m gives 50,528).

**Skips (110, all cached at the time; `data/skips.csv`).**

- **104** are GoPro **MAX2** uploads on the OSM-FR instance. Each declares 7680x3840 but
  serves a 7680x2940, vertically cropped `hd` image. The 2:1 check rejects them, which is
  correct ([Figure 7c](figures/panoramax-bayonne/fig7c_max2_skip.jpg)).
- **3** have no `view:azimuth`, which is also correct.
- **3** IGN panos (`139d37d3-…`, `1f8be062-…`, `2c902fc4-…`) have metadata and pixels
  that are fine today. main.py does not log per-pano skip reasons, and two causes fit:
  - a transient asset 404 (the source cached a 404 as permanent at the time);
  - a body that arrived incomplete and failed to decode. `_download_image`'s bare
    `except Exception` cached any decode exception as permanent.

  [#127](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127) tracked
  this. Its
  [amendment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127#issuecomment-5934720793)
  notes that a narrowed catch alone would not cover truncation. Both causes are now
  retryable: #137 made a truncated body a failure, and since 2026-10-05 a Panoramax 404
  (image asset or catalog item) is a failure too, never a cached skip, because the
  infrastructure is young and a 404 may be transient. A picture that really is gone is
  re-requested once per resume, forever, which is cheap. These 3 ids were removed from the
  local `already_processed.txt` (backup `already_processed.txt.bak-2026-10-05`), so the
  next resume retries them; the 107 explained skips stay cached. Whether any 404 cached
  as a skip before that change was still in a cache was checked on 2026-10-06: none
  remain. This run predates per-pano skip logging, so the original reasons are unknown;
  `data/skips.csv` re-queried each id afterwards, and every one of the 107 still cached
  reproduces as a deterministic skip under current code, so a 404 at run time could not
  have cost a pano. The 5 m resume (§1.1) retried all 3 and processed them.

After the 5 m resume the cache holds **119** skips: the 107 deterministic ones above and 12
new GoPro MAX2 cropped-`hd` skips among the added cells. No cached 404 skip remains.

No HTTP 429 or 5xx failures occurred on either instance.

**Census** (10 m pass; `data/census/`. [Figure 6](figures/panoramax-bayonne/fig6_census.png)
now draws the 5 m census, §1.1):

- **Rig:** GoPro Max 99.9% (5760x2880 for 58.0%, 5376x2688 for 41.9%).
- **Capture year:** 2026 56.0%, 2025 13.0%, 2024 28.0%, earlier 2.9%.
- **Producer:** `sig_bayonne` (the municipality, IGN instance, Etalab 2.0) holds 95.3%.
- **Pose:** absent 42.0%, real tilt 36.8%, reported as exactly 0/0 21.2%.
- **Detections:** at tier 0.55, 3,861 on 2,947 panos (**0.135 per pano**). At 0.30,
  10,011 (0.351 per pano).

The six runs at 0.55: Clovis 0.123, **Bayonne 0.135**, Laurens 0.158, Morgantown 0.203,
Annapolis 0.548, Richmond 1.048. Bayonne is inside the Mapillary range, and stays there at
5 m (0.148, [Figure 5](figures/panoramax-bayonne/fig5_detections.png)).

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

**Position check** (10 m run; `runs/bayonne/position_check.json` as of commit `c096987`,
since overwritten by the 5 m check in §1.1): median cross-track to OSM
centerlines **1.54 m**, IQR 0.70-3.13 m. That is at the metric's ~1.75 m floor. Panoramax
serves one position per pano, so the check reports and never gates.

4,027 panos (14%) sit more than 30 m from any street of the queried classes (6,113, 12%, at
5 m: orange in Figure 6D). What they are (paths, squares, parks) was not verified.

## 1.1 Densify to 5 m ([#147](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/147))

The run directory was rebound from 10 m to 5 m by hand (commit `c096987`; the manifest's
`thin_spacing_changed` block records the date, 10 → 5, why and how) and resumed on Hyak.
The resume reused the 2026-09-30 scan on purpose: re-thinning the same `scan.json` gives a
10 m set (28,634) that is an exact subset of the 5 m set (50,528), so the finished
`results.jsonl` is exactly the 5 m population and none of the 10 m work was redone. Panos
added to Panoramax after 2026-09-30 are not in the run.

| | |
|---|---|
| where | Hyak `ckpt-g2`, 1x NVIDIA L40S, 16 CPUs, 64 GB; Slurm job 41473672 (`sal_bayonne_ckpt.sbatch`, `--requeue`); not preempted |
| command | `main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --reuse-scan --thin-spacing 5` (batch 1, default concurrency), repo at `c096987` |
| scan | reused, 2026-09-30T22:56:20Z (137.8 h old): 73,161 in-area 360 pictures |
| thinning | 5 m: **50,528** panos; 28,631 already cached, **21,897** new |
| processed / skipped / failed | **21,885** / 12 / **0** in one pass |
| skips | all 12 are GoPro MAX2 cropped-`hd` uploads (7680x2940), cached as deterministic; the 3 retried ids (`139d37d3-…`, `1f8be062-…`, `2c902fc4-…`) were processed |
| wall time | 2.4 h (16:45:21 → 19:11:02 UTC); 2.50 panos/s (8,726 s in forward) |
| model | `rampnet-model@606a11956743` |
| `results.jsonl` | **50,409** lines, 50,409 unique `panorama_id`s; sha256 `cb03768cc4cefd0bcb9792e18ff339ad798e43fd3e560deee431387796d6eeef` |
| cache | 50,528 ids = 50,409 processed + 119 skips (107 from the 10 m pass + 12) |
| position check | 50,409 panos, **0 flagged** (Panoramax reports, never gates); median cross-track **1.58 m**, IQR 0.72-3.23 m; **6,113** panos (12%) not within 30 m of a street |

The position check re-run locally from the cached `osm_streets.json` reproduced the run's
own `position_check.json` exactly (only `checked_at` differed).

**Census at 5 m** (`data/census_5m/`, drawn in [Figure 6](figures/panoramax-bayonne/fig6_census.png);
same command as step 6 with `--out …/census_5m`; the 10 m census in `data/census/` is kept
as the evidence for §1):

| tier | panos with a detection | detections | per pano | on rig | in logo band |
|---|---:|---:|---:|---:|---:|
| 0.30 | 11,895 (23.6%) | 19,206 | **0.381** | 15 | 2 |
| 0.55 | 5,624 (11.2%) | 7,449 | **0.148** | 1 | 0 |

At 0.55 the rate moves from 0.135 (10 m) to 0.148 (5 m), still inside the Mapillary range of
takeaway 5. Rig, year, pose and producer shares move by at most 1.6 points (GoPro Max 99.9%;
2026 54.4%; pose absent 43.3%, tilt 37.0%, 0/0 19.7%; `sig_bayonne` 96.5%). No producer name
in the census is email-shaped.

## 2. Does `MAPILLARY_ERRORS` fit Panoramax?

Everything in this section is measured on the 5 m run. The first version of this document
measured the 10 m run; issue #147 asked what densifying changes, and the table below keeps
both readings.

### What densifying changed (10 m → 5 m)

| reading (tier 0.55 unless noted) | 10 m run (28,524 panos) | 5 m run (50,409 panos) | what moved |
|---|---|---|---|
| held-out views / sites | 1,039 / 281 | 3,626 / 793 | 3.5x the views |
| views / sequences per site (median) | 3 / 2 | 4 / 2 | one more view; Richmond 7 / 4 |
| **chi²/dof** [95% CI] | 0.580 [0.539, 0.618] | **0.556** [0.536, 0.576] | **lower bound 0.539 → 0.536, so the CI now includes the 0.538 bar** |
| median form [95% CI] | 0.582 [0.522, 0.647] | 0.559 [0.529, 0.588] | includes the bar, as it did at 10 m |
| over the bar | 8% | 3% | verdict weaker, not flipped |
| vs Richmond's GoPro Max views (0.499) | 1.16x | 1.11x | site-bootstrap CIs just separate (0.536 vs 0.532); sequence-bootstrap CIs overlap (Bayonne [0.534, 0.578], Richmond [0.454, 0.559], 14 sequences): not distinguishable |
| m p50 [95% CI] | 3.66 [3.43, 3.85] | 3.44 [3.33, 3.55] | still over the 2.93 m bar |
| tier 0.30: chi²/dof, m p50 | 0.584, 3.70 | 0.586, 3.52 | — |
| excess vs Richmond all, by range bin (m) | +2.2 / +1.6 / +1.2 / +1.8 | +1.9 / +1.4 / +1.2 / +1.8 | — |
| along-ray vs Richmond GoPro Max, re-solved (m) | +0.5 / −0.2 / −0.5 / +0.2 | +0.2 / −0.2 / −0.5 / 0.0 | still no along-ray excess |
| cross-ray vs Richmond GoPro Max, re-solved (m) | +0.6 / +0.8 / +1.0 / +0.7 | +0.6 / +0.7 / +0.8 / +0.6 | slightly smaller, same reading |
| chi²/dof by range bin | 0.60 / 0.57 / 0.60 / 0.56 | 0.57 / 0.55 / 0.54 / 0.55 | still flat |
| `sigma_gps` at chi²/dof 1 [95% CI] | 2.16 m [2.06, 2.26] | 2.10 m [2.06, 2.15] | — |
| `sigma_gps` / `sigma_pitch` at target 0.269 | 4.70 m / > 15° | 4.57 m / > 15° | pitch still cannot fix it |
| range scale k | 1.087 | 1.109 | inside the Mapillary range either way |
| same-sequence split: all-same vs none (n) | 0.423 (189) vs 0.715 (330) | 0.418 (403) vs 0.672 (1,157) | predicted way, CIs separate |
| same-site pairs from one sequence | 0.37 | 0.33 | — |
| chi²/dof by pose: absent / 0/0 / tilt | 0.51 / 0.60 / 0.77 | 0.47 / 0.58 / 0.72 | same order |
| pose ablation, real-tilt p90: flat vs others | 7.58 vs 9.45-11.27 m (97 groups) | 7.56 vs 9.85-13.20 m (225 groups) | stronger |
| detections per pano | 0.135 | 0.148 | — |

The 10 m figures are in this branch's history: `data/` at commit `9c6fe6c`.

**No takeaway flipped.** One weakened: the TOO TIGHT verdict still holds on the point
estimate, but it is now 3% over the bar with the bar inside its CI. One confound shrank:
site size (takeaway 1). One finding got stronger: the pose ablation now rests on 2.3x the
groups.

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

**Sequence-bootstrap CIs** (added in the
[PR #162 review](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/162#issuecomment-6608889865)).
Views of one capture sequence share its position error, and a site bootstrap treats them
as independent, so `fig1_verdict.csv`, `fig2a_range.csv`, `fig2b_sigma_gps.csv` and
`cuts.csv` also carry a CI that resamples **capture sequences** (same draws and seed;
`*_seq_lo`/`*_seq_hi`, `seq_lo`/`seq_hi`, with `n_sequences`). For view-level statistics
the cluster is the held-out view's sequence; for the site-level `sigma_gps` calibration it
is the sequence most of the site's views come from. A view without a sequence id is its
own cluster. Where few sequences carry a population, as with Richmond's GoPro Max views
(14), the two CIs differ, and claims below say which one they rest on.

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
| Bayonne | 0.418 (403) | 0.672 (1,157) | predicted, CIs separate |
| Clovis | 0.205 (1,438) | 0.355 (538) | predicted, CIs separate |
| Laurens | 0.338 (25) | 0.414 (87) | predicted, **CIs overlap heavily** |
| Annapolis | 0.194 (1,900) | 0.240 (1,967) | predicted, CIs separate |
| Morgantown | 0.146 (1,237) | 0.217 (369) | predicted, CIs separate |

In two runs the groups also differ in site size, so the comparison is not a clean test of
cancellation either way (`views_per_site_p50` in the CSV):

- Richmond: median 4 views per site for all-same-sequence views against 8 for none.
- Annapolis: 5 against 8.
- Bayonne, Clovis, Laurens and Morgantown: the two groups match within one view
  (Bayonne 3 against 4).

The share of same-site view pairs that come from one sequence (`data/fig3_pair_share.csv`)
is 0.19 in Richmond, 0.33 in Bayonne, 0.36 in Annapolis, 0.40 in Laurens, 0.47 in
Morgantown and 0.49 in Clovis. It is below half in every run (0.19-0.49), but well
above Richmond's 0.19 elsewhere. The Richmond and Annapolis shares were measured before
the amendment and not disclosed then.

So the rule stands, but the stated reason for it is replaced by measured numbers. The
`sigma_gps` that brings chi²/dof to **1** (`data/fig2b_sigma_gps.csv`) is:

- **1.11 m** for Richmond (CI 1.05-1.16) and **2.10 m** for Bayonne (CI 2.06-2.15);
- 0.74-1.81 m across the other populations, including the GoPro Max ones: Richmond's
  GoPro Max views 1.81 m, Laurens 1.74 m, Morgantown 0.80 m.
- Resampling capture sequences instead of sites (`seq_lo`/`seq_hi`): Bayonne 2.05-2.16,
  Richmond 0.96-1.22, Laurens 1.61-1.87, and Richmond's GoPro Max views **1.55-2.09**
  (their 208 sites fall under 12 sequences), the one comparison whose CIs then overlap.

`MAPILLARY_ERRORS`' 3 m is about 2.7x Richmond's between-view scatter, and it is still
looser than Bayonne's. The constant reads as standing for absolute position error, which
a leave-one-out cannot see, more than for between-view scatter.

### Results ([Figure 1](figures/panoramax-bayonne/fig1_verdict.png); `data/fig1_verdict.csv`, `data/reprojection/`)

| run (views) | held-out views | sites | chi²/dof [95% CI] | median form | m p50 [95% CI] |
|---|---:|---:|---|---:|---|
| **Bayonne, 5 m (GoPro Max)** | 3,626 | 793 | **0.556** [0.536, 0.576] | 0.559 [0.529, 0.588] | **3.44** [3.33, 3.55] |
| *Bayonne, 10 m (superseded)* | *1,039* | *281* | *0.580 [0.539, 0.618]* | *0.582 [0.522, 0.647]* | *3.66 [3.43, 3.85]* |
| Richmond, all views | 7,235 | 945 | 0.269 [0.257, 0.282] | 0.181 | 1.95 [1.89, 2.02] |
| Richmond, GoPro Max views | 1,301 | 472 | **0.499** [0.467, 0.532] | **0.453** [0.417, 0.503] | 3.13 [2.96, 3.31] |
| Richmond, iSTAR Pulsar views | 5,230 | 813 | 0.194 | 0.136 | 1.69 |
| Laurens (GoPro Max) | 585 | 109 | 0.431 [0.387, 0.474] | 0.381 | 2.77 |
| Morgantown (GoPro Max) | 9,084 | 1,171 | 0.177 [0.167, 0.187] | 0.098 | 1.38 |
| Clovis (GoPro Fusion) | 6,841 | 1,135 | 0.219 | 0.151 | 1.81 |
| Annapolis (Trimble MX7) | 24,228 | 2,727 | 0.216 | 0.146 | 1.69 |

(CIs omitted from the table are in the CSV; the 10 m row is from `data/` at `9c6fe6c`.) At
tier 0.30 Bayonne reads 0.586, m p50 3.52 (`data/reprojection_t0.3/`). The comparators did
not change: their `results.jsonl` hashes match the 10 m analysis's.

### Verdict

**TOO TIGHT under the amended rule, but only just: 0.556 against the 0.538 bar (3% over).
The 95% CI [0.536, 0.576] includes the bar, and so does the median form's.** At 10 m it was
8% over, with the pooled CI's lower bound 0.001 clear of the bar (0.539; the median form's
CI already included it), so densifying weakened the verdict without flipping it. The rule
reads the point estimate, so the reading stays TOO TIGHT. In practice Bayonne
sits at the edge of the ADEQUATE band, not clearly outside it.

- **Rig.** Richmond's anchor is 72% iSTAR Pulsar views. Matched to rig, Bayonne is 1.11x
  Richmond's GoPro Max views (0.499; median form 0.559 vs 0.453), but not distinguishable
  from them. The site-bootstrap CIs just separate (0.536 against 0.532). Richmond's 1,301
  GoPro Max views come from only 14 capture sequences, though, and a per-sequence position
  offset is shared by every view of one; resampling the held-out view's sequence instead
  gives Richmond [0.454, 0.559] and Bayonne [0.534, 0.578] (591 sequences), which overlap
  (`fig1_verdict.csv` `chi2_dof_seq_lo`/`_hi`, `n_sequences`). Bayonne is still the
  highest point in Figure 1. Mapillary
  GoPro Max populations span 0.18-0.50 (Morgantown 0.177, Laurens 0.431, Richmond GoPro
  Max 0.499), so the rig alone does not set the level. This confound stands.
- **Site size.** At 10 m Bayonne sites had a median of 3 views, against Richmond's 7, and
  that was the second confound. The 5 m run tests it. The median site grew to 4 views
  (still 2 sequences; Richmond 7 and 4), and chi²/dof fell only from 0.580 to 0.556.
  Within the 5 m run it has no trend with site size: 0.549 [0.508, 0.588] (3 views, 933
  held-out views), 0.535 [0.499, 0.569] (4-5, 1,308) and 0.579 [0.546, 0.615] (6+, 1,385)
  (`data/cuts.csv`, `dimension = n_views`; site bootstrap). Site size explains little of
  the excess. The 6+ sites are Richmond-sized (median 7 views from 4 sequences, as
  Richmond's), and their CI sits above the bar under either bootstrap (sequence:
  [0.546, 0.614]): on sites like Richmond's, Bayonne is over
  it, and "at the edge of the band" is a reading of the pooled number.

So the excess cannot be separated from a single-consumer-rig effect, but it is no longer
plausibly a site-size effect. The metre clause fails (3.44 m against the 2.93 m bar), but
Richmond's own GoPro Max views also sit above that bar (3.13 m).

The original absolute rule would have left Bayonne unclassified: its chi² clause passes
and its metre clause fails.

**What the excess is.** The pre-registered remedy, a wider `sigma_pitch`, does not reach
0.269 at any value up to 15° (`data/reprojection/gtfree_summary.csv`). The evidence is in
[Figure 2](figures/panoramax-bayonne/fig2_position.png) and `data/fig2a_range.csv`:

- **The metre excess is roughly constant across range bins.** Against all of Richmond it
  is +1.9 / +1.4 / +1.2 / +1.8 m (0-8 / 8-12 / 12-18 / 18-25 m). A pitch error would
  grow with range.
- **Rig-matched, the excess sits across the ray.** The reference is Richmond's GoPro
  Max views re-solved from GoPro Max views only (`_rig_sites`), so each is held out
  against GoPro Max mates, as Bayonne's are. Against it, the along-ray difference is
  +0.2 / −0.2 / −0.5 / 0.0 m, while the cross-ray excess is +0.6 / +0.7 / +0.8 / +0.6 m.
  - Site bootstrap: the along-ray CIs overlap in all four bins; Bayonne's cross-ray CI
    clears Richmond's GoPro Max CI in all four, the 18-25 m bin by 0.04 m (lower bound
    1.673 against upper 1.629; `fig2a_range.csv` `cross_p50_lo`/`_hi`). Sequence
    bootstrap (Richmond's GoPro Max bins hold 11-13 sequences): the cross-ray CIs still
    separate at 8-12 m (1.540 against 1.163) and 12-18 m (1.562 against 1.224) but
    overlap at 0-8 m and 18-25 m (`cross_p50_seq_lo`/`_hi`). The reading holds on the
    bins with most of the views; it is not a four-bin result once sequences are resampled.
  - Bayonne's bins hold 503-1,415 views; Richmond's GoPro Max bins hold 128-346.
  - At 10 m the along-ray difference was +0.5 / −0.2 / −0.5 / +0.2 and the cross-ray
    excess +0.6 / +0.8 / +1.0 / +0.7. The reading is unchanged.
  - An earlier 10 m version only filtered the mixed-rig solution, so those views were held
    out against mostly iSTAR Pulsar mates. Re-solving narrowed the cross-ray gap at the
    outer bins; the reading did not change then either.
  - A range-independent cross-ray offset is camera position.
  - Heading error would grow with range: the cross-ray part grows only 1.64 to 1.88 m over
    a ~16 m span, which bounds heading at about 1° or less.
- **Camera height is not unusual.** The range-scale fit gives k 1.109 (1.087 at 10 m),
  inside the Mapillary range (Laurens 0.930, Richmond 1.055, Clovis 1.091, Morgantown
  1.114, Annapolis 1.132; `range_slope.csv`).
- Chi²/dof is also flat by range (0.57 / 0.55 / 0.54 / 0.55). This only says that the
  model's predicted variance tracks the residual's range dependence equally in every bin.
  The constant metre excess above is the evidence.

**Recommendation (exploratory, outside the pre-registered rule).** Bayonne's
between-view position scatter is about 1.9x Richmond's (2.10 vs 1.11 m, calibrated to
chi²/dof = 1). It is 1.2x the scatter of Richmond's GoPro Max views (1.81 m) and of
Laurens (1.74 m); against Richmond's GoPro Max views the sequence-bootstrap CIs overlap,
so that ratio is not distinguishable from 1. The 5 m run moved it by 0.06 m.

The `sigma_gps` that brings Bayonne to Richmond's 0.269 is 4.57 m (4.70 m at 10 m). That
follows only if Richmond's ~2.7x inflation is kept on purpose, with 3 m standing for
absolute error. Used for association gating, it would loosen gates past anything
measured. Nothing is adopted. The 5 m run was one of the two things this waited for, and
it does not argue for a change: with Bayonne at the edge of the band, the case for a
Panoramax `ErrorModel` now rests on ground truth (`eval_sites.py bayonne`'s match-radius
ablation).

**By pose group** (0.55, `data/reprojection/gtfree_breakdown.csv`): chi²/dof is 0.47 for
pose absent (1,979 views), 0.58 for 0/0 (658) and 0.72 for real tilt (989). Panos that
report a tilt are the worst placed under a flat raycast, as at 10 m (0.51 / 0.60 / 0.77).

**By capture age (exploratory, not in the rule).** Views from the site's newest capture
month read 0.503 [0.479, 0.527] (2,367 views). Views 1-18 months older than the site's
newest read 0.620 [0.580, 0.660] (760), 19-36 months 0.701 [0.646, 0.760] (436), and older
0.742 [0.606, 0.874] (63 views from 13 sequences) (`data/cuts.csv`, `dimension =
delta_months`; site bootstrap, the sequence bootstrap is beside it).

This cut largely repeats the same-sequence split of takeaway 3. Every view whose site-mates
are all from its own sequence (`seq_mates = all_same`, 403 views at 0.418) is a
newest-month view, and those views pull the newest-month number down. Holding `seq_mates`
fixed (`dimension = delta_months|seq_mates=…`):

| `seq_mates` | newest month [95% CI] (views) | older [95% CI] (views) |
|---|---|---|
| none | 0.645 [0.589, 0.705] (424) | 0.688 [0.647, 0.730] (733) |
| some | 0.486 [0.457, 0.517] (1,540) | 0.607 [0.552, 0.661] (526) |
| all_same | 0.418 [0.359, 0.478] (403) | — (0) |

Among views with no same-sequence mate, the age effect mostly goes away (the CIs overlap)
and the newest-month views read 0.645, not about 0.50. An age effect remains among views
with some same-sequence mates. That still fits a position offset that differs between
capture campaigns, but it is confounded with the same-sequence split, it is one cut read
after the fact, and it was not pre-registered.

### Should `pers:pitch` / `pers:roll` be applied? No ([Figure 4](figures/panoramax-bayonne/fig4_pose.png); `data/fig4_pose.csv`)

On the same site set at the 25 m cap, real-tilt members only:

| convention | 0.55: median / p90 (m) | 0.30: median / p90 (m) |
|---|---|---|
| **off (flat)** | **4.49 / 7.56** | **4.52 / 7.81** |
| +pitch +roll | 6.08 / 13.20 | 5.90 / 13.01 |
| +pitch −roll | 5.67 / 13.17 | 5.58 / 12.28 |
| −pitch +roll | 5.54 / 10.53 | 5.49 / 10.51 |
| −pitch −roll | 5.54 / 9.85 | 5.46 / 10.10 |
| +pitch, roll 0 | 5.31 / 11.21 | 5.27 / 11.21 |

These are 225 of 430 groups at 0.55 and 539 of 1,210 at 0.30 (97 of 176 and 249 of 523 at
10 m); groups a convention cannot place inside the cap are excluded. The text output is in
`data/pose_ablation_t{0.55,0.3}.txt`.

Every convention loosens the spread. On p90 the CIs do not overlap the flat arm at either
tier. Flat stays right for this source, as for GSV and the withheld Mapillary
road-relative default.

A hypothesis, not tested here: the GoPro Max levels the horizon in-camera, so the images
may already be gravity-rectified. Applying `pers:pitch`/`pers:roll` on top would then be a
double correction, which would explain why every sign hurts.

[Figure 7b](figures/panoramax-bayonne/fig7b_site_plan.png) shows ground-point scatter for
one site, with position, heading and range error mixed. It plots site 487 (7 views from 5
sequences), chosen by a fixed rule: ≥ 5 views, with a median leave-one-out closest to the
run's (3.45 m against 3.44 m). The 10 m version plotted site 183; site ids are per-run
serials, so the two are not the same site.

## 3. RampNet hand-off

Bundle: **`D:/Git/labeler-wt/bayonne-bundle/`** (outside every repo). A RampNet session
copies it to `../RampNet/benchmark/bayonne/`.

- 125 panos (seed 0, 30 m minimum spacing, tier 0.55): `top` 5, `random` 95, `empty` 25.
- 147 detections ≥ 0.55 to judge.
- Native-resolution fetch 125/125, reconcile **OK 1:1**.
- The bundle's records carry only ≥ 0.55 detections.
- It was sampled from the **10 m run** (step 10 ran before the densify) and is not
  re-sampled: every 10 m pano is also in the 5 m run, and a RampNet session may already
  hold it. The 5 m run's extra panos are simply not in the sample.

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
at 1.90 panos/s, unpaced (before [#161](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/161); see "Re-running it today" in §5).

**After the 5 m densify (§1.1) the archive covers 28,524 of 50,409 panos.** It holds the 10 m
run's `results.jsonl`, not the 5 m one. Archiving the 21,885 added panos (about **6 h** at
today's 1 picture/s; `export_benchmark.py` is resumable) is **a decision for Jon** and has
not been done.

## 5. Replication

**Inputs** (`data/inputs.csv`, written by the `data` step):

| file | sha256 |
|---|---|
| `runs/bayonne/results.jsonl` (5 m run) | `cb03768cc4cefd0bcb9792e18ff339ad798e43fd3e560deee431387796d6eeef` |
| `runs/richmond/results.jsonl` | `109e7645ebf5ab982d2cc1388b50e837f6d622a4ff14752c1895b194a5c0d88c` |
| `runs/clovis/results.jsonl` | `f6a896f19f4c7036186b201bbd2a1bfa4d6a20f34c14f5586a72da936475c15d` |
| `runs/laurens/results.jsonl` | `16c5a348b739274bf8e7623b63551da5245547d0ca4e5a52956fd2413b4fe213` |
| `runs/annapolis/results.jsonl` | `f1b228f742dc95909067cc3ba9b045f1acf76617747b6e21eb9e7df58335b9dd` |
| `runs/morgantown/results.jsonl` | `7dbf24e03574c31b0a3011490177a5d51dc20e30badf7fdd35da443e870b98bc` |
| `runs/bayonne/osm_streets.json` (Overpass, cached by step 2) | `60d212b73d4cd7a7757b16f3f90e189d829958bbcc025f6d774ae165a11069b7` |

**Regenerating the 10 m evidence.** The table above is the 5 m run, and steps 4, 5, 6 (its
`census_5m/` half) and 7 read it from `runs/bayonne`. The 10 m readings are a different
input: `data/census/`, `data/skips.csv`, the example thumbnails, the bundle, and the 10 m
column of §2's table, whose CSVs are `data/` as committed at
[`7a7a1a6`](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/commit/7a7a1a6)
(before the 5 m re-run replaced them; their `inputs.csv` carries the 10 m hash). The 10 m
file is the 5 m file's byte prefix, because the densify appended to it: the first 28,524
lines of `runs/bayonne/results.jsonl` (`cb03768c…6eeef`, 50,409 lines) hash to
`4f38ff528d4445b21eb02e99427e0ba3c88dfb5c940cc4702a283182e379b4b3`. Build a run root of its
own from that prefix, beside the cached Overpass pull, the area and the five comparator runs
(each checked against the table above), and point every step that reads the 10 m run at it:

```bash
R10=/path/to/bayonne10m                     # any scratch dir; not runs/
mkdir -p $R10/bayonne
head -n 28524 runs/bayonne/results.jsonl > $R10/bayonne/results.jsonl
sha256sum $R10/bayonne/results.jsonl         # 4f38ff528d4445b21eb02e99427e0ba3c88dfb5c940cc4702a283182e379b4b3
cp runs/bayonne/osm_streets.json runs/bayonne/area.geojson $R10/bayonne/
# step 8 only: the 10 m cache as it stood when skips.csv was made (28,524 panos + 110 skips).
# Its first 28,631 lines are that cache after the 3 retried ids left it on 2026-10-05 (§1);
# the densify appended after them. Do NOT copy the live cache: against the prefix it holds
# 22,004 unprocessed ids, which step 8 would fetch one by one into a 5 m skips.csv.
{ head -n 28631 runs/bayonne/already_processed.txt
  printf '%s\n' 139d37d3-9c50-4202-964e-c50c92712a49 1f8be062-1ae4-4f68-a5d4-07f3ac494b86 \
                2c902fc4-669d-4b9c-9e9e-1eafb68a94f7; } > $R10/bayonne/already_processed.txt
for c in richmond clovis laurens annapolis morgantown; do
    mkdir -p $R10/$c && cp runs/$c/results.jsonl $R10/$c/; done
```

Checked 2026-11-03 from that root (Windows, the analysis environment below), against the
10 m files as committed then (`7a7a1a6`'s `data/`, before this PR replaced them with the 5 m
run's):

- step 4 reproduces both 10 m `pose_ablation_t*.txt`;
- step 6 reproduces the five `data/census/` CSVs. `census.md` gains a per-rig table that
  `run_census.py` added after this census was committed, and the script writes CRLF on
  Windows, so compare after LF normalisation;
- step 7 (a full `data` run, about 50 min) rewrote `inputs.csv` and every `fig*.csv` raw
  byte-identical to the 10 m ones. That check predates the sequence-bootstrap columns and
  `cuts.csv`; today the same run adds those columns and part. Run it in a scratch checkout
  and compare with `git show 7a7a1a6:<path>`: in this tree it overwrites the committed 5 m
  CSVs.
- step 8's rebuilt cache is set-equal to the 10 m cache's backup of 2026-10-05, and
  `skips` run on it with the Panoramax calls stubbed out asks for exactly the 110 ids of
  `data/skips.csv`. Step 8 itself was not re-run against Panoramax.

Step 5 was not re-run from the prefix.

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
| 1 | `python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --scan-only` | net | 6 s (≥ ~11 s today) |
| 2 | `python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --reuse-scan --thin-spacing 10` (ends with the position check; Overpass) | GPU + net | 6.7 h (≥ ~8.0 h today) |
| 2b | on Hyak `ckpt-g2` (L40S), after the hand rebind in `c096987`: `python main.py example_geojson/bayonne.geojson --name bayonne --source panoramax --reuse-scan --thin-spacing 5` (§1.1; resumes step 2's run, ends with the position check) | GPU + net | 2.4 h (≥ ~6.1 h today) |
| 3 | `python scripts/position_check.py runs/bayonne --report` (only to redo step 2b's check) | net (cached after) | < 1 min |
| 4 | `python scripts/fuse_sites.py runs/bayonne --pose-ablation --min-confidence 0.55 > docs/figures/panoramax-bayonne/data/pose_ablation_t0.55.txt` (and `0.3` → `…_t0.3.txt`) | — | 1 s |
| 5 | `python scripts/reprojection_residual.py bayonne richmond clovis laurens annapolis morgantown --camera-height-m 2.6 --refuse --fit-sigma-pitch 0.269 --benchmark-root /nonexistent [--min-confidence 0.3] --publish docs/figures/panoramax-bayonne/data/reprojection[_t0.3]` | — | ~1.5 min per tier |
| 6 | `python scripts/run_census.py runs/bayonne --out docs/figures/panoramax-bayonne/data/census_5m --band-y 0.79` (5 m); `python scripts/run_census.py $R10/bayonne --out docs/figures/panoramax-bayonne/data/census --band-y 0.79` (10 m, §1) | — | 10 s each |
| 7 | `python scripts/panoramax_bayonne_figures.py data` → `data/inputs.csv`, `data/fig*.csv`, `data/cuts.csv` (bootstraps, seed 57; `--only` regenerates parts) | — | ~2 h on the 5 m run (the two `sigma_gps` bootstraps, site and sequence; measured on a shared machine) |
| 8 | `python scripts/panoramax_bayonne_figures.py skips --run-root $R10` → `data/skips.csv` (10 m: the 110 ids of the rebuilt cache above, minus the 10 m results; each reason is re-read from Panoramax today, so the output is a snapshot and may differ from the committed file) | net | 1 min |
| 9 | `python scripts/panoramax_bayonne_figures.py examples --run-root $R10` → `data/examples/` (7a thumbnails; 7c, the first cropped-MAX2 skip by pano id; 10 m) | net | 1 min |
| 10 | `python scripts/export_benchmark.py $R10/bayonne/results.jsonl --bundle D:/Git/labeler-wt/bayonne-bundle --sample 100 --empty-sample 25` (10 m) | net | 1.5 min |
| 11 | `python scripts/panoramax_bayonne_figures.py crops` → `data/examples/crops/` + `crops.csv` (from the bundle) | — | 10 s |
| 12 | `python scripts/panoramax_bayonne_figures.py figures` → `fig*.png` / `.jpg` (+ `.svg` for figures 1-5 and 7b) | — | 30 s |
| 13 | on makelab2, `bayonne_archive.sh`: `export_benchmark.py runs/bayonne/results.jsonl --out runs/bayonne/panos` (the 10 m file, then) | net | 4.2 h (≥ ~7.9 h today) |

- **Steps 2 / 2b.** The run directory is now bound to **5 m** (top-level `thin_spacing_m`,
  changed by hand from the 10 m backfill in `c096987`, recorded in `thin_spacing_changed`), so
  any resume must pass `--thin-spacing 5` (the default) and a 10 m resume is refused.
- **Which run each step read.** Steps 4, 5 and 7 were re-run on 2026-10-07 on the 5 m
  run's `results.jsonl` (`data/inputs.csv`), with `--run-root` pointed at a directory
  holding the 5 m Bayonne run and the five comparators (steps 4 and 5 take the run
  directory and `--run-root`/`--out-root` the same way). Step 7's `fig1`, `fig2a`, `fig2b`
  and `cuts` outputs were regenerated from the same inputs on 2026-11-03 by the PR #162
  review fix (see the next bullet). Step 6's 5 m output is
  `data/census_5m/`, which Figure 6 draws; `data/census/` is the 10 m census of §1. Steps
  8-11 (skip list, example thumbnails, bundle, crops) need the network and stay on the
  10 m run, so steps 6 (its `data/census/` half), 8, 9 and 10 read `$R10`, the 10 m prefix
  root above; pointed at `runs/bayonne` they read the 5 m file and give different numbers.
  Any partial `data --only <part>` run (with or without `inputs`, `cuts` included) refuses
  while a run file's sha256 differs from `data/inputs.csv`, so one panel cannot be
  refreshed from one run beside parts from the other; re-run all parts
  (`--allow-changed-inputs` overrides, for scratch comparisons only).
- **Step 7 is byte-reproducible.** It was run twice on the 5 m inputs and every CSV it
  writes hashed identical. The PR #162 review fix added the sequence-bootstrap columns and
  `cuts.csv`; re-running every part then left every previously committed CSV
  byte-identical (`inputs`, `fig3`, `pairs`, `fig4`-`fig7b`), and in `fig1`, `fig2a` and
  `fig2b` every pre-existing column is unchanged, cell for cell (new columns appended).
- **Step 5's** `bayonne_report.md` copies under `data/reprojection*/` are that step's
  per-city `report.md`, copied by hand.
- **Step 12** reads only the committed `data/`. It is byte-reproducible for the listed
  versions: it was run twice and every PNG, JPEG and SVG hashed identical. SVGs are written
  with LF line endings, and `.gitattributes` keeps them LF.
- **Figure formats.** Figures 1-5 and 7b are PNG + SVG. The map (6) is PNG with a
  deterministic 256-colour palette. The photo figures (7a, 7c, 7d) are quality-92 JPEGs
  (4:4:4) rendered from the same 200 dpi raster: a palette posterised the photos, and an
  SVG would embed them a second time.

**Re-running it today.** Every time in the steps table outside the parentheses, and every
rate in this doc (1.18 and 2.50 panos/s for detection, 1.90 for the archive), was measured
before [#161](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/161) /
[PR #173](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/173), unpaced.
Panoramax's maintainers asked for about one full-resolution picture per second, and since
PR #173 `sources/panoramax.py` paces every request: `picture` 1/s, `metadata` 1/s (batched
when main.py hints the order), `tile` 5/s, with the pool capped at 4 threads. The download
then binds, so the floors at 1 picture/s are:

| what | panos | floor today |
|---|---|---|
| step 1, the scan | 55 tiles | ~11 s |
| step 2, the 10 m run | 28,634 | ~8.0 h |
| step 2b, the densify resume | 21,897 | ~6.1 h |
| step 13, the 10 m archive (one item GET + one picture GET per pano, each on its own 1/s limiter; `export_benchmark.py` gives no order hint) | 28,524 | ~7.9 h |
| archiving the 21,885 added panos (§4) | 21,885 | ~6.1 h |
| the whole 5 m set from scratch | 50,528 | ~14 h |

The limits are per process, so never run the archive and a detection run against the public
instances at the same time: two processes get 1/s each. Never set
`PANORAMAX_PICTURES_PER_SECOND` / `_METADATA_` / `_TILES_` to 0 (unlimited) except against a
private mirror.

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
| 73,161 / 28,634 / 28,524 / 110 / 0 | `runs/bayonne/manifest.json` (`runs[2]`, the 10 m pass; `runs[-1]` is now the 5 m resume) and `run.log` | `panos_found_in_area`, `processed`, `skipped`, `failed`; 73,161 from `scan.log` |
| 6.7 h, 1.18 panos/s, 24,062 s | `run.log` | last `detector:` line; `started_at`/`finished_at` in the manifest's `runs[2]` |
| 21.1 h estimate, 50,528 | `scan.log` | — |
| 104 / 3 / 3 skips | `data/skips.csv` | `reason` |
| rig / year / pose / producer shares | `data/census/{rigs,years,pose,producers}.csv` | `share` |
| 0.135 / 0.351 per pano; 2 of 10,011, 0 of 3,861 in band; 9 / 1 on rig | `data/census/detections.csv` | `detections_per_pano`, `in_band`, `on_rig` |
| detections per pano, six runs | `data/fig5_detections.csv` | `per_pano` by `run`, `tier` |
| 1.54 m, IQR 0.70-3.13, 4,027 (10 m) | `runs/bayonne/position_check.json` at commit `c096987` | `fields.submitted.cross_track`, `panos_not_near_a_street` |
| 5 m: 73,161 / 50,528 / 21,885 / 12 / 0; reused scan, 137.8 h old; started/finished | `runs/bayonne/manifest.json` (`runs[3]`) and `run_densify_41473672.log` (untracked) | `panos_before_thinning`, `panos_found_in_area`, `processed`, `skipped`, `failed`, `scan`, `scan_age_hours`, `thin_spacing_m`, `started_at`/`finished_at` |
| 5 m: 28,631 cached / 21,897 new; 2.50 panos/s, 8,726 s | `run_densify_41473672.log` | `Processing Summary`; last `detector:` line |
| 5 m: 10 m → 5 m rebind | `runs/bayonne/manifest.json` | `thin_spacing_m`, `thin_spacing_changed` |
| 5 m: 50,409 panos, 0 flagged, 1.58 m, IQR 0.72-3.23, 6,113 | `runs/bayonne/position_check.json` (and `manifest.json` `position_check`) | `panos`, `flagged_sequences`, `fields.submitted.cross_track`, `panos_not_near_a_street` |
| 5 m: 0.381 / 0.148 per pano; 15 / 1 on rig; 2 / 0 in band; shares | `data/census_5m/{detections,rigs,years,pose,producers}.csv` | `detections_per_pano`, `on_rig`, `in_band`, `share` |
| chi²/dof, median form, m p50 (+ CIs), incl. rig-matched rows | `data/fig1_verdict.csv` | `chi2_dof`, `chi2_dof_median`, `m_p50`, `*_lo`/`*_hi` by `run`, `subset` |
| sequence-bootstrap CIs (Bayonne [0.534, 0.578], Richmond GoPro Max [0.454, 0.559], 14 sequences) | `data/fig1_verdict.csv` | `*_seq_lo`/`*_seq_hi`, `n_sequences` by `run`, `subset` |
| 0.586 / 3.52 at 0.30 | `data/reprojection_t0.3/gtfree_summary.csv` | `bayonne` row |
| `sigma_pitch` > 15°, `sigma_gps` 4.57 m at target 0.269 | `data/reprojection/gtfree_summary.csv` | `sigma_pitch_deg_at_chi2_target`, `sigma_gps_m_at_chi2_target` |
| `sigma_gps` at chi²/dof 1 (2.10, 1.11, 1.81, 1.74, 0.80, …) | `data/fig2b_sigma_gps.csv` | `sigma_gps_chi2_1_m`, `lo`, `hi` by `run`, `subset` |
| `sigma_gps` sequence-bootstrap CIs (Richmond GoPro Max 1.55-2.09, 12 sequences) | `data/fig2b_sigma_gps.csv` | `seq_lo`, `seq_hi`, `n_sequences` |
| along/cross CIs by range bin, both bootstraps | `data/fig2a_range.csv` | `along_p50_lo`/`_hi`, `cross_p50_lo`/`_hi`, `*_seq_lo`/`*_seq_hi` |
| by-range excess (total, along, cross), chi²/dof by range | `data/fig2a_range.csv` | `m_p50`, `along_p50`, `cross_p50`, `chi2_dof` by `run`, `subset`, `range_bin` |
| k 1.109 vs 0.930 / 1.055 / 1.091 / 1.114 / 1.132 | `data/reprojection/range_slope.csv` | `implied_range_scale_k`, `capture_year = all` |
| same-sequence split; views / sequences per site; site size per group | `data/fig3_seqsplit.csv` | `chi2_dof`, `chi2_dof_lo`/`_hi`, `n_views`, `views_per_site_p50`; `site_makeup` rows |
| same-sequence pair shares 0.19 … 0.49 | `data/fig3_pair_share.csv` | `same_sequence_share` |
| pose ablation medians / p90s | `data/fig4_pose.csv` (and `pose_ablation_t*.txt`) | `median_m`, `p90_m` by `tier`, `subset`, `convention` |
| chi²/dof by pose group | `data/reprojection/gtfree_breakdown.csv` | `dimension = pose_group`, `city = bayonne` |
| chi²/dof by site size (0.549 / 0.535 / 0.579) and capture age (0.503 … 0.74), with CIs; capture age within each `seq_mates` stratum (0.645 vs 0.688, 0.486 vs 0.607); 6+ make-up 7 / 4 | `data/cuts.csv` | `chi2_dof`, `chi2_dof_lo`/`_hi`, `chi2_dof_seq_lo`/`_hi`, `site_views_p50`, `site_sequences_p50` by `dimension`, `bucket` |
| every 10 m reading in §2's comparison table | `docs/figures/panoramax-bayonne/data/` at commit `9c6fe6c` (same files and columns as the 5 m rows) | — |
| Figure 7b site 487, 7 views, 3.45 vs 3.44 m | `data/fig7b_site.csv` | `site_id`, `kind = view` rows, `site_median_loo_m`, `run_median_loo_m` |
| logo band y 0.791 / 52.4° | measured on two panos; drawn in Figure 7a | `LOGO_BAND_Y` in `scripts/panoramax_bayonne_figures.py` |
| Figure 7d crops, ranges, confidences | `data/examples/crops.csv` | `confidence`, `range_m`, `privacy_skip` |
| bundle strata, 147 detections (10 m run) | `D:/Git/labeler-wt/bayonne-bundle/sample.json`, `records.jsonl` | `groups` |
| archive 28,524, rc 0 | makelab2 `runs/bayonne/index.csv`, `runs/bayonne_archive.done` | — |

## Figures

| # | question it answers | file |
|---|---|---|
| 1 | Is Bayonne outside the Richmond-anchored band? Barely: 3% over the bar, with the bar inside its CI; 1.11x Richmond's GoPro Max views but not distinguishable from them (site-bootstrap CIs just separate; sequence-bootstrap CIs, the pale bars, overlap). | [fig1_verdict](figures/panoramax-bayonne/fig1_verdict.png) |
| 2 | Tilt or position? Rig-matched (re-solved), the excess is 0.6-0.8 m across the ray. Scatter is 1.9x Richmond's and 1.2x its and Laurens' GoPro Max views (against Richmond's GoPro Max views the sequence-bootstrap CIs, the pale bars, overlap). | [fig2_position](figures/panoramax-bayonne/fig2_position.png) |
| 3 | Does same-sequence error cancel? Not established: the anchor reads the opposite way. | [fig3_seqsplit](figures/panoramax-bayonne/fig3_seqsplit.png) |
| 4 | Apply the reported pose? No: every sign convention loosens. | [fig4_pose](figures/panoramax-bayonne/fig4_pose.png) |
| 5 | Is the detection rate anomalous? No: it is inside the Mapillary range. | [fig5_detections](figures/panoramax-bayonne/fig5_detections.png) |
| 6 | What did the 5 m run cover? Census and map. | [fig6_census](figures/panoramax-bayonne/fig6_census.png) |
| 7a-d | What do the panos, a site, a MAX2 skip and the detections themselves look like? (7b is the 5 m run; 7a, 7c and 7d were drawn from the 10 m run.) | [7a](figures/panoramax-bayonne/fig7a_contact_sheet.jpg), [7b](figures/panoramax-bayonne/fig7b_site_plan.png), [7c](figures/panoramax-bayonne/fig7c_max2_skip.jpg), [7d](figures/panoramax-bayonne/fig7d_crops.jpg) |

**Colour key across figures:** blue is Bayonne, orange is a GoPro Max population, and
grey is other or mixed rigs.

**Credits:**
- Figures 7a, 7c and 7d: imagery via Panoramax, with producer and licence per panel
  (municipal: © sig_bayonne, Licence Ouverte / Etalab 2.0; 7c: © lplm, CC BY-SA 4.0).
- Figure 6: © OpenStreetMap contributors (ODbL).

Alt text, in order:

1. Dot plot of chi²/dof and median residual for eight rows. Bayonne at 0.56 sits just
   above the shaded 0.13-0.54 band, its CI reaching into it. Richmond's GoPro Max views (0.50) and Laurens (0.43)
   sit just inside it, Morgantown (GoPro Max) at 0.18, and other rigs near 0.2. A pale
   bar behind each pooled CI is the sequence-bootstrap CI; Richmond's GoPro Max bar
   reaches past Bayonne's point.
2. Three line panels by range: Bayonne, all of Richmond, and Richmond's GoPro Max views
   re-solved on their own. Bayonne tracks Richmond's GoPro Max along the ray but sits
   0.6-0.8 m above it across the ray. A dot plot shows calibrated `sigma_gps` from 0.74 m
   (Annapolis) to 2.10 m (Bayonne), with the GoPro Max populations at 0.80, 1.74 and 1.81 m, all left of the
   3 m line. Pale bars behind the CIs are the sequence-bootstrap CIs; Richmond's GoPro
   Max bar reaches Bayonne's.
3. Paired dots per run: chi²/dof for held-out views whose site-mates are all from the
   same sequence vs none. Four runs are lower with same-sequence mates, Laurens too but
   with overlapping CIs, and Richmond is higher.
4. Median and p90 within-site distance for six pose conventions at two tiers. The flat
   arm is lowest everywhere.
5. Detections per pano for six runs. Bayonne (0.148) sits between Clovis (0.123) and
   Laurens (0.158).
6. Bar charts of capture year, rig and pose availability, and a map of 50,409 panos
   coloured by distance to the nearest OSM street, with a 1 km scale bar.
7. Four example figures:
   - (a) Eight panoramas with detections, the nadir-mask line and the shaded logo band.
   - (b) Plan view of seven cameras, rays and ground points around a fused site.
   - (c) A MAX2 image that is shorter than its declared 2:1 frame.
   - (d) Twelve zoomed crops of detections, mostly lowered kerbs at pedestrian crossings.

## 6. What remains

- **Ground truth.** Review the bundle in RampNet, then run `eval_sites.py bayonne`.
- **Position scatter.** Re-measured on the 5 m run (§2): Bayonne is now at the edge of
  the ADEQUATE band. A second Panoramax city, or ground truth, decides whether Panoramax
  gets its own `ErrorModel`.
- **Archive** the 21,885 panos the 5 m densify added (§4), if Jon decides to.
- **Follow-ups:** [#126](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/126)
  (record and bind `--thin-spacing`) and
  [#127](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127) (the
  decode catch and truncation).
