# GSV partial pose: does a fraction of the rig tilt improve the flat raycast? (issue #116)

`scripts/gsv_partial_pose.py` runs the pre-registered test that #113 called for. The design
and rule were posted on
[#116](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/116#issuecomment-5917738112)
before any scoring run, and the script was committed at the same time.

## Verdict

**FAIL under the pre-registered rule, on one clause. Production does not change:** GSV
still raycasts flat, and `--apply-pose` gains no `partial` mode.

- **Clause (i) passes (self-agreement).** At the production height (`auto`), on held-out
  panos, with each arm re-associated under its own pose, a partial pose cuts the median
  within-site pair distance by 5.7-10.7% and the p90 by 4.7-8.5%. This holds in all four
  gated cities, for both candidates (per-city fit and leave-one-city-out fit). It also beats
  the magnitude-matched shuffled control everywhere, by a similar margin. The mirrored sign
  is worse than flat everywhere.
- **Clause (iv) passes (external referee).** Against the Bend and Gainesville curb-ramp
  inventories, the partial pose is slightly better than flat in both cities: Bend median
  -0.02 m and p90 -0.05 m, Gainesville median -0.04 to -0.06 m and p90 -0.07 to -0.10 m.
- **Clause (iii) passes (unplaceable GT marks).**
- **Clause (ii) fails in Bend.** Off-pool recall at 2.5 m falls 1.3 points for both
  candidates. That is **2 of 157 ramps lost and 0 gained** (exact paired p = 0.5). At the
  half split, the 1.0-point threshold is smaller than two ramps in every gated city, so the
  clause cannot tell a two-ramp loss from noise. The Bend loss is also the smallest of any
  rotated arm there: the shuffled controls lose 4 and 6 ramps, the full pose 10.
- **At 2.6 m (reported, not deciding), it also fails.** Paterson's p90 ties flat (+0.001 /
  +0.003 m). Gainesville loses 6 / 3 ramps of 97, which is where 2.6 m is the wrong height
  for most of the city (below).
- **An exploratory reading passes all four clauses at both heights.** It was added after
  the verdict and gates nothing. It uses the leave-one-city-out coefficients, which never
  saw the city scored, applied to the whole run (about twice the GT pool). Bend's recall
  there is unchanged: 2 ramps lost and 2 gained of 298.

**What this settles.** The #113 mechanism now holds out of sample and against a control. A
fraction of each pano's own stored tilt, about 0.18 of the pitch term and 0.38 of the roll
term, is a real placement error: the magnitude-matched shuffle, which applies corrections
of the same size to the wrong panos, does worse than flat. That is the control #42's
road-relative correction failed. **What it does not settle** is whether the correction is
safe for recall. The pre-registered survivorship clause was too fine for the half-split
pool. That question needs a confirmatory run whose recall clause is sized to its pool (see
"What remains").

## Background

GSV equirectangulars are in the capture rig's frame, not gravity-rectified (#113,
`docs/` corrections in PR #115). Rotating rays by the full stored pose loosens multi-view
agreement (#27's `--pose-ablation`, #52). #113 explained why. The car rides the road, so
the local ground shares most of the rig's tilt, and only a fraction of it is a placement
error. A triangulation regression read 0.14-0.25 of the pitch term and 0.41-0.54 of the
roll term, and a frozen-at-flat `(-0.25 pitch, +0.5 roll)` arm tightened Paterson, Bend and
São Paulo. Those arms were in-sample, frozen at flat, and had no shuffled control.

## Design (as pre-registered; the posted comment is authoritative)

- **Cities:** paterson, bend, gainesville, sao_paulo, laurens_gsv. Operational tier 0.30,
  rig mask on, 25 m cap. `auto` heights are resolved on the full run before the split.
- **Split:** a pano is TEST when the low bit of `sha256("116:<pano_id>")` is 1, else
  TRAIN. The two halves are separate runs: TRAIN fits, TEST is scored.
- **Fit (#113's estimator):** fuse TRAIN flat. For each pair of operational members of a
  site whose flat bearings cross at >= 30 deg, triangulate by bearing only (3-30 m). For
  each member, compute the residual as observed elevation minus expected elevation at that
  range and the pano's height. Regress it by OLS on `pitch cos b` and `roll sin b`
  (streetlevel's stored sign, roll wrapped to [-180, 180), b = azimuth from the centre
  column), with an intercept and pano-clustered SEs. The slopes are the leaked fractions
  `(k_pitch, k_roll)`, and the arm feeds `geo._world_ray` the pose
  `(-k_pitch x pitch, +k_roll x roll)`. The fit is run per city, leave-one-city-out
  (LOCO), and pooled.
- **Arms (TEST):**
  - `off`
  - `partial` (the city's own fit)
  - `partial-loco`
  - `partial-shuffled` and `partial-loco-shuffled`: each pano gets another TEST pano's
    `(pitch, roll)` pair, permuted within |tilt| buckets [0, 0.5, 1, 2, 3, 5, inf) deg,
    seed 116
  - `full`
  - `partial-mirror`
  - `fixed-113`: descriptive only
- **Primary frame, `reassoc`:** each arm re-fuses TEST under its own pose. The unit is the
  pair of operational members (different panos) co-associated under every decision arm
  (off, both candidates, both shuffles), measured between each arm's own placements.
  Secondary frame `frozen@off`: #113's design on TEST.
- **External referee:** `inventory_oracle.score_frozen` on TEST, with membership and the
  5 m pool anchored on `off`.
- **Survivorship:** RampNet GT on TEST-half judged panos. It covers off-pool recall
  against each arm's own sites (#42's definition), unplaceable GT marks, and unplaceable
  operational members.
- **Rule (at `auto`, both candidates):** (i) in every city with >= 500 common pairs,
  median and p90 strictly below off and below the candidate's own shuffle; (ii) off-pool
  recall@2.5 m drop <= 1.0 pt in every GT city; (iii) extra unplaceable GT marks <= 5% of
  the off pool; (iv) inventory median and p90 no worse than off by more than 0.10 m. A
  pass would wire `--apply-pose partial` as opt-in, and the default switch would be a
  separate PR.

laurens_gsv has 300 common pairs on its test half, below the 500 bar, so (i) reports it
without gating it. Its GT still enters (ii) and (iii).

## Results

Every table below is regenerated into `runs/_pooled/partial_pose/summary.md` by
`gsv_partial_pose.py verdict`. The per-city `runs/<city>/partial_pose/report.md` has
all of them, including mean distances and the 2.6 m tables.

### Coefficients (tier 0.30, TRAIN halves)

| height | fit | k_pitch (SE) | k_roll (SE) | intercept (deg) |
|---|---|---|---|---|
| auto | paterson | 0.196 (0.013) | 0.345 (0.020) | -0.18 |
| auto | bend | 0.126 (0.011) | 0.399 (0.018) | -0.14 |
| auto | gainesville | 0.294 (0.025) | 0.541 (0.024) | +0.01 |
| auto | sao_paulo | 0.192 (0.020) | 0.312 (0.028) | -0.27 |
| auto | laurens_gsv | 0.424 (0.099) | 0.356 (0.090) | -0.49 |
| auto | **pooled** | **0.183 (0.008)** | **0.382 (0.012)** | -0.15 |
| 2.6 | paterson | 0.155 (0.016) | 0.392 (0.025) | +0.96 |
| 2.6 | bend | 0.112 (0.011) | 0.410 (0.018) | +0.32 |
| 2.6 | gainesville | 0.146 (0.036) | 0.571 (0.038) | +1.32 |
| 2.6 | sao_paulo | 0.191 (0.018) | 0.344 (0.028) | +0.10 |
| 2.6 | laurens_gsv | 0.463 (0.097) | 0.389 (0.088) | -0.07 |
| 2.6 | **pooled** | **0.147 (0.009)** | **0.404 (0.013)** | +0.53 |

The LOCO fits (in `coefficients.csv`) sit within 0.03 of the pooled values at both
heights.

- **Every slope is positive and between 0 and 1**, which is #113's sign. The roll fraction
  is about twice the pitch fraction, as #113 found.
- **The coefficients move with the height, mostly on pitch.** Pooled k_pitch goes from
  0.183 at `auto` to 0.147 at 2.6 m, about 4 SE; k_roll moves only 0.382 to 0.404. The
  shift is concentrated in Gainesville, whose pitch fraction halves at 2.6 m (0.294 to
  0.146) while its intercept jumps to +1.3 deg. Gainesville is 64% the 2026 rig, whose
  true height is about 2.0 m (#79), so at 2.6 m the regression absorbs a large range error,
  and part of it is correlated with pitch. The `auto` fit is the one that describes the
  pose, because it runs at the height that ships.
- Bend's 0.30 and 0.55 fits are identical because Bend's run predates the storage floor:
  its 0.30 tier is its 0.55 tier.

![Two dot plots of the fitted fractions k_pitch and k_roll with 95% intervals, one row per
own-city fit, leave-one-city-out fit and pooled fit, blue for the auto height and orange for
2.6 m. Pitch fractions cluster near 0.15-0.2 and roll fractions near 0.35-0.55, left of
#113's fixed arm at 0.25 and 0.5. Laurens GSV has the widest intervals; Gainesville's pitch
fraction halves at 2.6 m.](figures/gsv-partial-pose/fig2_coefficients.png)

*Figure 2. The fitted fractions, with 95% intervals. The leave-one-city-out and pooled fits
agree within about 0.03, so the partial pose is not a per-city artefact. Gainesville's pitch
fraction is the one that moves with the camera height.*

### Primary: within-site pair distance, re-associated, `auto` (median / p90, m)

| city | pairs | off | partial | partial-loco | partial-shuffled | loco-shuffled | full | partial-mirror |
|---|---|---|---|---|---|---|---|---|
| paterson | 12,649 | 1.736 / 3.883 | **1.587 / 3.604** | **1.588 / 3.604** | 1.758 / 4.030 | 1.755 / 4.031 | 1.840 / 4.624 | 1.955 / 4.561 |
| bend | 20,233 | 1.516 / 3.728 | **1.429 / 3.553** | **1.415 / 3.546** | 1.549 / 3.816 | 1.562 / 3.873 | 1.634 / 4.353 | 1.673 / 4.124 |
| gainesville | 8,534 | 1.913 / 4.178 | **1.708 / 3.824** | **1.773 / 3.840** | 1.937 / 4.318 | 1.924 / 4.123 | 1.706 / 4.273 | 2.307 / 5.194 |
| sao_paulo | 12,621 | 1.945 / 4.100 | **1.828 / 3.866** | **1.816 / 3.862** | 1.949 / 4.120 | 1.969 / 4.168 | 1.926 / 4.689 | 2.107 / 4.546 |
| laurens_gsv (not gated) | 300 | 1.854 / 4.039 | 1.587 / 3.657 | 1.587 / 3.658 | 1.712 / 4.011 | 1.782 / 3.878 | 1.557 / 3.660 | 2.038 / 4.563 |

- **The shuffle is the control that matters, and it loses.** The shuffle gives each pano a
  correction of the same size, and the same mean tilt per bucket, taken from another pano.
  It is no better than flat on the median in any gated city, and worse on the p90 in 7 of
  8 city/arm cells (the exception is Gainesville's LOCO shuffle, 4.123 against 4.178). The
  partial arms beat their shuffles by 0.12-0.23 m on the median and 0.25-0.49 m on the
  p90. So the gain comes from each pano's own tilt, not from a systematic offset. This is
  the control #42's road-relative correction failed.
- **The sign is confirmed again.** The mirror is worse than flat everywhere, and much worse
  than the partial arm.
- **The full pose overshoots on the tail**, as in #113. It is worse than flat on the p90
  in every gated city, though in Gainesville its median alone matches the partial arm.
- **Re-association costs nothing.** The partial arms form as many or more multi-view sites
  than off (Paterson 4,284 against 4,217), and their count in the common set is within 10
  of off's. Only `full` and `partial-mirror` lose sites (full: about 300 in Paterson, 420
  in Bend). The frozen-at-off frame gives the same ranking and similar gains
  (`summary.md`).

![Two panels, median and p90 within-site pair distance, each showing every arm's change
from off in metres for five cities. In every city both partial arms sit left of zero
(members agree better), the two shuffled controls sit at or right of zero, and the mirror
sits well right of zero. The full pose is right of zero on the p90 everywhere except
laurens_gsv.](figures/gsv-partial-pose/fig1_pair_distance.png)

*Figure 1. Change from off in held-out within-site pair distance, at `auto`. Left of zero
means a site's members agree better. The partial arms move left in every city; their
magnitude-matched shuffles do not, and the mirror moves right. laurens_gsv has 300 pairs,
under the 500-pair bar, so it is shown but not gated.*

![Plan view of one Gainesville site in two panels. Left: five cameras up to 18 m away with
their rays converging on the site. Right: a 4 m zoom around the city inventory ramp, with
each member's placement under off (grey) and partial (blue) joined by an arrow. The fused
off and partial sites both sit about 0.4 m from the inventory ramp.](figures/gsv-partial-pose/fig3_site_example.png)

*Figure 3. A typical site, not a showcase. It was chosen by a rule fixed before looking: of
the 898 held-out Gainesville sites with at least three members on three panos and an
inventory ramp within 2.5 m, the one whose partial/off member-spread ratio is the median
(0.87). The partial pose moves each placement by a few decimetres along its ray and the
members close up. The fused site barely moves (0.37 m to 0.38 m from the inventory ramp),
which is why the inventory referee sees only 0.02-0.06 m. None of the five member panos is
in the local RampNet bundle, so the figure has no image crops.*

### Survivorship (RampNet GT, TEST-half judged panos, `auto`)

| city | off pool | off R@2.5 | partial (lost/gained) | partial-loco (lost/gained) | shuffled (lost/gained) | full (lost/gained) |
|---|---|---|---|---|---|---|
| paterson | 185 | 0.876 | 0.892 (1/4) | 0.892 (1/4) | 0.881 (2/3) | 0.865 (12/10) |
| bend | 157 | 0.898 | **0.885 (2/0)** | **0.885 (2/0)** | 0.873 (4/0) | 0.847 (10/2) |
| gainesville | 102 | 0.863 | 0.882 (1/3) | 0.853 (1/0) | 0.863 (3/3) | 0.892 (3/6) |
| sao_paulo | 109 | 0.908 | 0.917 (0/1) | 0.917 (0/1) | 0.908 (0/0) | 0.890 (6/4) |
| laurens_gsv | 104 | 0.827 | 0.827 (1/1) | 0.817 (3/2) | 0.817 (1/0) | 0.837 (4/5) |

The `lost/gained` pairs are ramps that off recalls and the arm does not, and the reverse.
They were added after the verdict as a descriptive column; the gated recall values are
byte-identical to the verdict run.

- **Unplaceable GT marks** (clause iii): every candidate is within one or two of off, and
  mostly below it.
- **Unplaceable operational members:** at `auto` the partial arm places more members than
  off inside 25 m (Paterson 304 more, Gainesville 341). At 2.6 m it places up to 77 fewer.
  `full` swings by hundreds either way.
- **The failed clause.** Bend loses 2 ramps and gains none, under both candidates. Every
  rotated arm in Bend loses some recall, and the partial arm loses the least. The GT pool
  is grouped under `off` (#42's design), which favours `off`. Two ramps is below what the
  clause can resolve at this pool size: 1.0 pt of 157 is 1.6 ramps.

![Three scatter plots of each Bend GT ramp's distance to its nearest site, off on the x
axis against the candidate on the y axis, with dashed lines at the 2.5 m match radius.
Almost every point lies on the diagonal. In the two held-out panels, one lost ramp moves
from 2.28 m to about 2.6 m, just across the radius, and a second lost ramp is drawn at the
top because the candidate can no longer place it. The exploratory full-run panel loses that
second ramp and one other that moves from 2.29 m to 2.67 m, and gains two that move from
2.64 m to 1.76 m and 1.49 m. Below, six street-level crops
show each changed ramp's GT mark circled.](figures/gsv-partial-pose/fig4_bend_recall.png)

*Figure 4. The two ramps behind the failed clause, which are the same two for both
candidates. Both sit at a threshold, and neither is a large misplacement. One ramp's
nearest site moves from 2.28 m to 2.60 m (2.56 m under LOCO), just across the 2.5 m match
radius. The other is a reviewer's missed-ramp mark near the horizon whose off raycast lands
at 24.8 m. The partial pose lengthens that ray to 25.9 m, past the production 25 m cap, so
the candidate cannot place the mark at all. The right panel is EXPLORATORY: on the whole
run, the LOCO arm loses the same beyond-the-cap mark and one other threshold ramp (2.29 m
to 2.67 m), and gains two that sat just outside the radius (2.64 m to 1.76 m and 1.49 m).*

### External referee: city inventories (frozen@off, 5 m pool on off)

| height | city | pool | off median / p90 | partial | partial-loco | partial-shuffled | full | mirror |
|---|---|---|---|---|---|---|---|---|
| auto | bend | 9,092 | 0.718 / 1.825 | 0.697 / 1.774 | 0.695 / 1.773 | 0.732 / 1.859 | 0.807 / 2.156 | 0.767 / 1.991 |
| auto | gainesville | 2,324 | 1.278 / 3.044 | 1.217 / 2.943 | 1.240 / 2.974 | 1.264 / 3.043 | 1.240 / 3.094 | 1.346 / 3.323 |
| 2.6 | bend | 8,872 | 0.720 / 1.864 | 0.696 / 1.816 | 0.695 / 1.817 | 0.733 / 1.911 | 0.798 / 2.177 | 0.767 / 1.995 |
| 2.6 | gainesville | 1,992 | 1.867 / 3.905 | 1.806 / 3.826 | 1.838 / 3.838 | 1.868 / 3.862 | 1.812 / 3.999 | 1.900 / 4.022 |

The inventory never passed through our raycast. Against it, both partial arms beat flat
on median and p90 in both cities at both heights. The mirror is worse than flat
everywhere, and so is the full pose on the p90. The shuffles land within 0.05 m of flat
either way. The effect is small (0.02-0.06 m on the median) because inventory distance is
dominated by other errors (height, survey placement). Own-association coverage moves by
less than 1 pt in either direction (`summary.md`).

### At 2.6 m (reported)

The partial arm still beats flat and the shuffle on the median in every city. The p90 in
Paterson ties (4.544 / 4.546 against 4.543), and in Gainesville the arm loses 6 (partial)
or 3 (LOCO) of 97 ramps. Gainesville at 2.6 m is the case #79 already ruled against: most
of its panos are the 2026 rig, about 2.0 m, so ranges run long, and the fitted pitch
fraction drops by half. A roll or pitch term and a height error both move range (the
issue's point 6), and at the wrong height the pose correction partly corrects the wrong
thing. The default ships at `auto`, which is why `auto` decides.

### EXPLORATORY: whole run, LOCO coefficients (added after the verdict, never gated)

This reading was added once the recall clause proved too coarse for the half split. The
LOCO coefficients never saw the city scored, so the whole run stays held out for them, and
the GT pool roughly doubles. The same four clauses are applied, for reading only
(`runs/_pooled/partial_pose/explore_full/verdict.md`):

| height | city | pairs | off med / p90 | loco med / p90 | loco-shuffled med / p90 | off-pool R@2.5 (lost/gained) | inventory med / p90, off -> loco |
|---|---|---|---|---|---|---|---|
| auto | paterson | 49,045 | 1.731 / 3.913 | 1.590 / 3.631 | 1.764 / 4.062 | 0.904 -> 0.916 (2/6) | — |
| auto | bend | 80,285 | 1.516 / 3.770 | 1.425 / 3.572 | 1.565 / 3.943 | 0.929 -> 0.929 (2/2) | 0.625 / 1.373 -> 0.615 / 1.335 |
| auto | gainesville | 35,889 | 1.927 / 4.276 | 1.762 / 3.903 | 1.925 / 4.268 | 0.917 -> 0.913 (4/3) | 1.135 / 2.919 -> 1.105 / 2.820 |
| auto | sao_paulo | 50,915 | 1.906 / 4.158 | 1.773 / 3.934 | 1.930 / 4.253 | 0.922 -> 0.926 (1/2) | — |
| auto | laurens_gsv | 1,236 | 1.749 / 3.908 | 1.576 / 3.534 | 1.722 / 3.796 | 0.907 -> 0.912 (2/3) | — |

All four clauses pass at `auto` and at 2.6 m. In this reading, laurens_gsv clears the
500-pair bar and passes (i) as well. This is **not** a verdict. It amends a clause after
that clause failed, which is exactly what pre-registration exists to prevent. What it
does show is that the failure is consistent with the pool's size rather than a trend:
with twice the ramps, Bend's recall change is zero.

### Tie-back to #113

On the full run at 2.6 m, frozen at flat, the median / mean / p90 figures below come from
`tieback_113.csv`, tier 0.55:

| city | off | full | full-mirror | fixed-113 |
|---|---|---|---|---|
| paterson | 2.147 / 2.413 / 4.565 | 2.332 / 2.797 / 5.587 | 2.648 / 3.105 / 5.986 | 2.096 / 2.393 / 4.597 |
| bend | 1.539 / 1.871 / 3.808 | 1.655 / 2.115 / 4.450 | 2.170 / 2.671 / 5.468 | 1.459 / 1.806 / 3.753 |
| sao_paulo | 1.643 / 1.949 / 3.865 | 1.544 / 1.954 / 4.018 | 2.315 / 2.748 / 5.440 | 1.495 / 1.810 / 3.635 |

#113's scratch table had paterson 2.125 / 2.394 / 4.548 → 2.074 / 2.366 / 4.556, bend
1.524 / 1.852 / 3.764 → 1.444 / 1.787 / 3.705, and sao_paulo 1.635 / 1.940 / 3.852 →
1.492 / 1.802 / 3.619. These figures are 0.2-1.2% above those, on a site set that is not
identical: #113's scratch code is not in the repo, so its exact filter is unknown. Every
ordering reproduces. The full pose is worse than flat on the p90 in all three cities, the
mirror is worst, and `fixed-113` is the best median.

## What changed and what did not

- **New:** `scripts/gsv_partial_pose.py` (`fit`, `score`, `tieback`, `verdict`, `explore`),
  `tests/test_gsv_partial_pose.py`, the per-city and pooled outputs, and this document.
- **Not changed:** `geo.py` and `fuse_sites.py` behaviour. There is no `--apply-pose
  partial`, `auto` still resolves flat for GSV, and nothing in `send_to_ps.py` or the
  analysis scripts moves. The comments at `fuse_sites.AUTO_ROAD_SOURCES` and
  `geo._world_ray` now point here.

## What remains

1. **A confirmatory decision on recall.** This is Jon's call. Either (a) accept the
   exploratory reading and wire `--apply-pose partial` as opt-in with the pooled `auto`
   constants (k_pitch 0.183, k_roll 0.382; a small change to `fuse_sites.pano_pose` plus
   tests), or (b) pre-register a confirmatory run. Option (b) needs a recall clause sized
   to its pool, for example a paired count of lost minus gained ramps with a stated
   minimum pool. It should run on data this study never scored: a new GSV city with pitch
   and roll, or the LOCO arm on a fresh benchmark split.
2. **The coefficient is attenuated.** The fit runs on a flat association that gates out
   large residuals, as #113 warned, so 0.18 / 0.38 is probably a lower bound. `fixed-113`
   (0.25 / 0.5) does slightly better than the fitted arm on several medians. A fixed-point
   fit (re-associate under the fitted arm, refit) was not tried, and #87/#89 show why any
   fixed point needs its own validation.
3. **Height and pose interact.** The pitch fraction depends on the height model (0.18 at
   `auto`, 0.15 at 2.6 m, and halved in Gainesville). A production constant should be fit
   at the height that ships and refit if the height default changes.
4. **The AI-vs-human label frame (beta, #113, sidewalk-panorama-tools#191) is a different
   quantity** and must not borrow these numbers. Beta is about 0.9 because it concerns the
   viewer; these fractions concern the ground.

## Limits

- **Self-agreement is not accuracy.** Clause (i) measures how tightly members agree. The
  inventory is the only external check, it covers two cities, and there the effect is
  0.02-0.06 m.
- **The recall pools are small** (97-185 ramps per city on the half), which is the reason
  for the verdict above.
- **Pairs weight large sites quadratically.** This is #113's and #27's metric, kept for
  continuity.
- **GSV only.** Mapillary's pose question is #42 and #51. Panoramax's is #57.

## Reproduce

```bash
python scripts/gsv_partial_pose.py fit                      # ~40 s; coefficients.csv
python scripts/gsv_partial_pose.py score --benchmark-root ../RampNet/benchmark   # ~4 min
python scripts/gsv_partial_pose.py tieback paterson bend sao_paulo gainesville
python scripts/gsv_partial_pose.py verdict                  # verdict.md + summary.md
python scripts/gsv_partial_pose.py explore --benchmark-root ../RampNet/benchmark  # exploratory
python scripts/gsv_partial_pose.py figures --benchmark-root ../RampNet/benchmark  # docs/figures/gsv-partial-pose/
```

Inputs are read in place: `runs/<city>/results.jsonl`, `runs/<city>/depth/index.csv` (for
`auto`), `runs/{bend,gainesville}/inventory_oracle/inventory.geojson`, and RampNet's
`benchmark/<city>/{verdicts.json,records.jsonl}`. No network, no GPU.

`figures` redraws Figures 1-2 from the committed pooled CSVs, which it copies into
`docs/figures/gsv-partial-pose/data/`. Figures 3-4 need a re-fuse. Their data
(`site_example.json`, `bend_recall_ramps.csv`) is written there once, with the Bend
lost/gained counts asserted against the committed `gt.csv`, and re-read after that. Pass
`--refresh` to recompute it. Crops come from local bundle panos only.
