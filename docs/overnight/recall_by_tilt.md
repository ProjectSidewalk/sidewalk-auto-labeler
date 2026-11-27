# Does RampNet's open-rig recall depend on pano tilt? (overnight `recall_by_tilt`)

Run 2026-11-26, CPU only, no network, on judged RampNet bundles. Script:
`scripts/recall_by_tilt.py`. Data: `docs/overnight/data/recall_by_tilt/`.

## 0. Pre-registration (written before any recall-by-covariate number was computed)

What I had seen before writing this: the per-split recall at 0.55 (to check that my unit
definition reproduces the published numbers), the pose-field coverage per split, and the
published RampNet docs (`open_rig_misses_258.md`, `pano_tilt_estimate.md`,
`lyon_review_159.md` section 10, which already shows Lyon recall by gravity-corrected
elevation and the same recall, about 0.26, for Lyon's GoPro Max and Ladybug groups). I had
not seen recall split by tilt, signed tilt, rig at matched dip, or any model fit.

### Units

- Splits: the **open-imagery pool** is laurens_mapillary, bayonne, juneau, lyon, richmond,
  clovis, morgantown, annapolis (all have verdicts). budapest_district5 is reported alone and
  never pooled (RampNet holds it out: single rater, low confidence). The five GSV splits
  (laurens_gsv, paterson, bend, gainesville, sao_paulo) are a descriptive contrast only.
- Panos: `rampnet.validation.collect` semantics, re-implemented on the bundle files: the
  verdict list must line up with the bundle record's detections, a pano with any `None`
  verdict is dropped, and it enters the recall pool only if its missed-ramp check is
  confirmed (`no_missed` set or a missed mark present; legacy entries trusted).
  **Primary population: the unbiased subset** (group `top`, the 5 hand-picked dense panos per
  split, excluded; this is also Lyon's posted plan). Sensitivity: all panos.
- GT ramp = a detection with verdict `True` (all bundle detections are >= 0.55, asserted)
  or a missed mark without `unsure`. Detected = verdict `True`. `duplicate` / `False` /
  `unsure` detections are not GT ramps. Recall = detected / GT ramps.
- Reproduction check: laurens_mapillary all panos 97/249 = 0.390, lyon unbiased 244/921 =
  0.265, bayonne unbiased 84/261 = 0.322.

### Pose, in RampNet's #113 convention (pitch > 0 nose down, roll > 0 left side up)

- Mapillary: `geo.mapillary_pitch_roll(source_metadata.computed_rotation)` (pitch > 0 nose
  up, roll in PS sign = left side up) -> pitch113 = -pitch, roll113 = roll. Over-tilt or
  missing rotation -> no pose.
- Lyon: RampNet's image-based estimate `analysis_out/pano_tilt/lyon.json` (status `ok`
  only; declined -> no pose). Not imputed.
- Bayonne: Panoramax `pers:pitch` / `pers:roll` only where `semantics` says
  `computed_by[pitch] = panoflat/...` and the pair is not exactly 0/0; pitch113 =
  -pers:pitch (Panoramax pitch is nose up, `pano_tilt_estimate.md` section 4.3), roll113 =
  pers:roll. 0/0 and absent -> no pose. Not imputed.
- GSV: record `camera_pitch`, wrapped `camera_roll` (already #113's convention).
- Convention check (outcome-free, uses GT positions only): the number of GT points above the
  horizon must fall when the pose is applied and rise when its sign is flipped; on Lyon it
  must reproduce `lyon_review_159.md`'s 87 -> 2 (declined panos read as level, as there).

### Covariates per GT ramp

- **T** = degrees the gravity horizon sits above the image's middle row at the ramp's
  column, computed exactly by rotating the ray with `geo._world_ray` (first-order:
  T = pitch113 cos b + roll113 sin b, b = bearing from the centre column). Primary: |T|.
- |tilt| = acos(cos pitch cos roll), per pano (secondary covariate).
- gravity dip = degrees below the gravity horizon (range proxy); image dip = (y - 0.5) x 180.
  Dip buckets (fixed): < 4, 4-6, 6-10, 10-20, >= 20 degrees.
- rig class: gopro_max, gopro_fusion, trimble_mx7, istar_pulsar, ladybug, other_360
  (QooCam, insta360, VIRB, THETA, GoPro MAX2), unknown (no make/model); capture year; city.
- Descriptive only: local in-plane rotation at the ramp, -pitch113 sin b + roll113 cos b.

### Decision rule (open-imagery pool, unbiased subset, posed panos only)

A tilt dose-response is **established** only if all three hold for **|T|**:

1. **Bucket gap.** |T| is cut at the quartiles of the pooled posed GT units (outcome-blind).
   recall(Q1) - recall(Q4) >= 0.10, and its pano-cluster bootstrap 95% CI (2,000 draws,
   panos resampled within each split, seed 20261126) excludes 0.
2. **Survives covariates.** Logistic model detected ~ 1 + min(|T|, 20)/5 + rig-class dummies
   + gravity-dip-bucket dummies (numpy IRLS). The |T| coefficient's pano-cluster bootstrap
   95% CI (1,000 draws) lies wholly below 0.
3. **Shuffle control clean.** Pose (pitch, roll pairs) shuffled among posed panos within rig
   class x city, 200 seeded draws, quartile edges held at the real ones: fewer than 10 of 200
   draws reach a Q1 - Q4 gap >= 0.10. (The permutation p of the observed gap is reported.)

The same three tests on **pano |tilt|** are secondary: if only |tilt| passes, the verdict is
"secondary only, not established".

Non-gating, descriptive: recall by signed T (bins <= -3, -3..-1, -1..1, 1..3, >= 3); a
within-pano contrast (logistic on |T| minus its pano mean, plus the pano mean); recall by dip
bucket per rig; recall by rig class standardized to the pooled dip distribution; a rig x city
table; the GSV contrast; sensitivities (all panos, city fixed effects, image dip instead of
gravity dip, first-order T). Coverage of pose and the recall of unposed units are reported;
unposed units are left out of every tilt test, never imputed.

## 1. Verdict

**NOT ESTABLISHED.** Pano tilt does not explain RampNet's low recall on the open rigs.

| covariate | 1. Q1 − Q4 recall gap (95% CI) | 2. logistic coef per 5° (95% CI) | 3. shuffles reaching the gap | permutation p | passes |
|---|---|---|---|---|---|
| **\|T\|** at the ramp's bearing (primary) | +0.060 (−0.009, +0.129): **fail** | −0.186 (−0.334, −0.035): pass | 40 of 200: **fail** | 0.78 | no |
| pano \|tilt\| (secondary) | +0.101 (+0.027, +0.174): pass | −0.282 (−0.420, −0.152): pass | 107 of 200: **fail** | 0.52 | no |

Open-imagery pool, unbiased subset, posed panos: 2,334 GT ramps on 678 panos. Quartile
edges: |T| 0.77 / 1.76 / 3.43°, |tilt| 1.72 / 3.02 / 5.21°.

The pooled gap and the pooled logistic slope come from **which city a pano is in**, not from
its tilt:

- **Shuffle control.** Shuffling poses among panos of the same rig and city reproduces the
  pooled gap as often as not (|tilt|: median shuffled gap +0.103, against +0.101 observed).
  A pano's own tilt adds nothing beyond its rig × city stratum.
- **City fixed effects** (pre-registered sensitivity): the |T| coefficient goes from −0.186 to
  **+0.034 (−0.145, +0.215)**. Image dip in place of gravity dip: +0.001 (−0.152, +0.173).
- **Within split** (post hoc, see section 5): the Mantel-Haenszel-weighted Q1 − Q4 gap is
  **−0.005 (−0.060, +0.051)** for |T| and −0.028 (−0.083, +0.028) for |tilt|.
- **Within pano** (pre-registered, non-gating): ramps at bearings where the same pano is more
  tilted are detected **more** often, not less: +0.40 (+0.13, +0.66) per 5° of |T| minus the pano
  mean. The between-pano term is −0.38 (−0.56, −0.21). Adding a front/back-versus-side bearing
  term (post hoc) leaves it at +0.42. I have no explanation for the positive sign. Either way
  it points away from "levelling recovers ramps".

The pool mixes high-tilt, low-recall Lyon and Bayonne with low-tilt, high-recall US Mapillary
cities. Inside any single split there is no dose-response. Lyon by |T| quartile reads 0.24 /
0.25 / 0.29 / 0.27, Laurens 0.31 / 0.30 / 0.32 / 0.40, Morgantown 0.68 / 0.69 / 0.62 / 0.76
(`absT_by_split.csv`).

## 2. Results tables (unbiased subset)

### Recall by quartile (`quartiles.csv`)

| pool | covariate | Q1 | Q2 | Q3 | Q4 | Q1 − Q4 (95% CI) |
|---|---|---:|---:|---:|---:|---|
| open (8 splits) | \|T\| | 0.498 | 0.480 | 0.509 | 0.438 | +0.060 (−0.009, +0.129) |
| open | \|tilt\| | 0.503 | 0.482 | 0.539 | 0.402 | +0.101 (+0.027, +0.174) |
| open, all panos (sensitivity) | \|T\| | 0.546 | 0.532 | 0.565 | 0.466 | +0.081 (+0.013, +0.152) |
| open, all panos (sensitivity) | \|tilt\| | 0.543 | 0.554 | 0.584 | 0.429 | +0.114 (+0.036, +0.190) |
| open, first-order T | \|T\| | 0.498 | 0.480 | 0.508 | 0.440 | +0.058 (−0.011, +0.128) |
| GSV (5 splits; own quartiles, max 8°) | \|T\| | 0.624 | 0.645 | 0.633 | 0.645 | −0.022 (−0.104, +0.056) |
| budapest (held out) | \|T\| | 0.455 | 0.508 | 0.415 | 0.439 | +0.015 (−0.159, +0.200) |

On all panos the |T| gap clears 0.10's CI test but not the bar itself (+0.081). It would fail
the shuffle as well, because the same composition drives it.

### Signed T and in-plane rotation (pooled, open, descriptive)

| signed T (°) | ≤ −3 | −3..−1 | −1..1 | 1..3 | ≥ 3 |
|---|---:|---:|---:|---:|---:|
| recall | 0.429 (301) | 0.494 (415) | 0.487 (753) | 0.519 (474) | 0.453 (391) |
| in-plane rotation at the ramp | 0.415 (330) | 0.518 (479) | 0.497 (724) | 0.488 (471) | 0.452 (330) |

Both tails sit 3 to 8 points low, with no sign asymmetry. The tails are mostly Lyon and Bayonne,
so this is the same between-city composition.

### Rig, city and apparent scale (presentation or content?)

Recall by rig × city (`rig_city.csv`, all units, unbiased):

| rig | city: recall (GT) |
|---|---|
| GoPro Max | richmond 0.71 (52), morgantown 0.68 (225), budapest 0.46 (268), laurens 0.33 (212), bayonne 0.32 (261), lyon 0.26 (453) |
| Lyon, every rig | GoPro Max 0.26 (453), Ladybug 0.25 (178), other 360 0.26 (125), unknown 0.28 (165) |
| Trimble MX7 | annapolis 0.69, juneau 0.66 |
| GSV | bend 0.74, paterson 0.65, gainesville 0.65, sao_paulo 0.63, **laurens_gsv 0.46** |

- **The same camera spans 0.26 to 0.71 across cities.**
- **In Lyon, four rig groups read the same 0.25 to 0.28**, although the median tilt of
  ecartip's GoPro Max panos is 10.4° and of the Ladybug panos 1.7° (`pano_tilt_estimate.md`).
- **Laurens is low on both of its rigs** (GoPro Max 0.33, GSV 0.46).

Laurens also shows a real rig gap in the same town (#151), so presentation is not zero.
But most of the variance is at city level. That is ramp design and kerb style (Lyon's lipped
kerbs, `lyon_review_159.md`), rubric fit, or the single rater. This data cannot tell those
apart.

**Apparent scale is a strong, common gradient, not a rig difference.** Every rig's recall
rises with gravity dip (for example, GoPro Max 0.15 / 0.19 / 0.38 / 0.45 / 0.45 across <4 /
4-6 / 6-10 / 10-20 / ≥20°; `dip_by_rig.csv`). Every rig has a similar dip mix (52 to 59% of
GT at ≥ 10°, except other_360 at 38%). Standardizing each rig to the pooled dip distribution
barely moves it:

| rig | crude recall | dip-standardized |
|---|---:|---:|
| GoPro Max | 0.386 | 0.384 |
| Ladybug | 0.253 | 0.263 |
| other 360 | 0.264 | 0.313 |
| unknown | 0.350 | 0.355 |
| GoPro Fusion | 0.650 | 0.633 |
| Trimble MX7 | 0.677 | 0.686 |
| iSTAR Pulsar | 0.729 | 0.734 |

So the rig gaps are not a difference in how far away the ramps are.

### Pose coverage (`coverage.csv`) and convention check (`horizon_check.csv`)

| split | GT | posed | source | recall posed / unposed |
|---|---:|---:|---|---|
| Mapillary splits (6) | 1,374 | 100% | computed_rotation | — |
| lyon | 921 | 98.0% | RampNet image estimate (18 GT on declined panos) | 0.263 / 0.389 |
| **bayonne** | 261 | **21.8%** | panoflat (57); 0/0 on 78 and absent on 126 are not poses | 0.298 / 0.328 |
| budapest | 268 | 97.8% | computed_rotation (6 missing) | 0.454 / 0.667 |
| GSV (5) | 1,296 | 100% | record | — |

Bayonne is thin. It enters the tilt tests with 57 GT ramps, and its 204 unposed GT ramps are
excluded, not imputed.

The convention check passed on every source:

- GT points above the horizon fall when the pose is applied and rise when its sign is flipped.
  Lyon reads 75 → 1 (flipped 141), bayonne 8 → 0 (14), budapest 12 → 0 (35) and richmond
  3 → 0 (14).
- Lyon reproduces `lyon_review_159.md`'s 87 → 2 once its unsure marks are dropped.
- Juneau goes 0 → 3 (flipped 16), which is SfM noise on a level vehicle rig.
- Juneau's stored `camera_pitch` / `camera_roll` equal `geo.mapillary_pitch_roll` to 7e-15°.
- The vectorized rotation equals `geo._world_ray` exactly.

## 3. Recommendation

**Do not run the GPU "level, then detect" test as a fix for the open-rig recall gap.** Tilt
fails its pre-registered rule. Inside a split or a rig × city stratum it predicts nothing, and
within a pano it goes the wrong way. The apparent-camera-height arm is not supported either:
recall's dip gradient is shared by every rig, and matching dip does not close any rig gap, so
there is no rig-specific scale offset to normalize. The deficit is at city level: the same
GoPro Max reads 0.26 to 0.71 across cities, and every Lyon rig reads about 0.26. GPU time is
better spent on content and labels (Lyon ramp types, #258's sub-threshold calibration on
Laurens/Bayonne).

If Jon still wants one cheap confirmation, run a single paired arm: **Lyon ecartip GoPro Max,
raw vs levelled with the cached estimates** (56 panos, median tilt 10.4°, `level_equirect`
already exists). Score recall at 0.30 and 0.55 against the existing verdicts, with missed
marks re-projected through the same rotation. The prediction from this analysis is a null
within ±0.05. Skip the camera-height arm.

## 4. Caveats (what this cannot see)

- **GT is RampNet-anchored and single-rater.** Every miss is a reviewer-added mark, made on the
  same tilted image. A ramp both missed is invisible. If tilt hurt the reviewer and the model
  alike, this would understate the effect.
- **Pose quality.**
  - Mapillary's SfM tilt has unmeasured absolute error.
  - The image estimator is validated only to 8° (Lyon reaches 27°).
  - Bayonne's panoflat sign is inferred from 11 panos (confirmed here by the horizon check).
  - Measurement noise in T attenuates any slope toward zero. But noise cannot create the
    positive within-pano sign, nor the zero within-split gap.
- **Clustering by pano only.** Panos of one sequence are not independent, so the CIs are
  somewhat narrow, which makes them generous to the effect.
- **Strata are small.** Some rig × city shuffle strata have fewer than 20 panos (Richmond GoPro
  Max 17, Juneau GoPro Max 1). Rig class and city are nearly collinear for half the pool.
- **Recall at 0.55 only.** The bundles were exported at 0.55, so no 0.30 reading is possible
  from verdicts alone.
- **Levelling is not tested.** A model whose response really depends on the in-plane rotation
  of the crop could still benefit from levelling while showing no dose-response here. That
  would need tilt to vary within the same ramp, and only the GPU arm above tests it.

## 5. Amendment (2026-11-26, post hoc, after the pre-registered results were read)

I added two checks after seeing the results. Neither changes the verdict.

1. A within-split Q1 − Q4 gap: the pooled quartile edges, per-split gaps combined with
   n1·n4/(n1+n4) weights, and a pano-cluster bootstrap. I added it because the city-FE
   sensitivity showed the pooled slope vanishing, and a gap is easier to read than a
   coefficient.
2. The within-pano model with |cos b| added, to test whether the positive within-pano term is
   a front/back-versus-side bearing effect. It is not: +0.42 (+0.12, +0.70); the bearing term
   is −0.17 (−0.54, +0.20).

The rig × split quartile tables (`posthoc_*_by_rig_split.csv`) are also post hoc.

## 6. Replication

```bash
# from the repo root; needs the RampNet checkout beside this repo (benchmark/ and
# analysis_out/pano_tilt/lyon.json; the script refuses to run without the latter)
python scripts/recall_by_tilt.py \
    --benchmark-root ../RampNet/benchmark \
    --lyon-tilt ../RampNet/analysis_out/pano_tilt/lyon.json \
    --out docs/overnight/data/recall_by_tilt          # ~10 s, seed 20261126
python -m pytest -q tests/test_recall_by_tilt.py
```

Inputs: RampNet bundles (`records.jsonl`, `verdicts.json`) for the 14 splits, plus RampNet's
`analysis_out/pano_tilt/lyon.json`. It needs no run files, no network and no GPU.
Outputs: `summary.json` (verdict, every test), `units_unbiased.csv` (one row per GT ramp with
all covariates; `pano_id` is globally unique), and the tables named above.
