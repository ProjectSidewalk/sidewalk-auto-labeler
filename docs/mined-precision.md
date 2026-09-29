# Precision of mined hard positives (RampNet#158: steps 1, 2 and phase 2)

`scripts/mined_precision.py` measures how often a label mined from multi-view consensus is
right. A **mined label** is a strong fused site (≥ 3 operational panos) projected into a
nearby judged benchmark pano that did not detect it. The reviewer's verdicts in
`../RampNet/benchmark/<city>/` decide whether the projected point is a ramp. The script's
docstring and the CLAUDE.md section describe the buckets. There are two denominators, and
neither is quoted without the other:

- **hard-only** = tp / (tp + fp). The share of mined targets that are misses the model does
  not already make.
- **all-mined** = (tp + already_detected) / (… + fp). The share of shipped labels that are
  correct. A miner has no verdicts, so it cannot drop `already_detected`.

Pre-registered rule (RampNet#158): ≥ 0.80 → build the miner; 0.50–0.80 → add the visibility
test first; < 0.50 → drop this label source. The rule is read at ≤ 15 m on the point
estimate, and a reading whose 95% Wilson CI crosses a band edge is "not decisive".

Step 1 (2026-09-21, [labeler#54](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/54))
found mined labels about half right, with placement as the main failure: 48 of 65 false
positives have a verdict-true detection as their nearest GT point. Its numbers are in the
[correction comment on RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158#issuecomment-5769032890).
Step 2 is below.

## Step 2 (2026-09-29): measured camera heights

**Question.** Does moving fusion, GT placement and projection from the 2.6 m constant to
measured heights raise mined precision? Step 2 was planned as per-pano GSV depth heights
(#40) plus Mapillary rig-class heights (#53). Nothing else changed from step 1: candidates
within 15 m, a 5 m match radius, operational members only, `apply_pose` off, fusion at the
benchmark threshold, and the same rule and CI.

**Arms.** All arms run on the same five runs and the same verdicts. Only `--camera-height`
differs:

| arm | `--camera-height` (richmond, paterson, bend, gainesville, sao_paulo) | what it is |
|---|---|---|
| `step1_2.6` | 2.6 for all | step 1's pre-registered headline |
| `step1_gsv2.2` | 2.6 2.2 2.2 2.2 2.2 | step 1's measured-constant row |
| **`step2_perpano`** | **per-rig per-pano per-pano per-pano per-pano** | **step 2 as planned** |
| `auto` | auto for all | the labeler's production fuse default since #79: GSV per capture-year rig, 2.0 / 2.5 m; Mapillary 2.6 m |

The `auto` arm is also RampNet PR #210's `proj_height_auto`. That arm calls the labeler's
own resolver (`fs.HEIGHT_AUTO`, pose off; `scripts/analysis/crossview_arms/geometry.py` on
`analysis/crossview-align-48`), so the rule is the same and it is not reported separately.

**Richmond is identical in every arm.** `per-rig` reads `runs/richmond/camera_heights.json`.
Since #89 every group in that table is 2.6 m with `recommended: false`, so 0 of 9,091 panos
take a per-rig height (the report says so). `auto` also keeps Mapillary at 2.6 m. #53 exists,
but no validated Mapillary height does, so step 2 has nothing to apply to richmond.

**Inputs are frozen to step 1.** After step 1, the 2026-09-21 gap-fill phases (#32) appended
2,231 panos to gainesville and 7,293 to são paulo. `results.jsonl` is append-only, so the
step-1 files are the first 35,204 and 22,741 lines. They are byte-identical to the makelab2
archive copies. The `step1_*` arms reproduce the step-1 correction exactly (0.327 32/98 and
0.511 69/135; 0.440 40/91 and 0.585 72/123), so every difference below comes from the
height. Hashes and sources are in
[`figures/mined-precision/data/inputs.json`](figures/mined-precision/data/inputs.json).
The same arms on the gap-filled runs are in `data/current/` (see Caveats).

### Headline, pooled over five cities

| arm | hard-only ≤ 15 m | all-mined ≤ 15 m | hard-only ≤ 10 m | all-mined ≤ 10 m | cand. | fp placed |
|---|---|---|---|---|--:|--:|
| step1_2.6 | 0.327 [0.24, 0.42] (32/98) | 0.511 [0.43, 0.59] (69/135) | 0.471 [0.31, 0.63] (16/34) | 0.660 [0.53, 0.77] (35/53) | 148 | 48 |
| step1_gsv2.2 | 0.440 [0.34, 0.54] (40/91) | 0.585 [0.50, 0.67] (72/123) | 0.632 [0.47, 0.77] (24/38) | 0.725 [0.59, 0.83] (37/51) | 136 | 36 |
| **step2_perpano** | **0.483 [0.38, 0.59] (42/87)** | **0.605 [0.51, 0.69] (69/114)** | 0.657 [0.49, 0.79] (23/35) | 0.745 [0.60, 0.85] (35/47) | 127 | 29 |
| auto | 0.452 [0.35, 0.56] (38/84) | 0.562 [0.47, 0.65] (59/105) | 0.647 [0.48, 0.79] (22/34) | 0.727 [0.58, 0.84] (32/44) | 118 | 31 |

*fp placed* counts false positives (≤ 15 m) whose nearest GT point in that pano is a
verdict-true detection. The pano saw a real ramp there and the site landed more than 5 m
from it.

### GSV cities only (paterson, bend, gainesville, são paulo)

| arm | hard-only ≤ 15 m | all-mined ≤ 15 m | hard-only ≤ 10 m | all-mined ≤ 10 m |
|---|---|---|---|---|
| step1_2.6 | 0.339 [0.23, 0.47] (19/56) | 0.560 [0.45, 0.66] (47/84) | 0.375 [0.18, 0.61] (6/16) | 0.667 [0.49, 0.81] (20/30) |
| step1_gsv2.2 | 0.551 [0.41, 0.68] (27/49) | 0.694 [0.58, 0.79] (50/72) | 0.700 [0.48, 0.85] (14/20) | 0.786 [0.60, 0.90] (22/28) |
| **step2_perpano** | **0.644 [0.50, 0.77] (29/45)** | **0.746 [0.63, 0.84] (47/63)** | 0.765 [0.53, 0.90] (13/17) | 0.833 [0.64, 0.93] (20/24) |
| auto | 0.595 [0.44, 0.73] (25/42) | 0.685 [0.55, 0.79] (37/54) | 0.750 [0.51, 0.90] (12/16) | 0.810 [0.60, 0.92] (17/21) |

### Per city, ≤ 15 m: step 1 (2.6 m) → step 2 (per-pano / per-rig)

| city | hard-only step 1 | hard-only step 2 | all-mined step 1 | all-mined step 2 | fp placed |
|---|---|---|---|---|---|
| richmond (Mapillary) | 0.310 [0.19, 0.46] (13/42) | 0.310 [0.19, 0.46] (13/42), unchanged | 0.431 [0.31, 0.57] (22/51) | 0.431, unchanged | 20 → 20 |
| paterson | 0.350 [0.18, 0.57] (7/20) | **0.800 [0.58, 0.92] (16/20)** | 0.629 [0.46, 0.77] (22/35) | 0.852 [0.68, 0.94] (23/27) | 10 → 2 |
| bend | 0.636 [0.35, 0.85] (7/11) | 0.750 [0.41, 0.93] (6/8) | 0.714 [0.45, 0.88] (10/14) | 0.800 [0.49, 0.94] (8/10) | 3 → 2 |
| gainesville | 0.133 [0.04, 0.38] (2/15) | 0.333 [0.06, 0.79] (1/3) | 0.350 [0.18, 0.57] (7/20) | 0.750 [0.41, 0.93] (6/8) | 12 → 2 |
| são paulo | 0.300 [0.11, 0.60] (3/10) | 0.429 [0.21, 0.67] (6/14) | 0.533 [0.30, 0.75] (8/15) | 0.556 [0.34, 0.75] (10/18) | 3 → 3 |

### By range band, pooled: step 1 (2.6 m) → step 2

| band | hard-only step 1 | hard-only step 2 | all-mined step 1 | all-mined step 2 |
|---|---|---|---|---|
| 0–8 m | 0.545 [0.35, 0.73] (12/22) | 0.667 [0.45, 0.83] (14/21) | 0.667 [0.49, 0.81] (20/30) | 0.741 [0.55, 0.87] (20/27) |
| 8–12 m | 0.333 [0.19, 0.51] (10/30) | 0.500 [0.32, 0.68] (13/26) | 0.583 [0.44, 0.71] (28/48) | 0.629 [0.46, 0.77] (22/35) |
| 12–18 m | 0.217 [0.12, 0.36] (10/46) | 0.375 [0.24, 0.53] (15/40) | 0.368 [0.26, 0.50] (21/57) | 0.519 [0.39, 0.65] (27/52) |

Candidates stop at 15 m, so the 12–18 m band holds 12–15 m. The full per-city × band ×
arm table, with both denominators and CIs, is
[`data/frozen/compare.md`](figures/mined-precision/data/frozen/compare.md) (CSV beside it).
Each arm's own reports, including support and confidence strata, heights resolved and
fall-back counts, are in `data/frozen/<arm>/<city>/report.md`.

### Reading against the pre-registered rule

- **Pooled, ≤ 15 m, step 2.** hard-only 0.483 [0.38, 0.59]: *drop* on the point estimate,
  with a CI that spans drop and visibility, so not decisive. all-mined 0.605 [0.51, 0.69]:
  *visibility*, and the whole CI is inside that band, so decisive. At step 1's 2.6 m,
  hard-only was decisively *drop* and all-mined's CI straddled 0.50.
- **GSV only, ≤ 15 m, step 2.** hard-only 0.644 [0.498, 0.768] and all-mined 0.746
  [0.627, 0.837]. Both point estimates read *visibility*, and **neither is decisive** by the
  rule above: hard-only's lower bound (0.498) is just under 0.50, and all-mined's upper
  bound (0.837) is over 0.80.
- **Richmond reads *drop*** (hard-only 0.310 [0.19, 0.46]; all-mined 0.431 [0.31, 0.57]) and
  did not move, because step 2 had no Mapillary height to apply.

So, on GSV, measured heights move the point estimates from drop into visibility on both
denominators. That is a move, but not a decisive reading: both CIs cross a band edge. On
open imagery, the case #158 exists for, nothing has changed. Of the 29 placed false
positives left after step 2, 20 are richmond's.

### What the change is made of

- **Most of the GSV gain comes from leaving 2.6 m, not from per-pano over a constant.**
  GSV hard-only goes 0.339 (2.6) → 0.551 (2.2 constant) → 0.644 (per-pano), and pooled goes
  0.327 → 0.440 → 0.483. Per-pano's gain over the 2.2 constant is inside every CI.
- **Per-pano vs `auto`: no resolvable difference here.** Pooled hard-only is 0.483 vs 0.452
  and GSV-only 0.644 vs 0.595, with overlapping CIs. RampNet PR #210 scored single pairs and
  found per-pano heights (`proj_height_perpano`) worse than `auto` on paired gain. That
  measures placement directly; this measures a precision rate after re-fusion. The two do
  not contradict each other, and neither separates the two frames.
- **Placement false positives fall 48 → 29, and the candidate set shrinks** (148 → 127;
  `already_detected` 37 → 27). A different height means a different fuse, so the arms are
  not paired candidate by candidate. Some of the gain is candidates that no longer exist.
  For example, a pano that detected the ramp now joins its site as a member instead of
  being mined as a non-member (gainesville: 20 candidates → 8). That is also how a real
  miner would behave, so it counts, but a fixed-candidate comparison is not available.
- **Where step 1's range cliff went.** On GSV under per-pano, hard-only reads 0.778 / 0.643 /
  0.591 across the three bands, against 0.500 / 0.333 / 0.286 at 2.6 m. Pooled, the 12–15 m
  band is still the weakest (0.375), and richmond's 0.111 is most of that.

### Caveats (they travel with the numbers)

- **Small n.** Per-city cells run n = 3–20 on hard-only. gainesville's hard-only is 1/3 under
  per-pano. Read pooled rows and CIs, not per-city point estimates.
- **Not every pano has a measured height.** paterson 30,325 / 34,687, bend 65,866 / 78,560,
  gainesville 30,458 / 35,204, são paulo 18,601 / 22,741. The rest fall back to 2.6 m.
  Measured heights flagged by #44's QC gate (vintage deviation) are kept, as `per-pano`
  does by design; each report lists the counts.
- **são paulo stays low** (0.429 hard-only) under every arm. The flat-ground model is wrong
  on hills, as step 1 expected, and a per-pano height does not fix slope.
- **Current (gap-filled) runs.** The same arms on the runs as they stand today read the same
  way: pooled per-pano hard-only 0.495 [0.40, 0.59] (47/95), all-mined 0.625 [0.54, 0.70]
  (80/128); 2.6 m gives 0.333 / 0.510. The 9,524 gap-fill panos have no depth row, so they
  fall back to 2.6 m. Tables: `data/current/compare.md`.
- **Paterson's input is in the archive under its own name (gap closed 2026-09-29).** The
  frozen paterson file (sha256 `651226f9…`, 34,687 panos, including the 260-pano gap fill)
  was at first only in Jon's desktop checkout. On 2026-09-29 it was copied, with Jon's
  approval, to `runs/paterson/results.gapfill-2026-09-21.jsonl` in the makelab2 archive
  (read-only, byte-identical). The archive's `results.jsonl` beside it is still the
  pre-gap-fill 34,427-pano file and is not the input used here. The current (gap-filled)
  gainesville and são paulo files are still desktop-only; only the secondary `current/`
  tables use them.

### Runtime and cost

CPU only, no network, on Jon's desktop. Each arm takes 16–21 s wall-clock for all five
cities. No GPU, no paid API, and nothing to record in a ledger. A re-run with the same
inputs rewrites every `candidates.csv` byte-identically (checked).

### Reproduce

```bash
# Build the frozen runs root from the makelab2 archive (runs/<city>/{results.jsonl,depth/index.csv});
# paterson from runs/paterson/results.gapfill-2026-09-21.jsonl there (renamed to
# results.jsonl in FROZEN); richmond's per-rig
# table is git-tracked. Verify every sha256 against data/inputs.json before running.
FROZEN=/path/to/frozen   # <city>/results.jsonl, <city>/depth/index.csv, richmond/camera_heights.json
C="richmond paterson bend gainesville sao_paulo"
B="--runs-root $FROZEN --benchmark-root ../RampNet/benchmark"   # RampNet at 4a859f1
D=docs/figures/mined-precision/data/frozen
python scripts/mined_precision.py $C $B --out $D/h2.6
python scripts/mined_precision.py $C $B --out $D/h2.6_gsv2.2 --camera-height 2.6 2.2 2.2 2.2 2.2
python scripts/mined_precision.py $C $B --out $D/perrig_perpano \
    --camera-height per-rig per-pano per-pano per-pano per-pano
python scripts/mined_precision.py $C $B --out $D/auto --camera-height auto
python scripts/mined_precision_compare.py --cities $C --mapillary richmond \
    --arm step1_2.6=$D/h2.6 --arm step1_gsv2.2=$D/h2.6_gsv2.2 \
    --arm step2_perpano=$D/perrig_perpano --arm auto=$D/auto \
    --out $D/compare.csv --md $D/compare.md
# the gap-filled runs: --runs-root runs (a checkout holding them), --out data/current/<arm>,
# arms h2.6 / perrig_perpano / auto, labels 2.6 / perpano / auto
```

### Next

Phase 2 (below) replaced the flat projection with image-based placement.

## Phase 2 (2026-09-29): image-based placement of the mined target

**Question.** Step 2 could not move richmond, and richmond holds 20 of the 29 placement false
positives left after it. RampNet [#210](https://github.com/ProjectSidewalk/RampNet/pull/210)
found image-based transfers that place a ramp from one view into another better than flat
ground does, most clearly on Mapillary. If the miner places its target with one of those
instead of projecting the fused site, does mined precision rise?

**Design. Fixed and posted before any placement was scored**
([RampNet#158 comment](https://github.com/ProjectSidewalk/RampNet/issues/158#issuecomment-5894832666);
code in `44784ef`):

- **Same candidates, so the comparison is paired.** The 127 candidates are step 2's
  (frozen inputs, per-rig / per-pano frame). Each arm is scored candidate by candidate
  against the step-2 flat run (`step2_flat`).
- **Source rule.** An image transfer needs one view of the ramp to start from. It is the
  site's operational member from another pano whose camera is nearest the target camera;
  ties go to higher confidence, then pano_id. The rule never reads a verdict or the target
  pano's detections. Output: `frozen/perrig_perpano/<city>/sources.csv`, which carries no GT.
- **Views** are the #48 harness's (1024 × 768, 75°). The source view is centred on the
  source detection, and the target view on the step-2 projection.
- **Arms**, run through the harness's own `Context` / `run_arm`, which hide the answer:
  - `mapa_posed_pair`: MapAnything on the pair, given the #48 pose priors (auto height, pose
    off). Post hoc in #48. When this plan was fixed its fresh-pair re-test was running; it
    has since **confirmed** it (see Caveats).
  - `roma`: RoMa with the RANSAC ground homography. Its matcher and estimator were
    pre-specified in #48, but it inherits `lg`'s 5° ground band, which #48 lists as post hoc.
  - `roma_local`: post hoc in #48.
  - Neither RoMa arm was in #48's fresh-pair re-test.
  - A fallback, or a pixel that does not reach the ground, keeps the step-2 point.
- **Adjudication.** The placed pixel is raycast from the target camera in the step-2 frame,
  exactly as that pano's GT marks are. Match radius 5 m, same buckets, same rule, same
  Wilson CIs. Range bands use the candidate's site range, so the strata are the flat run's.
  `placement.csv` records per candidate whether the arm placed it and how far it moved.
- **Driver.** RampNet `scripts/analysis/mined_placement_158.py`, branch
  `analysis/mined-placement-158`. Predictions are copied to
  `figures/mined-precision/data/placement/`.

**Coverage.** `mapa_posed_pair` placed 127 of 127 targets. `roma` placed 114: 9 fell back
and 4 missed the ground. `roma_local` placed 116: 7 fell back and 4 missed the ground.
Median shift of a placed target from the site: 1.84 / 2.19 / 2.23 m.

### Richmond (Mapillary), ≤ 15 m

| arm | hard-only [95% CI] (k/n) | all-mined [95% CI] (k/n) | fp placed | fixed / broken vs flat | sign-test p |
|---|---|---|--:|---|--:|
| step 1, 2.6 m = step 2 flat | 0.310 [0.19, 0.46] (13/42) | 0.431 [0.31, 0.57] (22/51) | 20 | – | – |
| `mapa_posed_pair` | 0.342 [0.21, 0.50] (13/38) | 0.500 [0.37, 0.63] (25/50) | 19 | 4 / 2 | 0.69 |
| `roma` | 0.394 [0.25, 0.56] (13/33) | 0.608 [0.47, 0.73] (31/51) | 13 | 8 / 0 | 0.008 |
| **`roma_local`** | **0.406 [0.26, 0.58] (13/32)** | **0.627 [0.49, 0.75] (32/51)** | 12 | **9 / 0** | **0.004** |

*Fixed* means a false positive under flat placement that is correct under the arm; *broken*
is the reverse. Both are defined by the pre-registered rubric (5 m world match), so a "fix"
is any landing within 5 m of a verdict-true detection or missed mark, **not necessarily the
site's own ramp**; the post hoc attribution below shows how many are not. The sign test is
two-sided and exact, and it is **post hoc**: the plan said "paired candidate by candidate"
but named no test, and the test was added in `cd34e97` after the predictions existed.
Richmond `roma_local`'s p = 0.004 survives Bonferroni over the 3 arms (0.012) and over the 9
arm × group cells (0.035).

### Pooled over five cities, ≤ 15 m

| arm | hard-only | all-mined | fixed / broken vs flat (p) |
|---|---|---|---|
| step 1, 2.6 m | 0.327 [0.24, 0.42] (32/98) | 0.511 [0.43, 0.59] (69/135) | – |
| step 2 flat | 0.483 [0.38, 0.59] (42/87) | 0.605 [0.51, 0.69] (69/114) | – |
| `mapa_posed_pair` | 0.487 [0.38, 0.60] (39/80) | 0.637 [0.55, 0.72] (72/113) | 5 / 4 (1.00) |
| `roma` | 0.500 [0.39, 0.61] (37/74) | 0.678 [0.59, 0.76] (78/115) | 9 / 3 (0.15) |
| **`roma_local`** | **0.534 [0.42, 0.64] (39/73)** | **0.702 [0.61, 0.78] (80/114)** | 10 / 1 (0.012) |

**GSV only (pooled, ≤ 15 m):**

| arm | hard-only | all-mined | fixed / broken vs flat |
|---|---|---|---|
| step 2 flat | 0.644 (29/45) | 0.746 (47/63) | – |
| `mapa_posed_pair` | 0.619 (26/42) | 0.746 (47/63) | 1 / 2 |
| `roma` | 0.585 (24/41) | 0.734 (47/64) | 1 / 3 |
| `roma_local` | 0.634 (26/41) | 0.762 (48/63) | 1 / 1 |

No arm helps on GSV. On paterson, `roma` turns 3 correct labels into false positives. More
important for a miner, the image arms turn **true hard positives into `already_detected`**:
`roma_local` moves 7 GSV `tp` there (paterson 5, são paulo 2) and gains 4 elsewhere, so
**pooled `tp` falls 42 → 39**. The pooled hard-only rise (0.483 → 0.534) is false positives
leaving its denominator *and* hard positives being lost, not new misses found. The post hoc
attribution below shows where those 7 land.

### By range band, pooled, ≤ 15 m: step 2 flat → `roma_local`

| band | hard-only | all-mined |
|---|---|---|
| 0–8 m | 0.667 (14/21) → 0.667 (14/21) | 0.741 → 0.731 |
| 8–12 m | 0.500 (13/26) → 0.476 (10/21) | 0.629 → 0.686 |
| 12–15 m | 0.375 (15/40) → 0.484 (15/31) | 0.519 → 0.698 |

For richmond alone, the far band's all-mined goes 0.238 → 0.591 (5/21 → 13/22). The full
city × band × arm table, with both denominators, CIs and the paired transitions, is
[`data/frozen/compare_phase2.md`](figures/mined-precision/data/frozen/compare_phase2.md).

### Post hoc: which ramp do the "fixes" land on? (added after review, 2026-09-29)

**This subsection is not part of the pre-registered design.** It was added after the
independent review of this PR, which found that several richmond "fixes" land on a
detection that fusion assigns to a *different* site. The all-mined and hard-only numbers
above stand as measured under the pre-registered rubric. What this subsection changes is
what they are evidence *for*.

`scripts/mined_placement_attribution.py` re-runs the exact step-2 fuse and adjudication
(it refuses unless every recomputed bucket equals the committed `candidates.csv`). For every
candidate whose bucket changes, it finds the GT point that adjudicated it. When that point
is one of the target pano's own detections, it looks the detection up in the fused sites.
`fixed` and `tp_to_ad` are attributed where the placed point landed, and `broken` where the
flat point had matched. The categories:

- **own site:** the detection is a member of the candidate's own site. This is impossible by
  construction, because a pano is a candidate only if it has no member in the site. It is
  counted anyway, and it is 0.
- **other multi-pano site:** a different site that ≥ 2 panos corroborate, so per the fuse a
  distinct ramp. The table also counts the sharpest case: the source pano (the view the arm
  transferred *from*) is itself a member of the landed site. In that case the source view
  detects the two ramps separately.
- **singleton:** a site of that one detection. The fuse linked it to nothing, so whether it
  is the mined ramp or another one is not determined here.
- **none:** the adjudicating point is not a detection (a missed or unsure mark), or nothing
  is within 5 m.

**`roma_local`, the transitions that carry the headline** (full table for all arms:
[`data/frozen/attribution_phase2.md`](figures/mined-precision/data/frozen/attribution_phase2.md),
per candidate in the CSV beside it):

| group | transition | n | own site | other multi-pano site (source pano a member) | singleton | none |
|---|---|--:|--:|--:|--:|--:|
| richmond | fixed (fp → right) | 9 | 0 | 6 (1) | 3 | 0 |
| richmond | broken | 0 | – | – | – | – |
| GSV pooled | fixed | 1 | 0 | 1 (0) | 0 | 0 |
| GSV pooled | broken | 1 | 0 | 1 (0) | 0 | 0 |
| GSV pooled | tp → already_detected | 7 | 0 | 5 (5) | 2 | 0 |
| GSV pooled | already_detected → tp | 3 | 0 | 0 | 0 | 3 |

**Richmond's 9 fixes, one by one.**
- **All 9** move from `fp` to `already_detected`. The placed pixel lands on a target detection
  that is 5.4–11 m from the candidate site, and for 8 of the 9 it lands within 2° of it.
- **6 of the 9** land on a detection of another multi-pano site: 732→842 (5 panos), 232→414
  (3), 312→1081 (2), 509→683 (10), 1287→916 (8), 350→916 (8).
- **The other 3** land on singletons 8.1–8.2 m away.
- **At least 2 are demonstrably a different ramp.** Site 1287's own source pano
  (`2965822660266034`) detects both ramps separately, about 45° apart: one detection in site
  1287, the other in site 916. RoMa carried the click from the first onto the target's
  detection of the second, about 22° off it. Site 350 (3.4 m from site 1287) lands on the
  same target detection.
- **The other 7 are undetermined** without looking at the imagery. The fuse's site identity
  is not a ramp identity on richmond. Every one of the 27 flat `already_detected`
  candidates pooled, richmond's 9 included, is also on "another multi-pano site". That
  follows from the candidate rule (the target pano detected something near the site, and
  that detection did not join the site), whether the fuse split one ramp or the ramps are
  distinct.

**GSV's 7 lost hard positives.** In 5 of the 7 the landed detection belongs to a site that
the source pano *also* detects separately, 0.9–3.8 m from the candidate site (paterson
sites 4800, 4993, 5917; são paulo 674, 999). The other 2 (paterson 1348, 2579) land on the
same singleton detection about 5 m away. In each, the flat point had landed within 5 m of a
missed ramp the reviewer marked. The image transfer pulled it onto a neighbouring ramp that
the target pano already detected, on the same corner.

**Re-tally of the sign test.** The original count, 9 / 0 (p = 0.004), stands under the
rubric. Read as "placed on the mined ramp", the paired count lies between 0 / 0 and 7 / 0:
- Removing the 2 demonstrated wrong-ramp transfers leaves 7 / 0 (p = 0.016). This is the
  upper bound: no other fix is shown to be a wrong ramp.
- None of the 9 is shown to be the mined ramp, so the lower bound is 0 / 0. With no
  discordant pair the sign test is undefined there.
- 3 / 0 (p = 0.25) is a reading, not a bound: it counts the 3 singletons as the mined ramp.
  Their landed detections are 8.08–8.17 m from the candidate site, inside the 5.4–11 m of
  the 6 other-multi-pano landings, and their identity is undetermined (see the category
  definitions above).

**Strict sensitivity.** A cruder reading of all-mined counts every `already_detected` that
lands on another multi-pano site as wrong, for the flat run and each arm alike. Richmond
reads 0.255 (13/51) flat and 0.314 (16/51) with `roma_local`. Pooled it is 0.368 → 0.386,
and GSV pooled 0.460 → 0.444. This strict reading is a sensitivity reading, not a bound on
the precision of "a label on the mined ramp": it still counts a singleton
`already_detected` as correct, and any `tp` whose nearest point is a *different* missed
mark within 5 m. Every arm's gain over flat sits almost entirely in the gap between it and
the rubric's all-mined.

**What this means for the 5 m world match (a finding for step 3).** The adjudication rule
takes the nearest GT point within 5 m. On a dense corner that cannot tell the mined ramp from
a neighbour: paired corner ramps are 1–10 m apart, which is inside the match radius and
often inside the placement error. A placed point that moves 1–2 m toward a neighbour's
detection flips `tp` → `already_detected` (GSV above). A point that lands 5–11 m off the
site on a detected neighbour flips `fp` → `already_detected` (richmond above). Neither flip
says the placement got better or worse at finding *this* ramp. Two things follow for any
later step that adjudicates mined targets this way, step 3's peak-anchored targets included:

- report the own / other / none attribution beside every all-mined number;
- treat an arm's gain in `already_detected` as unresolved until the landed detection is
  shown not to belong to another corroborated site, above all one the source view detects
  separately.

### Reading against the pre-registered rule

- **Richmond, best arm (`roma_local`).**
  - hard-only 0.406 [0.26, 0.58]: *drop* on the point estimate, and the CI spans drop and
    visibility, so it is not decisive.
  - all-mined 0.627 [0.49, 0.75]: *visibility*, with a CI that dips just under 0.50, so it
    is not decisive either.
  - Under flat placement richmond read *drop* on both denominators.
- **Pooled, `roma_local`.**
  - hard-only 0.534 [0.42, 0.64]: *visibility*, not decisive.
  - all-mined 0.702 [0.61, 0.78]: *visibility*, decisive (the whole CI is inside the band).
- **GSV is unchanged.** The step-2 reading stands there.

### What the change is made of

- **On richmond, every fix lands on a detection the target pano already had, and it is
  not shown to be the mined ramp.** RoMa creates no new hard positive there: `tp` stays at 13
  in every arm, and all 8–9 fixed candidates move from `fp` to `already_detected`
  (9 → 18 / 19). An earlier version of this doc said the transfer "lands on that ramp",
  meaning the site's own. **That is retracted.** The post hoc attribution above finds 6 of the
  9 on a detection of another multi-pano site 5.4–11 m away, at least 2 of them demonstrably
  a different ramp, and 3 on singletons whose identity is undetermined. What holds: under
  the pre-registered rubric, where a label on any real ramp within 5 m is a correct label,
  the labels a `roma_local` miner ships on richmond are right 63% of the time instead of 43%
  (paired 9 / 0). Hard-only rises only because false positives leave its denominator
  (0.310 → 0.406). Whether image placement puts the label on the *mined* ramp more often
  than flat projection does is not shown; the bounds above run from 0 / 0 to 7 / 0 (3 / 0
  if the 3 singleton landings are taken to be the mined ramp).
- **RoMa and MapAnything on richmond.** On #48's Mapillary pairs the two were close
  (RoMa 1.75–1.87° gain, MapAnything 1.73°). Here MapAnything fixes 4 (2 onto another
  multi-pano site, 2 onto singletons) and breaks 2. RoMa's larger count is at least partly
  its readiness to snap onto an existing detection: 8 of `roma_local`'s 9 fixes land within 2° of
  one. That is what a feature matcher does, and a 5 m world match rewards it whether or not
  the detection is the mined ramp.
- **GSV does not need it.** After step 2's measured heights, the flat projection is already
  about as good as the image transfers on GSV, as in #48 (no GSV arm beat auto height except
  the posed MapAnything pair, by 0.63°).

### Caveats (they travel with the numbers)

- **Where each arm stands in #48.**
  - `mapa_posed_pair` is post hoc in #48, but #48's fresh-pair re-test
    ([RampNet#220](https://github.com/ProjectSidewalk/RampNet/pull/220), open;
    [results](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5895553631))
    has since confirmed `mapa_posed_pair` against `proj_height_auto` (Mapillary +1.32°
    [0.73, 2.34]), and it stays robust under #220's sensitivity reads (GSV α/3 lower bound
    ≥ +0.105°). It also confirmed `mapa_posed_corner`. `mapa_k_pair` is confirmed as
    pre-specified but borderline on GSV: under #220's review and
    [sensitivity reads](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5896570175)
    its GSV lower bound is 0.000 / −0.009 / −0.048, so that cell is not confirmed under any
    of them. On Mapillary `mapa_k_pair` was best (+1.89°); it was not run here.
  - `roma` and `roma_local` were **not** in that re-test, so neither is confirmed on fresh
    pairs. In #48 no matching arm beats auto on GSV. Their Mapillary gains (+1.75 / +1.87°)
    come from a stratum #48 did not screen for multiplicity.
  - `roma_local` is post hoc in #48. `roma`'s matcher and estimator were pre-specified
    there, but it inherits `lg`'s post hoc 5° ground band, so it is not fully pre-specified
    either.
  - All three were named here before scoring.
- **Richmond n is small.** 32–42 adjudicable candidates on hard-only, 59 in all. The paired
  sign test (9 / 0, p = 0.004, post hoc; see above) counts rubric fixes, not placements on
  the mined ramp. The unpaired CIs overlap.
- **The 5 m world match accepts a neighbouring ramp.** See the post hoc subsection. Richmond's
  and GSV's `already_detected` gains are partly or wholly transfers onto another detected
  ramp, and on GSV they cost 7 true hard positives.
- **One source view per candidate.** A different source rule (for example, the most
  confident member) could transfer differently. It was not tried, so that no rule was chosen
  on these scores.
- **Adjudication stays in world space.** The placed pixel is raycast from the target camera,
  as the GT marks are, so the step-2 frame still sets the metre scale of the 5 m match. For
  a target and a GT mark in the same pano this is close to a pixel comparison. It is not
  identical.
- **Views and arms are #48's harness and code at `e00fcda`,** except that the pair list is
  this one. The #48 caveats about reference noise do not apply here, because the reference
  is the reviewer's verdict, not a detection peak.

### Runtime and cost

| step | where | wall-clock | GPU-h | $ |
|---|---|---|---|---|
| `--emit-sources` + adjudication, per arm | desktop CPU | 13–16 s | 0 | 0 |
| `mined_placement_158.py build` | desktop CPU | 8 s | 0 | 0 |
| `cut-views` (254 views from 211 panos) | makelab2 CPU, 8 workers | 40 s | 0 | 0 |
| `mapa_posed_pair` (127 pairs, incl. model load) | makelab2 A40 (shared; 8.8 GB held by an unrelated process, checked first) | 134 s | 0.037 | 0 |
| `roma` + `roma_local` (127 pairs, cached matches) | desktop RTX 3070 | 204 s | 0.057 (upper bound) | 0 |

No Tillicum and no paid API. The two GPU runs have `paid: false` rows in RampNet's
`analysis_out/usage_log.jsonl`. RoMa (`romatch` 0.1.2, `kornia` 0.8.3, `loguru` 0.7.3) was
installed `--no-deps` into a scratch directory and put on `PYTHONPATH` beside RampNet's
`.venv`, as #48 did; the exact install lines and the match-cache location are in RampNet's
`docs/mined_placement_158.md`. The attribution pass is CPU only, about 16 s. MapAnything ran in makelab2's `crossview48_sfm/venv`.

### Reproduce (phase 2)

```bash
# labeler: the source rows (already committed) come from the step-2 command plus --emit-sources
python scripts/mined_precision.py $C $B --out $D/perrig_perpano \
    --camera-height per-rig per-pano per-pano per-pano per-pano --emit-sources
# RampNet (branch analysis/mined-placement-158):
python scripts/analysis/mined_placement_158.py build --sources-root <labeler>/$D/perrig_perpano \
    --labeler-root <labeler> --runs-root $FROZEN
python scripts/analysis/mined_placement_158.py cut-views \
    --archive-root /projects/makeabilitylab/sidewalk-auto-labeler/runs --out VIEWS   # makelab2
python scripts/analysis/mined_placement_158.py predict --arm mapa_posed_pair --views VIEWS
PYTHONPATH=<romatch pkgs> python scripts/analysis/mined_placement_158.py predict --arm roma \
    --views VIEWS --extra match_cache=CACHE
PYTHONPATH=<romatch pkgs> python scripts/analysis/mined_placement_158.py predict --arm roma_local \
    --views VIEWS --extra match_cache=CACHE
# labeler: adjudicate each arm, then the paired table
for a in mapa_posed_pair roma roma_local; do
  python scripts/mined_precision.py $C $B --out $D/place_$a \
      --camera-height per-rig per-pano per-pano per-pano per-pano \
      --placement docs/figures/mined-precision/data/placement/$a.jsonl --placement-label $a
done
python scripts/mined_precision_compare.py --cities $C --mapillary richmond \
    --arm step1_2.6=$D/h2.6 --arm step2_flat=$D/perrig_perpano \
    --arm mapa_posed_pair=$D/place_mapa_posed_pair --arm roma=$D/place_roma \
    --arm roma_local=$D/place_roma_local --paired-base step2_flat \
    --paired-arms mapa_posed_pair roma roma_local \
    --out $D/compare_phase2.csv --md $D/compare_phase2.md
```

```bash
# POST HOC attribution (added after review; refuses unless its buckets equal the committed ones)
python scripts/mined_placement_attribution.py $C $B \
    --camera-height per-rig per-pano per-pano per-pano per-pano \
    --arm mapa_posed_pair=docs/figures/mined-precision/data/placement/mapa_posed_pair.jsonl \
    --arm roma=docs/figures/mined-precision/data/placement/roma.jsonl \
    --arm roma_local=docs/figures/mined-precision/data/placement/roma_local.jsonl \
    --verify $D --mapillary richmond \
    --out $D/attribution_phase2.csv --md $D/attribution_phase2.md
```

Every output above is written with LF line endings, so a re-run on Windows is
byte-identical on disk (checked 2026-09-29 for all phase-2 and step-2 files under
`data/frozen/`).

The views are not published: 254 JPEGs, `views.tar` sha256
`8a8bd69029736ffa1207de69484826b453109046485de028fadf2cabd7cccad6`, on makelab2 at
`/homes/gws/jonf/mined158/views`. They regenerate from the archive with `cut-views`. A
GPU re-run can differ by a pair or two (#48 matching.md §2).

### Next

- **`mapa_posed_pair`** is now confirmed on fresh pairs in #48, and it does not lead here.
  `mapa_k_pair`, best on Mapillary in the re-test, was not run here (step 3 adds it).
- **Richmond now reads *visibility* on all-mined, not decisively, and *drop* on
  hard-only.** Under the rubric RoMa moves richmond's false positives onto detected ramps,
  but it is not shown to place the mined ramp, and it adds no new misses. So the next lever
  is about which targets are hard misses, not where they land:
  - the peak-anchored target definition from step 1's plan, which needs a richmond
    inference pass at the storage floor;
  - the visibility test the rule names for the 0.50–0.80 band.
- **A miner that uses `roma_local` on Mapillary and the flat projection on GSV** is what the
  rubric's all-mined numbers favour. Given the post hoc attribution, it is not yet
  supported as a *placement* choice. It has not been measured as one arm.
- **Adjudication on dense corners** needs the own / other attribution reported beside it
  (see the post hoc subsection) before any `already_detected` gain is read as better
  placement.

## Step 3 (2026-09-29): peak-anchored targets. Plan, fixed before inference and scoring

*(Timing, added 2026-09-29 after the review of #112. The plan was committed at 17:59:08Z
(7e1f391), before any inference. The RampNet#158 comment that posts it went up at
18:03:33Z, about two minutes after the richmond floor pass had started: that pass ended at
about 18:05Z after 214 s. So the plan was committed before inference and posted as it
started. When 7e1f391 was pushed is not recorded; it was on origin by 18:16:07Z. Scoring
began after all of this, at d23e2fc, 18:21:39Z.)*

**Question.** In phase 2, image placement raised richmond's all-mined under the rubric, but
it added no new misses, and its "fixes" are not shown to land on the mined ramp (see the
post hoc attribution above). *(Edited 2026-09-29 at the merge with the reviewed phase 2,
after inference and scoring: the plan as committed in 7e1f391 said image placement "made
richmond's shipped labels correct more often ... every label it fixed was one the model
already detected", wording the phase-2 correction withdrew. Only the description of phase 2
changed; the question, the definition and the gate below are as committed.)* Step
1's plan proposed a stricter target, where geometry proposes and the model localizes. Emit a
target only where the target pano holds a **sub-threshold peak** near the geometric anchor,
and put the target on that peak. Does that make mined targets genuine misses?

**Target definition** (`scripts/peak_anchor.py`):

- **Window:** the benchmark's per-pano match radius, 0.022 in RampNet's pano geometry (x
  scaled by 1024, y by 512, x cyclic at the seam), around the anchor pixel. That is about
  7.9°.
- **Peaks:** the target pano's stored peaks down to the storage floor (0.1), at most 50 per
  pano, from the published model (`projectsidewalk/rampnet-model`, paper weights).
- **Rule:**
  - If a peak ≥ 0.55 is in the window, the model already fires there, so the candidate is
    **not emitted**.
  - Otherwise, if any peak in [0.1, 0.55) is in the window, the miner **emits the
    highest-confidence one's pixel** as the target.
  - Otherwise (no response at all), the candidate is **not emitted**.
- **Anchors**, all named now:
  - **`peak_flat`** (primary): the step-2 flat projection of the site.
  - `peak_roma_local`: phase 2's `roma_local` placement, falling back to flat.
  - `peak_mapa_k_pair`: a new placement arm, MapAnything on the pair given intrinsics only.
    #48's fresh-pair re-test ([RampNet#220](https://github.com/ProjectSidewalk/RampNet/pull/220))
    found it the best Mapillary arm. `mapa_k_pair` is also scored on its own as a placement
    arm, like phase 2's. *(Status as of #220's final reads, added at the merge with the
    reviewed phase 2: confirmed as pre-specified but **borderline on GSV**, lower bounds
    0.000 / −0.009 / −0.048 under the
    [sensitivity reads](https://github.com/ProjectSidewalk/RampNet/issues/48#issuecomment-5896570175);
    best on Mapillary, +1.89°. The plan above is unchanged.)*

**Where the floor peaks come from:**

- **paterson, gainesville, são paulo:** these runs already store peaks to 0.1 (post-#28),
  and their frozen `results.jsonl` are used as they are.
- **richmond and bend:** these runs predate the floor, so both get one inference pass on
  makelab2's A40 (`scripts/floor_infer_archive.py`):
  - Scope: every benchmark-judged pano (richmond 124, bend 110; `step3/<city>_ids.txt`,
    taken from `benchmark/<city>/records.jsonl`). That covers all 39 + 10 candidate targets.
    *(Note added 2026-09-29 after the review of #112: 39 + 10 counts distinct target
    **panos**; they carry 59 + 12 candidates.)*
  - Pixels: the native-resolution JPEGs in the makelab2 run archive, read-only.
  - Pano blocks: unchanged, taken from the pinned archived `results.jsonl` (sha256 in
    `inputs.json`).

**Instrument check, a gate run before any scoring.** A pano reproduces if its re-inferred
peaks ≥ 0.55 match the pinned run's ≥ 0.55 detections: the same count, each within one
heatmap cell. If fewer than **95%** of a city's panos reproduce, that city's pass is not
used, and the check is reported instead of a precision.

**Adjudication** is unchanged. The emitted peak's pixel goes through
`mined_precision.py --placement` in the step-2 frame: raycast from the target camera, 5 m
match, same buckets. A candidate that is not emitted leaves both denominators and is
counted.

**What is reported, and how it is read:**

- Both denominators, per city, by range band, on the emitted subset.
- **Yield:** emitted / 127, per city.
- Beside every number, the **flat arm on the same emitted candidates** (paired), so that a
  gain is not just a different subset.
- The decision rule is the pre-registered one, unchanged: ≥ 0.80 build, 0.50–0.80 add
  visibility, < 0.50 drop, read on the point estimate and CI, both denominators, at ≤ 15 m.
- **Primary reading:** `peak_flat`, on richmond and pooled.
- **Yield changes what "build" means.** A definition that emits few targets can read
  "build" and still not be worth building. Yield is stated beside the reading, and no
  yield threshold is set.

**Cost** gets `paid: false` rows in RampNet's `analysis_out/usage_log.jsonl`.

### Step 3: addendum written after inference, before scoring (2026-09-29)

1. **Instrument check.** richmond passes: **124 of 124** panos reproduce their ≥ 0.55
   detections (267 before and after). bend **fails**: 77 of 110 reproduce (0.70; 265
   operational detections before, 257 after). The pixel sizes of bend's archived JPEGs match
   its pano blocks (101 are 16384 × 8192 and 9 are 13312 × 6656), so it is not a size
   mismatch. bend's run is the oldest (2026-07-03), and why its peaks move was not
   investigated. *(Post hoc, 2026-09-29: the size check compared against the max-zoom
   size, so it could not see the likely cause, zoom 3 in production against max zoom in
   the archive. See "Post hoc: why bend fails the gate".)* **Under the gate, bend's pass is
   not used.** Every step-3 arm, and the flat
   comparison beside it, is therefore over **richmond, paterson, gainesville and são paulo:
   115 of the 127 candidates**. Outputs are in `step3/<city>.floor.jsonl` and
   `step3/<city>.check.json`.
2. **Own-site read, secondary, added after the review of RampNet#219 and before any step-3
   scoring.** The reviewer found that 6 of phase 2's 9 richmond `roma_local` fixes were
   matched to a detection of a *different* fused site 5.4–11 m away. The 5 m world match
   accepts a neighbouring ramp on a dense corner. `--own-site` therefore reads each
   adjudication a second way:
   - A right adjudication (tp or already_detected) whose GT point is strictly closer to
     another site of the same fuse than to the candidate's own becomes `other_site`.
   - `other_site` is false under both denominators.

   It is reported beside the pre-registered read for every arm, flat and phase 2 included.
   It is not a replacement for that read.
3. **`mapa_k_pair` ran.** 127 pairs, 1 fell back, 50 s on the A40.

### Step 3 results (2026-09-29)

**Scope.** Four cities: richmond, paterson, gainesville and são paulo. bend's floor pass
failed the instrument gate (addendum above), so bend is excluded, and every "flat" and
phase-2 number in this section is recomputed on the same four cities (115 candidates). The
read is at ≤ 15 m. Each cell gives the point estimate [95% Wilson CI] (k/n).

#### Pre-registered read

| group | arm | emitted / cand. | hard-only | all-mined | flat on the SAME candidates, hard / all |
|---|---|---|---|---|---|
| richmond | step 1 = step 2 flat | 59 / 59 | 0.310 [0.19, 0.46] (13/42) | 0.431 [0.31, 0.57] (22/51) | – |
| richmond | phase 2 `roma_local` | 59 / 59 | 0.406 [0.26, 0.58] (13/32) | 0.627 [0.49, 0.75] (32/51) | 0.310 / 0.431 |
| richmond | `mapa_k_pair` (placement) | 59 / 59 | 0.448 [0.28, 0.62] (13/29) | 0.680 [0.54, 0.79] (34/50) | 0.310 / 0.431 |
| richmond | **`peak_flat`** | **16 / 59** | **0.769 [0.50, 0.92] (10/13)** | **0.769 [0.50, 0.92] (10/13)** | 0.769 / 0.769 |
| richmond | `peak_roma_local` | 17 / 59 | 0.615 [0.36, 0.82] (8/13) | 0.615 (8/13) | 0.643 / 0.643 |
| richmond | `peak_mapa_k_pair` | 19 / 59 | 0.600 [0.36, 0.80] (9/15) | 0.600 (9/15) | 0.625 / 0.625 |
| 4 cities | step 2 flat | 115 / 115 | 0.456 [0.35, 0.57] (36/79) | 0.587 [0.49, 0.68] (61/104) | – |
| 4 cities | `mapa_k_pair` | 115 / 115 | 0.525 [0.40, 0.64] (32/61) | 0.716 [0.62, 0.79] (73/102) | 0.456 / 0.587 |
| 4 cities | **`peak_flat`** | **20 / 115** | **0.688 [0.44, 0.86] (11/16)** | **0.706 [0.47, 0.87] (12/17)** | 0.688 / 0.706 |
| GSV (3 cities) | step 2 flat | 56 / 56 | 0.622 [0.46, 0.76] (23/37) | 0.736 [0.60, 0.84] (39/53) | – |
| GSV (3 cities) | `peak_flat` | 4 / 56 | 0.333 (1/3) | 0.500 (2/4) | 0.333 / 0.500 |

**Richmond `peak_flat` by band.** 0–8 m 0.800 (4/5), 8–12 m 0.800 (4/5), 12–15 m 0.667 (2/3).
Within 10 m it is 0.778 [0.45, 0.94] (7/9). The full city × band × arm table, with the
paired transitions, is [`data/frozen/compare_step3.md`](figures/mined-precision/data/frozen/compare_step3.md).

**Which candidates the rule withholds (`peak_flat`):**

| city | no peak in window | operational (≥ 0.55) peak in window | emitted |
|---|--:|--:|--:|
| richmond | 35 | 8 | 16 |
| paterson | 8 | 18 | 1 |
| gainesville | 5 | 2 | 1 |
| são paulo | 10 | 9 | 2 |

#### Reading against the pre-registered rule

- **One reading, not two.** For the peak arms the two denominators coincide by
  construction: the rule withholds every candidate with an operational peak in its window,
  so the emitted subset holds no `already_detected` in richmond and one per arm in the four
  cities pooled (gainesville's). In richmond hard-only therefore equals all-mined, and
  pooled they differ by that one candidate. "Both denominators" below is one reading, not
  two that agree. *(Added 2026-09-29 after the review of #112.)*
- **richmond, `peak_flat`.** 0.769 on both denominators, which is *visibility* on the point
  estimate. The CI [0.50, 0.92] spans drop, visibility and build, so the reading is **not
  decisive**. It is the highest richmond has read at any step. Yield is 16 of 59 candidates.
- **Four cities, `peak_flat`.** hard-only 0.688 and all-mined 0.706, both *visibility* and
  not decisive. Yield is 20 of 115.
- **GSV.** The definition emits almost nothing: 4 of 56 candidates.

#### What it is made of

- **The gain is selection, not relocation.** On the 16 candidates it emits in richmond, the
  flat target scores exactly the same (0.769). Moving the target onto the peak changed 2
  buckets and fixed none. What the peak test does is **choose** candidates: of richmond
  flat's 13 true misses it keeps 10, and of its 29 false positives it keeps 3. So the
  model's own sub-threshold response is a working precision filter, and it needs no verdict.
- **On GSV the filter removes the true misses too, and the reason is dense corners.** Of GSV
  flat's 23 true misses, `peak_flat` keeps 1. 15 of the 23 are withheld because an
  operational peak is in the 7.9° window, and in all 15 that peak is a reviewer-judged
  **true detection of a neighbouring ramp**, 2.1–6.9° from the missed ramp the candidate
  points at. The model produces no separate peak for the missed ramp, even at
  `min_distance=1`, although its heatmap there is at operational level; whether that is
  the neighbour's shoulder or a merged response to both ramps is not separated. The other
  7 have no peak at all. So GSV's low yield is **confounded by dense corners**; it is not
  evidence that sub-threshold response is Mapillary-specific. See "Post hoc: why GSV's true
  misses are withheld" below. *(Rewritten 2026-09-29 after the review of #112. The text
  first read this as "the model already fires here", which the post hoc diagnostic does
  not show. No number changed. Edited again 2026-09-29 after the final re-review: an
  interim version said the model "is not firing on the mined ramp", which overstates the
  other way.)*
- **Image anchors do not help the peak definition.** `peak_roma_local` (0.615) and
  `peak_mapa_k_pair` (0.600) are both below `peak_flat` on richmond. More of their windows
  land on an operational peak (21 and 25 against 8), which fits the review finding that
  those transfers often land on a *neighbouring* ramp's detection. The exclusion side has
  the same dense-corner problem (see the post hoc diagnostic below).
- **`mapa_k_pair` as a placement arm.** Under the rubric, 11 of richmond's false positives
  become correct and none break (a post hoc sign test gives p = 0.001). All 11 become
  `already_detected`, so, as with `roma_local` in phase 2, they are **not shown to land on
  the mined ramp**. The own-site read below removes most of them. This is a rubric reading,
  not evidence of better placement.

#### Own-site read (secondary; added before scoring after the #219 review)

| group | arm | hard-only | all-mined | `other_site` |
|---|---|---|---|--:|
| richmond | step 2 flat | 0.224 [0.13, 0.36] (11/49) | 0.255 (13/51) | 9 |
| richmond | `roma_local` | 0.250 (12/48) | 0.294 (15/51) | 17 |
| richmond | `mapa_k_pair` | 0.255 (12/47) | 0.300 (15/50) | 19 |
| richmond | **`peak_flat`** | **0.538 [0.29, 0.77] (7/13)** | **0.538 (7/13)** | 3 |
| 4 cities | step 2 flat | 0.263 [0.19, 0.36] (26/99) | 0.298 (31/104) | 30 |
| 4 cities | **`peak_flat`** | **0.471 [0.26, 0.69] (8/17)** | **0.471 (8/17)** | 4 |

The own-site read **is biased low by construction**, and it has to be read with that in
mind:

- An `already_detected` candidate is one where the target pano's own detection was fused
  into a *different* site. That detection's GT point is a member of that site, so it is
  nearly always closer to it. Under this read almost every `already_detected` becomes
  `other_site`: for the flat arm in four cities, 25 become 5.
- The read also cannot tell a different ramp from the same ramp split across two sites by
  fusion.

So it is a sensitivity reading biased low, not a bound and not a better estimate. It can
also still credit a different ramp that has no fused site of its own.

**It tests site attribution, not label correctness.** *(Added 2026-09-29 after the review
of #112.)* An `other_site` true miss is a ramp this pano missed that fusion put in a
different site. As a hard-positive **training label** it is still correct: the pixel is on a
real ramp the model missed. Only the site identity is wrong. So the own-site read answers
"did the target land on the mined site's ramp?", a placement and identity question. It is
not a bound on label precision in either direction. It also sees only emitted rows, so it
cannot see what the exclusion side withholds (see the post hoc diagnostic below). It is a distance test
on fused-site positions, which differs from phase 2's post hoc attribution
(`mined_placement_attribution.py`), which asks which site the landed *detection* belongs to.
Step 3's peak-anchored arms have almost no `already_detected` (0–1 per arm), so the
dense-corner flip that attribution addresses barely arises for them. What the read shows:

- **Most of the image arms' right adjudications do not survive it.** 17 of richmond's
  `roma_local` right adjudications and 19 of `mapa_k_pair`'s are closer to another site.
  This is consistent with phase 2's attribution: none of the 9 `roma_local` fixes is shown
  to be the mined ramp, and the mined-ramp count is between 0 / 0 and 7 / 0.
- **`peak_flat` loses 3 of richmond's 10 true misses to it.** Those 3 hold GT points closer
  to another fused site, which could be a split or a different ramp. Under this read
  `peak_flat` is still the best arm: 0.538 against flat's 0.224.

#### Caveats

- **Small n.** richmond `peak_flat` is 13 adjudicable candidates, and the CI is wide. Its
  Wilson lower bound is 0.497, just below 0.50, so the CI reaches into *drop*. *(Corrected
  2026-09-29 after the review of #112; this line first said the bound "sits on 0.50".)*
- **bend is excluded.** Its floor pass failed the gate. The cause was not known when this
  was written; the likely cause, found after the review of #112, is the pixel source (see
  "Post hoc: why bend fails the gate" below).
- **Yield.** The peak-anchored miner emits about 27% of richmond's candidates and 7% of
  GSV's. Scaled by #102's richmond yield (upper bounds of 1,210 targets at 10 m and 3,011 at
  15 m), that is roughly 300–800 targets. That is an extrapolation, not a measurement.
- **The floor pass is per pano.** Only the 124 judged richmond panos were re-inferred.
  Building the miner for real needs floor peaks over the whole run. *(Added 2026-09-29:
  the existing `results.f01.jsonl` already passes the gate on 9,089 of 9,091 richmond
  panos; see "Post hoc" below.)*
- **The window is the benchmark radius** (0.022), chosen before scoring and not tuned.
- **The own-site read was added after the review**, before scoring, and is secondary.

#### Runtime and cost

| step | where | wall-clock | GPU-h |
|---|---|---|---|
| richmond floor pass (124 panos) | makelab2 A40 (shared; checked first) | 214 s | 0.059 |
| bend floor pass (110 panos; failed the gate) | makelab2 A40 | 245 s | 0.068 |
| `mapa_k_pair` (127 pairs) | makelab2 A40 | 50 s | 0.014 |
| peak_anchor + 17 adjudication runs + compare | desktop CPU | ~3 min | 0 |
| makelab2 venv (labeler `requirements.txt`: torch 2.14.0+cu130, transformers 5.12.1) | makelab2 CPU | ~2 min | 0 |
| post hoc (review of #112): zoom-3 re-fetch + re-inference of 20 GSV panos | desktop **CPU** (no GPU) | ~12 min | 0 |
| post hoc: gate displacement, f01 check (9,091 panos), input sidecars | desktop CPU | ~1 min | 0 |

Total: 0.14 A40-hours, $0. The `paid: false` rows are in RampNet's
`analysis_out/usage_log.jsonl` (branch `analysis/mined-placement-158-step3`). The labeler has
no ledger. Nothing ran on Tillicum, no paid API was used, and nothing was written to the
shared archive.

#### Reproduce (step 3)

```bash
# makelab2: the floor passes (labeler branch analysis/mined-precision-step3; A = the archive)
for c in richmond bend; do
  python scripts/floor_infer_archive.py --results $A/$c/results.jsonl --panos $A/$c/panos \
      --ids docs/figures/mined-precision/data/step3/${c}_ids.txt --out $c.floor.jsonl
  python scripts/floor_infer_archive.py --results $A/$c/results.jsonl \
      --ids docs/figures/mined-precision/data/step3/${c}_ids.txt --out $c.floor.jsonl \
      --check --check-out $c.check.json
done
# RampNet (branch analysis/mined-placement-158-step3): mapa_k_pair, as phase 2's arms
python scripts/analysis/mined_placement_158.py predict --arm mapa_k_pair --views VIEWS
# desktop: peak-anchored placements, then adjudicate (plain and --own-site), then compare
DD=docs/figures/mined-precision/data; C4="richmond paterson gainesville sao_paulo"
for a in flat roma_local mapa_k_pair; do
  python scripts/peak_anchor.py --sources-root $DD/frozen/perrig_perpano --cities $C4 \
      --runs-root $FROZEN --peaks richmond=$DD/step3/richmond.floor.jsonl \
      $( [ $a != flat ] && echo --anchor-placement $DD/placement/$a.jsonl ) \
      --out $DD/placement/peak_$a.jsonl
  python scripts/mined_precision.py $C4 $B --camera-height per-rig per-pano per-pano per-pano \
      --placement $DD/placement/peak_$a.jsonl --placement-label peak_$a --out $DD/frozen/peak_$a
  # ... and again with --own-site --out $DD/frozen/peak_${a}__own; the flat and placement arms
  # likewise (5 cities) into perrig_perpano__own / place_<arm>[__own]
done
python scripts/mined_precision_compare.py --cities $C4 --mapillary richmond \
    --arm step2_flat=$DD/frozen/perrig_perpano --arm roma_local=$DD/frozen/place_roma_local \
    --arm mapa_k_pair=$DD/frozen/place_mapa_k_pair --arm peak_flat=$DD/frozen/peak_flat \
    --arm peak_roma_local=$DD/frozen/peak_roma_local \
    --arm peak_mapa_k_pair=$DD/frozen/peak_mapa_k_pair --paired-base step2_flat \
    --out $DD/frozen/compare_step3.csv --md $DD/frozen/compare_step3.md   # and the __own set
# POST HOC (after the review of #112). The exclusion diagnostic; --heatmap re-fetches the
# GSV target panos at zoom 3 (network) and runs the model (CPU is fine, ~35 s a pano; the
# committed run used the desktop CPU, torch 2.12.1+cu126, transformers 5.12.1)
python scripts/step3_exclusion_diag.py richmond $C4_GSV --runs-root $FROZEN \
    --benchmark-root ../RampNet/benchmark --camera-height per-rig per-pano per-pano per-pano \
    --verify $DD/frozen --peak-flat $DD/placement/peak_flat.jsonl \
    --peaks richmond=$DD/step3/richmond.floor.jsonl --mapillary richmond --heatmap HM \
    --out $DD/step3/posthoc_exclusion.csv --md $DD/step3/posthoc_exclusion.md
#   ($C4_GSV = paterson gainesville sao_paulo; HM = a copy of makelab2
#   /homes/gws/jonf/mined158/hm_zoom3/. Into $DD the script refuses unless HM holds exactly
#   the 20 files of step3/zoom3_heatmaps.sha256.json; an empty HM re-fetches from Google
#   and must be written elsewhere.)
# the gate diagnosis and the f01 check (no model, no network)
for c in richmond bend; do
  python scripts/step3_gate_diag.py displacement --city $c --results $FROZEN/$c/results.jsonl \
      --floor $DD/step3/$c.floor.jsonl --json-out $DD/step3/posthoc_gate_displacement_$c.json
done
python scripts/step3_gate_diag.py f01 --results $FROZEN/richmond/results.jsonl \
    --f01 runs/richmond/results.f01.jsonl --floor $DD/step3/richmond.floor.jsonl \
    --json-out $DD/step3/posthoc_f01_gate.json
# the floor passes' input sidecars (A/<city>/index.csv read from the makelab2 archive;
# SW = the recorded software versions as a JSON object)
for c in richmond bend; do
  python scripts/floor_infer_archive.py --results $FROZEN/$c/results.jsonl \
      --ids $DD/step3/${c}_ids.txt --out $DD/step3/$c.floor.jsonl --manifest \
      --index $A/$c/index.csv --imagery-manifest ../RampNet/benchmark/$c/imagery_manifest.json \
      --software-json SW --manifest-out $DD/step3/$c.floor.inputs.json
done
```

### Step 3: post hoc diagnostics (added 2026-09-29, after the review of #112)

None of this is part of the pre-registered step-3 design. The rule, the gate and every
number above stand as posted. Each item below answers a question the review raised, with a
committed script and committed output. One input is not committed: the exclusion
diagnostic's 20 zoom-3 heatmaps are hash-pinned and kept on makelab2 (see "Where the
heatmaps are" below).

#### Post hoc: why GSV's true misses are withheld

`scripts/step3_exclusion_diag.py` takes every flat `tp` at ≤ 15 m and re-runs the step-2
fuse and adjudication. It refuses to continue unless every bucket equals the committed
`perrig_perpano` run. It then finds the missed mark that adjudicated each candidate, and
re-applies the step-3 rule, which must agree with the committed `peak_flat.jsonl`. Output:
[`step3/posthoc_exclusion.csv`](figures/mined-precision/data/step3/posthoc_exclusion.csv) and
[`.md`](figures/mined-precision/data/step3/posthoc_exclusion.md).

| imagery | `peak_flat` status | flat `tp` | in-window operational peak judged **true** | its distance to the missed mark | within 10 heatmap cells of it |
|---|---|--:|--:|---|--:|
| GSV (3 cities) | operational peak in window | 15 | 15 of 15 | 2.1–6.9° | 10 of 15 |
| GSV | no peak in window | 7 | – | – | – |
| GSV | emitted | 1 | – | – | – |
| Mapillary (richmond) | operational peak in window | 0 | – | – | – |
| Mapillary | no peak in window | 3 | – | – | – |
| Mapillary | emitted | 10 | – | – | – |

- **The withheld GSV misses sit next to a detected ramp.** In all 15, the operational peak
  in the window is a detection the reviewer judged true. The reviewer also marked a
  separate *missed* ramp 2.1–6.9° from it, inside the 7.9° window. That is two ramps: a
  detected neighbour, and the missed ramp the candidate points at. The rule reads the
  neighbour's peak as "the model already fires here". Whether the model also responds to
  the missed ramp is the next bullet's question, and the data do not settle it.
- **The missed ramp has no peak of its own, even without suppression.** The 15 candidates
  point at 14 distinct missed marks on 12 panos: `paterson:2579` and `paterson:1348` share
  one mark on `zB7_9mtLQuVbRyZrttlMXg`. Counts below are per candidate. None of the 15
  has a stored sub-threshold peak within the window of its missed mark. The production
  extraction (`peak_local_max`, `min_distance=10`) keeps a pixel only if it is the maximum
  of the 21 × 21 cells around it, so 10 of the 15 marks (≤ 10 cells, about 3.5°, from the
  stronger peak) could not have held one. To check the other 5 and see what suppression
  hides, the script re-fetched the 20 target panos at **zoom 3 through the production
  path** and re-ran the published model on the desktop CPU. All 20 reproduce their stored
  peak sets, floor included, so these are the production heatmaps. Re-extracting every
  local maximum (`min_distance=1`): for all 15, the nearest local maximum to the missed
  mark is the neighbour's own peak. Yet the heatmap at the missed mark is at operational
  level (median 0.77 within 2 cells; 13 of 15 are ≥ 0.55). So the model produces no
  separate peak for the missed ramp; its response there belongs to one blob with the
  detected neighbour. **Whether that is the neighbour's shoulder or a merged response to
  both ramps is not separated here.** Two heatmap-profile checks in the final re-review of
  #112 (scratch, not committed) disagree: against the axis-averaged profile of 40 isolated
  peaks, about 10 of 15 marks sit above the isolated blobs' 90th percentile at that
  distance (merged); against the most generous direction of those blobs, all 15 fall
  inside (shoulder). *(Edited 2026-09-29 after that re-review; the previous text said
  both "shoulder" and "merges". No number changed.)*
- **Where the heatmaps are.** The `hm_*` columns rest on 20 zoom-3 heatmaps, one float32
  512 × 1024 array per GSV target pano. They are **not published**. Their sha256s are
  committed in
  [`step3/zoom3_heatmaps.sha256.json`](figures/mined-precision/data/step3/zoom3_heatmaps.sha256.json),
  and the files are on makelab2 at `/homes/gws/jonf/mined158/hm_zoom3/` (copied
  2026-09-29; remote sha256s verified equal to the manifest). Without those files,
  regenerating the `hm_*` columns means re-fetching the 20 panos from Google's unofficial
  GSV endpoints and re-running the model; a re-fetch cannot be proven equal to these
  hashes, only re-checked through `hm_reproduces` / `hm_same_peak_set`. The script refuses
  to write into the committed `data/` dir unless `--heatmap` holds exactly these 20 files
  with these hashes, so a re-run without them can no longer blank the columns.
- **Richmond has no such case.** None of its 13 flat true misses has an operational peak in
  the window. 10 of the 13 are emitted.
- **So the GSV/Mapillary gap is confounded by dense corners.** It is not evidence that
  sub-threshold response is Mapillary-specific. On GSV, the rule's exclusion mostly removes
  misses that sit beside a detected ramp, and the flat anchors on GSV land close to that
  neighbour. Whether that is GSV's tighter anchors or GSV's denser benchmark corners is not
  separated here.
- **This answers the review's question on the window.** The 0.022 window *is* wide enough
  to take in a neighbouring ramp's peak. On GSV that shows up on the exclusion side, which
  the own-site read cannot see because it looks only at emitted rows.

**Design implication, a follow-up and not implemented.** A miner that wants these misses
needs a rule that can tell "the model fires on *this* ramp" from "the model fires on a
neighbour":
- an exclusion radius tied to the missed ramp, for example an operational peak within a
  cell or two of the anchor rather than anywhere in the 7.9° window; and
- a target that need not be a peak, such as the anchor pixel itself when the heatmap there
  is high. On these 15, extracting peaks with a smaller `min_distance` would **not** help:
  even at `min_distance=1` there is no separate peak.

Either one would change the pre-registered definition, so it would need its own plan and
its own read. The diagnostic's three GSV groups are small (15 + 7 + 1 misses), and the
reviewer's missed marks are clicks, not precise positions.

#### Post hoc: why bend fails the gate

The likely cause is the **pixel source**, not the weights:

- **Production fed the model zoom-3 GSV imagery.** `panorama.py` fetches at
  `PREFERRED_ZOOM = 3`, and it had that value at 5805dd7 (2026-07-01), the code bend's
  production run used (the run is from 2026-07-03). Zoom 3 is Google's own 4096 × 2048
  rendition of a 16384-px pano. For the 9 bend panos that are 13312 px wide, zoom 3 is
  3328 × 1664, which production upscaled to 4096.
- **The floor pass read max zoom.** The archive holds each pano at max zoom, re-encoded as
  a JPEG at quality 95 (`scripts/export_benchmark.py`, `fetch_native`), and the floor pass
  resized that to the model's input. So the model saw different pixels.
- **The addendum's size check could not see this.** It compared the JPEG with the pano
  block's `width` / `height`, and those record the max-zoom size.
- **Richmond is unaffected.** Mapillary's production path resizes `thumb_original` to
  4096 × 2048 with PIL bilinear, and the floor pass's `transforms.Resize` does the same
  operation on a PIL image. The archive's richmond JPEGs are the `thumb_original` bytes.
- **The committed outputs fit a pixel-source difference** (`scripts/step3_gate_diag.py
  displacement`; output with the sha256 of both inputs in
  [`step3/posthoc_gate_displacement_richmond.json`](figures/mined-precision/data/step3/posthoc_gate_displacement_richmond.json)
  and [`_bend.json`](figures/mined-precision/data/step3/posthoc_gate_displacement_bend.json),
  added after the final re-review):

  | city | pinned operational detections | same heatmap cell | within one cell | exact confidence | median / max &#124;Δconf&#124; |
  |---|--:|--:|--:|--:|---|
  | richmond | 267 | 267 | 267 | 1 | 0.00002 / 0.00006 |
  | bend | 265 | 208 | 238 (90%) | 0 | 0.020 / 0.29 |

  A weights difference would not leave richmond reproducing to 1e-4. The two runs also
  used value-identical weights: bend's production predates the 2026-07-24 re-export
  (606a119), which `detectors.KNOWN_REVISIONS` records as paper weights, value-identical.
- **Supporting evidence from GSV at zoom 3.** The exclusion diagnostic above re-fetched 20
  paterson, gainesville and são paulo panos at zoom 3 through the production path. All 20
  reproduce their stored peak sets, floor included, on a different machine (desktop CPU,
  torch 2.12.1). So GSV reproduces when the pixels match.

**Consequence.** A GSV floor pass must re-fetch zoom 3 through `panorama.fetch_panorama`,
not read the archive. A bend floor pass would need that. None was run for this review; a
zoom-3 re-fetch of one failing bend pano would confirm the diagnosis directly. The gate did
its job: it caught a changed instrument before anything was scored.

#### Post hoc: `results.f01.jsonl` already passes the gate on richmond

`results.f01.jsonl` (#20, 2026-09-22; sha256 `e81954f4…`) is a full richmond re-inference
at the 0.1 floor, re-fetched from Mapillary. Running the gate's own check over every pano
against the pinned run (`scripts/step3_gate_diag.py f01`):

- **9,089 of 9,091 panos reproduce** (0.9998), well above the 0.95 gate.
- **On the 124 judged panos, its full peak sets equal the floor pass**: the same positions
  for every peak down to 0.1, and |Δconf| ≤ 5.2e-5. That covers the one thing the gate
  does not check, sub-threshold reproduction.

Output: [`step3/posthoc_f01_gate.json`](figures/mined-precision/data/step3/posthoc_f01_gate.json),
with the sha256 of all three inputs and the two non-reproducing pano ids. So a
richmond-wide floor pass is not needed: f01's peaks, with the pinned pano blocks, can stand
in for one. `results.f01.jsonl` is on Jon's desktop only and is not published; that is the
gap a replicator would hit.

#### Post hoc: floor-pass inputs by hash, and the environment

- **Input hashes.** `step3/<city>.floor.inputs.json` lists the sha256, byte size and pixel
  size of every image each floor pass read. The hashes come from the archive's `index.csv`,
  recorded when the archive was exported, and were not re-hashed at pass time. All 124
  (richmond) and 110 (bend) equal RampNet's `benchmark/<city>/imagery_manifest.json`, hash
  and size. So a replicator can run the pass on the published benchmark imagery (HF
  `projectsidewalk/rampnet-benchmark`) instead of the makelab2 archive.
- **Software.** The same file records the pass's environment: torch 2.14.0+cu130,
  transformers 5.12.1, torchvision 0.29.0+cu130, Pillow 12.3.0, scikit-image 0.26.0. These
  were read afterwards from the venv the pass ran in (makelab2
  `/homes/gws/jonf/mined158/venv`); the pass itself did not record them. The sidecars omit
  numpy; the same venv has numpy 2.5.3 (read 2026-09-29, after the final re-review; the
  sidecars are left as written). New passes write `<out>.software.json` themselves, with
  numpy, and `floor_infer_archive.py --manifest` writes the inputs file. A
  `--software-json` that omits any recorded package, numpy included, is now refused.

#### Post hoc: `mapa_k_pair.meta.json` provenance

As written by the run, the meta says `"pre_specified_for_158": false`, because RampNet's
`PLANNED_ARMS` then listed only the phase-2 arms. The step-3 plan named the arm before the
run and before scoring, so the file now also carries `"pre_specified_in": "step 3"` and a
dated `provenance_correction` (RampNet#222). This copy stays byte-identical to RampNet's.
No prediction changed.

#### Post hoc: `compare_phase2.md` layout

`mined_precision_compare.py` gained two columns in the paired table in 7e1f391
(`not emitted`, `base on same cand.`). `compare_phase2.md` is regenerated with them, so it
again regenerates byte for byte at HEAD. `compare_phase2.csv` and every number are
unchanged.

### Step 3: next

- **A richmond-wide floor pass** so that the miner can actually be built. The published
  model over 9,091 panos on the A40 is about 3.5 h. An existing re-inference,
  `results.f01.jsonl` (#20, 2026-09-22), is on Jon's desktop. It was re-fetched from
  Mapillary, not taken from the archive, so it was not used here. It would need the same
  instrument check before it could replace a new pass. *(Update 2026-09-29, post hoc: it
  passes that check on 9,089 of 9,091 panos and matches the floor pass's full peak sets on
  the 124 judged panos, so no new pass is needed; see "Post hoc" above.)*
- **The visibility test** the rule names for the 0.50–0.80 band, applied to
  `peak_flat`'s emitted targets.
- **A look at the 3 `other_site` true misses** under `peak_flat`, to tell a split from a
  different ramp.
- **bend.** Find out why the archived JPEGs do not reproduce its run before using bend in
  any floor-based result. *(Update 2026-09-29, post hoc: the likely cause is zoom 3 in
  production against max zoom in the archive; a bend floor pass needs zoom-3 imagery.)*
- **A peak rule that survives dense corners** (post hoc follow-up, not implemented): an
  exclusion tied to the missed ramp rather than the whole window, and a target that need
  not be a peak. A smaller `min_distance` alone would not recover the 15 withheld GSV
  misses. See "Post hoc: why GSV's true misses are withheld".

## Step 4 (2026-09-29): the Richmond miner and a rated gallery. Plan, fixed before any crop is cut

**Question.** On the 13 adjudicable benchmark candidates, step 3's `peak_flat` read 0.769
[0.497, 0.92]. That is *visibility*, but not decisively. Two things are needed to decide
it: (1) run the miner over the whole of Richmond, and (2) have Jon rate about 100 of its
labels. That sample is about 8× step 3's n, and it comes from panos no benchmark verdict
has seen.

### 1. Floor pass and miner

- **Floor pass.** Same as step 3, over all 9,091 panos of the pinned richmond
  `results.jsonl` (sha256 `109e7645…`):
  - published model at revision 606a119;
  - the makelab2 archive's native-resolution JPEGs, read-only;
  - pano blocks unchanged.

  It started on the A40 at 2026-09-29T20:22:48Z (`floor_infer_archive.py --workers 6`).
  - **Gate:** step 3's gate, unchanged. At least 95% of the 124 benchmark panos inside
    the pass must reproduce their ≥ 0.55 detections within one heatmap cell. If it fails,
    stop.
  - **Cross-check:** the step-3 review found that `results.f01.jsonl` (#20) meets the
    gate city-wide. The pass had already started, so f01 is not used in its place. It is
    compared with the pass instead: the share of panos whose full peak sets down to 0.1
    agree (same count, same cell, |Δconf| ≤ 1e-3). The share is reported whatever it is.
    Both files are pinned by sha256.
- **Miner.** This is step 3's `peak_flat` rule, unchanged, applied city-wide:
  - fuse the pinned run exactly as in step 3 (benchmark threshold, pose off, per-rig
    frame, which resolves to 2.6 m in Richmond);
  - take every site with ≥ 3 operational panos;
  - for each such site, take every run pano within 15 m that is not a member;
  - anchor on the flat projection of the site;
  - use the pass's peaks: a peak ≥ 0.55 in the 0.022 window means not emitted; otherwise
    emit the strongest [0.1, 0.55) peak; with no peak, not emitted.

  **Reported:** the emitted total; counts by band (0–8, 8–12, 12–15 m); counts by status;
  and how many emitted labels fall on benchmark panos.

  **Consistency check.** On the 124 benchmark panos, the city-wide miner must reproduce
  step 3's 16 emitted richmond targets, with the same pixels. If it does not, say so.
- **Known limitation, stated before the numbers (review of labeler#112, finding 1).** The
  rule's exclusion test drops a candidate when *any* ≥ 0.55 peak falls in its 7.9°
  window, including a neighbouring ramp's detection. `peak_local_max(min_distance=10)`
  suppresses a weaker peak within about 3.5° of a stronger one, so a missed ramp beside a
  detected one cannot hold its own peak. On GSV this dropped 15 of 23 true misses. In
  step 3, Richmond had none dropped that way, but city-wide it can happen on any dense
  corner. **If Richmond's yield or gallery precision disappoints, this is the first cause
  to check.** The rule is not changed mid-stream.

### 2. Gallery (RampNet `scripts/analysis/mined_label_check_158.py`)

- **Population:** every emitted label whose target pano is **not** one of the 124
  benchmark panos.
- **Sample:** 100 labels, stratified by range band (0–8, 8–12, 12–15 m).
  - Allocation is **proportional** to each band's share of the population, by largest
    remainder, so the pooled sample is self-weighting and one Wilson interval is valid
    for the pooled rate.
  - Within a band, draw uniformly without replacement with `random.Random(158)` over the
    labels sorted by (site_id, pano_id).
  - If the population has fewer than 100 labels, all of them are taken.
- **Instrument items: 10.** These are drawn with the same seed from step 3's richmond
  `peak_flat` targets that the benchmark adjudicated `tp` or `fp` (13 labels). Their known
  answer is Yes for a `tp` and No for an `fp` (Jon's earlier verdicts, through the 5 m
  world match).
  - They are flagged in the committed item file, but not on the cards or in the per-rater
    file. Cards are shuffled with seed 158.
  - They are scored separately, as agreement with his earlier verdicts, and are never
    part of the precision.
- **Cards.** Each card shows:
  - one ringed crop of the target pano, a 36 × 24° window centred on the emitted pixel,
    cut with the #48 harness's `cut_one`;
  - one unringed context crop: the source-rule view (the member pano whose camera is
    nearest the target's), centred on that member's detection.

  Cards show no model confidence, no range, no band and no instrument flag. The header is
  a card number plus an opaque id.
- **Question:** "Is there a curb ramp at the ring?" The answers are Yes, No and Can't tell.
  The rubric, stored in every per-rater file:
  - **Yes:** a curb ramp is at the ring or touching it. It may be partly hidden or faint,
    as long as it is visibly a ramp.
  - **No:** there is no curb ramp at that spot. The ring is on plain curb, a driveway,
    sidewalk, street or something else, and the nearest ramp, if any, is more than about
    one ramp width away.
  - **Can't tell:** the view does not let you decide (too dark, blocked, too far, too
    blurry). These are excluded from every rate and counted.
- **Rules:** answer from the ringed view; the context view is not rated; judge the spot
  under the ring, not whether a ramp is somewhere in the crop.
- **Per-rater file:** `analysis_out/mined_label_check_158/mined_label_check__<rater>.json`.
  It carries the question, rubric, rules, item list, manifest digest (items plus crop
  sha256s) and empty verdicts. The page asks for a rater id, so a second rater gets their
  own file. `rates` refuses a file whose digest, items, question, rubric or rules differ
  from the committed ones. `agreement` compares two raters.
- **Precision:** Yes / (Yes + No) over the 100 sample items, with a 95% Wilson interval,
  pooled and per band. It is read against #158's rule: ≥ 0.80 build, 0.50–0.80 visibility
  test, < 0.50 drop, judged on the point estimate and the CI.
  - A label is emitted only where no ≥ 0.55 peak lies in its window, so a Yes is a ramp
    the model did not detect there. The rate therefore reads as hard-only and all-mined at
    once (as in step 3, the two coincide by construction). The gallery cannot tell a missed
    ramp from a ramp detected just outside the window; that is stated beside the number.
  - The Can't-tell count and the instrument agreement are reported beside it.
- **Publishing:** crops committed under `benchmark/mined_label_check_158/crops/`, with a
  sha256 per crop in `manifest.json`. The gallery is a local HTML file,
  `benchmark/mined_label_check_158/gallery.html`, as for the #48 GT check.
