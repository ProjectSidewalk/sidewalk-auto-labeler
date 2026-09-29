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
rubric. Read as "placed on the mined ramp":
- Removing the 2 demonstrated wrong-ramp transfers leaves 7 / 0 (p = 0.016).
- Counting only the 3 singletons, the only fixes whose landed detection fusion did not
  assign to another corroborated ramp, gives 3 / 0 (p = 0.25).
- The first is an upper bound and the second a lower bound on how often RoMa placed the
  site's own ramp.

**Strict sensitivity.** A cruder bound on all-mined counts every `already_detected` that
lands on another multi-pano site as wrong, for the flat run and each arm alike. Richmond
reads 0.255 (13/51) flat and 0.314 (16/51) with `roma_local`. Pooled it is 0.368 → 0.386,
and GSV pooled 0.460 → 0.444. The rubric's all-mined and this strict reading bracket the
precision of "a label on the mined ramp". Every arm's gain over flat sits almost entirely
in the gap between them.

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
  than flat projection does is not shown; the bounds above run from 3 / 0 to 7 / 0.
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
    has since **confirmed** all three MapAnything arms against `proj_height_auto`, Mapillary
    included (`mapa_posed_pair` +1.32° [0.73, 2.34]). On Mapillary `mapa_k_pair` was best
    (+1.89°); it was not run here.
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

The views are not published: 254 JPEGs, `views.tar` sha256 `8a8bd690…cad6`, on makelab2 at
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
