# How often do RampNet and the crowd agree on Gainesville's curb ramps?

**Status:** measured 2026-09-24 against a frozen snapshot of the crowd labels; nothing was
submitted to the server. Revised the same day after review: the reading of the AI → crowd
gap was overstated and is corrected below, and the one-to-one matcher's effect on the
headline is now reported beside it.

**One caveat up front:** the "crowd" here is effectively one auditor. One account made 4,247
of the 4,498 CurbRamp labels in the pull. Every crowd figure below is therefore closer to
"AI vs one careful labeller" than to "AI vs a crowd", and none of it says how much labellers
disagree with each other. This is goal 2 of
[issue #31](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/31). The plan it
follows is the most recent plan comment on that issue. The tool is `scripts/agree_rate.py`,
and the full numbers are in `runs/gainesville/agree_rate/report.md` and its CSVs.

## Why Gainesville

Gainesville is the first city where this comparison is possible by construction. The run's
boundary is the Project Sidewalk deployment's opened area, not the city limits. The crowd
audited that area between April and August 2026, and about two thirds of the run's GSV
imagery is from 2026. So the AI and the crowd looked at the same streets, and in two thirds
of cases at the same pictures. No AI label has ever been submitted to `sidewalk-gainesville`.
That is deliberate: an AI label on the server would contaminate the baseline measured here
([#21](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/21)). The report
checks this and finds zero crowd labels sitting exactly on a stored detection pixel.

## Snapshot

The crowd corpus keeps growing, so every number here is relative to this pull. The tool
pulls it once and reuses the cached copy on every re-run. `--refresh` takes a new snapshot,
and every number changes with it.

| file | url | fetched (UTC) | sha256 | features |
|---|---|---|---|---:|
| `raw_labels_CurbRamp.geojson` | `/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson` | 2026-09-24T05:28:36Z | `a3cd657e8e028be59174d85fa344a237c95924642529f18067bc9c9ff6c279de` | 4,498 |
| `raw_labels_NoCurbRamp.geojson` | `/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson` | 2026-09-24T05:28:36Z | `a2c6c4fd08ea152aa3eba5a1c4e1176eb34fbf7a8ef19df63f5efeb506b4895e` | 59 |
| `regions.geojson` | `/v3/api/regions?filetype=geojson` | 2026-09-24T05:28:10Z | `25ce88180ef1995f0279ff8f09d0d3bd1a1580ecadb42e173d65fec76d0064cb` | 57 |

All three are from `https://sidewalk-gainesville.cs.washington.edu`. The geojson files are
not committed; their hashes above identify them.

**Run:** `results.jsonl` sha256 `9f4a57f3…ab857e8`, 37,435 panos. That is the 35,204 from the
main pass plus 2,231 from the 2026-09-21 gap-fill. 114 of the in-scope crowd labels sit on 55
of the gap-fill panos, which the tile scan never returned. The first version of the script
found the gap-fill records by comparing a raw line index with a unique-pano count. This file
has no blank or repeated lines, so the two coincide and the 114 stands (it was re-checked
independently); the code now counts records on both sides.

**Per-pano heights:** 30,458 of the 37,435 panos have a measured camera height; the other
6,977 use the 2.6 m default. Most of these heights come from `runs/gainesville/depth/index.csv`,
a local artifact that is not in git. Without it the per-pano rows fall back to the default for
most panos, so the script now warns when the file is missing and the report prints the count.

## Scope

- **Regions.** 40 of the snapshot's 57 regions make up the run's footprint. The other 17 were
  opened after the run, and 44 crowd labels in them are excluded. Of the 40, **37 are fully
  audited** (`completion_rate` = 1) and **3 are partial**: region 7 (0.907), region 30 (0.372)
  and region 59 (0.740). This does not match the "39 of 40" recorded on 2026-07-29, and the
  earlier split cannot be recovered from today's feed. So the tables use the snapshot's own
  `completion_rate`, and the threshold is a flag (`--full-completion`, default 0.999).
- **Crowd labels.** 4,454 CurbRamp labels are in scope: 4,253 in full regions and 201 in
  partial ones. There are also 59 NoCurbRamp labels. One account made 4,247 of the 4,498
  CurbRamp labels in the pull.
- **Validation.** 4,439 of the in-scope labels carry a vote from PS's own AI validator
  (`validator_type: AI`). Only 263 carry a human majority verdict. So the feed's `correct`
  field mostly reflects another model's opinion. The validation buckets below use human votes
  only (excluding the labeller's own vote), and `correct` is reported separately.

## Method

Two frames, reported side by side:

- **Pano frame (the headline).** A crowd label carries its pano id and pixel. When the run
  processed that pano, the crowd mark and the AI detections are compared in the same image.
  The matcher uses RampNet's own geometry (`rampnet.detection_eval`): x is scaled by 1024 and
  y by 512, which gives equal angle per unit on a 2:1 equirectangular image. x wraps at the
  seam, and matching is greedy one-to-one strictly within **0.022** x-units (7.9°).
  - The plan said "RampNet's 5% tolerance". RampNet's constant is actually 0.022, so that is
    the default here. The sweep still includes 0.05.
  - This frame involves no camera height, no raycast, no RampNet#101 scale error and no PS
    placement error. The report checks that the two pixel conventions really coincide: over
    2,325 matched pairs, the median signed offset is −0.15° in x and +0.15° in y, and p10 to
    p90 is within ±2.7°.
  - Its limit is coverage: **66.3% (2,952 / 4,454)** of in-scope crowd labels are on a pano
    the run processed.
  - The one-to-one constraint is a choice, and it moves the headline by 6 points. Where two
    crowd marks sit under one AI peak, only one of them can agree. The report therefore also
    gives the **any-detection** rate (a crowd label agrees when any operational detection on
    its pano is within the radius) and the count of **shadowed** labels (a detection is
    within the radius, but another crowd mark on the same pano claimed it).
- **World frame (a bound).** Every crowd label is placed where PS placed it (the feed's
  lat/lng, from the viewer's estimator,
  [SidewalkWebpage#4766](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4766)). It
  is then matched one-to-one (`eval_sites.match_one_to_one` semantics) to operational fused
  sites from `fuse_sites.fuse`.
  - This frame includes the third of labels on panos the run never processed.
  - It is a bound, not the headline, because two independent placement models sit between
    the two sides: PS's linear estimator and the labeller's flat-ground cotangent. A miss can
    therefore be geometry rather than detection.
  - A match can also be coincidence. With 8,000 AI sites on 47 km², a crowd label displaced
    25 m in a random direction still matches a site **8.5%** of the time (the chance floor
    in the report).
  - The raw union (any single operational raycast within the radius) is the loosest version
    and is reported beside it.

Tiers and heights:

- Every AI figure is given at the **operational tier 0.30**, which is what production ships
  (rig-masked). The **benchmark tier 0.55** is given beside it.
- Every world figure is given at the production camera height of 2.6 m. The `per-pano`
  heights from [#40](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/40) are
  an ablation.

## Results

### Pano frame, crowd → AI (headline)

Statistic: the share of crowd CurbRamp labels on a processed pano that have an AI detection on
the same pano within 0.022, matched one-to-one. The denominator is crowd labels on processed
panos.

| regions | operational 0.30 | benchmark 0.55 |
|---|---|---|
| 37 fully audited | **0.725** (2,058/2,838) [0.708, 0.741] | 0.604 (1,714/2,838) [0.586, 0.622] |
| 3 partial | 0.789 (90/114) [0.706, 0.854] | 0.702 (80/114) [0.612, 0.778] |
| all 40 | **0.728** (2,148/2,952) [0.711, 0.743] | 0.608 (1,794/2,952) [0.590, 0.625] |

The same labels without the one-to-one constraint (all 40 regions, radius 0.022):

| tier | one-to-one | any detection | shadowed |
|---|---|---|---:|
| 0.30 | **0.728** (2,148/2,952) | 0.788 (2,325/2,952) [0.772, 0.802] | 177 |
| 0.55 | 0.608 (1,794/2,952) | 0.655 (1,934/2,952) [0.638, 0.672] | 140 |

| radius (x-units) | degrees | operational 0.30 | 0.30, any detection | benchmark 0.55 |
|---:|---:|---|---|---|
| 0.011 | 4.0 | 0.643 | 0.692 | 0.547 |
| **0.022** | 7.9 | **0.728** | 0.788 | 0.608 |
| 0.033 | 11.9 | 0.752 | 0.814 | 0.624 |
| 0.05 | 18.0 | 0.770 | 0.834 | 0.637 |

So at 0.30 the pano-frame crowd → AI rate is 0.728 one-to-one, 0.788 with any detection, and
0.643–0.770 across the radius sweep. The matcher and the radius each move it by a few points,
so quote the headline with its matcher and radius.

**By validation** (pano frame, 0.30):

| label status | agree rate |
|---|---|
| human majority agreed | 0.705 (124/176) |
| human majority disagreed | 0.000 (0/3) |
| no human verdict | 0.730 (2,024/2,773) |
| feed `correct` = true | 0.765 (1,983/2,591) |
| feed `correct` = false | 0.242 (16/66) |
| feed `correct` = null | 0.505 (149/295) |

Human validation is too thin to split the misses: 176 agreed labels, and 3 disagreed. The
`correct` split mostly shows that PS's AI validator and RampNet tend to agree with each other,
not that either one is right.

**Pano frame, AI → crowd (lower bound only):** 0.493 (2,148/4,360) of operational detections
on crowd-labelled panos are claimed by a crowd mark on the same pano at 0.30, and 0.562
(1,794/3,192) at 0.55. This is a lower bound because an auditor labels each ramp once, from
one of the several panos that see it.

### World frame (bound)

Crowd → AI, one-to-one against operational sites, over all 4,454 in-scope labels:

| tier | height | 2.5 m | **5 m** | 7.5 m | union at 5 m |
|---|---|---|---|---|---|
| 0.30 | 2.6 m | 0.659 | **0.793** (3,532/4,454) | 0.848 | 0.941 |
| 0.30 | per-pano | 0.624 | 0.784 | 0.819 | 0.945 |
| 0.55 | 2.6 m | 0.587 | 0.717 | 0.767 | 0.883 |
| 0.55 | per-pano | 0.568 | 0.705 | 0.734 | 0.890 |

Chance floor at 5 m: 0.085 at 0.30 and 0.063 at 0.55. Per-pano heights do not raise
crowd → AI agreement (−0.9 points at 5 m, 0.30). The crowd's positions come from PS's own
estimator, not from any height model, so a better AI height does not bring the two closer.

**The two frames on the same 2,952 labels** (0.30, 0.022 / 5 m at 2.6 m):

| outcome | labels |
|---|---:|
| agree in both frames | 1,900 |
| pano frame only (lost to placement or the 25 m raycast envelope) | 248 |
| world frame only (this view missed; another view's site is near) | 513 |
| neither | 291 |

On these labels the pano frame gives **0.728** and the world frame gives **0.817**. The world
frame is higher because other views recover ramps that the crowd's own pano missed. The pano
frame therefore measures per-image agreement, and the world frame measures what the
multi-view system finds, loosened by geometry and chance. Labels on panos the run never
processed (1,502) agree at 0.745 in the world frame. They can only be scored there.

### World frame, AI → crowd (37 fully audited regions)

Denominator: operational fused sites that fall in a fully audited region.

| tier | height | one-to-one, 5 m | any label, 5 m | on a NoCurbRamp label, 5 m |
|---|---|---|---|---|
| 0.30 | 2.6 m | **0.399** (3,298/8,264) | 0.504 (4,167/8,264) | 0.005 (42/8,264) |
| 0.30 | per-pano | 0.454 (3,252/7,168) | 0.532 | 0.007 |
| 0.55 | 2.6 m | 0.516 (2,985/5,788) | 0.614 | 0.005 |
| 0.55 | per-pano | 0.616 (2,922/4,744) | 0.680 | 0.005 |

The per-pano rows look better mostly because the denominator shrinks. At 0.30, per-pano
heights leave 7,168 sites in the full regions instead of 8,264 (−1,096), while the matched
sites barely move (3,252 instead of 3,298, −46). So the rise from 0.399 to 0.454 is mostly
fewer unmatched sites, not more agreement.

Where the unlabelled 0.30 sites sit, by the distance to the nearest crowd CurbRamp label:

| nearest crowd label | sites | seen from ≥ 2 panos | has a member ≥ 0.55 |
|---|---:|---|---|
| matched one-to-one (≤ 5 m) | 3,298 | 0.839 | 0.895 |
| another site took the label within 5 m | 869 | 0.541 | 0.770 |
| 5–10 m | 1,514 | 0.225 | 0.671 |
| 10–20 m | 705 | 0.330 | 0.423 |
| none within 20 m | 1,878 | 0.332 | 0.454 |

Only 42 AI sites sit on a crowd NoCurbRamp label, so explicit disagreement is rare. Of the
4,966 unmatched sites, 869 (18%) are fragmentation: a crowd label within 5 m went to a
neighbouring site. Another 2,219 (45%) have a crowd label 5–20 m away, which is placement
error or a second ramp at the same corner. Only 1,878 (38%) have no crowd label within 20 m.

### Who is right when they disagree (RampNet GT)

RampNet's Gainesville benchmark (125 judged panos) can adjudicate part of the gap. The world
frame is used here at 0.55, the tier the bundle was judged at, with a 2.6 m camera height and
a 5 m radius. The GT was built during a RampNet review, so it is anchored to RampNet and
favours the AI. Read the crowd figures as lower bounds.

- **GT ramps in fully audited regions (195):** a crowd CurbRamp label is within 5 m for
  **0.749** (146/195) [0.683, 0.804]. An operational AI site is within 5 m for **0.923**
  (180/195) [0.877, 0.953].
  - Split: both 138, crowd only 8, AI only 42, neither 7.
- **Precision of AI sites:** 0.992 (118/119) where a crowd label is within 5 m, and **0.837**
  (36/43) [0.700, 0.919] where there is none. Both are measured on **0.55-tier** sites, the
  tier the bundle was judged at. The AI → crowd gap discussed above is at 0.30, and the
  precision of the sites the 0.30 tier adds is **not measured**.

How much of the 0.30 gap can crowd omissions explain? If the crowd's 4,253 full-region labels
cover 0.749 of the real ramps, the regions hold about 4,253 / 0.749 ≈ 5,680 ramps, so about
**1,430 were skipped** (1,030–1,970 over the recall's 95% CI). Each skipped ramp can account
for at most one unmatched site, so skipped ramps explain **at most about 29%** of the 4,966
unmatched sites (21–40%). The report computes these figures.

So the AI → crowd rate should not be read as AI precision, and it should not be read as crowd
recall either. A sizeable minority of the unmatched sites are real ramps the crowd skipped.
The rest are fragmentation (869 sites), placement beyond 5 m (a crowd label 5–20 m away), and
low-confidence single-view sites that nobody judged. Precision at 0.30 is unmeasured.

### Vintage: how much of the crowd → AI miss rate is imagery age

Each label is placed against the run's imagery in one of three ways:

- **Same pano**, if the run processed the crowd's pano.
- Otherwise, the run pano whose GSV `history` lists the crowd's pano.
- Otherwise, the nearest run pano within 30 m.

Months are counted as "ours minus theirs". All strata use the world rate (0.30, 2.6 m, 5 m),
so they are comparable.

| stratum | labels | world agree | excess misses |
|---|---:|---|---:|
| same pano | 2,952 | 0.817 | — |
| different pano, same month | 912 | 0.787 | +27.5 |
| crowd imagery newer than ours | 348 | 0.716 | +35.5 |
| crowd 1–18 months older | 166 | 0.699 | +19.7 |
| crowd 19–36 months older | 13 | 0.615 | +2.6 |
| crowd > 36 months older | 42 | 0.667 | +6.3 |
| no run pano within 30 m | 21 | 0.000 | +17.2 |

"Excess misses" are the misses beyond what the same-pano rate predicts for a stratum of that
size. Together they come to **108.8 of 922 world-frame misses (12%)**. This is an upper bound
on the vintage effect, because any other difference between the strata also lands here. So
imagery age explains at most about an eighth of the crowd → AI misses. Most misses happen on
the very same pictures.

Two further observations:

- The 912 labels on a different pano from the same month are on panos that are not in the
  run and not in any run pano's history. Presumably they come from the same drive but were
  not returned by the tile scan. This was not verified pano by pano.
- The 21 labels with no run pano nearby all miss. They are far from any imagery the run
  processed.

## Reading it

- **Headline:** on identical imagery, RampNet at the production tier marks **72.8%** of the
  crowd's curb ramps (pano frame, 0.30, 0.022, one-to-one; 95% CI 71.1–74.3%). Without the
  one-to-one constraint it is 78.8% (177 labels are shadowed by a neighbouring crowd mark),
  and across the radius sweep it runs 64.3–77.0%. At the old 0.55 tier it marks 60.8%.
  Coverage is 66% of the crowd's labels. The "crowd" is effectively one auditor (4,247 of
  4,498 labels), so this is agreement with one labeller's judgement.
- **Same image vs whole system:** counting other views, 79–82% of crowd labels have an AI site
  within 5 m. That is a bound, against an 8.5% chance floor.
- **The other direction:** only 40% of the AI's operational sites (0.30) have a crowd label.
  Crowd recall against the RampNet GT is 0.75, which implies about 1,430 skipped ramps: at
  most about 29% of the 4,966 unmatched sites. So a sizeable minority of the gap is real
  ramps the crowd skipped. The rest is fragmentation (869 sites), placement beyond 5 m, and
  low-confidence single-view sites nobody judged. AI sites with no crowd label are 84%
  precise at **0.55**; precision at 0.30, the tier this gap is measured at, is unmeasured.
- **Explicit disagreement** (an AI site on a crowd NoCurbRamp label) is 0.5%.

## Caveats

- **The crowd is effectively one auditor.** One account made 4,247 of the 4,498 CurbRamp
  labels. Agreement with the crowd here is agreement with that person's judgement, and
  disagreement between labellers cannot be measured from this corpus.
- **Crowd labels are not ground truth.** Only 263 carry a human verdict. The feed's `correct`
  field is dominated by PS's own AI validator.
- **The world frame mixes geometry with detection.** The frame comparison above shows both
  directions: 248 labels are lost to placement or range, and 513 are gained from other views.
- **The GT is RampNet-anchored and small** (195 pool ramps, and 43 judged sites with no crowd
  label).
- **Region completeness comes from today's feed**, not from the 2026-07-29 state.
- **Training data:** as noted in
  [#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56), open curb-ramp
  inventories (Portland's 45k ramps among them) entered RampNet's Stage-1 training.
  Gainesville is not in the training registry (the #31 contamination pre-check), so this is
  not leakage. It does mean the model learned what counts as a curb ramp partly from other
  cities' inventories. That is stated here and not re-measured.
- **Not covered here:** submission (goal 3, which stays last) and any PS-side change.

## Reproduce

```bash
python scripts/agree_rate.py gainesville \
    --server https://sidewalk-gainesville.cs.washington.edu
```

This reuses the cached snapshot if it is present. It needs `runs/gainesville/{results.jsonl,
area.geojson,manifest.json}` (pass `--run-dir` to read them in place) and optionally
`../RampNet/benchmark/gainesville`. It uses only the stdlib and shapely, with no GPU.
