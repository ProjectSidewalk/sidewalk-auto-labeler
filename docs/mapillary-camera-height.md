# Per-rig camera height for Mapillary (issue #53)

Mapillary serves no depth, so every Mapillary raycast uses `geo.DEFAULT_CAMERA_HEIGHT_M`
(2.6 m) whatever carried the camera. This study measures a camera height per capture rig
with the two instruments the repo already has, applies a decision rule fixed before any
number was produced, and ships the result as an **opt-in** height mode
(`--camera-height-m per-rig`). The default stays 2.6 m.

> **Revised after review** ([PR #82 review](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/82#issuecomment-5835367514)).
> Instrument A's bootstrap now keeps draw multiplicity; before, repeated site draws were
> merged. The tables were regenerated from a fresh run, and the CIs below are the new ones.
> The pooled k_net values are written to `runs/_summary/camera_height/k_net.csv` and to
> §3. The `reprojection_residual` note (§5), the direction-of-disagreement claim and the
> gate discussion (§4) are corrected. **The verdict is unchanged.**

**Verdict.** One of the seven rig classes passed the decision rule: Annapolis's Trimble MX7,
at 2.376 m. The resulting table then **failed the production gate** on three of its four
clauses. Every table ships with `recommended: false`, and the default does not change.

Reproduce (no GPU, no network; well under an hour on a laptop CPU for all five runs):

```bash
python scripts/mapillary_height.py richmond laurens clovis morgantown annapolis
python scripts/fuse_sites.py runs/annapolis --camera-height-m per-rig --out /tmp/s.jsonl
```

Outputs: `runs/<city>/camera_heights.json` (git-tracked, bound to `results.jsonl` by
sha256), `runs/<city>/camera_height/groups.csv`, and
`runs/_summary/camera_height/{gate.csv,instrument_c.csv,groups.csv,gate.txt}` (local).

## 1. Instruments

- **A. Bearing-only fixed point** (`fuse_sites.implied_heights`, #68). Two views of a site
  fix the ramp's position from bearings alone. Each view then implies a height,
  `range × tan(dip)`, that does not depend on the height model. Which pairs exist,
  however, depends on the association height. So the run is associated at 1.4, 1.6, 1.8,
  2.0, 2.3, 2.6 and 3.0 m, and the group's median implied height is fitted as
  `implied = a + b·h`. The fixed point is `h* = a / (1 − b)`. The 95% CI comes from 200
  bootstrap draws over sites, taken independently at each association height. Draws keep
  their multiplicity (`resampled_median`): a site drawn m times weighs m times. The slope
  `b` is the instrument's blind spot: at `b = 1` the implied height only follows the
  association height. Support counts are taken at the 2.6 m association.
- **B. Leave-one-out scale identity** (`reprojection_residual.fit_scale`, #76). The run is
  fused at 2.6 m, and a joint fit gives one scale `s` per group; groups that share sites
  are separated. The same fit is run on the error model's own simulated noise, which gives
  a **per-group** errors-in-variables null. The height is `h_B = 2.6·(1 − (s − s_null))`,
  that is, `2.6 / k_net`. The scale-plus-dip-offset fit (`range_offset_levers`) is
  reported beside it. `fit_scale` takes arbitrary group labels; it only indexes the name
  list, which was checked before building on it.
- **C. GT-anchored slope.** In the five judged splits, this is the slope of the left-out
  along-ray residual of reviewer references (box, missed and peak) on predicted range,
  cluster-robust by site, under 2.6 m and under the table. A correct height should flatten
  it.

Groupings: rig class, meaning a normalised (make, model) pair (`rig_key`: case-folded,
make dropped from the model, firmware tokens dropped, so Clovis's `GoPro Fusion
FS1.04.01.80.00` and `Fusion` are one class); creator; sequence; and speed class (walk
< 2.5 m/s, mixed 2.5–7, drive > 7, from the median speed between consecutive frames) as a
sub-split of rig class.

Tiers: A and B use production fusion (`FuseParams` defaults, operational tier 0.30,
`mask_rig` on, pose off), because the table feeds production fusion. C and the gate use
the benchmark tier (0.55, `mask_rig` off, pose off), which the judged bundles are keyed to.
The pooled B scale is the same at either tier (§4), which ties back to #76.

## 2. The rule (pre-registered on #53, unchanged)

A group's height replaces 2.6 m only if it passes all four rules:

1. **Support:** at least 100 panos with an implied height and at least 50 sites
   (instrument A), and at least 300 held-out views (instrument B).
2. **Identifiable:** A's slope `b ≤ 0.6`.
3. **Agreement:** `|h*_A − h_B| ≤ 0.25 m`. The group's height `h_g` is then the mean of
   the two.
4. **Material:** A's CI excludes 2.6 m and `|h_g − 2.6| > 0.2 m`.

The grain is rig class per city. If a rig class's qualifying sequences disagree (IQR of
`h*` above 0.4 m, where qualifying means meeting rule 1 at a quarter of the counts), the
class goes per sequence. Sequences that fail rule 1 then inherit the rig-class value. The
production gate compares per-rig with `off` (2.6 m) on the common site set, built the way
`eval_sites --pose-precondition` builds it (frozen association from `off`, production
25 m cap, benchmark tier). It requires:

- (i) p90 GT-to-site distance no worse by more than 0.1 m in any city where the table
  changes at least 10% of panos;
- (ii) a better median in at least 3 of those cities;
- (iii) off-pool recall at 2.5 m down by at most 1.0 point in any city;
- (iv) instrument C's |slope| no larger in any changed city.

Two interpretations were fixed before the numbers were run. A per-sequence group that
passes rule 1 but fails a later rule keeps 2.6 m rather than inheriting the rig-class
value. Instrument C's gate slope uses every reference kind (box, missed and peak).

## 3. Per-rig results

> **Superseded by §8.4 (#89, 2026-09-27).** Read as a fixed point, B has no validated
> estimator in any city, so no group passes and Annapolis's 2.376 m below is withdrawn.
> The table here is kept as #53 ran it, for comparison.

Instrument A medians across the association sweep (m):

| city / rig | 1.4 | 1.6 | 1.8 | 2.0 | 2.3 | 2.6 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| richmond / gopro max | 1.721 | 1.785 | 1.898 | 2.002 | 2.093 | 2.273 | 2.506 |
| richmond / nctech istar pulsar | 2.196 | 2.296 | 2.387 | 2.493 | 2.549 | 2.587 | 2.668 |
| richmond / unknown (`none/none`) | 2.175 | 2.271 | 2.450 | 2.517 | 2.575 | 2.672 | 2.768 |
| laurens / gopro max | 2.141 | 2.286 | 2.512 | 2.738 | 2.951 | 3.118 | 3.248 |
| clovis / gopro fusion | 2.269 | 2.310 | 2.347 | 2.384 | 2.407 | 2.428 | 2.498 |
| morgantown / gopro max | 2.193 | 2.240 | 2.276 | 2.324 | 2.355 | 2.390 | 2.431 |
| annapolis / trimble mx7 | 2.058 | 2.116 | 2.178 | 2.240 | 2.272 | 2.320 | 2.393 |

Richmond's four insta360 x4 panos contribute no pair. In the table below, A panos and A
sites are the support counts; the other columns are h\*\_A with its 95% CI, the slope,
h\_B with its 95% CI, h\_B with the dip-offset term, and the rule outcome.

| city / rig | panos | A panos | A sites | B views | h*_A (95% CI) | slope | h_B (95% CI) | h_B+offset | rule outcome |
|---|---:|---:|---:|---:|---|---:|---|---:|---|
| richmond / gopro max | 3,318 | 698 | 503 | 1,301 | 1.983 (1.939–2.043) | 0.486 | 2.377 (2.336–2.418) | 2.800 | fails 3, agreement (0.39 m apart) |
| richmond / nctech istar pulsar | 4,809 | 2,156 | 875 | 5,230 | 2.595 (2.577–2.623) | 0.286 | 2.680 (2.659–2.700) | 3.126 | fails 4, material |
| richmond / unknown | 960 | 320 | 179 | 704 | 2.708 (2.632–2.779) | 0.359 | 2.640 (2.580–2.701) | 3.042 | fails 4, material |
| richmond / insta360 x4 | 4 | 0 | 0 | 0 | — | — | — | — | fails 1, support |
| laurens / gopro max | 4,495 | 644 | 219 | 1,371 | **4.314** (3.802–4.841) | **0.723** | 2.949 (2.897–3.000) | 3.438 | fails 2, identifiable, and 3, agreement; suspect |
| clovis / gopro fusion | 72,776 | 4,665 | 1,250 | 6,841 | 2.420 (2.410–2.433) | 0.133 | 2.553 (2.535–2.572) | 3.018 | fails 4, material (h_g 2.487) |
| morgantown / gopro max | 51,692 | 5,008 | 1,186 | 9,065 | 2.352 (2.342–2.357) | 0.146 | 2.520 (2.504–2.536) | 2.757 | fails 4, material (h_g 2.436) |
| **annapolis / trimble mx7** | 53,232 | 13,174 | 2,766 | 24,228 | 2.257 (2.253–2.264) | 0.202 | 2.495 (2.486–2.504) | 2.784 | **passes: h_g = 2.376 m, sigma 0.119** |

**Sequence grain.** Two rig classes crossed the IQR threshold. Richmond's GoPro Max had 8
qualifying sequences with an h* IQR of 0.61 m, and Laurens's GoPro Max had 7 with an IQR of
0.74 m. In both, **no sequence met full rule-1 support**, so every sequence inherits the
rig-class value, which is 2.6 m because the class failed. Because no per-sequence group was
written, the tables' `grain` is `rig`; after review it is derived from the groups actually
written. Clovis (18 sequences, IQR
0.25 m), Morgantown (37, 0.13 m) and Annapolis (97, 0.13 m) stay at rig grain.

**Speed and creator splits** are reported only; they do not enter the table. In Richmond
the GoPro Max split separates by speed. Instrument A gives drive 1.84 m and mixed 2.25 m;
instrument B gives drive 2.27 m and mixed 2.52 m. By creator, `yo_scottie_oh` reads
1.76 m (A) against 2.22 m (B). In the four one-rig cities, rig, creator and city are the
same group, so those splits add nothing there. Annapolis's MX7 reads 2.27 m on drive and
2.21 m on mixed (A), so speed makes little difference for a vehicle rig. Full rows are in
`runs/<city>/camera_height/groups.csv`.

### Findings that do not depend on the verdict

- **B reads higher than A wherever A < 2.6 m.** The gap is 0.13 m in Clovis, 0.17 m in
  Morgantown, 0.24 m in Annapolis, and 0.39 m for Richmond's GoPro Max; for Richmond's
  Pulsar it is 0.09 m (A 2.595, B 2.680). In the two groups where A > 2.6 m, B reads lower:
  Richmond's unknown class (2.708 vs 2.640) and Laurens (4.314 vs 2.949). Relative to A,
  then, B is pulled toward about 2.6 m. The dip-offset variant of B moves every rig group up
  by a further 0.24–0.49 m. Rule 3 bites almost wherever the effect is large. The cause was
  not tested here; §7 (#87) later traced it to association pulling B toward the 2.6 m
  height it was read at. Two candidates remain open: B's residuals are truncated by the association
  gate, which attenuates `s` toward 0, and A's linear extrapolation past the swept range.
- **Laurens is not identifiable by A, as the plan predicted.** The slope is 0.72, and the
  implied height sits above the association height at every height from 1.4 to 2.6 m.
  The fixed point is therefore an extrapolation to 4.3 m, outside the 1.0–3.5 m sanity
  band. Instrument B places Laurens above 2.6 m as well (2.95 m, `k_net` 0.88). Laurens is
  the furthest above the default, but not the only group above it: Richmond's Pulsar reads
  2.680 on B, Richmond's unknown class 2.708 on A and 2.640 on B, and Richmond's pooled
  `k_net` of 0.989 corresponds to 2.63 m. The instrument was not tuned to change any of
  this.
- **The [#76](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/76) tie-back.**
  Instrument B's pooled `k_net` (null-corrected) is below; it is also written to
  `runs/_summary/camera_height/k_net.csv`. The benchmark-tier column reproduces #76's
  committed `k_null_corrected` (`docs/figures/reprojection-residual/data/range_slope.csv`).
  The plan's sizing section quoted #76's *uncorrected* `k` values (1.055, 0.930, 1.091,
  1.114, 1.132). The Mapillary errors-in-variables null is large (`s_null` of about 0.04 to
  0.08), so the corrected scale is much nearer 1.

  | city | k_net, production tier | s | s_null | k_net, benchmark tier | s | s_null |
  |---|---:|---:|---:|---:|---:|---:|
  | richmond | 0.9893 | 0.0518 | 0.0626 | 0.9893 | 0.0518 | 0.0626 |
  | laurens | 0.8817 | −0.0946 | 0.0396 | 0.8854 | −0.0758 | 0.0536 |
  | clovis | 1.0182 | 0.0834 | 0.0655 | 1.0182 | 0.0834 | 0.0655 |
  | morgantown | 1.0316 | 0.1024 | 0.0717 | 1.0374 | 0.1023 | 0.0663 |
  | annapolis | 1.0420 | 0.1167 | 0.0764 | 1.0420 | 0.1167 | 0.0764 |

## 4. Production gate

> **Superseded by §8.5 (#89).** With no group applied, no city changes and the gate fails
> vacuously on (ii). The #53 gate below is kept for comparison.

Only Annapolis changes: the MX7 value covers all 53,232 of its panos. In the other four
cities the per-rig arm is identical to `off` by construction. Common site set, benchmark
tier, 25 m cap:

| city | arm | sites scored | common ramps | median m | p90 m | mean m | R@2.5 | off-pool R@2.5 | C slope (all refs, n) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| annapolis | off (2.6 m) | 4,018 | 207 | 1.367 | 3.471 | 1.704 | 0.880 | 0.880 | −0.176 (223) |
| annapolis | per-rig (2.376 m) | 4,018 | 207 | **1.268** | **3.130** | 1.560 | 0.867 | **0.867** | **−0.207** (221) |
| richmond | both | 1,570 | 227 | 1.554 | 3.855 | 1.835 | 0.905 | 0.905 | −0.086 (226) |
| laurens | both | 186 | 151 | 1.879 | 3.829 | 2.067 | 0.536 | 0.536 | −0.186 (164) |
| clovis | both | 2,495 | 159 | 1.323 | 3.279 | 1.624 | 0.874 | 0.874 | −0.082 (151) |
| morgantown | both | 1,733 | 227 | 1.097 | 2.990 | 1.374 | 0.892 | 0.892 | +0.012 (234) |

Clause by clause:

- **(i) PASS.** Annapolis p90 improves by 0.341 m, so no changed city is worse by more
  than 0.1 m.
- **(ii) FAIL, on its wording.** The median improves by 0.100 m in Annapolis, but
  Annapolis is the only changed city and the clause as written needs 3. With fewer than
  3 changed cities it cannot pass. Read as "better in min(3, n_changed) changed cities",
  which is plausibly what was meant, it **passes**.
- **(iii) FAIL, within noise.** Annapolis off-pool recall at 2.5 m falls from 0.880 to
  0.867 (212 → 209 of 241 ramps), a drop of 1.2 points against a limit of 1.0. That is
  **3 ramps**.
- **(iv) FAIL, within noise.** Annapolis instrument C slope moves from −0.176 to −0.207,
  a change of −0.031 against slope SEs of about 0.032 and 0.038. The two arms are also
  scored on slightly different reference sets (n 223 vs 221). With independent
  references only, the slope moves from −0.222 to −0.286.

**Verdict: FAIL.** Every `camera_heights.json` is written with `recommended: false` and
the gate's clause outcomes. The per-rig mode ships opt-in, and the default stays 2.6 m.
Annapolis's lower height tightens placement on the common set (median −0.10 m, p90
−0.34 m). The gate fails under either reading of (ii), because (iii) and (iv) fail on their
own. Both of those failures are within noise, though. So the evidence says the gate could
not show a benefit beyond the p90/median tightening. It does not show that the lower height
is harmful.

## 5. Caveats

- The bootstrap CIs on h* are narrow (for example 2.253–2.264 m in Annapolis) because they
  resample sites only. They carry no model uncertainty: the linear fit, the extrapolation
  to the fixed point, and the pair filters. Rule 4's "CI excludes 2.6" is therefore easy to
  pass, and rules 2 and 3 are doing the real work.
- Rule 1 counts support at the 2.6 m association.
- **A pre-existing issue in `reprojection_residual.gt_anchored_rows`, filed as
  [#85](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/85) and not fixed
  here** (the file is outside this PR).
  - **What it does:** the function passes `params.apply_pose`, a mode *string*, to
    `geo.detection_ground_point`'s boolean `apply_pose`. Any non-empty string is truthy,
    including `'auto'` and `'off'`, so every posed pano's reference is raycast rotated.
    That covers GSV and Mapillary alike.
  - **When it arrived:** it came in when the string pose modes
    ([#74](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/74)) merged with
    [#76](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/76). #76's
    committed numbers predate it and are correct.
  - **What is affected:** any re-run of #76's GT-anchored rows from current `main`, for
    GSV and Mapillary cities alike. The pixel residuals and all GT-free output are not
    affected.
  - **Here:** this study hands that code copies of the panos with pitch and roll stripped,
    so every raycast here is flat.
- The table and the gate were measured on each run's `results.jsonl`. Laurens's live
  submission file is `results.raw.jsonl` (raw GPS positions); the table's sha256 binds it
  to `results.jsonl` only, and `load_results` refuses it for any other file.

## 6. What ships

- `geo.PER_RIG` and a separate branch in `geo.camera_height_for`. The `PER_PANO` branch is
  untouched.
- `fuse_sites.load_results(..., height_table=)` and `apply_height_table`. They refuse a
  table whose `results_sha256` is not the file's, and a run holding non-crowdsourced panos.
  `sites_meta.json` gains `camera_heights: {mode, table, sha256, grain, applied, fallback,
  applied_by_group}`. A group applies when its `applied` flag is set. Panos are keyed by
  `pano.sequence_id` or else `source_metadata.sequence`, the same expression the table is
  built with. A sequence whose panos report two rig classes goes to the majority class. `--camera-height-m per-rig` and `--height-table` are added to
  `fuse_sites.py`, and `per-rig` is accepted by `eval_sites.py`.
- `scripts/mapillary_height.py`, which contains instruments A, B and C, the rule, the
  table writer and the gate.
- `runs/{richmond,laurens,clovis,morgantown,annapolis}/camera_heights.json`, all with
  `recommended: false`. Only Annapolis's applies a height other than 2.6 m. (Since #89,
  §8, none does: every group is at 2.6 m.)
- Under the default, fused output is byte-identical: Richmond's `sites.jsonl` and
  `sites_meta.json` fused at 2.6 m match before and after the change.

## 7. The Richmond GoPro Max gap ([#87](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/87)), 2026-09-26

Instruments A and B read Richmond's GoPro Max 0.39 m apart (§3). The decision rules for
this section were pre-registered on #87 before any number was produced, and
`scripts/height_gap.py` implements them as written (its `RULE_*` constants). Everything
here is offline CPU and changes nothing in production. No `camera_heights.json` is
rewritten, `per-rig` stays opt-in with `recommended: false`, and the default stays 2.6 m.

**Corrections after review (PR #90, same day).** The first version of this section had
four problems. All are fixed below, and the overall verdict is unchanged.

- **Candidate 2's reading.** The first version read candidate 2 on the raw gap. The
  results commit made that switch after the numbers were in. The pre-registered reading
  is the fixed-point gap (the plan's E4: "both fixed points, from E1"). Under that reading
  candidate 2 is **partial**, not rejected.
- **The zero-noise mechanism.** The first version said the pull at zero noise comes from
  association alone. Most of it comes from the null, which is not noise-matched in the
  simulation (see E2).
- **h\*\_B and the Pulsar and unknown classes.** The first version said h\*\_B overshoots
  and that rule 3 then fails for Richmond's Pulsar and unknown class. That is largely an
  artifact of the line fit. It is now reported beside a local-crossing estimate.
- **h\*\_B's CI.** It is too narrow, and is now quoted with that caveat.

```bash
python scripts/height_gap.py sweep richmond --group gopro/max --sequences          # E1, E4
python scripts/height_gap.py sweep richmond laurens clovis morgantown annapolis     # E1, all rigs
python scripts/height_gap.py simulate richmond --group gopro/max --null-unmatched --true-height 1.8 2.0 2.2 2.4 --noise-scale 0 0.5 1 --seeds 2
python scripts/height_gap.py simulate richmond --group gopro/max --null-unmatched --true-height 2.0 --offset-px -4 -2 -1 0 1 2 4 --noise-scale 0.5 --seeds 2
python scripts/height_gap.py simulate richmond --group gopro/max --null-unmatched --true-height 2.0 --offset-px -4 -2 -1 0 1 2 4 --noise-scale 0.5 --seeds 4 --seed-start 2 --out runs/richmond/camera_height/gap/simulate_e3b_seeds.csv   # E3b robustness
python scripts/height_gap.py gt richmond --group gopro/max --heights 1.98 2.2 2.38 2.6   # E3a, E5
python scripts/height_gap.py sweep laurens --results results.jsonl results.raw.jsonl     # E6
python scripts/height_gap.py simulate laurens --group gopro/max --null-unmatched --true-height 2.6 3.0 3.4 --noise-scale 0.5 --seeds 2
python scripts/height_gap.py sweep paterson --group-by year                          # E6, GSV
python scripts/height_gap.py verdict richmond                                        # -> report.md
```

The tracked outputs are
`runs/richmond/camera_height/gap/{sweep,sequences,simulate,simulate_e3b_seeds,gt_offset,instrument_c}.csv`,
`report.md`, and `runs/laurens/camera_height/gap/{sweep,simulate}.csv`. The five-city
sweep and the Paterson arm are local: `runs/_summary/camera_height/gap_sweep.csv` and
`gap_paterson_year.csv`. With the jobs run in parallel, everything took about 35 minutes
of wall-clock time on a 16-core desktop. The review re-run reproduced every earlier
column exactly (the seeds are fixed) and added only new columns.

**Notation.** h\*\_A is A's fixed point. B(h) = h·(1 − (s − s\_null)) is B read after
associating at height h; #53 read B only at 2.6 m. B\_raw(h) = h·(1 − s) is the same
reading without the null correction. h\*\_B is the fixed point of the line
B(h) = a\_B + b\_B·h, as pre-registered. The **local crossing** is where B(h) = h,
interpolated between the two sweep heights that bracket it; it is reported beside h\*\_B
and is not used by any rule. The null s\_null is now averaged over 10 seeds, where #53
used one. For this rig the null's SD across seeds is 0.030 m at 2.6 m, under the plan's
0.05 m alarm, so #53's CI was not too narrow on that account. The 10-seed mean does move
B(2.6) from #53's 2.377 m to 2.396 m. So here **G = 2.396 − 1.983 = 0.414 m**, not the
0.394 m quoted in the issue.

**Caveat on every h\*\_B CI below.** The CI draws each association height independently.
But every height re-associates the same views, so their errors are correlated, and a
common-mode shift is exactly what the fixed point amplifies, by 1/(1 − b\_B) (3.4 for
this rig). The intervals are therefore too narrow. For example, 1.948–2.040 excludes the
local crossing, 2.07. The verdict does not depend on them: the gap is 0.011 m against a
0.25 m bar.

### Verdict

**Explained, by candidate 3: association pulls B toward the height it ran at.** Read as
a fixed point, B agrees with A to 1 cm; read at the local crossing, to 9 cm.

| candidate | outcome | the numbers the rule reads |
|---|---|---|
| 3. association pull | **confirmed** | Real data: h\*\_B 1.993 (95% CI 1.948–2.040, too narrow, see the caveat) vs h\*\_A 1.983; b\_B 0.708. The local crossing is 2.072 (+0.090). E2 at h\_true 2.0, noise 0.5 / 1.0: B(2.6) − h\_true +0.359 / +0.480; h\*\_A − h\_true −0.048 / +0.018; h\*\_B − h\_true +0.046 / +0.059. |
| 1. vertical peak offset | **rejected at the bar; the margin is inside 2-seed noise** | GoPro Max ε (peak minus box) is +1.62 px (95% CI +0.47 to +2.20). Inside that CI the fixed-point gap moves −0.015 m, the wrong way. The largest move for \|ε\| ≤ 4 px is 0.090 m (at ε = −4), against the 0.10 m bar. The pooled per-seed SD of the gap is 0.019 m, twice the 0.010 m margin. With six seeds (not the pre-registered input) the largest move is 0.087 m, still under the bar. |
| 2. mixed mountings | **partial** (pre-registered reading) | Median within-sequence fixed-point gap h\*\_B − h\*\_A is +0.129 m over 8 quarter-support sequences (confirm ≤ 0.10, reject ≥ 0.25); the mixture arm closes 56% of G (confirm ≥ 75%). Secondary, the raw gap B(2.6) − h\*\_A has a median of +0.392 m, which would reject. |
| E5, instrument C | not decisive | C slope at 2.6 m is −0.111 ± 0.150 (needs SE < 0.05), from only 36–41 GoPro Max references. |
| overall | **explained** | Only confirmed candidates count, so candidate 2's partial does not change it. Corrected fixed points 1.983 / 1.993, residual +0.011 m (needs ≤ 0.25). Simulated raw gap +0.407 / +0.462 m against G = 0.414 (needs within 0.10). |

### E1: B swept over the association heights

For the GoPro Max, B(h) at 1.4, 1.6, 1.8, 2.0, 2.3, 2.6 and 3.0 m is 1.548, 1.721, 1.870,
2.033, 2.197, 2.396 and 2.717. B tracks the association height with slope 0.71, steeper
than A's 0.49. Its fixed point is where the two lines meet.

| city / rig | h\*\_A | A slope | B(2.6) | b\_B | 1/(1 − b\_B) | h\*\_B, line (95% CI\*) | h\*\_B, local crossing | line − h\*\_A | local − h\*\_A | B(2.6) − h\*\_A (#53's rule 3) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| richmond / gopro max | 1.983 | 0.486 | 2.396 | 0.708 | 3.43 | 1.993 (1.948–2.040) | 2.072 | **+0.011** | +0.090 | +0.413 |
| richmond / nctech istar pulsar | 2.595 | 0.285 | 2.680 | 0.668 | 3.01 | 2.846 (2.809–2.891) | 2.744 | +0.251 | +0.149 | +0.085 |
| richmond / unknown | 2.708 | 0.359 | 2.666 | 0.815 | 5.41 | 2.997 (2.800–3.325) | 2.820 | +0.288 | +0.111 | −0.043 |
| laurens / gopro max | 4.314 | 0.723 | 2.977 | 1.036 | — | none (b\_B ≥ 1) | 3.749, extrapolated | — | −0.565 | −1.337 |
| clovis / gopro fusion | 2.420 | 0.133 | 2.545 | 0.494 | 1.98 | 2.541 (2.524–2.562) | 2.518 | +0.121 | +0.098 | +0.124 |
| morgantown / gopro max | 2.352 | 0.146 | 2.507 | 0.567 | 2.31 | 2.387 (2.370–2.405) | 2.431 | +0.035 | +0.079 | +0.154 |
| annapolis / trimble mx7 | 2.257 | 0.202 | 2.486 | 0.637 | 2.75 | 2.309 (2.298–2.321) | 2.337 | +0.052 | +0.081 | +0.229 |

\* Too narrow; see the caveat above.

With the line's h\*\_B in place of B(2.6), rule 3 passes for Richmond's GoPro Max, Clovis,
Morgantown and Annapolis. It fails for Richmond's Pulsar (by 1 mm) and its unknown class.
**That failure is mostly an artifact of the line fit, not of B.** The fixed point
a\_B / (1 − b\_B) amplifies any bias of the line near the crossing by 1/(1 − b\_B), 3–5×
here. B(h) is concave: the Pulsar's step slopes are 0.88, 0.91, 1.26, 0.53, 0.38 and
0.45. So a straight line through the sweep's centroid overshoots a crossing near the top
of the sweep. Read locally, the Pulsar crosses at 2.74 m and the unknown class at 2.82 m,
and both pass rule 3 (+0.149 and +0.111).

The local crossing is **not validated** either. On Richmond's simulated view graph it
reads high by 0.03–0.21 m (E2 below), more than the line does for h\_true 2.0–2.4. So
neither estimator is established. Choosing one is a pre-registration question for
[#89](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/89), not something
this table settles.

### E2: simulation on Richmond's real view graph

The simulation keeps Richmond's panos, headings, confidences and (pano, detection) keys,
and plants the 2.6 m fused sites as the true ramps. Each member detection is re-projected
at a known GoPro Max height; the Pulsar is placed at 2.6 m and the unknown class at
2.65 m. Noise is the Mapillary error model, scaled. Each pano gets a GPS shift, a heading
error, a pitch/roll tilt and a height jitter; each detection gets peak jitter. Every cell
averages two seeds.

| h\_true | noise | h\*\_A (slope) | B(2.6) | h\*\_B (b\_B) | B(2.6) − h\_true | h\*\_B − h\_true | h\*\_A − h\_true |
|---:|---:|---|---:|---|---:|---:|---:|
| 1.8 | 0 | 1.800 (0.000) | 2.193 | 1.910 (0.451) | +0.393 | +0.110 | +0.000 |
| 1.8 | 0.5 | 1.763 (0.282) | 2.284 | 1.904 (0.562) | +0.484 | +0.104 | −0.037 |
| 1.8 | 1 | 1.816 (0.631) | 2.448 | 1.945 (0.784) | +0.648 | +0.145 | +0.016 |
| 2.0 | 0 | 2.000 (0.000) | 2.293 | 2.060 (0.468) | +0.293 | +0.060 | +0.000 |
| 2.0 | 0.5 | 1.952 (0.257) | 2.359 | 2.046 (0.573) | +0.359 | +0.046 | −0.048 |
| 2.0 | 1 | 2.018 (0.606) | 2.480 | 2.059 (0.777) | +0.480 | +0.059 | +0.018 |
| 2.2 | 0 | 2.200 (0.000) | 2.434 | 2.232 (0.513) | +0.234 | +0.032 | +0.000 |
| 2.2 | 0.5 | 2.147 (0.278) | 2.446 | 2.194 (0.613) | +0.246 | −0.006 | −0.053 |
| 2.2 | 1 | 2.204 (0.609) | 2.531 | 2.193 (0.780) | +0.331 | −0.007 | +0.004 |
| 2.4 | 0 | 2.400 (0.000) | 2.577 | 2.466 (0.582) | +0.177 | +0.066 | +0.000 |
| 2.4 | 0.5 | 2.345 (0.297) | 2.542 | 2.405 (0.661) | +0.142 | +0.005 | −0.055 |
| 2.4 | 1 | 2.378 (0.608) | 2.558 | 2.333 (0.787) | +0.158 | −0.067 | −0.022 |

**The null is not noise-matched, and that changes how the zero-noise row reads.**
`mapillary_height._null_views` draws the errors-in-variables null at the error model's
full sigmas, whatever `noise_scale` the simulation injected. At noise 0, then, B is
corrected for noise that was never injected, and at 0.5 it is over-corrected. Only at 1.0
does the null match. The raw reading B\_raw separates association from that mismatch
(`simulate.csv`, `b_raw_at_*`). At h\_true 2.0:

| noise | B\_raw at 1.8 / 2.0 / 2.6 | B at 1.8 / 2.0 / 2.6 | B\_raw(2.6) − h\_true | B(2.6) − B(h\_true) |
|---:|---|---|---:|---:|
| 0 | 1.823 / 1.939 / 2.064 | 1.972 / 2.111 / 2.293 | +0.064 | +0.182 |
| 0.5 | 1.784 / 1.906 / 2.135 | 1.945 / 2.073 / 2.359 | +0.135 | +0.286 |
| 1 | 1.712 / 1.871 / 2.310 | 1.852 / 2.036 / 2.480 | +0.310 | +0.444 |

The zero-noise pull of +0.29 m (B(2.6) − h\_true) splits into about +0.06 m of
association (B\_raw(2.6) − h\_true) and about +0.23 m of mismatched null. The earlier
reading, "present at zero noise, so it comes from association and not from the null", had
the attribution backwards.

The mechanism, restated: **at matched noise (1.0), B is unbiased at the true height
(B(2.0) = 2.036) and associating at 2.6 m pulls it up by +0.44 m. At zero noise, the pull
on the raw reading is only about +0.06 m, so the pull grows with noise.** That fits the
chi-square gate: with more noise, more of the opposite-side views that carry the scale
signal fall outside the gate at a wrong height. Rule 3's simulation clause still holds at
both pre-registered levels (+0.359 at 0.5, +0.480 at 1.0, both ≥ 0.25). At 0.5 part of
that is the over-correcting null; the raw pull there is +0.135 m. This is also the
setting that mirrors the real data, if #76's finding holds that the model overstates
Mapillary noise about 2×: the real B is corrected by a full-sigma null too.

B's line fixed point recovers the truth to within about 0.07 m for h\_true 2.0–2.4 m and
overshoots by 0.10–0.15 m at 1.8 m. The local crossing does worse here, at +0.15 to
+0.18 m for h\_true 2.0 (`report.md`, E2). A recovers the truth to within 0.06 m
everywhere. At h\_true = 2.0 m the simulated raw gap B(2.6) − h\*\_A is +0.407 m at
noise 0.5 and +0.462 m at noise 1.0; the real G is 0.414 m.

### E3: a constant vertical peak offset

The box-minus-peak offset comes from the benchmark's reviewer boxes, bootstrapped by pano:

| rig | boxes | panos | box − peak dy, px (95% CI) |
|---|---:|---:|---|
| gopro max | 38 | 14 | −1.62 (−2.20 to −0.47) |
| nctech istar pulsar | 144 | 55 | −0.53 (−1.24 to +0.02) |
| unknown | 20 | 8 | −1.55 (−3.28 to −0.05) |

The GoPro Max peak sits about 1.6 heatmap px *below* the box centre. E3b plants an offset
ε in the E2 simulation (h\_true 2.0, noise 0.5):

| ε (px) | −4 | −2 | −1 | 0 | +1 | +2 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| h\*\_A | 1.680 | 1.825 | 1.885 | 1.952 | 2.015 | 2.080 | 2.216 |
| h\*\_B | 1.864 | 1.943 | 1.989 | 2.046 | 2.077 | 2.137 | 2.273 |
| h\*\_B − h\*\_A (2 seeds, the rule's input) | +0.184 | +0.118 | +0.104 | +0.094 | +0.062 | +0.057 | +0.057 |
| h\*\_B − h\*\_A (6 seeds, robustness) | +0.172 | +0.122 | +0.105 | +0.085 | +0.062 | +0.053 | +0.055 |

The offset moves both instruments together. An offset of the measured size, ε ≈ +1.6 px,
raises both by about 0.1 m, so it is a shared bias rather than a disagreement, and it
cannot open G. It does mean both instruments probably read this rig about 0.1 m high.

The rejection is at the bar, and not by a comfortable margin. The largest move for
\|ε\| ≤ 4 px is 0.090 m against the 0.10 m bar. The two pre-registered seeds disagree by
up to 0.05 m in a cell (ε +1: 0.039 / 0.085), and the pooled per-seed SD is 0.019 m, so
the 0.010 m margin is inside 2-seed noise. Four more seeds (`simulate_e3b_seeds.csv`,
seeds 2–5; not what the rule reads) give a largest move of 0.087 m (at ε = −4), with
per-cell SDs of 0.009–0.021 m. So candidate 1 is still rejected, still narrowly. The
part of the reading that is robust is the direction: inside the measured CI the offset
*closes* the gap slightly rather than opening it.

### E4: per sequence

For the 8 GoPro Max sequences with quarter support, ordered by B views, the
**fixed-point gap** h\*\_B − h\*\_A is +0.316, +0.114, +0.210, +0.069, +0.011, −0.375,
+0.145 and +0.179 m. Its median is **+0.129 m**, between the confirm bar (≤ 0.10) and
the reject bar (≥ 0.25). With
the mixture arm closing 56% of G (it needs 75%), candidate 2 is **partial** under the
pre-registered reading. A share of G may come from mixed mountings. It does not enter
"explained", which counts confirmed candidates only.

The raw gap B(2.6) − h\*\_A is reported as secondary: +0.415, +0.259, −0.151, +0.369,
+0.621, −0.700, +0.596 and +0.622 m, median +0.392 m, so G is present inside single
sequences. The first version of this section applied the rule to this raw gap, which
would reject. The switch came after the numbers, and has been reverted.
`height_gap.candidate2_gaps` feeds the rule, and a test pins that input. The mixture
arm weights each sequence's h\*\_A by its held-out views: the mean is 2.215 m, which
closes 56% of G. Over all 11 sequences with both readings, the mean is 2.181 m and closes
48%. Full rows are in `sequences.csv`.

### E5: instrument C on GoPro Max references

| height (m) | 1.98 | 2.2 | 2.38 | 2.6 | 1.993 (h\*\_B) |
|---|---:|---:|---:|---:|---:|
| C slope (SE) | −0.220 (0.107) | −0.067 (0.097) | −0.093 (0.093) | −0.111 (0.150) | −0.215 (0.108) |

Each slope rests on 36–41 references, all boxed or reviewer-missed, so none is the
model's own peak. The weighted zero crossing is 2.80 m (SE 0.63). The SEs are two to
three times the 0.05 bar, so C cannot choose between 2.0 m and 2.38 m; it is reported
only. The plan predicted positive slopes at 2.6 m: +0.23 for a 2.0 m rig and +0.09 for a
2.38 m one. Every observed slope is negative. That is not a sign error, and it has
precedent. `run_gt` uses #53's own regression: along = reference raycast minus site,
positive away from the camera (`mapillary_height.instrument_c`). #53's gate already
recorded negative C slopes, for example Annapolis −0.176 at 2.6 m and −0.207 at
2.376 m (§4). The plan's predicted slopes did not allow for that offset, which is why
its sizing missed.

### E6: second cases (report only)

- **Laurens.** A is not identifiable on either file: its slope is 0.72 on `results.jsonl`
  and 0.84 on `results.raw.jsonl`. B's line is as steep. b\_B is 1.04 on `results.jsonl`,
  so the line has no fixed point, and 0.93 on `results.raw.jsonl`, where h\*\_B is 4.22.
  The local crossings are 3.75 and 3.38 m, and both are **extrapolations**, since B stays
  above the identity up to the 3.0 m sweep top. Simulated on Laurens's graph at
  noise 0.5, a tall rig gives A a slope of 0.38 at 2.6 m, 0.50 at 3.0 m and 0.63 at
  3.4 m. **A tall rig alone does not reach 0.7** within 3.4 m, so Laurens's 0.72 needs
  something more. B's fixed point for h\_true 2.6, 3.0 and 3.4 m:

  | h\_true | line h\*\_B | local crossing |
  |---:|---:|---:|
  | 2.6 | 2.836 | 2.746 |
  | 3.0 | 4.450, extrapolated | 3.177, extrapolated |
  | 3.4 | 9.432, extrapolated | 4.412, extrapolated |

  The plan's risk note said heights at or above the sweep top must be quoted as
  extrapolations. The first version quoted 4.45 and 9.43 m as overshoots without that
  label, and most of that size is the line's 1/(1 − b\_B) (7.4 and 19). Inside the sweep,
  at 2.6 m, both estimators read high: +0.24 on the line and +0.15 at the local crossing.
- **Paterson (GSV, with a depth-measured height).** The 2025–26 rig has 15,155 panos with
  a depth height. It reads h\*\_A 1.975, B(2.6) 2.237, h\*\_B **1.843** (local crossing
  2.022), and a depth median of 1.862. So the same pattern holds on GSV: B read at 2.6 m
  sits 0.26 m above A, and both fixed-point readings land near A. For the 2018–2024
  vintages, h\*\_A is 2.46–2.56. The line's h\*\_B is 2.25–2.47 and the local crossing
  2.47–2.53, against a depth median of 2.35–2.37. The 2007–2017 vintages have few
  multi-view sites and A slopes of 0.60–0.93, so they are not identifiable.

### What follows

- #53's rule 3 compared unlike quantities: A as a fixed point and B at 2.6 m. Read as
  fixed points, Richmond's GoPro Max passes agreement, and with its §3 support, slope and
  CI it would then pass all four rules at h\_g ≈ 1.99 m. On the line fit, Richmond's
  Pulsar and unknown class fail agreement; at the local crossing they pass. Nothing here
  rewrites `camera_heights.json`.
- The follow-up,
  [#89](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/89), is to make B
  a fixed point in `mapillary_height.py` and re-run the #53 rule and gate. The review
  findings above set three requirements for its pre-registration:
  - pre-register the estimator: the local crossing, or the line over a sweep extended
    above 3.0 m;
  - report 1/(1 − b\_B) beside every h\*\_B;
  - run its simulation gate with a noise-matched null.

  Without these, a gate such as \|h\*\_B − h\_true\| ≤ 0.10 m measures the line's
  amplification and the null mismatch, not B.
- RampNet#101 measures the same identity at a single association height. Its 11% slope is
  therefore a lower bound on the range error, and the fixed-point form is the right
  estimator.

## 8. Instrument B as a validated fixed point ([#89](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/89)), 2026-09-27

§7 showed that #53's rule 3 compared unlike quantities: A as a fixed point, B read once at
the 2.6 m association. This section makes B a fixed point in `mapillary_height.py`,
validates the two candidate estimators in simulation on every city's real view graph, and
re-runs the #53 rule and production gate with the selected estimator. The default height
stays 2.6 m whatever comes out; `recommended` in each table records the verdict.

```bash
python scripts/mapillary_height.py --validate richmond laurens clovis morgantown annapolis   # the estimator gate
python scripts/mapillary_height.py richmond laurens clovis morgantown annapolis              # rule + gate + tables
```

`--validate` first times one real sweep per city (the budget clock, below), then runs the
simulation grid in a process pool with fixed seeds. It is resumable: per-seed rows land in
`runs/<city>/camera_height/estimator_validation_seeds.csv` as they finish, and the cell
means rule V reads are in `estimator_validation.csv` beside it. The measurement step
refuses a city with no `estimator_validation.csv`.

### 8.1 Pre-registration

The plan below was posted on #89
([comment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/89#issuecomment-5857916579))
and committed with the code and tests before any real-data or simulation number was
produced. It is reproduced verbatim, headings demoted.

**Implementation readings, fixed with the code (same commit, before any number).** Where
the plan leaves a choice open, the code makes it as follows. None of these was chosen
after seeing a number.

1. **Which groups rule V reads.** "The city's pooled rig class(es) that meet rule 1
   (`RULE_MIN_VIEWS_B`)" is read as: the rig classes (each pooled over its sequences)
   with at least 300 held-out views at the 2.6 m association **on the real data**,
   `other` excluded (`validation_groups`). Where a city has several (Richmond), rule V
   takes the maximum over classes as well as cells: an estimator is valid in the city
   only if it is valid for every class.
2. **"One real sweep"** for the budget rule is the real run's B sweep over
   `SWEEP_HEIGHTS_B` with the rig keying and 10 null seeds (`time_real_sweep`): the work
   one simulation cell repeats, less the re-synthesis. It prints support counts only, no
   h\*\_B. The desktop is shared with other sessions, so the clock is wall time under
   whatever load there was; it is recorded per city.
3. **Undefined and extrapolated.** The line is undefined when b\_B ≥ 1; the local
   crossing when no segment crosses the identity. The line is extrapolated when h\*\_B
   lies outside [1.4, 3.8] m; the local crossing when no adjacent pair of swept heights
   brackets it (`local_crossing`'s flag). A cell is undefined or extrapolated if any of
   its seeds is.
4. **Real-data extrapolation.** When the selected estimator's real-data h\*\_B lies
   outside the sweep, it is flagged (`extrapolated` in the table) and quoted as an
   extrapolation, but rule 3 still reads it: the plan adds no rule for that case.
5. **Sequence grain.** A per-sequence B (only when a rig class goes per sequence) is a
   separate joint sweep over `SWEEP_HEIGHTS_B` and uses the city's selected estimator.
6. **B(2.6)** is still reported (`h_scale` in groups.csv and the table, `h_at_2p6` in the
   `instrument_b` block), now with the null averaged over 10 seeds. It therefore moves
   slightly from #53's single-seed values (§7: Richmond GoPro Max 2.377 → 2.396).
7. **Seeds.** Cell seed *s* is `resynthesize`'s seed; every cell uses null seeds 0–9. The
   simulation reads B only; instrument A is not re-run in the grid.
8. **Cities.** The grid runs on the same five Mapillary runs as #53: Richmond, Laurens,
   Clovis, Morgantown and Annapolis. `height_gap.py simulate` now defaults to the
   noise-matched null and writes `simulate_matched.csv`; `--null-unmatched` reproduces
   #87's `simulate.csv`, which is not rewritten.

#### Plan (pre-registration): instrument B as a validated fixed point, then the #53 rule and gate re-run

Branch `mapillary-height-fixed-point-89`. Everything runs offline (CPU, no network, no GPU, no production change). The default height stays 2.6 m whatever comes out; `recommended` in each table records the verdict. The rule below is fixed before any real-data number is produced: the estimator-selection step and the gate are committed to the branch and posted here first, the real run comes after.

##### 1. What changes in `scripts/mapillary_height.py`

- **Noise-matched null.** `_null_views(sv, rng, noise_scale=1.0)` scales every sigma it draws (along, cross, GPS) by `noise_scale`. `instrument_b` and the sweep take the same argument and pass it through. `scripts/height_gap.py simulate` passes its own `noise_scale` to the null, so the null matches the injected noise (the 2026-09-26 correction on this issue, item 3). At noise 0 the null is drawn at zero sigma and contributes nothing.
- **Null averaged over seeds.** Instrument B's null is the mean over `NULL_SEEDS = 10` seeds (as #87 ran it), with its SD across seeds recorded (`null_sd_m`).
- **B swept over association heights.** New `SWEEP_HEIGHTS_B = (1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.4, 3.8)`: #53's seven heights plus two above 3.0 m so a rig anywhere in `SANE_HEIGHT_M` (1.0-3.5) is bracketed rather than extrapolated. Instrument A keeps `SWEEP_HEIGHTS` unchanged so #53's A numbers reproduce exactly. `b_at`, `local_crossing` and `b_fixed_point` move from `height_gap.py` into `mapillary_height.py` (height_gap imports them back; its tests keep passing).
- **Two estimators of h*_B, both always reported**, per row: `h_b_line` = a_B / (1 - b_B) from the line B(h) = a_B + b_B·h over `SWEEP_HEIGHTS_B`, with **`amplification = 1/(1 - b_B)` beside it**; `h_b_local` = the local crossing of B(h) = h between the bracketing sweep heights, with `extrapolated` flagged when no pair brackets it. Anything outside the sweep is quoted as an extrapolation. The old single reading B(2.6) stays as `h_b_at_2p6` for comparison with #53.
- **CI on h*_B** (parametric draws as in `b_fixed_point`) is printed with the independence caveat: the heights share views, so the draws understate the common mode and the interval is too narrow.

##### 2. The estimator gate (simulation only, pre-registered)

Neither estimator is validated (this issue, 2026-09-26). So each is validated per city on that city's **real view graph** with `height_gap.py`'s existing re-synthesis (`planted_sites` / `resynthesize`: the 2.6 m fused sites are the true ramps, every member re-projected at a known height), extended from one group to the whole run: **every group is planted at the same `h_true`**.

- Grid per city: `h_true ∈ {1.8, 2.2, 2.6, 3.0}` × `noise_scale ∈ {0.5, 1.0}` × 2 seeds, null noise-matched, null seeds 10. The cell value is the mean over seeds of `h*_B - h_true` per estimator, on the city's pooled rig class(es) that meet rule 1 (`RULE_MIN_VIEWS_B`).
- **Rule V (validation).** An estimator is VALID in a city iff over all 8 cells `max |mean error| ≤ 0.10 m` and no cell is undefined (b_B ≥ 1) or extrapolated.
- **Selection.** Both valid → the line (the proposal on this issue). One valid → that one. Neither → B has no validated fixed point in that city: its groups are reported with both estimators but **rule 3 reads `b_unvalidated` and fails** (conservative), so the group cannot pass and stays at 2.6 m.
- Budget rule, declared now: if one real sweep of a city takes longer than 15 minutes on this desktop, that city's grid drops to 1 seed per cell (timed on the real sweep, which reveals no simulation result). The runtime per city is recorded in the report.

##### 3. Re-run the #53 rule and the production gate

With the selected estimator's h*_B in place of B(2.6): rules 1-4 (`decide_group`, unchanged constants, `RULE_MAX_DISAGREE_M` 0.25 on |h*_A - h*_B|), the per-sequence split, then `height_gate` / `gate_verdict` exactly as #53 ran them. Tables are re-issued: `runs/<city>/camera_heights.json` (git-tracked, sha256-bound) gains an `instrument_b` block per group (`estimator`, `h_star`, `amplification`, `extrapolated`, `validated`, `h_at_2p6`) and a top-level `estimator_validation` block; `recommended` follows the gate. The five-city groups.csv and the gate CSVs are regenerated. Expected direction, stated now so the reading is honest: on #87's numbers rule 3 passes for GoPro Max, Clovis, Morgantown and Annapolis under either estimator and Richmond's Pulsar / unknown class under the local one; the gate is the open question.

##### 4. Files

- `scripts/mapillary_height.py`: the above; a `validate` step (writes `runs/<city>/camera_height/estimator_validation.csv`) and the report lines.
- `scripts/height_gap.py`: import the moved helpers; `simulate` gains the noise-matched null (its committed `simulate.csv` is NOT rewritten; a re-run writes `simulate_matched.csv` beside it so #87's verdict inputs stay as they were).
- `tests/test_mapillary_height.py`: `_null_views` at noise 0 returns the views unchanged and at 0.5 halves the spread; line vs local estimator on a synthetic concave B(h) (the line overshoots, the crossing lands, extrapolation flagged past the sweep); `amplification`; rule V on boundary rows (0.10 exactly passes, one undefined cell fails, neither-valid → `b_unvalidated`); `decide_group` with the new row shape; `camera_heights.json` round-trip with the new blocks and `fuse_sites --camera-height-m per-rig` still loading it.
- `docs/mapillary-camera-height.md`: new §8 with this pre-registration verbatim, then the results; §3/§4 tables re-issued with the old ones kept for comparison; CLAUDE.md's per-rig paragraph updated.
- Regression: `height_gap.py sweep richmond --group gopro/max --sequences` re-run and diffed against the committed `runs/richmond/camera_height/gap/sweep.csv` (old columns identical; new columns only). `fuse_sites.py runs/annapolis` default output byte-identical to main (the table is opt-in).

##### 5. Commit order

(1) code + tests + the pre-registration section of the doc, no numbers; (2) validation + real-data results, tables, doc findings; results comment here. PR opened for review; nothing merged by the agent.

##### Out of scope

Instrument A's sweep, the gate constants, any default change, GSV runs, `reprojection_residual.py`, RampNet.

### 8.2 Verdict

**Neither estimator of B's fixed point passes rule V in any of the five cities.** So in
every city rule 3 reads `b_unvalidated` and fails, no group passes the #53 rule, and every
`camera_heights.json` applies 2.6 m everywhere. That includes Annapolis's MX7, which #53
applied at 2.376 m: its agreement clause used B(2.6), and B(2.6) is not a fixed point
(§7). The production gate then has no changed city. It fails on clause (ii), vacuously
("median better in 0 changed cities, needs 3"), and every table is re-issued with
`recommended: false`. The default stays 2.6 m, as it would have whatever came out.

What fails rule V is B itself at the model's full noise, not the choice of estimator. At
noise 0.5 the local crossing recovers the planted height within 0.071 m in four cities
(0.008 m in Laurens) and within 0.135 m in Richmond. At noise 1.0 both estimators read
**high at low heights**, by +0.25 to +0.53 m at h\_true 1.8, in every city and every rig class, with the error
falling as the true height rises (−0.28 to +0.18 m at 3.0 m). The line is worse than the
local crossing in most noise-0.5 cells, as §7 predicted from its amplification.

| city | real sweep | seeds / cell | rig classes validated | line: max \|mean error\| (0.5 / 1.0) | local: max \|mean error\| (0.5 / 1.0) | selected |
|---|---:|---:|---|---|---|---|
| richmond | 30 s | 2 | gopro/max, nctech/istar pulsar, unknown | 0.462 / 0.525 | 0.135 / 0.444 | none (`b_unvalidated`) |
| laurens | 5 s | 2 | gopro/max | 0.280 / 0.361 | 0.008 / 0.418 | none |
| clovis | 30 s | 2 | gopro/fusion | 0.113 / 0.370 | 0.028 / 0.324 | none |
| morgantown | 42 s | 2 | gopro/max | 0.255 / 0.268 | 0.067 / 0.252 | none |
| annapolis | 99 s | 2 | trimble/mx7 | 0.250 / 0.344 | 0.071 / 0.306 | none |

No cell of either estimator was undefined or extrapolated, so every failure is on the
0.10 m error bar alone. The budget rule never bit: the slowest real sweep took 99 s
against 900 s. The whole `--validate` step (five timing sweeps, then 80 cell-seeds on 10 worker
processes) took about 14 minutes of wall time, and the measurement plus gate about
34 minutes (Richmond 370 s, Laurens 53 s, Clovis 224 s, Morgantown 296 s, Annapolis
1,072 s, then the gate). The desktop was shared with other sessions throughout.

### 8.3 Rule V cells

Mean error over two seeds, h\*\_B − h\_true in metres, as **line / local**. Bold is
outside ±0.10 m. Every row is the city's real view graph re-synthesized with every pano at
h\_true; B is swept over `SWEEP_HEIGHTS_B` with a noise-matched null averaged over 10
seeds. Inputs: `runs/<city>/camera_height/estimator_validation{,_seeds}.csv`.

| city / rig class | noise | h_true 1.8 | h_true 2.2 | h_true 2.6 | h_true 3.0 |
|---|---:|---:|---:|---:|---:|
| richmond / gopro/max | 0.5 | **−0.191** / −0.056 | **−0.194** / −0.085 | **−0.352** / **−0.103** | **−0.462** / **−0.135** |
| richmond / gopro/max | 1.0 | **+0.310** / **+0.257** | **+0.121** / **+0.141** | **−0.137** / +0.010 | **−0.276** / −0.052 |
| richmond / nctech/istar pulsar | 0.5 | −0.008 / +0.039 | +0.010 / +0.026 | **−0.104** / +0.022 | **−0.144** / +0.013 |
| richmond / nctech/istar pulsar | 1.0 | **+0.525** / **+0.444** | **+0.330** / **+0.299** | **+0.187** / **+0.240** | **+0.115** / **+0.182** |
| richmond / unknown | 0.5 | −0.056 / +0.005 | −0.014 / −0.019 | **−0.123** / −0.022 | **−0.163** / −0.016 |
| richmond / unknown | 1.0 | **+0.496** / **+0.385** | **+0.260** / **+0.254** | **+0.147** / **+0.139** | −0.003 / +0.093 |
| laurens / gopro/max | 0.5 | −0.058 / −0.001 | **−0.102** / −0.006 | **−0.234** / −0.008 | **−0.280** / −0.001 |
| laurens / gopro/max | 1.0 | **+0.361** / **+0.418** | **+0.179** / **+0.258** | +0.038 / **+0.130** | −0.044 / +0.049 |
| clovis / gopro/fusion | 0.5 | −0.056 / −0.006 | +0.011 / −0.020 | −0.072 / −0.028 | **−0.113** / −0.027 |
| clovis / gopro/fusion | 1.0 | **+0.370** / **+0.324** | **+0.218** / **+0.210** | +0.077 / **+0.104** | +0.014 / +0.062 |
| morgantown / gopro/max | 0.5 | **−0.116** / −0.028 | **−0.113** / −0.049 | **−0.211** / −0.055 | **−0.255** / −0.067 |
| morgantown / gopro/max | 1.0 | **+0.268** / **+0.252** | +0.098 / +0.097 | −0.054 / +0.034 | **−0.127** / −0.028 |
| annapolis / trimble/mx7 | 0.5 | **−0.117** / −0.029 | −0.098 / −0.050 | **−0.198** / −0.058 | **−0.250** / −0.071 |
| annapolis / trimble/mx7 | 1.0 | **+0.344** / **+0.306** | **+0.156** / **+0.159** | −0.005 / +0.069 | −0.094 / −0.008 |

**Reading the noise-1.0 bias.** B's slope b\_B is 0.36–0.84 across cells and is steeper at
noise 1.0 (0.69–0.84) than at 0.5 (0.36–0.68): with more noise, association pulls B more
strongly toward the height it runs at (§7's mechanism). Where the true height is low,
every association height above it pulls B up, and the crossing moves up with it. The
matched-null re-run of #87's E2 cell (`simulate_matched.csv`, Richmond GoPro Max,
h\_true 2.0) shows the same thing at one rig. Seed-mean B at the true height is
1.951 (noise 0.5) and 2.036 (noise 1.0), so B is nearly unbiased *at* the true
association. But the local crossing reads 1.877 and 2.151, and the line 1.782 and 2.059.
At noise 0.5 the null is now matched: #87's full-sigma null had over-corrected B(2.6) to
2.359, while the matched null gives 2.198.

**The noise level decides it, and the real noise level is not known.** #76 found the
error model overstates Mapillary noise about 2×. If that holds, the real data sit near
the noise-0.5 rows, where the local crossing would pass everywhere except Richmond
(−0.103 and −0.135 m for GoPro Max at 2.6 and 3.0 m). The rule was fixed at both noise
levels before any number, precisely because the real level is uncertain, and it is not
re-read here. A follow-up that measures the real noise scale could license a narrower
grid; this section does not.

### 8.4 Per-rig results, re-issued (§3's table, B as a fixed point)

A reproduces §3 exactly (A's sweep and bootstrap are unchanged). B(2.6) now averages 10
null seeds, so it moves a few mm from §3. Both h\*\_B estimators are reported with the
line's amplification 1/(1 − b\_B). **The CIs are too narrow**: the association heights
share views, so the parametric draws understate the common mode. None of the local
crossings is extrapolated: the two added heights (3.4, 3.8 m) bracket Laurens, whose
local crossing moves from §7's extrapolated 3.749 to a bracketed 3.481.

| city / rig | h\*\_A (slope) | B(2.6) | b\_B | 1/(1 − b\_B) | h\*\_B line (CI) | h\*\_B local (CI) | rule outcome |
|---|---|---:|---:|---:|---|---|---|
| richmond / gopro max | 1.983 (0.486) | 2.396 | 0.684 | 3.17 | 1.982 (1.940–2.022) | 2.072 (1.986–2.131) | fails 3 (`b_unvalidated`), 4 |
| richmond / nctech istar pulsar | 2.595 (0.286) | 2.680 | 0.588 | 2.43 | 2.664 (2.639–2.688) | 2.744 (2.717–2.769) | fails 3 (`b_unvalidated`), 4 |
| richmond / unknown | 2.708 (0.359) | 2.666 | 0.742 | 3.87 | 2.690 (2.601–2.786) | 2.820 (2.655–3.018) | fails 3 (`b_unvalidated`), 4 |
| laurens / gopro max | 4.314 (0.723) | 2.977 | 0.850 | 6.67 | 3.755 (3.587–3.957) | 3.481 (3.422–3.528) | fails 2, 3 (`b_unvalidated`), 4; suspect |
| clovis / gopro fusion | 2.420 (0.133) | 2.544 | 0.500 | 2.00 | 2.545 (2.526–2.563) | 2.518 (2.499–2.541) | fails 3 (`b_unvalidated`), 4 |
| morgantown / gopro max | 2.352 (0.146) | 2.507 | 0.531 | 2.13 | 2.347 (2.330–2.362) | 2.431 (2.411–2.452) | fails 3 (`b_unvalidated`), 4 |
| annapolis / trimble mx7 | 2.257 (0.202) | 2.486 | 0.615 | 2.60 | 2.280 (2.269–2.290) | 2.338 (2.319–2.355) | fails 3 (`b_unvalidated`), 4; was **passes, 2.376 m** in §3 |

Rule 4 (material) needs h\_g, the mean of A and B, so it cannot pass once rule 3 fails. Richmond's and Laurens's
GoPro Max still go per sequence (IQR 0.614 and 0.736 m), and as in §3 no sequence meets
full rule-1 support, so every sequence inherits the rig class's 2.6 m. The insta360 x4
class (4 panos) fails support.

Unvalidated, the fixed points are descriptive only. For what they are worth, the line's
h\*\_B lands within 0.025 m of h\*\_A for Richmond's GoPro Max, Morgantown and Annapolis
(the §7 pattern) and within 0.13 m for Clovis. The local crossing sits 0.08–0.10 m above
A for the same four classes, the direction the noise-1.0 simulation biases it.

### 8.5 Production gate, re-issued (§4's table)

With no group applied, the per-rig arm is the `off` arm in all five cities. Common site
set, benchmark tier, 25 m cap:

| city | sites scored | common ramps | median m | p90 m | off-pool R@2.5 | C slope (all refs) |
|---|---:|---:|---:|---:|---:|---:|
| richmond | 1,570 | 227 | 1.553 | 3.854 | 0.905 | −0.086 |
| laurens | 186 | 151 | 1.879 | 3.828 | 0.536 | −0.186 |
| clovis | 2,495 | 159 | 1.322 | 3.279 | 0.873 | −0.082 |
| morgantown | 1,733 | 227 | 1.097 | 2.989 | 0.892 | +0.012 |
| annapolis | 4,018 | 211 | 1.379 | 3.546 | 0.879 | −0.176 |

(i) PASS, (ii) FAIL (no changed city), (iii) PASS, (iv) PASS. **Verdict: FAIL**, and
`recommended: false` in every table. Annapolis's common set is now 211 ramps, not §4's
207, because the set is built from GT marks both arms can place and the arms are now
identical.

### 8.6 Regression checks

- `height_gap.py sweep richmond --group gopro/max --sequences` re-run after the helpers
  moved: `sweep.csv` and `sequences.csv` match the committed files in every column and
  value (the only byte difference is CRLF vs LF line endings).
- `fuse_sites.py runs/annapolis` (default `auto`) writes a byte-identical `sites.jsonl`
  from `main`'s code and from this branch's, before and after the tables were re-issued
  (sha256 `BC019B8F…02FC`). `--camera-height-m per-rig` loads the new table.
- #87's `simulate.csv` is untouched; `simulate_matched.csv` is new.

### 8.7 What follows

- Per-rig heights for Mapillary stay opt-in and currently apply nowhere. #53's one applied
  value, Annapolis 2.376 m, rested on B(2.6) and is withdrawn.
- B's fixed point is biased high at low heights at the error model's full noise, under
  either estimator. Before B can be validated, the real Mapillary noise scale has to be
  measured rather than assumed (#76's 2× is an estimate from the residuals, not a
  calibration). A pre-registered re-run at a measured noise scale is the natural next step.
- Instrument A needs no B to be read, and on the placement oracle (#79) an external
  inventory, not agreement between A and B, is what moved the GSV default. None of the
  five Mapillary cities is among the oracle's inventory cities (Bend, Gainesville,
  Vancouver).
