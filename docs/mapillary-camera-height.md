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
  `recommended: false`. Only Annapolis's applies a height other than 2.6 m.
- Under the default, fused output is byte-identical: Richmond's `sites.jsonl` and
  `sites_meta.json` fused at 2.6 m match before and after the change.

## 7. The Richmond GoPro Max gap ([#87](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/87)), 2026-09-26

Instruments A and B read Richmond's GoPro Max 0.39 m apart (§3). The decision rules for
this section were pre-registered on #87 before any number was produced, and
`scripts/height_gap.py` implements them as written (its `RULE_*` constants). Everything
here is offline CPU and changes nothing in production. No `camera_heights.json` is
rewritten, `per-rig` stays opt-in with `recommended: false`, and the default stays 2.6 m.

```bash
python scripts/height_gap.py sweep richmond --group gopro/max --sequences          # E1, E4
python scripts/height_gap.py sweep richmond laurens clovis morgantown annapolis     # E1, all rigs
python scripts/height_gap.py simulate richmond --group gopro/max --true-height 1.8 2.0 2.2 2.4 --noise-scale 0 0.5 1 --seeds 2
python scripts/height_gap.py simulate richmond --group gopro/max --true-height 2.0 --offset-px -4 -2 -1 0 1 2 4 --noise-scale 0.5 --seeds 2
python scripts/height_gap.py gt richmond --group gopro/max --heights 1.98 2.2 2.38 2.6   # E3a, E5
python scripts/height_gap.py sweep laurens --results results.jsonl results.raw.jsonl     # E6
python scripts/height_gap.py simulate laurens --group gopro/max --true-height 2.6 3.0 3.4 --noise-scale 0.5 --seeds 2
python scripts/height_gap.py sweep paterson --group-by year                          # E6, GSV
python scripts/height_gap.py verdict richmond                                        # -> report.md
```

The tracked outputs are
`runs/richmond/camera_height/gap/{sweep,sequences,simulate,gt_offset,instrument_c}.csv`,
`report.md`, and `runs/laurens/camera_height/gap/{sweep,simulate}.csv`. The five-city
sweep and the Paterson arm are local: `runs/_summary/camera_height/gap_sweep.csv` and
`gap_paterson_year.csv`. With the jobs run in parallel, everything took about 35 minutes
of wall-clock time on a 16-core desktop.

**Notation.** h\*\_A is A's fixed point. B(h) = h·(1 − (s − s\_null)) is B read after
associating at height h; #53 read B only at 2.6 m. h\*\_B is the fixed point of the line
B(h) = a\_B + b\_B·h. The null s\_null is now averaged over 10 seeds, where #53 used one.
For this rig the null's SD across seeds is 0.030 m at 2.6 m, under the plan's 0.05 m
alarm, so #53's CI was not too narrow on that account. The 10-seed mean does move B(2.6)
from #53's 2.377 m to 2.396 m. So here **G = 2.396 − 1.983 = 0.414 m**, not the 0.394 m
quoted in the issue.

### Verdict

**Explained, by candidate 3: association pulls B toward the height it ran at.** Read as
a fixed point, B agrees with A to 1 cm.

| candidate | outcome | the numbers the rule reads |
|---|---|---|
| 3. association pull | **confirmed** | Real data: h\*\_B 1.993 (95% CI 1.948–2.040) vs h\*\_A 1.983; b\_B 0.708. E2 at h\_true 2.0, noise 0.5 / 1.0: B(2.6) − h\_true +0.359 / +0.480; h\*\_A − h\_true −0.048 / +0.018; h\*\_B − h\_true +0.046 / +0.059. |
| 1. vertical peak offset | **rejected**, narrowly | GoPro Max ε (peak minus box) is +1.62 px (95% CI +0.47 to +2.20). Inside that CI the fixed-point gap moves −0.015 m, the wrong way. The largest move for \|ε\| ≤ 4 px is 0.090 m (at ε = −4), under the 0.10 m bar. |
| 2. mixed mountings | **rejected** | Median within-sequence gap B(2.6) − h\*\_A is +0.392 m over 8 quarter-support sequences; the mixture arm closes 56% of G. On the fixed points instead, the median is +0.129 m, which would be **partial**. |
| E5, instrument C | not decisive | C slope at 2.6 m is −0.111 ± 0.150 (needs SE < 0.05), from only 36–41 GoPro Max references. |
| overall | **explained** | Corrected fixed points 1.983 / 1.993, residual +0.011 m (needs ≤ 0.25). Simulated raw gap +0.407 / +0.462 m against G = 0.414 (needs within 0.10). |

### E1: B swept over the association heights

For the GoPro Max, B(h) at 1.4, 1.6, 1.8, 2.0, 2.3, 2.6 and 3.0 m is 1.548, 1.721, 1.870,
2.033, 2.197, 2.396 and 2.717. B tracks the association height with slope 0.71, steeper
than A's 0.49. Its fixed point is where the two lines meet.

| city / rig | h\*\_A | A slope | B(2.6) | b\_B | h\*\_B (95% CI) | h\*\_B − h\*\_A | B(2.6) − h\*\_A (#53's rule 3) |
|---|---:|---:|---:|---:|---|---:|---:|
| richmond / gopro max | 1.983 | 0.486 | 2.396 | 0.708 | 1.993 (1.948–2.040) | **+0.011** | +0.413 |
| richmond / nctech istar pulsar | 2.595 | 0.285 | 2.680 | 0.668 | 2.846 (2.809–2.891) | **+0.251** | +0.085 |
| richmond / unknown | 2.708 | 0.359 | 2.666 | 0.815 | 2.997 (2.800–3.325) | **+0.288** | −0.043 |
| laurens / gopro max | 4.314 | 0.723 | 2.977 | 1.036 | none (b\_B ≥ 1) | — | −1.337 |
| clovis / gopro fusion | 2.420 | 0.133 | 2.545 | 0.494 | 2.541 (2.524–2.562) | +0.121 | +0.124 |
| morgantown / gopro max | 2.352 | 0.146 | 2.507 | 0.567 | 2.387 (2.370–2.405) | +0.035 | +0.154 |
| annapolis / trimble mx7 | 2.257 | 0.202 | 2.486 | 0.637 | 2.309 (2.298–2.321) | +0.052 | +0.229 |

With h\*\_B in place of B(2.6), rule 3 passes for Richmond's GoPro Max, Clovis,
Morgantown and Annapolis. It **fails** for Richmond's Pulsar (by 1 mm) and for Richmond's
unknown class; both passed rule 3 as #53 ran it. So the fixed-point form is not a drop-in
improvement. For rigs near 2.6 m it lands 0.25–0.29 m above A, and the Laurens simulation
below shows the same overshoot at a known 2.6 m (+0.24 m).

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

On a known world with this view graph, B read at 2.6 m is pulled toward 2.6 m. The pull
is present **at zero noise**, so it comes from association, that is, from which views end
up sharing a site, and not from the errors-in-variables null. B's fixed point recovers
the truth to within about 0.07 m for h\_true 2.0–2.4 m, and overshoots by 0.10–0.15 m at
1.8 m. A recovers the truth to within 0.06 m everywhere. At h\_true = 2.0 m the simulated
raw gap B(2.6) − h\*\_A is +0.407 m at noise 0.5 and +0.462 m at noise 1.0; the real G is
0.414 m.

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
| h\*\_B − h\*\_A | +0.184 | +0.118 | +0.104 | +0.094 | +0.062 | +0.057 | +0.057 |

The offset moves both instruments together. An offset of the measured size, ε ≈ +1.6 px,
raises both by about 0.1 m, so it is a shared bias rather than a disagreement, and it
cannot open G. It does mean both instruments probably read this rig about 0.1 m high.

### E4: per sequence

For the 8 GoPro Max sequences with quarter support, ordered by B views, the raw gap
B(2.6) − h\*\_A is +0.415, +0.259, −0.151, +0.369, +0.621, −0.700, +0.596 and +0.622 m.
The median is **+0.392 m**, so G is present inside single sequences. The mixture arm
weights each sequence's h\*\_A by its held-out views: the mean is 2.215 m, which closes 56%
of G. Over all 11 sequences with both readings, the mean is 2.181 m and closes 48%. The
per-sequence fixed-point gaps have a median of +0.129 m. Full rows are in `sequences.csv`.

### E5: instrument C on GoPro Max references

| height (m) | 1.98 | 2.2 | 2.38 | 2.6 | 1.993 (h\*\_B) |
|---|---:|---:|---:|---:|---:|
| C slope (SE) | −0.220 (0.107) | −0.067 (0.097) | −0.093 (0.093) | −0.111 (0.150) | −0.215 (0.108) |

Each slope rests on 36–41 references, all boxed or reviewer-missed, so none is the
model's own peak. The weighted zero crossing is 2.80 m (SE 0.63). The SEs are two to
three times the 0.05 bar, so C cannot choose between 2.0 m and 2.38 m; it is reported
only. The plan predicted positive slopes at 2.6 m: +0.23 for a 2.0 m rig and +0.09 for a
2.38 m one. Every observed slope is negative. The plan's sizing did not anticipate that,
and it is not explained here.

### E6: second cases (report only)

- **Laurens.** A is not identifiable on either file: its slope is 0.72 on `results.jsonl`
  and 0.84 on `results.raw.jsonl`. B's line is as steep. b\_B is 1.04 on `results.jsonl`,
  so there is no fixed point, and 0.93 on `results.raw.jsonl`, where h\*\_B is 4.22, an
  extrapolation. Simulated on Laurens's graph at noise 0.5, a tall rig gives A a slope of
  0.38 at 2.6 m, 0.50 at 3.0 m and 0.63 at 3.4 m. **A tall rig alone does not reach 0.7**
  within 3.4 m, so Laurens's 0.72 needs something more. B's fixed point is poor there:
  2.84, 4.45 and 9.43 m against true heights of 2.6, 3.0 and 3.4 m.
- **Paterson (GSV, with a depth-measured height).** The 2025–26 rig has 15,155 panos with
  a depth height. It reads h\*\_A 1.975, B(2.6) 2.237, h\*\_B **1.843**, and a depth
  median of 1.862. So the same pattern holds on GSV: B read at 2.6 m sits 0.26 m above A,
  and its fixed point lands near A and the depth median. For the 2018–2024 vintages,
  h\*\_A is 2.46–2.56, h\*\_B is 2.25–2.47 and the depth median is 2.35–2.37. The
  2007–2017 vintages have few multi-view sites and A slopes of 0.60–0.93, so they are not
  identifiable.

### What follows

- #53's rule 3 compared unlike quantities: A as a fixed point and B at 2.6 m. Read as
  fixed points, Richmond's GoPro Max passes agreement, and with its §3 support, slope and
  CI it would then pass all four rules at h\_g ≈ 1.99 m. Richmond's Pulsar and unknown
  class would instead fail agreement. Nothing here rewrites `camera_heights.json`.
- The follow-up is to make B a fixed point in `mapillary_height.py` and re-run the #53
  rule and gate. The Pulsar result and the Laurens simulation show that B's fixed point
  overshoots for rigs near or above 2.6 m, so that change needs its own simulation check
  and is not a pure win.
- RampNet#101 measures the same identity at a single association height. Its 11% slope is
  therefore a lower bound on the range error, and the fixed-point form is the right
  estimator.
