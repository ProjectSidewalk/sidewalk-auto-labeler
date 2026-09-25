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
  not tested. Two candidates remain open: B's residuals are truncated by the association
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
