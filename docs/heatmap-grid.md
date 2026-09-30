# The heatmap's 8-cell grid: error model and reproduction rules (#111)

Issue [#111](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/111), labeler
half (plan steps 2 and 3). Step 1, the sub-cell decode against ground truth, runs in
RampNet#221 and is not repeated here; `detectors/curb_ramp.py::detections_from_heatmap` is
unchanged. No GPU, no network. Rules were posted on #111 before any number was measured.

## 1. The grid, on every local run

RampNet's head upsamples a stride-32 (64x128) map 8x bilinearly before a 1x1 conv. With
`align_corners=False`, coarse sample *k* sits at hi-res coordinate 8*k* + 3.5, and a
bilinear surface peaks only at its sample points, so an integer argmax lands on hi-res
column/row residue 3 or 4 mod 8. `scripts/heatmap_grid.py grid` (data:
`figures/heatmap-grid/data/grid.csv`, full residue histograms per axis and tier):

| run | detections >= 0.55 | on residue 3/4 in both axes | all stored | on grid |
|---|---:|---:|---:|---:|
| paterson | 29,756 | 0.9980 | 46,312 | 0.9988 |
| bend | 51,543 | 0.9978 | 51,543 | 0.9978 |
| gainesville | 15,574 | 0.9997 | 43,871 | 0.9999 |
| sao_paulo | 16,159 | 0.9990 | 51,908 | 0.9997 |
| richmond | 9,526 | 0.9980 | 9,526 | 0.9980 |
| annapolis | 29,188 | 0.9983 | 29,188 | 0.9983 |
| clovis | 8,961 | 0.9998 | 8,961 | 0.9998 |
| morgantown | 10,482 | 0.9993 | 10,482 | 0.9993 |
| laurens | 708 | 0.9986 | 5,727 | 0.9998 |
| laurens_gsv | 473 | 1.0000 | 1,656 | 1.0000 |
| budapest_district5 | 67 | 1.0000 | 67 | 1.0000 |
| bayonne | 6 | 1.0000 | 18 | 1.0000 |
| vancouver (store run) | 60,104 | 0.9995 | 92,067 | 0.9997 |
| vancouver (zoom-3 control) | 294 | 1.0000 | 435 | 1.0000 |

The 0.30 tier sits between the two columns in every run (in the CSV). The few off-grid
detections are almost all at residue 2, one cell off. The quantum of a stored detection is
one coarse cell, 8 heatmap px = 2.8 deg, not one heatmap px.

## 2. Error model

**The quantization term.** An integer argmax on this map is off from the continuous peak by
up to half a coarse cell, +/-4 heatmap px. If the peak's position inside its cell is uniform,
the 1-sigma is 8/sqrt(12) = **2.31 px** (0.81 deg; `geo.SIGMA_PEAK_COARSE_CELL_PX`). The
current `sigma_peak_px = 1.0` (0.35 deg) is below the quantization alone. For GSV the
change matters: 0.81 deg is larger than the model's pitch (0.3 deg) and heading (0.5 deg)
terms. For Mapillary it is small next to the 1.5 deg / 1.0 deg terms and the 3 m GPS sigma.

**No separate flip term.** A flip is a near-tie between two neighbouring coarse cells that
a small input change resolves the other way, moving the argmax 7-8 heatmap px. It is how
often an argmax fails to *reproduce*, not an extra *error* relative to the ramp. When two
cells are near-tied, the continuous peak lies between them, within about 4 px of each, and
the uniform +/-4 term already covers that. A flip term on top would count the same
uncertainty twice. The flip rate depends on the perturbation, measured on the Vancouver
gate files (section 3): **1.0%** of matched labels (3 of 295) when the same zoom-3 pipeline is
re-run a year later, and **11.3%** (6,706 of 59,529) when the pixels are a bilinear
downsample of a different JPEG (the store run). The uniform term does not cover the model's
own coarse-level error, where the coarse peak falls in a cell that does not hold the ramp.
That is step 1's measurement (RampNet#221), and `sigma_peak_px` should become the measured
residual when it arrives.

**Measured effect of 1.0 -> 2.31 on fusion.** `scripts/heatmap_grid.py sigma`
(`figures/heatmap-grid/data/sigma_peak.csv`, `sigma_peak_verdict.csv`). Each cell re-fuses at
the benchmark tier (0.55, rig mask off, pose off, exactly as `eval_sites.py`) and scores
against the RampNet bundle at 5 m. The baseline is re-scored on today's `results.jsonl`, so
Paterson, Gainesville and Sao Paulo differ from their 2026-08-02 `fusion_eval` reports by
the gap-fill panos added since. Bend and Richmond reproduce theirs exactly.

| city | height | world R 1.0 -> 2.31 | world P | union lift | sites | chi2-gate rej. | residual rej. |
|---|---|---|---|---|---|---|---|
| paterson | 2.6 | 0.957 -> 0.951 | 0.975 -> 0.975 | 0.194 -> 0.188 | 14,363 -> 12,935 | 2,210 -> 921 | 1,181 -> 565 |
| paterson | auto | 0.957 -> 0.957 | 0.975 -> 0.975 | 0.238 -> 0.238 | 12,703 -> 12,216 | 1,191 -> 700 | 356 -> 227 |
| bend | 2.6 | 0.956 -> 0.956 | 0.957 -> 0.957 | 0.139 -> 0.139 | 14,190 -> 13,746 | 1,674 -> 1,348 | 620 -> 363 |
| bend | auto | 0.953 -> 0.953 | 0.957 -> 0.957 | 0.141 -> 0.141 | 14,061 -> 13,806 | 1,712 -> 1,441 | 509 -> 383 |
| gainesville | 2.6 | 0.927 -> 0.932 | 0.956 -> 0.956 | 0.142 -> 0.146 | 16,411 -> 14,797 | 3,246 -> 1,702 | 1,456 -> 821 |
| gainesville | auto | 0.943 -> 0.943 | 0.956 -> 0.956 | 0.196 -> 0.196 | 14,320 -> 13,716 | 2,195 -> 1,482 | 548 -> 395 |
| sao_paulo | 2.6 | 0.953 -> 0.937 | 0.893 -> 0.893 | 0.231 -> 0.216 | 17,318 -> 16,391 | 4,202 -> 3,102 | 976 -> 721 |
| sao_paulo | auto | 0.949 -> 0.942 | 0.893 -> 0.893 | 0.233 -> 0.226 | 17,428 -> 16,570 | 4,355 -> 3,330 | 968 -> 732 |
| richmond | 2.6 | 0.941 -> 0.941 | 0.959 -> 0.959 | 0.111 -> 0.111 | 1,570 -> 1,569 | 0 -> 0 | 0 -> 0 |
| richmond | auto | 0.941 -> 0.941 | 0.959 -> 0.959 | 0.111 -> 0.111 | 1,570 -> 1,569 | 0 -> 0 | 0 -> 0 |

`chi2-gate rej.` counts detections that had a candidate site inside the 8 m cap but opened
a new site because none passed the gate. `residual rej.` counts detections that passed the
gate but would have pushed the refit residual past 3.0 per dof. Both are new counters in
`fuse_sites.fuse` (`stats['rejections']`).

**Verdict (pre-registered rule): KEEP 1.0.** The rule allowed |delta world R| and |delta
world P| up to one binomial SE of the baseline in every cell, and gate and residual
rejections could rise by at most 10%. Nine of ten cells pass. Sao Paulo at 2.6 m fails:
world recall falls 1.57 pts (4 of 255 pool ramps) against an SE of 1.33 pts. So
`geo.SIGMA_PEAK_PX_DEFAULT` stays 1.0, and 2.31 is opt-in through `--sigma-peak-px`
(`fuse_sites.py`, `eval_sites.py`) or `FuseParams.sigma_peak_px`. **Every published fusion,
clustering, residual and camera-height table used 1.0.**

What the table says besides the verdict:
- **World precision cannot see this change.** TP and FP are identical in every cell. The
  judged benchmark panos are spatially de-clustered, so no two judged detections ever share a
  site, and association changes cannot reach them.
- **Rejections fall 16-58% on GSV, and sites fall 2-10%.** A wider peak sigma merges more.
  Where recall moves, it moves at 2.6 m, the height known to run new-rig ranges long (#79):
  Paterson -0.7 pts, Sao Paulo -1.6 pts, Gainesville +0.5 pts. At `auto` it moves only in Sao
  Paulo (-0.8 pts). The losses look like over-merges (union lift falls with them), which is
  what an honest covariance does when the ranges it is fed are biased.
- **Richmond (Mapillary) is structurally insensitive.** Its 3 m GPS sigma alone gives any
  pair a combined variance of at least 18 m^2, so nothing inside the 8 m cap can exceed the
  9.21 gate (64/18 = 3.6). Both rejection counters are 0 at either sigma.

## 3. Reproduction rules in coarse cells

**`scripts/provenance_gate.py`** now matches Arms S and Z within +/-1 coarse cell:
+/-(8W/1024 + 1) px in x and +/-(8H/512 + 1) px in y, Chebyshev, seam-wrapped
(`--rule coarse-cell`, the default). The report splits every match by distance in heatmap
cells: same cell (0), grid neighbour (1: residue 3 <-> 4), off-grid (2-6), **adjacent-coarse-
cell flip (7-8)**, further. It also counts tier detections claimed by labels at two
different pixels, because the join is nearest-within-tolerance, not one-to-one, and a coarse
tolerance can let two ramps within about 2.8 deg share one detection. `--rule pixel-96`
reproduces PR #108's report number for number (checked: S 52,819 matched, Z 288 of 299).
Under that flag the report differs from PR #108's only by additions: the match-class table
and the shared-claims line. One change is not an addition: the unmatched-distance table's
fixed 64 px bucket was replaced by coarse-cell buckets.

**Vancouver under the coarse-cell rule. Exploratory:** the rule was amended after the gate
had run, and the #56 decision was taken under the amended scope (PR #118). This does not
reopen it. Full report: [`figures/heatmap-grid/vancouver_gate_coarse_cell.md`](figures/heatmap-grid/vancouver_gate_coarse_cell.md).

| check | #96 rule (PR #108) | coarse-cell rule | bar |
|---|---:|---:|---|
| Arm S share | 0.8252 | 0.9301 | >= 0.98, fail |
| Arm Z share | 0.9632 | **0.9866** | >= 0.98, pass |
| unclaimed tier detections / joinable | 0.1316 | 0.0288 | <= 0.02, fail |
| verdict | STOP (S, Z, P) | STOP (S, P) | |

Of Arm S's 59,529 matches, 79.4% are on the same heatmap cell, 9.3% are grid neighbours,
11.3% are flips, and 4 are off-grid. The coarse-cell rule accounts for the flips, and Arm Z
(the same pipeline a year later) then passes. The store run still fails, but no longer
because of position: 4,268 of its 4,477 remaining misses have a detection **below 0.55
within one coarse cell**. Resampling moved the confidence across the tier. The 0.55 tier
is a confidence rule, and a cell-level position rule cannot absorb that. Only 144 misses
have no detection within two coarse cells.

**`scripts/reinfer.py --verify`** keeps **exact pixel keys** as the reproduction test,
because `--write-band-file` builds on it. The band file must hold, at the server's tier,
exactly the live labels. A label that moved one coarse cell is a different pixel on the
server, so if a cell-level match counted as reproduced, its band would ship beside a live
label that no longer matches the file. Coarse-cell agreement is a **diagnostic**
(`summary['coarse_cell']` plus one printed line). For each pano whose keys differ, old and
new keys are paired one-to-one within +/-1 coarse cell and counted by class, flips named.
It explains a mismatch and never moves a pano out of `carry_over` (pinned in
`tests/test_reinfer.py`).

## 4. What waits on RampNet#221

- **The decode.** If a sub-cell refinement lands nearer the reviewer's box centre than the
  argmax, `detections_from_heatmap` changes. Every stored position then moves by up to 4
  heatmap px, which creates a frame difference against every live campaign (see the rollout
  caution on #111).
- **`sigma_peak_px`.** It should become the measured residual of whichever decode ships. That
  residual includes the model's coarse-level error, which the 2.31 px quantization term does
  not.
- **The reproduction rules.** Under a sub-cell decode, a flip would move a detection by up to
  8 px, and near-ties could land anywhere between the two cells. The +/-1 coarse-cell
  tolerance still covers that. The distance classes, which assume residues 3 and 4, would
  need restating.
