# The heatmap's 8-cell grid: error model and reproduction rules (#111)

Issue [#111](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/111). Sections
1-3 are the labeler half (plan steps 2 and 3; PR #120): no GPU, no network, rules posted on #111
before any number was measured. Section 4 is the decode half (2026-10-02, PR #129): RampNet#221's
sub-cell decode ported as an opt-in (`--decode gaussian`; the default stays argmax) and measured
through this repo's detector, in world space and under a perturbation.

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

![Two log-scale bar charts of heatmap column and row residue mod 8, pooled over 13 runs and 351,326 stored detections: residues 3 and 4 hold about 169k and 182k columns and 151k and 200k rows, every other residue holds at most 162. A dot strip beside them shows each run's share of detections on residue 3 or 4 in both axes, from 99.78% in Bend to 100% in Bayonne, Budapest and Laurens GSV.](figures/heatmap-grid/grid_mod8.png)

*Figure 1. Where stored detections land inside each 8-px block, pooled over every run's stored
detections (the Vancouver zoom-3 control is left out, since its panos are a subset of the store
run). Residues 3 and 4 are the two hi-res pixels beside each coarse sample.*

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

![Six panels. Top: a Richmond Mapillary pano crop with a curb ramp circled (confidence 0.91), its 64 by 64 px heatmap crop with the 8-px coarse-cell grid overlaid and a smooth peak, and the row and column profiles through the peak, which are piecewise linear with kinks at the coarse samples. Bottom: a second Richmond pano's heatmap before and after resampling to 0.75x with JPEG re-encoding; the two look identical, yet the argmax moves 7 px right and 7 px up, and a profile along the line through both argmaxes shows two peaks of 0.697 and 0.698 that trade places.](figures/heatmap-grid/heatmap_crop.png)

*Figure 2. A real RampNet heatmap (Richmond bundle, `scripts/heatmap_grid.py examples`). Top:
the peak is a bilinear surface with straight segments between coarse samples, so the argmax can
only land on a sample. Bottom: an adjacent-coarse-cell flip, the section 3 mechanism. The
resample changes the heatmap by about 0.001, which is enough to reorder two samples tied at 0.70.
Imagery: Mapillary contributors vukmercd23 and HKocen, CC BY-SA 4.0.*

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

![Two dot plots over ten city-by-height cells. Left: change in world recall from sigma 1.0 to 2.31 px, each with a grey band of plus or minus one binomial standard error; nine dots sit inside their band, and Sao Paulo at 2.6 m sits outside at minus 1.57 points against a 1.33-point bound, marked FAIL. World precision is unchanged in every cell. Right: relative change in chi-square gate and residual rejections, all between minus 58% and 0%, well left of the dashed plus-10% limit.](figures/heatmap-grid/sigma_sweep.png)

*Figure 3. The pre-registered rule, cell by cell. Recall moves only inside its noise band except
in Sao Paulo at 2.6 m (4 of 255 pool ramps). The rejection clause is nowhere near binding: a
wider sigma merges more, so both counters fall.*

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

**Why each miss did not reproduce (#111 item 3, 2026-10-05; diagnostics only).** Both checks
now give every miss a class; no rule, verdict or existing report line moved. In the gate, each
unmatched label gets a `miss_class`, first that applies: `dims_differ`, `no_detection`,
`below_tier_in_tolerance` (a detection below 0.55 inside the tolerance: the position
reproduced, the confidence did not), `tier_same_cell` / `tier_grid_neighbour` /
`tier_off_grid` / `tier_flip` (a detection at the tier within 8 heatmap cells but outside the
tolerance), `below_tier_shift` / `below_tier_flip` (the nearest detection within 8 cells is
below the tier and also moved), and `beyond` (nothing at any confidence within 8 cells).
Under the coarse-cell rule everything within 8 cells is inside the tolerance, so only
`dims_differ`, `no_detection`, `below_tier_in_tolerance` and `beyond` can occur; the tier and
shift classes are what `pixel-96` exposes. On Vancouver's Arm S:

| miss class | `pixel-96` | coarse-cell |
|---|---:|---:|
| `below_tier_in_tolerance` | 3,502 | 4,268 |
| `tier_flip` (7-8 cells; 6,606 at exactly 7) | 6,706 | 0 |
| `below_tier_flip` (733 at exactly 7) | 766 | 0 |
| `tier_off_grid` | 4 | 0 |
| `beyond` (> 8 cells) | 207 | 207 |
| `no_detection` | 2 | 2 |
| unmatched | 11,187 | 4,477 |

The two columns account for each other. The 6,706 tier flips are exactly the 6,706 matches the
coarse-cell rule classes as flips, and the 766 below-tier flips join the 3,502 to make its
4,268. PR #108's "7,339 of 7,679 far misses at exactly 7 cells" is 6,606 + 733 out of
6,706 + 766 + 207. Under the coarse-cell rule, then, 95% of the store run's misses are a
confidence that crossed 0.55 at the right spot, and 207 (4.6%) have nothing within 8 cells.
`unmatched.csv` gains `nearest_cells`, `nearest_class`, `nearest_tier_cells`,
`nearest_tier_class` and `miss_class` after its original 13 columns, which are unchanged. Arm Z
gets the same table in the report (`pixel-96`: 4 grid-neighbour, 3 flip, 2 below-tier shift,
1 below tier in tolerance, 1 beyond, of 11).

`reinfer.py --verify` classes each carried-over pano, first that applies: `pano_drift`,
`border_band_only` (the #130 diagnostic), `new_or_lost` (an unpaired tier key with no below-tier
detection within one coarse cell on the other side and no tier key left there to pair with;
a tier detection already paired with another key, e.g. two peaks merged into one, may still
sit nearby), `threshold` (each unpaired tier key has a
below-tier detection within a coarse cell on the other side), `flip`, `off_grid`, `jitter`.
The classes go in `summary['mismatch_classes']`, with `summary['key_classes']` pooling the
unpaired keys (`threshold_down` / `threshold_up` / `vanished` / `appeared`), and
`--mismatch-csv PATH` writes one row per pano. Richmond's re-inference (`results.f01.jsonl`
against the live 0.55 campaign) carries over 2 of 9,091 panos: one `threshold` (the 0.550011
-> 0.549993 peak, a `threshold_down`) and one `flip`. Neither is `new_or_lost`.

## 4. The sub-cell decode (decode half, 2026-10-02)

RampNet#221 landed a sub-cell decode (`rampnet/subcell.py`, RampNet PRs 226 / 229 / 233). This
section ports it as an **opt-in** and measures it through this repo's detector. Plan:
[#111 comment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/111#issuecomment-5956604334).
**No default changes; nothing is sent to any server.**

### Key takeaways

1. **RampNet's residual table reproduces here, and the labeler's own decode gives the same
   positions.** One forward pass per bundle pano through `CurbRampDetector` on makelab2, RampNet's
   extraction applied to that heatmap: every split's argmax and gaussian mean residual equals
   RampNet's published value to the third decimal, except manual_gold argmax (5.082 vs 5.080, the
   1-px cross-machine tie RampNet already names) and, through it, the all-5 pool (5.172 vs 5.170
   argmax, 4.472 vs 4.473 gaussian). On every peak both paths return, the labeler's
   gaussian position equals RampNet's `detect_peaks` exactly. (Figure 4, `decode_residual.csv`,
   `decode_agreement.csv`.)
2. **Through the labeler path the decode moves detections 0.73 px nearer the box centre on
   manual_gold** ([-0.79, -0.67], 3,496 pairs) and 0.50 px pooled over the four box splits;
   annapolis (Mapillary) is the one split whose CI crosses zero. Scores and peaks do not change.
3. **The measured `sigma_peak_px` is 4.2 px for argmax and 3.8 px for gaussian** (manual_gold,
   rms of the two axes, bias removed). Both are upper bounds on the model's own per-view error.
   At the measured value PR #120's ten-cell table fails the #111 rule in the same cell 2.31 px
   failed, Sao Paulo, now at both heights (world recall -1.6 pts, 4 of 255 ramps). The default
   stays 1.0. (`decode_sigma.csv`, `decode_sigma_table*.csv`.)
4. **In world space, at the benchmark tier (0.55), the decode buys nothing measurable.** One forward pass, both decodes, fused and scored as `eval_sites.py` does. The instrument that can see a sub-cell move is the frozen-association leave-one-out residual: it tightens Laurens GSV sites at 2.6 m (-0.18 m [-0.28, -0.07], chi2/dof -0.07) but not at `auto`, the GSV production height, and not in pixels, and it makes Laurens Mapillary sites slightly looser (+0.87 px [+0.49, +1.28], chi2/dof +0.02). World precision and recall are nearly blind to it by construction (4.4); paired, the ramps one arm recovers and the other does not are 6 gained / 9 lost on Laurens Mapillary (sign test p 0.61) and 0 / 0 and 1 / 2 on Laurens GSV. The production tier (0.30) was not run.
   (Figure 5, `decode_world_*.csv`.)
5. **The decode removes the 7-8 px flip from reproduction.** Between a pano and a resampled copy (0.75x + JPEG q90), 44 of 1,204 paired argmax peaks (3.7%) jump 7-8 heatmap px to the neighbouring coarse cell. Under the gaussian decode no paired peak moves more than 5.4 px; 99% move under 2.5 px.
   (Figure 6, `decode_stability*.csv`.)

### 4.1 What was built

- **`detectors/decode.py`** (torch-free): `detections_from_heatmap(heatmap, decode="argmax")`.
  `argmax` is the pre-#111 extractor, bit for bit (pinned in `tests/test_decode.py` against a
  verbatim copy of the old function, on synthetic maps and on four real manual_gold heatmaps
  stored as coarse maps in `tests/fixtures/decode_heatmaps.npz`). `gaussian` finds the SAME peaks
  (storage floor, top 50, `exclude_border` as production) with the same scores and order, then
  moves each with RampNet's `refine_peaks(method="gaussian")`. It refuses (`ValueError`) a heatmap
  that is not an exact x8 upsample (clipped, TTA-combined, wrong input size), RampNet's
  `UPSAMPLE_RTOL` guard. `CurbRampDetector.heatmap()` returns the raw, unclipped head output,
  which passes (relative residual <= 4.2e-6 on every pano measured here, 8,131 panos).
- **`detectors/rampnet_subcell.py`** is RampNet's `rampnet/subcell.py` verbatim, at RampNet
  commit `cf7aecd` (identical at RampNet main `459ea9e`), sha256 `b0712dfe...`
  (`detectors.decode.RAMPNET_SUBCELL_SHA256`; `test_vendored_subcell_is_rampnets_file_verbatim`).
  It is vendored, not taken from the Hub, because the published model package does not yet ship
  `rampnet_subcell.py` (the re-export is Jon's). To update: copy the file verbatim and change
  both constants.
- **The decode is bound wherever runs combine.** A record written under `gaussian` carries
  `"detection_decode": "gaussian"`; argmax records carry nothing, so every argmax
  `results.jsonl` record and every argmax `*.submission.json` is byte-identical to before (the
  latter pinned against origin/main's output in `tests/test_decode_guards.py`), and every older
  line reads as argmax (`detectors.record_decode`). Two run artifacts that no code hashes do
  change for argmax runs: a new run's `manifest.json` gains `"detection_decode": "argmax"` (so
  does a pre-#111 manifest whose `results.jsonl` is absent or empty, e.g. a scan-only dir), and
  every `sites_meta.json` gains `detection_decode`.
  - `main.py --decode {argmax,gaussian}` (default argmax): the run directory is bound to its
    decode (`manifest.json` `detection_decode`; a pre-#111 manifest over existing records is
    argmax), and a resume or gap fill under the other decode is refused.
  - `scripts/reinfer.py --decode` (default: the run's own). An `--out` file is bound to the
    decode its records carry. `--verify` refuses a pair written under different decodes
    (`--allow-mixed-decode` compares them as a diagnostic); `--write-band-file` refuses a mix
    outright, because the band would ship in another frame than the live labels.
  - `scripts/fuse_sites.py` refuses a file, or a pano set, that mixes decodes
    (`--allow-mixed-decode`); `sites_meta.json` records `detection_decode`.
  - `scripts/eval_sites.py` (and everything that loads through `load_city_at_height`) refuses a
    non-argmax run: the bundles were judged on argmax positions, and the exact drift gate would
    otherwise drop every GT pano silently. A gaussian run is scored with
    `scripts/subcell_decode.py world` (4.4).
  - `scripts/provenance_gate.py` refuses a non-argmax arm: the live labels it checks were
    placed by argmax. So a city that receives a gaussian campaign has **no provenance gate**
    until one is built (4.5).
  - The research scorers that do their own bundle join (`agree_rate.py`, `site_explorer.py`,
    `gsv_partial_pose.py`, `mapillary_tilt.py`) refuse a non-argmax run the same way
    (`eval_sites.require_argmax`).
  - `send_to_ps.py` refuses a file that mixes decodes, and a file whose decode differs from any
    campaign already recorded beside it (`*.submission.json`; a record without
    `detection_decode` was sent before #111 and is argmax). `--allow-mixed-decode` overrides; the
    record then keeps the reason and the full mix (`"detection_decode": {"argmax": 2,
    "gaussian": 1}`), and any record holding a mix or an override counts as differing from every
    single-decode file, so a later argmax campaign beside it is refused too. A non-argmax
    campaign's record says `detection_decode`. The guard is per directory, like
    `check_live_positions`: it sees only campaigns recorded beside the file (every `runs/<name>/`
    here is one city). Nothing else in `send_to_ps.py` changed.

### 4.2 Agreement with RampNet, through this repo's detector

`scripts/subcell_decode.py detect` runs `CurbRampDetector.heatmap()` once per pano and applies
both decodes to that one heatmap, plus RampNet's own extraction
(`detect_peaks(h, 0.30, clip=True)`, `exclude_border=False`) under both. `residual` then
reproduces RampNet#221's protocol: GT = manual_gold YOLO box centres and the reviewer boxes
(`boxes.json`, status `boxed`; `cant` entries kept as match decoys), pairs fixed once on the
argmax positions (greedy by confidence, radius 0.022, x wrapped), both decodes scored on the
same pairs, pano-cluster bootstrap (2,000 reps, seed 111).

| split | pairs | argmax mean px: here (RampNet) | gaussian mean px: here (RampNet) | labeler path: pairs, d [95% CI] |
|---|---:|---|---|---|
| manual_gold | 3,517 | 5.082 (5.080) | 4.353 (4.353) | 3,496, -0.729 [-0.786, -0.673] |
| annapolis | 102 | 7.133 (7.133) | 6.945 (6.945) | 102, -0.188 [-0.635, +0.243] |
| paterson | 85 | 5.665 (5.665) | 4.874 (4.874) | 85, -0.792 [-1.137, -0.473] |
| richmond | 246 | 5.327 (5.327) | 4.796 (4.796) | 245, -0.529 [-0.765, -0.275] |
| sao_paulo | 93 | 5.563 (5.563) | 5.066 (5.066) | 93, -0.497 [-0.928, -0.097] |
| pooled, 4 box splits | 526 | 5.774 (5.774) | 5.273 (5.273) | 525, -0.499 [-0.669, -0.328] |
| pooled, all 5 | 4,043 | 5.172 (5.170) | 4.472 (4.473) | 4,021, -0.699 [-0.756, -0.645] |

The labeler path differs from RampNet's only in its peak finder: `exclude_border` (skimage's
default) drops peaks within 10 heatmap px of the edge, including the 360-degree seam. That is
43 peaks over the five splits (34 on manual_gold) and 22 pairs. RampNet#132 calls that drop a
defect, and for the labeler it is filed as
[#130](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/130); it is kept here
because changing it moves stored detections, and it is outside #111. On every peak both paths return, the two gaussian positions are identical
(`decode_agreement.csv`, max difference 0.0 px).

![Two panels over seven rows (five benchmark splits and two pools). Left: mean distance to the box centre under argmax (orange) and the gaussian decode (blue), labeler detector, with RampNet's published values as open triangles that sit on top of the dots; the blue dot is left of the orange one on every row, from 0.19 px on annapolis to 0.79 px on paterson. Right: the paired change with its 95% CI; every interval lies left of zero except annapolis, which spans -0.64 to +0.24.](figures/heatmap-grid/decode_residual.png)

*Figure 4. Same peaks, same scores, positions moved by the decode. The labeler's detector
reproduces RampNet#221's table, and its own decode (labeler peak finder) gives the same changes.*

#### Seam band (#130)

The 43 peaks above are the border band: peaks within 10 heatmap px of any edge, of which the
360-degree seam (left and right edges) is the part that can hold a ramp. Since
[#130](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/130) the peak finder
has a second opt-in beside `--decode`: `--border keep` (main.py, reinfer.py) is RampNet's rule
(`exclude_border=False`, no NMS across the seam), bound in the manifest
(`detection_border`) and refused in a mix exactly like the decode. The default stays
`exclude`. What the band costs on both Laurens arms, and whether its peaks are lost ramps or
only lost views, is in [seam-band-130.md](seam-band-130.md). A third opt-in, `--border wrap`, is `keep`
plus NMS wrapped across the seam (this repo's rule; one peak per straddling ramp), estimated in
section 9 of that doc.

### 4.3 A measured `sigma_peak_px`

`decode_sigma.csv`: the residual's SD per axis with the bias removed, labeler path.

| split | decode | SD x | SD y | rms | robust rms (1.4826 MAD) |
|---|---|---:|---:|---:|---:|
| manual_gold | argmax | 5.03 | 3.26 | **4.24** | 3.67 |
| manual_gold | gaussian | 4.67 | 2.63 | **3.79** | 2.99 |
| 4 box splits | argmax | 6.28 | 3.20 | 4.98 | 4.05 |
| 4 box splits | gaussian | 6.00 | 2.66 | 4.64 | 3.71 |

**What part of it is the reviewer's box.** These SDs are upper bounds on the model's per-view
error. They also hold (a) where the labeller put the point, and (b) the systematic offset between
the point the model marks (Stage 1's auto-placed point, which it was trained on) and the box
centre. Neither can be separated from the model's error with one rater. Two reads bound it:

- manual_gold "boxes" are a median 3 heatmap px wide (`box_extent_x_median_px`): they are point
  labels, so on manual_gold the box extent contributes nothing and the SD is model error plus
  point placement plus concept offset.
- On the box splits, regressing the squared residual on the box's squared extent leaves a y
  intercept of 0.98 px (gaussian) and 2.2 px (argmax): almost all of the y residual beyond the
  quantization grows with the ramp's size. In x the intercept is about 5 px whatever the box
  size, which is the concept offset along the ramp's width, not box extent.

The concept offset is largely a property of the ramp in 3-D, shared by every view of it, so it
does not make views disagree with each other the way per-view noise does. The y axis (the one
that sets range) carries 2.6-3.3 px; x carries most of the rest. A single isotropic sigma is
therefore a compromise; `ErrorModel` has one, and the rms is what is reported.

**PR #120's ten cells at the measured value.** `subcell_decode.py sigma-table` re-fuses the five
cities at the benchmark tier at 2.6 m and `auto` exactly as PR #120 did (stored runs are argmax,
so the argmax value applies), and applies the pre-registered #111 rule
(`decode_sigma_table.csv`, `decode_sigma_table_verdict.csv`). The 1.0 baseline reproduces PR
#120's row in every cell.

| city | height | world R 1.0 -> 4.24 | -> 3.67 | sites 1.0 -> 4.24 | gate rej. | resid. rej. | verdict |
|---|---|---|---|---|---|---|---|
| paterson | 2.6 | 0.957 -> 0.947 | 0.951 | 14,363 -> 12,346 | 2,210 -> 526 | 1,181 -> 165 | pass |
| paterson | auto | 0.957 -> 0.957 | 0.957 | 12,703 -> 11,978 | 1,191 -> 451 | 356 -> 108 | pass |
| bend | 2.6 | 0.956 -> 0.956 | 0.956 | 14,190 -> 13,707 | 1,674 -> 1,127 | 620 -> 329 | pass |
| bend | auto | 0.953 -> 0.953 | 0.953 | 14,061 -> 13,751 | 1,712 -> 1,198 | 509 -> 357 | pass |
| gainesville | 2.6 | 0.927 -> 0.932 | 0.932 | 16,411 -> 13,941 | 3,246 -> 1,031 | 1,456 -> 275 | pass |
| gainesville | auto | 0.943 -> 0.939 | 0.943 | 14,320 -> 13,351 | 2,195 -> 1,022 | 548 -> 274 | pass |
| sao_paulo | 2.6 | 0.953 -> 0.937 | 0.937 | 17,318 -> 15,678 | 4,202 -> 2,185 | 976 -> 523 | **FAIL** |
| sao_paulo | auto | 0.949 -> 0.934 | 0.934 | 17,428 -> 15,865 | 4,355 -> 2,334 | 968 -> 562 | **FAIL** |
| richmond | 2.6 | 0.941 -> 0.941 | 0.941 | 1,570 -> 1,571 | 0 -> 0 | 0 -> 0 | pass |
| richmond | auto | 0.941 -> 0.941 | 0.941 | 1,570 -> 1,571 | 0 -> 0 | 0 -> 0 | pass |

World precision is identical in every cell (judged sites never share a site, section 2). Sao
Paulo loses 4 of 255 (2.6 m) / 257 (auto) pool ramps against a 1.3-1.4 pt SE at both values, the
same loss 2.31 px produced at 2.6 m; the other eight cells are within noise and, where nonzero, gate and residual rejections fall 30-86%. **The default stays 1.0** (the PR changes no default); this table is what a change
would be decided on. The verdict is the same for the robust value 3.67.

### 4.4 World space: one forward pass, two decodes

`subcell_decode.py world` re-detects an archived run once (`detect --source-resize`: the native
archive pano resized to 4096x2048 as `sources/` do, then the detector), writes the run's own
records with each decode's detections as two results files, and fuses and scores both exactly
as `eval_sites.py` does (tier 0.55, rig mask off, pose off, 5 m match, `sigma_peak_px` 1.0).
Two things make the pair comparable:

- **The bundle is re-keyed to the pass's peaks.** The judged bundle was exported from the
  original run, whose pixels differ (GSV: zoom-3 tiles; the archive is native resolution), so
  `eval_sites`' exact drift gate would drop most GT panos. For each judged pano the bundle's
  >= 0.55 detections are paired one-to-one with the pass's argmax >= 0.55 peaks within one
  coarse cell (9 heatmap px, the provenance gate's coarse-cell rule); the pano is kept only if
  every detection pairs and none is left over. The SAME panos, verdicts and peaks then score
  both arms.
- **The GT-free test freezes the association.** Sites come from the argmax fuse; every member
  is then re-placed at its gaussian position (same pano, same peak) and the leave-one-view-out
  residual and site chi2/dof are recomputed, paired by view, with a site-cluster bootstrap.
  Re-associated metrics (sites, fragmentation) are reported too, but they mix the decode with
  association changes.

Re-associated (each arm fused on its own; "->" is argmax -> gaussian; Laurens Mapillary is shown at 2.6 m only, because `auto` resolves to 2.6 m for Mapillary):

| city | height | GT panos kept | pool ramps | world R | world P | operational sites | sites with another < 5 m | pool ramps with >= 2 sites < 5 m | chi2/dof median | LOO m median |
|---|---|---|---|---|---|---|---|---|---|---|
| laurens | 2.6 | 94 of 94 | 235 -> 236 | 0.6979 -> 0.6864 | 0.8922 -> 0.8922 | 186 -> 184 | 50 -> 48 | 50 -> 43 | 0.3581 -> 0.3685 | 2.7668 -> 2.7463 |
| laurens_gsv | 2.6 | 64 of 86 | 108 -> 107 | 0.8148 -> 0.8131 | 0.9104 -> 0.9104 | 178 -> 175 | 53 -> 55 | 23 -> 23 | 0.5421 -> 0.4088 | 1.0707 -> 0.9515 |
| laurens_gsv | auto | 64 of 86 | 109 -> 108 | 0.8073 -> 0.7963 | 0.9104 -> 0.9104 | 175 -> 174 | 44 -> 48 | 22 -> 21 | 0.5359 -> 0.4541 | 1.0745 -> 1.019 |

Frozen association (sites with >= 3 operational views from the argmax fuse; every member re-placed at its gaussian position; paired by view; site-cluster bootstrap, 2,000 reps):

| city | height | sites / views | LOO m mean, argmax | d LOO m [95% CI] | LOO px mean, argmax | d LOO px [95% CI] | chi2/dof mean, argmax | d chi2/dof [95% CI] |
|---|---|---|---|---|---|---|---|---|
| laurens | 2.6 | 107 / 573 | 3.0624 | +0.019 [-0.043, +0.080] | 28.0904 | +0.869 [+0.488, +1.284] | 0.4249 | +0.018 [+0.009, +0.028] |
| laurens_gsv | 2.6 | 68 / 263 | 1.4213 | -0.175 [-0.276, -0.072] | 7.7395 | +0.013 [-0.616, +0.677] | 0.5912 | -0.074 [-0.126, -0.025] |
| laurens_gsv | auto | 71 / 279 | 1.4351 | -0.068 [-0.165, +0.024] | 9.079 | +0.581 [-0.049, +1.278] | 0.6397 | -0.000 [-0.051, +0.052] |

![Three dot-and-interval panels over three rows (Laurens Mapillary at 2.6 m, Laurens GSV at 2.6 m and at auto). Left, leave-one-out residual in metres: Laurens GSV at 2.6 m sits left of zero at -0.175 with its interval clear of zero; the other two rows straddle zero. Middle, the same residual in heatmap pixels: Laurens Mapillary sits right of zero at +0.87 with its interval clear of zero; both GSV rows straddle zero. Right, site chi-square per degree of freedom: GSV at 2.6 m -0.074 (clear of zero), GSV at auto 0.00, Mapillary +0.018 (clear of zero, small).](figures/heatmap-grid/decode_world.png)

*Figure 5. Frozen association: each multi-view site's members re-placed at their gaussian
positions. A negative value means the decode made the views agree better.*

**What each measure can see.** World precision is membership-based (verdicts on a site's
members), and a GT ramp seen by a verdict-true detection in its own view counts as recalled
wherever that detection lands; the GT `det` points are the detections themselves, placed by each
arm's own decode. So P and R can respond to the decode only through association, and only for
pool ramps that are not self-detected (46-47 of 107-109 on Laurens GSV, 140 of 235 on Laurens
Mapillary). They are read paired, ramp by ramp (`decode_world_pair_<city>.csv`:
`responding_ramps`, `gained_gaussian`, `lost_gaussian`, `sign_test_p`):

| city | height | ramps that can respond | recovered by both | gaussian only | argmax only | sign test p |
|---|---|---:|---:|---:|---:|---:|
| laurens (Mapillary) | 2.6 | 140 | 60 | 6 | 9 | 0.6072 |
| laurens_gsv (GSV) | 2.6 | 46 | 26 | 0 | 0 | 1.0 |
| laurens_gsv (GSV) | auto | 47 | 24 | 1 | 2 | 1.0 |

The leave-one-out residual and chi2/dof under frozen association are the instruments that can see
a sub-cell move, and they carry the reading.

**Reading.** The decode does not buy anything measurable in fusion at the benchmark tier. The one
gain, Laurens GSV at 2.6 m in metres and chi2/dof, does not hold at `auto` (the production height
for GSV), and in pixels it is zero. On Laurens Mapillary the decode makes the views agree slightly
worse: +0.87 px on a leave-one-out residual whose mean is 28 px, i.e. 3%. RampNet#221's two
Mapillary splits disagree on the x axis, so they neither predict nor explain this: annapolis's x
spread did not improve (SD x +0.05 [-0.35, +0.41], labeler path) and richmond's did (-0.39
[-0.60, -0.15]). The scale explains the null: the decode moves a view by 1.5 heatmap px per axis
at the median, while the views of a site disagree by 7.7 px (GSV) to 28 px (Mapillary) on
average, dominated by camera height, pose and position error. Paired, world recall changes by 0
to 3 ramps net, none of them distinguishable from noise (table above); world precision and the
judged sites are identical. Caveats: two cities, one of each source, at the benchmark tier only;
the GT `det` points move with each arm (the pool differs by one ramp, and 1 responding ramp on
Laurens GSV is keyed in one arm only); and on Laurens GSV 22 of 86 judged panos drop out of the
re-keyed bundle (20 change their >= 0.55 count between the original zoom-3 pixels and the native
archive, 2 move further than a coarse cell), on Laurens Mapillary none.

### 4.5 Reproduction rules for a gaussian campaign

Section 3's distance classes (same cell 0, grid neighbour 1, off-grid 2-6, flip 7-8) assume an argmax on residues 3 and 4. A gaussian position is continuous, so for a gaussian-vs-gaussian comparison they are restated here, measured. `subcell_decode.py detect --perturb` re-ran the four box splits (499 panos) on a 0.75x + JPEG q90 copy of each pano, PR #120's stand-in for a store JPEG, and `stability` pairs each >= 0.30 peak with its counterpart (argmax positions, one-to-one within 12 px) and measures how far it moved under each decode (`decode_stability.csv`, `decode_stability_hist.csv`):

| split | paired peaks | argmax: same px / flip 7-8 | gaussian: median / p99 / max shift (px) | gaussian shift on the argmax flips, median |
|---|---:|---|---|---:|
| annapolis (Mapillary) | 261 | 225 / 16 (6.1%) | 0.28 / 2.46 / 3.32 | 1.35 |
| paterson (GSV) | 307 | 297 / 7 (2.3%) | 0.09 / 0.58 / 1.33 | 0.21 |
| richmond (Mapillary) | 319 | 287 / 13 (4.1%) | 0.16 / 2.11 / 5.45 | 0.68 |
| sao_paulo (GSV) | 317 | 303 / 8 (2.5%) | 0.13 / 1.45 / 2.72 | 0.46 |

![Grouped bar chart on a log scale of how far the same peak moved between a pano and its resampled copy, 1,204 paired peaks. Argmax (orange): 1,112 under 0.5 px, 48 at about 1 px, nothing from 1.5 to 6.5 px, and 44 at 6.5-8.5 px (the flip). Gaussian (blue): 1,062 under 0.5 px, 124 at 0.5-1.5, 13 at 1.5-2.5, 4 at 2.5-4.5, 1 at 4.5-6.5, and none beyond.](figures/heatmap-grid/decode_stability.png)

*Figure 6. Under a small input change the argmax either stays or jumps a whole coarse cell; the gaussian position moves a fraction of a pixel, and the peaks that flipped under argmax move about 0.2-1.4 px under gaussian, because a near-tie decodes to a point between the two tied cells whichever of them wins.*

**Proposed classes for a gaussian campaign (not implemented):** `< 0.5 px` (same), `0.5-2.5 px`
(sub-cell jitter, 11.4% of pairs here), `2.5-6.5 px` (0.4%), `>= 6.5 px` (none observed: under
gaussian this is not the same peak position). **Nothing uses them today.**
`scripts/provenance_gate.py` refuses every non-argmax arm, because the live labels it checks were
placed by argmax, so a city that receives a gaussian campaign has no provenance gate until a
gate that reads the campaign's decode is built (listed as a decision in the PR). The coarse-cell
tolerance would cover every shift measured here, but that is an observation about these 1,204
pairs, not a rule anyone runs. `--rule pixel-96` and the coarse-cell rule run exactly as before
on argmax files, so PR #108's and PR #120's reports reproduce.

`reinfer.py --verify` still decides band eligibility on EXACT pixel keys, and that matters for a
gaussian campaign: argmax returned the identical heatmap pixel for 92% of the paired peaks above,
while a gaussian position moved a median 0.09-0.28 heatmap px. On the two GSV splits (16,384 px
wide panos) that is about 1.5-2 native px; on the Mapillary splits it depends on each pano's
width. Exact native keys would therefore rarely reproduce, and `--write-band-file` would carry
most panos over: a gaussian campaign's later band has to come from the same forward pass as its
base, or the band rule has to be restated. That is listed as a decision, not changed here. This
was one perturbation on 499 panos, not the Vancouver store run; the store run's 11.3% flip rate
was not re-measured under the decode.

### 4.6 Rollout (restated from the #111 issue)

Live labels never move: PS places a label once, at insert. A gaussian campaign in a city whose
live labels are argmax is a **frame change** of 1.5 heatmap px per axis at the median, p99 3.5 px,
and up to about one coarse cell for a re-anchored peak (max 8.7 px, 3.1 deg; 3 of 14,817 committed
peaks moved more than 4.5 px),
the same kind of decision as the Richmond reposition (`--reposition-live-city`). It is per
city and Jon's. The guards in 4.1 make it impossible to drift into one: a gaussian band or
gap-fill cannot be appended to, verified against, fused with or sent beside an argmax campaign
without `--allow-mixed-decode`, and the override is recorded.

### 4.7 Replication

Inputs: the released model `projectsidewalk/rampnet-model` at Hub commit
`606a11956743f7eb328d9207769034752f6191f4` (the one `detectors.KNOWN_REVISIONS` binds); RampNet
at main `459ea9e` for `benchmark/<split>/{records.jsonl,verdicts.json,boxes.json}`,
`manual_labels/` and `analysis_out/subcell_decode_221/results.json`; bundle panos at native
resolution from RampNet's Hub benchmark imagery (as RampNet#221 section 8 fetches them; checked
there against `imagery_manifest.json`); and the native archives of laurens_gsv and laurens on
makelab2 (`/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/panos`, not published),
whose `results.jsonl` sha256 are `83f49aae...` (laurens_gsv) and `16c5a348...` (laurens), equal
to the copies in `runs/`. **The `detect` outputs are committed**: `figures/heatmap-grid/data/decode/`
holds each `decode_<name>.jsonl` gzipped (`gzip -n`, 1.4 MB in all) with its `.meta.json` (model
revision, software versions, host, wall-clock); the CPU steps read them by default.

**What re-derives from the repo alone, and what does not.**

- From committed data only (no GPU, no network, nothing outside the repo): the stability CSVs
  and Figure 6 (`tests/test_decode.py` re-derives them byte for byte on every run), and
  Figures 4-5 from the committed CSVs.
- With a RampNet checkout whose files match main `459ea9e` (sibling `../RampNet`, or
  `RAMPNET_ROOT`): the residual, sigma and agreement CSVs. `decode_inputs.csv` records the content
  sha256 of every RampNet file `residual` reads, `residual` refuses a RampNet root without the
  published report, and the re-derivation test skips (saying why) unless the checkout matches.
- **Not from the repo alone**: the ten-cell table (4.3) and the world tables (4.4) also read
  gitignored run files. Each is named here with its sha256 (also in `decode_sigma_table.csv` and
  `decode_world_pair_<city>.csv`), and `sigma-table` / `world` refuse to run when a file present
  on disk has another hash than the committed CSV recorded (`--allow-input-mismatch` overrides).
  They live in the run directories (`runs/<city>/` of Jon's desktop checkout) and, for the two
  Laurens runs, in the makelab2 archive beside the panos; none is published.

| file (gitignored) | used by | sha256 |
|---|---|---|
| `runs/paterson/results.jsonl` | ten-cell table (4.3) | `651226f9f1e6...` |
| `runs/paterson/depth/index.csv` | ten-cell table, `auto` | `06e878e2d77e...` |
| `runs/bend/results.jsonl` | ten-cell table (4.3) | `1307faa8041a...` |
| `runs/bend/depth/index.csv` | ten-cell table, `auto` | `a4ae7d3743ca...` |
| `runs/gainesville/results.jsonl` | ten-cell table (4.3) | `9f4a57f35d24...` |
| `runs/gainesville/depth/index.csv` | ten-cell table, `auto` | `02ad2a5d87e2...` |
| `runs/sao_paulo/results.jsonl` | ten-cell table (4.3) | `54348f367dfa...` |
| `runs/sao_paulo/depth/index.csv` | ten-cell table, `auto` | `46315eb54fbb...` |
| `runs/richmond/results.jsonl` | ten-cell table (4.3) | `109e7645ebf5...` |
| `runs/laurens_gsv/results.jsonl` | world (4.4), pano blocks | `83f49aaecac1...` |
| `runs/laurens_gsv/depth/index.csv` | world, `auto` height | `74e56eb5514e...` |
| `runs/laurens/results.jsonl` | world (4.4), pano blocks | `16c5a348b739...` |

The 64x128 coarse maps are on makelab2 only (`/homes/gws/jonf/decode111/coarse/`); each line of
a `decode_<name>.jsonl` carries its map's sha256.

As run (2026-10-02, makelab2, one A40 shared with other jobs, torch 2.14.0+cu130, transformers
5.12.1, Python 3.12; branch `subcell-decode-111` at `f74ded9` for the original passes and
`417c728` for the perturbed ones):

```bash
# GPU: bundles as RampNet#221 ran them (no pre-resize), then the two archives
python scripts/subcell_decode.py detect --panos-dir $RN/manual_gold/panos --ids ids_manual_gold.txt \
    --out decode_manual_gold.jsonl --coarse-dir coarse/manual_gold    # ... and the 4 box splits
python scripts/subcell_decode.py detect --panos-dir $AR/laurens_gsv/panos \
    --results $AR/laurens_gsv/results.jsonl --source-resize --out decode_laurens_gsv.jsonl
python scripts/subcell_decode.py detect --panos-dir $AR/laurens/panos \
    --results $AR/laurens/results.jsonl --source-resize --out decode_laurens.jsonl
# GPU: the stability arm (4 box splits, 0.75x + JPEG q90)
python scripts/subcell_decode.py detect --panos-dir $RN/richmond/panos --ids ids_richmond.txt \
    --perturb --out decode_richmond_perturbed.jsonl                    # ... x4
# CPU (desktop, minutes): every CSV under docs/figures/heatmap-grid/data/decode_*, from the
# committed decode/ outputs (pass --decode-dir / --decode-file for your own GPU pass)
python scripts/subcell_decode.py residual --rampnet-root ../RampNet
python scripts/subcell_decode.py sigma-table paterson bend gainesville sao_paulo richmond \
    --sigma 4.24 3.67 --benchmark-root ../RampNet/benchmark
python scripts/subcell_decode.py world laurens_gsv --split laurens_gsv \
    --results runs/laurens_gsv/results.jsonl --decode-file docs/figures/heatmap-grid/data/decode/decode_laurens_gsv.jsonl.gz \
    --work-dir /tmp/w
python scripts/subcell_decode.py world laurens --split laurens_mapillary --heights 2.6 \
    --results runs/laurens/results.jsonl --decode-file docs/figures/heatmap-grid/data/decode/decode_laurens.jsonl.gz --work-dir /tmp/w
python scripts/subcell_decode.py stability annapolis paterson richmond sao_paulo
# no GPU, no network: Figures 4-6 from the committed CSVs, byte-reproducible
python scripts/subcell_decode.py figures
```

`ids_<split>.txt` is the pano order of RampNet's `benchmark/<split>/records.jsonl`. Ids are
read in that order; `residual` refuses a decode file missing any bundle pano. On the same
software stack the GPU pass is expected to reproduce bit for bit; across machines, fp32 noise
can flip a within-cell 1-px tie (the 0.002 px above) or move a peak across a tier.

**Cost.** Free compute. GPU wall-clock on the shared A40: bundles 1,491 s for 1,499 panos
(manual_gold 836 s); laurens_gsv 4,739 s for 2,137 native 16k panos;
laurens 7,277 s for 4,495 Mapillary panos; perturbed box splits 812 s for 499 panos. Image decoding
ran in 4 threads beside the forward pass, and time in the forward pass was within 5% of wall-clock
in every pass; the A40 was shared with other jobs for most of the day, so 0.8-2.2 s per pano
reflects that sharing, not the model (0.78 s on an idle A40, CLAUDE.md batch-size note). CPU
steps: a few minutes on the desktop. Total about 4.0 GPU-hours (upper bound; GPU share unmeasured).

### 4.8 Where each number lives

| number | file | column(s) | produced by |
|---|---|---|---|
| residual table, reproduction | `data/decode_residual.csv` | `mean_px`, `published_mean_px`, `d_mean_px*` (path `rampnet` / `labeler`) | `residual` |
| gaussian agreement with RampNet | `data/decode_agreement.csv` | `max_gaussian_diff_px`, `only_*` | `residual` |
| measured sigma | `data/decode_sigma.csv` | `sd_*`, `robust_sd_*`, `size_*` | `residual` |
| ten cells at 4.24 / 3.67 | `data/decode_sigma_table.csv`, `decode_sigma_table_verdict.csv` | `world_recall`, `pass`, input sha256 | `sigma-table` |
| world, re-associated | `data/decode_world_<city>.csv` | one row per height x decode | `world` |
| world, frozen association and paired recall | `data/decode_world_pair_<city>.csv` | `d_loo_*`, `d_chi2_dof_*`, `responding_ramps`, `gained_gaussian`, `lost_gaussian`, `sign_test_p`, `rekey_*`, input sha256 | `world` |
| reproduction under perturbation | `data/decode_stability.csv`, `decode_stability_hist.csv` | shift columns | `stability` |
| inputs | `data/decode_inputs.csv` | sha256 of each `detect` output read | `residual` |
| Vancouver miss classes, `pixel-96` (§3) | `runs/vancouver/provenance_gate/report.md`, `unmatched.csv` | "why not reproduced" tables; `miss_class`, `nearest_cells`, `nearest_tier_cells` | `provenance_gate.py vancouver --control runs/vancouver/control_zoom3.jsonl --rule pixel-96` |
| Vancouver miss classes, coarse-cell (§3) | not committed (the default-rule report, re-run) | same | the same command without `--rule` |
| Richmond `--verify` classes (§3) | not committed (stdout) | `mismatch_classes`, `key_classes`; `--mismatch-csv` rows | `reinfer.py runs/richmond --verify --band-floor 0.30` |

### 4.9 What was not done

- **No default changed**: `decode` stays `argmax` and `sigma_peak_px` stays 1.0 everywhere.
- **No Hub re-export.** The published package still lacks `rampnet_subcell.py`; this repo
  vendors it.
- **Only two cities in world space**, at the benchmark tier. The production tier (0.30) and the
  PS-clustering arms were not re-run under the decode.
- **No flip TTA**: the labeler runs single pass, and so does everything here.
- **`detect_from_store.py` has no `--decode`**: a store run exists to reproduce live (argmax)
  labels.
- **No provenance gate for a gaussian campaign.** The gate refuses non-argmax arms; the classes
  in 4.5 are a proposal that nothing runs.
