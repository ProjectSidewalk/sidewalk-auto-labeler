# Does a per-rig confidence threshold beat one global threshold on open imagery? (overnight `per_rig_threshold`)

Run 2026-11-27, CPU only, no network, on judged RampNet bundles. Script:
`scripts/per_rig_threshold.py`. Data: `docs/overnight/data/per_rig_threshold/`.

## 0. Pre-registration (written before any threshold was scored on real data)

What I had seen before writing this: the published RampNet sweep (`../RampNet/docs/operating_point.md`,
including its per-tier table, whose tier optima "cluster at 0.26-0.35, which argues against per-tier
thresholds"), the per-split recall at 0.55 in `recall_by_tilt.md`, the rig mix of each bundle
(camera make/model counts), and a parity check that Lyon's stored run peaks at >= 0.55 equal its
bundle's records (200 of 200 panos identical). I had not computed any precision, recall or utility at any
threshold on these splits, by rig or otherwise.

### Data and units

- **Cities (8, open imagery):** richmond, clovis, morgantown, annapolis, laurens_mapillary, juneau
  (Mapillary); bayonne, lyon (Panoramax). budapest_district5 is excluded, as RampNet excludes it from
  every pooled recommendation (single rater, low reviewer confidence). GSV splits are out of scope.
- **Predictions.** The judged bundles hold only >= 0.55 detections, and the four oldest runs (richmond,
  clovis, morgantown, annapolis) store nothing below 0.55 either. So I use the predictions RampNet's own
  sweep used: `../RampNet/analysis_out/op_cache/<split>.json`, one no-TTA forward pass per bundle pano
  with every peak down to 0.05, parity-gated against `records.jsonl` (all Mapillary splits bit-exact,
  bayonne 98%). Lyon has no op_cache; its predictions are the production run's stored peaks for the 200
  audited panos (`../RampNet/analysis_out/lyon_review_159/run_peaks.json`, >= 0.10, pinned to the
  run's sha256). Then, to reproduce what production would ship:
  1. drop peaks below the production storage floor, 0.10 (production cannot threshold lower);
  2. drop peaks within 10 heatmap px (1024 x 512 heatmap) of the image edge, production's
     `exclude_border` rule (#130; the op_caches were extracted with `exclude_border=False`; Lyon's run
     peaks already obey it);
  3. drop peaks on the camera rig, `detectors.on_camera_rig` (the 49-degree nadir mask), as
     `send_to_ps.transform_record` does.
  A parity gate re-checks, per city, that the kept peaks at >= 0.55 reproduce the bundle's records
  (>= 95% of record detections have a kept peak within 0.5 match radius); a failure is reported.
- **Ground truth and matching: exactly RampNet's sweep semantics** (`rampnet.detection_eval.build_ground_truth`
  + `score_pano`, re-implemented here and checked against the GT embedded in each op_cache):
  GT ramps = detections with verdict `True` + missed-ramp marks without `unsure`; ignore points =
  `unsure` detections and `unsure` marks; `False` and `duplicate` are neither. Predictions are matched
  greedily in descending confidence to the nearest unclaimed GT strictly within the 0.022 radius
  (1024 x 512 scaled space, seam-wrapped); an unmatched prediction within radius of an ignore point is
  ignored, otherwise it is a false positive (a second hit on a claimed ramp included).
- **What this assumes about sub-0.55 detections (the bundle trap).** Reviewers judged only >= 0.55
  detections and clicked every ramp the model missed. A sub-0.55 peak is a TP only if it matches a
  missed-ramp click; if no click is near it, it is a FP. That is RampNet's sweep convention, and it
  assumes the reviewer clicked every missed ramp. Where they did not, a real ramp scores as a FP, so
  sub-0.55 precision is a **lower bound** (RampNet#55 found 13-35% of the incremental [0.25, 0.55) FPs
  were real ramps, per split). The #55 A/B tags exist for only five of these eight cities, so I do not
  apply them (applying them unevenly could itself create a rig effect); the bias pushes every arm
  toward higher thresholds, and a per-rig arm only gains from it if the A-rate differs by rig.
- **Unit set:** every pano whose verdict list aligns with its record, has no `None` verdict, and whose
  missed-ramp check is confirmed (`no_missed` or a missed mark). The same panos enter every arm, for
  precision and recall alike. The 5 hand-picked `top` panos per split are kept (as in RampNet's sweep);
  excluding them is a sensitivity.
- **Threshold grid:** 0.10 to 0.90 in steps of 0.01. A detection is kept at threshold t iff conf >= t.

### Rig key

`rig_class` from `scripts/recall_by_tilt.py` (camera make/model): gopro_max, gopro_fusion,
trimble_mx7, istar_pulsar, ladybug, other_360, unknown (no make/model). Panoramax producer is not used
in the primary key.

**Eligibility (computed on each training set):** a rig gets its own threshold only if the training
cities hold >= 100 GT ramps of it **and** at least 2 training cities each hold >= 20 GT ramps of it.
The two-city rule is what makes a rig threshold a rig threshold, not a city threshold under another
name. Every other rig is pooled into `other`, which gets its own threshold if its training panos hold
>= 100 GT ramps, else the global procedure's threshold. In the held-out city, a pano whose rig is not
eligible takes `other`'s threshold.

### Objective (recall-first, as a cost ratio)

Utility U = TP - lambda * FP, summed over panos (equivalently a cost of 1 per missed ramp and lambda
per false positive, since FN = GT - TP and GT is fixed). A threshold is chosen by maximizing U on the
training panos; ties go to the **lowest** threshold (recall-first).

**lambda = 1 (primary)**, derived from production's own choice, not from these data: on RampNet's
published raw pooled US sweep, the marginal precision between 0.25 and 0.30 is 0.445 and between 0.30
and 0.35 is 0.51 (from the table's P x detections/pano), so 0.30 maximizes TP - lambda FP for lambda in
[0.80, 1.05]. lambda = 1 is the round value in that range: keep a detection when it is at least as
likely real as not. lambda = 0.5 (more recall-first) and lambda = 2 are non-gating sensitivities.

**Precision floor 0.85**: the lowest raw precision production accepted on a trusted pooled US split at
0.30 (clovis, raw P 0.851 in `operating_point.md`).

### Leave-one-city-out and arms

For each held-out city, thresholds are chosen on the other seven cities and scored on the held-out one.

- **A0 production:** global 0.30.
- **A1 global-procedure (the fair control):** one threshold, maximizing U on all training panos.
- **A2 per-rig:** one threshold per eligible rig (and `other`), each maximizing U on that rig's training
  panos.
- **A3 shuffled-rig control:** as A2, but before fitting, rig labels are permuted across panos within
  each training city (200 seeded draws). Held-out panos keep their true labels. City composition is
  preserved, so A3 shows how much of A2 any arbitrary within-city partition would buy.

### Metric and decision rule

Primary metric: pooled held-out dU = (sum over held-out cities of U_A2 - U_A1) / (sum of GT ramps),
reported per 100 GT ramps. CI: cluster bootstrap by pano, 2,000 draws, seed 20261127; each draw
resamples panos within every city and **re-runs the whole procedure** (eligibility and threshold
selection included), so the CI carries selection noise.

**Per-rig PASSES only if all three hold (lambda = 1):**

1. pooled dU(A2 - A1) > 0 with its 95% bootstrap CI excluding 0;
2. it beats the shuffled control: fewer than 10 of the 200 shuffled draws reach a pooled dU(A3 - A1) at
   least as large as the real dU(A2 - A1);
3. precision guard, in every held-out city: if A1's precision is >= 0.85, A2's must be >= 0.85; if A1's is
   below 0.85, A2's must not be more than 0.03 below A1's.

Non-gating, reported: A1 vs A0 (does the procedure itself beat production's 0.30?); per-city P/R/U for
every arm; the thresholds each fold chose per rig; the held-out-oracle threshold per city (ceiling,
chosen on the test city itself); sensitivities (lambda 0.5 and 2; rig key x imagery source; eligibility
with one city allowed; equal weight per city; `top` panos excluded); survivorship.

What this cannot see: anything below the 0.10 storage floor; GT incompleteness below 0.55 (above);
rigs that appear in only one city cannot be separated from that city by any design, which the
two-city rule makes explicit rather than solving; Lyon's sample is stratified by rig group, so its
rig mix is not the city's.

## 1. Verdict

**FAIL.** A per-rig threshold does not beat one global threshold chosen the same way. It fails
criteria 1 and 2 and passes only the precision guard.

| criterion (lambda = 1, 8 held-out cities pooled) | result | passes |
|---|---|---|
| 1. dU(A2 per-rig − A1 global-procedure), per 100 GT ramps, 95% CI | **−0.66 (−2.21, +1.48)** | no |
| 2. shuffled-rig draws reaching that dU | **94 of 200** (shuffle median −0.73, 95th pct +0.00) | no |
| 3. precision guard in every held-out city | holds in 8 of 8 | yes |

The per-rig arm is slightly worse than the global one, and no better than an arbitrary partition
of the panos inside each training city. Rig labels carry no threshold information that survives
leave-one-city-out.

Production's 0.30 holds up as well: the procedure-chosen global threshold (0.32 in 7 folds, 0.42
with Laurens held out) scores dU(A1 − A0) = **−0.31 (−2.01, +0.78)** per 100 GT against 0.30.

**Why it fails: the best threshold is a property of the city, not the camera.** On the full data,
the U-optimal threshold for one rig moves across cities by more than the rigs differ from each
other (`rig_curves.csv`, `best_lam1` rows):

| rig | city: U-optimal threshold (lambda = 1) |
|---|---|
| GoPro Max | laurens_mapillary **0.13**, bayonne 0.25, richmond 0.33, morgantown 0.38, lyon **0.45** |
| Trimble MX7 | juneau 0.26, annapolis 0.44 |
| Lyon, four rigs in one city | other_360 0.19, ladybug 0.38, unknown 0.42, GoPro Max 0.45 |
| one rig, one city | GoPro Fusion (clovis) 0.35, iSTAR Pulsar (richmond) 0.45 |

So a GoPro Max threshold learned elsewhere is wrong for Laurens in one direction (fit 0.45, Laurens
wants 0.13) and for Lyon in the other (fit 0.25, Lyon wants 0.45). The same pattern is in RampNet's
per-tier table (`../RampNet/docs/operating_point.md`: tier optima 0.26-0.35) and in
`recall_by_tilt.md` (recall is a city effect, not a rig effect).

**The ceiling is small anyway.** A held-out oracle (each city's own best single threshold, chosen
in-sample, so optimistic) gains only +2.2 per 100 GT over 0.30 pooled at lambda = 1. Almost all of
that is Laurens, where 0.13 instead of 0.30 gains +7.2 per 100 GT (recall 0.526 → 0.675 at precision
0.885 → 0.824). Nothing trained on other cities predicts that.

## 2. Results

### Primary, per held-out city (`per_city.csv`; U = TP − FP per 100 GT ramps)

| held out | GT | A0 0.30: P / R / U | A1 global: P / R / U | A2 per-rig: P / R / U | oracle t: U |
|---|---:|---|---|---|---|
| richmond | 310 | 0.902 / 0.829 / 73.9 | 0.911 / 0.829 / 74.8 | 0.907 / 0.816 / 73.2 | 0.33: 75.5 |
| clovis | 195 | 0.851 / 0.821 / 67.7 | 0.855 / 0.815 / 67.7 | 0.886 / 0.795 / 69.2 | 0.35: 69.7 |
| morgantown | 267 | 0.896 / 0.805 / 71.2 | 0.911 / 0.805 / 72.7 | 0.870 / 0.805 / 68.5 | 0.38: 73.0 |
| annapolis | 294 | 0.902 / 0.810 / 72.1 | 0.912 / 0.806 / 72.8 | 0.928 / 0.786 / 72.4 | 0.44: 73.8 |
| laurens_mapillary | 249 | 0.885 / 0.526 / 45.8 | 0.892 / 0.430 / 37.8 | 0.897 / 0.418 / 36.9 | 0.13: 53.0 |
| juneau | 297 | 0.915 / 0.758 / 68.7 | 0.922 / 0.751 / 68.7 | 0.935 / 0.731 / 68.0 | 0.26: 69.4 |
| bayonne | 301 | 0.760 / 0.495 / 33.9 | 0.762 / 0.478 / 32.9 | 0.762 / 0.478 / 32.9 | 0.25: 34.9 |
| lyon | 967 | 0.745 / 0.409 / 26.9 | 0.765 / 0.396 / 27.4 | 0.749 / 0.411 / 27.3 | 0.42: 29.2 |
| **pooled** | 2,880 | U 50.10 | U 49.79 | U 49.13 | U 52.33 |

Per-rig wins in one city (clovis +1.5) and loses in five (morgantown −4.1, richmond −1.6, laurens
−0.8, juneau −0.7, annapolis −0.3). It ties bayonne, where the only eligible rig's threshold equals
the global one, and loses 0.1 in lyon.

### Thresholds each fold chose (`thresholds.csv`, primary)

| held out | A1 global | GoPro Max | Trimble MX7 | unknown | other |
|---|---:|---:|---:|---:|---:|
| richmond | 0.32 | 0.25 | 0.44 | n.e. | 0.38 |
| clovis | 0.32 | 0.27 | 0.44 | 0.42 | 0.38 |
| morgantown | 0.32 | 0.25 | 0.44 | 0.42 | 0.38 |
| annapolis | 0.32 | 0.27 | n.e. | 0.42 | 0.38 |
| laurens_mapillary | 0.42 | 0.45 | 0.44 | 0.42 | 0.38 |
| juneau | 0.32 | 0.27 | n.e. | 0.42 | 0.38 |
| bayonne | 0.32 | 0.32 | 0.44 | 0.42 | 0.38 |
| lyon | 0.32 | 0.25 | 0.44 | n.e. | 0.35 |

n.e. = not eligible in that fold (fewer than two training cities with >= 20 GT of the rig), so it
falls into `other`. GoPro Fusion, iSTAR Pulsar, Ladybug and other_360 are single-city rigs and are
never eligible under the two-city rule; they always take `other`. Only GoPro Max is eligible in every
fold, so in practice the primary arm is "GoPro Max vs everything else".

### Sensitivities (non-gating; `summary.csv`, 500 bootstrap draws, 50 shuffles)

| analysis | dU(A2 − A1) (95% CI) | dU(A1 − A0) (95% CI) | shuffles reaching | guard |
|---|---|---|---|---|
| **primary** (lambda 1) | −0.66 (−2.21, +1.48) | −0.31 (−2.01, +0.78) | 94 / 200 | holds |
| lambda 0.5 | −1.34 (−2.12, +0.36) | **+1.98 (+0.07, +2.92)** | 48 / 50 | holds |
| lambda 2 | −0.10 (−2.80, +1.12) | **+5.63 (+2.84, +7.92)** | 14 / 50 | holds |
| rigs from one city allowed | −0.80 (−1.91, +1.02) | −0.31 (−1.97, +0.74) | 21 / 50 | holds |
| rig x source (Mapillary / Panoramax) | −0.49 (−3.23, +1.18) | −0.31 (−1.97, +0.74) | 7 / 50 | **fails** (morgantown 0.911 → 0.810) |
| equal weight per city | −0.76 (−2.92, +1.85) | −0.67 (−2.71, +0.51) | — | holds |
| `top` panos excluded | −0.78 (−2.34, +1.36) | −0.35 (−2.19, +0.86) | 39 / 50 | **fails** (morgantown 0.892 → 0.833) |

- **No variant favours per-rig.** Every dU(A2 − A1) point estimate is negative and every CI
  includes 0.
- **The two guard failures are the same mechanism.** Holding Morgantown out leaves Laurens as the
  largest GoPro Max (Mapillary) training city. Laurens wants a very low threshold, so the rig
  threshold drops to 0.20 (0.24 with `top` panos excluded), and Morgantown's precision falls below
  the floor. A rig threshold
  learned from a few cities inherits whichever city dominates it.
- **lambda 0.5 and 2 move the global threshold, as they should.** The procedure picks 0.19 at
  lambda 0.5 and 0.52-0.55 at lambda 2, and each beats 0.30 under its own objective. lambda = 1 is
  the cost ratio at which 0.30 is optimal (section 0), so this only confirms that the objective
  is calibrated. It is not a finding about 0.30.

### Survivorship and parity (`survivorship.csv`, `parity.csv`)

- **Panos.** All 1,043 bundle panos of the 8 cities enter every arm: none misaligned, none with a
  `None` verdict, every one recall-confirmed. They hold 2,880 GT ramps.
- **Peaks.** Of 4,115 low-floor peaks, these were dropped before scoring:
  - 767 below the 0.10 storage floor, from the op_caches; Lyon's run peaks start at 0.10.
  - 22 by production's border rule: bayonne 15, juneau 5, laurens 2.
  - 63 by the 49-degree rig mask: laurens 33, lyon 21, morgantown 9.
- **Parity.** The kept peaks at >= 0.55 reproduce 100% of every bundle's record detections within
  0.5 match radius (1,676 of 1,676 outside the rig mask). My rebuilt GT equals the GT embedded in every op_cache, on
  every pano.
- **Sanity check against RampNet's sweep.** At 0.55 these give clovis P 0.915 / R 0.713,
  annapolis 0.973 / 0.738 and laurens_mapillary 0.898 / 0.390, against RampNet's 0.914 / 0.713,
  0.973 / 0.738 and 0.898 / 0.390. Bayonne and morgantown differ slightly, only because of the
  border rule and the rig mask.

## 3. Caveats

- **Sub-0.55 precision is a lower bound.** A sub-0.55 peak is credited only if a reviewer clicked
  that ramp as missed. RampNet#55 measured 13-35% of incremental [0.25, 0.55) FPs as real ramps on
  the tagged splits. Tags do not exist for laurens_mapillary, bayonne or lyon, and were not applied
  anywhere here. If Laurens's or Bayonne's A-rate is high, their true optima sit lower still. That
  widens the city spread; it does not create a rig signal.
- **Small and unbalanced.** The design has 8 cities, and Lyon is 34% of the GT. Most rigs live in
  one city, so no design can separate them from their city. The two-city rule makes that explicit.
  It also leaves the primary test mostly "GoPro Max vs the rest".
- **The CI is wide** (about ±2 per 100 GT). An effect of +1 per 100 GT could not be detected. But
  the point estimate is negative, and the city spread above shows no mechanism for a positive one.
- **Lyon's sample was stratified by rig group** (`group_map.json`), so its rig mix is not the city's.
  Its predictions are production's run peaks, not a no-TTA re-extraction. Parity holds exactly.
- **What this does not test:** a per-city threshold tuned on that city's own validations, which is
  the only lever with a visible ceiling (Laurens). That needs judged data from the target city, the
  thing a new city lacks. Producer within one rig (Lyon's ecartip vs others) is also untested; it is
  too thin here.
- **Out of scope:** thresholds below the 0.10 storage floor, GSV, and budapest_district5.

## 4. Replication

```bash
cd <labeler checkout>   # with ../RampNet holding benchmark/ and analysis_out/{op_cache,lyon_review_159}
python scripts/per_rig_threshold.py --rampnet-root ../RampNet \
    --out docs/overnight/data/per_rig_threshold          # ~40 s, CPU, no network
python -m pytest -q tests/test_per_rig_threshold.py
```

Inputs, all read-only:
- `../RampNet/benchmark/<split>/{records.jsonl,verdicts.json}` for the 8 splits.
- `../RampNet/analysis_out/op_cache/<split>.json` for the 7 non-Lyon splits.
- `../RampNet/analysis_out/lyon_review_159/run_peaks.json`, pinned to Lyon run sha256 `60e688b2…`.

Outputs:
- `summary.json` holds the verdict and the per-fold detail.
- `summary.csv` holds every arm and sensitivity.
- `per_city.csv`, `thresholds.csv`, `guard.csv`, `rig_curves.csv`, `rig_sizes.csv`,
  `survivorship.csv` and `parity.csv` hold the rest.

The run is seeded (`--seed 20261127`) and deterministic.
