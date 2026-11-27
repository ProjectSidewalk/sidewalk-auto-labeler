# GT-free geometric filters for Juneau AI Crosswalk false positives (overnight `crosswalk_fp_filters`)

Run 2026-11-27, CPU only, no network. Script: `scripts/crosswalk_fp_filters.py`. Data:
`docs/overnight/data/crosswalk_fp_filters/`. Context: https://github.com/ProjectSidewalk/RampNet/issues/298
(Juneau Crosswalk pilot), https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/168 (multi-type).

## 0. Pre-registration (written before any filter was scored against a vote)

**What I had seen before writing this.** The validation README, `random_sample_2026-11-17.json`, the
validator report (unfiltered precision 0.72 [0.65, 0.79] on the random sample), the brief's
description of the false-positive classes (capture-car roof "mostly at the 360-degree seam"; brick/paver
sidewalk 29-38 degrees below the horizon beside the vehicle), and these GT-free distributions of the
3,579 sent labels: dip and image-azimuth histograms per rig, the most recurrent heatmap cells per rig
(Trimble MX7: dip 7-13 degrees at the seam and at the centre column, each in 7-10 of 33 sequences), and
the camera-heading-to-travel offset per rig (below). I had not cross-tabulated any vote, `fp_class` or
comment against any geometric feature.

**A GT-free fact that changes the brief's framing.** In `runs/juneau/results.jsonl` (the same 4,999
panos), 97% of Trimble MX7 frame-to-next-frame steps have the camera heading (the image centre column)
within +/-30 degrees of the travel bearing **plus 180**: the MX7's centre column faces backward, so its
360-degree seam faces **forward** (unless capture times run backwards, which I cannot check offline). Most
GoPro Max steps face forward (one sequence about -120). So "at the seam" is in front of the Trimble, not
behind it. The filters below therefore use a vehicle-relative azimuth computed from the data, not the
image seam.

### Data and units

- **Population:** all 3,579 labels in `runs/juneau_crosswalk/validation/validator_r3_all_labels_scored.csv`
  (3,525 `juneau_crosswalk` + 54 `juneau_expansion_crosswalk`; 3,287 Trimble MX7, 292 GoPro Max). They are
  already rig-masked (49 deg), range-gated (5.625 deg) and >= 0.23. Keys are `juneau-ak:<label_id>`.
- **Heatmap position:** col = pano_x / width x 1024, row = pano_y / height x 512 (rounded); dip =
  (row/512 - 0.5) x 180 deg below the horizon (the image frame; no pose correction, as production);
  image azimuth az_img = (col/1024 - 0.5) x 360 (0 = centre column).
- **Vehicle-relative azimuth** rel_az = wrap(az_img + offset(seq)), 0 = direction of travel. offset(seq)
  = circular mean, over consecutive frames of the sequence (ordered by `captured_at`) at least 2 m apart,
  of (camera_heading - bearing(frame -> next frame)), from `runs/juneau/results.jsonl`. A sequence with
  fewer than 3 such pairs, or absent from that file (all 54 expansion labels), uses its rig's offset
  (circular mean over all of the rig's pairs).
- **Votes:** `validation_result` of the scored CSV (both of Jon's accounts; the two doubly voted labels
  are Agree twice). y = 1 for Agree, 0 for Disagree. **Unsure is treated as missing** (dropped from its
  set and its stratum's n); sensitivity: Unsure = Disagree.
- **Confirmatory set:** the seeded random sample (`random_sample_2026-11-17.json`, seed 20261117), 80
  labels drawn 50 from the 2,717 an OSM-proximity rule would keep and 30 from the 862 it would drop (the
  rule is a stratifier only; it is not adopted). Voted: 79 (`juneau-ak:7666` has no vote in the CSV).
  **Weights** w_h = N_h / n_h, with n_h the labels of stratum h holding an Agree/Disagree vote: expected
  2,717/49 = 55.45 (kept stratum) and 862/30 = 28.73 (dropped stratum) if no sample label is Unsure.
- **Tuning set:** every voted label NOT in the random sample (the first ~236-vote batch and the second
  batch's non-sample labels). The 5 random-sample labels that were already voted in the first batch belong
  to the confirmatory set and are excluded from tuning.

### Arms (all GT-free in what they remove; only their parameters are tuned)

- **A. Rig-cell mask (persistence across sequences).** Per camera model, cut the 1024 x 512 heatmap into
  square cells of c px (x wraps; cell = (col // c, row // c)). A cell is in the mask iff the sent labels
  falling in it come from >= S distinct sequences of that camera model. Remove every label in a masked
  cell. Grid: c in {8, 16}, S in {3, 4, 5, 6, 8, 10}.
- **B. Steep-dip cut.** Remove labels with dip > D. Grid: D in {20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5,
  40, 45}.
- **C. Dip x beside-the-vehicle band.** Remove labels with D_lo <= dip <= D_hi and
  ||rel_az| - 90| <= H (either side of the vehicle; H = 90 is a pure dip band). Grid: D_lo in {20, 25,
  27.5, 30}, D_hi in {35, 40, 45, 49}, H in {30, 45, 60, 90}.
- **D. Combination:** A* union X*, where X* is whichever of B*, C* has the higher tuning-set kept
  precision (tie: fewer labels removed). Not tuned further.
- **Control:** random removal of the same number of labels as the arm, uniform over the 3,579 (5,000
  seeded draws).

**Tuning rule (tuning set only, unweighted).** Eligible: the cell removes <= 5% of the tuning set's
Agree labels. Pick the eligible cell with the highest kept precision on the tuning set; ties -> fewer
labels removed from the 3,579 -> the grid's earlier entry. If no cell is eligible, the arm FAILS tuning
and the highest-precision cell is scored anyway, for description.

### Metrics (confirmatory, random sample, weighted to 3,579)

- Kept-set precision P(F) = sum_{sampled, kept} w y / sum_{sampled, kept} w; the unfiltered P(none) should
  reproduce 0.72.
- Paired gain Delta = P(F) - P(none), 95% CI from a cluster bootstrap by `sequence_id` (10,000 replicates,
  seed 20261127; weights re-derived as N_h / n_h in each replicate).
- Labels removed: exact count and share of 3,579. Estimated true crosswalks lost = sum_{sampled, removed}
  w y, and as a share of the estimated total sum w y (bootstrap CI).
- Control p-value: share of random-removal draws whose Delta >= the arm's Delta.

### Decision rule (per arm, A*, B*, C*, D*)

An arm is **RECOMMENDED** iff, on the random sample, (i) the paired Delta's 95% CI excludes 0 (lower bound
> 0); (ii) it beats the random-removal control (p < 0.05); and (iii) its estimated loss of true crosswalks
is <= 5% of estimated true positives (point estimate; CI reported). Otherwise NOT RECOMMENDED; if (i) fails
only because the sample cannot resolve the effect (e.g. the arm touches few sampled labels), I say so and
report the tuning-set numbers as **exploratory only** (they are optimistic: tuned there). Four arms are
tested with no multiplicity correction; a pass of only D* with a borderline CI should be read with that in
mind.

### Reported, exploratory (never gate)

Survivorship per arm (removed by rig, run, stratum, sample set); per-arm metrics on the tuning set; the
`fp_class` of every voted Disagree left after the best arm (all votes) as the hard-negative list for the
next retrain; Unsure-as-Disagree sensitivity.

### What this cannot see

Recall losses on crosswalks never sent (below 0.23, rig-masked or range-gated); other cities and rigs
(Juneau is 92% one Trimble MX7 fleet); Jon is the only rater; the random sample is 79 labels, so effects
on a class of a few percent of the population are hard to resolve.

---

## 1. Results (computed 2026-11-27, after the pre-registration above)

**Inputs.** 3,579 labels (none below the nadir mask, checked); 37 sequences with a measured offset from
`runs/juneau/results.jsonl` (every main-run label uses its sequence's offset; the 54 expansion labels use
their rig's). Tuning set: 265 labels (159 Agree, 106 Disagree; the 4 Unsure votes are all outside the
sample and are dropped). Confirmatory set: 79 voted labels in 20 sequences, **no Unsure among them**, so
the Unsure-as-Disagree sensitivity is identical to the primary analysis and is not repeated below.
**Weights: kept stratum 2,717/49 = 55.45, dropped stratum 862/30 = 28.73.** Unfiltered precision
reproduces the README: **0.722 [0.656, 0.783]** (sequence-cluster bootstrap).

**Camera offsets (GT-free, `offsets.csv`).** Every Trimble MX7 sequence has its centre column 176-180 deg
from the travel bearing (share of steps within 30 deg of the sequence mean: 0.97-1.00). The GoPro Max
"rig" is at least two mounts: 2 sequences face forward (-2, +1 deg), 3 face about -121 deg. The rig-mean
fallback for GoPro (-34 deg) is therefore a poor stand-in for the 34 expansion GoPro labels.

### Tuned parameters (`tuning_grid.csv`)

| arm | chosen | eligible (<= 5% tuning Agrees lost)? | tuning precision kept (from 0.600) | tuning Agree / Disagree removed |
|---|---|---|---|---|
| A rig-cell mask | c = 16 px, S = 3 | **no cell eligible: A FAILS tuning** | 0.605 | 136/159 / 91/106 |
| B steep-dip cut | D = 35 deg | yes | 0.647 | 7/159 / 23/106 |
| C dip x beside band | dip 30-40 deg, within 45 deg of abeam | yes | 0.654 | 6/159 / 25/106 |
| D = A* union C* | (C beat B on tuning) | n/a | 0.677 | 136/159 / 95/106 |

Why A failed: the cells that recur across sequences are the near-horizon cells straight ahead and behind
(dip 7-13 deg), where real crosswalks at intersections appear in drive after drive. Even the strictest
cell (c = 8, S = 10, 296 labels) removed 20 tuning Agrees and **0** Disagrees. Cross-sequence pixel
recurrence finds popular crosswalk positions, not the rig, the same failure `rig_persistence.md` found for
CurbRamp.

### Confirmatory: random sample, weighted to 3,579 (`arms.csv`, mode `primary`)

| arm | labels removed (share) | kept precision [95% CI] | paired gain [95% CI] | est. true crosswalks lost (share [CI]) | control p | sample removed (Agree/Disagree) | (i) | (ii) | (iii) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.722 [0.656, 0.783] | - | - | - | - | | | | |
| A (failed tuning) | 3,258 (91.0%) | 0.537 [0.000, 0.797] | -0.185 [-0.716, +0.077] | 2,417 (93.6% [84, 100]) | 0.85 | 46 / 25 | no | no | no | NOT RECOMMENDED |
| B dip > 35 | 347 (9.7%) | 0.737 [0.667, 0.795] | +0.016 [0.000, +0.027] | 111 (4.3% [0, 12.1]) | 0.16 | 2 / 4 | no | no | yes | NOT RECOMMENDED |
| **C dip 30-40, abeam +/-45** | **366 (10.2%)** | **0.754 [0.681, 0.808]** | **+0.033 [+0.004, +0.060]** | **55 (2.1% [0, 8.0])** | **0.024** | **1 / 6** | yes | yes | yes | **RECOMMENDED (narrowly)** |
| D = A union C | 3,296 (92.1%) | 0.537 [0.000, 0.797] | -0.185 [-0.716, +0.077] | 2,417 (93.6%) | 0.84 | 46 / 25 | no | no | no | NOT RECOMMENDED |

Control p = share of 5,000 same-size random removals whose gain was at least the arm's. Bootstrap: 10,000
sequence-cluster replicates, seed 20261127.

### Verdict against the pre-registered rule

- **C (dip 30-40 deg below the horizon AND within 45 deg of straight abeam of the vehicle, either side)
  passes all three criteria and is RECOMMENDED**, but by the thinnest margin the rule allows. The pass
  rests on **7 sampled labels** (6 brick-sidewalk Disagrees, 6 of the 7 in the OSM-"dropped" stratum, and
  1 Agree, `juneau-ak:6835`). The paired CI's lower bound is +0.004; one more sampled Agree in the band
  would put it at or below 0. Four arms were tested with no multiplicity correction. Read it as "a cheap
  filter that removes about 10% of labels, most of them brick sidewalk, at an estimated 2% (CI up to 8%)
  loss of real crosswalks", not as a precise effect size.
- **B** removes about as many labels, but also roof hits on the Trimble and more Agrees (9 of 208 Agree
  votes overall vs C's 7); its gain's CI touches 0 and it does not beat random removal (p 0.16).
- **A and D fail**: the rig-cell mask cannot be tuned to lose <= 5% of real crosswalks.

### What C removes, and what is left (all votes, exploratory; `remaining_fp.csv`)

C removes 27 of the 40 brick/paver Disagrees and 4 of 11 fence/tree shadows, and **none of the 52
capture-car-roof Disagrees**. It removes 7 of the 208 Agree votes (all Trimble, dip 32-35 deg, abeam).
Survivorship (`survivorship.csv`): 351 of 3,287 Trimble and 15 of 292 GoPro labels; 14 of the 54
expansion labels (whose azimuth uses the rig fallback offset).

Disagree classes left after C, the hard-negative list for the next retrain (RampNet#298): **capture-car
roof 52**, brick/paver sidewalk 13, fence/tree shadows 7, stairs 6, parking-lot / access-aisle stripes 5,
other road markings (chalk, paint, bike lane) 4, turn lane 4, curb ramp 3, driveway crossing 3, and one
each of fence, median, other car roof, placement off the crosswalk, plain sidewalk, window reflection
(plus 2 without a comment).

### Exploratory, post hoc: where the roof hits are, and Amendment 1

Written 2026-11-27 AFTER the confirmatory results above. Nothing here changes the verdict.

**Roof geometry.** 37 of the 52 roof Disagrees are on the **GoPro Max**, 33 of them in the two
forward-facing GoPro sequences (`eLOsiMypD8rIQc0qlHVAPg` 25, `9hLvjleCpqNisY1EJdm7nS` 8), at the image
seam (cols 0-28 and 1012) and dip 6.7-9.8 deg, i.e. directly **behind** the vehicle just below the horizon.
The 15 Trimble roof hits sit at the image seam too, but the Trimble seam faces **forward**, and they are
steep (dip 24-49 deg; one at 9.5 deg abeam). So "the seam" is the right place for both rigs, but it means
behind on the GoPro and ahead on the Trimble. The pre-registered arms missed the GoPro roof for two reasons.
Only 6 GoPro sequences exist, the roof shows in 2, and the cross-sequence threshold was S >= 3. And the
same cells hold real crosswalks on the Trimble fleet.

**Amendment 1: within-sequence recurrence** (`amendment1_within_sequence.csv`). Flag a label whose 8- or
16-px cell holds labels from >= k panos of its own sequence (a fixture riding along). At c = 8, k = 8 it
removes 82 labels, 14 of the 52 roof Disagrees and 15 Disagrees in all, but also 4 Agrees, and **3 of the 4
random-sample labels it removes are Agrees**. No setting separates roof from real crosswalks seen ahead
on long straight drives. Not worth pursuing in this form. A rig-specific "GoPro rear seam, dip < 12 deg"
band would remove most GoPro roof hits, but I derived it from the votes, and the random sample holds only
5 GoPro labels, so it cannot be confirmed here. It needs a fresh GoPro test set, or per-sequence masks built
from the full stored detections (>= 0.05) rather than the sent labels.

### Context

The offline production Crosswalk validator (`validator_r3_REPORT.md`) reaches 0.94 by dropping its own
Disagree verdicts. It keeps 93% of real crosswalks and rejects 51 of 52 roof and 40 of 40 brick labels.
C is a much smaller, GT-free, image-free complement: about +3 points.

## 2. Caveats: what this cannot see

- **Tiny confirmatory set.** 79 labels in 20 sequences. C's pass hinges on 7 of them, and only 5 sampled
  labels are GoPro, so nothing about the GoPro roof can be confirmed.
- **One rater.** Jon is the only rater, under two accounts. The CSV is one vote short of the database,
  and `juneau-ak:7666`, a sample label, is unvoted.
- **Recall.** Crosswalks never sent (below 0.23, masked, range-gated) are invisible. "True crosswalks lost"
  counts only the sent labels' Agrees.
- **One city.** Juneau, 92% one Trimble MX7 fleet. Brick sidewalks beside the road are a downtown-Juneau
  feature, and the 30-40 deg band is in image coordinates for a rig of this height. Other cities and rigs,
  and GSV, are untested. A city whose crosswalks are often seen abeam at 30-40 deg (camera at an
  intersection, crossing street to the side) could lose more.
- **Azimuth.** The vehicle-relative azimuth assumes `captured_at` order is travel order. If it were reversed,
  "abeam" is unaffected (the band is symmetric front-to-back), but "Trimble seam = forward" would flip. The
  expansion GoPro labels use a poor fallback offset.
- **Tuning set.** It is clustered, non-random voting from the same city and rater, so the tuning precisions
  are optimistic.

## 3. Replication

```bash
cd <worktree>
python scripts/crosswalk_fp_filters.py \
    --validation-dir runs/juneau_crosswalk/validation \
    --results runs/juneau/results.jsonl \
    --out docs/overnight/data/crosswalk_fp_filters
pytest -q tests/test_crosswalk_fp_filters.py
```

The run is deterministic: bootstrap and control seeds are 20261127, and two runs gave byte-identical
outputs. It takes about 20 s on one core. Outputs: `offsets.csv` (camera-to-travel offset per sequence),
`tuning_grid.csv`, `arms.csv` (`primary` and `unsure_as_disagree` modes), `survivorship.csv`,
`remaining_fp.csv`, `rig_cells.csv` (A*'s mask), `amendment1_within_sequence.csv`, `summary.json`.
