# Footway: do an off-the-shelf segmenter and GSV depth agree on the walkable surface?

**Issue:** [#47](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/47), step 2: *run a pretrained
Mask2Former/SegFormer over a few hundred panos and measure agreement against "depth says a near-horizontal
plane ~camera-height below". Disagreement is the interesting output either way.*

**Date:** 2026-09-30, revised the same day after review of
[#124](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/124) (§9). **Branch:** `footway-depth-47`.

**Reproduce:** `scripts/footway_segmentation.py` produces every number and figure below; the full replication recipe, input hashes and versions are in §8.

**Tests:** `tests/test_footway_segmentation.py`, pure functions only; the model is never loaded. It covers:
- the equirect/perspective tile mapping round trip, seam wrap, and the render-then-stitch recovery of a
  labelled equirect;
- the vote tie-break and the class collapse;
- a synthetic 512x256 payload that pins `compare_pano`'s grid↔payload indexing, the mirrored null and the
  reference-pixel lookup;
- kappa and the lift arithmetic, and the verdict rule.

**Pre-registration:** the [dated #47 comment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/47#issuecomment-5921395361)
fixed the sample, model, tiling (including the unseen zenith above ~+43° and nadir below ~−79°), class mapping and
readings before any GT-conditioned number was computed.
- Commit `d1ffbb6` put the rule into code (`verdict()`), together with the headline denominators that leave
  out pixels no tile sees.
- `75f2d05` only fixed a payload-parse crash and range-restricted the per-pano `n_*_px` counts.
- Anything marked *exploratory* or *post hoc* was added after scoring and enters no verdict.
- The null arm, the solid-angle and per-pano columns, kappa, and the 4096 px direct arm were added on review;
  they change interpretation, not a rule (§9).

**Builds on:** step 1, [docs/depth-at-detection-study.md](depth-at-detection-study.md). Its depth classes, plane
lookup and `offset_local` are imported, not reimplemented.

> **Key takeaways**
>
> 1. **Surface "agreement" between GSV depth and the segmenter is a base rate, not evidence.** P(WALK or ROAD | depth
>    surface) is 0.897 against a marginal of 0.884, and a scrambled-depth null gives 0.894 / 0.895 (fig. 1;
>    `data/headline.csv` columns `p_walkroad_given_surface`, `p_walkroad`, rows `city=pooled, range=le25m,
>    weighting=pixels`, `depth_arm=correct/yaw180/mirror`).
> 2. **The instruments do agree where depth is informative, on the walls.** P(STRUCTURE | depth wall) is 0.661
>    against 0.415 / 0.470 under the nulls, and kappa is 0.250 against 0.169 / 0.188 (figs. 1 and 3;
>    `headline.csv` `p_structure_given_wall`, `kappa`; `data/fig3_where_signal.csv` `lift`).
> 3. **Depth has no object-shaped holes.** 94.7% of segmenter-OBJECT pixels within 25 m sit on a depth floor plane,
>    against 93.5% / 93.6% when depth's geometry is scrambled. Above the near-nadir rows (y ≤ 0.75), the figures
>    are 94.1% against 92.8% / 92.8%. The scrambled-geometry null is the valid control here (fig. 1, with fig. 2 as
>    examples; `headline.csv` `p_surface_given_object`, rows `weighting=pixels` and `pixels_y_le_0.75`).
> 4. **Neither instrument is a false-positive filter at the peak pixel, and GSV depth shows no ~0.15 m curb
>    step, at most a few centimetres.** 2 of 42 verdict-False detections are off WALK/ROAD, against 43 of 770 True
>    (registered (C): NOT SUPPORTED). The sidewalk plane sits 0.022 m [0.014, 0.030] above its local reference
>    plane; the exploratory road-referenced reading is 0.042 m [0.025, 0.055] (figs. 4 and 5;
>    `data/verdict.json` `tiers.0.55.*`, `curb_height`; `data/curb_height.csv` `median_of_pano_medians`).
> 5. **Feeding the equirect directly fails on the nadir fill, and that failure varies by city.** At the tiles'
>    resolution the direct arm keeps Curb Cut (post hoc: 0.301 vs 0.271 tiled). At −80..−70° it instead labels
>    the tiled arm's ROAD under the car as SKY: 0.000 in Bend, 0.110 in Paterson, 0.097 in São Paulo and 0.247 in
>    Gainesville (fig. 6; `data/detection_classes.csv` `share_curb_cut`; `data/trap.csv`
>    `direct4096_share_SKY`, rows by city).
>
> `python scripts/footway_segmentation.py check-numbers` re-reads 236 quoted numbers from committed files, keyed by
> file, row and column, and exits non-zero on any mismatch. Each row records where the number is quoted
> (`data/numbers.csv` `quoted_in`; §10). The list is curated by hand, so a number missing from it is not checked.

## 0. Summary

The sample is 416 judged benchmark panoramas from bend, paterson, gainesville and São Paulo, each with a
`measured` depth payload. Each was segmented with `facebook/mask2former-swin-large-mapillary-vistas-semantic` on 16
perspective tiles, and the label maps were stitched back onto a 1024x512 equirect grid.

1. **Below the horizon, depth puts a floor plane almost everywhere, so "agreement on the surface" is mostly a base
   rate.**
   - Within 25 m, depth's floor (ground ∪ floor ∪ stand-in floor) covers 98.3% of the pixels a tile sees.
   - The segmenter calls 88.4% of the seen within-25 m pixels WALK or ROAD.
   - P(WALK or ROAD | depth surface) = 0.897 is therefore a lift of only 1.01 over that marginal. A null with the
     depth index rotated 180° in azimuth gives 0.894, and a mirrored one 0.895.
   - The pixel pool is also nadir-heavy: 71% of within-25 m pixels lie in the 0–5 m bin, mostly road under the
     car.
   - Solid-angle weighting gives 0.863 (marginal 0.845), and the per-pano median gives 0.917 (marginal 0.904).
     Neither changes the reading.
   - The first version of this study called 0.871/0.897 "agreement". It is not.
2. **Where depth is informative, the instruments do agree: the walls.**
   - P(STRUCTURE | depth's steep planes) is 0.661 within 25 m, against 0.415 under the yaw-180 null and 0.470
     mirrored. That is a lift of 29 over STRUCTURE's 2.3% marginal.
   - Cohen's kappa on a common surface/vertical/unmodelled partition is 0.250 within 25 m, against 0.169 / 0.188
     under the nulls (0.430 vs 0.341 / 0.364 over all ranges).
   - The per-pano median kappa is only 0.097. The agreement is real but weak, and it is concentrated in the few
     pixels where depth models something other than floor.
3. **"Depth draws the ground through objects": supported, and the scrambled-geometry null is what shows it.**
   - 94.7% of segmenter-OBJECT pixels within 25 m sit on a depth floor plane. The nulls give 93.5% / 93.6%, so
     moving depth's geometry away from the objects does not change the rate: depth has no object-shaped holes.
   - Why the null and not the marginal: objects sit higher in the frame than the nadir-heavy marginal. On the
     complement scale, 5.3% of OBJECT pixels are off-floor against 1.7% of all pixels. The null holds elevation
     fixed; the marginal does not.
   - Above the near-nadir rows (y ≤ 0.75), the reading is 0.941 against 0.928 / 0.928.
   - Without Ego Vehicle / Car Mount the figure is 94.4% (94.0% above the near-nadir rows). Those two classes are
     6.2% of the depth-surface ∩ OBJECT pixels within 25 m.
   - The camera car can also hide under `Car`. In 9 of 336 panos, the largest on-floor Car component within 25 m
     has its centroid below y = 0.75 (`examples/objects_family_stats.csv`).
   - OBJECT is 5.5% of depth-surface pixels within 25 m (5.1% without the camera car; 7.1% solid-angle weighted).
     This is the term a segmenter supplies and depth cannot.
4. **(C) The segmenter class at the detection as a false-positive filter: NOT SUPPORTED.**
   - 2 of 42 verdict-False detections (0.048, Wilson 0.013–0.158) sit on a non-surface class, against 43 of 770
     verdict-True (0.056, 0.042–0.074). The registered margin is +20 pts; the False share is lower.
   - `underpowered` is **false**. The False interval is 14.5 pts wide, and its top end is 10 pts above the True
     share.
   - The reviewed false positives are footway things: 33 of 42 are Sidewalk/Curb/Curb Cut, and 7 are ROAD-group
     (Road 5, Catch Basin 1, Manhole 1).
5. **(C') 0.944 of verdict-True detections (727/770) are on WALK or ROAD.** Most exceptions are the peak pixel landing
   on a ramp's grass edge (Terrain) or an adjacent pole: one-pixel label noise.
6. **(B) GSV depth shows no ~0.15 m curb step, at most a few centimetres.**
   - Over 399 panos, the median of per-pano medians of `offset_local` for Vistas-`Sidewalk` pixels on a measured
     floor plane is **0.022 m (CI 0.014–0.030)**. That is **not consistent** with ~0.15 m.
   - The road-on-floor control reads 0.008 m. 34% of sidewalk offsets fall in [0.05, 0.30), against 23% for the
     road control.
   - Exploratory: requiring the local reference to be segmenter-ROAD gives 0.042 m (0.025–0.055).
   - `offset_local` is measured against the local reference plane, which may itself be sidewalk. The plane model
     carries at most a few centimetres; step 1 found ~0.03 m for real ramp planes.
7. **The trap: separating projection from resolution changed the answer** (registered: latitude band and seam).
   - The registered control arm fed the equirect at 2048 px wide, 5.7 px/deg, half the tiles' 11.4 px/deg.
   - A 4096 px direct arm, added on review at the tiles' angular resolution, fails on the **nadir fill**. Below
     −40° it labels 6–15% of what the tiled arm calls ROAD under the car as SKY (pooled), and agreement with the
     tiled arm falls to 0.82–0.91 there, where the 2048 px arm read 0.95–0.97.
   - **The failure varies by city.** At −80..−70° (interior) the SKY share is 0.000 in Bend, 0.110 in Paterson,
     0.097 in São Paulo and 0.247 in Gainesville (seam 0.318).
   - The crops (fig. 6b) are featureless fill, so "ROAD" there is the tiled arm's call, not ground truth. This
     is consistent with an effect that depends on the rig or the fill. It is not the general pole distortion
     the issue predicted, and Bend does not show it at all.
   - Near the horizon the 4096 arm agrees better than the 2048 arm: 0.95–0.96 against 0.91–0.92 at −10..0°.
   - The fine-class drop reported first (`Curb Cut` under True detections 0.271 tiled → 0.179 direct) was
     **resolution, not projection**. At 4096 px the direct arm reads 0.301. That comparison at detection pixels is
     post hoc and descriptive, not a registered trap reading. Pano-cluster bootstrap intervals match the Wilson
     ones to within 0.01 (`data/fig6_resolution_projection.csv`).
   - Seam vs interior is content-confounded. The seam band is straight behind the car: 95% ROAD at −20..−10°
     against 44% in the interior. It says nothing about how the model handles the wrap.
8. **Exploratory lead, outside every reading.** `Curb Cut` at the peak pixel covers 0.271 of True detections (Wilson
   0.241–0.304) against 0.071 of False (3/42, 0.025–0.190) in the tiled arm. In the 4096 direct arm the figures
   are 0.301 and 0.119. That is a candidate for a pre-registered follow-up (§7), not a result.

**Net for #47.**
- **The two instruments describe different things.** Depth says "a floor is modelled here" nearly everywhere, and
  says nothing about objects. The segmenter says what is there.
- **Their coarse agreement is mostly the base rate**, and where depth is informative (walls) it agrees with the
  segmenter well above chance.
- **For stage 3, the useful combination is the complement:** an object region from the segmenter, standing on a
  floor plane whose range depth supplies.
- **Neither instrument discriminates a ramp from the reviewed false positives** at a detection pixel.
- **GSV depth does not resolve a ~0.15 m curb step**; it carries at most a few centimetres.

## 1. Background and goals

Step 1 showed that GSV depth puts a near-horizontal plane under 99.4% of detections, and under verdict-False
detections as often as verdict-True ones. Depth locates the surface but does not discriminate a ramp from a
non-ramp.

The issue's premise is that depth models the **surface** and omits the **objects**, so that stage 3, the
"impedance region", needs a segmenter for the object and surface-class term. Step 2 measures three things:
- whether the two instruments agree, and where they disagree;
- whether the segmenter adds a discriminating term at a detection that depth lacked;
- how much the equirect-direct shortcut distorts the segmenter.

## 2. Research questions

- **RQ-A.** Over below-horizon pixels, how do depth plane classes and segmenter classes co-occur, beyond what their
  marginals imply? What share of depth's surface is actually an object?
- **RQ-B.** Where the segmenter sees sidewalk on a secondary depth plane, how high does depth put it above the road?
  Is it the ~0.15 m the issue quotes?
- **RQ-C.** At a detection, does the segmenter class separate verdict-False from verdict-True? This is step 1's
  rule (ii), with the segmenter in place of depth.
- **RQ-D.** How badly does feeding the equirect directly (the issue's "known trap") distort the answer?

## 3. Data

| city | judged GT panos | measured payload | stand-in / degenerate / implausible (excluded) |
|---|---:|---:|---|
| bend | 110 | 90 | 20 / 0 / 0 |
| paterson | 125 | 113 | 10 / 2 / 0 |
| gainesville | 125 | 112 | 12 / 0 / 1 |
| São Paulo | 125 | 101 | 24 / 0 / 0 |
| **total** | **485** | **416** | 69 |

- **Inputs.** `runs/_pooled/footway/sample.csv` lists the sample with a JPEG and payload sha256 per pano;
  `sample_counts.csv` holds the counts above.
- **GT join.** `eval_sites.judged_gt_panos`, with the same drift gate as every sibling study. No pano was skipped.
- **Pixels.** `../RampNet/benchmark/<city>/panos/`: 16384x8192 JPEGs, read-only.
- **Depth.** The runs' `depth/` payloads.
- **Detections.** The stored ones at ≥ 0.30. Verdicts are at the benchmark tier, 0.55.
- **Counts on these 416 panos:**
  - 770 verdict-True and **42 verdict-False**;
  - 47 unsure and 4 duplicate;
  - 331 non-unsure and 162 unsure missed marks;
  - 140 detections in the 0.30–0.55 band.

## 4. Methods

### 4.1 Segmenter

The segmenter is `facebook/mask2former-swin-large-mapillary-vistas-semantic` at revision
`4772b6bf101d91f2534c106dc524d906aeb3c68a` (Mapillary Vistas v1.2, 65 classes, including Curb Cut). It runs in
fp16 on the local RTX 3070 (torch 2.8.0+cu126, transformers 5.12.1).

- **No resize.** The processor's default resize to 384x384 is switched off (`do_resize=False`).
- **Semantic maps.** Computed the standard Mask2Former way: softmax(class) without the no-object column, times
  sigmoid(mask), at mask resolution; then bilinear upsampling to the target and argmax.
- **Load report.** It flags `swin.layernorm` as newly initialised. That layer feeds only the backbone's pooled
  output, which Mask2Former's feature maps do not use.
- **Provenance.** Every one of the 6,656 tile maps comes from a batch-2 forward. On review, the 404 maps an aborted
  batch-4 invocation had written were re-segmented at batch 2; the batch-4 versions had differed in 0.022% of
  pixels, an fp16 batching effect. The 32 smoke-run maps re-segment byte-identically. The notes are in
  `masks_manifest.json`.

### 4.2 Tiling, the stitch, and the control arms

- **Tiles.** Each pano is resized to 4096x2048 (LANCZOS) and rendered bilinearly to **16 perspective tiles**. Each
  tile is 90° FOV, 1024x1024, at 11.4 px/deg. Headings are every 45°, at pitch 0° and −35°.
- **Frame.** The geometry is pure image-frame: x runs from the left of the JPEG, longitude increases to the right,
  and heading 0 is the centre column.
- **Stitch.** Every tile whose frustum contains a grid pixel votes with its nearest tile pixel. A tie goes to the tile
  nearest in angle.
- **Coverage.** No tile sees the zenith above ~+43°, nor the nadir below ~−79°. The nadir gap is 11.4% of
  below-horizon pixels, all within ~0.5 m of the car. As the pre-registration stated, these pixels are labelled
  `NONE` and left out of every denominator.
- **Control arms** (never in a headline):
  - **direct 2048** (registered): the 2048x1024 equirect, 5.7 px/deg.
  - **direct 4096** (added on review): the 4096x2048 equirect at the tiles' angular resolution, as a single
    whole-image forward with no sliding window. It peaked at 5.7 GiB and ran at 0.72 img/s.

  Both are read out on the same grid and compared over the pixels the tiled arm sees.

![Three stacked panels for one Bend panorama: the equirect image with the 16 tile outlines drawn on it (blue for
pitch 0, orange for pitch -35); a coverage map showing how many tiles see each pixel, with the zenith above +43 deg and
the nadir below -79 deg in red; and the stitched label map in the collapsed classes.](figures/footway-depth/fig7_tiling.jpg)

*Figure 7. The tiling covers everything from +43° down to −79°, and the stitched map is what every later number is
read from.* (a) The first pano of `sample.csv` with the 16 tile footprints. (b) The number of tiles that see each
grid pixel; red = none (the `NONE` label). (c) The stitched, collapsed label map. The geometry is computed by the
same functions the tests pin (`tile_pixel_dirs`, `StitchGeometry`). Data: `examples/tiling.csv`,
`examples/tiling/`.

### 4.3 Class collapse

`COLLAPSE` in the script maps every Vistas name explicitly; an unmapped name refuses.

| group | Vistas classes |
|---|---|
| WALK | Sidewalk, Curb, Curb Cut, Crosswalk - Plain, Pedestrian Area |
| ROAD | Road, Lane Marking - Crosswalk, Lane Marking - General, Parking, Bike Lane, Service Lane; flush fixtures Manhole, Catch Basin, Pothole; Rail Track |
| OBJECT | every vehicle, person and rider, Bird, Ground Animal, Vegetation, Guard Rail, Barrier, all poles, signs, lights and street furniture, Car Mount, Ego Vehicle |
| STRUCTURE | Building, Wall, Fence, Bridge, Tunnel |
| SKY | Sky |
| OTHER | Terrain, Mountain, Sand, Snow, Water |

**Kappa partition** (added on review). Kappa needs one partition on both sides:
- depth {ground, floor, stand-in} ↔ segmenter {WALK, ROAD, OTHER} as `surface`;
- depth `non_horizontal` ↔ STRUCTURE as `vertical`;
- depth {no_plane, horizontal_nonfloor} ↔ {OBJECT, SKY} as `unmodelled`.

Under this partition an object standing on a depth floor plane counts *against* kappa, by design.

### 4.4 Depth side

- **Classes.** Step 1's classes come from `depth_at_detection.plane_class`, with `floor` split by `depth.is_standin`.
  The depth surface is ground ∪ floor ∪ stand-in floor.
- **Lookup.** `depth._plane_at`, in the image frame, with an exact ray. `offset_local` comes from
  `depth_at_detection.classify`, and the local reference plane is read where its column walk stopped
  (`reference_plane`; its stand-in test is `depth.is_standin`).
- **Grid.** There is one grid pixel per payload pixel below the horizon: 512x128 per pano.
- **Range.** Bins use the flat raycast at 2.6 m.
- **Null arms (A), added on review.** The same pixels and denominators, with the payload index array rotated 180° in
  azimuth (`yaw180`) or mirrored left-right (`mirror`).
- **Weightings (A), added on review.** Three columns: pooled pixels (the registered reading), solid angle
  (cos(latitude) per pixel row), and the per-pano median (each pano one vote).

### 4.5 The readings, as registered

- **(A)** A descriptive agreement matrix. Its headlines are P(WALK or ROAD | depth surface) and
  P(OBJECT | depth surface).
- **(B)** The median, across panos, of each pano's median `offset_local` over `Sidewalk` pixels.
  - Pixels: on a measured floor plane within 25 m, whose plane and local reference are not stand-ins.
  - Panos: a pano needs ≥ 20 such pixels.
  - Reading: "consistent with ~0.15 m" iff the median lies in [0.05, 0.30).
  - The cap of 1,500 pixels per pano and group (a seeded subsample) is set in the pre-scoring code (`d1ffbb6`),
    not in the comment.
- **(C)** `non_surface` = the tiled group at the detection pixel is not WALK or ROAD.
  - **SUPPORTED** iff the pooled False share exceeds the True share by ≥ 20 pts **and** the False Wilson lower bound
    exceeds the True upper bound.
  - `underpowered` iff the False interval is wider than 20 pts.
  - Tier 0.55.
- **(C')** Descriptive: the share of True detections on WALK or ROAD.
- **Trap.** Tiled vs direct agreement by latitude band and seam band.

## 5. Findings

### 5.1 RQ-A: agreement, against its base rate (`headline.csv`, `agreement.csv`, `pano_counts_le25m.csv`; figs. 1–3)

Pooled, within 25 m unless noted. NULL columns use pixel weighting.

| statistic | pixels | solid angle | per-pano median | NULL yaw 180 | NULL mirror | all ranges, pixels |
|---|---:|---:|---:|---:|---:|---:|
| P(WALK or ROAD \| depth surface) | 0.897 | 0.863 | 0.917 | 0.894 | 0.895 | 0.871 |
| marginal P(WALK or ROAD) | 0.884 | 0.845 | 0.904 | 0.884 | 0.884 | 0.826 |
| lift | 1.01 | 1.02 | 1.00 | 1.01 | 1.01 | 1.05 |
| P(depth surface \| WALK or ROAD) | 0.997 | 0.997 | 1.000 | 0.994 | 0.995 | 0.997 |
| marginal P(depth surface) | 0.983 | 0.976 | 0.996 | 0.983 | 0.983 | 0.946 |
| P(OBJECT \| depth surface) | 0.055 | 0.071 | 0.033 | 0.054 | 0.054 | 0.070 |
| ... without Ego Vehicle / Car Mount | 0.051 | | 0.032 | | | 0.067 |
| P(depth surface \| OBJECT) | 0.947 | 0.942 | 0.979 | 0.935 | 0.936 | 0.804 |
| ... without Ego Vehicle / Car Mount | 0.944 | | 0.979 | | | 0.797 |
| P(STRUCTURE \| depth wall) | **0.661** | 0.687 | 0.692 | **0.415** | **0.470** | 0.674 |
| marginal P(STRUCTURE) | 0.023 | 0.033 | 0.013 | 0.023 | 0.023 | 0.053 |
| P(WALK or ROAD \| depth wall) | 0.153 | 0.120 | 0.007 | 0.336 | 0.280 | 0.052 |
| Cohen's kappa | **0.250** | 0.263 | 0.097 | **0.169** | **0.188** | 0.430 (nulls 0.341 / 0.364) |

Per city, within 25 m (correct / yaw180 / mirror):

| city | P(WALK∪ROAD \| surf) [marginal] | P(surf \| OBJECT) | P(STRUCTURE \| wall) | kappa |
|---|---|---|---|---|
| bend | 0.906 / 0.905 / 0.905 [0.902] | 0.957 / 0.966 / 0.963 | 0.488 / 0.209 / 0.222 | 0.104 / 0.053 / 0.057 |
| paterson | 0.905 / 0.902 / 0.903 [0.892] | 0.955 / 0.941 / 0.946 | 0.674 / 0.403 / 0.486 | 0.225 / 0.146 / 0.170 |
| gainesville | 0.882 / 0.881 / 0.881 [0.878] | 0.966 / 0.960 / 0.955 | 0.553 / 0.174 / 0.216 | 0.117 / 0.055 / 0.065 |
| São Paulo | 0.896 / 0.889 / 0.891 [0.866] | 0.929 / 0.907 / 0.907 | 0.684 / 0.466 / 0.512 | 0.341 / 0.242 / 0.266 |

**Reading.**
- **The surface headline is a base rate.** Depth calls nearly every seen below-horizon pixel within 25 m a
  floor, and most of those pixels are road. Scrambling depth's geometry moves P(WALK or ROAD | depth surface) by
  0.003. Every weighting agrees.
- **The pooled pixels are nadir-heavy.** 71% of within-25 m pixels lie in the 0–5 m bin, mostly road under the car.
  That is why the solid-angle and per-pano columns are shown beside the pooled one.
- **The walls carry the signal.** Depth's steep planes are segmenter STRUCTURE 66% of the time, against 42–47% when
  the depth geometry is scrambled. Kappa sits 0.05–0.10 above the nulls in the pool and in every city. The
  agreement is real but weak, and it lives in the few non-floor pixels.
- **Depth does not exclude objects.** P(depth surface | OBJECT) matches its scrambled-geometry null: 0.947 against
  0.935 pooled, and 0.941 against 0.928 above the near-nadir rows. Per city it sits within ~0.02 of the null in
  either direction; Bend's 0.957 is just below its 0.966 / 0.963.
  - The null, not the floor's marginal, is the control. Objects sit higher in the frame than the nadir-heavy
    marginal: 5.3% of OBJECT pixels are off-floor against 1.7% of all pixels, so a lift against the marginal mixes
    elevation into the comparison.
  - Depth's floor therefore runs under objects as if they were not there. It "draws the ground through objects",
    the issue's premise. OBJECT is 5.5% of the modelled surface within 25 m (7.1%
  solid-angle weighted), rising with range from 2% at 0–5 m to 24% at 15–25 m (`data/derived_numbers.csv`). That share is what a
  segmenter adds.

![Grouped bars for four statistics within 25 m, pooled: P(WALK or ROAD given depth surface), P(STRUCTURE given
depth wall), P(depth surface given OBJECT) and Cohen's kappa. Each group shows the measured value in blue beside two
gray null bars, with a dashed marginal line and bootstrap whiskers. Only the wall statistic and kappa stand clear of
the nulls. A second row repeats the four statistics per city as dot-and-interval plots.](figures/footway-depth/fig1_base_rate.png)

*Figure 1. Surface "agreement" is a base rate: scrambling depth's geometry leaves it unchanged, and only the walls
and kappa carry signal.* The bars show the pooled pixel statistic within 25 m. Whiskers are 95% pano-bootstrap
intervals (1,000 resamples, seed 47; the pano is the unit, because pixels inside a pano are not independent).
Dashed lines are the marginal each conditional must be read against. Bottom row: per city. Data:
`data/fig1_base_rate.csv` (point estimate and interval per scope, arm and statistic), from
`runs/_pooled/footway/pano_counts_le25m.csv`.

![Two heatmaps side by side of depth plane class (rows) against segmenter group (columns), row-normalised, within
25 m. The left (measured) panel annotates each sizeable cell with its lift over the right panel, a 180-degree
rotated depth null. Rows under 10,000 pixels are greyed. Only the non_horizontal row's STRUCTURE cell rises
clearly, at x1.59.](figures/footway-depth/fig3_where_signal.png)

*Figure 3. The signal sits in the wall row; the floor rows barely move against the null.* These are row shares
over below-horizon pixels within 25 m, pooled. Left cells carry the lift over the 180°-rotated null, shown where
the share is ≥ 0.05 and the row has ≥ 10,000 pixels. Rows under 10,000 pixels are greyed: within 25 m,
`no_plane` is a single pixel and `horizontal_nonfloor` 916. The null shown is the 180° rotation. Data:
`data/fig3_where_signal.csv`.

![Contact sheet in two parts. Top: ten objects in the 5-15 m band (two parked cars, a truck, a bus, two people
walking behind a car, a cyclist, shrubs and two utility poles), each shown as photo, segmenter overlay and depth
overlay, all on a depth floor, ground or stand-in plane. Bottom: four counter-examples (two hedges and two cars)
whose OBJECT pixels sit on a depth wall plane instead.](figures/footway-depth/fig2_objects_on_floor.jpg)

*Figure 2. Examples of both outcomes. Most objects sit on a depth floor plane, and a minority sit on a wall
plane; the rate against its null is fig. 1, not this sheet.* The selection rule is fixed, with seed 47.
- **Band:** only objects in the 5–15 m flat-range band are eligible, which keeps the sheet clear of the camera car
  and the nadir fill.
- **Order:** candidates are taken in a seeded order of the top 10 by component size (on-floor rows, per family:
  car, bus/truck, person, vegetation, pole) or the top 20 (counter-example rows, OBJECT component NOT on a floor
  plane), then in rank order.
- **What a pick must satisfy:**
  - the crop is less than 70% OBJECT, so the reader can see what the object is;
  - the pano is not already on the sheet;
  - for the person family, the person is a small standing or moving pedestrian (< 40% of the crop height,
    ≥ 8 m away).
- **Privacy skips:** 3 picks were skipped by the stated privacy rule (a seated individual centred in the crop;
  listed in `examples/skipped.csv`, a skip list that applies to every example sheet), and the next in rank was
  used.
- **Labels:** each crop's label is the segmenter's own majority class.
- **Palette:** depth overlays use their own palette (violet / red / lavender / charcoal), so they never share a
  colour with the segmenter's groups.

Data: `examples/objects.csv`, `examples/objects_family_stats.csv`.

The matrix rows (`agreement.csv`, all ranges):
- **The dominant ground plane** is 87% ROAD and 7% WALK.
- **Secondary floor planes** are 66% ROAD, 12% WALK and 12% OBJECT. WALK makes up a larger share of secondary
  floors than of the dominant plane (11.8% vs 6.9% of seen pixels). In absolute terms, though, the sidewalk is
  split about evenly: 0.97 M WALK pixels sit on the dominant plane and 1.03 M on secondary floors (0.91 M
  measured, 0.12 M stand-in).
- **Stand-in floors** are the most object-laden surface class (23% OBJECT).
- **Steep planes** are 67% STRUCTURE.

Where depth reports a wall and the segmenter a sidewalk (15% of wall pixels within 25 m, about half the nulls'
28–34%), the fixed-rule gallery (fig. 8, row 3) shows mostly **steeply sloped pavement**: São Paulo's sloped
sidewalks, driveway aprons, a curb face.

### 5.2 RQ-B: curb height (`curb_height.csv`; per-pano medians in `runs/_pooled/footway/curb_pano_medians.csv`; fig. 4)

| reading | panos | median of pano medians [95% CI] | pixel p10 / median / p90 | share in [0.05, 0.30) |
|---|---:|---|---|---:|
| **Sidewalk, measured planes (registered)** | 399 | **0.022 [0.014, 0.030]** | −0.10 / 0.02 / 0.19 | 0.339 |
| Sidewalk, with stand-ins | 401 | 0.033 [0.025, 0.039] | −0.09 / 0.03 / 0.22 | 0.366 |
| Sidewalk, reference = segmenter ROAD (*exploratory*) | 366 | 0.042 [0.025, 0.055] | −0.13 / 0.05 / 0.26 | 0.428 |
| all WALK, measured | 407 | 0.021 [0.013, 0.028] | −0.10 / 0.02 / 0.18 | 0.330 |
| ROAD on a floor plane (control) | 413 | 0.008 [0.002, 0.012] | −0.09 / 0.01 / 0.11 | 0.234 |

Positive means above the road. Per city, the registered Sidewalk reading is bend 0.031, paterson 0.024, gainesville
0.020 and São Paulo 0.009. **Reading: not consistent with ~0.15 m.**

![Step histograms of per-pano median offset above the local reference plane for three readings: Sidewalk
(registered, blue), Sidewalk with a ROAD-labelled reference (exploratory, violet dashed) and the Road control
(orange). Their medians are 0.022, 0.042 and 0.008 m, marked by lines with CI bands. The claimed band from 0.05 to
0.30 m is shaded, with a dotted line at 0.15 m.](figures/footway-depth/fig4_curb_height.png)

*Figure 4. GSV depth shows no ~0.15 m curb step, at most a few centimetres: 0.022 m against the local
reference, and 0.042 m road-referenced (exploratory).* These are histograms of per-pano medians, with stand-ins
excluded, within 25 m. Lines and shading mark each median of pano medians and its 95% order-statistic CI. The
x-axis is the height above the local reference plane, the next floor plane down the column. Data:
`runs/_pooled/footway/curb_pano_medians.csv`, `data/curb_height.csv`, `data/fig4_curb_height.csv`. (Fig. 4b, a
strip of single-pixel offsets, was dropped on the replication audit. Its crops did not show the curb edge, and
single pixels scatter by ±0.2 m.)

What the median does and does not cover:
- **It covers only sidewalk on a secondary plane.** Sidewalk pixels on the dominant plane, shared with the road,
  have no offset to measure and are excluded. Including them as zeros would lower the figure, not raise it. That
  share is 50.5% pixel-pooled, but the per-pano median is 0.27 over 410 panos, so it is concentrated in some panos.
- **It can understate a real step.** The column walk's reference is "the first different floor plane below", which
  can be another piece of sidewalk. The exploratory road-referenced reading addresses this and moves the median only
  to 0.042 m.
- **Tilt can push it either way.** `offset_local` carries the reference plane's tilt, extrapolated to the hit (step
  1 §5.4).

Sidewalk offsets fall in the claimed band more often than the road control's (34% vs 23% of pixel offsets),
which suggests a faint step. No reading puts the typical sidewalk plane near 0.15 m: there is no ~0.15 m step,
at most a few centimetres.

### 5.3 RQ-C: at the detection (`runs/_pooled/footway/detections.csv`, `detection_classes.csv`, `verdict.json`, fig. 5)

| group | n | non-surface share [Wilson 95%] | WALK | ROAD |
|---|---:|---|---:|---:|
| verdict True | 770 | 0.056 [0.042, 0.074] | 0.870 | 0.074 |
| verdict False | 42 | 0.048 [0.013, 0.158] | 0.786 | 0.167 |
| missed (non-unsure) | 331 | 0.030 [0.016, 0.055] | 0.825 | 0.145 |
| unsure missed | 162 | 0.056 [0.029, 0.102] | 0.765 | 0.179 |

**(C) NOT SUPPORTED.**
- The gap is −0.8 pts against a registered +20.
- `underpowered` is false.
- Per city, the False non-surface counts are 1/7, 0/5, 0/9 and 1/21.

**(C') 0.944 of True detections are on WALK or ROAD** (bend 0.955, paterson 0.930, gainesville 0.924, São Paulo
0.975).

![Three panels. Left: the non-surface share with Wilson intervals for verdict True (0.056), verdict False (0.048)
and missed marks (0.030), with a red dashed line marking the registered bar of True + 20 points, far to the right of
every point. Middle: stacked bars of segmenter groups, mostly WALK, for the three groups. Right, labelled
exploratory: Curb Cut share, True 0.271, False 0.071, missed 0.254. Each point carries two intervals, Wilson
and a pano-cluster bootstrap, and a dotted line marks the rule's second condition.](figures/footway-depth/fig5_fp_rule.png)

*Figure 5. The segmenter class at the peak pixel is no false-positive filter: the 42 false positives are footway
too.*
- **Left:** the registered reading (C), with both rule conditions drawn: False ≥ True + 20 pts (dashed), and
  False's Wilson lower bound above True's upper bound (dotted).
- **Middle:** the segmenter group at the pixel.
- **Right:** the post hoc Curb Cut share, labelled exploratory.
- **Intervals:** filled markers carry Wilson 95%, the registered interval. Hollow markers carry a 95% pano-cluster
  bootstrap (1,000 resamples, seed 47), because detections on one pano are not independent (770 True detections
  sit on ≤ 416 panos). The two agree closely.

Data:
`data/fig5_fp_rule.csv`, from `runs/_pooled/footway/detections.csv`.

**What the false positives are.** They sit on the footway: Sidewalk 25, Curb 5, Curb Cut 3. Seven more are
ROAD-group (Road 5, Catch Basin 1, Manhole 1). They are driveway cuts, ramp-like curb transitions and paving
changes. A surface-class filter cannot remove them.

**Depth at the same pixels agrees with step 1.**
- True: floor 500, ground 143, stand-in floor 122, steep 4, overhang 1.
- False: floor 26, stand-in 10, ground 5, steep 1.

### 5.4 RQ-D: the trap (`trap.csv`, `trap_pano.csv`; figs. 6, 6b)

Agreement with the tiled arm is on the collapsed group, over the pixels the tiled arm sees. Shares also exclude
`NONE`, which was corrected on review.

| latitude band | interior, 2048 | interior, 4096 | seam, 2048 | seam, 4096 |
|---|---:|---:|---:|---:|
| −80..−70° | 0.957 | **0.856** | 0.948 | **0.819** |
| −60..−50° | 0.971 | 0.890 | 0.956 | 0.854 |
| −40..−30° | 0.962 | 0.945 | 0.967 | 0.919 |
| −20..−10° | 0.951 | 0.970 | 0.986 | 0.989 |
| −10..0° | 0.919 | 0.959 | 0.907 | 0.953 |
| 0..10° | 0.962 | 0.983 | 0.944 | 0.975 |
| +20..+40° | 0.99 | 1.00 | 0.99 | 1.00 |

Per city, −80..−70° (`trap.csv`; the 4096 px arm's SKY share among pixels the tiled arm sees):

| city | interior SKY share | seam SKY share |
|---|---:|---:|
| Bend | 0.000 | 0.000 |
| Paterson | 0.110 | 0.127 |
| São Paulo | 0.097 | 0.108 |
| Gainesville | 0.247 | 0.318 |

![Left: bars of the Curb Cut share under verdict-True detections and under missed marks for the 2048 px direct arm
(0.18, 0.13), the tiled arm (0.27, 0.25) and the 4096 px direct arm (0.30, 0.27), with Wilson intervals. Right:
agreement with the tiled arm by latitude band for both direct arms, interior solid and seam dashed, with bootstrap
bands. The 4096 px arm falls to 0.86 (interior) and 0.82 (seam) at the nadir, while the 2048 px arm stays near
0.95. A bottom row of per-city panels shows the drop is absent in Bend and largest in Gainesville.](figures/footway-depth/fig6_resolution_projection.png)

*Figure 6. Resolution, not projection, caused the Curb Cut loss. At matched resolution the direct arm instead
fails on the nadir fill, and the failure varies by city (absent in Bend).*
- **Top left (post hoc):** Curb Cut at the detection, with Wilson and pano-cluster bootstrap intervals.
- **Top right (registered):** agreement with the tiled arm by latitude band, with 95% pano-bootstrap bands
  (1,000 resamples, seed 47).
- **Bottom:** the interior bands per city.

Data:
`data/fig6_resolution_projection.csv`, from `runs/_pooled/footway/trap_pano.csv` and `detections.csv`.

![Six rows. Each starts with a wide context strip, 180 degrees across from the horizon to the nadir with a red
box at the crop, followed by the featureless nadir crop, the tiled overlay in ROAD orange, and the 4096 px overlay
mostly in SKY pink, labelled 69% to 97%. Five of the six are from Gainesville or São Paulo.](figures/footway-depth/fig6b_sky_examples.jpg)

*Figure 6b. At matched resolution, the 4096 px equirect paints the nadir fill under the car as SKY. The
pixels are featureless, so "ROAD" is the tiled arm's call, not ground truth.*
- **Selection rule** (fixed, seed 47): 6 panos drawn from the 20 with the most pixels below −40° that the tiled
  arm calls ROAD and the 4096 px arm SKY. Each crop is centred on the largest such component. The privacy skip list skipped 0
  picks on this sheet.
- **Context strip:** the left panel of each row places the crop within the lower half of the panorama.

Data: `examples/sky.csv`.

**Projection vs resolution.**
- **At matched angular resolution the direct arm fails on the nadir fill.** Pooled, the 4096 px arm labels 12% of
  the tiled arm's ROAD at −80..−70° as **SKY** (15% in the seam band), and still 6–8% at −50..−40°
  (`trap.csv`, `direct4096_share_SKY`).
  - The failure varies by city: from 0.000 in Bend to 0.247 in Gainesville (table above).
  - The crops are featureless fill, so this reads as a rig- or fill-dependent effect, not the generic pole
    distortion. We cannot tell which from this sample.
  - That the largest single disagreement pair is tiled `Road` → direct `Sky` comes from an uncommitted check over
    every 4th pano's −70..−51° rows. It is not in a committed CSV.
- **The 2048 arm hides this.** It reads the nadir as road (0.95). Why is speculation: perhaps at half the
  resolution the stretched nadir is a featureless grey band either way.
- **Near the horizon, resolution matters more than projection.** The 4096 arm agrees better than the 2048 arm there.
- **The registered seam-vs-interior split cannot isolate the wrap.** The seam band is straight behind the car: 95%
  ROAD at −20..−10°, against 44% in the interior. The first version's "the model handles the wrap" claim is
  withdrawn.

**At the detection pixels** (post hoc, descriptive; not a registered trap reading):

| arm | agrees with tiled on group / fine class (True + False + missed) | `Curb Cut` under True | under False | under missed |
|---|---|---:|---:|---:|
| tiled | — | 0.271 | 0.071 | 0.254 |
| direct 2048 | 0.896 / 0.684 | 0.179 | 0.024 | 0.130 |
| direct 4096 | 0.958 / 0.843 | 0.301 | 0.119 | 0.269 |

The fine-class loss reported first was resolution, not projection. Fed at full resolution, the equirect keeps Curb
Cut at the detections, 98% of which lie between −2° and −36° (full range −49° to +4°), where the projection distortion is small. The projection
cost shows up at the nadir instead.

### 5.5 The disagreement gallery (`examples/disagreement.csv`, fig. 8)

**Selection rule** (fixed, seed 47):
- For each of three pixel categories, panos are ranked by the size of the category's largest 8-connected component
  within 25 m, above the near-nadir rows (y ≤ 0.75). That exclusion was added on the replication audit; row 1 was
  previously mostly near-camera crops. Six are drawn from the top 20, each marked at the component pixel nearest its centroid.
- The fourth row is six draws from all 45 detections at ≥ 0.55 that the segmenter calls non-surface.

![A 4 by 6 contact sheet of photo crops with a red ring and in-image labels naming the depth class and segmenter
class. Row 1: objects on a depth floor (parked cars, a palm, a trailer, a trash can). Row 2: fences and walls on a
depth floor. Row 3: sloped sidewalks that depth calls a wall. Row 4: verdict-True detections whose peak pixel lands on
grass or a pole.](figures/footway-depth/fig8_disagreement_gallery.jpg)

*Figure 8. Where depth and the segmenter disagree: objects on the floor, see-through fences, sloped pavement, and
peak pixels just off the ramp.* The selection rule is stated above (fixed, seed 47). The privacy skip list skipped 0
picks on this sheet. Data:
`examples/disagreement.csv`.

1. **Depth surface, segmenter OBJECT.** Parked cars, a palm, a trailer and a trash can. Depth draws its plane
   through each. Near-nadir rows are now excluded; before that, the camera car's own nadir patch took 2 of the 6
   picks.
2. **Depth surface, segmenter STRUCTURE.** See-through fences, behind which depth models the ground, and walls at
   the ground-contact line.
3. **Depth wall, segmenter WALK.** Steeply sloped sidewalks and driveway aprons, a curb face, a store threshold.
   Here depth carries slope information that the segmenter's class lacks.
4. **True detections the segmenter calls non-surface.** The peak lands on the grass edge or a pole beside the ramp.

## 6. Discussion

**Stages 2 and 3 of #47.**
- **The two instruments are complementary, not redundant.** Depth's floor is close to unconditional below the
  horizon and ignores objects. The segmenter's surface classes add no information about *whether* there is a floor,
  but they add the object term: where something stands on the floor.
- **Stage 3 is well-posed as that complement.** Take a segmenter-OBJECT region and read the depth floor plane
  under it for a metric contact-line range.
- **Where depth models something other than floor** (walls, steep pavement), it agrees with the segmenter well above
  chance. It also carries a slope signal (gallery row 3) that may matter for stage 3's severity question.
- **The curb step is the gap.** A height under ~0.3 m needs another instrument: monocular depth, multi-view
  triangulation, or the box annotations.
- **Tile, or at least run at full resolution and handle the nadir.** At 2048 px a direct equirect loses fine classes;
  at 4096 px it hallucinates sky under the car.

**What it does not do.**
- A class at a detection's single peak pixel is no false-positive filter, from depth (step 1) or from the
  segmenter (here).
- Discrimination has to come from the detector's appearance model, from multi-view consistency (#27), or from a
  region-level rule.

**For the heuristic cropper** (ProjectSidewalk/sidewalk-panorama-tools#32): the segmenter's WALK region around a
detection is a plausible context-extent input, and the depth plane under it gives the range. This is a direction;
nothing here measures crop quality.

**Limits.**
- 42 verdict-False detections.
- One model.
- The pixel test is at the heatmap peak, which RampNet's upsampled heatmap quantises to an 8-cell grid
  ([RampNet#221](https://github.com/ProjectSidewalk/RampNet/issues/221)).
- The kappa partition is a choice, stated in §4.3.
- The benchmark panos are corner-heavy by construction.

## 7. Recommendations

1. **Never quote a conditional over below-horizon pixels without its marginal and a scrambled-geometry null.**
2. **Tile.** If a direct equirect is used, run it at full angular resolution and mask or tile the nadir. At 2048 px
   the fine classes go; at 4096 px the road under the car turns to sky.
3. **Do not build a surface-class false-positive filter** from depth or the segmenter at the peak pixel.
4. **Next pre-registered step (not done here): a region-level `Curb Cut` score** around the detection.
   - Choose the window on a held-out city.
   - Use it as a ranking and mining signal (ties to RampNet#158). 25% of non-unsure missed marks already sit on
     tiled Curb Cut pixels.
5. **For stage 3's metric term,** take ground-contact range from depth. Do not rely on depth for heights under
   ~0.3 m.

## 8. Replication

**Obtaining inputs.** Three inputs are not in any git repository.
- **Benchmark pixels:** `../RampNet/benchmark/<city>/panos/` (gitignored in RampNet). These are the
  native-resolution benchmark JPEGs, produced by this repo's `scripts/export_benchmark.py`.
- **Depth payloads:** `runs/<city>/depth/*.json.gz` (untracked harvests), produced by
  `scripts/harvest_depth.py`.
- **Per-city runs:** `runs/<city>/results.jsonl`, written by `main.py`.

All three are kept with the lab's run archive on makelab2 (`/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/`)
and are available on request. The arbiter is `runs/_pooled/footway/sample.csv`, which records each sampled pano's
JPEG and payload sha256; `inputs.json` records each run's `results.jsonl` sha256. A fresh harvest may differ from
the archived one if Google has revised a payload since. Compare its hash with `sample.csv` before trusting a
re-run.

**From raw inputs to every table and figure, in order.** `<runs>` is the run tree holding the four GSV cities'
`results.jsonl` and `depth/` payloads; on the original machine that is `D:/Git/sidewalk-auto-labeler/runs`.

| # | command | needs | output | runtime here |
|---|---|---|---|---|
| 1 | `python scripts/footway_segmentation.py sample --run-root <runs> --benchmark-root ../RampNet/benchmark` | CPU | `runs/_pooled/footway/sample.csv`, `sample_counts.csv` (committed) | 38 s |
| 2 | `python scripts/footway_segmentation.py tiles --benchmark-root ../RampNet/benchmark --workers 8` | CPU | `work/tiles/` (6,656 JPEGs), `work/direct/` (416), untracked | 14 min |
| 3 | `python scripts/footway_segmentation.py tiles --benchmark-root ../RampNet/benchmark --workers 8 --direct-width 4096` | CPU | `work/direct4096/`, untracked | 2.4 min |
| 4 | `python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/tiles --out runs/_pooled/footway/work/tile_labels --fp16 --batch-size 2` | **GPU**; network once (model download) | `work/tile_labels/` + `segment_manifest.json` | 19 min (5.9 img/s) |
| 5 | `python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/direct --out runs/_pooled/footway/work/direct_labels --target 1024x512 --fp16 --batch-size 1` | **GPU** | `work/direct_labels/` | 2.3 min (3.0 img/s) |
| 6 | `python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/direct4096 --out runs/_pooled/footway/work/direct4096_labels --target 1024x512 --fp16 --batch-size 1` | **GPU** (5.7 GiB) | `work/direct4096_labels/` | 9.6 min (0.72 img/s) |
| 7 | `python scripts/footway_segmentation.py stitch --run-root <runs> --benchmark-root ../RampNet/benchmark` | CPU | `work/stitched/` | 2.5 min |
| 8 | `python scripts/footway_segmentation.py compare --run-root <runs> --benchmark-root ../RampNet/benchmark` (runs `verdict` too) | CPU | every CSV / JSON / report under `runs/<city>/footway/` and `runs/_pooled/footway/` (committed) | 2.2 min |
| 9 | `python scripts/footway_segmentation.py examples --run-root <runs> --benchmark-root ../RampNet/benchmark` | CPU; untracked `work/` + JPEGs | `docs/figures/footway-depth/examples/` (panels + CSVs, 1.44 MB, committed) | 2.0 min |
| 10 | `python scripts/footway_segmentation.py figures` | **committed files only**; no GPU, no network, no `work/` | every figure, `data/*.csv`, `data/numbers.csv`, `data/derived_numbers.csv`, `data/figures_manifest.json`; exits 1 on any number mismatch | 30 s |
| 11 | `python scripts/footway_segmentation.py check-numbers` | committed files only | `data/numbers.csv`, `data/derived_numbers.csv`; exits 1 on any mismatch | 3 s |

- **Byte-reproducible figures, under the listed versions.** Step 10 reads only committed files and drops every
  timestamp: PNG `Software`, SVG `Date`/`Creator` (SVG written as LF bytes), a fixed `svg.hashsalt`, and seeded
  bootstraps. Two consecutive runs gave identical sha256 for all 14 outputs; they are recorded in
  `data/figures_manifest.json` and match the committed blobs.
  - JPEG sheets depend on Pillow/libjpeg, and PNG/SVG on matplotlib/FreeType. With other versions, expect
    pixel-level differences rather than identical bytes.
- **Committed vs regenerated.** Committed:
  - `runs/_pooled/footway/`: sample, per-pano and per-detection tables, reports, `verdict.json`, `inputs.json`
    and `masks_manifest.json`;
  - `runs/<city>/footway/`;
  - `docs/figures/footway-depth/`: figures, `data/` and `examples/`.

  Regenerated and untracked: `runs/_pooled/footway/work/` (tiles and label maps).
  - Committed for them: every stitched and direct label map's sha256 (`masks_manifest.json`), the per-tile
    sha256 of all 6,656 tile label maps (`runs/_pooled/footway/tile_label_hashes.txt`), and the model's
    `id2label` (`masks_manifest.json`).
  - `stitch`, `compare` and `examples` still read the untracked `work/tile_labels/segment_manifest.json`, which
    `segment` writes.
- **Steps needing neither GPU nor network:** 1–3 and 7–10. Steps 4–6 need a CUDA GPU, and network access the
  first time, to fetch the pinned model.
- **Seeds:** 47 everywhere. That covers the curb pixel subsample (`random.Random('47:<pano_id>')`), every example
  rule (`random.Random('47:<set>')`, or `random.Random(47)` for the disagreement gallery), and the bootstraps
  (`numpy.random.default_rng(47)`, 1,000 resamples).
- **Model:** `facebook/mask2former-swin-large-mapillary-vistas-semantic` at revision
  `4772b6bf101d91f2534c106dc524d906aeb3c68a`, fp16, processor resize off. The Vistas id order is pinned in the
  script (`VISTAS_V12_ORDER`), and `examples` asserts it.
- **Environment, read from the env that ran it:**
  - Python 3.12.13, torch 2.8.0+cu126, transformers 5.12.1, Pillow 12.3.0, numpy 2.5.1, scipy 1.18.0,
    matplotlib 3.11.2.
  - GPU: NVIDIA GeForce RTX 3070 (8 GB, driver 591.86), Windows 11.
  - torch and transformers are needed only by `segment` and are deliberately not in requirements.txt.
- **Provenance notes.** All 6,656 tile maps come from batch-2 forwards (§4.1). The notes are carried in
  `masks_manifest.json`.

**Inputs (sha256).** `runs/_pooled/footway/inputs.json` records the same hashes, written by `compare`. The per-pano
JPEG and depth-payload hashes are in `sample.csv`.

| input | sha256 |
|---|---|
| bend `results.jsonl` | `1307faa8041acbbf4cba78fd53979e2215511b8371c018f427c356f6b0e26153` |
| paterson `results.jsonl` | `651226f9f1e66d60fc8504423e7400730aad649e20a4a448d25bd4e5c620a2a8` |
| gainesville `results.jsonl` | `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8` |
| sao_paulo `results.jsonl` | `54348f367dfa3ecb9cde587d106d493d729fb5e5744687e3f38d1c9f1549bf7f` |
| bend `depth/index.csv` (recorded; not read: payloads are read per pano) | `a4ae7d3743ca11b8e3ccce652009dcc57309dbd410ea8c90e57b4c3d044b448d` |
| paterson `depth/index.csv` | `06e878e2d77e60dbf2652d8e7b31ca8763d0aa49876d02c9e67338a2e90ebfa1` |
| gainesville `depth/index.csv` | `02ad2a5d87e2d45a2fb5eb6e3b4215546f07a8425ea7032a49e9cf8b2f67c527` |
| sao_paulo `depth/index.csv` | `46315eb54fbb14c17669be6a08f6c3ec1d7560f8c7d8864a8c600a147438aa09` |
| RampNet bend `verdicts.json` / `records.jsonl` | `9d9835db27f66c1904fcfd4fc87b06f26638a3c033ef10936387db40feb84b05` / `c17c2503b22cf78f8c38ebefa445f2709845550bacd84dff75fa5e9e1616b29a` |
| RampNet paterson | `5e43c54389eb9ea1d8daf0a075e562e27eb6b484dfcb3397cb6f438619365dad` / `363dba08d49d60ce9d9322115c89b27f450b35cd74598f127b261ab321979096` |
| RampNet gainesville | `776ddb2ca1a5a72fb9b27965ae8e5727d6fecc8bf4cc56b2ac94543c6cfad1e4` / `f59e73db5c95017b30d8687eb4e14bccfc10f516bf98fbcb13e502185607c8fc` |
| RampNet sao_paulo | `dbe32bcc8f8d87b3caedcc54597b6547720d98202abe149fb707dd560aa0db4c` / `c483538442d84a6e29b765d9e897253e833839f240d36af84dc82c69ed1c6b90` |
| RampNet commit | `6c252bbac30343da7fcebb57e6310ff82f54d708` |
| `runs/_pooled/footway/masks_manifest.json` | `323306cc07c2c5c06a2d22575a09707557233927966c2574af1262dca7fc801c` |
| `runs/_pooled/footway/sample.csv` | `9ea7f0e02a1c5e65392af34e795a723e6e06e210820ccb5a6c19fcf64701dd12` |

- **Guards.**
  - `compare` refuses if any sampled pano no longer passes the GT join.
  - `verdict` and `figures` print which directory they read.
  - `figures` and `check-numbers` exit 1 if any quoted number no longer matches its committed file.

## 9. Corrections on review (2026-09-30)

A deep review of #124 verified the pipeline, frames, decoding, GT join, pre-registration order and the (B)/(C)
readings, which reproduce byte for byte. It found the following, and all of it is corrected above.

1. **The (A) headlines were base rates presented as agreement.** The null arm, lifts, kappa, and the solid-angle and
   per-pano columns were added. The honest reading is §0.1–0.3. (A) was registered as descriptive, so this corrects
   an interpretation, not a rule.
2. **The trap conclusion was confounded.**
   - The registered direct arm ran at half the tiles' angular resolution. The 4096 px arm moves the effect from the
     Curb Cut class to the nadir.
   - The detection-pixel fine-class comparison is post hoc.
   - Seam vs interior is content-confounded.
   - The nadir shares divided by a total that included `NONE`.
   - "0.92–0.99 in every band" was false: the seam reads 0.907 at −10..0°.
3. **(B) wording.**
   - The dominant-plane exclusion does not make the median understate; including those pixels as zeros would lower
     it.
   - The 50.5% share was pixel-pooled; the per-pano median is 0.27.
   - "Secondary floors are where the sidewalk lives" overstated a share difference.
4. **Provenance.**
   - The pre-registration did state the unseen zenith.
   - The denominator rule was already in `d1ffbb6`.
   - The pixel cap is in the pre-scoring code only.
   - The module docstring said −75°.
5. **Nits.**
   - The ROAD group of the False detections is 7, not 5.
   - Ego Vehicle / Car Mount are now reported both ways.
   - `standin_ref` now uses `depth.is_standin` on the reference plane.
   - `figures` takes `--fig-dir`.
   - The large tables are tracked once.
   - Outputs are written with LF line endings.
   - The batch-4 tile maps were redone at batch 2.

### Replication audit (2026-10-01)

An independent replication audit of 026f7e7 reproduced every step byte for byte from a clean archive. It asked
for the following, all applied in this revision:
- **Inputs:** the "obtaining inputs" paragraph.
- **The nadir-fill result:** reported per city, with a context strip per example.
- **Fig. 2:** rebuilt so it no longer selects only one outcome. It now uses the 5–15 m band, drops the camera-car
  family, adds counter-examples, and gives the depth overlay its own palette.
- **Curb wording:** "no ~0.15 m step, at most a few cm".
- **Objects claim:** grounded in the scrambled-geometry null rather than a lift against the marginal, plus a
  near-nadir-excluded reading.
- **Number checker:** keyed by file, row and column, with `quoted_in`, and failing non-zero on a mismatch.
- **Hashes:** the per-tile label-map hashes are committed.
- **Intervals:** pano-cluster bootstrap intervals beside Wilson.
- **Links:** commit-pinned image links.
- **Figure polish:** darker null and NONE colours, greyed thin rows, fig. 4b dropped.

## 10. Where each number lives

Every number quoted in this document, the PR body, the CLAUDE.md block and the #47 comments, with the committed file, row and column it comes from, and where it is quoted. `check-numbers` (and `figures`) regenerate this list as `data/numbers.csv` and look each value up by key (file | row | column), not by value. Each value is compared at the quoted rounding, and any mismatch exits non-zero. Last run: 236 numbers, 0 mismatches. The list is curated by hand.

Files: `headline.csv`, `curb_height.csv`, `detection_classes.csv`, `trap.csv`, `verdict.json` and `sample_counts.csv` are in `runs/_pooled/footway/`, with copies in `docs/figures/footway-depth/data/`. `derived_numbers.csv` is in `data/`; each of its rows carries its formula. Percentages in the text are the fractions below.

| quoted | file | row | column | value | quoted in |
|---:|---|---|---|---:|---|
| 0.897 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad_given_surface` | 0.896958 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.894 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_walkroad_given_surface` | 0.893784 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.895 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=mirror | `p_walkroad_given_surface` | 0.89475 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.884 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad` | 0.884216 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.983 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_surface` | 0.982868 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 1.01 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `lift_walkroad_given_surface` | 1.01441 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.997 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_walkroad` | 0.997031 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.863 | `headline.csv` | city=pooled, range=le25m, weighting=solid_angle, depth_arm=correct | `p_walkroad_given_surface` | 0.862858 | doc §0.1-0.3, §5.1 table; #47 comment 5923015034 |
| 0.845 | `headline.csv` | city=pooled, range=le25m, weighting=solid_angle, depth_arm=correct | `p_walkroad` | 0.845257 | doc §0.1-0.3, §5.1 table; #47 comment 5923015034 |
| 0.917 | `headline.csv` | city=pooled, range=le25m, weighting=per_pano_median, depth_arm=correct | `p_walkroad_given_surface` | 0.917351 | doc §0.1-0.3, §5.1 table; #47 comment 5923015034 |
| 0.904 | `headline.csv` | city=pooled, range=le25m, weighting=per_pano_median, depth_arm=correct | `p_walkroad` | 0.903899 | doc §0.1-0.3, §5.1 table; #47 comment 5923015034 |
| 0.71 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `share_0_5m_of_le25m_px` | 0.713433 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.661 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_structure_given_wall` | 0.661394 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.415 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_structure_given_wall` | 0.415146 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.470 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=mirror | `p_structure_given_wall` | 0.469654 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.023 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_structure` | 0.0230326 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 29 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `lift_structure_given_wall` | 28.7155 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.250 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `kappa` | 0.249886 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.169 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=yaw180 | `kappa` | 0.169121 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.188 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=mirror | `kappa` | 0.18828 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.097 | `headline.csv` | city=pooled, range=le25m, weighting=per_pano_median, depth_arm=correct | `kappa` | 0.0972365 | doc §0.1-0.3, §5.1 table; #47 comment 5923015034 |
| 0.430 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=correct | `kappa` | 0.429845 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.341 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=yaw180 | `kappa` | 0.341014 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.364 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=mirror | `kappa` | 0.363594 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.947 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_object` | 0.947364 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.935 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_surface_given_object` | 0.935343 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.936 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=mirror | `p_surface_given_object` | 0.935681 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.944 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_object_excl_ego` | 0.944151 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.062 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `ego_share_of_surface_object` | 0.0619377 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.055 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_object_given_surface` | 0.0546315 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.051 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_object_given_surface_excl_ego` | 0.0512478 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.071 | `headline.csv` | city=pooled, range=le25m, weighting=solid_angle, depth_arm=correct | `p_object_given_surface` | 0.0711377 | doc §0.1-0.3, §5.1 table; #47 comment 5923015034 |
| 0.153 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad_given_wall` | 0.153066 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.336 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_walkroad_given_wall` | 0.335974 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.280 | `headline.csv` | city=pooled, range=le25m, weighting=pixels, depth_arm=mirror | `p_walkroad_given_wall` | 0.279671 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.871 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=correct | `p_walkroad_given_surface` | 0.870654 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.826 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=correct | `p_walkroad` | 0.826335 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.070 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=correct | `p_object_given_surface` | 0.070157 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.804 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=correct | `p_surface_given_object` | 0.804358 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.674 | `headline.csv` | city=pooled, range=all, weighting=pixels, depth_arm=correct | `p_structure_given_wall` | 0.674046 | doc §5.1 table; #47 comments 5921870275 (findings: 0.871, 0.070) and 5923015034 (kappa) |
| 0.022 | `curb_height.csv` | city=pooled, group=sidewalk, reading=measured | `median_of_pano_medians` | 0.0215624 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.014 | `curb_height.csv` | city=pooled, group=sidewalk, reading=measured | `median_lo` | 0.0144774 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.030 | `curb_height.csv` | city=pooled, group=sidewalk, reading=measured | `median_hi` | 0.0299256 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 399 | `curb_height.csv` | city=pooled, group=sidewalk, reading=measured | `n_panos` | 399 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.339 | `curb_height.csv` | city=pooled, group=sidewalk, reading=measured | `share_in_claim_band` | 0.338766 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.033 | `curb_height.csv` | city=pooled, group=sidewalk, reading=with_standins | `median_of_pano_medians` | 0.0332609 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.042 | `curb_height.csv` | city=pooled, group=sidewalk, reading=measured_ref_road | `median_of_pano_medians` | 0.0424041 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.021 | `curb_height.csv` | city=pooled, group=walk, reading=measured | `median_of_pano_medians` | 0.0208638 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.008 | `curb_height.csv` | city=pooled, group=road, reading=measured | `median_of_pano_medians` | 0.00794625 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.234 | `curb_height.csv` | city=pooled, group=road, reading=measured | `share_in_claim_band` | 0.233894 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.031 | `curb_height.csv` | city=bend, group=sidewalk, reading=measured | `median_of_pano_medians` | 0.0305415 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.024 | `curb_height.csv` | city=paterson, group=sidewalk, reading=measured | `median_of_pano_medians` | 0.0243227 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.020 | `curb_height.csv` | city=gainesville, group=sidewalk, reading=measured | `median_of_pano_medians` | 0.0196867 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 0.009 | `curb_height.csv` | city=sao_paulo, group=sidewalk, reading=measured | `median_of_pano_medians` | 0.00934119 | doc §0.6, §5.2; fig. 4; #47 comment 5921870275 |
| 2 | `verdict.json` | - | `tiers.0.55.false_non_surface` | 2 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 42 | `verdict.json` | - | `tiers.0.55.false_n` | 42 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.048 | `verdict.json` | - | `tiers.0.55.false_share` | 0.047619 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.013 | `verdict.json` | - | `tiers.0.55.false_lo` | 0.0131572 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.158 | `verdict.json` | - | `tiers.0.55.false_hi` | 0.157901 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 43 | `verdict.json` | - | `tiers.0.55.true_non_surface` | 43 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 770 | `verdict.json` | - | `tiers.0.55.true_n` | 770 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.056 | `verdict.json` | - | `tiers.0.55.true_share` | 0.0558442 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.042 | `verdict.json` | - | `tiers.0.55.true_lo` | 0.0417209 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.074 | `verdict.json` | - | `tiers.0.55.true_hi` | 0.0743772 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.944 | `verdict.json` | - | `tiers.0.55.surface_reading.share` | 0.944156 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 727 | `verdict.json` | - | `tiers.0.55.surface_reading.true_on_walkroad` | 727 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.030 | `detection_classes.csv` | city=pooled, group=missed, arm=tiled | `share_non_surface` | 0.0302115 | doc §0.4, §5.3; fig. 5 |
| 0.271 | `detection_classes.csv` | city=pooled, group=true, arm=tiled | `share_curb_cut` | 0.271429 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.071 | `detection_classes.csv` | city=pooled, group=false, arm=tiled | `share_curb_cut` | 0.0714286 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.254 | `detection_classes.csv` | city=pooled, group=missed, arm=tiled | `share_curb_cut` | 0.253776 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.179 | `detection_classes.csv` | city=pooled, group=true, arm=direct | `share_curb_cut` | 0.179221 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.130 | `detection_classes.csv` | city=pooled, group=missed, arm=direct | `share_curb_cut` | 0.129909 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.301 | `detection_classes.csv` | city=pooled, group=true, arm=direct4096 | `share_curb_cut` | 0.301299 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.269 | `detection_classes.csv` | city=pooled, group=missed, arm=direct4096 | `share_curb_cut` | 0.268882 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.119 | `detection_classes.csv` | city=pooled, group=false, arm=direct4096 | `share_curb_cut` | 0.119048 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.957 | `trap.csv` | city=pooled, lat_band_deg=-80..-70, band=interior | `agreement_direct` | 0.956784 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.856 | `trap.csv` | city=pooled, lat_band_deg=-80..-70, band=interior | `agreement_direct4096` | 0.855679 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.948 | `trap.csv` | city=pooled, lat_band_deg=-80..-70, band=seam | `agreement_direct` | 0.948197 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.819 | `trap.csv` | city=pooled, lat_band_deg=-80..-70, band=seam | `agreement_direct4096` | 0.818679 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.907 | `trap.csv` | city=pooled, lat_band_deg=-10..0, band=seam | `agreement_direct` | 0.906929 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.919 | `trap.csv` | city=pooled, lat_band_deg=-10..0, band=interior | `agreement_direct` | 0.918653 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.959 | `trap.csv` | city=pooled, lat_band_deg=-10..0, band=interior | `agreement_direct4096` | 0.959366 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.12 | `trap.csv` | city=pooled, lat_band_deg=-80..-70, band=interior | `direct4096_share_SKY` | 0.119991 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.15 | `trap.csv` | city=pooled, lat_band_deg=-80..-70, band=seam | `direct4096_share_SKY` | 0.146264 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.06 | `trap.csv` | city=pooled, lat_band_deg=-50..-40, band=interior | `direct4096_share_SKY` | 0.0642953 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.08 | `trap.csv` | city=pooled, lat_band_deg=-50..-40, band=seam | `direct4096_share_SKY` | 0.0822902 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.95 | `trap.csv` | city=pooled, lat_band_deg=-20..-10, band=seam | `tiled_share_ROAD` | 0.948836 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.44 | `trap.csv` | city=pooled, lat_band_deg=-20..-10, band=interior | `tiled_share_ROAD` | 0.441171 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 90 | `sample_counts.csv` | city=bend | `n_sample` | 90 | doc §3; #47 pre-registration comment 5921395361 |
| 113 | `sample_counts.csv` | city=paterson | `n_sample` | 113 | doc §3; #47 pre-registration comment 5921395361 |
| 112 | `sample_counts.csv` | city=gainesville | `n_sample` | 112 | doc §3; #47 pre-registration comment 5921395361 |
| 101 | `sample_counts.csv` | city=sao_paulo | `n_sample` | 101 | doc §3; #47 pre-registration comment 5921395361 |
| 0.906 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad_given_surface` | 0.906431 | doc §5.1 per-city table; fig. 1 |
| 0.905 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_walkroad_given_surface` | 0.905043 | doc §5.1 per-city table; fig. 1 |
| 0.905 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=mirror | `p_walkroad_given_surface` | 0.905233 | doc §5.1 per-city table; fig. 1 |
| 0.902 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad` | 0.902405 | doc §5.1 per-city table; fig. 1 |
| 0.957 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_object` | 0.95739 | doc §5.1 per-city table; fig. 1 |
| 0.966 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_surface_given_object` | 0.965509 | doc §5.1 per-city table; fig. 1 |
| 0.963 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=mirror | `p_surface_given_object` | 0.963288 | doc §5.1 per-city table; fig. 1 |
| 0.488 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=correct | `p_structure_given_wall` | 0.488458 | doc §5.1 per-city table; fig. 1 |
| 0.209 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_structure_given_wall` | 0.208825 | doc §5.1 per-city table; fig. 1 |
| 0.222 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=mirror | `p_structure_given_wall` | 0.222498 | doc §5.1 per-city table; fig. 1 |
| 0.104 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=correct | `kappa` | 0.103606 | doc §5.1 per-city table; fig. 1 |
| 0.053 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=yaw180 | `kappa` | 0.0529467 | doc §5.1 per-city table; fig. 1 |
| 0.057 | `headline.csv` | city=bend, range=le25m, weighting=pixels, depth_arm=mirror | `kappa` | 0.0567275 | doc §5.1 per-city table; fig. 1 |
| 0.905 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad_given_surface` | 0.905211 | doc §5.1 per-city table; fig. 1 |
| 0.902 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_walkroad_given_surface` | 0.902184 | doc §5.1 per-city table; fig. 1 |
| 0.903 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=mirror | `p_walkroad_given_surface` | 0.903316 | doc §5.1 per-city table; fig. 1 |
| 0.892 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad` | 0.892118 | doc §5.1 per-city table; fig. 1 |
| 0.955 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_object` | 0.95494 | doc §5.1 per-city table; fig. 1 |
| 0.941 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_surface_given_object` | 0.940604 | doc §5.1 per-city table; fig. 1 |
| 0.946 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=mirror | `p_surface_given_object` | 0.945611 | doc §5.1 per-city table; fig. 1 |
| 0.674 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=correct | `p_structure_given_wall` | 0.674159 | doc §5.1 per-city table; fig. 1 |
| 0.403 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_structure_given_wall` | 0.403134 | doc §5.1 per-city table; fig. 1 |
| 0.486 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=mirror | `p_structure_given_wall` | 0.486484 | doc §5.1 per-city table; fig. 1 |
| 0.225 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=correct | `kappa` | 0.224529 | doc §5.1 per-city table; fig. 1 |
| 0.146 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=yaw180 | `kappa` | 0.146448 | doc §5.1 per-city table; fig. 1 |
| 0.170 | `headline.csv` | city=paterson, range=le25m, weighting=pixels, depth_arm=mirror | `kappa` | 0.169909 | doc §5.1 per-city table; fig. 1 |
| 0.882 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad_given_surface` | 0.881623 | doc §5.1 per-city table; fig. 1 |
| 0.881 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_walkroad_given_surface` | 0.880575 | doc §5.1 per-city table; fig. 1 |
| 0.881 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=mirror | `p_walkroad_given_surface` | 0.880952 | doc §5.1 per-city table; fig. 1 |
| 0.878 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad` | 0.878477 | doc §5.1 per-city table; fig. 1 |
| 0.966 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_object` | 0.965568 | doc §5.1 per-city table; fig. 1 |
| 0.960 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_surface_given_object` | 0.959865 | doc §5.1 per-city table; fig. 1 |
| 0.955 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=mirror | `p_surface_given_object` | 0.955332 | doc §5.1 per-city table; fig. 1 |
| 0.553 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=correct | `p_structure_given_wall` | 0.553287 | doc §5.1 per-city table; fig. 1 |
| 0.174 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_structure_given_wall` | 0.174417 | doc §5.1 per-city table; fig. 1 |
| 0.216 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=mirror | `p_structure_given_wall` | 0.215795 | doc §5.1 per-city table; fig. 1 |
| 0.117 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=correct | `kappa` | 0.117087 | doc §5.1 per-city table; fig. 1 |
| 0.055 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=yaw180 | `kappa` | 0.0547912 | doc §5.1 per-city table; fig. 1 |
| 0.065 | `headline.csv` | city=gainesville, range=le25m, weighting=pixels, depth_arm=mirror | `kappa` | 0.0652774 | doc §5.1 per-city table; fig. 1 |
| 0.896 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad_given_surface` | 0.896395 | doc §5.1 per-city table; fig. 1 |
| 0.889 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_walkroad_given_surface` | 0.888948 | doc §5.1 per-city table; fig. 1 |
| 0.891 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=mirror | `p_walkroad_given_surface` | 0.89112 | doc §5.1 per-city table; fig. 1 |
| 0.866 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=correct | `p_walkroad` | 0.865532 | doc §5.1 per-city table; fig. 1 |
| 0.929 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=correct | `p_surface_given_object` | 0.929034 | doc §5.1 per-city table; fig. 1 |
| 0.907 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_surface_given_object` | 0.907338 | doc §5.1 per-city table; fig. 1 |
| 0.907 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=mirror | `p_surface_given_object` | 0.906939 | doc §5.1 per-city table; fig. 1 |
| 0.684 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=correct | `p_structure_given_wall` | 0.683716 | doc §5.1 per-city table; fig. 1 |
| 0.466 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=yaw180 | `p_structure_given_wall` | 0.465804 | doc §5.1 per-city table; fig. 1 |
| 0.512 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=mirror | `p_structure_given_wall` | 0.512496 | doc §5.1 per-city table; fig. 1 |
| 0.341 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=correct | `kappa` | 0.341231 | doc §5.1 per-city table; fig. 1 |
| 0.242 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=yaw180 | `kappa` | 0.24191 | doc §5.1 per-city table; fig. 1 |
| 0.266 | `headline.csv` | city=sao_paulo, range=le25m, weighting=pixels, depth_arm=mirror | `kappa` | 0.2658 | doc §5.1 per-city table; fig. 1 |
| 500 | `derived_numbers.csv` | name=det_true_depth_floor | `value` | 500 | doc (see the formula in data/derived_numbers.csv) |
| 143 | `derived_numbers.csv` | name=det_true_depth_ground | `value` | 143 | doc (see the formula in data/derived_numbers.csv) |
| 122 | `derived_numbers.csv` | name=det_true_depth_floor_standin | `value` | 122 | doc (see the formula in data/derived_numbers.csv) |
| 4 | `derived_numbers.csv` | name=det_true_depth_non_horizontal | `value` | 4 | doc (see the formula in data/derived_numbers.csv) |
| 1 | `derived_numbers.csv` | name=det_true_depth_horizontal_nonfloor | `value` | 1 | doc (see the formula in data/derived_numbers.csv) |
| 26 | `derived_numbers.csv` | name=det_false_depth_floor | `value` | 26 | doc (see the formula in data/derived_numbers.csv) |
| 10 | `derived_numbers.csv` | name=det_false_depth_floor_standin | `value` | 10 | doc (see the formula in data/derived_numbers.csv) |
| 5 | `derived_numbers.csv` | name=det_false_depth_ground | `value` | 5 | doc (see the formula in data/derived_numbers.csv) |
| 1 | `derived_numbers.csv` | name=det_false_depth_non_horizontal | `value` | 1 | doc (see the formula in data/derived_numbers.csv) |
| 25 | `derived_numbers.csv` | name=false_seg_Sidewalk | `value` | 25 | doc (see the formula in data/derived_numbers.csv) |
| 5 | `derived_numbers.csv` | name=false_seg_Curb | `value` | 5 | doc (see the formula in data/derived_numbers.csv) |
| 3 | `derived_numbers.csv` | name=false_seg_Curb Cut | `value` | 3 | doc (see the formula in data/derived_numbers.csv) |
| 5 | `derived_numbers.csv` | name=false_seg_Road | `value` | 5 | doc (see the formula in data/derived_numbers.csv) |
| 1 | `derived_numbers.csv` | name=false_seg_Catch Basin | `value` | 1 | doc (see the formula in data/derived_numbers.csv) |
| 1 | `derived_numbers.csv` | name=false_seg_Manhole | `value` | 1 | doc (see the formula in data/derived_numbers.csv) |
| 33 | `derived_numbers.csv` | name=false_group_WALK | `value` | 33 | doc (see the formula in data/derived_numbers.csv) |
| 7 | `derived_numbers.csv` | name=false_group_ROAD | `value` | 7 | doc (see the formula in data/derived_numbers.csv) |
| 0.97 | `derived_numbers.csv` | name=walk_px_millions_ground | `value` | 0.970004 | doc (see the formula in data/derived_numbers.csv) |
| 1.03 | `derived_numbers.csv` | name=walk_px_millions_secondary | `value` | 1.03193 | doc (see the formula in data/derived_numbers.csv) |
| 0.91 | `derived_numbers.csv` | name=walk_px_millions_floor | `value` | 0.913435 | doc (see the formula in data/derived_numbers.csv) |
| 0.12 | `derived_numbers.csv` | name=walk_px_millions_floor_standin | `value` | 0.118497 | doc (see the formula in data/derived_numbers.csv) |
| 0.069 | `derived_numbers.csv` | name=walk_share_of_seen_ground | `value` | 0.068997 | doc (see the formula in data/derived_numbers.csv) |
| 0.118 | `derived_numbers.csv` | name=walk_share_of_seen_floor | `value` | 0.117569 | doc (see the formula in data/derived_numbers.csv) |
| 0.02 | `derived_numbers.csv` | name=object_share_of_surface_0-5 | `value` | 0.0218299 | doc (see the formula in data/derived_numbers.csv) |
| 0.24 | `derived_numbers.csv` | name=object_share_of_surface_15-25 | `value` | 0.243274 | doc (see the formula in data/derived_numbers.csv) |
| 0.505 | `derived_numbers.csv` | name=sidewalk_on_dominant_pixel_pooled | `value` | 0.504796 | doc (see the formula in data/derived_numbers.csv) |
| 0.27 | `derived_numbers.csv` | name=sidewalk_on_dominant_pano_median | `value` | 0.273399 | doc (see the formula in data/derived_numbers.csv) |
| 410 | `derived_numbers.csv` | name=sidewalk_on_dominant_n_panos | `value` | 410 | doc (see the formula in data/derived_numbers.csv) |
| -2 | `derived_numbers.csv` | name=det_lat_p99_deg | `value` | -2.46538 | doc (see the formula in data/derived_numbers.csv) |
| -36 | `derived_numbers.csv` | name=det_lat_p01_deg | `value` | -36.0102 | doc (see the formula in data/derived_numbers.csv) |
| -49 | `derived_numbers.csv` | name=det_lat_min_deg | `value` | -49.1454 | doc (see the formula in data/derived_numbers.csv) |
| 4 | `derived_numbers.csv` | name=det_lat_max_deg | `value` | 4.21884 | doc (see the formula in data/derived_numbers.csv) |
| 0.955 | `derived_numbers.csv` | name=true_on_walkroad_bend | `value` | 0.954545 | doc (see the formula in data/derived_numbers.csv) |
| 0.930 | `derived_numbers.csv` | name=true_on_walkroad_paterson | `value` | 0.930328 | doc (see the formula in data/derived_numbers.csv) |
| 0.924 | `derived_numbers.csv` | name=true_on_walkroad_gainesville | `value` | 0.923977 | doc (see the formula in data/derived_numbers.csv) |
| 0.975 | `derived_numbers.csv` | name=true_on_walkroad_sao_paulo | `value` | 0.974522 | doc (see the formula in data/derived_numbers.csv) |
| 0.896 | `derived_numbers.csv` | name=det_group_agree_direct | `value` | 0.895888 | doc (see the formula in data/derived_numbers.csv) |
| 0.684 | `derived_numbers.csv` | name=det_class_agree_direct | `value` | 0.684164 | doc (see the formula in data/derived_numbers.csv) |
| 0.958 | `derived_numbers.csv` | name=det_group_agree_direct4096 | `value` | 0.958005 | doc (see the formula in data/derived_numbers.csv) |
| 0.843 | `derived_numbers.csv` | name=det_class_agree_direct4096 | `value` | 0.84252 | doc (see the formula in data/derived_numbers.csv) |
| 1143 | `derived_numbers.csv` | name=det_n_true_false_missed | `value` | 1143 | doc (see the formula in data/derived_numbers.csv) |
| 1 | `verdict.json` | - | `tiers.0.55.per_city.bend.false_non_surface` | 1 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 7 | `verdict.json` | - | `tiers.0.55.per_city.bend.false_n` | 7 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0 | `verdict.json` | - | `tiers.0.55.per_city.gainesville.false_non_surface` | 0 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 9 | `verdict.json` | - | `tiers.0.55.per_city.gainesville.false_n` | 9 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0 | `verdict.json` | - | `tiers.0.55.per_city.paterson.false_non_surface` | 0 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 5 | `verdict.json` | - | `tiers.0.55.per_city.paterson.false_n` | 5 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 1 | `verdict.json` | - | `tiers.0.55.per_city.sao_paulo.false_non_surface` | 1 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 21 | `verdict.json` | - | `tiers.0.55.per_city.sao_paulo.false_n` | 21 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures); #47 comment 5921870275 (findings) |
| 0.024 | `detection_classes.csv` | city=pooled, group=false, arm=direct | `share_curb_cut` | 0.0238095 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.191 | `detection_classes.csv` | city=pooled, group=unsure_missed, arm=tiled | `share_curb_cut` | 0.191358 | doc takeaways + §0 + §5; PR #124 body; CLAUDE.md block; #47 comments 5923015034 (correction) and 5932049764 (figures) |
| 0.056 | `detection_classes.csv` | city=pooled, group=unsure_missed, arm=tiled | `share_non_surface` | 0.0555556 | doc §0.4, §5.3; fig. 5 |
| 0.870 | `detection_classes.csv` | city=pooled, group=true, arm=tiled | `share_WALK` | 0.87013 | doc §0.4, §5.3; fig. 5 |
| 0.786 | `detection_classes.csv` | city=pooled, group=false, arm=tiled | `share_WALK` | 0.785714 | doc §0.4, §5.3; fig. 5 |
| 0.167 | `detection_classes.csv` | city=pooled, group=false, arm=tiled | `share_ROAD` | 0.166667 | doc §0.4, §5.3; fig. 5 |
| 0.825 | `detection_classes.csv` | city=pooled, group=missed, arm=tiled | `share_WALK` | 0.824773 | doc §0.4, §5.3; fig. 5 |
| 0.145 | `detection_classes.csv` | city=pooled, group=missed, arm=tiled | `share_ROAD` | 0.145015 | doc §0.4, §5.3; fig. 5 |
| 0.074 | `detection_classes.csv` | city=pooled, group=true, arm=tiled | `share_ROAD` | 0.074026 | doc §0.4, §5.3; fig. 5 |
| 0.107 | `derived_numbers.csv` | name=band_non_surface_share | `value` | 0.107143 | doc (see the formula in data/derived_numbers.csv) |
| 485 | `derived_numbers.csv` | name=gt_panos_total | `value` | 485 | doc (see the formula in data/derived_numbers.csv) |
| 69 | `derived_numbers.csv` | name=excluded_not_measured | `value` | 69 | doc (see the formula in data/derived_numbers.csv) |
| 47 | `derived_numbers.csv` | name=n_unsure | `value` | 47 | doc (see the formula in data/derived_numbers.csv) |
| 4 | `derived_numbers.csv` | name=n_duplicate | `value` | 4 | doc (see the formula in data/derived_numbers.csv) |
| 331 | `derived_numbers.csv` | name=n_missed | `value` | 331 | doc (see the formula in data/derived_numbers.csv) |
| 162 | `derived_numbers.csv` | name=n_unsure_missed | `value` | 162 | doc (see the formula in data/derived_numbers.csv) |
| 140 | `derived_numbers.csv` | name=n_band_0p30 | `value` | 140 | doc (see the formula in data/derived_numbers.csv) |
| 0.114 | `derived_numbers.csv` | name=unseen_below_horizon_share | `value` | 0.114075 | doc (see the formula in data/derived_numbers.csv) |
| 14.5 | `derived_numbers.csv` | name=false_wilson_width_pts | `value` | 14.4744 | doc (see the formula in data/derived_numbers.csv) |
| 0.241 | `derived_numbers.csv` | name=curb_cut_true_wilson_lo | `value` | 0.24121 | doc (see the formula in data/derived_numbers.csv) |
| 0.304 | `derived_numbers.csv` | name=curb_cut_true_wilson_hi | `value` | 0.303916 | doc (see the formula in data/derived_numbers.csv) |
| 0.025 | `derived_numbers.csv` | name=curb_cut_false_wilson_lo | `value` | 0.02459 | doc (see the formula in data/derived_numbers.csv) |
| 0.190 | `derived_numbers.csv` | name=curb_cut_false_wilson_hi | `value` | 0.190097 | doc (see the formula in data/derived_numbers.csv) |
| 0.97 | `derived_numbers.csv` | name=walkroad_given_surface_0-5 | `value` | 0.966328 | doc (see the formula in data/derived_numbers.csv) |
| 0.80 | `derived_numbers.csv` | name=walkroad_given_surface_5-10 | `value` | 0.797928 | doc (see the formula in data/derived_numbers.csv) |
| 0.63 | `derived_numbers.csv` | name=walkroad_given_surface_10-15 | `value` | 0.626838 | doc (see the formula in data/derived_numbers.csv) |
| 0.50 | `derived_numbers.csv` | name=walkroad_given_surface_15-25 | `value` | 0.495618 | doc (see the formula in data/derived_numbers.csv) |
| 0.29 | `derived_numbers.csv` | name=walkroad_given_surface_beyond | `value` | 0.287166 | doc (see the formula in data/derived_numbers.csv) |
| 0.79 | `derived_numbers.csv` | name=false_group_share_WALK | `value` | 0.785714 | doc (see the formula in data/derived_numbers.csv) |
| 0.17 | `derived_numbers.csv` | name=false_group_share_ROAD | `value` | 0.166667 | doc (see the formula in data/derived_numbers.csv) |
| 0.053 | `derived_numbers.csv` | name=object_off_floor_le25m | `value` | 0.052636 | doc (see the formula in data/derived_numbers.csv) |
| 0.017 | `derived_numbers.csv` | name=pixels_off_floor_le25m | `value` | 0.017132 | doc (see the formula in data/derived_numbers.csv) |
| 9 | `derived_numbers.csv` | name=car_components_near_nadir | `value` | 9 | doc (see the formula in data/derived_numbers.csv) |
| 336 | `derived_numbers.csv` | name=car_components_total | `value` | 336 | doc (see the formula in data/derived_numbers.csv) |
| 0.941 | `headline.csv` | city=pooled, range=le25m, weighting=pixels_y_le_0.75, depth_arm=correct | `p_surface_given_object` | 0.94136 | doc §0.3, §5.1 |
| 0.928 | `headline.csv` | city=pooled, range=le25m, weighting=pixels_y_le_0.75, depth_arm=yaw180 | `p_surface_given_object` | 0.927968 | doc §0.3, §5.1 |
| 0.928 | `headline.csv` | city=pooled, range=le25m, weighting=pixels_y_le_0.75, depth_arm=mirror | `p_surface_given_object` | 0.928345 | doc §0.3, §5.1 |
| 0.940 | `headline.csv` | city=pooled, range=le25m, weighting=pixels_y_le_0.75, depth_arm=correct | `p_surface_given_object_excl_ego` | 0.939922 | doc §0.3, §5.1 |
| 0.000 | `trap.csv` | city=bend, lat_band_deg=-80..-70, band=interior | `direct4096_share_SKY` | 0 | doc takeaways, §0.7, §5.4; fig. 6; PR body |
| 0.110 | `trap.csv` | city=paterson, lat_band_deg=-80..-70, band=interior | `direct4096_share_SKY` | 0.110284 | doc takeaways, §0.7, §5.4; fig. 6; PR body |
| 0.247 | `trap.csv` | city=gainesville, lat_band_deg=-80..-70, band=interior | `direct4096_share_SKY` | 0.246577 | doc takeaways, §0.7, §5.4; fig. 6; PR body |
| 0.318 | `trap.csv` | city=gainesville, lat_band_deg=-80..-70, band=seam | `direct4096_share_SKY` | 0.317661 | doc takeaways, §0.7, §5.4; fig. 6; PR body |
| 0.097 | `trap.csv` | city=sao_paulo, lat_band_deg=-80..-70, band=interior | `direct4096_share_SKY` | 0.0974014 | doc takeaways, §0.7, §5.4; fig. 6; PR body |
