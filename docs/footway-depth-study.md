# Footway: do an off-the-shelf segmenter and GSV depth agree on the walkable surface?

**Issue:** [#47](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/47), step 2: *run a pretrained
Mask2Former/SegFormer over a few hundred panos and measure agreement against "depth says a near-horizontal
plane ~camera-height below". Disagreement is the interesting output either way.*
**Date:** 2026-09-30. **Branch:** `footway-depth-47`.
**Reproduce:** `scripts/footway_segmentation.py` (every number and figure below; see §8). **Tests:**
`tests/test_footway_segmentation.py` covers the equirect/perspective tile mapping round trip, seam wrap, the
render-then-stitch recovery of a labelled equirect, the tie-break, the class collapse and the verdict rule. The
model is never loaded.
**Pre-registration:** the [dated #47 comment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/47#issuecomment-5921395361)
fixed the sample, model, tiling, class mapping and readings before any GT-conditioned number was computed.
Commit `d1ffbb6` put the rule into code (`verdict()`). One clarification was made before the first scoring run
(`75f2d05`): headline denominators exclude the pixels no tile sees. Everything marked *exploratory* was added
after scoring and does not enter a verdict.
**Builds on:** step 1, [docs/depth-at-detection-study.md](depth-at-detection-study.md). Its depth classes, plane
lookup and `offset_local` are imported, not reimplemented.

## 0. Summary

The sample is 416 judged benchmark panoramas from bend, paterson, gainesville and São Paulo, each with a
`measured` depth payload. Each was segmented with `facebook/mask2former-swin-large-mapillary-vistas-semantic`
on 16 perspective tiles, and the label maps were stitched back onto a 1024x512 equirect grid. Findings, by
question:

1. **Where depth models a floor, the segmenter mostly sees walkable surface.** Over every below-horizon pixel,
   P(WALK or ROAD | depth surface) is 0.871 (0.897 within 25 m), and the four cities fall within 0.850–0.881.
   The reverse holds almost without exception: 0.997 of the pixels the segmenter calls WALK or ROAD are on a
   depth floor plane. The agreement falls with range, from 0.97 at 0–5 m to 0.50 at 15–25 m (fig. 2), because
   the remainder is objects and terrain that depth folds into the ground.
2. **"Depth omits objects", now with a number.** 7.0% of depth-surface pixels are OBJECT by the segmenter (5.5%
   within 25 m; 2% at 0–5 m, rising to 24% at 15–25 m). Seen from the other side, **94.7% of segmenter-OBJECT
   pixels within 25 m sit on a depth floor plane.** Depth models the ground *through* cars, people, vegetation
   and poles. It draws a plane where they stand, not an obstacle. Walls are the exception: 67% of depth's
   steep planes are segmenter STRUCTURE.
3. **(C) Segmenter class at the detection as a false-positive filter: NOT SUPPORTED.**
   - Only 2 of 42 verdict-False detections (0.048, Wilson 0.013–0.158) sit on a non-surface class, against 43 of
     770 verdict-True (0.056, 0.042–0.074). The False share is *lower*, against a registered margin of +20 pts.
   - `underpowered` is **false**. The False interval is 14.5 pts wide, and even its top end is only 10 pts above
     the True share, so an effect of the registered size is implausible, not just undetected.
   - The reviewed false positives are ramp-like things *on the footway*: 79% Sidewalk/Curb/Curb Cut, 17% road
     (fig. 4). The segmenter's surface class carries no more ramp-vs-not signal than depth did in step 1.
4. **(C') The surface reading: 0.944 of verdict-True detections (727/770) are on WALK or ROAD.** Missed marks
   read 0.970. Most of the 43 True detections the segmenter calls non-surface are Terrain at a ramp's grass edge
   or a pole at the corner (fig. 6, bottom row). That is label noise at one pixel, not a missing surface.
5. **(B) The sidewalk plane is not ~0.15 m above the road in GSV depth.**
   - Over 399 panos, the median height of Vistas-`Sidewalk` pixels on a measured floor plane above the local
     road is **0.022 m (CI 0.014–0.030)**. By the registered reading that is **not consistent** with ~0.15 m. The
     road-on-floor control reads 0.008 m.
   - About half of sidewalk pixels (50.5%) are not on a secondary plane at all: they share the dominant ground
     plane with the road, so their height step is zero by construction and they are not in the median.
   - Exploratory: restricting to pixels whose local reference is segmenter-ROAD raises the median to 0.042 m
     (0.025–0.055). That is still a third of a real curb.
   - GSV's plane model does not carry the curb step. This matches step 1's finding that real ramp planes sit
     ~0.03 m up.
6. **The trap is real, and it shows at the fine class, not the coarse group.**
   - Feeding the equirect directly agrees with the tiled arm on the collapsed group for 0.92–0.99 of pixels in
     every latitude band, and the seam band is no worse than the interior (fig. 5). The coarse surface/object
     split survives the projection.
   - At the detection pixels, however, the arms disagree on the **fine** Vistas class 32% of the time (on the
     group, 10%). **`Curb Cut` under a verdict-True detection falls from 0.271 tiled to 0.179 direct, and under
     missed marks from 0.254 to 0.130**, with the difference going to Sidewalk and Curb. The direct arm quietly
     loses exactly the class this project cares about, and it still looks plausible in a spot check.
7. **Exploratory lead, outside every reading.** The fine class `Curb Cut` at the pixel is a *positive* signal:
   0.271 of True detections (Wilson 0.241–0.304) against 0.071 of False (3/42, 0.025–0.190), with
   non-overlapping intervals. It is not registered and n(False) is small. It is the candidate rule for a
   pre-registered follow-up (§7).

**Net for #47.** Depth and a segmenter agree on *where the walkable surface is*. The segmenter adds exactly the
object term the issue predicted depth would lack (item 2). That makes stage 3, the impedance region, the
product of the two, as the issue argued. Neither signal, alone or at a detection's single pixel, separates real
ramps from the reviewed false positives, and GSV depth does not resolve the curb step that a "surface above the
road" rule would need.

## 1. Background and goals

Step 1 showed that GSV's depth puts a near-horizontal plane under 99.4% of detections and under verdict-False
detections as often as verdict-True ones. Depth locates the surface, but it does not discriminate a ramp from a
non-ramp. The issue's premise is that depth models the **surface** and omits the **objects**. Stage 3, the
"impedance region", would then need a segmenter to supply the object and surface-class term. Step 2 measures
whether the two agree, where they disagree, and whether the segmenter adds the discriminating term at a
detection that depth lacked.

## 2. Research questions

- **RQ-A.** Over below-horizon pixels, how do depth plane classes and segmenter classes co-occur? In
  particular, what is P(WALK or ROAD | depth surface), and what share of depth's surface is actually an object?
- **RQ-B.** Where the segmenter sees sidewalk on a secondary depth plane, how high does depth put it above the
  road, and is that the ~0.15 m the issue quotes?
- **RQ-C.** At a detection, does the segmenter class separate verdict-False from verdict-True? (Step 1's rule
  (ii), with the segmenter in place of depth.)
- **RQ-D.** How badly does the equirect-direct shortcut (the issue's "known trap") distort the answer?

## 3. Data

| city | judged GT panos | measured payload | stand-in / degenerate / implausible (excluded) |
|---|---:|---:|---|
| bend | 110 | 90 | 20 / 0 / 0 |
| paterson | 125 | 113 | 10 / 2 / 0 |
| gainesville | 125 | 112 | 12 / 0 / 1 |
| São Paulo | 125 | 101 | 24 / 0 / 0 |
| **total** | **485** | **416** | 69 |

The source files are `runs/_pooled/footway/sample.csv` (with the JPEG and payload sha256 per pano) and
`sample_counts.csv`. The GT join is `eval_sites.judged_gt_panos`: the same drift gate as every sibling study,
with 0 panos skipped. Pixels come from `../RampNet/benchmark/<city>/panos/` (16384x8192, read-only). Depth
payloads come from the runs' `depth/`. Detections are the stored ones, ≥ 0.30. The verdicts are at the
benchmark tier (0.55), the tier the bundles were judged at. On these 416 panos: 770 verdict-True, **42
verdict-False**, 47 unsure, 4 duplicate, 331 non-unsure missed marks, 162 unsure missed marks, and 140
detections in the 0.30–0.55 band (no verdict).

## 4. Methods

### 4.1 Segmenter

The segmenter is `facebook/mask2former-swin-large-mapillary-vistas-semantic`, revision
`4772b6bf101d91f2534c106dc524d906aeb3c68a` (Mapillary Vistas v1.2, 65 classes). The vocabulary this question
needs is there: Sidewalk, Curb, **Curb Cut**, Crosswalk - Plain, Pedestrian Area. It loaded, so the Cityscapes
SegFormer fallback was not used. It runs in fp16 on the local RTX 3070 (5.9 tiles/s, 1.8 GiB peak; torch
2.8.0+cu126, transformers 5.12.1).

Two choices differ from a naive HF pipeline. Both are deliberate:
- **The processor's default resize to 384x384 is off** (`do_resize=False`). The model sees the tiles at their
  rendered 1024x1024.
- **Semantic maps are computed the standard Mask2Former way.** Take softmax(class) without the no-object column,
  multiply by sigmoid(mask) at mask resolution, upsample bilinearly to the target, then take the argmax. This
  avoids `post_process_semantic_segmentation`, which pushes every mask through a fixed 384x384 intermediate
  (anisotropically, for a 2:1 input).

The load report flags `swin.layernorm` as newly initialised. That norm feeds only the backbone's pooled output,
which Mask2Former does not read; the stage feature maps use their own `hidden_states_norms`. The output was
checked visually on a smoke pano before the full run.

### 4.2 Tiling and the stitch

Each pano is resized to 4096x2048 (LANCZOS), then rendered bilinearly to **16 perspective tiles**. Each tile is
90° FOV and 1024x1024, the same ~11.4 px/deg as the resized pano. There are 8 headings every 45° and 2 pitch
rows, 0° and −35°. The geometry is pure image-frame: x from the left of the JPEG, longitude increasing to the
right, heading 0 = the centre column, with no compass involved.

Each tile's label map is voted back onto the 1024x512 grid. Every tile whose frustum contains the pixel
contributes its nearest tile pixel. The majority label wins, and a tie goes to the tile whose optical axis is
nearest in angle.

**Coverage.** No tile sees the zenith above ~+43° (irrelevant below the horizon) or the far nadir below
~−79°. That is 11.4% of below-horizon pixels, all within ~0.5 m of the car. They carry `NONE` and are left out
of every headline denominator, which is the pre-scoring clarification. The pre-registration comment had
assumed only the nadir would go unseen; the zenith gap is stated here instead.

**Control arm.** The same model sees the 2048x1024 equirect directly, read out on the same grid. It is reported
by latitude band and seam band (within 11° of x = 0/1), and never in a headline.

### 4.3 Class collapse

The mapping is `COLLAPSE` in the script. Every Vistas name is mapped explicitly, and an unmapped name refuses.

| group | Vistas classes |
|---|---|
| WALK | Sidewalk, Curb, Curb Cut, Crosswalk - Plain, Pedestrian Area |
| ROAD | Road, Lane Marking - Crosswalk, Lane Marking - General, Parking, Bike Lane, Service Lane; flush fixtures Manhole, Catch Basin, Pothole; Rail Track |
| OBJECT | every vehicle, person and rider, Bird, Ground Animal, Vegetation, Guard Rail, Barrier, all poles, signs, lights and street furniture, Car Mount, Ego Vehicle |
| STRUCTURE | Building, Wall, Fence, Bridge, Tunnel |
| SKY | Sky |
| OTHER | Terrain, Mountain, Sand, Snow, Water |

Two placements are routine calls, stated here. Flush fixtures count as ROAD because they are part of the
drivable surface, not objects on it. Ego Vehicle and Car Mount count as OBJECT because they are the camera car.

### 4.4 Depth side

The depth classes are step 1's, taken from `depth_at_detection.plane_class`: `ground` (the dominant plane),
`floor`, `horizontal_nonfloor`, `non_horizontal` and `no_plane`. `floor` is split by `depth.is_standin` into a
measured floor and a stand-in floor. "Depth surface" = ground ∪ floor ∪ stand-in floor.

The plane lookup is `depth._plane_at` in the image frame, with an exact ray, and `offset_local` comes from
`depth_at_detection.classify`. `depth.depth_at` is never called.

The pixel grid is one grid pixel per payload pixel below the horizon, 512x128 per pano. Range bins come from
the flat raycast at 2.6 m (`geo.detection_ground_point`, `apply_pose=False`). "Beyond" means past 25 m or at
the horizon.

### 4.5 The readings, as registered

- **(A)** Descriptive agreement matrix. Headlines: P(WALK or ROAD | depth surface) and P(OBJECT | depth surface).
- **(B)** For each pano, take the median `offset_local` of `Sidewalk` pixels on a measured floor plane within
  25 m, excluding pixels whose plane or local reference is a stand-in. At most 1,500 pixels per pano are used,
  seeded subsample; a pano needs ≥ 20 pixels. The reading is the median across panos, with an order-statistic
  CI, and it is "consistent with ~0.15 m" iff it lies in [0.05, 0.30). The headline uses `Sidewalk` rather
  than all of WALK because Crosswalk - Plain lies at road level (the one deviation from the plan, made in the
  pre-registration). All of WALK, ROAD (the control) and a with-stand-ins reading are reported beside it.
- **(C)** `non_surface` = the tiled group at the detection pixel is not WALK or ROAD.
  - **SUPPORTED** iff the pooled False share exceeds the True share by ≥ 20 pts **and** the False Wilson lower
    bound exceeds the True upper bound.
  - `underpowered` iff the False interval is wider than 20 pts.
  - Tier 0.55.
- **(C')** Descriptive: the share of True detections on WALK or ROAD.

## 5. Findings

### 5.1 RQ-A: agreement (`agreement.csv`, `headline.csv`; figs. 1–2)

![fig1](figures/footway-depth/fig1_agreement.png)

| city | P(WALK∪ROAD \| depth surface) | within 25 m | P(OBJECT \| depth surface) | within 25 m | P(depth surface \| OBJECT), 25 m | P(WALK∪ROAD \| depth wall), 25 m |
|---|---:|---:|---:|---:|---:|---:|
| bend | 0.877 | 0.906 | 0.060 | 0.042 | 0.957 | 0.044 |
| paterson | 0.881 | 0.905 | 0.076 | 0.063 | 0.955 | 0.151 |
| gainesville | 0.850 | 0.882 | 0.052 | 0.031 | 0.966 | 0.088 |
| São Paulo | 0.877 | 0.896 | 0.095 | 0.084 | 0.929 | 0.172 |
| **pooled** | **0.871** | **0.897** | **0.070** | **0.055** | **0.947** | **0.153** |

- **The dominant ground plane is road.** 87% of its pixels are ROAD and 7% WALK.
- **Secondary floor planes are where the sidewalk lives.** They are 66% ROAD, 12% WALK and 12% OBJECT.
- **Stand-in floors are the most object-laden surface class.** They are 23% OBJECT. That fits step 1's picture
  of a plane Google puts where it has no reconstruction.
- **Depth's steep planes are mostly buildings and walls.** They are 67% STRUCTURE and 27% OBJECT.
- **`no_plane` is 76% OBJECT** (vehicles and vegetation at the horizon). Below the horizon it is rare: 80,775
  of 27.2 M pixels.
- **Range drives the surface agreement** (fig. 2). P(WALK or ROAD | depth surface) falls from 0.97 at 0–5 m to
  0.80, 0.63, 0.50 and 0.29 (beyond). The object share rises from 0.02 to 0.10, 0.18, 0.24 and 0.42. Cities track each other within
  a few points; São Paulo is the most object-dense.

![fig2](figures/footway-depth/fig2_by_range.png)

- **Gallery rows 3 and 1 show the disagreements.** Where depth reports a *wall* and the segmenter a sidewalk
  (15% of wall pixels within 25 m), the fixed-rule gallery (§5.5, row 3) shows mostly **steeply sloped
  pavement**: São Paulo's sloped sidewalks, a raked driveway apron, a curb face. Depth's 18° tilt cut calls
  these walls; a ramp built steeper than 18° would read the same way. Where depth reports a surface and the
  segmenter an object (row 1), it is a parked bus, vegetation, a car body, or the camera vehicle's own
  blur patch at the nadir.

### 5.2 RQ-B: curb height (`curb_height.csv`, `curb_pano_medians.csv`, fig. 3)

| reading | panos | median of pano medians [95% CI] | pixel p10 / median / p90 |
|---|---:|---|---|
| **Sidewalk, measured planes (registered)** | 399 | **0.022 [0.014, 0.030]** | −0.10 / 0.02 / 0.19 |
| Sidewalk, with stand-ins | 401 | 0.033 [0.025, 0.039] | −0.09 / 0.03 / 0.22 |
| Sidewalk, reference = segmenter ROAD (*exploratory*) | 366 | 0.042 [0.025, 0.055] | −0.13 / 0.05 / 0.26 |
| all WALK, measured | 407 | 0.021 [0.013, 0.028] | −0.10 / 0.02 / 0.18 |
| ROAD on a floor plane (control) | 413 | 0.008 [0.002, 0.012] | −0.09 / 0.01 / 0.11 |

Per city, the registered Sidewalk reading is bend 0.031, paterson 0.024, gainesville 0.020 and São Paulo 0.009.
Every city is below the 0.05 band edge. **Reading: not consistent with ~0.15 m.**

![fig3](figures/footway-depth/fig3_curb_height.png)

The median understates any step depth might carry, for three reasons:

1. Only half the sidewalk is on a secondary plane at all: 50.5% of Sidewalk pixels within 25 m share the
   dominant plane with the road.
2. The column walk's reference is "the first different floor plane below", which can be another piece of
   sidewalk. The exploratory road-referenced reading addresses this and moves the median only to 0.042 m.
3. `offset_local` carries the reference plane's tilt extrapolated to the hit (step 1 §5.4).

None of these lifts the sidewalk into the band. GSV's plane model does not resolve a 15 cm curb. The issue's
"~0.15 m above the modelled road" is not a property of GSV depth that can be measured here, neither for ramps
(step 1: ~0.03 m) nor for sidewalks (here: 0.02–0.04 m).

### 5.3 RQ-C: at the detection (`detections.csv`, `detection_classes.csv`, `verdict.json`, fig. 4)

| group | n | non-surface share [Wilson 95%] | WALK | ROAD | `Curb Cut` (*exploratory*) |
|---|---:|---|---:|---:|---:|
| verdict True | 770 | 0.056 [0.042, 0.074] | 0.870 | 0.074 | 0.271 |
| verdict False | 42 | 0.048 [0.013, 0.158] | 0.786 | 0.167 | 0.071 |
| missed (non-unsure) | 331 | 0.030 [0.016, 0.055] | 0.825 | 0.145 | 0.254 |
| unsure missed | 162 | 0.056 [0.029, 0.102] | 0.765 | 0.179 | 0.191 |
| 0.30–0.55 band | 140 | 0.107 | — | — | 0.157 |

**(C) NOT SUPPORTED.** The gap is −0.8 pts against a registered +20. `underpowered` is false, because the
False interval is 14.5 pts wide. Per city, the False non-surface counts are 1/7, 0/5, 0/9 and 1/21.

**(C') 0.944 of True on WALK or ROAD** (bend 0.955, paterson 0.930, gainesville 0.924, São Paulo 0.975).

![fig4](figures/footway-depth/fig4_detections.png)

The reviewed false positives sit on the footway (Sidewalk 25, Curb 5, Curb Cut 3) or the road (5). They are
driveway cuts, ramp-like curb transitions and paving changes, which is what a ramp detector's errors should look
like. A surface-class filter cannot remove them. The 0.30–0.55 band is somewhat more off-surface (0.107) than the
0.55 tier (0.052), but it has no verdicts.

Depth at the same pixels agrees with step 1. True detections sit on floor 500 / ground 143 / stand-in floor 122
/ steep 4 / overhang 1; False on floor 26 / stand-in 10 / ground 5 / steep 1.

### 5.4 RQ-D: the trap (`trap.csv`, fig. 5)

![fig5](figures/footway-depth/fig5_trap.png)

- **On the collapsed group, the direct arm agrees with the tiled arm over most of the sphere.** Pixel agreement
  is 0.95–0.97 below −20°, dips to 0.92 at −10..0° (the horizon band, where small objects and the sidewalk edge
  live), and is ≥ 0.98 above +10°.
- **The seam is not worse than the interior.** The model handles the wrap as well as it handles anything
  else.
- **At the nadir the direct arm shifts toward ROAD.** At −80..−70° it reads 0.95 ROAD / 0.03 WALK against the
  tiled 0.89 / 0.05.
- **Above +43° nothing can be compared.** The tiled arm does not see it.

The coarse numbers therefore make the trap look harmless. It is not harmless where it matters. At the 1,143
judged detection and missed-mark pixels the two arms disagree on the group 10% of the time and on the fine class
32% of the time. **`Curb Cut` drops from 0.271 to 0.179 of True detections and from 0.254 to 0.130 of missed
marks** in the direct arm. This is the issue's "plausible garbage that will not be obvious in a spot check",
measured: the map looks right at a glance and quietly loses the one fine class a curb-ramp pipeline would use.
Any downstream use of this segmenter on our panoramas should tile.

### 5.5 The disagreement gallery (`gallery.csv`, fig. 6)

The selection rule is fixed, with seed 47. For each of three pixel categories, panos are ranked by the size of
the category's largest 8-connected component within 25 m. Six are drawn from the top 20, and each is marked at
the component pixel nearest its centroid. The fourth row is six draws from all 45 detections at ≥ 0.55 that
the segmenter calls non-surface.

![fig6](figures/footway-depth/fig6_gallery.jpg)

1. **Row 1: depth surface, segmenter OBJECT.** A bus, a hedge, a parked car's body and the camera car's own
   nadir patch. Depth draws its ground plane straight through each.
2. **Row 2: depth surface, segmenter STRUCTURE.** Chain-link and wood fences, a graffiti wall and a stone
   retaining wall. The fences are see-through: depth models the ground behind them, while the segmenter labels
   the fence. The walls sit at the ground-contact line.
3. **Row 3: depth wall, segmenter WALK.** Steeply sloped sidewalk and driveway aprons (São Paulo), a curb face
   and a store threshold. The plane really is tilted more than 18°. This is the case where depth carries slope
   information the segmenter's class does not, and it is relevant to stage 3's severity question.
4. **Row 4: True detections the segmenter calls non-surface.** All six are verdict-True ramps whose peak pixel
   lands on the grass edge (Terrain) or a signal pole beside the ramp. This is one-pixel label noise, which is
   why a class *at the peak* is a weak test (§6).

## 6. Discussion

**What this means for stage 2/3 of #47.**
- **Stage 2 works off the shelf, with tiling.** A pretrained Vistas model and GSV depth agree on the walkable
  surface to ~0.9 within 25 m, and the segmenter supplies the object term depth structurally lacks: 95% of object
  pixels within 25 m sit on a depth floor plane.
- **Stage 3 is therefore well-posed.** An obstacle's footprint is where a segmenter-OBJECT region meets a depth
  floor plane, and depth gives that contact line a metric range.
- **The curb step is the gap.** Depth cannot supply it, so any stage-3 quantity that needs "how high above the
  road" (curb height, a missing-ramp inference) needs another instrument: monocular depth, multi-view
  triangulation, or the box annotations.

**What it does not do.** A class at a detection's single peak pixel is no false-positive filter, from depth
(step 1) or from the segmenter (here). The false positives are footway things. Discrimination has to come from
the detector's own appearance model, from multi-view consistency (#27), or from a region-level rule rather than
one pixel. The exploratory `Curb Cut` result suggests the region-level version is worth one pre-registered try
(§7).

**For the heuristic cropper** (ProjectSidewalk/sidewalk-panorama-tools#32): the segmenter's WALK region around a
detection is a plausible context-extent input. The depth plane under it gives the range (step 1's
`range_depth_m`). This is a direction, not a result: nothing here measures crop quality.

**Limits.**
- 42 verdict-False detections; the null for (C) is informative only because the observed gap has the wrong
  sign.
- One model; no second segmenter was run.
- The pixel test is at the heatmap peak, which RampNet's upsampled heatmap quantises to an 8-cell grid
  ([RampNet#221](https://github.com/ProjectSidewalk/RampNet/issues/221)), so the peak can land a few pixels
  off the ramp itself (fig. 6, row 4).
- The zenith is unseen by the tiled arm, which matters for nothing below the horizon.
- The benchmark panos are corner-heavy by construction.

## 7. Recommendations

1. Tile. Never feed the equirect to a perspective segmenter for fine classes. The curb-cut loss is ~9–12 pts at
   the pixels that matter.
2. Do not build a surface-class false-positive filter from either depth or the segmenter at the peak pixel.
3. Next pre-registered step (not done here): **a region-level `Curb Cut` score**. Take the share of Curb Cut in
   a window around the detection, choose the window size on a held-out city, and score it on the others.
   Evaluate it as a confirmation/ranking signal and as a missed-ramp miner. 25% of non-unsure missed marks
   already sit on Curb Cut pixels, so this ties into RampNet#158's mining.
4. For stage 3's metric term, take the ground-contact range from depth, and do not rely on depth for heights
   under ~0.3 m.

## 8. Reproducibility

```
python scripts/footway_segmentation.py sample   --run-root <runs> --benchmark-root ../RampNet/benchmark
python scripts/footway_segmentation.py tiles    --benchmark-root ../RampNet/benchmark --workers 8   # ~14 min
python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/tiles --out runs/_pooled/footway/work/tile_labels --fp16 --batch-size 2
python scripts/footway_segmentation.py segment --in runs/_pooled/footway/work/direct --out runs/_pooled/footway/work/direct_labels --target 1024x512 --fp16 --batch-size 1
python scripts/footway_segmentation.py stitch   --run-root <runs> --benchmark-root ../RampNet/benchmark
python scripts/footway_segmentation.py compare  --run-root <runs> --benchmark-root ../RampNet/benchmark
python scripts/footway_segmentation.py figures  --run-root <runs> --benchmark-root ../RampNet/benchmark
```

- `segment` needs torch + transformers, deliberately not in requirements.txt. On the local RTX 3070 it took
  ~19 min for the 6,656 tiles and ~2.5 min for the 416 direct equirects.
- Tiles and label maps live in `runs/_pooled/footway/work/` and are untracked. The model id, revision, runtime,
  every stitched and direct label map's sha256, and an aggregate hash over the 6,656 tile label maps are in
  `runs/_pooled/footway/masks_manifest.json`, copied to `docs/figures/footway-depth/data/`.
- `verdict` falls back to the committed copies.
- `compare` refuses if any sampled pano no longer passes the GT join.
