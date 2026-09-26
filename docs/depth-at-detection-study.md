# Depth at the detection: does harvested GSV depth alone locate a curb ramp's surface?

**Issue:** [#47](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/47), step 1 — *take panos with a
RampNet detection and a harvested depth payload, read the modelled plane at the detection's pixel: is it a
near-horizontal plane at plausible range, how often is it "no plane", and does the ~0.15 m ramp-above-road offset
eat the signal?*
**Date:** 2026-09-26. **Branch:** `depth-at-detection-47`.
**Reproduce:** `scripts/depth_at_detection.py` (every number and figure below; see §9). **Tests:**
`tests/test_depth_at_detection.py` (the plane class in the image frame, the local height offset, the column walk,
the verdict rule), all on synthetic payloads.
**Pre-registration:** the plan comment on #47 fixed the classes, the offset definition, the bins and the reading
before any GT-conditioned number was computed. The commit that added this section also put the reading into code
(`verdict()`), before the full measurement ran. Nothing in it changes afterwards.

## 0. Summary

*Filled in with the results (§5).*

## 1. Background and goals

#47 proposes computing the *impedance region* (where an object's footprint meets the walkable surface) from
depth x sidewalk segmentation x detection, instead of inferring it from a click and a size formula. Its first step
is the cheapest: GSV's depth payload is Google's plane model of the scene (`depth.py`: a list of planes plus one
plane index per pixel of a 512x256 grid), and it models the **surface** while omitting objects. A curb ramp is
part of the surface, not an object on it, so for this class depth alone may already be enough. The
sidewalk-panorama-tools notes on the same payload say two things that bear on that: "no plane" (index 0) means sky
or anything unmodelled, and curb ramps sit ~0.15 m above the modelled road surface rather than appearing as their
own geometry.

**Goals.** (G1) Measure, for every stored detection on a pano with a payload, what the depth model puts under its
pixel. (G2) Measure how far a ramp's plane sits above the road beside it. (G3) Measure the horizontal range the
plane gives, against the flat-ground raycast. (G4) Against RampNet ground truth: does depth carry the surface for
real ramps, and does it separate false positives for free?

## 2. Research questions

| | Question | Answered by |
|---|---|---|
| RQ1 | What plane is under a detection — the dominant ground, another floor plane, a wall, an overhang, or nothing? How often nothing? | plane class per detection, by city, tier, vintage, and range (§5.1) |
| RQ2 | When the plane is a secondary floor plane, how high above the local road is it? Is it ~0.15 m? | `offset_local` of `floor` detections (§5.2) |
| RQ3 | Is the plane's range plausible, and how does it compare with the flat raycast at 2.6 m and at the depth-measured height? | `ratio_flat`, `ratio_pp` (§5.3) |
| RQ4 | Do verdict-True detections sit on the surface? Do verdict-False detections sit off it more often? | pre-registered rules (i) and (ii) (§5.4) |

## 3. Data

The four GSV runs whose depth was harvested (`scripts/harvest_depth.py`, #41) — bend, paterson, gainesville,
sao_paulo — read in place and not modified, and RampNet's benchmark bundles for the same cities
(`../RampNet/benchmark/<city>/{verdicts.json,records.jsonl}`, read as data). The per-city coverage table (panos
with a detection, with a payload, by payload status) is in §5.0 and `coverage.csv`. Not every pano has a payload:
gap-fill panos (#32) were added after the harvest, most of them in sao_paulo.

## 4. Methods

### 4.1 Unit and tiers

Every stored detection at or above `OPERATIONAL_CONFIDENCE` (0.30), reported at 0.30 and at
`BENCHMARK_CONFIDENCE` (0.55). bend predates the storage floor, so its stored detections are all ≥ 0.55 and its
two tiers coincide. The payload's status is `depth.classify_height`'s (`measured`, `synthetic_ground`,
`degenerate`, `no_ground`, `implausible`, `unparsed`, plus `no_file` when no payload was harvested). **Only
`measured` payloads enter the headline shares and both rules**; the others are reported apart, never pooled
with them.

### 4.2 Frames: image-frame lookup, exact ray

Detections are stored in the image frame (x from the left of the panorama JPEG), and a raw depth-index column is
an image column (#80). The plane lookup is `depth._plane_at(payload, x, y)` in that frame. `depth.depth_at` is
never used: it takes streetlevel's mirrored raster frame and snaps the ray to a pixel centre, worth up to 6.6% of
range near the horizon. Only the plane lookup is quantized, which is correct because the segmentation is per
pixel; the hit point is the exact ray `depth._direction_continuous(x, y)` intersected with that plane, and the
horizontal range is computed exactly as `depth.ground_range_at` computes it (the tests assert equality). Depth
frame: +z is down.

### 4.3 Plane class

| class | rule |
|---|---|
| `no_plane` | index 0 at the pixel |
| `ground` | the pixel's plane is the dominant ground plane (`gsv_ground_plane.dominant_ground`, cross-checked against `depth.ground_plane` on every payload) |
| `floor` | another plane meeting `ground_plane`'s candidate rule: tilt ≤ `GROUND_MAX_TILT_DEG` (18°), ≥ 90% of its pixels below the horizon |
| `horizontal_nonfloor` | tilt ≤ 18° but fails the below-horizon share (an overpass or awning underside) |
| `non_horizontal` | tilt > 18° (the tilt is reported; a wall reads ~90°) |

"No plane" is also reported in the 3x3 payload-pixel neighbourhood (rows clamped, columns wrapping the seam),
binned by flat range (0–5, 5–10, 10–15, 15–20, 20–25, ≥ 25 m, and above the raycast horizon), so a zero at the
pixel is measured rather than assumed and a detection on a plane boundary is visible.

### 4.4 Height above the local road (primary) and above the dominant plane (secondary)

For a `floor` detection, P is the exact ray's hit on its plane. The **local reference** is the first floor-candidate
plane *different* from the detection's own, met walking the detection's payload column down toward the nadir —
the road in front of the ramp. `offset_local = z_ref(P_x, P_y) − P_z`, positive = above the road (both planes are
floor planes, so each is the one below the camera: `n·X = d·sign(n_z)`). `offset_dominant` does the same against
the dominant ground plane and is reported for completeness only: extrapolating a plane with the typical 1.5° tilt
over 20 m moves it about 0.5 m, so it is not a height. Differences of perpendicular distance (`ground.d − plane.d`)
are not height offsets and are not used. Bins: < −0.30, [−0.30, −0.05), [−0.05, 0.05) ("same surface, different
segment"), **[0.05, 0.30) = the curb-height band**, ≥ 0.30 m.

A `floor` detection whose column walk finds no reference has no `offset_local`. It is outside the surface band and
counts as `non_surface` (a clarification made while implementing, before any GT-conditioned number was computed;
the plan's probe found 2 such of 2,478 floor hits).

### 4.5 Range instrument

For `ground` and `floor`, `ratio_flat = range_depth / range_flat(2.6 m)` (`geo.detection_ground_point`,
`apply_pose=False`, `geo.DEFAULT_CAMERA_HEIGHT_M`) and `ratio_pp` against the flat raycast at the pano's own
depth-measured height (the harvested `depth/index.csv` through `fuse_sites.load_depth_index`; these runs predate
the #40 block fields). `ground` has `ratio_pp` ≈ 1 by construction, so the two classes are quoted separately.
Ratios are taken over rays the production raycast places (≤ 25 m, below `MIN_DEPRESSION_RAD`); the rest keep
their depth range and are counted as `beyond_25m`. Class shares and ratio medians are also reported per
(city, capture year) where the cell has ≥ 300 detections.

### 4.6 The GT join

Benchmark tier only (every bundle was exported and reviewed at 0.55). Panos come from `eval_sites.judged_gt_panos`,
so the drift gate and skip rules — and so the denominators — are eval_sites' and mined_precision's. Groups:
verdict-True detections, verdict-False detections, and non-unsure missed marks (`missed[].x/.y` are normalized
image-frame coordinates, the frame `x_normalized` is in, and are classified with the same lookup). `unsure` and
`duplicate` verdicts and unsure marks are counted apart. Intervals are Wilson 95% (`eval_sites.wilson`); the median
offset's interval is the distribution-free binomial order-statistic one.

### 4.7 The pre-registered reading, operationalised

In `verdict()` (committed before the full run), on `measured` payloads at the benchmark tier:

- **(i) Depth carries the surface.** `surface` = `ground`, or `floor` with −0.30 ≤ `offset_local` < 0.30.
  **SUPPORTED** iff the pooled surface share of verdict-True detections is ≥ 0.85 and ≥ 0.80 in every city;
  **UNDERCUT** iff pooled < 0.70; otherwise **NOT SUPPORTED**. The 0.15 m claim is read descriptively: the median
  `offset_local` of True `floor` detections, with its CI, is "consistent" if it lies in [0.05, 0.30).
- **(ii) A free false-positive signal.** `non_surface` = everything else (`no_plane`, `non_horizontal`,
  `horizontal_nonfloor`, `floor` outside ±0.30 m or with no local reference). **SUPPORTED** iff the pooled
  non_surface share among verdict-False detections exceeds the True share by ≥ 20 points AND the False share's
  Wilson lower bound exceeds the True share's upper bound; otherwise **NOT SUPPORTED**. `underpowered: true` is
  printed when the False share's Wilson 95% interval is wider than the 20-point margin: a null is then weak
  evidence. With the pooled verdict-False count near 49, it will be.
- Reported without a gate: missed marks against True on the same classes (a depth-based miner for step 2 needs the
  surface to be there for the ramps the model misses, too).

## 5. Findings

*Filled in after the full run (commit 2).*

## 9. Reproducibility

Everything reads `runs/<city>/{results.jsonl,depth/}` (from `--run-root`, read-only) and
`../RampNet/benchmark/<city>/`, writes per-detection CSVs to `runs/<city>/depth_at_detection/` and the aggregates
to `runs/_summary/depth_at_detection/` (both gitignored), and figures to `docs/figures/depth-at-detection/`.
Stdlib on top of `depth.py`, `geo.py`, `fuse_sites.py`, `eval_sites.py`, `gsv_ground_plane.py`,
`mapillary_tilt.py`; `figures` needs matplotlib. No network, no GPU.

```bash
python scripts/depth_at_detection.py measure --limit 500   # smoke: first 500 panos per city, no summaries
python scripts/depth_at_detection.py measure               # all four cities (run first)
python scripts/depth_at_detection.py gt                    # the GT join; needs ../RampNet/benchmark
python scripts/depth_at_detection.py verdict               # the pre-registered reading -> verdict.json
python scripts/depth_at_detection.py figures               # copies the aggregates to docs/figures/.../data/, redraws
pytest tests/test_depth_at_detection.py tests/test_depth.py
```
