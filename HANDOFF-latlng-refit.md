# Handoff: the lat/lng refit, from the auto-labeler's side

**For:** the SidewalkWebpage / label-latlng-estimation Claude Code session.
**From:** the sidewalk-auto-labeler session, 2026-08-07.
**Why now:** the auto-labeler is about to submit **9,091 Richmond (Mapillary) records /
9,526 labels** to production. 13 of them are already live on richmond-test. Whether that
run happens before or after the #3 refit is a real sequencing decision, and this note
gives you the evidence to make it.

---

## 1. The question that gates our submission

**Does the #3 refit ship with a recompute of existing `label_point` rows, or only apply
to new labels?**

We read the code (`ExploreService.scala:744-757` on a `qa-develop` clone at #4661 — please
re-verify against current `develop`, ours is stale) and the answer looks like "stored at
insert":

```scala
val latLng = PanoDataService.toLatLng(pano.lat.get, pano.lng.get, pov.heading, pov.zoom,
                                      canvasX, canvasY, label.panoY, pano.height.get)
...
labelPoint = LabelPointSubmission(..., lat = Some(latLng._1), lng = Some(latLng._2),
                                  computationMethod = Some(ComputationMethod.Approximation2))
```

Two consequences we care about:

1. **It is persisted, not derived on read.** So labels submitted today keep today's
   estimate forever unless something recomputes them.
2. **It is not only a map pin.** Lines 748-749 use the estimated lat/lng to choose
   `streetEdgeId` (nearest street edge) and then `regionId`, both stored on the label. A
   biased estimate can attach a label to the wrong street or region, which feeds street
   completion and region aggregates — not just where the dot renders.

**The encouraging part:** `ComputationMethod` is already an enum (`Depth`,
`Approximation2`). That is exactly the hook a recompute needs — add `approximation3`, and
target the old rows. If the refit lands with that backfill, we will submit Richmond now and
let the recompute sweep it up. **If it does not, we would rather wait**, because 9,526
labels attached to possibly-wrong street edges is not something we want to hand-repair.

Please tell us which it is.

## 2. Evidence for #4765, measured on real AI submissions

We transcribed the zoom-1 estimator from `PanoDataService.scala:135-140` and ran it over
the 13 labels we actually submitted to richmond-test, against flat-ground geometry
(depression angle below horizon, 2.6 m camera height):

| pano | height | panoY | y_norm | PS est | geometry | error |
|---|---|---|---|---|---|---|
| 1731432574 | 5500 | 3469 | 0.6309 | 6.5 m | 6.0 m | +0.6 |
| 1731432574 | 5500 | 3297 | 0.5996 | 11.1 m | 8.0 m | +3.0 |
| 1731432574 | 5500 | 3211 | 0.5840 | 13.3 m | 9.6 m | +3.7 |
| 5835857246 | 5500 | 3136 | 0.5703 | 15.3 m | 11.6 m | +3.7 |
| 1731432574 | 5500 | 2964 | 0.5391 | 19.9 m | 21.1 m | −1.2 |
| 1009452203 | 2880 | 1507 | 0.5234 | 23.7 m | 35.2 m | −11.5 |

Over-long near, short far — the linear-vs-cotangent shape of #4766.

**The cleaner demonstration of #4765** holds the physical geometry fixed and varies only
the pano's native pixel height. Same ramp, 8 m away:

```
   pano height 2048 -> PS estimates 20.1 m        5500 -> 11.0 m
               2880 -> 17.9 m                     6656 ->  7.9 m   <-- correct
               4096 -> 14.7 m                     8192 ->  3.9 m
```

Because `(panoHeight / 2 - panoY)` is a **raw pixel count** against a fixed slope, the
estimator is only calibrated at the resolution it was fit on — and it comes out correct at
**6656**, i.e. GSV zoom-5 (13312×6656). That is #4765 reproduced from first principles, and
it independently dates the fit.

**Why this bites AI submissions harder than crowd labels.** Mapillary native heights vary
*within a single city* — our 4-record Richmond sample alone contains 2880 and 5500 — so the
error varies label to label rather than being a constant offset you could calibrate away.
Our GSV cities are more uniform but sit at a different height than 6656 as well.

Reproduction script (stdlib only, no PS code):
`sidewalk-auto-labeler` scratch — formula transcribed verbatim from
`PanoDataService.scala:135-140`, `canvasY = LabelPointTable.canvasHeight / 2 = 240`.

## 3. A wrinkle specific to AI labels: the canvas inputs are synthetic

`ExploreService.scala:742-743` sets `canvasX = canvasWidth / 2` and
`canvasY = canvasHeight / 2` for every AI label, because there is no human viewport. So:

- `estHeadingDiff` collapses to a **constant** `-13.5675945 + 0.0396061 × 360 = +0.69°`
  applied to every AI label (small — ~0.12 m at 10 m — but it is a systematic bias, not
  noise, and it is an artifact of faking a canvas).
- the `canvasY` term contributes a constant **+0.27 m** to every AI distance.

The estimator was parameterized around a human's viewport. When the refit re-derives its
inputs, it is worth deciding deliberately what those inputs mean for a labeler that has no
canvas at all — rather than inheriting the centre-of-canvas convention by default.

## 4. Cross-link: latlng#7 overlaps work that already exists and is validated

**latlng#7 — "Bearing-only triangulation: estimator for multiply-observed labels"** is
close to what `sidewalk-auto-labeler` already ships as `scripts/fuse_sites.py` (merged
labeler#35, issue #27 stages 2-3):

- `geo.py` — ENU tangent frame + ground raycast with closed-form anisotropic error from
  heatmap quantization, dropping rays beyond 25 m.
- `fuse_sites.py` — associates detections across panos into physical sites:
  descending-confidence greedy, same-pano cannot-link, chi-square gating,
  inverse-covariance refit, residual rejection.
- `eval_sites.py` — scored against RampNet GT in world space across **nine cities**:
  world precision 0.89–0.98, recall 0.93–0.96, versus own-view recall 0.72–0.83.

Two findings from that work worth having before you build #7:

1. **GSV equirects are already gravity-rectified.** A pose ablation over pitch/roll sign
   conventions showed applying the metadata pitch/roll makes within-site spread *worse*.
   Don't apply it. (`geo._world_ray` docstring has the numbers.)
2. **Camera height is per-pano, not constant.** streetlevel returns per-pano depth for free
   as a flag on the metadata call, and it shows heights of **1.11–2.50 m** tracking pano
   vintage. Our own `geo.py` hardcodes 2.6 m and therefore runs ~18% long on every range —
   filed as RampNet#101, and it is the same class of error as #4765. Your #9 depth
   validation is directly reusable for this, in both directions.

**latlng#8** (ray–street-edge intersection as a map prior) is likewise a variant of the
raycast in `geo.py`, and PS already uses nearest-street-edge snapping at
`ExploreService.scala:748` — so the prior is half-built on both sides.

Worth a conversation before either side reimplements the other's work.

## Claim provenance

- **Verified in code, on a stale `qa-develop` clone at #4661 — re-verify on `develop`:**
  everything in §1 and §3, plus the estimator formula in §2.
- **Computed by us, reproducible:** every number in §2's tables.
- **From the auto-labeler / RampNet side, ours to stand behind:** all of §4.
