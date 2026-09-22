"""The detection-confidence contract shared across the pipeline (issue #27).

Two distinct thresholds govern detections, and they are deliberately not the same:

- DETECTION_STORAGE_FLOOR: results.jsonl stores every heatmap peak at or above this
  floor (capped at MAX_PEAKS_PER_PANO per pano, highest-intensity first) — including
  candidates well below the operational threshold. Sub-threshold evidence is the raw
  material for multi-view consensus fusion, and discarding it at extraction is
  irreversible: the only way to get it back is re-running inference over a whole
  city. Storing low costs only disk, and the per-pano cap bounds that wherever the
  heatmap's noise floor turns out to sit.

- OPERATIONAL_CONFIDENCE: the decision threshold (the pre-#27 ``threshold_abs``).
  Everything that ACTS on detections must filter here: PS submission
  (send_to_ps.py --min-confidence), production fusion (scripts/fuse_sites.py), and
  any analysis that treats a detection as a believed ramp. A record whose stored
  peaks all fall below it is still a valid "checked, nothing found" pano.
  0.30 since 2026-09-21 (issue #20), replacing 0.55: RampNet's threshold sweep
  (RampNet docs/operating_point.md, GT-completeness corrected, seven US splits pooled)
  gives P 0.919 / R 0.796 / F1 0.853 at 0.30 against 0.964 / 0.722 / 0.826 at 0.55 —
  +7.4 recall points for -4.5 precision, 1.86 -> 2.23 detections per pano; 27% of the
  extra "false positives" were ramps the GT had missed. Clovis is the binding split
  (corrected P 0.883). The policy is deliberately recall-first: a false positive is
  cheap to validate on the server, a false negative is never seen again.

- BENCHMARK_CONFIDENCE: the threshold every judged RampNet benchmark bundle was
  exported (scripts/export_benchmark.py) and reviewed at. Verdicts exist only for
  detections at or above it, so anything that JOINS a run to a bundle — the drift
  gate in scripts/eval_sites.py / scripts/mined_precision.py, the export strata, the
  tilt and thinning studies whose numbers are committed — filters here, never at the
  policy threshold above. The two are decoupled on purpose: moving the operating
  point must not silently re-key nine cities of ground truth. Changing THIS value
  means re-exporting and re-judging every bundle.

A third filter is geometric rather than confidence-based — see NADIR_MASK_DEG.

This module imports no torch so main.py, send_to_ps.py, and scripts/ can read the
contract without pulling in the model stack (the test suite runs torch-free); the
detector itself lives in detectors.curb_ramp.
"""

DETECTION_STORAGE_FLOOR = 0.1
MAX_PEAKS_PER_PANO = 50
OPERATIONAL_CONFIDENCE = 0.30
BENCHMARK_CONFIDENCE = 0.55

# --- The nadir (camera-rig) mask -----------------------------------------------------
#
# "Nadir" is the direction straight down from the camera (the opposite of zenith) — in 360
# photography, the bottom of the sphere, which is where the tripod or vehicle appears. It is
# the region panoramic rigs traditionally cover with a "nadir patch" or logo.
#
# In an equirectangular pano the vertical axis IS the dip angle: y_normalized 0.5 is the
# horizon and 1.0 is the nadir. Straight down is the vehicle the camera is mounted on, so a
# detection steep enough lands on the rig itself, never on the street.
#
# NOTE this rule uses only the pixel's position in the image. It says nothing about where
# the label sits on the street network, so a legitimate mid-block ramp — a school crossing,
# a mid-block crosswalk, a driveway cut — is untouched by it. That distinction matters: the
# same Laurens validations show the rig's false positives also sit far from intersections,
# but that is a CONSEQUENCE of the rig artifact (it lands wherever the car drove, while real
# ramps cluster at corners), not an independent signal. Measured 2026-09-22: after this mask,
# a ">30 m from an intersection" rule would have cut 3 validated-true labels for every 1
# false, so it is deliberately NOT implemented.
#
# Measured on Laurens prod, 2026-09-22, from human validations of live AI labels. Of the 158
# labels in this region, 50 have been judged and **every one is false** (0 true; 95% CI on
# precision [0.000, 0.071]). Counted from the database; a /v3/api/rawLabels pull the same day
# reported 48, for reasons never established (the feed itself was then measured to be
# real-time, agreeing with the database row-for-row during live validation), so prefer the
# database when a count has to be exact. They are not scattered: the 158 fall on 156 panos
# across 19 sequences at just seven discrete y values, every one of them GoPro Max — a roof
# rack, fixed in the rig's own frame, re-detected pano after pano. At 2.6 m camera height
# those dips are 1.5-2.1 m of ground range, i.e. on the vehicle.
#
# The separation is exact and has held as the judged set grew from 379 to 536 labels: the
# steepest validated TRUE label is at 46.1 deg, the shallowest validated FALSE one below the
# horizon-ish band is at 51.7 deg, and nothing lies between. 49 deg sits mid-gap. It is
# expressed as an ANGLE, not a range, on purpose: range needs a camera height, which is
# per-pano and currently a constant known to be too high (issue #40), whereas the dip is read
# straight off the pixel and cannot drift.
#
# This mask is NOT a general false-positive filter, and must not be sold as one. With it
# applied, the band's remaining errors are ordinary model mistakes — driveway cuts and the
# like — running at 0.828 precision (77/93 judged) against 0.972 for the >=0.55 tier.
#
# This mask was invisible at the old 0.55 operating point (0 of 708 Laurens labels, 2 of
# 9,526 Richmond) — dropping to 0.30 is what surfaced it.
NADIR_MASK_DEG = 49.0
NADIR_MASK_Y = 0.5 + NADIR_MASK_DEG / 180.0


def on_camera_rig(y_normalized: float) -> bool:
    """Is this detection steep enough to be on the camera vehicle rather than the street?

    The single definition of the rule, so the submission path and any analysis that adopts
    it cannot drift apart. Callers reconstructing a campaign that shipped BEFORE the mask
    existed must not apply it — what is already live is what is already live.
    """
    return y_normalized > NADIR_MASK_Y
