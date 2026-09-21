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

This module imports no torch so main.py, send_to_ps.py, and scripts/ can read the
contract without pulling in the model stack (the test suite runs torch-free); the
detector itself lives in detectors.curb_ramp.
"""

DETECTION_STORAGE_FLOOR = 0.1
MAX_PEAKS_PER_PANO = 50
OPERATIONAL_CONFIDENCE = 0.30
BENCHMARK_CONFIDENCE = 0.55
