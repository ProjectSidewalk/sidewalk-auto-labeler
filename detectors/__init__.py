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
  (send_to_ps.py --min-confidence), benchmark export (scripts/export_benchmark.py),
  and any analysis that treats a detection as a believed ramp. A record whose stored
  peaks all fall below it is still a valid "checked, nothing found" pano.

This module imports no torch so main.py, send_to_ps.py, and scripts/ can read the
contract without pulling in the model stack (the test suite runs torch-free); the
detector itself lives in detectors.curb_ramp.
"""

DETECTION_STORAGE_FLOOR = 0.1
MAX_PEAKS_PER_PANO = 50
OPERATIONAL_CONFIDENCE = 0.55
