POST HOC diagnostic (not pre-registered); see scripts/step3_exclusion_diag.py.
Every flat `tp` at <= 15 m, by the `peak_flat` rule's status.

| imagery | status | n | in-window op. detection judged true | ... within 10 cells (suppressed by construction) | op. to missed mark, deg (min-max) | stored sub-threshold peak near the missed mark |
|---|---|--:|--:|--:|---|--:|
| GSV | operational_in_window | 15 | 15 of 15 | 10 | 2.1-6.9 | 0 |
| GSV | no_peak | 7 | 0 of 0 | 0 | - | 2 |
| GSV | emit | 1 | 0 of 0 | 0 | - | 1 |
| Mapillary | no_peak | 3 | 0 of 0 | 0 | - | 1 |
| Mapillary | emit | 10 | 0 of 0 | 0 | - | 10 |

Zoom-3 re-inference over the 20 distinct target panos of these 23 candidates: 20 reproduce their >= 0.55 detections and 20 their full stored peak set (floor 0.1, one-cell tolerance).

| status | n | nearest local max to the missed mark with NO suppression (min_distance=1) is a stored peak | ... and is the in-window op. detection | heatmap max within 2 cells of the mark: median; n >= 0.55 |
|---|--:|--:|--:|---|
| operational_in_window | 15 | 15 | 15 | 0.774; 13 |
| no_peak | 7 | 7 | 0 | 0.231; 1 |
| emit | 1 | 1 | 0 | 0.150; 0 |

Provenance: {"imagery": "zoom 3 via panorama.fetch_panorama (the production path)", "model": {"api_version": "1.0.0", "model_id": "rampnet-model@606a11956743", "model_repo": "projectsidewalk/rampnet-model", "model_revision": "606a11956743f7eb328d9207769034752f6191f4", "model_training_date": "08-21-2025"}, "torch": "2.12.1+cu126", "transformers": "5.12.1"}
