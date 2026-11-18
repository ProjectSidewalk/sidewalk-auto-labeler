# Juneau AI Crosswalk labels: validations on juneau-test

Jon validated the AI Crosswalk labels sent to juneau-test (issue #168, 3,579 labels,
`rampnet-crosswalk@ebad379b5f7c`, min confidence 0.23, range gate 5.625 deg) on 2026-11-17.
The test database is not backed up, so the votes are snapshotted here.

| File | What |
|---|---|
| `juneau_test_votes_<date>.csv` | One row per validation: `label_uid` (`juneau-ak:<label_id>`; label ids are per city), pano and pixel, dip below the horizon, the label's stored lat/lng, confidence, Agree/Disagree/Unsure, validator `user_id`, time, and the validator's comment. |
| `random_sample_2026-11-17.json` | The seeded random draw for an unbiased precision check: 50 labels from the 2,717 an OSM-proximity rule would keep and 30 from the 862 it would drop. The rule is recorded in the file. It was a candidate only, and it is not adopted. |

The first ~236 votes were **not** a random sample: they came from the validation queue and
the gallery in geographic clusters. Use them to describe failure modes, not to estimate
city-wide precision. The comments name the false-positive classes: own capture-car roof,
brick/paver sidewalk, fence/tree shadows, stairs, turn-lane markings, sidewalks across
driveways, and parking-lot stripes.

## AI validator scores (`validator_r3_*`)

The production Crosswalk validator (sidewalk-validator-ai round 3, hub
`projectsidewalk/sidewalk-validator-ai-crosswalk@b8498b82eb`, plus the 2025-08 model as a
secondary) scored all 3,579 labels offline on klone, 2026-11-17. The run used
`coord_frame "rig"` crops cut from the PS pano store. The results are in
ProjectSidewalk/sidewalk-validator-ai#114 and `validator_r3_REPORT.md`.

| File | What |
|---|---|
| `validator_r3_labels.csv` | The validator's input rows. Its sha256 is `labels_sha256` in `validator_r3_crops_run.json`. |
| `validator_r3_manifest.csv` | Crop manifest (status, depth, crop size), one row per label. |
| `validator_r3_predictions_r3.csv` | Round-3 predictions. |
| `validator_r3_predictions_2025.csv` | 2025-08 model predictions. |
| `validator_r3_predictions_*_provenance.json` | Model revisions and manifest sha256. |
| `validator_r3_all_labels_scored.csv` | Per-label join of scores, votes and metadata. |

They are copied here because the crops and scratch copies live on klone's scrubbed storage
and in a session scratchpad, both of which are purged.

## Regenerating the snapshot

Read-only on makelab1 (test Postgres, port 6432), then the export:

```sql
-- labels
SELECT l.label_id, l.pano_id, lp.lat, lp.lng, lp.pano_x, lp.pano_y, pd.width, pd.height,
       to_jsonb(i)->>'confidence'
FROM sidewalk_juneau_ak.label l
JOIN sidewalk_juneau_ak.label_point lp USING (label_id)
JOIN sidewalk_juneau_ak.pano_data pd ON pd.pano_id = l.pano_id
JOIN sidewalk_juneau_ak.label_ai_info i USING (label_id)
WHERE l.label_type = 'Crosswalk' AND l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4';
-- validations: label_validation (validation_result, user_id, end_timestamp) joined on the same labels
-- comments:    validation_task_comment (label_id, comment) joined on the same labels
```
