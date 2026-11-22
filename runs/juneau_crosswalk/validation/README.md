# Juneau AI Crosswalk labels: validations on juneau-test

Jon validated the AI Crosswalk labels sent to juneau-test (issue #168, 3,579 labels,
`rampnet-crosswalk@ebad379b5f7c`, min confidence 0.23, range gate 5.625 deg) on 2026-11-17.
The test database is not backed up, so the votes are snapshotted here.

The labels come from `runs/juneau_crosswalk` (3,525) and `runs/juneau_expansion_crosswalk` (54),
both made by `scripts/detect_from_store.py --records`. Their manifests' `pixels.store` is the
scratch copy of the store the detector read on klone, which is purged storage; the durable
source of those pixels is the PS pano store (`juneau-ak`). Their `pixels.records.path` is the
klone checkout the runs started in, and a resume is bound to it.

| File | What |
|---|---|
| `juneau_test_votes_<date>.csv` | One row per validation: `label_uid` (`juneau-ak:<label_id>`; label ids are per city), pano and pixel, dip below the horizon, the label's stored lat/lng, confidence, Agree/Disagree/Unsure, validator `user_id`, time, and the validator's comment. |
| `random_sample_2026-11-17.json` | The seeded random draw for an unbiased precision check: 50 labels from the 2,717 an OSM-proximity rule would keep and 30 from the 862 it would drop. The rule is recorded in the file. It was a candidate only, and it is not adopted. |

**Who voted.** Every vote is Jon's, cast under two accounts. Nobody else validated these labels (checked
on the test DB, 2026-11-18).

| `validator_user_id` | Username on juneau-test | Votes in the CSV | When, in the CSV (PST, 2026-11-17) | Votes in the test DB on 2026-11-18 |
|---|---|---|---|---|
| `4ff63c52-88d6-40bb-ad08-ceff4e14f8bc` | `29vs30oz1c7daxjc` | 236 | 16:20–16:44 | 236 |
| `2d8335ef-daff-4382-afe2-7e1eb6567fd9` | `jonfroehlich` | 114 | 16:52–17:32:34 | 115, the last at 17:56 |

**The CSV is one vote short of the database.** The committed
`juneau_test_votes_2026-11-17.csv` was exported at 17:33 on 2026-11-17 and holds 350 votes
(236 + 114) on 348 labels. When the test DB was read on 2026-11-18 for this table, the
`jonfroehlich` account had 115 votes, the last at 17:56, so one vote was cast after the
export. In the CSV, `juneau-ak:7666` is the only label of the 80-label random sample with no
vote (79 of 80 voted); whether the later vote is on that label has not been checked. The vote
columns of `validator_r3_all_labels_scored.csv`, and so every vote count in
`validator_r3_REPORT.md`, are the CSV's.
**Open (Jon):** re-export the votes from the test DB, then re-score the sample.

The first account is an auto-generated anonymous one: Jon was logged out without noticing, and he
confirmed on 2026-11-18 that those votes are his. Treat both ids as Jon. Two labels have a vote
from each account, both Agree (`juneau-ak:6640`, `juneau-ak:8379`).

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
