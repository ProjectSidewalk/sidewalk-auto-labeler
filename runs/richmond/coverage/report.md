# Post-submission coverage check: richmond

Server `https://sidewalk-richmond.cs.washington.edu` (records matched on `https://sidewalk-richmond.cs.washington.edu/ai/submitLabelsOnPano`), run `runs/richmond`, checked 2026-09-26T14:09:52+00:00 by `scripts/coverage_check.py` (issue #46). Read-only: GET requests only.

**Exit 0**: every expected pano with a live AI label is backed up.

## Server pulls

| file | url | fetched (UTC) | rows | sha256 |
|---|---|---|---:|---|
| `panos.json` | https://sidewalk-richmond.cs.washington.edu/adminapi/panos | 2026-09-26T14:09:49+00:00 | 9,282 | `c3cb0bfa75f4ed71a4d02c8fedc908c061c4500eba5ffafe626a9d1def911515` |
| `labels_all.geojson` | https://sidewalk-richmond.cs.washington.edu/labels/all | 2026-09-26T14:09:50+00:00 | 13,346 | `50878f975f4b7eade58ccfd2462a4ccb2557d59e4a4cd66d584c451b949452ab` |
| `raw_labels_CurbRamp.geojson` | https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-26T14:09:51+00:00 | 13,078 | `00e86369e3313a08a10b39f0ba9211c550182e0879fb96ed3e7a8c9fca362f05` |

The pulls are cached beside this report and not tracked; `--refresh` re-pulls.

## Expected set (campaigns unioned)

Each campaign replayed through `send_to_ps.transform_record` at the range it sent; `rig_masked` absent in a record means the campaign predates the nadir mask.

| campaign | sent range | rig masked | lines | labels (record) | labels (replayed) | panos with labels |
|---|---|---|---:|---:|---:|---:|
| `results.band.jsonl` | [0.55, ∞) | no | 9,091 | 12962 | 9,526 | 3,685 |
| `results.band.jsonl (band 0.3-0.55)` | [0.3, 0.55) | yes | 9,091 | 3436 | 3,436 | 2,593 |
| `results.jsonl` | [0.55, ∞) | no | 9,091 | 9526 | 9,526 | 3,685 |
| `results.posfix3seq.raw.jsonl` | [0.3, ∞) | yes | 72 | 143 | 143 | 51 |

Distinct expected panos: **4,613**.

## Classification

| status | panos | meaning |
|---|---:|---|
| covered | 4,613 | a live AI label on it reads `has_backup: true` |
| covered_metadata | 0 | `/backupImage/<id>/metadata` answered 200 |
| missing | 0 | metadata 404 with a non-null `camera_pitch` (decisive) |
| unconfirmed | 0 | flag false and not settled by the probe (e.g. 404 with a null pitch) |
| retired | 0 | no live AI label on the server: nothing renders, not a gap |

**4,613 of 4,613** expected panos with a live AI label are confirmed backed up.

## Server cross-checks (informational)

- Live AI labels of the checked type(s) on the server: 12,962 on 4,613 panos.
- Server AI panos NOT in the expected set: 0 (labels sent from files with no record here, or from another run).
- Live AI labels that rawLabels does not list (unjoinable, no pano id): 0.
- `/adminapi/panos`: 9,282 rows, 4,738 with `has_labels` (ever labelled, soft-deleted included); 125 of those are not expected (crowd-only, retired-only or other); 38 of them carry a live label of the checked type(s), and 28 of those read `has_backup: true`.
- Expected panos absent from the server's pano table: 0.

## Reading this

- `has_backup: true` is a cache of a past disk check by the server; a file removed later would still read true. `false` means unconfirmed (NULL is read as false), never confirmed absent.
- `/backupImage/<id>/metadata` also 404s when the pano row has no `camera_pitch`, so it can confirm only panos whose pose has been pushed.
- Human cross-checks: the admin health page's backup aggregate (admin session), and the panorama scraper's `pano_id_log.csv` (`downloaded=0` rows) on the server host.
