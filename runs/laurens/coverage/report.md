# Post-submission coverage check: laurens

Server `https://sidewalk-laurens.cs.washington.edu` (records matched on `https://sidewalk-laurens.cs.washington.edu/ai/submitLabelsOnPano`), run `runs/laurens`, checked 2026-09-26T14:09:47+00:00 by `scripts/coverage_check.py` (issue #46). Read-only: GET requests only.

**Exit 0**: every expected pano with a live AI label is backed up.

## Server pulls

| file | url | fetched (UTC) | rows | sha256 |
|---|---|---|---:|---|
| `panos.json` | https://sidewalk-laurens.cs.washington.edu/adminapi/panos | 2026-09-26T14:09:15+00:00 (cached) | 4,564 | `e3ed37327de22cf1357a7c6f28880c45441489eedb2ade7d482cffcc47508a3e` |
| `labels_all.geojson` | https://sidewalk-laurens.cs.washington.edu/labels/all | 2026-09-26T14:09:15+00:00 (cached) | 2,847 | `1ccda445033acb6ec939bacd67bdf33966638de44996aa284c3cd6e98b2ec19b` |
| `raw_labels_CurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-26T14:09:16+00:00 (cached) | 1,790 | `9a4f6e3ca619538a112c5c58e321ae9297648d7e64ee489556371c9906d3e419` |

The pulls are cached beside this report and not tracked; `--refresh` re-pulls.

## Expected set (campaigns unioned)

Each campaign replayed through `send_to_ps.transform_record` at the range it sent; `rig_masked` absent in a record means the campaign predates the nadir mask.

| campaign | sent range | rig masked | lines | labels (record) | labels (replayed) | panos with labels |
|---|---|---|---:|---:|---:|---:|
| `results.raw.jsonl` | [0.55, ∞) | no | 4,495 | 1733 | 708 | 420 |
| `results.raw.jsonl (band 0.3-0.55)` | [0.3, 0.55) | no | 4,495 | 1025 | 1,025 | 723 |

Distinct expected panos: **875**.

## Classification

| status | panos | meaning |
|---|---:|---|
| covered | 741 | a live AI label on it reads `has_backup: true` |
| covered_metadata | 0 | `/backupImage/<id>/metadata` answered 200 |
| missing | 0 | metadata 404 with a non-null `camera_pitch` (decisive) |
| unconfirmed | 0 | flag false and not settled by the probe (e.g. 404 with a null pitch) |
| retired | 134 | no live AI label on the server: nothing renders, not a gap |

**741 of 741** expected panos with a live AI label are confirmed backed up.

## Server cross-checks (informational)

- Live AI labels of the checked type(s) on the server: 1,575 on 741 panos.
- Server AI panos NOT in the expected set: 0 (labels sent from files with no record here, or from another run).
- Live AI labels that rawLabels does not list (unjoinable, no pano id): 0.
- `/adminapi/panos`: 4,564 rows, 1,254 with `has_labels` (ever labelled, soft-deleted included); 379 of those are not expected (crowd-only, retired-only or other); 26 of them carry a live label of the checked type(s), and 1 of those read `has_backup: true`.
- Expected panos absent from the server's pano table: 0.

## Reading this

- `has_backup: true` is a cache of a past disk check by the server; a file removed later would still read true. `false` means unconfirmed (NULL is read as false), never confirmed absent.
- `/backupImage/<id>/metadata` also 404s when the pano row has no `camera_pitch`, so it can confirm only panos whose pose has been pushed.
- Human cross-checks: the admin health page's backup aggregate (admin session), and the panorama scraper's `pano_id_log.csv` (`downloaded=0` rows) on the server host.
