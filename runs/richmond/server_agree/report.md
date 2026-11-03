# richmond: server-side read of the AI's curb-ramp labels

Tool: `scripts/server_agree_check.py`. Every number is relative to this pull.

## Snapshot

| file | url | fetched (UTC) | sha256 | features |
|---|---|---|---|---:|
| `raw_labels_CurbRamp.geojson` | https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:07:56+00:00 | `d737de7cded675a5a8e93db7d2919c8d4dd1e890bf7012b5307ec50670ecba66` | 13079 |
| `raw_labels_NoCurbRamp.geojson` | https://sidewalk-richmond.cs.washington.edu/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson | 2026-09-30T15:07:56+00:00 | `fc9fb0df005d6b9f804636bbf92f236f9b30420616ce946ff2fdb783841bfd07` | 6 |
| `label_clusters_CurbRamp.geojson` | https://sidewalk-richmond.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:07:57+00:00 | `fd35126dcf14b8aab9d9ada477fda8cca51151296caf14bbb5299d6f464a8978` | 3026 |
| `streets.geojson` | https://sidewalk-richmond.cs.washington.edu/v3/api/streets?filetype=geojson | 2026-09-30T15:07:58+00:00 | `9d193a7433c5ec457cdfc499fa41580c36f215928f6f50699f362112de451260` | 16365 |

- CurbRamp labels: 13079 (12962 by the AI `51b0b927…`, 117 by 5 human users); pano_source: {'mapillary': 13079}; dropped for a null/corrupt position: 0 CurbRamp, 0 NoCurbRamp, 0 clusters.

## 1. Human validations of the AI's CurbRamp labels

Per label, the majority of human Agree vs Disagree votes; PS's own AI validator is excluded, so this is people judging the model, not a model judging a model. Precision = agreed / (agreed + disagreed). **Not a random sample:** labels reach validators through PS's validation queue, so this is the precision of the labels that were shown, and it speaks for the rest only as far as the queue is representative of them.

| AI labels | human-validated | agreed | disagreed | tie / unsure only | precision [95% Wilson] |
|---:|---:|---:|---:|---:|---:|
| 12962 | 1069 | 1004 | 49 | 16 | 0.953 [0.939, 0.965] |

- Human votes on AI labels: 1069 in all. Per label: 0 votes 11893, 1 vote 1069, 2 votes 0, 3+ votes 0.
- AI-validator votes on AI labels: 0.
- Human validators: 3. Top: `549187e0…` 999, `fa28a62f…` 65, `51c7df6e…` 5. **When one account cast most of the votes, this is one rater's precision read, not a crowd's.**
- **Per cluster** (the label rate counts a ramp seen from k panos k times; a cluster's status is the majority of its AI labels' human statuses). The server leaves labels already marked incorrect out of its clustering: 49 AI labels sit outside every cluster, 49 of them disagreed. So its 3026 clusters as served read high: agreed 775, disagreed 0, precision 775/775 = 1.000 [0.995, 1.000]. **With those labels put back** (each joined to any AI label within 7.5 m, a single-linkage approximation of the server's 7.5 m complete linkage; 2988 groups): agreed 763, disagreed 34, neither 2191, precision **763/797 = 0.957 [0.941, 0.969]**.

### 1b. By confidence tier

Each AI label joined to the run's stored detection on (pano_id, pano_x, pano_y), the pixel rounding send_to_ps.py used; `ambiguous` = the key carries two tiers across the files given. Run files: `results.band.jsonl` (sha256 `04e7325eb07c…`), `results.posfix3seq.raw.jsonl` (sha256 `e7eee73bd402…`).

| tier | AI labels | agreed | disagreed | precision [95% Wilson] |
|---|---:|---:|---:|---:|
| >= 0.55 | 9526 | 920 | 21 | 920/941 = 0.978 [0.966, 0.985] |
| 0.30-0.55 | 3436 | 84 | 28 | 84/112 = 0.750 [0.662, 0.821] |

