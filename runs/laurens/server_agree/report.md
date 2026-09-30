# laurens: server-side read of the AI's curb-ramp labels

Tool: `scripts/server_agree_check.py`. Every number is relative to this pull.

## Snapshot

| file | url | fetched (UTC) | sha256 | features |
|---|---|---|---|---:|
| `raw_labels_CurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:07:17+00:00 | `e2b9df2f53f8eb9ece33b3cdf92f1f9c39e98469c8f400c99c524948437179cd` | 1809 |
| `raw_labels_NoCurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson | 2026-09-30T15:07:17+00:00 | `58414153b9e199e195e3525286bf915def3aff7858dda52d666edc8ba5b304fd` | 57 |
| `label_clusters_CurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:07:18+00:00 | `daec411c1768e9457c1d3aca630eecd437aaa25f071a0ac0691c051bdbb72cce` | 671 |
| `streets.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson | 2026-09-30T15:07:18+00:00 | `3306464dc453d9eff138bd120ae721e6e0bd1105e3555ef93060eaa9bbd61599` | 169 |

- CurbRamp labels: 1809 (1575 by the AI `51b0b927…`, 234 by 2 human users); pano_source: {'mapillary': 1809}; dropped for a null/corrupt position: 0 CurbRamp, 0 NoCurbRamp, 0 clusters.

## 1. Human validations of the AI's CurbRamp labels

Per label, the majority of human Agree vs Disagree votes; PS's own AI validator is excluded, so this is people judging the model, not a model judging a model. Precision = agreed / (agreed + disagreed).

| AI labels | human-validated | agreed | disagreed | tie / unsure only | precision [95% Wilson] |
|---:|---:|---:|---:|---:|---:|
| 1575 | 1112 | 1006 | 94 | 12 | 0.915 [0.897, 0.930] |

- Human votes per AI label: 0 votes 463, 1 vote 1112, 2 votes 0, 3+ votes 0.
- AI-validator votes on AI labels: 0.
- Human validators: 2. Top: `549187e0…` 1111, `f813a0b6…` 1. **When one account cast most of the votes, this is one rater's precision read, not a crowd's.**

## 2. Human `549187e0…` vs the AI

- Streets: 169 in the city; 84 carry a label of the human's; **44 completed** (audit_count ≥ 1 and the human among its users). The audit is treated as incomplete unless those two numbers are the whole city.
- The human's labels: **230 CurbRamp**, 54 NoCurbRamp. AI CurbRamp labels: 1575, in 639 server clusters (127 of which also hold a label of the human's).

### 2a. Human → AI: share of the human's CurbRamp labels the AI also has

| test | labels | share [95% Wilson] |
|---|---:|---:|
| in a server cluster that also holds an AI label | 166/230 | 0.722 [0.661, 0.776] |
| … or an AI cluster within 5 m | 191/230 | 0.830 [0.777, 0.873] |
| … or an AI cluster within 7.5 m | 210/230 | 0.913 [0.870, 0.943] |
| … or an AI cluster within 10 m | 224/230 | 0.974 [0.944, 0.988] |

**6 of the human's ramps have no AI cluster within 10 m** (candidate AI misses; `misses.csv`): `1408`, `2727`, `2728`, `2774`, `2775`, `3011`.

Pano frame, for reference: 178 of 230 of the human's labels are on a pano the AI labelled at all; in 92 of those 178 an AI label sits within RampNet's match radius (0.022 × 1024) in that same pano. The gap to the world-frame rows is the AI finding the ramp from a different pano.

### 2b. AI → human: AI clusters on the streets the human COMPLETED

224 clusters holding an AI label sit on the 44 completed streets. A cluster is confirmed when it holds the human's label or the human has a CurbRamp label within r. **This is a lower bound on coverage, not a false-positive rate** (see 2c): an audit is one pass from one pano, and a corner ramp is attributed to one street edge but may be labelled from another.

| test | clusters | share [95% Wilson] | human NoCurbRamp within r instead |
|---|---:|---:|---:|
| holds a label of the human's | 71/224 | 0.317 [0.260, 0.381] | |
| … or human CurbRamp within 5 m | 88/224 | 0.393 [0.331, 0.458] | 9 |
| … or human CurbRamp within 7.5 m | 132/224 | 0.589 [0.524, 0.652] | 4 |
| … or human CurbRamp within 10 m | 154/224 | 0.688 [0.624, 0.745] | 8 |

By AI views in the cluster (confirmed = holds the label or human CurbRamp within 7.5 m):

| AI views | clusters | confirmed | share [95% Wilson] |
|---|---:|---:|---:|
| 1 | 78 | 30 | 0.385 [0.284, 0.496] |
| 2 | 79 | 46 | 0.582 [0.472, 0.685] |
| 3+ | 67 | 56 | 0.836 [0.729, 0.906] |

### 2c. The human's own validation vote × whether the human labelled nearby

Over the 537 AI labels on completed streets (label level, so a ramp seen from k panos counts k times). An Agree vote with no label of the human's within 7.5 m is a ramp the human accepted when validating and did not place when auditing: a coverage or attribution gap, not a false positive.

| own vote | human CurbRamp within 7.5 m | none within 7.5 m |
|---|---:|---:|
| agree | 228 | 105 |
| disagree | 3 | 25 |
| none | 117 | 59 |

