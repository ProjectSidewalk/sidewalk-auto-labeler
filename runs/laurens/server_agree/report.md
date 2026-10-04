# laurens: server-side read of the AI's curb-ramp labels

Tool: `scripts/server_agree_check.py`. Every number is relative to this pull.

## Snapshot

| file | url | fetched (UTC) | sha256 | features |
|---|---|---|---|---:|
| `raw_labels_CurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:07:17+00:00 | `7e0104b641c813b7a98f4f2dec72fd2742d67a26f994022a60db92934f0478ea` | 1809 |
| `raw_labels_NoCurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson | 2026-09-30T15:07:17+00:00 | `9e78f67aec67e8d24ec73e7be2766a6b894e544a520345dca5e392eb3310c1d1` | 57 |
| `label_clusters_CurbRamp.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:07:18+00:00 | `2ddfe95757590b431aaade75d0c391813266f750628c18e42c4734e998623a83` | 671 |
| `streets.geojson` | https://sidewalk-laurens.cs.washington.edu/v3/api/streets?filetype=geojson | 2026-09-30T15:07:18+00:00 | `4c388312cdf3177ba9b093b996b31abb16cdcddf7d1ce8e137a3ab6c39da2993` | 169 |
- `raw_labels_CurbRamp.geojson` is a redacted copy (human user ids cut to 8 characters; no number depends on the full id); as served, sha256 `e2b9df2f53f8eb9ece33b3cdf92f1f9c39e98469c8f400c99c524948437179cd`.
- `raw_labels_NoCurbRamp.geojson` is a redacted copy (human user ids cut to 8 characters; no number depends on the full id); as served, sha256 `58414153b9e199e195e3525286bf915def3aff7858dda52d666edc8ba5b304fd`.
- `label_clusters_CurbRamp.geojson` is a redacted copy (human user ids cut to 8 characters; no number depends on the full id); as served, sha256 `daec411c1768e9457c1d3aca630eecd437aaa25f071a0ac0691c051bdbb72cce`.
- `streets.geojson` is a redacted copy (human user ids cut to 8 characters; no number depends on the full id); as served, sha256 `3306464dc453d9eff138bd120ae721e6e0bd1105e3555ef93060eaa9bbd61599`.

- CurbRamp labels: 1809 (1575 by the AI `51b0b927…`, 234 by 2 human users); pano_source: {'mapillary': 1809}; dropped for a null/corrupt position: 0 CurbRamp, 0 NoCurbRamp, 0 clusters.

## 1. Human validations of the AI's CurbRamp labels

Per label, the majority of human Agree vs Disagree votes; PS's own AI validator is excluded, so this is people judging the model, not a model judging a model. Precision = agreed / (agreed + disagreed). **Not a random sample:** labels reach validators through PS's validation queue, so this is the precision of the labels that were shown, and it speaks for the rest only as far as the queue is representative of them.

| AI labels | human-validated | agreed | disagreed | tie / unsure only | precision [95% Wilson] |
|---:|---:|---:|---:|---:|---:|
| 1575 | 1112 | 1006 | 94 | 12 | 0.915 [0.897, 0.930] |

- Human votes on AI labels: 1112 in all. Per label: 0 votes 463, 1 vote 1112, 2 votes 0, 3+ votes 0.
- AI-validator votes on AI labels: 0.
- Human validators: 2. Top: `549187e0…` 1111, `f813a0b6…` 1. **When one account cast most of the votes, this is one rater's precision read, not a crowd's.**
- **Per cluster** (the label rate counts a ramp seen from k panos k times; a cluster's status is the majority of its AI labels' human statuses). The server leaves labels already marked incorrect out of its clustering: 84 AI labels sit outside every cluster, 84 of them disagreed. So its 639 clusters as served read high: agreed 516, disagreed 10, precision 516/526 = 0.981 [0.965, 0.990]. **With those labels put back** (7.5 m single linkage to any AI label; 650 groups): agreed 491, disagreed 53, neither 106, precision **491/544 = 0.903 [0.875, 0.925]**.

### 1b. By confidence tier

Each AI label joined to the run's stored detection on (pano_id, pano_x, pano_y), the pixel rounding send_to_ps.py used; `ambiguous` = the key carries two tiers across the files given. Run files: `results.raw.jsonl` (sha256 `a7da290de5b0…`).

| tier | AI labels | agreed | disagreed | precision [95% Wilson] |
|---|---:|---:|---:|---:|
| >= 0.55 | 708 | 560 | 15 | 560/575 = 0.974 [0.957, 0.984] |
| 0.30-0.55 | 867 | 446 | 79 | 446/525 = 0.850 [0.816, 0.878] |

## 2. Human `549187e0…` vs the AI

Auditor chosen by rule (`--human auto`): the human user with the most CurbRamp labels in this snapshot, `549187e0…`, with 230 of the 234 human CurbRamp labels (next: `18b26a38…` 4). The rule re-derives the same account from the pull, so no full id is recorded.

- Streets: 169 in the city; 84 carry a label of the human's; **44 completed** (audit_count ≥ 1 and the human among its users). The audit is treated as incomplete unless those two numbers are the whole city.
- The human's labels: **230 CurbRamp**, 54 NoCurbRamp. AI CurbRamp labels: 1575, in 639 server clusters (127 of which also hold a label of the human's).

### 2a. Human → AI: share of the human's CurbRamp labels the AI also has

Three matchers, because they disagree and the truth sits between the two one-to-one rows. **Headline: one-to-one vs AI clusters** (agree_rate's matcher: greedy by distance, d ≤ r, a cluster credits at most one label). It is strict where the server's 7.5 m single linkage merged two ramps at a corner into one cluster. One-to-one vs raw AI labels is lenient the other way, since a ramp has several AI views. `any` lets one cluster credit several labels (paired corner ramps), so it is coverage, not agreement. **Chance** = the same matcher after every human label is moved 25 m in a random direction (seed 31, as agree_rate.chance_floor): what street-bound density alone would match.

| reading | 5 m | 7.5 m | 10 m |
|---|---:|---:|---:|
| **one-to-one vs AI clusters** (headline) | 134/230 = 0.583 [0.518, 0.644] | 157/230 = 0.683 [0.620, 0.739] | 179/230 = 0.778 [0.720, 0.827] |
| one-to-one vs raw AI labels | 159/230 = 0.691 [0.629, 0.747] | 195/230 = 0.848 [0.796, 0.889] | 209/230 = 0.909 [0.864, 0.940] |
| any AI cluster (holds the label, or within r) | 191/230 = 0.830 [0.777, 0.873] | 210/230 = 0.913 [0.870, 0.943] | 224/230 = 0.974 [0.944, 0.988] |
| chance: one-to-one vs AI clusters | 30/230 = 0.130 | 57/230 = 0.248 | 75/230 = 0.326 |
| chance: one-to-one vs raw AI labels | 40/230 = 0.174 | 68/230 = 0.296 | 87/230 = 0.378 |
| chance: any AI cluster (holds the label, or within r) | 34/230 = 0.148 | 65/230 = 0.283 | 92/230 = 0.400 |

- Chance at 7.5 m over seeds 1-20: one-to-one vs AI clusters 50-66 of 230 (0.22-0.29); one-to-one vs raw AI labels 60-77 of 230 (0.26-0.33); any AI cluster (holds the label, or within r) 55-75 of 230 (0.24-0.33).
- In a server cluster that also holds an AI label (no radius): 166/230 = 0.722 [0.661, 0.776].

**6 of the human's CurbRamp labels have no AI cluster within 10 m** (candidate AI misses; `misses.csv`; two labels can be one location): `laurens:1408`, `laurens:2727`, `laurens:2728`, `laurens:2774`, `laurens:2775`, `laurens:3011`.

Pano frame, for reference: 178 of 230 of the human's labels are on a pano the AI labelled at all; 90 of those 178 match an AI label on that same pano one-to-one, strictly within RampNet's match radius (0.022 of the pano width; agree_rate's matcher), and 92 have any AI label within it. The gap to the world-frame rows is the AI finding the ramp from a different pano.

### 2b. AI → human: AI clusters on the streets the human COMPLETED

224 clusters holding an AI label sit on the 44 completed streets. A cluster is confirmed when it holds the human's label or the human has a CurbRamp label within r. **This is a lower bound on coverage, not a false-positive rate** (see 2c): an audit is one pass from one pano, and a corner ramp is attributed to one street edge but may be labelled from another.

| test | clusters | share [95% Wilson] | human NoCurbRamp within r instead |
|---|---:|---:|---:|
| holds a label of the human's | 71/224 | 0.317 [0.260, 0.381] | |
| … or human CurbRamp within 5 m | 88/224 | 0.393 [0.331, 0.458] | 9 |
| … or human CurbRamp within 7.5 m | 132/224 | 0.589 [0.524, 0.652] | 4 |
| … or human CurbRamp within 10 m | 154/224 | 0.688 [0.624, 0.745] | 8 |

By AI views in the cluster (AI labels only; confirmed = holds the label or human CurbRamp within 7.5 m):

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

