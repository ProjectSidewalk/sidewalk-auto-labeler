# vancouver: server-side read of the AI's curb-ramp labels

Tool: `scripts/server_agree_check.py`. Every number is relative to this pull.

## Snapshot

| file | url | fetched (UTC) | sha256 | features |
|---|---|---|---|---:|
| `raw_labels_CurbRamp.geojson` | https://sidewalk-vancouver.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:08:04+00:00 | `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d` | 64847 |
| `raw_labels_NoCurbRamp.geojson` | https://sidewalk-vancouver.cs.washington.edu/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson | 2026-09-30T15:08:05+00:00 | `e6646d1573614c092044035562b1cc58ced11f9548ad805c53a70d4b177a17b2` | 34 |
| `label_clusters_CurbRamp.geojson` | https://sidewalk-vancouver.cs.washington.edu/v3/api/labelClusters?labelType=CurbRamp&filetype=geojson | 2026-09-30T15:08:08+00:00 | `c58949536ef5b4b0e2e0471d5886e93cb856c6fdc030b6d57a8b4029547d46c9` | 18684 |
| `streets.geojson` | https://sidewalk-vancouver.cs.washington.edu/v3/api/streets?filetype=geojson | 2026-09-30T15:08:09+00:00 | `0d7ce79794a28146e811e2161c2019e2d96f3c8ec76d5b12b95183970d45f8b4` | 12567 |

- CurbRamp labels: 64847 (64814 by the AI `51b0b927…`, 33 by 8 human users); pano_source: {'gsv': 64847}; dropped for a null/corrupt position: 0 CurbRamp, 0 NoCurbRamp, 0 clusters.

## 1. Human validations of the AI's CurbRamp labels

Per label, the majority of human Agree vs Disagree votes; PS's own AI validator is excluded, so this is people judging the model, not a model judging a model. Precision = agreed / (agreed + disagreed).

| AI labels | human-validated | agreed | disagreed | tie / unsure only | precision [95% Wilson] |
|---:|---:|---:|---:|---:|---:|
| 64814 | 2911 | 2713 | 81 | 117 | 0.971 [0.964, 0.977] |

- Human votes per AI label: 0 votes 61903, 1 vote 2883, 2 votes 28, 3+ votes 0.
- AI-validator votes on AI labels: 0.
- Human validators: 19. Top: `549187e0…` 1572, `c9b7327d…` 500, `7ed1205b…` 320, `5ecf68e9…` 152, `81b72179…` 100. **When one account cast most of the votes, this is one rater's precision read, not a crowd's.**

