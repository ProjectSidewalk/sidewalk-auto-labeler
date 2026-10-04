# vancouver: provenance gate (#56)

**Verdict: STOP** -- failing: Arm S (store usability), Arm Z (pipeline identity), precision (unclaimed detections).

## Rule (`pixel-96`: pre-registered; amended after the PR #96 review, before any Vancouver number)

- **Arm S, store usability:** a label matches when a stored detection >= 0.55 lies within +/-(W/1024 + 1) px in x and +/-(H/512 + 1) px in y of it (key `round(x*W), round(y*H)`, x wrapping at the seam). Passes iff >= 0.98 of joinable labels match. `exact_share` (+/-1 px) is reported, never gated.
- **Arm Z, pipeline identity (with --control):** the same join at +/-1 px on a zoom-3 control file (200 seeded labeled panos, seed 56, via `reinfer.py --ids`); passes iff >= 0.98.
- `--rule pixel-96` reproduces this report; the default, `--rule coarse-cell` (#111), writes the exploratory coarse-cell reading instead.
- **Coverage floor:** UNDETERMINED unless no selected pano is pending and joinable labels are >= 0.95 of the AI labels whose pano has a store JPEG.
- **Precision (P):** detections >= 0.55 on labeled panos that no label claims must be <= 0.02 x joinable labels, else STOP. Soft-deleted AI labels are a known source of such detections: a STOP here is a signal to investigate, not a verdict on the store.
- PASS only if every arm that runs passes; UNDETERMINED takes precedence over STOP. STOP means nothing downstream runs.

| check | value | rule | result |
|---|---:|---|---|
| Arm S share | 0.8252 | >= 0.98 | fail |
| exact_share (+/-1 px, not gated) | 0.7385 | -- | -- |
| Arm Z share | 0.9632 | >= 0.98 | fail |
| coverage (joinable / labels with a JPEG) | 1.0000 | >= 0.95 | pass |
| pending selected panos | 0 | 0 | pass |
| unclaimed tier detections / joinable | 0.1316 | <= 0.02 | fail |

## Inputs

- Labels: `raw_labels.geojson`: 64,847 features, sha256 `57c31c73dc75a6139b2694fdc5e7c0823bf0d0504348f1e1f38d3681b713098d`, fetched 2026-09-28T23:01:58+00:00 from https://sidewalk-vancouver.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson
- AI account: `51b0b927-3c8a-45b2-93de-bd878d1e5cf4` (64,814 CurbRamp labels); every account: `51b0b927-3c8a-45b2-93de-bd878d1e5cf4` 64,814, `c6030d8f-9163-498b-a102-d147a27b8c44` 7, `f34410b6-90d2-4176-a590-f371b75ab4c5` 6, `de9a2eb6-52e3-4854-b488-b970c5c1c567` 5, `aab0b9c1-bffc-4884-9c76-365762503a95` 5, `fb9b61d7-c613-454e-82de-63722f6baa2a` 5, `71a31933-61d9-4474-b0d0-1d41f7aba0ac` 2, `964eb6f2-da36-4a4f-bb95-aa99ff6eae6f` 2, `0ff4a61a-8f80-4ef9-a5f5-1c85d8917e0d` 1
- Run: `runs/vancouver/results.jsonl`, 28,830 panos, sha256 `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28`
- Control: `runs/vancouver/control_zoom3.jsonl`, sha256 `748734afa2a2057e3eeb678f973e3787981addd13d7b7a3b22ccdac63caba2f7`
- Control panos: 141 of the 200 drawn panos are in the control file; 59 absent (reinfer.py wrote no record, e.g. no longer served by id)
- Generated 2026-10-04T18:10:59+00:00

## Coverage

- Selected panos: 29,181; pending (neither processed nor cached-skipped): 0
- AI labels: 64,814; on a pano with a store JPEG: 64,006; joinable: 64,006
- Metadata 404 (`metadata_404_null_field_or_no_file`: a null pano_data field -- most often camera_pitch -- or no server-side file): 0 (0.00%) of selected panos (0 of them labeled), carrying 0 (0.00%) of the AI labels
- Store JPEGs whose native size differs from the server's width/height (manifest): 0

## Arm S join

- Joinable labels (pano in the run): 64,006
- Matched under Arm S: 52,819 (82.52%); of those exactly on the pixel: 47,270
- Within +/-1 px at >= 0.55 (exact_share): 47,270 (73.85%)
- Unmatched: 11,187; of those, a stored detection BELOW 0.55 sits within tolerance (a threshold flip): 3,502
- Tier detections claimed by labels at two or more different pixels (the join is nearest-within-tolerance, not one-to-one): 0
- Labels whose pano width/height differ from the run's: 0
- Duplicate label keys (same pano and pixel; PS is insert-only, and the join is many-to-one): 1,357 key(s) carrying 2,714 labels

| panos with AI labels | count |
|---|---:|
| all matched | 19,064 |
| partly matched | 6,402 |
| none matched | 3,064 |

### Matched labels: where the detection sits (heatmap cells, Chebyshev; #111)

| matched detection sits | labels | share of matched |
|---|---:|---:|
| same heatmap cell (0) | 47,270 | 0.8949 |
| grid neighbour (1: residue 3 <-> 4) | 5,549 | 0.1051 |
| off-grid shift (2-6) | 0 | 0.0000 |
| adjacent-coarse-cell flip (7-8) | 0 | 0.0000 |
| further (> 8) | 0 | 0.0000 |

A flip (7-8 cells) is the same ramp decoded at the neighbouring coarse cell of a near-tied pair; residue 3 <-> 4 (1 cell) is the same coarse cell.

### Unmatched labels: nearest stored detection (any confidence)

| distance | labels |
|---|---:|
| <= 2 px | 2,982 |
| <= 4 px | 0 |
| <= 1 heatmap cell | 495 |
| <= 2 heatmap cells | 27 |
| <= 1 coarse cell (8 heatmap cells) | 6,893 |
| <= 2 coarse cells | 644 |
| > 2 coarse cells | 144 |
| no detection on the pano | 2 |

A heatmap cell is W/1024 px (16 px on a 16384-wide pano).

## Detections >= 0.55 that no label claims (under Arm S)

- on panos carrying AI labels (gated by P): 8,424
- on panos carrying none (e.g. the sampled empty stratum; not gated): 1

## Arm Z (control)

- Panos: 141 of the 200 drawn panos are in the control file; 59 absent (reinfer.py wrote no record, e.g. no longer served by id). The Arm Z share is over the panos present only, so it is conditioned on whatever removed the absent ones.
- Joinable labels on the control's panos: 299; matched under Arm Z: 288 (96.32%); within +/-1 px: 288 (96.32%)
- Unmatched: 11; threshold flips within tolerance: 1

| matched detection sits | labels | share of matched |
|---|---:|---:|
| same heatmap cell (0) | 288 | 1.0000 |
| grid neighbour (1: residue 3 <-> 4) | 0 | 0.0000 |
| off-grid shift (2-6) | 0 | 0.0000 |
| adjacent-coarse-cell flip (7-8) | 0 | 0.0000 |
| further (> 8) | 0 | 0.0000 |

| on the control's panos | labels |
|---|---:|
| both | 247 |
| S only | 7 |
| Z only | 41 |
| neither | 4 |

`S only`: matched under Arm S on the store run but not under Arm Z on the control; `Z only` the reverse.

## AI labels whose pano is not in the run

| why | labels |
|---|---:|
| skipped: jpg_missing | 808 |
