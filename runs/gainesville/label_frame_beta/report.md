# gainesville: label-frame beta (issue #113)

How far a human label's `pano_y` sits from the image's own frame, as a fraction of the rig tilt at the label's bearing. beta = 0: one frame; beta = 1: off by the whole tilt. **This is not a placement coefficient** (that is #116).

## Inputs

- labels: `raw_labels_CurbRamp.geojson`, 4498 features, fetched 2026-09-29T22:09:02+00:00 from https://sidewalk-gainesville.cs.washington.edu/v3/api/rawLabels?labelType=CurbRamp&filetype=geojson, sha256 `c0e5000b10b9341fff2ab9af400139506d694a10067922da21d1c519b306c691`
- run: `results.jsonl` sha256 `9f4a57f35d24715856d4464dbc0950febbf20de810ec9653432f339a5ab857e8`
- labels kept: 4498; dropped while loading: excluded_user 0, not_gsv 0, no_pixel 0
- panos with a pose and detections: 37435

## Headline (tier 0.55, window 3 x 6 deg)

| group | pairs | panos | beta (SE) | beta pitch (SE) | beta roll (SE) | intercept | abs T p50 / p90 / max | pairs at abs T >= 3 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| all | 1704 | 1096 | 0.936 (0.021) | 0.949 (0.031) | 0.929 (0.028) | -0.31 | 0.84 / 1.92 / 6.19 | 34 |
| human-agreed | 99 | 85 | 0.915 (0.083) | 0.919 (0.128) | 0.912 (0.108) | -0.15 | 0.79 / 1.71 / 3.40 | 3 |
| abs T >= 3 deg | 34 | 29 | 0.962 (0.051) | 0.886 (0.066) | 1.049 (0.081) | -0.37 | 3.33 / 4.42 / 6.19 | 34 |

Of 4498 labels: 1546 on a pano with no pose or detections, 0 whose stored pano size differs from the image's, 1248 with no detection in the window.

## Every tier and window (group `all`)

| tier | window (deg) | pairs | beta (SE) | beta pitch (SE) | beta roll (SE) |
|---|---:|---:|---:|---:|---:|
| 0.55 | 4 | 1683 | 0.901 (0.022) | 0.921 (0.034) | 0.891 (0.028) |
| 0.55 | 6 | 1704 | 0.936 (0.021) | 0.949 (0.031) | 0.929 (0.028) |
| 0.55 | 10 | 1712 | 0.951 (0.022) | 0.959 (0.033) | 0.947 (0.030) |
| 0.3 | 4 | 1968 | 0.891 (0.021) | 0.908 (0.032) | 0.882 (0.027) |
| 0.3 | 6 | 1994 | 0.918 (0.022) | 0.926 (0.031) | 0.913 (0.029) |
| 0.3 | 10 | 2004 | 0.934 (0.023) | 0.921 (0.037) | 0.941 (0.031) |

A wider window admits more wrong pairings (noise, not bias) and cuts off fewer large shifts, so beta rising with the window is truncation at the narrow end.

## Median offset by signed tilt (headline pairs, group `all`)

| T from | T to | n | median T | median (el_detection - el_human) |
|---:|---:|---:|---:|---:|
| -3 | -2 | 44 | -2.37 | -2.80 |
| -2 | -1 | 222 | -1.31 | -1.81 |
| -1 | -0.5 | 199 | -0.73 | -0.92 |
| -0.5 | 0.5 | 547 | 0.01 | -0.22 |
| 0.5 | 1 | 255 | 0.75 | 0.46 |
| 1 | 2 | 335 | 1.36 | 0.97 |
| 2 | 3 | 68 | 2.33 | 1.85 |
| 3 | 4 | 17 | 3.28 | 3.08 |

## Reading it

- Standard errors are clustered by pano: labels on one pano share its pose.
- The intercept is where a person clicks on a ramp against where the detector peaks. It is not a tilt effect.
- Detections are rig-masked, as production ships them.
- `pairs.csv` holds the headline pairs, keyed on `label_uid`.
