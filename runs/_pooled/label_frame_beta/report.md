# vouched pool (CurbRamp): label-frame beta (issue #113)

How far a human label's `pano_y` sits from the image's own frame, as a fraction of the rig tilt at the label's bearing. beta = 0: one frame; beta = 1: off by the whole tilt. **This is not a placement coefficient** (that is #116).

## Inputs

- pool: `2026-09-29-tilt-jm-pool.csv.gz` sha256 `c048471769b3835b422b5df5c130561503fe0a6969fa3ff469bf402c8b774dcd`
- pose scan: `2026-09-29-tilt-pose-jm.csv.gz` sha256 `2c4c2ab0c75ac7b05ea2d960223eafb282fbcecbcf08bf155499921ab5f38362`
- detections: `pool_detections.jsonl` sha256 `3dc69b34f2f8481666eca0278c78e17b4a7940ba96e21628df24dfc1b76bf00d`
- labels kept: 20857; dropped while loading: other_type_or_source 42721, no_pose_row 0, no_jpg 78, no_pose 5920, dims_differ 846, not_detected 0
- panos with a pose and detections: 13171

## Headline (tier 0.55, window 3 x 6 deg)

| group | pairs | panos | beta (SE) | beta pitch (SE) | beta roll (SE) | intercept | abs T p50 / p90 / max | pairs at abs T >= 3 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| all | 14701 | 9734 | 0.863 (0.010) | 0.880 (0.013) | 0.846 (0.014) | -0.02 | 0.91 / 2.48 / 10.17 | 866 |
| era legacy | 6410 | 4194 | 0.855 (0.014) | 0.880 (0.017) | 0.832 (0.020) | -0.06 | 0.96 / 2.61 / 10.17 | 431 |
| era mid | 5312 | 3447 | 0.852 (0.020) | 0.869 (0.024) | 0.832 (0.028) | -0.07 | 0.85 / 2.41 / 9.07 | 291 |
| era post179 | 2979 | 2151 | 0.905 (0.018) | 0.902 (0.027) | 0.908 (0.025) | 0.17 | 0.90 / 2.36 / 6.26 | 144 |
| pose npz | 3352 | 2363 | 0.896 (0.017) | 0.885 (0.025) | 0.906 (0.024) | 0.15 | 0.90 / 2.36 / 8.24 | 164 |
| pose xml | 11349 | 7371 | 0.855 (0.012) | 0.879 (0.015) | 0.831 (0.017) | -0.07 | 0.92 / 2.51 / 10.17 | 702 |
| abs T >= 3 deg | 866 | 760 | 0.783 (0.019) | 0.818 (0.024) | 0.735 (0.038) | -0.11 | 3.60 / 5.25 / 10.17 | 866 |

Of 20857 labels: 0 on a pano with no pose or detections, 0 whose stored pano size differs from the image's, 6156 with no detection in the window.

## Every tier and window (group `all`)

| tier | window (deg) | pairs | beta (SE) | beta pitch (SE) | beta roll (SE) |
|---|---:|---:|---:|---:|---:|
| 0.55 | 4 | 14034 | 0.780 (0.010) | 0.803 (0.013) | 0.758 (0.014) |
| 0.55 | 6 | 14701 | 0.863 (0.010) | 0.880 (0.013) | 0.846 (0.014) |
| 0.55 | 10 | 14878 | 0.902 (0.010) | 0.921 (0.012) | 0.880 (0.015) |
| 0.3 | 4 | 15071 | 0.782 (0.010) | 0.805 (0.012) | 0.761 (0.014) |
| 0.3 | 6 | 15794 | 0.865 (0.010) | 0.881 (0.012) | 0.849 (0.014) |
| 0.3 | 10 | 15997 | 0.902 (0.010) | 0.920 (0.012) | 0.883 (0.014) |

A wider window admits more wrong pairings (noise, not bias) and cuts off fewer large shifts, so beta rising with the window is truncation at the narrow end.

## Median offset by signed tilt (headline pairs, group `all`)

| T from | T to | n | median T | median (el_detection - el_human) |
|---:|---:|---:|---:|---:|
| -99 | -4 | 123 | -4.73 | -4.11 |
| -4 | -3 | 280 | -3.38 | -3.13 |
| -3 | -2 | 836 | -2.36 | -2.13 |
| -2 | -1 | 2141 | -1.39 | -1.33 |
| -1 | -0.5 | 1841 | -0.73 | -0.68 |
| -0.5 | 0.5 | 4371 | -0.00 | 0.02 |
| 0.5 | 1 | 1724 | 0.74 | 0.75 |
| 1 | 2 | 2155 | 1.38 | 1.36 |
| 2 | 3 | 767 | 2.33 | 2.29 |
| 3 | 4 | 304 | 3.39 | 3.05 |
| 4 | 99 | 159 | 4.71 | 3.98 |

## Reading it

- Standard errors are clustered by pano: labels on one pano share its pose.
- The intercept is where a person clicks on a ramp against where the detector peaks. It is not a tilt effect.
- Detections are rig-masked, as production ships them.
- `pairs.csv` holds the headline pairs, keyed on `label_uid`.
