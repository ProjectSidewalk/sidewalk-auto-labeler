# Height QC verdicts (#44), pooled over bend, paterson, gainesville, sao_paulo

Pre-registered on [#44](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/44); scripts/height_qc.py has the rules verbatim and the interpretation choices fixed before the run.

## Verdicts

| test | verdict |
|---|---|
| T1 | **undecided** (per-pano association: undecided; 2.3 m: believe the vintage) |
| T2 | fallback: none |
| T3 | pooled verdict ships: |depth - vintage_median| >= 0.4; **adopted: none** (per-city results below; the gate only flags) |
| T4 | constant sigma (p68 rising: per-pano False, 2.3 m False; kept-pano p68 0.255 (per-pano), 0.259 (2.3)); **not adopted**: scored against RampNet GT, the constant made p90 GT-to-site placement worse in all four cities, so the spread sigma stays (docs/camera-height-study.md) |

## T1: pano-weighted slope per city

| city | slope (per-pano) | slope (2.3) |
|---|---:|---:|
| bend | 0.227 | 0.075 |
| paterson | 0.522 | 0.197 |
| gainesville | 0.422 | 0.051 |
| sao_paulo | 0.503 | 0.090 |

## T2: low tail, pooled

| group | n (per-pano) | implied/depth (per-pano) | n (2.3) | implied/depth (2.3) |
|---|---:|---:|---:|---:|
| tilt < 6 | 513 | 1.297 | 562 | 1.525 |
| tilt >= 6 | 62 | 1.297 | 61 | 1.782 |

## T3: gates, pooled

| gate | rate (per-pano) | ratio (per-pano) | rate (2.3) | ratio (2.3) | ships |
|---|---:|---:|---:|---:|---|
| ground_tilt_deg >= 6 | 0.0339 | 1.542 | 0.0339 | 1.738 | no |
| height_spread_m >= 0.3 | 0.0776 | 1.267 | 0.0776 | 1.347 | no |
| ground_pixel_share < 0.15 | 0.0165 | 1.030 | 0.0165 | 1.013 | no |
| depth_planes < 60 | 0.0269 | 1.230 | 0.0269 | 1.229 | no |
| sky_fraction > 0.6 | 0.0000 | — | 0.0000 | — | no |
| |depth - vintage_median| >= 0.4 | 0.0403 | 2.384 | 0.0403 | 3.536 | yes |

## T3 per city (context for the pooled verdict; not pre-registered)

| gate | city | measured | flag rate | ratio (per-pano / 2.3) | passes in this city |
|---|---|---:|---:|---|---|
| ground_tilt_deg >= 6 | bend | 65866 | 0.0193 | 1.382 / 1.394 | no |
| ground_tilt_deg >= 6 | paterson | 30325 | 0.0462 | 1.497 / 1.699 | no |
| ground_tilt_deg >= 6 | gainesville | 30458 | 0.0093 | 1.464 / 1.780 | no |
| ground_tilt_deg >= 6 | sao_paulo | 18601 | 0.1060 | 1.427 / 1.565 | no |
| height_spread_m >= 0.3 | bend | 65866 | 0.0715 | 1.232 / 1.235 | no |
| height_spread_m >= 0.3 | paterson | 30325 | 0.1002 | 1.429 / 1.557 | no |
| height_spread_m >= 0.3 | gainesville | 30458 | 0.0233 | 1.128 / 1.228 | no |
| height_spread_m >= 0.3 | sao_paulo | 18601 | 0.1516 | 1.113 / 1.225 | no |
| ground_pixel_share < 0.15 | bend | 65866 | 0.0183 | 1.098 / 1.050 | no |
| ground_pixel_share < 0.15 | paterson | 30325 | 0.0163 | 1.155 / 1.088 | no |
| ground_pixel_share < 0.15 | gainesville | 30458 | 0.0074 | 0.752 / 0.843 | no |
| ground_pixel_share < 0.15 | sao_paulo | 18601 | 0.0252 | 1.059 / 0.932 | no |
| depth_planes < 60 | bend | 65866 | 0.0142 | 0.977 / 1.017 | no |
| depth_planes < 60 | paterson | 30325 | 0.0227 | 1.236 / 1.249 | no |
| depth_planes < 60 | gainesville | 30458 | 0.0668 | 1.026 / 1.026 | no |
| depth_planes < 60 | sao_paulo | 18601 | 0.0133 | 0.669 / 0.648 | no |
| sky_fraction > 0.6 | bend | 65866 | 0.0000 | — / — | no |
| sky_fraction > 0.6 | paterson | 30325 | 0.0000 | — / — | no |
| sky_fraction > 0.6 | gainesville | 30458 | 0.0000 | — / — | no |
| sky_fraction > 0.6 | sao_paulo | 18601 | 0.0000 | — / — | no |
| |depth - vintage_median| >= 0.4 | bend | 65866 | 0.0156 | 3.034 / 3.484 | yes |
| |depth - vintage_median| >= 0.4 | paterson | 30325 | 0.0162 | 2.145 / 3.161 | yes |
| |depth - vintage_median| >= 0.4 | gainesville | 30458 | 0.1106 | 2.339 / 3.626 | no |
| |depth - vintage_median| >= 0.4 | sao_paulo | 18601 | 0.0519 | 1.376 / 2.064 | no |

Share of the pooled measured panos: bend 45%, paterson 21%, gainesville 21%, sao_paulo 13%.

## POST-HOC: what a flagged pano should be raycast at

Not pre-registered (asked for by the PR #81 review). For panos the vintage-deviation gate flags (≥ 0.4 m from the vintage median): median |H / implied − 1| if H is the depth height, the 2.6 m default (the plan's fallback), or the vintage median.

| association | rig | side of median | n | keep depth | 2.6 m | vintage median |
|---|---|---|---:|---:|---:|---:|
| per-pano | low | below | 285 | 0.269 | 0.524 | 0.115 |
| per-pano | low | above | 85 | 0.120 | 0.118 | 0.317 |
| per-pano | other | below | 434 | 0.219 | 0.137 | 0.090 |
| 2.3 | low | below | 322 | 0.390 | 0.285 | 0.124 |
| 2.3 | low | above | 87 | 0.115 | 0.108 | 0.315 |
| 2.3 | other | below | 519 | 0.287 | 0.082 | 0.090 |

## T4: p68 of |resid| (m) by spread quartile, pooled

| quartile | spread (per-pano) | p68 (per-pano) | spread (2.3) | p68 (2.3) |
|---|---|---:|---|---:|
| Q1 | 0.000–0.042 | 0.261 | 0.000–0.043 | 0.271 |
| Q2 | 0.042–0.099 | 0.257 | 0.043–0.100 | 0.261 |
| Q3 | 0.099–0.166 | 0.249 | 0.100–0.167 | 0.252 |
| Q4 | 0.166–1.662 | 0.269 | 0.167–1.662 | 0.281 |

## Sensitivity (one threshold at a time, ×0.75 and ×1.25)

| knob | value | verdicts | moved? |
|---|---:|---|---|
| dev_m ×0.75 | 0.3 | T1 undecided; T2 fallback none; T3 ship none; T4 constant sigma | **yes** |
| dev_m ×1.25 | 0.5 | T1 undecided; T2 fallback none; T3 ship ['|depth - vintage_median| >= 0.5']; T4 constant sigma | no |
| tilt_deg ×0.75 | 4.5 | T1 undecided; T2 fallback none; T3 ship ['|depth - vintage_median| >= 0.4']; T4 constant sigma | no |
| tilt_deg ×1.25 | 7.5 | T1 undecided; T2 fallback none; T3 ship ['|depth - vintage_median| >= 0.4']; T4 constant sigma | no |
| ratio_max ×0.75 | 0.975 | T1 undecided; T2 fallback ['tilt < 6', 'tilt >= 6']; T3 ship ['|depth - vintage_median| >= 0.4']; T4 constant sigma | **yes** |
| ratio_max ×1.25 | 1.625 | T1 undecided; T2 fallback none; T3 ship ['|depth - vintage_median| >= 0.4']; T4 constant sigma | no |
| gate_factor ×0.75 | 1.5 | T1 undecided; T2 fallback none; T3 ship ['ground_tilt_deg >= 6', '|depth - vintage_median| >= 0.4']; T4 constant sigma | **yes** |
| gate_factor ×1.25 | 2.5 | T1 undecided; T2 fallback none; T3 ship none; T4 constant sigma | **yes** |

## Default-height decision table (the default is NOT changed)

Range error = H / implied − 1 per pano (positive = ranges run long), median signed / median absolute. "Low rig" = vintages whose median depth is below 2.1 m. Options: (a) 2.6 m everywhere; (b) 1.08 × the depth height; (c) depth < 2.1 → 2.0 m, else 2.5 m. No QC gate rejects (ADOPTED_GATES is empty), so every measured height is used; unmeasured panos raycast at 2.6 m under every option. T1 said: **undecided**. The pre-registered verdicts behind this table are fragile: the ±25% sensitivity pass moves them in 4 of 8 runs.

Associated at per-pano:

| city | option | low rig (n) | low rig err | other (n) | other err | panos changed |
|---|---|---:|---:|---:|---:|---:|
| bend | a | 3 | +17.4% / +17.4% | 19124 | +3.9% / +6.6% | +0.0% |
| bend | b | 3 | +4.5% / +18.2% | 19124 | +2.2% / +6.4% | +83.8% |
| bend | c | 3 | +12.9% / +19.9% | 19124 | +0.0% / +5.9% | +83.8% |
| paterson | a | 5315 | +31.1% / +31.1% | 5742 | +2.6% / +6.6% | +0.0% |
| paterson | b | 5315 | +1.9% / +8.2% | 5742 | +0.8% / +6.6% | +87.4% |
| paterson | c | 5315 | +1.2% / +8.2% | 5742 | -1.3% / +6.4% | +87.4% |
| gainesville | a | 6069 | +33.9% / +33.9% | 3306 | +3.9% / +7.9% | +0.0% |
| gainesville | b | 6069 | -0.2% / +10.4% | 3306 | +1.0% / +7.9% | +81.4% |
| gainesville | c | 6069 | +4.0% / +9.8% | 3306 | +0.4% / +7.6% | +81.4% |
| sao_paulo | a | 175 | +17.7% / +18.6% | 9159 | +2.8% / +7.8% | +0.0% |
| sao_paulo | b | 175 | +0.3% / +12.0% | 9159 | -1.3% / +8.0% | +61.9% |
| sao_paulo | c | 175 | +5.7% / +15.3% | 9159 | -1.2% / +8.0% | +61.9% |

Associated at 2.3:

| city | option | low rig (n) | low rig err | other (n) | other err | panos changed |
|---|---|---:|---:|---:|---:|---:|
| bend | a | 3 | +17.4% / +17.4% | 19079 | +4.1% / +6.6% | +0.0% |
| bend | b | 3 | +4.5% / +18.2% | 19079 | +2.5% / +6.5% | +83.8% |
| bend | c | 3 | +12.9% / +20.4% | 19079 | +0.3% / +5.9% | +83.8% |
| paterson | a | 5031 | +26.6% / +26.6% | 5715 | +3.0% / +6.5% | +0.0% |
| paterson | b | 5031 | -1.4% / +8.0% | 5715 | +1.1% / +6.6% | +87.4% |
| paterson | c | 5031 | -2.2% / +7.7% | 5715 | -0.9% / +6.4% | +87.4% |
| gainesville | a | 5835 | +27.1% / +27.2% | 3315 | +4.1% / +7.7% | +0.0% |
| gainesville | b | 5835 | -5.4% / +11.0% | 3315 | +1.3% / +7.9% | +81.4% |
| gainesville | c | 5835 | -1.2% / +8.8% | 3315 | +0.6% / +7.5% | +81.4% |
| sao_paulo | a | 167 | +8.1% / +11.8% | 9215 | +3.4% / +7.6% | +0.0% |
| sao_paulo | b | 167 | -5.9% / +14.1% | 9215 | -0.7% / +8.5% | +61.9% |
| sao_paulo | c | 167 | -3.9% / +15.1% | 9215 | -0.6% / +8.3% | +61.9% |
