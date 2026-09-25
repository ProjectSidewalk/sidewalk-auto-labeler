# Height QC verdicts (#44), pooled over bend, paterson, gainesville, sao_paulo

Pre-registered on [#44](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/44); scripts/height_qc.py has the rules verbatim and the interpretation choices fixed before the run.

## Verdicts

| test | verdict |
|---|---|
| T1 | **undecided** (per-pano association: undecided; 2.3 m: believe the vintage) |
| T2 | fallback: none |
| T3 | ships: |depth - vintage_median| >= 0.4 |
| T4 | constant sigma (p68 rising: per-pano False, 2.3 m False; kept-pano p68 0.255 (per-pano), 0.259 (2.3)) |

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

## Default-height decision table (rule applied; the default is NOT changed)

Range error = H / implied − 1 per pano (positive = ranges run long), median signed / median absolute. "Low rig" = vintages whose median depth is below 2.1 m. Options: (a) 2.6 m everywhere; (b) 1.08 × the depth height where the QC rule keeps it, else 2.6; (c) depth < 2.1 → 2.0 m, else 2.5 m where kept, else 2.6. Unmeasured and rejected panos raycast at 2.6 m under every option. T1 said: **undecided**.

Associated at per-pano:

| city | option | low rig (n) | low rig err | other (n) | other err | panos changed |
|---|---|---:|---:|---:|---:|---:|
| bend | a | 3 | +17.4% / +17.4% | 19124 | +3.9% / +6.6% | +0.0% |
| bend | b | 3 | +4.5% / +18.2% | 19124 | +2.4% / +6.4% | +82.5% |
| bend | c | 3 | +12.9% / +19.9% | 19124 | +0.1% / +5.9% | +82.5% |
| paterson | a | 5315 | +31.1% / +31.1% | 5742 | +2.6% / +6.6% | +0.0% |
| paterson | b | 5315 | +2.2% / +8.3% | 5742 | +1.0% / +6.6% | +86.0% |
| paterson | c | 5315 | +1.2% / +8.2% | 5742 | -1.2% / +6.4% | +86.0% |
| gainesville | a | 6069 | +33.9% / +33.9% | 3306 | +3.9% / +7.9% | +0.0% |
| gainesville | b | 6069 | +1.1% / +10.6% | 3306 | +1.2% / +7.9% | +72.4% |
| gainesville | c | 6069 | +4.2% / +10.2% | 3306 | +0.5% / +7.6% | +72.4% |
| sao_paulo | a | 175 | +17.7% / +18.6% | 9159 | +2.8% / +7.8% | +0.0% |
| sao_paulo | b | 175 | +4.9% / +15.0% | 9159 | -0.9% / +8.0% | +58.7% |
| sao_paulo | c | 175 | +7.9% / +16.0% | 9159 | -0.9% / +8.0% | +58.7% |

Associated at 2.3:

| city | option | low rig (n) | low rig err | other (n) | other err | panos changed |
|---|---|---:|---:|---:|---:|---:|
| bend | a | 3 | +17.4% / +17.4% | 19079 | +4.1% / +6.6% | +0.0% |
| bend | b | 3 | +4.5% / +18.2% | 19079 | +2.6% / +6.4% | +82.5% |
| bend | c | 3 | +12.9% / +20.4% | 19079 | +0.4% / +5.8% | +82.5% |
| paterson | a | 5031 | +26.6% / +26.6% | 5715 | +3.0% / +6.5% | +0.0% |
| paterson | b | 5031 | -1.2% / +7.9% | 5715 | +1.2% / +6.5% | +86.0% |
| paterson | c | 5031 | -2.1% / +7.7% | 5715 | -0.8% / +6.3% | +86.0% |
| gainesville | a | 5835 | +27.1% / +27.2% | 3315 | +4.1% / +7.7% | +0.0% |
| gainesville | b | 5835 | -3.6% / +10.9% | 3315 | +1.5% / +7.8% | +72.4% |
| gainesville | c | 5835 | -0.5% / +9.3% | 3315 | +0.9% / +7.4% | +72.4% |
| sao_paulo | a | 167 | +8.1% / +11.8% | 9215 | +3.4% / +7.6% | +0.0% |
| sao_paulo | b | 167 | -3.1% / +16.0% | 9215 | -0.2% / +8.3% | +58.7% |
| sao_paulo | c | 167 | -1.7% / +15.7% | 9215 | -0.2% / +8.1% | +58.7% |
