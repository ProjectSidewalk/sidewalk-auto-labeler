# Lyon thinning: 5 m or 10 m? (#148)

**Question.** Lyon ran at 10 m thinning (86,942 panos), chosen for runtime before
[#144](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/144) measured that 10 m
loses 10-18 points of well-seen sites on Panoramax (Bayonne). Lyon detects about 3x
Bayonne's rate per pano ([docs/panoramax-lyon.md](panoramax-lyon.md)), and #144's own
reading is that density pays most where each view rarely fires. So does Bayonne's result
transfer to Lyon, and is densifying Lyon to 5 m worth its cost?

This doc reuses #144's protocol and script ([docs/thinning-experiment.md](thinning-experiment.md))
unchanged, on three Lyon sub-areas. **Nothing in production changes.**

> **Key takeaways**
>
> PENDING: filled in after the GPU runs (section 3 onward).

## 1. Densify cost, from the scan alone (no GPU)

`python scripts/thinning_experiment.py subset runs/lyon --from 10 --to 5 --panos-per-second 2.505 --out docs/figures/thinning-experiment-lyon/data`
reads the canonical run's own `scan.json` (sha256 `20339fea...`, scanned 2026-10-06 04:46 UTC).

| | panos |
|---|---:|
| raw scan | 381,295 |
| 5 m | 174,525 |
| 10 m (the run) | 86,942 |
| 20 m | 38,796 |
| 10 m panos *not* kept at 5 m | **0** |
| added by densifying 10 m -> 5 m | **87,583** |
| GPU time for the added set at Hyak's measured 2.505 panos/s (job 41472534, L40S) | **9.7 h** |

**The 10 m set is an exact subset of the 5 m set**, as it was in Bayonne. Densifying later
only adds panos: letting the 10 m run finish cost nothing.

The added set is **older** than the 10 m set. Newest-capture-wins already gave 10 m the
newest pano in each 10 m cell, so the extra 5 m panos are mostly the next-newest:

| capture year | 10 m set | added by 5 m | share of added |
|---|---:|---:|---:|
| 2026 | 35,200 | 16,170 | 18.5% |
| 2025 | 14,661 | 11,105 | 12.7% |
| 2024 | 1,616 | 752 | 0.9% |
| 2023 | 5,651 | 9,721 | 11.1% |
| 2022 | 13,458 | 22,450 | 25.6% |
| 2021 | 7,041 | 9,410 | 10.7% |
| 2020 | 8,662 | 16,987 | 19.4% |
| <= 2019 | 653 | 988 | 1.1% |

(`data/densify_years.csv`; 2011, 2018 and 2019 summed into `<= 2019`.)

## 2. Pre-registered rule (committed and pushed before any GPU job)

**Why sub-areas, not a full densify.** #144's protocol needs an un-thinned
(`--thin-spacing 0`) run as the full-density denominator, which the 10 m run cannot supply.
A full 5 m densify would take ~9.7 h of ckpt GPU and answers a different question (it has no
denominator either). So: three sub-areas, run un-thinned, scored offline at every spacing.

**Boxes.** Chosen to span the strata RampNet#159's audit cares about (camera and operator),
read from the canonical 10 m run's producer mix inside each box. Each was then sized with
`--scan-only` (fresh scans, 2026-10-07 04:28-04:30 UTC) to fall within the budget.

| box | run name | geojson | centre, size | stratum (10 m set inside the box) | raw panos (scan-only) |
|---|---|---|---|---|---:|
| A | `thinexp_lyon_a` | `example_geojson/lyon_thinexp_a.geojson` | 45.7667 N 4.8452 E, 330 x 330 m (Presqu'île) | ecartip GoPro Max 5760x2880 (70% of the box's 10 m panos; raw scan 75% 2020 Ladybug, which newest-wins mostly drops) | 4,226 |
| B | `thinexp_lyon_b` | `example_geojson/lyon_thinexp_b.geojson` | 45.7487 N 4.8436 E, 520 x 520 m (Confluence) | Grand Lyon 8192x4096 (make-less, 2021 campaign; 72%) | 4,379 |
| C | `thinexp_lyon_c` | `example_geojson/lyon_thinexp_c.geojson` | 45.7487 N 4.8629 E, 550 x 550 m (east bank, 7e) | mixed accounts, no producer > 50% (ecartip 48%, Grand Lyon Ladybug 24%, IGN 10%, others) | 3,941 |
| | | | | **total** | **12,546** |

**Budget rule.**

1. Each box holds 3,000-6,000 raw panos by `--scan-only`; the total is at most 20,000.
   (A box over 8,000 would have been shrunk before submission.)
2. Detection runs on Hyak `ckpt-g2` (one L40S per box, the three in parallel), from a
   separate fresh clone (`repo-thinexp`), never the canonical `repo-main`. Each job runs
   `main.py example_geojson/lyon_thinexp_<x>.geojson --name thinexp_lyon_<x> --source panoramax --thin-spacing 0 --reuse-scan`
   on the scan committed here (uploaded with the run dir), so the processed set is the
   pre-registered one.
3. **Early abort:** after ~500 panos, a job whose failures exceed 5% or whose skips exceed
   10% is cancelled and not blindly retried.
4. **Time box:** if the jobs have not finished ~6 h after submission, what is done is
   written up and the PR stays a draft.

**Analysis (unchanged from #144).** `scripts/thinning_experiment.py runs/thinexp_lyon_<x>`
at 0.55 and with `--min-confidence 0.3`: 7.5 m greedy sites from flat 2.6 m raycasts, rig
mask on, spacings 0 / 2.5 / 5 / 7.5 / 10 / 15 / 20 m, random same-count control over 20
seeds. Figures go to `docs/figures/thinning-experiment-lyon/` (never #144's directory).

**Sanity checks.**

- Each box's 10 m kept set should sit inside the canonical run's processed set
  (`runs/lyon/already_processed.txt`), apart from cells cut by the box edge, where the
  newest pano of the cell can lie outside the box. Report the count.
- Detections on panos shared with the canonical run should match it (same model revision).
  Report the share that match exactly and the largest confidence difference.

**Reading guide for the recommendation** (fixed now; the call stays Jon's). The headline
is the robust-site gap, 5 m minus 10 m, as points of the box's full-density robust count,
at 0.30 (the tier production ships and training mines), read with 0.55 beside it.

- **Favours 5 m** if the gap is >= 5 points in at least 2 of the 3 boxes (Bayonne, #144: 10 points
  at 0.55, 18 at 0.30).
- **Favours staying at 10 m** if the gap is <= 2 points in at least 2 of the 3 boxes.
- **Otherwise mixed**: the per-box pattern (stratum) and the 2+-view counts carry the
  reading.

The 2+-view column (what multi-view mining and fusion need) and all-site retention are
reported beside it, as in #144, but do not change the reading above.

## 3. Results

PENDING.
