# Lyon: first Panoramax slice, and detection rates per rig

Lyon is a **training-only** city under
[RampNet#159](https://github.com/ProjectSidewalk/RampNet/issues/159): a city either trains
or evaluates, never both. So this run makes **no benchmark bundle and no GT export**, and
nothing is submitted to Project Sidewalk. What it gives
[RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158) and #159 is:

- the first detection slice of the city (Lyon commune, OSM relation 120965, 48.0 km²);
- which rigs are in the data, and how the model's detection rate differs by rig.

Bayonne ([docs/panoramax-bayonne.md](panoramax-bayonne.md)) was missing that second
diagnostic. Its rate was low but in range: 0.135 per pano at 0.55, against Richmond's 1.048.

This is **slice 1**: 40,000 of the 86,942 panos in the 10 m set. It is not "Lyon".

> **Key takeaways**
>
> 1. **Lyon's slice detects about 3x Bayonne's rate.** It reads **0.391** detections per
>    pano at 0.55 and 0.869 at 0.30 over 39,975 panos. Bayonne read 0.135 / 0.351.
>    Among the six earlier runs, only Annapolis (0.548) and Richmond (1.048) are higher.
>    *Figure 1; `data/census/lyon/detections.csv` `detections_per_pano`.*
> 2. **The same rig reads 3-4x higher in Lyon than in Bayonne, so the rig does not explain
>    Bayonne's low rate.** GoPro Max 5760x2880 reads 0.558 per pano in Lyon in 2026
>    (14,328 panos) and 0.547 in 2023. In Bayonne the same camera reads 0.104-0.156.
>    *Figure 1; `data/census/{lyon,bayonne}/rig_detections.csv` `detections_per_pano_0.55`.*
> 3. **Within one rig and one year, the operator moves the rate about 3x.** GoPro Max 2026
>    reads 0.633 for ecartip (10,944 panos, 350 sequences), 0.387 for ign_ddc_dtce_np and
>    0.224 for ign_ddc_dtce_fa. The Métropole's 8192x4096 Ladybug campaign reads
>    0.138-0.392 by year. Rig, operator, vintage and district are confounded, and nothing
>    here is a recall claim: there is no GT.
>    *`data/rig_producer_detections.csv` `detections_per_pano_0.55`.*
> 4. **Thinning favours the newest captures.** 2026 is 19% of the raw scan, but 40% of the
>    10 m set and of the slice. The slice's year mix matches the 10 m set's to within a
>    point, so it is a fair sample of what a full 10 m run would process.
>    *Figure 2; `data/scan_years.csv`, `data/census/lyon/years.csv`.*
> 5. **Clean run.** 0 failed and 25 skipped (0.06%), all for a missing `view:azimuth`.
>    The position check's median cross-track is **1.58 m**, at the metric's ~1.75 m floor
>    (it reports and never gates for Panoramax). The A40 ran at 1.25 panos/s for 8 h 53 m.
>    *`runs/lyon/manifest.json`, `run.log`, `runs/lyon/position_check.json`.*

## 0. Pre-registered run rule (fixed before launch)

1. **Thinning: 10 m.** This matches Bayonne and the RampNet#159 census row, and gives a
   GSV-like density. 5 m would be about 174k panos (about 41 h at the A40 rate). 20 m would
   bind `runs/lyon` to a sparser set than Bayonne's, and the two Panoramax cities would no
   longer be comparable. `--thin-spacing` is bound into the manifest ([#126](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/126)), so **20 m
   would be a new run name** (`lyon_20m`), not a resume.
2. **Budget: ≤ 12 h on the A40, so `--limit 40000`.** That is about 9.4 h at Bayonne's
   1.18 panos/s, with margin because the Ladybug `hd` files are larger. If another job had
   held more than 30% GPU utilisation or 30 GB of VRAM (not counting the resident
   ~9 GB gunicorn service), the limit would have dropped to 20,000. If the GPU had been
   pegged, the run would have stopped after the scan.
3. **Early abort:** after the first ~500 panos, stop if failures exceed 5% or skips exceed
   10%.
4. **Reporting:** the slice is reported as a slice. That is N of M thinned panos, uniform by
   sorted UUIDv4 id, because `--limit N` takes the first N of `sorted(thinned - processed)`.

**Outcome.** At launch the A40 held only the gunicorn service (8.8 GB, 0% utilisation), so
the limit stayed at 40,000. At 1,564 panos there were 0 failures and 0 skips. No STOP
condition fired.

## 1. The run

| | |
|---|---|
| command | `main.py example_geojson/lyon.geojson --name lyon --source panoramax --reuse-scan --thin-spacing 10 --limit 40000` (makelab2 A40, batch 1, default concurrency; tmux session `lyon`) |
| scan | 99 of 168 bbox z15 tiles within 50 m of the area; **381,295** in-area 360 pictures; **0** failed tiles (2026-10-06 04:46 UTC) |
| thinning | 5 m 174,525 · **10 m 86,942** · 20 m 38,796 |
| slice | `--limit 40000`: the first 40,000 of the 86,942 sorted ids; **46,942 remain** |
| processed / skipped / failed | **39,975** / 25 / **0** in one pass |
| wall time | 04:49:36 to 13:43:09 UTC (8 h 53 m); **1.251 panos/s** (`detector:` line: 39,975 forward passes, 31,944 s in forward over 31,966 s) |
| model | `rampnet-model@606a11956743` (trained 08-21-2025) |
| host / code | makelab2, worktree `~/sal-lyon` at `d929ccf` (detached), Python 3.12.13, torch 2.14.0+cu130, Pillow 12.3.0, A40 driver 595.80 |

The scan-only estimate is 36.2 h for the 10 m set at the 1.5 s/pano RTX 3070 constant. The
A40 ran at 0.80 s/pano, so the whole 10 m set would take about 19.3 h.

**Why a separate worktree on makelab2.** The usual py312 checkout
(`~/sidewalk-auto-labeler-py312`) holds the Bayonne and Vancouver run artifacts, and they
are uncommitted. It sat at `632e819`, 145 commits behind main. A fast-forward to main
would have collided with untracked `runs/vancouver/store_*` files that main now tracks. The
checkout was left untouched. The run used a detached `git worktree` of the same repo at
`origin/main` (`d929ccf`), running the checkout's `.venv/bin/python`. `runs/lyon` lives
under `~/sal-lyon`, and later slices run there.

**Census** (`data/census/lyon/`; the slice's processed panos):

- **Rig:** GoPro Max 5760x2880 56.4%. Point Grey Ladybug 8192x4096 21.1%. Make-less
  8192x4096 15.1% (Grand Lyon, the same Métropole campaign; see the pixel look below).
  Then Kandao QooCam 3 2.2%, RICOH THETA S 1.3%, insta360 x3 1.2%, and a tail of nine
  other models.
- **GoPro MAX2:** 191 panos at 7680x3840 were *processed*. In Bayonne, 104 MAX2 uploads
  served a vertically cropped `hd` image and were skipped. Lyon's MAX2 uploads pass the
  2:1 check.
- **Capture year:** 2026 40.3%, 2025 16.9%, 2022 15.5%, 2020 10.1%, 2021 8.0%,
  2023 6.6%, 2024 1.9%, ≤ 2019 0.7%.
- **Producer:** ecartip 33.7% and Grand Lyon 36.2% (two spellings, `grand lyon` and
  `grand-lyon`), both on the IGN instance under Etalab 2.0. 30 accounts in all, some on
  the OSM-FR instance under CC BY-SA 4.0.
- **Pose:** absent 54.5%, reported as exactly 0/0 40.1%, real tilt 5.4%. Bayonne's real
  tilt share was 36.8%.

**Pixel look (optional in the plan; five panos fetched by hand, not committed).** One
random pano was drawn per main group with `random.Random(159)` from the first 1,500
results:

- **Grand Lyon 8192x4096 (Ladybug, and the make-less 2021 pano):** solid **black** below
  y ≈ 0.81, because the rig has no downward camera. The green survey vehicle's roof and the
  mast base sit just above the black.
- **ecartip GoPro Max 2026:** a large roof-rack rig across the bottom ~25%.

There is no logo band like Bayonne's. Detections in the bottom band (y ≥ 0.79) are 4 of
15,617 at 0.55 and 91 of 34,726 at 0.30. On-rig detections (dip ≥ 49°) are 31 and 242
(0.20% / 0.70%). `send_to_ps.transform_record` masks those anyway.

## 2. Detections per pano by rig and capture year

([Figure 1](figures/panoramax-lyon/fig1_rig_rates.png); `data/census/{lyon,bayonne}/rig_detections.csv`)

| rig, year | Lyon panos | Lyon @0.55 / @0.30 | Bayonne panos | Bayonne @0.55 / @0.30 |
|---|---:|---:|---:|---:|
| GoPro Max 5760x2880, 2026 | 14,328 | **0.558** / 1.202 | 4,031 | 0.134 / 0.370 |
| GoPro Max 5760x2880, 2025 | 5,932 | 0.286 / 0.669 | 3,663 | 0.149 / 0.365 |
| GoPro Max 5760x2880, 2024 | 759 | 0.107 / 0.208 | 7,994 | 0.104 / 0.270 |
| GoPro Max 5760x2880, 2023 | 1,520 | 0.547 / 1.049 | 557 | 0.117 / 0.294 |
| GoPro Max 5376x2688, 2026 | — | — | 11,956 | 0.156 / 0.402 |
| Ladybug 8192x4096, 2022 | 5,435 | 0.299 / 0.703 | — | — |
| Ladybug 8192x4096, 2020 | 1,844 | 0.264 / 0.874 | — | — |
| Ladybug 8192x4096, 2021 | 1,113 | 0.188 / 0.520 | — | — |
| (no make) 8192x4096, 2020 / 2021 / 2023 / 2022 | 2,177 / 2,082 / 1,016 / 777 | 0.392 / 0.345 / 0.177 / 0.138 at 0.55 | — | — |
| Kandao QooCam 3 11136x5568, 2026 | 885 | 0.443 / 0.911 | — | — |
| **whole run** | 39,975 | **0.391** / 0.869 | 28,524 | **0.135** / 0.351 |

Whole-run rates at 0.55 for the six earlier runs: Clovis 0.123, Bayonne 0.135, Laurens
0.158, Morgantown 0.203, Annapolis 0.548, Richmond 1.048.

**Operator within rig** (`data/rig_producer_detections.csv`, groups ≥ 200 panos):

| rig, year | producer | panos (sequences) | @0.55 |
|---|---|---:|---:|
| GoPro Max 2026 | ecartip | 10,944 (350) | 0.633 |
| GoPro Max 2026 | ign_ddc_dtce_np | 1,439 (19) | 0.387 |
| GoPro Max 2026 | luppano / trucbidule | 369 / 265 | 0.360 / 0.343 |
| GoPro Max 2026 | ign_ddc_dtce_fa | 1,302 (35) | 0.224 |
| GoPro Max 2023 | mike en gyroroue | 1,064 (158) | 0.626 |
| GoPro Max 2024 | mike en gyroroue | 622 (44) | 0.059 |
| Ladybug 2022 / 2020 / 2021 | grand lyon | 5,434 / 1,788 / 1,054 | 0.299 / 0.265 / 0.190 |

**What this can and cannot say.**

- **There is no GT here.** A lower rate means fewer detections. It does not mean lower
  recall: the street may have fewer ramps, or the model may be more conservative on it.
- **Rig, operator, vintage, district and season are confounded.**
  - One operator holds most of each large group: ecartip for GoPro Max 2026, and the
    Métropole for every 8192x4096 pano.
  - The same contributor reads 0.626 in 2023 and 0.059 in 2024. That points to where and
    how the operator drove, not to the camera.
  - The Ladybug and make-less 8192x4096 years each cover different parts of the city.
- **What the data does support:** a GoPro Max is not a low-rate rig on Panoramax. Lyon's
  GoPro Max groups mostly read 0.22-0.63 against Bayonne's 0.10-0.16. Bayonne's low rate is
  therefore a property of Bayonne's imagery or streets, and the camera model does not
  explain it. One candidate is its single municipal operator (95% of panos) with the white
  nadir logo band. Another is French kerb design in that town. Bayonne GT
  ([PR #125](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/125)'s bundle)
  is what would separate them.
- **No uncertainty intervals.** The rates have no CIs. Detections cluster by sequence, so a
  per-pano Poisson interval would overstate precision. Read the sequence counts beside the
  operator rows instead.

## 3. Skips

All 25 skips are `No heading (view:azimuth)`, a deterministic skip that is cached and never
retried. `run.log` lists each skipped id. Since [#127](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/127), main.py logs a per-pano reason, which
Bayonne's run predated. The other known Panoramax deterministic skip, a MAX2 `hd` that is
not 2:1, did not occur in this slice.

## 4. Position check

`runs/lyon/position_check.json` (`rule` paired-unsigned-v1, issue [#62](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/62)):

- median cross-track to OSM centerlines **1.58 m**, IQR 0.65-3.64 m, p95 14.18 m (36,104
  panos scored);
- 3,871 panos (9.7%) more than 30 m from any street of the queried classes (Bayonne: 14%).

Panoramax serves one position per pano, so the check reports and never gates: `flagged` 0,
`submitted_field` null.

## 5. Replication

**Inputs**

| file | sha256 |
|---|---|
| `runs/lyon/results.jsonl` (39,975 lines; makelab2 and local, verified equal) | `802230ea8afd9194d359cf14a97c1a6a6ed2dd577da55357c548a893a0e841dc` |
| `runs/lyon/scan.json` (scanned 2026-10-06T04:46:02Z, local, uploaded to makelab2 and verified equal) | `20339fea38e9476e089977de29cccb52c8d3e8497c4f1377d21d4175ea93561b` |
| `runs/lyon/osm_streets.json` (Overpass, fetched by the run's position check) | `92915232b3ea34162882dac7df899545e9352281c2ee51e48bac13847c5fe8da` |
| `runs/lyon/manifest.json` (committed) | `1e880831964faa8b6a1f93d6234bd496f9e4c267880bb6cf5188192a249fcb9c` |
| `D:/Git/sidewalk-auto-labeler/runs/bayonne/results.jsonl` (for the Bayonne census) | `4f38ff528d4445b21eb02e99427e0ba3c88dfb5c940cc4702a283182e379b4b3` (as in the Bayonne report) |

**Environment**

| | |
|---|---|
| model | `projectsidewalk/rampnet-model` revision `606a11956743f7eb328d9207769034752f6191f4` |
| detection | makelab2: Python 3.12.13, torch 2.14.0+cu130, Pillow 12.3.0, NVIDIA A40 (driver 595.80), repo `d929ccf` |
| scan + analysis | Windows: Python 3.12.13, matplotlib 3.11.2, Pillow 12.3.0, repo `d929ccf` + this branch |

**Steps, in order.** "Net" means it needs the network.

| # | command | needs | time |
|---|---|---|---|
| 1 | `python main.py example_geojson/lyon.geojson --name lyon --source panoramax --scan-only` (stdout → `data/scan_only_fresh.txt`) | net | 70 s |
| 2 | the same with `--reuse-scan --thin-spacing {5,10,20}` → `data/scan_only_{5,10,20}m.txt` | — | 5 s each |
| 3 | `python scripts/panoramax_lyon_figures.py census-scan` → `data/scan_years.csv` | — | 5 s |
| 4 | upload `runs/lyon/scan.json` to the GPU host; compare `sha256sum` | net | 1 min |
| 5 | `python main.py example_geojson/lyon.geojson --name lyon --source panoramax --reuse-scan --thin-spacing 10 --limit 40000` (ends with the position check; Overpass) | GPU + net | 8 h 53 m |
| 6 | `python scripts/run_census.py runs/lyon --out docs/figures/panoramax-lyon/data/census/lyon --band-y 0.79` | — | 20 s |
| 7 | `python scripts/run_census.py runs/bayonne --results <bayonne results.jsonl> --out docs/figures/panoramax-lyon/data/census/bayonne --band-y 0.79` | — | 10 s |
| 8 | `python scripts/panoramax_lyon_figures.py rig-producers` → `data/rig_producer_detections.csv` | — | 20 s |
| 9 | `python scripts/panoramax_lyon_figures.py figures` → `fig1_rig_rates`, `fig2_years` (`.png` + `.svg`) | — | 5 s |

- **Step 2** binds nothing: `--scan-only` records no run entry.
- **Step 5** must keep `--thin-spacing 10`. The manifest now binds `thin_spacing_m: 10`, and
  a resume at any other spacing is refused.
- **Step 7:** the Bayonne CSVs that already existed (`detections`, `pose`, `producers`,
  `rigs`, `years`) were compared with the committed
  `docs/figures/panoramax-bayonne/data/census/` copies. They match line for line. Python's
  csv writer emits CRLF, and git stores LF. `rig_detections.csv` is the new table.
- **Step 9** reads only committed CSVs, plus the six-run rates from
  `docs/figures/panoramax-bayonne/data/fig5_detections.csv`. It was run twice, and every PNG
  and SVG hashed identical. SVGs are written with LF line endings.

**Committed vs regenerated.**

- **Committed:**
  - `runs/lyon/{manifest.json, area.geojson, position_check.json, position_report.html}`;
  - everything under `docs/figures/panoramax-lyon/`.
- **Regenerated, not committed:**
  - `results.jsonl`, `scan.json`, `osm_streets.json`, `already_processed.txt`, `run.log`
    and `run.times` (local `runs/lyon/` and makelab2 `~/sal-lyon/runs/lyon/`);
  - the five pixel-look thumbnails.

**Next slice** (on makelab2, in `~/sal-lyon`; it continues past the 40,000 already done and
processes ids 40,001-80,000 of the sorted 10 m set). The scan must be the same `scan.json`,
or a fresh scan must be taken and said so:

```
cd ~/sal-lyon && tmux new-session -d -s lyon2 "/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python \
  main.py example_geojson/lyon.geojson --name lyon --source panoramax --reuse-scan --thin-spacing 10 \
  --limit 40000 > runs/lyon/run2.log 2>&1"
```

A third slice with `--limit 6942` or no limit finishes the 10 m set (about 1.5 h).
`--reuse-scan` prints the scan's age. Coverage churns, so a fresh scan for the last slice
would add panos uploaded since 2026-10-06, and the manifest records `scan: fresh|reused`.

### Where each number lives

| number | file | column / row |
|---|---|---|
| 381,295 / 99 of 168 tiles / 0 failed | `data/scan_only_fresh.txt`; `runs/lyon/scan.json` (`pano_count`, `tile_count`, `failed_tiles`) | — |
| 174,525 / 86,942 / 38,796; 72.7 / 36.2 / 16.2 h estimates | `data/scan_only_{5,10,20}m.txt` | `Spatial thinning`, `Estimated` lines |
| 39,975 / 25 / 0; 86,942; 381,295 before thinning | `runs/lyon/manifest.json` `runs[0]` | `processed`, `skipped`, `failed`, `panos_found_in_area`, `panos_before_thinning` |
| 1.251 panos/s, 31,944 s, 8 h 53 m | `run.log` last `detector:` line; `run.times`; manifest `started_at` / `finished_at` | — |
| 25 skips, all `view:azimuth` | `run.log` `Skipped ... (cached, never retried)` lines | — |
| 0.391 / 0.869 per pano; on-rig 31 / 242; in-band 4 / 91 | `data/census/lyon/detections.csv` | `detections_per_pano`, `on_rig`, `in_band` |
| rig / year / pose / producer shares | `data/census/lyon/{rigs,years,pose,producers}.csv` | `share` |
| per rig × year rates, Lyon and Bayonne | `data/census/{lyon,bayonne}/rig_detections.csv` | `detections_per_pano_0.55`, `detections_per_pano_0.3`, `panos` |
| per operator rates | `data/rig_producer_detections.csv` | `detections_per_pano_0.55`, `panos`, `sequences` |
| six-run rates | `docs/figures/panoramax-bayonne/data/fig5_detections.csv` | `per_pano`, `tier = 0.55` |
| year mix of raw / thinned / slice | `data/scan_years.csv` | `raw`, `thin_*m`, `slice_10m_first40000` |
| 1.58 m, IQR 0.65-3.64, p95 14.18, 3,871 | `runs/lyon/position_check.json` | `fields.submitted.cross_track`, `panos_not_near_a_street` |
| black nadir from y ≈ 0.81 | pixel look (five panos, not committed) | — |

## Figures

| # | question it answers | file |
|---|---|---|
| 1 | Does the detection rate differ by rig, and does the rig explain Bayonne? Yes, it differs by rig, and no, it does not explain Bayonne: the same GoPro Max reads 0.56 per pano in Lyon in 2026 against 0.13-0.16 in Bayonne. | [fig1_rig_rates](figures/panoramax-lyon/fig1_rig_rates.png) |
| 2 | Is the slice representative, and what does thinning do to the year mix? Thinning lifts 2026 from 19% to 40%, and the slice matches the 10 m set. | [fig2_years](figures/panoramax-lyon/fig2_years.png) |

Alt text, in order:

1. Three horizontal bar panels on one axis of detections per processed pano. The top panel
   has 16 Lyon rig-and-year groups with at least 200 panos, each with a dark bar at 0.55
   and a light bar at 0.30. GoPro Max 2026 leads at 0.558 / 1.202 (n = 14,328); GoPro Max
   2023 reads 0.547, and Ladybug and make-less 8192x4096 groups read 0.14-0.39. The middle
   panel has Bayonne's six GoPro Max groups, all between 0.063 and 0.156 at 0.55. The bottom
   panel has the six earlier runs' whole-run rates at 0.55, from Clovis 0.123 to Richmond
   1.048.
2. Four stacked horizontal bars of capture-year shares: the raw scan (381,295), the 10 m set
   (86,942), the slice as drawn from the scan (40,000) and the slice as processed (39,975).
   2026 grows from 19% raw to 40% in the other three, and the three thinned bars are nearly
   identical.

## 6. What remains, and decisions for Jon

- **Remaining slices** (46,942 panos, about 10.4 h at 1.25 panos/s): the command above.
- **Native-res archive (deferred; not started).** The retrain needs pixels. On makelab2:
  - read `~/bayonne_archive.sh`, or the copy under
    `/projects/makeabilitylab/sidewalk-auto-labeler/`, and clone it for `lyon`;
  - check `df -h /projects/makeabilitylab` first. Lyon's 8192x4096 files are ~5 MB each,
    and the pool had 18 TB free on 2026-10-06;
  - a non-zero exit means the archive is not 1:1 (`makelab-archive-protocol`).
- **Decisions:**
  1. 10 m spacing is now bound to `runs/lyon`. The alternative is 20 m under a new name.
  2. Should later slices run as recorded?
  3. Archive now, or after more slices?
  4. Confirm: no GT bundle for Lyon (the train-or-evaluate rule).
- **For RampNet:** the operator effect (takeaway 3) suggests stratifying training samples
  by producer as well as by rig, so that one high-yield operator (ecartip) does not
  dominate the Lyon training set.
