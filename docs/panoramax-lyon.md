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

**Update 2026-10-07:** the whole 10 m set has since finished (86,889 processed). Sections
0-6 below still describe slice 1 and its committed data; the completed run is in
[section 7](#7-full-10-m-run-completed-2026-10-06). The 5 m evidence is
[docs/thinning-experiment-lyon.md](thinning-experiment-lyon.md) (#148); Jon decided on
2026-10-07 to densify Lyon to 5 m.

> **Key takeaways**
>
> 1. **Lyon's slice detects about 3x Bayonne's rate.** It reads **0.391** detections per
>    pano at 0.55 and 0.869 at 0.30 over 39,975 panos. Bayonne read 0.135 / 0.351.
>    Among the six earlier runs, only Annapolis (0.548) and Richmond (1.048) are higher.
>    *Figure 1; `data/census/lyon/detections.csv` `detections_per_pano`.*
> 2. **The camera model does not explain Bayonne's low rate, but the gap is about 2x, not
>    3-4x.** GoPro Max 5760x2880 2026 reads 0.558 per pano in Lyon against 0.134 in
>    Bayonne. 76% of that Lyon group is one operator, ecartip (0.633). Without ecartip it
>    reads **0.317** (1,072 detections on 3,384 panos), still **2.4x** Bayonne.
>    - 2025 without ecartip reads 0.302, 2.0x Bayonne's 0.149.
>    - 2024 is level: 0.107 vs 0.104.
>    - 2023's 0.547 is 70% one contributor (mike en gyroroue), who reads 0.059 in 2024.
>
>    *Figure 1; `data/census/{lyon,bayonne}/rig_detections.csv` `detections_0.55`, `panos`,
>    minus the ecartip rows' `detections_0.55`, `panos` in
>    `data/rig_producer_detections.csv`.*
> 3. **Within one camera model and one year, the operator moves the rate about 3x.**
>    - GoPro Max 2026 reads 0.633 for ecartip (10,944 panos, 350 sequences), 0.387 for
>      ign_ddc_dtce_np and 0.224 for ign_ddc_dtce_fa.
>    - The candidates are route and district, and **mount** (camera height). ecartip's
>      below-horizon detections sit at a median dip of 18.3°, against 12.3° for both IGN
>      accounts. At a fixed ramp distance that is the geometry of a higher camera.
>    - It is not rig false positives: the rig mask removes 17 of ecartip's 6,930 detections.
>    - District alone is as large. The Métropole's make-less 8192x4096 campaign in 2021 reads
>      **0.711** under the `grand lyon` account (692 panos) and **0.163** under `grand-lyon`
>      (1,390 panos): one campaign, one year, two spellings.
>    - Camera model, operator, mount, vintage and district are confounded, and nothing here
>      is a recall claim: there is no GT.
>
>    *`data/rig_producer_detections.csv` `detections_per_pano_0.55`, `on_rig_0.55`,
>    `median_dip_deg_below_horizon_0.55`.*
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
  Then Kandao QooCam 3 2.2%, RICOH THETA S 1.3%, insta360 x3 1.2%, and a tail of other
  models (`rigs.csv`).
- **GoPro MAX2:** 191 panos at 7680x3840 were *processed*. In Bayonne, 104 MAX2 uploads
  served a vertically cropped `hd` image and were skipped. Lyon's MAX2 uploads pass the
  2:1 check.
- **Capture year:** 2026 40.3%, 2025 16.9%, 2022 15.5%, 2020 10.1%, 2021 8.0%,
  2023 6.6%, 2024 1.9%, ≤ 2019 0.8%.
- **Producer:** ecartip 33.7% and Grand Lyon 36.2% (two spellings, `grand lyon` and
  `grand-lyon`), both on the IGN instance under Etalab 2.0. The slice holds 29 producer
  names (28 with the two Grand Lyon spellings merged), some on the OSM-FR instance under
  CC BY-SA 4.0. The 30 accounts in the 2026-09-04 census were counted over the whole city.
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
(0.20% / 0.70%). `send_to_ps.transform_record` masks those anyway. **Every rate in this
report includes on-rig detections**, as `detections.csv` does. In Lyon that moves nothing.

**A fixed-pixel artifact at 0.30** (from the PR review's ad-hoc check, not a committed
script):
- 207 of 9,973 Grand Lyon detections at 0.30 (2.1%) sit in 1024x512 cells that repeat in
  ≥ 5 panos of one sequence.
- The main one is (x 0.503, y 0.756), about 46° dip, straight ahead, just above the black
  nadir: probably the survey vehicle or mast. Others sit at the seam near the horizon.
- None reaches 0.55. It matters only if Lyon's 0.30 tier feeds training mining.

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

**Operator within camera model** (`data/rig_producer_detections.csv`, groups ≥ 200
panos):

| camera, year | producer | panos (sequences) | @0.55 | median dip below horizon | above horizon | pose 0/0 |
|---|---|---:|---:|---:|---:|---:|
| GoPro Max 2026 | ecartip | 10,944 (350) | 0.633 | 18.3° | 10.3% | 99.9% |
| GoPro Max 2026 | ign_ddc_dtce_np | 1,439 (19) | 0.387 | 12.3° | 1.4% | 0% |
| GoPro Max 2026 | luppano / trucbidule | 369 / 265 | 0.360 / 0.343 | 15.1° / 12.7° | 0% / 4.4% | 0% |
| GoPro Max 2026 | ign_ddc_dtce_fa | 1,302 (35) | 0.224 | 12.3° | 1.7% | 0% |
| GoPro Max 2025 | ecartip | 2,500 (177) | 0.263 | 20.7° | 12.3% | 100% |
| GoPro Max 2023 | mike en gyroroue | 1,064 (158) | 0.626 | 12.3° | 1.4% | 0% |
| GoPro Max 2024 | mike en gyroroue | 622 (44) | 0.059 | 15.1° | 0% | 0% |
| Ladybug 2022 / 2020 / 2021 | grand lyon | 5,434 / 1,788 / 1,054 | 0.299 / 0.265 / 0.190 | 15.5° / 17.9° / 15.5° | 0.1% / 5.9% / 0% | 0% |
| make-less 8192x4096, 2021 | grand lyon / grand-lyon | 692 / 1,390 | **0.711 / 0.163** | 15.5° / 12.7° | 0% / 3.1% | 0% |

The dip medians come from detections on a coarse grid of about 2.8° steps (RampNet's
8-cell heatmap grid). Read 12.3° vs 18.3° as two grid steps apart, not as a precise ratio.

**What this can and cannot say.**

- **There is no GT here.** A lower rate means fewer detections. It does not mean lower
  recall: the street may have fewer ramps, or the model may be more conservative on it.
- **Camera model, operator, mount, vintage, district and season are confounded.**
  - One operator holds most of each large group: ecartip for GoPro Max 2026, and the
    Métropole for every 8192x4096 pano.
  - The same contributor reads 0.626 in 2023 and 0.059 in 2024. That points to where and
    how the operator drove, not to the camera.
  - The Ladybug and make-less 8192x4096 years each cover different parts of the city. One
    2021 campaign splits 0.711 vs 0.163 across two account spellings, the clearest sign
    that district matters as much as anything here.
- **Mount, not rig false positives.**
  - ecartip's detections sit lower in the image: median dip 18.3° against 12.3° for the
    IGN accounts. At a fixed distance that is a camera about 1.5x higher. That fits its
    large roof rack, though narrower streets could also produce it.
  - Camera height changes how many pixels a ramp covers, and so detectability. It is a
    property of the *rig* (camera plus mount), in the sense CLAUDE.md's nadir-mask section
    uses the word.
  - The rig mask removes only 17 of ecartip's 6,930 detections at 0.55 (`on_rig_0.55`).
  - The PR review found no repeated-pixel signature like Laurens' roof rack in ecartip's
    sequences.
- **Unexplained: ecartip's above-horizon detections.**
  - 10.3% of ecartip's 0.55 detections sit above the horizon (y < 0.5, mostly
    0.475-0.49), against 0-4.4% for the other GoPro Max 2026 producers and at most 9.5% in any other group (`above_horizon_share_0.55`). They cluster straight
    ahead and behind.
  - ecartip reports pose as exactly 0/0 on 99.9% of its panos (`pose_zeros_share`), so the
    images may not be levelled. These could be far ramps on Lyon's slopes, or a horizon
    offset.
  - Without them ecartip still reads about 0.57. But a ray cast from them hits no ground,
    so they matter for any geometric use of these panos in training or mining.
- **What the data does support: the camera model does not explain Bayonne's low rate.**
  - Non-ecartip GoPro Max groups in Lyon mostly read 0.17-0.39, against Bayonne's
    0.06-0.16. The exceptions are mike en gyroroue, at 0.626 in 2023 and 0.059 in 2024.
  - Without the dominant operator, the gap is about 2x in 2026 and 2025, and level in 2024.
  - Operator and mount, which include camera height, are confounded with Bayonne's rate.
    Bayonne's municipal rig (95% of panos) is also car-mounted, at a height nobody has
    measured, and carries the white nadir logo band.
  - Another candidate is French kerb design in that town.
  - Bayonne GT
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
| per operator rates; 0.711 / 0.163 | `data/rig_producer_detections.csv` | `detections_per_pano_0.55`, `panos`, `sequences` |
| without ecartip: 0.317 (2.4x) in 2026, 0.302 in 2025 | `data/census/lyon/rig_detections.csv` minus the ecartip rows of `data/rig_producer_detections.csv` | `detections_0.55`, `panos` |
| 17 of 6,930 on rig; dip 18.3° vs 12.3°; 10.3% above horizon; 99.9% pose 0/0 | `data/rig_producer_detections.csv` | `on_rig_0.55`, `detections_0.55`, `median_dip_deg_below_horizon_0.55`, `above_horizon_share_0.55`, `pose_zeros_share` |
| Grand Lyon 0.30 fixed-pixel artifact (207 of 9,973) | PR #145 review, ad-hoc check | — |
| six-run rates | `docs/figures/panoramax-bayonne/data/fig5_detections.csv` | `per_pano`, `tier = 0.55` |
| year mix of raw / thinned / slice | `data/scan_years.csv` | `raw`, `thin_*m`, `slice_10m_first40000` |
| 1.58 m, IQR 0.65-3.64, p95 14.18, 3,871 | `runs/lyon/position_check.json` | `fields.submitted.cross_track`, `panos_not_near_a_street` |
| black nadir from y ≈ 0.81 | pixel look (five panos, not committed) | — |

## Figures

| # | question it answers | file |
|---|---|---|
| 1 | Does the detection rate differ by camera model and year, and does the camera model explain Bayonne? It differs, and the camera model does not explain Bayonne. GoPro Max 2026 reads 0.56 per pano in Lyon against 0.13 in Bayonne, and 0.32 (2.4x) without ecartip. | [fig1_rig_rates](figures/panoramax-lyon/fig1_rig_rates.png) |
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
  by producer as well as by camera model, so that one high-yield operator (ecartip) does
  not dominate the Lyon training set. ecartip's above-horizon detections and unlevelled
  pose need a look before its panos feed any geometric mining.

## 7. Full 10 m run (completed 2026-10-06)

The remaining 46,942 panos of the 10 m set ran on Hyak, not makelab2: job **41472534**
(`ckpt-g2`, one L40S, `sal_lyon_ckpt.sbatch` in a plain clone of main at `8490974`),
`main.py example_geojson/lyon.geojson --name lyon --source panoramax --reuse-scan --thin-spacing 10`
with no `--limit`, on the same 2026-10-06 04:46 UTC scan. The run dir there
(`/gscratch/makelab/jfroehli/sidewalk-auto-labeler/repo-main/runs/lyon`) is canonical; the
makelab2 `~/sal-lyon` copy holds slice 1 only.

| | slice 1 (makelab2 A40) | slice 2 (Hyak L40S, job 41472534) | whole 10 m set |
|---|---:|---:|---:|
| processed | 39,975 | 46,914 | **86,889** of 86,942 |
| skipped | 25 | 28 | 53 (0.06%) |
| failed | 0 | 0 | 0 |
| wall time (UTC) | 10-06 04:49-13:42 | 10-06 16:12-21:24 | |
| rate (`detector:` line) | 1.251 panos/s | **2.505** panos/s (18,436 s in forward over 18,728 s) | |

- **Skips:** 52 `No heading (view:azimuth)` and one `Not a full 360x180 equirectangular
  (12416x2096)` (a vertically cropped `hd`, like Bayonne's MAX2 uploads). All are cached,
  never retried. `run2.log` lists each id.
- **Detections, whole set:** **0.390** per pano at 0.55 (33,872) and 0.867 at 0.30 (75,372);
  on rig 66 / 549. Slice 1 read 0.391 / 0.869, so the slice was a fair sample.
  *`docs/figures/panoramax-lyon/data/census/lyon_full/detections.csv`.*
- **Mix, whole set:** GoPro Max 5760x2880 56.4%, Ladybug 8192x4096 21.1%, make-less
  8192x4096 15.1%; 2026 40.5%; ecartip 33.7%, Grand Lyon 36.1% (both spellings). The
  operator reading of section 2 holds: ecartip GoPro Max 2026 reads 0.640 per pano at 0.55
  (23,776 panos, 360 sequences), ign_ddc_dtce_np 0.392, ign_ddc_dtce_fa 0.228.
  *`data/census/lyon_full/{rigs,years,producers,rig_detections}.csv`,
  `data/full/rig_producer_detections.csv`.*
- **Position check:** median cross-track 1.58 m (IQR 0.65-3.61, p95 13.98; 78,506 panos
  scored), 8,383 panos more than 30 m from a street, `flagged` 0.
  *`runs/lyon/position_check.json`.*

The census CSVs went to `data/census/lyon_full/` and `data/full/`, not over slice 1's, so
every number in sections 0-6 still reads from the files it names. `census/lyon_full/` was
written with [PR #150](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/150)'s
`run_census.py` (email-shaped producer names masked); this branch carries none of #150's
code.

| file | sha256 |
|---|---|
| `runs/lyon/results.jsonl` (86,889 lines; Hyak and local, verified equal) | `6e8a11739a75eac1db85cd971323d95283d7f40c8cad98cc59ec4b8f907578e8` |
| `runs/lyon/scan.json` (unchanged since slice 1) | `20339fea38e9476e089977de29cccb52c8d3e8497c4f1377d21d4175ea93561b` |
| `runs/lyon/already_processed.txt` (86,942 ids) | `816fa4c98a59216d6309a30197e95533efd7757fef67edcedad1db867fd614ac` |
| `runs/lyon/manifest.json` (committed) | `c566a75a835de83b32b14f17788edd4fc3ed064b33f0daf525ca5c6efd2b36c0` |
| `runs/lyon/run2.log` (Hyak `lyon_41472534.log`, not committed) | `ced0ff777cd4150402169036cbb6a2aedeaf44b2b882068c39eb2ac595cd0c73` |

Regenerate the two tables (no network, no GPU, ~1 min each):

```
python scripts/run_census.py runs/lyon --out docs/figures/panoramax-lyon/data/census/lyon_full --band-y 0.79
python scripts/panoramax_lyon_figures.py rig-producers --out docs/figures/panoramax-lyon/data/full
```

**Run the first command only once #150 is on main.** This branch's `run_census.py` is
main's unmasked one, so running it here would overwrite the masked `lyon_full/producers.csv`
with an email-shaped producer name. It regenerates byte for byte with #150's version
(checked in the #156 review). `rig-producers` masks the same way since that review.

Section 6's "remaining slices" is done. The native-res archive (#146) now covers 86,889
panos. Jon decided on 2026-10-07 to densify to 5 m on the evidence in
[docs/thinning-experiment-lyon.md](thinning-experiment-lyon.md).

## 8. Densify to 5 m (completed 2026-10-07)

Jon's decision of 2026-10-07 (evidence: [docs/thinning-experiment-lyon.md](thinning-experiment-lyon.md),
#148) was carried out in place, the way Bayonne was (#147, PR #154). Re-thinning the same
reused `scan.json` at 5 m gives 174,525 panos, and the 10 m set (86,942) is an exact subset
of it, so a resume at `--thin-spacing 5` appends exactly the 87,583 missing cells. The run
dir's binding was moved from 10 m to 5 m by hand first; the manifest's
`thin_spacing_changed` block records how and why. The 10 m files were backed up beside the
run dir (`runs/lyon.bak-10m-2026-10-07/`, with `SHA256SUMS`) before the job started.
Tracked in #159.

Hyak job **41504145** (`ckpt-g2`, one L40S, `sal_lyon_densify_ckpt.sbatch`, plain clone
of main at `fd08b8f`):
`main.py example_geojson/lyon.geojson --name lyon --source panoramax --reuse-scan --thin-spacing 5`,
no `--limit`. It was requeued once, which `ckpt` does:

| | attempt 1 | attempt 2 (resume) | densify total |
|---|---:|---:|---:|
| wall time (UTC) | 10-07 12:38-20:45, requeued | 10-07 20:48-23:42 | |
| added to the cache | 61,876 | 25,707 | **87,583** |
| processed | 61,818 | 25,698 | **87,516** |
| skipped | 58 | 9 | 67 |
| failed | see below | 0 | 0 |
| rate (`detector:` line) | not printed | **2.512** panos/s | |

- **Attempt 1 left no `runs[]` entry and no Final Report.** A requeued job is killed, so
  `main.py` never wrote its manifest entry. Its per-pano stdout lines were lost too; they
  were still buffered, while tqdm on stderr was not. `runs[2]` therefore records only
  attempt 2 (25,698 processed / 9 skipped). The attempt-1 column above is counted from the
  files: lines and cache ids added between the 10 m backup and the restart's
  `Already processed (skipped): 148818`. tqdm had counted 75,988 completions at the
  requeue, against 61,876 cached. The difference is retryable failures, which `main.py`
  never caches; their reasons went with the lost stdout. Attempt 2 retried every one of
  them with 0 failures.
- **Skips: all 67 are `No heading (view:azimuth)`.** 9 are in the log. The other 58 were
  re-derived on 2026-10-07 from each live STAC item, applying `fetch_pano`'s metadata
  checks in order; all 58 lack `view:azimuth`. Over the whole 5 m set that is 120 skips
  (0.07%): 119 no heading and the one cropped `hd` from section 7.
- **Checks.** `results.jsonl` has 174,405 lines and 174,405 unique `panorama_id`s, none
  unparseable. `already_processed.txt` holds 174,525 unique ids. Every results id is in
  it, and the 120 extra ids are exactly the skips. The end-of-run position check reads
  0 flagged. Its `results_sha256` equals the file's, so the check covers the final file.
- **Detections, whole 5 m set:** **0.424** per pano at 0.55 (73,925) and 0.940 at 0.30
  (163,987); on rig 141 / 1,061. The 87,516 added panos alone read 0.458 / 1.013, above
  the 10 m set's 0.390 / 0.867. The cause is the mix the densify adds. *`data/census/lyon_5m/detections.csv`.*
- **Mix, whole 5 m set:** the densify adds mostly the Métropole's older imagery. Grand
  Lyon (both spellings) is 49.4% of panos, up from 36.1%. ecartip falls to 26.1% (from
  33.7%). Capture years 2020-22 are 44.7% (from 33.6%), and 2026 is 29.5% (from 40.5%).
  By rig: GoPro Max 5760x2880 45.0%, Ladybug 8192x4096 30.4%, make-less 8192x4096 19.0%.
  *`data/census/lyon_5m/{rigs,years,producers,rig_detections}.csv`,
  `data/full_5m/rig_producer_detections.csv`.* One producer name is email-shaped and is
  masked (`j***@***`, 66 panos), per #150.
- **Position check:** median cross-track 1.45 m (IQR 0.60-3.26, p95 12.72; 159,858 panos
  scored). 14,547 panos are more than 30 m from a street, and `flagged` is 0.
  *`runs/lyon/position_check.json`.*

RampNet#159's per-producer caps may limit how many of the added Grand Lyon panos train.
That is a training-side question, and it does not block this run.

| file | sha256 |
|---|---|
| `runs/lyon/results.jsonl` (174,405 lines; Hyak and local, verified equal) | `60e688b28b62f85e71f51c603dc1b6f4ff020348220c6cc7955891dcfef061ef` |
| `runs/lyon/already_processed.txt` (174,525 ids) | `6ae9cafc04bdb3f8d0b69b3565eb83655038aa0448acf46c57ad110a942c29fa` |
| `runs/lyon/scan.json` (unchanged since slice 1) | `20339fea38e9476e089977de29cccb52c8d3e8497c4f1377d21d4175ea93561b` |
| `runs/lyon/osm_streets.json` (cached, not committed) | `92915232b3ea34162882dac7df899545e9352281c2ee51e48bac13847c5fe8da` |
| `runs/lyon/manifest.json` (committed) | `f4b50585b5b5b5d73431acdf0d19f0a095e661d8316c6f0d07890353f7b3f4fa` |
| `runs/lyon/position_check.json` (committed) | `47feda1bfa95246d44d945be05bbff56e9fd625244fbf4a91d40e65b44e626c2` |
| `runs/lyon/position_report.html` (committed) | `b94b6a718ed331e92f04baa1d8c893b3d5f412018865928f8e3a2635b764fe9c` |
| `runs/lyon/run3.log` (Hyak `lyon_densify_41504145.log`, not committed) | `22defce1410b303c1a20cb09ffeab1655f89cc3694e8254e20314755aefdcc94` |

Sections 1-7 and their files still describe the 10 m run. `data/census/lyon_full/` and
`data/full/` are kept as section 7's evidence. Regenerate the 5 m tables with no network
and no GPU, about 1 min each. The census step needs #150's `run_census.py`, as in section 7:

```
python scripts/run_census.py runs/lyon --out docs/figures/panoramax-lyon/data/census/lyon_5m --band-y 0.79
python scripts/panoramax_lyon_figures.py rig-producers --out docs/figures/panoramax-lyon/data/full_5m
```

| number | file | column / row |
|---|---|---|
| 174,525 found, 25,698 / 9 / 0 for attempt 2, 2.512 panos/s | `runs/lyon/manifest.json` `runs[2]`; `run3.log` last `detector:` line | `panos_found_in_area`, `processed`, `skipped`, `failed` |
| 61,876 / 61,818 / 58 for attempt 1; 75,988 at requeue | `run3.log` (`Already processed (skipped): 148818`, last tqdm line before `CANCELLED ... DUE TO JOB REQUEUE`); line and cache counts against the 10 m backup | — |
| 67 skips, all `view:azimuth` | `data/full_5m/densify_skips.csv` (`source` = `log` for 9, `api` for 58, re-derived 2026-10-07) | `reason` |
| 0.424 / 0.940 per pano; on rig 141 / 1,061 | `data/census/lyon_5m/detections.csv` | `detections_per_pano`, `on_rig` |
| 0.458 / 1.013 for the added panos | `results.jsonl` lines 86,890 onward, counted at 0.55 / 0.30 | — |
| mix shares | `data/census/lyon_5m/{rigs,years,producers}.csv` | `share` |
| 1.45 m, IQR 0.60-3.26, p95 12.72, 14,547 | `runs/lyon/position_check.json` | `fields.submitted.cross_track`, `panos_not_near_a_street` |

The native-res archive on makelab2 (#146) covers the 86,889 panos of the 10 m set. The
87,516 added panos are archived next, as a follow-up step under #159.
