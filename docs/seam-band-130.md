# The seam band: what `exclude_border` costs, on both Laurens arms (#130)

Issue [#130](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/130); PR
[#131](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/131), stacked on #129.
Measured 2026-10-03, CPU only, no network. Revised the same day after an independent review
(PR #131 review: the split classifier, the geometric expectation, the frame labels). Every
number below re-derives from `scripts/seam_band_130.py` and the files in
`docs/figures/seam-band-130/data/`. Section 7 says which inputs are not published, and where
they live.

**Two frames, never mixed silently.** Mapillary numbers are in the **run frame**: the run's own
stored detections, from 4,492 of 4,495 panos whose heatmap reproduces them. GSV numbers are in
the **archive frame**: both rules applied to the #111 pass's heatmap of the native-resolution
re-fetch. That is a valid paired test of the border rule on GSV imagery, but those are not the
deployed GSV campaign's detections (section 2). Every GSV number below carries that label.

## Key takeaways

- **The seam band is real, and smaller than the geometry predicts once the operating point
  applies.** At 0.30, `keep` finds 15 more peaks than `exclude` on Laurens Mapillary (0.86% of
  all peaks, 95% CI [0.52, 1.41]). On Laurens GSV (archive frame) it finds 2 more (0.24%
  [0.06, 0.85]). If peaks were spread evenly in azimuth, the band would hold 1.5625%.
  - Why 1.5625%: the heatmap is an exact x8 bilinear upsample, so a peak in the 20-pixel band
    can only come from coarse column 0 or 127. That is 2/128, or 5.6 degrees, not 20/1024
    (section 3).
  - At the 0.1 floor, Mapillary sits at that expectation (1.67% [1.37, 2.03]). At 0.30 and 0.55
    both arms are below it.
  - The seam is the direction straight behind the pano's `camera_heading`.
- **Most lost seam peaks are lost views, but the missing NMS across the seam creates duplicate
  sites.** On Mapillary (run frame), the 15 seam peaks at or above 0.30 break down as:
  - 8 join a site the `exclude` run already had as operational.
  - **1 promotes sub-threshold support to an operational site: a lost ramp.**
  - **2 open a second operational site beside an existing one.** One of these is the left half
    of a seam-straddling pair. The same-pano cannot-link kept it out of the site its right half
    joined, and it took two stored members with it. That is a duplicate site for one ramp.
  - 4 don't project: they are just below the horizon, past the 25 m range cap.
  - Operational sites: 330 -> 333 (+1 ramp, +2 sites beside existing ones).
  - On GSV (archive frame), of 2 seam peaks: 1 starts a new site and 1 doesn't project
    (254 -> 255).
- **There is no precision read.** No gained seam peak falls on a pano of either RampNet bundle.
- **RampNet does not wrap NMS at the seam, and neither does `keep`.** A ramp that straddles the
  seam gives two peaks, at x = 0 and x = 1020. Counts of pairs with both halves at or above 0.30:
  3 on Mapillary, and in 1 of them both halves projected and became two sites. At the 0.1 floor:
  8 pairs on Mapillary, and 1 on GSV (archive frame).

## 1. What the band is, and what RampNet does

`detectors/decode.py::_peaks` calls `peak_local_max(min_distance=10)` with skimage's default
`exclude_border=True`. That zeroes heatmap rows and columns [0, 10) and [size - 10, size) before
it looks for maxima. On the 512x1024 heatmap, the left and right strips are the 360-degree seam.
The top and bottom strips are zenith and nadir, where no ramp appears, and in both Laurens arms
no top or bottom peak was gained at any threshold.

The heatmap is an exact x8 bilinear upsample of a 64x128 coarse map (#111). Its maxima sit at
coarse-cell centres (columns 8c+3 or 8c+4) or on the clamped edge plateaus (columns 0-3 and
1020-1023), so the band holds exactly coarse columns 0 and 127. Every gained peak in the
committed `*_gained.csv` files is at column 0 or 1020.

RampNet fixed its own extractor in RampNet#132 (RampNet f4c71c8). Today its single entry point,
`rampnet/subcell.py::detect_peaks`, calls `peak_local_max(..., exclude_border=False)` and **does
not suppress across the seam**. The sub-cell decode does not wrap either (`wrap_x=False`).

The labeler's `--border keep` (this PR) is that rule, and `tests/test_border.py` pins its peak
pixels against `detect_peaks`. Two differences predate #130 and hold under both rules:
- The labeler keeps at most 50 peaks per pano. No Laurens pano reached that cap.
- It scores a peak by the raw heatmap value, where `detect_peaks(clip=True)` reports the clipped
  value. The two differ only above 1.0.

The default stays `exclude`. `keep` is bound in the manifest and refused in a mix, the same way
`--decode` is (`CLAUDE.md`, SEAM BAND block).

## 2. Instrument check

The #111 detect pass saved the model's 64x128 coarse map for every pano of both runs. `check`
rebuilds each heatmap from its map and runs the production extractor (`exclude`, argmax, floor
0.1). It then compares the result with the pinned run's stored detections, on exact pixel keys
and on scores.

| arm | panos | keys exact, scores within 1e-6 | keys exact, scores within 1e-4 | rebuild vs #111 pass's peaks: keys equal, max \|d score\| | coarse sha256 = #111 pass |
|---|---:|---:|---:|---:|---:|
| laurens (Mapillary) | 4,495 | 1,932 (43.0%) | **4,492 (99.93%)** | 4,495, 1.79e-7 | 4,495 |
| laurens_gsv (GSV) | 2,137 | 1,423 (66.6%) | 1,423 (66.6%) | 2,137, 1.19e-7 | 2,137 |

**The tolerance is 1e-4, which deviates from the plan's 1e-6.**
- Of the 2,563 Mapillary panos that miss the 1e-6 check, 2,560 have identical pixel keys and
  peak counts, with scores differing by at most 6.4e-5 (median 8e-6). The other 3 fail at 1e-4
  too: they differ in a key below 0.30, or in a near-tied cell at or above it.
- The rebuild itself is exact. Against the #111 pass's own peaks it has no key mismatch, and its
  largest score difference is 1.79e-7 (`max_dscore_detect_pass`, `<arm>_check.csv`).
- So the gap is between the original run (2026-08-31) and the #111 pass. They ran the same
  weights on different GPU and software stacks, and float32 kernels are not bit-identical across
  stacks.
- 1e-4 is above that spread and four orders of magnitude below any threshold used here. Pixel
  keys still have to match exactly. The 1e-6 result is the `reproduces_1e6` column.

**GSV.** The GSV archive is a native-resolution re-fetch, and the run saw zoom-3 tiles.
- Every GSV pano that reproduces (1,423) has zero stored detections.
- Of the 714 that don't, all but 49 have at least one peak. Their scores differ by up to 0.39
  (median 0.07), and only 478 agree on pixel keys at 0.30.
- The plan said to restrict GSV to the reproducing panos. **That restriction leaves nothing to
  measure:** the 1,423 empty panos gain 5 peaks below 0.30 and none at or above it (row
  `laurens_gsv` below).
- So GSV is measured in the archive frame, on all 2,137 panos.

## 3. What `keep` adds

Peaks under each rule, from the same heatmaps. "Gained" means present under `keep` and absent
under `exclude` at the same pixel key. No peak present under `exclude` was ever absent under
`keep`, and no pano reached the 50-peak cap (both checked by a test on the committed counts).

The share is gained / keep peaks: of all the peaks `keep` finds, the fraction `exclude` dropped.
This is the issue's "3 of 150", and it carries a Wilson 95% interval. The plan asked for gained /
exclude peaks. That is not a proportion, so it gets no interval, but it is in `summary.json`
(`gained_per_exclude_peak`).

The **pooled** rows add Mapillary (run frame) to GSV (archive frame). They are a property of the
border rule on Laurens imagery, not of any deployed campaign.

| arm | frame | panos counted | tier | exclude peaks | keep peaks | gained | gained / keep [95% CI] | seam (L/R) | top/bottom | panos w/ seam gain | straddle pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laurens | run | 4492 | 0.1 | 5721 | 5818 | 97 | 1.67% [1.37, 2.03] | 97 (35/62) | 0 | 89 | 8 |
| laurens | run | 4492 | 0.3 | 1730 | 1745 | 15 | 0.86% [0.52, 1.41] | 15 (7/8) | 0 | 12 | 3 |
| laurens | run | 4492 | 0.55 | 707 | 710 | 3 | 0.42% [0.14, 1.23] | 3 (2/1) | 0 | 2 | 1 |
| laurens_gsv | run | 1423 | 0.1 | 0 | 5 | 5 | -- | 5 (4/1) | 0 | 5 | 0 |
| laurens_gsv | run | 1423 | 0.3 | 0 | 0 | 0 | -- | 0 | 0 | 0 | 0 |
| laurens_gsv | archive | 2137 | 0.1 | 1609 | 1619 | 10 | 0.62% [0.34, 1.13] | 10 (6/4) | 0 | 9 | 1 |
| laurens_gsv | archive | 2137 | 0.3 | 847 | 849 | 2 | 0.24% [0.06, 0.85] | 2 (1/1) | 0 | 2 | 0 |
| laurens_gsv | archive | 2137 | 0.55 | 449 | 450 | 1 | 0.22% [0.04, 1.25] | 1 (0/1) | 0 | 1 | 0 |
| pooled | mixed | 6629 | 0.1 | 7330 | 7437 | 107 | 1.44% [1.19, 1.74] | 107 (41/66) | 0 | 98 | 9 |
| pooled | mixed | 6629 | 0.3 | 2577 | 2594 | 17 | 0.66% [0.41, 1.05] | 17 (8/9) | 0 | 14 | 3 |
| pooled | mixed | 6629 | 0.55 | 1156 | 1160 | 4 | 0.34% [0.13, 0.88] | 4 (2/2) | 0 | 3 | 1 |

**Against the geometry.** Under a uniform azimuth, the band (coarse columns 0 and 127) holds 2/128
= 1.5625% of peaks.
- At the 0.1 floor, Mapillary (run frame) sits at that value: its interval [1.37, 2.03] contains
  it. GSV (archive frame) is below it (0.62% [0.34, 1.13]).
- At 0.30 and 0.55, every interval on both arms excludes 1.5625%. So in Laurens, peaks that pass
  the operating point are under-represented at the seam, while low-confidence ones are not.
- The seam is straight behind `camera_heading` (`geo.ground_point_to_pano`:
  x = 0.5 + (bearing - heading) / 360).
- Why confident peaks are rarer there has not been examined. Rig occlusion behind the camera, or
  ramps behind the camera being farther away on average, are untested guesses.

**Straddling pairs** are two kept peaks on one pano with wrapped |dx| <= 10 px and |dy| <= 10 px.
Without wrapping, the peak finder never returns two peaks that close, so every such pair straddles
the seam: one ramp seen at both x = 0 and x = 1020. Both peaks of a pair count as gained above.
The same-pano cannot-link keeps the two halves out of one site, so a pair whose halves both
project makes two sites (section 4).

## 4. Lost ramps or lost views?

`world` builds a second results file for each arm: every gained peak is appended to the pano's
detections. In the run frame those are the stored detections. In the archive frame they are the
#111 pass's `exclude` peaks. Either way, the two files differ only by the band.

It then fuses both files with identical parameters: `fuse_sites` defaults, camera height 2.6 m,
min confidence 0.30. The full parameter set is in `<stem>_world.json`. The 2.6 m is the value
the earlier local Laurens fusions used (`runs/laurens{,_gsv}/sites_meta.json`). Those files are
gitignored run state, so `<stem>_world.json` is where a replicator reads it.

Let S be the keep-fusion site a gained seam peak at or above 0.30 joined, and E the `exclude`
operational sites that held S's stored members. Each such site in E has one **continuation** in
the keep fusion: the keep site that holds most of its stored operational members. Each gained
peak is then classified as:

- **lost view**: S is the continuation of a site in E. The seam peak is one more view of a ramp
  the run already had.
- **split**: E is non-empty, but S is not the continuation of any site in E. S is a second
  operational site beside a ramp the run had: a duplicate of it, or a neighbouring ramp.
- **promoted**: S has stored members, but none was in an operational `exclude` site. A ramp the
  run had only as sub-threshold support becomes an operational site.
- **new site**: S has no stored member.
- **not projected**: the peak was dropped before association (below the horizon past the 25 m
  range cap, or on the camera rig).

| arm (frame) | seam peaks >= 0.30 | lost view | split | promoted | new site | not projected | operational sites exclude -> keep | straddle pairs with both halves fused |
|---|---|---|---|---|---|---|---|---|
| laurens (run) | 15 | 8 | 2 | 1 | 0 | 4 | 330 -> 333 (+3) | 1 |
| laurens_gsv (archive) | 2 | 0 | 0 | 0 | 1 | 1 | 254 -> 255 (+1) | 0 |

**Mapillary (run frame).** The +3 is fully accounted for:
- The **promotion** (`1424244379059068`) is a ramp the operational sites did not have.
- The first **split** (`1496478694911820`) opened an operational site from sub-threshold support
  whose operational site still exists.
- The second **split** is the left half of the straddling pair on `1466581971069523`. Its right
  half (0.708) joined the existing operational site. The left half (0.666) was kept out of it by
  the same-pano cannot-link, and it took stored members `845899741254441` (0.567, operational)
  and `1400380601441935` (0.394) out of that site, leaving **two operational sites for one ramp**.
- So, on Mapillary in the run frame, `keep` adds 1 ramp and 2 sites beside existing ones, at
  least one of them certainly a duplicate.
- The lack of NMS across the seam therefore does real damage after fusion. Of the 3 straddling
  pairs at or above 0.30, the 1 whose halves both projected made two sites.
- The 4 unprojected Mapillary peaks sit at y = 0.508-0.523, just below the horizon. At 2.6 m
  that is 35-106 m of ground range, past the default 25 m cap that `fuse_sites` applies to every
  detection. The four are two straddling pairs.

**GSV (archive frame).** 1 new site and 1 unprojected peak, out of 2. The pano with the new site
(`tM1gQR6oOHTcwLItwgD3nA`) has a single stored detection at 0.299 in the deployed run, so this
count describes the border rule on the re-fetched imagery, not a ramp the deployed GSV campaign
lost.

**Ground truth.** No gained seam peak falls on a pano of the Laurens Mapillary bundle (94 panos)
or the Laurens GSV bundle (86 panos). Both bundles are from RampNet `origin/main` 459ea9e, and the
sha256 of each is in `<stem>_world.json`. So there is no precision read, and none is quoted. The
plan's floor for quoting one was 10 judged peaks.

## 5. Wall-clock time (CPU, no cost)

All CPU, so no money was spent.
- `check`: 60-67 s (Mapillary) and 27-53 s (GSV) on makelab2, 2 BLAS threads, over three runs.
- `peaks`: 200 s (Mapillary, run frame), 47 s (GSV, run frame) and 78 s (GSV, archive frame), on
  makelab2.
- `world`: about 1 s per arm on the desktop. `summary` / `verify`: under 1 s.
- A first `check` at the default BLAS thread count used 45 cores for tiny matrix products. I
  stopped it and re-ran with `OMP_NUM_THREADS=2`.

## 6. Decisions for Jon

These are the numbers to decide on. None of these is decided here.

1. **Option 1, or option 2?** Option 1 is this PR: `--border keep` as an opt-in, default
   unchanged. Option 2 is to change the default and treat that as a campaign boundary for every
   city. Measured at 0.30:
   - Mapillary (run frame): `exclude` drops 0.86% of peaks, below the 1.5625% geometric share.
     `keep` would add 1 operational ramp (a promotion), 8 extra views of existing sites, and
     **2 extra operational sites beside existing ones, at least one a duplicate made by the
     missing NMS across the seam**.
   - GSV (archive frame, not the deployed campaign): 0.24%, and 1 new site.
   - Bayonne's single bundle was 2.0% at 0.55.
   - Laurens is one rural town, and no city where multi-view coverage is thin has been measured.
   - If the default changes, NMS across the seam (one peak per straddling pair) is worth deciding
     with it. RampNet does not do it today.
2. **Whether to re-run any deployed city under `keep`.**
   - `send_to_ps.py` refuses a `keep` file without `--allow-mixed-border` when a live `exclude`
     campaign on the same endpoint is recorded in the same run directory or in a sibling
     `runs/*/` directory. So a re-run under a new `--name` is caught.
   - The guard keys on the endpoint, not the city's area. That holds because production runs
     one server per city; a shared dev or localhost host would make one city's campaign block
     another's. A sibling record that can't be read is skipped with a warning on stderr.
   - `reinfer.py --write-band-file` refuses the mix outright.
   - At Laurens (Mapillary, run frame), a re-run would add about 1 operational ramp and 2
     near-duplicate sites per 4,500 panos.

## 7. Replication

What is **not** published, and where it lives:

- **Coarse maps.** 4,495 + 2,137 `.npy` files (265 MB) on makelab2 at
  `/homes/gws/jonf/decode111/coarse/{laurens,laurens_gsv}/`, not in git or on HF.
  - A replicator can check a regenerated map against the committed `coarse_sha256` column of
    `<arm>_check.csv`. That is also the sha256 in the committed #111 detect outputs
    (`docs/figures/heatmap-grid/data/decode/decode_<arm>.jsonl.gz`).
  - Regenerating a map needs the #111 GPU pass (`subcell_decode.py detect --source-resize`)
    over the archived panos.
- **Pinned runs.** `results.jsonl` (Mapillary sha256 `16c5a348...`, GSV `83f49aae...`) and the
  pano archives are at `/projects/makeabilitylab/sidewalk-auto-labeler/runs/{laurens,laurens_gsv}/`
  on makelab2. `results.jsonl` is gitignored here, as everywhere in this repo.
- **Peak files.** `peaks.{run,archive}.jsonl` (both rules and both decodes, per pano) are
  untracked. Their sha256 is in `<stem>_counts.json`.

What **is** committed, under `docs/figures/seam-band-130/data/` (pinned LF by `.gitattributes`):
- `<arm>_check.csv`: per pano, the coarse sha256 and reproduction, plus diagnostic columns.
- `<stem>_gained.csv`: every gained peak under both decodes, with edge and straddle flag.
- `<stem>_counts.json`: peak counts per tier, lost peaks, straddle pairs.
- `<stem>_world.csv` / `.json`: classification, straddle flag, GT, fusion parameters, bundle
  sha256s.
- `summary.json`.

`python scripts/seam_band_130.py verify` re-derives `summary.json` byte for byte from those files,
with no makelab2 access, and `tests/test_seam_band_130.py` runs it.

Exact commands, in order. On makelab2 the Python is
`/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python`, run with
`export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`. `runs/<arm>/` holds copies of the pinned
`results.jsonl` and `manifest.json`.

```bash
# makelab2: instrument check, then both rules per pano
for a in laurens laurens_gsv; do
  python scripts/seam_band_130.py check $a --results runs/$a/results.jsonl \
    --coarse-dir /homes/gws/jonf/decode111/coarse/$a \
    --decode-file /homes/gws/jonf/decode111/decode_$a.jsonl
  python scripts/seam_band_130.py peaks $a --results runs/$a/results.jsonl \
    --coarse-dir /homes/gws/jonf/decode111/coarse/$a --frame run
done
python scripts/seam_band_130.py peaks laurens_gsv --results runs/laurens_gsv/results.jsonl \
  --coarse-dir /homes/gws/jonf/decode111/coarse/laurens_gsv --frame archive
# anywhere: copy back the data files and the two peaks files world needs, then
# (--benchmark-root holds <split>/{verdicts.json,records.jsonl} from RampNet 459ea9e)
python scripts/seam_band_130.py world laurens --results runs/laurens/results.jsonl \
  --peaks runs/laurens/seam_band_130/peaks.run.jsonl --benchmark-root ../RampNet/benchmark
python scripts/seam_band_130.py world laurens_gsv --frame archive \
  --results runs/laurens_gsv/results.jsonl \
  --peaks runs/laurens_gsv/seam_band_130/peaks.archive.jsonl --benchmark-root ../RampNet/benchmark
python scripts/seam_band_130.py summary
python scripts/seam_band_130.py verify      # no makelab2, no network
```

## 8. What was not done

- **No GPU re-detection.** The Mapillary run frame reproduces from the saved maps (section 2).
  - A GSV run-frame count would need the run's original zoom-3 pixels, which were never archived.
  - A GSV re-run through the zoom-3 path (`reinfer.py --border keep`) would be a new forward pass
    on today's imagery, not the run's.
- **No other city.** Laurens is one rural town with two rigs. The seam rate probably depends on
  the rig and on how far apart panos are captured, and neither was varied here.
- **No reviewer pass on the gained seam peaks.** With no gained peak on a bundle pano, precision
  is unknown. The 17 peaks at or above 0.30 are listed in `<stem>_world.csv`.
- **No NMS across the seam.** RampNet does not do it either. It is a decision for option 2
  (section 6), not part of this PR.
