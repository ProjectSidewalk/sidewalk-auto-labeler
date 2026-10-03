# The seam band: what `exclude_border` costs, on both Laurens arms (#130)

Issue [#130](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/130); PR
[#131](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/131), stacked on #129.
Measured 2026-10-03, CPU only, no network. Every number below re-derives from
`scripts/seam_band_130.py` and the files in `docs/figures/seam-band-130/data/`. Section 7 says
which inputs are not published, and where they live.

## Key takeaways

- **The seam band is real, but at Laurens it is smaller than the geometry predicts.** At the
  0.30 operating point, `keep` finds 15 more peaks than `exclude` on Laurens Mapillary (0.86% of
  all peaks, 95% CI [0.52, 1.41]) and 2 more on Laurens GSV (0.24% [0.06, 0.85]). If peaks were
  spread evenly in azimuth, the band (20 of 1,024 columns) would hold 1.95%. So at Laurens there
  are fewer ramps at the seam than the geometry predicts. The seam is the direction straight
  behind the pano's `camera_heading`. Bayonne measured 2.0% at 0.55 (3 of 150, one 125-pano
  bundle). Laurens at 0.55 is 0.42% (Mapillary) and 0.22% (GSV).
- **Most of the lost seam peaks are lost views, not lost ramps.** Of the 17 seam peaks at or
  above 0.30 (both arms), 9 join a site the `exclude` run already had as operational. 5 drop
  out in projection (above the horizon or beyond the 25 m range cap), 1 opens a second
  operational site next to an existing one, 1 promotes sub-threshold support to an operational
  site, and 1 starts a new site. That is **2 lost ramps at the operating point over the two arms**
  (1 promoted, 1 new). Operational sites go from 330 to 333 on Mapillary and from 254 to 255 on GSV.
- **There is no precision read.** No gained seam peak falls on a pano of either RampNet bundle,
  so none of them can be checked against reviewer marks.
- **RampNet does not wrap NMS at the seam.** A ramp that straddles the seam can give two peaks,
  one at x = 0 and one at x = 1023. `keep` follows RampNet exactly. That happened 3 times at
  0.30 on Mapillary (8 times at the 0.1 floor) and once at the floor on GSV.
- **The GSV run frame can't be measured from the archive.** The only GSV panos whose stored
  detections the archive heatmap reproduces are the 1,423 panos with no peaks at all. GSV
  counts therefore come from the archive frame (section 2).

## 1. What the band is, and what RampNet does

`detectors/decode.py::_peaks` calls `peak_local_max(min_distance=10)` with skimage's default
`exclude_border=True`. That zeroes heatmap rows and columns [0, 10) and [size - 10, size) before
it looks for maxima. On the 512x1024 heatmap, the left and right strips are the 360-degree seam:
20 columns, or 7.0 degrees of azimuth. The top and bottom strips are zenith and nadir, where no
ramp appears. In both Laurens arms, no top or bottom peak was gained at any threshold.

RampNet fixed its own extractor in RampNet#132 (RampNet f4c71c8). Today its single entry point,
`rampnet/subcell.py::detect_peaks`, calls `peak_local_max(..., exclude_border=False)` and
**does not suppress across the seam**. The sub-cell decode does not wrap either (`wrap_x=False`).
The labeler's `--border keep` (this PR) is exactly that rule, and `tests/test_border.py` pins it
against `detect_peaks`. The default stays `exclude`. `keep` is bound in the manifest and refused
in a mix, the same way `--decode` is (`CLAUDE.md`, SEAM BAND block).

## 2. Instrument check

The #111 detect pass saved the model's 64x128 coarse map for every pano of both runs. The head's
output is an exact bilinear x8 upsample of that map, so the map is the heatmap. `check` rebuilds
each heatmap and runs the production extractor (`exclude`, argmax, floor 0.1). It then compares
the result with the pinned run's stored detections, on exact pixel keys and on scores.

| arm | panos | keys exact, scores within 1e-6 | keys exact, scores within 1e-4 | rebuild = #111 pass's peaks | coarse sha256 = #111 pass |
|---|---:|---:|---:|---:|---:|
| laurens (Mapillary) | 4,495 | 1,932 (43.0%) | **4,492 (99.93%)** | 4,495 | 4,495 |
| laurens_gsv (GSV) | 2,137 | 1,423 (66.6%) | 1,423 (66.6%) | 2,137 | 2,137 |

**The tolerance (a deviation from the plan, which set 1e-6).** On Mapillary, all 2,563 panos
that missed the 1e-6 check have identical pixel keys and identical peak counts. Their scores
differ by at most 6.4e-5 (median 8e-6). The rebuild itself is exact: it matches the #111 pass's
own peaks on every pano. What differs is the stack. The original run (2026-08-31) and the #111
pass ran the same weights on different GPU and software stacks, and float32 kernels are not
bit-identical across stacks. So the check uses 1e-4, which is above that spread and four orders
of magnitude below any threshold used here. Pixel keys still have to match exactly. The 1e-6
result is the `reproduces_1e6` column of `laurens_check.csv`. The 3 Mapillary panos that still
fail differ in a peak below 0.30 or in a near-tied cell. All Mapillary counts below use the
**4,492 panos that reproduce (the run frame)**.

**GSV.** The GSV archive is a native-resolution re-fetch, and the run saw zoom-3 tiles. Every
GSV pano that reproduces (1,423) has zero stored detections. Of the 714 that don't, all but 49
have at least one peak. Their scores differ by up to 0.39 (median 0.07), and only 478 agree on
pixel keys at 0.30. The plan said to restrict GSV to the panos that reproduce. **That restriction
leaves nothing to measure:** the 1,423 empty panos gain 5 peaks below 0.30 and none at or above
it (row `laurens_gsv` below). So GSV is measured in the **archive frame**: both rules are applied
to the same #111-pass heatmap for every one of the 2,137 panos. That is a valid paired comparison
of the border rule on GSV imagery. It is not the run's own detections. The pooled rows use
Mapillary in the run frame and GSV in the archive frame.

## 3. What `keep` adds

Peaks under each rule, from the same heatmaps. "Gained" means present under `keep` and absent
under `exclude` at the same pixel key. No peak present under `exclude` was ever absent under
`keep`, and no pano reached the 50-peak cap. The share is gained / keep peaks: of all the peaks
`keep` finds, the fraction `exclude` dropped. This is the issue's "3 of 150", and it carries a
Wilson 95% interval. The plan asked for gained / exclude peaks. That is not a proportion, so it
gets no interval, but it is in `summary.json` (`gained_per_exclude_peak`) and differs from the
share only in the third decimal.

| arm | panos counted | tier | exclude peaks | keep peaks | gained | gained / keep [95% CI] | seam (L/R) | top/bottom | panos w/ seam gain | straddle pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| laurens | 4492 | 0.1 | 5721 | 5818 | 97 | 1.67% [1.37, 2.03] | 97 (35/62) | 0 | 89 | 8 |
| laurens | 4492 | 0.3 | 1730 | 1745 | 15 | 0.86% [0.52, 1.41] | 15 (7/8) | 0 | 12 | 3 |
| laurens | 4492 | 0.55 | 707 | 710 | 3 | 0.42% [0.14, 1.23] | 3 (2/1) | 0 | 2 | 1 |
| laurens_gsv (run frame) | 1423 | 0.1 | 0 | 5 | 5 | -- | 5 (4/1) | 0 | 5 | 0 |
| laurens_gsv (run frame) | 1423 | 0.3 | 0 | 0 | 0 | -- | 0 | 0 | 0 | 0 |
| laurens_gsv_archive | 2137 | 0.1 | 1609 | 1619 | 10 | 0.62% [0.34, 1.13] | 10 (6/4) | 0 | 9 | 1 |
| laurens_gsv_archive | 2137 | 0.3 | 847 | 849 | 2 | 0.24% [0.06, 0.85] | 2 (1/1) | 0 | 2 | 0 |
| laurens_gsv_archive | 2137 | 0.55 | 449 | 450 | 1 | 0.22% [0.04, 1.25] | 1 (0/1) | 0 | 1 | 0 |
| pooled | 6629 | 0.1 | 7330 | 7437 | 107 | 1.44% [1.19, 1.74] | 107 (41/66) | 0 | 98 | 9 |
| pooled | 6629 | 0.3 | 2577 | 2594 | 17 | 0.66% [0.41, 1.05] | 17 (8/9) | 0 | 14 | 3 |
| pooled | 6629 | 0.55 | 1156 | 1160 | 4 | 0.34% [0.13, 0.88] | 4 (2/2) | 0 | 3 | 1 |

**Against the geometry.** If peaks were spread evenly in azimuth, the seam band would hold 20/1024
= 1.95% of them. At the 0.1 floor Mapillary is near that (1.67% [1.37, 2.03]). At 0.30 and 0.55
both arms are well below it, and every interval at 0.30 excludes 1.95%. Confident ramps are
under-represented at the seam in Laurens. The seam is straight behind `camera_heading`
(`geo.ground_point_to_pano`: x = 0.5 + (bearing - heading) / 360). Why fewer confident ramps sit there
has not been examined. Two possible reasons: the rear view of the vehicle rig is partly occluded,
or ramps behind the camera are farther away on average. Neither has been checked.

**Straddling pairs** are two kept peaks on one pano with wrapped |dx| <= 10 px and |dy| <= 10 px.
Unwrapped, the peak finder never returns two peaks that close, so every such pair straddles the
seam: one ramp seen at both x = 0 and x = 1023. Mapillary has 3 pairs with both peaks at or above
0.30 (8 at the floor). GSV has 1, at the floor only. Both peaks of a pair are counted as gained
above, so at 0.30, 3 of the 15 Mapillary peaks are second copies of a ramp already in the count.
Fusion never merges such a pair: the two peaks come from one pano, and the same-pano
cannot-link keeps them out of one site (`fuse_sites.fuse`).

## 4. Lost ramps or lost views?

`world` writes each arm's records with every gained peak appended to the pano's detections. In the
run frame these are the stored detections. In the archive frame they are the #111 pass's
`exclude` peaks, so the two files differ only by the band. It then fuses both files with
identical parameters: `fuse_sites` defaults, `--camera-height-m 2.6` (the value in both committed
`runs/laurens{,_gsv}/sites_meta.json`) and `--min-confidence 0.30`. The full parameter set is in
`<arm>_world.json`. Each gained seam peak at or above 0.30 is classified by the keep-fusion site
it joined:

- **lost view**: the site holds a stored operational member. The ramp was already an operational
  site, and the seam peak is one more view of it.
- **split**: no stored operational member, but its stored sub-threshold support belonged to an
  operational site in the `exclude` fusion. The seam peak opened a second operational site
  beside an existing one (greedy association depends on order, and a new peak changes the order).
- **promoted**: stored support only, none of it operational in `exclude` either. A ramp the run
  had only as sub-threshold support becomes an operational site.
- **new site**: no stored member. A site the `exclude` run does not have.
- **not projected**: dropped before association (above the horizon, beyond the 25 m range cap,
  or on the camera rig).

| arm | seam peaks >= 0.30 | lost view | split | promoted | new site | not projected | operational sites exclude -> keep |
|---|---|---|---|---|---|---|---|
| laurens | 15 | 9 | 1 | 1 | 0 | 4 | 330 -> 333 (+3) |
| laurens_gsv_archive | 2 | 0 | 0 | 0 | 1 | 1 | 254 -> 255 (+1) |
| pooled | 17 | 9 | 1 | 1 | 1 | 5 | 584 -> 588 (+4) |

So 2 of the 17 seam peaks at the operating point are ramps the `exclude` run does not have as
operational sites: 1 promoted on Mapillary and 1 new on GSV. 9 are views of ramps it already has.
The Mapillary site count rises by 3, not 2. The split adds one, and one more comes from stored
detections re-associating once the new peaks change the greedy order (a stored sub-threshold
member moved between sites). The 4 Mapillary peaks that did not project sit at y = 0.51-0.52,
just below the horizon, so their ground range is beyond the 25 m cap. The four are two
straddling pairs.

**Ground truth.** No gained seam peak falls on a pano of the Laurens Mapillary bundle (94 panos)
or the Laurens GSV bundle (86 panos). Both bundles are from RampNet `origin/main` 459ea9e, and the
sha256 of each is in `<arm>_world.json`. So there is no precision read, and none is quoted. The
plan's floor for quoting one was 10 judged peaks. With 15 and 2 seam peaks per arm against bundles
of about 90 panos out of 4,495 and 2,137, a hit was unlikely from the start.

## 5. Wall-clock time (CPU, no cost)

All CPU, so no money was spent. `check`: 67 s (Mapillary) and 27 s (GSV) on makelab2 with 2 BLAS
threads. `peaks`: 200 s (Mapillary, run frame), 47 s (GSV, run frame) and 78 s (GSV, archive
frame), also on makelab2. `world`: about 1 s per arm on the desktop. `summary` / `verify`: under
1 s. A first `check` run at the default BLAS thread count used 45 cores for tiny matrix products,
so it was stopped and re-run with `OMP_NUM_THREADS=2`.

## 6. Decisions for Jon

Numbers to decide on. None of these is decided here.

1. **Option 1 (this PR: opt-in `--border keep`, default unchanged) or option 2 (change the
   default, and treat that as a campaign boundary for every city).** What is measured: at
   Laurens, `exclude` drops 0.86% (Mapillary) and 0.24% (GSV) of peaks at 0.30, below the
   geometric 1.95%. 9 of the 17 dropped seam peaks are views of ramps that are already
   operational sites, and 2 are ramps the fusion does not otherwise have. Bayonne's single
   bundle was 2.0% at 0.55. One rural town, and no city with multi-view gaps, has been measured.
2. **Whether to re-run any deployed city under `keep`.** A re-run under a new `--name` would add
   seam-band labels beside live `exclude` labels. `send_to_ps.py` refuses that without
   `--allow-mixed-border`, and `reinfer.py --write-band-file` refuses it outright. At Laurens it
   would add about 2 operational ramps per 6,600 panos.

## 7. Replication

What is **not** published, and where it lives:

- The 64x128 coarse maps (4,495 + 2,137 `.npy` files, 265 MB) are on makelab2 at
  `/homes/gws/jonf/decode111/coarse/{laurens,laurens_gsv}/`. They are not in git or on HF. A
  replicator can check a regenerated map against the committed `coarse_sha256` column of
  `<arm>_check.csv`, which is also the sha256 in the committed #111 detect outputs
  (`docs/figures/heatmap-grid/data/decode/decode_<arm>.jsonl.gz`). Regenerating a map needs the
  #111 GPU pass (`subcell_decode.py detect --source-resize`) over the archived panos.
- The pinned runs' `results.jsonl` (Mapillary sha256 `16c5a348...`, GSV `83f49aae...`) and the
  pano archives are at `/projects/makeabilitylab/sidewalk-auto-labeler/runs/{laurens,laurens_gsv}/`
  on makelab2. `results.jsonl` is gitignored here, as everywhere in this repo.
- `peaks.{run,archive}.jsonl` (both rules and both decodes, per pano) are untracked. Their sha256
  is in `<stem>_counts.json`.

What **is** committed, under `docs/figures/seam-band-130/data/`: `<arm>_check.csv` (per pano:
coarse sha256 and reproduction, with the diagnostic columns), `<stem>_gained.csv` (every gained
peak, both decodes, with edge and straddle flag), `<stem>_counts.json` (peak counts per tier, lost
peaks, straddle pairs), `<stem>_world.csv` / `.json` (classification, GT, fusion parameters,
bundle sha256s), and `summary.json`. `python scripts/seam_band_130.py verify` re-derives
`summary.json` byte for byte from those files, with no makelab2 access, and
`tests/test_seam_band_130.py` runs it.

Exact commands, in order (makelab2 Python `/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python`,
with `export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`; `runs/<arm>/` holds copies of the pinned
`results.jsonl` and `manifest.json`):

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
# (--benchmark-root holds benchmark/<split>/{verdicts.json,records.jsonl} from RampNet 459ea9e)
python scripts/seam_band_130.py world laurens --results runs/laurens/results.jsonl \
  --peaks runs/laurens/seam_band_130/peaks.run.jsonl --benchmark-root ../RampNet/benchmark
python scripts/seam_band_130.py world laurens_gsv --frame archive \
  --results runs/laurens_gsv/results.jsonl \
  --peaks runs/laurens_gsv/seam_band_130/peaks.archive.jsonl --benchmark-root ../RampNet/benchmark
python scripts/seam_band_130.py summary
python scripts/seam_band_130.py verify      # no makelab2, no network
```

## 8. What was not done

- **No GPU re-detection.** The Mapillary run frame reproduces from the saved maps (section 2). A
  GSV run-frame count would need the run's original zoom-3 pixels, which were never archived. A
  GSV re-run through the zoom-3 path (`reinfer.py --border keep`) would be a new forward pass on
  today's imagery, not the run's.
- **No other city.** Laurens is one rural town with two rigs. The rate at the seam probably
  depends on the rig and on how far apart panos are captured. Neither was varied here.
- **No reviewer pass on the gained seam peaks.** With no gained peak on a bundle pano, precision
  is unknown. The 17 peaks at or above 0.30 are listed in `<stem>_world.csv`, so they could be
  reviewed by hand.
