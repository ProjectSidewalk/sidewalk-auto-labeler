# Panoramax SD vs HD: does the lighter asset cost detections? (issue #161)

Panoramax's maintainers asked two things in #161: whether we really need the
full-resolution `hd` original for the computer-vision step, since the `sd` derivative
(2,048 px wide) is far lighter to serve, and whether we could pace downloads at about one
per second. The pacing is now the source's default, along with batched metadata and a
concurrency cap; see `sources/panoramax.py`. This study answers the first question with
the production detector, on a seeded sample of the Lyon run, under a rule written down
before any arm was scored.

## 1. Pre-registration (committed before the first scoring run)

Everything in this section was committed and pushed before `score` was run on any arm.
Section 2 onwards is filled in afterwards.

**Inputs.** The Lyon 5 m run's `results.jsonl` as archived on the lab file server
(`runs/lyon/results.jsonl` beside the native-res `panos/`; 174,405 records, sha256
`60e688b28b62f85e71f51c603dc1b6f4ff020348220c6cc7955891dcfef061ef`, verified when the
sample was drawn). Rigs (`camera_make camera_model`) and HD dimensions come from that file;
the HD pixels are the archive's JPEGs; the SD pixels are the live `sd` assets, fetched once
at 1 picture/s through the source's own limiter.

**Sample** (`sample`, seed 161). 500 detection-bearing panos (at least one stored
detection >= 0.30 after the camera-rig mask) and 100 empty ones, stratified by rig within
each stratum: every rig with at least 30 detection-bearing panos gets 30, the remaining
seats are proportional to the rig's pool (largest remainder); the empty stratum is
proportional. Within a rig, ids are sorted, shuffled with `random.Random(161)` (one
generator, strata in order), and the first k taken. The drawn sample is
`runs/_summary/panoramax_asset/sample.csv`.

**Arms.** Every arm's image is decoded to RGB and resized to 4,096 x 2,048 BILINEAR, which
is what `sources/panoramax._download_image` feeds the detector, then run through
`CurbRampDetector(batch_size=1)` with the default decode (argmax) and border rule
(exclude) -- the settings the Lyon run used (its records carry neither key).

- `hd`: the archived original.
- `sd`: the real `sd` asset.
- `sd_sim` (exploratory, outside the rule): the archived original resized to 2,048 px wide
  with LANCZOS and back up, which separates resampling from SD's JPEG re-encode.

**Sanity gate.** The `hd` arm on the archived bytes must reproduce the run's stored
detections, or the pipeline under test is not production's. Pass iff (a) at least 99% of
sampled panos have the identical set of rig-masked peaks >= 0.30 (exact normalized x, y),
ignoring any peak within 0.005 of 0.30 in either list, and (b) the largest |confidence
difference| over peaks at the same position, at least one of them >= 0.30, is <= 0.01. If
the gate fails the study stops, the default stays `hd`, and that is reported.

**Matcher.** RampNet's pano-frame geometry, as in `scripts/agree_rate.py`: x scaled by
1,024 and y by 512, x cyclic at the seam, greedy ascending-distance one-to-one, strictly
within 0.022 normalized-x units (~22.5 heatmap px, ~3 coarse cells). Detections are
rig-masked (`detectors.on_camera_rig`) throughout, as production ships them.

**Metrics** for `sd` and `sd_sim` against `hd`, at 0.30 (operational) and 0.55
(benchmark), pooled over the sample as drawn and per rig:

- strict recall-of-HD: HD peaks >= t matched one-to-one by arm peaks >= t, over HD peaks >= t;
- lenient recall-of-HD: HD peaks >= t matched by arm peaks >= the 0.1 storage floor;
- precision-vs-HD: matched over arm peaks >= t (strict matching);
- matched confidence delta (arm - HD): median, and the 90th percentile of |delta|;
- where matched pairs sit on the stride-8 heatmap grid: the share at <= 1 px Chebyshev
  offset (same coarse cell), at 7-9 px (one cell over), and other;
- the frame facts: how often the item's `sensor_array_dimensions` equal the archived HD
  file's dimensions, and how often the SD asset is 2,048 wide and 2:1.

95% intervals are percentile intervals of a pano-level (cluster) bootstrap, 2,000
resamples, seed 161: peaks on one pano are not independent, so panos are resampled.

**Default rule.** `sd` becomes the default for `--panoramax-asset` iff, for the `sd` arm at
0.30,

1. pooled strict recall-of-HD >= 0.95 AND pooled precision-vs-HD >= 0.95 (point
   estimates; the lower 95% bounds are reported beside them), AND
2. no rig with >= 30 sampled panos has a 95% UPPER bound below 0.90 on either.

Otherwise the default stays `hd` and the write-up says what SD would cost. `sd_sim`, the
0.55 tier, the lenient recall and the offset shares are reported and never read by the
rule. There is no Lyon ground truth in `../RampNet/benchmark/` (checked 2026-10-28), so
this study measures agreement with HD, not correctness; an SD-vs-GT arm is not possible yet.

**What the rule does not settle.** Agreement with HD is the question the maintainers asked
("is HD needed?"), not whether SD finds more or fewer *real* ramps. And the PS crops, the
PS pano store and the native-res archive all read HD whichever asset detection uses.
