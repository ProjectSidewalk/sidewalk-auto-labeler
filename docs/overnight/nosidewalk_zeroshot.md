# Zero-shot NoSidewalk per block face from street-level panoramas (Seattle vs SDOT)

Study slug `nosidewalk_zeroshot`. Approved by Jon Froehlich 2026-11-27. Code:
`scripts/nosidewalk_zeroshot.py`; small data: `docs/overnight/data/nosidewalk_zeroshot/`.
Written by Claude Opus 5.5 (`claude-opus-5-5`).

## 1. Pre-registration (written 2026-11-27, after drawing the sample and before any pano was segmented or any score computed against SDOT)

**Question.** Can a zero-training, segmentation-based method decide sidewalk presence per block face from
street-level panoramas well enough to be worth pursuing as automatic NoSidewalk labeling?

**Silver ground truth.** The #5222 study's `sp_face` (Planning `reference/sidewalk-presence-sdot-study/results/sp_face.csv.gz`,
sha256 `3768d98a…6a259`, recorded in `population.json`): SDOT's `Sidewalks_CDL` units joined to PS Seattle street edges
and sides. `truth2 = present` (built coverage of the trimmed interior >= 0.2, i.e. #5222's full + partial) vs `none`
(built < 0.2 and unimproved >= 0.5). Side = left/right of the edge's digitized direction (positive cross product), the
`label_point.street_side` convention; verified on 2,538 of 2,538 faces against the PS API geometry before sampling.

**Unit.** Block face `(seattle, street_edge_id, street_side)`, keyed as `seattle:<edge>:<side>`.

**Eligibility.** #5222 `in_scope` (>= 1 completed audit and a binary SDOT verdict); `way_type` in residential, tertiary,
secondary, primary (trunk excluded; PS Seattle has no *_link / service / footway edges in `sp_face`); edge `status = open`
in the PS API now; edge length >= 40 m (so >= 16 m of mid-block interior). 31,974 eligible faces.

**Sample** (seed `20261127`, `draw_sample`). Strata, using the SDOT verdict only to stratify:

| stratum | definition | population faces | sampled faces | weight |
|---|---|---|---|---|
| D | discordant street: both faces eligible, verdicts differ (both faces sampled) | 2,748 (1,374 streets) | 180 (90 streets) | 15.27 |
| A | SDOT-absent face on a non-discordant street | 4,183 | 210 | 19.92 |
| P | SDOT-present face on a non-discordant street | 25,043 | 210 | 119.25 |

300 absent + 300 present faces on 504 streets. Population-weighted numbers use weight = population/sample per stratum
(Horvitz-Thompson). Dev split: a seeded 15% of sampled streets per stratum (91 faces); the other 509 faces are the
locked test set (A 179, D 152, P 178).

**Panos.** GSV coverage panos (z17 coverage tiles) whose foot on the PS edge is > 12 m from either end and within 10 m of
the centerline; ordered newest capture first, taken greedily with >= 8 m spacing along the edge, up to 5 per street.
Every selected pano gives a view of BOTH faces. Capture dates recorded. Secondary arm: Mapillary 360 images, same rule
(SfM position, compass from `computed_compass_angle`), coverage reported; not required for the primary.

**Method v0 (no training).** Per pano and side: a 90 deg FOV, 1024 px perspective view, pitched -30 deg, aimed along
the face's outward normal from the PS edge tangent (chord over +-10 m around the pano's foot; left = tangent - 90 deg),
not the vehicle heading. Mask2Former Swin-L Mapillary Vistas semantic (`facebook/mask2former-swin-large-mapillary-vistas-semantic`
@ `4772b6bf…`), run by `footway_segmentation.py segment` unchanged (no resize, class x mask scores, argmax). Ground
band: flat ground 2.5 m below the camera, 3-15 m from the camera along the face normal, |along-street| <= 15 m and
inside the view frustum, sampled on a 0.25 m ground grid (equal area per sample). Occluders (unobserved, not absent):
Car, Truck, Bus, Other Vehicle, Trailer, Caravan, Motorcycle, Bicycle, On Rails, Wheeled Slow, Boat, Person, Bicyclist,
Motorcyclist, Other Rider, Car Mount, Ego Vehicle, Bird, Ground Animal. A pano-side is `observed` if >= 50% of its band
samples are not occluders. Its score is the **walkway share** = (Sidewalk + Pedestrian Area + Curb Cut) / observed
samples. **Primary pooling:** the 75th percentile (numpy linear) of a face's observed pano-side shares; a face with
< 2 observed pano-sides is `unobserved`. Secondary (reported, not decisive): max / median / mean pooling; Sidewalk-only,
Curb, walkway+Curb shares; near (3-8 m) and far (8-15 m) half-bands; the Mapillary arm.

**Threshold (the one fitted number).** On dev observed faces: tau = the smallest pooled share at which the
population-weighted recall of SDOT-absent (score <= tau) reaches 0.67. Fixed before the test set is scored.

**Primary metrics (test set, observed faces).**
1. AUC, SDOT-absent vs present (population-weighted; unweighted also shown).
2. Precision of `absent` at tau (population-weighted, i.e. at the eligible population's 17% absent prevalence; the
   balanced-sample precision is also shown but is not used by the rule). Test recall at tau reported.
3. Discordant-street AUC: AUC over observed faces of stratum D (the only test of side resolution).
4. Coverage: share of sampled test faces that are observed.

95% CIs: stratified cluster bootstrap (streets resampled with replacement within stratum), 2,000 reps, seed 20261127.

**Controls.**
1. *Side-swap null*: each face scored with the other face's pooled evidence (same panos, other side). Note, fixed now
   before scoring: on a discordant street a perfectly side-resolving method gives swap AUC = 1 - AUC (it inverts), and a
   side-blind method gives 0.5 for both; so "collapses" is read as swap discordant AUC <= 0.55. A side-blind street-mean
   score (mean of both faces) is reported too.
2. *Way-type prior*: score = P(absent | way type) over eligible faces on non-test streets. "Beats the prior" = the
   bootstrap 95% CI of (method AUC - prior AUC) excludes 0 from below.
3. *OSM `sidewalk=*`* tags for the edge's OSM way (side mapped through `same_direction`), with tag coverage.
4. *PS crowd face rule* from #5222 (`m1`: >= 1 NoSidewalk label on the face; `m2_k2`: >= 2), on the same faces.

**Decision rule (advisors', kept; point estimates decide, CIs are reported).**
- CONTINUE if precision >= 0.80 at tau AND discordant AUC >= 0.80 AND it beats the way-type prior AND the side-swap null
  collapses (<= 0.55).
- STOP the zero-shot route if discordant AUC < 0.70, or |discordant AUC - swap discordant AUC| <= 0.05, or precision at
  tau < 0.70, or coverage < 50%.
- Otherwise INCONCLUSIVE. Each clause is reported.

**Exploratory slices** (test, observed): way type; SDOT `truth3` (full vs partial); stratum; SDOT unit offset from the
centerline (a set-back proxy: <8, 8-12, >=12 m); newest observed capture year; mean occlusion; arterial class. A seeded
gallery list (pano id, side bearing, scores; no images) of false-absent and false-present faces.

**Caveats known in advance.** SDOT is silver, not gold (geometry "general location", install dates sparse, the PS↔SDOT
join has its own failure modes per #5222). Seattle is a dense-sidewalk GSV city. GSV equirects are not
gravity-rectified (#113) and the band assumes a flat road and a 2.5 m camera, so band edges are approximate. No human
rater is involved yet.

### Amendments (dated; all made before any score against SDOT was computed)

- **A0, 2026-11-27, implementation only.** Mapillary returned 3.24 million 360 images in the 87 z14 tiles over the
  sample, so candidates are pre-selected from the tile's position and `captured_at` (the same newest-first, spaced rule
  with a looser 15 m corridor and up to 15 per street) before the Graph call that supplies the SfM position and compass;
  the final selection then applies the pre-registered rule. The GSV and Mapillary scans use a coarse lat/lng grid as a
  prefilter before exact projection (no change in what is selected).
- **A1, 2026-11-27, a secondary variant.** A truth-blind look at the pooled band class histogram (all 4,716 GSV views,
  no SDOT verdicts) showed Vegetation is 32% of all band samples. The primary keeps Vegetation as observed, as
  pre-registered; a secondary variant `walk_share_vegocc` also treats Vegetation as an occluder. It is not decisive.

<!-- RESULTS BELOW ARE WRITTEN AFTER THE PRE-REGISTRATION ABOVE -->

## 2. What ran

| step | where | result |
|---|---|---|
| `sample` | Mac | 600 faces on 504 streets (table above); `face_sample.csv`, `edges.geojson`, `population.json` |
| `gsv-views` | klone CPU job, 11 min | 1,087 z17 coverage tiles (0 failed), 5,757 mid-block candidates, **2,358 panos selected on all 504 streets** (0 download failures), 4,716 views. All `launch` (Google car) panos; newest capture 2025 for 1,847 of them (78%) |
| `mly-views` | Mac (token stays local) | 3.24 M 360 images in the 87 z14 tiles; 1,263 pre-selected; **530 panos on 125 of 504 streets**; 1,060 views |
| `segment` | klone GPU (`ckpt-all`, RTX 2080 Ti), `footway_segmentation.py segment --fp16 --batch-size 4` | 5,776 label maps, 5.6 img/s; model revision `4772b6bf…` |
| `bands` | klone (in the GPU job) | 3,545 band samples per view → `view_bands.jsonl.gz` |
| `osm-tags` | Mac, Overpass | 492 OSM ways, 736 of 1,008 faces tagged |
| `score` | Mac | `metrics.json`, `face_scores.csv`, `gallery.csv`; re-running from the committed files reproduces `metrics.json` byte for byte |

GPU time: 24 min wall on one GPU (17 + 4 min, plus a 3 min run that failed on macOS `._*` files), **0.40 GPU-hours**,
free `ckpt` partition.

## 3. Results (test set; the threshold was fixed on dev first)

tau (dev, 89 observed faces) = **0.0305**: a face is called `absent` when the 75th percentile of its pano-sides' walkway
share is <= 3%. 497 of 509 test faces observed; most observed faces have all 5 pano-sides observed (378) or 4 (65).

| metric | point | 95% CI (cluster bootstrap) | pre-registered bar |
|---|---|---|---|
| AUC, absent vs present (pop.-weighted) | **0.947** | 0.923-0.967 | |
| AUC, unweighted | 0.923 | 0.896-0.946 | |
| precision of `absent` at tau (pop.-weighted, 17% prevalence) | **0.924** | 0.854-0.973 | CONTINUE >= 0.80; STOP < 0.70 |
| recall of `absent` at tau (pop.-weighted) | 0.721 | 0.659-0.776 | target 0.67 |
| precision / recall, balanced sample (not used by the rule) | 0.941 / 0.720 | | |
| discordant-street AUC (145 faces) | **0.844** | 0.783-0.900 | CONTINUE >= 0.80; STOP < 0.70 |
| side-swap null, discordant AUC | **0.158** | 0.101-0.219 | collapses if <= 0.55 |
| discordant AUC minus swap null | 0.685 | 0.564-0.799 | STOP if <= 0.05 |
| side-blind street-mean score, discordant AUC | 0.500 | (by construction) | |
| side-swap null, all faces (pop.-weighted AUC) | 0.860 | 0.832-0.888 | (street-level signal, expected high) |
| way-type prior AUC | 0.591 | 0.557-0.626 | |
| method AUC minus prior AUC | **0.356** | 0.312-0.396 | beats if CI > 0 |
| coverage (observed / sampled test faces) | **0.976** | 0.959-0.990 | STOP < 0.50 |

Test-optimal precision at recall >= 0.67 (an oracle, not the rule): 0.927 at tau 0.021, i.e. the dev threshold lost
nothing.

### Verdict: **CONTINUE**

| clause | value | met? |
|---|---|---|
| precision >= 0.80 at recall >= 0.67 | 0.924 at recall 0.721 | yes |
| discordant-street AUC >= 0.80 | 0.844 | yes (but its CI lower bound, 0.783, is below 0.80) |
| beats the way-type prior | +0.356, CI 0.312-0.396 | yes |
| side-swap null collapses (<= 0.55) | 0.158 | yes (it inverts, as a side-resolving method must) |
| STOP: discordant AUC < 0.70 | 0.844 | no |
| STOP: within 0.05 of the swap null | 0.685 apart | no |
| STOP: precision < 0.70 | 0.924 | no |
| STOP: coverage < 50% | 97.6% | no |

### Context: the other evidence on the same test faces (pop.-weighted precision / recall of `absent`)

| source | faces | precision | recall | note |
|---|---|---|---|---|
| this method at tau | 497 observed | 0.924 | 0.721 | no labels, no training |
| PS crowd `m1` (>= 1 NoSidewalk label on the face) | 497 | 0.789 | 0.884 | every eligible face has >= 1 audit by construction |
| PS crowd `m2_k2` (>= 2 labels) | 497 | 0.918 | 0.779 | |
| OSM `sidewalk=*` | 359 tagged (70.5%) | 0.895 | 0.959 | Seattle's OSM sidewalk tagging very likely descends from SDOT data (the Seattle sidewalk import), so this is not an independent check; not verified here |

So on audited Seattle faces the zero-shot method is roughly as precise as two agreeing crowd labels, at somewhat lower
recall, without needing an audit.

### Secondary variants (test, not decisive)

| variant | coverage | AUC | disc. AUC | precision | recall |
|---|---|---|---|---|---|
| **primary: q75, walkway share, 3-15 m** | 0.976 | 0.947 | 0.844 | 0.924 | 0.721 |
| median pooling | 0.976 | 0.974 | 0.869 | 0.925 | 0.650 |
| mean pooling | 0.976 | 0.960 | 0.846 | 0.901 | 0.721 |
| max pooling | 0.976 | 0.903 | 0.788 | 0.714 | 0.756 |
| Sidewalk class only | 0.976 | 0.949 | 0.847 | 0.895 | 0.733 |
| walkway + Curb | 0.976 | 0.945 | 0.830 | 0.921 | 0.777 |
| Curb only | 0.976 | 0.785 | 0.587 | 0.559 | 0.630 |
| near half-band 3-8 m | 0.969 | 0.893 | 0.825 | 0.647 | 0.602 |
| far half-band 8-15 m | 0.976 | 0.907 | 0.768 | 0.678 | 0.783 |
| A1: Vegetation as occluder | 0.766 | 0.935 | 0.783 | 0.838 | 0.745 |

Each variant's tau was refit on dev the same way. Max pooling (the "one clear view counts" idea) is clearly worse than
q75: a single driveway or cross-street view is enough to lift an absent face. Median does slightly better on AUC; it is
reported, not adopted, since q75 was pre-registered. Curb alone carries little side information.

### Mapillary arm (secondary)

Mapillary 360 imagery reached 149 of the 600 sampled faces (125 streets), and 136 faces (22.7%) were observed. On the
135 faces observed by both sources (all splits; 45 on discordant streets): GSV AUC 0.930 vs Mapillary 0.797,
discordant AUC 0.870 vs 0.828; at the GSV tau, Mapillary's weighted precision is 0.19 (recall 0.79). Mapillary is
much weaker here; plausible causes, not tested: mixed rigs and heights (some captured from bikes or on foot, so the
camera may stand on the sidewalk and the 3-15 m band misses it), un-rectified tilt, and an untuned tau.

### Exploratory slices (test, observed; small cells are noisy)

- **Way type:** residential (410 faces) AUC 0.943, disc. 0.802, P 0.96 / R 0.71; secondary (41) P 0.62 / R 1.00;
  tertiary (40) AUC 0.999. **Arterials** reach high recall (0.90) but lower weighted precision (0.72): present faces on
  arterials are heavily weighted, so each false `absent` there costs a lot.
- **SDOT category** (`truth3`): partial faces (11) score lower than full ones (mean share 0.13 vs 0.19), as expected;
  1 of 11 partial and 10 of 240 full faces were called absent at tau (unweighted).
- **SDOT unit offset from the centerline** (set-back proxy): <8 m AUC 0.952 (381), 8-12 m 0.928 (100), >=12 m 0.984
  (16). No sign that set-back sidewalks escape the 15 m band in Seattle.
- **Capture year:** 418 of 497 observed faces have a 2025 newest pano (AUC 0.948); older years are too small to read.
- **Occlusion:** 432 faces < 20% occluded (AUC 0.95); only 7 test faces had mean occlusion >= 40% (4 unobserved).

### Failure modes (four gallery views inspected by eye; `gallery.csv` lists all 80 test errors' panos)

- **False `absent` (SDOT built, share ~ 0), 11 test faces:** asphalt walkways. One is an asphalt path behind a grass
  planting strip, another an asphalt strip at the curb behind wheel stops; Mask2Former labels both Road or leaves them
  out of Sidewalk. Seattle's inventory counts such surfaces as built.
- **False `present` (SDOT absent, share > tau), 69 test faces:** driveways and paved aprons, which Vistas labels Sidewalk;
  and the band's far ends: it reaches +-15 m along the street while panos need only be 12 m from an end, so a
  cross street's sidewalk or a corner plaza can enter the band near an intersection. Both are fixable without training
  (a narrower along-street window or a larger end buffer; a minimum along-street extent of walkway pixels to tell a
  sidewalk from a driveway crossing).
- Vegetation is 32% of all band samples (A1). Treating it as occluded loses 21% coverage and does not help.

## 4. Caveats

- **SDOT is silver.** The #5222 join has its own errors (side flips, partial faces, recently built sidewalks), and some
  of the 80 test errors may be SDOT's. No human rater has looked at a sample yet.
- **Seattle is easy in two ways:** dense, recent (2025) Google car coverage on every sampled street, and a city where
  sidewalks, when present, are mostly concrete and visible from the road. Cities with asphalt or gravel walkways, sparse
  or old coverage, or only Mapillary imagery will do worse (the Mapillary arm already does).
- **The precision is the eligible population's** (17% absent): weighted by stratum, residential through primary, edges
  >= 40 m, audited faces. Short edges, trunks, alleys and unaudited streets are outside it.
- **The discordant AUC passes on its point estimate;** its 95% CI (0.78-0.90) straddles 0.80. Side resolution is real
  (the swap null inverts to 0.16) but not yet shown to be >= 0.80 with confidence.
- The band assumes flat ground and a 2.5 m camera; GSV equirects are not gravity-rectified (#113).
- OSM is not an independent comparator in Seattle (see above).

## 5. Replication

All paths relative to the repo root. Outputs that are committed live in `docs/overnight/data/nosidewalk_zeroshot/`.

```bash
# inputs: the #5222 face table (Planning repo, origin/main) and the PS Seattle street edges
git -C ../Planning show origin/main:reference/sidewalk-presence-sdot-study/results/sp_face.csv.gz > sp_face.csv.gz
curl -o streets.geojson "https://sidewalk-sea.cs.washington.edu/v3/api/streets?filetype=geojson"
python scripts/nosidewalk_zeroshot.py sample --sp-face sp_face.csv.gz --streets streets.geojson
# (the committed population.json records both inputs' sha256; the streets API drifts)

# GPU host (klone): views, segmentation, bands. WORK is any scratch dir.
# scripts/nosidewalk_zeroshot_klone.sh wraps these as the two sbatch jobs that were run.
python scripts/nosidewalk_zeroshot.py gsv-views --work $WORK          # needs streetlevel; PROJ_DATA set
python scripts/nosidewalk_zeroshot.py mly-views --work $WORK          # needs MAPILLARY_ACCESS_TOKEN (.env)
python scripts/footway_segmentation.py segment --in $WORK/views --out $WORK/labels --fp16 --batch-size 4
python scripts/nosidewalk_zeroshot.py bands --work $WORK
python scripts/nosidewalk_zeroshot.py manifest --out $WORK/input_manifest.json $WORK/views $WORK/labels

# scoring, from the committed files alone (reproduces metrics.json exactly)
python scripts/nosidewalk_zeroshot.py osm-tags                         # optional; Overpass drifts
python scripts/nosidewalk_zeroshot.py score --sp-face sp_face.csv.gz
```

Coverage and imagery drift: the selected pano ids and capture dates are in `gsv_panos.csv` / `mly_panos.csv`, and
`input_manifest.json.gz` holds the sha256 of every rendered view and label map used.

Files in `docs/overnight/data/nosidewalk_zeroshot/`: `face_sample.csv` (sample, strata, weights, split),
`population.json`, `edges.geojson`, `gsv_panos.csv`, `mly_panos.csv`, `views_gsv.csv`, `views_mly.csv`,
`view_bands.jsonl.gz` (per-view band class histograms), `bands_meta.json` (Vistas id2label, model revision),
`osm_sidewalk_tags.csv`, `face_scores.csv`, `metrics.json`, `gallery.csv`, `input_manifest.json.gz`,
`*_views_summary.json`.
