# Sidewalks from aerial imagery (Tile2Net), projected into panos (issue #104)

A study, not production. Nothing here is wired into the pipeline, and no dependency was
added to `requirements.txt`. Tile2Net runs in its own environment on the GPU host.
`scripts/aerial_sidewalks.py` reads its polygons as data.

**Bend is a RampNet training city.** Every Bend number built on RampNet GT or on the
run's detections carries that caveat. Paterson is the held-out city. The Bend inventory
(#79) is image-free.

The rules were pre-registered on #104 before any GT-joined number existed:
[#104 (comment)](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/104#issuecomment-5880437702).
They are committed as code in `q1_verdict`, `q3_verdict`, `q4_verdict` and the constants at
the top of the script
([54e8852](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/commit/54e8852)).

## Answers

| question | pre-registered rule | answer |
|---|---|---|
| Q1 mask quality | USABLE iff ≥ 0.90 of Bend inventory ramps lie within 2 m of a sidewalk/crosswalk polygon | **NOT USABLE**: 0.703 (n = 12,504) |
| Q2 projection error | none (descriptive) | median 3.4 px (Paterson) and 3.6 px (Bend, training city) at `auto`, in 1024×512 heatmap px |
| Q3 corner anchors | an arm must cut frag 5 m by ≥ 0.03 in both cities, with coverage and dual separation each down ≤ 1 pt | **NO**: all four (base, arm) cells fail |
| Q4 precision signal | FILTER iff (i) ≥ 0.90 of true detections on surface in every city and (ii) pooled False − True off-surface share ≥ 0.15 with CI > 0; < 30 False → not established | **NOT ESTABLISHED (underpowered)**: only 17 False detections pooled. Clause (i) also fails in Bend (0.70). |

In short, Tile2Net's polygons fit the streets well where they exist. In Paterson they
contain or touch nine in ten ramps, and their edges land about 3 px from a reviewer's box
centre. But Bend's sidewalks are too often simply missing from them. Corner anchors do not
fix fragmentation without merging dual ramps. The precision question cannot be answered at
this GT size.

## Q1: mask quality

Share of ramps within d of a sidewalk or crosswalk polygon (0 m inside). "Chance" is the
same points displaced 10 m in a seeded random direction.

| set | n | inside | ≤ 1 m | ≤ 2 m | ≤ 3 m | chance ≤ 2 m |
|---|---:|---:|---:|---:|---:|---:|
| Paterson GT ramps | 323 | 0.653 | 0.867 | **0.913** | 0.944 | 0.396 |
| Bend GT ramps (training city) | 298 | 0.373 | 0.604 | **0.675** | 0.711 | 0.326 |
| Bend inventory, all in area | 12,504 | 0.464 | 0.667 | **0.703** | 0.726 | 0.217 |
| Bend inventory, visible pool (≤ 20 m from a pano) | 12,065 | 0.474 | 0.682 | 0.719 | 0.743 | 0.222 |

- **The failure is missing polygons, not misplaced ones.** Most ramps sit inside or within
  1 m of a polygon. The rest are far away: Bend inventory ramps beyond 1 m have a median
  distance of 12.8 m (p90 122 m). The gallery shows the same thing. Where Tile2Net drew a
  Bend sidewalk, the projected edge hugs the curb. Plainly visible sidewalks on other
  corners have no polygon at all.
- **Paterson would pass the bar that Bend fails.** In Paterson, 0.91 of GT ramps are within
  2 m, against a 0.40 chance floor. That is a dense Northeastern grid, like the cities in
  Tile2Net's paper. Bend is suburban and high-desert, with a 2019 image.
- **By capture year (Bend GT, descriptive):** 2024 panos 0.68 (n = 245), 2018 panos 0.94
  (n = 16), 2019 panos 0.62 (n = 13). The newer-pano bins read lower, which fits sidewalks
  built after the 2019 flight, but the older bins are too small to separate that from domain
  shift. Paterson (2020 orthos) shows no such pattern: 0.94 for 2025 panos (n = 133), 0.84
  for 2024 panos (n = 69).
- Sidewalk polygons alone, without crosswalks, reach 0.86 (Paterson) and 0.63 (Bend)
  within 2 m. Tile2Net emitted no footpath polygons in either city, so "walkable" (Q4)
  equals sidewalk ∪ crosswalk here.

Tables: `runs/<city>/aerial/report_q1.md`, `q1.csv`, `gt_ramps.csv`. Figure:
`figures/aerial-sidewalks/q1_distance_cdf.png`.

## Q2: projection error

The measure is the distance from each reference mark to the nearest projected
sidewalk/crosswalk edge, in 1024×512 heatmap px (1 px = 0.35°), in judged panos whose 25 m
disk is fully covered.

| city | height | reference | n | px p50 | px p90 | share ≤ 5 px | dy p50 |
|---|---|---|---:|---:|---:|---:|---:|
| Paterson | auto | all | 395 | 3.4 | 14.6 | 0.63 | +0.3 |
| Paterson | auto | box centre | 109 | 3.9 | 18.9 | 0.60 | +0.2 |
| Paterson | 2.6 m | all | 395 | 4.6 | 17.0 | 0.52 | +1.4 |
| Paterson | 2.6 m | box centre | 109 | 5.1 | 24.8 | 0.49 | +1.6 |
| Bend (training city) | auto | all (peaks + missed clicks) | 327 | 3.6 | 83.0 | 0.59 | −0.4 |
| Bend (training city) | 2.6 m | all | 327 | 3.7 | 83.0 | 0.56 | +0.0 |

- **Where a polygon exists, the projected edge lands a few pixels from the ramp.** The
  metric mixes mask error with height, heading and GPS error by construction, so it is an
  upper bound on the projection error alone.
- **Paterson, the held-out city with reviewer boxes, prefers `auto` over 2.6 m** (3.9 vs
  5.1 px at the box centre). The 2.6 m edges sit about 1 px lower, i.e. nearer. That
  agrees with #79's per-rig heights: 2.6 m runs ranges long on the 2025 rig, which is 48%
  of Paterson. Bend (84% 2024, an older 2.5 m rig) barely moves.
- Bend's p90 of 83 px is Q1's missing polygons again: the nearest edge is on another
  street.
- The galleries (20 seeded panos per city, `runs/<city>/aerial/gallery/index.html`,
  untracked because they hold pixels) show `auto` edges following curbs closely.

## Q3: corner anchors vs fragmentation

- **Anchors:** 5,815 in Paterson; 1,833 in Bend, where polygon contacts alone were
  available because Tile2Net's network step crashed (below).
- **How many ramps have an anchor nearby:** 0.54 of Paterson's GT pool ramps have an anchor
  within 3 m, but only 0.12 of Bend's (median distance 115 m). Bend paints few crosswalks.
- **Reproduction check:** the base rows reproduce the committed #106 offline `arms.csv`
  exactly.

| city | base | arm | clusters | coverage | frag 5 m | dual kept | same-pano pairs |
|---|---|---|---:|---:|---:|---:|---:|
| Paterson | fusion | base | 7,172 | 0.944 | 0.128 | 85/100 | 0 |
| Paterson | fusion | anchored | 7,172 | 0.944 | 0.141 | 85/100 | 0 |
| Paterson | fusion | merge-at-anchor | 6,197 | 0.777 | 0.076 | 31/100 | 1,768 |
| Paterson | ps @ 7.5 m | base | 8,383 | 0.944 | 0.298 | 85/100 | 0 |
| Paterson | ps @ 7.5 m | anchored | 8,383 | 0.941 | 0.313 | 84/100 | 0 |
| Paterson | ps @ 7.5 m | merge-at-anchor | 7,114 | 0.805 | 0.142 | 39/100 | 1,755 |
| Bend (training city) | fusion | base | 14,058 | 0.950 | 0.145 | 29/34 | 0 |
| Bend (training city) | fusion | anchored | 14,058 | 0.940 | 0.143 | 29/34 | 0 |
| Bend (training city) | fusion | merge-at-anchor | 13,851 | 0.930 | 0.134 | 26/34 | 495 |
| Bend (training city) | ps @ 7.5 m | base | 14,650 | 0.943 | 0.206 | 27/34 | 0 |
| Bend (training city) | ps @ 7.5 m | anchored | 14,650 | 0.940 | 0.204 | 27/34 | 0 |
| Bend (training city) | ps @ 7.5 m | merge-at-anchor | 14,354 | 0.930 | 0.173 | 24/34 | 484 |

- **Snapping (`anchored`) cannot merge anything.** Two clusters snapped to one corner are
  still two clusters, so frag 5 m does not move (±0.015).
- **Merging at the anchor does cut fragmentation, by merging real neighbours.** Paterson's
  fusion frag 5 m halves, but only by uniting the two ramps of a corner. Dual separation
  collapses from 85/100 to 31/100, coverage drops 17 pts, and 1,768 same-pano pairs land in
  one cluster. A corner has several ramps and one crosswalk end per crossing, so "one anchor
  per ramp" does not hold at this density.
- The polygon-contact-only sensitivity (`q3_polygon_anchors.csv`) reads the same.

## Q4: precision signal

Scope: operational detections on judged panos, off-surface = more than 2 m from every
walkable polygon.

| city | true on surface | True off | False off | gap (False − True) | 95% CI | n True / False |
|---|---:|---:|---:|---:|---|---|
| Paterson | **0.927** | 0.073 | 0.167 | +0.094 | −0.098 to +0.545 | 233 / 6 |
| Bend (training city) | 0.701 | 0.299 | 0.455 | +0.155 | −0.132 to +0.492 | 244 / 11 |
| pooled | | | | +0.164 | −0.065 to +0.435 | 477 / 17 |

- **Only 17 detections were judged false** in the two benchmarks. That is below the
  pre-registered 30, so the answer is NOT ESTABLISHED, whatever the point estimate.
- **Clause (i) fails in Bend** (0.70), because Q1's missing polygons put three in ten real
  ramps "off surface". Even with more False detections, a Bend-quality mask would throw
  away real ramps.
- In Paterson, where the mask is good, True detections are on surface 93% of the time. The
  signal is not ruled out there. It needs a larger false set to test.
- Paterson's 0.30–0.55 band (17 detections, unjudged) is 0.29 off surface. Bend's run
  predates the storage floor and has no band.
- Tile2Net has no occlusion class, so the tree/awning share cannot be measured. In the
  gallery panos inspected by eye, the missing Bend sidewalks were not under canopy, which
  points at the aerial model or the imagery date. That is a spot check, not a measurement.

## What this cannot say

- **Two cities, one of them a RampNet training city.** Paterson is the only held-out GT,
  and only Bend has an inventory. The Q1 rule was read on Bend because that is where the
  image-free reference exists. Paterson's GT would pass the same 0.90 bar, but GT ramps are
  not an inventory.
- **Imagery dates:** Bend 2019, Paterson 2020, against mostly 2024–2025 panos. Neither the
  imagery date nor Tile2Net's training domain can be separated from the Bend failure with
  one city of each kind.
- **Q2 mixes mask error with projection error.** It bounds the projection error from above
  and does not isolate it.
- **Q3's anchors come from crosswalk polygons.** Where crossings are unpainted (most of
  Bend) there are none. A curb-line or curb-return anchor would need a different
  extraction.
- **Default settings only:** Tile2Net ran with its default checkpoint and settings at zoom
  19. No zoom-20 run, no fine-tuning.

## Method

### Tile2Net

[Tile2Net](https://github.com/VIDA-NYU/tile2net) (BSD-3) segments orthorectified aerial
tiles into sidewalk, crosswalk, road and footpath polygons. It also writes a pedestrian
network, in WGS 84 as GeoParquet. Citation: Hosseini, M., Sevtsuk, A., Miranda, F., Cesar
Jr., R. M., and Silva, C. T. "Mapping the walk: A scalable computer vision approach for
generating sidewalk network datasets from aerial imagery." *Computers, Environment and Urban
Systems* 101 (2023).

- **Version and hardware:** 0.5.0 at commit `c737a4e8c8b907ee3c80739673c67140ce95483d`,
  with its pinned default checkpoints (`satellite_2021.pth`, HRNetV2-W48). It ran in its own
  uv venv (Python 3.12, torch 2.14 + CUDA 13) on the makelab2 A40.
- **Settings:** zoom 19 (about 0.3 m/px), 256 px tiles stitched 4×4 into 1024 px inputs
  (its defaults), `inference --local --eval test --dump_percent 0 --deterministic`.
- **Conversion:** `t2n_convert.py` (in the archive beside the outputs) converts the
  GeoParquet to GeoJSON (`f_type` + geometry, 7-decimal coordinates). Per-city provenance
  is in `runs/<city>/aerial/tile2net.json`: source, date, URL, tile count, runtime, sha256.
- **GPU memory.** Without `--deterministic`, Tile2Net turns on cudnn benchmark mode, and
  its algorithm probing peaks at about **34 GB** of VRAM whatever the input size (measured
  at stitch step 4 and 2). Three short runs (the Boston example and two 1 km² pilots, about
  50 s each) peaked there while another job held about 9 GB. That left about 3 GB free,
  below the 20 GB this shared GPU is meant to keep free. `--deterministic` disables the
  probing and peaks at **1.4 GB**, so every run that produced results used it.
  Determinism changes only which convolution algorithms are picked, not the model.

### Imagery: one deviation from the issue

The issue assumed Tile2Net's statewide sources covered both cities. New Jersey's does. The
**Oregon** source is commented out upstream ("Oregon also has some SSL issues"), and on
2026-09-28 the Oregon Statewide Imagery Program server's TLS certificate had expired two
days earlier. Fetching it without certificate verification was not done.

| city | imagery | date | how |
|---|---|---|---|
| Paterson | NJ Orthos Natural 2020 (Web Mercator cache, `maps.nj.gov/.../Orthos_Natural_2020_NJ_WM/MapServer`) | 2020 | Tile2Net's built-in `nj` source |
| Bend | City of Bend 2019 imagery (`tiles.arcgis.com/tiles/JisFYcK2mIVg9ueP/.../City_of_Bend_2019_Imagery/MapServer`) | 2019 | standard z19 slippy tiles fetched by `scripts/aerial_fetch_tiles.py` (TLS verified, 4 workers, per-tile sha256), then `tile2net generate --input` |

**Date gap.** Bend's GSV panos are mostly from 2024 (84%), five years after its aerial
imagery. Paterson's largest share (48%) is from 2025, five years after its 2020 orthos. Sidewalks and ramps
built or rebuilt in between exist in the panos but not in the polygons. That is one
possible source of disagreement in Q1 and Q2, so Q1 is also reported by the capture year of
each GT ramp's newest pano (descriptive; the rule is unchanged).

### Scoring frame

- **Camera height:** the production `auto` height (fuse_sites' per-rig GSV rule, #79),
  flat raycast, 25 m cap, rig mask on.
- **GT:** RampNet's judged benchmark panos at tier 0.55. Verdict-true detections and
  missed-ramp clicks are placed at that height and merged at 2.5 m, exactly as
  `eval_ps_clustering.py` does.
- **Classes:** surface = sidewalk ∪ crosswalk, walkable = surface ∪ footpath. Distance is 0
  inside a polygon.
- **Coverage:** everything is restricted to the Tile2Net coverage, which is each run's
  `area.geojson` bounding box. Q2 additionally keeps only panos whose whole 25 m disk lies
  inside it.

### Pedestrian network

Paterson's network was written (30,365 segments). Bend's network step failed after the
polygons were written: `centerline.exceptions.TooFewRidgesError` in
`pednet.create_crosswalk`, a degenerate crosswalk polygon. Bend therefore has polygons
only. Q3 anchors there come from polygon contacts alone, and Paterson is scored both ways.

### Runtime

| city | tiles (z19) | fetch | generate + inference to polygons | peak VRAM |
|---|---:|---|---|---|
| Paterson | 12,064 | Tile2Net's own downloader | 15.6 min | 1.4 GB |
| Bend | 52,480 | 42.6 min (4 workers) | 19.9 min (the network step then failed after 16.6 more min) | 1.4 GB |

### Reproduce

```bash
python scripts/aerial_sidewalks.py gt bend paterson --run-root ../sidewalk-auto-labeler/runs
python scripts/aerial_sidewalks.py project bend paterson --run-root ../sidewalk-auto-labeler/runs --gallery
python scripts/aerial_sidewalks.py anchors bend paterson --run-root ../sidewalk-auto-labeler/runs
python scripts/aerial_sidewalks.py precision bend paterson --run-root ../sidewalk-auto-labeler/runs
python scripts/aerial_sidewalks.py verdict && python scripts/aerial_sidewalks.py figures
```

Figures: `docs/figures/aerial-sidewalks/` (`q1_distance_cdf.png`, `q2_edge_px_cdf.png`,
`q3_frag5.png`, `q4_off_surface.png`).
