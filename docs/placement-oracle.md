# Placement oracle: city curb-ramp inventories vs the camera-height default (issue #79)

`scripts/inventory_oracle.py` scores where fusion puts curb ramps against where two
cities' own inventories say they are. It exists to decide one thing: whether the
production camera height (`geo.DEFAULT_CAMERA_HEIGHT_M = 2.6`) should be replaced by one
of the two options `docs/camera-height-study.md` left open.

**This PR does not change the default.** It ships the instrument, the pre-registered rule
and the verdict. If the rule selects (b) or (c), the flip is a follow-up PR, because it
moves every fused number in the four GSV runs.

## Why an external oracle

The camera height sets every raycast's range. The RampNet benchmark cannot check it: GT
marks are pixels projected through the same raycast, so every height model moves the
truth along with the prediction. World P/R lands within ±3 points under every option
(camera-height study, "GT world P/R under each height model"). The implied-height
instrument (bearing-only triangulation) is what the options were calibrated on, so it
cannot referee them either.

A city's own curb-ramp inventory can. Each point was surveyed or digitized by city staff,
one per corner ramp, and never went through our raycast, the depth ground plane, or
Project Sidewalk's placement estimator.

## Where the inventories are (searched 2026-09-26)

| city | result | source | points: layer / in `area.geojson` / kept |
|---|---|---|---|
| **bend** | **yes** | City of Bend [Curb Ramps](https://data.bendoregon.gov/datasets/curb-ramps): `services5.arcgis.com/JisFYcK2mIVg9ueP/.../sCurbRamps/FeatureServer/0` (item `aa70ed04816040ea9c6dd5de6789448c`) | 14,822 / 14,328 / 12,504 with `LifeCycleStatus = 'I'` (installed) |
| **gainesville** | **yes** | City of Gainesville Public Works, layer 3 `CurbRamp`: `services2.arcgis.com/Zzhtlau4ccHkQgTu/.../PublicWorksInfrastructure_AGO/FeatureServer/3` (item `817b0156f40a4cb8a1ccdb88510d6dce`; mirrored on dataGNV as "ADA Ramps" `3um4-3vb3`, Public Domain) | 7,248 / 3,246 / 3,208 with `LIFECYCLE = 'Active'` |
| paterson | no | Passaic County ArcGIS (`gis.passaiccountynj.org`, 76 hosted services: parks, facilities, parcels, one pedestrian-bridges layer); NJDOT open-data portal and its ArcGIS Online org (road LRS, `NJ_Sidewalks` polylines, crash data, no ramp layer); NJGIN hub; ArcGIS Online item search "Paterson" (nothing). The city has no GIS portal. | — |
| sao_paulo | no | GeoSampa WFS (`wfs.geosampa.prefeitura.sp.gov.br`, 483 layers): `calcada` is sidewalk polygons with width and slope per segment, no ramp geometry; `acessibilidade_smped` is 972 establishments holding an accessibility seal, not ramps; no `rampa`/`rebaixamento` layer. | — |
| vancouver (future, #56) | known, down | `PublicWorks/transSidewalkPUB/MapServer/0` (16,960 points per #56) answers "Service ... not started" (2026-09-06 and 2026-09-26). Recorded in the script's registry; `runs/vancouver/inventory_oracle/inventory.json` holds the status. | — |

The exact pull (url, query, fetch time, sha256, byte length, counts, field histograms) is
in each city's tracked `runs/<city>/inventory_oracle/inventory.json`. The geojson itself
is cached beside it and not tracked; `fetch` refetches it and the sha256 shows whether the
layer changed. Bend's layer is edited continuously (last edit 2026-09-25), so a refetch
will not be byte-identical. The planning probe the day before counted 14,330 in the area;
this pull counts 14,328, two fewer decommissioned points, with the kept count unchanged.

Terms: Bend's item is public with a "reference purposes only, not for survey or
engineering" disclaimer and no explicit licence; Gainesville's is Public Domain on the
dataGNV mirror. Both are used for analysis only, and cited.

**Are the points at the ramp?** Yes. Both are one point per corner ramp (`Direction` /
`CORNER` = NE/SE/SW/NW). Bend's `Method` field says 85% were digitized and 14% GPS
surveyed. Gainesville's points are coarser, with about 2× Bend's per-point scatter. The
`characterize` subcommand and each report's first section give the nearest-site distance by
field.

**Which city decides.** Gainesville's imagery is 64% 2026, the low (2025–26) GSV rig, so it
is the only inventory city where the effect the options target can show. Bend is 84% 2024
and has 3 low-rig panos, so it guards against a regression on the old rig. Paterson's 2025
rig (48% of its panos) has no oracle. If an option passes, Paterson inherits it on
Gainesville's evidence, untested.

### Pilot (not pre-registered)

Measured during planning on the on-disk `sites.jsonl` (2.6 m, 0.55 tier). Single-vintage
sites within 5 m of an inventory point, and the site's signed offset from the point along
the mean member ray (+ = ranges run long):

| city | vintage | n | median along | IQR | along/range |
|---|---|---:|---:|---|---:|
| gainesville | 2026 (low rig) | 2,552 | +1.92 m | +0.78 … +3.17 | +0.135 |
| gainesville | 2024 | 121 | +0.29 m | −0.57 … +1.26 | +0.025 |
| bend | 2024 | 8,604 | +0.04 m | −0.33 … +0.42 | +0.004 |
| bend | 2012 | 277 | −0.58 m | −1.06 … −0.07 | −0.056 |

The pilot motivated the design. It is not the evidence. The pre-registered measurement
below re-fuses at the production tier (0.30) with the rig mask, and never reads the stale
0.55-tier `sites.jsonl` on disk.

## Pre-registration (committed before the first scoring run)

Everything in this section, the constants and `verdict()` in
`scripts/inventory_oracle.py`, and its tests, were committed before `score` was run on
either city. The results section below was added in a later commit, and nothing in this
section changed after it.

### Arms

All GSV, flat raycast (`apply_pose` auto, which is off for GSV), `OPERATIONAL_CONFIDENCE`
(0.30), rig mask on, 25 m cap. Heights come from `fs.load_results(read_heights=True)`,
which reads the harvested `depth/index.csv`; `score` refuses a city without it.

- **(a)** 2.6 m everywhere.
- **(b)** `geo.PER_PANO` with every measured `camera_height_m` (and its
  `camera_height_vintage_m`, so the #44 QC flag reads the same) multiplied by 1.08 on the
  loaded objects. `geo` is not changed. Unmeasured panos fall back to 2.6 m. The unscaled
  ×1.00 row is reported as a reference and is never a candidate.
- **(c)** `geo.PER_RIG` with `camera_height_m` set per vintage. A vintage is (run, capture
  year). Where its median measured depth height is < 2.1 m, every pano of the vintage
  raycasts at 2.0 m; in every other vintage, at 2.5 m. This covers unmeasured panos too,
  because a rig is not a payload. Undated panos, and vintages with no measured pano, fall
  back to 2.6 m. The vintage median is taken over every measured pano of the year, with no
  minimum count. That is `height_qc.load_city`'s `vmed`, which the study's decision table
  used to classify the low rig. The constants are imported from `height_qc`
  (`LOW_VINTAGE_M`, `OPTION_B_SCALE`, `OPTION_C`).

  (c) is not the same as the study's decision-table (c). That one keyed on each pano's
  *own* depth height and left unmeasured panos at 2.6 m. This plan's (c) is the per-rig
  constant #79 describes.

### Scoring

- **Frozen association** (the primary frame, `frozen@a`): `fs.fuse` once under (a), and
  keep that membership. Each arm re-places every operational member with
  `geo.detection_ground_point` under its own height and re-solves the site with the
  inverse-covariance refit (`eval_sites.refit_frozen`, factored out of the #42
  precondition unchanged). A site counts only if **every** arm places **every**
  operational member at the 25 m cap. The ×1.00 reference arm is in that intersection.
  It cannot drop a site the decision arms keep, because its heights never exceed (b)'s.
  Sensitivity arms (`--scale`, `--rig-cut`) join the intersection too, which is one
  reason they are never verdict inputs.
- **The (a)-anchored pool:** kept inventory points matched one-to-one (greedy, ascending
  distance) to a site under (a)'s positions within the radius. Since membership is frozen,
  every arm is scored on the same point–site pairs. Per arm: median and p90 (nearest rank,
  `eval_sites._pct`) of the point-to-site distance, and the share within 3 m.
- **Own-match per arm**, in the same frame: that arm's re-placed sites matched to the
  inventory on their own. Coverage is matched / kept points, plus the share within 3 m. This
  is the survivorship check.
- **`frozen@b`, `frozen@c`:** the same design, with membership from (b)'s or (c)'s own fuse.
  The pool stays anchored on (a)'s positions within that membership. Rule 3 reads it.
- **Own association (`own`):** each arm fused under itself and matched on its own. It
  favours each arm by construction. It is reported beside the primary frame, and rule 3
  reads it.
- **Along-ray offset by capture year:** for pool sites (5 m) whose operational members all
  share one capture year, this is the signed offset of the arm's site position from its
  inventory point, projected on the normalized mean of the member ray directions (the arm's
  own bearings). + means the site lies beyond the point as seen from the cameras, so ranges
  run long. The table reports the median, IQR, and median offset / mean member range. A
  site whose member rays cancel exactly is skipped.
- **Chance floor:** the kept points displaced 25 m in a random direction (seed 31,
  `agree_rate.chance_floor`) and matched again. It shows how much a 5 m match rate owes to
  coincidence.
- Radii 2.5 / 5 / 8 m; 5 m is primary. One-to-one matching uses `agree_rate`'s
  grid-accelerated `match_one_to_one`, which returns exactly the pairs of
  `eval_sites.match_one_to_one` (tested there).

### The decision rule

Primary frame: `frozen@a`, `OPERATIONAL_CONFIDENCE`, 5 m radius, the (a)-anchored pool,
rig mask on. A candidate X ∈ {(b), (c)} replaces 2.6 m only if **all** of these hold:

1. **gainesville:** X's p90 and median site-to-inventory distance both improve on (a) by
   more than 0.10 m (the #42 tolerance).
2. **bend:** X's p90 is not worse than (a)'s by more than 0.10 m. This is the old-rig
   no-regression check; bend has 3 low-rig panos and cannot carry rule 1.
3. **Not by construction (gainesville):** X, fused under itself (`own`), has an own-match
   p90 below (a)'s primary p90. **And** in `frozen@X` (X's membership, pool anchored on
   (a)'s positions), X's p90 is below (a)'s. The advantage must not exist only inside
   (a)'s membership.
4. **Survivorship (both cities, primary frame):** X's own-match coverage at 5 m is within
   1.0 point of (a)'s. Sites dropped by the every-arm-places filter are ≤ 5% of (a)'s
   operational sites; if not, the candidate is a **caveat**, not a verdict, and is never
   selected.
5. **Direction:** gainesville's 2026-vintage median along-ray offset under X (primary
   frame, 5 m) lies within ±0.75 m of zero. It is +1.92 m under (a) in the pilot. A
   correction that overshoots fails.
6. **Tie-break:** if both pass, (c) wins, unless (b)'s p90 beats (c)'s by more than
   0.10 m in both cities, since a per-rig constant carries no per-pano noise. If neither
   passes, 2.6 m stays, and the negative result is the deliverable.

`python scripts/inventory_oracle.py verdict` applies this to the committed `arms.csv` and
`vintage.csv` files. It warns if they were not scored at 0.30.

Reported, but not part of the verdict: radius 2.5 and 8 m, tier 0.55, ×1.06 and ×1.10, the
2.1 m rig threshold ±0.1, and the ×1.00 reference row.

### Risks named in advance

- Inventory accuracy. Bend is 85% digitized; its tail is ramps we have no site for, not
  bad points. Gainesville's points scatter about 2× more. The along-ray offset is the
  discriminant, and rule 1 asks for > 0.10 m against a pilot effect of +1.92 m.
- Dual ramps 1.5–3 m apart on one corner. One-to-one matching can pair the wrong twin; the
  2.5 m row shows what that costs.
- Points on state roads (802 in Gainesville owned by the State of Florida) and ramps outside
  GSV coverage inflate the unmatched share equally across arms. Coverage is compared across
  arms, never read absolutely.
- The verdict rests on one city and one rig (Gainesville 2026). Paterson's 2025 rig
  inherits it untested.

## Results (scored 2026-09-26, after the pre-registration commit)

`score bend gainesville` took 96 s wall for both cities on the desktop (bend 64 s, 78,560
panos, 12,504 kept points; gainesville 30 s, 37,435 panos, 3,208 kept points). It needs no
GPU and no network. Each arm is one full `fs.fuse`, plus three frozen refits. The committed
tables are `runs/{bend,gainesville}/inventory_oracle/{report.md,arms.csv,vintage.csv}`.

### Primary frame: frozen@a, 5 m, the (a)-anchored pool, tier 0.30

| city | arm | pool | median m | p90 m | ≤ 3 m | own coverage | 2026 / 2024 median along-ray m |
|---|---|---:|---:|---:|---:|---:|---:|
| gainesville | (a) 2.6 m | 2,677 | 1.50 | 3.29 | 87.3% | 83.4% | **+1.17** (2026) |
| gainesville | (b) per-pano × 1.08 | 2,677 | 1.27 | 3.18 | 88.1% | 85.5% | −0.62 |
| gainesville | (c) per-rig 2.0/2.5 | 2,677 | 1.22 | 2.94 | 90.4% | 85.6% | −0.47 |
| gainesville | per-pano × 1.00 (ref) | 2,677 | 1.42 | 3.47 | 85.4% | 85.5% | −1.01 |
| bend | (a) 2.6 m | 10,914 | 0.61 | 1.33 | 98.8% | 87.3% | +0.03 (2024) |
| bend | (b) per-pano × 1.08 | 10,914 | 0.62 | 1.35 | 98.7% | 87.2% | −0.04 |
| bend | (c) per-rig 2.0/2.5 | 10,914 | 0.63 | 1.37 | 98.8% | 87.3% | −0.14 |
| bend | per-pano × 1.00 (ref) | 10,914 | 0.69 | 1.49 | 98.6% | 87.2% | −0.34 |

No site was dropped by the every-arm-places filter in either city's primary frame (0 of
8,967 and 0 of 14,187). The chance floor at 5 m is 8.9% of Gainesville's points under (a)
and 6.7% under (c); Bend's is 4.3%.

Own association (each arm fused under itself, 5 m): Gainesville (a) 8,967 sites, coverage
83.4%, p90 3.29 m; (b) 7,488, 84.7%, 2.86 m; (c) 7,301, 84.4%, 2.84 m. Bend's three are
within 0.04 m on p90 and 0.1 pt on coverage of each other.

### The verdict under the committed rule: (a), 2.6 m stays

`python scripts/inventory_oracle.py verdict` prints:

| rule | (b) per-pano × 1.08 | (c) per-rig |
|---|---|---|
| 1 gainesville p90 and median improve > 0.10 m | pass (p90 +0.105, median +0.236) | pass (+0.348, +0.281) |
| 2 bend p90 not worse by > 0.10 m | pass (+0.017) | pass (+0.044) |
| 3 not by construction | **FAIL**: under (b)'s own membership (a) beats it (p90 3.03 vs 3.40) | pass (own-association 2.84 < 3.29; under (c)'s membership 2.87 < 2.97) |
| 4 coverage within 1.0 pt; site drop ≤ 5% | **FAIL**: gainesville +2.05 pt | **FAIL**: gainesville **+2.18 pt** |
| 5 gainesville 2026 along-ray within ±0.75 m | pass (−0.62) | pass (−0.47) |

Neither candidate passes, so rule 6 keeps 2.6 m.

**Decision point for the maintainer: how rule 4 is read.** (c) fails only rule 4, and only
because its own-match coverage in Gainesville is 2.18 points *higher* than (a)'s. More
inventory ramps get a site within 5 m once ranges stop running long. The committed
`verdict()` reads "within 1.0 pt of (a)'s" literally, as a two-sided band, and that is the
verdict above. The rule is headed *survivorship*, though, and the plan's own test
description reads "rule 4 fails on a coverage drop". Its #42 precedent is also a
one-sided drop limit. All of these suggest the intent was "coverage must not fall by more
than 1.0 point". Under that one-sided reading (c) passes all five rules while (b) still
fails rule 3, and rule 6 selects **(c)**.

The reading was not changed after the numbers were seen: `verdict()` is exactly as
pre-registered, and so is its answer, (a). Choosing the one-sided reading, and with it (c),
is a call for the maintainer, and would be made in the follow-up PR that flips the default.

### What the numbers say, rule aside

- **The oracle sees the new rig's range bias, and both corrections remove it.** Under (a),
  Gainesville's 2026 sites sit a median +1.17 m beyond their inventory ramp (along/range
  +0.088). (c) brings that to −0.47 m and (b) to −0.62 m: both overshoot slightly, and both
  stay inside the pre-registered ±0.75 m. The unscaled per-pano height overshoots to −1.01 m,
  which independently confirms that the depth frame runs short.
- **(c) beats (b) on this oracle.** It has the lower p90 in Gainesville (2.94 vs 3.18 m).
  (b) also loses to (a) inside (b)'s own membership, which is exactly the
  favoured-by-construction pattern rule 3 exists to catch.
- **Bend does not care.** Its 2024 imagery (84% of the run) is already unbiased at 2.6 m
  (+0.03 m along-ray). All three arms land within 0.05 m on median and p90, and (c)'s
  2.5 m costs a hair (−0.14 m along-ray on 2024, p90 +0.04 m).
- **(c)'s vintage rule has a visible flaw.** Gainesville's 2015 and 2018 vintages (606 and
  1,059 panos) have depth medians of 1.94 and 2.07 m, so (c) puts them at 2.0 m. Their sites
  then land 1.8–2.1 m *short* (2018: −0.50 → −2.13 m along-ray), which suggests those are old
  rigs that happen to read low. The rule has no minimum vintage size and no second signal. A
  follow-up that adopts (c) should look at that; the cut sensitivity below is where it shows.
- **The pilot's +1.92 m was an upper figure.** The pilot matched on-disk 0.55-tier sites with
  the rig mask off. Re-fused here, the 2026 offset is +1.35 m at 0.55 and +1.17 m at 0.30.
  The pre-registered +1.92 m only motivated rule 5's ±0.75 m.

### Sensitivity (not verdict inputs; scratch runs, not committed)

- **Tier 0.55** (`score --tier 0.55`): the same pattern. (c) passes rules 1, 2, 3 and 5
  (Gainesville p90 3.39 → 2.94, median 1.56 → 1.20) and fails the two-sided rule 4 on
  +2.61 pt coverage. (b) fails rule 1 (p90 unchanged at 3.39) and rule 3. Bend's tables are
  identical at 0.30 and 0.55, because its `results.jsonl` predates the storage floor and
  holds no sub-0.55 detections.
- **Radius.** At 8 m, (c) has p90 3.45 vs (a) 3.84 in Gainesville. At 2.5 m the pool is, by
  construction, only the points (a) already places within 2.5 m, so (a)'s p90 is capped
  there (2.15 vs (c)'s 2.25). The informative 2.5 m number is coverage: (c) matches 75.1% of
  points within 2.5 m, against (a)'s 66.4%.
- **Scale** (× 1.06 / × 1.10, Gainesville 5 m): p90 3.27 / 3.17 and median 1.30 / 1.24,
  bracketing × 1.08. None reaches (c)'s p90.
- **Rig threshold** (2.0 / 2.2 m instead of 2.1): Gainesville p90 2.92 / 2.97, against 2.94
  for (c) at 2.1. A 2.0 m cut moves the 2018 vintage back to 2.5 m, which is the better
  of the two. Bend is unchanged, because its three low vintages (2007–2009, 34 panos) have
  medians below 2.0 m. These runs add four arms to the site intersection, which drops 2
  of Bend's 14,187 sites; the primary rows move by at most 0.001.

### What this means for Paterson

Paterson has no inventory. Its 2025 vintage is the same low rig (depth median ~1.86 m, 48%
of its panos), so a flip to (c) would apply to it on Gainesville's evidence alone,
untested. São Paulo's single low vintage (2021, ~175 panos) is the case the camera-height
study found ambiguous. Ortho imagery
([label-latlng-estimation#20](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/20))
remains the route to an oracle there.

## Related

- [label-latlng-estimation#20](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/20)
  (ortho imagery) is the remaining oracle route for Paterson and São Paulo.
  [label-latlng-estimation#7](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7):
  the same two inventories are an external check on the server's own placements in
  Gainesville.
- [RampNet#101](https://github.com/ProjectSidewalk/RampNet/issues/101): the along/range
  column is a direct, external measurement of its range scale `k` per vintage.
- #56 / [SidewalkWebpage#4706](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4706):
  if Vancouver's layer comes back, `fetch vancouver` already knows where it lives.
