# NoCurbRamp labels from absent corners (issue #169)

Tool: `scripts/corner_labels.py` (`build`, `rank`, `export`, `score`, `sample`). Tests:
`tests/test_corner_labels.py`. Outputs: `runs/vancouver/corner_labels169/na_<read>/` (no
sidewalk layer) and `.../na_<read>_sw/` (with the city sidewalk layer, **the recommended pilot
input**), for `<read>` = `discount` and `count`. The layer's provenance is
`.../sidewalks/cov_sidewalks.geojson.source.json`. Issue: [#169](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/169)
(umbrella ProjectSidewalk/Planning#36).

**Revised 2026-11-06 (see "Pre-registration revision" below):** the test now runs on the
split-wide corner table, `runs/vancouver/corner_labels169_split/na_discount_sw/` and the bundle
`runs/vancouver/corner_gallery169_split/`. The criterion is unchanged. The `corner_labels169/`
builds, their sample and the pilot bundle `corner_gallery169/` stay as they were.

**Status (2026-11-11): the rule PASSES, under a declared amendment for n = 97, and nothing
goes to any server under this PR.** Jon rated all 100 sampled corners blind: 96 absent, 1
present, 3 can't tell, so absence precision is 0.990 [Wilson 0.944, 0.998] against a bar of 0.90
[0.85]. `score` prints NOT SCORED because only 97 corners are decided; no value of the 3
missing ratings could change the outcome ("Result" below). The primary `NA` read is **settled
as `discount`** (gallery 243, PR
[#180](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/180); see "The NA read"
below). **The pilot build is `na_discount_sw`** (discount read, city sidewalk layer). Next: a
test-server pilot, which needs Jon's clearance of the exact send.

## Definition

Jon Froehlich's definition of the label, 2026-11-05: *"is there a sidewalk at the corner and
does it have a curb ramp? If not, the corner is 'missing a curb ramp' and needs a NoCurbRamp
label."* So a corner gets a NoCurbRamp label when a sidewalk reaches it and no curb ramp
serves it.

He considered and rejected a stricter definition: a sidewalk, plus somewhere to cross to, plus
a painted crosswalk or a controlled intersection. It would hide exactly the corners where the
city never finished the pedestrian infrastructure, which are the corners the label exists to
find.

The `_sw` builds implement this definition: a corner is proposed when the detector finds no
ramp there (absent in the #241 corner table under `fusion/primary`, not excluded by the
inventory under the `NA` read) **and** a kept city sidewalk feature lies within 10 m of the
corner point (`--sidewalk-presence`, `--sidewalk-keep STATUS=Available,Pathway,Private`). The
pilot build is `na_discount_sw`.

Two known residuals under this definition:

- **Arterial and highway sidewalks with medians still get labels.** A corner on a wide
  arterial, or where a sidewalk meets a median or a highway ramp, gets a label if a sidewalk
  reaches it and no ramp serves it, even where nobody is expected to cross. That is correct
  under the definition, which deliberately does not ask whether a crossing exists. Watch the
  validation rate of these corners in the pilot.
- **Corner points that land off the curb (partly fixed 2026-11-06).** The corner point is
  12 m from the intersection centre along the sector bisector. On the far side of a T it
  landed midway along the far curb, between the two crossing landings; the split-wide table
  fixes that case (two corners, one per landing; "Pre-registration revision" below). On wide
  arterials, 12 m can still fall in the roadway or past the curb return. That part is open.

## What ships

1. **Corner -> label (`build`).** The input is the merged #241 corner table
   (`corners_full.jsonl`). A corner gets one NoCurbRamp proposal when all of these hold:
   - Its state under the decision rule's arm and observability variant (`fusion/primary` by
     default, `--arm` / `--variant`) is `absent`.
   - The inventory does not exclude it under the chosen `NA` read (`inventory_excludes`).
   - A sidewalk reaches it, when a sidewalk layer is given (`--sidewalk-presence`,
     `--sidewalk-keep`; see "The sidewalk layer" below).
   - At least one pano is a candidate.

   **Pano choice ("the clearest view").** The candidates are the corner's `pano_ids_25`, the
   panos the table says observe it. A candidate must:
   - be within `--max-range-m` (25 m; larger is refused, since the table lists only 25 m);
   - be at least `--min-view-m` from the point (3 m; nearer panos look straight down).

   There is **no view-direction test** (since 2026-11-05): a pano facing away from the corner
   is a candidate like any other. `--hemisphere-deg 90` restores the first build's
   forward-hemisphere test; `build.json` records the value (`params.hemisphere_deg`, `null` =
   off). See "Forward hemisphere" under Caveats.

   Candidates are ranked nearest first. Among the `--top-k` nearest (3), the newest capture date
   wins, and the nearest pano of that date carries the label. This reuses
   `corner_gallery._distance_order` and `newest_date`. `corner_gallery.choose_views` applies the
   newest-date slot to its whole 40 m pool, because a rater sees three crops. A label sits on
   one pano, so here that slot is confined to the top k. Up to four runner-ups are recorded.

   **Projection.** The corner point is projected with
   `geo.ground_point_to_pano(pose, lat, lng, camera_height=2.5)`, flat pose. 2.5 m is the fixed
   height of the #241 fusion, which built the corner table (`--camera-height-m`). A dropped
   projection is counted and listed in `unlabeled.csv`; Vancouver has none.

   **Evidence.** Each proposal carries a `source` block:
   - the observing panos within 25 m;
   - the nearest operational fused site within 50 m, with its id and distance;
   - the inventory status, counts and points;
   - all six arm/variant states;
   - the sector geometry;
   - the candidate count and the rejection counts;
   - the runner-ups.

   `unlabeled.csv` lists every absent corner that did not become a proposal, with its reason.
2. **Review order (`rank`).** This joins validator-ai's ranked CSV
   (ProjectSidewalk/sidewalk-validator-ai#69, unmerged; columns not final) on PS's pixel key
   `(pano_id, ps_pixel(x), ps_pixel(y))`. It writes `rank` (1 = highest score) and `rank_score`
   into `proposals.ranked.jsonl`, plus `rank.json`. Unmatched proposals get `rank: null` and are
   counted. Nothing is dropped on the score. The assumed contract is one row per label, with
   columns `pano_id,pano_x,pano_y,score` (all renameable by flag), pixels as PS stores them,
   and higher = more likely real. It lives in one function (`read_ranked_csv`), so adopting
   the final #69 columns touches only that. Not run on Vancouver: no ranked CSV exists yet.
3. **Export (`export`).** Proposals become a `results.jsonl` that send_to_ps.py accepts
   unchanged:
   - one line per pano and one detection per corner (two corners on one pano give one line
     with two detections);
   - the line is written by `main.build_output_line`, so it follows the #168 multi-type rules
     (per-detection `label_type`, sorted `label_types`, line-level `label_type`,
     `models[NoCurbRamp]`);
   - the pano block is copied verbatim from the `results.jsonl` of the run the proposal names
     (`pano.run`), never first-wins across runs; a run `build.json` does not record is refused;
   - each detection also carries its `corner` key, `rank` and the `source` evidence, which
     `transform_record` does not send;
   - lines are ordered by best rank, unranked last.

   Export refuses in four cases:
   - proposals not beside a `build.json` whose recorded sha256 they match (a ranked file is
     tied to it through `rank.json`);
   - a results file whose sha256 changed since the build. Export re-reads and re-hashes the
     absolute input paths `build.json` recorded on the build machine, so it runs where those
     files are;
   - a build made from a dirty tree: tracked files outside `runs/` with uncommitted edits
     (`--allow-dirty-build`);
   - a `git_sha` that is not a commit.
4. **Checks (`score`, `sample`).** See "What is not scored" below.

### The NoCurbRamp spec (detectors/registry.py)

There is no model. The "detector" is this rule, so the spec has `module=None`.
`load_detectors(['NoCurbRamp'])` raises `NoDetectorError`: "NoCurbRamp has no detector; it is
produced by scripts/corner_labels.py". `main.py --types NoCurbRamp` exits with that message.

**Thresholds.** Every proposal carries `confidence: 1.0`, because the rule is binary. All
three thresholds (storage floor, operational, benchmark) are 0.5, so a proposal always passes
`--min-confidence` and nothing below 0.5 could. The alternative was a confidence derived from
the evidence, such as the number of observing panos. It was rejected as the more complex
choice, and because it would imply a calibration nobody has measured.

**`mask_rig=False`.** The pixel is the projection of a known ground point, not a heatmap peak.
It reaches the nadir mask's dip (49 deg) only at about 2.2 m of range, and the build excludes
panos nearer than 3 m. So the mask could only drop a correct label.

**Provenance** (`models[NoCurbRamp]`):
- `model_id` is `corner-rule@<12 hex>` of the repo's HEAD commit when the proposals were
  built (`build.json` `tool.git_sha`). It is HEAD, not the script's last commit, because
  `geo.py`, `corner_inventory.py`, `corner_gallery.py` and the registry are part of the rule
  too. The commit holds the code; the outputs it built land in a later commit;
- `model_revision` is the full SHA;
- `model_training_date` is the rule's version date, `RULE_VERSION_DATE` 2026-11-05, as
  `11-05-2026`;
- `model_repo` is `ProjectSidewalk/sidewalk-auto-labeler`;
- `api_version` is `1.0.0`.

## The rule, as pre-registered on #169

- **Precondition.** Gallery 243 is scored and the primary `NA` read is written into
  `docs/corner-inventory.md` first. **Met 2026-11-05**: scored in PR
  [#180](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/180), `discount`
  (read (b)) is primary, and the follow-up section of `docs/corner-inventory.md` says so.
- **Corner-level absence precision against blind ratings >= 0.90**, with Wilson lower bound
  >= 0.85. It is measured on a seeded sample of >= 100 absent corners drawn under the primary
  read, rated by the gallery 243 rubric. `can't tell` corners are excluded from the
  denominator, and their share is reported.
- The per-direction arm is scored separately on the same sample.
- Pilot: a test-server campaign only.

`score` implements this:
- precision = absent / (absent + present), over the proposals that carry a **blind** verdict;
- `cant_tell` is excluded, and its share is reported;
- n = absent + present (`can't tell` is out, so the 100 is counted after it).

It prints PASS / FAIL only when both of these hold, and NOT SCORED otherwise:
- the scored corners are a sample drawn for this rule (`--sample sample.csv`);
- n >= 100.

`score` also reports a secondary **label precision**, which counts `no_sidewalk` absences as
wrong (guide rule 3: no label where no sidewalk reaches the corner). `unspecified` absences
count as right there, so it is an upper bound.

## Pre-registration revision (2026-11-06): the far side of a T is two corners

**What the pilot found.** Jon rated 7 of the 99 units of `corner_gallery169/` (8 corners) and
exported them at 2026-11-06T13:33Z; the file is kept as
`runs/vancouver/corner_gallery169/pilot_verdicts__jonf.json`, deliberately not under the scored
name. His notes on `vancouver:art:n47372178` corner 0 and `vancouver:res:n958723078` corner 2:
on the far side of a T, the corner point sits midway along the far curb, between the two places
a crossing of the through street lands, so a NoCurbRamp label there is misplaced. The corner
table flagged such sectors (wider than 150°) `wide`, as "usually no crossing". **51 of the 100
sampled corners were wide**, as were 1,010 of the 1,821 `na_discount_sw` proposals and 3,177 of
the table's 14,391 corners.

**The fix (Jon's rule).** There are two crossings on the far side of a T, one per landing, and
each needs a ramp. `corner_inventory.py build --split-wide` splits every sector wider than 150°
at a *phantom leg* opposite each real leg whose opposite falls inside it at least 30° from both
bounds, so the far side of a T becomes the two corners the four-way would have if the stem
continued. Rule, margin, regression and table counts: `docs/corner-inventory.md`, amendment of
2026-11-06. The flag is off by default; the #238 / #241 numbers are untouched.

**What changes for the test.**

- The test runs on `runs/vancouver/corner_labels169_split/na_discount_sw/` (the same flags as
  `corner_labels169/na_discount_sw/`, on the split table) and its `sample.csv`: seed 169, 100 of
  3,070 proposals, drawn 2026-11-06 before any rating of them.
- It is rated in `runs/vancouver/corner_gallery169_split/` under rubric v2 (v1 plus a paragraph:
  phantom legs are dashed on the aerial, the far side of a T has two corners, one per crossing
  landing, and a ramp counts only for its own landing; crops may be landscape).
- **The criterion is unchanged**: corner-level absence precision >= 0.90 with Wilson lower
  bound >= 0.85, on >= 100 decided blind ratings, `can't tell` excluded and its share reported;
  the secondary label precision as before.
- The old sample (`corner_labels169/na_discount_sw/sample.csv`), the pilot bundle and the pilot
  verdicts stay in the repo. **The pilot is not scored as the test**: it rated a sample of the
  old table, and it is the observation that motivated the fix.

**Counts, old table → split table** (all intersections, `fusion/primary`):

| build | proposals | wide share of proposals | `dropped_no_sidewalk` | excluded by inventory |
|---|---:|---:|---:|---:|
| `na_discount` | 6,131 → 8,529 | 34.4% → 3.2% (2,107 → 270) | - | 253 → 283 |
| `na_count` | 2,894 → 5,274 | 70.7% → 4.9% (2,045 → 258) | - | 3,490 → 3,538 |
| `na_discount_sw` | **1,821 → 3,070** | 55.5% → 5.6% (1,010 → 171) | 4,311 → 5,461 | 253 → 283 |
| `na_count_sw` | 1,376 → 2,620 | 72.5% → 6.5% (998 → 171) | 1,519 → 2,656 | 3,490 → 3,538 |

Absent corners go from 6,385 to 8,814 of 17,063 (was 14,391); `no_candidate` is 2 in the
no-layer builds (was 1). `na_discount_sw` by stratum: signalised 10 → 34, arterial 639 → 1,052,
residential 1,172 → 1,984; 2,088 of its 3,070 proposals are bounded by a phantom leg. Proposals
rise mostly because a far side with a ramp at one landing was one present corner and is now one
present and one absent corner. The new sample: 66 residential and 34 arterial corners at 98
units; 67 bounded by a phantom leg; 5 still wide. The crowd-overlap numbers below were not
recomputed for the split builds (they need the server's API; nothing was fetched).

## The NA read (settled 2026-11-05: `discount`)

`--na-read` is **required and has no default**. The two reads:

- `count` is corner-inventory read (a), "as written". Any inventory point excludes the corner,
  `NA` included, because the city recorded something there. Only corners with no point at all
  are labeled.
- `discount` is read (b) of amendment A8. `NA` points with no `RAMPTYPE` (`NA_noramp`) are read
  as the city's record of no ramp and do not exclude. Every other class does: `Available`,
  `NA_typed`, `RMV`, `Expired/Removed` and other. This is exactly read (b), the read the issue
  quotes at 0.965 at unit level.

**Settled: `discount` is primary.** Gallery 243 (RampNet#243) was rated blind, 95 of 95
units (`verdicts__jonf.json`), and scored in draft PR
[#180](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/180) (branch
`corner-gallery-243-verdicts`, `runs/vancouver/corner_gallery243/score/report.md`;
`score/score.json` sha256 `447c6bb5db0fd2e2d18d7f01c8812f313db28d25cd23c64da14d23e873fb12bc`).
From that report:

- the 40 sampled units whose only inventory points are `NA` with no `RAMPTYPE` were absent at
  every corner, **39/39** of the decided units (1 undetermined), Wilson 95% [0.910, 1.000];
- the corners holding those `NA` points were absent **99/99** (1 can't tell);
- the 35 false absences split into 4 real misses, 16 "inventory built after the imagery" and
  15 inventory or geometry artifacts;
- the control (clean units) agreed 17/18;
- the stratified absence precision is 0.983 [0.926, 1.000] (descriptive).

So an `NA` point with no `RAMPTYPE` is the city's record of *no ramp*. That is read (b) of
amendment A8, the `discount` read. `count` stays buildable as the comparison.

**In the builds.** `build --na-read-settled` records the settlement in `build.json` `params`:

- `na_read_settled: true` and `na_read_primary: "discount"`;
- `na_read_is_primary`: true for the discount dirs, false for the count dirs;
- `na_read_settlement`: the evidence pointer (PR #180, the report and score paths, the sha256
  of `score.json` and of the verdicts file, and the two headline numbers), copied from
  `NA_READ_SETTLEMENT` in the script.

Without the flag a build records `na_read_settled: false`, as every build before 2026-11-05 did.

## Vancouver counts

Merged #241 build, intersections only (mid-block units have no corners), `fusion/primary`.
Inputs are pinned by sha256 (`PINNED_SHA256`: the inventory build's inputs and gallery 243's
`corners_full`). All four dirs were built at aea5dc6 on a clean tree, then rebuilt at 80fc070
with `--na-read-settled` to record the settled `NA` read. The rebuild changed only metadata:

- `proposals.jsonl`, `unlabeled.csv` and `counts.csv` are byte-identical (the same sha256 in
  `build.json` `outputs`), so every count below is unchanged and `sample.csv` was not redrawn;
- in `build.json` only `built_at`, `seconds`, `tool.git_sha` and the new `params.na_read_*`
  fields differ;
- in `export.json` and the exported `results.jsonl` only the provenance changed, from
  `corner-rule@aea5dc6fa00c` to `corner-rule@80fc070659f7`, because `model_id` names the
  commit that built the proposals.

**Effect of the two review changes** (all intersections; proposals):

| read | before (hemisphere on, no layer; 7fc5cbf) | after (view test off) | after + sidewalk layer | `dropped_no_sidewalk` |
|---|---:|---:|---:|---:|
| discount | 5,770 | 6,131 | **1,821** | 4,311 |
| count | 2,715 | 2,894 | **1,376** | 1,519 |

Turning the view test off moved `no_candidate` from 362 to 1 (discount) and from 180 to 1
(count); the one left has no pano within 25 m that is at least 3 m away. It also changes which
pano carries a label at corners that had one before, since the nearest pano now wins whichever
way it faces: about half the chosen panos now face away from their corner (3,006 of 6,131
under discount). The sidewalk layer then drops 70% of the discount read's corners and 52% of
the count read's (below for why that is plausible).

Per stratum, view test off:

| stratum | corners | absent | excluded by inventory (discount / count) | no candidate | proposed, no layer (discount / count) | no sidewalk (discount / count) | **proposed with layer** (discount / count) |
|---|---:|---:|---:|---:|---:|---:|---:|
| signalised | 1,004 | 51 | 14 / 23 | 0 | 37 / 28 | 27 / 21 | **10 / 7** |
| arterial | 4,743 | 1,754 | 73 / 686 | 0 | 1,681 / 1,068 | 1,042 / 490 | **639 / 578** |
| residential | 8,644 | 4,580 | 166 / 2,781 | 1 | 4,413 / 1,798 | 3,242 / 1,008 | **1,172 / 791** |
| all | 14,391 | 6,385 | 253 / 3,490 | 1 | 6,131 / 2,894 | 4,311 / 1,519 | **1,821 / 1,376** |

(With the layer the sidewalk test runs before the pano test, so the one `no_candidate` corner
is counted as `no_sidewalk` there.) Projection dropped nothing in any build.

| | discount | count | discount_sw | count_sw |
|---|---:|---:|---:|---:|
| proposals | 6,131 | 2,894 | 1,821 | 1,376 |
| panos | 5,053 | 2,677 | 1,664 | 1,292 |
| corners with no inventory point / only `NA_noramp` | 2,894 / 3,237 | 2,894 / 0 | 1,376 / 445 | 1,376 / 0 |
| range to the corner point, p10 / median / p90 (m) | 6.2 / 10.2 / 16.1 | 7.3 / 12.3 / 17.2 | 6.7 / 11.4 / 17.6 | 7.2 / 12.3 / 17.5 |
| newest date beat a nearer candidate | 1,197 | 521 | 311 | 240 |
| chosen pano faces away (off axis > 90 deg) | 3,006 | 1,431 | 925 | 701 |
| pano from `vancouver_posdetect241` / `vancouver` / `vancouver_posdetect241_gsv` | 4,205 / 1,913 / 13 | 1,666 / 1,219 / 9 | 817 / 998 / 6 | 561 / 809 / 6 |

`proposals.jsonl` is 10.1 MB (discount), 4.6 MB (count), 2.9 MB (discount_sw) and 2.2 MB
(count_sw), all over the 2 MB line. So they and the exported `results.jsonl` are not committed.
`build.json` and `export.json` carry their sha256, line and byte counts, and `build`
reproduces them from the pinned inputs and the sidewalk file (whose sha256 `build.json` records
under `inputs.sidewalk_presence` and `sidewalk_rule.sha256`).

### The sidewalk layer

**Source.** City of Vancouver, WA, "City of Vancouver Sidewalks (View)": ArcGIS item
`6506a0f173bf4329aaa0baac61b61877`, layer
`https://services.arcgis.com/oNvpY90qsPDizwkN/arcgis/rest/services/COV_TransSidewalk_View/FeatureServer/0`.
It sits in the same city ArcGIS org as `COV_TransCurbRamp`, the curb-ramp inventory
(`runs/vancouver/inventory_oracle/inventory.json`); the item says the data come from Clark
County GIS and city transportation, updated weekly. Downloaded 2026-11-06T00:01:02Z: every
feature intersecting the bbox of `runs/vancouver/area.geojson` (the bbox the inventory pull
used), 75,349 LineString sidewalk centrelines, WGS84, sha256 `11503ed4cd7d...` (full value,
query, histograms and licence text in `sidewalks/cov_sidewalks.geojson.source.json`). The
layer covers the area: only 56 of the 6,132 absent, non-excluded discount corners have no
feature of any status within 50 m.

**Licence.** The city's GIS Data Terms of Use grant a limited, non-exclusive licence for
personal use and forbid selling, reproducing or distributing the data. It is not an open
licence. So the 59 MB GeoJSON is gitignored and only its provenance is committed, as with the
curb-ramp inventory. The labels are derived from it (a corner is dropped or kept), and the
layer itself goes nowhere. **Whether a label campaign derived from it is within those terms is
Jon's call**; nothing here decides it.

**Which features count as sidewalk (`--sidewalk-keep STATUS=Available,Pathway,Private`).**
The layer digitizes street sides that have **no** sidewalk too: 28,203 features have `STATUS`
`NA`, and all but 14 of them carry no `UNITTYPE` (attached / detached). Read like `NA` with no
`RAMPTYPE` in the curb-ramp layer (corner-inventory amendment A6), they are the city's record
of a street side without a sidewalk, so they do not count. `RMV` (removed, 1,105), `CONST`
(under construction, 263), `Clarify` (3) and null (15) do not count either. Kept: `Available`
(45,083), `Pathway` (475), `Private` (202), 45,760 in all. This filter was chosen from the
layer's attributes before any build with it. Radius: the existing 10 m default
(`--sidewalk-radius-m`).

Most drops are `NA` geometry: of the 4,311 discount corners dropped, 4,100 have a feature of
some status within 10 m.

**Descriptive check against the gallery 243 blind ratings** (discount read, the 120 rated
corners among its absent, non-excluded corners; after the filter was fixed, not used to choose
it):

| rated | kept by the layer | dropped |
|---|---:|---:|
| absent, `no_sidewalk` | 2 | 44 |
| absent, `curb_no_ramp` | 14 | 0 |
| absent, no kind given | 15 | 41 |
| present | 1 | 0 |
| can't tell | 1 | 2 |

The layer drops 44 of the 46 corners the rater called `no_sidewalk` and none of the 14 called
`curb_no_ramp`. That is the guide's rule 3 doing what it should, on a sample that was not drawn
for this purpose (see the next section), so it is encouraging, not a measurement.

## What is not scored, and why

`score` against the gallery 243 blind ratings (`verdicts__jonf.json`) gives these reads.
**All are NOT SCORED.**

| build | rated corners | absent | present | can't tell | precision [Wilson 95%] | `no_sidewalk` | label precision [Wilson 95%] |
|---|---:|---:|---:|---:|---|---:|---|
| discount | 120 | 116 | 1 | 3 (0.025) | 0.992 [0.953, 0.999] | 46 | 0.598 [0.508, 0.683] |
| count | 54 | 51 | 1 | 2 (0.037) | 0.981 [0.899, 0.997] | 19 | 0.615 [0.480, 0.735] |
| discount_sw | 33 | 31 | 1 | 1 (0.030) | 0.969 [0.843, 0.995] | 2 | 0.906 [0.758, 0.968] |
| count_sw | 24 | 22 | 1 | 1 (0.042) | 0.957 [0.790, 0.992] | 2 | 0.870 [0.679, 0.955] |

(The first build, with the hemisphere test, had 115 and 51 rated corners under discount and
count, with the same single present corner.)

There are three reasons:
- that sample was drawn for #243, not for this rule: its strata are false absences, `NA`-only
  units and clean controls, so the corners are not a random draw of the proposals;
- the precondition was not met when these were computed (it is now);
- n < 100 in every build but the no-layer discount one.

These descriptive scores used the `verdicts__jonf.json` on this branch, exported when 50 of
gallery 243's 95 units were rated. They were not rerun on PR #180's complete file: that would
change only these NOT SCORED descriptions, and the gallery 243 bundle on this branch is left
as it was.

`score_sample.json` scores each dir's seeded sample (`sample.csv`, 100 corners,
`sample --seed 169`, drawn from that dir's proposals before any rating of it exists). Only
3 (discount), 0 (count), 3 (discount_sw) and 2 (count_sw) sampled corners are already rated, so
all are NOT SCORED (n < 100). The sample to rate for the pilot is `na_discount_sw/sample.csv`
(see "The precision gallery").

What these descriptive numbers say before any pilot:
- **Guide rule 3 was not cosmetic, and the layer addresses it.** Without a layer, 46 of the
  116 corners rated absent under discount (19 of 51 under count) were rated `no_sidewalk`,
  where the guide asks for no label, so roughly 4 in 10 proposals at rated corners would have
  been labels the guide says not to place. With the layer, 2 of 31 (2 of 22) are. Label
  precision rises from about 0.6 to about 0.9, with wide intervals.
- **Only one rated corner was present**, in every build. Absence precision in the rubric's own
  sense is high wherever rated.

## The precision gallery (`runs/vancouver/corner_gallery169/`)

The pre-registered rule is judged on a blind rating of a sample of the rule's **own**
proposals. Gallery 243 is not that sample (its strata are inventory classes), so a second
bundle rates `na_discount_sw/sample.csv`: 100 corners drawn with seed 169 from the pilot
build's 1,821 proposals, before any rating of them.

**Built with the gallery 243 tooling.** `corner_gallery.py build --items-from <sample.csv>`
skips gallery 243's three parts and its population checks and rates the sample instead:

- one unit per sampled intersection (99 units, since one unit holds two sampled corners);
  **only the sampled corners** of a unit are shown and rated, numbered as in the corner table;
- views, crops, aerials, rubric v1, the blind freeze, timing, notes and Export are gallery
  243's, unchanged; the view pool is the whole unit's, as in gallery 243;
- **the reveal is unchanged: it shows the city inventory** after a unit is completed. The
  rule's proposal (chosen pano, pixel, range) is stored in `items.jsonl` per corner
  (`proposal`) and never sent to the page. The pano that carries the label is among the views
  shown at all 100 corners;
- `snapshot.json` (`mode: sample`) records the sha256 of the sample, `sample.json`, the
  proposals and their `build.json`, plus the #241 inputs; the build refuses a sample or
  proposals that do not match their records, and a `corners_full` other than the one the
  proposals were built on.

By stratum: 64 residential, 35 arterial, 1 signalised corner. Items sha256
`1735f7bd2d17c45429d219f204a70423d8d0e0297f502f8c399488f9c2f04806`.

**Crops: not cut yet.** The 281 crops come from 279 panos of the PS pano store on makelab2. No
local copy of those panos exists on this machine (searched D: and the user profile), and this
session must not write on makelab2, so the bundle was built with `--no-crops`: items,
snapshot, report and the 99 aerials (Esri tiles) are in place. `CROPS_NEEDED.md` in the bundle
gives the one command that cuts them (the same build without `--no-crops`, which drives the
crop worker on makelab2 and pulls the crops back) and a by-hand alternative using
`crop_job.json`. `render` refuses until the crops are on disk and in `images.sha256`.

**Scoring.** Export saves `verdicts__<rater>.json` in the gallery 243 format, keyed by unit
and corner, which is what `score` reads. `--bundle` checks the export against the bundle
before scoring: the sample's and the proposals' sha256 are the ones `snapshot.json` recorded,
`items.jsonl` is unchanged, the file passes gallery 243's validator
(`corner_gallery_score.validate`: schema, items sha256, rubric text, blind reads) and every
sampled corner is an item corner. `corner_gallery_score.py` does not score this bundle; it is built for gallery 243's
three parts.

```bash
B=runs/vancouver/corner_gallery169; D=runs/vancouver/corner_labels169/na_discount_sw
# 1. crops (Jon; drives makelab2): the build without --no-crops, see $B/CROPS_NEEDED.md
python scripts/corner_gallery.py build --items-from $D/sample.csv \
    --corners-full ../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl \
    --bundle $B
# 2. the page; open $B/gallery/jonf/index.html, rate, Export, save as $B/verdicts__jonf.json
python scripts/corner_gallery.py render --bundle $B --rater jonf
# 3. the pre-registered score
python scripts/corner_labels.py score --proposals $D/proposals.jsonl --sample $D/sample.csv \
    --bundle $B --verdicts $B/verdicts__jonf.json --out $D/score_gallery169.json
```

Tracked in the bundle: `items.jsonl`, `snapshot.json`, `images.sha256`, `report.md`,
`CROPS_NEEDED.md` and `crop_job.json` (both deleted by the build once the crops exist), and
later `verdicts__*.json`. The aerials, crops, tile cache and `gallery/` stay local, as in
gallery 243.

### The split-wide bundle (`runs/vancouver/corner_gallery169_split/`, the revised test)

Built 2026-11-06 like the pilot bundle, with three changes Jon asked for after the pilot:
- **The split-wide sample** (`corner_labels169_split/na_discount_sw/sample.csv`, 100 corners at
  98 units: 66 residential, 34 arterial), on the split table's `corners_full.jsonl`;
- **up to 6 views per corner** (`--views-per-corner 6`; the newest-date slot is kept, the rest
  fill in distance order): 471 crops, 46 corners with 6 views, 20 with 5, 12 with 4, 10 with 3,
  5 with 2 and 7 with 1; the pano that carries the label is among them at 100 of 100 corners;
- **landscape crops**, 120° x 60° of the equirect at 896 x 448 px (`--crop-aspect 2
  --crop-fov-deg 120 --crop-px 896`), the orange ring still on the corner point, laid out one
  per row at a 1200 px window; the aerial stays, and **phantom legs are drawn as thin dashed
  white lines** (76 units carry one; 67 of the 100 corners are bounded by one);
- **rubric v2** (above). Gallery 243 and the pilot bundle keep v1; each bundle's
  `snapshot.json` holds its version and text, `render` shows that text, and the validators
  accept v1 or v2 but only the bundle's own.

The crops were cut on makelab2 by the crop worker on 2026-11-06 (0 missing) and the 98
aerials made from Esri tiles; `sha256sum -c images.sha256` passes for all 569 images. Items
sha256 `e7648a487a92ef508d488e40d8425de333f0d3719d2412a72a80138e15e26910`. The page is
`runs/vancouver/corner_gallery169_split/gallery/jonf/index.html`. As for the pilot bundle,
everything but `cache/` is committed.

```bash
B=runs/vancouver/corner_gallery169_split; D=runs/vancouver/corner_labels169_split/na_discount_sw
python scripts/corner_gallery.py build --items-from $D/sample.csv \
    --corners-full runs/vancouver/corner_inventory169_split/corners_full.jsonl --bundle $B \
    --views-per-corner 6 --crop-aspect 2 --crop-fov-deg 120 --crop-px 896
python scripts/corner_gallery.py render --bundle $B --rater jonf
# open $B/gallery/jonf/index.html, rate, Export, save as $B/verdicts__jonf.json
python scripts/corner_labels.py score --proposals $D/proposals.jsonl --sample $D/sample.csv \
    --bundle $B --verdicts $B/verdicts__jonf.json --out $D/score_gallery169.json
```

## Result (2026-11-11): PASS, under an amendment for n = 97

**Rated.** Jon rated all 98 units (100 corners) of `corner_gallery169_split/` under rubric v2.
The committed `verdicts__jonf.json` is his export of 2026-11-11T19:17:47Z (sha256
`da3f0114…`), and `score_gallery169.json` scores it. Its blind reads are identical to those of
his first complete export, made at 19:01:39Z (sha256 `e546e70e…`, not committed). The second
export adds only post-reveal edits.

| rated corners | absent | present | can't tell | precision [Wilson 95%] | `no_sidewalk` | unspecified | label precision [Wilson 95%] |
|---:|---:|---:|---:|---|---:|---:|---|
| 100 | 96 | 1 | 3 (0.03) | **0.990 [0.944, 0.998]** | 5 | 2 | 0.938 [0.872, 0.971] |

By stratum: arterial 33 absent, 1 can't tell; residential 63 absent, 1 present, 2 can't tell.
Label precision counts the 2 `unspecified` absences as right, so it is an upper bound.

**Why `score` says NOT SCORED, and the amendment.** The rule needs n >= 100 decided blind
ratings, with `can't tell` removed before counting. The sample was drawn at exactly 100
corners, so any `can't tell` was bound to leave n short. That was a design slip in the draw,
not something the ratings did. **Amendment, declared after the ratings were seen:** the
verdict is taken at n = 97, because it cannot depend on the 3 missing ratings. If all 3 were
`present`, the read would be 96/100: precision 0.960, Wilson lower bound 0.902, still a pass on
both clauses. The script's NOT SCORED stays as printed. The PASS is this amendment's, and it
holds for every value the missing ratings could take. **Next time, draw a seeded reserve** (for
example 110, rated in order until 100 are decided) or size the sample for the expected
can't-tell share. `sample --reserve` now does this; see
[Reserve and sequential stop rule](#reserve-and-sequential-stop-rule-183).

**Post-reveal edits (recorded, never scored).** After the reveal Jon changed 5 corners: the one
`present` (`vancouver:res:n47255336` #2) to can't tell, since his note had doubted which landing
the ramp served; the driveway apron at `vancouver:art:n47372480` #2 from can't tell to
absent / `curb_no_ramp`; and the missing reason at `vancouver:art:n2966616369` #1 and
`vancouver:res:n47360202` #0 (`curb_no_ramp`). Earlier he had changed the reason at
`vancouver:art:n47301278` #0 (`no_sidewalk` to `curb_no_ramp`, "a gravel sidewalk"). His later
reads therefore hold no present corner. They only point the same way as the blind read.

**Crossing notes (descriptive, from free text).** 70 of the 98 units carry a note. A keyword pass
(`safe`, `busy road`, `median`) finds **23 units whose note says, in some form, that it is not a
safe place to cross: 22 of the 34 arterial units** and 1 residential. 14 units (11 residential)
say there is nothing to cross to on the far side. Jon rated these absent as the rubric asks:
every leg of a Washington intersection is a legal unmarked crosswalk unless it is signed closed.
The tally is a floor, since he did not note every unit. It bears on the pilot, though: on
arterials, a NoCurbRamp label will often mark a place where the rater thinks the right fix is no
crossing, not a ramp. That is a question for the pilot's scope (residential first?) and for the
city, not for this precision test.

**Rating cost.** The page's timer logged about 46 min for the blind pass (median 20 s a unit)
and 52 min with the re-review, and Jon found the task exhausting. The levers for the next
sample are a sequential design (at 40 decided corners, all absent gives a lower bound of 0.912
and one present 0.871, both passes), a model pre-rater that a person audits, one-click tags in
place of the recurring free-text notes, and a cap on the residential far-side-of-T stratum.
Filed as [#183](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/183) (reserve,
stop rule, caps), [#184](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/184)
(tags) and [#185](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/185) (VLM
pre-rater).

## Reserve and sequential stop rule (#183)

For the **next** rated sample; #169's recorded result above is unchanged. `corner_labels.py
sample --reserve K --looks 40,60,80,100` draws n + K corners in one seeded order and writes
`sample.json` schema `/2` with a `design` block. The design is fixed at that moment. `sample`
never overwrites a sample, `corner_gallery.py build` refuses a changed `sample.json` without
`--rebuild`, and `score --bundle` refuses a `sample.json` other than the one the bundle was
built from. Commit `sample.csv`, `sample.json` and the bundle before the first rating.
Without `--reserve` / `--looks`, `sample` writes the `/1` record exactly as before.

**Counting.** The gallery serves a `/2` sample's units in sample order, not sha1 order. `score`
reads the blind verdicts in sample order. At look k it takes the first n_k *decided*
(absent or present) corners before the first unrated one; `cant_tell` is skipped and the next
corner, eventually a reserve corner, takes its place. A rating past an unrated unit is refused,
and the message names the units to rate to clear it. The refusal is a protocol check. The
validity control is the counted prefix: a corner after an unrated one never enters a look.
Unrated reserve corners are fine. The first look that crosses a boundary settles the verdict,
and ratings after it never change it (`rated_beyond_stop`).

**The rule is a set of integer boundaries.** At look k, PASS needs `pass_min_absent` absent of
n_k. FAIL is settled at `fail_max_absent` or fewer, when no later look could pass even if every
remaining decided corner were absent. The boundaries come from the #169 clauses (precision
>= 0.90 and Wilson lower bound >= 0.85) at one z shared by every look. That z is the smallest
on a 0.001 grid from 1.96 for which exact binomial enumeration keeps P(PASS | precision = 0.85)
<= 0.025 (`make_design`). z only derives the integers, and `score` checks that `sample.json`'s
integers are what the shipped code derives. A one-look design keeps z = 1.96, the #169 rule.
A final look below 100 is refused. FAIL by curtailment costs no false-pass rate. Looking early
does, and the calibrated z pays for it:

| look n | uncorrected z = 1.96 (PASS >= / FAIL <=) | calibrated z = 2.192 (PASS >= / FAIL <=) |
|---:|---|---|
| 40 | 39 / 31 | 39 / 32 |
| 60 | 57 / 51 | 58 / 52 |
| 80 | 75 / 71 | 76 / 72 |
| 100 | 92 / 91 | **93** / 92 |

Looking four times at z = 1.96 would raise P(PASS | 0.85) to 0.043. **The cost: the final bar
is 93 of 100, not #169's 92**, since 92/100 has a lower bound of 0.840 at z = 2.192. At 40, the
issue's numbers (lower bound 0.912 for 40/40 and 0.871 for 39/40) are for z = 1.96. At z = 2.192
they are 0.893 and 0.852, so 39 of 40 still passes, and 38 of 40 (0.816) does not. The z
depends on the schedule: 2.179 for 30/50/75/100, 2.101 for 25/50/75/100, 1.961 for 40/100.

Operating characteristics, P(PASS) / expected decided corners, by the true precision
(`design_oc`; each decided corner absent independently):

| true precision | one look at 100 (#169) | looks 40/60/80/100 |
|---:|---|---|
| 0.85 | 0.0275 / 100 | 0.0234 / 62 |
| 0.90 | 0.321 / 100 | 0.244 / 77 |
| 0.95 | 0.937 / 100 | 0.885 / 67 |
| 0.97 | 0.997 / 100 | 0.991 / 53 |
| 0.99 | 1.000 / 100 | 1.000 / 42 |

**Reserve size.** With can't-tell at 3% of corners, P(fewer than 100 decided among 100 + K)
is 0.0005 at K = 10. At 5% it is 0.022 at K = 10 and 0.0002 at K = 15. **Use K = 15.** At #169's
precision the rating stops near 40 decided corners, so the reserve matters only when a sample
runs to look 4.

**#169 replayed (illustration only).** Read in sample order, #169's ratings put can't-tell at
sample positions 22, 97 and 98, and the one present at position 76 (the 75th decided). Under
40/60/80/100 the first 40 decided are all absent and end at position 41: **PASS at look 1**,
Wilson lower bound 0.893 at z = 2.192, after 41 units and 22.4 of the 52.0 logged minutes. The
can't-tell at position 22 is in that window; counted as present, the read is 40/41 with a lower
bound of 0.855, still above 0.85. #169 was rated in sha1 order, so this is not what a rater
would have seen. Its result remains the n = 97 amendment.

**What can't-tell can still do.** With a reserve, rating a doubtful present `cant_tell` costs
nothing, because a reserve corner replaces it. If a rater did that with a share q of true
presents, a true precision of 0.85 would pass with probability 0.045 at q = 0.10, 0.119 at
q = 0.25 and 0.47 at q = 0.50, while can't-tell would be only 1.7%, 4.2% and 8.1% of corners.
A ceiling on the can't-tell share would not catch it. `score` therefore lists the can't-tell keys
inside the counted window beside the verdict (`sequential.cant_tell_keys`) and adds a
non-binding `worst_case` that counts each one present. Making the worst case binding would
cost too much power (at 0.95 precision and 5% can't-tell, P(PASS) would fall from about 0.88 to
0.27 by Monte Carlo, in the #183 review). A reviewer reads the listed can't-tells.

**Descriptive fields.** In a `/2` score JSON, `n_rated` through `wilson95` and `by_stratum`
cover every rated sample corner, reserve and beyond the stop included. They are descriptive, and
optimistic after an early stop. The verdict is `sequential`. Stratum caps (the issue's third
item) are not built.

Where each number lives: the boundaries and the 0.85 / 0.90 / 0.95 / 0.99 characteristics in
`sample.json` `design` (from `make_design`); the 40/60/80/100 boundaries, z and the 0.85 row are
pinned in `tests/test_corner_labels.py::test_sample_records_the_reserve_and_the_look_design`.
Regenerate the tables:

```bash
cd scripts && python -c "
import corner_labels as cl
fx, ex = cl.make_design([100]), cl.make_design([40, 60, 80, 100])
print(cl.design_text(ex)); print(cl.look_boundaries([40, 60, 80, 100], 1.96))
for p in (0.85, 0.90, 0.95, 0.97, 0.99):
    print(p, cl.design_oc(fx['looks'], p), cl.design_oc(ex['looks'], p))"
```

## Dry run (no server contacted)

`python send_to_ps.py runs/vancouver/corner_labels169/na_<read>/results.jsonl --dry-run`.
The full output is in each dir's `dry_run_summary.txt`.

| build | records | errors | labels | below 0.5 |
|---|---:|---:|---:|---:|
| discount | 5,053 | 0 | 6,131 | 0 |
| count | 2,677 | 0 | 2,894 | 0 |
| discount_sw | 1,664 | 0 | 1,821 | 0 |
| count_sw | 1,292 | 0 | 1,376 | 0 |

Every payload has `label_type: NoCurbRamp`, `model_id: corner-rule@80fc070659f7` and
`model_training_date: 11-05-2026`. `file_label_types` reads `['NoCurbRamp']`, so it is one POST
per line and the legacy sidecar.

A dry run skips the pre-send guards. Run directly, they give:
- `check_model_provenance`, `check_detection_decode` and `check_detection_border` pass;
- **`check_store_built` refuses all four files on this branch.** 5,043 of 5,053 (discount), 2,671 of 2,677
  (count), 1,661 of 1,664 (discount_sw) and 1,289 of 1,292 (count_sw) records carry
  `source_detail: "ps_store"`, because the Vancouver runs were rebuilt from the PS pano store
  (#56).

That guard exists so that a store-built file's CurbRamp detections, which are the city's live
AI labels, are not inserted twice. A NoCurbRamp file built on store panos duplicates no AI
label: these panos are on the server already, which is what makes them labelable. On this
branch the guard is still file-level, and only `--allow-store-file` passes it. **Decided: the
guard learns the label type.** Draft PR #179 (branch `store-guard-label-type`) makes
`send_to_ps.check_store_built` refuse only when CurbRamp is among `file_label_types`. All four
exports carry NoCurbRamp only, so they pass once #179 merges, without `--allow-store-file`.
`dry_run_summary.txt` was generated before #179 and still shows the refusal.

## Caveats

- **Observed is conditioned on labels.** The base run and the #241 run were built from the PS
  pano store, which holds the panos people labeled (plus a 300-pano unlabeled sample;
  `docs/corner-inventory.md`). A corner is "absent" only where such a pano saw it. Many of these
  panos will already carry crowd labels; the overlap with crowd NoCurbRamp labels is measured
  in "Crowd NoCurbRamp overlap" below.
- **Corner geometry.** At corner level #241 found the leg over-merge (A1), and the clean read
  was 0.58-0.64 there. That is the reason the pre-registered rule is at corner level and
  rated by people.
- **Forward hemisphere: reversed 2026-11-05.** The issue asked for it, and the first build
  required the corner point within 90 deg of `camera_heading`. That was a choice of the most
  natural view, not a visibility test, and it cost 362 (discount) and 180 (count) corners whose
  only panos face away. On review Jon Froehlich reversed it: a GSV equirect sees behind the car
  at the same resolution as ahead; in thousands of panos he has labeled he sees no front/rear
  difference; and the label is a point, with PS's pixel key wrapping x modulo the width. So
  the test is off by default and survives only as the opt-in `--hemisphere-deg`. Distance
  ranking is unchanged.

## Crowd NoCurbRamp overlap

Measured 2026-11-05 by a read-only pass over the Vancouver server's public API, two GETs, and
nothing sent: `/v3/api/rawLabels?labelType=NoCurbRamp&filetype=geojson` (fetched
2026-11-05T23:59:40Z) and the same for `NoSidewalk` (23:59:42Z). Labels were matched by
`pano_id` and by haversine distance, never by `label_id`. **These numbers are for the
first build** (hemisphere test on, no sidewalk layer): the first build's
`na_discount/proposals.jsonl` (commit 7fc5cbf), sha256
`1062ac646afebe23f713fb0c3eaaa0151b00046ecf017cd46755527787488281` (5,770 proposals), and the
first build's `na_count/proposals.jsonl` (commit 7fc5cbf), sha256
`07b8d0368414e1722a613ac2c368b1bb139ea01363932475b2188abb7137e5cc` (2,715). Both files have
since been rebuilt, so these sha256s do not match the current `proposals.jsonl`. The measuring script and the API pulls are not committed.

- **The crowd's NoCurbRamp set is tiny.** 34 labels in the run area, from 6 users, none by the
  AI account; 27 were never validated (`correct`: 4 true, 3 false, 27 null).
- **Proposals on top of them.** Under discount, 24 of 5,770 proposals fall within 15 m of a
  crowd NoCurbRamp label and 7 share its pano; under count, 2 and 3. Most same-pano pairs point
  at a different part of the pano (heading gaps mostly 35-154 deg).
- **The reverse: the rule re-finds the crowd's labels under discount, not under count.** The
  discount read has a proposal within 15 m of 30 of the 34 crowd labels; the count read, of 2.
  Nearly all crowd NoCurbRamp labels sit at corners whose inventory holds only untyped `NA`
  points, which only the discount read proposes (22 of the 24 discount hits are such corners).
  That is weak evidence for the discount read, in the same direction as A8's read (b), but 27
  of the 34 labels are unvoted. Gallery 243 has since settled the read (discount, above).
- **Crowd NoSidewalk labels.** 42 of the 5,770 discount proposals lie within 15 m of one of the
  crowd's 116 NoSidewalk labels (9 of 2,715 under count). Only 1 of the 116 is validated, so this
  is a floor on the crowd's coverage, not a rate, and it says nothing against requiring the
  sidewalk layer.

Duplicates on the test server are not an issue at this scale (Jon's decision); a crowd and an
AI label at the same corner are what the validators compare.

## Deferred

- **The per-direction crossing-ends arm.** No crossing-line source exists in the run dirs
  (RampNet crosswalk sub-issue), so it is not built. The issue scores it separately, so its
  absence changes nothing above.
- **The #69 ranked CSV.** `rank` is implemented against the minimal contract above and is
  tested. It has not been run, because there is no CSV yet.
- **A sidewalk layer for other cities.** Vancouver now uses the city's own layer (above).
  The hook (`--sidewalk-presence <geojson>`, `--sidewalk-keep`, shapely >= 2, pinned in
  requirements.txt: `STRtree.query` returns indices) drops corners with no kept sidewalk
  geometry within `--sidewalk-radius-m` (10 m) of the corner point, records
  `dropped_no_sidewalk`, and puts the file's sha256 in `build.json`. Without a layer
  `build.json` records `sidewalk_rule: not applied`. PS's own sidewalk-presence table
  (SidewalkWebpage#5279) or OSM `footway=sidewalk` would be the source where a city publishes
  none; OSM was not needed here.
- **Scoring.** Done 2026-11-11 on the split-wide bundle: PASS under the n = 97 amendment
  ("Result" above).
- **The test-server pilot.** It needs Jon's clearance of the exact send (build, endpoint,
  confidence, stratum scope; see the crossing notes under "Result").
- **`corner_gallery_score.py` on a sample bundle: fixed 2026-11-11.** It used to validate this
  bundle's verdicts and then crash in its report (`KeyError: 'na_noramp'`, since a sample
  bundle's `snapshot.json` has no gallery 243 populations). It now validates, prints the
  `corner_labels.py score` command recorded in `snapshot.json`, and writes no report
  (`test_generic_scorer_validates_a_sample_bundle_and_stops`).
- **The ASSETS'19 DC placement check.** It needs a store-built DC run.

## Replication

**Archive of the uncommitted inputs and outputs (2026-11-06).** Every file over the 2 MB line that
the builds above read or wrote is on makelab2 under
`/projects/makeabilitylab/sidewalk-auto-labeler/runs/`, in the same relative layout as `runs/`
here, with a `sha256sum -c` manifest beside it that was verified on the host after upload:

- `vancouver/corner_labels169.archive.sha256` (19 files, verified 2026-11-06T04:08:11Z, stamp
  `vancouver/corner_labels169_archive.done`): the four `proposals.jsonl` and four exported
  `results.jsonl` of `corner_labels169/na_*`, `sidewalks/cov_sidewalks.geojson`
  (sha256 `11503ed4…`), `corner_inventory_posdetect241/corners_full.jsonl` (`ec206060…`), and
  `results.jsonl` / `sites.jsonl` / `sites_meta.json` of the three runs the proposals draw on
  (`vancouver`, `vancouver_posdetect241`, `vancouver_posdetect241_gsv`).
- `vancouver/corner_labels169_split.archive.sha256` (6 files, verified 2026-11-06, stamp
  `vancouver/corner_labels169_split_archive.done`): `corner_inventory169_split/corners_full.jsonl`
  (`b50da32e…`, the pinned split build) and `corners_224.jsonl`, and the four
  `corner_labels169_split/na_*/proposals.jsonl` (`b4e9e97e…`, `16c0a5dc…`, `1562e076…`,
  `57105551…`).

The sha256s match the ones `build.json` records, so a re-run can be checked against the archive
without re-downloading the sidewalk layer.

The sidewalk file is not committed; re-download it with the query recorded in
`sidewalks/cov_sidewalks.geojson.source.json` and check its sha256 (the weekly-updated layer
will have moved, so a re-download builds a different `_sw` set; `build.json` would show the new
sha256). `dry_run_summary.txt` holds the dry run's summary lines plus the four pre-send guards
called directly.

```bash
C=../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl
SW=runs/vancouver/corner_labels169/sidewalks/cov_sidewalks.geojson
for R in discount count; do for V in "" _sw; do
  D=runs/vancouver/corner_labels169/na_$R$V
  LAYER=(); [ -n "$V" ] && LAYER=(--sidewalk-presence $SW \
      --sidewalk-keep STATUS=Available,Pathway,Private)
  python scripts/corner_labels.py build --na-read $R --na-read-settled --corners $C \
      --run-dir ../../sal-vancouver/runs/vancouver \
      --extra-run ../posdetect241/runs/vancouver_posdetect241 \
      --extra-run ../posdetect241/runs/vancouver_posdetect241_gsv --out $D "${LAYER[@]}"
  python scripts/corner_labels.py sample --proposals $D/proposals.jsonl --out $D/sample.csv
  python scripts/corner_labels.py export --dir $D
  python scripts/corner_labels.py score --proposals $D/proposals.jsonl \
      --verdicts runs/vancouver/corner_gallery243/verdicts__jonf.json --out $D/score_gallery243.json
  python scripts/corner_labels.py score --proposals $D/proposals.jsonl --sample $D/sample.csv \
      --verdicts runs/vancouver/corner_gallery243/verdicts__jonf.json --out $D/score_sample.json
  python send_to_ps.py $D/results.jsonl --dry-run > /dev/null
done; done
```
