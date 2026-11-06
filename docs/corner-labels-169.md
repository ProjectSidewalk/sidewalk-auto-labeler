# NoCurbRamp labels from absent corners (issue #169)

Tool: `scripts/corner_labels.py` (`build`, `rank`, `export`, `score`, `sample`). Tests:
`tests/test_corner_labels.py`. Outputs: `runs/vancouver/corner_labels169/na_<read>/` (no
sidewalk layer) and `.../na_<read>_sw/` (with the city sidewalk layer, **the recommended pilot
input**), for `<read>` = `discount` and `count`. The layer's provenance is
`.../sidewalks/cov_sidewalks.geojson.source.json`. Issue: [#169](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/169)
(umbrella ProjectSidewalk/Planning#36).

**Status: plumbing only. Nothing is scored, and nothing goes to any server under this PR.**
The pre-registered rule (below) cannot be scored yet, and the primary `NA` read it depends on
is not settled. When a pilot does go ahead, its input is the `_sw` build (with the sidewalk
layer) of whichever read gallery 243 makes primary, not the no-layer build.

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
  `docs/corner-inventory.md` first. **Not met.**
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

## The NA-read caveat

`--na-read` is **required and has no default**, because the primary read is not settled:

- `count` is corner-inventory read (a), "as written". Any inventory point excludes the corner,
  `NA` included, because the city recorded something there. Only corners with no point at all
  are labeled.
- `discount` is read (b) of amendment A8. `NA` points with no `RAMPTYPE` (`NA_noramp`) are read
  as the city's record of no ramp and do not exclude. Every other class does: `Available`,
  `NA_typed`, `RMV`, `Expired/Removed` and other. This is exactly read (b), the read the issue
  quotes at 0.965 at unit level.

Gallery 243 (RampNet#243) decides between them, and only 50 of its 95 units are rated so far.
Both reads were built.

## Vancouver counts

Merged #241 build, intersections only (mid-block units have no corners), `fusion/primary`.
Inputs are pinned by sha256 (`PINNED_SHA256`: the inventory build's inputs and gallery 243's
`corners_full`). All four dirs were built at aea5dc6 on a clean tree.

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
- the precondition is not met;
- n < 100 in every build but the no-layer discount one.

`score_sample.json` scores each dir's seeded sample (`sample.csv`, 100 corners,
`sample --seed 169`, drawn from that dir's proposals before any rating of it exists). Only
3 (discount), 0 (count), 3 (discount_sw) and 2 (count_sw) sampled corners are already rated, so
all are NOT SCORED (n < 100). The sample to rate for the pilot is the `_sw` dir's of whichever
read gallery 243 makes primary.

What these descriptive numbers say before any pilot:
- **Guide rule 3 was not cosmetic, and the layer addresses it.** Without a layer, 46 of the
  116 corners rated absent under discount (19 of 51 under count) were rated `no_sidewalk`,
  where the guide asks for no label, so roughly 4 in 10 proposals at rated corners would have
  been labels the guide says not to place. With the layer, 2 of 31 (2 of 22) are. Label
  precision rises from about 0.6 to about 0.9, with wide intervals.
- **Only one rated corner was present**, in every build. Absence precision in the rubric's own
  sense is high wherever rated.

## Dry run (no server contacted)

`python send_to_ps.py runs/vancouver/corner_labels169/na_<read>/results.jsonl --dry-run`.
The full output is in each dir's `dry_run_summary.txt`.

| build | records | errors | labels | below 0.5 |
|---|---:|---:|---:|---:|
| discount | 5,053 | 0 | 6,131 | 0 |
| count | 2,677 | 0 | 2,894 | 0 |
| discount_sw | 1,664 | 0 | 1,821 | 0 |
| count_sw | 1,292 | 0 | 1,376 | 0 |

Every payload has `label_type: NoCurbRamp`, `model_id: corner-rule@aea5dc6fa00c` and
`model_training_date: 11-05-2026`. `file_label_types` reads `['NoCurbRamp']`, so it is one POST
per line and the legacy sidecar.

A dry run skips the pre-send guards. Run directly, they give:
- `check_model_provenance`, `check_detection_decode` and `check_detection_border` pass;
- **`check_store_built` refuses all four files.** 5,043 of 5,053 (discount), 2,671 of 2,677
  (count), 1,661 of 1,664 (discount_sw) and 1,289 of 1,292 (count_sw) records carry
  `source_detail: "ps_store"`, because the Vancouver runs were rebuilt from the PS pano store
  (#56).

That guard exists so that a store-built file's CurbRamp detections, which are the city's live
AI labels, are not inserted twice. A NoCurbRamp file built on store panos duplicates no AI
label: these panos are on the server already, which is what makes them labelable. But the
guard is file-level, and only `--allow-store-file` passes it. **Whether a NoCurbRamp pilot may
use that override, or the guard should learn the label type, is a decision for review.** It is
not taken here.

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
pre-revision proposal files** (hemisphere test on, no sidewalk layer): `na_discount`
sha256 `1062ac646afebe23f713fb0c3eaaa0151b00046ecf017cd46755527787488281` (5,770 proposals)
and `na_count` sha256 `07b8d0368414e1722a613ac2c368b1bb139ea01363932475b2188abb7137e5cc`
(2,715). The measuring script and the API pulls are not committed.

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
  of the 34 labels are unvoted. The real test remains the `na_noramp` units of gallery 243 still
  to be rated.
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
- **Scoring.** This needs the gallery 243 verdict on the `NA` read, then a gallery over
  `sample.csv` of the primary read's dir, then `score --sample`.
- **The ASSETS'19 DC placement check.** It needs a store-built DC run.

## Replication

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
  python scripts/corner_labels.py build --na-read $R --corners $C \
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
