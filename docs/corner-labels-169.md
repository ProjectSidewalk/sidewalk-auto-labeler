# NoCurbRamp labels from absent corners (issue #169)

Tool: `scripts/corner_labels.py` (`build`, `rank`, `export`, `score`, `sample`). Tests:
`tests/test_corner_labels.py`. Outputs: `runs/vancouver/corner_labels169/na_discount/` and
`.../na_count/`. Issue: [#169](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/169)
(umbrella ProjectSidewalk/Planning#36).

**Status: plumbing only. Nothing is scored, and nothing goes to any server under this PR.**
The pre-registered rule (below) cannot be scored yet, and the primary `NA` read it depends on
is not settled.

## What ships

1. **Corner -> label (`build`).** The input is the merged #241 corner table
   (`corners_full.jsonl`). A corner gets one NoCurbRamp proposal when all of these hold:
   - Its state under the decision rule's arm and observability variant (`fusion/primary` by
     default, `--arm` / `--variant`) is `absent`.
   - The inventory does not exclude it under the chosen `NA` read (`inventory_excludes`).
   - A sidewalk reaches it, when a sidewalk layer is given (`--sidewalk-presence`).
   - At least one pano passes the view test.

   **Pano choice ("the clearest view").** The candidates are the corner's `pano_ids_25`, the
   panos the table says observe it. A candidate must:
   - be within `--max-range-m` (25 m; larger is refused, since the table lists only 25 m);
   - be at least `--min-view-m` from the point (3 m; nearer panos look straight down);
   - hold the corner point in its forward hemisphere, i.e. within 90 deg of `camera_heading`.

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

Merged #241 build, intersections only (mid-block units have no corners), `fusion/primary`, no
sidewalk layer. Inputs are pinned by sha256 (`PINNED_SHA256`: the inventory build's inputs and
gallery 243's `corners_full`). Built at 7fc5cbf on a clean tree. The review rebuild left
`proposals.jsonl` byte-identical to the first build (same sha256).

| stratum | corners | absent | excluded by inventory | no candidate pano | proposed (discount) |
|---|---:|---:|---:|---:|---:|
| signalised | 1,004 | 51 | 14 | 4 | **33** |
| arterial | 4,743 | 1,754 | 73 | 71 | **1,610** |
| residential | 8,644 | 4,580 | 166 | 287 | **4,127** |
| all | 14,391 | 6,385 | 253 | 362 | **5,770** |

| stratum | excluded by inventory | no candidate pano | proposed (count) |
|---|---:|---:|---:|
| signalised | 23 | 3 | **25** |
| arterial | 686 | 51 | **1,017** |
| residential | 2,781 | 126 | **1,673** |
| all | 3,490 | 180 | **2,715** |

Projection dropped nothing under either read. The proposals sit on 4,781 panos (discount) and
2,515 panos (count). Under discount, the 5,770 proposals split into 2,715 corners with no
inventory point (the `count` set) and 3,055 holding only `NA_noramp` points. The chosen pano's
range to the corner point is 7.0 / 11.3 / 17.8 m (p10 / median / p90) under discount, and
8.7 / 13.4 / 19.0 m under count. Under discount, the newest-date preference moved the label
off the nearest candidate at 967 corners.

The chosen panos come from these runs:

| run | discount | count |
|---|---:|---:|
| vancouver_posdetect241 (the #241 position-selected run) | 4,143 | 1,632 |
| vancouver (base) | 1,614 | 1,074 |
| vancouver_posdetect241_gsv | 13 | 9 |

`proposals.jsonl` is 8.7 MB (discount) and 3.9 MB (count), both over the 2 MB line. So they and
the exported `results.jsonl` are not committed. `build.json` and `export.json` carry their
sha256, line and byte counts, and `build` reproduces them from the pinned inputs.

## What is not scored, and why

`score` against the gallery 243 blind ratings (`verdicts__jonf.json`) gives these reads.
**Both are NOT SCORED.**

| read | rated corners | absent | present | can't tell | precision [Wilson 95%] | `no_sidewalk` | label precision |
|---|---:|---:|---:|---:|---|---:|---|
| discount | 115 | 111 | 1 | 3 (0.026) | 0.991 [0.951, 0.998] | 45 | 0.589 [0.497, 0.676] |
| count | 51 | 48 | 1 | 2 (0.039) | 0.980 [0.893, 0.996] | 18 | 0.612 [0.473, 0.736] |

There are three reasons:
- that sample was drawn for #243, not for this rule: its strata are false absences, `NA`-only
  units and clean controls, so the corners are not a random draw of the proposals;
- the precondition is not met;
- under `count`, n = 49 < 100.

`score_sample.json` scores the seeded sample (`sample.csv`, 100 corners, `sample --seed 169`,
drawn from each read's proposals before any rating of it exists). Only 4 discount corners and 0
count corners in it are already rated, so it is NOT SCORED (n < 100).

Two things in these descriptive numbers matter before any pilot:
- **Guide rule 3 is not cosmetic.** 45 of the 111 corners rated absent under discount (and 18
  of 48 under count) were rated `no_sidewalk`, where the guide asks for no label. Another 52
  (20) carry no absence kind at all, so the true share may be higher. Without a sidewalk
  signal, roughly 4 in 10 proposals at rated corners would be labels the guide says not to
  place.
- **Only one rated corner was present.** Absence precision in the rubric's own sense is high
  wherever rated.

## Dry run (no server contacted)

`python send_to_ps.py runs/vancouver/corner_labels169/na_<read>/results.jsonl --dry-run`.
The full output is in each dir's `dry_run_summary.txt`.

| read | records | errors | labels | below 0.5 |
|---|---:|---:|---:|---:|
| discount | 4,781 | 0 | 5,770 | 0 |
| count | 2,515 | 0 | 2,715 | 0 |

Every payload has `label_type: NoCurbRamp`, `model_id: corner-rule@7fc5cbff8fbc` and
`model_training_date: 11-05-2026`. `file_label_types` reads `['NoCurbRamp']`, so it is one POST
per line and the legacy sidecar.

A dry run skips the pre-send guards. Run directly, they give:
- `check_model_provenance`, `check_detection_decode` and `check_detection_border` pass;
- **`check_store_built` refuses both files.** 4,771 of 4,781 and 2,507 of 2,515 records carry
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
  panos will already carry crowd labels, possibly crowd NoCurbRamp labels at the same corner.
  This PR does not check for duplicates against the crowd; the server is the place to.
- **Corner geometry.** At corner level #241 found the leg over-merge (A1), and the clean read
  was 0.58-0.64 there. That is the reason the pre-registered rule is at corner level and
  rated by people.
- **Forward hemisphere.** The issue asks for it. A 360 pano sees behind it too, so this is a
  choice of the most natural view, not a visibility test. It cost 362 (discount) and 180
  (count) corners whose only panos face away.

## Deferred

- **The per-direction crossing-ends arm.** No crossing-line source exists in the run dirs
  (RampNet crosswalk sub-issue), so it is not built. The issue scores it separately, so its
  absence changes nothing above.
- **The #69 ranked CSV.** `rank` is implemented against the minimal contract above and is
  tested. It has not been run, because there is no CSV yet.
- **The sidewalk-presence table** (SidewalkWebpage#5279). The `--sidewalk-presence <geojson>`
  hook (shapely >= 2, pinned in requirements.txt: `STRtree.query` returns indices) drops corners with no sidewalk geometry within `--sidewalk-radius-m` (10 m) of the corner
  point, and records `dropped_no_sidewalk`. Without it, `build.json` records
  `sidewalk_rule: not applied`, as here. No source was fabricated. The rated `no_sidewalk`
  share above says this hook needs a real layer before a pilot.
- **Scoring.** This needs the gallery 243 verdict on the `NA` read, then a gallery over
  `sample.csv` of the primary read's dir, then `score --sample`.
- **The ASSETS'19 DC placement check.** It needs a store-built DC run.

## Replication

```bash
C=../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl
for R in discount count; do
  D=runs/vancouver/corner_labels169/na_$R
  python scripts/corner_labels.py build --na-read $R --corners $C \
      --run-dir ../../sal-vancouver/runs/vancouver \
      --extra-run ../posdetect241/runs/vancouver_posdetect241 \
      --extra-run ../posdetect241/runs/vancouver_posdetect241_gsv --out $D
  python scripts/corner_labels.py sample --proposals $D/proposals.jsonl --out $D/sample.csv
  python scripts/corner_labels.py export --dir $D
  python scripts/corner_labels.py score --proposals $D/proposals.jsonl \
      --verdicts runs/vancouver/corner_gallery243/verdicts__jonf.json --out $D/score_gallery243.json
  python scripts/corner_labels.py score --proposals $D/proposals.jsonl --sample $D/sample.csv \
      --verdicts runs/vancouver/corner_gallery243/verdicts__jonf.json --out $D/score_sample.json
  python send_to_ps.py $D/results.jsonl --dry-run > /dev/null
done
```
