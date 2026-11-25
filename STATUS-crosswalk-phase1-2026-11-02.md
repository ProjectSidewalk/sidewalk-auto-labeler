# STATUS: crosswalk Phase 1 items 2–3, autonomous run 2026-11-02

Plan: `PLAN-crosswalk-phase1-2026-11-02.md` (this dir). Run by Claude Opus 5.5 (claude-opus-5-5). Nothing
public was posted: no PRs, no issue comments, no PS writes, no makelab writes.

## What's done

**Item 2 (primary): DONE and pushed.** Branch `crosswalk-gt-phase1` (RampNet), worktree
`D:/Git/RampNet-wt/crosswalk-gt-phase1`. Commits:
- [34a8871](https://github.com/ProjectSidewalk/RampNet/commit/34a8871): city table, seeded draw, archive fetcher, blind page, Seattle parity, pre-rate tooling, tests
- [b1c9600](https://github.com/ProjectSidewalk/RampNet/commit/b1c9600): the 320-row sheet + archive log (tile/crop sha256s, skips)
- [ce402c2](https://github.com/ProjectSidewalk/RampNet/commit/ce402c2): `docs/crosswalk_gt_phase1_274.md` (protocol, sample, rubric, reading rule, replication, where-each-number-lives) + docs index row
- [5f45508](https://github.com/ProjectSidewalk/RampNet/commit/5f45508): blind AI pre-rate `prerate_ai.csv` + the rubric gap it exposed
- [253cabd](https://github.com/ProjectSidewalk/RampNet/commit/253cabd): doc fix (archive size)

**Item 3 (stretch): STOPPED AT A DESIGN FORK, by the plan's rule.** Branch `stage1-label-type-274`,
worktree `D:/Git/RampNet-wt/stage1-label-type-274`:
- [6771440](https://github.com/ProjectSidewalk/RampNet/commit/6771440): `docs/stage1_label_type_274.md` (the options + a recommendation) and the one fork-independent piece, `stage_one/dataset_generation/crossing_source.py` (OSM `highway=crossing` → Stage 1 location rows under Phase 0's positive rule, with tests). No Stage 1 pipeline script was changed, so CurbRamp output is unchanged by construction. The parity test waits for the schema choice; the doc says what it will pin.

Tests: `tests/test_crosswalk_gt_phase1_274.py` 7 pass (the full Seattle page parity runs when
`CROSSWALK_SEATTLE_RAWLABELS` is set, and passed here); `tests/test_crossing_source_274.py` 4 pass; the docs
tests pass on both branches. Env: `C:\Users\jonf\anaconda3\envs\sidewalk-auto-labeler\python.exe`.

## The sample (320 rows)

| city | rows | marked / untagged / unmarked / osm_only | imagery |
|---|---|---|---|
| Amsterdam | 120 | 49 / 11 / 12 / 48 | PDOK 8 cm, 2024 + 2025 |
| Columbus | 120 | 31 / 31 / 27 / 31 | OGRIP 7.6 cm, 2025 |
| Chicago | 40 | 10 / 10 / 10 / 10 | Cook 15 cm, 2024 + 2025 |
| São Paulo | 40 | 12 / 11 / 6 / 11 | GeoSampa 12 / 10 cm, 2017 + 2020 |

The unit is the OSM node. Labeled strata prefer nodes with a human-confirmed label. `osm_only` = no crowd label
within 15 m but one within 100 m (an audited block). Amsterdam tags nearly everything `marked`, so its untagged
and unmarked cells are the whole population (11, 12) and the shortfall went to marked and osm_only. São Paulo
has **zero** human-confirmed Crosswalk labels, so every labeled row there is undecided. Three Chicago
candidates were outside Cook County imagery (blank tiles); the next ranks replaced them, as logged. Every one
of the 220 labeled rows has its crop.

## How to open the page

**`D:\Git\sal-scratch-keep\phase1-crosswalk-gt\page\crosswalk_gt.html`**: double-click it, or run
`Start-Process D:\Git\sal-scratch-keep\phase1-crosswalk-gt\page\crosswalk_gt.html`. It renders offline from
the archive beside it. Type your initials in the **rater** box first (save refuses without them). Open it in
ONE Chrome profile and stay there; Export when done (`crosswalk_gt_verdicts_<initials>.json` in Downloads).

## What you rate, and how long

320 rows. **Q1** on every row: Y (paint at the node, inner ring) / R (paint within 5 m, not at the node) /
N / U. **Q2** on the 220 rows with a red crowd label: 1–4 / 0, as in Seattle; B for a bike crossing. Nothing
is pre-selected and there is no AI line (blind). Rows are shuffled across cities; a city filter is at the top.
At your Seattle pace (120 rows over a few sessions with three panels) budget **about 2–2.5 hours**: the 100
osm_only rows are Q1 only and fast. The rubric is collapsible at the top of the page.

**Decide BEFORE rating (rubric gap the AI raters exposed):** does a **bike-only crossing** (Dutch red or
block-and-dash cycle crossing) or **shark teeth** at the node count as paint for Q1? The four AI raters split
on it. My proposal is pedestrian paint only: rate it N or R and add a note. If you agree, nothing on the page
changes; I recode the ~8 affected AI rows before agreement is read.

**After rating:** commit the export to `analysis_out/crosswalk_gt_phase1_274/`, then run
`python scripts/analysis/crosswalk_gt_phase1_274/gt_page.py merge --sheet analysis_out/crosswalk_gt_phase1_274/sheet.csv --verdicts <export> --prerate analysis_out/crosswalk_gt_phase1_274/prerate_ai.csv --out analysis_out/crosswalk_gt_phase1_274/sheet.rated_<initials>.csv`.
It prints the per-city × stratum table and the human-vs-AI agreement. **Don't open `prerate_ai.csv` before you
rate.**

## Decisions I made (reversible)

1. **One multi-city page, not four**: one export and one rater box; the localStorage split across browser profiles in Phase 0 is the reason.
2. **The page renders from the local archive only**, not live services. It is not committed (it would point at files that are not in the repo).
3. **Blind page**: no stratum, OSM tags, label tier, votes or AI in the payload (enforced by a test). Unlike Phase 0, nothing is pre-selected. The OSM node link still shows tags if clicked; the page says to rate first.
4. **Rubric change: Q1 gains R** (paint within 5 m, not at the node), for São Paulo's junction-tagged nodes; Y alone equals Phase 0's yes.
5. **The unit is the OSM node** (Phase 0 drew labels). Label tier: confirmed first, undecided only to fill.
6. **`osm_only` rule**: no label within 15 m, at least one within 100 m; 10–15 m is a deliberate gap.
7. **Tiles**: one 30 m (ground) × 800 px tile per year; the zoom is a CSS 2x. This sidesteps the small-extent blank-tile trap (KC 2025) for every service.
8. **GeoSampa TLS**: curl/Schannel rejects the Sectigo R46 chain; Python's verified TLS accepts it. Nothing ran with verification off.
9. **Seattle parity**: the Phase 0 template was generalized, and filling it from the city table reproduces the committed Phase 0 page byte for byte (sha256 `7c3506e4…f288`). Phase 0's `aerial_spotcheck.py build` was split into `render()` + `build()`, output unchanged.
10. **AI pre-rate**: four fresh Opus 5.5 subagents, 80 rows each, given only composites + rubric (1.10M subagent tokens, ~5 min, subscription, no API spend).
11. **No city dropped.** Tucson and Madison are config-only (terms unconfirmed), as planned.

## Blocked / for Jon

- **The replay archive lives on one machine**: `D:\Git\sal-scratch-keep\phase1-crosswalk-gt\` (964 MB, `SHA256SUMS` 1,356 files, `manifest.json`, `README.md`). Copy it to `/projects/makeabilitylab/RampNet/archives/crosswalk_phase1_274/` when convenient (Claude cannot write on makelab). No tarball is staged yet; ask and I will stage one with an install script, as in Phase 0.
- **Item 3 needs your decision** on fork A (run-level vs per-point vs parallel keys) and B (what places a crossing). My recommendation is A1 run-level + B1 node-projected for a first run, with B decided after this GT is scored.
- PRs and the #274 comment are drafted below, not posted.

---

## DRAFT PR body: `crosswalk-gt-phase1` → main

```markdown
## Crosswalk Phase 1 item 2: multi-city GT page and seeded draw (#274)

Built, not yet rated. A seeded 320-row ground-truth sheet in four cities, a blind local rating page, a
blind AI pre-rate held apart from the page, and the protocol, written before anyone rates:
`docs/crosswalk_gt_phase1_274.md`.

- **City table** (`scripts/analysis/crosswalk_gt_phase1_274/cities.py`): PS server, ortho sources
  (service kind, URL, layer, GSD, minimum fillable extent, credit) and terms per city. Drawn: Amsterdam
  (PDOK 8 cm, CC BY 4.0) 120, Columbus (OGRIP 7.6 cm) 120, Chicago (Cook 15 cm) 40, São Paulo (GeoSampa
  10 cm, CC BY-SA) 40. Tucson and Madison are config-only until their terms are confirmed.
- **Seattle parity**: the Phase 0 page's Seattle-specific strings now come from the table; filling the
  generic template from the Seattle row reproduces the committed Phase 0 page byte for byte (test included).
- **Draw** (`draw.py`, seed `crosswalk-gt-phase1-274`): one row per OSM node; strata marked / untagged /
  unmarked (nodes with a crowd label within 10 m, confirmed labels first) plus **osm_only** (no crowd label
  within 15 m, in an audited block).
- **Archive** (`archive.py`): tiles and crops go to a local replay archive with manifest + sha256 and
  SHA256SUMS; blank tiles are skipped and logged (3 Chicago candidates outside Cook coverage). No imagery
  is committed; `archive_log.json` carries every file's sha256.
- **Page** (`gt_page.py build`): one page for all cities, rendered from the archive. The payload carries no
  stratum, tags, tier or AI rating (tested). Rubric: Phase 0's, plus Q1 **R** (paint within 5 m, not at the
  node) for junction-tagged nodes.
- **AI pre-rate** (`prerate.py`): four fresh raters given only composites + rubric; `prerate_ai.csv` is read
  only by `merge`.

Rating: Jon, ~320 rows (~2–2.5 h). Then `gt_page.py merge` gives the per-city × stratum table and the
agreement; results are added to the doc in a follow-up commit on this PR.

Tests: `python -m pytest tests/test_crosswalk_gt_phase1_274.py`.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5
```

## DRAFT PR body: `stage1-label-type-274` → main

```markdown
## Stage 1 label_type axis: the design fork, and an OSM crossing source (#274 Phase 1 item 3)

Stops at a design fork, as the Phase 1 plan asks. `docs/stage1_label_type_274.md` lays out:
- **A, where `label_type` lives**: run-level (A1), per-point field (A2) or parallel keys (A3);
- **B, what places a crossing** in the bearing strip: the OSM node projected (B1), the Vistas mask (B2), or a
  crosswalk crop model (B3);
- **C, the positive rule per city**, which the Phase 1 GT measures.

It recommends A1 + B1 for a first run, with B settled after the Phase 1 GT is scored.

Built: `stage_one/dataset_generation/crossing_source.py`, which turns Overpass `highway=crossing` nodes into
Stage 1 location rows (the curb ramp columns unchanged, plus `label_type`, `source_id` and `osm_class`) under
Phase 0's Seattle positive rule, passed as arguments. Tests: `tests/test_crossing_source_274.py`. No pipeline
script changed; the CurbRamp parity test is specified in the doc for when A is chosen.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5
```

## DRAFT comment for RampNet#274

```markdown
## Phase 1 items 2–3: GT round built (not yet rated); Stage 1 label_type at a design fork

**Item 2.** A seeded 320-row crosswalk GT sheet: Amsterdam 120 (PDOK 8 cm), Columbus 120 (OGRIP 7.6 cm),
Chicago 40 (Cook 15 cm, the presence-at-15-cm test) and São Paulo 40 (GeoSampa 10 cm, the junction probe).
The unit is the OSM crossing node, stratified marked / untagged / `unmarked` (nodes with a crowd label within
10 m) plus a new **osm_only** stratum: OSM crossings with no crowd label within 15 m in a block the crowd
audited, so the round measures more than tag precision on crowd-found crossings. Every service was probed
before fetching; no city was dropped. The rating page is a blind local page (no stratum, tags or AI rating on
it); a blind AI pre-rate is held apart for agreement afterwards. The Phase 0 Seattle page is now built from
the same city table, byte for byte. Two things to read before the numbers: Amsterdam tags almost every
labeled crossing marked (its untagged and `unmarked` cells are 11 and 12 nodes, the whole population), and São
Paulo has no human-confirmed Crosswalk label, so its labeled rows are undecided ones. Protocol, rubric and
the reading rule, stated before rating: `docs/crosswalk_gt_phase1_274.md` on branch `crosswalk-gt-phase1`.

**Item 3.** Stage 1's label type is implicit everywhere (`curb_ramps_coords`) and its placer is a curb ramp
crop model, so adding Crosswalk forks on where `label_type` lives (run-level, per-point, parallel keys) and on
what places a crossing (the OSM node, the Vistas mask, a new crop model). Options and a recommendation
(run-level + node-projected for a first run) are in `docs/stage1_label_type_274.md` on branch
`stage1-label-type-274`, with an OSM crossing source reader that does not depend on the choice.

Next: the human rating of the 320 rows, then the per-city table and the agreement.

🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5
```
