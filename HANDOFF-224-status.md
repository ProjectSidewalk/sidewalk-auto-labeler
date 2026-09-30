# STATUS: RampNet#224 cluster-review GT — night run 2026-09-29 (Opus 5.5 implementer)

Ready for Jon to open the pilot gallery. Nothing was posted to GitHub (no PR, no issue comment),
nothing was sent to any Project Sidewalk server, no frozen pull was refreshed, and no
assignments file exists anywhere outside test fixtures. All five plan milestones are done.

## What exists, and where

| repo / branch | HEAD | pushed | contents |
|---|---|---|---|
| RampNet `cluster-review-224` (off main) | `6550ad4` (review fixes; was `244098b`) | yes | M1 `benchmark/RUBRICS.md` §6 + `docs/cluster_review_protocol.md`; M3 `scripts/cluster_review_gallery.py`, `rampnet/cluster_review.py`, `scripts/cluster_review_agreement.py`, tests, README/city-doc pointers, .gitignore; M5 `benchmark/vancouver/cluster_review/{snapshot.json, corners.jsonl, crops_missing.csv, report.md, export_runs.json}` |
| sidewalk-auto-labeler `cluster-review-224` (off `vancouver-56-scoring`, worktree `D:/Git/sal-cluster-review`) | `bd1d428` (review fixes; was `39ecc9c`) | yes | M2 `scripts/export_cluster_review.py`, `inventory_clustering.server_arms`, site_explorer crop-worker extension, tests; M4 `assignment_metrics` / `paired_ramp_table` / `assignment_verdict`, `scripts/cluster_review_score.py`, tests, CLAUDE.md block, `docs/ps-clustering-eval.md` section; M5 `runs/vancouver/cluster_review/report.md` + `score/{report.md, arms.csv, ramps.csv}` |
| sal-working-notes `working-notes` | (this commit) | local commit | `224-preregistration-comment.md` (DRAFT, not posted), this note |

Untracked, local only: pixels `D:/Git/RampNet/benchmark/vancouver/cluster_review/{crops (78 MB, 1,365 files), aerial (4.2 MB, 80 files)}`, the built pilot gallery `.../cluster_review/gallery/index.html`, the Overpass cache `D:/Git/sal-cluster-review/runs/vancouver/cluster_review/osm.json` and Esri tile cache `.../tiles/` (1,050 tiles). `D:/Git/sal-vancouver` was only read.

## Commands for Jon

All from `D:/Git/RampNet` unless noted. The gallery references `../crops` and `../aerial` by
relative path, so open it from where it was built (a `file://` URL works; tested in headless Edge).

**Rater A (pilot, deployed seed) — already built:**
```
start benchmark\vancouver\cluster_review\gallery\index.html
# rebuild: python scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review --pilot
```
Put your name in the "Review notes" panel. Export downloads `assignments.json`; save it as
`benchmark/vancouver/cluster_review/assignments.json`. (If you prefer a named file, add
`--rater jonf`; the export is then `assignments__jonf.json` and the scorer needs
`--assignments assignments__jonf.json`.)

**Rater B (pilot, every unit seeded with its `rater_b_seed`: 15 fusion / 15 deployed):**
```
python scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review --pilot \
    --rater <name> --role b --seed-arm auto --out benchmark/vancouver/cluster_review/gallery/<name>
start benchmark\vancouver\cluster_review\gallery\<name>\index.html
```
Export downloads `assignments__<name>.json`; save it into the bundle. Two raters in one browser
do not collide (localStorage is keyed by bundle + rater + label-snapshot sha256).

**Keys:** click crops / aerial dots to select; `1`-`9` assign to group N; `n` new group (split);
`m` merge selected groups (click headers) or the groups of the selected labels; `x` not a ramp;
`u` unsure; `0` unassign; `a` add-uncovered mode (click = add, click a mark = toggle unsure,
shift-click = remove); `p` place the selected group's ramp point, `P` reset it; `c` complete
(refused with an unassigned label); `z` undo; `←/→` units; `?` help. Inventory squares appear
only after a unit is complete. `elapsed_s` is shown in the title line.

**Agreement (after both exports):**
```
python scripts/cluster_review_agreement.py benchmark/vancouver/cluster_review \
    --a assignments.json --b assignments__<name>.json --pilot
```
**Score (from `D:/Git/sal-cluster-review`, ~45 s):**
```
python scripts/cluster_review_score.py vancouver --run-dir ../sal-vancouver/runs/vancouver \
    --bundle ../RampNet/benchmark/vancouver/cluster_review [--assignments assignments.json]
```
Writes `runs/vancouver/cluster_review/score/{report.md, arms.csv, ramps.csv}`. Today it says
NO GT YET (exit 0). The decision rule is read on rater A's full pass; pilot scores are descriptive.

**Re-export (resumable; never re-samples corners.jsonl; refuses a changed label pull):**
```
python scripts/export_cluster_review.py vancouver --run-dir ../sal-vancouver/runs/vancouver \
    --bundle ../RampNet/benchmark/vancouver/cluster_review --workers 16
```

## Decisions taken for Jon (plan table, unchanged)

| open question | decision tonight | why |
|---|---|---|
| Aerial imagery | Esri World Imagery tiles (z20) with the same attribution line `split_figures.py` uses; source recorded per unit and per bundle | it is what the #56 figures already used; a lab review tool; swapping the tile source later is one constant |
| Post the #224 pre-registration comment | NO. Draft it to a file (`224-preregistration-comment.md` in `D:/Git/sal-working-notes`) and as RampNet docs | handoff: "post it as a comment on #224 only with his OK" |
| #56 headline correction / SW stale-clusters issue | NO action | Jon's call |
| PR stack merge | NO action; the new branch stacks on `vancouver-56-scoring` (#118) | Jon's call |
| Vancouver's 33 human labels | IN the window, reviewed like any label; each label carries `user_kind: ai|human` so the scorer can exclude them per arm | the deployed partition holds them; excluding them would make `deployed` unscorable |
| 5+-leg / closely spaced OSM nodes | OSM intersection nodes within 25 m of each other are one unit (union of legs, centre = centroid), recorded as `node_ids` | one window per physical intersection |
| Gainesville / Richmond | not exported tonight; Vancouver is the only run | pilot first |
| Pushing | feature branches pushed; NO PRs, NO issue comments, nothing POSTed to any PS server | RampNet CLAUDE.md |

## Decisions I took that the plan did not cover (all recorded in the protocol's "Decisions" table)

1. **Eligibility rule** (protocol rule 6): a unit centre must be inside the area polygon and
   ≤ 20 m from an OPEN street of the PS street pull. Without it, no-label units could sit where no
   imagery was ever looked at (28,437 OSM candidates were outside the area, 6,144 off PS streets).
2. **A label in two served clusters** (10 in the bundle; the stale deployed pull): seed = the lower
   `label_cluster_id`, all listed in `seed_group_all`. The scorer counts both clusters (so such a
   label can read as a split under `deployed`, which is the stale pull's real state).
3. **A label with no seed group** opens unassigned (1 label under deployed, 34 under fusion — the
   fusion arm drops labels on the 32 unplaceable panos).
4. **File names**: `:` → `_` in `aerial/` and `crops/` (Windows).
5. **Who is rater B**: the gallery's `--role b` (shows only double-rated units; `--seed-arm auto`
   then uses `rater_b_seed`). Rater A = no `--rater` → `assignments.json`.
6. **Overpass by GET** (the brief said GET; `position_check` POSTs) with `highway=traffic_signals`
   and `crossing=traffic_signals` nodes in the same query.
7. **Crops**: native crop at 45° resampled to 512 px (= 4096×2048-equivalent sampling), with a
   JPEG draft decode never below 4096 px wide (speed). site_explorer's worker gained an optional
   per-item `path` and job `draft_width`, and its `main()` is now `__main__`-guarded (a
   spawn-start Pool on Windows re-imports the file; behaviour on makelab2 is unchanged). Jobs
   without the new keys behave exactly as before.
8. **elapsed_s**: 1 s ticks while the tab is visible, each capped at 2 s; idle time on screen is
   included; the unit shown when the page loads starts accruing immediately.
9. **Rounding** of shares is half-up (3 of 20 no-label, 1 of 7 or 8 in the pilot, 10 of 50
   double-rated).
10. **Merge denominator** counts a cluster holding ≥ 2 ramp-assigned labels even when they are
    all one ramp (per the plan's text); primary merge = ≥ 2 ramps, strict (≥ 2 labels each) beside it.
11. **`server_arms`** is a verbatim duplicate of `score_server_arms`' construction (not a refactor),
    so `runs/vancouver/inventory_clustering/report.md` cannot change; it reproduces
    `split_figures/state.pkl`'s deployed / ps @ 7.5 m / fusion+attach partitions exactly.

## Measured tonight (nothing here is invented; sources: bundle `report.md`, `export_runs.json`)

- Inputs: labels sha256 `57c31c73dc75…` (64,847 features, fetched 2026-09-28T23:01:58Z);
  deployed clusters sha256 `d8a1e706…` (18,684, fetched 2026-09-30T01:39:55Z); OSM payload
  sha256 `5c7fbd74…` (16,588 street ways, 2,447 signal nodes, fetched 2026-09-30T04:55:33Z from
  overpass-api.de); PS streets sha256 `0d7ce797…`; inventory 11,355 kept points.
- Arms rebuilt: deployed 18,684; ps @ 7.5 / 10 / 12.5 / 15 m 17,451 / 16,034 / 15,421 / 15,080;
  fusion_server 18,414; fusion_server+attach 16,101 (2,274 labels attached). Identical to state.pkl.
- Candidates: 10,492 intersection nodes → 9,509 units after 25 m merging; 46,835 mid-block points;
  eligible 21,763 (signalised 303 [6 no-label], arterial 1,532 [433], residential 2,652 [1,098],
  mid_block 17,276 [14,692]). Spacing rejections: 0 in every stratum.
- Units: 80 = 20 per stratum; no-label 12/80 = 0.150; pilot 30 (8/7/8/7, 4 no-label, 622 labels);
  double-rated 40 (30 pilot + 10 of 50 others); rater_b_seed fusion 20 / deployed 20 (pilot 15/15).
- Labels: 1,406 on 559 panos (signalised 666, arterial 385, residential 278, mid_block 77);
  **0 human labels** fell in any window. Pano widths 16384: 1,230, 13312: 142, 3328: 34.
  Cameras: 1,365 from the run, 41 inverted from the labels.
- Labels per unit: all p0 0 / p25 2 / p50 12 / p75 29 / p90 47 / max 94; labelled units p50 16;
  pilot per-unit counts 0,0,0,0,1,1,2,3,3,4,6,6,10,12,12,13,19,20,21,23,30,34,34,36,38,48,49,51,52,94.
- Seed groups per labelled unit: deployed p50 4 (max 18), fusion p50 4 (max 12).
- Crops: 1,406 requested, **1,365 cut, 41 missing** — 14 panos absent from the PS store (verified
  with `ls` on makelab2; the same panos are the 41 inverted cameras). Aerials 80/80. Reconcile OK.
- Inventory points in the 80 windows: 225.
- Wall clock: arms 41.7 s, Overpass GET 6.4 s (first, trial run into the scratchpad; console /
  scratch `export_runs.json` output, NOT in a tracked file — the tracked `export_runs.json` shows
  the cached re-use, 0.2 s), sampling 12.3 s,
  aerials 360.7 s (1,050 Esri tiles fetched, none seeded — sal-vancouver's 94 cached tiles did not
  overlap), crops 55.9 s on makelab2 (16 workers), scorer ~45 s.
- Scorer self-consistency on the real bundle: fusion_server+attach vs an assignment derived from
  itself over all 80 units: split 0/296, merge 0/245 — OK (after the review fixes, with the 34
  labels fusion does not hold scored as its singletons: split 0/330, merge 0/245). Singletons
  added per arm on the bundle: deployed 1, ps @ t 0, fusion_server 34, fusion_server+attach 34;
  human labels excluded: 0 (none fell in a window).
- Tests after the review fixes: RampNet 2,290 passed, 1 skipped, 1 failed (the same pre-existing
  test below); auto-labeler 894 passed, 2 skipped.
- Tests before the review: RampNet `pytest -q` 2,285 passed, 1 skipped, **1 failed — pre-existing and untouched by the
  branch** (`tests/test_stage1_bearing_residual.py::test_great_circle_matches_the_geodesic_used_by_stage_1`:
  great-circle vs pyproj bearing differs by 0.08°; it runs only where pyproj is installed, which
  CI does not). New RampNet tests: 28 (`test_cluster_review.py`, `test_cluster_review_gallery.py`,
  node bootstrap included). Auto-labeler `pytest`: 887 passed, 2 skipped (opt-in / no local depth
  archive); new tests: 16 exporter + 8 scorer.
- The local Python lacks both repos' test deps (pyarrow/transformers/opencv/sklearn for RampNet;
  geojson/dotenv/mapbox-vector-tile/shapely for the auto-labeler), so the suites ran in two scratch
  venvs (`--system-site-packages` + the repos' own requirements) under the session scratchpad; the
  system Python was not modified. To re-run: `pip install -r requirements-dev.txt` (RampNet) /
  `-r requirements-test.txt` (auto-labeler) in any venv.
- Reviewer time: **not measured** — no unit has been reviewed.

## What failed or was skipped, and how to retry

- 41 crops cannot be cut: their panos are not in the store (`crops_missing.csv`). Those labels still
  appear in the gallery (greyed-out image) with their aerial dot, so the reviewer can still assign
  them from the aerial/other views or mark them unsure. Retry is only possible if the panos reach
  the store: re-run the export command (it cuts only what is missing and is not in the csv —
  delete the rows first to retry them).
- Not done by design: posting the pre-registration comment (draft in `224-preregistration-comment.md`),
  Gainesville/Richmond bundles, any GT, any PR.
- The gallery was verified by headless Edge (render screenshot + scripted clicks/keys: split,
  not_ramp, unsure, merge, undo, place, uncovered, complete-refusal, export shape), not by a human;
  expect UI polish requests after the first real session — that is what the pilot is for.

## What is left for Jon

1. Read RUBRICS §6 / the protocol; if OK, post `224-preregistration-comment.md` on #224.
2. Review the 30 pilot units as rater A; recruit rater B for the same 30 with `--role b`.
3. Run the agreement CLI (pilot rule: pairwise ≥ 0.90 and κ ≥ 0.6) and score; revise rubric to v2
   before the full pass if the pilot fails.
4. Decide on the Esri imagery licence for a lab tool, the #56 headline correction, and the PR stack.

## Review fixes (2026-09-29, after an independent review; still before any unit was reviewed)

RampNet `6550ad4` ("M3 review fixes"), auto-labeler `bd1d428` ("M4 review fixes"). The pilot gallery
was rebuilt (`gallery/index.html`, 30 units, 622 labels) and re-probed in headless Edge (undo keeps
elapsed_s/note, inventory flags set on reveal and on reopen, uncovered-on-ramp click refused).

| # | defect | fix |
|---|---|---|
| 1 | a later-placed assignments file was shadowed by merely seeded browser state | local state wins only on units `seen` or `complete`; otherwise the file; conflicts listed in the notice; node test added |
| 2 | `--role b` without `--rater` shared rater A's storage/prefill/export | refused; a rater's export made under the other role is refused (`role_problem`) |
| 3 | undo rewound `elapsed_s` and the note | undo restores labels / ramps / uncovered / complete only |
| 4 | inventory could be seen, then the unit reopened and edited | sticky `inventory_seen`, `edited_after_inventory` on any later edit (reopen included), exported + validated; the scorer drops edited-after units from the inventory calibration |
| 5 | scorer printed but never compared the deployed-clusters / results.jsonl sha256 | refused on mismatch; `--allow-arm-mismatch` overrides and is written into the report |
| 6 | malformed label values accepted; foreign rubric only warned | values must match `^r\d+$` / not_ramp / unsure; `rubric_version != 1` refused; tests added |
| 7+8 | arms were scored on different label sets | **pre-registration change**: human labels excluded in every arm; a label an arm does not hold is its singleton; coverage arm-independent (`inventory_clustering.complete_arm`). Protocol "Metrics" + "Amendment", RUBRICS §6 "What is scored", draft comment updated |
| 9 | idle time accrued; saves only every 5 ticks | clock pauses after 60 s without keyboard/mouse input; save on `pagehide` and when hidden; documented in protocol + RUBRICS §6 |
| 10 | storage key could collide across re-samples | key adds the `corners.jsonl` sha256 |
| 11 | uncovered point on a ramp only caught by validate() | refused at click (and ramp placement onto an uncovered mark refused); Export refuses a complete unit that still has one |
| 12 | docstring `--out .../gallery_mikey` not git-ignored | now `gallery/<rater>` |
| 13 | first-run Overpass timing cited as if tracked | marked as console / scratch output above |

Decision taken while fixing that the brief did not spell out: flagging on *every* revealed unit would
empty the Vancouver inventory calibration (every completed unit reveals it), so the calibration drops
only units **edited after** the reveal; `inventory_seen` alone is reported, not dropped. Reopening a
unit counts as an edit (conservative).
