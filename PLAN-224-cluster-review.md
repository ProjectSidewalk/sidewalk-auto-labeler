# PLAN: corner-level cluster-review GT (RampNet#224) — implementation plan (2026-09-29, night run)

Written by Fable 5.1 from `HANDOFF-224-cluster-review-gt.md` + RampNet#224, for one Opus 5.5
implementer agent working both repos overnight without Jon. Jon's checkpoints (posting to GitHub,
reviewing GT, imagery-licence decision, PR-stack merge) are NOT taken tonight: the work stops at
"ready for Jon to open the gallery and review the pilot", with the pre-registration text drafted
as files he can post.

## Decisions taken for Jon tonight (each is reversible; all are flagged in the status note)

| open question | decision tonight | why |
|---|---|---|
| Aerial imagery | Esri World Imagery tiles (z20) with the same attribution line `split_figures.py` uses; source recorded per unit and per bundle | it is what the #56 figures already used; a lab review tool; swapping the tile source later is one constant |
| Post the #224 pre-registration comment | NO. Draft it to a file (`224-preregistration-comment.md` in `D:/Git/sal-working-notes`) and as RampNet docs | handoff: "post it as a comment on #224 only with his OK" |
| #56 headline correction / SW stale-clusters issue | NO action | Jon's call |
| PR stack merge | NO action; the new branch stacks on `vancouver-56-scoring` (#118) | Jon's call |
| Vancouver's 33 human labels | IN the window, reviewed like any label; each label carries `user_kind: ai|human` so the scorer can exclude them per arm | the deployed partition holds them; excluding them would make `deployed` unscorable |
| 5+-leg / closely spaced OSM nodes | OSM intersection nodes within 25 m of each other are one unit (union of legs, centre = centroid), recorded as `node_ids` | one window per physical intersection |
| Gainesville / Richmond | not exported tonight (no live AI labels in Gainesville; scope is Vancouver pilot). The exporter stays city-agnostic where cheap (a `--labels-from results` synthesised-label path may be stubbed with a clear error), Vancouver is the only run. | pilot first |
| Pushing | commit early and often on both branches; `git push -u origin <branch>` is allowed (feature branches). NO PRs, NO issue comments, nothing POSTed to any PS server. | RampNet CLAUDE.md: uncommitted work is one cleanup away from gone |

## Branching (do this first)

- auto-labeler: `git -C D:/Git/sidewalk-auto-labeler worktree add D:/Git/sal-cluster-review -b cluster-review-224 vancouver-56-scoring`.
  Vancouver run data is untracked and lives ONLY in `D:/Git/sal-vancouver/runs/vancouver/`
  (results.jsonl 35 MB, provenance_gate/raw_labels.geojson + .source.json, ps_clustering_eval/
  clusters.geojson, streets.geojson, inventory_oracle/inventory.geojson + inventory.json,
  sites*.jsonl, split_figures/state.pkl + tiles/). Read it by path (`--run-dir`), copy nothing,
  move nothing, `--refresh` nothing. Never write into sal-vancouver.
- RampNet: `git -C D:/Git/RampNet checkout -b cluster-review-224 main` (it is clean on main).
  Leave `D:/Git/RampNet-review` alone.
- Working notes: `D:/Git/sal-working-notes` is a worktree on branch `working-notes`; write the
  status note + draft comment there and commit.

Environment: local Python 3.12 has numpy, pandas, scipy, haversine, Pillow, torch-cpu; NO shapely.
node v20 is available (RampNet's box_gallery tests run viewer JS under node). makelab2 is reachable
through `D:\Git\dotfiles\wsl-ssh.ps1 makelab2 run "<cmd>"` (site_explorer.run_helper); the pano
store `/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa` is SHARDED `<id[:2]>/<id>.jpg`
(4,101 shard dirs); remote python is `/projects/makeabilitylab/sidewalk-auto-labeler/.venv/bin/python`
(3.9 — the crop worker string must stay 3.9-compatible). Read-only on makelab2; stage only under
`/homes/gws/jonf/.sidewalk_explorer/` as site_explorer does.

## Milestone 1 — pre-registration text (RampNet, files only)

1. `benchmark/RUBRICS.md` §6 "Corner-level cluster review — `benchmark/<city>/cluster_review/assignments.json`":
   what is one ramp (a dual-direction corner apron is TWO ramps, per the #116 box rule; a
   driveway is not a ramp; a ramp with no label is one `uncovered` point at the ramp), what
   `not_ramp` / `unsure` mean (unsure abstains: excluded from every metric), the complete
   attestation per unit, the note box, the resolution rule (crops at model resolution, 45° / 512 px),
   and that the seed biases toward itself (why the inter-rater subset is re-seeded from fusion).
   State `rubric_version: 1` and that files under different versions must not be compared.
2. `docs/cluster_review_protocol.md`: the pre-registered protocol — sampling rule (below), the exact
   file schemas (below), metric definitions (below), the decision rule (reuse
   `inventory_clustering.RULE_SPLIT_DROP=0.05 / RULE_MERGE_RISE=0.01 / RULE_COVERED_DROP=0.01`
   with `fusion_server+attach` vs `ps @ 7.5 m` as the pre-registered comparison), inter-rater
   agreement definitions and the pilot pass/fail rule, the calibration checks against the Vancouver
   inventory and against verdicts.json. Say explicitly which numbers are guesses (reviewer time).
3. `D:/Git/sal-working-notes/224-preregistration-comment.md`: the same content condensed as a
   ready-to-post issue comment, with a first line "DRAFT — Jon to post".

### Sampling rule (pre-registered)
- Units from OSM via one Overpass query over the run's area bbox (pattern: `position_check.fetch_osm_streets`;
  same `STREET_HIGHWAY_RE`; add `node["highway"="traffic_signals"]` and `node["crossing"="traffic_signals"]`
  to the union; `out geom;` returns way `nodes` ids + geometry). Cache the payload under the
  auto-labeler worktree's `runs/vancouver/cluster_review/osm.json` (untracked; sha256 + fetched_at in the report).
- Intersection node = node id shared by ways with total leg count >= 3 (a way passing through
  contributes 2 legs, an endpoint 1). Nodes within 25 m merge into one unit.
- Type: `signalised` if the node (or a merged node, or a traffic_signals node within 30 m) carries
  signals; else `residential` if every leg's highway is residential / living_street / unclassified;
  else `arterial`. `mid_block`: points along street geometry > 60 m from any intersection node,
  candidates every ~20 m along the way, window 30 m.
- Window radius 30 m about the unit centre. A label is in a unit if its SERVER lat/lng is within
  30 m. Units with 0 labels are drawn at a 15% share of each stratum. No two unit centres within
  60 m (`_SpatialIndex` rule from gt_gallery / export_benchmark). Candidates are shuffled with
  `random.Random(seed)`; seed 224; default per-stratum target 20 → 80 Vancouver units.
- `pilot: true` on 30 units chosen by sha1(corner_id) order with stratum quotas
  signalised 8 / arterial 7 / residential 8 / mid_block 7, keeping the 15% no-label share (≈4–5 of 30).
- Inter-rater: `rater_b_seed` = `fusion` on every other pilot unit in hash order (so half of the
  pilot is re-seeded), `deployed` on the rest. For the full pass, 20% of non-pilot units get
  `double_rate: true` with the same alternation.

### Schemas (pre-registered; the gallery and the scorer both read these)

`benchmark/<city>/cluster_review/snapshot.json`
```json
{"schema": "rampnet.cluster_review.snapshot/1", "city": "vancouver",
 "labels": {"kind": "server_pull", "path_in_run": "provenance_gate/raw_labels.geojson",
            "url": "...", "fetched_at": "...", "sha256": "...", "n_features": 64847,
            "ai_user_id": "51b0b927-3c8a-45b2-93de-bd878d1e5cf4", "tier": 0.55},
 "seed_arms": {"deployed": {"path_in_run": "ps_clustering_eval/clusters.geojson", "sha256": "...", "fetched_at": "..."},
               "fusion":   {"definition": "fusion_server+attach, auto frame, inventory_clustering.score_server_arms construction", "results_sha256": "..."}},
 "osm": {"query": "...", "fetched_at": "...", "sha256": "..."},
 "aerial": {"source": "Esri World Imagery", "url_template": "...", "zoom": 20,
            "attribution": "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"},
 "crops": {"fov_deg": 45, "px": 512, "resolution": "model (4096x2048-equivalent)", "store": "makelab2 vancouver-wa"},
 "sampling": {"seed": 224, "per_stratum": 20, "empty_share": 0.15, "window_m": 30, "spacing_m": 60, "merge_nodes_m": 25, "rule_version": 1},
 "exported_at": "...", "exporter": "sidewalk-auto-labeler scripts/export_cluster_review.py@<git sha>"}
```

`corners.jsonl` — one line per unit:
```json
{"corner_id": "vancouver:sig:000123", "city": "vancouver", "type": "signalised",
 "centre": {"lat": 0, "lng": 0}, "window_m": 30, "node_ids": [123, 456],
 "has_labels": true, "pilot": true, "double_rate": true, "rater_b_seed": "fusion",
 "aerial": {"file": "aerial/vancouver:sig:000123.jpg", "px": 512, "north_up": true,
            "bbox": {"south": 0, "west": 0, "north": 0, "east": 0}},
 "labels": [{"key": "24921", "label_id": 24921, "pano_id": "...", "user_kind": "ai",
             "pano_x": 8463, "pano_y": 3692, "pano_width": 13312, "pano_height": 6656,
             "x": 0.6357, "y": 0.5547, "lat": 0, "lng": 0,
             "camera": {"lat": 0, "lng": 0, "heading_deg": 0, "source": "run|inverted"},
             "capture_date": "2017-08", "crop": "crops/24921.jpg",
             "seed_group": {"deployed": "d3353", "fusion": "f17"}}],
 "inventory": [{"lat": 0, "lng": 0, "unit_id": "CR26188"}]}
```
Label `key` is the server `label_id` as a string for a live pull, `<pano_id>:<pano_x>:<pano_y>` for
synthesised labels (the pixel key `label_to_detection` uses). `inventory` is present only for
cities with one and is shown by the gallery ONLY after the unit is marked complete.

`assignments.json` (and `assignments__<rater>.json`), following verdicts.json conventions:
```json
{"schema": "rampnet.cluster_review/1", "city": "vancouver", "snapshot_sha256": "<labels sha256>",
 "rubric_version": 1, "seed_arm": "deployed",
 "review_notes": {"reviewer": "...", "reviewed_at": "...", "confidence": "high|medium|low", "summary": "", "caveats": []},
 "corners": {"<corner_id>": {
    "seed_arm": "deployed|fusion", "stratum": {"city": "vancouver", "type": "signalised", "has_labels": true},
    "labels": {"<key>": "r1" | "not_ramp" | "unsure"},
    "ramps": {"r1": {"lat": 0, "lng": 0}},
    "uncovered": [{"lat": 0, "lng": 0, "unsure": false}],
    "complete": true, "elapsed_s": 47.2, "note": ""}}}
```
Ramp positions: the gallery sets `ramps[r].lat/lng` to the mean of the group's label positions unless
the reviewer drags/clicks a point. `elapsed_s` accumulates while the unit is on screen and the tab
is visible (the pilot measures reviewer time; never fabricate it).

### Metrics (pre-registered; `assignment_metrics` in the auto-labeler)
Inputs: an arm = partition of label keys into clusters (city-wide); the assignment. Only `complete`
units count. Within a unit, only that unit's labels are looked at (a cluster's labels outside the
window are ignored). `unsure` labels are dropped from everything; `not_ramp` labels count only for validity.
- per GT ramp r (a key in `ramps` with ≥ 1 label assigned to it that the arm holds): `clusters(r)` =
  distinct clusters holding any of its labels; **covered** = |clusters(r)| ≥ 1; **split** = ≥ 2.
  split_rate = split / covered (Wilson CI). Per ramp, the number of clusters is kept so arms can be
  compared PAIRED (fixed / broken counts, as the #56 table).
- per cluster c touching the unit: `ramps(c)` = distinct r among its in-window labels; **merge** if
  ≥ 2 ramps each contribute ≥ 1 label (also report the stricter ≥ 2-each variant that
  `inventory_metrics` uses); merge_rate over clusters with ≥ 2 in-window ramp-assigned labels.
- **validity**: clusters whose in-window labels are all `not_ramp` / clusters touching the unit;
  labels `not_ramp` / labels.
- **coverage of the ramp population**: ramps with ≥ 1 label held by the arm / (ramps + uncovered
  points with `unsure: false`).
- pooled overall and by stratum type; per-city.
- Decision rule: `fusion_server+attach` vs `ps @ 7.5 m` on the same units: split_rate lower by
  ≥ 0.05 absolute AND merge_rate not higher by > 0.01 AND coverage not lower by > 0.01 → PASS;
  else NOT ESTABLISHED. Also report `deployed` and `ps @ {10, 12.5, 15} m`.
- Inter-rater (RampNet `rampnet/cluster_review.py::agreement`): over units both raters completed,
  pairwise same-ramp agreement on label pairs where both raters gave both labels a ramp key
  (Rand-style: agree if both say same-ramp or both say different-ramp; report the count of pairs
  and the agreement rate with a Wilson CI); Cohen's κ on `not_ramp` vs ramp per label (reuse
  `tag_review.cohen_kappa`); uncovered-point counts per unit (both totals and the per-unit
  |difference| distribution); refuse two files with different `rubric_version` or `snapshot_sha256`.
  Pilot pass/fail (pre-registered guess, to be revised only before the full pass): pairwise
  agreement ≥ 0.90 and κ(not_ramp) ≥ 0.6 → proceed to the full pass; else revise the rubric (v2).
- Calibration (auto-labeler scorer, only when a city has an inventory): reviewer ramps (assigned
  ramps + non-unsure uncovered) vs inventory points within the window, matched one-to-one within
  5 m, both directions; and `assignment_metrics` vs `inventory_metrics` on the same units.
- Against verdicts.json (RampNet loader): where a label key maps to a judged detection, its
  verdict must agree with `not_ramp` (report disagreements; nothing here for Vancouver, which has no bundle).

## Milestone 2 — exporter (auto-labeler, `scripts/export_cluster_review.py`)

`python scripts/export_cluster_review.py vancouver --run-dir D:/Git/sal-vancouver/runs/vancouver --bundle ../RampNet/benchmark/vancouver/cluster_review [--per-stratum 20] [--seed 224] [--no-crops] [--local-panos DIR] [--html-only-ish: --skip-aerial]`
- Reuse: `eval_ps_clustering.load_labels / label_to_detection / load_server_clusters / invert_camera_position / server_panos / attach_unplaceable`, `fuse_sites.load_at_height / project / fuse`, `inventory_oracle.load_inventory`, `position_check.fetch_osm_streets` pattern, `gt_gallery`-style `_SpatialIndex` (the auto-labeler has the original in `geo.py` / `export_benchmark.py`), `split_figures.tile / world_px / basemap` for the aerial mosaic (cache tiles under `runs/vancouver/cluster_review/tiles/`, same `20_x_y.jpg` naming; you may seed it from sal-vancouver's `split_figures/tiles` by COPY), `site_explorer.CROP_WORKER / fetch_crops_remote / fetch_crops_local` for crops.
- Fusion seed: use the same arm construction as `inventory_clustering.score_server_arms` (factor a `server_arms(...)` helper out of it ONLY if the existing tests still pass and `runs/vancouver/inventory_clustering/report.md` would be unchanged; otherwise duplicate the ~25 lines). `split_figures.load_state()` (state.pkl in sal-vancouver) holds exactly these three partitions and may be used as a cross-check, not as the source of truth.
- Crops: extend the crop worker to accept an optional per-item `path` (or a `sharded: true` flag) so it reads `<root>/<id[:2]>/<id>.jpg`; keep the default behaviour byte-identical for site_explorer. Cut at fov 45°, output 512 px, crosshair as today. Job → makelab2 → tarball → `crops/<key>.jpg`. Resumable: existing crops are never re-cut; `_missing.txt` names go to a `crops_missing.csv` in the bundle (tracked, small).
- Aerial: one 512 px north-up mosaic per unit covering ±35 m, written to `aerial/<corner_id>.jpg`, bbox recorded on the unit (Web Mercator, so latitude spacing is very slightly non-linear across 70 m — record `bbox` and let the gallery map lat/lng with the Mercator formula, not linear; or write `world_px` origin + scale). Attribution recorded.
- Reconcile at the end (like export_benchmark): every label in corners.jsonl has a crop or is listed in crops_missing.csv; every unit has an aerial file; write `report.md` (tracked, in the bundle) with counts per stratum, no-label share, spacing rejections, OSM/labels/clusters provenance (url, fetched_at, sha256), timing, and a copy under the worktree's `runs/vancouver/cluster_review/report.md`.
- .gitignore (RampNet): `benchmark/*/cluster_review/crops/`, `benchmark/*/cluster_review/aerial/`, `benchmark/*/cluster_review/gallery/`. (auto-labeler): allow `runs/*/cluster_review/report.md` + `*.csv`; ignore `osm.json`, `tiles/`.
- Tests (`tests/test_export_cluster_review.py`, lean, no network): intersection detection + typing on a tiny synthetic Overpass payload; node merging at 25 m; mid-block candidates; sampler spacing + empty share + determinism; window membership; corners.jsonl line validates against the schema; crop-worker sharded path (run the worker string under the local python on a 2-pano temp dir).

## Milestone 3 — gallery + loader (RampNet)

`scripts/cluster_review_gallery.py benchmark/vancouver/cluster_review [--rater NAME] [--seed-arm deployed|fusion|auto] [--pilot] [--out DIR] [--html-only]`
- A sibling of `gt_gallery.py` / `box_gallery.py`: self-contained `index.html` (crops and aerials referenced by relative path — copy or symlink nothing; the gallery dir sits beside `crops/`), localStorage autosave keyed by bundle + rater + snapshot sha256, Export downloads `assignments__<rater>.json` (or `assignments.json` when no rater), an existing file prefills for revision, `review_notes` panel, units interleaved by sha1(corner_id).
- `--seed-arm auto` = deployed for rater A; for a second rater the unit's `rater_b_seed`. Record the seed per unit in the export.
- Per unit: aerial panel with labels as dots coloured by current group, cameras as small triangles with a thin ray to the label (`site_explorer.plan_svg` conventions), north arrow + 10 m scale bar + attribution; inventory points appear only once the unit is `complete`. Crop strip per group (group header: id, n labels, capture dates), each crop 256 px on screen with the key and pano id, click to select. Keyboard: digits assign the selection to group N, `n` new group (split), `m` merge selected groups, `x` not_ramp, `u` unsure, `a` then click on the aerial adds an uncovered point (shift-click removes), `c` complete, `←/→` units, `?` help. Undo (`z`) within a unit.
- `elapsed_s` per unit accumulates while the unit is shown and `document.visibilityState === 'visible'`.
- Export refuses (with a dialog) when a complete unit has an unassigned label; warns when incomplete units exist (like gt_gallery's export warning).
- The one JS path that can destroy work — bootstrapping local state against a prefilled file when the snapshot changed — follows `box_gallery.STATE_BOOTSTRAP_JS`: a standalone pure function, tested under node.
- `rampnet/cluster_review.py`: `load_bundle(dir)` (snapshot + corners + optional assignments), `validate(assignments, corners, snapshot)` (schema id, snapshot sha256 match, every label of a complete unit has an entry, every ramp key referenced exists and has ≥ 1 label, uncovered points inside the window + not on top of an assigned ramp within 1 m, `elapsed_s` ≥ 0), `pairs(unit)`, `agreement(a, b)` per the pre-registration, `summary(assignments)` (units complete, labels by class, ramps, uncovered, median elapsed), `verdict_consistency(assignments, corners, verdicts)`; a small CLI `scripts/cluster_review_agreement.py` printing summary + agreement like `tag_review`.
- Tests (`tests/test_cluster_review.py` + `tests/test_cluster_review_gallery.py`): synthetic bundle round trip (load → build_html → export shape), validate() refusals (each rule), agreement on a hand-built pair with known Rand agreement and κ, the node bootstrap test.
- Docs: `benchmark/README.md` gains a short "cluster_review/" line in Layout pointing to the protocol; `docs/adding_a_benchmark_city.md` gets one sentence (optional pass).

## Milestone 4 — scorer (auto-labeler)

- `assignment_metrics(arms_or_one_arm, assignment_corners, ...)` beside `inventory_metrics` in `scripts/inventory_clustering.py` (pure, numpy-free is fine), plus `paired_ramp_table(arm_a, arm_b, ...)` (fixed / broken / both / neither).
- `scripts/cluster_review_score.py vancouver --run-dir <sal-vancouver run> --bundle ../RampNet/benchmark/vancouver/cluster_review [--assignments assignments.json] [--frame auto]`: rebuilds the arms (`deployed`, `ps @ 7.5/10/12.5/15 m`, `fusion_server`, `fusion_server+attach`) on the snapshot labels — the same construction as `score_server_arms` — restricts to the reviewed units, applies the metrics + decision rule, the inventory calibration when the city has one, and writes `runs/<city>/cluster_review/score/{report.md, ramps.csv, arms.csv}` (tracked). Reads assignments.json AS DATA (no RampNet import). If no assignments.json exists yet it must say so and exit 0 after writing a "no GT yet" report — Jon has not reviewed anything.
- Self-consistency check built in: scoring an arm against an assignment derived from that arm itself must give split 0 / merge 0 (used as a test and printed once as a sanity line).
- Tests (`tests/test_cluster_review_score.py`): toy arms + toy assignment with one split, one merge, one not_ramp cluster, one unsure label, one uncovered point → exact expected numbers; the paired table; decision rule PASS / NOT ESTABLISHED at the boundaries; the "no GT yet" path.
- CLAUDE.md (auto-labeler): ONE compact command block for `export_cluster_review.py` and `cluster_review_score.py` in the style of the existing ones; `docs/ps-clustering-eval.md`: a short "Cluster-review GT (RampNet#224)" section pointing to RampNet's protocol, stating that no GT exists yet.

## Milestone 5 — run the exporter for real, build the pilot gallery, hand off

1. Run the exporter for Vancouver (network: Overpass GET, Esri tile GETs, makelab2 SSH for crops). Expect ~80 units, a few hundred to ~1,500 labels, one crop each. If makelab2 crops fail, finish everything else and leave `--no-crops` output + the exact retry command in the status note.
2. Build `benchmark/vancouver/cluster_review/gallery/index.html` with `--pilot` and open nothing; record the path.
3. Run both repos' full test suites (`pytest -q`), commit, push both branches (no PR).
4. Write `D:/Git/sal-working-notes/HANDOFF-224-status.md` (commit on `working-notes`): what exists, exact commands to open the pilot gallery as rater A and as rater B (`--rater jonf --seed-arm auto`), how to export and where to save, how to score (`cluster_review_score.py`), what was decided for Jon (table above), what failed or was skipped, and every number measured tonight (unit counts per stratum, labels per unit distribution, crops cut / missing, time taken) — none invented.

## Guardrails (from the handoff; non-negotiable)
- GET-only against every PS server; nothing submitted; no `--refresh` of any frozen pull.
- The agent never fills in GT itself: no assignments.json is ever written by code except the
  gallery's Export in a browser, driven by a human. Test fixtures are synthetic and live under tests/.
- Never quote reviewer-time numbers as measured; the pilot measures them.
- Numbers to carry unchanged (Vancouver, server placement, r = 5 m): deployed 0.196, ps @ 7.5 m 0.138,
  fusion_server 0.199, fusion_server+attach 0.118; fusion+attach vs ps fixes 601 / breaks 399 / both 724;
  deployed vs ps 678 stale splits. Raycast-frame 0.255 / 0.200 / 0.091 are survivorship-flattered.
- Commit small and often; each commit message says which milestone; end with the attribution line
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` is NOT right for the Opus agent — use
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` on commits the Opus agent authors.
