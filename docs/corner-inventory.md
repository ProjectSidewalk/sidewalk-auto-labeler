# Corner-level ramp inventory: present / absent / unobservable (RampNet#238)

**Status:** PROTOCOL written 2026-10-04, before any number in this study was computed. The
RESULTS section is appended after the run and does not change the protocol; any deviation
found while implementing is recorded under "Deviations" with its reason. Companion tool:
`scripts/corner_inventory.py` (`build`, `score`, `score-assignments`, `census`). Issue:
[RampNet#238](https://github.com/ProjectSidewalk/RampNet/issues/238).

## Question

For every intersection corner in Vancouver, WA, does the shipped detector (through the
labeler's fusion, or through the deployed server clusters) say a curb ramp is **present**,
**absent**, or that we **could not see** the corner? Read against the city's curb-ramp
inventory, how often is an "absent" call clean? That last number decides whether the
longitudinal study (RampNet#238 experiment 2) is built.

**One hard-coded Vancouver fact.** The run is the Project Sidewalk pano store rebuilt
offline (`detect_from_store.py`, #56): 28,830 GSV panos the deployment looked at. It is
**not** every GSV pano in the city. "Unobservable" therefore means "no pano in the PS store
near the corner", not "no imagery exists".

## Inputs (read in place, never modified, hashes asserted by the tool)

| input | path | sha256 |
|---|---|---|
| detections | `sal-vancouver/runs/vancouver/results.jsonl` | `7fdf4005824f3edbebb93c6f365d25c54d1c61384b801b0213aaafa97ef79f28` (asserted) |
| fused sites | `…/sites.jsonl` + `sites_meta.json` | recorded at run time |
| deployed clusters | `…/ps_clustering_eval/clusters.geojson` | recorded |
| deployed labels (frame only) | `…/provenance_gate/raw_labels.geojson` | recorded |
| PS streets (eligibility) | `…/ps_clustering_eval/streets.geojson` | recorded |
| area | `…/area.geojson` | recorded |
| city inventory | `…/inventory_oracle/inventory.geojson` | must equal `inventory.json`'s sha256 (asserted) |
| OSM payload | `sal-cluster-review/runs/vancouver/cluster_review/osm.json` | `58ce83ffa4b8eea199d7017b53fe125079e2cfa2f3d5181f9e22d61bc0c2a030` (asserted) |
| #224 sampled units | `RampNet/benchmark/vancouver/cluster_review/corners.jsonl` | recorded |

The OSM functions (`street_ways`, `intersection_nodes`, `merge_nodes`, `classify`,
`build_candidates`, `PointIndex`, `SegmentIndex`, `open_ps_streets`, `validate_osm`, the
constants) are imported from `scripts/export_cluster_review.py`, not copied. The git SHA of
that file at run time is recorded in RESULTS.

## Definitions

**Unit.** An eligible #224 intersection unit, built by `build_candidates` exactly as the
exporter builds it (sampling rule v2 of RampNet `docs/cluster_review_protocol.md`): OSM
nodes with >= 3 street legs (rule 2), merged within 25 m (rule 3), typed signalised /
arterial / residential (rule 4), centre inside the area polygon and within 20 m of an open
PS street (rule 6), and no grade-separated way within the 30 m window (rule 6b). There is
**no sampling**: every eligible unit is in the table. The local ENU frame is the exporter's
(mean of the deployed labels' positions), and the build asserts that its eligible counts per
stratum equal the exporter's `snapshot.json` (`eligible`: signalised 284, arterial 1,497,
residential 2,648, mid_block 16,912). Mid-block candidates (rule 5, every 20 m along a way,
> 60 m from any intersection node) are built and reported **separately, at unit level only**:
a mid-block ramp is a crossing, and absence there is not a defect. Mid-block windows overlap
(20 m spacing, 30 m radius), so mid-block shares are shares of points along streets, not of
independent places. Neighbouring intersection windows can overlap too (units are >= 25 m
apart, windows 30 m); a site in two windows counts in both.

**Unit key.** `vancouver:<sig|art|res>:n<lowest OSM node id>` (mid-block:
`vancouver:mid:<lat>,<lng>` at 7 decimals). Stable across re-runs on the same OSM payload.

**Legs.** From each node of the unit, every OSM street edge (consecutive way nodes, the
same street filter as rule 1) leading to a node outside the unit is walked outward through
degree-2 nodes (way splits) until the walk first leaves a circle of `LEG_PROBE_M = 20 m`
about the unit centre; the leg bearing is the bearing from the centre to that exit point
(linear interpolation on the crossing segment). A walk that stops inside the circle (a dead
end, or another intersection node of degree >= 3) gives the bearing to where it stopped,
kept only if that point is >= `LEG_STUB_MIN_M = 5 m` from the centre. Edges between two nodes
of the same unit are internal and skipped. Bearings within `LEG_MERGE_DEG = 30°` of each
other (circular single linkage) are one leg, at their circular mean: the two carriageways of
a divided road and a slip lane beside its parent road are one leg, a real skewed leg
(>= 30° from its neighbour) is kept. 20 m is inside the 30 m window and far enough out that a
curving way's bearing is its approach direction, not its last metre.

**Corner.** The sector between two adjacent leg bearings, clockwise from north. A unit with
k >= 2 legs has k corners; a unit with fewer than 2 distinct legs after merging has one
corner covering 360° (counted and reported). Sites, deployed clusters, panos and inventory
points are assigned to a corner by their bearing from the unit centre, among points within
the 30 m window (`WINDOW_M`). A corner whose sector is wider than 150° (the far side of a T,
where there is usually no crossing) is flagged `wide`; corner-level tables are reported for
all corners and for non-wide corners. **The OSM corner is not always a legal crossing**: a
sector with no crosswalk needs no ramp, so corner-level absence over-counts defects.

**Corner point.** The unit centre moved `CORNER_OFFSET_M = 12 m` along the sector's bisector.
For two perpendicular streets with a curb-to-centreline half-width w, the curb corner sits at
w·√2 from the centre; w ≈ 8.5 m (two travel lanes plus parking or gutter) gives 12 m. It is
used only for the observability test below; membership is by sector.

**Observed.** Primary: at least one processed pano (a `results.jsonl` record, at its stored
lat/lng) within the fusion range, 25 m (`sites_meta.params.max_range_m`), of the corner point
(corner level) or of the unit centre (unit level). Two sensitivity variants reported beside
it: `>=2` (at least two panos within 25 m) and `15m` (at least one pano within 15 m). A rule
that a ray actually crossed the corner is a follow-up, not the primary.

**Present.** Arm `fusion`: at least one operational fused site (`n_operational >= 1`, i.e. a
member detection with confidence >= `sites_meta.params.min_confidence` = 0.30) in the corner
(unit level: in the window). Arm `deployed`: at least one deployed server cluster (its served
point, `clusters.geojson`, labels went live at tier 0.55) in the corner. The two arms share
the observability test.

**States**, in precedence order: **present** if the arm has something in the corner; else
**absent** if observed; else **unobservable**. A present corner that fails the observability
test (a site detected from a pano > 25 m from the corner point) is counted present, and the
number of such corners is reported.

**Inventory.** Points of `inventory.geojson` (all 17,440 in area, not only the 11,355 kept
`Available` points) in the corner or window, counted by status class: `Available`;
`NA_noramp` (`NA` with no `RAMPTYPE`, which `inventory_oracle.py` documents as the city
recording a corner without a ramp); `NA_typed` (`NA` with a `RAMPTYPE`); `RMV`;
`Expired/Removed`; `other` (null, FV, Clarify, Private, CONST). Per corner the table also
carries `INSTDATE`, `PROJNAME`, `DEFECT`, `CORNER`, `OWNER` of its points.

## Metrics

All at unit level and corner level, per stratum (signalised, arterial, residential, and the
three pooled; mid-block separately, unit level), for both arms, under the primary
observability definition and the two variants. Shares carry Wilson 95% intervals.

- **City sentence**: share of units / corners present, absent, unobservable.
- **Corner recall against the inventory**: among units / corners with >= 1 `Available` point,
  the share called present, absent, unobservable.
- **Absence precision**: among units / corners called absent, the share with no inventory
  point of any status (**clean read**), the share with >= 1 `Available` point (**false
  absence**: a detector miss or an observability error), and the share with only `RMV` / `NA`
  / other points (**RMV/NA only**). The three sum to 1.
  A second read is reported beside it and is **not** the decision rule: `1 − false absence`
  (absent with no `Available` point), because `NA_noramp` points are the city's own record of a
  corner with no ramp and so support an absence rather than contradict it.
- **Inventory gaps**: corners called present (fusion arm) with no inventory point of any
  status, listed with the site ids and the pano ids of their operational members (for a
  later gallery). If the detector is right, these are rows the city is missing; the deployed
  arm's count is reported beside it.

## Decision rule (verbatim from RampNet#238, written before scoring)

> **If absence precision against the inventory (clean read) is >= 0.90 at unit level, the
> longitudinal study (experiment 2) is unblocked.** Below that, the false-absence rows are
> triaged into detector misses vs observability before anything else is built, using the 12
> no-label #224 units and a 30-corner sample.

Applied to: the `fusion` arm, primary observability, unit level, the three intersection
strata pooled. The `deployed` arm, the per-stratum reads, the variants and the second read are
reported beside it and do not change the outcome. A point estimate >= 0.90 whose Wilson lower
bound is below 0.90 is reported as passing with that interval stated.

## #224 hook (`score-assignments`)

States for the 80 #224 units are emitted from their `corners.jsonl` centres and `node_ids`
(`corners_224.csv`, unit and corner rows). When a rater's `assignments.json` exists, every
complete, non-`cant_judge` unit is scored: the rater says **present** if the unit (or corner)
holds >= 1 ramp or >= 1 non-unsure `uncovered` point, else **absent**; `unsure` uncovered
points are counted separately. Ramps and uncovered points are assigned to corners by bearing,
as above. Reported: the confusion of our state (present / absent / unobservable, both arms)
against the rater's present / absent, absence precision against the rater, and recall against
the rater.

## GSV history census (experiment 2, step 1)

A stratified sample of 200 eligible intersection units (signalised 67, arterial 67,
residential 66; `random.Random(238)` over units sorted by key, one `sample` call per stratum
in that order). For every processed pano within 25 m of a sampled unit's centre: one
streetlevel metadata request (`api.find_panorama_by_id`, no depth, no image), the retry policy
of `sources/gsv.py` (`METADATA_ATTEMPTS`, same backoff), plus 0.25 s between requests. Every
raw response is cached as JSON with its fetch time; the cache's sha256 is recorded. Reported:
calls, failures (exact error), panos the endpoint no longer serves, captures (distinct
year-months over current + historical panos) per unit and per corner, earliest capture year,
span, the number of distinct historical panos in the sample, and an **extrapolation** (stated
as one) of the historical panos a full pass over every eligible unit would process. Wall-clock
recorded. No GPU, no paid API.

## Caveats that travel with every number

One city (Vancouver, WA); one rig (GSV); the inventory's completeness is unknown (it is the
city's inventory of city-owned ramps, `OWNER` null on ~4.9k points, and `NA` / null statuses
whose exact meaning is not documented); the PS store holds the panos the deployment looked
at, not every GSV pano; an OSM corner is not always a legal crossing; mid-block and
neighbouring windows overlap.

## Deviations from the protocol (found while implementing, before scoring)

- **Mid-block unit key.** The protocol named mid-block points `vancouver:mid:<lat>,<lng>`.
  Overlapping OSM ways put two mid-block candidates at the same position (the build stopped on
  a duplicate key), so mid-block points are keyed by their serial in `build_candidates` order,
  `vancouver:mid:m<serial>`. The exporter counts both points too, so the eligible counts still
  match. Intersection keys are as written.
- **Base commit.** `origin/cluster-review-224` (bd1d428) is still on sampling rule v1. Rule 6b
  (grade separation) and the re-drawn #224 bundle are two commits that exist only in the local
  `sal-cluster-review` worktree (5b21cd9, be54326), and the asserted OSM payload is the v2 one.
  This branch is stacked on be54326, so pushing it carries those two commits; the PR shows
  them until `cluster-review-224` is pushed.
- **Census fetch.** The census calls streetlevel's `api.find_panorama_by_id(download_depth=False)`
  directly rather than `sources.gsv.fetch_metadata_with_retry`, because the latter returns
  only parsed metadata (and asks for the depth payload), and the protocol caches the raw
  response. The retry count and backoff are `sources/gsv.py`'s. A response with status code 2
  (the endpoint no longer serves that pano) is recorded as `not_found` and not retried.
- **Added after the first score, reported, not used by the decision:** a table of which
  inventory classes sit at absent units and corners, and a "how the pano set was selected"
  diagnostic (below). Neither changes a definition or the rule.

## Amendments after review (2026-10-04, after the first RESULTS were published)

An independent review of the PR
([review](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/132#pullrequestreview-5407384626))
found that the leg rule did not do what the Definitions above say. Each finding below was
reproduced against the code before it was fixed. The protocol text above is left as written;
these amendments override it where they conflict. Unit-level numbers and the decision
computation do not use sectors and did not move. Every corner-level number in RESULTS was
re-derived. The superseded values are listed here so they stay visible.

- **A1. Divided roads (review B1).** The Definitions promise that "the two carriageways of a
  divided road ... are one leg". With a 20 m probe and a 30° merge, two carriageways only merge
  when their centrelines are less than 2·20·tan 15° ≈ 10.7 m apart. Most divided arterials are
  wider than that, so the median between them became a ~35° "corner" that was almost always
  called absent. Reproduced: `vancouver:sig:n47223902` had legs [1, 73, 109, 181, 253, 288];
  73/109 and 253/288 are the West Mill Plain Blvd carriageways, both `oneway=yes`.
  - **New rule** (`merge_legs`), applied before the 30° chaining. Each raw walk carries the tags
    of its first edge out of the unit.
    - (a) A `oneway`, non-`_link` walk joins the nearest other `oneway`, non-`_link` walk with
      the same `name` (or, when both are unnamed, the same `ref`) within `LEG_PAIR_DEG = 90°`.
      These are the two halves of one divided road, whatever the median width. 90° keeps the
      two directions of a road through the junction (~180° apart) separate.
    - (b) A `*_link` walk joins the nearest non-link walk within `LEG_LINK_DEG = 60°`. This is
      the slip lane the Definitions already promised to merge.
  - **Effect:**
    - Sectors < 45° fell from 466 to 53.
    - Among multi-node signalised units, 4 legs is now the most common count (77 of 110).
    - Corners went from 14,901 to 14,391. Legs per unit went from {1: 3, 2: 180, 3: 2,770,
      4: 1,286, 5–8: 190} to {1: 3, 2: 273, 3: 2,778, 4: 1,368, 5: 6, 6: 1}.
    - The units that went from 3 legs to 2, checked by hand on six, are slip-lane merges
      (`*_link` + a through road) and split oneway pairs of one street (teardrop islands); see
      "Known over-merges" below for the kinds a full sweep found.
  - **Tests:** a 4-leg cross with one divided arterial whose carriageways are 14 m apart gives
    4 corners, not 6; a slip lane joins its parent leg.
  - **Known over-merges (re-review, kept as a limitation; the rule was not changed).** The
    six hand-checked 3→2 units above are not the only kind of merge. A sweep of all 4,429 units
    under the current build found four kinds of over-merge. This was a scratch script that re-ran
    `leg_bearings` + `merge_legs` per unit and asserted the committed legs; it is not committed.
    - **Two ends of a loop or circle road.** The pair rule joins two oneway walks of one named
      road 60–90° apart. 16 units change leg count between a 60° and a 90° pair cap.
      - `res:n958722944` (Northeast 34th Way): walks at 35° and 121° give legs [78, 258]; it was
        a 3-leg T.
      - `res:n1253177406` (East 38th Loop): [83, 181, 250] → [83, 215].
      - `res:n1245728425` (Officers Row): [21, 109, 192] → [65, 192].
      - `art:n47278105`: 4 legs → [86, 269].
    - **Interchange ramp terminals taken by the `*_link` rule.** Here the link is the crossing
      approach, not a slip lane beside its parent. 42 link walks have their nearest non-link walk
      30–60° away, so they merge only because of the 60° link cap. Of the 8 multi-node signalised
      units that now have 2 legs, five are such terminals:
      - `sig:n249208576`: [69, 119, 252, 308] → [89, 289];
      - `sig:n697467152`: [129, 171, 320] → [150, 320];
      - `sig:n1723250519`: 6 legs → [142, 325];
      - `sig:n3993169612`: [70, 108, 238, 301] → [89, 269];
      - `sig:n47266006` (East 39th St): links at 45° and 110° both joined the 77° leg;
        [45, 77, 110, 257] → [77, 257].
    - **Chaining past 90°.** In `sig:n47250613` (Fourth Plain × Ward × 147th, 4 nodes), the
      Fourth Plain walks at 148°, 189° and 240° pair one after another and the links join them:
      6 legs → [9, 190].
    - **Links shift a leg's bearing.** The circular mean includes the link walk. In
      `art:n47196196`, the South Garrison walk at 214° and a `tertiary_link` at 265° become one
      leg at 239°, which moves a sector boundary by about 25°.
    - **Two more of the 8 two-leg multi-node signalised units:**
      - `sig:n441985883` (a median crossover on SE 164th Ave) is a correct 2.
      - `sig:n47282521` (Lincoln × W Fourth Plain, 4 nodes) was already 2 legs under the old
        rule, because its nodes sit beyond the 20 m probe and Lincoln Avenue's legs are mostly
        lost. That is a known gap for wide multi-node units, not a regression.
    - **Scale:** about 20–60 of the 4,429 units. Unit-level numbers and the decision do not use
      sectors and are untouched. The corner tables were not re-derived under a tighter rule.
    - **Candidate tighter rule (follow-up, not applied):**
      - pair only mutually nearest walks, with a cap of about 60°;
      - merge a `*_link` walk only when its far end rejoins a leg of the same unit (a true slip
        lane), and never a `motorway_link`;
      - take each leg's bearing from its non-link walks only.
- **A2. Internal paths through shape nodes (review S3).** A walk that reached another node of
  the same unit through degree-2 shape nodes ended as a stub leg. The Definitions say internal
  edges are skipped; only direct edges were. Such walks now give no leg. A test covers it.
- **A3. The census changed after it ran (review S4).** Both changes were made to the summary
  only; no response was re-fetched.
  - In 6e0f5a5, "captures" grew from the served current + historical panos to also include the
    run's own capture date of every pano. The reason: 36.5% of the panos are no longer served,
    and their capture is still a capture.
  - In 8058b2f, the extrapolation ratio moved from distinct historical ids per *served* pano to
    per *queried* pano, because a full pass would also meet the ~36% unserved ids. Both figures:
    - per queried pano: 5,248 / 1,308 × 18,461 ≈ **74,070** (the one quoted);
    - per served pano: 5,248 / 830 × 18,461 ≈ **116,727** (emitted by the script before 8058b2f).
  - After review, the captures quantiles leave out units and corners with no run pano within
    25 m (30 of 200 units, 104 of 656 corners), which are counted separately. `census.json`
    `wall_clock_s` is now the network time (701.2 s), not the last cache-only re-run.
- **A4. Order (review S5).** The census ran before the experiment-1 decision existed. It is step 1
  of experiment 2, it needs no GPU, and the issue's milestones list it (5) before the decision
  (6). The issue body also says experiment 2 comes after experiment 1 passes, so the issue
  contradicts itself here. Its result does not depend on the decision.
- **A5. Panos are not assigned to corners by bearing (review M7).** The Definitions list panos
  among the things assigned to a corner by bearing. The code, and the Observed definition, use
  distance to the corner point. Observed is what holds.
- **A6. `NA` with no `RAMPTYPE` (review S2).** Reading this as "the city records no ramp here"
  is an inference from the field values (`inventory_oracle.py` notes), not documented by the
  city; "Not Available" could also mean not surveyed. Everything below is worded as that
  inference.
- **A7. Smaller fixes.**
  - `build.json` records absolute input paths, so `score` and `census` work from any cwd.
    Before this change, re-running them only worked from the cwd that `build` ran in.
  - `frame` is rounded to 9 decimals.
  - The tool sha is the last commit touching `corner_inventory.py`, plus a dirty flag.
  - `score-assignments` refuses a `rubric_version` other than 1 and, with `--snapshot`, a
    `snapshot_sha256` other than the bundle's. It counts, rather than drops, ramps and uncovered
    points more than 30 m out at corner level.
  - `census --cache-only` refuses to fetch.

**Open question (review M2, not acted on).** "No longer served" is the same share in every
capture year (2024 captures 218/567, pre-2020 57/140, per the review), so 36.5% reads as id churn
rather than imagery age. Status code 2 is never retried. Re-querying ~50 not-served ids once
would rule out transient code-2 answers; this has not been done.

## RESULTS (Vancouver, WA; first computed 2026-10-04 after the protocol was posted; corner level re-derived after the amendments above)

Everything below comes from `runs/vancouver/corner_inventory/` (tracked: `report.md`,
`counts.csv`, `decision.json`, `false_absences.csv`, `gaps.csv`, `corners_224_*.csv`,
`build.json`, `census/census.json`), produced by `scripts/corner_inventory.py` at the commit
recorded in `build.json` (`tool_git_sha`, `tool_dirty: false`), with the exporter at
`export_cluster_review.py@5b21cd9`. Shares carry Wilson 95% intervals. Input sha256s are in
`build.json` and at the top of `report.md`. The two asserted ones are `results.jsonl`
`7fdf4005…9f28` and `osm.json` `58ce83ff…c2a030`.

**Caveats that travel with every number here:**
- one city, one rig (GSV);
- the inventory's completeness is unknown;
- the run is the PS pano store **selected on labels** (next section), not every GSV pano;
- an OSM corner is not always a legal crossing;
- neighbouring and mid-block windows overlap.

### Units

- 4,429 eligible intersection units (signalised 284, arterial 1,497, residential 2,648). The
  build reproduces the exporter's `eligible` counts exactly.
- 14,391 corners, 3,177 of them wider than 150°.
- 16,912 mid-block points.
- Legs per unit: 3 legs 2,778; 4 legs 1,368; 2 legs 273; 5–6 legs 7; 1 leg 3.
- Wall-clock: build 16.6 s, score 2.2 s (CPU, desktop).

### The run's pano set is selected on the outcome

This finding governs every absence number.
- `detect_from_store.py` built the run from the panos that carry a deployed CurbRamp label
  (28,881 ids), plus a seeded sample of 300 unlabeled store panos (`store_selection.json`).
- In the run, 26,930 of 28,830 panos have a detection >= 0.55.
- So "observed" here mostly means "a label was made in a pano near this corner". A corner with no
  ramp in view of any pano tends to have no pano in the run at all.

| unit state (fusion, primary) | panos within 25 m | units |
|---|---|---:|
| present | labeled panos only | 2,845 |
| present | a sampled-unlabeled pano within 25 m | 22 |
| present | no pano within 25 m | 25 |
| absent | labeled panos only | 55 |
| absent | sampled-unlabeled panos only | 29 |
| unobservable | no pano within 25 m | 1,453 |

- All 29 of these absent units have *only* sampled-unlabeled panos within 25 m, so they are
  reached only through the 300 random panos.
- 51 of those 300 panos lie within 25 m of an intersection unit centre.
- 11 of the 12 no-label #224 units are unobservable in this run (`corners_224_units.csv`). On this
  run's panos, the 12 no-label units cannot do the triage the decision rule names.

### City sentence (primary observability)

| stratum | arm | units | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 284 | 0.982 [0.959, 0.992] | 0.007 [0.002, 0.025] | 0.011 [0.004, 0.031] |
| arterial | fusion | 1,497 | 0.719 [0.695, 0.741] | 0.026 [0.019, 0.035] | 0.255 [0.234, 0.278] |
| residential | fusion | 2,648 | 0.580 [0.562, 0.599] | 0.016 [0.012, 0.022] | 0.403 [0.385, 0.422] |
| intersections | fusion | 4,429 | 0.653 [0.639, 0.667] | 0.019 [0.015, 0.023] | 0.328 [0.314, 0.342] |
| intersections | deployed | 4,429 | 0.654 [0.640, 0.668] | 0.018 [0.015, 0.023] | 0.328 [0.314, 0.342] |
| mid-block | fusion | 16,912 | 0.148 [0.142, 0.153] | 0.028 [0.026, 0.031] | 0.824 [0.818, 0.830] |

| stratum | arm | corners | present | absent | unobservable |
|---|---|---:|---|---|---|
| signalised | fusion | 1,004 | 0.946 [0.930, 0.959] | 0.043 [0.032, 0.057] | 0.011 [0.006, 0.020] |
| arterial | fusion | 4,743 | 0.603 [0.589, 0.617] | 0.154 [0.144, 0.164] | 0.244 [0.232, 0.256] |
| residential | fusion | 8,644 | 0.432 [0.422, 0.443] | 0.161 [0.153, 0.168] | 0.407 [0.397, 0.418] |
| intersections | fusion | 14,391 | 0.524 [0.516, 0.532] | 0.150 [0.144, 0.156] | 0.326 [0.318, 0.333] |
| intersections | deployed | 14,391 | 0.508 [0.500, 0.516] | 0.167 [0.161, 0.173] | 0.326 [0.318, 0.333] |
| intersections, sectors <= 150° | fusion | 11,214 | 0.599 [0.590, 0.608] | 0.099 [0.094, 0.105] | 0.302 [0.294, 0.311] |

Superseded corner rows (before amendment A1): signalised 0.783 / 0.208 / 0.009 over 1,254
corners, arterial 0.578 / 0.179 / 0.243, residential 0.432 / 0.161 / 0.407; pooled fusion
0.510 / 0.171 / 0.319 over 14,901, deployed 0.496 / 0.185 / 0.319; sectors <= 150° 0.579 /
0.128 / 0.294 over 11,833. Most of the signalised "absent" corners were median slivers.

Read these with the selection effect above. The unobservable share is mostly "no label was made
nearby", so it is not a statement about GSV coverage in Vancouver, and the absent share is a
floor. Per-stratum deployed-arm rows and the observability variants are in `report.md`. The two
arms agree to within 0.002 on every unit-level share of the pooled intersections.

### Recall against the `Available` inventory

| level | arm | with Available | present | absent | unobservable |
|---|---|---:|---|---|---|
| unit | fusion | 2,617 | 0.971 [0.964, 0.977] | 0.003 [0.002, 0.007] | 0.025 [0.020, 0.032] |
| unit | deployed | 2,617 | 0.971 [0.964, 0.977] | 0.003 [0.002, 0.006] | 0.026 [0.020, 0.032] |
| corner | fusion | 6,778 | 0.958 [0.953, 0.962] | 0.026 [0.022, 0.030] | 0.016 [0.013, 0.020] |
| corner | deployed | 6,778 | 0.943 [0.937, 0.948] | 0.040 [0.036, 0.045] | 0.017 [0.014, 0.020] |

(Superseded: corner row n = 6,810, same shares to 3 decimals within 0.001.) Signalised units:
274 of 274 with an `Available` ramp are called present (fusion). This recall is inflated by the
same selection: a unit with an inventory ramp is in the run largely because a label was made
there.

### Absence precision and the decision

| level | arm | absent | clean (no point) | false absence (Available) | RMV/NA only | no-Available read |
|---|---|---:|---|---|---|---|
| unit | fusion | 84 | **0.548 [0.441, 0.650]** | 0.107 [0.057, 0.191] | 0.345 [0.252, 0.452] | 0.893 [0.809, 0.943] |
| unit | deployed | 81 | 0.519 [0.411, 0.624] | 0.099 [0.051, 0.183] | 0.383 [0.284, 0.492] | 0.901 [0.817, 0.949] |
| corner | fusion | 2,160 | 0.579 [0.558, 0.599] | 0.081 [0.070, 0.093] | 0.340 [0.321, 0.361] | 0.919 [0.907, 0.930] |
| corner | deployed | 2,397 | 0.542 [0.522, 0.561] | 0.114 [0.102, 0.128] | 0.344 [0.325, 0.363] | 0.886 [0.872, 0.898] |

Superseded corner rows (before A1):
- fusion: 2,552 absent, clean 0.641, false 0.071, no-Available 0.929;
- deployed: 2,762 absent, clean 0.598, no-Available 0.898;
- signalised fusion: 261 absent, clean 0.908. That 0.908 was almost all median slivers; the
  re-derived signalised corner read is 43 absent, clean 0.512 [0.368, 0.654], false 0.279.

What the inventory holds at the 84 absent units:
- nothing at 46;
- `NA` with no `RAMPTYPE` at 32 (amendment A6: we read this as the city recording no ramp, an
  inference from the field values, not documented);
- `Available` at 9 (3 units hold both).

All 29 "RMV/NA only" units are `NA_noramp` units. At corner level, of 2,160 absent corners 1,250
hold nothing, 726 hold `NA_noramp` and 175 hold `Available`.

How the 84 absent units were observed, by bucket:

| observed through | units | clean | RMV/NA only | false (Available) |
|---|---:|---:|---:|---:|
| labeled panos only | 55 | 37 | 11 | 7 |
| sampled-unlabeled panos only | 29 | 9 | 18 | 2 |

**Decision: FAIL as written; on this run the absence read is not decidable.**
- **The rule as pre-registered.** Clean-read absence precision, unit level, fusion arm, primary
  observability, intersections pooled = 46/84 = 0.548 (Wilson [0.441, 0.650]). That is below
  0.90, so the rule fails and experiment 2 is not unblocked by it. That outcome stays on record.
- **Why it does not settle the question.**
  - The clean-read shortfall comes mostly from `NA_noramp` records, which the protocol comment
    flagged before scoring as plausibly supporting an absence. 18 of the 29 such units are the
    absent units reached only through the 300 random panos.
  - The second read (absent with no `Available` point) is 75/84 = 0.893. Its Wilson interval
    [0.809, 0.943] contains 0.90, and one more clean unit (76/84 = 0.905) would pass.
  - On the deployed arm the same read is 0.901 [0.817, 0.949], which passes.
  - n = 84 units, selected by how the run was built.
- So 0.548 is not a city-wide absence precision, and neither read separates the detector from
  0.90 here. Per stratum (fusion, unit level, clean read): signalised 2/2, arterial 29/39 =
  0.744, residential 15/43 = 0.349.

The 9 false absences (`false_absences.csv`, unit and corner rows, with pano ids and inventory ids):
- 8 of the 9 units have exactly one pano within 25 m. In 6 of those 8 the pano is 17–25 m away;
  the other two are at 9.8 m and 7.3 m.
- The ninth unit has two panos, the nearest at 21.3 m.
- They look mostly like observability-edge cases. Nobody has looked at them yet.

Observability sensitivity (fusion, intersections pooled, unit level): requiring >= 2 panos leaves
10 absent units (7 clean, 1 false); requiring a pano within 15 m leaves 27 (15 clean, 2 false).

### Inventory gaps

Present with no inventory point of any status (fusion, primary): 116 units (signalised 4,
arterial 39, residential 73) and 412 corners, listed in `gaps.csv` with site ids and the pano ids
of their operational members. Deployed arm: 121 units, 365 corners. (Superseded corner counts:
440 fusion, 424 deployed.) None has been looked at; a gallery is needed before calling any of
them a city omission.

### #224 hook

- `corners_224_units.csv` and `corners_224_corners.csv` (from `corners_224.jsonl`, untracked)
  hold our states for the 80 #224 units.
- All 80 match a unit of the full build: intersections by node ids, mid-block by position.
- `score-assignments` is tested end to end on a synthetic `assignments.json`. No real ratings
  exist yet.
- Command once `assignments.json` exists (run `build` first; writes `assignments_score/`):

```
python scripts/corner_inventory.py score-assignments \
    --assignments ../RampNet/benchmark/vancouver/cluster_review/assignments.json \
    --snapshot ../RampNet/benchmark/vancouver/cluster_review/snapshot.json \
    --out runs/vancouver/corner_inventory
```

### GSV history census (experiment 2, step 1; see amendments A3, A4)

200 units (signalised 67, arterial 67, residential 66; seed 238) and the 1,308 run panos within
25 m of them, one metadata request each, no images.
- **Wall-clock 701.2 s** for the 1,308 requests (0.25 s sleep between them, from the Windows
  desktop). No errors, no early stop.
- Response cache sha256 `9ef15aacbd46533f0398137f4247e84ee3d69b8bc3da6c75ede04c7c9e75471a`
  (`census/census.json`; the cache itself is untracked). The summary was recomputed from the
  cache with `--cache-only`; nothing was re-fetched.

Findings:
- **478 of 1,308 panos (36.5%) are no longer served** (status code 2); 830 are. This is the #56
  observation (store panos that no longer resolve on GSV) again; see the M2 open question above.
  Those panos contribute their own capture date and no history.
- Captures (distinct year-months over the run's capture dates and the served current and
  historical panos):
  - per unit, over the 170 units with >= 1 run pano within 25 m: median 11, IQR 5–14, max 26;
  - per corner, over the 552 corners with >= 1: median 11, IQR 5–14;
  - median per unit by stratum: signalised 15, arterial 11, residential 5.
- Earliest capture year per unit: 2007 at the median and at the 75th percentile. Span: median 18
  years, IQR 15–19.
- 30 of the 200 sampled units have no run pano within 25 m. 10 more have panos of which none is
  still served.
- 5,248 distinct historical panos in the sample are not already in the run: 4.01 per queried pano.
- **Extrapolation, stated as one:** a full pass over every eligible intersection unit would
  process about 74,000 historical panos (4.01 × the 18,461 run panos within 25 m of an eligible
  unit = 74,070).
  - The per-unit, per-stratum route gives 73,256. It is an upper bound, because neighbouring
    windows share panos.
  - The per-*served*-pano ratio gives 116,727 (amendment A3).
  - The census reaches history only through panos still served, and not at all at corners with no
    run pano.

(Superseded census quantiles, which included the no-pano units: captures per unit median 10, IQR
3–13; per corner median 10, IQR 4–14; residential median 4, arterial 10.)

### What would make the absence read decisive (not done here)

1. Run the detector on panos chosen by position rather than by label (the rest of the PS store,
   or current GSV coverage) at the 1,453 unobservable units, so that "observed" stops depending
   on a label having been made. Do the CPU metadata pass first, then GPU on the panos that turn
   out to exist.
2. Look at the 9 false-absence units and a sample of the 46 clean absences in a gallery.
3. Decide how `NA_noramp` is read: the rule as written counts it against an absence.

### Reproduce

From a checkout of this branch beside `sal-vancouver` (branch `vancouver-56-scoring`),
`sal-cluster-review` (for `osm.json`, Overpass payload of sampling rule v2) and `RampNet`
(branch `cluster-review-224`), with the test requirements plus nothing else (no pandas):

```
python scripts/corner_inventory.py build --run-dir ../sal-vancouver/runs/vancouver \
    --osm ../sal-cluster-review/runs/vancouver/cluster_review/osm.json \
    --units224 ../RampNet/benchmark/vancouver/cluster_review/corners.jsonl \
    --out runs/vancouver/corner_inventory          # asserts results.jsonl + osm.json sha256
python scripts/corner_inventory.py score --out runs/vancouver/corner_inventory
python scripts/corner_inventory.py census --out runs/vancouver/corner_inventory   # network, ~12 min
# ...or, from an existing response cache, with no network:
python scripts/corner_inventory.py census --cache-only --out runs/vancouver/corner_inventory
```

`osm.json` and `results.jsonl` are not public files: they live in the lab's working copies
(the OSM payload is re-fetchable with the exporter's query, recorded in the #224
`snapshot.json`, but a new fetch will not match the asserted sha256; the run's results are
#56's). The census cache is untracked; `census.json` carries its sha256, and a re-run fetches
GSV metadata as of that day, so the not-served share and the history will drift.

`build.json` stores absolute input paths, so `score` and `census` can be run from any cwd once
`build` has run. The committed run was built from `D:\Git\labeler-wt\corner238` with
`../../sal-vancouver/...`; the commands above assume a checkout beside `sal-vancouver`.

## Amendment A8: position-selected panos at the unobservable units (RampNet#241, 2026-10-05)

[RampNet#241](https://github.com/ProjectSidewalk/RampNet/issues/241) follows "What would make
the absence read decisive", item 1. The plan was posted on the issue before any number
([comment](https://github.com/ProjectSidewalk/RampNet/issues/241#issuecomment-5994700349)).
Nothing above is changed: the definitions, the decision rule and the #238 outputs in
`runs/vancouver/corner_inventory/` stand as written. This section adds panos and re-scores.

### What was done

- **Target units.** The 1,453 intersection units whose fusion/primary unit state is
  `unobservable` in the #238 build: signalised 3, arterial 382, residential 1,068.
- **Selection by position, PS store first** (`scripts/corner_posdetect.py select`, CPU).
  Pano positions come from one GET of the Vancouver server's `/adminapi/panos`, the list
  sidewalk-panorama-tools downloads from (142,623 panos, fetched 2026-10-05T12:49:41Z, sha256
  `77ea8a3e…6b45`, cached untracked). The store's JPEG ids come from
  `detect_from_store.store_ids` on makelab2 (139,278 ids, sha256 `dbc9d976…390a`, cached
  untracked). A pano is selected when it is within 25 m (the build's `obs_m`) of a target
  unit's centre, is not in the #56 run, and has a JPEG in the store. **Every** such pano is
  selected, not only the nearest, because the observability rule counts any pano within
  25 m; a nearest-pano-only read is reported as a sensitivity check.
- **Current GSV coverage for the rest.** Units left with no usable store pano get a 25 m
  circle each (`select --gsv-fallback` -> `gsv_area.geojson`), and `main.py` scans and
  processes current GSV coverage inside the circles (its own z17 tile scan, then its GSV fetch
  and detection path). This is a different pixel source (GSV tiles, not the store's
  native-resolution JPEGs).
- **Detection** (makelab2 A40). Store panos: `detect_from_store.py --ids` into a new run dir,
  `runs/vancouver_posdetect241`. The #56 run is read only and never appended to. GSV panos:
  `main.py` into `runs/vancouver_posdetect241_gsv`. Both used the code at this branch, whose
  `detectors/`, `main.py`, `panorama.py`, `sources/`, `depth.py`, `geo.py`,
  `scripts/detect_from_store.py` and `scripts/fuse_sites.py` are byte-identical to 660ccd5,
  the commit that produced the #56 run (`git diff --stat 660ccd5 HEAD -- <those paths>` is
  empty). Same model revision (`rampnet-model@606a11956743`), storage floor 0.1, batch size
  1, store and metadata server as #56.
- **Fusion.** `fuse_sites.py` on each new run alone, with the #56 `sites_meta.json`
  parameters (floor 0.1, operational 0.30, max range 25 m, rig mask, flat pose). Camera height
  is **fixed at 2.5 m**, the value `auto` gave every capture year in the #56 run; it is fixed
  because the new runs have no harvested depth table. The GSV panos' own measured heights are
  2.0–2.3 m, so their sites sit up to ~20% too far out along each ray.
- **Merge.** `corner_inventory.py build --extra-run <dir>` (repeatable) adds each extra run's
  panos to the observed set and its operational sites to the fusion arm. Site ids are
  prefixed with the run dir's name. A pano shared with the base run, or a fusion parameter
  that differs (other than camera height), is refused. **The deployed arm is emulated for the
  added panos**: they were never submitted, so no server cluster exists for them, and an added
  site with a member detection >= 0.55 (the tier that went live) stands in for one. Without
  `--extra-run`, every scored output (`counts.csv`, `decision.json`, `false_absences.csv`,
  `gaps.csv`, `corners_224_*.csv`) is byte-identical to the committed #238 outputs. That was
  checked by rebuilding them on this branch.
- **Reads** (`corner_posdetect.py compare`). The decision rule is applied exactly as
  pre-registered: the clean read, unit level, fusion arm, primary observability,
  intersections pooled, over the merged build. Beside it are the three arms the issue names:
  (a) the rule as written, (b) the same clean read without counting the city's `NA` points
  with no `RAMPTYPE` (amendment A6: our inference that they mean "no ramp"), and (c) the
  deployed arm. Each is read over three subsets: all units, the units this pass moved (the
  1,453 targets), and the 84 units that were absent before. The no-`Available` read is in
  every row. The capture years of the added panos are reported beside the #56 run's.

### Results (computed 2026-10-05, after the plan comment)

Outputs:
- `runs/vancouver/corner_posdetect241/select/`: the selection;
- `runs/vancouver/corner_posdetect241/compare/`: `report.md`, `compare.json`, `transitions.csv`,
  `reads.csv`;
- `runs/vancouver/corner_inventory_posdetect241/`: the merged build's `counts.csv`,
  `decision.json`, `false_absences.csv`, `gaps.csv`, `report.md` and `build.json`.

A full regeneration (fuse, build, score, compare) reproduced every one of these byte for byte,
except `build.json` and `report.md`, which carry the build time as the #238 ones do.

**Coverage.**
- **Store panos.** The store held at least one usable pano within 25 m for 1,443 of the 1,453
  target units, 10,337 panos in all. The metadata pass fetched all 10,337 with 0 skips (38.7
  min, CPU). The detection pass wrote all 10,337 with 0 failures.
- **GSV fallback.** The other 10 units had either no store pano within 25 m (4) or only store
  panos with no JPEG (6). Current GSV coverage inside their 25 m circles held 30 panos (z17
  scan, `scan.json` sha256 `e00b074b…dbcd`), and all 30 were processed.
- **Merged.** 1,450 of the 1,453 target units have an added pano within 25 m of the centre (by
  the record's position; median 7 panos per unit within 25 m).
- **3 units stay unobservable.** `res:n3784186964` and `res:n3788306130` have no pano at all,
  and `res:n3784186894`'s nearest pano is 25.25 m away. Neither source has imagery within
  25 m of these 3.
- **Fusion.** The store run gave 1,475 sites, 269 of them operational. The GSV run gave 13
  sites, 8 operational. Emulated deployed points (a member >= 0.55): 24 from the store run and
  3 from the GSV run.
- **Run sha256s.** Store `results.jsonl` `69f46b26…bf0e` (verified on both ends of the copy);
  GSV `results.jsonl` `953716d1…57b2`. Every input's sha256 is in the merged `build.json`.

**Old -> new unit states, fusion arm** (`transitions.csv`, which also has the deployed arm):

| stratum | target units | -> absent | -> present | -> still unobservable |
|---|---:|---:|---:|---:|
| signalised | 3 | 2 | 1 | 0 |
| arterial | 382 | 332 | 50 | 0 |
| residential | 1,068 | 905 | 160 | 3 |
| **all** | **1,453** | **1,239** | **211** | **3** |

- **Deployed arm.** Over its own 1,452 unobservable units, the arm (emulated for added panos)
  sends 1,422 to absent and 21 to present; 9 stay unobservable.
- **Other units.** Outside the targets, one mid-block point changed (absent -> present).
  No other intersection unit changed state.
- **City sentence after the merge** (fusion, unit level, intersections pooled):
  - present 3,103 / 4,429 = 0.701;
  - absent 1,323 = 0.299;
  - unobservable 3.

  Deployed arm: 2,917 present, 1,503 absent, 9 unobservable. Per-stratum rows are in the
  merged `report.md`.

**The decision rule.**

| read (unit level, intersections pooled) | absent | clean | share [Wilson 95%] |
|---|---:|---:|---|
| (a) as written: no inventory point of any status, fusion arm | 1,323 | 333 | **0.252 [0.229, 0.276]** |
| (b) same, not counting `NA` points with no `RAMPTYPE` | 1,323 | 1,277 | 0.965 [0.954, 0.974] |
| (c) as written, deployed arm (emulated for added panos) | 1,503 | 365 | 0.243 [0.222, 0.265] |
| no-`Available` read, fusion arm | 1,323 | 1,279 | 0.967 [0.956, 0.975] |
| no-`Available` read, deployed arm | 1,503 | 1,437 | 0.956 [0.945, 0.965] |

- **On the 1,239 units this pass moved to absent** (fusion): (a) 0.232 [0.209, 0.256], (b)
  0.970 [0.959, 0.978], no-`Available` 0.972 [0.961, 0.980].
- **Nearest added pano only, per unit:** 1,401 absent and 49 present. (a) is 0.226 [0.205,
  0.249] and no-`Available` 0.965 [0.954, 0.973].
- **Per stratum, among the moved units:** (a) is 0.485 for arterial and 0.138 for residential;
  (b) is 0.958 and 0.975.

What the inventory holds at the 1,239 units moved to absent:
- nothing at 287;
- **only `NA` points with no `RAMPTYPE` at 915**;
- an `Available` point at 35;
- other classes at 2.

**Decision: the rule as written FAILS, and n is now large enough that this is not a sampling
accident. Under reading (b) it would pass.**
- **The rule as pre-registered.** Read (a) is 333/1,323 = 0.252, Wilson [0.229, 0.276]. #238
  had n = 84 selected by label; this n is not small. The rule fails, so it does not unblock
  experiment 2.
- **What drives the failure.** Almost all of it is the city's `NA` points with no
  `RAMPTYPE`. Of the 990 absent units that are not clean, 944 hold nothing else.
  - Without them, read (b) is 0.965, and its lower bound (0.954) is above 0.90.
  - The no-`Available` read is 0.967 [0.956, 0.975] on the fusion arm and 0.956 [0.945,
    0.965] on the deployed arm.
- **The detector and the city mostly agree at those points.** Among target units whose only
  inventory is `NA` with no `RAMPTYPE`, 915 are called absent and 135 present (fusion arm, 0.30
  operating point). This supports the amendment A6 inference that such a point records a
  corner with no ramp. It does not establish it: the city has not documented the code.
- **The open question.** It is item 3 of "What would make the absence read decisive": how to
  read `NA` with no `RAMPTYPE`. That is Jon's call and is not made here.

**False absences and recall.**
- **All intersection units with an `Available` inventory point:** 2,573 of 2,617 (0.983) are
  now present, 44 absent and 0 unobservable. In #238 the split was 2,542 present, 9 absent and
  66 unobservable.
- **The 66 target units with an `Available` point:** only 31 are now present; 35 are absent.
  At the units the deployment did not label, recall against the inventory is about one half.
- **For triage.** The 35 are rows of the merged `false_absences.csv` (44 units in all, plus
  corner rows, with pano ids), ready for the triage the rule names. Nobody has looked at them
  yet.

**Capture dates of the added panos** (`compare.json`):
- Median 2023-04, IQR 2022-11 to 2023-05, range 2007-08 to 2026-07.
- By year: 2023 3,414; 2022 2,624; 2024 2,182; 2014 665; 2019 500; 2021 239; others smaller.
- 2007 (17 panos) and 2026 (38) are not in the #56 run at all. The 2007 panos come from the GSV
  fallback.
- The #56 run is dominated by 2024 (10,621) and 2023 (8,653), so the added panos are about a
  year older.

**Cost** (makelab2 A40, free; rows in RampNet `analysis_out/usage_log.jsonl`, `paid: false`):
- store detection: 8,113 s including model load (the pass itself 7,971.6 s, 1.297 panos/s,
  batch 1) = 2.25 GPU-hours;
- GSV detection: 51.9 s for 30 panos (0.014 GPU-h);
- store metadata: 2,319 s, CPU only;
- on the desktop CPU: fuse about 1 min, build about 19 s, score and compare a few seconds.

`docs/compute_cost.md` routes makelab2 time to `usage_log.jsonl`, not `compute_log.jsonl`,
because the latter is asserted equal to the `sacct` dumps. That is why the rows went there.

**Caveats that travel with these numbers:**
- one city, one rig (GSV);
- the added panos are PS-store panos (the panos the server holds) chosen by position, plus 30
  current GSV panos; they are not a complete coverage scan;
- no added store pano carries a deployed AI CurbRamp label. Whether the 2025 deployment
  processed a given one and found nothing >= 0.55, or never processed it, is not recorded
  here. Either way, 0.55 is the operating point that made no label there. That is why the
  emulated deployed arm sends only 21 targets to present, against the fusion arm's 211;
- fusion of the added panos used a fixed 2.5 m camera height, while the GSV fallback panos
  measure 2.0–2.3 m;
- the deployed arm for added panos is an emulation;
- the inventory's completeness is unknown, and so is the meaning of `NA` with no `RAMPTYPE`;
- the decision uses unit level only. Corner-level numbers in the merged `report.md` carry
  the amendment A1 over-merge limitation and are not read here.

### Reproduce (A8)

Run from this branch, with `sal-vancouver`, `sal-cluster-review` and `RampNet` checked out
beside it, as in the #238 Reproduce section. The #238 build (`runs/vancouver/corner_inventory`)
must exist first.

```
O=runs/vancouver/corner_posdetect241
# 1. selection (CPU). fetch-panos re-fetches the live list, which will not match the recorded
#    sha256 once the server changes; select checks the cached copy against ps_panos_fetch.json.
python scripts/corner_posdetect.py fetch-panos --server https://sidewalk-vancouver.cs.washington.edu --out $O
#    $O/cache/store_jpg_ids.txt: the store's JPEG ids, one per line, from
#    detect_from_store.store_ids(<store>, 'sharded') on makelab2, where the store is mounted.
python scripts/corner_posdetect.py select --build runs/vancouver/corner_inventory \
    --store-ids $O/cache/store_jpg_ids.txt --out $O --gsv-fallback
# 2. detection (makelab2 A40)
STORE=/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa
SERVER=https://sidewalk-vancouver.cs.washington.edu
python scripts/detect_from_store.py --run-dir runs/vancouver_posdetect241 --store $STORE \
    --server $SERVER --ids $O/select/store_ids.txt --metadata-only
python scripts/detect_from_store.py --run-dir runs/vancouver_posdetect241 --store $STORE \
    --server $SERVER --ids $O/select/store_ids.txt
python main.py $O/select/gsv_area.geojson --name vancouver_posdetect241_gsv --scan-only
python main.py $O/select/gsv_area.geojson --name vancouver_posdetect241_gsv --reuse-scan \
    --no-gap-fill --no-position-check
# 3. fusion, merge, score, compare (CPU)
python scripts/fuse_sites.py runs/vancouver_posdetect241 --camera-height-m 2.5
python scripts/fuse_sites.py runs/vancouver_posdetect241_gsv --camera-height-m 2.5
python scripts/corner_inventory.py build --run-dir ../sal-vancouver/runs/vancouver \
    --osm ../sal-cluster-review/runs/vancouver/cluster_review/osm.json \
    --units224 ../RampNet/benchmark/vancouver/cluster_review/corners.jsonl \
    --out runs/vancouver/corner_inventory_posdetect241 \
    --extra-run runs/vancouver_posdetect241 --extra-run runs/vancouver_posdetect241_gsv
python scripts/corner_inventory.py score --out runs/vancouver/corner_inventory_posdetect241
python scripts/corner_posdetect.py compare --old runs/vancouver/corner_inventory \
    --new runs/vancouver/corner_inventory_posdetect241 --out $O
```

The two detection runs' `results.jsonl` are not public files, as with #56's. A re-run of step
2 can differ, because the store grows and the GSV scan returns whatever coverage exists that
day. Step 3 is deterministic from the two `results.jsonl` files, whose sha256s are above.
The store pass's metadata cache is kept on makelab2 as `~/posdetect241_store_metadata.tgz`.

## Follow-up: rating gallery for the `NA` read and the 35 false absences (RampNet#243)

[RampNet#243](https://github.com/ProjectSidewalk/RampNet/issues/243) rates 95 absent units per
corner (present / absent / can't tell): the 35 false absences at the #241 targets, 40 of the
944 `NA`-no-`RAMPTYPE` units and 20 of the 333 clean units (seed 243). Protocol, how to rate
and the scoring rules: [`corner-gallery-243.md`](corner-gallery-243.md). Bundle:
`runs/vancouver/corner_gallery243/`. Nothing in this document changes until the ratings are
scored; the `NA` read stays open until then.
