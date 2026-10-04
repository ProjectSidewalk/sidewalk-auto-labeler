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

## RESULTS (Vancouver, WA; computed 2026-10-04 after the protocol above was committed and posted)

Everything below comes from `runs/vancouver/corner_inventory/` (tracked: `report.md`,
`counts.csv`, `decision.json`, `false_absences.csv`, `gaps.csv`, `corners_224_*.csv`,
`build.json`, `census/census.json`), produced by `scripts/corner_inventory.py@8058b2f`
with the exporter at `export_cluster_review.py@5b21cd9`. Shares carry Wilson 95% intervals.
Input sha256s are in `build.json` and at the top of `report.md`; the two asserted ones are
`results.jsonl` `7fdf4005…9f28` and `osm.json` `58ce83ff…c2a030`.

**Caveats that travel with every number here:** one city, one rig (GSV); the inventory's
completeness is unknown; the run is the PS pano store **selected on labels** (next section), not
every GSV pano; an OSM corner is not always a legal crossing; neighbouring and mid-block windows
overlap.

### Units

4,429 eligible intersection units (signalised 284, arterial 1,497, residential 2,648; the build
reproduces the exporter's `eligible` counts exactly), 14,901 corners (3,068 wider than 150°),
16,912 mid-block points. Legs per unit: 3 legs 2,770; 4 legs 1,286; 2 legs 180; 5–8 legs 190;
1 leg 3. Wall-clock: build 17.5 s, score about 2 s (CPU, desktop).

### The run's pano set is selected on the outcome

This finding governs every absence number. `detect_from_store.py` built the run from the panos
that carry a deployed CurbRamp label (28,881 ids) plus a seeded sample of 300 unlabeled store
panos (`store_selection.json`). In the run, 26,930 of 28,830 panos have a detection >= 0.55. So
"observed" here mostly means "a label was made in a pano near this corner", and a corner with no
ramp in view of any pano tends to have no pano in the run at all.

| unit state (fusion, primary) | panos within 25 m | units |
|---|---|---:|
| present | labeled panos only | 2,845 |
| present | a sampled-unlabeled pano within 25 m | 22 |
| present | no pano within 25 m | 25 |
| absent | labeled panos only | 55 |
| absent | a sampled-unlabeled pano within 25 m | 29 |
| unobservable | no pano within 25 m | 1,453 |

29 of the 84 absent units are seen through the 300-pano random sample (51 of those 300 panos lie
within 25 m of an intersection unit centre). 11 of the 12 no-label #224 units are unobservable in
this run (`corners_224_units.csv`), so the 12 no-label units cannot do the triage the decision
rule names, not on this run's panos.

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
| signalised | fusion | 1,254 | 0.783 [0.759, 0.805] | 0.208 [0.187, 0.231] | 0.009 [0.005, 0.016] |
| arterial | fusion | 4,976 | 0.578 [0.564, 0.591] | 0.179 [0.169, 0.190] | 0.243 [0.231, 0.255] |
| residential | fusion | 8,671 | 0.432 [0.422, 0.442] | 0.161 [0.154, 0.169] | 0.407 [0.397, 0.417] |
| intersections | fusion | 14,901 | 0.510 [0.502, 0.518] | 0.171 [0.165, 0.177] | 0.319 [0.311, 0.326] |
| intersections | deployed | 14,901 | 0.496 [0.488, 0.504] | 0.185 [0.179, 0.192] | 0.319 [0.311, 0.326] |
| intersections, sectors <= 150° | fusion | 11,833 | 0.579 [0.570, 0.588] | 0.128 [0.122, 0.134] | 0.294 [0.286, 0.302] |

Read these with the selection effect above: the unobservable share is mostly "no label was made
nearby", so it is not a statement about GSV coverage in Vancouver, and the absent share is a
floor. Per-stratum deployed-arm rows and the observability variants are in `report.md`; the two
arms agree to within 0.002 on every unit-level share of the pooled intersections.

### Recall against the `Available` inventory

| level | arm | with Available | present | absent | unobservable |
|---|---|---:|---|---|---|
| unit | fusion | 2,617 | 0.971 [0.964, 0.977] | 0.003 [0.002, 0.007] | 0.025 [0.020, 0.032] |
| unit | deployed | 2,617 | 0.971 [0.964, 0.977] | 0.003 [0.002, 0.006] | 0.026 [0.020, 0.032] |
| corner | fusion | 6,810 | 0.958 [0.953, 0.962] | 0.026 [0.023, 0.031] | 0.016 [0.013, 0.019] |
| corner | deployed | 6,810 | 0.943 [0.937, 0.948] | 0.041 [0.037, 0.046] | 0.016 [0.013, 0.019] |

Signalised units: 274 of 274 with an `Available` ramp are called present (fusion). This recall is
inflated by the same selection: a unit with an inventory ramp is in the run largely because a
label was made there.

### Absence precision and the decision

| level | arm | absent | clean (no point) | false absence (Available) | RMV/NA only | no-Available read |
|---|---|---:|---|---|---|---|
| unit | fusion | 84 | **0.548 [0.441, 0.650]** | 0.107 [0.057, 0.191] | 0.345 [0.252, 0.452] | 0.893 [0.809, 0.943] |
| unit | deployed | 81 | 0.519 [0.411, 0.624] | 0.099 [0.051, 0.183] | 0.383 [0.284, 0.492] | 0.901 [0.817, 0.949] |
| corner | fusion | 2,552 | 0.641 [0.622, 0.659] | 0.071 [0.061, 0.081] | 0.289 [0.272, 0.307] | 0.929 [0.919, 0.939] |
| corner | deployed | 2,762 | 0.598 [0.580, 0.616] | 0.102 [0.091, 0.114] | 0.300 [0.283, 0.318] | 0.898 [0.886, 0.909] |

What the inventory holds at the 84 absent units: nothing at 46; `NA` with no `RAMPTYPE` (the
city's own record of a corner without a ramp) at 32; `Available` at 9 (3 units hold both). All 29
"RMV/NA only" units are `NA_noramp` units. At corner level, of 2,552 absent corners 1,635 hold
nothing, 728 hold `NA_noramp`, 180 hold `Available`.

**Decision: FAIL as written.** Clean-read absence precision, unit level, fusion arm, primary
observability, intersections pooled = 46/84 = 0.548 (Wilson [0.441, 0.650]), below 0.90.
Experiment 2 is not unblocked by this rule. The second read (absent with no `Available` point) is
75/84 = 0.893 [0.809, 0.943], also below 0.90, so the outcome does not hinge on how `NA` is read.
Per stratum (fusion, unit level, clean read): signalised 2/2, arterial 29/39 = 0.744,
residential 15/43 = 0.349.

What this FAIL does and does not say about the detector:

- n = 84 absent units, a third of them reached only through the 300 random unlabeled panos. The
  absent set is shaped by how the run was selected, so 0.548 is not a city-wide absence precision.
- The clean read is low mostly because the city has an `NA_noramp` record at 29 of the absent
  units, i.e. the city agrees there is no ramp. Against the city's records where it has one, the
  absences agree 29 times and disagree 9 times.
- The 9 false absences (`false_absences.csv`, unit and corner rows, with pano ids and inventory
  ids): 8 of the 9 units have exactly one pano within 25 m, and in 7 the nearest pano is 17–25 m
  away. They look like observability-edge cases; nobody has looked at them yet.

Observability sensitivity (fusion, intersections pooled, unit level): requiring >= 2 panos leaves
10 absent units (7 clean, 1 false); requiring a pano within 15 m leaves 27 (15 clean, 2 false).

### Inventory gaps

Present with no inventory point of any status (fusion, primary): 116 units (signalised 4,
arterial 39, residential 73) and 440 corners, listed in `gaps.csv` with site ids and the pano ids
of their operational members. Deployed arm: 121 units, 424 corners. None has been looked at; a
gallery is needed before calling any of them a city omission.

### #224 hook

`corners_224_units.csv` and `corners_224_corners.csv` (from `corners_224.jsonl`, untracked) hold
our states for the 80 #224 units; all 80 match a unit of the full build (intersections by node
ids, mid-block by position). `score-assignments` is tested end to end on a synthetic
`assignments.json`; no real ratings exist yet. Command once `assignments.json` exists:
`python scripts/corner_inventory.py score-assignments --assignments
../RampNet/benchmark/vancouver/cluster_review/assignments.json --out
runs/vancouver/corner_inventory` (needs `build` run first; writes `assignments_score/`).

### GSV history census (experiment 2, step 1)

200 units (signalised 67, arterial 67, residential 66; seed 238), 1,308 run panos within 25 m
of them, one metadata request each, no images. **Wall-clock 701.2 s** for the 1,308 requests
(0.25 s sleep between them, desktop on a home connection), no errors, no early stop. Response
cache sha256 `9ef15aacbd46533f0398137f4247e84ee3d69b8bc3da6c75ede04c7c9e75471a`
(`census/census.json`; the cache itself is untracked).

- **478 of 1,308 panos (36.5%) are no longer served** (status code 2); 830 are. This is the #56
  observation (store panos that no longer resolve on GSV) again. Those panos contribute their own
  capture date and no history.
- Captures per unit (distinct year-months over the run's capture dates and the served current and
  historical panos): median 10, IQR 3–13, max 26. Per corner: median 10, IQR 4–14. Median per unit
  by stratum: signalised 15, arterial 10, residential 4.
- Earliest capture year per unit: 2007 at the median and at the 75th percentile. Span: median 18
  years, IQR 15–19.
- 30 of the 200 sampled units have no run pano within 25 m; 10 more have panos of which none is
  still served.
- 5,248 distinct historical panos in the sample that are not already in the run: 4.01 per queried
  pano. **Extrapolation, stated as one:** a full pass over every eligible intersection unit would
  process about 74,000 historical panos (4.01 × the 18,461 run panos within 25 m of an eligible
  unit = 74,070; the per-unit, per-stratum route gives 73,256 and is an upper bound because
  neighbouring windows share panos). The census reaches history only through panos still served,
  and not at all at corners with no run pano.

### What would make the absence read decisive (not done here)

1. Run the detector on panos chosen by position rather than by label (the rest of the PS store,
   or current GSV coverage) at the 1,453 unobservable units, so that "observed" stops depending on
   a label having been made. CPU metadata first, then GPU on the panos that turn out to exist.
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
```

`osm.json` and `results.jsonl` are not public files: they live in the lab's working copies
(the OSM payload is re-fetchable with the exporter's query, recorded in the #224
`snapshot.json`, but a new fetch will not match the asserted sha256; the run's results are
#56's). The census cache is untracked; `census.json` carries its sha256, and a re-run fetches
GSV metadata as of that day, so the not-served share and the history will drift.
