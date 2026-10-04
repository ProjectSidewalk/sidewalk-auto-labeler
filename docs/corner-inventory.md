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
