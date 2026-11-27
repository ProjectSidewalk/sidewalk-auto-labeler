# Best-view emission: one label per fused site, at server placement

Agent `best_view_emission`, overnight batch 2026-11-26. Script `scripts/best_view_emission.py`,
tests `tests/test_best_view_emission.py`, CSVs `docs/overnight/data/best_view_emission/`.

Question: today every per-pano detection >= 0.30 ships as its own label, and the server
places each one once from its pixel, with an estimator that caps range near 23.85 m and
ignores pose (https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/192). Its
clustering then fragments ramps about 2x more than fusion does
(https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/106). Would emitting ONE
label per fused site, at the stored pixel of the member that sees the ramp steepest, place
labels closer to the city inventories at server placement, without losing coverage? It
needs no server change.

## 1. Pre-registration (written 2026-11-26, before any score on real data)

Nothing below was changed after scores existed. Deviations are listed as dated amendments in
section 6.

**Inputs.** `runs/<city>/results.jsonl` and `depth/index.csv` (Bend, Gainesville; read in
place from the main checkout). Inventory points: the `inventory_oracle` snapshot
(`inventory.geojson`, verified against the sha256 in `inventory.json`), filtered by
`inventory_oracle.kept`. Any substitute is labelled as one and does not count toward the
rule.

**Labels (server placement).** Synthesized exactly as `eval_ps_clustering.synthesize_labels`
does: one label per stored CurbRamp detection with confidence >= tier, rig-masked
(`detectors.on_camera_rig`), at `detectors.ps_pixel`, placed by `ps_placement.label_latlng`
(the server's estimator, 2.341 m, range capped near 23.85 m, no pose). CurbRamp is not
range-gated. Every arm scores labels at these server positions. No arm moves a label; arms
only choose which labels to emit.

**Association.** `fuse_sites.fuse` over the run, at the tier, rig-masked, flat pose
(`apply_pose=off`), as `inventory_clustering.py` does. The `auto` frame is primary and the
2.6 m frame is secondary. A site's candidates are its operational members (confidence >=
tier, raycast inside 25 m at the fuse height). Every candidate is also an arm-1 label.

**Arms.**
- **A1 `all`**: every label (today's behaviour).
- **A2 `best`**: one label per operational site, from the operational member with the
  largest dip (largest `y_normalized`, all of which already pass the rig mask). Ties go to
  the higher confidence, then to `(pano_id, det_index)`. Labels in no site (fusion dropped
  the ray: beyond 25 m or above the horizon at the fuse height) are NOT emitted.
- **A2' `random`** (control): one label per operational site, from a member drawn uniformly
  at random. `random.Random(192)`, sites visited in `site.id` order (fusion is
  deterministic). Unplaced labels are not emitted. Seeds 0-19 are reported as a
  sensitivity check. The rule reads seed 192.
- **A3 `gate20`**: every label, except a label whose server range is > 20.0 m when its site
  has at least one operational member whose server range is <= 20.0 m. Labels in no site
  are kept.
- Sensitivity arms, not in the rule: `best+u` and `random+u` (A2 and A2' plus the unplaced
  labels passed through unchanged).

**Visible pool.** Inventory ramps within 20 m of a processed pano position (as in
`inventory_clustering.visible_pool`). The same pool is used for every arm.

**Metrics** (computed per city, tier and frame).
- **M1, anchored one-to-one distance (PRIMARY for the distance clause).** Pairs come from a
  greedy ascending-distance one-to-one match between pool ramps and operational sites' fused
  positions (the fuse frame), within 5 m. The pairs are therefore fixed across arms. For each
  pair, the distance from the ramp to the label the arm emits for that site (A2, A2':
  exactly one, uncapped). The statistics are p50, p90 and mean. The paired difference
  A2' - A2 gets a bootstrap CI over pairs (one pair per site, so this is a site bootstrap):
  2000 reps, seed 192. For A1 and A3 the same pairs are scored label-weighted (every label
  the arm emits for that site, to the paired ramp). This is descriptive only.
- **M2, own one-to-one at 5 m (secondary).** A greedy one-to-one match between pool ramps
  and the arm's own labels within 5 m. Reported as p50 and p90 of the matched distances,
  plus `coverage_1to1` = matched / pool. This favours arms with more labels (more chances),
  and its distances are truncated at 5 m.
- **M3, coverage at 5 m (PRIMARY for the coverage clause).** The `inventory_clustering`
  definition: each label is assigned to its nearest kept inventory ramp within r. A pool
  ramp is covered if at least one label is assigned to it. r = 5 m is primary; 3 and 8 m
  are also reported.
- **Workload and fragmentation.** Labels emitted. Labels per covered pool ramp (mean), and
  the share of covered ramps with >= 2 labels (r = 5 m). Labels with no ramp within 5 m.
- **Survivorship.** Labels in A1, in operational sites and unplaced. Operational sites. M1
  pairs. Pool size.
- **Secondary, approximate.** The server's clustering re-run on each arm's labels (complete
  linkage at 7.5 m, citywide, same-pano cannot-link, ENU distances), each cluster at the
  mean of its labels' server positions, scored with `inventory_clustering.inventory_metrics`
  (covered, split and merge at 5 m).

**Decision rule** (tier 0.30, `auto` frame). PASS iff in BOTH Bend and Gainesville:
`p90_M1(A2') - p90_M1(A2) >= 0.5 m` AND `covered_r5(A2) >= covered_r5(A1) - 0.010`. Otherwise
FAIL. If either city cannot be scored on the verified snapshot, the verdict is UNDETERMINED.
The 2.6 m frame, tier 0.55, seeds 0-19 and the bootstrap CIs are reported but do not gate.

What this cannot see: whether the emitted label is a true ramp (the inventory has no false
positives to score), what happens to human validation workload beyond the label count, and
regional clustering (Gainesville's server clusters per region; the secondary clustering here
is citywide).

## 2. Verdict: FAIL, on fresh inventory pulls that Jon ruled may stand in (the script still prints UNDETERMINED)

**Update, 2026-11-27 (Amendment 3).** Both inventories were re-pulled from the cities' ArcGIS
layers (`inventory_oracle.py fetch bend gainesville --out <scratch>`; nothing tracked was
overwritten). Neither matches its recorded fingerprint, since both layers were edited after
2026-09-26: Bend keeps 12,591 ramps against the record's 12,504, and Gainesville keeps 3,225
against 3,208. Jon ruled that a fresh snapshot may stand in for the frozen one. The script
still labels both SUBSTITUTE and prints UNDETERMINED, because its check is unchanged. Under
the ruling, the rule reads **FAIL**: the distance clause passes in both cities, and the
coverage clause fails in both.

| city (tier 0.30, `auto`) | p90 random - best (need >= +0.50 m) | covered at 5 m, best vs all (need >= -1.0 pt) | labels sent, all -> best | lost ramps whose site's label went to another ramp |
|---|---|---|---|---|
| Bend | 3.61 - 1.91 = **+1.70 m** [+1.59, +1.79] | 0.890 vs 0.910 = **-2.0 pts** | 51,529 -> 14,058 (-73%) | 239 of 248 |
| Gainesville | 4.30 - 3.37 = **+0.93 m** [+0.73, +1.09] | 0.853 vs 0.890 = **-3.7 pts** | 21,874 -> 6,857 (-69%) | 85 of 111 |

The 0.55 tier and the 2.6 m frame read the same way (`data/best_view_emission/*_arms.csv`,
`*_diagnostics.csv`). At 2.6 m Gainesville loses fewer ramps (65 against 111). The
mechanism is the one Bend showed overnight. Fusion merges the two ramps of a dual corner
into one site, so a single label can cover only one of them. Next step: split dual corners
before emitting one label per site. The CSVs under `data/best_view_emission/` now hold the
2026-11-27 run. The section 3 tables below are from the overnight run on RampNet's July Bend
substitute, which differs from the new run by about 0.02 m and 1 point of coverage.

### Overnight reading (2026-11-26, before the fetch)

The rule needs both cities scored on the frozen `inventory_oracle` snapshots. Neither is on
this machine. `inventory.geojson` is untracked, and only `inventory.json` (metadata and
sha256) is present.
- **Gainesville: not scored.** No local copy of its inventory exists. I searched Spotlight,
  `~/Git/*/runs/gainesville/inventory_oracle/`, `RampNet/data/inventories/` (which holds
  Bend but not Gainesville), `RampNet/analysis_out/` and label-latlng-estimation.
- **Bend: scored on a SUBSTITUTE.** The substitute is RampNet's
  `data/inventories/bend-or-2026-07-31.jsonl.gz`, the same ArcGIS layer pulled 8 weeks
  earlier (sha256 `bbc78f4c...`, not the oracle's `30acadb2...`). After the oracle's own
  area and `LifeCycleStatus == 'I'` filter it keeps **12,420** ramps, against the record's
  12,504 (-0.7%). Its in-area count is 14,311 against 14,328. As a sanity check, the
  approximate server clustering of arm `all` reproduces the tracked
  `runs/bend/inventory_clustering` server-centroid row: 14,655 clusters vs 14,650, and split
  0.0566 vs 0.057.

On Bend (substitute, tier 0.30, `auto`):
- The **distance clause holds by a wide margin**: p90 random 3.60 m minus best 1.89 m is
  **+1.70 m** (95% paired bootstrap over sites [+1.61, +1.80]), against the +0.50 m the rule
  needs.
- The **coverage clause fails**: covered at 5 m is best **0.900** vs all **0.920**, i.e.
  **-2.0 points** against a 1.0-point tolerance.

An inventory 0.7% smaller is very unlikely to move a 2-point gap. So I expect the Bend half
to read "does NOT meet" on the frozen snapshot too, which would make the verdict FAIL
whatever Gainesville shows. This is an expectation, not a measurement.

## 3. Results (Bend, substitute inventory)

Tier 0.30. Bend's run predates the storage floor, so every stored detection is >= 0.55 and
the 0.55 tier is identical. Visible pool: 11,982 of 12,420 ramps. M1 pairs: 10,899 sites
matched to pool ramps (91% of the pool).

**Primary metrics, `auto` frame** (2.6 m in brackets where it differs):

| arm | labels emitted | M1 p50 / p90 (m) | M1 label-weighted p90 | covered r3 / r5 / r8 | M2 one-to-one cov. / p90 | labels per covered ramp | labels > 5 m from any ramp |
|---|---:|---|---:|---|---|---:|---:|
| all (today) | 51,529 | n/a (several per site) | 3.88 [4.01] | 0.911 / **0.920** / 0.923 | 0.932 / 1.31 | 3.95 | 8,027 |
| **best** | 14,058 [14,187] | **0.79 / 1.89** [0.80 / 1.91] | 1.89 | 0.884 / **0.900** / 0.903 | 0.907 / 1.77 | 1.06 | 2,652 |
| random (control, seed 192) | 14,058 | 1.15 / **3.60** [1.15 / 3.73] | 3.60 | 0.785 / 0.852 / 0.874 | 0.875 / 2.93 | 1.09 | 2,980 |
| gate20 | 51,529 | n/a | 3.88 | identical to all | | | |
| best+u (sensitivity) | 15,886 | 0.79 / 1.89 | 1.89 | 0.885 / 0.901 / 0.905 | 0.908 / 1.75 | 1.16 | 3,411 |
| random+u | 15,886 | 1.15 / 3.60 | 3.60 | 0.790 / 0.857 / 0.880 | 0.879 / 2.91 | 1.18 | 3,739 |
| gate_tail (EXPLORATORY) | 31,850 | n/a | 2.36 [2.40] | 0.904 / 0.915 / 0.918 | 0.927 / 1.44 | 2.46 | 4,906 |

- **Paired difference, random minus best** (2000 site-bootstrap reps, seed 192):
  - p50 +0.35 m [+0.33, +0.37].
  - p90 +1.70 m [+1.61, +1.80] at `auto`, and +1.82 m [+1.73, +1.94] at 2.6 m.
  - The two arms emit the same label at 3,527 of 10,899 sites, mostly single-member sites.
  - Over seeds 0-19 the control's p90 runs 3.59-3.72 m (`auto`) and its coverage
    0.847-0.854. The control is stable, and no seed moves the distance clause.
- **Today's labels, label-weighted:** p90 3.88 m from the paired ramp. The best view halves
  that, to 1.89 m. Most of the spread is the server placing tail-range labels (depression
  < 11.25 deg) short. 23,352 of the 51,529 labels (45%) sit in that tail.

**Workload.** `best` emits 14,058 labels instead of 51,529, i.e. **-73%** validator work.
Labels with no inventory ramp within 5 m fall from 8,027 to 2,652. The inventory has no
false positives to judge, so whether those are false detections, unmapped ramps or
misplacements cannot be told here.

**Survivorship.**
- 1,828 of the 51,529 labels (3.5%) are in no fused site: fusion dropped their rays (beyond
  25 m, or above the horizon at the fuse height). `best` and `random` do not emit them;
  `best+u` and `random+u` do. Adding them back recovers only 0.15 points of coverage.
- `best` has 14,058 operational sites, of which 2,387 have a single member.
- The pool and the M1 pairs are the same for every arm within a frame.

**`gate20` is a no-op, structurally.** A server range > 20 m means a depression < 3.6 deg.
Fusion drops every ray past 25 m at a 2.5 m camera height (depression < ~5.7 deg). So no
label past 20 m is ever in a site, and the gate never fires. Only 27 labels exceed 20 m at
all. The exploratory `gate_tail` gates at the server's tail instead (11.77 m). It drops
19,679 labels (38%) for -0.4 points of coverage, and brings label-weighted p90 from 3.88 to
2.36 m.

**Where `best` loses coverage (exploratory, amendment 1).** `all` covers 239 pool ramps that
`best` does not:
- 230 are covered under `all` by labels of a site that is paired with ANOTHER ramp. That is
  typically the second ramp of a dual corner, which fusion holds in one site (its merge rate
  on Bend is about 3%, per `runs/bend/inventory_clustering`).
- 6 come from unpaired sites, and 3 from unplaced labels only.
- 113 of the 239 are covered by a single label under `all`.
- A chance floor puts these numbers in proportion. Moving every label 25 m at random still
  "covers" 11.7% of the pool under `all` and 4.1% under `best`. In an inventory this dense,
  coverage at 5 m grows with label count. So part of `all`'s 2-point lead is density, and
  part is ramps fusion merged. This analysis cannot say which part is which.

**Under the server's own clustering (secondary, approximate) `best` is clearly worse.** The
server cuts complete linkage at 7.5 m. Two lone labels from two panos 3-6 m apart (a dual
corner) always merge. Two clouds of labels usually do not, because a cloud's diameter
exceeds 7.5 m.

| arm (`auto`) | server clusters | covered r5 | split r5 | merge (strict) | merge (loose, amendment 2) |
|---|---:|---:|---:|---:|---:|
| all | 14,655 | **0.901** | 0.057 | 0.013 | 0.078 (825 / 10,630) |
| best | 11,498 | **0.759** | 0.011 | 0.000 | **0.801** (1,704 / 2,128) |
| random | 10,881 | 0.696 | 0.008 | 0.000 | 0.750 |
| gate_tail | 13,582 | 0.892 | 0.024 | 0.006 | 0.055 |

At the cluster level the server would hold 14 points less coverage under `best`: 2,560 of
its 14,058 labels are absorbed into a neighbour's cluster, and 80% of its multi-label
clusters straddle two ramps. The approximation is citywide with ENU distances. Bend has no
server, so per-region clustering is moot there; it is not moot for Gainesville.

## 4. Reading

- **Choosing the steepest member is a real placement win at server placement.** It beats a
  random member by 1.7 m at p90, with a tight CI, in both frames. This is not a frame
  artefact: the label positions are the server's, and the frame only sets which sites
  exist.
- **It does not come for free.**
  - Against today's labels it loses 2 points of per-label coverage on Bend, which fails
    the pre-registered tolerance. That loss is mostly the second ramp of corners fusion
    holds as one site.
  - Without a server change it would make the server's clusters merge dual corners
    wholesale (covered 0.90 -> 0.76).
  - "It needs no server change" is therefore true only of label placement, not of what the
    server builds from the labels.
- **The far-field gate as specified cannot do anything**, because fusion already drops what
  it would have dropped. The exploratory tail gate is the more promising version:
  - it removes 38% of labels and cuts the server clusters' split rate from 0.057 to 0.024;
  - it keeps coverage within half a point;
  - it does not trigger the corner merges.
  It was not pre-registered. It would need its own rule and the Gainesville inventory before
  anyone acts on it.

## 5. Caveats

- The Bend numbers are on a substitute inventory (-0.7% ramps, 8 weeks older). Gainesville
  is unscored, and Gainesville is the city that is not a RampNet training city.
- The inventory has no false positives, so emitting fewer labels can only lose on coverage
  and never gain on precision. The 73% workload cut is unpriced.
- M1's pairs are anchored on fusion's positions, so ramps fusion does not resolve are not in
  the distance comparison. M2, unanchored, agrees in direction: best's p90 is 1.77 m vs
  random's 2.93 m.
- The server-clustering rows are an approximation: citywide, ENU distances, no regions.
  They match the tracked Bend report to 5 clusters.

## Replication

```bash
PY=python
RUNS=runs
$PY -m pytest -q tests/test_best_view_emission.py
# Bend on the substitute (what this report shows; ~1 min, no network):
$PY scripts/best_view_emission.py score bend --runs-root $RUNS \
    --inventory bend=../RampNet/data/inventories/bend-or-2026-07-31.jsonl.gz
$PY scripts/best_view_emission.py verdict
# To settle the rule (NEEDS NETWORK, Jon's approval): pull both inventories to a scratch dir
# (never over the tracked inventory.json), then score from those files.
$PY scripts/inventory_oracle.py fetch bend gainesville --out /tmp/inv
$PY scripts/best_view_emission.py score bend gainesville --runs-root $RUNS \
    --inventory bend=/tmp/inv/bend/inventory.geojson \
    --inventory gainesville=/tmp/inv/gainesville/inventory.geojson
$PY scripts/best_view_emission.py verdict
```

A fresh pull is `verified` only if its sha256 equals the one the committed `inventory.json`
records. That is plausible for Gainesville, whose layer was last edited 2026-07-09. It is
unlikely for Bend, which is edited often. If the hash differs, the script labels the file a
SUBSTITUTE and the verdict stays UNDETERMINED. Whether a fresh snapshot may stand in is a
decision for Jon; this report does not make it. pandas and haversine are NOT needed: the
script imports inventory_clustering's metric functions without eval_ps_clustering.

## 6. Amendments (dated; the rule in section 1 is unchanged)

**Amendment 1, 2026-11-26. Added after a smoke run on Bend's substitute inventory (tier
0.30, `auto`) and before any other cell was scored.** That run showed three things.
`gate20` drops no label at all. `best` lost about 2 points of coverage against `all`. And
the approximate server clustering of `best`'s labels covered far fewer ramps than `best`'s
labels did. Three EXPLORATORY additions follow from it. None of them enters the rule.
- Arm `gate_tail`: the same construction as `gate20`, with the gate at the start of the
  server's linear range tail (server range > 11.77 m, i.e. a depression < 11.25 deg) instead
  of 20 m.
- `coverage_loss`: the pool ramps `all` covers and `best` does not, split by what covered
  them under `all`. The categories are labels fusion could not place, labels of a site
  paired with another ramp, and labels of an unpaired site.
- `chance_covered_r5_<arm>`: coverage after every label is moved 25 m in a seeded random
  direction. This is the coverage a label cloud of that density earns by chance.

**Amendment 2, 2026-11-26. Added after the full Bend run (substitute inventory).** For the
approximate server clustering, the strict merge metric of `inventory_metrics` cannot fire on
a one-label-per-site arm. That metric needs two ramps holding two labels each. So a loose
merge rate (`ps_merge_loose_r5`) is added: clusters whose assigned members span at least two
ramps, over clusters with at least two assigned members. It is EXPLORATORY and does not
enter the rule.

**Amendment 3, 2026-11-27. Added after the overnight report, at Jon's direction.** Both
inventories were re-pulled (network, approved). Jon ruled that a fresh snapshot may stand in
for the frozen one even though its sha256 differs. The rule in section 1 is unchanged, and
the script's `verified` check is unchanged too, so it still prints UNDETERMINED. This
document reads the verdict under the ruling: FAIL (section 2).
