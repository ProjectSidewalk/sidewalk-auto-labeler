# Dual-corner split: one best-view label per split fused site

Agent `dual_corner_split`, overnight batch round 2, 2026-11-27. Script
`scripts/dual_corner_split.py`, tests `tests/test_dual_corner_split.py`, CSVs
`docs/overnight/data/dual_corner_split/`. Builds on `docs/overnight/best_view_emission.md`
(BVE below) and imports its functions; BVE's code and outputs are unchanged.

Question: BVE's `best` arm (one label per fused site, from the member with the largest dip, at
server placement) beat a random member by +1.70 m (Bend) / +0.93 m (Gainesville) at p90 and
cut labels ~70%, but lost 2.0 / 3.7 points of inventory coverage at 5 m against today's `all`,
mostly because fusion holds the two ramps of a dual corner in one site. Can a split rule that
reads only fusion's own output (no inventory, no ground truth) recover that coverage while
keeping the placement and label-count gains?

## 1. Pre-registration (written 2026-11-27, before any split arm was scored on any city)

Nothing in this section was changed after scores existed. The concrete split rule is chosen
on Bend (the tuning city) by the procedure below and frozen in section 2 before Gainesville
(the confirmatory city) is loaded at all. Deviations go to section 8 as dated amendments.

**Inputs.** `runs/<city>/results.jsonl` and `depth/index.csv` for Bend and Gainesville, read
in place. Inventories: the fresh ArcGIS pulls of 2026-11-27 that Jon approved as stand-ins
for the frozen `inventory_oracle` snapshots (BVE amendment 3). Their sha256 differs from the
record, so the script labels them SUBSTITUTE; under Jon's ruling they stand in, and the
verdict below is read under that ruling. Filtering, labels, fusion, frames and tiers are
exactly BVE's (`best_view_emission.load_points`, `synthesize`, `fs.fuse` with
`apply_pose=off`, rig mask on, tiers 0.30 and 0.55, frames `auto` and 2.6 m).

**The split rule's constraints.** It reads only fusion's output for a site: its operational
members' raycast ground points, covariances, bearings, ranges, pano ids, pixel coordinates,
confidences, and the other detections of the same panos. It never reads an inventory or a
RampNet verdict. It partitions a site's operational members into two non-empty parts, or
leaves it whole. Each part then emits one label: `best_view_emission.best_key` over that
part's members (largest dip, then confidence, then key). A site is split at most once.

**Tuning (Bend only).** Candidate rules and parameter grids are explored on Bend, tier 0.30,
`auto` frame, with the Bend inventory. Every Bend number is a TUNING number and is
optimistic. The setting is selected by this objective, fixed now: among candidates whose
Bend clause (ii) gain is >= +1.0 m, take the one with the FEWEST labels whose Bend
`covered_r5` is >= `covered_r5(all) - 0.005`; if none reaches that, take the highest
`covered_r5` (ties: fewer labels). The selected rule and its constants are written into
section 2 and into the script as constants before Gainesville is loaded.

**Arms** (every arm scores labels at the server's positions, `ps_placement`; no arm moves a
label, an arm only chooses which of today's labels to emit):
- `all`: every label (today).
- `best`: BVE's arm, one label per operational site.
- `best+split`: one best-view label per sub-site after the split rule.
- `best+random_split` (control): the same NUMBER of sites is split, chosen uniformly at
  random (`random.Random(192)`) among the sites the rule could have split by size alone
  (operational members >= 2 x the rule's minimum part size, which is >= 2 members); each
  chosen site's members are divided into two parts at random (each member to a part with
  probability 1/2, redrawn until both parts meet the minimum part size); one best-view label
  per part. The rule reads seed 192; seeds 0-19 are reported as a sensitivity check.
- `split+randmember` (reference for clause ii): the rule's sub-sites, one uniformly random
  member each (`random.Random(192)`, sub-sites in (site id, part) order).

**Metrics.**
- Coverage `covered_r5`: BVE's M3 (each label assigned to its nearest kept inventory ramp
  within 5 m; a pool ramp is covered if >= 1 label is assigned). 3 and 8 m reported.
- Placement M1 on split sub-sites: pool ramps matched one-to-one (greedy ascending, BVE's
  `anchored_pairs` logic) to the `best+split` sub-sites' fused positions (the information-
  form refit of each part's members; an unsplit site keeps its position) within 5 m. For each
  pair, the distance from the ramp to the label `best+split` emits for that sub-site, and to
  the label `split+randmember` emits for it. Same pairs for both arms. Paired bootstrap of the
  p90 difference over pairs (one pair per sub-site, 2000 reps, seed 192). BVE's original
  M1 (`best` vs `random` on unsplit sites) is reported beside it.
- Paired coverage difference: `covered_r5(best+split) - covered_r5(other arm)` with a
  95% cluster bootstrap. Each pool ramp is put in the cluster of its nearest operational
  fused site (original fuse) when that site is within 10 m, otherwise in a singleton
  cluster; clusters are resampled (2000 reps, seed 192).
- Labels sent per arm; labels per covered ramp; labels with no ramp within 5 m.
- Over-splitting: for each split site, the number of kept inventory ramps within 8 m of
  its original fused position (0 / 1 / >= 2). A split site with exactly one ramp within 8 m
  most likely gives a duplicate label. Reported for the rule and for the control, beside the
  same distribution over all multi-member sites.
- Secondary (approximate, BVE's): the server's complete-linkage clustering re-run on each
  arm's labels; covered, split and loose merge at 5 m.
- Survivorship: sites, split sites, sub-sites, pairs, pool, unplaced labels (not emitted by
  any one-per-site arm, as in BVE).

**Decision rule** (Gainesville = confirmatory; tier 0.30, `auto` frame). `best+split` PASSES
iff all three hold in Gainesville:
1. `covered_r5(best+split) >= covered_r5(all) - 0.010` (point estimates, as in BVE);
2. `p90_M1(split+randmember) - p90_M1(best+split) >= +0.5 m` (point estimate);
3. the 95% cluster-bootstrap CI of `covered_r5(best+split) - covered_r5(best+random_split)`
   (seed 192) lies entirely above 0.
Otherwise FAIL. The same three clauses are reported for Bend (tuning, not part of the rule),
and for tier 0.55 and the 2.6 m frame (reported, not gating).

What this cannot see: whether a split's second label is a true ramp the inventory lacks or
a false detection (the inventory has no false positives), what validators would do with the
labels, and the server's per-region clustering.

## 2. The rule, frozen on Bend (2026-11-27, before Gainesville was loaded)

**How the rule family was designed (exploratory, Bend only).** Before the grid, I looked at
Bend's 248 ramps that `all` covers and `best` does not. 204 of them sit beside one site whose
labels cover them. In those sites the members that cover the lost ramp have fuse ground
points nearer that ramp than the best label's ramp in 238 of 308 cases, and are farther
views than the best member in 293 of 308. The two ramps are 3.5 m apart at the median (p25
2.5 m). That is the same size as a single ramp's member scatter (the best two-part split of
a site with one inventory ramp within 8 m is 2.0 m apart at the median), so splitting on
geometry alone is a weak signal. Three GT-free quantities separated sites whose split would
gain a ramp from the rest (Bend, AUC over 11,671 multi-member sites, 224 gaining): the
chi-square gain of the best two-part split (0.74), the distance between the two labels the
parts would emit at server placement (0.75), and the server range of the new label (0.71).
An oracle that adds the one best extra member label per site (chosen WITH the inventory)
reaches 0.910, i.e. `all`'s coverage, so the ceiling is there; the question is whether a
GT-free rule finds it.

**Rule family.** For each operational site with >= 2 operational members:
1. Candidate partition: of the cuts perpendicular to 36 directions (5-degree steps) through
   the members' raycast ground points, the one minimizing chi2(part 1) + chi2(part 2), each
   part's chi2 being the information-form GLS residual fusion itself uses
   (`dual_corner_split.bipartition`; both parts non-empty).
2. Part K holds the site's best-view member; part O is the other. Each emits its own
   best-view label (`best_view_emission.best_key`).
3. Split iff `dchi2 = chi2(site) - chi2(K) - chi2(O) >= T`, AND the two emitted labels are
   >= D apart at server placement, AND part O's label has server range <= R.

Grid: T in {0, 1, 2, 3, 4, 6, 9}, D in {0, 1.5, 2, 2.5, 3, 4} m, R in {8, 10, 11.77 (the
server's tail start), 14, inf} m; 210 cells, scored on Bend tier 0.30 `auto`
(`data/dual_corner_split/bend_tuning_grid.csv`).

**Selection, by the pre-registered objective.** No cell short of splitting every
multi-member site reaches `covered_r5(all) - 0.005` = 0.9051. The objective therefore takes
the highest coverage, which is the cell with every threshold off:

> **FROZEN RULE: split every operational site with >= 2 operational members at its
> minimum-chi-square cut** (T = 0, D = 0, R = inf). Bend (TUNING): 11,671 sites split,
> 25,729 labels, `covered_r5` 0.9054 vs `all` 0.9101 and `best` 0.8897; clause (ii) gain
> +1.31 m.

The frontier is smooth: each extra point of coverage costs labels, with no knee where a
selective rule jumps ahead (table in section 4). With every threshold off, the rule no
longer chooses WHICH sites to split, only HOW to divide them. The control
`best+random_split` then splits the same 11,671 sites with random partitions, so clause
(iii) tests whether the chi-square partition is better than a random one. That is not the
question the arms were designed to answer, but it is what the pre-registered objective
gives. Amendment A1 adds an exploratory selective setting.

**Amendment A1 (2026-11-27, after the Bend grid, before Gainesville was loaded).** The
pre-registered pick is degenerate, so a second setting is scored as EXPLORATORY, with its
own matched random-split control and the same three clauses. It is reported, never gates.
`knee`: the fewest-label Bend cell within 1.0 pt of `all` with clause (ii) gain >= 1.0 m:
**T = 2, D = 2.0 m, R = 14 m**. Bend (TUNING): 2,610 sites split, 16,668 labels,
`covered_r5` 0.9005 (`all` - 0.96 pt), gain +1.52 m. It sits exactly at the 1.0-point edge
on its own tuning city, so it has no headroom for Gainesville.

## 3. Verdict: FAIL (Gainesville, confirmatory; fresh inventories standing in under Jon's ruling)

The frozen rule fails all three clauses in Gainesville at tier 0.30, `auto`
(`python scripts/dual_corner_split.py verdict`):

| clause (Gainesville, tier 0.30, `auto`) | needed | measured | |
|---|---|---|---|
| (i) `covered_r5(best+split) - covered_r5(all)` | >= -1.0 pt | 0.8760 - 0.8898 = **-1.38 pt** [-1.80, -1.01] | fails |
| (ii) p90 `split+randmember` - `best+split` | >= +0.50 m | 3.69 - 3.21 = **+0.48 m** [+0.34, +0.60] | fails (narrowly) |
| (iii) `best+split` - `best+random_split` coverage, 95% CI | excludes 0 | **+0.20 pt** [-0.17, +0.56] | fails |

On Bend, where it was tuned, the same rule passes all three (-0.47 pt; +1.31 m; +0.15 pt
[+0.02, +0.26]). The other Gainesville cells fail too: tier 0.55 `auto` fails all three, and
the 2.6 m frame passes (i) and (ii) (-0.86 pt, +0.99 m) but not (iii) at either tier. The
exploratory `knee` setting (amendment A1) fails (i) in every Gainesville cell, by 1.7 to 2.9
points.

So the answer to the question is **no**, not with a rule that reads only fusion's geometry.
Splitting does recover coverage: `best+split` beats `best` by +2.27 pt [+1.77, +2.81] in
Gainesville. But a random partition of the same sites recovers almost as much. What buys the
coverage is the second label per site, not the rule's choice of where to cut.

## 4. Results

CIs are 95% cluster bootstraps by site (coverage) or pair (p90), 2000 reps, seed 192.
Inventories are the 2026-11-27 pulls, SUBSTITUTE by sha256 (Bend 12,591 ramps kept, pool
12,129; Gainesville 3,225 kept, pool 3,040).

**Arms, tier 0.30, `auto`, the frozen rule** (`data/dual_corner_split/<city>_arms.csv`):

| arm | Bend labels | Bend covered r3 / **r5** / r8 | Gainesville labels | Gainesville covered r3 / **r5** / r8 |
|---|---:|---|---:|---|
| all (today) | 51,529 | 0.901 / **0.910** / 0.914 | 21,874 | 0.845 / **0.890** / 0.907 |
| best (BVE) | 14,058 | 0.873 / **0.890** / 0.893 | 6,857 | 0.754 / **0.853** / 0.881 |
| **best+split** | 25,729 | 0.895 / **0.905** / 0.909 | 11,132 | 0.807 / **0.876** / 0.895 |
| best+random_split (seed 192) | 25,729 | 0.892 / **0.904** / 0.907 | 11,132 | 0.808 / **0.874** / 0.893 |
| split+randmember | 25,729 | 0.880 / **0.901** / 0.906 | 11,132 | 0.788 / **0.873** / 0.896 |
| knee best+split (A1) | 16,668 | 0.888 / **0.901** / 0.904 | 7,758 | 0.768 / **0.861** / 0.887 |
| knee best+random_split (A1) | 16,668 | 0.878 / **0.894** / 0.897 | 7,758 | 0.766 / **0.858** / 0.884 |

- **Labels.** The rule splits every multi-member site (Bend 11,671 of 14,058; Gainesville
  4,275 of 6,857). That is 1.8x (Bend) / 1.6x (Gainesville) `best`'s labels, and still 50%
  (Bend) / 49% (Gainesville) fewer than `all`.
- **The control is stable.** Over seeds 0-19 the control's `covered_r5` runs 0.9025-0.9044
  (Bend) and 0.8714-0.8760 (Gainesville). In Gainesville the rule's 0.8760 equals the best
  control seed.
- **Placement.** The rule's sub-sites keep a placement gain over a random member, but it is
  smaller than BVE's on whole sites. On Bend it is +1.31 m [+1.22, +1.40] against BVE's
  +1.70 m. On Gainesville it is +0.48 m against BVE's +0.93 m. A part with one member has no
  choice to make: its best member IS its random member. The `knee` setting, which splits fewer
  sites, keeps more of the gain (Gainesville +0.87 m).
- **The 3 m coverage gap is larger.** At r = 3 m, `best+split` sits 0.6 pt (Bend) and 3.9 pt
  (Gainesville) below `all`. Both are larger than the 5 m gaps.

**Coverage differences with CIs, tier 0.30, `auto`** (`<city>_bootstrap.csv`):

| | Bend | Gainesville |
|---|---|---|
| best+split - all | -0.47 pt [-0.60, -0.35] | -1.38 pt [-1.80, -1.01] |
| best+split - best | +1.57 pt [+1.36, +1.80] | +2.27 pt [+1.77, +2.81] |
| best+split - best+random_split | +0.15 pt [+0.02, +0.26] | +0.20 pt [-0.17, +0.56] |
| knee: best+split - best+random_split | +0.67 pt [+0.50, +0.85] | +0.26 pt [-0.07, +0.59] |

**Other cells** (`verdict` prints every one):

| cell | rule: vs all / p90 gain / vs random_split | knee: vs all / vs random_split |
|---|---|---|
| Bend 2.6 m | -0.55 pt / +1.36 m / +0.13 [+0.00, +0.26] | -1.14 / +0.73 [+0.58, +0.89] |
| Gainesville 2.6 m | -0.86 pt / +0.99 m / +0.13 [-0.10, +0.39] | -1.74 / +0.16 [-0.10, +0.43] |
| Gainesville 0.55 `auto` | -1.22 pt / +0.27 m / +0.33 [-0.07, +0.72] | -2.86 / +0.39 [+0.10, +0.69] |
| Gainesville 0.55 2.6 m | -0.89 pt / +0.58 m / +0.03 [-0.17, +0.26] | -1.94 / +0.20 [-0.03, +0.46] |

Bend's run predates the storage floor, so its 0.55 tier equals its 0.30 tier.

**Over-splitting** (inventory ramps within 8 m of a split site's original fused position,
tier 0.30, `auto`; `<city>_oversplit.csv`):

| sites split | Bend 0 / **1** / >= 2 ramps | Gainesville 0 / **1** / >= 2 ramps |
|---|---|---|
| rule (= every multi-member site) | 1,253 / **3,846** / 6,572 (33% one ramp) | 1,465 / **2,109** / 701 (49% one ramp) |
| knee (A1) | 247 / **754** / 1,609 (29%) | 349 / **395** / 157 (44%) |
| knee's random control | 289 / **901** / 1,420 (35%) | 295 / **440** / 166 (49%) |

Under the frozen rule, a third of Bend's splits and half of Gainesville's split a site with
only one inventory ramp nearby. Those second labels are most likely duplicates. A further 34%
of Gainesville's split sites have NO inventory ramp within 8 m. Gainesville corners are mostly
single-ramp per the inventory: only 16% of its multi-member sites have >= 2 ramps within 8 m,
against 56% in Bend. That is why a rule tuned on dual-corner-heavy Bend transfers poorly. The
`knee` setting does lean toward dual corners: 29% one-ramp splits against its control's 35% on
Bend, 44% against 49% on Gainesville. The lean is too small to matter.

**Approximate server clustering** (secondary; BVE's citywide complete linkage at 7.5 m,
tier 0.30, `auto`):

| arm | Bend clusters / covered r5 / loose merge | Gainesville clusters / covered r5 / loose merge |
|---|---|---|
| all | 14,655 / 0.891 / 0.078 | 7,405 / 0.861 / 0.074 |
| best | 11,498 / 0.751 / 0.800 | 5,771 / 0.796 / 0.311 |
| best+split | 12,843 / 0.833 / 0.119 | 6,254 / 0.833 / 0.070 |
| best+random_split | 13,018 / 0.857 / 0.074 | 6,220 / 0.839 / 0.055 |

Splitting undoes most of `best`'s wholesale corner merges at the server: Bend's loose merge
rate falls from 0.80 to 0.12. The server-side coverage still trails `all` by 6 pt (Bend) and
3 pt (Gainesville). The random control does as well as the rule here, or slightly better.

**Survivorship** (tier 0.30, `auto`). Every arm except `all` drops the labels fusion could not
place: 1,828 of 51,529 in Bend (3.5%) and **2,716 of 21,874 in Gainesville (12.4%)**. The
pool and the cluster units are the same for every arm within a frame. The clause (ii) pairs
are the rule's sub-sites matched to pool ramps: 11,133 (Bend) and 2,703 (Gainesville).
BVE's own pairs were 10,907 and 2,634.

## 5. Reading

- **Geometry cannot tell a dual corner from a single ramp seen from several directions.** In
  Bend the two ramps of a lost pair are 3.5 m apart at the median. A single ramp's members
  scatter by about 2 m at the median under the raycast's range error. The best two-part cut's
  chi-square gain, the label separation and the new label's range each rank gaining sites
  above the rest only moderately (AUC 0.71-0.75). The base rate is 224 gaining sites in
  11,671. No setting of the three gets most of the gaining sites without splitting most of the
  sites.
- **The rule's edge over a random cut is real on Bend and does not carry over.** On Bend it is
  +0.15 pt (rule) or +0.67 pt (knee). On Gainesville it is +0.20 pt and +0.26 pt, with CIs
  crossing 0. Gainesville is the city that is not a RampNet training city, and its sites are
  mostly single-ramp.
- **What "split everything" amounts to is two labels per multi-member site.** In Bend that
  closes 1.6 of `best`'s 2.0 points for 1.8x `best`'s labels (still half of `all`). In
  Gainesville it closes 2.3 of 3.7 points. That is a workload trade-off, not a dual-corner
  detector. A random second member does the same.
- **Gainesville's remaining gap is not dual corners alone.** 12.4% of its labels are never in
  a site, against 3.5% in Bend, and no one-per-site arm emits them. BVE's `best+u`
  sensitivity arm passes them through. Combining it with a split is not pre-registered here.

## 6. Caveats

- The inventories are fresh pulls (SUBSTITUTE by sha256; Jon ruled they may stand in). Their
  record was taken on 2026-09-26, and both layers were edited since.
- The rule family was designed after looking at Bend's lost ramps with the inventory (section
  2). Bend numbers are tuning numbers and are optimistic. Gainesville is the only
  confirmatory evidence.
- The pre-registered objective selected a degenerate cell (split every multi-member site).
  So clause (iii) measured the partition, not the site selection. The `knee` amendment shows
  that the selective version fares no better in Gainesville. It is exploratory.
- The inventory has no false positives. A second label that covers no ramp is a duplicate, an
  unmapped ramp or a false detection, and the inventory cannot say which. The over-splitting
  table uses "one ramp within 8 m" as a duplicate proxy only.
- The coverage metric rewards label density (BVE's chance floor: a cloud of `all`'s density
  moved 25 m still covers 11.7% of Bend). Part of what any extra label buys is that floor.
- The server clustering rows are citywide with ENU distances. Gainesville's real server
  clusters per region.

## 7. Replication

```bash
PY=python
RUNS=runs
INV=<scratch>/inv        # inventory_oracle.py fetch bend gainesville --out <scratch>/inv (network)
$PY -m pytest -q tests/test_dual_corner_split.py
# tuning grid (Bend only; ~70 s):
$PY scripts/dual_corner_split.py tune bend --runs-root $RUNS \
    --inventory bend=$INV/bend/inventory.geojson
# both cities, tiers 0.30/0.55, frames auto/2.6 m, rule + knee (~4 min):
$PY scripts/dual_corner_split.py score bend gainesville --runs-root $RUNS \
    --inventory bend=$INV/bend/inventory.geojson \
    --inventory gainesville=$INV/gainesville/inventory.geojson
$PY scripts/dual_corner_split.py verdict
```

Inventory sha256s: Bend and Gainesville `data/dual_corner_split/<city>_inventory.json`.
Results sha256s are recorded there too (Bend `1307faa8...`, Gainesville `bb8a7872...`).
Everything is deterministic (seed 192; the partition search is a fixed 36-direction sweep
with fixed tie-breaks).

## 8. Amendments (dated; section 1 is unchanged)

- **A1, 2026-11-27, after the Bend tuning grid and before Gainesville was loaded.** An
  exploratory `knee` setting was added (section 2) because the pre-registered objective
  selected the no-threshold cell. It is scored with its own matched random-split control and
  never gates.
