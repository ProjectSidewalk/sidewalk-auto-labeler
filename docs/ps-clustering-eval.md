# Does Project Sidewalk's label clustering group multi-view AI curb-ramp labels correctly?

**Status:** protocol written 2026-09-05 before any result was computed; findings appended
below after the run, and **revised 2026-09-21 after code review** — the metric the protocol
called world recall could barely see the thing under test, so everything is rescored on a
strict coverage metric and one conclusion changed. Companion tool:
`scripts/eval_ps_clustering.py`. Server-side issue:
[SidewalkWebpage#4706](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4706).

## Study goal

The labeler submits one label per pano per detection and deliberately keeps every one of
them: the same physical ramp seen from five panos arrives as five labels. That is by design
(the per-view labels are training signal), so the server's label clustering is what turns
labels into ramps for the API, city stats and validation. Richmond is the first city where
every ramp carries roughly five AI labels, and the server has run its clustering over them.
This study measures whether that clustering represents physical ramps well enough to keep
the per-view labels, and if not, which part is short: the distance threshold, the label
placement, or the merge criterion.

## Research questions

- **RQ1 Validity.** When the server groups AI labels into a cluster, how often is that
  cluster a real ramp?
- **RQ2 One ramp, one cluster.** For a ramp a reviewer can see from a judged pano: is there
  a cluster at it (coverage), is there exactly one (fragmentation), and are two adjacent
  ramps on one corner kept apart (dual-ramp separation)?
- **RQ3 Attribution.** If clustering falls short, is the cause the 7.5 m threshold, the
  server's lat/lng estimate for each label, or the proximity merge criterion itself?

## Method

### Data

- **Labels.** Every `CurbRamp` label on `sidewalk-richmond.cs.washington.edu`, pulled from
  `/v3/api/rawLabels`. The tables below are the 2026-09-21 re-run: 9,639 labels, 9,526 of
  them from the AI user and 113 from three human users. Each AI label maps one-to-one to a
  stored detection in `runs/richmond/results.jsonl` by pano id and pixel position (the tool
  now checks both halves: 9,526 of 9,526 map, every mapped label belongs to one account,
  and no pixel key is ambiguous), so cluster membership can be scored against per-detection
  verdicts. The exact pull is identified by url, timestamp and sha256 at the top of each
  committed report.
- **Deployed clusters.** `/v3/api/labelClusters?includeRawLabels=true`, same pull: 2,156
  clusters covering 9,634 labels (5 labels unclustered).
- **Clustering code.** `scripts/label_clustering.py` from SidewalkWebpage `develop` at
  `99b6d93` (2026-09-05): complete-linkage hierarchical clustering over haversine distance,
  cut at 7.5 m for CurbRamp, run per region, with a cannot-link between labels from the
  same (user, pano).
- **Ground truth.** RampNet's Richmond benchmark: 124 spatially de-clustered panos, every
  operational detection judged true/false/duplicate/unsure and every missed ramp marked,
  with a per-pano attestation that the missed-ramp check was done. World-space GT ramps are
  built exactly as `scripts/eval_sites.py` builds them (raycast of verdict-true detections
  and non-unsure missed marks at 2.6 m camera height, 25 m envelope, cross-pano merge at
  2.5 m). Richmond has 253 placeable GT ramps and 0 cross-pano merges, i.e. the judged panos
  never see each other's ramps.

### Arms

Every arm is a partition of the same AI labels, scored by one scorer in one world frame.

| arm | merge criterion | label positions | scope |
|---|---|---|---|
| `deployed` | server clusters as served | server lat/lng | per region |
| `ps_repro` | PS script, verbatim `cluster()` | server lat/lng | per region |
| `ps @ t` | PS algorithm, t in {2.5, 5, 7.5, 10, 12.5, 15} m | server lat/lng | per region |
| `ps_citywide @ 7.5` | PS algorithm | server lat/lng | whole city |
| `ps_placeable @ t` | PS algorithm, same sweep, restricted to placeable labels | server lat/lng | per region |
| `ps_raycast @ t` | PS algorithm, same sweep | labeler raycast | per region |
| `fusion` | ray-aware associator (`fuse_sites.py`) | labeler raycast | whole city |

`ps_repro` exists only to prove the offline harness reproduces the server (membership
identity is the check). `ps_placeable` holds the label set fixed so that `ps_raycast` and
`fusion` differ from it only in positions and merge rule. `ps_raycast` isolates placement
from algorithm: same merge rule, different positions. `fusion` is the reference the labeler
already measured.

### Cluster placement for scoring

Every cluster is placed at the **mean of its members' labeler-raycast positions** (members
beyond the 25 m envelope are unplaceable and contribute nothing; a cluster with no placeable
member is counted but not placed — the `placed` column in the report). This puts every arm
in the frame the GT is placed in and makes the scores depend only on *who was grouped with
whom*. The server's own centroid is reported separately as a descriptive offset, not used
for scoring. `fusion` is additionally reported at its refit position (`fusion_refit`) to tie
back to the published numbers.

### Metrics (match radius 5 m unless stated)

1. **Precision** (RQ1), membership-based and GT-incompleteness-safe, as `eval_sites.py`:
   over clusters with at least one judged member, TP if any member verdict is true or
   duplicate, FP if all decided verdicts are false, excluded if unsure-only. Note that a
   fragment of a real ramp is a TP, so precision is insensitive to over-splitting by
   construction; that is what metric 3 is for.
2. **Coverage** (RQ2a): pool GT ramps matched one-to-one to a cluster of this arm within
   5 m. *(Revised after review — see "Revised after review" below. The protocol originally
   named `eval_sites`' union recall here; that metric counts a self-detected ramp as
   recovered whether or not any cluster landed on it, so about 83% of it is constant across
   arms. It is still reported, as `recall (union)`, to tie back to `fusion_eval/report.md`,
   alongside `no cluster` — the count of ramps it credits although no cluster is within the
   radius.)*
3. **Fragmentation** (RQ2b): over covered GT ramps, the number of *additional* clusters
   within 3 m (and 5 m) of the ramp that are not the one-to-one match of any GT ramp.
   Reported as the fraction of ramps with at least one extra cluster and extras per ramp.
4. **Dual-ramp separation** (RQ2c): same-pano GT pairs under 5 m apart, both matched to
   distinct clusters ("kept separate"), one matched, or neither.
5. **Coherence**: for each self-detected GT ramp, the distance from the GT ramp to the
   centroid of the cluster that contains that very label. Zero for a singleton; grows as
   wrongly attached members pull the centroid. Median, p90, and the fraction over 5 m.
6. **Descriptives**: cluster count, labels per cluster, size distribution, same-pano pairs
   inside one cluster (must be 0 under the cannot-link), and the server-centroid vs
   raycast-centroid offset per cluster.

### Validation checks, run before reading any result

- `ps_repro` reproduces the deployed partition (fraction of deployed clusters with an
  identical member set).
- The vectorized PS distance reproduces the script's `cluster()` partition exactly.
- `fusion_refit` reproduces `runs/richmond/fusion_eval/report.md`: precision 0.959,
  union recall 0.941, dual ramps 23 / 4 / 3. Those published figures were computed in the
  labeler's 2.6 m frame, so only the 2.6 m run reproduces the world-space half of them;
  in the server frame precision still matches (it is frame-free) and the rest is expected
  to differ. The report line says which frame it ran in.
- *(added after review)* every label that maps to a stored detection belongs to one
  account, and none of that account's labels is left unmapped — so "AI label", which is
  inferred from a pixel match, really is one submitter's label set.
- *(added after review)* the `fusion` arm and the `ps_*` arms cover the same labels: every
  placeable operational detection has a label on the server. They coincide only because
  Richmond submitted every one of them; a gap-filled run or a partial campaign would not.

### Decision rules, fixed in advance

- **Already good enough:** `deployed` is within 3 points of `fusion` on precision and
  recall, fragmentation is not worse by more than 5 points, and dual-ramp separation is not
  worse. Then keep the per-view labels and close the clustering question.
- **Threshold is the lever:** some `ps @ t` reaches the bar above. Then the fix is a
  constant.
- **Placement is the lever:** no `ps @ t` reaches the bar but some `ps_raycast @ t` does.
  Then the merge rule is fine and the label lat/lng estimate is what to fix.
- **Merge criterion is the lever:** neither sweep reaches the bar. Then the ray-aware
  association in SidewalkWebpage#4706 is what it takes.

*(Read after review with **coverage** in the "recall" slot, since that is the quantity the
rule was meant to protect. The rules themselves are unchanged.)*

### Limitations

- The benchmark panos are de-clustered, so GT never says "these two labels from different
  panos are the same ramp". Fragmentation and dual-ramp separation are therefore measured
  in world space, and an "extra" cluster within 3 m could in principle be a real ramp hidden
  from the judged pano. The bias is the same for every arm.
- **The same-ramp scatter statistic borrows its definition of "same ramp" from
  `fuse_sites.py`** — i.e. from the reference arm. GT cannot supply one here: Richmond has
  0 cross-pano merges, so the benchmark never asserts that two labels from different panos
  are the same ramp, which is exactly the assertion the statistic needs. If fusion
  over-merges anywhere, the measured scatter is inflated there. Fusion's precision (0.959)
  and its dual-ramp separation (26 of 32 and 24 of 30 mean-placed, the best of any arm;
  25 and 23 at the refit position, level with `deployed`) bound how much
  over-merging is plausible, but they do not rule it out.
- Placement uses one camera height for every arm in a given run; Richmond's rigs sit lower
  than either value scored here (about 1.7 to 2.0 m), so raycast positions run long. Again
  the same for every arm, and it does not touch membership-based metrics.
- One city, one imagery source (Mapillary). GSV cities have different scatter.

## Findings (run 2026-09-05, revised after review 2026-09-21)

Full tables: `runs/richmond/ps_clustering_eval/report.md` (labeler frame, 2.6 m) and
`runs/richmond/ps_clustering_eval_h2.34/report.md` (server frame, see the deviation note).
Both are committed, and each names the url, fetch time, sha256 and feature count of the two
API pulls it was computed from. The pulls themselves are tens of MB and are not committed;
the sha256 is what makes a re-run auditable.

### Revised after review

A review of the tool found that the metric the protocol called "world recall" — inherited
from `eval_sites.py` — **counts a self-detected GT ramp as recovered whether or not any
cluster landed on it**. 210 of Richmond's 253 pool ramps are self-detected, so about 83% of
that number was constant across every arm by construction, and 9 ramps were credited to the
deployed clustering with no cluster within 5 m of them. It was the weakest possible test of
the thing under test.

The tool now reports **coverage** — a cluster of this arm matched one-to-one to the ramp
within the match radius — as the RQ2a metric, keeps the union recall beside it for the
tie-back to `fusion_eval/report.md`, and prints how many ramps the union metric credits with
no cluster present. Everything below is rescored on that basis. Three of the four
conclusions survive unchanged; **one does not, and it is the actionable one**:

- Unchanged: validity is not the problem (RQ1); fragmentation is the whole shortfall (2 to
  4 times the reference); same-ramp scatter of about 7 m is the mechanism.
- **Strengthened:** coverage no longer merely "matches" the ray-aware reference. Measured
  strictly, the deployed clustering is slightly *ahead* of it — 0.931 against 0.908 in the
  server frame, 0.917 against 0.909 in the labeler frame.
- **Corrected:** the first pass said a 12.5 to 15 m cut recovers most of the fragmentation
  "at no measured cost". It has a cost, and the old metric could not see it. Widening the
  cut **loses coverage**: 0.4 to 1.2 points in the server frame, 3.2 to 3.5 points in the
  labeler frame — the latter outside the pre-registered 3-point bar. The recommendation to
  measure a wider cut stands; the phrase "no measured cost" does not.

### Validation

All checks passed exactly: the verbatim `SidewalkWebpage/scripts/label_clustering.py`
reproduces the server's partition (2,156 of 2,156 clusters identical), the vectorized
distance reproduces the script (2,156 of 2,156), the 2.6 m run's `fusion_refit` reproduces
the published fusion numbers (precision 0.959, union recall 0.941, dual ramps 23 / 4 / 3 —
those were published in the 2.6 m frame, so the server-frame run matches only on the
frame-free precision and is not a failed check), every label that maps to a stored detection
belongs to one account with none of that account's labels left over, and the fusion arm and
the `ps_*` arms cover the same 8,098 labels (0 placeable operational detections without a
label on the server). No arm ever put two labels from one pano in one cluster, no pixel key
in `results.jsonl` is ambiguous, and no two server labels share a pixel — all three counts
are printed unconditionally in the committed reports.

### Deviation from the protocol

The protocol fixed the scoring frame at the labeler's 2.6 m raycast. Reading
[SidewalkWebpage#4819](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/4819) (in the
deployed release) showed the server's estimator is the same flat-ground cotangent at a
calibrated 2.341 m, blending into a bounded linear tail beyond about 12 m. Rescoring at that
height confirmed it: the server's per-label position and the labeler's raycast differ by
0.0 m at the median and 2.7 m at p90 (the tail). So a second run scores everything in the
server's own frame (`--camera-height-m 2.341219672825709`). Membership-only metrics
(precision, cluster counts) are identical across frames; the world-space ones (coverage,
fragmentation, dual ramps) are reported for both. The camera-height constant itself is not
testable on fusion's sites, whose membership was selected at 2.6 m (a circular test that
was run and discarded).

### Headline numbers

Match radius 5 m. "coverage" is the share of pool GT ramps with a cluster of that arm
matched one-to-one within 5 m; "no cluster" is how many ramps the union recall credits
although no cluster is within 5 m; "frag" is the share of covered GT ramps with at least one
extra cluster within 3 m / 5 m; "dual" is same-pano ramp pairs under 5 m kept in separate
clusters.

Server frame (2.341 m), 260 pool ramps:

| arm | clusters | labels | precision | coverage | no cluster | recall (union) | frag 3 m / 5 m | dual kept | coherence > 5 m |
|---|---:|---:|---|---|---:|---|---|---|---:|
| deployed (7.5 m, server positions) | 2,156 | 9,634 | 0.964 (238/9) | **0.931** (242/260) | 4 | 0.946 | 0.17 / 0.48 | 25 of 32 | 3 |
| ps @ 12.5 m, server positions | 1,585 | 9,526 | 0.963 | 0.904 | 10 | 0.942 | 0.11 / 0.25 | 25 of 32 | 9 |
| ps @ 15 m, server positions | 1,487 | 9,526 | 0.963 | 0.896 | 11 | 0.938 | 0.11 / 0.21 | 25 of 32 | 11 |
| ps @ 7.5 m, placeable labels only | 1,887 | 8,098 | 0.959 | 0.931 | 4 | 0.946 | 0.13 / 0.41 | 25 of 32 | 4 |
| ps @ 12.5 m, placeable labels only | 1,461 | 8,098 | 0.959 | 0.900 | 10 | 0.938 | 0.10 / 0.23 | 25 of 32 | 9 |
| ps @ 7.5 m, pure cotangent positions | 1,896 | 8,098 | 0.959 | 0.938 | 1 | 0.942 | 0.09 / 0.30 | 23 of 32 | 1 |
| fusion (ray-aware) | 1,514 | 8,098 | 0.959 (211/9) | 0.908 (236/260) | 8 | 0.938 | 0.08 / 0.17 | 26 of 32 | 12 |

Labeler frame (2.6 m), 253 pool ramps:

| arm | clusters | labels | precision | coverage | no cluster | recall (union) | frag 3 m / 5 m | dual kept | coherence > 5 m |
|---|---:|---:|---|---|---:|---|---|---|---:|
| deployed | 2,156 | 9,634 | 0.964 | **0.917** (232/253) | 9 | 0.953 | 0.24 / 0.47 | 23 of 30 | 11 |
| ps @ 12.5 m, server positions | 1,585 | 9,526 | 0.963 | 0.877 | 17 | 0.945 | 0.14 / 0.29 | 22 of 30 | 18 |
| ps @ 15 m, server positions | 1,487 | 9,526 | 0.963 | 0.874 | 17 | 0.941 | 0.13 / 0.26 | 22 of 30 | 18 |
| ps @ 7.5 m, placeable labels only | 1,887 | 8,098 | 0.959 | 0.897 | 13 | 0.949 | 0.20 / 0.44 | 23 of 30 | 14 |
| ps @ 7.5 m, raycast positions | 1,979 | 8,098 | 0.959 | 0.949 | 1 | 0.953 | 0.07 / 0.31 | 23 of 30 | 0 |
| ps @ 12.5 m, raycast positions | 1,488 | 8,098 | 0.959 | 0.889 | 13 | 0.941 | 0.06 / 0.19 | 23 of 30 | 13 |
| fusion (ray-aware) | 1,570 | 8,098 | 0.959 | 0.909 (230/253) | 9 | 0.945 | 0.06 / 0.16 | 24 of 30 | 12 |

### RQ1, validity: not the problem

Of 267 deployed clusters that contain a judged label, 247 are decided (the other 20 hold
only unsure verdicts and are excluded), and 238 of those 247 are real ramps (precision
0.964). The 9 false clusters are the same 9 false detections every arm carries, so they are
detector errors, not clustering errors. Precision is unchanged at every threshold from 2.5 m to 15 m.
Note what this metric cannot see: a fragment of a real ramp is a true positive, so precision
is insensitive to over-splitting by construction. That is what RQ2b is for.

### RQ2, one ramp one cluster: fragmentation is the whole shortfall

Coverage is fine, and measured strictly it is a little better than the ray-aware reference:
0.931 against fusion's 0.908 in the server frame, 0.917 against 0.909 in the labeler frame.
Dual-ramp separation is **exactly level with `fusion_refit` and one pair short of
mean-placed `fusion`**: 25 of 32 against 25 and 26 in the server frame, 23 of 30 against 23
and 24 in the labeler frame. Which of the two fusion variants the pre-registered "not worse"
clause is read against therefore decides that clause, and it is read against `fusion_refit`
— the refit position is the one `fuse_sites.py` actually writes to `sites.jsonl`, and the
mean-placed variant exists here only to put fusion in the same placement model as every
other arm. Against mean-placed fusion the clause fails by one pair, for `deployed` and for
`ps @ 15 m` alike. What the deployed clustering gets wrong is splitting: 17 to 24 percent of covered ramps have a second
cluster within 3 m and 47 to 48 percent within 5 m, against 8 and 6 percent and 17 and 16
percent for fusion. On the same 8,098 labels the server's rule makes 1,887 clusters where
fusion makes 1,514 to 1,570, so roughly one deployed cluster in five is a fragment of a ramp
that already has one. 411 of the 2,156 deployed clusters are singletons in a city where a
ramp carries five labels.

The mechanism is same-ramp scatter. Over the 945 to 954 **fusion sites** seen from three or
more panos (fusion's grouping is the only available definition of "the same ramp across
panos" here — see Limitations), the largest distance between two members is 7.0 to 7.4 m at
the median under server and raycast positions alike, and 44 to 49 percent of them exceed the
7.5 m cut. Complete linkage keeps a group only if every pair is within the threshold, so
about half of the multi-view ramps cannot survive it. The scatter is in the label positions
(Mapillary's structure-from-motion camera positions, mixed rig heights, and far-view range
error), not in the grouping rule.

### RQ3, attribution

- **Threshold.** In the server's own frame, widening the cut on the server's positions to
  12.5 to 15 m removes most of the fragmentation: 0.11 / 0.21 to 0.25 against fusion's
  0.08 / 0.17, cluster count 1,487 to 1,585 against 1,514, precision and dual-ramp
  separation unchanged (25 of 32 at every cut from 7.5 m up). **It is not free.** Coverage
  falls from 0.931 to 0.904 at 12.5 m and 0.896 at 15 m — 0.4 and 1.2 points below fusion,
  inside the pre-registered 3-point bar but no longer "no measured cost". In the labeler's
  2.6 m frame the same sweep costs far more: coverage 0.877 and 0.874 against fusion's
  0.909, i.e. 3.2 and 3.5 points short, **outside** the bar, while fragmentation only
  reaches 0.14 / 0.29 to 0.13 / 0.26. How far a threshold alone gets therefore still depends
  on which placement model scores it — and under the strict metric one of the two frames
  says it does not get there at all.
- **Why widening costs coverage.** A wider complete-linkage cut merges genuinely distinct
  nearby ramps into one cluster; with one-to-one matching the second ramp is then left
  without a cluster of its own. The "no cluster" column tracks it monotonically: 4 → 10 → 11
  in the server frame and 9 → 17 → 17 in the labeler frame as the cut goes 7.5 → 12.5 → 15 m.
  Dual-ramp separation stays flat because the pairs it counts are the ones a single observer
  saw together, which the cannot-link protects; the ramps that lose their cluster are the
  ones no single pano saw together.
- **Why dual ramps survive a 15 m cut.** With about five views per ramp, nearly every pair
  of adjacent ramps is seen together from at least one pano, the same-(user, pano)
  cannot-link fires, and complete linkage propagates it. The failure SidewalkWebpage#4706
  predicts, two ramps that no single observer saw together, becomes rare precisely because
  every per-view label is kept. This protection is a property of label density, not of the
  algorithm, and a human-labeled city with one label per ramp does not have it.
- **Placement.** At matched height, the server's estimator and a pure cotangent differ only
  in the bounded tail, and that tail costs some fragmentation at 5 m (0.41 against 0.30 at
  7.5 m) and a little coverage (0.931 against 0.938). The larger effect in the 2.6 m frame
  (fragmentation 0.20 to 0.07 at 3 m, coverage 0.897 to 0.949, coherence failures 14 to 0)
  is mostly the frame agreeing with itself, and is not evidence that 2.6 m is a better
  height. Region-wise clustering costs 34 clusters (1.6 percent) at region boundaries and
  nothing else.
- **Merge criterion.** The ray-aware associator groups tighter than any isotropic cut at
  7.5 m on the same positions (1,514 to 1,570 clusters against 1,887 to 1,979) and holds its
  coverage while doing it. A 12.5 to 15 m cut on consistent positions comes within 3 to 4
  points of it on fragmentation but pays 0.4 to 3.5 points of coverage for that. On this
  benchmark the criterion is second-order in the server's frame and first-order in the
  labeler's; its advantage should be clearest where the cannot-link does not cover pairs,
  which is sparse human labeling, and that is not what this data tests.

### Decision, per the pre-registered rules

The bar has four clauses, and the dual-ramp clause depends on which fusion variant it is
read against: `deployed` and `ps @ 15 m` are both exactly level with `fusion_refit`
(25 of 32, 23 of 30) and both one pair short of mean-placed `fusion` (26, 24). Read against
`fusion_refit`, as below.

- **"Already good enough"** — no, and for the same reason as before: fragmentation is 2 to 4
  times the reference (worse by 9 and 31 points, against a 5-point bar). Precision and
  coverage clear their bars easily, coverage in the deployed clustering's favour, and dual
  separation is level.
- **"Threshold is the lever"** — **only in the server's own frame.** There `ps @ 15 m` is
  within 1.2 points of fusion on coverage, within 0.4 on precision, within 3 and 4 points on
  fragmentation, and level with `fusion_refit` on dual separation (one pair short of
  mean-placed fusion). In the labeler's 2.6 m frame no cut in the sweep clears the bar: the
  best fragmentation comes with a 3.2 to 3.5 point coverage loss. The first pass read this
  as "no measured cost" and that reading is withdrawn. Note that read against mean-placed
  fusion instead, the dual clause costs `ps @ 15 m` the server-frame verdict too — the
  threshold conclusion is one pair away from not holding in either frame, which is another
  reason to treat it as "measure it per city", not "set the constant".
- Both frames still agree the grouping rule is sound and the remaining gap is positional
  scatter, which per-rig camera heights
  ([RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158) step 2) attack
  directly.

### What to do with it

1. **Keep the per-view labels.** Precision and coverage of the deployed clusters are as good
   as the reference — coverage is slightly better — and nothing here argues for
   deduplicating at submission. Unchanged by the review.
2. **Measure a wider CurbRamp cut for AI-dense cities before changing code, and measure
   coverage when you do.** 12.5 m on the server's positions removes about half the fragments
   on Richmond for 0.4 points of coverage in the server's own frame — but 3.2 points in the
   labeler's, which is exactly why this has to be measured per city rather than set as a
   constant. The protection that makes a wide cut safe is view density, so it should not be
   applied to human-only cities without the same measurement.
3. **The ray-aware criterion stays the structural fix** for the sparse case a threshold
   cannot cover, as SidewalkWebpage#4706 proposes. It is also the only arm that improves
   fragmentation without paying coverage for it, which is a stronger argument for it than
   the first pass made.
4. **Reduce the scatter itself.** About half the multi-view sites span more than 7.5 m under
   either placement. Per-rig heights and better camera poses shrink that for every arm at
   once, and unlike a wider cut they cost no coverage.
5. **A second city is the next run**, ideally GSV where per-pano depth gives true heights.
   The tool takes a city name and the two API downloads.

## Fusion on what the server holds (`fusion_server`, 2026-09-28)

The `fusion` arm runs on the labeler's `results.jsonl`, which the server does not have.
`fusion_server` runs the same associator (`fuse_sites.fuse`) on **only what the server
holds**, the partition the server would compute if its clustering were fusion. It is an
evaluation arm only and changes no SidewalkWebpage code.

- **Labels:** every live CurbRamp label, AI and human, at its stored pixel. All of them are
  operational: they are live.
- **Confidence:** AI labels take their detection's confidence, which the server stores in
  `label_ai_info`. Human labels get 1.0, so they seed sites first.
- **Camera:** heading from the label row. The position is `pano_data`'s: the run's pano
  block for a pano the labeler submitted, else inverted from that pano's labels. The server
  placed each label with a flat raycast at 2.341 m, so the camera is the label minus that
  offset. Only labels within 15 m are used, and the median is taken.
- **Frame:** the scoring frame, like every other labeler arm (height fields are copied from
  the run's pano when there is one).

Richmond (Mapillary; 2026-09-21 pull), 2.6 m frame, 5 m match radius:

| arm | clusters | coverage | frag 5 m (extra) | dual both/one/neither |
|---|---:|---|---|---|
| deployed | 2156 | 0.917 | 0.47 (132) | 23/4/3 |
| fusion | 1570 | 0.909 | 0.16 (38) | 24/3/3 |
| fusion_server | 3025 | 0.913 | 0.17 (44) | 24/3/3 |

- **Server-only data loses nothing.** 1,521 of `fusion_server`'s 1,587 clusters with AI
  members are, member for member, clusters of `fusion`. Fragmentation stays at fusion's level
  (0.17 vs 0.16, against the deployed 0.47), coverage and dual-ramp separation are unchanged,
  and precision equals the deployed 0.964. Inverting camera positions is accurate: over the
  3,024 panos where both are known, the inverted position is a median 0.01 m and a p90
  0.20 m from the run's.
- **The open design question is the labels fusion cannot place.** 1,434 of 9,639 labels (15%)
  lie beyond the 25 m raycast cap or at the horizon; no site holds them, so the arm makes
  each one a singleton cluster, which is where 3,025 clusters against 1,587 placed ones comes
  from. They are unscored (no raycast position), so the scores above are unaffected. A server
  still has to put them somewhere. Candidates: attach by bearing to a site their ray passes
  near, fall back to the PS distance rule on the server's own lat/lng, or leave them
  unclustered. That choice needs its own measurement, which the scorer cannot give today
  because it places clusters by raycast.

## Step 2 (Vancouver) runbook

Issue [#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56). Vancouver, WA
is the second city: all GSV and AI-dense, and its server has already clustered it. The labels
went live in September 2025, and that run's `results.jsonl` was not kept. By 2026-09 about a
third of the labeled panos no longer resolve on GSV by id. So the run is rebuilt from the
Project Sidewalk pano store on makelab2, with each pano's metadata taken from the Vancouver
server. The tooling is in place; the two launches left are the detection run (makelab2 A40,
~4-6 h) and the RampNet GT session.

`STORE=/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa` (`<id[:2]>/<id>.jpg`,
with pano-tools' `<id>.depth.npz` beside each JPEG), and
`SERVER=https://sidewalk-vancouver.cs.washington.edu`.

1. **Labels and the AI account** (any machine; one GET, cached):
   `python scripts/provenance_gate.py vancouver --server $SERVER --fetch-only` writes
   `runs/vancouver/provenance_gate/raw_labels.geojson` and prints every account's CurbRamp
   count. The AI account is the one with the most labels, and the default everywhere below.
2. **Pano metadata** (on makelab2, no GPU; can run while the A40 is busy):
   `python scripts/detect_from_store.py --run-dir runs/vancouver --store $STORE --labels runs/vancouver/provenance_gate/raw_labels.geojson --labels-user <AI user_id> --sample-unlabeled 300 --seed 56 --server $SERVER --metadata-only`.
   This fixes the pano list (`store_ids.txt` + `store_selection.json` + `store_sampled_ids.txt`:
   the labeled panos plus 300 seeded unlabeled store panos for the benchmark's empty stratum)
   and caches one `/backupImage/<id>/metadata` JSON per pano under `store_metadata/`. The
   selection JSON and the sampled ids are git-tracked; commit them. The selection binds the
   store (resolved path), its layout and the server, so a later call with any of them different
   is refused. The GETs are sequential, spaced 0.2 s apart, and a 429's Retry-After is
   honoured, so ~29k panos take a couple of hours.
   - **Skips.** A pano with no JPEG in the store (`jpg_missing`) or whose metadata answers 404
     is logged in `store_skipped.jsonl` and not retried. **A metadata 404 does not mean "no
     backup":** the server answers 404 unless its file is on disk *and* the pano_data row has
     non-null width, height, lat, lng, camera_heading and camera_pitch
     (`PanoDataService.getLocalBackupImage`). The runner only asks after finding the JPEG, so a
     404 here is most often a null camera_pitch. The reason is recorded as
     `metadata_404_null_field_or_no_file`, and the gate reports that set by count and share,
     since it is not a random hole.
   - **Poison guard.** Either of those two reasons above 5% of a pass (minimum 20) is **not**
     cached: a wrong or unmounted `--store`, or a wrong `--server`, produces exactly that for
     every pano. The pass says so and the panos stay pending. If the rate is real, re-run with
     `--accept-skip-rate` after checking by hand.
3. **Launch 1: detection** (makelab2 A40): the same command without `--metadata-only`. It
   appends to `runs/vancouver/results.jsonl` through main.py's own record writer and resume
   cache. It binds the existing scan-only `manifest.json` to the model revision and adds a
   `pixels` block (store, server, id-list sha256). `--workers` defaults to 8, because a
   16384x8192 JPEG decodes to ~400 MB and the RGB copy doubles the peak to ~800 MB per worker.
   From here on **main.py refuses `runs/vancouver`** (its manifest carries `pixels`), and
   **`send_to_ps.py` refuses the file**: its >= 0.55 detections are the labels already live,
   and PS is insert-only ([SidewalkWebpage#5382](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5382)).
   `--allow-store-file` overrides that; nothing in this runbook needs it.
4. **Provenance gate**: `python scripts/provenance_gate.py vancouver`. The rule below was
   **amended after review ([PR #96](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/96)),
   before any Vancouver number existed**. The first version, PASS iff >= 0.98 of joinable labels
   match within +/-1 px, would have made a STOP from resampling alone likely. The 2025 run
   detected on Google's zoom-3 4096x2048 JPEG. This run detects on a PIL-bilinear 4x downsample
   of pano-tools' native JPEG, which is a different low-pass filter and a different JPEG. A
   peak that moves one heatmap cell (16 px at W = 16384) is therefore expected, and after a STOP
   there would have been no pre-registered path. The amended `verdict()`:
   - **Arm S, store usability (always run).** A label matches when a >= 0.55 detection on the
     same pano lies within +/-(W/1024 + 1) px in x and +/-(H/512 + 1) px in y: one heatmap cell
     plus the rounding pixel, with W and H the pano's stored native size. S passes iff >= 0.98
     of joinable labels match. The +/-1 px share is reported beside it as `exact_share` and
     never gates.
   - **Arm Z, pipeline identity (optional control).** It needs about 200 GSV requests:
     `python scripts/provenance_gate.py vancouver --draw-control` draws 200 labeled ids of the
     run with seed 56 into `provenance_gate/control_ids.txt` (tracked). Then
     `python scripts/reinfer.py runs/vancouver --ids runs/vancouver/provenance_gate/control_ids.txt --out runs/vancouver/control_zoom3.jsonl`
     re-detects them through the 2025 path, zoom-3 pixels via `panorama.fetch_panorama`. A pano
     gone from GSV is skipped there. Finally, `python scripts/provenance_gate.py vancouver --control runs/vancouver/control_zoom3.jsonl`.
     Z passes iff >= 0.98 of the labels on the control's panos match at +/-1 px. Threshold
     flips between the arms (labels matched under S but not Z, and vice versa) are reported.
   - **Coverage floor.** The gate is UNDETERMINED unless no selected pano is pending (every id
     is processed or cached-skipped) and joinable labels are >= 0.95 of the AI labels whose
     pano has a store JPEG. The metadata-404 set is reported by count and share.
   - **Precision.** Detections >= 0.55 on labeled panos that no label claims must be <= 0.02 x
     joinable labels, else STOP. The 2025 run shipped every such detection. Soft-deleted AI
     labels are a known source of them (the detection is still in the run, the label is no
     longer in the pull), so a STOP on this arm is a signal to investigate, not a verdict on
     the store.
   - **Verdict.** PASS only if S passes, precision passes, and (when a control is given) Z
     passes. Otherwise STOP, naming the failing arm. UNDETERMINED takes precedence over STOP.
     On STOP or UNDETERMINED, nothing below runs.
5. **Depth**: `python scripts/harvest_depth.py runs/vancouver --from-store $STORE --check-store-frame 5`
   draws in a seeded order until five panos have been **checked**. A pano gone from GSV does
   not count, and the draw is capped at 4N + 10 live requests. The result is written into
   `runs/vancouver/depth/store.json` (tracked). Then run
   `python scripts/harvest_depth.py runs/vancouver --from-store $STORE`, which says "frame
   unchecked" if no passing check is recorded. This writes `runs/vancouver/depth/index.csv` in
   the harvest's schema from the artifacts, reading them in place. The depth dir is then
   bound to the store: every later call must pass `--from-store` (`--verify --rehash` included),
   and one without it is refused. `fuse_sites.load_depth_index` and `height_qc.py` read the
   index unchanged. `depth_at_detection.py` and `gsv_ground_plane.py` read harvested
   `*.json.gz` payloads rather than the index, so they would need `harvest_depth.payload_from_npz`
   to run on a store index. Panos without an artifact go to `unavailable.txt` when the store's
   `depth_log.csv` says `unavailable` (pano-tools' word for "gone OR served no depth", which a
   harvest would split into `gone.txt` and `no_depth.txt`), and are otherwise listed as pending.
6. **Launch 2: benchmark and GT session.**
   `python scripts/export_benchmark.py runs/vancouver/results.jsonl --bundle ../RampNet/benchmark/vancouver --sample 100 --empty-sample 25 --records-only`.
   Copy those panos' JPEGs in from `$STORE`, re-run without `--records-only` to reconcile,
   then review them in RampNet (Richmond's session took about a day).
7. **The evaluation**, in three frames (four with `auto`). Each writes its own directory, and
   the first call pulls the two API files into `ps_clustering_eval/`; the others read that
   pull through `--labels`/`--clusters`, so every frame scores the same snapshot:

   ```bash
   E=runs/vancouver/ps_clustering_eval
   python scripts/eval_ps_clustering.py vancouver --server $SERVER            # 2.6 m -> $E/
   python scripts/eval_ps_clustering.py vancouver --camera-height-m 2.341219672825709 \
       --labels $E/raw_labels.geojson --clusters $E/clusters.geojson          # server -> ${E}_h2.34/
   python scripts/eval_ps_clustering.py vancouver --camera-height-m per-pano \
       --labels $E/raw_labels.geojson --clusters $E/clusters.geojson          # -> ${E}_per-pano/
   python scripts/eval_ps_clustering.py vancouver --camera-height-m auto \
       --labels $E/raw_labels.geojson --clusters $E/clusters.geojson          # -> ${E}_auto/
   ```

   `per-pano` and `auto` read heights exactly as `fuse_sites.py` does (#56: one shared
   resolver, `fuse_sites.load_at_height`): the pano block's measured height, else
   `runs/vancouver/depth/index.csv` from step 5's `harvest_depth.py --from-store`. The frame
   applies to every labeler raycast (detections, GT, `ps_raycast`, fusion); arms on the
   server's own positions are untouched. The report states the resolution, including how many
   panos had no measured height and fell back to 2.6 m -- read that line before reading the
   per-pano arms, since a thin index makes the "per-pano" frame mostly the constant one.
   `mined_precision.py --camera-height per-pano|auto` goes through the same resolver
   (step 2 of [RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158)).

**Store frame check (2026-09-27, five panos).** pano-tools flips its depth *raster* on write
(sidewalk-panorama-tools#58), and `depth.py` has its own raster mirror (#80). Neither convention touches the
plane indices, which the bridge reads. To show that directly, five store panos were compared
with their live payloads fetched today. The five were the first artifacts in five shards
spread across the id space.

- Four were byte-identical: plane indices, normals and offsets. None matched a mirrored
  index array. On the two whose artifacts were copied here, the ground plane, the image-frame
  `ground_range_at` at eight asymmetric points (8 and 2 of them discriminate a mirror), and
  the artifact's own raster against `depth_at(payload, 1 - x, y)` (0 mismatches) all agree.
- One (`0A5ipVeTM-1W8-lsIhLePg`, saved to the store 2026-09-17) has since been revised
  upstream: 130 planes today against 127. 98.3% of its pixels keep the same plane index,
  against 65.8% for the mirrored artifact. The sky mask is identical. Per-pixel depth is
  identical wherever both have a plane, and the ground plane (2.323 m, 1.01°) is identical.
  `compare_store_frame` reports such a pair as `revised`, not as a frame error.

Heights on those five, for the record: 2.323, 2.199, 1.823 and 2.389 m measured, plus one
2.5 m stand-in ground.
