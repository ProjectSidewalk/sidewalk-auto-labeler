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
