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
- **The live Richmond reports are pinned to that 2026-09-21 pull** (the cached
  `raw_labels.geojson` / `clusters.geojson`, passed with `--labels` / `--clusters`). A
  fresh pull no longer maps to `results.jsonl`: the server now also holds about 3.4k
  0.30-0.55 band labels submitted from `results.band.jsonl`, and the 72 posfix3seq panos'
  labels re-inserted at raw GPS. Those band labels belong to the AI account but match no
  stored detection, so the run refuses (`unmapped_ai > 0`), and the all-label placement gate
  would FAIL on the repositioned panos. Both are correct behaviour; scoring a current pull
  needs the band file as `--results` and the repositioned panos accounted for.
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

- **Labels:** every live CurbRamp label, AI and human, at its stored pixel; never a run
  detection the server has no label for. All of them are operational: they are live. A label
  is AI when its account is the one that submitted the run. An AI-account label that maps to
  no stored detection in `results.jsonl` (a band submitted from `results.band.jsonl`, a
  re-inferred campaign) makes the run refuse rather than enter as human. Two live labels on
  one stored detection (a re-submitted campaign) are two detections; only the first carries
  the run's index, so it is scored once, and the report checks that every label id is in
  exactly one cluster.
- **Confidence:** AI labels take their detection's confidence, which the server stores in
  `label_ai_info`. Human labels get 1.0, so they seed sites first.
- **Camera:** heading from the label row. The position is **inverted from the pano's own
  labels**. The server placed each label with a flat raycast at 2.341 m, so the camera is
  the label minus that offset. Only labels within 15 m are used, and the median is taken.
  Inversion is used rather than the run's pano block, because the block stops being
  `pano_data`'s once a pano is repositioned: Richmond's 72 posfix3seq panos are live at raw
  GPS (since 2026-09-24) while `results.jsonl` holds SfM, a median 4.3 m apart. Inversion
  is also used rather than `position_check.live_positions`, which describes the server
  *today*: on the 2026-09-21 pull, the 27 posfix3seq panos that can be inverted sit a median
  0.01 m from SfM and 4.8 m from raw, so today's records would put them in the wrong
  frame for that pull. Inversion reads the same pull as the labels, so it cannot disagree
  with them. A pano with no label within 15 m falls back to the run's block, and one with
  neither is left out, but its labels still become singleton clusters. The report counts
  each case, and it warns when an inverted position sits more than 1 m from the run's
  block. Two refinements (2026-10-04, #107 review pass): the AI account's labels are
  inverted when a pano has any, and the other accounts' only when it has none, because a
  human label keeps the lat/lng it was inserted at. On Laurens that is a pano position the
  server no longer holds: human-only inversion sits a median 8.7 m from the live block on
  70 panos (AI-only: 0.000 m, the placement gate), and mixed into the median it moved 57 of
  695 panos and cost Laurens' `fusion_server` 5 ramps of coverage (0.769 -> 0.748); with AI
  first it is back at 0.769 (183/238), and 17 human-only panos still warn. Richmond does not
  move. **Offline** there is nothing to invert against: the synthesized labels were placed
  *from* the run's block, so that block is the server's position by construction and is
  used directly (inverting it back only added its own error, p90 0.24-0.33 m).
- **Frame:** the scoring frame, like every other labeler arm. Height fields, peak decode and
  border rule are copied from the run's pano when there is one, so fuse's mixed-decode guard
  (#111) sees the run's real values. A pano only humans labeled takes the run's single
  decode.

Richmond (Mapillary; 2026-09-21 pull), 2.6 m frame, 5 m match radius:

| arm | clusters | coverage | frag 5 m (extra) | dual both/one/neither |
|---|---:|---|---|---|
| deployed | 2156 | 0.917 (232) | 0.47 (132) | 23/4/3 |
| fusion | 1570 | 0.909 (230) | 0.16 (38) | 24/3/3 |
| fusion_server | 3030 | 0.909 (230) | 0.17 (44) | 24/3/3 |

- **Server-only data matches fusion on coverage and the dual split, and is close on
  fragmentation:** coverage 230 vs 230 of 253, the same dual-ramp split; frag 5 m 0.17 vs 0.16
  (40 vs 36 ramps fragmented, 44 vs 38 extra fragments), frag 3 m 15 vs 14, both far below the
  deployed 0.47 / 132. Precision equals *deployed*'s 0.964, not fusion's 0.959, because both
  score every live label. The
  partitions are close but not identical: 1,485 of `fusion_server`'s 1,589 clusters with AI
  members are, member for member, clusters of `fusion`. Of the 3,721 labeled panos, 3,057
  are positioned by inversion and 661 from the run's block. 3 have neither: they are
  human-only panos whose 3 labels are singletons. Inversion is accurate. Over the 3,024 panos
  where both positions are known (mostly AI labels), the inverted position is a median
  0.010 m and a p90 0.20 m from the run's. From human labels alone, over the 12 run panos
  that have them, it is a median 0.007 m, a p90 0.07 m and a max 0.54 m.
- **The open design question is the labels fusion cannot place.** 1,434 of 9,639 labels (15%)
  lie beyond the 25 m raycast cap or at the horizon. No site holds them, so the arm makes
  each one a singleton cluster. That is why there are 3,030 clusters: 1,593 sites (1,589
  with AI members), plus 1,434 singletons, plus the 3 unpositioned labels. Every one of the
  9,639 labels is in exactly one cluster. The singletons are unscored (no raycast
  position), so the scores above do not depend on them. A server still has to put them
  somewhere. The candidates are: attach each to a site its ray passes near (by bearing),
  fall back to the PS distance rule on the server's own lat/lng, or leave them
  unclustered. That choice needs its own measurement, which the scorer cannot give today,
  because it places clusters by raycast.

The numbers above were updated on 2026-10-04, after the PR #105 review. Two changes moved
them, both to the `fusion_server` row only:

- Cameras are now inverted from the labels. Before, they came from the run's block.
- The 3 labels on unpositioned panos are now singletons. Before, they were dropped.

| | clusters | placed | labels | coverage | small-cluster pool |
|---|---:|---:|---:|---|---|
| before | 3,025 | 1,587 | 9,636 | 231 | 21/25 |
| after | 3,030 | 1,589 | 9,639 | 230 | 20/24 |

The member-for-member match with `fusion` went from 1,521 to 1,485.

**Is a small cluster a false positive?** The report's "Precision by cluster size" section
answers this per partition, against RampNet verdicts (Richmond, 2.6 m):

- **Unplaceable labels are not false positives:** 27/27 judged true (Wilson 95% CI
  0.88-1.00). They sit just below the horizon: the median y is 0.523 (the report's
  `median y` column). These are real ramps, too far for the flat raycast. So the fix for
  them is association (e.g. by bearing), not rejection.
- **Placed clusters of 1-2 labels are weaker:** under `fusion_server`, 11/13 and 9/11
  (pooled 20/24 = 0.83, CI 0.64-0.93), against 190/195 = 0.974 for 3+ (Fisher p = 0.010).
  `deployed` shows the same shape: 22/26 vs 188/193, p = 0.013. Its one placeable AI label
  that no server cluster holds is bucketed `unclustered`, not as a cluster of 1. With 24
  judged labels this is a thin sample. Read it as
  validation priority, not a filter: most small clusters are still real ramps, and under
  the recall-first policy a false positive costs one validation while a dropped ramp is
  never seen again. Vancouver's run will add a GSV city with far more labels.

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
4. **Provenance gate**: `python scripts/provenance_gate.py vancouver --rule pixel-96`. The rule
   below is `--rule pixel-96`; since [#111](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/111) (2026-09-30) the
   script's default is `--rule coarse-cell`, which matches within one coarse heatmap cell and is
   exploratory for Vancouver (`docs/heatmap-grid.md`). The rule below was
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
     gone from GSV is skipped there, so the gate refuses a control pano that is not in
     `control_ids.txt` and reports how many drawn panos the control holds. Finally, `python scripts/provenance_gate.py vancouver --control runs/vancouver/control_zoom3.jsonl`.
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
     On STOP or UNDETERMINED, nothing below runs. (For Vancouver this was amended on
     2026-09-29, after the STOP and before any score: see "What followed" in Step 2.)
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

## Beyond Richmond (issue #106, Part 1)

Richmond is one Mapillary city with a live server. Issue
[#106](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/106) asks the same
questions of every RampNet benchmark city, most of which have no server. Tools:
`eval_ps_clustering.py --offline` per city and `scripts/clustering_eval_pooled.py` for the
pooled table. No SidewalkWebpage code changed, and every server call is a GET.

### Method

- **Server labels without a server.** `--offline` synthesizes one label per stored detection at
  the tier, rig-masked, at the integer pixel `send_to_ps.py` sends, and places it with
  `ps_placement.py`: a stdlib port of the server's own estimator (`PanoDataService.toLatLng`,
  "approximation3" at 2.341 m, spherical destination at R = 6371 km). The port passes all 59
  cases of SidewalkWebpage's cross-implementation parity fixture at 1e-9.
- **Regions.** The server gives an AI label the region of the street nearest the label's own
  lat/lng (`ExploreService.submitAiLabelData`), so offline labels do the same against the
  city's `/v3/api/streets`. Point-in-region-polygon was tried first: it disagreed with the
  live region on 6% of Richmond's labels and cost the partition check 6 points. Cities with
  no server are one region, so `ps @ t` equals `ps_citywide` there.
- **Blocked PS partition.** Complete-linkage clusters at a cut t lie inside the single-linkage
  components at t, so the linkage runs per component at the widest threshold + 0.5 m instead
  of on one N x N matrix. It is exact: on Richmond the blocked and dense partitions are
  identical cluster for cluster at every threshold from 2.5 to 15 m, per region and citywide.
  `ps_citywide` therefore runs at any size (largest block over all 25 cells: 301 labels).
- **Unplaceable labels (`fusion_server+attach`).** One rule, fixed before any result and not
  tuned on GT: a label the raycast cannot place joins the placed `fusion_server` site nearest
  along its bearing ray, if one lies 15-60 m ahead and within 3 m of the ray and holds no
  label from the same pano. The site does not move. 15 m is where the server's tail departs
  from the flat raycast, 60 m is about where a ramp stops being resolvable at 4096 px, and
  3 m is the same-ramp lateral scatter.
- **Cells.** Tier 0.55 in the 2.6 m frame is the headline, since every judged bundle was
  reviewed at 0.55 and every published GT number is in that frame. Tier 0.30 at 2.6 m and
  tier 0.55 in the `auto` frame (GSV only) are secondary. Laurens (Mapillary) is scored on
  `results.raw.jsonl`, the file prod went live from.
- **Excluded.** Budapest District V: this machine holds 300 of the run's 18,183 panos and 2 of
  its 125 benchmark panos (the full file was never copied back). Five runs (Bend, Clovis,
  Morgantown, Annapolis, Richmond) predate the storage floor and stored only >= 0.55
  detections. Their 0.30 cells equal their 0.55 cells and are kept out of the 0.30 table.

### Validation gates

| gate | bar | measured |
|---|---|---|
| `ps_placement` vs SidewalkWebpage parity fixture | all cases at 1e-9 | 59 / 59 |
| blocked vs dense partition, Richmond, every threshold, per region and citywide | identical | identical at all 24 (threshold, scope, label set) cells |
| offline placement vs live lat/lng, Richmond (9,526 AI labels, 2026-09-21 pull) | median < 0.05 m | median, p90 and max 0.000000 m; 0 panos moved |
| offline placement vs live lat/lng, Laurens (1,575 AI labels, 2026-09-28 pull) | median < 0.05 m | median, p90 and max 0.000000 m; 0 panos moved |
| offline placement check over ALL labels (moved panos included), Richmond / Laurens | max <= 0.5 m | PASS / PASS (max 0.000000 m) |
| offline `ps @ 7.5 m` vs all-AI deployed clusters, Richmond | >= 0.98 identical | 2,080 / 2,088 = 0.996 (open-street regions; 0.982 before the fix) |
| ...with the live human labels added (all deployed clusters), Richmond | (diagnostic) | 2,156 / 2,156 = 1.000 |
| offline `ps @ 7.5 m` vs all-AI deployed clusters, Laurens | (not gated) | 480 / 508 = 0.945 |
| ...with the live human labels added (all deployed clusters), Laurens | (diagnostic) | 671 / 671 = 1.000 |
| `ps_repro` reproduces deployed (verbatim SW script) | identical | Richmond 2,156 / 2,156; Laurens 671 / 671 |
| same-pano pairs inside any cluster, all 25 offline cells | 0 | 0 in every arm |

The Richmond residual before the #107 review fix was the region rule, not the network: the
offline arm snapped labels to every street `/v3/api/streets` returns (16,365 in Richmond),
while the server snaps to OPEN streets only (704). With open streets only, 9,525 of 9,525
labels take their live region (was 9,394), the all-AI partition matches 2,080 of 2,088
(0.982 -> 0.996) and the with-humans check matches 2,156 of 2,156. (The server's snap set
also holds the tutorial street, which the API omits; no real label is near it.) Where the
all-AI check falls short of the with-humans check, the gap is human labels, which the
offline arm does not have: complete linkage lets a nearby human label change how AI labels
group, which is why Laurens' all-AI figure (0.945) sits below its with-humans 1.000. The
placement check is gated on ALL labels, because the moved-pano exclusion inverts the
estimator it validates (an error consistent within a pano would be excluded, not failed).
Laurens' live report scores the fusion arms rig-masked (`--mask-rig`), because its 158 rig
labels were soft-deleted.

### Pooled results, tier 0.55, 2.6 m frame (headline)

Ten cities, 2,420 pool GT ramps, 5 m match radius. Counts are summed over cities.
Full tables: `runs/_pooled/ps_clustering_eval/report.md`.

| arm | clusters | coverage | frag 3 m / 5 m | dual both/one/neither | precision |
|---|---:|---|---|---|---|
| ps @ 7.5 m (server rule) | 44,575 | 0.872 (2110/2420) | 0.11 / 0.24 | 246/89/20 | 0.950 |
| ps_citywide @ 7.5 m | 44,006 | 0.871 | 0.10 / 0.23 | 245/90/20 | 0.950 |
| ps @ 12.5 m | 39,359 | 0.834 | 0.07 / 0.15 | 219/112/24 | 0.951 |
| ps @ 15 m | 38,351 | 0.821 | 0.07 / 0.14 | 213/113/29 | 0.951 |
| ps_raycast @ 7.5 m | 45,183 | 0.891 | 0.05 / 0.16 | 252/87/16 | 0.950 |
| fusion | 44,359 | 0.876 (2121/2420) | 0.04 / 0.12 | 255/81/19 | 0.949 |
| fusion_server | 57,960 | 0.876 | 0.04 / 0.12 | 255/81/19 | 0.950 |
| fusion_server+attach | 45,360 | 0.876 | 0.04 / 0.12 | 255/81/19 | 0.950 |

By source: GSV (5 cities, 1,264 ramps) `ps @ 7.5 m` 0.894 and 0.11 / 0.23 against fusion 0.921
and 0.04 / 0.14. Mapillary (5 cities, 1,156 ramps) 0.848 and 0.11 / 0.25 against 0.828 and
0.04 / 0.09.

- **Richmond generalizes.** The server's rule fragments about twice as much as fusion
  (frag 5 m 0.24 against 0.12 pooled), in 9 of 10 cities. Gainesville is level (0.19
  against 0.20). Coverage and precision are level: +0.5
  points coverage for fusion pooled, +2.7 on GSV, -2.0 on Mapillary.
- **Widening the cut costs coverage everywhere.** 12.5 m cuts fragmentation most of the way to
  fusion's level (0.15 vs 0.12 at 5 m) but loses 3.8 points of coverage (5.1 at 15 m) and 27 kept dual pairs. That
  is Richmond's labeler-frame result, now pooled. A wider constant is not the fix.
- **Placement matters on its own.** The same rule on the labeler's raycast positions
  (`ps_raycast @ 7.5 m`) has the best coverage of any arm (0.891) and fragmentation between
  the two (0.05 / 0.16).
- **Regions are second-order.** Citywide vs per region differs by 0.1 point of coverage and
  1 point of fragmentation.
- **Small clusters are weaker, unplaceable labels are not.** Per-label precision is 0.72
  (99/137) in `ps @ 7.5 m` singletons and 0.88 in pairs, against 0.98 in clusters of 3+. The
  labels the raycast cannot place are 0.96 (130/135), so Richmond's 27/27 holds at ten times
  the sample.
- **The attach rule** puts 12,600 of the 13,584 unplaceable labels (0.93) on a site: 0.79
  (Richmond) to 0.98 (Paterson, Bend). `fusion_server` then goes from 57,960 clusters to 45,360
  (fusion alone: 44,359). Coverage, frag, dual and coherence cannot move, because
  attached labels have no raycast position. Cluster-level precision can: attaching a
  judged-true singleton to a site that is already TP removes one TP cluster (pooled TP 1,893
  -> 1,891). Richmond's judged ones: 28 of 33 attached (24 of those judged true). Only 1
  attached to a site holding a verdict-true member, since de-clustered benchmark panos rarely
  share sites.

Secondary cells, in the same report:

- **Tier 0.30** (5 cities with a band, 1,206 ramps). The band is unjudged. Pooled `ps @ 7.5 m`
  coverage is 0.907 with frag 0.16 / 0.33. Fusion is 0.910 with 0.06 / 0.18. Fragmentation
  under the server rule grows with the extra labels.
- **Auto frame** (GSV). `ps @ 7.5 m` 0.911 and 0.11 / 0.23 against fusion 0.919 and 0.06 / 0.13.
- **Live `deployed` rows**, for comparison: Richmond at 0.55 has coverage 0.917 and frag
  0.24 / 0.47. Laurens at 0.30 has 0.807 and 0.05 / 0.28.

## City-inventory scoring (issue #106, Part 2)

RampNet GT judges a de-clustered sample of panos, so it never says "these labels from
different panos are one ramp", and it cannot score a split or a merge directly. Bend and
Gainesville publish per-corner curb-ramp inventories (one point per ramp, fetched for #79 by
`scripts/inventory_oracle.py`). Those are an external answer to exactly that question, so
Part 2 scores the partitions against them. Tool: `scripts/inventory_clustering.py`.

### Pre-registration (committed and posted on #106 before any score was computed)

**Inputs.** `runs/<city>/inventory_oracle/inventory.geojson` as filtered by
`inventory_oracle.load_inventory` (Bend 12,504 ramps with `LifeCycleStatus == 'I'`,
Gainesville 3,208), and the run's `results.jsonl`. Labels are synthesized as
`eval_ps_clustering.py --offline` does: one per stored detection at or above the tier,
rig-masked, at the integer pixel `send_to_ps.py` sends. Tier **0.30** (the operating point:
what a server would hold today) is primary; 0.55 is reported beside it.

**Visible pool.** Inventory ramps within 20 m of at least one processed pano position (every
pano of the run with a position). Everything is scored over the pool, so a ramp no pano came
near is not a miss. The pool size and the excluded count are reported.

**Arms.**
- `ps @ 7.5 m`: the server's method. Labels at the server's own positions (`ps_placement`,
  2.341 m), complete linkage with the same-(user, pano) cannot-link, cut at 7.5 m, per region
  for Gainesville (regions from the Gainesville server's street network, the server's own
  nearest-street rule; one GET of `/v3/api/streets`) and citywide for Bend (no server).
  `ps_citywide @ 7.5 m` is reported for both (for Bend it is the same partition).
- `ps @ 10 / 12.5 / 15 m`: secondary, descriptive.
- `fusion`: `fuse_sites.fuse` over the run at the tier (rig-masked, flat pose), in the `auto`
  frame (primary, fuse_sites' default since #79) and in the 2.6 m frame. A cluster is a
  site's operational members.
- `fusion_server+attach`: cluster count only.

**Cluster placement (common frame).** Every cluster of every arm is placed at the mean of its
members' labeler raycast positions in the frame under test (`auto` primary, 2.6 m
secondary), so split and merge respond to the partition only. A member the raycast cannot
place contributes nothing; a cluster with no placeable member is counted in `n_clusters` but
not placed. The server arm's native centroid (the mean of its members' 2.341 m server
positions) is one extra sensitivity row.

**Metrics, at r in {3, 5, 8} m, r = 5 m primary.** Each placed cluster is assigned to its
nearest inventory ramp (any kept ramp, pool or not) if that ramp is within r, else it is
unassigned.
- `covered` = pool ramps with at least one assigned cluster / pool. (`missed` = 1 - covered
  is an upper bound on misses: occlusion and construction since capture are in it.)
- `split` = covered pool ramps with at least two assigned clusters / covered pool ramps. Also
  `extra per covered ramp` = (clusters assigned to covered pool ramps - covered pool ramps) /
  covered pool ramps.
- `merge`: each placeable member of a cluster is assigned to its nearest inventory ramp within
  r the same way. Denominator: clusters with at least two assigned members. Numerator: those
  in which at least two distinct ramps each hold at least two of the cluster's assigned
  members. Both are reported.
- `clusters / covered ramp`, `n_clusters`, `n_labels`.

**Decision rule (read at r = 5 m, tier 0.30, common `auto` frame).** `fusion` is judged better
than `ps @ 7.5 m` if in BOTH cities: split is lower by at least 5 points (absolute), merge is
not higher by more than 1 point, and covered is not lower by more than 1 point. The same
comparison in the 2.6 m frame must not reverse any of the three; a reversal (fusion's split
not lower at all, its merge higher by more than 1 point, or its covered lower by more than
1 point) reads "NOT ESTABLISHED (frame-dependent)". Gainesville decides generalization (not a
RampNet training city); Bend is the guard (a training city, with the larger inventory).
Anything else reads NOT ESTABLISHED. The threshold sweep and the 0.55 tier are descriptive
only.

**Named risks.** Inventory points sit at the ramp, labels at the ramp's foot (about 1-2 m); a
consistent offset moves covered at r = 3 m for every arm equally, which is why r = 5 m is
primary. Bend's inventory is 'Digitized' for 97% of points (map-traced, not surveyed).
Perpendicular ramps at one corner are 1.5-3 m apart and are distinct inventory ramps: a
cluster spanning them counts as a merge under this rule, as intended. Bend was a RampNet
training city, so its detections are not a generalization test; clustering is what is
measured there. Gainesville is read-only (no submission, no re-detection).

The pre-registration above was committed in `ebc3652` and posted on #106
([comment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/106#issuecomment-5875088671))
before `scripts/inventory_clustering.py` existed. It was scored once, unamended. After the
#107 review it was re-scored once with the region rule corrected to OPEN streets only, the
implementation catching up to the registered "server's own nearest-street rule" (noted on
#106 before re-scoring,
[comment](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/106#issuecomment-5879886962));
the rule is unchanged. Only Gainesville's `ps @ t` rows moved (7,936 -> 7,930 clusters at
0.30), and the verdict did not.

### Results (2026-09-28)

`runs/<city>/inventory_clustering/report.md` (all tiers, frames and radii) and
`runs/_pooled/inventory_clustering/{report.md,verdict.json}`. Visible pool: Gainesville
3,107 of 3,208 ramps, Bend 12,065 of 12,504. Values at r = 5 m and tier 0.30.

| city | frame | arm | clusters | covered | split | merge (k/n) |
|---|---|---|---:|---|---|---|
| gainesville | auto | ps @ 7.5 m | 7,930 | 0.864 | 0.311 | 0.009 (29/3117) |
| gainesville | auto | fusion | 7,151 | 0.863 | **0.211** | 0.017 (48/2763) |
| gainesville | 2.6 m | ps @ 7.5 m | 7,930 | 0.833 | 0.288 | 0.024 (69/2866) |
| gainesville | 2.6 m | fusion | 8,967 | 0.851 | **0.331** | 0.016 (43/2633) |
| bend | auto | ps @ 7.5 m | 14,650 | 0.896 | 0.093 | 0.016 (170/10780) |
| bend | auto | fusion | 14,058 | 0.897 | 0.048 | 0.031 (316/10324) |
| bend | 2.6 m | ps @ 7.5 m | 14,650 | 0.893 | 0.098 | 0.020 (220/10788) |
| bend | 2.6 m | fusion | 14,187 | 0.897 | 0.048 | 0.034 (346/10287) |

**Verdict: NOT ESTABLISHED.** Gainesville, the deciding city, meets the rule in the `auto`
frame: split falls 10.0 points, merge rises 0.8 and covered falls 0.1. In the 2.6 m frame it
reverses, and fusion splits more than the server rule (0.331 against 0.288). Bend meets
neither clause that matters there: split falls 4.6 points against a 5-point bar, and merge
rises 1.5 points against a 1-point allowance. Bend's 2.6 m comparison also reads as a
reversal under the rule, on merge alone (+1.3 points; its split still halves).

What the numbers say beyond the verdict (descriptive, not part of the rule):

- **In Gainesville the split advantage depends on the height frame; in Bend and in the Part 1
  GT pool it holds at 2.6 m** (Bend split 0.048 against 0.098; Part 1 pooled frag 5 m 0.12
  against 0.24, in 9 of 10 cities). 64% of Gainesville's panos are the 2026 GSV rig (depth
  median 1.76 m), which `auto` raycasts at 2.0 m. At 2.6 m fusion splits more there (8,967
  clusters against 7,151), while the server's distance-only rule moves less (0.311 to
  0.288). A plausible mechanism is that 2.6 m runs that rig's ranges long
  (docs/camera-height-study.md) and the ray-aware gate then refuses to merge views that
  disagree on range, but that is descriptive: nothing here tests it.
- **Fusion trades split for merge in Bend.** Split halves (0.093 to 0.048), but merge doubles
  (1.6% to 3.1%). Widening the server cut to 10 m gets a similar split (0.057) with less
  merge (2.0%) and 0.5 points less coverage.
- **The server's own centroid** (sensitivity row) scores the PS clusters better than their
  raycast mean does. Gainesville's split is 0.267 and Bend's 0.057, and in Gainesville the
  2.6 m covered rises from 0.833 to 0.871. So part of the frame effect is in the scoring
  frame, not in the partition.
- **Bend's tiers are one tier.** Its `results.jsonl` predates the storage floor and holds only
  >= 0.55 detections, so the 0.30 read there is the 0.55 read. The rule was applied as
  written to what the file holds.

Caveats (from the issue). `missed` = 1 - covered is an upper bound, since occlusion and
construction since capture are in it. Bend was a RampNet training city. Gainesville was read
only: no submission, no re-detection. Vancouver's inventory is down, so two cities is all
there is.

## Step 2: Vancouver (run 2026-09-28): the gate stopped, then an amended scope was scored

Issue [#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56). The runbook above
was followed through step 4. **The gate returned STOP, and Arm Z, run after it, failed too.** Depth,
fusion and the benchmark bundle do not depend on the gate, so they were run. On 2026-09-29 the
scope was amended on #56, before any score was computed (see "What followed" below): the scoring
of the server-label arms is in [PR #118](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/118), not here.

**The run.** On makelab2 (A40), `detect_from_store.py` finished on 2026-09-28: 28,830 panos, 0
failed, 351 cached `jpg_missing`. That covers the 28,881 labeled panos plus 300 seeded unlabeled
ones, 29,181 selected ids in all. There were 0 metadata 404s and 0 native-size mismatches. The
run was copied home with sha256 verified on both ends (results.jsonl `7fdf4005…9f28`).

**The gate** (`runs/vancouver/provenance_gate/report.md`, rule unchanged, `--rule pixel-96`). The
labels were pulled fresh on 2026-09-28T23:01Z: 64,847 CurbRamp features, 64,814 of them from the
AI account on 28,881 panos, and 64,006 of those joinable.

| check | value | rule | result |
|---|---:|---|---|
| Arm S share | 0.8252 (52,819 / 64,006) | >= 0.98 | fail |
| exact_share (+/-1 px) | 0.7385 | -- | -- |
| Arm Z share (zoom-3 control, +/-1 px) | 0.9632 (288 / 299), on 141 of the 200 drawn panos | >= 0.98 | fail |
| coverage | 1.0000 | >= 0.95 | pass |
| unclaimed tier detections / joinable | 0.1316 (8,424) | <= 0.02 | fail |

To reproduce the report from its untracked inputs (`results.jsonl`, `control_zoom3.jsonl`,
`provenance_gate/raw_labels.geojson`; each sha256 is in the report):

```bash
python scripts/provenance_gate.py vancouver --control runs/vancouver/control_zoom3.jsonl --rule pixel-96
```

Without `--rule pixel-96`, the script's default since #111 (`coarse-cell`) writes a different,
exploratory reading into the same path: S 0.9301, Z 0.9866 (pass), P 0.0288, still STOP
(`docs/heatmap-grid.md`).

**Arm Z** (run 2026-09-28, after the gate, rule unchanged). 200 labeled panos drawn with seed 56
(`provenance_gate/control_ids.txt`) were re-detected through the 2025 path, zoom-3 GSV pixels via
`reinfer.py --ids`. **141 were written; the other 59 (29.5%) are no longer served by id**, so the
Z share is over the panos that survived on GSV, a non-random subset (#56 measured survival by
capture year). 288 of the 299 labels on them match at +/-1 px: 0.9632, below 0.98. On those
panos 247 labels match under both arms, 7 only under S, 41 only under Z and 4 under neither.
Plateau end-flips (exactly 7 cells) are 3 of 299 in Z, against 33 of 299 from the store.

**What the misses are.** This diagnostic is not part of the rule. Of the 11,187 unmatched labels:

- 3,502 are threshold flips: a stored detection sits within tolerance, but below 0.55.
- 7,679 have no detection within 64 px. **7,339 of these sit exactly 7 heatmap cells from a
  stored detection** (Chebyshev distance). 6,170 of them are axis-aligned: (+7, 0) 1,877,
  (-7, 0) 1,875, (0, +7) 1,283, (0, -7) 1,135. The two horizontal directions are even; the
  vertical ones are about a third smaller.
- Almost no miss lies between 1 and 6 cells: widening the tolerance from 1 to 2 cells adds 3
  labels, and from 2 to 6 cells 1 more (0.8252 to 0.8253). At 7 cells Arm S would read 0.9285, at
  8 cells 0.9301 at >= 0.55 and 0.9967 at any stored confidence.
- The rate is flat across capture years (0.81-0.84), pano widths and label days, and no global
  heading shift fits.

Re-running the model on one far-miss pano (`-65GoVmwedYvlbgkb8nAgA`) shows the mechanism. The
heatmap has an **8-cell plateau** at 0.91 (row 307, columns 236-243). `peak_local_max` keeps one
pixel of it: column 236 today, and column 243 for the 2025 label. The unclaimed detections are the
other ends of the same plateaus. So the STOP comes mostly from the model's output, not from the
store, and a +/-1-cell rule cannot be met by any rebuild that perturbs the input pixels.

A second, rare pattern also appeared. On about 19 panos, every label looked moved by one linear
horizontal map, for example x2025 = (x - 0.125)/0.875 on `qWxSMzkdRIaDQegUsxsY2w`. Re-fetching all
19 from GSV after Arm Z showed it is **mostly a store-vs-GSV pixel difference**, not misplaced
labels: 13 are still served, and 12 of those 13 reproduce every label at +/-1 px from fresh zoom-3
pixels and fail only from the store. One, `qWxSMzkdRIaDQegUsxsY2w`, is displaced against both (0 of
6 from fresh GSV, whose detections agree with the store run's to within a cell), so for that pano
the 2025 input differed from both today's GSV and the store. Most of the 19 were fitted with the
smallest shrink the fit allows, so some may be plateau flips the fit mislabelled.

**What followed (2026-09-29, on #56).** The scope was amended before any score was computed. The
STOP's cause is the heatmap grid ([#111](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/111)), which touches
only the arms built from the rebuilt run's detections; the arms built from the server's own labels
(`deployed`, `ps@t`, `fusion_server`, `fusion_server+attach`) and the inventory never depended on
the gate. Those are pre-registered on #56 and scored in [PR #118](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/118),
with the rebuilt run's own arms reported there as exploratory. The gate verdict above stands as
recorded and is not re-scored under any other tolerance; the coarse-cell re-read for #111 is
exploratory (`docs/heatmap-grid.md`).

**Depth** (`runs/vancouver/depth/store.json`, tracked). The frame check found 5 of 5 store
artifacts identical to the live payloads, with 0 revised. The index covers 18,473 of 28,830 run
panos:

- 13,572 have a measured height (47.1% of the run). The median is 2.356 m (p25 2.268, p75 2.417).
- 4,890 have a stand-in ground and 11 are degenerate.
- 8,952 panos are `unavailable` in pano-tools' ledger, and 1,405 had no artifact yet (the depth
  phase was still running).

Under `per-pano`, 15,258 panos (52.9%) therefore fall back to 2.6 m.

**Fusion** (`fuse_sites.py`; `sites_meta.json` untracked). Every capture year's depth median is
above the 2.1 m cut (2.24-2.43 m), so `auto` puts **all 28,830 panos at 2.5 m**. The low 2025-26
rig is absent: the labels were made in 2025-09, and the 2025 captures here (497 panos) read
2.362 m.

| capture year | panos | measured | median (m) |
|---|---:|---:|---:|
| 2011-2018 (seven years) | 3,036 | 153 | 2.24-2.43 |
| 2019 | 1,663 | 968 | 2.324 |
| 2021 | 926 | 506 | 2.370 |
| 2022 | 3,434 | 1,624 | 2.354 |
| 2023 | 8,653 | 4,562 | 2.366 |
| 2024 | 10,621 | 5,482 | 2.349 |
| 2025 | 497 | 277 | 2.362 |

| frame | sites | operational | multi-pano |
|---|---:|---:|---:|
| auto (2.5 m) | 24,080 | 17,650 | 15,769 |
| 2.6 m | 23,925 | 17,551 | 15,609 |
| per-pano | 24,612 | 18,094 | 15,896 |

**Benchmark bundle.** It sits on makelab2 at
`/projects/makeabilitylab/sidewalk-auto-labeler/runs/vancouver/benchmark/`: 125 panos (top 5,
random 95, empty 25), with native JPEGs copied from the store. The reconcile reads OK, 125/125
into `index.csv`. The pixels are not committed. The bundle's README notes two caveats: Portland
(the same metro) was in RampNet's Stage-1 training, and the gate stopped.

**City inventory.** The City's hosted `COV_TransCurbRamp` layer replaces the dead proxy (see
`docs/placement-oracle.md`). It was fetched on 2026-09-28: 11,355 in-area ramps with
`STATUS = 'Available'`. It was scored under the amended scope below, in
[PR #118](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/118).

### Scoring under the amended scope (2026-09-29; corrected 2026-10-04)

Jon decided to score (2026-09-29) under an amended scope, pre-registered as the last comment
on [#56](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56) before any
number was computed. The gate's STOP comes from the heatmap grid (#111), which affects only
the arms built from the rebuilt run's detections. So the arms built from the server's own
labels (`deployed`, `ps @ t` on server positions, `fusion_server`, `fusion_server+attach`)
carry metrics (a)-(d), and everything built from the rebuilt run's detections is
exploratory (`fusion` at auto / 2.6 m / per-pano, the offline synthesis at 0.55 and 0.30).
Vancouver is confirmatory for Part 1's fragmentation claim only, and the pre-registration
names the test: metric (b), the near-cluster proxy, with the expected direction
"`fusion_server` fragmentation at 5 m about half of `ps@7.5`". It also says "(c), (d) and the
inventory are descriptive here". The gate verdict stands as recorded and was not re-scored.
No GT exists, so there is no coverage or recall column.

**Pre-registered result: fusion did not halve fragmentation on the pre-registered proxy; it
read more.** At 5 m, `fusion_server` has another cluster within reach for 0.420 of its
clusters against 0.274 for `ps @ 7.5 m` (ratio 1.53; 0.408 / 0.274 in the `auto` frame,
0.426 / 0.274 per-pano). The expected direction was about half. Everything below that
bears on fragmentation is descriptive or post hoc, and none of it replaces this result.

**This section was corrected on 2026-10-04** after the
[#118 review](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/118#pullrequestreview-5407479509).
The first version dropped the proxy and answered the fragmentation question from the
inventory split rate, which the pre-registration called descriptive. That switch is withdrawn
(see "Deviations" below). The re-run also takes #105's label map, which puts each of the
1,058 labels that share a pixel with another label in exactly one cluster; the numbers that
moved are listed in the PR thread.

**Inputs, pulled once.** Labels: the gate's frozen `provenance_gate/raw_labels.geojson`
(64,847 CurbRamp features, sha256 `57c31c73…3098d`, 2026-09-28T23:01Z). Deployed clusters:
`ps_clustering_eval/clusters.geojson` (18,684 clusters over 64,918 memberships, sha256
`d8a1e706…c1c3`, 2026-09-30T01:39Z). Streets: `ps_clustering_eval/streets.geojson` (12,567
features, sha256 `0d7ce797…f8b4`, 2026-09-30T01:39Z). Every report reads these files by path,
so every report records the same streets sha256. `inventory_clustering.py` reads the same
streets file and refuses if it is absent. Inventory: `COV_TransCurbRamp`, 11,355 kept (sha256
in `inventory.json`). Tier 0.55 (what went live). **No `--mask-rig`**: Vancouver's rig labels
were never soft-deleted, so the server holds them and the arms score what it holds. Scorer
106.3.

#### Deviations from the pre-registration

1. **`--ai-user` (forced by the data).** The scorer identifies an AI label by a pixel-exact
   match to a stored detection. Here 50,252 of the AI account's 64,814 labels match the
   rebuilt run; 14,562 (22.5%) do not, for the plateau reason the gate found. `--ai-user`
   names the account, so all 64,814 are AI labels in every server-label arm. The 14,562
   unmapped ones are never placeable by the raycast and enter `fusion_server` at the tier
   (0.55, the lowest confidence the server can hold; the feed does not carry
   `label_ai_info`). Without the flag the scorer refuses such labels.
2. **The label-level anchor** is 2,911 human-voted AI labels in the frozen pull, not the
   2,632 quoted in the pre-registration from an earlier server read.
3. **The fragmentation read was switched, then withdrawn.** After (b) came out against
   Part 1, the first version of this section called the proxy uninformative (from a post-hoc
   Richmond calibration) and answered the fragmentation question from the inventory split
   rate. That was a post-hoc change of the confirmatory metric. It is withdrawn: (b) is the
   result, and the inventory is reported below as descriptive, in both placement frames.
4. **Post hoc, labelled so:** the placeable-member column of the (b) table, the Richmond
   calibration of the proxy, the deployed-partition diagnostics under (a), and the
   matched-label-set inventory table. None of them is a confirmatory test.

#### (a) Verbatim reproduction: 0.765, not the expected >= 0.98

`ps_repro` (SidewalkWebpage `label_clustering.py` at `origin/develop` 0062ed0, per region, on
the labels the server has clustered) reproduces 14,300 of 18,684 deployed clusters (15,431
labels in clusters that differ). The vectorized partition reproduces the script 17,412 /
17,414. Richmond and Laurens were 1.000 on the same check. What differs is the deployed
table itself (the `Deployed-partition diagnostics` section of
`runs/vancouver/ps_clustering_eval/report.md`, at the labels' current positions):

- 276 deployed clusters (1.5%) span more than 7.5 m, which complete linkage at 7.5 m cannot
  produce;
- 3,665 cluster geometries (19.6%) sit more than 1 m from the mean of their current members
  (max 11.1 m);
- 154 labels sit in two clusters;
- 2,896 of the 4,384 clusters `ps_repro` does not reproduce are proper subsets of one
  `ps_repro` cluster (deployed is the more fragmented one there), and 929 of those 2,896
  were created all before or all after the rest of their fresh cluster;
- running the script with each label in its deployed cluster's region instead of its own
  reproduces 14,175 / 18,684 (0.759), so regions do not explain it.

The server re-clusters a region only when its label MEMBERSHIP changes
(`ClusteringSessionTable.getRegionsToCluster`), so a label position recomputed after the
September 2025 clustering never triggers a re-run. That is the reading most consistent with
these symptoms; it is not proven here. `ps @ 7.5 m` (the vectorized twin of `ps_repro`) is
what the tables below compare against.

#### (b) Fragmentation proxy (the confirmatory test)

Share of an arm's clusters with another cluster of the same arm within r, clusters at the
mean of their labels' server positions. 2.6 m frame for the fusion arms, from
`runs/vancouver/ps_clustering_eval/arms.csv` (`near5`, `near7_5`, `near12_5`,
`per_1000_labels`, `nearp5`); the `auto` and `per-pano` reports have the same table.

| arm | clusters | labels | clusters / 1,000 labels | near 5 m | near 7.5 m | near 12.5 m | placeable-member clusters, near 5 m (post hoc) |
|---|---:|---:|---:|---:|---:|---:|---:|
| deployed | 18,684 | 64,918 | 287.8 | 0.352 | 0.583 | 0.799 | 0.338 |
| ps_repro | 17,414 | 64,764 | 268.9 | 0.274 | 0.516 | 0.771 | 0.277 |
| ps @ 5 m | 21,049 | 64,814 | 324.8 | 0.439 | 0.678 | 0.842 | 0.426 |
| ps @ 7.5 m | 17,451 | 64,814 | 269.2 | 0.274 | 0.515 | 0.769 | 0.276 |
| ps @ 10 m | 16,034 | 64,814 | 247.4 | 0.260 | 0.445 | 0.723 | 0.260 |
| ps @ 12.5 m | 15,421 | 64,814 | 237.9 | 0.262 | 0.431 | 0.693 | 0.262 |
| ps_citywide @ 7.5 m | 17,310 | 64,814 | 267.1 | 0.261 | 0.506 | 0.765 | 0.264 |
| fusion_server (2.6 m) | 18,424 | 64,847 | 284.1 | **0.420** | 0.581 | 0.785 | 0.280 |
| fusion_server+attach (2.6 m) | 16,109 | 64,847 | 248.4 | 0.310 | 0.472 | 0.728 | 0.278 |
| fusion_server (auto) | 18,484 | 64,847 | 285.0 | 0.408 | 0.581 | 0.787 | 0.263 |
| fusion_server+attach (auto) | 16,171 | 64,847 | 249.4 | 0.294 | 0.471 | 0.730 | 0.262 |

`fusion_server` reads higher than `ps @ 7.5 m` at every radius. Its 2,369 labels the
raycast cannot place are singletons at their server position, beside the site they could
not join, and those count as near pairs. That is part of what the arm does on a live server,
not an artifact to remove. On the post-hoc placeable-member read the two are level (0.280
against 0.276).

What the proxy tracks, read post hoc on Richmond, where GT exists
(`runs/richmond/ps_clustering_eval/arms.csv`, columns `nearp5` and `frag5_with_extra` /
`frag3_ramps`): placeable-member near 5 m is 0.414 (`deployed`), 0.409 (`ps @ 7.5 m`) and
0.425 (`fusion_server`), where GT `frag` at 5 m is 0.47 (109 / 232), 0.47 (108 / 231) and
0.17 (40 / 230). On Richmond the proxy does not follow `frag`: most neighbours are other
ramps of the same corner. That is a reason to read (b) with care. It does not change what
(b) found.

#### (c) Validation-based precision, human votes (descriptive)

The frozen pull holds 2,911 AI labels with a human vote: 2,713 true, 81 false, 117
tie/unsure, so the label-level anchor is **0.971** [0.96, 0.98]. The feed's `correct` reads the
same over the AI labels (2,713 true, 81 false). **`deployed` holds no false label at all.** The server clusters only
labels not marked incorrect (`labelsForApiQuery`), and all 81 human-false labels are among
the 83 labels in no cluster. So (c) on `deployed` is 0 by construction, and the informative
rows are the re-clustered arms, which hold every label (2.6 m, from
`runs/vancouver/ps_clustering_eval/validation_precision.csv`):

| arm | cluster of 1: any false | cluster of 2 | cluster of 3+ |
|---|---|---|---|
| ps @ 7.5 m | 0.243 (41 / 169) | 0.062 (15 / 243) | 0.011 (25 / 2,253) |
| fusion_server (2.6 m) | 0.171 (45 / 263) | 0.097 (19 / 195) | 0.008 (17 / 2,183) |
| fusion_server+attach (2.6 m) | 0.277 (44 / 159) | 0.091 (18 / 197) | 0.008 (19 / 2,265) |

(Share of clusters holding a validated label that hold one voted false; counts in
parentheses.) A false label sits in a singleton about a quarter of the time and in a 3+
cluster about 1% of the time, whichever rule made the cluster: the Richmond size-precision
pattern, on 6.7x the labels. Every label is in exactly one cluster of each arm now
(64,847 label ids, 64,847 distinct), so these 81 false labels are counted once.

#### (d) Deployed vs offline partition: 0.733 (descriptive)

`--offline-check`: placement of the 64,006 live AI labels the file can re-place is exact
(median 0, p90 0) with one label at 0.550 m, so the 0.5 m gate reads FAIL on that one label
(the scorer exits 1; the report is complete). 46,257 of 60,104 synthesized labels match a
live AI label by pixel; 18,557 live AI labels have no twin (the gate's finding again). Of the
6,956 deployed clusters whose labels are all AI and all synthesized, 5,099 are reproduced
label for label (0.733). Richmond read 0.996 and Laurens 0.945, but the ceiling here is (a):
`ps_repro` itself reproduces only 0.765 of `deployed`.

#### Inventory placement (descriptive; `inventory_clustering.py score vancouver`)

Visible pool 10,685 of 11,355 ramps within 20 m of one of the run's 28,830 panos; r = 5 m.
The split rate depends on the placement frame, and which arm reads lower depends on it too.
The table below is the `label_set: mapped` block of
`runs/vancouver/inventory_clustering/report.md` (`auto` frame): every arm on the same labels
(the 50,252 AI labels the rebuilt run maps, plus the 33 human labels), placed both ways.

| arm (same label set) | split, raycast placement | split, server placement | merge (raycast / server) |
|---|---:|---:|---|
| ps @ 7.5 m | 0.163 (1,518 / 9,285) | 0.098 (905 / 9,257) | 0.020 / 0.019 |
| fusion_server | 0.090 (840 / 9,332) | 0.160 (1,483 / 9,256) | 0.024 / 0.031 |
| fusion_server+attach | 0.090 (840 / 9,332) | 0.093 (861 / 9,212) | 0.024 / 0.032 |

Each arm reads lowest in the frame it clusters in. The PS rule clusters on server
positions; fusion clusters on raycast positions. Coverage is flat (0.87 in both frames), so
the inventory cannot pick between the frames here. The 2.6 m and per-pano blocks show the
same pattern (raycast 0.172 / 0.085 and 0.160 / 0.098; server 0.098 / 0.166 and 0.098 /
0.169 for `ps @ 7.5 m` / `fusion_server`).

On all labels (`label_set: all`; the arms as the server would hold them, unmapped labels
included):

| arm | placement | frame | clusters | placed | covered r5 | split r5 | merge r5 |
|---|---|---|---:|---:|---:|---:|---:|
| deployed | server | - | 18,684 | 18,684 | 0.903 | 0.196 | 0.020 |
| ps @ 7.5 m | server | - | 17,451 | 17,451 | 0.902 | 0.138 | 0.019 |
| ps_citywide @ 7.5 m | server | - | 17,310 | 17,310 | 0.901 | 0.127 | 0.020 |
| fusion_server | server | auto | 18,484 | 18,484 | 0.900 | 0.201 | 0.039 |
| fusion_server+attach | server | auto | 16,171 | 16,171 | 0.897 | 0.120 | 0.041 |
| deployed | raycast | auto | 18,684 | 16,790 | 0.877 | 0.255 | 0.014 |
| ps @ 7.5 m | raycast | auto | 17,451 | 15,927 | 0.878 | 0.200 | 0.014 |
| fusion_server | raycast | auto | 18,484 | 14,674 | 0.874 | 0.092 | 0.024 |

The raycast rows place only clusters with a label the rebuilt run maps. `fusion_server`
leaves 3,810 of its clusters unplaced there, against 1,524 for `ps @ 7.5 m`, so the
all-label raycast ratio (0.46x) mixes the frame with which clusters survive placement. It is
not a test of Part 1's claim and is not quoted as one. The deployed partition is the most
fragmented arm in the raycast frame (0.255 against 0.200 for a fresh `ps @ 7.5 m`) and more
fragmented than a fresh `ps @ 7.5 m` in the server frame too (0.196 against 0.138), though
there `fusion_server` reads higher still (0.201); this is (a) seen from the inventory. Merge rises from the PS rule to
fusion in every block (the Bend trade-off). The split figures in
`docs/figures/vancouver-splits/` illustrate the server-frame rows.

#### Exploratory (rebuilt-run detections; never the deployed labels' fusion)

Offline synthesis at 0.55: 60,094 labels, `ps @ 7.5 m` 16,200 clusters against `fusion`
14,820 (2.6 m); at 0.30: 73,478 labels, 20,205 against 17,551. Inventory split at r = 5 m,
`auto` frame, both arms placed by raycast: 0.55 `ps @ 7.5 m` 0.174 vs `fusion` 0.077
(0.44x), 0.30 0.274 vs 0.126 (0.46x); 2.6 m 0.182 / 0.074 and 0.286 / 0.119; per-pano 0.169 /
0.088 and 0.268 / 0.136. Merge 0.021 to 0.033 (0.55, auto). Coverage flat (0.895-0.897 at
0.55, 0.926-0.928 at 0.30). This is Part 1's pattern on the run's own detections, in the
frame fusion clusters in. `fusion` holds fewer labels than `ps @ 7.5 m` (57,972 against
60,094 at 0.55), because labels the raycast cannot place leave fusion and stay in the PS
partition. It says what fusion would do on a fresh Vancouver run, not what it would do on
the deployed labels. `fusion` alone: auto 14,869 clusters (256 per 1,000 labels), 2.6 m
14,824, per-pano 15,196.

**Pooled driver.** `clustering_eval_pooled.py` is unchanged: it pools GT-based columns
(coverage, frag, precision), none of which exist for Vancouver.

#### Replication

All inputs are the frozen, untracked pulls named above plus `runs/vancouver/results.jsonl`
(sha256 `7fdf4005…9f28`), its depth index (per-pano) and the inventory. No command contacts a
Project Sidewalk server: every pull is passed by path, with no `--server`. `<SW>` is a
SidewalkWebpage checkout at `origin/develop` 0062ed0.

```bash
V=runs/vancouver; P=$V/ps_clustering_eval
LIVE="vancouver --no-gt --ai-user 51b0b927-3c8a-45b2-93de-bd878d1e5cf4 \
  --labels $V/provenance_gate/raw_labels.geojson --clusters $P/clusters.geojson \
  --streets $P/streets.geojson --offline-check --ps-script <SW>/scripts/label_clustering.py"
python scripts/eval_ps_clustering.py $LIVE                              # -> ps_clustering_eval/
python scripts/eval_ps_clustering.py $LIVE --camera-height-m auto       # -> ps_clustering_eval_auto/
python scripts/eval_ps_clustering.py $LIVE --camera-height-m per-pano   # -> ps_clustering_eval_per-pano/
python scripts/eval_ps_clustering.py vancouver --offline --no-gt --min-confidence 0.55 \
  --streets $P/streets.geojson                                          # -> ..._offline_t0.55/
python scripts/eval_ps_clustering.py vancouver --offline --no-gt --min-confidence 0.3 \
  --streets $P/streets.geojson                                          # -> ..._offline_t0.3/
python scripts/inventory_clustering.py score vancouver                  # -> inventory_clustering/
python scripts/split_figures.py --rebuild                               # -> docs/figures/vancouver-splits/
# the Richmond calibration, from its 2026-09-21 pulls
R=runs/richmond/ps_clustering_eval
python scripts/eval_ps_clustering.py richmond --labels $R/raw_labels.geojson \
  --clusters $R/clusters.geojson --streets $R/streets.geojson --offline-check \
  --ps-script <SW>/scripts/label_clustering.py
```

The three live Vancouver runs exit 1 on the documented 0.550 m placement label; their
reports are complete. Wall time is about 15 minutes each on a desktop CPU.

Where each number lives:

| number | file | where |
|---|---|---|
| (a) 0.765, 17,412 / 17,414 | `runs/vancouver/ps_clustering_eval/report.md` | Validation checks |
| (a) 276, 19.6% / 11.1 m, 154, 4,384 / 2,896 / 929, 0.759 | same | Deployed-partition diagnostics |
| (b) the table and the pre-registered reading | `runs/vancouver/ps_clustering_eval{,_auto,_per-pano}/arms.csv`, `report.md` | `near*`, `nearp*`, `per_1000_labels`; "Pre-registered reading" line |
| (b) Richmond calibration | `runs/richmond/ps_clustering_eval/arms.csv` | `nearp5`, `frag5_with_extra`, `frag3_ramps` |
| (c) anchor and size table | `runs/vancouver/ps_clustering_eval/report.md`, `validation_precision.csv` | Validation-based precision |
| (d) 0.733, 0.550 m | `runs/vancouver/ps_clustering_eval/report.md` | Offline server arm |
| inventory, both tables | `runs/vancouver/inventory_clustering/arms.csv`, `report.md` | `label_set` `mapped` / `all`, `scope: descriptive` |
| exploratory | `runs/vancouver/inventory_clustering/arms.csv` (`scope: exploratory`), `runs/vancouver/ps_clustering_eval_offline_t0.{55,3}/` | |
