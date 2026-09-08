# Does Project Sidewalk's label clustering group multi-view AI curb-ramp labels correctly?

**Status:** protocol written 2026-09-05 before any result was computed; findings appended
below after the run. Companion tool: `scripts/eval_ps_clustering.py`. Server-side issue:
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
  `/v3/api/rawLabels` on 2026-09-05: 9,631 labels, 9,526 of them from the AI user and 105
  from three human users. Each AI label maps one-to-one to a stored detection in
  `runs/richmond/results.jsonl` by pano id and pixel position (verified: 9,526 of 9,526,
  zero collisions), so cluster membership can be scored against per-detection verdicts.
- **Deployed clusters.** `/v3/api/labelClusters?includeRawLabels=true`, same day: 2,156
  clusters covering 9,627 labels (4 labels unclustered).
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
| `ps_raycast @ t` | PS algorithm, same sweep | labeler raycast (2.6 m) | per region |
| `fusion` | ray-aware associator (`fuse_sites.py`) | labeler raycast (2.6 m) | whole city |

`ps_repro` exists only to prove the offline harness reproduces the server (membership
identity is the check). `ps_raycast` isolates placement from algorithm: same merge rule,
different positions. `fusion` is the reference the labeler already measured.

### Cluster placement for scoring

Every cluster is placed at the **mean of its members' labeler-raycast positions** (members
beyond the 25 m envelope are unplaceable and contribute nothing; a cluster with no placeable
member is counted but not placed). This puts every arm in the frame the GT is placed in and
makes the scores depend only on *who was grouped with whom*. The server's own centroid is
reported separately as a descriptive offset, not used for scoring. `fusion` is additionally
reported at its refit position (`fusion_refit`) to tie back to the published numbers.

### Metrics (match radius 5 m unless stated)

1. **Precision** (RQ1), membership-based and GT-incompleteness-safe, as `eval_sites.py`:
   over clusters with at least one judged member, TP if any member verdict is true or
   duplicate, FP if all decided verdicts are false, excluded if unsure-only.
2. **World recall** (RQ2a): pool GT ramps matched one-to-one to a cluster within 5 m, with
   the self-detected / other-view / unmatched decomposition.
3. **Fragmentation** (RQ2b): over matched GT ramps, the number of *additional* clusters
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
  world recall 0.941, dual ramps 23 / 4 / 3.

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

### Limitations

- The benchmark panos are de-clustered, so GT never says "these two labels from different
  panos are the same ramp". Fragmentation and dual-ramp separation are therefore measured
  in world space, and an "extra" cluster within 3 m could in principle be a real ramp hidden
  from the judged pano. The bias is the same for every arm.
- Placement uses the 2.6 m camera height everywhere; Richmond's rigs sit lower (about 1.7 to
  2.0 m), so raycast positions run long. Again the same for every arm, and it does not touch
  membership-based metrics.
- One city, one imagery source (Mapillary). GSV cities have different scatter.

## Findings (run 2026-09-05)

Full tables: `runs/richmond/ps_clustering_eval/report.md` (labeler frame, 2.6 m) and
`runs/richmond/ps_clustering_eval_h2.34/report.md` (server frame, see the deviation note).

### Validation

All three checks passed exactly: the verbatim script reproduces the server's partition
(2,156 of 2,156 clusters identical), the vectorized distance reproduces the script (2,156 of
2,156), and `fusion_refit` reproduces the published fusion numbers (precision 0.959, recall
0.941, dual ramps 23 / 4 / 3). No arm ever put two labels from one pano in one cluster.

### Deviation from the protocol

The protocol fixed the scoring frame at the labeler's 2.6 m raycast. Reading
[SidewalkWebpage#4819](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/4819) (in the
deployed release) showed the server's estimator is the same flat-ground cotangent at a
calibrated 2.341 m, blending into a bounded linear tail beyond about 12 m. Rescoring at that
height confirmed it: the server's per-label position and the labeler's raycast differ by
0.0 m at the median and 2.7 m at p90 (the tail). So a second run scores everything in the
server's own frame (`--camera-height-m 2.341219672825709`). Membership-only metrics
(precision, cluster counts) are identical across frames; the world-space ones (recall,
fragmentation, dual ramps) are reported for both. The camera-height constant itself is not
testable on fusion's sites, whose membership was selected at 2.6 m (a circular test that
was run and discarded).

### Headline numbers

Match radius 5 m. "frag" is the share of matched GT ramps with at least one extra cluster
within 3 m / 5 m; "dual" is same-pano ramp pairs under 5 m kept in separate clusters.

Server frame (2.341 m):

| arm | clusters | labels | precision | recall | frag 3 m / 5 m | dual kept | coherence > 5 m |
|---|---:|---:|---|---|---|---|---:|
| deployed (7.5 m, server positions) | 2,156 | 9,627 | 0.964 (238/9) | 0.946 | 0.17 / 0.48 | 25 of 32 | 3 |
| ps @ 12.5 m, server positions | 1,585 | 9,526 | 0.963 | 0.942 | 0.11 / 0.25 | 25 of 32 | 9 |
| ps @ 15 m, server positions | 1,487 | 9,526 | 0.963 | 0.938 | 0.11 / 0.21 | 25 of 32 | 11 |
| ps @ 7.5 m, placeable labels only | 1,887 | 8,098 | 0.959 | 0.946 | 0.13 / 0.41 | 25 of 32 | 4 |
| ps @ 7.5 m, pure cotangent positions | 1,896 | 8,098 | 0.959 | 0.942 | 0.09 / 0.30 | 23 of 32 | 1 |
| fusion (ray-aware) | 1,514 | 8,098 | 0.959 (211/9) | 0.938 | 0.08 / 0.17 | 26 of 32 | 12 |

Labeler frame (2.6 m):

| arm | clusters | labels | precision | recall | frag 3 m / 5 m | dual kept | coherence > 5 m |
|---|---:|---:|---|---|---|---|---:|
| deployed | 2,156 | 9,627 | 0.964 | 0.953 | 0.24 / 0.47 | 23 of 30 | 11 |
| ps @ 15 m, server positions | 1,487 | 9,526 | 0.963 | 0.941 | 0.13 / 0.26 | 22 of 30 | 18 |
| ps @ 7.5 m, placeable labels only | 1,887 | 8,098 | 0.959 | 0.949 | 0.20 / 0.44 | 23 of 30 | 14 |
| ps @ 7.5 m, raycast positions | 1,979 | 8,098 | 0.959 | 0.953 | 0.07 / 0.31 | 23 of 30 | 0 |
| ps @ 12.5 m, raycast positions | 1,488 | 8,098 | 0.959 | 0.941 | 0.06 / 0.19 | 23 of 30 | 13 |
| fusion (ray-aware) | 1,570 | 8,098 | 0.959 | 0.945 | 0.06 / 0.16 | 24 of 30 | 12 |

### RQ1, validity: not the problem

Of 247 deployed clusters that contain a judged label, 238 are real ramps. The 9 false clusters
are the same 9 false detections every arm carries, so they are detector errors, not
clustering errors. Precision is unchanged at every threshold from 2.5 m to 15 m.

### RQ2, one ramp one cluster: fragmentation is the whole shortfall

Coverage (recall 0.946 to 0.953) and dual-ramp separation (25 of 32 pairs, 23 of 30 in the
other frame) match the ray-aware reference. What the deployed clustering gets wrong is
splitting: 17 to 24 percent of matched ramps have a second cluster within 3 m and 47 to 48
percent within 5 m, against 6 to 8 percent and 16 to 17 percent for fusion. On the same
8,098 labels the server's rule makes 1,887 clusters where fusion makes 1,514 to 1,570, so
roughly one deployed cluster in five is a fragment of a ramp that already has one. 412 of
the 2,156 deployed clusters are singletons in a city where a ramp carries five labels.

The mechanism is same-ramp scatter. Over the 945 ramps seen from three or more panos, the
largest distance between two labels of the same ramp is 7.1 to 7.4 m at the median (server
and raycast positions alike), and 45 to 49 percent of these ramps exceed the 7.5 m cut.
Complete linkage keeps a group only if every pair is within the threshold, so half of the
multi-view ramps cannot survive it. The scatter is in the label positions (Mapillary's
structure-from-motion camera positions, mixed rig heights, and far-view range error), not
in the grouping rule.

### RQ3, attribution

- **Threshold.** In the server's own frame, widening the cut on the server's positions to
  12.5 to 15 m closes most of the gap: fragmentation 0.11 / 0.21 to 0.25 against fusion's
  0.08 / 0.17, cluster count 1,487 to 1,585 against 1,514, and precision, recall and
  dual-ramp separation unchanged (25 of 32 at every cut from 7.5 m up). Under the
  pre-registered bar, `ps @ 15 m` is within 3 points on precision and recall and within
  4 points on fragmentation, and one dual pair short of the mean-placed fusion arm (equal
  to the refit one). In the labeler's 2.6 m frame the same sweep stops halfway (0.13 / 0.26
  at 15 m) and coherence failures climb from 11 to 18, so how far a threshold alone gets
  depends on which placement model scores it. Region-wise clustering costs 34 clusters
  (1.6 percent) at region boundaries and nothing else.
- **Why dual ramps survive a 15 m cut.** With about five views per ramp, nearly every pair
  of adjacent ramps is seen together from at least one pano, the same-(user, pano)
  cannot-link fires, and complete linkage propagates it. The failure SidewalkWebpage#4706
  predicts, two ramps that no single observer saw together, becomes rare precisely because
  every per-view label is kept. This protection is a property of label density, not of the
  algorithm, and a human-labeled city with one label per ramp does not have it.
- **Placement.** At matched height, the server's estimator and a pure cotangent differ only
  in the bounded tail, and that tail costs some fragmentation at 5 m (0.41 against 0.30 at
  7.5 m) and nothing at 12.5 m. The larger effect in the 2.6 m frame (0.20 to 0.07 at 3 m,
  coherence failures 14 to 0) is mostly the frame agreeing with itself, and is not evidence
  that 2.6 m is a better height.
- **Merge criterion.** The ray-aware associator groups tighter than any isotropic cut at
  7.5 m on the same positions (1,514 to 1,570 clusters against 1,887 to 1,979), but a 12.5
  to 15 m cut on consistent positions comes within 3 to 4 points of it on every metric here.
  On this benchmark the criterion is second-order; its advantage should appear where the
  cannot-link does not cover pairs, which is sparse human labeling, and that is not what
  this data tests.

### Decision, per the pre-registered rules

Not "already good enough": fragmentation is 2 to 4 times the reference. In the server's own
frame the threshold is the lever; in the labeler's frame placement is. Both frames agree the
grouping rule is sound, the merge criterion is second-order here, and the remaining gap is
positional scatter, which per-rig camera heights
([RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158) step 2) attack directly.

### What to do with it

1. **Keep the per-view labels.** Precision and coverage of the deployed clusters are as good
   as the reference; nothing here argues for deduplicating at submission.
2. **Measure a wider CurbRamp cut for AI-dense cities before changing code.** 12.5 m on the
   server's positions removes about half the fragments on Richmond with no measured cost to
   precision, coverage or dual-ramp separation. The protection that makes it safe is view
   density, so it should not be applied to human-only cities without the same measurement.
   A second city, ideally GSV where per-pano depth gives true heights, is the next run; the
   tool takes a city name and the two API downloads.
3. **The ray-aware criterion stays the structural fix** for the sparse case the threshold
   cannot cover, as SidewalkWebpage#4706 proposes; this study neither needs it for Richmond
   nor argues against it.
4. **Reduce the scatter itself.** Half the multi-view ramps span more than 7.5 m under either
   placement. Per-rig heights and better camera poses shrink that for every arm at once.
