# Server-side read: validations of the AI's labels, and one auditor vs the AI

**Status:** measured 2026-09-30 against frozen pulls of three city servers; restated
2026-10-04 after the [PR #119 review](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/119#pullrequestreview-5407428975)
(one-to-one headline with a chance floor, per-cluster precision, confidence tiers). Tool:
`scripts/server_agree_check.py`. Full numbers, with each pull's url, fetch time, sha256 and
feature count, in `runs/<city>/server_agree/report.md`. Nothing was submitted anywhere.

**What this is and is not.** `agree_rate.py` (`docs/agree-rate-gainesville.md`) compares a
run's raw detections with crowd labels in the pano frame and needs the run directory. This
tool reads only what the server serves: the AI labels as deployed, the human votes on them,
the server's own clusters, and the streets table. It is a quick read, and its main use is a
city where the AI labels are live and one person has been validating them and auditing
streets. It was written for [RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158),
which asks whether RampNet needs a rig-native retrain for open imagery.

**Two caveats up front.**

- **Mostly one rater.** One account cast 1,111 of Laurens's 1,112 human votes, 999 of
  Richmond's 1,069 and 1,572 of Vancouver's 2,939, and it is also the auditor in section 2.
  Vancouver has 18 other validators; Laurens and Richmond have effectively none. Nothing
  here says how much validators disagree with each other.
- **Not a random sample.** Labels reach validators through PS's validation queue, so
  section 1 is the precision of the labels that were shown. Laurens is 71% validated, so
  this matters little there; Richmond (8%) and Vancouver (4%) depend on the queue being
  representative.

## 1. Human validations of the AI's CurbRamp labels

Per label, the majority of human Agree vs Disagree votes; PS's own AI validator is excluded.
Precision = agreed / (agreed + disagreed), Wilson 95%.

| city | imagery | AI labels | human-validated | per label | per cluster |
|---|---|---:|---:|---:|---:|
| Vancouver | GSV | 64,814 | 2,911 (4%) | 0.971 [0.964, 0.977] (2,713 / 2,794) | 0.971 [0.964, 0.977] (2,487 / 2,561) |
| Richmond | Mapillary, 11000 px rig | 12,962 | 1,069 (8%) | 0.953 [0.939, 0.965] (1,004 / 1,053) | 0.957 [0.941, 0.969] (763 / 797) |
| Laurens | Mapillary, GoPro Max | 1,575 | 1,112 (71%) | 0.915 [0.897, 0.930] (1,006 / 1,100) | 0.903 [0.875, 0.925] (491 / 544) |

**The units differ, so quote each with its unit.** Per label counts a ramp seen from k panos
k times. Per cluster takes the majority of each cluster's AI labels' statuses, which is
closer to "per ramp".

**The server's clusters as served are not a fair unit for precision.** The server leaves
labels already marked incorrect out of its clustering. In all three pulls, every AI label
outside a cluster is a disagreed one: 84 in Laurens, 49 in Richmond, 81 in Vancouver. So a
rejected ramp mostly has no cluster, and per-cluster precision over the served clusters
reads high: 0.981 in Laurens (10 disagreed clusters, validated after the last re-cluster),
and 1.000 in Richmond and Vancouver. The per-cluster column above puts those labels back
first, by 7.5 m single linkage to any AI label (the server's threshold, without its
per-region split), and that is the figure to quote.

**By confidence tier** (each label joined to its stored detection on pano id and the pixel
send_to_ps.py sent; label level). Both Mapillary cities hold the 0.30-0.55 band (#20) on top
of the 0.55 core:

| city | ≥ 0.55 | 0.30-0.55 band |
|---|---:|---:|
| Laurens | 0.974 [0.957, 0.984] (560 / 575; 708 live) | 0.850 [0.816, 0.878] (446 / 525; 867 live) |
| Richmond | 0.978 [0.966, 0.985] (920 / 941; 9,526 live) | 0.750 [0.662, 0.821] (84 / 112; 3,436 live) |

Every live label joined to exactly one tier. Vancouver's tier is not split: its run file is
not on this machine, and the tier it went live at is not recorded here. The pooled Laurens
0.915 is therefore a mix of a 0.97 core and a 0.85 band, at 45 / 55 by label count. Almost
every validated label has exactly one vote. The Laurens figures are after the roof-rack
false positives were masked (`detectors.on_camera_rig`); the 50 judged labels that were all
false in September are not in this pull.

## 2. Laurens: the auditor's own labels vs the AI

The auditor labels streets in the Explore interface, independently of the AI labels. The
auditor is chosen by rule (`--human auto`: the human user with the most CurbRamp labels in
the snapshot). The rule is not a post-hoc choice: that account holds 230 of the 234 human
CurbRamp labels on the server. At the pull, 84 of Laurens's 169 streets carried a label of
theirs and **44 were completed**; the audit is in progress, so everything below is
provisional. Their labels: 230 CurbRamp, 54 NoCurbRamp.

**Human → AI**, all 230 CurbRamp labels, in world space. Three matchers, because they
disagree, each beside a chance floor (every auditor label moved 25 m in a random direction,
seed 31, as `agree_rate.chance_floor`):

| reading at 7.5 m | share | chance |
|---|---:|---:|
| **one-to-one vs AI clusters** (headline) | **157/230 = 0.683** [0.620, 0.739] | 0.248 |
| one-to-one vs raw AI labels | 195/230 = 0.848 [0.796, 0.889] | 0.296 |
| any AI cluster (holds the label, or within r) | 210/230 = 0.913 [0.870, 0.943] | 0.283 |

At 5 / 10 m the headline reads 0.583 / 0.778. Over seeds 1-20 the 7.5 m chance floor
ranges 0.22-0.29 (one-to-one vs clusters) and 0.24-0.33 (any).

- **One-to-one vs clusters** credits each server cluster to at most one auditor label. It
  is strict where the server's 7.5 m single linkage merged two ramps at a corner into one
  cluster, which happens at corners where two ramps sit a few metres apart.
- **One-to-one vs raw AI labels** is lenient the other way, because one ramp has several AI
  views.
- **Any** lets one cluster credit several auditor labels (paired corner ramps). It is
  coverage, not agreement, and it is the figure first reported to #158.

The truth sits between the two one-to-one rows.

Six of the auditor's labels, at four locations, have no AI cluster within 10 m
(`runs/laurens/server_agree/misses.csv`): `laurens:1408`, `laurens:2727` and `laurens:2728`
(2 m apart), `laurens:2774` and `laurens:2775` (one pano, 2 m apart), and `laurens:3011`.
They are the candidate misses of the deployment on this rig, and they are the recall
question the #158 mined-label gallery cannot answer.

Pano frame, for reference: 178 of the 230 labels are on a pano the AI labelled at all. In
90 of those, an AI label on the same pano matches one-to-one, strictly within RampNet's
radius (agree_rate's matcher); 92 have any AI label within it. The gap to the world-frame
rows is the AI finding the ramp from another pano.

**AI → human**, the 224 AI clusters on the 44 completed streets, confirmed = holds the
auditor's label or has their CurbRamp label within 7.5 m: **132/224 = 0.589**. By AI views
in the cluster: 1 view 30/78 (0.385), 2 views 46/79 (0.582), 3+ views 56/67 (0.836).

**That 0.589 is a lower bound on coverage, not a false-positive rate.** The auditor's own
votes show it. Of the 537 AI labels on completed streets, 105 carry their Agree vote and
have no label of theirs within 7.5 m: accepted when validating, not placed when auditing. An
audit is one pass from one pano, and a corner ramp is attributed to one street edge but
labelled from another. The Disagree votes behave as expected: 25 of 28 have no label of
theirs nearby.

## What it says for #158

The case for a rig-native retrain rests on per-pano recall (0.39 on the laurens_mapillary
benchmark). At the deployment level, after fusion, the AI matches **0.68-0.85 of the ramps
the auditor placed, one-to-one at 7.5 m** (vs clusters / vs raw labels; any-cluster coverage
0.91; chance ~0.25-0.30). Validated precision on this rig is **0.903 per cluster, 0.915 per
label**, and it splits by tier: 0.97 for the 0.55 core and 0.85 for the 0.30-0.55 band. A
retrain's gain on this rig would show up as fewer views needed per ramp, fewer single-view
false positives (the 1-view row above) and a cleaner band, more than as many new ramps. The
six misses are where to look before deciding.

The number first posted on #158 was the any-cluster 0.913 and a label-level 0.92
precision, with no chance floor; the restatement above replaces it.

## Re-running

```
python scripts/server_agree_check.py laurens --human auto --results runs/laurens/results.raw.jsonl
python scripts/server_agree_check.py richmond \
    --results runs/richmond/results.band.jsonl --results runs/richmond/results.posfix3seq.raw.jsonl
python scripts/server_agree_check.py vancouver
```

The four pulls are cached in `runs/<city>/server_agree/` and reused; `--refresh` takes a
new snapshot, and the servers move (Laurens changed between two pulls an hour apart on
2026-09-30, while the auditor was working). **Laurens's pulls are tracked in git**, so its
numbers re-score from a clone. Human user ids in them are cut to 8 characters
(`--redact-users`). No number depends on the full id, and each sidecar keeps the served
sha256. Richmond (22 MB) and Vancouver (84 MB) are not tracked. The tier split also needs
the run files (`--results`), which are not tracked; the report records each one's sha256.
Offline test: `tests/test_server_agree_check.py`.
