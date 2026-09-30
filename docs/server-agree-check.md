# Server-side read: validations of the AI's labels, and one auditor vs the AI

**Status:** measured 2026-09-30 against frozen pulls of three city servers. Tool:
`scripts/server_agree_check.py`. Full numbers, with each pull's url, fetch time, sha256 and
feature count, in `runs/<city>/server_agree/report.md`. Nothing was submitted anywhere.

**What this is and is not.** `agree_rate.py` (`docs/agree-rate-gainesville.md`) compares a
run's raw detections with crowd labels in the pano frame and needs the run directory. This
tool reads only what the server serves: the AI labels as deployed, the human votes on them,
the server's own clusters, and the streets table. It is a quick read, and its main use is a
city where the AI labels are live and one person has been validating them and auditing
streets. It was written for [RampNet#158](https://github.com/ProjectSidewalk/RampNet/issues/158),
which asks whether RampNet needs a rig-native retrain for open imagery.

**One caveat up front: this is one rater.** The account that cast 1,111 of Laurens's 1,112
human votes, 999 of Richmond's 1,069 and 1,572 of Vancouver's 2,911 is the same person, who
is also the auditor in section 2. Nothing here says how much validators disagree with each
other.

## 1. Human validations of the AI's CurbRamp labels

Per label, the majority of human Agree vs Disagree votes; PS's own AI validator is excluded.
Precision = agreed / (agreed + disagreed), Wilson 95%.

| city | imagery | AI labels | human-validated | agreed | disagreed | precision |
|---|---|---:|---:|---:|---:|---:|
| Vancouver | GSV | 64,814 | 2,911 (4%) | 2,713 | 81 | 0.971 [0.964, 0.977] |
| Richmond | Mapillary, 11000 px rig | 12,962 | 1,069 (8%) | 1,004 | 49 | 0.953 [0.939, 0.965] |
| Laurens | Mapillary, GoPro Max | 1,575 | 1,112 (71%) | 1,006 | 94 | 0.915 [0.897, 0.930] |

Almost every validated label has exactly one vote. The Laurens figure is after the roof-rack
false positives were masked (`detectors.on_camera_rig`); the 50 judged labels that were all
false in September are not in this pull.

## 2. Laurens: the auditor's own labels vs the AI's

The auditor labels streets in the Explore interface, independently of the AI labels. At the
pull, 84 of Laurens's 169 streets carried a label of theirs and **44 were completed**; the
audit is in progress, so everything below is provisional. Their labels: 230 CurbRamp, 54
NoCurbRamp. Comparison is in world space against the server's clusters (7.5 m single
linkage), because the AI labels a ramp from every pano that sees it and a person labels it
once.

**Human → AI**, all 230 CurbRamp labels:

| test | share |
|---|---:|
| sits in a server cluster that also holds an AI label | 166/230 = 0.722 |
| … or an AI cluster within 5 m | 191/230 = 0.830 |
| … or within 7.5 m | 210/230 = 0.913 [0.870, 0.943] |
| … or within 10 m | 224/230 = 0.974 |

Six ramps have no AI cluster within 10 m (`runs/laurens/server_agree/misses.csv`):
`laurens:1408`, `2727`, `2728`, `2774`, `2775`, `3011`. Those are the candidate misses of
the deployment on this rig, and they are the recall question the #158 mined-label gallery
cannot answer.

Pano frame, for reference: 178 of the 230 labels are on a pano the AI labelled at all, and
in 92 of those an AI label sits within RampNet's match radius in that same pano. The gap
between 92/178 and 210/230 is fusion: the AI finds the ramp from another pano.

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
benchmark). At the deployment level, after fusion, the AI already has a cluster within
7.5 m of 91% of the ramps the auditor placed, at 92% validated precision. A retrain's gain
on this rig would show up as fewer views needed per ramp and fewer single-view false
positives (the 1-view row above), not as many new ramps. The six misses are where to look
before deciding.

## Re-running

```
python scripts/server_agree_check.py laurens --human <user_id>
python scripts/server_agree_check.py richmond
python scripts/server_agree_check.py vancouver
```

The four pulls are cached in `runs/<city>/server_agree/` and reused; `--refresh` takes a
new snapshot, and the servers move (Laurens changed between two pulls an hour apart on
2026-09-30, while the auditor was working). Offline test: `tests/test_server_agree_check.py`.
