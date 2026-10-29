# Overnight dispatch 2026-10-28: status for Jon

Written by Fable 5.1 (claude-fable-5-1), session sidewalk-auto-labeler-de. Plan:
`PLAN-overnight-2026-10-28.md` (same branch). Nothing was merged, no issue was commented on,
no Project Sidewalk server was written to. Four PRs are open; CI green on all four as of 2026-10-29 ~00:30 PT. A trial merge of #173 + #170 + #171 had no conflicts and passed the suite (1,338 tests).

## PRs, in suggested merge order

1. **#172** `pytest-9-approx` → main. One test fix: CI's unpinned pytest moved to 9.2, whose
   `approx()` rejects a dict of tuples (`test_height_gap`). Every branch's CI was red until
   this; #170, #171 and #173 each carry the same commit so they are green now. Merge first.
2. **#173** `panoramax-politeness-161` → main (#161). Limiter (1 picture/s, 1 metadata/s,
   5 tiles/s, Retry-After pauses the whole kind), batched `/search?ids=` metadata (100 per
   request, 174k → ~1,750 for Lyon), 4 workers max, `--panoramax-asset {hd,sd}` bound in the
   manifest (old runs = hd), archive step through the limiter and always HD, study + doc.
   **Study: HD stays the default.** 600 Lyon panos, pre-registered rule: SD reproduces 0.629
   [0.597, 0.661] of HD's 0.30-tier detections (precision 0.847); every rig with ≥ 30 panos
   fails. SD is 18x lighter. Review (Opus 5.5): no blockers, all study numbers replicated independently, fixes applied (head 840f5e1).
3. **#170** `multitype-records-168` → main (#168 items 1, 3, 6). Registry, `--types`,
   `label_types` binding, writer rule; CurbRamp-only runs byte-identical (hash pin). Review: no blockers, fixes applied (head b6bc182).
4. **#171** `multitype-submit-168` → `multitype-records-168` (#168 item 2, stacked). One POST
   per (line, type), per-type sidecars `<file>.type-<T>.submitted`, record `types[T]`;
   single-type files byte-identical (golden fixtures captured from main before the change). Review found the fixtures were gitignored (CI red); fixed, head 36cd3cd, green.
   Merge after #170 (retarget to main first if you merge #170 with branch deletion).

## Your decisions

- **Post the #161 reply.** `DRAFT-161-reply.md` (this branch) is written as you, numbers
  from the run records and the study; no tool mentions. It is three weeks overdue. All [CHECK]s are resolved; the one judgement call left in it is wording.
- The sd/hd default is settled by the pre-registered rule (hd); nothing to decide unless you
  disagree with the rule.
- #168 items 4–5 (per-type FuseParams, ranker review-order export) are deliberately not
  done; nothing needs them until a second detector exists.
- Follow-ups NOT filed (your call): archive the bytes during a run so the native-res archive
  never re-downloads from Panoramax; a Mapillary limiter; decode/border guards per type in
  send_to_ps before a second spec lands.

## Housekeeping

- Worktrees added tonight: `D:\Git\labeler-wt\pmx161`, `mt168a`, `mt168b`, `pytest9`; study
  data (1.8 GB HD + SD pixels, uncommitted) in `D:\Git\labeler-wt\pmx161-data\`.
- Phase 0 census scratchpad copied out of Temp to `D:\Git\sal-scratch-keep\phase0-census-2026-10-28\`.
- makelab2: one SSH master opened 21:29 PT, read-only use only (the auto-mode classifier
  denies remote writes there for Claude).
- The main checkout `D:\Git\sidewalk-auto-labeler` is ~188 commits behind origin/main on
  `clustering-eval-106`, with uncommitted runs/ files nobody claims. Left untouched.

## Merged (2026-10-29 ~05:45 PT, by Jon's instruction)

All four PRs merged with merge commits in the suggested order: #172 (72534e9), #173
(6c86628), #170 (b1fd38f), #171 retargeted to main and merged (dbbdbac, now main's head).
CI on dbbdbac passed. The four feature branches were deleted on origin. Worktrees
`D:\Git\labeler-wt\{pmx161,mt168a,mt168b,pytest9}` still exist (pmx161 now sits on `main`);
the study pixels in `pmx161-data` are untouched. Still open: posting the #161 reply.
