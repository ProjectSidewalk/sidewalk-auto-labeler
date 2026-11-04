# STATUS: labeler PR review sweep (2026-11-03)

Plan: C:\Users\jonf\.claude-team\plans\modular-beaming-dragonfly.md (approved by Jon 2026-11-03).
Shared agent rules: session scratchpad `SWEEP-RULES.md`.
Jon's decisions: open a PR for cluster-review-224; include #114 last; #134 keeps only the Richmond 1.7 m eval
(revert the Lyon stub); post review/fix/final comments on each PR.

Phases: 0 = merge main in (no rebase) · 1 = deep review · 2 = fixes · 3 = fresh final review (READY / NOT READY).

| PR | branch | worktree | P0 | P1 review | P2 fixes | P3 final | head |
|---|---|---|---|---|---|---|---|
| #158 | pin-scikit-image | labeler-wt/pinskimage | running | | | | |
| #157 | position-check-systemexit | labeler-wt/pcexit | running | | | | |
| #152 | pano-x-seam-wrap | labeler-wt/seamx | running | | | | |
| #150 | census-email-mask-149 | labeler-wt/mask149 | running | | | | |
| #153 | panoramax-404-skips | labeler-wt/skips404 | running | done earlier | done earlier | | |
| #134 | run-artifacts-2026-10-04 | labeler-wt/artifacts1004 | running | | | | |
| #155 | offgrid-misses-151 | labeler-wt/offgrid151 | running | done earlier | done earlier | | |
| #160 | paterson-offgrid-151 | labeler-wt/paterson151 | running | | | | |
| #154 | bayonne-densify-147 | labeler-wt/densify147 | running | done earlier | done earlier | | |
| #162 | bayonne-5m-analysis-147 | labeler-wt/bayonne5m | running | | | | |
| #156 | lyon-thinning-148 | labeler-wt/lyon148 | running | done earlier | done earlier | | |
| #163 | lyon-densify-159 | labeler-wt/lyon5m | running | | | | |
| new | cluster-review-224 | sal-cluster-review | running | | | | |
| #132 | corner-inventory-238 | labeler-wt/corner238 | running | done earlier | done earlier | | |
| #135 | corner-posdetect-241 | labeler-wt/posdetect241 | running | done earlier | done earlier | | |
| #141 | corner-gallery-243 | labeler-wt/gallery243 | running | 3 new commits only | | | |
| #114 | analysis/mined-precision-step4 | labeler-wt/mined158s4 | running | | | | |

## Merge order (Jon runs; retarget the child BEFORE merging its base)
1. #158, #157, #152, #150, #153, #134
2. #155 → `gh pr edit 160 --base main` → merge #160
3. #154 → #162
4. #156 → #163
5. cluster-review-224 PR → #132 → #135 → #141
6. #114 (`gh pr ready 114` first)

## Log
- 2026-11-03: Phase 0 launched as two agents (A: small PRs + three study stacks; B: corner stack + new PR + #114).
- P0B done: new PR #174 (cluster-review-224, a3f189e); #132 a86b647, #135 1435b55, #141 6c020fa, #114 fc2e3ff; all CI pass.
  Merge caught a silent semantic conflict in inventory_clustering.server_arms (positional arg) — fixed in the merge.
  Open for Jon (maybe): re-seed #224 bundle after #118 change; #141 items.jsonl rebuild identity (reviewer checking).
- P1 launched: #174, #141 (delta since 1ed3af9), #114. #132/#135 final reviews wait for #174 fixes to propagate.
- P0A done: #158 e2c7833, #157 91b2845, #152 ba13737, #150 6e5e720, #153 74a0e40, #134 c95e731, #155 3c23810,
  #160 ca01ec8, #154 403e981, #162 a42f2cc, #156 6913e4c, #163 651160e; all CI pass, no conflicts.
  #134: Lyon-stub revert resolved as EMPTY (fe64d59) — a literal revert would delete main's real runs/lyon files.
  Cross-PR: #155 CLAUDE.md says skimage unpinned vs #158 pin; #150 vs #156 two producer masks; #152 five analysis
  scripts use round(x*w) without % w; #157 also swallows position_check.py:335 stale-filter SystemExit.
- P1 launched: #152, #157+#158, #150+#134, #160, #162, #163.
- Pending P3 (final review only): #153, #155, #154, #156; #132, #135 after #174 fixes.
- #114 P1 done: needs fixes, 6 Should-fix (stale ratings text — RampNet#223 DID rate: 0.843/0.855/0.831; tier
  0.55-vs-0.30 wording; 28 duplicate targets; replication block; rig status; test gaps). mine_city reproduces
  byte-identically at the merged head with f01 inputs. P2 fixer launched.
- #153 P3 final review launched.
- P1 results: #174 needs fixes (8 S; S3 re-seed + S4 amendment + S5 RampNet PR = RampNet side); #141 needs fixes
  (Jon's 50 ratings VALID: items.jsonl rebuilds byte-identical); #157 + #158 ready w/ 1 S each; #152 needs fixes
  (ps_pixel helper across 5 scripts; 0 of 1.83M stored detections hit the seam, no live impact); #163 needs fixes
  (docs/rate-limit/manifest); #150 + #134 needs fixes; #160 needs fixes (overclaims near-tie; tie-order + GPU
  dependent); #162 needs fixes (sequence-cluster CI). #153 FINAL: READY at 74a0e40.
- P2 done: #114 → 3b71d6a; #174 (labeler side) → 75b907e. Frozen-input archive for #174 staged at
  D:/Git/sal-scratch-keep/p2-174-cluster-review-frozen-inputs/ (Jon runs the 2 wsl-ssh lines in its install script).
- Jon 2026-11-03: include a RampNet PR for cluster-review-224 (yes). Re-seed: Jon unsure whether v2 rating started —
  no assignments.json export found in Downloads or repos; definitive check = gallery localStorage.
- P2 launched: #141, #152, #157+#150+#134, #155(wording)+#160, #163, #162; RampNet #224 branch merge + PR + Amendment 4
  draft (worktree D:/Git/RampNet-wt/cr224-sweep). P3 launched: #114, #158.
- P3 FINAL: #114 READY at 3b71d6a (still draft; `gh pr ready 114` at merge time); #158 READY at e2c7833.
  Follow-up to file later: stamp scikit-image version into manifest runs[] (makelab2/Hyak venvs not re-resolved).
- P2 #163 → 6cad11f (all fixed). P3 launched: #163, #154.
- #156 final review DEFERRED until #150 merges: #150's S1 changes the mask (sub vs fullmatch), so #156's two copies
  would diverge; after batch 1 merges, #156 merges main and imports run_census.mask_producer, then final review.
- P2 #152 → 9c0c401 (ps_pixel helper; 440k records byte-identical). P3 #152 launched.
- P2 #157 → 1f306a0, #150 → 808c162, #134 → f0bad0d (dir renamed _h1.70, scorer 106.4). P3 launched for all three.
  Jon decisions logged: #150 mask on PS submission path (copyright/source_metadata) — licence attribution call;
  email in main history 55f4cc0 (rewrite?); main checkout untracked runs/richmond/ps_clustering_eval_h1.7 superseded.
- P2 #141 → fd1377a (Jon's verdicts/items/images hash-identical; live_imagery__jonf.json sidecar added).
- Propagating #174 (75b907e) → #132 → #135 → #141; then P3 for all three.
- RampNet PR #280 opened (cluster-review-224, e079b57; main merged 26e801e; worktree D:/Git/RampNet-wt/cr224-sweep).
  Amendment 4 draft (2 alternatives, seed decision pending Jon) in session scratchpad amendment4-draft.md.
  P1 review of #280 launched (worktree D:/Git/RampNet-wt/r280).
- Corner stack propagated: #132 325c4b6, #135 bf47b60, #141 a683732 (CI pass; corner_inventory/posdetect outputs
  rebuild byte-identical; Jon's bundle unchanged). P3 launched: #132+#135, #141.
- P3 FINAL READY: #157 1f306a0, #150 808c162, #134 f0bad0d. Batch 1 now waits only on #152's final review.
  Out-of-scope notes: Bayonne census.md (9d40873) no longer regenerates byte-identically (rig-detections table
  added since); eval reports don't record which label_clustering.py --ps-script ran.
- P3 FINAL READY: #152 9c0c401 (525/525 hashed payloads identical over 1.2M detections; mutation-tested).
- BATCH 1 READY, handed to Jon: #158 e2c7833, #157 1f306a0, #152 9c0c401, #150 808c162, #153 74a0e40, #134 f0bad0d —
  all CI green, MERGEABLE, no file overlaps between them.
- P3 #141: NOT READY at a683732 — SF1 stale page can silently drop frozen blind reads on export. Round-2 fixer
  launched (blind_regressions guard in score main vs git HEAD + prefilled_from_sha256). Jon's 50 ratings intact.
- P2 #155 → df55364 (skimage wording), #160 → 44ff5fc (tie split, backend subcommand, wording). P3 launched for both.
- RampNet #280 P1: needs fixes, BLOCKER B1 (pilot gate pairwise >=0.90 can't discriminate; pre-fill accepter scores
  0.963) + S1-S9. Jon 2026-11-03: "Amend all" (Amendment 5) and "Nothing rated: re-seed" (Amendment 4 alt A).
- P3: #163 READY 6cad11f; #155 READY df55364; #160 NOT READY (wording F1/F2); #154 NOT READY (partial-run guard hole,
  10 m commands read 5 m file, runtimes pre-#173); #132 + #135 NOT READY (replicability doc gaps; #135 false
  "byte-identical to 660ccd5"). #141 round-2 fix → a16f9a3 (stale-page guard).
- Launched: #224 two-repo fixer (#174 + #280: re-seed, Amendments 4+5, B1/S1-S9); #132+#135 fixer; #160+#154 fixer;
  #141 final review round 2. Still running from before: #162 fixer.
- P3 #141 READY at a16f9a3 (guard catches every blind-read loss tried, no false alarms; Jon's files intact).
  Will need one more propagation merge after #132/#135 fixes land (then a quick re-check).
- P2 round 2: #132 → 94223d4 (figures subcommand, takeaways, table), #135 → 526d593 (re-anchored to 028c9c7, fig3).
  Final review round 2 launched for both.
- P3 round 2: #132 READY 94223d4, #135 READY 526d593. Small wording fix to fold into the next propagation: #132
  takeaway 3 / PR body "1,453 of 4,429 units have no pano within 25 m" → "1,453 are unobservable" (1,478 have none).
  PLAN: once the #224 fixer pushes #174, propagate #174 → #132 (+ that wording) → #135 → #141 in one pass, re-check.
- #224 fixed: #174 → 8ec49bc, RampNet #280 → e869be2 (re-seeded bundle d55c71c8…; Amendments 4 (alt A) + 5 appended;
  B1 gate = same-ramp pair F1 on 15 different-seed pilot units, pre-fill rater 0.782 fails). Consequence for Jon:
  first pilot pass shows NO inventory (all pilot units are double-rated). Final reviews launched for #280 + #174;
  corner-stack propagation (+#132 wording) launched.
- P2 #162 → d5047a7 (sequence-cluster CIs: Bayonne now "not distinguishable" from Richmond GoPro Max; cross-ray claim
  weakened). Final review waits for #154 fixer → then merge base into #162 and final-review #154 + #162 together.
- P3 #174 READY at 8ec49bc (CI green; score report byte-identical on regen; mutation-tested refusal).
  Jon-only: run the archive install (D:/Git/sal-scratch-keep/p2-174-cluster-review-frozen-inputs), merge RampNet#280
  with a MERGE commit (keeps e869be2 + 39155d8 reachable — #174 permalinks).
- Corner stack propagated again: #132 7367f8b (wording + Fig 1), #135 1613070 (Fig 3 CIs), #141 679cafc (merge, Jon's
  files unchanged); CI green. Delta re-check launched for all three.
- RampNet #280 final: NOT READY (SF1 role-b inventory guard untested; nits: 14 not 15 gated units, pooled leg untested,
  rubric procedure, <7-day rule). B1 replay confirmed (seed-accepter 0.782/0.830 fail; near-perfect 0.990 pass;
  random 0.201 fail). Round-2 fixer launched (erratum appended, not edited in place).
- Re-check READY: #132 7367f8b, #135 1613070, #141 679cafc. Whole corner stack (#174 8ec49bc → #141) merges clean vs
  main 2e3f238. Batch 1 NOT yet merged by Jon (main still 2e3f238). After batch 1 merges: merge main into corner
  stack (#152/#134 touch CLAUDE.md + docs/ps-clustering-eval.md like #174), re-run CI, then hand corner batch.
- P2: #160 → 7e3131d (wording + tie_env_warning), #154 → c7b2679 (guard on every partial run, 10 m prefix recipe,
  pacing note; full 10 m data regen byte-identical). #154 and #162 both added tests/test_panoramax_bayonne_figures.py →
  merge agent resolving #154 → #162. Re-check of #160 launched. Then final review #154 + #162 together.
- #160 READY at 7e3131d. Off-grid stack (#155 df55364 → #160 7e3131d) fully READY.
- #154 merged into #162 → b9cb8c4 (one combined test file; guard covers `cuts`; CI green). Final review of #154 + #162
  launched.
- RampNet #280 round 2 → da10928 (SF1 + pooled-leg tests; Amendment 5a appended: 15 gated / 14 defined, <7-day rule;
  RUBRICS §6 procedure). Re-check launched. #174 permalinks stay at e869be2 (resolve; repoint to RampNet main after
  #280 merges — follow-up, avoids another corner-stack propagation).
- #154 + #162 final: both NOT READY on ONE doc defect (step 8 `skips --run-root $R10` crashes: recipe lacks
  already_processed.txt; fix = head -n 28631 + 3 retried ids → exactly 110 skips.csv ids). Past the 2-round cap but
  doc-only with a verified remedy → applied (told Jon), then delta re-check. #162 nits (CLAUDE.md runtime ~1.5-2 h).
- RampNet #280 READY at da10928 (CI green). Merge with a MERGE commit. Follow-up after it merges: repoint #174's
  RampNet permalinks (e869be2) to RampNet main.
- Step-8 fix: #154 → 9ad0e4b, #162 → 5de2b88 (offline-verified: 110 ids set-equal to skips.csv). Delta re-check launched.
- Re-check READY: #154 9ad0e4b, #162 5de2b88.

## STATE AT END OF SWEEP (2026-11-03) — every head verified = its READY sha, CI green, MERGEABLE
READY: #158 e2c7833 · #157 1f306a0 · #152 9c0c401 · #150 808c162 · #153 74a0e40 · #134 f0bad0d · #155 df55364 ·
#160 7e3131d · #154 9ad0e4b · #162 5de2b88 · #163 6cad11f · #174 8ec49bc · #132 7367f8b · #135 1613070 ·
#141 679cafc · #114 3b71d6a (draft) · RampNet#280 da10928.
NOT DONE: #156 (6913e4c) — after #150 merges: merge main, import run_census.mask_producer (drop its 2 copies), final review.
Merge simulation (local, D:/Git/labeler-review/merge-sim): all merge clean in order EXCEPT #154/#162 conflict in
docs/panoramax-bayonne.md once #153 is on main → after batch 1, merge main into #154 → #162, resolve, CI, re-check.
Follow-ups: repoint #174 RampNet permalinks to main after #280 merges; stamp skimage version in manifest runs[];
#150 PS-path masking decision + email in main history (Jon); install #174 archive on makelab2 (Jon); cleanup of
D:/Git/labeler-review/* scratch worktrees + RampNet-wt/{r280,r280-final,r280-rc,r174f-rampnet,cr224-sweep} (Jon's OK).

## 2026-11-04 — batch 1 merged (Claude, on Jon's explicit permission)
- Merged with `--merge --delete-branch --match-head-commit` at the READY heads: #158 #157 #152 #150 #153 #134 -> main 85f7e9e; RampNet#280 (da10928, merge commit).
- Tidy: removed all D:/Git/labeler-review/* scratch worktrees + the folder, RampNet-wt/{r280,r280-final,r280-rc,r174f-rampnet,cr224-sweep}, and the 6 merged PRs' labeler-wt worktrees + local branches. D:/Git/RampNet main checkout is still on the merged cluster-review-224 branch (left alone; may be shared with the peer session).
- Post-batch merges, pytest run on each, pushed: #154 bcaade9 (docs/panoramax-bayonne.md conflict resolved, both sides kept), #162 5a89c1c, #155 4ead074, #160 92388f8, #156 974d9af + 13e2d4e (import run_census.mask_producer), #163 88e41c5, #174 964fdfd, #132 d25bc3c, #135 10d1018, #141 c2d0b5d, #114 c91b2b2.
- Running: #156 final review; delta semantic-conflict check of the other 10 branches.
- Follow-up: CLAUDE.md says a Panoramax image 404 is a cached skip; code (sources/panoramax.py, since 2026-10-05) treats it as retryable failure. Fix the CLAUDE.md line.

## 2026-11-04 — batch 2 ready (awaiting Jon's go to merge)
- Delta semantic check of all post-batch merges: no code conflict; stale #150-era doc text fixed on #154 (3a03c61), #162 (5d47bcb, conflict resolved), #156 (ef7f652 + PR body), #163 (21562d3).
- #156 final review round 1 NOT READY (stale docs only) -> round 2 READY at ef7f652 (issuecomment-6643308051). Delta checks OK on #154/#162/#163.
- READY heads, all CI green + MERGEABLE: #155 4ead074, #160 92388f8, #154 3a03c61, #162 5d47bcb, #156 ef7f652, #163 21562d3, #174 964fdfd, #132 d25bc3c, #135 10d1018, #141 c2d0b5d, #114 c91b2b2 (draft).
- Nits left: "Python's csv writer emits CRLF" in docs/panoramax-lyon.md (pre-existing on main); #156 body "two lines in CLAUDE.md" (three); CLAUDE.md Panoramax-404 line stale.
