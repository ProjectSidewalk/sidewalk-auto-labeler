# PLAN: #169 NoCurbRamp test-server pilot, residential corners only

Written 2026-11-11 by Claude Fable 5.2 (claude-fable-5-2) for one Opus 5.5 implementer.
Issue: https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/169
PR (draft): https://github.com/ProjectSidewalk/sidewalk-auto-labeler/pull/177
Doc of record: `docs/corner-labels-169.md` on branch `corner-labels-169`
(worktree `D:/Git/labeler-wt/corner169`, HEAD c8e495e, clean, pushed).

## Goal

Get the pilot to the point where Jon can clear **one exact send**: a file with a sha256, an
endpoint, the commands, the expected counts, a verification query and a rollback draft.
**The implementer sends nothing.** Jon's clearance is per operation and covers test servers too.

## Where things stand

- The pre-registered rule passed on the split-wide table: 96 absent, 1 present, 3 can't tell;
  0.990 [0.944, 0.998] (amendment for n = 97).
- Pilot build: `runs/vancouver/corner_labels169_split/na_discount_sw/`, 3,070 proposals:
  residential 1,984, arterial 1,052, signalised 34. `proposals.jsonl` sha256 `1562e076…`.
  **That dir has no exported `results.jsonl` yet** and `export` has no stratum filter.
- Why residential only: 22 of the 34 arterial units in the rated sample carry Jon's note that
  it is not a safe place to cross (1 residential unit does). 14 units, 11 of them residential,
  say there is nothing to cross to on the far side; that stays in scope and is a thing to watch.
- Store guard: `check_store_built` refuses these files on `corner-labels-169`. Draft PR #179
  (`store-guard-label-type`, base main) fixes it and is **not** in the branch.
- Checked 2026-11-11: `https://sidewalk-vancouver-test.cs.washington.edu/v3/api/cities` returned
  **503**; `sidewalk-vancouver` returned 200. Whether a Vancouver test stage exists is unknown.
- The local SidewalkWebpage clone (`D:/Git/ProjectSidewalk_Webpage`) is at 2026-09-30 and
  stale. Read current server code with `gh api` / `gh browse`-style reads, not from that clone,
  and do not pull or edit it.

## Decisions already made here (reversible; change them on the clearance sheet if Jon says so)

| # | Decision | Why |
|---|---|---|
| 1 | Population = residential stratum of the split `na_discount_sw` build (1,984 proposals). Arterial and signalised are out, not deferred to a later look within this pilot. | Jon's crossing notes |
| 2 | Recommended send = a **seeded subset of 200 labels** in a fixed order (seed 169, recorded). The full 1,984 is the alternative; build both files so Jon can pick. | PS is insert-only for AI labels (SW#5382); the pilot measures a sample either way |
| 3 | Rating = sequential on the seeded order with a reserve: looks at 40 / 60 / 100 decided labels, can't-tell skipped and absorbed by the reserve, schedule and correction fixed before any send (the #183 design). This amends the issue's "100-label sample" and is written as a declared amendment for Jon to approve. | Jon's rating burden (2026-11-11) |
| 4 | Work on a new branch `corner-pilot-169` = `corner-labels-169` + a merge of `store-guard-label-type`. Push it. No PR, no merge of #177 / #179 / #176. | merge order is Jon's |
| 5 | No `rank` step: no validator-ai #69 CSV exists. The seeded order is the review order. | |

## Work items

### A. Code (branch `corner-pilot-169`, new worktree `D:/Git/labeler-wt/pilot169`)

1. `corner_labels.py export --stratum <name>` (repeatable) and `--only-corners <csv>` (the
   corner keys of a pilot draw). Both recorded in `export.json` with the counts kept and
   dropped. Refuse a key that is not in the proposals.
2. `corner_labels.py sample --stratum <name> --reserve K`: one fixed seeded order,
   `sample.json` records seed, n, reserve, strata, look schedule. Keep it minimal; the full
   #183 scorer is not in scope unless it is small. If `score` is not extended, say so.
3. Tests in `tests/test_corner_labels.py`: stratum filter, only-corners, unknown key refused,
   order stable under the seed, end-to-end build -> sample -> export -> `send_to_ps` dry run.
   Run the whole suite with the conda env's python.exe (see memory `running-tests-conda-env`).
4. Docs: a "Test-server pilot" section in `docs/corner-labels-169.md`, the commands in
   `docs/commands.md`. CLAUDE.md stays under 50k (a test enforces it).

### B. Build the pilot files (local only)

Out dir `runs/vancouver/corner_labels169_split/na_discount_sw/pilot_residential/`:

- `pilot.csv` + `pilot.json`: the seeded order over the 1,984 residential proposals (first 200
  = recommended send; the first 100 of those = the rated sample with its reserve).
- `results.pilot200.jsonl` and `results.residential.jsonl`, each with its `export.json`.
- `send_to_ps.py --dry-run` on both; then call the pre-send guards directly, as
  `dry_run_summary.txt` did before (`check_model_provenance`, `check_detection_decode`,
  `check_detection_border`, `check_store_built`, and whatever else a real send runs for a GSV
  NoCurbRamp file: position check, live positions, resume state). Record what each prints.
- Report: labels, panos, labels per pano, range to the corner p10/median/p90, share bounded
  by a phantom leg, share still wide, how many of the 66 residential corners Jon already
  rated in `corner_gallery169_split` fall in the 200 (keep them; say how many).
- Files over 2 MB stay untracked; `export.json` carries sha256 / lines / bytes.

### C. Server facts, read-only (each one is a line on the clearance sheet)

1. **Does a Vancouver test stage exist?** Start from SidewalkWebpage `docs/deployment-and-stages.md`
   (via `gh api`), then the public URL. If makelab1 is needed: `wsl-ssh.ps1 makelab1 check`
   first, one batched `run`, **zero retries on a timeout** (global CLAUDE.md rule), and only
   read-only psql (`PGOPTIONS='-c default_transaction_read_only=on'`, port 6432 for test).
   Is there a `sidewalk_vancouver` schema in `sidewalk_test`, and how old is it against prod?
2. **Does `/ai/submitLabelsOnPano` accept `NoCurbRamp` on current develop?** Read
   `AiController.submitAiLabel` and `AiService` (label-type parsing, `getAiValidateMissionId`,
   anything keyed to CurbRamp). Does the AI validator then vote on the new label, and is
   that wanted in a pilot?
3. **Are the pilot's panos in the test DB?** They are store-built, so they exist on prod; the
   test copy may be older. Count the 200 (and the 1,984 file's panos) present / missing.
4. **Baseline.** Existing NoCurbRamp labels on the test stage, by the AI user and by people,
   so the pilot's labels can be told apart afterwards. Any ref to a label carries the city
   (`vancouver:12345`).
5. **How a lead rates on the live site.** Read `docs/server-agree-check.md`,
   `scripts/server_agree_check.py` and the Richmond/Laurens live-validation notes. Write the
   exact steps; if the site has no way to serve a fixed seeded list, build a small local HTML
   page of the sample in order that links each label on the test site (local HTML, never a
   Claude Artifact).

If C1 says there is no Vancouver test stage, stop the server checks there, finish A and B, and
put "stand up a Vancouver test stage (Mikey, via a SidewalkWebpage issue)" at the top of the
sheet as the blocker. Do not propose prod as a substitute; the issue forbids it.

### D. The clearance sheet: `docs/corner-labels-169-pilot.md`

One page Jon can approve line by line:

- file, sha256, lines, labels, panos; endpoint URL; `--min-confidence`;
- the exact commands in order: dry run, a `--limit 3` smoke send with what to look at on the
  site, then the rest; which sidecar and `.submission.json` appear;
- every guard's verdict, and which branch the send must run from;
- expected rows after the send, the read-only SELECT that verifies them;
- rollback: AI labels are insert-only, so a **draft** soft-delete SQL for Jon, keyed on the AI
  `user_id`, label type and time window, with the count it should touch;
- the rating protocol and its amendment text, with the stop rule's arithmetic;
- open decisions for Jon, each with a recommendation: 200 vs 1,984;
  whether the AI validator should run on these; who the lead is; whether #179 merges first.
- a draft of the comment for #169 (file only, not posted).

### E. Wrap-up

- Commit in small commits on `corner-pilot-169`, push the branch.
- `D:/Git/sal-working-notes/STATUS-169-test-pilot.md`: what was done, what was found, exact
  commands, what Jon has to decide. Commit it on `working-notes` (local).
- List the over-2 MB files that should go to makelab2 under
  `/projects/makeabilitylab/sidewalk-auto-labeler/runs/vancouver/` with the command; Jon runs it.

## Hard limits

- **No POST to any PS server. No non-dry-run `send_to_ps.py`.** GETs to public APIs are fine.
- No writes on makelab1 / makelab2. No Hyak.
- No GitHub writes other than `git push` of `corner-pilot-169`: no PR, no comments, no merges,
  no issue edits.
- Do not touch `corner-labels-169`'s existing builds, samples, verdicts or bundles.
- Do not delete untracked files anywhere.
- Anything that could become public states practical constraints only.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
