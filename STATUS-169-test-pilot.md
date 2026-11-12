# STATUS: #169 NoCurbRamp test-server pilot (residential), 2026-11-11

Plan: `PLAN-169-test-pilot.md`. Implementer: Claude Opus 5.5 (claude-opus-5-5).
**Nothing was sent to any server.** Read-only GETs and one read-only makelab1 probe only.

## Branch

`corner-pilot-169` (worktree `D:/Git/labeler-wt/pilot169`), pushed to origin; HEAD `c26b22a`.
Its base is `corner-labels-169` (`c8e495e`), plus a merge of `store-guard-label-type`
(`c0ce3e9`, draft PR #179). The merge was clean. No PR was opened. Full suite: 1603 passed,
3 skipped.

Commits:
- `d13bd80` merge of #179
- `6b8a28a` `export --stratum/--only-corners/--only-first`; `sample --stratum/--reserve/--send/--looks`
  (one full seeded permutation; stop rule in the .json); `pilot-score`; tests
- `b97eef2` the pilot files, `dry_run_summary.txt`, `scripts/pilot_guards.py`,
  `scripts/corner_pilot_stats.py`
- `f18e07f` `pilot-links` (server label ids -> ordered Gallery review-list URL)
- `c26b22a` clearance sheet `docs/corner-labels-169-pilot.md`, doc section, commands,
  `.gitattributes` LF pin for the pilot dir

## Files (`runs/vancouver/corner_labels169_split/na_discount_sw/pilot_residential/`)

- **`results.pilot200.jsonl`** (committed): sha256 `d38c9f3b1fba530bac9d6b6ecfcacf9403ebe5b55f116bf18b5a35a7035da0c6`;
  197 lines/panos, 200 labels, 506,047 B.
- **`results.residential.jsonl`** (untracked, 4.7 MB): sha256 `a4dce3575e852ff9c4389ff094fbfd156c946f6bd003bcca905a14900ba6aa0c`;
  1,717 lines/panos, 1,984 labels.
- **`pilot.csv`**: sha256 `157ff268…`, seed 169, rate 100 / reserve 20 / send 200.
- **Labels.** `model_id` is `corner-rule@30168fddeb6e`, the commit of the split build.
- **Checks.** The dry runs give 0 errors. Every guard passes against the test endpoint:
  `check_store_built` passes through the type-aware guard of #179, and `check_live_positions`
  is blind to the 2025-09 campaign.

## Server facts

- **C1. No running Vancouver test stage.**
  - `.sidewalk-vancouver-test` is dot-disabled (dated 2025-09-19) and not running. The URL
    returns 503.
  - The test DB (6432) has a `sidewalk_vancouver` schema: 64,814 labels, all AI CurbRamp, the
    newest from 2025-09-18, and no crowd labels.
  - Test `sidewalk_login` user ids differ from prod for people (jonfroehlich is
    `2d8335ef…` on test).
  - Juneau-test was re-enabled 2026-10-29, so this is doable.
- **C2. NoCurbRamp is accepted** on develop `428341d`.
  - The pano upsert overwrites lat/lng/heading/pitch/roll (COALESCE) and keeps the stored
    width/height.
  - The AI validator never votes on AI-created labels of any type.
  - A 409 is returned for a pano already holding a live AI label of the same type.
- **C3. Pano presence in the test DB: NOT measured.**
  - The probe's pano dump failed: my SQL had `ORDER BY 2` with one select column.
  - Per the plan (C1 = no stage → stop server checks), it was not re-run.
  - Query to re-run is in the sheet's spirit: `pano_data` where `pano_id` is in
    `panos.pilot200.txt`. The id list is too long for one `wsl-ssh run` command (32K command
    line), so it needs a different shape.
- **C4. Baseline.**
  - Test vancouver: 0 NoCurbRamp labels.
  - Prod vancouver: 34 live non-tutorial human, 13 tutorial, 2 deleted, 0 by the AI.
  - Prod NoCurbRamp validations: human 2 Agree / 2 Disagree / 1 Unsure; AI validator 2 / 1 / 2.
- **C5. Rating route.** Gallery `?labelIds=` review list, in order, max 500 (SW#5444). There is
  also `/label/<id>`. No local HTML was needed; `pilot-links` builds the URL.

## Decisions for Jon (sheet section 8)

1. Re-enable `sidewalk-vancouver-test` via Mikey. Refresh its schema from prod? Recommended yes.
2. 200 vs 1,984: recommended 200.
3. Sequential rating amendment (looks 40/60/100, Bonferroni z 2.394, PASS at ≤ 0/2/6 wrong,
   FAIL at 7 wrong, reserve 20): recommended approve.
4. Sidewalk-layer terms: Jon's call.
5. The lead: Jon unless an independent second rater is available.
6. The 2007 GSV label at position 200 (no store image): recommended keep.
7. Merge #179 and then #177 before sending: recommended.
8. AI validator: no decision needed.

## Deviations from the plan

- **Export record location.** Export writes its record beside a non-default output as
  `<stem>.export.json`, not as `<dir>/export.json`. Otherwise two exports of one build would
  overwrite each other.
- **The sample is a full permutation of 1,984.** The old `draw_sample` is not prefix-stable.
- **Extra tools:** `pilot-links`, `scripts/pilot_guards.py` and `scripts/corner_pilot_stats.py`.
- **C3 not measured** (above). C4 came from the same single probe, read before stopping.
- **The rated measure is label precision** (Agree/Disagree on the live site), not corner-absence
  precision; the sheet says so.
- **Two full dry-run payload dumps** were written in the pilot dir and left untracked:
  `dry_run.pilot200.full.txt` (268 KB) and `dry_run.residential.full.txt` (2.4 MB).

## Over-2 MB files for makelab2 (Jon runs; Claude does not write there)

```powershell
$R='/projects/makeabilitylab/sidewalk-auto-labeler/runs/vancouver/corner_labels169_split/na_discount_sw/pilot_residential'
& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 run "mkdir -p $R"
& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 put 'D:\Git\labeler-wt\pilot169\runs\vancouver\corner_labels169_split\na_discount_sw\pilot_residential\results.residential.jsonl' "$R/results.residential.jsonl"
& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 put 'D:\Git\labeler-wt\pilot169\runs\vancouver\corner_labels169_split\na_discount_sw\pilot_residential\dry_run.residential.full.txt' "$R/dry_run.residential.full.txt"
& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 run "cd $R && sha256sum results.residential.jsonl"   # a4dce357...0aa0c
```

`proposals.jsonl` (`1562e076…`) is already archived (`corner_labels169_split.archive.sha256`).

## Update 2026-11-11 evening: placement fix after the smoke send (Opus 5.5)

**Context, from the parent session.** Jon started vancouver-test (v11.20.0, schema not
refreshed). He smoke-sent positions 1-3 of the first file (`vancouver-test:89735`-`89737`), and
all three sat a couple of metres inland. The remaining 197 of that file are on hold for good.

**Branch `corner-pilot-169`**, pushed; HEAD `fc1aafe`. Full suite: 1607 passed, 3 skipped.
- `bc40622`: `build --label-point site-median` (the default) moves only the label point along
  the bisector. Distance: the unit's median centre-to-site distance (sites from
  `corners[].site_ids` in the merged base + extra-run sites), else the stratum's median, less
  `--label-inset-m` 0.8, clamped to [stratum p10, 12 m]. `--label-point corner` is the old
  behaviour.
- `d24c4d6`: `corner_range_fallback`. The first site-median build dropped 2 corners
  (`vancouver:art:n3559864516#0`, `vancouver:res:n47211243#2`): the chosen panos sit 24.7 m
  from the corner, so the moved label fell past 25 m. Those labels now keep the corner point.
- `4bc0762`: `--skip-first` / `--exclude-panos` for export, pilot-links and pilot-score;
  `scripts/corner_label_shift.py`.
- `e8a2aa0`: the build `na_discount_sw_sitemed` (at `d24c4d6`) and the new file. The smoke
  send's `.submission.json` is committed.
- `fc1aafe`: the sheet rewritten with a short "what Jon has to decide / do" top.

**Verified by running:**
- **Same corner set.** `na_discount_sw_sitemed` has the same 3,070 keys and chosen panos;
  `counts.csv` and `unlabeled.csv` are byte-identical.
- **Shift.** Residential labels moved p10 / median / p90 0.47 / 2.50 / 4.09 m, with 71% from
  the unit's own sites. The smoke three would move 3.60, 2.48 and 2.00 m.
- **Short ranges.** 40 proposals now have a range under 3 m; 4 of them are in the new file
  (positions 91, 144, 152, 157).
- **New file.** `na_discount_sw_sitemed/pilot_residential/results.pilot200.jsonl`: sha256
  `567b88632fa83b21516e37ca8f8a06edf9252fe6d6f555ef99f75cdebbd0b446`, 197 lines/panos, 200
  labels, positions 4-203. No corner sits on a smoke pano.
- **Dry run and guards.** The dry run is clean and every guard passes. `check_live_positions`
  cannot see the smoke record next door. Read directly, that record holds 3 live panos and
  none is in the new file.
- **Hazard.** The old file's resume state passes, so a re-run of it would continue at line 4
  with the old placement. The sheet says never to resume it.

**Deviations:**
- `corner_range_fallback` was added because the first rebuild did differ, by 2 corners. The
  instruction was to stop if the sets differ; I made the smallest change that keeps the
  corner set fixed and recorded it.
- The rated order now starts at position 4. That is declared in the sheet.
