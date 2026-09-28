# PLAN — labeler #56: Vancouver, WA clustering eval step 2 (2026-09-28, Fable 5.1 planner)

Issue: https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/56
Parent: SidewalkWebpage#4706 step 2; sibling of #106 (PR #107). Read CLAUDE.md ("A RUN REBUILT FROM
THE PANO STORE", "BEYOND RICHMOND (#106)", "POST-SUBMISSION COVERAGE CHECK") and
docs/ps-clustering-eval.md ("Step 2" runbook) before touching anything.

## State (measured 2026-09-28 22:26Z)
- Step 3 detection FINISHED on makelab2 (`~/vancouver_run.log`: `== step 3 exit 0 2026-09-28T21:23:56Z`;
  28,830 written, 0 skipped, 0 failed, 1.30 panos/s). Run dir is in the Python-3.12 checkout
  `/homes/gws/jonf/sidewalk-auto-labeler-py312/runs/vancouver/` (NOT `~/git/sidewalk-auto-labeler`,
  NOT the old Py3.9 checkouts — leave those untouched). Store:
  `/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa` (sharded), 29,181 ids
  (28,881 labeled panos' ids + 300 seeded unlabeled, seed 56).
- Locally, runs/vancouver holds only area.geojson, manifest.json (0 runs), inventory_oracle/inventory.json.
- Server: https://sidewalk-vancouver.cs.washington.edu — 64,814 AI CurbRamp labels on 28,881 panos,
  18,813 deployed clusters, 2,632 AI labels with human validations (2,553 agree / 79 disagree). GET only.
- City curb-ramp inventory (16,960 pts): ArcGIS service STILL "not started" (probed 22:30Z today).
- No RampNet GT for Vancouver. Portland (same metro) was in RampNet Stage-1 training: caveat, not a blocker.

## Goal
Score the server's deployed Vancouver partition and the #106 offline arms on an AI-dense GSV city with
per-pano depth heights, publish the numbers on #56 and in docs/ps-clustering-eval.md ("Step 2"), and leave a
benchmark bundle ready for a RampNet GT session. Everything GET-only, nothing submitted, nothing on makelab2
deleted or overwritten.

## Steps (in order; each has a stop condition)
1. **Bring the run home.** sha256 the remote results.jsonl, manifest.json, store_selection.json,
   store_sampled_ids.txt, store_ids.txt, store_skipped.jsonl, already_processed.txt; `get` them into the
   worktree's runs/vancouver/; verify sha256 both ends (never size). Line count must be 28,830 + the 351
   pre-existing = what manifest/history says; explain any difference. Do NOT copy store_metadata/ (29k files)
   — tar it remotely to `runs/vancouver/store_metadata.tar.gz` and `get` that if < 200 MB, else leave it.
2. **Provenance gate (Arm S).** `provenance_gate.py vancouver --server ... --fetch-only` (fresh labels pull),
   then `provenance_gate.py vancouver`. Read report.md. PASS → continue. UNDETERMINED/STOP → post the report
   summary on #56 and stop the scoring steps (5-6); still do steps 3, 4 and 7. Commit store_selection.json,
   store_sampled_ids.txt, provenance_gate/report.md (+ CSVs the runbook says are tracked).
3. **Depth from the store.** On makelab2, in the py312 checkout:
   `harvest_depth.py runs/vancouver --from-store <store> --check-store-frame 5` → depth/index.csv, store.json.
   Then `--from-store --verify`. `get` index.csv, store.json, unavailable.txt, no_depth.txt home. Report the
   measured-height share and the frame check result. (pano-tools' depth phase for Vancouver was expected done
   ~09-29; if many are `unavailable`, say so and continue.)
4. **Fuse.** `fuse_sites.py runs/vancouver` (auto = per-rig GSV) and `--camera-height-m 2.6` and `per-pano`
   (to --out files). Record sites_meta.json camera_heights/year table: Vancouver 2025-09 submission panos are
   mostly which rig year? This is the first AI-dense GSV city, so the rig split matters.
5. **Pre-register on #56 BEFORE scoring** (one comment, then never edited): the metrics below and which are
   confirmatory vs exploratory. GT-free metrics only (no verdicts exist):
   (a) verbatim reproduction: SW label_clustering.py (read-only copy via
       `git -C D:/Git/ProjectSidewalk_Webpage show origin/develop:scripts/label_clustering.py`) re-run on the
       server's positions per region reproduces the deployed partition (partition_agreement; Richmond 0.982);
   (b) fragmentation proxy without GT: per arm, share of clusters with another cluster of the same arm within
       5 m / 7.5 m / 12.5 m, and clusters per 1,000 labels — arms: deployed, ps@7.5 (server positions),
       ps@t sweep, fusion(auto), fusion(2.6), fusion(per-pano), fusion_server, +attach; the #106 offline
       synthesis at tier 0.55 (what went live) and 0.30 (what would ship today);
   (c) validation-based precision: the 2,632 human-validated AI labels → per-cluster "any member validated
       false" rate by cluster size and by arm; the label-level agree rate 2,553/2,632 = 0.970 is the anchor;
   (d) deployed-vs-#106-offline partition agreement on Vancouver (the gate #106 measured at 0.982/0.945 on
       Richmond/Laurens, here on a GSV city with 6.7x the labels).
   State the expected direction from #106 (fusion ~half the fragmentation of the 7.5 m rule) and that Vancouver
   is confirmatory for Part 1's fragmentation claim, exploratory for everything else. No GT → no coverage,
   no verdict rule; say so.
6. **Score.** If eval_ps_clustering.py needs GT to run at all, add the smallest possible `--no-gt` path (skip
   coverage/frag-vs-GT, keep partition agreement, cluster counts, near-pair rates, size buckets, validation
   join) rather than a second scorer; keep the #106 code untouched otherwise (PR #107 is being fixed in the
   main checkout right now — do NOT edit D:\Git\sidewalk-auto-labeler; work only in your worktree and merge
   `origin/clustering-eval-106` into your branch when you need its code). Outputs → runs/vancouver/
   ps_clustering_eval*/ (report.md + CSVs tracked; pulled geojson untracked, sha256 + fetch time in report).
7. **Benchmark bundle for a GT session** (so #56 can get real coverage numbers): `export_benchmark.py
   runs/vancouver/results.jsonl --bundle <worktree>/benchmark/vancouver --sample 100 --empty-sample 25
   --records-only`, then copy the sampled ids' JPEGs from the store on makelab2 into the bundle's panos/ (a
   remote script; native res already), pull the bundle back (or leave it on makelab2 under
   /projects/makeabilitylab/sidewalk-auto-labeler/runs/vancouver/benchmark/ and say where), re-run without
   --records-only to verify index.csv. Do NOT commit pixels. Note the Portland-training caveat in the bundle
   README line.
8. **Inventory:** probe the ArcGIS URL once; spend ≤ 20 min looking for the same layer elsewhere (City of
   Vancouver open-data hub / ArcGIS Online item search "Vancouver curb ramp"). If found: record url + sha256,
   fill inventory_oracle.py's vancouver entry (status ok, fields), `fetch vancouver`, then
   `inventory_clustering.py score vancouver` under the #106 pre-registered rule as a THIRD city (say on #56
   before scoring that it is confirmatory of #106 Part 2). If not found: record the negative in inventory.json.
9. **Write up**: docs/ps-clustering-eval.md "Step 2: Vancouver" section (numbers from CSVs, never typed from
   memory), CLAUDE.md command block if a flag was added, results comment on #56 with the tables, PR against
   `main` (base main unless you merged clustering-eval-106 code — then base `clustering-eval-106` and say so
   in the body). Attribution trailer on every comment/PR body: `🤖 Generated with [Claude Code]
   (https://claude.com/claude-code) — <friendly name>, <exact model id>` using what YOUR session says; plan by
   Fable 5.1, claude-fable-5-1. Never a claude.ai/code/session_ URL anywhere.

## Constraints
- makelab2 via `& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 run '<cmd>'` (PowerShell tool, full path; single
  quotes so nothing expands locally; `put`/`get` for files; `-TimeoutSec`). The control master is already
  open. ONE call at a time, never parallel calls, never a polling loop tighter than 5 minutes — the makelab
  hosts blackhole an IP that opens connections too fast. Long remote work: `tmux new-session -d -s
  <name> "<cmd> > ~/<log> 2>&1"` then check the log later.
- Remote python: `/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python`, cwd that checkout. `git
  pull` there only if a script you need is missing; run dirs are untracked and survive a pull.
- Local python: `C:\Users\jonf\anaconda3\envs\sidewalk-auto-labeler\python.exe` (conda not on PATH; pytest
  -q → branch on exit code). Never `git add` with globs under runs/. Key labels on (city, label_id).
- Gainesville is untouched. Nothing is POSTed anywhere. No SidewalkWebpage edits (read via `git show` only).
