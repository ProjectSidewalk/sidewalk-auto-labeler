# HANDOFF: corner-level cluster-review GT (RampNet#224), one agent, both repos (2026-09-29)

Issue with the full plan: https://github.com/ProjectSidewalk/RampNet/issues/224 (read it
first; this note covers only what the issue doesn't). Filed 2026-09-29 by a Fable 5.1
subagent from the Vancouver #56 session. Jon wants ONE agent to do both halves, driven from
the sidewalk-auto-labeler checkout.

## Why this exists (one paragraph)
Label-clustering fragmentation (one real ramp split across two or more PS clusters) can
only be measured against city curb-ramp inventories, which exist for Bend, Gainesville and
Vancouver only. The GT-free near-cluster proxy (eval_ps_clustering.py) is broken: it
penalises correctly separating two ramps at one corner, and it fails the Richmond
calibration. RampNet's verdicts.json judges detections, not "same ramp?". #224 fills the
gap: reviewers correct seeded clusters corner by corner and produce a label -> ramp
assignment on a frozen label snapshot, plus ramp points for ramps no label covers. Every
arm clusters the same labels, so one GT scores them all, with no placement, radius or
camera height involved.

## Numbers to carry (Vancouver, auto frame; don't re-derive)
- Split rate at SERVER placement (every cluster placed; the fair frame): deployed 0.196,
  ps @ 7.5 m 0.138, fusion_server 0.199, fusion_server+attach 0.118.
  Fusion+attach vs ps: fixes 601 ramps, breaks 399, both split 724; deployed vs ps: 678
  stale splits.
- The raycast-frame figures quoted on #56 (0.255 / 0.200 / 0.091) are flattered by
  survivorship: `placed_clusters` drops clusters with no raycastable label, and fusion
  loses 3,757 against ps's 1,524. The #56 comment has NOT been corrected yet. Ask Jon
  whether to post the correction, pointing to docs/figures/vancouver-splits/README.md.
- Deployed clusters look stale: ps re-run reproduces them at 0.765 (Richmond 0.996); 154
  labels sit in two clusters. Not yet filed on SidewalkWebpage (Jon hasn't decided).

## Where things are
- auto-labeler Vancouver work: worktree `D:/Git/sal-vancouver`, branch
  `vancouver-56-scoring` = PR #118 (stacked; merge order #105, #107, #109, #108, #118,
  pending Jon's go). HEAD d86cbd4 added `scripts/split_figures.py` +
  `docs/figures/vancouver-splits/` (12 PNGs on Esri imagery, groups.csv, README).
- Vancouver run data is UNTRACKED and lives only in that worktree:
  `runs/vancouver/{results.jsonl, provenance_gate/raw_labels.geojson (frozen label pull,
  provenance in .source.json), ps_clustering_eval/clusters.geojson, inventory_oracle/
  inventory.geojson, sites*.jsonl}`, plus the partition cache
  `runs/vancouver/split_figures/state.pkl` (pickle of deployed / ps@7.5 / fusion+attach
  with server-position centroids and 5 m ramp assignments; regenerate with
  `split_figures.py --rebuild`). A new worktree must copy or point at these, never move them.
- Code to reuse: `scripts/inventory_clustering.py` (`inventory_metrics`,
  `SERVER_LABELS`, `score_server_arms`: the exact arm construction),
  `scripts/eval_ps_clustering.py` (`load_labels`, `label_to_detection`, `Cluster`,
  `ps_partition`, `server_panos`, `attach_unplaceable`), `scripts/site_explorer.py` (crop
  worker keyed (pano, x, y, fov, px); makelab2 over SSH or `--local-panos`),
  `scripts/split_figures.py` (Esri tile mosaic + ENU plan view; a starting point for the
  gallery's aerial panel).
- Pano pixels for Vancouver: the PS pano store on makelab2,
  `/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa` (sharded
  `<id[:2]>/<id>.jpg`). The run's py312 checkout there is
  `/homes/gws/jonf/sidewalk-auto-labeler-py312/`; read only, overwrite nothing.
- RampNet: `D:/Git/RampNet` (clean, on main). Precedents the issue builds on:
  `scripts/gt_gallery.py`, `scripts/box_gallery.py` (#116), `rampnet/validation.py`,
  `rampnet/tag_review.py`, `RUBRICS.md`, `benchmark/README.md`,
  `docs/adding_a_benchmark_city.md`. Leave `D:/Git/RampNet-review` alone (someone else's
  review checkout).

## Suggested branching
- auto-labeler: a new worktree + branch off `vancouver-56-scoring` (e.g.
  `D:/Git/sal-cluster-review`, branch `cluster-review-224`), so it stacks on #118 and has
  its code. Point it at sal-vancouver's run data by path.
- RampNet: branch `cluster-review-224` off main in `D:/Git/RampNet`.

## Order of work (matches the issue's milestones; stop at each checkpoint for Jon)
1. **Rubric + schemas.** Draft rubric v1 (RUBRICS.md section) and the exact
   `snapshot.json` / `corners.jsonl` / `assignments.json` schemas, plus the pre-registered
   scoring rules (assignment_metrics definitions, inter-rater agreement, pilot pass/fail).
   CHECKPOINT: show Jon, then post it as a comment on #224 only with his OK.
2. **Exporter (auto-labeler).** Corner sampler (OSM intersections typed + mid-block
   windows, ~15% no-label units, seeded), seed groups (deployed where AI labels are live,
   else ps @ 7.5 m; inter-rater subset seeded from fusion), crops through site_explorer's
   worker, writing a bundle under `../RampNet/benchmark/vancouver/cluster_review/`.
   Tests in the lean style (tests/ covers PS-facing contracts; keep it small).
3. **Gallery (RampNet).** `scripts/cluster_review_gallery.py` in the gt_gallery /
   box_gallery mould (self-contained HTML, localStorage autosave, Export, prefill), and
   `rampnet/cluster_review.py` (loader + validation) with tests.
4. **Scorer (auto-labeler).** `assignment_metrics` beside `inventory_metrics`, reading
   assignments.json as data (no RampNet import, like eval_sites.py reads verdicts.json).
5. **Pilot.** 30 Vancouver units, timed, double-rated. CHECKPOINT: Jon (and any other
   rater) does the reviewing; the agent never fills in GT itself.
6. Full pass and re-score: only after the pilot is read.

## Open decisions for Jon (ask, don't assume)
- Aerial imagery source and licence for the gallery (Esri World Imagery was used for the
  #56 figures with an attribution line; OK for a lab review tool? or a county orthophoto?).
- Whether to post the #56 headline correction, and whether to file the stale-clusters
  finding on SidewalkWebpage (SW#4706 thread).
- Merge of the PR stack (#105 -> #118), which the branch in "Suggested branching" stacks on.

## Guardrails
- GET-only against every PS server; nothing submitted.
- Frozen snapshot: use the provenance_gate raw_labels pull as-is (sha256 in its
  .source.json); never --refresh it for this work.
- Don't fabricate the reviewer-time numbers: the issue's ~1 min/unit is a guess the pilot
  measures.
