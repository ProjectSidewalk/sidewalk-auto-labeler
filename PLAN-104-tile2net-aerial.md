# PLAN — labeler #104: sidewalks from aerial imagery (Tile2Net) projected into panos (2026-09-28, Fable 5.1)

Issue: https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/104 (read all of it; its four
questions are the deliverable). Parent direction: #47. A STUDY: no production wiring, no new dependency in
requirements.txt; Tile2Net runs in its own env on makelab2. Read CLAUDE.md first (fusion, eval_sites,
eval_ps_clustering, geo.ground_point_to_pano, the pre-registration convention used by #47/#52/#79/#106).

## Why this is the breakthrough bet
Every pano-side geometry lever has been measured and came back small or negative (per-pano height #44,
ground-plane normal #52, road-frame pose #42, depth as FP filter #47). Aerial polygons are an independent
modality with independent failure modes (trees hide from above; cars hide from the street). Two open
problems it can attack: cluster fragmentation (corner anchors) and off-surface false positives.

## Cities
Bend (Oregon statewide tiles; GSV run, GT, fused sites, city curb-ramp inventory from #79) and Paterson (New
Jersey statewide tiles; GSV run, GT, fused sites). Start with the benchmark panos' neighbourhoods (fast
iteration), then the full area.geojson of each run. Data lives read-only in D:\Git\sidewalk-auto-labeler\
runs\<city>\ (results.jsonl, sites.jsonl, depth/index.csv, inventory_oracle/) and D:\Git\RampNet\benchmark\
<city>\ (records.jsonl, verdicts.json). Do NOT edit anything under D:\Git\sidewalk-auto-labeler — work in your
own worktree; your script takes --run-root (default REPO_ROOT/runs) so it can read that repo's runs/ as data.

## Steps
1. **Tile2Net on makelab2** (A40). Own venv (`~/.local/bin/uv venv ~/tile2net-venv --python 3.12`, or what
   its README requires; install from the GitHub repo, BSD-3). Verify with its example, then run `generate`
   for a ~1 km² pilot around Bend's benchmark panos (bbox from records.jsonl), zoom 19/20, then the full Bend
   and Paterson areas (tmux, log). Outputs (polygons: sidewalk/crosswalk/road/footpath; network) to
   `/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/aerial/` (the archive root; never delete
   anything there). Record tile source, zoom, Tile2Net commit, runtime, tile count in a tracked
   `runs/<city>/aerial/tile2net.json` (+ sha256 of the polygon file). Polygons: untracked here (size); if a
   simplified sidewalk+crosswalk GeoJSON per city is < 25 MB, `get` it into the worktree's runs/<city>/aerial/
   untracked and note the sha256. Respect the tile servers (Tile2Net's defaults; no extra parallelism).
2. **Pre-register on #104 BEFORE any GT-joined number**, one comment, never edited, with the rules:
   - Q1 (mask quality): share of RampNet verdict-TRUE GT ramp positions (eval_sites' judged GT panos, tier
     0.55, world positions at the production `auto` height) within d of a sidewalk∪crosswalk polygon, d = 1,
     2, 3 m; the distance distribution for the rest; same for the Bend inventory points (#79's inventory, an
     image-free reference). Pre-state what "usable" means: ≥ 0.90 of inventory ramps within 2 m.
   - Q2 (projection error): for judged benchmark panos, project the nearest polygon boundary into the pano
     (geo.ground_point_to_pano, `auto` height and 2.6 m) and measure the pixel offset between the reviewer's
     box centre and the projected sidewalk edge nearest to it; also render an HTML gallery (20 panos/city,
     crops from the makelab2 native-res archive the way site_explorer.py does) for Jon's eyes. Quote pixels.
   - Q3 (anchors): corner anchors = points where a crosswalk polygon meets a sidewalk polygon (or the
     network's crossing endpoints). Arms: fuse_sites `auto` (baseline), `anchored` (sites snapped to the
     nearest anchor within r_a = 3 m when one exists, else unchanged), `merge-at-anchor` (sites sharing an
     anchor merged). Score with eval_ps_clustering's GT scorer (coverage, frag 5 m, dual, precision) on the
     same GT pool; rule: anchored must cut frag 5 m by ≥ 0.03 absolute in both cities with coverage ≥ −1 pt
     and dual ≥ −1 pt. Also score against the server-rule offline arm from #106 (its code is on
     origin/clustering-eval-106, PR #107; merge that branch into yours, do not modify its files).
   - Q4 (precision signal): among operational (0.30) detections on judged panos, share whose ground point is
     > 2 m from every walkable polygon, split by verdict (True/False). Rule as #47's: (i) ≥ 0.90 of True on
     surface; (ii) False − True off-surface share ≥ 0.15 with a bootstrap 95% CI excluding 0; else "not a
     filter". Tree/awning occlusion is the expected failure; report the occluded share if Tile2Net gives it.
   Everything at tier 0.55 for GT joins (BENCHMARK_CONFIDENCE), 0.30 only where named.
3. **Script**: one `scripts/aerial_sidewalks.py` (numpy + shapely + stdlib; matplotlib only in `figures`),
   subcommands `gt` (Q1), `project` (Q2 + gallery), `anchors` (Q3), `precision` (Q4), `verdict` (the
   pre-registered reading, committed before the run, like gsv_ground_plane.verdict), `figures`. Outputs to
   runs/<city>/aerial/ (CSVs + report.md tracked) and runs/_summary/aerial/. Small tests for the geometry
   helpers only (anchor extraction on a toy crosswalk+sidewalk pair; on-surface distance), importorskip
   shapely. Keep it lean.
4. **Run, then write** docs/aerial-sidewalk-study.md (method, tile provenance, the four answers, what it
   cannot say, Tile2Net citation: Hosseini et al., Mapping the walk, CEUS 2023), figures under
   docs/figures/aerial-sidewalks/, a CLAUDE.md command block, results comment on #104, PR against main.
   Attribution trailer on every comment/PR body: `🤖 Generated with [Claude Code](https://claude.com/
   claude-code) — <friendly name>, <exact model id>` using what YOUR session says; plan by Fable 5.1,
   claude-fable-5-1. Never a claude.ai/code/session_ URL anywhere.

## Constraints
- makelab2 via `& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 run '<cmd>'` (PowerShell tool, full path, single
  quotes; `put`/`get`; `-TimeoutSec`). Master already open. ONE call at a time, no parallel calls, no polling
  tighter than 5 minutes (rate-based IP blackhole). Long work in tmux with a log. Another agent is also using
  makelab2 for Vancouver (#56) — do not touch its runs/vancouver dirs or tmux sessions; check `nvidia-smi`
  before launching and leave ≥ 20 GB VRAM free.
- Local python: `C:\Users\jonf\anaconda3\envs\sidewalk-auto-labeler\python.exe` (conda not on PATH; pytest
  -q → branch on exit code). Never `git add` with globs under runs/. Bend is a TRAINING city for RampNet
  Stage 1 (say so wherever a Bend number is quoted). Gainesville untouched. GET only against PS servers.
- If Tile2Net cannot be made to run within ~2 hours of effort, stop, post what blocked it on #104, and do
  Q1 for the OSM footway baseline instead (position_check's Overpass pull; lines not polygons) so the
  session still yields a number.
