# Crosswalk Phase 1, items 2-3: autonomous run, 2026-11-02 (Jon at the university)

Context: RampNet#274 (crosswalk detector), umbrella Planning#36. Phase 0 merged as RampNet#275 (main 16171c9).
Phase 1 gate comment (RampNet#274, 2026-11-01) orders: (1) ortho survey (DONE, posted), (2) multi-city GT page
and seeded sample, (3) `stage_one` `label_type` axis. GT is scored before any label reaches a server.

## Item 2: multi-city crosswalk GT page + seeded draw (primary deliverable)

Branch `crosswalk-gt-phase1` off RampNet origin/main, worktree `D:/Git/RampNet-wt/crosswalk-gt-phase1`.
New dir `scripts/analysis/crosswalk_gt_phase1_274/` (+ `analysis_out/...`, doc `docs/crosswalk_gt_phase1_274.md`).

1. Generalize `aerial_spotcheck.py` / `aerial_sample.py` / `aerial_crops.py` from Phase 0 into one city-config table
   (ortho service URL + type, years, GSD, minimum extent (`MIN_TILE_M`, King County's 2025 quirk), attribution/licence
   string, crowd server). Seattle reproduces its Phase 0 page byte-for-byte or within a documented template diff (parity).
2. Cities, from the survey shortlist: Amsterdam (PDOK 8 cm WMS/WMTS, CC BY 4.0), Columbus (OGRIP 7.6 cm ImageServer),
   Chicago (Cook County 15 cm: a 40-row presence test), Sao Paulo (GeoSampa 10 cm WMS: 40-row junction probe).
   Tucson/Madison: config entries only, terms unconfirmed, no draw.
3. Seeded draw (seed recorded, stratified by OSM class marked/untagged/unmarked like Phase 0; plus a stratum of
   OSM crossing nodes WITHOUT a crowd label, so the set measures more than tag precision on crowd-found crossings).
   Sizes: Amsterdam 120, Columbus 120, Chicago 40, Sao Paulo 40. Second rater can repeat it from the seed.
4. Rubric: the Phase 0 conventions written into the page (Q1 paint in either year or crop; Q2 position codes 0-4 with
   the measured ~5.5-6 m near/off boundary; B tag for bike crossings; faded-but-visible = yes).
5. Probe every service live (one fetch per service first; check for blank tiles like KC 2025), then build the pages.
   Imagery stays OUT of git (local replay archive under `D:/Git/sal-scratch-keep/phase1-crosswalk-gt/`, manifest +
   sha256, same as Phase 0). Pages are local HTML (gt_gallery pattern), never Claude Artifacts. Verdicts embed + export.
6. AI blind pre-rate of paint presence (Phase 0 measured 0.98 agreement given the same imagery), written to a separate
   CSV, NEVER shown on the page Jon rates (he rates blind; agreement is computed after).
7. Tests: small pytest for the city config + seeded draw determinism + Seattle parity.

## Item 3: `stage_one` `label_type` axis (stretch, separate branch `stage1-label-type-274`)

Read RampNet's stage_one code first. Add a crossing source reader (OSM crossing nodes; inventories later), keep the
bearing rule shared, candidate crossings per pano, and a parity test that CurbRamp output is unchanged. If the design
has a real fork (e.g. where label_type lives in the data schema), write the options in the doc and stop there.

## Rules for the run

- Push feature branches. Do NOT open PRs, comment on issues, or post anywhere public; Jon does that.
- No PS server writes of any kind. No makelab writes (classifier denies; stage tarballs + install script for Jon).
- No imagery or GSV pixels committed. Licence notes per city in the doc.
- Replication section + "where each number lives" table + figures regenerator in the doc.
- Status note for Jon at the end: `D:/Git/sal-working-notes/STATUS-crosswalk-phase1-2026-11-02.md`.
