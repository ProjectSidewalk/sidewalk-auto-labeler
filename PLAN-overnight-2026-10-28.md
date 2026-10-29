# Overnight dispatch plan, 2026-10-28 → 29

Author: Fable 5.1 (claude-fable-5-1), session sidewalk-auto-labeler-de. Jon asked for 2–3
important open issues that need no oversight, a plan, an adversarial review of the plan
(Opus 5.5), implementation (Opus 5.5), PRs, then deep code review (Opus 5.5). Jon is asleep;
nothing here may post to an issue, write to any Project Sidewalk server, or merge. PRs are
explicitly authorised tonight.

Coordination: peer session sidewalk-auto-labeler-eb owns RampNet#274 Phase 0 and the local
RTX 3070 (busy ~1–2 h from 21:30 PT with Mask2Former). It touches no labeler code tonight.
makelab2 is free. The main checkout `D:\Git\sidewalk-auto-labeler` is on
`clustering-eval-106` with uncommitted `runs/` files nobody owns: do not touch it. All work
happens in new worktrees under `D:\Git\labeler-wt\`.

## Picks and why

Open labeler issues without a PR: #161, #168, #169. #169 is gated on Jon's gallery-243
ratings, so it is out. Every other open issue already has its PR (#146/#159 → #163, #147 →
#154/#162, #148 → #156, #149 → #150, #151 → #155/#160).

1. **#161 — Panoramax asked us to stop hammering them.** Filed 2026-10-07 by `cquest`
   (OSM France / Panoramax). Unanswered for three weeks, and on 2026-10-08/09 the Lyon
   native-res archive re-downloaded all 174,405 HD pictures (~1.3/s from a thread pool),
   on top of the 5 m densify (87,583 HD) the day before and Bayonne's 50,409 + archive.
   Highest-importance item by a wide margin: a community-relations debt on an open-data
   partner. The *reply* is Jon's; the engineering (rate limit, SD asset, batched metadata,
   a measured answer to "is HD needed?") needs no oversight.
2. **#168 — multi-type plumbing**, the prerequisite of both routes in Planning#36
   (crosswalk and NoCurbRamp), "needed whichever label type comes first". Items 1–3 and 6
   are mechanical and decision-free; items 4 (per-type fusion params) and 5 (ranker CSV
   from validator-ai#69) wait on a second detector and on #69's format, so they are out
   of tonight's scope. Split into two implementers along a frozen record contract (below).

Three implementers: A (#161), B (#168 records/registry/main), C (#168 send_to_ps).

Housekeeping already done: the Phase 0 census scratchpad (205 MB, Temp dir) is copied to
`D:\Git\sal-scratch-keep\phase0-census-2026-10-28\` so a Temp clean cannot lose it.

## Shared rules for every implementer

- Branch from `origin/main` (`4993105`). Worktree: `git -C D:\Git\sidewalk-auto-labeler worktree add D:\Git\labeler-wt\<dir> -b <branch> origin/main`.
- Tests: `& 'C:\Users\jonf\anaconda3\envs\sidewalk-auto-labeler\python.exe' -m pytest` from the worktree; `addopts=-q` hides the summary, so branch on the exit code. Whole suite must exit 0 before a PR (1,200+ tests, a few minutes).
- Keep the suite lean: test the PS-facing contract, no new test infrastructure.
- CLAUDE.md gets a short block per change (it is the operator manual); docstrings in the repo's existing voice (why, not what).
- Commit messages: conventional subject ≤ 72 chars, body says why; trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the implementer) — the harness adds its own; keep both.
- PR body ends with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code) — plan: Fable 5.1 (claude-fable-5-1); implementation: Opus 5.5 (claude-opus-5-5)`.
  **No claude.ai session links anywhere in a PR or issue.** Bare `#161` is fine in PR bodies.
- Open the PR with `gh pr create --base main --head <branch> --title ... --body-file <file>` (body from a file, never a bare `/`-prefixed arg: MSYS mangles it). Do not merge, do not comment on issues, do not label.
- Do not push to `main`, do not touch other worktrees, do not run anything on the local GPU.
- Public writing norms if anything you write could be read outside the lab (PR bodies are public): no Google ToS / legal framing, no GSV depth-payload talk; positive and factual.
- If blocked, write what you have, push the branch, and report; do not improvise around a permission denial.

## Task A — #161 Panoramax politeness (implementer A)

Branch `panoramax-politeness-161`, worktree `D:\Git\labeler-wt\pmx161`.

### Facts measured tonight (2026-10-28, one item from Bayonne, `002aaf61…` on panoramax.ign.fr)

- Item assets: `hd` (roles data, 1,329,819 B), `sd` (roles visual, 263,140 B, 2048 px wide
  per cquest), `thumb`. `sensor_array_dimensions` = [5376, 2688] (GoPro Max). Asset
  responses are `binary/octet-stream`, no rate-limit headers.
- `GET https://api.panoramax.xyz/api/search?ids=a,b,c&limit=10` returns full items
  (assets + `pers:interior_orientation`) for several ids in one request; POST `/api/search`
  with `{"ids": [...]}` too. Chunk size limit unmeasured — probe 100, then 50 if refused.
- PS's own viewer (`SidewalkWebpage/frontend/js/common/pano-viewer/PanoramaxViewer.js`) uses
  the **SD picture as its base image** plus the tile pyramid, and sets PanoData width/height
  from `sensor_array_dimensions`. So the SD asset is what Panoramax itself expects viewers
  to pull, and PS's pixel frame for a Panoramax pano is the sensor frame.
- Today: `sources/panoramax.py` makes 1 item GET + 1 HD GET per pano, from
  `PROCESSING_CONCURRENCY=50` threads, with no limiter; `scripts/export_benchmark.py
  fetch_native` repeats the HD GET for every pano of a run from a pool of `WORKERS`.

### Deliverables

1. **A shared limiter.** `sources/ratelimit.py` (stdlib, thread-safe token bucket:
   `RateLimiter(per_second, burst=1).wait()`; retries also pay). `sources/panoramax.py`
   routes every request through one `_get(url, kind, **kw)` that waits on the limiter for
   its kind and honours `Retry-After` on 429/503 (sleep it, then retry; counts as an attempt).
   Kinds and defaults: `picture` (asset downloads) **1.0/s**, `metadata` (item GET and
   `/search`) **1.0/s**, `tile` 5/s. Env overrides `PANORAMAX_PICTURES_PER_SECOND`,
   `PANORAMAX_METADATA_PER_SECOND`, `PANORAMAX_TILES_PER_SECOND`; values are recorded in the
   manifest run entry (`panoramax_rate_limits`). `export_benchmark.fetch_native` and any
   other direct `requests.get` to a Panoramax host (grep `panoramax.USER_AGENT`,
   `api.panoramax`) go through the same `_get`.
2. **Batched metadata.** `fetch_items(ids)` via `/search?ids=` in chunks (size: measured,
   ≤ 100). To keep `fetch_pano(pano_id, lat, lon)`'s interface, add an OPTIONAL source hook
   `queue_hint(ordered_pano_ids)` (document it in `sources/__init__.py`), called by main.py
   right before the processing pool with the post-thinning todo list in submission order.
   The module keeps that order; on a cache miss `fetch_item` fetches the window of up to one
   chunk starting at the missing id's position (unprocessed ids only) into a bounded cache
   (entries are popped on use; cap ~4 chunks). Per-id GET stays as the fallback when no hint
   was given (gap-fill, reinfer, export_benchmark). Net: Lyon-size run = ~1,750 metadata
   requests instead of 174k.
3. **`--panoramax-asset {hd,sd}`** in main.py (default decided by the study below, rule
   pre-registered here). The module setting `IMAGE_ASSET`; `image_url(item, asset)`. SD
   missing on an item → fall back to HD for that pano and count it (`asset_fallbacks` in the
   run entry). **Record frame under SD:** `pano.width/height` = `sensor_array_dimensions`
   when the item carries them (that is PS's own frame for this source, see above; the study
   reports how often they equal the HD file's real dimensions on the sample), else the
   downloaded SD dimensions. Detections stay normalized, so `send_to_ps` is untouched. The 2:1
   aspect check runs on whatever was downloaded. `source_metadata` gains `image_asset`
   (`hd`|`sd`), `image_url` (what was downloaded) and `image_dimensions_downloaded`; `hd_url`
   stays. The manifest **binds** `panoramax_asset` like `thin_spacing_m` (#136): resuming
   with a different asset is refused; legacy manifests read as `hd`.
4. **Processing concurrency cap.** A source may declare `MAX_PROCESSING_CONCURRENCY`
   (Panoramax: 8); main.py takes `min(flag, cap)` and prints it. 50 idle threads holding
   connections against a 1/s bucket buy nothing.
5. **The study** `scripts/panoramax_asset_study.py` (stdlib + numpy/Pillow + the detector),
   run on **makelab2** (A40; py312 checkout `/homes/gws/jonf/sidewalk-auto-labeler-py312`,
   `git pull` it first or `git fetch && git checkout panoramax-politeness-161`; the Lyon
   archive is local there under `/projects/makeabilitylab/sidewalk-auto-labeler/runs/lyon/`
   — read its `index.csv` for ids, rigs and real dimensions; verify the exact path with `ls`
   before assuming). Verified 2026-10-28 21:29 PT: the A40 shows 8.7 GB used by someone else at
   0% util (fine); `index.csv` has only `panorama_id,filename,bytes,sha256`, so rigs and real
   dimensions come from the archive dir's own `results.jsonl` (`pano.camera_model`,
   `pano.width/height`); the py312 checkout sits at 632e819 with dirty `runs/` files — do NOT
   stash or pull it; instead `git -C /homes/gws/jonf/sidewalk-auto-labeler-py312 fetch origin
   panoramax-politeness-161 && git -C ... worktree add /homes/gws/jonf/sal-pmx161
   panoramax-politeness-161` and run with `/homes/gws/jonf/sidewalk-auto-labeler-py312/.venv/bin/python`
   (set `PYTHONPATH` to the worktree). An SSH master is already open. **makelab rules:** open ONE master first
   (`& 'D:\Git\dotfiles\wsl-ssh.ps1' makelab2 open`), batch work into few `run`/`script`
   calls, never fan out, and **one timeout means stop: zero retries**, report. Check
   `nvidia-smi` is free before using the GPU; the archive root is also a frozen Py3.9
   checkout — do not `git pull` there, run code only from the py312 checkout.
   - Sample: 300 Lyon panos stratified by `camera_model` (≥ 30 per rig with that many; seed 161), plus the 300 matching SD assets fetched at 1/s through the new limiter (≈ 80 MB, 5 min; this IS the behaviour we are adopting, and is well under what cquest asked for).
   - Arms per pano, all resized to 4096×2048 BILINEAR as the pipeline does: `hd` (archive bytes), `sd` (real derivative), `sd_sim` (archive HD → 2048×1024 LANCZOS → 4096×2048, to separate resampling from JPEG).
   - Metrics vs the `hd` arm at 0.30 and 0.55: one-to-one match within radius 0.022 normalized (RampNet's matcher geometry: dx·1024 / dy·512 with seam wrap; copy the rule from `scripts/agree_rate.py`, do not import pandas), recall-of-HD, precision-vs-HD, matched confidence delta (median, p90), pixel offset (median px at 1024×512), per rig and pooled; sensor-dims == archive-dims rate.
   - **Pre-registered default rule:** `sd` becomes the default iff pooled recall-of-HD and precision-vs-HD at 0.30 are both ≥ 0.95 AND every rig with ≥ 30 sample panos has both ≥ 0.90. Otherwise the default stays `hd` and the PR says what SD would cost.
   - Outputs: `docs/panoramax-asset-study.md` (replication section + where-each-number-lives table, like the other study docs) and `runs/_summary/panoramax_asset/{summary.csv,per_rig.csv,per_pano.csv}` committed. The 300 SD files are not committed.
6. Tests (monkeypatched `requests`, no network): limiter spacing with a fake clock; `Retry-After` honoured; `/search` chunking and the window cache (order, pop-on-use, fallback GET without a hint); `sd` record frame (sensor dims vs downloaded dims, fallback to hd when `sd` is absent, `image_asset` recorded); manifest binding refusal; concurrency cap; `fetch_native` goes through the limiter.
7. Docs: CLAUDE.md block (flag, env knobs, binding, the archive-re-download caveat), module docstring update, and `docs/panoramax-asset-study.md`.
8. **Draft reply for Jon** (not posted): `D:\Git\sal-working-notes\DRAFT-161-reply.md`. Plain, apologetic, factual: what we pulled and when (numbers above), what changed (defaults, limits, batched metadata, SD if adopted), an offer to coordinate before any further large run, and a question whether they would prefer a bulk channel. Written as a human maintainer; no AI mention, no provenance of other projects beyond what the issue already shows.

### Out of scope (list in the PR body as follow-ups, do not file issues)

Archiving the bytes *during* a run so the native-res archive never re-downloads (the
archive step is what hit Panoramax hardest); Mapillary gets no limiter tonight (it has its
own token-gated quota).

## Frozen record contract for #168 (binds B and C)

Readers always go through helpers in `detectors/registry.py` (torch-free):

- `detection_type(record, detection) -> str`: `detection['label_type']`, else
  `record['label_type']`, else `'CurbRamp'`.
- `record_types(record) -> list[str]`: `record['label_types']` if present, else
  `[record.get('label_type') or 'CurbRamp']`. This is "which detectors RAN on this pano",
  so a pano with zero detections still says what it was checked for.
- `provenance_for(record, label_type) -> dict`: `record['models'][label_type]` if present,
  else the line-level `model_id / model_training_date / api_version / model_repo /
  model_revision`.

Writer rule (byte-identical equivalence is the issue's done-when, and the peer session
asked for it explicitly): **a run whose type set is exactly `{CurbRamp}` writes today's
format with no new key.** Any other type set writes, per line:

```
"detections": [{"x_normalized", "y_normalized", "confidence", "label_type": "<T>"}, ...]
"label_type": "<T>" when every detection on the line is one type (or label_types has one entry), else null
"label_types": ["<T1>", "<T2>"]            # sorted; the detectors that ran
"models": {"<T1>": {model_id, model_training_date, api_version, model_repo, model_revision}, ...}
<line-level model_* keys> = models[label_types[0]]   # legacy readers see the first type's
```

`API_VERSION` stays `1.0.0`: each POST is still the same wire format. Detector specs:

```
DetectorSpec(label_type, module, cls, storage_floor, operational_confidence,
             benchmark_confidence, mask_rig)
SPECS = {'CurbRamp': DetectorSpec('CurbRamp', 'detectors.curb_ramp', 'CurbRampDetector',
                                  DETECTION_STORAGE_FLOOR, OPERATIONAL_CONFIDENCE,
                                  BENCHMARK_CONFIDENCE, True)}
```

The module constants in `detectors/__init__.py` keep their names and values (every import
keeps working); the CurbRamp spec references them. The thresholds and the rig mask are read
**from the spec of the detection's type** wherever a consumer acts on a detection.

## Task B — #168 records, registry, main.py (implementer B)

Branch `multitype-records-168`, worktree `D:\Git\labeler-wt\mt168a`.

1. `detectors/registry.py`: the contract above plus `load_detectors(label_types, **kw) ->
   {type: detector}` (lazy import per spec; each detector exposes `.detect(image)`,
   `.provenance`, `.stats()`, `.close()` — the CurbRampDetector surface, unchanged).
2. `main.py --types CurbRamp` (comma list; default `CurbRamp`; unknown type = parser error
   listing SPECS). `_process` runs every loaded detector on the fetched image and tags each
   detection with its type; `build_output_line(result, provenances)` implements the writer
   rule; the manifest binds `label_types` (sorted) like geometry/source/model (legacy
   manifests read `['CurbRamp']`; a mismatch is refused with the same voice as
   `thin_spacing` — "new run name, never a resume"); the manifest `model` block stays for
   the first type and gains `models` when there are several. `report_detector` per type.
   Keep a module global `main.detectors` (dict) and keep `main.curb_ramp_detector` as the
   CurbRamp entry so `scripts/reinfer.py` and `scripts/detect_from_store.py` keep working
   (they set it directly; update them to go through the registry with `--types` defaulting
   to CurbRamp only if that is a small change, else leave them and say so).
3. Consumers read through the helpers and filter by type: `fuse_sites.load_results`
   (`--label-type`, default CurbRamp; sub-floor band logic unchanged), `eval_sites`,
   `position_check` (counts only), `coverage_check` (already keys label_types; make it use
   `record_types`), `export_benchmark.py --label-type` (strata at the type's benchmark tier;
   `records.jsonl` carries the type), `provenance_gate`, `agree_rate`, `mined_precision`
   where they read `detections` — the minimal edit is "iterate detections of the requested
   type"; do not restructure them.
4. Tests: golden equivalence — `build_output_line` for a CurbRamp-only run equals, key for
   key and in key order, the dict a committed line of `runs/bayonne/results.jsonl` has
   (strip the pano block; compare the rest), and `json.dumps` of it is byte-identical; the
   two-type line shape; helper fallbacks on a legacy line; manifest binding (legacy,
   match, mismatch); `--types` parsing; `fuse_sites.load_results` ignores the other type.
5. CLAUDE.md: a "Multiple label types (#168)" block: the writer rule, the helpers, the
   binding, what is NOT done (per-type FuseParams, ranker export, a second detector).

## Task C — #168 one POST per (line, type) in send_to_ps (implementer C)

Branch `multitype-submit-168`, worktree `D:\Git\labeler-wt\mt168b`, **based on
`multitype-records-168`** once B has pushed `detectors/registry.py` (start from
`origin/main` against the contract above with a local copy of the helpers if B is not there
yet, then rebase onto B's branch before the PR; the PR targets B's branch — a stacked PR).

1. `transform_record(data, min_confidence, max_confidence, mask_rig, label_type=None)`:
   with a type, keep that type's detections only, set the payload's `label_type`, take
   provenance from `provenance_for`, apply `mask_rig` only if the spec says so, and use the
   spec's `operational_confidence` as the default when the caller passed none. Without a
   type: today's behaviour exactly (single-type files). `split_record(data) -> [(type,
   payload)]` over `record_types`.
2. The POST loop sends one `AiLabelsSubmission` per (line, type). **Sidecar:** a single-type
   line completes as the bare line number (today's bytes); a multi-type line writes
   `N:Type` per completed type and the line counts as done when all of `record_types` are
   present. `load_submitted_lines` returns the done set (ints) and the partial map.
   **Record:** `labels_submitted_by_type` and `min_confidence_by_type` per endpoint, written
   only for files with more than one type, so a CurbRamp-only record is unchanged.
3. Guards: `check_resume_state` / `append_check` / `check_band_state` work on line numbers
   as today; a **band on a multi-type file is refused** with a message (bands exist for
   legacy CurbRamp cities; say so). `--min-confidence` on a multi-type file applies to every
   type (explicit flag wins over per-type defaults) and the record says which.
4. Tests: golden — a CurbRamp-only file produces a sidecar and a submission record
   byte-identical to the current code's (capture them from `main` with the same fixture
   before changing anything, commit the fixture); two-type dry run shows two POSTs per line
   with the right pixel transform and provenance; resume after a partial line sends only the
   missing type; `mask_rig` follows the spec; band on a multi-type file refused.
5. CLAUDE.md: the sidecar/record extension, in the Stage 2 section.

## Review gates (after implementation)

One Opus 5.5 reviewer per PR, fresh context, given the diff and the plan, asked for
correctness findings ranked by severity, each with a failure scenario, plus a replication
check of every number in A's study doc against its CSVs. Findings go to the implementer
(same agent, by message) for fixes; the reviewer's summary is appended to the PR body as a
"Review" section. Nothing is merged.

## What Jon decides in the morning

- Post the #161 reply (draft in `DRAFT-161-reply.md`) and whether to flip the asset
  default if the study came out near the bar.
- Merge order: A is independent; B then C (stacked).
- Whether #168 items 4–5 (per-type FuseParams, ranker review-order export) are wanted
  before a second detector exists.
