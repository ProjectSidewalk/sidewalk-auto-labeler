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


---

# Amendments after the adversarial review (Opus 5.5, 2026-10-28 ~22:00 PT)

**These override anything above that conflicts.** The reviewer read `origin/main` (4993105);
the main checkout `D:\Git\sidewalk-auto-labeler` is 188 commits behind it, so every
implementer reads and edits code ONLY in its own worktree (created from origin/main), never
the main checkout.

## Everyone

- Merge order A -> B -> C. New tests go in NEW files (`tests/test_panoramax_politeness.py`,
  `tests/test_multitype_records.py`, `tests/test_multitype_submit.py`), not into
  `test_main_units.py` / `test_send_to_ps.py`.
- CLAUDE.md anchors so hunks never touch: A appends under the `- **panoramax**:` source
  bullet; B adds a new block right after the "Output format notes" section; C adds under
  "Stage 2 - submission".
- `main.py` is shared by A and B. **A's main.py footprint is generic source hooks only:**
  (a) `--panoramax-asset` handled in `main()` beside the Mapillary `POSITION_FIELD` block;
  (b) `hint = getattr(source, 'queue_hint', None)` called once before the processing pool;
  (c) `PROCESSING_CONCURRENCY = min(flag, getattr(source, 'MAX_PROCESSING_CONCURRENCY', flag))`
  in `main()` so gap fill inherits it; (d) one generic `source.manifest_bindings()` ->
  `{'panoramax_asset': 'sd'}` checked by one helper called at the END of
  `load_or_init_run_dir` (bind like `bind_decode`: a legacy manifest that already holds
  records is known to be `hd`; written for Panoramax manifests only); (e) one
  `source.run_record()` dict merged into the run entry (rate limits, `asset_fallbacks`).
  B owns everything detector-side in main.py (`_process`, `build_output_line`, detector
  loading, `report_detector`, the types binding).

## A (#161)

- **Frame under `sd`:** `pano.width/height` = `sensor_array_dimensions` when present and
  2:1; otherwise **download HD for that pano** and count it in `asset_fallbacks`. Never
  record SD dimensions: PS's viewer (PanoramaxViewer.js:533-542) and the PS pano store
  (pano-tools saves HD; CropRunner skips labels whose stored width differs from the JPEG's)
  both live in the HD/sensor frame, and in this repo `agree_rate`, `reinfer.PANO_INVARIANTS`
  and `detect_from_store` assume it. Under `hd` write **no new keys** (absent = hd, like
  `detection_decode`), so resumed Lyon/Bayonne lines keep their shape. Under `sd`,
  `source_metadata` gains `image_asset`, `image_url`, `image_dimensions_downloaded`.
- `export_benchmark.fetch_native` passes `asset='hd'` explicitly (test it): the native
  archive must never silently become SD. `scripts/reinfer.py` sets `source.IMAGE_ASSET`
  from `manifest['panoramax_asset']` (default hd), as it does Mapillary's `POSITION_FIELD`.
- **Cache, simplest correct form:** `queue_hint(ids)` stores the list + a cursor. `fetch_item`
  takes ONE lock; on a miss it fetches that id plus the next <= 99 hinted, not-yet-cached ids
  from the cursor in one `/search` call (`limit=len(chunk)`, always), advances the cursor;
  pop on use; <= 4 chunks cached. No hint -> per-id GET (today). An id ABSENT from a search
  result falls back to the per-id GET (never read absence as gone). Live check on ~20 ids
  that `build_pano_record(search_item)` == `build_pano_record(pictures_item)` (minus volatile
  keys); record the result in the PR; if they differ, normalize or do not batch.
- **Limiter:** injectable clock/sleep; 429/503 pauses the whole KIND for
  `min(Retry-After, 300)` s (every thread waits, not one); exhaustion is a retryable failure;
  tiles and `prepare()` get the same handling. Per process (a makelab2 study + a local run
  = 2/s combined; say so in CLAUDE.md). An autouse fixture in `tests/conftest.py` sets the
  Panoramax limiters to unlimited so the 31 existing tests do not sleep. Keep
  `fetch_item(picture_id)` and `_download_image(url)` callable with today's arguments (asset
  as a keyword with a default) and keep calling `requests.get` so existing monkeypatches hit.
- Arithmetic in CLAUDE.md and in `--scan-only`'s estimate: 1 picture/s means Lyon-size (174k)
  is ~48 h of downloads, Bayonne ~14 h. `MAX_PROCESSING_CONCURRENCY = 4`.
- **Study inputs:** the archive `index.csv` has NO rig and NO dimensions; rigs and HD dims
  come from the archive dir's `results.jsonl` (verify it is the 5 m file: sha256 begins
  `60e688b2`; if not, say so and use what is there, reporting its sha). Sample
  ~500 detection-bearing (HD >= 0.30) + ~100 empty panos, seed 161, stratified by rig within
  each. SD fetch ~600 requests at 1/s ~10 min. **Sanity gate:** the `hd` arm on archive
  bytes must reproduce the stored detections (same revision/decode/border) or the pipeline
  under test is not production's: stop and report if it does not.
- **Metrics:** strict (both >= 0.30) AND lenient (HD >= 0.30 matched by SD >= storage floor
  0.1) recall-of-HD; precision-vs-HD; matched confidence delta; instead of median px,
  the share of matches in the same 8-px coarse cell / at 7-8 px / beyond (the heatmap is a
  stride-8 grid; radius 0.022 is ~22.5 px, ~3 cells). `sd_sim` stays exploratory, outside
  the rule. If `../RampNet/benchmark/lyon` holds verdicts, add an exploratory SD-vs-GT arm.
- **Rule (final):** pooled strict recall-of-HD AND precision-vs-HD at 0.30 >= 0.95 (lower
  95% bound reported), and a rig with >= 30 sampled panos fails only if its 95% UPPER bound
  on either is < 0.90. Commit the pre-registration into `docs/panoramax-asset-study.md` on
  the branch BEFORE the first scoring run (the repo's convention; this plan file does not
  count).
- **Doc conventions** (memory `replicable-reports-and-figures.md`): Key takeaways box;
  Replication section (commands, sha256 of every input, seed, model id + revision, package
  versions, host/GPU, runtime); where-each-number-lives table; a `figures` subcommand that
  is byte-reproducible from the committed CSVs (no GPU/network; load the `dataviz` skill
  before drawing); one seeded contact sheet of SD/HD disagreements with CC BY-SA attribution
  on the crops. The matcher and the rule get torch-free tests.
- **makelab2:** push the branch first; worktree there as described above; run the study
  **detached in tmux** (`tmux new-session -d -s pmx161 ...`) and poll through the live
  master at multi-minute intervals (a foreground `run` longer than `-TimeoutSec` returns 124
  and would force a self-inflicted STOP); `--batch-size 1`; read the archive and its
  `results.jsonl` in place, read-only, never write under
  `/projects/makeabilitylab/sidewalk-auto-labeler`; `get` the outputs back and commit from
  Windows. If any `ls` on `/homes` or `/projects` hangs: stop, report.

## Contract (B and C)

- Line-level `label_type` = the type iff `len(label_types) == 1`, else `null`. Never derived
  from the detections on the line.
- Line-level `model_*` = CurbRamp's provenance when CurbRamp is among the types, else
  `label_types[0]`'s. `detection_decode` / `detection_border` (merged since the plan was
  drafted; `bind_decode` / `bind_border` in main.py) describe the CurbRamp detector; another
  type's decode/border go inside `models[T]`.
- `detection_type` RAISES on an untyped detection in a multi-type line (never falls through
  to CurbRamp there). `transform_record` without `label_type=` RAISES on a multi-type line.

## B (#168 records)

- First commit, pushed within ~30 min: `detectors/registry.py` + its tests. C waits for it.
- Bind `label_types` with `bind_decode`'s pattern (legacy = `['CurbRamp']`, bound without a
  manifest rewrite); bound for `--gap-fill-only` too (it writes records); `None` only for
  `--scan-only`. Bind EVERY type's revision (`models` map in the manifest).
  `scripts/detect_from_store.bind_manifest` binds types or refuses non-CurbRamp.
- Golden test: extend the existing byte pin `PRE_130_LINE_SHA256` in `tests/test_border.py`
  (it hashes `json.dumps(main.build_output_line(...))`): the same hash must hold through the
  registry path; plus a `fuse_sites.load_results` equivalence on a fixture (legacy line vs the
  same line through the helpers -> identical). `runs/bayonne/results.jsonl` is gitignored and
  absent in CI, so it cannot be the golden source.
- Consumers trimmed: only `fuse_sites` (`--label-type`), `export_benchmark` (`--label-type`)
  and `position_check` (counts) iterate typed detections. `agree_rate`, `mined_precision`,
  `eval_sites`, `provenance_gate` get a one-line refusal of a file whose `record_types` is
  not `{CurbRamp}`. `coverage_check` moves to C.

## C (#168 submission)

- Start after B's registry commit is on `origin/multitype-records-168`; base the worktree on it.
- **Per-type sidecars, not `N:Type` lines:** `<file>.submitted` for single-type files (today's
  bytes), `<file>.type-<T>.submitted` (plain ints) per type for multi-type files; the record
  gets `endpoints[ep].types[T] = {submitted_lines, labels_submitted, min_confidence,
  rig_masked}`, mirroring `bands`. Reason: `position_check._read_sidecar` (int parsing, reached
  by `campaigns_for` -> `live_positions` -> `send_to_ps.check_live_positions`, `reposition`,
  `coverage_check`) would raise on `N:Type`; and a half-sent line must still count as "placed".
- Update `position_check._read_sidecar` / `campaigns_for` / `live_positions` (a pano is
  placed if ANY type's sidecar sent it), the `backfill_metadata.py` sidecar globs
  (`.type-*.submitted`), `coverage_check`'s replay (per type, tag with `detection_type`),
  `count_labels` / `count_band_labels_in_file` (per type), `check_model_provenance` (every
  `models[T]`), and `check_resume_state`'s min_confidence comparison (per type).
- Refuse a band on a multi-type file in `process_jsonl_file` BEFORE `check_band_state`;
  `reinfer --write-band-file` refuses multi-type files too.
