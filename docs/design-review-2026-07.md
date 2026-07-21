# Design Review & Bend Deployment Readiness — July 2026

A code review of **`sidewalk-auto-labeler`** (this repo) and
**[`RampNet`](https://github.com/ProjectSidewalk/RampNet)** ahead of the Bend, OR deployment
(2026-07-02), plus a longer-term roadmap for continually improving auto-labeling with Project
Sidewalk (PS) data. Actionable findings are filed as GitHub issues (tables below); critical
run-blocking items were fixed directly on the `bend-predeploy-fixes` branch.

## 1. System overview

```
GeoJSON area → coverage tiles (zoom 17) → pano IDs → pano download (panorama.py)
  → RampNet heatmap (2048×4096 in, 512×1024 out) → peak_local_max(threshold 0.55, min_distance 10)
  → normalized detections → <area>.jsonl → send_to_ps.py → PS /ai/submitLabelsOnPano
  → human validation in PS
```

RampNet (`projectsidewalk/rampnet-model` on HuggingFace) is a ConvNeXtV2-Base backbone with a
heatmap head, trained on ~214k auto-labeled GSV panoramas (RampNet "Stage 1" dataset, itself
bootstrapped from open-government curb-ramp point data for **NYC, Portland, and Bend** and a
crop model pretrained on PS labels from 12 cities). ICCV'25 workshop paper: arXiv:2508.09415.

## 2. Bend readiness checklist

| ✔ | Item |
|---|------|
| ☐ | **PS Bend server exists and is reachable** — the long pole; see `docs/bend-onboarding.md` (Phases A–G). Nothing in this repo can proceed to submission without it. |
| ☐ | **Server accepts AI labels for Bend** — `AiController.submitAiLabel` in SidewalkWebpage currently hard-rejects every city except `vancouver-wa`. Change the gate (ideally to a per-city `cityparams.conf` flag) before submission. |
| ☑ | `example_geojson/bend.geojson` is a bare `MultiPolygon` (verified) — correct format for `main.py`. |
| ☐ | Environment created on the deploy machine (`environment.yml` on linux-64; `requirements.txt` elsewhere) and `torch.cuda.is_available()` returns `True`. |
| ☐ | **`streetlevel >= 0.12.10` installed** — 0.12.4 silently skips every pano (broken metadata endpoint; see §3.5). Verify with a one-pano `find_panorama_by_id` call. |
| ☐ | `CurbRampDetector()` instantiated once beforehand (pre-caches the HF model download). |
| ☐ | Smoke test: tiny Bend sub-polygon run completes; immediate re-run reports "No new panoramas". |
| ☐ | One-pano visual check: `(x_normalized·width, y_normalized·height)` plotted on the full-res pano lands on curb ramps (validates the crop → 2:1-clamp → normalize → pixel chain). |
| ☐ | `python send_to_ps.py <file> --dry-run` output eyeballed; `PS_INTERNAL_API_KEY` exported if the server requires it. |
| ☐ | 2–3 records submitted to the Bend server and visible in the Validate UI before the full submission. |
| ☐ | Disk headroom for `cache/` + JSONL; long run launched under `nohup`/`tmux`. |

## 3. Pre-deployment fixes applied (this branch)

1. **Run-aborting crash fixed** — JSONL record construction ran in the writer loop *outside*
   the per-pano `try/except`; a single pano with `date=None` (or missing `image_sizes` /
   `tile_size`, or a `None` `historical`/`links` list) raised and aborted the entire run — and
   since the pano was never cached, every re-run died at the same pano. Record construction now
   lives in `build_output_line()` and is guarded per pano; incomplete-metadata panos are cached
   as deterministic skips.
2. **Skipped panos are now cached** — indoor panos and panos without metadata were re-fetched
   and reported as failures on every run. They are now written to `already_processed.txt`
   (they produce no JSONL line) and counted separately in the final report.
3. **`send_to_ps.py` hardened** — argparse CLI (`jsonl_file`, `--endpoint`, `--api-key-env`,
   `--dry-run`), resumable submission via a `<file>.submitted` sidecar, and retry with backoff
   on connection errors/5xx (4xx = permanent, logged, not retried).
4. **Portable `requirements.txt`** — `environment.yml` is a linux-64 conda export and cannot
   solve on Windows/macOS; README documents both paths. `streetlevel` is pinned (it parses
   undocumented GSV endpoints; behavior can shift across versions).
5. **`streetlevel` upgraded 0.12.4 → 0.12.10** — found live during smoke testing:
   **0.12.4's `find_panorama_by_id` always returns `None`** against Google's current metadata
   endpoint (Google changed it ~June 2026; fixed upstream in
   [sk-zk/streetlevel#40](https://github.com/sk-zk/streetlevel/issues/40)). With 0.12.4 every
   pano is silently skipped — the run "succeeds" with zero output. This would have broken the
   Bend run on **any** machine. Also per that issue, the endpoint can be intermittent — which
   is why a `None` metadata result is treated as a retryable failure, not a cacheable skip.
6. **Windows console-encoding guard** — progress messages use emoji; on a cp1252 console the
   failure-report `print` itself raised `UnicodeEncodeError` and aborted the run. `stdout` is
   now reconfigured with `errors='replace'`.
7. **`panorama.py` custom tile downloader replaced with `streetlevel.get_panorama`** — also
   found live during smoke testing: Google's anonymous tile URL
   (`streetviewpixels-pa.googleapis.com/v1/tile?...`) now returns **403 PERMISSION_DENIED**,
   so the hand-rolled downloader produced zero images (every pano "failed"). streetlevel
   0.12.10 sends whatever the endpoint now requires and stitches/crops using the metadata's
   tile grid — which also eliminates the fragile black-tile dimension probing (the bulk of
   issue [#1](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/1)).
   `fetch_panorama(metadata)` now returns a 4096×2048 RGB PIL image directly.
8. **gevent replaced with real threads** — found live during the Bend smoke test (2026-07-02):
   streetlevel's sync `get_panorama` runs an internal asyncio loop (`asyncio.run` + aiohttp)
   per call. Under gevent's `monkey.patch_all()` every worker greenlet shares one OS thread,
   so those loops collide and the patched thread-executor shutdown hangs (~300 s per join) —
   observed ~10 min per pano. `main.py` now uses `concurrent.futures.ThreadPoolExecutor`
   (same two concurrency knobs); GPU inference is serialized by a lock in `CurbRampDetector`
   since real threads — unlike greenlets — would otherwise overlap full-resolution forward
   passes and exhaust VRAM. `gevent` is removed from `requirements.txt`.

## 4. Deferred findings — sidewalk-auto-labeler

| Issue | Priority | Finding |
|---|---|---|
| [#1](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/1) | P1 | `panorama.py` probes tile-grid dimensions with a black-tile heuristic even though `process_pano` already fetched metadata containing `image_sizes`/`tile_size`; genuine black tiles truncate panos; a new `requests.Session` per tile defeats connection pooling; no 429 backoff anywhere. |
| [#2](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/2) | P1 | No GPU batching: inference runs single-image inside gevent greenlets (GIL-serialized), so `PROCESSING_CONCURRENCY=50` only parallelizes downloads while up to 50 full panoramas sit in RAM; the pano is also resized twice (4096×2048 in `panorama.py`, again in the detector transform). |
| [#3](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/3) | P1 | **Deployed inference doesn't match the evaluated configuration** — see §6. |
| [#4](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/4) | P2 | Coverage scan: per-tile exceptions silently swallowed (silent pano loss), pano-ID list rescanned every run, tiles enumerated by bbox rather than polygon intersection; output named `<basename>.jsonl` in CWD while cache is keyed by geometry hash — renaming or editing the geojson silently forks state. |
| [#5](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/5) | P2 | Only 3 indoor sources are skipped; user-generated photospheres may be out-of-distribution for RampNet. `source` is recorded per JSONL line — join against Bend validation outcomes and decide a filter empirically. |
| [#6](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/6) | P2 | `model_id`/`model_training_date`/`api_version` are hardcoded literals that can drift from the actual HF weights; detector thresholds are magic numbers; no logging framework; no tests. |
| [#7](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/7) | P3 | GeoJSON input must be a bare geometry object; accepting `Feature`/`FeatureCollection` (extracting the geometry before hashing) removes a documented footgun. |

## 5. Findings — RampNet

> **Process constraint:** before *any* code change lands in RampNet, create a **tagged release
> preserving the repo exactly as it accompanied the ICCV'25 workshop paper** (e.g.
> `v1.0-iccv2025`), so the published paper always points at reproducible code. This is noted
> in every RampNet issue.

| Issue | Priority | Finding |
|---|---|---|
| [RampNet#2](https://github.com/ProjectSidewalk/RampNet/issues/2) | P1 | Tag an ICCV'25 paper release before any other change (see above). |
| [RampNet#3](https://github.com/ProjectSidewalk/RampNet/issues/3) | P1 | **No HuggingFace export/publish script** — the deployed `projectsidewalk/rampnet-model` (a `trust_remote_code` wrapper) cannot be regenerated from the repo; the checkpoint→Hub conversion is undocumented and unversioned. Blocking for any retraining loop. |
| [RampNet#4](https://github.com/ProjectSidewalk/RampNet/issues/4) | P1 | `KeypointModel` is **defined 5× with divergent backbone construction** (`[:-2]` vs `num_classes=0, global_pool=''`); eval scripts fall back to `strict=False` state-dict loading (silently drops mismatched weights → skewed metrics) and fabricate dummy checkpoints/images when inputs are missing, so misconfigured runs "succeed". |
| [RampNet#5](https://github.com/ProjectSidewalk/RampNet/issues/5) | P1 | `stage_two/train.py` has **no warm-start flag** — training always initializes from ImageNet weights; fine-tuning the released model requires new code (the crop model already demonstrates the pattern: load `ps_model.pth`, continue training). Prerequisite for the continual-improvement loop. |
| [RampNet#6](https://github.com/ProjectSidewalk/RampNet/issues/6) | P2 | linux-64 conda export only; README says CUDA 11.8 but the env pins cu126; train/eval configured via hardcoded module constants — no argparse, so eval can't run headless as a retraining gate. |
| [RampNet#7](https://github.com/ProjectSidewalk/RampNet/issues/7) | P2 | Peak thresholds are inconsistent across scripts (README 0.5, demo 0.4, eval 0.0, deployed 0.55) with no documented provenance; the committed PR-curve CSVs support principled operating-point selection (see §6). |
| [RampNet#8](https://github.com/ProjectSidewalk/RampNet/issues/8) | P3 | Dataset regeneration depends on undocumented Google endpoints with positional-index JSON parsing (brittle, silently breakable); no registry of which cities' open-gov data entered training (NYC, Portland, **Bend**) — needed to avoid contaminated evaluations. |

## 6. Threshold & expected accuracy

From `RampNet/stage_two/evaluation_results/pr_rc_vs_c_data_manual_r0.022_pt0.0.csv`
(1,000-pano manually-labeled gold set, match radius 0.022 normalized):

| Threshold | Precision | Recall |
|---|---|---|
| 0.40 | 0.904 | 0.958 |
| 0.50 | 0.927 | 0.946 |
| **0.55 (deployed)** | **0.938** | **0.935** |

Two caveats for Bend:

1. **TTA mismatch.** Those curves were computed **with horizontal-flip test-time augmentation**
   (`stage_two/evaluate.py` maxes the original and mirrored heatmaps); the deployed
   `CurbRampDetector.detect` is single-pass. Deployed precision/recall are therefore likely
   somewhat below the table. Fix either way (add TTA at 2× GPU cost, or regenerate the curve
   without TTA and re-pick the threshold) — issue S6.
2. **Contamination.** Bend's open-government curb-ramp points were input to RampNet's Stage-1
   dataset generation, so Bend is effectively in-distribution — expect strong performance, but
   treat any Bend-derived offline eval as optimistically biased. The unbiased Bend metric is
   the **fresh post-deployment PS validation agree-rate**.

Relevance: `cityparams.conf` exposes `ai-validation-min-accuracy` per city; the deployed
threshold should be chosen to clear it with margin (see roadmap step 5).

## 7. Continual-improvement roadmap

The strategic goal: close the loop **PS validations → better model → better labels**, per city
milestone or quarterly.

1. **PS validation export** *(blocking prerequisite — needs a PS endpoint or admin SQL)*: for
   every AI-attributed label — pano id, pixel coords, confidence, agree/unsure/disagree counts,
   final outcome, city, **model version**. Everything below depends on this.
2. **Curate a fine-tuning set:**
   - *Positives:* AI labels with net agreement ≥ 2 — deliberately the same criterion RampNet's
     crop-model training used on PS `rawLabels` (AgreeCount − DisagreeCount ≥ 2).
   - *Hard negatives:* disagreed AI labels (confirmed false positives at the deployed
     threshold) — the most informative examples. With the MSE heatmap loss these are encoded
     simply by including the pano with **no Gaussian** at the disagreed point: the current
     model fires there, so the loss actively suppresses exactly those activations.
   - *Recall signal:* validations only cover what the model predicted — false negatives are
     invisible. Mine human-placed CurbRamp labels on already-processed panos that match no AI
     detection within radius 0.022 → missed-ramp positives.
   - *Anti-forgetting:* validated data is biased toward the current model's high-confidence
     regime; mix ~30% replay of the original Stage-1 dataset.
3. **Fine-tune:** warm-start from the released checkpoint (issue R3), lr 1e-6–1e-5, 1–2
   epochs, h-flip augmentation — small cheap runs, not 16-GPU retrains.
4. **Eval gates (must pass to ship):** (a) the 1k manual gold set must not regress; (b) frozen
   per-city holdout samples never used for fine-tuning; (c) contamination registry (R6) —
   Bend/NYC/Portland offline evals are biased; use fresh PS agree-rate instead.
5. **Per-city threshold calibration** *(cheap win, available before any retraining)*: sweep the
   threshold on the city's validated sample; pick the lowest threshold whose precision clears
   the city's `ai-validation-min-accuracy`; ship as per-city detector config (S5).
6. **Versioned redeploy:** HF export script (R1) publishing a model card with data-snapshot
   hash, eval numbers, and recommended thresholds; the auto-labeler reads
   `model_id`/`model_training_date` from the model config (S5) so PS attributes each validation
   to a model version — making the loop's improvements measurable.
7. **Drift monitoring:** rolling agree-rate per city × model version, label volume, confidence
   distribution; alert when agree-rate drops below the city's min-accuracy (catches camera
   generation changes, seasonal effects, geographic drift).

**New label types (later).** `detectors/` is already pluggable. Obstacles/surface problems: PS
has 12-city human labels via `rawLabels`, so RampNet's crop-model pipeline is a direct
template. Missing curb ramps: a different formulation — detecting *absence* at expected
locations; likely candidate corner points from OSM crossings + a crop classifier rather than a
dense heatmap. Prove the curb-ramp loop end-to-end first; it exercises all the plumbing.

**Prerequisite order:** PS validation export → R0 (tag) → R2 → R1 → R3 → S5 → R4.

## 8. Appendix — per-city verification runbook

Repeatable smoke-test procedure before any full city run (Bend or future cities):

1. **Env sanity** on the deploy machine: create the env, then
   `python -c "import torch; print(torch.cuda.is_available())"` and instantiate
   `CurbRampDetector()` once to pre-cache the HF weights.
1. **Scope the city:** `python main.py <city>.geojson --name <city> --scan-only` — pano
   count and runtime estimate before committing to a multi-day run.
2. **Tiny sub-polygon run:** craft a bare-geometry geojson covering a few downtown blocks
   (keep it out of the repo); `python main.py tiny.geojson --name <city>-smoke`. Expect tens
   of panos and JSONL lines in `runs/<city>-smoke/results.jsonl`. **Re-run immediately** — it
   must report "No new panoramas to process" (proves the cache path, including skip caching).
3. **Coordinate spot-check:** `python scripts/spot_check_gallery.py runs/<city>-smoke` and
   open the generated `index.html` — the numbered detection circles (clickable verdict
   markers, yellow until judged) must land on curb ramps
   (`scripts/visual_check.py` remains for checking a single pano at full resolution).
4. **Submission dry-run:** `python send_to_ps.py runs/<city>-smoke/results.jsonl --dry-run` and eyeball
   `pano_x`/`pano_y` against `width`/`height`; then submit 2–3 lines to the real server and
   confirm they appear in the Validate UI.
5. Launch the full run under `nohup`/`tmux`; re-run `main.py` after completion to catch
   transient failures (failed panos are retried; skips are not).
