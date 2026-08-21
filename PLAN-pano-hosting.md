# Implementation plan: pano hosting + server-side crops

**Written:** 2026-08-11. Companion to `HANDOFF-pano-hosting.md` (same directory) — read that first for the
full problem statement and evidence. This document is the *implementation* plan, reviewed against the code
on `origin/develop` of `SidewalkWebpage` on 2026-08-11. It is written to be executed by a model/developer
with no other context from the planning session.

**Repos:** `SidewalkWebpage` (`~/git/SidewalkWebpage` on WSL — note the checkout may be sitting on a stale
QA branch; always branch from `origin/develop`) and `sidewalk-auto-labeler` (this repo).
Crop-geometry reference: `sidewalk-panorama-tools` (`D:\Git\sidewalk-panorama-tools`).

**Status of decisions:** Jon approved the overall design direction (handoff §4 + the review that produced
this plan). The Richmond production submission **waits** for all of this plus the next release cut — that
decision stands and is not revisited here.

---

## 0. The design in three sentences

The labeler uploads its archived native-resolution panoramas into Project Sidewalk's **existing**
self-hosted imagery store (`pano.images.directory`, served via `/backupImage/:panoId`, tracked by
`pano_data.has_backup`). The server treats crops and display derivatives as **derived data**: a nightly
reconciliation job (plus admin trigger) scans for *label whose pano has stored pixels ∧ crop file missing*
and cuts the crop server-side — never inline with ingest, so submission order can never strand a label.
The Gallery serving path needs **zero changes**: `PanoDataService.cropUrl` is a bare filesystem check, so
cards heal the moment `crop_<labelId>.png` exists.

**Why reconciliation, not an ingest hook (the one structural change vs. the handoff):** the codebase
already treats `has_backup` exactly this way — `CheckImageExpiryActor` → `PanoDataService.panoExists`
rewrites `has_backup` from `backupExists(panoId)` (a disk stat) on every check, for both GSV and
Mapillary, and `ImageController.serveBackupImage` calls `markHasBackup` opportunistically. Following that
precedent makes crop generation idempotent, order-independent, repairable for any past city (including
Vancouver later, via the option-B live-fetch floor), and immune to the handoff §9 irreversibility risk.
It also deletes the need for the handoff §5.1's "return label IDs / resolve-by-(pano_id, x, y)" machinery.

**No database evolution is needed anywhere in this plan.** `has_backup` exists (evolution 319); crops and
derivatives are files. Do not add one.

---

## 0.5 Ownership

| Repo | Owns | Deliverable |
|---|---|---|
| **SidewalkWebpage** | Upload intake endpoint; crop generation **and, from PR-A2 on, the canonical display-crop geometry**; display derivative; attribution; reconciliation actor | One issue; PRs A1 + A2 |
| **sidewalk-auto-labeler** | Pano selection + upload client + ledger; the labeler#42 rotation measurement | One issue + the #42 promotion note; PR-B |
| **sidewalk-panorama-tools** | **Nothing.** It contributes only the settled crop-geometry *reference* (post-#47 `CropRunner.py`), read once at a pinned commit | No changes |

Two ownership rules worth stating outright:

- **The geometry port is a one-time extraction, not an ongoing sync — and the end state is one
  implementation, not two.** `CropRunner.py` has **zero consumers anywhere** (verified 2026-08-11: the
  nightly scraper entrypoint runs only `DownloadRunner.py`; the auto-labeler never imports it;
  SidewalkWebpage's only crop producer is the browser `saveImage` path; **RampNet doesn't use it either**
  — its Stage-1 crop model trains on crop *images* downloaded from Project Sidewalk, and its dataset
  generation fetches GSV tiles with its own code; nor do validator-ai/tagger-ai/quality-analysis). So
  PR-A2 reads the post-#47 mechanics and the v1 sizing heuristic at a **pinned commit SHA** (recorded in
  the fixture README), and after that **SidewalkWebpage is the single source of truth for *production*
  crop geometry**. Research consumes the crop *corpus* PS produces (RampNet already trains on it), not
  crop code.
- **CropRunner's ongoing role (decided with Jon, 2026-08-11): research bench, never a production
  dependency.** The team keeps experimenting with auto-cropping in pano-tools (the ground-truth study).
  The flow is strictly one-way: experiments iterate there freely; a *converged, validated* result crosses
  to SidewalkWebpage as a **sizing-rule spec** — formula + constants + validation crops (which become PS
  test fixtures) — and PS regenerates all crops via the reconciliation job. This is the lat/lng estimator
  precedent (#4819). PS never tracks CropRunner's experimental states, and nothing pins the two after the
  initial port SHA. Implementers: do not add any runtime or test dependency on pano-tools.
- **Cross-repo contract:** the only interface between the two owning repos is the
  `PUT /ai/panoImage/:panoId` endpoint (auth, headers, status codes as specified in PR-A1). Neither side
  should assume anything else about the other's internals.

---

## 1. Workstream A — SidewalkWebpage (two PRs)

### PR-A1: keyed panorama upload endpoint

Small, self-contained, and lands first so the ~9 GB richmond-test transfer can start while PR-A2 is in
review.

**Route** (in `conf/routes`, next to the existing AI block at `/ai/submitLabelsOnPano`):

```
+ nocsrf
PUT     /ai/panoImage/:panoId    controllers.AiController.uploadPanoImage(panoId: String)
```

`+ nocsrf` is **mandatory** — Play's CSRF filter 403s any POST/PUT carrying an `Authorization` header.
The existing `/ai/submitLabelsOnPano` route has a comment block explaining exactly this (it was a
production bug, #4809); copy that comment style.

**Handler** in `app/controllers/AiController.scala` (it owns the keyed ingest surface):

- **Auth:** `ControllerUtils.internalKeyValid(request, config internal-api-key)`, same as
  `submitAiLabel`. 401 on failure.
- **Body:** raw streamed to a temp file — `parse.maxLength(<cap>, parse.temporaryFile)` or equivalent.
  Never buffer the body in memory; panos are 10–40 MB. Add a config key
  `pano.upload.max-bytes` (default 64 MB) to `conf/application.conf`. Over-cap → 413.
- **Pano ID validation:** reuse the `PANO_ID_PATTERN` regex currently private in
  `ImageController` — move it somewhere shared (e.g. `PanoDataService` companion or
  `controllers.helper.ControllerUtils`) rather than duplicating. Invalid → 400.
- **Content validation, in order:**
  1. `Content-Type` must be `image/jpeg` or `image/png`.
  2. Magic-byte sniff of the temp file must agree (JPEG `FF D8 FF`, PNG `89 50 4E 47`).
  3. Dimension peek **without full decode** — `ImageIO.getImageReaders` → `reader.getWidth/getHeight`
     reads only the header. Reject > 2.2×10⁸ pixels (decode-bomb guard; 16384×8192 = 1.34×10⁸ passes)
     and < 1024×512. Do **not** hard-require 2:1 aspect (warn-log only).
- **Integrity:** require an `X-Content-Sha256` header; compute the digest of the received file and 400 on
  mismatch (delete the temp file).
- **Idempotency / conflict:** resolve the target via the same rule as
  `PanoDataService.localBackupImageFile` — `<pano.images.directory>/<city-id>/<panoId[0:2]>/<panoId>.<ext>`
  (ext from the sniffed type). If a file already exists: same hash → 200 `{"status": "exists"}`;
  different hash → **409** with both hashes in the body, never overwrite. No `overwrite` flag in v1.
- **Atomicity:** write is temp-file + `Files.move(..., ATOMIC_MOVE)` into place. Create the two-level
  directory if needed (`ImageController.initializeDirIfNeeded` is prior art).
- **On success:** call the existing `panoDataService.markHasBackup(panoId)` fire-and-forget (exactly as
  `serveBackupImage` does). If the `pano_data` row doesn't exist yet (pixels uploaded before metadata),
  that's fine — the nightly `CheckImageExpiryActor` reconciliation sets `has_backup` once the row appears.
  Do not make row-existence a precondition.
- **Response:** JSON `{status, pano_id, bytes, sha256}`; 201 for a new file.

**Tests** (`test/controllers/` — DB-backed spec, ScalaTest): 401 without key; 400 bad pano id; 400 magic
mismatch; 400 sha mismatch; 413 oversize; 201 happy path (file exists on disk afterwards at the exact
`localBackupImageFile` path, `has_backup` set when the seeded `pano_data` row exists); 200 idempotent
re-upload; 409 different content. Seed the `pano_data` fixture in the spec — do not rely on dump data —
and negative-control (assert the 409 case did *not* change the stored file).

### PR-A2: crop generation + display derivative + attribution

**New service `app/service/CropService.scala`** — don't grow `PanoDataService` further. Three parts:

#### (a) Crop geometry: settled mechanics + an explicitly swappable sizing rule

`CropRunner.py` (read at a pinned commit SHA — see §0.5) contains two very different kinds of logic, and
this plan treats them differently. **Do not port it as one undifferentiated blob.**

**Mechanics — settled; port faithfully.** These are equirectangular *topology* facts, debugged via
panorama-tools #47/#77, and no study can change how a seam works:

- `compute_crop_box(pano_x, pano_y, crop_size, pano_width, pano_height)`:
  `size = min(round(crop_size), pano_width, pano_height)`; `left = round(pano_x − size/2) mod pano_width`
  (x wraps at the equirectangular seam); `top = clamp(round(pano_y − size/2), 0, pano_height − size)`
  (y clamps *by shifting* — the poles are not adjacent, and no crop may contain synthetic black).
  Integers throughout.
- `extract_crop`: when `left + size > pano_width`, the window crosses the seam — read **two** segments
  (`[left, pano_width)` and `[0, size − first_width)`) and stitch.

**Sizing — known-imperfect; isolate it and make it swappable.** `predict_crop_size` is a ~decade-old
distance regression, self-flagged in its own docstring, and **currently being replaced via a
ground-truth study in panorama-tools**. Treat it as *sizing-rule v1*, not as settled geometry:

- Implement sizing as **one isolated function** (e.g. `cropWindowSize(label, panoDims)`) with the
  provenance and a version marker documented at the definition — the single place a future rule swaps in.
- v1 ports the current heuristic: `old_pano_y = pano_height/2 − pano_y`;
  `distance = max(0, 19.80546390 + 0.01523952 × old_pano_y)`;
  `crop_size = 8725.6 × distance^−1.192`, clamped to [50, 1500]; `1500` when `distance == 0`.
  **With resolution normalization:** the constants were calibrated on GSV panos 6656 px high, so compute
  in reference space and scale back — `ref_y_offset = old_pano_y × (6656 / pano_height)`, run the formula
  and clamp on that, then `crop_size_native = crop_size_ref × (pano_height / 6656)`. Bit-identical to the
  Python at `pano_height = 6656`.
- Structure the call site to **prefer a per-label stored bounding box when one exists** (none exist today
  — see the RampNet note in §5 — so v1 always falls through to the heuristic; the seam is what matters).
- **Why shipping a known-imperfect sizing rule bakes nothing in:** crops are derived data under the
  reconciliation job. When the ground-truth study lands a better rule, bump the sizing version, port the
  new constants, delete (or version-stamp) the crop files, and the nightly job regenerates every crop —
  sizing improvements are retroactive for free, across all cities. An imperfectly sized context crop now
  beats a broken-image icon; do not wait for the study.

**Memory discipline (this is where a naive port kills the shared 56-city JVM):** never fully decode a
pano for a crop. Use `ImageIO` `ImageReadParam.setSourceRegion` to read only each segment's window — a
full 16384×8192 decode is a ~512 MB `BufferedImage`. Process panos strictly sequentially, labels grouped
by pano so each file is opened once.

**Output:** PNG at exactly the path `PanoDataService.cropFile` resolves —
`<cropped.image.directory>/<city-id>/<LabelType>/crop_<labelId>.png` — via temp + atomic rename. That
makes `cropUrl`/`serveCropImage` and the Gallery work with zero further changes.

**Golden-fixture tests — split along the same mechanics/sizing line:** generate fixtures **once** with
`CropRunner.py` (record the panorama-tools commit SHA used) from a small *synthetic* pano (e.g. 2048×1024
gradient/checkerboard PNG, so nothing heavy is committed).

- **Mechanics fixtures** (permanent): a seam-wrapping window (near x=0 and near x=width) and a
  pole-clamped window, cut at *fixed, hand-chosen sizes* — they pin seam/clamp/stitch behavior and never
  need regenerating when the sizing rule changes.
- **Sizing fixtures** (versioned with the rule): expected window sizes for a centered label and one at a
  non-6656 height exercising the normalization. Regenerated when sizing-v2 lands.

Commit pano + expected outputs under `test/resources/` with a short README naming the generating script,
commit SHA, and command. ScalaTest asserts pixel-identical output (PNG in, PNG out — lossless, so exact
equality is achievable). The fixtures are acceptance proof for the port, then regression fixtures for the
Scala implementation; they create **no ongoing dependency on `CropRunner.py`** — its revision or removal
doesn't touch them.

#### (b) Reconciliation actor

`app/actor/CropGenerationActor.scala`, modeled directly on `CheckImageExpiryActor` (self-scheduling;
register in `app/modules/ActorModule.scala` via `bindActor`). Nightly, offset from the other actors'
times. Candidate query: labels (with `label_point.pano_x/pano_y`) joined to `pano_data` where
`has_backup = true`; then a per-label `cropFile(...).exists()` stat filters to the missing ones. This is
deliberately **not** AI-only — it also heals any crowd label whose browser `saveImage` POST failed, and
it re-covers everything if crop geometry ever changes again (delete files, let the job rebuild).

Admin trigger: `POST /adminapi/generateCrops` (admin-secured), same pattern as `/runClustering`, so a
backfill doesn't wait for midnight. Log a per-run summary (panos opened, crops written, failures).

#### (c) Display derivative

`serveBackupImage` currently sends the native file. That breaks two ways at 16384 px wide: per-view
bandwidth, and — decisively — Pannellum renders a single-image equirect as **one WebGL texture**, and
8192 is a common `MAX_TEXTURE_SIZE`, so native 16k panos won't render on many GPUs at all.

- New config key `pano.derived.images.directory` (default `.panos-derived`), same
  `<city-id>/<panoId[0:2]>/<panoId>.jpg` internal layout. Keep it **outside** `pano.images.directory` so
  `localBackupImageFile`'s extension scan can never resolve a derivative as the native file.
- Generated in the same reconciliation job: for any backup image wider than 8192, write an 8192×4096 JPEG
  derivative. Decode with `ImageReadParam.setSourceSubsampling` (e.g. 2×2 for a 16k pano → ~34 MB read),
  then the existing `ImageController.resize` helper. This also fixes any *existing* oversized GSV backups.
- `serveBackupImage`: serve the derivative when present, else the native file. `/backupImage` URL shape,
  signing, and metadata are unchanged.
- **Verification item for the implementer:** confirm the pano-viewer's marker math uses *ratios* of
  `pano_x/width` (metadata `width`/`height` stay native — they describe the label coordinate space, not
  the served pixels). Check `PannellumViewer.js` and `PanoMarker.js` before assuming.

#### (d) Attribution (CC BY-SA is an obligation, not just permission)

Richmond's imagery is `© jacobwhall / Mapillary (CC BY-SA 4.0)` — redistribution is permitted **with
visible attribution**. `pano_data.copyright` already flows into `buildBackupImageData`
(`utilitiesSidewalk.js`) → Pannellum metadata; verify the viewer actually renders it on self-hosted
imagery. For crops: surface the © line wherever the crop is shown at meaningful size (the Gallery
expanded-card/modal at minimum; the small card can defer). This likely means adding `copyright` to the
label metadata the Gallery already fetches — keep it minimal, but it ships in PR-A2, not "later".

#### PR-A2 tests

- `CropServiceSpec` (pure, no DB): geometry vs. golden fixtures; windowed-read seam stitch equals a
  full-decode crop on the synthetic pano.
- A DB-backed reconciliation spec: seed a label + `pano_data(has_backup = true)` + a pano file in a temp
  `pano.images.directory` → run the service → crop file exists; second run is a no-op; a label whose pano
  has no backup is untouched (negative control).

### CI trap — applies to both PRs

`.github/workflows/ci.yml` runs DB-backed specs via an **enumerated allowlist**
(`sbt 'testOnly controllers.HealthDashboardSpec service.HealthServiceSpec ...'`, ~line 277). A new spec
class not added there silently never runs and looks identical to passing. Add every new spec class to
that line.

### House rules the implementer must follow (from CLAUDE.md + hard-won project memory)

- `make scalafmt-fix` after Scala edits; the build is `-Xfatal-warnings` (an unused import fails CI).
- Frontend edits (if any JS is touched for attribution): `make eslint` etc. to zero; edit `src/`, never
  `build/`. Note `make eslint dir=.claude/worktrees/...` lints nothing — the lint targets `cd /home`.
- `sbt --client` from `/home` compiles the **main** repo — `cd` to the worktree first when working in one.
- Never start a second sbt/docker server; Jon's app is usually already running (`docker ps` first).
- Run new/changed tests locally (in the web container) before pushing.
- Comments state current behavior only — no "previously/renamed/no longer" narration (a hook flags it).
- Push branches freely; **no `gh pr create` and no merge without Jon's explicit OK.**
- Branch names start with the issue number once issues are filed.

---

## 2. Workstream B — sidewalk-auto-labeler (one PR)

New script **`upload_panos_to_ps.py`** at the repo root (sibling of `send_to_ps.py`), run as a separate
phase from metadata submission (decided: two-phase — the metadata POSTs are small and fast; a ~9 GB pixel
transfer needs its own retry policy).

- **Selection:** parse the run's `results.jsonl`; a pano is *label-bearing* iff it has ≥ 1 detection at
  `OPERATIONAL_CONFIDENCE` (import from `detectors`, mirroring `send_to_ps.py`'s filtering). Default =
  label-bearing only (decided, handoff §7.1); add `--all-panos` for the future widening. Resolve
  pano → file via the archive (`runs/<city>/panos/` + `index.csv`; `scripts/site_explorer.py` is prior
  art for the access pattern). Panos listed in `decayed.txt` are *included* — the archive predates the
  decay; that's the point.
- **Transfer:** per pano, `requests.put(f"{endpoint}/{pano_id}", data=<open file handle>)` — streams from
  disk, no multipart, no base64 — with headers `Authorization` (internal key), `X-Content-Sha256`
  (compute before sending), `Content-Type` from the file extension. Reuse `check_endpoint_security`
  (never the key in cleartext off-machine) and the `MAX_ATTEMPTS`/backoff pattern from `send_to_ps.py`.
- **Ledger:** sidecar keyed by **pano_id** (not line number), e.g. `<results.jsonl>.panos-uploaded`,
  appended after each 200/201. Treat server responses as: 201 → success; 200 "exists" → success (record
  it); 409 → log loudly, record in a separate conflicts list, **do not overwrite**, continue; anything
  else → retry then fail that pano and continue. Exit non-zero if any pano ultimately failed.
- **Flags:** `--endpoint` (default `http://localhost:9000/ai/panoImage`), `--limit N`, `--dry-run`
  (print selection + total bytes, send nothing), `--all-panos`.
- **Summary:** uploaded / already-present / conflicts / failed counts, total bytes, and coverage vs. the
  expected label-bearing count — so a resumed run's completeness is auditable.
- **Tests** (`tests/`, pytest, matching this repo's style): selection logic, ledger resume, endpoint
  security, response handling (mock `requests`). Pure logic separated from I/O per the repo's
  conventions.

---

## 3. Workstream C — labeler#42 (parallel track, deliberately out of this plan's critical path)

`camera_pitch`/`camera_roll` are null for every Mapillary record, and develop's
`util.misc.BACKUP_IMAGE_REQUIRED_FIELDS` gate (`utilitiesSidewalk.js`, requires numeric `cameraPitch`)
blocks Pannellum rendering without them. But **crops need no camera pose** — `pano_x/pano_y` are pixel
coordinates. So #42 gates only the interactive-viewer half of the definition of done, not the Gallery
card fix. Run the #42 measurement (Mapillary rotation convention from `computed_rotation`, per the
existing issue — it's a measurement task, not a parse) in parallel; do not serialize Workstreams A/B
behind it. When it lands, backfill the labeler's records and re-submit pano metadata (the `pano_data`
upsert path already exists).

---

## 4. Sequencing & rollout

1. **File issues first**: one in SidewalkWebpage ("AI-submitted labels have no imagery on non-GSV
   panoramas" — covers PR-A1 + PR-A2), one here ("Upload archived panoramas alongside submitted
   labels"), plus the promotion note on labeler#42. Reference `HANDOFF-pano-hosting.md` + this plan.
2. **PR-A1** (upload endpoint) → merge deploys to richmond-test automatically (test tracks develop).
3. **PR-B** (labeler upload script) — can develop in parallel against a local server.
4. **PR-A2** (crops + derivative + attribution).
5. **Full rehearsal on richmond-test**: all 9,091 records (use a *copy* of the JSONL — the `.submitted`
   sidecar is endpoint-agnostic) + ~9 GB pano upload + `/adminapi/generateCrops`. Check disk headroom on
   the test host first.
6. **Production** after the next release cut (v11.8.0 was 2026-08-10; prod deploys only on a tag):
   upload panos, submit with `--limit 20`, eyeball Gallery/Validate/Explore, then the remainder.

**Definition of done** (unchanged from the handoff): a Richmond AI label opened in Gallery shows a crop;
the same label renders in Validate/Explore from self-hosted imagery when Mapillary is unavailable (this
half gated on #42); `has_backup` is true for every label-bearing Richmond pano.

---

## 5. Decisions made (don't re-litigate) and what's still open

**Decided in planning — implement as specified:** reconciliation job, not inline generation; raw streamed
PUT, not base64 JSON; never-overwrite conflict semantics (409); crops are PNG at the existing
`cropFile` path; the job covers *all* crop-less labels with stored pixels, not just AI labels;
crop *sizing* isolated as a swappable v1 heuristic (resolution-normalized `predict_crop_size`) separate
from the settled seam/clamp mechanics; crop determination lives server-side, never with the submitter
(Jon, 2026-08-11); 8192×4096 derivative cap (WebGL texture limit); derivatives in a separate directory
root; label-bearing panos only for Richmond; two-phase upload; no DB evolution; after PR-A2,
SidewalkWebpage — not `CropRunner.py` — is the canonical display-crop geometry (§0.5).

**Reaffirmed 2026-08-12, after Mikey raised the counter-proposal.** Mikey leaned toward the labeler
uploading *crops only*, on backend-pressure grounds (10–20 MB per pano over the wire). Jon's call stands
at full panoramas, **stored at native resolution**, so the store can serve future CV experimentation.
The supporting measurements, for anyone re-opening this:

- **The 10–20 MB figure is a GSV figure, and GSV needs none of this** (`getImageUrl` already covers its
  cards). Measured per-pano averages from the archives: morgantown 0.8 MB, richmond 2.6 MB, annapolis
  2.8 MB, clovis 11.9 MB — against paterson 14.7 MB, são paulo 16.6 MB, gainesville 18.2 MB. Richmond's
  ask is 3,685 label-bearing panos ≈ 9 GB, once.
- **Crop-only saves far less than intuition suggests, and costs more requests.** Crop side over 436k
  production labels is p10/p50/p90 = 359/606/1500 px with 19.2% clamped at 1500, written as lossless
  PNG — a clamped crop can exceed a whole Richmond pano. Estimated 40–80% of the pano bytes, across
  9,526 crop PUTs instead of 3,685 pano PUTs.
- **The real form of the backend concern is decode memory in the shared 56-city JVM**, not transfer —
  which is why §1's PR-A2 mandates windowed `setSourceRegion` reads. Crop-only would move that decode
  onto the labeler's machine; that is its one genuine advantage.
- **Crop-only has no repair path.** pano-tools #83 (6,634 black-padded crops from before the #77 seam
  fix, never re-cut) is that failure mode already live in production, and Study 2 is *actively
  replacing* the sizing rule — so crops shipped today are stale by construction.

**Consequences of native-res storage, for implementers:** the display derivative (§1 PR-A2c) is
**mandatory, not an optimization** — Richmond's panos are 11000×5500 and would otherwise be served
past the common 8192 WebGL texture limit. Do not add an upload-side resolution cap. PR-A1's
`pano.upload.max-bytes` (64 MB) and the 2.2×10⁸-pixel decode-bomb guard both already clear native GSV
(16384×8192 = 1.34×10⁸) and Richmond (6.1×10⁷), so neither needs revisiting.

**Two stores, and why neither is redundant (settled 2026-08-12).** PS ends up with a native store
(`pano.images.directory`, existing) and a derived store (`pano.derived.images.directory`, new) — kept
under separate roots because `localBackupImageFile` resolves by extension scan and would otherwise
serve a derivative as the native file. The derived store is disposable data like crops: no backup, and
the reconciliation job rebuilds it. Only panos wider than 8192 need one — ~64% of a 3,000-record
Richmond sample (11000×5500, 12288×6144); the 4096×2048 and 5760×2880 panos are served natively. At
comparable compression the derived store estimates to ~30–40% of the native bytes, so **Richmond is
~9 GB native + ~3 GB derived**.

The lab's makelab2 archive is **not** made redundant by this, and does not make it redundant either —
they serve different consumers. makelab2 remains the research corpus (RampNet training, benchmark
bundles, re-inference). The PS-side native copy exists for what makelab2 structurally cannot reach:
unbounded-quality crop re-cutting as the sizing rule evolves (pano-tools #32 guarantees it will —
an 8192-capped store would cap every future re-cut permanently); durability of the pixel evidence
behind published labels, whose lifecycle is PS's, not the lab's; third-party access for anyone
consuming PS data; and PS-side server jobs (validator-ai, tagger-ai, depth work) that can read PS's
store and not the lab's.

**Resolved by that split: label-bearing panos only** (as §1 originally specified). "Future
experimentation" briefly seemed to argue for uploading every processed pano, but the negatives are
exactly what makelab2 already serves; every PS-side justification above is co-extensive with *panos
that carry labels*. PR-B keeps label-bearing as the default and `--all-panos` as the unused widening.

**Open — needs Jon, blocking nothing:**

1. **Storage owner/provisioning — not a blocker (Jon, 2026-08-12).** Richmond's ~12 GB all-in is noise
   and only a few cities are in scope. Worth telling Mikey the trajectory once so it is never a
   surprise: ~125 GB for all four Mapillary cities label-bearing, and ~3.7 TB if pano hosting ever
   extends to the GSV cities for decay defense.
2. Whether GSV cities ever get pano hosting (decay defense — file separately if pursued); exact visual
   treatment of the attribution line on cards.

**Contingent follow-on (do not build now) — labeler-supplied bounding boxes.** Considered and settled
with Jon (2026-08-11): **crop determination belongs to Project Sidewalk, not the submitter.** The
supporting facts: the current detector is a heatmap-peak model (`detectors/curb_ramp`) — detections in
`results.jsonl` are `{x_normalized, y_normalized, confidence}` points, so no boxes exist in Richmond's
archive and producing them would mean re-running inference over all 9,091 panos; and while RampNet is
training a YOLO-based model whose output *would* carry boxes (its gold labels are YOLO-format
`class cx cy w h`), it is not yet established that it beats the current approach. The server-side sizing
rule (v1 heuristic now, ground-truth-study-informed later) is therefore the canonical and durable path —
it must exist regardless, since crowd labels, Vancouver's ~64.8k AI labels, and any point-only submitter
never have boxes.

If a box-emitting detector ever ships and proves out, the upgrade is contained and pre-seamed: nullable
per-label bbox columns (an evolution — the box must be *persisted* to stay reconciliation-compatible),
`AiLabelsSubmission` reader fields, and PR-A2's sizing call site preferring padded-bbox over the
heuristic. Stored boxes would also have independent value (ray-aware clustering #4706, ML provenance).
File that issue when the trigger is real, not before.
