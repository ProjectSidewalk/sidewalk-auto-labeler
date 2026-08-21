# Plan: pixels for AI labels — pano hosting + server-side crops

**Written:** 2026-08-11, from the sidewalk-auto-labeler session.
**Repos involved:** `SidewalkWebpage` (server) and `sidewalk-auto-labeler` (this repo).
**Status:** proposal for review. Nothing has been filed or built.

**Decision taken 2026-08-11 (Jon):** the Richmond production submission **waits** until both
halves of this land. That reverses my earlier "not a blocker" read, and the reasoning is sound —
see [§9](#9-why-waiting-is-the-right-call-and-what-it-costs).

---

## 1. The one-paragraph version

Project Sidewalk has no way to show the pixels behind an AI-submitted label on non-GSV imagery.
Crop images are written only by a browser POSTing a canvas snapshot, and the AI ingest has no
browser; the fallback that rescued the Vancouver pilot is GSV-only by construction. Richmond is
the first city to sit in the gap. The fix is to have the labeler ship the panorama it already
archived at native resolution into the store Project Sidewalk already has for self-hosted
imagery, and to have the server cut crops from that local file. That is one issue per repo, plus
one existing labeler issue promoted to a prerequisite.

---

## 2. The gap, traced in code

Everything below was read on `develop` on 2026-08-11.

**Gallery card image selection** — `public/js/gallery/src/cards/Card.js`:

```js
const primaryUrl = this.#cropUrl || this.#gsvImageUrl;
...
img.src = primaryUrl;
```

Both are `null` for a Mapillary AI label, so the browser assigns the literal string `"null"` and
issues `GET /null` → 404 → broken-image icon. That request was captured in the browser dev tools
on richmond-test with `Sec-Fetch-Dest: image`, confirming the path.

**Why `cropUrl` is null** — `app/service/PanoDataService.scala`:

```scala
def cropExists(labelId: Int, labelType: LabelTypeEnum.Base): Boolean =
  cropFile(labelId, labelType.name).exists()

def cropUrl(labelId: Int, labelType: LabelTypeEnum.Base): Option[String] =
  if (cropExists(labelId, labelType)) Some(signingService.signedUrl(s"/cropImage/${labelType.name}/$labelId"))
  else None
```

It is a bare filesystem check against
`<cropped.image.directory>/<city-id>/<LabelType>/crop_<labelId>.png`.

**Who writes that file** — exactly one code path in the whole application: `POST /saveImage` →
`ImageController.saveImage`, a `SecuredAction` that base64-decodes a canvas snapshot sent by
`public/js/explore/src/canvas/Canvas.js` / `label/Label.js`. It requires a browser and a user
session. **`CropRunner` is no longer used in production** (confirmed by Jon), so this is the only
producer.

**Why `gsvImageUrl` is null** — `PanoDataService.getImageUrl` opens with:

```scala
if (panoSrc != PanoSource.Gsv) return None
```

For GSV it returns a signed Google Static Street View URL built from the label's own POV. For
anything else there is no fallback at all.

**What the AI ingest does** — `AiController.submitLabelsOnPano` → `savePanoInfo` →
`submitAiLabelData` → `Ok("success!")`. `savePanoInfo` writes the `pano_data` row plus
`pano_link`/`pano_history` and nothing else. The insert is explicit about it:

```scala
panoDataTable.insert(PanoData(..., hasBackup = None, ...))
```

There is no background job that would fill the gap either. The scheduled actors are
`AuthTokenCleaner`, `CheckImageExpiry`, `Clustering`, `FunnelStat`, `GetAiValidations`,
`OsmWayRefresh`, `RecalculateStreetPriority`, `UserStat` — none downloads panoramas or generates
crops.

**Conclusion:** submitting metadata today produces a `pano_data` row and label rows. Nothing
fetches pixels, nothing crops. Every option requires a server change; there is no dormant
machinery to switch on.

---

## 3. Why the Vancouver pilot worked

Worth stating precisely, because the intuitive answer ("something we changed since") is wrong.

**No AI label has ever had a crop, in any city — including Vancouver's ~64,824.** The ingest has
never touched the crop path. Vancouver rendered because its imagery is GSV, so `getImageUrl`
synthesized a card image on demand from Google's Static Street View API.

| | crop from Explore canvas | GSV static fallback | card renders |
|---|---|---|---|
| Vancouver — GSV + AI | ✗ | **✓** | ✓ |
| Richmond — Mapillary + crowd | **✓** | ✗ | ✓ |
| **Richmond — Mapillary + AI** | ✗ | ✗ | **✗** |

Richmond is simply the first time both sources are absent at once. It is the imagery source, not
the code churn since the pilot.

**This affects four of our nine cities** — richmond, clovis, morgantown, annapolis — every
Mapillary run we have.

---

## 4. The design

Three options were considered. The chosen design is a merge of two of them.

| | option | verdict |
|---|---|---|
| A | Extend `getImageUrl` with a Mapillary equivalent (`thumb_2048_url`) | Rejected as the target — a full frame, not a POV crop; still needs a live fetch and a token; still no pixels for the interactive tools |
| B | Server generates crops on ingest by fetching imagery | Kept as the **floor**, not the target — see §4.3 |
| C | Labeler uploads crops | **Dropped.** Duplicates crop geometry across repos, and a crop alone doesn't serve Explore/Validate, which need the full pano |

### 4.1 Chosen: the labeler ships panoramas, the server cuts crops

Two moving parts, each in the repo that should own it:

- **`sidewalk-auto-labeler` supplies the panorama** at native resolution, from the archive it
  already maintains. Only this repo can fetch these pixels, and its archives are a snapshot taken
  at detection time.
- **`SidewalkWebpage` cuts the crop server-side** from that local file, using its own crop
  geometry, at ingest or in a job behind it.

Why this shape rather than the labeler cutting crops:

1. **Crop geometry stays in one implementation.** That logic is actively changing —
   `sidewalk-panorama-tools` PR #77 just landed a seam fix (`Image.crop()` zero-fills
   out-of-range boxes, so labels near the equirectangular seam got a black bar) and a vertical
   clamp. A second implementation in this repo would drift from it.
2. **No tokens, no rate limits, no network in the ingest path.** The server reads a local file.
3. **No decay.** Server-side generation from a *live fetch* is best-effort against imagery that
   is disappearing — Bend has already lost 8 panoramas. Our archives predate that loss. This is
   the one advantage that can never be recovered later.
4. **It fixes the interactive tools, not just the thumbnail.** Explore, Validate and the label
   card need the full panorama; the crop is only the Gallery card image.

### 4.2 Storage: native on disk, derived on serve

**Store native resolution.** Downgraded pixels in Project Sidewalk are a one-way loss: the
degraded copy becomes the copy people use, and for decayed panoramas there is no re-fetch. The
imagery is also CV-experiment input, not just a picture.

**Serve a derivative.** A 16384×8192 single JPEG is a heavy per-view load.
`ImageController.resize(img, newWidth, newHeight)` already exists, so the server can cache a
display-resolution derivative and cut the crop from the native file.

The store already exists — `PanoDataService.localBackupImageFile` resolves
`<pano.images.directory>/<city-id>/<panoId[0:2]>/<panoId>.<ext>` (jpg/jpeg/png), served through
`/backupImage/:panoId` behind a signed URL, with a `has_backup` column (evolution 319) surfaced
on the ops health dashboard and consumed by the Validate, Gallery, Label and Admin controllers.
This is a live feature, not a vestigial one.

### 4.3 Keep option B as the floor

Server-side generation from a live fetch is still worth having *behind* this, because it is the
only thing that reaches:

- Vancouver's ~64,824 existing AI labels, whose pixels we do not have
- any future submitter without a native-res archive
- any label whose panorama upload failed

It is a fallback, not the primary path.

---

## 5. Work split

### 5.1 SidewalkWebpage issue — "AI-submitted labels have no imagery on non-GSV panoramas"

- Accept a panorama upload on the internal-key path, writing into the existing self-hosted store
  and setting `has_backup`. Needs a size cap, a format allowlist and decode-bomb protection —
  `saveImage` already does base64 image intake, but from a session-authenticated browser, which
  is a different trust posture from a keyed API client.
- Generate the crop server-side from the local file, reusing the crop geometry that
  `sidewalk-panorama-tools` PR #77 settled, so AI and crowd cards look alike.
- Cache a display-resolution derivative via the existing `resize` helper.
- **Make it backfillable.** Richmond's rows may already exist by the time this ships, and the
  ingest currently returns a bare `Ok("success!")` with no label IDs — so a design that only
  accepts pixels inline with the original submission can never repair an earlier city. Either
  return created label IDs, or expose a keyed endpoint that resolves a label by
  `(pano_id, pano_x, pano_y)`.
- Optional follow-on: extend the same crop generation to GSV AI labels for consistency. Not
  required — the Static API already covers their cards.

### 5.2 sidewalk-auto-labeler issue — "Upload archived panoramas alongside submitted labels"

- Add an upload path to `send_to_ps.py` (or a sibling script) that sends the native-res panorama
  for every label-bearing pano.
- Source the pixels from the makelab2 archive
  (`/projects/makeabilitylab/sidewalk-auto-labeler/runs/<city>/panos`), which is already verified
  1:1 against `results.jsonl` with an `index.csv` and a `decayed.txt`. `scripts/site_explorer.py`
  already reads that archive, so there is prior art for the access pattern.
- Must be **resumable and idempotent**, in the same spirit as the `.submitted` sidecar — a
  ~9 GB upload for Richmond will not complete in one clean pass every time.
- Decide whether uploads ride with `send_to_ps.py` or run as a separate phase. A separate phase
  is probably right: the metadata POSTs are small and fast, the pixel transfer is neither, and
  coupling them makes one retry policy serve two very different failure modes.

### 5.3 Prerequisite: labeler#42, promoted

`utilitiesSidewalk.js` declares:

```js
util.misc.BACKUP_IMAGE_REQUIRED_FIELDS = ['width','height','lat','lng','cameraHeading','cameraPitch'];
```

and `backupImageDataIsComplete` rejects nulls (per #4804). **Our Mapillary records write
`camera_pitch: null` and `camera_roll: null`**, so the self-hosted-panorama path cannot render
our panoramas even once the pixels are there.

The cause is an API-shape difference, not a considered decision. `sources/gsv.py` gets scalars
from streetlevel and records them (`camera_pitch: 1.51, camera_roll: 1.40` on a Paterson record);
Mapillary exposes rotation only inside `computed_rotation`, an axis-angle vector, which
`sources/mapillary.py` deliberately leaves unparsed:

> Pitch/roll only exist inside `computed_rotation` (an axis-angle vector) and are left null.

That is exactly labeler#42. It was filed as a fusion-accuracy improvement; it is now also a
**submission** prerequisite. Note #42 is explicit that the rotation convention must be *measured*
(re-running the pose ablation on a Mapillary city), not assumed — the GSV ablation proved those
equirectangulars are already gravity-rectified, and Mapillary is a different pipeline. So this is
a measurement task, not a one-line parse.

---

## 6. Sizing

Computed from the local runs (label-bearing = at least one detection at
`OPERATIONAL_CONFIDENCE`), with per-pano byte sizes derived from the measured archive totals in
`docs/production-deployment.md`. **The GB column is an estimate from the archive average, not a
measured per-file sum.**

| city | panos | label-bearing | labels | archive | est. label-bearing bytes |
|---|---:|---:|---:|---:|---:|
| **richmond** | 9,091 | **3,685 (40.5%)** | 9,526 | 23 GB | **~9 GB** |
| annapolis | 53,232 | 14,884 (28.0%) | 29,188 | 143 GB | ~40 GB |
| clovis | 72,776 | 6,092 (8.4%) | 8,961 | 847 GB | ~71 GB |
| morgantown | 51,692 | 5,795 (11.2%) | 10,482 | 39 GB | ~4 GB |

**Richmond alone is about 9 GB** — the deployment actually on the table. All four Mapillary
cities together are roughly 125 GB.

The GSV cities never enter this: `getImageUrl` already covers their cards, so Bend's 1.2 TB and
Gainesville's 626 GB are out of scope. That is what makes native resolution affordable here.

---

## 7. Open questions for review

Flagged rather than decided, because each changes the scope materially.

1. **Label-bearing panoramas only, or every processed panorama?** Hosting only label-bearing
   panos is 40.5% of Richmond. Hosting all of them would give Explore continuity when Mapillary
   imagery is unavailable, at 2.5× the storage. My inclination is label-bearing only to start.
2. **Do GSV cities ever get panorama hosting?** Not needed for cards, but it is the only defence
   against decay — Bend has already lost 8 panoramas permanently. That is a much larger storage
   conversation and probably its own issue.
3. **Where does the storage live, and who provisions it?** All 56 city instances share one
   server. ~9 GB for Richmond is modest; ~125 GB for all four Mapillary cities is a real request
   that needs an owner.
4. **Inline vs. two-phase upload** (§5.1's backfill point). Inline is atomic; two-phase keeps the
   metadata POSTs small and makes retry and backfill tractable at 9.5k scale. I lean two-phase.
5. **Does hosting change the licensing posture per source?** Richmond's panos carry
   `© jacobwhall / Mapillary (CC BY-SA 4.0)`, so redistribution is explicitly permitted and the
   Mapillary cities — precisely the ones with the gap — are the straightforward case. GSV
   hosting, if it ever happens under question 2, is a separate call.

---

## 8. What is already done — not at risk, and not part of this plan

Established earlier in this session, so the reviewer knows what this plan does *not* need to
revisit:

- **All three original Richmond blockers shipped in v11.8.0**, verified live against
  `sidewalk-richmond.cs.washington.edu`: the `richmond-va` submission flag (#4803), the CSRF fix
  that had silently 403'd every keyed POST (#4809), and `pano_data.source_metadata` (evolution
  348). A Bearer POST now returns 401, not 403, which is the deployed/not-deployed probe.
- **The label lat/lng estimator was replaced in the same release** (#4819, `approximation3`). I
  recomputed all 16 test submissions against the documented blend: **worst deviation 0.000 m**.
  Resolution independence confirmed directly (same `y_norm` at 2048 px and 6144 px both placed at
  10.73 m), and the saturating tail held a 1.055°-depression detection at 22.72 m where a raw
  cotangent would have thrown it 127 m. This was the one thing genuinely worth waiting for, and
  it is already in.
- **The test submission is green.** 6 records / 16 labels on richmond-test, all confirmed via
  `/v3/api/rawLabels`, attached across 10 street edges in 5 regions — so `pano_source: "mapillary"`
  works end to end through the POV math, street snapping and the `pano_data` upsert.
- **Richmond's data is preflight-clean**: no nulls in any field the `.get` landmine (#4808)
  touches; all 9,091 records carry `source_metadata`, forwarded verbatim by `transform_pano`;
  9,083 of 9,091 panos fall inside Project Sidewalk's 9 Richmond regions and the 8 outside carry
  zero labels; `results.jsonl.submitted` does not exist, so the endpoint-agnostic sidecar trap is
  not armed.
- **labeler#22 (pano provenance lands nowhere) is resolved** by #4809 and can be closed.
- **The operating point stays at 0.55.** Richmond's run predates the storage floor — every stored
  detection is already ≥ 0.55 — so adopting 0.30 would mean re-running inference over all 9,091
  panos, and the re-triaged world-space gain is only +0.4 to +3.2 recall points.
- **Submit raw detections, not fused sites.** 9,526 detections ↔ 1,570 fused sites. Fusion drops
  15% of detections before it starts, and SidewalkWebpage#4706's proposed ray-aware clustering
  needs exactly what raw submission preserves: per-label `pano_x` plus the panorama's pose.

---

## 9. Why waiting is the right call, and what it costs

**The argument for waiting is stronger than I first gave it credit for.** I called this
non-blocking because the labels are correct and render everywhere except the Gallery thumbnail.
But if the server change lands in an inline-only shape (§5.1), labels submitted first can never
be repaired — 9,526 Richmond labels would be permanently pixel-less. Submitting after removes
that risk entirely rather than mitigating it with a backfill requirement.

**Waiting is also cheap here**, which is the part that makes it easy. Mapillary imagery does not
decay the way GSV does, our archive is already captured and verified, and nothing about the run
degrades while it sits.

**The real cost is the release cycle.** Test deploys from `develop` on merge, but production
needs a full cut — version bump, version-table evolution, a `vX.Y.Z` tag on `master`. v11.8.0 was
cut on 2026-08-10, so the next window may be some weeks out. Richmond's submission is now bound
to that cadence rather than to a config toggle. That is the trade being made, and it should be
made with open eyes.

**One mitigation worth considering:** run the full Richmond submission against **richmond-test**
once the labeler half is ready, before production. It exercises the whole 9,091-record path,
including the panorama upload, at real scale — and test tracks `develop`, so it will have the
server change weeks before production does. Use a copy of the JSONL, since the `.submitted`
sidecar is endpoint-agnostic.

---

## 10. Sequencing

1. File both issues; add the promotion note to labeler#42.
2. **labeler#42** — measure the Mapillary rotation convention, parse `computed_rotation` into
   `camera_pitch`/`camera_roll`. Blocks the panorama path from rendering.
3. **SidewalkWebpage** — upload intake, server-side crop generation, served derivative. Merge to
   `develop` deploys it to richmond-test.
4. **sidewalk-auto-labeler** — the upload phase, resumable.
5. Full-scale rehearsal against richmond-test: all 9,091 records plus ~9 GB of panoramas.
6. Production, after the next release cut: `--limit 20`, eyeball, then the remainder.

**Definition of done:** a Richmond AI label opened in Gallery shows a crop; the same label in
Validate and Explore renders from self-hosted imagery when Mapillary is unavailable; and
`has_backup` is set for every label-bearing panorama in the city.

---

## Claim provenance

- **Read in code on `develop`, 2026-08-11:** everything in §2, §3, §4.2, §5.1 and §5.3's
  `BACKUP_IMAGE_REQUIRED_FIELDS` quote.
- **Measured live, 2026-08-11:** the v11.8.0 version string and the 401-not-403 probe on
  richmond prod; the 6-record/16-label test submission and its `/v3/api/rawLabels` verification;
  the 0.000 m estimator agreement; the region-containment and null-field preflights.
- **Computed from local run data:** every figure in §6's first three columns. The GB column is an
  estimate from archive averages, not a per-file sum.
- **From Jon, this session:** that `CropRunner` is no longer used in production, and the
  native-resolution requirement.
- **Not verified:** that the makelab2 archive is reachable and complete for the three Mapillary
  cities other than Richmond (Richmond's was verified 1:1 zero-decay on 2026-07-28).

---

*Prepared with Claude Code (claude-opus-5[1m]).*
