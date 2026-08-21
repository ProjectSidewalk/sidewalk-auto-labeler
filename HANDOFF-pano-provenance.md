# Handoff: give submitted pano provenance somewhere to land (SidewalkWebpage)

**For:** the SidewalkWebpage Claude Code session (repo `/home/jonf/git/SidewalkWebpage`).
**Written:** 2026-08-07, from the sidewalk-auto-labeler session, on v11.8 release day.
**Goal:** decide whether a `pano_data` provenance column ships in v11.8 (cut today) or in
develop immediately after — and if in, land it at the scope below.

Jon leans toward shipping it in v11.8. This note argues that is reasonable **only at a
deliberately narrow scope** (one JSONB column), gives the abort criteria, and flags the
claims you must re-verify in your repo before deciding. You have the code; the final call
is yours and Jon's.

**Hard constraint either way: cutting v11.8, and SW#4803 (the `richmond-va` AI-submission
flag), do not wait on this.** That PR is the release-critical piece; this is an
enhancement with a proven recovery path (see "Why Richmond doesn't need it").

---

## What the labeler already sends — verified today, labeler-side

No labeler change is needed for any version of this. `send_to_ps.transform_pano`
(send_to_ps.py:84-115) deliberately forwards every key it doesn't rename, so each
submitted pano block **already carries**: `camera_make`, `camera_model`, `camera_type`,
`sequence_id`, `quality_score`, and `source_metadata` (the verbatim Mapillary Graph API
blob minus the expiring thumb URL). Your `PanoSubmission` reader is path-based and
silently drops them today.

Re-verified today across all **9,091 Richmond records**: all six fields non-null in every
record, and **zero** records with a null `width`/`height`/`lat`/`lng`/`camera_heading`
(the `.get` landmine in `submitAiLabelData` — see catch #4).

**The fact that changes the design debate:** `source_metadata` is a **verbatim superset**
of the five scalars (sources/mapillary.py:56-85):

| top-level field | = blob key |
|---|---|
| `camera_make` | `make` |
| `camera_model` | `model` |
| `camera_type` | `camera_type` |
| `sequence_id` | `sequence` |
| `quality_score` | `quality_score` |

So storing the blob alone loses **nothing**, and promoting scalars later is pure SQL in a
calmer release — `UPDATE pano_data SET camera_make = source_metadata->>'make'` — with no
re-POST and no labeler involvement.

## Recommended v11.8 scope: ONE nullable JSONB column

```sql
ALTER TABLE pano_data ADD COLUMN source_metadata jsonb;
```

Nullable, no default, no data migration — in Postgres this is a metadata-only change (no
table rewrite), the safest evolution class there is. Plus:

- `PanoSubmission` reader: one optional `JsValue` field.
- Write it in the AI insert path **and** its already-exists/update branch (catch #1).
- That's it. No frontend, no scalars, no indexes yet.

**Why not the 5-scalars + JSONB shape on release day:**

1. `pano_data` has 19 columns. +6 = 25, past **Slick's 22-column flat mapping limit** —
   if the table mapping is a plain tupled `<>` case-class mapping, the wide version
   forces an HList/nested-tuple refactor mid-release-day. **Check the mapping style
   before even considering the wide shape.** 19 + 1 = 20 stays clear.
2. Wide-vs-JSONB is a genuine design call; the superset fact above means you don't have
   to settle it today to capture 100% of the data.
3. GSV records carry no `source_metadata` today (it's Mapillary-source-only) — nullable
   handles that, and the column is ready if the labeler ever emits a streetlevel
   metadata blob for GSV cities.

## Catches — each is a claim from the other session's read of your repo; re-verify in code

1. **The upsert's update branch.** `savePanoInfo` upserts, but the update path
   (`updateFromExplore`) reportedly writes only
   lat/lng/heading/pitch/roll/address/expired/timestamps. If the AI-path update doesn't
   include the new column, the backfill pass **silently no-ops** on panos already
   inserted (e.g. Richmond's, once labels are submitted). Test: POST the same record
   twice against a dev instance → column populated after the second POST.
2. **The inverse:** Explore-path updates must NOT null the column out — crowd payloads
   have no `source_metadata`. If insert/update share a column list, an Explore update
   after an AI submission would erase it.
3. **Backfill semantics** (the recovery path if this ships after Richmond's labels):
   `submitAiLabelData` reportedly has no dedup — re-POSTing a record duplicates every
   label — but a record with `labels: []` upserts the pano and inserts nothing. The
   labeler half is verified today: `send_to_ps.py --min-confidence 2.0` still POSTs
   every record, with empty labels (send_to_ps.py:118-140 — "checked, nothing found"
   records are deliberately submitted). So a pano-only provenance backfill over
   Richmond's file is one command, idempotent, using flags that exist today.
4. **Not for today, but file it:** `submitAiLabelData` reportedly calls `.get` on
   `pano.width/height/lat/lng/cameraHeading` — a null is a 500, not a validation error.
   Richmond is clean (0/9,091, re-verified), but it's a landmine for any future
   submitter. Worth a SidewalkWebpage issue, not a release-day fix.

## Decision framing

**Ship in v11.8:** additive nullable JSONB is about the lowest-risk evolution possible;
the first-ever production AI submission (Richmond) gets full provenance from day one with
no backfill pass; the next deploy window is otherwise unknown.

**Punt to develop-after-cut:** any release-day schema change across 56 production
instances is nonzero risk on top of a release that already carries the AI flag; the
backfill path above means nothing is unrecoverable, ever.

**Suggested rule — timebox with explicit abort criteria.** Attempt the JSONB-only scope.
If it stays within [1 evolution + reader field + insert/update writes + one double-POST
round-trip test] and every gate passes before the cut: land it. The moment it demands a
Slick mapping refactor, a design debate, an `updateFromExplore` restructure, or the
Evolutions-lint gate pushes back: stop, land it in develop right after the cut, and
backfill Richmond with the `--min-confidence 2.0` pass later.

## Why Richmond doesn't need it

Submission works today — the provenance rides along and is dropped. Every dropped byte is
still in `runs/richmond/results.jsonl` and re-submittable as a pano-only pass whenever
the column exists. This is day-one-completeness vs. a one-command backfill, not
blocker vs. non-blocker.

## Claim provenance

- **Verified today in the labeler repo/data:** everything in "What the labeler already
  sends"; the empty-labels submit behavior; the superset table.
- **From the other session's read of SidewalkWebpage (re-verify):** 19-column count,
  upsert/update semantics, no-dedup on `submitAiLabelData`, the `.get` landmine,
  `labels: []` inserting nothing.
