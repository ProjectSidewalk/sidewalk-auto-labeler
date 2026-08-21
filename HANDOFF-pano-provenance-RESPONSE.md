# Response: pano provenance landed — plus a bug your handoff helped catch

**From:** the SidewalkWebpage Claude Code session, 2026-08-07 (same day, pre-cut).
**Status:** implemented at exactly your recommended scope; PR #4809 open against develop, targeting the v11.8 cut.

## Decision & outcome

Shipped the JSONB-only scope, in v11.8 (Jon confirmed release-today intent). Nothing hit your abort criteria:

- `pano_data` is indeed 19 columns with a plain flat tupled `<>` mapping — the 20th column fit with no refactor.
- Evolution is **348** (347 was taken by the geodesic PR, merged 08-05): `ALTER TABLE pano_data ADD COLUMN source_metadata JSONB;`
- `PanoSubmission` gained `sourceMetadata: Option[JsValue]` read from `source_metadata` — your key name, verbatim; **no labeler change needed**, as you said.
- Catches #1 and #2 were both real and are both handled: the update branch writes the blob via a dedicated `updateSourceMetadata` query *only when the payload carries the key* — so your `--min-confidence 2.0` pano-only backfill works on already-recorded panos, and Explore submissions can never clear a stored blob. Both invariants are locked by a new DB-backed spec (`AiSubmissionSpec`) that does the double-POST round trip with `labels: []`.
- Catch #4 (the `.get` landmine) is filed as SidewalkWebpage **#4808**, not fixed today. Your read was exact, plus a third `.get` on the region lookup.

## ⚠ The thing you need to know: authenticated submission was broken

Your "submission works today" claim was **false for authenticated POSTs** — and the new spec caught it. Play's default CSRF filter protects any unsafe request carrying an `Authorization` header, and `send_to_ps.py` authenticates with `Authorization: Bearer`. So every keyed POST to `/ai/submitLabelsOnPano` returned **403 before reaching the controller** — Richmond could not have submitted a single record. (Historic Vancouver-WA submissions predate the Bearer auth, which is why this never surfaced.)

Fixed in the same PR (`+ nocsrf` on the route, the established pattern for internal-key ingests). No labeler change needed — but **do not start the Richmond submission run until v11.8 (or at least PR #4809) is deployed to the target server**, and treat a 403 from the ingest as "server predates the fix," not an auth problem.

## Bottom line for Richmond

Once v11.8 is live: run submission normally; provenance persists from the first record, no backfill pass needed. If anything was somehow submitted against a pre-#4809 server (it would have 403'd, so this is unlikely), the backfill you designed remains valid and spec-verified.
