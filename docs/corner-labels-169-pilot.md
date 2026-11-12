# #169 test-server pilot: clearance sheet (residential corners only)

Prepared 2026-11-11 on branch `corner-pilot-169` (= `corner-labels-169` + a merge of
`store-guard-label-type`, draft PR #179). **Nothing has been sent.** Each line below is for
Jon to approve, change or strike. One approval covers one operation, test servers included.
Background and the rule's result: [corner-labels-169.md](corner-labels-169.md), "Result" and
"Test-server pilot".

## 0. Blocker: there is no running Vancouver test stage

Read-only, makelab1, 2026-11-11T22:16Z (one batched `wsl-ssh.ps1 makelab1 run`):

- `/www/sidewalk/tools/city-configs/.sidewalk-vancouver-test` is **dot-disabled** (dated
  2025-09-19); `sidewalk-vancouver` (prod) is enabled. No `sidewalk-parent-process.sh
  sidewalk-vancouver-test` is running. 11 test stages are running, among them
  `sidewalk-juneau-test` (config dated 2026-10-29), `-richmond-test`, `-laurens-test`.
- The public URL agrees: `https://sidewalk-vancouver-test.cs.washington.edu/v3/api/cities` →
  **503** (2026-11-11, twice); `sidewalk-vancouver` → 200; `sidewalk-juneau-test` → 200.
- The **test database still holds a `sidewalk_vancouver` schema** (port 6432, `sidewalk_test`):
  64,814 labels, **all of them the AI's CurbRamp labels** (`rampnet-model`, 1.0.0), newest
  `time_created` 2025-09-18 19:57 PDT, **no crowd label at all** and no NoCurbRamp label. Prod's
  schema has 65,183 labels (newest 2026-10-11). So the test schema is a 2025-09 copy taken
  around the AI pilot.
- `sidewalk_login` differs between the two databases: `jonfroehlich` is
  `2d8335ef-daff-4382-afe2-7e1eb6567fd9` on test and `549187e0-82c9-4014-a48d-31f18083d575` on
  prod. `SidewalkAI` is `51b0b927-3c8a-45b2-93de-bd878d1e5cf4` on both. **Key every query on
  usernames, never on a person's `user_id` copied from prod.**

**To do (Jon, from the makelab1 sidewalk dashboard; updated 2026-11-11, an earlier version
routed this through Mikey):** start `sidewalk-vancouver-test`, as was done for Juneau on
2026-10-29. Decision 1 below: refresh the
test schema from prod first, or not.

Prod is **not** a substitute. The issue confines the pilot to a test server.

## 1. The file

| | recommended: 200 labels | alternative: all residential |
|---|---|---|
| path (under `runs/vancouver/corner_labels169_split/na_discount_sw/pilot_residential/`) | `results.pilot200.jsonl` (committed) | `results.residential.jsonl` (4.7 MB, untracked; rebuild below) |
| sha256 | `d38c9f3b1fba530bac9d6b6ecfcacf9403ebe5b55f116bf18b5a35a7035da0c6` | `a4dce3575e852ff9c4389ff094fbfd156c946f6bd003bcca905a14900ba6aa0c` |
| lines (= panos, = POSTs) | 197 | 1,717 |
| labels | 200 | 1,984 |
| bytes | 506,047 | 4,734,498 |
| pilot order positions | 1-200 | 1-1,984 |
| export record | `results.pilot200.export.json` | `results.residential.export.json` |

- **Source.** Both files come from the split `na_discount_sw` build (`proposals.jsonl` sha256
  `1562e076…`, built at `30168fd`). The labels are its **residential** stratum: 1,984 of the
  3,070 proposals. The 1,052 arterial and 34 signalised proposals are out of this pilot.
- **Order.** `pilot.csv` is one seed-169 permutation of the 1,984 residential corners
  (`sample --stratum residential --n 100 --reserve 20 --send 200`; sha256 `157ff268…`). The
  200-label file is its first 200 positions. Lines follow that order, so `--limit` sends the
  head of the draw.
- **Payloads.** Every payload has `label_type: NoCurbRamp`, `model_id:
  corner-rule@30168fddeb6e`, `model_training_date: 11-05-2026`, `api_version: 1.0.0` and
  `confidence: 1.0`.
- **Endpoint.** `https://sidewalk-vancouver-test.cs.washington.edu/ai/submitLabelsOnPano`.
- **`--min-confidence`.** Omit it. The NoCurbRamp spec's default is 0.5, and every label is 1.0.
- **Branch.** Send from `corner-pilot-169` (or from `main` once #179 is merged). `main` without
  #179 refuses both files in `check_store_built`. The branch never changes the labels:
  `model_id` names the build commit, `30168fd`.

**What the 200 hold** (`pilot_stats.json`, from `scripts/corner_pilot_stats.py`):

| | first 120 (rate + reserve) | 200 file | 1,984 file |
|---|---:|---:|---:|
| labels / panos | 120 / 119 | 200 / 197 | 1,984 / 1,717 |
| range to the corner, p10 / median / p90 (m) | 6.9 / 10.2 / 18.2 | 6.8 / 9.9 / 17.1 | 6.8 / 10.1 / 17.5 |
| bounded by a phantom leg (far side of a T) | 81 (0.68) | 130 (0.65) | 1,354 (0.68) |
| still `wide` | 0 | 1 | 11 |
| inventory: none / only `NA` with no `RAMPTYPE` | 100 / 20 | 166 / 34 | 1,601 / 383 |
| pano dims 16384 / 13312 / 3328 wide | 109 / 11 / 0 | 184 / 15 / 1 | 1,844 / 135 / 5 |

- **Jon's earlier ratings.** 6 of the 66 residential corners he rated in
  `corner_gallery169_split` fall in the 200, at positions 133, 139, 140, 148, 150 and 175. None
  is in the first 120, so the rated prefix is all new. They stay in the file.
- **Store panos.** 196 of the 197 panos are PS store panos (`source_detail: ps_store`). Their
  pano block is prod's own `/backupImage/<id>/metadata` (scripts/detect_from_store.py), so the
  positions sent are prod's.
- **The one exception** is position 200, `vancouver:res:n4600722929#1`, on GSV pano
  `7s0_v8TVpDa7WsPBDCPpmw`: 3328 px wide, captured 2007-08, fetched from Google for #241.
  `/backupImage/7s0_v8TVpDa7WsPBDCPpmw/metadata` on prod answers 404 ("No stored image"), so
  this label will have no store imagery and no crop (decision 6).

## 2. Guards (verbatim in `dry_run_summary.txt`)

`python scripts/pilot_guards.py <file> --endpoint <the test endpoint>` calls every guard a real
send runs, in send order, and sends nothing. Both files give the same verdicts:

| guard | verdict |
|---|---|
| `check_endpoint_security` | PASS |
| `check_model_provenance` | PASS |
| `check_store_built` | PASS: "196 record(s) … have source_detail "ps_store" … but it carries only NoCurbRamp labels; the store's live AI labels are all CurbRamp, so none is duplicated (type-aware store guard, #169)." (1,713 for the 1,984 file.) **Refuses on a branch without #179.** |
| `check_detection_decode` / `_border` | PASS: argmax / exclude |
| `file_label_types` | `['NoCurbRamp']`, so one POST per line and the legacy sidecar |
| `check_resume_state` | PASS (fresh: no sidecar, no record) |
| `check_position_state` | PASS (GSV, not gated) |
| `check_live_positions` | PASS, but **blind here**: it reads only submission records in this directory, and the 2025-09 CurbRamp campaign was sent from elsewhere. See section 6 for what the server does with the pano rows. |

`send_to_ps.py --dry-run` on both: 197 and 1,717 records, 0 errors, 0 labels below 0.5. A dry
run skips the guards, hence the table.

## 3. Commands, in order (Jon runs them; Claude does not)

```bash
cd D:/Git/labeler-wt/pilot169          # branch corner-pilot-169, or main after #179
PY=C:/Users/jonf/anaconda3/envs/sidewalk-auto-labeler/python.exe
P=runs/vancouver/corner_labels169_split/na_discount_sw/pilot_residential
EP=https://sidewalk-vancouver-test.cs.washington.edu/ai/submitLabelsOnPano
sha256sum $P/results.pilot200.jsonl     # must print d38c9f3b...a0c6
# 1. dry run (contacts nothing) and the guards
$PY send_to_ps.py $P/results.pilot200.jsonl --endpoint $EP --dry-run > /dev/null
$PY scripts/pilot_guards.py $P/results.pilot200.jsonl --endpoint $EP
# 2. smoke: 3 POSTs (positions 1-3), PS_INTERNAL_API_KEY in the environment or ./.env
$PY send_to_ps.py $P/results.pilot200.jsonl --endpoint $EP --limit 3
#    -> run the SELECT in section 4 (expect 3), open /label/<id> for each, check below
# 3. the rest (resumes from the sidecar; 194 more POSTs)
$PY send_to_ps.py $P/results.pilot200.jsonl --endpoint $EP
```

- **What to look at after the smoke send.** Run the section 4 SELECT; it should return 3
  labels. Open `https://sidewalk-vancouver-test.cs.washington.edu/label/<label_id>` for each.
  The marker should sit at the curb corner, where a ramp would be, on the pano named in
  `pilot.csv` positions 1-3. The run should end with 0 errors and print `Submission record:
  results.pilot200.jsonl.submission.json`.
- **Files a real send writes, beside the file:** `results.pilot200.jsonl.submitted` (line
  numbers) and `results.pilot200.jsonl.submission.json`. Commit the `.submission.json`, as for
  every campaign.
- **A 409 for a pano is the server's duplicate guard.** It refuses a pano that already holds a
  live AI label of the same type (`ExploreService` `countLiveLabelsOnPano(pano, aiUser,
  labelType)`), and `send_to_ps.py` sends no `overwrite`.
- **The 200 file and the 1,984 file are alternatives, never a sequence.** After the 200, the
  1,984 file would 409 on the 197 panos already sent. Any later extension would 409 on the 45
  pilot panos that hold 50 more residential corners. An extension needs `overwrite` or a
  per-pano merge, which is a separate decision.

## 4. Expected rows and the read-only check

After the full 200 send, the test `sidewalk_vancouver` schema should hold:
- **200** live `label` rows with `user_id` = SidewalkAI, `label_type` NoCurbRamp, and
  `label_ai_info.model_id` = `corner-rule@30168fddeb6e`, against **0** NoCurbRamp labels of any
  kind before (measured 2026-11-11);
- 200 `label_point` rows;
- `pano_data` rows for the 197 panos: inserted where missing, and upserted where present
  (section 6).

```sql
-- psql -X -p 6432 -d sidewalk_test, with PGOPTIONS='-c default_transaction_read_only=on'
SET search_path = sidewalk_vancouver, sidewalk_login, public;
SELECT l.label_id, l.pano_id, lp.pano_x, lp.pano_y, l.time_created, l.deleted
FROM label l JOIN label_point lp USING (label_id) JOIN label_ai_info a USING (label_id)
WHERE l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4' AND l.label_type = 'NoCurbRamp'
  AND a.model_id = 'corner-rule@30168fddeb6e'
ORDER BY l.label_id;
-- expect 200 rows, deleted = f, 197 distinct pano_id; save as label_id,pano_id,pano_x,pano_y
-- (psql --csv) for `corner_labels.py pilot-links`; every (pano_id, pano_x, pano_y) must be
-- one of the file's PS pixel keys (pilot-links refuses a duplicate key).
```

Every label id is a **`vancouver:` test-database id**. It is not a prod id, and it is
meaningless without the city and the stage.

## 5. Rollback (DRAFT; not run; Jon's clearance needed)

AI labels are insert-only through the API (SW#5382). Retiring the pilot means soft-deleting it
in the test database. The draft below touches exactly the pilot's labels and refuses to commit
on any other count:

```sql
-- DRAFT. psql -X -p 6432 -d sidewalk_test (peer auth, from a makelab1 shell). Fill the window
-- from the send's first and last timestamps in results.pilot200.jsonl.submission.json.
BEGIN;
SET search_path = sidewalk_vancouver, sidewalk_login, public;
DO $$ DECLARE n int; BEGIN
  UPDATE label l SET deleted = TRUE
  FROM label_ai_info a
  WHERE a.label_id = l.label_id
    AND l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4'
    AND l.label_type = 'NoCurbRamp'
    AND a.model_id = 'corner-rule@30168fddeb6e'
    AND l.time_created >= '<first_submission_utc>' AND l.time_created <= '<last_submission_utc>'
    AND NOT l.deleted;
  GET DIAGNOSTICS n = ROW_COUNT;
  IF n <> 200 THEN RAISE EXCEPTION 'expected 200 pilot labels, touched %', n; END IF;
END $$;
COMMIT;   -- reversible: the same UPDATE with deleted = FALSE
```

TODO before use: `label` also has `deleted_by`, `deleted_at` and `deleted_source`, whose types
and conventions were not read. Set them the way the server's own overwrite path does
(`deleted_source` reads `SidewalkAI` there), or leave them null and say so. Pano rows the send
upserted are not rolled back (section 6).

## 6. What the server does with a submission (SidewalkWebpage `develop` 428341d, read 2026-11-11)

- **NoCurbRamp is accepted.** `label_type` parses as any `LabelType` name, and no code on the
  submit path is keyed to CurbRamp. The mission is one AI explore mission per region, not per
  type. The duplicate check is per type, so existing AI CurbRamp labels on a pano do not block
  it. The submit path is gated only by the city flag `ai-label-submission-enabled`, which is
  true for `vancouver-wa`. Only CurbRamp payloads appear in the server's tests, so NoCurbRamp
  is accepted by the code but has never been exercised by a test.
- **`pano_data` is upserted.**
  - `INSERT … ON CONFLICT (pano_id) DO UPDATE SET lat = COALESCE(EXCLUDED.lat, pano_data.lat)`,
    and the same for `lng`, `camera_heading`, `camera_pitch` and `camera_roll`. So the pano
    position and pose sent overwrite the stored ones.
  - `width`, `height` and the tile sizes keep the stored value.
  - For the 196 store panos, the values sent are prod's own (from `/backupImage` metadata). The
    test schema's 2025-09 rows, where present, would be moved to prod's values.
  - A label's lat/lng is computed once, at insert, from the payload's pano position and
    width/height (`PanoDataService.toLatLng`). Existing labels never move.
- **AI validation.** The nightly AI validator **never votes on labels created by the AI
  account, of any type**. `LabelTable` filters `ur.role =!= Role.Ai` after `labelType inSet
  aiLabelTypes`, and `aiLabelTypes` includes NoCurbRamp. So the pilot's labels get human
  validations only, and there is nothing to switch off.
- **Crops.** The nightly crop job cuts a crop for every live label of any type whose pano has
  a store image. The 2007 GSV pano at position 200 has none.

Source of these reads: a delegated read of develop with file and line citations. Spot-checked
directly: the `submitLabelsOnPano` and `gallery?labelIds` routes, `MaxLabelIds = 500`, the
`ON CONFLICT … COALESCE(EXCLUDED.lat …)` and `width = COALESCE(pano_data.width …)` lines, and
`LabelTable.scala:3213`. Whether a re-enabled test stage runs this develop commit is not known.

## 7. Rating protocol (declared amendment to #169's "100-label sample")

The issue says: "a lead rates a seeded 100-label sample on the live site". **Amendment,
declared before any send:** the lead rates **in the seeded order** and stops by a
pre-registered sequential rule. A can't-tell is skipped and absorbed by a 20-label reserve.
Recorded in `pilot.json` `stop_rule`; implemented and tested as `corner_labels.py
pilot-score`.

- **What is rated.** The live label, on the test site. Agree = a NoCurbRamp label belongs
  there. Disagree = it does not, for any reason: a ramp serves the corner, no sidewalk, or
  misplaced. Unsure = can't tell. This measures **label precision**, which is broader than the
  test's corner-absence precision. The bar is the same: precision ≥ 0.90 and Wilson lower
  bound ≥ 0.85.
- **The order.** `pilot.csv` positions 1-100 are `rate` and 101-120 are `reserve`. If 20
  can't-tells exhaust the reserve, rating continues in order inside the 200 and the
  can't-tell share is reported. Only the lead's **first** validation of a label counts. A
  rating after an unrated earlier label is refused: no skipping ahead.
- **Looks** at 40, 60 and 100 decided labels. The Wilson bound uses z = 2.394, which is the
  one-sided 0.025 of the rule split evenly across 3 looks (Bonferroni, conservative):

| look (decided) | PASS if wrong ≤ | Wilson lower, all right | unadjusted (z 1.96) would allow |
|---:|---:|---:|---:|
| 40 | 0 | 0.875 | 1 |
| 60 | 2 | 0.913 (2 wrong: 0.859) | 3 |
| 100 | 6 | 0.946 (6 wrong: 0.856) | 8 |

- **FAIL** stops at once when the 7th wrong label arrives, since no outcome at 100 can pass
  after that. Reaching 100 decided without a pass is also FAIL.
- **Expected effort.** If the rate matches the precision test (1 wrong in 97), the likely stop
  is at 40 decided, about 13-15 min at the gallery's median 20 s per label. One wrong in the
  first 40 moves it to 60.
- **How the lead rates.**
  1. After the send, run the section 4 SELECT and save it as CSV.
  2. `corner_labels.py pilot-links --pilot $P/pilot.csv --proposals <build>/proposals.jsonl
     --labels labels.csv --server https://sidewalk-vancouver-test.cs.washington.edu --out
     $P/pilot_label_ids.csv` prints one Gallery review-list URL, `/gallery?labelIds=<id,…>`.
     It holds the first 120 labels in draw order, under the server's cap of 500 (SW#5444).
  3. The lead validates them top to bottom in the Gallery.
  4. The read is a read-only SELECT of the lead's first validation per label (below), then
     `corner_labels.py pilot-score --pilot $P/pilot.csv --proposals <build>/proposals.jsonl
     --validations validations.csv --out $P/pilot_score.json`.
  5. If the Gallery review list is not on the re-enabled stage, use `/label/<id>` pages in
     `pilot_label_ids.csv` order instead.

```sql
-- the lead's first validation of each pilot label (read-only; save with psql --csv)
SET search_path = sidewalk_vancouver, sidewalk_login, public;
SELECT DISTINCT ON (l.label_id) l.pano_id, lp.pano_x, lp.pano_y,
       v.validation_result::text AS validation
FROM label l JOIN label_point lp USING (label_id) JOIN label_ai_info a USING (label_id)
JOIN label_validation v USING (label_id) JOIN sidewalk_user u ON u.user_id = v.user_id
WHERE l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4' AND l.label_type = 'NoCurbRamp'
  AND a.model_id = 'corner-rule@30168fddeb6e' AND NOT l.deleted
  AND u.username = '<lead username>'
ORDER BY l.label_id, v.start_timestamp;
```

- **Watch item, not scored.** 11 residential units in the precision sample carry Jon's note
  that the far side has nothing to cross to. Those corners stay in scope. A lead's tag or note
  on them is worth reading afterwards.

## 8. Decisions for Jon (each with a recommendation)

1. **Re-enable `sidewalk-vancouver-test` (blocker), and refresh its schema from prod?**
   Jon can start the stage himself from the dashboard. Whether the dashboard can also refresh
   the schema is not known; if it cannot, the refresh is the only part that needs Mikey.
   Recommend both. The test schema is a 2025-09 copy with only the AI's
   CurbRamp labels and no crowd labels. A refresh lets a lead see the crowd's 34 NoCurbRamp
   labels and the city as it is now, and lets the 200 be compared with them. Without a refresh
   the pilot still works, because store pano imagery is shared by test and prod.
2. **200 vs 1,984.** Recommend **200**. The rating needs at most 120 and likely about 40. PS
   is insert-only. And the 200 file cannot be followed by the 1,984 file (section 3).
3. **The rating amendment** (section 7: sequential, Bonferroni, reserve 20). Recommend
   approve as written, before any send.
4. **The city sidewalk layer's terms.** The layer's terms grant personal use and forbid
   redistribution. The labels are derived from it (it decides which corners are kept); the
   layer itself is not sent. This is Jon's call. Recommend settling it before prod. For a test
   server it is still his call; nothing here decides it.
5. **Who the lead is.** Recommend Jon if no one else is available, since the likely stop at 40
   is about a quarter-hour. A second person would be an independent rater, because Jon also
   rated the precision sample. Their username goes into the section 7 SELECT.
6. **The 2007 GSV label at position 200** (no store image, no crop). Recommend keep. It is
   outside the rated 120, and dropping it would change the file and its sha256.
7. **Merge #179 first?** Recommend merging #179, then #177, then sending from `main`. Sending
   from `corner-pilot-169` is equivalent: the labels are byte-identical, and only the send
   tooling differs.
8. **AI validator.** No decision needed: it never votes on AI-created labels (section 6).

## 9. Rebuild

```bash
D=runs/vancouver/corner_labels169_split/na_discount_sw; P=$D/pilot_residential
# proposals.jsonl must be the archived one (sha256 1562e076..., makelab2, see corner-labels-169.md)
$PY scripts/corner_labels.py sample --proposals $D/proposals.jsonl --stratum residential \
    --n 100 --reserve 20 --send 200 --out $P/pilot.csv
$PY scripts/corner_labels.py export --dir $D --stratum residential --only-corners $P/pilot.csv \
    --only-first 200 --out $P/results.pilot200.jsonl
$PY scripts/corner_labels.py export --dir $D --stratum residential --only-corners $P/pilot.csv \
    --out $P/results.residential.jsonl
python scripts/corner_pilot_stats.py
```

`pilot.csv` is reproducible byte for byte from the proposals (seeded). The exports differ only
in their records' `exported_at`; the results files are byte-identical.

## 10. Draft comment for #169 (not posted)

> **Test-server pilot: prepared, not sent.**
>
> - **Scope.** Residential corners only. Jon noted "not safe to cross" on 22 of the 34 arterial
>   units he rated. The pilot draws from the 1,984 residential proposals of the split
>   `na_discount_sw` build.
> - **Files.** A seed-169 order over those 1,984 (`pilot.csv`). The recommended send is its
>   first 200: 197 panos, sha256 `d38c9f3b…`. All pre-send guards pass against the test
>   endpoint with the label-type-aware store guard (#179).
> - **Rating.** A declared amendment to "a seeded 100-label sample": the lead rates in the
>   seeded order with a 20-label reserve for can't-tell, looking at 40 / 60 / 100 decided
>   labels. The Wilson bound uses a Bonferroni z of 2.394. PASS at ≤ 0 / 2 / 6 wrong; FAIL once
>   7 are wrong.
> - **Blocker.** `sidewalk-vancouver-test` is disabled. Its test schema is a 2025-09 copy that
>   holds only the AI's CurbRamp labels. It needs re-enabling, and ideally a refresh from prod,
>   before anything is sent.
>
> Clearance sheet: `docs/corner-labels-169-pilot.md` on `corner-pilot-169`.
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5
