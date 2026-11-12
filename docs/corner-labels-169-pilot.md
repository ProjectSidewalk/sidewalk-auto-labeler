# #169 test-server pilot: clearance sheet (residential corners only)

## What Jon has to decide / do

1. **Approve the new file.** `runs/vancouver/corner_labels169_split/na_discount_sw_sitemed/pilot_residential/results.pilot200.jsonl`:
   sha256 `567b8863…d0b446`, 200 labels on 197 panos (pilot positions 4-203), to `sidewalk-vancouver-test`.
2. **First, a 3-POST smoke send of it** (positions 4-6; section 3). Check that the markers now sit at the curb.
3. **Approve the placement rule** (section 1: site-median with a 0.8 m inset), or name another inset.
4. **Approve the rated-order deviation.** Smoke labels 1-3 leave the rated sample, so rate = positions 4-103 and reserve = 104-123 (section 5).
5. **Decide the 3 smoke labels** (`vancouver-test:89735`, `:89736`, `:89737`, old placement): soft-delete them (draft SQL in section 4) or leave them.
6. **Never resume the old file** (`na_discount_sw/pilot_residential/results.pilot200.jsonl`). Its sidecar would continue at line 4 with the old, inland placement.
7. **Still open from before:** who the lead is; the city sidewalk layer's terms; refreshing the test schema from prod (not done); merging #179 before #177.

Nothing in this sheet was sent by Claude. One approval covers one operation, test servers
included.

## 0. Where things stand (2026-11-11)

- **Test stage running.** Jon started `sidewalk-vancouver-test` from the makelab1 sidewalk
  dashboard. It runs v11.20.0 (release of 2026-11-04), whose `conf/routes` has
  `/ai/submitLabelsOnPano` and the Gallery `labelIds` review list. The test schema was **not**
  refreshed from prod: it is a 2025-09 copy holding only the AI's 64,814 CurbRamp labels, and
  `sidewalk_login` user ids for people differ from prod (`jonfroehlich` is `2d8335ef…` on test,
  `549187e0…` on prod; `SidewalkAI` is `51b0b927…` on both).
- **Panos.**
  - Of the new file's 197 panos, 196 have a `pano_data` row on test. The parent session
    measured 196 of the first file's 197. I checked the 3 panos new to this file
    (`A1CWMRcfc8BFFtqizgcf5w`, `txLb-6-vO2E8hjiJHitGQQ`, `YMJViGgslnA7SYLAkHlCkg`): each
    answers 200 on test `/backupImage/<id>/metadata`.
  - The missing pano is `JBovpHBE3lAv7EW_7oU4Lw` (position 168, a 2026-04 capture). Prod has
    it and its image; test answers 404. The submit inserts its row from prod's own metadata,
    which should then serve the store image (inferred, not tested).
  - `7s0_v8TVpDa7WsPBDCPpmw` (position 200, a 2007 GSV pano, 3328 px) has a test row but no
    store image on either stage, so that label gets no crop.
- **Smoke send** (Jon, 2026-11-12T01:24:14Z, i.e. 11-11 17:24 PST):
  - Command: `send_to_ps.py results.pilot200.jsonl --limit 3` from
    `na_discount_sw/pilot_residential/`, 0 errors. `.submission.json` is committed;
    `.submitted` holds lines 1-3.
  - Labels: `vancouver-test:89735` (pano `R89SNAFa2yc770d_Z6PD1w`), `:89736`
    (`ohEOR_VckTCGrJ6HzdzU7w`) and `:89737` (`GnIlezUX8SPE_OlRLyK4-g`), pilot positions 1-3,
    `model_id corner-rule@30168fddeb6e`.
- **Jon's finding: every marker sits a couple of metres inland** (yard or driveway-apron side,
  not the curb). The site-median rebuild would move these three by 3.60, 2.48 and 2.00 m
  toward the centre. The remaining 197 of that file are on hold for good. It is replaced by
  the file below.

## 1. The placement fix (`build --label-point site-median`, the default since 2026-11-11)

**Cause.** The corner point is `CORNER_OFFSET_M` = 12 m from the unit centre along the sector
bisector, for every street type. The fused ramp sites at Vancouver residential corners sit at a
median of 10.3 m.

**Rule.** It moves **only the label point**, along the same bisector. The corner decision (absent
state, inventory rule, sidewalk within 10 m of the corner point) and the pano choice stay keyed
on the corner point. Distance from the centre:

| case | when | distance |
|---|---|---|
| (a) `unit` | the unit has fused sites at its corners | the median centre-to-site distance of those sites |
| (b) `stratum` | it has none | the stratum's median, from the same corner table at build time |
| `corner` | the stratum has no site at all | the corner point (does not occur in Vancouver) |
| `corner_range_fallback` | the moved point is past the chosen pano's 25 m projection range | the corner point (2 corners in Vancouver, neither in the pilot) |

- **Adjustments.** The distance is reduced by `--label-inset-m` (default 0.8 m) toward the road,
  then clamped to [the stratum's p10, 12 m]. Nothing is per city: the medians and p10 come from
  the corner table given.
- **Sites.** The corner table's `corners[].site_ids` refer to the merged operational sites of
  the base run plus both `--extra-run`s (ids prefixed `<run>:`). The rule resolves them in that
  same merged set. That is why its residential n is 5,563, not the 5,368 the parent session
  counted on the base run alone.
- **Old behaviour.** `--label-point corner` keeps it (no new proposal key). `build.json` records
  `label_point` (rule, flags, stratum stats, case counts) and `label_point_default`.

**Builds.**
- `na_discount_sw_sitemed` is the only site-median build. It was built at `d24c4d6`, so the
  labels' `model_id` is `corner-rule@d24c4d65e6a4`.
- Every earlier build used the corner point. That includes `na_discount_sw`, whose sample Jon
  rated (the precision test is unaffected).

**Stratum stats from the corner table** (m from the unit centre to a site):

| stratum | sites | p10 | median | p90 |
|---|---:|---:|---:|---:|
| residential | 5,563 | 7.88 | 10.31 | 15.77 |
| arterial | 5,119 | 8.63 | 12.36 | 20.14 |
| signalised | 2,257 | 11.23 | 16.85 | 24.64 |

**Checks** (`scripts/corner_label_shift.py`, `label_shift.json`):
- The new build has **the same 3,070 corner keys and the same chosen panos** as
  `na_discount_sw`. Its `counts.csv` and `unlabeled.csv` are byte-identical.

**How far labels moved:**

| | labels | shift p10 / median / p90 / max (m) | case (a) share | clamped max / min |
|---|---:|---|---:|---|
| residential | 1,984 | 0.47 / 2.50 / 4.09 / 4.18 | 0.71 | 157 / 240 |
| arterial | 1,052 | 0.03 / 0.44 / 2.66 / 3.43 | 0.63 | 300 / 56 |
| signalised | 34 | ≈0 (31 clamped at 12 m) | 0.91 | 31 / 1 |
| **new pilot file** | 200 | 0.90 / 2.50 / 4.05 / 4.18 | 0.695 (139 unit, 61 stratum) | 16 / 20 |

Pano-to-label range in the new file: median 8.1 m (was 9.9).

**Short-range labels.** The min-view rule (3 m) was applied to the old point and was not
re-applied. **40 proposals now have a range under 3 m** (27 residential, 13 arterial; minimum
0.19 m). All are listed in `label_shift.json`. Four are in the new file:

| position | corner | range |
|---:|---|---:|
| 91 | `vancouver:res:n47225557#1` | 2.74 m |
| 144 | `vancouver:res:n47307495#2` | 2.18 m |
| 152 | `vancouver:res:n47266095#3` | 2.90 m |
| 157 | `vancouver:res:n47318190#3` | 2.87 m |

Their panos were kept, as asked. They look steeply down; the NoCurbRamp spec does not mask.

## 2. The file and its guards

| | value |
|---|---|
| path | `runs/vancouver/corner_labels169_split/na_discount_sw_sitemed/pilot_residential/results.pilot200.jsonl` (committed) |
| sha256 | `567b88632fa83b21516e37ca8f8a06edf9252fe6d6f555ef99f75cdebbd0b446` |
| lines (= panos = POSTs) / labels / bytes | 197 / 200 / 505,595 |
| pilot positions | 4-203 of the unchanged `na_discount_sw/pilot_residential/pilot.csv` (seed 169) |
| smoke panos excluded | `smoke_panos.txt` (the 3 above). No other pilot corner sits on them, so none was dropped and nothing extends past 203 |
| `model_id` | `corner-rule@d24c4d65e6a4` (the 3 smoke labels carry `corner-rule@30168fddeb6e`) |
| endpoint | `https://sidewalk-vancouver-test.cs.washington.edu/ai/submitLabelsOnPano` |
| `--min-confidence` | omit (NoCurbRamp default 0.5; every label is 1.0) |
| branch | `corner-pilot-169` (needs #179's type-aware store guard) |
| first 3 records | position 4 `vancouver:res:n47220756#0` on `4KpliJAiXgqswes8G0g2XQ`; 5 `vancouver:res:n47279117#3` on `b08SQFYKM92x6TcvKEGVWw`; 6 `vancouver:res:n47271259#0` on `Rv-KWv2G7ecm0-4gw1WV4Q` |

**Guards** (`scripts/pilot_guards.py`, verbatim in `dry_run_summary.txt`):

| guard | verdict |
|---|---|
| endpoint security, model provenance | PASS |
| `check_store_built` | PASS ("196 record(s) … carries only NoCurbRamp labels … (type-aware store guard, #169)") |
| decode / border | PASS: argmax / exclude |
| `check_resume_state` | PASS: a fresh sidecar, since the new file is in a new directory |
| `check_position_state` | PASS (GSV) |
| `check_live_positions` | PASS |

**`check_live_positions` cannot see the smoke campaign from here.** It reads submission records
in the file's own directory only, and the smoke campaign is recorded next door. Read directly,
that record names 3 live panos, the new file holds none of them, and none would move. The
smoke panos are excluded, so the server's per-pano duplicate check (409) is not hit.

**Dry run.** `send_to_ps.py --dry-run` gives 197 records, 0 errors, and 200 NoCurbRamp labels.

## 3. Commands (Jon runs them; Claude does not)

```bash
cd D:/Git/labeler-wt/pilot169                        # branch corner-pilot-169
PY=C:/Users/jonf/anaconda3/envs/sidewalk-auto-labeler/python.exe
P=runs/vancouver/corner_labels169_split/na_discount_sw_sitemed/pilot_residential
EP=https://sidewalk-vancouver-test.cs.washington.edu/ai/submitLabelsOnPano
sha256sum $P/results.pilot200.jsonl                  # 567b8863...d0b446
$PY send_to_ps.py $P/results.pilot200.jsonl --endpoint $EP --dry-run > /dev/null
$PY scripts/pilot_guards.py $P/results.pilot200.jsonl --endpoint $EP
$PY send_to_ps.py $P/results.pilot200.jsonl --endpoint $EP --limit 3   # smoke: positions 4-6
#    -> SELECT below (3 rows with the new model_id); open /label/<id>: marker at the curb?
$PY send_to_ps.py $P/results.pilot200.jsonl --endpoint $EP             # the other 194 POSTs
```

The sidecar `$P/results.pilot200.jsonl.submitted` and the record `….submission.json` appear
beside the file. Commit the record.

## 4. Expected rows, the check, and the rollback draft

After the full send, the test `sidewalk_vancouver` schema should hold **203** live AI NoCurbRamp
labels:
- **200** with `label_ai_info.model_id = 'corner-rule@d24c4d65e6a4'`, on 197 panos;
- the 3 smoke labels (`corner-rule@30168fddeb6e`, 89735-89737), unless they are soft-deleted
  first.

```sql
-- read-only: psql -X -p 6432 -d sidewalk_test, PGOPTIONS='-c default_transaction_read_only=on'
SET search_path = sidewalk_vancouver, sidewalk_login, public;
SELECT a.model_id, count(*), count(DISTINCT l.pano_id), min(l.time_created), max(l.time_created)
FROM label l JOIN label_ai_info a USING (label_id)
WHERE l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4' AND l.label_type = 'NoCurbRamp'
  AND NOT l.deleted GROUP BY 1;
-- for pilot-links: label_id, pano_id, pano_x, pano_y of model_id corner-rule@d24c4d65e6a4
SELECT l.label_id, l.pano_id, lp.pano_x, lp.pano_y
FROM label l JOIN label_point lp USING (label_id) JOIN label_ai_info a USING (label_id)
WHERE l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4' AND l.label_type = 'NoCurbRamp'
  AND a.model_id = 'corner-rule@d24c4d65e6a4' AND NOT l.deleted ORDER BY l.label_id;
```

**Rollback, DRAFT, not run.** AI labels are insert-only through the API, so a rollback is a
soft delete in the test DB, guarded by an exact row count:

```sql
-- DRAFT. psql -X -p 6432 -d sidewalk_test from a makelab1 shell (peer auth).
BEGIN;
SET search_path = sidewalk_vancouver, sidewalk_login, public;
DO $$ DECLARE n int; BEGIN
  UPDATE label l SET deleted = TRUE FROM label_ai_info a
  WHERE a.label_id = l.label_id AND l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4'
    AND l.label_type = 'NoCurbRamp' AND NOT l.deleted
    AND a.model_id = 'corner-rule@30168fddeb6e' AND l.label_id IN (89735, 89737, 89736);
  GET DIAGNOSTICS n = ROW_COUNT;
  IF n <> 3 THEN RAISE EXCEPTION 'expected the 3 smoke labels, touched %', n; END IF;
END $$;
COMMIT;   -- reversible with deleted = FALSE
```

- **The smoke labels.** Run as written, the draft retires the 3 smoke labels.
- **The new file's 200.** Use the same block with `model_id = 'corner-rule@d24c4d65e6a4'`, the
  send's time window from its `.submission.json`, and `n <> 200`.
- **TODO before either:** `label` also has `deleted_by`, `deleted_at` and `deleted_source`, whose
  types and conventions were not read. The server's overwrite path sets `deleted_source`
  `SidewalkAI`.
- **Not rolled back:** the `pano_data` rows the submits upserted. For store panos those are
  prod's own values.

## 5. Rating protocol and its deviations

The issue said: "a lead rates a seeded 100-label sample on the live site". The protocol has two
declared changes.

**Amendment 1 (sequential stop rule).** Declared before any send.
- **Order.** The lead rates in the seeded order. Only the lead's first validation of a label
  counts. A can't-tell is skipped and absorbed by a 20-label reserve. A rating after an unrated
  earlier label is refused.
- **What is measured.** Agree / Disagree / Unsure on the live label, which is **label
  precision**. The bar is the rule's: precision ≥ 0.90 and Wilson lower bound ≥ 0.85.
- **Looks** at 40 / 60 / 100 decided labels, with the Wilson bound at a Bonferroni z of 2.394.
  PASS if wrong ≤ 0 / 2 / 6. FAIL as soon as 7 are wrong. Reaching 100 without a pass is FAIL.
- **Recorded** in `pilot.json` `stop_rule`. `pilot-score` applies it.

**Deviation 2 (declared 2026-11-11, after the smoke send and before any rating).**
- **Why.** The three smoke labels (positions 1-3) carry the old inland placement. Jon has seen
  them, so they are not blind.
- **The rated order now starts at position 4.** Rate = the first 100 labels of the new file
  (positions 4-103), reserve = the next 20 (104-123). Corners on the smoke panos would be
  dropped from the order; there are none.
- **Tools.** `pilot-links` and `pilot-score` take `--skip-first 3 --exclude-panos
  $P/smoke_panos.txt`. `pilot-score` also needs `--relabeled-proposals --proposals
  <na_discount_sw_sitemed>/proposals.jsonl`, because the order was drawn on the corner-point
  build and the pixel keys sent come from the site-median build.
- **What it measures.** The rated labels carry the new placement, so the pilot measures the
  site-median labels. The first file's three are reported apart, as the reason for the fix.

**The lead's steps after the send:**
1. Run the second SELECT in section 4 and save it as `labels.csv`.
2. `corner_labels.py pilot-links --pilot <na_discount_sw>/pilot_residential/pilot.csv
   --proposals <na_discount_sw_sitemed>/proposals.jsonl --labels labels.csv --skip-first 3
   --exclude-panos $P/smoke_panos.txt --server https://sidewalk-vancouver-test.cs.washington.edu`
   prints one `/gallery?labelIds=…` URL of the first 120 in order.
3. Validate them top to bottom.
4. The read is the lead's first validation per label, from the SELECT kept in section 7 (with
   `a.model_id = 'corner-rule@d24c4d65e6a4'`). Then run `pilot-score` with the flags above.

## 6. What the server does with a submission (develop 428341d; the stage runs v11.20.0)

- **NoCurbRamp is accepted.** No code on the submit path is keyed to CurbRamp. The duplicate
  check (409) is per pano, per AI user and per type. No `overwrite` is sent.
- **`pano_data` is upserted.** Position, heading, pitch and roll take the submitted value;
  width and height keep the stored value. For store panos these are prod's own values.
- **Label lat/lng** is computed once, at insert, from the payload.
- **AI validation.** The nightly AI validator never votes on AI-created labels of any type.
- **Crops.** The nightly job cuts one for every live label whose pano has a store image.

These are reads of `develop`. Whether v11.20.0 differs on any of them was not checked; the
smoke send behaved as expected.

## 7. Read-only SELECT for the lead's validations

```sql
SET search_path = sidewalk_vancouver, sidewalk_login, public;
SELECT DISTINCT ON (l.label_id) l.pano_id, lp.pano_x, lp.pano_y,
       v.validation_result::text AS validation
FROM label l JOIN label_point lp USING (label_id) JOIN label_ai_info a USING (label_id)
JOIN label_validation v USING (label_id) JOIN sidewalk_user u ON u.user_id = v.user_id
WHERE l.user_id = '51b0b927-3c8a-45b2-93de-bd878d1e5cf4' AND l.label_type = 'NoCurbRamp'
  AND a.model_id = 'corner-rule@d24c4d65e6a4' AND NOT l.deleted
  AND u.username = '<lead username>'          -- test-DB login; ids differ from prod
ORDER BY l.label_id, v.start_timestamp;
```

## 8. Decisions in detail (recommendations)

1. **Placement rule and inset.** Recommend approving as is. The median shift on residential
   corners is 2.5 m, which matches Jon's "a couple of metres". The inset is a flag if the next
   smoke send still looks inland or now looks in the road.
2. **Rated-order deviation.** Recommend approving (section 5).
3. **Smoke labels 89735-89737.** Recommend soft-deleting them before the full send, so the
   test stage holds one placement only. If they stay, they stay out of the rating.
4. **The 4 short-range labels** (positions 91, 144, 152, 157). Recommend keeping them; position 91
   falls in the rated 100. Changing a pano now would change the corner-to-pano choice the
   checks pin.
5. **Lead.** Jon, unless an independent second rater is available. The likely stop at 40 decided
   labels is about 15 min.
6. **Sidewalk layer terms.** Jon's call, as before.
7. **Schema refresh from prod.** Not needed for the pilot; it would let the 200 be compared with
   the crowd's labels.
8. **#179, then #177, before sending from `main`.** Sending from `corner-pilot-169` is
   equivalent.

## 9. History: the first file (superseded, do not resume)

`na_discount_sw/pilot_residential/results.pilot200.jsonl` (sha256 `d38c9f3b…`, 197 panos, 200
labels, corner-point placement, positions 1-200). Its lines 1-3 went to test in the smoke send.
Its `.submitted` sidecar would resume at line 4. **Do not run it again.**
`results.residential.jsonl` (1,984 labels) in the same directory is superseded as well.

## 10. Rebuild

```bash
D=runs/vancouver/corner_labels169_split; S=$D/na_discount_sw_sitemed; P=$S/pilot_residential
$PY scripts/corner_labels.py build --na-read discount --na-read-settled \
    --corners runs/vancouver/corner_inventory169_split/corners_full.jsonl \
    --run-dir ../../sal-vancouver/runs/vancouver --extra-run ../posdetect241/runs/vancouver_posdetect241 \
    --extra-run ../posdetect241/runs/vancouver_posdetect241_gsv --out $S \
    --sidewalk-presence runs/vancouver/corner_labels169/sidewalks/cov_sidewalks.geojson \
    --sidewalk-keep STATUS=Available,Pathway,Private          # --label-point site-median (default)
$PY scripts/corner_label_shift.py --old $D/na_discount_sw --new $S \
    --pilot $D/na_discount_sw/pilot_residential/pilot.csv --results $P/results.pilot200.jsonl
$PY scripts/corner_labels.py export --dir $S --stratum residential \
    --only-corners $D/na_discount_sw/pilot_residential/pilot.csv --skip-first 3 \
    --exclude-panos $P/smoke_panos.txt --only-first 200 --out $P/results.pilot200.jsonl
```

`proposals.jsonl` (sha256 `9925a736…`, 5.5 MB) is not committed. A rebuild from the pinned
inputs reproduces it; only `build.json`'s time and commit fields differ.

## 11. Draft comment for #169 (not posted)

> **Test-server pilot: smoke send, a placement fix, and a new file.**
>
> - **Smoke send.** Jon sent 3 labels to `sidewalk-vancouver-test` (positions 1-3 of the
>   residential pilot order). All three sat a couple of metres inland of the curb.
> - **Cause.** The label was placed at the corner point, 12 m from the intersection centre for
>   every street type. Fused ramp sites at residential corners sit at a median of 10.3 m.
> - **Fix.** The label point now moves along the corner bisector to the unit's own median
>   centre-to-site distance, else its stratum's, less a 0.8 m inset, clamped to [stratum p10,
>   12 m]. The corner set and the chosen panos are unchanged. Residential labels move a median
>   of 2.5 m.
> - **New file.** Positions 4-203: 200 labels on 197 panos, sha256 `567b8863…`. Every guard
>   passes. The rated order now starts at position 4, a declared deviation.
>
> Clearance sheet: `docs/corner-labels-169-pilot.md` on `corner-pilot-169`.
>
> 🤖 Generated with [Claude Code](https://claude.com/claude-code) — Opus 5.5, claude-opus-5-5
