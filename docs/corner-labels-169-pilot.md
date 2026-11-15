# #169 test-server pilot: clearance sheet (residential corners only)

## What Jon has to decide / do

**2026-11-15 status: ON HOLD.** The second smoke send (positions 4-6) put 2 of 3 markers at the
curb and 1 in the street (section 5, deviation 3). The rated order now starts at position 7.
Nothing more is sent until a local placement check passes. The list below is the 11-11 list,
kept for the record.

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
- **Second smoke send (new file, cleared by Jon 2026-11-15; run by Claude Opus 5.5 at
  2026-11-15T14:20:41Z):** `send_to_ps.py results.pilot200.jsonl --endpoint <test> --limit 3`
  from `na_discount_sw_sitemed/pilot_residential/`. Before it: sha256 `567b8863…d0b446`
  re-checked, dry run 0 errors, every `pilot_guards.py` check PASS, the three panos' test
  `/backupImage/<id>/metadata` answered 200. A first attempt got HTTP 401 on all three (no
  `PS_INTERNAL_API_KEY` in the worktree; nothing landed, no sidecar written); the re-run with the
  key 0 errors. `.submission.json` committed; `.submitted` holds lines 1-3.
  - Labels: `vancouver-test:89738` (pano `4KpliJAiXgqswes8G0g2XQ`, pilot position 4), `:89739`
    (`b08SQFYKM92x6TcvKEGVWw`, 5) and `:89740` (`Rv-KWv2G7ecm0-4gw1WV4Q`, 6).
  - Check (public `/v3/api/rawLabels?labelType=NoCurbRamp`, makelab1 ssh refused): test held 3
    NoCurbRamp labels before (89735-89737) and 6 after; pano_x/pano_y equal the file's. The
    server's lat/lng is 0.92 / 0.40 / 0.50 m from the build's site-median label point and 2.18 /
    1.71 / 2.94 m from the old corner point. Whether the markers sit at the curb is Jon's
    visual check.

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

**Deviation 3 (declared 2026-11-15, after the second smoke send and before any rating).**
- **Why.** Jon checked the placement of the second smoke send's three labels (positions 4-6,
  `vancouver-test:89738`-`89740`) in the live viewer, so they are not blind either.
- **The rated order now starts at position 7.** Rate = positions 7-106, reserve = 107-126. No
  other pilot corner sits on the three smoke panos, so nothing else drops.
- **Tools.** `--skip-first 6`; the three panos were appended to `$P/smoke_panos.txt`, so
  `--exclude-panos $P/smoke_panos.txt` covers both smoke sends.
- **Placement verdict (Jon, 2026-11-15):** 89738 and 89740 at the curb; **89739 in the street**,
  short of a rounded curb. The fixed distance along the bisector (10.41 m there, case `unit`)
  cannot follow each corner's own street widths and curb radius. **The remaining 197 POSTs are on
  hold** until a placement check on residential corners outside the rated range (positions
  127+, so blinding is untouched) passes a bar written before looking. If the rule changes, the
  file is rebuilt and gets a new sha256, guards and a fresh smoke send from outside the rated
  range, each cleared separately.

**The lead's steps after the send:**
1. Run the second SELECT in section 4 and save it as `labels.csv`.
2. `corner_labels.py pilot-links --pilot <na_discount_sw>/pilot_residential/pilot.csv
   --proposals <na_discount_sw_sitemed>/proposals.jsonl --labels labels.csv --skip-first 6
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

## 12. Placement check (2026-11-15)

Why: the second smoke send put 2 of 3 labels at the curb and 1 (`vancouver-test:89739`) in the
street (section 5, deviation 3). This check measures where each candidate rule puts the marker,
locally, before anything else is sent. No server is touched. Tool:
`scripts/corner_placement_check.py` (`sample`, `render`, `score`). Outputs:
`runs/vancouver/corner_placement169/`.

### 12.1 Pre-registration (written 2026-11-15, after `sample` and before any pano was fetched or any card rendered)

**Rules**, all on the corner's bisector and on the same chosen pano (the pano choice is keyed on
the corner point, so it does not depend on the rule):

| | `--label-point` | point |
|---|---|---|
| A | `site-median` (current default) | the build's stored pixel, exactly what is sent |
| B | `sidewalk-crossing` (new, opt-in) | the first crossing, 3-20 m out from the unit centre, of the bisector with ANY line of the city sidewalk layer (`cov_sidewalks.geojson`, every STATUS); site-median where there is none |
| C | `sidewalk-crossing-inset` (new, opt-in) | B moved back toward the centre by the crossed line's `UNITTYPE` inset: ATTACHED 0 m, DETACHED 2.5 m, other or none 1.0 m; never nearer than 3 m |

C's insets were set from the three calibration corners, which are therefore development data
and are scored apart: on the two ATTACHED corners Jon judged at the curb, B is within 0.8 m of
A; at 89739 (DETACHED) the curb lies between A's 10.4 m (in the street) and B's 15.3 m.

**Sample.** `sample` seed 20261115; 60 residential corners drawn from pilot positions >= 204
of `na_discount_sw/pilot_residential/pilot.csv` (sha256 `157ff268…`), keeping the build's
chosen pano. Corners whose chosen pano is also chosen at positions 1-203, or is a smoke pano,
are not eligible (200 panos blocked; 1,730 eligible). Keys sha256
`6de6873d9b30d24833905b1a3a1203376516d8d7c746a807ceee27c78f21eb83` (`sample.json`). Build
`na_discount_sw_sitemed`, proposals sha256 `9925a736…`; sidewalks sha256 `11503ed4…`. Of the
60, B finds a crossing at 59 (28 DETACHED, 23 ATTACHED, 9 no UNITTYPE).
*Deviation from the approved plan:* the plan said positions >= 127. Positions 127-203 are in
the pilot send file, and would be sent, and possibly rated if can't-tells run past the reserve.
So the draw starts at 204, which keeps every card off the send file.

**Cards.** One per (corner, rule), 180 + 9 calibration = 189, shuffled (seed 20261116) and
named `c001`-`c189`. A card shows a 50 x 30 degree window of the store pano centred on the
marker, which sits on PS's own pixel (`detectors.ps_pixel`). The rule of each card is only in
`key.json`.

**Tags** (one per card):
- `at_curb`: on the curb edge, or on the sidewalk corner right behind the curb (where a ramp
  would start).
- `inland`: behind the curb, beyond the sidewalk corner (yard, lawn, planter strip, driveway
  apron, fence).
- `in_street`: on the roadway (gutter included).
- `cant_tell`: occluded, too far or blurred, or no curb in view.

**Rater.** AI pre-pass by Claude Opus 5.5 (claude-opus-5-5), blind to rule: tags are written
for every card from the cards and montages alone, then `key.json` is joined by `score`. Not
independent: the same model designed rules B and C. The tags go to
`tags_ai_claude-opus-5-5.json`, marked as AI pre-rater tags, never as ground truth. Jon then
skims the page's skim section (every `inland`/`in_street` card and every corner where the
rules' tags differ).

**Bar, per rule** (the 60 sample corners; calibration excluded): `at_curb` >= 0.90 of the
decidable cards (`at_curb` + `inland` + `in_street`) AND at most 2 `in_street`. The Wilson 95%
interval is reported but does not gate; at n = 60 a share of 0.90 has a lower bound near 0.80.

**What each outcome means.**
- **Exactly one rule passes:** it is the recommendation.
- **Several pass:** the one with fewer `in_street`, then more `at_curb`, then the simpler one
  (A, then B, then C).
- **None passes:** nothing is rebuilt or sent. Next is a different approach (e.g. per-leg curb
  geometry), with its own check.
- **Calibration check:** if the AI's tag on any of rule A's three calibration cards disagrees
  with Jon's verdict (curb, street, curb), the result is marked AI-unconfirmed. Jon then rates
  the winning rule's 60 cards himself before any rebuild.
- **Whatever the result:** a rebuild under the chosen rule, its guards, and a 3-label smoke
  send from outside the rated range are each cleared separately.

### 12.2 Result (2026-11-15): no rule passes; nothing is rebuilt or sent

**Rater:** AI pre-pass, Claude Opus 5.5 (claude-opus-5-5), blind to rule. The tags were
committed (474d331) before `key.json` was joined. They are not ground truth.

| rule | at_curb / decidable | share | Wilson 95% | in_street | inland | can't tell | bar |
|---|---|---|---|---|---|---|---|
| A site-median (current) | 23 / 47 | 0.489 | 0.353-0.628 | 6 | 18 | 13 | FAIL |
| B sidewalk-crossing | 29 / 47 | 0.617 | 0.474-0.742 | 1 | 17 | 13 | FAIL |
| C sidewalk-crossing-inset | 32 / 47 | 0.681 | 0.538-0.796 | 5 | 10 | 13 | FAIL |

- **Calibration reproduced.** The AI's tags on rule A's three calibration cards match Jon's
  verdicts (89738 at the curb, 89739 in the street, 89740 at the curb), so the result is not
  marked AI-unconfirmed.
  - On 89739, B and C both land inland (on the grass of the wide corner).
  - On 89738 and 89740, B and C land at the curb.
- **The current rule is far from the bar.** About half of rule A's decidable labels sit at the
  curb, with 6 of 47 in the street. The second smoke send's 2 of 3 is consistent with that
  rate; the first send's 3 inland labels used the older corner-point rule.
- **The pre-registered consequence:** none passes, so nothing is rebuilt or sent. The 197
  POSTs stay on hold.

**Exploratory (after unblinding; not a test).** Split by the `UNITTYPE` of the sidewalk line B
crosses:

| rule | ATTACHED (n = 23) | DETACHED (n = 28) | no UNITTYPE (n = 9) |
|---|---|---|---|
| A | 10 / 19 at curb, 3 in street | 11 / 22, 2 in street | 2 / 6, 1 in street |
| B | **17 / 19, 0 in street** | 9 / 22, 0 in street (13 inland) | 3 / 6, 1 in street |
| C | 17 / 19, 0 in street | 13 / 22, 3 in street | 2 / 6, 2 in street |

- **B on attached sidewalks** puts 17 of 19 decidable labels at the curb with none in the
  street. On attached sidewalks the city line sits at, or just behind, the curb.
- **Detached corners are the open problem.** No rule exceeds 13 of 22. B lands on the
  sidewalk behind the planter strip, which this check's tag set counts as inland. C's fixed
  2.5 m inset overshoots into the street on 3.
- **Whether that is wrong is a definition question for Jon.** It turns on where a NoCurbRamp
  label belongs at a detached-sidewalk corner: on the curb, or on the sidewalk corner where
  the walking path ends. If the sidewalk corner is acceptable, B's detached "inland" count
  would shrink (not measured: the tags do not separate "sidewalk behind planter" from
  "lawn").
- **12 of the 60 corners are can't-tell under all three rules.** Most are curbless corners
  (dirt or gravel edges, an alley), some are occluded by vehicles or privacy blur. About 1 in
  5 sampled corners has no curb to put a label at. That bears on whether those corners should
  be labeled at all (guide rule 3 counts sidewalk presence, not curb presence).
- **7 corners miss under all three rules the same way** (6 inland, 1 in the street). They are
  listed in the page's skim section.

**Where each number lives** (all under `runs/vancouver/corner_placement169/`, committed except
the images):

| number | file | made by |
|---|---|---|
| sample, eligible count, keys digest, each rule's point and pixel | `sample.json` | `sample` |
| blind cards / rule per card | `cards.json` / `key.json` | `sample` |
| AI tags | `tags_ai_claude-opus-5-5.json` | by hand (AI pre-pass) |
| per-rule table, calibration, disagreements, all-miss corners | `summary.json`, `summary.md` | `score` |
| crops, montages, page | `crops/`, `review/`, `index.html` (36 MB, not committed) | `render` |
| the ATTACHED/DETACHED split | recomputed from `sample.json` (`rules.B.sidewalk.unittype`) + `key.json` + tags | ad hoc (exploratory) |

**Replication.**

```bash
PY=C:/Users/jonf/anaconda3/envs/sidewalk-auto-labeler/python.exe
O=runs/vancouver/corner_placement169
$PY scripts/corner_placement_check.py sample --out $O          # reproduces sample/cards/key
# pixels: the 63 panos of $O/panos_needed.txt from the makelab2 store
#   (/projects/makeabilitylab/sidewalk_panos/Panoramas/vancouver-wa), read-only, as one tar
$PY scripts/corner_placement_check.py render --out $O --panos <local copy> --summary $O/summary.json
$PY scripts/corner_placement_check.py score --out $O --tags $O/tags_ai_claude-opus-5-5.json
```

**What Jon decides next.**
1. **The definition:** at a detached-sidewalk corner, is the sidewalk corner (behind the
   planter) an acceptable place for a NoCurbRamp label, or must it be the curb?
2. **The route:**
   - *Option 1:* a second pre-registered check of a hybrid. B on ATTACHED. On DETACHED,
     either B, if (1) accepts the sidewalk corner, or a curb estimate from the sidewalk line
     offset by the layer's own planter width (not in the data as checked).
   - *Option 2:* restrict the pilot to ATTACHED corners, where B looks good, after its own
     check on a fresh sample.
   - *Option 3:* rethink placement entirely, e.g. a curb detector on the pano.
3. **Curbless corners:** whether to drop them from labeling.

Each later rebuild, guard run and smoke send is still cleared separately.

### 12.3 Confirmation check: rule B on ATTACHED corners (pre-registration, 2026-11-15)

**Status: approved by Jon 2026-11-15 and committed before the draw**, before any pano is
fetched and before any card is rendered. Only the draw's digest is added after this commit;
nothing else in this section changes.

**Jon's decisions (2026-11-15).**
1. A NoCurbRamp label must be **at the curb**, also at a DETACHED (planter-strip) corner. The
   sidewalk corner behind the strip is not acceptable.
2. Route: pilot on **ATTACHED corners only, with rule B** (`sidewalk-crossing`), after this
   check. DETACHED corners stay out until a curb-finding rule passes a check of its own.
3. Curbless corners: still open. Here they are `cant_tell`, as in 12.1.

**Why a new check.** 12.2's 17 of 19 is a split made after unblinding, on 23 corners. It is a
reason to test, not a result.

**ATTACHED corner** (fixed before the draw; `UNITTYPE` is the city sidewalk layer's own field):
- rule B finds a crossing, and
- the crossed line's `UNITTYPE` is `ATTACHED`, and
- every line of the layer within 6 m of the crossing point is `ATTACHED` (a line with no
  `UNITTYPE` fails this). This covers both legs where they are separate features. 6 m is
  under a residential street's width, so the opposite corners are not counted.

Counts, all 1,984 residential corners of the pilot order: 832 by the crossed line alone, **799**
with the 6 m clause (804 at 4 m, 793 at 8 m, so the radius hardly matters). All 23 ATTACHED
corners of the first check also meet the 6 m clause, so 12.2's 17 of 19 is the same set.

**Pilot block (reserved, never drawn here).** The ATTACHED corners in pilot order from position
7, less the first check's 60 corners and any corner on a smoke pano: 775. The **first 260** of
them (through position 661) are reserved for the rebuilt pilot file (3 smoke + 200 + spare).

**Sample.** Seed 20261117; **80** corners drawn with `random.Random(seed).sample` from the
other 515 ATTACHED corners, less any whose chosen pano is also chosen in the pilot block or at
positions 1-203, or is a smoke pano, or a pano of the first check: **469 eligible**. Rule B
only, one card per corner, same crop and marker as 12.1. Expected about 66 decidable cards
(12.2 had 4 of 23 `cant_tell`).

**Tags.** The four of 12.1, unchanged.

**Raters.**
- AI pre-pass by Claude Fable 5.2 (claude-fable-5-2), from the cards alone, committed before
  Jon sees the page. (Rules B and C were designed by Claude Opus 5.5, not by this model.)
- **Jon's tags are the verdict.** A local, self-contained HTML page shows each card with the
  AI's tag; one key confirms it, another key overrides it. The page exports a JSON of Jon's
  tags. Jon sees the AI's tag first, so his tags are not independent of it; that is the price
  of about 80 key presses in place of 80 fresh judgements.

**Bar** (Jon's tags): `at_curb` >= 0.90 of the decidable cards AND at most 2 `in_street`. The
Wilson 95% interval is reported and does not gate.

**What each outcome means.**
- **PASS:** rebuild the pilot file on the pilot block with rule B. Guards, a 3-label smoke
  send and the send follow, each cleared separately.
- **NEAR MISS** (share 0.80 to under 0.90, at most 2 `in_street`): apply the same bar to the
  cards the **AI** tagged `at_curb`. If that subset passes, the pilot may go ahead **screened**:
  the AI tags every pilot-block corner's rule-B card, and only corners it tags `at_curb` are
  sent. The write-up must then say the placement was screened by an AI tagger, not automatic.
  If the subset fails, treat as FAIL.
- **FAIL** (share under 0.80, or 3 or more `in_street`): rule B is not good enough even on
  ATTACHED corners. No rebuild. Placement goes to the image-based route (find the curb in the
  pano).
- **More than 30% `cant_tell`:** reported as a finding for decision 3 before anything is
  rebuilt, whatever the share.

**What the pilot can then claim.** Its precision speaks for ATTACHED residential corners in
Vancouver (about 40% of the pool), not for all corners. Re-drawing the rated sample from the
pilot block is deviation 4 of section 5, to be declared before the rebuild.

**Draw (added after the pre-registration commit dd759e4).** `sample --attached` -> `runs/vancouver/corner_placement169_attached/`:
80 corners of 469 eligible, keys sha256 `9ea680bb3d87567697e430f9b3d2bc378b4351f5708ecc2e48d7a73d523783f3`;
799 ATTACHED, pool 775, pilot block 260 through position 661, block keys sha256 `4256e288…715c497`
(`sample.json` `attached.pilot_block_keys`). Drawn positions 681-1981. At 2 of the 80 the rule-B point is
past the pano's range, so the card shows the corner point, which is what a build would send.

Nothing here touches a server. Pixels come from the makelab2 pano store, read-only.
