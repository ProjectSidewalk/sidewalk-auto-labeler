# Handoff: land the AI-label-submission city flag (SidewalkWebpage)

**For:** the SidewalkWebpage Claude Code session (repo `/home/jonf/git/SidewalkWebpage`).
**Written:** 2026-08-04, from the sidewalk-auto-labeler session, after a full open-issue review.
**Goal:** unblock AI label submission for cities other than Vancouver, WA.

This note is self-contained — you don't need anything from the labeler session.

---

## Why this matters

The auto-labeler has finished **nine cities** (detected, natively archived, ground-truthed):
bend, paterson, gainesville, sao_paulo, richmond, clovis, morgantown, annapolis,
budapest_district5. **Zero have ever been submitted to a Project Sidewalk server** — there is
not a single `.submitted` sidecar in the labeler repo.

The labeler side has been finished for weeks. The only thing in the way is one conditional in
this repo.

## The blocker

`app/controllers/AiController.scala` (~line 51 on `develop`):

```scala
if (configService.getCityId == "vancouver-wa") { ... }
else BadRequest("AI label submission beta is only supported in Vancouver, WA at the moment.")
```

Added Sept 2025 for the Vancouver pilot. Documented only as a "⚠ City gate" note in
`docs/ai-subsystems.md` — **there is no GitHub issue tracking its removal.**

## The fix already exists — it just never got a PR

Branch **`ai-label-submission-city-flag`**, commit **`6aa4dc381`**, authored by Jon
**2026-07-02**. One commit, **+17 / −6 across 4 files**:

| file | change |
|---|---|
| `app/controllers/AiController.scala` | `getCityId == "vancouver-wa"` → `configService.getAiLabelSubmissionEnabled`; rejection message becomes "AI label submission is not enabled for this city." |
| `app/service/ConfigService.scala` | adds `getAiLabelSubmissionEnabled: Boolean` to the trait + impl |
| `conf/cityparams.conf` | new `ai-label-submission-enabled { vancouver-wa = true }` block, with an explanatory comment |
| `docs/ai-subsystems.md` | rewrites the "⚠ City gate" note to describe the flag |

The impl:

```scala
def getAiLabelSubmissionEnabled: Boolean =
  config.getOptional[Boolean](s"city-params.ai-label-submission-enabled.$getCityId").getOrElse(false)
```

**The design decision that makes this low-risk:** every other `city-params` block uses
`config.get`, which throws unless all 57 cities have an entry. This one uses
`getOptional(...).getOrElse(false)`, so unlisted cities reject submissions. Only
`vancouver-wa` is listed — meaning **runtime behavior is identical to today for all 57
cities**, and the only user-visible difference anywhere is the rejection message string.

No DB migration. No evolution. No frontend change.

## Rebase state — verified 2026-08-04, it is clean

The branch is **1 ahead / 716 behind `develop`**. The 716 is misleading; I checked every hunk
against current `develop` and **all four anchors survive verbatim — only line numbers moved**:

| file | commits on `develop` since merge base (`b1fb53c14`) | anchor status |
|---|---|---|
| `AiController.scala` | **0 — untouched** | — |
| `ConfigService.scala` | 11 | intact; trait decl 526 → 620, impl 1305 → 1663 |
| `conf/cityparams.conf` | 4 | intact; insertion point 864 → 895 (between `ai-tag-suggestions-enabled`'s close and `ai-validation-min-accuracy {`) |
| `docs/ai-subsystems.md` | 1 | intact; "⚠ City gate" line 84 → 87 |

Two more checks, both clean:

- **`ConfigServiceImpl` is the only implementer** of the `ConfigService` trait
  (`ConfigService.scala:640`), and there are **no test stubs or mocks** of it. Adding a trait
  method therefore cannot break compilation anywhere else — the ~10 files referencing
  `ConfigService` all merely inject it as consumers.
- **Only one open PR touches any of these files** — #4759 ("Today & this week" on Across
  Cities), in a different region of `ConfigService`. Land this before that grows.

Expect a rebase with offsets only. If you hit a real conflict, something changed after
2026-08-04 — re-check rather than force through.

## Steps

1. **File the issue first.** There is no issue for this, and repo convention is to prefix the
   branch with the issue number (`CLAUDE.md`: *"If there is an associated Github issue, begin
   the branch name with the issue number"*). The existing branch name predates that. Suggested
   title: *"Replace the vancouver-wa hard gate on AI label submission with a per-city flag"*.
2. **Work in a worktree**, per the `.claude/worktrees/` convention — the main tree is on
   `qa-develop` with an untracked `docs/planning/`. Name it `<issue#>-ai-label-submission-flag`.
3. `git rebase origin/develop` onto the one commit. Consider renaming the branch to match the
   issue number.
4. **`make scalafmt-fix`** — mandatory after any Scala edit; `scalafmtCheckAll` is a blocking
   CI gate.
5. **Compile via the thin client** (do NOT run a plain second `sbt compile` — it deadlocks
   against a running `sbt ~ run` over target locks):
   ```bash
   docker exec projectsidewalk-web bash -lc "cd /home && sbt --client compile"
   ```
6. **PR → `develop`** (not `master`, which is the release branch).
7. Branch protection requires `Backend (compile + scalafmt)` and `Frontend (build)` green.
   **There are no required reviews — self-merge is preserved**, so this does not have to wait
   on Mikey (see below).

There is nothing meaningful to functionally test until a city is flagged on — the change is a
config read. Compile + scalafmt is the real verification.

## What comes after the merge (the actual goal)

Set the flag for the first city and submit. **The decision (Jon, 2026-08-04) is Richmond, VA
goes first — not Clovis**, which has no PS instance at all. Richmond, per `cityparams.conf`:

- `status = "private"` (not publicly listed), launched 2026-03-09, schema `sidewalk_richmond`
- `pano-viewer-type = "mapillary"` — our Mapillary pano ids will render
- effectively empty: 2 researcher users, 93 labels, 0.3 of 45.7 km explored — nothing to contaminate
- best-validated Mapillary run we have: 9,091 panos / 9,526 operational detections, GT P 0.965 / R 0.895

**Use the test instance first** — and note it comes free: merging to `develop` auto-deploys the
test stage, so `richmond-va = "https://sidewalk-richmond-test.cs.washington.edu"` gets the flag
with no extra step. Reaching **prod** is heavier: it needs a full release cut (version bump in
`build.sbt`, a version-table evolution, a `vX.Y.Z` tag on `master`), so plan it into a release
rather than treating it as a config toggle.
Nobody has ever run a non-GSV submission end to end, so the test deploy is where to find out
whether `pano_source: "mapillary"` actually works through POV math,
`getStreetEdgeIdClosestToLatLng`, and the `pano_data` upsert. Then 2–3 records on prod, eyeball
them in Validate, then the full ~9.5k.

Tracked on the labeler side as **sidewalk-auto-labeler#21**.

## Things worth knowing / raising

- **AI validation will not cover these labels.** `LabelTable.getLabelsToValidateWithAi` filters
  `pd.source === PanoSource.Gsv` (`LabelTable.scala:2464`) **and** `r.role =!= "AI"`
  (`:2449`). So the daily validator never runs on Mapillary panos and never validates
  AI-authored labels. Richmond has `ai-validation-enabled = true`, but for our labels that flag
  is **inert** — QA is human-only by construction. Was that deliberate? Worth its own issue.
- **Nobody's review gates this.** `develop` has no required reviews, and deployment is
  git-driven, not person-driven (`docs/deployment-and-stages.md`): pushing `develop`
  auto-deploys **test**, `staging` deploys staging, and a **semver tag `vX.Y.Z` on `master`**
  deploys prod. So merging this PR *is* the test deploy. Mikey (`misaugstad`) added the original
  gate for the Sept 2025 pilot and owns the AI-subsystems area, so it's worth flagging to him
  for judgment — but not as a permission step.
- **The one genuinely external dependency is `INTERNAL_API_KEY`.** It is a runtime env var that
  lives in **UW CSE IT's separate ops repo**, not in this codebase
  (`docs/deployment-and-stages.md:181` + the doc's standing "ask UW CSE IT" note for anything
  operational). `internalKeyValid` **fails closed** when the key is unset
  (`ControllerUtils.scala:75`), so submission returns 401 even with the flag on until a key
  exists on that instance. Local dev is fine — `docker-compose.yml` sets
  `INTERNAL_API_KEY=DUMMY_INTERNAL_API_KEY`.
- **Settle labeler#22 (pano provenance) before bulk-submitting**, not after — the `pano_data`
  column shape is a server-side call, and anything submitted first needs a backfill or resubmit.
- **Five of the nine finished cities have no PS instance at all** — clovis, bend, annapolis,
  morgantown, budapest are absent from `cityparams.conf` entirely. Bend has an external
  requester in SidewalkWebpage#4444 (open, quiet since 2026-07-03). That's city-onboarding work
  (#4291), not a flag.
- **Do not submit to gainesville early.** It has 15,121 live crowd labels *and* is the baseline
  for the AI-vs-crowd agree-rate study (labeler#31). Paterson (21,325 labels) is likewise a poor
  first target.

---

*Prepared with Claude Code (claude-opus-5[1m]). Every code/config claim above was verified
against `develop` on 2026-08-04 — re-verify if significant time has passed.*
