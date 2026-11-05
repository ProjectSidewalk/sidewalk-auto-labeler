# Corner present/absent gallery (RampNet#243)

**Status: partial, 50 of 95 rated (2026-10-07, `verdicts__jonf.json`); the item list is
final.** Built 2026-10-06 and rebuilt after the review
of PR #141 (crop choice now always includes the newest capture, B1): same seed, units and
order, items.jsonl sha256 `bbed3d6ddaf1835a66da3afb274c3f338cc80512d4af7c4567c96c1bf47a1459`
(was `8e5f881b...7bdc1d`). One rater (Jon) will rate it. The page and the
scorer are built so a second rater can repeat the pass later with their own file.
Issue: [RampNet#243](https://github.com/ProjectSidewalk/RampNet/issues/243). It follows
amendment A8 of [`corner-inventory.md`](corner-inventory.md) (RampNet#241).

## Why

After the position-selected run, the corner-inventory decision rule depends on one question:
what do the city's `NA` inventory points with no `RAMPTYPE` mean?

- Counted as "the city says no ramp", absence precision is 0.252 [0.229, 0.276].
- Not counted, it is 0.965 [0.954, 0.974].
- 944 of the 990 absent units that are not clean hold only such points.

Separately, recall at the #241 target units is 31/66. The 35 false absences there have not been
looked at.

## What is rated

Each unit is an intersection unit of the merged #241 build that the fusion arm calls
**absent** under primary observability (the decision rule's read). Every corner of the unit gets
one verdict: **present**, **absent** or **can't tell**. An absent verdict can say why:
`curb_no_ramp` (a sidewalk reaches the corner, the curb has no cut) or `no_sidewalk` (no
sidewalk reaches the corner). The rubric is `RUBRIC` in
`scripts/corner_gallery.py` (version 1). The page writes it into every verdicts file.

| part | population | drawn | question |
|---|---:|---:|---|
| `false_absence`: absent, >= 1 `Available` point, a #241 target unit | 35 | 35 (all) | Is the recall drop real, or an inventory or geometry artifact? |
| `na_noramp`: absent, every inventory point `NA` with no `RAMPTYPE` | 944 | 40 | Does `NA` with no type mean "no ramp"? |
| `clean`: absent, no inventory point | 333 | 20 | Control: calibrates the rater against units the rule already counts as clean |

95 units and 281 corners in all. The population sizes are checked against amendment A8 at
build time. Out of scope: the 412 inventory-gap corners, the 9 false absences outside the #241
targets, and the 2 absent units whose inventory is some other mix.

**Draw.** All 35 false absences. Then one `random.Random(243)`: `rng.sample` of 40 from the
sorted `na_noramp` unit keys, then 20 from the sorted `clean` keys. The lists are in
`snapshot.json` (`draw`). The review order is sha1 of the unit key, so a unit's position in
the gallery says nothing about its part. The page never receives the part.

**What the rater sees.**
- A north-up Esri aerial, about 70 m across, with the 30 m window and the unit's legs.
- Each corner's marker at its corner point (12 m out along the sector bisector), numbered.
- Up to 3 crops per corner. Each crop is a 60 deg square of the equirect pano, centred on the
  corner point with an orange ring. Candidates are the panos within 25 m of the unit centre
  or the corner point that are within 40 m of the corner point. **One slot always goes to the
  newest capture date among the candidates** (the nearest pano of that date). The other slots
  are the nearest panos at least 3 m from the corner point; nearer panos are used only to top
  up. The projection assumes a level camera at 2.5 m, the fixed height of the #241 fusion.

  The first build chose by distance only. The review of PR #141 found that this hid a newer
  pano at 24 of 281 corners (13 false-absence corners, 9 `na_noramp`, 2 clean), for example
  `vancouver:art:n47270030` corner 0: three 2014-08 crops shown, a 2021-12 pano in the pool.
  The rule above was adopted before any rating, and all 24 corners now show their newest
  pano. Each corner records `newest_available` (the newest capture date in its pool) in
  `items.jsonl`, and the build test asserts that the newest crop shown equals it.
- Each crop's capture date and distance, and the camera position on the aerial.

The city inventory points are hidden until the unit is completed. They are not in the page
at all: `render` writes each unit's inventory to `gallery/<rater>/reveal/<unit>.js`, and the
page inserts that script only when the unit is completed (a `<script>` tag, because Chrome
refuses `fetch()` of a `file://` URL). So view-source or devtools on the page before the reveal
shows no status, `RAMPTYPE`, `INSTDATE` or position. **What this does not do:** the reveal
files sit beside the page, and `items.jsonl` holds the same inventory, so a rater who opens
those files can still read it. Blindness rests on the rater not opening the bundle's files,
which is a much smaller ask than never opening devtools; it is not enforced. (The first
version embedded the inventory in the page; the review of PR #141 caught it before any rating.)
After the reveal the points appear as squares
on the aerial and as a list per corner: status, `RAMPTYPE`, install date, and a flag when the
install month is later than the newest crop of that corner. The verdicts given at the first
completion are frozen as `blind`. A change after the reveal is allowed. It is recorded as
`edited_after_inventory` and is never scored as blind.

**Reused from #224.** The page is a sibling of RampNet's cluster-review gallery
(`scripts/cluster_review_gallery.py` on RampNet branch `cluster-review-224`), and the bundle
uses the #224 exporter's aerial and crop helpers (`export_cluster_review.aerial_window`,
`make_aerial`, `site_explorer.fetch_crops_remote`). Shared mechanics:
- one unit per screen;
- browser storage keyed by rater + the sha256 of `items.jsonl`;
- the same prefill rule: local work wins only on units worked in this browser, and
  conflicts are listed;
- the same timing rule: 1 s ticks while visible and active within 60 s;
- review notes, and Export to a per-rater file;
- inventory revealed only after completion.

The task differs from #224 (a verdict per corner, not a label-to-ramp assignment), so the page
is a new file, not a mode of the #224 tool. A unit-level "can't judge" is not needed here:
"can't tell" per corner plus the unit note covers it.

## How to rate

```
python scripts/corner_gallery.py aerial-dates --bundle runs/vancouver/corner_gallery243   # optional, network
python scripts/corner_gallery.py render --bundle runs/vancouver/corner_gallery243 --rater jonf
```

`aerial-dates` writes `aerial_dates.json` (committed): the capture date Esri's imagery
metadata gives for each aerial, at zoom 20 (the 15 cm layer), sampled at the window centre
and its four corners. Queried 2026-10-06, the day after the tiles were fetched: 81 units
2025-09-19, 12 units 2025-04-14, 2 units 2025-06-08, and one unit
(`vancouver:art:n13090709646`) spans two dates. It is Esri's metadata for its *current*
release, not frozen with the tiles, so a republish between tile fetch and query would make a
date wrong; re-querying later can drift for the same reason. The page shows the date under
the aerial; without the file it says the date is unknown. The coarser layers (zoom 12-19)
carry different dates (2025-12-30 here), so the zoom matters.

1. Open `runs/vancouver/corner_gallery243/gallery/jonf/index.html` in a browser. The aerials
   and crops are git-ignored. They are in this worktree, and `build` remakes them.
2. Rate each corner with the radio buttons or the keys: `p` present, `a` absent, `t` can't tell.
   Each key rates the active corner (blue outline, tinted, marked "keys act here") and then
   moves to the next unrated corner, `a` included. Right after `a`, the optional `b` (sidewalk
   and curb, no ramp) or `n` (no sidewalk at the corner) describes the corner just rated
   absent. Once every corner is rated, a further verdict key is refused until you pick a
   corner with `1`-`9`, `j`/`k` or a click, so a verdict is never overwritten by accident.
   Click or press Enter on a crop to enlarge it.

   (The first version kept `a` on the same corner so that `b`/`n` could follow; a `p` typed
   next then silently overwrote the absent verdict. The review of PR #141 caught this before
   any rating. The key logic is `KEYS_JS` in `scripts/corner_gallery_page.py`, tested under
   node.)
3. Press `c` (or the button) to complete the unit. That shows the inventory.
4. Continue with `→` or "Next to do". Work is saved in the browser as you go.
5. Press **Export**, and save the download as
   `runs/vancouver/corner_gallery243/verdicts__jonf.json`.
6. Run `python scripts/corner_gallery_score.py runs/vancouver/corner_gallery243`, then commit
   the verdicts file. `runs/**` is git-ignored, so use `git add -f` (see below). `score/` is
   written for reading and is not committed (rerun the scorer to regenerate it). The scorer
   first compares the verdicts file with its committed version (`git show HEAD:...`) and
   **refuses** (exit 2, nothing written) if a blind read frozen in HEAD is missing or changed:
   an Export from a stale page drops the units rated since that page was rendered, and the
   smaller file still validates (final review SF1 of PR #141). Re-render, reload and export
   again; `--allow-blind-change` overrides for a deliberate change. Every export records
   `prefilled_from_sha256`, the sha256 of the file its page was rendered from (null when there
   was none), so the refusal can say the page was stale. Outside a git checkout, or before the
   file's first commit, the scorer prints a note and scores.

To revise later, re-run `render`: it prefills from the verdicts file. A second rater uses their
own `--rater` name and file. Agreement is computed with `--a jonf --b <name>`.

### Rating on another machine

The images (95 aerials + 813 crops, ~44 MB) are git-ignored. A copy of exactly the files in
`images.sha256` is on makelab2 (private, not published):
`/homes/gws/jonf/rampnet_share/corner_gallery243/corner_gallery243_images.tgz`
(sha256 `572b6bbc79a633dc6559e5fc1be314c3085b8dee1b3d82a35292aa79c9aa61eb`).

```
git fetch origin && git checkout corner-gallery-243 && git pull
scp makelab2:/homes/gws/jonf/rampnet_share/corner_gallery243/corner_gallery243_images.tgz .
tar -xzf corner_gallery243_images.tgz -C runs/vancouver/corner_gallery243
python scripts/corner_gallery.py check  --bundle runs/vancouver/corner_gallery243   # must say OK
python scripts/corner_gallery.py render --bundle runs/vancouver/corner_gallery243 --rater jonf
```

`render` prefills the page from the committed `verdicts__jonf.json`, so the ratings carry
over. Progress lives in the browser until Export: **export and commit before switching
machines**, or the other machine starts from the last committed file. **After every `git
pull`, re-run `render` and reload every open tab of the page**: the page knows the file only
as it was when `render` ran, so a tab left open across a machine switch, or an `index.html`
reopened without re-rendering, exports without the units rated elsewhere since (the scorer
refuses such a file; step 6). On a machine that has
already rated in its browser, a unit worked on locally wins over the file and the page
reports the conflict, **except the blind read**: once a unit's blind read is frozen in the
file, it is kept (review S1 of PR #141). A browser that never completed the unit takes the
file's entry whole; one whose own blind read differs keeps its later work but takes the
file's blind read, with the edit flags set from the data. Export warns before replacing a
file entry that differs.

## Changes made during the jonf pass (2026-10-06)

Jon rated 28 of 95 units, exported (committed as the partial `verdicts__jonf.json`,
`f0f467b`), and asked for these. None changes `items.jsonl`, the images or the rubric text
(the scorer checks the rubric text, so a mid-pass edit would mix two definitions).

- **Absence reasons after the reveal are not a re-adjudication.** `b` / `n` (and the reason
  radios) now work on a completed unit without reopening it. A reasons-only change sets
  `kinds_after_inventory`, not `edited_after_inventory`; the page re-flags older saved state
  on load, and the scorer measures "verdict edited" from the data (blind vs final verdicts),
  so the one partial-file unit flagged edited for adding reasons only
  (`vancouver:res:n3372672500`) now reads as a reasons-only change. A reason added after the
  reveal (to a corner rated absent before and after, with no blind reason) is reported on its
  own row and never enters the blind kinds of absence or present/absent. It is kept apart
  because the reveal can bias it: the inventory has no sidewalk field, but an `NA` status
  with no `RAMPTYPE` invites the reading "no sidewalk here", and `no_sidewalk` vs
  `curb_no_ramp` is the corner-level evidence on what `NA` means. (The first version merged
  these reasons into the blind kinds; review S3 of PR #141. On the 50-unit file the only such
  unit, `res:n3372672500`, is a false absence, so no `NA` number moved.)
  **Why it was needed:** the `b` / `n` keys were only announced to screen readers and listed
  in the help panel, and Jon found them around unit 24, so 32 of 37 absent `NA` corners in
  the partial file had no reason.
- **Visible prompt after `a`**: "Add why (optional): b ... n ...".
- **Complete button renamed** "Done: lock ratings (c)" with a one-line hint, because it was
  unclear that it had to be pressed once per unit.
- **Look-around links** in the enlarged crop, both to the same pano at the same heading, for
  when the crop framing is too tight to judge (Jon's notes on `art:n47331236`,
  `res:n47231852`, `sig:n1642453647`): **Project Sidewalk Explore**
  (`/explore?lat&lng&panoId&heading&pitch`), which serves PS's cached copy when Google has
  expired the pano, and **Google Street View** (the `@lat,lng,3a,...,/data=!3m4!1e1!3m2!1s<pano>`
  form). A first version linked only Google's `api=1&map_action=pano` URL and Jon got "No
  street view imagery available here". Probed 2026-10-06: 4 of 24 sampled panos return a
  blank tile from Google (expired), and all 4 have a PS backup (`/backupImage/<pano>/metadata`
  200); the backup image itself needs a signed URL, so Explore is the way in. Either way the
  view is served live, so opening a link now flags the unit `used_live_imagery` (see Scoring).
  **Explore writes to production.** `ExploreController.explore` starts an address drop-in for
  any visitor, logged in or not, so every click on the Explore link writes a mission row and an
  activity log on the production Vancouver server, and a label placed there is a real label.
  Look only: a rater (Jon or a second rater) must not label in Explore. The link text says so,
  and both links say they open a new tab.
- **Rubric clarification (no rubric text change):** a curb cut where a shared-use path or
  trail meets the street serves pedestrians and counts as **present**. A feature that is for
  bikes only is **can't tell**, with a note. Raised by Jon's notes on `art:n47270030` (a
  bike-style ramp) and `art:n47190958` (a path with a seamless transition to the street).
  Units rated before this clarification keep their verdicts; check those two notes when
  adjudicating.

## Scoring (`scripts/corner_gallery_score.py`)

Scoring uses the **blind** verdicts of every unit that has them. They are frozen at a unit's
first completion, so a unit that was completed, reopened and not completed again is still
scored on them (the first version dropped such units). The final verdicts, from complete units
only, are reported as a sensitivity read. Agreement between two raters uses the units where
both have blind verdicts. The unit outcome is:
- **present** if any corner is present;
- **absent** if every corner is absent;
- **undetermined** otherwise (no corner present, at least one can't tell).

- **NA with no RAMPTYPE.** The share of sampled units the rater confirms as no ramp is
  absent / (absent + present), with a Wilson 95% interval. Undetermined units are counted and
  left out. A conservative share (absent / all complete) is printed beside it. A corner-level
  read covers the corners that hold the `NA` points, with the kind of absence.
- **False absences.** Each of the 35 units gets one class:
  - `miss_at_inventory_corner`: a corner holding an `Available` point is present;
  - `miss_elsewhere`: another corner is present;
  - `artifact_built_after_imagery`: no corner is present, every inventory corner is absent,
    and every `Available` point there was installed (by month) after the newest crop shown
    at its corner. The scorer also reports each such corner's `newest_available` beside it
    (`dating` in `score.json`) and flags any corner where the two differ;
  - `artifact_inventory_or_geometry`: no corner is present and every inventory corner is
    absent, in any other case;
  - `undetermined`.

  Real misses are the two `miss_*` classes. Artifacts are the two `artifact_*` classes.
- **Control.** The share of clean units rated absent at every corner, with a Wilson interval,
  plus the corner-level read.
- **Stratified estimate (descriptive).** Absence precision against the rater over the three
  strata, weighted by population size. The 11 absent units outside every stratum are named in
  the report. Its 95% interval is **conservative and ad hoc**: a normal interval around the
  unadjusted estimate, with Agresti-Coull per-stratum variances ((k+2)/(n+4)), the finite
  population correction, and clipping to [0, 1]. It is not the decision rule; a standard
  interval for a stratified proportion near a boundary (Korn-Graubard, or a stratified Wilson
  interval) would replace it if it ever were.
- **Agreement** (two files). Corner-level percent agreement and Cohen's kappa over the three
  values, and unit-outcome agreement.
- **Live imagery** (review S2 of PR #141). A unit whose rater looked at live imagery is
  flagged `used_live_imagery`. The page ticks it when a look-around link is opened, and the
  rater can tick or clear it by hand (checkbox under the unit note). The 50 units Jon rated
  before the flag existed are covered by a hand-recorded sidecar,
  `live_imagery__jonf.json`, derived **only** from his unit notes: three units, each entry
  quoting the note it rests on (`res:n47287760` "I went into PS and explored area",
  `res:n47311369` "I checked on Google Street View for Corner 0", `res:n47362725` "I verified
  Corner 1 in Project Sidewalk"). `verdicts__jonf.json` is not edited. The scorer takes the
  union of the flag and the sidecar, names the units, marks them in the per-unit table, and
  repeats the `NA`, false-absence and control reads without them (`blind_without_live` in
  `score.json`). On the 50-unit file, dropping the three moves the false absences from 3 real
  misses of 15 decided to 2 of 13 (`res:n47287760` is one of the three real misses); the `NA`
  read (24/24) and the control (8/8 vs 7/7) do not move.

`validate` refuses a file with:
- the wrong schema, an `items_sha256` other than the bundle's, a `rubric_version` other than 1,
  or no rubric text;
- an unknown unit or corner;
- a verdict outside the three values, or an `absent_kind` without `absent`;
- a complete unit with an unrated corner or without blind verdicts;
- a complete unit whose `inventory_seen` is false;
- a negative `elapsed_s`;
- a rubric text other than the bundle's (`snapshot.json`), so an edit without a version bump
  cannot mix two definitions;
- final verdicts that differ from the blind ones while `edited_after_inventory` is false (the
  page sets that flag on any edit after the reveal, so this means the file was altered);
- absence reasons that differ from the blind ones while `kinds_after_inventory` is false (the
  page sets that flag on a reasons-only change after the reveal);
- a flag (`complete`, `inventory_seen`, `edited_after_inventory`, `kinds_after_inventory`,
  `used_live_imagery`) that is not a boolean.

A `live_imagery__<rater>.json` sidecar is refused when its schema, rater or `items_sha256` is
wrong, when it names a unit the verdicts file has not rated, or when an entry's `note_quote`
is not in that unit's note.

## Which `NA` read is primary

**Pending the ratings.** This section will state which read is primary, and why, once all 95
units are rated and scored (the committed `verdicts__jonf.json` is partial: 50 of 95, and the
scorer marks its report INTERIM). Nothing here is decided yet.

## What is a guess

- The corner point is a geometric construction (12 m along the bisector). A ramp can sit
  several metres away. The rubric says to judge the whole sector.
- The crop projection ignores pitch and roll and assumes 2.5 m. The ring can be off by a few
  degrees; the crops are 60 deg wide for that reason.
- The `artifact_built_after_imagery` class compares months. An `INSTDATE` can be a record
  date rather than a build date.
- The rated imagery is the newest crop shown, which is the newest capture in the 40 m pool.
  It is not necessarily the newest imagery that exists: GSV can hold a newer pano that no
  #56 or #241 run enumerated. The #241 panos are mostly 2022-2024; the inventory can be newer.
  **Except where the rater used live imagery.** The look-around links open Google Street View
  or Project Sidewalk Explore live, where the rater can also walk to other panos or change
  date. That imagery is neither pinned by `images.sha256` nor dated, so for those units the
  `artifact_built_after_imagery` test (install date vs the newest crop shown) compares with
  the wrong imagery, and a ramp seen only in newer live imagery scores as a real miss. Such
  units carry `used_live_imagery` (see "Live imagery" under Scoring) and the report gives
  every headline read with and without them.

## Caveats

- One city, one rig (GSV), one rater. With one rater, the control share measures the rater
  against the inventory, not rater reliability.
- The `na_noramp` sample is 40 of 944. Its Wilson interval is wide by construction: about
  ±0.1 if the share is near 0.9, about ±0.15 near 0.5. That is enough to tell "mostly no ramp"
  from "mostly ramp", and not enough to pin a rate.
- The 35 false absences are a census of the target units, not a sample.
- Corner sectors carry the amendment A1 over-merge limitation of the corner build.

## Reproduce

```
# inputs: the merged #241 build (corners_full.jsonl is untracked; its sha256 is in
# snapshot.json), the #56 and #241 results.jsonl and the inventory, at the paths in
# runs/vancouver/corner_inventory_posdetect241/build.json (each sha256 is checked)
python scripts/corner_gallery.py build \
    --corners-full ../posdetect241/runs/vancouver/corner_inventory_posdetect241/corners_full.jsonl \
    --bundle runs/vancouver/corner_gallery243 \
    --seed-tiles ../../sal-cluster-review/runs/vancouver/cluster_review/tiles
python scripts/corner_gallery.py render --bundle runs/vancouver/corner_gallery243 --rater jonf
python scripts/corner_gallery_score.py runs/vancouver/corner_gallery243
```

**Rebuilt at [6c020fa](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/commit/6c020fa)
from `build.json` `da61166a...`:** `corners_full.jsonl` (`ec206060...681f8d`) and `items.jsonl`
(`bbed3d6d...1459`) came out byte-identical (verified in the deep review of 2026-11-03, with
`build --skip-aerial --no-crops`). `snapshot.json` still records the earlier `build_json` sha
`b0d8fcdd...`; it is deliberately not rebuilt, because its hashes are what the verdicts are
bound to.

`build` cuts the crops on makelab2 from the PS store (`--local-panos` cuts them from a local
sharded copy). It never re-samples an existing `items.jsonl` with a different unit list.

**Image manifest.** The aerials and crops are git-ignored, and Esri World Imagery changes over
time, so a rebuild or a second rater's copy could show different pixels without anyone
noticing. `build` therefore writes `images.sha256` (sha256sum format) over every aerial and
crop the page shows, taken from the files on disk, and records its sha256 and counts in
`snapshot.json` (`images`). The committed manifest was taken from the exact files Jon rates
(95 aerials, 813 crops). Verify a copy with

```
python scripts/corner_gallery.py check --bundle runs/vancouver/corner_gallery243
# or, from the bundle dir: sha256sum -c images.sha256
```

`render` runs the same check and refuses to write a page over images that differ
(`--allow-image-drift` overrides, for looking only).

The tracked files are `items.jsonl`, `snapshot.json`, `images.sha256`, `aerial_dates.json`,
`report.md`, `verdicts__*.json` and `live_imagery__*.json` (`score/` is not tracked; the
scorer regenerates it). `runs/**` is git-ignored, so they were added with `git add -f`. A `.gitignore` rule
can replace that once labeler PR #135 merges. The aerials, crops, tile cache and `gallery/`
stay local.
