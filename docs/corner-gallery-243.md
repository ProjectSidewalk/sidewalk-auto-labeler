# Corner present/absent gallery (RampNet#243)

**Status (2026-10-05): built, not rated.** One rater (Jon) will rate it. The page and the
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
python scripts/corner_gallery.py render --bundle runs/vancouver/corner_gallery243 --rater jonf
```

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
   the verdicts file and `score/`. `runs/**` is git-ignored, so use `git add -f` (see below).

To revise later, re-run `render`: it prefills from the verdicts file. A second rater uses their
own `--rater` name and file. Agreement is computed with `--a jonf --b <name>`.

## Scoring (`scripts/corner_gallery_score.py`)

Scoring uses complete units only and their **blind** verdicts. The final verdicts are reported
as a sensitivity read. The unit outcome is:
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
  the report.
- **Agreement** (two files). Corner-level percent agreement and Cohen's kappa over the three
  values, and unit-outcome agreement.

`validate` refuses a file with:
- the wrong schema, an `items_sha256` other than the bundle's, a `rubric_version` other than 1,
  or no rubric text;
- an unknown unit or corner;
- a verdict outside the three values, or an `absent_kind` without `absent`;
- a complete unit with an unrated corner or without blind verdicts;
- a complete unit whose `inventory_seen` is false;
- a negative `elapsed_s`.

## Which `NA` read is primary

**Pending the ratings.** This section will state which read is primary, and why, once
`verdicts__jonf.json` is committed and scored. Nothing here is decided yet.

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

`build` cuts the crops on makelab2 from the PS store (`--local-panos` cuts them from a local
sharded copy). It never re-samples an existing `items.jsonl` with a different unit list. The
tracked files are `items.jsonl`, `snapshot.json`, `report.md`, `verdicts__*.json` and
`score/`. `runs/**` is git-ignored, so they were added with `git add -f`. A `.gitignore` rule
can replace that once labeler PR #135 merges. The aerials, crops, tile cache and `gallery/`
stay local.
