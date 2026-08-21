# Handoff: multi-view fusion (issue #27, stages 2–3)

**Written 2026-08-03. Untracked scratch — do not commit.** Every number below was
measured today against the five scored cities, not carried over from an earlier
estimate. Where an earlier claim was wrong, this file says so rather than quietly
replacing it, because the wrong version is already written down in a few places.

## State in one paragraph

Stages 2 and 3 are **built, measured, and written up**. Branch `multi-view-association`
(PR #35, open) carries `geo.py`, `scripts/fuse_sites.py`, `scripts/eval_sites.py`, and
now `scripts/site_explorer.py`. Five cities are scored: paterson, gainesville, sao_paulo,
bend (GSV) and richmond (Mapillary). The results comment is posted on
[#27](https://github.com/ProjectSidewalk/sidewalk-auto-labeler/issues/27#issuecomment-5169230788).
One new systematic was found and filed as
[RampNet#101](https://github.com/ProjectSidewalk/RampNet/issues/101). **Nothing is
committed for the explorer yet** and `send_to_ps.py` is deliberately untouched — no
submission-policy decision has been made.

## The correction that matters most

**Do not describe #27 as "+11–21 points of recall".** That framing is in the older
notes and in the pre-2026-08-03 memory, and it compares against a baseline production
never operated at.

`own-view recall` (0.72–0.83) is a **per-pano** metric. Production runs every pano and
submits every operational detection, so a city already receives the *union* of views.
Measured union coverage — any operational detection raycasting within 5 m of a GT ramp:

| city | own-view | **union today, no fusion** | fused | decoy @30 m |
| :--- | ---: | ---: | ---: | ---: |
| paterson | 0.763 | **1.000** | 0.957 | 0.03 |
| gainesville | 0.785 | **0.945** | 0.927 | 0.08 |
| sao_paulo | 0.722 | **0.980** | 0.933 | 0.04 |
| richmond | 0.830 | **0.972** | 0.941 | 0.11 |
| bend | 0.818 | **0.983** | 0.956 | 0.05 |

The decoy column is the control: displace every GT point 30 m in eight directions and
re-measure. Coverage is real, not the birthday problem. Fusion **costs** 1.5–4.7 points
relative to the raw union (gating + residual rejection + one-to-one matching are
stricter than "a ray landed nearby").

**The real case for #27**, in three numbers:

1. **Label economy** — operational detections per physical ramp collapsing to one site:
   gainesville 2.33, sao_paulo 2.78, paterson 3.35, bend 3.63, richmond 6.07.
2. **Placement tail** — p90 error, fused vs. best single view (what PS derives lat/lng
   from today): paterson 3.51/4.31, gainesville 3.10/3.81, sao_paulo 2.84/3.74,
   richmond 3.86/5.55, bend 2.70/3.72 m. ~20–30% tighter. **Median is a wash** and is
   slightly *worse* on gainesville and sao_paulo — averaging pulls in far noisy views.
3. **Support signal** — ghost check, sao_paulo: 0.967 of verdict-true detections have
   other-pano support vs 0.636 of verdict-false.

Caveat to keep attached to #2: GT "position" is itself a single-view raycast of a
reviewer's pixel mark, so these are *agreement* numbers, not absolute accuracy. There is
no absolute localization measurement yet — see "open threads".

## Counting ramps: split vs merge

- **Over-split 2.6–6.2%**, measured only on *isolated* GT ramps (no other GT ramp within
  5 m, so a second site cannot be a real second ramp): gainesville .026, sao_paulo .046,
  bend .051, paterson .054, richmond .062. GT-free corroboration: operational site pairs
  closer than **0.5 m** — which cannot be distinct ramps — are 0.0–0.8% of sites
  (paterson 71, bend 24, richmond 8, gainesville 1, sao_paulo 1).
- **Over-merge 17–33%** of same-pano GT pairs under 5 m: paterson 16/93, gainesville 3/9,
  sao_paulo 12/58, richmond 7/30, bend 6/31. **This is the weak spot**, and it is
  SidewalkWebpage#4706 reappearing on our side of the wire.

**Trap:** counting "sites within 2.5 m of a GT ramp" looks like a split/merge metric but
just reproduces each city's dual-ramp density — it gave paterson a 34% "error rate" that
was pure artifact. Don't re-derive it that way.

## The new systematic (RampNet#101)

Within a site, each member's signed along-ray residual grows **linearly with its range**.
Pooled within-site fixed-effects slope, sites with ≥3 refit members spanning ≥4 m:

| paterson | gainesville | richmond | bend | sao_paulo |
| ---: | ---: | ---: | ---: | ---: |
| +0.1159 | +0.1179 | +0.1275 | +0.1082 | +0.0572 m/m |

All 50–90σ. Far views land ~11% too far out; the disagreement drags every fused
position. **Only multi-view can see this** — one ray cannot.

Two things already ruled out or corroborated, so nobody repeats them:

- **The camera-height hypothesis FAILS.** +0.116 implies ~2.33 m against the assumed
  2.6 m, and re-fusing at 2.10–2.60 m does lower raw within-site scatter around 2.33 m —
  but lowering the height shrinks every range proportionally, so raw metres shrink for
  free. Normalize by the assumed height and **2.60 m is monotonically best** on both a
  GSV and a Mapillary city. Not the constant.
- **RampNet already half-saw it.** `docs/detection_recall_analysis.md` reports
  flat-ground vs DA3 metric depth agreeing only "to within 6.5–8.5%", and geometry
  putting four richmond ramps *above the horizon*. Read then as rig noise; multi-view
  says the residual is proportional, i.e. a scale error, i.e. a one-constant fix.
- Note the two repos disagree on the constant: **RampNet assumes 2.5 m, `geo.py` 2.6 m.**

**Decisive next experiment** (tooling already exists in RampNet): regress DA3 metric
depth on flat-ground distance and read the **slope**, not ρ. Slope ≠ 1 → geometry is
biased and calibratable, and the distance buckets in `detection_recall_analysis.md` §1
plus `precision_by_distance.py` are stretched. Slope ≈ 1 with the multi-view trend
surviving → the heatmap argmax drifts off the ground-contact point as ramps shrink,
which is a model property and lands next to RampNet#83.

## Tooling added today

`scripts/site_explorer.py` — one HTML card per fused site: a crop from every member
view, a plan view (cameras, rays, per-ray 1σ ellipses, fused covariance, GT points,
nearby sites), the GT verdict, and a neighbour panel so over-split is visible at all.

Pixels come from the **makelab2 native-res archive**, not the live APIs: it holds every
pano of every finished run, covers Mapillary with no token or expiring signed URL, and
is immune to pano decay (bend's archive is 78,552 vs 78,560 processed — those 8 are
already gone from Google). Crops are cut *on makelab2* (48 cores, Pillow 11.3 in
`$ROOT/.venv`) and pulled as one tarball — ~100 KB/view instead of ~16 MB — then cached
under `runs/<city>/explorer/crops/`, so re-renders are offline.

```bash
python scripts/site_explorer.py richmond                    # 40 GT-seen sites
python scripts/site_explorer.py sao_paulo --select fp       # false positives
python scripts/site_explorer.py richmond --select fragment  # over-split suspects
python scripts/site_explorer.py richmond --inline           # one shareable file
python scripts/site_explorer.py paterson --local-panos ../RampNet/benchmark/paterson/panos
```

Already rendered locally (all under the gitignored `runs/**`, so they never appear in
`git status`): `runs/richmond/explorer/` (40 sites), `runs/richmond/explorer_fp/` (6),
`runs/richmond/explorer_frag/` (12 — closest pair 0.11 m).

## Open threads, roughly in priority order

1. **RampNet#101** — the DA3 slope experiment. Largest named systematic on world
   position; blocks credible mensuration.
2. **Over-merge / dual ramps** — 17–33% is the real quality gap. Pairs with
   SidewalkWebpage#4706 (the server half).
3. **PR #33 (gap-fill link closure)** — independent and still open. Merging it and
   re-running `--gap-fill-only` closes the ~1% missing-view gap before any production
   fuse. Retrofit of paterson/gainesville is an open decision; bend is stale.
4. **Absolute localization is unmeasured** — everything is agreement-with-GT-raycast.
   Filed as **labeler#36** (leave-one-out reprojection). Two variants, and the
   distinction is the whole point: the **GT-free** one compares against the detector's
   own peak in the held-out view, measures *self-consistency*, and runs on every
   multi-view site in every city with no labelling; the **GT-anchored** one compares
   against the reviewer's pixel mark, measures *accuracy*, and is capped at 125 panos.
   Needs one new primitive — `geo.py` has only the forward raycast, no
   `ground_point_to_pixel` inverse.
5. **Submission policy / stage 4** — deliberately still late-bound. `sites.jsonl` carries
   cluster + members; `send_to_ps.py` untouched. Stage-4 promotion headroom is modest
   (k≥2 @ f=0.25: gainesville +1.8 pts, sao_paulo +1.2, paterson saturated); its value is
   the demotion signal, not recall.
6. **Mensuration (RampNet#83/#86)** — width needs the model to emit *extent*; fusing
   points gives a better point, never a size. Worth knowing when it lands: two edges seen
   from the *same* camera share that camera's GPS error, so it cancels in the difference —
   width could be far more accurate than our ~1–1.5 m absolute position suggests.
7. **Pseudo-label mining — RampNet#102.** The strategic argument for this whole layer:
   a site with ≥3 corroborating views is near-certainly real, so any nearby pano that
   detected nothing is a miss *at a known world position* = a training example targeted
   at a failure, with no human review, in cities with no inventory. Measured yield
   (upper bound, proximity as a visibility proxy): ≤10 m gives paterson 1,091 /
   gainesville 499 / sao_paulo 437 / richmond 1,210 / bend 1,281; ≤15 m gives
   3,974 / 1,652 / 1,382 / 3,011 / 5,448 (0.11–0.32× each city's operational
   detections). **Thousands per city, not millions** — the case is per-label value, not
   volume. **RampNet#101 is a hard prerequisite**: mining without it bakes the ~11%
   range-scale bias into the targets as *systematic* error. Occlusion is unmodelled and
   is the real ceiling. First step is not the miner — it is measuring mined *precision*
   against the existing missed-ramp marks in the 125 judged panos.

## Things not done / not verified

- ~~`pytest` was not run~~ — **run 2026-08-03, 101 tests pass, exit 0.** `conda` is on
  PATH in neither the Bash nor the PowerShell tool shell, and `!conda activate` fails the
  same way. Skip conda entirely and call the env's interpreter directly:

  ```powershell
  & 'C:\Users\jonf\anaconda3\envs\sidewalk-auto-labeler\python.exe' -m pytest
  ```

  `pytest.ini` sets `addopts = -q`, so a clean run prints only dots and no summary line —
  **branch on the exit code, not on finding "N passed" in the output.**
- ~~`scripts/site_explorer.py` and the CLAUDE.md entry are uncommitted~~ — committed as
  `e0e6cc3` on `multi-view-association`, **not pushed**.
- `runs/sao_paulo/` is untracked but its `manifest.json` + `area.geojson` are un-ignored
  by the `.gitignore` exceptions, i.e. they are *meant* to be tracked — a loose end from
  the São Paulo deployment, not this thread. **Never `git add -A` in this repo**: it
  would sweep that run's provenance (and the two smoke dirs) into an unrelated commit.
- Measurement scripts live only in the session scratchpad, not the repo:
  `crispness.py` (union + placement), `nullcheck.py` (decoy control), `splitmerge2.py`
  (isolated-ramp fragmentation), `tightpairs.py` (<0.5 m pairs), `heightbias.py` +
  `heightsweep.py` (RampNet#101). If any of these numbers need re-deriving, the method is
  described fully enough in #27 and RampNet#101 to rebuild them from `sites.jsonl`.
