# Camera height: what GSV depth measures, and why per-pano is opt-in (issue #40)

Measured 2026-09-23 on the four GSV runs whose depth payloads are fully harvested
(`runs/<city>/depth/`, 170,932 panos): paterson, bend, gainesville, sao_paulo.

## Summary

- **The depth ground plane is a per-pano rig measurement, but its absolute value is low.**
  Triangulating multi-view ramps from bearings alone (which does not depend on camera
  height) and backing out the height each pano's ray implies gives heights **6–16%
  above** the depth ground plane's distance, city by city. So this is not the
  "exact camera height" #40 assumed. The #40 headline (ranges 29–35% long at 2.6 m) was
  computed inside the depth frame, against a depth map that shares the bias, so it does
  not hold city-wide. It does hold, roughly, for the new rig (next bullet).
- **The rig ranking in the depth data is real.** The 2025–26 GSV rig is lower: it
  triangulates to ~1.9–2.0 m in paterson and gainesville, against ~2.5 m for every
  earlier vintage. So 2.6 m runs ranges ~2–4% long on pre-2025 imagery and **31–35% long
  on the new rig** (2.6/1.979, 2.6/1.924). That rig is about half of paterson (2025) and
  of gainesville (2026).
- **Using depth heights as-is does not measurably help.** World P/R against RampNet GT
  cannot tell any of the height models apart: every difference is ≤3 points, inside the
  Wilson CIs. Within-site multi-view spread gets *worse* when the association is held
  at 2.6 m, although that test favours 2.6 by construction (see below).
- **Google's stand-in ground is common:** 24,529 of 170,462 non-degenerate payloads
  (14%; 16% in bend alone) have a 100–200-plane reconstruction whose ground is a plane at exactly
  2.500 m with an exactly vertical normal. The existing `degenerate` (≤2 planes) test
  misses all of them. These panos triangulate to ≥2.6 m (2012–2018 imagery reads
  ~2.9 m), which is why unmeasured panos fall back to 2.6 m, not to the measured median.

**Since #44**, per-pano heights are run through pre-registered QC tests. Nothing they found
changes a raycast: one gate only flags, and per-pano output is still #68's. See "QC rule
(#44)" below, which ends with the decision table for the default.

**What shipped:** the pano block now carries the depth-derived height and its provenance
(`camera_height_status`). `geo.PER_PANO` / `--camera-height-m per-pano` raycast with it,
and `fuse_sites.py --implied-height` is the instrument below. **The production default
is unchanged at 2.6 m**, so `sites.jsonl` is byte-identical to before (checked on
paterson).

## The instrument: implied height by bearing-only triangulation

For two operational members of one fused site whose rays cross at ≥30°, the ramp's
ground position is the intersection of the two bearing rays. Each pano's implied height
is then `range × tan(depression)`. Take the median per pano, and compare it with that
pano's depth height by capture year.

**The trap is selection.** The pairs exist only because the association put them in one
site, and association runs under some height model. Implied heights drift toward that
height. Paterson's 2025 rig reads 1.93 m associated at 1.8 m, 1.98 m per-pano, and
2.14 m at 2.6 m. So read it as a fixed-point search: the answer is where the implied
height agrees with the one the association used.

```
python scripts/fuse_sites.py runs/paterson --implied-height                         # at 2.6
python scripts/fuse_sites.py runs/paterson --implied-height --camera-height-m per-pano
python scripts/fuse_sites.py runs/paterson --implied-height --camera-height-m 2.0   # iterate
```

### Median implied / measured, all panos with a measured height

| city | associated at 2.6 m | per-pano | per-pano × 1.08 | per-pano × 1.15 | self-consistent scale |
|---|---:|---:|---:|---:|---:|
| bend | 1.059 | 1.051 | 1.056 | 1.065 | ~1.06 |
| paterson | 1.106 | 1.063 | 1.075 | 1.083 | ~1.08 |
| gainesville | 1.168 | 1.075 | 1.096 | 1.111 | ~1.09–1.10 |
| sao_paulo | 1.150 | 1.102 | 1.125 | 1.145 | ~1.16 |

(The × k columns scale each measured height before associating, by monkeypatch. They
are how the fixed point is located; they are not a shipped option.)

### The new rig, by vintage (associated per-pano)

| city | capture | panos | depth height | implied |
|---|---|---:|---:|---:|
| paterson | 2018–2024 | ~5,000 | 2.35–2.38 | 2.48–2.54 |
| paterson | **2025** | 5,172 | **1.864** | **1.979** |
| gainesville | 2021–2025 | ~2,000 | 2.31–2.34 | 2.44–2.50 |
| gainesville | **2026** | 5,486 | **1.797** | **1.924** |
| bend | 2019–2025 | ~16,800 | 2.35–2.39 | 2.48–2.50 |
| sao_paulo | 2019–2026 | ~5,900 | 1.82–2.29 | 2.19–2.54 |

The implied height barely moves *within* a vintage across depth-height bins. For
example, bend panos whose depth reads 1.5–1.8 m still triangulate to 2.54 m (associated
at 2.6 m). So a single pano's depth deviation from its vintage looks mostly like noise in
the depth, not a different rig.

## GT world P/R under each height model

Benchmark tier (0.55), `mask_rig=False`, GT placed under the same model. Match radius 5 m,
with the 2.5 m recall in brackets.

| city | fixed 2.6 | per-pano | per-pano × 1.08 | fixed 2.3 |
|---|---|---|---|---|
| bend | P 0.957 R 0.956 (0.939) | P 0.957 R 0.954 (0.917) | P 0.957 R 0.957 (0.923) | P 0.957 R 0.957 (0.914) |
| gainesville | P 0.956 R 0.927 (0.872) | P 0.953 R 0.927 (0.853) | P 0.952 R 0.928 (0.861) | P 0.956 R 0.934 (0.872) |
| paterson | P 0.975 R 0.957 (0.872) | P 0.976 R 0.952 (0.908) | P 0.975 R 0.948 (0.896) | P 0.975 R 0.959 (0.902) |
| sao_paulo | P 0.893 R 0.953 (0.886) | P 0.893 R 0.946 (0.876) | P 0.893 R 0.946 (0.896) | P 0.893 R 0.946 (0.870) |

Pool sizes are 219–336 ramps, so the CIs are ±3–4 points. The GT denominator also moves
with the model, because GT marks are raycast under it. As the #27 work already found,
union coverage is 0.93–0.96 whatever the height, so this benchmark cannot resolve a
height change. Reproduce with `eval_sites.py <city> --camera-height-m per-pano --out <dir>`.

## Within-site spread, association frozen at 2.6 m

Within-site pairwise member distance, divided by the pair's mean range:

| city | per-pano | fixed 2.6 | fixed 2.3 |
|---|---:|---:|---:|
| paterson | 0.232 | 0.191 | 0.192 |
| bend | 0.170 | 0.152 | 0.183 |
| gainesville | 0.333 | 0.208 | 0.191 |
| sao_paulo | 0.302 | 0.174 | 0.249 |

This was the first test run, and on its face it rejects per-pano heights. It is biased
toward 2.6 m because the pairs are selected under 2.6 m, which is what led to the
fixed-point design above. It is not in the shipped code.

## Open questions and what would decide the default

1. **Why the depth frame runs short.** The candidates are a multiplicative depth-scale
   error, and a depth origin below the camera (an additive offset: ~0.12–0.26 m here). The
   data cannot separate them yet. It is not the obvious detection-point bias: a ramp that
   sits at sidewalk height would push the implied height *down*, not up.
2. **Which default to ship.** On this evidence the defensible options are (a) keep 2.6 m,
   (b) per-pano × a calibrated scale (~1.08), or (c) a per-vintage constant keyed on the
   depth height (e.g. depth < 2.1 m → ~2.0 m, else 2.5 m). (b) and (c) need a held-out
   check that the GT benchmark cannot supply, such as surveyed ramp positions or the PS
   validated-label placements.
3. **Mapillary.** The same triangulation needs no depth, so it could calibrate a height
   per Mapillary sequence. That is the open half of #40's scope note, and a follow-up.
4. **Ground tilt** is recorded (`ground_tilt_deg`) but not applied.

## QC rule (#44)

Measured 2026-09-25 with `python scripts/height_qc.py` over the same four GSV runs. That
is 145,250 measured panos, of which about 49k have an implied height. The tests and their
verdict rules were posted on #44 before any number ran. The script's docstring records
them and the choices the plan left open: T2–T4 are read on the pooled cities, T1 per city,
and every verdict must hold under both association heights.

`resid = implied − k_city × depth`, with k = 1.06 / 1.08 / 1.10 / 1.16 from the table
above. The full tables, including T3 per city and the post-hoc comparison below, are in
`runs/_pooled/height_qc/report.md` and `runs/<city>/height_qc/`.

**Nothing from these tests changes a raycast.** The PR #81 review found that the one gate
the pooled reading passed does not survive per city, and that its fallback is worse than
the height it replaces. The constant sigma that T4 pre-registered made GT placement worse.
So the shipped rule only flags, and `--camera-height-m per-pano` output is exactly what
#68 produced. Paterson's per-pano `sites.jsonl` hashes the same before and after, and so
does its default `sites.jsonl`.

### Pre-registered verdicts, and what was adopted

| test | question | result | pre-registered verdict | adopted |
|---|---|---|---|---|
| T1 | Does a pano's deviation from its vintage median show up in its implied height? | Pano-weighted Theil–Sen slope, bend / paterson / gainesville / sao_paulo: **0.23 / 0.52 / 0.42 / 0.50** associated per-pano, **0.08 / 0.20 / 0.05 / 0.09** at 2.3 m | **undecided**: ≤ 0.3 in 4 of 4 at 2.3 m, but only 1 of 4 per-pano | a kept height is used as measured |
| T2 | Is the sub-1.5 m tail a measurement? | Median implied/depth for tilt < 6° (n 513) and ≥ 6° (n 62): **1.297 / 1.297** per-pano, **1.53 / 1.78** at 2.3 m | **kept as measured**: not above 1.30 under both | same |
| T3 | Which features flag bad heights? | Pooled, only `\|depth − vintage_median\| ≥ 0.40 m` passes, at a 4.0% flag rate with residual ratio **2.38** per-pano / **3.54** at 2.3 m. The others fail: tilt ≥ 6° 1.54 / 1.74; spread ≥ 0.30 m 1.27 / 1.35 (7.8% flagged); pixel share < 0.15 1.03 / 1.01 (the expected failure); planes < 60 1.23; sky > 0.6 flags nothing. | ship the vintage-deviation gate | **not adopted; it flags only** (below) |
| T4 | Does the plane spread predict the error? | p68 of \|resid\| by spread quartile, pooled: 0.261 / 0.257 / 0.249 / 0.269 m per-pano, 0.271 / 0.261 / 0.252 / 0.281 at 2.3 m | **no**; use a constant sigma of 0.259 m (the kept-pano p68) | **not adopted**: it worsened GT placement (below), so the spread sigma stays |

**Why the T3 gate was not adopted.** The pooled pass was carried by bend, which is 45% of
the pool (65,866 of 145,250 measured). Per city, the gate fails its own rule in two of the
four cities.

| city | flag rate | ratio, per-pano / 2.3 m | passes in this city |
|---|---:|---|---|
| bend | 1.56% | 3.03 / 3.48 | yes |
| paterson | 1.62% | 2.15 / 3.16 | yes |
| gainesville | **11.06%** | 2.34 / 3.63 | **no** (rate above 5%) |
| sao_paulo | 5.19% | **1.38** / 2.06 | **no** (rate, and per-pano ratio) |

Even where the gate flags a worse height, the fallback the plan specified (2.6 m) can be
worse still. That is measured post-hoc below. Choosing a different fallback after seeing
the data would be a new rule, not the pre-registered one, so the gate only flags.
`depth.believe_height` returns `flagged_qc:vintage_deviation` and keeps the height.
`sites_meta.json` counts the flags under `camera_heights.flagged_qc`. The vintage median
comes from years with at least `QC_MIN_VINTAGE_PANOS` = 300 measured panos; undated panos
get none. The function and its plumbing stay in place so a future rule has somewhere to
live.

**Why the T4 constant was not adopted.** T4 was pre-registered and passed, but it only
shows that the spread does not predict the residual. It does not show that 0.259 m
associates better. So the constant was scored against RampNet GT with
`eval_sites.py --camera-height-m per-pano` at the benchmark tier (0.55) and
`mask_rig=False`. Each cell below is main (spread sigma) → branch (constant 0.259 m).
Median and p90 GT-to-site distance are under the production 25 m cap, over the ramps both
arms match within 5 m.

| city | precision | world recall (5 m) | median GT-to-site | p90 GT-to-site | sites |
|---|---|---|---|---|---|
| bend | 0.957 → 0.957 | 0.953 → 0.957 | 0.864 → 0.852 m | **2.201 → 2.318 m** | 14,197 → 13,956 |
| paterson | 0.976 → 0.976 | 0.958 → 0.961 | 0.915 → 0.958 m | **2.753 → 3.051 m** | 13,165 → 12,617 |
| gainesville | 0.953 → 0.953 | 0.926 → 0.926 | 0.979 → 1.066 m | **3.057 → 3.228 m** | 15,612 → 14,750 |
| sao_paulo | 0.893 → 0.893 | 0.950 → 0.942 | 1.115 → 1.111 m | **3.165 → 3.395 m** | 18,224 → 17,398 |

Precision is identical and recall moves within ±1 point, well inside the ±3–4-point CIs.
p90 placement is worse in **all four** cities, by 0.12–0.30 m, and each exceeds the 0.1 m
p90 tolerance that #42 used for the same statistic. The wider sigma loosens the chi-square
gate and merges more views, 4–6% fewer sites, and the extra merges cost placement. So the
constant was reverted.

The earlier justification for it was also wrong. It called the constant "an upper bound
on the depth height's error", but the residual is measured against `k × depth` while the
raycast uses the raw depth. The 6–16% depth-frame scale bias (roughly 0.1–0.35 m) is
therefore **not** inside that sigma.

**Reproducibility.** Default outputs are unchanged: paterson's `sites.jsonl` and
`sites_meta.json` are byte-identical to main. Per-pano `sites.jsonl` is byte-identical to
main too. Only per-pano `sites_meta.json` gains the `flagged_qc` counts, and no script
reads that key. So the committed per-pano artifacts from #76
(`docs/figures/reprojection-residual/data/`), #75 (`runs/gainesville/agree_rate/`) and #68
(the per-pano sections above) still reproduce. They would **not** have reproduced under
the review draft of this change, with its 2.6 m fallback and 0.259 m sigma.

**How firm the verdicts are.** The sensitivity pass moves each of 0.40 m, 6°, 1.30 and 2×
by ±25%, one at a time. The verdict moves in **4 of 8** runs:

- At 0.30 m the vintage gate flags more than 5% and nothing ships.
- At a 2.5× bar it fails, since 2.38 is below the bar.
- At a 1.5× bar the tilt gate ships too.
- At a T2 bar of 0.975 both low groups fall back. This run tells us nothing, because
  every ratio is above it.

T2 is also close under the per-pano association: 1.297 against a bar of 1.30.

**What T1 leaves open.** The two associations disagree in the direction the selection
trap predicts. The per-pano association pulls each pano's implied height toward its own
depth value, which raises the slope. The 2.3 m association pulls every pano toward one
value, which lowers it. Neither reading clears its rule under both associations.

A per-detection range from the depth payload (`depth.ground_range_at`) is **not** used
anywhere here. Its azimuth convention is the subject of #80, so that is deferred to #80.

### POST-HOC: what a flagged pano should be raycast at

This comparison was **not pre-registered**. The PR #81 review asked for it, and it is here
to inform the default-height decision, not as a verdict. Each cell is the median
`|H / implied − 1|` over flagged panos (≥ 0.40 m from their vintage median) for three
choices of H: keep the depth height, fall back to 2.6 m, or use the vintage median.

| association | rig | side of the median | n | keep depth | 2.6 m | vintage median |
|---|---|---|---:|---:|---:|---:|
| per-pano | low | below | 285 | 0.269 | **0.524** | **0.115** |
| per-pano | low | above | 85 | 0.120 | 0.118 | 0.317 |
| per-pano | other | below | 434 | 0.219 | 0.137 | **0.090** |
| 2.3 | low | below | 322 | 0.390 | 0.285 | **0.124** |
| 2.3 | low | above | 87 | 0.115 | 0.108 | 0.315 |
| 2.3 | other | below | 519 | 0.287 | 0.082 | 0.090 |

Of the flagged panos, 5,414 of 5,856 sit **below** their vintage median; gainesville 2026
alone accounts for 2,319. For those, the vintage median is the best of the three choices
in every cell but one, under both associations. The exception is 2.6 m on other vintages
at the 2.3 m association (0.082 against 0.090). The 2.6 m fallback doubles the error on
the low rig under the per-pano association.

Read with T1's leaning toward "believe the vintage" at 2.3 m, this points toward a
per-vintage height rather than a per-pano one. That is option (c) below. It is a post-hoc
observation and has not been pre-registered or tested as a rule.

### Decision table for the default height (the default is not changed here)

**These numbers rest on pre-registered verdicts that are fragile.** The ±25% sensitivity
pass moves them in 4 of 8 runs (see "How firm the verdicts are"). No QC gate is applied,
so every measured height is used.

Range error = H / implied − 1 per pano, where positive means ranges run long. Each cell is
the median signed error / median absolute error. The "low rig" is the vintages whose
median depth height is below 2.1 m: paterson 2025, gainesville 2026, sao_paulo 2021, and 3
bend panos. The options come from "Open questions" above:

- (a) 2.6 m everywhere.
- (b) 1.08 × the depth height.
- (c) depth < 2.1 m → 2.0 m, else 2.5 m.

Unmeasured panos raycast at 2.6 m under every option. "Changed" is the share of all the
run's panos whose height differs from 2.6 m. **T1 said: undecided.** The error is measured
against the implied height, the same instrument the options were calibrated on; it is not
ground truth.

Associated per-pano. Both associations are in `runs/_pooled/height_qc/decision.csv`.

| city | low rig n | (a) | (b) | (c) | other n | (a) | (b) | (c) | changed by (b)/(c) |
|---|---:|---|---|---|---:|---|---|---|---:|
| paterson | 5,315 | +31.1 / 31.1% | +1.9 / 8.2% | +1.2 / 8.2% | 5,742 | +2.6 / 6.6% | +0.8 / 6.6% | −1.3 / 6.4% | 87% |
| gainesville | 6,069 | +33.9 / 33.9% | −0.2 / 10.4% | +4.0 / 9.8% | 3,306 | +3.9 / 7.9% | +1.0 / 7.9% | +0.4 / 7.6% | 81% |
| sao_paulo | 175 | +17.7 / 18.6% | +0.3 / 12.0% | +5.7 / 15.3% | 9,159 | +2.8 / 7.8% | −1.3 / 8.0% | −1.2 / 8.0% | 62% |
| bend | 3 | — | — | — | 19,124 | +3.9 / 6.6% | +2.2 / 6.4% | +0.0 / 5.9% | 84% |

**The two associations do not rank every cell the same way.** At 2.3 m the low-rig
absolute error is:

| city | (a) | (b) | (c) |
|---|---:|---:|---:|
| paterson | 26.6% | 8.0% | 7.7% |
| gainesville | 27.2% | 11.0% | 8.8% |
| sao_paulo | **11.8%** | 14.1% | 15.1% |

So on sao_paulo's small low-rig group (n ≈ 170, all 2021 imagery) the ranking flips, and
(a) is best at 2.3 m. On other vintages the three options differ by at most ~1 point of
absolute error, and their order moves between associations.

Reading it for the decision:

- On the low rig in paterson and gainesville, (a) runs ranges 27–34% long. Both (b) and (c)
  bring the median within ±5.4% under both associations and cut the absolute error to
  8–11%. (c) is as good as or better than (b) there under both associations, which is
  consistent with the post-hoc vintage-median observation above.
- sao_paulo's low rig does not settle it either way.
- On every other vintage, the options differ by only a few percent in median. The absolute
  error stays at 6–8%, because it is dominated by per-pano scatter that no constant
  removes.
- (b) and (c) change 62–87% of panos. Neither has a check independent of the
  implied-height instrument (#40's open question 2 still stands), and GT world P/R could not
  separate the height models.

The choice stays with the default-height decision.
