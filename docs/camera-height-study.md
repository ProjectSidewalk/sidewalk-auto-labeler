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

**Since #44** per-pano also applies a QC rule to each measured height (one gate adopted,
see "QC rule (#44)" below), and the section ends with the decision table for the default.

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

Measured 2026-09-25 with `python scripts/height_qc.py` over the same four GSV runs
(145,250 measured panos; ~49k panos have an implied height). The tests and their verdict rules
were posted on #44 before any number ran. The script's docstring records them and the
choices the plan left open (T2–T4 read on the pooled cities, T1 per city, and "holds under
both association heights" for every verdict). `resid = implied − k_city × depth`, with
k = 1.06 / 1.08 / 1.10 / 1.16 from the table above. The full tables are in
`runs/_pooled/height_qc/report.md` and `runs/<city>/height_qc/`.

| test | question | result | verdict |
|---|---|---|---|
| T1 | does a pano's deviation from its vintage median show up in its implied height? | pano-weighted Theil–Sen slope bend / paterson / gainesville / sao_paulo: **0.23 / 0.52 / 0.42 / 0.50** associated per-pano, **0.08 / 0.20 / 0.05 / 0.09** at 2.3 m | **undecided** (≤ 0.3 in 4 of 4 at 2.3 m, but in 1 of 4 per-pano) |
| T2 | is the sub-1.5 m tail a measurement? | median implied/depth, tilt < 6° (n 513) and ≥ 6° (n 62): **1.297 / 1.297** per-pano, **1.53 / 1.78** at 2.3 m | **kept as measured** (not above 1.30 under both) |
| T3 | which features flag bad heights? | only `\|depth − vintage_median\| ≥ 0.40 m` passes: 4.0% flagged, residual ratio **2.38** per-pano / **3.54** at 2.3 m. Tilt ≥ 6°: 1.54 / 1.74. Spread ≥ 0.30 m: 1.27 / 1.35 (and 7.8% flagged). Pixel share < 0.15: 1.03 / 1.01, failing as predicted. Planes < 60: 1.23. Sky > 0.6 flags no pano | **ship the vintage-deviation gate only** |
| T4 | does the plane spread predict the error? | p68 of \|resid\| by spread quartile, pooled: 0.261 / 0.257 / 0.249 / 0.269 m per-pano (0.271 / 0.261 / 0.252 / 0.281 at 2.3 m) | **no**: constant sigma 0.259 m (kept-pano p68, the larger of the two) |

**Adopted rule** (`depth.believe_height`, applied by `geo.camera_height_for` under
per-pano): a measured height ≥ 0.40 m from the median measured height of its run and
capture year is refused (`rejected_qc:vintage_deviation`) and raycasts at 2.6 m, like an
unmeasured pano. A kept height is used as measured, with sigma max(error model, 0.259 m).
`sites_meta.json`'s `camera_heights` counts the refusals. In paterson that is 492 of
30,325 measured panos. The default (fixed 2.6 m) output is byte-identical: paterson's
`sites.jsonl` hashes the same before and after.

**How firm it is.** The sensitivity pass (each of 0.40 m, 6°, 1.30 and 2× moved ±25%, one at a
time) moves the verdict in four of eight runs:

- At 0.30 m the vintage gate flags more than 5% and no gate ships.
- At a 2.5× bar it fails (2.38 per-pano) and no gate ships.
- At a 1.5× bar the tilt gate ships as well.
- At a T2 bar of 0.975 both low groups are routed to fallback. That run is uninformative:
  it sits below the ratio every pano has.

The tilt ±25%, 0.50 m and 1.625 runs do not move the verdict. So the one adopted gate
passes its pre-registered bar, but not by a wide margin under the per-pano association. T2
is also close under that association: 1.297 against a 1.30 bar.

**What T1 leaves open.** The two associations disagree in the direction the selection trap
predicts. The per-pano association pulls implied heights toward each pano's own depth
height, which raises the slope. The 2.3 m association pulls every pano toward one value,
which lowers it. Neither reading clears its rule under both, so the rule does not replace
a kept pano's height with its vintage median. The vintage median is used only for the
gate.

A per-detection range from the depth payload (`depth.ground_range_at`) is **not** used
anywhere in this rule. Its azimuth convention is under review in #80, so that is deferred
to #80.

### Decision table for the default height (rule applied; the default is not changed here)

Range error = H / implied − 1 per pano: positive means ranges run long. Each cell is the
median signed error / median absolute error. The "low rig" is the vintages whose median
depth height is below 2.1 m: paterson 2025, gainesville 2026, sao_paulo 2021, and 3 bend
panos. The options come from "Open questions" below:

- (a) 2.6 m everywhere.
- (b) 1.08 × the depth height where the rule keeps it.
- (c) depth < 2.1 m → 2.0 m, else 2.5 m, where kept.

Unmeasured and refused panos raycast at 2.6 m under every option. "Changed" is the share
of all the run's panos whose height differs from 2.6 m. **T1 said: undecided.** The
error is measured against the implied height, which is the same instrument the options
were calibrated on. It is not ground truth.

Associated per-pano. Both associations are in `runs/_pooled/height_qc/decision.csv` and
rank the options the same way. At 2.3 m, (a)'s low-rig error is smaller (+26.6 / +27.1 /
+8.1% for paterson / gainesville / sao_paulo), because that association pulls implied
heights toward 2.3 m.

| city | low rig n | (a) | (b) | (c) | other n | (a) | (b) | (c) | changed by (b)/(c) |
|---|---:|---|---|---|---:|---|---|---|---:|
| paterson | 5,315 | +31.1 / 31.1% | +2.2 / 8.3% | +1.2 / 8.2% | 5,742 | +2.6 / 6.6% | +1.0 / 6.6% | −1.2 / 6.4% | 86% |
| gainesville | 6,069 | +33.9 / 33.9% | +1.1 / 10.6% | +4.2 / 10.2% | 3,306 | +3.9 / 7.9% | +1.2 / 7.9% | +0.5 / 7.6% | 72% |
| sao_paulo | 175 | +17.7 / 18.6% | +4.9 / 15.0% | +7.9 / 16.0% | 9,159 | +2.8 / 7.8% | −0.9 / 8.0% | −0.9 / 8.0% | 59% |
| bend | 3 | — | — | — | 19,124 | +3.9 / 6.6% | +2.4 / 6.4% | +0.1 / 5.9% | 83% |

Reading it for the decision:

- On the low rig in paterson and gainesville, (a) runs ranges 27–34% long. Either (b) or
  (c) brings the median within ±4.2% under both associations, and cuts the absolute error
  to 8–11%.
- On every other vintage, all three options are within ~4% of the median. The absolute
  error barely moves (6–8%) because it is dominated by per-pano scatter, which no
  constant removes.
- (b) and (c) change 59–86% of panos. Neither has a check independent of the implied-height
  instrument (#40's open question 2 still stands), and GT world P/R could not separate the
  height models.

The choice stays with the default-height decision.
