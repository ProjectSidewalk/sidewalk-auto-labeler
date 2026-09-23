# Camera height: what GSV depth measures, and why per-pano is opt-in (issue #40)

Measured 2026-09-23 on the four GSV runs whose depth payloads are fully harvested
(`runs/<city>/depth/`, 170,932 panos): paterson, bend, gainesville, sao_paulo.

## Summary

- **The depth ground plane is a per-pano rig measurement, but its absolute value is low.**
  Triangulating multi-view ramps from bearings alone (which does not depend on camera
  height) and backing out the height each pano's ray implies gives heights **6–16%
  above** the depth ground plane's distance, city by city. So this is not the
  "exact camera height" #40 assumed, and the #40 headline (ranges 29–35% long at 2.6 m)
  was computed inside the depth frame, i.e. against a depth map that shares the bias.
- **The rig ranking in the depth data is real.** The 2025–26 GSV rig is lower: it
  triangulates to ~1.9–2.0 m in paterson and gainesville, against ~2.5 m for every
  earlier vintage. So the 2.6 m constant is ~2–4% high for pre-2025 imagery and ~25–30%
  high for the new rig. That rig dominates paterson (2025) and gainesville (2026).
- **Using depth heights as-is does not measurably help.** World P/R against RampNet GT
  cannot tell any of the height models apart: every difference is ≤3 points, inside the
  Wilson CIs. Within-site multi-view spread gets *worse* when the association is held
  at 2.6 m, although that test favours 2.6 by construction (see below).
- **Google's stand-in ground is common:** 24,529 of 170,462 non-degenerate payloads
  (16% of bend) have a 100–200-plane reconstruction whose ground is a plane at exactly
  2.500 m with an exactly vertical normal. The existing `degenerate` (≤2 planes) test
  misses all of them. These panos triangulate to ≥2.6 m (2012–2018 imagery reads
  ~2.9 m), which is why unmeasured panos fall back to 2.6 m, not to the measured median.

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
