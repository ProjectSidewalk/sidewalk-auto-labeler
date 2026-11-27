# Rig persistence: finding camera-rig artifacts by sequence persistence (overnight, `rig_persistence`)

Status: PRE-REGISTRATION written 2026-11-26 before any flag was computed on real data. Results
are appended below it; the pre-registration is not edited after that (deviations go in dated
amendments).

## Pre-registration

**Question.** Can a rule that needs no hand-set angle, "the same image position fires frame after
frame along one capture sequence", find camera-rig artifacts (roof racks, mounts, logo bands) in
open imagery? Production's answer today is the fixed 49 deg dip mask (`detectors.on_camera_rig`).

**Data.** Stored detections in `runs/<city>/results.jsonl` (read-only, main checkout) for
laurens (tuning), richmond, juneau, clovis, morgantown, annapolis (Mapillary), bayonne, lyon
(Panoramax). The nadir mask is NOT applied anywhere in the rule (it is the thing under test).
Five runs (richmond, clovis, morgantown, annapolis; also bend, not used) predate the 0.1 storage
floor, so their stored detections are >= 0.55 only: there the 0.30 tier IS the 0.55 tier and
support can only come from >= 0.55 detections. This is reported, not corrected.

**Unit definitions.**
- Sequence: `pano.sequence_id` (Mapillary = `source_metadata.sequence`, Panoramax =
  `source_metadata.collection`; a mismatch or a missing id is counted and the pano is dropped
  from the rule, never guessed).
- Frame order: capture time (Mapillary `source_metadata.captured_at` ms; Panoramax
  `source_metadata.datetime`), ties broken by `geovisio:rank_in_collection` then pano id.
- Stationary collapse: walk the sequence in time order; a frame within 2.0 m (haversine, the
  record's lat/lng) of the last KEPT frame joins that frame's group, otherwise it starts a new
  group. A "frame" below means a group. n_frames(S) = number of groups in sequence S (in the run,
  i.e. after the run's own thinning).
- Positions are heatmap pixels: col = x_normalized * 1024, row = y_normalized * 512 (stored
  values are exact multiples of 1/1024, 1/512). Distance is Chebyshev with x wrapped at the seam.

**The rule.** Members are stored detections with confidence >= 0.30 (OPERATIONAL). For a member d
in frame g of sequence S, support(d) = number of distinct frames g' != g of S holding at least one
member within tolerance T of d. d is FLAGGED iff 1 + support(d) >= K AND
(1 + support(d)) / n_frames(S) >= F.

**Tuning (Laurens only, GT-free).** Grid: T in {4, 8, 16} heatmap px (half, one, two coarse
cells of the stride-32 map), K in {3, 4, 5, 6, 8, 10}, F in {0, 0.05, 0.10, 0.20}.
Below = dip > 49 deg (`on_camera_rig(y)` true), above = everything else.
Eligible cells: flag >= 0.80 of Laurens members below 49 deg (bar (a)). Among eligible cells
pick the one flagging the FEWEST Laurens members above 49 deg; ties -> larger K, then larger F,
then smaller T. If no cell is eligible, (a) FAILS and the cell with the highest below-49 recall
is frozen anyway (ties as above) so (b), (c) can still be read. The bundle verdicts are not used
in tuning. The frozen (T, K, F) is then applied unchanged to every other city.

**Pre-registered checks.**
- (a) Laurens: frozen rule flags >= 0.80 of members (>= 0.30) below 49 deg dip.
- (b) Pooled over every split with a `verdicts.json` among laurens_mapillary, richmond, juneau,
  clovis, morgantown, annapolis, bayonne, lyon: flagged share of verdict-TRUE detections at the
  0.55 tier <= 0.2% (point estimate; Wilson 95% CI reported). Detections are joined through
  `eval_sites.judged_gt_panos` (the drift guard; skipped panos counted). The flag a GT detection
  gets is the one computed on the FULL run (members >= 0.30), so it sees the real sequence
  context. Lyon has no `verdicts.json` (only `verdicts_backups/`): it is excluded from (b) and
  reported exploratorily from the newest backup.
- (c) Null, per city: permute the (x, y) of all members across the city's members (seeded,
  seed 20261126 + draw, 20 draws), keeping each member's frame, sequence and confidence; recompute
  flags. PASS iff the MEAN null flagged count <= 0.10 x the real flagged count (max reported). A
  city with zero real flags is "n/a". Expected weakness, stated in advance: where one rig dominates
  the city's position marginal (Laurens), the permuted positions still pile onto the rig spot, so
  the null is conservative there.

**Reported (descriptive, no rule).** Per city: members and flags at 0.30 and 0.55, below/above
49 deg; y-histogram of flagged members (0.025 bins); frame span of flagged clusters (connected
components of flagged members of one sequence linked within T: distinct frames, along-track
metres first->last kept frame); verdict-FALSE detections at 0.55 caught, split below/above 49 deg
(above = a precision gain beyond the mask); ~10 examples (pano_id, x, y, conf) per city,
above-mask flags first.

**Exploratory arms (never gate).** (E1) support counted from every stored detection >= 0.10
(where the floor exists), flagged members still >= 0.30. (E2) Lyon (b) from the newest verdicts
backup.

**What this cannot see.** A rig that fires in fewer than K frames of a sequence (short or heavily
thinned sequences, thinning keeps the newest pano per cell across sequences, so sequences are
fragmented); a rig that moves in the image (a re-mounted camera mid-sequence); real ramps that
persist (a ramp far ahead on a straight drive) are flagged by construction, which is what (b)
measures, but only on judged panos.

### Amendment A (2026-11-26, written AFTER the frozen rule's results were seen; EXPLORATORY)

Why: the frozen rule (T=4, K=3, F=0) passed (a) but failed (b) and (c) everywhere; its flagged
clusters span a median 0.26-1.8 km of drive, i.e. it counts sequence-wide coincidences at popular
image positions, not a fixture riding along. A rig recurs in NEIGHBOURING frames; a coincidence
does not. This variant is a new hypothesis, tuned on Laurens only with the same GT-free selection
rule, then scored once on the other cities with the same checks (a)-(c). Nothing below changes the
pre-registered verdict above.

- Supporters: every stored detection >= 0.10 (the storage floor; in the four pre-floor runs that
  is >= 0.55). Targets (what can be flagged): members >= 0.30, as before.
- Local support s(d) = number of distinct frames g' != g with |g' - g| <= W (frame-group index
  within the sequence) holding a supporter within T heatmap px (Chebyshev, x wrapped).
- Flag iff s(d) >= K.
- Grid: T in {4, 8}, W in {5, 10}, K in {3, 4, 5, 6, 8}. Selection: eligible iff Laurens below-49
  recall >= 0.80; pick the fewest Laurens above-49 flags; ties -> larger K, smaller T, smaller W.
- Null (c): permute (x, y) across ALL stored detections >= 0.10 of the city (targets and
  supporters together), seeded 20261126 + draw, 20 draws.
- Checks (a), (b), (c) and the reporting are otherwise identical to the pre-registration.

---

## Results (computed 2026-11-26)

Inputs as on disk: Laurens 4,495 panos / 19 sequences; Richmond 9,091 / 358; Juneau 4,999 / 37;
Clovis 72,776 / 514; Morgantown 51,692 / 338; Annapolis 53,232 / 669; Bayonne 50,409 / 1,358;
Lyon 174,405 / 1,872. No pano was dropped for a missing or mismatched sequence id or time.
Richmond, Clovis, Morgantown and Annapolis store only >= 0.55 detections (min stored 0.550), so
their 0.30 tier is their 0.55 tier and their support comes from >= 0.55 only. All eight bundles
have a `verdicts.json` (Lyon's was added after the pre-registration was drafted, so Lyon is in
(b) and E2 was not needed); every judged pano joined (0 skipped, 0 partial).

### Pre-registered rule: VERDICT (a) PASS, (b) FAIL, (c) FAIL in all 8 cities

Tuning picked T = 4 px, K = 3, F = 0 (`tuning_grid.csv`; 6 of 72 cells were eligible). Laurens's
rig peaks sit at x 548 / 555-556 (adjacent coarse cells, 7 px apart) and y 403-428, so T = 4 links
only same-cell repeats; the selection rule still preferred it because it flags fewest members above
49 deg (149 vs 324 for T = 8, K = 5).

| check | result | bar | verdict |
|---|---|---|---|
| (a) Laurens below-49 recall | 129 / 158 = 0.816 [0.749, 0.869] | >= 0.80 | PASS |
| (b) verdict-true flagged, 0.55, pooled 8 splits | 146 / 1,482 = 9.85% [8.44, 11.47] | <= 0.2% | **FAIL** |
| (c) null mean / real | 0.22 (Bayonne) to 0.72 (Juneau) | <= 0.10 | **FAIL** in all 8 |

What it flags: 4-17% of every city's members, nearly all in the 0-18 deg dip band (`band_summary.csv`).
Flagged clusters span a median 260 m (Bayonne) to 1.8 km (Lyon) of drive (`cluster_spans.csv`): the
rule counts coincidences at popular image positions over a whole sequence, not a fixture riding
along. E1 (support from >= 0.10) raises Laurens below-49 recall to 157/158 but also raises
above-49 flags (149 -> 354), so it does not help.

### Amendment A (exploratory, local window): (a) PASS, (b) FAIL, (c) PASS in 3 of 8

Tuning picked T = 8 px (one coarse cell), W = +/-10 frames, K = 4 (`amendment_a/tuning_grid.csv`).

| check | result | bar | verdict |
|---|---|---|---|
| (a) Laurens below-49 recall | 138 / 158 = 0.873 [0.813, 0.917] (13 above-49 flags) | >= 0.80 | PASS |
| (b) verdict-true flagged, 0.55, pooled | 32 / 1,482 = 2.16% [1.53, 3.03] | <= 0.2% | **FAIL** |
| (c) null mean / real | PASS Morgantown 0.03, Bayonne 0.045, Lyon 0.044; FAIL Laurens 2.54, Richmond 0.16, Juneau 0.39, Clovis 0.55 (2 flags), Annapolis 0.20 | <= 0.10 | 3 of 8 |

Flags by city, members >= 0.30 (at 0.55 in brackets):

| city | below-49 members | flagged | above-49 members | flagged |
|---|---|---|---|---|
| laurens | 158 [0] | 138 [0] | 1,575 [708] | 13 [7] |
| richmond | 2 | 0 | 9,524 | 17 |
| juneau | 1 [0] | 1 [0] | 4,823 [3,301] | 221 [126] |
| clovis | 0 | 0 | 8,961 | 2 |
| morgantown | 81 | 49 | 10,401 | 92 |
| annapolis | 0 | 0 | 29,188 | 174 |
| bayonne | 15 [1] | 1 [0] | 19,191 [7,448] | 641 [209] |
| lyon | 1,061 [141] | 205 [17] | 162,926 [73,784] | 9,093 [2,605] |

Verdict-labelled detections at 0.55 the rule flags (`gt_check.csv`, `gt_flagged.csv`):

| rule | true flagged / 1,482 | false below-49 flagged / 1 | false above-49 flagged / 93 |
|---|---|---|---|
| pre-registered | 146 | 1 | 8 |
| Amendment A | 32 | 1 | 3 |

On judged panos the flagged detections are 32 true to 4 false under Amendment A (146 to 9 under the
frozen rule): removing them would cost about eight real ramps per false positive caught. The
flagged true ramps sit at y 0.50-0.59 (1-16 deg dip), mostly at x ~0.5 (dead ahead) or x ~0 / 1
(dead behind, across the seam): the straight-drive pitfall. A ramp on the travel axis far away
moves less than one coarse cell between 5-10 m frames, so persistence alone cannot tell it from a
fixture. Excluding that zone would mean putting back a hand-set angle, which is what the question
ruled out.

Null caveat (c), as anticipated: Laurens's 19 sequences all carry the same rig, so permuting
positions across sequences piles them onto the rig spot and the null flags MORE than the real data
(383 vs 151). The across-sequence null cannot test a city-wide rig; it only tests a
sequence-specific one.

### Eyeball check of the three candidates (added 2026-11-27, after the results)

Eight panos per cell, one per sequence (seeded draw, highest-confidence detection within
+/-0.004 x and +/-0.006 y, conf >= 0.30), were cropped from the archived native-resolution
panos. The contact sheets are in `data/rig_persistence/eyeball/`.

| cell | what is there | rig? |
|---|---|---|
| Lyon, 38 deg above horizon, x 0.027 (at the seam, behind the vehicle) | A dish-shaped fixture with a bracket on the Grand Lyon Ladybug vehicle. It is visible in every pano. 206 of the 219 detections there (>= 0.30) are Ladybug, and all 8 drawn panos are in road tunnels, where the fixture is lit against a dark ceiling | **yes**, 8 of 8 |
| Lyon, 27 deg dip, dead ahead | The camera unit mounted on the front of the same green vehicle. 194 of 199 detections are Ladybug | **yes**, 7 of 8. The 8th is a GoPro Max pano from another producer, and it shows a real curb ramp with tactile paving |
| Bayonne, 15.5 deg dip, dead ahead | Street content about 9 m ahead of a GoPro Max car (sig_bayonne): crosswalk ends at curbs, curb edges, a manhole. In 4 of 8 the car's hood is just below the spot | **no**. It is a place where corners recur, plus a confusing hood-to-road edge, not a rig part |

The two Lyon cells are rig artifacts of one vehicle. A cell mask keyed on that vehicle's
producer and camera (`grand lyon` / `grand-lyon`, Point Grey Ladybug) would remove them
without touching other rigs. A global mask would cost the real ramps other cameras see at
the same pixel. Lyon is a training-only city (RampNet#159, no submission), so this matters
for training data, not for production. Bayonne's cell should stay unmasked.

### Candidate rig finds beyond the mask (need eyeballing; GT-unconfirmed except where noted)

From `amendment_a/band_summary.csv`. The most common coarse cell per dip band, and how many
sequences share it:

| city | band | cell (x, y) | flags | sequences | note |
|---|---|---|---|---|---|
| lyon | above horizon | (0.027, 0.289), 38 deg ABOVE horizon | 212 | 19 | the band's flags are 63% Point Grey Ladybug (producer "grand lyon") and 34% GoPro Max ("ecartip"), from a scratch breakdown of the whole band, not this cell; the one judged detection there (`5d9e19f2-38fa-4ceb-a877-d40b9a013d2c`, 0.026, 0.287) is verdict FALSE |
| lyon | 18-49 deg | (0.504, 0.648), 27 deg dip dead ahead | 194 | 14 | a fixed point ahead at that dip cannot be a ramp seen from a moving car across 4+ frames |
| bayonne | 0-18 deg | (0.551, 0.586), 15.5 deg dip ahead | 103 | 16 | judged FALSE example in the bundle: `86278d43-1d90-482d-b1e7-e61b0006a99e` (0.566, 0.586) |
| lyon | below mask | (0.363, 0.773), 49.2 deg | 19 | 11 | just inside the mask |
| morgantown | below mask | (0.590, 0.914) | 32 | 1 | inside the mask, one sequence |
| laurens | below mask | (0.543, 0.820) | 59 | 12 | the known roof rack, re-found |

Bayonne's logo band (y >= 0.791) is not re-found: only 15 members >= 0.30 sit below 49 deg there,
so there is almost nothing to persist. Examples for eyeballing (pano_id, x, y, conf):
`examples.csv` (10 per city, above-mask first, highest confidence: these are mostly far ramps on
the travel axis, the failure mode) and `band_examples.csv` (3 per city and band from the band's
top cell: the rig candidates above).

### Verdict

No. A sequence-persistence rule without a hand-set angle re-finds the Laurens roof rack (0.82
pre-registered, 0.87 with a local window), but it fails the safety bar by 10x (2.2%) to 50x (9.9%)
on verdict-true ramps, because real ramps on the travel axis persist too. It is not a replacement
for the 49 deg mask. What it is good for is a GT-free SCREEN: the local-window variant passes the
null in Lyon, Bayonne and Morgantown and points at fixed cells (Lyon 38 deg above the horizon, Lyon
27 deg ahead, Bayonne 15.5 deg ahead) that recur across 14-19 sequences and that the mask cannot
reach. Those want a human look at the band_examples panos, and if confirmed, a per-rig cell mask
(keyed on producer + camera model), not a persistence filter.

### Caveats

- Thinning (5 m Mapillary, 10 m Bayonne/Lyon, newest per cell across sequences) fragments
  sequences, so frame indices are not uniform in distance; W = 10 frames is 50-100+ m.
- In the four pre-floor runs, support comes from >= 0.55 only, which weakens any rig signal there.
- Cluster spans under Amendment A count flagged members only; their supporters (>= 0.10) are not
  in the span, so spans read short (median 1-3 frames).
- (b) is measured on judged panos only (the bundles' strata), where rigs are rare; it measures the
  cost to real ramps, not rig recall outside Laurens.
- Bayonne's on-disk results.jsonl holds 50,409 panos, more than the 28,524 in
  docs/panoramax-bayonne.md; it was used as is.

### Replication

```bash
# from the repo root; runs/ must hold the archived results.jsonl files (gitignored)
python scripts/rig_persistence.py run \
    --runs-root runs \
    --benchmark-root ../RampNet/benchmark \
    --out docs/overnight/data/rig_persistence          # both rules, ~3 min, deterministic
python -m pytest -q tests/test_rig_persistence.py
```
Two runs produced byte-identical output directories. The pre-registered rule's outputs are in
`docs/overnight/data/rig_persistence/`, Amendment A's in its `amendment_a/` subdirectory.
