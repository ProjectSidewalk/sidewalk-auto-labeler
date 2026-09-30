DRAFT — Jon to post (on RampNet#224, only with Jon's OK; nothing below has been posted)

## Pre-registration: corner-level cluster review, rubric v1

Committed before any unit was reviewed and before any assignment-based number existed:
rubric in `benchmark/RUBRICS.md` §6, full protocol (sampling, schemas, metrics, decision rule,
agreement, calibration) in `docs/cluster_review_protocol.md` on branch `cluster-review-224`.

**Unit.** An OSM intersection (node with ≥ 3 street legs; nodes within 25 m merge into one unit)
or a mid-block point (> 60 m from every intersection node), window radius 30 m, labels in the
window by their SERVER position. Types: `signalised` (a signal node merged in or within 30 m),
`residential` (every leg residential / living_street / unclassified), `arterial`, `mid_block`.
Eligible only inside the area and ≤ 20 m from an open PS street.

**Draw.** Seed 224; 20 units per stratum, 3 of them from no-label units (15%); ≥ 60 m between any
two unit centres across strata; no top-ups. Pilot = 30 units (8 / 7 / 8 / 7) by sha1(corner_id),
4 no-label, all double-rated; the double-rated pass's seed alternates fusion / deployed in hash order.
Full pass: 20% of the rest double-rated the same way.

**One rater (Amendment 2).** There is a single rater for now. "Double-rated" means the same rater
re-reviews the unit ≥ 7 days later, blind to the first pass, seeded with its alternate arm. The
agreement below is therefore **intra-rater** (test-retest), not inter-rater. The first pass is
seeded `deployed`, so any anchoring favours the baseline, and a PASS is conservative with respect
to it. The Vancouver inventory calibration is the only check independent of the rater.

**Rubric v1.** A dual-direction apron is two ramps (#116 box rule); a driveway is not a ramp; an
unlabelled ramp is one `uncovered` point at the ramp; `unsure` abstains from every metric; a unit
counts only when attested `complete`; crops 45° / 512 px (4096×2048-equivalent); inventory points
hidden until the unit is complete; a unit edited after the reveal is flagged and dropped from
the inventory calibration. Reviewer time accrues only while the unit is visible and there was input
in the last 60 s.

**Same labels in every arm.** Human labels are excluded from scoring in every arm; a label an
arm does not hold (ps has no humans; fusion drops 34 labels on unplaceable panos) is scored as
that arm's singleton cluster, so coverage is arm-independent and only split / merge / validity
differ.

**Metrics.** Per GT ramp: clusters holding its labels → covered (≥ 1), split (≥ 2); per cluster:
ramps its in-window labels span → merge (≥ 2 ramps; the ≥ 2-labels-each variant beside it);
validity (all-`not_ramp` clusters); coverage = ramps the arm holds / (ramps + sure uncovered).
Pooled and by stratum; arms paired per ramp (fixed / broken / both / neither).

**Decision rule.** Rater A's full pass: `fusion_server+attach` vs `ps @ 7.5 m` on the same units —
split lower by ≥ 0.05 AND merge not higher by > 0.01 AND coverage not lower by > 0.01 → PASS, else
NOT ESTABLISHED. `deployed`, `fusion_server` and `ps @ 10 / 12.5 / 15 m` reported alongside.

**Agreement (intra-rater, see above).** Pairwise same-ramp agreement over label pairs both raters put in ramps (Wilson CI;
also split by same / different seed), Cohen's κ on not_ramp vs ramp, uncovered-point counts per
unit. Pilot proceeds if agreement ≥ 0.90 and κ ≥ 0.6 (a guess; revisable once, as rubric v2,
before the full pass).

**Calibration.** Vancouver inventory: reviewer ramps vs inventory points one-to-one within 5 m,
both directions; `assignment_metrics` vs `inventory_metrics` on the same units.

**Guesses, labelled as such.** Reviewer time (~1 min/unit) is unmeasured until the pilot; the
thresholds and shares above are judgment calls.

Decisions taken without Jon (all reversible): Esri World Imagery aerials with attribution; the 33
human labels reviewed in-window; close nodes merged at 25 m; the eligibility rule; stale
double-clustered labels seeded to the lower cluster id.
