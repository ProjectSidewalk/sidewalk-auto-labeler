# Overnight CPU studies (2026-11-26/27)

Pre-registered studies, run on committed data and archived `results.jsonl` files with
no GPU. Each doc's pre-registration was written before any score on real data. Results and
dated amendments are appended below it.

| study | question | verdict |
|---|---|---|
| [rig_persistence.md](rig_persistence.md) | Can "same pixel fires frame after frame along one sequence" find camera-rig artifacts without the 49 deg mask? | **FAIL** as a filter: it flags 2-10% of verdict-true ramps against a 0.2% bar. Keep the mask. As a screen it found two Lyon rig spots on one vehicle (both confirmed by human review, one after an open-street control) and one Bayonne spot that is not a rig (human review agrees) |
| [bearing_refit.md](bearing_refit.md) | Does bearing-only triangulation place multi-view sites better than the flat raycast on Mapillary and Panoramax? | **FAIL** (1 of 3 cities). It helps where camera positions agree (Richmond, GSV), and hurts on Panoramax across sequences (Bayonne) |
| [recall_by_tilt.md](recall_by_tilt.md) | Does RampNet's recall on open-imagery rigs fall with pano tilt? | **NOT ESTABLISHED**: the pooled effect is a city confound and disappears under city fixed effects |
| [best_view_emission.md](best_view_emission.md) | Can one label per fused site, from its best view, replace every per-pano label without losing inventory coverage? | **FAIL** on fresh inventory pulls: p90 placement improves by 0.9-1.7 m and labels drop about 70%, but coverage falls 2.0 (Bend) and 3.7 (Gainesville) points, because dual corners fuse into one site |
| [per_rig_threshold.md](per_rig_threshold.md) (2026-11-27) | Does a confidence threshold chosen per camera rig beat one global threshold on held-out open-imagery cities? | **FAIL**: -0.66 utility per 100 GT ramps [-2.21, +1.48] vs a procedure-chosen global threshold, and 94 of 200 rig shuffles do as well. The best threshold belongs to the city, not the rig; 0.30 holds |
| [dual_corner_split.md](dual_corner_split.md) (2026-11-27) | Can a GT-free rule split fused sites holding two ramps (dual corners) and rescue one-label-per-site's coverage? | **FAIL** in Gainesville (confirmatory): coverage -1.38 pt vs all labels, and a random split of the same sites recovers nearly as much (+0.20 pt [-0.17, +0.56]). Geometry cannot tell a dual corner from one ramp seen from several sides |
| [crosswalk_fp_filters.md](crosswalk_fp_filters.md) (2026-11-27) | Do GT-free geometric filters remove the Juneau Crosswalk pilot's look-alikes (car roof, brick sidewalk)? | **PASS, narrowly**, for one arm: dropping labels 30-40 deg below the horizon and within 45 deg of abeam raises weighted precision 0.722 to 0.754 (paired +0.033 [+0.004, +0.060]) for 2.1% of true crosswalks. It rests on 7 sampled labels and removes no car-roof false positives |

Scripts are in `scripts/<study>.py`, tests in `tests/test_<study>.py`, and outputs in
`data/<study>/`. Each doc ends with a replication block.
