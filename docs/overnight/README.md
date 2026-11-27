# Overnight CPU studies (2026-11-26/27)

Four pre-registered studies, run on committed data and archived `results.jsonl` files with
no GPU. Each doc's pre-registration was written before any score on real data. Results and
dated amendments are appended below it.

| study | question | verdict |
|---|---|---|
| [rig_persistence.md](rig_persistence.md) | Can "same pixel fires frame after frame along one sequence" find camera-rig artifacts without the 49 deg mask? | **FAIL** as a filter: it flags 2-10% of verdict-true ramps against a 0.2% bar. Keep the mask. As a screen it found two Lyon rig spots on one vehicle (both confirmed by human review, one after an open-street control) and one Bayonne spot that is not a rig (human review agrees) |
| [bearing_refit.md](bearing_refit.md) | Does bearing-only triangulation place multi-view sites better than the flat raycast on Mapillary and Panoramax? | **FAIL** (1 of 3 cities). It helps where camera positions agree (Richmond, GSV), and hurts on Panoramax across sequences (Bayonne) |
| [recall_by_tilt.md](recall_by_tilt.md) | Does RampNet's recall on open-imagery rigs fall with pano tilt? | **NOT ESTABLISHED**: the pooled effect is a city confound and disappears under city fixed effects |
| [best_view_emission.md](best_view_emission.md) | Can one label per fused site, from its best view, replace every per-pano label without losing inventory coverage? | **FAIL** on fresh inventory pulls: p90 placement improves by 0.9-1.7 m and labels drop about 70%, but coverage falls 2.0 (Bend) and 3.7 (Gainesville) points, because dual corners fuse into one site |

Scripts are in `scripts/<study>.py`, tests in `tests/test_<study>.py`, and outputs in
`data/<study>/`. Each doc ends with a replication block.
