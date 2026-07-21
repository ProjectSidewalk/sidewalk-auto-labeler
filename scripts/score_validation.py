"""
Score a verdicts.json exported from the spot-check gallery against a run's results.

Reports precision (judged detections), recall (fully reviewed panos: correct detections
vs. human-marked missed ramps), and a confidence-threshold sweep — both overall and on
the unbiased subset (excluding the always-included densest "top" panos).

Caveats printed with the numbers:
  - Recall is relative to ramps *visible in the sampled panos as judged by the reviewer*,
    i.e. per-pano-comprehensive ground truth (which Project Sidewalk validation can't give).
  - A ramp double-counted by two detections inflates TP; mark duplicates INCORRECT if you
    want precision to penalize them.

Usage:
    python scripts/score_validation.py runs/bend
    python scripts/score_validation.py runs/bend path/to/bend_verdicts.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_inputs(run_arg, verdicts_arg):
    run_path = Path(run_arg)
    jsonl_path = run_path / "results.jsonl" if run_path.is_dir() else run_path
    if not jsonl_path.exists():
        sys.exit(f"No results file found at {jsonl_path}")

    if verdicts_arg is None:
        # The gallery exports <run name>_verdicts.json; look for it in the run directory.
        run_dir = jsonl_path.parent
        candidates = [run_dir / f"{run_dir.name}_verdicts.json", run_dir / "verdicts.json"]
        verdicts_arg = next((c for c in candidates if c.exists()), None)
        if verdicts_arg is None:
            sys.exit("No verdicts file found — looked for:\n  " +
                     "\n  ".join(str(c) for c in candidates) +
                     "\nExport it from the gallery and save it into the run directory, "
                     "or pass its path explicitly.")
        print(f"Scoring {verdicts_arg}\n")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        confs_by_pid = {
            r['pano']['panorama_id']: [d['confidence'] for d in r['detections']]
            for r in (json.loads(line) for line in f if line.strip())
        }

    with open(verdicts_arg, 'r', encoding='utf-8') as f:
        verdicts = json.load(f)

    manifest_path = jsonl_path.parent / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            area_hash = json.load(f).get('area_hash')
        if area_hash and verdicts.get('run_key') not in (area_hash, None):
            print(f"⚠ verdicts.json run_key does not match this run's area hash — "
                  f"are these verdicts from a different run?\n")
    return confs_by_pid, verdicts['panos']


def collect(panos, confs_by_pid, exclude_top=False):
    """
    Returns (judged, recall_judged, missed_total, n_seen, n_judged, n_unconfirmed,
             n_unsure, missed_unsure):
      judged        = (confidence, is_correct) for every *decided* detection on panos
                      whose detections are all judged — the PRECISION pool. A detection
                      verdict is valid whether or not the missed-ramp scan happened.
      recall_judged = the subset of judged from panos whose missed-ramp check is
                      confirmed — the RECALL pool. missed_total counts *confident*
                      missed-ramp marks on the same panos, so recall's numerator and
                      denominator cover the same set.
      n_unconfirmed = fully judged panos excluded from the recall pool because the
                      missed-ramp check was never confirmed.
      n_unsure      = detections marked 'unsure' (abstained: in neither pool).
      missed_unsure = missed marks flagged unsure (abstained: not in missed_total).

    'unsure' abstention: a crop verdict of 'unsure' and a missed mark with
    {'unsure': True} are the reviewer saying "can't tell from this imagery". They
    are dropped from both precision and recall (forcing a guess would bias either
    metric) and reported as separate counts. A pano with unsure marks still counts
    as fully judged — 'unsure' is a decision, unlike None (not yet looked at).

    The missed-ramp check is per entry: new-schema entries (exported by a gallery
    with the confirmation feature) carry 'no_missed' and must have it set or have a
    missed mark; legacy entries (no key) are trusted as before, with a file-level
    warning printed by main(). Mixed old/new files therefore score correctly.

    This gate mirrors reviewed()/fnChecked() in the viewer JS
    (scripts/spot_check_gallery.py) — keep the two in sync.
    """
    judged, recall_judged, missed_total, missed_unsure = [], [], 0, 0
    n_seen = n_judged = n_unconfirmed = n_unsure = 0
    for pid, entry in panos.items():
        if exclude_top and entry.get('group') == 'top':
            continue
        confs = confs_by_pid.get(pid)
        if confs is None or len(confs) != len(entry['dets']):
            print(f"⚠ skipping {pid}: verdicts don't match results.jsonl detections")
            continue
        n_seen += 1
        if any(d is None for d in entry['dets']):
            continue  # partially judged: unusable for either metric
        n_judged += 1
        n_unsure += sum(1 for d in entry['dets'] if d == 'unsure')
        # Decided verdicts only (True/False); 'unsure' abstains from both metrics.
        pano_judged = [(c, d) for c, d in zip(confs, entry['dets']) if d != 'unsure']
        judged += pano_judged
        fn_checked = ((entry['no_missed'] or entry['missed'])
                      if 'no_missed' in entry else True)
        if fn_checked:
            recall_judged += pano_judged
            missed_total += sum(1 for m in entry['missed'] if not m.get('unsure'))
            missed_unsure += sum(1 for m in entry['missed'] if m.get('unsure'))
        else:
            n_unconfirmed += 1
    return (judged, recall_judged, missed_total, n_seen, n_judged, n_unconfirmed,
            n_unsure, missed_unsure)


def report(title, judged, recall_judged, missed_total, n_seen, n_judged, n_unconfirmed,
           n_unsure, missed_unsure):
    tp = sum(1 for _, ok in judged if ok)
    fp = len(judged) - tp
    rtp = sum(1 for _, ok in recall_judged if ok)
    total_ramps = rtp + missed_total  # ramps found by model or reviewer, recall pool

    print(f"--- {title} ---")
    print(f"Panos fully judged:   {n_judged} (of {n_seen} seen)")
    if n_unconfirmed:
        print(f"⚠ {n_unconfirmed} of those excluded from RECALL only: their missed-ramp "
              f"check was never confirmed\n  (open the gallery and press 'm' / mark the "
              f"missed ramps, then re-export).")
    print(f"Detections judged:    {len(judged)}  (correct {tp}, incorrect {fp})")
    if n_unsure:
        print(f"Detections unsure:    {n_unsure}  (abstained — not in precision or recall)")
    print(f"Missed ramps marked:  {missed_total}"
          + (f"  (+{missed_unsure} unsure, abstained)" if missed_unsure else ""))
    if not judged:
        print("Nothing judged yet.\n")
        return
    p = tp / len(judged)
    lo, hi = wilson_interval(tp, len(judged))
    print(f"Precision: {p:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
    if total_ramps:
        r = rtp / total_ramps
        rlo, rhi = wilson_interval(rtp, total_ramps)
        print(f"Recall:    {r:.3f}  (95% CI {rlo:.3f}–{rhi:.3f})  "
              f"[vs ramps visible in the {n_judged - n_unconfirmed} recall-pool panos]")
    print()
    print(f"{'threshold':>9}  {'kept':>5}  {'precision':>9}  {'recall':>7}")
    for t in THRESHOLDS:
        kept = [(c, ok) for c, ok in judged if c >= t]
        if not kept:
            break
        ktp = sum(1 for _, ok in kept if ok)
        prec = ktp / len(kept)
        rtp_t = sum(1 for c, ok in recall_judged if c >= t and ok)
        rec = rtp_t / total_ramps if total_ramps else float('nan')
        print(f"{t:>9.2f}  {len(kept):>5}  {prec:>9.3f}  {rec:>7.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Score gallery verdicts: precision, recall, and a threshold sweep."
    )
    parser.add_argument("run", help="Run directory from main.py (or a results JSONL file).")
    parser.add_argument("verdicts", nargs="?",
                        help="Verdicts file exported from the gallery (default: "
                             "<run>/<name>_verdicts.json, then <run>/verdicts.json).")
    args = parser.parse_args()

    # Don't crash on ⚠/– when the console encoding can't represent them (Windows
    # cp1252). Done here, not at import time, so importing this module (e.g. from
    # tests) never mutates the process's streams.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')

    confs_by_pid, panos = load_inputs(args.run, args.verdicts)

    # Galleries with the missed-ramp confirmation export a per-pano 'no_missed' flag;
    # entries without it are legacy and trusted (see collect()).
    if not any('no_missed' in e for e in panos.values()):
        print("⚠ verdicts predate the missed-ramp confirmation: recall assumes every "
              "reviewed pano\n  was actually scanned for missed ramps and may be "
              "optimistic.\n")

    all_pools = collect(panos, confs_by_pid)
    report("All reviewed panos", *all_pools)

    unbiased_pools = collect(panos, confs_by_pid, exclude_top=True)
    if unbiased_pools[3] != all_pools[3]:  # n_seen differs -> top panos existed
        report("Unbiased subset (random + empty samples only)", *unbiased_pools)

    print("Recall here = per-pano-comprehensive, as judged by the reviewer on sampled panos.")
    print("For city-scale metrics, use PS validation agree-rate (precision) and attribute-level")
    print("comparison against human explore labels (recall) — see docs/design-review-2026-07.md §6–7.")


if __name__ == "__main__":
    main()
