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
    python scripts/score_validation.py runs/bend verdicts.json
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
    Returns (judged, missed_total, n_reviewed, n_seen):
      judged = list of (confidence, is_correct) for judged detections on included panos,
      missed_total = missed-ramp marks on *fully reviewed* panos,
      restricted to fully reviewed panos so recall's denominator is trustworthy.
    """
    judged, missed_total, n_reviewed, n_seen = [], 0, 0, 0
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
        n_reviewed += 1
        judged += list(zip(confs, entry['dets']))
        missed_total += len(entry['missed'])
    return judged, missed_total, n_reviewed, n_seen


def report(title, judged, missed_total, n_reviewed, n_seen):
    tp = sum(1 for _, ok in judged if ok)
    fp = len(judged) - tp
    total_ramps = tp + missed_total  # ramps found by model or reviewer on reviewed panos

    print(f"--- {title} ---")
    print(f"Panos fully reviewed: {n_reviewed} (of {n_seen} seen)")
    print(f"Detections judged:    {len(judged)}  (correct {tp}, incorrect {fp})")
    print(f"Missed ramps marked:  {missed_total}")
    if not judged:
        print("Nothing judged yet.\n")
        return
    p = tp / len(judged)
    lo, hi = wilson_interval(tp, len(judged))
    print(f"Precision: {p:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
    if total_ramps:
        r = tp / total_ramps
        rlo, rhi = wilson_interval(tp, total_ramps)
        print(f"Recall:    {r:.3f}  (95% CI {rlo:.3f}–{rhi:.3f})  "
              f"[vs ramps visible in reviewed panos]")
    print()
    print(f"{'threshold':>9}  {'kept':>5}  {'precision':>9}  {'recall':>7}")
    for t in THRESHOLDS:
        kept = [(c, ok) for c, ok in judged if c >= t]
        if not kept:
            break
        ktp = sum(1 for _, ok in kept if ok)
        prec = ktp / len(kept)
        rec = ktp / total_ramps if total_ramps else float('nan')
        print(f"{t:>9.2f}  {len(kept):>5}  {prec:>9.3f}  {rec:>7.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Score gallery verdicts: precision, recall, and a threshold sweep."
    )
    parser.add_argument("run", help="Run directory from main.py (or a results JSONL file).")
    parser.add_argument("verdicts", help="verdicts.json exported from the gallery.")
    args = parser.parse_args()

    confs_by_pid, panos = load_inputs(args.run, args.verdicts)

    judged, missed, n_rev, n_seen = collect(panos, confs_by_pid)
    report("All reviewed panos", judged, missed, n_rev, n_seen)

    judged_u, missed_u, n_rev_u, n_seen_u = collect(panos, confs_by_pid, exclude_top=True)
    if n_seen_u != n_seen:
        report("Unbiased subset (random + empty samples only)",
               judged_u, missed_u, n_rev_u, n_seen_u)

    print("Recall here = per-pano-comprehensive, as judged by the reviewer on sampled panos.")
    print("For city-scale metrics, use PS validation agree-rate (precision) and attribute-level")
    print("comparison against human explore labels (recall) — see docs/design-review-2026-07.md §6–7.")


if __name__ == "__main__":
    main()
