"""Score label clustering against corner-level cluster-review GT (RampNet#224).

Reads a RampNet cluster_review bundle (snapshot.json, corners.jsonl and the reviewer's
assignments.json) AS DATA -- no RampNet import, as eval_sites.py reads verdicts.json --
rebuilds every #56 arm on the snapshot's labels exactly as
inventory_clustering.score_server_arms does (`deployed`, `ps @ 7.5 / 10 / 12.5 / 15 m`,
`fusion_server`, `fusion_server+attach`; inventory_clustering.server_arms), restricts to
the reviewed (complete) units and applies the pre-registered metrics and decision rule
(RampNet docs/cluster_review_protocol.md): assignment_metrics per arm, pooled and by
stratum, paired per-ramp tables, `fusion_server+attach` vs `ps @ 7.5 m`, and -- where the
city has an inventory -- the calibration against it.

Nothing is placed: an assignment scores membership. The run dir is read in place.

Until a reviewer has exported an assignments file this writes a "no GT yet" report and
exits 0; it still rebuilds the arms and prints the self-consistency line (an arm scored
against an assignment derived from itself, in memory, must give split 0 / merge 0), so
the machinery is proven on the real bundle before any GT exists.

    python scripts/cluster_review_score.py vancouver --run-dir ../sal-vancouver/runs/vancouver \
        --bundle ../RampNet/benchmark/vancouver/cluster_review [--assignments assignments.json]
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geo  # noqa: E402
import inventory_clustering as ic  # noqa: E402

SCHEMA = 'rampnet.cluster_review/1'
RUBRIC_VERSION = 1
ARMS = ('deployed', 'ps @ 7.5 m', 'ps @ 10 m', 'ps @ 12.5 m', 'ps @ 15 m', 'fusion_server',
        'fusion_server+attach')
THRESHOLDS_M = (7.5, 10.0, 12.5, 15.0)
STRATA = ('signalised', 'arterial', 'residential', 'mid_block')
MATCH_M = 5.0
SANITY_ARM = 'fusion_server+attach'   # a true partition (deployed can hold a label twice)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(bundle):
    snapshot = json.loads((bundle / 'snapshot.json').read_text(encoding='utf-8'))
    with open(bundle / 'corners.jsonl', encoding='utf-8') as f:
        corners = [json.loads(line) for line in f if line.strip()]
    return snapshot, corners


def check_assignments(a, snapshot, corners):
    """Reasons this file cannot be scored (the subset of RampNet's validate() a scorer
    needs: schema, snapshot, every label of a complete unit assigned, known units/keys)."""
    problems = []
    if a.get('schema') != SCHEMA:
        problems.append(f"schema {a.get('schema')!r}")
    if a.get('snapshot_sha256') != snapshot['labels']['sha256']:
        problems.append('snapshot_sha256 is not the bundle\'s label snapshot')
    by_id = {c['corner_id']: c for c in corners}
    for cid, u in (a.get('corners') or {}).items():
        c = by_id.get(cid)
        if c is None:
            problems.append(f'{cid}: not a unit of this bundle')
            continue
        keys = {lab['key'] for lab in c['labels']}
        labels = u.get('labels') or {}
        if set(labels) - keys:
            problems.append(f'{cid}: labels not in the unit: {sorted(set(labels) - keys)[:3]}')
        if u.get('complete') and keys - set(labels):
            problems.append(f'{cid}: complete but {len(keys - set(labels))} label(s) unassigned')
    return problems


def derived_assignment(arm_of, corners):
    """An assignment built FROM an arm, in memory only (never written): each label the arm
    holds goes to ramp r<cluster>, every other label is unsure. Scoring the same arm on it
    must give split 0 and merge 0 -- the self-consistency check."""
    out = {}
    for c in corners:
        labels, ramps = {}, {}
        for lab in c['labels']:
            cl = arm_of.get(lab['key'])
            if cl and len(cl) == 1:
                r = f'r{next(iter(cl)) + 1}'
                labels[lab['key']] = r
                ramps[r] = {'lat': lab['lat'], 'lng': lab['lng']}
            else:
                labels[lab['key']] = ic.UNSURE
        out[c['corner_id']] = {'labels': labels, 'ramps': ramps, 'uncovered': [],
                               'complete': True}
    return out


def match_one_to_one(a_pts, b_pts, radius_m=MATCH_M):
    """Greedy one-to-one matching by distance within radius_m: number of matched pairs."""
    cand = sorted((geo.haversine_m(p[0], p[1], q[0], q[1]), i, j)
                  for i, p in enumerate(a_pts) for j, q in enumerate(b_pts))
    used_a, used_b, n = set(), set(), 0
    for d, i, j in cand:
        if d > radius_m:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        n += 1
    return n


def inventory_calibration(units, corners_by_id, arms, label_pos):
    """Reviewer ramps vs inventory points (one-to-one within 5 m, both directions) and each
    arm's inventory_metrics on the same units (clusters placed at the mean of their
    in-window labels' server positions; the pool = every inventory point in a window)."""
    import numpy as np
    rev_n = inv_n = matched = 0
    inv_all, windows = [], []
    for cid, u in sorted(units.items()):
        c = corners_by_id[cid]
        if 'inventory' not in c:
            continue
        windows.append(cid)
        rev = [(p['lat'], p['lng']) for p in (u.get('ramps') or {}).values()]
        rev += [(p['lat'], p['lng']) for p in u.get('uncovered') or [] if not p.get('unsure')]
        inv = [(p['lat'], p['lng']) for p in c['inventory']]
        rev_n += len(rev)
        inv_n += len(inv)
        matched += match_one_to_one(rev, inv)
        inv_all += inv
    if not windows:
        return None
    fr = geo.LocalFrame(corners_by_id[windows[0]]['centre']['lat'],
                        corners_by_id[windows[0]]['centre']['lng'])
    inv_xy = np.array([fr.to_enu(*p) for p in inv_all]).reshape(-1, 2)
    pool = np.ones(len(inv_xy), dtype=bool)
    keys = {lab['key'] for cid in windows for lab in corners_by_id[cid]['labels']}
    by_arm = {}
    for name, arm_of in arms.items():
        members = {}
        for k in keys:
            for cl in arm_of.get(k, ()):
                members.setdefault(cl, []).append(fr.to_enu(*label_pos[k]))
        placed = [((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)), pts)
                  for pts in members.values()]
        by_arm[name] = ic.inventory_metrics(placed, inv_xy, pool, MATCH_M) if len(inv_xy) else None
    return {'units': len(windows), 'reviewer_points': rev_n, 'inventory_points': inv_n,
            'matched': matched,
            'reviewer_matched_share': matched / rev_n if rev_n else None,
            'inventory_matched_share': matched / inv_n if inv_n else None,
            'inventory_metrics': by_arm}


def fmt(v, nd=3):
    return 'n/a' if v is None else f'{v:.{nd}f}'


def ci(t):
    return '' if not t else f' [{t[0]:.3f}, {t[1]:.3f}]'


ARM_FIELDS = ['arm', 'stratum', 'units', 'ramps', 'covered', 'split', 'split_rate', 'merge_k',
              'merge_strict_k', 'merge_n', 'merge_rate', 'merge_strict_rate', 'clusters',
              'invalid_clusters', 'invalid_cluster_rate', 'labels_held', 'not_ramp_held',
              'not_ramp_label_rate', 'uncovered_sure', 'coverage']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city')
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--assignments', default='assignments.json',
                    help='file name inside the bundle (default assignments.json = rater A)')
    ap.add_argument('--frame', default='auto', help='fuse frame of the fusion arms')
    ap.add_argument('--out', type=Path, default=None,
                    help='default runs/<city>/cluster_review/score in this checkout')
    args = ap.parse_args(argv)
    bundle, run_dir = args.bundle.resolve(), args.run_dir.resolve()
    out = (args.out or REPO_ROOT / 'runs' / args.city / 'cluster_review' / 'score').resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot, corners = load_bundle(bundle)
    corners_by_id = {c['corner_id']: c for c in corners}
    cfg = ic.SERVER_LABELS[args.city]
    run_labels = run_dir / cfg['labels']
    if sha256_file(run_labels) != snapshot['labels']['sha256']:
        raise SystemExit(f'{run_labels}: not the label snapshot the bundle was exported from')
    a_path = bundle / args.assignments
    lines = [f'# {args.city}: clustering scored against cluster-review GT (RampNet#224)', '',
             'Protocol and decision rule: RampNet docs/cluster_review_protocol.md (pre-registered); '
             'rubric: RampNet benchmark/RUBRICS.md §6.', '',
             f"- bundle: `{bundle.name}` of {snapshot['city']}: {len(corners)} units "
             f"({sum(1 for c in corners if c.get('pilot'))} pilot), "
             f"{sum(c['n_labels'] for c in corners)} labels; label snapshot sha256 "
             f"`{snapshot['labels']['sha256']}` ({snapshot['labels']['n_features']} features, "
             f"fetched {snapshot['labels']['fetched_at']})",
             f"- deployed clusters sha256 `{snapshot['seed_arms']['deployed']['sha256']}`; "
             f"results.jsonl sha256 `{sha256_file(run_dir / 'results.jsonl')}`; fusion frame "
             f'`{args.frame}`']

    labels, clusters, st = ic.server_arms(run_dir, cfg, frame=args.frame,
                                          thresholds_m=THRESHOLDS_M)
    arms = {name: ic.arm_index(clusters[name]) for name in ARMS}
    label_pos = {str(int(r.label_id)): (r.lat, r.lng) for r in labels.itertuples(index=False)}
    unit_keys = {c['corner_id']: [lab['key'] for lab in c['labels']] for c in corners}
    lines.append('- arms rebuilt: ' + ', '.join(f'{n} {len(clusters[n])}' for n in ARMS)
                 + ' clusters')

    derived = derived_assignment(arms[SANITY_ARM], corners)
    sm = ic.assignment_metrics(arms[SANITY_ARM], derived, unit_keys)
    sanity = (f'self-consistency: `{SANITY_ARM}` scored against an assignment derived from '
              f"itself (in memory, never written) over all {sm['units']} units: split "
              f"{sm['split']}/{sm['covered']}, merge {sm['merge_k']}/{sm['merge_n']} -- "
              + ('OK' if sm['split'] == 0 and sm['merge_k'] == 0 else 'FAILED'))
    print(sanity)
    if sm['split'] or sm['merge_k']:
        raise SystemExit('self-consistency check failed')

    if not a_path.exists():
        lines += ['', '## NO GT YET', '',
                  f'`{a_path.name}` does not exist in the bundle: no reviewer has exported an '
                  'assignment, so there is nothing to score and no number below is a result. '
                  'Review the pilot in RampNet `scripts/cluster_review_gallery.py`, save the '
                  'export into the bundle, and re-run this command.', '', f'- {sanity}']
        (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        for name, fields in (('arms.csv', ARM_FIELDS), ('ramps.csv', ['corner_id', 'type', 'ramp']
                                                        + list(ARMS))):
            with open(out / name, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(fields)
        print('\n'.join(lines))
        print(f'wrote {out / "report.md"} (no GT yet)')
        return 0

    a = json.loads(a_path.read_text(encoding='utf-8'))
    problems = check_assignments(a, snapshot, corners)
    if problems:
        raise SystemExit(f'{a_path}: cannot be scored: ' + '; '.join(problems[:10]))
    units = {cid: u for cid, u in (a.get('corners') or {}).items() if u.get('complete')}
    notes = a.get('review_notes') or {}
    lines += [f"- assignments: `{a_path.name}` sha256 `{sha256_file(a_path)}`, rater "
              f"{a.get('rater') or notes.get('reviewer') or '?'}, rubric v{a.get('rubric_version')}, "
              f"seed {a.get('seed_arm')}, exported {a.get('exported_at')}; "
              f'{len(units)} complete of {len(a.get("corners") or {})} units in the file',
              f'- {sanity}']
    if a.get('rubric_version') != RUBRIC_VERSION:
        lines.append(f"- WARNING: rubric v{a.get('rubric_version')} -- not comparable with v{RUBRIC_VERSION} files")
    if notes.get('caveats'):
        lines += ['- reviewer caveats:'] + [f'  - {c}' for c in notes['caveats']]

    rows, by_arm = [], {}
    for name in ARMS:
        for stratum in ('all',) + STRATA:
            sub = {cid: u for cid, u in units.items()
                   if stratum == 'all' or corners_by_id[cid]['type'] == stratum}
            m = ic.assignment_metrics(arms[name], sub, unit_keys)
            if stratum == 'all':
                by_arm[name] = m
            rows.append({'arm': name, 'stratum': stratum,
                         **{k: m[k] for k in ARM_FIELDS if k in m}})
    v, reasons = ic.assignment_verdict(by_arm)
    lines += ['', f'## Decision (pre-registered): **{v}**', '']
    lines += [f'- {r}' for r in reasons]
    lines += ['', f'Rule: `{ic.REVIEW_ARM}` vs `{ic.REVIEW_BASE}` -- split lower by >= '
              f'{ic.RULE_SPLIT_DROP}, merge not higher by > {ic.RULE_MERGE_RISE}, coverage not '
              f'lower by > {ic.RULE_COVERED_DROP}. Read on rater A\'s full pass; a pilot-only '
              'file is descriptive.', '',
              '## Arms (complete units, pooled)', '',
              '| arm | ramps | covered | split (95% CI) | merge k/n (strict) | invalid clusters | '
              'not_ramp labels | coverage |', '|---|---:|---:|---|---|---|---|---|']
    for name in ARMS:
        m = by_arm[name]
        lines.append(f"| {name} | {m['ramps']} | {m['covered']} | {fmt(m['split_rate'])} "
                     f"({m['split']}/{m['covered']}){ci(m['split_ci'])} | {fmt(m['merge_rate'])} "
                     f"({m['merge_k']}/{m['merge_n']}; strict {m['merge_strict_k']}) | "
                     f"{fmt(m['invalid_cluster_rate'])} ({m['invalid_clusters']}/{m['clusters']}) | "
                     f"{fmt(m['not_ramp_label_rate'])} ({m['not_ramp_held']}/{m['labels_held']}) | "
                     f"{fmt(m['coverage'])}{ci(m['coverage_ci'])} |")
    lines += ['', '## By stratum (split rate / merge rate / coverage)', '',
              '| arm | ' + ' | '.join(STRATA) + ' |', '|---|' + '---|' * len(STRATA)]
    for name in ARMS:
        cells = []
        for s in STRATA:
            r = next(x for x in rows if x['arm'] == name and x['stratum'] == s)
            cells.append(f"{fmt(r['split_rate'])} / {fmt(r['merge_rate'])} / {fmt(r['coverage'])} "
                         f"({r['units']} u)")
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired per ramp (A = baseline)', '',
              '| A | B | fixed (A splits, B not) | broken | both split | neither | only A covers | '
              'only B covers |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for base, arm in (('ps @ 7.5 m', 'fusion_server+attach'), ('ps @ 7.5 m', 'deployed'),
                      ('ps @ 7.5 m', 'fusion_server')):
        t = ic.paired_ramp_table(by_arm[base]['per_ramp'], by_arm[arm]['per_ramp'])
        lines.append(f"| {base} | {arm} | {t['fixed']} | {t['broken']} | {t['both']} | "
                     f"{t['neither']} | {t['only_a']} | {t['only_b']} |")
    cal = inventory_calibration(units, corners_by_id, arms, label_pos)
    lines += ['', '## Calibration against the city inventory', '']
    if cal is None:
        lines.append('- no reviewed unit carries inventory points (the city has none)')
    else:
        lines += [f"- {cal['units']} units: {cal['reviewer_points']} reviewer ramps (assigned + sure "
                  f"uncovered) vs {cal['inventory_points']} inventory points in the windows; "
                  f"{cal['matched']} matched one-to-one within {MATCH_M:g} m "
                  f"(reviewer share {fmt(cal['reviewer_matched_share'])}, inventory share "
                  f"{fmt(cal['inventory_matched_share'])})", '',
                  '| arm | inventory split r5 | assignment split | inventory merge r5 | '
                  'assignment merge |', '|---|---|---|---|---|']
        for name in ARMS:
            im = cal['inventory_metrics'].get(name) or {}
            lines.append(f"| {name} | {fmt(im.get('split_rate'))} | {fmt(by_arm[name]['split_rate'])} | "
                         f"{fmt(im.get('merge_rate'))} | {fmt(by_arm[name]['merge_rate'])} |")
    lines += ['', '## Against verdicts.json', '',
              '- not run here: RampNet `rampnet.cluster_review.verdict_consistency` does it for '
              'cities with a benchmark bundle (Vancouver has none).']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with open(out / 'arms.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ARM_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ARM_FIELDS})
    with open(out / 'ramps.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['corner_id', 'type', 'ramp'] + list(ARMS))
        for key in sorted(by_arm[ARMS[0]]['per_ramp']):
            w.writerow([key[0], corners_by_id[key[0]]['type'], key[1]]
                       + [by_arm[n]['per_ramp'].get(key, 0) for n in ARMS])
    print('\n'.join(lines))
    print(f'wrote {out}/report.md, arms.csv, ramps.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
