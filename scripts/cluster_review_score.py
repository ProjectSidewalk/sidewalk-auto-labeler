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

The decision is read as pre-registered only on rater A's full pass (role a, or no role, with
every unit of the bundle complete or can't judge). No arm-level number is computed on anything
else until rater A's full pass is exported (RampNet protocol Amendment 5, point S1): a pilot or
partial pass is REFUSED, and so is a role-b file while the bundle's assignments.json is not
that full pass, because with one rater who knows the hypothesis an arm table read mid-study is
the leak blinding exists to prevent (the pilot is read with RampNet's
scripts/cluster_review_agreement.py: agreement, timing, can't-judge counts).
--allow-partial-descriptive overrides; the tables are then headed DESCRIPTIVE (decision_scope)
and the report says the override was used. A role-b file after A's full pass is scored as
DESCRIPTIVE without the flag.

The report names the labeler commit it ran at (labeler_ref) beside RAMPNET_REF, and puts
a unit-level paired bootstrap 95% CI of the split-rate difference beside the verdict
(Amendment 5, S8; descriptive -- the rule reads the point estimates).

The RampNet side (the protocol, RUBRICS.md section 6, the gallery, rampnet/cluster_review.py
and the bundle) lives, for now, only on RampNet branch `cluster-review-224`; RAMPNET_REF pins
the commit every link here points at. TODO: repoint to RampNet main once that branch merges.

Until a reviewer has exported an assignments file this writes a "no GT yet" report and
exits 0; it still rebuilds the arms and prints the self-consistency line (an arm scored
against an assignment derived from itself, in memory, must give split 0 / merge 0), so
the machinery is proven on the real bundle before any GT exists.

    # --run-dir: a run dir holding the Vancouver run, e.g. runs/vancouver
    python scripts/cluster_review_score.py vancouver --run-dir runs/vancouver \
        --bundle ../RampNet/benchmark/vancouver/cluster_review [--assignments assignments.json]
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
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
VALUE_RE = re.compile(r'^(r\d+|not_ramp|unsure)$')
# The RampNet commit the protocol / rubric / gallery links point at: RampNet PR #280 (branch
# cluster-review-224) with Amendments 4 + 5. TODO: repoint to RampNet main once it merges.
RAMPNET_REF = 'e869be294f29321ae882521bed415a381eba40c0'


def labeler_ref():
    """(sha, dirty) of this checkout: the commit the scorer ran at, and whether any tracked
    .py file differed from it (then the report says so -- the commit alone would not
    reproduce the numbers). ('unknown', False) outside a git checkout."""
    def git(*a):
        return subprocess.run(['git', '-C', str(REPO_ROOT), *a], capture_output=True,
                              text=True, timeout=30)
    try:
        r = git('rev-parse', 'HEAD')
        if r.returncode != 0:
            return 'unknown', False
        dirty = git('status', '--porcelain', '--untracked-files=no', '--', '*.py').stdout.strip()
        return r.stdout.strip(), bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return 'unknown', False


def labeler_line():
    sha, dirty = labeler_ref()
    return (f'- scorer: sidewalk-auto-labeler `scripts/cluster_review_score.py` at commit `{sha}`'
            + (' **plus uncommitted changes to tracked .py files**' if dirty else '')
            + f'; RampNet links at `{RAMPNET_REF}`')


def rampnet_url(path):
    """Permalink to `path` in RampNet at RAMPNET_REF."""
    return f'https://github.com/ProjectSidewalk/RampNet/blob/{RAMPNET_REF}/{path}'


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
    if a.get('rubric_version') != RUBRIC_VERSION:
        problems.append(f"rubric_version {a.get('rubric_version')!r}: this scorer implements "
                        f'v{RUBRIC_VERSION} only, and passes under different rubrics are not '
                        'comparable')
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
        bad = sorted(k for k, v in labels.items()
                     if not (isinstance(v, str) and VALUE_RE.match(v)))
        if bad:
            problems.append(f'{cid}: label value(s) not r<N> / not_ramp / unsure: '
                            + ', '.join(f'{k}={labels[k]!r}' for k in bad[:3]))
        if u.get('complete') and keys - set(labels):
            problems.append(f'{cid}: complete but {len(keys - set(labels))} label(s) unassigned')
        cj = u.get('cant_judge', False)
        if not isinstance(cj, bool):
            problems.append(f'{cid}: cant_judge {cj!r} is not a boolean')
        elif cj and u.get('complete'):
            problems.append(f'{cid}: cant_judge and complete together')
        elif cj and not str(u.get('cant_judge_reason') or '').strip():
            problems.append(f'{cid}: cant_judge without a cant_judge_reason')
    return problems


def decision_scope(a, corners):
    """(pre_registered, why): whether the decision rule may be read as the pre-registered
    one. The protocol reads it "on the same complete units of rater A's full pass ...
    Pilot-only scores are descriptive", so it is pre-registered only for role a (or a file
    with no role) in which EVERY unit of the bundle is complete or can't judge; anything
    else (a pilot, a partial pass, a role-b retest) is descriptive, and `why` says which."""
    role = a.get('role')
    if role not in (None, 'a'):
        return False, f"role {role}: the decision is read on rater A's pass"
    units = a.get('corners') or {}
    done = {cid for cid, u in units.items() if u.get('complete') or u.get('cant_judge') is True}
    all_ids = {c['corner_id'] for c in corners}
    if all_ids <= done:
        return True, ''
    pilot = {c['corner_id'] for c in corners if c.get('pilot')}
    kind = 'pilot' if done and done <= pilot else 'partial pass'
    return False, (f"{kind}: {len(all_ids & done)} of {len(all_ids)} units complete or "
                   "can't judge")


def is_full_pass_a(a, corners):
    """True when `a` is rater A's full pass (decision_scope's pre-registered case)."""
    return decision_scope(a, corners)[0]


def arm_numbers_refusal(a, corners, bundle):
    """Why no arm-level number may be computed on this file (None = allowed). Protocol
    Amendment 5 (S1): nothing arm-level is computed or read until rater A's full pass is
    exported. Rater A's full pass itself is allowed; a role-b file is allowed once the
    bundle's assignments.json is that full pass; anything else is refused."""
    pre_registered, why = decision_scope(a, corners)
    if pre_registered:
        return None
    if a.get('role') not in (None, 'a'):
        a_path = bundle / 'assignments.json'
        if a_path.exists():
            try:
                rater_a = json.loads(a_path.read_text(encoding='utf-8'))
            except ValueError:
                rater_a = None
            if (rater_a is not None and rater_a.get('role') in (None, 'a')
                    and rater_a.get('snapshot_sha256') == a.get('snapshot_sha256')
                    and is_full_pass_a(rater_a, corners)):
                return None
        return (f"{why}, and the bundle's assignments.json is not rater A's full pass")
    return why


def cant_judge_section(a, corners_by_id):
    """report.md lines: every can't-judge unit (protocol Amendment 3) with its reason,
    counted overall and per stratum. Such units are excluded from every metric."""
    cj = {cid: u for cid, u in (a.get('corners') or {}).items() if u.get('cant_judge') is True}
    by = {s: 0 for s in STRATA}
    for cid in cj:
        t = (corners_by_id.get(cid) or {}).get('type')
        by[t] = by.get(t, 0) + 1
    lines = ['', "## Can't judge (excluded from every metric)", '',
             f"- {len(cj)} unit(s): " + ', '.join(f'{s} {by.get(s, 0)}' for s in STRATA)]
    if cj:
        lines += ['', '| unit | type | reason |', '|---|---|---|']
        for cid in sorted(cj):
            reason = str(cj[cid].get('cant_judge_reason') or '').strip().replace('|', '/')
            lines.append(f"| {cid} | {(corners_by_id.get(cid) or {}).get('type', '?')} | {reason} |")
    return lines, len(cj)


def arm_input_mismatches(run_dir, cfg, snapshot):
    """How the run dir's arm inputs differ from the ones snapshot.json records."""
    out = []
    want = (snapshot.get('seed_arms') or {})
    dep = sha256_file(run_dir / cfg['clusters'])
    if dep != (want.get('deployed') or {}).get('sha256'):
        out.append(f"deployed clusters sha256 {dep[:12]} != snapshot "
                   f"{str((want.get('deployed') or {}).get('sha256'))[:12]}")
    res = sha256_file(run_dir / 'results.jsonl')
    if res != (want.get('fusion') or {}).get('results_sha256'):
        out.append(f"results.jsonl sha256 {res[:12]} != snapshot "
                   f"{str((want.get('fusion') or {}).get('results_sha256'))[:12]}")
    return out


def derived_assignment(arm_of, corners):
    """An assignment built FROM an arm, in memory only (never written): each label the arm
    holds goes to ramp r<cluster>, every other label is unsure. Scoring the same arm on it
    must give split 0 and merge 0 -- the self-consistency check."""
    out = {}
    for c in corners:
        labels, ramps, name = {}, {}, {}
        for lab in c['labels']:
            cl = arm_of.get(lab['key'])
            if cl and len(cl) == 1:
                r = name.setdefault(next(iter(cl)), f'r{len(name) + 1}')
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


BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 224


def bootstrap_split_diff(per_base, per_arm, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """(diff, [lo, hi], units): split_rate(arm) - split_rate(base) with a unit-level PAIRED
    percentile bootstrap 95% CI (RampNet protocol Amendment 5, S8). Units are resampled
    with replacement and both arms are scored on the same draw, because ramps cluster in
    units (one unit can hold dozens) and a ramp-level Wilson interval is too narrow.
    per_* = assignment_metrics' per_ramp {(corner_id, ramp): n clusters}; covered = >= 1,
    split = >= 2. Descriptive: the decision rule reads the point estimates."""
    import random
    units = sorted({k[0] for k in per_base} | {k[0] for k in per_arm})
    if not units:
        return None, None, 0
    tally = {}
    for u in units:
        t = [0, 0, 0, 0]                       # base split, base covered, arm split, arm covered
        for (cid, _r), v in per_base.items():
            if cid == u:
                t[0] += v >= 2
                t[1] += v >= 1
        for (cid, _r), v in per_arm.items():
            if cid == u:
                t[2] += v >= 2
                t[3] += v >= 1
        tally[u] = t

    def diff(sample):
        bs, bc, as_, ac = (sum(tally[u][i] for u in sample) for i in range(4))
        return None if not bc or not ac else as_ / ac - bs / bc
    point = diff(units)
    rng = random.Random(seed)
    draws = sorted(d for d in (diff([units[rng.randrange(len(units))] for _ in units])
                               for _ in range(n)) if d is not None)
    if point is None or not draws:
        return point, None, len(units)
    return point, [draws[int(0.025 * (len(draws) - 1))],
                   draws[int(round(0.975 * (len(draws) - 1)))]], len(units)


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
    ap.add_argument('--allow-partial-descriptive', action='store_true',
                    help="compute arm-level numbers on a file that is not rater A's full pass "
                         "(a pilot, a partial pass, a role-b file before A's full pass); "
                         'refused without it (protocol Amendment 5, S1). Every table is '
                         'headed DESCRIPTIVE and the report records the override')
    ap.add_argument('--allow-arm-mismatch', action='store_true',
                    help='score although the deployed clusters or results.jsonl are not the '
                         'ones the bundle was exported from (recorded in the report)')
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
    mismatch = arm_input_mismatches(run_dir, cfg, snapshot)
    if mismatch and not args.allow_arm_mismatch:
        raise SystemExit('the arms would be rebuilt from other inputs than the bundle was '
                         'exported from: ' + '; '.join(mismatch)
                         + ' (--allow-arm-mismatch overrides, and says so in the report)')
    a_path = bundle / args.assignments
    lines = [f'# {args.city}: clustering scored against cluster-review GT (RampNet#224)', '',
             'Protocol and decision rule: RampNet '
             f"[docs/cluster_review_protocol.md]({rampnet_url('docs/cluster_review_protocol.md')}) "
             '(pre-registered); rubric: RampNet '
             f"[benchmark/RUBRICS.md]({rampnet_url('benchmark/RUBRICS.md')}) §6. Both live, for "
             'now, only on RampNet branch `cluster-review-224`, PR #280 (links pinned to '
             f'`{RAMPNET_REF[:7]}`; TODO: repoint to RampNet main once that branch merges).', '',
             f"- bundle: `{bundle.name}` of {snapshot['city']}: {len(corners)} units "
             f"({sum(1 for c in corners if c.get('pilot'))} pilot), "
             f"{sum(c['n_labels'] for c in corners)} labels; label snapshot sha256 "
             f"`{snapshot['labels']['sha256']}` ({snapshot['labels']['n_features']} features, "
             f"fetched {snapshot['labels']['fetched_at']})",
             labeler_line(),
             f"- deployed clusters sha256 `{snapshot['seed_arms']['deployed']['sha256']}`; "
             f"results.jsonl sha256 `{sha256_file(run_dir / 'results.jsonl')}`; fusion frame "
             f'`{args.frame}`']
    if mismatch:
        lines.append('- **ARM INPUT MISMATCH, overridden with --allow-arm-mismatch**: '
                     + '; '.join(mismatch))

    labels, clusters, st = ic.server_arms(run_dir, cfg, frame=args.frame,
                                          thresholds_m=THRESHOLDS_M)
    # the pre-registered label set (protocol, "Metrics"): human labels are out of every arm,
    # and a label an arm does not hold is that arm's singleton, so all arms cluster the same
    # labels and coverage cannot differ between them
    unit_keys = {c['corner_id']: [lab['key'] for lab in c['labels']
                                  if lab.get('user_kind') != 'human'] for c in corners}
    n_human = sum(1 for c in corners for lab in c['labels'] if lab.get('user_kind') == 'human')
    scored = [k for ks in unit_keys.values() for k in ks]
    arms, n_single = {}, {}
    for name in ARMS:
        arms[name], n_single[name] = ic.complete_arm(ic.arm_index(clusters[name]), scored)
    label_pos = {str(int(r.label_id)): (r.lat, r.lng) for r in labels.itertuples(index=False)}
    lines.append('- arms rebuilt: ' + ', '.join(f'{n} {len(clusters[n])}' for n in ARMS)
                 + ' clusters')
    lines.append(f'- scored label set: {len(scored)} labels in the bundle\'s windows; '
                 f'{n_human} human label(s) excluded from every arm; labels an arm does not '
                 'hold, scored as its singletons: '
                 + ', '.join(f'{n} {n_single[n]}' for n in ARMS))

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
                  'Review in RampNet `scripts/cluster_review_gallery.py` (branch '
                  '`cluster-review-224`) and save the export into the bundle. The pilot is '
                  'read with RampNet `scripts/cluster_review_agreement.py` only; this command '
                  "scores rater A's full pass and refuses anything earlier (protocol "
                  'Amendment 5, S1).', '', f'- {sanity}']
        (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        for name, fields in (('arms.csv', ARM_FIELDS), ('ramps.csv', ['corner_id', 'type', 'ramp']
                                                        + list(ARMS))):
            with open(out / name, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f, lineterminator='\n').writerow(fields)
        print('\n'.join(lines))
        print(f'wrote {out / "report.md"} (no GT yet)')
        return 0

    a = json.loads(a_path.read_text(encoding='utf-8'))
    problems = check_assignments(a, snapshot, corners)
    if problems:
        raise SystemExit(f'{a_path}: cannot be scored: ' + '; '.join(problems[:10]))
    refusal = arm_numbers_refusal(a, corners, bundle)
    if refusal and not args.allow_partial_descriptive:
        raise SystemExit(f"{a_path}: no arm-level number is computed before rater A's full pass "
                         f'is exported ({refusal}; RampNet protocol Amendment 5, S1). Read the '
                         'pilot with RampNet scripts/cluster_review_agreement.py (agreement, '
                         "timing, can't-judge counts). --allow-partial-descriptive overrides, "
                         'headed DESCRIPTIVE and recorded in the report.')
    units = {cid: u for cid, u in (a.get('corners') or {}).items()
             if u.get('complete') and not u.get('cant_judge')}
    cj_lines, n_cj = cant_judge_section(a, corners_by_id)
    notes = a.get('review_notes') or {}
    lines += [f"- assignments: `{a_path.name}` sha256 `{sha256_file(a_path)}`, rater "
              f"{a.get('rater') or notes.get('reviewer') or '?'}, rubric v{a.get('rubric_version')}, "
              f"seed {a.get('seed_arm')}, exported {a.get('exported_at')}; "
              f'{len(units)} complete, {n_cj} can\'t judge, of {len(a.get("corners") or {})} '
              'units in the file',
              f'- {sanity}']
    if notes.get('caveats'):
        lines += ['- reviewer caveats:'] + [f'  - {c}' for c in notes['caveats']]
    if refusal:
        lines.append(f'- **--allow-partial-descriptive**: arm-level numbers computed on a file '
                     f"that is not rater A's full pass ({refusal}); the protocol (Amendment 5, "
                     'S1) says none is read before that pass is exported')

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
    if v in (ic.PASS, ic.NOT_ESTABLISHED):
        # every arm clusters the same label set, so the coverage leg cannot differ
        reasons = [r + ' -- coverage is arm-independent by construction (same label set)'
                   for r in reasons]
    pre_registered, why = decision_scope(a, corners)
    scope = 'pre-registered' if pre_registered else f'DESCRIPTIVE -- {why}'
    lines += ['', f'## Decision ({scope}): **{v}**', '']
    lines += [f'- {r}' for r in reasons]
    d, d_ci, d_units = bootstrap_split_diff(by_arm[ic.REVIEW_BASE]['per_ramp'],
                                            by_arm[ic.REVIEW_ARM]['per_ramp'])
    lines.append(f'- descriptive, not part of the rule (Amendment 5, S8): split rate '
                 f'`{ic.REVIEW_ARM}` minus `{ic.REVIEW_BASE}` = {fmt(d)}, unit-level paired '
                 f'bootstrap 95% CI{ci(d_ci) or " n/a"} over {d_units} units '
                 f'({BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}); the bar is -{ic.RULE_SPLIT_DROP}')
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
    seen_inv = [cid for cid, u in units.items() if u.get('inventory_seen')]
    tainted = [cid for cid, u in units.items() if u.get('edited_after_inventory')]
    cal = inventory_calibration({cid: u for cid, u in units.items() if cid not in tainted},
                                corners_by_id, arms, label_pos)
    lines += ['', '## Calibration against the city inventory', '',
              f'- {len(seen_inv)} complete unit(s) revealed the inventory; {len(tainted)} were '
              'edited after the reveal and are DROPPED from this calibration (they stay in every '
              'other table)' + (f": {', '.join(sorted(tainted))}" if tainted else '')]
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
    lines += cj_lines
    lines += ['', '## Against verdicts.json', '',
              '- not run here: RampNet `rampnet.cluster_review.verdict_consistency` '
              f"([rampnet/cluster_review.py]({rampnet_url('rampnet/cluster_review.py')}), branch "
              '`cluster-review-224`) does it for cities with a benchmark bundle (Vancouver has '
              'none).']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    with open(out / 'arms.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ARM_FIELDS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ARM_FIELDS})
    with open(out / 'ramps.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['corner_id', 'type', 'ramp'] + list(ARMS))
        for key in sorted(by_arm[ARMS[0]]['per_ramp']):
            w.writerow([key[0], corners_by_id[key[0]]['type'], key[1]]
                       + [by_arm[n]['per_ramp'].get(key, 0) for n in ARMS])
    print('\n'.join(lines))
    print(f'wrote {out}/report.md, arms.csv, ramps.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
