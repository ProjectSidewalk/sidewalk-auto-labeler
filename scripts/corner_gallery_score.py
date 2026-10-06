"""Validate and score the corner present/absent gallery (RampNet#243).

Reads a bundle written by scripts/corner_gallery.py (items.jsonl, snapshot.json) and every
``verdicts__<rater>.json`` beside it. CPU only, stdlib only, no network.

Scoring reads each unit's **blind** verdicts (those given before the city inventory was
shown), from every unit that has them: a unit completed once and then reopened keeps its
blind verdicts and is scored on them. The final verdicts are reported beside them as a
sensitivity read, from complete units only. Unit outcome, from the corner verdicts:

  present        at least one corner rated present
  absent         every corner rated absent
  undetermined   no corner present and at least one can't tell

Reported per rater:

  na_noramp      the share of the sampled `NA`-no-`RAMPTYPE` absent units the rater confirms
                 as no ramp = absent / (absent + present), Wilson 95%; undetermined units
                 are counted and left out of that share, and a conservative share (absent /
                 all complete) is printed beside it. Also the corner-level read at the
                 corners holding the `NA` points, with the kind of absence.
  false_absence  each of the 35 units as one of
                   miss_at_inventory_corner   a corner holding an `Available` point is present
                   miss_elsewhere             no such corner present, another corner present
                   artifact_built_after_imagery  no corner present; every inventory corner
                                              absent; every `Available` point there has an
                                              INSTDATE (month) later than the newest crop
                                              SHOWN at its corner (the imagery the rater
                                              judged; since review B1 the build always shows
                                              the newest capture in the candidate pool, and
                                              each corner's `newest_available` is reported
                                              beside it in `dating`)
                   artifact_inventory_or_geometry  no corner present; every inventory corner
                                              absent; otherwise
                   undetermined               no corner present; an inventory corner can't tell
                 real misses = the two miss_*; artifacts = the two artifact_*.
  clean          control agreement: the share of the control units rated absent at every
                 corner (absent / (absent + present)), Wilson 95%; corner level beside it.
  stratified     a descriptive estimate of absence precision against the rater over the
                 sampled strata (weights = population sizes), with the units outside every
                 stratum (the 9 non-target false absences and 2 'other' units) named.

Given two raters (--a / --b), corner-level agreement on the blind verdicts (percent and
Cohen's kappa over the three values) and unit-outcome agreement.

    python scripts/corner_gallery_score.py runs/vancouver/corner_gallery243
    python scripts/corner_gallery_score.py runs/vancouver/corner_gallery243 --a jonf --b jonf-retest
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = 'sidewalk-auto-labeler.corner_gallery/1'
RUBRIC_VERSION = 1
VERDICTS = ('present', 'absent', 'cant_tell')
ABSENT_KINDS = ('curb_no_ramp', 'no_sidewalk')
OUTCOMES = ('present', 'absent', 'undetermined')
FA_CLASSES = ('miss_at_inventory_corner', 'miss_elsewhere', 'artifact_built_after_imagery',
              'artifact_inventory_or_geometry', 'undetermined')
#: units called absent in the merged build that belong to no sampled stratum (amendment A8:
#: 1,323 absent = 333 clean + 944 NA-only + 44 with an Available point + 2 other)
ABSENT_TOTAL = 1323


def wilson(k, n, z=1.96):
    """Wilson score interval; None when n == 0.

    >>> [round(x, 3) for x in wilson(9, 10)]
    [0.596, 0.982]
    """
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def share(k, n):
    ci = wilson(k, n)
    return {'k': k, 'n': n, 'share': round(k / n, 4) if n else None,
            'ci95': [round(ci[0], 4), round(ci[1], 4)] if ci else None}


def fmt(s):
    if not s['n']:
        return f"{s['k']}/0"
    return f"{s['k']}/{s['n']} = {s['share']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"


# ----------------------------------------------------------------------------- loading

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(bundle):
    """(snapshot, items, {rater: verdicts}). Raters come from ``verdicts__<rater>.json``."""
    d = Path(bundle)
    snapshot = json.loads((d / 'snapshot.json').read_text(encoding='utf-8'))
    items = [json.loads(x) for x in (d / 'items.jsonl').read_text(encoding='utf-8').splitlines()
             if x.strip()]
    files = {}
    for p in sorted(d.glob('verdicts__*.json')):
        files[p.name[len('verdicts__'):-len('.json')]] = json.loads(p.read_text(encoding='utf-8'))
    return snapshot, items, files


# -------------------------------------------------------------------------- validation

def _check_corner_map(cid, what, m, keys, problems, need_all):
    if not isinstance(m, dict):
        problems.append(f'{cid}: {what} is not an object')
        return
    for k in sorted(set(m) - keys):
        problems.append(f'{cid}: {what} names corner {k}, which the unit does not have')
    for k in sorted(keys - set(m)):
        problems.append(f'{cid}: {what} lacks corner {k}')
    for k, e in sorted(m.items()):
        if not isinstance(e, dict):
            problems.append(f'{cid}: {what} corner {k} is not an object')
            continue
        v, ak = e.get('verdict'), e.get('absent_kind')
        if v is not None and v not in VERDICTS:
            problems.append(f'{cid}: {what} corner {k} verdict {v!r}')
        if need_all and v is None:
            problems.append(f'{cid}: {what} corner {k} has no verdict')
        if ak is not None and ak not in ABSENT_KINDS:
            problems.append(f'{cid}: {what} corner {k} absent_kind {ak!r}')
        if ak is not None and v != 'absent':
            problems.append(f'{cid}: {what} corner {k} has absent_kind with verdict {v!r}')


def validate(verdicts, items, items_sha, rubric=None):
    """Every reason the file cannot be scored, as strings (empty = valid). rubric: the bundle's
    rubric text (snapshot.json); when given, the file's rubric must be that exact text, so an
    edit to the rubric without a version bump cannot mix two definitions silently."""
    problems = []
    if verdicts.get('schema') != SCHEMA:
        problems.append(f"schema {verdicts.get('schema')!r}, expected {SCHEMA!r}")
    if verdicts.get('items_sha256') != items_sha:
        problems.append(f"items_sha256 {str(verdicts.get('items_sha256'))[:12]} is not the "
                        f'bundle\'s {items_sha[:12]}')
    if verdicts.get('rubric_version') != RUBRIC_VERSION:
        problems.append(f"rubric_version {verdicts.get('rubric_version')!r}, expected "
                        f'{RUBRIC_VERSION}')
    if not isinstance(verdicts.get('rubric'), str) or not verdicts['rubric'].strip():
        problems.append('the rubric text is missing (it travels in the verdicts file)')
    elif rubric is not None and verdicts['rubric'] != rubric:
        problems.append('the rubric text differs from the bundle rubric (snapshot.json): the page '
                        'was rendered with another rubric')
    if not isinstance(verdicts.get('rater'), str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',
                                                                     verdicts.get('rater') or ''):
        problems.append(f"rater {verdicts.get('rater')!r}")
    by_id = {i['unit']: i for i in items}
    for cid, u in sorted((verdicts.get('units') or {}).items()):
        it = by_id.get(cid)
        if it is None:
            problems.append(f'{cid}: not a unit of this bundle')
            continue
        keys = {str(c['corner']) for c in it['corners']}
        complete = u.get('complete')
        for flag in ('complete', 'inventory_seen', 'edited_after_inventory'):
            if flag in u and not isinstance(u[flag], bool):
                problems.append(f'{cid}: {flag} {u[flag]!r} is not a boolean')
        _check_corner_map(cid, 'corners', u.get('corners'), keys, problems, bool(complete))
        if u.get('blind') is not None:
            _check_corner_map(cid, 'blind', u['blind'], keys, problems, True)
        if complete and u.get('blind') is None:
            problems.append(f'{cid}: complete with no blind verdicts')
        if complete and not u.get('inventory_seen'):
            problems.append(f'{cid}: complete but inventory_seen is false')
        if u.get('edited_after_inventory') and not u.get('inventory_seen'):
            problems.append(f'{cid}: edited_after_inventory without inventory_seen')
        e = u.get('elapsed_s', 0)
        if not isinstance(e, (int, float)) or e < 0:
            problems.append(f'{cid}: elapsed_s {e!r}')
    return problems


# ----------------------------------------------------------------------------- scoring

def outcome(corner_verdicts):
    """present / absent / undetermined from {corner: verdict}."""
    vs = list(corner_verdicts.values())
    if any(v == 'present' for v in vs):
        return 'present'
    if vs and all(v == 'absent' for v in vs):
        return 'absent'
    return 'undetermined'


def verdict_map(u, which='blind'):
    m = u.get(which) if which == 'blind' else u.get('corners')
    return {k: (e or {}).get('verdict') for k, e in (m or {}).items()}


def kind_map(u, which='blind'):
    m = u.get(which) if which == 'blind' else u.get('corners')
    return {k: (e or {}).get('absent_kind') for k, e in (m or {}).items()}


def classify_false_absence(item, verdicts):
    """One false-absence unit's class (FA_CLASSES); verdicts: {corner key: verdict}."""
    inv = [c for c in item['corners'] if c['inv_counts'].get('Available', 0) > 0]
    inv_keys = {str(c['corner']) for c in inv}
    if any(verdicts.get(k) == 'present' for k in inv_keys):
        return 'miss_at_inventory_corner'
    if any(v == 'present' for v in verdicts.values()):
        return 'miss_elsewhere'
    if any(verdicts.get(k) != 'absent' for k in inv_keys) or not inv_keys:
        return 'undetermined'
    late = True
    for c in inv:
        newest = max((v.get('capture_date') or '' for v in c['views']), default='')
        for p in c['inventory']:
            if p['class'] != 'Available':
                continue
            inst = (p.get('instdate') or '')[:7]
            if not (inst and newest and inst > newest):
                late = False
    return 'artifact_built_after_imagery' if late else 'artifact_inventory_or_geometry'


def fa_dating(item):
    """{corner key: {newest_shown, newest_available, instdates}} at the corners holding an
    `Available` point: the imagery the rater judged vs the newest the build had."""
    out = {}
    for c in item['corners']:
        if not c['inv_counts'].get('Available', 0):
            continue
        out[str(c['corner'])] = {
            'newest_shown': max((v.get('capture_date') or '' for v in c['views']), default='')
            or None,
            'newest_available': c.get('newest_available'),
            'instdates': sorted((p.get('instdate') or '')[:7] for p in c['inventory']
                                if p['class'] == 'Available')}
    return out


def scorable(u, which='blind'):
    """Whether a unit enters a read: the blind read takes every unit with blind verdicts
    (frozen at the first completion, so a reopened unit still has them); the final read
    takes complete units only."""
    u = u or {}
    return u.get('blind') is not None if which == 'blind' else bool(u.get('complete'))


def score_rater(items, verdicts, snapshot, which='blind'):
    """The per-rater report (dict) from the blind (default) or final verdicts; see scorable
    for which units enter each read."""
    units = verdicts.get('units') or {}
    done = {i['unit']: units[i['unit']] for i in items if scorable(units.get(i['unit']), which)}
    out = {'which': which, 'scored': len(done), 'units': len(items),
           'complete': sum(1 for i in items if (units.get(i['unit']) or {}).get('complete')),
           'reopened_with_blind': sum(1 for i in items if (units.get(i['unit']) or {}).get('blind')
                                      is not None and not units[i['unit']].get('complete')),
           'edited_after_inventory': sum(1 for u in done.values()
                                         if u.get('edited_after_inventory')),
           'parts': {}}
    rows = []
    for part in ('false_absence', 'na_noramp', 'clean'):
        its = [i for i in items if i['part'] == part]
        dn = [i for i in its if i['unit'] in done]
        oc = {o: 0 for o in OUTCOMES}
        corner = {v: 0 for v in VERDICTS}
        kinds = {k: 0 for k in ABSENT_KINDS + ('unspecified',)}
        for i in dn:
            vm, km = verdict_map(done[i['unit']], which), kind_map(done[i['unit']], which)
            o = outcome(vm)
            oc[o] += 1
            rows.append({'unit': i['unit'], 'part': part, 'outcome': o})
            for c in i['corners']:
                k = str(c['corner'])
                if part == 'na_noramp' and not c['inv_counts'].get('NA_noramp'):
                    continue     # corner level for NA: only the corners holding the NA points
                v = vm.get(k)
                if v in corner:
                    corner[v] += 1
                if v == 'absent':
                    kinds[km.get(k) or 'unspecified'] += 1
        p = {'n': len(its), 'scored': len(dn), 'outcomes': oc,
             'unit_absent_share': share(oc['absent'], oc['absent'] + oc['present']),
             'unit_absent_conservative': share(oc['absent'], len(dn)),
             'corner_verdicts': corner,
             'corner_absent_share': share(corner['absent'], corner['absent'] + corner['present']),
             'absent_kinds': kinds}
        if part == 'false_absence':
            cls = {c: 0 for c in FA_CLASSES}
            per, dating = {}, {}
            for i in dn:
                c = classify_false_absence(i, verdict_map(done[i['unit']], which))
                cls[c] += 1
                per[i['unit']] = c
                dating[i['unit']] = fa_dating(i)
            miss = cls['miss_at_inventory_corner'] + cls['miss_elsewhere']
            art = cls['artifact_built_after_imagery'] + cls['artifact_inventory_or_geometry']
            stale = sorted(u for u, d in dating.items() for e in d.values()
                           if (e['newest_available'] or '') > (e['newest_shown'] or ''))
            p.update({'classes': cls, 'per_unit': per, 'dating': dating,
                      'shown_older_than_available': stale,
                      'real_misses': miss, 'artifacts': art,
                      'real_miss_share': share(miss, miss + art)})
        out['parts'][part] = p
    out['rows'] = rows
    # stratified, descriptive: absence precision against the rater over the sampled strata
    pops = snapshot.get('populations') or {}
    strata = []
    for part, conf in (('clean', 'absent'), ('na_noramp', 'absent'),
                       ('false_absence', None)):
        p = out['parts'][part]
        if part == 'false_absence':
            k = p['artifacts']
            n = p['real_misses'] + p['artifacts']
        else:
            k, n = p['outcomes']['absent'], p['outcomes']['absent'] + p['outcomes']['present']
        strata.append((part, pops.get(part, 0), k, n))
    if all(n for _p, _N, _k, n in strata):
        tot = sum(N for _p, N, _k, _n in strata)
        est = sum(N * k / n for _p, N, k, n in strata) / tot
        var = sum((N / tot) ** 2 * (k / n) * (1 - k / n) / n * max(0.0, 1 - n / N)
                  for _p, N, k, n in strata if N)
        out['stratified'] = {'estimate': round(est, 4),
                             'ci95_normal': [round(max(0.0, est - 1.96 * math.sqrt(var)), 4),
                                             round(min(1.0, est + 1.96 * math.sqrt(var)), 4)],
                             'covered_units': tot, 'absent_units': ABSENT_TOTAL,
                             'not_covered': ABSENT_TOTAL - tot}
    else:
        out['stratified'] = None
    return out


# --------------------------------------------------------------------------- agreement

def cohen_kappa(pairs, cats=VERDICTS):
    """Cohen's kappa over (a, b) category pairs; None when undefined.

    >>> cohen_kappa([('present', 'present'), ('absent', 'absent'), ('absent', 'present')])
    0.4
    """
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n)
             for c in cats)
    if pe >= 1:
        return None
    return round((po - pe) / (1 - pe), 4)


def agreement(items, va, vb):
    ua, ub = va.get('units') or {}, vb.get('units') or {}
    both = [i for i in items if scorable(ua.get(i['unit'])) and scorable(ub.get(i['unit']))]
    pairs, unit_pairs = [], []
    for i in both:
        a, b = verdict_map(ua[i['unit']]), verdict_map(ub[i['unit']])
        for c in i['corners']:
            k = str(c['corner'])
            pairs.append((a.get(k), b.get(k)))
        unit_pairs.append((outcome(a), outcome(b)))
    return {'units_both_blind': len(both), 'corners': len(pairs),
            'corner_agree': share(sum(1 for a, b in pairs if a == b), len(pairs)),
            'corner_kappa': cohen_kappa(pairs),
            'unit_outcome_agree': share(sum(1 for a, b in unit_pairs if a == b), len(unit_pairs)),
            'unit_outcome_kappa': cohen_kappa(unit_pairs, OUTCOMES)}


# ------------------------------------------------------------------------------ report

def render_report(snapshot, items, results, problems, agree=None):
    lines = ['# Corner present/absent gallery (RampNet#243): scores', '',
             f"Items sha256 `{snapshot['items_sha256']}`, seed {snapshot['seed']}. Scored on the "
             'blind verdicts (given before the city inventory was shown), every unit that has '
             'them; the final verdicts, complete units only, are the sensitivity read. Definitions: the docstring of '
             '`scripts/corner_gallery_score.py`.', '']
    if not results:
        lines.append('No verdicts file yet: nothing has been rated.')
    for rater, res in results.items():
        b, f = res['blind'], res['final']
        lines += [f'## Rater `{rater}`', '']
        if problems.get(rater):
            lines += [f'**INVALID ({len(problems[rater])} problems); not scored.**', '']
            lines += [f'- {p}' for p in problems[rater][:30]] + ['']
            continue
        lines += [f"{b['complete']} of {b['units']} units complete; {b['scored']} with blind "
                  f"verdicts ({b['reopened_with_blind']} reopened and not completed again); "
                  f"{b['edited_after_inventory']} edited after the inventory was shown.", '']
        na, fa, cl = b['parts']['na_noramp'], b['parts']['false_absence'], b['parts']['clean']
        lines += ['### `NA` with no `RAMPTYPE`: does the city mean "no ramp"?', '',
                  f"- Units rated absent at every corner, of units decided: "
                  f"**{fmt(na['unit_absent_share'])}** (blind); final "
                  f"{fmt(f['parts']['na_noramp']['unit_absent_share'])}.",
                  f"- Outcomes: {na['outcomes']} of {na['scored']} scored "
                  f"(population {snapshot['populations']['na_noramp']}, sample {na['n']}). "
                  f"Conservative (undetermined counted against): "
                  f"{fmt(na['unit_absent_conservative'])}.",
                  f"- Corners holding the `NA` points: {na['corner_verdicts']}; absent share "
                  f"{fmt(na['corner_absent_share'])}; kinds of absence {na['absent_kinds']}.", '',
                  '### The 35 false absences', '',
                  f"- Classes: {fa.get('classes')}.",
                  f"- Real misses {fa.get('real_misses')}, artifacts {fa.get('artifacts')}; "
                  f"real-miss share of decided: **{fmt(fa['real_miss_share'])}** (blind); final "
                  f"{fmt(f['parts']['false_absence']['real_miss_share'])}.",
                  f"- Units where an inventory corner's newest crop shown is older than the "
                  f"newest pano available to the build: "
                  f"{len(fa.get('shown_older_than_available') or [])} (expected 0; per-corner "
                  f"dates are under `dating` in score.json).", '',
                  '### Control: clean absences', '',
                  f"- Rated absent at every corner, of decided: **{fmt(cl['unit_absent_share'])}**"
                  f"; outcomes {cl['outcomes']}; corner level {fmt(cl['corner_absent_share'])}.",
                  '']
        st = b.get('stratified')
        if st:
            lines += ['### Stratified estimate (descriptive, not the decision rule)', '',
                      f"Absence precision against the rater over the sampled strata "
                      f"({st['covered_units']} of {st['absent_units']} absent units; the other "
                      f"{st['not_covered']} are the 9 non-target false absences and 2 'other' "
                      f"units): {st['estimate']:.3f}, normal-approximation 95% "
                      f"[{st['ci95_normal'][0]:.3f}, {st['ci95_normal'][1]:.3f}].", '']
        lines += ['### Per unit', '', '| unit | part | outcome (blind) | false-absence class |',
                  '|---|---|---|---|']
        for r in b['rows']:
            lines.append(f"| `{r['unit']}` | {r['part']} | {r['outcome']} | "
                         f"{fa.get('per_unit', {}).get(r['unit'], '')} |")
        lines.append('')
    if agree:
        lines += [f"## Agreement `{agree['a']}` vs `{agree['b']}`", '',
                  f"{agree['units_both_blind']} units with blind verdicts from both, {agree['corners']} "
                  f"corners. Corner agreement {fmt(agree['corner_agree'])}, kappa "
                  f"{agree['corner_kappa']}. Unit outcome agreement "
                  f"{fmt(agree['unit_outcome_agree'])}, kappa {agree['unit_outcome_kappa']}.", '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('bundle', type=Path)
    ap.add_argument('--a', default=None, help='rater A for agreement')
    ap.add_argument('--b', default=None, help='rater B for agreement')
    ap.add_argument('--out', type=Path, default=None, help='default <bundle>/score')
    args = ap.parse_args(argv)
    snapshot, items, files = load_bundle(args.bundle)
    items_sha = sha256_file(Path(args.bundle) / 'items.jsonl')
    if items_sha != snapshot['items_sha256']:
        raise SystemExit('items.jsonl does not match snapshot.json')
    problems, results = {}, {}
    for rater, v in files.items():
        problems[rater] = validate(v, items, items_sha, snapshot.get('rubric'))
        if v.get('rater') != rater:
            problems[rater].append(f"file name says rater {rater!r}, the file {v.get('rater')!r}")
        results[rater] = {'blind': score_rater(items, v, snapshot, 'blind'),
                          'final': score_rater(items, v, snapshot, 'final')}
        print(f"verdicts__{rater}.json: "
              f"{'valid' if not problems[rater] else f'{len(problems[rater])} problem(s)'}; "
              f"{results[rater]['blind']['complete']} complete units, "
              f"{results[rater]['blind']['scored']} with blind verdicts")
        for p in problems[rater][:10]:
            print('  ' + p)
    agree = None
    if args.a and args.b:
        for r in (args.a, args.b):
            if r not in files:
                raise SystemExit(f'no verdicts__{r}.json in {args.bundle}')
        agree = dict(agreement(items, files[args.a], files[args.b]), a=args.a, b=args.b)
    out = args.out or Path(args.bundle) / 'score'
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(snapshot, items, results, problems, agree)
    with open(out / 'report.md', 'w', encoding='utf-8', newline='\n') as f:
        f.write(report)
    with open(out / 'score.json', 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({'items_sha256': items_sha, 'problems': problems,
                            'results': results, 'agreement': agree}, indent=1,
                           sort_keys=True) + '\n')
    print(report)
    return 1 if any(problems.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
