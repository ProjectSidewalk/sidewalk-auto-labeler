"""Stage 1 of the corner pre-rater plan (issue #185): export the requests, score the answers.

Called through scripts/corner_prerater.py (`export`, `score`); the paid calls themselves are
made by scripts/corner_vlm.py with RampNet's Gemini harness. Plan:
corner-prerater-absence-2026-11-15.md, Stage 1, approved by Jon on 2026-11-15. Write-up:
docs/corner-prerater-185.md.

Fixed before any call (plan section 3): the prompt is the bundle's own rubric text verbatim
plus a three-value schema (`present` / `absent` / `cant_tell`, with `absent_kind`); no
inventory and no filenames reach the model; temperature 0; no prompt tuning. PROMPT_* below are
frozen and their sha256 goes into requests.json and every ledger row.

Arms (plan Stage 1):
  argon_multi_r1   gemini-4-argon, all of a corner's crops plus the aerial, one call (decides)
  argon_multi_r2   the same again; measures run-to-run change only
  argon_noaerial   gemini-4-argon, all crops, no aerial; measures only
  flash_multi      gemini-3.8-flash, all crops plus the aerial
  argon_single     gemini-4-argon, one call per crop; the arm's answer is the nearest crop's
  flash_single     gemini-3.8-flash, the same
The kill rule uses the four arms argon_multi_r1, flash_multi, argon_single, flash_single.

Kill rule (plan, kill only): an arm fails if it catches <= 2 of the 6 present units (a unit is
caught when every blind-present corner in it is called `present`) or its flag rate on the 365
blind-absent corners exceeds 10% (flag = `present` or `cant_tell`). Kill if every arm fails.
A call that returned no parseable verdict is `no_answer`: never a catch, and counted as a flag
on an absent corner, because it would go to the rater (declared before any call; the plan is
silent on it).
"""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

GALLERIES = {
    # name: (bundle dir, items sha256, verdicts sha256, rubric version)
    'g243': ('../gallery243/runs/vancouver/corner_gallery243',
             'bbed3d6ddaf1835a66da3afb274c3f338cc80512d4af7c4567c96c1bf47a1459',
             '30a5008d125f143d6cc536e9787676310fc474b9c4cc127f59a419a16e2e80bd', 1),
    'g169': ('runs/vancouver/corner_gallery169_split',
             'e7648a487a92ef508d488e40d8425de333f0d3719d2412a72a80138e15e26910',
             'da3f01147df9f502584d84c5b4b479c5eedc37ee0d7dce99d8f3e16fe88d0a64', 2),
}
MODELS = {'argon': 'gemini-4-argon', 'flash': 'gemini-3.8-flash'}
ARMS = {
    'argon_multi_r1': ('argon', 'multi', 1),
    'argon_multi_r2': ('argon', 'multi', 2),
    'argon_noaerial': ('argon', 'noaerial', 1),
    'flash_multi': ('flash', 'multi', 1),
    'argon_single': ('argon', 'single', 1),
    'flash_single': ('flash', 'single', 1),
}
KILL_ARMS = ('argon_multi_r1', 'flash_multi', 'argon_single', 'flash_single')
KILL_MAX_UNITS = 2          # catches <= 2 of 6 units fails the arm
KILL_MAX_FLAG = 0.10        # flag rate above 10% fails the arm
SMOKE_N, SMOKE_SEED = 20, 185
#: the one corner Jon judged from live imagery the model cannot see (plan Stage 1)
LIVE_IMAGERY_KEY = 'g243:vancouver:res:n47287760#2'
VERDICTS = ('present', 'absent', 'cant_tell')
ABSENT_KINDS = ('curb_no_ramp', 'no_sidewalk', 'unspecified')

RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': list(VERDICTS)},
        'absent_kind': {'type': 'string', 'enum': list(ABSENT_KINDS), 'nullable': True},
    },
    'required': ['verdict', 'absent_kind'],
    'propertyOrdering': ['verdict', 'absent_kind'],
}

PROMPT_HEAD = (
    'You are rating one street corner for curb ramps, as a human rater does in a gallery. '
    'The rubric the human rater follows is below, verbatim.\n\n<rubric>\n{rubric}</rubric>\n\n')
PROMPT_MULTI = (
    'Rate corner {corner} of this unit only. The first image is the aerial: north up, about '
    '70 m across. The solid white lines are the unit\'s legs, dashed white lines (if any) are '
    'phantom legs, the dashed white circle is the 30 m window, and each numbered white circle '
    'is a corner marker; corner {corner} is the one with the thick blue outline. The yellow '
    'numbered dots are the cameras of the crops that follow, each joined to corner {corner} by '
    'a dashed yellow line. Then come {n} crop(s) of street-level panoramas, each centred on '
    'corner {corner}\'s corner point (the orange ring).\n')
PROMPT_NOAERIAL = (
    'Rate corner {corner} of this unit only. No aerial is given. The images are {n} crop(s) of '
    'street-level panoramas, each centred on corner {corner}\'s corner point (the orange '
    'ring).\n')
PROMPT_SINGLE = (
    'Rate corner {corner} of this unit only, from the one crop below; no aerial and no other '
    'crop is given. It is a crop of a street-level panorama centred on corner {corner}\'s '
    'corner point (the orange ring).\n')
CAPTION = 'Crop {i}: captured {date}, camera {dist:.1f} m from the corner point.'
PROMPT_TAIL = (
    '\nThe city inventory is not available. Answer in JSON: `verdict` is present, absent or '
    'cant_tell; `absent_kind` is curb_no_ramp, no_sidewalk or unspecified when the verdict is '
    'absent, and null otherwise.')
PROMPT_TEXTS = {'head': PROMPT_HEAD, 'multi': PROMPT_MULTI, 'noaerial': PROMPT_NOAERIAL,
                'single': PROMPT_SINGLE, 'caption': CAPTION, 'tail': PROMPT_TAIL}


def prompt_sha256():
    """One digest over every frozen prompt piece and the schema."""
    blob = json.dumps({'texts': PROMPT_TEXTS, 'schema': RESPONSE_SCHEMA}, sort_keys=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')


# ------------------------------------------------------------------- aerial rendering

def world_px(lat, lng, zoom):
    """Web-Mercator world pixel at zoom (the page's worldPx)."""
    s = 256 * 2 ** zoom
    x = (lng + 180.0) / 360.0 * s
    sl = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sl) / (1 - sl)) / (4 * math.pi)) * s
    return x, y


def to_img(aerial, lat, lng):
    """Pixel on the aerial jpg of (lat, lng): the page's toImg."""
    w = aerial['world_px']
    x, y = world_px(lat, lng, aerial['zoom'])
    return ((x - w['x0']) / (w['x1'] - w['x0']) * aerial['px'],
            (y - w['y0']) / (w['y1'] - w['y0']) * aerial['px'])


def px_per_metre(aerial, centre_lat):
    mpp = 156543.03392 * math.cos(math.radians(centre_lat)) / 2 ** aerial['zoom']
    return aerial['px'] / (aerial['world_px']['x1'] - aerial['world_px']['x0']) / mpp


def _dashed(draw, p, q, dash, gap, fill, width):
    length = math.hypot(q[0] - p[0], q[1] - p[1])
    if length == 0:
        return
    ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        draw.line([(p[0] + ux * t, p[1] + uy * t), (p[0] + ux * e, p[1] + uy * e)],
                  fill=fill, width=width)
        t = e + gap


def render_aerial(item, corner_idx, src, dst, scale=2):
    """The aerial as the page draws it with corner_idx active: window circle, legs, phantom
    legs dashed, every corner marker numbered (active one outlined blue), the active corner's
    crop cameras in yellow joined to its corner point. No verdict colours and no inventory.
    Drawn at `scale` x the jpg for legible marks."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(src).convert('RGB')
    img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    a = item['aerial']

    def P(lat, lng):
        x, y = to_img(a, lat, lng)
        return x * scale, y * scale
    try:
        font = ImageFont.truetype('arialbd.ttf', 14 * scale)
        small = ImageFont.truetype('arialbd.ttf', 12 * scale)
    except OSError:
        font = small = ImageFont.load_default()
    c0 = P(item['centre']['lat'], item['centre']['lng'])
    r = item['window_m'] * px_per_metre(a, item['centre']['lat']) * scale
    n_seg = 72
    for i in range(0, n_seg, 2):
        t0, t1 = 2 * math.pi * i / n_seg, 2 * math.pi * (i + 1) / n_seg
        d.line([(c0[0] + r * math.sin(t0), c0[1] - r * math.cos(t0)),
                (c0[0] + r * math.sin(t1), c0[1] - r * math.cos(t1))], fill='white',
               width=int(1.5 * scale))
    for b in item['legs']:
        t = math.radians(b)
        d.line([c0, (c0[0] + r * math.sin(t), c0[1] - r * math.cos(t))], fill='white',
               width=2 * scale)
    for b in item.get('phantom_legs') or ():
        t = math.radians(b)
        _dashed(d, c0, (c0[0] + r * math.sin(t), c0[1] - r * math.cos(t)), 9 * scale,
                6 * scale, 'white', int(1.5 * scale))
    ac = item['corners'][corner_idx]
    q = P(ac['lat'], ac['lng'])
    for vi, v in enumerate(ac['views']):
        p = P(v['cam']['lat'], v['cam']['lng'])
        _dashed(d, p, q, 4 * scale, 3 * scale, '#ffd400', int(1.5 * scale))
    for vi, v in enumerate(ac['views']):
        p = P(v['cam']['lat'], v['cam']['lng'])
        rr = 7 * scale
        d.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill='#ffd400', outline='black')
        d.rectangle([p[0] + 8 * scale, p[1] - 9 * scale, p[0] + 24 * scale, p[1] + 9 * scale],
                    fill='black')
        d.text((p[0] + 16 * scale, p[1]), str(vi + 1), fill='#ffd400', font=small, anchor='mm')
    for ci, c in enumerate(item['corners']):
        p = P(c['lat'], c['lng'])
        rr = 12 * scale
        active = ci == corner_idx
        d.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill='white',
                  outline='#0b63ce' if active else 'black', width=(5 if active else 2) * scale)
        d.text(p, str(c['corner']), fill='black', font=font, anchor='mm')
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, 'PNG', optimize=True)


# --------------------------------------------------------------------------- export

def load_gallery(name, root):
    bundle, items_sha, verdicts_sha, rubric_v = GALLERIES[name]
    b = Path(root) / bundle if not Path(bundle).is_absolute() else Path(bundle)
    for f, want in (('items.jsonl', items_sha), ('verdicts__jonf.json', verdicts_sha)):
        got = sha256_file(b / f)
        if got != want:
            raise SystemExit(f'{b / f}: sha256 {got} is not the pinned {want}')
    snap = json.loads((b / 'snapshot.json').read_text(encoding='utf-8'))
    if snap.get('rubric_version') != rubric_v:
        raise SystemExit(f'{b}: rubric v{snap.get("rubric_version")}, expected v{rubric_v}')
    pins = {}
    with open(b / 'images.sha256', encoding='utf-8') as f:
        for line in f:
            h, p = line.split(None, 1)
            pins[p.strip().lstrip('*')] = h
    return b, read_jsonl(b / 'items.jsonl'), json.loads(
        (b / 'verdicts__jonf.json').read_text(encoding='utf-8')), snap['rubric'], pins


def nearest_view(views):
    """Index of the crop nearest the corner point (first on a tie): the single-view arm's
    answer (plan Stage 1, review F4)."""
    return min(range(len(views)), key=lambda i: (views[i]['dist_m'], i))


def build_requests(root, out_dir, render=True):
    """One request per judged corner (381), with the rendered aerial, the crops (pinned to the
    bundle's images.sha256) and the blind verdict kept beside it for scoring only."""
    out_dir = Path(out_dir)
    reqs = []
    for g in GALLERIES:
        b, items, verdicts, rubric, pins = load_gallery(g, root)
        for it in items:
            uv = verdicts['units'][it['unit']]
            for ci, c in enumerate(it['corners']):
                key = f"{g}:{it['unit']}#{c['corner']}"
                crops = []
                for vi, v in enumerate(c['views']):
                    if pins.get(v['crop']) is None:
                        raise SystemExit(f'{key}: {v["crop"]} not in images.sha256')
                    got = sha256_file(b / v['crop'])
                    if got != pins[v['crop']]:
                        raise SystemExit(f'{key}: {v["crop"]} does not match images.sha256')
                    crops.append({'i': vi + 1, 'path': str((b / v['crop']).resolve()),
                                  'sha256': got, 'date': v['capture_date'],
                                  'dist_m': v['dist_m']})
                safe = f"{g}__{it['unit'].replace(':', '_')}__c{c['corner']}.png"
                aerial_path = out_dir / 'aerial' / safe
                if render:
                    render_aerial(it, ci, b / it['aerial']['file'], aerial_path)
                blind = uv['blind'][str(c['corner'])]
                final = uv['corners'][str(c['corner'])]
                reqs.append({
                    'key': key, 'gallery': g, 'unit': it['unit'], 'corner': c['corner'],
                    'rubric_version': GALLERIES[g][3], 'rubric': rubric,
                    'aerial': {'path': str(aerial_path.resolve()),
                               'sha256': sha256_file(aerial_path) if aerial_path.exists()
                               else None},
                    'crops': crops, 'nearest': nearest_view(c['views']) + 1,
                    'truth': {'blind': blind['verdict'],
                              'blind_kind': blind.get('absent_kind'),
                              'final': final['verdict'],
                              'used_live_imagery': bool(uv.get('used_live_imagery'))},
                })
    reqs.sort(key=lambda r: r['key'])
    return reqs


def smoke_keys(reqs, n=SMOKE_N, seed=SMOKE_SEED):
    """The seeded 20-corner smoke sample (sorted keys, random.Random(185).sample)."""
    return sorted(random.Random(seed).sample(sorted(r['key'] for r in reqs), n))


def prompt_parts(req, mode, crop_i=None):
    """The request body as an ordered list of ('text', str) / ('image', path) parts.

    multi: head, task, aerial, then a caption and image per crop, tail. noaerial: the same
    without the aerial. single: head, task, the one crop's caption and image, tail. The
    images go to the model as bytes; no path or filename ever reaches it."""
    head = PROMPT_HEAD.format(rubric=req['rubric'])
    crops = req['crops'] if mode != 'single' else [c for c in req['crops'] if c['i'] == crop_i]
    if mode == 'single':
        parts = [('text', head + PROMPT_SINGLE.format(corner=req['corner']))]
    elif mode == 'multi':
        parts = [('text', head + PROMPT_MULTI.format(corner=req['corner'], n=len(crops))),
                 ('text', 'Aerial:'), ('image', req['aerial']['path'])]
    elif mode == 'noaerial':
        parts = [('text', head + PROMPT_NOAERIAL.format(corner=req['corner'], n=len(crops)))]
    else:
        raise ValueError(mode)
    for c in crops:
        i = 1 if mode == 'single' else c['i']
        parts.append(('text', CAPTION.format(i=i, date=c['date'] or 'unknown',
                                             dist=c['dist_m'])))
        parts.append(('image', c['path']))
    parts.append(('text', PROMPT_TAIL))
    return parts


def call_keys(reqs, arm, keys=None, views='all'):
    """The call ids an arm makes: the corner key for a multi-image arm, `<key>@<crop i>` per
    crop for a single-view arm. views: nearest | rest | all (single-view arms only)."""
    _, mode, _ = ARMS[arm]
    want = None if keys is None else set(keys)
    out = []
    for r in reqs:
        if want is not None and r['key'] not in want:
            continue
        if mode != 'single':
            out.append(r['key'])
            continue
        for c in r['crops']:
            near = c['i'] == r['nearest']
            if views == 'all' or (views == 'nearest' and near) or (views == 'rest' and not near):
                out.append(f"{r['key']}@{c['i']}")
    return out


# ---------------------------------------------------------------------------- score

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, mid - half), min(1.0, mid + half)]


def arm_answers(reqs, responses, arm):
    """{corner key: verdict or 'no_answer' or None (not called)} for one arm."""
    _, mode, _ = ARMS[arm]
    by_id = {r['call_id']: r for r in responses}
    out = {}
    for q in reqs:
        cid = q['key'] if mode != 'single' else f"{q['key']}@{q['nearest']}"
        r = by_id.get(cid)
        out[q['key']] = None if r is None else (r.get('verdict') or 'no_answer')
    return out


def kappa(pairs):
    """Cohen's kappa over (truth, model) label pairs; None when undefined."""
    n = len(pairs)
    if not n:
        return None
    labels = sorted({x for p in pairs for x in p})
    po = sum(a == b for a, b in pairs) / n
    ta, tb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ta[l] * tb[l] for l in labels) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def present_sets(reqs):
    """The present-corner sets of the plan: blind (primary, 9), final (6), and both without
    the live-imagery corner. {name: {unit: [keys]}}."""
    sets = {'blind': {}, 'final': {}, 'blind_no_live': {}, 'final_no_live': {}}
    for q in reqs:
        u = f"{q['gallery']}:{q['unit']}"
        if q['truth']['blind'] == 'present':
            sets['blind'].setdefault(u, []).append(q['key'])
            if q['key'] != LIVE_IMAGERY_KEY:
                sets['blind_no_live'].setdefault(u, []).append(q['key'])
        if q['truth']['final'] == 'present':
            sets['final'].setdefault(u, []).append(q['key'])
            if q['key'] != LIVE_IMAGERY_KEY:
                sets['final_no_live'].setdefault(u, []).append(q['key'])
    return sets


def score_arm(reqs, answers, gallery=None):
    """Catches, flag rate and agreement for one arm's answers, optionally one gallery."""
    rs = [q for q in reqs if gallery is None or q['gallery'] == gallery]
    keys = {q['key'] for q in rs}
    called = [q for q in rs if answers.get(q['key']) is not None]
    absent = [q for q in rs if q['truth']['blind'] == 'absent']
    a_called = [q for q in absent if answers.get(q['key']) is not None]
    flagged = [q for q in a_called if answers[q['key']] in ('present', 'cant_tell', 'no_answer')]
    tally = Counter(answers.get(q['key']) or 'not_called' for q in rs)
    out = {'corners': len(rs), 'called': len(called), 'answers': dict(sorted(tally.items())),
           'absent': len(absent), 'absent_called': len(a_called), 'flagged': len(flagged),
           'flag_rate': (len(flagged) / len(a_called)) if a_called else None,
           'flag_wilson95': wilson(len(flagged), len(a_called)),
           'flag_present': sum(answers[q['key']] == 'present' for q in a_called),
           'flag_cant_tell': sum(answers[q['key']] == 'cant_tell' for q in a_called),
           'flag_no_answer': sum(answers[q['key']] == 'no_answer' for q in a_called)}
    for name, units in present_sets(reqs).items():
        units = {u: ks for u, ks in units.items() if all(k in keys for k in ks)}
        caught_units = [u for u, ks in units.items()
                        if all(answers.get(k) == 'present' for k in ks)]
        corners = [k for ks in units.values() for k in ks]
        out[f'present_{name}'] = {
            'units': len(units), 'units_caught': len(caught_units),
            'units_wilson95': wilson(len(caught_units), len(units)),
            'corners': len(corners),
            'corners_caught': sum(answers.get(k) == 'present' for k in corners),
            'corners_cant_tell': sum(answers.get(k) == 'cant_tell' for k in corners),
            'caught_units': sorted(caught_units)}
    pairs = [(q['truth']['blind'], answers[q['key']]) for q in called
             if answers[q['key']] in VERDICTS]
    out['kappa_blind'] = kappa(pairs)
    # absent_kind agreement on no_sidewalk, among corners both call absent
    return out


def kind_agreement(reqs, responses, arm):
    """no_sidewalk agreement among corners the rater and the arm both call absent."""
    _, mode, _ = ARMS[arm]
    by_id = {r['call_id']: r for r in responses}
    t = Counter()
    for q in reqs:
        cid = q['key'] if mode != 'single' else f"{q['key']}@{q['nearest']}"
        r = by_id.get(cid)
        if not r or r.get('verdict') != 'absent' or q['truth']['blind'] != 'absent':
            continue
        h = q['truth']['blind_kind'] == 'no_sidewalk'
        m = r.get('absent_kind') == 'no_sidewalk'
        t['both' if h and m else 'rater_only' if h else 'model_only' if m else 'neither'] += 1
    return dict(sorted(t.items()))


def kill_verdict(arm_scores):
    """The pre-registered kill rule over the four kill arms. Returns (verdict, per-arm)."""
    per = {}
    for arm in KILL_ARMS:
        s = arm_scores.get(arm)
        if s is None or s['called'] < s['corners']:
            per[arm] = 'incomplete'
            continue
        p = s['present_blind']
        fails = []
        if p['units_caught'] <= KILL_MAX_UNITS:
            fails.append(f"catches {p['units_caught']} of {p['units']} units (<= {KILL_MAX_UNITS})")
        if s['flag_rate'] > KILL_MAX_FLAG:
            fails.append(f"flag rate {s['flag_rate']:.3f} > {KILL_MAX_FLAG}")
        per[arm] = 'fail: ' + '; '.join(fails) if fails else 'pass'
    if any(v == 'incomplete' for v in per.values()):
        return 'NOT DECIDED (an arm is incomplete)', per
    if all(v.startswith('fail') for v in per.values()):
        return 'KILL (every arm fails)', per
    return 'NO KILL (at least one arm passes)', per


def replicate_change(reqs, r1, r2):
    """Argon multi r1 vs r2: corners answered by both, and how many differ."""
    a, b = arm_answers(reqs, r1, 'argon_multi_r1'), arm_answers(reqs, r2, 'argon_multi_r2')
    both = [k for k in a if a[k] is not None and b.get(k) is not None]
    diff = [k for k in both if a[k] != b[k]]
    return {'both': len(both), 'differ': len(diff),
            'differ_share': len(diff) / len(both) if both else None,
            'differ_keys': sorted(diff)}
