#!/usr/bin/env python3
"""Placement check for #169 NoCurbRamp label points: where does each rule put the marker?

The second test-server smoke send (2026-11-15) put 2 of 3 site-median labels at the curb and
one in the street (vancouver-test:89739). Before more labels are sent, this draws a seeded
sample of residential corners OUTSIDE the pilot's send file, computes each candidate rule's
label point on the SAME chosen pano (the pano choice is keyed on the corner point, so it does
not depend on the rule), and renders one blind card per (corner, rule) with the marker drawn
on Project Sidewalk's own pixel (detectors.ps_pixel). Rules (corner_labels.LABEL_POINT_RULES):

  A  site-median               the build's stored point (reprojected and checked against it)
  B  sidewalk-crossing         first crossing of the bisector with a city sidewalk line
  C  sidewalk-crossing-inset   B moved back by the crossed line's UNITTYPE inset

Pre-registration, tag set and pass bar: docs/corner-labels-169-pilot.md, "Placement check".

    python scripts/corner_placement_check.py sample --out runs/vancouver/corner_placement169
    #   -> sample.json, cards.json (blind), key.json (rule per card), panos_needed.txt
    python scripts/corner_placement_check.py render --out runs/vancouver/corner_placement169 \\
        --panos <dir holding <id[:2]>/<id>.jpg>
    #   -> crops/, review/ montages, index.html (no rule shown)
    python scripts/corner_placement_check.py score --out runs/vancouver/corner_placement169 \\
        --tags runs/vancouver/corner_placement169/tags_ai_claude-opus-5-5.json
    #   -> summary.json, summary.md; re-run render with --summary to add the skim section

Nothing here talks to a server. Pixels come from a local copy of the PS pano store.
"""
import argparse
import hashlib
import html
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corner_labels as cl  # noqa: E402
import corner_inventory as ci  # noqa: E402
from detectors import ps_pixel  # noqa: E402

D = 'runs/vancouver/corner_labels169_split'
DEFAULTS = {
    'build': f'{D}/na_discount_sw_sitemed',
    'pilot': f'{D}/na_discount_sw/pilot_residential/pilot.csv',
    'smoke_panos': f'{D}/na_discount_sw_sitemed/pilot_residential/smoke_panos.txt',
    'corners': 'runs/vancouver/corner_inventory169_split/corners_full.jsonl',
    'sidewalks': 'runs/vancouver/corner_labels169/sidewalks/cov_sidewalks.geojson',
}
SEED = 20261115
N_SAMPLE = 60
#: first pilot position eligible: the send file holds positions 4-203 (rated 7-106, reserve
#: 107-126 after deviation 3), so 204+ is never sent from that file nor rated
MIN_POSITION = 204
STRATUM = 'residential'
RULES = {'A': 'site-median', 'B': 'sidewalk-crossing', 'C': 'sidewalk-crossing-inset'}
#: Jon's verdicts on the 2026-11-15 smoke send (rule A, the live viewer)
CALIBRATION = {'vancouver:res:n47220756#0': ('vancouver-test:89738', 'at_curb'),
               'vancouver:res:n47279117#3': ('vancouver-test:89739', 'in_street'),
               'vancouver:res:n47271259#0': ('vancouver-test:89740', 'at_curb')}
TAGS = ('at_curb', 'inland', 'in_street', 'cant_tell')
#: pass bar, fixed before any card was seen (docs, "Placement check")
BAR_AT_CURB = 0.90
BAR_MAX_STREET = 2
CROP_FOV_H, CROP_FOV_V = 50.0, 30.0      # degrees of the equirect around the marker
CROP_W, CROP_H = 800, 480
MONTAGE_COLS, MONTAGE_ROWS = 2, 3


def wilson(k, n, z=1.96):
    """Wilson score interval (lo, hi) of k of n; (None, None) for n = 0.

    >>> [round(v, 3) for v in wilson(9, 10)]
    [0.596, 0.982]
    """
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def rule_verdict(counts):
    """Pass/fail of one rule's tag counts under the pre-registered bar: at_curb >= 90% of the
    decidable cards (at_curb + inland + in_street) and at most 2 in_street.

    >>> rule_verdict({'at_curb': 54, 'inland': 3, 'in_street': 2, 'cant_tell': 1})['pass']
    True
    >>> rule_verdict({'at_curb': 54, 'inland': 0, 'in_street': 3, 'cant_tell': 3})['pass']
    False
    """
    dec = sum(counts.get(t, 0) for t in ('at_curb', 'inland', 'in_street'))
    k = counts.get('at_curb', 0)
    lo, hi = wilson(k, dec)
    share = k / dec if dec else None
    ok = bool(dec) and share >= BAR_AT_CURB and counts.get('in_street', 0) <= BAR_MAX_STREET
    return {'decidable': dec, 'at_curb': k, 'share': share, 'wilson95': [lo, hi],
            'in_street': counts.get('in_street', 0), 'inland': counts.get('inland', 0),
            'cant_tell': counts.get('cant_tell', 0), 'pass': ok}


def draw(order, chosen_pano, blocked_panos, n=N_SAMPLE, seed=SEED, min_position=MIN_POSITION):
    """(eligible keys, drawn keys): positions >= min_position (1-based) whose chosen pano is
    not among blocked_panos, then random.Random(seed).sample of n, kept in pilot order.

    >>> order = [f'k{i}' for i in range(1, 11)]
    >>> pano = {k: k.upper() for k in order}
    >>> el, got = draw(order, pano, {'K9'}, n=3, seed=1, min_position=6)
    >>> el
    ['k6', 'k7', 'k8', 'k10']
    >>> len(got), set(got) <= set(el)
    (3, True)
    """
    eligible = [k for i, k in enumerate(order, 1)
                if i >= min_position and chosen_pano.get(k) is not None
                and chosen_pano[k] not in blocked_panos]
    pick = set(random.Random(seed).sample(eligible, min(n, len(eligible))))
    return eligible, [k for k in eligible if k in pick]


class _Fixed:
    """Fallback placement for SidewalkCrossingPlacement: the build's stored site-median point."""

    def __init__(self, props):
        self.props = props

    def place(self, rec, c):
        lp = self.props[cl.corner_key(rec['unit'], c['corner'])]['label_point']
        return dict(lp, base_m=lp['base_m'])

    def record(self):
        return {'rule': 'site-median (stored in the build)'}


def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def write_json(path, obj):
    ci.write_text_lf(path, json.dumps(obj, indent=1, sort_keys=True) + '\n')


def cmd_sample(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build = json.loads((Path(args.build) / 'build.json').read_text(encoding='utf-8'))
    props = {p['key']: p for p in ci.read_jsonl(Path(args.build) / 'proposals.jsonl')
             if p['stratum'] == STRATUM}
    order = cl.read_corner_order(args.pilot)
    chosen = {k: props[k]['pano']['pano_id'] for k in order if k in props}
    rated_panos = {chosen[k] for k in order[:MIN_POSITION - 1] if k in chosen}
    smoke = set(cl.read_pano_list(args.smoke_panos))
    eligible, keys = draw(order, chosen, rated_panos | smoke, args.n, args.seed)
    calib = [k for k in CALIBRATION if k in props]
    if len(calib) != len(CALIBRATION):
        raise SystemExit(f'calibration corners missing from the build: '
                         f'{sorted(set(CALIBRATION) - set(calib))}')
    want = keys + calib
    units = {}
    for rec in ci.read_jsonl(args.corners):
        units[rec['unit']] = rec
    res = cl.run_results_paths(build)
    pano_ids = {props[k]['pano']['pano_id'] for k in want}
    panos = cl.load_pano_index([(name, rec['path']) for name, rec in sorted(res.items())],
                               pano_ids)
    fixed = _Fixed(props)
    placers = {'B': cl.SidewalkCrossingPlacement(args.sidewalks, fixed, 'sidewalk-crossing'),
               'C': cl.SidewalkCrossingPlacement(args.sidewalks, fixed,
                                                 'sidewalk-crossing-inset')}
    cam_h = build['params']['camera_height_m']
    rng_m = build['params']['max_range_m']
    corners, cards = [], []
    for k in want:
        p = props[k]
        rec = units[p['unit']]
        c = next(c for c in rec['corners'] if c['corner'] == p['corner'])
        pano = panos[p['pano']['pano_id']]
        pts = {'A': dict(p['label_point'])}
        for r, pl in placers.items():
            pts[r] = pl.place(rec, c)
        row = {'key': k, 'unit': p['unit'], 'corner': p['corner'],
               'position': order.index(k) + 1, 'calibration': k in CALIBRATION,
               'pano_id': p['pano']['pano_id'], 'width': p['pano']['width'],
               'height': p['pano']['height'], 'rules': {}}
        for r, lp in pts.items():
            if r == 'A' or lp['case'] == 'site_median_fallback':
                # the build's own pixel, exactly what is sent: its stored lat/lng is rounded
                # to 1e-6 deg, which moves a reprojection by pixels when the pano is close
                xn, yn, rng = p['x_normalized'], p['y_normalized'], p['range_m']
            else:
                proj = cl.project(pano, lp['lat'], lp['lng'], cam_h, rng_m)
                if proj is None:    # past the pano's range: the build keeps the corner point
                    proj = cl.project(pano, c['lat'], c['lng'], cam_h, rng_m)
                    lp = dict(lp, case='corner_range_fallback')
                xn, yn, rng = proj.x_norm, proj.y_norm, proj.range_m
            px, py = ps_pixel(xn, yn, p['pano']['width'], p['pano']['height'])
            row['rules'][r] = {'x_normalized': round(xn, 6), 'y_normalized': round(yn, 6),
                               'pano_x': px, 'pano_y': py, 'range_m': round(rng, 2),
                               'dist_m': round(lp['dist_m'], 2), 'case': lp['case'],
                               'sidewalk': lp.get('sidewalk')}
            cards.append({'key': k, 'rule': r})
        corners.append(row)
    random.Random(args.seed + 1).shuffle(cards)
    for i, cd in enumerate(cards, 1):
        cd['card'] = f'c{i:03d}'
    by_key = {r['key']: r for r in corners}
    blind = [{'card': cd['card'], 'pano_id': by_key[cd['key']]['pano_id'],
              'width': by_key[cd['key']]['width'], 'height': by_key[cd['key']]['height'],
              **{f: by_key[cd['key']]['rules'][cd['rule']][f]
                 for f in ('x_normalized', 'y_normalized', 'pano_x', 'pano_y')}}
             for cd in cards]
    write_json(out / 'cards.json', blind)
    write_json(out / 'key.json', [{'card': cd['card'], 'key': cd['key'], 'rule': cd['rule'],
                                   'calibration': cd['key'] in CALIBRATION} for cd in cards])
    rec = {'schema': 'sidewalk-auto-labeler.corner_placement_check.sample/1',
           'issue': 'sidewalk-auto-labeler#169', 'seed': args.seed, 'n': args.n,
           'min_position': MIN_POSITION, 'stratum': STRATUM, 'rules': RULES,
           'sidewalk_inset_by_type_m': cl.SIDEWALK_INSET_BY_TYPE,
           'sidewalk_ray_m': list(cl.SIDEWALK_RAY_M),
           'build': {'dir': str(args.build), 'tool_git_sha': build['tool']['git_sha']},
           'proposals_sha256': ci.sha256_file(Path(args.build) / 'proposals.jsonl'),
           'pilot_sha256': ci.sha256_file(args.pilot),
           'sidewalks_sha256': ci.sha256_file(args.sidewalks),
           'n_eligible': len(eligible), 'blocked_panos': len(rated_panos | smoke),
           'keys': keys, 'keys_sha256': sha256_text('\n'.join(keys)),
           'calibration': {k: {'label': v[0], 'jon': v[1]} for k, v in CALIBRATION.items()},
           'corners': corners}
    write_json(out / 'sample.json', rec)
    store = sorted({r['pano_id'] for r in corners})
    ci.write_text_lf(out / 'panos_needed.txt',
                     ''.join(f'{pid[:2]}/{pid}.jpg\n' for pid in store))
    print(f'{len(keys)} corners of {len(eligible)} eligible + {len(calib)} calibration; '
          f'{len(cards)} cards; {len(store)} panos -> {out}')


# ------------------------------------------------------------------------- render

def crop_card(im, xn, yn, fov_h=CROP_FOV_H, fov_v=CROP_FOV_V, size=(CROP_W, CROP_H)):
    """(tile, (mx, my)): an equirect window centred on the marker, resized to size, and the
    marker's position in it. Wraps across the seam; clamps at the poles.

    >>> from PIL import Image
    >>> t, (mx, my) = crop_card(Image.new('RGB', (3600, 1800)), 0.999, 0.5, 50, 30, (500, 300))
    >>> t.size, round(mx), round(my)
    ((500, 300), 250, 150)
    """
    from PIL import Image
    W, H = im.size
    tw = int(round(W * fov_h / 360.0))
    th = min(int(round(H * fov_v / 180.0)), H)
    cx, cy = xn * W, yn * H
    x0 = int(round(cx - tw / 2.0))
    y0 = max(0, min(H - th, int(round(cy - th / 2.0))))
    if x0 < 0 or x0 + tw > W:
        tile = Image.new('RGB', (tw, th))
        a = x0 % W
        first = min(tw, W - a)
        tile.paste(im.crop((a, y0, a + first, y0 + th)), (0, 0))
        if first < tw:
            tile.paste(im.crop((0, y0, tw - first, y0 + th)), (first, 0))
    else:
        tile = im.crop((x0, y0, x0 + tw, y0 + th))
    sx, sy = size[0] / tw, size[1] / th
    return tile.resize(size, Image.LANCZOS), ((cx - x0) * sx, (cy - y0) * sy)


def draw_marker(tile, mx, my):
    """PS-like marker: a filled magenta disc with a white ring and a dark centre dot on the
    exact pixel (the label's point is the disc's centre)."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(tile)
    r = 13
    d.ellipse([mx - r - 2, my - r - 2, mx + r + 2, my + r + 2], fill=(255, 255, 255))
    d.ellipse([mx - r, my - r, mx + r, my + r], fill=(232, 84, 170))
    d.ellipse([mx - 2, my - 2, mx + 2, my + 2], fill=(20, 20, 20))
    return tile


def cmd_render(args):
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    out = Path(args.out)
    cards = json.loads((out / 'cards.json').read_text(encoding='utf-8'))
    (out / 'crops').mkdir(exist_ok=True)
    by_pano = {}
    for cd in cards:
        by_pano.setdefault(cd['pano_id'], []).append(cd)
    missing = []
    for pid, cds in sorted(by_pano.items()):
        if all((out / 'crops' / f"{cd['card']}.jpg").exists() for cd in cds):
            continue
        path = Path(args.panos) / pid[:2] / f'{pid}.jpg'
        if not path.exists():
            missing.append(pid)
            continue
        im = Image.open(path).convert('RGB')
        for cd in cds:
            # the marker sits on PS's integer pixel of the record's width x height, as the
            # viewer draws it, scaled to whatever size the store copy has
            xn, yn = cd['pano_x'] / cd['width'], cd['pano_y'] / cd['height']
            tile, (mx, my) = crop_card(im, xn, yn)
            draw_marker(tile, mx, my).save(out / 'crops' / f"{cd['card']}.jpg", 'JPEG',
                                           quality=85)
        im.close()
    if missing:
        raise SystemExit(f'{len(missing)} pano(s) not under {args.panos}, e.g. {missing[:3]}')
    (out / 'review').mkdir(exist_ok=True)
    per = MONTAGE_COLS * MONTAGE_ROWS
    for m in range(0, len(cards), per):
        sheet = Image.new('RGB', (MONTAGE_COLS * CROP_W, MONTAGE_ROWS * (CROP_H + 28)), 'white')
        d = ImageDraw.Draw(sheet)
        for j, cd in enumerate(cards[m:m + per]):
            x, y = (j % MONTAGE_COLS) * CROP_W, (j // MONTAGE_COLS) * (CROP_H + 28)
            d.text((x + 8, y + 6), cd['card'], fill=(0, 0, 0))
            sheet.paste(Image.open(out / 'crops' / f"{cd['card']}.jpg"), (x, y + 28))
        sheet.save(out / 'review' / f'm{m // per + 1:02d}.jpg', 'JPEG', quality=85)
    summary = None
    if args.summary:
        summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    ci.write_text_lf(out / 'index.html', page_html(cards, summary))
    print(f'{len(cards)} cards, {len(by_pano)} panos -> {out / "index.html"}')


def page_html(cards, summary=None):
    """The local review page: the skim section (only once scored) and every card, blind."""
    e = html.escape
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>#169 placement check</title><style>',
             'body{font:15px/1.4 system-ui,sans-serif;margin:16px;background:#fff;color:#111}',
             '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));'
             'gap:12px}figure{margin:0}img{width:100%;height:auto;display:block}',
             'figcaption{font-size:13px;color:#333}table{border-collapse:collapse}',
             'td,th{border:1px solid #bbb;padding:2px 6px;text-align:left}',
             '@media (prefers-color-scheme: dark){body{background:#111;color:#eee}'
             'figcaption{color:#ccc}td,th{border-color:#555}}</style></head><body>',
             '<h1>#169 NoCurbRamp placement check</h1>',
             '<p>One card per (corner, rule). The magenta disc is where Project Sidewalk would '
             'draw the label. Tags: <b>at_curb</b> (at the curb edge, or on the sidewalk corner '
             'right behind it), <b>inland</b> (yard, grass, planter strip or driveway apron '
             'behind the curb), <b>in_street</b> (roadway), <b>cant_tell</b>.</p>']
    if summary:
        parts.append('<h2>Skim (after unblinding)</h2>')
        parts.append('<table><tr><th>rule</th><th>at_curb / decidable</th><th>in_street</th>'
                     '<th>inland</th><th>Wilson 95%</th><th>bar</th></tr>')
        for r, v in summary['rules'].items():
            lo, hi = v['wilson95']
            parts.append(f"<tr><td>{e(r)} {e(RULES[r])}</td><td>{v['at_curb']} / "
                         f"{v['decidable']}</td><td>{v['in_street']}</td><td>{v['inland']}</td>"
                         f"<td>{lo:.2f}-{hi:.2f}</td><td>{'PASS' if v['pass'] else 'FAIL'}"
                         '</td></tr>')
        parts.append('</table><div class="grid">')
        for s in summary['skim']:
            parts.append(f'<figure><img src="crops/{e(s["card"])}.jpg" alt="card '
                         f'{e(s["card"])}, rule {e(s["rule"])}, tagged {e(s["tag"])}">'
                         f'<figcaption>{e(s["card"])} &middot; rule {e(s["rule"])} &middot; '
                         f'{e(s["tag"])} &middot; {e(s["key"])}</figcaption></figure>')
        parts.append('</div>')
    parts.append('<h2>All cards (blind)</h2><div class="grid">')
    for cd in cards:
        parts.append(f'<figure><img loading="lazy" src="crops/{e(cd["card"])}.jpg" '
                     f'alt="card {e(cd["card"])}"><figcaption>{e(cd["card"])}</figcaption>'
                     '</figure>')
    parts.append('</div></body></html>\n')
    return '\n'.join(parts)


# -------------------------------------------------------------------------- score

def score(key_rows, tags):
    """{'rules': {A/B/C: rule_verdict over the sample}, 'calibration': [...], 'skim': [...],
    'disagreements': [...]} from key.json rows and {card: tag}. Every card must be tagged."""
    bad = sorted(set(tags.values()) - set(TAGS))
    if bad:
        raise SystemExit(f'unknown tags {bad}; allowed {TAGS}')
    untagged = sorted(r['card'] for r in key_rows if r['card'] not in tags)
    if untagged:
        raise SystemExit(f'{len(untagged)} card(s) untagged, e.g. {untagged[:3]}')
    counts = {r: Counter() for r in RULES}
    per_key, calib, skim = {}, [], []
    for row in key_rows:
        t = tags[row['card']]
        if row['calibration']:
            calib.append({'key': row['key'], 'rule': row['rule'], 'card': row['card'],
                          'tag': t, 'jon': CALIBRATION[row['key']][1] if row['rule'] == 'A'
                          else None, 'label': CALIBRATION[row['key']][0]})
            continue
        counts[row['rule']][t] += 1
        per_key.setdefault(row['key'], {})[row['rule']] = (t, row['card'])
        if t in ('in_street', 'inland'):
            skim.append({'card': row['card'], 'rule': row['rule'], 'tag': t, 'key': row['key']})
    dis = []
    for k, rs in sorted(per_key.items()):
        if len({t for t, _ in rs.values()}) > 1:
            dis.append({'key': k, **{r: rs[r][0] for r in sorted(rs)}})
            for r, (t, card) in rs.items():
                if t == 'at_curb':
                    skim.append({'card': card, 'rule': r, 'tag': t, 'key': k})
    skim.sort(key=lambda s: (s['key'], s['rule']))
    calib.sort(key=lambda s: (s['key'], s['rule']))
    return {'rules': {r: rule_verdict(counts[r]) for r in RULES}, 'calibration': calib,
            'skim': skim, 'disagreements': dis}


def cmd_score(args):
    out = Path(args.out)
    key_rows = json.loads((out / 'key.json').read_text(encoding='utf-8'))
    tag_doc = json.loads(Path(args.tags).read_text(encoding='utf-8'))
    s = score(key_rows, tag_doc['tags'])
    s['tags_file'] = str(args.tags)
    s['tags_sha256'] = ci.sha256_file(args.tags)
    s['rater'] = tag_doc.get('rater')
    s['bar'] = {'at_curb_share_min': BAR_AT_CURB, 'in_street_max': BAR_MAX_STREET}
    write_json(out / 'summary.json', s)
    lines = ['| rule | at_curb / decidable | share | Wilson 95% | in_street | inland | '
             "can't tell | bar |", '|---|---|---|---|---|---|---|---|']
    for r, v in s['rules'].items():
        lo, hi = v['wilson95']
        lines.append(f"| {r} {RULES[r]} | {v['at_curb']} / {v['decidable']} | "
                     f"{v['share']:.3f} | {lo:.3f}-{hi:.3f} | {v['in_street']} | {v['inland']} | "
                     f"{v['cant_tell']} | {'PASS' if v['pass'] else 'FAIL'} |")
    lines += ['', 'Calibration (Jon judged rule A only):', '',
              '| corner | label | rule | AI tag | Jon |', '|---|---|---|---|---|']
    for c in s['calibration']:
        lines.append(f"| {c['key']} | {c['label']} | {c['rule']} | {c['tag']} | "
                     f"{c['jon'] or ''} |")
    lines += ['', f"Corners where the rules' tags differ: {len(s['disagreements'])}", '']
    ci.write_text_lf(out / 'summary.md', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample', help='draw the corners, compute each rule, write blind cards')
    s.add_argument('--out', required=True)
    s.add_argument('--build', default=DEFAULTS['build'])
    s.add_argument('--pilot', default=DEFAULTS['pilot'])
    s.add_argument('--smoke-panos', default=DEFAULTS['smoke_panos'])
    s.add_argument('--corners', default=DEFAULTS['corners'])
    s.add_argument('--sidewalks', default=DEFAULTS['sidewalks'])
    s.add_argument('--n', type=int, default=N_SAMPLE)
    s.add_argument('--seed', type=int, default=SEED)
    r = sub.add_parser('render', help='cut the crops, montages and index.html')
    r.add_argument('--out', required=True)
    r.add_argument('--panos', required=True, help='local copy of the store: <id[:2]>/<id>.jpg')
    r.add_argument('--summary', help="score's summary.json: adds the skim section")
    c = sub.add_parser('score', help='per-rule counts, the bar, calibration')
    c.add_argument('--out', required=True)
    c.add_argument('--tags', required=True)
    args = ap.parse_args(argv)
    {'sample': cmd_sample, 'render': cmd_render, 'score': cmd_score}[args.cmd](args)


if __name__ == '__main__':
    main()
