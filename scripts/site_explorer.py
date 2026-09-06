"""Visual explorer for fused multi-view sites (issue #27, review tool).

Renders a self-contained HTML page where each fused site is one card: a crop from
every member view side by side, a plan view showing the camera positions, their
rays, the per-detection error ellipses and the fused covariance, plus the RampNet
ground-truth verdict wherever the site was seen by a judged benchmark pano.

Association is recomputed in memory with `fuse_sites.fuse` (the same call
`eval_sites.py` makes), so the explorer always shows exactly what the eval scored
and never needs a prior `fuse_sites.py` run.

Pixels come from the makelab2 native-resolution archive rather than the live APIs:
it holds every processed pano of every finished run, it covers the Mapillary
cities without a token or an expiring signed URL, and it is immune to pano decay.
Crops are cut *on makelab2* and pulled back as a tarball -- a 7-view site costs
~7 x 100 KB instead of 7 x 16 MB -- then cached under
`runs/<city>/explorer/crops/`, so re-renders are offline and free.

Usage:
    python scripts/site_explorer.py richmond                    # GT sites, 40 of them
    python scripts/site_explorer.py sao_paulo --select fp       # the false positives
    python scripts/site_explorer.py paterson --select spread -n 60
    python scripts/site_explorer.py richmond --no-fetch         # cache only, offline
    # ...or skip makelab2 entirely and cut crops locally from a pano directory:
    python scripts/site_explorer.py paterson --local-panos ../RampNet/benchmark/paterson/panos
"""
import argparse
import base64
import hashlib
import html
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import fuse_sites as fs  # noqa: E402
import eval_sites as es  # noqa: E402

DEFAULT_HELPER = os.environ.get(
    'WSL_SSH_HELPER', r'D:\Git\dotfiles\wsl-ssh.ps1')
DEFAULT_REMOTE_ROOT = '/projects/makeabilitylab/sidewalk-auto-labeler'
DEFAULT_REMOTE_HOME = '/homes/gws/jonf'

PANO_URL = {
    'gsv': 'https://www.google.com/maps/@?api=1&map_action=pano&pano_id={id}',
    'mapillary': 'https://www.mapillary.com/app/?pKey={id}&focus=photo',
    # The federation viewer resolves any member instance's picture id, so one template
    # covers every Panoramax run regardless of which instance holds the picture.
    'panoramax': 'https://api.panoramax.xyz/#focus=pic&pic={id}',
}


# --------------------------------------------------------------------------
# the crop worker that runs on makelab2 (Python 3.9 + Pillow there; keep it
# dependency-free and 3.9-compatible -- it is shipped as a string)
# --------------------------------------------------------------------------
CROP_WORKER = r'''
import json, os, sys
from multiprocessing import Pool
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def one_pano(task):
    pano_id, items, panos_root, out_dir = task
    path = os.path.join(panos_root, pano_id + ".jpg")
    if not os.path.exists(path):
        return [it["name"] for it in items], []
    try:
        im = Image.open(path)
        im.load()
        im = im.convert("RGB")
    except Exception:
        return [it["name"] for it in items], []
    W, H = im.size
    done = []
    for it in items:
        side = max(64, int(round(W * it["fov_deg"] / 360.0)))
        side = min(side, H)
        cx, cy = it["x"] * W, it["y"] * H
        x0 = int(round(cx - side / 2.0))
        y0 = int(round(cy - side / 2.0))
        y0 = max(0, min(H - side, y0))
        if x0 < 0 or x0 + side > W:          # wrap across the equirect seam
            tile = Image.new("RGB", (side, side))
            a = x0 % W
            first = min(side, W - a)
            tile.paste(im.crop((a, y0, a + first, y0 + side)), (0, 0))
            if first < side:
                tile.paste(im.crop((0, y0, side - first, y0 + side)), (first, 0))
        else:
            tile = im.crop((x0, y0, x0 + side, y0 + side))
        d = ImageDraw.Draw(tile)
        mx, my = side / 2.0, cy - y0
        r = max(6.0, side * 0.055)
        w = max(2, side // 180)
        d.ellipse([mx - r, my - r, mx + r, my + r], outline=(255, 64, 0), width=w)
        d.line([mx - r * 2.2, my, mx - r * 1.25, my], fill=(255, 64, 0), width=w)
        d.line([mx + r * 1.25, my, mx + r * 2.2, my], fill=(255, 64, 0), width=w)
        d.line([mx, my - r * 2.2, mx, my - r * 1.25], fill=(255, 64, 0), width=w)
        d.line([mx, my + r * 1.25, mx, my + r * 2.2], fill=(255, 64, 0), width=w)
        px = it["px"]
        tile = tile.resize((px, px), Image.LANCZOS)
        tile.save(os.path.join(out_dir, it["name"]), "JPEG", quality=82,
                  optimize=True)
        done.append(it["name"])
    im.close()
    return [], done


def main():
    job = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    by_pano = {}
    for it in job["items"]:
        by_pano.setdefault(it["pano_id"], []).append(it)
    tasks = [(pid, items, job["panos_root"], out_dir)
             for pid, items in sorted(by_pano.items())]
    missing, made = [], 0
    with Pool(job["workers"]) as pool:
        for miss, done in pool.imap_unordered(one_pano, tasks, chunksize=1):
            missing.extend(miss)
            made += len(done)
    with open(os.path.join(out_dir, "_missing.txt"), "w") as f:
        f.write("\n".join(sorted(missing)))
    sys.stderr.write("cropped %d, missing %d\n" % (made, len(missing)))


main()
'''


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def site_verdicts(site, op_verdicts):
    """Verdicts of this site's members that a judged GT pano decided."""
    return [op_verdicts[(d.pano_id, d.det_index)]
            for d, _ in site.members
            if d.operational and (d.pano_id, d.det_index) in op_verdicts]


def gt_label(verdicts):
    """Collapse member verdicts to the site's GT status (eval_sites semantics)."""
    if not verdicts:
        return None
    if any(v is True or v == 'duplicate' for v in verdicts):
        return 'true'
    if any(v is False for v in verdicts):
        return 'false'
    return 'unsure'


def neighbors_within(op_sites, radius_m):
    """{site.id: [(distance, other_site), ...] sorted} over operational sites.

    A fragmented ramp shows up as two sites a couple of metres apart, which is
    invisible on a single card — this is what makes over-split auditable.
    """
    if radius_m <= 0:
        return {s.id: [] for s in op_sites}
    cell = radius_m
    grid = {}
    for s in op_sites:
        grid.setdefault((int(s.e // cell), int(s.n // cell)), []).append(s)
    out = {}
    for s in op_sites:
        ce, cn = int(s.e // cell), int(s.n // cell)
        near = []
        for de in (-1, 0, 1):
            for dn in (-1, 0, 1):
                for o in grid.get((ce + de, cn + dn), ()):
                    if o.id == s.id:
                        continue
                    d = math.hypot(o.e - s.e, o.n - s.n)
                    if d <= radius_m:
                        near.append((d, o))
        out[s.id] = sorted(near, key=lambda t: (t[0], t[1].id))
    return out


def best_view(site):
    """The most legible member: nearest operational view, else nearest of any."""
    ops = [m for m in site.members if m[0].operational]
    return min(ops or site.members, key=lambda m: m[0].ground.range_m)


def select_sites(sites, op_verdicts, mode, n, min_views, seed, explicit_ids,
                 nbrs=None, fragment_m=3.0):
    op_sites = [s for s in sites if s.n_operational > 0]
    if explicit_ids:
        wanted = set(explicit_ids)
        return [s for s in sites if s.id in wanted]
    pool = [s for s in op_sites if len(s.pano_ids) >= min_views]
    labelled = [(s, gt_label(site_verdicts(s, op_verdicts))) for s in pool]

    if mode == 'gt':
        chosen = [s for s, lab in labelled if lab is not None]
        chosen.sort(key=lambda s: (-len(s.pano_ids), s.id))
    elif mode in ('tp', 'fp', 'unsure'):
        want = {'tp': 'true', 'fp': 'false', 'unsure': 'unsure'}[mode]
        chosen = [s for s, lab in labelled if lab == want]
        chosen.sort(key=lambda s: (-len(s.pano_ids), s.id))
    elif mode == 'multiview':
        chosen = sorted(pool, key=lambda s: (-len(s.pano_ids), s.id))
    elif mode == 'fragment':
        # over-split suspects: another operational site sits within fragment_m,
        # closest pair first. Some are real dual ramps — that is the judgement.
        # Only the lower-id half of each pair is listed; the other half is the
        # neighbour shown on its card, so listing both just duplicates the page.
        cand = []
        for s in pool:
            near = nbrs.get(s.id) or []
            if near and near[0][0] <= fragment_m and s.id < near[0][1].id:
                cand.append((near[0][0], s))
        chosen = [s for _, s in sorted(cand, key=lambda t: (t[0], t[1].id))]
    elif mode == 'spread':
        scored = [(s.residual_per_dof(), s.id, s) for s in pool
                  if s.residual_per_dof() is not None]
        chosen = [s for _, _, s in sorted(scored, key=lambda t: (-t[0], t[1]))]
    elif mode == 'random':
        chosen = sorted(pool, key=lambda s: s.id)
        random.Random(seed).shuffle(chosen)
    else:
        raise SystemExit('unknown --select ' + mode)
    return chosen[:n]


# --------------------------------------------------------------------------
# crop fetching
# --------------------------------------------------------------------------
def crop_name(det, fov_deg, px):
    return '{}_{}_{:g}_{}.jpg'.format(det.pano_id, det.det_index, fov_deg, px)


def shown_views(site, cap):
    """Members to render, nearest first — the close views are the readable ones."""
    members = sorted(site.members, key=lambda m: m[0].ground.range_m)
    return members[:cap], max(0, len(members) - cap)


def build_job(sites, crops_dir, fov_deg, px, workers, panos_root, cap,
              neighbor_of=None):
    items, seen = [], set()

    def want(det):
        name = crop_name(det, fov_deg, px)
        if name in seen or (crops_dir / name).exists():
            return
        seen.add(name)
        items.append({'pano_id': det.pano_id, 'x': det.x, 'y': det.y,
                      'fov_deg': fov_deg, 'px': px, 'name': name})

    for site in sites:
        for det, _ in shown_views(site, cap)[0]:
            want(det)
        for _, other in (neighbor_of or {}).get(site.id, ()):
            want(best_view(other)[0])          # one crop per neighbouring site
    return {'items': items, 'workers': workers, 'panos_root': panos_root}


def run_helper(helper, host, *args, timeout=1800):
    cmd = ['pwsh', '-NoProfile', '-NonInteractive', '-File', helper, host]
    cmd += [str(a) for a in args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc


def fetch_crops_remote(job, crops_dir, helper, host, remote_root,
                       remote_home, timeout):
    """Cut the crops on makelab2 and pull them back as one tarball."""
    payload = json.dumps(job, separators=(',', ':'), sort_keys=True)
    tag = hashlib.sha1(payload.encode()).hexdigest()[:12]
    stage = '{}/.sidewalk_explorer/{}'.format(remote_home, tag)
    script = '\n'.join([
        'set -eu',
        'STAGE="{}"'.format(stage),
        'rm -rf "$STAGE" "$STAGE.tar.gz"',
        'mkdir -p "$STAGE/crops"',
        "cat > \"$STAGE/job.json\" <<'JOB_EOF'",
        payload,
        'JOB_EOF',
        "cat > \"$STAGE/crop.py\" <<'PY_EOF'",
        CROP_WORKER,
        'PY_EOF',
        '"{}/.venv/bin/python" "$STAGE/crop.py" "$STAGE/job.json" '
        '"$STAGE/crops"'.format(remote_root),
        'tar -czf "$STAGE.tar.gz" -C "$STAGE/crops" .',
        'rm -rf "$STAGE"',
        'echo TARBALL_READY',
        '',
    ])
    with tempfile.TemporaryDirectory() as tmp:
        sh = Path(tmp) / 'explorer_crop.sh'
        with open(sh, 'w', encoding='utf-8', newline='\n') as f:
            f.write(script)                      # LF only: CRLF breaks bash
        print('  cropping {} views on {} ...'.format(len(job['items']), host))
        proc = run_helper(helper, host, 'script', str(sh),
                          '-TimeoutSec', timeout, timeout=timeout + 120)
        out = (proc.stdout or '') + (proc.stderr or '')
        if 'TARBALL_READY' not in out:
            raise SystemExit('remote crop failed:\n' + out.strip())
        for line in out.splitlines():
            if line.startswith('cropped '):
                print('  ' + line.strip())
        local_tar = Path(tmp) / 'crops.tar.gz'
        proc = run_helper(helper, host, 'get', stage + '.tar.gz',
                          str(local_tar), '-TimeoutSec', timeout,
                          timeout=timeout + 120)
        if not local_tar.exists():
            raise SystemExit('pulling the tarball failed:\n'
                             + ((proc.stdout or '') + (proc.stderr or '')))
        with tarfile.open(local_tar) as tf:
            for member in tf.getmembers():   # write by hand: no path traversal
                if not member.isfile():
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(crops_dir / os.path.basename(member.name), 'wb') as f:
                    f.write(src.read())
        run_helper(helper, host, 'run', 'rm', '-f', stage + '.tar.gz',
                   timeout=120)
    missing = crops_dir / '_missing.txt'
    names = [n for n in missing.read_text().split('\n') if n.strip()] \
        if missing.exists() else []
    if missing.exists():
        missing.unlink()
    return set(names)


def fetch_crops_local(job, crops_dir, panos_dir):
    """Same crops, cut locally from a pano directory (no SSH)."""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    by_pano = {}
    for it in job['items']:
        by_pano.setdefault(it['pano_id'], []).append(it)
    missing = set()
    for pano_id, items in sorted(by_pano.items()):
        path = Path(panos_dir) / (pano_id + '.jpg')
        if not path.exists():
            missing.update(it['name'] for it in items)
            continue
        im = Image.open(path).convert('RGB')
        W, H = im.size
        for it in items:
            side = min(max(64, int(round(W * it['fov_deg'] / 360.0))), H)
            cx, cy = it['x'] * W, it['y'] * H
            x0 = int(round(cx - side / 2.0))
            y0 = max(0, min(H - side, int(round(cy - side / 2.0))))
            if x0 < 0 or x0 + side > W:
                tile = Image.new('RGB', (side, side))
                a = x0 % W
                first = min(side, W - a)
                tile.paste(im.crop((a, y0, a + first, y0 + side)), (0, 0))
                if first < side:
                    tile.paste(im.crop((0, y0, side - first, y0 + side)),
                               (first, 0))
            else:
                tile = im.crop((x0, y0, x0 + side, y0 + side))
            d = ImageDraw.Draw(tile)
            mx, my = side / 2.0, cy - y0
            r, w = max(6.0, side * 0.055), max(2, side // 180)
            d.ellipse([mx - r, my - r, mx + r, my + r], outline=(255, 64, 0),
                      width=w)
            tile.resize((it['px'], it['px']), Image.LANCZOS).save(
                crops_dir / it['name'], 'JPEG', quality=82, optimize=True)
        im.close()
    return missing


# --------------------------------------------------------------------------
# plan view
# --------------------------------------------------------------------------
def ellipse_axes(cov, nsig=1.0):
    """(semi-major, semi-minor, rotation-deg-CCW-from-east) of a sym2 covariance."""
    a, b, c = cov
    tr, det = a + c, a * c - b * b
    disc = max(0.0, (tr / 2.0) ** 2 - det)
    l1, l2 = tr / 2.0 + math.sqrt(disc), max(1e-9, tr / 2.0 - math.sqrt(disc))
    theta = 0.5 * math.atan2(2 * b, a - c)
    return nsig * math.sqrt(l1), nsig * math.sqrt(l2), math.degrees(theta)


def plan_svg(site, gt_points, neighbors=(), width=460, height=300, pad=26):
    """SVG plan view in site-local ENU metres. East right, north up."""
    pts = [(0.0, 0.0)]
    pts += [(o.e - site.e, o.n - site.n) for _, o in neighbors]
    cams = []
    for det, in_refit in site.members:
        de, dn = det.e - site.e, det.n - site.n
        b = math.radians(det.ground.bearing_deg)
        ce = de - det.ground.range_m * math.sin(b)
        cn = dn - det.ground.range_m * math.cos(b)
        cams.append((ce, cn, de, dn, det, in_refit))
        pts += [(de, dn), (ce, cn)]
    pts += [(g[0] - site.e, g[1] - site.n) for g in gt_points]

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(6.0, max(max(xs) - min(xs), max(ys) - min(ys)))
    cx0, cy0 = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
    scale = min(width - 2 * pad, height - 2 * pad) / span

    def T(e, n):
        return (width / 2.0 + (e - cx0) * scale,
                height / 2.0 - (n - cy0) * scale)

    out = ['<svg class="plan" viewBox="0 0 {} {}" width="{}" height="{}" '
           'role="img" aria-label="plan view">'.format(width, height,
                                                       width, height)]
    out.append('<rect width="{}" height="{}" class="planbg"/>'.format(
        width, height))
    for ce, cn, de, dn, det, in_refit in cams:
        x1, y1 = T(ce, cn)
        x2, y2 = T(de, dn)
        cls = 'ray' if in_refit else 'ray sub'
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                   'class="{}"/>'.format(x1, y1, x2, y2, cls))
        out.append('<path d="M{:.1f},{:.1f} l-5,9 l10,0 z" class="cam" '
                   'transform="rotate({:.1f} {:.1f} {:.1f})"><title>{} · '
                   '{:.1f} m · conf {:.2f}</title></path>'.format(
                       x1, y1 - 5, det.ground.bearing_deg, x1, y1,
                       html.escape(det.pano_id), det.ground.range_m, det.conf))
        smaj, smin, rot = ellipse_axes(det.cov)
        out.append('<ellipse cx="{:.1f}" cy="{:.1f}" rx="{:.1f}" ry="{:.1f}" '
                   'transform="rotate({:.1f} {:.1f} {:.1f})" class="derr"/>'
                   .format(x2, y2, smaj * scale, smin * scale, -rot, x2, y2))
        out.append('<circle cx="{:.1f}" cy="{:.1f}" r="2.6" class="det {}"/>'
                   .format(x2, y2, '' if in_refit else 'sub'))
    for ge, gn in gt_points:
        gx, gy = T(ge - site.e, gn - site.n)
        out.append('<path d="M{:.1f},{:.1f} l0,-9 M{:.1f},{:.1f} l0,9 '
                   'M{:.1f},{:.1f} l-9,0 M{:.1f},{:.1f} l9,0" class="gt"/>'
                   .format(gx, gy, gx, gy, gx, gy, gx, gy))
    for dist, other in neighbors:
        ox, oy = T(other.e - site.e, other.n - site.n)
        out.append('<circle cx="{:.1f}" cy="{:.1f}" r="5" class="nbr">'
                   '<title>site {} · {:.1f} m away · {} views</title></circle>'
                   .format(ox, oy, other.id, dist, len(other.pano_ids)))
        out.append('<text x="{:.1f}" y="{:.1f}" class="nbrlab">{:.1f} m</text>'
                   .format(ox + 7, oy + 3.5, dist))
    sx, sy = T(0, 0)
    smaj, smin, rot = ellipse_axes(site.cov_p)
    out.append('<ellipse cx="{:.1f}" cy="{:.1f}" rx="{:.1f}" ry="{:.1f}" '
               'transform="rotate({:.1f} {:.1f} {:.1f})" class="serr"/>'
               .format(sx, sy, smaj * scale, smin * scale, -rot, sx, sy))
    out.append('<circle cx="{:.1f}" cy="{:.1f}" r="4.5" class="site"/>'
               .format(sx, sy))
    bar_m = max(1, round(span / 4))
    bx, by = pad, height - 12
    out.append('<line x1="{}" y1="{}" x2="{:.1f}" y2="{}" class="bar"/>'
               .format(bx, by, bx + bar_m * scale, by))
    out.append('<text x="{}" y="{}" class="barlab">{} m</text>'.format(
        bx, by - 5, bar_m))
    out.append('</svg>')
    return ''.join(out)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
CSS = """
:root{--bg:#fbfaf9;--fg:#1c1a19;--mut:#6b6663;--card:#fff;--line:#e3dedb;
--true:#0f7b4f;--false:#b3261e;--unsure:#8a6d00;--sub:#8a8683;--accent:#c2410c;
--nbr:#7c3aed}
@media (prefers-color-scheme:dark){:root{--bg:#141312;--fg:#eceae8;--mut:#9c9691;
--card:#1e1c1b;--line:#33302e;--true:#4ade80;--false:#f87171;--unsure:#fbbf24;
--sub:#7a7570;--accent:#fb923c;--nbr:#c4b5fd}}
:root[data-theme=dark]{--bg:#141312;--fg:#eceae8;--mut:#9c9691;--card:#1e1c1b;
--line:#33302e;--true:#4ade80;--false:#f87171;--unsure:#fbbf24;--sub:#7a7570;
--accent:#fb923c;--nbr:#c4b5fd}
:root[data-theme=light]{--bg:#fbfaf9;--fg:#1c1a19;--mut:#6b6663;--card:#fff;
--line:#e3dedb;--true:#0f7b4f;--false:#b3261e;--unsure:#8a6d00;--sub:#8a8683;
--accent:#c2410c;--nbr:#7c3aed}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,
system-ui,-apple-system,"Segoe UI",sans-serif}
header.top{padding:22px 24px 14px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:19px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px}
.filters{display:flex;gap:7px;flex-wrap:wrap;padding:12px 24px;position:sticky;
top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5}
.filters button{font:inherit;font-size:12.5px;padding:4px 11px;border-radius:99px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.filters button[aria-pressed=true]{background:var(--fg);color:var(--bg);
border-color:var(--fg)}
main{padding:18px 24px 60px;display:flex;flex-direction:column;gap:16px}
article{background:var(--card);border:1px solid var(--line);border-radius:11px;
padding:13px 15px;display:grid;grid-template-columns:1fr auto;gap:14px;
align-items:start}
@media(max-width:900px){article{grid-template-columns:1fr}}
article.hidden{display:none}
.hd{grid-column:1/-1;display:flex;gap:11px;align-items:baseline;flex-wrap:wrap;
font-variant-numeric:tabular-nums}
.sid{font-weight:650}
.hd .m{color:var(--mut);font-size:12.5px}
.badge{font-size:11.5px;font-weight:650;padding:1.5px 8px;border-radius:99px;
border:1px solid currentColor}
.badge.true{color:var(--true)}.badge.false{color:var(--false)}
.badge.unsure{color:var(--unsure)}
.views{display:flex;gap:9px;overflow-x:auto;padding-bottom:5px;min-width:0}
figure{margin:0;flex:0 0 auto;width:170px}
figure img{width:170px;height:170px;border-radius:7px;display:block;
background:var(--line);object-fit:cover}
figure.subth img{opacity:.62;outline:2px dashed var(--sub);outline-offset:-2px}
figcaption{font-size:11.5px;color:var(--mut);margin-top:5px;line-height:1.35;
font-variant-numeric:tabular-nums}
figcaption .c{color:var(--fg);font-weight:600}
figcaption a{color:inherit}
.miss{width:170px;height:170px;border-radius:7px;border:1px dashed var(--line);
display:flex;align-items:center;justify-content:center;color:var(--mut);
font-size:11.5px;text-align:center;padding:8px}
svg.plan{border-radius:8px;overflow:hidden}
.planbg{fill:transparent}
.ray{stroke:var(--mut);stroke-width:1;opacity:.55}
.ray.sub{stroke-dasharray:3 3;opacity:.38}
.cam{fill:var(--mut)}
.det{fill:var(--accent)}
.det.sub{fill:var(--sub)}
.derr{fill:var(--accent);opacity:.10;stroke:var(--accent);stroke-opacity:.35;
stroke-width:.7}
.serr{fill:none;stroke:var(--fg);stroke-width:1.3;stroke-dasharray:4 3}
.site{fill:var(--fg)}
.gt{stroke:var(--true);stroke-width:2;fill:none}
.nbr{fill:none;stroke:var(--nbr);stroke-width:1.8;stroke-dasharray:3 2}
.nbrlab{fill:var(--nbr);font-size:9.5px}
.bar{stroke:var(--mut);stroke-width:1.5}
.barlab{fill:var(--mut);font-size:10px}
.legend{color:var(--mut);font-size:12px;padding:0 24px 4px}
.nbrs{grid-column:1/-1;border-top:1px dashed var(--line);padding-top:10px;
margin-top:2px}
.nbrs h4{margin:0 0 7px;font-size:12px;font-weight:650;color:var(--nbr);
text-transform:uppercase;letter-spacing:.04em}
.nbrs .views figure{width:120px}
.nbrs .views figure img,.nbrs .views .miss{width:120px;height:120px}
"""

JS = """
const btns=[...document.querySelectorAll('.filters button')];
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(o=>o.setAttribute('aria-pressed',o===b));
  const f=b.dataset.filter;
  document.querySelectorAll('article').forEach(a=>{
    a.classList.toggle('hidden', f!=='all' && a.dataset.verdict!==f);
  });
});
"""


def render(city, sites, op_verdicts, gt_by_site, missing, crops_dir, fov_deg,
           px, source, params, stats, cap, inline=False, neighbors=None):
    def src_for(name):
        """Relative path, or a data: URI when the page must stand alone."""
        if not inline:
            return 'crops/' + name
        blob = base64.b64encode((crops_dir / name).read_bytes()).decode()
        return 'data:image/jpeg;base64,' + blob

    def figure(det, in_refit, site_id, lead=None):
        name = crop_name(det, fov_deg, px)
        vd = op_verdicts.get((det.pano_id, det.det_index))
        vtxt = ''
        if vd is True:
            vtxt = ' <b class="badge true">GT true</b>'
        elif vd is False:
            vtxt = ' <b class="badge false">GT false</b>'
        elif vd is not None:
            vtxt = ' <b class="badge unsure">GT {}</b>'.format(
                html.escape(str(vd)))
        url = PANO_URL.get(source, '').format(id=det.pano_id)
        pid_html = html.escape(det.pano_id[:14])
        if url:
            pid_html = '<a href="{}" target="_blank" rel="noopener">{}</a>' \
                .format(html.escape(url), pid_html)
        if name in missing:
            img = '<div class="miss">not in the archive<br>(decayed or ' \
                  'un-archived)</div>'
        else:
            img = '<img loading="lazy" src="{}" alt="view of site {} from {}">' \
                .format(src_for(name), site_id, html.escape(det.pano_id))
        head = lead or '<span class="c">{:.1f} m</span>'.format(
            det.ground.range_m)
        return ('<figure class="{}">{}<figcaption>{} · conf {:.2f}{}<br>{}{}'
                '</figcaption></figure>').format(
                    '' if in_refit else 'subth', img, head, det.conf, vtxt,
                    pid_html, ' · ' + html.escape(str(det.capture_date))
                    if det.capture_date else '')

    counts = {'true': 0, 'false': 0, 'unsure': 0, 'none': 0}
    cards = []
    for site in sites:
        verdicts = site_verdicts(site, op_verdicts)
        lab = gt_label(verdicts) or 'none'
        counts[lab] += 1
        members, hidden = shown_views(site, cap)
        figs = [figure(det, in_refit, site.id) for det, in_refit in members]
        resid = site.residual_per_dof()
        span = (site.month_max - site.month_min) \
            if site.month_min is not None else None
        meta = ['{} views'.format(len(site.pano_ids)),
                'best conf {:.2f}'.format(
                    max(d.conf for d, _ in site.members)),
                'resid {}'.format('n/a' if resid is None
                                  else '{:.2f}/dof'.format(resid))]
        if span is not None:
            meta.append('vintage span {} mo'.format(span))
        nsub = sum(1 for d, r in site.members if not r)
        if nsub:
            meta.append('{} sub-threshold'.format(nsub))
        if hidden:
            meta.append('{} farther views not shown'.format(hidden))
        near = neighbors.get(site.id, []) if neighbors else []
        if near:
            meta.append('{} nearby site{} (closest {:.2f} m)'.format(
                len(near), '' if len(near) == 1 else 's', near[0][0]))
        badge = '' if lab == 'none' else \
            ' <b class="badge {0}">{0}</b>'.format(lab)

        nbr_html = ''
        if near:
            nfigs = []
            for dist, other in near:
                det, in_refit = best_view(other)
                lead = '<span class="c">{:.1f} m away</span> · site {} · ' \
                       '{} views'.format(dist, other.id, len(other.pano_ids))
                nfigs.append(figure(det, in_refit, other.id, lead=lead))
            nbr_html = ('<div class="nbrs"><h4>nearby sites — same ramp split '
                        'in two, or a real second ramp?</h4>'
                        '<div class="views">{}</div></div>').format(
                            ''.join(nfigs))
        cards.append(
            '<article data-verdict="{}"><div class="hd"><span class="sid">'
            'site {}</span>{}<span class="m">{}</span></div>'
            '<div class="views">{}</div>{}{}</article>'.format(
                lab, site.id, badge, html.escape(' · '.join(meta)),
                ''.join(figs),
                plan_svg(site, gt_by_site.get(site.id, []), near), nbr_html))

    filters = ''.join(
        '<button data-filter="{}" aria-pressed="{}">{} ({})</button>'.format(
            key, 'true' if key == 'all' else 'false', label,
            len(sites) if key == 'all' else counts.get(key, 0))
        for key, label in [('all', 'all'), ('true', 'GT true'),
                           ('false', 'GT false'), ('unsure', 'GT unsure'),
                           ('none', 'no GT')])
    subtitle = (
        '{} sites · {} · fused from {} panos into {} sites '
        '({} operational) · crops at {:g}° FOV · match gate χ²={} , '
        'max range {:g} m'.format(
            len(sites), html.escape(source), stats['n_panos'],
            stats['n_sites'], stats['n_operational_sites'],
            fov_deg, params.gate_chi2, params.max_range_m))
    legend = ('plan view: ▲ camera, thin line = ray, shaded ellipse = that '
              'ray’s 1σ, ● = per-view ground point, ◆ dashed = the fused site’s '
              '1σ, ✚ = RampNet ground-truth point, dashed violet ring = another '
              'operational site nearby. Dashed rays and dimmed crops are '
              'sub-threshold members (support only — they never move the '
              'position). A nearby site is either this ramp fragmented or a '
              'genuine second ramp; the crops are what tell them apart.')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>{city} · multi-view sites</title><style>{css}</style></head>'
        '<body><header class="top"><h1>{city} — fused multi-view sites</h1>'
        '<div class="sub">{sub}</div></header>'
        '<div class="filters">{filters}</div>'
        '<div class="legend">{legend}</div><main>{cards}</main>'
        '<script>{js}</script></body></html>'
    ).format(city=html.escape(city), css=CSS, sub=subtitle, filters=filters,
             legend=legend, cards=''.join(cards), js=JS)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('city')
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--benchmark-root', type=Path,
                    default=REPO_ROOT.parent / 'RampNet' / 'benchmark')
    ap.add_argument('--select', default='gt',
                    choices=['gt', 'tp', 'fp', 'unsure', 'multiview',
                             'spread', 'fragment', 'random'],
                    help='which sites to show (default: those a judged GT '
                         'pano saw; "fragment" = over-split suspects)')
    ap.add_argument('--neighbor-m', type=float, default=8.0,
                    help='show other operational sites within this radius '
                         '(0 disables the neighbour panel)')
    ap.add_argument('--fragment-m', type=float, default=3.0,
                    help='--select fragment: nearest-neighbour cutoff')
    ap.add_argument('-n', '--n', type=int, default=40)
    ap.add_argument('--site-ids', type=int, nargs='*',
                    help='explicit site ids, overrides --select')
    ap.add_argument('--min-views', type=int, default=1)
    ap.add_argument('--max-views', type=int, default=12,
                    help='crops rendered per site, nearest first (default 12)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fov-deg', type=float, default=40.0)
    ap.add_argument('--crop-px', type=int, default=384)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--inline', action='store_true',
                    help='embed the crops as data: URIs -> one shareable file')
    ap.add_argument('--no-fetch', action='store_true',
                    help='render from the local crop cache only')
    ap.add_argument('--local-panos', type=Path, default=None,
                    help='cut crops locally from this pano dir instead of '
                         'going to makelab2')
    ap.add_argument('--ssh-helper', default=DEFAULT_HELPER)
    ap.add_argument('--host', default='makelab2')
    ap.add_argument('--remote-root', default=DEFAULT_REMOTE_ROOT)
    ap.add_argument('--remote-home', default=DEFAULT_REMOTE_HOME)
    ap.add_argument('--timeout', type=int, default=1800)
    args = ap.parse_args()

    run_dir = args.run_dir or REPO_ROOT / 'runs' / args.city
    out_dir = args.out or run_dir / 'explorer'
    crops_dir = out_dir / 'crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    print('fusing {} ...'.format(args.city))
    run_panos, _ = fs.load_results(run_dir / 'results.jsonl')
    params = fs.FuseParams()
    sites, frame, stats = fs.fuse(run_panos, params)
    source = run_panos[0].source if run_panos else 'gsv'
    # Records store either a source name or, for legacy GSV runs, streetlevel's raw
    # source string ("launch", "scout", ...) — so match the known names and let anything
    # else fall back to GSV, the way send_to_ps.transform_pano reads the same field.
    source = next((name for name in PANO_URL if name in source.lower()), 'gsv')

    op_verdicts, gt_points = {}, []
    bench = args.benchmark_root / args.city
    if (bench / 'verdicts.json').exists():
        # inline rather than es.load_city_files: that would re-read results.jsonl
        with open(bench / 'verdicts.json', encoding='utf-8') as f:
            verdict_panos = json.load(f)['panos']
        bundle_ops = {}
        with open(bench / 'records.jsonl', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    bundle_ops[rec['pano']['panorama_id']] = [
                        (d['x_normalized'], d['y_normalized'], d['confidence'])
                        for d in rec['detections']]
        pts, op_verdicts, counts, warnings = es.build_gt(
            verdict_panos, bundle_ops, {p.pano_id: p for p in run_panos},
            params, frame)
        gt_points = [(p.e, p.n) for p in pts]
        print('  GT: {} judged panos, {} placeable points, {} verdicts'.format(
            counts['judged'], counts['placeable'], len(op_verdicts)))
        for w in warnings[:3]:
            print('  warn: ' + w)
    else:
        print('  no RampNet verdicts for this city — rendering without GT')

    nbrs = neighbors_within([s for s in sites if s.n_operational > 0],
                            max(args.neighbor_m,
                                args.fragment_m if args.select == 'fragment'
                                else 0.0))
    chosen = select_sites(sites, op_verdicts, args.select, args.n,
                          args.min_views, args.seed, args.site_ids,
                          nbrs, args.fragment_m)
    if not chosen:
        raise SystemExit('no sites matched --select {}'.format(args.select))
    near_of = {s.id: [(d, o) for d, o in nbrs.get(s.id, ())
                      if d <= args.neighbor_m] for s in chosen}
    print('  selected {} sites ({} views, {} neighbouring sites)'.format(
        len(chosen), sum(len(s.members) for s in chosen),
        sum(len(v) for v in near_of.values())))

    gt_by_site = {}
    for site in chosen:
        near = [(e, n) for e, n in gt_points
                if math.hypot(e - site.e, n - site.n) <= 12.0]
        if near:
            gt_by_site[site.id] = near

    job = build_job(chosen, crops_dir, args.fov_deg, args.crop_px,
                    args.workers,
                    '{}/runs/{}/panos'.format(args.remote_root, args.city),
                    args.max_views, near_of)
    missing = set()
    if job['items'] and not args.no_fetch:
        if args.local_panos:
            missing = fetch_crops_local(job, crops_dir, args.local_panos)
        else:
            missing = fetch_crops_remote(
                job, crops_dir, args.ssh_helper, args.host,
                args.remote_root, args.remote_home, args.timeout)
    elif job['items']:
        print('  {} crops absent from the cache (--no-fetch)'.format(
            len(job['items'])))
    for site in chosen:
        dets = [d for d, _ in shown_views(site, args.max_views)[0]]
        dets += [best_view(o)[0] for _, o in near_of.get(site.id, ())]
        for det in dets:
            name = crop_name(det, args.fov_deg, args.crop_px)
            if not (crops_dir / name).exists():
                missing.add(name)
    if missing:
        print('  {} views without imagery'.format(len(missing)))

    page = render(args.city, chosen, op_verdicts, gt_by_site, missing,
                  crops_dir, args.fov_deg, args.crop_px, source, params, stats,
                  args.max_views, inline=args.inline, neighbors=near_of)
    index = out_dir / ('standalone.html' if args.inline else 'index.html')
    index.write_text(page, encoding='utf-8')
    print('wrote {} ({:.1f} MB)'.format(
        index, index.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
