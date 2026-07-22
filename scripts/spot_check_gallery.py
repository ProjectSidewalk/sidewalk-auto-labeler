"""
Render a single-pano gallery viewer for visual QA *and* quick validation.

Reads a run directory produced by main.py (or a bare results JSONL file), samples
panoramas, re-downloads each at up to 4096x2048, and writes a static one-pano-at-a-time
viewer (index.html). The saved images are deliberately clean — detections are drawn as
HTML overlays so the viewer can recolor them live as verdicts change. Serve it however you like (VS Code
Live Server, `python -m http.server`, or just open the file).

The viewer doubles as a validation tool:
  - click a detection crop (or its numbered circle on the panorama) to cycle its
    verdict: unjudged -> correct -> incorrect -> unsure; circles start yellow and
    turn green/red/blue to match. "unsure" abstains — the scorer drops it from
    both precision and recall rather than forcing a guess on ambiguous imagery
  - click anywhere on the panorama to mark a curb ramp the model missed (click the
    magenta marker to downgrade it to "unsure" amber, again to remove it), or press
    "m" / the "No missed ramps" button to confirm there are none — a pano only counts
    as reviewed once every crop is judged AND the missed-ramp check is confirmed, so
    recall isn't inflated by panos nobody actually scanned. Unsure missed marks also
    abstain: they don't count toward the recall denominator
  - verdicts autosave to the browser's localStorage; "Export verdicts" downloads a
    verdicts.json to score with scripts/score_validation.py (precision + recall)

The sample always includes the densest panos (flagged so scoring can exclude them
from unbiased estimates), a random sample of panos with detections, and a few
zero-detection panos (for false negatives / recall).

Usage:
    python scripts/spot_check_gallery.py runs/bend
    python scripts/spot_check_gallery.py runs/bend --sample 100 --empty-sample 10
"""
import argparse
import json
import math
import random
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from streetlevel import streetview
from tqdm import tqdm

# Repo root on the path (for sources/) + local secrets: Mapillary-run galleries
# re-download imagery through the Graph API, which needs MAPILLARY_ACCESS_TOKEN.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

socket.setdefaulttimeout(30)

DISPLAY_WIDTH = 1600   # displayed full-pano width in the gallery
CROP_SIZE = 512        # per-detection close-up, taken at download resolution
TOP_N_BY_COUNT = 5     # the densest panos are always included (group "top")
DOWNLOAD_WORKERS = 8

# Two validation panos closer than this almost certainly show the same physical
# curb ramps, so the sampler keeps selected panos at least this far apart: a
# reviewer never judges the same ramps twice, and precision/recall aren't inflated
# by correlated duplicates. Tunable per city/imagery via --min-spacing; deliberately
# well above the Mapillary thinning spacing (~5 m) since that's coverage, not sampling.
DEFAULT_MIN_SPACING_M = 30


def load_records(path_arg):
    path = Path(path_arg)
    jsonl_path = path / "results.jsonl" if path.is_dir() else path
    if not jsonl_path.exists():
        sys.exit(f"No results file found at {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Verdicts are keyed by the run's area hash when available, so different runs
    # (and re-generated galleries) don't clobber each other's localStorage.
    run_key = str(jsonl_path)
    manifest_path = jsonl_path.parent / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            run_key = json.load(f).get('area_hash', run_key)
    return jsonl_path, records, run_key


def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _coords(record):
    """(lat, lng) for a record, or None if it carries no position."""
    p = record['pano']
    lat, lng = p.get('lat'), p.get('lng')
    return (lat, lng) if lat is not None and lng is not None else None


class _SpatialIndex:
    """Grid of accepted points for fast 'is anything within min_spacing?' checks.

    Cell size == min_spacing, so any point within range lies in one of the nine
    neighbouring cells — the acceptance test touches a handful of points, not the
    whole accepted set, so selection stays cheap on city-sized candidate pools.
    """

    def __init__(self, min_spacing):
        self.s = max(min_spacing, 1e-9)
        self.cells = {}

    def _key(self, lat, lng):
        clat = self.s / 111320.0
        clng = self.s / (111320.0 * max(0.01, math.cos(math.radians(lat))))
        return (math.floor(lat / clat), math.floor(lng / clng))

    def far_enough(self, lat, lng):
        kx, ky = self._key(lat, lng)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for plat, plng in self.cells.get((kx + dx, ky + dy), ()):
                    if _haversine_m(lat, lng, plat, plng) < self.s:
                        return False
        return True

    def add(self, lat, lng):
        self.cells.setdefault(self._key(lat, lng), []).append((lat, lng))


def _spread(candidates, k, index, min_spacing):
    """Greedily take up to k candidates (in the given order) that sit at least
    min_spacing metres from every already-accepted point in the shared `index`.

    Records without coordinates are accepted without constraint (graceful
    degradation — a coordinate-less run can't be de-clustered, but must still
    render). Returns (picked, rejected_for_spacing)."""
    picked, rejected = [], 0
    for rec in candidates:
        if len(picked) >= k:
            break
        c = _coords(rec)
        if c is None:
            picked.append(rec)
        elif min_spacing <= 0 or index.far_enough(*c):
            picked.append(rec)
            index.add(*c)
        else:
            rejected += 1
    return picked, rejected


def choose_panos(records, sample, empty_sample, seed, min_spacing=DEFAULT_MIN_SPACING_M):
    """Return [(record, group)] for a spatially de-clustered validation sample.

    Three strata, all held at least `min_spacing` metres apart — *across* strata as
    well as within — so a reviewer never judges the same physical ramps twice and
    the precision/recall estimate isn't inflated by correlated duplicates:
      - 'top':    the densest *distinct* intersections (up to TOP_N_BY_COUNT) — a
                  dense-scene stress test, excluded from unbiased scoring.
      - 'random': a spatially spread sample of panos with detections.
      - 'empty':  a spatially spread sample of zero-detection panos (for recall).

    Deterministic given `seed`. `min_spacing=0` disables spacing (the original pure
    random behaviour). Records without lat/lng are included unconstrained. The method
    is city-agnostic by design — it's meant to seed every city's ground-truth set.
    """
    rng = random.Random(seed)
    with_det = [r for r in records if r['detections']]
    without_det = [r for r in records if not r['detections']]
    index = _SpatialIndex(min_spacing)

    # top: densest first, but only one pano per distinct location.
    by_density = sorted(with_det, key=lambda r: len(r['detections']), reverse=True)
    top, _ = _spread(by_density, TOP_N_BY_COUNT, index, min_spacing)
    top_ids = {r['pano']['panorama_id'] for r in top}

    # random: spread across the remaining detection panos.
    rest = [r for r in with_det if r['pano']['panorama_id'] not in top_ids]
    rng.shuffle(rest)
    n_random = max(0, sample - len(top))
    random_sel, rej_r = _spread(rest, n_random, index, min_spacing)

    # empty: spread across the zero-detection panos.
    empties = list(without_det)
    rng.shuffle(empties)
    empty_sel, rej_e = _spread(empties, empty_sample, index, min_spacing)

    # No silent caps: if spacing (not scarcity) kept us short of a target, say so.
    if min_spacing > 0 and rej_r and len(random_sel) < n_random:
        print(f"  spatial sampling: kept {len(random_sel)}/{n_random} detection panos "
              f"at >= {min_spacing} m spacing (area too dense for more).")
    if min_spacing > 0 and rej_e and len(empty_sel) < empty_sample:
        print(f"  spatial sampling: kept {len(empty_sel)}/{empty_sample} empty panos "
              f"at >= {min_spacing} m spacing.")

    return ([(r, 'top') for r in top]
            + [(r, 'random') for r in random_sel]
            + [(r, 'empty') for r in empty_sel])


def fetch_display_image(pano):
    """Re-downloads the pano through the provider it came from."""
    if pano.get('source') == 'mapillary':
        from sources import mapillary
        img = mapillary.fetch_image(pano['panorama_id'])
        if img is None:
            raise RuntimeError("Mapillary metadata/image unavailable")
        return img
    metadata = streetview.find_panorama_by_id(pano['panorama_id'])
    if metadata is None:
        raise RuntimeError("metadata unavailable")
    return streetview.get_panorama(metadata, zoom=min(3, len(metadata.image_sizes) - 1))


def render_pano(record, group, images_dir):
    """Downloads one pano, saves the clean display image and per-detection crops,
    and returns the overlay geometry the viewer draws the detection circles from."""
    pano = record['pano']
    pid = pano['panorama_id']
    img = fetch_display_image(pano)

    # Detections are stored normalized; scale to this download's resolution.
    # Images are saved clean — detection circles are drawn as HTML overlays so the
    # viewer can recolor them live as verdicts change.
    sx, sy = img.size[0], img.size[1]

    crops = []
    for i, det in enumerate(record['detections']):
        px, py = det['x_normalized'] * sx, det['y_normalized'] * sy
        half = CROP_SIZE // 2
        left = int(min(max(px - half, 0), img.size[0] - CROP_SIZE))
        top = int(min(max(py - half, 0), img.size[1] - CROP_SIZE))
        crop = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        crop_name = f"{pid}_det{i}.jpg"
        crop.convert('RGB').save(images_dir / crop_name, quality=85)
        crops.append({'img': crop_name, 'conf': round(det['confidence'], 4),
                      # detection center: normalized in the pano, and in the crop
                      'x': round(det['x_normalized'], 5),
                      'y': round(det['y_normalized'], 5),
                      'cx': round((px - left) / CROP_SIZE, 4),
                      'cy': round((py - top) / CROP_SIZE, 4)})

    full_name = f"{pid}_full.jpg"
    img.resize((DISPLAY_WIDTH, DISPLAY_WIDTH // 2), Image.BILINEAR) \
       .convert('RGB').save(images_dir / full_name, quality=80)
    return {
        'pid': pid,
        'source': pano.get('source', ''),
        'date': str(pano['capture_date']),
        'group': group,
        'full': full_name,
        'crops': crops,
    }


HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Detection spot check</title>
<style>
  body{font-family:sans-serif;margin:20px auto;max-width:1650px;background:#fafafa;color:#222}
  a{color:#06c}
  .bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px}
  .bar button,.bar select{font-size:15px;padding:6px 14px;cursor:pointer}
  .meta{color:#666;font-size:13px}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#eee;color:#555}
  #panowrap{position:relative;line-height:0;cursor:crosshair}
  #panowrap img{width:100%;height:auto}
  .det{position:absolute;width:34px;height:34px;transform:translate(-50%,-50%);
       border-radius:50%;border:4px solid #ffd400;
       box-shadow:0 0 0 3px rgba(255,255,255,.85),0 0 6px #000;cursor:pointer}
  .det.ok{border-color:#1a9c3e}
  .det.bad{border-color:#d23}
  .det.unsure{border-color:#3a86ff}
  .det .num{position:absolute;top:-21px;left:50%;transform:translateX(-50%);
            font:bold 13px/1 sans-serif;color:#fff;text-shadow:0 0 3px #000,0 0 3px #000}
  .missed{position:absolute;width:36px;height:36px;transform:translate(-50%,-50%);
          border:4px dashed #e534eb;border-radius:50%;box-shadow:0 0 4px #000;
          cursor:pointer;background:rgba(229,52,235,.12)}
  .missed.unsure{border-color:#ff9f1c;background:rgba(255,159,28,.14)}
  #nomiss.active{background:#1a9c3e;border:1px solid #1a9c3e;color:#fff}
  .crops{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;line-height:0}
  .crops figure{margin:0;text-align:center;font-size:13px;line-height:1.3;cursor:pointer}
  /* verdict border lives on the wrapper so the ring's %-coords map exactly onto the image */
  .cropwrap{position:relative;display:inline-block;line-height:0;border:5px solid #bbb;border-radius:4px}
  .crops img{width:256px;height:256px;border-radius:2px;display:block}
  .crops .ok .cropwrap{border-color:#1a9c3e}
  .crops .bad .cropwrap{border-color:#d23}
  .crops .unsure .cropwrap{border-color:#3a86ff}
  /* ring diameter mirrors the old 40px-radius circle on the 512px crop (80/512) */
  .cropring{position:absolute;width:15.6%;height:15.6%;transform:translate(-50%,-50%);
            border:3px solid #ffd400;border-radius:50%;
            box-shadow:0 0 0 2px rgba(255,255,255,.85);pointer-events:none}
  .ok .cropring{border-color:#1a9c3e}.bad .cropring{border-color:#d23}
  .unsure .cropring{border-color:#3a86ff}
  .crops .verdict{font-weight:bold}
  .ok .verdict{color:#1a9c3e}.bad .verdict{color:#d23}.unsure .verdict{color:#3a86ff}
  #help{font-size:13px;color:#666;margin-top:14px}
  kbd{background:#eee;border:1px solid #ccc;border-radius:3px;padding:0 4px;font-size:12px}
</style>
<div class="bar">
  <button id="prev">&#8592; Prev</button>
  <button id="next">Next &#8594;</button>
  <select id="filter">
    <option value="all">All panos</option>
    <option value="det">With detections</option>
    <option value="empty">Zero detections</option>
    <option value="todo">Unreviewed</option>
  </select>
  <span id="pos" class="meta"></span>
  <span id="progress" class="meta"></span>
  <span style="flex:1"></span>
  <button id="export">Export verdicts</button>
</div>
<h2 id="title" style="margin:6px 0"></h2>
<div id="panowrap"><img id="panoimg" alt=""></div>
<div class="bar" id="fnbar" style="margin:8px 0 0">
  <button id="nomiss">No missed ramps</button>
  <span id="fnstate" class="meta"></span>
</div>
<div class="crops" id="crops"></div>
<p id="help">
  <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> pano &nbsp;&middot;&nbsp; <kbd>1</kbd>&#8211;<kbd>9</kbd> cycle a crop's
  verdict (unjudged &#8594; correct &#8594; incorrect &#8594; <span style="color:#3a86ff">unsure</span>) or click the
  crop / its numbered circle &nbsp;&middot;&nbsp; click the panorama to mark a <b>missed</b> curb ramp (click the
  marker to make it <span style="color:#ff9f1c">unsure</span>, again to remove), or press <kbd>m</kbd> if there
  are none &mdash; a pano counts as reviewed only once every crop is judged <b>and</b> the missed-ramp check is
  done &nbsp;&middot;&nbsp; <b>unsure</b> abstains (dropped from precision &amp; recall) &nbsp;&middot;&nbsp; verdicts autosave
  locally; Export downloads <span id="vname"></span> &mdash; save it into the run directory, then run
  <code>python scripts/score_validation.py runs/&lt;name&gt;</code>
</p>
<script>
const ENTRIES = __ENTRIES__;
const RUN_KEY = __RUN_KEY__;
const RUN_NAME = __RUN_NAME__;
const SOURCE = __SOURCE__;
const STORE = 'verdicts:' + RUN_KEY;

let verdicts = JSON.parse(localStorage.getItem(STORE) || '{}');
// verdicts[pid] = {dets: [null|true|false|'unsure', ...],
//                  missed: [{x, y, unsure?: bool}, ...],
//                  noMissed: bool (reviewer confirmed no missed ramps), seen: true}
// 'unsure' (crop) and unsure:true (missed) both mean "can't tell" — the scorer
// abstains on them (dropped from precision and from the recall pool).
function v(pid, n) {
  if (!verdicts[pid]) verdicts[pid] = {dets: Array(n).fill(null), missed: [], noMissed: false, seen: false};
  const s = verdicts[pid];
  // A rerun of the same area can change a pano's detection count (new model or
  // threshold); stored crop verdicts then no longer map to the current crops.
  // Reset them (missed marks and the no-missed affirmation are pano-level and
  // stay valid) rather than mis-align or over-count reviewed().
  if (s.dets.length !== n) s.dets = Array(n).fill(null);
  return s;
}
function save() { localStorage.setItem(STORE, JSON.stringify(verdicts)); }

let filterMode = 'all', view = ENTRIES.slice(), idx = 0;

// vcls/fnChecked/reviewed define what "reviewed" means; scripts/score_validation.py
// collect() applies the same gate to the exported verdicts — keep the two in sync.
function vcls(d) { return d === true ? 'ok' : d === false ? 'bad' : d === 'unsure' ? 'unsure' : ''; }
function fnChecked(s) { return s.missed.length > 0 || !!s.noMissed; }
function reviewed(e) {
  const s = verdicts[e.pid];
  if (!s || !s.seen || !fnChecked(s)) return false;
  if (s.dets.length !== e.crops.length) return false; // stale entry, see v()
  return s.dets.every(d => d !== null);
}
function applyFilter() {
  const cur = view[idx] && view[idx].pid;
  view = ENTRIES.filter(e =>
    filterMode === 'det' ? e.crops.length > 0 :
    filterMode === 'empty' ? e.crops.length === 0 :
    filterMode === 'todo' ? !reviewed(e) : true);
  if (!view.length) { idx = 0; render(); return; }
  const keep = view.findIndex(e => e.pid === cur);
  idx = keep >= 0 ? keep : 0;
  render();
}

function render() {
  const pos = document.getElementById('pos');
  const done = ENTRIES.filter(reviewed).length;
  document.getElementById('progress').textContent = done + '/' + ENTRIES.length + ' reviewed';
  // Overlays belong to the previously rendered pano; clear them on every path
  // (including the empty-filter one, where stale markers would still be clickable).
  document.querySelectorAll('.missed, .det').forEach(m => m.remove());
  if (!view.length) {
    document.getElementById('title').textContent = 'No panos match this filter';
    document.getElementById('panoimg').removeAttribute('src');
    document.getElementById('crops').innerHTML = '';
    document.getElementById('fnbar').style.display = 'none';
    pos.textContent = '';
    return;
  }
  document.getElementById('fnbar').style.display = '';
  const e = view[idx];
  const s = v(e.pid, e.crops.length);
  s.seen = true; save();

  pos.textContent = (idx + 1) + ' / ' + view.length;
  const viewerUrl = e.source === 'mapillary'
    ? 'https://www.mapillary.com/app/?pKey=' + e.pid + '&focus=photo'
    : 'https://www.google.com/maps/@?api=1&map_action=pano&pano=' + e.pid;
  document.getElementById('title').innerHTML =
    '<a href="' + viewerUrl + '" target="_blank">' +
    e.pid + '</a> <span class="meta">captured ' + e.date + ' &mdash; ' + e.crops.length +
    ' detection(s)</span> <span class="badge">' + e.group + '</span>';
  document.getElementById('panoimg').src = 'images/' + e.full;

  // Detection circles + missed-ramp markers, overlaid on the clean pano image.
  const wrap = document.getElementById('panowrap');
  e.crops.forEach((c, i) => {
    const d = document.createElement('div');
    d.className = ('det ' + vcls(s.dets[i])).trim();
    d.style.left = (c.x * 100) + '%';
    d.style.top = (c.y * 100) + '%';
    d.innerHTML = '<span class="num">' + (i + 1) + '</span>';
    d.title = 'detection ' + (i + 1) + ' — click to cycle verdict';
    d.onclick = ev => { ev.stopPropagation(); cycle(i); };
    wrap.appendChild(d);
  });
  s.missed.forEach((m, i) => {
    const d = document.createElement('div');
    d.className = 'missed' + (m.unsure ? ' unsure' : '');
    d.style.left = (m.x * 100) + '%';
    d.style.top = (m.y * 100) + '%';
    d.title = m.unsure ? 'unsure missed ramp — click to remove'
                       : 'missed ramp — click to mark unsure, again to remove';
    // Click cycles a marker confident -> unsure -> removed, mirroring the crop cycle.
    d.onclick = ev => { ev.stopPropagation();
      if (!m.unsure) m.unsure = true; else s.missed.splice(i, 1);
      save(); render();
    };
    wrap.appendChild(d);
  });

  // Missed-ramp confirmation state. The click handlers keep noMissed and missed
  // marks mutually exclusive; normalize once anyway so stale state can't disagree.
  const affirmed = !!s.noMissed && !s.missed.length;
  const nMissedSure = s.missed.filter(m => !m.unsure).length;
  const nMissedUnsure = s.missed.filter(m => m.unsure).length;
  const nm = document.getElementById('nomiss');
  nm.disabled = s.missed.length > 0;
  nm.classList.toggle('active', affirmed);
  nm.textContent = affirmed ? '✓ No missed ramps' : 'No missed ramps (m)';
  document.getElementById('fnstate').textContent =
    s.missed.length ? (nMissedSure + ' missed ramp(s) marked' +
      (nMissedUnsure ? ', ' + nMissedUnsure + ' unsure' : '')) :
    affirmed ? 'missed-ramp check done' :
    'scan the panorama for ramps the model missed, then click a ramp or confirm none';

  // Detection crops with verdict state.
  const crops = document.getElementById('crops');
  crops.innerHTML = '';
  e.crops.forEach((c, i) => {
    const fig = document.createElement('figure');
    fig.className = vcls(s.dets[i]);
    fig.innerHTML = '<span class="cropwrap"><img src="images/' + c.img + '" loading="lazy">' +
      '<span class="cropring" style="left:' + (c.cx * 100) + '%;top:' + (c.cy * 100) + '%"></span></span>' +
      '<figcaption>[' + (i + 1) + '] conf ' + c.conf.toFixed(2) +
      ' &mdash; <span class="verdict">' +
      (s.dets[i] === true ? 'correct' : s.dets[i] === false ? 'INCORRECT'
        : s.dets[i] === 'unsure' ? 'unsure' : 'unjudged') +
      '</span></figcaption>';
    fig.onclick = () => cycle(i);
    crops.appendChild(fig);
  });
}

function cycle(i) {
  const e = view[idx];
  if (!e || i >= e.crops.length) return;
  const s = v(e.pid, e.crops.length);
  s.dets[i] = s.dets[i] === null ? true
            : s.dets[i] === true ? false
            : s.dets[i] === false ? 'unsure'
            : null;
  save(); render();
}

document.getElementById('prev').onclick = () => { if (view.length) { idx = (idx - 1 + view.length) % view.length; render(); } };
document.getElementById('next').onclick = () => { if (view.length) { idx = (idx + 1) % view.length; render(); } };
document.getElementById('filter').onchange = ev => { filterMode = ev.target.value; applyFilter(); };
document.getElementById('panowrap').onclick = ev => {
  if (!view.length || ev.target.closest('.missed, .det')) return;
  const e = view[idx];
  const r = document.getElementById('panoimg').getBoundingClientRect();
  const s = v(e.pid, e.crops.length);
  s.missed.push({x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height});
  s.noMissed = false;
  save(); render();
};
document.getElementById('nomiss').onclick = () => {
  if (!view.length) return;
  const e = view[idx];
  const s = v(e.pid, e.crops.length);
  if (s.missed.length) return;
  s.noMissed = !s.noMissed;
  save(); render();
};
document.addEventListener('keydown', ev => {
  // Never hijack browser shortcuts (Ctrl+M mutes a tab, Cmd+M minimizes, Ctrl+1..9
  // switch tabs) — a modifier chord must not silently record a verdict.
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.key === 'ArrowLeft') document.getElementById('prev').click();
  else if (ev.key === 'ArrowRight') document.getElementById('next').click();
  else if (ev.key === 'm' || ev.key === 'M') document.getElementById('nomiss').click();
  else if (ev.key >= '1' && ev.key <= '9') cycle(Number(ev.key) - 1);
});
document.getElementById('export').onclick = () => {
  const out = {run_key: RUN_KEY, run_name: RUN_NAME, source: SOURCE,
               exported_at: new Date().toISOString(), panos: {}};
  for (const e of ENTRIES) {
    const s = verdicts[e.pid];
    if (!s || !s.seen) continue;
    out.panos[e.pid] = {group: e.group, dets: s.dets, missed: s.missed,
                        no_missed: !!s.noMissed};
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = RUN_NAME + '_verdicts.json';
  a.click();
};

document.getElementById('vname').textContent = RUN_NAME + '_verdicts.json';
render();
</script>
"""


def build_html(entries, jsonl_path, run_key, run_name):
    return (HTML_TEMPLATE
            .replace('__ENTRIES__', json.dumps(entries))
            .replace('__RUN_KEY__', json.dumps(run_key))
            .replace('__RUN_NAME__', json.dumps(run_name))
            .replace('__SOURCE__', json.dumps(str(jsonl_path))))


def main():
    parser = argparse.ArgumentParser(
        description="Render a single-pano gallery viewer for visual QA and validation."
    )
    parser.add_argument("run", help="Run directory from main.py (or a results JSONL file).")
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of panos with detections to include (default: %(default)s).")
    parser.add_argument("--empty-sample", type=int, default=10,
                        help="Number of zero-detection panos to include (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed, for a reproducible gallery (default: %(default)s).")
    parser.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_SPACING_M,
                        help="Minimum metres between sampled panos, so a reviewer never "
                             "judges the same ramps twice (default: %(default)s; 0 disables).")
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: <run dir>/spot_check).")
    args = parser.parse_args()

    jsonl_path, records, run_key = load_records(args.run)
    chosen = choose_panos(records, args.sample, args.empty_sample, args.seed, args.min_spacing)
    print(f"{len(records)} records; rendering {len(chosen)} panos "
          f"({sum(1 for r, _ in chosen if r['detections'])} with detections).")

    out_dir = args.out or jsonl_path.parent / "spot_check"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(render_pano, r, g, images_dir): r for r, g in chosen}
        with tqdm(total=len(futures), desc="Rendering panoramas") as pbar:
            for future in as_completed(futures):
                pid = futures[future]['pano']['panorama_id']
                try:
                    entries.append(future.result())
                except Exception as e:
                    print(f"  skipped {pid}: {e}")
                pbar.update(1)

    run_name = jsonl_path.parent.name if jsonl_path.parent.name else jsonl_path.stem
    entries.sort(key=lambda e: (-len(e['crops']), e['pid']))
    index_path = out_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_html(entries, jsonl_path, run_key, run_name))
    print(f"Gallery: {index_path}")
    print(f"After reviewing, save the exported {run_name}_verdicts.json into {jsonl_path.parent}")
    print(f"and score it with: python scripts/score_validation.py {jsonl_path.parent}")


if __name__ == "__main__":
    main()
