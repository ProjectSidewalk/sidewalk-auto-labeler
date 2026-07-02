"""
Render a single-pano gallery viewer for visual QA *and* quick validation.

Reads a run directory produced by main.py (or a bare results JSONL file), samples
panoramas, re-downloads each at up to 4096x2048, draws the detections, and writes a
static one-pano-at-a-time viewer (index.html). Serve it however you like (VS Code
Live Server, `python -m http.server`, or just open the file).

The viewer doubles as a validation tool:
  - click a detection crop to cycle its verdict: unjudged -> correct -> incorrect
  - click anywhere on the panorama to mark a curb ramp the model missed
    (click a marker again to remove it)
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
import random
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw
from streetlevel import streetview
from tqdm import tqdm

socket.setdefaulttimeout(30)

DISPLAY_WIDTH = 1600   # displayed full-pano width in the gallery
CROP_SIZE = 512        # per-detection close-up, taken at download resolution
TOP_N_BY_COUNT = 5     # the densest panos are always included (group "top")
DOWNLOAD_WORKERS = 8


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


def choose_panos(records, sample, empty_sample, seed):
    """Returns a list of (record, group) with group in {"top", "random", "empty"}."""
    rng = random.Random(seed)
    with_det = [r for r in records if r['detections']]
    without_det = [r for r in records if not r['detections']]

    top = sorted(with_det, key=lambda r: len(r['detections']), reverse=True)[:TOP_N_BY_COUNT]
    top_ids = {r['pano']['panorama_id'] for r in top}
    rest = [r for r in with_det if r['pano']['panorama_id'] not in top_ids]

    chosen = [(r, 'top') for r in top]
    if sample > len(chosen) and rest:
        chosen += [(r, 'random') for r in rng.sample(rest, min(sample - len(chosen), len(rest)))]
    if without_det and empty_sample:
        chosen += [(r, 'empty') for r in rng.sample(without_det, min(empty_sample, len(without_det)))]
    return chosen


def render_pano(record, group, images_dir):
    """Downloads one pano, draws its detections, and saves the gallery images."""
    pano = record['pano']
    pid = pano['panorama_id']
    metadata = streetview.find_panorama_by_id(pid)
    if metadata is None:
        raise RuntimeError("metadata unavailable")
    img = streetview.get_panorama(metadata, zoom=min(3, len(metadata.image_sizes) - 1))

    # Detections are stored normalized; scale to this download's resolution.
    sx, sy = img.size[0], img.size[1]

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    crops = []
    for i, det in enumerate(record['detections']):
        px, py = det['x_normalized'] * sx, det['y_normalized'] * sy
        radius = 40
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     outline=(255, 0, 0), width=8)

        half = CROP_SIZE // 2
        left = int(min(max(px - half, 0), img.size[0] - CROP_SIZE))
        top = int(min(max(py - half, 0), img.size[1] - CROP_SIZE))
        crop = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        crop_draw = ImageDraw.Draw(crop)
        crop_draw.ellipse([px - left - radius, py - top - radius,
                           px - left + radius, py - top + radius],
                          outline=(255, 0, 0), width=6)
        crop_name = f"{pid}_det{i}.jpg"
        crop.convert('RGB').save(images_dir / crop_name, quality=85)
        crops.append({'img': crop_name, 'conf': round(det['confidence'], 4)})

    full_name = f"{pid}_full.jpg"
    annotated.resize((DISPLAY_WIDTH, DISPLAY_WIDTH // 2), Image.BILINEAR) \
             .convert('RGB').save(images_dir / full_name, quality=80)
    return {
        'pid': pid,
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
  .missed{position:absolute;width:36px;height:36px;margin:-18px 0 0 -18px;border:4px solid #ffd400;
          border-radius:50%;box-shadow:0 0 4px #000;cursor:pointer}
  .crops{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;line-height:0}
  .crops figure{margin:0;text-align:center;font-size:13px;line-height:1.3;cursor:pointer}
  .crops img{width:256px;height:256px;border:5px solid #bbb;border-radius:4px}
  .crops .ok img{border-color:#1a9c3e}
  .crops .bad img{border-color:#d23}
  .crops .verdict{font-weight:bold}
  .ok .verdict{color:#1a9c3e}.bad .verdict{color:#d23}
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
<div class="crops" id="crops"></div>
<p id="help">
  <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> pano &nbsp;&middot;&nbsp; <kbd>1</kbd>&#8211;<kbd>9</kbd> cycle a crop's
  verdict (unjudged &#8594; correct &#8594; incorrect) or click the crop &nbsp;&middot;&nbsp; click the panorama
  to mark a <b>missed</b> curb ramp (click the yellow marker to remove) &nbsp;&middot;&nbsp; verdicts autosave
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
// verdicts[pid] = {dets: [null|true|false, ...], missed: [{x, y}, ...], seen: true}
function v(pid, n) {
  if (!verdicts[pid]) verdicts[pid] = {dets: Array(n).fill(null), missed: [], seen: false};
  return verdicts[pid];
}
function save() { localStorage.setItem(STORE, JSON.stringify(verdicts)); }

let filterMode = 'all', view = ENTRIES.slice(), idx = 0;

function reviewed(e) {
  const s = verdicts[e.pid];
  if (!s || !s.seen) return false;
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
  if (!view.length) {
    document.getElementById('title').textContent = 'No panos match this filter';
    document.getElementById('panoimg').removeAttribute('src');
    document.getElementById('crops').innerHTML = '';
    pos.textContent = '';
    return;
  }
  const e = view[idx];
  const s = v(e.pid, e.crops.length);
  s.seen = true; save();

  pos.textContent = (idx + 1) + ' / ' + view.length;
  document.getElementById('title').innerHTML =
    '<a href="https://www.google.com/maps/@?api=1&map_action=pano&pano=' + e.pid + '" target="_blank">' +
    e.pid + '</a> <span class="meta">captured ' + e.date + ' &mdash; ' + e.crops.length +
    ' detection(s)</span> <span class="badge">' + e.group + '</span>';
  document.getElementById('panoimg').src = 'images/' + e.full;

  // Missed-ramp markers.
  document.querySelectorAll('.missed').forEach(m => m.remove());
  const wrap = document.getElementById('panowrap');
  s.missed.forEach((m, i) => {
    const d = document.createElement('div');
    d.className = 'missed';
    d.style.left = (m.x * 100) + '%';
    d.style.top = (m.y * 100) + '%';
    d.title = 'missed ramp — click to remove';
    d.onclick = ev => { ev.stopPropagation(); s.missed.splice(i, 1); save(); render(); };
    wrap.appendChild(d);
  });

  // Detection crops with verdict state.
  const crops = document.getElementById('crops');
  crops.innerHTML = '';
  e.crops.forEach((c, i) => {
    const fig = document.createElement('figure');
    fig.className = s.dets[i] === true ? 'ok' : s.dets[i] === false ? 'bad' : '';
    fig.innerHTML = '<img src="images/' + c.img + '" loading="lazy">' +
      '<figcaption>[' + (i + 1) + '] conf ' + c.conf.toFixed(2) +
      ' &mdash; <span class="verdict">' +
      (s.dets[i] === true ? 'correct' : s.dets[i] === false ? 'INCORRECT' : 'unjudged') +
      '</span></figcaption>';
    fig.onclick = () => cycle(i);
    crops.appendChild(fig);
  });
}

function cycle(i) {
  const e = view[idx];
  if (!e || i >= e.crops.length) return;
  const s = v(e.pid, e.crops.length);
  s.dets[i] = s.dets[i] === null ? true : s.dets[i] === true ? false : null;
  save(); render();
}

document.getElementById('prev').onclick = () => { if (view.length) { idx = (idx - 1 + view.length) % view.length; render(); } };
document.getElementById('next').onclick = () => { if (view.length) { idx = (idx + 1) % view.length; render(); } };
document.getElementById('filter').onchange = ev => { filterMode = ev.target.value; applyFilter(); };
document.getElementById('panowrap').onclick = ev => {
  if (!view.length || ev.target.classList.contains('missed')) return;
  const e = view[idx];
  const r = document.getElementById('panoimg').getBoundingClientRect();
  const s = v(e.pid, e.crops.length);
  s.missed.push({x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height});
  save(); render();
};
document.addEventListener('keydown', ev => {
  if (ev.key === 'ArrowLeft') document.getElementById('prev').click();
  else if (ev.key === 'ArrowRight') document.getElementById('next').click();
  else if (ev.key >= '1' && ev.key <= '9') cycle(Number(ev.key) - 1);
});
document.getElementById('export').onclick = () => {
  const out = {run_key: RUN_KEY, run_name: RUN_NAME, source: SOURCE,
               exported_at: new Date().toISOString(), panos: {}};
  for (const e of ENTRIES) {
    const s = verdicts[e.pid];
    if (!s || !s.seen) continue;
    out.panos[e.pid] = {group: e.group, dets: s.dets, missed: s.missed};
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
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: <run dir>/spot_check).")
    args = parser.parse_args()

    jsonl_path, records, run_key = load_records(args.run)
    chosen = choose_panos(records, args.sample, args.empty_sample, args.seed)
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
