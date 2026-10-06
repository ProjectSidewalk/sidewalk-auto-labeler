"""The review page of scripts/corner_gallery.py (RampNet#243), kept apart so the Python stays
readable. Vanilla JS, no framework, no network: the page reads only the bundle's aerials
and crops by relative path.

Mechanics follow RampNet's #224 cluster-review gallery (scripts/cluster_review_gallery.py,
branch cluster-review-224): browser storage keyed by rater + item-list sha256, the same
prefill rule (STATE_BOOTSTRAP_JS below), the same timing rule, review notes, Export to a
per-rater file, and the city inventory revealed only after a unit is complete. What differs
is the task: each corner gets one verdict (present / absent / can't tell) instead of a
label -> ramp assignment, and the first completed verdicts are frozen as ``blind``.

Accessibility: every verdict is a native radio button inside a fieldset whose legend names
the corner; crops are buttons with alt text (capture date, distance, pano); status is
announced through an aria-live region; nothing is encoded by colour alone (the aerial's
corner markers carry their number, the cards carry the verdict in words).
"""

# Kept out of the template as a standalone pure function so tests can run it under node
# (tests/test_corner_gallery.py), as RampNet's cluster_review_gallery.STATE_BOOTSTRAP_JS is.
# It is the one piece of page JS that can destroy rating work. Its rules:
#   * a prefill made on another item list (items sha256) is ignored whole;
#   * local state wins per UNIT only when the rater has worked on it in this browser (seen
#     or complete); a unit the browser merely initialised takes the file's state, and every
#     unit where local work shadows a differing file entry is reported;
#   * every rendered unit is reconciled against its current corner keys -- a key that left
#     is dropped, a new one is added unrated, and either reopens a complete unit;
#   * state for units not rendered this session is kept verbatim, so Export round-trips it.
STATE_BOOTSTRAP_JS = r"""
function emptyUnit(u) {
  const corners = {};
  for (const c of u.corners) corners[c.k] = {verdict: null, absent_kind: null};
  return {corners: corners, blind: null, complete: false, elapsed_s: 0, note: '',
          seen: false, inventory_seen: false, edited_after_inventory: false};
}
function copyCorners(cs) {
  const out = {};
  for (const k in (cs || {})) out[k] = {verdict: cs[k].verdict || null,
                                        absent_kind: cs[k].absent_kind || null};
  return out;
}
function fromFile(f) {
  return {corners: copyCorners(f.corners), blind: f.blind ? copyCorners(f.blind) : null,
          complete: !!f.complete, elapsed_s: f.elapsed_s || 0, note: f.note || '',
          seen: true, inventory_seen: !!f.inventory_seen,
          edited_after_inventory: !!f.edited_after_inventory};
}
function sameUnit(s, f) {
  return JSON.stringify(copyCorners(s.corners)) === JSON.stringify(copyCorners(f.corners)) &&
         !!s.complete === !!f.complete;
}
function bootstrapState(INITIAL, local, UNITS, ITEMS_SHA) {
  const state = local || {};
  let prefilled = 0, reopened = 0, initialIgnored = false;
  const conflicts = [];
  if (INITIAL && INITIAL.items_sha256 !== ITEMS_SHA) initialIgnored = true;
  else if (INITIAL) {
    for (const id in (INITIAL.units || {})) {
      const s = state[id], f = INITIAL.units[id];
      if (s && (s.seen || s.complete)) { if (!sameUnit(s, f)) conflicts.push(id); continue; }
      state[id] = fromFile(f);
      prefilled++;
    }
  }
  for (const u of UNITS) {
    const s = state[u.id];
    if (!s) { state[u.id] = emptyUnit(u); continue; }
    const keys = new Set(u.corners.map(c => c.k));
    let changed = false;
    for (const k of Object.keys(s.corners)) if (!keys.has(k)) { delete s.corners[k]; changed = true; }
    for (const k of keys) if (!(k in s.corners)) { s.corners[k] = {verdict: null, absent_kind: null}; changed = true; }
    if (changed) { if (s.complete) reopened++; s.complete = false; }
  }
  return {state: state, prefilled: prefilled, reopened: reopened,
          initialIgnored: initialIgnored, conflicts: conflicts};
}
"""

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corner ramp ratings</title>
<style>
  :root{--aw:min(620px, calc(100vh - 150px));--ok:#1a7f37;--no:#b42318;--ct:#5c6670;--acc:#0b63ce}
  body{font-family:system-ui,sans-serif;margin:8px 16px;background:#fafafa;color:#1d1d1f}
  .bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
  .bar button,.btn{font-size:14px;padding:5px 12px;cursor:pointer}
  button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible,
  .view:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
  .meta{color:#555;font-size:13px}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#eee;color:#333}
  .badge.done{background:var(--ok);color:#fff}
  .badge.todo{background:#fff3bf;color:#5c4500}
  #notice{display:none;background:#fff2df;border:1px solid #c77700;border-radius:8px;
          padding:8px 14px;margin:0 0 10px;font-size:13px}
  #main{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
  #left{flex:0 0 auto;width:var(--aw);position:sticky;top:6px}
  #aerialwrap{position:relative;width:var(--aw);height:var(--aw);background:#222;border-radius:6px;overflow:hidden}
  #aerialwrap img,#aerialwrap svg{position:absolute;left:0;top:0;width:100%;height:100%}
  #attrib{font-size:11px;color:#555;margin:3px 0 8px}
  #right{flex:1;min-width:320px;display:flex;flex-direction:column;gap:10px}
  fieldset.corner{background:#fff;border:1px solid #ccc;border-left:8px solid #bbb;border-radius:6px;
                  padding:6px 10px 8px;margin:0}
  fieldset.corner.active{box-shadow:0 0 0 3px var(--acc)}
  fieldset.corner.v-present{border-left-color:var(--ok)}
  fieldset.corner.v-absent{border-left-color:var(--no)}
  fieldset.corner.v-cant_tell{border-left-color:var(--ct)}
  legend{font-size:14px;font-weight:600;padding:0 4px}
  legend .meta{font-weight:normal}
  .views{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 6px}
  .view{padding:0;border:2px solid transparent;border-radius:4px;background:#eee;cursor:zoom-in;text-align:left}
  .view img{width:200px;height:200px;display:block}
  .view .cap{display:block;font-size:11px;color:#333;padding:1px 3px}
  .nocrop{width:200px;height:200px;display:flex;align-items:center;justify-content:center;
          font-size:12px;color:#555;background:#ddd}
  .choices{display:flex;gap:14px;flex-wrap:wrap;font-size:14px}
  .choices label{cursor:pointer;padding:3px 6px;border-radius:4px}
  .choices label:has(input:checked){background:#e8f0fe;font-weight:600}
  .akind{font-size:13px;margin-top:4px;color:#333}
  .akind label{margin-right:12px;cursor:pointer}
  .inv{font-size:13px;margin-top:6px;background:#f3f6fb;border-radius:4px;padding:4px 8px}
  .inv .late{color:#8a4b00;font-weight:600}
  #unitnote{width:100%;box-sizing:border-box;font:13px sans-serif;padding:5px}
  #completebtn{font-size:15px;padding:6px 16px;margin:6px 0}
  #notes{background:#eef4ff;border:1px solid #9db8e8;border-radius:8px;padding:8px 12px;margin:0 0 10px;font-size:13px}
  #notes summary{cursor:pointer;font-weight:bold;color:#24457f}
  #notes input,#notes textarea,#notes select{font:13px sans-serif;padding:3px 6px}
  #notes textarea{width:100%;box-sizing:border-box}
  #help{display:none;position:fixed;right:20px;top:20px;bottom:20px;overflow:auto;max-width:760px;background:#fff;
        border:1px solid #888;border-radius:8px;padding:12px 18px;font-size:13px;line-height:1.5;z-index:20;
        box-shadow:0 4px 18px rgba(0,0,0,.25)}
  #help pre{white-space:pre-wrap;font-size:12px;background:#f6f6f6;padding:8px;border-radius:4px}
  #lb{display:none;position:fixed;inset:0;z-index:40;background:rgba(0,0,0,.85);
      align-items:center;justify-content:center;gap:14px;padding:20px;box-sizing:border-box}
  #lb .big img{width:min(82vh,900px);height:min(82vh,900px);border-radius:6px;background:#333}
  #lb .cap{color:#eee;font-size:14px;margin-top:6px;text-align:center}
  #lb .side{display:flex;flex-direction:column;gap:6px}
  #lb .side img{width:140px;height:140px;border:3px solid transparent;border-radius:4px;cursor:pointer}
  #lb .side img.on{border-color:#ffb000}
  kbd{background:#eee;border:1px solid #bbb;border-radius:3px;padding:0 4px;font-size:12px}
  .sr{position:absolute;left:-9999px}
  @media (max-width:900px){#left{position:static;width:100%}#aerialwrap{width:100%;height:auto;aspect-ratio:1}}
</style>

<div id="notice" role="status"></div>
<details id="notes">
  <summary>Review notes (exported as <code>review_notes</code>)</summary>
  <div><label>Reviewer <input id="n_reviewer" size="10"></label>
    <label>Date <input id="n_date" size="10" placeholder="YYYY-MM-DD"></label>
    <label>Confidence <select id="n_conf"><option value=""></option>
    <option>high</option><option>medium</option><option>low</option></select></label></div>
  <div><label>Summary <textarea id="n_summary" rows="2"></textarea></label></div>
  <div><label>Caveats (one per line) <textarea id="n_caveats" rows="3"></textarea></label></div>
</details>

<nav class="bar" aria-label="Units">
  <button id="prev">&#8592; Prev</button><button id="next">Next &#8594;</button>
  <button id="nexttodo">Next to do</button>
  <label class="sr" for="unitsel">Jump to unit</label><select id="unitsel"></select>
  <span id="progress" class="meta"></span>
  <span style="flex:1"></span>
  <button id="helpbtn" aria-expanded="false" aria-controls="help">Rubric and keys (?)</button>
  <button id="export">Export</button>
</nav>
<h1 id="title" style="margin:4px 0 8px;font-size:17px"></h1>
<div id="live" class="sr" aria-live="polite"></div>
<div id="main">
  <section id="left" aria-label="Aerial and unit controls">
    <div id="aerialwrap"><img id="aerial" alt=""><svg id="plan" role="img" aria-labelledby="plantitle"><title id="plantitle">Aerial plan</title></svg></div>
    <div id="attrib"></div>
    <button id="completebtn"></button>
    <p><label for="unitnote">Note on this unit (optional; say why for any can't tell)</label>
      <textarea id="unitnote" rows="3"></textarea></p>
    <p class="meta">Corner markers: number = corner; fill: green present, red absent, grey can't
      tell, white not yet rated; blue ring = the corner the keys act on. Dots with dashed lines =
      the cameras of that corner's crops. Squares (after completion only) = city inventory
      points: A Available, N NA with no RAMPTYPE, T NA with a RAMPTYPE, R RMV, X Expired/Removed,
      ? other.</p>
  </section>
  <section id="right" aria-label="Corners"></section>
</div>
<div id="lb" role="dialog" aria-modal="true" aria-label="Enlarged crop"></div>
<div id="help" role="dialog" aria-label="Rubric and keys">
  <button id="helpclose" style="float:right">Close</button>
  <b>Keys</b> (not while typing in a text box)<br>
  <kbd>1</kbd>-<kbd>9</kbd> pick the corner the keys act on ·
  <kbd>j</kbd>/<kbd>k</kbd> next / previous corner<br>
  <kbd>p</kbd> present · <kbd>a</kbd> absent · <kbd>t</kbd> can't tell (then the next unrated
  corner becomes active) · after absent: <kbd>b</kbd> curb, no ramp · <kbd>n</kbd> no curb or
  sidewalk<br>
  <kbd>c</kbd> complete / reopen the unit (shows the city inventory) ·
  <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> units · <kbd>Enter</kbd> on a crop or a click: enlarge
  (<kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> step, <kbd>Esc</kbd> close) · <kbd>?</kbd> this panel<br><br>
  State is saved in this browser as you go. <b>Export</b> downloads the verdicts file; save it
  into the bundle under the name it suggests. Opening the page again with that file in place
  prefills it.
  <pre>__RUBRIC_HTML__</pre>
</div>

<script>
const UNITS = __UNITS__;
const ITEMS_SHA = __ITEMS_SHA__;        // sha256 of items.jsonl: every export is bound to it
const CITY = __CITY__;
const RATER = __RATER__;
const RUBRIC_VERSION = __RUBRIC_V__;
const RUBRIC = __RUBRIC_JSON__;
const SCHEMA = __SCHEMA__;
const INITIAL = __INITIAL__;            // an existing verdicts file, or null
const ATTRIBUTION = __ATTRIBUTION__;
const FILE_NAME = __FILE_NAME__;
const STORE = 'cornergallery243:' + CITY + ':' + RATER + ':' + ITEMS_SHA;
const NSTORE = STORE + ':notes', ISTORE = STORE + ':idx';
const VERDICTS = ['present', 'absent', 'cant_tell'];
const VLABEL = {present: 'Present', absent: 'Absent', cant_tell: "Can't tell"};
const AKIND = {curb_no_ramp: 'curb, no ramp', no_curb: 'no curb or sidewalk'};
const VCOL = {present: '#1a7f37', absent: '#b42318', cant_tell: '#5c6670'};
const INVTAG = {Available: 'A', NA_noramp: 'N', NA_typed: 'T', RMV: 'R', 'Expired/Removed': 'X', other: '?'};
const INVTEXT = {Available: 'Available', NA_noramp: 'NA, no RAMPTYPE', NA_typed: 'NA with a RAMPTYPE',
                 RMV: 'RMV', 'Expired/Removed': 'Expired/Removed', other: 'other status'};

__STATE_BOOTSTRAP__

function loadLocal() { try { return JSON.parse(localStorage.getItem(STORE) || '{}'); } catch (e) { return {}; } }
const boot = bootstrapState(INITIAL, loadLocal(), UNITS, ITEMS_SHA);
const state = boot.state;
function save() { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} }
(function notice() {
  const bits = [];
  if (boot.initialIgnored) bits.push('<b>The verdicts file was made on another item list and was NOT loaded.</b>');
  if (boot.prefilled) bits.push(boot.prefilled + ' unit(s) prefilled from the verdicts file.');
  if (boot.conflicts.length) bits.push('<b>' + boot.conflicts.length + ' unit(s) kept this browser&#39;s work over a different entry in the verdicts file</b>: ' + boot.conflicts.join(', ') + '. Clear this page&#39;s site data to take the file instead.');
  if (boot.reopened) bits.push('<b>' + boot.reopened + ' complete unit(s) reopened</b>: their corners changed since they were rated.');
  if (bits.length) { const n = document.getElementById('notice'); n.style.display = ''; n.innerHTML = bits.join('<br>'); }
})();
document.getElementById('attrib').textContent = ATTRIBUTION;
function say(t) { document.getElementById('live').textContent = t; }
function esc(t) { return String(t == null ? '' : t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }

// --- review notes (as #224) ------------------------------------------------------------
let notes = Object.assign({}, (INITIAL && !boot.initialIgnored && INITIAL.review_notes) || {});
try { Object.assign(notes, JSON.parse(localStorage.getItem(NSTORE) || '{}')); } catch (e) {}
const NF = {reviewer: 'n_reviewer', reviewed_at: 'n_date', confidence: 'n_conf', summary: 'n_summary'};
for (const k in NF) document.getElementById(NF[k]).value = notes[k] || '';
document.getElementById('n_caveats').value = (notes.caveats || []).join('\n');
function saveNotes() {
  for (const k in NF) notes[k] = document.getElementById(NF[k]).value.trim();
  notes.caveats = document.getElementById('n_caveats').value.split('\n').map(s => s.trim()).filter(Boolean);
  try { localStorage.setItem(NSTORE, JSON.stringify(notes)); } catch (e) {}
}
document.querySelectorAll('#notes input, #notes textarea, #notes select').forEach(el => el.addEventListener('input', saveNotes));

// --- geometry: lat/lng -> aerial pixels through Web Mercator world pixels (as #224) ----
function worldPx(lat, lng, z) {
  const n = Math.pow(2, z) * 256;
  return [(lng + 180) / 360 * n, (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n];
}
function toImg(u, lat, lng) {
  const a = u.aerial, w = a.world_px, p = worldPx(lat, lng, a.zoom);
  return [(p[0] - w.x0) / (w.x1 - w.x0) * a.px, (p[1] - w.y0) / (w.y1 - w.y0) * a.px];
}
function pxPerMetre(u) {
  const a = u.aerial, mpp = 156543.03392 * Math.cos(u.centre.lat * Math.PI / 180) / Math.pow(2, a.zoom);
  return a.px / (a.world_px.x1 - a.world_px.x0) / mpp;
}

// --- navigation and edits ----------------------------------------------------------------
let idx = 0, active = 0;
try { const i = parseInt(localStorage.getItem(ISTORE), 10); if (i >= 0 && i < UNITS.length) idx = i; } catch (e) {}
function cur() { return UNITS[idx]; }
function S() { return state[cur().id]; }
function rated(u) { const s = state[u.id]; return u.corners.filter(c => s.corners[c.k].verdict).length; }
function go(d) { idx = (idx + d + UNITS.length) % UNITS.length; active = 0; renderUnit(); }
function goTo(i) { idx = i; active = 0; renderUnit(); }
function touched() {
  const s = S();
  s.seen = true;
  if (s.inventory_seen) s.edited_after_inventory = true;
}
function setVerdict(ci, v) {
  const u = cur(), s = S(), c = u.corners[ci];
  if (!c) return;
  if (s.complete) { say('Reopen the unit (c) before changing a verdict.'); alert('This unit is complete. Reopen it first (c or the Reopen button).'); return; }
  touched();
  const e = s.corners[c.k];
  e.verdict = v;
  if (v !== 'absent') e.absent_kind = null;
  save(); refresh();
  say('Corner ' + c.corner + ': ' + VLABEL[v]);
}
function setKind(ci, kind) {
  const u = cur(), s = S(), c = u.corners[ci];
  if (!c || s.complete || s.corners[c.k].verdict !== 'absent') return;
  touched();
  s.corners[c.k].absent_kind = kind || null;
  save(); refresh();
}
function nextUnrated() {
  const u = cur(), s = S();
  for (let d = 1; d <= u.corners.length; d++) {
    const i = (active + d) % u.corners.length;
    if (!s.corners[u.corners[i].k].verdict) return i;
  }
  return active;
}
function toggleComplete() {
  const u = cur(), s = S();
  if (!s.complete) {
    const left = u.corners.length - rated(u);
    if (left) { alert(left + ' corner(s) still need a verdict.'); return; }
    s.seen = true;
    s.complete = true;
    if (!s.blind) s.blind = copyCorners(s.corners);   // the first completion is the blind read
    s.inventory_seen = true;
    say('Unit complete. City inventory shown.');
  } else {
    s.complete = false;
    say('Unit reopened.');
  }
  save(); renderUnit();
}

// --- timing (as #224): elapsed_s while shown, visible and active in the last 60 s ------
const IDLE_S = 60;
let lastTick = performance.now(), sinceSave = 0, lastInput = performance.now();
for (const ev of ['keydown', 'mousedown', 'mousemove', 'wheel', 'scroll', 'touchstart'])
  window.addEventListener(ev, () => { lastInput = performance.now(); }, {passive: true, capture: true});
setInterval(() => {
  const now = performance.now(), dt = Math.min((now - lastTick) / 1000, 2);
  lastTick = now;
  if (document.visibilityState !== 'visible' || !UNITS.length) return;
  if (now - lastInput > IDLE_S * 1000) return;
  const s = S();
  s.elapsed_s = Math.round((s.elapsed_s + dt) * 10) / 10;
  s.seen = true;
  if (++sinceSave >= 5) { sinceSave = 0; save(); }
}, 1000);
document.addEventListener('visibilitychange', () => { lastTick = performance.now(); if (document.visibilityState === 'hidden') save(); });
window.addEventListener('pagehide', () => save());

// --- rendering ---------------------------------------------------------------------------
function newestDate(c) { return c.views.map(v => v.date || '').sort().pop() || ''; }
function invHtml(u, c) {
  if (!c.inventory.length) return '<div class="inv">City inventory: no point in this corner.</div>';
  const newest = newestDate(c);
  return '<div class="inv">City inventory in this corner:<ul style="margin:2px 0 0 18px;padding:0">' +
    c.inventory.map(p => {
      const late = p.instdate && newest && p.instdate.slice(0, 7) > newest;
      return '<li>' + esc(p.unit_id) + ': <b>' + esc(INVTEXT[p.class] || p.class) + '</b>' +
        (p.ramptype ? ', ' + esc(p.ramptype) : '') + (p.instdate ? ', installed ' + esc(p.instdate) : '') +
        (late ? ' <span class="late">(after the newest crop, ' + esc(newest) + ')</span>' : '') +
        (p.defect ? ', ' + esc(p.defect) : '') + '</li>';
    }).join('') + '</ul></div>';
}
function renderUnit() {
  const u = cur(), s = S();
  try { localStorage.setItem(ISTORE, String(idx)); } catch (e) {}
  document.getElementById('title').textContent = 'Unit ' + (idx + 1) + ' of ' + UNITS.length + ': ' + u.id +
    ' (' + u.type + ', ' + u.corners.length + ' corner' + (u.corners.length > 1 ? 's' : '') + ')';
  document.getElementById('aerial').src = u.aerial.file;
  document.getElementById('aerial').alt = 'Aerial image of ' + u.id + ', north up, about 70 m across';
  document.getElementById('unitnote').value = s.note || '';
  const right = document.getElementById('right');
  right.innerHTML = u.corners.map((c, ci) => {
    const e = s.corners[c.k];
    const end = (c.start_deg + c.width_deg) % 360;
    const views = c.views.length ? c.views.map((v, vi) =>
      '<button class="view" data-ci="' + ci + '" data-vi="' + vi + '"><img loading="lazy" src="' + esc(v.crop) +
      '" alt="Corner ' + c.corner + ', view ' + (vi + 1) + ': pano ' + esc(v.pano_id) + ', captured ' + esc(v.date || 'unknown') +
      ', ' + v.dist_m.toFixed(1) + ' m from the corner point" onerror="this.replaceWith(Object.assign(document.createElement(\'span\'),{className:\'nocrop\',textContent:\'crop missing\'}))">' +
      '<span class="cap">' + (vi + 1) + ' · ' + esc(v.date || '?') + ' · ' + v.dist_m.toFixed(1) + ' m</span></button>').join('')
      : '<span class="nocrop">no pano within 40 m</span>';
    return '<fieldset class="corner" id="corner-' + ci + '" data-ci="' + ci + '">' +
      '<legend>Corner ' + c.corner + ' <span class="meta">bearings ' + Math.round(c.start_deg) + '° to ' + Math.round(end) +
      '°, ' + Math.round(c.width_deg) + '° wide' + (c.wide ? ' (wide: often the far side of a T)' : '') + '</span> ' +
      '<span class="badge" id="vb-' + ci + '"></span></legend>' +
      '<div class="views">' + views + '</div>' +
      '<div class="choices" role="radiogroup" aria-label="Verdict for corner ' + c.corner + '">' +
      VERDICTS.map(v => '<label><input type="radio" name="v-' + ci + '" value="' + v + '"' + (e.verdict === v ? ' checked' : '') +
        (s.complete ? ' disabled' : '') + '> ' + VLABEL[v] + ' <kbd>' + {present: 'p', absent: 'a', cant_tell: 't'}[v] + '</kbd></label>').join('') +
      '</div>' +
      '<div class="akind" id="ak-' + ci + '" role="radiogroup" aria-label="Kind of absence, corner ' + c.corner + '">Absent because (optional): ' +
      Object.keys(AKIND).map(k => '<label><input type="radio" name="ak-' + ci + '" value="' + k + '"' + (e.absent_kind === k ? ' checked' : '') +
        (s.complete ? ' disabled' : '') + '> ' + AKIND[k] + ' <kbd>' + (k === 'no_curb' ? 'n' : 'b') + '</kbd></label>').join('') +
      '<label><input type="radio" name="ak-' + ci + '" value=""' + (!e.absent_kind ? ' checked' : '') + (s.complete ? ' disabled' : '') + '> not specified</label></div>' +
      (s.complete ? invHtml(u, c) : '') + '</fieldset>';
  }).join('');
  right.querySelectorAll('input[name^="v-"]').forEach(el => el.addEventListener('change', () => {
    const ci = +el.name.slice(2); active = ci; setVerdict(ci, el.value);
  }));
  right.querySelectorAll('input[name^="ak-"]').forEach(el => el.addEventListener('change', () => setKind(+el.name.slice(3), el.value)));
  right.querySelectorAll('fieldset.corner').forEach(el => el.addEventListener('focusin', () => { active = +el.dataset.ci; refresh(); }));
  right.querySelectorAll('fieldset.corner').forEach(el => el.addEventListener('mousedown', () => { active = +el.dataset.ci; refresh(); }));
  right.querySelectorAll('.view').forEach(el => el.addEventListener('click', () => openLightbox(+el.dataset.ci, +el.dataset.vi)));
  refresh();
}
function refresh() {
  const u = cur(), s = S();
  u.corners.forEach((c, ci) => {
    const e = s.corners[c.k], fs = document.getElementById('corner-' + ci);
    fs.className = 'corner' + (ci === active ? ' active' : '') + (e.verdict ? ' v-' + e.verdict : '');
    const b = document.getElementById('vb-' + ci);
    b.textContent = e.verdict ? VLABEL[e.verdict] + (e.absent_kind ? ' (' + AKIND[e.absent_kind] + ')' : '') : 'not rated';
    b.className = 'badge' + (e.verdict ? '' : ' todo');
    document.getElementById('ak-' + ci).style.display = e.verdict === 'absent' ? '' : 'none';
    fs.querySelectorAll('input[name="v-' + ci + '"]').forEach(el => { el.checked = el.value === e.verdict; });
    fs.querySelectorAll('input[name="ak-' + ci + '"]').forEach(el => { el.checked = el.value === (e.absent_kind || ''); });
  });
  const cb = document.getElementById('completebtn');
  cb.textContent = s.complete ? 'Reopen unit (c)' : 'Complete unit and show the inventory (c)';
  const done = UNITS.filter(x => state[x.id].complete).length;
  document.getElementById('progress').textContent = done + ' of ' + UNITS.length + ' units complete · this unit ' +
    rated(u) + '/' + u.corners.length + ' corners rated' + (s.complete ? ' · complete' : '') +
    (s.edited_after_inventory ? ' · edited after the inventory was shown' : '');
  const sel = document.getElementById('unitsel');
  sel.innerHTML = UNITS.map((x, i) => '<option value="' + i + '"' + (i === idx ? ' selected' : '') + '>' +
    (state[x.id].complete ? '✓ ' : '· ') + (i + 1) + ' ' + esc(x.id) + '</option>').join('');
  drawPlan();
}
function drawPlan() {
  const u = cur(), s = S(), svg = document.getElementById('plan'), out = ['<title id="plantitle">Aerial plan of ' + esc(u.id) + '</title>'];
  const ppm = pxPerMetre(u), c0 = toImg(u, u.centre.lat, u.centre.lng);
  svg.setAttribute('viewBox', '0 0 ' + u.aerial.px + ' ' + u.aerial.px);
  out.push('<circle cx="' + c0[0] + '" cy="' + c0[1] + '" r="' + (u.window_m * ppm) + '" fill="none" stroke="#fff" stroke-dasharray="6 5" stroke-width="1.5" opacity=".8"/>');
  for (const b of u.legs) {
    const r = u.window_m * ppm, t = b * Math.PI / 180;
    out.push('<line x1="' + c0[0] + '" y1="' + c0[1] + '" x2="' + (c0[0] + r * Math.sin(t)) + '" y2="' + (c0[1] - r * Math.cos(t)) + '" stroke="#fff" stroke-width="2" opacity=".85"/>');
  }
  const ac = u.corners[active];
  if (ac) for (const [vi, v] of ac.views.entries()) {
    const p = toImg(u, v.cam.lat, v.cam.lng), q = toImg(u, ac.lat, ac.lng);
    out.push('<line x1="' + p[0] + '" y1="' + p[1] + '" x2="' + q[0] + '" y2="' + q[1] + '" stroke="#ffd400" stroke-width="1.5" stroke-dasharray="4 3"/>');
    out.push('<circle cx="' + p[0] + '" cy="' + p[1] + '" r="6" fill="#ffd400" stroke="#000"/><text x="' + (p[0] + 8) + '" y="' + (p[1] + 4) + '" font-size="12" fill="#ffd400" stroke="#000" stroke-width=".4">' + (vi + 1) + '</text>');
  }
  if (s.complete) for (const c of u.corners) for (const p of c.inventory) {
    if (p.lat == null) continue;
    const q = toImg(u, p.lat, p.lng);
    out.push('<rect x="' + (q[0] - 8) + '" y="' + (q[1] - 8) + '" width="16" height="16" fill="#fff" stroke="#000" stroke-width="1.5"><title>' + esc(p.unit_id + ' ' + (INVTEXT[p.class] || p.class)) + '</title></rect>' +
             '<text x="' + q[0] + '" y="' + (q[1] + 4) + '" font-size="11" font-weight="bold" text-anchor="middle">' + (INVTAG[p.class] || '?') + '</text>');
  }
  u.corners.forEach((c, ci) => {
    const q = toImg(u, c.lat, c.lng), v = s.corners[c.k].verdict;
    out.push('<circle cx="' + q[0] + '" cy="' + q[1] + '" r="12" fill="' + (v ? VCOL[v] : '#fff') + '" stroke="' + (ci === active ? '#0b63ce' : '#000') + '" stroke-width="' + (ci === active ? 5 : 1.5) + '"/>' +
             '<text x="' + q[0] + '" y="' + (q[1] + 5) + '" font-size="14" font-weight="bold" text-anchor="middle" fill="' + (v ? '#fff' : '#000') + '">' + c.corner + '</text>');
  });
  svg.innerHTML = out.join('');
}

// --- enlarged crop -------------------------------------------------------------------------
let lb = null;
function openLightbox(ci, vi) {
  const c = cur().corners[ci], v = c.views[vi];
  if (!v) return;
  lb = {ci: ci, vi: vi, back: document.activeElement};
  const el = document.getElementById('lb');
  el.innerHTML = '<div class="big"><img src="' + esc(v.crop) + '" alt="Corner ' + c.corner + ', view ' + (vi + 1) + ', enlarged"><div class="cap">Corner ' + c.corner +
    ' · view ' + (vi + 1) + ' of ' + c.views.length + ' · captured ' + esc(v.date || '?') + ' · ' + v.dist_m.toFixed(1) + ' m · pano ' + esc(v.pano_id) +
    '<br>Esc or click outside: close · ←/→: other views</div></div>' +
    (c.views.length > 1 ? '<div class="side">' + c.views.map((w, i) => '<img data-vi="' + i + '" class="' + (i === vi ? 'on' : '') + '" src="' + esc(w.crop) + '" alt="view ' + (i + 1) + '">').join('') + '</div>' : '') +
    '<button id="lbclose" style="position:fixed;top:12px;right:16px">Close</button>';
  el.style.display = 'flex';
  document.getElementById('lbclose').focus();
}
function closeLightbox() { document.getElementById('lb').style.display = 'none'; const b = lb && lb.back; lb = null; if (b && b.focus) b.focus(); }
document.getElementById('lb').addEventListener('click', ev => {
  const t = ev.target.closest('[data-vi]');
  if (t && lb) { openLightbox(lb.ci, +t.dataset.vi); return; }
  if (ev.target.id === 'lbclose' || !ev.target.closest('.big img')) closeLightbox();
});

// --- controls ------------------------------------------------------------------------------
document.getElementById('prev').onclick = () => go(-1);
document.getElementById('next').onclick = () => go(1);
document.getElementById('nexttodo').onclick = () => {
  for (let d = 1; d <= UNITS.length; d++) { const i = (idx + d) % UNITS.length; if (!state[UNITS[i].id].complete) { goTo(i); return; } }
  say('Every unit is complete.'); alert('Every unit is complete.');
};
document.getElementById('unitsel').onchange = ev => goTo(+ev.target.value);
document.getElementById('completebtn').onclick = () => toggleComplete();
document.getElementById('unitnote').addEventListener('input', ev => { S().note = ev.target.value; S().seen = true; save(); });
function toggleHelp(show) {
  const h = document.getElementById('help'), on = show === undefined ? h.style.display !== 'block' : show;
  h.style.display = on ? 'block' : 'none';
  document.getElementById('helpbtn').setAttribute('aria-expanded', String(on));
  if (on) document.getElementById('helpclose').focus();
}
document.getElementById('helpbtn').onclick = () => toggleHelp();
document.getElementById('helpclose').onclick = () => toggleHelp(false);
document.addEventListener('keydown', ev => {
  const t = ev.target, tag = (t.tagName || '').toLowerCase();
  if (tag === 'textarea' || tag === 'select' || (tag === 'input' && t.type !== 'radio')) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (lb) {
    const n = cur().corners[lb.ci].views.length;
    if (ev.key === 'Escape') closeLightbox();
    else if (ev.key === 'ArrowRight') openLightbox(lb.ci, (lb.vi + 1) % n);
    else if (ev.key === 'ArrowLeft') openLightbox(lb.ci, (lb.vi - 1 + n) % n);
    else return;
    ev.preventDefault(); return;
  }
  const k = ev.key, u = cur();
  if (k === 'Escape') { toggleHelp(false); return; }
  if (k === 'ArrowLeft' && tag !== 'input') go(-1);
  else if (k === 'ArrowRight' && tag !== 'input') go(1);
  else if (k >= '1' && k <= '9') { if (+k - 1 < u.corners.length) { active = +k - 1; refresh(); scrollActive(); } }
  else if (k === 'j') { active = (active + 1) % u.corners.length; refresh(); scrollActive(); }
  else if (k === 'k') { active = (active - 1 + u.corners.length) % u.corners.length; refresh(); scrollActive(); }
  else if (k === 'p' || k === 'a' || k === 't') {
    setVerdict(active, {p: 'present', a: 'absent', t: 'cant_tell'}[k]);
    if (k !== 'a' && !S().complete) { active = nextUnrated(); refresh(); scrollActive(); }
  }
  else if (k === 'b' || k === 'n') { setKind(active, k === 'b' ? 'curb_no_ramp' : 'no_curb'); if (!S().complete) { active = nextUnrated(); refresh(); scrollActive(); } }
  else if (k === 'c') toggleComplete();
  else if (k === '?') toggleHelp();
  else return;
  ev.preventDefault();
});
function scrollActive() { const el = document.getElementById('corner-' + active); if (el) el.scrollIntoView({block: 'nearest'}); }

// --- export ----------------------------------------------------------------------------------
function exportUnit(u, s) {
  return {type: u.type, corners: copyCorners(s.corners), blind: s.blind ? copyCorners(s.blind) : null,
          complete: !!s.complete, elapsed_s: Math.round(s.elapsed_s * 10) / 10,
          note: (s.note || '').trim(), inventory_seen: !!s.inventory_seen,
          edited_after_inventory: !!s.edited_after_inventory};
}
document.getElementById('export').onclick = () => {
  const open = UNITS.filter(u => state[u.id].seen && !state[u.id].complete && rated(u)).length;
  if (open && !confirm(open + ' unit(s) have verdicts but are not complete; the scorer ignores them. Export anyway?')) return;
  saveNotes();
  const out = {schema: SCHEMA, city: CITY, items_sha256: ITEMS_SHA, rater: RATER,
               rubric_version: RUBRIC_VERSION, rubric: RUBRIC, exported_at: new Date().toISOString()};
  const rn = {};
  for (const k of ['reviewer', 'reviewed_at', 'confidence', 'summary']) if (notes[k]) rn[k] = notes[k];
  if ((notes.caveats || []).length) rn.caveats = notes.caveats;
  out.review_notes = rn;
  out.units = {};
  const known = new Set(UNITS.map(u => u.id));
  for (const u of UNITS) { const s = state[u.id]; if (s.complete || rated(u) || (s.note || '').trim()) out.units[u.id] = exportUnit(u, s); }
  if (INITIAL && !boot.initialIgnored) for (const id in (INITIAL.units || {})) if (!known.has(id)) out.units[id] = INITIAL.units[id];
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = FILE_NAME; a.click();
  say('Exported ' + Object.keys(out.units).length + ' units as ' + FILE_NAME);
};

save();
renderUnit();
</script>
</html>
"""
