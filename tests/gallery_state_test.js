// Behavioral test of the spot-check gallery's review state machine.
// Usage: node gallery_state_test.js <path-to-extracted-gallery.js>
// Exits 0 on success, 1 with a FAIL message on the first failed assertion.
// Expects the gallery built with two entries: [0] two crops, [1] zero crops.
const fs = require('fs');
const js = fs.readFileSync(process.argv[2], 'utf8');

function makeEl() {
  return {
    style: {}, classList: { toggle() {}, contains: () => false },
    innerHTML: '', textContent: '', title: '', className: '', disabled: false,
    appendChild() {}, remove() {}, removeAttribute() {},
    click() { if (this.onclick) this.onclick(); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1600, height: 800 }),
    href: '', download: '',
  };
}
const els = {};
let lastBlob = null;
global.document = {
  getElementById: id => (els[id] ||= makeEl()),
  querySelectorAll: () => [],
  createElement: () => makeEl(),
  addEventListener() {},
};
global.localStorage = { _s: {}, getItem(k) { return this._s[k] ?? null; }, setItem(k, v) { this._s[k] = v; } };
global.Blob = class { constructor(parts) { lastBlob = parts.join(''); } };
global.URL = { createObjectURL: () => 'blob:x' };

eval(js + `
// ---- assertions run in the gallery script's own scope ----
const assert = (cond, msg) => { if (!cond) { console.error('FAIL:', msg); process.exit(1); } };
const e0 = ENTRIES[0];
assert(e0.crops.length === 2, 'fixture entry 0 has 2 crops');
assert(!reviewed(e0), 'seen-only pano is not reviewed');

cycle(0); cycle(1);                         // judge both crops correct
assert(verdicts[e0.pid].dets.every(d => d === true), 'both dets judged correct');
assert(!reviewed(e0), 'crops judged but FN check missing -> not reviewed');

cycle(1);                                   // correct -> incorrect
assert(verdicts[e0.pid].dets[1] === false, 'cycle advances correct -> incorrect');
cycle(1);                                   // incorrect -> unsure
assert(verdicts[e0.pid].dets[1] === 'unsure', 'cycle advances incorrect -> unsure');

document.getElementById('nomiss').click();  // affirm no missed ramps
assert(verdicts[e0.pid].noMissed === true, 'noMissed set by button');
// 'unsure' is a decision (not null), so an all-judged pano with an unsure crop
// still counts as reviewed once the missed-ramp check is done.
assert(reviewed(e0), 'judged (incl. unsure) + affirmed -> reviewed');

// adding a missed marker revokes the affirmation
document.getElementById('panowrap').onclick({ clientX: 400, clientY: 200, target: { closest: () => null } });
const s0 = verdicts[e0.pid];
assert(s0.missed.length === 1, 'missed marker added at click point');
assert(Math.abs(s0.missed[0].x - 0.25) < 1e-9, 'marker x normalized to pano width');
assert(s0.noMissed === false, 'marker revokes noMissed');
assert(reviewed(e0), 'judged + missed mark -> still reviewed');
assert(document.getElementById('nomiss').disabled === true, 'affirm button disabled once a marker exists');

// zero-detection pano: paging past is NOT a review
document.getElementById('next').click();
const e1 = ENTRIES[1];
assert(view[idx].pid === e1.pid, 'navigated to the empty pano');
assert(!reviewed(e1), 'empty pano seen but unaffirmed -> not reviewed');
document.getElementById('nomiss').click();
assert(reviewed(e1), 'empty pano affirmed -> reviewed');

// export carries the scorer's contract
document.getElementById('export').click();
const out = JSON.parse(lastBlob);
assert(out.panos[e0.pid].no_missed === false && out.panos[e0.pid].missed.length === 1, 'export: pano with marker');
assert(out.panos[e1.pid].no_missed === true && out.panos[e1.pid].missed.length === 0, 'export: affirmed empty pano');
assert(Array.isArray(out.panos[e0.pid].dets) && out.panos[e0.pid].dets.length === 2, 'export: dets array intact');
console.log('gallery state machine OK');
// The pytest wrapper feeds this line into score_validation.collect(), so the
// viewer's export and the scorer's expectations are checked against each other.
console.log('EXPORT_JSON ' + JSON.stringify(out));
`);
