// Run with: node --test openstats-desktop/web/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Abi, OpenStats, GraphWindow, OUTCOME, SPECIAL, decodeFrame, keyCode, modBits, paint } from '../openstats.js';

const dir = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const frame = new Uint8Array(readFileSync(join(dir, 'frame.bin')));
const summary = JSON.parse(readFileSync(join(dir, 'frame.json'), 'utf8'));

// ---------------------------------------------------------------- the decoder against real Rust output

test('decodes a frame written by the Rust encoder exactly', () => {
  const cmds = decodeFrame(frame);
  assert.equal(cmds.length, summary.total);
  const counts = {};
  for (const c of cmds) counts[c.op] = (counts[c.op] || 0) + 1;
  assert.deepEqual(counts, summary.counts);
  assert.deepEqual(cmds.filter((c) => c.op === 'Text').map((c) => c.text), summary.texts);
  const first = cmds.find((c) => c.op === 'Text');
  assert.equal(first.x, summary.firstText.x);
  assert.equal(first.y, summary.firstText.y);
  assert.equal(first.size, summary.firstText.size);
  assert.deepEqual(first.color, summary.firstText.color);
  assert.equal(first.bold, summary.firstText.bold);
  assert.equal(first.anchor, summary.firstText.anchor);
  assert.equal(first.rotate, summary.firstText.rotate);
  assert.equal(first.overbars.length, summary.firstText.overbars);
  assert.deepEqual(cmds[0], { op: 'Clear', w: summary.width, h: summary.height, color: cmds[0].color });
  if (summary.dashedStroke) {
    const s = cmds.find((c) => c.op === 'StrokeStyle' && c.dash.length);
    assert.equal(s.width, summary.dashedStroke.width);
    assert.equal(s.dash.length, summary.dashedStroke.dash);
    assert.deepEqual(s.color, summary.dashedStroke.color);
  }
});

test('rejects malformed frames', () => {
  assert.throws(() => decodeFrame(new Uint8Array([1, 2, 3])), /not an OTC1/);
  assert.throws(() => decodeFrame(new Uint8Array([79, 84, 67, 50, 0, 0, 0, 0])), /not an OTC1/);
  const extra = new Uint8Array(frame.length + 1);
  extra.set(frame);
  assert.throws(() => decodeFrame(extra), /trailing/);
  const bad = frame.slice();
  bad[8] = 99; // first opcode
  assert.throws(() => decodeFrame(bad), /unknown opcode/);
});

// ---------------------------------------------------------------- painting

class Ctx {
  constructor() { this.calls = []; this.canvas = { width: 0, height: 0 }; }
  rec(name, ...a) { this.calls.push([name, ...a]); }
  measureText(s) { return { width: s.length * 6 }; }
}
for (const m of ['save', 'restore', 'beginPath', 'moveTo', 'lineTo', 'arc', 'rect', 'closePath', 'fill', 'stroke', 'clip', 'fillRect', 'fillText', 'translate', 'rotate', 'setLineDash', 'setTransform']) {
  Ctx.prototype[m] = function (...a) { this.rec(m, ...a); };
}

test('painting a real frame issues the matching canvas calls', () => {
  const ctx = new Ctx();
  paint(ctx, decodeFrame(frame));
  const n = (name) => ctx.calls.filter((c) => c[0] === name).length;
  assert.equal(n('fillText'), summary.counts.Text);
  assert.equal(n('clip'), summary.counts.Clip, 'each Clip clips');
  assert.equal(n('fill'), summary.counts.Fill);
  assert.equal(n('stroke'), summary.counts.Stroke);
  assert.equal(n('arc'), summary.counts.Arc);
  assert.equal(n('save'), summary.counts.Save + summary.counts.Text, 'texts are drawn inside save/restore');
  assert.equal(n('save'), n('restore'), 'balanced');
  assert.equal(ctx.calls[0][0], 'fillRect');
  assert.equal(n('setLineDash'), summary.counts.StrokeStyle, 'every stroke style sets its dash');
});

test('text anchors, rotation and overbars', () => {
  const ctx = new Ctx();
  const t = (o) => ({ op: 'Text', x: 10, y: 20, text: 'X̄ab', size: 10, color: [0, 0, 0, 255], bold: true, anchor: 1, rotate: 90, overbars: [], ...o });
  paint(ctx, [t({ overbars: [0] })]);
  const names = ctx.calls.map((c) => c[0]);
  assert.deepEqual(names.slice(0, 3), ['save', 'translate', 'rotate']);
  assert.equal(ctx.textAlign, 'center');
  assert.match(ctx.font, /^bold 10px/);
  assert.ok(names.includes('fillRect'), 'overbar drawn');
  assert.throws(() => paint(ctx, [{ op: 'Bogus' }]), /unknown command/);
});

// ---------------------------------------------------------------- key mapping

test('key and modifier mapping', () => {
  assert.equal(keyCode({ key: 'ArrowUp' }), SPECIAL + 0);
  assert.equal(keyCode({ key: 'F2' }), SPECIAL + 13);
  assert.equal(keyCode({ key: 'F6' }), SPECIAL + 14);
  assert.equal(keyCode({ key: 'F10' }), SPECIAL + 15);
  assert.equal(keyCode({ key: 'F5' }), SPECIAL + 16);
  assert.equal(keyCode({ key: 'F8' }), SPECIAL + 17);
  assert.equal(keyCode({ key: 'F9' }), SPECIAL + 18);
  assert.equal(keyCode({ key: 'a' }), 97);
  assert.equal(keyCode({ key: '✓' }), 0x2713);
  assert.equal(keyCode({ key: '😀' }), 0x1f600, 'astral characters are one code point');
  assert.equal(keyCode({ key: 'Shift' }), null);
  assert.equal(keyCode({ key: 'Dead' }), null);
  assert.equal(modBits({ shiftKey: true, ctrlKey: false, metaKey: true, altKey: true }), 7, 'Cmd counts as Ctrl');
  assert.equal(modBits({}), 0);
});

// ---------------------------------------------------------------- the application against a fake module

function fakeModule(opts = {}) {
  const memory = new WebAssembly.Memory({ initial: 4 });
  const calls = [];
  const IN = 1024, OUT = 20000, SLOT = [40000, 41000, 42000, 43000];
  const state = { out: new Uint8Array(0), slots: [[], [], [], []], queue: [], pages: [], title: 'OpenStats', anchor: false, input: new Uint8Array(0) };
  const enc = new TextEncoder();
  const put = (addr, bytes) => new Uint8Array(memory.buffer).set(bytes, addr);
  const rec = (n) => (...a) => { calls.push([n, ...a]); };
  const x = {
    memory,
    ot_abi_version: () => 1,
    ot_init: rec('init'),
    ot_resize: rec('resize'),
    ot_pointer: rec('pointer'),
    ot_wheel: rec('wheel'),
    ot_key: rec('key'),
    ot_ime_cancel: rec('ime_cancel'),
    ot_ime_x: () => 111, ot_ime_y: () => 222, ot_ime_h: () => 18,
    ot_ime_anchor: () => (state.anchor ? 1 : 0),
    ot_input_reserve: (len) => { calls.push(['reserve', len]); return IN; },
    ot_paste: (len) => { calls.push(['paste', new TextDecoder().decode(new Uint8Array(memory.buffer, IN, len))]); },
    ot_ime_update: (len, caret) => { calls.push(['ime_update', new TextDecoder().decode(new Uint8Array(memory.buffer, IN, len)), caret]); },
    ot_ime_commit: (len) => { calls.push(['ime_commit', new TextDecoder().decode(new Uint8Array(memory.buffer, IN, len))]); },
    ot_command: (len) => { const id = new TextDecoder().decode(new Uint8Array(memory.buffer, IN, len)); calls.push(['command', id]); return id.startsWith('ok') ? 1 : 0; },
    ot_open_project: (len) => { calls.push(['open_project', Array.from(new Uint8Array(memory.buffer, IN, len))]); },
    ot_import: (nl, tl) => { calls.push(['import', new TextDecoder().decode(new Uint8Array(memory.buffer, IN, nl)), Array.from(new Uint8Array(memory.buffer, IN + nl, tl - nl))]); },
    ot_save_project: () => { put(OUT, [9, 8, 7]); return 3; },
    ot_render: () => { calls.push(['render']); put(OUT, frame); return frame.length; },
    ot_menu_json: () => { const b = enc.encode('[{"label":"File","entries":[]}]'); put(OUT, b); return b.length; },
    ot_title: () => { const b = enc.encode(state.title); put(OUT, b); return b.length; },
    ot_out_ptr: () => OUT,
    ot_out_len: () => state.out.length,
    ot_outcome_next: () => (state.queue.length ? state.queue.shift()() : 0),
    ot_slot_ptr: (i) => SLOT[i],
    ot_slot_len: (i) => state.slots[i].length,
    ot_print_pages: () => state.pages.length,
    ot_print_frame: (i) => { put(OUT, state.pages[i]); return state.pages[i].length; },
    ot_graph_open: (len) => { const j = new TextDecoder().decode(new Uint8Array(memory.buffer, IN, len)); calls.push(['graph_open', j]); if (j === 'bad') { state.out = enc.encode('boom'); put(OUT, state.out); return 0; } return 7; },
    ot_graph_title: () => { const b = enc.encode('Histogram 1'); put(OUT, b); return b.length; },
    ot_graph_render: (h, w, ht) => { calls.push(['graph_render', h, w, ht]); put(OUT, frame); return frame.length; },
    ot_graph_close: rec('graph_close'),
  };
  const queueOutcome = (kind, slots = {}) => state.queue.push(() => {
    state.slots = [[], [], [], []];
    for (const [i, v] of Object.entries(slots)) { state.slots[i] = Array.from(typeof v === 'string' ? enc.encode(v) : v); put(SLOT[i], state.slots[i]); }
    return kind;
  });
  return { x, calls, state, queueOutcome, memory };
}

function fakeDom() {
  const listeners = {};
  const doc = {
    listeners: {},
    addEventListener(t, f) { (this.listeners[t] ||= []).push(f); },
    createElement() {
      const el = { style: {}, listeners: {}, attrs: {}, focused: 0,
        addEventListener(t, f) { (this.listeners[t] ||= []).push(f); },
        setAttribute(k, v) { this.attrs[k] = v; }, focus() { this.focused++; }, value: '' };
      return el;
    },
    body: { children: [], appendChild(e) { this.children.push(e); } },
  };
  const ctx = new Ctx();
  const canvas = {
    listeners,
    style: {},
    width: 0, height: 0,
    addEventListener(t, f) { (listeners[t] ||= []).push(f); },
    getContext: () => ctx,
    getBoundingClientRect: () => ({ left: 5, top: 7, width: 1000, height: 650 }),
  };
  return { doc, canvas, ctx };
}

const ev = (o = {}) => ({ preventDefault() { this.prevented = true; }, ...o });

function boot(handlers = {}) {
  const m = fakeModule();
  const { doc, canvas, ctx } = fakeDom();
  const app = new OpenStats(new Abi(m.x), canvas, handlers, doc);
  app.start();
  return { app, m, doc, canvas, ctx, ta: doc.body.children[0] };
}

test('start sizes the canvas, initialises Rust and draws a frame', () => {
  const { m, canvas, ctx } = boot();
  assert.deepEqual(m.calls.find((c) => c[0] === 'init'), ['init', 1000, 650]);
  assert.deepEqual(m.calls.find((c) => c[0] === 'resize'), ['resize', 1000, 650]);
  assert.equal(canvas.width, 1000);
  assert.ok(ctx.calls.some((c) => c[0] === 'fillText'));
  const hi = fakeModule();
  const d = fakeDom();
  const app = new OpenStats(new Abi(hi.x), d.canvas, {}, d.doc);
  app.start();
  app.fit(500, 300, 2);
  assert.equal(d.canvas.width, 1000);
  assert.equal(d.canvas.height, 600);
  assert.deepEqual(hi.calls.filter((c) => c[0] === 'resize').pop(), ['resize', 500, 300], 'Rust works in CSS pixels');
  assert.deepEqual(d.ctx.calls.filter((c) => c[0] === 'setTransform').pop(), ['setTransform', 2, 0, 0, 2, 0, 0]);
});

test('the theme is restored at start and saved when it changes', () => {
  const store = { 'openstats.theme': 'dark' };
  globalThis.localStorage = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  try {
    const m = fakeModule();
    let theme = 'light';
    m.x.ot_command = (len) => {
      const id = new TextDecoder().decode(new Uint8Array(m.memory.buffer, 1024, len));
      m.calls.push(['command', id]);
      if (id === 'view.theme_dark') theme = 'dark';
      if (id === 'view.theme_high_contrast') theme = 'high-contrast';
      return 1;
    };
    m.x.ot_theme = () => { const b = new TextEncoder().encode(theme); new Uint8Array(m.memory.buffer).set(b, 20000); return b.length; };
    const { doc, canvas } = fakeDom();
    const app = new OpenStats(new Abi(m.x), canvas, {}, doc);
    app.start();
    assert.deepEqual(m.calls.find((c) => c[0] === 'command'), ['command', 'view.theme_dark'], 'the saved theme is re-applied');
    theme = 'high-contrast';
    app.draw();
    assert.equal(store['openstats.theme'], 'high-contrast', 'a changed theme is saved');
  } finally {
    delete globalThis.localStorage;
  }
});

test('older wasm modules without ot_theme still start', () => {
  const { app } = boot();
  app.restoreTheme();
  app.saveTheme();
});

test('mouse events carry canvas-relative positions, button, modifiers and click count', () => {
  const { m, canvas, doc } = boot();
  canvas.listeners.mousedown[0](ev({ clientX: 105, clientY: 207, button: 2, shiftKey: true, detail: 2 }));
  assert.deepEqual(m.calls.find((c) => c[0] === 'pointer'), ['pointer', 0, 100, 200, 2, 1, 2]);
  canvas.listeners.mousemove[0](ev({ clientX: 55, clientY: 57, ctrlKey: true }));
  assert.deepEqual(m.calls.filter((c) => c[0] === 'pointer').pop(), ['pointer', 1, 50, 50, 0, 2, 0]);
  doc.listeners.mouseup[0](ev({ clientX: 55, clientY: 57, button: 0 }));
  assert.deepEqual(m.calls.filter((c) => c[0] === 'pointer').pop(), ['pointer', 2, 50, 50, 0, 0, 1]);
  const cm = ev();
  canvas.listeners.contextmenu[0](cm);
  assert.ok(cm.prevented, 'native context menu suppressed');
});

test('mousedown cancels the default focus change so the hidden textarea keeps focus', () => {
  const { canvas } = boot();
  const e = ev({ clientX: 10, clientY: 10, button: 0 });
  canvas.listeners.mousedown[0](e);
  assert.ok(e.prevented, 'default mousedown would blur the textarea and drop every keystroke');
});

test('wheel deltas are normalised to pixels', () => {
  const { m, canvas } = boot();
  const w = ev({ clientX: 15, clientY: 17, deltaX: 1, deltaY: 3, deltaMode: 1 });
  canvas.listeners.wheel[0](w);
  assert.ok(w.prevented);
  assert.deepEqual(m.calls.find((c) => c[0] === 'wheel'), ['wheel', 10, 10, 16, 48]);
  canvas.listeners.wheel[0](ev({ clientX: 15, clientY: 17, deltaX: 0, deltaY: 100, deltaMode: 0 }));
  assert.deepEqual(m.calls.filter((c) => c[0] === 'wheel').pop(), ['wheel', 10, 10, 0, 100]);
});

test('keys: typed characters, specials, app shortcuts; Ctrl+V and browser shortcuts pass through', () => {
  const { m, ta } = boot();
  const down = ta.listeners.keydown[0];
  const keys = () => m.calls.filter((c) => c[0] === 'key');
  let e = ev({ key: 'a' }); down(e);
  assert.deepEqual(keys().pop(), ['key', 97, 0]); assert.ok(e.prevented);
  e = ev({ key: 'ArrowDown', shiftKey: true }); down(e);
  assert.deepEqual(keys().pop(), ['key', SPECIAL + 1, 1]);
  e = ev({ key: 's', ctrlKey: true }); down(e);
  assert.deepEqual(keys().pop(), ['key', 115, 2]); assert.ok(e.prevented);
  let before = keys().length;
  e = ev({ key: 'v', ctrlKey: true }); down(e);
  assert.equal(keys().length, before); assert.ok(!e.prevented, 'the browser then fires its paste event');
  for (const [k, extra] of [['f', {}], ['i', {}], ['z', { shiftKey: true }], ['d', { shiftKey: true }], ['y', {}], ['n', { shiftKey: true }]]) {
    e = ev({ key: k, ctrlKey: true, ...extra }); down(e);
    assert.ok(e.prevented, `Ctrl+${k} is the app's, not the browser's`);
    assert.equal(keys().pop()[1], k.codePointAt(0));
  }
  for (const k of ['F6', 'F10', 'F5', 'F8', 'F9']) { e = ev({ key: k }); down(e); assert.ok(e.prevented, k); }
  const base = keys().length;
  e = ev({ key: 'f', altKey: true }); down(e);
  assert.equal(keys().length, base + 1, 'Alt+letter menu mnemonics reach the app');
  before = keys().length;
  e = ev({ key: 'r', ctrlKey: true }); down(e);
  assert.equal(keys().length, before); assert.ok(!e.prevented, 'reload is left to the browser');
  e = ev({ key: 'Shift' }); down(e);
  assert.equal(keys().length, before);
  e = ev({ key: 'x', metaKey: true }); down(e);
  assert.deepEqual(keys().pop(), ['key', 120, 2], 'Cmd+X acts as Ctrl+X');
});

test('paste events send the clipboard text', () => {
  const { m, ta } = boot();
  const e = ev({ clipboardData: { getData: (t) => (t === 'text/plain' ? 'a\tb\n1\t2' : '') } });
  ta.listeners.paste[0](e);
  assert.ok(e.prevented);
  assert.deepEqual(m.calls.find((c) => c[0] === 'paste'), ['paste', 'a\tb\n1\t2']);
  const n = m.calls.filter((c) => c[0] === 'paste').length;
  ta.listeners.paste[0](ev({ clipboardData: { getData: () => '' } }));
  assert.equal(m.calls.filter((c) => c[0] === 'paste').length, n, 'an empty paste is ignored');
});

test('input-method composition: updates, commit, cancel, and keys ignored while composing', () => {
  const { m, ta, app } = boot();
  ta.listeners.compositionstart[0](ev());
  assert.ok(app.composing);
  ta.listeners.keydown[0](ev({ key: 'k', isComposing: true }));
  ta.listeners.keydown[0](ev({ key: 'Process', keyCode: 229 }));
  assert.equal(m.calls.filter((c) => c[0] === 'key').length, 0);
  ta.listeners.compositionupdate[0](ev({ data: '日本' }));
  assert.deepEqual(m.calls.find((c) => c[0] === 'ime_update'), ['ime_update', '日本', 2]);
  ta.value = '日本語';
  ta.listeners.compositionend[0](ev({ data: '日本語' }));
  assert.ok(!app.composing);
  assert.deepEqual(m.calls.find((c) => c[0] === 'ime_commit'), ['ime_commit', '日本語']);
  assert.ok(m.calls.some((c) => c[0] === 'ime_cancel'), 'the preedit is dropped before the commit');
  assert.equal(ta.value, '', 'the textarea is cleared');
  ta.value = 'x';
  ta.listeners.input[0](ev());
  assert.equal(ta.value, '');
});

test('the candidate window follows the caret', () => {
  const { m, app, ta } = boot();
  m.state.anchor = true;
  app.draw();
  assert.equal(ta.style.left, '116px'); // canvas left 5 + 111
  assert.equal(ta.style.top, '229px');  // canvas top 7 + 222
});

test('outcomes reach the handlers', () => {
  const got = [];
  const { m, app } = boot({
    onFile: (f) => got.push(['file', f.name, f.mime, Array.from(f.bytes)]),
    onClipboard: (t) => got.push(['clip', t]),
    onPrint: (p) => got.push(['print', p.length, p[0][0].op]),
    onNeedOpen: () => got.push(['open']),
    onNeedImport: () => got.push(['import']),
    onError: (s) => got.push(['error', s]),
    onTearOff: (t) => got.push(['tear', t.name, t.json]),
    onTitle: (t) => got.push(['title', t]),
  });
  assert.deepEqual(got, [['title', 'OpenStats']], 'the first frame reports the title');
  got.length = 0;
  m.queueOutcome(OUTCOME.FILE, { 0: 'a.csv', 1: 'text/csv', 2: [1, 2, 3] });
  m.queueOutcome(OUTCOME.CLIPBOARD, { 0: 'héllo' });
  m.state.pages = [frame, frame];
  m.queueOutcome(OUTCOME.PRINT);
  m.queueOutcome(OUTCOME.NEED_OPEN);
  m.queueOutcome(OUTCOME.NEED_IMPORT);
  m.queueOutcome(OUTCOME.ERROR, { 0: 'bad file' });
  m.queueOutcome(OUTCOME.TEAR_OFF, { 0: 'Histogram 1', 3: '{"title":"t"}' });
  m.state.title = 'Project - OpenStats';
  app.draw();
  assert.deepEqual(got, [
    ['file', 'a.csv', 'text/csv', [1, 2, 3]],
    ['clip', 'héllo'],
    ['print', 2, 'Clear'],
    ['open'], ['import'],
    ['error', 'bad file'],
    ['tear', 'Histogram 1', '{"title":"t"}'],
    ['title', 'Project - OpenStats'],
  ]);
  got.length = 0;
  app.draw();
  assert.deepEqual(got, [], 'the title is reported only when it changes');
});

test('commands, files and the menu', () => {
  const { m, app } = boot();
  assert.equal(app.command('ok.thing'), true);
  assert.equal(app.command('nope'), false);
  assert.deepEqual(app.menu(), [{ label: 'File', entries: [] }]);
  app.openProject(new Uint8Array([4, 5, 6]));
  assert.deepEqual(m.calls.find((c) => c[0] === 'open_project'), ['open_project', [4, 5, 6]]);
  app.importData('données.csv', new Uint8Array([65, 10]));
  const imp = m.calls.find((c) => c[0] === 'import');
  assert.equal(imp[1], 'données.csv', 'the name is UTF-8 and its byte length is what Rust receives');
  assert.deepEqual(imp[2], [65, 10]);
  assert.deepEqual(Array.from(app.saveProject()), [9, 8, 7]);
});

test('graph windows', () => {
  const m = fakeModule();
  const d = fakeDom();
  const abi = new Abi(m.x);
  const g = new GraphWindow(abi, d.canvas, '{"title":"t"}');
  assert.equal(g.handle, 7);
  assert.equal(g.title, 'Histogram 1');
  g.draw(640, 400, 2);
  assert.deepEqual(m.calls.find((c) => c[0] === 'graph_render'), ['graph_render', 7, 640, 400]);
  assert.equal(d.canvas.width, 1280);
  assert.ok(d.ctx.calls.some((c) => c[0] === 'fillText'));
  g.close();
  assert.ok(m.calls.some((c) => c[0] === 'graph_close'));
  assert.throws(() => new GraphWindow(abi, d.canvas, 'bad'), /boom/);
});

test('ABI helper round-trips bytes through linear memory', () => {
  const m = fakeModule();
  const abi = new Abi(m.x);
  const n = abi.putText('héllo ✓');
  assert.equal(n, new TextEncoder().encode('héllo ✓').length);
  assert.equal(new TextDecoder().decode(new Uint8Array(m.memory.buffer, 1024, n)), 'héllo ✓');
});
