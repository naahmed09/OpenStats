// Run with: node --test openstats-desktop/web/test/tauri-shell.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { tauriHandlers, installNativeMenu, graphFromTauri } from '../tauri-shell.js';

function mockTauri(over = {}) {
  const log = [];
  const listeners = {};
  const T = {
    log, listeners,
    window: { getCurrentWindow: () => ({ setTitle: async (t) => log.push(['title', t]) }) },
    clipboardManager: { writeText: async (t) => log.push(['clip', t]) },
    dialog: {
      save: async (o) => { log.push(['save', o]); return over.savePath === undefined ? '/tmp/out.csv' : over.savePath; },
      open: async (o) => { log.push(['open', o]); return over.openPath === undefined ? '/home/u/data.xlsx' : over.openPath; },
    },
    fs: {
      writeFile: async (p, b) => log.push(['write', p, Array.from(b)]),
      readFile: async (p) => { if (over.readFails) throw new Error('denied'); log.push(['read', p]); return new Uint8Array([1, 2]); },
    },
    core: { invoke: async (c, a) => { log.push(['invoke', c, a]); } },
    event: {
      listen: async (n, f) => { listeners[n] = f; },
      emit: async (n, p) => log.push(['emit', n, p]),
    },
  };
  return T;
}

const fakeDoc = { createElement: () => ({ style: {} }), body: { appendChild() {} } };
const flush = () => new Promise((r) => setTimeout(r, 0));

function app() {
  const calls = [];
  return { calls, openProject: (b) => calls.push(['open', Array.from(b)]), importData: (n, b) => calls.push(['import', n, Array.from(b)]), command: (id) => calls.push(['command', id]), menu: () => [{ label: 'File', entries: [] }] };
}

test('exports go through the native save dialog and the file system', async () => {
  const T = mockTauri();
  const h = tauriHandlers(T, () => app(), fakeDoc);
  await h.onFile({ name: 'results.csv', mime: 'text/csv', bytes: new Uint8Array([7, 8]) });
  assert.deepEqual(T.log[0], ['save', { defaultPath: 'results.csv', filters: [{ name: 'CSV', extensions: ['csv'] }] }]);
  assert.deepEqual(T.log[1], ['write', '/tmp/out.csv', [7, 8]]);
});

test('cancelling the save dialog writes nothing', async () => {
  const T = mockTauri({ savePath: null });
  await tauriHandlers(T, () => app(), fakeDoc).onFile({ name: 'a.otb', mime: '', bytes: new Uint8Array(1) });
  assert.ok(!T.log.some((l) => l[0] === 'write'));
});

test('open and import read the chosen file and hand it to the app', async () => {
  const T = mockTauri();
  const a = app();
  const h = tauriHandlers(T, () => a, fakeDoc);
  await h.onNeedOpen();
  assert.deepEqual(T.log.find((l) => l[0] === 'open')[1].filters, [{ name: 'OpenStats project', extensions: ['otb'] }]);
  assert.deepEqual(a.calls[0], ['open', [1, 2]]);
  await h.onNeedImport();
  assert.deepEqual(a.calls[1], ['import', 'data.xlsx', [1, 2]], 'the import gets the file name, not the path');
  const filters = T.log.filter((l) => l[0] === 'open')[1][1].filters[0].extensions;
  assert.ok(['csv', 'xlsx'].every((e) => filters.includes(e)));
  const T2 = mockTauri({ openPath: null });
  const a2 = app();
  await tauriHandlers(T2, () => a2, fakeDoc).onNeedOpen();
  assert.deepEqual(a2.calls, [], 'cancel does nothing');
});

test('Windows-style paths yield the bare file name', async () => {
  const T = mockTauri({ openPath: 'C:\\Users\\me\\Desktop\\scores.csv' });
  const a = app();
  await tauriHandlers(T, () => a, fakeDoc).onNeedImport();
  assert.equal(a.calls[0][1], 'scores.csv');
});

test('clipboard, title and tear-off', async () => {
  const T = mockTauri();
  const h = tauriHandlers(T, () => app(), fakeDoc);
  await h.onClipboard('a\tb');
  h.onTitle('Proj - OpenStats');
  await h.onTearOff({ name: 'Histogram 1', json: '{"x":1}' });
  assert.deepEqual(T.log, [['clip', 'a\tb'], ['title', 'Proj - OpenStats'], ['invoke', 'open_graph_window', { name: 'Histogram 1', json: '{"x":1}' }]]);
});

test('file errors are reported, not thrown', async () => {
  const T = mockTauri({ readFails: true });
  const warned = [];
  const orig = console.warn; console.warn = (m) => warned.push(m);
  try {
    await tauriHandlers(T, () => app(), fakeDoc).onNeedOpen();
    await flush();
  } finally { console.warn = orig; }
  assert.match(warned[0], /denied/);
});

test('the native menu is published when it changes and its commands run in the app', async () => {
  const T = mockTauri();
  const a = app();
  const publish = await installNativeMenu(T, a);
  assert.deepEqual(T.log.filter((l) => l[0] === 'invoke'), [['invoke', 'set_menu', { tree: [{ label: 'File', entries: [] }] }]]);
  await publish();
  assert.equal(T.log.filter((l) => l[0] === 'invoke').length, 1, 'unchanged menus are not resent');
  a.menu = () => [{ label: 'File', entries: [{ label: 'Undo', id: 'edit.undo', enabled: true }] }];
  await publish();
  assert.equal(T.log.filter((l) => l[0] === 'invoke').length, 2);
  await T.listeners['menu-command']({ payload: 'file.new' });
  assert.deepEqual(a.calls, [['command', 'file.new']]);
});

test('a graph window announces itself, then draws the spec it receives', async () => {
  const T = mockTauri();
  const events = {};
  const win = { innerWidth: 640, innerHeight: 400, devicePixelRatio: 2, addEventListener: (t, f) => { events[t] = f; } };
  const drawn = [];
  class G { constructor(j) { this.json = j; this.title = 'Histogram 1'; }
    static async load(url, canvas, json) { drawn.push(['load', url, json]); return new G(json); }
    draw(w, h, d) { drawn.push(['draw', w, h, d]); } close() { drawn.push(['close']); } }
  await graphFromTauri(T, G, {}, './x.wasm', win);
  assert.deepEqual(T.log.find((l) => l[0] === 'emit'), ['emit', 'graph-ready', null]);
  await T.listeners['graph-open']({ payload: { name: 'n', json: '{"a":1}' } });
  assert.deepEqual(drawn, [['load', './x.wasm', '{"a":1}'], ['draw', 640, 400, 2]]);
  assert.deepEqual(T.log.find((l) => l[0] === 'title'), ['title', 'Histogram 1']);
  await T.listeners['graph-open']({ payload: { name: 'n', json: '{"a":2}' } });
  assert.equal(drawn[2][0], 'close', 'a second spec replaces the first');
});
