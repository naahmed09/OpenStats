// Tauri v2 integration for the OpenStats webview. Everything here takes the `__TAURI__`
// global as a parameter so it can be exercised under Node with a mock.
//
// Native pieces: menu bar (rebuilt in Rust from the workspace's menu tree), open/save
// dialogs, file reads/writes, clipboard, window title and graph tear-off windows.

import { browserHandlers } from './openstats.js';

const PROJECT_FILTER = [{ name: 'OpenStats project', extensions: ['otb'] }];
const DATA_FILTER = [{ name: 'Data files', extensions: ['csv', 'tsv', 'txt', 'xlsx', 'xlsm', 'json', 'gz'] }];

const basename = (p) => String(p).split(/[\\/]/).pop();

/** Handlers for `OpenStats` that use native dialogs, files and the system clipboard. */
export function tauriHandlers(T, getApp, doc = globalThis.document) {
  const fallback = browserHandlers(getApp, doc); // print and error reporting stay web-based
  const win = T.window.getCurrentWindow();
  const guard = (p) => p.catch((e) => fallback.onError(String(e)));
  return {
    ...fallback,
    onTitle: (t) => { win.setTitle(t); },
    onClipboard: (t) => guard(T.clipboardManager.writeText(t)),
    onFile: ({ name, bytes }) => guard((async () => {
      const ext = name.includes('.') ? [name.split('.').pop()] : [];
      const path = await T.dialog.save({ defaultPath: name, filters: ext.length ? [{ name: ext[0].toUpperCase(), extensions: ext }] : [] });
      if (path) await T.fs.writeFile(path, bytes);
    })()),
    onNeedOpen: () => guard((async () => {
      const path = await T.dialog.open({ multiple: false, filters: PROJECT_FILTER });
      if (path) getApp().openProject(await T.fs.readFile(path));
    })()),
    onNeedImport: () => guard((async () => {
      const path = await T.dialog.open({ multiple: false, filters: DATA_FILTER });
      if (path) getApp().importData(basename(path), await T.fs.readFile(path));
    })()),
    onTearOff: ({ name, json }) => guard(T.core.invoke('open_graph_window', { name, json })),
  };
}

/**
 * Keep the native menu bar in step with the workspace and route its commands back.
 * Returns a function that re-publishes the menu; call it after each frame.
 */
export async function installNativeMenu(T, app) {
  let last = '';
  const publish = async () => {
    const tree = app.menu();
    const key = JSON.stringify(tree);
    if (key === last) return;
    last = key;
    await T.core.invoke('set_menu', { tree });
  };
  await T.event.listen('menu-command', (e) => { app.command(e.payload); publish(); });
  await publish();
  return publish;
}

/** Entry point for the graph tear-off window: wait for its spec, then draw. */
export async function graphFromTauri(T, GraphWindow, canvas, wasmUrl, win = globalThis) {
  const self = T.window.getCurrentWindow();
  let g = null;
  await T.event.listen('graph-open', async (e) => {
    if (g) g.close();
    g = await GraphWindow.load(wasmUrl, canvas, e.payload.json);
    await self.setTitle(g.title);
    const redraw = () => g.draw(win.innerWidth, win.innerHeight, win.devicePixelRatio || 1);
    win.addEventListener('resize', redraw);
    redraw();
  });
  await T.event.emit('graph-ready', null);
  return () => g;
}
