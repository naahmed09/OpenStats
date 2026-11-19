// OpenStats browser/webview shell. Plain ES module: no build step, no dependencies.
//
// It loads the `openstats_host.wasm` module (the C ABI of the `openstats-host` crate), forwards DOM
// events to it as numbers, and paints the binary canvas command stream it returns onto a <canvas>.
// Everything the user sees (menus, dialogs, grid, Session window) is drawn by the Rust side.

export const SPECIAL = 0x200000;
export const SPECIAL_KEYS = {
  ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3, Home: 4, End: 5, PageUp: 6, PageDown: 7,
  Enter: 8, Tab: 9, Escape: 10, Backspace: 11, Delete: 12, F2: 13, F6: 14, F10: 15, F5: 16, F8: 17, F9: 18, F1: 19,
};

/** Modifier bits: 1 shift, 2 ctrl (Cmd on macOS counts as ctrl), 4 alt. */
export function modBits(e) {
  return (e.shiftKey ? 1 : 0) | ((e.ctrlKey || e.metaKey) ? 2 : 0) | (e.altKey ? 4 : 0);
}

/** The wire key code of a KeyboardEvent, or null when the key is not one the app handles. */
export function keyCode(e) {
  if (e.key in SPECIAL_KEYS) return SPECIAL + SPECIAL_KEYS[e.key];
  const cps = Array.from(e.key || '');
  if (cps.length === 1) return cps[0].codePointAt(0);
  return null;
}

/** Decode the `OTC1` binary command stream into command objects (the JSON shape of the Rust side). */
export function decodeFrame(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 8 || String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== 'OTC1') {
    throw new Error('not an OTC1 frame');
  }
  const n = dv.getUint32(4, true);
  let i = 8;
  const f = () => { const v = dv.getFloat32(i, true); i += 4; return v; };
  const col = () => { const c = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]; i += 4; return c; };
  const dec = new TextDecoder();
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    const op = bytes[i++];
    switch (op) {
      case 0: out[k] = { op: 'Clear', w: f(), h: f(), color: col() }; break;
      case 1: out[k] = { op: 'Save' }; break;
      case 2: out[k] = { op: 'Restore' }; break;
      case 3: out[k] = { op: 'Clip', x: f(), y: f(), w: f(), h: f() }; break;
      case 4: out[k] = { op: 'FillStyle', color: col() }; break;
      case 5: {
        const color = col(), width = f(), nd = bytes[i++], dash = [];
        for (let d = 0; d < nd; d++) dash.push(f());
        out[k] = { op: 'StrokeStyle', color, width, dash };
        break;
      }
      case 6: out[k] = { op: 'BeginPath' }; break;
      case 7: out[k] = { op: 'MoveTo', x: f(), y: f() }; break;
      case 8: out[k] = { op: 'LineTo', x: f(), y: f() }; break;
      case 9: out[k] = { op: 'Arc', x: f(), y: f(), r: f(), a0: f(), a1: f() }; break;
      case 10: out[k] = { op: 'Rect', x: f(), y: f(), w: f(), h: f() }; break;
      case 11: out[k] = { op: 'ClosePath' }; break;
      case 12: out[k] = { op: 'Fill' }; break;
      case 13: out[k] = { op: 'Stroke' }; break;
      case 14: {
        const x = f(), y = f(), size = f(), rotate = f(), color = col();
        const bold = bytes[i++] !== 0, anchor = bytes[i++];
        const len = dv.getUint16(i, true); i += 2;
        const text = dec.decode(bytes.subarray(i, i + len)); i += len;
        const no = bytes[i++], overbars = [];
        for (let o = 0; o < no; o++) { overbars.push(dv.getUint16(i, true)); i += 2; }
        out[k] = { op: 'Text', x, y, text, size, color, bold, anchor, rotate, overbars };
        break;
      }
      default: throw new Error(`unknown opcode ${op} at command ${k}`);
    }
  }
  if (i !== bytes.length) throw new Error(`frame has ${bytes.length - i} trailing bytes`);
  return out;
}

const rgba = (c) => `rgba(${c[0]},${c[1]},${c[2]},${c[3] / 255})`;

/** Replay decoded commands on a 2D context. A `Clear` fills the background (the canvas is sized by the shell). */
export function paint(ctx, cmds) {
  for (const c of cmds) {
    switch (c.op) {
      case 'Clear': ctx.fillStyle = rgba(c.color); ctx.fillRect(0, 0, c.w, c.h); break;
      case 'Save': ctx.save(); break;
      case 'Restore': ctx.restore(); break;
      case 'Clip': ctx.beginPath(); ctx.rect(c.x, c.y, c.w, c.h); ctx.clip(); break;
      case 'FillStyle': ctx.fillStyle = rgba(c.color); break;
      case 'StrokeStyle':
        ctx.strokeStyle = rgba(c.color); ctx.lineWidth = c.width; ctx.setLineDash(c.dash); ctx.lineJoin = 'round';
        break;
      case 'BeginPath': ctx.beginPath(); break;
      case 'MoveTo': ctx.moveTo(c.x, c.y); break;
      case 'LineTo': ctx.lineTo(c.x, c.y); break;
      case 'Arc': ctx.arc(c.x, c.y, c.r, c.a0, c.a1); break;
      case 'Rect': ctx.rect(c.x, c.y, c.w, c.h); break;
      case 'ClosePath': ctx.closePath(); break;
      case 'Fill': ctx.fill(); break;
      case 'Stroke': ctx.stroke(); break;
      case 'Text': {
        ctx.save();
        ctx.font = `${c.bold ? 'bold ' : ''}${c.size}px Helvetica, Arial, sans-serif`;
        ctx.fillStyle = rgba(c.color);
        ctx.textAlign = ['left', 'center', 'right'][c.anchor];
        ctx.translate(c.x, c.y);
        if (c.rotate) ctx.rotate(-c.rotate * Math.PI / 180);
        ctx.fillText(c.text, 0, 0);
        if (c.overbars.length) {
          const chars = Array.from(c.text);
          const total = ctx.measureText(c.text).width;
          const off = c.anchor === 1 ? -total / 2 : c.anchor === 2 ? -total : 0;
          for (const i of c.overbars) {
            const before = ctx.measureText(chars.slice(0, i).join('')).width;
            const w = ctx.measureText(chars[i] || '').width;
            ctx.fillRect(off + before, -c.size * 0.9, w, Math.max(1, c.size * 0.07));
          }
        }
        ctx.restore();
        break;
      }
      default: throw new Error(`unknown command ${c.op}`);
    }
  }
}

/** Thin typed wrapper over the wasm exports (also used with a fake in tests). */
export class Abi {
  constructor(exports) {
    this.x = exports;
    this.enc = new TextEncoder();
    this.dec = new TextDecoder();
  }
  mem() { return this.x.memory.buffer; }
  put(bytes) {
    const p = this.x.ot_input_reserve(bytes.length);
    new Uint8Array(this.mem(), p, bytes.length).set(bytes);
    return bytes.length;
  }
  putText(s) { return this.put(this.enc.encode(s)); }
  out(len) { return new Uint8Array(this.mem(), this.x.ot_out_ptr(), len).slice(); }
  outText(len) { return this.dec.decode(this.out(len)); }
  slot(i) { return new Uint8Array(this.mem(), this.x.ot_slot_ptr(i), this.x.ot_slot_len(i)).slice(); }
}

/** Outcome kinds returned by `ot_outcome_next`. */
export const OUTCOME = { FILE: 1, CLIPBOARD: 2, PRINT: 3, NEED_OPEN: 4, NEED_IMPORT: 5, ERROR: 6, TEAR_OFF: 7 };

/**
 * The application: owns a canvas, forwards events, paints frames and reports outcomes.
 * `handlers` may define onFile({name, mime, bytes}), onClipboard(text), onPrint(frames[]),
 * onNeedOpen(), onNeedImport(), onError(message), onTearOff({name, json}), onTitle(title), onFrame() (after every drawn frame).
 */
export class OpenStats {
  constructor(abi, canvas, handlers = {}, doc = globalThis.document) {
    this.abi = abi;
    this.canvas = canvas;
    this.h = handlers;
    this.doc = doc;
    this.ctx = canvas.getContext('2d');
    this.dpr = 1;
    this.composing = false;
    this.pending = false;
    this.lastTitle = null;
    this.input = null;
  }

  static async load(source, canvas, handlers, doc) {
    let bytes = source;
    if (typeof source === 'string') bytes = await (await fetch(source)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const abi = new Abi(instance.exports);
    if (abi.x.ot_abi_version() !== 1) throw new Error('openstats-host ABI version mismatch');
    const app = new OpenStats(abi, canvas, handlers, doc);
    app.start();
    return app;
  }

  /** Size the canvas to its CSS box, tell Rust, and draw. */
  fit(width, height, dpr = 1) {
    this.dpr = dpr;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    if (this.canvas.style) { this.canvas.style.width = `${width}px`; this.canvas.style.height = `${height}px`; }
    this.abi.x.ot_resize(width, height);
    this.draw();
  }

  start() {
    const r = this.canvas.getBoundingClientRect ? this.canvas.getBoundingClientRect() : { width: 1000, height: 700 };
    const w = r.width || 1000, h = r.height || 700;
    this.abi.x.ot_init(w, h);
    this.restoreTheme();
    this.attach();
    this.fit(w, h, globalThis.devicePixelRatio || 1);
    this.focusInput();
  }

  /** Re-apply the theme saved by an earlier session (older wasm builds without `ot_theme` skip this). */
  restoreTheme() {
    if (!this.abi.x.ot_theme) return;
    try {
      const saved = globalThis.localStorage && globalThis.localStorage.getItem('openstats.theme');
      const id = { light: 'view.theme_light', dark: 'view.theme_dark', 'high-contrast': 'view.theme_high_contrast' }[saved];
      if (id) this.abi.x.ot_command(this.abi.putText(id));
      this.lastTheme = saved || 'light';
    } catch { /* storage unavailable: start with the default theme */ }
  }

  /** Remember the theme whenever it changes. */
  saveTheme() {
    if (!this.abi.x.ot_theme) return;
    const name = this.abi.outText(this.abi.x.ot_theme());
    if (name === this.lastTheme) return;
    this.lastTheme = name;
    try { globalThis.localStorage && globalThis.localStorage.setItem('openstats.theme', name); } catch { /* ignore */ }
  }

  /** Request one repaint on the next animation frame. */
  schedule() {
    if (this.pending) return;
    this.pending = true;
    const raf = globalThis.requestAnimationFrame || ((fn) => setTimeout(fn, 0));
    raf(() => { this.pending = false; this.draw(); });
  }

  draw() {
    const n = this.abi.x.ot_render();
    const cmds = decodeFrame(this.abi.out(n));
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    paint(this.ctx, cmds);
    this.afterEvent();
  }

  /** Deliver queued outcomes, move the IME box, update the title. */
  afterEvent() {
    for (;;) {
      const k = this.abi.x.ot_outcome_next();
      if (!k) break;
      this.deliver(k);
    }
    if (this.input && this.abi.x.ot_ime_anchor()) {
      const rect = this.canvas.getBoundingClientRect ? this.canvas.getBoundingClientRect() : { left: 0, top: 0 };
      this.input.style.left = `${rect.left + this.abi.x.ot_ime_x()}px`;
      this.input.style.top = `${rect.top + this.abi.x.ot_ime_y()}px`;
    }
    const n = this.abi.x.ot_title();
    const t = this.abi.outText(n);
    if (t !== this.lastTitle) {
      this.lastTitle = t;
      if (this.h.onTitle) this.h.onTitle(t);
    }
    this.saveTheme();
    if (this.h.onFrame) this.h.onFrame();
  }

  deliver(kind) {
    const a = this.abi, h = this.h;
    switch (kind) {
      case OUTCOME.FILE:
        if (h.onFile) h.onFile({ name: a.dec.decode(a.slot(0)), mime: a.dec.decode(a.slot(1)), bytes: a.slot(2) });
        break;
      case OUTCOME.CLIPBOARD: if (h.onClipboard) h.onClipboard(a.dec.decode(a.slot(0))); break;
      case OUTCOME.PRINT: {
        const frames = [];
        const pages = a.x.ot_print_pages();
        for (let i = 0; i < pages; i++) frames.push(decodeFrame(a.out(a.x.ot_print_frame(i))));
        if (h.onPrint) h.onPrint(frames);
        break;
      }
      case OUTCOME.NEED_OPEN: if (h.onNeedOpen) h.onNeedOpen(); break;
      case OUTCOME.NEED_IMPORT: if (h.onNeedImport) h.onNeedImport(); break;
      case OUTCOME.ERROR: if (h.onError) h.onError(a.dec.decode(a.slot(0))); break;
      case OUTCOME.TEAR_OFF: if (h.onTearOff) h.onTearOff({ name: a.dec.decode(a.slot(0)), json: a.dec.decode(a.slot(3)) }); break;
      default: break;
    }
  }

  // ------------------------------------------------------------------ input

  pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  attach() {
    const c = this.canvas, d = this.doc;
    c.addEventListener('mousedown', (e) => {
      // Without this the browser's default action moves focus to the body *after* our focus call,
      // so the hidden textarea never receives key events.
      e.preventDefault();
      this.focusInput();
      const [x, y] = this.pos(e);
      this.abi.x.ot_pointer(0, x, y, e.button, modBits(e), e.detail || 1);
      this.draw();
    });
    c.addEventListener('mousemove', (e) => {
      const [x, y] = this.pos(e);
      this.abi.x.ot_pointer(1, x, y, 0, modBits(e), 0);
      this.schedule();
    });
    d.addEventListener('mouseup', (e) => {
      const [x, y] = this.pos(e);
      this.abi.x.ot_pointer(2, x, y, e.button, modBits(e), 1);
      this.focusInput();
      this.draw();
    });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('selectstart', (e) => e.preventDefault());
    c.addEventListener('dragstart', (e) => e.preventDefault());
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const scale = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.canvas.height / this.dpr : 1;
      const [x, y] = this.pos(e);
      this.abi.x.ot_wheel(x, y, e.deltaX * scale, e.deltaY * scale);
      this.schedule();
    }, { passive: false });

    // A hidden textarea receives keys, paste and input-method events.
    const ta = d.createElement('textarea');
    Object.assign(ta.style, { position: 'fixed', left: '0px', top: '0px', width: '1px', height: '1px', opacity: '0', padding: '0', border: '0', resize: 'none' });
    ta.setAttribute('autocomplete', 'off');
    ta.setAttribute('autocapitalize', 'off');
    ta.setAttribute('spellcheck', 'false');
    d.body.appendChild(ta);
    this.input = ta;
    ta.addEventListener('keydown', (e) => this.onKey(e));
    ta.addEventListener('paste', (e) => {
      e.preventDefault();
      const t = (e.clipboardData && e.clipboardData.getData('text/plain')) || '';
      if (t) { this.abi.x.ot_paste(this.abi.putText(t)); this.draw(); }
    });
    ta.addEventListener('compositionstart', () => { this.composing = true; });
    ta.addEventListener('compositionupdate', (e) => {
      const t = e.data || '';
      this.abi.x.ot_ime_update(this.abi.putText(t), Array.from(t).length);
      this.draw();
    });
    ta.addEventListener('compositionend', (e) => {
      this.composing = false;
      const t = e.data || '';
      this.abi.x.ot_ime_cancel();
      if (t) this.abi.x.ot_ime_commit(this.abi.putText(t));
      ta.value = '';
      this.draw();
    });
    ta.addEventListener('input', () => { if (!this.composing) ta.value = ''; });
  }

  focusInput() { if (this.input && this.input.focus) this.input.focus({ preventScroll: true }); }

  onKey(e) {
    if (this.composing || e.isComposing || e.keyCode === 229) return; // the input method owns it
    const bits = modBits(e);
    const ctrl = (bits & 2) !== 0;
    // Let the browser fire its own paste event for Ctrl+V so the clipboard text arrives with it.
    if (ctrl && !e.altKey && e.key.toLowerCase() === 'v') return;
    const code = keyCode(e);
    if (code === null) return;
    // Keep browser shortcuts (reload, devtools, tab switching) that the app does not use.
    const mine = code >= SPECIAL || !ctrl || 'nopsxcazyfidl'.includes(e.key.toLowerCase());
    if (!mine) return;
    e.preventDefault();
    this.abi.x.ot_key(code, bits);
    this.draw();
  }

  // ------------------------------------------------------------------ commands and files

  /** Run a command by its text id (from the menu JSON). Returns whether the id is known. */
  command(id) {
    const ok = this.abi.x.ot_command(this.abi.putText(id)) === 1;
    this.draw();
    return ok;
  }

  menu() { return JSON.parse(this.abi.outText(this.abi.x.ot_menu_json())); }

  openProject(bytes) { this.abi.x.ot_open_project(this.abi.put(bytes)); this.draw(); }

  importData(name, bytes) {
    const nb = this.abi.enc.encode(name);
    const all = new Uint8Array(nb.length + bytes.length);
    all.set(nb, 0); all.set(bytes, nb.length);
    this.abi.x.ot_import(nb.length, this.abi.put(all));
    this.draw();
  }

  saveProject() { return this.abi.out(this.abi.x.ot_save_project()); }
}

/** A torn-off graph window: draws one chart at the size of its canvas. */
export class GraphWindow {
  constructor(abi, canvas, json) {
    this.abi = abi;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.handle = abi.x.ot_graph_open(abi.putText(json));
    if (!this.handle) throw new Error(abi.outText(abi.x.ot_out_len()) || 'cannot open the graph');
    this.title = abi.outText(abi.x.ot_graph_title(this.handle));
  }
  static async load(source, canvas, json) {
    let bytes = source;
    if (typeof source === 'string') bytes = await (await fetch(source)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const abi = new Abi(instance.exports);
    abi.x.ot_init(100, 100);
    return new GraphWindow(abi, canvas, json);
  }
  draw(width, height, dpr = 1) {
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    const n = this.abi.x.ot_graph_render(this.handle, width, height);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paint(this.ctx, decodeFrame(this.abi.out(n)));
  }
  close() { this.abi.x.ot_graph_close(this.handle); }
}

/** Browser defaults for outcomes: download, clipboard, file pickers, printing. */
export function browserHandlers(getApp, doc = globalThis.document) {
  const pick = (accept, cb) => {
    const i = doc.createElement('input');
    i.type = 'file';
    i.accept = accept;
    i.onchange = async () => {
      const f = i.files && i.files[0];
      if (f) cb(f.name, new Uint8Array(await f.arrayBuffer()));
    };
    i.click();
  };
  return {
    onFile: ({ name, mime, bytes }) => {
      const a = doc.createElement('a');
      a.href = URL.createObjectURL(new Blob([bytes], { type: mime }));
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    },
    onClipboard: (t) => { if (navigator.clipboard) navigator.clipboard.writeText(t); },
    onNeedOpen: () => pick('.otb', (n, b) => getApp().openProject(b)),
    onNeedImport: () => pick('.csv,.tsv,.txt,.xlsx,.xlsm,.json,.gz', (n, b) => getApp().importData(n, b)),
    onError: (m) => console.warn(m),
    onTitle: (t) => { doc.title = t; },
    onPrint: (frames) => {
      const box = doc.createElement('div');
      box.className = 'openstats-print';
      for (const f of frames) {
        const cv = doc.createElement('canvas');
        const clear = f[0];
        cv.width = clear.w; cv.height = clear.h;
        paint(cv.getContext('2d'), f);
        box.appendChild(cv);
      }
      doc.body.appendChild(box);
      globalThis.print();
      box.remove();
    },
  };
}
