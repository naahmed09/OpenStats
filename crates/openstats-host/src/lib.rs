//! Host bindings for OpenStats (std only).
//!
//! A browser or desktop shell embeds the Rust workspace through three layers, all in this crate:
//!
//! * [`Host`] — safe Rust. One window's [`Workspace`] plus everything a shell needs: abstract
//!   input events (pointer, wheel, key, text, input-method), commands by text id, file bytes in,
//!   [`Outcome`]s queued out (files to save, clipboard text, print pages, …), and the current
//!   frame as the compact binary canvas command stream (`OTC1`) from `openstats-charts`.
//! * [`GraphHost`] — a second kind of window, for a graph torn off into its own window.
//! * [`native`] — the layer a *native* desktop shell (a real window, no web view) needs: frames as
//!   pixels with pluggable system fonts, input, and the actions to perform (save, open, clipboard,
//!   graph windows).
//! * [`abi`] — a flat `extern "C"` API over one [`Host`] (and a table of [`GraphHost`]s) using
//!   only integers, floats and offsets into linear memory. It is what `web/openstats.js` calls in the
//!   browser and what the Tauri webview calls in the desktop app; no `wasm-bindgen`, `web-sys` or
//!   `js-sys` is needed, so the crate stays dependency-free and builds for `wasm32-unknown-unknown`
//!   with plain `cargo build --target wasm32-unknown-unknown`.
//!
//! The wire formats are fixed: key codes in [`keys`], modifier bits `1 = shift, 2 = ctrl, 4 = alt`,
//! pointer kinds `0 = down, 1 = move, 2 = up`, buttons `0 = left, 2 = right`.

pub mod abi;
pub mod keys;
pub mod native;

use openstats_charts::render::canvas::{to_binary, to_commands};
use openstats_charts::{render_spec, PlotOptions, Scene};
use openstats_stats::report::{Block, ChartSpec, SessionReport};
use openstats_ui::grid::Mods;
use openstats_ui::workspace::{Button, Outcome, Workspace};

/// Modifier bits as sent by the shell.
pub fn mods_from_bits(bits: u32) -> Mods {
    Mods { shift: bits & 1 != 0, ctrl: bits & 2 != 0, alt: bits & 4 != 0 }
}

/// Encode a scene as the binary canvas command stream.
pub fn frame_bytes(scene: &Scene) -> Vec<u8> {
    to_binary(&to_commands(scene))
}

/// One application window.
#[derive(Debug)]
pub struct Host {
    ws: Workspace,
    frame: Vec<u8>,
    input: Vec<u8>,
    pending: Vec<Outcome>,
    current: Option<Outcome>,
    page_frames: Vec<Vec<u8>>,
}

impl Host {
    /// A new, empty project in a `width × height` window.
    pub fn new(width: f64, height: f64) -> Host {
        Host {
            ws: Workspace::new(width, height),
            frame: Vec::new(),
            input: Vec::new(),
            pending: Vec::new(),
            current: None,
            page_frames: Vec::new(),
        }
    }

    /// The workspace.
    pub fn workspace(&self) -> &Workspace {
        &self.ws
    }

    /// Mutable workspace (tests and embedding code).
    pub fn workspace_mut(&mut self) -> &mut Workspace {
        &mut self.ws
    }

    fn queue(&mut self, o: Outcome) {
        if o != Outcome::None {
            self.pending.push(o);
        }
    }

    /// Resize the window.
    pub fn resize(&mut self, width: f64, height: f64) {
        self.ws.set_size(width.max(200.0), height.max(150.0));
    }

    /// A pointer event: `kind` 0 = down, 1 = move, 2 = up; `button` 0 = left, 2 = right.
    pub fn pointer(&mut self, kind: u32, x: f64, y: f64, button: u32, mods: u32, clicks: u32) {
        match kind {
            0 => {
                let b = if button == 2 { Button::Right } else { Button::Left };
                let o = self.ws.pointer_down(x, y, b, mods_from_bits(mods), clicks.max(1));
                self.queue(o);
            }
            1 => self.ws.pointer_move(x, y),
            _ => self.ws.pointer_up(),
        }
    }

    /// A wheel or trackpad scroll.
    pub fn wheel(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        self.ws.wheel(x, y, dx, dy);
    }

    /// A key press (`code` as in [`keys`]). Unknown codes are ignored.
    pub fn key(&mut self, code: u32, mods: u32) {
        if let Some(k) = keys::decode(code) {
            let o = self.ws.key_down(k, mods_from_bits(mods));
            self.queue(o);
        }
    }

    /// Pasted text from the system clipboard.
    pub fn paste(&mut self, text: &str) {
        let o = self.ws.paste_text(text);
        self.queue(o);
    }

    /// Input-method composition update.
    pub fn ime_update(&mut self, text: &str, caret: usize) {
        self.ws.ime_update(text, caret);
    }

    /// Input-method commit.
    pub fn ime_commit(&mut self, text: &str) {
        self.ws.ime_commit(text);
    }

    /// Input-method composition cancelled.
    pub fn ime_cancel(&mut self) {
        self.ws.ime_cancel();
    }

    /// Where an input method should place its candidate window, if a cell is being edited.
    pub fn ime_anchor(&self) -> Option<(f64, f64, f64)> {
        self.ws.ime_anchor()
    }

    /// Run a command by its text id (see `Command::id`); returns whether the id is known.
    pub fn command(&mut self, id: &str) -> bool {
        match openstats_ui::dialogs::Command::from_id(id) {
            Some(c) => {
                let o = self.ws.execute(c);
                self.queue(o);
                true
            }
            None => false,
        }
    }

    /// Open a project.
    pub fn open_project(&mut self, bytes: &[u8]) {
        let o = self.ws.open_project(bytes);
        self.queue(o);
    }

    /// Import a data file as worksheets.
    pub fn import_data(&mut self, name: &str, bytes: &[u8]) {
        let o = self.ws.import_data(name, bytes);
        self.queue(o);
    }

    /// The project as `.otb` bytes.
    pub fn save_project(&mut self) -> Vec<u8> {
        self.ws.save_project()
    }

    /// Draw the window: the binary canvas command stream, valid until the next call.
    pub fn render(&mut self) -> &[u8] {
        self.frame = frame_bytes(&self.ws.render());
        &self.frame
    }

    /// Menu bar as JSON for native menus, with Undo/Redo enabled state filled in.
    pub fn menu_json(&self) -> String {
        let (undo, redo) = (self.ws.grid().history().can_undo(), self.ws.grid().history().can_redo());
        self.ws.menu().to_json(&|c| match c {
            openstats_ui::dialogs::Command::Undo => undo,
            openstats_ui::dialogs::Command::Redo => redo,
            _ => true,
        })
    }

    /// The active theme's name (`light`, `dark`, `high-contrast`).
    pub fn theme_name(&self) -> &'static str {
        self.ws.theme_kind().name()
    }

    /// The window title: the project title, with the last status or error message if any.
    pub fn title(&self) -> String {
        let t = self.ws.grid().workbook().title.clone();
        let t = if t.is_empty() { "OpenStats".to_string() } else { format!("{t} - OpenStats") };
        if self.ws.status().is_empty() {
            t
        } else {
            format!("{t} — {}", self.ws.status())
        }
    }

    /// The reserved input buffer, for the shell to fill (see [`Host::reserve_input`]).
    pub fn input(&self) -> &[u8] {
        &self.input
    }

    /// Resize the input buffer to `len` zeroed bytes and return it for the shell to fill. The
    /// buffer is owned by the host, so no memory crosses the boundary in the other direction.
    pub fn reserve_input(&mut self, len: usize) -> &mut [u8] {
        self.input.clear();
        self.input.resize(len, 0);
        &mut self.input
    }

    /// Make the next queued outcome current; returns whether there was one.
    pub fn next_outcome(&mut self) -> bool {
        self.page_frames.clear();
        if self.pending.is_empty() {
            self.current = None;
            return false;
        }
        let o = self.pending.remove(0);
        if let Outcome::Print(pages) = &o {
            self.page_frames = pages.iter().map(frame_bytes).collect();
        }
        self.current = Some(o);
        true
    }

    /// The current outcome.
    pub fn outcome(&self) -> Option<&Outcome> {
        self.current.as_ref()
    }

    /// The binary frame of print page `i` of the current outcome.
    pub fn page_frame(&self, i: usize) -> &[u8] {
        self.page_frames.get(i).map_or(&[], Vec::as_slice)
    }
}

/// A graph in its own window.
#[derive(Debug, Clone)]
pub struct GraphHost {
    /// Window title.
    pub name: String,
    spec: ChartSpec,
    frame: Vec<u8>,
}

impl GraphHost {
    /// Build from the `SessionReport` JSON of an [`Outcome::TearOffGraph`].
    pub fn from_json(json: &str) -> Result<GraphHost, String> {
        let r = SessionReport::from_json(json).map_err(|e| e.to_string())?;
        let spec = r
            .sections
            .iter()
            .flat_map(|s| s.blocks.iter())
            .find_map(|b| if let Block::Chart(c) = b { Some(c.clone()) } else { None })
            .ok_or("the report holds no chart")?;
        Ok(GraphHost { name: r.title, spec, frame: Vec::new() })
    }

    /// The chart.
    pub fn spec(&self) -> &ChartSpec {
        &self.spec
    }

    /// Draw the chart at `width × height`.
    pub fn render(&mut self, width: f64, height: f64) -> &[u8] {
        let o = PlotOptions { width: width.max(100.0), height: height.max(80.0), ..Default::default() };
        self.frame = frame_bytes(&render_spec(&self.spec, &o));
        &self.frame
    }
}
