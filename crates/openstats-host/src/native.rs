//! Everything a *native* desktop shell needs that is not the operating system itself (std only).
//!
//! The native app has no browser, web view or JavaScript. A shell crate (`openstats-desktop/native`)
//! owns a window, translates its events into calls on [`NativeApp`] and carries out the
//! [`Action`]s that come back: save a file, show a dialog, put text on the clipboard, open a
//! graph window. Frames come from the software rasteriser in `openstats-charts`; the shell can
//! plug the system's fonts in through [`TextPainter`]. Because all of that lives here, the
//! behaviour is covered by ordinary tests; the shell itself is a thin adapter over a windowing
//! library.

use openstats_charts::render::png::{encode, Image};
use openstats_charts::render::raster::{rasterize_with, TextPainter};
use openstats_charts::{render_spec, PlotOptions};
use openstats_core::io::json::{self, Json};
use openstats_ui::style::ThemeKind;
use openstats_ui::workspace::Outcome;

use crate::{GraphHost, Host};

/// Pixel scale used when print pages are written out as PNG files (2 = 144 dpi at 72 pt/inch).
pub const PRINT_SCALE: f64 = 2.0;

/// Something the shell must do on behalf of the workspace.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Ask where to save `bytes` (the suggested `name` carries the extension) and write them.
    SaveFile {
        /// Suggested file name.
        name: String,
        /// File contents.
        bytes: Vec<u8>,
    },
    /// Put this text on the system clipboard.
    Clipboard(String),
    /// Show an open dialog for an OpenStats project (`.otb`); call [`NativeApp::open_project`].
    PickProject,
    /// Show an open dialog for a data file; call [`NativeApp::import_data`].
    PickData,
    /// Open a graph in its own window ([`NativeGraph::from_json`]).
    TearOff {
        /// Window title.
        name: String,
        /// The chart as `SessionReport` JSON.
        json: String,
    },
    /// Write the pages of a print job as PNG files: `base` is the session name, one entry per page.
    SavePages {
        /// Base name (no extension).
        base: String,
        /// PNG bytes of each page.
        pages: Vec<Vec<u8>>,
    },
    /// Show an error message.
    Error(String),
}

/// One node of the native menu tree (parsed from [`Host::menu_json`]).
#[derive(Debug, Clone, PartialEq)]
pub enum MenuNode {
    /// A separator line.
    Separator,
    /// A command. `shortcut` is display text such as `Ctrl+S` (empty when there is none).
    Item {
        /// Menu label.
        label: String,
        /// Command id for [`NativeApp::command`].
        id: String,
        /// Shortcut text.
        shortcut: String,
        /// Whether the item can be chosen.
        enabled: bool,
    },
    /// A submenu.
    Submenu {
        /// Menu label.
        label: String,
        /// Children.
        entries: Vec<MenuNode>,
    },
}

/// Parse the menu JSON (`[{"label":…,"entries":[…]}, …]`) into the top-level menus.
pub fn parse_menu(text: &str) -> Result<Vec<MenuNode>, String> {
    fn node(j: &Json) -> Result<MenuNode, String> {
        if matches!(j.get("separator"), Some(Json::Bool(true))) {
            return Ok(MenuNode::Separator);
        }
        let label = j.get("label").and_then(Json::as_str).ok_or("menu entry without a label")?.to_string();
        if let Some(es) = j.get("entries").and_then(Json::as_array) {
            return Ok(MenuNode::Submenu { label, entries: es.iter().map(node).collect::<Result<_, _>>()? });
        }
        Ok(MenuNode::Item {
            label,
            id: j.get("id").and_then(Json::as_str).ok_or("menu item without an id")?.to_string(),
            shortcut: j.get("shortcut").and_then(Json::as_str).unwrap_or("").to_string(),
            enabled: !matches!(j.get("enabled"), Some(Json::Bool(false))),
        })
    }
    let root = json::parse(text).map_err(|e| e.to_string())?;
    root.as_array().ok_or("the menu is not an array")?.iter().map(node).collect()
}

/// Copy a rasterised image into a `0x00RRGGBB` pixel buffer of `width × height` (the format
/// softbuffer-style surfaces use), cropping or leaving the remainder black when sizes differ.
pub fn to_xrgb(img: &Image, out: &mut [u32], width: usize, height: usize) {
    for y in 0..height {
        for x in 0..width {
            out[y * width + x] = if x < img.width && y < img.height {
                let i = 4 * (y * img.width + x);
                (u32::from(img.rgba[i]) << 16) | (u32::from(img.rgba[i + 1]) << 8) | u32::from(img.rgba[i + 2])
            } else {
                0
            };
        }
    }
}

/// The main workspace window.
#[derive(Debug)]
pub struct NativeApp {
    host: Host,
    scale: f64,
    title: String,
    menu: String,
    last_frame: Option<u64>,
}

impl NativeApp {
    /// A new empty project in a window of `width × height` points at `scale` device pixels per point.
    pub fn new(width: f64, height: f64, scale: f64) -> NativeApp {
        NativeApp {
            host: Host::new(width, height),
            scale: sane_scale(scale),
            title: String::new(),
            menu: String::new(),
            last_frame: None,
        }
    }

    /// The window was resized (points) or moved to a screen with another scale factor.
    pub fn resize(&mut self, width: f64, height: f64, scale: f64) {
        self.scale = sane_scale(scale);
        self.host.resize(width, height);
    }

    /// Device pixels per point.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The underlying host (for tests and embedding code).
    pub fn host(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Pointer event in points: `kind` 0 = down, 1 = move, 2 = up; `button` 0 = left, 2 = right.
    pub fn pointer(&mut self, kind: u32, x: f64, y: f64, button: u32, mods: u32, clicks: u32) -> Vec<Action> {
        self.host.pointer(kind, x, y, button, mods, clicks);
        self.drain()
    }

    /// Scroll by `dx, dy` points at `x, y`.
    pub fn wheel(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        self.host.wheel(x, y, dx, dy);
    }

    /// Key press (codes in [`crate::keys`], modifier bits 1 shift, 2 ctrl/Cmd, 4 alt).
    pub fn key(&mut self, code: u32, mods: u32) -> Vec<Action> {
        self.host.key(code, mods);
        self.drain()
    }

    /// Text from the system clipboard to paste.
    pub fn paste(&mut self, text: &str) -> Vec<Action> {
        self.host.paste(text);
        self.drain()
    }

    /// Input-method composition update (`caret` counts characters).
    pub fn ime_update(&mut self, text: &str, caret: usize) {
        self.host.ime_update(text, caret);
    }

    /// Input-method commit: ends any composition, then inserts `text`.
    pub fn ime_commit(&mut self, text: &str) {
        self.host.ime_cancel();
        if !text.is_empty() {
            self.host.ime_commit(text);
        }
    }

    /// Input-method composition cancelled.
    pub fn ime_cancel(&mut self) {
        self.host.ime_cancel();
    }

    /// Where the input method's candidate window belongs: `(x, y, height)` in points.
    pub fn ime_anchor(&self) -> Option<(f64, f64, f64)> {
        self.host.ime_anchor()
    }

    /// Run a command by id (a menu choice). Returns the actions it caused.
    pub fn command(&mut self, id: &str) -> Vec<Action> {
        self.host.command(id);
        self.drain()
    }

    /// Open a project file the shell has read.
    pub fn open_project(&mut self, bytes: &[u8]) -> Vec<Action> {
        self.host.open_project(bytes);
        self.drain()
    }

    /// Import a data file the shell has read.
    pub fn import_data(&mut self, name: &str, bytes: &[u8]) -> Vec<Action> {
        self.host.import_data(name, bytes);
        self.drain()
    }

    /// Turn the workspace's queued outcomes into shell actions.
    pub fn drain(&mut self) -> Vec<Action> {
        let mut out = Vec::new();
        while self.host.next_outcome() {
            match self.host.outcome().cloned() {
                Some(Outcome::File(f)) => out.push(Action::SaveFile { name: f.name, bytes: f.bytes }),
                Some(Outcome::Clipboard(t)) => out.push(Action::Clipboard(t)),
                Some(Outcome::Print(pages)) => {
                    let pngs = pages.iter().map(|p| encode(&rasterize_with(p, PRINT_SCALE, None))).collect();
                    out.push(Action::SavePages { base: "Session".into(), pages: pngs });
                }
                Some(Outcome::NeedOpenFile) => out.push(Action::PickProject),
                Some(Outcome::NeedImportFile) => out.push(Action::PickData),
                Some(Outcome::TearOffGraph { name, json }) => out.push(Action::TearOff { name, json }),
                Some(Outcome::Error(m)) => out.push(Action::Error(m)),
                Some(Outcome::None) | None => {}
            }
        }
        out
    }

    /// The active theme's name (`light`, `dark`, `high-contrast`), for saving as a setting.
    pub fn theme_name(&self) -> &'static str {
        self.host.workspace().theme_kind().name()
    }

    /// Switch to the theme called `name` (as returned by [`NativeApp::theme_name`]); unknown
    /// names are ignored. Returns whether the theme was applied.
    pub fn set_theme_by_name(&mut self, name: &str) -> bool {
        match ThemeKind::from_name(name) {
            Some(k) => {
                self.host.workspace_mut().set_theme_kind(k);
                true
            }
            None => false,
        }
    }

    /// The window title if it changed since the last call.
    pub fn title_if_changed(&mut self) -> Option<String> {
        let t = self.host.title();
        if t == self.title {
            None
        } else {
            self.title = t.clone();
            Some(t)
        }
    }

    /// The menu tree if it changed since the last call (Undo/Redo enabled state, new analyses).
    pub fn menu_if_changed(&mut self) -> Option<Vec<MenuNode>> {
        let m = self.host.menu_json();
        if m == self.menu {
            return None;
        }
        let parsed = parse_menu(&m).ok()?;
        self.menu = m;
        Some(parsed)
    }

    /// Draw the window only when what it would show differs from the previous call (same
    /// scene, size and scale gives `None`). Rasterising a Retina window is the expensive part, and
    /// a shell asks for redraws on every mouse move and input-method event; skipping identical
    /// frames keeps an idle window at 0 % CPU.
    pub fn frame_if_changed(&mut self, painter: Option<&mut dyn TextPainter>) -> Option<Image> {
        use std::hash::{Hash, Hasher};
        let scene = self.host.workspace().render();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        crate::frame_bytes(&scene).hash(&mut h);
        self.scale.to_bits().hash(&mut h);
        let key = h.finish();
        if self.last_frame == Some(key) {
            return None;
        }
        self.last_frame = Some(key);
        Some(rasterize_with(&scene, self.scale, painter))
    }

    /// Forget the last drawn frame so the next [`NativeApp::frame_if_changed`] draws (after the
    /// window was exposed again or its surface was recreated).
    pub fn invalidate(&mut self) {
        self.last_frame = None;
    }

    /// Draw the window at the current size and scale. `painter` supplies system fonts.
    pub fn frame(&self, painter: Option<&mut dyn TextPainter>) -> Image {
        rasterize_with(&self.host.workspace().render(), self.scale, painter)
    }
}

/// A graph in its own native window.
#[derive(Debug, Clone)]
pub struct NativeGraph {
    graph: GraphHost,
}

impl NativeGraph {
    /// Build from the JSON of an [`Action::TearOff`].
    pub fn from_json(json: &str) -> Result<NativeGraph, String> {
        Ok(NativeGraph { graph: GraphHost::from_json(json)? })
    }

    /// Window title.
    pub fn name(&self) -> &str {
        &self.graph.name
    }

    /// Draw the chart filling `width × height` points at `scale` device pixels per point.
    pub fn frame(&self, width: f64, height: f64, scale: f64, painter: Option<&mut dyn TextPainter>) -> Image {
        let o = PlotOptions { width: width.max(100.0), height: height.max(80.0), ..Default::default() };
        rasterize_with(&render_spec(self.graph.spec(), &o), sane_scale(scale), painter)
    }
}

fn sane_scale(s: f64) -> f64 {
    if s.is_finite() && s >= 0.5 {
        s.min(8.0)
    } else {
        1.0
    }
}
