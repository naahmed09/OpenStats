//! The workspace shell: owns the worksheet grid, graphs, Session windows, navigator, menu bar
//! and the open dialog, routes input, and draws everything into one scene.

use super::cmdline::{CommandLine, HistoryKind};
use super::export::{session_docx, session_html};
use super::helpdoc;
use super::keymap::KeyMap;
use super::linked::{self, GraphMeta, GraphRecipe, GraphStyle};
use super::macro_editor::{EditorEvent, MacroEditor, RunState};
use super::navigator::{Button, NavAction, NavEvent, NavModel, Navigator, Target};
use super::project::{graph_stem, GraphItem, ItemId, MacroItem, SessionItem};
use super::session::{SessionHit, SessionView};
use crate::dialogs::assistant::recommend;
use crate::dialogs::menu::{Command, MenuBar, MenuResult, BAR_HEIGHT};
use crate::dialogs::{DialogEvent, DialogView, Registry, Values};
use crate::grid::{GridController, Key, Mods};
use crate::style::{Chrome, ThemeKind};
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Color, Point, Rect, Scene, Stroke, TextStyle, Theme};
use openstats_charts::render::pdf::to_pdf;
use openstats_charts::render::raster::render_png;
use openstats_charts::render::svg::to_svg;
use openstats_charts::{render_spec, PlotOptions};
use openstats_core::io::csv::{write_csv, CsvWriteOptions};
use openstats_core::io::otb;
use openstats_core::locale::Locale;
use openstats_core::Workbook;
use openstats_script::{Env, Library, Machine, MachineConfig, State};
use openstats_stats::report::{Block, ChartSpec, Section, SessionReport};
use std::collections::{BTreeMap, BTreeSet};

const SIZE: f64 = 12.0;
const FORMULA_BAR: f64 = 26.0;
const GAP: f64 = 4.0;
const SHEET_TABS: f64 = 26.0;
const STATUS_BAR: f64 = 22.0;
const CMD_BAR: f64 = 26.0;
const HIST_ROW: f64 = 18.0;
const HIST_ROWS: usize = 10;

/// Lines of the Help > Keyboard Shortcuts section (`Ctrl` is Cmd on macOS).
const SHORTCUTS: &[&str] = &[
    "Ctrl+N  New project            Ctrl+O  Open project            Ctrl+S  Save project",
    "Ctrl+I  Open data file         Ctrl+P  Print Session window",
    "Ctrl+Z  Undo                   Ctrl+Y or Ctrl+Shift+Z  Redo",
    "Ctrl+C / X / V  Copy, cut, paste       Ctrl+A  Select all",
    "Ctrl+F  Go to the formula bar  F2  Edit the active cell          F6  Next pane",
    "F10 or Alt+letter  Open a menu         Ctrl+Shift+D  Next theme (light, dark, high contrast)",
    "Arrow keys, Tab, Enter  Move in the worksheet      Ctrl+arrows  Jump to the edge of the data",
    "Home / End  Row start / end            PageUp / PageDown  Scroll a page",
];

/// What is shown in the upper pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Top {
    /// A Session window.
    Session(ItemId),
    /// A graph.
    Graph(ItemId),
    /// The Macro Editor for a macro, with the Session window below it.
    Macro(ItemId),
}

/// Which pane receives keyboard input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// The Project Navigator.
    Navigator,
    /// The upper pane (Session window or graph).
    Session,
    /// The worksheet grid and formula bar.
    Grid,
    /// The command line.
    Command,
}

/// A file produced for the host to save or download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportFile {
    /// Suggested file name.
    pub name: String,
    /// MIME type.
    pub mime: &'static str,
    /// Contents.
    pub bytes: Vec<u8>,
}

/// What the host must do after an input event.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Nothing.
    None,
    /// Save or download this file.
    File(ExportFile),
    /// Put this text on the system clipboard.
    Clipboard(String),
    /// Print these pages (one scene each).
    Print(Vec<Scene>),
    /// Ask the user for a project file and call [`Workspace::open_project`].
    NeedOpenFile,
    /// Ask the user for a data file (CSV, TSV, XLSX, JSON) and call [`Workspace::import_data`].
    NeedImportFile,
    /// Show a graph in its own window: `json` is a `SessionReport` holding the chart, to be
    /// given to the new window's host (see `openstats-host`).
    TearOffGraph {
        /// Window title (the graph's name).
        name: String,
        /// The chart as `SessionReport` JSON.
        json: String,
    },
    /// An error to show; also available through [`Workspace::status`].
    Error(String),
}

#[derive(Debug, Clone)]
struct OpenDialog {
    id: String,
    view: DialogView,
}

/// File format of an exported graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphFormat {
    Svg,
    Png,
    Pdf,
}

/// A macro run in progress (paused between steps).
#[derive(Debug)]
struct MacroRun {
    sheet: usize,
    machine: Machine,
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    nav: Rect,
    top: Rect,
    fbar: Rect,
    grid: Rect,
    tabs: Rect,
    cmd: Rect,
    status: Rect,
}

/// The whole application window.
#[derive(Debug)]
pub struct Workspace {
    grid: GridController,
    registry: Registry,
    menu: MenuBar,
    nav: Navigator,
    session_view: SessionView,
    graphs: Vec<GraphItem>,
    sessions: Vec<SessionItem>,
    macros: Vec<MacroItem>,
    active_macro: Option<ItemId>,
    macro_editor: MacroEditor,
    macro_run: Option<MacroRun>,
    cmd: CommandLine,
    output_focus: bool,
    next_id: u64,
    active_session: ItemId,
    top: Top,
    focus: Focus,
    dialog: Option<OpenDialog>,
    last_values: BTreeMap<String, Values>,
    /// Worksheet rows brushed in a graph, and the worksheet they belong to.
    brush: BTreeSet<usize>,
    brush_sheet: usize,
    /// Rubber band being dragged over a graph: start, current point, add to the brush.
    graph_drag: Option<(Point, Point, bool)>,
    /// The graph dialog being run: its charts are linked to it when they become graphs.
    pending: Option<(String, Values, usize)>,
    /// Custom keyboard shortcuts.
    keymap: KeyMap,
    /// Related documents: `(path, note)`.
    related: Vec<(String, String)>,
    /// Undo steps kept (Tools > Options).
    undo_steps: usize,
    width: f64,
    height: f64,
    nav_width: f64,
    split: f64,
    status: String,
    theme: Theme,
}

fn file(name: String, mime: &'static str, bytes: Vec<u8>) -> Outcome {
    Outcome::File(ExportFile { name, mime, bytes })
}

impl Workspace {
    /// A new project (one worksheet, one Session window) in a `width × height` window.
    pub fn new(width: f64, height: f64) -> Workspace {
        let registry = Registry::standard();
        let menu = MenuBar::standard(&registry);
        let mut w = Workspace {
            grid: GridController::new(Workbook::new(), 100.0, 100.0),
            registry,
            menu,
            nav: Navigator::new(200.0, 100.0),
            session_view: SessionView::new(100.0, 100.0),
            graphs: Vec::new(),
            sessions: vec![SessionItem {
                id: ItemId(1),
                name: "Session 1".into(),
                report: SessionReport::new("Session 1"),
            }],
            macros: Vec::new(),
            active_macro: None,
            macro_editor: MacroEditor::new(100.0, 100.0),
            macro_run: None,
            cmd: CommandLine::default(),
            output_focus: false,
            next_id: 2,
            active_session: ItemId(1),
            top: Top::Session(ItemId(1)),
            focus: Focus::Grid,
            dialog: None,
            last_values: BTreeMap::new(),
            brush: BTreeSet::new(),
            brush_sheet: 0,
            graph_drag: None,
            pending: None,
            keymap: KeyMap::default(),
            related: Vec::new(),
            undo_steps: 1000,
            width,
            height,
            nav_width: 210.0,
            split: 0.45,
            status: String::new(),
            theme: Theme::light(),
        };
        w.relayout();
        w.refresh_session_view();
        w
    }

    // ---------------------------------------------------------------- accessors

    /// The upper pane (Session window, graph or macro) in window coordinates.
    pub fn upper_pane(&self) -> Rect {
        self.geometry().top
    }

    /// The worksheet grid.
    pub fn grid(&self) -> &GridController {
        &self.grid
    }

    /// Mutable access to the grid (loading data, tests).
    pub fn grid_mut(&mut self) -> &mut GridController {
        &mut self.grid
    }

    /// Macros in Navigator order.
    pub fn macros(&self) -> &[MacroItem] {
        &self.macros
    }

    /// The macro shown in the Macro Editor, if any.
    pub fn active_macro(&self) -> Option<ItemId> {
        self.active_macro
    }

    /// The Macro Editor panel.
    pub fn macro_editor(&self) -> &MacroEditor {
        &self.macro_editor
    }

    /// Mutable Macro Editor (text entry in tests and hosts; call
    /// [`Workspace::macro_text_changed`] after changing its text).
    pub fn macro_editor_mut(&mut self) -> &mut MacroEditor {
        &mut self.macro_editor
    }

    /// Whether a macro run is paused (Step, a breakpoint or the step budget).
    pub fn macro_paused(&self) -> bool {
        self.macro_run.is_some()
    }

    /// The menu bar.
    pub fn menu(&self) -> &MenuBar {
        &self.menu
    }

    /// Mutable access to the menu bar.
    pub fn menu_mut(&mut self) -> &mut MenuBar {
        &mut self.menu
    }

    /// The Navigator.
    pub fn navigator(&self) -> &Navigator {
        &self.nav
    }

    /// The analysis registry.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Graphs in Navigator order.
    pub fn graphs(&self) -> &[GraphItem] {
        &self.graphs
    }

    /// Session windows in Navigator order.
    pub fn sessions(&self) -> &[SessionItem] {
        &self.sessions
    }

    /// The Session window that receives new output.
    pub fn active_session(&self) -> ItemId {
        self.active_session
    }

    /// What the upper pane shows.
    pub fn top(&self) -> Top {
        self.top
    }

    /// The pane with keyboard focus.
    pub fn focus(&self) -> Focus {
        self.focus
    }

    /// Move keyboard focus.
    pub fn set_focus(&mut self, f: Focus) {
        self.focus = f;
    }

    /// The Session view (scrolling, selection, printing).
    pub fn session_view(&self) -> &SessionView {
        &self.session_view
    }

    /// Mutable Session view.
    pub fn session_view_mut(&mut self) -> &mut SessionView {
        &mut self.session_view
    }

    /// The open dialog, if any.
    pub fn dialog(&self) -> Option<&DialogView> {
        self.dialog.as_ref().map(|d| &d.view)
    }

    /// Mutable access to the open dialog.
    pub fn dialog_mut(&mut self) -> Option<&mut DialogView> {
        self.dialog.as_mut().map(|d| &mut d.view)
    }

    /// Last status or error message.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Window size.
    pub fn size(&self) -> (f64, f64) {
        (self.width, self.height)
    }

    /// Resize the window.
    pub fn set_size(&mut self, w: f64, h: f64) {
        self.width = w;
        self.height = h;
        self.relayout();
    }

    /// Fraction of the right-hand height given to the upper pane (0.1 – 0.9).
    pub fn set_split(&mut self, fraction: f64) {
        self.split = fraction.clamp(0.1, 0.9);
        self.relayout();
    }

    fn geometry(&self) -> Geometry {
        let top_y = BAR_HEIGHT;
        let cmd_h = if self.cmd.visible { CMD_BAR } else { 0.0 };
        let status_y = (self.height - STATUS_BAR).max(top_y);
        let bottom = (status_y - cmd_h).max(top_y);
        let body_h = (bottom - top_y).max(0.0);
        let x = self.nav_width + GAP;
        let w = (self.width - x).max(0.0);
        let top_h = (body_h * self.split).floor();
        let fbar_y = top_y + top_h + GAP;
        let grid_y = fbar_y + FORMULA_BAR;
        let grid_h = (bottom - SHEET_TABS - grid_y).max(0.0);
        Geometry {
            nav: Rect::new(0.0, top_y, self.nav_width, body_h),
            top: Rect::new(x, top_y, w, top_h),
            fbar: Rect::new(x, fbar_y, w, FORMULA_BAR),
            grid: Rect::new(x, grid_y, w, grid_h),
            tabs: Rect::new(x, grid_y + grid_h, w, SHEET_TABS.min((bottom - grid_y).max(0.0))),
            cmd: Rect::new(x, bottom, w, (status_y - bottom).max(0.0)),
            status: Rect::new(0.0, status_y, self.width, (self.height - status_y).max(0.0)),
        }
    }

    /// The upper pane's parts: the editor (or the whole pane) and, for a macro, the Session
    /// window beneath it.
    fn top_parts(&self, top: Rect) -> (Rect, Option<Rect>) {
        if !matches!(self.top, Top::Macro(_)) || top.h < 220.0 {
            return (top, None);
        }
        let eh = (top.h * 0.6).floor().clamp(120.0, top.h - 80.0);
        (Rect::new(top.x, top.y, top.w, eh), Some(Rect::new(top.x, top.y + eh, top.w, top.h - eh)))
    }

    /// Worksheet tab rectangles (one per sheet) and the "+" button, in window coordinates.
    fn sheet_tabs(&self, strip: Rect) -> (Vec<Rect>, Rect) {
        let mut x = strip.x + 4.0;
        let mut tabs = Vec::new();
        for w in &self.grid.workbook().worksheets {
            let tw = (text_width(&w.name, SIZE, true) + 26.0).clamp(72.0, 200.0);
            tabs.push(Rect::new(x, strip.y + 2.0, tw, strip.h - 2.0));
            x += tw + 2.0;
        }
        (tabs, Rect::new(x + 2.0, strip.y + 2.0, 26.0, strip.h - 2.0))
    }

    fn relayout(&mut self) {
        let g = self.geometry();
        self.nav.width = g.nav.w;
        self.nav.height = g.nav.h;
        let (ed, out) = self.top_parts(g.top);
        self.macro_editor.set_size(ed.w, ed.h);
        match out {
            Some(o) => self.session_view.set_size(o.w, o.h),
            None => self.session_view.set_size(g.top.w, g.top.h),
        }
        self.grid.set_viewport(g.grid.w, g.grid.h);
    }

    fn model(&self) -> NavModel {
        NavModel {
            worksheets: self.grid.workbook().worksheets.iter().map(|w| w.name.clone()).collect(),
            graphs: self.graphs.iter().map(|g| (g.id, g.name.clone())).collect(),
            sessions: self.sessions.iter().map(|s| (s.id, s.name.clone())).collect(),
            macros: self.macros.iter().map(|m| (m.id, m.name.clone())).collect(),
            related: self.related.iter().map(|(p, _)| p.clone()).collect(),
        }
    }

    fn fail(&mut self, msg: impl Into<String>) -> Outcome {
        let m = msg.into();
        self.status.clone_from(&m);
        Outcome::Error(m)
    }

    fn new_id(&mut self) -> ItemId {
        self.next_id += 1;
        ItemId(self.next_id - 1)
    }

    fn session_index(&self, id: ItemId) -> Option<usize> {
        self.sessions.iter().position(|s| s.id == id)
    }

    fn refresh_session_view(&mut self) {
        let id = match self.top {
            Top::Session(id) => id,
            Top::Graph(_) | Top::Macro(_) => self.active_session,
        };
        self.relayout();
        if let Some(i) = self.session_index(id) {
            let report = self.sessions[i].report.clone();
            self.session_view.set_report(&report);
        }
    }

    // ---------------------------------------------------------------- output

    fn unique_name(taken: &[String], stem: &str) -> String {
        (1..)
            .map(|n| format!("{stem} {n}"))
            .find(|c| !taken.iter().any(|t| t.eq_ignore_ascii_case(c)))
            .unwrap_or_default()
    }

    /// Append `section` to the active Session window; each chart in it also becomes a graph in
    /// the Navigator. The Session window is shown, scrolled to the new output.
    pub fn append_section(&mut self, section: Section) {
        self.push_section(section, true);
    }

    /// Append `section` to the active Session window; `show` switches the upper pane to it
    /// (a running macro keeps its editor on screen, which shows the Session window below it).
    fn push_section(&mut self, section: Section, show: bool) {
        let Some(si) = self.session_index(self.active_session) else { return };
        let sec_idx = self.sessions[si].report.sections.len();
        let charts: Vec<(usize, ChartSpec)> = section
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if let Block::Chart(c) = b { Some((i, c.clone())) } else { None })
            .collect();
        self.sessions[si].report.push(section);
        for (k, (bi, spec)) in charts.into_iter().enumerate() {
            let names: Vec<String> = self.graphs.iter().map(|g| g.name.clone()).collect();
            let name = Self::unique_name(&names, graph_stem(&spec));
            let id = self.new_id();
            let (recipe, rows, fingerprint) = self.link_for(k);
            self.graphs.push(GraphItem {
                id,
                name,
                spec,
                source: Some((self.active_session, sec_idx, bi)),
                recipe,
                auto: true,
                style: GraphStyle::default(),
                labels: [None, None, None],
                rows,
                fingerprint,
            });
        }
        if show || !matches!(self.top, Top::Macro(_)) {
            self.top = Top::Session(self.active_session);
        }
        self.refresh_session_view();
        self.session_view.scroll_to_end();
    }

    /// Scene of graph `gr` with its style applied; `marks` adds the brushed points and the rubber band.
    fn graph_scene_of(&self, gr: &GraphItem, w: f64, h: f64, marks: bool) -> Scene {
        let (theme, viewport) = gr.style.apply(&self.theme);
        let ink = theme.ink;
        let o = PlotOptions { width: w, height: h, theme, viewport, ..Default::default() };
        let mut scene = render_spec(&gr.spec, &o);
        gr.style.draw_notes(&mut scene, ink);
        if marks {
            let chrome = Chrome::of(&self.theme);
            if gr.recipe.as_ref().is_some_and(|r| r.sheet == self.brush_sheet) {
                linked::draw_brush(&mut scene, &gr.rows, &self.brush, chrome.accent);
            }
            if let (Top::Graph(id), Some((a, b, _))) = (self.top, self.graph_drag) {
                if id == gr.id {
                    let r = Rect::new(a.x.min(b.x), a.y.min(b.y), (a.x - b.x).abs(), (a.y - b.y).abs());
                    scene.rect(r, Some(chrome.accent.with_alpha(0.12)), Some(Stroke::new(chrome.accent, 1.0)));
                }
            }
        }
        scene
    }

    /// Recipe, point-to-row map and data fingerprint for chart `k` of the dialog being run.
    fn link_for(&self, k: usize) -> (Option<GraphRecipe>, Vec<usize>, u64) {
        let Some((id, values, sheet)) = self.pending.as_ref() else { return (None, Vec::new(), 0) };
        if !self.registry.is_linkable(id) {
            return (None, Vec::new(), 0);
        }
        let Ok(ws) = self.grid.workbook().sheet(*sheet) else { return (None, Vec::new(), 0) };
        let rows = self.registry.brush_rows(id, ws, values).and_then(|r| r.into_iter().nth(k)).unwrap_or_default();
        let recipe = GraphRecipe { dialog: id.clone(), values: values.clone(), sheet: *sheet, chart: k };
        (Some(recipe), rows, linked::fingerprint(ws, values))
    }

    /// Redraw every automatically updating graph whose data changed, and keep the worksheet's
    /// highlight in step with the brush. Returns the names of the graphs that were redrawn.
    pub fn refresh_graphs(&mut self) -> Vec<String> {
        let mut redrawn = Vec::new();
        for i in 0..self.graphs.len() {
            if !self.graphs[i].auto {
                continue;
            }
            if let Some(name) = self.redraw_graph(i, false) {
                redrawn.push(name);
            }
        }
        self.sync_highlight();
        redrawn
    }

    /// Redraw graph `i` from its recipe when its data changed (or `force`). Returns its name if it was redrawn.
    fn redraw_graph(&mut self, i: usize, force: bool) -> Option<String> {
        let recipe = self.graphs[i].recipe.clone()?;
        let ws = self.grid.workbook().sheet(recipe.sheet).ok()?;
        let fp = linked::fingerprint(ws, &recipe.values);
        if fp == self.graphs[i].fingerprint && !force {
            return None;
        }
        let rows =
            self.registry.brush_rows(&recipe.dialog, ws, &recipe.values).and_then(|r| r.into_iter().nth(recipe.chart));
        let charts = match self.registry.charts(&recipe.dialog, ws, &recipe.values) {
            Some(Ok(c)) => c,
            Some(Err(e)) => {
                self.graphs[i].fingerprint = fp;
                self.status = format!("{} could not be updated: {e}", self.graphs[i].name);
                return None;
            }
            None => return None,
        };
        let mut spec = charts.into_iter().nth(recipe.chart)?;
        let g = &mut self.graphs[i];
        if let Some(t) = &g.labels[0] {
            spec.title.clone_from(t);
        }
        if let Some(t) = &g.labels[1] {
            spec.x_label.clone_from(t);
        }
        if let Some(t) = &g.labels[2] {
            spec.y_label.clone_from(t);
        }
        g.spec = spec.clone();
        g.rows = rows.unwrap_or_default();
        g.fingerprint = fp;
        let (name, source) = (g.name.clone(), g.source);
        if let Some((sid, sec, blk)) = source {
            if let Some(si) = self.session_index(sid) {
                if let Some(Block::Chart(c)) =
                    self.sessions[si].report.sections.get_mut(sec).and_then(|s| s.blocks.get_mut(blk))
                {
                    *c = spec;
                }
            }
        }
        self.refresh_session_view();
        Some(name)
    }

    fn sync_highlight(&mut self) {
        let rows = if self.grid.sheet_index() == self.brush_sheet { self.brush.clone() } else { BTreeSet::new() };
        if &rows != self.grid.highlight() {
            self.grid.set_highlight(rows);
        }
    }

    /// The rows brushed in graphs (0-based worksheet rows) and their worksheet.
    pub fn brushed(&self) -> (usize, &BTreeSet<usize>) {
        (self.brush_sheet, &self.brush)
    }

    /// Replace the brush; the worksheet highlights these rows.
    pub fn set_brush(&mut self, sheet: usize, rows: BTreeSet<usize>) {
        self.brush_sheet = sheet;
        self.brush = rows;
        self.status =
            if self.brush.is_empty() { String::new() } else { format!("{} row(s) brushed", self.brush.len()) };
        self.sync_highlight();
    }

    /// Finish a rubber band or click on graph `id` (scene coordinates) and update the brush.
    fn brush_in_graph(&mut self, id: ItemId, a: Point, b: Point, add: bool) {
        let Some(gr) = self.graphs.iter().find(|g| g.id == id) else { return };
        let Some(sheet) = gr.recipe.as_ref().map(|r| r.sheet) else {
            self.status = "This graph cannot be brushed: it was not drawn from worksheet columns".into();
            return;
        };
        if gr.rows.is_empty() {
            self.status = "Brushing works for scatterplots, time series, marginal and bubble plots".into();
            return;
        }
        let g = self.geometry();
        let scene = self.graph_scene_of(gr, g.top.w, g.top.h, false);
        let index = openstats_charts::core::HitIndex::new(scene.width, scene.height, &scene.targets);
        let (dx, dy) = ((a.x - b.x).abs(), (a.y - b.y).abs());
        let hits: Vec<(usize, usize)> = if dx < 3.0 && dy < 3.0 {
            index.query(b, 8.0).map(|h| vec![(h.series, h.index)]).unwrap_or_default()
        } else {
            index.query_rect(Rect::new(a.x.min(b.x), a.y.min(b.y), dx, dy))
        };
        let rows: BTreeSet<usize> =
            hits.iter().filter(|(s, _)| *s == 0).filter_map(|(_, i)| gr.rows.get(*i).copied()).collect();
        let mut next = if add && self.brush_sheet == sheet { self.brush.clone() } else { BTreeSet::new() };
        next.extend(rows);
        self.set_brush(sheet, next);
    }

    // ---------------------------------------------------------------- dialogs

    /// Open the dialog of analysis `id` over the active worksheet, with the values last used.
    pub fn open_dialog(&mut self, id: &str) -> Result<(), String> {
        let a = self.registry.get(id).ok_or_else(|| format!("unknown command {id}"))?;
        let mut view = DialogView::new(a.spec.clone(), self.grid.worksheet());
        if let Some(v) = self.last_values.get(id) {
            let mut v = v.clone();
            let n = self.grid.data_cols();
            for val in v.0.values_mut() {
                if let crate::dialogs::Value::Vars(cols) = val {
                    cols.retain(|c| *c < n);
                }
            }
            view.set_values(v);
        }
        if self.grid.worksheet().n_cols() > 0 {
            view.select_column(self.grid.active().col.min(self.grid.data_cols().saturating_sub(1)));
        }
        self.menu.close();
        view.theme = self.theme.clone();
        self.dialog = Some(OpenDialog { id: id.to_string(), view });
        Ok(())
    }

    /// Close the dialog without running it.
    pub fn cancel_dialog(&mut self) {
        self.dialog = None;
    }

    /// Run the open dialog's analysis if its input is valid. Output goes to the Session window;
    /// on an error the dialog stays open and shows it.
    fn run_dialog_raw(&mut self) -> Outcome {
        let Some(d) = self.dialog.as_mut() else { return Outcome::None };
        if d.view.try_ok() != DialogEvent::Ok {
            return Outcome::None;
        }
        let (id, values) = (d.id.clone(), d.view.values().clone());
        if id == "graph.edit" {
            return match self.edit_graph(
                values.text("graph"),
                values.text("title"),
                values.text("xlabel"),
                values.text("ylabel"),
            ) {
                Ok(name) => {
                    self.last_values.insert(id, values);
                    self.dialog = None;
                    self.status = format!("Updated {name}");
                    Outcome::None
                }
                Err(e) => {
                    if let Some(d) = self.dialog.as_mut() {
                        d.view.fail(&e);
                    }
                    self.status = e.clone();
                    Outcome::Error(e)
                }
            };
        }
        if crate::dialogs::analyses_workspace::OWNED.contains(&id.as_str()) {
            return match self.owned_dialog(&id, &values) {
                Ok(msg) => {
                    self.last_values.insert(id, values);
                    self.dialog = None;
                    self.status = msg;
                    Outcome::None
                }
                Err(e) => {
                    if let Some(d) = self.dialog.as_mut() {
                        d.view.fail(&e);
                    }
                    self.status = e.clone();
                    Outcome::Error(e)
                }
            };
        }
        if matches!(id.as_str(), "graph.update" | "graph.format" | "graph.annotate" | "graph.brush") {
            return match self.graph_dialog(&id, &values) {
                Ok(msg) => {
                    self.last_values.insert(id, values);
                    self.dialog = None;
                    self.status = msg;
                    Outcome::None
                }
                Err(e) => {
                    if let Some(d) = self.dialog.as_mut() {
                        d.view.fail(&e);
                    }
                    self.status = e.clone();
                    Outcome::Error(e)
                }
            };
        }
        self.pending = Some((id.clone(), values.clone(), self.grid.sheet_index()));
        let ran = self.registry.run(&id, &mut self.grid, &values);
        let ran = ran.map(|section| {
            let title = self.registry.get(&id).map_or(id.clone(), |a| a.spec.title.trim_end_matches("...").to_string());
            (section, title)
        });
        match ran {
            Ok((section, title)) => {
                self.cmd.record(HistoryKind::Dialog, &title, true);
                self.last_values.insert(id, values);
                self.dialog = None;
                self.status.clear();
                self.append_section(section);
                self.pending = None;
                self.grid.set_viewport(self.geometry().grid.w, self.geometry().grid.h);
                Outcome::None
            }
            Err(e) => {
                self.pending = None;
                if let Some(d) = self.dialog.as_mut() {
                    d.view.fail(&e.0);
                }
                self.status = e.0.clone();
                Outcome::Error(e.0)
            }
        }
    }

    /// Change the titles of graph `name` (empty values leave a text unchanged). The graph in the
    /// Navigator and its copy in the Session window are updated together.
    pub fn edit_graph(&mut self, name: &str, title: &str, x_label: &str, y_label: &str) -> Result<String, String> {
        let name = name.trim();
        let idx = if name.is_empty() {
            self.graphs.len().checked_sub(1).ok_or("there are no graphs to edit")?
        } else {
            self.graphs
                .iter()
                .position(|g| g.name == name)
                .ok_or_else(|| format!("there is no graph named \"{name}\" (graphs: {})", self.graph_names()))?
        };
        let apply = |c: &mut ChartSpec| {
            if !title.trim().is_empty() {
                c.title = title.trim().to_string();
            }
            if !x_label.trim().is_empty() {
                c.x_label = x_label.trim().to_string();
            }
            if !y_label.trim().is_empty() {
                c.y_label = y_label.trim().to_string();
            }
        };
        apply(&mut self.graphs[idx].spec);
        for (slot, text) in self.graphs[idx].labels.iter_mut().zip([title, x_label, y_label]) {
            if !text.trim().is_empty() {
                *slot = Some(text.trim().to_string());
            }
        }
        if let Some((sid, sec, blk)) = self.graphs[idx].source {
            if let Some(si) = self.session_index(sid) {
                if let Some(Block::Chart(c)) =
                    self.sessions[si].report.sections.get_mut(sec).and_then(|s| s.blocks.get_mut(blk))
                {
                    apply(c);
                }
            }
        }
        self.refresh_session_view();
        Ok(self.graphs[idx].name.clone())
    }

    fn graph_index(&self, name: &str) -> Result<usize, String> {
        let name = name.trim();
        if name.is_empty() {
            self.graphs.len().checked_sub(1).ok_or_else(|| "there are no graphs".to_string())
        } else {
            self.graphs
                .iter()
                .position(|g| g.name == name)
                .ok_or_else(|| format!("there is no graph named \"{name}\" (graphs: {})", self.graph_names()))
        }
    }

    /// Apply the graph-owned dialogs: update/freeze, format, notes and brushing.
    fn graph_dialog(&mut self, id: &str, v: &Values) -> Result<String, String> {
        match id {
            "graph.brush" => {
                if v.choice("action") == 0 {
                    self.set_brush(self.brush_sheet, BTreeSet::new());
                    return Ok("Brushing cleared".into());
                }
                if self.brush.is_empty() {
                    return Err("nothing is brushed: drag a rectangle over points of a graph first".into());
                }
                let n = self.grid.workbook().sheet(self.brush_sheet).map_err(|e| e.to_string())?.n_rows();
                let data: Vec<f64> = (0..n).map(|r| if self.brush.contains(&r) { 1.0 } else { 0.0 }).collect();
                let sheet = self.grid.sheet_index();
                if sheet != self.brush_sheet {
                    return Err("switch to the worksheet the brushed graph was drawn from first".into());
                }
                let first = self
                    .grid
                    .store_columns(vec![openstats_core::Column::numeric("Brushed", data)])
                    .map_err(|e| e.to_string())?;
                Ok(format!("Stored the brushed rows in {}", self.grid.worksheet().column_label(first)))
            }
            _ => {
                let i = self.graph_index(v.text("graph"))?;
                let name = self.graphs[i].name.clone();
                match id {
                    "graph.update" => match v.choice("action") {
                        0 => {
                            if self.graphs[i].recipe.is_none() {
                                return Err(format!(
                                    "{name} was not drawn from worksheet columns, so it cannot be updated"
                                ));
                            }
                            self.redraw_graph(i, true);
                            Ok(format!("Updated {name}"))
                        }
                        1 => {
                            self.graphs[i].auto = false;
                            Ok(format!("{name} is frozen"))
                        }
                        _ => {
                            self.graphs[i].auto = true;
                            self.redraw_graph(i, false);
                            Ok(format!("{name} follows the data again"))
                        }
                    },
                    "graph.format" => {
                        let st = &mut self.graphs[i].style;
                        if v.flag("reset") {
                            *st = GraphStyle::default();
                        }
                        let f = v.num("font");
                        if f.is_finite() && f != 1.0 {
                            st.font_scale = f;
                        }
                        match v.choice("grid") {
                            1 => st.grid = Some(true),
                            2 => st.grid = Some(false),
                            _ => {}
                        }
                        let (x0, x1, y0, y1) = (v.num("xmin"), v.num("xmax"), v.num("ymin"), v.num("ymax"));
                        for (lo, hi, slot, what) in [(x0, x1, 0, "X"), (y0, y1, 1, "Y")] {
                            match (lo.is_finite(), hi.is_finite()) {
                                (true, true) => {
                                    if lo >= hi {
                                        return Err(format!("the {what} axis minimum must be below its maximum"));
                                    }
                                    if slot == 0 {
                                        st.x_range = Some((lo, hi));
                                    } else {
                                        st.y_range = Some((lo, hi));
                                    }
                                }
                                (false, false) => {}
                                _ => return Err(format!("give both the {what} axis minimum and maximum, or neither")),
                            }
                        }
                        let colors = v.text("colors").trim();
                        if !colors.is_empty() {
                            let mut parsed = Vec::new();
                            for c in colors.split(',') {
                                let c = c.trim().trim_start_matches('#');
                                let hex = u32::from_str_radix(c, 16)
                                    .ok()
                                    .filter(|_| c.len() == 6)
                                    .ok_or_else(|| format!("\"{c}\" is not a colour like #1f77b4"))?;
                                parsed.push(Some(hex));
                            }
                            st.colors = parsed;
                        }
                        Ok(format!("Formatted {name}"))
                    }
                    _ => {
                        let st = &mut self.graphs[i].style;
                        if v.flag("clear") {
                            st.notes.clear();
                            return Ok(format!("Removed the notes of {name}"));
                        }
                        let text = v.text("note").trim();
                        if text.is_empty() {
                            return Err("type the note".into());
                        }
                        st.notes.push(linked::Annotation { x: v.num("x"), y: v.num("y"), text: text.to_string() });
                        Ok(format!("Added a note to {name}"))
                    }
                }
            }
        }
    }

    /// F1: the open dialog's full help in the Session window, or the command search.
    fn help_key(&mut self) -> Outcome {
        let section = self.dialog.as_ref().and_then(|d| self.registry.get(&d.id)).map(helpdoc::dialog_help);
        match section {
            Some(sec) => {
                self.append_section(sec);
                self.status = "The help for this dialog is in the Session window".into();
                Outcome::None
            }
            None if self.dialog.is_some() => Outcome::None,
            None => match self.open_dialog("help.search") {
                Ok(()) => Outcome::None,
                Err(e) => self.fail(e),
            },
        }
    }

    /// Apply the dialogs that act on the project: help search, shortcuts, related documents, options.
    fn owned_dialog(&mut self, id: &str, v: &Values) -> Result<String, String> {
        match id {
            "help.search" => {
                self.append_section(helpdoc::search(&self.registry, v.text("query")));
                Ok(String::new())
            }
            "tools.keys" => {
                if v.flag("reset") {
                    self.keymap.clear();
                    return Ok("Removed all custom shortcuts".into());
                }
                if v.flag("list") {
                    let mut sec = Section::new("Custom shortcuts");
                    if self.keymap.is_empty() {
                        sec.text("There are no custom shortcuts.");
                    }
                    for (chord, cmd) in self.keymap.entries() {
                        sec.text(format!("{chord}  {cmd}"));
                    }
                    self.append_section(sec);
                    return Ok(String::new());
                }
                self.keymap.set(&self.registry, v.text("chord"), v.text("command"))
            }
            "tools.related" => {
                let path = v.text("path").trim().to_string();
                match v.choice("action") {
                    0 => {
                        if path.is_empty() {
                            return Err("type the file name or address".into());
                        }
                        if self.related.iter().any(|(p, _)| *p == path) {
                            return Err(format!("{path} is already listed"));
                        }
                        self.related.push((path.clone(), v.text("note").trim().to_string()));
                        Ok(format!("Added {path} to the related documents"))
                    }
                    1 => match self.related.iter().position(|(p, _)| *p == path) {
                        Some(i) => {
                            self.related.remove(i);
                            Ok(format!("Removed {path}"))
                        }
                        None => Err(format!("{path} is not in the related documents")),
                    },
                    _ => {
                        let mut sec = Section::new("Related documents");
                        if self.related.is_empty() {
                            sec.text("There are no related documents.");
                        }
                        for (p, n) in &self.related {
                            sec.text(if n.is_empty() { p.clone() } else { format!("{p} - {n}") });
                        }
                        self.append_section(sec);
                        Ok(String::new())
                    }
                }
            }
            _ => {
                let n = v.num("undo");
                if !n.is_finite() || n < 1.0 {
                    return Err("keep at least 1 undo step".into());
                }
                self.undo_steps = n as usize;
                self.grid.set_undo_steps(self.undo_steps);
                let (name, loc) = Locale::PRESETS[v.choice("locale").min(Locale::PRESETS.len() - 1)];
                self.grid.set_locale(loc);
                Ok(format!("Keeping up to {} undo steps; regional settings: {name}", self.undo_steps))
            }
        }
    }

    /// The custom shortcuts.
    pub fn keymap(&self) -> &KeyMap {
        &self.keymap
    }

    /// The related documents as `(path, note)`.
    pub fn related(&self) -> &[(String, String)] {
        &self.related
    }

    fn graph_names(&self) -> String {
        self.graphs.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
    }

    fn dialog_event(&mut self, ev: DialogEvent) -> Outcome {
        match ev {
            DialogEvent::Ok => self.run_dialog(),
            DialogEvent::Cancel => {
                self.dialog = None;
                Outcome::None
            }
            _ => Outcome::None,
        }
    }

    fn dialog_origin(&self) -> Point {
        Point::new(
            ((self.width - crate::dialogs::dialog::WIDTH) / 2.0).max(0.0),
            ((self.height - crate::dialogs::dialog::HEIGHT) / 2.0).max(0.0),
        )
    }

    // ---------------------------------------------------------------- commands

    fn command_enabled(&self, c: &Command) -> bool {
        match c {
            Command::Undo => self.grid.history().can_undo(),
            Command::Redo => self.grid.history().can_redo(),
            _ => true,
        }
    }

    fn selected_columns(&self) -> Vec<usize> {
        let n = self.grid.data_cols();
        let mut cols: Vec<usize> = self
            .grid
            .selection()
            .ranges()
            .iter()
            .flat_map(|r| r.c0..=r.c1.min(n.saturating_sub(1)))
            .filter(|c| *c < n)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        if cols.is_empty() && self.grid.active().col < n {
            cols.push(self.grid.active().col);
        }
        cols
    }

    /// Carry out a menu command.
    fn execute_raw(&mut self, cmd: Command) -> Outcome {
        self.menu.close();
        let r = match cmd {
            Command::NewProject => {
                self.reset_project(Workbook::new());
                Ok(Outcome::None)
            }
            Command::OpenProject => Ok(Outcome::NeedOpenFile),
            Command::ImportData => Ok(Outcome::NeedImportFile),
            Command::SaveProject => {
                Ok(file(format!("{}.otb", self.grid.workbook().title), "application/octet-stream", self.save_project()))
            }
            Command::NewWorksheet => self
                .grid
                .add_sheet(None)
                .and_then(|i| self.grid.set_sheet(i))
                .map(|()| Outcome::None)
                .map_err(|e| e.to_string()),
            Command::ExportWorksheet => Ok(self.export_worksheet(self.grid.sheet_index())),
            Command::ExportSession => Ok(self.export_session(self.active_session)),
            Command::ExportSessionHtml => Ok(self.export_session_as(self.active_session, "html")),
            Command::ExportSessionDocx => Ok(self.export_session_as(self.active_session, "docx")),
            Command::PrintSession => Ok(Outcome::Print(self.session_view.print_pages(612.0, 792.0))),
            Command::CommandLine => {
                if self.cmd.visible && self.focus == Focus::Command {
                    self.cmd.visible = false;
                    self.focus = Focus::Grid;
                } else {
                    self.cmd.visible = true;
                    self.focus = Focus::Command;
                }
                self.relayout();
                Ok(Outcome::None)
            }
            Command::HistoryPane => {
                self.cmd.history_open = !self.cmd.history_open;
                if self.cmd.history_open && !self.cmd.visible {
                    self.cmd.visible = true;
                    self.relayout();
                }
                Ok(Outcome::None)
            }
            Command::Undo => self
                .grid
                .undo()
                .map(|l| self.status = format!("Undo: {l}"))
                .map(|()| Outcome::None)
                .map_err(|e| e.to_string()),
            Command::Redo => self
                .grid
                .redo()
                .map(|l| self.status = format!("Redo: {l}"))
                .map(|()| Outcome::None)
                .map_err(|e| e.to_string()),
            Command::MacroNew => {
                let id = self.add_macro(None, "");
                self.open_macro(id);
                Ok(Outcome::None)
            }
            Command::MacroImport => Ok(Outcome::NeedImportFile),
            Command::MacroExport => Ok(self.export_macro()),
            Command::MacroEditor => {
                let id = match self.active_macro.or_else(|| self.macros.first().map(|m| m.id)) {
                    Some(id) => id,
                    None => self.add_macro(None, ""),
                };
                self.open_macro(id);
                Ok(Outcome::None)
            }
            Command::MacroRun => Ok(self.macro_advance(false)),
            Command::MacroStep => Ok(self.macro_advance(true)),
            Command::MacroStop => Ok(self.macro_stop()),
            Command::MacroBreakpoint => {
                if self.active_macro.is_none() {
                    return self.fail("no macro is open; use File > Macros > New Macro");
                }
                let line = self.macro_editor.cursor().0;
                self.macro_editor.toggle_breakpoint(line);
                Ok(Outcome::None)
            }
            Command::Copy | Command::Cut if self.macro_has_keys() => {
                let t = if cmd == Command::Cut { self.macro_editor.cut_text() } else { self.macro_editor.copy_text() };
                if cmd == Command::Cut && t.is_some() {
                    self.macro_text_changed();
                }
                Ok(t.map_or(Outcome::None, Outcome::Clipboard))
            }
            Command::Paste if self.macro_has_keys() => {
                let t = self.grid.clipboard_text().unwrap_or_default().to_string();
                self.macro_editor.insert_text(&t);
                self.macro_text_changed();
                Ok(Outcome::None)
            }
            Command::SelectAll if self.macro_has_keys() => {
                self.macro_editor.select_all();
                Ok(Outcome::None)
            }
            Command::Copy | Command::Cut if self.grid.editor().is_some() => {
                let t = if cmd == Command::Cut { self.grid.editor_cut() } else { self.grid.editor_copy() };
                Ok(t.map_or(Outcome::None, Outcome::Clipboard))
            }
            Command::Copy | Command::Cut => {
                if self.focus == Focus::Session {
                    Ok(Outcome::Clipboard(self.session_view.copy_text()))
                } else if cmd == Command::Cut {
                    self.grid.cut().map(Outcome::Clipboard).map_err(|e| e.to_string())
                } else {
                    self.grid.copy().map(Outcome::Clipboard).map_err(|e| e.to_string())
                }
            }
            Command::Paste => {
                let t = self.grid.clipboard_text().unwrap_or_default().to_string();
                self.grid.paste_text(&t).map(|_| Outcome::None).map_err(|e| e.to_string())
            }
            Command::Delete => self.grid.clear_selection().map(|()| Outcome::None).map_err(|e| e.to_string()),
            Command::SelectAll => {
                if self.focus == Focus::Session {
                    self.session_view.select_all(true);
                } else {
                    self.grid.key_down(Key::Char('a'), Mods::CTRL);
                }
                Ok(Outcome::None)
            }
            Command::FormulaBar => {
                self.focus = Focus::Grid;
                self.grid.begin_edit(None, true);
                Ok(Outcome::None)
            }
            Command::NextPane => {
                self.focus = match self.focus {
                    Focus::Navigator => Focus::Session,
                    Focus::Session => Focus::Grid,
                    Focus::Grid if self.cmd.visible => Focus::Command,
                    Focus::Grid | Focus::Command => Focus::Navigator,
                };
                Ok(Outcome::None)
            }
            Command::ToggleTheme => {
                let k = self.theme_kind().toggled();
                self.set_theme_kind(k);
                Ok(Outcome::None)
            }
            Command::ThemeLight => {
                self.set_theme_kind(ThemeKind::Light);
                Ok(Outcome::None)
            }
            Command::ThemeDark => {
                self.set_theme_kind(ThemeKind::Dark);
                Ok(Outcome::None)
            }
            Command::ThemeHighContrast => {
                self.set_theme_kind(ThemeKind::HighContrast);
                Ok(Outcome::None)
            }
            Command::ShowShortcuts => {
                let mut sec = Section::new("Keyboard shortcuts");
                for line in SHORTCUTS {
                    sec.text((*line).to_string());
                }
                sec.text("F1  Help for the open dialog, or Help > Search Commands".to_string());
                if !self.keymap.is_empty() {
                    sec.text("Custom shortcuts:".to_string());
                    for (chord, cmd) in self.keymap.entries() {
                        sec.text(format!("  {chord}  {cmd}"));
                    }
                }
                self.top = Top::Session(self.active_session);
                self.append_section(sec);
                Ok(Outcome::None)
            }
            Command::About => {
                let mut sec = Section::new("About OpenStats");
                sec.text(format!(
                    "OpenStats {} - open-source statistical software written in Rust.",
                    env!("CARGO_PKG_VERSION")
                ));
                sec.text("Licensed under Apache-2.0 OR MIT.".to_string());
                self.top = Top::Session(self.active_session);
                self.append_section(sec);
                Ok(Outcome::None)
            }
            Command::ClearSession => {
                if let Some(i) = self.session_index(self.active_session) {
                    let name = self.sessions[i].name.clone();
                    self.sessions[i].report = SessionReport::new(name);
                    let id = self.active_session;
                    self.graphs.retain(|g| g.source.is_none_or(|s| s.0 != id));
                    if matches!(self.top, Top::Graph(g) if !self.graphs.iter().any(|x| x.id == g)) {
                        self.top = Top::Session(id);
                    }
                    self.refresh_session_view();
                }
                Ok(Outcome::None)
            }
            Command::InsertRows => self.grid.insert_rows().map(|()| Outcome::None).map_err(|e| e.to_string()),
            Command::DeleteRows => self.grid.delete_rows().map(|()| Outcome::None).map_err(|e| e.to_string()),
            Command::OpenDialog(id) => self.open_dialog(&id).map(|()| Outcome::None),
            Command::Assistant(topic) => {
                let cols = self.selected_columns();
                let rec = recommend(self.grid.worksheet(), topic, &cols);
                let mut s = Section::new(format!("Assistant: {}", topic.label().trim_end_matches("...")));
                s.text(rec.reason.clone());
                self.append_section(s);
                self.open_dialog(&rec.analysis).map(|()| Outcome::None)
            }
        };
        match r {
            Ok(o) => o,
            Err(e) => self.fail(e),
        }
    }

    // ---------------------------------------------------------------- theme

    /// Which built-in theme is active.
    pub fn theme_kind(&self) -> ThemeKind {
        ThemeKind::of(&self.theme)
    }

    /// Switch every component (grid, Navigator, Session window, graphs, menus, dialogs) to the
    /// light or dark theme. Graph thumbnails are redrawn in the new colours.
    pub fn set_theme_kind(&mut self, kind: ThemeKind) {
        self.theme = kind.theme();
        self.grid.theme = self.theme.clone();
        self.nav.theme = self.theme.clone();
        self.session_view.theme = self.theme.clone();
        self.macro_editor.theme = self.theme.clone();
        if let Some(d) = self.dialog.as_mut() {
            d.view.theme = self.theme.clone();
        }
        self.refresh_session_view();
        // The status bar already shows the theme name on the right; no message needed.
        self.status.clear();
    }

    // ---------------------------------------------------------------- project files

    fn reset_project(&mut self, wb: Workbook) {
        let theme = self.grid.theme.clone();
        self.grid = GridController::new(wb, 100.0, 100.0);
        self.grid.theme = theme;
        self.graphs.clear();
        self.keymap = KeyMap::default();
        self.related.clear();
        self.undo_steps = 1000;
        self.grid.set_locale(Locale::default());
        self.brush.clear();
        self.brush_sheet = 0;
        self.graph_drag = None;
        self.pending = None;
        self.sessions =
            vec![SessionItem { id: ItemId(1), name: "Session 1".into(), report: SessionReport::new("Session 1") }];
        self.macros.clear();
        self.active_macro = None;
        self.macro_run = None;
        self.cmd.clear_history();
        self.cmd.set_text("");
        self.output_focus = false;
        self.macro_editor.set_text("");
        self.next_id = 2;
        self.active_session = ItemId(1);
        self.top = Top::Session(ItemId(1));
        self.dialog = None;
        self.nav = {
            let mut n = Navigator::new(self.nav.width, self.nav.height);
            n.theme = self.theme.clone();
            n
        };
        self.relayout();
        self.refresh_session_view();
    }

    /// Serialize the project (worksheets, Session windows and graphs) to `.otb` bytes.
    pub fn save_project(&mut self) -> Vec<u8> {
        let mut keys = Vec::new();
        for (n, s) in self.sessions.iter().enumerate() {
            let mut r = s.report.clone();
            r.title.clone_from(&s.name);
            keys.push((format!("ui.session.{n:04}"), r.to_json()));
        }
        for (n, g) in self.graphs.iter().enumerate() {
            let mut r = SessionReport::new(g.name.clone());
            let mut sec = Section::new("");
            sec.chart(g.spec.clone());
            r.push(sec);
            keys.push((format!("ui.graph.{n:04}"), r.to_json()));
            let meta =
                GraphMeta { recipe: g.recipe.clone(), auto: g.auto, style: g.style.clone(), labels: g.labels.clone() };
            keys.push((format!("ui.graphmeta.{n:04}"), linked::encode(&meta)));
        }
        self.sync_macro_source();
        for (n, m) in self.macros.iter().enumerate() {
            keys.push((format!("ui.macro.{n:04}"), format!("{}\n{}", m.name.replace('\n', " "), m.source)));
        }
        if !self.cmd.entries().is_empty() {
            keys.push(("ui.history".to_string(), self.cmd.save_text()));
        }
        if !self.keymap.is_empty() {
            keys.push(("ui.keys".to_string(), self.keymap.encode()));
        }
        for (n, (p, note)) in self.related.iter().enumerate() {
            keys.push((
                format!("ui.related.{n:04}"),
                format!("{}\t{}", p.replace(['\t', '\n'], " "), note.replace(['\t', '\n'], " ")),
            ));
        }
        if self.undo_steps != 1000 || !self.grid.locale().is_default() {
            keys.push((
                "ui.options".to_string(),
                format!("undo\t{}\nlocale\t{}\n", self.undo_steps, self.grid.locale().preset_index()),
            ));
        }
        for (k, v) in &keys {
            self.grid.wb_mut().metadata.insert(k.clone(), v.clone());
        }
        let bytes = otb::encode(self.grid.workbook());
        for (k, _) in &keys {
            self.grid.wb_mut().metadata.remove(k);
        }
        bytes
    }

    /// Open a project saved by [`Workspace::save_project`] (or any `.otb` workbook).
    pub fn open_project(&mut self, bytes: &[u8]) -> Outcome {
        let mut wb = match otb::decode(bytes) {
            Ok(w) => w,
            Err(e) => return self.fail(format!("cannot open the project: {e}")),
        };
        let take = |wb: &mut Workbook, prefix: &str| -> Vec<SessionReport> {
            let keys: Vec<String> = wb.metadata.keys().filter(|k| k.starts_with(prefix)).cloned().collect();
            keys.iter()
                .filter_map(|k| wb.metadata.remove(k))
                .filter_map(|j| SessionReport::from_json(&j).ok())
                .collect()
        };
        let macro_keys: Vec<String> = wb.metadata.keys().filter(|k| k.starts_with("ui.macro.")).cloned().collect();
        let macros: Vec<(String, String)> = macro_keys
            .iter()
            .filter_map(|k| wb.metadata.remove(k))
            .map(|v| match v.split_once('\n') {
                Some((n, src)) => (n.to_string(), src.to_string()),
                None => (v, String::new()),
            })
            .collect();
        let history = wb.metadata.remove("ui.history");
        let keys_text = wb.metadata.remove("ui.keys");
        let options_text = wb.metadata.remove("ui.options");
        let related_keys: Vec<String> = wb.metadata.keys().filter(|k| k.starts_with("ui.related.")).cloned().collect();
        let related: Vec<(String, String)> = related_keys
            .iter()
            .filter_map(|k| wb.metadata.remove(k))
            .map(|t| match t.split_once('\t') {
                Some((p, n)) => (p.to_string(), n.to_string()),
                None => (t, String::new()),
            })
            .collect();
        let sessions = take(&mut wb, "ui.session.");
        let meta_keys: Vec<String> = wb.metadata.keys().filter(|k| k.starts_with("ui.graphmeta.")).cloned().collect();
        let metas: Vec<GraphMeta> =
            meta_keys.iter().filter_map(|k| wb.metadata.remove(k)).map(|t| linked::decode(&t)).collect();
        let graphs = take(&mut wb, "ui.graph.");
        self.reset_project(wb);
        if !sessions.is_empty() {
            self.sessions.clear();
            for r in sessions {
                let id = self.new_id();
                self.sessions.push(SessionItem { id, name: r.title.clone(), report: r });
            }
            self.active_session = self.sessions[0].id;
            self.top = Top::Session(self.active_session);
        }
        for (name, source) in macros {
            self.add_macro(Some(&name), &source);
        }
        if let Some(h) = history {
            self.cmd.load_text(&h);
        }
        if let Some(k) = keys_text {
            self.keymap = KeyMap::decode(&k, &self.registry);
        }
        self.related = related;
        for line in options_text.iter().flat_map(|t| t.lines()) {
            match line.split_once('\t') {
                Some(("undo", n)) => {
                    if let Ok(n) = n.trim().parse::<usize>() {
                        self.undo_steps = n.max(1);
                        self.grid.set_undo_steps(self.undo_steps);
                    }
                }
                Some(("locale", i)) => {
                    if let Some((_, loc)) = i.trim().parse::<usize>().ok().and_then(|i| Locale::PRESETS.get(i)) {
                        self.grid.set_locale(*loc);
                    }
                }
                _ => {}
            }
        }
        for (n, r) in graphs.into_iter().enumerate() {
            for b in r.sections.iter().flat_map(|s| s.blocks.iter()) {
                if let Block::Chart(c) = b {
                    let id = self.new_id();
                    let meta = metas.get(n).cloned();
                    let (recipe, auto, style, labels) = match meta {
                        Some(m) => (m.recipe, m.auto, m.style, m.labels),
                        None => (None, true, GraphStyle::default(), [None, None, None]),
                    };
                    let (rows, fingerprint) = match &recipe {
                        Some(rc) => match self.grid.workbook().sheet(rc.sheet) {
                            Ok(ws) => (
                                self.registry
                                    .brush_rows(&rc.dialog, ws, &rc.values)
                                    .and_then(|r| r.into_iter().nth(rc.chart))
                                    .unwrap_or_default(),
                                linked::fingerprint(ws, &rc.values),
                            ),
                            Err(_) => (Vec::new(), 0),
                        },
                        None => (Vec::new(), 0),
                    };
                    self.graphs.push(GraphItem {
                        id,
                        name: r.title.clone(),
                        spec: c.clone(),
                        source: None,
                        recipe,
                        auto,
                        style,
                        labels,
                        rows,
                        fingerprint,
                    });
                }
            }
        }
        self.refresh_session_view();
        self.status.clear();
        Outcome::None
    }

    // ---------------------------------------------------------------- navigator actions

    fn export_worksheet(&self, i: usize) -> Outcome {
        match self.grid.workbook().sheet(i) {
            Ok(w) => {
                let loc = self.grid.locale();
                let opts = if loc.decimal == ',' {
                    CsvWriteOptions { delimiter: ';', decimal: ',', ..CsvWriteOptions::default() }
                } else {
                    CsvWriteOptions::default()
                };
                file(format!("{}.csv", w.name), "text/csv", write_csv(w, &opts).into_bytes())
            }
            Err(e) => Outcome::Error(e.to_string()),
        }
    }

    /// Export a Session window as `html` or `docx`.
    fn export_session_as(&self, id: ItemId, format: &str) -> Outcome {
        let Some(i) = self.session_index(id) else { return Outcome::Error("no such Session window".into()) };
        let mut report = self.sessions[i].report.clone();
        report.title.clone_from(&self.sessions[i].name);
        let name = &self.sessions[i].name;
        match format {
            "html" => file(format!("{name}.html"), "text/html", session_html(&report).into_bytes()),
            _ => match session_docx(&report) {
                Ok(b) => file(
                    format!("{name}.docx"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    b,
                ),
                Err(e) => Outcome::Error(format!("cannot write the Word file: {e}")),
            },
        }
    }

    fn export_session(&self, id: ItemId) -> Outcome {
        match self.session_index(id) {
            Some(i) => file(
                format!("{}.txt", self.sessions[i].name),
                "text/plain",
                self.sessions[i].report.render_text().into_bytes(),
            ),
            None => Outcome::Error("no such Session window".into()),
        }
    }

    fn export_graph(&self, id: ItemId, format: GraphFormat) -> Outcome {
        let Some(g) = self.graphs.iter().find(|g| g.id == id) else { return Outcome::Error("no such graph".into()) };
        let scene = self.graph_scene_of(g, 800.0, 500.0, false);
        if format == GraphFormat::Pdf {
            file(format!("{}.pdf", g.name), "application/pdf", to_pdf(&scene))
        } else if format == GraphFormat::Png {
            file(format!("{}.png", g.name), "image/png", render_png(&scene, 1.0))
        } else {
            file(format!("{}.svg", g.name), "image/svg+xml", to_svg(&scene).into_bytes())
        }
    }

    /// The chart of graph `id` as `SessionReport` JSON (one section, one chart), the form a
    /// separate window needs to draw it.
    pub fn graph_json(&self, id: ItemId) -> Option<String> {
        let g = self.graphs.iter().find(|g| g.id == id)?;
        let mut r = SessionReport::new(g.name.clone());
        let mut sec = Section::new("");
        sec.chart(g.spec.clone());
        r.push(sec);
        Some(r.to_json())
    }

    fn tear_off(&mut self, id: ItemId) -> Outcome {
        match (self.graphs.iter().find(|g| g.id == id), self.graph_json(id)) {
            (Some(g), Some(json)) => Outcome::TearOffGraph { name: g.name.clone(), json },
            _ => self.fail("no such graph"),
        }
    }

    /// Import a data file as new worksheets (the first becomes active). `.otb` files are
    /// refused here; open them as projects. Format is taken from the content, then the name.
    pub fn import_data(&mut self, name: &str, bytes: &[u8]) -> Outcome {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".mac") || lower.ends_with(".mtb") {
            return match std::str::from_utf8(bytes) {
                Ok(text) => {
                    let stem = name.rsplit(['/', '\\']).next().unwrap_or(name);
                    let stem = stem.rsplit_once('.').map_or(stem, |(a, _)| a);
                    let id = self.add_macro(Some(stem), text);
                    self.open_macro(id);
                    self.status.clear();
                    Outcome::None
                }
                Err(_) => self.fail(format!("{name} is not a text file")),
            };
        }
        if openstats_core::io::FileFormat::sniff(bytes) == openstats_core::io::FileFormat::Otb {
            return self.fail("that is an OpenStats project; use File > Open Project");
        }
        match openstats_core::io::load_workbook_in(Some(name), bytes, &self.grid.locale()) {
            Ok(wb) => match self.grid.import_sheets(wb.worksheets).and_then(|i| self.grid.set_sheet(i).map(|()| i)) {
                Ok(_) => {
                    self.focus = Focus::Grid;
                    self.status.clear();
                    Outcome::None
                }
                Err(e) => self.fail(e.to_string()),
            },
            Err(e) => self.fail(format!("cannot read {name}: {e}")),
        }
    }

    /// Show the item `t` (worksheet in the lower pane, graph or Session window in the upper).
    pub fn open_item(&mut self, t: Target) -> Outcome {
        match t {
            Target::Worksheet(i) => {
                if let Err(e) = self.grid.set_sheet(i) {
                    return self.fail(e.to_string());
                }
                self.focus = Focus::Grid;
            }
            Target::Graph(id) if self.graphs.iter().any(|g| g.id == id) => {
                self.top = Top::Graph(id);
                self.focus = Focus::Session;
            }
            Target::Related(i) if i < self.related.len() => {
                let (path, note) = self.related[i].clone();
                let mut sec = Section::new("Related document");
                sec.text(path);
                if !note.is_empty() {
                    sec.text(note);
                }
                self.top = Top::Session(self.active_session);
                self.append_section(sec);
            }
            Target::Macro(id) if self.macros.iter().any(|m| m.id == id) => self.open_macro(id),
            Target::Session(id) if self.session_index(id).is_some() => {
                self.active_session = id;
                self.top = Top::Session(id);
                self.focus = Focus::Session;
                self.refresh_session_view();
            }
            _ => {}
        }
        Outcome::None
    }

    fn rename_item(&mut self, t: Target, name: &str) -> Outcome {
        let name = name.trim();
        if name.is_empty() {
            return self.fail("a name cannot be empty");
        }
        let dup = |names: Vec<(ItemId, &str)>, id: ItemId| {
            names.iter().any(|(i, n)| *i != id && n.eq_ignore_ascii_case(name))
        };
        match t {
            Target::Worksheet(i) => {
                if let Err(e) = self.grid.rename_sheet(i, name) {
                    return self.fail(e.to_string());
                }
            }
            Target::Graph(id) => {
                if dup(self.graphs.iter().map(|g| (g.id, g.name.as_str())).collect(), id) {
                    return self.fail(format!("a graph named {name} already exists"));
                }
                if let Some(g) = self.graphs.iter_mut().find(|g| g.id == id) {
                    g.name = name.to_string();
                }
            }
            Target::Macro(id) => {
                if dup(self.macros.iter().map(|g| (g.id, g.name.as_str())).collect(), id) {
                    return self.fail(format!("a macro named {name} already exists"));
                }
                if let Some(m) = self.macros.iter_mut().find(|m| m.id == id) {
                    m.name = name.to_string();
                }
            }
            Target::Session(id) => {
                if dup(self.sessions.iter().map(|g| (g.id, g.name.as_str())).collect(), id) {
                    return self.fail(format!("a Session window named {name} already exists"));
                }
                if let Some(s) = self.sessions.iter_mut().find(|s| s.id == id) {
                    s.name = name.to_string();
                    s.report.title = name.to_string();
                }
                self.refresh_session_view();
            }
            _ => {}
        }
        self.status.clear();
        Outcome::None
    }

    fn delete_item(&mut self, t: Target) -> Outcome {
        match t {
            Target::Worksheet(i) => {
                if let Err(e) = self.grid.remove_sheet(i) {
                    return self.fail(e.to_string());
                }
            }
            Target::Related(i) => {
                if i < self.related.len() {
                    self.related.remove(i);
                }
            }
            Target::Graph(id) => {
                self.graphs.retain(|g| g.id != id);
                if self.top == Top::Graph(id) {
                    self.top = Top::Session(self.active_session);
                }
            }
            Target::Macro(id) => {
                if self.active_macro == Some(id) {
                    self.macro_abandon();
                    self.active_macro = None;
                    self.macro_editor.set_text("");
                    if matches!(self.top, Top::Macro(_)) {
                        self.top = Top::Session(self.active_session);
                        self.relayout();
                        self.refresh_session_view();
                    }
                }
                self.macros.retain(|m| m.id != id);
            }
            Target::Session(id) => {
                if self.sessions.len() == 1 {
                    return self
                        .fail("a project keeps at least one Session window; use Edit > Clear Session Window instead");
                }
                self.sessions.retain(|s| s.id != id);
                self.graphs.retain(|g| g.source.is_none_or(|s| s.0 != id));
                if self.active_session == id {
                    self.active_session = self.sessions[0].id;
                }
                if matches!(self.top, Top::Session(s) if s == id)
                    || matches!(self.top, Top::Graph(g) if !self.graphs.iter().any(|x| x.id == g))
                {
                    self.top = Top::Session(self.active_session);
                }
                self.refresh_session_view();
            }
            _ => {}
        }
        self.nav.select(Target::Project);
        self.status.clear();
        Outcome::None
    }

    /// Carry out a Navigator action on `t`.
    fn navigator_action_raw(&mut self, t: Target, a: NavAction) -> Outcome {
        match a {
            NavAction::Open => self.open_item(t),
            NavAction::Delete => self.delete_item(t),
            NavAction::Rename => {
                let m = self.model();
                self.nav.begin_rename(t, &m);
                Outcome::None
            }
            NavAction::Export => match t {
                Target::Worksheet(i) => self.export_worksheet(i),
                Target::Graph(id) => self.export_graph(id, GraphFormat::Svg),
                Target::Session(id) => self.export_session(id),
                Target::Macro(id) => self.export_macro_by_id(id),
                _ => Outcome::None,
            },
            NavAction::TearOff => match t {
                Target::Graph(id) => self.tear_off(id),
                _ => Outcome::None,
            },
            NavAction::ExportPng => match t {
                Target::Graph(id) => self.export_graph(id, GraphFormat::Png),
                _ => Outcome::None,
            },
            NavAction::ExportPdf => match t {
                Target::Graph(id) => self.export_graph(id, GraphFormat::Pdf),
                _ => Outcome::None,
            },
            NavAction::Duplicate => match t {
                Target::Worksheet(i) => match self.grid.duplicate_sheet(i) {
                    Ok(_) => Outcome::None,
                    Err(e) => self.fail(e.to_string()),
                },
                _ => Outcome::None,
            },
            NavAction::NewWorksheet => match self.grid.add_sheet(None).and_then(|i| self.grid.set_sheet(i)) {
                Ok(()) => Outcome::None,
                Err(e) => self.fail(e.to_string()),
            },
            NavAction::NewRelated => match self.open_dialog("tools.related") {
                Ok(()) => Outcome::None,
                Err(e) => self.fail(e),
            },
            NavAction::NewMacro => {
                let id = self.add_macro(None, "");
                self.open_macro(id);
                Outcome::None
            }
            NavAction::NewSession => {
                let names: Vec<String> = self.sessions.iter().map(|s| s.name.clone()).collect();
                let name = Self::unique_name(&names, "Session");
                let id = self.new_id();
                self.sessions.push(SessionItem { id, name: name.clone(), report: SessionReport::new(name) });
                self.open_item(Target::Session(id))
            }
        }
    }

    fn nav_event(&mut self, ev: NavEvent) -> Outcome {
        match ev {
            NavEvent::None | NavEvent::Select(_) => Outcome::None,
            NavEvent::Open(t) => self.open_item(t),
            NavEvent::Action(t, a) => self.navigator_action(t, a),
            NavEvent::Rename(t, n) => self.rename_item(t, &n),
        }
    }

    // ---------------------------------------------------------------- input

    /// Paste text from the system clipboard into the worksheet (or the open dialog field).
    fn paste_text_raw(&mut self, text: &str) -> Outcome {
        if let Some(d) = self.dialog.as_mut() {
            d.view.insert_text(text);
            return Outcome::None;
        }
        if self.macro_has_keys() {
            self.macro_editor.insert_text(text);
            self.macro_text_changed();
            return Outcome::None;
        }
        if self.focus == Focus::Command && self.cmd.visible {
            self.cmd.editor.insert(text.trim_end_matches(['\r', '\n']));
            return Outcome::None;
        }
        match self.grid.paste_text(text) {
            Ok(_) => Outcome::None,
            Err(e) => self.fail(e.to_string()),
        }
    }

    /// Input-method composition update: `text` is the uncommitted text, `caret` a position in it.
    pub fn ime_update(&mut self, text: &str, caret: usize) {
        if self.dialog.is_none() {
            self.grid.ime_update(text, caret);
        }
    }

    /// Input-method commit of `text` (into the open dialog field, or the cell editor — which
    /// starts an edit of the active cell when none is open).
    fn ime_commit_raw(&mut self, text: &str) {
        if let Some(d) = self.dialog.as_mut() {
            d.view.insert_text(text);
        } else if self.macro_has_keys() {
            self.macro_editor.insert_text(text);
            self.macro_text_changed();
        } else if self.focus == Focus::Command && self.cmd.visible {
            self.cmd.editor.insert(text);
        } else if self.focus == Focus::Grid {
            self.grid.ime_commit(text);
        }
    }

    /// Input-method composition cancelled.
    pub fn ime_cancel(&mut self) {
        self.grid.ime_cancel();
    }

    /// Window position `(x, top, line height)` where an input method should show its candidate
    /// list: at the caret of the cell editor, when one is open.
    pub fn ime_anchor(&self) -> Option<(f64, f64, f64)> {
        let (x, y, h) = self.grid.ime_anchor()?;
        let g = self.geometry();
        Some((g.grid.x + x, g.grid.y + y, h))
    }

    /// Mouse press at window position `(x, y)`.
    pub fn pointer_down(&mut self, x: f64, y: f64, button: Button, mods: Mods, clicks: u32) -> Outcome {
        if self.dialog.is_some() {
            let o = self.dialog_origin();
            let ev = match self.dialog.as_mut() {
                Some(d) => d.view.pointer_down_with(x - o.x, y - o.y, clicks, mods),
                None => DialogEvent::None,
            };
            return self.dialog_event(ev);
        }
        let enabled = |c: &Command| self.command_enabled(c);
        let res = {
            let mut menu = self.menu.clone();
            let r = menu.pointer_down(x, y, &enabled);
            (menu, r)
        };
        self.menu = res.0;
        match res.1 {
            MenuResult::Command(c) => return self.execute(c),
            MenuResult::Consumed => return Outcome::None,
            MenuResult::Pass => {}
        }
        let g = self.geometry();
        let p = Point::new(x, y);
        if g.nav.contains(p) {
            self.focus = Focus::Navigator;
            let m = self.model();
            let ev = self.nav.pointer_down(x - g.nav.x, y - g.nav.y, button, clicks, &m);
            return self.nav_event(ev);
        }
        if g.top.contains(p) {
            self.focus = Focus::Session;
            if matches!(self.top, Top::Macro(_)) {
                let (ed, out) = self.top_parts(g.top);
                if ed.contains(p) {
                    self.output_focus = false;
                    let ev = self.macro_editor.pointer_down(x - ed.x, y - ed.y, clicks, mods.shift);
                    return self.macro_event(ev);
                }
                if let Some(o) = out.filter(|o| o.contains(p)) {
                    self.output_focus = true;
                    if let SessionHit::Chart(sec, b) = self.session_view.pointer_down(x - o.x, y - o.y) {
                        let id = self.active_session;
                        if let Some(gr) = self.graphs.iter().find(|gr| gr.source == Some((id, sec, b))) {
                            self.top = Top::Graph(gr.id);
                            self.relayout();
                        }
                    }
                }
                return Outcome::None;
            }
            if let Top::Graph(_) = self.top {
                let q = Point::new(x - g.top.x, y - g.top.y);
                self.graph_drag = Some((q, q, mods.shift));
                return Outcome::None;
            }
            if let Top::Session(_) = self.top {
                if let SessionHit::Chart(s, b) = self.session_view.pointer_down(x - g.top.x, y - g.top.y) {
                    let id = self.active_session;
                    if let Some(gr) = self.graphs.iter().find(|gr| gr.source == Some((id, s, b))) {
                        self.top = Top::Graph(gr.id);
                    }
                }
            }
            return Outcome::None;
        }
        if g.fbar.contains(p) {
            self.focus = Focus::Grid;
            if x > g.fbar.x + 110.0 && self.grid.editor().is_none() {
                self.grid.begin_edit(None, true);
            }
            return Outcome::None;
        }
        if self.cmd.visible {
            if let Some(rows) = self.history_rows(&g).filter(|_| self.cmd.history_open) {
                if let Some((i, _)) = rows.iter().enumerate().find(|(_, (r, _))| r.contains(p)) {
                    let entry = rows[i].1;
                    if let Some(e) = self.cmd.entries().get(entry).cloned() {
                        if e.kind == HistoryKind::Command {
                            self.cmd.set_text(&e.text);
                        }
                    }
                    self.focus = Focus::Command;
                    return Outcome::None;
                }
            }
            if g.cmd.contains(p) {
                self.focus = Focus::Command;
                let field = Rect::new(g.cmd.x + 58.0, g.cmd.y + 3.0, (g.cmd.w - 66.0).max(0.0), g.cmd.h - 6.0);
                let i = self.cmd.editor.index_at(x - field.x - 6.0, SIZE);
                self.cmd.editor.place_cursor(i, mods.shift);
                return Outcome::None;
            }
        }
        if g.tabs.contains(p) {
            self.focus = Focus::Grid;
            let (tabs, plus) = self.sheet_tabs(g.tabs);
            if plus.contains(p) {
                return match self.grid.add_sheet(None) {
                    Ok(i) => {
                        let _ = self.grid.set_sheet(i);
                        Outcome::None
                    }
                    Err(e) => self.fail(e.to_string()),
                };
            }
            if let Some(i) = tabs.iter().position(|r| r.contains(p)) {
                if let Err(e) = self.grid.set_sheet(i) {
                    return self.fail(e.to_string());
                }
            }
            return Outcome::None;
        }
        if g.grid.contains(p) {
            self.focus = Focus::Grid;
            self.grid.pointer_down(x - g.grid.x, y - g.grid.y, mods, clicks);
        }
        Outcome::None
    }

    /// Mouse moved (with or without a button down).
    pub fn pointer_move(&mut self, x: f64, y: f64) {
        if self.dialog.is_some() {
            return;
        }
        self.menu.pointer_move(x, y);
        let g = self.geometry();
        if let Some((a, _, add)) = self.graph_drag {
            self.graph_drag = Some((a, Point::new(x - g.top.x, y - g.top.y), add));
        }
        if matches!(self.top, Top::Macro(_)) {
            let (ed, _) = self.top_parts(g.top);
            self.macro_editor.pointer_move(x - ed.x, y - ed.y);
        }
        self.grid.pointer_move(x - g.grid.x, y - g.grid.y);
    }

    /// Mouse released.
    pub fn pointer_up(&mut self) {
        if let Some((a, b, add)) = self.graph_drag.take() {
            if let Top::Graph(id) = self.top {
                self.brush_in_graph(id, a, b, add);
            }
        }
        self.macro_editor.pointer_up();
        self.grid.pointer_up();
        self.refresh_graphs();
    }

    /// Wheel or trackpad scroll over `(x, y)`.
    pub fn wheel(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        if let Some(d) = self.dialog.as_mut() {
            d.view.wheel(dy);
            return;
        }
        let g = self.geometry();
        let p = Point::new(x, y);
        if g.top.contains(p) {
            let (ed, _) = self.top_parts(g.top);
            if matches!(self.top, Top::Macro(_)) && ed.contains(p) {
                self.macro_editor.scroll_by(dy);
            } else {
                self.session_view.scroll_by(dy);
            }
        } else if g.grid.contains(p) {
            self.grid.wheel(dx, dy);
        }
    }

    /// Key press. Returns what the host must do (clipboard text, files, printing).
    fn key_down_raw(&mut self, key: Key, mods: Mods) -> Outcome {
        if key == Key::F1 {
            return self.help_key();
        }
        if self.dialog.is_some() {
            let ev = match self.dialog.as_mut() {
                Some(d) => d.view.key_down(key, mods),
                None => DialogEvent::None,
            };
            return self.dialog_event(ev);
        }
        if self.menu.is_open() {
            let enabled = |c: &Command| self.command_enabled(c);
            let mut menu = self.menu.clone();
            let r = menu.key_down(key, &enabled);
            self.menu = menu;
            return match r {
                MenuResult::Command(c) => self.execute(c),
                _ => Outcome::None,
            };
        }
        if mods.ctrl && self.focus == Focus::Grid && self.grid.editor().is_some() {
            if let Key::Char(c) = key {
                match c.to_ascii_lowercase() {
                    'c' => return self.execute(Command::Copy),
                    'x' => return self.execute(Command::Cut),
                    _ => {}
                }
            }
        }
        if mods.ctrl && self.macro_has_keys() {
            if let Key::Char(c) = key {
                match c.to_ascii_lowercase() {
                    'c' => return self.execute(Command::Copy),
                    'x' => return self.execute(Command::Cut),
                    'a' | 'z' | 'y' => {
                        let ev = self.macro_editor.key_down(key, mods);
                        return self.macro_event(ev);
                    }
                    _ => {}
                }
            }
        }
        if self.focus == Focus::Command && self.cmd.visible {
            if let Some(o) = self.command_key(key, mods) {
                return o;
            }
        }
        if !self.keymap.is_empty() && !(self.focus == Focus::Command && self.cmd.visible) {
            let f_key = !matches!(key, Key::Char(_));
            if f_key || (self.grid.editor().is_none() && !self.macro_has_keys()) {
                if let Some(cmd) = self.keymap.lookup(&self.registry, key, mods) {
                    return self.execute(cmd);
                }
            }
        }
        match key {
            Key::F10 => {
                self.menu.open_by_label("File");
                return Outcome::None;
            }
            Key::F6 => return self.execute(Command::NextPane),
            Key::F5 => return self.execute(if mods.shift { Command::MacroStop } else { Command::MacroRun }),
            Key::F8 => return self.execute(Command::MacroStep),
            Key::F9 => return self.execute(Command::MacroBreakpoint),
            _ => {}
        }
        if mods.alt && !mods.ctrl {
            if let Key::Char(c) = key {
                if self.menu.open_by_initial(c) {
                    return Outcome::None;
                }
            }
        }
        if mods.ctrl && self.grid.editor().is_none() {
            if let Key::Char(c) = key {
                let cmd = match c.to_ascii_lowercase() {
                    'n' if mods.shift => Some(Command::NewWorksheet),
                    'n' => Some(Command::NewProject),
                    'o' => Some(Command::OpenProject),
                    's' => Some(Command::SaveProject),
                    'p' => Some(Command::PrintSession),
                    'z' if mods.shift => Some(Command::Redo),
                    'z' => Some(Command::Undo),
                    'y' => Some(Command::Redo),
                    'f' => Some(Command::FormulaBar),
                    'i' => Some(Command::ImportData),
                    'd' if mods.shift => Some(Command::ToggleTheme),
                    'l' => Some(Command::CommandLine),
                    // Cells and report text go to the system clipboard (the host acts on the outcome).
                    'c' if self.focus != Focus::Navigator => Some(Command::Copy),
                    'x' if self.focus == Focus::Grid => Some(Command::Cut),
                    _ => None,
                };
                if let Some(c) = cmd {
                    return self.execute(c);
                }
            }
        }
        match self.focus {
            Focus::Navigator => {
                let m = self.model();
                let ev = self.nav.key_down(key, &m);
                self.nav_event(ev)
            }
            Focus::Session if self.macro_has_keys() => {
                let ev = self.macro_editor.key_down(key, mods);
                self.macro_event(ev)
            }
            Focus::Session => {
                let page = self.session_view.height - 20.0;
                match key {
                    Key::Down => self.session_view.scroll_by(40.0),
                    Key::Up => self.session_view.scroll_by(-40.0),
                    Key::PageDown => self.session_view.scroll_by(page),
                    Key::PageUp => self.session_view.scroll_by(-page),
                    Key::Home => self.session_view.scroll_to_start(),
                    Key::End => self.session_view.scroll_to_end(),
                    Key::Char(c) if mods.ctrl && c.eq_ignore_ascii_case(&'a') => self.session_view.select_all(true),
                    _ => {}
                }
                Outcome::None
            }
            Focus::Grid => {
                self.grid.key_down(key, mods);
                if !self.grid.status().is_empty() {
                    self.status = self.grid.status().to_string();
                }
                Outcome::None
            }
            Focus::Command => Outcome::None,
        }
    }

    /// Keys for the command line. `None` means the key is not for the command line (menu
    /// shortcuts such as Ctrl+S, F6, Alt+letter).
    fn command_key(&mut self, key: Key, mods: Mods) -> Option<Outcome> {
        let ed = &mut self.cmd.editor;
        match key {
            Key::Char(c) if mods.ctrl => match c.to_ascii_lowercase() {
                'a' => {
                    ed.select_all();
                    Some(Outcome::None)
                }
                'c' => Some(ed.selected_text().map_or(Outcome::None, Outcome::Clipboard)),
                'x' => {
                    let t = ed.selected_text();
                    ed.delete_selection();
                    Some(t.map_or(Outcome::None, Outcome::Clipboard))
                }
                _ => None,
            },
            Key::Char(c) if !mods.alt => {
                ed.insert(&c.to_string());
                Some(Outcome::None)
            }
            Key::Backspace => {
                ed.backspace();
                Some(Outcome::None)
            }
            Key::Delete => {
                ed.delete();
                Some(Outcome::None)
            }
            Key::Left => {
                if mods.ctrl {
                    ed.move_word(-1, mods.shift);
                } else {
                    ed.move_cursor(-1, mods.shift);
                }
                Some(Outcome::None)
            }
            Key::Right => {
                if mods.ctrl {
                    ed.move_word(1, mods.shift);
                } else {
                    ed.move_cursor(1, mods.shift);
                }
                Some(Outcome::None)
            }
            Key::Home => {
                ed.cursor_to(false, mods.shift);
                Some(Outcome::None)
            }
            Key::End => {
                ed.cursor_to(true, mods.shift);
                Some(Outcome::None)
            }
            Key::Up => {
                self.cmd.recall_step(true);
                Some(Outcome::None)
            }
            Key::Down => {
                self.cmd.recall_step(false);
                Some(Outcome::None)
            }
            Key::Enter => {
                let text = self.cmd.text().to_string();
                self.cmd.set_text("");
                Some(self.run_command_line(&text))
            }
            Key::Escape => {
                if self.cmd.text().is_empty() {
                    self.focus = Focus::Grid;
                } else {
                    self.cmd.set_text("");
                }
                Some(Outcome::None)
            }
            _ => None,
        }
    }

    /// Run the commands in `text` (the macro language) as one undo step. The text is echoed to
    /// the Session window as `MTB > …`; output, errors and files written by `OUTFILE` follow.
    /// The command goes into the history.
    fn run_command_line_raw(&mut self, text: &str) -> Outcome {
        let text = text.trim_end().to_string();
        if text.trim().is_empty() {
            return Outcome::None;
        }
        let mut echo = Section::new("");
        for l in text.lines() {
            echo.text(format!("MTB > {l}"));
        }
        self.push_section(echo, false);
        self.sync_macro_source();
        let lib = Library { scripts: self.macros.iter().map(|m| (m.name.clone(), m.source.clone())).collect() };
        let first = text.lines().next().unwrap_or("").trim().to_string();
        let result = match Machine::new("Command Line", &text, lib, MachineConfig::default()) {
            Err(e) => Err(e.to_string()),
            Ok(mut machine) => {
                let label: String = format!("Command: {}", first.chars().take(40).collect::<String>());
                let state = self.grid.run_script_step(&label, |wb, hist| {
                    let mut env = Env { wb, hist };
                    let st = machine.run(&mut env, &Default::default());
                    if matches!(st, State::Paused) {
                        machine.stop();
                    }
                    if !matches!(st, State::Finished) {
                        machine.cleanup(&mut env);
                    }
                    st
                });
                for sec in machine.take_output() {
                    self.push_section(sec, false);
                }
                let files = machine.take_files();
                match state {
                    Err(e) => Err(e.to_string()),
                    Ok(State::Failed(e)) => Err(e.to_string()),
                    Ok(State::Paused) => Err("the command ran past the instruction limit and was stopped".to_string()),
                    Ok(_) => Ok(files),
                }
            }
        };
        match result {
            Ok(files) => {
                self.cmd.record(HistoryKind::Command, &text, true);
                self.grid.set_viewport(self.geometry().grid.w, self.geometry().grid.h);
                self.status.clear();
                let mut it = files.into_iter();
                match it.next() {
                    Some((name, body)) => {
                        let extra: Vec<String> = it.map(|(n, _)| n).collect();
                        if !extra.is_empty() {
                            self.status = format!("Only {name} was returned; not saved: {}", extra.join(", "));
                        }
                        file(name, "text/plain", body.into_bytes())
                    }
                    None => Outcome::None,
                }
            }
            Err(e) => {
                self.cmd.record(HistoryKind::Command, &text, false);
                let mut sec = Section::new("Command error");
                for l in e.lines() {
                    sec.warn(l.to_string());
                }
                self.push_section(sec, false);
                self.status = e.lines().next().unwrap_or("").to_string();
                Outcome::Error(self.status.clone())
            }
        }
    }

    /// The command line and History list.
    pub fn command_line(&self) -> &CommandLine {
        &self.cmd
    }

    /// Mutable command line (typing text in tests and hosts).
    pub fn command_line_mut(&mut self) -> &mut CommandLine {
        &mut self.cmd
    }

    // ---------------------------------------------------------------- drawing

    fn pane(&self, s: &mut Scene, r: Rect, inner: &Scene) {
        s.rect(r, Some(self.theme.background), Some(Stroke::new(self.theme.grid, 1.0)));
        s.clip_push(r);
        s.embed(inner, r.x, r.y);
        s.clip_pop();
    }

    fn render_formula_bar(&self, s: &mut Scene, r: Rect) {
        let t = &self.theme;
        let ink = TextStyle::new(SIZE, t.ink);
        let edge = Stroke::new(t.ink.with_alpha(0.4), 1.0);
        let chrome = Chrome::of(t);
        s.rect(r, Some(chrome.bar), Some(Stroke::new(t.grid, 1.0)));
        let name = Rect::new(r.x + 4.0, r.y + 3.0, 92.0, r.h - 6.0);
        s.rect(name, Some(t.plot_background), Some(edge));
        s.text(Point::new(name.x + 6.0, name.y + 14.0), self.grid.name_box(), ink);
        s.text(
            Point::new(name.right() + 8.0, r.y + 17.0),
            "fx".to_string(),
            TextStyle::new(SIZE, t.ink.with_alpha(0.7)).bold(),
        );
        let field = Rect::new(name.right() + 30.0, r.y + 3.0, (r.w - 92.0 - 42.0).max(0.0), r.h - 6.0);
        let editing = self.grid.editor().is_some_and(|e| e.in_formula_bar);
        s.rect(field, Some(t.plot_background), Some(if editing { Stroke::new(chrome.accent, 2.0) } else { edge }));
        let text = self.grid.formula_bar_text();
        s.clip_push(field);
        if let Some((a, b)) = self.grid.editor().and_then(|e| e.selection()) {
            let tw = |n: usize| text_width(&text.chars().take(n).collect::<String>(), SIZE, false);
            s.rect(
                Rect::new(field.x + 6.0 + tw(a), field.y + 3.0, tw(b) - tw(a), field.h - 6.0),
                Some(chrome.selection),
                None,
            );
        }
        s.text(Point::new(field.x + 6.0, field.y + 14.0), text.clone(), ink);
        if let Some(e) = self.grid.editor() {
            let cx =
                field.x + 6.0 + text_width(&text.chars().take(e.display_cursor()).collect::<String>(), SIZE, false);
            s.line(Point::new(cx, field.y + 4.0), Point::new(cx, field.bottom() - 4.0), Stroke::new(t.ink, 1.0));
        }
        s.clip_pop();
    }

    fn render_command_line(&self, s: &mut Scene, r: Rect) {
        let t = &self.theme;
        let chrome = Chrome::of(t);
        let ink = TextStyle::new(SIZE, t.ink);
        let edge = Stroke::new(t.ink.with_alpha(0.4), 1.0);
        s.rect(r, Some(chrome.bar), Some(Stroke::new(t.grid, 1.0)));
        s.text(Point::new(r.x + 8.0, r.y + 17.0), "MTB >".to_string(), TextStyle::new(SIZE, chrome.accent).bold());
        let field = Rect::new(r.x + 58.0, r.y + 3.0, (r.w - 66.0).max(0.0), r.h - 6.0);
        let editing = self.focus == Focus::Command;
        s.rect(field, Some(t.plot_background), Some(if editing { Stroke::new(chrome.accent, 2.0) } else { edge }));
        let ed = &self.cmd.editor;
        let text = ed.display_text();
        s.clip_push(field);
        if let Some((a, b)) = ed.selection() {
            let tw = |n: usize| text_width(&text.chars().take(n).collect::<String>(), SIZE, false);
            s.rect(
                Rect::new(field.x + 6.0 + tw(a), field.y + 3.0, tw(b) - tw(a), field.h - 6.0),
                Some(chrome.selection),
                None,
            );
        }
        s.text(Point::new(field.x + 6.0, field.y + 14.0), text.clone(), ink);
        if editing {
            let cx =
                field.x + 6.0 + text_width(&text.chars().take(ed.display_cursor()).collect::<String>(), SIZE, false);
            s.line(Point::new(cx, field.y + 4.0), Point::new(cx, field.bottom() - 4.0), Stroke::new(t.ink, 1.0));
        }
        s.clip_pop();
    }

    /// Row rectangles of the History list with the index of each entry (newest at the bottom).
    fn history_rows(&self, g: &Geometry) -> Option<Vec<(Rect, usize)>> {
        let n = self.cmd.entries().len();
        if n == 0 {
            return None;
        }
        let shown = n.min(HIST_ROWS);
        let h = shown as f64 * HIST_ROW + 22.0;
        let top = (g.cmd.y - h).max(BAR_HEIGHT);
        Some(
            (0..shown)
                .map(|k| {
                    let idx = n - shown + k;
                    (Rect::new(g.cmd.x, top + 22.0 + k as f64 * HIST_ROW, g.cmd.w, HIST_ROW), idx)
                })
                .collect(),
        )
    }

    fn render_history(&self, s: &mut Scene, g: &Geometry) {
        let t = &self.theme;
        let chrome = Chrome::of(t);
        let n = self.cmd.entries().len();
        let shown = n.min(HIST_ROWS);
        let h = shown as f64 * HIST_ROW + 22.0;
        let top = (g.cmd.y - h).max(BAR_HEIGHT);
        let panel = Rect::new(g.cmd.x, top, g.cmd.w, g.cmd.y - top);
        s.rect(panel, Some(t.plot_background), Some(Stroke::new(chrome.accent, 1.0)));
        s.text(
            Point::new(panel.x + 8.0, top + 15.0),
            if n == 0 { "History (empty)".to_string() } else { format!("History ({n})") },
            TextStyle::new(SIZE, chrome.accent).bold(),
        );
        for (r, idx) in self.history_rows(g).unwrap_or_default() {
            let e = &self.cmd.entries()[idx];
            let tag = match e.kind {
                HistoryKind::Command => "MTB >",
                HistoryKind::Dialog => "Dialog",
                HistoryKind::Macro => "Macro",
            };
            let first = e.text.lines().next().unwrap_or("");
            let more = if e.text.lines().count() > 1 { " ..." } else { "" };
            let fail = if e.ok { "" } else { "   (failed)" };
            s.clip_push(r);
            s.text(Point::new(r.x + 8.0, r.y + 13.0), tag.to_string(), TextStyle::new(SIZE, chrome.muted));
            s.text(Point::new(r.x + 62.0, r.y + 13.0), format!("{first}{more}{fail}"), TextStyle::new(SIZE, t.ink));
            s.clip_pop();
        }
    }

    fn render_sheet_tabs(&self, s: &mut Scene, strip: Rect) {
        let t = &self.theme;
        let chrome = Chrome::of(t);
        s.rect(strip, Some(chrome.bar), Some(Stroke::new(t.grid, 1.0)));
        let (tabs, plus) = self.sheet_tabs(strip);
        s.clip_push(strip);
        for (i, (r, w)) in tabs.iter().zip(&self.grid.workbook().worksheets).enumerate() {
            let active = i == self.grid.sheet_index();
            s.rect(*r, Some(if active { t.plot_background } else { chrome.button }), Some(Stroke::new(t.grid, 1.0)));
            if active {
                s.rect(Rect::new(r.x, r.y, r.w, 3.0), Some(chrome.accent), None);
            }
            let mut st = TextStyle::new(SIZE, if active { t.ink } else { chrome.muted });
            if active {
                st = st.bold();
            }
            s.text(Point::new(r.x + 12.0, r.y + 17.0), w.name.clone(), st);
        }
        s.rect(plus, Some(chrome.button), Some(Stroke::new(t.grid, 1.0)));
        s.text(Point::new(plus.x + 8.0, plus.y + 17.0), "+".to_string(), TextStyle::new(SIZE + 2.0, t.ink).bold());
        s.clip_pop();
    }

    fn render_status_bar(&self, s: &mut Scene, r: Rect) {
        let t = &self.theme;
        let chrome = Chrome::of(t);
        s.rect(r, Some(chrome.bar), Some(Stroke::new(t.grid, 1.0)));
        let ink = TextStyle::new(SIZE, t.ink);
        let title = &self.grid.workbook().title;
        let name = if title.is_empty() { "Untitled" } else { title.as_str() };
        s.text(Point::new(r.x + 10.0, r.y + 15.0), name.to_string(), ink.bold());
        let ws = self.grid.worksheet();
        let right = format!(
            "{}  |  {} rows x {} columns  |  {} theme",
            ws.name,
            self.grid.data_rows(),
            self.grid.data_cols(),
            self.theme_kind().label()
        );
        let rw = text_width(&right, SIZE, false);
        s.text(Point::new(r.right() - rw - 12.0, r.y + 15.0), right, TextStyle::new(SIZE, chrome.muted));
        if !self.status.is_empty() {
            let left = r.x + 20.0 + text_width(name, SIZE, true);
            let room = (r.right() - rw - 30.0 - left).max(0.0);
            s.clip_push(Rect::new(left, r.y, room, r.h));
            s.text(Point::new(left + 8.0, r.y + 15.0), self.status.clone(), ink);
            s.clip_pop();
        }
    }

    /// Draw the whole window.
    pub fn render(&self) -> Scene {
        let g = self.geometry();
        let mut s = Scene::new(self.width, self.height, Chrome::of(&self.theme).desk);
        s.title = "OpenStats".into();
        let model = self.model();
        self.pane(&mut s, g.nav, &self.nav.render(&model));
        let (editor_rect, output_rect) = self.top_parts(g.top);
        if matches!(self.top, Top::Macro(_)) {
            let focused = self.focus == Focus::Session && !self.output_focus;
            self.pane(&mut s, editor_rect, &self.macro_editor.render_focused(focused));
            if let Some(o) = output_rect {
                self.pane(&mut s, o, &self.session_view.render());
            }
        } else {
            let top = match self.top {
                Top::Graph(id) => match self.graphs.iter().find(|x| x.id == id) {
                    Some(gr) => self.graph_scene_of(gr, g.top.w, g.top.h, true),
                    None => self.session_view.render(),
                },
                _ => self.session_view.render(),
            };
            self.pane(&mut s, g.top, &top);
        }
        self.render_formula_bar(&mut s, g.fbar);
        self.pane(&mut s, g.grid, &self.grid.render());
        self.render_sheet_tabs(&mut s, g.tabs);
        if self.cmd.visible {
            self.render_command_line(&mut s, g.cmd);
        }
        self.render_status_bar(&mut s, g.status);
        let focus_rect = match self.focus {
            Focus::Navigator => g.nav,
            Focus::Session => match (self.output_focus, output_rect) {
                (true, Some(o)) => o,
                _ => editor_rect,
            },
            Focus::Grid => g.grid,
            Focus::Command => g.cmd,
        };
        s.rect(focus_rect, None, Some(Stroke::new(Chrome::of(&self.theme).accent, 1.5)));
        if self.cmd.visible && self.cmd.history_open {
            self.render_history(&mut s, &g);
        }
        let enabled = |c: &Command| self.command_enabled(c);
        self.menu.render(&mut s, self.width, &self.theme, &enabled);
        if let Some(d) = &self.dialog {
            s.rect(Rect::new(0.0, 0.0, self.width, self.height), Some(Color::BLACK.with_alpha(0.25)), None);
            let o = self.dialog_origin();
            s.embed(&d.view.render(), o.x, o.y);
        }
        s
    }
}

// ------------------------------------------------------------------------------- macros

impl Workspace {
    /// Add a macro to the project (not opened). `name` defaults to `Macro N`.
    pub fn add_macro(&mut self, name: Option<&str>, source: &str) -> ItemId {
        let names: Vec<String> = self.macros.iter().map(|m| m.name.clone()).collect();
        let name = match name.map(str::trim).filter(|n| !n.is_empty()) {
            Some(n) if !names.iter().any(|t| t.eq_ignore_ascii_case(n)) => n.replace('\n', " "),
            Some(n) => Self::unique_name(&names, n),
            None => Self::unique_name(&names, "Macro"),
        };
        let id = self.new_id();
        self.macros.push(MacroItem { id, name, source: source.replace("\r\n", "\n") });
        id
    }

    /// Show macro `id` in the Macro Editor (the Session window stays visible below it).
    pub fn open_macro(&mut self, id: ItemId) {
        if !self.macros.iter().any(|m| m.id == id) {
            return;
        }
        if self.active_macro != Some(id) {
            self.sync_macro_source();
            self.macro_abandon();
            let text = self.macros.iter().find(|m| m.id == id).map(|m| m.source.clone()).unwrap_or_default();
            self.macro_editor.set_text(&text);
            self.macro_editor.set_run_state(RunState::Idle, "");
            self.active_macro = Some(id);
        }
        self.top = Top::Macro(id);
        self.focus = Focus::Session;
        self.output_focus = false;
        self.relayout();
        self.refresh_session_view();
    }

    /// Whether keys and pasted text go to the Macro Editor.
    fn macro_has_keys(&self) -> bool {
        self.dialog.is_none() && self.focus == Focus::Session && matches!(self.top, Top::Macro(_)) && !self.output_focus
    }

    /// Copy the editor's text into the macro it belongs to.
    fn sync_macro_source(&mut self) {
        if let Some(id) = self.active_macro {
            let text = self.macro_editor.text();
            if let Some(m) = self.macros.iter_mut().find(|m| m.id == id) {
                if m.source != text {
                    m.source = text;
                }
            }
        }
    }

    /// Tell the workspace the editor's text changed (after [`Workspace::macro_editor_mut`] edits):
    /// the macro is updated and a paused run, whose line numbers no longer match, is abandoned.
    pub fn macro_text_changed(&mut self) {
        self.sync_macro_source();
        if self.macro_run.is_some() {
            self.macro_abandon();
            self.status =
                "The macro was edited, so the paused run was abandoned (use Edit > Undo to reverse its changes)".into();
        }
        self.macro_editor.set_exec_line(None);
        self.macro_editor.set_error(None);
        self.macro_editor.set_run_state(RunState::Idle, "");
    }

    fn macro_event(&mut self, ev: EditorEvent) -> Outcome {
        match ev {
            EditorEvent::None | EditorEvent::Breakpoints => Outcome::None,
            EditorEvent::Edited => {
                self.macro_text_changed();
                Outcome::None
            }
            EditorEvent::Run => self.execute(Command::MacroRun),
            EditorEvent::Step => self.execute(Command::MacroStep),
            EditorEvent::Stop => self.execute(Command::MacroStop),
        }
    }

    fn export_macro(&mut self) -> Outcome {
        match self.active_macro {
            Some(id) => {
                self.sync_macro_source();
                self.export_macro_by_id(id)
            }
            None => self.fail("no macro is open; use File > Macros > New Macro"),
        }
    }

    fn export_macro_by_id(&self, id: ItemId) -> Outcome {
        match self.macros.iter().find(|m| m.id == id) {
            Some(m) => file(format!("{}.mac", m.name), "text/plain", m.source.clone().into_bytes()),
            None => Outcome::Error("no such macro".into()),
        }
    }

    /// Drop a paused run, erasing the temporary variables of its active macros.
    fn macro_abandon(&mut self) {
        if let Some(mut run) = self.macro_run.take() {
            run.machine.stop();
            if run.sheet < self.grid.workbook().worksheets.len() {
                let _ = self.grid.run_script_step("Stop macro", |wb, hist| {
                    wb.active = run.sheet;
                    run.machine.cleanup(&mut Env { wb, hist });
                });
            }
        }
        self.macro_editor.set_exec_line(None);
        self.macro_editor.set_run_state(RunState::Idle, "");
    }

    fn macro_stop(&mut self) -> Outcome {
        if self.macro_run.is_none() {
            self.status = "No macro is paused".into();
            return Outcome::None;
        }
        self.macro_abandon();
        self.status = "Macro stopped".into();
        Outcome::None
    }

    /// Run (`step == false`) or step the active macro. The whole call is one undo step.
    fn macro_advance(&mut self, step: bool) -> Outcome {
        let Some(id) = self.active_macro else {
            return self.fail("no macro is open; use File > Macros > New Macro");
        };
        self.sync_macro_source();
        let Some(mi) = self.macros.iter().position(|m| m.id == id) else {
            return self.fail("no such macro");
        };
        let name = self.macros[mi].name.clone();
        if let Some(r) = &self.macro_run {
            if r.sheet != self.grid.sheet_index() {
                return self.fail("the active worksheet changed since the run started; use Stop first");
            }
        }
        if self.macro_run.is_none() {
            let lib = Library { scripts: self.macros.iter().map(|m| (m.name.clone(), m.source.clone())).collect() };
            match Machine::new(&name, &self.macros[mi].source, lib, MachineConfig::default()) {
                Ok(machine) => {
                    self.macro_run = Some(MacroRun { sheet: self.grid.sheet_index(), machine });
                    self.macro_editor.set_error(None);
                }
                Err(e) => return self.macro_failed(&name, &e.to_string(), e.line),
            }
        }
        let Some(mut run) = self.macro_run.take() else { return Outcome::None };
        let breakpoints = self.macro_editor.breakpoints().clone();
        let label = format!("Macro {name}");
        let state = self.grid.run_script_step(&label, |wb, hist| {
            wb.active = run.sheet;
            let mut env = Env { wb, hist };
            let st = if step { run.machine.step(&mut env) } else { run.machine.run(&mut env, &breakpoints) };
            if matches!(st, State::Failed(_) | State::Stopped) {
                run.machine.cleanup(&mut env);
            }
            st
        });
        let state = match state {
            Ok(st) => st,
            Err(e) => return self.fail(e.to_string()),
        };
        let sections = run.machine.take_output();
        let written = sections.len();
        for sec in sections {
            self.push_section(sec, false);
        }
        let steps = run.machine.steps();
        match state {
            State::Paused => {
                let line = run.machine.current_line_in_script();
                let why = run.machine.pause_reason.clone();
                self.macro_editor.set_exec_line(line);
                let at = line.map_or(String::new(), |l| format!(" before line {l}"));
                let msg = format!("Paused{at} ({why})");
                self.macro_editor.set_run_state(RunState::Paused, msg.clone());
                self.status = msg;
                self.macro_run = Some(run);
                Outcome::None
            }
            State::Failed(e) => {
                let line = run.machine.error_line_in_script().unwrap_or(0);
                self.macro_failed(&name, &e.to_string(), line)
            }
            State::Stopped | State::Ready => {
                self.macro_editor.set_exec_line(None);
                self.macro_editor.set_run_state(RunState::Idle, "Stopped");
                Outcome::None
            }
            State::Finished => {
                self.cmd.record(HistoryKind::Macro, &name, true);
                let msg =
                    format!("Finished: {steps} instruction(s), {written} section(s) written to the Session window");
                self.macro_editor.set_exec_line(None);
                self.macro_editor.set_run_state(RunState::Idle, msg.clone());
                self.status = msg;
                Outcome::None
            }
        }
    }

    fn macro_failed(&mut self, name: &str, text: &str, line: usize) -> Outcome {
        self.cmd.record(HistoryKind::Macro, name, false);
        self.macro_run = None;
        self.macro_editor.set_exec_line(None);
        self.macro_editor.set_error(Some((line, text.to_string())));
        self.macro_editor.set_run_state(RunState::Idle, "Failed");
        let mut sec = Section::new(format!("Macro error: {name}"));
        for l in text.lines() {
            sec.warn(l.to_string());
        }
        self.push_section(sec, false);
        self.status = text.lines().next().unwrap_or("").to_string();
        Outcome::Error(self.status.clone())
    }
}

impl Workspace {
    /// A key was pressed; graphs whose data changed are redrawn afterwards.
    pub fn key_down(&mut self, key: Key, mods: Mods) -> Outcome {
        let o = self.key_down_raw(key, mods);
        self.refresh_graphs();
        o
    }

    /// Text was pasted; graphs whose data changed are redrawn afterwards.
    pub fn paste_text(&mut self, text: &str) -> Outcome {
        let o = self.paste_text_raw(text);
        self.refresh_graphs();
        o
    }

    /// IME text was committed; graphs whose data changed are redrawn afterwards.
    pub fn ime_commit(&mut self, text: &str) {
        self.ime_commit_raw(text);
        self.refresh_graphs();
    }

    /// Carry out a menu command; graphs whose data changed are redrawn afterwards.
    pub fn execute(&mut self, cmd: Command) -> Outcome {
        let o = self.execute_raw(cmd);
        self.refresh_graphs();
        o
    }

    /// Run a command-line command; graphs whose data changed are redrawn afterwards.
    pub fn run_command_line(&mut self, text: &str) -> Outcome {
        let o = self.run_command_line_raw(text);
        self.refresh_graphs();
        o
    }

    /// Navigator action; graphs whose data changed are redrawn afterwards.
    pub fn navigator_action(&mut self, t: Target, a: NavAction) -> Outcome {
        let o = self.navigator_action_raw(t, a);
        self.refresh_graphs();
        o
    }

    /// Run the open dialog's analysis; graphs whose data changed are redrawn afterwards.
    pub fn run_dialog(&mut self) -> Outcome {
        let o = self.run_dialog_raw();
        self.refresh_graphs();
        o
    }
}
