//! The top-level menu bar: File, Edit, Data, Calc, Stat, Graph, Assistant.

use super::analysis::Registry;
use crate::grid::Key;
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Anchor, Color, Point, Rect, Scene, Stroke, TextStyle, Theme};

/// Topics of the Assistant menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topic {
    /// Compare means or find a relationship in sample data.
    HypothesisTests,
    /// Choose a control chart.
    ControlCharts,
    /// Process capability.
    Capability,
    /// Regression.
    Regression,
    /// Design of experiments.
    Doe,
}

impl Topic {
    /// Stable identifier (used by native menus).
    pub fn id(self) -> &'static str {
        match self {
            Topic::HypothesisTests => "hypothesis",
            Topic::ControlCharts => "control_charts",
            Topic::Capability => "capability",
            Topic::Regression => "regression",
            Topic::Doe => "doe",
        }
    }

    /// The topic with identifier `id`.
    pub fn from_id(id: &str) -> Option<Topic> {
        [Topic::HypothesisTests, Topic::ControlCharts, Topic::Capability, Topic::Regression, Topic::Doe]
            .into_iter()
            .find(|t| t.id() == id)
    }

    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            Topic::HypothesisTests => "Hypothesis Tests...",
            Topic::ControlCharts => "Control Charts...",
            Topic::Capability => "Capability Analysis...",
            Topic::Regression => "Regression...",
            Topic::Doe => "DOE (Design of Experiments)...",
        }
    }
}

/// What a menu entry does. The workspace carries the commands out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// File > New Project.
    NewProject,
    /// File > Open Project (the host supplies the bytes).
    OpenProject,
    /// File > Save Project.
    SaveProject,
    /// File > New Worksheet.
    NewWorksheet,
    /// File > Open Data File (CSV, TSV, XLSX, JSON): the host supplies the bytes.
    ImportData,
    /// File > Export Worksheet as CSV.
    ExportWorksheet,
    /// File > Export Session as text.
    ExportSession,
    /// File > Export Session as HTML.
    ExportSessionHtml,
    /// File > Export Session as a Word document.
    ExportSessionDocx,
    /// File > Print Session Window.
    PrintSession,
    /// Edit > Undo.
    Undo,
    /// Edit > Redo.
    Redo,
    /// Edit > Cut.
    Cut,
    /// Edit > Copy.
    Copy,
    /// Edit > Paste.
    Paste,
    /// Edit > Delete (clear cells).
    Delete,
    /// Edit > Select All.
    SelectAll,
    /// Edit > Clear Session Window.
    ClearSession,
    /// View > Go to Formula Bar (edit the active cell there).
    FormulaBar,
    /// View > Next Pane (cycle focus Navigator, Session, worksheet).
    NextPane,
    /// View > Command Line (show or hide the command line and move the keyboard there).
    CommandLine,
    /// View > History (show or hide the list of commands, dialogs and macros that were run).
    HistoryPane,
    /// View > Theme > Cycle (light, dark, high contrast).
    ToggleTheme,
    /// View > Theme > Light.
    ThemeLight,
    /// View > Theme > Dark.
    ThemeDark,
    /// View > Theme > High Contrast.
    ThemeHighContrast,
    /// Help > Keyboard Shortcuts (written to the Session window).
    ShowShortcuts,
    /// Help > About OpenStats (written to the Session window).
    About,
    /// Data > Insert Rows.
    InsertRows,
    /// Data > Delete Rows.
    DeleteRows,
    /// File > Macros > New Macro.
    MacroNew,
    /// File > Macros > Open Macro File (the host supplies the bytes; `.mac` is routed here).
    MacroImport,
    /// File > Macros > Save Macro As.
    MacroExport,
    /// Tools > Macro Editor (show the active macro).
    MacroEditor,
    /// Tools > Run Macro.
    MacroRun,
    /// Tools > Step Macro.
    MacroStep,
    /// Tools > Stop Macro.
    MacroStop,
    /// Tools > Toggle Breakpoint.
    MacroBreakpoint,
    /// Open the dialog of the analysis with this id.
    OpenDialog(String),
    /// Assistant > topic.
    Assistant(Topic),
}

impl Command {
    /// A stable, unique text identifier (`file.save`, `dialog:stat.basic.ttest1`, …) for native
    /// menus and scripting.
    pub fn id(&self) -> String {
        match self {
            Command::NewProject => "file.new".into(),
            Command::OpenProject => "file.open".into(),
            Command::SaveProject => "file.save".into(),
            Command::NewWorksheet => "file.new_worksheet".into(),
            Command::ImportData => "file.import".into(),
            Command::ExportWorksheet => "file.export_worksheet".into(),
            Command::ExportSession => "file.export_session".into(),
            Command::ExportSessionHtml => "file.export_session_html".into(),
            Command::ExportSessionDocx => "file.export_session_docx".into(),
            Command::PrintSession => "file.print".into(),
            Command::Undo => "edit.undo".into(),
            Command::Redo => "edit.redo".into(),
            Command::Cut => "edit.cut".into(),
            Command::Copy => "edit.copy".into(),
            Command::Paste => "edit.paste".into(),
            Command::Delete => "edit.delete".into(),
            Command::SelectAll => "edit.select_all".into(),
            Command::ClearSession => "edit.clear_session".into(),
            Command::FormulaBar => "view.formula_bar".into(),
            Command::NextPane => "view.next_pane".into(),
            Command::CommandLine => "view.command_line".into(),
            Command::HistoryPane => "view.history".into(),
            Command::ToggleTheme => "view.dark_theme".into(),
            Command::ThemeLight => "view.theme_light".into(),
            Command::ThemeDark => "view.theme_dark".into(),
            Command::ThemeHighContrast => "view.theme_high_contrast".into(),
            Command::ShowShortcuts => "help.shortcuts".into(),
            Command::About => "help.about".into(),
            Command::InsertRows => "data.insert_rows".into(),
            Command::DeleteRows => "data.delete_rows".into(),
            Command::MacroNew => "file.macro_new".into(),
            Command::MacroImport => "file.macro_open".into(),
            Command::MacroExport => "file.macro_save".into(),
            Command::MacroEditor => "tools.macro_editor".into(),
            Command::MacroRun => "tools.run_macro".into(),
            Command::MacroStep => "tools.step_macro".into(),
            Command::MacroStop => "tools.stop_macro".into(),
            Command::MacroBreakpoint => "tools.toggle_breakpoint".into(),
            Command::OpenDialog(id) => format!("dialog:{id}"),
            Command::Assistant(t) => format!("assistant:{}", t.id()),
        }
    }

    /// The command with identifier `id` (the inverse of [`Command::id`]).
    pub fn from_id(id: &str) -> Option<Command> {
        if let Some(d) = id.strip_prefix("dialog:") {
            return Some(Command::OpenDialog(d.to_string()));
        }
        if let Some(t) = id.strip_prefix("assistant:") {
            return Topic::from_id(t).map(Command::Assistant);
        }
        [
            Command::NewProject,
            Command::OpenProject,
            Command::SaveProject,
            Command::NewWorksheet,
            Command::ImportData,
            Command::ExportWorksheet,
            Command::ExportSession,
            Command::ExportSessionHtml,
            Command::ExportSessionDocx,
            Command::PrintSession,
            Command::Undo,
            Command::Redo,
            Command::Cut,
            Command::Copy,
            Command::Paste,
            Command::Delete,
            Command::SelectAll,
            Command::ClearSession,
            Command::FormulaBar,
            Command::NextPane,
            Command::CommandLine,
            Command::HistoryPane,
            Command::ToggleTheme,
            Command::ThemeLight,
            Command::ThemeDark,
            Command::ThemeHighContrast,
            Command::ShowShortcuts,
            Command::About,
            Command::InsertRows,
            Command::DeleteRows,
            Command::MacroNew,
            Command::MacroImport,
            Command::MacroExport,
            Command::MacroEditor,
            Command::MacroRun,
            Command::MacroStep,
            Command::MacroStop,
            Command::MacroBreakpoint,
        ]
        .into_iter()
        .find(|c| c.id() == id)
    }
}

/// One row of a menu.
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    /// A command.
    Item {
        /// Label.
        label: String,
        /// Command carried out when chosen.
        command: Command,
        /// Shortcut text (display only).
        shortcut: &'static str,
    },
    /// A divider.
    Separator,
    /// A nested menu.
    Submenu {
        /// Label.
        label: String,
        /// Rows.
        entries: Vec<Entry>,
    },
}

/// A top-level menu.
#[derive(Debug, Clone, PartialEq)]
pub struct Menu {
    /// Label in the bar.
    pub label: String,
    /// Rows.
    pub entries: Vec<Entry>,
}

fn item(label: &str, command: Command, shortcut: &'static str) -> Entry {
    Entry::Item { label: label.into(), command, shortcut }
}

fn insert(entries: &mut Vec<Entry>, path: &[&'static str], command: Command) {
    match path {
        [] => {}
        [last] => entries.push(Entry::Item { label: (*last).into(), command, shortcut: "" }),
        [head, rest @ ..] => {
            let pos = entries.iter().position(|e| matches!(e, Entry::Submenu { label, .. } if label == head));
            let idx = pos.unwrap_or_else(|| {
                entries.push(Entry::Submenu { label: (*head).into(), entries: Vec::new() });
                entries.len() - 1
            });
            if let Entry::Submenu { entries: sub, .. } = &mut entries[idx] {
                insert(sub, rest, command);
            }
        }
    }
}

/// Where a menu click landed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuResult {
    /// A command was chosen.
    Command(Command),
    /// The menu used the event.
    Consumed,
    /// The event is not for the menu.
    Pass,
}

/// One drop-down panel and its rows (for drawing and hit-testing).
#[derive(Debug, Clone, PartialEq)]
pub struct Panel {
    /// Panel rectangle.
    pub rect: Rect,
    /// Row rectangles and entry indices (separators included).
    pub rows: Vec<(Rect, usize)>,
}

/// The menu bar and its open/hover state.
#[derive(Debug, Clone, PartialEq)]
pub struct MenuBar {
    /// Top-level menus.
    pub menus: Vec<Menu>,
    open: Option<usize>,
    path: Vec<usize>,
    cursor: Option<usize>,
}

/// Height of the bar.
pub const BAR_HEIGHT: f64 = 26.0;
const ROW: f64 = 24.0;
const SEP: f64 = 9.0;
const SIZE: f64 = 12.0;

impl MenuBar {
    /// The whole menu tree as JSON for a native menu bar:
    /// `[{"label":"File","entries":[{"label":"Save Project","id":"file.save","shortcut":"Ctrl+S","enabled":true},
    /// {"separator":true},{"label":"Basic Statistics","entries":[…]}]}]`.
    pub fn to_json(&self, enabled: &dyn Fn(&Command) -> bool) -> String {
        fn esc(s: &str) -> String {
            let mut o = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => o.push_str("\\\""),
                    '\\' => o.push_str("\\\\"),
                    c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                    c => o.push(c),
                }
            }
            o.push('"');
            o
        }
        fn entries(es: &[Entry], enabled: &dyn Fn(&Command) -> bool) -> String {
            let rows: Vec<String> = es
                .iter()
                .map(|e| match e {
                    Entry::Separator => "{\"separator\":true}".to_string(),
                    Entry::Item { label, command, shortcut } => format!(
                        "{{\"label\":{},\"id\":{},\"shortcut\":{},\"enabled\":{}}}",
                        esc(label),
                        esc(&command.id()),
                        esc(shortcut),
                        enabled(command)
                    ),
                    Entry::Submenu { label, entries: sub } => {
                        format!("{{\"label\":{},\"entries\":{}}}", esc(label), entries(sub, enabled))
                    }
                })
                .collect();
            format!("[{}]", rows.join(","))
        }
        let menus: Vec<String> = self
            .menus
            .iter()
            .map(|m| format!("{{\"label\":{},\"entries\":{}}}", esc(&m.label), entries(&m.entries, enabled)))
            .collect();
        format!("[{}]", menus.join(","))
    }

    /// The standard OpenStats menus, with every registry analysis in its place.
    pub fn standard(reg: &Registry) -> MenuBar {
        let file = vec![
            item("New Project", Command::NewProject, "Ctrl+N"),
            item("Open Project...", Command::OpenProject, "Ctrl+O"),
            item("Save Project", Command::SaveProject, "Ctrl+S"),
            Entry::Separator,
            item("New Worksheet", Command::NewWorksheet, "Ctrl+Shift+N"),
            item("Open Data File...", Command::ImportData, "Ctrl+I"),
            Entry::Separator,
            item("Export Worksheet (CSV)...", Command::ExportWorksheet, ""),
            item("Export Session (Text)...", Command::ExportSession, ""),
            item("Export Session (HTML)...", Command::ExportSessionHtml, ""),
            item("Export Session (Word)...", Command::ExportSessionDocx, ""),
            item("Print Session Window...", Command::PrintSession, "Ctrl+P"),
            Entry::Separator,
            Entry::Submenu {
                label: "Macros".into(),
                entries: vec![
                    item("New Macro", Command::MacroNew, ""),
                    item("Open Macro File...", Command::MacroImport, ""),
                    item("Save Macro As...", Command::MacroExport, ""),
                    Entry::Separator,
                    item("Macro Editor", Command::MacroEditor, ""),
                ],
            },
        ];
        let edit = vec![
            item("Undo", Command::Undo, "Ctrl+Z"),
            item("Redo", Command::Redo, "Ctrl+Y / Ctrl+Shift+Z"),
            Entry::Separator,
            item("Cut", Command::Cut, "Ctrl+X"),
            item("Copy", Command::Copy, "Ctrl+C"),
            item("Paste", Command::Paste, "Ctrl+V"),
            item("Delete", Command::Delete, "Del"),
            Entry::Separator,
            item("Select All", Command::SelectAll, "Ctrl+A"),
            item("Clear Session Window", Command::ClearSession, ""),
        ];
        let assistant =
            [Topic::HypothesisTests, Topic::ControlCharts, Topic::Capability, Topic::Regression, Topic::Doe]
                .into_iter()
                .map(|t| item(t.label(), Command::Assistant(t), ""))
                .collect();
        let mut menus: Vec<Menu> = [
            ("File", file),
            ("Edit", edit),
            (
                "Data",
                vec![
                    item("Insert Rows", Command::InsertRows, ""),
                    item("Delete Rows", Command::DeleteRows, ""),
                    Entry::Separator,
                ],
            ),
            ("Calc", vec![]),
            ("Stat", vec![]),
            ("Graph", vec![]),
            (
                "View",
                vec![
                    item("Go to Formula Bar", Command::FormulaBar, "Ctrl+F"),
                    item("Next Pane", Command::NextPane, "F6"),
                    item("Command Line", Command::CommandLine, "Ctrl+L"),
                    item("History", Command::HistoryPane, ""),
                    Entry::Separator,
                    Entry::Submenu {
                        label: "Theme".into(),
                        entries: vec![
                            item("Light", Command::ThemeLight, ""),
                            item("Dark", Command::ThemeDark, ""),
                            item("High Contrast", Command::ThemeHighContrast, ""),
                            Entry::Separator,
                            item("Next Theme", Command::ToggleTheme, "Ctrl+Shift+D"),
                        ],
                    },
                ],
            ),
            (
                "Tools",
                vec![
                    item("Run Macro", Command::MacroRun, "F5"),
                    item("Step Macro", Command::MacroStep, "F8"),
                    item("Stop Macro", Command::MacroStop, "Shift+F5"),
                    item("Toggle Breakpoint", Command::MacroBreakpoint, "F9"),
                ],
            ),
            (
                "Help",
                vec![
                    item("Keyboard Shortcuts", Command::ShowShortcuts, ""),
                    item("About OpenStats", Command::About, ""),
                ],
            ),
            ("Assistant", assistant),
        ]
        .into_iter()
        .map(|(l, entries)| Menu { label: l.into(), entries })
        .collect();
        for a in reg.all() {
            if let Some(top) = a.menu.first() {
                if let Some(m) = menus.iter_mut().find(|m| m.label == *top) {
                    insert(&mut m.entries, &a.menu[1..], Command::OpenDialog(a.spec.id.clone()));
                }
            }
        }
        MenuBar { menus, open: None, path: Vec::new(), cursor: None }
    }

    /// Index of the open top-level menu.
    pub fn open_menu(&self) -> Option<usize> {
        self.open
    }

    /// Label of the open top-level menu.
    pub fn open_label(&self) -> Option<String> {
        self.open.and_then(|i| self.menus.get(i)).map(|m| m.label.clone())
    }

    /// Whether any menu is open.
    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Open the menu labelled `label` (used for Alt-mnemonics and tests).
    pub fn open_by_label(&mut self, label: &str) -> bool {
        match self.menus.iter().position(|m| m.label == label) {
            Some(i) => {
                self.open = Some(i);
                self.path.clear();
                self.cursor = None;
                true
            }
            None => false,
        }
    }

    /// Open the menu whose label starts with `c` (Alt+letter mnemonics; case-insensitive).
    pub fn open_by_initial(&mut self, c: char) -> bool {
        let want = c.to_ascii_lowercase();
        let label = self
            .menus
            .iter()
            .find(|m| m.label.chars().next().is_some_and(|f| f.to_ascii_lowercase() == want))
            .map(|m| m.label.clone());
        label.is_some_and(|l| self.open_by_label(&l))
    }

    /// Close any open menu.
    pub fn close(&mut self) {
        self.open = None;
        self.path.clear();
        self.cursor = None;
    }

    /// Rectangles of the top-level labels.
    pub fn bar_rects(&self) -> Vec<Rect> {
        let mut x = 4.0;
        self.menus
            .iter()
            .map(|m| {
                let w = text_width(&m.label, SIZE, false) + 24.0;
                let r = Rect::new(x, 0.0, w, BAR_HEIGHT);
                x += w;
                r
            })
            .collect()
    }

    fn entries_at(&self, depth: usize) -> Option<&Vec<Entry>> {
        let mut entries = &self.menus.get(self.open?)?.entries;
        for &i in self.path.iter().take(depth) {
            match entries.get(i)? {
                Entry::Submenu { entries: sub, .. } => entries = sub,
                _ => return None,
            }
        }
        Some(entries)
    }

    /// The open panels, outermost first.
    pub fn panels(&self) -> Vec<Panel> {
        let Some(open) = self.open else { return Vec::new() };
        let mut out = Vec::new();
        let mut origin = Point::new(self.bar_rects()[open].x, BAR_HEIGHT);
        for depth in 0..=self.path.len() {
            let Some(entries) = self.entries_at(depth) else { break };
            let width = entries
                .iter()
                .map(|e| match e {
                    Entry::Item { label, shortcut, .. } => {
                        text_width(label, SIZE, false)
                            + if shortcut.is_empty() { 0.0 } else { text_width(shortcut, SIZE, false) + 28.0 }
                    }
                    Entry::Submenu { label, .. } => text_width(label, SIZE, false) + 16.0,
                    Entry::Separator => 0.0,
                })
                .fold(80.0_f64, f64::max)
                + 36.0;
            let mut y = origin.y;
            let mut rows = Vec::new();
            for (i, e) in entries.iter().enumerate() {
                let h = if matches!(e, Entry::Separator) { SEP } else { ROW };
                rows.push((Rect::new(origin.x, y, width, h), i));
                y += h;
            }
            let rect = Rect::new(origin.x, origin.y, width, y - origin.y);
            if let (Some(&sel), true) = (self.path.get(depth), depth < self.path.len()) {
                origin = Point::new(rect.right() - 2.0, rows[sel].0.y);
            }
            out.push(Panel { rect, rows });
        }
        out
    }

    fn enabled_entry(e: &Entry, enabled: &dyn Fn(&Command) -> bool) -> bool {
        match e {
            Entry::Item { command, .. } => enabled(command),
            Entry::Submenu { .. } => true,
            Entry::Separator => false,
        }
    }

    /// Mouse press at `(x, y)`.
    pub fn pointer_down(&mut self, x: f64, y: f64, enabled: &dyn Fn(&Command) -> bool) -> MenuResult {
        let p = Point::new(x, y);
        if (0.0..BAR_HEIGHT).contains(&y) {
            if let Some(i) = self.bar_rects().iter().position(|r| r.contains(p)) {
                if self.open == Some(i) {
                    self.close();
                } else {
                    self.open = Some(i);
                    self.path.clear();
                    self.cursor = None;
                }
                return MenuResult::Consumed;
            }
        }
        let panels = self.panels();
        for (level, panel) in panels.iter().enumerate() {
            if let Some(&(_, idx)) = panel.rows.iter().find(|(r, _)| r.contains(p)) {
                let Some(entries) = self.entries_at(level) else { return MenuResult::Consumed };
                return match &entries[idx] {
                    Entry::Item { command, .. } => {
                        if enabled(command) {
                            let c = command.clone();
                            self.close();
                            MenuResult::Command(c)
                        } else {
                            MenuResult::Consumed
                        }
                    }
                    Entry::Submenu { .. } => {
                        self.path.truncate(level);
                        self.path.push(idx);
                        self.cursor = None;
                        MenuResult::Consumed
                    }
                    Entry::Separator => MenuResult::Consumed,
                };
            }
        }
        if self.is_open() {
            self.close();
            return if y < BAR_HEIGHT { MenuResult::Consumed } else { MenuResult::Pass };
        }
        MenuResult::Pass
    }

    /// Mouse moved: while a menu is open, hovering switches menus and opens submenus.
    pub fn pointer_move(&mut self, x: f64, y: f64) {
        if !self.is_open() {
            return;
        }
        let p = Point::new(x, y);
        if (0.0..BAR_HEIGHT).contains(&y) {
            if let Some(i) = self.bar_rects().iter().position(|r| r.contains(p)) {
                if self.open != Some(i) {
                    self.open = Some(i);
                    self.path.clear();
                    self.cursor = None;
                }
            }
            return;
        }
        let panels = self.panels();
        for (level, panel) in panels.iter().enumerate() {
            if let Some(&(_, idx)) = panel.rows.iter().find(|(r, _)| r.contains(p)) {
                let is_sub = matches!(self.entries_at(level).and_then(|e| e.get(idx)), Some(Entry::Submenu { .. }));
                let is_sep = matches!(self.entries_at(level).and_then(|e| e.get(idx)), Some(Entry::Separator));
                if is_sep {
                    return;
                }
                self.path.truncate(level);
                if is_sub {
                    self.path.push(idx);
                    self.cursor = None;
                } else {
                    self.cursor = Some(idx);
                }
                return;
            }
        }
    }

    fn step(&mut self, dir: isize, enabled: &dyn Fn(&Command) -> bool) {
        let depth = self.path.len();
        let Some(entries) = self.entries_at(depth) else { return };
        let n = entries.len() as isize;
        let mut i = match self.cursor {
            Some(c) => c as isize,
            None if dir > 0 => -1,
            None => n,
        };
        for _ in 0..n {
            i = (i + dir).rem_euclid(n);
            if Self::enabled_entry(&entries[i as usize], enabled) {
                self.cursor = Some(i as usize);
                return;
            }
        }
    }

    /// Keyboard navigation while a menu is open. Returns a command when Enter chooses one.
    pub fn key_down(&mut self, key: Key, enabled: &dyn Fn(&Command) -> bool) -> MenuResult {
        let Some(open) = self.open else { return MenuResult::Pass };
        match key {
            Key::Escape => self.close(),
            Key::Down => self.step(1, enabled),
            Key::Up => self.step(-1, enabled),
            Key::Right => {
                let depth = self.path.len();
                let sub = self
                    .cursor
                    .and_then(|c| self.entries_at(depth).and_then(|e| e.get(c)))
                    .is_some_and(|e| matches!(e, Entry::Submenu { .. }));
                if sub {
                    self.path.push(self.cursor.unwrap_or(0));
                    self.cursor = None;
                    self.step(1, enabled);
                } else if self.path.is_empty() {
                    self.open = Some((open + 1) % self.menus.len());
                    self.cursor = None;
                }
            }
            Key::Left => {
                if let Some(i) = self.path.pop() {
                    self.cursor = Some(i);
                } else {
                    self.open = Some((open + self.menus.len() - 1) % self.menus.len());
                    self.cursor = None;
                }
            }
            Key::Enter => {
                let depth = self.path.len();
                let chosen = self.cursor.and_then(|c| self.entries_at(depth).and_then(|e| e.get(c))).cloned();
                match chosen {
                    Some(Entry::Item { command, .. }) if enabled(&command) => {
                        self.close();
                        return MenuResult::Command(command);
                    }
                    Some(Entry::Submenu { .. }) => {
                        self.path.push(self.cursor.unwrap_or(0));
                        self.cursor = None;
                        self.step(1, enabled);
                    }
                    _ => {}
                }
            }
            _ => return MenuResult::Consumed,
        }
        MenuResult::Consumed
    }

    /// Draw the bar (and any open panels) into `scene` at the top-left corner, `width` wide.
    pub fn render(&self, scene: &mut Scene, width: f64, theme: &Theme, enabled: &dyn Fn(&Command) -> bool) {
        let chrome = crate::style::Chrome::of(theme);
        let bar = chrome.bar;
        let hi = chrome.selection;
        let line = Stroke::new(theme.grid, 1.0);
        scene.rect(Rect::new(0.0, 0.0, width, BAR_HEIGHT), Some(bar), None);
        scene.line(Point::new(0.0, BAR_HEIGHT), Point::new(width, BAR_HEIGHT), line);
        for (i, (m, r)) in self.menus.iter().zip(self.bar_rects()).enumerate() {
            if self.open == Some(i) {
                scene.rect(r, Some(hi), None);
            }
            scene.text(Point::new(r.x + 12.0, 17.0), m.label.clone(), TextStyle::new(SIZE, theme.ink));
        }
        for (level, panel) in self.panels().iter().enumerate() {
            scene.rect(
                Rect::new(panel.rect.x + 2.0, panel.rect.y + 2.0, panel.rect.w, panel.rect.h),
                Some(Color::BLACK.with_alpha(0.15)),
                None,
            );
            scene.rect(panel.rect, Some(theme.background), Some(Stroke::new(theme.ink.with_alpha(0.5), 1.0)));
            let Some(entries) = self.entries_at(level) else { continue };
            for (r, idx) in &panel.rows {
                let highlighted =
                    (level == self.path.len() && self.cursor == Some(*idx)) || self.path.get(level) == Some(idx);
                match &entries[*idx] {
                    Entry::Separator => {
                        let y = r.y + r.h / 2.0;
                        scene.line(Point::new(r.x + 6.0, y), Point::new(r.right() - 6.0, y), line);
                    }
                    Entry::Item { label, command, shortcut } => {
                        let on = enabled(command);
                        if highlighted && on {
                            scene.rect(*r, Some(hi), None);
                        }
                        let color = if on { theme.ink } else { theme.ink.with_alpha(0.4) };
                        scene.text(Point::new(r.x + 18.0, r.y + 16.0), label.clone(), TextStyle::new(SIZE, color));
                        if !shortcut.is_empty() {
                            scene.text(
                                Point::new(r.right() - 10.0, r.y + 16.0),
                                (*shortcut).to_string(),
                                TextStyle::new(SIZE, color.with_alpha(0.7)).anchored(Anchor::End),
                            );
                        }
                    }
                    Entry::Submenu { label, .. } => {
                        if highlighted {
                            scene.rect(*r, Some(hi), None);
                        }
                        scene.text(Point::new(r.x + 18.0, r.y + 16.0), label.clone(), TextStyle::new(SIZE, theme.ink));
                        scene.text(
                            Point::new(r.right() - 10.0, r.y + 16.0),
                            ">".to_string(),
                            TextStyle::new(SIZE, theme.ink).anchored(Anchor::End),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(_: &Command) -> bool {
        true
    }

    #[test]
    fn standard_menu_structure() {
        let bar = MenuBar::standard(&Registry::standard());
        let labels: Vec<&str> = bar.menus.iter().map(|m| m.label.as_str()).collect();
        assert_eq!(labels, ["File", "Edit", "Data", "Calc", "Stat", "Graph", "View", "Tools", "Help", "Assistant"]);
        let stat = &bar.menus[4];
        let subs: Vec<&str> = stat
            .entries
            .iter()
            .filter_map(|e| if let Entry::Submenu { label, .. } = e { Some(label.as_str()) } else { None })
            .collect();
        assert_eq!(
            subs,
            [
                "Basic Statistics",
                "Regression",
                "ANOVA",
                "Control Charts",
                "Quality Tools",
                "DOE",
                "Multivariate",
                "Predictive Analytics",
                "Time Series",
                "Tables",
                "EDA",
                "Resampling",
                "Nonparametrics",
                "Reliability/Survival",
                "Power and Sample Size"
            ]
        );
        assert!(
            matches!(&bar.menus[3].entries[0], Entry::Item { command: Command::OpenDialog(id), .. } if id == "calc.calculator")
        );
    }

    #[test]
    fn click_open_navigate_and_choose() {
        let mut bar = MenuBar::standard(&Registry::standard());
        let r = bar.bar_rects()[4];
        assert_eq!(bar.pointer_down(r.x + 3.0, 5.0, &all), MenuResult::Consumed);
        assert_eq!(bar.open_menu(), Some(4));
        let p0 = bar.panels();
        assert_eq!(p0.len(), 1);
        // Hover the first submenu: its panel opens to the right.
        let row = p0[0].rows[0].0;
        bar.pointer_move(row.x + 5.0, row.y + 5.0);
        let p1 = bar.panels();
        assert_eq!(p1.len(), 2);
        assert!(p1[1].rect.x > p1[0].rect.x);
        // Choose the first item of the submenu.
        let item_row = p1[1].rows[0].0;
        let res = bar.pointer_down(item_row.x + 5.0, item_row.y + 5.0, &all);
        assert_eq!(res, MenuResult::Command(Command::OpenDialog("stat.basic.descriptive".into())));
        assert!(!bar.is_open());
        // Clicking away closes and passes the click on.
        bar.open_by_label("File");
        assert_eq!(bar.pointer_down(500.0, 300.0, &all), MenuResult::Pass);
        assert!(!bar.is_open());
    }

    #[test]
    fn keyboard_and_enabled_state() {
        let mut bar = MenuBar::standard(&Registry::standard());
        let no_undo = |c: &Command| *c != Command::Undo;
        bar.open_by_label("Edit");
        // Undo is disabled, so the first Down lands on Redo; the next skips the separator.
        bar.key_down(Key::Down, &no_undo);
        bar.key_down(Key::Down, &no_undo);
        assert_eq!(bar.key_down(Key::Enter, &no_undo), MenuResult::Command(Command::Cut));
        bar.open_by_label("Edit");
        bar.key_down(Key::Down, &no_undo);
        assert_eq!(bar.key_down(Key::Enter, &no_undo), MenuResult::Command(Command::Redo));
        bar.open_by_label("Edit");
        bar.key_down(Key::Down, &no_undo);
        // Clicking a disabled item does nothing.
        let r = bar.panels()[0].rows[0].0;
        assert_eq!(bar.pointer_down(r.x + 4.0, r.y + 4.0, &no_undo), MenuResult::Consumed);
        assert!(bar.is_open());
        // Left/Right switch menus; Escape closes.
        bar.key_down(Key::Right, &all);
        assert_eq!(bar.open_menu(), Some(2));
        bar.key_down(Key::Left, &all);
        bar.key_down(Key::Left, &all);
        assert_eq!(bar.open_menu(), Some(0));
        bar.key_down(Key::Escape, &all);
        assert!(!bar.is_open());
        // Submenu by keyboard.
        bar.open_by_label("Stat");
        bar.key_down(Key::Down, &all);
        bar.key_down(Key::Right, &all);
        assert_eq!(bar.panels().len(), 2);
        assert_eq!(
            bar.key_down(Key::Enter, &all),
            MenuResult::Command(Command::OpenDialog("stat.basic.descriptive".into()))
        );
    }

    #[test]
    fn renders_labels_and_panels() {
        let mut bar = MenuBar::standard(&Registry::standard());
        let mut s = Scene::new(800.0, 400.0, Color::WHITE);
        bar.open_by_label("Edit");
        bar.render(&mut s, 800.0, &Theme::light(), &all);
        let t: Vec<String> = s
            .items
            .iter()
            .filter_map(|i| {
                if let openstats_charts::core::Shape::Text { text, .. } = &i.shape {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect();
        for want in ["File", "Assistant", "Undo", "Ctrl+Z", "Select All"] {
            assert!(t.iter().any(|x| x == want), "missing {want}");
        }
    }
}
