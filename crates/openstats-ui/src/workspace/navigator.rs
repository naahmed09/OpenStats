//! The Project Navigator: a tree of worksheets, graphs and Session windows with rename, delete
//! and export through a context menu.

use super::project::ItemId;
use crate::grid::Key;
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Anchor, Color, Marker, Point, Rect, Scene, Stroke, TextStyle, Theme};

const ROW: f64 = 22.0;
const SIZE: f64 = 12.0;
const INDENT: f64 = 16.0;

/// A folder of the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Folder {
    /// Worksheets.
    Worksheets,
    /// Graphs.
    Graphs,
    /// Session windows.
    Sessions,
    /// Macro scripts.
    Macros,
    /// Related documents.
    Related,
}

/// Something in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The project root.
    Project,
    /// A folder.
    Folder(Folder),
    /// A worksheet by index.
    Worksheet(usize),
    /// A graph.
    Graph(ItemId),
    /// A Session window.
    Session(ItemId),
    /// A macro script.
    Macro(ItemId),
    /// A related document by index.
    Related(usize),
}

/// Context-menu actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavAction {
    /// Show the item.
    Open,
    /// Edit the name.
    Rename,
    /// Delete the item.
    Delete,
    /// Export (worksheet → CSV, graph → SVG, session → text).
    Export,
    /// Export a graph as PNG.
    ExportPng,
    /// Export a graph as PDF.
    ExportPdf,
    /// Show a graph in its own window (tear-off).
    TearOff,
    /// Copy a worksheet.
    Duplicate,
    /// Add a worksheet (folder menu).
    NewWorksheet,
    /// Add a Session window (folder menu).
    NewSession,
    /// Add a macro (folder menu).
    NewMacro,
    /// Add a related document (folder menu).
    NewRelated,
}

impl NavAction {
    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            NavAction::Open => "Open",
            NavAction::Rename => "Rename",
            NavAction::Delete => "Delete",
            NavAction::Export => "Export...",
            NavAction::ExportPng => "Export as PNG...",
            NavAction::ExportPdf => "Export as PDF...",
            NavAction::TearOff => "Open in New Window",
            NavAction::Duplicate => "Duplicate",
            NavAction::NewWorksheet => "New Worksheet",
            NavAction::NewSession => "New Session",
            NavAction::NewMacro => "New Macro",
            NavAction::NewRelated => "Add Related Document...",
        }
    }
}

/// Names shown in the tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NavModel {
    /// Worksheet names in index order.
    pub worksheets: Vec<String>,
    /// Graph ids and names.
    pub graphs: Vec<(ItemId, String)>,
    /// Session ids and names.
    pub sessions: Vec<(ItemId, String)>,
    /// Macro ids and names.
    pub macros: Vec<(ItemId, String)>,
    /// Related documents (shown only when there are some).
    pub related: Vec<String>,
}

/// Mouse button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    /// Primary button.
    Left,
    /// Secondary button (context menu).
    Right,
}

/// What a navigator input produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavEvent {
    /// Nothing for the host to do.
    None,
    /// Selection changed.
    Select(Target),
    /// An item was double-clicked or Enter was pressed.
    Open(Target),
    /// A context-menu action was chosen.
    Action(Target, NavAction),
    /// A rename was committed.
    Rename(Target, String),
}

#[derive(Debug, Clone, PartialEq)]
struct Row {
    target: Target,
    label: String,
    depth: usize,
    folder: bool,
    expanded: bool,
}

/// Rename in progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Renaming {
    /// Item being renamed.
    pub target: Target,
    /// Text typed so far.
    pub buffer: String,
}

#[derive(Debug, Clone, PartialEq)]
struct ContextMenu {
    target: Target,
    at: Point,
    actions: Vec<NavAction>,
}

/// Tree state.
#[derive(Debug, Clone, PartialEq)]
pub struct Navigator {
    expanded: [bool; 5],
    selected: Target,
    renaming: Option<Renaming>,
    context: Option<ContextMenu>,
    /// Width of the pane.
    pub width: f64,
    /// Height of the pane.
    pub height: f64,
    /// Colours used when drawing.
    pub theme: Theme,
}

impl Navigator {
    /// A navigator of `width × height` pixels with every folder expanded.
    pub fn new(width: f64, height: f64) -> Navigator {
        Navigator {
            expanded: [true; 5],
            selected: Target::Project,
            renaming: None,
            context: None,
            width,
            height,
            theme: Theme::light(),
        }
    }

    /// The selected item.
    pub fn selected(&self) -> Target {
        self.selected
    }

    /// Select an item (no event).
    pub fn select(&mut self, t: Target) {
        self.selected = t;
    }

    /// The rename in progress.
    pub fn renaming(&self) -> Option<&Renaming> {
        self.renaming.as_ref()
    }

    /// Whether the context menu is open.
    pub fn context_open(&self) -> bool {
        self.context.is_some()
    }

    /// Start renaming `t` (folders and the project cannot be renamed).
    pub fn begin_rename(&mut self, t: Target, model: &NavModel) -> bool {
        let name = match t {
            Target::Worksheet(i) => model.worksheets.get(i).cloned(),
            Target::Graph(id) => model.graphs.iter().find(|g| g.0 == id).map(|g| g.1.clone()),
            Target::Session(id) => model.sessions.iter().find(|g| g.0 == id).map(|g| g.1.clone()),
            Target::Macro(id) => model.macros.iter().find(|g| g.0 == id).map(|g| g.1.clone()),
            _ => None,
        };
        match name {
            Some(n) => {
                self.renaming = Some(Renaming { target: t, buffer: n });
                self.context = None;
                true
            }
            None => false,
        }
    }

    fn folder_index(f: Folder) -> usize {
        match f {
            Folder::Worksheets => 0,
            Folder::Graphs => 1,
            Folder::Sessions => 2,
            Folder::Macros => 3,
            Folder::Related => 4,
        }
    }

    fn rows(&self, model: &NavModel) -> Vec<Row> {
        let mut rows =
            vec![Row { target: Target::Project, label: "Project".into(), depth: 0, folder: true, expanded: true }];
        let folder = |rows: &mut Vec<Row>, f: Folder, label: &str, children: Vec<(Target, String)>| {
            let e = self.expanded[Self::folder_index(f)];
            rows.push(Row { target: Target::Folder(f), label: label.into(), depth: 1, folder: true, expanded: e });
            if e {
                rows.extend(children.into_iter().map(|(target, label)| Row {
                    target,
                    label,
                    depth: 2,
                    folder: false,
                    expanded: false,
                }));
            }
        };
        folder(
            &mut rows,
            Folder::Sessions,
            "Session",
            model.sessions.iter().map(|(id, n)| (Target::Session(*id), n.clone())).collect(),
        );
        folder(
            &mut rows,
            Folder::Worksheets,
            "Worksheets",
            model.worksheets.iter().enumerate().map(|(i, n)| (Target::Worksheet(i), n.clone())).collect(),
        );
        folder(
            &mut rows,
            Folder::Graphs,
            "Graphs",
            model.graphs.iter().map(|(id, n)| (Target::Graph(*id), n.clone())).collect(),
        );
        folder(
            &mut rows,
            Folder::Macros,
            "Macros",
            model.macros.iter().map(|(id, n)| (Target::Macro(*id), n.clone())).collect(),
        );
        if !model.related.is_empty() {
            folder(
                &mut rows,
                Folder::Related,
                "Related Documents",
                model.related.iter().enumerate().map(|(i, n)| (Target::Related(i), n.clone())).collect(),
            );
        }
        rows
    }

    /// The tree as `(depth, label, target)` rows, top to bottom.
    pub fn visible(&self, model: &NavModel) -> Vec<(usize, String, Target)> {
        self.rows(model).into_iter().map(|r| (r.depth, r.label, r.target)).collect()
    }

    /// Actions offered by the context menu of `t`.
    pub fn actions_for(t: Target) -> Vec<NavAction> {
        match t {
            Target::Project => vec![],
            Target::Folder(Folder::Worksheets) => vec![NavAction::NewWorksheet],
            Target::Folder(Folder::Sessions) => vec![NavAction::NewSession],
            Target::Folder(Folder::Graphs) => vec![],
            Target::Folder(Folder::Macros) => vec![NavAction::NewMacro],
            Target::Folder(Folder::Related) => vec![NavAction::NewRelated],
            Target::Related(_) => vec![NavAction::Open, NavAction::Delete],
            Target::Worksheet(_) => {
                vec![NavAction::Open, NavAction::Rename, NavAction::Duplicate, NavAction::Export, NavAction::Delete]
            }
            Target::Graph(_) => {
                vec![
                    NavAction::Open,
                    NavAction::TearOff,
                    NavAction::Rename,
                    NavAction::Export,
                    NavAction::ExportPng,
                    NavAction::ExportPdf,
                    NavAction::Delete,
                ]
            }
            Target::Session(_) | Target::Macro(_) => {
                vec![NavAction::Open, NavAction::Rename, NavAction::Export, NavAction::Delete]
            }
        }
    }

    fn menu_rect(c: &ContextMenu) -> Rect {
        let w = c.actions.iter().map(|a| text_width(a.label(), SIZE, false)).fold(80.0, f64::max) + 32.0;
        Rect::new(c.at.x, c.at.y, w, c.actions.len() as f64 * ROW + 4.0)
    }

    fn row_at(&self, y: f64, rows: &[Row]) -> Option<usize> {
        let i = (y / ROW).floor();
        (i >= 0.0 && (i as usize) < rows.len()).then_some(i as usize)
    }

    /// Mouse press in pane coordinates.
    pub fn pointer_down(&mut self, x: f64, y: f64, button: Button, clicks: u32, model: &NavModel) -> NavEvent {
        if let Some(c) = self.context.clone() {
            let r = Self::menu_rect(&c);
            self.context = None;
            if r.contains(Point::new(x, y)) {
                let i = ((y - r.y - 2.0) / ROW).floor();
                if i >= 0.0 {
                    if let Some(a) = c.actions.get(i as usize) {
                        return self.perform(c.target, *a, model);
                    }
                }
            }
            return NavEvent::None;
        }
        if let Some(r) = self.renaming.take() {
            // Clicking elsewhere commits the rename.
            return NavEvent::Rename(r.target, r.buffer);
        }
        let rows = self.rows(model);
        let Some(i) = self.row_at(y, &rows) else { return NavEvent::None };
        let row = rows[i].clone();
        let arrow = row.depth as f64 * INDENT + 4.0;
        if button == Button::Right {
            self.selected = row.target;
            let actions = Self::actions_for(row.target);
            if !actions.is_empty() {
                self.context = Some(ContextMenu { target: row.target, at: Point::new(x, y), actions });
            }
            return NavEvent::Select(row.target);
        }
        if let Target::Folder(f) = row.target {
            if x >= arrow && x <= arrow + 14.0 || clicks >= 2 {
                let k = Self::folder_index(f);
                self.expanded[k] = !self.expanded[k];
                self.selected = row.target;
                return NavEvent::Select(row.target);
            }
        }
        self.selected = row.target;
        if clicks >= 2 && !row.folder {
            NavEvent::Open(row.target)
        } else {
            NavEvent::Select(row.target)
        }
    }

    fn perform(&mut self, t: Target, a: NavAction, model: &NavModel) -> NavEvent {
        if a == NavAction::Rename {
            self.begin_rename(t, model);
            return NavEvent::None;
        }
        NavEvent::Action(t, a)
    }

    /// Key press while the navigator has focus.
    pub fn key_down(&mut self, key: Key, model: &NavModel) -> NavEvent {
        if self.renaming.is_some() {
            match key {
                Key::Enter => {
                    return match self.renaming.take() {
                        Some(r) => NavEvent::Rename(r.target, r.buffer),
                        None => NavEvent::None,
                    };
                }
                Key::Escape => self.renaming = None,
                Key::Backspace => {
                    if let Some(r) = self.renaming.as_mut() {
                        r.buffer.pop();
                    }
                }
                Key::Char(c) if !c.is_control() => {
                    if let Some(r) = self.renaming.as_mut() {
                        r.buffer.push(c);
                    }
                }
                _ => {}
            }
            return NavEvent::None;
        }
        if self.context.is_some() {
            if key == Key::Escape {
                self.context = None;
            }
            return NavEvent::None;
        }
        let rows = self.rows(model);
        let cur = rows.iter().position(|r| r.target == self.selected).unwrap_or(0);
        match key {
            Key::Up | Key::Down => {
                let next = if key == Key::Up { cur.saturating_sub(1) } else { (cur + 1).min(rows.len() - 1) };
                self.selected = rows[next].target;
                NavEvent::Select(self.selected)
            }
            Key::Left | Key::Right => {
                if let Target::Folder(f) = self.selected {
                    self.expanded[Self::folder_index(f)] = key == Key::Right;
                }
                NavEvent::None
            }
            Key::Enter => {
                if rows[cur].folder {
                    NavEvent::None
                } else {
                    NavEvent::Open(self.selected)
                }
            }
            Key::F2 => {
                self.begin_rename(self.selected, model);
                NavEvent::None
            }
            Key::Delete => match self.selected {
                Target::Worksheet(_)
                | Target::Graph(_)
                | Target::Session(_)
                | Target::Macro(_)
                | Target::Related(_) => NavEvent::Action(self.selected, NavAction::Delete),
                _ => NavEvent::None,
            },
            _ => NavEvent::None,
        }
    }

    /// Draw the tree.
    pub fn render(&self, model: &NavModel) -> Scene {
        let t = &self.theme;
        let mut s = Scene::new(self.width, self.height, t.background);
        s.title = "Project Manager".into();
        let ink = TextStyle::new(SIZE, t.ink);
        let chrome = crate::style::Chrome::of(t);
        let accent = chrome.accent;
        s.clip_push(Rect::new(0.0, 0.0, self.width, self.height));
        for (i, r) in self.rows(model).iter().enumerate() {
            let y = i as f64 * ROW;
            if r.target == self.selected {
                s.rect(Rect::new(0.0, y, self.width, ROW), Some(accent.with_alpha(0.2)), None);
            }
            let x = r.depth as f64 * INDENT + 4.0;
            if r.folder && r.depth > 0 {
                let c = Point::new(x + 6.0, y + ROW / 2.0);
                s.rect(
                    Rect::new(c.x - 5.0, c.y - 5.0, 10.0, 10.0),
                    None,
                    Some(Stroke::new(t.ink.with_alpha(0.6), 1.0)),
                );
                s.line(Point::new(c.x - 3.0, c.y), Point::new(c.x + 3.0, c.y), Stroke::new(t.ink, 1.0));
                if !r.expanded {
                    s.line(Point::new(c.x, c.y - 3.0), Point::new(c.x, c.y + 3.0), Stroke::new(t.ink, 1.0));
                }
            }
            let icon = Point::new(x + 22.0, y + ROW / 2.0);
            let (kind, color) = match r.target {
                Target::Worksheet(_) => (Marker::Square, chrome.worksheet),
                Target::Graph(_) => (Marker::Diamond, chrome.graph),
                Target::Session(_) => (Marker::TriangleUp, chrome.session),
                Target::Macro(_) => (Marker::Cross, chrome.script),
                _ => (Marker::Circle, chrome.folder),
            };
            s.marker(icon, kind, 9.0, color, None);
            match &self.renaming {
                Some(rn) if rn.target == r.target => {
                    let w = (self.width - x - 40.0).max(40.0);
                    s.rect(
                        Rect::new(x + 32.0, y + 2.0, w, ROW - 4.0),
                        Some(t.plot_background),
                        Some(Stroke::new(accent, 2.0)),
                    );
                    s.text(Point::new(x + 36.0, y + 15.0), rn.buffer.clone(), ink);
                    let cx = x + 36.0 + text_width(&rn.buffer, SIZE, false);
                    s.line(Point::new(cx, y + 5.0), Point::new(cx, y + ROW - 5.0), Stroke::new(t.ink, 1.0));
                }
                _ => s.text(Point::new(x + 32.0, y + 15.0), r.label.clone(), if r.folder { ink.bold() } else { ink }),
            }
        }
        s.clip_pop();
        if let Some(c) = &self.context {
            let r = Self::menu_rect(c);
            s.rect(Rect::new(r.x + 2.0, r.y + 2.0, r.w, r.h), Some(Color::BLACK.with_alpha(0.15)), None);
            s.rect(r, Some(t.background), Some(Stroke::new(t.ink.with_alpha(0.5), 1.0)));
            for (i, a) in c.actions.iter().enumerate() {
                s.text(
                    Point::new(r.x + 14.0, r.y + 2.0 + (i as f64 + 1.0) * ROW - 6.0),
                    a.label().to_string(),
                    ink.anchored(Anchor::Start),
                );
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> NavModel {
        NavModel {
            worksheets: vec!["Worksheet 1".into(), "Data".into()],
            graphs: vec![(ItemId(2), "Histogram 1".into()), (ItemId(3), "Regression Plot 1".into())],
            sessions: vec![(ItemId(1), "Session 1".into())],
            macros: vec![(ItemId(7), "Clean".into())],
            related: Vec::new(),
        }
    }

    #[test]
    fn tree_structure_and_collapse() {
        let mut n = Navigator::new(200.0, 400.0);
        let m = model();
        let v = n.visible(&m);
        let labels: Vec<&str> = v.iter().map(|x| x.1.as_str()).collect();
        assert_eq!(
            labels,
            [
                "Project",
                "Session",
                "Session 1",
                "Worksheets",
                "Worksheet 1",
                "Data",
                "Graphs",
                "Histogram 1",
                "Regression Plot 1",
                "Macros",
                "Clean"
            ]
        );
        assert_eq!(v[4].0, 2);
        // Collapse Worksheets by clicking its box (row 3).
        n.pointer_down(1.0 * INDENT + 10.0, 3.0 * ROW + 5.0, Button::Left, 1, &m);
        assert_eq!(n.visible(&m).len(), 9);
        n.key_down(Key::Right, &m);
        assert_eq!(n.visible(&m).len(), 11);
    }

    #[test]
    fn select_open_and_keyboard() {
        let mut n = Navigator::new(200.0, 400.0);
        let m = model();
        let ev = n.pointer_down(60.0, 4.0 * ROW + 5.0, Button::Left, 1, &m);
        assert_eq!(ev, NavEvent::Select(Target::Worksheet(0)));
        let ev = n.pointer_down(60.0, 7.0 * ROW + 5.0, Button::Left, 2, &m);
        assert_eq!(ev, NavEvent::Open(Target::Graph(ItemId(2))));
        assert_eq!(n.key_down(Key::Down, &m), NavEvent::Select(Target::Graph(ItemId(3))));
        assert_eq!(n.key_down(Key::Enter, &m), NavEvent::Open(Target::Graph(ItemId(3))));
        assert_eq!(n.key_down(Key::Delete, &m), NavEvent::Action(Target::Graph(ItemId(3)), NavAction::Delete));
        n.select(Target::Folder(Folder::Graphs));
        assert_eq!(n.key_down(Key::Delete, &m), NavEvent::None);
        assert_eq!(n.key_down(Key::Enter, &m), NavEvent::None);
    }

    #[test]
    fn context_menu_rename_delete_export() {
        let mut n = Navigator::new(200.0, 400.0);
        let m = model();
        // Right-click the second worksheet.
        n.pointer_down(60.0, 5.0 * ROW + 5.0, Button::Right, 1, &m);
        assert!(n.context_open());
        assert_eq!(n.selected(), Target::Worksheet(1));
        let r = Navigator::menu_rect(n.context.as_ref().unwrap());
        // Items: Open, Rename, Duplicate, Export, Delete -> choose Export (index 3).
        let ev = n.pointer_down(r.x + 10.0, r.y + 2.0 + 3.0 * ROW + 5.0, Button::Left, 1, &m);
        assert_eq!(ev, NavEvent::Action(Target::Worksheet(1), NavAction::Export));
        assert!(!n.context_open());
        // Rename through the menu.
        n.pointer_down(60.0, 5.0 * ROW + 5.0, Button::Right, 1, &m);
        let r = Navigator::menu_rect(n.context.as_ref().unwrap());
        assert_eq!(n.pointer_down(r.x + 10.0, r.y + 2.0 + 1.0 * ROW + 5.0, Button::Left, 1, &m), NavEvent::None);
        assert_eq!(n.renaming().unwrap().buffer, "Data");
        for _ in 0..4 {
            n.key_down(Key::Backspace, &m);
        }
        for c in "Raw".chars() {
            n.key_down(Key::Char(c), &m);
        }
        assert_eq!(n.key_down(Key::Enter, &m), NavEvent::Rename(Target::Worksheet(1), "Raw".into()));
        // Escape cancels; F2 starts; folder menus offer "New ..." only.
        n.select(Target::Session(ItemId(1)));
        n.key_down(Key::F2, &m);
        n.key_down(Key::Escape, &m);
        assert!(n.renaming().is_none());
        assert_eq!(Navigator::actions_for(Target::Folder(Folder::Worksheets)), vec![NavAction::NewWorksheet]);
        assert!(Navigator::actions_for(Target::Graph(ItemId(1))).contains(&NavAction::ExportPng));
        // Clicking away from an open context menu just closes it.
        n.pointer_down(60.0, 2.0 * ROW + 5.0, Button::Right, 1, &m);
        assert_eq!(n.pointer_down(190.0, 380.0, Button::Left, 1, &m), NavEvent::None);
        assert!(!n.context_open());
        assert!(!n.begin_rename(Target::Project, &m));
    }

    #[test]
    fn renders_names_and_editor() {
        let mut n = Navigator::new(200.0, 400.0);
        let m = model();
        n.select(Target::Graph(ItemId(2)));
        n.begin_rename(Target::Graph(ItemId(2)), &m);
        let s = n.render(&m);
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
        for w in ["Project", "Session 1", "Worksheet 1", "Regression Plot 1", "Histogram 1"] {
            assert!(t.iter().any(|x| x == w), "missing {w}");
        }
    }
}
