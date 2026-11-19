//! The interactive dialog: renders a [`DialogSpec`] and edits its [`Values`].

use super::spec::{validate_info, ColInfo, DialogSpec, FieldKind, Page, Problem, Value, Values, VarKind};
use crate::grid::{Key, Mods};
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Anchor, Point, Rect, Scene, Stroke, TextStyle, Theme};
use openstats_core::column::ColumnKind;
use openstats_core::value::{format_number, parse_number};
use openstats_core::Worksheet;
use std::collections::BTreeMap;

/// Dialog width in pixels.
pub const WIDTH: f64 = 660.0;
/// Dialog height in pixels.
pub const HEIGHT: f64 = 470.0;
const SIZE: f64 = 12.0;
const ROW: f64 = 20.0;

/// What a click or key press did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DialogEvent {
    /// Nothing the host needs to know about.
    None,
    /// A value changed.
    Changed,
    /// OK was pressed and the input is valid.
    Ok,
    /// Cancel or Escape.
    Cancel,
}

/// An open dialog.
#[derive(Debug, Clone)]
pub struct DialogView {
    spec: DialogSpec,
    values: Values,
    cols: Vec<ColInfo>,
    buffers: BTreeMap<String, String>,
    page: Page,
    focus: Option<usize>,
    selected: Vec<usize>,
    anchor: Option<usize>,
    cursor: Option<usize>,
    list_scroll: usize,
    problems: Vec<Problem>,
    /// Colours used when drawing.
    pub theme: Theme,
}

impl DialogView {
    /// Open the dialog for `spec` over worksheet `ws`, with default values.
    pub fn new(spec: DialogSpec, ws: &Worksheet) -> DialogView {
        let values = spec.defaults();
        let mut d = DialogView {
            spec,
            values,
            cols: ColInfo::of(ws),
            buffers: BTreeMap::new(),
            page: Page::Main,
            focus: None,
            selected: Vec::new(),
            anchor: None,
            cursor: None,
            list_scroll: 0,
            problems: Vec::new(),
            theme: Theme::light(),
        };
        d.reset_buffers();
        d.focus =
            d.spec.fields.iter().position(|f| f.page == Page::Main && matches!(f.kind, FieldKind::Variables { .. }));
        // Pre-select the active column choice later via `select_column`.
        d
    }

    fn reset_buffers(&mut self) {
        self.buffers.clear();
        for f in &self.spec.fields {
            match &f.kind {
                FieldKind::Number { .. } => {
                    let x = self.values.num(&f.name);
                    self.buffers.insert(f.name.clone(), if x.is_nan() { String::new() } else { format_number(x) });
                }
                FieldKind::Text { .. } => {
                    self.buffers.insert(f.name.clone(), self.values.text(&f.name).to_string());
                }
                _ => {}
            }
        }
    }

    /// The dialog description.
    pub fn spec(&self) -> &DialogSpec {
        &self.spec
    }

    /// The current values.
    pub fn values(&self) -> &Values {
        &self.values
    }

    /// Replace the values (e.g. to re-open a dialog with its last settings).
    pub fn set_values(&mut self, v: Values) {
        self.values = v;
        self.reset_buffers();
    }

    /// Problems found by the last OK attempt.
    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }

    /// The visible page.
    pub fn page(&self) -> Page {
        self.page
    }

    /// Show a page (ignored if the dialog has no such page).
    pub fn set_page(&mut self, p: Page) {
        if self.spec.pages().contains(&p) {
            self.page = p;
            self.focus =
                self.spec.fields.iter().position(|f| f.page == p && !matches!(f.kind, FieldKind::Check { .. }));
        }
    }

    /// The focused field's name.
    pub fn focused(&self) -> Option<&str> {
        self.focus.map(|i| self.spec.fields[i].name.as_str())
    }

    /// Insert committed text (an input-method commit or a paste) into the focused number or
    /// text field. Returns whether the field changed.
    pub fn insert_text(&mut self, text: &str) -> bool {
        let Some(fi) = self.focus else { return false };
        if !matches!(self.spec.fields[fi].kind, FieldKind::Number { .. } | FieldKind::Text { .. }) {
            return false;
        }
        let mut changed = false;
        for c in text.chars().filter(|c| !c.is_control()) {
            changed |= self.type_into(fi, Some(c), false);
        }
        changed
    }

    /// Highlight column `c` in the list.
    pub fn select_column(&mut self, c: usize) {
        if c < self.cols.len() {
            self.selected = vec![c];
            self.anchor = Some(c);
            self.reveal(c);
        }
    }

    /// The highlighted columns, in ascending order.
    pub fn selected_columns(&self) -> &[usize] {
        &self.selected
    }

    /// Highlight with the usual list rules: plain replaces the selection, Ctrl toggles one
    /// column, Shift selects the range from the anchor (added to the selection with Ctrl).
    pub fn select_with(&mut self, c: usize, mods: Mods) {
        if c >= self.cols.len() {
            return;
        }
        if mods.shift {
            let a = self.anchor.unwrap_or(c);
            let (lo, hi) = (a.min(c), a.max(c));
            if !mods.ctrl {
                self.selected.clear();
            }
            self.selected.extend(lo..=hi);
        } else if mods.ctrl {
            if let Some(i) = self.selected.iter().position(|&x| x == c) {
                self.selected.remove(i);
            } else {
                self.selected.push(c);
            }
            self.anchor = Some(c);
        } else {
            self.selected = vec![c];
            self.anchor = Some(c);
        }
        self.selected.sort_unstable();
        self.selected.dedup();
        self.reveal(c);
    }

    /// Highlight every column.
    pub fn select_all_columns(&mut self) {
        self.selected = (0..self.cols.len()).collect();
        self.anchor = self.selected.first().copied();
    }

    fn reveal(&mut self, c: usize) {
        let rows = (self.list_rect().h / ROW) as usize;
        if c < self.list_scroll {
            self.list_scroll = c;
        } else if rows > 0 && c >= self.list_scroll + rows {
            self.list_scroll = c + 1 - rows;
        }
    }

    fn move_selection(&mut self, d: isize, mods: Mods) {
        if self.cols.is_empty() {
            return;
        }
        let cur = self.anchor_cursor();
        let next = (cur as isize + d).clamp(0, self.cols.len() as isize - 1) as usize;
        let keep_anchor = self.anchor;
        self.select_with(next, Mods { ctrl: false, ..mods });
        if mods.shift {
            self.anchor = keep_anchor;
        }
        self.cursor = Some(next);
    }

    fn anchor_cursor(&self) -> usize {
        self.cursor.or(self.anchor).unwrap_or(0)
    }

    /// Focus the field called `name` (switching page if needed).
    pub fn focus_field(&mut self, name: &str) {
        if let Some(i) = self.spec.fields.iter().position(|f| f.name == name) {
            self.page = self.spec.fields[i].page;
            self.focus = Some(i);
        }
    }

    /// Add the highlighted columns to the focused variable box (those of the wrong kind are
    /// skipped; a box that is full keeps the newest columns). Returns whether anything changed.
    pub fn add_selected(&mut self) -> bool {
        let Some(fi) = self.focus else { return false };
        let f = &self.spec.fields[fi];
        let FieldKind::Variables { kind, max, .. } = &f.kind else { return false };
        let name = f.name.clone();
        let max = *max;
        let mut v = self.values.vars(&name).to_vec();
        let before = v.clone();
        for &c in &self.selected {
            let ok = match kind {
                VarKind::Any => true,
                VarKind::Text => self.cols[c].kind == ColumnKind::Text,
                VarKind::Numeric => self.cols[c].kind != ColumnKind::Text,
            };
            if !ok || v.contains(&c) {
                continue;
            }
            if v.len() >= max {
                v.remove(0);
            }
            v.push(c);
        }
        if v == before {
            return false;
        }
        self.values.set(&name, Value::Vars(v));
        true
    }

    /// Remove the last column of the focused variable box.
    pub fn remove_last(&mut self) -> bool {
        let Some(fi) = self.focus else { return false };
        let name = self.spec.fields[fi].name.clone();
        if !matches!(self.spec.fields[fi].kind, FieldKind::Variables { .. }) {
            return false;
        }
        let mut v = self.values.vars(&name).to_vec();
        let had = v.pop().is_some();
        self.values.set(&name, Value::Vars(v));
        had
    }

    /// Show an error from running the analysis (the dialog stays open).
    pub fn fail(&mut self, message: &str) {
        self.problems = vec![Problem { field: String::new(), message: message.to_string() }];
    }

    /// Validate and report whether OK may proceed.
    pub fn try_ok(&mut self) -> DialogEvent {
        self.problems = validate_info(&self.spec, &self.values, &self.cols);
        if let Some(p) = self.problems.first() {
            // Jump to the page of the first problem so the user sees it.
            if let Some(f) = self.spec.field(&p.field) {
                self.page = f.page;
            }
            DialogEvent::None
        } else {
            DialogEvent::Ok
        }
    }

    fn type_into(&mut self, fi: usize, ch: Option<char>, backspace: bool) -> bool {
        let f = &self.spec.fields[fi];
        let name = f.name.clone();
        let is_num = matches!(f.kind, FieldKind::Number { .. });
        if !(is_num || matches!(f.kind, FieldKind::Text { .. })) {
            return false;
        }
        let buf = self.buffers.entry(name.clone()).or_default();
        if backspace {
            buf.pop();
        } else if let Some(c) = ch {
            buf.push(c);
        }
        let text = buf.clone();
        if is_num {
            let x = if text.trim().is_empty() { f64::NAN } else { parse_number(&text).unwrap_or(f64::NAN) };
            self.values.set(&name, Value::Num(x));
        } else {
            self.values.set(&name, Value::Text(text));
        }
        true
    }

    fn cycle_choice(&mut self, fi: usize, dir: isize) {
        let f = &self.spec.fields[fi];
        if let FieldKind::Choice { options, .. } = &f.kind {
            let n = options.len() as isize;
            let cur = self.values.choice(&f.name) as isize;
            let name = f.name.clone();
            self.values.set(&name, Value::Choice((cur + dir).rem_euclid(n.max(1)) as usize));
        }
    }

    fn page_fields(&self) -> Vec<usize> {
        (0..self.spec.fields.len()).filter(|&i| self.spec.fields[i].page == self.page).collect()
    }

    /// Key press.
    pub fn key_down(&mut self, key: Key, mods: Mods) -> DialogEvent {
        match key {
            Key::Escape => return DialogEvent::Cancel,
            Key::Enter => return self.try_ok(),
            Key::Tab => {
                let fields = self.page_fields();
                if !fields.is_empty() {
                    let pos = self.focus.and_then(|f| fields.iter().position(|x| *x == f));
                    let next = match (pos, mods.shift) {
                        (Some(p), false) => (p + 1) % fields.len(),
                        (Some(p), true) => (p + fields.len() - 1) % fields.len(),
                        (None, _) => 0,
                    };
                    self.focus = Some(fields[next]);
                }
                return DialogEvent::None;
            }
            _ => {}
        }
        let Some(fi) = self.focus else { return DialogEvent::None };
        let changed = match (&self.spec.fields[fi].kind, key) {
            (FieldKind::Variables { .. }, Key::Backspace | Key::Delete) => self.remove_last(),
            (FieldKind::Variables { .. }, Key::Char(' ')) => self.add_selected(),
            (FieldKind::Variables { .. }, Key::Char(c)) if mods.ctrl && c.eq_ignore_ascii_case(&'a') => {
                self.select_all_columns();
                false
            }
            (FieldKind::Variables { .. }, Key::Down) => {
                self.move_selection(1, mods);
                false
            }
            (FieldKind::Variables { .. }, Key::Up) => {
                self.move_selection(-1, mods);
                false
            }
            (FieldKind::Number { .. } | FieldKind::Text { .. }, Key::Backspace) => self.type_into(fi, None, true),
            (FieldKind::Number { .. } | FieldKind::Text { .. }, Key::Char(c)) if !mods.ctrl && !c.is_control() => {
                self.type_into(fi, Some(c), false)
            }
            (FieldKind::Choice { .. }, Key::Right | Key::Down) => {
                self.cycle_choice(fi, 1);
                true
            }
            (FieldKind::Choice { .. }, Key::Left | Key::Up) => {
                self.cycle_choice(fi, -1);
                true
            }
            (FieldKind::Check { .. }, Key::Char(' ')) => {
                self.toggle(fi);
                true
            }
            _ => false,
        };
        if changed {
            DialogEvent::Changed
        } else {
            DialogEvent::None
        }
    }

    fn toggle(&mut self, fi: usize) {
        let name = self.spec.fields[fi].name.clone();
        let cur = self.values.flag(&name);
        self.values.set(&name, Value::Bool(!cur));
    }

    // ------------------------------------------------------------------ geometry

    fn list_rect(&self) -> Rect {
        Rect::new(12.0, 44.0, 210.0, 330.0)
    }

    fn select_button(&self) -> Rect {
        Rect::new(12.0, 382.0, 210.0, 26.0)
    }

    fn tab_rects(&self) -> Vec<(Page, Rect)> {
        let mut x = 236.0;
        self.spec
            .pages()
            .into_iter()
            .map(|p| {
                let w = text_width(p.label(), SIZE, false) + 22.0;
                let r = Rect::new(x, 44.0, w, 26.0);
                x += w + 4.0;
                (p, r)
            })
            .collect()
    }

    /// `(field index, label rect, input rect)` of every field on the current page.
    fn field_rects(&self) -> Vec<(usize, Rect, Rect)> {
        let mut y = 82.0;
        let mut out = Vec::new();
        for i in self.page_fields() {
            let f = &self.spec.fields[i];
            let (h, label, input) = match &f.kind {
                FieldKind::Variables { .. } => {
                    (66.0, Rect::new(236.0, y, 410.0, 16.0), Rect::new(236.0, y + 18.0, 410.0, 40.0))
                }
                FieldKind::Check { .. } => {
                    (28.0, Rect::new(260.0, y, 380.0, 20.0), Rect::new(236.0, y + 2.0, 16.0, 16.0))
                }
                _ => (34.0, Rect::new(236.0, y + 4.0, 200.0, 20.0), Rect::new(446.0, y, 200.0, 26.0)),
            };
            out.push((i, label, input));
            y += h;
        }
        out
    }

    fn button_rects(&self) -> (Rect, Rect) {
        (Rect::new(470.0, 430.0, 82.0, 28.0), Rect::new(562.0, 430.0, 82.0, 28.0))
    }

    /// Mouse press at `(x, y)` (dialog coordinates). `clicks` ≥ 2 is a double click.
    pub fn pointer_down(&mut self, x: f64, y: f64, clicks: u32) -> DialogEvent {
        self.pointer_down_with(x, y, clicks, Mods::NONE)
    }

    /// Like [`DialogView::pointer_down`] with modifier keys, for Ctrl/Shift selection in the list.
    pub fn pointer_down_with(&mut self, x: f64, y: f64, clicks: u32, mods: Mods) -> DialogEvent {
        let p = Point::new(x, y);
        let (ok, cancel) = self.button_rects();
        if ok.contains(p) {
            return self.try_ok();
        }
        if cancel.contains(p) {
            return DialogEvent::Cancel;
        }
        if self.select_button().contains(p) {
            return if self.add_selected() { DialogEvent::Changed } else { DialogEvent::None };
        }
        let l = self.list_rect();
        if l.contains(p) {
            let idx = self.list_scroll + ((y - l.y) / ROW).floor() as usize;
            if idx < self.cols.len() {
                self.select_with(idx, mods);
                self.cursor = Some(idx);
                if clicks >= 2 && !mods.ctrl && !mods.shift && self.add_selected() {
                    return DialogEvent::Changed;
                }
            }
            return DialogEvent::None;
        }
        for (pg, r) in self.tab_rects() {
            if r.contains(p) {
                self.set_page(pg);
                return DialogEvent::None;
            }
        }
        for (i, label, input) in self.field_rects() {
            let hit =
                input.contains(p) || (matches!(self.spec.fields[i].kind, FieldKind::Check { .. }) && label.contains(p));
            if !hit {
                continue;
            }
            self.focus = Some(i);
            match &self.spec.fields[i].kind {
                FieldKind::Check { .. } => {
                    self.toggle(i);
                    return DialogEvent::Changed;
                }
                FieldKind::Choice { .. } => {
                    self.cycle_choice(i, if x > input.x + input.w / 2.0 { 1 } else { -1 });
                    return DialogEvent::Changed;
                }
                FieldKind::Variables { .. } => {
                    // Clicking a chosen column removes it.
                    let name = self.spec.fields[i].name.clone();
                    let chosen = self.values.vars(&name).to_vec();
                    let mut cx = input.x + 6.0;
                    for (k, c) in chosen.iter().enumerate() {
                        let w = text_width(&self.cols[*c].list_text(), SIZE, false) + 14.0;
                        if x >= cx && x < cx + w && clicks >= 2 {
                            let mut v = chosen.clone();
                            v.remove(k);
                            self.values.set(&name, Value::Vars(v));
                            return DialogEvent::Changed;
                        }
                        cx += w + 6.0;
                    }
                }
                _ => {}
            }
            return DialogEvent::None;
        }
        DialogEvent::None
    }

    /// Scroll the column list.
    pub fn wheel(&mut self, dy: f64) {
        let visible = (self.list_rect().h / ROW) as usize;
        let max = self.cols.len().saturating_sub(visible);
        let d = if dy > 0.0 { 1 } else { -1 };
        self.list_scroll = (self.list_scroll as isize + d).clamp(0, max as isize) as usize;
    }

    // ------------------------------------------------------------------ drawing

    /// Draw the dialog into a `WIDTH × HEIGHT` scene.
    pub fn render(&self) -> Scene {
        let t = &self.theme;
        let mut s = Scene::new(WIDTH, HEIGHT, t.background);
        let ink = TextStyle::new(SIZE, t.ink);
        let dim = TextStyle::new(SIZE, t.ink.with_alpha(0.6));
        let chrome = crate::style::Chrome::of(t);
        let accent = chrome.accent;
        let edge = Stroke::new(t.ink.with_alpha(0.5), 1.0);
        s.rect(Rect::new(0.0, 0.0, WIDTH, HEIGHT), Some(t.background), Some(edge));
        s.rect(Rect::new(0.0, 0.0, WIDTH, 30.0), Some(accent), None);
        s.text(Point::new(12.0, 20.0), self.spec.title.clone(), TextStyle::new(13.0, chrome.on_accent).bold());
        // Column list.
        let l = self.list_rect();
        s.rect(l, Some(t.plot_background), Some(edge));
        s.clip_push(l);
        let visible = (l.h / ROW) as usize + 1;
        for (k, c) in self.cols.iter().enumerate().skip(self.list_scroll).take(visible) {
            let r = Rect::new(l.x, l.y + (k - self.list_scroll) as f64 * ROW, l.w, ROW);
            if self.selected.contains(&k) {
                s.rect(r, Some(accent.with_alpha(0.25)), None);
            }
            s.text(Point::new(r.x + 6.0, r.y + 14.0), c.list_text(), ink);
        }
        s.clip_pop();
        let b = self.select_button();
        s.rect(b, Some(chrome.button), Some(edge));
        s.text(Point::new(b.x + b.w / 2.0, b.y + 17.0), "Select >>".to_string(), ink.anchored(Anchor::Middle));
        // Tabs.
        for (p, r) in self.tab_rects() {
            let on = p == self.page;
            s.rect(r, Some(if on { accent.with_alpha(0.2) } else { chrome.button }), Some(edge));
            s.text(Point::new(r.x + 11.0, r.y + 17.0), p.label().to_string(), ink);
        }
        // Fields.
        for (i, label, input) in self.field_rects() {
            let f = &self.spec.fields[i];
            let focused = self.focus == Some(i);
            let border =
                Stroke::new(if focused { accent } else { t.ink.with_alpha(0.5) }, if focused { 2.0 } else { 1.0 });
            match &f.kind {
                FieldKind::Variables { .. } => {
                    s.text(Point::new(label.x, label.y + 12.0), f.label.clone(), ink);
                    s.rect(input, Some(t.plot_background), Some(border));
                    let mut cx = input.x + 6.0;
                    for &c in self.values.vars(&f.name) {
                        let txt = self.cols.get(c).map_or_else(|| format!("C{}", c + 1), ColInfo::list_text);
                        let w = text_width(&txt, SIZE, false) + 14.0;
                        s.rect(
                            Rect::new(cx, input.y + 8.0, w, 22.0),
                            Some(accent.with_alpha(0.15)),
                            Some(Stroke::new(accent, 1.0)),
                        );
                        s.text(Point::new(cx + 7.0, input.y + 23.0), txt, ink);
                        cx += w + 6.0;
                    }
                }
                FieldKind::Number { .. } | FieldKind::Text { .. } => {
                    s.text(Point::new(label.x, label.y + 13.0), f.label.clone(), ink);
                    s.rect(input, Some(t.plot_background), Some(border));
                    let buf = self.buffers.get(&f.name).cloned().unwrap_or_default();
                    s.text(Point::new(input.x + 6.0, input.y + 18.0), buf.clone(), ink);
                    if focused {
                        let cx = input.x + 6.0 + text_width(&buf, SIZE, false);
                        s.line(
                            Point::new(cx, input.y + 5.0),
                            Point::new(cx, input.bottom() - 5.0),
                            Stroke::new(t.ink, 1.0),
                        );
                    }
                }
                FieldKind::Choice { options, .. } => {
                    s.text(Point::new(label.x, label.y + 13.0), f.label.clone(), ink);
                    s.rect(input, Some(t.plot_background), Some(border));
                    s.text(Point::new(input.x + 8.0, input.y + 18.0), "<".to_string(), dim);
                    s.text(Point::new(input.right() - 8.0, input.y + 18.0), ">".to_string(), dim.anchored(Anchor::End));
                    let o = options.get(self.values.choice(&f.name)).cloned().unwrap_or_default();
                    s.text(Point::new(input.x + input.w / 2.0, input.y + 18.0), o, ink.anchored(Anchor::Middle));
                }
                FieldKind::Check { .. } => {
                    s.rect(input, Some(t.plot_background), Some(border));
                    if self.values.flag(&f.name) {
                        s.line(
                            Point::new(input.x + 3.0, input.y + 8.0),
                            Point::new(input.x + 7.0, input.bottom() - 3.0),
                            Stroke::new(accent, 2.0),
                        );
                        s.line(
                            Point::new(input.x + 7.0, input.bottom() - 3.0),
                            Point::new(input.right() - 2.0, input.y + 3.0),
                            Stroke::new(accent, 2.0),
                        );
                    }
                    s.text(Point::new(label.x, label.y + 14.0), f.label.clone(), ink);
                }
            }
        }
        // Problems and help.
        if let Some(p) = self.problems.first() {
            s.text(Point::new(236.0, 398.0), p.message.clone(), TextStyle::new(SIZE, t.alert));
        } else {
            s.text(Point::new(236.0, 398.0), crate::grid::format::fit_text(&self.spec.help, 410.0, SIZE, false), dim);
        }
        let (ok, cancel) = self.button_rects();
        for (r, label, primary) in [(ok, "OK", true), (cancel, "Cancel", false)] {
            s.rect(r, Some(if primary { accent } else { chrome.button }), Some(edge));
            s.text(
                Point::new(r.x + r.w / 2.0, r.y + 18.0),
                label.to_string(),
                TextStyle::new(SIZE, if primary { chrome.on_accent } else { t.ink }).anchored(Anchor::Middle),
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialogs::analysis::Registry;
    use openstats_core::column::Column;

    fn ws() -> Worksheet {
        let mut w = Worksheet::new("W");
        w.add_column(Column::numeric("height", vec![1.0, 2.0, 3.0, 4.0])).unwrap();
        w.add_column(Column::text("group", vec!["a", "a", "b", "b"])).unwrap();
        w.add_column(Column::numeric("weight", vec![5.0, 6.0, 7.0, 9.0])).unwrap();
        w
    }

    fn dialog(id: &str) -> DialogView {
        DialogView::new(Registry::standard().get(id).unwrap().spec.clone(), &ws())
    }

    #[test]
    fn selecting_columns_and_ok() {
        let mut d = dialog("stat.basic.ttest1");
        assert_eq!(d.focused(), Some("var"));
        assert_eq!(d.pointer_down(500.0, 440.0, 1), DialogEvent::None, "invalid input blocks OK");
        assert!(d.problems()[0].message.contains("select a column"));
        // Highlight "height" (row 0 of the list) and press Select.
        assert_eq!(d.pointer_down(60.0, 54.0, 1), DialogEvent::None);
        assert_eq!(d.pointer_down(100.0, 395.0, 1), DialogEvent::Changed);
        assert_eq!(d.values().vars("var"), &[0]);
        // A text column is refused for a numeric box.
        assert_eq!(d.pointer_down(60.0, 74.0, 1), DialogEvent::None);
        assert_eq!(d.pointer_down(100.0, 395.0, 1), DialogEvent::None);
        assert_eq!(d.values().vars("var"), &[0]);
        // Double click replaces the single-column selection.
        assert_eq!(d.pointer_down(60.0, 94.0, 2), DialogEvent::Changed);
        assert_eq!(d.values().vars("var"), &[2]);
        assert_eq!(d.pointer_down(500.0, 440.0, 1), DialogEvent::Ok);
        assert_eq!(d.pointer_down(600.0, 440.0, 1), DialogEvent::Cancel);
    }

    #[test]
    fn options_page_number_choice_and_checks() {
        let mut d = dialog("stat.basic.ttest1");
        d.select_column(0);
        assert!(d.add_selected());
        let tabs = d.tab_rects();
        assert_eq!(tabs.iter().map(|t| t.0).collect::<Vec<_>>(), vec![Page::Main, Page::Options, Page::Graphs]);
        d.pointer_down(tabs[1].1.x + 5.0, tabs[1].1.y + 5.0, 1);
        assert_eq!(d.page(), Page::Options);
        assert_eq!(d.focused(), Some("mu"));
        // Type a test mean.
        for c in "12.5".chars() {
            d.key_down(Key::Char(c), Mods::NONE);
        }
        assert_eq!(d.values().num("mu"), 12.5);
        d.key_down(Key::Backspace, Mods::NONE);
        assert_eq!(d.values().num("mu"), 12.0);
        // Tab to the Alternative choice and cycle it.
        d.key_down(Key::Tab, Mods::NONE);
        d.key_down(Key::Right, Mods::NONE);
        assert_eq!(d.values().choice("alt"), 1);
        d.key_down(Key::Left, Mods::NONE);
        d.key_down(Key::Left, Mods::NONE);
        assert_eq!(d.values().choice("alt"), 2, "wraps around");
        // Level field is empty -> invalid; fix it.
        d.focus_field("level");
        for _ in 0..4 {
            d.key_down(Key::Backspace, Mods::NONE);
        }
        assert!(d.values().num("level").is_nan());
        assert_eq!(d.key_down(Key::Enter, Mods::NONE), DialogEvent::None);
        assert!(d.problems()[0].message.contains("Confidence level"));
        for c in "99".chars() {
            d.key_down(Key::Char(c), Mods::NONE);
        }
        assert_eq!(d.key_down(Key::Enter, Mods::NONE), DialogEvent::Ok);
        // Graph checkbox toggles by click.
        d.set_page(Page::Graphs);
        let (i, _, input) = d.field_rects()[0];
        assert_eq!(d.spec().fields[i].name, "hist");
        assert_eq!(d.pointer_down(input.x + 3.0, input.y + 3.0, 1), DialogEvent::Changed);
        assert!(d.values().flag("hist"));
        assert_eq!(d.key_down(Key::Escape, Mods::NONE), DialogEvent::Cancel);
    }

    #[test]
    fn multi_variable_box_and_removal() {
        let mut d = dialog("stat.basic.correlation");
        for c in [0, 2] {
            d.select_column(c);
            assert!(d.add_selected());
        }
        assert!(!d.add_selected(), "no duplicates");
        assert_eq!(d.values().vars("vars"), &[0, 2]);
        d.key_down(Key::Backspace, Mods::NONE);
        assert_eq!(d.values().vars("vars"), &[0]);
        assert_eq!(d.key_down(Key::Enter, Mods::NONE), DialogEvent::None);
        assert!(d.problems()[0].message.contains("at least 2"));
        d.select_column(2);
        d.key_down(Key::Char(' '), Mods::NONE);
        assert_eq!(d.key_down(Key::Enter, Mods::NONE), DialogEvent::Ok);
    }

    #[test]
    fn renders_columns_fields_and_errors() {
        let mut d = dialog("stat.regression.fit");
        d.try_ok();
        let s = d.render();
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
        for want in [
            "Regression",
            "C1 height",
            "C2-T group",
            "Response",
            "Predictors",
            "Select >>",
            "OK",
            "Cancel",
            "Storage...",
            "Response: select a column.",
        ] {
            assert!(t.iter().any(|x| x == want), "missing {want}: {t:?}");
        }
    }

    fn row_y(k: usize) -> f64 {
        44.0 + k as f64 * ROW + 5.0
    }

    #[test]
    fn ctrl_and_shift_select_several_columns_and_add_them_together() {
        let mut d = dialog("stat.basic.descriptive");
        d.pointer_down_with(30.0, row_y(0), 1, Mods::NONE);
        assert_eq!(d.selected_columns(), &[0]);
        d.pointer_down_with(30.0, row_y(2), 1, Mods::CTRL);
        assert_eq!(d.selected_columns(), &[0, 2], "Ctrl adds");
        d.pointer_down_with(30.0, row_y(0), 1, Mods::CTRL);
        assert_eq!(d.selected_columns(), &[2], "Ctrl toggles off");
        d.pointer_down_with(30.0, row_y(0), 1, Mods::NONE);
        d.pointer_down_with(30.0, row_y(2), 1, Mods::SHIFT);
        assert_eq!(d.selected_columns(), &[0, 1, 2], "Shift selects the range");
        d.focus_field("vars");
        assert_eq!(d.pointer_down(236.0 + 5.0, 400.0, 1), DialogEvent::None);
        assert!(d.add_selected());
        assert_eq!(d.values().vars("vars"), &[0, 2], "the text column is skipped for a numeric box");
        assert!(!d.add_selected(), "adding again changes nothing");
    }

    #[test]
    fn keyboard_selection_in_the_list() {
        let mut d = dialog("stat.basic.descriptive");
        d.focus_field("vars");
        d.select_column(0);
        d.key_down(Key::Down, Mods::SHIFT);
        d.key_down(Key::Down, Mods::SHIFT);
        assert_eq!(d.selected_columns(), &[0, 1, 2]);
        d.key_down(Key::Up, Mods::SHIFT);
        assert_eq!(d.selected_columns(), &[0, 1], "Shift+Up shrinks from the anchor");
        d.key_down(Key::Down, Mods::NONE);
        assert_eq!(d.selected_columns().len(), 1);
        d.key_down(Key::Char('a'), Mods::CTRL);
        assert_eq!(d.selected_columns(), &[0, 1, 2]);
        assert_eq!(d.key_down(Key::Char(' '), Mods::NONE), DialogEvent::Changed);
        assert_eq!(d.values().vars("vars"), &[0, 2]);
    }

    #[test]
    fn a_full_single_variable_box_takes_the_newest_column() {
        let mut d = dialog("stat.basic.ttest1");
        d.focus_field("var");
        d.select_with(0, Mods::NONE);
        d.select_with(2, Mods::CTRL);
        assert!(d.add_selected());
        assert_eq!(d.values().vars("var"), &[2]);
    }
}
