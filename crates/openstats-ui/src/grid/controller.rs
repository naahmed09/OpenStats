//! The grid's state and every operation on it: scrolling, selection, editing, formula bar,
//! undo/redo, clipboard and formatting. Input events are translated in [`super::input`] and
//! the picture is produced in [`super::render`].

use super::clipboard::{parse_tsv, to_tsv};
use super::format::{autosize, raw_text};
use super::metrics::{gutter_width, ColumnWidths, Metrics, View, MAX_COLS, MAX_ROWS};
use super::selection::{CellRange, CellRef, Selection};
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Rect, Theme};
use openstats_core::column::{Column, ColumnData, ColumnKind};
use openstats_core::datetime;
use openstats_core::formula::recalc::{recalc_edit, set_formula_edit};
use openstats_core::formula::FunctionRegistry;
use openstats_core::value::parse_number;
use openstats_core::{CoreError, Edit, History, NumericFormat, Result, Value, Workbook};
use std::collections::BTreeMap;

/// What the in-cell editor is editing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditTarget {
    /// A cell's value (or, when the text starts with `=`, its column's formula).
    Cell(CellRef),
    /// A column's name.
    ColumnName(usize),
}

/// The in-cell editor / formula-bar edit buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Editor {
    /// What is being edited.
    pub target: EditTarget,
    /// Text being typed (committed text only; an IME composition is kept apart).
    pub buffer: String,
    /// Caret position in characters.
    pub cursor: usize,
    /// Whether the edit was started from the formula bar.
    pub in_formula_bar: bool,
    /// The other end of the text selection (the selection is between it and the caret).
    pub anchor: Option<usize>,
    /// Text being composed by an input method (shown at the caret, not yet part of the buffer).
    pub composition: Option<Composition>,
}

/// An input-method composition in progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    /// The uncommitted text.
    pub text: String,
    /// Caret inside the composition, in characters.
    pub caret: usize,
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

impl Editor {
    pub(crate) fn new(target: EditTarget, buffer: String, in_formula_bar: bool) -> Editor {
        let cursor = buffer.chars().count();
        Editor { target, buffer, cursor, in_formula_bar, anchor: None, composition: None }
    }

    fn len(&self) -> usize {
        self.buffer.chars().count()
    }

    fn byte_at(&self, chars: usize) -> usize {
        self.buffer.char_indices().nth(chars).map_or(self.buffer.len(), |(i, _)| i)
    }

    /// The selected character range `(start, end)`, if any text is selected.
    pub fn selection(&self) -> Option<(usize, usize)> {
        let a = self.anchor?;
        (a != self.cursor).then(|| (a.min(self.cursor), a.max(self.cursor)))
    }

    /// The selected text.
    pub fn selected_text(&self) -> Option<String> {
        let (a, b) = self.selection()?;
        Some(self.buffer[self.byte_at(a)..self.byte_at(b)].to_string())
    }

    /// Remove the selection; returns whether there was one.
    pub fn delete_selection(&mut self) -> bool {
        let Some((a, b)) = self.selection() else {
            self.anchor = None;
            return false;
        };
        let (x, y) = (self.byte_at(a), self.byte_at(b));
        self.buffer.replace_range(x..y, "");
        self.cursor = a;
        self.anchor = None;
        true
    }

    /// Insert text at the caret, replacing the selection.
    pub fn insert(&mut self, s: &str) {
        self.composition = None;
        self.delete_selection();
        let at = self.byte_at(self.cursor);
        self.buffer.insert_str(at, s);
        self.cursor += s.chars().count();
    }

    /// Delete the selection, or the character before the caret.
    pub fn backspace(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor > 0 {
            let (a, b) = (self.byte_at(self.cursor - 1), self.byte_at(self.cursor));
            self.buffer.replace_range(a..b, "");
            self.cursor -= 1;
        }
    }

    /// Delete the selection, or the character after the caret.
    pub fn delete(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor < self.len() {
            let (a, b) = (self.byte_at(self.cursor), self.byte_at(self.cursor + 1));
            self.buffer.replace_range(a..b, "");
        }
    }

    fn set_cursor(&mut self, to: usize, extend: bool) {
        if extend {
            if self.anchor.is_none() {
                self.anchor = Some(self.cursor);
            }
        } else {
            self.anchor = None;
        }
        self.cursor = to.min(self.len());
    }

    /// Move the caret by `d` characters (clamped); `extend` grows the selection. Without
    /// `extend`, a selection collapses to its edge in the direction of travel.
    pub fn move_cursor(&mut self, d: isize, extend: bool) {
        if !extend {
            if let Some((a, b)) = self.selection() {
                self.anchor = None;
                self.cursor = if d < 0 { a } else { b };
                return;
            }
        }
        let t = (self.cursor as isize + d).clamp(0, self.len() as isize) as usize;
        self.set_cursor(t, extend);
    }

    /// Caret to the start or end (optionally extending the selection).
    pub fn cursor_to(&mut self, end: bool, extend: bool) {
        let t = if end { self.len() } else { 0 };
        self.set_cursor(t, extend);
    }

    /// Place the caret at character `i` (a click), optionally extending the selection (a drag).
    pub fn place_cursor(&mut self, i: usize, extend: bool) {
        self.set_cursor(i, extend);
    }

    /// Move to the previous (`dir < 0`) or next word boundary.
    pub fn move_word(&mut self, dir: isize, extend: bool) {
        let chars: Vec<char> = self.buffer.chars().collect();
        let mut i = self.cursor;
        if dir < 0 {
            while i > 0 && !is_word(chars[i - 1]) {
                i -= 1;
            }
            while i > 0 && is_word(chars[i - 1]) {
                i -= 1;
            }
        } else {
            while i < chars.len() && !is_word(chars[i]) {
                i += 1;
            }
            while i < chars.len() && is_word(chars[i]) {
                i += 1;
            }
        }
        self.set_cursor(i, extend);
    }

    /// Select the word around the caret (a double click), or the run of non-word characters.
    pub fn select_word(&mut self) {
        let chars: Vec<char> = self.buffer.chars().collect();
        if chars.is_empty() {
            return;
        }
        let i = self.cursor.min(chars.len() - 1);
        let kind = is_word(chars[i]);
        let (mut a, mut b) = (i, i + 1);
        while a > 0 && is_word(chars[a - 1]) == kind {
            a -= 1;
        }
        while b < chars.len() && is_word(chars[b]) == kind {
            b += 1;
        }
        self.anchor = Some(a);
        self.cursor = b;
    }

    /// Select everything.
    pub fn select_all(&mut self) {
        self.anchor = Some(0);
        self.cursor = self.len();
    }

    /// Show `text` as an input-method composition at the caret (it replaces the selection when
    /// it is committed). An empty `text` ends the composition.
    pub fn update_composition(&mut self, text: &str, caret: usize) {
        self.composition = if text.is_empty() {
            None
        } else {
            Some(Composition { text: text.to_string(), caret: caret.min(text.chars().count()) })
        };
    }

    /// Commit composed text: it is inserted like typing.
    pub fn commit_composition(&mut self, text: &str) {
        self.composition = None;
        if !text.is_empty() {
            self.insert(text);
        }
    }

    /// The text to draw: the buffer with any composition spliced in at the caret.
    pub fn display_text(&self) -> String {
        match &self.composition {
            None => self.buffer.clone(),
            Some(c) => {
                let at = self.byte_at(self.cursor);
                format!("{}{}{}", &self.buffer[..at], c.text, &self.buffer[at..])
            }
        }
    }

    /// Caret position in the displayed text (inside the composition while one is active).
    pub fn display_cursor(&self) -> usize {
        self.cursor + self.composition.as_ref().map_or(0, |c| c.caret)
    }

    /// Character index nearest to pixel offset `dx` from the start of the text.
    pub fn index_at(&self, dx: f64, size: f64) -> usize {
        let text = &self.buffer;
        let mut best = (0usize, f64::INFINITY);
        for i in 0..=self.len() {
            let w = text_width(&text.chars().take(i).collect::<String>(), size, false);
            let d = (w - dx).abs();
            if d < best.1 {
                best = (i, d);
            }
        }
        best.0
    }
}

/// Pointer drag in progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Drag {
    Select,
    Editor,
    Columns,
    Rows,
    Resize { col: usize, start_x: f64, start_w: f64 },
    VThumb { grab: f64 },
    HThumb { grab: f64 },
}

/// Pixel layout of the fixed parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    /// Width of the row-number gutter.
    pub gutter: f64,
    /// Height of the column header.
    pub header: f64,
    /// The scrolling body rectangle (view coordinates).
    pub body: Rect,
}

/// Metadata key prefix under which saved models are kept.
pub const MODEL_KEY: &str = "model.";

/// The grid: a workbook, its history and all view state.
#[derive(Debug)]
pub struct GridController {
    pub(crate) wb: Workbook,
    pub(crate) history: History,
    pub(crate) sheet: usize,
    pub(crate) registry: FunctionRegistry,
    pub(crate) now: Option<i64>,
    pub(crate) metrics: Metrics,
    pub(crate) widths: ColumnWidths,
    pub(crate) view: View,
    pub(crate) selection: Selection,
    pub(crate) editor: Option<Editor>,
    pub(crate) clipboard: Option<String>,
    pub(crate) drag: Option<Drag>,
    pub(crate) status: String,
    /// Decimal mark and date order used to show and read numbers and dates.
    pub(crate) locale: openstats_core::locale::Locale,
    /// Rows of the active worksheet highlighted by brushing in a graph.
    pub(crate) highlight: std::collections::BTreeSet<usize>,
    /// Colours and font sizes used by [`GridController::render`].
    pub theme: Theme,
}

impl GridController {
    /// A grid over sheet `0` of `wb` showing a `width` × `height` pixel window.
    pub fn new(wb: Workbook, width: f64, height: f64) -> GridController {
        let metrics = Metrics::default();
        let widths = ColumnWidths::new(metrics.default_col_width);
        let mut registry = FunctionRegistry::standard();
        openstats_math::calc::register(&mut registry);
        GridController {
            wb,
            history: History::default(),
            sheet: 0,
            registry,
            now: None,
            metrics,
            widths,
            view: View { width, height, scroll_x: 0.0, scroll_y: 0.0 },
            selection: Selection::new(),
            editor: None,
            clipboard: None,
            drag: None,
            status: String::new(),
            locale: openstats_core::locale::Locale::default(),
            highlight: std::collections::BTreeSet::new(),
            theme: Theme::light(),
        }
    }

    /// Set the value labels of column `col` (empty clears them) as one undo step.
    pub fn set_value_labels(&mut self, col: usize, labels: Vec<(String, String)>) -> Result<()> {
        let sheet = self.sheet;
        let c =
            self.ws().columns.get(col).ok_or_else(|| CoreError::InvalidArgument(format!("no column {}", col + 1)))?;
        let mut props = c.props.clone();
        props.value_labels = labels;
        if props == c.props {
            return Ok(());
        }
        self.history.execute(&mut self.wb, "Value labels".to_string(), Edit::SetColumnProps { sheet, col, props })
    }

    /// Use a locale's decimal mark and date order for display and for typed or pasted numbers and dates.
    pub fn set_locale(&mut self, locale: openstats_core::locale::Locale) {
        self.locale = locale;
    }

    /// The locale in use.
    pub fn locale(&self) -> openstats_core::locale::Locale {
        self.locale
    }

    /// The value for text typed or pasted into column `col`: numbers and dates are read with the locale when the
    /// column holds (or will hold) numbers or dates; anything else stays text.
    fn entry_value(&self, col: usize, s: &str) -> Value {
        let t = s.trim();
        if t.is_empty() || t == "*" {
            return Value::Missing;
        }
        if self.locale.is_default() {
            return Value::Text(s.to_string());
        }
        let kind = self.ws().columns.get(col).map(Column::kind);
        let numeric_ok = matches!(kind, None | Some(ColumnKind::Numeric));
        let date_ok = matches!(kind, None | Some(ColumnKind::DateTime));
        if numeric_ok && (self.locale.decimal != ',' || !t.contains('.') || grouped_thousands(t)) {
            if let Ok(x) = openstats_core::value::parse_number_with(t, self.locale.decimal) {
                return Value::Numeric(x);
            }
        }
        if date_ok {
            let o = datetime::ParseOptions { order: self.locale.date_order, ..Default::default() };
            if let Ok(ts) = datetime::parse_with(t, o) {
                return Value::DateTime(ts);
            }
        }
        Value::Text(s.to_string())
    }

    /// Keep at most `steps` undo steps (Tools > Options).
    pub fn set_undo_steps(&mut self, steps: usize) {
        let (_, bytes) = self.history.limits();
        self.history.set_limits(steps, bytes);
    }

    /// Rows to highlight (brushed in a graph); an empty set clears the highlight.
    pub fn set_highlight(&mut self, rows: std::collections::BTreeSet<usize>) {
        self.highlight = rows;
    }

    /// The highlighted rows.
    pub fn highlight(&self) -> &std::collections::BTreeSet<usize> {
        &self.highlight
    }

    // ---------------------------------------------------------------- accessors

    /// The workbook.
    pub fn workbook(&self) -> &Workbook {
        &self.wb
    }

    /// Names of the models saved in the project, alphabetically.
    pub fn model_names(&self) -> Vec<String> {
        self.wb.metadata.keys().filter_map(|k| k.strip_prefix(MODEL_KEY).map(str::to_string)).collect()
    }

    /// The JSON of the saved model `name`.
    pub fn model_json(&self, name: &str) -> Option<&str> {
        self.wb.metadata.get(&format!("{MODEL_KEY}{name}")).map(String::as_str)
    }

    /// Save (or replace) a model in the project. Models are stored with the project but are not part
    /// of the undo history.
    pub fn store_model(&mut self, name: &str, json: String) -> std::result::Result<(), String> {
        let n = name.trim();
        if n.is_empty() {
            return Err("give the model a name".into());
        }
        if n.chars().count() > 60 || n.chars().any(char::is_control) {
            return Err("a model name has at most 60 characters and no line breaks".into());
        }
        self.wb.metadata.insert(format!("{MODEL_KEY}{n}"), json);
        Ok(())
    }

    /// Delete a saved model; `false` when there is none of that name.
    pub fn remove_model(&mut self, name: &str) -> bool {
        self.wb.metadata.remove(&format!("{MODEL_KEY}{}", name.trim())).is_some()
    }

    /// Take the workbook back (history is dropped).
    pub fn into_workbook(self) -> Workbook {
        self.wb
    }

    /// The undo history.
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Index of the displayed sheet.
    pub fn sheet_index(&self) -> usize {
        self.sheet
    }

    /// Show another sheet; the selection, scroll and editor reset.
    pub fn set_sheet(&mut self, i: usize) -> Result<()> {
        self.wb.sheet(i)?;
        self.sheet = i;
        self.selection = Selection::new();
        self.editor = None;
        self.view.scroll_x = 0.0;
        self.view.scroll_y = 0.0;
        Ok(())
    }

    /// The selection.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// The active cell.
    pub fn active(&self) -> CellRef {
        self.selection.active()
    }

    /// The editor, while editing.
    pub fn editor(&self) -> Option<&Editor> {
        self.editor.as_ref()
    }

    /// Sizes.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Replace the sizes.
    pub fn set_metrics(&mut self, m: Metrics) {
        self.widths = ColumnWidths::new(m.default_col_width);
        self.metrics = m;
        self.clamp_scroll();
    }

    /// Scroll position and window size.
    pub fn view(&self) -> View {
        self.view
    }

    /// Last status or error message.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Set the clock used by date functions in formulas (microseconds since the epoch).
    pub fn set_now(&mut self, now: Option<i64>) {
        self.now = now;
    }

    /// Width of column `c`.
    pub fn column_width(&self, c: usize) -> f64 {
        self.widths.width(c)
    }

    /// Set the width of column `c` (clamped to the metric limits).
    pub fn set_column_width(&mut self, c: usize, w: f64) {
        self.widths.set(c, w.clamp(self.metrics.min_col_width, self.metrics.max_col_width));
        self.clamp_scroll();
    }

    /// Fit column `c` to its header and its first 2000 rows.
    pub fn autosize_column(&mut self, c: usize) {
        if let Ok(ws) = self.wb.sheet(self.sheet) {
            let w =
                autosize(ws, c, self.metrics.font_size, 2000, self.metrics.min_col_width, self.metrics.max_col_width);
            self.set_column_width(c, w);
        }
    }

    /// The displayed worksheet.
    pub fn worksheet(&self) -> &openstats_core::Worksheet {
        self.ws()
    }

    pub(crate) fn ws(&self) -> &openstats_core::Worksheet {
        // The sheet index is validated whenever it is set; fall back to the first sheet.
        self.wb.worksheets.get(self.sheet).unwrap_or_else(|| &self.wb.worksheets[0])
    }

    /// Number of data rows of the sheet.
    pub fn data_rows(&self) -> usize {
        self.ws().n_rows()
    }

    /// Number of data columns of the sheet.
    pub fn data_cols(&self) -> usize {
        self.ws().n_cols()
    }

    /// Rows shown (data rows plus blank rows to type into).
    pub fn display_rows(&self) -> usize {
        self.metrics.min_display_rows.max(self.data_rows() + self.metrics.spare_rows).min(MAX_ROWS)
    }

    /// Columns shown.
    pub fn display_cols(&self) -> usize {
        self.metrics.min_display_cols.max(self.data_cols() + self.metrics.spare_cols).min(MAX_COLS)
    }

    // ---------------------------------------------------------------- layout and scrolling

    /// Fixed-part layout for the current window.
    pub fn layout(&self) -> Layout {
        let gutter = gutter_width(self.display_rows(), self.metrics.font_size);
        let header = self.metrics.header_height;
        let body = Rect::new(
            gutter,
            header,
            (self.view.width - gutter - self.metrics.scrollbar).max(0.0),
            (self.view.height - header - self.metrics.scrollbar).max(0.0),
        );
        Layout { gutter, header, body }
    }

    /// Total content size `(width, height)` in pixels.
    pub fn content_size(&self) -> (f64, f64) {
        (self.widths.x_of(self.display_cols()), self.display_rows() as f64 * self.metrics.row_height)
    }

    /// Largest scroll offsets `(x, y)`.
    pub fn max_scroll(&self) -> (f64, f64) {
        let (cw, ch) = self.content_size();
        let b = self.layout().body;
        ((cw - b.w).max(0.0), (ch - b.h).max(0.0))
    }

    pub(crate) fn clamp_scroll(&mut self) {
        let (mx, my) = self.max_scroll();
        self.view.scroll_x = self.view.scroll_x.clamp(0.0, mx);
        self.view.scroll_y = self.view.scroll_y.clamp(0.0, my);
        if !self.view.scroll_x.is_finite() {
            self.view.scroll_x = 0.0;
        }
        if !self.view.scroll_y.is_finite() {
            self.view.scroll_y = 0.0;
        }
    }

    /// Resize the window.
    pub fn set_viewport(&mut self, width: f64, height: f64) {
        self.view.width = width.max(0.0);
        self.view.height = height.max(0.0);
        self.clamp_scroll();
    }

    /// Scroll by pixel deltas.
    pub fn scroll_by(&mut self, dx: f64, dy: f64) {
        self.view.scroll_x += dx;
        self.view.scroll_y += dy;
        self.clamp_scroll();
    }

    /// Scroll to absolute offsets.
    pub fn scroll_to(&mut self, x: f64, y: f64) {
        self.view.scroll_x = x;
        self.view.scroll_y = y;
        self.clamp_scroll();
    }

    /// Scroll so that row `row` is at the top.
    pub fn scroll_to_row(&mut self, row: usize) {
        self.view.scroll_y = row as f64 * self.metrics.row_height;
        self.clamp_scroll();
    }

    /// Scroll the minimum amount that makes `cell` fully visible.
    pub fn scroll_to_cell(&mut self, cell: CellRef) {
        let b = self.layout().body;
        let rh = self.metrics.row_height;
        let top = cell.row as f64 * rh;
        if top < self.view.scroll_y {
            self.view.scroll_y = top;
        } else if top + rh > self.view.scroll_y + b.h {
            self.view.scroll_y = top + rh - b.h;
        }
        let (left, w) = (self.widths.x_of(cell.col), self.widths.width(cell.col));
        if left < self.view.scroll_x {
            self.view.scroll_x = left;
        } else if left + w > self.view.scroll_x + b.w {
            self.view.scroll_x = (left + w - b.w).min(left);
        }
        self.clamp_scroll();
    }

    /// Visible data rows as `first .. end` (exclusive), including partially visible ones.
    pub fn visible_rows(&self) -> (usize, usize) {
        let b = self.layout().body;
        let rh = self.metrics.row_height;
        let first = (self.view.scroll_y / rh).floor() as usize;
        let end = (((self.view.scroll_y + b.h) / rh).ceil() as usize).min(self.display_rows());
        (first.min(end), end)
    }

    /// Visible columns as `first .. end` (exclusive).
    pub fn visible_cols(&self) -> (usize, usize) {
        let b = self.layout().body;
        let n = self.display_cols();
        let first = self.widths.col_at(self.view.scroll_x, n);
        let last = self.widths.col_at(self.view.scroll_x + b.w, n);
        (first, (last + 1).min(n))
    }

    /// Rectangle of `cell` in view coordinates (may lie outside the body).
    pub fn cell_rect(&self, cell: CellRef) -> Rect {
        let l = self.layout();
        Rect::new(
            l.gutter + self.widths.x_of(cell.col) - self.view.scroll_x,
            l.header + cell.row as f64 * self.metrics.row_height - self.view.scroll_y,
            self.widths.width(cell.col),
            self.metrics.row_height,
        )
    }

    // ---------------------------------------------------------------- selection

    /// Select a single cell and scroll to it.
    pub fn select_cell(&mut self, cell: CellRef) {
        let c = self.clamp_cell(cell);
        self.selection.select_cell(c);
        self.scroll_to_cell(c);
    }

    pub(crate) fn clamp_cell(&self, c: CellRef) -> CellRef {
        CellRef::new(c.row.min(self.display_rows() - 1), c.col.min(self.display_cols() - 1))
    }

    /// Move the active cell by `(dr, dc)`; with `extend` the current range grows instead.
    pub fn move_active(&mut self, dr: isize, dc: isize, extend: bool) {
        let a = self.selection.active();
        let n = |v: usize, d: isize| (v as isize + d).max(0) as usize;
        let t = self.clamp_cell(CellRef::new(n(a.row, dr), n(a.col, dc)));
        self.go_to(t, extend);
    }

    pub(crate) fn go_to(&mut self, t: CellRef, extend: bool) {
        if extend {
            self.selection.extend_to(t);
        } else {
            self.selection.move_to(t);
        }
        self.scroll_to_cell(t);
    }

    fn filled(&self, row: usize, col: usize) -> bool {
        self.ws().columns.get(col).is_some_and(|c| !c.data.is_missing(row))
    }

    /// Ctrl+arrow target: the edge of the current block of filled cells, or the next filled
    /// cell, or the edge of the sheet.
    pub fn jump_target(&self, from: CellRef, dr: isize, dc: isize) -> CellRef {
        let (rows, cols) = (self.display_rows(), self.display_cols());
        let step = |c: CellRef| -> Option<CellRef> {
            let (r, k) = (c.row as isize + dr, c.col as isize + dc);
            (r >= 0 && k >= 0 && (r as usize) < rows && (k as usize) < cols)
                .then(|| CellRef::new(r as usize, k as usize))
        };
        let Some(mut cur) = step(from) else { return from };
        let start_filled = self.filled(from.row, from.col);
        if start_filled && self.filled(cur.row, cur.col) {
            // Walk to the end of the block.
            while let Some(n) = step(cur) {
                if !self.filled(n.row, n.col) {
                    break;
                }
                cur = n;
            }
            return cur;
        }
        // Walk to the next filled cell (or the edge).
        while !self.filled(cur.row, cur.col) {
            match step(cur) {
                Some(n) => cur = n,
                None => break,
            }
        }
        cur
    }

    // ---------------------------------------------------------------- formula bar and editor

    /// Text of the cell `c` as the formula bar shows it: `=formula` for calculated columns,
    /// otherwise the full-precision value.
    pub fn cell_content(&self, c: CellRef) -> String {
        match self.ws().columns.get(c.col) {
            Some(col) => match &col.props.formula {
                Some(f) => format!("={f}"),
                None => raw_text(col, c.row),
            },
            None => String::new(),
        }
    }

    /// The formula bar text: the edit buffer while editing, otherwise the active cell.
    pub fn formula_bar_text(&self) -> String {
        match &self.editor {
            Some(e) => e.display_text(),
            None => self.cell_content(self.active()),
        }
    }

    /// The name box: column label and 1-based row, e.g. `C2[15]`.
    pub fn name_box(&self) -> String {
        let a = self.active();
        let (label, _) = super::format::header_parts(self.ws(), a.col);
        format!("{}[{}]", label.split('-').next().unwrap_or(&label), a.row + 1)
    }

    /// Start editing the active cell. `initial` replaces the content (typing a character);
    /// `None` edits the existing content (F2, double click, formula bar).
    pub fn begin_edit(&mut self, initial: Option<&str>, in_formula_bar: bool) {
        let a = self.active();
        let buffer = initial.map_or_else(|| self.cell_content(a), str::to_string);
        self.editor = Some(Editor::new(EditTarget::Cell(a), buffer, in_formula_bar));
        self.scroll_to_cell(a);
    }

    /// Start editing the name of column `c`.
    pub fn begin_rename(&mut self, c: usize) {
        let buffer = self.ws().columns.get(c).map(|x| x.name().to_string()).unwrap_or_default();
        self.editor = Some(Editor::new(EditTarget::ColumnName(c), buffer, false));
    }

    /// Text to copy from the editor: the selection, or `None` when nothing is selected.
    pub fn editor_copy(&self) -> Option<String> {
        self.editor.as_ref().and_then(Editor::selected_text)
    }

    /// Cut the editor's selection; returns the removed text.
    pub fn editor_cut(&mut self) -> Option<String> {
        let ed = self.editor.as_mut()?;
        let t = ed.selected_text()?;
        ed.delete_selection();
        Some(t)
    }

    /// Input-method composition update (preedit text and caret inside it).
    pub fn ime_update(&mut self, text: &str, caret: usize) {
        if let Some(ed) = self.editor.as_mut() {
            ed.update_composition(text, caret);
        }
    }

    /// Input-method commit. Outside an edit it starts one (so the first composed character of a
    /// language typed through an IME begins editing the active cell, like typing does).
    pub fn ime_commit(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.editor.is_none() {
            self.begin_edit(Some(text), false);
        } else if let Some(ed) = self.editor.as_mut() {
            ed.commit_composition(text);
        }
    }

    /// Cancel an input-method composition.
    pub fn ime_cancel(&mut self) {
        if let Some(ed) = self.editor.as_mut() {
            ed.composition = None;
        }
    }

    /// Where an input method should place its candidate window: the caret's position in view
    /// coordinates (`x`, top `y`, line height), if a cell is being edited.
    pub fn ime_anchor(&self) -> Option<(f64, f64, f64)> {
        let ed = self.editor.as_ref()?;
        let EditTarget::Cell(c) = &ed.target else { return None };
        let r = self.cell_rect(*c);
        let x = r.x
            + 4.0
            + text_width(
                &ed.display_text().chars().take(ed.display_cursor()).collect::<String>(),
                self.metrics.font_size,
                false,
            );
        Some((x, r.y, r.h))
    }

    /// Mutable access to the editor (typing, caret movement).
    pub fn editor_mut(&mut self) -> Option<&mut Editor> {
        self.editor.as_mut()
    }

    /// Abandon the edit.
    pub fn cancel_edit(&mut self) {
        self.editor = None;
    }

    /// Apply the edit. On failure the editor stays open and the error is returned.
    pub fn commit_edit(&mut self) -> Result<()> {
        let Some(ed) = self.editor.take() else { return Ok(()) };
        let r = match &ed.target {
            EditTarget::Cell(c) => self.enter_cell(*c, &ed.buffer),
            EditTarget::ColumnName(c) => self.rename_column(*c, ed.buffer.trim()),
        };
        match r {
            Ok(()) => {
                self.status.clear();
                Ok(())
            }
            Err(e) => {
                self.status = e.to_string();
                self.editor = Some(ed);
                Err(e)
            }
        }
    }

    /// Enter `text` into a cell: a value, `*` or empty for missing, or `=formula` to give the
    /// whole column a formula (`=` alone removes it).
    pub fn enter_cell(&mut self, cell: CellRef, text: &str) -> Result<()> {
        let t = text.trim();
        if let Some(f) = t.strip_prefix('=') {
            let f = f.trim();
            return self.set_formula(cell.col, (!f.is_empty()).then_some(f));
        }
        let v = self.entry_value(cell.col, text);
        self.set_cells("Edit cell", vec![(cell.row, cell.col, v)])
    }

    // ---------------------------------------------------------------- edits

    fn has_formulas(&self) -> bool {
        self.ws().columns.iter().any(|c| c.props.formula.is_some())
    }

    /// Run `f` as one undo step; on error everything done so far is rolled back.
    fn transaction(&mut self, label: &str, f: impl FnOnce(&mut GridController) -> Result<()>) -> Result<()> {
        self.history.begin_group(label);
        match f(self) {
            Ok(()) => self.history.end_group(),
            Err(e) => {
                let _ = self.history.cancel_group(&mut self.wb);
                Err(e)
            }
        }
    }

    /// Detect the kind of a column that is being filled for the first time.
    fn adopt_kind(values: &[(usize, Value)]) -> Option<ColumnKind> {
        let mut kind: Option<ColumnKind> = None;
        let mut numeric = true;
        let mut dates = true;
        let mut any = false;
        for (_, v) in values {
            match v {
                Value::Missing => {}
                Value::Numeric(x) if x.is_nan() => {}
                Value::Numeric(_) => {
                    any = true;
                    dates = false;
                    kind = Some(ColumnKind::Numeric);
                }
                Value::DateTime(_) => {
                    any = true;
                    numeric = false;
                    kind = Some(ColumnKind::DateTime);
                }
                Value::Text(s) => {
                    any = true;
                    let s = s.trim();
                    if s.is_empty() || s == "*" {
                        continue;
                    }
                    numeric &= parse_number(s).is_ok();
                    dates &= datetime::parse(s).is_ok();
                }
            }
        }
        if !any {
            return None;
        }
        if let Some(k) = kind {
            return Some(k);
        }
        Some(if numeric {
            ColumnKind::Numeric
        } else if dates {
            ColumnKind::DateTime
        } else {
            ColumnKind::Text
        })
    }

    /// Set many cells as one undo step. Columns past the end are created; an empty column
    /// takes the kind of what is entered first (number, date/time, else text). Missing values
    /// past the end of a column are ignored. Formula columns cannot be edited. Large changes
    /// replace whole columns instead of recording one edit per cell. Atomic: on any error
    /// nothing changes. Formulas are recalculated afterwards.
    pub fn set_cells(&mut self, label: &str, cells: Vec<(usize, usize, Value)>) -> Result<()> {
        if cells.is_empty() {
            return Ok(());
        }
        let max_row = cells.iter().map(|c| c.0).max().unwrap_or(0);
        let max_col = cells.iter().map(|c| c.1).max().unwrap_or(0);
        if max_row >= MAX_ROWS {
            return Err(CoreError::InvalidArgument(format!("a worksheet has at most {MAX_ROWS} rows")));
        }
        if max_col >= MAX_COLS {
            return Err(CoreError::InvalidArgument(format!("a worksheet has at most {MAX_COLS} columns")));
        }
        let mut by_col: BTreeMap<usize, Vec<(usize, Value)>> = BTreeMap::new();
        for (r, c, v) in cells {
            by_col.entry(c).or_default().push((r, v));
        }
        let sheet = self.sheet;
        self.transaction(label, move |g| {
            let n_cols = g.data_cols();
            let mut edits: Vec<Edit> = (n_cols..=max_col)
                .map(|at| Edit::InsertColumn { sheet, at, column: Column::numeric("", vec![]) })
                .collect();
            for (col, vals) in &by_col {
                let existing: Option<&Column> = g.ws().columns.get(*col);
                if let Some(c) = existing {
                    if let Some(f) = &c.props.formula {
                        return Err(CoreError::InvalidArgument(format!(
                            "{} is calculated by the formula ={f}; edit the formula instead",
                            g.ws().display_name(*col)
                        )));
                    }
                }
                let len = existing.map_or(0, Column::len);
                let vals: Vec<(usize, Value)> =
                    vals.iter().filter(|(r, v)| *r < len || !v.is_missing()).cloned().collect();
                if vals.is_empty() {
                    continue;
                }
                let adopt = if len == 0 {
                    GridController::adopt_kind(&vals).filter(|k| Some(*k) != existing.map(Column::kind))
                } else {
                    None
                };
                if vals.len() > 64 || adopt.is_some() && vals.len() > 1 {
                    let mut c = existing.cloned().unwrap_or_else(|| Column::numeric("", vec![]));
                    if let Some(k) = adopt {
                        c.data = ColumnData::empty(k);
                    }
                    for (r, v) in &vals {
                        c.data.set(*r, v)?;
                    }
                    edits.push(Edit::ReplaceColumn { sheet, col: *col, column: c });
                } else {
                    if let Some(k) = adopt {
                        let mut c = existing.cloned().unwrap_or_else(|| Column::numeric("", vec![]));
                        c.data = ColumnData::empty(k);
                        edits.push(Edit::ReplaceColumn { sheet, col: *col, column: c });
                    }
                    for (r, v) in vals {
                        edits.push(Edit::SetCell { sheet, col: *col, row: r, value: v });
                    }
                }
            }
            g.history.execute(&mut g.wb, label.to_string(), Edit::Batch(edits))?;
            g.recalc_if_needed()
        })
    }

    fn recalc_if_needed(&mut self) -> Result<()> {
        if self.has_formulas() {
            let e = recalc_edit(&self.wb, self.sheet, &self.registry, self.now)?;
            self.history.execute(&mut self.wb, "Recalculate", e)?;
        }
        Ok(())
    }

    /// Run `f` (a script run or step) over the workbook and its history as one undo step, then
    /// recalculate formula columns inside the same step.
    pub fn run_script_step<R>(
        &mut self,
        label: &str,
        f: impl FnOnce(&mut openstats_core::Workbook, &mut openstats_core::history::History) -> R,
    ) -> Result<R> {
        self.editor = None;
        self.wb.active = self.sheet;
        self.history.begin_group(label);
        let r = f(&mut self.wb, &mut self.history);
        let recalc = self.recalc_if_needed();
        self.history.end_group()?;
        recalc?;
        Ok(r)
    }

    /// Give column `col` a formula (`None` removes it, keeping the values) and recalculate it
    /// and the columns that depend on it. One undo step.
    pub fn set_formula(&mut self, col: usize, formula: Option<&str>) -> Result<()> {
        if col >= MAX_COLS {
            return Err(CoreError::InvalidArgument(format!("a worksheet has at most {MAX_COLS} columns")));
        }
        let sheet = self.sheet;
        self.transaction("Set formula", |g| {
            let n = g.data_cols();
            if col >= n {
                let ins: Vec<Edit> =
                    (n..=col).map(|at| Edit::InsertColumn { sheet, at, column: Column::numeric("", vec![]) }).collect();
                g.history.execute(&mut g.wb, "Add columns", Edit::Batch(ins))?;
            }
            let e = set_formula_edit(&g.wb, sheet, col, formula, &g.registry, g.now)?;
            g.history.execute(&mut g.wb, "Set formula", e)
        })
    }

    /// Rename a column (empty clears the name).
    pub fn rename_column(&mut self, col: usize, name: &str) -> Result<()> {
        let ws = self.ws();
        ws.column(col)?;
        ws.validate_name(name, Some(col))?;
        let e = Edit::rename_column(&self.wb, self.sheet, col, name)?;
        self.history.execute(&mut self.wb, "Rename column", e)
    }

    /// Clear (make missing) every selected cell that holds data; cells of formula columns are
    /// left alone. One undo step.
    pub fn clear_selection(&mut self) -> Result<()> {
        let cells = self.selected_data_cells(true);
        self.set_cells("Clear cells", cells.into_iter().map(|(r, c)| (r, c, Value::Missing)).collect())
    }

    /// Cells inside the data extent covered by the selection.
    fn selected_data_cells(&self, skip_formulas: bool) -> Vec<(usize, usize)> {
        let ws = self.ws();
        let mut out = Vec::new();
        for r in self.selection.ranges() {
            for c in r.c0..=r.c1.min(ws.n_cols().saturating_sub(1)) {
                let Some(col) = ws.columns.get(c) else { continue };
                if skip_formulas && col.props.formula.is_some() {
                    continue;
                }
                for row in r.r0..=r.r1.min(col.len().saturating_sub(1)) {
                    if row < col.len() {
                        out.push((row, c));
                    }
                }
            }
        }
        out
    }

    /// Undo the last edit; returns its label.
    pub fn undo(&mut self) -> Result<String> {
        self.editor = None;
        let l = self.history.undo(&mut self.wb)?;
        self.clamp_scroll();
        Ok(l)
    }

    /// Redo the last undone edit; returns its label.
    pub fn redo(&mut self) -> Result<String> {
        self.editor = None;
        let l = self.history.redo(&mut self.wb)?;
        self.clamp_scroll();
        Ok(l)
    }

    // ---------------------------------------------------------------- clipboard

    /// Text of the current range as tab-separated values: numbers at full precision, `*` for
    /// missing values inside the data, blank beyond it. Also kept as the internal clipboard.
    pub fn copy(&mut self) -> Result<String> {
        let r = self.selection.current().clamp(self.data_rows(), self.data_cols());
        if r.n_cells() > 20_000_000 {
            return Err(CoreError::InvalidArgument(String::from("the selection is too large to copy")));
        }
        let ws = self.ws();
        let rows: Vec<Vec<String>> = (r.r0..=r.r1)
            .map(|row| {
                (r.c0..=r.c1)
                    .map(|c| match ws.columns.get(c) {
                        Some(col) if row < col.len() => {
                            if col.data.is_missing(row) {
                                "*".to_string()
                            } else {
                                raw_text(col, row)
                            }
                        }
                        _ => String::new(),
                    })
                    .collect()
            })
            .collect();
        let text = to_tsv(&rows);
        self.clipboard = Some(text.clone());
        Ok(text)
    }

    /// Copy, then clear the copied cells (one undo step for the clear).
    pub fn cut(&mut self) -> Result<String> {
        let t = self.copy()?;
        self.clear_selection()?;
        Ok(t)
    }

    /// The internal clipboard text, for hosts without system clipboard access.
    pub fn clipboard_text(&self) -> Option<&str> {
        self.clipboard.as_deref()
    }

    /// Paste tab-separated text at the top-left of the current range; a single pasted value
    /// fills a larger selection. Returns the number of cells written. One undo step; atomic.
    pub fn paste_text(&mut self, text: &str) -> Result<usize> {
        if let Some(ed) = self.editor.as_mut() {
            // While editing, pasted text goes into the editor (one line: line breaks become spaces).
            let one: String = text.trim_end_matches(['\r', '\n']).replace("\r\n", " ").replace(['\r', '\n', '\t'], " ");
            ed.insert(&one);
            return Ok(0);
        }
        let rows = parse_tsv(text);
        if rows.is_empty() {
            return Ok(0);
        }
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        let sel = self.selection.current().clamp(self.data_rows(), self.data_cols());
        let to_value = |col: usize, s: &str| self.entry_value(col, s);
        let mut cells: Vec<(usize, usize, Value)> = Vec::new();
        let target: CellRange;
        if rows.len() == 1 && width == 1 && sel.n_cells() > 1 {
            target = sel;
            for r in sel.r0..=sel.r1 {
                for c in sel.c0..=sel.c1 {
                    cells.push((r, c, to_value(c, &rows[0][0])));
                }
            }
        } else {
            let (r0, c0) = (sel.r0, sel.c0);
            if r0 + rows.len() > MAX_ROWS || c0 + width > MAX_COLS {
                return Err(CoreError::InvalidArgument(String::from("the pasted block does not fit in the worksheet")));
            }
            target = CellRange { r0, c0, r1: r0 + rows.len() - 1, c1: c0 + width - 1 };
            for (i, row) in rows.iter().enumerate() {
                for (j, f) in row.iter().enumerate() {
                    cells.push((r0 + i, c0 + j, to_value(c0 + j, f)));
                }
            }
        }
        let n = cells.len();
        self.set_cells("Paste", cells)?;
        self.selection.select_cell(CellRef::new(target.r0, target.c0));
        self.selection.extend_to(CellRef::new(target.r1, target.c1));
        self.scroll_to_cell(CellRef::new(target.r0, target.c0));
        Ok(n)
    }

    // ---------------------------------------------------------------- sheets

    /// Store matrix `M{m}` of the active worksheet (`None` erases it) as one undo step.
    pub fn set_matrix(&mut self, m: usize, matrix: Option<openstats_core::StoredMatrix>) -> Result<()> {
        let sheet = self.sheet;
        self.run_script_step("Set matrix", |wb, hist| {
            hist.execute(wb, "Set matrix", openstats_core::Edit::SetMatrix { sheet, m, matrix })
        })?
    }

    /// Store constant `K{k}` of the active worksheet as one undo step.
    pub fn set_constant(&mut self, k: usize, value: f64) -> Result<()> {
        let sheet = self.sheet;
        self.run_script_step("Set constant", |wb, hist| {
            hist.execute(
                wb,
                "Set constant",
                openstats_core::Edit::SetConstant {
                    sheet,
                    k,
                    value: Some(openstats_core::value::Value::Numeric(value)),
                },
            )
        })?
    }

    pub(crate) fn wb_mut(&mut self) -> &mut Workbook {
        &mut self.wb
    }

    /// Number of worksheets.
    pub fn sheet_count(&self) -> usize {
        self.wb.worksheets.len()
    }

    fn check_sheet_name(&self, name: &str, except: Option<usize>) -> Result<()> {
        let n = name.trim();
        if n.is_empty() {
            return Err(CoreError::InvalidArgument(String::from("a worksheet needs a name")));
        }
        if self.wb.worksheets.iter().enumerate().any(|(i, w)| Some(i) != except && w.name.eq_ignore_ascii_case(n)) {
            return Err(CoreError::Duplicate(n.to_string()));
        }
        Ok(())
    }

    /// Rename worksheet `i`. Structural sheet operations are not part of the undo history.
    pub fn rename_sheet(&mut self, i: usize, name: &str) -> Result<()> {
        self.wb.sheet(i)?;
        self.check_sheet_name(name, Some(i))?;
        self.wb.worksheets[i].name = name.trim().to_string();
        Ok(())
    }

    /// Append an empty worksheet (a name like `Worksheet 3` is chosen when `name` is `None`)
    /// and return its index. The undo history is cleared because edits address sheets by index.
    pub fn add_sheet(&mut self, name: Option<&str>) -> Result<usize> {
        let name = match name {
            Some(n) => n.trim().to_string(),
            None => (self.wb.worksheets.len() + 1..)
                .map(|k| format!("Worksheet {k}"))
                .find(|n| self.check_sheet_name(n, None).is_ok())
                .unwrap_or_default(),
        };
        self.check_sheet_name(&name, None)?;
        self.wb.worksheets.push(openstats_core::Worksheet::new(name));
        self.history.clear();
        Ok(self.wb.worksheets.len() - 1)
    }

    /// Copy worksheet `i` under a new name and return the copy's index.
    /// Append worksheets (a data import). Names are made unique with a numeric suffix. A project
    /// that is still a single empty worksheet has that sheet replaced, so a first import does not
    /// leave an unused "Worksheet 1" behind. Clears the undo history. Returns the index of the
    /// first imported sheet.
    pub fn import_sheets(&mut self, sheets: Vec<openstats_core::Worksheet>) -> Result<usize> {
        if sheets.is_empty() {
            return Err(CoreError::InvalidArgument("the file contains no worksheets".into()));
        }
        if self.wb.worksheets.len() == 1 && self.wb.worksheets[0].n_cols() == 0 {
            self.wb.worksheets.clear();
        }
        let first = self.wb.worksheets.len();
        for mut w in sheets {
            let base = if w.name.trim().is_empty() { "Data".to_string() } else { w.name.trim().to_string() };
            w.name = (1..)
                .map(|k| if k == 1 { base.clone() } else { format!("{base} ({k})") })
                .find(|n| self.check_sheet_name(n, None).is_ok())
                .unwrap_or(base);
            self.wb.worksheets.push(w);
        }
        self.history.clear();
        Ok(first)
    }

    /// Duplicate worksheet `i` as "NAME (copy)".
    pub fn duplicate_sheet(&mut self, i: usize) -> Result<usize> {
        let mut w = self.wb.sheet(i)?.clone();
        let base = format!("{} (copy)", w.name);
        w.name = (1..)
            .map(|k| if k == 1 { base.clone() } else { format!("{base} {k}") })
            .find(|n| self.check_sheet_name(n, None).is_ok())
            .unwrap_or(base);
        self.wb.worksheets.push(w);
        self.history.clear();
        Ok(self.wb.worksheets.len() - 1)
    }

    /// Delete worksheet `i` (the last worksheet cannot be deleted).
    pub fn remove_sheet(&mut self, i: usize) -> Result<()> {
        self.wb.sheet(i)?;
        if self.wb.worksheets.len() == 1 {
            return Err(CoreError::InvalidArgument(String::from("a project needs at least one worksheet")));
        }
        self.wb.worksheets.remove(i);
        self.history.clear();
        self.editor = None;
        self.sheet = if self.sheet > i { self.sheet - 1 } else { self.sheet.min(self.wb.worksheets.len() - 1) };
        self.wb.active = self.sheet;
        self.selection = Selection::new();
        self.clamp_scroll();
        Ok(())
    }

    // ---------------------------------------------------------------- stored results

    /// Store `cols` after the last used column (Minitab's default for stored results) as one
    /// undo step. Colliding names get a numeric suffix. Returns the index of the first column
    /// written.
    pub fn store_columns(&mut self, cols: Vec<Column>) -> Result<usize> {
        if cols.is_empty() {
            return Ok(self.ws().next_free_column());
        }
        let first = self.ws().next_free_column();
        if first + cols.len() > MAX_COLS {
            return Err(CoreError::InvalidArgument(String::from("the worksheet would exceed the column limit")));
        }
        let sheet = self.sheet;
        self.transaction("Store results", |g| {
            let e = openstats_core::ops::store_columns(&g.wb, sheet, cols, openstats_core::ops::Target::Append)?;
            g.history.execute(&mut g.wb, "Store results", e)?;
            g.recalc_if_needed()
        })?;
        Ok(first)
    }

    /// Add worksheets to the project as one undo step. Colliding names get a numeric suffix.
    /// Returns the index of the first worksheet added.
    pub fn add_worksheets(&mut self, sheets: Vec<openstats_core::Worksheet>) -> Result<usize> {
        let first = self.wb.worksheets.len();
        if sheets.is_empty() {
            return Ok(first);
        }
        self.transaction("Add worksheets", |g| {
            let e = openstats_core::ops::add_worksheets(&g.wb, sheets)?;
            g.history.execute(&mut g.wb, "Add worksheets", e)?;
            Ok(())
        })?;
        Ok(first)
    }

    /// Sort all rows by column `key` (missing values last) as one undo step. Calculated
    /// columns are recomputed rather than permuted.
    pub fn sort_rows(&mut self, key: usize, descending: bool) -> Result<()> {
        self.ws().column(key)?;
        let carry: Vec<usize> =
            (0..self.data_cols()).filter(|c| self.ws().columns[*c].props.formula.is_none()).collect();
        let sort_key = if descending {
            openstats_core::ops::sort::SortKey::desc(key)
        } else {
            openstats_core::ops::sort::SortKey::asc(key)
        };
        let cols = openstats_core::ops::sort::sort_columns(self.ws(), &carry, &[sort_key])?;
        let sheet = self.sheet;
        self.transaction("Sort", |g| {
            let e =
                openstats_core::ops::store_columns(&g.wb, sheet, cols, openstats_core::ops::Target::Columns(carry))?;
            g.history.execute(&mut g.wb, "Sort", e)?;
            g.recalc_if_needed()
        })
    }

    /// Calculator: evaluate `expr` and store the values (not the formula) in column `col`.
    pub fn calculate(&mut self, col: usize, expr: &str) -> Result<()> {
        if self.ws().columns.get(col).is_some_and(|c| c.props.formula.is_some()) {
            return Err(CoreError::InvalidArgument(String::from(
                "that column is calculated by a formula; remove the formula first",
            )));
        }
        self.transaction("Calculator", |g| {
            g.set_formula(col, Some(expr))?;
            g.set_formula(col, None)
        })
    }

    // ---------------------------------------------------------------- formatting and rows

    fn selected_columns(&self) -> Vec<usize> {
        let n = self.data_cols();
        let mut cols: Vec<usize> = self
            .selection
            .ranges()
            .iter()
            .flat_map(|r| r.c0..=r.c1.min(n.saturating_sub(1)))
            .filter(|c| *c < n)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        cols
    }

    fn update_props(
        &mut self,
        label: &str,
        only_numeric: bool,
        f: impl Fn(&mut openstats_core::ColumnProps),
    ) -> Result<()> {
        let sheet = self.sheet;
        let mut edits = Vec::new();
        for c in self.selected_columns() {
            let col = &self.ws().columns[c];
            if only_numeric && col.kind() != ColumnKind::Numeric {
                continue;
            }
            let mut props = col.props.clone();
            f(&mut props);
            if props != col.props {
                edits.push(Edit::SetColumnProps { sheet, col: c, props });
            }
        }
        if edits.is_empty() {
            return Ok(());
        }
        self.history.execute(&mut self.wb, label.to_string(), Edit::Batch(edits))
    }

    /// Numeric display format of the selected numeric columns (affects display only).
    pub fn set_number_format(&mut self, fmt: NumericFormat) -> Result<()> {
        self.update_props("Number format", true, |p| p.numeric_format = fmt)
    }

    /// Date/time pattern of the selected columns (`None` = default pattern).
    pub fn set_datetime_format(&mut self, pattern: Option<&str>) -> Result<()> {
        let p = pattern.map(str::to_string);
        let sheet = self.sheet;
        let mut edits = Vec::new();
        for c in self.selected_columns() {
            let col = &self.ws().columns[c];
            if col.kind() != ColumnKind::DateTime || col.props.datetime_format == p {
                continue;
            }
            let mut props = col.props.clone();
            props.datetime_format.clone_from(&p);
            edits.push(Edit::SetColumnProps { sheet, col: c, props });
        }
        if edits.is_empty() {
            return Ok(());
        }
        self.history.execute(&mut self.wb, "Date/time format".to_string(), Edit::Batch(edits))
    }

    /// Show one more (`+1`) or fewer (`-1`) decimals in the selected numeric columns.
    pub fn change_decimals(&mut self, delta: i32) -> Result<()> {
        self.update_props("Decimals", true, |p| {
            let cur = match p.numeric_format {
                NumericFormat::Fixed(d) => i32::from(d),
                _ => 2 - delta.signum(),
            };
            p.numeric_format = NumericFormat::Fixed((cur + delta).clamp(0, 15) as u8);
        })
    }

    /// Insert blank rows above the first selected row (as many as the selection spans).
    pub fn insert_rows(&mut self) -> Result<()> {
        let r = self.selection.current();
        let at = r.r0.min(self.data_rows());
        let n = r.r1.min(r.r0.saturating_add(10_000)).saturating_sub(r.r0) + 1;
        if self.data_rows() + n > MAX_ROWS {
            return Err(CoreError::InvalidArgument(String::from("the worksheet would exceed the row limit")));
        }
        let e = Edit::insert_blank_rows(&self.wb, self.sheet, at, n)?;
        self.history.execute(&mut self.wb, "Insert rows".to_string(), e)
    }

    /// Delete the rows of the current range (data rows only).
    pub fn delete_rows(&mut self) -> Result<()> {
        let r = self.selection.current();
        let rows = self.data_rows();
        if r.r0 >= rows {
            return Ok(());
        }
        let n = r.r1.min(rows - 1) - r.r0 + 1;
        let e = Edit::remove_rows_all(&self.wb, self.sheet, r.r0, n)?;
        self.history.execute(&mut self.wb, "Delete rows".to_string(), e)?;
        self.clamp_scroll();
        Ok(())
    }
}

/// Whether `t` is digits with `.` used only as a thousands mark (`1.234`, `12.345.678`, with an optional `,` decimal part).
fn grouped_thousands(t: &str) -> bool {
    let int = t.split(',').next().unwrap_or(t).trim_start_matches(['-', '+']);
    let parts: Vec<&str> = int.split('.').collect();
    parts.len() > 1
        && parts[0].len() <= 3
        && !parts[0].is_empty()
        && parts[1..].iter().all(|p| p.len() == 3)
        && parts.iter().all(|p| p.bytes().all(|b| b.is_ascii_digit()))
}
