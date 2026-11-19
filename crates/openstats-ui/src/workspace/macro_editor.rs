//! The Macro Editor panel: a plain-text editor for `.mac` scripts with line numbers, syntax
//! colours, breakpoints, a Run / Step / Stop toolbar and an error strip.
//!
//! The editor owns the text and the cursor; running the script is the workspace's job (it owns
//! the worksheet). The workspace tells the editor where execution is
//! ([`MacroEditor::set_exec_line`]), what went wrong ([`MacroEditor::set_error`]) and what to
//! write in the toolbar ([`MacroEditor::set_run_state`]).

use crate::grid::{Key, Mods};
use crate::style::Chrome;
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Color, Point, Rect, Scene, Stroke, TextStyle, Theme};
use openstats_script::highlight::{highlight_line, Class};
use std::collections::BTreeSet;

const SIZE: f64 = 12.0;
const LINE_H: f64 = 17.0;
const TOOLBAR: f64 = 30.0;
const GUTTER: f64 = 58.0;
const ERROR_STRIP: f64 = 36.0;
const MAX_UNDO: usize = 200;
const TAB: &str = "  ";

/// What an input did that the workspace must react to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorEvent {
    /// Nothing outside the editor.
    None,
    /// The text changed.
    Edited,
    /// The Run button, F5.
    Run,
    /// The Step button, F8.
    Step,
    /// The Stop button, Shift+F5.
    Stop,
    /// A breakpoint was set or cleared.
    Breakpoints,
}

/// Where the current run is, for the toolbar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunState {
    /// No run in progress.
    #[default]
    Idle,
    /// Stopped between lines; Run continues, Step advances, Stop abandons.
    Paused,
}

/// Position in the text: line index and character index in that line.
type Pos = (usize, usize);

#[derive(Debug, Clone, PartialEq)]
struct Snapshot {
    lines: Vec<String>,
    cursor: Pos,
}

/// Which toolbar button a point is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Button {
    Run,
    Step,
    Stop,
}

/// The editor.
#[derive(Debug, Clone)]
pub struct MacroEditor {
    lines: Vec<String>,
    cursor: Pos,
    anchor: Option<Pos>,
    scroll_y: f64,
    scroll_x: f64,
    breakpoints: BTreeSet<usize>,
    exec_line: Option<usize>,
    error: Option<(usize, String)>,
    state: RunState,
    message: String,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    typing: bool,
    dragging: bool,
    /// Whether the editor has keyboard focus (draws the caret).
    pub focused: bool,
    /// Width in pixels.
    pub width: f64,
    /// Height in pixels.
    pub height: f64,
    /// Colours.
    pub theme: Theme,
}

impl MacroEditor {
    /// An empty editor of `width × height` pixels.
    pub fn new(width: f64, height: f64) -> MacroEditor {
        MacroEditor {
            lines: vec![String::new()],
            cursor: (0, 0),
            anchor: None,
            scroll_y: 0.0,
            scroll_x: 0.0,
            breakpoints: BTreeSet::new(),
            exec_line: None,
            error: None,
            state: RunState::Idle,
            message: String::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            typing: false,
            dragging: false,
            focused: false,
            width,
            height,
            theme: Theme::light(),
        }
    }

    // ---------------------------------------------------------------- text

    /// The whole text, lines joined with `\n`.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Replace the text (clears undo history, breakpoints, error and execution marker).
    pub fn set_text(&mut self, text: &str) {
        self.lines = text.replace("\r\n", "\n").replace('\r', "\n").split('\n').map(str::to_string).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor = (0, 0);
        self.anchor = None;
        self.undo.clear();
        self.redo.clear();
        self.typing = false;
        self.breakpoints.clear();
        self.exec_line = None;
        self.error = None;
        self.scroll_y = 0.0;
        self.scroll_x = 0.0;
    }

    /// Number of lines.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Text of 1-based line `n`.
    pub fn line(&self, n: usize) -> Option<&str> {
        self.lines.get(n.wrapping_sub(1)).map(String::as_str)
    }

    /// Cursor as 1-based `(line, column)`.
    pub fn cursor(&self) -> (usize, usize) {
        (self.cursor.0 + 1, self.cursor.1 + 1)
    }

    /// Put the cursor at 1-based `(line, column)` (clamped) and scroll it into view.
    pub fn set_cursor(&mut self, line: usize, col: usize) {
        let l = line.saturating_sub(1).min(self.lines.len() - 1);
        let c = col.saturating_sub(1).min(self.lines[l].chars().count());
        self.cursor = (l, c);
        self.anchor = None;
        self.typing = false;
        self.reveal();
    }

    /// Resize.
    pub fn set_size(&mut self, w: f64, h: f64) {
        self.width = w;
        self.height = h;
        self.clamp_scroll();
    }

    // ---------------------------------------------------------------- run state

    /// Breakpoints (1-based lines).
    pub fn breakpoints(&self) -> &BTreeSet<usize> {
        &self.breakpoints
    }

    /// Set or clear the breakpoint on 1-based `line`.
    pub fn toggle_breakpoint(&mut self, line: usize) {
        if line >= 1 && line <= self.lines.len() && !self.breakpoints.remove(&line) {
            self.breakpoints.insert(line);
        }
    }

    /// Mark the line that will execute next (1-based), scrolling it into view.
    pub fn set_exec_line(&mut self, line: Option<usize>) {
        self.exec_line = line.filter(|l| *l >= 1 && *l <= self.lines.len());
        if let Some(l) = self.exec_line {
            self.reveal_line(l - 1);
        }
    }

    /// The line marked as next to execute.
    pub fn exec_line(&self) -> Option<usize> {
        self.exec_line
    }

    /// Show an error at 1-based `line` (`0` for no particular line) or clear it.
    pub fn set_error(&mut self, err: Option<(usize, String)>) {
        if let Some((l, _)) = &err {
            if *l >= 1 && *l <= self.lines.len() {
                self.reveal_line(l - 1);
            }
        }
        self.error = err;
    }

    /// The error being shown.
    pub fn error(&self) -> Option<&(usize, String)> {
        self.error.as_ref()
    }

    /// Toolbar state and message (shown right of the buttons).
    pub fn set_run_state(&mut self, state: RunState, message: impl Into<String>) {
        self.state = state;
        self.message = message.into();
    }

    /// Toolbar state.
    pub fn run_state(&self) -> RunState {
        self.state
    }

    /// Toolbar message.
    pub fn message(&self) -> &str {
        &self.message
    }

    // ---------------------------------------------------------------- geometry

    fn text_area(&self) -> Rect {
        let strip = if self.error.is_some() { ERROR_STRIP } else { 0.0 };
        Rect::new(GUTTER, TOOLBAR, (self.width - GUTTER).max(0.0), (self.height - TOOLBAR - strip).max(0.0))
    }

    fn buttons(&self) -> [(Button, Rect); 3] {
        let w = 64.0;
        [
            (Button::Run, Rect::new(6.0, 4.0, w, TOOLBAR - 8.0)),
            (Button::Step, Rect::new(6.0 + w + 4.0, 4.0, w, TOOLBAR - 8.0)),
            (Button::Stop, Rect::new(6.0 + 2.0 * (w + 4.0), 4.0, w, TOOLBAR - 8.0)),
        ]
    }

    fn button_enabled(&self, b: Button) -> bool {
        match b {
            Button::Run | Button::Step => true,
            Button::Stop => self.state == RunState::Paused,
        }
    }

    fn content_height(&self) -> f64 {
        self.lines.len() as f64 * LINE_H
    }

    fn clamp_scroll(&mut self) {
        let max = (self.content_height() - self.text_area().h + LINE_H).max(0.0);
        self.scroll_y = self.scroll_y.clamp(0.0, max);
        self.scroll_x = self.scroll_x.max(0.0);
    }

    /// Scroll by `dy` pixels.
    pub fn scroll_by(&mut self, dy: f64) {
        self.scroll_y += dy;
        self.clamp_scroll();
    }

    /// Current vertical scroll in pixels.
    pub fn scroll_y(&self) -> f64 {
        self.scroll_y
    }

    fn reveal_line(&mut self, l: usize) {
        let h = self.text_area().h;
        let top = l as f64 * LINE_H;
        if top < self.scroll_y {
            self.scroll_y = top;
        } else if top + LINE_H > self.scroll_y + h {
            self.scroll_y = top + LINE_H - h;
        }
        self.clamp_scroll();
    }

    fn reveal(&mut self) {
        self.reveal_line(self.cursor.0);
        let w = self.text_area().w - 12.0;
        let x = self.x_of(self.cursor.0, self.cursor.1);
        if x < self.scroll_x {
            self.scroll_x = (x - 20.0).max(0.0);
        } else if x > self.scroll_x + w {
            self.scroll_x = x - w + 20.0;
        }
    }

    fn x_of(&self, line: usize, col: usize) -> f64 {
        let s: String = self.lines[line].chars().take(col).collect();
        text_width(&s, SIZE, false)
    }

    fn col_at(&self, line: usize, x: f64) -> usize {
        let chars: Vec<char> = self.lines[line].chars().collect();
        let mut prev = 0.0;
        for i in 1..=chars.len() {
            let w = text_width(&chars[..i].iter().collect::<String>(), SIZE, false);
            if x < (prev + w) / 2.0 {
                return i - 1;
            }
            prev = w;
        }
        chars.len()
    }

    // ---------------------------------------------------------------- selection helpers

    fn selection(&self) -> Option<(Pos, Pos)> {
        let a = self.anchor?;
        if a == self.cursor {
            return None;
        }
        Some(if a < self.cursor { (a, self.cursor) } else { (self.cursor, a) })
    }

    /// The selected text, if any.
    pub fn selected_text(&self) -> Option<String> {
        let (a, b) = self.selection()?;
        let mut out = String::new();
        for l in a.0..=b.0 {
            let chars: Vec<char> = self.lines[l].chars().collect();
            let from = if l == a.0 { a.1 } else { 0 };
            let to = if l == b.0 { b.1 } else { chars.len() };
            out.extend(&chars[from..to]);
            if l < b.0 {
                out.push('\n');
            }
        }
        Some(out)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot { lines: self.lines.clone(), cursor: self.cursor }
    }

    fn checkpoint(&mut self, typing: bool) {
        if !(typing && self.typing) {
            self.undo.push(self.snapshot());
            if self.undo.len() > MAX_UNDO {
                self.undo.remove(0);
            }
        }
        self.redo.clear();
        self.typing = typing;
    }

    fn delete_selection(&mut self) -> bool {
        let Some((a, b)) = self.selection() else { return false };
        let head: String = self.lines[a.0].chars().take(a.1).collect();
        let tail: String = self.lines[b.0].chars().skip(b.1).collect();
        self.lines.splice(a.0..=b.0, [head + &tail]);
        self.cursor = a;
        self.anchor = None;
        true
    }

    /// Insert `text` at the cursor, replacing the selection. Newlines start new lines.
    pub fn insert_text(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        if text.is_empty() {
            return;
        }
        self.checkpoint(text.chars().count() == 1 && text != "\n");
        self.delete_selection();
        let (l, c) = self.cursor;
        let head: String = self.lines[l].chars().take(c).collect();
        let tail: String = self.lines[l].chars().skip(c).collect();
        let parts: Vec<&str> = text.split('\n').collect();
        if parts.len() == 1 {
            self.lines[l] = format!("{head}{}{tail}", parts[0]);
            self.cursor = (l, c + parts[0].chars().count());
        } else {
            let mut new = vec![format!("{head}{}", parts[0])];
            new.extend(parts[1..parts.len() - 1].iter().map(|s| (*s).to_string()));
            let last = parts[parts.len() - 1];
            new.push(format!("{last}{tail}"));
            let end = (l + parts.len() - 1, last.chars().count());
            self.lines.splice(l..=l, new);
            self.cursor = end;
        }
        self.reveal();
    }

    fn newline(&mut self) {
        let indent: String = self.lines[self.cursor.0].chars().take_while(|c| *c == ' ').collect();
        let before: String = self.lines[self.cursor.0].chars().take(self.cursor.1).collect();
        let indent = if before.trim().is_empty() { before.clone() } else { indent };
        self.insert_text(&format!("\n{indent}"));
    }

    /// Copy the selection (or nothing).
    pub fn copy_text(&self) -> Option<String> {
        self.selected_text()
    }

    /// Cut the selection.
    pub fn cut_text(&mut self) -> Option<String> {
        let t = self.selected_text()?;
        self.checkpoint(false);
        self.delete_selection();
        self.reveal();
        Some(t)
    }

    /// Select everything.
    pub fn select_all(&mut self) {
        self.anchor = Some((0, 0));
        let l = self.lines.len() - 1;
        self.cursor = (l, self.lines[l].chars().count());
        self.typing = false;
    }

    /// Undo the last edit.
    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(s) => {
                self.redo.push(self.snapshot());
                self.lines = s.lines;
                self.cursor = s.cursor;
                self.anchor = None;
                self.typing = false;
                self.reveal();
                true
            }
            None => false,
        }
    }

    /// Redo the last undone edit.
    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(s) => {
                self.undo.push(self.snapshot());
                self.lines = s.lines;
                self.cursor = s.cursor;
                self.anchor = None;
                self.typing = false;
                self.reveal();
                true
            }
            None => false,
        }
    }

    // ---------------------------------------------------------------- input

    fn move_to(&mut self, p: Pos, extend: bool) {
        if extend {
            if self.anchor.is_none() {
                self.anchor = Some(self.cursor);
            }
        } else {
            self.anchor = None;
        }
        self.cursor = p;
        self.typing = false;
        self.reveal();
    }

    fn line_len(&self, l: usize) -> usize {
        self.lines[l].chars().count()
    }

    /// Key press. Ctrl+C/X/V are left to the workspace (they need the clipboard).
    pub fn key_down(&mut self, key: Key, mods: Mods) -> EditorEvent {
        let (l, c) = self.cursor;
        match key {
            Key::F5 => return if mods.shift { EditorEvent::Stop } else { EditorEvent::Run },
            Key::F8 => return EditorEvent::Step,
            Key::F9 => {
                self.toggle_breakpoint(l + 1);
                return EditorEvent::Breakpoints;
            }
            Key::Char(ch) if mods.ctrl => {
                match ch.to_ascii_lowercase() {
                    'a' => self.select_all(),
                    'z' if mods.shift => {
                        let c = self.redo();
                        return self.edited(c);
                    }
                    'z' => {
                        let c = self.undo();
                        return self.edited(c);
                    }
                    'y' => {
                        let c = self.redo();
                        return self.edited(c);
                    }
                    _ => {}
                }
                return EditorEvent::None;
            }
            Key::Char(ch) if !ch.is_control() && !mods.alt => {
                self.insert_text(&ch.to_string());
                return EditorEvent::Edited;
            }
            Key::Enter => {
                self.newline();
                return EditorEvent::Edited;
            }
            Key::Tab => {
                if self.selection().is_some_and(|(a, b)| a.0 != b.0) {
                    return self.indent(mods.shift);
                }
                if mods.shift {
                    return self.indent(true);
                }
                self.insert_text(TAB);
                return EditorEvent::Edited;
            }
            Key::Backspace => {
                self.checkpoint(false);
                if !self.delete_selection() {
                    if c > 0 {
                        let s: String =
                            self.lines[l].chars().take(c - 1).chain(self.lines[l].chars().skip(c)).collect();
                        self.lines[l] = s;
                        self.cursor = (l, c - 1);
                    } else if l > 0 {
                        let cur = self.lines.remove(l);
                        let at = self.line_len(l - 1);
                        self.lines[l - 1].push_str(&cur);
                        self.cursor = (l - 1, at);
                    } else {
                        self.undo.pop();
                        return EditorEvent::None;
                    }
                }
                self.reveal();
                return EditorEvent::Edited;
            }
            Key::Delete => {
                self.checkpoint(false);
                if !self.delete_selection() {
                    if c < self.line_len(l) {
                        let s: String =
                            self.lines[l].chars().take(c).chain(self.lines[l].chars().skip(c + 1)).collect();
                        self.lines[l] = s;
                    } else if l + 1 < self.lines.len() {
                        let next = self.lines.remove(l + 1);
                        self.lines[l].push_str(&next);
                    } else {
                        self.undo.pop();
                        return EditorEvent::None;
                    }
                }
                return EditorEvent::Edited;
            }
            Key::Left => {
                if let (Some((a, _)), false) = (self.selection(), mods.shift) {
                    self.move_to(a, false);
                } else if c > 0 {
                    self.move_to((l, c - 1), mods.shift);
                } else if l > 0 {
                    self.move_to((l - 1, self.line_len(l - 1)), mods.shift);
                }
            }
            Key::Right => {
                if let (Some((_, b)), false) = (self.selection(), mods.shift) {
                    self.move_to(b, false);
                } else if c < self.line_len(l) {
                    self.move_to((l, c + 1), mods.shift);
                } else if l + 1 < self.lines.len() {
                    self.move_to((l + 1, 0), mods.shift);
                }
            }
            Key::Up => {
                if l > 0 {
                    let x = self.x_of(l, c);
                    let nc = self.col_at(l - 1, x);
                    self.move_to((l - 1, nc), mods.shift);
                } else {
                    self.move_to((0, 0), mods.shift);
                }
            }
            Key::Down => {
                if l + 1 < self.lines.len() {
                    let x = self.x_of(l, c);
                    let nc = self.col_at(l + 1, x);
                    self.move_to((l + 1, nc), mods.shift);
                } else {
                    self.move_to((l, self.line_len(l)), mods.shift);
                }
            }
            Key::Home => {
                let first = self.lines[l].chars().take_while(|ch| *ch == ' ').count();
                let target = if c == first || mods.ctrl { 0 } else { first };
                self.move_to(if mods.ctrl { (0, 0) } else { (l, target) }, mods.shift);
            }
            Key::End => {
                let p = if mods.ctrl {
                    (self.lines.len() - 1, self.line_len(self.lines.len() - 1))
                } else {
                    (l, self.line_len(l))
                };
                self.move_to(p, mods.shift);
            }
            Key::PageUp | Key::PageDown => {
                let page = ((self.text_area().h / LINE_H).floor() as usize).max(2) - 1;
                let nl = if key == Key::PageUp { l.saturating_sub(page) } else { (l + page).min(self.lines.len() - 1) };
                let x = self.x_of(l, c);
                let nc = self.col_at(nl, x);
                self.move_to((nl, nc), mods.shift);
            }
            _ => {}
        }
        EditorEvent::None
    }

    fn edited(&self, changed: bool) -> EditorEvent {
        if changed {
            EditorEvent::Edited
        } else {
            EditorEvent::None
        }
    }

    /// Indent or unindent the selected lines (or the cursor line when Shift+Tab).
    fn indent(&mut self, out: bool) -> EditorEvent {
        let (a, b) = self.selection().unwrap_or((self.cursor, self.cursor));
        self.checkpoint(false);
        let mut changed = false;
        for l in a.0..=b.0 {
            if out {
                let n = self.lines[l].chars().take(TAB.len()).take_while(|c| *c == ' ').count();
                if n > 0 {
                    self.lines[l] = self.lines[l].chars().skip(n).collect();
                    changed = true;
                }
            } else {
                self.lines[l].insert_str(0, TAB);
                changed = true;
            }
        }
        if !changed {
            self.undo.pop();
            return EditorEvent::None;
        }
        self.cursor.1 = self.cursor.1.min(self.line_len(self.cursor.0));
        if let Some(an) = self.anchor.as_mut() {
            an.1 = an.1.min(self.lines[an.0].chars().count());
        }
        EditorEvent::Edited
    }

    fn button_at(&self, x: f64, y: f64) -> Option<Button> {
        self.buttons().iter().find(|(_, r)| r.contains(Point::new(x, y))).map(|(b, _)| *b)
    }

    /// Mouse press at pane coordinates.
    pub fn pointer_down(&mut self, x: f64, y: f64, clicks: u32, shift: bool) -> EditorEvent {
        if y < TOOLBAR {
            return match self.button_at(x, y) {
                Some(b) if self.button_enabled(b) => match b {
                    Button::Run => EditorEvent::Run,
                    Button::Step => EditorEvent::Step,
                    Button::Stop => EditorEvent::Stop,
                },
                _ => EditorEvent::None,
            };
        }
        let area = self.text_area();
        if y >= area.bottom() {
            return EditorEvent::None;
        }
        let line = (((y - TOOLBAR + self.scroll_y) / LINE_H).floor().max(0.0) as usize).min(self.lines.len() - 1);
        if x < GUTTER {
            if x < GUTTER - 30.0 || (y - TOOLBAR + self.scroll_y) / LINE_H < self.lines.len() as f64 {
                self.toggle_breakpoint(line + 1);
                return EditorEvent::Breakpoints;
            }
            return EditorEvent::None;
        }
        let col = self.col_at(line, x - GUTTER - 6.0 + self.scroll_x);
        match clicks {
            0 | 1 => {
                self.move_to((line, col), shift);
                if !shift {
                    self.anchor = Some((line, col));
                }
                self.dragging = true;
            }
            2 => {
                let chars: Vec<char> = self.lines[line].chars().collect();
                let word = |ch: char| ch.is_alphanumeric() || ch == '_';
                let mut a = col.min(chars.len());
                let mut b = a;
                while a > 0 && word(chars[a - 1]) {
                    a -= 1;
                }
                while b < chars.len() && word(chars[b]) {
                    b += 1;
                }
                self.anchor = Some((line, a));
                self.cursor = (line, b);
            }
            _ => {
                self.anchor = Some((line, 0));
                self.cursor = (line, self.line_len(line));
            }
        }
        self.typing = false;
        EditorEvent::None
    }

    /// Mouse moved with the button held: extend the selection.
    pub fn pointer_move(&mut self, x: f64, y: f64) {
        if !self.dragging {
            return;
        }
        let line = (((y - TOOLBAR + self.scroll_y) / LINE_H).floor().max(0.0) as usize).min(self.lines.len() - 1);
        let col = self.col_at(line, x - GUTTER - 6.0 + self.scroll_x);
        self.cursor = (line, col);
        self.reveal();
    }

    /// Mouse released.
    pub fn pointer_up(&mut self) {
        self.dragging = false;
        if self.anchor == Some(self.cursor) {
            self.anchor = None;
        }
    }

    // ---------------------------------------------------------------- drawing

    fn colors(&self) -> [(Class, Color); 9] {
        let dark = Chrome::of(&self.theme).dark;
        let pick = |l: u32, d: u32| Color::hex(if dark { d } else { l });
        let ink = self.theme.ink;
        [
            (Class::Plain, ink),
            (Class::Comment, pick(0x5C6B5C, 0x8FA08F)),
            (Class::Keyword, pick(0x7A2FD0, 0xD2A8FF)),
            (Class::Command, pick(0x0057A0, 0x79C0FF)),
            (Class::Number, pick(0xB04A00, 0xFFA657)),
            (Class::Text, pick(0x00704F, 0x7EE787)),
            (Class::Column, pick(0x00704F, 0x7EE787)),
            (Class::Variable, pick(0x9B1F7A, 0xFF9BD2)),
            (Class::Call, pick(0x9A5B00, 0xE3B341)),
        ]
    }

    fn color_of(&self, class: Class) -> Color {
        self.colors().iter().find(|(c, _)| *c == class).map_or(self.theme.ink, |(_, col)| *col)
    }

    /// Draw the panel (with the caret when `focused` was set).
    pub fn render(&self) -> Scene {
        self.render_focused(self.focused)
    }

    /// Draw the panel, with the caret when `focused`.
    pub fn render_focused(&self, focused: bool) -> Scene {
        let t = &self.theme;
        let chrome = Chrome::of(t);
        let mut s = Scene::new(self.width, self.height, t.plot_background);
        s.title = "Macro Editor".into();
        // Toolbar.
        s.rect(Rect::new(0.0, 0.0, self.width, TOOLBAR), Some(chrome.bar), Some(Stroke::new(t.grid, 1.0)));
        for (b, r) in self.buttons() {
            let on = self.button_enabled(b);
            let label = match b {
                Button::Run => {
                    if self.state == RunState::Paused {
                        "Continue"
                    } else {
                        "Run"
                    }
                }
                Button::Step => "Step",
                Button::Stop => "Stop",
            };
            s.rect(r, Some(chrome.button), Some(Stroke::new(t.grid, 1.0)));
            let ink = if on { t.ink } else { chrome.muted.with_alpha(0.5) };
            s.text(Point::new(r.x + 8.0, r.y + 14.0), label.to_string(), TextStyle::new(SIZE, ink));
        }
        let left = self.buttons()[2].1.right() + 14.0;
        let (ln, col) = self.cursor();
        let info = if self.message.is_empty() { format!("Line {ln}, column {col}") } else { self.message.clone() };
        s.clip_push(Rect::new(left, 0.0, (self.width - left).max(0.0), TOOLBAR));
        s.text(Point::new(left, 19.0), info, TextStyle::new(SIZE, chrome.muted));
        s.clip_pop();
        // Text area.
        let area = self.text_area();
        s.rect(Rect::new(0.0, TOOLBAR, GUTTER, area.h), Some(chrome.bar), None);
        s.clip_push(Rect::new(0.0, TOOLBAR, self.width, area.h));
        let first = (self.scroll_y / LINE_H).floor() as usize;
        let count = (area.h / LINE_H).ceil() as usize + 1;
        let sel = self.selection();
        for l in first..(first + count).min(self.lines.len()) {
            let y = TOOLBAR + l as f64 * LINE_H - self.scroll_y;
            let row = Rect::new(GUTTER, y, area.w, LINE_H);
            if self.exec_line == Some(l + 1) {
                s.rect(row, Some(chrome.accent.with_alpha(0.22)), None);
            }
            if self.error.as_ref().is_some_and(|(e, _)| *e == l + 1) {
                s.rect(row, Some(Color::hex(0xD62728).with_alpha(0.22)), None);
            }
            if let Some((a, b)) = sel {
                if l >= a.0 && l <= b.0 {
                    let from = if l == a.0 { self.x_of(l, a.1) } else { 0.0 };
                    let to = if l == b.0 { self.x_of(l, b.1) } else { self.x_of(l, self.line_len(l)) + 6.0 };
                    s.rect(
                        Rect::new(GUTTER + 6.0 + from - self.scroll_x, y, (to - from).max(2.0), LINE_H),
                        Some(chrome.selection),
                        None,
                    );
                }
            }
            // Text, one coloured run per span.
            let text = &self.lines[l];
            s.clip_push(Rect::new(GUTTER, y, area.w, LINE_H));
            for sp in highlight_line(text) {
                let before = text_width(&text[..sp.start], SIZE, false);
                s.text(
                    Point::new(GUTTER + 6.0 - self.scroll_x + before, y + 13.0),
                    text[sp.start..sp.end].to_string(),
                    TextStyle::new(SIZE, self.color_of(sp.class)),
                );
            }
            s.clip_pop();
            // Gutter: number and breakpoint.
            if self.breakpoints.contains(&(l + 1)) {
                s.marker(
                    Point::new(10.0, y + LINE_H / 2.0),
                    openstats_charts::core::Marker::Circle,
                    9.0,
                    Color::hex(0xD62728),
                    None,
                );
            }
            if self.exec_line == Some(l + 1) {
                s.marker(
                    Point::new(22.0, y + LINE_H / 2.0),
                    openstats_charts::core::Marker::TriangleUp,
                    8.0,
                    chrome.accent,
                    None,
                );
            }
            let num = (l + 1).to_string();
            let w = text_width(&num, SIZE, false);
            s.text(Point::new(GUTTER - 8.0 - w, y + 13.0), num, TextStyle::new(SIZE, chrome.muted));
        }
        s.line(Point::new(GUTTER, TOOLBAR), Point::new(GUTTER, TOOLBAR + area.h), Stroke::new(t.grid, 1.0));
        if focused {
            let (l, c) = self.cursor;
            let y = TOOLBAR + l as f64 * LINE_H - self.scroll_y;
            let x = GUTTER + 6.0 + self.x_of(l, c) - self.scroll_x;
            if y >= TOOLBAR - 1.0 && y < TOOLBAR + area.h {
                s.line(Point::new(x, y + 2.0), Point::new(x, y + LINE_H - 2.0), Stroke::new(t.ink, 1.5));
            }
        }
        s.clip_pop();
        // Error strip.
        if let Some((line, msg)) = &self.error {
            let r = Rect::new(0.0, self.height - ERROR_STRIP, self.width, ERROR_STRIP);
            s.rect(r, Some(Color::hex(0xD62728).with_alpha(0.16)), Some(Stroke::new(Color::hex(0xD62728), 1.0)));
            let head = if *line > 0 { format!("Error on line {line}") } else { "Error".to_string() };
            s.clip_push(r);
            s.text(Point::new(8.0, r.y + 14.0), head, TextStyle::new(SIZE, t.ink).bold());
            s.text(
                Point::new(8.0, r.y + 29.0),
                msg.lines().next().unwrap_or("").to_string(),
                TextStyle::new(SIZE, t.ink),
            );
            s.clip_pop();
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(e: &mut MacroEditor, s: &str) {
        for ch in s.chars() {
            if ch == '\n' {
                e.key_down(Key::Enter, Mods::default());
            } else {
                e.key_down(Key::Char(ch), Mods::default());
            }
        }
    }

    #[test]
    fn typing_enter_and_backspace() {
        let mut e = MacroEditor::new(600.0, 300.0);
        typed(&mut e, "LET K1 = 1\nLET K2 = 2");
        assert_eq!(e.text(), "LET K1 = 1\nLET K2 = 2");
        assert_eq!(e.cursor(), (2, 11));
        for _ in 0..10 {
            e.key_down(Key::Backspace, Mods::default());
        }
        assert_eq!(e.cursor(), (2, 1));
        e.key_down(Key::Backspace, Mods::default());
        assert_eq!((e.text().as_str(), e.cursor()), ("LET K1 = 1", (1, 11)));
    }

    #[test]
    fn enter_keeps_indentation() {
        let mut e = MacroEditor::new(600.0, 300.0);
        typed(&mut e, "DO K1 = 1:3\n  LET K2 = 1\n");
        assert_eq!(e.line(3), Some("  "));
    }

    #[test]
    fn selection_copy_cut_paste() {
        let mut e = MacroEditor::new(600.0, 300.0);
        e.set_text("one\ntwo\nthree");
        e.key_down(Key::Char('a'), Mods::CTRL);
        assert_eq!(e.copy_text().as_deref(), Some("one\ntwo\nthree"));
        e.set_cursor(1, 2);
        e.key_down(Key::Down, Mods { shift: true, ..Mods::default() });
        e.key_down(Key::Right, Mods { shift: true, ..Mods::default() });
        assert_eq!(e.selected_text().as_deref(), Some("ne\ntw"));
        assert_eq!(e.cut_text().as_deref(), Some("ne\ntw"));
        assert_eq!(e.text(), "oo\nthree");
        e.insert_text("X\nY");
        assert_eq!(e.text(), "oX\nYo\nthree");
        assert_eq!(e.line_count(), 3);
    }

    #[test]
    fn undo_and_redo_restore_text() {
        let mut e = MacroEditor::new(600.0, 300.0);
        typed(&mut e, "abc");
        e.key_down(Key::Enter, Mods::default());
        typed(&mut e, "def");
        assert!(e.undo());
        assert_eq!(e.text(), "abc\n");
        assert!(e.undo());
        assert_eq!(e.text(), "abc");
        assert!(e.undo());
        assert_eq!(e.text(), "");
        assert!(!e.undo());
        assert!(e.redo());
        assert_eq!(e.text(), "abc");
    }

    #[test]
    fn tab_indents_and_shift_tab_unindents_selected_lines() {
        let mut e = MacroEditor::new(600.0, 300.0);
        e.set_text("a\nb\nc");
        e.key_down(Key::Char('a'), Mods::CTRL);
        assert_eq!(e.key_down(Key::Tab, Mods::default()), EditorEvent::Edited);
        assert_eq!(e.text(), "  a\n  b\n  c");
        e.key_down(Key::Tab, Mods { shift: true, ..Mods::default() });
        assert_eq!(e.text(), "a\nb\nc");
    }

    #[test]
    fn breakpoints_toggle_by_key_and_by_gutter_click() {
        let mut e = MacroEditor::new(600.0, 300.0);
        e.set_text("a\nb\nc");
        e.set_cursor(2, 1);
        assert_eq!(e.key_down(Key::F9, Mods::default()), EditorEvent::Breakpoints);
        assert!(e.breakpoints().contains(&2));
        // Click the gutter beside line 3 (y = toolbar + 2.5 lines).
        assert_eq!(e.pointer_down(10.0, TOOLBAR + 2.5 * LINE_H, 1, false), EditorEvent::Breakpoints);
        assert!(e.breakpoints().contains(&3));
        e.key_down(Key::F9, Mods::default());
        assert!(!e.breakpoints().contains(&2));
    }

    #[test]
    fn toolbar_buttons_and_function_keys_emit_run_events() {
        let mut e = MacroEditor::new(600.0, 300.0);
        assert_eq!(e.key_down(Key::F5, Mods::default()), EditorEvent::Run);
        assert_eq!(e.key_down(Key::F8, Mods::default()), EditorEvent::Step);
        assert_eq!(e.key_down(Key::F5, Mods { shift: true, ..Mods::default() }), EditorEvent::Stop);
        let [(_, run), (_, step), (_, stop)] = e.buttons();
        assert_eq!(e.pointer_down(run.x + 3.0, run.y + 3.0, 1, false), EditorEvent::Run);
        assert_eq!(e.pointer_down(step.x + 3.0, step.y + 3.0, 1, false), EditorEvent::Step);
        // Stop is disabled until a run is paused.
        assert_eq!(e.pointer_down(stop.x + 3.0, stop.y + 3.0, 1, false), EditorEvent::None);
        e.set_run_state(RunState::Paused, "Paused at line 2");
        assert_eq!(e.pointer_down(stop.x + 3.0, stop.y + 3.0, 1, false), EditorEvent::Stop);
    }

    #[test]
    fn mouse_places_cursor_selects_words_and_drags() {
        let mut e = MacroEditor::new(600.0, 300.0);
        e.set_text("LET K1 = 5\nsecond line");
        e.pointer_down(GUTTER + 6.0 + 1.0, TOOLBAR + 3.0, 1, false);
        assert_eq!(e.cursor(), (1, 1));
        e.pointer_down(GUTTER + 6.0 + 3.0, TOOLBAR + 3.0, 2, false);
        assert_eq!(e.selected_text().as_deref(), Some("LET"));
        e.pointer_down(GUTTER + 6.0, TOOLBAR + 3.0, 1, false);
        e.pointer_move(GUTTER + 200.0, TOOLBAR + LINE_H + 3.0);
        e.pointer_up();
        let sel = e.selected_text().unwrap();
        assert!(sel.starts_with("LET K1 = 5\n"), "{sel:?}");
    }

    #[test]
    fn exec_line_and_error_scroll_into_view() {
        let mut e = MacroEditor::new(600.0, 120.0);
        e.set_text(&(1..=60).map(|i| format!("LET K1 = {i}")).collect::<Vec<_>>().join("\n"));
        assert_eq!(e.scroll_y(), 0.0);
        e.set_exec_line(Some(50));
        assert!(e.scroll_y() > 0.0);
        assert_eq!(e.exec_line(), Some(50));
        e.set_exec_line(None);
        e.set_error(Some((3, "boom".into())));
        assert_eq!(e.error().map(|x| x.0), Some(3));
        assert!(e.scroll_y() <= 2.0 * LINE_H);
    }

    #[test]
    fn render_draws_toolbar_numbers_and_error_strip() {
        let mut e = MacroEditor::new(500.0, 200.0);
        e.set_text("# a comment\nLET K1 = 1");
        e.set_error(Some((2, "something failed".into())));
        e.focused = true;
        let scene = e.render();
        let texts = format!("{:?}", scene);
        for needle in ["Run", "Step", "Stop", "# a comment", "Error on line 2", "something failed"] {
            assert!(texts.contains(needle), "missing {needle}");
        }
    }

    #[test]
    fn all_themes_render() {
        for k in [crate::style::ThemeKind::Light, crate::style::ThemeKind::Dark, crate::style::ThemeKind::HighContrast]
        {
            let mut e = MacroEditor::new(400.0, 150.0);
            e.theme = k.theme();
            e.set_text("DO K1 = 1:3\n  LET C1(K1) = K1 * 2 # x\nENDDO");
            let _ = e.render();
        }
    }
}
