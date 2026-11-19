//! Keyboard, pointer and wheel input.

use super::controller::{Drag, GridController};
use super::selection::CellRef;

/// A key press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// Arrow up.
    Up,
    /// Arrow down.
    Down,
    /// Arrow left.
    Left,
    /// Arrow right.
    Right,
    /// Home.
    Home,
    /// End.
    End,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Enter / Return.
    Enter,
    /// Tab.
    Tab,
    /// Escape.
    Escape,
    /// Backspace.
    Backspace,
    /// Delete.
    Delete,
    /// F2 (edit the active cell).
    F2,
    /// F6 (move to the next pane).
    F6,
    /// F10 (open the menu bar).
    F10,
    /// F1 (help).
    F1,
    /// F5 (run the macro).
    F5,
    /// F8 (step the macro).
    F8,
    /// F9 (toggle a macro breakpoint).
    F9,
    /// A printable character.
    Char(char),
}

/// Modifier keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mods {
    /// Shift.
    pub shift: bool,
    /// Ctrl (Cmd on macOS).
    pub ctrl: bool,
    /// Alt.
    pub alt: bool,
}

impl std::ops::BitOr for Mods {
    type Output = Mods;

    /// Combine modifier sets (`Mods::CTRL | Mods::SHIFT`).
    fn bitor(self, o: Mods) -> Mods {
        Mods { shift: self.shift || o.shift, ctrl: self.ctrl || o.ctrl, alt: self.alt || o.alt }
    }
}

impl Mods {
    /// No modifiers.
    pub const NONE: Mods = Mods { shift: false, ctrl: false, alt: false };
    /// Shift only.
    pub const SHIFT: Mods = Mods { shift: true, ctrl: false, alt: false };
    /// Ctrl only.
    pub const CTRL: Mods = Mods { shift: false, ctrl: true, alt: false };
}

/// What lies under a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hit {
    /// The top-left corner (select all).
    Corner,
    /// A column header.
    ColumnHeader(usize),
    /// The right edge of a column header (resize grip).
    ColumnEdge(usize),
    /// A row-number cell.
    RowHeader(usize),
    /// A body cell.
    Cell(CellRef),
    /// The vertical scrollbar track.
    VScroll,
    /// The horizontal scrollbar track.
    HScroll,
    /// Outside everything.
    None,
}

impl GridController {
    fn cell_at(&self, x: f64, y: f64) -> CellRef {
        let l = self.layout();
        let row = (((y - l.header + self.view.scroll_y) / self.metrics.row_height).floor().max(0.0)) as usize;
        let col = self.widths.col_at((x - l.gutter + self.view.scroll_x).max(0.0), self.display_cols());
        self.clamp_cell(CellRef::new(row, col))
    }

    /// What is at view position `(x, y)`.
    pub fn hit_test(&self, x: f64, y: f64) -> Hit {
        let l = self.layout();
        if x < 0.0 || y < 0.0 || x > self.view.width || y > self.view.height {
            return Hit::None;
        }
        if x >= l.body.right() && y >= l.body.y && y < l.body.bottom() {
            return Hit::VScroll;
        }
        if y >= l.body.bottom() && x >= l.body.x && x < l.body.right() {
            return Hit::HScroll;
        }
        if x >= l.body.right() || y >= l.body.bottom() {
            return Hit::None;
        }
        if y < l.header {
            if x < l.gutter {
                return Hit::Corner;
            }
            let c = self.cell_at(x, l.header).col;
            let left = l.gutter + self.widths.x_of(c) - self.view.scroll_x;
            let right = left + self.widths.width(c);
            let grip = self.metrics.resize_grip;
            return if c > 0 && (x - left) <= grip {
                Hit::ColumnEdge(c - 1)
            } else if (right - x).abs() <= grip {
                Hit::ColumnEdge(c)
            } else {
                Hit::ColumnHeader(c)
            };
        }
        if x < l.gutter {
            return Hit::RowHeader(self.cell_at(l.gutter, y).row);
        }
        Hit::Cell(self.cell_at(x, y))
    }

    /// Pointer pressed. `clicks` is the click count (2 = double click).
    pub fn pointer_down(&mut self, x: f64, y: f64, mods: super::input::Mods, clicks: u32) {
        let hit = self.hit_test(x, y);
        let additive = mods.ctrl;
        if !matches!(hit, Hit::Cell(_)) && self.editor.is_some() {
            let _ = self.commit_edit();
            if self.editor.is_some() {
                return;
            }
        }
        match hit {
            Hit::Corner => self.selection.select_all(),
            Hit::ColumnEdge(c) => {
                if clicks >= 2 {
                    self.autosize_column(c);
                } else {
                    self.drag = Some(Drag::Resize { col: c, start_x: x, start_w: self.widths.width(c) });
                }
            }
            Hit::ColumnHeader(c) => {
                if clicks >= 2 {
                    self.begin_rename(c);
                } else {
                    self.selection.select_column(c, mods.shift, additive);
                    self.drag = Some(Drag::Columns);
                }
            }
            Hit::RowHeader(r) => {
                self.selection.select_row(r, mods.shift, additive);
                self.drag = Some(Drag::Rows);
            }
            Hit::Cell(cell) => {
                if self.editor.as_ref().is_some_and(|e| e.target == super::controller::EditTarget::Cell(cell)) {
                    let size = self.metrics.font_size;
                    let er = self.cell_rect(cell);
                    if let Some(ed) = self.editor.as_mut() {
                        let i = ed.index_at(x - er.x - 4.0, size);
                        ed.place_cursor(i, mods.shift);
                        if clicks >= 2 {
                            ed.select_word();
                        }
                    }
                    self.drag = Some(Drag::Editor);
                    return;
                }
                if self.editor.is_some() {
                    let _ = self.commit_edit();
                    if self.editor.is_some() {
                        return;
                    }
                }
                if mods.shift {
                    self.selection.extend_to(cell);
                } else if additive {
                    self.selection.add_cell(cell);
                } else {
                    self.selection.select_cell(cell);
                }
                self.drag = Some(Drag::Select);
                if clicks >= 2 {
                    self.begin_edit(None, false);
                }
            }
            Hit::VScroll | Hit::HScroll => self.press_scrollbar(hit, x, y),
            Hit::None => {}
        }
    }

    fn press_scrollbar(&mut self, hit: Hit, x: f64, y: f64) {
        let (v, h) = self.scrollbar_thumbs();
        let l = self.layout();
        match hit {
            Hit::VScroll => match v {
                Some(t) if y >= t.y && y <= t.bottom() => self.drag = Some(Drag::VThumb { grab: y - t.y }),
                Some(_) => self.scroll_by(0.0, if y < l.body.y + (l.body.h / 2.0) { -l.body.h } else { l.body.h }),
                None => {}
            },
            _ => match h {
                Some(t) if x >= t.x && x <= t.right() => self.drag = Some(Drag::HThumb { grab: x - t.x }),
                Some(t) => self.scroll_by(if x < t.x { -l.body.w } else { l.body.w }, 0.0),
                None => {}
            },
        }
    }

    /// Pointer moved (with the button down this continues a drag).
    pub fn pointer_move(&mut self, x: f64, y: f64) {
        let Some(d) = self.drag else { return };
        let l = self.layout();
        match d {
            Drag::Select => {
                let t = self.cell_at(x, y);
                self.go_to(t, true);
            }
            Drag::Editor => {
                let size = self.metrics.font_size;
                let start = self.editor.as_ref().and_then(|e| match &e.target {
                    super::controller::EditTarget::Cell(c) => Some(self.cell_rect(*c).x),
                    _ => None,
                });
                if let (Some(x0), Some(ed)) = (start, self.editor.as_mut()) {
                    let i = ed.index_at(x - x0 - 4.0, size);
                    ed.place_cursor(i, true);
                }
            }
            Drag::Columns => {
                let c = self.cell_at(x, l.header).col;
                self.selection.select_column(c, true, false);
                self.scroll_to_cell(CellRef::new(self.view_top_row(), c));
            }
            Drag::Rows => {
                let r = self.cell_at(l.gutter, y).row;
                self.selection.select_row(r, true, false);
                self.scroll_to_cell(CellRef::new(r, self.visible_cols().0));
            }
            Drag::Resize { col, start_x, start_w } => self.set_column_width(col, start_w + x - start_x),
            Drag::VThumb { grab } => {
                if let (Some(t), (_, my)) = (self.scrollbar_thumbs().0, self.max_scroll()) {
                    let span = (l.body.h - t.h).max(1.0);
                    self.scroll_to(self.view.scroll_x, ((y - grab - l.body.y) / span).clamp(0.0, 1.0) * my);
                }
            }
            Drag::HThumb { grab } => {
                if let (Some(t), (mx, _)) = (self.scrollbar_thumbs().1, self.max_scroll()) {
                    let span = (l.body.w - t.w).max(1.0);
                    self.scroll_to(((x - grab - l.body.x) / span).clamp(0.0, 1.0) * mx, self.view.scroll_y);
                }
            }
        }
    }

    fn view_top_row(&self) -> usize {
        self.visible_rows().0
    }

    /// Pointer released.
    pub fn pointer_up(&mut self) {
        self.drag = None;
    }

    /// Wheel / trackpad scroll by pixel deltas.
    pub fn wheel(&mut self, dx: f64, dy: f64) {
        self.scroll_by(dx, dy);
    }

    /// A key press; returns whether the grid handled it. Errors (a rejected edit, nothing to
    /// undo) are reported through [`GridController::status`].
    pub fn key_down(&mut self, key: Key, mods: Mods) -> bool {
        if self.editor.is_some() {
            return self.key_editing(key, mods);
        }
        let page = ((self.layout().body.h / self.metrics.row_height).floor() as isize - 1).max(1);
        if mods.ctrl {
            return self.key_ctrl(key, mods);
        }
        let sh = mods.shift;
        match key {
            Key::Up => self.move_active(-1, 0, sh),
            Key::Down | Key::Enter => {
                self.move_active(if key == Key::Enter && sh { -1 } else { 1 }, 0, sh && key != Key::Enter)
            }
            Key::Left => self.move_active(0, -1, sh),
            Key::Right => self.move_active(0, 1, sh),
            Key::Tab => self.move_active(0, if sh { -1 } else { 1 }, false),
            Key::PageUp => self.move_active(-page, 0, sh),
            Key::PageDown => self.move_active(page, 0, sh),
            Key::Home => {
                let a = self.active();
                self.go_to(CellRef::new(a.row, 0), sh);
            }
            Key::End => {
                let a = self.active();
                let last = self.data_cols().saturating_sub(1);
                self.go_to(CellRef::new(a.row, last), sh);
            }
            Key::F2 => self.begin_edit(None, false),
            Key::Delete | Key::Backspace => self.report(GridController::clear_selection),
            Key::Escape => {}
            // Pane and menu keys belong to the workspace, not the grid.
            Key::F1 | Key::F6 | Key::F10 | Key::F5 | Key::F8 | Key::F9 => return false,
            Key::Char(c) if !mods.alt && !c.is_control() => self.begin_edit(Some(&c.to_string()), false),
            Key::Char(_) => return false,
        }
        true
    }

    fn report(&mut self, f: impl FnOnce(&mut GridController) -> openstats_core::Result<()>) {
        match f(self) {
            Ok(()) => self.status.clear(),
            Err(e) => self.status = e.to_string(),
        }
    }

    fn key_ctrl(&mut self, key: Key, mods: Mods) -> bool {
        let sh = mods.shift;
        match key {
            Key::Up | Key::Down | Key::Left | Key::Right => {
                let (dr, dc) = match key {
                    Key::Up => (-1, 0),
                    Key::Down => (1, 0),
                    Key::Left => (0, -1),
                    _ => (0, 1),
                };
                let t = self.jump_target(self.selection.active(), dr, dc);
                self.go_to(t, sh);
            }
            Key::Home => self.go_to(CellRef::new(0, 0), sh),
            Key::End => {
                let t = CellRef::new(self.data_rows().saturating_sub(1), self.data_cols().saturating_sub(1));
                self.go_to(t, sh);
            }
            Key::Char(c) => match c.to_ascii_lowercase() {
                'a' => self.selection.select_all(),
                'c' => self.report(|g| g.copy().map(|_| ())),
                'x' => self.report(|g| g.cut().map(|_| ())),
                'v' => {
                    let text = self.clipboard.clone().unwrap_or_default();
                    self.report(|g| g.paste_text(&text).map(|_| ()));
                }
                'z' if sh => self.report(|g| g.redo().map(|_| ())),
                'z' => self.report(|g| g.undo().map(|_| ())),
                'y' => self.report(|g| g.redo().map(|_| ())),
                _ => return false,
            },
            _ => return false,
        }
        true
    }

    fn key_editing(&mut self, key: Key, mods: Mods) -> bool {
        match key {
            Key::Escape => self.cancel_edit(),
            Key::Enter | Key::Tab | Key::Up | Key::Down => {
                if self.commit_edit().is_ok() {
                    let (dr, dc) = match key {
                        Key::Enter | Key::Down => (if key == Key::Enter && mods.shift { -1 } else { 1 }, 0),
                        Key::Up => (-1, 0),
                        _ => (0, if mods.shift { -1 } else { 1 }),
                    };
                    self.move_active(dr, dc, false);
                }
            }
            Key::Left | Key::Right | Key::Home | Key::End | Key::Backspace | Key::Delete | Key::Char(_) => {
                let Some(ed) = self.editor.as_mut() else { return false };
                let sh = mods.shift;
                match key {
                    Key::Left if mods.ctrl => ed.move_word(-1, sh),
                    Key::Right if mods.ctrl => ed.move_word(1, sh),
                    Key::Left => ed.move_cursor(-1, sh),
                    Key::Right => ed.move_cursor(1, sh),
                    Key::Home => ed.cursor_to(false, sh),
                    Key::End => ed.cursor_to(true, sh),
                    Key::Backspace => ed.backspace(),
                    Key::Delete => ed.delete(),
                    Key::Char(c) if mods.ctrl && c.eq_ignore_ascii_case(&'a') => ed.select_all(),
                    Key::Char(c) if !mods.ctrl && !c.is_control() => ed.insert(&c.to_string()),
                    _ => return false,
                }
            }
            _ => return false,
        }
        true
    }
}
