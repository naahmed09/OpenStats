//! Canvas rendering: only the visible window is laid out, so the cost of a frame depends on
//! the window size and never on the number of rows.

use super::controller::{EditTarget, GridController};
use super::format::{cell_text_with, fit_number, fit_text, header_parts, Align};
use super::selection::CellRef;
use openstats_charts::core::font::text_width;
use openstats_charts::core::{Anchor, Color, Point, Rect, Scene, Stroke, TextStyle};

/// Grid colours derived from the chart theme.
#[derive(Debug, Clone, Copy)]
struct Colors {
    cell: Color,
    ink: Color,
    dim: Color,
    line: Color,
    header: Color,
    header_sel: Color,
    sel: Color,
    accent: Color,
    thumb: Color,
}

impl GridController {
    fn colors(&self) -> Colors {
        let t = &self.theme;
        let ch = crate::style::Chrome::of(t);
        Colors {
            cell: t.background,
            ink: t.ink,
            dim: t.ink.with_alpha(0.45),
            line: t.grid,
            header: ch.bar,
            header_sel: ch.header_selected,
            sel: ch.selection,
            accent: ch.accent,
            thumb: t.ink.with_alpha(0.35),
        }
    }

    /// Vertical and horizontal scrollbar thumbs (view coordinates); `None` when the content
    /// fits.
    pub fn scrollbar_thumbs(&self) -> (Option<Rect>, Option<Rect>) {
        let l = self.layout();
        let (cw, ch) = self.content_size();
        let (mx, my) = self.max_scroll();
        let sb = self.metrics.scrollbar;
        let min: f64 = 20.0;
        let v = (my > 0.0).then(|| {
            let len = (l.body.h * l.body.h / ch).clamp(min.min(l.body.h), l.body.h);
            let y = l.body.y + (l.body.h - len) * (self.view.scroll_y / my);
            Rect::new(l.body.right(), y, sb, len)
        });
        let h = (mx > 0.0).then(|| {
            let len = (l.body.w * l.body.w / cw).clamp(min.min(l.body.w), l.body.w);
            let x = l.body.x + (l.body.w - len) * (self.view.scroll_x / mx);
            Rect::new(x, l.body.bottom(), len, sb)
        });
        (v, h)
    }

    /// Draw the visible part of the grid.
    pub fn render(&self) -> Scene {
        let c = self.colors();
        let l = self.layout();
        let m = &self.metrics;
        let ws = self.ws();
        let mut s = Scene::new(self.view.width, self.view.height, c.cell);
        s.title = format!("Worksheet {}", ws.name);
        let (r0, r1) = self.visible_rows();
        let (c0, c1) = self.visible_cols();
        let hair = Stroke::new(c.line, 1.0);
        let size = m.font_size;
        let base = |rect: Rect| rect.y + rect.h / 2.0 + size * 0.35;

        // Body.
        s.clip_push(l.body);
        for row in r0..r1 {
            let y = l.header + row as f64 * m.row_height - self.view.scroll_y;
            s.line(Point::new(l.body.x, y + m.row_height), Point::new(l.body.right(), y + m.row_height), hair);
        }
        for col in c0..c1 {
            let x = l.gutter + self.widths.x_of(col) - self.view.scroll_x;
            s.line(
                Point::new(x + self.widths.width(col), l.body.y),
                Point::new(x + self.widths.width(col), l.body.bottom()),
                hair,
            );
        }
        // Selection shading under the text.
        for rg in self.selection.ranges() {
            let a = rg.r0.max(r0).min(r1);
            let b = (rg.r1.saturating_add(1)).min(r1).max(a);
            let x0 = self.widths.x_of(rg.c0.min(self.display_cols()));
            let x1 = self.widths.x_of(rg.c1.saturating_add(1).min(self.display_cols()));
            if a < b && x1 > x0 {
                let top = l.header + a as f64 * m.row_height - self.view.scroll_y;
                s.rect(
                    Rect::new(l.gutter + x0 - self.view.scroll_x, top, x1 - x0, (b - a) as f64 * m.row_height),
                    Some(c.sel),
                    None,
                );
            }
        }
        // Rows brushed in a graph.
        for &row in self.highlight.range(r0..r1) {
            let top = l.header + row as f64 * m.row_height - self.view.scroll_y;
            s.rect(Rect::new(l.body.x, top, l.body.w, m.row_height), Some(c.accent.with_alpha(0.22)), None);
        }
        for col in c0..c1 {
            let Some(data) = ws.columns.get(col) else { continue };
            let w = self.widths.width(col);
            let x = l.gutter + self.widths.x_of(col) - self.view.scroll_x;
            let end = r1.min(data.len());
            for row in r0..end {
                let Some(ct) = cell_text_with(data, row, &self.locale) else { continue };
                let rect = Rect::new(x, l.header + row as f64 * m.row_height - self.view.scroll_y, w, m.row_height);
                let text = if ct.align == Align::Right && !ct.missing {
                    self.locale.localize_number(&fit_number(data, row, &ct.text, w - 8.0, size))
                } else {
                    fit_text(&ct.text, w - 8.0, size, false)
                };
                let color = if ct.missing { c.dim } else { c.ink };
                let (at, anchor) = match ct.align {
                    Align::Left => (Point::new(rect.x + 4.0, base(rect)), Anchor::Start),
                    Align::Right => (Point::new(rect.right() - 4.0, base(rect)), Anchor::End),
                };
                s.text(at, text, TextStyle::new(size, color).anchored(anchor));
            }
        }
        // Active cell and editor.
        let a = self.selection.active();
        let ar = self.cell_rect(a);
        if let Some(ed) = &self.editor {
            if let EditTarget::Cell(ec) = &ed.target {
                let er = self.cell_rect(*ec);
                let shown = ed.display_text();
                let w = er.w.max(text_width(&shown, size, false) + 12.0);
                let box_ = Rect::new(er.x, er.y, w, er.h);
                s.rect(box_, Some(c.cell), Some(Stroke::new(c.accent, 2.0)));
                let tw = |n: usize| text_width(&shown.chars().take(n).collect::<String>(), size, false);
                if let Some((a, b)) = ed.selection() {
                    let (x0, x1) = (box_.x + 4.0 + tw(a), box_.x + 4.0 + tw(b));
                    s.rect(Rect::new(x0, box_.y + 3.0, x1 - x0, box_.h - 6.0), Some(c.accent.with_alpha(0.3)), None);
                }
                s.text(Point::new(box_.x + 4.0, base(box_)), shown.clone(), TextStyle::new(size, c.ink));
                if let Some(comp) = &ed.composition {
                    let x0 = box_.x + 4.0 + tw(ed.cursor);
                    let x1 = box_.x + 4.0 + tw(ed.cursor + comp.text.chars().count());
                    s.line(
                        Point::new(x0, box_.bottom() - 3.0),
                        Point::new(x1, box_.bottom() - 3.0),
                        Stroke::new(c.ink, 1.5),
                    );
                }
                let cx = box_.x + 4.0 + tw(ed.display_cursor());
                s.line(Point::new(cx, box_.y + 3.0), Point::new(cx, box_.bottom() - 3.0), Stroke::new(c.ink, 1.0));
            }
        } else {
            s.rect(ar, None, Some(Stroke::new(c.accent, 2.0)));
        }
        s.clip_pop();

        // Frozen column header.
        s.clip_push(Rect::new(l.gutter, 0.0, l.body.w, l.header));
        for col in c0..c1 {
            let w = self.widths.width(col);
            let x = l.gutter + self.widths.x_of(col) - self.view.scroll_x;
            let rect = Rect::new(x, 0.0, w, l.header);
            let sel = self.selection.touches_column(col);
            s.rect(rect, Some(if sel { c.header_sel } else { c.header }), Some(hair));
            let (label, name) = header_parts(ws, col);
            let mid = Point::new(rect.x + w / 2.0, 0.0);
            s.text(
                Point::new(mid.x, 15.0),
                fit_text(&label, w - 6.0, size, true),
                TextStyle::new(size, c.ink).anchored(Anchor::Middle).bold(),
            );
            s.text(
                Point::new(mid.x, 32.0),
                fit_text(&name, w - 6.0, size, false),
                TextStyle::new(size, c.ink).anchored(Anchor::Middle),
            );
        }
        s.clip_pop();

        // Frozen row gutter.
        s.clip_push(Rect::new(0.0, l.header, l.gutter, l.body.h));
        for row in r0..r1 {
            let rect =
                Rect::new(0.0, l.header + row as f64 * m.row_height - self.view.scroll_y, l.gutter, m.row_height);
            let sel = self.selection.touches_row(row) || self.highlight.contains(&row);
            s.rect(rect, Some(if sel { c.header_sel } else { c.header }), Some(hair));
            s.text(
                Point::new(rect.right() - 6.0, base(rect)),
                (row + 1).to_string(),
                TextStyle::new(size, c.ink).anchored(Anchor::End),
            );
        }
        s.clip_pop();

        // Corner and scrollbars.
        s.rect(Rect::new(0.0, 0.0, l.gutter, l.header), Some(c.header), Some(hair));
        let (v, h) = self.scrollbar_thumbs();
        if let Some(v) = v {
            s.rect(v, Some(c.thumb), None);
        }
        if let Some(h) = h {
            s.rect(h, Some(c.thumb), None);
        }
        s
    }

    /// Whether `cell` is inside the visible window (used by hosts to decide on scrolling).
    pub fn is_visible(&self, cell: CellRef) -> bool {
        let (r0, r1) = self.visible_rows();
        let (c0, c1) = self.visible_cols();
        cell.row >= r0 && cell.row < r1 && cell.col >= c0 && cell.col < c1
    }
}
