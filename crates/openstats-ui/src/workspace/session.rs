//! The Session window: a scrollable, printable view of a [`SessionReport`].

use openstats_charts::core::font::text_width;
use openstats_charts::core::{Anchor, Color, Point, Rect, Scene, Stroke, TextStyle, Theme};
use openstats_charts::{render_spec, PlotOptions};
use openstats_stats::report::{Block, ColumnKind, SessionReport};

const SIZE: f64 = 12.0;
const LINE: f64 = 17.0;
const ROW: f64 = 19.0;
const MARGIN: f64 = 14.0;
const THUMB_W: f64 = 360.0;
const THUMB_H: f64 = 240.0;
const GAP: f64 = 10.0;

#[derive(Debug, Clone)]
enum Item {
    Title(String),
    Heading(String),
    Text(Vec<String>),
    Table {
        caption: String,
        header: Vec<String>,
        cols: Vec<Vec<String>>,
        right: Vec<bool>,
        widths: Vec<f64>,
        footnotes: Vec<String>,
    },
    Charts(Vec<(Scene, usize, usize)>),
    Warning(String),
}

#[derive(Debug, Clone)]
struct Laid {
    y: f64,
    h: f64,
    section: usize,
    item: Item,
}

/// What a click in the Session window landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionHit {
    /// Empty space.
    None,
    /// Text or table of section `.0` (it becomes the selected section).
    Section(usize),
    /// A chart thumbnail: section and block index.
    Chart(usize, usize),
}

/// Scrollable view of a report.
#[derive(Debug, Clone)]
pub struct SessionView {
    report: SessionReport,
    laid: Vec<Laid>,
    content_h: f64,
    /// Window size in pixels.
    pub width: f64,
    /// Window height in pixels.
    pub height: f64,
    scroll: f64,
    selected: Option<usize>,
    /// Colours used when drawing.
    pub theme: Theme,
}

fn wrap(text: &str, width: f64) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split(' ') {
            let cand = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
            if !line.is_empty() && text_width(&cand, SIZE, false) > width {
                out.push(std::mem::take(&mut line));
                line = word.to_string();
            } else {
                line = cand;
            }
        }
        out.push(line);
    }
    out
}

impl SessionView {
    /// An empty view of `width × height` pixels.
    pub fn new(width: f64, height: f64) -> SessionView {
        let mut v = SessionView {
            report: SessionReport::default(),
            laid: Vec::new(),
            content_h: 0.0,
            width,
            height,
            scroll: 0.0,
            selected: None,
            theme: Theme::light(),
        };
        v.relayout();
        v
    }

    /// Show `report` (scroll resets only if the content shrank below the scroll position).
    pub fn set_report(&mut self, report: &SessionReport) {
        self.report = report.clone();
        self.selected = None;
        self.relayout();
        self.scroll = self.scroll.clamp(0.0, self.max_scroll());
    }

    /// Resize the window.
    pub fn set_size(&mut self, w: f64, h: f64) {
        self.width = w;
        self.height = h;
        self.relayout();
        self.scroll = self.scroll.clamp(0.0, self.max_scroll());
    }

    /// The report shown.
    pub fn report(&self) -> &SessionReport {
        &self.report
    }

    /// Total content height.
    pub fn content_height(&self) -> f64 {
        self.content_h
    }

    fn max_scroll(&self) -> f64 {
        (self.content_h - self.height).max(0.0)
    }

    /// Current scroll offset.
    pub fn scroll(&self) -> f64 {
        self.scroll
    }

    /// Scroll by `dy` pixels.
    pub fn scroll_by(&mut self, dy: f64) {
        self.scroll = (self.scroll + dy).clamp(0.0, self.max_scroll());
    }

    /// Scroll to the end (where new output appears).
    pub fn scroll_to_end(&mut self) {
        self.scroll = self.max_scroll();
    }

    /// Scroll to the top.
    pub fn scroll_to_start(&mut self) {
        self.scroll = 0.0;
    }

    fn relayout(&mut self) {
        let avail = (self.width - 2.0 * MARGIN).max(80.0);
        let mut laid: Vec<Laid> = Vec::new();
        let mut y = MARGIN;
        let push = |laid: &mut Vec<Laid>, y: &mut f64, section: usize, h: f64, item: Item| {
            laid.push(Laid { y: *y, h, section, item });
            *y += h + GAP;
        };
        if !self.report.title.is_empty() {
            push(&mut laid, &mut y, usize::MAX, LINE + 4.0, Item::Title(self.report.title.clone()));
        }
        for (si, sec) in self.report.sections.iter().enumerate() {
            if !sec.heading.is_empty() {
                push(&mut laid, &mut y, si, LINE + 2.0, Item::Heading(sec.heading.clone()));
            }
            for (bi, b) in sec.blocks.iter().enumerate() {
                match b {
                    Block::Text(t) => {
                        let lines = wrap(t, avail);
                        let h = lines.len() as f64 * LINE;
                        push(&mut laid, &mut y, si, h, Item::Text(lines));
                    }
                    Block::Table(t) => {
                        let cols = t.formatted_columns();
                        let header: Vec<String> = t.columns.iter().map(|c| c.header.clone()).collect();
                        let right: Vec<bool> = t.columns.iter().map(|c| c.kind != ColumnKind::Text).collect();
                        let widths: Vec<f64> = header
                            .iter()
                            .zip(&cols)
                            .map(|(h, c)| {
                                c.iter().map(|s| text_width(s, SIZE, false)).fold(text_width(h, SIZE, true), f64::max)
                                    + 20.0
                            })
                            .collect();
                        let cap = if t.title.is_empty() { 0.0 } else { LINE };
                        let h = cap + ROW * (t.rows.len() as f64 + 1.0) + 4.0 + t.footnotes.len() as f64 * LINE;
                        push(
                            &mut laid,
                            &mut y,
                            si,
                            h,
                            Item::Table {
                                caption: t.title.clone(),
                                header,
                                cols,
                                right,
                                widths,
                                footnotes: t.footnotes.clone(),
                            },
                        );
                    }
                    Block::Chart(c) => {
                        let o = PlotOptions {
                            width: THUMB_W,
                            height: THUMB_H,
                            theme: self.theme.clone(),
                            ..Default::default()
                        };
                        let scene = render_spec(c, &o);
                        let fits = matches!(laid.last(), Some(Laid { item: Item::Charts(v), .. }) if (v.len() as f64 + 1.0) * (THUMB_W + GAP) - GAP <= avail && laid.last().is_some_and(|l| l.section == si));
                        if fits {
                            if let Some(Laid { item: Item::Charts(v), .. }) = laid.last_mut() {
                                v.push((scene, si, bi));
                            }
                        } else {
                            push(&mut laid, &mut y, si, THUMB_H, Item::Charts(vec![(scene, si, bi)]));
                        }
                    }
                }
            }
            for w in &sec.warnings {
                push(&mut laid, &mut y, si, LINE, Item::Warning(w.clone()));
            }
            y += GAP;
        }
        for w in &self.report.warnings {
            push(&mut laid, &mut y, usize::MAX, LINE, Item::Warning(w.clone()));
        }
        self.content_h = y + MARGIN;
        self.laid = laid;
    }

    fn draw(&self, s: &mut Scene, l: &Laid, x0: f64, y0: f64, highlight: bool) {
        let t = &self.theme;
        let ink = TextStyle::new(SIZE, t.ink);
        let y = y0 + l.y;
        if highlight {
            s.rect(
                Rect::new(x0, y - 3.0, self.width, l.h + 6.0),
                Some(crate::style::Chrome::of(t).selection.with_alpha(0.08)),
                None,
            );
        }
        let text_y = |i: usize| y + (i as f64 + 1.0) * LINE - 4.0;
        match &l.item {
            Item::Title(t) => {
                s.text(Point::new(x0 + MARGIN, y + LINE - 2.0), t.clone(), TextStyle::new(14.0, self.theme.ink).bold());
            }
            Item::Heading(h) => {
                s.text(
                    Point::new(x0 + MARGIN, y + LINE - 3.0),
                    h.clone(),
                    TextStyle::new(SIZE + 1.0, self.theme.ink).bold(),
                );
                s.line(
                    Point::new(x0 + MARGIN, y + LINE),
                    Point::new(x0 + self.width - MARGIN, y + LINE),
                    Stroke::new(self.theme.grid, 1.0),
                );
            }
            Item::Text(lines) => {
                for (i, line) in lines.iter().enumerate() {
                    s.text(Point::new(x0 + MARGIN, text_y(i)), line.clone(), ink);
                }
            }
            Item::Warning(w) => {
                s.text(
                    Point::new(x0 + MARGIN, y + LINE - 4.0),
                    format!("* WARNING * {w}"),
                    TextStyle::new(SIZE, self.theme.alert),
                );
            }
            Item::Table { caption, header, cols, right, widths, footnotes } => {
                let mut ty = y;
                if !caption.is_empty() {
                    s.text(Point::new(x0 + MARGIN, ty + LINE - 4.0), caption.clone(), ink.bold());
                    ty += LINE;
                }
                let total: f64 = widths.iter().sum();
                let mut cx = x0 + MARGIN;
                for (j, h) in header.iter().enumerate() {
                    let (px, anchor) =
                        if right[j] { (cx + widths[j] - 10.0, Anchor::End) } else { (cx + 10.0, Anchor::Start) };
                    s.text(Point::new(px, ty + ROW - 5.0), h.clone(), ink.bold().anchored(anchor));
                    for (i, cell) in cols[j].iter().enumerate() {
                        s.text(Point::new(px, ty + (i as f64 + 2.0) * ROW - 5.0), cell.clone(), ink.anchored(anchor));
                    }
                    cx += widths[j];
                }
                let rule = Stroke::new(self.theme.ink.with_alpha(0.7), 1.0);
                s.line(Point::new(x0 + MARGIN, ty + ROW), Point::new(x0 + MARGIN + total, ty + ROW), rule);
                let rows = cols.first().map_or(0, Vec::len);
                let bottom = ty + ROW * (rows as f64 + 1.0) + 2.0;
                s.line(
                    Point::new(x0 + MARGIN, bottom),
                    Point::new(x0 + MARGIN + total, bottom),
                    Stroke::new(self.theme.grid, 1.0),
                );
                for (i, f) in footnotes.iter().enumerate() {
                    s.text(
                        Point::new(x0 + MARGIN, bottom + (i as f64 + 1.0) * LINE - 3.0),
                        f.clone(),
                        TextStyle::new(SIZE - 1.0, self.theme.ink.with_alpha(0.8)),
                    );
                }
            }
            Item::Charts(v) => {
                for (k, (scene, _, _)) in v.iter().enumerate() {
                    let dx = x0 + MARGIN + k as f64 * (THUMB_W + GAP);
                    s.rect(
                        Rect::new(dx, y, THUMB_W, THUMB_H),
                        Some(self.theme.background),
                        Some(Stroke::new(self.theme.grid, 1.0)),
                    );
                    s.clip_push(Rect::new(dx, y, THUMB_W, THUMB_H));
                    s.embed(scene, dx, y);
                    s.clip_pop();
                }
            }
        }
    }

    /// Draw the visible part.
    pub fn render(&self) -> Scene {
        let mut s = Scene::new(self.width, self.height, self.theme.background);
        s.title = "Session".into();
        s.clip_push(Rect::new(0.0, 0.0, self.width, self.height));
        for l in &self.laid {
            if l.y + l.h < self.scroll || l.y > self.scroll + self.height {
                continue;
            }
            self.draw(&mut s, l, 0.0, -self.scroll, self.selected == Some(l.section));
        }
        s.clip_pop();
        if self.max_scroll() > 0.0 {
            let len = (self.height * self.height / self.content_h).max(20.0);
            let y = (self.height - len) * self.scroll / self.max_scroll();
            s.rect(Rect::new(self.width - 8.0, y, 6.0, len), Some(self.theme.ink.with_alpha(0.3)), None);
        }
        s
    }

    /// What is under `(x, y)` (window coordinates).
    pub fn hit_test(&self, x: f64, y: f64) -> SessionHit {
        let cy = y + self.scroll;
        for l in &self.laid {
            if cy >= l.y && cy <= l.y + l.h {
                if let Item::Charts(v) = &l.item {
                    let k = ((x - MARGIN) / (THUMB_W + GAP)).floor();
                    if k >= 0.0 && (k as usize) < v.len() && x - MARGIN - k * (THUMB_W + GAP) <= THUMB_W {
                        let (_, s, b) = v[k as usize];
                        return SessionHit::Chart(s, b);
                    }
                    return SessionHit::None;
                }
                return if l.section == usize::MAX { SessionHit::None } else { SessionHit::Section(l.section) };
            }
        }
        SessionHit::None
    }

    /// Click: selects the section under the pointer. Returns what was hit.
    pub fn pointer_down(&mut self, x: f64, y: f64) -> SessionHit {
        let h = self.hit_test(x, y);
        self.selected = match h {
            SessionHit::Section(s) | SessionHit::Chart(s, _) => Some(s),
            SessionHit::None => None,
        };
        h
    }

    /// The selected section, if any.
    pub fn selected_section(&self) -> Option<usize> {
        self.selected
    }

    /// Select every section / clear the selection.
    pub fn select_all(&mut self, on: bool) {
        self.selected = if on { Some(usize::MAX - 1) } else { None };
    }

    /// Text for the clipboard: the selected section, or the whole report.
    pub fn copy_text(&self) -> String {
        match self.selected {
            Some(i) if i < self.report.sections.len() => {
                let mut r = SessionReport::new("");
                r.push(self.report.sections[i].clone());
                r.render_text()
            }
            _ => self.report.render_text(),
        }
    }

    /// Paginate for printing: one scene per page of `page_w × page_h` pixels with a footer.
    /// Blocks are not split across pages unless one is taller than a page.
    pub fn print_pages(&self, page_w: f64, page_h: f64) -> Vec<Scene> {
        let footer = 24.0;
        let usable = (page_h - 2.0 * MARGIN - footer).max(50.0);
        let mut pages: Vec<Vec<&Laid>> = vec![Vec::new()];
        let mut page_top = 0.0;
        for l in &self.laid {
            let empty = pages.last().is_none_or(Vec::is_empty);
            if empty {
                page_top = l.y;
            } else if l.y + l.h - page_top > usable {
                pages.push(Vec::new());
                page_top = l.y;
            }
            if let Some(p) = pages.last_mut() {
                p.push(l);
            }
        }
        let n = pages.len();
        let mut starts = Vec::new();
        for p in &pages {
            starts.push(p.first().map_or(0.0, |l| l.y));
        }
        let mut view = self.clone();
        view.width = page_w;
        pages
            .iter()
            .enumerate()
            .map(|(i, items)| {
                let mut s = Scene::new(page_w, page_h, Color::WHITE);
                s.title = format!("Session page {}", i + 1);
                s.clip_push(Rect::new(0.0, 0.0, page_w, page_h - footer));
                for l in items {
                    view.draw(&mut s, l, 0.0, MARGIN - starts[i], false);
                }
                s.clip_pop();
                s.text(
                    Point::new(page_w / 2.0, page_h - 10.0),
                    format!("Page {} of {n}", i + 1),
                    TextStyle::new(10.0, crate::style::Chrome::light().muted).anchored(Anchor::Middle),
                );
                s
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_stats::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind as K, Section, Series, Table};

    fn report() -> SessionReport {
        let mut r = SessionReport::new("Session 1");
        let mut s = Section::new("Descriptive Statistics: x");
        s.text("Mean of the sample is shown below. ".repeat(12));
        let mut t = Table::new(
            "Stats",
            vec![Column::new("Variable", K::Text), Column::new("Mean", K::Fixed(2)), Column::new("P-Value", K::PValue)],
        );
        t.push_row(vec![Cell::text("x"), Cell::Num(10.0), Cell::Num(0.5)]).unwrap();
        t.push_row(vec![Cell::text("y"), Cell::Num(1234.5678), Cell::Num(0.00001)]).unwrap();
        t.footnotes.push("A footnote".into());
        s.table(t);
        for k in 0..3 {
            s.chart(ChartSpec {
                kind: ChartKind::Scatter,
                title: format!("Chart {k}"),
                x_label: "x".into(),
                y_label: "y".into(),
                categories: vec![],
                series: vec![Series { name: "s".into(), x: vec![1.0, 2.0, 3.0], y: vec![1.0, 4.0, 9.0] }],
                reference_lines: vec![],
            });
        }
        s.warn("something to note");
        r.push(s);
        let mut s2 = Section::new("Second");
        s2.text("second section");
        r.push(s2);
        r
    }

    fn texts(s: &Scene) -> Vec<(String, f64, f64)> {
        s.items
            .iter()
            .filter_map(|i| {
                if let openstats_charts::core::Shape::Text { at, text, .. } = &i.shape {
                    Some((text.clone(), at.x, at.y))
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn renders_tables_with_aligned_decimals() {
        let mut v = SessionView::new(900.0, 3000.0);
        v.set_report(&report());
        let t = texts(&v.render());
        let find = |s: &str| t.iter().find(|x| x.0 == s).unwrap_or_else(|| panic!("missing {s}: {t:?}")).clone();
        let (a, b) = (find("10.00"), find("1234.57"));
        assert!((a.1 - b.1).abs() < 1e-9, "right-aligned numeric column shares one edge");
        assert!(b.2 > a.2);
        assert_eq!(find("Variable").1, find("x").1, "text column is left-aligned with its header");
        assert!(t.iter().any(|x| x.0 == "0.000"), "tiny p-value prints as 0.000");
        assert!(t.iter().any(|x| x.0.starts_with("* WARNING *")));
        assert!(t.iter().any(|x| x.0 == "Session 1") && t.iter().any(|x| x.0 == "A footnote"));
    }

    #[test]
    fn charts_flow_into_rows_and_hit_test() {
        let mut v = SessionView::new(900.0, 900.0);
        v.set_report(&report());
        let rows = v.laid.iter().filter(|l| matches!(l.item, Item::Charts(_))).count();
        assert_eq!(rows, 2, "two thumbnails fit in 900px, the third wraps");
        let ch = v.laid.iter().find(|l| matches!(l.item, Item::Charts(_))).unwrap();
        let (y, x2) = (ch.y + 10.0, MARGIN + THUMB_W + GAP + 10.0);
        assert_eq!(v.hit_test(MARGIN + 5.0, y), SessionHit::Chart(0, 2));
        assert_eq!(v.hit_test(x2, y), SessionHit::Chart(0, 3));
        assert_eq!(v.hit_test(x2 + THUMB_W, y), SessionHit::None);
        // Clicking text selects its section and highlights it.
        let txt = v.laid.iter().find(|l| matches!(l.item, Item::Text(_))).unwrap();
        assert_eq!(v.pointer_down(30.0, txt.y + 3.0), SessionHit::Section(0));
        assert_eq!(v.selected_section(), Some(0));
    }

    #[test]
    fn scrolling_clips_to_the_window() {
        let mut v = SessionView::new(500.0, 200.0);
        v.set_report(&report());
        assert!(v.content_height() > 600.0);
        let all = v.render().items.len();
        v.scroll_by(1e9);
        assert_eq!(v.scroll(), v.content_height() - 200.0);
        let t = texts(&v.render());
        assert!(t.iter().any(|x| x.0 == "second section"));
        assert!(!t.iter().any(|x| x.0 == "Descriptive Statistics: x"));
        assert!(v.render().items.len() < all + 50);
        v.scroll_by(-1e9);
        assert_eq!(v.scroll(), 0.0);
        v.scroll_to_end();
        assert!(v.scroll() > 0.0);
    }

    #[test]
    fn copy_selection_or_everything() {
        let mut v = SessionView::new(900.0, 700.0);
        v.set_report(&report());
        let all = v.copy_text();
        assert!(all.contains("Descriptive Statistics: x") && all.contains("Second") && all.contains("1234.57"));
        let txt = v.laid.iter().rev().find(|l| matches!(l.item, Item::Text(_))).unwrap().y;
        v.pointer_down(30.0, txt + 3.0);
        let one = v.copy_text();
        assert!(one.contains("Second") && !one.contains("1234.57"));
    }

    #[test]
    fn printing_paginates_without_splitting_blocks() {
        let mut v = SessionView::new(900.0, 700.0);
        v.set_report(&report());
        let pages = v.print_pages(612.0, 400.0);
        assert!(pages.len() >= 2);
        for (i, p) in pages.iter().enumerate() {
            let t = texts(p);
            assert!(t.iter().any(|x| x.0 == format!("Page {} of {}", i + 1, pages.len())));
            assert_eq!(p.width, 612.0);
        }
        // Everything is printed exactly once.
        let joined: Vec<String> = pages.iter().flat_map(texts).map(|x| x.0).collect();
        assert_eq!(joined.iter().filter(|s| s.as_str() == "second section").count(), 1);
        assert_eq!(joined.iter().filter(|s| s.as_str() == "A footnote").count(), 1);
    }
}
