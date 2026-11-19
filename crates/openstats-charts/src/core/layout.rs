//! Axis frame: margins, titles, tick labels, grid and legend, laid out from
//! measured text so nothing overlaps.

use super::font::{text_width, FontMetrics};
use super::geom::{Point, Rect};
use super::scale::Scale;
use super::scene::{Anchor, AxesMeta, Scene, TextStyle};
use super::style::{Color, Dash, Marker, Stroke, Theme};
use super::ticks::Tick;

/// One legend row.
#[derive(Debug, Clone, PartialEq)]
pub struct LegendEntry {
    /// Label.
    pub name: String,
    /// Colour.
    pub color: Color,
    /// Marker (drawn at the centre of the swatch).
    pub marker: Option<Marker>,
    /// Line style (a line swatch when present).
    pub line: Option<Dash>,
}

/// Everything the frame needs to know.
#[derive(Debug, Clone)]
pub struct FrameSpec {
    /// Scene width.
    pub width: f64,
    /// Scene height.
    pub height: f64,
    /// Title.
    pub title: String,
    /// x-axis label.
    pub x_label: String,
    /// y-axis label.
    pub y_label: String,
    /// x scale.
    pub x: Scale,
    /// y scale.
    pub y: Scale,
    /// Category labels for a band x scale.
    pub x_categories: Vec<String>,
    /// Category labels for a band y scale.
    pub y_categories: Vec<String>,
    /// Secondary (right-hand) y axis.
    pub y2: Option<(Scale, String)>,
    /// Legend rows (none = no legend).
    pub legend: Vec<LegendEntry>,
    /// Theme.
    pub theme: Theme,
    /// Suppress the grid even if the theme shows one.
    pub no_grid: bool,
    /// Extra right margin (for labels drawn outside the plot area).
    pub extra_right: f64,
}

impl FrameSpec {
    /// Spec with the given size and scales.
    pub fn new(width: f64, height: f64, x: Scale, y: Scale, theme: &Theme) -> FrameSpec {
        FrameSpec {
            width,
            height,
            title: String::new(),
            x_label: String::new(),
            y_label: String::new(),
            x,
            y,
            x_categories: Vec::new(),
            y_categories: Vec::new(),
            y2: None,
            legend: Vec::new(),
            theme: theme.clone(),
            no_grid: false,
            extra_right: 0.0,
        }
    }
}

/// A laid-out frame with its chrome already drawn into `scene`.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The scene being built (chrome drawn; add data on top).
    pub scene: Scene,
    /// Plot rectangle in pixels.
    pub plot: Rect,
    /// x scale.
    pub x: Scale,
    /// y scale.
    pub y: Scale,
    /// Secondary y scale.
    pub y2: Option<Scale>,
    /// Theme.
    pub theme: Theme,
}

fn label_of(scale: &Scale, cats: &[String], t: &Tick) -> String {
    match scale {
        Scale::Band { .. } => cats.get(t.value as usize).cloned().unwrap_or_default(),
        _ => t.label.clone(),
    }
}

impl Frame {
    /// Lay out and draw the chrome.
    pub fn new(spec: FrameSpec) -> Frame {
        let th = &spec.theme;
        let (w, h) = (spec.width, spec.height);
        let tick_f = FontMetrics::new(th.tick_size);
        let label_f = FontMetrics::new(th.label_size);
        let pad = 8.0;
        let title_h = if spec.title.is_empty() { 0.0 } else { th.title_size * 1.5 };
        let top = pad + title_h;
        let legend_w = if spec.legend.is_empty() {
            0.0
        } else {
            spec.legend.iter().map(|e| text_width(&e.name, th.tick_size, false)).fold(0.0, f64::max) + 40.0
        };
        let ylab_h = if spec.y_label.is_empty() { 0.0 } else { label_f.line_height() };
        let xlab_h = if spec.x_label.is_empty() { 0.0 } else { label_f.line_height() };

        // First pass: tick labels of the y axis (height estimate).
        let est_h = (h - top - 60.0).max(40.0);
        let y_ticks = |ph: f64| -> Vec<Tick> { spec.y.ticks(((ph / 38.0) as usize).clamp(2, 12)) };
        let y_lab_w = |ticks: &[Tick]| -> f64 {
            ticks
                .iter()
                .map(|t| text_width(&label_of(&spec.y, &spec.y_categories, t), th.tick_size, false))
                .fold(0.0, f64::max)
        };
        let yt0 = y_ticks(est_h);
        let y2_w = spec.y2.as_ref().map_or(0.0, |(s, l)| {
            let tk = s.ticks(5);
            let lw = tk.iter().map(|t| text_width(&t.label, th.tick_size, false)).fold(0.0, f64::max);
            lw + 10.0 + if l.is_empty() { 0.0 } else { label_f.line_height() }
        });
        let left = pad + ylab_h + y_lab_w(&yt0) + 12.0;
        let right = pad + 6.0 + legend_w + y2_w + spec.extra_right;
        let plot_w = (w - left - right).max(20.0);

        // x tick labels decide rotation and the bottom margin.
        let xt = spec.x.ticks(((plot_w / 70.0) as usize).clamp(2, 14));
        let x_labels: Vec<String> = xt.iter().map(|t| label_of(&spec.x, &spec.x_categories, t)).collect();
        let max_lw = x_labels.iter().map(|s| text_width(s, th.tick_size, false)).fold(0.0, f64::max);
        let slot = plot_w / xt.len().max(1) as f64;
        let rotate = matches!(spec.x, Scale::Band { .. }) && max_lw + 6.0 > slot && max_lw > 0.0;
        let tick_label_h = if rotate { max_lw + 4.0 } else { tick_f.line_height() };
        let bottom = pad + xlab_h + tick_label_h + 10.0;
        let plot_h = (h - top - bottom).max(20.0);
        let plot = Rect::new(left, top, plot_w, plot_h);
        let yt = y_ticks(plot_h);

        let mut s = Scene::new(w, h, th.background);
        s.title = spec.title.clone();
        s.rect(plot, Some(th.plot_background), None);
        let grid = Stroke::new(th.grid, 1.0);
        let ink = Stroke::new(th.ink, 1.0);
        let show_grid = th.show_grid && !spec.no_grid;
        let tx = |v: f64| spec.x.map(v, plot.x, plot.right());
        let ty = |v: f64| spec.y.map(v, plot.bottom(), plot.y);

        // Grid and ticks.
        if show_grid {
            for t in xt.iter().filter(|t| t.major) {
                let p = tx(t.value);
                s.line(Point::new(p, plot.y), Point::new(p, plot.bottom()), grid);
            }
            for t in yt.iter().filter(|t| t.major) {
                let p = ty(t.value);
                s.line(Point::new(plot.x, p), Point::new(plot.right(), p), grid);
            }
        }
        s.rect(plot, None, Some(ink));
        let tick_style = TextStyle::new(th.tick_size, th.ink);
        // x ticks and labels (thinned so they never overlap).
        let mut last_right = f64::NEG_INFINITY;
        for (t, lab) in xt.iter().zip(&x_labels) {
            let p = tx(t.value);
            if !p.is_finite() {
                continue;
            }
            s.line(Point::new(p, plot.bottom()), Point::new(p, plot.bottom() + if t.major { 5.0 } else { 3.0 }), ink);
            if !t.major || lab.is_empty() {
                continue;
            }
            let lw = text_width(lab, th.tick_size, false);
            if rotate {
                s.text(
                    Point::new(p + th.tick_size * 0.35, plot.bottom() + 8.0 + lw),
                    lab.clone(),
                    tick_style.anchored(Anchor::Start).rotated(90.0),
                );
            } else if p - lw / 2.0 > last_right + 6.0 || last_right.is_infinite() {
                s.text(
                    Point::new(p, plot.bottom() + 8.0 + tick_f.ascent()),
                    lab.clone(),
                    tick_style.anchored(Anchor::Middle),
                );
                last_right = p + lw / 2.0;
            }
        }
        // y ticks and labels.
        let mut last_top = f64::INFINITY;
        let mut ytk: Vec<(&Tick, f64)> = yt.iter().map(|t| (t, ty(t.value))).collect();
        ytk.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (t, p) in ytk {
            if !p.is_finite() {
                continue;
            }
            s.line(Point::new(plot.x - if t.major { 5.0 } else { 3.0 }, p), Point::new(plot.x, p), ink);
            let lab = label_of(&spec.y, &spec.y_categories, t);
            if !t.major || lab.is_empty() {
                continue;
            }
            if p + tick_f.ascent() / 2.0 < last_top - 2.0 || last_top.is_infinite() {
                s.text(
                    Point::new(plot.x - 8.0, p + tick_f.ascent() / 2.0 - 0.5),
                    lab,
                    tick_style.anchored(Anchor::End),
                );
                last_top = p - tick_f.ascent() / 2.0;
            }
        }
        // Secondary axis.
        let mut y2 = None;
        if let Some((sc, lab)) = &spec.y2 {
            let mut x_off = plot.right() + 8.0;
            for t in sc.ticks(5) {
                let p = sc.map(t.value, plot.bottom(), plot.y);
                s.line(Point::new(plot.right(), p), Point::new(plot.right() + 5.0, p), ink);
                s.text(Point::new(plot.right() + 8.0, p + tick_f.ascent() / 2.0), t.label.clone(), tick_style);
                x_off = x_off.max(plot.right() + 8.0 + text_width(&t.label, th.tick_size, false));
            }
            if !lab.is_empty() {
                s.text(
                    Point::new(x_off + 6.0 + label_f.ascent(), plot.center().y),
                    lab.clone(),
                    TextStyle::new(th.label_size, th.ink).anchored(Anchor::Middle).rotated(-90.0),
                );
            }
            y2 = Some(sc.clone());
        }
        // Axis labels and title.
        let lab_style = TextStyle::new(th.label_size, th.ink).anchored(Anchor::Middle);
        if !spec.x_label.is_empty() {
            s.text(Point::new(plot.center().x, h - pad - label_f.descent()), spec.x_label.clone(), lab_style);
        }
        if !spec.y_label.is_empty() {
            s.text(Point::new(pad + label_f.ascent(), plot.center().y), spec.y_label.clone(), lab_style.rotated(90.0));
        }
        if !spec.title.is_empty() {
            s.text(
                Point::new(plot.center().x, pad + th.title_size * 0.95),
                spec.title.clone(),
                TextStyle::new(th.title_size, th.ink).anchored(Anchor::Middle).bold(),
            );
        }
        // Legend.
        if !spec.legend.is_empty() {
            let lx = w - right + 6.0 + y2_w + spec.extra_right;
            let mut ly = plot.y + 4.0;
            for e in &spec.legend {
                let c = Point::new(lx + 10.0, ly + 7.0);
                if let Some(d) = e.line {
                    s.line(
                        Point::new(lx, c.y),
                        Point::new(lx + 20.0, c.y),
                        Stroke { color: e.color, width: 2.0, dash: d },
                    );
                    if let Some(m) = e.marker {
                        s.marker(c, m, 7.0, e.color, None);
                    }
                } else if let Some(m) = e.marker {
                    s.marker(c, m, 7.0, e.color, None);
                } else {
                    s.rect(Rect::new(lx + 3.0, ly + 1.0, 14.0, 12.0), Some(e.color), None);
                }
                s.text(Point::new(lx + 26.0, ly + 7.0 + tick_f.ascent() / 2.0 - 0.5), e.name.clone(), tick_style);
                ly += tick_f.line_height() + 3.0;
            }
        }
        s.axes.push(AxesMeta { plot, x: spec.x.clone(), y: spec.y.clone() });
        Frame { scene: s, plot, x: spec.x, y: spec.y, y2, theme: spec.theme }
    }

    /// Pixel x of data value `v`.
    pub fn px(&self, v: f64) -> f64 {
        self.x.map(v, self.plot.x, self.plot.right())
    }

    /// Pixel y of data value `v`.
    pub fn py(&self, v: f64) -> f64 {
        self.y.map(v, self.plot.bottom(), self.plot.y)
    }

    /// Pixel y on the secondary axis.
    pub fn py2(&self, v: f64) -> f64 {
        self.y2.as_ref().map_or(f64::NAN, |s| s.map(v, self.plot.bottom(), self.plot.y))
    }

    /// Pixel point of a data coordinate.
    pub fn pt(&self, x: f64, y: f64) -> Point {
        Point::new(self.px(x), self.py(y))
    }

    /// Start clipping to the plot area.
    pub fn begin_data(&mut self) {
        let r = self.plot;
        self.scene.clip_push(r);
    }

    /// Stop clipping.
    pub fn end_data(&mut self) {
        self.scene.clip_pop();
    }

    /// Finish and return the scene.
    pub fn finish(self) -> Scene {
        self.scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scene::Shape;

    #[test]
    fn frame_layout_is_inside_the_scene() {
        let th = Theme::light();
        let mut spec = FrameSpec::new(
            640.0,
            400.0,
            Scale::Linear { min: 0.0, max: 100.0 },
            Scale::Linear { min: -5.0, max: 5.0 },
            &th,
        );
        spec.title = "Title".into();
        spec.x_label = "x".into();
        spec.y_label = "y".into();
        spec.legend = vec![LegendEntry {
            name: "Series one".into(),
            color: th.palette.color(0),
            marker: Some(Marker::Circle),
            line: Some(Dash::Solid),
        }];
        let f = Frame::new(spec);
        assert!(f.plot.x > 20.0 && f.plot.right() < 640.0 - 60.0 && f.plot.y > 10.0 && f.plot.bottom() < 400.0 - 20.0);
        assert!((f.px(0.0) - f.plot.x).abs() < 1e-9 && (f.py(5.0) - f.plot.y).abs() < 1e-9);
        // All text stays inside the canvas.
        for it in &f.scene.items {
            if let Shape::Text { at, text, style } = &it.shape {
                if style.rotate == 0.0 {
                    let tw = text_width(text, style.size, style.bold);
                    let (l, r) = match style.anchor {
                        Anchor::Start => (at.x, at.x + tw),
                        Anchor::Middle => (at.x - tw / 2.0, at.x + tw / 2.0),
                        Anchor::End => (at.x - tw, at.x),
                    };
                    assert!(l >= 0.0 && r <= 640.0 && at.y <= 400.0 && at.y >= 0.0, "{text}");
                }
            }
        }
        assert!(f.scene.clips_balanced());
    }

    #[test]
    fn category_labels_rotate_when_crowded() {
        let th = Theme::light();
        let cats: Vec<String> = (0..12).map(|i| format!("Category number {i}")).collect();
        let mut spec = FrameSpec::new(500.0, 300.0, Scale::Band { n: 12 }, Scale::Linear { min: 0.0, max: 1.0 }, &th);
        spec.x_categories = cats;
        let f = Frame::new(spec);
        assert!(f.scene.items.iter().any(|i| matches!(&i.shape, Shape::Text { style, .. } if style.rotate == 90.0)));
        assert!(f.plot.bottom() < 300.0 - 50.0);
    }
}
