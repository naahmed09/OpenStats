//! Quality plots: control charts with zones, stages and special-cause labels, Pareto,
//! capability histogram and multi-panel layouts (sixpack, four-in-one, Gage R&R).

use super::basic::{histogram_with_curves, Curve};
use super::common::*;
use super::relation::{line_scene, LineExtras};
use super::{render_spec, PlotOptions};
use crate::core::geom::{Point, Rect};
use crate::core::hit::Region;
use crate::core::layout::{Frame, FrameSpec};
use crate::core::scale::Scale;
use crate::core::scene::{Anchor, Scene, TextStyle};
use crate::core::style::Stroke;
use openstats_stats::report::{Axis, ChartSpec, ReferenceLine};
use openstats_stats::spc::charts::ChartPanel;

/// Options of [`control_chart_panel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ControlChartOptions {
    /// Shade the ±1σ/±2σ/±3σ zones.
    pub zones: bool,
    /// Label flagged points with the numbers of the tests they failed.
    pub rule_labels: bool,
    /// Write limit values at the right edge.
    pub end_labels: bool,
    /// 0-based indices where a new stage starts.
    pub stage_breaks: Vec<usize>,
    /// Stage names (first = stage 1).
    pub stage_labels: Vec<String>,
}

impl Default for ControlChartOptions {
    fn default() -> Self {
        ControlChartOptions {
            zones: false,
            rule_labels: true,
            end_labels: true,
            stage_breaks: Vec::new(),
            stage_labels: Vec::new(),
        }
    }
}

/// Draw one control-chart panel: the statistic, stepped limits and centre line, red squares
/// on points that failed a test (with the failed test numbers), optional zones and stages.
pub fn control_chart_panel(
    panel: &ChartPanel,
    title: &str,
    x_label: &str,
    cc: &ControlChartOptions,
    opts: &PlotOptions,
) -> Scene {
    let spec = panel.chart_spec(title, x_label);
    let mut ex = LineExtras {
        end_labels: cc.end_labels,
        stage_breaks: cc.stage_breaks.clone(),
        stage_labels: cc.stage_labels.clone(),
        ..LineExtras::default()
    };
    if cc.zones || opts.zones {
        ex.zones = Some((panel.center.clone(), panel.sigma.clone()));
    }
    if cc.rule_labels {
        let mut by_point: Vec<Vec<u8>> = vec![Vec::new(); panel.values.len()];
        for v in &panel.violations {
            for &p in &v.points {
                if let Some(slot) = by_point.get_mut(p) {
                    slot.push(v.test);
                }
            }
        }
        for (i, tests) in by_point.iter().enumerate() {
            if !tests.is_empty() {
                let t: Vec<String> = tests.iter().map(u8::to_string).collect();
                ex.point_labels.push((i, t.join(",")));
            }
        }
    }
    line_scene(&spec, opts, &ex)
}

/// Pareto chart: bars sorted in decreasing order plus the cumulative-percentage line on a
/// secondary 0–100 % axis.
pub fn pareto_chart(
    categories: &[String],
    values: &[f64],
    title: &str,
    x_label: &str,
    y_label: &str,
    opts: &PlotOptions,
) -> Scene {
    let mut rows: Vec<(String, f64)> =
        categories.iter().cloned().zip(values.iter().copied()).filter(|(_, v)| v.is_finite() && *v >= 0.0).collect();
    let total: f64 = rows.iter().map(|r| r.1).sum();
    if rows.is_empty() || total <= 0.0 {
        return empty_scene(opts, title, "No data to display");
    }
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let n = rows.len();
    let ymax = rows[0].1;
    let mut fs = FrameSpec::new(
        opts.width,
        opts.height,
        Scale::Band { n },
        Scale::Linear { min: 0.0, max: ymax * 1.08 },
        &opts.theme,
    );
    fs.title = title.into();
    fs.x_label = x_label.into();
    fs.y_label = y_label.into();
    fs.x_categories = rows.iter().map(|r| r.0.clone()).collect();
    fs.y2 = Some((Scale::Linear { min: 0.0, max: 100.0 }, "Cumulative percent".into()));
    let mut f = Frame::new(fs);
    f.begin_data();
    let slot = f.plot.w / n as f64;
    let bw = slot * 0.8;
    let col = opts.theme.palette.color(0);
    let mut cum = 0.0;
    let mut line: Vec<Point> = vec![Point::new(f.plot.x, f.py2(0.0))];
    for (i, (name, v)) in rows.iter().enumerate() {
        let cx = f.px(i as f64);
        let (y0, y1) = (f.py(0.0), f.py(*v));
        let r = Rect::new(cx - bw / 2.0, y1, bw, y0 - y1);
        f.scene.rect(r, Some(col), Some(Stroke::new(col, 0.5)));
        f.scene.target(Region::Rect(r), 0, i, format!("{name}: {} ({:.1}%)", num(*v), v / total * 100.0));
        cum += v;
        let p = Point::new(cx, f.py2(cum / total * 100.0));
        line.push(p);
        f.scene.target(Region::Point(p, 6.0), 1, i, format!("{name}: cumulative {:.1}%", cum / total * 100.0));
    }
    f.scene.polyline_gaps(&line, Stroke::new(opts.theme.palette.color(1), 2.0));
    for p in line.iter().skip(1) {
        f.scene.marker(*p, crate::core::style::Marker::Circle, 5.5, opts.theme.palette.color(1), None);
    }
    let y = f.py2(80.0);
    f.scene.line(
        Point::new(f.plot.x, y),
        Point::new(f.plot.right(), y),
        Stroke::new(opts.theme.ink.lighten(0.5), 1.0).dashed(crate::core::style::Dash::Dotted),
    );
    f.end_data();
    f.finish()
}

/// Inputs of [`capability_histogram`].
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityInput {
    /// Observations.
    pub data: Vec<f64>,
    /// Lower specification limit.
    pub lsl: Option<f64>,
    /// Upper specification limit.
    pub usl: Option<f64>,
    /// Target.
    pub target: Option<f64>,
    /// Process mean.
    pub mean: f64,
    /// Within (short-term) standard deviation.
    pub sd_within: Option<f64>,
    /// Overall (long-term) standard deviation.
    pub sd_overall: f64,
}

/// Capability histogram: histogram of the data with the Within and Overall normal curves
/// and the specification limits and target as reference lines.
pub fn capability_histogram(c: &CapabilityInput, title: &str, x_label: &str, opts: &PlotOptions) -> Scene {
    let v: Vec<f64> = c.data.iter().copied().filter(|x| x.is_finite()).collect();
    let Some((lo, hi)) = range_of(&v) else { return empty_scene(opts, title, "No data to display") };
    let n = v.len() as f64;
    let k = ((n.log2() + 1.0).ceil() as usize).clamp(4, 25);
    let ticks = crate::core::ticks::extended_ticks(lo, hi, k + 1);
    let step = if ticks.len() >= 2 { ticks[1].value - ticks[0].value } else { ((hi - lo) / k as f64).max(1e-9) };
    let start = (lo / step).floor() * step;
    let nb = (((hi - start) / step).ceil() as usize).max(1);
    let mut counts = vec![0.0; nb];
    for &x in &v {
        counts[(((x - start) / step).floor() as usize).min(nb - 1)] += 1.0;
    }
    let edges: Vec<f64> = (0..=nb).map(|i| start + step * i as f64).collect();
    let mut a = edges[0];
    let mut b = edges[nb];
    for s in [c.sd_within, Some(c.sd_overall)].into_iter().flatten() {
        a = a.min(c.mean - 3.5 * s);
        b = b.max(c.mean + 3.5 * s);
    }
    let curve = |name: &str, s: f64| -> Option<Curve> {
        (s.is_finite() && s > 0.0).then(|| {
            let xs: Vec<f64> = (0..=120).map(|i| a + (b - a) * f64::from(i) / 120.0).collect();
            let ys = xs
                .iter()
                .map(|&x| {
                    n * step * (-0.5 * ((x - c.mean) / s).powi(2)).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())
                })
                .collect();
            (name.to_string(), xs, ys)
        })
    };
    let mut curves: Vec<Curve> = Vec::new();
    curves.extend(c.sd_within.and_then(|s| curve("Within", s)));
    curves.extend(curve("Overall", c.sd_overall));
    let mut refs = Vec::new();
    if let Some(l) = c.lsl {
        refs.push(ReferenceLine { axis: Axis::X, value: l, label: "LSL".into() });
    }
    if let Some(t) = c.target {
        refs.push(ReferenceLine { axis: Axis::X, value: t, label: "Target".into() });
    }
    if let Some(u) = c.usl {
        refs.push(ReferenceLine { axis: Axis::X, value: u, label: "USL".into() });
    }
    histogram_with_curves(&edges, &counts, &curves, &refs, title, x_label, "Frequency", opts)
}

/// Arrange finished scenes in a `cols`-column grid below an optional overall title. Each
/// cell is `opts.width / cols` wide; scenes are embedded (hit targets and axes included).
pub fn grid_scene(cells: &[Scene], cols: usize, title: &str, opts: &PlotOptions) -> Scene {
    let th = &opts.theme;
    let mut s = Scene::new(opts.width, opts.height, th.background);
    s.title = title.into();
    let top = if title.is_empty() { 0.0 } else { th.title_size * 1.8 };
    if !title.is_empty() {
        s.text(
            Point::new(opts.width / 2.0, 6.0 + th.title_size),
            title,
            TextStyle::new(th.title_size + 2.0, th.ink).anchored(Anchor::Middle).bold(),
        );
    }
    let cols = cols.max(1);
    let rows = cells.len().div_ceil(cols).max(1);
    let (cw, ch) = (opts.width / cols as f64, (opts.height - top) / rows as f64);
    for (i, c) in cells.iter().enumerate() {
        s.embed(c, (i % cols) as f64 * cw, top + (i / cols) as f64 * ch);
    }
    s
}

/// Render several specs as panels of one scene (sixpack, four-in-one, Gage R&R …).
pub fn panels_scene(specs: &[ChartSpec], cols: usize, title: &str, opts: &PlotOptions) -> Scene {
    let th = &opts.theme;
    let cols = cols.max(1);
    let rows = specs.len().div_ceil(cols).max(1);
    let top = if title.is_empty() { 0.0 } else { th.title_size * 1.8 };
    let (cw, ch) = (opts.width / cols as f64, (opts.height - top) / rows as f64);
    let mut cell_opts = opts.clone();
    cell_opts.width = cw;
    cell_opts.height = ch;
    cell_opts.viewport = Default::default();
    let cells: Vec<Scene> = specs.iter().map(|s| render_spec(s, &cell_opts)).collect();
    grid_scene(&cells, cols, title, opts)
}
