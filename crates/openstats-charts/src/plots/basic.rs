//! Basic and distribution plots: histogram, box plot, individual value plot,
//! interval plot, bar chart, pie chart, empirical CDF and probability plot.

use super::common::*;
use super::PlotOptions;
use crate::core::geom::{Point, Rect};
use crate::core::hit::Region;
use crate::core::layout::{Frame, FrameSpec, LegendEntry};
use crate::core::scale::{ProbabilityKind, Scale};
use crate::core::scene::{Anchor, Scene, TextStyle};
use crate::core::style::{Color, Dash, Marker, Stroke};
use crate::core::ticks::extended_ticks;
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::special::erf::norm_ppf;
use openstats_stats::graphical::boxplot;
use openstats_stats::report::{ChartSpec, ReferenceLine};

fn base_spec(opts: &PlotOptions, x: Scale, y: Scale, title: &str, xl: &str, yl: &str) -> FrameSpec {
    let mut f = FrameSpec::new(
        opts.width,
        opts.height,
        with_view(x, opts.viewport.x),
        with_view(y, opts.viewport.y),
        &opts.theme,
    );
    f.title = title.into();
    f.x_label = xl.into();
    f.y_label = yl.into();
    f
}

// ------------------------------------------------------------------ histogram

/// A curve drawn over a histogram (in count units).
pub type Curve = (String, Vec<f64>, Vec<f64>);

/// Overlay of [`histogram_from_data`].
#[derive(Debug, Clone, PartialEq)]
pub enum HistogramOverlay {
    /// No curve.
    None,
    /// Normal density with the sample mean and standard deviation.
    Normal,
    /// Normal density with given parameters.
    NormalWith {
        /// Mean.
        mean: f64,
        /// Standard deviation.
        sd: f64,
    },
    /// An arbitrary density curve `(x, density)`; scaled to counts by `n × bin width`.
    Density {
        /// Legend name.
        name: String,
        /// Abscissae.
        x: Vec<f64>,
        /// Density values.
        y: Vec<f64>,
    },
}

/// Histogram from bin edges and counts with optional curves and reference lines.
#[allow(clippy::too_many_arguments)]
pub fn histogram_with_curves(
    edges: &[f64],
    counts: &[f64],
    curves: &[Curve],
    refs: &[ReferenceLine],
    title: &str,
    x_label: &str,
    y_label: &str,
    opts: &PlotOptions,
) -> Scene {
    if edges.len() != counts.len() + 1 || counts.is_empty() || edges.iter().any(|e| !e.is_finite()) {
        return empty_scene(opts, title, "No data to display");
    }
    let xr = union(range_of(edges), range_of(curves.iter().flat_map(|c| c.1.iter())));
    let ymax = counts
        .iter()
        .chain(curves.iter().flat_map(|c| c.2.iter()))
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f64, f64::max);
    let xs = Scale::linear_padded(xr.map_or(0.0, |r| r.0), xr.map_or(1.0, |r| r.1), 0.03);
    let ys = Scale::Linear { min: 0.0, max: if ymax > 0.0 { ymax * 1.08 } else { 1.0 } };
    let mut spec = base_spec(opts, xs, ys, title, x_label, y_label);
    if !curves.is_empty() {
        spec.legend = curves
            .iter()
            .enumerate()
            .map(|(i, c)| LegendEntry {
                name: c.0.clone(),
                color: opts.theme.palette.color(i + 1),
                marker: None,
                line: Some(opts.theme.palette.dash(i)),
            })
            .collect();
    }
    let mut f = Frame::new(spec);
    let fill = opts.theme.palette.color(0).lighten(0.35);
    let edge = opts.theme.palette.color(0);
    f.begin_data();
    for (i, &c) in counts.iter().enumerate() {
        let (x0, x1) = (f.px(edges[i]), f.px(edges[i + 1]));
        let (y0, y1) = (f.py(0.0), f.py(c));
        let r = Rect::new(x0, y1, x1 - x0, y0 - y1);
        f.scene.rect(r, Some(fill), Some(Stroke::new(edge, 1.0)));
        f.scene.target(Region::Rect(r), 0, i, format!("[{}, {}): {}", num(edges[i]), num(edges[i + 1]), num(c)));
    }
    for (i, c) in curves.iter().enumerate() {
        let pts: Vec<Point> = c.1.iter().zip(&c.2).map(|(&x, &y)| f.pt(x, y)).collect();
        f.scene.polyline_gaps(
            &pts,
            Stroke { color: opts.theme.palette.color(i + 1), width: 2.0, dash: opts.theme.palette.dash(i) },
        );
    }
    reference_lines(&mut f, refs);
    f.end_data();
    f.finish()
}

/// Bin `data` (Sturges' rule unless `bins` is given) and draw a histogram with an overlay.
pub fn histogram_from_data(
    data: &[f64],
    bins: Option<usize>,
    overlay: &HistogramOverlay,
    title: &str,
    x_label: &str,
    opts: &PlotOptions,
) -> Scene {
    let v: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return empty_scene(opts, title, "No data to display");
    }
    let n = v.len() as f64;
    let (lo, hi) = range_of(&v).unwrap_or((0.0, 1.0));
    let k = bins.unwrap_or_else(|| ((n.log2() + 1.0).ceil() as usize).max(1)).max(1);
    let (start, step, nb) = if hi - lo <= 0.0 {
        (lo - 0.5, 1.0, 1)
    } else {
        let t = extended_ticks(lo, hi, k + 1);
        let step = if t.len() >= 2 { t[1].value - t[0].value } else { (hi - lo) / k as f64 };
        let start = (lo / step).floor() * step;
        let nb = (((hi - start) / step).ceil() as usize).max(1);
        (start, step, nb)
    };
    let mut counts = vec![0.0; nb];
    for &x in &v {
        let i = (((x - start) / step).floor() as usize).min(nb - 1);
        counts[i] += 1.0;
    }
    let edges: Vec<f64> = (0..=nb).map(|i| start + step * i as f64).collect();
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let normal = |m: f64, s: f64| -> Option<Curve> {
        if !(s.is_finite() && s > 0.0) {
            return None;
        }
        let (a, b) = (edges[0].min(m - 3.5 * s), edges[nb].max(m + 3.5 * s));
        let xs: Vec<f64> = (0..=120).map(|i| a + (b - a) * f64::from(i) / 120.0).collect();
        let ys = xs
            .iter()
            .map(|&x| n * step * (-0.5 * ((x - m) / s).powi(2)).exp() / (s * (2.0 * std::f64::consts::PI).sqrt()))
            .collect();
        Some(("Normal".into(), xs, ys))
    };
    let curves: Vec<Curve> = match overlay {
        HistogramOverlay::None => Vec::new(),
        HistogramOverlay::Normal => normal(mean, sd).into_iter().collect(),
        HistogramOverlay::NormalWith { mean, sd } => normal(*mean, *sd).into_iter().collect(),
        HistogramOverlay::Density { name, x, y } => {
            vec![(name.clone(), x.clone(), y.iter().map(|d| d * n * step).collect())]
        }
    };
    histogram_with_curves(&edges, &counts, &curves, &[], title, x_label, "Frequency", opts)
}

pub(crate) fn histogram_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(h) = spec.series.first() else { return empty_scene(opts, &spec.title, "No data to display") };
    let curves: Vec<Curve> = spec.series.iter().skip(1).map(|s| (s.name.clone(), s.x.clone(), s.y.clone())).collect();
    histogram_with_curves(&h.x, &h.y, &curves, &spec.reference_lines, &spec.title, &spec.x_label, &spec.y_label, opts)
}

// ------------------------------------------------------------------ box plot

/// One group of raw data for a box plot.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxGroup {
    /// Label.
    pub label: String,
    /// Observations (missing values are ignored).
    pub data: Vec<f64>,
}

struct BoxData {
    label: String,
    pos: f64,
    lo: f64,
    q1: f64,
    med: f64,
    q3: f64,
    hi: f64,
    mean: Option<f64>,
    outliers: Vec<f64>,
    notch: Option<(f64, f64)>,
}

fn draw_boxes(
    boxes: &[BoxData],
    title: &str,
    xl: &str,
    yl: &str,
    use_band: bool,
    refs: &[ReferenceLine],
    opts: &PlotOptions,
) -> Scene {
    if boxes.is_empty() {
        return empty_scene(opts, title, "No data to display");
    }
    let yr = boxes
        .iter()
        .flat_map(|b| {
            [b.lo, b.hi].into_iter().chain(b.outliers.iter().copied()).chain(b.notch.iter().flat_map(|n| [n.0, n.1]))
        })
        .fold(None, |acc: Option<(f64, f64)>, v| union(acc, Some((v, v))));
    let n = boxes.len();
    let xs = if use_band {
        Scale::Band { n }
    } else {
        let r = range_of(boxes.iter().map(|b| &b.pos)).unwrap_or((0.0, 1.0));
        Scale::linear_padded(r.0 - 0.6, r.1 + 0.6, 0.0)
    };
    let mut spec = base_spec(opts, xs, linear_scale(yr, false, 0.06), title, xl, yl);
    if use_band {
        spec.x_categories = boxes.iter().map(|b| b.label.clone()).collect();
    }
    let mut f = Frame::new(spec);
    let col = opts.theme.palette.color(0);
    let ink = Stroke::new(opts.theme.ink, 1.2);
    f.begin_data();
    let slot = if use_band { f.plot.w / n as f64 } else { f.plot.w / (f.x.domain().1 - f.x.domain().0) };
    let half = (slot * 0.5 * 0.55).min(48.0);
    for (i, b) in boxes.iter().enumerate() {
        let cx = f.px(if use_band { i as f64 } else { b.pos });
        let (yq1, yq3, ym) = (f.py(b.q1), f.py(b.q3), f.py(b.med));
        let rect = Rect::new(cx - half, yq3, 2.0 * half, yq1 - yq3);
        match b.notch {
            Some((nl, nu)) => {
                let (nl, nu) = (nl.max(b.q1), nu.min(b.q3));
                let (yl_, yu) = (f.py(nl), f.py(nu));
                let w2 = half * 0.55;
                let poly = vec![
                    Point::new(cx - half, yq3),
                    Point::new(cx + half, yq3),
                    Point::new(cx + half, yu),
                    Point::new(cx + w2, ym),
                    Point::new(cx + half, yl_),
                    Point::new(cx + half, yq1),
                    Point::new(cx - half, yq1),
                    Point::new(cx - half, yl_),
                    Point::new(cx - w2, ym),
                    Point::new(cx - half, yu),
                ];
                f.scene.polygon(poly, Some(col.lighten(0.5)), Some(ink));
                f.scene.line(Point::new(cx - w2, ym), Point::new(cx + w2, ym), Stroke::new(opts.theme.ink, 2.0));
            }
            None => {
                f.scene.rect(rect, Some(col.lighten(0.5)), Some(ink));
                f.scene.line(Point::new(cx - half, ym), Point::new(cx + half, ym), Stroke::new(opts.theme.ink, 2.0));
            }
        }
        f.scene.line(Point::new(cx, yq3), Point::new(cx, f.py(b.hi)), ink);
        f.scene.line(Point::new(cx, yq1), Point::new(cx, f.py(b.lo)), ink);
        for w in [b.hi, b.lo] {
            f.scene.line(Point::new(cx - half * 0.5, f.py(w)), Point::new(cx + half * 0.5, f.py(w)), ink);
        }
        for &o in &b.outliers {
            f.scene.marker(Point::new(cx, f.py(o)), Marker::Cross, 6.0, opts.theme.ink, None);
        }
        if let Some(m) = b.mean {
            f.scene.marker(Point::new(cx, f.py(m)), Marker::Diamond, 6.0, Color::WHITE, Some(ink));
        }
        f.scene.target(
            Region::Rect(Rect::new(cx - half, f.py(b.hi), 2.0 * half, f.py(b.lo) - f.py(b.hi))),
            i,
            i,
            format!("{}: median {}, Q1 {}, Q3 {}", b.label, num(b.med), num(b.q1), num(b.q3)),
        );
    }
    reference_lines(&mut f, refs);
    f.end_data();
    f.finish()
}

/// Box plots of raw groups (Minitab quartiles, 1.5 IQR whiskers); optionally notched
/// (median ± 1.58 IQR/√n) and with the mean marked.
pub fn boxplot_groups(
    groups: &[BoxGroup],
    notched: bool,
    show_mean: bool,
    title: &str,
    x_label: &str,
    y_label: &str,
    opts: &PlotOptions,
) -> Scene {
    let mut boxes = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        let Ok(b) = boxplot(&g.data) else { continue };
        let half = 1.58 * (b.q3 - b.q1) / (b.n as f64).sqrt();
        boxes.push(BoxData {
            label: g.label.clone(),
            pos: (i + 1) as f64,
            lo: b.lower_whisker,
            q1: b.q1,
            med: b.median,
            q3: b.q3,
            hi: b.upper_whisker,
            mean: show_mean.then_some(b.mean),
            outliers: b.low_outliers.iter().chain(&b.high_outliers).copied().collect(),
            notch: notched.then_some((b.median - half, b.median + half)),
        });
    }
    draw_boxes(&boxes, title, x_label, y_label, true, &[], opts)
}

pub(crate) fn boxplot_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let mut boxes = Vec::new();
    let mut outliers: Vec<(f64, f64)> = Vec::new();
    for s in &spec.series {
        if s.name.to_lowercase().contains("outlier") {
            outliers.extend(s.x.iter().copied().zip(s.y.iter().copied()));
        } else if s.y.len() == 5 && s.x.len() == 5 {
            boxes.push(BoxData {
                label: s.name.clone(),
                pos: s.x[0],
                lo: s.y[0],
                q1: s.y[1],
                med: s.y[2],
                q3: s.y[3],
                hi: s.y[4],
                mean: None,
                outliers: Vec::new(),
                notch: None,
            });
        }
    }
    for b in boxes.iter_mut() {
        b.outliers = outliers.iter().filter(|(x, _)| (*x - b.pos).abs() < 1e-9).map(|p| p.1).collect();
    }
    let band = !spec.categories.is_empty() || boxes.len() > 1;
    if !spec.categories.is_empty() {
        for (i, b) in boxes.iter_mut().enumerate() {
            if let Some(c) = spec.categories.get(i) {
                b.label = c.clone();
            }
        }
    }
    draw_boxes(&boxes, &spec.title, &spec.x_label, &spec.y_label, band, &spec.reference_lines, opts)
}

// ------------------------------------------------------------------ individual value / interval

/// Individual value plot: every observation as a marker, with the group mean tick.
pub fn individual_value_plot(
    groups: &[BoxGroup],
    title: &str,
    x_label: &str,
    y_label: &str,
    opts: &PlotOptions,
) -> Scene {
    let n = groups.len();
    let yr = range_of(groups.iter().flat_map(|g| g.data.iter()));
    if n == 0 || yr.is_none() {
        return empty_scene(opts, title, "No data to display");
    }
    let mut spec = base_spec(opts, Scale::Band { n }, linear_scale(yr, false, 0.06), title, x_label, y_label);
    spec.x_categories = groups.iter().map(|g| g.label.clone()).collect();
    let mut f = Frame::new(spec);
    f.begin_data();
    let slot = f.plot.w / n as f64;
    let total: usize = groups.iter().map(|g| g.data.len()).sum();
    let ms = marker_size(total);
    let mut seed = 12345u64;
    let mut targets = 0;
    for (gi, g) in groups.iter().enumerate() {
        let cx = f.px(gi as f64);
        let vals: Vec<f64> = g.data.iter().copied().filter(|v| v.is_finite()).collect();
        for (i, &v) in g.data.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * slot * 0.45;
            let p = Point::new(cx + j, f.py(v));
            f.scene.marker(p, Marker::OpenCircle, ms, opts.theme.palette.color(0), None);
            if targets < MAX_TARGETS {
                f.scene.target(Region::Point(p, ms), gi, i, tip(&g.label, "", gi as f64 + 1.0, y_label, v));
                targets += 1;
            }
        }
        if !vals.is_empty() {
            let m = vals.iter().sum::<f64>() / vals.len() as f64;
            f.scene.line(
                Point::new(cx - slot * 0.3, f.py(m)),
                Point::new(cx + slot * 0.3, f.py(m)),
                Stroke::new(opts.theme.palette.color(1), 2.0),
            );
        }
    }
    f.end_data();
    f.finish()
}

/// One group of an interval plot.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalGroup {
    /// Label.
    pub label: String,
    /// Point estimate.
    pub estimate: f64,
    /// Lower limit.
    pub lower: f64,
    /// Upper limit.
    pub upper: f64,
}

fn draw_intervals(
    groups: &[IntervalGroup],
    horizontal: bool,
    title: &str,
    xl: &str,
    yl: &str,
    refs: &[ReferenceLine],
    opts: &PlotOptions,
) -> Scene {
    let n = groups.len();
    let vr = range_of(groups.iter().flat_map(|g| [&g.estimate, &g.lower, &g.upper]));
    if n == 0 || vr.is_none() {
        return empty_scene(opts, title, "No data to display");
    }
    let vs = linear_scale(vr, false, 0.1);
    let (xs, ys) = if horizontal { (vs, Scale::Band { n }) } else { (Scale::Band { n }, vs) };
    let mut spec = base_spec(opts, xs, ys, title, xl, yl);
    let labels: Vec<String> = groups.iter().map(|g| g.label.clone()).collect();
    if horizontal {
        spec.y_categories = labels;
    } else {
        spec.x_categories = labels;
    }
    let mut f = Frame::new(spec);
    f.begin_data();
    let st = Stroke::new(opts.theme.palette.color(0), 1.8);
    for (i, g) in groups.iter().enumerate() {
        let (c, l, u, e) = if horizontal {
            let c = f.py(i as f64);
            (c, f.px(g.lower), f.px(g.upper), Point::new(f.px(g.estimate), c))
        } else {
            let c = f.px(i as f64);
            (c, f.py(g.lower), f.py(g.upper), Point::new(c, f.py(g.estimate)))
        };
        let cap = 5.0;
        if horizontal {
            f.scene.line(Point::new(l, c), Point::new(u, c), st);
            f.scene.line(Point::new(l, c - cap), Point::new(l, c + cap), st);
            f.scene.line(Point::new(u, c - cap), Point::new(u, c + cap), st);
        } else {
            f.scene.line(Point::new(c, l), Point::new(c, u), st);
            f.scene.line(Point::new(c - cap, l), Point::new(c + cap, l), st);
            f.scene.line(Point::new(c - cap, u), Point::new(c + cap, u), st);
        }
        f.scene.marker(e, Marker::Circle, 7.0, opts.theme.palette.color(0), Some(Stroke::new(Color::WHITE, 1.0)));
        f.scene.target(
            Region::Point(e, 8.0),
            0,
            i,
            format!("{}: {} ({}, {})", g.label, num(g.estimate), num(g.lower), num(g.upper)),
        );
    }
    reference_lines(&mut f, refs);
    f.end_data();
    f.finish()
}

/// Vertical interval plot (estimate with confidence interval per group).
pub fn interval_plot(groups: &[IntervalGroup], title: &str, x_label: &str, y_label: &str, opts: &PlotOptions) -> Scene {
    draw_intervals(groups, false, title, x_label, y_label, &[], opts)
}

pub(crate) fn interval_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let find = |n: &str| spec.series.iter().find(|s| s.name == n);
    let (Some(e), Some(l), Some(u)) = (find("estimate"), find("lower"), find("upper")) else {
        return empty_scene(opts, &spec.title, "No data to display");
    };
    // Horizontal when the positions (y) are the ranks 1..n and the estimates vary in x.
    let horizontal = e.y.iter().enumerate().all(|(i, &y)| (y - (i + 1) as f64).abs() < 1e-9) && e.x.len() > 1
        || spec.x_label.contains("Mean")
        || spec.x_label.contains("Percent");
    let n = e.x.len().min(l.x.len()).min(u.x.len());
    let groups: Vec<IntervalGroup> = (0..n)
        .map(|i| {
            let (a, b) = if horizontal { (l.x[i], u.x[i]) } else { (l.y[i], u.y[i]) };
            IntervalGroup {
                label: spec.categories.get(i).cloned().unwrap_or_else(|| format!("{}", i + 1)),
                estimate: if horizontal { e.x[i] } else { e.y[i] },
                lower: a,
                upper: b,
            }
        })
        .collect();
    draw_intervals(&groups, horizontal, &spec.title, &spec.x_label, &spec.y_label, &spec.reference_lines, opts)
}

// ------------------------------------------------------------------ bars and pie

/// Clustered bar chart: `series[k].1[i]` is the height of series `k` at category `i`.
pub fn bar_chart(
    categories: &[String],
    series: &[(String, Vec<f64>)],
    title: &str,
    x_label: &str,
    y_label: &str,
    opts: &PlotOptions,
) -> Scene {
    bar_draw(categories, series, title, x_label, y_label, &[], opts)
}

fn bar_draw(
    categories: &[String],
    series: &[(String, Vec<f64>)],
    title: &str,
    xl: &str,
    yl: &str,
    refs: &[ReferenceLine],
    opts: &PlotOptions,
) -> Scene {
    let n = categories.len().max(series.iter().map(|s| s.1.len()).max().unwrap_or(0));
    let yr = range_of(
        series
            .iter()
            .flat_map(|s| s.1.iter())
            .chain(refs.iter().filter(|r| r.axis == openstats_stats::report::Axis::Y).map(|r| &r.value)),
    );
    if n == 0 || yr.is_none() {
        return empty_scene(opts, title, "No data to display");
    }
    let mut spec = base_spec(opts, Scale::Band { n }, linear_scale(yr, true, 0.08), title, xl, yl);
    spec.x_categories = (0..n).map(|i| categories.get(i).cloned().unwrap_or_else(|| format!("{}", i + 1))).collect();
    if series.len() > 1 {
        let names: Vec<&str> = series.iter().map(|s| s.0.as_str()).collect();
        spec.legend = series_legend(&names, &opts.theme, false, false);
    }
    let mut f = Frame::new(spec);
    f.begin_data();
    let slot = f.plot.w / n as f64;
    let k = series.len().max(1);
    let group_w = slot * 0.8;
    let bw = group_w / k as f64;
    for (si, s) in series.iter().enumerate() {
        let col = opts.theme.palette.color(si);
        for (i, &v) in s.1.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let x0 = f.px(i as f64) - group_w / 2.0 + bw * si as f64;
            let (y0, y1) = (f.py(0.0), f.py(v));
            let r = Rect::new(x0, y1.min(y0), bw - 1.0, (y0 - y1).abs());
            f.scene.rect(r, Some(col), Some(Stroke::new(col.lighten(0.0), 0.5)));
            f.scene.target(Region::Rect(r), si, i, format!("{} / {}: {}", s.0, f.x_cat(i), num(v)));
        }
    }
    reference_lines(&mut f, refs);
    f.end_data();
    f.finish()
}

pub(crate) fn bar_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let series: Vec<(String, Vec<f64>)> = spec.series.iter().map(|s| (s.name.clone(), s.y.clone())).collect();
    bar_draw(&spec.categories, &series, &spec.title, &spec.x_label, &spec.y_label, &spec.reference_lines, opts)
}

trait CatLabel {
    fn x_cat(&self, i: usize) -> String;
}

impl CatLabel for Frame {
    fn x_cat(&self, i: usize) -> String {
        format!("{}", i + 1)
    }
}

/// Pie chart with percentage labels. Values must be finite and ≥ 0 with a positive sum.
pub fn pie_chart(labels: &[String], values: &[f64], title: &str, opts: &PlotOptions) -> Result<Scene, String> {
    if labels.len() != values.len() {
        return Err("labels and values must have the same length".into());
    }
    if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("pie values must be finite and non-negative".into());
    }
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        return Err("pie values must have a positive sum".into());
    }
    let th = &opts.theme;
    let mut s = Scene::new(opts.width, opts.height, th.background);
    s.title = title.into();
    let top = if title.is_empty() { 12.0 } else { 12.0 + th.title_size * 1.5 };
    if !title.is_empty() {
        s.text(
            Point::new(opts.width * 0.4, 8.0 + th.title_size * 0.95),
            title,
            TextStyle::new(th.title_size, th.ink).anchored(Anchor::Middle).bold(),
        );
    }
    let legend_w =
        labels.iter().map(|l| crate::core::font::text_width(l, th.tick_size, false)).fold(0.0, f64::max) + 44.0;
    let area_w = opts.width - legend_w - 20.0;
    let r = ((area_w.min(opts.height - top - 12.0)) / 2.0 - 34.0).max(20.0);
    let c = Point::new(10.0 + area_w / 2.0, top + (opts.height - top - 12.0) / 2.0);
    let mut a = -std::f64::consts::FRAC_PI_2;
    for (i, (&v, l)) in values.iter().zip(labels).enumerate() {
        let sweep = v / total * std::f64::consts::TAU;
        if v > 0.0 {
            let col = th.palette.color(i);
            s.push(
                crate::core::scene::Shape::Wedge { c, r, a0: a, a1: a + sweep },
                Some(Stroke::new(th.background, 1.5)),
                Some(col),
            );
            let mid = a + sweep / 2.0;
            let pct = v / total * 100.0;
            if pct >= 3.0 {
                let p = Point::new(c.x + (r + 14.0) * mid.cos(), c.y + (r + 14.0) * mid.sin() + th.tick_size * 0.35);
                let anchor = if mid.cos() > 0.2 {
                    Anchor::Start
                } else if mid.cos() < -0.2 {
                    Anchor::End
                } else {
                    Anchor::Middle
                };
                s.text(p, format!("{pct:.1}%"), TextStyle::new(th.tick_size, th.ink).anchored(anchor));
            }
            // Hit region: bounding box of the wedge centroid.
            let cp = Point::new(c.x + r * 0.6 * mid.cos(), c.y + r * 0.6 * mid.sin());
            s.target(Region::Point(cp, r * 0.35), 0, i, format!("{l}: {} ({pct:.1}%)", num(v)));
        }
        a += sweep;
    }
    let mut ly = top + 8.0;
    let lx = opts.width - legend_w;
    for (i, l) in labels.iter().enumerate() {
        s.rect(Rect::new(lx, ly, 14.0, 12.0), Some(th.palette.color(i)), None);
        s.text(Point::new(lx + 20.0, ly + 10.0), l.clone(), TextStyle::new(th.tick_size, th.ink));
        ly += th.tick_size * 1.2 + 4.0;
    }
    Ok(s)
}

// ------------------------------------------------------------------ ECDF

/// Empirical CDF step plots, one line per group.
pub fn ecdf_plot(groups: &[BoxGroup], title: &str, x_label: &str, opts: &PlotOptions) -> Scene {
    let sorted: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let mut v: Vec<f64> = g.data.iter().copied().filter(|x| x.is_finite()).collect();
            v.sort_by(f64::total_cmp);
            v
        })
        .collect();
    let xr = range_of(sorted.iter().flatten());
    if xr.is_none() {
        return empty_scene(opts, title, "No data to display");
    }
    let mut spec = base_spec(
        opts,
        linear_scale(xr, false, 0.05),
        Scale::Linear { min: -0.02, max: 1.02 },
        title,
        x_label,
        "Cumulative probability",
    );
    if groups.len() > 1 {
        let names: Vec<&str> = groups.iter().map(|g| g.label.as_str()).collect();
        spec.legend = series_legend(&names, &opts.theme, false, true);
    }
    let mut f = Frame::new(spec);
    f.begin_data();
    let (xl, xh) = f.x.domain();
    for (gi, v) in sorted.iter().enumerate() {
        if v.is_empty() {
            continue;
        }
        let n = v.len() as f64;
        let mut pts = vec![f.pt(xl, 0.0)];
        for (i, &x) in v.iter().enumerate() {
            pts.push(f.pt(x, i as f64 / n));
            pts.push(f.pt(x, (i + 1) as f64 / n));
        }
        pts.push(f.pt(xh, 1.0));
        f.scene.polyline_gaps(
            &pts,
            Stroke { color: opts.theme.palette.color(gi), width: 2.0, dash: opts.theme.palette.dash(gi) },
        );
        for (i, &x) in v.iter().enumerate().take(MAX_TARGETS / groups.len().max(1)) {
            f.scene.target(
                Region::Point(f.pt(x, (i + 1) as f64 / n), 5.0),
                gi,
                i,
                tip(&groups[gi].label, x_label, x, "F(x)", (i + 1) as f64 / n),
            );
        }
    }
    f.end_data();
    f.finish()
}

// ------------------------------------------------------------------ probability plot

struct Fit {
    a: f64,
    b: f64,
    s: f64,
    n: f64,
    tbar: f64,
    stt: f64,
}

fn ols_xt(t: &[f64], x: &[f64]) -> Option<Fit> {
    let n = t.len() as f64;
    if t.len() < 3 {
        return None;
    }
    let (tbar, xbar) = (t.iter().sum::<f64>() / n, x.iter().sum::<f64>() / n);
    let stt: f64 = t.iter().map(|v| (v - tbar).powi(2)).sum();
    if stt <= 0.0 {
        return None;
    }
    let b = t.iter().zip(x).map(|(a, c)| (a - tbar) * (c - xbar)).sum::<f64>() / stt;
    let a = xbar - b * tbar;
    let sse: f64 = t.iter().zip(x).map(|(ti, xi)| (xi - a - b * ti).powi(2)).sum();
    Some(Fit { a, b, s: (sse / (n - 2.0)).sqrt(), n, tbar, stt })
}

/// Probability plot of raw data on a normal, lognormal or Weibull probability axis with the
/// fitted line (least squares of the transformed data on the transformed plotting positions,
/// Benard median ranks) and a pointwise `level` confidence band from the regression.
pub fn probability_plot(
    data: &[f64],
    kind: ProbabilityKind,
    level: Option<f64>,
    title: &str,
    x_label: &str,
    opts: &PlotOptions,
) -> Scene {
    let mut v: Vec<f64> =
        data.iter().copied().filter(|x| x.is_finite() && (kind == ProbabilityKind::Normal || *x > 0.0)).collect();
    v.sort_by(f64::total_cmp);
    if v.len() < 3 {
        return empty_scene(opts, title, "At least 3 observations are needed");
    }
    let n = v.len() as f64;
    let p: Vec<f64> = (1..=v.len()).map(|i| (i as f64 - 0.3) / (n + 0.4)).collect();
    let tf = |p: f64| match kind {
        ProbabilityKind::Weibull => (-(1.0 - p).ln()).ln(),
        _ => norm_ppf(p),
    };
    let fx = |x: f64| if kind == ProbabilityKind::Normal { x } else { x.ln() };
    let inv = |u: f64| if kind == ProbabilityKind::Normal { u } else { u.exp() };
    let t: Vec<f64> = p.iter().map(|&q| tf(q)).collect();
    let xt: Vec<f64> = v.iter().map(|&x| fx(x)).collect();
    let xr = range_of(&v);
    let xs = match kind {
        ProbabilityKind::Normal => linear_scale(xr, false, 0.05),
        _ => {
            let (a, b) = xr.unwrap_or((1.0, 10.0));
            Scale::Log { min: a / 1.2, max: b * 1.2 }
        }
    };
    let ys = Scale::probability_for(kind, p[0] * 0.5, 1.0 - (1.0 - p[p.len() - 1]) * 0.5);
    let mut spec = base_spec(opts, xs, ys, title, x_label, "Percent");
    spec.no_grid = false;
    let mut f = Frame::new(spec);
    f.begin_data();
    if let Some(fit) = ols_xt(&t, &xt) {
        let (lo, hi) = f.y.domain();
        let (t0, t1) = (tf(lo), tf(hi));
        let grid: Vec<f64> = (0..=60).map(|i| t0 + (t1 - t0) * f64::from(i) / 60.0).collect();
        let pr = |u: f64| match kind {
            ProbabilityKind::Weibull => 1.0 - (-u.exp()).exp(),
            _ => 0.5 * (1.0 + openstats_math::special::erf::erf(u / std::f64::consts::SQRT_2)),
        };
        if let Some(l) = level {
            if let Ok(tq) = StudentT::new(fit.n - 2.0) {
                let q = tq.quantile(1.0 - (1.0 - l) / 2.0);
                let st = Stroke::new(opts.theme.palette.color(1), 1.2).dashed(Dash::Dashed);
                for sign in [-1.0, 1.0] {
                    let pts: Vec<Point> = grid
                        .iter()
                        .map(|&u| {
                            let se = fit.s * (1.0 / fit.n + (u - fit.tbar).powi(2) / fit.stt).sqrt();
                            Point::new(f.px(inv(fit.a + fit.b * u + sign * q * se)), f.py(pr(u)))
                        })
                        .collect();
                    f.scene.polyline_gaps(&pts, st);
                }
            }
        }
        let line: Vec<Point> = grid.iter().map(|&u| Point::new(f.px(inv(fit.a + fit.b * u)), f.py(pr(u)))).collect();
        f.scene.polyline_gaps(&line, Stroke::new(opts.theme.palette.color(1), 1.8));
    }
    let ms = marker_size(v.len());
    for (i, (&x, &q)) in v.iter().zip(&p).enumerate() {
        let pt = f.pt(x, q);
        f.scene.marker(pt, Marker::Circle, ms, opts.theme.palette.color(0), None);
        if i < MAX_TARGETS {
            f.scene.target(Region::Point(pt, ms), 0, i, tip("", x_label, x, "Percent", q * 100.0));
        }
    }
    f.end_data();
    f.finish()
}

/// `ChartKind::ProbabilityPlot` (x = data, y = normal scores): normal probability axis, LS line.
pub(crate) fn probability_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(s) = spec.series.first() else { return empty_scene(opts, &spec.title, "No data to display") };
    let pts: Vec<(f64, f64)> =
        s.x.iter().zip(&s.y).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
    if pts.len() < 3 {
        return empty_scene(opts, &spec.title, "At least 3 observations are needed");
    }
    let p: Vec<f64> =
        pts.iter().map(|q| 0.5 * (1.0 + openstats_math::special::erf::erf(q.1 / std::f64::consts::SQRT_2))).collect();
    let xr = range_of(pts.iter().map(|q| &q.0));
    let ys = Scale::probability_for(
        ProbabilityKind::Normal,
        p.iter().copied().fold(1.0, f64::min) * 0.5,
        1.0 - (1.0 - p.iter().copied().fold(0.0, f64::max)) * 0.5,
    );
    let mut fs = base_spec(opts, linear_scale(xr, false, 0.05), ys, &spec.title, &spec.x_label, "Percent");
    fs.no_grid = false;
    let mut f = Frame::new(fs);
    f.begin_data();
    let t: Vec<f64> = pts.iter().map(|q| q.1).collect();
    let x: Vec<f64> = pts.iter().map(|q| q.0).collect();
    if let Some(fit) = ols_xt(&t, &x) {
        let (lo, hi) = f.y.domain();
        let (t0, t1) = (norm_ppf(lo), norm_ppf(hi));
        let line: Vec<Point> = (0..=40)
            .map(|i| {
                let u = t0 + (t1 - t0) * f64::from(i) / 40.0;
                Point::new(
                    f.px(fit.a + fit.b * u),
                    f.py(0.5 * (1.0 + openstats_math::special::erf::erf(u / std::f64::consts::SQRT_2))),
                )
            })
            .collect();
        f.scene.polyline_gaps(&line, Stroke::new(opts.theme.palette.color(1), 1.8));
    }
    let ms = marker_size(pts.len());
    for (i, ((xv, _), q)) in pts.iter().zip(&p).enumerate() {
        let pt = f.pt(*xv, *q);
        f.scene.marker(pt, Marker::Circle, ms, opts.theme.palette.color(0), None);
        if i < MAX_TARGETS {
            f.scene.target(Region::Point(pt, ms), 0, i, tip("", &spec.x_label, *xv, "Percent", q * 100.0));
        }
    }
    f.end_data();
    f.finish()
}
