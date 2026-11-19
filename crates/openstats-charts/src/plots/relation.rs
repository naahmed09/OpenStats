//! Relationship and modelling plots: scatter plots with fits and bands, LOWESS, generic
//! line charts (main effects, interaction, control-chart style), cube, contour and surface.

use super::common::*;
use super::PlotOptions;
use crate::core::geom::Point;
use crate::core::hit::Region;
use crate::core::layout::{Frame, FrameSpec, LegendEntry};
use crate::core::scale::Scale;
use crate::core::scene::{Anchor, Scene, TextStyle};
use crate::core::style::{Color, Dash, Marker, Stroke};
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_stats::report::ChartSpec;

// ------------------------------------------------------------------ fitting helpers

/// A least-squares polynomial fit in a centred, scaled variable.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyFit {
    /// Coefficients in `u = (x - mean) / scale`, lowest order first.
    pub coef: Vec<f64>,
    /// Centre of x.
    pub mean: f64,
    /// Scale of x.
    pub scale: f64,
    /// Residual standard deviation.
    pub s: f64,
    /// Residual degrees of freedom.
    pub df: f64,
    inv: Vec<Vec<f64>>,
}

fn invert(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        inv.swap(c, p);
        let d = a[c][c];
        for j in 0..n {
            a[c][j] /= d;
            inv[c][j] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                if f != 0.0 {
                    for j in 0..n {
                        a[r][j] -= f * a[c][j];
                        inv[r][j] -= f * inv[c][j];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Least-squares polynomial of degree `deg` (1 = straight line). `None` when there are
/// too few distinct points or the design is singular.
pub fn poly_fit(x: &[f64], y: &[f64], deg: usize) -> Option<PolyFit> {
    let pts: Vec<(f64, f64)> =
        x.iter().zip(y).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).collect();
    let n = pts.len();
    let k = deg + 1;
    if deg == 0 || n <= k {
        return None;
    }
    let mean = pts.iter().map(|p| p.0).sum::<f64>() / n as f64;
    let scale = (pts.iter().map(|p| (p.0 - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if scale.is_nan() || scale <= 0.0 {
        return None;
    }
    let row = |x: f64| -> Vec<f64> {
        let u = (x - mean) / scale;
        (0..k).map(|j| u.powi(j as i32)).collect()
    };
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for &(a, b) in &pts {
        let r = row(a);
        for i in 0..k {
            xty[i] += r[i] * b;
            for j in 0..k {
                xtx[i][j] += r[i] * r[j];
            }
        }
    }
    let inv = invert(xtx)?;
    let coef: Vec<f64> = (0..k).map(|i| (0..k).map(|j| inv[i][j] * xty[j]).sum()).collect();
    let sse: f64 =
        pts.iter().map(|&(a, b)| (b - row(a).iter().zip(&coef).map(|(r, c)| r * c).sum::<f64>()).powi(2)).sum();
    let df = (n - k) as f64;
    Some(PolyFit { coef, mean, scale, s: (sse / df).sqrt(), df, inv })
}

impl PolyFit {
    fn row(&self, x: f64) -> Vec<f64> {
        let u = (x - self.mean) / self.scale;
        (0..self.coef.len()).map(|j| u.powi(j as i32)).collect()
    }

    /// Fitted value at `x`.
    pub fn predict(&self, x: f64) -> f64 {
        self.row(x).iter().zip(&self.coef).map(|(r, c)| r * c).sum()
    }

    /// Standard error of the mean response at `x`.
    pub fn se_mean(&self, x: f64) -> f64 {
        let r = self.row(x);
        let mut h = 0.0;
        for i in 0..r.len() {
            for j in 0..r.len() {
                h += r[i] * self.inv[i][j] * r[j];
            }
        }
        self.s * h.max(0.0).sqrt()
    }

    /// Standard error of a new observation at `x`.
    pub fn se_pred(&self, x: f64) -> f64 {
        let m = self.se_mean(x);
        (self.s * self.s + m * m).sqrt()
    }
}

/// LOWESS (locally weighted linear regression, tricube weights, `iters` robustness
/// iterations). Returns `(sorted x, fitted y)`; `span` is the fraction of points used.
pub fn lowess(x: &[f64], y: &[f64], span: f64, iters: usize) -> (Vec<f64>, Vec<f64>) {
    let mut pts: Vec<(f64, f64)> =
        x.iter().zip(y).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = pts.len();
    if n < 3 {
        return (pts.iter().map(|p| p.0).collect(), pts.iter().map(|p| p.1).collect());
    }
    let k = ((span.clamp(0.05, 1.0) * n as f64).ceil() as usize).clamp(2, n);
    let mut rw = vec![1.0; n];
    let mut fit = vec![0.0; n];
    for it in 0..=iters {
        for i in 0..n {
            let xi = pts[i].0;
            let mut d: Vec<f64> = pts.iter().map(|p| (p.0 - xi).abs()).collect();
            d.sort_by(f64::total_cmp);
            let h = d[k - 1].max(1e-12);
            let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (j, p) in pts.iter().enumerate() {
                let u = (p.0 - xi).abs() / h;
                if u < 1.0 {
                    let w = (1.0 - u.powi(3)).powi(3) * rw[j];
                    sw += w;
                    swx += w * p.0;
                    swy += w * p.1;
                    swxx += w * p.0 * p.0;
                    swxy += w * p.0 * p.1;
                }
            }
            fit[i] = if sw <= 0.0 {
                pts[i].1
            } else {
                let det = sw * swxx - swx * swx;
                if det.abs() < 1e-12 * (sw * swxx).abs().max(1e-300) {
                    swy / sw
                } else {
                    let b = (sw * swxy - swx * swy) / det;
                    let a = (swy - b * swx) / sw;
                    a + b * xi
                }
            };
        }
        if it < iters {
            let mut res: Vec<f64> = pts.iter().zip(&fit).map(|(p, f)| (p.1 - f).abs()).collect();
            res.sort_by(f64::total_cmp);
            let m = 0.5 * (res[(n - 1) / 2] + res[n / 2]);
            if m <= 1e-12 {
                break;
            }
            for j in 0..n {
                let u = (pts[j].1 - fit[j]).abs() / (6.0 * m);
                rw[j] = if u < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
            }
        }
    }
    (pts.iter().map(|p| p.0).collect(), fit)
}

// ------------------------------------------------------------------ scatter

/// Fit drawn over a scatter plot.
#[derive(Debug, Clone, PartialEq)]
pub enum Fit {
    /// No fit.
    None,
    /// Least-squares polynomial (degree 1 = line, 2 = quadratic, 3 = cubic).
    Poly(usize),
    /// LOWESS with the given span.
    Lowess(f64),
}

/// Confidence and prediction bands of a polynomial fit.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FitBands {
    /// Confidence level of the mean response band (e.g. 0.95).
    pub confidence: Option<f64>,
    /// Confidence level of the prediction band.
    pub prediction: Option<f64>,
}

/// Points of a scatter plot, optionally in groups.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterData {
    /// x values.
    pub x: Vec<f64>,
    /// y values.
    pub y: Vec<f64>,
    /// Group index of each point (empty = one group).
    pub groups: Vec<usize>,
    /// Group names.
    pub group_names: Vec<String>,
}

impl ScatterData {
    /// A single group.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> ScatterData {
        ScatterData { x, y, groups: Vec::new(), group_names: Vec::new() }
    }
}

/// Scatter plot with optional fit, bands and groups (marker/colour per group).
pub fn scatter_plot(
    data: &ScatterData,
    fit: &Fit,
    bands: FitBands,
    title: &str,
    x_label: &str,
    y_label: &str,
    opts: &PlotOptions,
) -> Scene {
    let xr = range_of(&data.x);
    let yr = range_of(&data.y);
    if xr.is_none() || yr.is_none() || data.x.len() != data.y.len() {
        return empty_scene(opts, title, "No data to display");
    }
    // Compute the fit first so bands can widen the y range.
    let (xlo, xhi) = xr.unwrap_or((0.0, 1.0));
    let grid: Vec<f64> = (0..=100).map(|i| xlo + (xhi - xlo) * f64::from(i) / 100.0).collect();
    let mut line: Vec<(f64, f64)> = Vec::new();
    let mut ci: Vec<(f64, f64, f64)> = Vec::new();
    let mut pi: Vec<(f64, f64, f64)> = Vec::new();
    match fit {
        Fit::None => {}
        Fit::Poly(d) => {
            if let Some(pf) = poly_fit(&data.x, &data.y, *d) {
                line = grid.iter().map(|&g| (g, pf.predict(g))).collect();
                if let Ok(t) = StudentT::new(pf.df) {
                    if let Some(l) = bands.confidence {
                        let q = t.quantile(1.0 - (1.0 - l) / 2.0);
                        ci = grid
                            .iter()
                            .map(|&g| (g, pf.predict(g) - q * pf.se_mean(g), pf.predict(g) + q * pf.se_mean(g)))
                            .collect();
                    }
                    if let Some(l) = bands.prediction {
                        let q = t.quantile(1.0 - (1.0 - l) / 2.0);
                        pi = grid
                            .iter()
                            .map(|&g| (g, pf.predict(g) - q * pf.se_pred(g), pf.predict(g) + q * pf.se_pred(g)))
                            .collect();
                    }
                }
            }
        }
        Fit::Lowess(span) => {
            let (lx, ly) = lowess(&data.x, &data.y, *span, 2);
            line = lx.into_iter().zip(ly).collect();
        }
    }
    let ymin = line
        .iter()
        .map(|p| p.1)
        .chain(ci.iter().map(|p| p.1))
        .chain(pi.iter().map(|p| p.1))
        .chain(data.y.iter().copied())
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let ymax = line
        .iter()
        .map(|p| p.1)
        .chain(ci.iter().map(|p| p.2))
        .chain(pi.iter().map(|p| p.2))
        .chain(data.y.iter().copied())
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let ng = data.groups.iter().copied().max().map_or(1, |m| m + 1);
    let mut spec = FrameSpec::new(
        opts.width,
        opts.height,
        with_view(linear_scale(xr, false, 0.05), opts.viewport.x),
        with_view(linear_scale(Some((ymin, ymax)), false, 0.05), opts.viewport.y),
        &opts.theme,
    );
    spec.title = title.into();
    spec.x_label = x_label.into();
    spec.y_label = y_label.into();
    let name_of = |g: usize| data.group_names.get(g).cloned().unwrap_or_else(|| format!("Group {}", g + 1));
    if ng > 1 {
        spec.legend = (0..ng)
            .map(|g| LegendEntry {
                name: name_of(g),
                color: opts.theme.palette.color(g),
                marker: Some(marker_of(&opts.theme, g)),
                line: None,
            })
            .collect();
    }
    let mut f = Frame::new(spec);
    f.begin_data();
    let band = |f: &mut Frame, b: &[(f64, f64, f64)], color: Color, dash: Dash| {
        if b.len() < 2 {
            return;
        }
        let mut poly: Vec<Point> = b.iter().map(|p| f.pt(p.0, p.1)).collect();
        poly.extend(b.iter().rev().map(|p| f.pt(p.0, p.2)));
        f.scene.polygon(poly, Some(color.with_alpha(0.12)), None);
        let lo: Vec<Point> = b.iter().map(|p| f.pt(p.0, p.1)).collect();
        let hi: Vec<Point> = b.iter().map(|p| f.pt(p.0, p.2)).collect();
        let st = Stroke::new(color, 1.0).dashed(dash);
        f.scene.polyline_gaps(&lo, st);
        f.scene.polyline_gaps(&hi, st);
    };
    let accent = opts.theme.palette.color(1);
    band(&mut f, &pi, accent, Dash::Dotted);
    band(&mut f, &ci, accent, Dash::Dashed);
    if !line.is_empty() {
        let pts: Vec<Point> = line.iter().map(|p| f.pt(p.0, p.1)).collect();
        f.scene.polyline_gaps(&pts, Stroke::new(accent, 2.0));
    }
    let ms = marker_size(data.x.len());
    for (i, (&x, &y)) in data.x.iter().zip(&data.y).enumerate() {
        if !(x.is_finite() && y.is_finite()) {
            continue;
        }
        let g = data.groups.get(i).copied().unwrap_or(0);
        let p = f.pt(x, y);
        f.scene.marker(p, marker_of(&opts.theme, g), ms, opts.theme.palette.color(g), None);
        if i < MAX_TARGETS {
            let gname = if ng > 1 { name_of(g) } else { String::new() };
            f.scene.target(Region::Point(p, ms.max(5.0)), g, i, tip(&gname, x_label, x, y_label, y));
        }
    }
    f.end_data();
    f.finish()
}

/// Line style of a scatter-spec series that is a curve rather than a point cloud: the fitted line
/// (`Fit`), its confidence band (`Lower CI` / `Upper CI`) or its prediction band (`Lower PI` /
/// `Upper PI`).
fn fit_curve_dash(name: &str) -> Option<Dash> {
    match name {
        "Fit" => Some(Dash::Solid),
        n if n.starts_with("Fit: ") => Some(Dash::Solid),
        "Lower CI" | "Upper CI" => Some(Dash::Dashed),
        "Lower PI" | "Upper PI" => Some(Dash::Dotted),
        _ => None,
    }
}

pub(crate) fn scatter_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    if spec.title.starts_with("Cube Plot") && !spec.categories.is_empty() {
        return cube_scene(spec, opts);
    }
    let xr = range_of(spec.series.iter().flat_map(|s| s.x.iter()));
    let yr = union(
        range_of(spec.series.iter().flat_map(|s| s.y.iter())),
        range_of(spec.reference_lines.iter().filter(|r| r.axis == openstats_stats::report::Axis::Y).map(|r| &r.value)),
    );
    if xr.is_none() || yr.is_none() {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let mut fs = FrameSpec::new(
        opts.width,
        opts.height,
        with_view(linear_scale(xr, false, 0.05), opts.viewport.x),
        with_view(linear_scale(yr, false, 0.08), opts.viewport.y),
        &opts.theme,
    );
    fs.title = spec.title.clone();
    fs.x_label = spec.x_label.clone();
    fs.y_label = spec.y_label.clone();
    if spec.series.len() > 1 {
        let names: Vec<&str> = spec.series.iter().map(|s| s.name.as_str()).collect();
        fs.legend = series_legend(&names, &opts.theme, true, false);
    }
    let mut f = Frame::new(fs);
    f.begin_data();
    for (si, s) in spec.series.iter().enumerate() {
        // A fitted line and its confidence / prediction bands are drawn as curves, not markers.
        if let Some(dash) = fit_curve_dash(&s.name) {
            let pts: Vec<Point> = s.x.iter().zip(&s.y).map(|(&x, &y)| f.pt(x, y)).collect();
            // `Fit: <group>` takes the colour of the point series named `<group>`.
            let ci =
                s.name.strip_prefix("Fit: ").and_then(|g| spec.series.iter().position(|o| o.name == g)).unwrap_or(si);
            f.scene.polyline_gaps(&pts, Stroke::new(opts.theme.palette.color(ci), 1.5).dashed(dash));
            continue;
        }
        let ms = marker_size(s.x.len());
        for (i, (&x, &y)) in s.x.iter().zip(&s.y).enumerate() {
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            let p = f.pt(x, y);
            f.scene.marker(p, marker_of(&opts.theme, si), ms, opts.theme.palette.color(si), None);
            if i < MAX_TARGETS / spec.series.len().max(1) {
                f.scene.target(
                    Region::Point(p, ms.max(5.0)),
                    si,
                    i,
                    tip(if spec.series.len() > 1 { &s.name } else { "" }, &spec.x_label, x, &spec.y_label, y),
                );
            }
        }
    }
    reference_lines(&mut f, &spec.reference_lines);
    f.end_data();
    f.finish()
}

// ------------------------------------------------------------------ cube

fn cube_scene(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let th = &opts.theme;
    let mut s = Scene::new(opts.width, opts.height, th.background);
    s.title = spec.title.clone();
    s.text(
        Point::new(opts.width / 2.0, 8.0 + th.title_size * 0.95),
        spec.title.clone(),
        TextStyle::new(th.title_size, th.ink).anchored(Anchor::Middle).bold(),
    );
    let side = (opts.width.min(opts.height) * 0.46).max(40.0);
    let off = side * 0.38;
    let ox = (opts.width - side - off) / 2.0;
    let oy = (opts.height + th.title_size * 1.5 - side - off) / 2.0 + off;
    // corner (a, b, c) with a, b, c in {-1, 1}
    let at = |a: f64, b: f64, c: f64| {
        Point::new(
            ox + (a + 1.0) / 2.0 * side + (c + 1.0) / 2.0 * off,
            oy + (1.0 - b) / 2.0 * side - (c + 1.0) / 2.0 * off,
        )
    };
    let ink = Stroke::new(th.ink, 1.2);
    for c in [-1.0, 1.0] {
        let q = [at(-1.0, -1.0, c), at(1.0, -1.0, c), at(1.0, 1.0, c), at(-1.0, 1.0, c)];
        for i in 0..4 {
            s.line(q[i], q[(i + 1) % 4], ink);
        }
    }
    for a in [-1.0, 1.0] {
        for b in [-1.0, 1.0] {
            s.line(at(a, b, -1.0), at(a, b, 1.0), ink);
        }
    }
    let mut k = 0;
    for (si, ser) in spec.series.iter().enumerate() {
        let c = if ser.name.ends_with("-1") { -1.0 } else { 1.0 };
        for i in 0..ser.x.len() {
            let p = at(ser.x[i], ser.y[i], c);
            s.marker(p, Marker::Circle, 6.0, th.palette.color(0), None);
            let cat = spec.categories.get(k).cloned().unwrap_or_default();
            let val = cat.rsplit(": ").next().unwrap_or("").to_string();
            let right = ser.x[i] > 0.0;
            s.text(
                Point::new(p.x + if right { 8.0 } else { -8.0 }, p.y - 6.0),
                val,
                TextStyle::new(th.tick_size, th.ink).anchored(if right { Anchor::Start } else { Anchor::End }),
            );
            s.target(Region::Point(p, 8.0), si, i, cat);
            k += 1;
        }
    }
    let lab = TextStyle::new(th.label_size, th.ink).anchored(Anchor::Middle);
    s.text(Point::new(ox + side / 2.0, oy + side + th.label_size + 6.0), format!("{} →", spec.x_label), lab);
    s.text(
        Point::new(ox + side + off + 8.0, oy - side / 2.0),
        format!("{} ↑", spec.y_label),
        TextStyle::new(th.label_size, th.ink),
    );
    s
}

// ------------------------------------------------------------------ line charts

/// Extras for control-chart style line plots.
#[derive(Debug, Clone, Default)]
pub struct LineExtras {
    /// Zone shading: per-point centre and sigma (draws ±1σ, ±2σ, ±3σ bands).
    pub zones: Option<(Vec<f64>, Vec<f64>)>,
    /// Indices (0-based) where a new stage begins (a vertical divider is drawn).
    pub stage_breaks: Vec<usize>,
    /// Stage labels, one per stage (optional).
    pub stage_labels: Vec<String>,
    /// Red labels placed next to flagged points: `(point index, text)`.
    pub point_labels: Vec<(usize, String)>,
    /// Write the limit values at the right end of the limit lines.
    pub end_labels: bool,
    /// The x values are Excel serial days: draw a calendar axis.
    pub time_axis: bool,
}

#[derive(PartialEq)]
enum Role {
    Data,
    Limit,
    Center,
    Flag,
    /// A thin background curve (an individual conditional expectation line).
    Faint,
}

fn role(name: &str) -> Role {
    match name {
        "UCL" | "LCL" | "Upper limit" | "Lower limit" => Role::Limit,
        "Center" => Role::Center,
        "Flagged" => Role::Flag,
        n if n.starts_with("ICE ") => Role::Faint,
        _ => Role::Data,
    }
}

pub(crate) fn line_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    line_scene(spec, opts, &LineExtras::default())
}

pub(crate) fn timeline_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    line_scene(spec, opts, &LineExtras { time_axis: true, ..LineExtras::default() })
}

/// Draw a [`ChartSpec`] of kind `Line` (limits stepped, flagged points in the alert colour).
pub fn line_scene(spec: &ChartSpec, opts: &PlotOptions, ex: &LineExtras) -> Scene {
    let finite = |s: &openstats_stats::report::Series| s.y.iter().any(|v| v.is_finite());
    if !spec.series.iter().any(finite) {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let xr = range_of(spec.series.iter().flat_map(|s| s.x.iter()));
    let mut yr = range_of(spec.series.iter().flat_map(|s| s.y.iter()));
    yr = union(
        yr,
        range_of(spec.reference_lines.iter().filter(|r| r.axis == openstats_stats::report::Axis::Y).map(|r| &r.value)),
    );
    if let Some((c, sg)) = &ex.zones {
        let ext: Vec<f64> = c.iter().zip(sg).flat_map(|(c, s)| [c - 3.0 * s, c + 3.0 * s]).collect();
        yr = union(yr, range_of(&ext));
    }
    let pad = if ex.end_labels { 0.08 } else { 0.06 };
    let xscale = match (ex.time_axis, xr) {
        (true, Some((lo, hi))) => {
            let p = ((hi - lo) * 0.03).max(0.5);
            Scale::Time { min: lo - p, max: hi + p }
        }
        _ => linear_scale(xr, false, 0.03),
    };
    let mut fs = FrameSpec::new(
        opts.width,
        opts.height,
        with_view(xscale, opts.viewport.x),
        with_view(linear_scale(yr, false, pad), opts.viewport.y),
        &opts.theme,
    );
    fs.title = spec.title.clone();
    fs.x_label = spec.x_label.clone();
    fs.y_label = spec.y_label.clone();
    if ex.end_labels {
        fs.extra_right = 62.0;
    }
    let data_names: Vec<&str> =
        spec.series.iter().filter(|s| role(&s.name) == Role::Data).map(|s| s.name.as_str()).collect();
    if data_names.len() > 1 {
        fs.legend = series_legend(&data_names, &opts.theme, true, true);
    }
    let mut f = Frame::new(fs);
    let th = opts.theme.clone();
    f.begin_data();
    if let Some((c, sg)) = &ex.zones {
        let xs: Vec<f64> =
            spec.series.iter().find(|s| role(&s.name) == Role::Data).map(|s| s.x.clone()).unwrap_or_default();
        let n = xs.len().min(c.len()).min(sg.len());
        if n >= 2 {
            let shades = [(3.0, 2.0, 0xFADBD8u32), (2.0, 1.0, 0xFCF3CF), (1.0, 0.0, 0xD5F5E3)];
            for (hi, lo, col) in shades {
                for sign in [1.0, -1.0] {
                    let top: Vec<Point> = (0..n).map(|i| f.pt(xs[i], c[i] + sign * hi * sg[i])).collect();
                    let bot: Vec<Point> = (0..n).map(|i| f.pt(xs[i], c[i] + sign * lo * sg[i])).collect();
                    let mut poly = top;
                    poly.extend(bot.into_iter().rev());
                    if poly.iter().all(|p| p.is_finite()) {
                        f.scene.polygon(
                            poly,
                            Some(Color::hex(col).with_alpha(if th.background.r < 100 { 0.25 } else { 0.8 })),
                            None,
                        );
                    }
                }
            }
        }
    }
    for (i, &b) in ex.stage_breaks.iter().enumerate() {
        let Some(s0) = spec.series.first() else { break };
        if b > 0 && b < s0.x.len() {
            let x = f.px((s0.x[b - 1] + s0.x[b]) / 2.0);
            f.scene.line(
                Point::new(x, f.plot.y),
                Point::new(x, f.plot.bottom()),
                Stroke::new(th.ink.lighten(0.4), 1.0).dashed(Dash::Dashed),
            );
            if let Some(l) = ex.stage_labels.get(i + 1) {
                f.scene.text(
                    Point::new(x + 3.0, f.plot.y + th.tick_size + 2.0),
                    l.clone(),
                    TextStyle::new(th.tick_size, th.ink),
                );
            }
        }
    }
    let mut di = 0;
    for (si, s) in spec.series.iter().enumerate() {
        match role(&s.name) {
            Role::Limit | Role::Center => {
                let is_c = role(&s.name) == Role::Center;
                let color = if is_c { th.center } else { th.limit };
                // Stepped line so staged limits jump at the stage change.
                let mut pts: Vec<Point> = Vec::new();
                for i in 0..s.x.len().min(s.y.len()) {
                    if i > 0 && s.y[i] != s.y[i - 1] && s.y[i].is_finite() && s.y[i - 1].is_finite() {
                        let xm = (s.x[i - 1] + s.x[i]) / 2.0;
                        pts.push(f.pt(xm, s.y[i - 1]));
                        pts.push(f.pt(xm, s.y[i]));
                    }
                    pts.push(f.pt(s.x[i], s.y[i]));
                }
                f.scene
                    .polyline_gaps(&pts, Stroke::new(color, 1.3).dashed(if is_c { Dash::Solid } else { Dash::Dashed }));
                if ex.end_labels {
                    if let (Some(&xl), Some(&yl)) = (s.x.last(), s.y.iter().rev().find(|v| v.is_finite())) {
                        let p = f.pt(xl, yl);
                        let label = format!("{}={}", s.name, num(yl));
                        f.scene.text(
                            Point::new(f.plot.right() + 4.0, p.y + th.tick_size * 0.35),
                            label,
                            TextStyle::new(th.tick_size, color),
                        );
                    }
                }
            }
            Role::Faint => {
                let pts: Vec<Point> = s.x.iter().zip(&s.y).map(|(&x, &y)| f.pt(x, y)).collect();
                f.scene.polyline_gaps(&pts, Stroke::new(th.ink.lighten(0.65), 0.8));
            }
            Role::Flag => {
                for (i, (&x, &y)) in s.x.iter().zip(&s.y).enumerate() {
                    if x.is_finite() && y.is_finite() {
                        let p = f.pt(x, y);
                        f.scene.marker(p, Marker::Square, 7.0, th.alert, None);
                        f.scene.target(
                            Region::Point(p, 7.0),
                            si,
                            i,
                            tip("Flagged", &spec.x_label, x, &spec.y_label, y),
                        );
                    }
                }
            }
            Role::Data => {
                let col = th.palette.color(di);
                let pts: Vec<Point> = s.x.iter().zip(&s.y).map(|(&x, &y)| f.pt(x, y)).collect();
                f.scene.polyline_gaps(
                    &pts,
                    Stroke {
                        color: col,
                        width: 1.8,
                        dash: if data_names.len() > 1 { th.palette.dash(di) } else { Dash::Solid },
                    },
                );
                let ms = if s.x.len() > 120 { 0.0 } else { 5.5 };
                for (i, (&x, &y)) in s.x.iter().zip(&s.y).enumerate() {
                    if x.is_finite() && y.is_finite() {
                        let p = f.pt(x, y);
                        if ms > 0.0 {
                            f.scene.marker(p, marker_of(&th, di), ms, col, None);
                        }
                        if i < MAX_TARGETS / spec.series.len().max(1) {
                            f.scene.target(
                                Region::Point(p, 6.0),
                                si,
                                i,
                                tip(
                                    if data_names.len() > 1 { &s.name } else { "" },
                                    &spec.x_label,
                                    x,
                                    &spec.y_label,
                                    y,
                                ),
                            );
                        }
                    }
                }
                di += 1;
            }
        }
    }
    if let Some(d0) = spec.series.iter().find(|s| role(&s.name) == Role::Data) {
        for (i, text) in &ex.point_labels {
            if let (Some(&x), Some(&y)) = (d0.x.get(*i), d0.y.get(*i)) {
                if x.is_finite() && y.is_finite() {
                    let p = f.pt(x, y);
                    f.scene.text(
                        Point::new(p.x + 5.0, p.y - 6.0),
                        text.clone(),
                        TextStyle::new(th.tick_size, th.alert).bold(),
                    );
                }
            }
        }
    }
    reference_lines(&mut f, &spec.reference_lines);
    f.end_data();
    f.finish()
}

// ------------------------------------------------------------------ contour

pub(crate) fn contour_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let xr = range_of(spec.series.iter().flat_map(|s| s.x.iter()));
    let yr = range_of(spec.series.iter().flat_map(|s| s.y.iter()));
    if xr.is_none() || yr.is_none() {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let mut fs = FrameSpec::new(
        opts.width,
        opts.height,
        with_view(linear_scale(xr, false, 0.02), opts.viewport.x),
        with_view(linear_scale(yr, false, 0.02), opts.viewport.y),
        &opts.theme,
    );
    fs.title = spec.title.clone();
    fs.x_label = spec.x_label.clone();
    fs.y_label = spec.y_label.clone();
    let n = spec.series.len();
    let colour = |i: usize| ramp(if n > 1 { i as f64 / (n - 1) as f64 } else { 0.5 });
    fs.legend = spec
        .series
        .iter()
        .enumerate()
        .map(|(i, s)| LegendEntry {
            name: s.name.parse::<f64>().map_or_else(|_| s.name.clone(), num),
            color: colour(i),
            marker: None,
            line: Some(Dash::Solid),
        })
        .collect();
    let mut f = Frame::new(fs);
    f.begin_data();
    for (i, s) in spec.series.iter().enumerate() {
        let pts: Vec<Point> = s.x.iter().zip(&s.y).map(|(&x, &y)| f.pt(x, y)).collect();
        // Segments are pairs separated by NaN; polyline_gaps splits at the gaps.
        f.scene.polyline_gaps(&pts, Stroke::new(colour(i), 1.8));
    }
    reference_lines(&mut f, &spec.reference_lines);
    f.end_data();
    f.finish()
}

// ------------------------------------------------------------------ surface

pub(crate) fn surface_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    crate::render_3d::surface_spec_scene(spec, opts)
}
