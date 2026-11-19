//! Renderers for the chart kinds added with the Graph menu: pie, empirical CDF, individual
//! values, areas, bubbles, heat maps, dotplots, matrix and marginal plots, stacked bars, probability
//! plots of raw data and three-dimensional scatter plots.

use super::basic::{
    ecdf_plot, histogram_from_data, individual_value_plot, pie_chart, probability_plot, BoxGroup, HistogramOverlay,
};
use super::common::*;
use super::quality::panels_scene;
use super::relation::{scatter_plot, Fit, FitBands, ScatterData};
use super::PlotOptions;
use crate::core::geom::{Point, Rect};
use crate::core::hit::Region;
use crate::core::layout::{Frame, FrameSpec};
use crate::core::scale::{ProbabilityKind, Scale};
use crate::core::scene::{Anchor, Scene, Shape, TextStyle};
use crate::core::style::{Marker, Stroke};
use crate::render_3d::{Camera, Vec3, ViewTransform};
use openstats_stats::report::{ChartKind, ChartSpec, Series};

fn frame_spec(opts: &PlotOptions, x: Scale, y: Scale, title: &str, xl: &str, yl: &str) -> FrameSpec {
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

fn groups_of(spec: &ChartSpec, use_y: bool) -> Vec<BoxGroup> {
    spec.series
        .iter()
        .map(|s| BoxGroup { label: s.name.clone(), data: if use_y { s.y.clone() } else { s.x.clone() } })
        .collect()
}

/// Pie chart from `categories` and `series[0].y`.
pub(crate) fn pie_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(s) = spec.series.first() else { return empty_scene(opts, &spec.title, "No data to display") };
    match pie_chart(&spec.categories, &s.y, &spec.title, opts) {
        Ok(sc) => sc,
        Err(e) => empty_scene(opts, &spec.title, &e),
    }
}

pub(crate) fn ecdf_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    ecdf_plot(&groups_of(spec, false), &spec.title, &spec.x_label, opts)
}

pub(crate) fn individual_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    individual_value_plot(&groups_of(spec, true), &spec.title, &spec.x_label, &spec.y_label, opts)
}

pub(crate) fn data_probability_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(s) = spec.series.first() else { return empty_scene(opts, &spec.title, "No data to display") };
    let kind = match s.name.to_ascii_lowercase().as_str() {
        "lognormal" => ProbabilityKind::Lognormal,
        "weibull" => ProbabilityKind::Weibull,
        _ => ProbabilityKind::Normal,
    };
    probability_plot(&s.x, kind, Some(0.95), &spec.title, &spec.x_label, opts)
}

/// Stacked bars: series are stacked in order (non-negative heights).
pub(crate) fn stacked_bar_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let n = spec.categories.len().max(spec.series.iter().map(|s| s.y.len()).max().unwrap_or(0));
    if n == 0 || spec.series.is_empty() {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let mut totals = vec![0.0f64; n];
    for s in &spec.series {
        for (i, &v) in s.y.iter().enumerate() {
            if v.is_finite() {
                totals[i] += v.max(0.0);
            }
        }
    }
    let top = totals.iter().copied().fold(0.0, f64::max);
    let mut fs = frame_spec(
        opts,
        Scale::Band { n },
        linear_scale(Some((0.0, top.max(1e-9))), true, 0.06),
        &spec.title,
        &spec.x_label,
        &spec.y_label,
    );
    fs.x_categories = (0..n).map(|i| spec.categories.get(i).cloned().unwrap_or_else(|| format!("{}", i + 1))).collect();
    if spec.series.len() > 1 {
        let names: Vec<&str> = spec.series.iter().map(|s| s.name.as_str()).collect();
        fs.legend = series_legend(&names, &opts.theme, false, false);
    }
    let mut f = Frame::new(fs);
    f.begin_data();
    let slot = f.plot.w / n as f64;
    let bw = slot * 0.7;
    let mut base = vec![0.0f64; n];
    for (si, s) in spec.series.iter().enumerate() {
        let col = opts.theme.palette.color(si);
        for (i, &v) in s.y.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            let (y0, y1) = (f.py(base[i]), f.py(base[i] + v));
            let r = Rect::new(f.px(i as f64) - bw / 2.0, y1.min(y0), bw, (y0 - y1).abs());
            f.scene.rect(r, Some(col), Some(Stroke::new(opts.theme.background, 0.8)));
            f.scene.target(Region::Rect(r), si, i, format!("{}: {}", s.name, num(v)));
            base[i] += v;
        }
    }
    f.end_data();
    f.finish()
}

/// Filled areas, each series drawn in order down to zero.
pub(crate) fn area_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let xr = range_of(spec.series.iter().flat_map(|s| s.x.iter()));
    let yr = range_of(spec.series.iter().flat_map(|s| s.y.iter()));
    if xr.is_none() || yr.is_none() {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let mut fs = frame_spec(
        opts,
        linear_scale(xr, false, 0.02),
        linear_scale(yr, true, 0.06),
        &spec.title,
        &spec.x_label,
        &spec.y_label,
    );
    if spec.series.len() > 1 {
        let names: Vec<&str> = spec.series.iter().map(|s| s.name.as_str()).collect();
        fs.legend = series_legend(&names, &opts.theme, false, true);
    }
    let mut f = Frame::new(fs);
    f.begin_data();
    for (si, s) in spec.series.iter().enumerate() {
        let col = opts.theme.palette.color(si);
        let pts: Vec<(f64, f64)> =
            s.x.iter().zip(&s.y).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
        if pts.len() < 2 {
            continue;
        }
        let mut poly: Vec<Point> = pts.iter().map(|&(x, y)| f.pt(x, y)).collect();
        let line = poly.clone();
        poly.push(f.pt(pts[pts.len() - 1].0, 0.0));
        poly.push(f.pt(pts[0].0, 0.0));
        f.scene.polygon(poly, Some(col.with_alpha(0.55)), None);
        f.scene.polyline_gaps(&line, Stroke::new(col, 1.6));
        for (i, &(x, y)) in pts.iter().enumerate() {
            if pts.len() <= 200 {
                let p = f.pt(x, y);
                f.scene.target(Region::Point(p, 6.0), si, i, tip(&s.name, &spec.x_label, x, &spec.y_label, y));
            }
        }
    }
    f.end_data();
    f.finish()
}

/// Bubble plot: marker area proportional to the `Size` series.
pub(crate) fn bubble_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(pts) = spec.series.iter().find(|s| !s.name.eq_ignore_ascii_case("size")) else {
        return empty_scene(opts, &spec.title, "No data to display");
    };
    let sizes: Vec<f64> = spec
        .series
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case("size"))
        .map(|s| s.y.clone())
        .unwrap_or_else(|| vec![1.0; pts.x.len()]);
    let (xr, yr) = (range_of(&pts.x), range_of(&pts.y));
    if xr.is_none() || yr.is_none() {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let mut f = Frame::new(frame_spec(
        opts,
        linear_scale(xr, false, 0.08),
        linear_scale(yr, false, 0.08),
        &spec.title,
        &spec.x_label,
        &spec.y_label,
    ));
    f.begin_data();
    let smax = sizes.iter().copied().filter(|v| v.is_finite() && *v > 0.0).fold(0.0, f64::max).max(1e-300);
    let col = opts.theme.palette.color(0);
    // Largest bubbles first so small ones stay visible.
    let mut order: Vec<usize> = (0..pts.x.len()).collect();
    order.sort_by(|&a, &b| sizes.get(b).copied().unwrap_or(0.0).total_cmp(&sizes.get(a).copied().unwrap_or(0.0)));
    for i in order {
        let (x, y) = (pts.x[i], pts.y[i]);
        let sz = sizes.get(i).copied().unwrap_or(f64::NAN);
        if !(x.is_finite() && y.is_finite() && sz.is_finite() && sz > 0.0) {
            continue;
        }
        let r = 3.0 + 22.0 * (sz / smax).sqrt();
        let c = f.pt(x, y);
        f.scene.push(Shape::Circle { c, r }, Some(Stroke::new(col, 1.0)), Some(col.with_alpha(0.45)));
        f.scene.target(
            Region::Point(c, r),
            0,
            i,
            format!("{}: {}, {}: {}, size {}", spec.x_label, num(x), spec.y_label, num(y), num(sz)),
        );
    }
    f.end_data();
    f.finish()
}

/// Heat map from the surface layout (one series per row).
pub(crate) fn heatmap_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let rows: Vec<&Series> = spec.series.iter().filter(|s| !s.x.is_empty()).collect();
    let ncol = rows.iter().map(|s| s.x.len()).max().unwrap_or(0);
    if rows.is_empty() || ncol == 0 {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let ys: Vec<f64> = rows.iter().enumerate().map(|(i, s)| s.name.parse::<f64>().unwrap_or(i as f64)).collect();
    let xs: Vec<f64> = rows[0].x.clone();
    let all: Vec<f64> = rows.iter().flat_map(|s| s.y.iter().copied()).filter(|v| v.is_finite()).collect();
    let Some((zlo, zhi)) = range_of(&all) else { return empty_scene(opts, &spec.title, "No data to display") };
    let step = |v: &[f64]| if v.len() > 1 { (v[v.len() - 1] - v[0]) / (v.len() - 1) as f64 } else { 1.0 };
    let (dx, dy) = (step(&xs).abs().max(1e-12), step(&ys).abs().max(1e-12));
    let xr = (xs[0].min(xs[xs.len() - 1]) - dx / 2.0, xs[0].max(xs[xs.len() - 1]) + dx / 2.0);
    let yr = (ys[0].min(ys[ys.len() - 1]) - dy / 2.0, ys[0].max(ys[ys.len() - 1]) + dy / 2.0);
    let mut fs = frame_spec(
        opts,
        Scale::linear_padded(xr.0, xr.1, 0.0),
        Scale::linear_padded(yr.0, yr.1, 0.0),
        &spec.title,
        &spec.x_label,
        &spec.y_label,
    );
    fs.extra_right = 54.0;
    fs.no_grid = true;
    let mut f = Frame::new(fs);
    f.begin_data();
    for (j, row) in rows.iter().enumerate() {
        for (i, &z) in row.y.iter().enumerate() {
            if !z.is_finite() || i >= xs.len() {
                continue;
            }
            let t = if zhi > zlo { (z - zlo) / (zhi - zlo) } else { 0.5 };
            let (xa, xb) = (f.px(xs[i] - dx / 2.0), f.px(xs[i] + dx / 2.0));
            let (ya, yb) = (f.py(ys[j] - dy / 2.0), f.py(ys[j] + dy / 2.0));
            let r = Rect::new(xa.min(xb), ya.min(yb), (xb - xa).abs() + 0.5, (yb - ya).abs() + 0.5);
            f.scene.rect(r, Some(ramp(t)), None);
            f.scene.target(Region::Rect(r), j, i, format!("{}, {}: {}", num(xs[i]), num(ys[j]), num(z)));
        }
    }
    f.end_data();
    // Colour bar.
    let th = opts.theme.clone();
    let bar = Rect::new(f.plot.right() + 12.0, f.plot.y, 12.0, f.plot.h);
    let steps = 40;
    for k in 0..steps {
        let t = k as f64 / (steps - 1) as f64;
        let h = bar.h / steps as f64;
        f.scene.rect(Rect::new(bar.x, bar.bottom() - (k as f64 + 1.0) * h, bar.w, h + 0.5), Some(ramp(t)), None);
    }
    let st = TextStyle::new(th.tick_size, th.ink);
    f.scene.text(Point::new(bar.right() + 4.0, bar.y + th.tick_size * 0.8), num(zhi), st);
    f.scene.text(Point::new(bar.right() + 4.0, bar.bottom()), num(zlo), st);
    f.finish()
}

/// Dotplot: one panel per group, dots stacked in bins.
pub(crate) fn dotplot_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let groups: Vec<&Series> = spec.series.iter().collect();
    let all: Vec<f64> = groups.iter().flat_map(|s| s.x.iter().copied()).filter(|v| v.is_finite()).collect();
    let Some((lo, hi)) = range_of(&all) else { return empty_scene(opts, &spec.title, "No data to display") };
    if groups.len() > 1 {
        // Same x range in every panel.
        let panels: Vec<ChartSpec> = groups
            .iter()
            .map(|g| ChartSpec {
                kind: ChartKind::Dotplot,
                title: g.name.clone(),
                x_label: spec.x_label.clone(),
                y_label: String::new(),
                categories: vec![format!("{lo}"), format!("{hi}")],
                series: vec![(*g).clone()],
                reference_lines: vec![],
            })
            .collect();
        return panels_scene(&panels, 1, &spec.title, opts);
    }
    let (lo, hi) = match (spec.categories.first(), spec.categories.get(1)) {
        (Some(a), Some(b)) => (a.parse().unwrap_or(lo), b.parse().unwrap_or(hi)),
        _ => (lo, hi),
    };
    let data: Vec<f64> = groups[0].x.iter().copied().filter(|v| v.is_finite()).collect();
    let span = (hi - lo).max(1e-12);
    let nbins = 50usize;
    let bw = span / nbins as f64;
    let mut counts = vec![0usize; nbins + 1];
    let bin_of = |x: f64| (((x - lo) / bw).floor() as usize).min(nbins);
    for &x in &data {
        counts[bin_of(x)] += 1;
    }
    let tallest = counts.iter().copied().max().unwrap_or(1).max(1);
    let mut fs = frame_spec(
        opts,
        Scale::linear_padded(lo - bw, hi + bw, 0.0),
        Scale::linear_padded(0.0, tallest as f64 + 1.0, 0.0),
        &spec.title,
        &spec.x_label,
        "",
    );
    fs.no_grid = true;
    let mut f = Frame::new(fs);
    f.begin_data();
    let slot = f.plot.w / (nbins as f64 + 2.0);
    let dy = (f.plot.h / (tallest as f64 + 1.0)).min(slot);
    let d = dy.min(slot).max(2.0);
    let col = opts.theme.palette.color(0);
    let mut fill = vec![0usize; nbins + 1];
    let mut sorted = data.clone();
    sorted.sort_by(f64::total_cmp);
    for (k, &x) in sorted.iter().enumerate() {
        let b = bin_of(x);
        fill[b] += 1;
        let cx = f.px(lo + (b as f64 + 0.5) * bw);
        let cy = f.plot.bottom() - d * (fill[b] as f64 - 0.5);
        f.scene.push(Shape::Circle { c: Point::new(cx, cy), r: d / 2.0 * 0.9 }, None, Some(col));
        f.scene.target(Region::Point(Point::new(cx, cy), d / 2.0), 0, k, format!("{}: {}", spec.x_label, num(x)));
    }
    f.end_data();
    f.finish()
}

/// Matrix plot: scatterplots of every ordered pair of variables, names on the diagonal.
pub(crate) fn matrix_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let n = spec.categories.len();
    if n < 2 {
        return empty_scene(opts, &spec.title, "At least two variables are needed");
    }
    let mut cells: Vec<Scene> = Vec::new();
    let mut cell_opts = opts.clone();
    cell_opts.width = opts.width / n as f64;
    cell_opts.height = (opts.height - opts.theme.title_size * 1.8) / n as f64;
    cell_opts.viewport = Default::default();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                cells.push(empty_scene(&cell_opts, "", &spec.categories[i]));
                continue;
            }
            let key = format!("{i}|{j}");
            match spec.series.iter().find(|s| s.name == key) {
                Some(s) => {
                    let d = ScatterData::new(s.x.clone(), s.y.clone());
                    cells.push(scatter_plot(
                        &d,
                        &Fit::None,
                        FitBands::default(),
                        "",
                        &spec.categories[j],
                        &spec.categories[i],
                        &cell_opts,
                    ));
                }
                None => cells.push(empty_scene(&cell_opts, "", "")),
            }
        }
    }
    super::quality::grid_scene(&cells, n, &spec.title, opts)
}

/// Marginal plot: scatter with histograms of both variables.
pub(crate) fn marginal_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(s) = spec.series.first() else { return empty_scene(opts, &spec.title, "No data to display") };
    let mut cell = opts.clone();
    cell.width = opts.width / 2.0;
    cell.height = (opts.height - opts.theme.title_size * 1.8) / 2.0;
    cell.viewport = Default::default();
    let d = ScatterData::new(s.x.clone(), s.y.clone());
    let scatter = scatter_plot(&d, &Fit::None, FitBands::default(), "", &spec.x_label, &spec.y_label, &cell);
    let hx = histogram_from_data(&s.x, None, &HistogramOverlay::None, "", &spec.x_label, &cell);
    let hy = histogram_from_data(&s.y, None, &HistogramOverlay::None, "", &spec.y_label, &cell);
    let blank = empty_scene(&cell, "", "");
    super::quality::grid_scene(&[hx, blank, scatter, hy], 2, &spec.title, opts)
}

/// Three-dimensional scatter: points projected with the default orbit camera inside a
/// reference box with axis labels. `series[0]` has `x`, `y`; the series named `Z` has `y` = z.
pub(crate) fn scatter3d_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    let Some(p) = spec.series.iter().find(|s| !s.name.eq_ignore_ascii_case("z")) else {
        return empty_scene(opts, &spec.title, "No data to display");
    };
    let Some(z) = spec.series.iter().find(|s| s.name.eq_ignore_ascii_case("z")) else {
        return empty_scene(opts, &spec.title, "A Z series is needed");
    };
    let n = p.x.len().min(p.y.len()).min(z.y.len());
    let pts: Vec<(f64, f64, f64)> = (0..n)
        .filter(|&i| p.x[i].is_finite() && p.y[i].is_finite() && z.y[i].is_finite())
        .map(|i| (p.x[i], p.y[i], z.y[i]))
        .collect();
    if pts.is_empty() {
        return empty_scene(opts, &spec.title, "No data to display");
    }
    let rng = |f: &dyn Fn(&(f64, f64, f64)) -> f64| {
        let (a, b) = pts.iter().map(f).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        (a, if b > a { b } else { a + 1.0 })
    };
    let (rx, ry, rz) = (rng(&|q| q.0), rng(&|q| q.1), rng(&|q| q.2));
    let unit = |v: f64, r: (f64, f64)| (v - r.0) / (r.1 - r.0) - 0.5;
    let cam = Camera::orthographic();
    let th = &opts.theme;
    let mut s = Scene::new(opts.width, opts.height, th.background);
    s.title = spec.title.clone();
    let top = if spec.title.is_empty() { 8.0 } else { 12.0 + th.title_size * 1.4 };
    if !spec.title.is_empty() {
        s.text(
            Point::new(opts.width / 2.0, 8.0 + th.title_size),
            &spec.title,
            TextStyle::new(th.title_size, th.ink).anchored(Anchor::Middle).bold(),
        );
    }
    let area = Rect::new(24.0, top, opts.width - 48.0, opts.height - top - 24.0);
    let corners: Vec<Vec3> = (0..8)
        .map(|k| Vec3::new(f64::from(k & 1) - 0.5, f64::from((k >> 1) & 1) - 0.5, f64::from((k >> 2) & 1) - 0.5))
        .collect();
    let vt = ViewTransform::fit(&cam, &corners, area);
    let proj = |v: Vec3| cam.project_ndc(v).map(|(x, y, d)| (vt.to_pixel((x, y)), d));
    let edge = Stroke::new(th.grid.lighten(0.0), 1.0);
    for a in 0..8usize {
        for b in (a + 1)..8usize {
            if (a ^ b).count_ones() == 1 {
                if let (Some((pa, _)), Some((pb, _))) = (proj(corners[a]), proj(corners[b])) {
                    s.line(pa, pb, edge);
                }
            }
        }
    }
    let label = |s: &mut Scene, v: Vec3, t: &str| {
        if let Some((p, _)) = proj(v) {
            s.text(p, t, TextStyle::new(th.label_size, th.ink).anchored(Anchor::Middle));
        }
    };
    label(&mut s, Vec3::new(0.0, -0.62, -0.5), &spec.x_label);
    label(&mut s, Vec3::new(0.62, 0.0, -0.5), &spec.y_label);
    label(&mut s, Vec3::new(-0.55, -0.55, 0.58), &format!("{} - {}", z.name, "up"));
    let mut order: Vec<(f64, usize)> = Vec::new();
    for (k, q) in pts.iter().enumerate() {
        if let Some((_, d)) = proj(Vec3::new(unit(q.0, rx), unit(q.1, ry), unit(q.2, rz))) {
            order.push((d, k));
        }
    }
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (zlo, zhi) = rz;
    for (_, k) in order {
        let q = pts[k];
        if let Some((pp, _)) = proj(Vec3::new(unit(q.0, rx), unit(q.1, ry), unit(q.2, rz))) {
            let t = (q.2 - zlo) / (zhi - zlo);
            s.marker(pp, Marker::Circle, marker_size(pts.len()), ramp(t), None);
            s.target(
                Region::Point(pp, 6.0),
                0,
                k,
                format!("{}: {}, {}: {}, {}: {}", spec.x_label, num(q.0), spec.y_label, num(q.1), z.name, num(q.2)),
            );
        }
    }
    s
}
