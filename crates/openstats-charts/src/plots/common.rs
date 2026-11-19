//! Helpers shared by the plot builders.

use super::PlotOptions;
use crate::core::geom::Point;
use crate::core::layout::{Frame, LegendEntry};
use crate::core::scale::Scale;
use crate::core::scene::{Anchor, Scene, TextStyle};
use crate::core::style::{Color, Dash, Marker, Stroke, Theme};
use openstats_stats::report::{Axis, ReferenceLine};

/// Finite minimum and maximum of a sequence.
pub fn range_of<'a>(it: impl IntoIterator<Item = &'a f64>) -> Option<(f64, f64)> {
    let mut r: Option<(f64, f64)> = None;
    for &v in it {
        if v.is_finite() {
            r = Some(match r {
                None => (v, v),
                Some((a, b)) => (a.min(v), b.max(v)),
            });
        }
    }
    r
}

/// Merge two optional ranges.
pub fn union(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> Option<(f64, f64)> {
    match (a, b) {
        (Some(x), Some(y)) => Some((x.0.min(y.0), x.1.max(y.1))),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Up to six significant digits, no trailing zeros.
pub fn num(v: f64) -> String {
    if !v.is_finite() {
        return "*".into();
    }
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if !(1e-4..1e7).contains(&a) {
        return format!("{v:.5e}");
    }
    let d = (5 - a.log10().floor() as i32).clamp(0, 10) as usize;
    let s = format!("{v:.d$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Tooltip "x = .., y = .." text.
pub fn tip(series: &str, xl: &str, x: f64, yl: &str, y: f64) -> String {
    let xl = if xl.is_empty() { "x" } else { xl };
    let yl = if yl.is_empty() { "y" } else { yl };
    if series.is_empty() {
        format!("{xl} = {}, {yl} = {}", num(x), num(y))
    } else {
        format!("{series}: {xl} = {}, {yl} = {}", num(x), num(y))
    }
}

/// A scene with a title and a "No data" notice.
pub fn empty_scene(opts: &PlotOptions, title: &str, msg: &str) -> Scene {
    let mut s = Scene::new(opts.width, opts.height, opts.theme.background);
    s.title = title.to_string();
    if !title.is_empty() {
        s.text(
            Point::new(opts.width / 2.0, 8.0 + opts.theme.title_size * 0.95),
            title,
            TextStyle::new(opts.theme.title_size, opts.theme.ink).anchored(Anchor::Middle).bold(),
        );
    }
    s.text(
        Point::new(opts.width / 2.0, opts.height / 2.0),
        msg,
        TextStyle::new(opts.theme.label_size, opts.theme.ink).anchored(Anchor::Middle),
    );
    s
}

/// Replace a scale's domain by the viewport range when one is set.
pub fn with_view(base: Scale, view: Option<(f64, f64)>) -> Scale {
    match (view, &base) {
        (Some((a, b)), Scale::Linear { .. } | Scale::Log { .. } | Scale::Time { .. } | Scale::Probability { .. })
            if a < b =>
        {
            base.with_domain(a, b)
        }
        _ => base,
    }
}

/// Linear scale over data padded by 5 %, optionally including zero.
pub fn linear_scale(r: Option<(f64, f64)>, include_zero: bool, pad: f64) -> Scale {
    let (mut lo, mut hi) = r.unwrap_or((0.0, 1.0));
    if include_zero {
        lo = lo.min(0.0);
        hi = hi.max(0.0);
    }
    Scale::linear_padded(lo, hi, pad)
}

/// Draw reference lines (dashed) with their labels.
pub fn reference_lines(f: &mut Frame, refs: &[ReferenceLine]) {
    let th = f.theme.clone();
    for r in refs {
        let lab = r.label.to_lowercase();
        let color =
            if lab.contains("target") || lab.contains("mean") || lab.contains("center") || lab.contains("optimum") {
                th.center
            } else if lab.contains("lsl")
                || lab.contains("usl")
                || lab.contains("limit")
                || lab.contains("critical")
                || lab.contains("ucl")
                || lab.contains("lcl")
            {
                th.limit
            } else {
                th.ink.lighten(0.35)
            };
        let st = Stroke::new(color, 1.2).dashed(Dash::Dashed);
        let ts = TextStyle::new(th.tick_size, color);
        match r.axis {
            Axis::Y => {
                let y = f.py(r.value);
                if y.is_finite() && y >= f.plot.y - 0.5 && y <= f.plot.bottom() + 0.5 {
                    f.scene.line(Point::new(f.plot.x, y), Point::new(f.plot.right(), y), st);
                    f.scene.text(Point::new(f.plot.right() - 4.0, y - 3.0), r.label.clone(), ts.anchored(Anchor::End));
                }
            }
            Axis::X => {
                let x = f.px(r.value);
                if x.is_finite() && x >= f.plot.x - 0.5 && x <= f.plot.right() + 0.5 {
                    f.scene.line(Point::new(x, f.plot.y), Point::new(x, f.plot.bottom()), st);
                    f.scene.text(Point::new(x + 3.0, f.plot.y + th.tick_size + 2.0), r.label.clone(), ts);
                }
            }
        }
    }
}

/// Legend rows for named series: colour, marker and dash cycle with the series index.
pub fn series_legend(names: &[&str], theme: &Theme, markers: bool, lines: bool) -> Vec<LegendEntry> {
    names
        .iter()
        .enumerate()
        .map(|(i, n)| LegendEntry {
            name: (*n).to_string(),
            color: theme.palette.color(i),
            marker: markers.then(|| theme.palette.marker(i)),
            line: lines.then(|| theme.palette.dash(i)),
        })
        .collect()
}

/// Perceptually ordered colour ramp (dark blue → teal → yellow), `t` in [0, 1].
pub fn ramp(t: f64) -> Color {
    const STOPS: [(f64, [f64; 3]); 5] = [
        (0.0, [68.0, 1.0, 84.0]),
        (0.25, [59.0, 82.0, 139.0]),
        (0.5, [33.0, 145.0, 140.0]),
        (0.75, [94.0, 201.0, 98.0]),
        (1.0, [253.0, 231.0, 37.0]),
    ];
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    for w in STOPS.windows(2) {
        if t <= w[1].0 {
            let u = (t - w[0].0) / (w[1].0 - w[0].0);
            let c = |i: usize| (w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u).round() as u8;
            return Color { r: c(0), g: c(1), b: c(2), a: 255 };
        }
    }
    Color { r: 253, g: 231, b: 37, a: 255 }
}

/// Marker size by point count (smaller when crowded).
pub fn marker_size(n: usize) -> f64 {
    match n {
        0..=60 => 6.5,
        61..=300 => 5.0,
        _ => 3.5,
    }
}

/// Maximum number of hit targets registered per chart.
pub const MAX_TARGETS: usize = 20_000;

/// Marker for series `i` honouring the theme palette.
pub fn marker_of(theme: &Theme, i: usize) -> Marker {
    theme.palette.marker(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_numbers() {
        assert_eq!(range_of(&[3.0, f64::NAN, -1.0, 7.0]), Some((-1.0, 7.0)));
        assert_eq!(range_of(&[f64::NAN]), None);
        assert_eq!(union(Some((0.0, 1.0)), Some((-2.0, 0.5))), Some((-2.0, 1.0)));
        assert_eq!(union(None, Some((1.0, 2.0))), Some((1.0, 2.0)));
        assert_eq!(num(0.0), "0");
        assert_eq!(num(1.5), "1.5");
        assert_eq!(num(f64::NAN), "*");
        assert_eq!(num(123456.0), "123456");
        assert!(num(1.0e-9).contains('e'));
    }

    #[test]
    fn ramp_is_monotone_in_brightness_and_clamped() {
        assert_eq!(ramp(-1.0), ramp(0.0));
        assert_eq!(ramp(2.0), ramp(1.0));
        assert_eq!(ramp(f64::NAN), ramp(0.0));
        let lum = |c: Color| u32::from(c.r) + u32::from(c.g) + u32::from(c.b);
        assert!(lum(ramp(1.0)) > lum(ramp(0.0)));
    }

    #[test]
    fn marker_sizes_shrink_with_count() {
        assert!(marker_size(10) > marker_size(1000));
    }
}
