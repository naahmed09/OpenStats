//! Chart data for designed experiments: main effects, interaction, cube,
//! Pareto and normal plots of effects, contour and response-surface plots.

use super::analysis::FactorialAnalysis;
use super::design::{FactorSpec, PointType};
use super::model::PolynomialModel;
use crate::error::{invalid, Result};
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::special::erf::norm_ppf;

fn mean_where(a: &FactorialAnalysis, pred: impl Fn(usize) -> bool) -> Option<f64> {
    let (mut s, mut n) = (0.0, 0usize);
    for i in 0..a.y.len() {
        if pred(i) {
            s += a.y[i];
            n += 1;
        }
    }
    (n > 0).then(|| s / n as f64)
}

/// One chart per factor: the mean response at each level of the factor
/// (corner points at −1 and +1, plus the centre-point mean at 0 when there are centre points).
pub fn main_effects_charts(a: &FactorialAnalysis) -> Vec<ChartSpec> {
    let d = &a.design;
    let has_center = d.point_type.contains(&PointType::Center);
    (0..d.n_factors())
        .map(|j| {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for lv in [-1.0, 1.0] {
                if let Some(m) = mean_where(a, |i| d.point_type[i] == PointType::Factorial && d.coded[i][j] == lv) {
                    x.push(lv);
                    y.push(m);
                }
            }
            if has_center {
                if let Some(m) = mean_where(a, |i| d.point_type[i] == PointType::Center) {
                    x.insert(1, 0.0);
                    y.insert(1, m);
                }
            }
            ChartSpec {
                kind: ChartKind::Line,
                title: format!("Main Effects Plot for {}", d.factors[j].name),
                x_label: format!("{} (coded)", d.factors[j].name),
                y_label: "Mean response".into(),
                categories: Vec::new(),
                series: vec![Series { name: "Mean".into(), x, y }],
                reference_lines: vec![ReferenceLine {
                    axis: Axis::Y,
                    value: a.y.iter().sum::<f64>() / a.y.len() as f64,
                    label: "Grand mean".into(),
                }],
            }
        })
        .collect()
}

/// One chart per pair of factors: lines of the cell means of the first factor's levels, one per level of the second.
pub fn interaction_charts(a: &FactorialAnalysis) -> Vec<ChartSpec> {
    let d = &a.design;
    let k = d.n_factors();
    let mut out = Vec::new();
    for i in 0..k {
        for j in i + 1..k {
            let mut series = Vec::new();
            for lj in [-1.0, 1.0] {
                let mut x = Vec::new();
                let mut y = Vec::new();
                for li in [-1.0, 1.0] {
                    if let Some(m) = mean_where(a, |r| {
                        d.point_type[r] == PointType::Factorial && d.coded[r][i] == li && d.coded[r][j] == lj
                    }) {
                        x.push(li);
                        y.push(m);
                    }
                }
                series.push(Series {
                    name: format!("{}={}", d.factors[j].name, if lj < 0.0 { "-1" } else { "+1" }),
                    x,
                    y,
                });
            }
            out.push(ChartSpec {
                kind: ChartKind::Line,
                title: format!("Interaction Plot: {} * {}", d.factors[i].name, d.factors[j].name),
                x_label: format!("{} (coded)", d.factors[i].name),
                y_label: "Mean response".into(),
                categories: Vec::new(),
                series,
                reference_lines: Vec::new(),
            });
        }
    }
    out
}

/// Cube plot of three factors: the mean response at the eight corners. Each
/// corner is a point (x = first factor's level, y = second's) in the series
/// of the third factor's level; `categories` label the points in series
/// order with the corner and its mean.
pub fn cube_chart(a: &FactorialAnalysis, factors: [usize; 3]) -> Result<ChartSpec> {
    let d = &a.design;
    if factors.iter().any(|&f| f >= d.n_factors())
        || factors[0] == factors[1]
        || factors[0] == factors[2]
        || factors[1] == factors[2]
    {
        return invalid("a cube plot needs three distinct factors of the design");
    }
    let mut series = Vec::new();
    let mut cats = Vec::new();
    for lc in [-1.0, 1.0] {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for lb in [-1.0, 1.0] {
            for la in [-1.0, 1.0] {
                let m = mean_where(a, |r| {
                    d.point_type[r] == PointType::Factorial
                        && d.coded[r][factors[0]] == la
                        && d.coded[r][factors[1]] == lb
                        && d.coded[r][factors[2]] == lc
                });
                if let Some(m) = m {
                    x.push(la);
                    y.push(lb);
                    cats.push(format!(
                        "{}={:+},{}={:+},{}={:+}: {:.4}",
                        d.factors[factors[0]].name,
                        la,
                        d.factors[factors[1]].name,
                        lb,
                        d.factors[factors[2]].name,
                        lc,
                        m
                    ));
                }
            }
        }
        series.push(Series { name: format!("{}={:+}", d.factors[factors[2]].name, lc), x, y });
    }
    Ok(ChartSpec {
        kind: ChartKind::Scatter,
        title: "Cube Plot (data means)".into(),
        x_label: d.factors[factors[0]].name.clone(),
        y_label: d.factors[factors[1]].name.clone(),
        categories: cats,
        series,
        reference_lines: Vec::new(),
    })
}

/// Pareto chart of the standardized effects (|t|, with the critical t as the
/// reference line) or, without error degrees of freedom, of |effect| with Lenth's margin of error.
pub fn pareto_chart(a: &FactorialAnalysis) -> Result<ChartSpec> {
    let mut rows: Vec<(String, f64)> = Vec::new();
    let mut refline = None;
    if a.df_error > 0 {
        for e in &a.effects {
            if let Some(t) = e.t {
                rows.push((e.term.clone(), t.abs()));
            }
        }
        let tc = StudentT::new(a.df_error as f64)?.quantile(1.0 - (1.0 - a.level) / 2.0);
        refline = Some(ReferenceLine { axis: Axis::Y, value: tc, label: format!("t critical = {tc:.4}") });
    } else {
        for e in &a.effects {
            if let Some(v) = e.effect {
                rows.push((e.term.clone(), v.abs()));
            }
        }
        if let Some(l) = &a.lenth {
            refline =
                Some(ReferenceLine { axis: Axis::Y, value: l.margin, label: format!("Lenth ME = {:.4}", l.margin) });
        }
    }
    rows.sort_by(|x, y| y.1.total_cmp(&x.1));
    Ok(ChartSpec {
        kind: ChartKind::Bar,
        title: "Pareto Chart of the Effects".into(),
        x_label: "Term".into(),
        y_label: if a.df_error > 0 { "Standardized effect |t|" } else { "|Effect|" }.into(),
        categories: rows.iter().map(|r| r.0.clone()).collect(),
        series: vec![Series {
            name: "Effect".into(),
            x: (1..=rows.len()).map(|i| i as f64).collect(),
            y: rows.iter().map(|r| r.1).collect(),
        }],
        reference_lines: refline.into_iter().collect(),
    })
}

/// Normal probability plot of the effects (Blom plotting positions); the
/// category labels name the points in x-sorted order.
pub fn normal_effects_chart(a: &FactorialAnalysis) -> ChartSpec {
    let mut rows: Vec<(String, f64)> = a.effects.iter().filter_map(|e| e.effect.map(|v| (e.term.clone(), v))).collect();
    rows.sort_by(|x, y| x.1.total_cmp(&y.1));
    let m = rows.len() as f64;
    ChartSpec {
        kind: ChartKind::ProbabilityPlot,
        title: "Normal Plot of the Effects".into(),
        x_label: "Effect".into(),
        y_label: "Normal score".into(),
        categories: rows.iter().map(|r| r.0.clone()).collect(),
        series: vec![Series {
            name: "Effects".into(),
            x: rows.iter().map(|r| r.1).collect(),
            y: (1..=rows.len()).map(|i| norm_ppf((i as f64 - 0.375) / (m + 0.25))).collect(),
        }],
        reference_lines: Vec::new(),
    }
}

/// Grid of a model over two factors (others held at `fixed`, coded values).
/// Grid coordinates and the model values `z[row][col]`.
type Grid = (Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

fn grid(p: &SurfacePlot<'_>) -> Result<Grid> {
    let SurfacePlot { model, factors, ix, iy, fixed, range, grid_points: n } = *p;
    let k = model.n_factors();
    if ix >= k || iy >= k || ix == iy || fixed.len() != k || factors.len() != k {
        return invalid("a plot needs two distinct factors and one fixed coded value per factor");
    }
    if n < 3 {
        return invalid("the grid needs at least 3 points per axis");
    }
    let xs: Vec<f64> = (0..n).map(|i| range.0[0] + (range.0[1] - range.0[0]) * i as f64 / (n - 1) as f64).collect();
    let ys: Vec<f64> = (0..n).map(|i| range.1[0] + (range.1[1] - range.1[0]) * i as f64 / (n - 1) as f64).collect();
    let z: Vec<Vec<f64>> = ys
        .iter()
        .map(|&y| {
            xs.iter()
                .map(|&x| {
                    let mut p = fixed.to_vec();
                    p[ix] = x;
                    p[iy] = y;
                    model.predict(&p)
                })
                .collect()
        })
        .collect();
    Ok((xs, ys, z))
}

type Seg = ([f64; 2], [f64; 2]);

/// Marching-squares segments of the level set `z = level` (rows of `z` indexed by y).
pub fn contour_segments(xs: &[f64], ys: &[f64], z: &[Vec<f64>], level: f64) -> Vec<Seg> {
    let mut out = Vec::new();
    for j in 0..ys.len() - 1 {
        for i in 0..xs.len() - 1 {
            let c = [
                (xs[i], ys[j], z[j][i]),
                (xs[i + 1], ys[j], z[j][i + 1]),
                (xs[i + 1], ys[j + 1], z[j + 1][i + 1]),
                (xs[i], ys[j + 1], z[j + 1][i]),
            ];
            let mut pts: [Option<[f64; 2]>; 4] = [None; 4];
            for e in 0..4 {
                let (a, b) = (c[e], c[(e + 1) % 4]);
                if (a.2 < level) != (b.2 < level) {
                    let t = (level - a.2) / (b.2 - a.2);
                    pts[e] = Some([a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)]);
                }
            }
            let have: Vec<usize> = (0..4).filter(|&e| pts[e].is_some()).collect();
            let get = |e: usize| pts[e].unwrap_or([f64::NAN; 2]);
            match have.len() {
                2 => out.push((get(have[0]), get(have[1]))),
                4 => {
                    let centre = 0.25 * (c[0].2 + c[1].2 + c[2].2 + c[3].2);
                    if (centre < level) == (c[0].2 < level) {
                        out.push((get(0), get(1)));
                        out.push((get(2), get(3)));
                    } else {
                        out.push((get(3), get(0)));
                        out.push((get(1), get(2)));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// What to draw in a contour or surface plot: the model, the two plotted
/// factors and the coded values at which the others are held.
#[derive(Debug, Clone, Copy)]
pub struct SurfacePlot<'a> {
    /// Fitted model (coded factors).
    pub model: &'a PolynomialModel,
    /// Factor definitions (natural units on the axes).
    pub factors: &'a [FactorSpec],
    /// Factor on the x axis.
    pub ix: usize,
    /// Factor on the y axis.
    pub iy: usize,
    /// Coded value of every factor (entries `ix` and `iy` are ignored).
    pub fixed: &'a [f64],
    /// Coded `[low, high]` of the x and y axes.
    pub range: ([f64; 2], [f64; 2]),
    /// Grid points per axis (≥ 3).
    pub grid_points: usize,
}

/// Contour plot of the model over factors `ix` × `iy` (axes in natural units);
/// the other factors are held at `fixed`. `levels` equally spaced contour
/// levels strictly between the grid's minimum and maximum are drawn.
pub fn contour_chart(p: &SurfacePlot<'_>, levels: usize) -> Result<ChartSpec> {
    let SurfacePlot { factors, ix, iy, .. } = *p;
    let (xs, ys, z) = grid(p)?;
    let (zmin, zmax) = z.iter().flatten().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    let nat = |f: &FactorSpec, c: f64| f.to_natural(c);
    let mut series = Vec::new();
    for l in 1..=levels {
        let level = zmin + (zmax - zmin) * l as f64 / (levels + 1) as f64;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (a, b) in contour_segments(&xs, &ys, &z, level) {
            x.push(nat(&factors[ix], a[0]));
            y.push(nat(&factors[iy], a[1]));
            x.push(nat(&factors[ix], b[0]));
            y.push(nat(&factors[iy], b[1]));
            x.push(f64::NAN);
            y.push(f64::NAN);
        }
        if !x.is_empty() {
            series.push(Series { name: format!("{level:.6}"), x, y });
        }
    }
    Ok(ChartSpec {
        kind: ChartKind::Contour,
        title: format!("Contour Plot of Response vs {}, {}", factors[iy].name, factors[ix].name),
        x_label: factors[ix].name.clone(),
        y_label: factors[iy].name.clone(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    })
}

/// Response-surface data over factors `ix` × `iy` (natural-unit axes): one
/// series per grid row, named by the row's natural y value.
pub fn surface_chart(p: &SurfacePlot<'_>) -> Result<ChartSpec> {
    let SurfacePlot { factors, ix, iy, .. } = *p;
    let (xs, ys, z) = grid(p)?;
    let series = ys
        .iter()
        .zip(&z)
        .map(|(&y, row)| Series {
            name: format!("{}", factors[iy].to_natural(y)),
            x: xs.iter().map(|&x| factors[ix].to_natural(x)).collect(),
            y: row.clone(),
        })
        .collect();
    Ok(ChartSpec {
        kind: ChartKind::Surface,
        title: format!("Surface Plot of Response vs {}, {}", factors[iy].name, factors[ix].name),
        x_label: factors[ix].name.clone(),
        y_label: "Response".into(),
        categories: vec![factors[iy].name.clone()],
        series,
        reference_lines: Vec::new(),
    })
}
