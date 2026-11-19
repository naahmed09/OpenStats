//! The Graph menu (Phase 5j): bar and pie charts, dotplots, stem-and-leaf displays, individual
//! value, interval and empirical-CDF plots, area graphs, bubble, matrix and marginal plots, heat
//! maps, 3-D scatter, surface and contour plots, and probability plots for three distributions.
//! Every dialog yields a `ChartSpec` (so the graph appears in the Navigator and can be exported
//! as SVG, PNG or PDF) except the stem-and-leaf display, which is a text table.

use super::analyses::add;
use super::analysis::{col_title, labels, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Value, Values, VarKind};
use openstats_charts::render_3d::{contour_lines, nice_levels};
use openstats_core::Worksheet;
use openstats_stats::graphical::ci_mean;
use openstats_stats::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};
use openstats_stats::stemleaf::stem_and_leaf;

fn mk(kind: ChartKind, title: String, xl: &str, yl: &str, cats: Vec<String>, series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind,
        title,
        x_label: xl.into(),
        y_label: yl.into(),
        categories: cats,
        series,
        reference_lines: Vec::new(),
    }
}

fn names(ws: &Worksheet, cols: &[usize]) -> String {
    cols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
}

/// Distinct labels in order: numeric order when every label is a number, else alphabetical.
fn sorted_levels(l: &[String]) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for x in l {
        if !x.is_empty() && !v.contains(x) {
            v.push(x.clone());
        }
    }
    if v.iter().all(|s| s.parse::<f64>().is_ok()) {
        v.sort_by(|a, b| a.parse::<f64>().unwrap_or(0.0).total_cmp(&b.parse::<f64>().unwrap_or(0.0)));
    } else {
        v.sort();
    }
    v
}

/// The numeric samples to plot: one per selected column, or one per level of `by`.
fn samples(ws: &Worksheet, v: &Values) -> Res<Vec<(String, Vec<f64>)>> {
    let cols = v.vars("vars");
    if cols.is_empty() {
        return Err("select at least one graph variable".into());
    }
    match v.var("by") {
        None => cols.iter().map(|&c| Ok((col_title(ws, c), numeric(ws, c)?))).collect(),
        Some(b) => {
            if cols.len() != 1 {
                return Err("with a grouping variable, select exactly one graph variable".into());
            }
            let y = numeric(ws, cols[0])?;
            let g = labels(ws, b)?;
            if g.len() != y.len() {
                return Err("the graph variable and the grouping variable have different lengths".into());
            }
            Ok(sorted_levels(&g)
                .into_iter()
                .map(|lv| {
                    let d = y.iter().zip(&g).filter(|(_, gi)| **gi == lv).map(|(&a, _)| a).collect();
                    (lv, d)
                })
                .collect())
        }
    }
}

fn finite(d: &[f64]) -> Vec<f64> {
    d.iter().copied().filter(|x| x.is_finite()).collect()
}

fn single(title: &str, c: ChartSpec) -> Res<Output> {
    let mut s = Section::new(title);
    s.chart(c);
    super::analyses::out(s)
}

// ------------------------------------------------------------------ bar and pie

struct Tally {
    cats: Vec<String>,
    groups: Vec<String>,
    /// `h[g][c]`: value of group `g` in category `c`.
    h: Vec<Vec<f64>>,
}

fn tally(ws: &Worksheet, v: &Values) -> Res<Tally> {
    let cat_col = v.var("cat").ok_or("select a categorical variable")?;
    let cat = labels(ws, cat_col)?;
    let by = match v.var("group") {
        Some(g) => Some(labels(ws, g)?),
        None => None,
    };
    let val = match v.var("value") {
        Some(c) => Some(numeric(ws, c)?),
        None => None,
    };
    let stat = v.choice("stat");
    if stat > 0 && val.is_none() {
        return Err("select the value variable to sum or average".into());
    }
    for (n, len) in [("grouping", by.as_ref().map(Vec::len)), ("value", val.as_ref().map(Vec::len))] {
        if len.is_some_and(|l| l != cat.len()) {
            return Err(format!("the {n} variable and the categorical variable have different lengths").into());
        }
    }
    let cats = sorted_levels(&cat);
    if cats.is_empty() {
        return Err("the categorical variable has no values".into());
    }
    let groups = match &by {
        Some(g) => sorted_levels(g),
        None => vec![String::new()],
    };
    let mut sum = vec![vec![0.0; cats.len()]; groups.len()];
    let mut cnt = vec![vec![0usize; cats.len()]; groups.len()];
    for i in 0..cat.len() {
        let Some(ci) = cats.iter().position(|c| *c == cat[i]) else { continue };
        let gi = match &by {
            Some(g) => match groups.iter().position(|x| *x == g[i]) {
                Some(k) => k,
                None => continue,
            },
            None => 0,
        };
        match &val {
            Some(x) if stat > 0 => {
                if x[i].is_finite() {
                    sum[gi][ci] += x[i];
                    cnt[gi][ci] += 1;
                }
            }
            _ => {
                sum[gi][ci] += 1.0;
                cnt[gi][ci] += 1;
            }
        }
    }
    if stat == 2 {
        for g in 0..groups.len() {
            for c in 0..cats.len() {
                sum[g][c] = if cnt[g][c] > 0 { sum[g][c] / cnt[g][c] as f64 } else { f64::NAN };
            }
        }
    }
    Ok(Tally { cats, groups, h: sum })
}

fn bar_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut t = tally(ws, v)?;
    let stat = v.choice("stat");
    if v.flag("percent") {
        let total: f64 = t.h.iter().flatten().filter(|x| x.is_finite()).sum();
        if total <= 0.0 {
            return Err("percentages need a positive total".into());
        }
        for x in t.h.iter_mut().flatten() {
            *x = *x / total * 100.0;
        }
    }
    if v.choice("order") == 1 {
        let tot: Vec<f64> =
            (0..t.cats.len()).map(|c| t.h.iter().map(|g| if g[c].is_finite() { g[c] } else { 0.0 }).sum()).collect();
        let mut idx: Vec<usize> = (0..t.cats.len()).collect();
        idx.sort_by(|&a, &b| tot[b].total_cmp(&tot[a]));
        t.cats = idx.iter().map(|&i| t.cats[i].clone()).collect();
        for g in &mut t.h {
            *g = idx.iter().map(|&i| g[i]).collect();
        }
    }
    let cat_name = col_title(ws, v.var("cat").unwrap_or(0));
    let ylab = match (stat, v.flag("percent")) {
        (_, true) => "Percent",
        (0, _) => "Count",
        (1, _) => "Sum",
        _ => "Mean",
    };
    let series: Vec<Series> = t
        .groups
        .iter()
        .zip(&t.h)
        .map(|(g, h)| Series { name: if g.is_empty() { ylab.to_string() } else { g.clone() }, x: vec![], y: h.clone() })
        .collect();
    let stacked = v.choice("multi") == 1 && series.len() > 1;
    let kind = if stacked { ChartKind::StackedBar } else { ChartKind::Bar };
    let mut s = Section::new("Bar Chart");
    let mut tb = Table::new(
        "Bar chart data",
        std::iter::once(Column::new(cat_name.clone(), ColumnKind::Text))
            .chain(series.iter().map(|x| Column::new(x.name.clone(), ColumnKind::Significant(6))))
            .collect(),
    );
    for (c, cat) in t.cats.iter().enumerate() {
        let mut row = vec![Cell::text(cat)];
        row.extend(series.iter().map(|x| Cell::Num(x.y[c])));
        tb.push_row(row).map_err(|e| e.to_string())?;
    }
    s.table(tb);
    s.chart(mk(kind, format!("Chart of {ylab} by {cat_name}"), &cat_name, ylab, t.cats, series));
    super::analyses::out(s)
}

fn pie_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut v = v.clone();
    if v.var("value").is_some() {
        v.set("stat", Value::Choice(1));
    }
    let v = &v;
    let t = tally(ws, v)?;
    let mut vals: Vec<f64> =
        (0..t.cats.len()).map(|c| t.h.iter().map(|g| if g[c].is_finite() { g[c] } else { 0.0 }).sum()).collect();
    let mut cats = t.cats;
    if vals.iter().any(|x| *x < 0.0) {
        return Err("a pie chart needs non-negative values".into());
    }
    let total: f64 = vals.iter().sum();
    if total <= 0.0 {
        return Err("the pie has no positive values".into());
    }
    let minp = v.num("combine");
    if minp.is_finite() && minp > 0.0 {
        let (mut keep_c, mut keep_v, mut other) = (Vec::new(), Vec::new(), 0.0);
        for (c, x) in cats.iter().zip(&vals) {
            if x / total * 100.0 < minp {
                other += x;
            } else {
                keep_c.push(c.clone());
                keep_v.push(*x);
            }
        }
        if other > 0.0 {
            keep_c.push("Other".into());
            keep_v.push(other);
        }
        cats = keep_c;
        vals = keep_v;
    }
    let name = col_title(ws, v.var("cat").unwrap_or(0));
    single(
        "Pie Chart",
        mk(
            ChartKind::Pie,
            format!("Pie Chart of {name}"),
            "",
            "",
            cats,
            vec![Series { name: "Size".into(), x: vec![], y: vals }],
        ),
    )
}

// ------------------------------------------------------------------ distribution plots

fn dot_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let g = samples(ws, v)?;
    let nm = names(ws, v.vars("vars"));
    let series: Vec<Series> =
        g.into_iter().map(|(n, d)| Series { name: n, x: finite(&d), y: vec![] }).filter(|s| !s.x.is_empty()).collect();
    if series.is_empty() {
        return Err("there are no values to plot".into());
    }
    single("Dotplot", mk(ChartKind::Dotplot, format!("Dotplot of {nm}"), &nm, "", vec![], series))
}

fn stem_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let unit = {
        let t = v.text("unit").trim();
        if t.is_empty() {
            None
        } else {
            Some(t.parse::<f64>().map_err(|_| format!("'{t}' is not a number (leaf unit)"))?)
        }
    };
    let g = samples(ws, v)?;
    let mut s = Section::new("Stem-and-Leaf");
    for (name, d) in g {
        let sl = stem_and_leaf(&d, unit).map_err(|e| format!("{name}: {e}"))?;
        let part = sl.report_section();
        s.text(format!("Stem-and-Leaf Display: {name}"));
        s.blocks.extend(part.blocks);
        if v.var("by").is_some() {
            s.text("");
        }
    }
    super::analyses::out(s)
}

fn ivp_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let g = samples(ws, v)?;
    let nm = names(ws, v.vars("vars"));
    let series: Vec<Series> =
        g.into_iter().map(|(n, d)| Series { name: n, x: vec![], y: finite(&d) }).filter(|s| !s.y.is_empty()).collect();
    if series.is_empty() {
        return Err("there are no values to plot".into());
    }
    single(
        "Individual Value Plot",
        mk(ChartKind::IndividualValue, format!("Individual Value Plot of {nm}"), "", &nm, vec![], series),
    )
}

fn interval_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let level = v.num("level") / 100.0;
    if !(level > 0.0 && level < 1.0) {
        return Err("the confidence level must be between 0 and 100".into());
    }
    let g = samples(ws, v)?;
    let (mut cats, mut est, mut lo, mut hi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut s = Section::new("Interval Plot");
    for (n, d) in g {
        let d = finite(&d);
        match ci_mean(&d, level) {
            Ok(ci) => {
                cats.push(n);
                est.push(d.iter().sum::<f64>() / d.len() as f64);
                lo.push(ci.lower);
                hi.push(ci.upper);
            }
            Err(_) => {
                s.warn(format!("{n}: at least two values are needed for an interval; the group is left out"));
            }
        }
    }
    if cats.is_empty() {
        return Err("no group has enough values for an interval".into());
    }
    let pos: Vec<f64> = (1..=cats.len()).map(|i| i as f64).collect();
    let nm = names(ws, v.vars("vars"));
    s.chart(mk(
        ChartKind::IntervalPlot,
        format!("Interval Plot of {nm} ({}% CI for the Mean)", level * 100.0),
        "Mean",
        "Level",
        cats,
        vec![
            Series { name: "estimate".into(), x: est, y: pos.clone() },
            Series { name: "lower".into(), x: lo, y: pos.clone() },
            Series { name: "upper".into(), x: hi, y: pos },
        ],
    ));
    super::analyses::out(s)
}

fn ecdf_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let g = samples(ws, v)?;
    let nm = names(ws, v.vars("vars"));
    let series: Vec<Series> =
        g.into_iter().map(|(n, d)| Series { name: n, x: finite(&d), y: vec![] }).filter(|s| !s.x.is_empty()).collect();
    if series.is_empty() {
        return Err("there are no values to plot".into());
    }
    single("Empirical CDF", mk(ChartKind::Ecdf, format!("Empirical CDF of {nm}"), &nm, "", vec![], series))
}

fn prob_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let d = numeric(ws, c)?;
    let name = col_title(ws, c);
    match v.choice("dist") {
        0 => {
            let mut s = Section::new("Probability Plot");
            s.chart(super::analyses::prob_chart(&d, &name)?);
            super::analyses::out(s)
        }
        k => {
            let dist = if k == 1 { "Lognormal" } else { "Weibull" };
            let x = finite(&d);
            if x.iter().any(|&t| t <= 0.0) {
                return Err(format!("a {dist} probability plot needs positive values").into());
            }
            if x.len() < 3 {
                return Err("a probability plot needs at least 3 values".into());
            }
            single(
                "Probability Plot",
                mk(
                    ChartKind::DataProbability,
                    format!("Probability Plot of {name} ({dist}, 95% CI)"),
                    &name,
                    "Percent",
                    vec![],
                    vec![Series { name: dist.into(), x, y: vec![] }],
                ),
            )
        }
    }
}

// ------------------------------------------------------------------ relationships

type XyData = (usize, usize, Vec<usize>, Vec<f64>, Vec<f64>);
type XyzData = (String, String, String, Vec<f64>, Vec<f64>, Vec<f64>);

fn xy(ws: &Worksheet, v: &Values) -> Res<XyData> {
    let (y, x) = (v.var("y").ok_or("select Y")?, v.var("x").ok_or("select X")?);
    let (yd, xd) = (numeric(ws, y)?, numeric(ws, x)?);
    let (rows, d) = super::analysis::complete_rows(&[&xd, &yd]);
    if rows.len() < 2 {
        return Err("at least two complete rows are needed".into());
    }
    Ok((y, x, rows, d[0].clone(), d[1].clone()))
}

fn area_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    if cols.is_empty() {
        return Err("select at least one variable".into());
    }
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let n = data[0].len();
    if data.iter().any(|d| d.len() != n) {
        return Err("the variables have different lengths".into());
    }
    let (x, xl): (Vec<f64>, String) = match v.var("x") {
        Some(c) => {
            let x = numeric(ws, c)?;
            if x.len() != n {
                return Err("the X variable and the variables have different lengths".into());
            }
            (x, col_title(ws, c))
        }
        None => ((1..=n).map(|i| i as f64).collect(), "Index".into()),
    };
    let stacked = v.flag("stacked") && cols.len() > 1;
    let mut cum = vec![0.0; n];
    let mut series = Vec::new();
    for (k, d) in data.iter().enumerate() {
        let y: Vec<f64> = if stacked {
            for i in 0..n {
                cum[i] += if d[i].is_finite() { d[i] } else { 0.0 };
            }
            cum.clone()
        } else {
            d.clone()
        };
        series.push((col_title(ws, cols[k]), y));
    }
    if stacked {
        series.reverse(); // largest total first, so the lower areas stay visible
    }
    let series: Vec<Series> = series
        .into_iter()
        .map(|(name, y)| {
            let (xs, ys): (Vec<f64>, Vec<f64>) =
                x.iter().zip(&y).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).unzip();
            Series { name, x: xs, y: ys }
        })
        .collect();
    let nm = names(ws, cols);
    single(
        "Area Graph",
        mk(
            ChartKind::Area,
            format!("Area Graph of {nm}"),
            &xl,
            if stacked { "Cumulative" } else { "Value" },
            vec![],
            series,
        ),
    )
}

fn bubble_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, x) = (v.var("y").ok_or("select Y")?, v.var("x").ok_or("select X")?);
    let sz = v.var("size").ok_or("select the size variable")?;
    let (yd, xd, sd) = (numeric(ws, y)?, numeric(ws, x)?, numeric(ws, sz)?);
    let (rows, d) = super::analysis::complete_rows(&[&xd, &yd, &sd]);
    let keep: Vec<usize> = (0..rows.len()).filter(|&i| d[2][i] > 0.0).collect();
    if keep.is_empty() {
        return Err("the size variable has no positive values".into());
    }
    let pick = |k: usize| keep.iter().map(|&i| d[k][i]).collect::<Vec<f64>>();
    let mut s = Section::new("Bubble Plot");
    if keep.len() < rows.len() {
        s.warn(format!("{} row(s) with a non-positive size were left out", rows.len() - keep.len()));
    }
    s.chart(mk(
        ChartKind::Bubble,
        format!("Bubble Plot of {} vs {} (size {})", col_title(ws, y), col_title(ws, x), col_title(ws, sz)),
        &col_title(ws, x),
        &col_title(ws, y),
        vec![],
        vec![
            Series { name: col_title(ws, y), x: pick(0), y: pick(1) },
            Series { name: "Size".into(), x: vec![], y: pick(2) },
        ],
    ));
    super::analyses::out(s)
}

fn matrix_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    if cols.len() < 2 {
        return Err("select at least two variables".into());
    }
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let mut series = Vec::new();
    for i in 0..cols.len() {
        for j in 0..cols.len() {
            if i == j {
                continue;
            }
            let (_, d) = super::analysis::complete_rows(&[&data[j], &data[i]]);
            series.push(Series { name: format!("{i}|{j}"), x: d[0].clone(), y: d[1].clone() });
        }
    }
    single(
        "Matrix Plot",
        mk(
            ChartKind::Matrix,
            format!("Matrix Plot of {}", names(ws, cols)),
            "",
            "",
            cols.iter().map(|&c| col_title(ws, c)).collect(),
            series,
        ),
    )
}

fn marginal_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, x, _, xd, yd) = xy(ws, v)?;
    single(
        "Marginal Plot",
        mk(
            ChartKind::Marginal,
            format!("Marginal Plot of {} vs {}", col_title(ws, y), col_title(ws, x)),
            &col_title(ws, x),
            &col_title(ws, y),
            vec![],
            vec![Series { name: col_title(ws, y), x: xd, y: yd }],
        ),
    )
}

fn heat_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    if cols.len() < 2 {
        return Err("select at least two variables (the columns of the map)".into());
    }
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let n = data.iter().map(Vec::len).max().unwrap_or(0);
    if n == 0 {
        return Err("the variables are empty".into());
    }
    if n > 400 {
        return Err(format!("a heat map shows at most 400 rows (the variables have {n})").into());
    }
    let series: Vec<Series> = (0..n)
        .map(|r| Series {
            name: format!("{}", r + 1),
            x: (1..=cols.len()).map(|c| c as f64).collect(),
            y: data.iter().map(|d| d.get(r).copied().unwrap_or(f64::NAN)).collect(),
        })
        .collect();
    single(
        "Heatmap",
        mk(
            ChartKind::Heatmap,
            format!("Heatmap of {}", names(ws, cols)),
            "Variable number (in the order selected)",
            "Row",
            vec![],
            series,
        ),
    )
}

// ------------------------------------------------------------------ 3-D

fn xyz(ws: &Worksheet, v: &Values) -> Res<XyzData> {
    let (z, y, x) = (v.var("z").ok_or("select Z")?, v.var("y").ok_or("select Y")?, v.var("x").ok_or("select X")?);
    let (zd, yd, xd) = (numeric(ws, z)?, numeric(ws, y)?, numeric(ws, x)?);
    let (_, d) = super::analysis::complete_rows(&[&xd, &yd, &zd]);
    if d[0].len() < 3 {
        return Err("at least three complete rows are needed".into());
    }
    Ok((col_title(ws, x), col_title(ws, y), col_title(ws, z), d[0].clone(), d[1].clone(), d[2].clone()))
}

fn scatter3d_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (xn, yn, zn, x, y, z) = xyz(ws, v)?;
    single(
        "3D Scatterplot",
        mk(
            ChartKind::Scatter3D,
            format!("3D Scatterplot of {zn} vs {yn} vs {xn}"),
            &xn,
            &yn,
            vec![],
            vec![Series { name: yn.clone(), x, y }, Series { name: "Z".into(), x: vec![], y: z }],
        ),
    )
}

/// Regular grid of the data: used as is when the `(x, y)` pairs form a complete grid, otherwise
/// interpolated by inverse-distance weighting (power 2) onto an `n × n` grid over the data range.
fn grid_of(x: &[f64], y: &[f64], z: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, bool) {
    let uniq = |v: &[f64]| {
        let mut u = v.to_vec();
        u.sort_by(f64::total_cmp);
        u.dedup();
        u
    };
    let (ux, uy) = (uniq(x), uniq(y));
    if ux.len() >= 2 && uy.len() >= 2 && ux.len() * uy.len() == x.len() {
        let mut g = vec![vec![f64::NAN; ux.len()]; uy.len()];
        let mut filled = 0;
        for i in 0..x.len() {
            let (a, b) = (ux.binary_search_by(|p| p.total_cmp(&x[i])), uy.binary_search_by(|p| p.total_cmp(&y[i])));
            if let (Ok(a), Ok(b)) = (a, b) {
                if g[b][a].is_nan() {
                    filled += 1;
                }
                g[b][a] = z[i];
            }
        }
        if filled == x.len() {
            return (ux, uy, g, false);
        }
    }
    let (x0, x1) = (ux[0], ux[ux.len() - 1]);
    let (y0, y1) = (uy[0], uy[uy.len() - 1]);
    let gx: Vec<f64> = (0..n).map(|i| x0 + (x1 - x0) * i as f64 / (n - 1) as f64).collect();
    let gy: Vec<f64> = (0..n).map(|i| y0 + (y1 - y0) * i as f64 / (n - 1) as f64).collect();
    // Distances are measured in units of the data ranges.
    let (sx, sy) = ((x1 - x0).max(1e-300), (y1 - y0).max(1e-300));
    let mut g = vec![vec![0.0; n]; n];
    for (j, &py) in gy.iter().enumerate() {
        for (i, &px) in gx.iter().enumerate() {
            let (mut num, mut den, mut exact) = (0.0, 0.0, None);
            for k in 0..x.len() {
                let d2 = ((px - x[k]) / sx).powi(2) + ((py - y[k]) / sy).powi(2);
                if d2 < 1e-24 {
                    exact = Some(z[k]);
                    break;
                }
                num += z[k] / d2;
                den += 1.0 / d2;
            }
            g[j][i] = exact.unwrap_or(num / den);
        }
    }
    (gx, gy, g, true)
}

fn surface_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (xn, yn, zn, x, y, z) = xyz(ws, v)?;
    if x.len() > 20_000 {
        return Err("surface and contour plots take at most 20000 rows".into());
    }
    let (gx, gy, g, interpolated) = grid_of(&x, &y, &z, 25);
    let series: Vec<Series> =
        gy.iter().zip(&g).map(|(&yy, row)| Series { name: format!("{yy}"), x: gx.clone(), y: row.clone() }).collect();
    let mut s = Section::new("Surface Plot");
    if interpolated {
        s.text("The points do not form a regular grid, so the surface is interpolated (inverse-distance weighting) on a 25 x 25 grid.");
    }
    s.chart(mk(ChartKind::Surface, format!("Surface Plot of {zn} vs {yn}, {xn}"), &xn, &zn, vec![yn], series));
    super::analyses::out(s)
}

fn contour_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (xn, yn, zn, x, y, z) = xyz(ws, v)?;
    if x.len() > 20_000 {
        return Err("surface and contour plots take at most 20000 rows".into());
    }
    let nl = v.num("levels") as usize;
    let (gx, gy, g, interpolated) = grid_of(&x, &y, &z, 40);
    let (lo, hi) = g
        .iter()
        .flatten()
        .filter(|t| t.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |a, &t| (a.0.min(t), a.1.max(t)));
    if (hi - lo).is_nan() || hi - lo <= 1e-9 * hi.abs().max(lo.abs()).max(1.0) {
        return Err("the Z values are constant, so there are no contours".into());
    }
    let mut series = Vec::new();
    for level in nice_levels(lo, hi, nl.max(2)) {
        let (mut px, mut py) = (Vec::new(), Vec::new());
        for line in contour_lines(&gx, &gy, &g, level) {
            if !px.is_empty() {
                px.push(f64::NAN);
                py.push(f64::NAN);
            }
            for (a, b) in line.points {
                px.push(a);
                py.push(b);
            }
        }
        if !px.is_empty() {
            series.push(Series { name: format!("{level:.6}"), x: px, y: py });
        }
    }
    let mut s = Section::new("Contour Plot");
    if interpolated {
        s.text("The points do not form a regular grid, so the surface is interpolated (inverse-distance weighting) on a 40 x 40 grid.");
    }
    s.chart(mk(ChartKind::Contour, format!("Contour Plot of {zn} vs {yn}, {xn}"), &xn, &yn, vec![], series));
    super::analyses::out(s)
}

// ------------------------------------------------------------------ registry

fn by_field(d: DialogSpec) -> DialogSpec {
    d.vars("by", "Group by (optional)", VarKind::Any, 0, 1)
}

pub(crate) fn register(r: &mut Registry) {
    use Page::Options;
    use VarKind::{Any, Numeric};
    let g = |last: &'static str| -> Vec<&'static str> { vec!["Graph", last] };

    add(
        r,
        DialogSpec::new(
            "graph.edit",
            "Edit Graph",
            "Change the title and axis labels of a graph (blank keeps the current text).",
        )
        .text(Page::Main, "graph", "Graph name (blank = the latest graph)", "")
        .text(Page::Main, "title", "Title", "")
        .text(Page::Main, "xlabel", "X axis label", "")
        .text(Page::Main, "ylabel", "Y axis label", ""),
        &g("Edit Graph..."),
        Runner::Report(|_, _| Err("Edit Graph is applied by the workspace, which owns the graphs".into())),
    );
    let owned: fn(&Worksheet, &Values) -> Res<Output> =
        |_, _| Err("this command is applied by the workspace, which owns the graphs".into());
    add(
        r,
        DialogSpec::new(
            "graph.update",
            "Update or Freeze Graph",
            "Graphs drawn from worksheet columns redraw when the data change. Update one now, freeze it, or let it follow the data again.",
        )
        .text(Page::Main, "graph", "Graph name (blank = the latest graph)", "")
        .choice(
            Page::Main,
            "action",
            "Action",
            &["Update now", "Freeze (stop updating)", "Resume automatic updating"],
            0,
        ),
        &g("Update or Freeze Graph..."),
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new(
            "graph.format",
            "Format Graph",
            "Change the look of a graph: font size, grid lines, axis ranges and series colours (blank keeps the current setting).",
        )
        .text(Page::Main, "graph", "Graph name (blank = the latest graph)", "")
        .number(Page::Main, "font", "Font size scale (1 = normal)", 1.0, 0.5, 3.0)
        .choice(Page::Main, "grid", "Grid lines", &["Keep", "Show", "Hide"], 0)
        .optional_number(Page::Main, "xmin", "X axis minimum")
        .optional_number(Page::Main, "xmax", "X axis maximum")
        .optional_number(Page::Main, "ymin", "Y axis minimum")
        .optional_number(Page::Main, "ymax", "Y axis maximum")
        .text(Page::Main, "colors", "Series colours, e.g. #1f77b4, #d62728 (blank = keep)", "")
        .check(Page::Main, "reset", "Reset to the default look first", false),
        &g("Format Graph..."),
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new(
            "graph.annotate",
            "Add Note to Graph",
            "Put a text note at a position given in the graph's data units.",
        )
        .text(Page::Main, "graph", "Graph name (blank = the latest graph)", "")
        .number(Page::Main, "x", "X position", 0.0, -1e300, 1e300)
        .number(Page::Main, "y", "Y position", 0.0, -1e300, 1e300)
        .text(Page::Main, "note", "Note", "")
        .check(Page::Main, "clear", "Remove all notes from the graph instead", false),
        &g("Add Note..."),
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new(
            "graph.brush",
            "Brushing",
            "Drag a rectangle over points of a scatterplot, time series, marginal or bubble plot to brush them; the worksheet and other graphs from the same data highlight the same rows.",
        )
        .choice(
            Page::Main,
            "action",
            "Action",
            &["Clear brushing", "Store brushed rows as an indicator column (1 = brushed)"],
            0,
        ),
        &g("Brushing..."),
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new("graph.bar", "Bar Chart", "Bars for the categories of a variable: counts, sums or means.")
            .vars("cat", "Categorical variable", Any, 1, 1)
            .vars("group", "Group by (optional)", Any, 0, 1)
            .vars("value", "Value variable (for sums and means)", Numeric, 0, 1)
            .choice(Options, "stat", "Bars represent", &["Counts", "Sum of values", "Mean of values"], 0)
            .choice(Options, "multi", "Groups are shown", &["Clustered", "Stacked"], 0)
            .choice(Options, "order", "Order categories", &["Alphabetical or numeric", "Decreasing height"], 0)
            .check(Options, "percent", "Show percent of the total", false),
        &g("Bar Chart..."),
        Runner::Report(bar_run),
    );
    add(
        r,
        DialogSpec::new("graph.pie", "Pie Chart", "Slices for the categories of a variable.")
            .vars("cat", "Categorical variable", Any, 1, 1)
            .vars("value", "Value variable (summed; blank counts rows)", Numeric, 0, 1)
            .optional_number(Options, "combine", "Combine slices smaller than this percent into Other"),
        &g("Pie Chart..."),
        Runner::Report(pie_run),
    );
    add(
        r,
        by_field(
            DialogSpec::new("graph.dotplot", "Dotplot", "Stacked dots for the values of one or more variables.").vars(
                "vars",
                "Graph variables",
                Numeric,
                1,
                8,
            ),
        ),
        &g("Dotplot..."),
        Runner::Report(dot_run),
    );
    add(
        r,
        by_field(
            DialogSpec::new("graph.stem", "Stem-and-Leaf", "Stem-and-leaf display with depths.")
                .vars("vars", "Graph variables", Numeric, 1, 8)
                .text(Options, "unit", "Leaf unit (a power of ten; blank chooses)", ""),
        ),
        &g("Stem-and-Leaf..."),
        Runner::Report(stem_run),
    );
    add(
        r,
        by_field(
            DialogSpec::new("graph.ivp", "Individual Value Plot", "Every observation, by variable or group.").vars(
                "vars",
                "Graph variables",
                Numeric,
                1,
                10,
            ),
        ),
        &g("Individual Value Plot..."),
        Runner::Report(ivp_run),
    );
    add(
        r,
        by_field(
            DialogSpec::new("graph.interval", "Interval Plot", "Confidence intervals for means, by variable or group.")
                .vars("vars", "Graph variables", Numeric, 1, 10)
                .number(Options, "level", "Confidence level (%)", 95.0, 50.0, 99.999),
        ),
        &g("Interval Plot..."),
        Runner::Report(interval_run),
    );
    add(
        r,
        by_field(DialogSpec::new("graph.ecdf", "Empirical CDF", "Empirical cumulative distribution function.").vars(
            "vars",
            "Graph variables",
            Numeric,
            1,
            8,
        )),
        &g("Empirical CDF..."),
        Runner::Report(ecdf_run),
    );
    add(
        r,
        DialogSpec::new("graph.area", "Area Graph", "Filled areas under one or more variables.")
            .vars("vars", "Series", Numeric, 1, 6)
            .vars("x", "X variable (blank uses the row number)", Numeric, 0, 1)
            .check(Options, "stacked", "Stack the series", false),
        &g("Area Graph..."),
        Runner::Report(area_run),
    );
    add(
        r,
        DialogSpec::new(
            "graph.bubble",
            "Bubble Plot",
            "Y against X with marker area proportional to a third variable.",
        )
        .vars("y", "Y variable", Numeric, 1, 1)
        .vars("x", "X variable", Numeric, 1, 1)
        .vars("size", "Size variable", Numeric, 1, 1),
        &g("Bubble Plot..."),
        Runner::Report(bubble_run),
    );
    add(
        r,
        DialogSpec::new("graph.matrix", "Matrix Plot", "Scatterplots of every pair of variables.").vars(
            "vars",
            "Graph variables",
            Numeric,
            2,
            8,
        ),
        &g("Matrix Plot..."),
        Runner::Report(matrix_run),
    );
    add(
        r,
        DialogSpec::new("graph.marginal", "Marginal Plot", "Scatterplot with a histogram of each variable.")
            .vars("y", "Y variable", Numeric, 1, 1)
            .vars("x", "X variable", Numeric, 1, 1),
        &g("Marginal Plot..."),
        Runner::Report(marginal_run),
    );
    add(
        r,
        DialogSpec::new("graph.heatmap", "Heatmap", "Colour-coded values of several columns (rows are observations).")
            .vars("vars", "Variables", Numeric, 2, 60),
        &g("Heatmap..."),
        Runner::Report(heat_run),
    );
    let xyz3 = |d: DialogSpec| {
        d.vars("z", "Z variable", Numeric, 1, 1).vars("y", "Y variable", Numeric, 1, 1).vars(
            "x",
            "X variable",
            Numeric,
            1,
            1,
        )
    };
    add(
        r,
        xyz3(DialogSpec::new("graph.scatter3d", "3D Scatterplot", "Points in three dimensions, coloured by Z.")),
        &g("3D Scatterplot..."),
        Runner::Report(scatter3d_run),
    );
    add(
        r,
        xyz3(DialogSpec::new(
            "graph.surface",
            "3D Surface Plot",
            "Surface of Z over X and Y (interpolated if not a grid).",
        )),
        &g("3D Surface Plot..."),
        Runner::Report(surface_run),
    );
    add(
        r,
        xyz3(DialogSpec::new(
            "graph.contour",
            "Contour Plot",
            "Contour lines of Z over X and Y (interpolated if not a grid).",
        ))
        .integer(Options, "levels", "Approximate number of levels", 8.0, 2.0, 30.0),
        &g("Contour Plot..."),
        Runner::Report(contour_run),
    );
}

pub(super) fn probability_dialog() -> (DialogSpec, Runner) {
    (
        DialogSpec::new(
            "graph.probability",
            "Probability Plot",
            "Probability plot on a normal, lognormal or Weibull axis.",
        )
        .vars("var", "Graph variables", VarKind::Numeric, 1, 1)
        .choice(Page::Options, "dist", "Distribution", &["Normal", "Lognormal", "Weibull"], 0),
        Runner::Report(prob_run),
    )
}
