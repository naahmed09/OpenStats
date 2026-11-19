//! The concrete analyses behind the Stat, Graph, Calc and Data menus. Each binds dialog values
//! to routines of `openstats-stats` and turns the result into a Session-window section.

use super::analysis::{
    col_title, complete_rows, expand, labels, level, numeric, Analysis, DialogError, Output, Registry, Res, Runner,
};
use super::spec::{DialogSpec, Page, Values, VarKind};
use crate::grid::GridController;
use openstats_core::column::{Column, ColumnData};
use openstats_core::Worksheet;
use openstats_stats::anova::{one_way_stacked, OneWayOptions};
use openstats_stats::correlation::{kendall, spearman};
use openstats_stats::descriptive::{describe, describe_by};
use openstats_stats::graphical::{boxplot, graphical_summary, normal_probability_plot, Binning, PlotPosition};
use openstats_stats::hypothesis::{Alternative, VarianceAssumption};
use openstats_stats::model::{Data, DesignOptions};
use openstats_stats::normality::{ks_normal, lilliefors, ryan_joiner, shapiro_wilk};
use openstats_stats::regression::{fit_xy, regress_weighted, RegressionOptions};
use openstats_stats::report::{
    Block, Cell, ChartKind, ChartSpec, Column as RCol, ColumnKind as RKind, Reportable, Section, Series, Table,
};
use openstats_stats::spc::capability::{capability_individuals, capability_subgroups, CapabilityOptions, Specs};
use openstats_stats::spc::charts::subgroups_by_size;
use openstats_stats::ttest::{
    one_sample_t, one_sample_t_summary, paired_t, paired_t_summary, two_sample_t, two_sample_t_summary, z_test,
    z_test_summary,
};

const ALTS: [&str; 3] = ["not equal", "less than", "greater than"];
const MODES: [&str; 2] = ["Samples in columns", "Summarized data"];

fn alt(v: &Values) -> Alternative {
    match v.choice("alt") {
        1 => Alternative::Less,
        2 => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

fn level_field(s: DialogSpec) -> DialogSpec {
    s.number(Page::Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
}

pub(super) fn out(section: Section) -> Res<Output> {
    Ok(Output { section, stored: Vec::new(), models: Vec::new() })
}

fn chart(kind: ChartKind, title: String, xl: &str, yl: &str, series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind,
        title,
        x_label: xl.into(),
        y_label: yl.into(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    }
}

/// Histogram and boxplot charts of one sample, taken from the graphical summary.
fn summary_charts(data: &[f64], name: &str) -> Res<(ChartSpec, ChartSpec)> {
    let g = graphical_summary(data, 0.95, &Binning::Sturges)?;
    let mut charts =
        g.report_section().blocks.into_iter().filter_map(|b| if let Block::Chart(c) = b { Some(c) } else { None });
    let (mut h, mut b) = match (charts.next(), charts.next()) {
        (Some(h), Some(b)) => (h, b),
        _ => return Err(DialogError("the sample is too small to draw a histogram".into())),
    };
    h.title = format!("Histogram of {name}");
    b.title = format!("Boxplot of {name}");
    Ok((h, b))
}

pub(super) fn prob_chart(data: &[f64], name: &str) -> Res<ChartSpec> {
    let pts = normal_probability_plot(data, PlotPosition::Benard)?;
    Ok(chart(
        ChartKind::ProbabilityPlot,
        format!("Probability Plot of {name}"),
        name,
        "Percent",
        vec![Series {
            name: name.into(),
            x: pts.iter().map(|p| p.x).collect(),
            y: pts.iter().map(|p| p.p * 100.0).collect(),
        }],
    ))
}

fn stat_table(title: &str, header: &str) -> Table {
    Table::new(
        title,
        vec![RCol::new(header, RKind::Text), RCol::new("N", RKind::Integer), RCol::new("N*", RKind::Integer)],
    )
}

// --------------------------------------------------------------------------- Basic statistics

fn descriptive(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let by = v.var("by");
    let mut s = Section::new(format!(
        "Descriptive Statistics: {}",
        cols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    ));
    let mut t = stat_table("", "Variable");
    let want: [(&str, bool); 8] = [
        ("Mean", true),
        ("SE Mean", v.flag("semean")),
        ("StDev", true),
        ("Minimum", true),
        ("Q1", true),
        ("Median", true),
        ("Q3", true),
        ("Maximum", true),
    ];
    let mut shown = Vec::new();
    for (name, on) in want {
        if on {
            t.columns.push(RCol::new(name, RKind::Significant(6)));
            shown.push(name);
        }
    }
    if by.is_some() {
        t.columns.insert(1, RCol::new("Group", RKind::Text));
    }
    let mut n_rows = 0;
    let mut add = |name: String, group: Option<String>, d: &openstats_stats::descriptive::Descriptives| {
        let mut row = vec![Cell::text(name)];
        if let Some(g) = group {
            row.push(Cell::text(g));
        }
        row.push(Cell::Num(d.n as f64));
        row.push(Cell::Num(d.n_missing as f64));
        for n in &shown {
            row.push(Cell::opt(match *n {
                "Mean" => d.mean,
                "SE Mean" => d.se_mean,
                "StDev" => d.std_dev,
                "Minimum" => d.minimum,
                "Q1" => d.q1,
                "Median" => d.median,
                "Q3" => d.q3,
                _ => d.maximum,
            }));
        }
        let _ = t.push_row(row);
        n_rows += 1;
    };
    for &c in cols {
        let data = numeric(ws, c)?;
        let name = col_title(ws, c);
        match by {
            Some(b) => {
                let g = labels(ws, b)?;
                for (lvl, d) in describe_by(&data, &g)? {
                    add(name.clone(), Some(lvl), &d);
                }
            }
            None => add(name, None, &describe(&data)?),
        }
        if v.flag("hist") || v.flag("box") {
            let (h, b) = summary_charts(&data, &col_title(ws, c))?;
            if v.flag("hist") {
                s.chart(h);
            }
            if v.flag("box") {
                s.chart(b);
            }
        }
    }
    if n_rows > 0 {
        s.table(t);
    }
    out(s)
}

fn summary_n(v: &Values, key: &str) -> Res<usize> {
    let x = v.num(key);
    if x.is_finite() && x >= 1.0 && x.fract() == 0.0 {
        Ok(x as usize)
    } else {
        Err(DialogError(format!("the sample size ({key}) must be a whole number of at least 1")))
    }
}

fn summary_stat(v: &Values, key: &str) -> Res<f64> {
    let x = v.num(key);
    if x.is_finite() {
        Ok(x)
    } else {
        Err(DialogError(format!("enter a value for {key}")))
    }
}

fn ttest1(ws: &Worksheet, v: &Values) -> Res<Output> {
    if v.choice("mode") == 1 {
        let r = one_sample_t_summary(
            summary_n(v, "n")?,
            summary_stat(v, "mean")?,
            summary_stat(v, "sd")?,
            v.num("mu"),
            alt(v),
            level(v)?,
        )?;
        let mut s = r.report_section();
        s.heading = "One-Sample T (summarized data)".into();
        return out(s);
    }
    let c = v.var("var").ok_or("select a column")?;
    let data = numeric(ws, c)?;
    let r = one_sample_t(&data, v.num("mu"), alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("One-Sample T: {}", col_title(ws, c));
    if v.flag("hist") {
        s.chart(summary_charts(&data, &col_title(ws, c))?.0);
    }
    if v.flag("box") {
        s.chart(summary_charts(&data, &col_title(ws, c))?.1);
    }
    out(s)
}

fn ztest1(ws: &Worksheet, v: &Values) -> Res<Output> {
    let sigma = v.num("sigma");
    if v.choice("mode") == 1 {
        let r = z_test_summary(summary_n(v, "n")?, summary_stat(v, "mean")?, v.num("mu"), sigma, alt(v), level(v)?)?;
        let mut s = r.report_section();
        s.heading = "One-Sample Z (summarized data)".into();
        return out(s);
    }
    let c = v.var("var").ok_or("select a column")?;
    let r = z_test(&numeric(ws, c)?, v.num("mu"), sigma, alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("One-Sample Z: {}", col_title(ws, c));
    out(s)
}

fn ttest2(ws: &Worksheet, v: &Values) -> Res<Output> {
    let var = if v.flag("pooled") { VarianceAssumption::Pooled } else { VarianceAssumption::Welch };
    if v.choice("mode") == 1 {
        let a = (summary_n(v, "n1")?, summary_stat(v, "mean1")?, summary_stat(v, "sd1")?);
        let b = (summary_n(v, "n2")?, summary_stat(v, "mean2")?, summary_stat(v, "sd2")?);
        let r = two_sample_t_summary(a, b, v.num("diff"), var, alt(v), level(v)?)?;
        let mut s = r.report_section();
        s.heading = "Two-Sample T (summarized data)".into();
        return out(s);
    }
    let (a, b) = (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
    let (x, y) = (numeric(ws, a)?, numeric(ws, b)?);
    let r = two_sample_t(&x, &y, v.num("diff"), var, alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Two-Sample T: {} vs {}", col_title(ws, a), col_title(ws, b));
    if v.flag("box") {
        let bx = |d: &[f64], n: String| -> Res<openstats_stats::report::Series> {
            let b = boxplot(d)?;
            Ok(Series { name: n, x: vec![], y: vec![b.lower_whisker, b.q1, b.median, b.q3, b.upper_whisker] })
        };
        s.chart(chart(
            ChartKind::Boxplot,
            format!("Boxplot of {}, {}", col_title(ws, a), col_title(ws, b)),
            "",
            "Data",
            vec![bx(&x, col_title(ws, a))?, bx(&y, col_title(ws, b))?],
        ));
    }
    out(s)
}

fn paired(ws: &Worksheet, v: &Values) -> Res<Output> {
    if v.choice("mode") == 1 {
        let r = paired_t_summary(
            summary_n(v, "n")?,
            summary_stat(v, "mean")?,
            summary_stat(v, "sd")?,
            v.num("diff"),
            alt(v),
            level(v)?,
        )?;
        let mut s = r.report_section();
        s.heading = "Paired T (summarized data)".into();
        return out(s);
    }
    let (a, b) = (v.var("first").ok_or("select the first sample")?, v.var("second").ok_or("select the second sample")?);
    let r = paired_t(&numeric(ws, a)?, &numeric(ws, b)?, v.num("diff"), alt(v), level(v)?)?;
    let mut s = r.report_section();
    s.heading = format!("Paired T: {} - {}", col_title(ws, a), col_title(ws, b));
    out(s)
}

fn normality(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let data = numeric(ws, c)?;
    let name = col_title(ws, c);
    let mut s = match v.choice("test") {
        0 => openstats_stats::graphical::anderson_darling(&data)?.report_section(),
        1 => ryan_joiner(&data)?.report_section(),
        2 => shapiro_wilk(&data)?.report_section(),
        3 => lilliefors(&data)?.report_section(),
        _ => {
            let d = describe(&data)?;
            ks_normal(&data, d.mean.ok_or("too few values")?, d.std_dev.ok_or("too few values")?)?.report_section()
        }
    };
    s.heading = format!("Normality Test: {name}");
    s.chart(prob_chart(&data, &name)?);
    out(s)
}

fn correlation(ws: &Worksheet, v: &Values) -> Res<Output> {
    let cols = v.vars("vars");
    let data: Vec<Vec<f64>> = cols.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let names: Vec<String> = cols.iter().map(|&c| col_title(ws, c)).collect();
    let mut s = Section::new(format!("Correlations: {}", names.join(", ")));
    let kind = v.choice("method");
    let mut t = Table::new(
        match kind {
            0 => "Pearson correlation",
            1 => "Spearman rho",
            _ => "Kendall tau",
        },
        vec![
            RCol::new("Pair", RKind::Text),
            RCol::new("N", RKind::Integer),
            RCol::new("Correlation", RKind::Fixed(3)),
            RCol::new("P-Value", RKind::PValue),
        ],
    );
    for i in 0..cols.len() {
        for j in 0..i {
            let (r, n, p) = match kind {
                0 => {
                    let r = openstats_stats::correlation::pearson(&data[j], &data[i], level(v)?)?;
                    (r.r, r.n, r.p_value)
                }
                1 => {
                    let r = spearman(&data[j], &data[i])?;
                    (r.rho, r.n, r.p_value)
                }
                _ => {
                    let r = kendall(&data[j], &data[i])?;
                    (r.tau_b, r.n, Some(r.p_value))
                }
            };
            t.push_row(vec![
                Cell::text(format!("{} * {}", names[j], names[i])),
                Cell::Num(n as f64),
                Cell::Num(r),
                Cell::opt(p),
            ])
            .map_err(DialogError::from)?;
        }
    }
    s.table(t);
    if v.flag("scatter") {
        for i in 0..cols.len() {
            for j in 0..i {
                let (_, d) = complete_rows(&[&data[j], &data[i]]);
                s.chart(chart(
                    ChartKind::Scatter,
                    format!("Scatterplot of {} vs {}", names[i], names[j]),
                    &names[j],
                    &names[i],
                    vec![Series { name: names[i].clone(), x: d[0].clone(), y: d[1].clone() }],
                ));
            }
        }
    }
    out(s)
}

fn graphical(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let data = numeric(ws, c)?;
    let g = graphical_summary(&data, level(v)?, &Binning::Sturges)?;
    let mut s = g.report_section();
    s.heading = format!("Graphical Summary: {}", col_title(ws, c));
    out(s)
}

// --------------------------------------------------------------------------- Regression, ANOVA

fn regression(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let preds = v.vars("preds");
    if preds.contains(&y) {
        return Err("the response cannot also be a predictor".into());
    }
    let mut ycol = numeric(ws, y)?;
    let mut note = None;
    let tr = v.choice("transform");
    if tr > 0 {
        use openstats_stats::spc::transform::{BoxCox, LambdaChoice};
        let choice = match tr {
            1 => LambdaChoice::Optimal,
            2 => LambdaChoice::Rounded,
            _ => {
                let l = v.num("lambda");
                if !l.is_finite() {
                    return Err("enter the Box-Cox lambda".into());
                }
                LambdaChoice::Fixed(l)
            }
        };
        let bc = BoxCox::fit(&ycol, choice)?;
        ycol = ycol.iter().map(|&x| bc.forward(x)).collect();
        note = Some(format!(
            "The response was Box-Cox transformed with lambda = {:.4} (optimal lambda {:.4}); fits and residuals are in transformed units.",
            bc.lambda, bc.optimal_lambda
        ));
    }
    let xcols: Vec<Vec<f64>> = preds.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let wcol = match v.var("weights") {
        Some(w) if w == y || preds.contains(&w) => return Err("the weights must be a separate column".into()),
        Some(w) => Some(numeric(ws, w)?),
        None => None,
    };
    let mut all: Vec<&[f64]> = vec![&ycol];
    all.extend(xcols.iter().map(Vec::as_slice));
    if let Some(w) = &wcol {
        all.push(w);
    }
    let (rows, data) = complete_rows(&all);
    let names: Vec<String> = preds.iter().map(|&c| col_title(ws, c)).collect();
    let o = RegressionOptions { level: level(v)?, ..Default::default() };
    let fit = if wcol.is_some() {
        let wv = &data[data.len() - 1];
        if wv.iter().any(|&x| x <= 0.0) {
            return Err("the weights must be positive".into());
        }
        let ids: Vec<String> = (0..preds.len()).map(|i| format!("x{}_", i + 1)).collect();
        let mut d = Data::new().with_numeric("y_", data[0].clone())?;
        for (i, id) in ids.iter().enumerate() {
            d = d.with_numeric(id, data[i + 1].clone())?;
        }
        let formula = format!("y_ ~ {}{}", ids.join(" + "), if v.flag("intercept") { "" } else { " - 1" });
        let mut f = regress_weighted(&formula, &d, &DesignOptions::default(), wv, &o)?;
        for (c, n) in f
            .coefficients
            .iter_mut()
            .zip(std::iter::once("Constant".to_string()).filter(|_| v.flag("intercept")).chain(names.iter().cloned()))
        {
            c.name = n;
        }
        f
    } else {
        let cols: Vec<(&str, &[f64])> = names.iter().zip(&data[1..]).map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        fit_xy(&cols, &data[0], v.flag("intercept"), &o)?
    };
    let mut s = fit.report_section();
    s.heading = format!("Regression Analysis: {} versus {}", col_title(ws, y), names.join(", "));
    if let Some(w) = v.var("weights") {
        s.text(format!("Weighted least squares with weights {}.", col_title(ws, w)));
    }
    if let Some(n) = note {
        s.text(n);
    }
    if !v.flag("residual_plots") {
        s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    }
    let mut stored = Vec::new();
    let n = ycol.len();
    if v.flag("store_fits") {
        stored.push(expand("FITS", n, &rows, &fit.diagnostics.fitted));
    }
    if v.flag("store_resid") {
        stored.push(expand("RESI", n, &rows, &fit.diagnostics.residuals));
    }
    if v.flag("store_stdres") {
        stored.push(expand("SRES", n, &rows, &fit.diagnostics.standardized));
    }
    let d = &fit.diagnostics;
    for (flag, name, vals) in [
        ("store_delres", "TRES", &d.deleted),
        ("store_hi", "HI", &d.leverage),
        ("store_cook", "COOK", &d.cooks_d),
        ("store_dffits", "DFIT", &d.dffits),
        ("store_press", "PRES", &d.press_residuals),
    ] {
        if v.flag(flag) {
            stored.push(expand(name, n, &rows, vals));
        }
    }
    if v.flag("store_coef") {
        let est = fit.estimates();
        let idx: Vec<usize> = (0..est.len()).collect();
        stored.push(expand("COEF", est.len(), &idx, &est));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn oneway(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, f) = (v.var("response").ok_or("select a response")?, v.var("factor").ok_or("select a factor")?);
    let (data, factor) = (numeric(ws, y)?, labels(ws, f)?);
    let o = OneWayOptions { level: level(v)? };
    let a = one_way_stacked(&data, &factor, &o)?;
    let mut s = a.report_section();
    s.heading = format!("One-Way ANOVA: {} versus {}", col_title(ws, y), col_title(ws, f));
    let alpha = 1.0 - o.level;
    if v.flag("tukey") {
        for b in a.tukey(alpha)?.report_section().blocks {
            push_block(&mut s, b);
        }
    }
    if v.flag("fisher") {
        for b in a.fisher(alpha)?.report_section().blocks {
            push_block(&mut s, b);
        }
    }
    if v.flag("interval") {
        s.chart(a.interval_plot());
    }
    if v.flag("ivp") {
        s.chart(a.individual_value_plot());
    }
    out(s)
}

fn push_block(s: &mut Section, b: Block) {
    s.blocks.push(b);
}

// --------------------------------------------------------------------------- Quality

fn capability(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = v.var("var").ok_or("select a column")?;
    let specs = Specs {
        lsl: Some(v.num("lsl")).filter(|x| x.is_finite()),
        target: Some(v.num("target")).filter(|x| x.is_finite()),
        usl: Some(v.num("usl")).filter(|x| x.is_finite()),
    };
    if specs.lsl.is_none() && specs.usl.is_none() {
        return Err("enter a lower or an upper specification limit".into());
    }
    let o = CapabilityOptions { level: level(v)?, ..Default::default() };
    let data = numeric(ws, c)?;
    let size = v.count("size");
    let mut groups = None;
    let r = if size <= 1 {
        capability_individuals(&data, &specs, &o)?
    } else {
        let g = subgroups_by_size(&data, size)?;
        let r = capability_subgroups(&g, &specs, &o)?;
        groups = Some(g);
        r
    };
    let mut s = r.report_section();
    if v.flag("intervals") {
        let extra = super::analyses_qual::capability_intervals_section(&r, groups.as_deref(), o.level)?;
        s.blocks.extend(extra.blocks);
        s.warnings.extend(extra.warnings);
    }
    s.heading = format!("Process Capability of {}", col_title(ws, c));
    out(s)
}

// --------------------------------------------------------------------------- DOE

// --------------------------------------------------------------------------- Graph menu

fn graph_hist(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut s = Section::new("Histogram");
    for &c in v.vars("vars") {
        s.chart(summary_charts(&numeric(ws, c)?, &col_title(ws, c))?.0);
    }
    out(s)
}

fn graph_box(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut s = Section::new("Boxplot");
    let cols = v.vars("vars");
    let mut series = Vec::new();
    for &c in cols {
        let b = boxplot(&numeric(ws, c)?)?;
        series.push(Series {
            name: col_title(ws, c),
            x: vec![],
            y: vec![b.lower_whisker, b.q1, b.median, b.q3, b.upper_whisker],
        });
    }
    s.chart(chart(
        ChartKind::Boxplot,
        format!("Boxplot of {}", cols.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")),
        "",
        "Data",
        series,
    ));
    out(s)
}

fn graph_scatter(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (y, x) = (v.var("y").ok_or("select Y")?, v.var("x").ok_or("select X")?);
    let (yd, xd) = (numeric(ws, y)?, numeric(ws, x)?);
    let (_, d) = complete_rows(&[&xd, &yd]);
    let mut s = Section::new("Scatterplot");
    let kind = if v.flag("connect") { ChartKind::Line } else { ChartKind::Scatter };
    s.chart(chart(
        kind,
        format!("Scatterplot of {} vs {}", col_title(ws, y), col_title(ws, x)),
        &col_title(ws, x),
        &col_title(ws, y),
        vec![Series { name: col_title(ws, y), x: d[0].clone(), y: d[1].clone() }],
    ));
    out(s)
}

fn graph_series(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut s = Section::new("Time Series Plot");
    let (kind, x, xlabel) = match v.var("time") {
        Some(t) => {
            let col = ws.column(t)?;
            let x: Vec<f64> = match &col.data {
                ColumnData::DateTime(d) => d
                    .iter()
                    .map(|&m| {
                        if m == openstats_core::missing::DATETIME_MISSING {
                            f64::NAN
                        } else {
                            m as f64 / openstats_core::datetime::MICROS_PER_DAY as f64 + 25569.0
                        }
                    })
                    .collect(),
                _ => numeric(ws, t)?,
            };
            let is_date = matches!(col.data, ColumnData::DateTime(_));
            (if is_date { ChartKind::TimeLine } else { ChartKind::Line }, x, col_title(ws, t))
        }
        None => (ChartKind::Line, Vec::new(), "Index".to_string()),
    };
    for &c in v.vars("vars") {
        let d = numeric(ws, c)?;
        let xs: Vec<f64> = if x.is_empty() { (1..=d.len()).map(|i| i as f64).collect() } else { x.clone() };
        let n = xs.len().min(d.len());
        let (xs, d): (Vec<f64>, Vec<f64>) =
            xs.iter().zip(&d).take(n).filter(|(a, _)| a.is_finite()).map(|(a, b)| (*a, *b)).unzip();
        s.chart(chart(
            kind,
            format!("Time Series Plot of {}", col_title(ws, c)),
            &xlabel,
            &col_title(ws, c),
            vec![Series { name: col_title(ws, c), x: xs, y: d }],
        ));
    }
    out(s)
}

// --------------------------------------------------------------------------- Calc and Data

fn standardize(ws: &Worksheet, v: &Values) -> Res<Output> {
    let mut stored = Vec::new();
    for &c in v.vars("vars") {
        let d = numeric(ws, c)?;
        let ds = describe(&d)?;
        let (m, sd) = (ds.mean.ok_or("too few values")?, ds.std_dev.ok_or("too few values")?);
        if sd == 0.0 {
            return Err(DialogError(format!("{} is constant; it cannot be standardized", col_title(ws, c))));
        }
        stored.push(Column::numeric(format!("Z_{}", col_title(ws, c)), d.iter().map(|x| (x - m) / sd).collect()));
    }
    let mut s = Section::new("Standardize");
    s.text("Each column was centred on its mean and divided by its standard deviation.");
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn calculator(g: &mut GridController, v: &Values) -> Res<String> {
    let target = v.count("target");
    if target == 0 {
        return Err("enter the number of the column to store the result in (1, 2, …)".into());
    }
    let expr = v.text("expr").trim().to_string();
    if expr.is_empty() {
        return Err("enter an expression".into());
    }
    g.calculate(target - 1, &expr)?;
    Ok(format!("Calculator: C{target} = {expr}"))
}

fn sort_data(g: &mut GridController, v: &Values) -> Res<String> {
    let key = v.var("by").ok_or("select a column to sort by")?;
    let name = col_title(g.worksheet(), key);
    g.sort_rows(key, v.choice("order") == 1)?;
    Ok(format!("Sorted all rows by {name} ({}).", if v.choice("order") == 1 { "descending" } else { "ascending" }))
}

// --------------------------------------------------------------------------- registration

pub(super) fn add(r: &mut Registry, spec: DialogSpec, menu: &[&'static str], runner: Runner) {
    r.add(Analysis { spec, menu: menu.to_vec(), runner });
}

/// Register every analysis.
pub(crate) fn register_all(r: &mut Registry) {
    use Page::{Graphs, Options, Storage};
    use VarKind::{Any, Numeric};
    let basic = ["Stat", "Basic Statistics"];
    let p = |path: &[&'static str], last: &'static str| -> Vec<&'static str> {
        let mut v = path.to_vec();
        v.push(last);
        v
    };

    add(
        r,
        DialogSpec::new(
            "stat.basic.descriptive",
            "Display Descriptive Statistics",
            "Summary statistics for one or more numeric columns, optionally by group.",
        )
        .vars("vars", "Variables", Numeric, 1, 50)
        .vars("by", "By variable (optional)", Any, 0, 1)
        .check(Options, "semean", "Standard error of mean", true)
        .check(Graphs, "hist", "Histogram of data", false)
        .check(Graphs, "box", "Boxplot of data", false),
        &p(&basic, "Display Descriptive Statistics..."),
        Runner::Report(descriptive),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.basic.ttest1",
                "1-Sample t",
                "Test whether the mean of a column differs from a hypothesized value.",
            )
            .choice(Page::Main, "mode", "Data are", &MODES, 0)
            .vars("var", "Samples in column", Numeric, 1, 1)
            .vars_optional_when("var", "mode", 1)
            .integer(Page::Main, "n", "Sample size (summarized data)", 10.0, 1.0, 1e9)
            .number(Page::Main, "mean", "Sample mean (summarized data)", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .number(Page::Main, "sd", "Standard deviation (summarized data)", 1.0, 0.0, f64::INFINITY)
            .number(Options, "mu", "Test mean", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .choice(Options, "alt", "Alternative", &ALTS, 0)
            .check(Graphs, "hist", "Histogram of data", false)
            .check(Graphs, "box", "Boxplot of data", false),
        ),
        &p(&basic, "1-Sample t..."),
        Runner::Report(ttest1),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.basic.ztest1",
                "1-Sample Z",
                "Test whether the mean differs from a hypothesized value when the population standard deviation is known.",
            )
            .choice(Page::Main, "mode", "Data are", &MODES, 0)
            .vars("var", "Samples in column", Numeric, 1, 1)
            .vars_optional_when("var", "mode", 1)
            .integer(Page::Main, "n", "Sample size (summarized data)", 10.0, 1.0, 1e9)
            .number(Page::Main, "mean", "Sample mean (summarized data)", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .number(Page::Main, "sigma", "Known standard deviation", 1.0, 1e-300, f64::INFINITY)
            .number(Options, "mu", "Test mean", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .choice(Options, "alt", "Alternative", &ALTS, 0),
        ),
        &p(&basic, "1-Sample Z..."),
        Runner::Report(ztest1),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.basic.ttest2",
                "2-Sample t",
                "Compare the means of two samples stored in separate columns.",
            )
            .choice(Page::Main, "mode", "Data are", &MODES, 0)
            .vars("first", "First", Numeric, 1, 1)
            .vars("second", "Second", Numeric, 1, 1)
            .vars_optional_when("first", "mode", 1)
            .vars_optional_when("second", "mode", 1)
            .integer(Page::Main, "n1", "Sample 1 size (summarized data)", 10.0, 1.0, 1e9)
            .number(Page::Main, "mean1", "Sample 1 mean (summarized data)", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .number(Page::Main, "sd1", "Sample 1 standard deviation (summarized data)", 1.0, 0.0, f64::INFINITY)
            .integer(Page::Main, "n2", "Sample 2 size (summarized data)", 10.0, 1.0, 1e9)
            .number(Page::Main, "mean2", "Sample 2 mean (summarized data)", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .number(Page::Main, "sd2", "Sample 2 standard deviation (summarized data)", 1.0, 0.0, f64::INFINITY)
            .check(Options, "pooled", "Assume equal variances", false)
            .number(Options, "diff", "Test difference", 0.0, f64::NEG_INFINITY, f64::INFINITY)
            .choice(Options, "alt", "Alternative", &ALTS, 0)
            .check(Graphs, "box", "Boxplots of data", false),
        ),
        &p(&basic, "2-Sample t..."),
        Runner::Report(ttest2),
    );
    add(
        r,
        level_field(
            DialogSpec::new("stat.basic.paired", "Paired t", "Test the mean difference of paired observations.")
                .choice(Page::Main, "mode", "Data are", &MODES, 0)
                .vars("first", "Sample 1", Numeric, 1, 1)
                .vars("second", "Sample 2", Numeric, 1, 1)
                .vars_optional_when("first", "mode", 1)
                .vars_optional_when("second", "mode", 1)
                .integer(Page::Main, "n", "Number of pairs (summarized data)", 10.0, 1.0, 1e9)
                .number(
                    Page::Main,
                    "mean",
                    "Mean of the differences (summarized data)",
                    0.0,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                )
                .number(
                    Page::Main,
                    "sd",
                    "Standard deviation of the differences (summarized data)",
                    1.0,
                    0.0,
                    f64::INFINITY,
                )
                .number(Options, "diff", "Test mean difference", 0.0, f64::NEG_INFINITY, f64::INFINITY)
                .choice(Options, "alt", "Alternative", &ALTS, 0),
        ),
        &p(&basic, "Paired t..."),
        Runner::Report(paired),
    );
    add(
        r,
        DialogSpec::new(
            "stat.basic.normality",
            "Normality Test",
            "Test whether a column is consistent with a normal distribution.",
        )
        .vars("var", "Variable", Numeric, 1, 1)
        .choice(
            Options,
            "test",
            "Tests",
            &["Anderson-Darling", "Ryan-Joiner", "Shapiro-Wilk", "Lilliefors", "Kolmogorov-Smirnov"],
            0,
        ),
        &p(&basic, "Normality Test..."),
        Runner::Report(normality),
    );
    add(
        r,
        level_field(
            DialogSpec::new("stat.basic.correlation", "Correlation", "Correlation between pairs of numeric columns.")
                .vars("vars", "Variables", Numeric, 2, 50)
                .choice(Options, "method", "Method", &["Pearson", "Spearman", "Kendall"], 0)
                .check(Graphs, "scatter", "Scatterplot for each pair", false),
        ),
        &p(&basic, "Correlation..."),
        Runner::Report(correlation),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.basic.graphsummary",
                "Graphical Summary",
                "Descriptive statistics, intervals and plots for one column.",
            )
            .vars("var", "Variables", Numeric, 1, 1),
        ),
        &p(&basic, "Graphical Summary..."),
        Runner::Report(graphical),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.regression.fit",
                "Regression",
                "Least-squares regression of a response on numeric predictors.",
            )
            .vars("response", "Response", Numeric, 1, 1)
            .vars("preds", "Predictors", Numeric, 1, 50)
            .vars("weights", "Weights (optional)", Numeric, 0, 1)
            .check(Options, "intercept", "Fit intercept", true)
            .choice(
                Options,
                "transform",
                "Transform the response",
                &["None", "Box-Cox (optimal lambda)", "Box-Cox (rounded lambda)", "Box-Cox (lambda given)"],
                0,
            )
            .optional_number(Options, "lambda", "Box-Cox lambda (for the given-lambda choice)")
            .check(Graphs, "residual_plots", "Residual plots", false)
            .check(Storage, "store_fits", "Fits", false)
            .check(Storage, "store_resid", "Residuals", false)
            .check(Storage, "store_stdres", "Standardized residuals", false)
            .check(Storage, "store_delres", "Deleted (studentized) residuals", false)
            .check(Storage, "store_hi", "Leverages (hi)", false)
            .check(Storage, "store_cook", "Cook's distance", false)
            .check(Storage, "store_dffits", "DFITS", false)
            .check(Storage, "store_press", "PRESS residuals", false)
            .check(Storage, "store_coef", "Coefficients", false),
        ),
        &["Stat", "Regression", "Regression..."],
        Runner::Report(regression),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.anova.oneway",
                "One-Way ANOVA",
                "Compare the means of several groups (response in one column, group labels in another).",
            )
            .vars("response", "Response", Numeric, 1, 1)
            .vars("factor", "Factor", Any, 1, 1)
            .check(Options, "tukey", "Tukey comparisons", true)
            .check(Options, "fisher", "Fisher comparisons", false)
            .check(Graphs, "interval", "Interval plot", false)
            .check(Graphs, "ivp", "Individual value plot", false),
        ),
        &["Stat", "ANOVA", "One-Way..."],
        Runner::Report(oneway),
    );
    let ctrl = |s: DialogSpec| {
        s.number(Options, "k", "Sigma limit position", 3.0, 0.5, 10.0).check(
            Options,
            "all_tests",
            "Perform all tests for special causes",
            false,
        )
    };
    add(
        r,
        super::analyses_qual::ctrl_extra(
            ctrl(DialogSpec::new("stat.spc.imr", "I-MR Chart", "Individuals and moving-range control chart.").vars(
                "var",
                "Variables",
                Numeric,
                1,
                1,
            )),
            super::analyses_qual::Kind::Individuals,
        ),
        &["Stat", "Control Charts", "Variables Charts for Individuals", "I-MR..."],
        Runner::Report(super::analyses_qual::imr_run),
    );
    add(
        r,
        super::analyses_qual::ctrl_extra(
            ctrl(
                DialogSpec::new(
                    "stat.spc.xbar",
                    "Xbar Chart",
                    "Subgroup means with a range or standard-deviation chart.",
                )
                .vars("var", "Observations in one column", Numeric, 1, 1)
                .vars("ids", "Subgroup identifiers (column; overrides the size)", Any, 0, 1)
                .integer(Page::Main, "size", "Subgroup size", 5.0, 2.0, 100.0)
                .choice(
                    Options,
                    "sigma",
                    "Estimate sigma from",
                    &["Subgroup ranges", "Subgroup standard deviations"],
                    0,
                ),
            ),
            super::analyses_qual::Kind::Subgroup,
        ),
        &["Stat", "Control Charts", "Variables Charts for Subgroups", "Xbar-R / Xbar-S..."],
        Runner::Report(super::analyses_qual::xbar_run),
    );
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.quality.capability",
                "Capability Analysis (Normal)",
                "Process capability and performance indices against specification limits.",
            )
            .vars("var", "Single column", Numeric, 1, 1)
            .integer(Page::Main, "size", "Subgroup size (1 = individuals)", 1.0, 1.0, 100.0)
            .optional_number(Page::Main, "lsl", "Lower spec")
            .optional_number(Page::Main, "usl", "Upper spec")
            .optional_number(Options, "target", "Target")
            .check(Options, "intervals", "Confidence intervals for the indices", false),
        ),
        &["Stat", "Quality Tools", "Capability Analysis", "Normal..."],
        Runner::Report(capability),
    );
    add(
        r,
        super::analyses_doe::extend_factorial(
            DialogSpec::new(
                "stat.doe.factorial",
                "Create Factorial Design",
                "Create a two-level full factorial design and store it in the worksheet.",
            )
            .integer(Page::Main, "factors", "Number of factors", 3.0, 2.0, 15.0)
            .integer(Page::Main, "replicates", "Number of replicates", 1.0, 1.0, 100.0)
            .integer(Page::Main, "center", "Number of center points", 0.0, 0.0, 100.0)
            .check(Options, "randomize", "Randomize runs", false)
            .integer(Options, "seed", "Base for random data", 1.0, 0.0, 1e9)
            .check(Storage, "store", "Store design in worksheet", true),
        ),
        &["Stat", "DOE", "Factorial", "Create Factorial Design..."],
        Runner::Report(super::analyses_doe::factorial_run),
    );
    add(
        r,
        DialogSpec::new("graph.histogram", "Histogram", "Histogram of one or more columns.").vars(
            "vars",
            "Graph variables",
            Numeric,
            1,
            10,
        ),
        &["Graph", "Histogram..."],
        Runner::Report(graph_hist),
    );
    add(
        r,
        DialogSpec::new("graph.boxplot", "Boxplot", "Boxplots of one or more columns.").vars(
            "vars",
            "Graph variables",
            Numeric,
            1,
            10,
        ),
        &["Graph", "Boxplot..."],
        Runner::Report(graph_box),
    );
    add(
        r,
        DialogSpec::new("graph.scatter", "Scatterplot", "Plot of Y against X.")
            .vars("y", "Y variable", Numeric, 1, 1)
            .vars("x", "X variable", Numeric, 1, 1)
            .check(Options, "connect", "Connect points", false),
        &["Graph", "Scatterplot..."],
        Runner::Report(graph_scatter),
    );
    {
        let (spec, runner) = super::analyses_graph::probability_dialog();
        add(r, spec, &["Graph", "Probability Plot..."], runner);
    }
    add(
        r,
        DialogSpec::new(
            "graph.timeseries",
            "Time Series Plot",
            "Plot of values against row number or a date/time column.",
        )
        .vars("vars", "Series", Numeric, 1, 10)
        .vars("time", "Time scale: date/time or numeric column (blank = row number)", Numeric, 0, 1),
        &["Graph", "Time Series Plot..."],
        Runner::Report(graph_series),
    );
    add(
        r,
        DialogSpec::new("calc.calculator", "Calculator", "Evaluate an expression and store the values in a column.")
            .integer(Page::Main, "target", "Store result in column number", 0.0, 0.0, 4000.0)
            .text(Page::Main, "expr", "Expression", ""),
        &["Calc", "Calculator..."],
        Runner::Mutate(calculator),
    );
    add(
        r,
        DialogSpec::new(
            "calc.standardize",
            "Standardize",
            "Center and scale columns to z-scores, stored in new columns.",
        )
        .vars("vars", "Input column(s)", Numeric, 1, 50),
        &["Calc", "Standardize..."],
        Runner::Report(standardize),
    );
    add(
        r,
        DialogSpec::new("data.sort", "Sort", "Sort all rows by one column.")
            .vars("by", "Sort by column", Any, 1, 1)
            .choice(Options, "order", "Order", &["Ascending", "Descending"], 0),
        &["Data", "Sort..."],
        Runner::Mutate(sort_data),
    );
    super::analyses_adv::register(r);
    super::analyses_ml::register(r);
    super::analyses_ml2::register(r);
    super::analyses_graph::register(r);
    super::analyses_ts::register(r);
    super::analyses_data2::register(r);
    super::analyses_tables::register(r);
    super::analyses_qual2::register(r);
    super::analyses_basic::register(r);
    super::analyses_np::register(r);
    super::analyses_mv::register(r);
    super::analyses_rel::register(r);
    super::analyses_rel2::register(r);
    super::analyses_power::register(r);
    super::analyses_reg::register(r);
    super::analyses_qual::register(r);
    super::analyses_doe::register(r);
    super::analyses_doe2::register(r);
    super::analyses_data::register(r);
    super::analyses_regfam::register(r);
    super::analyses_anova::register(r);
    super::analyses_workspace::register(r);
}
