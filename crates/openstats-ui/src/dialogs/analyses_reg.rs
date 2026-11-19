//! Regression and ANOVA extras: stepwise regression, best subsets, prediction at new values, the
//! fitted line plot and pairwise comparisons of general-linear-model least-squares means.

use super::analyses::{add, out};
use super::analyses_adv::ident;
use super::analysis::{
    col_title, complete_rows, expand, labels, level, numeric, DialogError, Output, Registry, Res, Runner,
};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::Worksheet;
use openstats_stats::anova::{
    dunnett, fisher, glm, hsu_mcb, tukey, Alternative as DunnettSide, Best, GlmOptions, Means,
};
use openstats_stats::model::{Data, DesignOptions};
use openstats_stats::regression::{
    best_subsets, regress, stepwise, RegressionOptions, StepCriterion, StepMethod, StepwiseOptions, SubsetOptions,
};
use openstats_stats::report::{Block, Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Series, Table};

const WARN_APPROX: &str =
    "The standard errors of the differences are derived from the standard errors of the least-squares means, \
which makes the comparisons exact for balanced designs without covariates and approximate otherwise.";

fn push(t: &mut Table, row: Vec<Cell>) -> Res<()> {
    t.push_row(row).map_err(DialogError::from)
}

/// Columns of the worksheet as formula-safe names and a [`Data`] set (missing values stay `NaN`).
fn numeric_data(ws: &Worksheet, response: usize, preds: &[usize]) -> Res<(Data, String, Vec<String>)> {
    let mut taken: Vec<String> = Vec::new();
    let yn = ident(ws, response, &taken);
    taken.push(yn.clone());
    let mut data = Data::new().with_numeric(&yn, numeric(ws, response)?)?;
    let mut names = Vec::new();
    for &c in preds {
        let n = ident(ws, c, &taken);
        taken.push(n.clone());
        data = data.with_numeric(&n, numeric(ws, c)?)?;
        names.push(n);
    }
    Ok((data, yn, names))
}

fn check_response(y: usize, preds: &[usize]) -> Res<()> {
    if preds.contains(&y) {
        return Err("the response cannot also be a predictor".into());
    }
    Ok(())
}

// --------------------------------------------------------------------------------- stepwise

fn stepwise_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let preds = v.vars("preds").to_vec();
    let force = v.vars("force").to_vec();
    check_response(y, &preds)?;
    if let Some(&f) = force.iter().find(|f| !preds.contains(f)) {
        return Err(format!("{} is forced into the model but is not a predictor", col_title(ws, f)).into());
    }
    let (data, yn, names) = numeric_data(ws, y, &preds)?;
    let method = match v.choice("method") {
        1 => StepMethod::Forward,
        2 => StepMethod::Backward,
        _ => StepMethod::Stepwise,
    };
    let mut opts = StepwiseOptions::new(method);
    let (def_in, def_out) = match StepCriterion::default_for(method) {
        StepCriterion::PValue { alpha_enter, alpha_remove } => (alpha_enter, alpha_remove),
        _ => (0.15, 0.15),
    };
    opts.criterion = match v.choice("criterion") {
        1 => StepCriterion::Aic,
        2 => StepCriterion::Aicc,
        3 => StepCriterion::Bic,
        _ => {
            let pick = |x: f64, d: f64| if x.is_finite() { x } else { d };
            let (a_in, a_out) = (pick(v.num("alpha_enter"), def_in), pick(v.num("alpha_remove"), def_out));
            if !(a_in > 0.0 && a_in < 1.0 && a_out > 0.0 && a_out < 1.0) {
                return Err("alpha to enter and to remove must be between 0 and 1".into());
            }
            StepCriterion::PValue { alpha_enter: a_in, alpha_remove: a_out }
        }
    };
    opts.force_in = force.iter().filter_map(|f| preds.iter().position(|p| p == f)).map(|i| names[i].clone()).collect();
    let ropts = RegressionOptions { level: level(v)?, ..Default::default() };
    let rhs = names.join(" + ");
    let formula = format!("{yn} ~ {rhs}");
    let r = stepwise(&formula, &data, &DesignOptions::default(), &opts, &ropts)?;
    let mut s = r.report_section();
    s.heading = format!(
        "Stepwise Regression: {} versus {}",
        col_title(ws, y),
        preds.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    );
    if !v.flag("residual_plots") {
        s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    }
    let n = ws.column(y)?.len();
    let mut stored = Vec::new();
    let rows = r.fit.rows_used().to_vec();
    if v.flag("store_fits") {
        stored.push(expand("SWFITS", n, &rows, &r.fit.diagnostics.fitted));
    }
    if v.flag("store_resid") {
        stored.push(expand("SWRESI", n, &rows, &r.fit.diagnostics.residuals));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------- best subsets

fn subsets_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let preds = v.vars("preds").to_vec();
    let force = v.vars("force").to_vec();
    check_response(y, &preds)?;
    if let Some(&f) = force.iter().find(|f| !preds.contains(f)) {
        return Err(format!("{} is forced into the models but is not a predictor", col_title(ws, f)).into());
    }
    let ycol = numeric(ws, y)?;
    let xcols: Vec<Vec<f64>> = preds.iter().map(|&c| numeric(ws, c)).collect::<Res<_>>()?;
    let mut all: Vec<&[f64]> = vec![&ycol];
    all.extend(xcols.iter().map(Vec::as_slice));
    let (_, data) = complete_rows(&all);
    let names: Vec<String> = preds.iter().map(|&c| col_title(ws, c)).collect();
    let cols: Vec<(&str, &[f64])> = names.iter().zip(&data[1..]).map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    let max = v.num("max_size");
    let o = SubsetOptions {
        intercept: v.flag("intercept"),
        min_size: v.count("min_size").max(1),
        max_size: if max.is_finite() && max >= 1.0 { Some(max as usize) } else { None },
        n_best: v.count("n_best").max(1),
        force_in: force.iter().map(|&c| col_title(ws, c)).collect(),
    };
    let ropts = RegressionOptions { level: 0.95, ..Default::default() };
    let b = best_subsets(&cols, &data[0], &o, &ropts)?;
    let mut s = b.report_section();
    s.heading = format!("Best Subsets Regression: {} versus {}", col_title(ws, y), names.join(", "));
    out(s)
}

// ---------------------------------------------------------------------------------- prediction

fn predict_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let preds = v.vars("preds").to_vec();
    let newc = v.vars("new").to_vec();
    check_response(y, &preds)?;
    if newc.len() != preds.len() {
        return Err(format!(
            "give one column of new values for each of the {} predictors, in the same order (got {})",
            preds.len(),
            newc.len()
        )
        .into());
    }
    let (data, yn, names) = numeric_data(ws, y, &preds)?;
    let rhs = names.join(" + ");
    let formula = if v.flag("intercept") { format!("{yn} ~ {rhs}") } else { format!("{yn} ~ {rhs} - 1") };
    let lvl = level(v)?;
    let fit =
        regress(&formula, &data, &DesignOptions::default(), &RegressionOptions { level: lvl, ..Default::default() })?;
    let mut new = Data::new();
    let mut len = None;
    for (name, &c) in names.iter().zip(&newc) {
        let col = numeric(ws, c)?;
        if *len.get_or_insert(col.len()) != col.len() {
            return Err("the columns of new values have different lengths".into());
        }
        new = new.with_numeric(name, col)?;
    }
    let n_new = len.unwrap_or(0);
    let preds_out = fit.predict(&new, lvl)?;
    if preds_out.is_empty() {
        return Err("no row of new values is complete".into());
    }
    let mut s = fit.report_section();
    s.heading = format!(
        "Prediction for {} versus {}",
        col_title(ws, y),
        preds.iter().map(|&c| col_title(ws, c)).collect::<Vec<_>>().join(", ")
    );
    s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    let pct = lvl * 100.0;
    let mut t = Table::new(
        "Prediction for New Observations",
        vec![
            Column::new("Row", ColumnKind::Integer),
            Column::new("Fit", ColumnKind::Significant(6)),
            Column::new("SE Fit", ColumnKind::Significant(6)),
            Column::new(format!("{pct:.4}% CI Lower").replace(".0000", ""), ColumnKind::Significant(6)),
            Column::new(format!("{pct:.4}% CI Upper").replace(".0000", ""), ColumnKind::Significant(6)),
            Column::new(format!("{pct:.4}% PI Lower").replace(".0000", ""), ColumnKind::Significant(6)),
            Column::new(format!("{pct:.4}% PI Upper").replace(".0000", ""), ColumnKind::Significant(6)),
        ],
    );
    for p in &preds_out {
        push(
            &mut t,
            vec![
                Cell::Int(i64::try_from(p.row + 1).unwrap_or(i64::MAX)),
                Cell::Num(p.fit),
                Cell::Num(p.se_fit),
                Cell::Num(p.ci.lower),
                Cell::Num(p.ci.upper),
                Cell::Num(p.pi.lower),
                Cell::Num(p.pi.upper),
            ],
        )?;
    }
    s.table(t);
    let mut stored = Vec::new();
    if v.flag("store") {
        let rows: Vec<usize> = preds_out.iter().map(|p| p.row).collect();
        let col = |name: &str, f: &dyn Fn(&openstats_stats::regression::Prediction) -> f64| {
            let vals: Vec<f64> = preds_out.iter().map(f).collect();
            expand(name, n_new, &rows, &vals)
        };
        stored.push(col("PFITS", &|p| p.fit));
        stored.push(col("PSEFIT", &|p| p.se_fit));
        stored.push(col("PCLOW", &|p| p.ci.lower));
        stored.push(col("PCUP", &|p| p.ci.upper));
        stored.push(col("PPLOW", &|p| p.pi.lower));
        stored.push(col("PPUP", &|p| p.pi.upper));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ------------------------------------------------------------------------- fitted line plot

fn fitted_line(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let x = v.var("predictor").ok_or("select a predictor")?;
    if x == y {
        return Err("the response and predictor must be different columns".into());
    }
    let degree = v.choice("type") + 1;
    let (yc, xc) = (numeric(ws, y)?, numeric(ws, x)?);
    let (rows, d) = complete_rows(&[&yc, &xc]);
    if d[0].len() < degree + 2 {
        return Err(format!("a degree {degree} fit needs at least {} complete rows", degree + 2).into());
    }
    let (xs, ys) = (&d[1], &d[0]);
    let power_name = |k: usize| format!("XP{k}");
    let build = |vals: &[f64], with_y: bool| -> Res<Data> {
        let mut data = Data::new();
        if with_y {
            data = data.with_numeric("YV", ys.clone())?;
        }
        for k in 1..=degree {
            data = data.with_numeric(&power_name(k), vals.iter().map(|a| a.powi(k as i32)).collect())?;
        }
        Ok(data)
    };
    let terms: Vec<String> = (1..=degree).map(power_name).collect();
    let formula = format!("YV ~ {}{}", terms.join(" + "), if v.flag("intercept") { "" } else { " - 1" });
    let lvl = level(v)?;
    let fit = regress(
        &formula,
        &build(xs, true)?,
        &DesignOptions::default(),
        &RegressionOptions { level: lvl, ..Default::default() },
    )?;
    // The curve and its bands on a grid across the data.
    let (lo, hi) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &a| (l.min(a), h.max(a)));
    if hi <= lo {
        return Err(format!("{} has only one value", col_title(ws, x)).into());
    }
    let grid: Vec<f64> = (0..=100).map(|i| lo + (hi - lo) * f64::from(i) / 100.0).collect();
    let pred = fit.predict(&build(&grid, false)?, lvl)?;
    let mut s = fit.report_section();
    s.heading = format!(
        "Fitted Line Plot: {} versus {} ({})",
        col_title(ws, y),
        col_title(ws, x),
        ["linear", "quadratic", "cubic"][degree - 1]
    );
    s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
    let series_of = |name: &str, f: &dyn Fn(&openstats_stats::regression::Prediction) -> f64| Series {
        name: name.into(),
        x: grid.clone(),
        y: pred.iter().map(f).collect(),
    };
    let mut series =
        vec![Series { name: col_title(ws, y), x: xs.clone(), y: ys.clone() }, series_of("Fit", &|p| p.fit)];
    if v.flag("ci") {
        series.push(series_of("Lower CI", &|p| p.ci.lower));
        series.push(series_of("Upper CI", &|p| p.ci.upper));
    }
    if v.flag("pi") {
        series.push(series_of("Lower PI", &|p| p.pi.lower));
        series.push(series_of("Upper PI", &|p| p.pi.upper));
    }
    s.chart(ChartSpec {
        kind: ChartKind::Scatter,
        title: format!("Fitted Line Plot: {} versus {}", col_title(ws, y), col_title(ws, x)),
        x_label: col_title(ws, x),
        y_label: col_title(ws, y),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    });
    let n = ws.column(y)?.len();
    let mut stored = Vec::new();
    if v.flag("store_fits") {
        stored.push(expand("FITS", n, &rows, &fit.diagnostics.fitted));
    }
    if v.flag("store_resid") {
        stored.push(expand("RESI", n, &rows, &fit.diagnostics.residuals));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ------------------------------------------------------------------------ GLM comparisons

fn glm_comparisons(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    let target = v.var("compare").ok_or("select the factor whose levels are compared")?;
    if !factors.contains(&target) {
        return Err(format!("{} is not one of the model factors", col_title(ws, target)).into());
    }
    if factors.iter().chain(&covs).any(|&c| c == y) {
        return Err("the response cannot also be a model term".into());
    }
    if factors.iter().any(|f| covs.contains(f)) {
        return Err("a column cannot be both a factor and a covariate".into());
    }
    let mut taken: Vec<String> = Vec::new();
    let mut name_of = |c: usize| {
        let n = ident(ws, c, &taken);
        taken.push(n.clone());
        n
    };
    let yn = name_of(y);
    let mut data = Data::new().with_numeric(&yn, numeric(ws, y)?)?;
    let mut atoms = Vec::new();
    let mut target_name = String::new();
    for &c in &factors {
        let n = name_of(c);
        data = data.with_factor(&n, &labels(ws, c)?)?;
        if c == target {
            target_name.clone_from(&n);
        }
        atoms.push(n);
    }
    for &c in &covs {
        let n = name_of(c);
        data = data.with_numeric(&n, numeric(ws, c)?)?;
        atoms.push(n);
    }
    let order = v.count("order").max(1);
    let rhs = if order > 1 && atoms.len() > 1 {
        format!("({})^{}", atoms.join(" + "), order.min(atoms.len()))
    } else {
        atoms.join(" + ")
    };
    let lvl = level(v)?;
    let opts =
        GlmOptions { regression: RegressionOptions { level: lvl, ..Default::default() }, ..GlmOptions::default() };
    let r = glm(&format!("{yn} ~ {rhs}"), &data, &opts)?;
    let table = r.ls_means.iter().find(|t| t.term == target_name).ok_or(
        "least-squares means are not available for this factor (the model has nested terms or too many cells)",
    )?;
    let mut lab = Vec::new();
    let (mut n_eff, mut mean) = (Vec::new(), Vec::new());
    for row in &table.rows {
        if !row.mean.is_finite() || row.se.is_nan() || row.se <= 0.0 {
            return Err(
                format!("level {} of {} has no observations", row.levels.join("*"), col_title(ws, target)).into()
            );
        }
        lab.push(row.levels.join("*"));
        n_eff.push(r.fit.mse / (row.se * row.se));
        mean.push(row.mean);
    }
    let exact = covs.is_empty() && n_eff.iter().all(|&n| (n - n_eff[0]).abs() <= 1e-8 * n_eff[0].abs());
    let means = Means::new(lab.clone(), n_eff, mean, r.fit.mse, r.fit.df_error as f64)?;
    let alpha = 1.0 - lvl;
    let mut s = r.report_section();
    s.heading = format!(
        "General Linear Model with {} comparisons of {}: {} versus {}",
        ["Tukey", "Fisher", "Dunnett", "Hsu MCB"][v.choice("method").min(3)],
        col_title(ws, target),
        col_title(ws, y),
        atoms.join(", ")
    );
    let blocks = match v.choice("method") {
        1 => fisher(&means, alpha)?.report_section().blocks,
        2 => {
            let ctrl = v.text("control").trim();
            let idx = if ctrl.is_empty() {
                0
            } else {
                lab.iter()
                    .position(|l| l == ctrl)
                    .ok_or_else(|| DialogError(format!("'{ctrl}' is not a level of {}", col_title(ws, target))))?
            };
            let side = match v.choice("side") {
                1 => DunnettSide::Less,
                2 => DunnettSide::Greater,
                _ => DunnettSide::TwoSided,
            };
            dunnett(&means, idx, side, alpha)?.report_section().blocks
        }
        3 => {
            let best = if v.choice("best") == 1 { Best::Smallest } else { Best::Largest };
            hsu_mcb(&means, best, alpha)?.report_section().blocks
        }
        _ => tukey(&means, alpha)?.report_section().blocks,
    };
    s.blocks.extend(blocks);
    if !exact {
        s.warn(WARN_APPROX);
    }
    out(s)
}

// ------------------------------------------------------------------------------ registration

/// Register the regression and ANOVA extras.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Options, Storage};
    use VarKind::{Any, Numeric};
    let lvl = |s: DialogSpec| s.number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999);

    add(
        r,
        lvl(DialogSpec::new(
            "stat.regression.stepwise",
            "Stepwise Regression",
            "Forward, backward or stepwise selection of predictors by p-value, AIC, AICc or BIC.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("preds", "Potential predictors", Numeric, 1, 50)
        .vars("force", "Predictors in every model", Numeric, 0, 50)
        .choice(Page::Main, "method", "Method", &["Stepwise", "Forward selection", "Backward elimination"], 0)
        .choice(Page::Main, "criterion", "Criterion", &["P-value", "AIC", "AICc", "BIC"], 0)
        .optional_number(Options, "alpha_enter", "Alpha to enter (blank = method default)")
        .optional_number(Options, "alpha_remove", "Alpha to remove (blank = method default)")
        .check(Graphs, "residual_plots", "Residual plots", false)
        .check(Storage, "store_fits", "Fits of the final model", false)
        .check(Storage, "store_resid", "Residuals of the final model", false)),
        &["Stat", "Regression", "Stepwise..."],
        Runner::Report(stepwise_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.regression.bestsubsets",
            "Best Subsets Regression",
            "The best models of each size by R-squared, with adjusted and predicted R-squared, Mallows' Cp, S, AICc and BIC.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("preds", "Free predictors", Numeric, 1, 31)
        .vars("force", "Predictors in all models", Numeric, 0, 31)
        .integer(Page::Main, "n_best", "Models of each size to print", 2.0, 1.0, 20.0)
        .integer(Options, "min_size", "Smallest model (predictors)", 1.0, 1.0, 100.0)
        .optional_number(Options, "max_size", "Largest model (blank = all)")
        .check(Options, "intercept", "Fit intercept", true),
        &["Stat", "Regression", "Best Subsets..."],
        Runner::Report(subsets_run),
    );
    add(
        r,
        lvl(DialogSpec::new(
            "stat.regression.predict",
            "Predict",
            "Fit a regression and predict the response at new predictor values with confidence and prediction intervals.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("preds", "Predictors", Numeric, 1, 50)
        .vars("new", "Columns of new values (one per predictor, same order)", Numeric, 1, 50)
        .check(Options, "intercept", "Fit intercept", true)
        .check(Storage, "store", "Store fits, SE and intervals", false)),
        &["Stat", "Regression", "Predict..."],
        Runner::Report(predict_run),
    );
    add(
        r,
        lvl(DialogSpec::new(
            "stat.regression.fittedline",
            "Fitted Line Plot",
            "Scatter plot with a linear, quadratic or cubic regression line and optional confidence and prediction bands.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("predictor", "Predictor", Numeric, 1, 1)
        .choice(Page::Main, "type", "Type of regression model", &["Linear", "Quadratic", "Cubic"], 0)
        .check(Options, "intercept", "Fit intercept", true)
        .check(Graphs, "ci", "Display confidence interval", true)
        .check(Graphs, "pi", "Display prediction interval", false)
        .check(Storage, "store_fits", "Fits", false)
        .check(Storage, "store_resid", "Residuals", false)),
        &["Stat", "Regression", "Fitted Line Plot..."],
        Runner::Report(fitted_line),
    );
    add(
        r,
        lvl(DialogSpec::new(
            "stat.anova.glmcompare",
            "General Linear Model: Comparisons",
            "Fit a general linear model and compare the least-squares means of one factor: Tukey, Fisher, Dunnett or Hsu MCB.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .vars("factors", "Factors", Any, 1, 8)
        .vars("covariates", "Covariates", Numeric, 0, 8)
        .vars("compare", "Compare levels of", Any, 1, 1)
        .choice(Page::Main, "method", "Comparison method", &["Tukey", "Fisher", "Dunnett", "Hsu MCB"], 0)
        .integer(Options, "order", "Interactions through order", 1.0, 1.0, 8.0)
        .text(Options, "control", "Dunnett control level (blank = first level)", "")
        .choice(Options, "side", "Dunnett alternative", &["not equal", "less than control", "greater than control"], 0)
        .choice(Options, "best", "Hsu MCB: best is the", &["largest mean", "smallest mean"], 0)),
        &["Stat", "ANOVA", "General Linear Model: Comparisons..."],
        Runner::Report(glm_comparisons),
    );
}
