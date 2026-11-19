//! The regression family: binary, ordinal and nominal logistic regression, Poisson / negative-binomial /
//! gamma / Gaussian generalized linear models, nonlinear regression, partial least squares and orthogonal
//! regression.

use super::analyses::{add, out};
use super::analyses_adv::ident;
use super::analysis::{col_title, expand, labels, level, numeric, DialogError, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_math::linalg::Matrix;
use openstats_stats::glm::ordinal::{nominal_logistic, ordinal_logistic, CategoricalOptions};
use openstats_stats::glm::stepwise::{glm_stepwise, GlmStepwiseOptions};
use openstats_stats::glm::{
    association, classification_table, glm, hosmer_lemeshow, roc_curve, Family, GlmFit, GlmOptions, Link,
};
use openstats_stats::model::{Data, DesignOptions, Expr};
use openstats_stats::nonlinear::{nonlinear_regression, NonlinearOptions, Parameter};
use openstats_stats::orthogonal::orthogonal_regression;
use openstats_stats::pls::{pls_regression, CrossValidation, PlsOptions};
use openstats_stats::regression::StepMethod;
use openstats_stats::report::{Block, ChartKind, ChartSpec, Reportable, Section, Series};

// ------------------------------------------------------------------------------------ helpers

/// Distinct non-empty values sorted numerically when they all parse as numbers, otherwise as text.
fn sorted_levels(values: &[String]) -> Vec<String> {
    let mut v: Vec<String> = values.iter().filter(|s| !s.is_empty()).cloned().collect();
    v.sort();
    v.dedup();
    if v.iter().all(|s| s.trim().parse::<f64>().is_ok()) {
        v.sort_by(|a, b| a.trim().parse::<f64>().unwrap_or(0.0).total_cmp(&b.trim().parse::<f64>().unwrap_or(0.0)));
    }
    v
}

/// The category order: the comma-separated `order` text when given (it must name every level), otherwise sorted.
fn category_order(present: &[String], order_text: &str) -> Res<Vec<String>> {
    let sorted = sorted_levels(present);
    let given: Vec<String> = order_text.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if given.is_empty() {
        return Ok(sorted);
    }
    if let Some(missing) = sorted.iter().find(|l| !given.contains(l)) {
        return Err(format!("the category order does not list the value '{missing}'").into());
    }
    if let Some(extra) = given.iter().find(|g| !sorted.contains(g)) {
        return Err(format!("'{extra}' in the category order does not occur in the response").into());
    }
    Ok(given)
}

struct Terms {
    data: Data,
    rhs: String,
    titles: String,
}

/// Add the factors and covariates of a model to `data` under formula-safe names and build the right-hand side.
fn model_terms(
    ws: &Worksheet,
    mut data: Data,
    mut taken: Vec<String>,
    factors: &[usize],
    covs: &[usize],
    order: usize,
) -> Res<Terms> {
    if factors.is_empty() && covs.is_empty() {
        return Err("select at least one factor or covariate".into());
    }
    if factors.iter().any(|f| covs.contains(f)) {
        return Err("a column cannot be both a factor and a covariate".into());
    }
    let mut names = Vec::new();
    for &c in factors {
        let n = ident(ws, c, &taken);
        taken.push(n.clone());
        data = data.with_factor(&n, &labels(ws, c)?)?;
        names.push(n);
    }
    for &c in covs {
        let n = ident(ws, c, &taken);
        taken.push(n.clone());
        data = data.with_numeric(&n, numeric(ws, c)?)?;
        names.push(n);
    }
    let order = order.max(1);
    let rhs = if order > 1 && names.len() > 1 {
        format!("({})^{}", names.join(" + "), order.min(names.len()))
    } else {
        names.join(" + ")
    };
    let mut t: Vec<String> = factors.iter().chain(covs).map(|&c| col_title(ws, c)).collect();
    if order > 1 && t.len() > 1 {
        t.push(format!("(interactions to order {})", order.min(t.len())));
    }
    Ok(Terms { data, rhs, titles: t.join(", ") })
}

fn check_not_terms(y: usize, others: &[usize]) -> Res<()> {
    if others.contains(&y) {
        return Err("the response cannot also be a model term".into());
    }
    Ok(())
}

/// Append another section's blocks under its heading as a text line.
fn append(s: &mut Section, sub: Section) {
    s.blocks.push(Block::Text(sub.heading));
    s.blocks.extend(sub.blocks);
    s.warnings.extend(sub.warnings);
}

fn drop_charts(s: &mut Section) {
    s.blocks.retain(|b| !matches!(b, Block::Chart(_)));
}

fn store_if(v: &Values, n: usize, rows: &[usize], list: &[(&str, &str, &[f64])]) -> Vec<Column> {
    list.iter().filter(|(flag, _, _)| v.flag(flag)).map(|(_, name, vals)| expand(name, n, rows, vals)).collect()
}

fn step_method(v: &Values) -> Option<StepMethod> {
    match v.choice("stepwise") {
        1 => Some(StepMethod::Stepwise),
        2 => Some(StepMethod::Forward),
        3 => Some(StepMethod::Backward),
        _ => None,
    }
}

fn fit_model(
    formula: &str,
    data: &Data,
    opts: &GlmOptions,
    v: &Values,
    notes: &mut Vec<String>,
) -> Res<(GlmFit, Option<Section>)> {
    let design = DesignOptions::default();
    match step_method(v) {
        None => Ok((glm(formula, data, &design, opts)?, None)),
        Some(m) => {
            let mut so = GlmStepwiseOptions::new(m);
            if v.num("alpha_enter").is_finite() {
                so.alpha_enter = v.num("alpha_enter");
            }
            if v.num("alpha_remove").is_finite() {
                so.alpha_remove = v.num("alpha_remove");
            }
            if !(so.alpha_enter > 0.0 && so.alpha_enter < 1.0 && so.alpha_remove > 0.0 && so.alpha_remove < 1.0) {
                return Err("the stepwise alpha values must be between 0 and 1".into());
            }
            let r = glm_stepwise(formula, data, &design, opts, &so)?;
            notes.push(format!(
                "Stepwise selection ({}): alpha to enter {}, alpha to remove {}.",
                ["Stepwise", "Forward", "Backward"][v.choice("stepwise") - 1],
                so.alpha_enter,
                so.alpha_remove
            ));
            let sec = r.report_section();
            Ok((r.fit, Some(sec)))
        }
    }
}

fn glm_storage(v: &Values, n: usize, fit: &GlmFit, fits_name: &str) -> Vec<Column> {
    let rows = fit.rows_used();
    let mut stored = store_if(
        v,
        n,
        rows,
        &[
            ("store_fits", fits_name, &fit.fitted),
            ("store_eta", "ETA", &fit.eta),
            ("store_dres", "DRES", &fit.deviance_residuals),
            ("store_pres", "PRES", &fit.pearson_residuals),
            ("store_sdres", "SDRES", &fit.std_deviance_residuals),
            ("store_hi", "HI", &fit.leverage),
            ("store_cook", "COOK", &fit.cooks_d),
            ("store_dchi", "DCHI", &fit.delta_chi2),
            ("store_ddev", "DDEV", &fit.delta_deviance),
            ("store_dbeta", "DBETA", &fit.delta_beta),
        ],
    );
    if v.flag("store_coef") {
        let est = fit.estimates();
        let idx: Vec<usize> = (0..est.len()).collect();
        stored.push(expand("COEF", est.len(), &idx, &est));
    }
    stored
}

fn add_model_storage(spec: DialogSpec, fits_label: &str) -> DialogSpec {
    use Page::Storage;
    spec.check(Storage, "store_fits", fits_label, false)
        .check(Storage, "store_eta", "Linear predictor", false)
        .check(Storage, "store_dres", "Deviance residuals", false)
        .check(Storage, "store_pres", "Pearson residuals", false)
        .check(Storage, "store_sdres", "Standardized deviance residuals", false)
        .check(Storage, "store_hi", "Leverages (hi)", false)
        .check(Storage, "store_cook", "Cook's distance", false)
        .check(Storage, "store_dchi", "Delta chi-square", false)
        .check(Storage, "store_ddev", "Delta deviance", false)
        .check(Storage, "store_dbeta", "Delta beta", false)
        .check(Storage, "store_coef", "Coefficients", false)
}

fn add_stepwise_fields(spec: DialogSpec) -> DialogSpec {
    use Page::Options;
    spec.choice(
        Options,
        "stepwise",
        "Stepwise selection of terms",
        &["None (fit the full model)", "Stepwise", "Forward selection", "Backward elimination"],
        0,
    )
    .optional_number(Options, "alpha_enter", "Alpha to enter (blank = 0.15 / 0.25 / 0.10 by method)")
    .optional_number(Options, "alpha_remove", "Alpha to remove (blank = method default)")
}

// -------------------------------------------------------------------------------- binary logistic

fn binary_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let trials_col = v.var("trials");
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    check_not_terms(y, &factors)?;
    check_not_terms(y, &covs)?;
    if trials_col == Some(y) {
        return Err("the number of trials cannot be the response column".into());
    }
    let n = ws.column(y)?.len();
    let mut taken = vec!["y_".to_string()];
    let mut opts = GlmOptions::logistic();
    opts.level = level(v)?;
    opts.link = [Link::Logit, Link::Probit, Link::Cloglog][v.choice("link").min(2)];
    let mut notes = Vec::new();
    let resp: Vec<f64>;
    let info: String;
    if let Some(tc) = trials_col {
        resp = numeric(ws, y)?;
        let t = numeric(ws, tc)?;
        taken.push("t_".into());
        opts.trials = Some(t);
        info = format!("Response: {} events in {} trials.", col_title(ws, y), col_title(ws, tc));
    } else {
        let lab = labels(ws, y)?;
        let levels = sorted_levels(&lab);
        if levels.len() != 2 {
            return Err(format!(
                "{} has {} distinct values; a binary response needs exactly 2 (use ordinal or nominal logistic regression otherwise)",
                col_title(ws, y),
                levels.len()
            )
            .into());
        }
        let ev = v.text("event").trim().to_string();
        let event = if ev.is_empty() { levels[1].clone() } else { ev };
        if !levels.contains(&event) {
            return Err(format!("'{event}' is not a value of {}", col_title(ws, y)).into());
        }
        let other = levels.iter().find(|l| **l != event).cloned().unwrap_or_default();
        resp = lab
            .iter()
            .map(|s| {
                if s.is_empty() {
                    f64::NAN
                } else if *s == event {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        info = format!(
            "Response: {} (event: {event}; reference: {other}); {} events and {} non-events.",
            col_title(ws, y),
            resp.iter().filter(|&&x| x == 1.0).count(),
            resp.iter().filter(|&&x| x == 0.0).count()
        );
    }
    let data = Data::new().with_numeric("y_", resp)?;
    let terms = model_terms(ws, data, taken, &factors, &covs, v.count("order"))?;
    let formula = format!("y_ ~ {}", terms.rhs);
    let (fit, step) = fit_model(&formula, &terms.data, &opts, v, &mut notes)?;
    let mut s = step.unwrap_or_else(|| fit.report_section());
    s.heading = format!("Binary Logistic Regression: {} versus {}", col_title(ws, y), terms.titles);
    s.blocks.insert(0, Block::Text(info));
    for note in notes {
        s.text(note);
    }
    if !v.flag("residual_plots") {
        drop_charts(&mut s);
    }
    if v.flag("classification") {
        let cutoff = v.num("cutoff");
        if !(cutoff > 0.0 && cutoff < 1.0) {
            return Err("the classification cutoff must be between 0 and 1".into());
        }
        append(&mut s, classification_table(&fit, cutoff)?.report_section());
    }
    if v.flag("association") {
        append(&mut s, association(&fit)?.report_section());
    }
    if v.flag("hl_table") {
        append(&mut s, hosmer_lemeshow(&fit, v.count("hl_groups").max(3))?.report_section());
    }
    if v.flag("roc") {
        append(&mut s, roc_curve(&fit)?.report_section());
    }
    Ok(Output { stored: glm_storage(v, n, &fit, "EPRO"), section: s, models: Vec::new() })
}

// ------------------------------------------------------------------------ ordinal and nominal

struct Categorical {
    labels: Vec<String>,
    terms: Terms,
    n: usize,
}

fn categorical_input(ws: &Worksheet, v: &Values) -> Res<Categorical> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    check_not_terms(y, &factors)?;
    check_not_terms(y, &covs)?;
    let lab = labels(ws, y)?;
    let order = category_order(&lab, v.text("order"))?;
    if order.len() < 2 {
        return Err(format!("{} needs at least two distinct values", col_title(ws, y)).into());
    }
    if order.len() > 20 {
        return Err(format!(
            "{} has {} distinct values; at most 20 categories are supported",
            col_title(ws, y),
            order.len()
        )
        .into());
    }
    let codes: Vec<f64> =
        lab.iter().map(|s| order.iter().position(|o| o == s).map_or(f64::NAN, |i| i as f64)).collect();
    let n = lab.len();
    let data = Data::new().with_numeric("y_", codes)?;
    let terms = model_terms(ws, data.clone(), vec!["y_".into()], &factors, &covs, v.count("order_terms"))?;
    Ok(Categorical { labels: order, terms, n })
}

fn prob_columns(n: usize, rows: &[usize], probs: &Matrix, labels: &[String], predicted: &[usize]) -> Vec<Column> {
    let mut cols = Vec::new();
    for (c, name) in labels.iter().enumerate() {
        let vals: Vec<f64> = (0..probs.rows()).map(|i| probs[(i, c)]).collect();
        cols.push(expand(&format!("PROB_{name}"), n, rows, &vals));
    }
    let mut text = vec![String::new(); n];
    for (&r, &p) in rows.iter().zip(predicted) {
        text[r].clone_from(&labels[p]);
    }
    cols.push(Column::text("PREDCAT", text));
    cols
}

fn ordinal_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = categorical_input(ws, v)?;
    let y = v.var("response").ok_or("select a response")?;
    let mut o = CategoricalOptions::new(c.labels.clone());
    o.level = level(v)?;
    o.link = [Link::Logit, Link::Probit, Link::Cloglog][v.choice("link").min(2)];
    let fit = ordinal_logistic(&format!("y_ ~ {}", c.terms.rhs), &c.terms.data, &DesignOptions::default(), &o)?;
    let mut s = fit.report_section();
    s.heading = format!("Ordinal Logistic Regression: {} versus {}", col_title(ws, y), c.terms.titles);
    s.text(format!("Categories in order: {}.", c.labels.join(" < ")));
    let stored = if v.flag("store_probs") {
        prob_columns(c.n, &fit.rows_used, &fit.probabilities, &c.labels, &fit.predicted)
    } else {
        vec![]
    };
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn nominal_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let c = categorical_input(ws, v)?;
    let y = v.var("response").ok_or("select a response")?;
    let mut o = CategoricalOptions::new(c.labels.clone());
    o.level = level(v)?;
    let r = v.text("reference").trim();
    if !r.is_empty() {
        o.reference = c
            .labels
            .iter()
            .position(|l| l == r)
            .ok_or_else(|| DialogError(format!("'{r}' is not a value of {}", col_title(ws, y))))?;
    }
    let fit = nominal_logistic(&format!("y_ ~ {}", c.terms.rhs), &c.terms.data, &DesignOptions::default(), &o)?;
    let mut s = fit.report_section();
    s.heading = format!("Nominal Logistic Regression: {} versus {}", col_title(ws, y), c.terms.titles);
    let stored = if v.flag("store_probs") {
        prob_columns(c.n, &fit.rows_used, &fit.probabilities, &c.labels, &fit.predicted)
    } else {
        vec![]
    };
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ------------------------------------------------------------------------------------ GLM family

fn family_run(ws: &Worksheet, v: &Values, family: Family, link: Link, heading: &str) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let factors = v.vars("factors").to_vec();
    let covs = v.vars("covariates").to_vec();
    check_not_terms(y, &factors)?;
    check_not_terms(y, &covs)?;
    let n = ws.column(y)?.len();
    let data = Data::new().with_numeric("y_", numeric(ws, y)?)?;
    let terms = model_terms(ws, data, vec!["y_".into()], &factors, &covs, v.count("order"))?;
    let mut opts = GlmOptions::new(family);
    opts.link = link;
    opts.level = level(v)?;
    if family == Family::NegativeBinomial && v.num("theta").is_finite() {
        opts.theta = Some(v.num("theta"));
    }
    let mut notes = Vec::new();
    let (fit, step) = fit_model(&format!("y_ ~ {}", terms.rhs), &terms.data, &opts, v, &mut notes)?;
    let mut s = step.unwrap_or_else(|| fit.report_section());
    s.heading = format!("{heading}: {} versus {}", col_title(ws, y), terms.titles);
    for note in notes {
        s.text(note);
    }
    if !v.flag("residual_plots") {
        drop_charts(&mut s);
    }
    Ok(Output { stored: glm_storage(v, n, &fit, "FITS"), section: s, models: Vec::new() })
}

fn poisson_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let family = if v.choice("family") == 1 { Family::NegativeBinomial } else { Family::Poisson };
    let link = if v.choice("link") == 1 { Link::Identity } else { Link::Log };
    let heading = if family == Family::Poisson { "Poisson Regression" } else { "Negative Binomial Regression" };
    family_run(ws, v, family, link, heading)
}

fn glm_family_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let family =
        [Family::Gamma, Family::Gaussian, Family::Poisson, Family::NegativeBinomial][v.choice("family").min(3)];
    let link = match v.choice("link") {
        1 => Link::Log,
        2 => Link::Identity,
        3 => Link::Inverse,
        _ => family.default_link(),
    };
    family_run(ws, v, family, link, "Generalized Linear Model")
}

// ----------------------------------------------------------------------------- nonlinear regression

fn parse_assignments(text: &str, what: &str) -> Res<Vec<(String, f64)>> {
    let mut out = Vec::new();
    for part in text.split([',', ';', '\n']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, val) =
            part.split_once('=').ok_or_else(|| DialogError(format!("{what}: expected name=value, got '{part}'")))?;
        let value: f64 =
            val.trim().parse().map_err(|_| DialogError(format!("{what}: '{}' is not a number", val.trim())))?;
        out.push((name.trim().to_string(), value));
    }
    Ok(out)
}

fn column_by_name(ws: &Worksheet, name: &str) -> Option<usize> {
    (0..ws.n_cols())
        .find(|&c| col_title(ws, c) == name)
        .or_else(|| (0..ws.n_cols()).find(|&c| ident(ws, c, &[]) == name))
}

fn nonlinear_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let equation = v.text("equation").trim().to_string();
    if equation.is_empty() {
        return Err("enter the equation, for example  a*exp(-b*x) + c".into());
    }
    let starts = parse_assignments(v.text("params"), "starting values")?;
    if starts.is_empty() {
        return Err("enter starting values for the parameters, for example  a=2, b=0.3, c=0".into());
    }
    let lower = parse_assignments(v.text("lower"), "lower bounds")?;
    let upper = parse_assignments(v.text("upper"), "upper bounds")?;
    for (name, _) in lower.iter().chain(&upper) {
        if !starts.iter().any(|(s, _)| s == name) {
            return Err(format!("a bound is given for '{name}', which is not a parameter").into());
        }
    }
    let params: Vec<Parameter> = starts
        .iter()
        .map(|(name, start)| Parameter {
            name: name.clone(),
            start: *start,
            lower: lower.iter().find(|(l, _)| l == name).map(|(_, b)| *b),
            upper: upper.iter().find(|(l, _)| l == name).map(|(_, b)| *b),
        })
        .collect();
    let rhs = equation.split(['=', '~']).next_back().unwrap_or(&equation).to_string();
    let expr = Expr::parse(&rhs)?;
    let mut data = Data::new().with_numeric("response_", numeric(ws, y)?)?;
    let predictors: Vec<String> =
        expr.variables().into_iter().filter(|n| !params.iter().any(|p| &p.name == n)).collect();
    for name in &predictors {
        let c = column_by_name(ws, name)
            .ok_or_else(|| DialogError(format!("'{name}' is neither a parameter nor the name of a column")))?;
        data = data.with_numeric(name, numeric(ws, c)?)?;
    }
    let o = NonlinearOptions { level: level(v)?, max_iter: v.count("max_iter").max(1) };
    let fit = nonlinear_regression("response_", &equation, &params, &data, &o)?;
    let mut s = fit.report_section();
    s.heading = format!("Nonlinear Regression: {} = {}", col_title(ws, y), rhs.trim());
    if predictors.len() == 1 && (v.flag("ci") || v.flag("pi")) {
        let xs = data.numeric(&predictors[0]).map(<[f64]>::to_vec).unwrap_or_default();
        let ys = data.numeric("response_").map(<[f64]>::to_vec).unwrap_or_default();
        let (lo, hi) = xs
            .iter()
            .filter(|x| x.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
        let grid: Vec<f64> = (0..=100).map(|i| lo + (hi - lo) * i as f64 / 100.0).collect();
        let gd = Data::new().with_numeric(&predictors[0], grid.clone())?;
        let pred = fit.predict(&gd, o.level)?;
        let mk = |name: &str, f: &dyn Fn(&openstats_stats::nonlinear::NlPrediction) -> f64| Series {
            name: name.into(),
            x: grid.clone(),
            y: pred.iter().map(f).collect(),
        };
        let used: Vec<(f64, f64)> = fit.rows_used.iter().map(|&i| (xs[i], ys[i])).collect();
        let mut series = vec![
            Series {
                name: col_title(ws, y),
                x: used.iter().map(|p| p.0).collect(),
                y: used.iter().map(|p| p.1).collect(),
            },
            mk("Fit", &|p| p.fit),
        ];
        if v.flag("ci") {
            series.push(mk("Lower CI", &|p| p.ci.lower));
            series.push(mk("Upper CI", &|p| p.ci.upper));
        }
        if v.flag("pi") {
            series.push(mk("Lower PI", &|p| p.pi.lower));
            series.push(mk("Upper PI", &|p| p.pi.upper));
        }
        s.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: format!("Fitted Curve: {} versus {}", col_title(ws, y), predictors[0]),
            x_label: predictors[0].clone(),
            y_label: col_title(ws, y),
            categories: Vec::new(),
            series,
            reference_lines: Vec::new(),
        });
    }
    let n = ws.column(y)?.len();
    let stored =
        store_if(v, n, &fit.rows_used, &[("store_fits", "FITS", &fit.fitted), ("store_resid", "RESI", &fit.residuals)]);
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ----------------------------------------------------------------------------------------- PLS

fn pls_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let ys = v.vars("responses").to_vec();
    let xs = v.vars("preds").to_vec();
    if ys.is_empty() || xs.is_empty() {
        return Err("select at least one response and one predictor".into());
    }
    if let Some(c) = ys.iter().find(|c| xs.contains(c)) {
        return Err(format!("{} cannot be both a response and a predictor", col_title(ws, *c)).into());
    }
    let xdata: Vec<(String, Vec<f64>)> =
        xs.iter().map(|&c| Ok((col_title(ws, c), numeric(ws, c)?))).collect::<Res<_>>()?;
    let ydata: Vec<(String, Vec<f64>)> =
        ys.iter().map(|&c| Ok((col_title(ws, c), numeric(ws, c)?))).collect::<Res<_>>()?;
    let xr: Vec<(&str, &[f64])> = xdata.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    let yr: Vec<(&str, &[f64])> = ydata.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    let cv = match v.choice("cv") {
        1 => CrossValidation::KFold(v.count("folds").max(2)),
        2 => CrossValidation::None,
        _ => CrossValidation::LeaveOneOut,
    };
    let o = PlsOptions { components: v.count("components").max(1), standardize: v.flag("standardize"), cv };
    let fit = pls_regression(&xr, &yr, &o)?;
    let mut s = fit.report_section();
    s.heading = format!(
        "Partial Least Squares Regression: {} versus {}",
        ydata.iter().map(|d| d.0.as_str()).collect::<Vec<_>>().join(", "),
        xdata.iter().map(|d| d.0.as_str()).collect::<Vec<_>>().join(", ")
    );
    if fit.n_components < o.components {
        s.warn(format!("Only {} of the {} requested components could be extracted.", fit.n_components, o.components));
    }
    let n = ws.column(ys[0])?.len();
    let mut stored = Vec::new();
    for (k, (name, _)) in ydata.iter().enumerate() {
        let fitted: Vec<f64> = (0..fit.n).map(|i| fit.fitted[(i, k)]).collect();
        let resid: Vec<f64> = (0..fit.n).map(|i| fit.residuals[(i, k)]).collect();
        if v.flag("store_fits") {
            stored.push(expand(&format!("FITS_{name}"), n, &fit.rows_used, &fitted));
        }
        if v.flag("store_resid") {
            stored.push(expand(&format!("RESI_{name}"), n, &fit.rows_used, &resid));
        }
        if v.flag("store_coef") {
            let mut c = vec![fit.intercepts[k]];
            c.extend((0..xdata.len()).map(|j| fit.coefficients[(j, k)]));
            let idx: Vec<usize> = (0..c.len()).collect();
            stored.push(expand(&format!("COEF_{name}"), c.len(), &idx, &c));
        }
    }
    if v.flag("store_scores") {
        for c in 0..fit.n_components {
            stored.push(expand(&format!("XSCORE{}", c + 1), n, &fit.rows_used, &fit.x_scores.col(c)));
        }
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

// ------------------------------------------------------------------------------------ orthogonal

fn orthogonal_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let y = v.var("response").ok_or("select a response")?;
    let x = v.var("predictor").ok_or("select a predictor")?;
    if x == y {
        return Err("the response and the predictor must be different columns".into());
    }
    let xv = numeric(ws, x)?;
    let yv = numeric(ws, y)?;
    let fit = orthogonal_regression(&col_title(ws, x), &xv, &col_title(ws, y), &yv, v.num("ratio"), level(v)?)?;
    let n = yv.len();
    let stored = store_if(
        v,
        n,
        &fit.rows_used,
        &[
            ("store_fits", "FITS", &fit.fitted),
            ("store_resid", "RESI", &fit.residuals),
            ("store_dist", "ODIST", &fit.orthogonal_distances),
        ],
    );
    out(fit.report_section()).map(|o| Output { stored, ..o })
}

// ------------------------------------------------------------------------------ registration

/// Register the regression-family dialogs.
pub(crate) fn register(r: &mut Registry) {
    use Page::{Graphs, Main, Options, Storage};
    use VarKind::{Any, Numeric};
    let lvl = |s: DialogSpec| s.number(Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999);
    let order = |s: DialogSpec| s.integer(Options, "order", "Interactions through order", 1.0, 1.0, 8.0);

    let binary = DialogSpec::new(
        "stat.regression.binary",
        "Binary Logistic Regression",
        "Model a two-valued response (or events out of trials) on factors and covariates; odds ratios, deviance table, Hosmer-Lemeshow, classification table, ROC curve.",
    )
    .vars("response", "Response (binary, or events if trials are given)", Any, 1, 1)
    .vars("trials", "Number of trials (optional)", Numeric, 0, 1)
    .vars("factors", "Categorical predictors", Any, 0, 8)
    .vars("covariates", "Continuous predictors", Numeric, 0, 8)
    .text(Main, "event", "Response event (blank = last value in sorted order)", "")
    .choice(Main, "link", "Link function", &["Logit", "Probit (normit)", "Complementary log-log"], 0)
    .check(Options, "classification", "Classification table", true)
    .number(Options, "cutoff", "Classification cutoff (probability)", 0.5, 1e-6, 0.999_999)
    .check(Options, "association", "Measures of association", true)
    .check(Options, "hl_table", "Hosmer-Lemeshow groups table", false)
    .integer(Options, "hl_groups", "Hosmer-Lemeshow groups", 10.0, 3.0, 50.0)
    .check(Graphs, "roc", "ROC curve", false)
    .check(Graphs, "residual_plots", "Deviance residuals versus fits", false);
    add(
        r,
        add_model_storage(add_stepwise_fields(order(lvl(binary))), "Fits (event probabilities)"),
        &["Stat", "Regression", "Binary Logistic Regression..."],
        Runner::Report(binary_run),
    );

    let ordinal = DialogSpec::new(
        "stat.regression.ordinal",
        "Ordinal Logistic Regression",
        "Model an ordered categorical response with the proportional-odds (cumulative link) model.",
    )
    .vars("response", "Response (ordered categories)", Any, 1, 1)
    .vars("factors", "Categorical predictors", Any, 0, 8)
    .vars("covariates", "Continuous predictors", Numeric, 0, 8)
    .text(Main, "order", "Category order, low to high (comma-separated; blank = sorted)", "")
    .choice(Main, "link", "Link function", &["Logit", "Probit (normit)", "Complementary log-log"], 0)
    .integer(Options, "order_terms", "Interactions through order", 1.0, 1.0, 8.0)
    .check(Storage, "store_probs", "Category probabilities and predicted category", false);
    add(r, lvl(ordinal), &["Stat", "Regression", "Ordinal Logistic Regression..."], Runner::Report(ordinal_run));

    let nominal = DialogSpec::new(
        "stat.regression.nominal",
        "Nominal Logistic Regression",
        "Model an unordered categorical response with baseline-category logits.",
    )
    .vars("response", "Response (categories)", Any, 1, 1)
    .vars("factors", "Categorical predictors", Any, 0, 8)
    .vars("covariates", "Continuous predictors", Numeric, 0, 8)
    .text(Main, "order", "Category order (comma-separated; blank = sorted; first is the default reference)", "")
    .text(Main, "reference", "Reference event (blank = first category)", "")
    .integer(Options, "order_terms", "Interactions through order", 1.0, 1.0, 8.0)
    .check(Storage, "store_probs", "Category probabilities and predicted category", false);
    add(r, lvl(nominal), &["Stat", "Regression", "Nominal Logistic Regression..."], Runner::Report(nominal_run));

    let poisson = DialogSpec::new(
        "stat.regression.poisson",
        "Poisson Regression",
        "Model a count response; the negative-binomial family handles over-dispersed counts.",
    )
    .vars("response", "Response (counts)", Numeric, 1, 1)
    .vars("factors", "Categorical predictors", Any, 0, 8)
    .vars("covariates", "Continuous predictors", Numeric, 0, 8)
    .choice(Main, "family", "Distribution", &["Poisson", "Negative binomial"], 0)
    .choice(Main, "link", "Link function", &["Log", "Identity"], 0)
    .optional_number(Options, "theta", "Negative-binomial theta (blank = estimate)")
    .check(Graphs, "residual_plots", "Deviance residuals versus fits", false);
    add(
        r,
        add_model_storage(add_stepwise_fields(order(lvl(poisson))), "Fits (means)"),
        &["Stat", "Regression", "Poisson Regression..."],
        Runner::Report(poisson_run),
    );

    let glm_spec = DialogSpec::new(
        "stat.regression.glmfamily",
        "Generalized Linear Model",
        "Fit a generalized linear model: gamma, Gaussian, Poisson or negative-binomial response with a choice of link.",
    )
    .vars("response", "Response", Numeric, 1, 1)
    .vars("factors", "Categorical predictors", Any, 0, 8)
    .vars("covariates", "Continuous predictors", Numeric, 0, 8)
    .choice(Main, "family", "Distribution", &["Gamma", "Gaussian", "Poisson", "Negative binomial"], 0)
    .choice(Main, "link", "Link function", &["Canonical", "Log", "Identity", "Inverse"], 0)
    .optional_number(Options, "theta", "Negative-binomial theta (blank = estimate)")
    .check(Graphs, "residual_plots", "Deviance residuals versus fits", false);
    add(
        r,
        add_model_storage(add_stepwise_fields(order(lvl(glm_spec))), "Fits (means)"),
        &["Stat", "Regression", "Generalized Linear Model..."],
        Runner::Report(glm_family_run),
    );

    add(
        r,
        lvl(DialogSpec::new(
            "stat.regression.nonlinear",
            "Nonlinear Regression",
            "Fit a user-written equation by Levenberg-Marquardt least squares with standard errors, confidence and prediction bands.",
        )
        .vars("response", "Response", Numeric, 1, 1)
        .text(Main, "equation", "Equation (columns by name; e.g. a*exp(-b*x) + c)", "")
        .text(Main, "params", "Parameters and starting values (e.g. a=2, b=0.3, c=0)", "")
        .text(Options, "lower", "Lower bounds (e.g. a=0, b=0)", "")
        .text(Options, "upper", "Upper bounds", "")
        .integer(Options, "max_iter", "Maximum iterations", 500.0, 1.0, 100_000.0)
        .check(Graphs, "ci", "Confidence band (one predictor)", true)
        .check(Graphs, "pi", "Prediction band (one predictor)", false)
        .check(Storage, "store_fits", "Fits", false)
        .check(Storage, "store_resid", "Residuals", false)),
        &["Stat", "Regression", "Nonlinear Regression..."],
        Runner::Report(nonlinear_run),
    );

    add(
        r,
        DialogSpec::new(
            "stat.regression.pls",
            "Partial Least Squares Regression",
            "Regress one or more responses on many correlated predictors through latent components, with cross-validated model selection.",
        )
        .vars("responses", "Responses", Numeric, 1, 10)
        .vars("preds", "Predictors", Numeric, 1, 100)
        .integer(Main, "components", "Maximum number of components", 2.0, 1.0, 100.0)
        .choice(Options, "cv", "Cross-validation", &["Leave-one-out", "k-fold", "None"], 0)
        .integer(Options, "folds", "Number of groups for k-fold", 10.0, 2.0, 1000.0)
        .check(Options, "standardize", "Standardize predictors and responses", false)
        .check(Storage, "store_fits", "Fits", false)
        .check(Storage, "store_resid", "Residuals", false)
        .check(Storage, "store_coef", "Coefficients", false)
        .check(Storage, "store_scores", "X scores", false),
        &["Stat", "Regression", "Partial Least Squares..."],
        Runner::Report(pls_run),
    );

    add(
        r,
        lvl(DialogSpec::new(
            "stat.regression.orthogonal",
            "Orthogonal Regression",
            "Fit a line when both variables carry measurement error (Deming regression) with a known error-variance ratio.",
        )
        .vars("response", "Response (Y)", Numeric, 1, 1)
        .vars("predictor", "Predictor (X)", Numeric, 1, 1)
        .number(Main, "ratio", "Error variance ratio (variance of Y error / variance of X error)", 1.0, 1e-9, 1e9)
        .check(Storage, "store_fits", "Fits", false)
        .check(Storage, "store_resid", "Residuals", false)
        .check(Storage, "store_dist", "Orthogonal distances", false)),
        &["Stat", "Regression", "Orthogonal Regression..."],
        Runner::Report(orthogonal_run),
    );
}
