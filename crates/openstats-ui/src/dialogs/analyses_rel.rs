//! Reliability and survival analyses (Stat ▸ Reliability/Survival): parametric distribution
//! analysis with right, left and interval censoring, distribution identification,
//! Kaplan–Meier with group comparison, accelerated life testing and warranty prediction.
//!
//! A censoring column marks a row as censored when it equals the "censoring value" (default 0);
//! every other value is a failure. A "start" column switches to arbitrary censoring, in which the
//! time column is the end of the interval (blank start = failed before the end, blank end =
//! survived past the start, equal values = exact time).

use super::analyses::{add, out};
use super::analysis::{col_title, labels, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::Worksheet;
use openstats_stats::reliability::report_impl::{life_section, AltReport, KmGroups, LifeReportOptions};
use openstats_stats::reliability::{
    compare_groups, cox_formula, fit_alt, fit_distribution, kaplan_meier, rank_distributions, warranty_forecast,
    CiMethod, CoxOptions, CoxTies, Family, LifeData, ObsKind, Relationship, StressVariable, WarrantyInput, WeightKind,
};
use openstats_stats::report::{
    Cell, ChartKind, ChartSpec, Column as RCol, ColumnKind, Reportable, Section, Series, Table,
};

const DISTS: [&str; 6] = ["Weibull", "Lognormal", "Exponential", "Normal", "Logistic", "Loglogistic"];

fn family(v: &Values) -> Family {
    match v.choice("dist") {
        1 => Family::Lognormal,
        2 => Family::Exponential,
        3 => Family::Normal,
        4 => Family::Logistic,
        5 => Family::Loglogistic,
        _ => Family::Weibull,
    }
}

/// Numbers separated by spaces, commas or semicolons.
pub(super) fn parse_numbers(text: &str, what: &str) -> Res<Vec<f64>> {
    text.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f64>().map_err(|_| format!("'{t}' is not a number in {what}").into()))
        .collect()
}

/// Life data from the time / start / censor / frequency fields.
pub(super) fn life_data(ws: &Worksheet, v: &Values) -> Res<(LifeData, String)> {
    let tcol = v.var("time").ok_or("select the time column")?;
    let time = numeric(ws, tcol)?;
    let freq = match v.var("freq") {
        Some(c) => Some(numeric(ws, c)?),
        None => None,
    };
    let name = col_title(ws, tcol);
    if let Some(sc) = v.var("start") {
        if v.var("censor").is_some() {
            return Err("use either a censoring column or a start column, not both".into());
        }
        let start = numeric(ws, sc)?;
        return Ok((LifeData::from_intervals(&start, &time, freq.as_deref())?, name));
    }
    let censor = match v.var("censor") {
        Some(c) => Some(numeric(ws, c)?),
        None => None,
    };
    let cv = v.num("censor_value");
    if censor.is_some() && cv.is_nan() {
        return Err("enter the value that marks a censored observation".into());
    }
    Ok((LifeData::from_times(&time, censor.as_deref(), cv, freq.as_deref())?, name))
}

pub(super) fn add_data_fields(s: DialogSpec, with_start: bool) -> DialogSpec {
    let mut s = s.vars("time", "Variable (failure or end time)", VarKind::Numeric, 1, 1);
    if with_start {
        s = s.vars("start", "Start variable (arbitrary censoring, optional)", VarKind::Numeric, 0, 1);
    }
    s.vars("censor", "Censoring column (optional)", VarKind::Numeric, 0, 1)
        .vars("freq", "Frequency column (optional)", VarKind::Numeric, 0, 1)
        .number(Page::Options, "censor_value", "Censoring value (rows with this value are censored)", 0.0, -1e12, 1e12)
}

pub(super) fn level_field(s: DialogSpec) -> DialogSpec {
    s.number(Page::Options, "level", "Confidence level (%)", 95.0, 0.001, 99.999)
}

// --------------------------------------------------------------------------- distribution analysis

fn dist_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let fam = family(v);
    let fit = fit_distribution(&data, fam)?;
    let mut o = LifeReportOptions {
        level: level(v)?,
        percents: parse_numbers(v.text("percents"), "the percentiles")?,
        times: parse_numbers(v.text("times"), "the times")?,
        charts: v.flag("charts"),
    };
    if o.percents.iter().any(|p| !(*p > 0.0 && *p < 100.0)) {
        return Err("percentiles must be between 0 and 100".into());
    }
    if o.times.iter().any(|t| fam.is_log() && *t <= 0.0) {
        return Err("survival times must be positive for this distribution".into());
    }
    o.times.retain(|t| t.is_finite());
    let km = if v.flag("overlay") && data.obs.iter().all(|x| matches!(x.kind(), ObsKind::Exact | ObsKind::Right)) {
        kaplan_meier(&data, o.level, CiMethod::Plain).ok()
    } else {
        None
    };
    let mut s = life_section(&fit, &o, km.as_ref(), &format!("Distribution Analysis: {name}"));
    if data.n_skipped > 0 {
        s.warnings.push(format!("{} rows with a missing value were left out.", data.n_skipped));
    }
    out(s)
}

// --------------------------------------------------------------------------- distribution ID

fn dist_id_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let fits = rank_distributions(&data);
    if fits.is_empty() {
        return Err("no distribution could be fitted to these data".into());
    }
    let mut s = Section::new(format!("Distribution Identification: {name}"));
    s.text(format!(
        "Observations: {} ({} failures, {} right-censored). Distributions are ordered by AIC; smaller is better.",
        data.total(),
        data.failures(),
        data.right_censored()
    ));
    let mut t = Table::new(
        "Goodness of Fit",
        vec![
            RCol::new("Distribution", ColumnKind::Text),
            RCol::new("Log-Likelihood", ColumnKind::Significant(6)),
            RCol::new("AIC", ColumnKind::Significant(6)),
            RCol::new("BIC", ColumnKind::Significant(6)),
            RCol::new("Parameters", ColumnKind::Text),
        ],
    );
    for f in &fits {
        let ps: Vec<String> = f.parameters(0.95).iter().map(|(n, e)| format!("{n} = {:.6}", e.est)).collect();
        let _ = t.push_row(vec![
            Cell::text(f.family.name()),
            Cell::Num(f.loglik),
            Cell::Num(f.aic()),
            Cell::Num(f.bic()),
            Cell::text(ps.join(", ")),
        ]);
    }
    s.table(t);
    if v.flag("charts") {
        let hi = fits.iter().map(|f| f.percentile(0.99)).filter(|x| x.is_finite()).fold(0.0_f64, f64::max);
        let obs_max =
            data.obs.iter().map(|o| if o.upper.is_finite() { o.upper } else { o.lower }).fold(0.0_f64, f64::max);
        let top = hi.min(obs_max * 2.0).max(obs_max);
        if top > 0.0 {
            let xs: Vec<f64> = (1..=120).map(|i| top * i as f64 / 120.0).collect();
            let mut series: Vec<Series> = fits
                .iter()
                .map(|f| Series {
                    name: f.family.name().into(),
                    x: xs.clone(),
                    y: xs.iter().map(|&t| f.survival(t)).collect(),
                })
                .collect();
            if let Ok(km) = kaplan_meier(&data, 0.95, CiMethod::Plain) {
                series.extend(km.chart_series("", false).into_iter().take(1).map(|mut s| {
                    s.name = "Kaplan-Meier".into();
                    s
                }));
            }
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: "Survival Plot for the Candidate Distributions".into(),
                x_label: "Time".into(),
                y_label: "Probability of Survival".into(),
                categories: Vec::new(),
                series,
                reference_lines: Vec::new(),
            });
        }
    }
    out(s)
}

// --------------------------------------------------------------------------- Kaplan-Meier

fn km_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let lv = level(v)?;
    let method = if v.choice("ci") == 1 { CiMethod::LogLog } else { CiMethod::Plain };
    let kinds = [
        (v.flag("logrank"), WeightKind::LogRank),
        (v.flag("wilcoxon"), WeightKind::Wilcoxon),
        (v.flag("tarone"), WeightKind::TaroneWare),
    ];
    let Some(gc) = v.var("group") else {
        let km = kaplan_meier(&data, lv, method)?;
        let mut s = km.report_section();
        s.heading = format!("Kaplan-Meier: {name}");
        return out(s);
    };
    let lab = labels(ws, gc)?;
    let mut names: Vec<String> = lab.iter().filter(|l| !l.is_empty()).cloned().collect();
    names.sort();
    names.dedup();
    if names.len() < 2 {
        return Err("the grouping column needs at least two groups".into());
    }
    let mut parts: Vec<LifeData> = Vec::new();
    for n in &names {
        let obs: Vec<_> = data.obs.iter().filter(|o| lab.get(o.row).is_some_and(|l| l == n)).copied().collect();
        if obs.is_empty() {
            return Err(format!("group '{n}' has no usable observations").into());
        }
        parts.push(LifeData { obs, n_skipped: 0 });
    }
    let curves = parts.iter().map(|d| kaplan_meier(d, lv, method)).collect::<Result<Vec<_>, _>>()?;
    let mut tests = Vec::new();
    for (on, kind) in kinds {
        if on {
            tests.push(compare_groups(&parts, kind)?);
        }
    }
    let mut s = KmGroups { names, curves, tests }.report_section();
    s.heading = format!("Kaplan-Meier by {}: {name}", col_title(ws, gc));
    out(s)
}

// --------------------------------------------------------------------------- accelerated life

fn alt_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, _) = life_data(ws, v)?;
    let rel = match v.choice("relationship") {
        1 => Relationship::Linear,
        2 => Relationship::Log,
        3 => Relationship::Reciprocal,
        _ => Relationship::Arrhenius,
    };
    let sel = v.vars("stress");
    if sel.is_empty() {
        return Err("select at least one stress column".into());
    }
    let mut stresses = Vec::new();
    for &c in sel {
        stresses.push(StressVariable { name: col_title(ws, c), values: numeric(ws, c)?, relationship: rel });
    }
    let fam = family(v);
    let fit = fit_alt(&data, &stresses, fam)?;
    let use_stress = parse_numbers(v.text("use_stress"), "the use-condition stress")?;
    if !use_stress.is_empty() && use_stress.len() != sel.len() {
        return Err(format!("enter {} use-condition stress value(s), one per stress column", sel.len()).into());
    }
    let percents = parse_numbers(v.text("percents"), "the percentiles")?;
    if percents.iter().any(|p| !(*p > 0.0 && *p < 100.0)) {
        return Err("percentiles must be between 0 and 100".into());
    }
    let times = parse_numbers(v.text("times"), "the times")?;
    let mut s = AltReport { fit, level: level(v)?, use_stress, percents, times }.report_section();
    if data.n_skipped > 0 {
        s.warnings.push(format!("{} rows with a missing value were left out.", data.n_skipped));
    }
    out(s)
}

// --------------------------------------------------------------------------- warranty

fn warranty_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let ship = numeric(ws, v.var("ship").ok_or("select the shipment column")?)?;
    let rets: Vec<Vec<f64>> = v.vars("returns").iter().map(|&c| numeric(ws, c)).collect::<Result<_, _>>()?;
    if rets.iter().any(|c| c.len() != ship.len()) {
        return Err("the shipment and return columns must have the same length".into());
    }
    let mut returns = Vec::with_capacity(ship.len());
    let mut shipments = Vec::with_capacity(ship.len());
    for (i, s) in ship.iter().enumerate() {
        if s.is_nan() {
            if rets.iter().all(|c| c[i].is_nan()) {
                continue; // blank row
            }
            return Err(format!("row {}: the number shipped is missing", i + 1).into());
        }
        shipments.push(*s);
        let mut row = Vec::new();
        let mut ended = false;
        for c in &rets {
            let x = c[i];
            if x.is_nan() {
                ended = true;
            } else if ended {
                return Err(
                    format!("row {}: returns must be filled from the first return column without gaps", i + 1).into()
                );
            } else {
                row.push(x);
            }
        }
        returns.push(row);
    }
    let horizon = v.count("periods");
    let each = v.num("future");
    let future = if each.is_nan() { Vec::new() } else { vec![each; horizon] };
    let fam = match v.choice("dist") {
        1 => Family::Lognormal,
        2 => Family::Exponential,
        3 => Family::Loglogistic,
        _ => Family::Weibull,
    };
    let f = warranty_forecast(&WarrantyInput { shipments, returns, future_shipments: future }, fam, horizon)?;
    out(f.report_section())
}

// --------------------------------------------------------------------------- Cox regression

fn cox_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    use super::analyses_anova::{prepare, retitle};
    let tcol = v.var("time").ok_or("select the time column")?;
    let covs = v.vars("covariates").to_vec();
    let facs = v.vars("factors").to_vec();
    if covs.is_empty() && facs.is_empty() {
        return Err("select at least one covariate or factor".into());
    }
    let mut time = numeric(ws, tcol)?;
    let n = time.len();
    let mut event = vec![true; n];
    if let Some(cc) = v.var("censor") {
        let c = numeric(ws, cc)?;
        let cv = v.num("censor_value");
        if cv.is_nan() {
            return Err("enter the value that marks a censored observation".into());
        }
        for i in 0..n {
            if c[i].is_nan() {
                time[i] = f64::NAN;
            } else {
                event[i] = c[i] != cv;
            }
        }
    }
    if time.iter().any(|&t| t < 0.0) {
        return Err("times must not be negative".into());
    }
    let strata = match v.var("strata") {
        Some(sc) => {
            let lab = labels(ws, sc)?;
            let mut names: Vec<&String> = lab.iter().filter(|l| !l.is_empty()).collect();
            names.sort();
            names.dedup();
            let mut ids = vec![0usize; n];
            for i in 0..n {
                match names.iter().position(|x| **x == lab[i]) {
                    Some(k) => ids[i] = k,
                    None => time[i] = f64::NAN,
                }
            }
            Some(ids)
        }
        None => None,
    };
    let mut all_terms = facs.clone();
    all_terms.extend(&covs);
    if all_terms.contains(&tcol) || v.var("censor").is_some_and(|c| all_terms.contains(&c)) {
        return Err("the time and censoring columns cannot also be model terms".into());
    }
    let prep = prepare(ws, tcol, &facs, &covs)?;
    let text = v.text("model");
    let rhs = if text.trim().is_empty() { prep.default_rhs(1) } else { prep.translate(text)? };
    let ties = if v.choice("ties") == 1 { CoxTies::Breslow } else { CoxTies::Efron };
    let fit = cox_formula(
        &rhs,
        &prep.data,
        &time,
        &event,
        strata.as_deref(),
        &Default::default(),
        &CoxOptions { ties, level: level(v)?, ..CoxOptions::default() },
    )?;
    let mut s = fit.report_section();
    s.heading = format!("Cox Regression: {}", col_title(ws, tcol));
    retitle(&mut s, &prep);
    let mut stored = Vec::new();
    if v.flag("store_lp") {
        stored.push(super::analysis::expand("LINPRED", n, &fit.rows_used, &fit.linear_predictor));
    }
    if v.flag("store_hr") {
        let hr: Vec<f64> = fit.linear_predictor.iter().map(|e| e.exp()).collect();
        stored.push(super::analysis::expand("HAZRATIO", n, &fit.rows_used, &hr));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

pub(crate) fn register(r: &mut Registry) {
    use Page::Options;
    let menu = |last: &'static str| ["Stat", "Reliability/Survival", last];
    add(
        r,
        level_field(
            DialogSpec::new(
                "stat.rel.cox",
                "Cox Regression",
                "Proportional hazards regression for survival times with censoring, tied events and strata.",
            )
            .vars("time", "Survival time", VarKind::Numeric, 1, 1)
            .vars("censor", "Censoring column (optional)", VarKind::Numeric, 0, 1)
            .vars("covariates", "Continuous predictors", VarKind::Numeric, 0, 50)
            .vars("factors", "Categorical predictors", VarKind::Any, 0, 50)
            .vars("strata", "Strata (optional)", VarKind::Any, 0, 1)
            .number(Options, "censor_value", "Censoring value (rows with this value are censored)", 0.0, -1e12, 1e12)
            .text(Options, "model", "Model terms (optional, e.g. Age Sex Age*Sex)", "")
            .choice(Options, "ties", "Ties", &["Efron", "Breslow"], 0)
            .check(Page::Storage, "store_lp", "Linear predictor", false)
            .check(Page::Storage, "store_hr", "Hazard ratio relative to a zero predictor", false),
        ),
        &menu("Cox Regression..."),
        Runner::Report(cox_run),
    );
    add(
        r,
        level_field(add_data_fields(
            DialogSpec::new(
                "stat.rel.dist",
                "Distribution Analysis",
                "Fit a life distribution to failure data with right, left or interval censoring.",
            ),
            true,
        ))
        .choice(Options, "dist", "Assumed distribution", &DISTS, 0)
        .text(Options, "percents", "Percentiles to estimate (%)", "1 5 10 50 90")
        .text(Options, "times", "Times for survival probabilities", "")
        .check(Page::Graphs, "charts", "Survival, hazard and density plots", true)
        .check(Page::Graphs, "overlay", "Overlay the Kaplan-Meier estimate (when possible)", true),
        &menu("Distribution Analysis..."),
        Runner::Report(dist_run),
    );
    add(
        r,
        add_data_fields(
            DialogSpec::new(
                "stat.rel.dist_id",
                "Distribution ID Plot",
                "Fit all six life distributions and rank them by AIC.",
            ),
            true,
        )
        .check(Page::Graphs, "charts", "Survival plot of the candidate distributions", true),
        &menu("Distribution ID Plot..."),
        Runner::Report(dist_id_run),
    );
    add(
        r,
        level_field(add_data_fields(
            DialogSpec::new(
                "stat.rel.km",
                "Nonparametric Distribution Analysis",
                "Kaplan-Meier survival estimate, optionally by group with log-rank type tests.",
            ),
            false,
        ))
        .vars("group", "By variable (optional)", VarKind::Any, 0, 1)
        .choice(Options, "ci", "Confidence interval", &["Plain Greenwood", "Log-log"], 0)
        .check(Options, "logrank", "Log-rank test", true)
        .check(Options, "wilcoxon", "Wilcoxon (Breslow) test", false)
        .check(Options, "tarone", "Tarone-Ware test", false),
        &menu("Nonparametric Distribution Analysis..."),
        Runner::Report(km_run),
    );
    add(
        r,
        level_field(add_data_fields(
            DialogSpec::new(
                "stat.rel.alt",
                "Accelerated Life Testing",
                "Regress life on stress with a Weibull, lognormal or other life distribution and project to use conditions.",
            ),
            false,
        ))
        .vars("stress", "Accelerating variable(s)", VarKind::Numeric, 1, 5)
        .choice(Options, "relationship", "Relationship", &["Arrhenius (temperature, °C)", "Linear", "Log", "Reciprocal"], 0)
        .choice(Options, "dist", "Assumed distribution", &DISTS, 0)
        .text(Options, "use_stress", "Stress at use conditions (one value per accelerating variable)", "")
        .text(Options, "percents", "Percentiles to estimate (%)", "1 10 50")
        .text(Options, "times", "Times for survival probabilities", ""),
        &menu("Accelerated Life Testing..."),
        Runner::Report(alt_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.rel.warranty",
            "Warranty Prediction",
            "Forecast returns from shipment counts and the returns seen so far in each service period.",
        )
        .vars("ship", "Shipment (units shipped per period)", VarKind::Numeric, 1, 1)
        .vars("returns", "Return columns (service period 1, 2, ...)", VarKind::Numeric, 1, 200)
        .choice(Options, "dist", "Assumed distribution", &["Weibull", "Lognormal", "Exponential", "Loglogistic"], 0)
        .integer(Options, "periods", "Periods to forecast", 12.0, 1.0, 1000.0)
        .optional_number(Options, "future", "Units shipped in each future period (blank = none)"),
        &menu("Warranty Prediction..."),
        Runner::Report(warranty_run),
    );
}
