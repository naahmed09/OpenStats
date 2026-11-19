//! Reliability completion dialogs (Phase 5n): repairable systems (mean cumulative function and
//! Crow-AMSAA), demonstration test plans, Turnbull estimation for interval data, threshold and
//! gamma distributions, Eyring accelerated life testing and censored probability plots.

use super::analyses::{add, out};
use super::analyses_rel::{add_data_fields, level_field, life_data, parse_numbers};
use super::analysis::{col_title, labels, level, numeric, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use openstats_core::column::Column;
use openstats_core::Worksheet;
use openstats_stats::reliability::{
    acceptance_probability, crow_amsaa, fit_distribution, fit_eyring, fit_gamma, fit_threshold,
    mean_cumulative_function, plan_sample_size, plan_test_time, systems_from_rows, turnbull, Family, System,
    Termination,
};
use openstats_stats::report::{
    Cell, ChartKind, ChartSpec, Column as TCol, ColumnKind as TKind, Section, Series, Table,
};

use Page::{Graphs, Main, Options, Storage};
use VarKind::{Any, Numeric};

const TCOLS: fn(&[(&str, TKind)]) -> Vec<TCol> = |h| h.iter().map(|(n, k)| TCol::new(*n, *k)).collect();

fn chart(kind: ChartKind, title: &str, xl: &str, yl: &str, series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind,
        title: title.into(),
        x_label: xl.into(),
        y_label: yl.into(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    }
}

fn percents(v: &Values) -> Res<Vec<f64>> {
    let p = parse_numbers(v.text("percents"), "the percentiles")?;
    if p.iter().any(|x| !(*x > 0.0 && *x < 100.0)) {
        return Err("percentiles must be between 0 and 100".into());
    }
    Ok(p)
}

// --------------------------------------------------------------------------- repairable systems

fn systems(ws: &Worksheet, v: &Values) -> Res<Vec<System>> {
    let id = labels(ws, v.var("sys").ok_or("select the system ID column")?)?;
    let t = numeric(ws, v.var("time").ok_or("select the time column")?)?;
    let end = match v.var("end") {
        Some(c) => Some(numeric(ws, c)?),
        None => None,
    };
    Ok(systems_from_rows(&id, &t, end.as_deref())?)
}

fn repairable_fields(s: DialogSpec) -> DialogSpec {
    s.vars("sys", "System ID", Any, 1, 1).vars("time", "Event (repair or failure) time", Numeric, 1, 1).vars(
        "end",
        "End-of-observation time (optional; default is each system's last event)",
        Numeric,
        0,
        1,
    )
}

fn step_points(t: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut xs, mut ys) = (vec![0.0], vec![0.0]);
    let mut prev = 0.0;
    for (&a, &b) in t.iter().zip(y) {
        xs.push(a);
        ys.push(prev);
        xs.push(a);
        ys.push(b);
        prev = b;
    }
    (xs, ys)
}

fn mcf_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let sys = systems(ws, v)?;
    let lev = level(v)?;
    let m = mean_cumulative_function(&sys, lev)?;
    let mut s = Section::new(format!("Mean Cumulative Function: {}", col_title(ws, v.var("time").unwrap_or(0))));
    s.text(format!(
        "Systems: {}\nEvents: {}\nConfidence limits: {}% (normal on the log scale; Lawless-Nadeau variance)",
        sys.len(),
        sys.iter().map(|x| x.events.len()).sum::<usize>(),
        lev * 100.0
    ));
    let mut t = Table::new(
        "Mean Cumulative Function",
        TCOLS(&[
            ("Time", TKind::Significant(6)),
            ("Events", TKind::Integer),
            ("At Risk", TKind::Integer),
            ("MCF", TKind::Fixed(4)),
            ("SE", TKind::Fixed(4)),
            ("Lower", TKind::Fixed(4)),
            ("Upper", TKind::Fixed(4)),
        ]),
    );
    for p in &m {
        t.push_row(vec![
            Cell::Num(p.time),
            Cell::Int(p.events as i64),
            Cell::Int(p.at_risk as i64),
            Cell::Num(p.mcf),
            Cell::Num(p.se),
            Cell::Num(p.lower),
            Cell::Num(p.upper),
        ])?;
    }
    s.table(t);
    if v.flag("charts") {
        let times: Vec<f64> = m.iter().map(|p| p.time).collect();
        let (x, y) = step_points(&times, &m.iter().map(|p| p.mcf).collect::<Vec<_>>());
        let (xl, lo) = step_points(&times, &m.iter().map(|p| p.lower).collect::<Vec<_>>());
        let (xu, hi) = step_points(&times, &m.iter().map(|p| p.upper).collect::<Vec<_>>());
        s.chart(chart(
            ChartKind::Line,
            "Mean Cumulative Function",
            "Time",
            "Mean cumulative number of events",
            vec![
                Series { name: "MCF".into(), x, y },
                Series { name: "Lower".into(), x: xl, y: lo },
                Series { name: "Upper".into(), x: xu, y: hi },
            ],
        ));
    }
    let mut stored = Vec::new();
    if v.flag("store") {
        stored.push(Column::numeric("MCF_TIME", m.iter().map(|p| p.time).collect()));
        stored.push(Column::numeric("MCF", m.iter().map(|p| p.mcf).collect()));
        stored.push(Column::numeric("MCF_LOWER", m.iter().map(|p| p.lower).collect()));
        stored.push(Column::numeric("MCF_UPPER", m.iter().map(|p| p.upper).collect()));
    }
    Ok(Output { section: s, stored, models: Vec::new() })
}

fn crow_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let sys = systems(ws, v)?;
    let lev = level(v)?;
    let term = if v.choice("term") == 1 { Termination::FailureTruncated } else { Termination::TimeTruncated };
    let c = crow_amsaa(&sys, term, lev)?;
    let mut s = Section::new(format!("Crow-AMSAA Reliability Growth: {}", col_title(ws, v.var("time").unwrap_or(0))));
    s.text(format!(
        "Systems: {}   Events: {}   Total observation time: {}\nModel: intensity = λβt^(β-1), expected events per system = λt^β\nβ < 1: reliability is improving; β > 1: it is deteriorating.",
        c.n_systems, c.n_events, c.total_time
    ));
    let mut p = Table::new(
        "Parameter Estimates",
        TCOLS(&[
            ("Parameter", TKind::Text),
            ("Estimate", TKind::Significant(6)),
            ("Lower", TKind::Significant(6)),
            ("Upper", TKind::Significant(6)),
        ]),
    );
    p.push_row(vec![Cell::text("Shape (β)"), Cell::Num(c.beta), Cell::Num(c.beta_lower), Cell::Num(c.beta_upper)])?;
    p.push_row(vec![Cell::text("Shape, bias-corrected"), Cell::Num(c.beta_unbiased), Cell::Missing, Cell::Missing])?;
    p.push_row(vec![Cell::text("Scale (λ)"), Cell::Num(c.lambda), Cell::Missing, Cell::Missing])?;
    s.table(p);
    let mut tt = Table::new(
        "Trend Tests (null: homogeneous Poisson process, β = 1)",
        TCOLS(&[("Test", TKind::Text), ("Statistic", TKind::Fixed(4)), ("P-Value", TKind::PValue)]),
    );
    tt.push_row(vec![Cell::text("2N/β̂ against chi-square"), Cell::Num(c.chi_stat), Cell::Num(c.chi_p)])?;
    tt.push_row(vec![Cell::text("Laplace (z)"), Cell::Num(c.laplace_u), Cell::Num(c.laplace_p)])?;
    s.table(tt);
    let tmax = sys.iter().map(|x| x.end).fold(0.0_f64, f64::max);
    let mut times = parse_numbers(v.text("times"), "the times")?;
    if times.is_empty() {
        times.push(tmax);
    }
    let mut f = Table::new(
        "Intensity and MTBF",
        TCOLS(&[
            ("Time", TKind::Significant(6)),
            ("Expected Events per System", TKind::Significant(5)),
            ("Intensity", TKind::Significant(5)),
            ("Instantaneous MTBF", TKind::Significant(5)),
            ("Cumulative MTBF", TKind::Significant(5)),
        ]),
    );
    for &t in &times {
        if t.is_nan() || t <= 0.0 {
            return Err("times must be positive".into());
        }
        f.push_row(vec![
            Cell::Num(t),
            Cell::Num(c.expected(t)),
            Cell::Num(c.intensity(t)),
            Cell::Num(c.mtbf_instant(t)),
            Cell::Num(c.mtbf_cumulative(t)),
        ])?;
    }
    s.table(f);
    if v.flag("charts") {
        let m = mean_cumulative_function(&sys, lev)?;
        let grid: Vec<f64> = (1..=80).map(|i| tmax * i as f64 / 80.0).collect();
        s.chart(chart(
            ChartKind::Scatter,
            "Cumulative Events per System",
            "Time",
            "Events",
            vec![
                Series {
                    name: "MCF".into(),
                    x: m.iter().map(|p| p.time).collect(),
                    y: m.iter().map(|p| p.mcf).collect(),
                },
                Series { name: "Fit".into(), x: grid.clone(), y: grid.iter().map(|&t| c.expected(t)).collect() },
            ],
        ));
        s.chart(chart(
            ChartKind::Line,
            "Instantaneous MTBF",
            "Time",
            "MTBF",
            vec![Series { name: "MTBF".into(), x: grid.clone(), y: grid.iter().map(|&t| c.mtbf_instant(t)).collect() }],
        ));
    }
    out(s)
}

// --------------------------------------------------------------------------- demonstration plans

fn demo_run(_ws: &Worksheet, v: &Values) -> Res<Output> {
    let shape = v.num("shape");
    let req = v.num("req_time");
    let rel = v.num("reliability");
    let conf = v.num("confidence") / 100.0;
    let c = v.num("failures").max(0.0) as u64;
    let plan = if v.choice("solve") == 0 {
        let t = v.num("test_time");
        if t.is_nan() {
            return Err("enter the test time".into());
        }
        plan_sample_size(shape, req, rel, conf, c, t)?
    } else {
        let n = v.num("n");
        if n.is_nan() {
            return Err("enter the sample size".into());
        }
        plan_test_time(shape, req, rel, conf, c, n as u64)?
    };
    let mut s = Section::new("Test Plan for Reliability Demonstration");
    s.text(format!(
        "Demonstrate R({}) ≥ {} with {}% confidence, Weibull shape {}.\nThe test passes when at most {} of the {} units fail.",
        req,
        rel,
        conf * 100.0,
        shape,
        plan.failures_allowed,
        plan.n
    ));
    let mut t = Table::new(
        "Plan",
        TCOLS(&[
            ("Reliability", TKind::Fixed(4)),
            ("Confidence Level (%)", TKind::Fixed(2)),
            ("Achieved Confidence (%)", TKind::Fixed(2)),
            ("Failures Allowed", TKind::Integer),
            ("Sample Size", TKind::Integer),
            ("Test Time", TKind::Significant(6)),
        ]),
    );
    t.push_row(vec![
        Cell::Num(rel),
        Cell::Num(conf * 100.0),
        Cell::Num(plan.achieved_confidence * 100.0),
        Cell::Int(plan.failures_allowed as i64),
        Cell::Int(plan.n as i64),
        Cell::Num(plan.test_time),
    ])?;
    s.table(t);
    let alts = parse_numbers(v.text("alts"), "the alternative reliabilities")?;
    if !alts.is_empty() {
        let mut o = Table::new(
            "Probability of Passing the Test",
            TCOLS(&[("True Reliability at Requirement Time", TKind::Fixed(4)), ("P(Pass)", TKind::Fixed(4))]),
        );
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for &r in &alts {
            let p = acceptance_probability(&plan, r)?;
            o.push_row(vec![Cell::Num(r), Cell::Num(p)])?;
            xs.push(r);
            ys.push(p);
        }
        s.table(o);
        s.chart(chart(
            ChartKind::Line,
            "Operating Characteristic",
            "True reliability",
            "Probability of passing",
            vec![Series { name: "P(Pass)".into(), x: xs, y: ys }],
        ));
    }
    out(s)
}

// --------------------------------------------------------------------------- Turnbull

fn turnbull_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let tb = turnbull(&data, 1e-10, 50_000)?;
    let mut s = Section::new(format!("Turnbull Nonparametric Estimate: {name}"));
    s.text(format!(
        "Observations: {}   EM iterations: {}   Log-likelihood: {:.4}\nProbability mass sits in the innermost intervals, where the estimate can place it; the survival value is evaluated at the right end of each interval.",
        data.total(),
        tb.iterations,
        tb.loglik
    ));
    let mut t = Table::new(
        "Turnbull Intervals",
        TCOLS(&[
            ("Lower", TKind::Significant(6)),
            ("Upper", TKind::Significant(6)),
            ("Probability", TKind::Fixed(5)),
            ("Survival", TKind::Fixed(5)),
        ]),
    );
    for ((q, p), (m, sv)) in tb.intervals.iter().zip(tb.mass.iter().zip(&tb.survival)) {
        if *m > 1e-9 {
            t.push_row(vec![
                Cell::Num(*q),
                if p.is_finite() { Cell::Num(*p) } else { Cell::text("*") },
                Cell::Num(*m),
                Cell::Num(*sv),
            ])?;
        }
    }
    s.table(t);
    if v.flag("charts") {
        let (mut x, mut y) = (vec![0.0], vec![1.0]);
        for ((_, p), sv) in tb.intervals.iter().zip(&tb.survival) {
            if p.is_finite() {
                x.push(*p);
                y.push(*y.last().unwrap_or(&1.0));
                x.push(*p);
                y.push(*sv);
            }
        }
        s.chart(chart(
            ChartKind::Line,
            "Turnbull Survival Estimate",
            "Time",
            "Survival probability",
            vec![Series { name: "Survival".into(), x, y }],
        ));
    }
    out(s)
}

// --------------------------------------------------------------------------- threshold and gamma

const EXT: [&str; 4] = ["3-Parameter Weibull", "3-Parameter Lognormal", "3-Parameter Loglogistic", "Gamma"];

fn dist_ext_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let lev = level(v)?;
    let pct = percents(v)?;
    let times = parse_numbers(v.text("times"), "the times")?;
    let which = v.choice("dist");
    let mut s = Section::new(format!("{} Distribution Analysis: {name}", EXT[which.min(3)]));
    s.text(format!(
        "Observations: {} ({} failures, {} censored)",
        data.total(),
        data.failures(),
        data.total() - data.failures()
    ));
    let mut ptab = Table::new(
        "Parameter Estimates",
        TCOLS(&[
            ("Parameter", TKind::Text),
            ("Estimate", TKind::Significant(6)),
            ("Std Err", TKind::Significant(5)),
            ("Lower", TKind::Significant(6)),
            ("Upper", TKind::Significant(6)),
        ]),
    );
    let mut fit_tab = Table::new("Goodness of Fit", TCOLS(&[("Statistic", TKind::Text), ("Value", TKind::Fixed(4))]));
    let surv: Box<dyn Fn(f64) -> f64>;
    let quant: Box<dyn Fn(f64) -> f64>;
    let mean;
    let mut warn: Option<String> = None;
    if which == 3 {
        let g = fit_gamma(&data)?;
        for (n, e) in g.parameters(lev) {
            ptab.push_row(vec![
                Cell::text(n),
                Cell::Num(e.est),
                Cell::Num(e.se),
                Cell::Num(e.lower),
                Cell::Num(e.upper),
            ])?;
        }
        fit_tab.push_row(vec![Cell::text("Log-likelihood"), Cell::Num(g.loglik)])?;
        fit_tab.push_row(vec![Cell::text("AIC"), Cell::Num(g.aic())])?;
        if !g.converged {
            warn = Some("The optimiser did not meet its convergence test.".into());
        }
        mean = g.mean();
        let g2 = g.clone();
        surv = Box::new(move |t| g.survival(t));
        quant = Box::new(move |p| g2.percentile(p));
    } else {
        let fam = [Family::Weibull, Family::Lognormal, Family::Loglogistic][which];
        let f = fit_threshold(&data, fam, lev)?;
        for (n, e) in f.fit.parameters(lev) {
            ptab.push_row(vec![
                Cell::text(n),
                Cell::Num(e.est),
                Cell::Num(e.se),
                Cell::Num(e.lower),
                Cell::Num(e.upper),
            ])?;
        }
        ptab.push_row(vec![
            Cell::text("Threshold"),
            Cell::Num(f.threshold),
            Cell::Missing,
            if f.threshold_lower.is_finite() { Cell::Num(f.threshold_lower) } else { Cell::text("*") },
            Cell::Num(f.threshold_upper),
        ])?;
        let two = fit_distribution(&data, fam)?;
        let lr = 2.0 * (f.loglik - two.loglik);
        fit_tab.push_row(vec![Cell::text("Log-likelihood"), Cell::Num(f.loglik)])?;
        fit_tab.push_row(vec![Cell::text("AIC"), Cell::Num(-2.0 * f.loglik + 6.0)])?;
        fit_tab.push_row(vec![Cell::text("Log-likelihood, 2-parameter model"), Cell::Num(two.loglik)])?;
        fit_tab.push_row(vec![Cell::text("AIC, 2-parameter model"), Cell::Num(two.aic())])?;
        fit_tab.push_row(vec![Cell::text("LRT statistic (threshold = 0)"), Cell::Num(lr.max(0.0))])?;
        s.text("The threshold limits come from the profile likelihood (the smallest observed time bounds it above). Parameter rows describe the distribution of time − threshold; the threshold is not included in the Scale / Location values.");
        if !f.interior {
            warn = Some("The likelihood has no interior maximum in the threshold; the estimate is at the edge of the search range.".into());
        }
        mean = f.mean();
        let f2 = f.clone();
        surv = Box::new(move |t| f.survival(t));
        quant = Box::new(move |p| f2.percentile(p));
    }
    s.table(ptab);
    s.table(fit_tab);
    let mut pt =
        Table::new("Percentiles", TCOLS(&[("Percent", TKind::Fixed(2)), ("Percentile", TKind::Significant(6))]));
    for &p in &pct {
        pt.push_row(vec![Cell::Num(p), Cell::Num(quant(p / 100.0))])?;
    }
    s.table(pt);
    if !times.is_empty() {
        let mut st = Table::new(
            "Survival Probabilities",
            TCOLS(&[("Time", TKind::Significant(6)), ("Probability", TKind::Fixed(5))]),
        );
        for &t in &times {
            st.push_row(vec![Cell::Num(t), Cell::Num(surv(t))])?;
        }
        s.table(st);
    }
    s.text(format!("Mean life: {mean:.6}"));
    if let Some(w) = warn {
        s.warn(w);
    }
    if v.flag("charts") {
        let hi = quant(0.995);
        let grid: Vec<f64> = (0..=100).map(|i| hi * i as f64 / 100.0).collect();
        s.chart(chart(
            ChartKind::Line,
            "Survival Plot",
            "Time",
            "Probability",
            vec![Series { name: "Survival".into(), x: grid.clone(), y: grid.iter().map(|&t| surv(t)).collect() }],
        ));
    }
    out(s)
}

// --------------------------------------------------------------------------- Eyring

const LOGFAMS: [(&str, Family); 4] = [
    ("Weibull", Family::Weibull),
    ("Lognormal", Family::Lognormal),
    ("Exponential", Family::Exponential),
    ("Loglogistic", Family::Loglogistic),
];

fn eyring_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let temps = numeric(ws, v.var("temp").ok_or("select the temperature column")?)?;
    let fam = LOGFAMS[v.choice("dist").min(3)].1;
    let lev = level(v)?;
    let fit = fit_eyring(&data, &temps, fam)?;
    let mut s = Section::new(format!("Eyring Accelerated Life Test: {name}"));
    s.text(format!(
        "Distribution: {}   Observations: {}   Log-likelihood: {:.4}\nModel: ln(t) = a − ln(T) + b/T with T in kelvin. The regression below is of ln(t·T) on 1/T, so Intercept = a and the 1/T coefficient = b.",
        fam.name(),
        fit.alt.n,
        fit.loglik
    ));
    let mut ct = Table::new(
        "Regression Table",
        TCOLS(&[
            ("Predictor", TKind::Text),
            ("Coef", TKind::Significant(6)),
            ("SE", TKind::Significant(5)),
            ("Z", TKind::Fixed(2)),
            ("P", TKind::PValue),
            ("Lower", TKind::Significant(6)),
            ("Upper", TKind::Significant(6)),
        ]),
    );
    for (n, e, z, p) in fit.alt.coefficients(lev) {
        let n = if n == "Intercept" { n } else { "1/T (K)".to_string() };
        ct.push_row(vec![
            Cell::text(n),
            Cell::Num(e.est),
            Cell::Num(e.se),
            Cell::Num(z),
            Cell::Num(p),
            Cell::Num(e.lower),
            Cell::Num(e.upper),
        ])?;
    }
    s.table(ct);
    let use_t = v.num("use_temp");
    if !use_t.is_nan() {
        let pct = percents(v)?;
        let life = fit.at_temperature(use_t)?;
        let mut pt = Table::new(
            format!("Percentiles at {use_t} °C"),
            TCOLS(&[
                ("Percent", TKind::Fixed(2)),
                ("Percentile", TKind::Significant(6)),
                ("Lower", TKind::Significant(6)),
                ("Upper", TKind::Significant(6)),
            ]),
        );
        for &p in &pct {
            let e = life.percentile_ci(p / 100.0, lev);
            pt.push_row(vec![Cell::Num(p), Cell::Num(e.est), Cell::Num(e.lower), Cell::Num(e.upper)])?;
        }
        s.table(pt);
        let mut tv: Vec<f64> = temps.iter().copied().filter(|t| !t.is_nan()).collect();
        tv.sort_by(f64::total_cmp);
        tv.dedup();
        let mut af = Table::new(
            "Acceleration Factors (median life)",
            TCOLS(&[("Test Temperature (°C)", TKind::Significant(5)), ("Factor", TKind::Significant(5))]),
        );
        for t in tv {
            af.push_row(vec![Cell::Num(t), Cell::Num(fit.acceleration_factor(use_t, t)?)])?;
        }
        s.table(af);
    }
    out(s)
}

// --------------------------------------------------------------------------- censored probability plot

fn cprob_run(ws: &Worksheet, v: &Values) -> Res<Output> {
    let (data, name) = life_data(ws, v)?;
    let fam = [
        Family::Weibull,
        Family::Lognormal,
        Family::Exponential,
        Family::Normal,
        Family::Logistic,
        Family::Loglogistic,
    ][v.choice("dist").min(5)];
    let fit = fit_distribution(&data, fam)?;
    let tb = turnbull(&data, 1e-10, 50_000)?;
    let mut s = Section::new(format!("Probability Plot for {name} ({}, censored data)", fam.name()));
    s.text("Plotting positions are the Turnbull / Kaplan-Meier estimate of the failure probability at each failure time. The line is the maximum-likelihood fit.");
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let mut cum = 0.0;
    let mut tab = Table::new(
        "Plotting Positions",
        TCOLS(&[("Time", TKind::Significant(6)), ("Percent Failed", TKind::Fixed(3))]),
    );
    for ((_, p), m) in tb.intervals.iter().zip(&tb.mass) {
        cum += m;
        if *m > 1e-9 && p.is_finite() && cum < 1.0 - 1e-9 && (!fam.is_log() || *p > 0.0) {
            xs.push(fam.g(*p));
            ys.push(fam.ppf0(cum));
            tab.push_row(vec![Cell::Num(*p), Cell::Num(cum * 100.0)])?;
        }
    }
    if xs.len() < 2 {
        return Err("fewer than two plotting positions; add failures at different times".into());
    }
    s.table(tab);
    let (lo, hi) = (xs[0], xs[xs.len() - 1]);
    let line = |x: f64| (x - fit.mu) / fit.sigma;
    let yl = match fam {
        Family::Weibull => "ln(-ln(1-F))",
        Family::Normal | Family::Lognormal => "Normal score z",
        _ => "Standardised quantile",
    };
    s.chart(chart(
        ChartKind::Scatter,
        "Probability Plot",
        if fam.is_log() { "ln(Time)" } else { "Time" },
        yl,
        vec![
            Series { name: "Data".into(), x: xs, y: ys },
            Series { name: "Fit".into(), x: vec![lo, hi], y: vec![line(lo), line(hi)] },
        ],
    ));
    out(s)
}

pub(crate) fn register(r: &mut Registry) {
    let menu = |last: &'static str| ["Stat", "Reliability/Survival", last];
    add(
        r,
        level_field(repairable_fields(DialogSpec::new(
            "stat.rel.mcf",
            "Mean Cumulative Function",
            "Nonparametric estimate of the cumulative number of repairs per system.",
        )))
        .check(Graphs, "charts", "Mean cumulative function plot", true)
        .check(Storage, "store", "Store the MCF table", false),
        &["Stat", "Reliability/Survival", "Repairable Systems", "Mean Cumulative Function..."],
        Runner::Report(mcf_run),
    );
    add(
        r,
        level_field(repairable_fields(DialogSpec::new(
            "stat.rel.crow",
            "Crow-AMSAA Reliability Growth",
            "Power-law process fit for repair or failure times with trend tests and MTBF projection.",
        )))
        .choice(Options, "term", "Termination", &["Time truncated", "Failure truncated (one system)"], 0)
        .text(Options, "times", "Times for intensity and MTBF (blank = end of observation)", "")
        .check(Graphs, "charts", "Cumulative events and MTBF plots", true),
        &["Stat", "Reliability/Survival", "Repairable Systems", "Crow-AMSAA Growth Model..."],
        Runner::Report(crow_run),
    );
    add(
        r,
        DialogSpec::new(
            "stat.rel.demo",
            "Test Plan for Reliability Demonstration",
            "Sample size or test time to demonstrate a reliability target for a Weibull life with known shape.",
        )
        .choice(Main, "solve", "Solve for", &["Sample size", "Test time"], 0)
        .number(Main, "shape", "Weibull shape (1 = exponential)", 1.0, 0.0001, 1000.0)
        .number(Main, "req_time", "Requirement time", 1000.0, 1e-12, 1e15)
        .number(Main, "reliability", "Reliability to demonstrate at the requirement time", 0.9, 0.0001, 0.9999)
        .number(Main, "confidence", "Confidence level (%)", 90.0, 0.001, 99.999)
        .integer(Main, "failures", "Failures allowed", 0.0, 0.0, 100000.0)
        .optional_number(Main, "test_time", "Test time per unit (when solving for sample size)")
        .optional_number(Main, "n", "Sample size (when solving for test time)")
        .text(Options, "alts", "True reliabilities for the operating characteristic", "0.8 0.85 0.9 0.95 0.99"),
        &["Stat", "Reliability/Survival", "Test Plans", "Demonstration Test Plan..."],
        Runner::Report(demo_run),
    );
    add(
        r,
        add_data_fields(
            DialogSpec::new(
                "stat.rel.turnbull",
                "Turnbull Estimate (Arbitrary Censoring)",
                "Nonparametric survival estimate for exact, right, left and interval-censored data.",
            ),
            true,
        )
        .check(Graphs, "charts", "Survival plot", true),
        &menu("Turnbull Estimate..."),
        Runner::Report(turnbull_run),
    );
    add(
        r,
        level_field(add_data_fields(
            DialogSpec::new(
                "stat.rel.dist_ext",
                "Distribution Analysis (3-Parameter and Gamma)",
                "Threshold Weibull, lognormal and log-logistic fits, and the gamma distribution, with censoring.",
            ),
            true,
        ))
        .choice(Options, "dist", "Assumed distribution", &EXT, 0)
        .text(Options, "percents", "Percentiles to estimate (%)", "1 5 10 50 90")
        .text(Options, "times", "Times for survival probabilities", "")
        .check(Graphs, "charts", "Survival plot", true),
        &menu("Distribution Analysis (3-Parameter and Gamma)..."),
        Runner::Report(dist_ext_run),
    );
    add(
        r,
        level_field(add_data_fields(
            DialogSpec::new(
                "stat.rel.eyring",
                "Accelerated Life Testing (Eyring)",
                "Eyring temperature-acceleration model for life data.",
            ),
            true,
        ))
        .vars("temp", "Temperature (°C)", Numeric, 1, 1)
        .choice(Options, "dist", "Assumed distribution", &["Weibull", "Lognormal", "Exponential", "Loglogistic"], 0)
        .optional_number(Options, "use_temp", "Use temperature (°C)")
        .text(Options, "percents", "Percentiles to estimate (%)", "1 10 50"),
        &menu("Accelerated Life Testing (Eyring)..."),
        Runner::Report(eyring_run),
    );
    add(
        r,
        level_field(add_data_fields(
            DialogSpec::new(
                "stat.rel.cprob",
                "Probability Plot (Censored Data)",
                "Probability plot with Kaplan-Meier / Turnbull plotting positions and the fitted line.",
            ),
            true,
        ))
        .choice(
            Options,
            "dist",
            "Assumed distribution",
            &["Weibull", "Lognormal", "Exponential", "Normal", "Logistic", "Loglogistic"],
            0,
        ),
        &menu("Probability Plot (Censored Data)..."),
        Runner::Report(cprob_run),
    );
}
