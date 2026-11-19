//! [`Reportable`] implementations for the reliability results.

use super::alt::AltFit;
use super::fit::{Estimate, LifeFit};
use super::km::{GroupTest, KaplanMeier, WeightKind};
use super::warranty::WarrantyForecast;
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const T: ColumnKind = ColumnKind::Text;

fn pct(level: f64) -> String {
    let p = level * 100.0;
    if (p - p.round()).abs() < 1e-9 {
        format!("{p:.0}")
    } else {
        format!("{p}")
    }
}

fn est_row(name: &str, e: &Estimate) -> Vec<Cell> {
    vec![Cell::text(name), Cell::Num(e.est), Cell::Num(e.se), Cell::Num(e.lower), Cell::Num(e.upper)]
}

fn est_table(title: &str, first: &str, level: f64) -> Table {
    let l = pct(level);
    Table::new(
        title,
        vec![
            Column::new(first, T),
            Column::new("Estimate", S),
            Column::new("Standard Error", S),
            Column::new(format!("{l}% Normal CI Lower"), S),
            Column::new(format!("{l}% Normal CI Upper"), S),
        ],
    )
}

fn line(title: &str, xl: &str, yl: &str, series: Vec<Series>) -> ChartSpec {
    ChartSpec {
        kind: ChartKind::Line,
        title: title.into(),
        x_label: xl.into(),
        y_label: yl.into(),
        categories: Vec::new(),
        series,
        reference_lines: Vec::new(),
    }
}

/// Options for the Distribution Analysis report.
#[derive(Debug, Clone)]
pub struct LifeReportOptions {
    /// Confidence level.
    pub level: f64,
    /// Percentages for the percentile table (e.g. `1.0` for the first percentile).
    pub percents: Vec<f64>,
    /// Times at which to tabulate the reliability.
    pub times: Vec<f64>,
    /// Draw the survival, hazard and density plots.
    pub charts: bool,
}

impl Default for LifeReportOptions {
    fn default() -> Self {
        LifeReportOptions { level: 0.95, percents: vec![1.0, 5.0, 10.0, 50.0, 90.0], times: Vec::new(), charts: true }
    }
}

fn grid(fit: &LifeFit) -> Vec<f64> {
    let (lo, hi) = (fit.percentile(0.0005), fit.percentile(0.9995));
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Vec::new();
    }
    let n = 120;
    (0..=n).map(|i| lo + (hi - lo) * i as f64 / n as f64).collect()
}

/// Build the section for a fitted distribution, with an optional Kaplan–Meier overlay.
pub fn life_section(fit: &LifeFit, opts: &LifeReportOptions, km: Option<&KaplanMeier>, heading: &str) -> Section {
    let mut s = Section::new(heading);
    s.text(format!(
        "Distribution: {}\nObservations: {} ({} failures, {} right-censored)",
        fit.family.name(),
        fit.n,
        fit.failures,
        fit.censored
    ));
    if !fit.converged {
        s.warnings.push("The optimiser did not meet its convergence test; check the results.".into());
    }
    if fit.cov.is_none() {
        s.warnings.push("The likelihood surface is flat: standard errors and intervals are not available.".into());
    }
    let mut t = est_table("Distribution Parameters", "Parameter", opts.level);
    for (name, e) in fit.parameters(opts.level) {
        let _ = t.push_row(est_row(&name, &e));
    }
    s.table(t);
    let mut g = Table::new(
        "Goodness of Fit",
        vec![Column::new("Log-Likelihood", S), Column::new("AIC", S), Column::new("BIC", S)],
    );
    let _ = g.push_row(vec![Cell::Num(fit.loglik), Cell::Num(fit.aic()), Cell::Num(fit.bic())]);
    s.table(g);

    let mut c = est_table("Characteristics of Distribution", "", opts.level);
    let _ = c.push_row(est_row("Mean(MTTF)", &fit.mean_ci(opts.level)));
    let nan = f64::NAN;
    let _ = c.push_row(vec![
        Cell::text("Standard Deviation"),
        Cell::Num(fit.std_dev()),
        Cell::Num(nan),
        Cell::Num(nan),
        Cell::Num(nan),
    ]);
    for (label, p) in [("Median", 0.5), ("First Quartile(Q1)", 0.25), ("Third Quartile(Q3)", 0.75)] {
        let _ = c.push_row(est_row(label, &fit.percentile_ci(p, opts.level)));
    }
    let _ = c.push_row(vec![
        Cell::text("Interquartile Range(IQR)"),
        Cell::Num(fit.percentile(0.75) - fit.percentile(0.25)),
        Cell::Num(nan),
        Cell::Num(nan),
        Cell::Num(nan),
    ]);
    s.table(c);

    if !opts.percents.is_empty() {
        let mut pt = est_table("Table of Percentiles", "Percent", opts.level);
        for &p in &opts.percents {
            if p > 0.0 && p < 100.0 {
                let mut row = est_row("", &fit.percentile_ci(p / 100.0, opts.level));
                row[0] = Cell::Num(p);
                let _ = pt.push_row(row);
            }
        }
        pt.columns[0].kind = ColumnKind::Significant(4);
        s.table(pt);
    }
    if !opts.times.is_empty() {
        let l = pct(opts.level);
        let mut rt = Table::new(
            "Table of Survival Probabilities",
            vec![
                Column::new("Time", S),
                Column::new("Probability", S),
                Column::new(format!("{l}% Normal CI Lower"), S),
                Column::new(format!("{l}% Normal CI Upper"), S),
            ],
        );
        for &t in &opts.times {
            let e = fit.reliability_ci(t, opts.level);
            let _ = rt.push_row(vec![Cell::Num(t), Cell::Num(e.est), Cell::Num(e.lower), Cell::Num(e.upper)]);
        }
        s.table(rt);
    }
    if opts.charts {
        let xs = grid(fit);
        if !xs.is_empty() {
            let name = fit.family.name();
            let mut surv = vec![Series {
                name: format!("{name} fit"),
                x: xs.clone(),
                y: xs.iter().map(|&t| fit.survival(t)).collect(),
            }];
            if let Some(k) = km {
                let (x, y) = step_xy(k, |st| st.survival, 1.0);
                surv.push(Series { name: "Kaplan-Meier".into(), x, y });
            }
            s.chart(line(&format!("Survival Plot for {name}"), "Time", "Probability of Survival", surv));
            s.chart(line(
                &format!("Hazard Function for {name}"),
                "Time",
                "Hazard rate",
                vec![Series { name: "Hazard".into(), x: xs.clone(), y: xs.iter().map(|&t| fit.hazard(t)).collect() }],
            ));
            s.chart(line(
                &format!("Probability Density Function for {name}"),
                "Time",
                "Density",
                vec![Series { name: "Density".into(), x: xs.clone(), y: xs.iter().map(|&t| fit.pdf(t)).collect() }],
            ));
        }
    }
    s
}

impl Reportable for LifeFit {
    fn report_section(&self) -> Section {
        life_section(
            self,
            &LifeReportOptions::default(),
            None,
            &format!("Distribution Analysis: {}", self.family.name()),
        )
    }
}

/// Step-function coordinates of a Kaplan–Meier column.
fn step_xy(km: &KaplanMeier, f: impl Fn(&super::km::KmStep) -> f64, start: f64) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (vec![0.0], vec![start]);
    let mut prev = start;
    for st in &km.steps {
        x.push(st.time);
        y.push(prev);
        prev = f(st);
        x.push(st.time);
        y.push(prev);
    }
    (x, y)
}

impl KaplanMeier {
    /// The survival and (if present) interval series for a chart, named with `prefix`.
    pub fn chart_series(&self, prefix: &str, with_ci: bool) -> Vec<Series> {
        let (x, y) = step_xy(self, |s| s.survival, 1.0);
        let mut v = vec![Series { name: format!("{prefix}Survival"), x, y }];
        if with_ci {
            let (xl, yl) = step_xy(self, |s| s.lower, 1.0);
            let (xu, yu) = step_xy(self, |s| s.upper, 1.0);
            v.push(Series { name: format!("{prefix}Lower"), x: xl, y: yl });
            v.push(Series { name: format!("{prefix}Upper"), x: xu, y: yu });
        }
        v
    }

    fn summary_rows(&self, label: &str, t: &mut Table) {
        let f = |x: f64| Cell::Num(x);
        let _ = t.push_row(vec![
            Cell::text(label),
            f(self.n),
            f(self.failures),
            f(self.mean),
            f(self.mean_se),
            f(self.median),
            f(self.q1.0),
            f(self.q3.0),
        ]);
    }
}

fn km_summary_table(title: &str, first: &str) -> Table {
    Table::new(
        title,
        vec![
            Column::new(first, T),
            Column::new("N", ColumnKind::Integer),
            Column::new("Failures", ColumnKind::Integer),
            Column::new("Mean", S),
            Column::new("SE of Mean", S),
            Column::new("Median", S),
            Column::new("Q1", S),
            Column::new("Q3", S),
        ],
    )
}

fn km_table(k: &KaplanMeier) -> Table {
    let l = pct(k.level);
    let mut t = Table::new(
        "Kaplan-Meier Estimates",
        vec![
            Column::new("Time", S),
            Column::new("Number at Risk", ColumnKind::Integer),
            Column::new("Number Failed", ColumnKind::Integer),
            Column::new("Survival Probability", S),
            Column::new("Standard Error", S),
            Column::new(format!("{l}% Lower"), S),
            Column::new(format!("{l}% Upper"), S),
        ],
    );
    for st in k.steps.iter().filter(|s| s.failed > 0.0) {
        let _ = t.push_row(vec![
            Cell::Num(st.time),
            Cell::Int(st.at_risk.round() as i64),
            Cell::Int(st.failed.round() as i64),
            Cell::Num(st.survival),
            Cell::Num(st.se),
            Cell::Num(st.lower),
            Cell::Num(st.upper),
        ]);
    }
    t
}

impl Reportable for KaplanMeier {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Nonparametric Distribution Analysis: Kaplan-Meier");
        let mut t = km_summary_table("Characteristics of Variable", "");
        self.summary_rows("All", &mut t);
        if self.mean_restricted {
            t.footnotes.push("The mean is restricted to the largest observation, which is censored.".into());
        }
        s.table(t);
        s.table(km_table(self));
        s.chart(line("Survival Plot", "Time", "Probability of Survival", self.chart_series("", true)));
        s
    }
}

/// Kaplan–Meier curves of several groups with the comparison tests.
#[derive(Debug, Clone)]
pub struct KmGroups {
    /// Group labels.
    pub names: Vec<String>,
    /// Estimate of each group.
    pub curves: Vec<KaplanMeier>,
    /// Comparison tests (log-rank, Wilcoxon, …).
    pub tests: Vec<GroupTest>,
}

impl Reportable for KmGroups {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Nonparametric Distribution Analysis: Kaplan-Meier by Group");
        let mut t = km_summary_table("Characteristics of Variable", "Group");
        for (n, c) in self.names.iter().zip(&self.curves) {
            c.summary_rows(n, &mut t);
        }
        s.table(t);
        if !self.tests.is_empty() {
            let mut cmp = Table::new(
                "Comparison of Survival Curves",
                vec![
                    Column::new("Method", T),
                    Column::new("Chi-Square", S),
                    Column::new("DF", ColumnKind::Integer),
                    Column::new("P-Value", P),
                ],
            );
            for g in &self.tests {
                let name = match g.kind {
                    WeightKind::LogRank => "Log-Rank",
                    WeightKind::Wilcoxon => "Wilcoxon",
                    WeightKind::TaroneWare => "Tarone-Ware",
                };
                let _ = cmp.push_row(vec![
                    Cell::text(name),
                    Cell::Num(g.statistic),
                    Cell::Int(g.df as i64),
                    Cell::Num(g.p_value),
                ]);
            }
            s.table(cmp);
            let mut oe = Table::new(
                "Observed and Expected Failures",
                vec![Column::new("Group", T), Column::new("Observed", S), Column::new("Expected", S)],
            );
            for (i, n) in self.names.iter().enumerate() {
                let _ = oe.push_row(vec![
                    Cell::text(n.clone()),
                    Cell::Num(self.tests[0].observed[i]),
                    Cell::Num(self.tests[0].expected[i]),
                ]);
            }
            s.table(oe);
        }
        let series: Vec<Series> =
            self.names.iter().zip(&self.curves).flat_map(|(n, c)| c.chart_series(&format!("{n} "), false)).collect();
        s.chart(line("Survival Plot", "Time", "Probability of Survival", series));
        s
    }
}

/// An accelerated-life fit with the report settings.
#[derive(Debug, Clone)]
pub struct AltReport {
    /// The model.
    pub fit: AltFit,
    /// Confidence level.
    pub level: f64,
    /// Stress values at which to describe the life distribution.
    pub use_stress: Vec<f64>,
    /// Percentages for the percentile table.
    pub percents: Vec<f64>,
    /// Times for the survival table.
    pub times: Vec<f64>,
}

impl Reportable for AltReport {
    fn report_section(&self) -> Section {
        let f = &self.fit;
        let mut s = Section::new(format!("Accelerated Life Testing: {}", f.family.name()));
        let rels: Vec<String> = f.stresses.iter().map(|(n, r)| format!("{n} ({})", r.name())).collect();
        s.text(format!(
            "Distribution: {}\nStress: {}\nObservations: {} ({} failures)",
            f.family.name(),
            rels.join(", "),
            f.n,
            f.failures
        ));
        if !f.converged {
            s.warnings.push("The optimiser did not meet its convergence test; check the results.".into());
        }
        let l = pct(self.level);
        let mut t = Table::new(
            "Regression Table",
            vec![
                Column::new("Predictor", T),
                Column::new("Coef", S),
                Column::new("Standard Error", S),
                Column::new("Z", ColumnKind::Fixed(2)),
                Column::new("P", P),
                Column::new(format!("{l}% Normal CI Lower"), S),
                Column::new(format!("{l}% Normal CI Upper"), S),
            ],
        );
        for (name, e, z, p) in f.coefficients(self.level) {
            let _ = t.push_row(vec![
                Cell::text(name),
                Cell::Num(e.est),
                Cell::Num(e.se),
                Cell::Num(z),
                Cell::Num(p),
                Cell::Num(e.lower),
                Cell::Num(e.upper),
            ]);
        }
        let (name, e) = f.scale_estimate(self.level);
        let nan = f64::NAN;
        let _ = t.push_row(vec![
            Cell::text(name),
            Cell::Num(e.est),
            Cell::Num(e.se),
            Cell::Num(nan),
            Cell::Num(nan),
            Cell::Num(e.lower),
            Cell::Num(e.upper),
        ]);
        s.table(t);
        let k = f.beta.len() + usize::from(!f.family.fixed_scale());
        let mut g = Table::new(
            "Log-Likelihood",
            vec![Column::new("Log-Likelihood", S), Column::new("AIC", S), Column::new("BIC", S)],
        );
        let _ = g.push_row(vec![
            Cell::Num(f.loglik),
            Cell::Num(-2.0 * f.loglik + 2.0 * k as f64),
            Cell::Num(-2.0 * f.loglik + k as f64 * f.n.ln()),
        ]);
        s.table(g);
        if self.use_stress.len() == f.stresses.len() {
            match f.at_stress(&self.use_stress) {
                Ok(life) => {
                    let label = self.use_stress.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ");
                    let mut pt = est_table(&format!("Table of Percentiles at stress {label}"), "Percent", self.level);
                    for &p in &self.percents {
                        if p > 0.0 && p < 100.0 {
                            let mut row = est_row("", &life.percentile_ci(p / 100.0, self.level));
                            row[0] = Cell::Num(p);
                            let _ = pt.push_row(row);
                        }
                    }
                    pt.columns[0].kind = ColumnKind::Significant(4);
                    s.table(pt);
                    if !self.times.is_empty() {
                        let mut rt = Table::new(
                            format!("Table of Survival Probabilities at stress {label}"),
                            vec![
                                Column::new("Time", S),
                                Column::new("Probability", S),
                                Column::new(format!("{l}% Normal CI Lower"), S),
                                Column::new(format!("{l}% Normal CI Upper"), S),
                            ],
                        );
                        for &t in &self.times {
                            let e = life.reliability_ci(t, self.level);
                            let _ = rt.push_row(vec![
                                Cell::Num(t),
                                Cell::Num(e.est),
                                Cell::Num(e.lower),
                                Cell::Num(e.upper),
                            ]);
                        }
                        s.table(rt);
                    }
                    let xs = grid(&life);
                    if !xs.is_empty() {
                        s.chart(line(
                            &format!("Survival Plot at stress {label}"),
                            "Time",
                            "Probability of Survival",
                            vec![Series {
                                name: "Survival".into(),
                                x: xs.clone(),
                                y: xs.iter().map(|&t| life.survival(t)).collect(),
                            }],
                        ));
                    }
                }
                Err(e) => s.warnings.push(format!("Use stress not evaluated: {e}")),
            }
        }
        s
    }
}

impl Reportable for WarrantyForecast {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Warranty Prediction");
        s.text(format!(
            "Distribution: {}\nReturns so far: {}\nUnits still in service: {}",
            self.fit.family.name(),
            self.returned,
            self.in_service
        ));
        let mut t = est_table("Distribution Parameters", "Parameter", 0.95);
        for (name, e) in self.fit.parameters(0.95) {
            let _ = t.push_row(est_row(&name, &e));
        }
        s.table(t);
        let mut f = Table::new(
            "Forecast of Returns",
            vec![
                Column::new("Period", ColumnKind::Integer),
                Column::new("Expected Returns", ColumnKind::Fixed(2)),
                Column::new("Cumulative", ColumnKind::Fixed(2)),
            ],
        );
        for ((p, e), c) in self.periods.iter().zip(&self.expected).zip(&self.cumulative) {
            let _ = f.push_row(vec![Cell::Int(*p as i64), Cell::Num(*e), Cell::Num(*c)]);
        }
        s.table(f);
        s.chart(line(
            "Expected Returns per Period",
            "Period",
            "Expected returns",
            vec![Series {
                name: "Expected".into(),
                x: self.periods.iter().map(|&p| p as f64).collect(),
                y: self.expected.clone(),
            }],
        ));
        s
    }
}

impl Reportable for super::cox::CoxFit {
    fn report_section(&self) -> Section {
        use super::cox::CoxTies;
        let mut s = Section::new("Cox Regression");
        let ties = match self.ties {
            CoxTies::Breslow => "Breslow",
            CoxTies::Efron => "Efron",
        };
        let mut sum = Table::new(
            "Model Summary",
            vec![
                Column::new("N", ColumnKind::Integer),
                Column::new("Events", ColumnKind::Integer),
                Column::new("Strata", ColumnKind::Integer),
                Column::new("Log-likelihood", S),
                Column::new("Concordance", S),
            ],
        );
        sum.row(vec![
            Cell::Int(self.n as i64),
            Cell::Int(self.events as i64),
            Cell::Int(self.n_strata as i64),
            Cell::Num(self.loglik),
            Cell::Num(self.concordance),
        ]);
        sum.footnotes.push(format!("Ties handled by the {ties} method."));
        if !self.converged {
            sum.footnotes.push("The Newton iteration did not converge.".into());
        }
        s.table(sum);
        let mut t = Table::new(
            "Coefficients",
            vec![
                Column::new("Term", T),
                Column::new("Coef", S),
                Column::new("SE Coef", S),
                Column::new("Z-Value", S),
                Column::new("P-Value", P),
                Column::new("Hazard Ratio", S),
                Column::new("HR Lower", S),
                Column::new("HR Upper", S),
            ],
        );
        for c in &self.coefficients {
            t.row(vec![
                Cell::text(c.name.clone()),
                Cell::Num(c.coef),
                Cell::Num(c.se),
                Cell::Num(c.z),
                Cell::Num(c.p_value),
                Cell::Num(c.hazard_ratio),
                Cell::Num(c.hr_lower),
                Cell::Num(c.hr_upper),
            ]);
        }
        s.table(t);
        let mut g = Table::new(
            "Tests of the Global Null Hypothesis (all coefficients are 0)",
            vec![
                Column::new("Test", T),
                Column::new("Chi-Square", S),
                Column::new("DF", ColumnKind::Integer),
                Column::new("P-Value", P),
            ],
        );
        for (name, v, p) in [
            ("Likelihood ratio", self.lr, self.p_lr),
            ("Wald", self.wald, self.p_wald),
            ("Score", self.score, self.p_score),
        ] {
            g.row(vec![Cell::text(name), Cell::Num(v), Cell::Int(self.df as i64), Cell::Num(p)]);
        }
        s.table(g);
        s
    }
}
