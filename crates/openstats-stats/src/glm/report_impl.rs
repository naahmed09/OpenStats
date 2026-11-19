//! [`Reportable`] implementations for generalized-linear-model results.

use super::ordinal::{CategoricalTests, LogitCoef, NominalFit, OrdinalFit};
use super::stepwise::GlmStepwiseResult;
use super::{hosmer_lemeshow, Association, Classification, Family, GlmFit, HosmerLemeshow, Link, Roc};
use crate::regression::StepAction;
use crate::report::{
    Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Reportable, Section, Series, Table,
};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn ci_text(lo: f64, hi: f64) -> Cell {
    Cell::text(format!("({lo:.6}, {hi:.6})"))
}

fn link_name(l: Link) -> &'static str {
    match l {
        Link::Logit => "Logit",
        Link::Probit => "Probit",
        Link::Cloglog => "Complementary log-log",
        Link::Log => "Log",
        Link::Identity => "Identity",
        Link::Inverse => "Inverse",
    }
}

fn family_name(f: Family) -> &'static str {
    match f {
        Family::Binomial => "Binomial",
        Family::Poisson => "Poisson",
        Family::NegativeBinomial => "Negative binomial",
        Family::Gamma => "Gamma",
        Family::Gaussian => "Gaussian",
    }
}

fn scatter(title: &str, x_label: &str, y_label: &str, x: Vec<f64>, y: Vec<f64>, zero: bool) -> ChartSpec {
    ChartSpec {
        kind: ChartKind::Scatter,
        title: title.into(),
        x_label: x_label.into(),
        y_label: y_label.into(),
        categories: vec![],
        series: vec![Series { name: y_label.into(), x, y }],
        reference_lines: if zero {
            vec![ReferenceLine { axis: Axis::Y, value: 0.0, label: "0".into() }]
        } else {
            vec![]
        },
    }
}

impl Reportable for GlmFit {
    fn report_section(&self) -> Section {
        let heading = match (self.family, self.link) {
            (Family::Binomial, _) => "Binary Logistic Regression".to_string(),
            (Family::Poisson, _) => "Poisson Regression".to_string(),
            (f, _) => format!("Generalized Linear Model ({})", family_name(f)),
        };
        let mut sec = Section::new(heading);
        sec.text(format!("Link function: {}   Family: {}", link_name(self.link), family_name(self.family)));
        let stat_head =
            if self.family == Family::Gamma || self.family == Family::Gaussian { "F-Value" } else { "Chi-Square" };
        let mut dt = Table::new(
            "Deviance Table",
            vec![
                Column::new("Source", T),
                Column::new("DF", I),
                Column::new("Adj Dev", S),
                Column::new("Adj Mean", S),
                Column::new(stat_head, S),
                Column::new("P-Value", P),
            ],
        );
        for r in &self.deviance_table {
            dt.row(vec![
                Cell::text(&r.source),
                int(r.df),
                Cell::Num(r.deviance),
                Cell::opt(r.mean),
                Cell::opt(r.stat),
                Cell::opt(r.p_value),
            ]);
        }
        sec.table(dt);
        let mut ms = Table::new(
            "Model Summary",
            vec![
                Column::new("Deviance R-Sq", ColumnKind::Fixed(2)),
                Column::new("Deviance R-Sq(adj)", ColumnKind::Fixed(2)),
                Column::new("AIC", S),
                Column::new("AICc", S),
                Column::new("BIC", S),
            ],
        );
        ms.row(vec![
            Cell::Num(self.deviance_r2() * 100.0),
            Cell::Num(self.deviance_r2_adj() * 100.0),
            Cell::opt(self.aic),
            Cell::opt(self.aicc),
            Cell::opt(self.bic),
        ]);
        sec.table(ms);
        let stat_name =
            if self.family == Family::Gamma || self.family == Family::Gaussian { "T-Value" } else { "Z-Value" };
        let mut co = Table::new(
            "Coefficients",
            vec![
                Column::new("Term", T),
                Column::new("Coef", S),
                Column::new("SE Coef", S),
                Column::new(format!("{}% CI", self.level * 100.0), T),
                Column::new(stat_name, S),
                Column::new("P-Value", P),
            ],
        );
        for c in &self.coefficients {
            co.row(vec![
                Cell::text(&c.name),
                Cell::Num(c.estimate),
                Cell::Num(c.std_err),
                ci_text(c.ci.lower, c.ci.upper),
                Cell::Num(c.stat),
                Cell::Num(c.p_value),
            ]);
        }
        sec.table(co);
        if matches!(self.link, Link::Logit | Link::Log) && self.coefficients.iter().any(|c| c.ratio.is_some()) {
            let title = if self.link == Link::Logit {
                "Odds Ratios for Continuous/Categorical Predictors"
            } else {
                "Rate Ratios (exp(Coef))"
            };
            let mut rt = Table::new(
                title,
                vec![
                    Column::new("Term", T),
                    Column::new(if self.link == Link::Logit { "Odds Ratio" } else { "Ratio" }, S),
                    Column::new(format!("{}% CI", self.level * 100.0), T),
                ],
            );
            for c in &self.coefficients {
                if let (Some(r), Some(ci)) = (c.ratio, &c.ratio_ci) {
                    rt.row(vec![Cell::text(&c.name), Cell::Num(r), ci_text(ci.lower, ci.upper)]);
                }
            }
            sec.table(rt);
        }
        if let Some((t, se)) = self.theta {
            sec.text(format!(
                "Negative binomial theta = {t:.6}{}",
                if se.is_finite() { format!(" (SE {se:.6})") } else { String::new() }
            ));
        }
        if self.family.estimates_dispersion() {
            sec.text(format!("Dispersion (Pearson χ²/DF) = {:.6}", self.dispersion));
        }
        if let Ok(g) = self.goodness_of_fit() {
            let mut gt = Table::new(
                "Goodness-of-Fit Tests",
                vec![
                    Column::new("Test", T),
                    Column::new("DF", I),
                    Column::new("Chi-Square", S),
                    Column::new("P-Value", P),
                ],
            );
            gt.row(vec![Cell::text("Deviance"), int(g.deviance.1), Cell::Num(g.deviance.0), Cell::Num(g.deviance.2)]);
            gt.row(vec![Cell::text("Pearson"), int(g.pearson.1), Cell::Num(g.pearson.0), Cell::Num(g.pearson.2)]);
            if self.family == Family::Binomial {
                if let Ok(h) = hosmer_lemeshow(self, 10) {
                    gt.row(vec![Cell::text("Hosmer-Lemeshow"), int(h.df), Cell::Num(h.chi_sq), Cell::Num(h.p_value)]);
                }
            }
            sec.table(gt);
        }
        if !self.aliased.is_empty() {
            sec.warn(format!("Columns dropped as aliased: {}", self.aliased.join(", ")));
        }
        if !self.converged {
            sec.warn("The iteration did not converge; the data may show complete or quasi-complete separation.");
        }
        sec.chart(scatter(
            "Deviance Residuals versus Fits",
            "Fitted value",
            "Deviance residual",
            self.fitted.clone(),
            self.deviance_residuals.clone(),
            true,
        ));
        sec
    }
}

impl Reportable for HosmerLemeshow {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Hosmer-Lemeshow Goodness-of-Fit Test");
        let mut t = Table::new(
            "Observed and Expected Frequencies",
            vec![
                Column::new("Group", I),
                Column::new("Cases", S),
                Column::new("Observed", S),
                Column::new("Expected", S),
            ],
        );
        for (i, g) in self.groups.iter().enumerate() {
            t.row(vec![int(i + 1), Cell::Num(g.n), Cell::Num(g.observed), Cell::Num(g.expected)]);
        }
        sec.table(t);
        let mut s =
            Table::new("Test", vec![Column::new("DF", I), Column::new("Chi-Square", S), Column::new("P-Value", P)]);
        s.row(vec![int(self.df), Cell::Num(self.chi_sq), Cell::Num(self.p_value)]);
        sec.table(s);
        sec
    }
}

impl Reportable for Classification {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(format!("Classification Table (cutoff {})", self.cutoff));
        let mut t = Table::new(
            "Observed versus Predicted",
            vec![Column::new("Observed", T), Column::new("Predicted event", S), Column::new("Predicted non-event", S)],
        );
        t.row(vec![Cell::text("Event"), Cell::Num(self.true_pos), Cell::Num(self.false_neg)]);
        t.row(vec![Cell::text("Non-event"), Cell::Num(self.false_pos), Cell::Num(self.true_neg)]);
        sec.table(t);
        let mut m = Table::new(
            "Measures",
            vec![
                Column::new("Accuracy %", ColumnKind::Fixed(2)),
                Column::new("Sensitivity %", ColumnKind::Fixed(2)),
                Column::new("Specificity %", ColumnKind::Fixed(2)),
                Column::new("PPV %", ColumnKind::Fixed(2)),
                Column::new("NPV %", ColumnKind::Fixed(2)),
            ],
        );
        m.row(vec![
            Cell::Num(self.accuracy() * 100.0),
            Cell::Num(self.sensitivity() * 100.0),
            Cell::Num(self.specificity() * 100.0),
            Cell::Num(self.ppv() * 100.0),
            Cell::Num(self.npv() * 100.0),
        ]);
        sec.table(m);
        sec
    }
}

impl Reportable for Roc {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Receiver Operating Characteristic (ROC) Curve");
        sec.text(format!("Area under the ROC curve = {:.6}", self.auc));
        let mut c = scatter("ROC Curve", "1 − Specificity", "Sensitivity", self.fpr.clone(), self.tpr.clone(), false);
        c.kind = ChartKind::Line;
        c.series.push(Series { name: "Chance".into(), x: vec![0.0, 1.0], y: vec![0.0, 1.0] });
        sec.chart(c);
        sec
    }
}

impl Reportable for Association {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Measures of Association");
        let mut t = Table::new(
            "Pairs",
            vec![Column::new("Pairs", T), Column::new("Number", S), Column::new("Percent", ColumnKind::Fixed(1))],
        );
        for (n, v) in [("Concordant", self.concordant), ("Discordant", self.discordant), ("Ties", self.ties)] {
            t.row(vec![Cell::text(n), Cell::Num(v), Cell::Num(100.0 * v / self.pairs)]);
        }
        t.row(vec![Cell::text("Total"), Cell::Num(self.pairs), Cell::Num(100.0)]);
        sec.table(t);
        let mut m = Table::new(
            "Measures",
            vec![
                Column::new("Somers' D", S),
                Column::new("Goodman-Kruskal Gamma", S),
                Column::new("Kendall's Tau-a", S),
            ],
        );
        m.row(vec![Cell::Num(self.somers_d), Cell::Num(self.gamma), Cell::Num(self.tau_a)]);
        sec.table(m);
        sec
    }
}

fn logit_table(title: &str, rows: &[LogitCoef], level: f64) -> Table {
    let mut t = Table::new(
        title,
        vec![
            Column::new("Predictor", T),
            Column::new("Coef", S),
            Column::new("SE Coef", S),
            Column::new("Z", S),
            Column::new("P", P),
            Column::new("Odds Ratio", S),
            Column::new(format!("{}% CI", level * 100.0), T),
        ],
    );
    for c in rows {
        t.row(vec![
            Cell::text(&c.name),
            Cell::Num(c.estimate),
            Cell::Num(c.std_err),
            Cell::Num(c.z),
            Cell::Num(c.p_value),
            Cell::opt(c.odds_ratio),
            match &c.or_ci {
                Some(ci) => ci_text(ci.lower, ci.upper),
                None => Cell::Missing,
            },
        ]);
    }
    t
}

fn tests_blocks(sec: &mut Section, t: &CategoricalTests) {
    sec.text(format!("Log-Likelihood = {:.4}", t.loglik));
    let mut g = Table::new(
        "Test that all slopes are zero",
        vec![Column::new("G", S), Column::new("DF", I), Column::new("P-Value", P)],
    );
    g.row(vec![Cell::Num(t.g), int(t.g_df), Cell::Num(t.g_p)]);
    sec.table(g);
    let mut gof = Table::new(
        "Goodness-of-Fit Tests",
        vec![Column::new("Method", T), Column::new("Chi-Square", S), Column::new("DF", I), Column::new("P", P)],
    );
    if let Some((s, df, p)) = t.pearson {
        gof.row(vec![Cell::text("Pearson"), Cell::Num(s), int(df), Cell::Num(p)]);
    }
    if let Some((s, df, p)) = t.deviance {
        gof.row(vec![Cell::text("Deviance"), Cell::Num(s), int(df), Cell::Num(p)]);
    }
    if t.pearson.is_some() || t.deviance.is_some() {
        sec.table(gof);
    }
}

fn response_info(cats: &[String], counts: &[usize], n: usize, extra: Option<usize>) -> Table {
    let mut t = Table::new("Response Information", vec![Column::new("Value", T), Column::new("Count", I)]);
    for (i, c) in cats.iter().enumerate() {
        let label = if extra == Some(i) { format!("{c} (reference event)") } else { c.clone() };
        t.row(vec![Cell::text(label), int(counts[i])]);
    }
    t.row(vec![Cell::text("Total"), int(n)]);
    t
}

impl Reportable for OrdinalFit {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Ordinal Logistic Regression");
        sec.text(format!("Link function: {}", link_name(self.link)));
        sec.table(response_info(&self.categories, &self.counts, self.n, None));
        let mut all = self.thresholds.clone();
        all.extend(self.coefficients.iter().cloned());
        sec.table(logit_table("Logistic Regression Table", &all, self.level));
        tests_blocks(&mut sec, &self.tests);
        if !self.aliased.is_empty() {
            sec.warn(format!("Columns dropped as aliased: {}", self.aliased.join(", ")));
        }
        if !self.converged {
            sec.warn("The iteration did not converge.");
        }
        sec
    }
}

impl Reportable for NominalFit {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Nominal Logistic Regression");
        sec.table(response_info(&self.categories, &self.counts, self.n, Some(self.reference)));
        for (name, rows) in &self.logits {
            sec.table(logit_table(&format!("Logit {name}"), rows, self.level));
        }
        tests_blocks(&mut sec, &self.tests);
        if !self.aliased.is_empty() {
            sec.warn(format!("Columns dropped as aliased: {}", self.aliased.join(", ")));
        }
        if !self.converged {
            sec.warn("The iteration did not converge; the data may show separation.");
        }
        sec
    }
}

impl Reportable for GlmStepwiseResult {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Stepwise Selection");
        let mut t = Table::new(
            "Steps",
            vec![
                Column::new("Step", I),
                Column::new("Action", T),
                Column::new("Term", T),
                Column::new("P-Value", P),
                Column::new("Terms", I),
                Column::new("Deviance", S),
                Column::new("AIC", S),
            ],
        );
        for s in &self.steps {
            t.row(vec![
                int(s.number),
                Cell::text(if s.action == StepAction::Added { "Added" } else { "Removed" }),
                Cell::text(&s.term),
                Cell::Num(s.p_value),
                int(s.n_terms),
                Cell::Num(s.deviance),
                Cell::opt(s.aic),
            ]);
        }
        sec.table(t);
        sec.text(format!(
            "Selected terms: {}",
            if self.selected.is_empty() { "(none)".to_string() } else { self.selected.join(", ") }
        ));
        let inner = self.fit.report_section();
        for b in inner.blocks {
            sec.blocks.push(b);
        }
        sec.warnings.extend(inner.warnings);
        sec
    }
}
