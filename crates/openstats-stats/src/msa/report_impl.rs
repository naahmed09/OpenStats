//! [`Reportable`] implementation for Gage R&R results.

use super::types::{AnovaRow, GageMethod, GageResult, Source};
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn anova_table(title: &str, rows: &[AnovaRow]) -> Table {
    let mut t = Table::new(
        title,
        vec![
            Column::new("Source", T),
            Column::new("DF", I),
            Column::new("SS", S),
            Column::new("MS", S),
            Column::new("F", S),
            Column::new("P", P),
        ],
    );
    for r in rows {
        t.row(vec![Cell::text(&r.source), int(r.df), Cell::Num(r.ss), Cell::opt(r.ms), Cell::opt(r.f), Cell::opt(r.p)]);
    }
    t
}

fn label(s: Source) -> String {
    match s {
        Source::Operator | Source::OperatorByPart => format!("  {}", s.label()),
        Source::Repeatability | Source::Reproducibility => format!(" {}", s.label()),
        _ => s.label().to_string(),
    }
}

impl Reportable for GageResult {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(match self.method {
            GageMethod::CrossedAnova => "Gage R&R Study - ANOVA Method",
            GageMethod::CrossedXbarR => "Gage R&R Study - Xbar-R Method",
            GageMethod::NestedAnova => "Gage R&R Study (Nested) - ANOVA Method",
        });
        let design = if self.method == GageMethod::NestedAnova {
            format!(
                "{} operators, {} parts per operator (nested), {} replicates.",
                self.operators, self.parts, self.replicates
            )
        } else {
            format!("{} parts, {} operators, {} replicates.", self.parts, self.operators, self.replicates)
        };
        sec.text(design);
        if let Some(full) = &self.anova_with_interaction {
            sec.table(anova_table("Two-Way ANOVA Table With Operator * Part Interaction", full));
            sec.text("The Operator * Part interaction was removed from the model.");
            sec.table(anova_table("Two-Way ANOVA Table Without Interaction", &self.anova));
        } else if !self.anova.is_empty() {
            let title = match self.method {
                GageMethod::NestedAnova => "Nested ANOVA Table",
                _ => "Two-Way ANOVA Table With Operator * Part Interaction",
            };
            sec.table(anova_table(title, &self.anova));
        }
        if let Some(p) = self.interaction_p {
            sec.text(format!(
                "Operator * Part interaction: p = {p:.4}{}.",
                if self.interaction_dropped { " (at or above the removal level)" } else { "" }
            ));
        }
        let mut vc = Table::new(
            "Gage R&R: Variance Components",
            vec![
                Column::new("Source", T),
                Column::new("VarComp", S),
                Column::new("%Contribution (of VarComp)", ColumnKind::Fixed(2)),
            ],
        );
        for r in &self.components {
            vc.row(vec![Cell::text(label(r.source)), Cell::Num(r.var), Cell::Num(r.pct_contribution)]);
        }
        sec.table(vc);
        let k = self.study_sigma;
        let mut cols = vec![
            Column::new("Source", T),
            Column::new("StdDev (SD)", S),
            Column::new(format!("Study Var ({k} × SD)"), S),
            Column::new("%Study Var (%SV)", ColumnKind::Fixed(2)),
        ];
        if self.tolerance.is_some() {
            cols.push(Column::new("%Tolerance (SV/Toler)", ColumnKind::Fixed(2)));
        }
        let mut sv = Table::new("Gage Evaluation", cols);
        for r in &self.components {
            let mut row =
                vec![Cell::text(label(r.source)), Cell::Num(r.sd), Cell::Num(r.study_var), Cell::Num(r.pct_study_var)];
            if let Some(t) = r.pct_tolerance {
                row.push(Cell::Num(t));
            }
            sv.row(row);
        }
        if let Some(t) = self.tolerance {
            sv.footnotes.push(format!("Process tolerance = {t}"));
        }
        sec.table(sv);
        match self.ndc {
            Some(n) => sec.text(format!("Number of Distinct Categories = {n}")),
            None => sec.text("Number of Distinct Categories is undefined: the gage variation is 0."),
        };
        if let Some(x) = &self.xbar_r {
            let mut t = Table::new("Xbar-R Method Detail", vec![Column::new("Quantity", T), Column::new("Value", S)]);
            for (n, v) in [
                ("Average range (R-bar)", x.r_bar),
                ("Range of operator means", x.operator_range),
                ("Range of part means", x.part_range),
                ("Equipment variation (EV)", x.equipment_variation),
                ("Appraiser variation (AV)", x.appraiser_variation),
                ("Part variation (PV)", x.part_variation),
            ] {
                t.row(vec![Cell::text(n), Cell::Num(v)]);
            }
            sec.table(t);
        }
        if self.operator_fixed {
            sec.text("Operators are a fixed factor: the operator (reproducibility) component is the variance of the operator effects, (o−1)/o × (MS operator − MS operator×part)/(parts × replicates).");
        }
        for s in &self.negative_set_to_zero {
            sec.warn(format!("The variance component estimate for {} was negative and was set to 0.", s.label()));
        }
        for c in &self.charts {
            sec.chart(c.clone());
        }
        sec
    }
}

// ------------------------------------------------- linearity and bias

use super::attribute::{Agreement, AttributeAgreement, CohenKappa, KappaRow};
use super::linearity::LinearityBias;

impl Reportable for LinearityBias {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Gage Linearity and Bias Study");
        sec.text(format!(
            "{} measurements; process variation {}; {:.0}% confidence intervals.",
            self.n,
            self.process_variation,
            self.level * 100.0
        ));
        let r = &self.regression;
        let mut t = Table::new(
            "Regression of Bias on Reference Value",
            vec![
                Column::new("Predictor", T),
                Column::new("Coef", S),
                Column::new("SE Coef", S),
                Column::new("T", S),
                Column::new("P", P),
                Column::new("Lower", S),
                Column::new("Upper", S),
            ],
        );
        for (name, c) in [("Constant", &r.intercept), ("Slope", &r.slope)] {
            t.row(vec![
                Cell::text(name),
                Cell::Num(c.coef),
                Cell::Num(c.se),
                Cell::Num(c.t),
                Cell::Num(c.p),
                Cell::Num(c.ci.lower),
                Cell::Num(c.ci.upper),
            ]);
        }
        t.footnotes.push(format!(
            "S = {:.6}; R-Sq = {}; residual df = {}.",
            r.s,
            r.r_squared.map_or("*".to_string(), |v| format!("{:.2}%", v * 100.0)),
            r.df
        ));
        sec.table(t);
        let mut l = Table::new("Gage Linearity", vec![Column::new("Item", T), Column::new("Value", S)]);
        l.row(vec![Cell::text("Linearity"), Cell::Num(self.linearity)]);
        l.row(vec![Cell::text("% Linearity"), Cell::Num(self.pct_linearity)]);
        sec.table(l);
        let mut b = Table::new(
            "Gage Bias",
            vec![
                Column::new("Reference", T),
                Column::new("N", I),
                Column::new("Mean", S),
                Column::new("Bias", S),
                Column::new("% Bias", S),
                Column::new("P", P),
            ],
        );
        for row in self.biases.iter().chain(std::iter::once(&self.average_bias)) {
            b.row(vec![
                row.reference.map_or_else(|| Cell::text("Average"), |v| Cell::text(format!("{v}"))),
                int(row.n),
                Cell::Num(row.mean_measured),
                Cell::Num(row.bias),
                Cell::Num(row.pct_bias),
                Cell::opt(row.p),
            ]);
        }
        sec.table(b);
        sec.chart(self.chart_spec());
        sec
    }
}

// --------------------------------------------------- attribute agreement

fn agreement_table(title: &str, rows: &[&Agreement], level: f64) -> Table {
    let mut t = Table::new(
        title,
        vec![
            Column::new("Appraiser", T),
            Column::new("# Inspected", I),
            Column::new("# Matched", I),
            Column::new("Percent", ColumnKind::Fixed(2)),
            Column::new(format!("{:.0}% CI lower", level * 100.0), ColumnKind::Fixed(2)),
            Column::new(format!("{:.0}% CI upper", level * 100.0), ColumnKind::Fixed(2)),
        ],
    );
    for a in rows {
        t.row(vec![
            Cell::text(&a.label),
            int(a.inspected),
            int(a.matched),
            Cell::Num(a.percent),
            Cell::Num(a.ci.lower),
            Cell::Num(a.ci.upper),
        ]);
    }
    t
}

fn kappa_table(title: &str, rows: &[KappaRow]) -> Table {
    let mut t = Table::new(
        title,
        vec![
            Column::new("Response", T),
            Column::new("Kappa", ColumnKind::Fixed(4)),
            Column::new("SE Kappa", ColumnKind::Fixed(4)),
            Column::new("Z", ColumnKind::Fixed(4)),
            Column::new("P(vs > 0)", P),
        ],
    );
    for r in rows {
        t.row(vec![Cell::text(&r.category), Cell::Num(r.kappa), Cell::Num(r.se), Cell::Num(r.z), Cell::Num(r.p)]);
    }
    t
}

fn cohen_row(t: &mut Table, label: &str, c: &CohenKappa) {
    t.row(vec![
        Cell::text(label),
        Cell::Num(c.kappa),
        Cell::Num(c.se_null),
        Cell::Num(c.z),
        Cell::Num(c.p),
        Cell::Num(c.ci.lower),
        Cell::Num(c.ci.upper),
    ]);
}

fn cohen_table(title: &str) -> Table {
    Table::new(
        title,
        vec![
            Column::new("Item", T),
            Column::new("Kappa", ColumnKind::Fixed(4)),
            Column::new("SE Kappa (H0)", ColumnKind::Fixed(4)),
            Column::new("Z", ColumnKind::Fixed(4)),
            Column::new("P(vs > 0)", P),
            Column::new("CI lower", ColumnKind::Fixed(4)),
            Column::new("CI upper", ColumnKind::Fixed(4)),
        ],
    )
}

impl Reportable for AttributeAgreement {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Attribute Agreement Analysis");
        sec.text(format!(
            "{} appraisers, {} samples, {} trials; responses: {}.",
            self.appraisers.len(),
            self.samples,
            self.trials,
            self.categories.join(", ")
        ));
        let rows: Vec<&Agreement> = self.within.iter().map(|w| &w.agreement).collect();
        sec.table(agreement_table("Within Appraisers: Assessment Agreement", &rows, self.level));
        for (name, w) in self.appraisers.iter().zip(&self.within) {
            if let Some(k) = &w.kappa {
                sec.table(kappa_table(&format!("Fleiss' Kappa within {name}"), k));
            }
        }
        if let Some(vs) = &self.vs_standard {
            let rows: Vec<&Agreement> = vs.iter().map(|w| &w.agreement).collect();
            sec.table(agreement_table("Each Appraiser vs Standard: Assessment Agreement", &rows, self.level));
            let mut t = cohen_table("Cohen's Kappa: Each Appraiser vs Standard");
            for (name, v) in self.appraisers.iter().zip(vs) {
                cohen_row(&mut t, name, &v.kappa);
            }
            sec.table(t);
            let dis: Vec<_> = vs.iter().filter_map(|v| v.disagreement.as_ref()).collect();
            if !dis.is_empty() {
                let mut d = Table::new(
                    "Assessment Disagreement",
                    vec![
                        Column::new("Appraiser", T),
                        Column::new("# Missed", I),
                        Column::new("Miss rate %", ColumnKind::Fixed(2)),
                        Column::new("# False alarms", I),
                        Column::new("False alarm rate %", ColumnKind::Fixed(2)),
                        Column::new("# Mixed", I),
                    ],
                );
                for x in dis {
                    d.row(vec![
                        Cell::text(&x.appraiser),
                        int(x.misses),
                        Cell::opt(x.miss_rate),
                        int(x.false_alarms),
                        Cell::opt(x.false_alarm_rate),
                        int(x.mixed),
                    ]);
                }
                sec.table(d);
            }
        }
        sec.table(agreement_table("Between Appraisers: Assessment Agreement", &[&self.between.agreement], self.level));
        sec.table(kappa_table("Fleiss' Kappa between Appraisers", &self.between.kappa));
        if let Some((a, k)) = &self.all_vs_standard {
            sec.table(agreement_table("All Appraisers vs Standard: Assessment Agreement", &[a], self.level));
            let mut t = cohen_table("Cohen's Kappa: All Appraisers vs Standard");
            cohen_row(&mut t, "Overall", k);
            sec.table(t);
        }
        if !self.pairs.is_empty() {
            let mut t = cohen_table("Cohen's Kappa: Appraiser vs Appraiser");
            for p in &self.pairs {
                cohen_row(&mut t, &format!("{} vs {}", p.a, p.b), &p.kappa);
            }
            sec.table(t);
        }
        for c in self.charts() {
            sec.chart(c);
        }
        sec
    }
}
