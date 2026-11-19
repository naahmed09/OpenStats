//! [`Reportable`] implementations for regression results.

use super::{BestSubsets, RegressionFit, ResidualKind, StepAction, StepwiseResult, UnusualFlag};
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

impl Reportable for RegressionFit {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Regression Analysis");
        let mut anova = Table::new(
            "Analysis of Variance",
            vec![
                Column::new("Source", T),
                Column::new("DF", I),
                Column::new("Seq SS", S),
                Column::new("Adj SS", S),
                Column::new("Adj MS", S),
                Column::new("F-Value", S),
                Column::new("P-Value", P),
            ],
        );
        for r in &self.anova {
            anova.row(vec![
                Cell::text(&r.source),
                int(r.df),
                Cell::opt(r.seq_ss),
                Cell::Num(r.adj_ss),
                Cell::opt(r.adj_ms),
                Cell::opt(r.f),
                Cell::opt(r.p_value),
            ]);
        }
        sec.table(anova);
        let mut ms = Table::new(
            "Model Summary",
            vec![
                Column::new("S", S),
                Column::new("R-sq", ColumnKind::Fixed(2)),
                Column::new("R-sq(adj)", ColumnKind::Fixed(2)),
                Column::new("R-sq(pred)", ColumnKind::Fixed(2)),
            ],
        );
        ms.row(vec![
            Cell::Num(self.s),
            Cell::Num(self.r2 * 100.0),
            Cell::Num(self.r2_adj * 100.0),
            Cell::Num(self.r2_pred * 100.0),
        ]);
        sec.table(ms);
        let ci = format!("{}% CI", self.level * 100.0);
        let mut co = Table::new(
            "Coefficients",
            vec![
                Column::new("Term", T),
                Column::new("Coef", S),
                Column::new("SE Coef", S),
                Column::new(ci, T),
                Column::new("T-Value", S),
                Column::new("P-Value", P),
                Column::new("VIF", S),
            ],
        );
        for c in &self.coefficients {
            co.row(vec![
                Cell::text(&c.name),
                Cell::Num(c.estimate),
                Cell::Num(c.std_err),
                Cell::text(format!("({:.6}, {:.6})", c.ci.lower, c.ci.upper)),
                Cell::Num(c.t),
                Cell::Num(c.p_value),
                Cell::opt(c.vif),
            ]);
        }
        sec.table(co);
        if !self.aliased.is_empty() {
            sec.warn(format!("The following terms cannot be estimated and were removed: {}.", self.aliased.join(", ")));
        }
        sec.text(format!("Durbin-Watson statistic = {:.6}", self.diagnostics.durbin_watson));
        if let Some(l) = &self.lack_of_fit {
            sec.text(format!("Lack-of-fit test: F = {:.4}, P = {}", l.f, crate::report::format_p_value(l.p_value)));
        }
        if let Ok(unusual) = self.unusual_observations() {
            if !unusual.is_empty() {
                let mut t = Table::new(
                    "Fits and Diagnostics for Unusual Observations",
                    vec![
                        Column::new("Obs", I),
                        Column::new("Fit", S),
                        Column::new("Resid", S),
                        Column::new("Std Resid", S),
                        Column::new("Flag", T),
                    ],
                );
                for u in &unusual {
                    let flag: String = u
                        .flags
                        .iter()
                        .map(|f| match f {
                            UnusualFlag::LargeResidual => "R",
                            UnusualFlag::HighLeverage => "X",
                            UnusualFlag::Influential => "D",
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    let d = &self.diagnostics;
                    t.row(vec![
                        int(u.index + 1),
                        Cell::Num(d.fitted[u.index]),
                        Cell::Num(d.residuals[u.index]),
                        Cell::Num(d.standardized[u.index]),
                        Cell::text(flag),
                    ]);
                }
                sec.table(t);
            }
        }
        if let Ok(f) = self.diagnostics.four_in_one(ResidualKind::Standardized) {
            for c in f.charts() {
                sec.chart(c);
            }
        }
        sec
    }
}

impl Reportable for StepwiseResult {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Stepwise Regression");
        let mut t = Table::new(
            "Steps",
            vec![
                Column::new("Step", I),
                Column::new("Action", T),
                Column::new("Term", T),
                Column::new("P / IC", S),
                Column::new("Terms", I),
                Column::new("S", S),
                Column::new("R-sq", ColumnKind::Fixed(2)),
                Column::new("R-sq(adj)", ColumnKind::Fixed(2)),
            ],
        );
        for s in &self.steps {
            t.row(vec![
                int(s.number),
                Cell::text(if s.action == StepAction::Added { "Added" } else { "Removed" }),
                Cell::text(&s.term),
                Cell::Num(s.value),
                int(s.n_terms),
                Cell::Num(s.s),
                Cell::Num(s.r2 * 100.0),
                Cell::Num(s.r2_adj * 100.0),
            ]);
        }
        sec.table(t);
        sec.text(format!(
            "Selected terms: {}",
            if self.selected.is_empty() { "(none)".to_string() } else { self.selected.join(", ") }
        ));
        let inner = self.fit.report_section();
        sec.blocks.extend(inner.blocks);
        sec.warnings.extend(inner.warnings);
        sec
    }
}

impl Reportable for BestSubsets {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Best Subsets Regression");
        let mut t = Table::new(
            "Models",
            vec![
                Column::new("Vars", I),
                Column::new("R-Sq", ColumnKind::Fixed(1)),
                Column::new("R-Sq(adj)", ColumnKind::Fixed(1)),
                Column::new("R-Sq(pred)", ColumnKind::Fixed(1)),
                Column::new("Mallows Cp", ColumnKind::Fixed(1)),
                Column::new("S", S),
                Column::new("Predictors", T),
            ],
        );
        for m in &self.models {
            t.row(vec![
                int(m.size),
                Cell::Num(m.r2 * 100.0),
                Cell::Num(m.r2_adj * 100.0),
                Cell::Num(m.r2_pred * 100.0),
                Cell::Num(m.cp),
                Cell::Num(m.s),
                Cell::text(m.names.join(" ")),
            ]);
        }
        sec.table(t);
        if !self.excluded.is_empty() {
            sec.warn(format!("Excluded as linearly dependent: {}.", self.excluded.join(", ")));
        }
        sec
    }
}
