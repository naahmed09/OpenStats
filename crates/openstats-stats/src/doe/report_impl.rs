//! [`Reportable`] implementations for the design-of-experiments results.

use super::analysis::FactorialAnalysis;
use super::design::Design;
use super::factorial::AliasStructure;
use super::optimize::OptimizationResult;
use super::plots::{interaction_charts, main_effects_charts, normal_effects_chart, pareto_chart};
use super::rsm::{StationaryKind, SurfaceAnalysis};
use super::taguchi::{ResponseTable, TaguchiAnalysis, TaguchiAnovaRow};
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn table(title: &str, cols: &[(&str, ColumnKind)], rows: Vec<Vec<Cell>>) -> Table {
    let mut t = Table::new(title, cols.iter().map(|(h, k)| Column::new(*h, *k)).collect());
    for r in rows {
        t.row(r);
    }
    t
}

fn text(s: &str) -> Cell {
    Cell::Text(s.to_string())
}

fn int(n: usize) -> Cell {
    Cell::Int(n as i64)
}

impl Reportable for Design {
    fn report_section(&self) -> Section {
        let mut s = Section::new(format!("{} Design", self.kind.name()));
        s.text(format!("Factors: {}   Runs: {}   Blocks: {}", self.n_factors(), self.n_runs(), self.blocks));
        if let Some(d) = &self.defining {
            s.text(format!("Resolution: {}", roman(d.resolution)));
        }
        let mut cols: Vec<(&str, ColumnKind)> = vec![("StdOrder", I), ("PtType", I), ("Blocks", I)];
        let names: Vec<String> = self.factors.iter().map(|f| f.name.clone()).collect();
        for n in &names {
            cols.push((n.as_str(), S));
        }
        let nat = self.natural();
        let rows = (0..self.n_runs())
            .map(|r| {
                let mut v = vec![int(self.std_order[r]), Cell::Int(self.point_type[r].code()), int(self.block[r])];
                v.extend(nat[r].iter().map(|&x| Cell::Num(x)));
                v
            })
            .collect();
        s.table(table("Design table (natural units)", &cols, rows));
        s
    }
}

fn roman(r: usize) -> String {
    match r {
        1 => "I",
        2 => "II",
        3 => "III",
        4 => "IV",
        5 => "V",
        6 => "VI",
        7 => "VII",
        8 => "VIII",
        _ => return r.to_string(),
    }
    .to_string()
}

impl Reportable for AliasStructure {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Alias Structure");
        let rows = self.letters.iter().map(|(c, n)| vec![text(&c.to_string()), text(n)]).collect();
        s.table(table("Factors", &[("Letter", T), ("Name", T)], rows));
        if self.defining_relation.is_empty() {
            s.text("Full factorial: no confounding.");
        } else {
            let rel = self.defining_relation.iter().map(|w| format!("I = {w}")).collect::<Vec<_>>().join(",  ");
            s.text(format!("Defining relation: {rel}"));
            if let Some(r) = self.resolution {
                s.text(format!("Resolution: {}", roman(r)));
            }
        }
        let rows = self.classes.iter().map(|c| vec![text(&c.join(" + "))]).collect();
        s.table(table("Aliased terms", &[("Alias", T)], rows));
        s
    }
}

impl Reportable for FactorialAnalysis {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Factorial Fit");
        let mut all = vec![&self.constant];
        all.extend(self.blocks.iter());
        all.extend(self.effects.iter());
        if let Some(c) = &self.curvature {
            all.push(c);
        }
        let rows = all
            .iter()
            .map(|e| {
                vec![
                    text(&e.term),
                    Cell::opt(e.effect),
                    Cell::Num(e.coef),
                    Cell::opt(e.se_coef),
                    Cell::opt(e.t),
                    Cell::opt(e.p),
                    text(&e.aliases.join(" + ")),
                ]
            })
            .collect();
        s.table(table(
            "Estimated Effects and Coefficients",
            &[("Term", T), ("Effect", S), ("Coef", S), ("SE Coef", S), ("T", S), ("P", P), ("Aliases", T)],
            rows,
        ));
        if !self.dropped.is_empty() {
            s.warn(format!("Terms confounded with earlier terms were not estimated: {}", self.dropped.join(", ")));
        }
        if let Some(sd) = self.s {
            s.text(format!(
                "S = {sd:.6}   R-Sq = {:.2}%   R-Sq(adj) = {}   R-Sq(pred) = {}",
                100.0 * self.r2,
                self.r2_adj.map_or("*".to_string(), |v| format!("{:.2}%", 100.0 * v)),
                self.r2_pred.map_or("*".to_string(), |v| format!("{:.2}%", 100.0 * v)),
            ));
        } else {
            s.warn("There are no degrees of freedom for error; p-values cannot be computed. Use Lenth's method.");
        }
        let rows = self
            .anova
            .iter()
            .map(|a| {
                vec![
                    text(&a.source),
                    int(a.df),
                    Cell::opt(a.seq_ss),
                    Cell::Num(a.adj_ss),
                    Cell::opt(a.adj_ms),
                    Cell::opt(a.f),
                    Cell::opt(a.p),
                ]
            })
            .collect();
        s.table(table(
            "Analysis of Variance",
            &[("Source", T), ("DF", I), ("Seq SS", S), ("Adj SS", S), ("Adj MS", S), ("F", S), ("P", P)],
            rows,
        ));
        if let Some(l) = &self.lenth {
            s.text(format!(
                "Lenth's PSE = {:.6}   DF = {:.4}   ME = {:.6}   SME = {:.6}",
                l.pse, l.df, l.margin, l.simultaneous_margin
            ));
            let rows = l.t_pse.iter().map(|(n, v)| vec![text(n), Cell::Num(*v)]).collect();
            s.table(table("Effects relative to Lenth's PSE", &[("Term", T), ("Effect/PSE", S)], rows));
        }
        if let Some(a) = &self.alias {
            for b in a.report_section().blocks {
                s.blocks.push(b);
            }
        }
        for c in main_effects_charts(self) {
            s.chart(c);
        }
        for c in interaction_charts(self) {
            s.chart(c);
        }
        if let Ok(c) = pareto_chart(self) {
            s.chart(c);
        }
        if !self.effects.is_empty() {
            s.chart(normal_effects_chart(self));
        }
        s
    }
}

impl Reportable for SurfaceAnalysis {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Response Surface Regression");
        let rows = self
            .fit
            .coefficients
            .iter()
            .map(|c| {
                vec![text(&c.name), Cell::Num(c.estimate), Cell::Num(c.std_err), Cell::Num(c.t), Cell::Num(c.p_value)]
            })
            .collect();
        s.table(table("Coded Coefficients", &[("Term", T), ("Coef", S), ("SE Coef", S), ("T", S), ("P", P)], rows));
        if !self.dropped.is_empty() {
            s.warn(format!("Terms confounded with earlier terms were not estimated: {}", self.dropped.join(", ")));
        }
        s.text(format!(
            "S = {:.6}   R-Sq = {:.2}%   R-Sq(adj) = {:.2}%   R-Sq(pred) = {}",
            self.fit.s,
            100.0 * self.fit.r2,
            100.0 * self.fit.r2_adj,
            if self.fit.r2_pred.is_nan() { "*".to_string() } else { format!("{:.2}%", 100.0 * self.fit.r2_pred) }
        ));
        let rows = self
            .anova
            .iter()
            .map(|a| {
                vec![
                    text(&a.source),
                    int(a.df),
                    Cell::opt(a.seq_ss),
                    Cell::Num(a.adj_ss),
                    Cell::opt(a.adj_ms),
                    Cell::opt(a.f),
                    Cell::opt(a.p),
                ]
            })
            .collect();
        s.table(table(
            "Analysis of Variance",
            &[("Source", T), ("DF", I), ("Seq SS", S), ("Adj SS", S), ("Adj MS", S), ("F", S), ("P", P)],
            rows,
        ));
        if let Some(c) = &self.canonical {
            let kind = match c.kind {
                StationaryKind::Maximum => "maximum",
                StationaryKind::Minimum => "minimum",
                StationaryKind::Saddle => "saddle point",
                StationaryKind::Ridge => "ridge (near-singular)",
            };
            let names = &self.model.factor_names;
            let rows = (0..names.len())
                .map(|i| vec![text(&names[i]), Cell::Num(c.point_coded[i]), Cell::Num(c.point_natural[i])])
                .collect();
            s.table(table(
                &format!("Stationary point ({kind}); predicted response {:.6}", c.response),
                &[("Factor", T), ("Coded", S), ("Natural", S)],
                rows,
            ));
            let rows = c.eigenvalues.iter().map(|&v| vec![Cell::Num(v)]).collect();
            s.table(table("Eigenvalues of the quadratic form", &[("Eigenvalue", S)], rows));
        }
        s
    }
}

fn response_rows(t: &[ResponseTable]) -> Vec<Vec<Cell>> {
    let l = t.iter().map(|r| r.level_means.len()).max().unwrap_or(0);
    t.iter()
        .map(|r| {
            let mut v = vec![text(&r.factor)];
            for i in 0..l {
                v.push(r.level_means.get(i).map_or(Cell::Missing, |&m| Cell::Num(m)));
            }
            v.push(Cell::Num(r.delta));
            v.push(int(r.rank));
            v
        })
        .collect()
}

fn anova_rows(a: &[TaguchiAnovaRow]) -> Vec<Vec<Cell>> {
    a.iter()
        .map(|r| {
            vec![
                text(&r.source),
                int(r.df),
                Cell::Num(r.ss),
                Cell::opt(r.ms),
                Cell::opt(r.f),
                Cell::opt(r.p),
                Cell::Num(r.percent),
            ]
        })
        .collect()
}

impl Reportable for TaguchiAnalysis {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Taguchi Analysis");
        s.text(format!("Signal-to-noise ratio: {}   Array: {}", self.sn_type.name(), self.design.array.name));
        let l = self.design.n_levels.iter().copied().max().unwrap_or(0);
        let level_names: Vec<String> = (1..=l).map(|i| format!("Level {i}")).collect();
        let mut cols: Vec<(&str, ColumnKind)> = vec![("Factor", T)];
        for n in &level_names {
            cols.push((n.as_str(), S));
        }
        cols.push(("Delta", S));
        cols.push(("Rank", I));
        s.table(table("Response Table for Signal to Noise Ratios", &cols, response_rows(&self.sn_table)));
        s.table(table("Response Table for Means", &cols, response_rows(&self.mean_table)));
        let acols =
            [("Source", T), ("DF", I), ("SS", S), ("MS", S), ("F", S), ("P", P), ("% of SS", ColumnKind::Fixed(2))];
        s.table(table("Analysis of Variance for SN Ratios", &acols, anova_rows(&self.sn_anova)));
        s.table(table("Analysis of Variance for Means", &acols, anova_rows(&self.mean_anova)));
        let rows = (0..self.design.factors.len())
            .map(|f| vec![text(&self.design.factors[f]), int(self.best_levels[f])])
            .collect();
        s.table(table("Levels with the largest S/N ratio", &[("Factor", T), ("Level", I)], rows));
        s.text(format!(
            "Predicted S/N ratio at those levels: {:.6}   Predicted mean: {:.6}",
            self.predicted_sn, self.predicted_mean
        ));
        let rows = (0..self.run_sn.len())
            .map(|r| {
                vec![int(r + 1), Cell::Num(self.run_mean[r]), Cell::opt(self.run_sd[r]), Cell::Num(self.run_sn[r])]
            })
            .collect();
        s.table(table("Run statistics", &[("Run", I), ("Mean", S), ("StDev", S), ("S/N", S)], rows));
        for c in self.charts() {
            s.chart(c);
        }
        s
    }
}

impl Reportable for OptimizationResult {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Response Optimization");
        let rows = self
            .responses
            .iter()
            .map(|g| {
                let (kind, lo, tg, up) = match g.goal {
                    super::optimize::Goal::Maximize { lower, target, .. } => ("Maximum", Some(lower), target, None),
                    super::optimize::Goal::Minimize { target, upper, .. } => ("Minimum", None, target, Some(upper)),
                    super::optimize::Goal::Target { lower, target, upper, .. } => {
                        ("Target", Some(lower), target, Some(upper))
                    }
                };
                let weight = match g.goal {
                    super::optimize::Goal::Maximize { shape, .. } | super::optimize::Goal::Minimize { shape, .. } => {
                        format!("{shape}")
                    }
                    super::optimize::Goal::Target { shape_low, shape_high, .. } => format!("{shape_low}, {shape_high}"),
                };
                vec![
                    text(&g.name),
                    text(kind),
                    Cell::opt(lo),
                    Cell::Num(tg),
                    Cell::opt(up),
                    text(&weight),
                    Cell::Num(g.importance),
                ]
            })
            .collect();
        s.table(table(
            "Parameters",
            &[
                ("Response", T),
                ("Goal", T),
                ("Lower", S),
                ("Target", S),
                ("Upper", S),
                ("Weight", T),
                ("Importance", S),
            ],
            rows,
        ));
        for (i, sol) in self.solutions.iter().enumerate() {
            let fac_rows = (0..self.factors.len())
                .map(|f| vec![text(&self.factors[f].name), Cell::Num(sol.natural[f]), Cell::Num(sol.coded[f])])
                .collect();
            s.table(table(
                &format!("Solution {}: composite desirability = {:.6}", i + 1, sol.composite),
                &[("Factor", T), ("Setting", S), ("Coded", S)],
                fac_rows,
            ));
            let rows = self
                .responses
                .iter()
                .enumerate()
                .map(|(r, g)| vec![text(&g.name), Cell::Num(sol.fit[r]), Cell::opt(sol.se_fit[r]), Cell::Num(sol.d[r])])
                .collect();
            s.table(table(
                "Predicted responses",
                &[("Response", T), ("Fit", S), ("SE Fit", S), ("Desirability", S)],
                rows,
            ));
        }
        for c in self.charts(None, 41) {
            s.chart(c);
        }
        s
    }
}
