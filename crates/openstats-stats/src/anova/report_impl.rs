//! [`Reportable`] implementations for the ANOVA results.

use super::comparisons::{Alternative, Comparisons, McbResult, Method};
use super::glm::GlmAnova;
use super::oneway::OneWayAnova;
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn pct(level: f64) -> String {
    format!("{}%", (level * 1e6).round() / 1e4)
}

impl Reportable for OneWayAnova {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("One-way ANOVA");
        let alpha = 1.0 - self.level;
        sec.text(format!(
            "Null hypothesis: all means are equal. Alternative hypothesis: not all means are equal. Significance level α = {alpha}. Equal variances were assumed for this analysis."
        ));
        let mut a = Table::new(
            "Analysis of Variance",
            vec![
                Column::new("Source", T),
                Column::new("DF", I),
                Column::new("Adj SS", S),
                Column::new("Adj MS", S),
                Column::new("F-Value", S),
                Column::new("P-Value", P),
            ],
        );
        a.row(vec![
            Cell::text("Factor"),
            int(self.df_factor),
            Cell::Num(self.ss_factor),
            Cell::Num(self.ms_factor),
            Cell::opt(self.f),
            Cell::opt(self.p_value),
        ]);
        a.row(vec![Cell::text("Error"), int(self.df_error), Cell::Num(self.ss_error), Cell::Num(self.ms_error)]);
        a.row(vec![Cell::text("Total"), int(self.df_factor + self.df_error), Cell::Num(self.ss_total)]);
        sec.table(a);
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
            Cell::opt(self.r2_pred.map(|v| v * 100.0)),
        ]);
        sec.table(ms);
        let ci = format!("{} CI", pct(self.level));
        let mut m = Table::new(
            "Means",
            vec![
                Column::new("Factor", T),
                Column::new("N", I),
                Column::new("Mean", S),
                Column::new("StDev", S),
                Column::new(format!("{ci} (pooled)"), T),
            ],
        );
        for l in &self.levels {
            m.row(vec![
                Cell::text(&l.label),
                int(l.n),
                Cell::Num(l.mean),
                Cell::opt(l.sd),
                Cell::text(format!("({:.4}, {:.4})", l.ci_pooled.lower, l.ci_pooled.upper)),
            ]);
        }
        m.footnotes.push(format!("Pooled StDev = {:.6}", self.s));
        sec.table(m);
        if let Some(w) = &self.welch {
            let mut t = Table::new(
                "Welch's Test (equal variances not assumed)",
                vec![
                    Column::new("DF Num", ColumnKind::Fixed(0)),
                    Column::new("DF Den", ColumnKind::Fixed(2)),
                    Column::new("F-Value", S),
                    Column::new("P-Value", P),
                ],
            );
            t.row(vec![Cell::Num(w.df1), Cell::Num(w.df2), Cell::Num(w.f), Cell::Num(w.p_value)]);
            sec.table(t);
        } else {
            sec.warn("Welch's test needs at least two observations and a positive standard deviation in every level.");
        }
        sec.chart(self.interval_plot());
        sec.chart(self.individual_value_plot());
        sec
    }
}

impl Reportable for Comparisons {
    fn report_section(&self) -> Section {
        let name = match self.method {
            Method::Tukey => "Tukey",
            Method::Fisher => "Fisher",
            Method::Dunnett => "Dunnett",
        };
        let conf = pct(self.confidence);
        let mut sec = Section::new(match self.method {
            Method::Tukey => "Tukey Pairwise Comparisons",
            Method::Fisher => "Fisher Pairwise Comparisons",
            Method::Dunnett => "Dunnett Multiple Comparisons with a Control",
        });
        let control = self.control.map(|c| self.means.labels[c].clone());
        sec.text(match &control {
            Some(c) => format!("Control level: {c}. Dunnett simultaneous {conf} confidence intervals."),
            None => format!("Grouping information using the {name} method and {conf} confidence. Means that do not share a letter are significantly different."),
        });
        let mut g = Table::new(
            "Grouping Information",
            vec![
                Column::new("Factor", T),
                Column::new("N", ColumnKind::Fixed(0)),
                Column::new("Mean", S),
                Column::new("Grouping", T),
            ],
        );
        for r in &self.grouping {
            g.row(vec![Cell::text(&r.label), Cell::Num(r.n), Cell::Num(r.mean), Cell::text(&r.letters)]);
        }
        sec.table(g);
        let mut cols = vec![
            Column::new("Difference of Levels", T),
            Column::new("Difference of Means", S),
            Column::new("SE of Difference", S),
        ];
        match self.alternative {
            Alternative::TwoSided => {
                cols.push(Column::new(format!("{conf} CI lower"), S));
                cols.push(Column::new(format!("{conf} CI upper"), S));
            }
            Alternative::Greater => cols.push(Column::new(format!("{conf} lower bound"), S)),
            Alternative::Less => cols.push(Column::new(format!("{conf} upper bound"), S)),
        }
        cols.push(Column::new("T-Value", S));
        cols.push(Column::new("Adjusted P-Value", P));
        let title = match self.method {
            Method::Tukey => "Tukey Simultaneous Tests for Differences of Means",
            Method::Fisher => "Fisher Individual Tests for Differences of Means",
            Method::Dunnett => "Dunnett Simultaneous Tests for Level Mean − Control Mean",
        };
        let mut d = Table::new(title, cols);
        for x in &self.differences {
            let mut row = vec![
                Cell::text(format!("{} − {}", self.means.labels[x.level], self.means.labels[x.reference])),
                Cell::Num(x.diff),
                Cell::Num(x.se),
            ];
            match self.alternative {
                Alternative::TwoSided => {
                    row.push(Cell::Num(x.interval.lower));
                    row.push(Cell::Num(x.interval.upper));
                }
                Alternative::Greater => row.push(Cell::Num(x.interval.lower)),
                Alternative::Less => row.push(Cell::Num(x.interval.upper)),
            }
            row.push(Cell::Num(x.t));
            row.push(Cell::Num(x.p_adjusted));
            d.row(row);
        }
        d.footnotes.push(match self.method {
            Method::Fisher => format!("Individual confidence level = {}", pct(self.individual_confidence)),
            _ => format!(
                "Individual confidence level = {} (critical value {:.5})",
                pct(self.individual_confidence),
                self.critical
            ),
        });
        sec.table(d);
        sec
    }
}

impl Reportable for McbResult {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Hsu's Multiple Comparisons with the Best (MCB)");
        let which = match self.best {
            super::comparisons::Best::Largest => "largest",
            super::comparisons::Best::Smallest => "smallest",
        };
        sec.text(format!(
            "The {which} mean is best; simultaneous {} confidence intervals for the mean minus the best of the other levels. Levels whose upper limit is 0 are significantly worse than the best; a level whose lower limit is above 0 is significantly the best.",
            pct(self.confidence)
        ));
        let mut t = Table::new(
            "Hsu MCB Intervals",
            vec![
                Column::new("Level", T),
                Column::new("Lower", S),
                Column::new("Center", S),
                Column::new("Upper", S),
                Column::new("Status", T),
            ],
        );
        for r in &self.rows {
            let status = if r.significantly_best {
                "significantly best"
            } else if r.possibly_best {
                "possibly best"
            } else {
                "not best"
            };
            t.row(vec![
                Cell::text(&r.label),
                Cell::Num(r.lower),
                Cell::Num(r.center),
                Cell::Num(r.upper),
                Cell::text(status),
            ]);
        }
        sec.table(t);
        sec
    }
}

impl Reportable for GlmAnova {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(format!("General Linear Model: {}", self.formula));
        sec.text("Factors are coded with sum-to-zero contrasts (−1, 0, +1); sums of squares are adjusted (Type III) with the sequential (Type I) table below.");
        let fit = &self.fit;
        let mut a = Table::new(
            "Analysis of Variance (adjusted sums of squares, Type III)",
            vec![
                Column::new("Source", T),
                Column::new("DF", I),
                Column::new("Adj SS", S),
                Column::new("Adj MS", S),
                Column::new("F-Value", S),
                Column::new("P-Value", P),
            ],
        );
        for r in &fit.anova {
            let name = if r.source == "Regression" { "Model".to_string() } else { r.source.clone() };
            a.row(vec![
                Cell::text(name),
                int(r.df),
                Cell::Num(r.adj_ss),
                Cell::opt(r.adj_ms),
                Cell::opt(r.f),
                Cell::opt(r.p_value),
            ]);
        }
        sec.table(a);
        let mut q = Table::new(
            "Sequential Analysis of Variance (Type I)",
            vec![
                Column::new("Source", T),
                Column::new("DF", I),
                Column::new("Seq SS", S),
                Column::new("Seq MS", S),
                Column::new("F-Value", S),
                Column::new("P-Value", P),
            ],
        );
        for r in &self.sequential {
            q.row(vec![
                Cell::text(&r.source),
                int(r.df),
                Cell::Num(r.seq_ss),
                Cell::Num(r.seq_ms),
                Cell::opt(r.f),
                Cell::opt(r.p_value),
            ]);
        }
        sec.table(q);
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
            Cell::Num(fit.s),
            Cell::Num(fit.r2 * 100.0),
            Cell::Num(fit.r2_adj * 100.0),
            Cell::Num(fit.r2_pred * 100.0),
        ]);
        sec.table(ms);
        let mut co = Table::new(
            "Coefficients",
            vec![
                Column::new("Term", T),
                Column::new("Coef", S),
                Column::new("SE Coef", S),
                Column::new("T-Value", S),
                Column::new("P-Value", P),
                Column::new("VIF", S),
            ],
        );
        for c in &fit.coefficients {
            co.row(vec![
                Cell::text(&c.name),
                Cell::Num(c.estimate),
                Cell::Num(c.std_err),
                Cell::Num(c.t),
                Cell::Num(c.p_value),
                Cell::opt(c.vif),
            ]);
        }
        sec.table(co);
        for t in &self.ls_means {
            let mut cols: Vec<Column> = t.factors.iter().map(|f| Column::new(f.clone(), T)).collect();
            cols.push(Column::new("Mean", S));
            cols.push(Column::new("SE Mean", S));
            cols.push(Column::new(format!("{} CI lower", pct(fit.level)), S));
            cols.push(Column::new(format!("{} CI upper", pct(fit.level)), S));
            let mut tb = Table::new(format!("Least Squares Means for {}", t.term), cols);
            for r in &t.rows {
                let mut row: Vec<Cell> = r.levels.iter().map(Cell::text).collect();
                row.extend([Cell::Num(r.mean), Cell::Num(r.se), Cell::Num(r.ci.lower), Cell::Num(r.ci.upper)]);
                tb.row(row);
            }
            sec.table(tb);
        }
        for f in self.factors() {
            if let Ok(c) = self.main_effects_chart(&f) {
                sec.chart(c);
            }
        }
        for t in &self.ls_means {
            if t.factors.len() == 2 {
                if let Ok(c) = self.interaction_chart(&t.factors[0], &t.factors[1]) {
                    sec.chart(c);
                }
            }
        }
        sec
    }
}

impl Reportable for super::random::RandomAnova {
    fn report_section(&self) -> Section {
        let mut sec = Section::new("Analysis of Variance (random factors)");
        sec.text(format!(
            "Model: {}. Sums of squares are adjusted (Type III); {} model form; {} design.",
            self.formula,
            if self.restricted { "restricted" } else { "unrestricted" },
            if self.balanced { "balanced" } else { "unbalanced" }
        ));
        let mut a = Table::new(
            "Analysis of Variance",
            vec![
                Column::new("Source", T),
                Column::new("DF", I),
                Column::new("Adj SS", S),
                Column::new("Adj MS", S),
                Column::new("F-Value", S),
                Column::new("P-Value", P),
                Column::new("Error term", T),
            ],
        );
        for (i, r) in self.rows.iter().enumerate() {
            let label = if r.random { format!("{} (random)", r.source) } else { r.source.clone() };
            a.row(vec![
                Cell::text(label),
                int(r.df),
                Cell::Num(r.ss),
                Cell::Num(r.ms),
                Cell::opt(r.f),
                Cell::opt(r.p_value),
                Cell::text(self.denominator_label(i)),
            ]);
        }
        a.row(vec![Cell::text("Error"), int(self.df_error), Cell::Num(self.ss_error), Cell::Num(self.ms_error)]);
        a.row(vec![Cell::text("Total"), int(self.df_total), Cell::Num(self.ss_total)]);
        sec.table(a);
        if self.rows.iter().any(|r| r.synthesized) {
            sec.text("Where the error term is a combination of mean squares, the denominator degrees of freedom are Satterthwaite's approximation. * means no test can be formed.");
        }
        let mut e = Table::new(
            "Expected Mean Squares",
            vec![Column::new("Source", T), Column::new("Expected Mean Square for Each Term", T)],
        );
        for (i, r) in self.rows.iter().enumerate() {
            e.row(vec![Cell::text(format!("({}) {}", i + 1, r.source)), Cell::text(self.ems_text(i))]);
        }
        e.row(vec![
            Cell::text(format!("({}) Error", self.rows.len() + 1)),
            Cell::text(format!("({})", self.rows.len() + 1)),
        ]);
        sec.table(e);
        let mut v = Table::new(
            "Variance Components (ANOVA method)",
            vec![
                Column::new("Source", T),
                Column::new("Variance", S),
                Column::new("% of Total", ColumnKind::Fixed(2)),
                Column::new("StDev", S),
            ],
        );
        for c in &self.components {
            v.row(vec![Cell::text(&c.source), Cell::Num(c.estimate), Cell::opt(Some(c.percent)), Cell::Num(c.std_dev)]);
        }
        sec.table(v);
        if self.components.iter().any(|c| c.estimate < 0.0) {
            sec.text("Negative variance estimates are set to zero for the percentages and standard deviations.");
        }
        for m in &self.means {
            let mut t = Table::new(
                format!("Means: {}", m.term),
                vec![Column::new("Level", T), Column::new("N", I), Column::new("Mean", S)],
            );
            for (lv, n, mean) in &m.rows {
                t.row(vec![Cell::text(lv.join(" ")), int(*n), Cell::Num(*mean)]);
            }
            sec.table(t);
        }
        sec
    }
}
