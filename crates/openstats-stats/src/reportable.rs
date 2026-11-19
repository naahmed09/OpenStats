//! [`Reportable`] implementations: every result type of `openstats-stats`
//! renders itself as a [`Section`] of the unified session report.
//!
//! Section layouts follow Minitab's session window: a descriptive table
//! followed by the test table, p-values in the `0.000` style, `*` for
//! missing numbers. The charts attached to a section (histogram, box plot,
//! power curve, …) are [`ChartSpec`]s that front ends draw natively.

use crate::correlation::{Kendall, PartialCorrelation, Pearson, Spearman};
use crate::crosstab::{ChiSquareGof, ChiSquareTest, Crosstab, FisherExact, McNemar};
use crate::descriptive::Descriptives;
use crate::equivalence::Tost;
use crate::graphical::{AndersonDarling, GraphicalSummary};
use crate::normality::{KsTest, RyanJoiner, ShapiroWilk};
use crate::outlier::{DixonQ, GeneralizedEsd, Grubbs};
use crate::poisson::{OnePoisson, TwoPoisson};
use crate::power::{PowerModel, PowerSolution, SolvedFor};
use crate::proportion::{OneProportion, TwoProportions};
use crate::report::{
    Axis, Cell, ChartKind, ChartSpec, Column, ColumnKind, ReferenceLine, Reportable, Section, Series, Table,
};
use crate::tolerance::{NonparametricTolerance, NormalMethod, NormalTolerance, ToleranceSides};
use crate::ttest::{TTest, ZTest};
use crate::variance::{BonettTest, FTest, LeveneTest};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

type Col<'a> = (&'a str, ColumnKind, Cell);

fn hrow(title: &str, cols: Vec<Col<'_>>) -> Table {
    let columns = cols.iter().map(|(h, k, _)| Column::new(*h, *k)).collect();
    let mut t = Table::new(title, columns);
    t.row(cols.into_iter().map(|(_, _, c)| c).collect());
    t
}

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn uint(n: u64) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn level_label(level: f64) -> String {
    format!("{}% CI", level * 100.0)
}

fn ci_cols<'a>(lo: f64, hi: f64, level: f64, lo_h: &'a str, hi_h: &'a str) -> Vec<Col<'a>> {
    let _ = level;
    vec![(lo_h, S, Cell::Num(lo)), (hi_h, S, Cell::Num(hi))]
}

impl Reportable for Descriptives {
    fn report_section(&self) -> Section {
        let mut t = Table::new("Descriptive Statistics", vec![Column::new("Statistic", T), Column::new("Value", S)]);
        let n = |x: usize| Cell::Num(x as f64);
        let rows: Vec<(&str, Cell)> = vec![
            ("N", n(self.n)),
            ("N*", n(self.n_missing)),
            ("Mean", Cell::opt(self.mean)),
            ("SE Mean", Cell::opt(self.se_mean)),
            ("TrMean", Cell::opt(self.trimmed_mean)),
            ("StDev", Cell::opt(self.std_dev)),
            ("Variance", Cell::opt(self.variance)),
            ("CoefVar", Cell::opt(self.coef_var)),
            ("Sum", Cell::opt(self.sum)),
            ("Minimum", Cell::opt(self.minimum)),
            ("Q1", Cell::opt(self.q1)),
            ("Median", Cell::opt(self.median)),
            ("Q3", Cell::opt(self.q3)),
            ("Maximum", Cell::opt(self.maximum)),
            ("Range", Cell::opt(self.range)),
            ("IQR", Cell::opt(self.iqr)),
            ("Mode", self.modes.first().map_or(Cell::Missing, |m| Cell::Num(*m))),
            ("N for Mode", n(self.mode_count)),
            ("Skewness", Cell::opt(self.skewness)),
            ("Kurtosis", Cell::opt(self.kurtosis)),
            ("MSSD", Cell::opt(self.mssd)),
        ];
        for (k, v) in rows {
            t.row(vec![Cell::text(k), v]);
        }
        if self.modes.len() > 1 {
            t.footnotes.push(format!("Multiple modes exist. The smallest is shown ({} modes).", self.modes.len()));
        }
        let mut s = Section::new("Descriptive Statistics");
        s.table(t);
        s
    }
}

impl Reportable for GraphicalSummary {
    fn report_section(&self) -> Section {
        let mut s = self.descriptives.report_section();
        s.heading = "Graphical Summary".into();
        if let Some(ad) = &self.anderson_darling {
            s.table(
                ad.report_section()
                    .blocks
                    .into_iter()
                    .find_map(|b| if let crate::report::Block::Table(t) = b { Some(t) } else { None })
                    .unwrap_or_default(),
            );
        }
        let mut ci = Table::new(
            "Confidence Intervals",
            vec![Column::new("Parameter", T), Column::new("Lower", S), Column::new("Upper", S)],
        );
        if let Some(i) = &self.ci_mean {
            ci.row(vec![
                Cell::text(format!("Mean ({})", level_label(i.level))),
                Cell::Num(i.lower),
                Cell::Num(i.upper),
            ]);
        }
        if let Some(i) = &self.ci_median {
            ci.row(vec![Cell::text("Median"), Cell::Num(i.lower), Cell::Num(i.upper)]);
            ci.footnotes.push(format!("Achieved confidence for the median interval: {:.4}.", i.achieved));
        }
        if let Some(i) = &self.ci_std_dev {
            ci.row(vec![
                Cell::text(format!("StDev ({})", level_label(i.level))),
                Cell::Num(i.lower),
                Cell::Num(i.upper),
            ]);
        }
        if !ci.rows.is_empty() {
            s.table(ci);
        }
        if let Some(h) = &self.histogram {
            s.chart(ChartSpec {
                kind: ChartKind::Histogram,
                title: "Histogram".into(),
                x_label: "Value".into(),
                y_label: "Frequency".into(),
                categories: Vec::new(),
                series: vec![Series {
                    name: "Frequency".into(),
                    x: h.edges.clone(),
                    y: h.counts.iter().map(|&c| c as f64).collect(),
                }],
                reference_lines: Vec::new(),
            });
        }
        if let Some(b) = &self.boxplot {
            s.chart(ChartSpec {
                kind: ChartKind::Boxplot,
                title: "Boxplot".into(),
                x_label: String::new(),
                y_label: "Value".into(),
                categories: Vec::new(),
                series: vec![
                    Series {
                        name: "whiskers/quartiles".into(),
                        x: vec![0.0; 5],
                        y: vec![b.lower_whisker, b.q1, b.median, b.q3, b.upper_whisker],
                    },
                    Series {
                        name: "outliers".into(),
                        x: vec![0.0; b.low_outliers.len() + b.high_outliers.len()],
                        y: b.low_outliers.iter().chain(&b.high_outliers).copied().collect(),
                    },
                ],
                reference_lines: Vec::new(),
            });
        }
        s
    }
}

impl Reportable for AndersonDarling {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Anderson-Darling Normality Test");
        s.table(hrow(
            "Anderson-Darling Normality Test",
            vec![
                ("N", I, int(self.n)),
                ("A-Squared", S, Cell::Num(self.a2)),
                ("Adjusted A²", S, Cell::Num(self.a2_adjusted)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for Crosstab {
    fn report_section(&self) -> Section {
        let mut cols = vec![Column::new("", T)];
        cols.extend(self.col_labels.iter().map(|l| Column::new(l.clone(), I)));
        cols.push(Column::new("All", I));
        let mut t = Table::new("Tabulated Statistics: Counts", cols);
        let mut col_tot = vec![0u64; self.col_labels.len()];
        for (r, row) in self.counts.iter().enumerate() {
            let mut cells = vec![Cell::text(self.row_labels.get(r).cloned().unwrap_or_default())];
            for (c, &v) in row.iter().enumerate() {
                cells.push(uint(v));
                if let Some(x) = col_tot.get_mut(c) {
                    *x += v;
                }
            }
            cells.push(uint(row.iter().sum()));
            t.row(cells);
        }
        let mut last = vec![Cell::text("All")];
        last.extend(col_tot.iter().map(|&v| uint(v)));
        last.push(uint(col_tot.iter().sum()));
        t.row(last);
        if self.n_excluded > 0 {
            t.footnotes.push(format!("{} observation(s) with missing values were excluded.", self.n_excluded));
        }
        let mut s = Section::new("Cross-Tabulation");
        s.table(t);
        s
    }
}

impl Reportable for ChiSquareTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Chi-Square Test");
        let mut t = hrow(
            "Chi-Square Test",
            vec![
                ("Chi-Square", S, Cell::Num(self.statistic)),
                ("DF", I, int(self.df)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        );
        t.footnotes.push(format!(
            "{} of {} cells have expected counts less than 5 (minimum expected {:.3}).",
            self.cells_expected_below_5, self.n_cells, self.min_expected
        ));
        s.table(t);
        if self.cells_expected_below_5 > 0 {
            s.warn("Chi-square approximation may be invalid: some expected counts are less than 5.");
        }
        s
    }
}

impl Reportable for ChiSquareGof {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Chi-Square Goodness-of-Fit Test");
        let mut t = Table::new(
            "Observed and Expected Counts",
            vec![
                Column::new("Category", T),
                Column::new("Observed", S),
                Column::new("Test Proportion", S),
                Column::new("Expected", S),
                Column::new("Contribution to Chi-Square", S),
            ],
        );
        for i in 0..self.labels.len() {
            t.row(vec![
                Cell::text(self.labels[i].clone()),
                Cell::Num(self.observed[i]),
                Cell::Num(self.proportion[i]),
                Cell::Num(self.expected[i]),
                Cell::Num(self.contribution[i]),
            ]);
        }
        s.table(t);
        s.table(hrow(
            "Chi-Square Test",
            vec![
                ("N", S, Cell::Num(self.n)),
                ("DF", I, int(self.df)),
                ("Chi-Square", S, Cell::Num(self.statistic)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        if self.cells_expected_below_5 > 0 {
            s.warn("Chi-square approximation may be invalid: some expected counts are less than 5.");
        }
        s
    }
}

impl Reportable for FisherExact {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Fisher's Exact Test");
        s.table(hrow(
            "Fisher's Exact Test",
            vec![
                ("P-Value", P, Cell::Num(self.p_value)),
                ("Left-tailed P", P, Cell::opt(self.p_left)),
                ("Right-tailed P", P, Cell::opt(self.p_right)),
                ("Odds Ratio", S, Cell::opt(self.odds_ratio)),
            ],
        ));
        s
    }
}

impl Reportable for McNemar {
    fn report_section(&self) -> Section {
        let mut s = Section::new("McNemar's Test");
        s.table(hrow(
            "McNemar's Test",
            vec![
                ("b", I, uint(self.b)),
                ("c", I, uint(self.c)),
                ("Chi-Square", S, Cell::opt(self.statistic)),
                ("P-Value", P, Cell::opt(self.p_value)),
                ("Corrected Chi-Sq", S, Cell::opt(self.statistic_corrected)),
                ("Corrected P", P, Cell::opt(self.p_value_corrected)),
                ("Exact P", P, Cell::Num(self.p_value_exact)),
            ],
        ));
        s
    }
}

impl Reportable for Pearson {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Pearson Correlation");
        let mut cols: Vec<Col<'_>> = vec![("N", I, int(self.n)), ("Pearson r", S, Cell::Num(self.r))];
        if let Some(ci) = &self.ci {
            cols.extend(ci_cols(ci.lower, ci.upper, ci.level, "CI Lower", "CI Upper"));
        }
        cols.push(("T-Value", S, Cell::opt(self.t)));
        cols.push(("P-Value", P, Cell::opt(self.p_value)));
        s.table(hrow("Pearson Correlation", cols));
        s
    }
}

impl Reportable for Spearman {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Spearman Correlation");
        s.table(hrow(
            "Spearman Rank Correlation",
            vec![
                ("N", I, int(self.n)),
                ("Spearman rho", S, Cell::Num(self.rho)),
                ("P-Value", P, Cell::opt(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for Kendall {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Kendall Correlation");
        s.table(hrow(
            "Kendall Rank Correlation",
            vec![
                ("N", I, int(self.n)),
                ("Concordant", I, uint(self.concordant)),
                ("Discordant", I, uint(self.discordant)),
                ("Tau-a", S, Cell::Num(self.tau_a)),
                ("Tau-b", S, Cell::Num(self.tau_b)),
                ("Tau-c", S, Cell::Num(self.tau_c)),
                ("Z", S, Cell::Num(self.z)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s.text(if self.exact {
            "The p-value is exact."
        } else {
            "The p-value uses the tie-corrected normal approximation."
        });
        s
    }
}

impl Reportable for PartialCorrelation {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Partial Correlation");
        let mut cols: Vec<Col<'_>> =
            vec![("N", I, int(self.n)), ("Partial r", S, Cell::Num(self.r)), ("DF", I, int(self.df))];
        if let Some(ci) = &self.ci {
            cols.extend(ci_cols(ci.lower, ci.upper, ci.level, "CI Lower", "CI Upper"));
        }
        cols.push(("T-Value", S, Cell::Num(self.t)));
        cols.push(("P-Value", P, Cell::Num(self.p_value)));
        s.table(hrow("Partial Correlation", cols));
        s
    }
}

impl Reportable for TTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new(if self.n2.is_some() { "Two-Sample T-Test" } else { "T-Test" });
        let mut d = Table::new(
            "Descriptive Statistics",
            vec![
                Column::new("Sample", T),
                Column::new("N", I),
                Column::new("Mean", S),
                Column::new("StDev", S),
                Column::new("SE Mean", S),
            ],
        );
        d.row(vec![
            Cell::text(if self.n2.is_some() { "Sample 1" } else { "Sample" }),
            int(self.n1),
            Cell::Num(self.mean1),
            Cell::Num(self.sd1),
            Cell::Num(self.sd1 / (self.n1 as f64).sqrt()),
        ]);
        if let (Some(n2), Some(m2), Some(s2)) = (self.n2, self.mean2, self.sd2) {
            d.row(vec![
                Cell::text("Sample 2"),
                int(n2),
                Cell::Num(m2),
                Cell::Num(s2),
                Cell::Num(s2 / (n2 as f64).sqrt()),
            ]);
        }
        s.table(d);
        let mut cols: Vec<Col<'_>> = vec![("Estimate", S, Cell::Num(self.estimate)), ("SE", S, Cell::Num(self.se))];
        cols.extend(ci_cols(self.ci.lower, self.ci.upper, self.ci.level, "CI Lower", "CI Upper"));
        cols.push(("Null value", S, Cell::Num(self.null_value)));
        cols.push(("T-Value", S, Cell::Num(self.t)));
        cols.push(("DF", S, Cell::Num(self.df)));
        cols.push(("P-Value", P, Cell::Num(self.p_value)));
        let mut t = hrow("Test", cols);
        t.footnotes.push(format!("{} confidence interval.", level_label(self.ci.level)));
        if let Some(sp) = self.pooled_sd {
            t.footnotes.push(format!("Both samples use a pooled standard deviation of {sp:.6}."));
        }
        s.table(t);
        s
    }
}

impl Reportable for ZTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Z-Test");
        let mut cols: Vec<Col<'_>> = vec![
            ("N", I, int(self.n)),
            ("Mean", S, Cell::Num(self.mean)),
            ("Known StDev", S, Cell::Num(self.sigma)),
            ("SE Mean", S, Cell::Num(self.se)),
        ];
        cols.extend(ci_cols(self.ci.lower, self.ci.upper, self.ci.level, "CI Lower", "CI Upper"));
        cols.push(("Z-Value", S, Cell::Num(self.z)));
        cols.push(("P-Value", P, Cell::Num(self.p_value)));
        s.table(hrow("One-Sample Z", cols));
        s
    }
}

impl Reportable for OneProportion {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test and CI for One Proportion");
        s.table(hrow(
            "Sample",
            vec![
                ("X", I, uint(self.x)),
                ("N", I, uint(self.n)),
                ("Sample p", S, Cell::Num(self.p_hat)),
                ("Test p", S, Cell::Num(self.p0)),
            ],
        ));
        let mut t = Table::new(
            "Test and Confidence Intervals",
            vec![
                Column::new("Method", T),
                Column::new("CI Lower", S),
                Column::new("CI Upper", S),
                Column::new("Z-Value", S),
                Column::new("P-Value", P),
            ],
        );
        t.row(vec![
            Cell::text("Normal approximation"),
            Cell::Num(self.ci_normal.lower),
            Cell::Num(self.ci_normal.upper),
            Cell::Num(self.z),
            Cell::Num(self.p_value_normal),
        ]);
        t.row(vec![
            Cell::text("Exact (Blaker CI)"),
            Cell::Num(self.ci_blaker.lower),
            Cell::Num(self.ci_blaker.upper),
            Cell::Missing,
            Cell::Num(self.p_value_exact),
        ]);
        t.row(vec![
            Cell::text("Exact (Clopper-Pearson)"),
            Cell::Num(self.ci_exact.lower),
            Cell::Num(self.ci_exact.upper),
            Cell::Missing,
            Cell::Num(self.p_value_exact),
        ]);
        if self.continuity_correction {
            t.footnotes.push("The normal approximation uses a continuity correction.".into());
        }
        s.table(t);
        s
    }
}

impl Reportable for TwoProportions {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test and CI for Two Proportions");
        let mut d = Table::new(
            "Sample",
            vec![Column::new("Sample", T), Column::new("X", I), Column::new("N", I), Column::new("Sample p", S)],
        );
        d.row(vec![Cell::text("1"), uint(self.x1), uint(self.n1), Cell::Num(self.p1)]);
        d.row(vec![Cell::text("2"), uint(self.x2), uint(self.n2), Cell::Num(self.p2)]);
        s.table(d);
        let mut cols: Vec<Col<'_>> =
            vec![("Difference", S, Cell::Num(self.diff)), ("Null difference", S, Cell::Num(self.diff0))];
        cols.extend(ci_cols(self.ci.lower, self.ci.upper, self.ci.level, "CI Lower", "CI Upper"));
        cols.push(("Z-Value", S, Cell::Num(self.z)));
        cols.push(("P-Value", P, Cell::Num(self.p_value_normal)));
        cols.push(("Fisher P", P, Cell::opt(self.p_value_fisher)));
        let mut t = hrow("Estimation and Test", cols);
        if self.continuity_correction {
            t.footnotes.push("The z test uses a continuity correction.".into());
        }
        s.table(t);
        s
    }
}

impl Reportable for OnePoisson {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test and CI for One-Sample Poisson Rate");
        s.table(hrow(
            "Sample",
            vec![
                ("Total occurrences", I, uint(self.total)),
                ("Exposure", S, Cell::Num(self.exposure)),
                ("Rate", S, Cell::Num(self.rate)),
                ("Test rate", S, Cell::Num(self.rate0)),
            ],
        ));
        let mut t = Table::new(
            "Test and Confidence Intervals",
            vec![
                Column::new("Method", T),
                Column::new("CI Lower", S),
                Column::new("CI Upper", S),
                Column::new("Z-Value", S),
                Column::new("P-Value", P),
            ],
        );
        t.row(vec![
            Cell::text("Normal approximation"),
            Cell::Num(self.ci_normal.lower),
            Cell::Num(self.ci_normal.upper),
            Cell::Num(self.z),
            Cell::Num(self.p_value_normal),
        ]);
        t.row(vec![
            Cell::text("Exact"),
            Cell::Num(self.ci_exact.lower),
            Cell::Num(self.ci_exact.upper),
            Cell::Missing,
            Cell::Num(self.p_value_exact),
        ]);
        s.table(t);
        s
    }
}

impl Reportable for TwoPoisson {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test and CI for Two-Sample Poisson Rates");
        let mut d = Table::new(
            "Sample",
            vec![
                Column::new("Sample", T),
                Column::new("Occurrences", I),
                Column::new("Exposure", S),
                Column::new("Rate", S),
            ],
        );
        d.row(vec![Cell::text("1"), uint(self.x1), Cell::Num(self.t1), Cell::Num(self.rate1)]);
        d.row(vec![Cell::text("2"), uint(self.x2), Cell::Num(self.t2), Cell::Num(self.rate2)]);
        s.table(d);
        let mut cols: Vec<Col<'_>> =
            vec![("Rate difference", S, Cell::Num(self.diff)), ("Null difference", S, Cell::Num(self.diff0))];
        cols.extend(ci_cols(self.ci_diff.lower, self.ci_diff.upper, self.ci_diff.level, "CI Lower", "CI Upper"));
        cols.push(("Z-Value", S, Cell::Num(self.z)));
        cols.push(("P-Value", P, Cell::Num(self.p_value_normal)));
        cols.push(("Exact P", P, Cell::opt(self.p_value_exact)));
        let mut t = hrow("Estimation and Test", cols);
        if let Some(r) = &self.ci_ratio_exact {
            t.footnotes.push(format!(
                "Exact {} for the rate ratio: ({:.6}, {:.6}).",
                level_label(r.level),
                r.lower,
                r.upper
            ));
        }
        s.table(t);
        s
    }
}

impl Reportable for FTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test for Two Variances (F-Test)");
        let mut d =
            Table::new("Sample", vec![Column::new("Sample", T), Column::new("N", I), Column::new("Variance", S)]);
        d.row(vec![Cell::text("1"), int(self.n1), Cell::Num(self.var1)]);
        d.row(vec![Cell::text("2"), int(self.n2), Cell::Num(self.var2)]);
        s.table(d);
        let mut cols: Vec<Col<'_>> =
            vec![("Ratio of variances", S, Cell::Num(self.ratio)), ("Null ratio", S, Cell::Num(self.ratio0))];
        cols.extend(ci_cols(self.ci_ratio.lower, self.ci_ratio.upper, self.ci_ratio.level, "CI Lower", "CI Upper"));
        cols.push(("F", S, Cell::Num(self.f)));
        cols.push(("DF1", S, Cell::Num(self.df1)));
        cols.push(("DF2", S, Cell::Num(self.df2)));
        cols.push(("P-Value", P, Cell::Num(self.p_value)));
        let mut t = hrow("Ratio of Variances", cols);
        t.footnotes.push(format!(
            "{} for the ratio of standard deviations: ({:.6}, {:.6}).",
            level_label(self.ci_sd_ratio.level),
            self.ci_sd_ratio.lower,
            self.ci_sd_ratio.upper
        ));
        s.table(t);
        s
    }
}

impl Reportable for LeveneTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test for Equal Variances (Levene / Brown-Forsythe)");
        let mut d = Table::new("Groups", vec![Column::new("Group", T), Column::new("N", I), Column::new("Center", S)]);
        for (i, (n, c)) in self.n.iter().zip(&self.centers).enumerate() {
            d.row(vec![Cell::text((i + 1).to_string()), int(*n), Cell::Num(*c)]);
        }
        s.table(d);
        s.table(hrow(
            "Test",
            vec![
                ("Test Statistic", S, Cell::Num(self.statistic)),
                ("DF1", S, Cell::Num(self.df1)),
                ("DF2", S, Cell::Num(self.df2)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for BonettTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Test for Equal Variances (Bonett)");
        let mut d = Table::new(
            "Groups",
            vec![
                Column::new("Group", T),
                Column::new("N", I),
                Column::new("StDev", S),
                Column::new("Kurtosis estimate", S),
                Column::new("Correction c", S),
            ],
        );
        for (i, g) in self.groups.iter().enumerate() {
            d.row(vec![
                Cell::text((i + 1).to_string()),
                int(g.n),
                Cell::Num(g.std_dev),
                Cell::Num(g.kurtosis),
                Cell::Num(g.correction),
            ]);
        }
        s.table(d);
        s.table(hrow(
            "Test",
            vec![
                ("Test Statistic", S, Cell::Num(self.statistic)),
                ("DF", S, Cell::Num(self.df)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for ShapiroWilk {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Shapiro-Wilk Normality Test");
        s.table(hrow(
            "Shapiro-Wilk Normality Test",
            vec![("N", I, int(self.n)), ("W", S, Cell::Num(self.w)), ("P-Value", P, Cell::Num(self.p_value))],
        ));
        s
    }
}

impl Reportable for RyanJoiner {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Ryan-Joiner Normality Test");
        s.table(hrow(
            "Ryan-Joiner Normality Test",
            vec![("N", I, int(self.n)), ("RJ", S, Cell::Num(self.rj)), ("P-Value", P, Cell::Num(self.p_value))],
        ));
        s
    }
}

impl Reportable for KsTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Kolmogorov-Smirnov Normality Test");
        s.table(hrow(
            "Kolmogorov-Smirnov Test",
            vec![
                ("N", I, int(self.n)),
                ("D", S, Cell::Num(self.d)),
                ("D+", S, Cell::Num(self.d_plus)),
                ("D-", S, Cell::Num(self.d_minus)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for Grubbs {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Grubbs' Test for Outliers");
        s.table(hrow(
            "Grubbs' Test",
            vec![
                ("N", I, int(self.n)),
                ("Mean", S, Cell::Num(self.mean)),
                ("StDev", S, Cell::Num(self.std_dev)),
                ("Suspect value", S, Cell::Num(self.value)),
                ("Row", I, int(self.index + 1)),
                ("G", S, Cell::Num(self.g)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for GeneralizedEsd {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Generalized ESD Test for Outliers");
        let mut t = Table::new(
            "Steps",
            vec![
                Column::new("i", I),
                Column::new("Row", I),
                Column::new("Value", S),
                Column::new("R", S),
                Column::new("Critical value", S),
            ],
        );
        for st in &self.steps {
            t.row(vec![int(st.i), int(st.index + 1), Cell::Num(st.value), Cell::Num(st.r), Cell::Num(st.lambda)]);
        }
        t.footnotes.push(format!("{} outlier(s) detected among {} observations.", self.n_outliers, self.n));
        s.table(t);
        s
    }
}

impl Reportable for DixonQ {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Dixon's Q Test for an Outlier");
        s.table(hrow(
            "Dixon's Q Test",
            vec![
                ("N", I, int(self.n)),
                ("Suspect value", S, Cell::Num(self.value)),
                ("Row", I, int(self.index + 1)),
                ("Q", S, Cell::Num(self.q)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for Tost {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Equivalence Test (TOST)");
        let mut cols: Vec<Col<'_>> = vec![
            ("Estimate", S, Cell::Num(self.estimate)),
            ("SE", S, Cell::Num(self.se)),
            ("DF", S, Cell::Num(self.df)),
        ];
        cols.extend(ci_cols(self.ci.lower, self.ci.upper, self.ci.level, "CI Lower", "CI Upper"));
        s.table(hrow("Difference", cols));
        let mut t = Table::new(
            "Null Hypotheses",
            vec![Column::new("Null hypothesis", T), Column::new("T-Value", S), Column::new("P-Value", P)],
        );
        t.row(vec![
            Cell::text(format!("Difference ≤ {}", self.lower)),
            Cell::Num(self.t_lower),
            Cell::Num(self.p_lower),
        ]);
        t.row(vec![
            Cell::text(format!("Difference ≥ {}", self.upper)),
            Cell::Num(self.t_upper),
            Cell::Num(self.p_upper),
        ]);
        t.footnotes
            .push(format!("Overall p-value (the larger of the two): {}.", crate::report::format_p_value(self.p_value)));
        s.table(t);
        s.text(if self.equivalent {
            format!(
                "Equivalence is concluded at α = {}: the {} interval lies within ({}, {}).",
                self.alpha,
                level_label(self.ci.level),
                self.lower,
                self.upper
            )
        } else {
            format!("Equivalence cannot be concluded at α = {}.", self.alpha)
        });
        s
    }
}

impl Reportable for NormalTolerance {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Normal Tolerance Interval");
        s.table(hrow(
            "Tolerance Interval",
            vec![
                ("N", I, int(self.n)),
                ("Mean", S, Cell::Num(self.mean)),
                ("StDev", S, Cell::Num(self.std_dev)),
                ("Coverage", ColumnKind::Fixed(4), Cell::Num(self.coverage)),
                ("Confidence", ColumnKind::Fixed(4), Cell::Num(self.confidence)),
                ("k", S, Cell::Num(self.k)),
                ("Lower", S, Cell::Num(self.lower)),
                ("Upper", S, Cell::Num(self.upper)),
            ],
        ));
        s.text(format!(
            "{}% of the population is covered with {}% confidence ({}).",
            self.coverage * 100.0,
            self.confidence * 100.0,
            match (self.sides, self.method) {
                (ToleranceSides::TwoSided, NormalMethod::Howe) => "two-sided, Howe approximation",
                (ToleranceSides::TwoSided, NormalMethod::Exact) => "two-sided, exact",
                (ToleranceSides::Upper, _) => "upper bound, exact",
                (ToleranceSides::Lower, _) => "lower bound, exact",
            }
        ));
        s
    }
}

impl Reportable for NonparametricTolerance {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Nonparametric Tolerance Interval");
        s.table(hrow(
            "Tolerance Interval",
            vec![
                ("N", I, int(self.n)),
                ("Coverage", ColumnKind::Fixed(4), Cell::Num(self.coverage)),
                ("Requested confidence", ColumnKind::Fixed(4), Cell::Num(self.requested_confidence)),
                ("Achieved confidence", ColumnKind::Fixed(4), Cell::Num(self.achieved_confidence)),
                ("Lower rank", I, self.lower_rank.map_or(Cell::Missing, int)),
                ("Upper rank", I, self.upper_rank.map_or(Cell::Missing, int)),
                ("Lower", S, Cell::Num(self.lower)),
                ("Upper", S, Cell::Num(self.upper)),
            ],
        ));
        s
    }
}

/// The report section of a solved power / sample-size problem: the fixed
/// parameters, the results table (effect, sample size, target and actual
/// power) and a power curve (power against the effect at the solved sample
/// size, with the target power and α as reference lines).
pub fn power_section<M: PowerModel>(model: &M, sol: &PowerSolution) -> Section {
    let mut s = Section::new(format!("Power and Sample Size: {}", model.name()));
    let mut p = Table::new("Method", vec![Column::new("Parameter", T), Column::new("Value", T)]);
    for (k, v) in model.parameters() {
        p.row(vec![Cell::text(k), Cell::text(v)]);
    }
    s.table(p);
    let mut r = Table::new(
        "Results",
        vec![
            Column::new(model.effect_label(), S),
            Column::new(model.sample_size_label(), I),
            Column::new("Target Power", ColumnKind::Fixed(4)),
            Column::new("Actual Power", ColumnKind::Fixed(4)),
        ],
    );
    r.row(vec![Cell::Num(sol.effect), int(sol.sample_size), Cell::opt(sol.target_power), Cell::Num(sol.power)]);
    if sol.solved_for == SolvedFor::SampleSize {
        r.footnotes.push("The sample size is the smallest whose power reaches the target; the actual power is the power at that sample size.".into());
    }
    s.table(r);

    // Power curve over the effect, from the null end past the solved effect.
    let (lo, _) = model.effect_range();
    let span = sol.effect - lo;
    if span != 0.0 && span.is_finite() {
        let xs: Vec<f64> = (0..=40).map(|i| lo + span * 1.6 * f64::from(i) / 40.0).collect();
        if let Ok(ys) = model.power_curve(sol.sample_size, &xs) {
            let mut lines = vec![ReferenceLine { axis: Axis::Y, value: model.alpha(), label: "alpha".into() }];
            if let Some(t) = sol.target_power {
                lines.push(ReferenceLine { axis: Axis::Y, value: t, label: "target power".into() });
            }
            lines.push(ReferenceLine { axis: Axis::X, value: sol.effect, label: model.effect_label().into() });
            s.chart(ChartSpec {
                kind: ChartKind::Line,
                title: format!("Power Curve for {}", model.name()),
                x_label: model.effect_label().into(),
                y_label: "Power".into(),
                categories: Vec::new(),
                series: vec![Series {
                    name: format!("{} = {}", model.sample_size_label(), sol.sample_size),
                    x: xs,
                    y: ys,
                }],
                reference_lines: lines,
            });
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypothesis::Alternative;
    use crate::power::{OneSampleT, TwoSampleT};
    use crate::report::{Block, SessionReport};

    fn data() -> Vec<f64> {
        vec![4.1, 5.2, 3.9, 6.3, 5.5, 4.8, 5.1, 4.4, 5.9, 5.0]
    }

    #[test]
    fn every_result_type_renders_and_round_trips() {
        let x = data();
        let y: Vec<f64> = x.iter().map(|v| v * 0.9 + 0.3).collect();
        let mut r = SessionReport::new("All procedures");
        r.push(crate::descriptive::describe(&x).unwrap().report_section());
        r.push(
            crate::graphical::graphical_summary(&x, 0.95, &crate::graphical::Binning::Sturges)
                .unwrap()
                .report_section(),
        );
        r.push(crate::ttest::one_sample_t(&x, 5.0, Alternative::TwoSided, 0.95).unwrap().report_section());
        r.push(
            crate::ttest::two_sample_t(
                &x,
                &y,
                0.0,
                crate::hypothesis::VarianceAssumption::Welch,
                Alternative::TwoSided,
                0.95,
            )
            .unwrap()
            .report_section(),
        );
        r.push(crate::ttest::z_test(&x, 5.0, 0.8, Alternative::Greater, 0.95).unwrap().report_section());
        r.push(
            crate::proportion::one_proportion(7, 20, 0.5, Alternative::TwoSided, 0.95, true).unwrap().report_section(),
        );
        r.push(
            crate::proportion::two_proportions(7, 20, 12, 25, 0.0, Alternative::TwoSided, 0.95, false)
                .unwrap()
                .report_section(),
        );
        r.push(crate::poisson::one_sample_poisson(12, 4.0, 2.0, Alternative::TwoSided, 0.95).unwrap().report_section());
        r.push(
            crate::poisson::two_sample_poisson(12, 4.0, 20, 5.0, 0.0, Alternative::TwoSided, 0.95)
                .unwrap()
                .report_section(),
        );
        r.push(crate::variance::f_test(&x, &y, 1.0, Alternative::TwoSided, 0.95).unwrap().report_section());
        let groups = vec![x.clone(), y.clone(), x.iter().map(|v| v * 1.3).collect()];
        r.push(crate::variance::levene_test(&groups, crate::variance::Center::Median).unwrap().report_section());
        r.push(crate::variance::bonett_test(&groups, 0.95).unwrap().report_section());
        r.push(crate::normality::shapiro_wilk(&x).unwrap().report_section());
        r.push(crate::normality::ryan_joiner(&x).unwrap().report_section());
        r.push(crate::normality::lilliefors(&x).unwrap().report_section());
        r.push(crate::graphical::anderson_darling(&x).unwrap().report_section());
        r.push(crate::outlier::grubbs_test(&x, crate::outlier::OutlierSide::Either).unwrap().report_section());
        r.push(crate::outlier::generalized_esd(&x, 2, 0.05).unwrap().report_section());
        r.push(crate::outlier::dixon_q_test(&x, crate::outlier::OutlierSide::Either).unwrap().report_section());
        r.push(crate::equivalence::tost_one_sample(&x, 5.0, -1.0, 1.0, 0.05).unwrap().report_section());
        r.push(crate::correlation::pearson(&x, &y, 0.95).unwrap().report_section());
        r.push(crate::correlation::spearman(&x, &y).unwrap().report_section());
        r.push(crate::correlation::kendall(&x, &y).unwrap().report_section());
        r.push(
            crate::tolerance::normal_tolerance(&x, 0.95, 0.95, ToleranceSides::TwoSided, NormalMethod::Exact)
                .unwrap()
                .report_section(),
        );
        let big: Vec<f64> = (1..=100).map(f64::from).collect();
        r.push(
            crate::tolerance::nonparametric_tolerance(&big, 0.9, 0.95, ToleranceSides::TwoSided)
                .unwrap()
                .report_section(),
        );
        r.validate().unwrap();
        assert!(r.sections.len() >= 25);
        let text = r.render_text();
        assert!(text.contains("P-Value"));
        let back = SessionReport::from_json(&r.to_json()).unwrap();
        assert_eq!(back.to_json(), r.to_json());
        assert_eq!(back.sections.len(), r.sections.len());
        // The graphical summary carries two chart specs.
        assert_eq!(r.sections[1].blocks.iter().filter(|b| matches!(b, Block::Chart(_))).count(), 2);
    }

    #[test]
    fn t_test_section_layout() {
        let x = data();
        let t = crate::ttest::one_sample_t(&x, 5.0, Alternative::TwoSided, 0.95).unwrap();
        let rep = t.to_report("One-Sample T");
        let tables = rep.tables();
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].columns[0].header, "Sample");
        let last = tables[1].columns.last().unwrap();
        assert_eq!(last.header, "P-Value");
        assert_eq!(last.kind, ColumnKind::PValue);
        let txt = rep.render_text();
        assert!(txt.starts_with("One-Sample T\n============\n"));
    }

    #[test]
    fn crosstab_totals_and_warnings() {
        let ct = Crosstab {
            row_labels: vec!["A".into(), "B".into()],
            col_labels: vec!["x".into(), "y".into()],
            counts: vec![vec![3, 1], vec![2, 4]],
            n_excluded: 1,
        };
        let sec = ct.report_section();
        let Block::Table(t) = &sec.blocks[0] else { panic!("table expected") };
        assert_eq!(t.rows.last().unwrap(), &vec![Cell::text("All"), Cell::Int(5), Cell::Int(5), Cell::Int(10)]);
        assert!(t.footnotes[0].contains("excluded"));
        let chi = crate::crosstab::pearson_chi_square(&ct).unwrap();
        assert!(chi.report_section().warnings.iter().any(|w| w.contains("expected counts")));
    }

    #[test]
    fn power_section_contains_results_and_curve() {
        let m = TwoSampleT { sigma: 1.0, alpha: 0.05, alternative: Alternative::TwoSided };
        let sol = m.solve(None, Some(0.8), Some(0.5)).unwrap();
        let sec = power_section(&m, &sol);
        let rep = {
            let mut r = SessionReport::new("Power");
            r.push(sec);
            r
        };
        let t = &rep.tables()[1];
        assert_eq!(t.rows[0][1], Cell::Int(64));
        let text = t.render_text();
        assert!(text.contains("Target Power") && text.contains("0.8000"));
        let charts = rep.charts();
        assert_eq!(charts.len(), 1);
        let c = charts[0];
        assert_eq!(c.series[0].x.len(), 41);
        assert!(c.series[0].y.windows(2).all(|w| w[1] >= w[0] - 1e-12), "power curve must increase");
        assert_eq!(c.reference_lines.len(), 3);
        // Solving for the effect works for the one-sample design as well.
        let o = OneSampleT { sigma: 2.0, alpha: 0.05, alternative: Alternative::TwoSided };
        let s2 = o.solve(Some(20), Some(0.9), None).unwrap();
        assert!(power_section(&o, &s2).blocks.len() >= 3);
    }
}
