//! [`Reportable`] implementations for the nonparametric tests.

use super::{
    AnsariBradley, Friedman, KruskalWallis, KsTwoSample, MannKendall, MannWhitney, MoodMedian, MoodScale, NormalScores,
    RunsTest, SignTest, WilcoxonSignedRank,
};
use crate::report::{Cell, Column, ColumnKind, Reportable, Section, Table};

const S: ColumnKind = ColumnKind::Significant(6);
const P: ColumnKind = ColumnKind::PValue;
const I: ColumnKind = ColumnKind::Integer;
const T: ColumnKind = ColumnKind::Text;

fn int(n: usize) -> Cell {
    Cell::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn one_row(title: &str, cols: Vec<(&str, ColumnKind, Cell)>) -> Table {
    let columns = cols.iter().map(|(h, k, _)| Column::new(*h, *k)).collect();
    let mut t = Table::new(title, columns);
    t.row(cols.into_iter().map(|(_, _, c)| c).collect());
    t
}

impl Reportable for SignTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Sign Test for Median");
        s.table(one_row(
            "Descriptive Statistics",
            vec![
                ("N", I, int(self.n)),
                ("Median", S, Cell::Num(self.median)),
                ("Below", I, int(self.below)),
                ("Equal", I, int(self.equal)),
                ("Above", I, int(self.above)),
                ("Null median", S, Cell::Num(self.null_median)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        if let (Some((lo, hi)), Some(c), Some((rl, ru))) = (self.ci, self.achieved_level, self.ci_ranks) {
            let mut t = one_row(
                "Confidence Interval for the Median",
                vec![
                    ("Achieved confidence", S, Cell::Num(100.0 * c)),
                    ("Lower", S, Cell::Num(lo)),
                    ("Upper", S, Cell::Num(hi)),
                    ("Lower position", I, int(rl)),
                    ("Upper position", I, int(ru)),
                ],
            );
            t.footnotes
                .push("Order-statistic interval; the achieved confidence is the largest not above the target.".into());
            s.table(t);
        }
        if self.equal > 0 {
            s.warn(format!("{} observation(s) equal to the null median were not used.", self.equal));
        }
        s
    }
}

impl Reportable for WilcoxonSignedRank {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Wilcoxon Signed Rank Test");
        s.table(one_row(
            "Test",
            vec![
                ("N", I, int(self.n)),
                ("N for test", I, int(self.n_used)),
                ("Wilcoxon statistic (W+)", S, Cell::Num(self.w_plus)),
                ("Null median", S, Cell::Num(self.null_median)),
                ("Z", S, self.z.map_or(Cell::Missing, Cell::Num)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        let mut t = one_row(
            "Hodges-Lehmann Estimate",
            vec![
                ("Estimate", S, Cell::Num(self.estimate)),
                ("Achieved confidence", S, Cell::Num(100.0 * self.achieved_level)),
                ("Lower", S, Cell::Num(self.ci.0)),
                ("Upper", S, Cell::Num(self.ci.1)),
            ],
        );
        t.footnotes.push(if self.exact { "Exact p-value.".into() } else { "Normal approximation".to_string() });
        s.table(t);
        if self.n_used < self.n {
            s.warn(format!("{} observation(s) equal to the null median were dropped.", self.n - self.n_used));
        }
        s
    }
}

impl Reportable for MannWhitney {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Mann-Whitney Test");
        let mut d = Table::new(
            "Descriptive Statistics",
            vec![Column::new("Sample", T), Column::new("N", I), Column::new("Median", S)],
        );
        d.row(vec![Cell::text("Sample 1"), int(self.n1), Cell::Num(self.median1)]);
        d.row(vec![Cell::text("Sample 2"), int(self.n2), Cell::Num(self.median2)]);
        s.table(d);
        s.table(one_row(
            "Test",
            vec![
                ("W", S, Cell::Num(self.w)),
                ("U", S, Cell::Num(self.u)),
                ("Z", S, self.z.map_or(Cell::Missing, Cell::Num)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        let mut t = one_row(
            "Estimation for Difference (eta1 - eta2)",
            vec![
                ("Difference", S, Cell::Num(self.estimate)),
                ("Achieved confidence", S, Cell::Num(100.0 * self.achieved_level)),
                ("Lower", S, Cell::Num(self.ci.0)),
                ("Upper", S, Cell::Num(self.ci.1)),
            ],
        );
        t.footnotes.push(if self.exact {
            "Exact p-value.".into()
        } else if self.ties {
            "Normal approximation, adjusted for ties.".into()
        } else {
            "Normal approximation.".to_string()
        });
        s.table(t);
        s
    }
}

impl KruskalWallis {
    /// Section with the group labels supplied by the caller.
    pub fn section_with_labels(&self, labels: &[String]) -> Section {
        let mut s = Section::new("Kruskal-Wallis Test");
        let mut d = Table::new(
            "Descriptive Statistics",
            vec![
                Column::new("Sample", T),
                Column::new("N", I),
                Column::new("Median", S),
                Column::new("Ave Rank", S),
                Column::new("Z-Value", S),
            ],
        );
        for (i, g) in self.groups.iter().enumerate() {
            let label = labels.get(i).cloned().unwrap_or_else(|| format!("Group {}", i + 1));
            d.row(vec![Cell::text(label), int(g.n), Cell::Num(g.median), Cell::Num(g.mean_rank), Cell::Num(g.z)]);
        }
        d.row(vec![
            Cell::text("Overall"),
            int(self.n),
            Cell::Missing,
            Cell::Num(self.overall_mean_rank),
            Cell::Missing,
        ]);
        s.table(d);
        let mut t = Table::new(
            "Test",
            vec![
                Column::new("Null hypothesis", T),
                Column::new("DF", I),
                Column::new("H-Value", S),
                Column::new("P-Value", P),
            ],
        );
        t.row(vec![Cell::text("All medians are equal"), int(self.df), Cell::Num(self.h), Cell::Num(self.p_value)]);
        if self.ties {
            t.row(vec![
                Cell::text("Adjusted for ties"),
                int(self.df),
                Cell::Num(self.h_adjusted),
                Cell::Num(self.p_value_adjusted),
            ]);
        }
        s.table(t);
        if self.groups.iter().any(|g| g.n < 5) {
            s.warn(
                "One or more samples have fewer than 5 observations; the chi-square approximation may be inaccurate.",
            );
        }
        s
    }
}

impl Reportable for KruskalWallis {
    fn report_section(&self) -> Section {
        self.section_with_labels(&[])
    }
}

impl MoodMedian {
    /// Section with the group labels supplied by the caller.
    pub fn section_with_labels(&self, labels: &[String]) -> Section {
        let mut s = Section::new("Mood Median Test");
        let mut t = Table::new(
            "Descriptive Statistics",
            vec![
                Column::new("Sample", T),
                Column::new("N<=Median", I),
                Column::new("N>Median", I),
                Column::new("Median", S),
                Column::new("Q3-Q1", S),
            ],
        );
        for (i, g) in self.groups.iter().enumerate() {
            let label = labels.get(i).cloned().unwrap_or_else(|| format!("Group {}", i + 1));
            t.row(vec![Cell::text(label), int(g.at_or_below), int(g.above), Cell::Num(g.median), Cell::Num(g.iqr)]);
        }
        t.footnotes.push(format!("Overall median = {:.6}", self.overall_median));
        s.table(t);
        s.table(one_row(
            "Test",
            vec![
                ("DF", I, int(self.df)),
                ("Chi-Square", S, Cell::Num(self.chi_square)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        if self.min_expected < 5.0 {
            s.warn("Some expected counts are less than 5; the chi-square approximation may be inaccurate.");
        }
        s
    }
}

impl Reportable for MoodMedian {
    fn report_section(&self) -> Section {
        self.section_with_labels(&[])
    }
}

impl Friedman {
    /// Section with the treatment labels supplied by the caller.
    pub fn section_with_labels(&self, labels: &[String]) -> Section {
        let mut s = Section::new("Friedman Test");
        let mut t = Table::new(
            "Treatments",
            vec![
                Column::new("Treatment", T),
                Column::new("N", I),
                Column::new("Est Median", S),
                Column::new("Sum of Ranks", S),
            ],
        );
        for j in 0..self.treatments {
            let label = labels.get(j).cloned().unwrap_or_else(|| format!("Treatment {}", j + 1));
            t.row(vec![
                Cell::text(label),
                int(self.blocks),
                Cell::Num(self.estimated_medians[j]),
                Cell::Num(self.rank_sums[j]),
            ]);
        }
        s.table(t);
        let mut tt = Table::new(
            "Test",
            vec![Column::new("Statistic", T), Column::new("DF", I), Column::new("S", S), Column::new("P-Value", P)],
        );
        tt.row(vec![Cell::text("S"), int(self.df), Cell::Num(self.statistic), Cell::Num(self.p_value)]);
        tt.row(vec![
            Cell::text("S (adjusted for ties)"),
            int(self.df),
            Cell::Num(self.statistic_adjusted),
            Cell::Num(self.p_value_adjusted),
        ]);
        s.table(tt);
        s
    }
}

impl Reportable for Friedman {
    fn report_section(&self) -> Section {
        self.section_with_labels(&[])
    }
}

impl Reportable for RunsTest {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Runs Test");
        s.table(one_row(
            "Runs",
            vec![
                ("K", S, Cell::Num(self.k)),
                ("Above K", I, int(self.above)),
                ("Below K", I, int(self.below)),
                ("Observed runs", I, int(self.runs)),
                ("Expected runs", S, Cell::Num(self.expected)),
            ],
        ));
        let mut cols = vec![("Z", S, Cell::Num(self.z)), ("P-Value (approx)", P, Cell::Num(self.p_value))];
        if let Some(p) = self.exact_p_value {
            cols.push(("P-Value (exact)", P, Cell::Num(p)));
        }
        s.table(one_row("Test", cols));
        if self.equal > 0 {
            s.warn(format!("{} value(s) equal to K were omitted.", self.equal));
        }
        s
    }
}

impl Reportable for KsTwoSample {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Two-Sample Kolmogorov-Smirnov Test");
        let mut t = one_row(
            "Test",
            vec![
                ("N1", I, int(self.n1)),
                ("N2", I, int(self.n2)),
                ("D+", S, Cell::Num(self.d_plus)),
                ("D-", S, Cell::Num(self.d_minus)),
                ("D", S, Cell::Num(self.d)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        );
        t.footnotes.push(if self.exact { "Exact p-value.".into() } else { "Asymptotic p-value.".to_string() });
        s.table(t);
        s
    }
}

impl Reportable for AnsariBradley {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Ansari-Bradley Test of Equal Scale");
        let mut t = one_row(
            "Test",
            vec![
                ("N1", I, int(self.n1)),
                ("N2", I, int(self.n2)),
                ("AB", S, Cell::Num(self.ab)),
                ("Mean", S, Cell::Num(self.mean)),
                ("Variance", S, Cell::Num(self.variance)),
                ("Z", S, self.z.map_or(Cell::Missing, Cell::Num)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        );
        t.footnotes.push(if self.exact { "Exact p-value.".into() } else { "Normal approximation.".to_string() });
        s.table(t);
        s
    }
}

impl Reportable for MoodScale {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Mood's Test of Equal Scale");
        s.table(one_row(
            "Test",
            vec![
                ("N1", I, int(self.n1)),
                ("N2", I, int(self.n2)),
                ("T", S, Cell::Num(self.t)),
                ("Mean", S, Cell::Num(self.mean)),
                ("Variance", S, Cell::Num(self.variance)),
                ("Z", S, Cell::Num(self.z)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for NormalScores {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Normal Scores (van der Waerden) Test");
        let mut d =
            Table::new("Groups", vec![Column::new("Group", T), Column::new("N", I), Column::new("Average score", S)]);
        for g in &self.groups {
            d.row(vec![Cell::text(g.label.clone()), int(g.n), Cell::Num(g.mean_score)]);
        }
        s.table(d);
        s.table(one_row(
            "Test",
            vec![
                ("DF", I, int(self.df)),
                ("Chi-Square", S, Cell::Num(self.statistic)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}

impl Reportable for MannKendall {
    fn report_section(&self) -> Section {
        let mut s = Section::new("Mann-Kendall Trend Test");
        s.table(one_row(
            "Test",
            vec![
                ("N", I, int(self.n)),
                ("S", S, Cell::Num(self.s)),
                ("Variance", S, Cell::Num(self.variance)),
                ("Z", S, Cell::Num(self.z)),
                ("Kendall tau", S, Cell::Num(self.tau)),
                ("Sen slope", S, Cell::Num(self.sen_slope)),
                ("P-Value", P, Cell::Num(self.p_value)),
            ],
        ));
        s
    }
}
