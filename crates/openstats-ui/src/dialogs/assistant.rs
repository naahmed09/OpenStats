//! The Assistant menu: recommend an analysis from the shape of the data.

use super::analysis::{labels, numeric};
use super::menu::Topic;
use openstats_core::Worksheet;
use openstats_stats::normality::shapiro_wilk;

/// A recommended analysis and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recommendation {
    /// Registry id of the dialog to open.
    pub analysis: String,
    /// Plain-language reason.
    pub reason: String,
}

fn rec(id: &str, reason: impl Into<String>) -> Recommendation {
    Recommendation { analysis: id.into(), reason: reason.into() }
}

/// Number of distinct non-empty labels in a column.
fn distinct(ws: &Worksheet, c: usize) -> usize {
    let mut v: Vec<String> = labels(ws, c).unwrap_or_default().into_iter().filter(|s| !s.is_empty()).collect();
    v.sort();
    v.dedup();
    v.len()
}

/// Recommend an analysis for `topic`, looking at the selected columns `cols` (the active
/// column when the user has not selected any).
pub fn recommend(ws: &Worksheet, topic: Topic, cols: &[usize]) -> Recommendation {
    let numeric_cols: Vec<usize> = cols
        .iter()
        .copied()
        .filter(|&c| ws.columns.get(c).is_some_and(|x| x.kind() != openstats_core::ColumnKind::Text))
        .collect();
    let text_cols: Vec<usize> = cols
        .iter()
        .copied()
        .filter(|&c| ws.columns.get(c).is_some_and(|x| x.kind() == openstats_core::ColumnKind::Text))
        .collect();
    match topic {
        Topic::HypothesisTests => {
            match (numeric_cols.len(), text_cols.first()) {
                (1, Some(&g)) => {
                    let k = distinct(ws, g);
                    if k == 2 {
                        rec("stat.basic.ttest2", "One numeric column and a grouping column with two groups: compare the group means (stack the groups into two columns first, or use One-Way ANOVA, which gives the same test).")
                    } else if k > 2 {
                        rec(
                            "stat.anova.oneway",
                            format!("One numeric column and {k} groups: compare the group means with one-way ANOVA."),
                        )
                    } else {
                        rec(
                            "stat.basic.descriptive",
                            "The grouping column has fewer than two groups; start with descriptive statistics.",
                        )
                    }
                }
                (2, None) => {
                    let (a, b) = (
                        numeric(ws, numeric_cols[0]).unwrap_or_default(),
                        numeric(ws, numeric_cols[1]).unwrap_or_default(),
                    );
                    if a.len() == b.len() && !a.is_empty() {
                        rec("stat.basic.paired", "Two numeric columns of equal length: if each row is a pair of measurements on the same unit, use the paired t-test.")
                    } else {
                        rec(
                            "stat.basic.ttest2",
                            "Two numeric columns of different length are independent samples: use the 2-sample t-test.",
                        )
                    }
                }
                (1, None) => {
                    let d = numeric(ws, numeric_cols[0]).unwrap_or_default();
                    let clean: Vec<f64> = d.into_iter().filter(|x| !x.is_nan()).collect();
                    let p = shapiro_wilk(&clean).map(|r| r.p_value).ok();
                    match p {
                    Some(p) if p < 0.05 => rec("stat.basic.normality", format!("Shapiro-Wilk p = {p:.3}: the data look non-normal; check normality before testing a mean.")),
                    _ => rec("stat.basic.ttest1", "One numeric column: test its mean against a target with the 1-sample t-test."),
                }
                }
                _ => rec(
                    "stat.basic.descriptive",
                    "Select one numeric column (optionally with a grouping column) to get a specific recommendation.",
                ),
            }
        }
        Topic::ControlCharts => {
            let n = numeric_cols.first().and_then(|&c| numeric(ws, c).ok()).map_or(0, |d| d.len());
            if n >= 100 {
                rec("stat.spc.xbar", format!("{n} observations: group them into subgroups (for example 5) and use an Xbar chart, which detects smaller shifts."))
            } else {
                rec("stat.spc.imr", "Individual measurements: use the I-MR chart.")
            }
        }
        Topic::Capability => rec(
            "stat.quality.capability",
            "Enter the specification limits to get Cp, Cpk, Pp, Ppk and expected PPM; check normality first.",
        ),
        Topic::Regression => {
            if numeric_cols.len() >= 2 {
                rec("stat.regression.fit", "Use the first column as the response and the others as predictors.")
            } else {
                rec(
                    "graph.scatter",
                    "Plot the relationship first; select at least two numeric columns to fit a regression.",
                )
            }
        }
        Topic::Doe => rec("stat.doe.factorial", "Start with a two-level full factorial design to screen factors."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openstats_core::column::Column;

    fn ws() -> Worksheet {
        let mut w = Worksheet::new("W");
        w.add_column(Column::numeric("y", (0..30).map(|i| (i % 7) as f64 + 0.1 * i as f64).collect())).unwrap();
        w.add_column(Column::text("g2", (0..30).map(|i| if i % 2 == 0 { "a" } else { "b" }).collect::<Vec<_>>()))
            .unwrap();
        w.add_column(Column::text("g3", (0..30).map(|i| ["a", "b", "c"][i % 3]).collect::<Vec<_>>())).unwrap();
        w.add_column(Column::numeric("z", (0..30).map(f64::from).collect())).unwrap();
        w
    }

    #[test]
    fn recommendations_follow_the_data() {
        let w = ws();
        assert_eq!(recommend(&w, Topic::HypothesisTests, &[0, 1]).analysis, "stat.basic.ttest2");
        assert_eq!(recommend(&w, Topic::HypothesisTests, &[0, 2]).analysis, "stat.anova.oneway");
        assert_eq!(recommend(&w, Topic::HypothesisTests, &[0, 3]).analysis, "stat.basic.paired");
        assert!(matches!(
            recommend(&w, Topic::HypothesisTests, &[0]).analysis.as_str(),
            "stat.basic.ttest1" | "stat.basic.normality"
        ));
        assert_eq!(recommend(&w, Topic::HypothesisTests, &[]).analysis, "stat.basic.descriptive");
        assert_eq!(recommend(&w, Topic::ControlCharts, &[0]).analysis, "stat.spc.imr");
        assert_eq!(recommend(&w, Topic::Capability, &[0]).analysis, "stat.quality.capability");
        assert_eq!(recommend(&w, Topic::Regression, &[0, 3]).analysis, "stat.regression.fit");
        assert_eq!(recommend(&w, Topic::Regression, &[0]).analysis, "graph.scatter");
        assert_eq!(recommend(&w, Topic::Doe, &[]).analysis, "stat.doe.factorial");
    }

    #[test]
    fn long_series_suggest_subgroups() {
        let mut w = Worksheet::new("W");
        w.add_column(Column::numeric("x", (0..200).map(f64::from).collect())).unwrap();
        assert_eq!(recommend(&w, Topic::ControlCharts, &[0]).analysis, "stat.spc.xbar");
    }
}
