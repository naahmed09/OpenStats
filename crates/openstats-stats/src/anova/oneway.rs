//! One-way analysis of variance with Welch's test, model-summary statistics,
//! per-level intervals and chart data.

use super::comparisons::{self, Alternative, Best, Comparisons, McbResult, Means};
use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::report::{ChartKind, ChartSpec, Series};
use openstats_math::dist::{ContinuousDistribution, FisherF, StudentT};

/// Options of a one-way analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneWayOptions {
    /// Confidence level of the interval estimates.
    pub level: f64,
}

impl Default for OneWayOptions {
    fn default() -> Self {
        OneWayOptions { level: 0.95 }
    }
}

/// Summary of one factor level.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelSummary {
    /// Level label.
    pub label: String,
    /// Number of non-missing observations.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Sample standard deviation (`None` for a single observation).
    pub sd: Option<f64>,
    /// Interval for the mean using the pooled standard deviation.
    pub ci_pooled: Interval,
    /// Interval for the mean using the level's own standard deviation (`None`
    /// when the level has fewer than two observations).
    pub ci_individual: Option<Interval>,
}

/// Welch's test of equal means without assuming equal variances.
#[derive(Debug, Clone, PartialEq)]
pub struct Welch {
    /// F statistic.
    pub f: f64,
    /// Numerator degrees of freedom `k − 1`.
    pub df1: f64,
    /// Denominator degrees of freedom (Satterthwaite-type).
    pub df2: f64,
    /// p-value.
    pub p_value: f64,
}

/// Result of a one-way analysis of variance.
#[derive(Debug, Clone, PartialEq)]
pub struct OneWayAnova {
    /// Per-level summaries, in level order.
    pub levels: Vec<LevelSummary>,
    /// Factor degrees of freedom.
    pub df_factor: usize,
    /// Error degrees of freedom.
    pub df_error: usize,
    /// Between-level sum of squares.
    pub ss_factor: f64,
    /// Within-level (error) sum of squares.
    pub ss_error: f64,
    /// Total sum of squares.
    pub ss_total: f64,
    /// Factor mean square.
    pub ms_factor: f64,
    /// Error mean square.
    pub ms_error: f64,
    /// F statistic (`None` when MSE is 0).
    pub f: Option<f64>,
    /// p-value of `f`.
    pub p_value: Option<f64>,
    /// Pooled standard deviation `√MSE`.
    pub s: f64,
    /// R².
    pub r2: f64,
    /// Adjusted R².
    pub r2_adj: f64,
    /// Predicted R² = 1 − PRESS/SST (`None` when a level has one observation).
    pub r2_pred: Option<f64>,
    /// Welch's test (`None` unless every level has ≥ 2 observations and a positive SD).
    pub welch: Option<Welch>,
    /// Confidence level of the intervals.
    pub level: f64,
    /// The observations per level (missing values removed).
    pub groups: Vec<Vec<f64>>,
}

/// One-way ANOVA of `groups` (missing values are dropped). Labels default to
/// `1, 2, …`.
pub fn one_way(groups: &[&[f64]], labels: Option<&[String]>, opts: &OneWayOptions) -> Result<OneWayAnova> {
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
    }
    if let Some(l) = labels {
        if l.len() != groups.len() {
            return invalid("labels and groups differ in number");
        }
    }
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        if g.iter().any(|v| v.is_infinite()) {
            return Err(StatsError::NonFinite);
        }
        let clean: Vec<f64> = g.iter().copied().filter(|v| !v.is_nan()).collect();
        if clean.is_empty() {
            continue;
        }
        names.push(labels.map_or_else(|| (i + 1).to_string(), |l| l[i].clone()));
        data.push(clean);
    }
    let k = data.len();
    if k < 2 {
        return Err(StatsError::InsufficientData { what: "one-way ANOVA (levels)", needed: 2, got: k });
    }
    let total_n: usize = data.iter().map(Vec::len).sum();
    if total_n <= k {
        return Err(StatsError::InsufficientData { what: "one-way ANOVA", needed: k + 1, got: total_n });
    }
    let nf = total_n as f64;
    let means: Vec<f64> = data.iter().map(|g| g.iter().sum::<f64>() / g.len() as f64).collect();
    let grand = data.iter().flatten().sum::<f64>() / nf;
    let ss_factor: f64 = data.iter().zip(&means).map(|(g, m)| g.len() as f64 * (m - grand).powi(2)).sum();
    let ss_error: f64 = data.iter().zip(&means).map(|(g, m)| g.iter().map(|v| (v - m).powi(2)).sum::<f64>()).sum();
    let ss_total = ss_factor + ss_error;
    let (df_factor, df_error) = (k - 1, total_n - k);
    let ms_factor = ss_factor / df_factor as f64;
    let ms_error = ss_error / df_error as f64;
    let (f, p_value) = if ms_error > 0.0 {
        let f = ms_factor / ms_error;
        (Some(f), Some(FisherF::new(df_factor as f64, df_error as f64)?.sf(f)))
    } else {
        (None, None)
    };
    let s = ms_error.sqrt();
    let r2 = if ss_total > 0.0 { ss_factor / ss_total } else { 0.0 };
    let r2_adj = if ss_total > 0.0 { 1.0 - ms_error / (ss_total / (nf - 1.0)) } else { 0.0 };
    let r2_pred = if data.iter().any(|g| g.len() < 2) || ss_total <= 0.0 {
        None
    } else {
        let press: f64 = data
            .iter()
            .zip(&means)
            .map(|(g, m)| {
                let h = 1.0 - 1.0 / g.len() as f64;
                g.iter().map(|v| ((v - m) / h).powi(2)).sum::<f64>()
            })
            .sum();
        Some(1.0 - press / ss_total)
    };
    let sds: Vec<Option<f64>> = data
        .iter()
        .zip(&means)
        .map(|(g, m)| {
            (g.len() >= 2).then(|| (g.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (g.len() - 1) as f64).sqrt())
        })
        .collect();
    let tq = StudentT::new(df_error as f64)?.isf((1.0 - opts.level) / 2.0);
    let levels: Vec<LevelSummary> = (0..k)
        .map(|i| {
            let n = data[i].len();
            let half = tq * (ms_error / n as f64).sqrt();
            let indiv = sds[i].map(|sd| {
                let t = StudentT::new((n - 1) as f64).map(|d| d.isf((1.0 - opts.level) / 2.0)).unwrap_or(f64::NAN);
                let h = t * sd / (n as f64).sqrt();
                Interval { lower: means[i] - h, upper: means[i] + h, level: opts.level }
            });
            LevelSummary {
                label: names[i].clone(),
                n,
                mean: means[i],
                sd: sds[i],
                ci_pooled: Interval { lower: means[i] - half, upper: means[i] + half, level: opts.level },
                ci_individual: indiv,
            }
        })
        .collect();
    let welch = welch_test(&data, &means, &sds)?;
    Ok(OneWayAnova {
        levels,
        df_factor,
        df_error,
        ss_factor,
        ss_error,
        ss_total,
        ms_factor,
        ms_error,
        f,
        p_value,
        s,
        r2,
        r2_adj,
        r2_pred,
        welch,
        level: opts.level,
        groups: data,
    })
}

fn welch_test(data: &[Vec<f64>], means: &[f64], sds: &[Option<f64>]) -> Result<Option<Welch>> {
    if sds.iter().any(|s| s.is_none_or(|v| v <= 0.0)) {
        return Ok(None);
    }
    let k = data.len() as f64;
    let w: Vec<f64> = data.iter().zip(sds).map(|(g, s)| g.len() as f64 / s.unwrap_or(1.0).powi(2)).collect();
    let wsum: f64 = w.iter().sum();
    let xt: f64 = w.iter().zip(means).map(|(w, m)| w * m).sum::<f64>() / wsum;
    let lam: f64 = data.iter().zip(&w).map(|(g, w)| (1.0 - w / wsum).powi(2) / (g.len() - 1) as f64).sum();
    let num: f64 = w.iter().zip(means).map(|(w, m)| w * (m - xt).powi(2)).sum::<f64>() / (k - 1.0);
    let f = num / (1.0 + 2.0 * (k - 2.0) / (k * k - 1.0) * lam);
    let df2 = (k * k - 1.0) / (3.0 * lam);
    Ok(Some(Welch { f, df1: k - 1.0, df2, p_value: FisherF::new(k - 1.0, df2)?.sf(f) }))
}

/// One-way ANOVA of stacked data: `values[i]` belongs to the level `factor[i]`
/// (empty labels and `NaN` values are skipped). Levels are ordered numerically
/// when all labels are numbers, alphabetically otherwise.
pub fn one_way_stacked(values: &[f64], factor: &[String], opts: &OneWayOptions) -> Result<OneWayAnova> {
    if values.len() != factor.len() {
        return invalid("values and factor differ in length");
    }
    let mut levels: Vec<String> = Vec::new();
    for f in factor.iter().filter(|f| !f.is_empty()) {
        if !levels.contains(f) {
            levels.push(f.clone());
        }
    }
    crate::model::level_sort(&mut levels);
    let groups: Vec<Vec<f64>> =
        levels.iter().map(|l| values.iter().zip(factor).filter(|(_, f)| *f == l).map(|(v, _)| *v).collect()).collect();
    let refs: Vec<&[f64]> = groups.iter().map(Vec::as_slice).collect();
    one_way(&refs, Some(&levels), opts)
}

impl OneWayAnova {
    /// The level means with the pooled error estimate, ready for the
    /// multiple-comparison procedures.
    pub fn means(&self) -> Result<Means> {
        Means::new(
            self.levels.iter().map(|l| l.label.clone()).collect(),
            self.levels.iter().map(|l| l.n as f64).collect(),
            self.levels.iter().map(|l| l.mean).collect(),
            self.ms_error,
            self.df_error as f64,
        )
    }

    /// Tukey (Tukey–Kramer) comparisons at family confidence `1 − alpha`.
    pub fn tukey(&self, alpha: f64) -> Result<Comparisons> {
        comparisons::tukey(&self.means()?, alpha)
    }

    /// Fisher's LSD comparisons at individual confidence `1 − alpha`.
    pub fn fisher(&self, alpha: f64) -> Result<Comparisons> {
        comparisons::fisher(&self.means()?, alpha)
    }

    /// Dunnett's comparisons with the level at index `control`.
    pub fn dunnett(&self, control: usize, alternative: Alternative, alpha: f64) -> Result<Comparisons> {
        comparisons::dunnett(&self.means()?, control, alternative, alpha)
    }

    /// Hsu's multiple comparisons with the best.
    pub fn hsu_mcb(&self, best: Best, alpha: f64) -> Result<McbResult> {
        comparisons::hsu_mcb(&self.means()?, best, alpha)
    }

    /// Interval plot of the level means (pooled-SD intervals): `x` is the mean
    /// and `y` the level position `1..k`; the series `lower` / `upper` carry the
    /// interval ends. Level labels are in `categories`.
    pub fn interval_plot(&self) -> ChartSpec {
        let pos: Vec<f64> = (1..=self.levels.len()).map(|i| i as f64).collect();
        ChartSpec {
            kind: ChartKind::IntervalPlot,
            title: format!("Interval Plot ({}% CI for the Mean)", self.level * 100.0),
            x_label: "Mean".into(),
            y_label: "Level".into(),
            categories: self.levels.iter().map(|l| l.label.clone()).collect(),
            series: vec![
                Series { name: "estimate".into(), x: self.levels.iter().map(|l| l.mean).collect(), y: pos.clone() },
                Series {
                    name: "lower".into(),
                    x: self.levels.iter().map(|l| l.ci_pooled.lower).collect(),
                    y: pos.clone(),
                },
                Series { name: "upper".into(), x: self.levels.iter().map(|l| l.ci_pooled.upper).collect(), y: pos },
            ],
            reference_lines: Vec::new(),
        }
    }

    /// Individual value plot: every observation at its level position, plus a
    /// `Mean` series.
    pub fn individual_value_plot(&self) -> ChartSpec {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (i, g) in self.groups.iter().enumerate() {
            for &v in g {
                x.push((i + 1) as f64);
                y.push(v);
            }
        }
        ChartSpec {
            kind: ChartKind::Scatter,
            title: "Individual Value Plot".into(),
            x_label: "Level".into(),
            y_label: "Value".into(),
            categories: self.levels.iter().map(|l| l.label.clone()).collect(),
            series: vec![
                Series { name: "Values".into(), x, y },
                Series {
                    name: "Mean".into(),
                    x: (1..=self.levels.len()).map(|i| i as f64).collect(),
                    y: self.levels.iter().map(|l| l.mean).collect(),
                },
            ],
            reference_lines: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_example_and_errors() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 3.0, 4.0];
        let c = [5.0, 6.0, 7.0];
        let r = one_way(&[&a, &b, &c], None, &OneWayOptions::default()).unwrap();
        assert_eq!((r.df_factor, r.df_error), (2, 6));
        assert!((r.ss_factor - 26.0).abs() < 1e-12 && (r.ss_error - 6.0).abs() < 1e-12);
        assert!((r.f.unwrap() - 13.0).abs() < 1e-12 && r.welch.is_some());
        assert_eq!(r.interval_plot().series.len(), 3);
        assert_eq!(r.individual_value_plot().series[0].y.len(), 9);
        let st = one_way_stacked(
            &[1.0, 2.0, f64::NAN, 5.0, 6.0],
            &["x", "x", "x", "y", "y"].map(String::from),
            &OneWayOptions::default(),
        );
        assert!(st.is_ok());
        assert!(one_way(&[&a], None, &OneWayOptions::default()).is_err());
        assert!(one_way(&[&a, &[f64::INFINITY]], None, &OneWayOptions::default()).is_err());
        assert!(one_way(&[&a, &b], Some(&["only".to_string()]), &OneWayOptions::default()).is_err());
    }
}
