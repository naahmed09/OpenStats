//! Orthogonal (Deming) regression: a straight line fitted when both the
//! predictor and the response carry measurement error, with a known ratio
//! of the error variances.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::report::{Cell, ChartKind, ChartSpec, Column, ColumnKind, Reportable, Section, Series, Table};
use openstats_math::dist::{ContinuousDistribution, StudentT};

/// Result of an orthogonal regression.
#[derive(Debug, Clone)]
pub struct OrthogonalFit {
    /// Observations used.
    pub n: usize,
    /// Intercept.
    pub intercept: f64,
    /// Slope.
    pub slope: f64,
    /// Jackknife standard error of the intercept.
    pub se_intercept: f64,
    /// Jackknife standard error of the slope.
    pub se_slope: f64,
    /// Confidence interval of the intercept.
    pub ci_intercept: Interval,
    /// Confidence interval of the slope.
    pub ci_slope: Interval,
    /// Error variance ratio δ = Var(error in y) / Var(error in x).
    pub delta: f64,
    /// Fitted response values.
    pub fitted: Vec<f64>,
    /// Vertical residuals `y − fit`.
    pub residuals: Vec<f64>,
    /// Signed distances to the line along the direction implied by δ
    /// (perpendicular when δ = 1).
    pub orthogonal_distances: Vec<f64>,
    /// Worksheet rows used.
    pub rows_used: Vec<usize>,
    /// Confidence level.
    pub level: f64,
    /// Predictor name.
    pub x_name: String,
    /// Response name.
    pub y_name: String,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

fn deming(x: &[f64], y: &[f64], delta: f64) -> Result<(f64, f64)> {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>();
    let syy: f64 = y.iter().map(|v| (v - my).powi(2)).sum::<f64>();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>();
    if sxy == 0.0 {
        return invalid("the sample covariance of x and y is zero, so the orthogonal slope is not defined");
    }
    let d = syy - delta * sxx;
    let slope = (d + (d * d + 4.0 * delta * sxy * sxy).sqrt()) / (2.0 * sxy);
    Ok((my - slope * mx, slope))
}

/// Fit `y = α + β x` with error-variance ratio `delta` (1 = orthogonal).
/// Standard errors are jackknife estimates.
pub fn orthogonal_regression(
    x_name: &str,
    x_all: &[f64],
    y_name: &str,
    y_all: &[f64],
    delta: f64,
    level: f64,
) -> Result<OrthogonalFit> {
    if x_all.len() != y_all.len() {
        return invalid("x and y must have the same length");
    }
    if !(delta > 0.0 && delta.is_finite()) {
        return invalid("the error variance ratio must be positive");
    }
    if !(level > 0.0 && level < 1.0) {
        return invalid(format!("confidence level {level} must be in (0, 1)"));
    }
    let rows_used: Vec<usize> = (0..x_all.len()).filter(|&i| x_all[i].is_finite() && y_all[i].is_finite()).collect();
    let n = rows_used.len();
    if n < 4 {
        return Err(StatsError::InsufficientData { what: "orthogonal regression", needed: 4, got: n });
    }
    let x: Vec<f64> = rows_used.iter().map(|&i| x_all[i]).collect();
    let y: Vec<f64> = rows_used.iter().map(|&i| y_all[i]).collect();
    let (a, b) = deming(&x, &y, delta)?;
    // Delete-one jackknife.
    let mut ja = Vec::with_capacity(n);
    let mut jb = Vec::with_capacity(n);
    for i in 0..n {
        let xi: Vec<f64> = x.iter().enumerate().filter(|(k, _)| *k != i).map(|(_, v)| *v).collect();
        let yi: Vec<f64> = y.iter().enumerate().filter(|(k, _)| *k != i).map(|(_, v)| *v).collect();
        let (a_i, b_i) = deming(&xi, &yi, delta)?;
        ja.push(a_i);
        jb.push(b_i);
    }
    let nf = n as f64;
    let jack = |v: &[f64]| -> f64 {
        let m = v.iter().sum::<f64>() / nf;
        (((nf - 1.0) / nf) * v.iter().map(|t| (t - m).powi(2)).sum::<f64>()).sqrt()
    };
    let (se_a, se_b) = (jack(&ja), jack(&jb));
    let tq = StudentT::new(nf - 2.0)?.isf((1.0 - level) / 2.0);
    let fitted: Vec<f64> = x.iter().map(|v| a + b * v).collect();
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(p, q)| p - q).collect();
    let norm = (b * b + delta).sqrt();
    let orthogonal_distances = residuals.iter().map(|r| r * delta.sqrt() / norm).collect();
    Ok(OrthogonalFit {
        n,
        intercept: a,
        slope: b,
        se_intercept: se_a,
        se_slope: se_b,
        ci_intercept: Interval { lower: a - tq * se_a, upper: a + tq * se_a, level },
        ci_slope: Interval { lower: b - tq * se_b, upper: b + tq * se_b, level },
        delta,
        fitted,
        residuals,
        orthogonal_distances,
        rows_used,
        level,
        x_name: x_name.into(),
        y_name: y_name.into(),
        xs: x,
        ys: y,
    })
}

impl OrthogonalFit {
    /// Predicted response at `x`.
    pub fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
}

impl Reportable for OrthogonalFit {
    fn report_section(&self) -> Section {
        let mut sec = Section::new(format!("Orthogonal Regression Analysis: {} versus {}", self.y_name, self.x_name));
        sec.text(format!("Error variance ratio ({} / {}) = {}", self.y_name, self.x_name, self.delta));
        let mut t = Table::new(
            "Coefficients",
            vec![
                Column::new("Predictor", ColumnKind::Text),
                Column::new("Coef", ColumnKind::Significant(6)),
                Column::new("SE Coef", ColumnKind::Significant(6)),
                Column::new(format!("{}% CI", self.level * 100.0), ColumnKind::Text),
            ],
        );
        t.row(vec![
            Cell::text("Constant"),
            Cell::Num(self.intercept),
            Cell::Num(self.se_intercept),
            Cell::text(format!("({:.6}, {:.6})", self.ci_intercept.lower, self.ci_intercept.upper)),
        ]);
        t.row(vec![
            Cell::text(&self.x_name),
            Cell::Num(self.slope),
            Cell::Num(self.se_slope),
            Cell::text(format!("({:.6}, {:.6})", self.ci_slope.lower, self.ci_slope.upper)),
        ]);
        sec.table(t);
        sec.text(format!("{} = {:.6} + {:.6} {}", self.y_name, self.intercept, self.slope, self.x_name));
        sec.text("Standard errors are delete-one jackknife estimates.");
        let (lo, hi) = self.xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        sec.chart(ChartSpec {
            kind: ChartKind::Scatter,
            title: format!("Orthogonal Fit: {} versus {}", self.y_name, self.x_name),
            x_label: self.x_name.clone(),
            y_label: self.y_name.clone(),
            categories: vec![],
            series: vec![
                Series { name: self.y_name.clone(), x: self.xs.clone(), y: self.ys.clone() },
                Series { name: "Fit".into(), x: vec![lo, hi], y: vec![self.predict(lo), self.predict(hi)] },
            ],
            reference_lines: vec![],
        });
        sec
    }
}
