//! Gage linearity and bias (AIAG / Minitab conventions).
//!
//! Every measurement `y` of a reference part `x` gives a *bias* `b = y − x`.
//!
//! * **Linearity**: ordinary least squares of the biases on the reference values
//!   (all `N` measurements, `N − 2` df); `Linearity = |slope| · process variation`
//!   and `%Linearity = 100·|slope|`.
//! * **Bias** of a reference value: mean of its measurements minus the
//!   reference, tested against 0 with a one-sample t test (`nⱼ − 1` df);
//!   `%Bias = 100·|bias|/process variation`. The *average* bias uses all `N`
//!   measurements (`N − 1` df).
//!
//! The process variation is the study variation of the process (typically
//! `6σ` from a Gage R&R study) and must be supplied.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::report::{Axis, ChartKind, ChartSpec, ReferenceLine, Series};
use openstats_math::dist::{ContinuousDistribution, StudentT};

/// Data of a linearity and bias study.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearityBiasStudy {
    reference: Vec<f64>,
    measured: Vec<f64>,
}

/// Options of a linearity and bias analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearityBiasOptions {
    /// Process variation (study variation of the process, e.g. 6σ); positive.
    pub process_variation: f64,
    /// Confidence level of the intervals (0.95).
    pub level: f64,
}

/// Coefficient of the bias-on-reference regression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficient {
    /// Estimate.
    pub coef: f64,
    /// Standard error.
    pub se: f64,
    /// t statistic.
    pub t: f64,
    /// Two-sided p-value.
    pub p: f64,
    /// Confidence interval.
    pub ci: Interval,
}

/// Regression of the bias on the reference value.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearityRegression {
    /// Intercept.
    pub intercept: Coefficient,
    /// Slope.
    pub slope: Coefficient,
    /// Residual standard deviation.
    pub s: f64,
    /// R² (a fraction; `None` when the biases have no variation).
    pub r_squared: Option<f64>,
    /// Residual degrees of freedom `N − 2`.
    pub df: usize,
}

/// Bias at one reference value (or on average).
#[derive(Debug, Clone, PartialEq)]
pub struct BiasRow {
    /// Reference value (`None` for the average row).
    pub reference: Option<f64>,
    /// Number of measurements.
    pub n: usize,
    /// Mean of the measurements.
    pub mean_measured: f64,
    /// Bias = mean measurement − reference (mean bias for the average row).
    pub bias: f64,
    /// `100·|bias|/process variation`.
    pub pct_bias: f64,
    /// Standard error of the bias (`None` when `n = 1` or the measurements do not vary).
    pub se: Option<f64>,
    /// t statistic of H₀: bias = 0.
    pub t: Option<f64>,
    /// Two-sided p-value.
    pub p: Option<f64>,
    /// Confidence interval of the bias.
    pub ci: Option<Interval>,
}

/// Result of a linearity and bias analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearityBias {
    /// Number of measurements.
    pub n: usize,
    /// Process variation used.
    pub process_variation: f64,
    /// Confidence level.
    pub level: f64,
    /// The regression.
    pub regression: LinearityRegression,
    /// `|slope| · process variation`.
    pub linearity: f64,
    /// `100·|slope|`.
    pub pct_linearity: f64,
    /// Bias per reference value, in increasing reference order.
    pub biases: Vec<BiasRow>,
    /// Average bias over all measurements.
    pub average_bias: BiasRow,
    /// Individual biases as (reference, bias) pairs.
    pub points: Vec<[f64; 2]>,
    /// Fitted line over the reference range with its confidence band (x, fit, lower, upper).
    pub band: Vec<[f64; 4]>,
}

impl LinearityBiasStudy {
    /// A study from parallel columns of reference values and measurements.
    pub fn new(reference: &[f64], measured: &[f64]) -> Result<Self> {
        if reference.len() != measured.len() {
            return invalid(format!(
                "reference ({}) and measurement ({}) columns must have the same length",
                reference.len(),
                measured.len()
            ));
        }
        let mut r = Vec::new();
        let mut m = Vec::new();
        for (&x, &y) in reference.iter().zip(measured) {
            if x.is_infinite() || y.is_infinite() {
                return Err(StatsError::NonFinite);
            }
            if x.is_nan() || y.is_nan() {
                continue;
            }
            r.push(x);
            m.push(y);
        }
        Ok(LinearityBiasStudy { reference: r, measured: m })
    }

    /// Analyse the study.
    pub fn analyze(&self, opts: &LinearityBiasOptions) -> Result<LinearityBias> {
        if !(opts.process_variation > 0.0 && opts.process_variation.is_finite()) {
            return invalid("the process variation must be positive and finite");
        }
        if !(opts.level > 0.0 && opts.level < 1.0) {
            return invalid(format!("confidence level {} must be in (0, 1)", opts.level));
        }
        let n = self.reference.len();
        if n < 3 {
            return Err(StatsError::InsufficientData { what: "linearity and bias study", needed: 3, got: n });
        }
        let mut refs: Vec<f64> = self.reference.clone();
        refs.sort_by(|a, b| a.total_cmp(b));
        refs.dedup();
        if refs.len() < 2 {
            return invalid("at least two distinct reference values are needed to assess linearity");
        }
        let nf = n as f64;
        let bias: Vec<f64> = self.measured.iter().zip(&self.reference).map(|(y, x)| y - x).collect();
        let xbar = self.reference.iter().sum::<f64>() / nf;
        let ybar = bias.iter().sum::<f64>() / nf;
        let sxx: f64 = self.reference.iter().map(|x| (x - xbar) * (x - xbar)).sum();
        let sxy: f64 = self.reference.iter().zip(&bias).map(|(x, b)| (x - xbar) * (b - ybar)).sum();
        let syy: f64 = bias.iter().map(|b| (b - ybar) * (b - ybar)).sum();
        let slope = sxy / sxx;
        let intercept = ybar - slope * xbar;
        let sse: f64 = self
            .reference
            .iter()
            .zip(&bias)
            .map(|(x, b)| {
                let e = b - intercept - slope * x;
                e * e
            })
            .sum::<f64>()
            .max(0.0);
        let df = n - 2;
        let dff = df as f64;
        let s = (sse / dff).sqrt();
        let tdist = StudentT::new(dff)?;
        let tcrit = tdist.quantile(1.0 - (1.0 - opts.level) / 2.0);
        let coef = |est: f64, se: f64| -> Coefficient {
            let t = est / se;
            Coefficient {
                coef: est,
                se,
                t,
                p: if se > 0.0 {
                    2.0 * tdist.sf(t.abs())
                } else if est == 0.0 {
                    1.0
                } else {
                    0.0
                },
                ci: Interval { lower: est - tcrit * se, upper: est + tcrit * se, level: opts.level },
            }
        };
        let se_slope = s / sxx.sqrt();
        let se_int = s * (1.0 / nf + xbar * xbar / sxx).sqrt();
        let regression = LinearityRegression {
            intercept: coef(intercept, se_int),
            slope: coef(slope, se_slope),
            s,
            r_squared: if syy > 0.0 { Some(1.0 - sse / syy) } else { None },
            df,
        };
        let pv = opts.process_variation;
        let row = |reference: Option<f64>, ys: &[f64], ref_value: f64| -> Result<BiasRow> {
            let m = ys.len();
            let mf = m as f64;
            let mean = ys.iter().sum::<f64>() / mf;
            let b = mean - ref_value;
            let (se, t, p, ci) = if m >= 2 {
                let sd = (ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / (mf - 1.0)).sqrt();
                if sd > 0.0 {
                    let d = StudentT::new(mf - 1.0)?;
                    let se = sd / mf.sqrt();
                    let t = b / se;
                    let c = d.quantile(1.0 - (1.0 - opts.level) / 2.0);
                    (
                        Some(se),
                        Some(t),
                        Some(2.0 * d.sf(t.abs())),
                        Some(Interval { lower: b - c * se, upper: b + c * se, level: opts.level }),
                    )
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };
            Ok(BiasRow { reference, n: m, mean_measured: mean, bias: b, pct_bias: 100.0 * b.abs() / pv, se, t, p, ci })
        };
        let mut biases = Vec::new();
        for &r in &refs {
            let ys: Vec<f64> =
                self.reference.iter().zip(&self.measured).filter(|(x, _)| **x == r).map(|(_, y)| *y).collect();
            biases.push(row(Some(r), &ys, r)?);
        }
        // Average bias: one-sample t test of the N individual biases against 0.
        let average_bias = {
            let mut b = row(None, &bias, 0.0)?;
            b.mean_measured = self.measured.iter().sum::<f64>() / nf;
            b
        };
        let (lo, hi) = (refs[0], refs[refs.len() - 1]);
        let steps = 40;
        let band = (0..=steps)
            .map(|i| {
                let x = lo + (hi - lo) * i as f64 / steps as f64;
                let fit = intercept + slope * x;
                let se = s * (1.0 / nf + (x - xbar) * (x - xbar) / sxx).sqrt();
                [x, fit, fit - tcrit * se, fit + tcrit * se]
            })
            .collect();
        Ok(LinearityBias {
            n,
            process_variation: pv,
            level: opts.level,
            regression,
            linearity: slope.abs() * pv,
            pct_linearity: 100.0 * slope.abs(),
            biases,
            average_bias,
            points: self.reference.iter().zip(&bias).map(|(&x, &b)| [x, b]).collect(),
            band,
        })
    }
}

impl LinearityBias {
    /// Chart: individual biases, the mean bias per reference value, the fitted
    /// line with its confidence band, and the zero-bias line.
    pub fn chart_spec(&self) -> ChartSpec {
        let col = |k: usize| self.band.iter().map(|r| r[k]).collect::<Vec<f64>>();
        ChartSpec {
            kind: ChartKind::Line,
            title: "Gage Linearity and Bias Study".into(),
            x_label: "Reference value".into(),
            y_label: "Bias".into(),
            categories: Vec::new(),
            series: vec![
                Series { name: "Fit".into(), x: col(0), y: col(1) },
                Series { name: "Lower limit".into(), x: col(0), y: col(2) },
                Series { name: "Upper limit".into(), x: col(0), y: col(3) },
                Series {
                    name: "Individual bias".into(),
                    x: self.points.iter().map(|p| p[0]).collect(),
                    y: self.points.iter().map(|p| p[1]).collect(),
                },
                Series {
                    name: "Mean bias".into(),
                    x: self.biases.iter().filter_map(|b| b.reference).collect(),
                    y: self.biases.iter().map(|b| b.bias).collect(),
                },
            ],
            reference_lines: vec![ReferenceLine { axis: Axis::Y, value: 0.0, label: "Bias = 0".into() }],
        }
    }
}
