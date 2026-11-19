//! Sample quantiles: the nine definitions of Hyndman & Fan (1996).
//!
//! Minitab computes quartiles and percentiles with the position (n+1)p and
//! linear interpolation, which is [`QuantileMethod::Type6`]
//! ([`QuantileMethod::MINITAB`]). Positions outside 1..=n are clamped to the
//! minimum / maximum.

use crate::error::{invalid, Result, StatsError};

/// Hyndman–Fan quantile definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// Inverse empirical cdf.
    Type1,
    /// Type 1 with averaging at discontinuities.
    Type2,
    /// Nearest even order statistic (SAS default).
    Type3,
    /// Linear interpolation of the empirical cdf, p = k/n.
    Type4,
    /// Piecewise linear, p = (k − ½)/n.
    Type5,
    /// p = k/(n+1): Minitab, SPSS, Weibull plotting position.
    Type6,
    /// p = (k−1)/(n−1): R and NumPy default.
    Type7,
    /// Median-unbiased, p = (k−⅓)/(n+⅓).
    Type8,
    /// Normal-unbiased, p = (k−⅜)/(n+¼).
    Type9,
}

impl QuantileMethod {
    /// The definition used by Minitab.
    pub const MINITAB: QuantileMethod = QuantileMethod::Type6;
}

/// Quantile of an **already sorted**, non-missing sample.
///
/// `p` must lie in `[0, 1]`; the sample must be non-empty.
pub fn quantile_sorted(sorted: &[f64], p: f64, method: QuantileMethod) -> Result<f64> {
    let n = sorted.len();
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "quantile", needed: 1, got: 0 });
    }
    if !(0.0..=1.0).contains(&p) {
        return invalid(format!("probability {p} is outside [0, 1]"));
    }
    let nf = n as f64;
    // 1-based order statistic with the usual end conventions.
    let x = |k: i64| -> f64 { sorted[(k.clamp(1, n as i64) - 1) as usize] };
    // floor(v) and v - floor(v), snapping values within rounding of an integer.
    let split = |v: f64| -> (i64, f64) {
        let r = v.round();
        if (v - r).abs() <= 8.0 * f64::EPSILON * v.abs().max(1.0) {
            (r as i64, 0.0)
        } else {
            let j = v.floor();
            (j as i64, v - j)
        }
    };
    use QuantileMethod::*;
    Ok(match method {
        Type1 | Type2 | Type3 => {
            let m = if method == Type3 { -0.5 } else { 0.0 };
            let (j, g) = split(nf * p + m);
            let gamma = match method {
                Type1 => {
                    if g > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                Type2 => {
                    if g > 0.0 {
                        1.0
                    } else {
                        0.5
                    }
                }
                _ => {
                    if g == 0.0 && j % 2 == 0 {
                        0.0
                    } else {
                        1.0
                    }
                }
            };
            if gamma == 0.0 {
                x(j)
            } else if gamma == 1.0 {
                x(j + 1)
            } else {
                0.5 * (x(j) + x(j + 1))
            }
        }
        _ => {
            let m = match method {
                Type4 => 0.0,
                Type5 => 0.5,
                Type6 => p,
                Type7 => 1.0 - p,
                Type8 => (p + 1.0) / 3.0,
                _ => p / 4.0 + 3.0 / 8.0,
            };
            let (j, g) = split(nf * p + m);
            let (a, b) = (x(j), x(j + 1));
            if g == 0.0 {
                a
            } else {
                a + g * (b - a)
            }
        }
    })
}

/// Quantile of unsorted data; missing values (`NaN`) are ignored.
pub fn quantile(data: &[f64], p: f64, method: QuantileMethod) -> Result<f64> {
    let v = crate::clean_sorted(data)?;
    quantile_sorted(&v, p, method)
}

/// Several quantiles of one sample (sorted once).
pub fn quantiles(data: &[f64], ps: &[f64], method: QuantileMethod) -> Result<Vec<f64>> {
    let v = crate::clean_sorted(data)?;
    ps.iter().map(|&p| quantile_sorted(&v, p, method)).collect()
}

/// Median of the non-missing values.
pub fn median(data: &[f64]) -> Result<f64> {
    quantile(data, 0.5, QuantileMethod::Type7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minitab_quartiles() {
        // n = 8: Q1 at position 2.25, Q3 at 6.75.
        let x = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0];
        let q = quantiles(&x, &[0.25, 0.5, 0.75], QuantileMethod::MINITAB).unwrap();
        assert!((q[0] - 2.5).abs() < 1e-14);
        assert!((q[1] - 9.0).abs() < 1e-14);
        assert!((q[2] - 20.5).abs() < 1e-14);
    }

    #[test]
    fn clamping_and_errors() {
        let x = [3.0, 1.0, 2.0];
        assert_eq!(quantile(&x, 0.01, QuantileMethod::MINITAB).unwrap(), 1.0);
        assert_eq!(quantile(&x, 0.99, QuantileMethod::MINITAB).unwrap(), 3.0);
        assert!(quantile(&[], 0.5, QuantileMethod::MINITAB).is_err());
        assert!(quantile(&x, 1.5, QuantileMethod::MINITAB).is_err());
        assert_eq!(quantile(&[f64::NAN, 5.0], 0.5, QuantileMethod::MINITAB).unwrap(), 5.0);
        assert!(quantile(&[f64::INFINITY, 5.0], 0.5, QuantileMethod::MINITAB).is_err());
    }
}
