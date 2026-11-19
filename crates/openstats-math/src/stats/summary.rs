//! Sums, means, variances and autocorrelation with compensated arithmetic.

use crate::error::{MathError, Result};

/// Error-free sum `a + b = s + e` (Knuth's TwoSum).
#[inline]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Error-free product `a·b = p + e` using a fused multiply-add.
#[inline]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Neumaier's compensated sum of an iterator: the rounding error of every
/// addition is accumulated separately and added back at the end.
pub fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for v in values {
        let (t, e) = two_sum(s, v);
        s = t;
        c += e;
    }
    s + c
}

fn check(x: &[f64], min: usize, what: &'static str) -> Result<()> {
    if x.len() < min {
        return Err(MathError::InvalidArgument(format!("{what}: need at least {min} observations (got {})", x.len())));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(MathError::NonFinite { op: what });
    }
    Ok(())
}

/// Arithmetic mean by the corrected two-pass method (the second pass adds the
/// mean of the residuals, which removes the rounding error of the first).
pub fn mean(x: &[f64]) -> Result<f64> {
    check(x, 1, "mean")?;
    let n = x.len() as f64;
    let m0 = sum(x.iter().copied()) / n;
    Ok(m0 + sum(x.iter().map(|v| v - m0)) / n)
}

/// Sum of squared deviations from the mean, Σ(xᵢ − x̄)², by the corrected
/// two-pass algorithm (Chan, Golub & LeVeque).
pub fn sum_squared_deviations(x: &[f64]) -> Result<f64> {
    check(x, 1, "sum of squares")?;
    let n = x.len() as f64;
    let m = mean(x)?;
    let s2 = sum(x.iter().map(|v| (v - m) * (v - m)));
    let s1 = sum(x.iter().map(|v| v - m));
    Ok(s2 - s1 * s1 / n)
}

/// Sample variance with divisor n − 1.
pub fn variance(x: &[f64]) -> Result<f64> {
    check(x, 2, "variance")?;
    Ok(sum_squared_deviations(x)? / (x.len() as f64 - 1.0))
}

/// Sample standard deviation (divisor n − 1).
pub fn std_dev(x: &[f64]) -> Result<f64> {
    Ok(variance(x)?.sqrt())
}

/// Lag-`k` sample autocorrelation in NIST's definition:
/// Σ_{i>k}(xᵢ − x̄)(x_{i−k} − x̄) / Σ(xᵢ − x̄)².
pub fn autocorrelation(x: &[f64], lag: usize) -> Result<f64> {
    check(x, lag + 1, "autocorrelation")?;
    let m = mean(x)?;
    let den = sum(x.iter().map(|v| (v - m) * (v - m)));
    if den == 0.0 {
        return Err(MathError::InvalidArgument("autocorrelation: all observations are equal".into()));
    }
    let num = sum((lag..x.len()).map(|i| (x[i] - m) * (x[i - lag] - m)));
    Ok(num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_beats_naive() {
        let mut v = vec![1.0e16];
        v.extend(std::iter::repeat_n(1.0, 1000));
        v.push(-1.0e16);
        assert_eq!(sum(v.iter().copied()), 1000.0);
        assert_eq!(v.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn two_prod_is_exact() {
        let (p, e) = two_prod(1.0 + 2.0f64.powi(-30), 1.0 + 2.0f64.powi(-30));
        assert_eq!(p + e, p + e);
        assert_eq!(e, 2.0f64.powi(-60));
    }

    #[test]
    fn small_samples() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&x).unwrap(), 5.0);
        assert!((variance(&x).unwrap() - 32.0 / 7.0).abs() < 1e-15);
        assert!(mean(&[]).is_err() && variance(&[1.0]).is_err());
        assert!(mean(&[1.0, f64::NAN]).is_err());
        assert!(autocorrelation(&[1.0, 1.0, 1.0], 1).is_err());
        // Alternating series has lag-1 autocorrelation −(n−1)/n.
        let alt = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert!((autocorrelation(&alt, 1).unwrap() + 5.0 / 6.0).abs() < 1e-15);
    }
}
