//! t-tests and the one-sample Z-test.
//!
//! All functions drop missing values (`NaN`) and report ±∞ as an error. Paired
//! tests use only the pairs where both values are present.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::hypothesis::{check_level, t_p_value, wald_interval, z_p_value, Alternative, VarianceAssumption};
use openstats_math::stats::summary;

/// Result of a t-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TTest {
    /// Sample sizes (the second is `None` for one-sample and paired tests).
    pub n1: usize,
    pub n2: Option<usize>,
    pub mean1: f64,
    pub sd1: f64,
    pub mean2: Option<f64>,
    pub sd2: Option<f64>,
    /// Estimated mean (one-sample), mean difference (paired: mean of x − y;
    /// two-sample: mean₁ − mean₂).
    pub estimate: f64,
    pub se: f64,
    /// Null value of the estimated quantity.
    pub null_value: f64,
    pub t: f64,
    /// Degrees of freedom (non-integer for Welch).
    pub df: f64,
    pub p_value: f64,
    /// Confidence interval for the estimated quantity (one-sided for
    /// `Less`/`Greater`).
    pub ci: Interval,
    /// Pooled SD (pooled two-sample test only).
    pub pooled_sd: Option<f64>,
}

fn sample(x: &[f64], what: &'static str) -> Result<(Vec<f64>, f64, f64)> {
    let v = crate::clean(x)?;
    if v.len() < 2 {
        return Err(StatsError::InsufficientData { what, needed: 2, got: v.len() });
    }
    let m = summary::mean(&v)?;
    let sd = summary::std_dev(&v)?;
    Ok((v, m, sd))
}

fn finish(est: f64, se: f64, null: f64, df: f64, alt: Alternative, level: f64) -> Result<(f64, f64, Interval)> {
    if !(se > 0.0 && se.is_finite()) {
        return invalid("the standard error is zero (no variation), so the t statistic is undefined");
    }
    let t = (est - null) / se;
    Ok((t, t_p_value(t, df, alt)?, wald_interval(est, se, df, alt, level)?))
}

/// One-sample t-test of H₀: μ = `mu0`.
pub fn one_sample_t(x: &[f64], mu0: f64, alt: Alternative, level: f64) -> Result<TTest> {
    check_level(level)?;
    let (v, mean, sd) = sample(x, "one-sample t-test")?;
    one_sample_t_summary(v.len(), mean, sd, mu0, alt, level)
}

/// One-sample t-test from summarized data (sample size, mean and standard deviation).
pub fn one_sample_t_summary(n: usize, mean: f64, sd: f64, mu0: f64, alt: Alternative, level: f64) -> Result<TTest> {
    check_level(level)?;
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "one-sample t-test", needed: 2, got: n });
    }
    if !(mean.is_finite() && sd.is_finite() && sd >= 0.0) {
        return invalid("the mean and standard deviation must be finite and the standard deviation non-negative");
    }
    let se = sd / (n as f64).sqrt();
    let df = (n - 1) as f64;
    let (t, p, ci) = finish(mean, se, mu0, df, alt, level)?;
    Ok(TTest {
        n1: n,
        n2: None,
        mean1: mean,
        sd1: sd,
        mean2: None,
        sd2: None,
        estimate: mean,
        se,
        null_value: mu0,
        t,
        df,
        p_value: p,
        ci,
        pooled_sd: None,
    })
}

/// Paired t-test from summarized data: the sample size, mean and standard deviation of the paired differences.
pub fn paired_t_summary(
    n: usize,
    mean_diff: f64,
    sd_diff: f64,
    mu0: f64,
    alt: Alternative,
    level: f64,
) -> Result<TTest> {
    one_sample_t_summary(n, mean_diff, sd_diff, mu0, alt, level)
}

/// Paired t-test of H₀: mean(x − y) = `mu0`, on complete pairs.
pub fn paired_t(x: &[f64], y: &[f64], mu0: f64, alt: Alternative, level: f64) -> Result<TTest> {
    check_level(level)?;
    if x.len() != y.len() {
        return invalid(format!("paired samples have different lengths ({} and {})", x.len(), y.len()));
    }
    if x.iter().chain(y).any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let d: Vec<f64> = x.iter().zip(y).filter(|(a, b)| !a.is_nan() && !b.is_nan()).map(|(a, b)| a - b).collect();
    let mut r = one_sample_t(&d, mu0, alt, level)?;
    // Report the moments of the paired samples, not of the differences.
    let (px, py): (Vec<f64>, Vec<f64>) =
        x.iter().zip(y).filter(|(a, b)| !a.is_nan() && !b.is_nan()).map(|(a, b)| (*a, *b)).unzip();
    r.mean1 = summary::mean(&px)?;
    r.sd1 = summary::std_dev(&px)?;
    r.mean2 = Some(summary::mean(&py)?);
    r.sd2 = Some(summary::std_dev(&py)?);
    r.n2 = Some(px.len());
    Ok(r)
}

/// Two-sample t-test of H₀: μ₁ − μ₂ = `diff0`.
pub fn two_sample_t(
    x: &[f64],
    y: &[f64],
    diff0: f64,
    var: VarianceAssumption,
    alt: Alternative,
    level: f64,
) -> Result<TTest> {
    check_level(level)?;
    let (vx, m1, s1) = sample(x, "two-sample t-test (first sample)")?;
    let (vy, m2, s2) = sample(y, "two-sample t-test (second sample)")?;
    two_sample_t_summary((vx.len(), m1, s1), (vy.len(), m2, s2), diff0, var, alt, level)
}

/// Two-sample t-test from summarized data: `(n, mean, standard deviation)` of each sample.
pub fn two_sample_t_summary(
    a: (usize, f64, f64),
    b: (usize, f64, f64),
    diff0: f64,
    var: VarianceAssumption,
    alt: Alternative,
    level: f64,
) -> Result<TTest> {
    check_level(level)?;
    let ((n1u, m1, s1), (n2u, m2, s2)) = (a, b);
    if n1u < 2 || n2u < 2 {
        return Err(StatsError::InsufficientData { what: "two-sample t-test", needed: 2, got: n1u.min(n2u) });
    }
    if ![m1, s1, m2, s2].iter().all(|v| v.is_finite()) || s1 < 0.0 || s2 < 0.0 {
        return invalid("the means and standard deviations must be finite and the standard deviations non-negative");
    }
    let (n1, n2) = (n1u as f64, n2u as f64);
    let (a, b) = (s1 * s1 / n1, s2 * s2 / n2);
    let (se, df, pooled) = match var {
        VarianceAssumption::Welch => {
            ((a + b).sqrt(), (a + b) * (a + b) / (a * a / (n1 - 1.0) + b * b / (n2 - 1.0)), None)
        }
        VarianceAssumption::Pooled => {
            let sp2 = ((n1 - 1.0) * s1 * s1 + (n2 - 1.0) * s2 * s2) / (n1 + n2 - 2.0);
            ((sp2 * (1.0 / n1 + 1.0 / n2)).sqrt(), n1 + n2 - 2.0, Some(sp2.sqrt()))
        }
    };
    let est = m1 - m2;
    let (t, p, ci) = finish(est, se, diff0, df, alt, level)?;
    Ok(TTest {
        n1: n1u,
        n2: Some(n2u),
        mean1: m1,
        sd1: s1,
        mean2: Some(m2),
        sd2: Some(s2),
        estimate: est,
        se,
        null_value: diff0,
        t,
        df,
        p_value: p,
        ci,
        pooled_sd: pooled,
    })
}

/// Result of a one-sample Z-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZTest {
    pub n: usize,
    pub mean: f64,
    /// Known population standard deviation used.
    pub sigma: f64,
    pub se: f64,
    pub z: f64,
    pub p_value: f64,
    pub ci: Interval,
}

/// One-sample Z-test of H₀: μ = `mu0` with known σ.
pub fn z_test(x: &[f64], mu0: f64, sigma: f64, alt: Alternative, level: f64) -> Result<ZTest> {
    check_level(level)?;
    let v = crate::clean(x)?;
    if v.is_empty() {
        return Err(StatsError::InsufficientData { what: "Z-test", needed: 1, got: 0 });
    }
    z_test_summary(v.len(), summary::mean(&v)?, mu0, sigma, alt, level)
}

/// One-sample Z-test from summarized data (sample size and mean) with known σ.
pub fn z_test_summary(n: usize, mean: f64, mu0: f64, sigma: f64, alt: Alternative, level: f64) -> Result<ZTest> {
    check_level(level)?;
    if !(sigma > 0.0 && sigma.is_finite()) {
        return invalid(format!("the known standard deviation {sigma} must be positive"));
    }
    if n == 0 || !mean.is_finite() {
        return invalid("a Z-test needs a positive sample size and a finite mean");
    }
    let se = sigma / (n as f64).sqrt();
    let z = (mean - mu0) / se;
    Ok(ZTest {
        n,
        mean,
        sigma,
        se,
        z,
        p_value: z_p_value(z, alt),
        ci: wald_interval(mean, se, f64::INFINITY, alt, level)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use Alternative::*;

    const A: [f64; 6] = [5.1, 4.9, 5.6, 5.8, 6.0, 5.3];
    const B: [f64; 7] = [4.2, 4.8, 5.0, 4.4, 4.9, 5.2, 4.6];

    #[test]
    fn one_sample_by_hand() {
        let r = one_sample_t(&[1.0, 2.0, 3.0, 4.0, f64::NAN], 2.0, TwoSided, 0.95).unwrap();
        assert_eq!(r.n1, 4);
        assert!((r.t - 0.5 / (1.2909944487358056 / 2.0)).abs() < 1e-12);
        assert_eq!(r.df, 3.0);
        assert!(r.ci.lower < 2.5 && r.ci.upper > 2.5);
        let g = one_sample_t(&[1.0, 2.0, 3.0, 4.0], 2.0, Greater, 0.95).unwrap();
        assert_eq!(g.ci.upper, f64::INFINITY);
        assert!((g.p_value - r.p_value / 2.0).abs() < 1e-12);
        let l = one_sample_t(&[1.0, 2.0, 3.0, 4.0], 2.0, Less, 0.95).unwrap();
        assert!((l.p_value + g.p_value - 1.0).abs() < 1e-12);
        assert_eq!(l.ci.lower, f64::NEG_INFINITY);
    }

    #[test]
    fn errors() {
        assert!(one_sample_t(&[1.0], 0.0, TwoSided, 0.95).is_err());
        assert!(one_sample_t(&[2.0, 2.0, 2.0], 0.0, TwoSided, 0.95).is_err());
        assert!(one_sample_t(&[1.0, 2.0], 0.0, TwoSided, 1.0).is_err());
        assert!(paired_t(&[1.0, 2.0], &[1.0], 0.0, TwoSided, 0.95).is_err());
        assert!(two_sample_t(&A, &[1.0], 0.0, VarianceAssumption::Welch, TwoSided, 0.95).is_err());
        assert!(one_sample_t(&[1.0, f64::INFINITY], 0.0, TwoSided, 0.95).is_err());
    }

    #[test]
    fn two_sample_variants() {
        let w = two_sample_t(&A, &B, 0.0, VarianceAssumption::Welch, TwoSided, 0.95).unwrap();
        let p = two_sample_t(&A, &B, 0.0, VarianceAssumption::Pooled, TwoSided, 0.95).unwrap();
        assert_eq!(p.df, 11.0);
        assert!(w.df > 5.0 && w.df < 11.0 && w.df.fract() != 0.0);
        assert!((w.estimate - p.estimate).abs() < 1e-15);
        assert!(p.pooled_sd.is_some() && w.pooled_sd.is_none());
        // diff0 shifts t
        let s = two_sample_t(&A, &B, w.estimate, VarianceAssumption::Welch, TwoSided, 0.95).unwrap();
        assert!(s.t.abs() < 1e-12 && (s.p_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn paired_uses_complete_pairs() {
        let x = [10.0, 12.0, f64::NAN, 15.0, 11.0];
        let y = [9.0, 10.5, 8.0, 13.0, f64::NAN];
        let r = paired_t(&x, &y, 0.0, TwoSided, 0.95).unwrap();
        assert_eq!(r.n1, 3);
        assert!((r.estimate - (1.0 + 1.5 + 2.0) / 3.0).abs() < 1e-14);
        assert!((r.mean2.unwrap() - 32.5 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn z() {
        let r = z_test(&[10.0, 11.0, 12.0, 13.0], 10.0, 2.0, TwoSided, 0.95).unwrap();
        assert!((r.z - 1.5).abs() < 1e-14);
        assert!((r.p_value - 0.13361440253771617).abs() < 1e-12);
        assert!((r.ci.lower - (11.5 - 1.959963984540054)).abs() < 1e-10);
        assert!(z_test(&[], 0.0, 1.0, TwoSided, 0.95).is_err());
        assert!(z_test(&[1.0], 0.0, 0.0, TwoSided, 0.95).is_err());
    }
}
