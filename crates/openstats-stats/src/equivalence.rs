//! Equivalence testing by two one-sided tests (TOST, Schuirmann 1987).
//!
//! Equivalence is concluded at level α when the difference is shown to lie
//! inside (`lower`, `upper`): both H₀: Δ ≤ lower and H₀: Δ ≥ upper are
//! rejected, i.e. the larger of the two one-sided p-values is below α. This is
//! equivalent to the (1 − 2α) confidence interval for Δ lying inside the
//! limits, which is reported.

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::hypothesis::{check_alpha, VarianceAssumption};
use openstats_math::dist::{ContinuousDistribution, StudentT};
use openstats_math::stats::summary;

/// Result of a TOST equivalence test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tost {
    /// Estimated difference Δ (mean − target; mean of differences; or
    /// mean₁ − mean₂).
    pub estimate: f64,
    pub se: f64,
    pub df: f64,
    pub lower: f64,
    pub upper: f64,
    /// t = (Δ̂ − lower)/se and its p-value P(T > t), for H₀: Δ ≤ lower.
    pub t_lower: f64,
    pub p_lower: f64,
    /// t = (Δ̂ − upper)/se and its p-value P(T < t), for H₀: Δ ≥ upper.
    pub t_upper: f64,
    pub p_upper: f64,
    /// max(p_lower, p_upper): the TOST p-value.
    pub p_value: f64,
    /// (1 − 2α) two-sided confidence interval for Δ.
    pub ci: Interval,
    pub alpha: f64,
    /// `p_value < alpha`.
    pub equivalent: bool,
}

fn tost_from(estimate: f64, se: f64, df: f64, lower: f64, upper: f64, alpha: f64) -> Result<Tost> {
    if lower >= upper || lower.is_nan() || upper.is_nan() || !lower.is_finite() || !upper.is_finite() {
        return invalid(format!("equivalence limits must satisfy lower < upper, got ({lower}, {upper})"));
    }
    if !(se > 0.0 && se.is_finite()) {
        return invalid("the standard error is zero (no variation), so the equivalence test is undefined");
    }
    let d = StudentT::new(df)?;
    let (t_lower, t_upper) = ((estimate - lower) / se, (estimate - upper) / se);
    let (p_lower, p_upper) = (d.sf(t_lower), d.cdf(t_upper));
    let q = d.isf(alpha);
    let p = p_lower.max(p_upper);
    Ok(Tost {
        estimate,
        se,
        df,
        lower,
        upper,
        t_lower,
        p_lower,
        t_upper,
        p_upper,
        p_value: p,
        ci: Interval { lower: estimate - q * se, upper: estimate + q * se, level: 1.0 - 2.0 * alpha },
        alpha,
        equivalent: p < alpha,
    })
}

/// One-sample TOST: is the mean within (`lower`, `upper`) *of `target`*,
/// i.e. is lower < μ − target < upper? 0 < α < ½.
pub fn tost_one_sample(x: &[f64], target: f64, lower: f64, upper: f64, alpha: f64) -> Result<Tost> {
    check_alpha(alpha)?;
    let v = crate::clean(x)?;
    if v.len() < 2 {
        return Err(StatsError::InsufficientData { what: "equivalence test", needed: 2, got: v.len() });
    }
    let n = v.len() as f64;
    let se = summary::std_dev(&v)? / n.sqrt();
    tost_from(summary::mean(&v)? - target, se, n - 1.0, lower, upper, alpha)
}

/// Paired TOST on the complete pairs: is lower < mean(x − y) < upper?
pub fn tost_paired(x: &[f64], y: &[f64], lower: f64, upper: f64, alpha: f64) -> Result<Tost> {
    if x.len() != y.len() {
        return invalid(format!("paired samples have different lengths ({} and {})", x.len(), y.len()));
    }
    if x.iter().chain(y).any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let d: Vec<f64> = x.iter().zip(y).filter(|(a, b)| !a.is_nan() && !b.is_nan()).map(|(a, b)| a - b).collect();
    tost_one_sample(&d, 0.0, lower, upper, alpha)
}

/// Two-sample TOST: is lower < μ₁ − μ₂ < upper?
pub fn tost_two_sample(
    x: &[f64],
    y: &[f64],
    lower: f64,
    upper: f64,
    var: VarianceAssumption,
    alpha: f64,
) -> Result<Tost> {
    check_alpha(alpha)?;
    let (vx, vy) = (crate::clean(x)?, crate::clean(y)?);
    if vx.len() < 2 || vy.len() < 2 {
        return Err(StatsError::InsufficientData { what: "equivalence test", needed: 2, got: vx.len().min(vy.len()) });
    }
    let (n1, n2) = (vx.len() as f64, vy.len() as f64);
    let (s1, s2) = (summary::variance(&vx)?, summary::variance(&vy)?);
    let est = summary::mean(&vx)? - summary::mean(&vy)?;
    let (a, b) = (s1 / n1, s2 / n2);
    let (se, df) = match var {
        VarianceAssumption::Welch => ((a + b).sqrt(), (a + b) * (a + b) / (a * a / (n1 - 1.0) + b * b / (n2 - 1.0))),
        VarianceAssumption::Pooled => {
            let sp2 = ((n1 - 1.0) * s1 + (n2 - 1.0) * s2) / (n1 + n2 - 2.0);
            ((sp2 * (1.0 / n1 + 1.0 / n2)).sqrt(), n1 + n2 - 2.0)
        }
    };
    tost_from(est, se, df, lower, upper, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 8] = [10.1, 9.9, 10.2, 10.0, 9.8, 10.1, 10.05, 9.95];
    const Y: [f64; 8] = [10.0, 10.1, 10.15, 9.9, 9.85, 10.2, 10.0, 10.1];

    #[test]
    fn equivalence_iff_ci_inside_limits() {
        for (lo, hi) in [(-0.1, 0.1), (-0.5, 0.5), (-0.02, 0.02), (-0.05, 0.3)] {
            let r = tost_one_sample(&X, 10.0, lo, hi, 0.05).unwrap();
            let inside = r.ci.lower > lo && r.ci.upper < hi;
            assert_eq!(r.equivalent, inside, "({lo},{hi}) p={} ci={:?}", r.p_value, r.ci);
            assert_eq!(r.p_value, r.p_lower.max(r.p_upper));
            assert!((r.ci.level - 0.9).abs() < 1e-15);
        }
    }

    #[test]
    fn two_sample_and_paired() {
        for var in [VarianceAssumption::Welch, VarianceAssumption::Pooled] {
            let r = tost_two_sample(&X, &Y, -0.3, 0.3, var, 0.05).unwrap();
            assert!(r.equivalent);
            let n = tost_two_sample(&X, &Y, -0.01, 0.01, var, 0.05).unwrap();
            assert!(!n.equivalent && n.p_value > r.p_value);
        }
        let p = tost_paired(&X, &Y, -0.3, 0.3, 0.05).unwrap();
        assert!(p.equivalent);
        let m = tost_paired(&[1.0, f64::NAN, 3.0, 4.0], &[1.5, 2.0, 3.4, f64::NAN], -1.0, 1.0, 0.05).unwrap();
        assert_eq!(m.df, 1.0);
    }

    #[test]
    fn errors() {
        assert!(tost_one_sample(&X, 10.0, 0.1, -0.1, 0.05).is_err());
        assert!(tost_one_sample(&X, 10.0, -0.1, 0.1, 0.5).is_err());
        assert!(tost_one_sample(&X, 10.0, -0.1, 0.1, 0.0).is_err());
        assert!(tost_one_sample(&[1.0], 0.0, -1.0, 1.0, 0.05).is_err());
        assert!(tost_one_sample(&[2.0, 2.0, 2.0], 0.0, -1.0, 1.0, 0.05).is_err());
        assert!(tost_paired(&X, &Y[..3], -1.0, 1.0, 0.05).is_err());
        assert!(tost_two_sample(&X, &[1.0], -1.0, 1.0, VarianceAssumption::Welch, 0.05).is_err());
        assert!(tost_one_sample(&X, 10.0, f64::NEG_INFINITY, 0.1, 0.05).is_err());
    }
}
