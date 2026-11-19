//! Reliability demonstration test plans (Weibull with a known shape; shape 1 is exponential).
//!
//! A plan tests `n` units for `T` time units and accepts when at most `c` fail. It demonstrates
//! `R(t) ≥ R*` with confidence `C` when `P(X ≤ c | p*) ≤ 1 − C`, with `X ~ Binomial(n, p*)` and
//! `p* = 1 − R*^((T/t)^β)`.

use crate::error::{invalid, Result};
use openstats_math::dist::{Binomial, DiscreteDistribution};

/// A demonstration test plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DemoPlan {
    /// Weibull shape β.
    pub shape: f64,
    /// Requirement time `t`.
    pub req_time: f64,
    /// Reliability to demonstrate at `t`.
    pub reliability: f64,
    /// Target confidence.
    pub confidence: f64,
    /// Failures allowed.
    pub failures_allowed: u64,
    /// Units on test.
    pub n: u64,
    /// Test duration per unit.
    pub test_time: f64,
    /// Confidence actually achieved (`≥ confidence` for the integer `n`).
    pub achieved_confidence: f64,
}

fn check(shape: f64, t: f64, r: f64, c: f64) -> Result<()> {
    if !(shape > 0.0) || !shape.is_finite() {
        return invalid("the Weibull shape must be positive");
    }
    if !(t > 0.0) || !t.is_finite() {
        return invalid("the requirement time must be positive");
    }
    if !(r > 0.0 && r < 1.0) {
        return invalid("the reliability to demonstrate must be between 0 and 1");
    }
    if !(c > 0.0 && c < 1.0) {
        return invalid("the confidence must be between 0 and 1");
    }
    Ok(())
}

/// Failure probability over the test for a unit with `R(t) = r`.
pub fn failure_probability(shape: f64, req_time: f64, r: f64, test_time: f64) -> f64 {
    1.0 - r.powf((test_time / req_time).powf(shape))
}

/// Probability of passing the test (≤ `c` failures) when the true reliability at `t` is `r_true`.
pub fn acceptance_probability(plan: &DemoPlan, r_true: f64) -> Result<f64> {
    if !(r_true > 0.0 && r_true < 1.0) {
        return invalid("reliability must be between 0 and 1");
    }
    let p = failure_probability(plan.shape, plan.req_time, r_true, plan.test_time);
    Ok(Binomial::new(plan.n, p.clamp(0.0, 1.0))?.cdf(plan.failures_allowed as f64))
}

/// The failure probability `p*` at which `P(X ≤ c) = 1 − C` for `n` units.
fn p_star(n: u64, c: u64, confidence: f64) -> Result<f64> {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let f = Binomial::new(n, mid)?.cdf(c as f64);
        if f > 1.0 - confidence {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Sample size for a given test time and number of allowed failures.
pub fn plan_sample_size(
    shape: f64,
    req_time: f64,
    reliability: f64,
    confidence: f64,
    failures_allowed: u64,
    test_time: f64,
) -> Result<DemoPlan> {
    check(shape, req_time, reliability, confidence)?;
    if !(test_time > 0.0) || !test_time.is_finite() {
        return invalid("the test time must be positive");
    }
    let p = failure_probability(shape, req_time, reliability, test_time);
    if !(p > 0.0 && p < 1.0) {
        return invalid("this test time gives a degenerate failure probability");
    }
    let ok = |n: u64| Binomial::new(n, p).map(|b| b.cdf(failures_allowed as f64) <= 1.0 - confidence);
    let mut hi = failures_allowed + 1;
    while !ok(hi)? {
        hi = hi.saturating_mul(2);
        if hi > 100_000_000 {
            return invalid("the required sample size exceeds 100,000,000; lengthen the test");
        }
    }
    let mut lo = failures_allowed; // not feasible with n ≤ c
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if ok(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let achieved = 1.0 - Binomial::new(hi, p)?.cdf(failures_allowed as f64);
    Ok(DemoPlan {
        shape,
        req_time,
        reliability,
        confidence,
        failures_allowed,
        n: hi,
        test_time,
        achieved_confidence: achieved,
    })
}

/// Test time for a given sample size and number of allowed failures.
pub fn plan_test_time(
    shape: f64,
    req_time: f64,
    reliability: f64,
    confidence: f64,
    failures_allowed: u64,
    n: u64,
) -> Result<DemoPlan> {
    check(shape, req_time, reliability, confidence)?;
    if n <= failures_allowed {
        return invalid("the sample size must exceed the number of allowed failures");
    }
    let ps = p_star(n, failures_allowed, confidence)?;
    let ratio = (1.0 - ps).ln() / reliability.ln();
    let t = req_time * ratio.powf(1.0 / shape);
    Ok(DemoPlan {
        shape,
        req_time,
        reliability,
        confidence,
        failures_allowed,
        n,
        test_time: t,
        achieved_confidence: confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_failure_exponential_classic() {
        // R=0.9 at t, 90% confidence, test time = t, c=0, shape 1: n = ln(.1)/ln(.9) = 21.85 -> 22
        let p = plan_sample_size(1.0, 100.0, 0.9, 0.9, 0, 100.0).unwrap();
        assert_eq!(p.n, 22);
        assert!(p.achieved_confidence >= 0.9);
    }

    #[test]
    fn shape_and_time_trade_off() {
        // doubling test time with shape 2 multiplies the failure exponent by 4
        let a = plan_sample_size(2.0, 100.0, 0.9, 0.9, 0, 200.0).unwrap();
        assert_eq!(a.n, 6); // ln(.1)/(4 ln .9) = 5.46 -> 6
    }

    #[test]
    fn test_time_inverts_sample_size() {
        let p = plan_test_time(2.0, 100.0, 0.9, 0.9, 1, 30).unwrap();
        let q = plan_sample_size(2.0, 100.0, 0.9, 0.9, 1, p.test_time * 1.0001).unwrap();
        assert!(q.n <= 30);
        let r = plan_sample_size(2.0, 100.0, 0.9, 0.9, 1, p.test_time * 0.98).unwrap();
        assert!(r.n >= 30);
    }

    #[test]
    fn acceptance_probability_is_monotone() {
        let p = plan_sample_size(1.0, 100.0, 0.9, 0.9, 1, 100.0).unwrap();
        let lo = acceptance_probability(&p, 0.85).unwrap();
        let hi = acceptance_probability(&p, 0.99).unwrap();
        assert!(hi > lo);
        let at_req = acceptance_probability(&p, 0.9).unwrap();
        assert!(at_req <= 1.0 - p.confidence + 1e-12);
    }

    #[test]
    fn errors() {
        assert!(plan_sample_size(0.0, 1.0, 0.9, 0.9, 0, 1.0).is_err());
        assert!(plan_sample_size(1.0, 1.0, 1.0, 0.9, 0, 1.0).is_err());
        assert!(plan_test_time(1.0, 1.0, 0.9, 0.9, 3, 3).is_err());
    }
}
