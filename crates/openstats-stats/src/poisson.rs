//! One- and two-sample Poisson rate tests.
//!
//! Counts are totals over an *exposure* (number of units × length of each
//! unit, or time). Exact methods follow Garwood (intervals) and the
//! conditional binomial test (two samples).

use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::hypothesis::{check_level, wald_interval, z_p_value, Alternative};
use crate::proportion::{binomial_p, clopper_pearson};
use openstats_math::dist::{ChiSquared, ContinuousDistribution, DiscreteDistribution, Poisson};

/// Total count and exposure from per-unit counts, each observed over
/// `length` units of exposure (non-negative whole numbers; missing dropped).
pub fn poisson_totals(counts: &[f64], length: f64) -> Result<(u64, f64)> {
    if !(length > 0.0 && length.is_finite()) {
        return invalid(format!("the exposure per observation {length} must be positive"));
    }
    let v = crate::clean(counts)?;
    if v.is_empty() {
        return Err(StatsError::InsufficientData { what: "Poisson rate", needed: 1, got: 0 });
    }
    if v.iter().any(|&c| c < 0.0 || c.fract() != 0.0) {
        return invalid("Poisson counts must be non-negative whole numbers");
    }
    Ok((v.iter().sum::<f64>() as u64, v.len() as f64 * length))
}

pub(crate) fn poisson_two_sided(mu: f64, x: u64) -> Result<f64> {
    let d = Poisson::new(mu)?;
    let t = d.pmf(x as f64) * (1.0 + 1e-7);
    let mode = mu.floor() as u64;
    if x == mode {
        return Ok(1.0);
    }
    let total = if x < mode {
        // pmf non-increasing on [mode, ∞): smallest k0 with pmf(k0) <= t
        let mut hi = mode + 1;
        while d.pmf(hi as f64) > t {
            hi = hi.saturating_mul(2);
            if hi > u64::MAX / 4 {
                break;
            }
        }
        let (mut lo, mut hi) = (mode, hi);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if d.pmf(mid as f64) <= t {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        d.cdf(x as f64) + d.sf(lo as f64 - 1.0)
    } else {
        let (mut lo, mut hi) = (0u64, mode + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if d.pmf(mid as f64) <= t {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (if lo > 0 { d.cdf(lo as f64 - 1.0) } else { 0.0 }) + d.sf(x as f64 - 1.0)
    };
    Ok(total.min(1.0))
}

/// Exact Poisson p-value of count `x` under mean `mu` for the alternative.
pub(crate) fn poisson_p(mu: f64, x: u64, alt: Alternative) -> Result<f64> {
    let d = Poisson::new(mu)?;
    Ok(match alt {
        Alternative::TwoSided => poisson_two_sided(mu, x)?,
        Alternative::Less => d.cdf(x as f64),
        Alternative::Greater => {
            if x == 0 {
                1.0
            } else {
                d.sf(x as f64 - 1.0)
            }
        }
    })
}

/// Result of a one-sample Poisson rate test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePoisson {
    pub total: u64,
    pub exposure: f64,
    /// Estimated rate, total/exposure.
    pub rate: f64,
    pub rate0: f64,
    /// Exact p-value from Poisson(rate₀·exposure) (two-sided by small p-values).
    pub p_value_exact: f64,
    /// z = (rate − rate₀)/√(rate₀/exposure).
    pub z: f64,
    pub p_value_normal: f64,
    /// Garwood (χ²) interval.
    pub ci_exact: Interval,
    /// rate ± z√(rate/exposure).
    pub ci_normal: Interval,
}

/// One-sample Poisson rate test of H₀: λ = `rate0`.
pub fn one_sample_poisson(total: u64, exposure: f64, rate0: f64, alt: Alternative, level: f64) -> Result<OnePoisson> {
    check_level(level)?;
    if !(exposure > 0.0 && exposure.is_finite()) {
        return invalid(format!("the exposure {exposure} must be positive"));
    }
    if !(rate0 > 0.0 && rate0.is_finite()) {
        return invalid(format!("the hypothesised rate {rate0} must be positive"));
    }
    let rate = total as f64 / exposure;
    let mu0 = rate0 * exposure;
    let d = Poisson::new(mu0)?;
    let p_exact = match alt {
        Alternative::TwoSided => poisson_two_sided(mu0, total)?,
        Alternative::Less => d.cdf(total as f64),
        Alternative::Greater => {
            if total == 0 {
                1.0
            } else {
                d.sf(total as f64 - 1.0)
            }
        }
    };
    let tail = match alt {
        Alternative::TwoSided => (1.0 - level) / 2.0,
        _ => 1.0 - level,
    };
    let lo = if total == 0 { 0.0 } else { ChiSquared::new(2.0 * total as f64)?.quantile(tail) / (2.0 * exposure) };
    let hi = ChiSquared::new(2.0 * total as f64 + 2.0)?.quantile(1.0 - tail) / (2.0 * exposure);
    let ci_exact = match alt {
        Alternative::TwoSided => Interval { lower: lo, upper: hi, level },
        Alternative::Less => Interval { lower: 0.0, upper: hi, level },
        Alternative::Greater => Interval { lower: lo, upper: f64::INFINITY, level },
    };
    let z = (rate - rate0) / (rate0 / exposure).sqrt();
    Ok(OnePoisson {
        total,
        exposure,
        rate,
        rate0,
        p_value_exact: p_exact,
        z,
        p_value_normal: z_p_value(z, alt),
        ci_exact,
        ci_normal: wald_interval(rate, (rate / exposure).sqrt(), f64::INFINITY, alt, level)?,
    })
}

/// Result of a two-sample Poisson rate comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPoisson {
    pub x1: u64,
    pub t1: f64,
    pub x2: u64,
    pub t2: f64,
    pub rate1: f64,
    pub rate2: f64,
    /// rate₁ − rate₂.
    pub diff: f64,
    pub diff0: f64,
    /// z for the difference (pooled variance when `diff0 = 0`, else unpooled).
    pub z: f64,
    pub p_value_normal: f64,
    /// Exact conditional test of equal rates (`diff0 = 0` only): X₁ given
    /// X₁ + X₂ is Binomial(x₁ + x₂, t₁/(t₁ + t₂)).
    pub p_value_exact: Option<f64>,
    /// Wald interval for rate₁ − rate₂ (unpooled).
    pub ci_diff: Interval,
    /// Exact interval for the rate ratio rate₁/rate₂ (from Clopper–Pearson on
    /// the conditional binomial); `None` when x₁ + x₂ = 0.
    pub ci_ratio_exact: Option<Interval>,
}

/// Two-sample Poisson test of H₀: λ₁ − λ₂ = `diff0`.
pub fn two_sample_poisson(
    x1: u64,
    t1: f64,
    x2: u64,
    t2: f64,
    diff0: f64,
    alt: Alternative,
    level: f64,
) -> Result<TwoPoisson> {
    check_level(level)?;
    if !(t1 > 0.0 && t1.is_finite() && t2 > 0.0 && t2.is_finite()) {
        return invalid("exposures must be positive");
    }
    let (r1, r2) = (x1 as f64 / t1, x2 as f64 / t2);
    let diff = r1 - r2;
    let var_test = if diff0 == 0.0 { (x1 + x2) as f64 / (t1 + t2) * (1.0 / t1 + 1.0 / t2) } else { r1 / t1 + r2 / t2 };
    let var_ci = r1 / t1 + r2 / t2;
    if var_test <= 0.0 {
        if diff == diff0 {
            return Ok(TwoPoisson {
                x1,
                t1,
                x2,
                t2,
                rate1: r1,
                rate2: r2,
                diff,
                diff0,
                z: 0.0,
                p_value_normal: if alt == Alternative::TwoSided { 1.0 } else { 0.5 },
                p_value_exact: (diff0 == 0.0).then_some(1.0),
                ci_diff: Interval { lower: diff, upper: diff, level },
                ci_ratio_exact: None,
            });
        }
        return invalid("both counts are zero, so the test statistic is undefined");
    }
    let z = (diff - diff0) / var_test.sqrt();
    let n = x1 + x2;
    let p0 = t1 / (t1 + t2);
    let p_exact = if diff0 == 0.0 { Some(if n == 0 { 1.0 } else { binomial_p(n, p0, x1, alt)? }) } else { None };
    let ci_ratio = if n == 0 {
        None
    } else {
        let cp = clopper_pearson(x1, n, alt, level)?;
        let to_ratio = |p: f64| if p >= 1.0 { f64::INFINITY } else { p / (1.0 - p) * (t2 / t1) };
        Some(Interval { lower: to_ratio(cp.lower), upper: to_ratio(cp.upper), level })
    };
    Ok(TwoPoisson {
        x1,
        t1,
        x2,
        t2,
        rate1: r1,
        rate2: r2,
        diff,
        diff0,
        z,
        p_value_normal: z_p_value(z, alt),
        p_value_exact: p_exact,
        ci_diff: wald_interval(diff, var_ci.sqrt(), f64::INFINITY, alt, level)?,
        ci_ratio_exact: ci_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use Alternative::*;

    #[test]
    fn totals() {
        assert_eq!(poisson_totals(&[2.0, 3.0, f64::NAN, 5.0], 2.0).unwrap(), (10, 6.0));
        assert!(poisson_totals(&[1.5], 1.0).is_err());
        assert!(poisson_totals(&[-1.0], 1.0).is_err());
        assert!(poisson_totals(&[], 1.0).is_err());
        assert!(poisson_totals(&[1.0], 0.0).is_err());
    }

    #[test]
    fn exact_matches_brute_force() {
        let mu = 4.5;
        let d = Poisson::new(mu).unwrap();
        for x in [0u64, 1, 2, 4, 5, 9, 12] {
            let t = d.pmf(x as f64) * (1.0 + 1e-7);
            let brute: f64 = (0..400).map(|k| d.pmf(f64::from(k))).filter(|&p| p <= t).sum();
            assert!((poisson_two_sided(mu, x).unwrap() - brute.min(1.0)).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn one_sample_features() {
        let r = one_sample_poisson(12, 4.0, 2.0, TwoSided, 0.95).unwrap();
        assert_eq!(r.rate, 3.0);
        assert!((r.z - 1.0 / (0.5f64).sqrt()).abs() < 1e-12);
        assert!(r.ci_exact.lower < 3.0 && r.ci_exact.upper > 3.0);
        // Garwood 95% for x = 12: [6.2, 21.0]/4
        assert!((r.ci_exact.lower - 6.200575 / 4.0).abs() < 1e-3 && (r.ci_exact.upper - 20.962 / 4.0).abs() < 1e-3);
        let z = one_sample_poisson(0, 3.0, 1.0, Greater, 0.9).unwrap();
        assert_eq!(z.p_value_exact, 1.0);
        let z2 = one_sample_poisson(0, 3.0, 1.0, TwoSided, 0.9).unwrap();
        assert_eq!(z2.ci_exact.lower, 0.0);
        assert!(one_sample_poisson(1, 0.0, 1.0, TwoSided, 0.95).is_err());
        assert!(one_sample_poisson(1, 1.0, 0.0, TwoSided, 0.95).is_err());
    }

    #[test]
    fn two_sample_features() {
        let r = two_sample_poisson(20, 10.0, 35, 12.0, 0.0, TwoSided, 0.95).unwrap();
        assert!((r.diff - (2.0 - 35.0 / 12.0)).abs() < 1e-14);
        assert!(r.p_value_exact.is_some());
        let ci = r.ci_ratio_exact.unwrap();
        assert!(ci.lower < r.rate1 / r.rate2 && ci.upper > r.rate1 / r.rate2);
        assert!(two_sample_poisson(20, 10.0, 35, 12.0, 0.1, TwoSided, 0.95).unwrap().p_value_exact.is_none());
        let z = two_sample_poisson(0, 1.0, 0, 1.0, 0.0, TwoSided, 0.95).unwrap();
        assert_eq!((z.z, z.p_value_exact), (0.0, Some(1.0)));
        assert!(two_sample_poisson(0, 1.0, 0, 1.0, 0.5, TwoSided, 0.95).is_err());
        assert!(two_sample_poisson(1, -1.0, 1, 1.0, 0.0, TwoSided, 0.95).is_err());
    }
}
