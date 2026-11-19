//! One- and two-proportion tests.
//!
//! Exact two-sided binomial p-values use the "small p-values" definition
//! (probability of outcomes no more likely than the observed one, relative
//! tolerance 1e-7), as R's `binom.test` and Minitab.

use crate::crosstab::{fisher_exact, Crosstab};
use crate::error::{invalid, Result, StatsError};
use crate::graphical::Interval;
use crate::hypothesis::{check_level, wald_interval, z_p_value, Alternative};
use openstats_math::dist::{Beta, Binomial, ContinuousDistribution, DiscreteDistribution};

/// Two-sided exact binomial p-value by the small-p-values rule.
pub(crate) fn binomial_two_sided(n: u64, p: f64, x: u64) -> Result<f64> {
    let b = Binomial::new(n, p)?;
    let d = b.pmf(x as f64) * (1.0 + 1e-7);
    let mode = (((n + 1) as f64 * p).floor() as u64).min(n);
    if x == mode {
        return Ok(1.0);
    }
    let total = if x < mode {
        // pmf is non-increasing on [mode, n]: smallest k0 with pmf(k0) <= d
        let (mut lo, mut hi) = (mode, n + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if b.pmf(mid as f64) <= d {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        b.cdf(x as f64) + if lo <= n { b.sf(lo as f64 - 1.0) } else { 0.0 }
    } else {
        // pmf is non-decreasing on [0, mode]: largest k0 with pmf(k0) <= d
        let (mut lo, mut hi) = (0u64, mode + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if b.pmf(mid as f64) <= d {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo = number of k in [0, mode] with pmf <= d
        (if lo > 0 { b.cdf(lo as f64 - 1.0) } else { 0.0 }) + b.sf(x as f64 - 1.0)
    };
    Ok(total.min(1.0))
}

/// Exact binomial p-value for the given alternative.
pub(crate) fn binomial_p(n: u64, p: f64, x: u64, alt: Alternative) -> Result<f64> {
    let b = Binomial::new(n, p)?;
    Ok(match alt {
        Alternative::TwoSided => binomial_two_sided(n, p, x)?,
        Alternative::Less => b.cdf(x as f64),
        Alternative::Greater => {
            if x == 0 {
                1.0
            } else {
                b.sf(x as f64 - 1.0)
            }
        }
    })
}

/// Clopper–Pearson interval for a binomial proportion (one-sided for
/// `Less`/`Greater`).
pub fn clopper_pearson(x: u64, n: u64, alt: Alternative, level: f64) -> Result<Interval> {
    check_level(level)?;
    if n == 0 || x > n {
        return invalid(format!("{x} successes in {n} trials is not valid"));
    }
    let tail = match alt {
        Alternative::TwoSided => (1.0 - level) / 2.0,
        _ => 1.0 - level,
    };
    let lo =
        || -> Result<f64> { Ok(if x == 0 { 0.0 } else { Beta::new(x as f64, (n - x + 1) as f64)?.quantile(tail) }) };
    let hi = || -> Result<f64> {
        Ok(if x == n { 1.0 } else { Beta::new((x + 1) as f64, (n - x) as f64)?.quantile(1.0 - tail) })
    };
    Ok(match alt {
        Alternative::TwoSided => Interval { lower: lo()?, upper: hi()?, level },
        Alternative::Less => Interval { lower: 0.0, upper: hi()?, level },
        Alternative::Greater => Interval { lower: lo()?, upper: 1.0, level },
    })
}

/// Blaker's p-value for H₀: p = `p0` (two-sided): the probability, under `p0`, of outcomes whose acceptability
/// `min(P(X ≤ k), P(X ≥ k))` does not exceed the observed one.
fn blaker_p_value(x: u64, n: u64, p0: f64) -> Result<f64> {
    let b = Binomial::new(n, p0)?;
    // running sums give P(X <= k) and P(X >= k) for every k in one pass
    let pm: Vec<f64> = (0..=n).map(|k| b.pmf(k as f64)).collect();
    let mut cdf = vec![0.0; pm.len()];
    let mut acc = 0.0;
    for (k, p) in pm.iter().enumerate() {
        acc += p;
        cdf[k] = acc;
    }
    let total = acc;
    let sf = |k: usize| total - if k == 0 { 0.0 } else { cdf[k - 1] };
    let a = |k: usize| cdf[k].min(sf(k));
    let g = a(x as usize) * (1.0 + 1e-9);
    Ok((0..=n as usize).filter(|&k| a(k) <= g).map(|k| pm[k]).sum::<f64>().min(1.0))
}

/// Blaker's exact two-sided confidence interval for a binomial proportion (the interval Minitab prints for
/// 1 Proportion): all `p0` whose Blaker p-value exceeds `1 − level`. It is nested, always contained in the
/// Clopper–Pearson interval and never wider. Limited to `n ≤ 20 000` (larger samples: use the normal methods).
pub fn blaker_interval(x: u64, n: u64, level: f64) -> Result<Interval> {
    check_level(level)?;
    if n == 0 || x > n {
        return invalid(format!("{x} successes in {n} trials is not valid"));
    }
    if n > 20_000 {
        return invalid("Blaker's exact interval is limited to 20,000 trials");
    }
    let alpha = 1.0 - level;
    let phat = x as f64 / n as f64;
    let sd = (phat * (1.0 - phat)).max(1.0 / n as f64).sqrt() / (n as f64).sqrt();
    // walk outward from p-hat until the p-value drops to alpha, then bisect that bracket
    let edge = |dir: f64| -> Result<f64> {
        let limit = if dir < 0.0 { 0.0 } else { 1.0 };
        if (dir < 0.0 && x == 0) || (dir > 0.0 && x == n) {
            return Ok(limit);
        }
        let mut inside = phat.clamp(1e-12, 1.0 - 1e-12);
        let step = sd / 8.0;
        let mut outside = inside;
        loop {
            outside = (outside + dir * step).clamp(1e-12, 1.0 - 1e-12);
            if blaker_p_value(x, n, outside)? <= alpha || outside <= 1e-12 || outside >= 1.0 - 1e-12 {
                break;
            }
            inside = outside;
        }
        for _ in 0..60 {
            let mid = 0.5 * (inside + outside);
            if blaker_p_value(x, n, mid)? > alpha {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        Ok(inside)
    };
    let lower = if x == 0 { 0.0 } else { edge(-1.0)? };
    let upper = if x == n { 1.0 } else { edge(1.0)? };
    Ok(Interval { lower, upper, level })
}

/// Result of a one-proportion test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneProportion {
    pub x: u64,
    pub n: u64,
    pub p_hat: f64,
    pub p0: f64,
    /// z = (p̂ − p₀)/√(p₀(1−p₀)/n), with the continuity correction if requested.
    pub z: f64,
    pub p_value_normal: f64,
    pub p_value_exact: f64,
    /// Clopper–Pearson interval.
    pub ci_exact: Interval,
    /// Blaker exact interval (two-sided; for one-sided alternatives the Clopper–Pearson bound).
    pub ci_blaker: Interval,
    /// Wald interval p̂ ± z√(p̂(1−p̂)/n) (not clamped to [0, 1]).
    pub ci_normal: Interval,
    pub continuity_correction: bool,
}

/// One-proportion test of H₀: p = `p0` (0 < p₀ < 1), exact and normal.
///
/// The continuity correction moves |p̂ − p₀| toward 0 by min(0.5/n, |p̂ − p₀|).
pub fn one_proportion(
    x: u64,
    n: u64,
    p0: f64,
    alt: Alternative,
    level: f64,
    continuity_correction: bool,
) -> Result<OneProportion> {
    check_level(level)?;
    if n == 0 {
        return Err(StatsError::InsufficientData { what: "one-proportion test", needed: 1, got: 0 });
    }
    if x > n {
        return invalid(format!("{x} successes exceed the {n} trials"));
    }
    if !(p0 > 0.0 && p0 < 1.0) {
        return invalid(format!("the hypothesised proportion {p0} must be in (0, 1)"));
    }
    let nf = n as f64;
    let p_hat = x as f64 / nf;
    let se0 = (p0 * (1.0 - p0) / nf).sqrt();
    let diff = p_hat - p0;
    let adj = if continuity_correction { diff.signum() * (0.5 / nf).min(diff.abs()) } else { 0.0 };
    let z = (diff - adj) / se0;
    let se_hat = (p_hat * (1.0 - p_hat) / nf).sqrt();
    Ok(OneProportion {
        x,
        n,
        p_hat,
        p0,
        z,
        p_value_normal: z_p_value(z, alt),
        p_value_exact: binomial_p(n, p0, x, alt)?,
        ci_exact: clopper_pearson(x, n, alt, level)?,
        ci_blaker: match alt {
            Alternative::TwoSided if n <= 20_000 => blaker_interval(x, n, level)?,
            _ => clopper_pearson(x, n, alt, level)?,
        },
        ci_normal: wald_interval(p_hat, se_hat, f64::INFINITY, alt, level)?,
        continuity_correction,
    })
}

/// Result of a two-proportion test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoProportions {
    pub x1: u64,
    pub n1: u64,
    pub x2: u64,
    pub n2: u64,
    pub p1: f64,
    pub p2: f64,
    /// p̂₁ − p̂₂.
    pub diff: f64,
    pub diff0: f64,
    pub z: f64,
    pub p_value_normal: f64,
    /// Fisher's exact p-value; only for `diff0 = 0`.
    pub p_value_fisher: Option<f64>,
    /// Wald interval for p₁ − p₂ (unpooled variance).
    pub ci: Interval,
    pub continuity_correction: bool,
}

/// Two-proportion test of H₀: p₁ − p₂ = `diff0`.
///
/// For `diff0 = 0` the test variance uses the pooled proportion; otherwise
/// the unpooled variance (a Wald test). The confidence interval is always
/// unpooled. The continuity correction moves |p̂₁ − p̂₂ − d₀| toward 0 by
/// min(0.5(1/n₁ + 1/n₂), |p̂₁ − p̂₂ − d₀|).
pub fn two_proportions(
    x1: u64,
    n1: u64,
    x2: u64,
    n2: u64,
    diff0: f64,
    alt: Alternative,
    level: f64,
    continuity_correction: bool,
) -> Result<TwoProportions> {
    check_level(level)?;
    if n1 == 0 || n2 == 0 {
        return Err(StatsError::InsufficientData { what: "two-proportion test", needed: 1, got: 0 });
    }
    if x1 > n1 || x2 > n2 {
        return invalid("successes exceed trials");
    }
    if !(-1.0..=1.0).contains(&diff0) {
        return invalid(format!("the hypothesised difference {diff0} must be in [-1, 1]"));
    }
    let (f1, f2) = (n1 as f64, n2 as f64);
    let (p1, p2) = (x1 as f64 / f1, x2 as f64 / f2);
    let diff = p1 - p2;
    let var_test = if diff0 == 0.0 {
        let pp = (x1 + x2) as f64 / (f1 + f2);
        pp * (1.0 - pp) * (1.0 / f1 + 1.0 / f2)
    } else {
        p1 * (1.0 - p1) / f1 + p2 * (1.0 - p2) / f2
    };
    let var_ci = p1 * (1.0 - p1) / f1 + p2 * (1.0 - p2) / f2;
    let d = diff - diff0;
    if var_test <= 0.0 {
        if d == 0.0 {
            // both samples are all-success or all-failure and agree with H0
            return Ok(TwoProportions {
                x1,
                n1,
                x2,
                n2,
                p1,
                p2,
                diff,
                diff0,
                z: 0.0,
                p_value_normal: if alt == Alternative::TwoSided { 1.0 } else { 0.5 },
                p_value_fisher: (diff0 == 0.0).then_some(1.0),
                ci: Interval { lower: diff, upper: diff, level },
                continuity_correction,
            });
        }
        return invalid("the variance of the test statistic is zero, so z is undefined");
    }
    let adj = if continuity_correction { d.signum() * (0.5 * (1.0 / f1 + 1.0 / f2)).min(d.abs()) } else { 0.0 };
    let z = (d - adj) / var_test.sqrt();
    let fisher = if diff0 == 0.0 {
        let t = Crosstab::from_counts(vec![vec![x1, n1 - x1], vec![x2, n2 - x2]])?;
        Some(match fisher_exact(&t) {
            Ok(f) => match alt {
                Alternative::TwoSided => f.p_value,
                Alternative::Less => f.p_left.unwrap_or(1.0),
                Alternative::Greater => f.p_right.unwrap_or(1.0),
            },
            // an empty row/column: the margins force the table, p = 1
            Err(_) => 1.0,
        })
    } else {
        None
    };
    Ok(TwoProportions {
        x1,
        n1,
        x2,
        n2,
        p1,
        p2,
        diff,
        diff0,
        z,
        p_value_normal: z_p_value(z, alt),
        p_value_fisher: fisher,
        ci: wald_interval(diff, var_ci.sqrt(), f64::INFINITY, alt, level)?,
        continuity_correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use Alternative::*;

    #[test]
    fn exact_two_sided_symmetric_case() {
        // Bin(10, .5): x = 3 → 2·P(X<=3) = 0.34375
        assert!((binomial_two_sided(10, 0.5, 3).unwrap() - 0.34375).abs() < 1e-14);
        assert!((binomial_two_sided(10, 0.5, 7).unwrap() - 0.34375).abs() < 1e-14);
        assert_eq!(binomial_two_sided(10, 0.5, 5).unwrap(), 1.0);
        // asymmetric: Bin(20, .2), x = 8: brute force
        let b = Binomial::new(20, 0.2).unwrap();
        let d = b.pmf(8.0) * (1.0 + 1e-7);
        let brute: f64 = (0..=20).map(|k| b.pmf(f64::from(k))).filter(|&p| p <= d).sum();
        assert!((binomial_two_sided(20, 0.2, 8).unwrap() - brute).abs() < 1e-13);
        let d0 = b.pmf(0.0) * (1.0 + 1e-7);
        let brute0: f64 = (0..=20).map(|k| b.pmf(f64::from(k))).filter(|&p| p <= d0).sum();
        assert!((binomial_two_sided(20, 0.2, 0).unwrap() - brute0).abs() < 1e-13);
    }

    #[test]
    fn one_prop_features() {
        let r = one_proportion(14, 20, 0.5, TwoSided, 0.95, false).unwrap();
        assert!((r.p_hat - 0.7).abs() < 1e-15);
        assert!((r.z - 0.2 / (0.25f64 / 20.0).sqrt()).abs() < 1e-12);
        let c = one_proportion(14, 20, 0.5, TwoSided, 0.95, true).unwrap();
        assert!((c.z - (0.2 - 0.025) / (0.25f64 / 20.0).sqrt()).abs() < 1e-12);
        assert!(c.p_value_normal > r.p_value_normal);
        assert!(r.ci_exact.lower < 0.7 && r.ci_exact.upper > 0.7);
        let g = one_proportion(14, 20, 0.5, Greater, 0.95, false).unwrap();
        assert_eq!(g.ci_exact.upper, 1.0);
        assert_eq!(g.ci_normal.upper, f64::INFINITY);
        let edge = one_proportion(0, 10, 0.3, TwoSided, 0.95, false).unwrap();
        assert_eq!(edge.ci_exact.lower, 0.0);
        let all = one_proportion(10, 10, 0.3, TwoSided, 0.95, false).unwrap();
        assert_eq!(all.ci_exact.upper, 1.0);
        assert!(one_proportion(11, 10, 0.5, TwoSided, 0.95, false).is_err());
        assert!(one_proportion(1, 0, 0.5, TwoSided, 0.95, false).is_err());
        assert!(one_proportion(1, 10, 1.0, TwoSided, 0.95, false).is_err());
    }

    #[test]
    fn two_prop_features() {
        let r = two_proportions(30, 100, 18, 90, 0.0, TwoSided, 0.95, false).unwrap();
        let pp: f64 = 48.0 / 190.0;
        let se = (pp * (1.0 - pp) * (1.0 / 100.0 + 1.0 / 90.0)).sqrt();
        assert!((r.z - (0.3 - 0.2) / se).abs() < 1e-12);
        assert!(r.p_value_fisher.is_some());
        let nz = two_proportions(30, 100, 18, 90, 0.05, TwoSided, 0.95, false).unwrap();
        assert!(nz.p_value_fisher.is_none());
        let cc = two_proportions(30, 100, 18, 90, 0.0, TwoSided, 0.95, true).unwrap();
        assert!(cc.p_value_normal > r.p_value_normal);
        // degenerate
        let d = two_proportions(0, 10, 0, 12, 0.0, TwoSided, 0.95, false).unwrap();
        assert_eq!((d.z, d.p_value_normal, d.p_value_fisher), (0.0, 1.0, Some(1.0)));
        assert!(two_proportions(0, 10, 0, 12, 0.1, TwoSided, 0.95, false).is_err());
        assert!(two_proportions(1, 0, 1, 2, 0.0, TwoSided, 0.95, false).is_err());
    }
}
