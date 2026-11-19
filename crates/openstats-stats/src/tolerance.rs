//! Tolerance intervals.
//!
//! A (*p*, γ) tolerance interval is computed from a sample so that, with
//! confidence γ, it contains at least a proportion *p* (the coverage) of the
//! population.
//!
//! * **Normal, one-sided** — exact: the factor is `k = t′⁻¹_{n−1,δ}(γ)/√n`
//!   with δ = z_p·√n (non-central t).
//! * **Normal, two-sided** — [`NormalMethod::Exact`] solves
//!   `P(Φ(Z/√n + kW) − Φ(Z/√n − kW) ≥ p) = γ` (Z ~ N(0,1), W² ~ χ²_{n−1}/(n−1))
//!   by integrating over the χ² variable, the inner coverage condition being
//!   solved exactly for |Z|; [`NormalMethod::Howe`] is Howe's (1969) χ²
//!   approximation `k = z_{(1+p)/2}·√((n−1)(1+1/n)/χ²_{1−γ,n−1})`.
//! * **Nonparametric** — the interval between order statistics
//!   X₍ₗ₎ … X₍ᵤ₎; coverage of such an interval follows Beta(u−l, n−u+l+1), so the
//!   achieved confidence is `P(Bin(n, p) ≤ u−l−1)`, exact for every
//!   continuous population.

use crate::error::{invalid, Result, StatsError};
use openstats_math::dist::{Binomial, ChiSquared, ContinuousDistribution, DiscreteDistribution, NonCentralT, Normal};
use openstats_math::integrate::adaptive;
use openstats_math::roots::brent;

/// Which side(s) of the distribution the interval bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceSides {
    /// Interval `[lower, upper]`.
    TwoSided,
    /// Upper bound only: at least *p* of the population lies below it.
    Upper,
    /// Lower bound only: at least *p* of the population lies above it.
    Lower,
}

/// How the two-sided normal tolerance factor is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalMethod {
    /// Exact (numerical integral); one-sided factors use the non-central t.
    Exact,
    /// Howe's χ² approximation (one-sided factors are still exact).
    Howe,
}

fn check_pg(coverage: f64, confidence: f64) -> Result<()> {
    if !(coverage > 0.0 && coverage < 1.0) {
        return invalid(format!("coverage {coverage} must be in (0, 1)"));
    }
    if !(confidence > 0.0 && confidence < 1.0) {
        return invalid(format!("confidence {confidence} must be in (0, 1)"));
    }
    Ok(())
}

/// Howe's approximate two-sided factor.
fn howe_factor(n: usize, coverage: f64, confidence: f64) -> Result<f64> {
    let nu = (n - 1) as f64;
    let z = Normal::standard().quantile(0.5 + coverage / 2.0);
    let chi = ChiSquared::new(nu)?.quantile(1.0 - confidence);
    Ok(z * (nu * (1.0 + 1.0 / n as f64) / chi).sqrt())
}

/// Exact one-sided factor `t′⁻¹_{n−1, z_p√n}(γ)/√n`.
fn one_sided_factor(n: usize, coverage: f64, confidence: f64) -> Result<f64> {
    let nf = n as f64;
    let zp = Normal::standard().quantile(coverage);
    let d = NonCentralT::new(nf - 1.0, zp * nf.sqrt())?;
    Ok(d.quantile(confidence) / nf.sqrt())
}

/// Coverage of `x̄ ± k s` given `W = s/σ`: the set of standardised means
/// with coverage ≥ p is |Z| ≤ √n·r(W); returns `2Φ(√n r) − 1`.
fn prob_given_w(n: f64, k: f64, w: f64, p: f64) -> f64 {
    let a = k * w;
    let norm = Normal::standard();
    let cov = |r: f64| norm.sf(r - a) - norm.sf(r + a);
    if cov(0.0) < p {
        return 0.0;
    }
    let r = brent(|r| cov(r) - p, 0.0, a + 40.0, 1e-15, 300).unwrap_or(0.0);
    2.0 * norm.cdf(n.sqrt() * r) - 1.0
}

/// Exact confidence that `x̄ ± k s` covers at least proportion `coverage`
/// of a normal population, for a sample of size `n`.
pub fn normal_two_sided_confidence(n: usize, k: f64, coverage: f64) -> Result<f64> {
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "tolerance interval", needed: 2, got: n });
    }
    if !(coverage > 0.0 && coverage < 1.0) || !(k > 0.0 && k.is_finite()) {
        return invalid("coverage must be in (0, 1) and k positive and finite");
    }
    let nu = (n - 1) as f64;
    let nf = n as f64;
    let z = Normal::standard().quantile(0.5 + coverage / 2.0);
    let w0 = z / k;
    let chi = ChiSquared::new(nu)?;
    let w_hi = (chi.isf(1e-17) / nu).sqrt();
    if w0 >= w_hi {
        return Ok(0.0);
    }
    // Substitute w = w0 + v²: removes the square-root behaviour of r(w) at
    // w0 and makes the integrand smooth.
    let v_hi = (w_hi - w0).sqrt();
    let mut f = |v: f64| {
        let w = w0 + v * v;
        let q = nu * w * w;
        let dens_w = chi.pdf(q) * 2.0 * nu * w;
        prob_given_w(nf, k, w, coverage) * dens_w * 2.0 * v
    };
    let c = adaptive(&mut f, 0.0, v_hi, 1e-14, 1e-11)?;
    Ok(c.clamp(0.0, 1.0))
}

/// Exact confidence that the one-sided bound `x̄ + k s` (or `x̄ − k s`)
/// covers at least proportion `coverage`.
pub fn normal_one_sided_confidence(n: usize, k: f64, coverage: f64) -> Result<f64> {
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "tolerance interval", needed: 2, got: n });
    }
    if !(coverage > 0.0 && coverage < 1.0) {
        return invalid("coverage must be in (0, 1)");
    }
    let nf = n as f64;
    let zp = Normal::standard().quantile(coverage);
    Ok(NonCentralT::new(nf - 1.0, zp * nf.sqrt())?.cdf(k * nf.sqrt()))
}

/// The tolerance factor `k` for a sample of size `n`: the bound(s) are
/// `x̄ ± k s`.
pub fn normal_tolerance_factor(
    n: usize,
    coverage: f64,
    confidence: f64,
    sides: ToleranceSides,
    method: NormalMethod,
) -> Result<f64> {
    check_pg(coverage, confidence)?;
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "tolerance interval", needed: 2, got: n });
    }
    if sides != ToleranceSides::TwoSided {
        return one_sided_factor(n, coverage, confidence);
    }
    let howe = howe_factor(n, coverage, confidence)?;
    if method == NormalMethod::Howe {
        return Ok(howe);
    }
    let z = Normal::standard().quantile(0.5 + coverage / 2.0);
    let g = |k: f64| normal_two_sided_confidence(n, k, coverage).map(|c| c - confidence);
    let mut lo = z;
    let mut guard = 0;
    while g(lo)? >= 0.0 {
        lo *= 0.5;
        guard += 1;
        if guard > 60 {
            return Err(StatsError::Math(openstats_math::MathError::NoConvergence {
                algorithm: "tolerance factor bracket",
                iterations: guard,
            }));
        }
    }
    let mut hi = howe.max(lo * 1.01) * 1.5;
    guard = 0;
    while g(hi)? < 0.0 {
        hi *= 2.0;
        guard += 1;
        if guard > 60 {
            return Err(StatsError::Math(openstats_math::MathError::NoConvergence {
                algorithm: "tolerance factor bracket",
                iterations: guard,
            }));
        }
    }
    let mut err = None;
    let k = brent(
        |k| match g(k) {
            Ok(v) => v,
            Err(e) => {
                err = Some(e);
                f64::NAN
            }
        },
        lo,
        hi,
        1e-13,
        200,
    );
    if let Some(e) = err {
        return Err(e);
    }
    Ok(k?)
}

/// A normal-theory tolerance interval.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalTolerance {
    /// Sample size (non-missing observations).
    pub n: usize,
    /// Sample mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Minimum proportion of the population covered.
    pub coverage: f64,
    /// Confidence level.
    pub confidence: f64,
    /// Tolerance factor `k`.
    pub k: f64,
    /// Lower limit (`-∞` for an upper bound).
    pub lower: f64,
    /// Upper limit (`+∞` for a lower bound).
    pub upper: f64,
    /// Sides bounded.
    pub sides: ToleranceSides,
    /// Method used for the two-sided factor.
    pub method: NormalMethod,
}

/// Normal tolerance interval from data (missing values `NaN` are dropped).
pub fn normal_tolerance(
    data: &[f64],
    coverage: f64,
    confidence: f64,
    sides: ToleranceSides,
    method: NormalMethod,
) -> Result<NormalTolerance> {
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let x: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    if x.len() < 2 {
        return Err(StatsError::InsufficientData { what: "tolerance interval", needed: 2, got: x.len() });
    }
    let (mean, std_dev) = (openstats_math::stats::summary::mean(&x)?, openstats_math::stats::summary::std_dev(&x)?);
    let k = normal_tolerance_factor(x.len(), coverage, confidence, sides, method)?;
    let (lower, upper) = match sides {
        ToleranceSides::TwoSided => (mean - k * std_dev, mean + k * std_dev),
        ToleranceSides::Upper => (f64::NEG_INFINITY, mean + k * std_dev),
        ToleranceSides::Lower => (mean - k * std_dev, f64::INFINITY),
    };
    Ok(NormalTolerance { n: x.len(), mean, std_dev, coverage, confidence, k, lower, upper, sides, method })
}

/// A distribution-free tolerance interval from order statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct NonparametricTolerance {
    /// Sample size.
    pub n: usize,
    /// Minimum proportion of the population covered.
    pub coverage: f64,
    /// Confidence level requested.
    pub requested_confidence: f64,
    /// Confidence actually achieved (≥ requested).
    pub achieved_confidence: f64,
    /// 1-based rank of the lower limit, if there is one.
    pub lower_rank: Option<usize>,
    /// 1-based rank of the upper limit, if there is one.
    pub upper_rank: Option<usize>,
    /// Lower limit (`-∞` when absent).
    pub lower: f64,
    /// Upper limit (`+∞` when absent).
    pub upper: f64,
    /// Sides bounded.
    pub sides: ToleranceSides,
}

/// Exact confidence that the interval between the order statistics of rank
/// `lower` and `upper` (1-based, `None` = unbounded side) covers at least
/// `coverage` of the population.
pub fn nonparametric_confidence(n: usize, coverage: f64, lower: Option<usize>, upper: Option<usize>) -> Result<f64> {
    if !(coverage > 0.0 && coverage < 1.0) {
        return invalid("coverage must be in (0, 1)");
    }
    // Number of sample points the interval spans on the probability scale:
    // coverage ~ Beta(m, n − m + 1) with m = u − l (two-sided),
    // m = n − l + 1 (lower bound only: X₍ₗ₎, ∞) or m = u (upper bound only).
    let m = match (lower, upper) {
        (Some(l), Some(u)) => {
            if l < 1 || u > n || l >= u {
                return invalid("ranks must satisfy 1 ≤ lower < upper ≤ n");
            }
            u - l
        }
        (Some(l), None) => {
            if l < 1 || l > n {
                return invalid("lower rank must be in 1..=n");
            }
            n - l + 1
        }
        (None, Some(u)) => {
            if u < 1 || u > n {
                return invalid("upper rank must be in 1..=n");
            }
            u
        }
        (None, None) => return invalid("at least one bound is required"),
    };
    // P(Beta(m, n−m+1) ≥ p) = P(Bin(n, p) ≤ m − 1)
    Ok(Binomial::new(n as u64, coverage)?.cdf((m - 1) as f64))
}

/// Symmetric order-statistic tolerance interval with the smallest span
/// whose exact confidence is at least `confidence`.
///
/// Two-sided uses ranks `(r, n+1−r)` with the largest feasible `r`; the
/// one-sided bounds use rank `r` (lower) / `n+1−r` (upper) likewise. Fails
/// with an explanatory error when even the sample extremes do not reach the
/// requested confidence (see [`nonparametric_min_sample_size`]).
pub fn nonparametric_tolerance(
    data: &[f64],
    coverage: f64,
    confidence: f64,
    sides: ToleranceSides,
) -> Result<NonparametricTolerance> {
    check_pg(coverage, confidence)?;
    if data.iter().any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite);
    }
    let mut x: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientData { what: "tolerance interval", needed: 2, got: n });
    }
    x.sort_by(f64::total_cmp);
    let conf_of = |r: usize| -> Result<f64> {
        match sides {
            ToleranceSides::TwoSided => nonparametric_confidence(n, coverage, Some(r), Some(n + 1 - r)),
            ToleranceSides::Lower => nonparametric_confidence(n, coverage, Some(r), None),
            ToleranceSides::Upper => nonparametric_confidence(n, coverage, None, Some(n + 1 - r)),
        }
    };
    let r_max = if sides == ToleranceSides::TwoSided { (n - 1) / 2 } else { n };
    let mut best: Option<(usize, f64)> = None;
    // Confidence decreases as r grows: take the largest r that still reaches it.
    for r in 1..=r_max {
        let c = conf_of(r)?;
        if c >= confidence {
            best = Some((r, c));
        } else {
            break;
        }
    }
    let Some((r, achieved)) = best else {
        let need = nonparametric_min_sample_size(coverage, confidence, sides)?;
        return invalid(format!(
            "n = {n} is too small for a {:.1}% confidence / {:.1}% coverage nonparametric tolerance interval; at least {need} observations are needed",
            confidence * 100.0,
            coverage * 100.0
        ));
    };
    let (lower_rank, upper_rank) = match sides {
        ToleranceSides::TwoSided => (Some(r), Some(n + 1 - r)),
        ToleranceSides::Lower => (Some(r), None),
        ToleranceSides::Upper => (None, Some(n + 1 - r)),
    };
    Ok(NonparametricTolerance {
        n,
        coverage,
        requested_confidence: confidence,
        achieved_confidence: achieved,
        lower_rank,
        upper_rank,
        lower: lower_rank.map_or(f64::NEG_INFINITY, |l| x[l - 1]),
        upper: upper_rank.map_or(f64::INFINITY, |u| x[u - 1]),
        sides,
    })
}

/// Smallest sample size for which the sample extremes give a nonparametric
/// tolerance interval (X₍₁₎, X₍ₙ₎) — or X₍₁₎ / X₍ₙ₎ alone for one-sided
/// bounds — with confidence at least `confidence`.
pub fn nonparametric_min_sample_size(coverage: f64, confidence: f64, sides: ToleranceSides) -> Result<usize> {
    check_pg(coverage, confidence)?;
    let conf = |n: usize| match sides {
        ToleranceSides::TwoSided => nonparametric_confidence(n, coverage, Some(1), Some(n)),
        ToleranceSides::Lower => nonparametric_confidence(n, coverage, Some(1), None),
        ToleranceSides::Upper => nonparametric_confidence(n, coverage, None, Some(n)),
    };
    let min = if sides == ToleranceSides::TwoSided { 2 } else { 1 };
    // Monotone increasing in n: bracket by doubling, then bisect.
    let mut hi = min.max(2);
    while conf(hi)? < confidence {
        // 2^30 fits `usize` on 32-bit targets (wasm32) as well as 64-bit ones.
        hi = match hi.checked_mul(2) {
            Some(next) if next <= 1usize << 30 => next,
            _ => return Err(StatsError::TooLarge("nonparametric tolerance sample size".into())),
        };
    }
    let mut lo = min;
    if conf(lo)? >= confidence {
        return Ok(lo);
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if conf(mid)? >= confidence {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_sided_factors_match_published_table() {
        // NIST / Natrella Table A-12 (95 % coverage, 95 % confidence).
        for (n, k) in [(10, 2.911), (20, 2.396), (30, 2.220), (50, 2.065)] {
            let got = normal_tolerance_factor(n, 0.95, 0.95, ToleranceSides::Upper, NormalMethod::Exact).unwrap();
            assert!((got - k).abs() < 6e-4, "n={n}: {got} vs {k}");
        }
    }

    #[test]
    fn two_sided_factors_match_independent_integration() {
        // 95 % coverage / 95 % confidence; reference values from an independent
        // SciPy quad/brentq evaluation of the same defining integral (and a
        // 400,000-sample Monte Carlo, see the next test).
        for (n, k) in [(10, 3.3934294787), (20, 2.7603461784), (30, 2.5548928133), (50, 2.3815597422)] {
            let got = normal_tolerance_factor(n, 0.95, 0.95, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
            assert!((got - k).abs() < 2e-8, "n={n}: {got} vs {k}");
        }
    }

    #[test]
    fn two_sided_factor_has_nominal_confidence_by_simulation() {
        use openstats_math::random::Pcg64;
        let norm = Normal::standard();
        let mut rng = Pcg64::with_stream(77, 3);
        let (n, p, g) = (10usize, 0.95, 0.95);
        let k = normal_tolerance_factor(n, p, g, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
        let reps = 100_000;
        let mut ok = 0u32;
        for _ in 0..reps {
            let x = norm.sample_n(&mut rng, n);
            let m = x.iter().sum::<f64>() / n as f64;
            let s = (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1) as f64).sqrt();
            if norm.cdf(m + k * s) - norm.cdf(m - k * s) >= p {
                ok += 1;
            }
        }
        let freq = f64::from(ok) / f64::from(reps);
        assert!((freq - g).abs() < 0.003, "{freq}");
    }

    #[test]
    fn exact_factor_round_trips_through_confidence() {
        for (n, p, g) in [(5usize, 0.9, 0.9), (12, 0.99, 0.95), (40, 0.5, 0.99), (2, 0.9, 0.9)] {
            let k = normal_tolerance_factor(n, p, g, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
            let back = normal_two_sided_confidence(n, k, p).unwrap();
            assert!((back - g).abs() < 1e-9, "n={n}: {back}");
            let k1 = normal_tolerance_factor(n, p, g, ToleranceSides::Lower, NormalMethod::Exact).unwrap();
            assert!((normal_one_sided_confidence(n, k1, p).unwrap() - g).abs() < 1e-9);
        }
    }

    #[test]
    fn howe_is_close_to_exact_for_moderate_n() {
        let e = normal_tolerance_factor(30, 0.95, 0.95, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
        let h = normal_tolerance_factor(30, 0.95, 0.95, ToleranceSides::TwoSided, NormalMethod::Howe).unwrap();
        assert!((e - h).abs() / e < 5e-3);
    }

    #[test]
    fn factor_tends_to_normal_quantile() {
        let k = normal_tolerance_factor(100_000, 0.95, 0.95, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
        assert!((k - 1.959964).abs() < 0.01, "{k}");
    }

    #[test]
    fn interval_from_data() {
        let x = [9.8, 10.1, 10.4, 9.9, 10.0, 10.3, 9.7, 10.2];
        let t = normal_tolerance(&x, 0.95, 0.95, ToleranceSides::TwoSided, NormalMethod::Exact).unwrap();
        assert!((t.mean - 10.05).abs() < 1e-12);
        assert!((t.upper - t.mean - t.k * t.std_dev).abs() < 1e-12);
        let u = normal_tolerance(&x, 0.95, 0.95, ToleranceSides::Upper, NormalMethod::Exact).unwrap();
        assert!(u.lower == f64::NEG_INFINITY && u.upper.is_finite());
        assert!(normal_tolerance(&[1.0], 0.9, 0.9, ToleranceSides::Upper, NormalMethod::Exact).is_err());
        assert!(normal_tolerance(&x, 1.0, 0.9, ToleranceSides::Upper, NormalMethod::Exact).is_err());
    }

    #[test]
    fn nonparametric_sample_sizes_match_classical_values() {
        // Two-sided (min, max) with 95 % coverage / 95 % confidence needs n = 93;
        // one-sided needs n = 59 (the "rule of 59"); 90/90 two-sided needs 38.
        assert_eq!(nonparametric_min_sample_size(0.95, 0.95, ToleranceSides::TwoSided).unwrap(), 93);
        assert_eq!(nonparametric_min_sample_size(0.95, 0.95, ToleranceSides::Upper).unwrap(), 59);
        assert_eq!(nonparametric_min_sample_size(0.95, 0.95, ToleranceSides::Lower).unwrap(), 59);
        assert_eq!(nonparametric_min_sample_size(0.90, 0.90, ToleranceSides::TwoSided).unwrap(), 38);
        assert_eq!(nonparametric_min_sample_size(0.99, 0.99, ToleranceSides::TwoSided).unwrap(), 662);
    }

    #[test]
    fn nonparametric_confidence_closed_forms() {
        // (X(1), X(n)): confidence = 1 − n p^{n−1} + (n−1) p^n.
        let (n, p) = (20usize, 0.9f64);
        let c = nonparametric_confidence(n, p, Some(1), Some(n)).unwrap();
        let exact = 1.0 - 20.0 * p.powi(19) + 19.0 * p.powi(20);
        assert!((c - exact).abs() < 1e-13);
        // One-sided max: 1 − p^n.
        let c1 = nonparametric_confidence(n, p, None, Some(n)).unwrap();
        assert!((c1 - (1.0 - p.powi(20))).abs() < 1e-13);
        assert!(nonparametric_confidence(5, 0.9, Some(3), Some(3)).is_err());
        assert!(nonparametric_confidence(5, 0.9, None, None).is_err());
    }

    #[test]
    fn nonparametric_interval_picks_tightest_feasible_ranks() {
        let data: Vec<f64> = (1..=100).map(f64::from).collect();
        let t = nonparametric_tolerance(&data, 0.90, 0.95, ToleranceSides::TwoSided).unwrap();
        let (l, u) = (t.lower_rank.unwrap(), t.upper_rank.unwrap());
        assert_eq!(l + u, 101);
        assert!(t.achieved_confidence >= 0.95);
        // One rank further in would fall below the requested confidence.
        let worse = nonparametric_confidence(100, 0.90, Some(l + 1), Some(u - 1)).unwrap();
        assert!(worse < 0.95);
        assert_eq!(t.lower, l as f64);
        let up = nonparametric_tolerance(&data, 0.95, 0.95, ToleranceSides::Upper).unwrap();
        assert_eq!(up.lower, f64::NEG_INFINITY);
        assert!(up.achieved_confidence >= 0.95);
    }

    #[test]
    fn nonparametric_too_small_sample_explains_requirement() {
        let data: Vec<f64> = (1..=30).map(f64::from).collect();
        let e = nonparametric_tolerance(&data, 0.95, 0.95, ToleranceSides::TwoSided).unwrap_err();
        assert!(e.to_string().contains("at least 93"), "{e}");
    }

    #[test]
    fn nonparametric_coverage_is_calibrated() {
        // Monte Carlo: uniform samples, the interval must cover ≥ p with the
        // stated frequency (exact for continuous populations).
        use openstats_math::random::{Pcg64, Rng};
        let mut rng = Pcg64::with_stream(2024, 5);
        let (n, p) = (40usize, 0.8);
        let mut hits = 0;
        let reps = 4000;
        let mut conf = 0.0;
        for _ in 0..reps {
            let x: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
            let t = nonparametric_tolerance(&x, p, 0.9, ToleranceSides::TwoSided).unwrap();
            conf = t.achieved_confidence;
            if t.upper - t.lower >= p {
                hits += 1;
            }
        }
        let freq = f64::from(hits) / f64::from(reps);
        assert!((freq - conf).abs() < 0.02, "{freq} vs {conf}");
    }
}
