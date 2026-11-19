//! Continuous probability distributions.
//!
//! Every distribution implements [`ContinuousDistribution`] with density,
//! log-density, distribution function, survival function, quantile and
//! inverse survival function. Upper-tail functions (`sf`, `isf`) are computed
//! directly — never as `1 − cdf` — so tiny tail probabilities keep full
//! relative accuracy. Parameterisations follow Minitab (scale/shape/threshold
//! names and meanings).
//!
//! | family | types |
//! |---|---|
//! | normal-based | [`Normal`], [`Lognormal`], [`HalfNormal`], [`FoldedNormal`], [`Johnson`] |
//! | gamma-based | [`Gamma`], [`ChiSquared`], [`Exponential`], [`Weibull`] |
//! | beta-based | [`Beta`], [`StudentT`], [`FisherF`] |
//! | discrete (see [`DiscreteDistribution`]) | [`Binomial`], [`Poisson`], [`Hypergeometric`], [`NegativeBinomial`], [`Geometric`], [`DiscreteUniform`] (= [`Integer`]), [`Discrete`] |
//! | non-central | [`NonCentralT`], [`NonCentralChiSquared`], [`NonCentralF`], [`NonCentralBeta`] |
//! | location–scale | [`Logistic`], [`Loglogistic`], [`Cauchy`], [`Laplace`], [`Uniform`], [`Triangular`], [`SmallestExtremeValue`], [`LargestExtremeValue`] |

mod beta_family;
mod discrete;
mod gamma_family;
mod location_scale;
mod noncentral;
mod normal_family;

pub use beta_family::{Beta, FisherF, StudentT};
pub use discrete::{
    Binomial, Discrete, DiscreteDistribution, DiscreteUniform, Geometric, Hypergeometric, Integer, NegativeBinomial,
    Poisson,
};
pub use gamma_family::{ChiSquared, Exponential, Gamma, Weibull};
pub use location_scale::{
    Cauchy, Laplace, LargestExtremeValue, Logistic, Loglogistic, SmallestExtremeValue, Triangular, Uniform,
};
pub use noncentral::{NonCentralBeta, NonCentralChiSquared, NonCentralF, NonCentralT};
pub use normal_family::{FoldedNormal, HalfNormal, Johnson, JohnsonFamily, Lognormal, Normal};

use crate::error::{MathError, Result};

/// Interface shared by all continuous distributions.
pub trait ContinuousDistribution {
    /// Short name ("Normal", "Weibull", …).
    fn name(&self) -> &'static str;
    /// Probability density f(x).
    fn pdf(&self, x: f64) -> f64;
    /// ln f(x).
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
    /// P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;
    /// P(X > x), computed without cancellation.
    fn sf(&self, x: f64) -> f64;
    /// x with P(X ≤ x) = p.
    fn quantile(&self, p: f64) -> f64;
    /// x with P(X > x) = q.
    fn isf(&self, q: f64) -> f64;
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    fn median(&self) -> f64 {
        self.quantile(0.5)
    }
    /// (lower, upper) bounds of the support.
    fn support(&self) -> (f64, f64);
    /// One random variate. The default is inversion, `quantile(U)` with
    /// U uniform on (0, 1); distributions with a dedicated exact algorithm
    /// (see [`crate::random::variates`]) override it.
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.quantile(rng.next_f64_open())
    }
    /// `n` independent variates.
    fn sample_n(&self, rng: &mut dyn crate::random::Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

pub(crate) fn require(cond: bool, msg: impl FnOnce() -> String) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(MathError::InvalidArgument(msg()))
    }
}

/// `true` if p is a valid probability; NaN propagates through callers.
pub(crate) fn valid_p(p: f64) -> bool {
    (0.0..=1.0).contains(&p)
}

/// Quantile by bracketing and Brent's method on the cdf (or on the survival
/// function for upper-tail targets), used where no closed form exists.
pub(crate) fn numeric_quantile<D: ContinuousDistribution + ?Sized>(d: &D, p: f64, upper: bool) -> f64 {
    if !valid_p(p) {
        return f64::NAN;
    }
    let (lo_s, hi_s) = d.support();
    if (!upper && p == 0.0) || (upper && p == 1.0) {
        return lo_s;
    }
    if (!upper && p == 1.0) || (upper && p == 0.0) {
        return hi_s;
    }
    let f = |x: f64| if upper { p - d.sf(x) } else { d.cdf(x) - p };
    let m = d.median();
    let s = d.std_dev();
    let s = if s.is_finite() && s > 0.0 { s } else { 1.0 };
    let start_lo = if lo_s.is_finite() { lo_s.max(m - s) } else { m - s };
    let start_hi = if hi_s.is_finite() { hi_s.min(m + s) } else { m + s };
    match crate::roots::expand_bracket(f, start_lo, start_hi.max(start_lo + f64::EPSILON), lo_s, hi_s) {
        Ok((a, b)) => crate::roots::brent(f, a, b, 0.0, 300).unwrap_or(f64::NAN),
        Err(_) => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generic consistency checks every distribution must satisfy.
    pub(crate) fn check<D: ContinuousDistribution>(d: &D, xs: &[f64]) {
        for &x in xs {
            let (c, s) = (d.cdf(x), d.sf(x));
            assert!((c + s - 1.0).abs() < 1e-14, "{}: cdf+sf at {x}", d.name());
            assert!((0.0..=1.0).contains(&c), "{}: cdf range", d.name());
            let pd = d.pdf(x);
            assert!(pd >= 0.0, "{}: pdf ≥ 0", d.name());
            if pd > 0.0 && pd.is_finite() {
                assert!((d.ln_pdf(x) - pd.ln()).abs() < 1e-12 * pd.ln().abs().max(1.0), "{}: ln_pdf at {x}", d.name());
                // Numerical derivative of the cdf matches the density.
                let (lo, hi) = d.support();
                // Shrink the step near finite support boundaries where the
                // density may be singular or steep.
                let mut h = 1e-6 * x.abs().max(1.0);
                if lo.is_finite() {
                    h = h.min(1e-3 * (x - lo));
                }
                if hi.is_finite() {
                    h = h.min(1e-3 * (hi - x));
                }
                if x - h > lo && x + h < hi {
                    let num = (d.cdf(x + h) - d.cdf(x - h)) / (2.0 * h);
                    assert!((num - pd).abs() < 1e-5 * pd.max(1e-3), "{}: pdf vs dcdf at {x}: {pd} vs {num}", d.name());
                }
            }
            if c > 1e-300 && c < 1.0 {
                let q = d.quantile(c);
                assert!(
                    (d.cdf(q) - c).abs() <= 1e-12 * c.max(1e-300) + 1e-15,
                    "{}: quantile round trip at {x}",
                    d.name()
                );
            }
            if s > 1e-300 && s < 1.0 {
                let q = d.isf(s);
                assert!((d.sf(q) - s).abs() <= 1e-12 * s + 1e-15, "{}: isf round trip at {x}", d.name());
            }
        }
        assert!((d.quantile(0.5) - d.median()).abs() <= 1e-12 * d.median().abs().max(1.0));
    }
}
