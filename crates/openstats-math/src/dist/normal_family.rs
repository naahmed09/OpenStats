//! Normal, lognormal, half-normal, folded normal and Johnson distributions.

use super::{numeric_quantile, require, valid_p, ContinuousDistribution};
use crate::error::Result;
use crate::special::erf::{erf_inv, norm_cdf, norm_isf, norm_ln_pdf, norm_pdf, norm_ppf, norm_sf};
use crate::special::tukey::normal_points;
use std::f64::consts::{PI, SQRT_2};

/// Normal(μ, σ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        require(mu.is_finite() && sigma > 0.0 && sigma.is_finite(), || {
            format!("Normal: need finite μ and σ > 0 (got {mu}, {sigma})")
        })?;
        Ok(Normal { mu, sigma })
    }
    pub fn standard() -> Self {
        Normal { mu: 0.0, sigma: 1.0 }
    }
}

impl ContinuousDistribution for Normal {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.mu + self.sigma * crate::random::variates::std_normal(rng)
    }
    fn name(&self) -> &'static str {
        "Normal"
    }
    fn pdf(&self, x: f64) -> f64 {
        norm_pdf((x - self.mu) / self.sigma) / self.sigma
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        norm_ln_pdf((x - self.mu) / self.sigma) - self.sigma.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        norm_cdf((x - self.mu) / self.sigma)
    }
    fn sf(&self, x: f64) -> f64 {
        norm_sf((x - self.mu) / self.sigma)
    }
    fn quantile(&self, p: f64) -> f64 {
        self.mu + self.sigma * norm_ppf(p)
    }
    fn isf(&self, q: f64) -> f64 {
        self.mu + self.sigma * norm_isf(q)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
    fn median(&self) -> f64 {
        self.mu
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Lognormal: ln(X − θ) ~ Normal(μ, σ) (Minitab: location μ, scale σ, threshold θ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lognormal {
    pub mu: f64,
    pub sigma: f64,
    pub threshold: f64,
}

impl Lognormal {
    pub fn new(mu: f64, sigma: f64, threshold: f64) -> Result<Self> {
        require(mu.is_finite() && sigma > 0.0 && sigma.is_finite() && threshold.is_finite(), || {
            format!("Lognormal: need finite location/threshold and scale > 0 (got {mu}, {sigma}, {threshold})")
        })?;
        Ok(Lognormal { mu, sigma, threshold })
    }
    fn z(&self, x: f64) -> f64 {
        ((x - self.threshold).ln() - self.mu) / self.sigma
    }
}

impl ContinuousDistribution for Lognormal {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.threshold + (self.mu + self.sigma * crate::random::variates::std_normal(rng)).exp()
    }
    fn name(&self) -> &'static str {
        "Lognormal"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            return 0.0;
        }
        norm_pdf(self.z(x)) / (self.sigma * (x - self.threshold))
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            return f64::NEG_INFINITY;
        }
        norm_ln_pdf(self.z(x)) - self.sigma.ln() - (x - self.threshold).ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            0.0
        } else {
            norm_cdf(self.z(x))
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            1.0
        } else {
            norm_sf(self.z(x))
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        self.threshold + (self.mu + self.sigma * norm_ppf(p)).exp()
    }
    fn isf(&self, q: f64) -> f64 {
        self.threshold + (self.mu + self.sigma * norm_isf(q)).exp()
    }
    fn mean(&self) -> f64 {
        self.threshold + (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }
    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        s2.exp_m1() * (2.0 * self.mu + s2).exp()
    }
    fn median(&self) -> f64 {
        self.threshold + self.mu.exp()
    }
    fn support(&self) -> (f64, f64) {
        (self.threshold, f64::INFINITY)
    }
}

/// Half-normal: X − θ = |Z|·σ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfNormal {
    pub sigma: f64,
    pub threshold: f64,
}

impl HalfNormal {
    pub fn new(sigma: f64, threshold: f64) -> Result<Self> {
        require(sigma > 0.0 && sigma.is_finite() && threshold.is_finite(), || {
            format!("Half-normal: need scale > 0 (got {sigma})")
        })?;
        Ok(HalfNormal { sigma, threshold })
    }
}

impl ContinuousDistribution for HalfNormal {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.threshold + self.sigma * crate::random::variates::std_normal(rng).abs()
    }
    fn name(&self) -> &'static str {
        "Half-normal"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x < self.threshold {
            return 0.0;
        }
        2.0 * norm_pdf((x - self.threshold) / self.sigma) / self.sigma
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.threshold {
            return f64::NEG_INFINITY;
        }
        std::f64::consts::LN_2 + norm_ln_pdf((x - self.threshold) / self.sigma) - self.sigma.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            return 0.0;
        }
        let z = (x - self.threshold) / self.sigma;
        if z < 1.0 {
            crate::special::erf::erf(z / SQRT_2)
        } else {
            1.0 - 2.0 * norm_sf(z)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            return 1.0;
        }
        2.0 * norm_sf((x - self.threshold) / self.sigma)
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.threshold + self.sigma * SQRT_2 * erf_inv(p)
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.threshold + self.sigma * norm_isf(0.5 * q)
    }
    fn mean(&self) -> f64 {
        self.threshold + self.sigma * (2.0 / PI).sqrt()
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma * (1.0 - 2.0 / PI)
    }
    fn support(&self) -> (f64, f64) {
        (self.threshold, f64::INFINITY)
    }
}

/// Folded normal: |Y| with Y ~ Normal(μ, σ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoldedNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl FoldedNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        require(mu.is_finite() && sigma > 0.0 && sigma.is_finite(), || {
            format!("Folded normal: need σ > 0 (got {sigma})")
        })?;
        Ok(FoldedNormal { mu, sigma })
    }
}

impl ContinuousDistribution for FoldedNormal {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        (self.mu + self.sigma * crate::random::variates::std_normal(rng)).abs()
    }
    fn name(&self) -> &'static str {
        "Folded normal"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        (norm_pdf((x - self.mu) / self.sigma) + norm_pdf((x + self.mu) / self.sigma)) / self.sigma
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (a, b) = ((x - self.mu) / self.sigma, (-x - self.mu) / self.sigma);
        if a - b < 0.5 {
            // Short interval: integrate the (smooth) density directly to keep
            // relative accuracy as x → 0 (the difference below would cancel).
            return small_interval_rule().integrate(0.0, x, |t| self.pdf(t));
        }
        // Φ(a) − Φ(b); choose the form without cancellation.
        if a <= 0.0 {
            norm_cdf(a) - norm_cdf(b)
        } else if b >= 0.0 {
            norm_sf(b) - norm_sf(a)
        } else {
            1.0 - norm_sf(a) - norm_cdf(b)
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        norm_sf((x - self.mu) / self.sigma) + norm_sf((x + self.mu) / self.sigma)
    }
    fn quantile(&self, p: f64) -> f64 {
        if p > 0.5 {
            numeric_quantile(self, 1.0 - p, true)
        } else {
            numeric_quantile(self, p, false)
        }
    }
    fn isf(&self, q: f64) -> f64 {
        numeric_quantile(self, q, true)
    }
    fn mean(&self) -> f64 {
        let r = self.mu / self.sigma;
        self.sigma * (2.0 / PI).sqrt() * (-0.5 * r * r).exp() + self.mu * (1.0 - 2.0 * norm_cdf(-r))
    }
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.mu * self.mu + self.sigma * self.sigma - m * m
    }
    fn median(&self) -> f64 {
        numeric_quantile(&Approx(*self), 0.5, false)
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

fn small_interval_rule() -> &'static crate::integrate::GaussLegendre {
    static R: std::sync::OnceLock<crate::integrate::GaussLegendre> = std::sync::OnceLock::new();
    R.get_or_init(|| crate::integrate::GaussLegendre::new(20))
}

/// Wrapper used to compute the folded-normal median without recursion
/// through `median()` inside `numeric_quantile`.
struct Approx(FoldedNormal);

impl ContinuousDistribution for Approx {
    fn name(&self) -> &'static str {
        "Folded normal"
    }
    fn pdf(&self, x: f64) -> f64 {
        self.0.pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        self.0.cdf(x)
    }
    fn sf(&self, x: f64) -> f64 {
        self.0.sf(x)
    }
    fn quantile(&self, _p: f64) -> f64 {
        f64::NAN
    }
    fn isf(&self, _q: f64) -> f64 {
        f64::NAN
    }
    fn mean(&self) -> f64 {
        self.0.mean()
    }
    fn variance(&self) -> f64 {
        self.0.variance()
    }
    fn median(&self) -> f64 {
        self.0.mean()
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

/// Johnson system families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JohnsonFamily {
    /// Unbounded: z = γ + δ·asinh((x−ξ)/λ).
    SU,
    /// Bounded on (ξ, ξ+λ): z = γ + δ·ln(u/(1−u)), u = (x−ξ)/λ.
    SB,
    /// Lognormal: z = γ + δ·ln((x−ξ)/λ).
    SL,
}

/// Johnson distribution: Z = γ + δ·g((X − ξ)/λ) is standard normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Johnson {
    pub family: JohnsonFamily,
    pub gamma: f64,
    pub delta: f64,
    pub xi: f64,
    pub lambda: f64,
}

impl Johnson {
    pub fn new(family: JohnsonFamily, gamma: f64, delta: f64, xi: f64, lambda: f64) -> Result<Self> {
        require(
            gamma.is_finite()
                && xi.is_finite()
                && delta > 0.0
                && lambda > 0.0
                && delta.is_finite()
                && lambda.is_finite(),
            || format!("Johnson: need δ > 0 and λ > 0 (got δ={delta}, λ={lambda})"),
        )?;
        Ok(Johnson { family, gamma, delta, xi, lambda })
    }

    /// (z, dz/dx) or None outside the support.
    fn z(&self, x: f64) -> Option<(f64, f64)> {
        let u = (x - self.xi) / self.lambda;
        match self.family {
            JohnsonFamily::SU => Some((self.gamma + self.delta * u.asinh(), self.delta / (self.lambda * u.hypot(1.0)))),
            JohnsonFamily::SB => {
                if u <= 0.0 || u >= 1.0 {
                    None
                } else {
                    Some((self.gamma + self.delta * (u / (1.0 - u)).ln(), self.delta / (self.lambda * u * (1.0 - u))))
                }
            }
            JohnsonFamily::SL => {
                if u <= 0.0 {
                    None
                } else {
                    Some((self.gamma + self.delta * u.ln(), self.delta / (self.lambda * u)))
                }
            }
        }
    }

    /// x for a standard-normal deviate z.
    fn x_of_z(&self, z: f64) -> f64 {
        let w = (z - self.gamma) / self.delta;
        let u = match self.family {
            JohnsonFamily::SU => w.sinh(),
            JohnsonFamily::SB => 1.0 / (1.0 + (-w).exp()),
            JohnsonFamily::SL => w.exp(),
        };
        self.xi + self.lambda * u
    }

    /// E[g(X)] by Gauss–Legendre over the normal deviate.
    fn expect(&self, g: impl Fn(f64) -> f64) -> f64 {
        normal_points().iter().map(|&(z, w, _)| w * g(self.x_of_z(z))).sum()
    }
}

impl ContinuousDistribution for Johnson {
    fn name(&self) -> &'static str {
        match self.family {
            JohnsonFamily::SU => "Johnson SU",
            JohnsonFamily::SB => "Johnson SB",
            JohnsonFamily::SL => "Johnson SL",
        }
    }
    fn pdf(&self, x: f64) -> f64 {
        match self.z(x) {
            Some((z, dz)) => norm_pdf(z) * dz,
            None => 0.0,
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        match self.z(x) {
            Some((z, _)) => norm_cdf(z),
            None => {
                if x <= self.xi {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }
    fn sf(&self, x: f64) -> f64 {
        match self.z(x) {
            Some((z, _)) => norm_sf(z),
            None => {
                if x <= self.xi {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        self.x_of_z(norm_ppf(p))
    }
    fn isf(&self, q: f64) -> f64 {
        self.x_of_z(norm_isf(q))
    }
    fn mean(&self) -> f64 {
        let (g, d, xi, l) = (self.gamma, self.delta, self.xi, self.lambda);
        match self.family {
            JohnsonFamily::SU => xi - l * (0.5 / (d * d)).exp() * (g / d).sinh(),
            JohnsonFamily::SL => xi + l * (-g / d + 0.5 / (d * d)).exp(),
            JohnsonFamily::SB => self.expect(|x| x),
        }
    }
    fn variance(&self) -> f64 {
        let (g, d, l) = (self.gamma, self.delta, self.lambda);
        match self.family {
            JohnsonFamily::SU => {
                let w = (1.0 / (d * d)).exp();
                0.5 * l * l * (w - 1.0) * (w * (2.0 * g / d).cosh() + 1.0)
            }
            JohnsonFamily::SL => {
                let s2 = 1.0 / (d * d);
                l * l * (-2.0 * g / d).exp() * s2.exp() * s2.exp_m1()
            }
            JohnsonFamily::SB => {
                let m = self.mean();
                self.expect(|x| (x - m) * (x - m))
            }
        }
    }
    fn median(&self) -> f64 {
        self.x_of_z(0.0)
    }
    fn support(&self) -> (f64, f64) {
        match self.family {
            JohnsonFamily::SU => (f64::NEG_INFINITY, f64::INFINITY),
            JohnsonFamily::SB => (self.xi, self.xi + self.lambda),
            JohnsonFamily::SL => (self.xi, f64::INFINITY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::check;
    use super::*;

    #[test]
    fn consistency() {
        check(&Normal::new(3.0, 2.0).unwrap(), &[-20.0, -1.0, 3.0, 4.5, 30.0]);
        check(&Lognormal::new(0.5, 0.8, 1.0).unwrap(), &[1.0001, 1.5, 3.0, 20.0, 400.0]);
        check(&HalfNormal::new(1.5, 0.5).unwrap(), &[0.5001, 1.0, 2.0, 8.0]);
        check(&FoldedNormal::new(1.0, 2.0).unwrap(), &[0.01, 0.5, 2.0, 6.0, 15.0]);
        check(&Johnson::new(JohnsonFamily::SU, 0.5, 1.3, 2.0, 1.5).unwrap(), &[-10.0, 0.0, 2.0, 5.0]);
        check(&Johnson::new(JohnsonFamily::SB, -0.3, 0.9, 1.0, 4.0).unwrap(), &[1.01, 2.0, 3.0, 4.99]);
        check(&Johnson::new(JohnsonFamily::SL, 0.2, 1.1, 0.0, 2.0).unwrap(), &[0.01, 1.0, 5.0, 50.0]);
    }

    #[test]
    fn moments_and_special_relations() {
        let l = Lognormal::new(0.0, 0.5, 0.0).unwrap();
        // Johnson SL with γ = −μ/σ·…: ln(X) ~ N(−γ/δ, 1/δ).
        let j = Johnson::new(JohnsonFamily::SL, 0.0, 2.0, 0.0, 1.0).unwrap();
        for &x in &[0.3, 1.0, 2.5] {
            assert!((l.cdf(x) - j.cdf(x)).abs() < 1e-15);
        }
        assert!((j.mean() - l.mean()).abs() < 1e-15 && (j.variance() - l.variance()).abs() < 1e-15);
        // Folded normal with μ = 0 is half-normal.
        let f = FoldedNormal::new(0.0, 1.7).unwrap();
        let h = HalfNormal::new(1.7, 0.0).unwrap();
        for &x in &[0.2, 1.0, 4.0] {
            assert!((f.cdf(x) - h.cdf(x)).abs() < 1e-15 && (f.pdf(x) - h.pdf(x)).abs() < 1e-15);
        }
        assert!((f.mean() - h.mean()).abs() < 1e-15 && (f.variance() - h.variance()).abs() < 1e-14);
        // SB mean/variance by quadrature vs a direct numerical integral of x·f(x).
        let sb = Johnson::new(JohnsonFamily::SB, 0.4, 1.2, 0.0, 1.0).unwrap();
        let m = crate::integrate::adaptive(&mut |x| x * sb.pdf(x), 0.0, 1.0, 1e-14, 1e-13).unwrap();
        assert!((sb.mean() - m).abs() < 1e-12);
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Johnson::new(JohnsonFamily::SB, 0.0, -1.0, 0.0, 1.0).is_err());
    }
}
