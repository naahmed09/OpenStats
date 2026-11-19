//! Logistic, loglogistic, Cauchy, Laplace, uniform, triangular and the two
//! extreme-value (Gumbel) distributions.

use super::{require, valid_p, ContinuousDistribution};
use crate::error::Result;
use crate::special::gamma::{gamma, sin_pi, EULER_GAMMA};
use std::f64::consts::PI;

fn check_ls(name: &str, loc: f64, scale: f64) -> Result<()> {
    require(loc.is_finite() && scale > 0.0 && scale.is_finite(), || {
        format!("{name}: need finite location and scale > 0 (got {loc}, {scale})")
    })
}

/// Standard logistic cdf 1/(1+e^{−z}) without overflow or cancellation.
fn logistic_cdf(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// logit(p) = ln(p/(1−p)).
fn logit(p: f64) -> f64 {
    p.ln() - (-p).ln_1p()
}

/// Logistic(μ, s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Logistic {
    pub mu: f64,
    pub scale: f64,
}

impl Logistic {
    pub fn new(mu: f64, scale: f64) -> Result<Self> {
        check_ls("Logistic", mu, scale)?;
        Ok(Logistic { mu, scale })
    }
}

impl ContinuousDistribution for Logistic {
    fn name(&self) -> &'static str {
        "Logistic"
    }
    fn pdf(&self, x: f64) -> f64 {
        let e = (-((x - self.mu) / self.scale).abs()).exp();
        e / (self.scale * (1.0 + e) * (1.0 + e))
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let a = ((x - self.mu) / self.scale).abs();
        -a - 2.0 * (-a).exp().ln_1p() - self.scale.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        logistic_cdf((x - self.mu) / self.scale)
    }
    fn sf(&self, x: f64) -> f64 {
        logistic_cdf(-(x - self.mu) / self.scale)
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.mu + self.scale * logit(p)
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.mu - self.scale * logit(q)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        (self.scale * PI).powi(2) / 3.0
    }
    fn median(&self) -> f64 {
        self.mu
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Loglogistic: ln(X − θ) ~ Logistic(μ, σ) (Minitab parameterisation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Loglogistic {
    pub mu: f64,
    pub sigma: f64,
    pub threshold: f64,
}

impl Loglogistic {
    pub fn new(mu: f64, sigma: f64, threshold: f64) -> Result<Self> {
        check_ls("Loglogistic", mu, sigma)?;
        require(threshold.is_finite(), || "Loglogistic: threshold must be finite".into())?;
        Ok(Loglogistic { mu, sigma, threshold })
    }
    fn z(&self, x: f64) -> f64 {
        ((x - self.threshold).ln() - self.mu) / self.sigma
    }
}

impl ContinuousDistribution for Loglogistic {
    fn name(&self) -> &'static str {
        "Loglogistic"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            return 0.0;
        }
        let e = (-self.z(x).abs()).exp();
        e / (self.sigma * (1.0 + e) * (1.0 + e) * (x - self.threshold))
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            0.0
        } else {
            logistic_cdf(self.z(x))
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            1.0
        } else {
            logistic_cdf(-self.z(x))
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.threshold + (self.mu + self.sigma * logit(p)).exp()
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.threshold + (self.mu - self.sigma * logit(q)).exp()
    }
    fn mean(&self) -> f64 {
        if self.sigma < 1.0 {
            self.threshold + self.mu.exp() * PI * self.sigma / sin_pi(self.sigma)
        } else {
            f64::INFINITY
        }
    }
    fn variance(&self) -> f64 {
        let s = self.sigma;
        if s < 0.5 {
            let m1 = gamma(1.0 + s) * gamma(1.0 - s);
            (2.0 * self.mu).exp() * (gamma(1.0 + 2.0 * s) * gamma(1.0 - 2.0 * s) - m1 * m1)
        } else {
            f64::INFINITY
        }
    }
    fn median(&self) -> f64 {
        self.threshold + self.mu.exp()
    }
    fn support(&self) -> (f64, f64) {
        (self.threshold, f64::INFINITY)
    }
}

/// Cauchy(location, scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cauchy {
    pub loc: f64,
    pub scale: f64,
}

impl Cauchy {
    pub fn new(loc: f64, scale: f64) -> Result<Self> {
        check_ls("Cauchy", loc, scale)?;
        Ok(Cauchy { loc, scale })
    }
}

/// Standard Cauchy cdf with full accuracy in the lower tail.
fn cauchy_cdf(z: f64) -> f64 {
    if z < -1.0 {
        (-1.0 / z).atan() / PI
    } else {
        0.5 + z.atan() / PI
    }
}

impl ContinuousDistribution for Cauchy {
    fn name(&self) -> &'static str {
        "Cauchy"
    }
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (PI * self.scale * (1.0 + z * z))
    }
    fn cdf(&self, x: f64) -> f64 {
        cauchy_cdf((x - self.loc) / self.scale)
    }
    fn sf(&self, x: f64) -> f64 {
        cauchy_cdf(-(x - self.loc) / self.scale)
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        if p == 0.5 {
            return self.loc;
        }
        if p < 0.5 {
            self.loc - self.scale / (PI * p).tan()
        } else {
            self.loc + self.scale / (PI * (1.0 - p)).tan()
        }
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        if q == 0.5 {
            return self.loc;
        }
        if q < 0.5 {
            self.loc + self.scale / (PI * q).tan()
        } else {
            self.loc - self.scale / (PI * (1.0 - q)).tan()
        }
    }
    fn mean(&self) -> f64 {
        f64::NAN
    }
    fn variance(&self) -> f64 {
        f64::NAN
    }
    fn median(&self) -> f64 {
        self.loc
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Laplace (double exponential)(μ, b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Laplace {
    pub mu: f64,
    pub scale: f64,
}

impl Laplace {
    pub fn new(mu: f64, scale: f64) -> Result<Self> {
        check_ls("Laplace", mu, scale)?;
        Ok(Laplace { mu, scale })
    }
}

impl ContinuousDistribution for Laplace {
    fn name(&self) -> &'static str {
        "Laplace"
    }
    fn pdf(&self, x: f64) -> f64 {
        (-((x - self.mu) / self.scale).abs()).exp() / (2.0 * self.scale)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        -((x - self.mu) / self.scale).abs() - (2.0 * self.scale).ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.scale;
        if z < 0.0 {
            0.5 * z.exp()
        } else {
            1.0 - 0.5 * (-z).exp()
        }
    }
    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.scale;
        if z > 0.0 {
            0.5 * (-z).exp()
        } else {
            1.0 - 0.5 * z.exp()
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        if p < 0.5 {
            self.mu + self.scale * (2.0 * p).ln()
        } else {
            self.mu - self.scale * (2.0 * (1.0 - p)).ln()
        }
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        if q < 0.5 {
            self.mu - self.scale * (2.0 * q).ln()
        } else {
            self.mu + self.scale * (2.0 * (1.0 - q)).ln()
        }
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        2.0 * self.scale * self.scale
    }
    fn median(&self) -> f64 {
        self.mu
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Continuous uniform on [a, b].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Result<Self> {
        require(a.is_finite() && b.is_finite() && a < b, || format!("Uniform: need a < b (got {a}, {b})"))?;
        Ok(Uniform { a, b })
    }
}

impl ContinuousDistribution for Uniform {
    fn name(&self) -> &'static str {
        "Uniform"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (self.b - self.a)
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        ((x - self.a) / (self.b - self.a)).clamp(0.0, 1.0)
    }
    fn sf(&self, x: f64) -> f64 {
        ((self.b - x) / (self.b - self.a)).clamp(0.0, 1.0)
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.a + p * (self.b - self.a)
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.b - q * (self.b - self.a)
    }
    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
    fn support(&self) -> (f64, f64) {
        (self.a, self.b)
    }
}

/// Triangular(lower a, mode c, upper b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangular {
    pub a: f64,
    pub c: f64,
    pub b: f64,
}

impl Triangular {
    pub fn new(a: f64, c: f64, b: f64) -> Result<Self> {
        require(a.is_finite() && b.is_finite() && c.is_finite() && a < b && a <= c && c <= b, || {
            format!("Triangular: need a ≤ c ≤ b with a < b (got {a}, {c}, {b})")
        })?;
        Ok(Triangular { a, c, b })
    }
}

impl ContinuousDistribution for Triangular {
    fn name(&self) -> &'static str {
        "Triangular"
    }
    fn pdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        if x < a || x > b {
            0.0
        } else if x < c {
            2.0 * (x - a) / ((b - a) * (c - a))
        } else if x == c {
            2.0 / (b - a)
        } else {
            2.0 * (b - x) / ((b - a) * (b - c))
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        if x <= a {
            0.0
        } else if x >= b {
            1.0
        } else if x <= c {
            (x - a) * (x - a) / ((b - a) * (c - a))
        } else {
            1.0 - (b - x) * (b - x) / ((b - a) * (b - c))
        }
    }
    fn sf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        if x <= a {
            1.0
        } else if x >= b {
            0.0
        } else if x <= c {
            1.0 - (x - a) * (x - a) / ((b - a) * (c - a))
        } else {
            (b - x) * (b - x) / ((b - a) * (b - c))
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        let (a, b, c) = (self.a, self.b, self.c);
        let fc = (c - a) / (b - a);
        if p <= fc {
            a + (p * (b - a) * (c - a)).sqrt()
        } else {
            b - ((1.0 - p) * (b - a) * (b - c)).sqrt()
        }
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        let (a, b, c) = (self.a, self.b, self.c);
        let fc = (c - a) / (b - a);
        if q <= 1.0 - fc {
            b - (q * (b - a) * (b - c)).sqrt()
        } else {
            a + ((1.0 - q) * (b - a) * (c - a)).sqrt()
        }
    }
    fn mean(&self) -> f64 {
        (self.a + self.b + self.c) / 3.0
    }
    fn variance(&self) -> f64 {
        let (a, b, c) = (self.a, self.b, self.c);
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }
    fn support(&self) -> (f64, f64) {
        (self.a, self.b)
    }
}

/// Smallest extreme value (Gumbel minimum): F(x) = 1 − exp(−e^{(x−μ)/σ}).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmallestExtremeValue {
    pub mu: f64,
    pub sigma: f64,
}

impl SmallestExtremeValue {
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        check_ls("Smallest extreme value", mu, sigma)?;
        Ok(SmallestExtremeValue { mu, sigma })
    }
}

impl ContinuousDistribution for SmallestExtremeValue {
    fn name(&self) -> &'static str {
        "Smallest extreme value"
    }
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        z - z.exp() - self.sigma.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        -(-((x - self.mu) / self.sigma).exp()).exp_m1()
    }
    fn sf(&self, x: f64) -> f64 {
        (-((x - self.mu) / self.sigma).exp()).exp()
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.mu + self.sigma * (-(-p).ln_1p()).ln()
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.mu + self.sigma * (-q.ln()).ln()
    }
    fn mean(&self) -> f64 {
        self.mu - EULER_GAMMA * self.sigma
    }
    fn variance(&self) -> f64 {
        (PI * self.sigma).powi(2) / 6.0
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Largest extreme value (Gumbel maximum): F(x) = exp(−e^{−(x−μ)/σ}).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LargestExtremeValue {
    pub mu: f64,
    pub sigma: f64,
}

impl LargestExtremeValue {
    pub fn new(mu: f64, sigma: f64) -> Result<Self> {
        check_ls("Largest extreme value", mu, sigma)?;
        Ok(LargestExtremeValue { mu, sigma })
    }
}

impl ContinuousDistribution for LargestExtremeValue {
    fn name(&self) -> &'static str {
        "Largest extreme value"
    }
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        -z - (-z).exp() - self.sigma.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        (-(-(x - self.mu) / self.sigma).exp()).exp()
    }
    fn sf(&self, x: f64) -> f64 {
        -(-(-(x - self.mu) / self.sigma).exp()).exp_m1()
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.mu - self.sigma * (-p.ln()).ln()
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.mu - self.sigma * (-(-q).ln_1p()).ln()
    }
    fn mean(&self) -> f64 {
        self.mu + EULER_GAMMA * self.sigma
    }
    fn variance(&self) -> f64 {
        (PI * self.sigma).powi(2) / 6.0
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::check;
    use super::*;

    #[test]
    fn consistency() {
        check(&Logistic::new(1.0, 2.0).unwrap(), &[-80.0, -3.0, 1.0, 4.0, 90.0]);
        check(&Loglogistic::new(0.5, 0.4, 1.0).unwrap(), &[1.001, 2.0, 5.0, 100.0]);
        check(&Cauchy::new(-1.0, 0.5).unwrap(), &[-1e6, -3.0, -1.0, 2.0, 1e6]);
        check(&Laplace::new(2.0, 1.5).unwrap(), &[-30.0, 0.0, 2.0, 3.0, 40.0]);
        check(&Uniform::new(-1.0, 3.0).unwrap(), &[-0.5, 0.0, 2.9]);
        check(&Triangular::new(0.0, 1.0, 4.0).unwrap(), &[0.1, 0.9, 1.0, 2.0, 3.99]);
        check(&SmallestExtremeValue::new(10.0, 2.0).unwrap(), &[-20.0, 5.0, 10.0, 13.0, 16.0]);
        check(&LargestExtremeValue::new(10.0, 2.0).unwrap(), &[4.0, 7.0, 10.0, 20.0, 60.0]);
    }

    #[test]
    fn relations_and_moments() {
        // −X with X ~ SEV(μ, σ) is LEV(−μ, σ).
        let (s, l) = (SmallestExtremeValue::new(1.0, 0.7).unwrap(), LargestExtremeValue::new(-1.0, 0.7).unwrap());
        for &x in &[-3.0, 0.0, 2.0] {
            assert!((s.cdf(x) - l.sf(-x)).abs() < 1e-15);
        }
        assert!((s.mean() + l.mean()).abs() < 1e-15);
        // Loglogistic median and mean.
        let ll = Loglogistic::new(0.0, 0.25, 0.0).unwrap();
        assert!((ll.cdf(1.0) - 0.5).abs() < 1e-15);
        let m = crate::integrate::adaptive(&mut |x| ll.sf(x), 0.0, 1e4, 1e-13, 1e-13).unwrap();
        assert!((ll.mean() - m).abs() < 1e-9, "{} vs {m}", ll.mean());
        // Cauchy tails keep relative precision.
        let c = Cauchy::new(0.0, 1.0).unwrap();
        assert!((c.cdf(-1e10) - 1.0 / (PI * 1e10)).abs() < 1e-25);
        assert!((c.isf(1e-12) - 1.0 / (PI * 1e-12).tan()).abs() < 1.0);
        assert!(Triangular::new(0.0, 5.0, 4.0).is_err());
        assert!(Uniform::new(1.0, 1.0).is_err());
        // Degenerate triangular modes.
        check(&Triangular::new(0.0, 0.0, 2.0).unwrap(), &[0.5, 1.5]);
        check(&Triangular::new(0.0, 2.0, 2.0).unwrap(), &[0.5, 1.5]);
    }
}
