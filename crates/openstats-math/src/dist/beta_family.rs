//! Beta, Student t and Fisher F distributions (incomplete-beta based).

use super::{require, valid_p, ContinuousDistribution};
use crate::error::Result;
use crate::special::erf::{norm_cdf, norm_isf, norm_pdf, norm_ppf, norm_sf};
use crate::special::gamma::ln_beta;
use crate::special::incbeta::{beta_inc_inv_pair, beta_inc_pair, beta_prefix};
use std::f64::consts::PI;

/// Beta(a, b) on [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    pub a: f64,
    pub b: f64,
}

impl Beta {
    pub fn new(a: f64, b: f64) -> Result<Self> {
        require(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite(), || {
            format!("Beta: need shapes > 0 (got {a}, {b})")
        })?;
        Ok(Beta { a, b })
    }
}

impl ContinuousDistribution for Beta {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::beta(rng, self.a, self.b)
    }
    fn name(&self) -> &'static str {
        "Beta"
    }
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 || x == 1.0 {
            let shape = if x == 0.0 { self.a } else { self.b };
            return match shape.partial_cmp(&1.0) {
                Some(std::cmp::Ordering::Less) => f64::INFINITY,
                Some(std::cmp::Ordering::Equal) => (-ln_beta(self.a, self.b)).exp(),
                _ => 0.0,
            };
        }
        beta_prefix(self.a, self.b, x, 1.0 - x) / (x * (1.0 - x))
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return self.pdf(x).ln();
        }
        (self.a - 1.0) * x.ln() + (self.b - 1.0) * (-x).ln_1p() - ln_beta(self.a, self.b)
    }
    fn cdf(&self, x: f64) -> f64 {
        beta_inc_pair(self.a, self.b, x.clamp(0.0, 1.0), (1.0 - x).clamp(0.0, 1.0)).0
    }
    fn sf(&self, x: f64) -> f64 {
        beta_inc_pair(self.a, self.b, x.clamp(0.0, 1.0), (1.0 - x).clamp(0.0, 1.0)).1
    }
    fn quantile(&self, p: f64) -> f64 {
        beta_inc_inv_pair(self.a, self.b, p, 1.0 - p).0
    }
    fn isf(&self, q: f64) -> f64 {
        beta_inc_inv_pair(self.a, self.b, 1.0 - q, q).0
    }
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
    fn support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
}

/// Student t with ν degrees of freedom (ν = ∞ gives the standard normal).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT {
    pub df: f64,
}

impl StudentT {
    pub fn new(df: f64) -> Result<Self> {
        require(df > 0.0, || format!("t: need df > 0 (got {df})"))?;
        Ok(StudentT { df })
    }

    /// (P(T > |t|), P(|T| < |t|)) both with full relative accuracy.
    fn tails(&self, t: f64) -> (f64, f64) {
        let nu = self.df;
        let t2 = t * t;
        let (x, y) =
            if t2 < nu { (nu / (nu + t2), t2 / (nu + t2)) } else { (1.0 / (1.0 + t2 / nu), 1.0 / (1.0 + nu / t2)) };
        let (ix, jx) = beta_inc_pair(0.5 * nu, 0.5, x, y);
        (0.5 * ix, jx)
    }

    /// Upper-tail quantile for q ≤ ½: t > 0 with P(T > t) = q.
    fn upper(&self, q: f64) -> f64 {
        let nu = self.df;
        if q == 0.5 {
            return 0.0;
        }
        if nu == 1.0 {
            return 1.0 / (PI * q).tan();
        }
        if nu == 2.0 {
            return (1.0 - 2.0 * q) / (2.0 * q * (1.0 - q)).sqrt();
        }
        let (x, y) = beta_inc_inv_pair(0.5 * nu, 0.5, 2.0 * q, 1.0 - 2.0 * q);
        (nu * y / x).sqrt()
    }
}

impl ContinuousDistribution for StudentT {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::student_t(rng, self.df)
    }
    fn name(&self) -> &'static str {
        "t"
    }
    fn pdf(&self, t: f64) -> f64 {
        if self.df.is_infinite() {
            return norm_pdf(t);
        }
        self.ln_pdf(t).exp()
    }
    fn ln_pdf(&self, t: f64) -> f64 {
        let nu = self.df;
        if nu.is_infinite() {
            return crate::special::erf::norm_ln_pdf(t);
        }
        -0.5 * (nu + 1.0) * (t * t / nu).ln_1p() - 0.5 * nu.ln() - ln_beta(0.5 * nu, 0.5)
    }
    fn cdf(&self, t: f64) -> f64 {
        if self.df.is_infinite() {
            return norm_cdf(t);
        }
        if t.is_nan() {
            return f64::NAN;
        }
        let (tail, center) = self.tails(t);
        if t < 0.0 {
            tail
        } else {
            0.5 + 0.5 * center
        }
    }
    fn sf(&self, t: f64) -> f64 {
        if self.df.is_infinite() {
            return norm_sf(t);
        }
        if t.is_nan() {
            return f64::NAN;
        }
        let (tail, center) = self.tails(t);
        if t > 0.0 {
            tail
        } else {
            0.5 + 0.5 * center
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        if self.df.is_infinite() {
            return norm_ppf(p);
        }
        if p == 0.0 {
            return f64::NEG_INFINITY;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }
        if p < 0.5 {
            -self.upper(p)
        } else {
            self.upper(1.0 - p)
        }
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        if self.df.is_infinite() {
            return norm_isf(q);
        }
        if q == 0.0 {
            return f64::INFINITY;
        }
        if q == 1.0 {
            return f64::NEG_INFINITY;
        }
        if q <= 0.5 {
            self.upper(q)
        } else {
            -self.upper(1.0 - q)
        }
    }
    fn mean(&self) -> f64 {
        if self.df > 1.0 {
            0.0
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        let nu = self.df;
        if nu.is_infinite() {
            1.0
        } else if nu > 2.0 {
            nu / (nu - 2.0)
        } else if nu > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
    fn median(&self) -> f64 {
        0.0
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// Fisher F(d₁, d₂).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherF {
    pub d1: f64,
    pub d2: f64,
}

impl FisherF {
    pub fn new(d1: f64, d2: f64) -> Result<Self> {
        require(d1 > 0.0 && d2 > 0.0 && d1.is_finite() && d2.is_finite(), || {
            format!("F: need df > 0 (got {d1}, {d2})")
        })?;
        Ok(FisherF { d1, d2 })
    }
    /// (u, v) = (d₁x/(d₁x+d₂), d₂/(d₁x+d₂)) computed without cancellation.
    fn uv(&self, x: f64) -> (f64, f64) {
        let s = self.d1 * x + self.d2;
        (self.d1 * x / s, self.d2 / s)
    }
}

impl ContinuousDistribution for FisherF {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::fisher_f(rng, self.d1, self.d2)
    }
    fn name(&self) -> &'static str {
        "F"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            return match self.d1.partial_cmp(&2.0) {
                Some(std::cmp::Ordering::Less) => f64::INFINITY,
                Some(std::cmp::Ordering::Equal) => 1.0,
                _ => 0.0,
            };
        }
        let (u, v) = self.uv(x);
        beta_prefix(0.5 * self.d1, 0.5 * self.d2, u, v) / x
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let (u, v) = self.uv(x);
        beta_inc_pair(0.5 * self.d1, 0.5 * self.d2, u, v).0
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let (u, v) = self.uv(x);
        beta_inc_pair(0.5 * self.d1, 0.5 * self.d2, u, v).1
    }
    fn quantile(&self, p: f64) -> f64 {
        let (u, v) = beta_inc_inv_pair(0.5 * self.d1, 0.5 * self.d2, p, 1.0 - p);
        self.d2 * u / (self.d1 * v)
    }
    fn isf(&self, q: f64) -> f64 {
        let (u, v) = beta_inc_inv_pair(0.5 * self.d1, 0.5 * self.d2, 1.0 - q, q);
        self.d2 * u / (self.d1 * v)
    }
    fn mean(&self) -> f64 {
        if self.d2 > 2.0 {
            self.d2 / (self.d2 - 2.0)
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        let (a, b) = (self.d1, self.d2);
        if b > 4.0 {
            2.0 * b * b * (a + b - 2.0) / (a * (b - 2.0) * (b - 2.0) * (b - 4.0))
        } else {
            f64::NAN
        }
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::check;
    use super::*;

    #[test]
    fn consistency_and_relations() {
        check(&Beta::new(2.0, 5.0).unwrap(), &[0.001, 0.2, 0.5, 0.9, 0.999]);
        check(&Beta::new(0.4, 0.7).unwrap(), &[1e-6, 0.1, 0.6, 0.99]);
        check(&StudentT::new(4.0).unwrap(), &[-50.0, -2.0, -0.1, 0.0, 0.3, 2.5, 80.0]);
        check(&StudentT::new(1.0).unwrap(), &[-1e3, -1.0, 0.5, 30.0]);
        check(&StudentT::new(2.0).unwrap(), &[-10.0, 0.5, 20.0]);
        check(&FisherF::new(3.0, 12.0).unwrap(), &[0.01, 0.5, 1.0, 4.0, 40.0]);
        // F(1, ν) = T²: P(F > t²) = 2·P(T > t).
        let (f, t) = (FisherF::new(1.0, 9.0).unwrap(), StudentT::new(9.0).unwrap());
        for &x in &[0.5, 2.0, 6.0] {
            assert!((f.sf(x * x) / (2.0 * t.sf(x)) - 1.0).abs() < 1e-13);
        }
        // t with ν = 1 is Cauchy: cdf = 1/2 + atan(t)/π.
        let c = StudentT::new(1.0).unwrap();
        for &x in &[-3.0, 0.2, 7.0] {
            assert!((c.cdf(x) - (0.5 + x.atan() / PI)).abs() < 1e-15);
        }
        // Large df approaches the normal.
        let big = StudentT::new(1e8).unwrap();
        assert!((big.cdf(1.5) - norm_cdf(1.5)).abs() < 1e-8);
        let inf = StudentT::new(f64::INFINITY).unwrap();
        assert_eq!(inf.quantile(0.975), norm_ppf(0.975));
        // Precision near the centre and deep tail.
        let p = 0.5 + 1e-12;
        let d = p - 0.5; // offset actually represented
        assert!((t.quantile(p) / (d / t.pdf(0.0)) - 1.0).abs() < 1e-9);
        assert!(t.isf(1e-200) > 1e20);
        assert!(Beta::new(0.0, 1.0).is_err());
    }
}
