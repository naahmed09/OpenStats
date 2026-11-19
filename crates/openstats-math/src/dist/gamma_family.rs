//! Gamma, chi-square, exponential and Weibull distributions.

use super::{require, valid_p, ContinuousDistribution};
use crate::error::Result;
use crate::special::gamma::{gamma, ln_gamma};
use crate::special::incgamma::{gamma_p_inv, gamma_pq, gamma_prefix, gamma_q_inv};

/// Gamma(shape k, scale θ, threshold t).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    pub shape: f64,
    pub scale: f64,
    pub threshold: f64,
}

impl Gamma {
    pub fn new(shape: f64, scale: f64, threshold: f64) -> Result<Self> {
        require(shape > 0.0 && scale > 0.0 && shape.is_finite() && scale.is_finite() && threshold.is_finite(), || {
            format!("Gamma: need shape > 0 and scale > 0 (got {shape}, {scale})")
        })?;
        Ok(Gamma { shape, scale, threshold })
    }
    fn y(&self, x: f64) -> f64 {
        (x - self.threshold) / self.scale
    }
}

impl ContinuousDistribution for Gamma {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        self.threshold + self.scale * crate::random::variates::std_gamma(rng, self.shape)
    }
    fn name(&self) -> &'static str {
        "Gamma"
    }
    fn pdf(&self, x: f64) -> f64 {
        let y = self.y(x);
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return match self.shape.partial_cmp(&1.0) {
                Some(std::cmp::Ordering::Less) => f64::INFINITY,
                Some(std::cmp::Ordering::Equal) => 1.0 / self.scale,
                _ => 0.0,
            };
        }
        gamma_prefix(self.shape, y) / (y * self.scale)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let y = self.y(x);
        if y <= 0.0 {
            return self.pdf(x).ln();
        }
        (self.shape - 1.0) * y.ln() - y - ln_gamma(self.shape) - self.scale.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        let y = self.y(x);
        if y <= 0.0 {
            0.0
        } else {
            gamma_pq(self.shape, y).0
        }
    }
    fn sf(&self, x: f64) -> f64 {
        let y = self.y(x);
        if y <= 0.0 {
            1.0
        } else {
            gamma_pq(self.shape, y).1
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        self.threshold + self.scale * gamma_p_inv(self.shape, p)
    }
    fn isf(&self, q: f64) -> f64 {
        self.threshold + self.scale * gamma_q_inv(self.shape, q)
    }
    fn mean(&self) -> f64 {
        self.threshold + self.shape * self.scale
    }
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
    fn support(&self) -> (f64, f64) {
        (self.threshold, f64::INFINITY)
    }
}

/// Chi-square with ν degrees of freedom (= Gamma(ν/2, 2)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    pub df: f64,
    g: Gamma,
}

impl ChiSquared {
    pub fn new(df: f64) -> Result<Self> {
        require(df > 0.0 && df.is_finite(), || format!("Chi-square: need df > 0 (got {df})"))?;
        Ok(ChiSquared { df, g: Gamma { shape: df / 2.0, scale: 2.0, threshold: 0.0 } })
    }
}

impl ContinuousDistribution for ChiSquared {
    fn sample(&self, rng: &mut dyn crate::random::Rng) -> f64 {
        crate::random::variates::chi_square(rng, self.df)
    }
    fn name(&self) -> &'static str {
        "Chi-square"
    }
    fn pdf(&self, x: f64) -> f64 {
        self.g.pdf(x)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        self.g.ln_pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        self.g.cdf(x)
    }
    fn sf(&self, x: f64) -> f64 {
        self.g.sf(x)
    }
    fn quantile(&self, p: f64) -> f64 {
        self.g.quantile(p)
    }
    fn isf(&self, q: f64) -> f64 {
        self.g.isf(q)
    }
    fn mean(&self) -> f64 {
        self.df
    }
    fn variance(&self) -> f64 {
        2.0 * self.df
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
}

/// Exponential with mean `scale` and threshold (Minitab parameterisation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    pub scale: f64,
    pub threshold: f64,
}

impl Exponential {
    pub fn new(scale: f64, threshold: f64) -> Result<Self> {
        require(scale > 0.0 && scale.is_finite() && threshold.is_finite(), || {
            format!("Exponential: need scale > 0 (got {scale})")
        })?;
        Ok(Exponential { scale, threshold })
    }
}

impl ContinuousDistribution for Exponential {
    fn name(&self) -> &'static str {
        "Exponential"
    }
    fn pdf(&self, x: f64) -> f64 {
        if x < self.threshold {
            0.0
        } else {
            (-(x - self.threshold) / self.scale).exp() / self.scale
        }
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.threshold {
            f64::NEG_INFINITY
        } else {
            -(x - self.threshold) / self.scale - self.scale.ln()
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            0.0
        } else {
            -(-(x - self.threshold) / self.scale).exp_m1()
        }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= self.threshold {
            1.0
        } else {
            (-(x - self.threshold) / self.scale).exp()
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.threshold - self.scale * (-p).ln_1p()
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.threshold - self.scale * q.ln()
    }
    fn mean(&self) -> f64 {
        self.threshold + self.scale
    }
    fn variance(&self) -> f64 {
        self.scale * self.scale
    }
    fn support(&self) -> (f64, f64) {
        (self.threshold, f64::INFINITY)
    }
}

/// Weibull(shape β, scale η, threshold t).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull {
    pub shape: f64,
    pub scale: f64,
    pub threshold: f64,
}

impl Weibull {
    pub fn new(shape: f64, scale: f64, threshold: f64) -> Result<Self> {
        require(shape > 0.0 && scale > 0.0 && shape.is_finite() && scale.is_finite() && threshold.is_finite(), || {
            format!("Weibull: need shape > 0 and scale > 0 (got {shape}, {scale})")
        })?;
        Ok(Weibull { shape, scale, threshold })
    }
}

impl ContinuousDistribution for Weibull {
    fn name(&self) -> &'static str {
        "Weibull"
    }
    fn pdf(&self, x: f64) -> f64 {
        let y = (x - self.threshold) / self.scale;
        if y < 0.0 {
            return 0.0;
        }
        if y == 0.0 {
            return match self.shape.partial_cmp(&1.0) {
                Some(std::cmp::Ordering::Less) => f64::INFINITY,
                Some(std::cmp::Ordering::Equal) => 1.0 / self.scale,
                _ => 0.0,
            };
        }
        let yb = y.powf(self.shape);
        self.shape / self.scale * yb / y * (-yb).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        let y = (x - self.threshold) / self.scale;
        if y <= 0.0 {
            return self.pdf(x).ln();
        }
        (self.shape / self.scale).ln() + (self.shape - 1.0) * y.ln() - y.powf(self.shape)
    }
    fn cdf(&self, x: f64) -> f64 {
        let y = (x - self.threshold) / self.scale;
        if y <= 0.0 {
            0.0
        } else {
            -(-y.powf(self.shape)).exp_m1()
        }
    }
    fn sf(&self, x: f64) -> f64 {
        let y = (x - self.threshold) / self.scale;
        if y <= 0.0 {
            1.0
        } else {
            (-y.powf(self.shape)).exp()
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        if !valid_p(p) {
            return f64::NAN;
        }
        self.threshold + self.scale * (-(-p).ln_1p()).powf(1.0 / self.shape)
    }
    fn isf(&self, q: f64) -> f64 {
        if !valid_p(q) {
            return f64::NAN;
        }
        self.threshold + self.scale * (-q.ln()).powf(1.0 / self.shape)
    }
    fn mean(&self) -> f64 {
        self.threshold + self.scale * gamma(1.0 + 1.0 / self.shape)
    }
    fn variance(&self) -> f64 {
        let g1 = gamma(1.0 + 1.0 / self.shape);
        self.scale * self.scale * (gamma(1.0 + 2.0 / self.shape) - g1 * g1)
    }
    fn median(&self) -> f64 {
        self.threshold + self.scale * std::f64::consts::LN_2.powf(1.0 / self.shape)
    }
    fn support(&self) -> (f64, f64) {
        (self.threshold, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::check;
    use super::*;

    #[test]
    fn consistency_and_relations() {
        check(&Gamma::new(2.5, 1.5, 0.0).unwrap(), &[0.01, 1.0, 4.0, 20.0, 80.0]);
        check(&Gamma::new(0.3, 2.0, 1.0).unwrap(), &[1.0001, 1.1, 3.0, 20.0]);
        check(&ChiSquared::new(7.0).unwrap(), &[0.1, 3.0, 7.0, 30.0, 120.0]);
        check(&Exponential::new(2.0, 1.0).unwrap(), &[1.0001, 2.0, 10.0, 100.0]);
        check(&Weibull::new(1.7, 3.0, 0.5).unwrap(), &[0.51, 1.0, 3.0, 10.0, 20.0]);
        check(&Weibull::new(0.6, 1.0, 0.0).unwrap(), &[1e-4, 0.5, 5.0, 100.0]);
        // Gamma(1, θ) = Exponential(θ); Weibull(1, η) = Exponential(η).
        let (g, e, w) = (
            Gamma::new(1.0, 2.0, 0.0).unwrap(),
            Exponential::new(2.0, 0.0).unwrap(),
            Weibull::new(1.0, 2.0, 0.0).unwrap(),
        );
        for &x in &[0.1, 1.0, 7.0, 60.0] {
            assert!((g.sf(x) / e.sf(x) - 1.0).abs() < 1e-14);
            assert!((w.sf(x) / e.sf(x) - 1.0).abs() < 1e-14);
            assert!((g.pdf(x) / e.pdf(x) - 1.0).abs() < 1e-14);
        }
        // Chi-square with 2 df: sf = e^{−x/2}.
        let c = ChiSquared::new(2.0).unwrap();
        assert!((c.isf(1e-100) - 200.0 * 10f64.ln()).abs() < 1e-11);
        assert!((Weibull::new(2.0, 1.0, 0.0).unwrap().mean() - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-15);
        assert_eq!(Gamma::new(0.5, 1.0, 0.0).unwrap().pdf(0.0), f64::INFINITY);
        assert!(Gamma::new(0.0, 1.0, 0.0).is_err());
    }
}
