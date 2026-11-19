//! Dunnett's many-to-one comparison distribution.
//!
//! The k test statistics Tᵢ = Zᵢ/S (comparison of treatment i with a common
//! control) are multivariate t with correlations ρᵢⱼ = λᵢλⱼ, where
//! λᵢ = √(nᵢ/(nᵢ + n₀)). This product structure reduces the k-dimensional
//! probability to a two-dimensional integral (Dunnett 1955):
//!
//! ```text
//! two-sided: P(max|Tᵢ| ≤ c) = ∫ f_ν(s) ∫ φ(z) Πᵢ [Φ((cs + λᵢz)/√(1−λᵢ²)) − Φ((−cs + λᵢz)/√(1−λᵢ²))] dz ds
//! one-sided: P(max Tᵢ ≤ c)  = ∫ f_ν(s) ∫ φ(z) Πᵢ  Φ((cs + λᵢz)/√(1−λᵢ²)) dz ds
//! ```
//!
//! evaluated with the same Gauss–Legendre machinery as the studentized range.

use super::erf::norm_cdf;
use super::tukey::{mix_over_scaled_chi, normal_points};
use crate::error::{MathError, Result};

/// Dunnett distribution for given correlation factors λᵢ and error df.
#[derive(Debug, Clone, PartialEq)]
pub struct Dunnett {
    lambdas: Vec<f64>,
    df: f64,
}

impl Dunnett {
    /// From correlation factors λᵢ ∈ [0, 1).
    pub fn new(lambdas: Vec<f64>, df: f64) -> Result<Self> {
        if lambdas.is_empty() || lambdas.iter().any(|l| !(0.0..1.0).contains(l)) {
            return Err(MathError::InvalidArgument("Dunnett: need at least one λ in [0, 1)".into()));
        }
        if df.is_nan() || df < 1.0 {
            return Err(MathError::InvalidArgument(format!("Dunnett: df must be ≥ 1, got {df}")));
        }
        Ok(Dunnett { lambdas, df })
    }

    /// From group sizes: control n₀ and treatments nᵢ.
    pub fn from_sizes(n_control: f64, n_treatments: &[f64], df: f64) -> Result<Self> {
        if n_control.is_nan() || n_control <= 0.0 || n_treatments.iter().any(|&n| n.is_nan() || n <= 0.0) {
            return Err(MathError::InvalidArgument("Dunnett: group sizes must be positive".into()));
        }
        Self::new(n_treatments.iter().map(|&n| (n / (n + n_control)).sqrt()).collect(), df)
    }

    /// k treatments, all groups of equal size (ρ = ½).
    pub fn balanced(k: usize, df: f64) -> Result<Self> {
        Self::new(vec![0.5f64.sqrt(); k], df)
    }

    fn inner(&self, cs: f64, two_sided: bool) -> f64 {
        let scaled: Vec<(f64, f64)> = self.lambdas.iter().map(|&l| (l, (1.0 - l * l).sqrt())).collect();
        normal_points()
            .iter()
            .map(|&(z, wt, _)| {
                let mut prod = 1.0;
                for &(l, r) in &scaled {
                    let up = norm_cdf((cs + l * z) / r);
                    let v = if two_sided { up - norm_cdf((-cs + l * z) / r) } else { up };
                    prod *= v.max(0.0);
                    if prod == 0.0 {
                        break;
                    }
                }
                wt * prod
            })
            .sum()
    }

    /// P(max |Tᵢ| ≤ c).
    pub fn cdf_two_sided(&self, c: f64) -> f64 {
        if c <= 0.0 {
            return 0.0;
        }
        mix_over_scaled_chi(self.df, |s| self.inner(c * s, true)).clamp(0.0, 1.0)
    }

    /// P(max Tᵢ ≤ c).
    pub fn cdf_one_sided(&self, c: f64) -> f64 {
        if c == f64::NEG_INFINITY {
            return 0.0;
        }
        if c == f64::INFINITY {
            return 1.0;
        }
        mix_over_scaled_chi(self.df, |s| self.inner(c * s, false)).clamp(0.0, 1.0)
    }

    fn solve(&self, p: f64, two_sided: bool) -> Result<f64> {
        if !(0.0..1.0).contains(&p) || p.is_nan() || (two_sided && p == 0.0) {
            return Err(MathError::InvalidArgument(format!("probability {p} outside the valid range")));
        }
        let f = |c: f64| if two_sided { self.cdf_two_sided(c) } else { self.cdf_one_sided(c) } - p;
        let lo0 = if two_sided { 1e-3 } else { -5.0 };
        let (lo, hi) = crate::roots::expand_bracket(f, lo0, 4.0, if two_sided { 0.0 } else { -1e6 }, 1e6)?;
        crate::roots::brent(f, lo, hi, 1e-12, 200)
    }

    /// Critical value c with P(max |Tᵢ| ≤ c) = p (e.g. p = 0.95).
    pub fn quantile_two_sided(&self, p: f64) -> Result<f64> {
        self.solve(p, true)
    }

    /// Critical value c with P(max Tᵢ ≤ c) = p.
    pub fn quantile_one_sided(&self, p: f64) -> Result<f64> {
        self.solve(p, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_comparison_is_t() {
        // k = 1: P(|T_ν| ≤ c) = 1 − I_{ν/(ν+c²)}(ν/2, ½), and one-sided = Student t cdf.
        for &(c, df) in &[(2.0, 10.0), (1.5, 3.0), (2.5, 60.0)] {
            let d = Dunnett::new(vec![0.6], df).unwrap();
            let x = df / (df + c * c);
            let two = 1.0 - super::super::incbeta::beta_inc(df / 2.0, 0.5, x);
            assert!((d.cdf_two_sided(c) - two).abs() < 1e-11, "two-sided c={c} df={df}");
            let one = 1.0 - 0.5 * super::super::incbeta::beta_inc(df / 2.0, 0.5, x);
            assert!((d.cdf_one_sided(c) - one).abs() < 1e-11, "one-sided c={c} df={df}");
        }
    }

    #[test]
    fn independence_limit_and_inputs() {
        // λ = 0: independent ⇒ product of univariate probabilities (df = ∞).
        let d = Dunnett::new(vec![0.0, 0.0, 0.0], f64::INFINITY).unwrap();
        let c = 2.0;
        let one = norm_cdf(c);
        assert!((d.cdf_one_sided(c) - one.powi(3)).abs() < 1e-13);
        assert!((d.cdf_two_sided(c) - (2.0 * one - 1.0).powi(3)).abs() < 1e-13);
        assert!(Dunnett::new(vec![1.0], 5.0).is_err());
        assert!(Dunnett::from_sizes(0.0, &[3.0], 5.0).is_err());
        let b = Dunnett::from_sizes(4.0, &[4.0, 4.0], 10.0).unwrap();
        assert_eq!(b, Dunnett::balanced(2, 10.0).unwrap());
    }
}
