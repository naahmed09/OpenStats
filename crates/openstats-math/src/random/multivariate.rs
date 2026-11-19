//! Multivariate samplers: normal, Wishart, Dirichlet, multinomial.

use super::variates::{binomial, ln_gamma_variate, std_gamma, std_normal};
use super::Rng;
use crate::error::{MathError, Result};
use crate::linalg::{Cholesky, Matrix, SymmetricEigen};

fn invalid(msg: impl Into<String>) -> MathError {
    MathError::InvalidArgument(msg.into())
}

/// A square root factor `A` of a covariance matrix (`A Aᵀ = Σ`): the Cholesky
/// factor when Σ is positive definite, otherwise `V·diag(√max(λ,0))` from the
/// symmetric eigendecomposition (positive semi-definite Σ).
fn covariance_factor(cov: &Matrix) -> Result<Matrix> {
    let n = cov.require_square()?;
    cov.require_finite("covariance")?;
    for i in 0..n {
        for j in 0..i {
            let (a, b) = (cov[(i, j)], cov[(j, i)]);
            if (a - b).abs() > 1e-10 * (1.0 + a.abs().max(b.abs())) {
                return Err(invalid("covariance matrix must be symmetric"));
            }
        }
    }
    // Accept the Cholesky factor only when no conditional variance is a
    // negligible fraction of its marginal variance; otherwise the matrix is
    // (numerically) singular and the eigen route handles it exactly.
    if let Ok(c) = Cholesky::new(cov) {
        let l = c.l();
        if (0..n).all(|i| l[(i, i)] * l[(i, i)] > 1e-10 * cov[(i, i)]) {
            return Ok(l.clone());
        }
    }
    let e = SymmetricEigen::new(cov)?;
    let scale = e.values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if e.values.iter().any(|&v| v < -1e-10 * scale.max(1.0)) {
        return Err(invalid("covariance matrix must be positive semi-definite"));
    }
    // Eigenvalues below the eigensolver's own error level are zero.
    let floor = 16.0 * n as f64 * f64::EPSILON * scale;
    Ok(Matrix::from_fn(n, n, |i, j| {
        let v = e.values[j];
        e.vectors[(i, j)] * if v > floor { v.sqrt() } else { 0.0 }
    }))
}

/// Sampler for the multivariate normal N(μ, Σ); the factorisation is done
/// once and reused for every draw.
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    factor: Matrix,
}

impl MultivariateNormal {
    pub fn new(mean: &[f64], cov: &Matrix) -> Result<Self> {
        if cov.rows() != mean.len() {
            return Err(MathError::DimensionMismatch {
                op: "multivariate normal",
                left: (mean.len(), 1),
                right: cov.shape(),
            });
        }
        if mean.iter().any(|m| !m.is_finite()) {
            return Err(invalid("mean must be finite"));
        }
        Ok(MultivariateNormal { mean: mean.to_vec(), factor: covariance_factor(cov)? })
    }

    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// One draw μ + A z.
    pub fn sample(&self, rng: &mut dyn Rng) -> Vec<f64> {
        let n = self.dim();
        let z: Vec<f64> = (0..n).map(|_| std_normal(rng)).collect();
        (0..n)
            .map(|i| {
                let row = self.factor.row(i);
                self.mean[i] + row.iter().zip(&z).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect()
    }

    /// `n` draws as the rows of an `n × dim` matrix.
    pub fn sample_n(&self, rng: &mut dyn Rng, n: usize) -> Matrix {
        let d = self.dim();
        let mut out = Matrix::zeros(n, d);
        for i in 0..n {
            out.row_mut(i).copy_from_slice(&self.sample(rng));
        }
        out
    }
}

/// One draw from the Wishart distribution W_p(df, Σ) (Bartlett
/// decomposition), for `df > p − 1`. Mean `df·Σ`.
pub fn wishart(rng: &mut dyn Rng, df: f64, scale: &Matrix) -> Result<Matrix> {
    let p = scale.require_square()?;
    if df.is_nan() || df <= p as f64 - 1.0 || !df.is_finite() {
        return Err(invalid(format!("Wishart: need df > p − 1 = {} (got {df})", p as f64 - 1.0)));
    }
    let l = covariance_factor(scale)?;
    // Bartlett factor: lower triangular with √χ²_{df−i} on the diagonal and
    // standard normals below.
    let mut a = Matrix::zeros(p, p);
    for i in 0..p {
        a[(i, i)] = (2.0 * std_gamma(rng, 0.5 * (df - i as f64))).sqrt();
        for j in 0..i {
            a[(i, j)] = std_normal(rng);
        }
    }
    let la = l.matmul(&a)?;
    la.matmul(&la.t())
}

/// One draw from Dirichlet(α), α_i > 0. Gammas are generated on the log
/// scale and normalised with a log-sum-exp, so very small α_i are safe.
pub fn dirichlet(rng: &mut dyn Rng, alpha: &[f64]) -> Result<Vec<f64>> {
    if alpha.is_empty() || alpha.iter().any(|&a| a.is_nan() || a <= 0.0 || !a.is_finite()) {
        return Err(invalid("Dirichlet: need a non-empty vector of finite positive parameters"));
    }
    let lg: Vec<f64> = alpha.iter().map(|&a| ln_gamma_variate(rng, a)).collect();
    let m = lg.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f64> = lg.iter().map(|&v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    Ok(e.into_iter().map(|v| v / s).collect())
}

/// One draw from Multinomial(n, probs): sequential conditional binomials.
/// `probs` need not be normalised (they are scaled to sum to 1).
pub fn multinomial(rng: &mut dyn Rng, n: u64, probs: &[f64]) -> Result<Vec<u64>> {
    if probs.is_empty() || probs.iter().any(|&p| p.is_nan() || p < 0.0 || !p.is_finite()) {
        return Err(invalid("multinomial: probabilities must be finite and non-negative"));
    }
    let total: f64 = probs.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return Err(invalid("multinomial: probabilities sum to zero"));
    }
    let mut out = vec![0u64; probs.len()];
    let (mut left, mut rest) = (n, total);
    for (i, &p) in probs.iter().enumerate() {
        if left == 0 {
            break;
        }
        if i + 1 == probs.len() {
            out[i] = left;
            break;
        }
        let q = (p / rest).clamp(0.0, 1.0);
        let k = binomial(rng, left, q);
        out[i] = k;
        left -= k;
        rest -= p;
        if rest <= 0.0 {
            // Remaining categories have no mass; everything left went above.
            break;
        }
    }
    Ok(out)
}
